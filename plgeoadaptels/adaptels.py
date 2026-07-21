"""
plGeoAdaptels - Scale-Adaptive Superpixels for geospatial data.

Main API module providing the high-level function to create adaptels.

Based on: R. Achanta, P. Marquez-Neila, P. Fua, S. Susstrunk,
"Scale-Adaptive Superpixels", Color and Imaging Conference, 2018.

Original C implementation: Pawel Netzel, University of Agriculture
in Krakow, Poland.

Python + Numba reimplementation for portability.
"""

import numpy as np
import time
from .core import _create_adaptels
from .io import read_raster, write_raster, normalize_layers

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# Distance name to int mapping
DISTANCE_MAP = {
    'minkowski': 0,
    'cosine': 1,
    'angular': 2,
}


def _validate_params(threshold, distance):
    """
    Check the parameters both entry points share, and resolve the distance.

    Kept in one place deliberately. When create_adaptels() and
    adaptels_from_array() each validated separately, the array version
    quietly fell back to minkowski on an unrecognised distance, so a typo
    like 'cosin' returned a full minkowski segmentation with no warning —
    a different result by orders of magnitude, silently mislabelled.

    Returns
    -------
    int
        The integer distance code for the Numba kernel.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be greater than 0")
    if distance not in DISTANCE_MAP:
        raise ValueError(
            f"Unknown distance '{distance}'. Use: {list(DISTANCE_MAP.keys())}"
        )
    # The metrics live on different scales, so one threshold does not carry
    # across them. 'cosine' and 'angular' are bounded by 1 by construction;
    # a threshold above that means "merge everything" and silently returns
    # a single adaptel, which looks like a broken algorithm rather than a
    # misread parameter. 'minkowski' grows with the data range and band
    # count, so it has no upper bound to check against.
    if distance in ("cosine", "angular") and threshold > 1.0:
        raise ValueError(
            f"threshold={threshold} is outside the range of the '{distance}' "
            f"metric, which produces distances in [0, 1]. Anything above 1 "
            f"merges the whole raster into one adaptel. Typical values are "
            f"0.005-0.2; try 0.03 to start. (The default threshold of 60 is "
            f"scaled for 'minkowski', whose distances follow the data range.)"
        )
    return DISTANCE_MAP[distance]


def _validate_normalized_threshold(threshold, distance, n_layers,
                                   minkowski_p):
    """Reject a threshold that no pair of pixels can reach once normalized.

    Called only when normalize=True, and only after the layer count is
    known. normalize rescales every band to [0, 1], which caps the largest
    possible minkowski distance at n_layers**(1/p) — about 1.73 for three
    bands at p=2. The package default of 60 is scaled for raw data, so
    normalize=True at the default merged the whole raster in silence: on
    the test scene 2792 adaptels became 1, with no error and no warning.

    Same failure as an out-of-scale 'cosine' threshold, so it fails the
    same way — loudly, naming the ceiling it cannot cross.
    """
    if distance != "minkowski":
        return  # cosine and angular are bounded by 1 regardless; checked above
    ceiling = float(n_layers) ** (1.0 / float(minkowski_p))
    if threshold >= ceiling:
        raise ValueError(
            f"threshold={threshold} cannot be reached with normalize=True. "
            f"Normalizing puts every band in [0, 1], so the largest possible "
            f"minkowski distance across {n_layers} band(s) at p={minkowski_p} "
            f"is {ceiling:.4f}. A threshold at or above that merges the "
            f"entire raster into one adaptel. Rescale the threshold to the "
            f"normalized range — on 3-band imagery 0.1-0.5 is a working "
            f"span, and the right value depends on the scene — or drop "
            f"normalize and keep the raw-data threshold."
        )


def enforce_connectivity(labels, min_size=0):
    """
    Split any adaptel that is not a single connected region.

    Adaptels compete for pixels: a later adaptel takes a pixel from an
    earlier one whenever it reaches it with a smaller accumulated distance.
    That competition is what gives the algorithm its boundary adherence, but
    it can also cut an earlier adaptel in two, leaving one label spread over
    separate patches. On a 400x400 3-band raster at the default threshold of
    60, about 10% of adaptels come out in more than one piece.

    Whether that matters depends on what you do next. It is harmless if the
    labels are only a lookup. It is not harmless for object-based analysis:
    zonal statistics over a split adaptel average two spatially separate
    patches into one "object".

    This relabels every connected component as its own adaptel, so the
    adaptel count rises and each output region is contiguous by
    construction. Nothing is merged and no pixel changes hands.

    Parameters
    ----------
    labels : ndarray, shape (rows, cols), dtype int32
        Adaptel ids, as returned by :func:`adaptels_from_array`. Negative
        values are nodata and pass through untouched.
    min_size : int
        Fragments of at most this many pixels are absorbed into an adjacent
        adaptel instead of becoming their own. 0 keeps every fragment.

    Returns
    -------
    labels : ndarray, shape (rows, cols), dtype int32
    n : int
        Number of adaptels after the split.

    Examples
    --------
    >>> labels, n = adaptels_from_array(data, threshold=60.0)  # doctest: +SKIP
    >>> labels, n = enforce_connectivity(labels)               # doctest: +SKIP
    """
    try:
        from scipy import ndimage
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "enforce_connectivity needs scipy.\n"
            "  conda install -c conda-forge scipy"
        ) from e

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0, got {min_size}")

    out = np.full(labels.shape, -1, dtype=np.int32)
    valid = labels >= 0
    if not valid.any():
        return out, 0

    # 4-connectivity, matching the neighbourhood the grower uses by default
    structure = ndimage.generate_binary_structure(2, 1)
    new_id = 0
    small = []

    # find_objects gives each adaptel's bounding box, so the component search
    # touches only that window. Scanning the whole raster once per adaptel is
    # quadratic in the adaptel count: ~6 s versus ~0.2 s on a 400x400 raster
    # carrying 9000 adaptels.
    boxes = ndimage.find_objects(labels + 1)     # find_objects wants 1-based
    for lab, box in enumerate(boxes):
        if box is None:
            continue                             # that id is unused
        window = labels[box]
        cc, ncomp = ndimage.label(window == lab, structure=structure)
        if ncomp == 1:
            out[box][cc == 1] = new_id           # untouched, the common case
            new_id += 1
            continue
        for c in range(1, ncomp + 1):
            frag = cc == c
            if min_size and frag.sum() <= min_size:
                small.append((box, frag))
                continue
            out[box][frag] = new_id
            new_id += 1

    # absorb slivers into whichever labelled neighbour they touch most. Done
    # after the main pass so they can attach to ids created during it.
    for box, frag in small:
        full = np.zeros(labels.shape, dtype=bool)
        full[box] = frag
        ring = ndimage.binary_dilation(full, structure=structure) & ~full
        nb = out[ring & (out >= 0)]
        if nb.size:
            out[full] = np.bincount(nb).argmax()
        else:
            out[full] = new_id                   # nothing adjacent: keep it
            new_id += 1

    out[~valid] = labels[~valid]
    return out, new_id


def create_adaptels(input_files, output_file=None,
                    threshold=60.0, distance='minkowski',
                    minkowski_p=2.0, queen_topology=False,
                    normalize=False, quiet=False):
    """
    Create scale-adaptive superpixels (adaptels) from geospatial raster data.
    
    Parameters
    ----------
    input_files : str or list of str
        Path(s) to input GeoTIFF file(s). Multiple files are treated
        as separate bands/layers.
    output_file : str, optional
        Path to output GeoTIFF with adaptel labels. If None, returns
        the labels array without writing to disk.
    threshold : float, default 60.0
        Energy threshold T controlling adaptel size. Lower values 
        produce smaller, more numerous adaptels. Higher values produce 
        fewer, larger adaptels.
    distance : str, default 'minkowski'
        Distance metric: 'minkowski', 'cosine', or 'angular'.
    minkowski_p : float, default 2.0
        Parameter p for Minkowski distance (2.0 = Euclidean).
    queen_topology : bool, default False
        If True, use 8-connectivity (Queen). If False, 4-connectivity (Rook).
    normalize : bool, default False
        If True, normalize input layers to [0, 1] before processing.
    quiet : bool, default False
        If True, suppress progress messages.
    
    Returns
    -------
    labels : np.ndarray, shape (rows, cols), dtype int32
        Adaptel labels. -9999 for nodata pixels.
    n_adaptels : int
        Number of adaptels created.
    
    Examples
    --------
    >>> from plgeoadaptels import create_adaptels
    >>> labels, n = create_adaptels('input.tif', 'output.tif', threshold=60.0)
    >>> print(f"Created {n} adaptels")
    
    >>> # Multiple input layers
    >>> labels, n = create_adaptels(['band1.tif', 'band2.tif', 'band3.tif'],
    ...                              'adaptels.tif', threshold=40.0)
    
    >>> # Return labels without writing file
    >>> labels, n = create_adaptels('input.tif', threshold=80.0)
    """
    # Validate parameters
    if isinstance(input_files, str):
        input_files = [input_files]
    
    distance_type = _validate_params(threshold, distance)
    connectivity = 8 if queen_topology else 4
    
    # Progress bar (optional)
    steps = ["Read", "Segment", "Write"] if output_file else ["Read", "Segment"]
    if normalize:
        steps.insert(1, "Normalize")
    use_tqdm = _HAS_TQDM and not quiet
    pbar = _tqdm(steps, desc="plGeoAdaptels", unit="step") if use_tqdm else None
    
    def _step(name):
        if pbar is not None:
            pbar.set_postfix_str(name)
            pbar.update(1)
    
    # Read input data
    if not quiet and not use_tqdm:
        print("Reading input files...", end=" ", flush=True)
    
    layers, mask, meta, cols, rows = read_raster(input_files)
    n_layers = layers.shape[0]
    
    if not quiet and not use_tqdm:
        print(f"OK ({cols}x{rows}, {n_layers} layer(s))")
    _step(f"Read {cols}x{rows}, {n_layers}b")
    
    # Normalize if requested
    if normalize:
        _validate_normalized_threshold(threshold, distance, n_layers,
                                       minkowski_p)
        if not quiet and not use_tqdm:
            print("Normalizing data...", end=" ", flush=True)
        layers = normalize_layers(layers, mask)
        if not quiet and not use_tqdm:
            print("OK")
        _step("Normalize")
    
    if not quiet and not use_tqdm:
        print(f"Distance: {distance}")
        print("Creating adaptels...", end=" ", flush=True)
    
    # Run algorithm
    t_start = time.time()
    
    labels, n_adaptels = _create_adaptels(
        layers, np.int32(n_layers), mask,
        np.int32(cols), np.int32(rows),
        float(threshold), np.int32(connectivity),
        np.int32(distance_type), float(minkowski_p)
    )
    
    t_elapsed = time.time() - t_start
    
    if not quiet and not use_tqdm:
        print(f"{n_adaptels} adaptels created in {t_elapsed:.4f} seconds")
    _step(f"{n_adaptels} adaptels in {t_elapsed:.1f}s")
    
    # Write output
    if output_file is not None:
        if not quiet and not use_tqdm:
            print("Writing results...", end=" ", flush=True)
        write_raster(output_file, labels, meta, cols, rows)
        if not quiet and not use_tqdm:
            print("OK")
        _step("Write")
    
    if pbar is not None:
        pbar.close()
    
    # Reshape to 2D
    labels_2d = labels.reshape(rows, cols)
    
    return labels_2d, n_adaptels


def adaptels_from_array(data, mask=None, threshold=60.0,
                        distance='minkowski', minkowski_p=2.0,
                        queen_topology=False, normalize=False):
    """
    Create adaptels directly from numpy arrays (no file I/O).
    
    Parameters
    ----------
    data : np.ndarray
        Input data. Shape can be:
        - (rows, cols) for single band
        - (bands, rows, cols) for multi-band
    mask : np.ndarray, optional, shape (rows, cols)
        Boolean or uint8 mask. True/1 = nodata.
        If None, NaN values are treated as nodata.
    threshold : float, default 60.0
    distance : str, default 'minkowski'
    minkowski_p : float, default 2.0
    queen_topology : bool, default False
    normalize : bool, default False
    
    Returns
    -------
    labels : np.ndarray, shape (rows, cols), dtype int32
    n_adaptels : int
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    n_layers, rows, cols = data.shape
    size = rows * cols
    
    # Flatten layers
    layers = data.reshape(n_layers, size).astype(np.float64)
    
    # Create mask
    if mask is None:
        mask_flat = np.zeros(size, dtype=np.uint8)
        for i in range(n_layers):
            mask_flat[np.isnan(layers[i])] = 1
    else:
        mask_flat = mask.ravel().astype(np.uint8)
    
    if normalize:
        _validate_normalized_threshold(threshold, distance, n_layers,
                                       minkowski_p)
        layers = normalize_layers(layers, mask_flat)

    distance_type = _validate_params(threshold, distance)
    connectivity = 8 if queen_topology else 4
    
    labels, n_adaptels = _create_adaptels(
        layers, np.int32(n_layers), mask_flat,
        np.int32(cols), np.int32(rows),
        float(threshold), np.int32(connectivity),
        np.int32(distance_type), float(minkowski_p)
    )
    
    return labels.reshape(rows, cols), n_adaptels
