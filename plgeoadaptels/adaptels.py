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
    return DISTANCE_MAP[distance]


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
