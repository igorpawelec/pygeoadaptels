"""grow_seeds -- seeded spectral region growing ("inverse OBIA").

Each operator-placed point is grown into the region that looks like the pixel
it was placed on, and everything unseeded is left unassigned (-1). This is the
inverse of adaptels/sicle, which partition the whole image.

The growth itself is the SICLE IFT core, unchanged: ``_ift_fmax`` from
``sicle.py`` is called ONCE, with all seeds, and every feature in the spec is
obtained by preparing that kernel's inputs and post-processing its outputs.
The kernel stays frozen so the R and Python twins share one verified
algorithm and differ only in a thin wrapper -- see docs/SPEC_grow_seeds.md
section 3.

Copyright (C) 2026 Igor Pawelec. Licence: GPLv3.
"""

import math
from pathlib import Path

import numpy as np

from .sicle import _ift_fmax


def _point_to_pixel(x, y, x0, y0, w, h):
    """Map a map coordinate to a 0-based (row, col) pixel index.

    The contract (SPEC_grow_seeds section 5.1): pixel ``(r, c)`` covers the
    half-open extent ``[x0 + c*w, x0 + (c+1)*w) x (y0 - r*h, y0 - (r+1)*h]``,
    so a point exactly on a shared grid line belongs to the pixel to the
    *right* / *below*. ``floor`` delivers exactly that, and resolves the edge
    deterministically instead of leaving it to floating-point noise.

    The arithmetic is written out here, not delegated to
    ``rasterio.transform.rowcol``, so that the *specification* is the contract
    and the R twin can implement the identical formula. They may well agree;
    the point is not to depend on it (SPEC 5.2).

    Parameters
    ----------
    x, y : float
        Map coordinates in the raster's CRS.
    x0, y0 : float
        Coordinates of the raster's top-left *corner* -- the west edge of
        column 0 and the north edge of row 0. From a rasterio transform these
        are ``transform.c`` and ``transform.f``.
    w, h : float
        Pixel width and height, both positive. From a rasterio transform,
        ``transform.a`` and ``-transform.e``.

    Returns
    -------
    (row, col) : tuple of int
        0-based indices. May fall outside ``[0, rows) x [0, cols)`` when the
        point is outside the raster (including exactly on the outer east or
        south edge); the conversion is total and deterministic, and bounds
        are a separate validation step (SPEC 5.4).
    """
    col = math.floor((x - x0) / w)
    row = math.floor((y0 - y) / h)
    return int(row), int(col)


def _prepare(data, mask):
    """Reshape to (n_layers, size) float64 and build a (size,) uint8 mask.

    Mirrors sicle_from_array's prep exactly (0 = valid, nonzero = nodata,
    NaN-derived when mask is None), so grow_seeds sees the raster the same way
    SICLE does.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    if data.ndim != 3:
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    n_layers, rows, cols = data.shape
    size = rows * cols
    layers = data.reshape(n_layers, size)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (rows, cols):
            raise ValueError(
                f"mask must be shaped {(rows, cols)} to match the raster, "
                f"got {mask.shape}")
        mask_flat = mask.ravel().astype(np.uint8)
    else:
        mask_flat = np.zeros(size, dtype=np.uint8)
        for l in range(n_layers):
            mask_flat[np.isnan(layers[l])] = 1
    return layers, mask_flat, n_layers, rows, cols


def _validate_seeds(seeds, mask_flat, rows, cols):
    """Return flat int64 seed indices, in input order, or raise.

    The same checks sicle uses (unique, inside the raster, on unmasked
    pixels), but grow_seeds raises rather than dropping, because dropping
    would renumber the labels and break the point-order contract in SPEC
    section 4.3: labels == i must stay the segment grown from seeds[i].
    """
    seeds = np.asarray(seeds)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError(
            f"seeds must be an (n, 2) array of (row, col) pairs, got shape "
            f"{seeds.shape}")
    if seeds.shape[0] == 0:
        raise ValueError("seeds is empty; grow_seeds needs at least one point")
    seeds = seeds.astype(np.int64)
    r, c = seeds[:, 0], seeds[:, 1]
    outside = (r < 0) | (r >= rows) | (c < 0) | (c >= cols)
    if outside.any():
        raise ValueError(
            f"seed(s) at input index {list(np.where(outside)[0])} lie outside "
            f"the raster ({rows}x{cols})")
    flat = (r * cols + c).astype(np.int64)
    on_nodata = mask_flat[flat] != 0
    if on_nodata.any():
        raise ValueError(
            f"seed(s) at input index {list(np.where(on_nodata)[0])} fall on "
            f"nodata pixels")
    # Duplicate pixels would give two labels the same seed and silently merge
    # two of the operator's points; raise with the offending input indices so
    # they can find them in their point layer.
    uniq, first, counts = np.unique(flat, return_index=True,
                                    return_counts=True)
    if (counts > 1).any():
        dup_pixels = uniq[counts > 1]
        dupes = sorted(int(i) for i in np.where(np.isin(flat, dup_pixels))[0])
        raise ValueError(
            f"seed(s) at input index {dupes} land on the same pixel(s); two "
            f"points in one pixel would merge two segments and break the "
            f"label contract")
    return flat


def _window_median(stack, mask_flat, r, c, k, rows, cols):
    """Per-band median over the unmasked pixels of the k*k window at (r, c).

    Computed on ``stack`` -- i.e. after band weighting -- so the seed
    signature is in the same units as the neighbours it will be compared
    against. The SPEC section 6 sketch took the median of the raw layers and
    wrote it into the weighted stack; that mixes units. Masked pixels are
    excluded; the seed itself is unmasked, so there is always a value.
    """
    half = k // 2
    r0, r1 = max(0, r - half), min(rows, r + half + 1)
    c0, c1 = max(0, c - half), min(cols, c + half + 1)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    idx = (rr * cols + cc).ravel()
    idx = idx[mask_flat[idx] == 0]
    return np.median(stack[:, idx], axis=1)


def _dist_to_own_seed(labels, seeds, rows, cols):
    """Euclidean pixel distance from each assigned pixel to its own seed.

    ``labels[p]`` is the index into ``seeds`` of the seed that conquered p
    (SPEC 4.3), so the seed coordinate is a gather. Unassigned pixels (-1) get
    +inf, so a radius cap never resurrects one.
    """
    # labels is the flat (size,) array at the call site, so the coordinate
    # grids are flat too -- a reshape here would mismatch the boolean index.
    out = np.full(labels.shape, np.inf, dtype=np.float64)
    assigned = labels >= 0
    rr, cc = np.divmod(np.arange(rows * cols), cols)
    lab = labels[assigned]
    dr = rr[assigned] - seeds[:, 0][lab]
    dc = cc[assigned] - seeds[:, 1][lab]
    out[assigned] = np.sqrt(dr * dr + dc * dc)
    return out


def _fill_holes(labels2d, mask2d):
    """Fill unassigned pockets that sit fully inside one crown.

    A pocket is filled only when it is a 4-connected group of valid ``-1``
    pixels that does not reach the raster border, does not touch nodata, and
    is bounded by exactly one label. Those conditions are what keep it from
    swallowing the nodata region or bridging two crowns, and they are
    topological, so the R twin reaches the same result regardless of how
    components are enumerated.
    """
    from scipy import ndimage
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    out = labels2d.copy()
    fillable = (labels2d < 0) & (mask2d == 0)
    if not fillable.any():
        return out
    border = np.zeros(labels2d.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    outside = ndimage.binary_propagation(border & fillable, mask=fillable,
                                         structure=struct)
    holes = fillable & ~outside
    cc, n = ndimage.label(holes, structure=struct)
    for k in range(1, n + 1):
        pocket = cc == k
        nb = ndimage.binary_dilation(pocket, structure=struct) & ~pocket
        if (mask2d[nb] != 0).any():          # touches nodata: not interior
            continue
        nb_labels = np.unique(labels2d[nb & (labels2d >= 0)])
        if len(nb_labels) == 1:
            out[pocket] = nb_labels[0]
    return out


def grow_seeds(data, seeds, mask=None, max_cost=None, band_weights=None,
               compactness=0.0, seed_window=1, max_radius=None,
               fill_holes=False, return_cost=False, quiet=False):
    """Grow each seed into the region that looks like the pixel it sits on.

    Calls the SICLE IFT kernel once, with every seed, and never removes one,
    so ``labels == i`` is exactly the region grown from ``seeds[i]`` and -1 is
    unassigned (SPEC_grow_seeds section 4.3). Every option is prepared into or
    read out of that single unmodified call.

    Parameters
    ----------
    data : ndarray (bands, rows, cols) or (rows, cols)
    seeds : ndarray (n, 2) of (row, col), 0-based
    mask : ndarray (rows, cols), optional
        0 = valid, nonzero = nodata. NaN-derived when None.
    max_cost : float, optional
        Cost cap in band units -- a tolerance on the minimax spectral
        deviation from the seed. None keeps every reachable pixel (a
        partition, as the kernel does today). If the input was CIELAB this is
        a Delta-E tolerance (SPEC 3.1, 10).
    band_weights : array-like (bands,), optional
        Per-band multipliers applied before the distance (SPEC 3.2).
    compactness : float, default 0.0
        SLIC-style spatial term, in feature-units per pixel; 0 reproduces pure
        spectral growth exactly (SPEC 3.3).
    seed_window : int, default 1
        Side k of a k*k median used as the seed signature; 1 is the raw pixel
        (SPEC 3.4).
    max_radius : float, optional
        Hard limit in pixels; pixels further than this from their seed go back
        to -1 (SPEC 3.5).
    fill_holes : bool, default False
        Fill unassigned pockets that sit fully inside one crown -- the
        interior pixels a ``max_cost`` cut left as -1 (a bright spot or a
        shadow gap). A pocket is filled only when it does not reach the raster
        border, does not touch nodata, and is bounded by a single label, so it
        cannot swallow the nodata region or bridge two crowns. Needs scipy.
    return_cost : bool, default False
    quiet : bool, default False

    Returns
    -------
    labels : (rows, cols) int32   -- label i is seeds[i]; -1 = unassigned
    cost   : (rows, cols) float64 -- only if return_cost
    """
    layers, mask_flat, n_layers, rows, cols = _prepare(data, mask)
    flat_seeds = _validate_seeds(seeds, mask_flat, rows, cols)
    seeds_rc = np.asarray(seeds, dtype=np.int64)
    size = rows * cols

    # A copy, because everything below rewrites the band stack and the caller
    # keeps `data`.
    stack = layers.astype(np.float64, copy=True)

    if band_weights is not None:
        bw = np.asarray(band_weights, dtype=np.float64)
        if bw.shape != (n_layers,):
            raise ValueError(
                f"band_weights must have one entry per band ({n_layers}), got "
                f"{bw.shape}")
        stack = stack * bw[:, None]

    if seed_window > 1:
        if seed_window % 2 == 0:
            raise ValueError(
                f"seed_window must be odd, got {seed_window}: an even window "
                f"has no centre pixel to anchor on")
        # All medians from the pre-write stack, then written, so one seed's
        # window cannot be perturbed by another seed's overwrite when two sit
        # within seed_window of each other.
        meds = [_window_median(stack, mask_flat, int(r), int(c),
                               seed_window, rows, cols)
                for r, c in seeds_rc]
        for (r, c), med in zip(seeds_rc, meds):
            stack[:, int(r) * cols + int(c)] = med

    if compactness > 0:
        # Two coordinate bands scaled by lambda. Because wroot measures
        # distance to the *seed*, the appended terms evaluate to
        # lambda^2 * d_euclid(seed, pixel)^2 inside the norm -- SLIC-style
        # compactness anchored at the seed, with no kernel change (SPEC 3.3).
        # This relies on wroot being seed-anchored; if the kernel ever becomes
        # a neighbour-difference cost this silently turns into a gradient term.
        rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float64)
        extra = np.stack([(compactness * rr).ravel(),
                          (compactness * cc).ravel()])
        stack = np.concatenate([stack, extra], axis=0)

    n_used = stack.shape[0]
    stack = np.ascontiguousarray(stack, dtype=np.float64)

    labels = np.empty(size, dtype=np.int32)
    cost = np.empty(size, dtype=np.float64)
    _ift_fmax(stack, np.int32(n_used), mask_flat,
              np.int32(cols), np.int32(rows),
              flat_seeds, np.int64(len(flat_seeds)),
              labels, cost)

    if max_cost is not None:
        labels[cost > max_cost] = -1
    if max_radius is not None:
        far = _dist_to_own_seed(labels, seeds_rc, rows, cols) > max_radius
        labels[far] = -1

    labels_2d = labels.reshape(rows, cols)
    cost_2d = cost.reshape(rows, cols)

    # Fill unassigned pockets left inside a crown by the max_cost cut. Done on
    # the 2-D form, after the cuts that create the holes.
    if fill_holes:
        labels_2d = _fill_holes(labels_2d, mask_flat.reshape(rows, cols))

    if not quiet:
        n_assigned = int((labels_2d >= 0).sum())
        print(f"grow_seeds: {len(flat_seeds)} seeds, {n_assigned} px "
              f"assigned, {size - n_assigned} px unassigned")

    if return_cost:
        return labels_2d, cost_2d
    return labels_2d


def _read_points(points, points_layer, raster_crs, quiet):
    """Return a list of (x, y) map coordinates in the raster's CRS.

    ``points`` is either a path to any OGR-readable point layer (read with
    fiona, so .shp / .gpkg / .geojson all work) or an (n, 2) array already in
    the raster CRS. A layer that carries a CRS different from the raster's is
    reprojected, and the fact is reported unless quiet -- never assume they
    match (SPEC 5.3).
    """
    if not isinstance(points, (str, Path)):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "points array must be (n, 2) of (x, y) map coordinates in the "
                f"raster CRS, got shape {arr.shape}")
        return [(float(x), float(y)) for x, y in arr]

    try:
        import fiona
        from fiona.transform import transform as fiona_transform
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fiona is required to read a point layer.\n"
            "  conda install -c conda-forge fiona") from e
    from rasterio.crs import CRS as RioCRS

    xs, ys = [], []
    with fiona.open(str(points), layer=points_layer) as src:
        pts_crs = src.crs
        for feat in src:
            geom = feat["geometry"]
            if geom is None or geom["type"] != "Point":
                raise ValueError(
                    f"grow_seeds needs a Point layer; found geometry type "
                    f"{None if geom is None else geom['type']!r}. Digitise the "
                    f"objects as points.")
            x, y = geom["coordinates"][:2]
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        raise ValueError(f"the point layer {points} is empty")

    if raster_crs is not None and pts_crs:
        rc = RioCRS.from_user_input(raster_crs)
        pc = RioCRS.from_user_input(pts_crs)
        if rc != pc:
            xs, ys = fiona_transform(pc.to_wkt(), rc.to_wkt(), xs, ys)
            if not quiet:
                print(f"  Reprojected {len(xs)} points from {pc.to_string()} "
                      f"to the raster CRS {rc.to_string()}")

    return list(zip(xs, ys))


def grow_seeds_from_files(input_files, points, output_file=None, polygons=None,
                          points_layer=None, quiet=False, **kwargs):
    """Grow an operator's point layer into regions on a raster from disk.

    Reads the raster(s), reads the point layer, converts each point to the
    pixel it falls on (SPEC section 5), grows every point in one call, and
    writes a label raster and/or crown polygons. ``kwargs`` are passed straight
    to :func:`grow_seeds` (``max_cost``, ``band_weights``, ``compactness``,
    ``seed_window``, ``max_radius``, ...).

    Parameters
    ----------
    input_files : str or list of str
        Raster path(s); several are stacked band-wise (same extent/grid/CRS).
        Feed the bands you want the growth to see -- raw RGB, CIELAB, an index
        -- prepared upstream (SPEC section 1, 7).
    points : str, Path, or (n, 2) array
        A point layer (any OGR format) or map coordinates already in the
        raster CRS. One point per object of interest.
    output_file : str, optional
        Where to write the ``int32`` label raster (-1 = unassigned, set as its
        nodata). None writes no raster.
    polygons : str, optional
        Where to write crown polygons. Format follows the extension
        (``.gpkg`` / ``.shp`` / ``.geojson``); ``.gpkg`` is the default for an
        unrecognised one, because the label-to-attribute join a caller does
        next is cleaner without the shapefile field-name limits (SPEC 4.3).
        None writes no polygons.
    points_layer : str, optional
        Layer name inside a multi-layer vector source.
    quiet : bool, default False

    Returns
    -------
    labels : ndarray (rows, cols) int32
        As :func:`grow_seeds`; label ``i`` is the region grown from the i-th
        point, in the layer's feature order, so it joins back to the point's
        attributes.
    """
    from .io import read_raster, write_raster

    layers, mask_flat, meta, cols, rows = read_raster(input_files)
    n_layers = layers.shape[0]
    data = layers.reshape(n_layers, rows, cols)
    mask2d = mask_flat.reshape(rows, cols)

    transform = meta["transform"]
    x0, y0 = transform.c, transform.f
    w, h = transform.a, -transform.e
    raster_crs = meta.get("crs")

    xy = _read_points(points, points_layer, raster_crs, quiet)
    seeds = np.array([_point_to_pixel(x, y, x0, y0, w, h) for x, y in xy],
                     dtype=np.int64)

    labels = grow_seeds(data, seeds, mask=mask2d, quiet=quiet, **kwargs)

    if output_file is not None:
        # -1 as nodata so a GIS reads unassigned as missing rather than as a
        # segment numbered minus one.
        write_raster(output_file, labels, meta, cols, rows, nodata=-1)
        if not quiet:
            print(f"  Wrote labels: {output_file}")

    if polygons is not None:
        from .vectorize import vectorize_adaptels
        crs_wkt = raster_crs.to_wkt() if raster_crs is not None else None
        # connectivity=8, because the IFT grows over 8 neighbours, so a crown
        # is one 8-connected region. Polygonising it 4-connected (the adaptels
        # default) splits a crown that pinches on a diagonal into separate
        # rings, which looks like a defect and is only an artefact.
        n = vectorize_adaptels(labels, transform, crs_wkt, polygons,
                               driver="GPKG", nodata=-1, connectivity=8,
                               quiet=quiet)
        if not quiet:
            print(f"  Wrote {n} polygons: {polygons}")

    return labels
