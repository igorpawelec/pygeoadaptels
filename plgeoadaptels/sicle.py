"""
SICLE — Superpixels through Iterative CLEarcutting.

Based on: F.C. Belém, I.B. Barcelos, L.M. João, B. Perret, J. Cousty,
S.J.F. Guimarães, A.X. Falcão, "Novel Arc-Cost Functions and Seed
Relevance Estimations for Compact and Accurate Superpixels",
Journal of Mathematical Imaging and Vision, 65:770–786, 2023.
DOI: 10.1007/s10851-023-01156-9

Pure Python + Numba reimplementation for geospatial raster data.
Reuses min-heap from plgeoadaptels.core.
"""

import numpy as np
from numba import njit, int32, int64, float64
from .core import heap_insert, heap_extract


# ══════════════════════════════════════════════════════════════════════
#  IFT kernel — Image Foresting Transform with fmax path-cost
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _ift_fmax(layers, n_layers, mask, cols, rows, seeds, n_seeds,
              labels_out, cost_out):
    """
    Seed-restricted IFT with fmax path-cost and wroot arc-cost.

    For each seed s, grows an optimum-path tree T_s by minimizing
    fmax(ρ) = max arc cost along the path.
    Arc cost: wroot(x,y) = ‖F(seed) − F(y)‖₂

    Parameters
    ----------
    layers : (n_layers, size) float64 — raster bands (flattened)
    mask : (size,) uint8 — 0 = valid, 1 = nodata
    seeds : (n_seeds,) int64 — flat indices of seed pixels
    labels_out : (size,) int32 — output: label per pixel (-1 = unassigned)
    cost_out : (size,) float64 — output: path cost per pixel
    """
    size = int64(cols) * int64(rows)

    # Initialize
    for i in range(size):
        labels_out[i] = int32(-1)
        cost_out[i] = 1e30

    # Store seed features for wroot
    seed_features = np.empty((n_seeds, n_layers), dtype=np.float64)
    for si in range(n_seeds):
        idx = seeds[si]
        for l in range(n_layers):
            seed_features[si, l] = layers[l, idx]

    # Heap — sized for boundary pixels, not all pixels
    heap_alloc = max(int64(100000), int64(8) * int64(int(size ** 0.5)))
    h_dist = np.empty(heap_alloc + 1, dtype=np.float64)
    h_x = np.empty(heap_alloc + 1, dtype=np.int32)
    h_y = np.empty(heap_alloc + 1, dtype=np.int32)
    h_idx = np.empty(heap_alloc + 1, dtype=np.int64)
    h_size = int64(0)

    # Insert all seeds with cost 0
    for si in range(n_seeds):
        idx = seeds[si]
        if mask[idx] == 0:
            cost_out[idx] = 0.0
            labels_out[idx] = int32(si)
            col = int32(idx % cols)
            row = int32(idx // cols)
            if h_size < heap_alloc:
                h_size = heap_insert(h_dist, h_x, h_y, h_idx, h_size,
                                     0.0, col, row, idx)

    # 8-adjacency shifts
    dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)
    dy = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)

    # Main IFT loop
    while h_size > 0:
        c_dist, c_x, c_y, c_idx, h_size = heap_extract(
            h_dist, h_x, h_y, h_idx, h_size)

        # Skip if already conquered with a better cost
        if c_dist > cost_out[c_idx]:
            continue

        seed_label = labels_out[c_idx]

        # Expand to 8-neighbors
        for k in range(8):
            nx = c_x + dx[k]
            ny = c_y + dy[k]
            if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                continue
            nidx = int64(ny) * int64(cols) + int64(nx)
            if mask[nidx] != 0:
                continue

            # wroot: ‖F(seed) − F(neighbor)‖₂
            arc_cost = 0.0
            for l in range(n_layers):
                diff = seed_features[seed_label, l] - layers[l, nidx]
                arc_cost += diff * diff
            arc_cost = arc_cost ** 0.5

            # fmax: max along path
            new_cost = max(c_dist, arc_cost)

            if new_cost < cost_out[nidx]:
                cost_out[nidx] = new_cost
                labels_out[nidx] = seed_label
                if h_size < heap_alloc:
                    h_size = heap_insert(h_dist, h_x, h_y, h_idx, h_size,
                                         new_cost, nx, ny, nidx)


# ══════════════════════════════════════════════════════════════════════
#  Seed relevance — Vsc (size × min contrast with neighbors)
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_seed_relevance(layers, n_layers, labels, mask,
                            cols, rows, n_seeds, saliency, use_saliency):
    """
    Compute relevance V(s) for each seed.

    Vsc(s) = |T_s| / |V| × min_{neighbor t} ‖μF(T_s) − μF(T_t)‖₂
    With saliency: V(s) = Vsc(s) × max(mean_O(T_s), max_∇O(T_s, T_t))

    Returns relevance array (n_seeds,) float64.
    """
    size = int64(cols) * int64(rows)

    # Accumulate per-tree stats
    tree_sum = np.zeros((n_seeds, n_layers), dtype=np.float64)
    tree_count = np.zeros(n_seeds, dtype=np.int64)
    tree_sal_sum = np.zeros(n_seeds, dtype=np.float64)

    for i in range(size):
        lab = labels[i]
        if lab < 0 or mask[i] != 0:
            continue
        tree_count[lab] += 1
        for l in range(n_layers):
            tree_sum[lab, l] += layers[l, i]
        if use_saliency:
            tree_sal_sum[lab] += saliency[i]

    # Mean features and saliency per tree
    tree_mean = np.zeros((n_seeds, n_layers), dtype=np.float64)
    tree_sal_mean = np.zeros(n_seeds, dtype=np.float64)
    total_valid = int64(0)
    for s in range(n_seeds):
        if tree_count[s] > 0:
            for l in range(n_layers):
                tree_mean[s, l] = tree_sum[s, l] / float64(tree_count[s])
            if use_saliency:
                tree_sal_mean[s] = tree_sal_sum[s] / float64(tree_count[s])
        total_valid += tree_count[s]

    if total_valid == 0:
        total_valid = 1

    # Find adjacent trees and compute contrasts
    # Scan all edges — if labels differ, trees are adjacent
    # For each seed, track: min color contrast, max saliency contrast
    min_contrast = np.full(n_seeds, 1e30, dtype=np.float64)
    max_sal_contrast = np.zeros(n_seeds, dtype=np.float64)

    dx = np.array([-1, 0, 1, 0], dtype=np.int32)
    dy = np.array([0, -1, 0, 1], dtype=np.int32)

    for y in range(rows):
        for x in range(cols):
            idx = int64(y) * int64(cols) + int64(x)
            lab_s = labels[idx]
            if lab_s < 0:
                continue
            for k in range(4):
                nx = x + dx[k]
                ny = y + dy[k]
                if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
                    continue
                nidx = int64(ny) * int64(cols) + int64(nx)
                lab_t = labels[nidx]
                if lab_t < 0 or lab_t == lab_s:
                    continue
                # Color contrast ‖μF(T_s) − μF(T_t)‖₂
                c_dist = 0.0
                for l in range(n_layers):
                    diff = tree_mean[lab_s, l] - tree_mean[lab_t, l]
                    c_dist += diff * diff
                c_dist = c_dist ** 0.5
                if c_dist < min_contrast[lab_s]:
                    min_contrast[lab_s] = c_dist

                # Saliency contrast
                if use_saliency:
                    s_dist = abs(tree_sal_mean[lab_s] - tree_sal_mean[lab_t])
                    if s_dist > max_sal_contrast[lab_s]:
                        max_sal_contrast[lab_s] = s_dist

    # Compute final relevance
    relevance = np.zeros(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        if tree_count[s] == 0:
            relevance[s] = 0.0
            continue
        v_size = float64(tree_count[s]) / float64(total_valid)
        mc = min_contrast[s]
        if mc > 1e29:
            mc = 0.0
        vsc = v_size * mc

        if use_saliency:
            pen = max(tree_sal_mean[s], max_sal_contrast[s])
            relevance[s] = vsc * pen
        else:
            relevance[s] = vsc

    return relevance


# ══════════════════════════════════════════════════════════════════════
#  Main SICLE algorithm
# ══════════════════════════════════════════════════════════════════════

def _run_sicle(layers, n_layers, mask, cols, rows,
               n_segments, n_oversampling, n_iterations,
               saliency, quiet):
    """
    Core SICLE algorithm.

    Returns (labels_2d, n_final) where labels_2d is (rows, cols) int32.
    """
    size = cols * rows
    n_seeds_current = n_oversampling
    use_saliency = saliency is not None

    # Flatten saliency if provided
    if use_saliency:
        sal_flat = saliency.ravel().astype(np.float64)
    else:
        sal_flat = np.empty(0, dtype=np.float64)

    # Step 1: Seed oversampling — random sampling
    valid_indices = np.where(mask == 0)[0]
    if len(valid_indices) < n_segments:
        raise ValueError(
            f"Not enough valid pixels ({len(valid_indices)}) "
            f"for {n_segments} segments."
        )
    n_seeds_current = min(n_oversampling, len(valid_indices))
    rng = np.random.default_rng(42)
    seeds = rng.choice(valid_indices, size=n_seeds_current, replace=False)
    seeds = seeds.astype(np.int64)

    if not quiet:
        print(f"  Seeds: {n_seeds_current} initial, target {n_segments}")

    # Preallocate
    labels = np.full(size, int32(-1), dtype=np.int32)
    cost = np.full(size, 1e30, dtype=np.float64)

    # Seed preservation curve: M(i) = max(N0^(1 − i/(Ω−1)), Nf)
    omega = max(n_iterations, 2)

    for iteration in range(omega):
        # Run IFT
        _ift_fmax(layers, int32(n_layers), mask,
                  int32(cols), int32(rows),
                  seeds, int64(len(seeds)),
                  labels, cost)

        if len(seeds) <= n_segments:
            break

        # Compute relevance for each seed
        relevance = _compute_seed_relevance(
            layers, int32(n_layers), labels, mask,
            int32(cols), int32(rows), int32(len(seeds)),
            sal_flat, use_saliency)

        # How many seeds to keep?
        t = float(iteration + 1) / float(omega - 1) if omega > 1 else 1.0
        m_keep = max(int(n_oversampling ** (1.0 - t)), n_segments)
        m_keep = min(m_keep, len(seeds))

        # Keep top m_keep seeds by relevance
        order = np.argsort(relevance)[::-1]  # descending
        keep_mask = order[:m_keep]

        # Build new seed array — remap labels
        new_seeds = np.empty(m_keep, dtype=np.int64)
        old_to_new = np.full(len(seeds), int32(-1), dtype=np.int32)
        for ni in range(m_keep):
            old_idx = keep_mask[ni]
            new_seeds[ni] = seeds[old_idx]
            old_to_new[old_idx] = int32(ni)

        seeds = new_seeds

        # Remap labels for removed seeds (they'll be reconquered)
        for i in range(size):
            lab = labels[i]
            if lab >= 0:
                new_lab = old_to_new[lab]
                if new_lab < 0:
                    labels[i] = int32(-1)
                    cost[i] = 1e30
                else:
                    labels[i] = new_lab

        if not quiet:
            print(f"  Iteration {iteration + 1}/{omega}: "
                  f"{len(seeds)} seeds remaining")

    # Final relabeling: consecutive IDs from 0
    n_final = len(seeds)
    labels_2d = labels.reshape(rows, cols)
    return labels_2d, int(n_final)


# ══════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════

def sicle_from_array(data, mask=None, n_segments=200,
                     n_oversampling=3000, n_iterations=2,
                     saliency=None, quiet=False):
    """
    Generate SICLE superpixels from numpy arrays.

    Parameters
    ----------
    data : np.ndarray
        Input raster. Shape (bands, rows, cols) or (rows, cols).
    mask : np.ndarray, optional, shape (rows, cols)
        0 = valid pixel, nonzero = nodata. Auto-detected from NaN if None.
    n_segments : int, default 200
        Desired number of superpixels.
    n_oversampling : int, default 3000
        Initial number of seeds (N₀ ≫ n_segments).
    n_iterations : int, default 2
        Maximum number of IFT iterations (Ω). 2 is optimal per Belém 2023.
    saliency : np.ndarray, optional, shape (rows, cols)
        Object saliency map [0, 1]. E.g. normalized CHM for forestry.
        Seeds near saliency borders are favored during removal.
    quiet : bool, default False
        Suppress progress messages.

    Returns
    -------
    labels : np.ndarray, shape (rows, cols), dtype int32
        Superpixel labels (0-based).
    n_superpixels : int
        Number of superpixels produced.

    Examples
    --------
    >>> import numpy as np
    >>> from plgeoadaptels.sicle import sicle_from_array
    >>> data = np.random.rand(3, 200, 200)
    >>> labels, n = sicle_from_array(data, n_segments=100)
    """
    # Validate and reshape
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    if data.ndim != 3:
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")

    n_layers, rows, cols = data.shape
    size = rows * cols

    # Flatten bands to (n_layers, size)
    layers = data.reshape(n_layers, size)

    # Build mask
    if mask is not None:
        mask_flat = mask.ravel().astype(np.uint8)
    else:
        mask_flat = np.zeros(size, dtype=np.uint8)
        for l in range(n_layers):
            mask_flat[np.isnan(layers[l])] = 1

    # Validate n_segments
    if n_segments < 1:
        raise ValueError(f"n_segments must be ≥ 1, got {n_segments}")
    if n_oversampling < n_segments:
        n_oversampling = n_segments * 10

    labels, n_sp = _run_sicle(
        layers, n_layers, mask_flat, cols, rows,
        n_segments, n_oversampling, n_iterations,
        saliency, quiet)

    return labels, n_sp


def create_sicle(input_files, output_file=None,
                 n_segments=200, n_oversampling=3000,
                 n_iterations=2, saliency_file=None,
                 quiet=False):
    """
    Generate SICLE superpixels from GeoTIFF file(s).

    Parameters
    ----------
    input_files : str or list of str
        Path(s) to input GeoTIFF file(s).
    output_file : str, optional
        Path to output GeoTIFF with superpixel labels.
    n_segments : int, default 200
        Desired number of superpixels.
    n_oversampling : int, default 3000
        Initial number of seeds.
    n_iterations : int, default 2
        Maximum IFT iterations.
    saliency_file : str, optional
        Path to GeoTIFF saliency map (single band, [0,1]).
    quiet : bool, default False
        Suppress progress messages.

    Returns
    -------
    labels : np.ndarray (rows, cols) int32
    n_superpixels : int
    """
    from .io import read_raster, write_raster
    import time

    if not quiet:
        print(f"SICLE superpixels (n_segments={n_segments}, "
              f"N₀={n_oversampling}, Ω={n_iterations})")

    # Read input raster
    layers, mask, meta, cols, rows = read_raster(input_files)
    n_layers = layers.shape[0]

    if not quiet:
        print(f"  Input: {rows}×{cols}, {n_layers} band(s)")

    # Read saliency if provided
    saliency = None
    if saliency_file is not None:
        sal_layers, _, _, _, _ = read_raster(saliency_file)
        saliency = sal_layers[0].reshape(rows, cols).astype(np.float64)
        # Normalize to [0, 1] if needed
        smin, smax = np.nanmin(saliency), np.nanmax(saliency)
        if smax > smin:
            saliency = (saliency - smin) / (smax - smin)
        if not quiet:
            print(f"  Saliency: {saliency_file}")

    t0 = time.time()
    labels, n_sp = sicle_from_array(
        layers.reshape(n_layers, rows, cols),
        mask=mask.reshape(rows, cols) if mask is not None else None,
        n_segments=n_segments,
        n_oversampling=n_oversampling,
        n_iterations=n_iterations,
        saliency=saliency,
        quiet=quiet)
    dt = time.time() - t0

    if not quiet:
        print(f"  Result: {n_sp} superpixels in {dt:.2f}s")

    # Write output
    if output_file is not None:
        write_raster(output_file, labels.ravel().astype(np.int32),
                     meta, cols, rows)
        if not quiet:
            print(f"  Written: {output_file}")

    return labels, n_sp
