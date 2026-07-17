"""
Core algorithm for Scale-Adaptive Superpixels (Adaptels).

Based on: R. Achanta, P. Marquez-Neila, P. Fua, S. Susstrunk,
"Scale-Adaptive Superpixels", Color and Imaging Conference, 2018.

Implementation based on plGeoAdaptels by Pawel Netzel,
University of Agriculture in Krakow, Poland.

Pure Python + Numba reimplementation for portability.
"""

import numpy as np
from numba import njit, int32, int64, float64


# ==========================================================================
# Min-heap operations (mirrors minlist.c)
# ==========================================================================

@njit(cache=True)
def heap_insert(h_dist, h_x, h_y, h_idx, h_size, dist, x, y, idx):
    """Insert into min-heap. Returns new size."""
    h_size += 1
    pos = h_size
    h_dist[pos] = dist
    h_x[pos] = x
    h_y[pos] = y
    h_idx[pos] = idx
    # Sift up
    while pos > 1:
        parent = pos // 2
        if h_dist[pos] < h_dist[parent]:
            h_dist[pos], h_dist[parent] = h_dist[parent], h_dist[pos]
            h_x[pos], h_x[parent] = h_x[parent], h_x[pos]
            h_y[pos], h_y[parent] = h_y[parent], h_y[pos]
            h_idx[pos], h_idx[parent] = h_idx[parent], h_idx[pos]
            pos = parent
        else:
            break
    return h_size


@njit(cache=True)
def heap_extract(h_dist, h_x, h_y, h_idx, h_size):
    """Extract minimum from heap. Returns (dist, x, y, idx, new_size)."""
    dist = h_dist[1]
    x = h_x[1]
    y = h_y[1]
    idx = h_idx[1]
    
    h_dist[1] = h_dist[h_size]
    h_x[1] = h_x[h_size]
    h_y[1] = h_y[h_size]
    h_idx[1] = h_idx[h_size]
    h_size -= 1
    
    # Sift down
    i = int64(1)
    while True:
        smallest = i
        left = i * 2
        right = i * 2 + 1
        if left <= h_size and h_dist[left] < h_dist[smallest]:
            smallest = left
        if right <= h_size and h_dist[right] < h_dist[smallest]:
            smallest = right
        if smallest != i:
            h_dist[i], h_dist[smallest] = h_dist[smallest], h_dist[i]
            h_x[i], h_x[smallest] = h_x[smallest], h_x[i]
            h_y[i], h_y[smallest] = h_y[smallest], h_y[i]
            h_idx[i], h_idx[smallest] = h_idx[smallest], h_idx[i]
            i = smallest
        else:
            break
    
    return dist, x, y, idx, h_size


# ==========================================================================
# Distance calculation (mirrors generate.c calc_distance)
# ==========================================================================

@njit(cache=True)
def calc_distance(layers, n_layers, cumul_sum, index, 
                  superpixel_size, distance_type, minkowski_p):
    """
    Calculate distance between a candidate pixel and the adaptel's 
    cumulative color. Mirrors the C version exactly.
    """
    dist = 0.0
    sp_size = float64(superpixel_size)

    if distance_type == 0:  # minkowski
        for i in range(n_layers):
            diff = abs(cumul_sum[i] - layers[i, index] * sp_size)
            dist += diff ** minkowski_p
        dist = (dist ** (1.0 / minkowski_p)) / sp_size
    elif distance_type == 1:  # cosine
        A = 1.0
        B = 1.0
        AB = 1.0
        for i in range(n_layers):
            mean = cumul_sum[i] / sp_size
            A += mean * mean
            B += layers[i, index] * layers[i, index]
            AB += layers[i, index] * mean
        dist = AB / np.sqrt(A * B)
    elif distance_type == 2:  # angular
        A = 1.0
        B = 1.0
        AB = 1.0
        for i in range(n_layers):
            mean = cumul_sum[i] / sp_size
            A += mean * mean
            B += layers[i, index] * layers[i, index]
            AB += layers[i, index] * mean
        val = AB / np.sqrt(A * B)
        val = min(max(val, -1.0), 1.0)
        dist = np.arccos(val) / np.pi
    
    return dist


# ==========================================================================
# Neighbor access
# ==========================================================================

DX = np.array([-1, 0, 1, 0, -1, 1, 1, -1], dtype=np.int32)
DY = np.array([0, -1, 0, 1, -1, -1, 1, 1], dtype=np.int32)


# ==========================================================================
# Main adaptel creation (mirrors generate.c)
# ==========================================================================

@njit(cache=True)
def _create_adaptels(layers, n_layers, mask, cols, rows,
                     threshold, connectivity, distance_type, minkowski_p):
    """
    Create adaptels for the entire raster.
    
    Faithful translation of the C code in generate.c.
    
    The algorithm:
    1. Find a starting pixel
    2. Insert it as the first seed
    3. For each seed, grow an adaptel using a priority queue
    4. When growth exceeds threshold, boundary pixels become new seeds
    5. Repeat until all seeds are processed
    6. Look for any remaining unlabeled valid pixels and repeat
    """
    size = int64(cols) * int64(rows)
    cols_i = int32(cols)
    rows_i = int32(rows)
    
    labels = np.full(size, int32(-1), dtype=np.int32)
    distances = np.zeros(size, dtype=np.float64)
    
    # Pre-compute shift indices (mirrors create_shiftIdx in C)
    dIdx = np.empty(8, dtype=np.int64)
    dIdx[0] = -1
    dIdx[1] = -int64(cols)
    dIdx[2] = 1
    dIdx[3] = int64(cols)
    dIdx[4] = -1 - int64(cols)
    dIdx[5] = 1 - int64(cols)
    dIdx[6] = 1 + int64(cols)
    dIdx[7] = -1 + int64(cols)
    
    # Seeds arrays (mirrors SEEDS struct)
    # Allocated at sqrt(size)*16 — sufficient for typical adaptel growth.
    # Overflow is handled: if n_seeds >= seeds_alloc, seed is silently dropped.
    seeds_alloc = max(int64(100000), int64(16) * int64(int(size ** 0.5)))
    seeds_x = np.empty(seeds_alloc, dtype=np.int32)
    seeds_y = np.empty(seeds_alloc, dtype=np.int32)
    seeds_idx = np.empty(seeds_alloc, dtype=np.int64)
    n_seeds = int64(0)
    
    # Heap arrays (mirrors MINLIST struct)
    # Heap holds boundary pixels of the CURRENT adaptel only.
    # Max heap size ≈ perimeter of largest adaptel ≈ 4*sqrt(area).
    # sqrt(size)*8 is a safe upper bound with margin.
    heap_alloc = max(int64(100000), int64(8) * int64(int(size ** 0.5)))
    h_dist = np.empty(heap_alloc + 1, dtype=np.float64)
    h_x = np.empty(heap_alloc + 1, dtype=np.int32)
    h_y = np.empty(heap_alloc + 1, dtype=np.int32)
    h_idx = np.empty(heap_alloc + 1, dtype=np.int64)
    
    current_label = int32(0)
    
    # Cumulative sum buffer
    cumul_sum = np.zeros(n_layers, dtype=np.float64)
    
    # Outer loop: find connected regions of unlabeled valid pixels
    start_idx = int64(0)
    
    while start_idx < size:
        # Find next valid unlabeled pixel
        found = False
        while start_idx < size:
            if mask[start_idx] == 0 and labels[start_idx] == -1:
                found = True
                break
            start_idx += 1
        
        if not found:
            break
        
        seed_col = int32(start_idx % cols_i)
        seed_row = int32(start_idx // cols_i)
        
        # Insert initial seed
        n_seeds = int64(1)
        seeds_x[0] = seed_col
        seeds_y[0] = seed_row
        seeds_idx[0] = start_idx
        
        # Process all seeds (mirrors the for loop in main C code)
        si = int64(0)
        while si < n_seeds:
            s_x = seeds_x[si]
            s_y = seeds_y[si]
            s_idx = seeds_idx[si]
            
            # === grow_adaptel (mirrors generate.c grow_adaptel) ===
            if labels[s_idx] < 0:
                index = s_idx
                
                # Init heap
                h_size = int64(0)
                h_size = heap_insert(h_dist, h_x, h_y, h_idx, h_size,
                                     0.0, s_x, s_y, s_idx)
                
                distances[index] = 0.0
                labels[index] = current_label
                
                nodata = (mask[index] == 1)
                superpixel_size = int64(1)
                
                # Init cumul_sum
                for l in range(n_layers):
                    if nodata:
                        cumul_sum[l] = 0.0
                    else:
                        cumul_sum[l] = layers[l, index]
                
                # Grow loop (mirrors while(minlist->N) in C)
                while h_size > 0:
                    cell_dist, cell_x, cell_y, cell_idx, h_size = \
                        heap_extract(h_dist, h_x, h_y, h_idx, h_size)
                    
                    for conn in range(connectivity):
                        nx = cell_x + DX[conn]
                        ny = cell_y + DY[conn]
                        nidx = cell_idx + dIdx[conn]
                        
                        # Boundary check
                        if nx < 0 or nx >= cols_i or ny < 0 or ny >= rows_i:
                            continue
                        
                        nodata_n = (mask[nidx] == 1)
                        if nodata_n:
                            continue
                        
                        if distances[cell_idx] < threshold:
                            if labels[nidx] != labels[cell_idx]:
                                dist = distances[cell_idx] + \
                                    calc_distance(layers, n_layers, cumul_sum,
                                                  nidx, superpixel_size,
                                                  distance_type, minkowski_p)
                                
                                if dist < distances[nidx] or labels[nidx] < 0:
                                    distances[nidx] = dist
                                    labels[nidx] = labels[cell_idx]
                                    
                                    # Add to cumul_sum
                                    for l in range(n_layers):
                                        cumul_sum[l] += layers[l, nidx]
                                    superpixel_size += 1
                                    
                                    if h_size < heap_alloc:
                                        h_size = heap_insert(
                                            h_dist, h_x, h_y, h_idx, h_size,
                                            distances[nidx], nx, ny, nidx)
                        else:
                            # Beyond threshold => new seed
                            if labels[nidx] < 0:
                                if n_seeds < seeds_alloc:
                                    seeds_x[n_seeds] = nx
                                    seeds_y[n_seeds] = ny
                                    seeds_idx[n_seeds] = nidx
                                    n_seeds += 1
                
                current_label += 1
            
            si += 1
        
        start_idx += 1
    
    # === remove_nodata_labels (mirrors generate.c) ===
    max_lab = current_label if current_label > 0 else int32(1)
    labels_list = np.full(max_lab, int32(-1), dtype=np.int32)
    
    for i in range(size):
        if mask[i] == 1:
            labels[i] = -9999
    
    new_id = int32(0)
    for i in range(size):
        if mask[i] == 0 and labels[i] >= 0:
            lab = labels[i]
            if labels_list[lab] == -1:
                labels_list[lab] = new_id
                new_id += 1
    
    for i in range(size):
        if mask[i] == 0 and labels[i] >= 0:
            labels[i] = labels_list[labels[i]]
    
    return labels, new_id
