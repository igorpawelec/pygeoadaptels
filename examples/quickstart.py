# -*- coding: utf-8 -*-
"""
plGeoAdaptels quickstart, on the GeoTIFF shipped with the repository.

    pip install -e .
    python examples/quickstart.py

Written in cells, so it also runs a block at a time in Spyder, VS Code or
Positron. Paths are relative to the repository root; run it from there.
"""

#%% Setup
import os
import numpy as np
import time

# repo root, so the script works wherever it is launched from
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "examples", "output")
os.makedirs(OUT, exist_ok=True)

# If installed with pip install -e . this just works:
from plgeoadaptels import create_adaptels, adaptels_from_array

#%% Test 1: Basic numpy array test (no file needed)
print("=" * 60)
print("TEST 1: Numpy array - uniform data")
print("=" * 60)

data = np.ones((1, 100, 100), dtype=np.float64) * 42.0
labels, n = adaptels_from_array(data, threshold=30.0)
print(f"Result: {n} adaptels (expected: 1)")

#%% Test 2: Two distinct regions
print("=" * 60)
print("TEST 2: Numpy array - two regions")
print("=" * 60)

data2 = np.zeros((1, 100, 100), dtype=np.float64)
data2[0, :50, :] = 10.0
data2[0, 50:, :] = 200.0
labels2, n2 = adaptels_from_array(data2, threshold=30.0)
print(f"Result: {n2} adaptels (expected: ~2)")

#%% Test 3: With real GeoTIFF file
print("=" * 60)
print("TEST 3: GeoTIFF - SNP_21_2020_1.tif")
print("=" * 60)

inp = os.path.join(HERE, "test_data", "SNP_21_2020_1.tif")
out = os.path.join(OUT, "adaptels_python.tif")

t0 = time.time()
labels3, n3 = create_adaptels(inp, out, threshold=60.0)
dt = time.time() - t0
print(f"Result: {n3} adaptels in {dt:.2f}s")
print(f"Output: {out}")

#%% Test 4: With normalization
print("=" * 60)
print("TEST 4: GeoTIFF - with normalization")
print("=" * 60)

out4 = os.path.join(OUT, "adaptels_normalized.tif")

labels4, n4 = create_adaptels(inp, out4, threshold=20.0, normalize=True)
print(f"Result: {n4} adaptels")

#%% Test 5: Different distance metrics
print("=" * 60)
print("TEST 5: Cosine distance")
print("=" * 60)

out5 = os.path.join(OUT, "adaptels_cosine.tif")

labels5, n5 = create_adaptels(inp, out5, threshold=60.0, distance='cosine')
print(f"Result: {n5} adaptels")

#%% Test 6: 8-connectivity
print("=" * 60)
print("TEST 6: Queen topology (8-connectivity)")
print("=" * 60)

out6 = os.path.join(OUT, "adaptels_queen.tif")

labels6, n6 = create_adaptels(inp, out6, threshold=60.0, queen_topology=True)
print(f"Result: {n6} adaptels")

#%% Summary
print()
print("=" * 60)
print("ALL TESTS COMPLETE!")
print("=" * 60)
