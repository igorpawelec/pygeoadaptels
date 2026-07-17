# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **`cosine` distance was inverted.** The kernel returned cosine *similarity*
  where the caller compares against a growth threshold: 1.0 for identical
  spectra, falling towards 0 as they diverged. The threshold therefore worked
  backwards and barely moved the result — on a 200x200 raster the adaptel
  count went from 19,737 to 18,636 across the metric's entire useful range.
  Now returns `1 - similarity`.
- **A threshold outside a metric's scale is rejected.** `cosine` and `angular`
  produce distances in [0, 1] by construction, but the package default of 60
  is scaled for `minkowski`. Passing it merged whole rasters into a single
  adaptel and looked like a broken algorithm rather than a misread parameter.
  The error now says so, and suggests a starting value.
- **`adaptels_from_array` silently ignored `distance`.** It used
  `DISTANCE_MAP.get(distance, 0)`, so an unrecognised name fell back to
  minkowski without a word: `distance='cosin'` returned a full minkowski
  segmentation. `create_adaptels` raised correctly on the same input. Both
  entry points now share one validator and cannot drift apart again.
- **`adaptels_from_array` did not validate `threshold`.** A negative threshold
  returned one adaptel per pixel and reported it as a result.
- `MANIFEST.in` referenced `LICENSE.md`; the repository ships `LICENSE`.

### Changed
- **`plgeoadaptels/test_adaptels.py` moved to `examples/quickstart.py`.** It
  was not a test but a cell-based scratch script with hardcoded paths to
  `D:\Projects\Test\`. Being named `test_*` inside the package, it broke a
  plain `pytest` run with a collection error and shipped into site-packages.
  It now reads the raster the repository already contains and runs anywhere.
- `test_distance_metrics` asserted `n >= 1`, which holds for any result
  including a single degenerate adaptel — which is how the broken `cosine`
  passed CI. It now requires a real segmentation.
- Licence declared as an SPDX expression with `license-files` (PEP 639); the
  TOML table form is deprecated and stops working 2027-02-18. Needs
  setuptools >= 77.
- Unused imports removed; `pyflakes` is clean.

## [0.2.0] — 2026-03-04

### Added
- Full repo structure (pyproject.toml, CITATION.cff, CONTRIBUTING.md, CI)
- GitHub Actions CI smoke test
- `environment.yaml` for reproducible conda environments
- `requirements.txt` for reference
- `adaptels_from_array()` for numpy-only workflows
- `--no-deps` installation guide to avoid breaking conda envs

### Changed
- Replaced `setup.py` with `pyproject.toml`
- Single-source versioning via `importlib.metadata`
- Empty `dependencies` in pyproject.toml (install via conda)
- Version bump to 0.2.0

## [0.1.0] — 2026-03-04

### Added
- Initial Python + Numba reimplementation of plGeoAdaptels
- `create_adaptels()` — GeoTIFF file-based API
- `adaptels_from_array()` — numpy array API
- CLI: `plgeoadaptels -i input.tif -o output.tif -t 60`
- Distance metrics: Minkowski, cosine, angular
- 4-connectivity (rook) and 8-connectivity (queen)
- Layer normalization option
- Min-heap priority queue (Numba JIT)
- Rasterio-based I/O with nodata handling
