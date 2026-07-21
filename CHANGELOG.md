# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **SICLE dropped pixels on large rasters.** The IFT priority queue was
  capped at 100,000 entries, and an insert past the cap was skipped — but
  `cost_out` and `labels_out` were written *before* the capacity check, so
  the pixel counted as conquered while never entering the queue. Its subtree
  stopped growing and the pixels behind it kept label `-1`. At 700x700 that
  lost 7 valid pixels, at 900x900 it lost 93, and the loss scales with the
  raster. The heap now grows on demand. Verified bit-identical to the old
  output wherever the cap was not reached, including `SNP_21_2020_1.tif`.
- **A NaN in the SICLE saliency map hijacked the seed ranking.** Relevance
  was sorted with `argsort(...)[::-1]`; NaN sorts to the end ascending, so
  reversing moved every NaN-relevance seed to the *head* of the ranking,
  ahead of every seed whose relevance was real. One NaN band produced a
  superpixel covering 139,820 of 160,000 pixels, with no exception and no
  warning — and nodata in a saliency raster is ordinary, a CHM has it.
  `saliency` is now validated for shape and for NaN over pixels the mask
  calls valid, and the ranking puts NaN last regardless.
- **`normalize=True` merged the whole raster at the default threshold.**
  Normalizing rescales every band to [0, 1], which caps the largest possible
  minkowski distance at `n_layers**(1/p)` — about 1.73 for three bands at
  p=2 — while the package default of 60 is scaled for raw data. On the test
  scene 2,792 adaptels became exactly 1, silently. This is the same failure
  as an out-of-scale `cosine` threshold, which already raised; it now fails
  the same way, naming the ceiling it cannot cross.
- **`n_oversampling` below `n_segments` was corrected in silence.** SICLE
  only removes seeds, so the target is unreachable from there and the value
  was replaced by `n_segments * 10` with nothing in the output to say so.
  It still corrects, but warns.
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
- **The README's own CLI example was broken.** It showed
  `-t 40.0 -d cosine`, a threshold four times outside that metric's range,
  so anyone copying it got a collapsed raster. Now `-t 0.03 -d angular`,
  which is verified to run.
- **The CLI printed tracebacks.** A missing raster surfaced as a raw
  `RasterioIOError` stack; user mistakes now get one clean line on stderr.
- **`python -m plgeoadaptels` always exited 0.** `__main__.py` called
  `main()` without passing its return value to `sys.exit()`, so a failed run
  reported success and any script checking `$?` missed it.

### Removed
- **Dead label remapping in the SICLE seed-removal loop.** Between
  iterations it rewrote `labels` and `cost` for every pixel, but
  `_ift_fmax` reinitialises both on entry, so the work was discarded
  before it could be read: a full O(size) pass per iteration that changed
  nothing. Confirmed by feeding `_ift_fmax` deliberately corrupted arrays
  and getting bit-identical output.

### Changed
- **SICLE labels are documented as 8-connected, and must not be passed to
  `enforce_connectivity()`.** That helper tests 4-connectivity, matching the
  adaptels grower's default neighbourhood, while SICLE's IFT expands to all
  8 neighbours. Measured on both a random scene and `SNP_21_2020_1.tif`,
  SICLE labels are exactly 8-connected — one component per label — and
  around 20x fragmented under a 4-connected reading, so the helper reports
  a defect that is not there.
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
- README documents the threshold scale per metric, with measured adaptel
  counts, since one threshold does not carry across them.

### Added
- **`enforce_connectivity()`**, which splits any adaptel that is not a single
  connected region. The competition between adaptels can cut an earlier one
  in two, so about 10% of adaptels at the default threshold arrive in more
  than one piece — which matters as soon as anything computes zonal
  statistics per label. Not applied automatically: it changes the adaptel
  count, so it stays an explicit call.
- GitHub Actions CI: tests, lint and the example, on Linux/macOS/Windows
  across Python 3.9-3.12. The 0.2.0 entry below claimed a CI smoke test, but
  no workflow was ever committed.
- Tests for the distance metrics: each must return 0 for identical spectra
  and grow with dissimilarity, and each must respond to its own threshold.
  The previous `test_distance_metrics` asserted only `n >= 1`, which holds
  even when a metric collapses the raster to a single adaptel — which is
  exactly how the broken `cosine` passed.

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
