# Changelog

## [0.8.1] — 2026-07-22

### Fixed
- **`create_sicle` crashed on a cp1250 console with its default settings.**
  The progress header read `N₀=... Ω=...` and `{rows}×{cols}` — U+2080,
  U+03A9, U+00D7 — and `quiet` defaults to `False`, so the first `print`
  raised `UnicodeEncodeError` on a Polish Windows console before any work
  began. Two error messages had the same problem (`≥` U+2265 in the
  `n_segments` check, `—` U+2014 in the normalize-threshold guidance): once
  raised and printed by the CLI they would crash the same way. All are ASCII
  now. A regression test drives `create_sicle` through an ASCII-strict stream,
  and `test_console_ascii.py` statically rejects non-ASCII in any
  `print`/`raise`/`warn` string across the package.


All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.8.0] — 2026-07-22

### Added
- **`cli.build_parser()`, and `main()` now takes `argv`.** The CLI had no
  tests, and it is where this package's only two shipped defects lived: a raw
  traceback where a message belonged, and `python -m plgeoadaptels` returning
  0 after a failure, so a script checking `$?` saw success. Both were fixed
  in 0.3.0 with nothing to keep them fixed.

  `main()` read `sys.argv` directly, so testing it meant monkeypatching. It
  now accepts `argv`, defaulting to `sys.argv[1:]`, which is what pyHRG's CLI
  already did.
- Seven CLI tests, including a subprocess check that `python -m plgeoadaptels`
  propagates the exit code. Verified to discriminate: removing the `sys.exit()`
  from `__main__.py` makes it fail.

## [0.7.0] — 2026-07-22

### Changed
- **SICLE ranks seeds with a stable sort.** `np.argsort` defaults to an
  introsort, whose order among equal scores is unspecified. Two seeds can
  carry exactly the same relevance, and swapping them does more than
  renumber the output: the seed index becomes the label, and `_ift_fmax`
  awards a contested pixel to whichever seed reached it first, so the order
  decides the partition. One such swap moved 65 pixels on a 45x58 scene.

  It also meant this package's own output was not guaranteed across NumPy
  versions, since nothing obliges an introsort to order ties the same way
  twice.

  **Measured cost on `SNP_21_2020_1.tif`: none.** Bit-identical at 200 and
  800 segments. It is a minor rather than a patch release only because it
  can change the result wherever a tie exists, as it did on the synthetic
  scene above.

  Found while porting SICLE to R. Both C kernels matched the Python bit for
  bit -- the IFT labels and the relevance vector, zero difference -- and the
  divergence was entirely in how the two languages broke a tie in the
  ranking. R's `order()` is stable, so `rev(order(rank))` now reproduces
  `argsort(rank, kind="stable")[::-1]` element for element.

## [0.6.0] — 2026-07-22

### Added
- **`seeds` and `random_state` on `sicle_from_array()` and `create_sicle()`.**
  The starting seeds were sampled with `np.random.default_rng(42)`, hardcoded,
  with no way to vary or supply them.

  `random_state` closes the smaller gap. `seeds` closes the one that blocked
  the R port: `Generator.choice(replace=False)` cannot be reproduced outside
  NumPy — it needs PCG64 *and* the internals of `choice`, and neither carries
  a stability guarantee. Reimplementing an undocumented ordering detail of a
  third-party library is precisely what left rHRG disagreeing with
  `scikit-image`'s watershed on 0.25 % of pixels, and doing it again knowingly
  would be a poor trade. Belém et al. treat the sampling as a free choice
  ("one may opt for a simple random sampling"), so it is not part of the
  algorithm; supplying seeds lets two implementations be compared on the part
  that is.

  Seeds are `(n, 2)` (row, col) pairs, validated for shape, bounds, nodata,
  duplicates and count. Supplying the seeds the sampler would have drawn
  reproduces its output bit for bit, which is the property the cross-language
  check rests on.

### Fixed
- **The seed-preservation curve used the requested N0, not the placed one.**
  `M(i)` was computed from `n_oversampling` while the actual seed count is
  `min(n_oversampling, valid pixels)`. On a raster with fewer valid pixels
  than requested, the curve was shaped for more seeds than existed. It now
  uses the count actually placed, which is also the only sensible reading
  when seeds are supplied directly.

### Note
The default path is unchanged: `random_state` defaults to the 42 that was
hardcoded, and the output is bit-identical to 0.5.1 on `SNP_21_2020_1.tif`
at two segment counts and on synthetic scenes at two iteration counts.

## [0.5.1] — 2026-07-22

### Fixed
- **`pip install plgeoadaptels` produced a package that could not be imported.**
  `dependencies` was an empty list, paired with a README recommending
  `pip install --no-deps` so that pip would not overwrite conda's
  GDAL/PROJ-linked builds. That flag alone achieves it; the empty list
  additionally broke every ordinary pip install, which surfaced as
  `ModuleNotFoundError: No module named 'numpy'` in the rgeoadaptels CI job
  that installs this package to cross-check against it.

  `numpy` and `numba` are now declared, because they are what `import
  plgeoadaptels` actually needs. Neither is linked against GDAL or PROJ, so
  the conflict the README warns about does not apply to them. `rasterio`,
  `fiona` and `scipy` stay extras: all three are imported lazily inside the
  functions that use them, never at import time. Conda users lose nothing —
  `--no-deps` still does what it did.

## [0.5.0] — 2026-07-21

### Fixed
- **The adaptel seed buffer discarded seeds in silence.** It was capped at
  `max(100000, 16*sqrt(size))`, and the `sqrt` term never wins for a real
  raster, so the cap was a flat 100,000. A 260x260 noise scene at threshold 30
  already exceeds it and lost 702 seeds; a 1200x1200 scene lost 9,278; a
  400x400 scene at threshold 20 sat at 91% of it. Pixels were not lost — the
  outer scan picks the region up later as a fresh start — but the segmentation
  changed, and nothing said so. Measured on the 1200x1200 case, **27% of the
  partition** differed from the uncapped result, so more than a quarter of the
  output depended on an arbitrary constant rather than on the data. The buffer
  now doubles on demand, and the result is bit-identical to an uncapped run on
  every scene tested.

  Comparing adaptel ids directly reports 99% here, which is renumbering rather
  than disagreement — the honest figure comes from best-overlap matching.

### Changed
- **The IFT heap in `_create_adaptels` also grows on demand.** This is
  precaution, not a fix: the heap holds the boundary of the *current* adaptel
  only, and the measured peak was 128 slots against the old cap of 100,000,
  with the overflow branch never firing. But the old code wrote `distances`
  and `labels` before checking capacity, which is exactly the shape of the
  bug that did fire in SICLE, so the check is gone rather than merely unlikely
  to matter.

### Added
- Tests for the seed buffer, on a scene that exceeds the old cap. Verified to
  discriminate: the capped code produces 29,883 adaptels there and the growing
  one 29,975.

## [0.4.0] — 2026-07-21

### Changed
- **SICLE seed relevance now uses the 8-neighbourhood the IFT grows over, not
  4.** Belém et al. 2023 define tree adjacency as
  `A(Ts) = {Tt : ∃⟨x,y⟩ ∈ A, x ∈ Ts, y ∈ Tt}` over the same arc set the forest
  uses; this scanned four neighbours while `_ift_fmax` expands to eight. The
  defect that hid there: a tree touching its only neighbour diagonally was
  found to have *no* neighbours, so its minimum contrast stayed at the
  sentinel, collapsed to 0, and the seed was ranked least relevant and removed
  first on no evidence at all.

  **This is a fidelity fix, not a quality improvement, and the measurements say
  so.** On `SNP_21_2020_1.tif` the change is essentially neutral: mean
  within-superpixel variance 156.25 → 157.27 at `n_segments=200` and
  57.74 → 57.72 at 800; largest superpixel 13,880 → 14,300 and 3,434 → 3,385;
  size CV 2.63 → 2.56 and 1.46 → 1.44. It moves 10.6% of the partition at 200
  and 6.6% at 800, measured label-invariantly by best-overlap matching —
  comparing superpixel ids directly reports 98%, which is renumbering rather
  than disagreement, and is the reason this is a minor version bump.

  Re-run anything whose superpixel boundaries you depend on.

### Added
- Tests for the relevance neighbourhood, built on two trees that touch only at
  a corner. Verified to discriminate: on the 4-neighbour code both score
  relevance 0, on the 8-neighbour code both score `vsize × contrast` as the
  paper specifies.

### Fixed
- **The `n_iterations` documentation was wrong, and the code was not.** It
  claimed "2 is optimal per Belém 2023". Having now read the paper, Ω=2 is
  their *speed* setting, and the argument is specific to the differential IFT
  they optimise, where removing many seeds at once creates inconsistencies to
  repair. This implementation re-runs the full IFT each iteration, so that
  argument does not transfer, and the measured delineation at Ω=2 is the worst
  of the values tried. The docstring now says so.

### Verified against the source paper
Belém et al. 2023 (doi:10.1007/s10851-023-01156-9) was checked line by line
against this implementation. Four of the five components are faithful: the
`fmax` path-cost, the `wroot` arc-cost `‖F(R(x)) − F(y)‖₂`, the relevance
criterion `vminsc(s) = vsize(s)·∇⁻F(Ts)`, and the object penalisation
`pobj(s) = max{O(Ts), ∇⁺O(Ts)}` with `pnone = 1` when no saliency map is given.

The seed-preservation curve is also faithful, which retracts a defect reported
here earlier. The paper's curve really is `M(i) = max{N₀^(1−i/(Ω−1)), Nf}`,
with `i` starting at 1 — which is what `t = (iteration + 1)/(Ω − 1)` computes.
The exponent genuinely does not depend on `n_segments`; that is the authors'
design, not a porting error, and the paper states the resulting iteration count
as `⌈(Ω−1)(1 − log_N₀ Nf)⌉ + 1`, which this implementation reproduces exactly
for Ω = 2, 5 and 10.

## [0.3.0] — 2026-07-21

A bug-fix release, but the version moves to 0.3.0 rather than 0.2.2 because
several fixes change results: `cosine` returned a similarity where a distance
was expected, `normalize=True` at the default threshold now raises instead of
returning one adaptel, and a NaN saliency map now raises instead of returning a
plausible-looking segmentation. Re-run anything whose numbers you rely on.

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
