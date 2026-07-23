# `grow_seeds` — seeded spectral region growing ("inverse OBIA")

Specification for implementation in **plGeoAdaptels** (Python) and **rgeoadaptels** (R).

Status: design spec. Nothing here has been run. Every claim about existing code is a
source reading of plGeoAdaptels 0.8.1 / rgeoadaptels 0.2.0 and is marked with the file
and line it came from, so it can be checked rather than trusted.

---

## 1. Why this exists

The packages currently offer two segmenters, and both **partition the image**:

- `adaptels` — seeds are placed by the algorithm; the operator has no control over
  where they land. Every pixel ends up in some adaptel.
- `sicle` — seeds start as an oversampled set (N₀ ≫ Nf) and are then *removed* down to
  `n_segments`. Again every valid pixel ends up labelled.

That is the right behaviour for unsupervised segmentation. It is the wrong behaviour for
the task this spec addresses.

**The task.** An operator has an orthophoto (typically 0.25 m, RGB or CIR, sometimes a
true ortho, sometimes oblique) and a point layer where each point marks one object of
interest — in the motivating case, a standing dead tree. The operator wants each point
grown into a polygon that follows the actual crown of *that* tree, and wants everything
else — healthy crowns, shadow, ground, the neighbouring dead tree — to remain
**unassigned**.

This is the inverse of what `adaptels`/`sicle` do:

| | adaptels / sicle | `grow_seeds` |
|---|---|---|
| seed placement | algorithm decides | **operator decides** |
| seed count | tuned indirectly | **exactly the input points** |
| coverage | whole image partitioned | **only what grows; rest unassigned** |
| growth ends when | competition with other seeds | **competition *or* cost cap** |

In eCognition terms this is vector-based / seeded region growing rather than
multiresolution segmentation. The operator supplies the objects; the algorithm supplies
the boundaries.

**Deliberate non-goal.** This function does not decide what "spectrally different" means.
It consumes an N-band raster and treats the bands as a feature vector. Whether those
bands are raw RGB, CIELAB, CIECAM02, CIR, a vegetation index, a GLCM texture stack, or
any combination is the operator's decision, made upstream (e.g. with GeoPalette). See
§7 for why this matters more than it sounds.

---

## 2. What already exists (and why this is a small job)

The algorithm is already implemented, in both languages, in compiled code. It is the IFT
core that SICLE runs on.

**Python** — `plgeoadaptels/sicle.py`, `_ift_fmax`, decorated `@njit(cache=True)`
(sicle.py:43). Signature:

```
_ift_fmax(layers, n_layers, mask, cols, rows, seeds, n_seeds, labels_out, cost_out)
```

**R** — `rgeoadaptels/src/sicle.c`, `C_ift_fmax` (sicle.c:37), plain C via `.Call`,
registered in `src/init.c` as `"ift_fmax"` with 5 arguments. No Rcpp anywhere in the
package.

Both compute the same thing, documented in the Python docstring (sicle.py:46-49):

> For each seed s, grows an optimum-path tree T_s by minimizing
> `fmax(ρ) = max arc cost along the path`.
> Arc cost: `wroot(x,y) = ‖F(seed) − F(y)‖₂`

Read that carefully, because the whole design rests on it:

1. **`fmax` is a minimax path cost.** `cost[p]` is the *largest* arc cost encountered
   anywhere on the optimal path from the seed to `p` — not a sum. So `cost[p]` answers:
   "what is the worst thing I had to cross to get here?"
2. **`wroot` is measured against the seed, not the neighbour** (sicle.c:57-59 comments on
   this explicitly; the seed features are cached in `sf`). So the arc cost is the
   deviation from *the seed's own signature*, and it does not change as the tree grows.
   A gradient-based cost would let a region drift along a smooth ramp into shadow; this
   one cannot. For "grow the thing that looks like what I clicked on", this is the
   correct connectivity function and it is already the one implemented.
3. **Unassigned is already representable.** Both cores initialise `labels` to `-1`
   (sicle.py:346, sicle.c:53) and only overwrite a pixel when it is conquered. Masked
   pixels are never conquered (sicle.c:89, `if (mask[nidx] != 0) continue`).

**The cost array is computed and then thrown away.** In R, `C_ift_fmax` already returns a
two-element list — labels *and* cost (sicle.c:108-111) — and `sicle.R:162` takes
`res[[1]]` and drops the second element. In Python, `cost` is a local in the driver and
never reaches the return (`return labels_2d, int(n_final)`, sicle.py:414).

So the halting behaviour this spec asks for is one array away, in code that is already
written, compiled and tested.

---

## 3. Key design result: no changes to the compiled cores

The obvious implementation plan is "add a `max_cost` parameter and a compactness term to
the C and numba kernels". **Do not do that.** Every feature below can be obtained from
the *existing, unmodified* `_ift_fmax` / `C_ift_fmax` by preparing their inputs and
post-processing their outputs.

This matters for a reason specific to these packages: R/Python parity. If both languages
call the same already-verified kernel and differ only in a thin wrapper, the parity
surface is small and testable. Modifying two kernels in two languages in lockstep is how
`rHRG != pyHRG` happened. Keep the kernels frozen.

### 3.1 Cost cap → post-processing

```
labels[cost > max_cost] = -1
```

That is the entire feature. `cost` is already the minimax spectral deviation from the
seed, so thresholding it means: *keep the pixels reachable from the seed without ever
crossing a deviation larger than `max_cost`*. A healthy green crown, a shadow edge or a
differently-weathered neighbouring snag all produce a jump in `wroot`, the path cost
steps above the cap, and growth stops there.

Because the cost is a *max* and not a sum, the cap has a clean interpretation: it is a
**tolerance**, in the units of the input bands. If the operator fed CIELAB, `max_cost` is
a ΔE tolerance. That is worth stating in the user documentation — it is the difference
between a tunable parameter and a magic number.

### 3.2 Per-band weights → pre-scaling

The arc cost is a plain Euclidean norm over bands (sicle.c:91-97). Multiplying band `l`
by `w_l` before the call gives

```
sqrt( Σ_l w_l² (F_l(seed) − F_l(y))² )
```

which is exactly a weighted Euclidean distance. So `band_weights` is implemented by
scaling the stack, not by touching the kernel. Useful in practice: on RGB→Lab, weighting
`a*` up and `L*` down makes the growth follow hue/chroma (dead grey-brown vs living
green) and ignore illumination, which is most of what separates a crown from its own
sunlit/shaded halves.

### 3.3 Spatial compactness → two extra bands

This is the non-obvious one, and it solves a real defect.

**The defect.** With `wroot`, two adjacent objects with near-identical signatures produce
near-identical costs everywhere between them. The boundary is then decided by whichever
seed's wavefront arrived first. Both codebases admit this in comments: sicle.py:388-391
("`_ift_fmax` awards a contested pixel to whichever seed reached it first, so the order
decides the partition") and sicle.c:29. For two dead trees of the same species standing
side by side — the exact case the operator cares about — the split is essentially
arbitrary and depends on seed ordering, not on the image.

**The fix.** Because `wroot` measures distance to *the seed*, appending two coordinate
bands scaled by λ

```
band_row[r, c] = λ · r
band_col[r, c] = λ · c
```

makes the norm evaluate to

```
sqrt( spectral² + λ²·((r − r_seed)² + (c − c_seed)²) )
    = sqrt( spectral² + λ²·d_euclid(seed, pixel)² )
```

i.e. SLIC-style compactness, anchored at the seed, obtained with zero kernel changes.
Two spectrally identical neighbours now split at the geometric midpoint, deterministically.

Notes:
- λ = 0 reproduces current pure-spectral behaviour exactly. Default should be 0 so the
  function is a strict superset of what the kernel does today.
- λ has units of *feature-units per pixel*. Document it that way, and give the operator
  a worked example (e.g. "with Lab input, λ = 0.5 means moving 20 px away costs the same
  as a ΔE of 10").
- The trick relies on `wroot` being seed-anchored. If the kernel is ever changed to a
  neighbour-difference arc cost, this silently becomes a gradient term instead. Leave a
  comment in the wrapper saying so.
- Memory cost is two extra float64 bands over the raster. Trivial at 400×400,
  non-trivial on a full ortho tile — see §8.
- Under `fmax` the spatial term acts as a soft radius limit rather than a smooth
  compactness penalty, because the max along a path is normally attained at its far end.
  This is the intended behaviour here, but it is *not* identical to SLIC and the
  docstring should not claim it is.

### 3.4 Robust seed signature → patch the seed pixel

The entire segment is anchored on the band values of **one pixel** (sicle.c:57-64, the
`sf` cache). At 0.25 m with hand-placed points, one click landing on a sunlit highlight
or a shadow edge mis-anchors the whole object.

Mitigation without kernel changes: compute the k×k median around each seed and write it
into the seed pixel of a *copy* of the stack before calling the kernel. The kernel then
caches that value as the seed signature.

Caveat to document: this also alters the pixel as a growth *target*, not just as a
source. With k = 3 on a 0.25 m ortho the effect is negligible; state the caveat rather
than hiding it. Default `seed_window = 1` (no change), so the behaviour is opt-in.

### 3.5 Hard radius limit → post-processing

`max_radius` (in metres, converted to pixels via the transform) applied after the call:
any pixel further than that from its seed goes back to `-1`. Independent of λ, and
interpretable on its own. Cheap and often the most intuitive control for an operator who
knows roughly how big a crown is.

---

## 4. Public API

Names are a proposal; keep them consistent across the two packages.

### 4.1 Python

```python
def grow_seeds(
    data,                 # ndarray (bands, rows, cols) or (rows, cols)
    seeds,                # ndarray (n, 2) of (row, col), 0-based
    mask=None,            # (rows, cols) 0 = valid, nonzero = nodata; NaN-derived if None
    max_cost=None,        # cost cap in band units; None = no cap (partition, as today)
    band_weights=None,    # (bands,) multipliers applied before the distance
    compactness=0.0,      # lambda; 0 = pure spectral
    seed_window=1,        # k for the k*k median seed signature; 1 = raw pixel
    max_radius=None,      # in pixels; None = unlimited
    return_cost=False,
    quiet=False,
):
    """
    Returns
    -------
    labels : (rows, cols) int32   -- label i corresponds to seeds[i]; -1 = unassigned
    cost   : (rows, cols) float64 -- only if return_cost
    """
```

And a file-level convenience wrapper mirroring `create_sicle`:

```python
def grow_seeds_from_files(
    input_files, points, output_file=None, points_layer=None, **kwargs
)
```

where `points` is a path to a vector file (or an array of map coordinates), reprojected
to the raster CRS if needed, and converted to pixel indices per §5.

### 4.2 R

```r
grow_seeds(data, seeds, mask = NULL, max_cost = NULL, band_weights = NULL,
           compactness = 0, seed_window = 1L, max_radius = NULL,
           return_cost = FALSE, quiet = FALSE)
# -> list(labels = <matrix int>, n_segments = <int>, cost = <matrix double> | NULL)

grow_seeds_raster(input_files, points, output_file = NULL, ...)
```

Follow the conventions already established in the packages:

- mask: **non-zero = nodata**, as in `read_bands()`, `adaptels()`, `sicle()`.
- data layout: 3-D `(bands, rows, cols)` in Python; the same flattening R already uses in
  `sicle.R` (band-major, row-major within band — see the indexing at sicle.c:93).
- return a list in R with the labels as a `matrix(..., byrow = TRUE)`, matching
  `sicle()`'s return.

### 4.3 Label contract

**`labels == i` is the segment grown from `seeds[i]`**, in input order, and `-1` means
unassigned. This must hold unconditionally, so that the operator can join the result back
to the attributes of their point layer.

This is why `grow_seeds` must **not** reuse SICLE's driver loop: that loop removes seeds
and reorders the survivors (`seeds = seeds[order[:m_keep]]`, sicle.py:398; the R
equivalent at sicle.R:~178), which destroys the mapping. `grow_seeds` calls the kernel
**once**, with all seeds, and never removes any.

If a seed is dropped for a legitimate reason (masked pixel, duplicate, outside the
raster), do not silently renumber. Either raise, or keep the index and return an empty
segment — decide once, document it, and make both languages do the same thing.

---

## 5. Seed contract: point → pixel

This is the highest-risk part of the whole job, because it is where R and Python will
disagree silently if it is not pinned down. `rHRG != pyHRG` on real CHMs came from
exactly this class of underspecification.

Requirements:

1. **Convention.** Pixel `(r, c)` covers the half-open extent
   `[x0 + c·w, x0 + (c+1)·w) × (y0 − r·h, y0 − (r+1)·h]`. Compute
   `c = floor((x − x0) / w)`, `r = floor((y0 − y) / h)`. A point exactly on a shared edge
   therefore belongs to the pixel to the right / below. This matches "snap to the nearest
   pixel centre" for every point that is not exactly on an edge, and resolves the edge
   case deterministically instead of leaving it to floating-point noise.
2. **Do not delegate to `terra::cellFromXY` and `rasterio.transform.rowcol` and assume
   they agree.** Implement the arithmetic explicitly in both packages, from the same
   formula, and cross-test it (§9). They may well agree; the point is that the
   specification, not the dependency, is the contract.
3. **CRS.** If the points carry a CRS and it differs from the raster's, reproject the
   points, and say so in a message unless `quiet`. Never assume they match.
4. **Validation.** Seeds must be unique, inside the raster, and on unmasked pixels. Two
   points landing in the same pixel is a real possibility at 0.25 m — decide whether that
   is an error or a silent merge, and make both languages agree. Recommendation: raise,
   with the offending indices in the message, because a silent merge breaks the label
   contract in §4.3.
5. **Oblique imagery.** Worth a paragraph in the user documentation, not in the code: on
   an oblique ortho a crown leans away from nadir, so a point digitised on the visible
   apex may fall outside the crown's footprint, or onto the neighbour. The function
   cannot detect this. Recommend that the operator digitises on the object as it appears
   in *this* image, and validate visually before trusting a batch run.

---

## 6. Reference implementation sketch (Python)

Illustrative, not final. R follows the same steps.

```python
n_bands, rows, cols = _as_3d(data)                 # reuse sicle's existing prep
mask = _mask_from(data, mask)                      # 0 = valid, as everywhere else

stack = layers.astype(np.float64, copy=True)

if band_weights is not None:                       # 3.2
    stack *= np.asarray(band_weights)[:, None, None]

if seed_window > 1:                                # 3.4
    for (r, c) in seeds:
        stack[:, r, c] = _window_median(layers, r, c, seed_window)

if compactness > 0:                                # 3.3
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float64)
    stack = np.concatenate([stack,
                            (compactness * rr)[None],
                            (compactness * cc)[None]], axis=0)

labels = np.empty(rows * cols, np.int32)
cost   = np.empty(rows * cols, np.float64)
_ift_fmax(stack.reshape(len(stack), -1), len(stack),
          mask.ravel(), cols, rows,
          _flat_indices(seeds, cols), len(seeds),
          labels, cost)                            # kernel UNCHANGED

labels = labels.reshape(rows, cols)
cost   = cost.reshape(rows, cols)

if max_cost is not None:                           # 3.1
    labels[cost > max_cost] = -1
if max_radius is not None:                         # 3.5
    labels[_dist_to_own_seed(labels, seeds) > max_radius] = -1
```

Note the ordering: weights before the coordinate bands, so λ is not accidentally scaled
by a band weight.

---

## 7. On texture (GLCM) — read before implementing it

The motivating worry is: two adjacent objects that are spectrally almost identical but
are different crowns. The instinct is to add GLCM texture features. Two observations
before spending time on that:

1. **Multi-seed competition already separates seeded neighbours.** IFT assigns each pixel
   to the seed that reaches it with the lower `fmax` cost. If both trees carry a point,
   they are separated by construction, along the cost-balance line. Texture is not needed
   for that case. What texture was implicitly being asked to fix is the *tie fragility*
   described in §3.3 — and the compactness term fixes it more cheaply and more
   predictably.
2. **Texture is a windowed statistic, so it is blurred exactly where sharpness is
   needed.** A GLCM feature at pixel `p` summarises a neighbourhood of `p`; near a crown
   boundary that neighbourhood straddles both objects. At 0.25 m a dead crown is roughly
   20–40 px across, so a 7×7 window is a substantial fraction of the object. As an *edge*
   discriminator, GLCM is weak; as a *region* descriptor it is fine, which is a different
   job.

None of that forbids texture. Because the function takes an arbitrary band stack (§1),
an operator who wants GLCM bands can compute them upstream and stack them, with a low
`band_weight` so they nudge rather than dominate. **That is the whole point of the
band-stack design: the algorithm stays out of the feature-engineering argument.** Do not
build GLCM into this function.

For the two cases actually named in the motivation, cheaper tools exist:

- **Shadow** — do not model it as a stopping attribute; mask it. Shadow is dark and
  low-chroma, so a threshold on `L*` (or a shadow index) produces a mask, and masked
  pixels are never conquered (sicle.c:89). Growth then stops at the shadow edge by
  construction, sharply, at no cost.
- **Healthy vs dead** — this is a band-choice problem, not an algorithm problem. On RGB,
  `a*` in CIELAB or an excess-green index separates living green from grey-brown snags
  strongly. On CIR, use the NIR band. Feed it in and weight it up.

---

## 8. Performance

- One `_ift_fmax` call for the whole point set, not one per point. The kernel is
  seed-restricted and handles all seeds in a single wavefront; per-point calls would be
  O(n) times slower and would break the competition that separates neighbours.
- With `max_cost` set, the wavefront still explores the whole valid image before the cap
  is applied in post-processing. If profiling shows this dominates on large tiles, the
  optimisation is to stop the heap loop once the extracted cost exceeds `max_cost`, since
  `fmax` costs are non-decreasing along a path and the heap is ordered — every pixel
  behind that point is over the cap too. **That is a kernel change**, so it is out of
  scope for the first version; note it in the code as a known optimisation and measure
  before deciding.
- The coordinate bands (§3.3) add `2 · rows · cols · 8` bytes. On a large tile, consider
  computing them as `float32`, or gating them behind `compactness > 0` (which the sketch
  already does).

---

## 9. Testing

Follow the project's method: measure on real data in `test_data/`, do not assert from
theory. State in the test names what is being tested.

Unit-level, synthetic (no data files needed):

1. Two uniform blocks, one seed in each → two segments, boundary on the block edge,
   nothing unassigned when `max_cost=None`.
2. Same, with `max_cost` below the block-to-block distance → each segment confined to its
   own block; the far block is `-1` if only one seed is given.
3. `compactness > 0` on a **uniform** image with two seeds → boundary at the geometric
   midpoint (this is the §3.3 regression test; without the fix the boundary is
   order-dependent).
4. Label contract: shuffle the seed order, check `labels == i` still tracks `seeds[i]`.
5. `max_cost=None, compactness=0` reproduces a single `_ift_fmax` call exactly — i.e. the
   function is a strict superset of current kernel behaviour.

On real data (`SNP_21_2020_1.tif`):

6. Hand-place a handful of seeds, run, and report: number of pixels assigned, per-segment
   area, and how many pixels stayed `-1`. Record the numbers; that is the baseline for
   future changes.
7. Sweep `max_cost` over a range and plot segment area against it. There should be a
   plateau — a range where area is insensitive to the cap because growth is bounded by a
   real edge rather than by the cap. Operating in the plateau is the calibration advice
   to give users. If there is no plateau, the band choice is wrong, and that is worth
   knowing.

Cross-language parity (this is the one that matters most here):

8. Same raster, same seeds, same parameters → **identical label matrices** in R and
   Python, tested bit-for-bit, not approximately. The kernels are separate
   implementations of the same algorithm, so this is a real test, not a tautology. The
   package already has a cross-check harness in `tools/` (referenced at the tail of
   `sicle.R`) — extend it rather than inventing a second mechanism.
9. Point→pixel conversion tested separately from growth: a table of map coordinates,
   including points exactly on pixel edges and on the raster's outer boundary, with the
   expected `(row, col)` per §5, run through both languages.

---

## 10. Documentation the operator actually needs

Two things decide whether this is usable, and neither is code:

- **What `max_cost` means in the units of what they fed in.** Write it as: "if you passed
  CIELAB, `max_cost` is a ΔE tolerance; a value of 10 means the segment will not cross
  anything more than ΔE 10 away from the pixel you clicked." Give one worked example per
  supported input (Lab, CAM02, raw RGB, CIR).
- **The calibration recipe**, i.e. test 7 above turned into user-facing advice: run a
  sweep, look for the plateau, work in it.

Add a short "when not to use this" section: unseeded objects are absorbed into whichever
neighbour reaches them first (up to the cap), so this is a tool for delineating objects
you have already inventoried, not for finding new ones. `adaptels`/`sicle` remain the
tools for discovery.

---

## 11. Summary of the work

| Item | Kernel change? | Where |
|---|---|---|
| Return/expose `cost` | no — R already returns it, Python discards a local | `sicle.R:162`, `sicle.py:414` |
| `max_cost` cap | no | wrapper post-processing |
| `band_weights` | no | pre-scale the stack |
| `compactness` (λ) | no | two appended coordinate bands |
| `seed_window` median | no | patch seed pixels in a copy |
| `max_radius` | no | wrapper post-processing |
| point → pixel contract | no | new, both packages, spec §5 |
| early heap termination | **yes** | deferred; measure first (§8) |

The compiled cores stay frozen. The new code is a wrapper in each language plus a shared,
explicitly specified seed contract — which is also the only place the two languages can
drift.
