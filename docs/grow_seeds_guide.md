# `grow_seeds` — operator's guide

`grow_seeds` grows an operator-placed point layer into region polygons and
leaves everything unseeded unassigned. It is the inverse of `adaptels` and
`sicle`, which partition the whole image: here **you** supply the objects and
the algorithm supplies their boundaries. It is a tool for delineating objects
you have already inventoried — standing dead trees, in the case it was built
for — not for finding new ones.

The numbers below were measured on a 400×400, 0.25 m CIELAB ortho
(`SNP_21_2020_1_lab.tif`) with 36 hand-digitised dead-tree points
(`dead_trees_test.shp`). Yours will differ; the shapes of the curves are the
point.

## The one thing to understand: `max_cost` is a tolerance in your band units

The growth cost is the **largest** spectral step on the path from the seed to
a pixel — the worst thing crossed to get there — measured against the seed's
own signature. `max_cost` caps it. So:

- If you feed **CIELAB**, `max_cost` is a **ΔE tolerance**: `max_cost = 10`
  means the segment will not cross anything more than ΔE 10 from the pixel you
  clicked.
- Feed CIELAB, not raw RGB. `max_cost` is only meaningful as a tolerance
  because CIELAB is perceptually near-uniform; a Euclidean step in RGB is not.
  Convert with GeoPalette first.

## Calibrate `max_cost` by sweeping it

Run a sweep and plot segment area against `max_cost`. You are looking for a
**plateau** — a range where area barely changes because growth is stopped by a
real edge, not by the cap. Operate in the plateau.

On the dead-tree scene there was **no plateau**: area rose smoothly and then,
past ΔE ≈ 25, one seed flooded the whole background in a single step. That is
worth knowing, and it is the diagnostic the sweep exists for — a continuous
ΔE distribution from the seeds to the background means the bands alone do not
draw a sharp line. Two things fix it, below.

## `max_radius` stops a flood

When there is no plateau, cap the reach with `max_radius`, in pixels, from
what you know a crown's size to be. On this scene `max_cost = 12` with
`max_radius = 12` px (3 m) held the largest crown to 18 m² and kept every tree
a separate object, where `max_cost` alone let one seed take thousands of
pixels. `max_radius` is often the most intuitive control: you usually know
roughly how big the object is.

## The dead-vs-living problem is a *band* problem, and `band_weights` solves it

Two symptoms show up together: homogeneous crowns grow too little, and crowns
next to a **healthy** tree spill onto it. Both are one thing. Dead crowns have
`a* ≈ 4` (brown); healthy canopy has `a* ≈ -1` (green) — a gap of only ~5,
while the brightness variation *inside* a dead crown is just as large. So no
single `max_cost` fills the crown without also crossing onto the neighbour.

That is a **feature** limit, not the algorithm, and the fix is to reshape the
feature space with `band_weights`. Down-weighting `L*` and up-weighting `a*`:

```
band_weights = [0.5, 2.5, 1.0]     # L down, a* up, b* unchanged
```

Measured effect at `max_cost = 15`, `max_radius = 20`: assigned pixels went
from 3544 to **6174** (crowns fill *more*) while the share that is green
canopy dropped from **19 % to 5 %** (they spill onto healthy trees *less*) —
both at once. Down-weighting `L*` lets a crown's own highlights and shadows
cost less; up-weighting `a*` makes the step onto green cost more.

This is by design: `grow_seeds` stays out of the feature-engineering argument.
If you want it to follow texture, a vegetation index, or NIR, compute those
bands upstream and stack them, with a `band_weight` that nudges rather than
dominates. If you want shadow ignored, do not model it — **mask** it (shadow
is dark and low-chroma, so a threshold on `L*` gives a mask, and masked pixels
are never grown into).

## Clean polygons

The file wrappers write polygons with 8-connectivity, because the growth is
8-connected — a crown that pinches on a diagonal is one region, not two. Turn
on `fill_holes` to close the pockets a `max_cost` cut leaves inside a crown (a
bright spot or a shadow gap): it fills only pockets that sit fully inside one
crown, so it never swallows nodata or bridges two crowns.

## A working recipe for dead trees

```
grow_seeds_from_files(
    "ortho_lab.tif", "dead_trees.shp",
    output_file="labels.tif", polygons="crowns.gpkg",
    max_cost=15, max_radius=20,          # spectral tolerance + size cap
    band_weights=[0.5, 2.5, 1.0],        # follow dead-vs-living, ignore light
    fill_holes=True,                     # solid crowns
)
```

Then **look at it** over the ortho before trusting a batch. This is a starting
point found on one scene; the values are yours to calibrate against ground
truth.

## Caveats

- **Two points in one pixel** is an error, not a silent merge — it would break
  the label-to-point join. Move one.
- **A point on an oblique ortho** may fall outside the crown's footprint,
  because the crown leans away from nadir. Digitise on the object as it appears
  in *this* image.
- **Unseeded objects are absorbed** by whichever seeded neighbour reaches them
  first (up to the cap). This is a tool for objects you have inventoried; use
  `adaptels` / `sicle` to discover new ones.
- The label raster uses **-1** for unassigned, set as its nodata, and label
  `i` is the region grown from the i-th point in the layer, so it joins back
  to that point's attributes.
