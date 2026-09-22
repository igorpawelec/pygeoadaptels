"""The README figures, generated from test_data so they can be remade.

    python www/figures.py            # from the repository root; no install needed

Writes three figures and prints the numbers the README captions quote:
  www/pipeline.png    the orthophoto with hand-placed dead-tree points, the
                      adaptels at the default threshold, and the crowns
                      grow_seeds grows from the points;
  www/threshold.png   a 30 x 30 m window at threshold 30, 60 and 120 -- the
                      scene decides the count, the threshold decides the scale;
  www/max_cost.png    grow_seeds at max_cost 8, 15 and 25 -- the tolerance
                      that says how far from the seed's colour a crown may go.
One scene, test_data/SNP_21_2020_1.tif: a 100 x 100 m circular sample plot in
a spruce stand at 0.25 m, with 36 standing dead trees digitised as points.
The CIELAB raster beside it was made with pygeopalette (rgb_to_lab). The
orthophoto is contrast-stretched for display only; every algorithm sees the
raw bands.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pygeoadaptels import adaptels_from_array  # noqa: E402
from pygeoadaptels.grow import grow_seeds  # noqa: E402

RGB = os.path.join(ROOT, "test_data", "SNP_21_2020_1.tif")
LAB = os.path.join(ROOT, "test_data", "SNP_21_2020_1_lab.tif")
PTS = os.path.join(ROOT, "test_data", "dead_trees_test.shp")
ORANGE, INK = (0.922, 0.408, 0.204), "#1f1f1f"          # #eb6834 as floats, and the marker ink
RECIPE = dict(band_weights=[0.5, 2.5, 1.0], max_radius=20, fill_holes=True)   # docs/grow_seeds_guide.md
WINDOW = (120, 90, 120)                                  # row0, col0, size in px: a 30 x 30 m window with dead trees and shadow


def read_points(path, transform):
    import fiona
    with fiona.open(path) as src:
        xy = np.array([f["geometry"]["coordinates"] for f in src])
    inv = ~transform
    cols, rows = inv * (xy[:, 0], xy[:, 1])
    return np.column_stack([np.floor(rows), np.floor(cols)]).astype(np.int64)


def stretched(rgb, nodata, lo=1.0, hi=99.5):
    """Per-band percentile stretch of the valid pixels, for display only."""
    out = np.zeros(rgb.shape[1:] + (3,), np.uint8)
    for b in range(3):
        v = rgb[b][~nodata].astype(float)
        a, z = np.percentile(v, [lo, hi])
        out[..., b] = np.clip((rgb[b] - a) / max(z - a, 1) * 255, 0, 255).astype(np.uint8)
    out[nodata] = 255                                     # white outside the disc, like the CHM figures
    return out


with rasterio.open(RGB) as src:
    rgb = src.read()                                     # (3, rows, cols) uint8
    tf = src.transform
    res = src.res[0]
nodata = (rgb == 0).all(axis=0)                          # the plot is a disc; outside it every band is 0
data = rgb.astype(np.float64)
data[:, nodata] = np.nan
with rasterio.open(LAB) as src:
    lab = src.read().astype(np.float64)                  # L*, a*, b*, NaN outside the plot
seeds = read_points(PTS, tf)
img = stretched(rgb, nodata)


def panel(ax, label=None, window=None):
    r0, c0, n = window if window else (0, 0, img.shape[0])
    ax.imshow(img[r0:r0 + n, c0:c0 + n], interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#d9d9d9")
    if label:
        ax.set_xlabel(label, fontsize=9.5, color=INK, labelpad=7)


def overlay(ax, labels, fill=False, alpha=0.32, edge_alpha=1.0, window=None):
    """Region boundaries in orange; optionally the regions filled too."""
    r0, c0, n = window if window else (0, 0, labels.shape[0])
    lab_w = labels[r0:r0 + n, c0:c0 + n]
    valid = lab_w >= 0
    edge = np.zeros(lab_w.shape, bool)                   # one pixel per border, not one on each side
    edge[:, :-1] |= lab_w[:, :-1] != lab_w[:, 1:]
    edge[:-1, :] |= lab_w[:-1, :] != lab_w[1:, :]
    edge &= valid
    rgba = np.zeros(lab_w.shape + (4,))
    if fill:
        rgba[valid] = (*ORANGE, alpha)
    rgba[edge] = (*ORANGE, edge_alpha)
    ax.imshow(rgba, interpolation="nearest")


def draw_points(ax, size=22):
    ax.scatter(seeds[:, 1], seeds[:, 0], s=size, c=INK, edgecolors="white", linewidths=0.9, zorder=3)


# ---- the runs --------------------------------------------------------------------
adapt = {t: adaptels_from_array(data, threshold=t) for t in (30.0, 60.0, 120.0)}
grown = {c: grow_seeds(lab, seeds, max_cost=c, quiet=True, **RECIPE) for c in (8.0, 15.0, 25.0)}
n_px = int((~nodata).sum())
r0, c0, n = WINDOW
print(f"scene {os.path.basename(RGB)}: {rgb.shape[2]} x {rgb.shape[1]} px at {res:g} m, {n_px:,} px inside the plot, "
      f"{len(seeds)} dead-tree points; window rows {r0}-{r0 + n}, cols {c0}-{c0 + n} = {n * res:g} m")
for t, (lab_t, nt) in adapt.items():
    sizes = np.bincount(lab_t[lab_t >= 0])
    w = lab_t[r0:r0 + n, c0:c0 + n]
    print(f"threshold {t:>5g}: {nt:,} adaptels, median {np.median(sizes) * res * res:.1f} m2, "
          f"largest {sizes.max() * res * res:.0f} m2; {len(np.unique(w[w >= 0]))} in the window")
for c, g in grown.items():
    sizes = np.bincount(g[g >= 0], minlength=len(seeds))
    print(f"max_cost {c:>4g}: {int((sizes > 0).sum())} of {len(seeds)} points grew, median crown "
          f"{np.median(sizes[sizes > 0]) * res * res:.1f} m2, largest {sizes.max() * res * res:.0f} m2, "
          f"{int((g >= 0).sum()) / n_px:.1%} of the plot")

# ---- figure 1: the pipeline --------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
panel(axes[0], f"orthophoto, {res:g} m, {len(seeds)} dead-tree points")
draw_points(axes[0])
panel(axes[1], f"{adapt[60.0][1]:,} adaptels at threshold 60")
overlay(axes[1], adapt[60.0][0], edge_alpha=0.5)
panel(axes[2], "crowns grown from the points, max_cost 15")
overlay(axes[2], grown[15.0], fill=True)
draw_points(axes[2], size=12)
fig.subplots_adjust(wspace=0.05)
fig.savefig(os.path.join(HERE, "pipeline.png"), dpi=160, bbox_inches="tight", facecolor="white")

# ---- figure 2: the adaptel threshold, in a window ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
for ax, (t, (lab_t, nt)) in zip(axes, adapt.items()):
    w = lab_t[r0:r0 + n, c0:c0 + n]
    panel(ax, f"threshold = {t:g}\n{len(np.unique(w[w >= 0]))} adaptels in this 30 m window", window=WINDOW)
    overlay(ax, lab_t, window=WINDOW)
fig.subplots_adjust(wspace=0.05)
fig.savefig(os.path.join(HERE, "threshold.png"), dpi=160, bbox_inches="tight", facecolor="white")

# ---- figure 3: the grow_seeds tolerance ----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
for ax, (c, g) in zip(axes, grown.items()):
    covered = int((g >= 0).sum()) / n_px
    panel(ax, f"max_cost = {c:g}\n{covered:.1%} of the plot in crowns")
    overlay(ax, g, fill=True)
    draw_points(ax, size=12)
fig.subplots_adjust(wspace=0.05)
fig.savefig(os.path.join(HERE, "max_cost.png"), dpi=160, bbox_inches="tight", facecolor="white")
print("written: www/pipeline.png, www/threshold.png, www/max_cost.png")
