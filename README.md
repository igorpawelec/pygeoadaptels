# pygeoadaptels

<img src="https://raw.githubusercontent.com/igorpawelec/pygeoadaptels/main/www/pygeoadaptels.png" align="right" width="200"/>

[![tests](https://github.com/igorpawelec/pygeoadaptels/actions/workflows/tests.yml/badge.svg)](https://github.com/igorpawelec/pygeoadaptels/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/igorpawelec/pygeoadaptels)](https://github.com/igorpawelec/pygeoadaptels/releases)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Scale-Adaptive Superpixels (Adaptels) and SICLE superpixels for geospatial raster data — and `grow_seeds`, the same kernel run the other way round.**

A raster goes in; a segmentation comes out, as a label raster or as polygons. Pure Python + Numba, portable, pip-installable, no compiled binaries.

> **R users:** the same algorithms are in [rgeoadaptels](https://github.com/igorpawelec/rgeoadaptels). The two are separate repositories because their tooling and idioms do not mix, but they are **bit-identical** — checked, not asserted, across thirty cases.

## The problem it solves

Object-based analysis of an orthophoto starts by cutting the image into segments, and the usual superpixel methods ask you how many. A fixed count imposes a grid on a scene that has no grid: the same segment size that resolves a small crown is wasted on a stretch of shadow, and the count that suits one plot is wrong on the next.

**Adaptels** let the scene decide. A region grows from a seed until its internal colour distance passes a threshold *T*; the pixels beyond it become seeds in turn. Where the image is textured the regions stay small, where it is homogeneous they grow large, and the count follows from the data. One parameter, and it is a distance in the units of your bands.

**`grow_seeds`** is the inverse. When you already know where the objects are — a point layer of standing dead trees digitised by an operator — you do not want a partition of the whole image, you want the boundary of each object. Every point grows into the region that looks like the pixel it sits on, within a tolerance, and everything unseeded stays unassigned. The operator supplies the objects, the algorithm supplies their extent.

Both run on one frozen kernel — an image foresting transform with a min-heap — which is also what **SICLE** uses when you do want a fixed number of superpixels.

<img src="https://raw.githubusercontent.com/igorpawelec/pygeoadaptels/main/www/pipeline.png" alt="A 0.25 m orthophoto of a circular spruce plot with 36 dead-tree points; the same plot partitioned into 2,792 adaptels at threshold 60; and the crowns grown from the 36 points by grow_seeds" width="100%"/>

*A 100 × 100 m circular sample plot in a spruce stand at 0.25 m (`test_data/SNP_21_2020_1.tif`), with 36 standing dead trees digitised as points. Middle: the plot partitioned into 2,792 adaptels at the default threshold of 60 — median 1.8 m², largest 50 m². Right: the crowns `grow_seeds` grows from the 36 points on the CIELAB version of the same scene, with the dead-tree recipe (`max_cost` 15, `band_weights` 0.5/2.5/1, `max_radius` 20, `fill_holes`). The orthophoto is contrast-stretched for display only. Made by `www/figures.py`.*

## The one parameter that matters

`threshold` is the colour distance a region may accumulate before it stops growing, in the units of the input bands. It is not a size and not a count: it decides the *scale* at which the scene is cut, and the scene then decides how many pieces that takes.

<img src="https://raw.githubusercontent.com/igorpawelec/pygeoadaptels/main/www/threshold.png" alt="A 30 by 30 m window of the plot at threshold 30, 60 and 120: 825, 403 and 178 adaptels, small on the textured crowns and large in the shadow" width="100%"/>

*A 30 × 30 m window of the plot at threshold 30, 60 and 120 — 825, 403 and 178 adaptels in the window, 6,138, 2,792 and 1,159 on the whole plot. At every threshold the adaptels are small on the textured crowns and large in the homogeneous shadow between them; the threshold shifts that whole distribution rather than fixing a size.*

The metrics do not share a scale, so a threshold does not carry across them. The default of 60 is scaled for `minkowski`; passing it to `cosine` or `angular` would merge the whole raster into one adaptel, so it is rejected with an error rather than silently returning nonsense.

| `distance` | Range | Typical `threshold` | Notes |
|---|---|---|---|
| `minkowski` | 0 … data range | 10 – 120 | Grows with band count and bit depth. For 8-bit 3-band imagery distances reach ~441 |
| `cosine` | 0 … 1 | 0.002 – 0.05 | Spectral angle as `1 - cos`. Insensitive to brightness |
| `angular` | 0 … 1 | 0.005 – 0.2 | `arccos(cos) / π`. Also brightness-insensitive, more linear than `cosine` |

`cosine` and `angular` compare the *direction* of the spectral vector rather than its length, so the same material lit differently — a crown in sun versus in shade — lands in one adaptel. `minkowski` will split it. The default of 60 is inherited from the original method, which works in CIELAB, where a distance of 1 is roughly one just-noticeable difference; raster bands are not CIELAB, so treat 60 as a starting point to calibrate from, not a value with meaning on your data.

## The other way round: `grow_seeds`

`grow_seeds` takes the points and one tolerance, `max_cost`: how far from the seed pixel's colour a crown may reach, in band units. Feed it CIELAB — convert with [pygeopalette](https://github.com/igorpawelec/pygeopalette) first — and `max_cost` becomes a ΔE, a colour difference with a meaning.

<img src="https://raw.githubusercontent.com/igorpawelec/pygeoadaptels/main/www/max_cost.png" alt="The 36 dead-tree points grown at max_cost 8, 15 and 25: crowns covering 1.9, 4.9 and 5.5 percent of the plot; at 25 one crown floods to a disc that max_radius stops" width="100%"/>

*The same 36 points grown at `max_cost` 8, 15 and 25: crowns covering 1.9 %, 4.9 % and 5.5 % of the plot, median 2.5, 6.4 and 6.5 m². At 8 the crowns stop short of their own edges; at 15, the recipe, they fill the bleached crowns and stop at the living neighbours; at 25 one seed on a dark stem leaks into the surrounding canopy until `max_radius` (20 px, 5 m) stops it — the disc in the lower left is that cap doing its job. Calibrate `max_cost` by sweeping it and looking, exactly like this.*

`band_weights` reshapes the feature space — weighting `a*` up separates dead brown from living green, which is the whole dead-vs-living problem — `max_radius` bounds the reach, and `fill_holes` closes the pockets a cut leaves inside a crown. Label `i` is the region grown from the i-th point, so it joins back to that point's attributes. [`docs/grow_seeds_guide.md`](docs/grow_seeds_guide.md) is the operator's guide, with the worked recipe for dead trees.

## When to use it, and when not

Use adaptels when the segments are the units of a later analysis — zonal statistics, a classifier over segment features, a manual interpretation — and you want them to follow the scene rather than a grid. Use `grow_seeds` when the objects are already located and you need their boundaries. Use SICLE when a downstream method needs a fixed number of superpixels, or a saliency map (a normalised height model, say) should pull the boundaries towards objects.

Adaptels are not object detection: a segment is a homogeneous patch, not a tree, and a crown may be several of them. For crowns from a canopy height model, use [pycacumen](https://github.com/igorpawelec/pycacumen); for standing dead trees as points and crowns from the orthophoto alone, [pygeosnag](https://github.com/igorpawelec/pygeosnag), which runs on adaptels underneath.

The algorithms come from [Achanta et al. (2018)](#citation) for adaptels and [Belém et al. (2023)](#citation) for SICLE; the original C implementation of adaptels, `plGeoAdaptels`, was written by Paweł Netzel at the University of Agriculture in Kraków, and this package is a faithful reimplementation of it. The method was applied to standing dead tree delineation in [Pawelec et al. (2026)](#citation).

### The package family

pygeoadaptels is one step of a longer chain; the other steps are separate packages, each with a Python and an R twin.

| Step | Python | R |
|---|---|---|
| Colour-space conversion of orthophotos | [pygeopalette](https://github.com/igorpawelec/pygeopalette) | [rgeopalette](https://github.com/igorpawelec/rgeopalette) |
| Adaptive superpixels and seeded growing on orthophotos | **pygeoadaptels** | [rgeoadaptels](https://github.com/igorpawelec/rgeoadaptels) |
| Crowns from a canopy height model | [pycacumen](https://github.com/igorpawelec/pycacumen) | [rcacumen](https://github.com/igorpawelec/rcacumen) |
| Standing dead trees on orthophotos | [pygeosnag](https://github.com/igorpawelec/pygeosnag) | — |
| The same, inside QGIS | [qgis-geoadaptels-geopalette](https://github.com/igorpawelec/qgis-geoadaptels-geopalette), [qgis-geosnag](https://github.com/igorpawelec/qgis-geosnag) | |
| Polish national geodata (GUGiK, BDL) | — | [rgeopl](https://github.com/igorpawelec/rgeopl) |

## Installation

Native dependencies come from conda; pip then installs the package without touching them.

```bash
conda install -c conda-forge numpy numba rasterio fiona
pip install --no-deps git+https://github.com/igorpawelec/pygeoadaptels.git
```

The `--no-deps` flag keeps pip from overwriting conda's GDAL/PROJ stack.

## Quick start

```python
from pygeoadaptels import create_adaptels, adaptels_from_array

# From GeoTIFF: reads, segments, writes the label raster
labels, n = create_adaptels("input.tif", "adaptels.tif", threshold=60.0)

# Several single-band files as one multi-band input
labels, n = create_adaptels(["b1.tif", "b2.tif", "b3.tif"], "adaptels.tif", threshold=40.0, normalize=True)

# From numpy arrays, no file I/O: (bands, rows, cols) or (rows, cols)
labels, n = adaptels_from_array(data, threshold=30.0)
```

```python
from pygeoadaptels.grow import grow_seeds_from_files

grow_seeds_from_files(
    "ortho_lab.tif", "dead_trees.shp",
    output_file="labels.tif", polygons="crowns.gpkg",
    max_cost=15, band_weights=[0.5, 2.5, 1.0],   # ΔE tolerance; dead vs living
    max_radius=20, fill_holes=True,
)
```

```python
from pygeoadaptels.sicle import sicle_from_array, create_sicle

labels, n = sicle_from_array(data, n_segments=200)
labels, n = sicle_from_array(data, n_segments=200, saliency=chm_normalized)   # favours object boundaries
labels, n = create_sicle("input.tif", "sicle.tif", n_segments=200)
```

```python
from pygeoadaptels.vectorize import vectorize_from_file

n_poly = vectorize_from_file("adaptels.tif", "adaptels.gpkg")   # or .shp, .geojson; no geopandas needed
```

From the command line:

```bash
pygeoadaptels -i input.tif -o output.tif -t 60.0
pygeoadaptels -i b1.tif -i b2.tif -o result.tif -t 0.03 -8 -d angular -n
python -m pygeoadaptels --help
```

## Reference

### Adaptels

| Parameter | Default | Description |
|---|---|---|
| `threshold` | 60.0 | Energy threshold *T*. Lower → smaller adaptels, higher → larger. **Its scale depends on `distance` — see above** |
| `distance` | `'minkowski'` | Distance metric: `'minkowski'`, `'cosine'`, `'angular'` |
| `minkowski_p` | 2.0 | Minkowski *p* parameter (2.0 = Euclidean) |
| `queen_topology` | `False` | `True` = 8-connectivity, `False` = 4-connectivity |
| `normalize` | `False` | Normalize inputs to [0, 1] before processing |

Measured on `test_data/`, 200 × 200 px:

| `threshold` | `minkowski` | | `threshold` | `angular` |
|---|---|---|---|---|
| 10 | 4140 | | 0.002 | 7834 |
| 30 | 1512 | | 0.01 | 2654 |
| 60 | 700 | | 0.03 | 855 |
| 120 | 299 | | 0.2 | 99 |

**Contiguity.** Adaptels compete for pixels — a later adaptel takes a pixel from an earlier one whenever it reaches it with a smaller accumulated distance. That competition is what gives the method its boundary adherence, but it can also cut an earlier adaptel in two, leaving one label spread over separate patches. On the 400 × 400 px test raster at threshold 60, 265 of 2770 adaptels (9.6 %) come out in more than one piece, the worst in four. Harmless if the labels are only a lookup; not harmless for zonal statistics, which would average two separate patches into one "object":

```python
from pygeoadaptels import adaptels_from_array, enforce_connectivity

labels, n = adaptels_from_array(data, threshold=60.0)   # 2770, 265 split
labels, n = enforce_connectivity(labels)                # 3066, 0 split
```

Every connected component becomes its own adaptel, so the count rises by about 10 %. Nothing is merged and no pixel changes hands. `min_size=` absorbs fragments below a size into an adjacent adaptel instead. Needs `scipy`. Not applied automatically, because it changes the count.

### `grow_seeds`

| Parameter | Default | Description |
|---|---|---|
| `max_cost` | `None` | Tolerance in band units (a ΔE on CIELAB). `None` keeps every reachable pixel — a partition |
| `band_weights` | `None` | Per-band weights reshaping the feature space |
| `max_radius` | `None` | Pixels from the seed a region may reach |
| `fill_holes` | `False` | Close the pockets a cut leaves inside a region |
| `seed_window` | 1 | Seed colour as the median of a window around the point, instead of the pixel |
| `compactness` | 0.0 | Spatial term added to the colour cost |

`seeds` are `(row, col)` pixel pairs; `grow_seeds_from_files` takes a point layer in any CRS and does the conversion. Labels are 0-based, `-1` unassigned, and label `i` is the region grown from the i-th point — a seed that cannot be placed raises rather than being dropped, so the point order never silently shifts.

### SICLE

| Parameter | Default | Description |
|---|---|---|
| `n_segments` | 200 | Desired number of superpixels |
| `n_oversampling` | 3000 | Initial seed count (N₀ ≫ n_segments) |
| `n_iterations` | 2 | Max IFT iterations (Ω). 2 is Belém 2023's *speed* setting, not a quality one; its rationale is specific to the differential IFT this does not use, so raising it can improve delineation |
| `saliency` | `None` | Object saliency map (H, W) float64 in [0, 1], e.g. a normalised CHM |
| `seeds` | `None` | Seeds given instead of sampled |

<details>
<summary><b>Performance, repository layout, requirements, testing</b></summary>

First call includes Numba JIT compilation (~5 s). Subsequent calls run at near-C speed: a 400 × 400 px 3-band raster in 0.03 s (~2 800 adaptels), 2000 × 2000 px in about 1 s (~70 000).

```
pygeoadaptels/
├── pygeoadaptels/        # Package source
│   ├── __init__.py       # Public API (lazy imports)
│   ├── __main__.py       # CLI entry point
│   ├── adaptels.py       # High-level API (threshold-based)
│   ├── sicle.py          # SICLE superpixels (n_segments-based)
│   ├── grow.py           # grow_seeds — seeded region growing
│   ├── cli.py            # Command-line interface
│   ├── core.py           # Numba JIT algorithm kernel + min-heap
│   ├── io.py             # GeoTIFF read/write (rasterio)
│   └── vectorize.py      # Raster→vector (rasterio + fiona)
├── tests/                # Pytest suite
├── test_data/            # The sample plot: orthophoto, its CIELAB version, the dead-tree points
├── docs/                 # grow_seeds operator's guide and specification
├── www/                  # Logo and the README figures, with the script that makes them
├── pyproject.toml
├── environment.yaml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

- Python ≥ 3.9, NumPy ≥ 1.21, Numba ≥ 0.56
- Rasterio ≥ 1.3, Fiona ≥ 1.9 *(file I/O and vectorization)*

```bash
pip install pytest
pytest tests/ -v
```

The README figures are remade with `python www/figures.py` from the files in `test_data/`.

</details>

## Citation

If you use this software in your research, please cite:

1. **This implementation:**

> Pawelec, I. (2026). pygeoadaptels — Scale-Adaptive Superpixels for geospatial data [Software]. https://github.com/igorpawelec/pygeoadaptels

2. **The original algorithm:**

> Achanta, R., Marquez-Neila, P., Fua, P., & Süsstrunk, S. (2018). Scale-Adaptive Superpixels. *Color and Imaging Conference (CIC26)*.

3. **The original C implementation:**

> Netzel, P. plGeoAdaptels [Software]. University of Agriculture in Kraków. https://gitlist.netzel.pl/

4. **SICLE algorithm:**

> Belém, F.C., Barcelos, I.B., João, L.M., Perret, B., Cousty, J., Guimarães, S.J.F., & Falcão, A.X. (2023). Novel Arc-Cost Functions and Seed Relevance Estimations for Compact and Accurate Superpixels. *Journal of Mathematical Imaging and Vision*, 65, 770–786.

5. **The application to standing dead trees:**

> Pawelec, I., Hawryło, P., Netzel, P., & Socha, J. (2026). Evaluating superpixel algorithms for standing dead tree delineation using aerial orthoimagery. *International Journal of Applied Earth Observation and Geoinformation*, 147, 105180. https://doi.org/10.1016/j.jag.2026.105180

See also [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
