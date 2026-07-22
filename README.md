# plGeoAdaptels

<img src="https://raw.githubusercontent.com/igorpawelec/plGeoAdaptels/main/www/plGeoAdaptels.png" align="right" width="200"/>

[!\[License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Scale-Adaptive Superpixels (Adaptels) and SICLE superpixels for geospatial raster data.**

A pure Python + Numba package providing two complementary superpixel algorithms for geospatial raster data. Portable, pip-installable, no compiled binaries required.

## Background

The package implements two algorithms:

**Adaptels** — superpixels that **automatically adapt their size** to local image texture: small in complex/textured regions, large in homogeneous areas. The only required parameter is the energy threshold *T*. No need to specify the number of segments.

> R. Achanta, P. Marquez-Neila, P. Fua, S. Süsstrunk,
> \*"Scale-Adaptive Superpixels"\*, Color and Imaging Conference (CIC26), 2018.

**SICLE** — Superpixels through Iterative CLEarcutting. The user specifies the desired number of superpixels. Achieves state-of-the-art boundary delineation through iterative seed oversampling and relevance-based removal using the Image Foresting Transform (IFT). Supports optional saliency maps (e.g. normalized CHM) for object-aware segmentation.

> F.C. Belém, I.B. Barcelos, L.M. João, B. Perret, J. Cousty, S.J.F. Guimarães, A.X. Falcão,
> \*"Novel Arc-Cost Functions and Seed Relevance Estimations for Compact and Accurate Superpixels"\*,
> Journal of Mathematical Imaging and Vision, 65:770–786, 2023.

The original C implementation (`plGeoAdaptels`) was developed by **Paweł Netzel** at the University of Agriculture in Kraków, Poland. This Python package is a faithful reimplementation of that C code using Numba JIT compilation for near-native performance.

The algorithm was applied to standing dead tree detection in:

> Pawelec, I., Hawryło, P., Netzel, P., \& Socha, J. (2026).
> Evaluating superpixel algorithms for standing dead tree delineation using aerial orthoimagery.
> JAG, 147, 105180. doi:10.1016/j.jag.2026.105180

## Installation

**Recommended (conda + pip):**

```bash
# 1. Install native dependencies via conda
conda install -c conda-forge numpy numba rasterio fiona

# 2. Install plgeoadaptels
pip install --no-deps .          # from cloned repo
# or
pip install --no-deps git+https://github.com/igorpawelec/plgeoadaptels.git
```

> \*\*Note:\*\* Dependencies are installed via conda to avoid conflicts with GDAL/PROJ native libraries. The `--no-deps` flag prevents pip from overwriting conda packages.

## Quick start

### Python API

```python
from plgeoadaptels import create\_adaptels, adaptels\_from\_array

# From GeoTIFF — reads, segments, writes output
labels, n = create\_adaptels("input.tif", "output.tif", threshold=60.0)
print(f"Created {n} adaptels")

# Multiple input layers (treated as bands)
labels, n = create\_adaptels(
    \["band1.tif", "band2.tif", "band3.tif"],
    "adaptels.tif",
    threshold=40.0,
    normalize=True,
)

# From numpy arrays (no file I/O)
import numpy as np
data = np.random.rand(3, 500, 500)
labels, n = adaptels\_from\_array(data, threshold=30.0)
```

### Vectorization (no geopandas needed)

```python
from plgeoadaptels.vectorize import vectorize\_from\_file

# Adaptel raster → Shapefile (or .gpkg, .geojson)
n\_poly = vectorize\_from\_file("output.tif", "adaptels.shp")
```

### SICLE superpixels

```python
from plgeoadaptels.sicle import sicle\_from\_array, create\_sicle

# From numpy arrays — specify desired number of superpixels
labels, n = sicle\_from\_array(data, n\_segments=200)

# With saliency map (e.g. normalized CHM) — favors object boundaries
labels, n = sicle\_from\_array(data, n\_segments=200, saliency=chm\_normalized)

# From GeoTIFF — reads, segments, writes output
labels, n = create\_sicle("input.tif", "output.tif", n\_segments=200)
```

### Command line

```bash
plgeoadaptels -i input.tif -o output.tif -t 60.0
plgeoadaptels -i b1.tif -i b2.tif -o result.tif -t 0.03 -8 -d angular -n
```

### As Python module

```bash
python -m plgeoadaptels -i input.tif -o output.tif -t 60.0
```

## Parameters

### Adaptels

|Parameter|Default|Description|
|-|-|-|
|`threshold`|60.0|Energy threshold *T*. Lower → smaller adaptels, higher → larger. **Its scale depends on `distance` — see below**|
|`distance`|`'minkowski'`|Distance metric: `'minkowski'`, `'cosine'`, `'angular'`|
|`minkowski\_p`|2.0|Minkowski *p* parameter (2.0 = Euclidean)|
|`queen\_topology`|`False`|`True` = 8-connectivity, `False` = 4-connectivity|
|`normalize`|`False`|Normalize inputs to \[0, 1] before processing|

#### Choosing `threshold` for a metric

The metrics do not share a scale, so a threshold does not carry across them.
The default of 60 is scaled for `minkowski`; passing it to `cosine` or
`angular` would merge the whole raster into one adaptel, so it is rejected
with an error rather than silently returning nonsense.

|`distance`|Range|Typical `threshold`|Notes|
|-|-|-|-|
|`minkowski`|0 … data range|10 – 120|Grows with band count and bit depth. For 8-bit 3-band imagery distances reach \~441|
|`cosine`|0 … 1|0.002 – 0.05|Spectral angle as `1 - cos`. Insensitive to brightness|
|`angular`|0 … 1|0.005 – 0.2|`arccos(cos) / π`. Also brightness-insensitive, more linear than `cosine`|

Measured on the 3-band raster in `test\_data/`, 200 × 200 px:

|`threshold`|`minkowski`||`threshold`|`angular`|
|-|-|-|-|-|
|10|4140||0.002|7834|
|30|1512||0.01|2654|
|60|700||0.03|855|
|120|299||0.2|99|

`cosine` and `angular` compare the *direction* of the spectral vector rather
than its length, so the same material lit differently — a crown in sun versus
in shade — lands in one adaptel. `minkowski` will split it.

#### A note on where `60` comes from

The default is inherited from the original method, which works in **CIELAB**,
where a colour distance of 1 is roughly one just-noticeable difference and
40-80 is the recommended band. Raster bands are not CIELAB: reflectance, DN
values and indices all carry their own scales. Treat 60 as a starting point
to calibrate from, not a value with meaning on your data.

### Contiguity

Adaptels compete for pixels — a later adaptel takes a pixel from an earlier
one whenever it reaches it with a smaller accumulated distance. That
competition is what gives the method its boundary adherence, but it can also
cut an earlier adaptel in two, leaving one label spread over separate
patches.

Measured on the 3-band raster in `test\_data/`, 400 × 400 px:

|`threshold`|adaptels|not contiguous|worst|
|-|-|-|-|
|20|9289|242 (2.6%)|9 pieces|
|60|2770|265 (9.6%)|4 pieces|
|100|1488|167 (11.2%)|4 pieces|

This is harmless if the labels are only a lookup. It is **not** harmless for
object-based analysis: zonal statistics over a split adaptel average two
spatially separate patches into one "object".

```python
from plgeoadaptels import adaptels\_from\_array, enforce\_connectivity

labels, n = adaptels\_from\_array(data, threshold=60.0)   # 2770, 265 split
labels, n = enforce\_connectivity(labels)                # 3066, 0 split
```

Every connected component becomes its own adaptel, so the count rises by
about 10%. Nothing is merged and no pixel changes hands. Pass
`min\_size=` to absorb fragments below a size into an adjacent adaptel
instead of keeping them. Needs `scipy`.

### SICLE

|Parameter|Default|Description|
|-|-|-|
|`n\_segments`|200|Desired number of superpixels|
|`n\_oversampling`|3000|Initial seed count (N₀ ≫ n\_segments)|
|`n\_iterations`|2|Max IFT iterations (Ω). 2 is optimal per Belém 2023|
|`saliency`|`None`|Object saliency map (H,W) float64 \[0,1], e.g. normalized CHM|
|`quiet`|`False`|Suppress progress messages|

## Performance

First call includes Numba JIT compilation (\~5 s). Subsequent calls run at near-C speed:

|Raster|Bands|Time|Adaptels|
|-|-|-|-|
|400 × 400|3|0.03 s|\~2 800|
|2000 × 2000|3|\~1 s|\~70 000|

## Repository structure

```
plgeoadaptels/
├── plgeoadaptels/        # Package source
│   ├── \_\_init\_\_.py       # Public API (lazy imports)
│   ├── \_\_main\_\_.py       # CLI entry point
│   ├── adaptels.py       # High-level API (threshold-based)
│   ├── sicle.py          # SICLE superpixels (n\_segments-based)
│   ├── cli.py            # Command-line interface
│   ├── core.py           # Numba JIT algorithm kernel + min-heap
│   ├── io.py             # GeoTIFF read/write (rasterio)
│   └── vectorize.py      # Raster→vector (rasterio + fiona)
├── tests/                # Pytest suite
├── test\_data/            # Sample rasters
├── www/                  # Logo \& assets
├── pyproject.toml
├── environment.yaml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

* Python ≥ 3.9
* NumPy ≥ 1.21
* Numba ≥ 0.56
* Rasterio ≥ 1.3
* Fiona ≥ 1.9 *(for vectorization)*

## Citation

If you use this software in your research, please cite:

1. **This implementation:**

> Pawelec, I. (2026). plGeoAdaptels — Scale-Adaptive Superpixels for geospatial data \[Software]. https://github.com/igorpawelec/plgeoadaptels

2. **The original algorithm:**

> Achanta, R., Marquez-Neila, P., Fua, P., \& Süsstrunk, S. (2018). Scale-Adaptive Superpixels. \*Color and Imaging Conference (CIC26)\*.

3. **The original C implementation:**

> Netzel, P. plGeoAdaptels \[Software]. University of Agriculture in Kraków. https://gitlist.netzel.pl/

4. **SICLE algorithm:**

> Belém, F.C., Barcelos, I.B., João, L.M., Perret, B., Cousty, J., Guimarães, S.J.F., \& Falcão, A.X. (2023). Novel Arc-Cost Functions and Seed Relevance Estimations for Compact and Accurate Superpixels. \*Journal of Mathematical Imaging and Vision\*, 65, 770–786.

See also [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

