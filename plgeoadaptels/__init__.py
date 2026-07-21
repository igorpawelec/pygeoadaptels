"""
plGeoAdaptels — Scale-Adaptive Superpixels for geospatial data.

A pure Python + Numba implementation of two complementary superpixel
algorithms for geospatial raster data:

1. **Adaptels** — scale-adaptive superpixels controlled by an energy
   threshold (no need to specify the number of segments).
2. **SICLE** — Superpixels through Iterative CLEarcutting, controlled
   by a target number of segments (best boundary delineation in
   benchmarks).

Based on:
    Adaptels: R. Achanta et al., "Scale-Adaptive Superpixels", CIC26, 2018.
    SICLE: F.C. Belém et al., "Novel Arc-Cost Functions and Seed Relevance
    Estimations for Compact and Accurate Superpixels", JMIV, 65:770–786, 2023.

Original C implementation (Adaptels):
    Paweł Netzel, University of Agriculture in Kraków, Poland.

Python + Numba reimplementation:
    Igor Pawelec, Kraków, Poland.

Usage::

    from plgeoadaptels import create_adaptels, adaptels_from_array

    # Adaptels: threshold-based (no n_segments needed)
    labels, n = create_adaptels('input.tif', 'output.tif', threshold=60.0)
    labels, n = adaptels_from_array(data_array, threshold=60.0)

    # SICLE: n_segments-based (best boundary delineation)
    from plgeoadaptels.sicle import create_sicle, sicle_from_array
    labels, n = sicle_from_array(data_array, n_segments=200)
    labels, n = create_sicle('input.tif', 'output.tif', n_segments=200)

    # Vectorize to Shapefile (no geopandas needed)
    from plgeoadaptels.vectorize import vectorize_from_file
    vectorize_from_file('output.tif', 'adaptels.shp')
"""

try:
    from importlib.metadata import version as _version
    __version__ = _version("plgeoadaptels")
except Exception:
    __version__ = "0.4.0"

__author__ = "Igor Pawelec"

# ── Import strategy ──────────────────────────────────────────────
# Try eager imports first (fast path when everything is installed).
# Fall back to lazy __getattr__ if numba/rasterio are broken or missing.

try:
    from .adaptels import (create_adaptels, adaptels_from_array,
                           enforce_connectivity)
    from .io import read_raster, write_raster, normalize_layers
    _LAZY_MODE = False
except (ImportError, OSError):
    _LAZY_MODE = True
    _LAZY_IMPORTS = {
        "create_adaptels":     ".adaptels",
        "enforce_connectivity": ".adaptels",
        "adaptels_from_array": ".adaptels",
        "read_raster":         ".io",
        "write_raster":        ".io",
        "normalize_layers":    ".io",
    }

    def __getattr__(name):
        if name in _LAZY_IMPORTS:
            import importlib
            try:
                mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
            except (ImportError, OSError) as e:
                raise ImportError(
                    f"Cannot load plgeoadaptels.{name}: {e}\n"
                    f"Install deps: conda install -c conda-forge numpy numba rasterio"
                ) from e
            obj = getattr(mod, name)
            globals()[name] = obj
            return obj
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_adaptels",
    "enforce_connectivity",
    "adaptels_from_array",
    "read_raster",
    "write_raster",
    "normalize_layers",
]
