"""The output is a GeoTIFF whatever the input was.

`write_raster` builds its profile from the input's metadata, and that metadata
carries the input's driver. Without an explicit override the result is written
in the input's format under a .tif name -- and for a VRT input the write fails
outright with "Writing through VRTSourcedRasterBand is not supported", which is
how this was found: selecting bands in the QGIS plugin hands these functions a
VRT.

Run: pytest tests/test_output_driver.py -v
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from pygeoadaptels.io import write_raster  # noqa: E402


def _meta(driver):
    return {
        "driver": driver,
        "height": 8,
        "width": 8,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:2180",
        "transform": from_origin(500000, 300000, 0.25, 0.25),
        "nodata": 0.0,
    }


@pytest.mark.parametrize("driver", ["VRT", "ENVI", "HFA", "GTiff"])
def test_write_raster_always_writes_gtiff(tmp_path, driver):
    """A profile inherited from a non-GTiff input must not decide the output."""
    out = tmp_path / f"labels_{driver}.tif"
    labels = np.arange(64, dtype=np.int32)
    write_raster(str(out), labels, _meta(driver), 8, 8)
    with rasterio.open(str(out)) as src:
        assert src.meta["driver"] == "GTiff", (
            f"input driver {driver} leaked into the output")
        assert src.meta["dtype"] == "int32"
        assert src.count == 1
        assert np.array_equal(src.read(1).ravel(), labels)
