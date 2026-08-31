"""Features are written in batches; nothing may be lost at the batch boundary.

Writing a GeoPackage one feature at a time is one SQLite transaction per row and
is roughly 12x slower than batching, so ``vectorize_adaptels`` buffers and calls
``writerecords``. The failure that buys is silent: drop the final partial batch
after the loop and the file simply ends short, with the returned count still
claiming every polygon was written.

WRITE_BATCH is monkeypatched to a small value so the tail case is reached with a
handful of polygons instead of twenty thousand, and every assertion compares
what was *read back* against what was returned.

Run: pytest tests/test_vectorize_batching.py -v
"""
import numpy as np
import pytest

fiona = pytest.importorskip("fiona")
pytest.importorskip("rasterio")

from affine import Affine  # noqa: E402

from pygeoadaptels import vectorize as V  # noqa: E402

CRS_WKT = (
    'PROJCS["ETRF2000-PL / CS92",GEOGCS["ETRF2000-PL",'
    'DATUM["ETRF2000_Poland",SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",19],'
    'PARAMETER["scale_factor",0.9993],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",-5300000],'
    'UNIT["metre",1]]'
)


def checkerboard(n=12):
    """One label per cell, so the polygon count is known exactly: n*n."""
    labels = np.arange(n * n, dtype=np.int32).reshape(n, n)
    return labels, Affine(0.25, 0.0, 500000.0, 0.0, -0.25, 300000.0)


def _write(tmp_path, ext, batch, monkeypatch, n=12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(V, "WRITE_BATCH", batch)
    labels, transform = checkerboard(n)
    out = tmp_path / f"cells{ext}"
    returned = V.vectorize_adaptels(labels, transform, CRS_WKT, str(out),
                                    quiet=True)
    with fiona.open(str(out)) as src:
        props = [f["properties"] for f in src]
    return returned, props


@pytest.mark.parametrize("ext", [".gpkg", ".shp"])
@pytest.mark.parametrize("batch", [1, 7, 144, 1000])
def test_no_feature_is_lost_at_the_batch_boundary(tmp_path, monkeypatch, ext,
                                                  batch):
    """144 cells against batches that divide it exactly, and ones that do not.

    batch=7 leaves a partial final batch, batch=144 leaves none, and batch=1000
    means the whole thing is only ever flushed after the loop -- each takes a
    different path through the buffering.
    """
    returned, props = _write(tmp_path, ext, batch, monkeypatch)
    assert returned == 144, f"vectorize returned {returned}, expected 144"
    assert len(props) == returned, (
        f"{returned} reported written but {len(props)} are in the file "
        f"(batch={batch})")
    assert sorted(p["adaptel_id"] for p in props) == list(range(144))


def test_batching_does_not_change_the_result(tmp_path, monkeypatch):
    """Batch size is a performance knob and must not touch the output."""
    a_ret, a = _write(tmp_path / "a", ".gpkg", 5, monkeypatch)
    b_ret, b = _write(tmp_path / "b", ".gpkg", 100000, monkeypatch)
    assert a_ret == b_ret
    key = lambda ps: sorted((p["adaptel_id"], round(p["area_m2"], 6))
                            for p in ps)
    assert key(a) == key(b)


def test_write_batch_is_sane():
    assert isinstance(V.WRITE_BATCH, int)
    assert V.WRITE_BATCH >= 1
