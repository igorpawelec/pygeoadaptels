"""grow_seeds: seeded spectral region growing.

The point->pixel contract is tested first and on its own, per SPEC_grow_seeds
section 9.9, because it is the one place R and Python can disagree silently --
the same class of underspecification that left rHRG != pyHRG on real CHMs.

The contract (SPEC section 5.1): pixel (r, c) covers the half-open extent
    [x0 + c*w, x0 + (c+1)*w) x (y0 - r*h, y0 - (r+1)*h]
so  c = floor((x - x0) / w),  r = floor((y0 - y) / h).
A point exactly on a shared edge therefore belongs to the pixel to the
right / below. Python indices are 0-based; the R twin returns the same
(r, c) + 1.

Copyright (C) 2026 Igor Pawelec. Licence: GPLv3.
"""

import numpy as np
import pytest


# A concrete north-up transform, deliberately at real-world magnitude
# (EPSG:2180-ish easting/northing) and 0.25 m pixels -- the motivating
# resolution. 8 columns x 6 rows. Every coordinate below is exactly
# representable in float64, so the expectations are exact, not approximate.
X0, Y0 = 500000.0, 400000.0    # top-left corner (west edge, north edge)
W = H = 0.25
COLS, ROWS = 8, 6

# (label, x, y, expected_row, expected_col) with 0-based indices.
# Rows outside [0, ROWS) or cols outside [0, COLS) are the raw floor result;
# validation (SPEC 5.4) rejects them later, but the conversion itself is
# total and deterministic, and that is what is pinned here.
POINT_TABLE = [
    ("centre of (0,0)",              500000.125, 399999.875, 0, 0),
    ("centre of (2,3)",              500000.875, 399999.375, 2, 3),
    ("vertical edge col3|col4 -> right", 500001.0,   399999.9,   0, 4),
    ("horizontal edge row1|row2 -> below", 500000.1, 399999.5,   2, 0),
    ("top-left outer corner -> (0,0)", 500000.0,   400000.0,   0, 0),
    ("last valid pixel (5,7) centre", 500001.875, 399998.625, 5, 7),
    ("outer east edge -> col == COLS (outside)", 500002.0, 399999.9, 0, 8),
    ("outer south edge -> row == ROWS (outside)", 500000.1, 399998.5, 6, 0),
]


class TestPointToPixel:

    @pytest.mark.parametrize("label,x,y,er,ec", POINT_TABLE,
                             ids=[t[0] for t in POINT_TABLE])
    def test_contract(self, label, x, y, er, ec):
        from plgeoadaptels.grow import _point_to_pixel
        r, c = _point_to_pixel(x, y, X0, Y0, W, H)
        assert (r, c) == (er, ec), f"{label}: got ({r},{c}), want ({er},{ec})"

    def test_returns_python_ints(self):
        """Flat indexing later does r*cols+c; numpy scalars there are a
        silent source of int64/overflow surprises. Keep them plain ints."""
        from plgeoadaptels.grow import _point_to_pixel
        r, c = _point_to_pixel(500000.1, 399999.9, X0, Y0, W, H)
        assert type(r) is int and type(c) is int

    def test_edge_rule_is_right_and_below_not_left_and_above(self):
        """The half-open rule is the whole point of specifying this: a point
        on an internal grid line must land on exactly one pixel, the same one
        in both languages. Probe both axes at a shared line."""
        from plgeoadaptels.grow import _point_to_pixel
        # x exactly on the col2|col3 line, y mid-pixel-row-4
        r, c = _point_to_pixel(X0 + 3 * W, Y0 - 4.5 * H, X0, Y0, W, H)
        assert c == 3, "east edge must belong to the right pixel"
        # y exactly on the row2|row3 line, x mid-pixel-col-1
        r, c = _point_to_pixel(X0 + 1.5 * W, Y0 - 3 * H, X0, Y0, W, H)
        assert r == 3, "south edge must belong to the pixel below"

    def test_matches_a_bruteforce_floor_over_the_grid(self):
        """Cross-check the closed form against an independent per-pixel
        containment test on every pixel centre -- catches an off-by-one in
        either direction that hand-picked cases might miss."""
        from plgeoadaptels.grow import _point_to_pixel
        for r in range(ROWS):
            for c in range(COLS):
                x = X0 + (c + 0.5) * W
                y = Y0 - (r + 0.5) * H
                assert _point_to_pixel(x, y, X0, Y0, W, H) == (r, c)


class TestGrowth:
    """Synthetic growth tests (SPEC section 9.1-9.5), no data files needed."""

    @staticmethod
    def _two_blocks(rows=12, cols=20, mid=10, lo=10.0, hi=200.0):
        d = np.empty((1, rows, cols), dtype=np.float64)
        d[0, :, :mid] = lo
        d[0, :, mid:] = hi
        return d, mid

    def test_two_blocks_two_seeds_split_on_the_edge(self):
        """9.1: one seed per block, no cap -> two segments, boundary exactly on
        the block edge, nothing unassigned."""
        from plgeoadaptels.grow import grow_seeds
        d, mid = self._two_blocks()
        seeds = np.array([[3, 0], [3, 19]])
        labels = grow_seeds(d, seeds, quiet=True)
        assert set(np.unique(labels)) == {0, 1}
        assert (labels == -1).sum() == 0
        assert (labels[:, :mid] == 0).all()
        assert (labels[:, mid:] == 1).all()

    def test_max_cost_confines_a_single_seed_to_its_block(self):
        """9.2: one seed, cap below the block-to-block jump -> its block only,
        the far block stays -1."""
        from plgeoadaptels.grow import grow_seeds
        d, mid = self._two_blocks(lo=10.0, hi=200.0)   # jump = 190
        seeds = np.array([[3, 0]])
        labels = grow_seeds(d, seeds, max_cost=50.0, quiet=True)
        assert (labels[:, :mid] == 0).all()
        assert (labels[:, mid:] == -1).all()

    def test_compactness_splits_uniform_at_the_midpoint(self):
        """9.3: the section 3.3 regression. On a uniform image two seeds split
        at the geometric midpoint, deterministically; without compactness the
        boundary is decided by seed arrival order."""
        from plgeoadaptels.grow import grow_seeds
        rows, cols = 15, 20
        d = np.full((1, rows, cols), 42.0)
        # Seeds at col 4 and col 15 -> perpendicular bisector at col 9.5, which
        # no integer column hits, so there are no ties to break.
        seeds = np.array([[7, 4], [7, 15]])
        labels = grow_seeds(d, seeds, compactness=1.0, quiet=True)
        assert (labels[:, :10] == 0).all(), "left of midpoint -> nearer seed 0"
        assert (labels[:, 10:] == 1).all(), "right of midpoint -> nearer seed 1"

    def test_label_i_is_seeds_i_under_shuffle(self):
        """9.4: the label contract. Whatever the seed order, the pixel a seed
        sits on carries that seed's own index."""
        from plgeoadaptels.grow import grow_seeds
        rng = np.random.default_rng(0)
        d = rng.random((3, 30, 30)) * 255
        seeds = np.array([[5, 5], [5, 25], [25, 5], [25, 25], [15, 15]])
        for _ in range(4):
            order = rng.permutation(len(seeds))
            s = seeds[order]
            labels = grow_seeds(d, s, quiet=True)
            for i, (r, c) in enumerate(s):
                assert labels[r, c] == i, (i, r, c, labels[r, c])

    def test_defaults_reproduce_one_kernel_call(self):
        """9.5: with every option off, grow_seeds is a strict superset of a
        single _ift_fmax call -- bit-for-bit, not approximately."""
        from plgeoadaptels.grow import grow_seeds
        from plgeoadaptels.sicle import _ift_fmax
        rng = np.random.default_rng(1)
        rows, cols = 40, 50
        d = rng.random((3, rows, cols)) * 255
        seeds = np.array([[10, 10], [10, 40], [30, 25]])

        got = grow_seeds(d, seeds, quiet=True)

        layers = d.reshape(3, rows * cols).copy()
        mask = np.zeros(rows * cols, dtype=np.uint8)
        flat = (seeds[:, 0] * cols + seeds[:, 1]).astype(np.int64)
        lab = np.empty(rows * cols, dtype=np.int32)
        cost = np.empty(rows * cols, dtype=np.float64)
        _ift_fmax(np.ascontiguousarray(layers), np.int32(3), mask,
                  np.int32(cols), np.int32(rows), flat, np.int64(len(flat)),
                  lab, cost)
        np.testing.assert_array_equal(got, lab.reshape(rows, cols))

    def test_band_weights_can_switch_a_band_off(self):
        """3.2: a zero weight on the discriminating band makes growth ignore
        it. Two bands; band 1 is a step the seed cannot cross under the cap."""
        from plgeoadaptels.grow import grow_seeds
        rows, cols = 12, 20
        d = np.empty((2, rows, cols))
        d[0, :, :] = 100.0                 # uniform, carries no information
        d[1, :, :10] = 0.0                 # the step is entirely in band 1
        d[1, :, 10:] = 100.0
        seeds = np.array([[6, 0]])
        # Weighted normally, the band-1 step (100) exceeds the cap, so the far
        # half is unreachable.
        on = grow_seeds(d, seeds, max_cost=50.0, quiet=True)
        assert (on[:, 10:] == -1).all()
        # Zero the band, and the step disappears: the far half grows in.
        off = grow_seeds(d, seeds, max_cost=50.0, band_weights=[1.0, 0.0],
                         quiet=True)
        assert (off[:, 10:] == 0).all()

    def test_seed_window_median_rescues_an_outlier_click(self):
        """3.4: a seed dropped on a highlight anchors the whole object on a bad
        value. The k*k median replaces it with the local signature. Also
        exercises the unit fix: the median is taken on the (here unweighted)
        stack, matching what the kernel then compares against."""
        from plgeoadaptels.grow import grow_seeds
        rows, cols = 15, 15
        d = np.full((1, rows, cols), 50.0)
        d[0, 7, 7] = 200.0                 # the click landed on a highlight
        seeds = np.array([[7, 7]])
        # Raw pixel: signature 200, every neighbour is 50, cost 150 > cap, so
        # nothing but the seed grows.
        raw = grow_seeds(d, seeds, max_cost=50.0, seed_window=1, quiet=True)
        assert (raw >= 0).sum() == 1
        # 3x3 median = 50 (eight 50s and one 200): signature becomes the
        # region, and it grows.
        med = grow_seeds(d, seeds, max_cost=50.0, seed_window=3, quiet=True)
        assert (med >= 0).sum() == rows * cols

    def test_seed_window_must_be_odd(self):
        from plgeoadaptels.grow import grow_seeds
        d = np.full((1, 8, 8), 10.0)
        with pytest.raises(ValueError, match="odd"):
            grow_seeds(d, np.array([[4, 4]]), seed_window=2, quiet=True)

    def test_fill_holes_fills_an_interior_pocket(self):
        """A bright pixel inside a crown exceeds max_cost and is left -1, a
        donut hole in the polygon. fill_holes closes it; without it the pocket
        stays open."""
        from plgeoadaptels.grow import grow_seeds
        pytest.importorskip("scipy")
        d = np.full((1, 11, 11), 50.0)
        d[0, 3, 3] = 200.0                     # interior highlight
        seeds = np.array([[5, 5]])
        open_ = grow_seeds(d, seeds, max_cost=30.0, quiet=True)
        assert open_[3, 3] == -1, "the highlight is over the cap -> a hole"
        filled = grow_seeds(d, seeds, max_cost=30.0, fill_holes=True, quiet=True)
        assert filled[3, 3] == 0, "fill_holes closes the enclosed pocket"
        assert int((filled != open_).sum()) == 1, "only the pocket changed"

    def test_fill_holes_leaves_nodata_and_edges_alone(self):
        """A pocket that touches nodata is not interior to one crown, so it is
        left as-is -- the guard that stops fill from eating the nodata region."""
        from plgeoadaptels.grow import grow_seeds
        pytest.importorskip("scipy")
        d = np.full((1, 11, 11), 50.0)
        d[0, 3, 4] = 200.0                     # highlight -> -1
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[3, 3] = 1                         # nodata right next to it
        filled = grow_seeds(d, seeds=np.array([[5, 5]]), mask=mask,
                            max_cost=30.0, fill_holes=True, quiet=True)
        assert filled[3, 4] == -1, "a pocket touching nodata is not filled"
        assert filled[3, 3] == -1, "nodata itself stays unassigned"

    def test_max_radius_caps_the_reach(self):
        """3.5: on a uniform image a single seed would fill everything; the
        radius sends anything beyond it back to -1."""
        from plgeoadaptels.grow import grow_seeds
        rows, cols = 21, 21
        d = np.full((1, rows, cols), 42.0)
        seeds = np.array([[10, 10]])
        labels = grow_seeds(d, seeds, max_radius=5.0, quiet=True)
        assert labels[10, 13] == 0, "3 px away is within the radius"
        assert labels[10, 18] == -1, "8 px away is beyond it"
        rr, cc = np.mgrid[0:rows, 0:cols]
        dist = np.sqrt((rr - 10) ** 2 + (cc - 10) ** 2)
        assert (labels[dist > 5.0] == -1).all()


class TestFromFiles:
    """The file wrapper: point layer -> pixels -> growth -> raster + polygons,
    on the real raster (SPEC section 9.6). The point->pixel arithmetic is
    already pinned in TestPointToPixel; here it is wired to an actual transform
    and a real vector round-trip."""

    @staticmethod
    def _raster():
        import pathlib
        p = pathlib.Path(__file__).resolve().parent.parent / "test_data" / \
            "SNP_21_2020_1.tif"
        if not p.exists():
            pytest.skip("test_data/SNP_21_2020_1.tif not present")
        return str(p)

    def _valid_seed_pixels(self, path, n=5):
        """A few well-separated unmasked pixels and their map-coord centres."""
        rasterio = pytest.importorskip("rasterio")
        with rasterio.open(path) as s:
            t = s.transform
            band = s.read(1)
            nod = s.nodata
        pts_rc, pts_xy = [], []
        for (r, c) in [(80, 80), (80, 320), (320, 80), (320, 320), (200, 200)][:n]:
            if nod is not None and band[r, c] == nod:
                continue
            pts_rc.append((r, c))
            pts_xy.append((t.c + (c + 0.5) * t.a, t.f + (r + 0.5) * t.e))
        return pts_rc, pts_xy

    def test_array_points_land_on_their_pixels(self, tmp_path):
        pytest.importorskip("rasterio")
        from plgeoadaptels.grow import grow_seeds_from_files
        src = self._raster()
        pts_rc, pts_xy = self._valid_seed_pixels(src)
        out_tif = tmp_path / "labels.tif"
        labels = grow_seeds_from_files(src, np.array(pts_xy),
                                       output_file=str(out_tif), quiet=True)
        # Each point's own pixel carries that point's index -- the round trip
        # map-coord -> _point_to_pixel returned the pixel it started from.
        for i, (r, c) in enumerate(pts_rc):
            assert labels[r, c] == i, (i, r, c, labels[r, c])
        assert out_tif.exists()
        import rasterio
        with rasterio.open(out_tif) as s:
            assert s.nodata == -1
            assert s.read(1).shape == labels.shape

    def test_reads_a_gpkg_point_layer_the_same_as_an_array(self, tmp_path):
        fiona = pytest.importorskip("fiona")
        pytest.importorskip("rasterio")
        from plgeoadaptels.grow import grow_seeds_from_files
        src = self._raster()
        _, pts_xy = self._valid_seed_pixels(src)

        gpkg = tmp_path / "seeds.gpkg"
        schema = {"geometry": "Point", "properties": {"id": "int"}}
        with fiona.open(gpkg, "w", driver="GPKG", crs="EPSG:2180",
                        schema=schema) as dst:
            for i, (x, y) in enumerate(pts_xy):
                dst.write({"geometry": {"type": "Point", "coordinates": (x, y)},
                           "properties": {"id": i}})

        from_arr = grow_seeds_from_files(src, np.array(pts_xy),
                                         max_cost=25.0, quiet=True)
        from_gpkg = grow_seeds_from_files(src, str(gpkg), max_cost=25.0,
                                          quiet=True)
        np.testing.assert_array_equal(from_arr, from_gpkg)

    def test_writes_polygons(self, tmp_path):
        fiona = pytest.importorskip("fiona")
        pytest.importorskip("rasterio")
        from plgeoadaptels.grow import grow_seeds_from_files
        src = self._raster()
        _, pts_xy = self._valid_seed_pixels(src)
        out_gpkg = tmp_path / "crowns.gpkg"
        labels = grow_seeds_from_files(src, np.array(pts_xy), max_cost=20.0,
                                       polygons=str(out_gpkg), quiet=True)
        assert out_gpkg.exists()
        with fiona.open(out_gpkg) as s:
            ids = [f["properties"]["adaptel_id"] for f in s]
        # Every polygon carries a real segment id, and only grown segments
        # become polygons (unassigned -1 is dropped).
        assert ids, "no polygons written"
        assert all(0 <= i < len(pts_xy) for i in ids)
        assert set(ids) <= set(np.unique(labels[labels >= 0]).tolist())

    @staticmethod
    def _lab():
        import pathlib
        p = pathlib.Path(__file__).resolve().parent.parent / "test_data" / \
            "SNP_21_2020_1_lab.tif"
        if not p.exists():
            pytest.skip("test_data/SNP_21_2020_1_lab.tif not present")
        return str(p)

    @staticmethod
    def _dead_trees():
        import pathlib
        p = pathlib.Path(__file__).resolve().parent.parent / "test_data" / \
            "dead_trees_test.shp"
        if not p.exists():
            pytest.skip("test_data/dead_trees_test.shp not present")
        return str(p)

    def test_dead_trees_on_the_lab_raster(self, tmp_path):
        """The motivating case end to end: an operator's 36 dead-tree points
        on the CIELAB ortho. Pins the numbers so a future change to the growth
        or the wrapper has to own any shift in them."""
        pytest.importorskip("fiona")
        pytest.importorskip("rasterio")
        from plgeoadaptels.grow import grow_seeds_from_files
        lab, shp = self._lab(), self._dead_trees()

        labels = grow_seeds_from_files(lab, shp, max_cost=10, quiet=True)
        # 36 points in, 36 labels out, none dropped, label i present for all i.
        present = set(np.unique(labels[labels >= 0]).tolist())
        assert present == set(range(36)), sorted(present)
        # nothing is assigned a label outside the seed set.
        assert labels.max() == 35 and labels.min() == -1

        # fill_holes only ever fills, never removes.
        filled = grow_seeds_from_files(lab, shp, max_cost=10, fill_holes=True,
                                       quiet=True)
        assert (filled >= 0).sum() >= (labels >= 0).sum()
        assert set(np.unique(filled[filled >= 0]).tolist()) == set(range(36))

    def test_a_star_weight_fills_more_and_spills_less(self, tmp_path):
        """The dead-tree recipe found on this scene, kept honest: down-weighting
        L and up-weighting a* grows the crowns further while a smaller share of
        the assigned pixels are green (healthy) canopy. Measured on the a*
        band directly, not asserted from theory."""
        pytest.importorskip("fiona")
        rasterio = pytest.importorskip("rasterio")
        from plgeoadaptels.grow import grow_seeds_from_files
        lab, shp = self._lab(), self._dead_trees()
        a_star = rasterio.open(lab).read(2)

        def green_share(labels):
            a = a_star[labels >= 0]
            return (a < -2).mean()          # a* < -2 is green canopy

        flat = grow_seeds_from_files(lab, shp, max_cost=15, max_radius=20,
                                     quiet=True)
        weighted = grow_seeds_from_files(lab, shp, max_cost=15, max_radius=20,
                                         band_weights=[0.5, 2.5, 1.0], quiet=True)
        assert (weighted >= 0).sum() > (flat >= 0).sum(), "should fill more"
        assert green_share(weighted) < green_share(flat), "should spill less"

    def test_reprojects_points_from_another_crs(self, tmp_path):
        """5.3: a point layer in a different CRS must be reprojected, not
        assumed to match. Points are written in EPSG:4326 and must land on the
        same pixels as their EPSG:2180 originals."""
        fiona = pytest.importorskip("fiona")
        pytest.importorskip("rasterio")
        from fiona.transform import transform as fiona_transform
        from plgeoadaptels.grow import grow_seeds_from_files
        src = self._raster()
        pts_rc, pts_xy = self._valid_seed_pixels(src)

        xs, ys = zip(*pts_xy)
        lon, lat = fiona_transform("EPSG:2180", "EPSG:4326", list(xs), list(ys))
        gpkg = tmp_path / "seeds_wgs84.gpkg"
        schema = {"geometry": "Point", "properties": {"id": "int"}}
        with fiona.open(gpkg, "w", driver="GPKG", crs="EPSG:4326",
                        schema=schema) as dst:
            for i, (x, y) in enumerate(zip(lon, lat)):
                dst.write({"geometry": {"type": "Point", "coordinates": (x, y)},
                           "properties": {"id": i}})

        labels = grow_seeds_from_files(src, str(gpkg), quiet=True)
        # 0.25 m pixels and a mm-accurate reprojection: each point returns to
        # its own pixel.
        for i, (r, c) in enumerate(pts_rc):
            assert labels[r, c] == i, (i, r, c, labels[r, c])
