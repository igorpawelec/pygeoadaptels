"""
pytest suite for plGeoAdaptels.
Run: pytest tests/test_plgeoadaptels.py -v
"""

import numpy as np
import pytest


@pytest.fixture
def uniform_data():
    return np.ones((1, 100, 100), dtype=np.float64) * 42.0


@pytest.fixture
def split_data():
    d = np.zeros((1, 100, 100), dtype=np.float64)
    d[0, :50, :] = 10.0
    d[0, 50:, :] = 200.0
    return d


@pytest.fixture
def gradient_data():
    return np.linspace(0, 255, 100 * 100).reshape(1, 100, 100).astype(np.float64)


class TestAdaptelsFromArray:

    def test_uniform_gives_one(self, uniform_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(uniform_data, threshold=30.0)
        assert n == 1
        assert labels.shape == (100, 100)

    def test_split_gives_at_least_two(self, split_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(split_data, threshold=30.0)
        assert n >= 2

    def test_gradient_gives_many(self, gradient_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(gradient_data, threshold=30.0)
        assert n > 5

    def test_lower_threshold_more_adaptels(self, gradient_data):
        from plgeoadaptels import adaptels_from_array
        _, n_low = adaptels_from_array(gradient_data, threshold=10.0)
        _, n_high = adaptels_from_array(gradient_data, threshold=100.0)
        assert n_low > n_high

    def test_queen_topology(self, gradient_data):
        from plgeoadaptels import adaptels_from_array
        _, n4 = adaptels_from_array(gradient_data, threshold=30.0, queen_topology=False)
        _, n8 = adaptels_from_array(gradient_data, threshold=30.0, queen_topology=True)
        # 8-connectivity produces fewer or equal adaptels
        assert n8 <= n4

    def test_normalize(self, gradient_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(gradient_data, threshold=0.5, normalize=True)
        assert n >= 1
        assert labels.shape == (100, 100)

    # thresholds are per-metric: minkowski grows with the data range, while
    # cosine and angular are bounded by 1
    @pytest.mark.parametrize("dist,thresh", [("minkowski", 30.0),
                                             ("cosine", 0.03),
                                             ("angular", 0.03)])
    def test_distance_metrics(self, gradient_data, dist, thresh):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(gradient_data, threshold=thresh,
                                        distance=dist)
        # `n >= 1` would pass even when a metric collapses the whole raster
        # into one adaptel, which is exactly how the broken cosine went
        # unnoticed. Require a real segmentation instead.
        assert n > 1, f"{dist} produced {n} adaptel(s): the metric is degenerate"
        assert labels.shape == gradient_data.shape[1:]
        assert set(np.unique(labels)) <= set(range(n))

    def test_multiband(self):
        from plgeoadaptels import adaptels_from_array
        data = np.random.rand(3, 50, 50).astype(np.float64)
        labels, n = adaptels_from_array(data, threshold=30.0)
        assert n >= 1
        assert labels.shape == (50, 50)

    def test_labels_dtype_and_range(self, uniform_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(uniform_data, threshold=30.0)
        assert labels.dtype == np.int32
        assert labels.min() >= 0

    def test_invalid_threshold(self, uniform_data):
        from plgeoadaptels import adaptels_from_array
        with pytest.raises(ValueError):
            adaptels_from_array(uniform_data, threshold=-1.0)

    def test_invalid_distance(self, uniform_data):
        from plgeoadaptels import adaptels_from_array
        with pytest.raises(ValueError):
            adaptels_from_array(uniform_data, threshold=30.0, distance="invalid")


class TestImports:

    def test_import_package(self):
        import plgeoadaptels
        assert hasattr(plgeoadaptels, '__version__')

    def test_import_functions(self):
        from plgeoadaptels import create_adaptels, adaptels_from_array
        assert callable(create_adaptels)
        assert callable(adaptels_from_array)

    def test_import_vectorize(self):
        from plgeoadaptels.vectorize import vectorize_adaptels, vectorize_from_file
        assert callable(vectorize_adaptels)
        assert callable(vectorize_from_file)

    def test_import_sicle(self):
        from plgeoadaptels.sicle import sicle_from_array, create_sicle
        assert callable(sicle_from_array)
        assert callable(create_sicle)


# ── SICLE tests ──────────────────────────────────────────────────────

class TestSicleFromArray:

    def test_basic(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.rand(3, 50, 50).astype(np.float64)
        labels, n = sicle_from_array(data, n_segments=10, quiet=True)
        assert labels.shape == (50, 50)
        assert n == 10

    def test_single_band(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.rand(100, 100).astype(np.float64)
        labels, n = sicle_from_array(data, n_segments=20, quiet=True)
        assert labels.shape == (100, 100)
        assert n == 20

    def test_n_segments_respected(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.rand(3, 80, 80).astype(np.float64)
        for n_seg in [5, 50, 200]:
            labels, n = sicle_from_array(data, n_segments=n_seg, quiet=True)
            assert n == n_seg

    def test_labels_cover_image(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.rand(3, 60, 60).astype(np.float64)
        labels, n = sicle_from_array(data, n_segments=30, quiet=True)
        # Every pixel should be assigned
        assert np.all(labels >= 0)
        # All labels should be in range
        assert labels.max() < n

    def test_with_saliency(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.rand(3, 50, 50).astype(np.float64)
        sal = np.zeros((50, 50), dtype=np.float64)
        sal[15:35, 15:35] = 1.0  # "object" in center
        labels, n = sicle_from_array(data, n_segments=20,
                                     saliency=sal, quiet=True)
        assert labels.shape == (50, 50)
        assert n == 20

    def test_oversampling_auto(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.rand(3, 30, 30).astype(np.float64)
        # n_oversampling < n_segments auto-corrects, but says so: SICLE only
        # removes seeds, so the caller's N0 is silently discarded otherwise.
        with pytest.warns(UserWarning, match="n_oversampling"):
            labels, n = sicle_from_array(data, n_segments=10,
                                         n_oversampling=5, quiet=True)
        assert n == 10


class TestDistanceMetrics:
    """Each metric must behave as a distance, and honour its own scale.

    These exist because 'cosine' shipped returning a similarity: it was 1.0
    for identical pixels and fell towards 0 as they diverged, so the growth
    threshold worked backwards and the parameter barely moved the result.
    """

    @staticmethod
    def _dist(mean, px, code):
        from plgeoadaptels.core import calc_distance
        n = len(mean)
        layers = np.ascontiguousarray(np.array(px, dtype=np.float64).reshape(n, 1))
        return calc_distance(layers, n, np.array(mean, dtype=np.float64),
                             0, 1, code, 2.0)

    @pytest.mark.parametrize("code", [0, 1, 2])
    def test_zero_for_identical(self, code):
        base = [100.0, 150.0, 200.0]
        assert self._dist(base, base, code) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("code", [0, 1, 2])
    def test_grows_with_dissimilarity(self, code):
        base = [100.0, 150.0, 200.0]
        near = self._dist(base, [105.0, 155.0, 205.0], code)
        far = self._dist(base, [200.0, 50.0, 10.0], code)
        assert near <= far

    @pytest.mark.parametrize("distance", ["cosine", "angular"])
    def test_threshold_above_scale_rejected(self, distance):
        from plgeoadaptels import adaptels_from_array
        img = np.random.default_rng(0).uniform(0, 255, (3, 30, 30))
        # the package default of 60 is scaled for minkowski and would merge
        # a bounded metric into a single adaptel
        with pytest.raises(ValueError, match="outside the range"):
            adaptels_from_array(img, threshold=60.0, distance=distance)

    @pytest.mark.parametrize("distance", ["cosine", "angular"])
    def test_threshold_responds(self, distance):
        from plgeoadaptels import adaptels_from_array
        img = np.random.default_rng(0).uniform(0, 255, (3, 40, 40))
        _, n_tight = adaptels_from_array(img, threshold=0.002, distance=distance)
        _, n_loose = adaptels_from_array(img, threshold=0.2, distance=distance)
        assert n_tight > n_loose, "threshold must change the segmentation"

    def test_minkowski_accepts_large_threshold(self):
        from plgeoadaptels import adaptels_from_array
        img = np.random.default_rng(0).uniform(0, 255, (3, 30, 30))
        _, n = adaptels_from_array(img, threshold=60.0, distance="minkowski")
        assert n > 0

    def test_metrics_differ(self):
        from plgeoadaptels import adaptels_from_array
        img = np.random.default_rng(1).uniform(0, 255, (3, 40, 40))
        _, a = adaptels_from_array(img, threshold=0.03, distance="angular")
        _, c = adaptels_from_array(img, threshold=0.03, distance="cosine")
        assert a > 0 and c > 0


class TestEnforceConnectivity:
    """Adaptels compete for pixels, so a later one can cut an earlier one in
    two and leave a single label spread over separate patches. At the default
    threshold roughly 10% of adaptels come out in more than one piece, which
    matters as soon as anything computes zonal statistics per label.
    """

    @staticmethod
    def _n_split(labels, n):
        from scipy import ndimage
        c = 0
        for k in range(n):
            m = labels == k
            if m.any():
                _, nc = ndimage.label(m)
                if nc > 1:
                    c += 1
        return c

    @pytest.fixture
    def segmented(self):
        from plgeoadaptels import adaptels_from_array
        rng = np.random.default_rng(4)
        img = rng.uniform(0, 255, (3, 90, 90))
        return adaptels_from_array(img, threshold=30.0)

    def test_removes_all_splits(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, n = segmented
        out, n2 = enforce_connectivity(labels)
        assert self._n_split(out, n2) == 0

    def test_never_merges_across_adaptels(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, n = segmented
        out, n2 = enforce_connectivity(labels)
        for k in range(n2):
            assert len(np.unique(labels[out == k])) == 1, \
                "a new adaptel must lie inside exactly one old one"

    def test_preserves_coverage(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, _ = segmented
        out, _ = enforce_connectivity(labels)
        np.testing.assert_array_equal(labels >= 0, out >= 0)

    def test_labels_contiguous(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, _ = segmented
        out, n2 = enforce_connectivity(labels)
        assert sorted(np.unique(out[out >= 0])) == list(range(n2))

    def test_idempotent(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, _ = segmented
        out, _ = enforce_connectivity(labels)
        again, _ = enforce_connectivity(out)
        np.testing.assert_array_equal(out, again)

    def test_count_can_only_grow(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, n = segmented
        _, n2 = enforce_connectivity(labels)
        assert n2 >= n, "splitting can only add adaptels, never remove them"

    def test_min_size_absorbs_slivers(self):
        """A sliver below min_size joins a neighbour instead of becoming its own.

        Hand-built rather than taken from a segmented image: whether a random
        raster happens to contain a split adaptel small enough to absorb is
        not something a test should depend on.
        """
        from plgeoadaptels import enforce_connectivity
        labels = np.full((10, 10), -1, dtype=np.int32)
        labels[:, :] = 1              # background first...
        labels[0:5, 0:5] = 0          # a solid block
        labels[9, 9] = 0              # ...then one stray pixel with the same id
        assert labels[9, 9] == 0 and labels[9, 8] == 1, "fixture is set up wrong"

        keep, n_keep = enforce_connectivity(labels, min_size=0)
        assert n_keep == 3, "the stray pixel becomes its own adaptel"
        assert keep[9, 9] != keep[0, 0]

        absorb, n_absorb = enforce_connectivity(labels, min_size=1)
        assert n_absorb < n_keep, "the stray is absorbed, not kept"
        assert absorb[9, 9] == absorb[9, 8], "it joins the adjacent adaptel"
        assert self._n_split(absorb, n_absorb) == 0
        np.testing.assert_array_equal(labels >= 0, absorb >= 0)

    def test_min_size_preserves_coverage_on_real_segmentation(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, _ = segmented
        for ms in (0, 5, 20):
            out, n = enforce_connectivity(labels, min_size=ms)
            np.testing.assert_array_equal(labels >= 0, out >= 0)
            assert self._n_split(out, n) == 0

    def test_splits_a_hand_built_case(self):
        """Two separate blocks sharing one label must become two adaptels."""
        from plgeoadaptels import enforce_connectivity
        labels = np.full((10, 10), -1, dtype=np.int32)
        labels[1:3, 1:3] = 0
        labels[7:9, 7:9] = 0          # same id, nowhere near
        labels[5, 5] = 1
        out, n = enforce_connectivity(labels)
        assert n == 3
        assert out[1, 1] != out[7, 7]

    def test_all_nodata(self):
        from plgeoadaptels import enforce_connectivity
        out, n = enforce_connectivity(np.full((5, 5), -1, dtype=np.int32))
        assert n == 0
        assert (out < 0).all()

    def test_rejects_3d(self):
        from plgeoadaptels import enforce_connectivity
        with pytest.raises(ValueError, match="2-D"):
            enforce_connectivity(np.zeros((2, 2, 2), dtype=np.int32))

    def test_rejects_negative_min_size(self, segmented):
        from plgeoadaptels import enforce_connectivity
        labels, _ = segmented
        with pytest.raises(ValueError, match="min_size"):
            enforce_connectivity(labels, min_size=-1)


# ── Regression tests ─────────────────────────────────────────────────
#
# Every test below covers a bug that the suite above could not have
# caught, because it ran only on np.random.rand of at most 100x100 and
# never asserted anything about the shape of the result. The bugs were
# found by measuring on test_data/SNP_21_2020_1.tif; the reproductions
# here are synthetic so they need no data file.


class TestSicleHeapCapacity:
    """The IFT priority queue was capped at 100000 entries.

    An insert past the cap was skipped, but cost_out and labels_out had
    already been written a few lines earlier, so the pixel counted as
    conquered while never entering the queue. Its subtree stopped growing
    and the pixels behind it kept label -1. The cap was never hit on a
    small raster, which is why the original suite passed: at 700x700 the
    old code dropped 7 valid pixels, at 900x900 it dropped 93.
    """

    def test_no_valid_pixel_is_left_unlabelled(self):
        from plgeoadaptels.sicle import sicle_from_array
        # Fully valid raster, so every pixel is reachable from some seed
        # and any -1 in the output is a lost pixel, not nodata.
        data = np.random.default_rng(7).random((3, 700, 700))
        labels, n = sicle_from_array(data, n_segments=200, quiet=True)
        assert (labels >= 0).all(), (
            f"{int((labels < 0).sum())} valid pixels came back unlabelled"
        )
        assert n == 200

    def test_more_seeds_than_the_old_cap(self):
        from plgeoadaptels.sicle import sicle_from_array
        # 120000 seeds is above the old fixed 100000 heap slots, so the
        # seeding loop itself used to overflow and 20000 seeds never
        # propagated at all.
        data = np.random.default_rng(0).random((3, 400, 400))
        labels, n = sicle_from_array(data, n_segments=200,
                                     n_oversampling=120000, quiet=True)
        assert (labels >= 0).all()
        assert len(np.unique(labels)) == 200


class TestSicleSaliencyValidation:
    """NaN in the saliency map used to hijack the seed ranking.

    Relevance is sorted with argsort(...)[::-1]. NaN sorts to the end
    ascending, so reversing put NaN-relevance seeds at the *head* of the
    ranking — ahead of every seed whose relevance was real. On a 400x400
    scene one NaN band produced a superpixel covering 139820 of 160000
    pixels, with no exception and no warning.
    """

    def test_nan_over_valid_pixels_is_rejected(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.default_rng(0).random((3, 60, 60))
        sal = np.random.default_rng(3).random((60, 60))
        sal[:10, :] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            sicle_from_array(data, n_segments=20, saliency=sal, quiet=True)

    def test_nan_under_nodata_is_fine(self):
        """NaN where the mask already says nodata never reaches relevance."""
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.default_rng(0).random((3, 60, 60))
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[:10, :] = 1
        sal = np.random.default_rng(3).random((60, 60))
        sal[:10, :] = np.nan          # exactly the masked rows
        labels, n = sicle_from_array(data, mask=mask, n_segments=20,
                                     saliency=sal, quiet=True)
        assert n == 20
        assert (labels[:10, :] < 0).all()
        assert (labels[10:, :] >= 0).all()

    def test_wrong_shape_is_rejected(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.default_rng(0).random((3, 60, 60))
        with pytest.raises(ValueError, match="shape"):
            sicle_from_array(data, n_segments=20,
                             saliency=np.zeros((30, 30)), quiet=True)

    def test_clean_saliency_still_drives_the_result(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.default_rng(0).random((3, 60, 60))
        sal = np.zeros((60, 60))
        sal[20:40, 20:40] = 1.0
        with_sal, n1 = sicle_from_array(data, n_segments=20,
                                        saliency=sal, quiet=True)
        without, n2 = sicle_from_array(data, n_segments=20, quiet=True)
        assert n1 == n2 == 20
        assert not np.array_equal(with_sal, without)


class TestSicleDeterminism:
    """Seed removal drops the label remap; the result must not move.

    _ift_fmax reinitialises labels and cost on entry, so the remap that
    used to run between iterations was overwritten before it was read.
    Removing it is only safe if the output is unchanged and reproducible.
    """

    def test_repeated_runs_are_identical(self):
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.default_rng(11).random((3, 120, 120))
        a, na = sicle_from_array(data, n_segments=40, quiet=True)
        b, nb = sicle_from_array(data, n_segments=40, quiet=True)
        assert na == nb
        np.testing.assert_array_equal(a, b)

    def test_every_label_is_a_single_connected_region(self):
        """SICLE grows over an 8-adjacency, so labels must be 8-connected.

        Deliberately not enforce_connectivity(): that checks 4-connectivity,
        matching the adaptels grower's default neighbourhood, and SICLE's
        IFT expands to all 8 neighbours. Measured on both a random scene and
        SNP_21_2020_1.tif, SICLE labels are exactly 8-connected (one
        component each) and roughly 20x fragmented under a 4-connected
        reading — so the 4-connected tool reports a defect that is not there.
        """
        ndimage = pytest.importorskip("scipy.ndimage")
        from plgeoadaptels.sicle import sicle_from_array
        data = np.random.default_rng(11).random((3, 120, 120))
        labels, n = sicle_from_array(data, n_segments=40, quiet=True)
        eight = np.ones((3, 3), dtype=bool)
        split = [lab for lab in np.unique(labels)
                 if ndimage.label(labels == lab, structure=eight)[1] > 1]
        assert not split, f"{len(split)} superpixels are not 8-connected"
        assert len(np.unique(labels)) == n


class TestNormalizedThreshold:
    """normalize=True rescales bands to [0, 1] but the threshold did not.

    The package default of 60 is scaled for raw data. Once normalized, the
    largest minkowski distance possible is n_layers**(1/p) — 1.0 for one
    band, about 1.73 for three at p=2 — so the default merged the entire
    raster. On the test scene 2792 adaptels became exactly 1, silently.
    Same failure mode as an out-of-scale cosine threshold, which already
    raised; this one did not.
    """

    def test_default_threshold_is_rejected_when_normalized(self, split_data):
        from plgeoadaptels import adaptels_from_array
        with pytest.raises(ValueError, match="normalize"):
            adaptels_from_array(split_data, threshold=60.0, normalize=True)

    def test_error_names_the_ceiling(self):
        from plgeoadaptels import adaptels_from_array
        data = np.random.default_rng(0).random((3, 40, 40))
        with pytest.raises(ValueError, match=r"1\.7"):
            adaptels_from_array(data, threshold=60.0, normalize=True)

    def test_threshold_on_the_normalized_scale_works(self, split_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(split_data, threshold=0.5,
                                        normalize=True)
        assert n >= 2, "a two-valued raster must not collapse to one adaptel"

    def test_unnormalized_default_is_untouched(self, split_data):
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(split_data, threshold=60.0)
        assert n >= 2

    def test_bounded_metrics_are_unaffected(self, split_data):
        """cosine is capped at 1 either way, so normalize changes nothing."""
        from plgeoadaptels import adaptels_from_array
        labels, n = adaptels_from_array(split_data, threshold=0.03,
                                        distance='cosine', normalize=True)
        assert n >= 1


class TestSicleRelevanceAdjacency:
    """Seed relevance must use the same neighbourhood the forest grows over.

    Belem et al. 2023 define tree adjacency as A(Ts) = {Tt : exists <x,y> in
    A} over the arc set A the IFT itself uses, which here is 8-connected.
    Until 0.4.0 the relevance scan used 4, so a tree whose only neighbour
    touched it diagonally was found to have no neighbours: its minimum
    contrast stayed at the sentinel, collapsed to 0, and the seed was ranked
    least relevant and removed first on no evidence.
    """

    @staticmethod
    def _diagonal_pair():
        """Two 2x2 trees in a 4x4 grid, touching only at one corner."""
        labels = np.full(16, -1, dtype=np.int32)
        for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
            labels[r * 4 + c] = 0
        for r, c in ((2, 2), (2, 3), (3, 2), (3, 3)):
            labels[r * 4 + c] = 1
        layers = np.zeros((1, 16), dtype=np.float64)
        layers[0, labels == 1] = 50.0          # a contrast worth finding
        mask = np.zeros(16, dtype=np.uint8)
        return layers, labels, mask

    def test_diagonally_touching_trees_are_adjacent(self):
        from plgeoadaptels.sicle import _compute_seed_relevance
        layers, labels, mask = self._diagonal_pair()
        rel = _compute_seed_relevance(layers, 1, labels, mask, 4, 4, 2,
                                      np.empty(0, dtype=np.float64), False)
        assert rel[0] > 0, (
            "a tree touching its only neighbour diagonally was scored as "
            "having no neighbours at all"
        )
        assert rel[1] > 0

    def test_relevance_is_size_times_min_contrast(self):
        """vminsc(s) = vsize(s) * min contrast, per the paper."""
        from plgeoadaptels.sicle import _compute_seed_relevance
        layers, labels, mask = self._diagonal_pair()
        rel = _compute_seed_relevance(layers, 1, labels, mask, 4, 4, 2,
                                      np.empty(0, dtype=np.float64), False)
        # 4 of 8 labelled pixels each, contrast 50 between the two means.
        assert abs(rel[0] - 0.5 * 50.0) < 1e-9
        assert abs(rel[1] - 0.5 * 50.0) < 1e-9

    def test_isolated_tree_scores_zero(self):
        """With no neighbour at all the sentinel must not leak into the score."""
        from plgeoadaptels.sicle import _compute_seed_relevance
        labels = np.full(16, -1, dtype=np.int32)
        labels[0] = 0
        layers = np.zeros((1, 16), dtype=np.float64)
        mask = np.zeros(16, dtype=np.uint8)
        rel = _compute_seed_relevance(layers, 1, labels, mask, 4, 4, 1,
                                      np.empty(0, dtype=np.float64), False)
        assert rel[0] == 0.0
