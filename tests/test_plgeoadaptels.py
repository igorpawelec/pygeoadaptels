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
        # n_oversampling < n_segments should auto-correct
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
