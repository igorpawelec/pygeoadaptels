"""Sliver absorption in enforce_connectivity must not change its answer when
it stops touching the whole raster per fragment.

The rewrite works inside each fragment's bounding box padded by one pixel.
That padding is the entire correctness question: a 4-connected ring around a
fragment can reach outside the box that find_objects reports for its label,
and a version that forgot to pad would find no neighbour there and keep the
sliver as its own adaptel. Two guards:

  * a from-scratch reference that does the absorption on the full raster,
    the way the slow version did, compared on random label fields with many
    slivers, across several min_size values, with nodata and raster edges;
  * a hand-built case where every labelled neighbour of a fragment lies
    outside its label's box, so the unpadded implementation gives a wrong
    answer rather than a slow one.

One rule of the real function that a naive reference gets wrong: a label
that is a single connected component is never a "fragment", however small,
and keeps its id. enforce_connectivity repairs splits; it does not delete
small adaptels.

Run: pytest tests/test_enforce_connectivity_slivers.py -v
"""
import numpy as np
import pytest

ndimage = pytest.importorskip("scipy.ndimage")

from pygeoadaptels import enforce_connectivity  # noqa: E402

STRUCT = ndimage.generate_binary_structure(2, 1)


def reference(labels, min_size):
    """The previous algorithm spelled out on the full raster, including the
    rule that a single-component label is kept whole."""
    labels = np.asarray(labels)
    out = np.full(labels.shape, -1, np.int32)
    valid = labels >= 0
    new_id = 0
    small = []
    for lab in np.unique(labels[valid]):
        cc, n = ndimage.label(labels == lab, structure=STRUCT)
        if n == 1:
            out[cc == 1] = new_id
            new_id += 1
            continue
        for c in range(1, n + 1):
            frag = cc == c
            if min_size and frag.sum() <= min_size:
                small.append(frag)
                continue
            out[frag] = new_id
            new_id += 1
    for frag in small:
        ring = ndimage.binary_dilation(frag, structure=STRUCT) & ~frag
        nb = out[ring & (out >= 0)]
        if nb.size:
            out[frag] = np.bincount(nb).argmax()
        else:
            out[frag] = new_id
            new_id += 1
    out[~valid] = labels[~valid]
    return out


def canonical(lab):
    """Relabel by first appearance so two labelings can be compared."""
    lab = np.asarray(lab).ravel()
    out = np.full(lab.shape, -1, np.int64)
    seen = {}
    for i, v in enumerate(lab):
        if v >= 0:
            out[i] = seen.setdefault(int(v), len(seen))
    return out


def field(seed, shape=(40, 40), n_labels=12, nodata_frac=0.05):
    """Blobby random labels with plenty of slivers, some nodata, some at edges."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, n_labels, size=shape).astype(np.int32)
    base = ndimage.median_filter(base, size=3)
    base[rng.random(shape) < 0.08] = rng.integers(0, n_labels)
    base[rng.random(shape) < nodata_frac] = -1
    return base


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("min_size", [1, 2, 4, 9])
def test_matches_full_raster_reference(seed, min_size):
    lab = field(seed)
    got, _ = enforce_connectivity(lab, min_size=min_size)
    want = reference(lab, min_size)
    assert np.array_equal(canonical(got), canonical(want)), (
        f"seed={seed} min_size={min_size}: absorption differs from the "
        f"full-raster reference")
    assert np.array_equal(got < 0, lab < 0), "nodata must pass through"


def test_neighbour_outside_the_label_box_is_still_found():
    """Label 2 is two pixels in one column with nodata between them, so its
    box is that column and nothing else. Each fragment's only labelled
    neighbour is to the side, outside the box. Padding the box by one pixel
    finds it; not padding it does not, and both fragments would wrongly come
    out as adaptels of their own."""
    lab = np.zeros((6, 6), np.int32)
    lab[0, 0] = 2
    lab[1, 0] = -1
    lab[2, 0] = 2
    got, n = enforce_connectivity(lab, min_size=1)
    assert n == 1, f"expected the two slivers absorbed into label 0, got n={n}"
    assert got[0, 0] == got[2, 0] == got[0, 1], "slivers must join label 0"
    assert got[1, 0] == -1, "nodata must pass through"


def test_single_component_label_is_kept_whole():
    """A one-pixel label that is not a fragment of anything is not a sliver."""
    lab = np.zeros((5, 5), np.int32)
    lab[2, 2] = 7
    got, n = enforce_connectivity(lab, min_size=4)
    assert n == 2
    assert got[2, 2] != got[0, 0]


def test_min_size_zero_keeps_every_fragment():
    lab = field(7)
    got, n = enforce_connectivity(lab, min_size=0)
    want = reference(lab, 0)
    assert np.array_equal(canonical(got), canonical(want))
    total = sum(ndimage.label(lab == v, structure=STRUCT)[1]
                for v in np.unique(lab[lab >= 0]))
    assert n == total
