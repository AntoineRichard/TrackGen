import torch

from track_gen import geometry
from track_gen.geometry import ccw_sort, polygon_area


def test_scramble_is_reordered_to_a_simple_polygon():
    scrambled = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    out = ccw_sort(scrambled)
    assert out.shape == scrambled.shape
    assert torch.isclose(polygon_area(out).abs(), torch.tensor([1.0]), atol=1e-6)


def test_sorted_output_has_monotone_angles_around_centroid():
    pts = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, -1.0]]]
    )
    out = ccw_sort(pts)
    centroid = out.mean(dim=1, keepdim=True)
    d = out - centroid
    ang = torch.atan2(d[..., 0], d[..., 1])  # reproduce the ported convention
    diffs = ang[0, 1:] - ang[0, :-1]
    assert (diffs >= -1e-6).all()


def test_output_is_a_permutation_of_the_input():
    pts = torch.tensor([[[0.3, 0.9], [-0.5, 0.2], [0.7, -0.4], [0.1, 0.6]]])
    out = ccw_sort(pts)
    assert torch.allclose(out.sum(dim=1), pts.sum(dim=1), atol=1e-6)


def test_ccw_sort_count_sorts_kept_and_nans_tail():
    # P=6 corners, count=4: first 4 sorted about THEIR centroid; last 2 -> NaN.
    pts = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [5., 5.], [6., 6.]]])  # [1,6,2]
    count = torch.tensor([4])
    out = geometry.ccw_sort_count(pts, count)

    assert out.shape == (1, 6, 2)
    assert torch.isnan(out[0, 4:]).all()       # pruned tail
    assert torch.isfinite(out[0, :4]).all()    # kept rows finite

    # kept rows are a permutation of the first 4 inputs
    kept_in = pts[0, :4].sort(dim=0).values
    kept_out = out[0, :4].sort(dim=0).values
    assert torch.allclose(kept_out, kept_in)

    # kept rows are angularly monotone about their own centroid
    c = out[0, :4].mean(dim=0)
    d = out[0, :4] - c
    ang = torch.arctan2(d[:, 0], d[:, 1])
    assert (ang[1:] - ang[:-1] >= -1e-6).all()


def test_ccw_sort_count_full_count_matches_ccw_sort():
    # count == P sorts everything; equals plain ccw_sort (no pruned tail).
    pts = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]])  # [1,4,2]
    count = torch.tensor([4])
    out = geometry.ccw_sort_count(pts, count)
    ref = geometry.ccw_sort(pts)
    assert torch.equal(out, ref)
