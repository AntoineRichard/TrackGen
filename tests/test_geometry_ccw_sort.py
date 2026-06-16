import torch

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
