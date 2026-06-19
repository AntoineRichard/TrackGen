import torch

from tests._oracle.geometry import polygon_area


def test_unit_square_ccw_is_plus_one():
    sq = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    area = polygon_area(sq)
    assert area.shape == (1,)
    assert torch.allclose(area, torch.tensor([1.0]), atol=1e-6)


def test_unit_square_cw_is_minus_one():
    sq = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])
    area = polygon_area(sq)
    assert torch.allclose(area, torch.tensor([-1.0]), atol=1e-6)


def test_batched_mixed_orientation():
    ccw = [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]  # area +6
    cw = [[0.0, 0.0], [0.0, 3.0], [2.0, 3.0], [2.0, 0.0]]  # area -6
    pts = torch.tensor([ccw, cw])
    area = polygon_area(pts)
    assert torch.allclose(area, torch.tensor([6.0, -6.0]), atol=1e-6)
