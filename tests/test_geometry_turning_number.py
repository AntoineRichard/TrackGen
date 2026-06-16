import math

import torch

from track_gen.geometry import turning_number


def _regular_polygon(n: int, radius: float = 1.0) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_convex_ccw_polygon_is_plus_two_pi():
    pts = _regular_polygon(8)  # CCW by construction (increasing angle)
    tn = turning_number(pts)
    assert tn.shape == (1,)
    assert torch.isclose(tn, torch.tensor([2.0 * math.pi], dtype=torch.float64), atol=1e-4)


def test_convex_cw_polygon_is_minus_two_pi():
    pts = _regular_polygon(8).flip(dims=[1])  # reverse -> clockwise
    tn = turning_number(pts)
    assert torch.isclose(tn, torch.tensor([-2.0 * math.pi], dtype=torch.float64), atol=1e-4)


def test_figure_eight_turns_cancel_to_zero():
    pts = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]], dtype=torch.float64
    )
    tn = turning_number(pts)
    assert torch.isclose(tn, torch.tensor([0.0], dtype=torch.float64), atol=1e-4)
