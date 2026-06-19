import math

import torch

from tests._oracle.geometry import menger_curvature


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_circle_curvature_is_one_over_r():
    r = 2.5
    pts = _circle(256, r)
    kappa = menger_curvature(pts)
    assert kappa.shape == (1, 256)
    expected = torch.full_like(kappa, 1.0 / r)
    assert torch.allclose(kappa, expected, atol=1e-3)
    assert (kappa >= 0).all()


def test_straight_line_curvature_is_zero():
    xs = torch.linspace(0.0, 9.0, 10, dtype=torch.float64)
    pts = torch.stack([xs, torch.zeros_like(xs)], dim=-1).unsqueeze(0)
    kappa = menger_curvature(pts)
    assert torch.allclose(kappa[0, 1:-1], torch.zeros(8, dtype=torch.float64), atol=1e-6)
    assert (kappa >= 0).all()


def test_coincident_points_do_not_produce_nan():
    pts = torch.zeros((1, 5, 2), dtype=torch.float64)
    kappa = menger_curvature(pts)
    assert torch.isfinite(kappa).all()
