import math

import torch

from track_gen.geometry import nearest_nonadjacent_distance


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_circle_min_nonadjacent_distance_is_positive():
    pts = _circle(64, 1.0)
    d = nearest_nonadjacent_distance(pts, band=2)
    assert d.shape == (1, 64)
    assert torch.isfinite(d).all()
    assert (d > 0).all()


def test_immediate_neighbors_are_masked_not_self():
    pts = _circle(32, 1.0)
    d_band1 = nearest_nonadjacent_distance(pts, band=1)
    two_step = 2.0 * 1.0 * math.sin(2.0 * math.pi / 32 * 2 / 2)
    assert torch.allclose(d_band1, torch.full_like(d_band1, two_step), atol=1e-3)
    assert (d_band1 > 1e-6).all()


def test_decimation_path_runs_and_matches_full_resolution_shape():
    pts = _circle(80, 2.0)
    d_full = nearest_nonadjacent_distance(pts, band=2)
    d_dec = nearest_nonadjacent_distance(pts, band=2, decimation=40)
    assert d_dec.shape == d_full.shape == (1, 80)
    assert torch.isfinite(d_dec).all()
    assert (d_dec > 0).all()
    # Decimation is an intentional approximation: ballpark agreement only.
    assert torch.allclose(d_dec, d_full, atol=0.2)
