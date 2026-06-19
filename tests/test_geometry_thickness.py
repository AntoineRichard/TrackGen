import math
import torch
from tests._oracle.geometry import (
    perimeter, mean_seg_len, separation_min, curvature_radius_min, thickness,
)


def _circle(n=256, r=2.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0)


def test_perimeter_and_spacing_of_circle():
    c = _circle(n=256, r=2.0)
    assert torch.allclose(perimeter(c), torch.tensor([2 * math.pi * 2.0]), atol=1e-2)
    assert torch.allclose(mean_seg_len(c), perimeter(c) / 256)


def test_curvature_radius_min_of_circle_is_radius():
    c = _circle(n=400, r=2.0)
    assert torch.allclose(curvature_radius_min(c), torch.tensor([2.0]), atol=1e-2)


def test_separation_min_is_index_based():
    # band is an INDEX window (not a distance). With band=4 on a 256-gon of radius 2,
    # the nearest non-excluded pair is at index distance 5: chord = 2r*sin(pi*5/256).
    # (A Euclidean-distance exclusion would mask the whole circle -> inf, so this pins
    # the index-based semantics that the relaxation/validity logic depends on.)
    c = _circle(n=256, r=2.0)
    band = torch.tensor([4])
    sep = separation_min(c, band)
    expected = 2 * 2.0 * math.sin(math.pi * 5 / 256)
    assert sep.shape == (1,)
    assert torch.allclose(sep, torch.tensor([expected]), atol=1e-3)


def test_thickness_of_circle_is_radius():
    # thickness = min(curvature_radius, 0.5*separation_min). For separation/2 to reach
    # the radius, the nearest non-adjacent point must be ~diametrically opposite, which
    # needs a LARGE index band (~N/2). With band=N//2-2 the only non-excluded pairs are
    # near-antipodal -> separation ~ diameter -> thickness ~ radius.
    n = 400
    c = _circle(n=n, r=2.0)
    band = torch.tensor([n // 2 - 2])
    th = thickness(c, band)
    assert torch.allclose(th, torch.tensor([2.0]), atol=2e-2)
