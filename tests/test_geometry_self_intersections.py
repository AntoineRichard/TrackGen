import math
import torch
from track_gen.geometry import self_intersections


def _circle(n=64, r=1.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _figure_eight(n=200, s=1.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1)


def test_self_intersections_convex_is_zero():
    poly = _circle().unsqueeze(0)  # [1,N,2]
    assert int(self_intersections(poly)[0]) == 0


def test_self_intersections_figure_eight_is_positive():
    poly = _figure_eight().unsqueeze(0)
    assert int(self_intersections(poly)[0]) >= 1


def test_self_intersections_batched():
    poly = torch.stack([_circle(), _figure_eight(n=64)], dim=0)  # [2,64,2]
    out = self_intersections(poly)
    assert out.shape == (2,)
    assert int(out[0]) == 0 and int(out[1]) >= 1
