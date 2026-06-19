import math
import torch
from tests._oracle.geometry import self_intersections


def _circle(n=64, r=1.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _figure_eight(n=200, s=1.0):
    # Phase-offset so the self-crossing (at the origin, where sin t = 0, i.e. t = 0 and pi)
    # falls BETWEEN samples -> a genuine transversal edge crossing. Without the offset the
    # lemniscate lands its crossing exactly on the coincident vertices t=0 and t=pi, a
    # degenerate vertex-touch (not a proper crossing -- f32 AND f64 agree it's 0). Bezier-
    # sampled tracks never produce coincident samples, so the transversal case is the realistic one.
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1] + 0.123
    return torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1)


def test_self_intersections_convex_is_zero():
    poly = _circle().unsqueeze(0)  # [1,N,2]
    assert int(self_intersections(poly)[0]) == 0


def test_self_intersections_figure_eight_is_positive():
    poly = _figure_eight().unsqueeze(0)
    assert int(self_intersections(poly)[0]) >= 1


def test_self_intersections_batched():
    poly = torch.stack([_circle(n=65), _figure_eight(n=65)], dim=0)  # [2,65,2] (odd: transversal crossing)
    out = self_intersections(poly)
    assert out.shape == (2,)
    assert int(out[0]) == 0 and int(out[1]) >= 1
