import math
import torch
from track_gen.generators import Centerline
from track_gen.types import TrackGenConfig
from track_gen import inflation, geometry


def _circle_cl(r=3.0, m=300, e=1):
    t = torch.linspace(0, 2 * math.pi, m + 1)[:-1]
    pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    return Centerline(points=pts, valid=torch.ones(e, dtype=torch.bool))


def _fig8_cl(s=2.0, m=400, e=1):
    t = torch.linspace(0, 2 * math.pi, m + 1)[:-1]
    pts = torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    return Centerline(points=pts, valid=torch.ones(e, dtype=torch.bool))


def _cfg(**ov):
    base = dict(device="cpu", num_envs=1, output_mode="fixed", num_points=256,
                half_width=0.4, turning_tol=0.2, w_floor=1e-3, relax_enable=False)
    base.update(ov)
    return TrackGenConfig(**base)


def test_constant_width_on_circle():
    cl = _circle_cl(r=5.0, m=200, e=1)
    cfg = _cfg(num_points=256, half_width=0.4)
    track = inflation.inflate(cl, cfg)
    w = torch.linalg.norm(track.outer - track.center, dim=-1)
    assert torch.allclose(w, torch.full_like(w, 0.4), atol=1e-3)
    assert bool(track.valid[0])


def test_validity_flags_self_crossing_border():
    # Flat ellipse + a half_width larger than the minor-axis radius => the inner border
    # folds into a tight swallowtail. Sampled densely (num_points=1024) the fold's edges
    # actually cross, so this exercises the border self-intersection path of the gate.
    # (The thickness gate also flags it, so validity is robust even at coarser sampling
    # where the strict crossing test can miss such a tight fold.)
    t = torch.linspace(0, 2 * math.pi, 400 + 1)[:-1]
    pts = torch.stack([4.0 * torch.cos(t), 0.8 * torch.sin(t)], dim=-1).unsqueeze(0)
    cl = Centerline(points=pts, valid=torch.ones(1, dtype=torch.bool))
    cfg = _cfg(num_points=1024, half_width=2.0)
    track = inflation.inflate(cl, cfg)
    crossings = geometry.self_intersections(track.inner) + geometry.self_intersections(track.outer)
    assert int(crossings[0]) > 0
    assert not bool(track.valid[0])


def test_validity_flags_figure_eight():
    cl = _fig8_cl()
    cfg = _cfg(num_points=256, half_width=0.2)
    track = inflation.inflate(cl, cfg)
    assert not bool(track.valid[0])
