import math
import torch
from track_gen.types import TrackGenConfig
from tests._oracle import relaxation, geometry


def _flat_oval(n=256):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([torch.cos(t), 0.25 * torch.sin(t)], dim=-1).unsqueeze(0)


def test_tp_sobolev_backend_runs_and_increases_separation():
    c0 = _flat_oval()
    cfg = TrackGenConfig(device="cpu", num_envs=1, num_points=256, relax_solver="tp_sobolev",
                         half_width=0.05, tp_iters=60, tp_tau=0.7)
    band = relaxation._band(c0, cfg)
    out = relaxation.relax(c0, cfg)
    assert out.shape == c0.shape
    assert torch.isfinite(out).all()
    assert float(geometry.separation_min(out, band)[0]) > float(geometry.separation_min(c0, band)[0])
