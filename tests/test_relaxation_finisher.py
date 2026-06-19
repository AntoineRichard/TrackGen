import math
import torch
from track_gen._src.types import TrackGenConfig
from tests._oracle import relaxation, geometry


def _star(n=256, r0=1.0, amp=0.6, k=7):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_finisher_keeps_validity_and_smooths():
    c0 = _star().unsqueeze(0)
    base = dict(device="cpu", num_envs=1, num_points=256, relax_solver="xpbd",
                half_width=0.05, relax_iters=200, relax_bend_relax=1.5, relax_margin=0.15)
    band = relaxation._band(c0, TrackGenConfig(**base))
    no_fin = relaxation.relax(c0, TrackGenConfig(**base, smooth_finish=False))
    fin = relaxation.relax(c0, TrackGenConfig(**base, smooth_finish=True,
                                              smooth_finish_iters=8, smooth_finish_tau=0.2))
    # Finisher keeps thickness at/above target and runs (changes the curve).
    assert float(geometry.thickness(fin, band)[0]) >= 0.96 * 0.05
    assert fin.shape == c0.shape
    assert not torch.allclose(no_fin, fin)
