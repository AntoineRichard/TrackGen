import math
import pytest
import torch

pytest.importorskip("warp")
if not torch.cuda.is_available():
    pytest.skip("Warp separation path requires CUDA", allow_module_level=True)

from tests._oracle import relaxation, geometry
from tests._warp_compare import xpbd_solve
from track_gen._src.types import TrackGenConfig


def _star(n=256, r0=1.0, amp=0.6, k=7, phase=0.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1] + phase
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_warp_xpbd_matches_torch_solve():
    E, N, hw = 6, 256, 0.05
    c0 = torch.stack([_star(N, phase=p) for p in torch.linspace(0, 1.0, E)], 0).cuda()
    base = dict(device="cuda", num_envs=E, num_points=N, half_width=hw,
                relax_iters=150, relax_bend_relax=1.5, relax_margin=0.15)
    out_torch = relaxation.relax(c0, TrackGenConfig(**base, relax_use_warp=False))
    out_warp = xpbd_solve(c0, relaxation._band(c0, TrackGenConfig(**base)),
                          geometry.perimeter(c0) / N, TrackGenConfig(**base))
    # Resample to N (relaxation.relax returns resampled; xpbd_solve returns raw).
    out_warp_rs, _ = geometry.arc_length_resample(out_warp, num=N)
    # Same dynamics, separation differs only at ~1e-5/iter -> trajectories stay close.
    assert torch.allclose(out_torch, out_warp_rs, atol=1e-2)
    band = relaxation._band(c0, TrackGenConfig(**base))
    assert float(geometry.thickness(out_warp_rs, band).min()) >= 0.95 * hw
