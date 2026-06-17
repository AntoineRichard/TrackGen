import math
import pytest
import torch

pytest.importorskip("warp")
if not torch.cuda.is_available():
    pytest.skip("Warp separation path requires CUDA", allow_module_level=True)

from track_gen import relaxation, geometry
from track_gen.types import TrackGenConfig
from track_gen import warp_relax


def _star(n=256, r0=1.0, amp=0.6, k=7, phase=0.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1] + phase
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_warp_separation_matches_torch():
    torch.manual_seed(0)
    E, N, hw = 8, 256, 0.05
    c = (torch.randn(E, N, 2) * 0.5).cuda()
    cfg = TrackGenConfig(device="cuda", num_envs=E, num_points=N, half_width=hw, relax_margin=0.15)
    band = relaxation._band(c, cfg)
    D = 2 * hw; margin = 0.15; target = D * (1 + margin)
    circ = geometry.circ_index_dist(N, c.device)
    mask = circ[None] > band.view(E, 1, 1)
    ref = relaxation._separation_disp(c, mask, D, margin)        # torch pairwise
    got = warp_relax.separation_disp(c, band, target)           # fused warp kernel
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max().item()


def test_warp_xpbd_matches_torch_solve():
    E, N, hw = 6, 256, 0.05
    c0 = torch.stack([_star(N, phase=p) for p in torch.linspace(0, 1.0, E)], 0).cuda()
    base = dict(device="cuda", num_envs=E, num_points=N, half_width=hw,
                relax_iters=150, relax_bend_relax=1.5, relax_margin=0.15)
    out_torch = relaxation.relax(c0, TrackGenConfig(**base, relax_use_warp=False))
    out_warp = relaxation.relax(c0, TrackGenConfig(**base, relax_use_warp=True))
    # Same dynamics, separation differs only at ~1e-5/iter -> trajectories stay close.
    assert torch.allclose(out_torch, out_warp, atol=1e-2)
    band = relaxation._band(c0, TrackGenConfig(**base))
    assert float(geometry.thickness(out_warp, band).min()) >= 0.95 * hw
