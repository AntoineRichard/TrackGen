import math
import torch
from track_gen.types import TrackGenConfig
from tests._oracle import relaxation


def _star(n=256, r0=1.0, amp=0.6, k=7, phase=0.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1] + phase
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_chunked_equals_unchunked():
    c0 = torch.stack([_star(phase=p) for p in torch.linspace(0, 1.0, 5)], dim=0)  # [5,256,2]
    base = dict(device="cpu", num_envs=5, num_points=256, relax_solver="xpbd",
                half_width=0.05, relax_iters=150)
    full = relaxation.relax(c0, TrackGenConfig(**base, relax_chunk_size=None))
    chunked = relaxation.relax(c0, TrackGenConfig(**base, relax_chunk_size=2))
    assert torch.allclose(full, chunked, atol=1e-5)
