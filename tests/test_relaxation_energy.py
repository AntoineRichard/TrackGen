import math
import torch
from track_gen.types import TrackGenConfig
from tests._oracle import relaxation, geometry


def _star(n=256, r0=1.0, amp=0.5, k=6):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_energy_backend_raises_thickness():
    c0 = _star().unsqueeze(0)
    cfg = TrackGenConfig(device="cpu", num_envs=1, num_points=256, relax_solver="energy",
                         half_width=0.05, energy_steps=400)
    out = relaxation.relax(c0, cfg)
    band = relaxation._band(c0, cfg)
    assert out.shape == c0.shape
    # Soft solver: thickness should improve substantially over the init.
    assert float(geometry.thickness(out, band)[0]) > float(geometry.thickness(c0, band)[0])


def test_energy_backend_deterministic():
    c0 = _star().unsqueeze(0)
    cfg = TrackGenConfig(device="cpu", num_envs=1, num_points=256, relax_solver="energy",
                         half_width=0.05, energy_steps=100)
    torch.manual_seed(0); a = relaxation.relax(c0, cfg)
    torch.manual_seed(0); b = relaxation.relax(c0, cfg)
    assert torch.allclose(a, b)
