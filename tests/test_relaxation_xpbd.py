import math
import torch
from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry


def _star(n=256, r0=1.0, amp=0.6, k=7):
    """A wiggly star: low curvature radius at the spikes (sharp corners)."""
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _cfg(**ov):
    base = dict(device="cpu", num_envs=1, num_points=256, relax_solver="xpbd",
                half_width=0.05, relax_iters=200, relax_bend_relax=1.5, relax_margin=0.15)
    base.update(ov)
    return TrackGenConfig(**base)


def test_xpbd_rounds_sharp_corners_to_thickness_target():
    c0 = _star(n=256, r0=1.0, amp=0.6, k=7).unsqueeze(0)  # [1,256,2]
    cfg = _cfg(half_width=0.05)
    out = relaxation.relax(c0, cfg)
    band = relaxation._band(c0, cfg)
    th = geometry.thickness(out, band)
    assert out.shape == c0.shape
    assert float(th[0]) >= 0.98 * cfg.half_width  # reached thickness target


def test_xpbd_is_deterministic():
    c0 = _star().unsqueeze(0)
    cfg = _cfg()
    a = relaxation.relax(c0, cfg)
    b = relaxation.relax(c0, cfg)
    assert torch.allclose(a, b)


def test_relax_disabled_is_identity():
    c0 = _star().unsqueeze(0)
    cfg = _cfg(relax_enable=False)
    assert torch.allclose(relaxation.relax(c0, cfg), c0)
