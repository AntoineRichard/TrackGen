import math
import torch
from track_gen._src.types import TrackGenConfig
from tests._oracle import relaxation, geometry


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


def _dkappa_rms(pts):
    """RMS of adjacent Menger-curvature differences over one env's closed loop."""
    p = pts[0]
    pp, pn = torch.roll(p, 1, 0), torch.roll(p, -1, 0)
    a, b, c = p - pp, pn - p, pn - pp
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    area = 0.5 * cross.abs()
    la, lb, lc = (x.norm(dim=-1) for x in (a, b, c))
    kappa = 4.0 * area / (la * lb * lc).clamp_min(1e-12)
    dk = torch.roll(kappa, -1) - kappa
    return float((dk ** 2).mean().sqrt())


def test_smoothing_tail_reduces_curvature_noise():
    # The Taubin+polish tail (defaults 5/10) should smooth curvature noise while
    # keeping the thickness target, versus the tail disabled (0/0).
    c0 = _star(n=256, r0=1.0, amp=0.6, k=7).unsqueeze(0)
    cfg_on = _cfg(half_width=0.05)
    cfg_off = _cfg(half_width=0.05, relax_smooth_passes=0, relax_smooth_spacing_iters=0)
    out_on = relaxation.relax(c0, cfg_on)
    out_off = relaxation.relax(c0, cfg_off)
    band = relaxation._band(c0, cfg_on)
    # (a) smoothed result still meets the thickness target
    th_on = geometry.thickness(out_on, band)
    assert float(th_on[0]) >= 0.98 * cfg_on.half_width
    # (b) curvature-difference RMS strictly lower with the tail on
    assert _dkappa_rms(out_on) < _dkappa_rms(out_off)


def test_xpbd_pushes_apart_near_touch():
    # Two near-touching strands (a pinched oval): separation-limited, not curvature.
    import math
    n = 256
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    x = torch.cos(t)
    y = 0.15 * torch.sin(t)          # very flat oval -> top/bottom strands ~0.3 apart
    c0 = torch.stack([x, y], dim=-1).unsqueeze(0)
    cfg = _cfg(half_width=0.05, relax_iters=300)
    out = relaxation.relax(c0, cfg)
    band = relaxation._band(c0, cfg)
    assert float(geometry.separation_min(out, band)[0]) >= 2 * 0.05 * 0.98
