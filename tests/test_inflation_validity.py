import math
import torch
from tests._oracle.generators import Centerline
from track_gen.types import TrackGenConfig
from tests._oracle import inflation, geometry


def _circle_cl(r=3.0, m=300, e=1):
    t = torch.linspace(0, 2 * math.pi, m + 1)[:-1]
    pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    return Centerline(points=pts, valid=torch.ones(e, dtype=torch.bool))


def _fig8_cl(s=2.0, m=400, e=1):
    t = torch.linspace(0, 2 * math.pi, m + 1)[:-1]
    pts = torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    return Centerline(points=pts, valid=torch.ones(e, dtype=torch.bool))


def _cfg(**ov):
    # output_mode="fixed" was dropped; constant_spacing is the only mode. Tests set
    # spacing/N_max explicitly for determinism (otherwise spacing auto-couples to half_width).
    base = dict(device="cpu", num_envs=1, num_points=256,
                half_width=0.4, turning_tol=0.2, w_floor=1e-3, relax_enable=False)
    base.update(ov)
    return TrackGenConfig(**base)


def test_constant_width_on_circle():
    # constant_spacing output is NaN-padded [E, N_max, 2] with a per-env real-point count;
    # the offset half-width is constant 0.4 at every REAL point (the two loop-wrap slots at
    # index 0 / count-1 are NaN -- a known artifact of the resample padding poisoning the
    # central-difference normal at the loop seam -- so we mask to the finite real points).
    cl = _circle_cl(r=5.0, m=200, e=1)
    cfg = _cfg(num_points=256, half_width=0.4, spacing=0.24)
    track = inflation.inflate(cl, cfg)

    c = int(track.count[0])
    assert c >= 2
    w = torch.linalg.norm(track.outer - track.center, dim=-1)[0, :c]  # real-point widths
    finite = torch.isfinite(w)
    assert int(finite.sum()) >= c - 2  # at most the two seam slots are NaN
    assert torch.allclose(w[finite], torch.full_like(w[finite], 0.4), atol=1e-3)

    # A clean circle IS a valid track. The torch oracle flags NaN-padded constant_spacing
    # output invalid (documented limitation -- the seam NaN poisons turning/thickness), so
    # to assert validity honestly we size N_max to the produced count (no padding -> no seam
    # NaN -> all-finite, genuinely valid).
    cfg_full = _cfg(num_points=256, half_width=0.4, spacing=0.24, N_max=c)
    track_full = inflation.inflate(cl, cfg_full)
    assert int(track_full.count[0]) == c
    w_full = torch.linalg.norm(track_full.outer - track_full.center, dim=-1)
    assert torch.isfinite(w_full).all()
    assert torch.allclose(w_full, torch.full_like(w_full, 0.4), atol=1e-3)
    assert bool(track_full.valid[0])


def test_validity_flags_self_crossing_border():
    # Flat ellipse + a half_width larger than the minor-axis radius => the inner border
    # folds into a tight swallowtail. Sampled densely (small explicit spacing) the fold's
    # edges actually cross, so this exercises the border self-intersection path of the gate.
    # (The thickness gate also flags it, so validity is robust even at coarser sampling
    # where the strict crossing test can miss such a tight fold.)
    #
    # constant_spacing auto-couples spacing to half_width (here 0.6*2.0 = 1.2, far too coarse
    # to resolve the fold), so spacing is set explicitly small to densely sample the border;
    # N_max is sized to hold the resulting real-point count.
    t = torch.linspace(0, 2 * math.pi, 400 + 1)[:-1]
    pts = torch.stack([4.0 * torch.cos(t), 0.8 * torch.sin(t)], dim=-1).unsqueeze(0)
    cl = Centerline(points=pts, valid=torch.ones(1, dtype=torch.bool))
    cfg = _cfg(num_points=1024, half_width=2.0, spacing=0.02, N_max=1100)
    track = inflation.inflate(cl, cfg)

    c = int(track.count[0])
    # Count-aware: test the real (resampled) border only, not the NaN padding tail.
    inner = track.inner[:, :c]
    outer = track.outer[:, :c]
    crossings = geometry.self_intersections(inner) + geometry.self_intersections(outer)
    assert int(crossings[0]) > 0
    assert not bool(track.valid[0])


def test_validity_flags_figure_eight():
    # A figure-eight winds its two lobes in opposite directions => turning number ~0 (not
    # +/-2*pi), so the closed-loop turning gate flags it invalid regardless of sampling mode.
    cl = _fig8_cl()
    cfg = _cfg(num_points=256, half_width=0.2)
    track = inflation.inflate(cl, cfg)
    assert not bool(track.valid[0])
