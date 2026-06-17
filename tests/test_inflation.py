import math

import pytest
import torch

from track_gen.generators import Centerline
from track_gen.types import Track, TrackGenConfig
from track_gen import inflation
from track_gen import geometry


def make_circle_centerline(radius=2.0, m=200, e=1, center=(0.0, 0.0), device="cpu"):
    """Build a Centerline whose points lie exactly on a circle (closed, no NaN padding)."""
    theta = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]  # drop duplicate endpoint
    x = center[0] + radius * torch.cos(theta)
    y = center[1] + radius * torch.sin(theta)
    pts = torch.stack([x, y], dim=-1)  # [m, 2]
    pts = pts.unsqueeze(0).expand(e, m, 2).contiguous()  # [e, m, 2]
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def fixed_config(num_points=128, device="cpu", **overrides):
    """A TrackGenConfig in fixed output mode."""
    kwargs = dict(
        device=device,
        num_envs=1,
        output_mode="fixed",
        num_points=num_points,
    )
    kwargs.update(overrides)
    return TrackGenConfig(**kwargs)


def test_resample_stage_circle_is_arc_uniform_and_on_circle():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=200, e=3)
    cfg = fixed_config(num_points=128, num_envs=3)

    res = inflation._resample_stage(cl, cfg)

    assert res.center.shape == (3, 128, 2)
    assert torch.equal(res.count, torch.full((3,), 128, dtype=res.count.dtype))
    r = torch.linalg.norm(res.center, dim=-1)  # [E, N]
    assert torch.allclose(r, torch.full_like(r, radius), atol=1e-3)
    seg = torch.linalg.norm(torch.diff(res.center, dim=1, append=res.center[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-3


def test_frame_curvature_orthonormal_and_circle_kappa():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=500, e=2)
    cfg = fixed_config(num_points=256, num_envs=2)

    res = inflation._resample_stage(cl, cfg)
    T, Nrm, kappa = inflation._frame_curvature_stage(res.center)

    t_norm = torch.linalg.norm(T, dim=-1)  # [E, N]
    assert torch.allclose(t_norm, torch.ones_like(t_norm), atol=1e-4)
    n_norm = torch.linalg.norm(Nrm, dim=-1)
    assert torch.allclose(n_norm, torch.ones_like(n_norm), atol=1e-4)
    dot = (T * Nrm).sum(dim=-1)  # [E, N]
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-4)
    assert torch.allclose(kappa, torch.full_like(kappa, 1.0 / radius), atol=1e-2)


def test_width_bounded_by_w_max_on_circle():
    cl = make_circle_centerline(radius=5.0, m=200, e=1)
    cfg = fixed_config(num_points=256, num_envs=1, half_width=0.4)
    res = inflation._resample_stage(cl, cfg)
    _, _, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    assert w.shape == (1, 256)
    assert torch.all(w <= cfg.half_width + 1e-6)
    assert torch.allclose(w, torch.full_like(w, cfg.half_width), atol=1e-3)


def test_offset_orientation_outer_bigger_inner_smaller():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=4)
    cfg = fixed_config(num_points=256, num_envs=4, half_width=0.5)
    res = inflation._resample_stage(cl, cfg)
    _, Nrm, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)

    outer, inner = inflation._offset_stage(res.center, Nrm, w)

    assert outer.shape == res.center.shape
    assert inner.shape == res.center.shape

    a_outer = geometry.polygon_area(outer).abs()   # [E]
    a_center = geometry.polygon_area(res.center).abs()
    a_inner = geometry.polygon_area(inner).abs()

    assert torch.all(a_outer > a_center)
    assert torch.all(a_center > a_inner)
    w_scalar = cfg.half_width
    assert torch.allclose(a_outer, torch.full_like(a_outer, math.pi * (radius + w_scalar) ** 2), atol=1e-1)
    assert torch.allclose(a_inner, torch.full_like(a_inner, math.pi * (radius - w_scalar) ** 2), atol=1e-1)


def make_figure_eight_centerline(scale=2.0, m=400, e=1, device="cpu"):
    """A self-crossing figure-eight (lemniscate): turning number ~ 0, not +/- 2pi."""
    t = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]
    x = scale * torch.sin(t)
    y = scale * torch.sin(t) * torch.cos(t)
    pts = torch.stack([x, y], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def _run_to_width(cl, cfg):
    res = inflation._resample_stage(cl, cfg)
    _, Nrm, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    return res.center, Nrm, w, res.count


def test_validity_true_for_clean_circle():
    cl = make_circle_centerline(radius=3.0, m=300, e=2)
    cfg = fixed_config(num_points=256, num_envs=2, half_width=0.4,
                       turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert valid.dtype == torch.bool
    assert valid.shape == (2,)
    assert torch.all(valid)


def test_validity_false_for_self_crossing():
    cl = make_figure_eight_centerline(scale=2.0, m=400, e=1)
    cfg = fixed_config(num_points=256, num_envs=1, half_width=0.2,
                       turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert not bool(valid[0])


def test_validity_respects_gen_valid_flag():
    cl = make_circle_centerline(radius=3.0, m=300, e=2)
    cl.valid[1] = False
    cfg = fixed_config(num_points=256, num_envs=2, half_width=0.4,
                       turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert bool(valid[0]) is True
    assert bool(valid[1]) is False


def test_inflate_fixed_mode_full_track():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=3)
    cfg = fixed_config(num_points=128, num_envs=3, half_width=0.4,
                       turning_tol=0.2, w_floor=1e-3)

    track = inflation.inflate(cl, cfg)

    assert isinstance(track, Track)
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (3, 128, 2)
    assert track.arclen.shape == (3, 128)
    assert track.length.shape == (3,)
    assert track.valid.shape == (3,)
    assert track.count.shape == (3,)

    assert torch.equal(track.count, torch.full((3,), 128, dtype=track.count.dtype))
    assert torch.all(track.valid)
    assert torch.isfinite(track.center).all()
    assert torch.isfinite(track.outer).all()
    assert torch.isfinite(track.inner).all()

    assert torch.allclose(track.arclen[:, 0], torch.zeros(3), atol=1e-6)
    assert torch.all(track.arclen[:, 1:] - track.arclen[:, :-1] >= -1e-6)
    assert torch.allclose(track.length, torch.full((3,), 2 * math.pi * radius), atol=1e-1)


def test_inflate_constant_spacing_mode_padding_and_wrap_length():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=1)
    # Circumference ~ 18.85; spacing 0.5 -> ~38 real points, padded to N_max=128.
    cfg = TrackGenConfig(
        device="cpu", num_envs=1,
        output_mode="constant_spacing", spacing=0.5, N_max=128,
        half_width=0.3,
        turning_tol=0.2, w_floor=1e-3,
    )

    track = inflation.inflate(cl, cfg)

    assert track.center.shape == (1, 128, 2)
    c = int(track.count[0].item())
    assert 0 < c <= 128
    assert torch.isfinite(track.center[0, :c]).all()
    if c < 128:
        assert torch.isnan(track.center[0, c:]).all()
    assert abs(c - round(2 * math.pi * radius / 0.5)) <= 2
    # The closing wrap segment must be included -> total length ~ circumference,
    # not short by one segment.
    assert torch.allclose(track.length, torch.full((1,), 2 * math.pi * radius), atol=0.6)
