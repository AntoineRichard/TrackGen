import math

import pytest
import torch

from tests._oracle.generators import Centerline
from track_gen._src.types import Track, TrackGenConfig
from tests._oracle import inflation
from tests._oracle import geometry


def make_circle_centerline(radius=2.0, m=200, e=1, center=(0.0, 0.0), device="cpu"):
    """Build a Centerline whose points lie exactly on a circle (closed, no NaN padding)."""
    theta = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]  # drop duplicate endpoint
    x = center[0] + radius * torch.cos(theta)
    y = center[1] + radius * torch.sin(theta)
    pts = torch.stack([x, y], dim=-1)  # [m, 2]
    pts = pts.unsqueeze(0).expand(e, m, 2).contiguous()  # [e, m, 2]
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def circle_cs_config(radius, n_max=128, device="cpu", **overrides):
    """A constant_spacing TrackGenConfig tuned so a circle of ``radius`` resamples to
    EXACTLY ``n_max`` real points (count == n_max, no NaN padding).

    The legacy "fixed" output mode (constant point COUNT) was dropped; constant_spacing
    is the only mode. To preserve the intent of the circle stage-tests -- which want a
    fully populated [E, n_max, 2] resample on a known circle -- we pick the spacing that
    divides the circumference into exactly ``n_max`` arc-uniform segments. The resample
    then emits targets i*spacing for i in [0, n_max) (all < circumference) and drops the
    target at i == n_max (== circumference), giving count == n_max with no padding. This
    keeps the oracle's full-tensor turning/thickness metrics NaN-free for the on-circle
    cases while exercising the real constant_spacing path.
    """
    circumference = 2 * math.pi * radius
    kwargs = dict(
        device=device,
        num_envs=1,
        output_mode="constant_spacing",
        spacing=circumference / n_max,
        N_max=n_max,
    )
    kwargs.update(overrides)
    return TrackGenConfig(**kwargs)


def test_resample_stage_circle_is_arc_uniform_and_on_circle():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=200, e=3)
    cfg = circle_cs_config(radius=radius, n_max=128, num_envs=3)

    res = inflation._resample_stage(cl, cfg)

    # constant_spacing tuned to fill exactly N_max real points (no NaN padding).
    assert res.center.shape == (3, 128, 2)
    assert torch.equal(res.count, torch.full((3,), 128, dtype=res.count.dtype))
    r = torch.linalg.norm(res.center, dim=-1)  # [E, N]
    assert torch.allclose(r, torch.full_like(r, radius), atol=1e-3)
    seg = torch.linalg.norm(torch.diff(res.center, dim=1, append=res.center[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-3


def test_frame_curvature_orthonormal_and_circle_kappa():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=500, e=2)
    cfg = circle_cs_config(radius=radius, n_max=256, num_envs=2)

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
    radius = 5.0
    cl = make_circle_centerline(radius=radius, m=200, e=1)
    cfg = circle_cs_config(radius=radius, n_max=256, num_envs=1, half_width=0.4)
    res = inflation._resample_stage(cl, cfg)
    _, _, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    assert w.shape == (1, 256)
    assert torch.all(w <= cfg.half_width + 1e-6)
    assert torch.allclose(w, torch.full_like(w, cfg.half_width), atol=1e-3)


def test_offset_orientation_outer_bigger_inner_smaller():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=4)
    cfg = circle_cs_config(radius=radius, n_max=256, num_envs=4, half_width=0.5)
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


def figure_eight_cs_config(cl, n_max=256, device="cpu", **overrides):
    """A constant_spacing config tuned so the figure-eight resamples to exactly ``n_max``
    real points (count == n_max, no NaN padding), mirroring :func:`circle_cs_config`.

    Spacing = (closed-loop arc length) / n_max so the validity oracle's full-tensor
    turning/thickness metrics see a fully populated, NaN-free polygon -- isolating the
    self-crossing (turning ~ 0 / border-crossing) signal from the padding artifact.
    """
    pts = cl.points[0]
    closed = torch.cat([pts, pts[:1]], dim=0)
    total = torch.linalg.norm(closed[1:] - closed[:-1], dim=-1).sum().item()
    kwargs = dict(
        device=device,
        num_envs=int(cl.points.shape[0]),
        output_mode="constant_spacing",
        spacing=total / n_max,
        N_max=n_max,
    )
    kwargs.update(overrides)
    return TrackGenConfig(**kwargs)


def _run_to_width(cl, cfg):
    res = inflation._resample_stage(cl, cfg)
    _, Nrm, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    return res.center, Nrm, w, res.count


def test_validity_true_for_clean_circle():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=2)
    cfg = circle_cs_config(radius=radius, n_max=256, num_envs=2, half_width=0.4,
                           turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    # Tuned spacing -> count == N_max for every env, so the oracle's full-tensor
    # turning/thickness metrics are NaN-free and the clean circle is valid.
    assert torch.equal(count, torch.full((2,), 256, dtype=count.dtype))
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert valid.dtype == torch.bool
    assert valid.shape == (2,)
    assert torch.all(valid)


def test_validity_false_for_self_crossing():
    cl = make_figure_eight_centerline(scale=2.0, m=400, e=1)
    cfg = figure_eight_cs_config(cl, n_max=256, num_envs=1, half_width=0.2,
                                 turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    # No NaN padding (count == N_max), so the invalidity is the genuine self-crossing
    # signal (turning ~ 0), not the padding artifact.
    assert torch.equal(count, torch.full((1,), 256, dtype=count.dtype))
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert not bool(valid[0])


def test_validity_respects_gen_valid_flag():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=2)
    cl.valid[1] = False
    cfg = circle_cs_config(radius=radius, n_max=256, num_envs=2, half_width=0.4,
                           turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert bool(valid[0]) is True
    assert bool(valid[1]) is False


def test_inflate_constant_spacing_full_track():
    # Repurposed from the dropped "fixed mode full track" test: a fully-populated
    # constant_spacing track (spacing tuned so count == N_max for every env, no NaN
    # padding) is the closest meaningful equivalent of the old constant-COUNT mode.
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=3)
    cfg = circle_cs_config(radius=radius, n_max=128, num_envs=3, half_width=0.4,
                           turning_tol=0.2, w_floor=1e-3)

    track = inflation.inflate(cl, cfg)

    assert isinstance(track, Track)
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (3, 128, 2)
    assert track.arclen.shape == (3, 128)
    assert track.length.shape == (3,)
    assert track.valid.shape == (3,)
    assert track.count.shape == (3,)

    # Tuned spacing -> exactly N_max real points: count == 128 and fully finite.
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
