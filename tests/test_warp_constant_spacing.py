# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Constant-spacing Warp pipeline: per-stage parity (count==N_max matches fixed-N) and
variable-count behaviour, plus the end-to-end smoothness/yield win."""
import math
import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen import warp_pipeline as wpl, warp_relax, geometry  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _circle(N, r, dev):
    t = torch.linspace(0, 2 * math.pi, N + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], -1).to(torch.float32)


def _pad(center, n_max):
    """[E,N,2] -> [E,n_max,2] NaN-padded, count=[N]*E."""
    E, N, _ = center.shape
    buf = torch.full((E, n_max, 2), float("nan"), device=center.device, dtype=torch.float32)
    buf[:, :N] = center
    count = torch.full((E,), N, dtype=torch.int32, device=center.device)
    return buf, count


@pytest.mark.parametrize("dev", DEVS)
def test_constant_spacing_resample_matches_torch_oracle(dev):
    E, N = 3, 300
    src = torch.stack([_circle(N, r, dev) for r in (1.0, 2.5, 4.0)], 0)  # [3,N,2]
    spacing, n_max = 0.5, 128
    out_w, cnt_w = wpl.resample_constant_spacing(src, spacing, n_max)
    out_t, cnt_t = geometry.arc_length_resample(src, spacing=spacing, n_max=n_max)
    assert out_w.shape == (E, n_max, 2)
    assert torch.equal(cnt_w.cpu(), cnt_t.cpu()), f"{cnt_w} vs {cnt_t}"
    for e in range(E):
        c = int(cnt_w[e])
        assert torch.allclose(out_w[e, :c], out_t[e, :c], atol=1e-4)
        assert torch.isnan(out_w[e, c:]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_resample_uniform_count_aware(dev):
    # parity: count==N reproduces the fixed call
    src = torch.stack([_circle(64, 1.0, dev), _circle(64, 2.0, dev)], 0)
    base = wpl.resample_uniform(src, 64)
    buf, cnt = _pad(src, 64)
    out = wpl.resample_uniform(buf, 64, count=cnt)
    assert torch.allclose(out, base, atol=1e-5, equal_nan=True)
    # variable: env0 uses 40 real pts (rest NaN), env1 uses 64; env0 stays ~circle, pad NaN
    buf2 = torch.full((2, 64, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :40] = _circle(40, 1.0, dev); buf2[1, :64] = _circle(64, 2.0, dev)
    cnt2 = torch.tensor([40, 64], dtype=torch.int32, device=dev)
    out2 = wpl.resample_uniform(buf2, 64, count=cnt2)
    r0 = torch.linalg.norm(out2[0, :40], dim=-1)
    assert torch.allclose(r0, torch.ones_like(r0), atol=2e-2)
    assert torch.isnan(out2[0, 40:]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_thickness_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    band = torch.tensor([3, 3], dtype=torch.int32, device=dev)
    base = wpl.thickness(src, band)
    buf, cnt = _pad(src, 80)
    out = wpl.thickness(buf, band, count=cnt)
    assert torch.allclose(out, base, atol=1e-5)
    # variable: env0 real=50 (radius-1 circle) padded to 80; thickness finite, not poisoned by NaN tail
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    th = wpl.thickness(buf2, band, count=cnt2)
    assert th[0] > 0.0 and torch.isfinite(th[0])
    # the radius-1 50-pt circle thickness ~ min(curv_radius≈1, 0.5*sep) — sane, finite
    assert torch.isfinite(th[1])


@pytest.mark.parametrize("dev", DEVS)
def test_self_intersections_count_aware(dev):
    t = torch.linspace(0, 2 * math.pi, 100 + 1, device=dev)[:-1]
    fig8 = torch.stack([torch.sin(t), torch.sin(t) * torch.cos(t)], -1).to(torch.float32)  # 1 crossing
    circle = _circle(100, 1.0, dev)
    src = torch.stack([fig8, circle], 0)
    base = wpl.self_intersections(src)
    buf, cnt = _pad(src, 100)
    out = wpl.self_intersections(buf, count=cnt)
    assert torch.equal(out.cpu(), base.cpu())          # parity: count==N matches fixed
    # variable: env0 = 60-pt figure-eight padded to 100; crossing still detected, NaN tail ignored
    buf2 = torch.full((2, 100, 2), float("nan"), device=dev, dtype=torch.float32)
    t2 = torch.linspace(0, 2 * math.pi, 60 + 1, device=dev)[:-1]
    buf2[0, :60] = torch.stack([torch.sin(t2), torch.sin(t2) * torch.cos(t2)], -1).to(torch.float32)
    buf2[1, :100] = circle
    cnt2 = torch.tensor([60, 100], dtype=torch.int32, device=dev)
    xs = wpl.self_intersections(buf2, count=cnt2)
    assert int(xs[0]) >= 1 and int(xs[1]) == 0          # fig8 crossing; circle simple; pad ignored


@pytest.mark.parametrize("dev", DEVS)
def test_turning_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    base = wpl.turning_number(src)
    buf, cnt = _pad(src, 80)
    out = wpl.turning_number(buf, count=cnt)
    assert torch.allclose(out, base, atol=1e-5)
    # variable: 50-pt circle padded to 80; turning still ~ 2*pi (turning number 1)
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    tn = wpl.turning_number(buf2, count=cnt2)
    assert torch.allclose(tn.abs(), torch.full_like(tn, 2 * math.pi), atol=1e-2)


@pytest.mark.parametrize("dev", DEVS)
def test_frame_curvature_count_aware(dev):
    # --- parity: count==N_max reproduces the fixed call ---
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    T_base, Nrm_base, kap_base = wpl.frame_curvature(src)
    buf, cnt = _pad(src, 80)
    T_out, Nrm_out, kap_out = wpl.frame_curvature(buf, count=cnt)
    assert torch.allclose(T_out, T_base, atol=1e-4)
    assert torch.allclose(Nrm_out, Nrm_base, atol=1e-4)
    assert torch.allclose(kap_out, kap_base, atol=1e-4)

    # --- variable: env0 = 50-pt circle padded to 80, env1 = full 80-pt circle ---
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev)
    buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    T2, Nrm2, kap2 = wpl.frame_curvature(buf2, count=cnt2)

    # Real points (0..49) of env0: normals must be unit-length
    nrm0_real = Nrm2[0, :50]
    lengths = torch.linalg.norm(nrm0_real, dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4), \
        "env0 real normals should be unit-length"

    # Padding points (50..79) of env0: must be NaN
    assert torch.isnan(Nrm2[0, 50:]).all(), "env0 padding normals should be NaN"
    assert torch.isnan(T2[0, 50:]).all(), "env0 padding tangents should be NaN"
    assert torch.isnan(kap2[0, 50:]).all(), "env0 padding kappa should be NaN"


@pytest.mark.parametrize("dev", DEVS)
def test_offset_count_aware(dev):
    # --- parity: count==N_max reproduces the fixed call ---
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    _, Nrm_base, _ = wpl.frame_curvature(src)
    outer_base, inner_base = wpl.offset(src, Nrm_base, 0.1)
    buf, cnt = _pad(src, 80)
    _, Nrm_buf, _ = wpl.frame_curvature(buf, count=cnt)
    outer_out, inner_out = wpl.offset(buf, Nrm_buf, 0.1, count=cnt)
    assert torch.allclose(outer_out, outer_base, atol=1e-4)
    assert torch.allclose(inner_out, inner_base, atol=1e-4)

    # --- variable: env0 = 50-pt circle (r=1) padded to 80, env1 = full 80-pt circle (r=2) ---
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev)
    buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    _, Nrm2, _ = wpl.frame_curvature(buf2, count=cnt2)
    outer2, inner2 = wpl.offset(buf2, Nrm2, 0.1, count=cnt2)

    # Real points (0..49) of env0: outer ~ r=1.1, inner ~ r=0.9  (both finite)
    outer0_real = outer2[0, :50]
    inner0_real = inner2[0, :50]
    assert torch.isfinite(outer0_real).all(), "env0 real outer points should be finite"
    assert torch.isfinite(inner0_real).all(), "env0 real inner points should be finite"
    r_outer = torch.linalg.norm(outer0_real, dim=-1)
    r_inner = torch.linalg.norm(inner0_real, dim=-1)
    assert torch.allclose(r_outer, torch.full_like(r_outer, 1.1), atol=5e-2), \
        f"env0 outer radius ~ 1.1, got {r_outer.mean():.4f}"
    assert torch.allclose(r_inner, torch.full_like(r_inner, 0.9), atol=5e-2), \
        f"env0 inner radius ~ 0.9, got {r_inner.mean():.4f}"

    # Padding points (50..79) of env0: must be NaN
    assert torch.isnan(outer2[0, 50:]).all(), "env0 padding outer should be NaN"
    assert torch.isnan(inner2[0, 50:]).all(), "env0 padding inner should be NaN"


@pytest.mark.parametrize("dev", DEVS)
def test_arclength_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    a0, L0 = wpl._arclength(src)
    buf, cnt = _pad(src, 80)
    a, L = wpl._arclength(buf, count=cnt)
    assert torch.allclose(a[:, :80], a0, atol=1e-4)
    assert torch.allclose(L, L0, atol=1e-4)
    # variable: env0 = 50-pt unit circle padded to 80; length ~ 2*pi, arclen monotonic over real, NaN tail
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    a2, L2 = wpl._arclength(buf2, count=cnt2)
    assert torch.allclose(L2[0], torch.tensor(2 * math.pi, device=dev), atol=1e-2)
    diffs = a2[0, 1:50] - a2[0, :49]
    assert (diffs > 0).all()                 # strictly increasing over real points
    assert torch.isnan(a2[0, 50:]).all()     # NaN padding tail


@pytest.mark.parametrize("dev", DEVS)
def test_xpbd_count_aware(dev):
    src = torch.stack([_circle(96, 1.0, dev), _circle(96, 1.4, dev)], 0)
    band = torch.tensor([3, 2], dtype=torch.int32, device=dev)
    L0 = geometry.perimeter(src) / 96
    cfg = TrackGenConfig(num_envs=2, num_points=96, half_width=0.05, relax_iters=40, device=dev)
    base = warp_relax.xpbd_solve(src, band, L0, cfg)
    buf, cnt = _pad(src, 96)
    out = warp_relax.xpbd_solve(buf, band, L0, cfg, count=cnt)
    assert torch.allclose(out[:, :96], base, atol=1e-5)        # parity: count==N matches fixed
    # variable: env0 = 60-pt circle padded to 96; real points relax & stay finite, padding stays NaN
    buf2 = torch.full((2, 96, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :60] = _circle(60, 1.0, dev); buf2[1, :96] = _circle(96, 1.4, dev)
    cnt2 = torch.tensor([60, 96], dtype=torch.int32, device=dev)
    L02 = torch.tensor([float(geometry.perimeter(buf2[0:1, :60])[0]) / 60,
                        float(geometry.perimeter(buf2[1:2])[0]) / 96], device=dev)
    out2 = warp_relax.xpbd_solve(buf2, band, L02, cfg, count=cnt2)
    assert torch.isfinite(out2[0, :60]).all()                  # real points finite (not NaN-poisoned)
    assert torch.isnan(out2[0, 60:]).all()                     # padding stays NaN
    assert torch.isfinite(out2[1]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_validity_count_aware(dev):
    hw = 0.5
    cfg = TrackGenConfig(num_envs=1, num_points=120, half_width=hw, device=dev)
    src = _circle(120, 5.0, dev).unsqueeze(0)            # radius 5, 1m road -> easily valid
    w = torch.full((1, 120), hw, device=dev)
    gv = torch.ones(1, dtype=torch.int32, device=dev)
    cnt_full = torch.full((1,), 120, dtype=torch.int32, device=dev)
    # build outer/inner with the count-aware offset (count==N here)
    _, Nrm_full, _ = wpl.frame_curvature(src, count=cnt_full)
    o, i = wpl.offset(src, Nrm_full, hw, count=cnt_full)
    v_fixed = wpl.validity(src, w, cnt_full, gv, cfg, o, i)   # count==N (fixed mode)
    assert bool(v_fixed[0]) is True, "full circle radius-5 must be valid"

    # padded version: same circle in a 200-wide buffer, count=120
    buf = torch.full((1, 200, 2), float("nan"), device=dev, dtype=torch.float32)
    buf[0, :120] = src[0]
    w2 = torch.full((1, 200), hw, device=dev)
    cnt = torch.tensor([120], dtype=torch.int32, device=dev)
    _, Nrm2, _ = wpl.frame_curvature(buf, count=cnt)
    o2, i2 = wpl.offset(buf, Nrm2, hw, count=cnt)
    v_pad = wpl.validity(buf, w2, cnt, gv, cfg, o2, i2)
    assert bool(v_pad[0]) is True, "padded circle (NaN tail) must still be valid"

    # parity: fixed and padded must agree
    assert bool(v_pad[0]) == bool(v_fixed[0])

    # a genuinely too-sharp small circle (radius 0.3 < hw=0.5) must be INVALID, count-masked
    small = _circle(120, 0.3, dev).unsqueeze(0)
    _, Nrm_s, _ = wpl.frame_curvature(small, count=cnt_full)
    os, is_ = wpl.offset(small, Nrm_s, hw, count=cnt_full)
    v_small = wpl.validity(small, w, cnt_full, gv, cfg, os, is_)
    assert bool(v_small[0]) is False, "radius-0.3 circle with hw=0.5 must be invalid"


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_constant_spacing(dev):
    hw = 0.5
    buf = torch.full((1, 200, 2), float("nan"), device=dev, dtype=torch.float32)
    buf[0, :120] = _circle(120, 5.0, dev)
    cnt = torch.tensor([120], dtype=torch.int32, device=dev)
    gv = torch.ones(1, dtype=torch.bool, device=dev)
    cfg = TrackGenConfig(num_envs=1, num_points=120, half_width=hw,
                         output_mode="constant_spacing", spacing=0.30, N_max=200, device=dev)
    tr = wpl.inflate_warp(buf, cfg, valid=gv, count=cnt)
    assert tr.center.shape == (1, 200, 2)
    assert int(tr.count[0]) == 120
    assert torch.isnan(tr.center[0, 120:]).all()
    assert bool(tr.valid[0]) is True
    assert torch.allclose(tr.length, torch.tensor([2 * math.pi * 5.0], device=dev), atol=0.5)
