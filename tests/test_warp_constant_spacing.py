"""Constant-spacing Warp pipeline: per-stage parity (count==N_max matches fixed-N) and
variable-count behaviour, plus the end-to-end smoothness/yield win."""
import math
import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen._src import warp_pipeline as wpl
from tests._oracle import geometry  # noqa: E402
from tests._warp_compare import (  # noqa: E402
    to_t, thickness, self_intersections, turning_number, validity, xpbd_solve,
)
from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402

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


def _resample_cs_wp(src_t: torch.Tensor, spacing: float, n_max: int):
    """Constant-spacing resample from a torch tensor; returns (out_t, cnt_t) tensors."""
    E, N, _ = src_t.shape
    dev = str(src_t.device)
    cf = wp.from_torch(src_t.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    count_wp = wp.empty(E, dtype=wp.int32, device=dev)
    out_wp, count_wp = wpl.resample_constant_spacing(cf, spacing, n_max, count_wp=count_wp)
    return wp.to_torch(out_wp).view(E, n_max, 2), wp.to_torch(count_wp).long()


def _offset(center, Nrm, half_width, count=None):
    """Test helper: allocates buffers and calls the in-place wpl.offset, returns torch tensors."""
    E, n_max, _ = center.shape
    dev = center.device
    if count is None:
        count = torch.full((E,), n_max, dtype=torch.int32, device=dev)
    cf  = wp.from_torch(center.reshape(E * n_max, 2).contiguous(), dtype=wp.vec2f)
    nf  = wp.from_torch(Nrm.reshape(E * n_max, 2).contiguous(), dtype=wp.vec2f)
    oo  = wp.zeros(E * n_max, dtype=wp.vec2f, device=str(dev))
    oi  = wp.zeros(E * n_max, dtype=wp.vec2f, device=str(dev))
    aa  = wp.zeros(E, dtype=wp.float32, device=str(dev))
    ab  = wp.zeros(E, dtype=wp.float32, device=str(dev))
    cnt = wp.from_torch(count.to(torch.int32).contiguous(), dtype=wp.int32)
    wpl.offset(cf, nf, half_width, oo, oi, aa, ab, cnt)
    return wp.to_torch(oo).view(E, n_max, 2), wp.to_torch(oi).view(E, n_max, 2)


def _resample_uniform(center, n, count=None):
    """Test helper: allocates wp.array buffers, calls in-place resample_uniform, returns torch."""
    E, n_max, _ = center.shape
    assert n == n_max
    dev = str(center.device)
    flat = E * n_max
    if count is None:
        count_t = torch.full((E,), n_max, dtype=torch.int32, device=center.device)
    else:
        count_t = count.to(torch.int32).contiguous()
    cw = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    out_wp = wp.empty(flat, dtype=wp.vec2f, device=dev)
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    wpl.resample_uniform(cw, out_wp, n, cnt_wp, device=dev)
    if "cuda" in dev:
        wp.synchronize()
    return wp.to_torch(out_wp).view(E, n, 2)


@pytest.mark.parametrize("dev", DEVS)
def test_constant_spacing_resample_matches_torch_oracle(dev):
    E, N = 3, 300
    src = torch.stack([_circle(N, r, dev) for r in (1.0, 2.5, 4.0)], 0)  # [3,N,2]
    spacing, n_max = 0.5, 128
    out_w, cnt_w = _resample_cs_wp(src, spacing, n_max)
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
    base = _resample_uniform(src, 64)
    buf, cnt = _pad(src, 64)
    out = _resample_uniform(buf, 64, count=cnt)
    assert torch.allclose(out, base, atol=1e-5, equal_nan=True)
    # variable: env0 uses 40 real pts (rest NaN), env1 uses 64; env0 stays ~circle, pad NaN
    buf2 = torch.full((2, 64, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :40] = _circle(40, 1.0, dev); buf2[1, :64] = _circle(64, 2.0, dev)
    cnt2 = torch.tensor([40, 64], dtype=torch.int32, device=dev)
    out2 = _resample_uniform(buf2, 64, count=cnt2)
    r0 = torch.linalg.norm(out2[0, :40], dim=-1)
    assert torch.allclose(r0, torch.ones_like(r0), atol=2e-2)
    assert torch.isnan(out2[0, 40:]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_thickness_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    band = torch.tensor([3, 3], dtype=torch.int32, device=dev)
    base = thickness(src, band)
    buf, cnt = _pad(src, 80)
    out = thickness(buf, band, count=cnt)
    assert torch.allclose(out, base, atol=1e-5)
    # variable: env0 real=50 (radius-1 circle) padded to 80; thickness finite, not poisoned by NaN tail
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    th = thickness(buf2, band, count=cnt2)
    assert th[0] > 0.0 and torch.isfinite(th[0])
    # the radius-1 50-pt circle thickness ~ min(curv_radius≈1, 0.5*sep) — sane, finite
    assert torch.isfinite(th[1])


@pytest.mark.parametrize("dev", DEVS)
def test_self_intersections_count_aware(dev):
    # +0.123 phase: crossing falls between samples (transversal), not on the coincident
    # vertices t=0/pi (a degenerate touch the collinear-robust detector ignores).
    t = torch.linspace(0, 2 * math.pi, 100 + 1, device=dev)[:-1] + 0.123
    fig8 = torch.stack([torch.sin(t), torch.sin(t) * torch.cos(t)], -1).to(torch.float32)  # 1 crossing
    circle = _circle(100, 1.0, dev)
    src = torch.stack([fig8, circle], 0)
    base = self_intersections(src)
    buf, cnt = _pad(src, 100)
    out = self_intersections(buf, count=cnt)
    assert torch.equal(out.cpu(), base.cpu())          # parity: count==N matches fixed
    # variable: env0 = 60-pt figure-eight padded to 100; crossing still detected, NaN tail ignored
    buf2 = torch.full((2, 100, 2), float("nan"), device=dev, dtype=torch.float32)
    t2 = torch.linspace(0, 2 * math.pi, 60 + 1, device=dev)[:-1] + 0.123
    buf2[0, :60] = torch.stack([torch.sin(t2), torch.sin(t2) * torch.cos(t2)], -1).to(torch.float32)
    buf2[1, :100] = circle
    cnt2 = torch.tensor([60, 100], dtype=torch.int32, device=dev)
    xs = self_intersections(buf2, count=cnt2)
    assert int(xs[0]) >= 1 and int(xs[1]) == 0          # fig8 crossing; circle simple; pad ignored


@pytest.mark.parametrize("dev", DEVS)
def test_turning_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    base = turning_number(src)
    buf, cnt = _pad(src, 80)
    out = turning_number(buf, count=cnt)
    assert torch.allclose(out, base, atol=1e-5)
    # variable: 50-pt circle padded to 80; turning still ~ 2*pi (turning number 1)
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    tn = turning_number(buf2, count=cnt2)
    assert torch.allclose(tn.abs(), torch.full_like(tn, 2 * math.pi), atol=1e-2)


def _call_frame_curvature(center, count=None):
    """Allocate out/scratch buffers and call the in-place frame_curvature wrapper."""
    E, n_max, _ = center.shape
    flat = E * n_max
    dev = str(center.device)
    cf = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    if count is None:
        count_t = torch.full((E,), n_max, dtype=torch.int32, device=center.device)
    else:
        count_t = count.to(torch.int32).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_T = wp.zeros(flat, dtype=wp.vec2f, device=dev)
    out_Nrm = wp.zeros(flat, dtype=wp.vec2f, device=dev)
    kappa = wp.zeros(flat, dtype=wp.float32, device=dev)
    wpl.frame_curvature(cf, out_T, out_Nrm, kappa, cnt_wp)
    T = wp.to_torch(out_T).view(E, n_max, 2)
    Nrm = wp.to_torch(out_Nrm).view(E, n_max, 2)
    kap = wp.to_torch(kappa).view(E, n_max)
    return T, Nrm, kap


@pytest.mark.parametrize("dev", DEVS)
def test_frame_curvature_count_aware(dev):
    # --- parity: count==N_max reproduces the fixed call ---
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    T_base, Nrm_base, kap_base = _call_frame_curvature(src)
    buf, cnt = _pad(src, 80)
    T_out, Nrm_out, kap_out = _call_frame_curvature(buf, count=cnt)
    assert torch.allclose(T_out, T_base, atol=1e-4)
    assert torch.allclose(Nrm_out, Nrm_base, atol=1e-4)
    assert torch.allclose(kap_out, kap_base, atol=1e-4)

    # --- variable: env0 = 50-pt circle padded to 80, env1 = full 80-pt circle ---
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev)
    buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    T2, Nrm2, kap2 = _call_frame_curvature(buf2, count=cnt2)

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
    _, Nrm_base, _ = _call_frame_curvature(src)
    outer_base, inner_base = _offset(src, Nrm_base, 0.1)
    buf, cnt = _pad(src, 80)
    _, Nrm_buf, _ = _call_frame_curvature(buf, count=cnt)
    outer_out, inner_out = _offset(buf, Nrm_buf, 0.1, count=cnt)
    assert torch.allclose(outer_out, outer_base, atol=1e-4)
    assert torch.allclose(inner_out, inner_base, atol=1e-4)

    # --- variable: env0 = 50-pt circle (r=1) padded to 80, env1 = full 80-pt circle (r=2) ---
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev)
    buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    _, Nrm2, _ = _call_frame_curvature(buf2, count=cnt2)
    outer2, inner2 = _offset(buf2, Nrm2, 0.1, count=cnt2)

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


def _call_arclength(center, count=None):
    """Allocate out buffers and call the in-place _arclength wrapper."""
    E, n_max, _ = center.shape
    flat = E * n_max
    dev = str(center.device)
    cf = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    if count is None:
        count_t = torch.full((E,), n_max, dtype=torch.int32, device=center.device)
    else:
        count_t = count.to(torch.int32).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_arclen = wp.zeros(flat, dtype=wp.float32, device=dev)
    out_length = wp.zeros(E, dtype=wp.float32, device=dev)
    wpl._arclength(cf, out_arclen, out_length, cnt_wp)
    arclen = wp.to_torch(out_arclen).view(E, n_max)
    length = wp.to_torch(out_length)
    return arclen, length


@pytest.mark.parametrize("dev", DEVS)
def test_arclength_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    a0, L0 = _call_arclength(src)
    buf, cnt = _pad(src, 80)
    a, L = _call_arclength(buf, count=cnt)
    assert torch.allclose(a[:, :80], a0, atol=1e-4)
    assert torch.allclose(L, L0, atol=1e-4)
    # variable: env0 = 50-pt unit circle padded to 80; length ~ 2*pi, arclen monotonic over real, NaN tail
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    a2, L2 = _call_arclength(buf2, count=cnt2)
    assert torch.allclose(L2[0], torch.tensor(2 * math.pi, device=dev), atol=1e-2)
    diffs = a2[0, 1:50] - a2[0, :49]
    assert (diffs > 0).all()                 # strictly increasing over real points
    assert torch.isnan(a2[0, 50:]).all()     # NaN padding tail


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("relax_iters", [39, 40])
def test_xpbd_count_aware(dev, relax_iters):
    src = torch.stack([_circle(96, 1.0, dev), _circle(96, 1.4, dev)], 0)
    band = torch.tensor([3, 2], dtype=torch.int32, device=dev)
    L0 = geometry.perimeter(src) / 96
    cfg = TrackGenConfig(num_envs=2, num_points=96, half_width=0.05, relax_iters=relax_iters, device=dev)
    base = xpbd_solve(src, band, L0, cfg)
    buf, cnt = _pad(src, 96)
    out = xpbd_solve(buf, band, L0, cfg, count=cnt)
    assert torch.allclose(out[:, :96], base, atol=1e-5)        # parity: count==N matches fixed
    # variable: env0 = 60-pt circle padded to 96; real points relax & stay finite, padding stays NaN
    buf2 = torch.full((2, 96, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :60] = _circle(60, 1.0, dev); buf2[1, :96] = _circle(96, 1.4, dev)
    cnt2 = torch.tensor([60, 96], dtype=torch.int32, device=dev)
    L02 = torch.tensor([float(geometry.perimeter(buf2[0:1, :60])[0]) / 60,
                        float(geometry.perimeter(buf2[1:2])[0]) / 96], device=dev)
    out2 = xpbd_solve(buf2, band, L02, cfg, count=cnt2)
    assert torch.isfinite(out2[0, :60]).all()                  # real points finite (not NaN-poisoned)
    assert torch.isnan(out2[0, 60:]).all()                     # padding stays NaN
    assert torch.isfinite(out2[1]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_xpbd_separation_cadence_skips_intermediate_sweeps(dev):
    N = 32
    src = _circle(N, 0.1, dev).unsqueeze(0)
    band = torch.tensor([1], dtype=torch.int32, device=dev)
    L0 = geometry.perimeter(src) / N
    base = dict(
        num_envs=1,
        num_points=N,
        half_width=0.08,
        relax_iters=2,
        relax_margin=0.15,
        device=dev,
    )

    every = xpbd_solve(src, band, L0, TrackGenConfig(**base, relax_sep_every=1))
    sparse = xpbd_solve(src, band, L0, TrackGenConfig(**base, relax_sep_every=99))

    assert torch.isfinite(sparse).all()
    assert torch.max(torch.abs(every - sparse)) > 1.0e-4


@pytest.mark.parametrize("dev", DEVS)
def test_xpbd_cached_separation_runs_between_refreshes(dev):
    N = 32
    src = _circle(N, 0.1, dev).unsqueeze(0)
    band = torch.tensor([1], dtype=torch.int32, device=dev)
    L0 = geometry.perimeter(src) / N
    base = dict(
        num_envs=1,
        num_points=N,
        half_width=0.08,
        relax_iters=2,
        relax_margin=0.15,
        device=dev,
    )

    every = xpbd_solve(src, band, L0, TrackGenConfig(**base, relax_sep_every=1))
    sparse = xpbd_solve(src, band, L0, TrackGenConfig(**base, relax_sep_every=99))
    cached = xpbd_solve(
        src,
        band,
        L0,
        TrackGenConfig(**base, relax_sep_every=99, relax_sep_cache_slots=32, relax_sep_cache_skin=0.0),
    )

    sparse_err = torch.max(torch.abs(every - sparse))
    cached_err = torch.max(torch.abs(every - cached))
    assert torch.isfinite(cached).all()
    assert sparse_err > 1.0e-4
    assert cached_err < 1.0e-6


@pytest.mark.parametrize("dev", DEVS)
def test_validity_count_aware(dev):
    hw = 0.5
    cfg = TrackGenConfig(num_envs=1, num_points=120, half_width=hw, device=dev)
    src = _circle(120, 5.0, dev).unsqueeze(0)            # radius 5, 1m road -> easily valid
    w = torch.full((1, 120), hw, device=dev)
    gv = torch.ones(1, dtype=torch.int32, device=dev)
    cnt_full = torch.full((1,), 120, dtype=torch.int32, device=dev)
    # build outer/inner with the count-aware offset (count==N here)
    _, Nrm_full, _ = _call_frame_curvature(src, count=cnt_full)
    o, i = _offset(src, Nrm_full, hw, count=cnt_full)
    v_fixed = validity(src, w, cnt_full, gv, cfg, o, i)   # count==N (fixed mode)
    assert bool(v_fixed[0]) is True, "full circle radius-5 must be valid"

    # padded version: same circle in a 200-wide buffer, count=120
    buf = torch.full((1, 200, 2), float("nan"), device=dev, dtype=torch.float32)
    buf[0, :120] = src[0]
    w2 = torch.full((1, 200), hw, device=dev)
    cnt = torch.tensor([120], dtype=torch.int32, device=dev)
    _, Nrm2, _ = _call_frame_curvature(buf, count=cnt)
    o2, i2 = _offset(buf, Nrm2, hw, count=cnt)
    v_pad = validity(buf, w2, cnt, gv, cfg, o2, i2)
    assert bool(v_pad[0]) is True, "padded circle (NaN tail) must still be valid"

    # parity: fixed and padded must agree
    assert bool(v_pad[0]) == bool(v_fixed[0])

    # a genuinely too-sharp small circle (radius 0.3 < hw=0.5) must be INVALID, count-masked
    small = _circle(120, 0.3, dev).unsqueeze(0)
    _, Nrm_s, _ = _call_frame_curvature(small, count=cnt_full)
    os, is_ = _offset(small, Nrm_s, hw, count=cnt_full)
    v_small = validity(small, w, cnt_full, gv, cfg, os, is_)
    assert bool(v_small[0]) is False, "radius-0.3 circle with hw=0.5 must be invalid"


def _mean_jag(center, count):
    # Resolution-normalised jaggedness: (mean |turning angle|) / (2*pi/c), per env.
    E = center.shape[0]
    out = torch.zeros(E, device=center.device)
    for e in range(E):
        c = int(count[e])
        if c < 3:
            out[e] = 0.0
            continue
        p = center[e, :c]
        d = torch.roll(p, -1, 0) - p
        u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ang = torch.arccos((u * torch.roll(u, 1, 0)).sum(-1).clamp(-1, 1))
        out[e] = ang.mean() / (2 * math.pi / c)
    return out


def test_generate_tracks_constant_spacing_smoother_and_valid():
    # End-to-end: assert the absolute properties constant_spacing delivers in the fat-band
    # regime: near-total yield and smooth valid tracks.
    dev = "cpu"
    E = 96
    N_max = 384
    cfg = TrackGenConfig(output_mode="constant_spacing", spacing=0.30, N_max=N_max,
                         num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         relax_iters=150, device=dev)
    rng = PerEnvSeededRNG(seeds=42, num_envs=E, device=dev)
    gen = TrackGenerator(cfg, rng)
    cs = gen.generate(E)
    # Track fields are wp.array; convert to tensors for assertions.
    valid_t = to_t(cs.valid).bool()
    count_t = to_t(cs.count)
    center_t = wp.to_torch(cs.center).view(E, N_max, 3)[..., :2]
    assert valid_t.float().mean() > 0.95
    assert valid_t.any(), "need valid tracks to measure smoothness"
    jag = _mean_jag(center_t, count_t)[valid_t]
    assert torch.isfinite(jag).all(), "valid tracks must have finite jaggedness"
    assert jag.mean() < 8.0, f"valid constant_spacing tracks should be smooth, got {jag.mean():.3f}"
    assert center_t.shape == (E, N_max, 2)
    assert (count_t <= N_max).all() and (count_t[valid_t] >= 3).all()
    for e in torch.nonzero(valid_t).flatten().tolist():
        c = int(count_t[e])
        assert torch.isfinite(center_t[e, :c]).all(), "real points finite"
        if c < N_max:
            assert torch.isnan(center_t[e, c:]).all(), "padding tail is NaN"


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_constant_spacing(dev):
    hw = 0.5
    n_max = 200
    E = 1
    # Build NaN-padded wp.array center with 120 real points.
    buf_t = torch.full((E, n_max, 2), float("nan"), device=dev, dtype=torch.float32)
    buf_t[0, :120] = _circle(120, 5.0, dev)
    buf_wp = wp.from_torch(buf_t.reshape(E * n_max, 2).contiguous(), dtype=wp.vec2f)
    cnt_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(wpl._fill_i32_k, dim=E, inputs=[cnt_wp, 120], device=dev)
    gv_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(wpl._fill_i32_k, dim=E, inputs=[gv_wp, 1], device=dev)
    cfg = TrackGenConfig(num_envs=E, num_points=120, half_width=hw,
                         output_mode="constant_spacing", spacing=0.30, N_max=n_max, device=dev)
    tr = wpl.inflate_warp(buf_wp, cfg, valid=gv_wp, count=cnt_wp)
    # Track fields are wp.array; convert to tensors for assertions.
    center_t = wp.to_torch(tr.center).view(E, n_max, 3)[..., :2]
    count_t = to_t(tr.count)
    valid_t = to_t(tr.valid).bool()
    length_t = to_t(tr.length)
    assert center_t.shape == (E, n_max, 2)
    assert int(count_t[0]) == 120
    assert torch.isnan(center_t[0, 120:]).all()
    assert bool(valid_t[0]) is True
    assert torch.allclose(length_t, torch.tensor([2 * math.pi * 5.0], device=dev), atol=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_graph_capture_constant_spacing():
    E = 8
    cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device="cuda")
    # TrackGenerator auto-captures on first generate() call (cuda device); second replays.
    rng = PerEnvSeededRNG(seeds=42, num_envs=E, device="cuda")
    gen = TrackGenerator(cfg, rng)
    first = gen.generate(E)   # capture (returns self._track)
    # Clone count+valid before second call overwrites self._track in place.
    count_after_capture = to_t(first.count).cpu().clone()
    valid_after_capture = to_t(first.valid).cpu().clone()
    replay = gen.generate(E)  # replay (same rng seeds -> same result)
    # Track fields are wp.array; convert to tensors for comparison.
    assert torch.equal(to_t(replay.count).cpu(), count_after_capture), \
        "count differs between capture and replay"
    assert torch.equal(to_t(replay.valid).cpu(), valid_after_capture), \
        "valid mask differs between capture and replay"


@pytest.mark.parametrize("dev", DEVS)
def test_constant_spacing_handles_nan_env(dev):
    # a never-accepted env has an all-NaN centerline; resample must give count 0 for it,
    # and generate-style validity must mark it invalid (no crash, no garbage count).
    N = 200
    E = 2
    src = torch.stack([_circle(N, 2.0, dev),
                       torch.full((N, 2), float("nan"), device=dev, dtype=torch.float32)], 0)
    src_wp = wp.from_torch(src.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    cnt_wp = wp.empty(E, dtype=wp.int32, device=dev)
    out_wp, cnt_wp = wpl.resample_constant_spacing(src_wp, 0.3, 256, count_wp=cnt_wp)
    cnt_t = wp.to_torch(cnt_wp)
    out_t = wp.to_torch(out_wp).view(E, 256, 2)
    assert int(cnt_t[1]) == 0                       # NaN env -> count 0 (deterministic)
    assert int(cnt_t[0]) > 0
    assert torch.isnan(out_t[1]).all()              # whole NaN-env row stays NaN
