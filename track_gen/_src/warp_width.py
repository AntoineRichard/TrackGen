"""Per-point half-width variants of the width consumers (roadmap PR13a).

The track pipeline's per-point half-width is the channel ``cl.half_width``. At constant
width (``width.constant``, whose version the planner records constant) the consumers run
TrackGen's original kernels with the value as a host scalar, byte-identical to
0.2.0.dev3 (an in-kernel float32 XPBD target is not bit-exact against the host-double
scalars those kernels take, so the choice is made at plan time). Any other width runs
the kernels here, which read the channel per point:

- XPBD (:func:`band_l0_w_inplace`, :func:`xpbd_solve_w_inplace`): the separation band
  from ``2 * max_i w_i`` (real points); every bead's radius ``r_i = w_i * f32(1 +
  margin)`` precomputed once per solve (``_xpbd_radius_k``, scratch ``xpbd.r``); a pair
  separates to ``r_i + r_j`` (a plain add of two loaded values: symmetric), a bead
  bends toward its own ``R_min = r_i``, the broadphase radius is ``(r_i + r_j) *
  f32(1 + skin)``. The schedule (Chebyshev, cache refresh, Taubin tail, polish sweeps)
  is :func:`track_gen._src.warp_relax.xpbd_solve_inplace`'s, launch for launch, plus the
  radius launch.
- offset (:func:`offset_w`): ``c_t +- w_t n_t``; ``hw_out`` (``Track.half_width``)
  receives ``w_t`` on real points, NaN padding.
- validity (:func:`validity_w_inplace`): the band from ``2 * w_max``; thickness (bit 3)
  fails when two non-neighbouring points are closer than ``(1 - tol)(w_i + w_j)`` or a
  point's curvature radius is below ``(1 - tol) w_i``; the width floor (bit 4) is
  ``not (w_i > w_floor)`` at some real point (NaN fails).

Warp's ``min`` / ``max`` drop NaN: a NaN width does not enter ``w_max``, gives no bend
correction and never triggers a separation (XPBD treats the bead as unconstrained); the
env fails bit 4 anyway. Every wrapper launches through a ``LaunchContext`` and allocates
nothing.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.check import check_array
from ..ops.geometry import _self_intersections_func, _turning_func
from . import warp_relax as _relax
from .runtime import _init
from .warp_relax import _separation_band

__all__ = ["band_l0_w_inplace", "offset_w", "validity_w_inplace", "xpbd_solve_w_inplace"]


# ---------------------------------------------------------------------------------------
# XPBD
# ---------------------------------------------------------------------------------------

@wp.kernel
def _band_l0_w_k(center: wp.array(dtype=wp.vec2f), w: wp.array(dtype=wp.float32),
                 n_max: int, band_out: wp.array(dtype=wp.int32),
                 l0_out: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32)):
    # _band_l0_k with the band from 2 * max_i w_i over the real points (equal to the
    # scalar 2 * hw at constant width: scaling by 2 is exact).
    e = wp.tid()
    base = e * n_max
    cn = count[e]
    peri = float(0.0)
    wmax = float(0.0)
    for i in range(cn):
        peri += wp.length(center[base + (i + 1) % cn] - center[base + i])
        wmax = wp.max(wmax, w[base + i])
    l0 = peri / float(cn)
    l0_out[e] = l0
    band_out[e] = _separation_band(l0, 2.0 * wmax)


@wp.kernel
def _xpbd_radius_k(w: wp.array(dtype=wp.float32), k: wp.float32,
                   r: wp.array(dtype=wp.float32)):
    # One thread per (env, point): the bead radius r = w * f32(1 + margin).
    t = wp.tid()
    r[t] = w[t] * k


@wp.kernel
def _step_w_kernel(center: wp.array(dtype=wp.vec2f), r: wp.array(dtype=wp.float32),
                   band: wp.array(dtype=wp.int32), L0: wp.array(dtype=wp.float32),
                   sr: wp.float32, pr: wp.float32, br: wp.float32, do_sep: int,
                   n_max: int, count: wp.array(dtype=wp.int32),
                   out: wp.array(dtype=wp.vec2f)):
    # warp_relax._step_kernel with per-bead radii: pair target r_i + r_j, R_min = r_i.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    if i >= count[e]:
        out[t] = center[t]
        return
    xi = center[t]
    ne = count[e]
    band_e = band[e]
    l0_e = L0[e]
    r_i = r[t]
    sep = wp.vec2f(0.0, 0.0)
    cnt = int(0)
    if do_sep != 0:
        for j in range(ne):
            dd = wp.abs(i - j)
            circ = wp.min(dd, ne - dd)
            if circ > band_e:
                target = r_i + r[b + j]
                diff = xi - center[b + j]
                adx = wp.abs(diff[0])
                ady = wp.abs(diff[1])
                if adx < target and ady < target:
                    dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                    if dist2 < target * target:
                        dist = wp.max(wp.sqrt(dist2), 1.0e-9)
                        pen = target - dist
                        sep = sep + (0.5 * pen / dist) * diff
                        cnt += 1
        if cnt > 0:
            sep = sep / wp.float32(cnt)
    xn = center[b + ((i + 1) % ne)]
    xp = center[b + ((i + ne - 1) % ne)]
    dn = xn - xi
    ln = wp.max(wp.length(dn), 1.0e-9)
    dp = xi - xp
    lp = wp.max(wp.length(dp), 1.0e-9)
    spc = 0.25 * (((ln - l0_e) / ln) * dn - ((lp - l0_e) / lp) * dp)
    a = xi - xp
    bb = xn - xi
    la = wp.length(a)
    lb = wp.length(bb)
    lc = wp.length(xn - xp)
    denom = wp.max(la * lb * lc, 1.0e-12)
    cross = a[0] * bb[1] - a[1] * bb[0]
    area = 0.5 * wp.abs(cross)
    kappa = 4.0 * area / denom
    radius = 1.0 / wp.max(kappa, 1.0e-12)
    mid = 0.5 * (xp + xn)
    toward = mid - xi
    R_min = r_i
    deficit = wp.max((R_min - radius) / R_min, 0.0)
    bscale = wp.min(br * deficit, 1.0)
    step = sr * sep + pr * spc + bscale * toward
    out[t] = xi + step


@wp.kernel
def _build_sep_cache_w_kernel(center: wp.array(dtype=wp.vec2f),
                              r: wp.array(dtype=wp.float32),
                              band: wp.array(dtype=wp.int32), kskin: wp.float32,
                              n_max: int, cache_slots: int,
                              count: wp.array(dtype=wp.int32),
                              cache_count: wp.array(dtype=wp.int32),
                              cache_idx: wp.array(dtype=wp.int32),
                              overflow: wp.array(dtype=wp.int32)):
    # warp_relax._build_sep_cache_kernel with the pair radius (r_i + r_j) * f32(1 + skin).
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    if i >= count[e]:
        cache_count[t] = 0
        return
    xi = center[t]
    ne = count[e]
    band_e = band[e]
    r_i = r[t]
    n_cached = int(0)
    did_overflow = int(0)
    for j in range(ne):
        dd = wp.abs(i - j)
        circ = wp.min(dd, ne - dd)
        if circ > band_e:
            radius = (r_i + r[b + j]) * kskin
            diff = xi - center[b + j]
            adx = wp.abs(diff[0])
            ady = wp.abs(diff[1])
            if adx < radius and ady < radius:
                dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                if dist2 < radius * radius:
                    if n_cached < cache_slots:
                        cache_idx[t * cache_slots + n_cached] = j
                        n_cached += 1
                    else:
                        did_overflow = 1
    cache_count[t] = n_cached
    if did_overflow != 0:
        wp.atomic_add(overflow, 0, 1)


@wp.kernel
def _step_cached_w_kernel(center: wp.array(dtype=wp.vec2f), r: wp.array(dtype=wp.float32),
                          cache_count: wp.array(dtype=wp.int32),
                          cache_idx: wp.array(dtype=wp.int32),
                          L0: wp.array(dtype=wp.float32), sr: wp.float32, pr: wp.float32,
                          br: wp.float32, n_max: int, cache_slots: int,
                          count: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
    # warp_relax._step_cached_kernel with per-bead radii.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    if i >= count[e]:
        out[t] = center[t]
        return
    xi = center[t]
    ne = count[e]
    l0_e = L0[e]
    r_i = r[t]
    sep = wp.vec2f(0.0, 0.0)
    cnt = int(0)
    n_cached = cache_count[t]
    for slot in range(cache_slots):
        if slot < n_cached:
            j = cache_idx[t * cache_slots + slot]
            target = r_i + r[b + j]
            diff = xi - center[b + j]
            adx = wp.abs(diff[0])
            ady = wp.abs(diff[1])
            if adx < target and ady < target:
                dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                if dist2 < target * target:
                    dist = wp.max(wp.sqrt(dist2), 1.0e-9)
                    pen = target - dist
                    sep = sep + (0.5 * pen / dist) * diff
                    cnt += 1
    if cnt > 0:
        sep = sep / wp.float32(cnt)
    xn = center[b + ((i + 1) % ne)]
    xp = center[b + ((i + ne - 1) % ne)]
    dn = xn - xi
    ln = wp.max(wp.length(dn), 1.0e-9)
    dp = xi - xp
    lp = wp.max(wp.length(dp), 1.0e-9)
    spc = 0.25 * (((ln - l0_e) / ln) * dn - ((lp - l0_e) / lp) * dp)
    a = xi - xp
    bb = xn - xi
    la = wp.length(a)
    lb = wp.length(bb)
    lc = wp.length(xn - xp)
    denom = wp.max(la * lb * lc, 1.0e-12)
    cross = a[0] * bb[1] - a[1] * bb[0]
    area = 0.5 * wp.abs(cross)
    kappa = 4.0 * area / denom
    radius = 1.0 / wp.max(kappa, 1.0e-12)
    mid = 0.5 * (xp + xn)
    toward = mid - xi
    R_min = r_i
    deficit = wp.max((R_min - radius) / R_min, 0.0)
    bscale = wp.min(br * deficit, 1.0)
    step = sr * sep + pr * spc + bscale * toward
    out[t] = xi + step


def _f32(x: float) -> float:
    return float(np.float32(x))


def band_l0_w_inplace(center_wp, n_max: int, w_wp, band_wp, l0_wp, count_wp, config,
                      ctx: "LaunchContext | None" = None) -> None:
    """:func:`warp_relax.band_l0_inplace <track_gen._src.warp_relax.band_l0_inplace>`
    with the band from the per-point half-width ``w_wp`` (``[E*n_max]`` float32; the
    largest real point's). ``config.relax_band`` (when not None) overrides it."""
    _init()
    E = count_wp.shape[0]
    ctx = _resolve_ctx(ctx, str(center_wp.device))
    ctx.launch(_band_l0_w_k, E, [center_wp, w_wp, n_max, band_wp, l0_wp, count_wp])
    if config.relax_band is not None:
        ctx.launch(_relax._band_fill_k, E, [band_wp, int(config.relax_band)])


def xpbd_solve_w_inplace(center_wp, relaxed_wp, db_wp, band_wp, l0_wp, count_wp,
                         n_max: int, config, w_wp, r_wp,
                         sep_cache_idx_wp=None, sep_cache_count_wp=None,
                         sep_cache_overflow_wp=None, cheb_prev_wp=None,
                         ctx: "LaunchContext | None" = None) -> None:
    """:func:`warp_relax.xpbd_solve_inplace <track_gen._src.warp_relax.xpbd_solve_inplace>`
    with per-bead radii: ``r_wp`` (``[E*n_max]`` float32 scratch) receives
    ``w_wp * f32(1 + relax_margin)`` first (one launch), then the same schedule runs the
    per-point kernels. Zero allocation (every buffer pre-allocated by the caller)."""
    _init()
    ctx = _resolve_ctx(ctx, str(center_wp.device))
    E = count_wp.shape[0]
    k = _f32(1.0 + float(config.relax_margin))
    sr = float(config.relax_sep_relax)
    pr = float(config.relax_spc_relax)
    br = float(config.relax_bend_relax)
    sep_every = max(1, int(getattr(config, "relax_sep_every", 1)))
    cache_slots = max(0, int(getattr(config, "relax_sep_cache_slots", 0)))
    kskin = _f32(1.0 + max(0.0, float(getattr(config, "relax_sep_cache_skin", 0.0))))
    smooth_passes = max(0, int(getattr(config, "relax_smooth_passes", 0)))
    smooth_spacing_iters = max(0, int(getattr(config, "relax_smooth_spacing_iters", 0)))
    use_cache = cache_slots > 0 and sep_every > 1
    if use_cache and (sep_cache_idx_wp is None or sep_cache_count_wp is None
                      or sep_cache_overflow_wp is None):
        raise ValueError("relax_sep_cache_slots > 0 requires pre-allocated cache buffers")
    n_iters = int(config.relax_iters)
    use_cheby = str(getattr(config, "relax_accel", "none")) == "chebyshev"
    if use_cheby:
        gamma = float(getattr(config, "relax_cheby_gamma", 0.9))
        cheby_start = max(1, int(getattr(config, "relax_cheby_start", 8)))
        omegas = _relax._cheby_schedule(
            n_iters, float(getattr(config, "relax_cheby_rho", 0.98)), cheby_start)
        if cheb_prev_wp is None:
            raise ValueError("relax_accel='chebyshev' requires a pre-allocated "
                             "cheb_prev_wp buffer")
        prev_wp = cheb_prev_wp

    flat = E * n_max
    ctx.launch(_xpbd_radius_k, flat, [w_wp, wp.float32(k), r_wp])
    ctx.copy(relaxed_wp, center_wp)
    read_wp = relaxed_wp
    write_wp = db_wp
    for step_i in range(n_iters):
        if use_cache:
            if step_i % sep_every == 0:
                ctx.launch(_relax._band_fill_k, 1, [sep_cache_overflow_wp, 0])
                ctx.launch(_build_sep_cache_w_kernel, flat,
                           [read_wp, r_wp, band_wp, wp.float32(kskin), n_max, cache_slots,
                            count_wp, sep_cache_count_wp, sep_cache_idx_wp,
                            sep_cache_overflow_wp])
            ctx.launch(_step_cached_w_kernel, flat,
                       [read_wp, r_wp, sep_cache_count_wp, sep_cache_idx_wp, l0_wp,
                        sr, pr, br, n_max, cache_slots, count_wp, write_wp])
        else:
            do_sep = 1 if step_i % sep_every == 0 else 0
            ctx.launch(_step_w_kernel, flat,
                       [read_wp, r_wp, band_wp, l0_wp, sr, pr, br, do_sep, n_max,
                        count_wp, write_wp])
        if use_cheby and step_i >= cheby_start:
            x_prev = read_wp if step_i == cheby_start else prev_wp
            ctx.launch(_relax._accel_blend_kernel, flat,
                       [write_wp, read_wp, x_prev, wp.float32(omegas[step_i]),
                        wp.float32(gamma), write_wp])
            read_wp, write_wp, prev_wp = write_wp, prev_wp, read_wp
        else:
            read_wp, write_wp = write_wp, read_wp
    for _ in range(smooth_passes):
        ctx.launch(_relax._taubin_kernel, flat,
                   [read_wp, wp.float32(_relax._TAUBIN_LAMBDA), n_max, count_wp, write_wp])
        read_wp, write_wp = write_wp, read_wp
        ctx.launch(_relax._taubin_kernel, flat,
                   [read_wp, wp.float32(_relax._TAUBIN_MU), n_max, count_wp, write_wp])
        read_wp, write_wp = write_wp, read_wp
    for _ in range(smooth_spacing_iters):
        ctx.launch(_step_w_kernel, flat,
                   [read_wp, r_wp, band_wp, l0_wp, 0.0, pr, 0.0, 0, n_max, count_wp,
                    write_wp])
        read_wp, write_wp = write_wp, read_wp
    if read_wp is not relaxed_wp:
        ctx.copy(relaxed_wp, read_wp)


# ---------------------------------------------------------------------------------------
# Offset
# ---------------------------------------------------------------------------------------

@wp.kernel
def _offset_build_w_k(center: wp.array(dtype=wp.vec2f), Nrm: wp.array(dtype=wp.vec2f),
                      w: wp.array(dtype=wp.float32), n_max: int,
                      area_a: wp.array(dtype=wp.float32),
                      area_b: wp.array(dtype=wp.float32),
                      count: wp.array(dtype=wp.int32)):
    # warp_pipeline._offset_build_k with the per-point half-width.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b_base = e * n_max
    if i >= count[e]:
        return
    ct = center[t]
    nt = Nrm[t]
    wt = w[t]
    at = ct + wt * nt
    bt = ct - wt * nt
    next_idx = b_base + (i + 1) % count[e]
    cn = center[next_idx]
    nn = Nrm[next_idx]
    wn = w[next_idx]
    an = cn + wn * nn
    bn = cn - wn * nn
    cross_a = at[0] * an[1] - an[0] * at[1]
    cross_b = bt[0] * bn[1] - bn[0] * bt[1]
    wp.atomic_add(area_a, e, 0.5 * cross_a)
    wp.atomic_add(area_b, e, 0.5 * cross_b)


@wp.kernel
def _offset_assign_w_k(center: wp.array(dtype=wp.vec2f), Nrm: wp.array(dtype=wp.vec2f),
                       w: wp.array(dtype=wp.float32), n_max: int,
                       area_a: wp.array(dtype=wp.float32),
                       area_b: wp.array(dtype=wp.float32),
                       outer: wp.array(dtype=wp.vec2f), inner: wp.array(dtype=wp.vec2f),
                       count: wp.array(dtype=wp.int32),
                       hw_out: wp.array(dtype=wp.float32)):
    # warp_pipeline._offset_assign_k with the per-point half-width; hw_out (when
    # non-empty) receives it on real points, NaN padding.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    store = hw_out.shape[0] > 0
    if i >= count[e]:
        nan_val = wp.vec2f(wp.nan, wp.nan)
        outer[t] = nan_val
        inner[t] = nan_val
        if store:
            hw_out[t] = wp.nan
        return
    ct = center[t]
    nt = Nrm[t]
    wt = w[t]
    at = ct + wt * nt
    bt = ct - wt * nt
    aa = wp.abs(area_a[e])
    ab = wp.abs(area_b[e])
    if aa >= ab:
        outer[t] = at
        inner[t] = bt
    else:
        outer[t] = bt
        inner[t] = at
    if store:
        hw_out[t] = wt


def offset_w(center, Nrm, w, out_outer, out_inner, area_a, area_b, count, hw_out=None,
             ctx: "LaunchContext | None" = None) -> None:
    """:func:`warp_pipeline.offset <track_gen._src.warp_pipeline.offset>` with the
    per-point half-width ``w`` (``[E*n_max]`` float32). ``hw_out`` (``None``: nothing
    stored) receives ``w`` on real points, NaN padding (``ValueError`` unless it is a
    float32 array of ``out_outer``'s length on its device). Allocates nothing."""
    _init()
    ctx = _resolve_ctx(ctx, str(out_outer.device))
    E = count.shape[0]
    flat = out_outer.shape[0]
    n_max = flat // E
    if hw_out is not None:
        check_array("hw_out", hw_out, (int(flat),), wp.float32, out_outer.device)
    ctx.fill(area_a, 0.0)
    ctx.fill(area_b, 0.0)
    ctx.launch(_offset_build_w_k, flat, [center, Nrm, w, n_max, area_a, area_b, count])
    ctx.launch(_offset_assign_w_k, flat,
               [center, Nrm, w, n_max, area_a, area_b, out_outer, out_inner, count,
                hw_out])
    ctx.sync()


# ---------------------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------------------

@wp.func
def _thickness_ok_w(pts: wp.array(dtype=wp.vec2f), w: wp.array(dtype=wp.float32),
                    base: int, N: int, band: int, tol: float) -> int:
    # Per pair (circular distance > band): |p_i - p_j| >= (1 - tol) (w_i + w_j); per
    # point: its curvature radius 1 / max(kappa_i, 1e-12) >= (1 - tol) w_i. At constant
    # width this is _thickness_func's min(rad_min, 0.5 sep_min) >= (1 - tol) hw, exactly.
    c = 1.0 - tol
    ok = int(1)
    for i in range(N):
        for j in range(i + 1, N):
            diff = j - i
            circ = wp.min(diff, N - diff)
            if circ > band:
                d = wp.length(pts[base + i] - pts[base + j])
                if not (d >= c * (w[base + i] + w[base + j])):
                    ok = int(0)
    for i in range(N):
        xp = pts[base + (i + N - 1) % N]
        xc = pts[base + i]
        xn = pts[base + (i + 1) % N]
        a = xc - xp
        bb = xn - xc
        cc = xn - xp
        cross = a[0] * bb[1] - a[1] * bb[0]
        area = 0.5 * wp.abs(cross)
        denom = wp.max(wp.length(a) * wp.length(bb) * wp.length(cc), float(1.0e-12))
        kappa = 4.0 * area / denom
        rad = 1.0 / wp.max(kappa, float(1.0e-12))
        if not (rad >= c * w[base + i]):
            ok = int(0)
    return ok


@wp.kernel
def _validity_w_k(
    center: wp.array(dtype=wp.vec2f),
    w: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    gen_valid: wp.array(dtype=wp.int32),
    outer: wp.array(dtype=wp.vec2f),
    inner: wp.array(dtype=wp.vec2f),
    has_border: int,
    n_max: int,
    turning_tol: float,
    w_floor: float,
    relax_tol: float,
    z: wp.array(dtype=wp.float32),
    z_valid_grade: float,
    truncated: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
    fail: wp.array(dtype=wp.int32),
):
    # warp_pipeline._validity_k with the per-point half-width: the band from 2 * w_max,
    # the per-pair / per-point thickness test; the same bits.
    e = wp.tid()
    cnt = count[e]
    base = e * n_max
    if cnt == 0:
        out[e] = 0
        f0 = int(2)
        if gen_valid[e] == 0:
            f0 = f0 | 1
        if truncated[e] != 0:
            f0 = f0 | 128
        fail[e] = f0
        return
    turn = _turning_func(center, base, cnt)
    turn_ok = int(0)
    if wp.abs(wp.abs(turn) - 2.0 * wp.pi) <= turning_tol:
        turn_ok = int(1)
    w_ok = int(1)
    no_nan = int(1)
    wmax = float(0.0)
    for i in range(cnt):
        wi = w[base + i]
        if not (wi > w_floor):
            w_ok = int(0)
        wmax = wp.max(wmax, wi)
        ci = center[base + i]
        if not (wp.isfinite(ci[0]) and wp.isfinite(ci[1])):
            no_nan = int(0)
    peri = float(0.0)
    for i in range(cnt):
        peri += wp.length(center[base + (i + 1) % cnt] - center[base + i])
    band = _separation_band(peri / float(cnt), 2.0 * wmax)
    th_ok = _thickness_ok_w(center, w, base, cnt, band, relax_tol)
    border_ok = int(1)
    if has_border == 1:
        cross = _self_intersections_func(outer, base, cnt) + \
            _self_intersections_func(inner, base, cnt)
        if cross != 0:
            border_ok = int(0)
    grade_ok = int(1)
    if z_valid_grade > 0.0:
        for i in range(cnt):
            j = i + 1
            if j == cnt:
                j = 0
            a = center[base + i]
            b = center[base + j]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            dxy = wp.sqrt(dx * dx + dy * dy)
            if wp.abs(z[base + j] - z[base + i]) > z_valid_grade * wp.max(dxy, 1.0e-9):
                grade_ok = int(0)
    gv = int(0)
    if gen_valid[e] != 0:
        gv = int(1)
    trunc_ok = int(1)
    if truncated[e] != 0:
        trunc_ok = int(0)
    out[e] = gv & turn_ok & w_ok & no_nan & th_ok & border_ok & grade_ok & trunc_ok
    f = int(0)
    if gv == 0:
        f = f | 1
    if turn_ok == 0:
        f = f | 2
    if no_nan == 0:
        f = f | 4
    if th_ok == 0:
        f = f | 8
    if w_ok == 0:
        f = f | 16
    if border_ok == 0:
        f = f | 32
    if grade_ok == 0:
        f = f | 64
    if trunc_ok == 0:
        f = f | 128
    fail[e] = f


def validity_w_inplace(center_wp, w_wp, count_wp, gen_valid_wp, outer_wp, inner_wp,
                       has_border: int, n_max: int, z_wp, out_valid, config,
                       truncated_wp, fail_wp, ctx: "LaunchContext | None" = None) -> None:
    """:func:`warp_pipeline.validity_inplace
    <track_gen._src.warp_pipeline.validity_inplace>` with the per-point half-width
    ``w_wp`` (the thickness test per pair / per point, the band from ``2 * w_max``).
    Every buffer is the caller's (no allocation)."""
    _init()
    ctx = _resolve_ctx(ctx, str(out_valid.device))
    E = count_wp.shape[0]
    ctx.launch(_validity_w_k, E,
               [center_wp, w_wp, count_wp, gen_valid_wp, outer_wp, inner_wp,
                int(has_border), int(n_max), float(config.turning_tol),
                float(config.w_floor), float(config.relax_tol), z_wp,
                float(config.z_valid_grade), truncated_wp, out_valid, fail_wp])
    ctx.sync()


# ---------------------------------------------------------------------------------------
# width.from_channel
# ---------------------------------------------------------------------------------------

@wp.kernel
def _width_from_channel_k(src: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
                          scale: wp.array(dtype=wp.float32),
                          lo: wp.array(dtype=wp.float32), hi: wp.array(dtype=wp.float32),
                          S: int, out: wp.array(dtype=wp.float32)):
    # Real points: clamp(scale * src, lo, max(hi, lo)) (a run-time hi below lo clamps to
    # lo); a NaN source stays NaN (Warp's min / max drop NaN, hence the explicit branch);
    # padding NaN.
    t = wp.tid()
    e = t // S
    i = t - e * S
    if i >= count[e]:
        out[t] = wp.nan
        return
    v = src[t]
    if wp.isnan(v):
        out[t] = wp.nan
    else:
        out[t] = wp.min(wp.max(scale[e] * v, lo[e]), wp.max(hi[e], lo[e]))
