"""Pure-Warp track-generation pipeline kernels.

Every pipeline stage (generation, resample, relax, inflate) is expressed as Warp
kernels that run on BOTH the Warp ``cpu`` device (tests/CI, GPU-free) and ``cuda``
(production). The whole pipeline is graph-capturable on CUDA. During the port each
kernel is verified ``allclose`` against the oracle.

Convention: one thread per output element; flat arrays ``[E*N]`` of ``wp.vec2f`` and
``[E]`` per-env scalars; env index ``e = tid // N``; launch with
``device=str(tensor.device)``.
"""
from __future__ import annotations

import math

from . import warp_relax  # pure-Warp XPBD solve (cpu+cuda); part of the pure-Warp impl

import warp as wp

_INITED = False

# True only inside a CUDA graph capture region (set by TrackGenerator around the
# wp.ScopedCapture context). While set, every wrapper's _sync() is a no-op (host-blocking
# sync is ILLEGAL during CUDA graph capture) and warp_relax.xpbd_solve skips its own
# wp.synchronize(). Module-global because the whole pipeline (and warp_relax) must agree,
# and the captured region is single-threaded/serial by construction.
_CAPTURING = False


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    # Skip the host-blocking sync while a graph is being captured (it is illegal there);
    # the graph records stream ordering, so the sync is unnecessary on replay too.
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


@wp.kernel
def _offset_build_k(
    center: wp.array(dtype=wp.vec2f),
    Nrm: wp.array(dtype=wp.vec2f),
    half_width: float,
    n_max: int,
    area_a: wp.array(dtype=wp.float32),
    area_b: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
):
    # One thread per point t.  e = env index, i = point-within-env index.
    # Accumulates the per-env signed shoelace cross-product terms for
    # candidate polygons a (center + hw*Nrm) and b (center - hw*Nrm) via
    # atomic adds.  area_a/area_b must be zero-initialised before launch.
    # Padding threads (i >= count[e]) add nothing to the accumulators.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b_base = e * n_max

    # Guard: padding threads contribute nothing to the shoelace accumulation.
    if i >= count[e]:
        return

    # Offset points at this thread's index.
    ct = center[t]
    nt = Nrm[t]
    at = ct + half_width * nt
    bt = ct - half_width * nt

    # Offset points at the NEXT index (wraps within the real loop).
    next_idx = b_base + (i + 1) % count[e]
    cn = center[next_idx]
    nn = Nrm[next_idx]
    an = cn + half_width * nn
    bn = cn - half_width * nn

    # Shoelace edge contribution: x_i * y_{i+1} - x_{i+1} * y_i
    cross_a = at[0] * an[1] - an[0] * at[1]
    cross_b = bt[0] * bn[1] - bn[0] * bt[1]
    wp.atomic_add(area_a, e, 0.5 * cross_a)
    wp.atomic_add(area_b, e, 0.5 * cross_b)

@wp.kernel
def _offset_assign_k(
    center: wp.array(dtype=wp.vec2f),
    Nrm: wp.array(dtype=wp.vec2f),
    half_width: float,
    n_max: int,
    area_a: wp.array(dtype=wp.float32),
    area_b: wp.array(dtype=wp.float32),
    outer: wp.array(dtype=wp.vec2f),
    inner: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
):
    # One thread per point t.  Recompute a[t], b[t] (cheap), then assign
    # outer/inner based on which candidate has the larger |signed area|.
    # Padding threads (i >= count[e]) write NaN to outer/inner and return.
    t = wp.tid()
    e = t // n_max
    i = t % n_max

    # Guard: padding threads write NaN.
    if i >= count[e]:
        nan_val = wp.vec2f(wp.nan, wp.nan)
        outer[t] = nan_val
        inner[t] = nan_val
        return

    ct = center[t]
    nt = Nrm[t]
    at = ct + half_width * nt
    bt = ct - half_width * nt
    aa = wp.abs(area_a[e])
    ab = wp.abs(area_b[e])
    if aa >= ab:
        outer[t] = at
        inner[t] = bt
    else:
        outer[t] = bt
        inner[t] = at

@wp.kernel
def _frame_k(c: wp.array(dtype=wp.vec2f), n_max: int,
             T: wp.array(dtype=wp.vec2f), Nrm: wp.array(dtype=wp.vec2f),
             kappa: wp.array(dtype=wp.float32),
             count: wp.array(dtype=wp.int32)):
    # Per closed-loop point: central-difference unit tangent, left-normal, and
    # non-negative Menger curvature. Matches geometry.tangents_normals + menger_curvature.
    # Padding threads (i >= count[e]) write NaN and return.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max

    # Guard: padding threads write NaN to all outputs and return.
    if i >= count[e]:
        nan_val = wp.vec2f(wp.nan, wp.nan)
        T[t] = nan_val
        Nrm[t] = nan_val
        kappa[t] = wp.nan
        return

    real_n = count[e]
    xp = c[b + (i + real_n - 1) % real_n]
    xc = c[t]
    xn = c[b + (i + 1) % real_n]
    d = xn - xp
    inv = 1.0 / wp.max(wp.length(d), 1.0e-8)   # safe_normalize floor
    tan = d * inv
    T[t] = tan
    Nrm[t] = wp.vec2f(-tan[1], tan[0])
    a = xc - xp
    bb = xn - xc
    cc = xn - xp
    cross = a[0] * bb[1] - a[1] * bb[0]
    area = 0.5 * wp.abs(cross)
    denom = wp.max(wp.length(a) * wp.length(bb) * wp.length(cc), 1.0e-12)
    kappa[t] = 4.0 * area / denom

@wp.func
def _separation_band(l0: float, two_hw: float) -> int:
    # Excluded-neighbour half-window for the XPBD separation pass and the thickness
    # gate: round(2*half_width / L0).clamp_min(1), where L0 is the mean segment length.
    # The divisor is floored at 1e-9 to bound the ratio, and the isfinite guard maps a
    # NaN/inf ratio (invalid NaN-centerline envs) to band 1. Shared by _band_l0_k and
    # _validity_k so the band definition lives in exactly one place.
    bf = two_hw / wp.max(l0, float(1.0e-9))
    return wp.where(wp.isfinite(bf), wp.max(int(wp.round(bf)), 1), 1)

@wp.func
def _ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
    # Returns (q.y-o.y)*(p.x-o.x) - (p.y-o.y)*(q.x-o.x)
    return (qy - oy) * (px - ox) - (py - oy) * (qx - ox)

@wp.func
def _nan0(x: float) -> float:
    # NaN/inf -> 0.0 guard (nan_to_num equivalent) for the border self-intersection
    # check. wp.where(cond, if_true, if_false) is the Warp 1.14 non-deprecated select
    # primitive. A NO-OP for finite inputs.
    return wp.where(wp.isfinite(x), x, 0.0)

@wp.func
def _self_intersections_func(poly: wp.array(dtype=wp.vec2f), base: int, N: int) -> int:
    # Proper-crossing double-loop count for the env whose points start at `base`.
    # Each coordinate is read through _nan0 (NaN->0 guard), which is a no-op on finite
    # inputs and reproduces the validity border check's nan_to_num(outer/inner, nan=0.0).
    count = int(0)
    for i in range(N):
        for j in range(i + 1, N):
            diff = j - i
            circ_dist = wp.min(diff, N - diff)
            if circ_dist <= 1:
                continue
            Ai = poly[base + i]
            Bi = poly[base + (i + 1) % N]
            Aj = poly[base + j]
            Bj = poly[base + (j + 1) % N]
            aix = _nan0(Ai[0]); aiy = _nan0(Ai[1])
            bix = _nan0(Bi[0]); biy = _nan0(Bi[1])
            ajx = _nan0(Aj[0]); ajy = _nan0(Aj[1])
            bjx = _nan0(Bj[0]); bjy = _nan0(Bj[1])
            d1 = _ccw(ajx, ajy, bjx, bjy, aix, aiy)
            d2 = _ccw(ajx, ajy, bjx, bjy, bix, biy)
            d3 = _ccw(aix, aiy, bix, biy, ajx, ajy)
            d4 = _ccw(aix, aiy, bix, biy, bjx, bjy)
            # Scale-relative collinearity tolerance (matches geometry.SELF_X_REL): a
            # straddle counts only if both endpoints clear the other segment's line by
            # more than SELF_X_REL * (that segment length). |d| = len * perp-offset, so
            # the threshold eps = REL * len^2. Kills the float32 sign-flip false positives
            # on near-collinear/straight segments without missing genuine crossings.
            lj2 = (bjx - ajx) * (bjx - ajx) + (bjy - ajy) * (bjy - ajy)
            li2 = (bix - aix) * (bix - aix) + (biy - aiy) * (biy - aiy)
            ej = float(1.0e-3) * lj2
            ei = float(1.0e-3) * li2
            seg_ij = (d1 > ej and d2 < -ej) or (d1 < -ej and d2 > ej)
            seg_ji = (d3 > ei and d4 < -ei) or (d3 < -ei and d4 > ei)
            if seg_ij and seg_ji:
                count = count + 1
    return count

@wp.kernel
def _self_intersections_k(
    poly: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. Delegates to _self_intersections_func over [e*n_max, e*n_max+count[e]).
    e = wp.tid()
    out[e] = _self_intersections_func(poly, e * n_max, count[e])

@wp.func
def _thickness_func(pts: wp.array(dtype=wp.vec2f), base: int, N: int, band: int) -> float:
    # thickness = min(rad_min, 0.5 * sep_min) for the env whose points start at `base`.
    # sep_min = min pairwise distance over pairs with circ_dist > band;
    # rad_min  = 1 / max Menger curvature.
    # --- sep_min: min dist over pairs with circ_dist > band ---
    sep_min = float(1.0e30)
    for i in range(N):
        for j in range(i + 1, N):
            diff = j - i
            circ_dist = wp.min(diff, N - diff)
            if circ_dist > band:
                pi = pts[base + i]
                pj = pts[base + j]
                d = wp.length(pi - pj)
                sep_min = wp.min(sep_min, d)

    # --- rad_min: 1 / max Menger curvature ---
    kappa_max = float(0.0)
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
        kappa_max = wp.max(kappa_max, kappa)
    rad_min = 1.0 / wp.max(kappa_max, float(1.0e-12))

    return wp.min(rad_min, 0.5 * sep_min)

@wp.kernel
def _thickness_k(
    pts: wp.array(dtype=wp.vec2f),
    band: wp.array(dtype=wp.int32),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Delegates to _thickness_func with this env's band.
    e = wp.tid()
    out[e] = _thickness_func(pts, e * n_max, count[e], band[e])

@wp.kernel
def _resample_scan_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
):
    # One thread per env e. Segment lengths seg[e*n_max+i]=|c[i+1]-c[i]|
    # (i+1 wraps via %count[e]) and cumulative arc s[e*(n_max+1)+0]=0,
    # s[..+i+1]=s[..+i]+seg[i] for i in range(count[e]).
    # Running sum in float64 to limit accumulation drift.
    # PARITY: when count[e]==n_max for all e, produces identical output to the
    # former fixed-N kernel (same float64 accumulation order, same modular indexing).
    e = wp.tid()
    cn = count[e]
    b = e * n_max
    es = e * (n_max + 1)
    s[es] = float(0.0)
    acc = wp.float64(0.0)
    for i in range(cn):
        d = c[b + (i + 1) % cn] - c[b + i]
        l = wp.length(d)
        seg[b + i] = l
        acc = acc + wp.float64(l)
        s[es + i + 1] = wp.float32(acc)

@wp.kernel
def _resample_lookup_k(
    c: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per output slot t; e = t // n_max, k = t % n_max.
    # k >= count[e] -> NaN pad (padding region).
    # Otherwise: target tk = k * total / count[e], linear scan then lerp.
    # PARITY: when count[e]==n_max for all e, result is bit-identical to the
    # former fixed-N kernel (same formula, same scan bounds, same wrap modulus).
    t = wp.tid()
    e = t // n_max
    k = t % n_max
    cn = count[e]
    if k >= cn:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return
    eb = e * n_max
    es = e * (n_max + 1)
    total = s[es + cn]
    tk = float(k) * total / float(cn)
    idx = int(0)
    while idx < cn - 1 and s[es + idx + 1] < tk:
        idx = idx + 1
    s0 = s[es + idx]
    segl = wp.max(seg[eb + idx], float(1.0e-12))
    frac = wp.clamp((tk - s0) / segl, float(0.0), float(1.0))
    p0 = c[eb + idx]
    p1 = c[eb + (idx + 1) % cn]
    out[t] = p0 + frac * (p1 - p0)

@wp.kernel
def _cs_scan_k(c: wp.array(dtype=wp.vec2f), N: int, spacing: wp.float32, n_max: int,
               seg: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32),
               count: wp.array(dtype=wp.int32)):
    # One thread per env e. Closed-loop seg lengths + cumulative arc s (len N+1),
    # then count = floor(total/spacing)+1, capped so target (count-1)*spacing < total
    # and to n_max. Mirrors geometry._resample_one's spacing branch.
    e = wp.tid()
    b = e * N
    es = e * (N + 1)
    s[es] = float(0.0)
    acc = wp.float64(0.0)
    for i in range(N):
        d = c[b + (i + 1) % N] - c[b + i]
        l = wp.length(d)
        seg[b + i] = l
        acc = acc + wp.float64(l)
        s[es + i + 1] = wp.float32(acc)
    total = wp.float32(acc)
    if not (total > 0.0):      # catches total <= 0 AND NaN (never-accepted envs)
        count[e] = 0
        return
    k = int(wp.floor(total / spacing)) + 1
    while k > 1 and wp.float32(k - 1) * spacing >= total:
        k = k - 1
    count[e] = wp.min(wp.max(k, 1), n_max)

@wp.kernel
def _cs_lookup_k(c: wp.array(dtype=wp.vec2f), seg: wp.array(dtype=wp.float32),
                 s: wp.array(dtype=wp.float32), N: int, spacing: wp.float32, n_max: int,
                 count: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
    # One thread per OUTPUT slot t (dim = E*n_max). k >= count[e] -> NaN pad.
    t = wp.tid()
    e = t // n_max
    k = t % n_max
    if k >= count[e]:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return
    eb = e * N
    es = e * (N + 1)
    target = wp.float32(k) * spacing
    idx = int(0)
    while idx < N - 1 and s[es + idx + 1] < target:
        idx = idx + 1
    s0 = s[es + idx]
    segl = wp.max(seg[eb + idx], float(1.0e-12))
    frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))
    p0 = c[eb + idx]
    p1 = c[eb + (idx + 1) % N]
    out[t] = p0 + frac * (p1 - p0)

@wp.kernel
def _arc_scan_k(
    dense: wp.array(dtype=wp.vec2f),
    M: int,
    num: int,
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
):
    # One thread per env e. NaN-aware generalization of _resample_scan_k: first
    # COMPACT the real (both-components-finite) points IN ORDER (dropping interior
    # NaN too, like the oracle's pe[isfinite.all(-1)]), then build the closed-loop
    # segment lengths and cumulative arc length over those R real points.
    e = wp.tid()
    db = e * M
    rb = e * M
    es = e * (M + 1)

    # Compaction: walk i=0..M-1, append every finite point in order.
    r = int(0)
    for i in range(M):
        p = dense[db + i]
        if wp.isfinite(p[0]) and wp.isfinite(p[1]):
            real_pts[rb + r] = p
            r = r + 1
    count_r[e] = r
    # count_out: R>=2 -> num, else 0.
    count_out[e] = wp.where(r >= 2, num, 0)

    # R >= 2: closed-loop arc length over real_pts[0..R-1]. seg[j] = |real[(j+1)%R]
    # - real[j]| (j=R-1 is the wrap |real[0]-real[R-1]|); s[0]=0, s[j+1]=s[j]+seg[j].
    # Accumulate in float64 to limit accumulation drift.
    if r >= 2:
        s[es] = float(0.0)
        acc = wp.float64(0.0)
        for j in range(r):
            nxt = real_pts[rb + (j + 1) % r]
            l = wp.length(nxt - real_pts[rb + j])
            seg[rb + j] = l
            acc = acc + wp.float64(l)
            s[es + j + 1] = wp.float32(acc)
    # R < 2: leave seg/s untouched; _arc_lookup_k emits NaN for this env.

@wp.kernel
def _arc_lookup_k(
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    M: int,
    num: int,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per OUTPUT point t (dim = E*num); e = env index, k = point-within-env.
    # R = count_r[e]. R < 2 -> NaN (matches _resample_one's degenerate full-NaN row).
    # Else: linear-scan searchsorted(right=False).clamp(max=R-1) over the closed
    # real-loop arc length, then lerp. Mirrors _resample_lookup_k with variable R and
    # the wrap p1 = real[0] for idx == R-1 ((idx+1)%R).
    t = wp.tid()
    e = t // num
    k = t % num
    rb = e * M
    es = e * (M + 1)

    r = count_r[e]
    if r < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    total = s[es + r]
    target = float(k) * total / float(num)

    # Linear scan: first j in [0, R-1] with s[es+j+1] >= target, clamped to R-1.
    idx = int(0)
    while idx < r - 1 and s[es + idx + 1] < target:
        idx = idx + 1

    s0 = s[es + idx]
    segl = wp.max(seg[rb + idx], float(1.0e-12))
    frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))

    p0 = real_pts[rb + idx]
    p1 = real_pts[rb + (idx + 1) % r]   # closed[idx+1]: real[0] when idx == R-1
    out[t] = p0 + frac * (p1 - p0)

@wp.func
def _turning_func(c: wp.array(dtype=wp.vec2f), base: int, N: int) -> float:
    # Signed total turning of the closed polygon whose points start at `base`.
    # Edge angle theta_i = atan2(d_i.y, d_i.x) for raw edge d_i = c[(i+1)%N] - c[i]
    # (atan2 is scale-invariant, so no normalization is needed; the zero-length
    # edge gives atan2(0,0)=0, matching the oracle's safe_normalize-then-atan2).
    # Per edge, dtheta = theta_i - theta_{i-1} wrapped into (-pi, pi] via
    # atan2(sin, cos); the sum over all edges is the turning number.
    total = float(0.0)
    for i in range(N):
        di = c[base + (i + 1) % N] - c[base + i]
        ip = (i + N - 1) % N
        dp = c[base + (ip + 1) % N] - c[base + ip]
        theta_i = wp.atan2(di[1], di[0])
        theta_prev = wp.atan2(dp[1], dp[0])
        dth = theta_i - theta_prev
        total = total + wp.atan2(wp.sin(dth), wp.cos(dth))
    return total

@wp.kernel
def _turning_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Delegates to _turning_func over [e*n_max, e*n_max+count[e]).
    e = wp.tid()
    out[e] = _turning_func(c, e * n_max, count[e])

@wp.kernel
def _validity_k(
    center: wp.array(dtype=wp.vec2f),
    w: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    gen_valid: wp.array(dtype=wp.int32),
    outer: wp.array(dtype=wp.vec2f),
    inner: wp.array(dtype=wp.vec2f),
    has_border: int,
    n_max: int,
    half_width: float,
    turning_tol: float,
    w_floor: float,
    relax_tol: float,
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. Fuses inflation._validity_stage entirely in-kernel:
    # gen_valid AND turning AND width-floor AND no-NaN AND thickness AND border-simple.
    # All sub-results are 0/1 int flags (Warp can't AND Python bools in dynamic loops).
    # count[e] real points; NaN padding beyond is ignored in all sub-checks.
    e = wp.tid()
    cnt = count[e]
    base = e * n_max

    if cnt == 0:
        out[e] = 0
        return

    # --- turning (over cnt real points only) ---
    turn = _turning_func(center, base, cnt)
    turn_ok = int(0)
    if wp.abs(wp.abs(turn) - 2.0 * wp.pi) <= turning_tol:
        turn_ok = int(1)

    # --- real-point mask (i < cnt): width floor + no NaN over real points ---
    w_ok = int(1)
    no_nan = int(1)
    for i in range(cnt):
        if not (w[base + i] > w_floor):
            w_ok = int(0)
        ci = center[base + i]
        if not (wp.isfinite(ci[0]) and wp.isfinite(ci[1])):
            no_nan = int(0)

    # --- separation band from the mean segment length (perimeter over cnt real pts) ---
    peri = float(0.0)
    for i in range(cnt):
        peri += wp.length(center[base + (i + 1) % cnt] - center[base + i])
    band = _separation_band(peri / float(cnt), 2.0 * half_width)

    # --- thickness gate (over cnt real points only) ---
    th = _thickness_func(center, base, cnt, band)
    th_ok = int(0)
    if th >= (1.0 - relax_tol) * half_width:
        th_ok = int(1)

    # --- border self-intersection gate (skipped when has_border == 0; cnt real pts) ---
    border_ok = int(1)
    if has_border == 1:
        cross = _self_intersections_func(outer, base, cnt) + _self_intersections_func(inner, base, cnt)
        if cross != 0:
            border_ok = int(0)

    # --- generation flag ---
    gv = int(0)
    if gen_valid[e] != 0:
        gv = int(1)

    out[e] = gv & turn_ok & w_ok & no_nan & th_ok & border_ok

@wp.kernel
def _arclength_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
):
    # One thread per env e. Count-aware arc length:
    # Only count[e] real points are used; padding slots i in [count[e], n_max) get NaN.
    # seg_len[i] = |c[b+(i+1)%count[e]] - c[b+i]| for i in [0, count[e]):
    #   i=count[e]-1 is the closing wrap segment (last real pt -> first pt).
    # arclen[b+i] = cumulative length BEFORE segment i (arclen[b+0]=0).
    # length[e] = full closed perimeter (all count[e] segments including wrap).
    # Running sum in float64 to limit accumulation drift.
    # PARITY: when count[e]==n_max for all e, bit-identical to the former fixed-N path.
    e = wp.tid()
    cn = count[e]
    b = e * n_max
    acc = wp.float64(0.0)
    for i in range(cn):
        arclen[b + i] = wp.float32(acc)            # arc length BEFORE segment i
        d = c[b + (i + 1) % cn] - c[b + i]
        acc = acc + wp.float64(wp.length(d))       # add segment i (i=cn-1 is the wrap)
    length[e] = wp.float32(acc)
    # NaN-pad slots beyond the real count
    for i in range(cn, n_max):
        arclen[b + i] = wp.nan

@wp.kernel
def _corner_sample_k(
    seeds: wp.array(dtype=wp.int32),
    attempt: int,
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: float,
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # ACCEPTED RNG REDESIGN (does NOT match the oracle _sample_corner_points
    # bit-for-bit; validated by structural properties only). One thread per env e.
    #
    # Seeding: state = wp.rand_init(seeds[e] * 9781 + attempt) -> reproducible per
    # (env, attempt). 9781 is a large odd multiplier so distinct env seeds map to
    # well-separated rand_init states.
    #
    # Draw ORDER per corner c (fixed so a corner's noise is deterministic given its
    # retries): for each duplicate-rejection retry draw ONE cell-selection randf,
    # then once a cell is accepted draw the two noise randfs (nx, ny) in that order.
    #
    # Dedup: a corner's cell is redrawn (up to 8 tries) if it collides with any cell
    # already chosen for an EARLIER corner of this env, preserving the distinct-cell
    # spread the oracle's top-k subset gave. The per-thread chosen-cell history lives
    # in the scratch buffer used[e*P + 0 .. e*P + c] (pre-filled with -1 by the
    # wrapper); after the retry budget we accept whatever cell we have.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 9781 + attempt)
    base = e * P
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        # Bounded duplicate rejection against earlier corners of this env.
        # dup is an int flag (Warp can't mutate a Python bool in a dynamic loop).
        for _retry in range(8):
            dup = int(0)
            for k in range(c):
                if used[base + k] == cell:
                    dup = int(1)
            if dup == 0:
                break
            cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        used[base + c] = cell

        x = float(cell % num_cells)
        y = float(cell // num_cells)
        nx = wp.randf(state) - 0.5            # [0,1) -> [-0.5, 0.5)
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * scale,
                                 (y * cell_size + ny) * scale)

@wp.kernel
def _ccw_sort_k(
    points: wp.array(dtype=wp.vec2f),
    P: int,
    count: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Orders this env's FIRST m = count[e] corners ascending by
    # the centroid-relative angle key = atan2(dx, dy) (X FIRST), about the centroid of
    # those m corners; rows [m, P) are written NaN (the pruned tail). m == P sorts all P
    # slots with no NaN tail. The insertion sort reads only slots behind
    # its write frontier, so the uninitialised keys/out scratch is never consumed.
    e = wp.tid()
    base = e * P
    m = count[e]
    if m < 1:
        m = 1
    if m > P:
        m = P

    # Centroid over the first m corners (float64 for precision).
    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(P):
        if i < m:
            p = points[base + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])
    cx = wp.float32(sx / wp.float64(m))
    cy = wp.float32(sy / wp.float64(m))

    for c in range(P):
        if c < m:
            p = points[base + c]
            key = wp.atan2(p[0] - cx, p[1] - cy)   # X first!
            j = c - 1
            while j >= 0 and keys[base + j] > key:
                keys[base + j + 1] = keys[base + j]
                out[base + j + 1] = out[base + j]
                j = j - 1
            keys[base + j + 1] = key
            out[base + j + 1] = p
        else:
            out[base + c] = wp.vec2f(wp.nan, wp.nan)

@wp.func
def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    # Mirrors geometry.safe_normalize: v / clamp_min(||v||, 1e-8). The wp.max
    # floors finite lengths at 1e-8 and a NaN vector divides to (nan, nan)
    # so NaN propagates to BOTH components (matching oracle behavior on pruned corners).
    return v / wp.max(wp.length(v), 1.0e-8)

@wp.func
def _pruned_corner(c: wp.array(dtype=wp.vec2f), b: int, i: int, cnt: int) -> wp.vec2f:
    # Folds in _prune_corners' NaN step: corner i of the env at base b is real iff
    # i < cnt; rows i >= cnt are replaced by (nan, nan) (same NaN positions as the
    # oracle's where(arange(P) < count, corners, nan) prune).
    ci = c[b + i]
    return wp.where(i < cnt, ci, wp.vec2f(wp.nan, wp.nan))

@wp.kernel
def _vertex_tangents_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    P: int,
    p: float,
    tangents: wp.array(dtype=wp.vec2f),
    scale: wp.array(dtype=wp.float32),
):
    # One thread per corner t (dim = E*P); e = env index, i = corner-within-env.
    # Mirrors geometry.vertex_tangents: u_out_i = dir(i -> i+1), u_in_i = dir(i-1 -> i)
    # (== roll(u_out, +1)); tangent_i = safe_normalize(p*u_out + (1-p)*u_in). The
    # count->NaN prune is folded in-kernel (corner i is NaN iff i >= count[e]); NaN at
    # any pruned corner propagates the same way the oracle's safe_normalize does.
    t = wp.tid()
    e = t // P
    i = t % P
    b = e * P
    cnt = count[e]
    cm = wp.max(cnt, 1)                       # guard %0 for degenerate (cnt==0) envs
    c_i = _pruned_corner(c, b, i, cnt)
    # Wrap mod cnt (not P): a real corner's circular neighbours are real corners, so the
    # closing edge (corner cnt-1 -> corner 0) is a genuine segment. (Mod-P-with-NaN would
    # poison the seam tangents and drop the first/last corner + close with a straight chord.)
    c_next = _pruned_corner(c, b, (i + 1) % cm, cnt)
    c_prev = _pruned_corner(c, b, (i + cm - 1) % cm, cnt)
    u_out = _safe_normalize2(c_next - c_i)
    u_in = _safe_normalize2(c_i - c_prev)
    blended = p * u_out + (1.0 - p) * u_in
    tangents[t] = _safe_normalize2(blended)
    # Per-corner scale = shorter incident edge; the adaptive handle clamp caps handles by it.
    scale[t] = wp.min(wp.length(c_next - c_i), wp.length(c_i - c_prev))

@wp.kernel
def _assemble_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    tangents: wp.array(dtype=wp.vec2f),
    scale: wp.array(dtype=wp.float32),
    P: int,
    npseg: int,
    rad: float,
    clamp_frac: float,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per dense sample t (dim = E*P*npseg). Decodes (e, segment i, sample s),
    # rebuilds the cubic Bezier of segment i (corner i -> corner (i+1)%P) and evaluates
    # it at parameter u = s/(npseg-1) with the degree-3 Bernstein basis. Mirrors
    # BezierCenterlineGenerator._segment + _cubic_bezier: handle = rad*chord along the
    # corner tangents. The count->NaN prune is folded in-kernel (corner i is NaN iff
    # i >= count[e]); NaN corners/tangents propagate into the output as in the oracle.
    t = wp.tid()
    per_env = P * npseg
    e = t // per_env
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * P
    cnt = count[e]
    cm = wp.max(cnt, 1)                       # guard %0 for degenerate (cnt==0) envs

    c0 = _pruned_corner(c, b, i, cnt)
    # Closing segment i == cnt-1 wraps to corner 0 -> a real cubic Bezier (not the old
    # straight chord). Segments i >= cnt keep c0 = NaN and drop out via the resample.
    inext = (i + 1) % cm
    c1 = _pruned_corner(c, b, inext, cnt)
    t0 = tangents[b + i]
    t1 = tangents[b + inext]

    chord = wp.length(c1 - c0)
    # F2: clamp each end's handle by clamp_frac * (that corner's shorter incident edge),
    # so a long handle can't overshoot past a nearby corner and self-cross.
    h0 = wp.min(rad * chord, clamp_frac * scale[b + i])
    h1 = wp.min(rad * chord, clamp_frac * scale[b + inext])
    p1 = c0 + t0 * h0    # leave c0 along its tangent
    p2 = c1 - t1 * h1    # arrive at c1 along its tangent

    u = float(s) / float(npseg - 1)
    omu = 1.0 - u
    b0 = omu * omu * omu          # (1-u)^3
    b1 = 3.0 * u * omu * omu      # 3u(1-u)^2
    b2 = 3.0 * u * u * omu        # 3u^2(1-u)
    b3 = u * u * u                # u^3
    out[t] = b0 * c0 + b1 * p1 + b2 * p2 + b3 * c1

@wp.kernel
def _corner_angles_gate_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    P: int,
    min_angle: float,
    ok: wp.array(dtype=wp.int32),
):
    # One thread per env e. Reproduces generate()'s ANGLE gate over this env's P RAW
    # corners with the _prune_corners NaN step folded in: corner i is REAL iff
    # i < count[e] AND both components finite. angle_ok = ((angle > min_angle) |
    # ~constrained) over all corners; a corner is "constrained" only when i and BOTH
    # its circular neighbours i-1, i+1 are real; unconstrained corners are skipped
    # (mirrors the oracle's nan_to_num(angle, 0) | ~constrained passing them). For each
    # constrained corner: interior angle = pi - acos(clamp(dot(u_in, u_out))),
    # u_in = safe_normalize(c_i - c_prev), u_out = safe_normalize(c_next - c_i), with the
    # same [-1+1e-7, 1-1e-7] cos clamp. If any constrained corner fails (not > min_angle),
    # the env's flag is 0.
    e = wp.tid()
    b = e * P
    cnt = count[e]
    flag = int(1)
    for i in range(P):
        ip = (i + P - 1) % P
        inx = (i + 1) % P
        ci = _pruned_corner(c, b, i, cnt)
        cp = _pruned_corner(c, b, ip, cnt)
        cn = _pruned_corner(c, b, inx, cnt)
        real_i = wp.isfinite(ci[0]) and wp.isfinite(ci[1])
        real_p = wp.isfinite(cp[0]) and wp.isfinite(cp[1])
        real_n = wp.isfinite(cn[0]) and wp.isfinite(cn[1])
        if real_i and real_p and real_n:
            u_in = _safe_normalize2(ci - cp)
            u_out = _safe_normalize2(cn - ci)
            cos = wp.clamp(wp.dot(u_in, u_out), -1.0 + 1.0e-7, 1.0 - 1.0e-7)
            angle = wp.pi - wp.acos(cos)
            if not (angle > min_angle):
                flag = int(0)
    ok[e] = flag

@wp.kernel
def _gates_combine_k(
    angle_ok: wp.array(dtype=wp.int32),
    turn: wp.array(dtype=wp.float32),
    cnt_turn: wp.array(dtype=wp.int32),
    cross_simple: wp.array(dtype=wp.int32),
    turning_tol: float,
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. Fuses generate()'s gate conjunction:
    #   turn_ok   = |(|turn| - 2*pi)| <= turning_tol
    #   finite_ok = (cnt_turn >= 2) and isfinite(turn)
    #   simple_ok = (cross_simple == 0)
    #   out       = angle_ok & turn_ok & finite_ok & simple_ok   (int 0/1 flags)
    # cnt_turn is the [E] count returned by arc_length_resample_warp(dense, npseg).
    e = wp.tid()
    tu = turn[e]
    turn_ok = int(0)
    if wp.abs(wp.abs(tu) - 2.0 * wp.pi) <= turning_tol:
        turn_ok = int(1)
    finite_ok = int(0)
    if cnt_turn[e] >= 2 and wp.isfinite(tu):
        finite_ok = int(1)
    simple_ok = int(0)
    if cross_simple[e] == 0:
        simple_ok = int(1)
    out[e] = angle_ok[e] & turn_ok & finite_ok & simple_ok

@wp.kernel
def _corner_count_sample_k(
    seeds: wp.array(dtype=wp.int32),
    attempt: int,
    min_num: int,
    max_num: int,
    out: wp.array(dtype=wp.int32),
):
    # ACCEPTED RNG REDESIGN (does NOT match the oracle's per-env corner-count draw
    # bit-for-bit; validated by range/reproducibility only). One thread per env e.
    #
    # Seeding: state = wp.rand_init(seeds[e] * 6151 + attempt). The 6151 multiplier is
    # DISTINCT from corner_sample's 9781 so the count stream and the corner-position
    # stream stay decorrelated (different rand_init states for the same (seed, attempt)).
    #
    # Draw: a single uniform randf in [0, 1) maps to an inclusive integer count in
    # [min_num, max_num] via floor(randf * range) where range = max_num - min_num + 1,
    # then clamp to max_num to fold the measure-zero randf == 1.0 edge back in range.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 6151 + attempt)
    span = max_num - min_num + 1
    count = min_num + int(wp.randf(state) * float(span))
    out[e] = wp.min(count, max_num)

@wp.kernel
def _fill_f32_k(arr: wp.array(dtype=wp.float32), v: float):
    # One thread per element: constant float fill.
    arr[wp.tid()] = v

@wp.kernel
def _fill_i32_k(arr: wp.array(dtype=wp.int32), v: int):
    # One thread per element: constant int fill.
    arr[wp.tid()] = v

@wp.kernel
def _select_vec2_k(rs: wp.array(dtype=wp.vec2f), rs_poly: wp.array(dtype=wp.vec2f),
                   crossers: wp.array(dtype=wp.int32), N: int, out: wp.array(dtype=wp.vec2f)):
    # One thread per point t (dim=E*N). e = env index.
    # Selects rs_poly[t] if crossers[e] > 0, else rs[t] (polygon fallback for self-crossers).
    t = wp.tid()
    e = t // N
    if crossers[e] > 0:
        out[t] = rs_poly[t]
    else:
        out[t] = rs[t]

@wp.kernel
def _select_first_valid_k(
    accept: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    rs: wp.array(dtype=wp.vec2f),
    centerline: wp.array(dtype=wp.vec2f),
    N: int,
):
    # One thread per POINT t (dim = E*N); e = env index. Accept-FIRST-valid take:
    # write the freshly-resampled rs into centerline ONLY for envs newly accepted this
    # attempt (accept[e]==1 AND the OLD valid[e]==0). Reads valid but never writes it
    # (no race); _or_update_k updates valid AFTER so the take here saw the old flag.
    t = wp.tid()
    e = t // N
    if accept[e] == 1 and valid[e] == 0:
        centerline[t] = rs[t]

@wp.kernel
def _or_update_k(accept: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32)):
    # One thread per env e. valid |= accept (run AFTER _select_first_valid_k so the
    # select saw the OLD valid).
    e = wp.tid()
    if accept[e] != 0:
        valid[e] = 1

@wp.kernel
def _band_l0_k(
    center: wp.array(dtype=wp.vec2f),
    n_max: int,
    two_hw: float,
    band_out: wp.array(dtype=wp.int32),
    l0_out: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
):
    # One thread per env e. Count-aware: loop over count[e] real points, base e*n_max,
    # wrap index (i+1)%count[e]. L0 = perimeter/count[e] (mean segment length). The band
    # is _separation_band(L0, 2*hw); L0 itself may stay NaN for invalid (NaN-centerline)
    # envs (that flows untouched into xpbd, which propagates the NaN), while the band's
    # isfinite guard maps such envs to band 1.
    # PARITY: when count[e]==n_max for all e, produces identical output to the former
    # fixed-N kernel (same loop bounds, same formula).
    e = wp.tid()
    base = e * n_max
    cn = count[e]
    peri = float(0.0)
    for i in range(cn):
        peri += wp.length(center[base + (i + 1) % cn] - center[base + i])
    l0 = peri / float(cn)
    l0_out[e] = l0
    band_out[e] = _separation_band(l0, two_hw)

def corner_count_sample_inplace(seeds_wp, attempt, config, out_count):
    """In-place: writes per-env corner counts into out_count ([E] int32 wp.array). Zero alloc."""
    _init()
    E = out_count.shape[0]
    wp.launch(_corner_count_sample_k, dim=E,
              inputs=[seeds_wp, int(attempt), int(config.min_num_points),
                      int(config.max_num_points), out_count],
              device=str(out_count.device))
    _sync(out_count.device)


def corner_sample_inplace(seeds_wp, attempt, config, out_corners, used_scratch):
    """In-place: writes [E*P] vec2f into out_corners. used_scratch is [E*P] int32 scratch.
    Zero alloc."""
    _init()
    E = seeds_wp.shape[0]
    P = int(config.max_num_points)
    num_cells = int(1.0 / (config.min_point_distance * 2))
    nc2 = num_cells * num_cells
    cell_size = config.min_point_distance * 2.0
    dev = str(out_corners.device)
    # Fill scratch with -1 (dedup init)
    wp.launch(_fill_i32_k, dim=E * P, inputs=[used_scratch, -1], device=dev)
    wp.launch(_corner_sample_k, dim=E,
              inputs=[seeds_wp, int(attempt), num_cells, nc2, float(cell_size),
                      float(config.scale), P, used_scratch, out_corners],
              device=dev)
    _sync(out_corners.device)


def ccw_sort_inplace(corners_wp, count_wp, keys_scratch, out_wp, P):
    """In-place: writes sorted corners into out_wp ([E*P] vec2f). keys_scratch is [E*P] float32.
    Zero alloc."""
    _init()
    dev = str(out_wp.device)
    E = count_wp.shape[0]
    wp.launch(_ccw_sort_k, dim=E,
              inputs=[corners_wp, P, count_wp, keys_scratch, out_wp],
              device=dev)
    _sync(out_wp.device)


def assemble_inplace(corners_wp, count_wp, config, tan_scratch, scale_scratch, out_wp):
    """In-place: writes dense [E*P*npseg] vec2f into out_wp. Zero alloc.
    tan_scratch: [E*P] vec2f; scale_scratch: [E*P] float32."""
    _init()
    E = count_wp.shape[0]
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    assert npseg >= 2
    p = math.atan(config.edgy) / math.pi + 0.5
    clamp_frac = float(getattr(config, "handle_clamp_frac", 1.0e9))
    dev = str(out_wp.device)
    wp.launch(_vertex_tangents_k, dim=E * P,
              inputs=[corners_wp, count_wp, P, float(p), tan_scratch, scale_scratch],
              device=dev)
    wp.launch(_assemble_k, dim=E * P * npseg,
              inputs=[corners_wp, count_wp, tan_scratch, scale_scratch,
                      P, npseg, float(config.rad), clamp_frac, out_wp],
              device=dev)
    _sync(out_wp.device)


def self_intersections_inplace(poly_wp, count_wp, out_wp, n_max):
    """In-place: writes [E] int32 crossing counts into out_wp. Zero alloc.
    poly_wp: [E*n_max] vec2f; count_wp: [E] int32; out_wp: [E] int32."""
    _init()
    E = count_wp.shape[0]
    dev = str(out_wp.device)
    wp.launch(_self_intersections_k, dim=E,
              inputs=[poly_wp, n_max, count_wp, out_wp],
              device=dev)
    _sync(out_wp.device)


def _arc_resample_inplace(points_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp, out_wp, dev):
    """In-place NaN-aware arc-length resample. All args are wp.arrays. Zero alloc.

    points_wp: [E*M] vec2f input; real_wp: [E*M] vec2f scratch; seg_wp: [E*M] float32 scratch;
    s_wp: [E*(M+1)] float32 scratch; count_r_wp: [E] int32 scratch; count_out_wp: [E] int32 output;
    out_wp: [E*num] vec2f output; dev: warp device string.
    """
    E = points_wp.shape[0] // M
    wp.launch(_arc_scan_k, dim=E,
              inputs=[points_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp], device=dev)
    wp.launch(_arc_lookup_k, dim=E * num,
              inputs=[real_wp, seg_wp, s_wp, count_r_wp, M, num, out_wp], device=dev)


def arc_length_resample_inplace(
    points_wp: "wp.array",
    M: int,
    num: int,
    real_wp: "wp.array",
    seg_wp: "wp.array",
    s_wp: "wp.array",
    count_r_wp: "wp.array",
    count_out_wp: "wp.array",
    out_wp: "wp.array",
    dev: str,
) -> None:
    """Public in-place NaN-aware arc-length resample. All args are wp.arrays. Zero alloc.

    Identical to _arc_resample_inplace (which remains for internal use); this public name
    allows tests + the standalone generate path to call it without using the private alias.

    points_wp:   [E*M] vec2f input (may contain NaN).
    real_wp:     [E*M] vec2f scratch (compacted real points).
    seg_wp:      [E*M] float32 scratch (per-segment lengths).
    s_wp:        [E*(M+1)] float32 scratch (cumulative arc length).
    count_r_wp:  [E] int32 scratch (real-point count per env).
    count_out_wp:[E] int32 output (R>=2 -> num, else 0).
    out_wp:      [E*num] vec2f output.
    dev:         Warp device string.
    """
    _init()
    _arc_resample_inplace(points_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp, out_wp, dev)
    _sync(dev)


def turning_number_inplace(
    center_wp: "wp.array",
    n_max: int,
    count_wp: "wp.array",
    out_wp: "wp.array",
) -> None:
    """In-place signed total turning of each closed polygon. Zero alloc.

    center_wp: [E*n_max] vec2f flat input centerline (NaN-padded beyond count[e]).
    n_max:     int stride per env in the flat array.
    count_wp:  [E] int32 real point count per env.
    out_wp:    [E] float32 output (signed turning in radians per env).

    Pure Warp (cpu+cuda); matches geometry.turning_number to allclose(atol=1e-4).
    """
    _init()
    E = count_wp.shape[0]
    wp.launch(_turning_k, dim=E,
              inputs=[center_wp, n_max, count_wp, out_wp],
              device=str(out_wp.device))
    _sync(out_wp.device)


def generate_centerline_warp(seeds_wp: wp.array, config,
                              out_centerline: wp.array, out_valid_wp: wp.array,
                              scratch: "_Scratch") -> None:
    """Single-pass centerline generation — in-place owned path only.

    Pure-Warp: sample corners -> sort ccw -> assemble Bezier -> arc-resample -> polygon
    fallback (if self-crossing) -> write chosen centerline into out_centerline.
    Marks all envs valid (generation gate is always True; inflate does the real gate).

    Args:
        seeds_wp:      [E] int32 wp.array per-env base seeds.
        config:        TrackGenConfig.
        out_centerline:[E*N] vec2f wp.array — written in-place with the chosen centerline.
        out_valid_wp:  [E] int32 wp.array — filled with 1 (all valid at generation stage).
        scratch:       _Scratch with generation intermediates.
    """
    import dataclasses

    _init()

    assert scratch is not None, "generate_centerline_warp requires scratch"
    E = scratch.gen_count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M = P * npseg  # dense points per env
    dev = str(out_centerline.device)

    # Step 1-3: sample corners, sort
    corner_count_sample_inplace(seeds_wp, 0, config, scratch.gen_count)
    corner_sample_inplace(seeds_wp, 0, config, scratch.gen_corners, scratch.gen_used)
    ccw_sort_inplace(scratch.gen_corners, scratch.gen_count, scratch.gen_keys,
                     scratch.gen_ordered, P)

    # Step 4: assemble Bezier dense -> gen_dense
    assemble_inplace(scratch.gen_ordered, scratch.gen_count, config,
                     scratch.gen_tan, scratch.gen_scale, scratch.gen_dense)

    # Step 5: arc-resample Bezier dense -> gen_rs (N points per env)
    _arc_resample_inplace(scratch.gen_dense, M, N,
                          scratch.gen_arc_real, scratch.gen_arc_seg,
                          scratch.gen_arc_s, scratch.gen_arc_cr,
                          scratch.gen_arc_co, scratch.gen_rs, dev)

    # Step 6: assemble polygon dense -> gen_poly (handle_clamp_frac=0)
    cfg_poly = dataclasses.replace(config, handle_clamp_frac=0.0)
    assemble_inplace(scratch.gen_ordered, scratch.gen_count, cfg_poly,
                     scratch.gen_tan, scratch.gen_scale, scratch.gen_poly)

    # Step 7: arc-resample polygon dense -> out_centerline (temporarily holds rs_poly)
    _arc_resample_inplace(scratch.gen_poly, M, N,
                          scratch.gen_arc_real, scratch.gen_arc_seg,
                          scratch.gen_arc_s, scratch.gen_arc_cr,
                          scratch.gen_arc_co, out_centerline, dev)

    # Step 8: self-intersections of the Bezier resample -> gen_crossers
    # gen_arc_co was written by _arc_scan_k: R>=2 -> N, R<2 -> 0. Pass directly as count.
    self_intersections_inplace(scratch.gen_rs, scratch.gen_arc_co,
                               scratch.gen_crossers, N)

    # Step 9: select: rs if no crossings, rs_poly (= out_centerline temporarily) if crossings.
    # out_centerline aliases rs_poly arg: safe per-thread (each thread reads its own slot
    # then writes it for the crossers>0 branch; reads gen_rs for the else branch).
    wp.launch(_select_vec2_k, dim=E * N,
              inputs=[scratch.gen_rs, out_centerline, scratch.gen_crossers, N, out_centerline],
              device=dev)

    # Step 10: mark all envs valid (gen gate is always True; inflate does the real gate).
    wp.launch(_fill_i32_k, dim=E, inputs=[out_valid_wp, 1], device=dev)

    _sync(dev)


def _run_pipeline(config, seed_buf_wp: wp.array, out: "Track", scratch: "_Scratch") -> "Track":
    """Execute the owned pure-Warp pipeline into pre-allocated buffers. Zero alloc.

    Composes: generate_centerline_warp -> resample_constant_spacing -> band/L0 ->
    xpbd_solve_inplace -> inflate_warp. All buffers are pre-allocated in scratch/out.
    Safe to call inside a wp.ScopedCapture region (no host syncs, no allocations).

    Args:
        config:       TrackGenConfig.
        seed_buf_wp:  [E] int32 wp.array — per-env base seeds.
        out:          Pre-allocated Track (wp.array fields) — written in-place.
        scratch:      Pre-allocated _Scratch — all intermediates written in-place.

    Returns:
        out (the same Track instance).
    """
    E = scratch.gen_count.shape[0]
    dev = str(scratch.gen_centerline.device)
    hw = float(config.half_width)
    n_max = int(config.N_max)

    # 1. Generate centerline in-place.
    generate_centerline_warp(seed_buf_wp, config,
                             out_centerline=scratch.gen_centerline,
                             out_valid_wp=scratch.gen_valid,
                             scratch=scratch)

    # 2. Constant-spacing resample in-place.
    resample_constant_spacing(
        scratch.gen_centerline, float(config.spacing), n_max,
        out_wp=scratch.cs_center, count_wp=scratch.count,
        seg_wp=scratch.cs_seg, s_wp=scratch.cs_s,
    )

    # 3-4. Relaxation (band/L0 setup + XPBD solve) in-place, then choose its input to
    # inflate. relax_enable=False is an identity pass-through (matches the oracle's
    # `if not relax_enable: return center`): skip the relax band/L0 + solve and inflate
    # the constant-spacing centerline directly. The branch is resolved at capture time,
    # so the captured graph is fixed and allocation-free either way.
    if config.relax_enable:
        # 3. Band / L0 computation in-place.
        wp.launch(_band_l0_k, dim=E, inputs=[scratch.cs_center, n_max, 2.0 * hw,
                  scratch.band, scratch.L0, scratch.count], device=dev)
        if config.relax_band is not None:
            wp.launch(_fill_i32_k, dim=E, inputs=[scratch.band,
                      int(config.relax_band)], device=dev)
        _sync(dev)

        # 4. XPBD relaxation in-place. Thread the pipeline's capture state in explicitly
        # so warp_relax decides its host-sync without reaching back into this module.
        warp_relax.xpbd_solve_inplace(
            scratch.cs_center, scratch.relaxed, scratch.xpbd_db,
            scratch.band, scratch.L0, scratch.count, n_max, config,
            capturing=_CAPTURING,
        )
        relax_out = scratch.relaxed
    else:
        relax_out = scratch.cs_center

    # 5. Inflate (resample_uniform + frame + offset + validity) in-place.
    return inflate_warp(
        relax_out, config, out=out, valid=scratch.gen_valid,
        count=scratch.count, scratch=scratch,
    )


def offset(center, Nrm, half_width, out_outer, out_inner, area_a, area_b, count):
    """In-place: writes outer/inner into out_outer/out_inner (wp.array [E*n_max] vec2f),
    using area_a/area_b (wp.array [E] f32) scratch. All args are wp.array; nothing
    allocated. center/Nrm: wp.array [E*n_max] vec2f. count: wp.array [E] int32.
    half_width: float. n_max is inferred from out_outer.shape[0] // count.shape[0].

    Pure Warp (cpu+cuda); allclose to oracle to atol=1e-5.
    """
    _init()
    E = count.shape[0]
    flat = out_outer.shape[0]
    n_max = flat // E
    area_a.zero_()
    area_b.zero_()
    wp.launch(_offset_build_k, dim=flat,
              inputs=[center, Nrm, float(half_width), n_max, area_a, area_b, count],
              device=str(out_outer.device))
    wp.launch(_offset_assign_k, dim=flat,
              inputs=[center, Nrm, float(half_width), n_max, area_a, area_b,
                      out_outer, out_inner, count],
              device=str(out_outer.device))
    _sync(out_outer.device)


def frame_curvature(
    center_wp: "wp.array",
    out_T: "wp.array",
    out_Nrm: "wp.array",
    kappa_scratch: "wp.array",
    count_wp: "wp.array",
):
    """In-place: writes T/Nrm into out_T/out_Nrm (wp.array [E*n_max] vec2f);
    kappa_scratch (wp.array [E*n_max] float32) receives kappa (unused by pipeline).
    All args are wp.array; nothing allocated. n_max inferred from out_T.shape[0]
    and count_wp.shape[0] (== E).

    Pure Warp (cpu+cuda); allclose to oracle to atol=1e-5.
    """
    _init()
    E = count_wp.shape[0]
    flat = out_T.shape[0]
    n_max = flat // E
    wp.launch(_frame_k, dim=flat,
              inputs=[center_wp, n_max, out_T, out_Nrm, kappa_scratch, count_wp],
              device=str(out_T.device))
    _sync(out_T.device)


def resample_uniform(
    center_wp: "wp.array",
    out_wp: "wp.array",
    n: int,
    count_wp: "wp.array",
    seg_wp: "wp.array | None" = None,
    s_wp: "wp.array | None" = None,
    device: str = "cpu",
) -> None:
    """Arc-length-uniform resample of each closed loop to n points — in-place.

    Matches track_gen.relaxation._resample_uniform within FP tolerance (~1e-4).
    Two Warp kernels: scan (one thread per env, builds seg+cumulative s from points)
    then lookup (one thread per output point, linear-scan searchsorted + lerp).

    The owned pipeline path (``seg_wp`` and ``s_wp`` pre-allocated) is zero-alloc and
    safe inside a CUDA graph capture region. When ``seg_wp`` or ``s_wp`` is ``None``
    those arrays are allocated internally (not zero-alloc).

    Args:
        center_wp: [E*n_max] wp.vec2f flat input centerline.
        out_wp:    [E*n_max] wp.vec2f flat output buffer (written in-place).
        n:         output point count per env (== n_max, the buffer stride).
        count_wp:  [E] wp.int32 real point count per env.
        seg_wp:    [E*n_max] wp.float32 scan scratch (allocated internally if None).
        s_wp:      [E*(n_max+1)] wp.float32 scan scratch (allocated internally if None).
        device:    Warp device string (e.g. "cpu", "cuda:0").

    PARITY INVARIANT: count=full((E,), N) reproduces the former fixed-N behaviour
    bit-exactly.  Pure Warp (cpu+cuda), zero compute outside of Warp kernels.
    """
    _init()
    E = count_wp.shape[0]
    n_max = n
    flat = E * n_max

    if seg_wp is None:
        seg_wp = wp.empty(flat, dtype=wp.float32, device=device)
    if s_wp is None:
        s_wp = wp.empty(E * (n_max + 1), dtype=wp.float32, device=device)

    wp.launch(_resample_scan_k, dim=E,
              inputs=[center_wp, n_max, count_wp, seg_wp, s_wp], device=device)
    wp.launch(_resample_lookup_k, dim=flat,
              inputs=[center_wp, seg_wp, s_wp, n_max, count_wp, out_wp], device=device)


def resample_constant_spacing(
    center: "wp.array",
    spacing: float,
    n_max: int,
    out_wp: "wp.array | None" = None,
    count_wp: "wp.array | None" = None,
    seg_wp: "wp.array | None" = None,
    s_wp: "wp.array | None" = None,
) -> "tuple[wp.array, wp.array] | None":
    """Arc-length resample each fully-real closed loop to constant ``spacing``, padded to
    ``n_max`` with NaN.  Matches ``geometry.arc_length_resample(points, spacing=spacing,
    n_max=n_max)``. Pure Warp (cpu+cuda).

    ``center`` must be a ``wp.array [E*N] vec2f`` flat.

    In-place mode (all four pre-allocated buffers provided):
        ``out_wp``   — [E*n_max] wp.vec2f written in-place (center output).
        ``count_wp`` — [E] wp.int32 written in-place (real point count per env).
        ``seg_wp``   — [E*N] wp.float32 scan scratch.
        ``s_wp``     — [E*(N+1)] wp.float32 scan scratch.
        Returns ``None`` (caller reads out_wp / count_wp directly). This path is
        zero-alloc and safe inside a CUDA graph capture region.

    Allocating-mode (any buffer omitted -> allocated internally):
        Returns ``(out_wp, count_wp)`` — both freshly-allocated wp.arrays. This
        branch allocates; do not use inside a CUDA graph capture region.

    ``count[e]`` is capped at ``n_max`` (required for fixed-buffer sizing). If the
    true point count hits N_max, a ``RuntimeWarning`` is emitted on the non-capture
    path (the cap is silent during CUDA graph replay). The caller MUST choose
    ``N_max >= max(perimeter) / spacing + 1`` across all envs to avoid truncation."""
    _init()
    if count_wp is not None:
        E = count_wp.shape[0]
    elif out_wp is not None:
        E = out_wp.shape[0] // n_max
    else:
        raise ValueError(
            "resample_constant_spacing: need count_wp or out_wp to infer E from wp.array input")
    N = center.shape[0] // E
    dev = str(center.device)

    _allocating = out_wp is None or count_wp is None
    if out_wp is None:
        out_wp = wp.empty(E * n_max, dtype=wp.vec2f, device=dev)
    if count_wp is None:
        count_wp = wp.empty(E, dtype=wp.int32, device=dev)
    if seg_wp is None:
        seg_wp = wp.empty(E * N, dtype=wp.float32, device=dev)
    if s_wp is None:
        s_wp = wp.empty(E * (N + 1), dtype=wp.float32, device=dev)

    wp.launch(_cs_scan_k, dim=E, inputs=[center, N, float(spacing), n_max,
              seg_wp, s_wp, count_wp], device=dev)
    wp.launch(_cs_lookup_k, dim=E * n_max, inputs=[center, seg_wp, s_wp, N,
              float(spacing), n_max, count_wp, out_wp], device=dev)
    _sync(dev)

    # N_max truncation warning: only on the non-capture path (a host readback is
    # ILLEGAL during CUDA graph capture and unnecessary on replay — count is fixed).
    # This is the ONE justified host readback in _src; gated off during capture.
    if not _CAPTURING:
        import warnings
        import numpy as _np
        max_count = int(_np.max(count_wp.numpy()))
        if max_count >= n_max:
            warnings.warn(
                f"constant_spacing: a track's point count hit N_max={n_max} "
                f"(spacing={spacing}); it was truncated — increase N_max to avoid "
                f"truncation. (CUDA graph replay cannot re-check.)",
                RuntimeWarning,
                stacklevel=2,
            )

    if _allocating:
        return out_wp, count_wp


def validity_inplace(
    center_wp: "wp.array",
    w_wp: "wp.array",
    count_wp: "wp.array",
    gen_valid_wp: "wp.array",
    outer_wp: "wp.array",
    inner_wp: "wp.array",
    has_border: int,
    n_max: int,
    out_valid: "wp.array",
    config,
) -> None:
    """In-place validity gate: writes directly into out_valid ([E] int32 wp.array). Zero alloc.

    All arguments are wp.arrays. has_border=0 means the border self-intersection check is
    skipped (outer_wp/inner_wp are ignored by the kernel). n_max must be passed explicitly.

    Args:
        center_wp:    [E*n_max] vec2f resampled centerline (flat).
        w_wp:         [E*n_max] float32 per-point half-width (flat).
        count_wp:     [E] int32 real-point count per env.
        gen_valid_wp: [E] int32 generation flag (0/1).
        outer_wp:     [E*n_max] vec2f outer border (flat); ignored when has_border==0.
        inner_wp:     [E*n_max] vec2f inner border (flat); ignored when has_border==0.
        has_border:   int flag (1 -> run border self-intersection check, 0 -> skip).
        n_max:        int stride per env in the flat arrays.
        out_valid:    [E] int32 wp.array to write into (e.g. out.valid).
        config:       TrackGenConfig (uses half_width, turning_tol, w_floor, relax_tol).
    """
    _init()
    E = count_wp.shape[0]
    wp.launch(
        _validity_k, dim=E,
        inputs=[
            center_wp, w_wp, count_wp, gen_valid_wp, outer_wp, inner_wp,
            int(has_border), int(n_max),
            float(config.half_width), float(config.turning_tol),
            float(config.w_floor), float(config.relax_tol),
            out_valid,
        ],
        device=str(out_valid.device),
    )
    _sync(out_valid.device)


def _arclength(
    center_wp: "wp.array",
    out_arclen: "wp.array",
    out_length: "wp.array",
    count_wp: "wp.array",
):
    """In-place: writes arclen into out_arclen (wp.array [E*n_max] float32) and
    total perimeter into out_length (wp.array [E] float32). All args are wp.array;
    nothing allocated. n_max inferred from out_arclen.shape[0] and count_wp.shape[0].

    Pure Warp (cpu+cuda); allclose to oracle to atol~1e-3 (float32 drift).
    """
    _init()
    E = count_wp.shape[0]
    flat = out_arclen.shape[0]
    n_max = flat // E
    wp.launch(_arclength_k, dim=E,
              inputs=[center_wp, n_max, count_wp, out_arclen, out_length],
              device=str(out_arclen.device))
    _sync(out_arclen.device)


class _Scratch:
    """Pre-allocated per-env scratch arrays for in-place stage computations.

    Owned by TrackGenerator; threaded into inflate_warp on the owned path so the
    runtime generate() path makes zero per-call allocations for the converted stages.

    Inflate / offset / frame-curvature / validity fields:
    area_a/area_b: [E] float32 accumulators for the offset shoelace kernel.
    kappa:         [E*N_max] float32 Menger curvature scratch for frame_curvature
                   (kappa is computed by the kernel but unused by the pipeline).
    w:             [E*N_max] float32 per-point half-width buffer for validity_inplace.

    Resample / relax fields:
    cs_center:     [E*N_max] vec2f — constant-spacing resampled centerline output.
    cs_seg:        [E*N_max] float32 — scan scratch shared by resample_constant_spacing
                   and resample_uniform (sequential stages, safe to alias).
    cs_s:          [E*(N_max+1)] float32 — cumulative arc-length scratch (same sharing).
    count:         [E] int32 — real-point-count output of resample_constant_spacing;
                   also threaded through relax → inflate as Track.count.
    relaxed:       [E*N_max] vec2f — xpbd_solve output / resample_uniform input on the
                   generate path (written by xpbd_solve, read-then-overwritten by
                   resample_uniform which writes directly into out.center).
    band:          [E] int32 — _band_l0_k output (excluded-neighbour half-window).
    L0:            [E] float32 — _band_l0_k output (rest segment length per env).
    xpbd_db:       [E*N_max] vec2f — xpbd_solve double-buffer (displacement scratch).

    Generation fields (owned generate path):
    gen_count:     [E] int32 — per-env corner count from corner_count_sample_inplace.
    gen_corners:   [E*P] vec2f — raw corners from corner_sample_inplace (P=max_num_points).
    gen_ordered:   [E*P] vec2f — ccw-sorted corners from ccw_sort_inplace.
    gen_used:      [E*P] int32 — dedup scratch for corner_sample_inplace.
    gen_keys:      [E*P] float32 — sort-key scratch for ccw_sort_inplace.
    gen_tan:       [E*P] vec2f — vertex tangents scratch for assemble_inplace.
    gen_scale:     [E*P] float32 — vertex scale scratch for assemble_inplace.
    gen_dense:     [E*P*npseg] vec2f — Bezier assembled dense centerline.
    gen_poly:      [E*P*npseg] vec2f — polygon assembled dense centerline (the
                   self-crossing fallback, handle_clamp_frac=0).
    gen_rs:        [E*num_points] vec2f — Bezier arc-resampled N-point centerline.
    gen_crossers:  [E] int32 — self-intersection counts (fallback select input).
    gen_centerline:[E*num_points] vec2f — final chosen centerline (output of _select_vec2_k;
                   fed directly into resample_constant_spacing on the owned pipeline path).
    gen_valid:     [E] int32 — generation validity (always 1; passed to inflate_warp).
    gen_arc_real:  [E*P*npseg] vec2f — arc-resample compacted real points scratch.
    gen_arc_seg:   [E*P*npseg] float32 — arc-resample per-segment length scratch.
    gen_arc_s:     [E*(P*npseg+1)] float32 — arc-resample cumulative arc-length scratch.
    gen_arc_cr:    [E] int32 — arc-resample real-point-count scratch.
    gen_arc_co:    [E] int32 — arc-resample output-count scratch (also used as count=N
                   input to self_intersections_inplace after the Bezier resample).
    """

    __slots__ = (
        "area_a", "area_b", "kappa", "w",
        "cs_center", "cs_seg", "cs_s", "count",
        "relaxed", "band", "L0", "xpbd_db",
        # Generation intermediates
        "gen_count", "gen_corners", "gen_ordered", "gen_used", "gen_keys",
        "gen_tan", "gen_scale", "gen_dense", "gen_poly",
        "gen_rs", "gen_crossers", "gen_centerline", "gen_valid",
        "gen_arc_real", "gen_arc_seg", "gen_arc_s", "gen_arc_cr", "gen_arc_co",
    )

    def __init__(
        self,
        area_a: "wp.array",
        area_b: "wp.array",
        kappa: "wp.array",
        w: "wp.array",
        cs_center: "wp.array | None" = None,
        cs_seg: "wp.array | None" = None,
        cs_s: "wp.array | None" = None,
        count: "wp.array | None" = None,
        relaxed: "wp.array | None" = None,
        band: "wp.array | None" = None,
        L0: "wp.array | None" = None,
        xpbd_db: "wp.array | None" = None,
        # Generation intermediates
        gen_count: "wp.array | None" = None,
        gen_corners: "wp.array | None" = None,
        gen_ordered: "wp.array | None" = None,
        gen_used: "wp.array | None" = None,
        gen_keys: "wp.array | None" = None,
        gen_tan: "wp.array | None" = None,
        gen_scale: "wp.array | None" = None,
        gen_dense: "wp.array | None" = None,
        gen_poly: "wp.array | None" = None,
        gen_rs: "wp.array | None" = None,
        gen_crossers: "wp.array | None" = None,
        gen_centerline: "wp.array | None" = None,
        gen_valid: "wp.array | None" = None,
        gen_arc_real: "wp.array | None" = None,
        gen_arc_seg: "wp.array | None" = None,
        gen_arc_s: "wp.array | None" = None,
        gen_arc_cr: "wp.array | None" = None,
        gen_arc_co: "wp.array | None" = None,
    ) -> None:
        self.area_a = area_a
        self.area_b = area_b
        self.kappa = kappa
        self.w = w
        self.cs_center = cs_center
        self.cs_seg = cs_seg
        self.cs_s = cs_s
        self.count = count
        self.relaxed = relaxed
        self.band = band
        self.L0 = L0
        self.xpbd_db = xpbd_db
        # Generation intermediates
        self.gen_count = gen_count
        self.gen_corners = gen_corners
        self.gen_ordered = gen_ordered
        self.gen_used = gen_used
        self.gen_keys = gen_keys
        self.gen_tan = gen_tan
        self.gen_scale = gen_scale
        self.gen_dense = gen_dense
        self.gen_poly = gen_poly
        self.gen_rs = gen_rs
        self.gen_crossers = gen_crossers
        self.gen_centerline = gen_centerline
        self.gen_valid = gen_valid
        self.gen_arc_real = gen_arc_real
        self.gen_arc_seg = gen_arc_seg
        self.gen_arc_s = gen_arc_s
        self.gen_arc_cr = gen_arc_cr
        self.gen_arc_co = gen_arc_co


def _inflate_warp_alloc(config):
    """Allocate a Track with pre-sized wp.array buffers for TrackGenerator.__init__.

    Sizes: outer/center/inner/tangent/normal are [E*N_max] vec2f; arclen is
    [E*N_max] float32; length/valid/count are [E] float32/int32/int32.
    These are the flat storage shapes — reshape via wp bridge at the boundary.

    Also allocates a _Scratch holder with per-env area_a/area_b accumulators for the
    in-place offset stage.

    Args:
        config: TrackGenConfig (uses num_envs, N_max, device).

    Returns:
        (Track, _Scratch) — both with all arrays on config.device.
    """
    from .types import Track  # local import: keep warp_pipeline free of oracle modules

    _init()
    E = int(config.num_envs)
    n_max = int(config.N_max)
    dev = str(config.device)
    flat = E * n_max
    track = Track(
        outer=wp.empty(flat, dtype=wp.vec2f, device=dev),
        center=wp.empty(flat, dtype=wp.vec2f, device=dev),
        inner=wp.empty(flat, dtype=wp.vec2f, device=dev),
        tangent=wp.empty(flat, dtype=wp.vec2f, device=dev),
        normal=wp.empty(flat, dtype=wp.vec2f, device=dev),
        arclen=wp.empty(flat, dtype=wp.float32, device=dev),
        length=wp.empty(E, dtype=wp.float32, device=dev),
        valid=wp.empty(E, dtype=wp.int32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
    )
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M_dense = P * npseg  # dense points per env (assembled Bezier / polygon)
    N_gen = int(config.num_points)  # arc-resampled centerline length (input to cs-resample)

    scratch = _Scratch(
        area_a=wp.zeros(E, dtype=wp.float32, device=dev),
        area_b=wp.zeros(E, dtype=wp.float32, device=dev),
        kappa=wp.empty(flat, dtype=wp.float32, device=dev),
        w=wp.empty(flat, dtype=wp.float32, device=dev),
        # resample + relax intermediates
        cs_center=wp.empty(flat, dtype=wp.vec2f, device=dev),
        cs_seg=wp.empty(flat, dtype=wp.float32, device=dev),
        cs_s=wp.empty(E * (n_max + 1), dtype=wp.float32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        relaxed=wp.empty(flat, dtype=wp.vec2f, device=dev),
        band=wp.empty(E, dtype=wp.int32, device=dev),
        L0=wp.empty(E, dtype=wp.float32, device=dev),
        xpbd_db=wp.empty(flat, dtype=wp.vec2f, device=dev),
        # generation intermediates (owned generate path)
        gen_count=wp.empty(E, dtype=wp.int32, device=dev),
        gen_corners=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_ordered=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_used=wp.empty(E * P, dtype=wp.int32, device=dev),
        gen_keys=wp.empty(E * P, dtype=wp.float32, device=dev),
        gen_tan=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_scale=wp.empty(E * P, dtype=wp.float32, device=dev),
        gen_dense=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_poly=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_rs=wp.empty(E * N_gen, dtype=wp.vec2f, device=dev),
        gen_crossers=wp.empty(E, dtype=wp.int32, device=dev),
        gen_centerline=wp.empty(E * N_gen, dtype=wp.vec2f, device=dev),
        gen_valid=wp.empty(E, dtype=wp.int32, device=dev),
        gen_arc_real=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_arc_seg=wp.empty(E * M_dense, dtype=wp.float32, device=dev),
        gen_arc_s=wp.empty(E * (M_dense + 1), dtype=wp.float32, device=dev),
        gen_arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        gen_arc_co=wp.empty(E, dtype=wp.int32, device=dev),
    )
    return track, scratch


def inflate_warp(center, config, out=None,
                 valid: "wp.array | None" = None,
                 count: "wp.array | None" = None,
                 scratch: "_Scratch | None" = None):
    """Pure-Warp inflation: resample -> frame -> offset -> validity -> arclength.

    Writes results into out's wp.array Track buffers (or allocates a fresh Track).
    All inputs are wp.arrays; center is a flat [E*n_max] vec2f wp.array.

    resample_uniform writes DIRECTLY into out.center; all downstream stages read
    out.center. The offset stage writes DIRECTLY into out.outer/out.inner.
    Scratch (area_a/area_b/kappa/w) is passed in via scratch (zero allocation on owned
    path) or allocated here (the out=None / standalone path).

    Constant-spacing path (count provided — the pipeline always uses this):
        center is [E*n_max] vec2f flat, NaN-padded (real points in [0, count[e])).
        count is threaded into every sub-stage; Track.count = count.

    count=None convenience path (generic fixed-N, fully-finite centerline):
        Requires center to be [E*N] vec2f with no NaN and center.shape[0]//E == N ==
        config.num_points. All sub-stages run with full count (count[e] == N).

    Args:
        center:   [E*n_max] vec2f wp.array (flat).
        config:   TrackGenConfig.
        out:      Optional pre-allocated Track. Allocated fresh when None.
        valid:    [E] int32 wp.array generation flag. Defaults to all-1 when None.
        count:    [E] int32 wp.array real point count. None -> fixed-N path.
        scratch:  Optional _Scratch. Required when out is provided (owned path).

    Returns:
        track_gen.types.Track with wp.array fields.
    """
    from .types import Track  # local import: keep warp_pipeline free of oracle modules

    _init()

    # --- Resolve E, n_max, dev from wp.array center ---
    if count is not None:
        E = count.shape[0]
    else:
        # count=None convenience: infer E from out (if provided) or assume square
        # (caller must ensure center.shape[0] is divisible by E == num_envs).
        if out is not None:
            E = out.valid.shape[0]
        else:
            # Fall back: n_max == num_points, E = total / n_max.
            n_pts = int(config.num_points)
            E = center.shape[0] // n_pts

    n_max = center.shape[0] // E
    dev = str(center.device)
    hw = float(config.half_width)
    flat = E * n_max

    # --- Allocate Track + scratch when not pre-provided ---
    if out is None:
        out = Track(
            outer=wp.empty(flat, dtype=wp.vec2f, device=dev),
            center=wp.empty(flat, dtype=wp.vec2f, device=dev),
            inner=wp.empty(flat, dtype=wp.vec2f, device=dev),
            tangent=wp.empty(flat, dtype=wp.vec2f, device=dev),
            normal=wp.empty(flat, dtype=wp.vec2f, device=dev),
            arclen=wp.empty(flat, dtype=wp.float32, device=dev),
            length=wp.empty(E, dtype=wp.float32, device=dev),
            valid=wp.empty(E, dtype=wp.int32, device=dev),
            count=wp.empty(E, dtype=wp.int32, device=dev),
        )
        if scratch is None:
            scratch = _Scratch(
                area_a=wp.zeros(E, dtype=wp.float32, device=dev),
                area_b=wp.zeros(E, dtype=wp.float32, device=dev),
                kappa=wp.empty(flat, dtype=wp.float32, device=dev),
                w=wp.empty(flat, dtype=wp.float32, device=dev),
            )
    else:
        # Owned path: the caller MUST pass pre-allocated scratch.
        assert scratch is not None, (
            "inflate_warp(out=...) requires a pre-allocated scratch=_Scratch(...) "
            "(zero-allocation contract); pass scratch alongside out."
        )

    # --- Allocate resample scan scratch (seg/s) from _Scratch or standalone ---
    if scratch.cs_seg is not None and scratch.cs_s is not None:
        rs_seg_wp = scratch.cs_seg
        rs_s_wp = scratch.cs_s
    else:
        rs_seg_wp = None
        rs_s_wp = None

    if count is None:
        # --- count=None convenience: fixed-N, fully-finite centerline ---
        assert n_max == int(config.num_points), "center N must equal config.num_points"
        N = n_max

        # Build a full count wp.array (== N for all envs).
        cnt_wp = wp.empty(E, dtype=wp.int32, device=dev)
        wp.launch(_fill_i32_k, dim=E, inputs=[cnt_wp, N], device=dev)

        # 1. resample_uniform: writes directly into out.center.
        resample_uniform(center, out.center, N, cnt_wp,
                         seg_wp=rs_seg_wp, s_wp=rs_s_wp, device=dev)
        _sync(dev)

        # Per-point half-width kernel-filled into scratch.w.
        wp.launch(_fill_f32_k, dim=flat, inputs=[scratch.w, hw], device=dev)

        # Track.count == N for all envs.
        wp.launch(_fill_i32_k, dim=E, inputs=[out.count, N], device=dev)

    else:
        # --- Constant-spacing path: variable count per env, NaN-padded output ---
        cnt_wp = count  # wp.int32 array

        # 1. resample_uniform: writes directly into out.center.
        resample_uniform(center, out.center, n_max, cnt_wp,
                         seg_wp=rs_seg_wp, s_wp=rs_s_wp, device=dev)
        _sync(dev)

        # Per-point half-width kernel-filled into scratch.w.
        wp.launch(_fill_f32_k, dim=flat, inputs=[scratch.w, hw], device=dev)

        # Track.count == per-env real point count.
        wp.copy(out.count, cnt_wp)

    # 2. frame + curvature — in-place directly into out.tangent/out.normal.
    frame_curvature(out.center, out.tangent, out.normal, scratch.kappa, cnt_wp)

    # 3. cumulative arc length + total length — in-place into out.arclen/out.length.
    _arclength(out.center, out.arclen, out.length, cnt_wp)

    # 4. offset to outer/inner borders — in-place directly into out.outer/out.inner.
    offset(out.center, out.normal, hw, out.outer, out.inner,
           scratch.area_a, scratch.area_b, cnt_wp)

    # 5. per-track validity gate (in-place: writes directly into out.valid).
    if valid is not None:
        gv_wp = valid  # wp.int32 array
    else:
        # Standalone / test path: allocate a small all-ones temp.
        gv_wp = wp.empty(E, dtype=wp.int32, device=dev)
        wp.launch(_fill_i32_k, dim=E, inputs=[gv_wp, 1], device=dev)

    # Border self_intersections is optional (config.validity_border_check, default off).
    _bc = getattr(config, "validity_border_check", False)
    has_border = 1 if _bc else 0

    # validity_inplace writes into out.valid directly.
    validity_inplace(out.center, scratch.w, cnt_wp, gv_wp,
                     out.outer, out.inner, has_border, n_max, out.valid, config)

    return out


