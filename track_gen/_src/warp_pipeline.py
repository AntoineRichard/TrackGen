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

from . import warp_relax  # pure-Warp XPBD setup + solve (cpu+cuda); part of the pure-Warp impl
from . import warp_zprofile  # pluggable per-point altitude (Z) profiles for the 2.5D lift
# _separation_band is defined in warp_relax (relaxation owns the band definition); it is
# imported into this module's namespace so the _validity_k kernel can resolve the @wp.func.
from .warp_relax import _separation_band  # noqa: F401  (used inside _validity_k)

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
def _self_intersections_by_i_k(
    poly: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    # One thread per segment start i. This preserves the same pair predicate as
    # _self_intersections_func, but avoids serializing every O(N^2) pair check onto
    # one thread per environment. Crossing counts are rare, so atomics are cheap.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    N = count[e]
    if i >= N:
        return

    base = e * n_max
    Ai = poly[base + i]
    Bi = poly[base + (i + 1) % N]
    aix = _nan0(Ai[0]); aiy = _nan0(Ai[1])
    bix = _nan0(Bi[0]); biy = _nan0(Bi[1])

    for j in range(i + 1, N):
        diff = j - i
        circ_dist = wp.min(diff, N - diff)
        if circ_dist <= 1:
            continue
        Aj = poly[base + j]
        Bj = poly[base + (j + 1) % N]
        ajx = _nan0(Aj[0]); ajy = _nan0(Aj[1])
        bjx = _nan0(Bj[0]); bjy = _nan0(Bj[1])
        d1 = _ccw(ajx, ajy, bjx, bjy, aix, aiy)
        d2 = _ccw(ajx, ajy, bjx, bjy, bix, biy)
        d3 = _ccw(aix, aiy, bix, biy, ajx, ajy)
        d4 = _ccw(aix, aiy, bix, biy, bjx, bjy)
        lj2 = (bjx - ajx) * (bjx - ajx) + (bjy - ajy) * (bjy - ajy)
        li2 = (bix - aix) * (bix - aix) + (biy - aiy) * (biy - aiy)
        ej = float(1.0e-3) * lj2
        ei = float(1.0e-3) * li2
        seg_ij = (d1 > ej and d2 < -ej) or (d1 < -ej and d2 > ej)
        seg_ji = (d3 > ei and d4 < -ei) or (d3 < -ei and d4 > ei)
        if seg_ij and seg_ji:
            wp.atomic_add(out, e, int(1))

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

@wp.kernel
def _arc_scan_selected_k(
    dense: wp.array(dtype=wp.vec2f),
    active: wp.array(dtype=wp.int32),
    M: int,
    num: int,
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
):
    # Same as _arc_scan_k for active envs. Inactive envs write only their counts,
    # allowing downstream lookup to skip them and leave output rows untouched.
    e = wp.tid()
    if active[e] <= 0:
        count_r[e] = int(0)
        count_out[e] = int(0)
        return

    db = e * M
    rb = e * M
    es = e * (M + 1)

    r = int(0)
    for i in range(M):
        p = dense[db + i]
        if wp.isfinite(p[0]) and wp.isfinite(p[1]):
            real_pts[rb + r] = p
            r = r + 1
    count_r[e] = r
    count_out[e] = wp.where(r >= 2, num, 0)

    if r >= 2:
        s[es] = float(0.0)
        acc = wp.float64(0.0)
        for j in range(r):
            nxt = real_pts[rb + (j + 1) % r]
            l = wp.length(nxt - real_pts[rb + j])
            seg[rb + j] = l
            acc = acc + wp.float64(l)
            s[es + j + 1] = wp.float32(acc)

@wp.kernel
def _arc_lookup_selected_k(
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    M: int,
    num: int,
    out: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // num
    if active[e] <= 0:
        return

    k = t % num
    rb = e * M
    es = e * (M + 1)
    r = count_r[e]
    if r < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    total = s[es + r]
    target = float(k) * total / float(num)
    idx = int(0)
    while idx < r - 1 and s[es + idx + 1] < target:
        idx = idx + 1

    s0 = s[es + idx]
    segl = wp.max(seg[rb + idx], float(1.0e-12))
    frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))
    p0 = real_pts[rb + idx]
    p1 = real_pts[rb + (idx + 1) % r]
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
    z: wp.array(dtype=wp.float32),
    z_valid_grade: float,
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

    # --- grade gate (2.5D): reject any segment whose vertical rise over plan-view
    # run exceeds z_valid_grade. Skipped entirely when z_valid_grade <= 0 (flat
    # default) so z is read only when a grade cap is configured. z here is the
    # per-point altitude the profiler wrote; center is the 2D plan-view polyline. ---
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

    # --- generation flag ---
    gv = int(0)
    if gen_valid[e] != 0:
        gv = int(1)

    out[e] = gv & turn_ok & w_ok & no_nan & th_ok & border_ok & grade_ok

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
def _winding_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Sign of the closed centerline's signed shoelace area:
    # +1.0 counter-clockwise, -1.0 clockwise, 0.0 degenerate (count < 3, zero area,
    # or any NaN point -> comparisons are false -> 0.0). Fixed launch, no host
    # branch -> safe inside a CUDA graph capture region.
    e = wp.tid()
    cn = count[e]
    b = e * n_max
    area2 = float(0.0)
    for i in range(cn):
        p = c[b + i]
        q = c[b + (i + 1) % cn]
        area2 = area2 + (p[0] * q[1] - q[0] * p[1])
    out[e] = wp.where(area2 > 0.0, 1.0, wp.where(area2 < 0.0, -1.0, 0.0))

@wp.kernel
def _fill_f32_k(arr: wp.array(dtype=wp.float32), v: float):
    # One thread per element: constant float fill.
    arr[wp.tid()] = v

@wp.kernel
def _fill_i32_k(arr: wp.array(dtype=wp.int32), v: int):
    # One thread per element: constant int fill.
    arr[wp.tid()] = v

def self_intersections_inplace(poly_wp, count_wp, out_wp, n_max):
    """In-place: writes [E] int32 crossing counts into out_wp. Zero alloc.
    poly_wp: [E*n_max] vec2f; count_wp: [E] int32; out_wp: [E] int32."""
    _init()
    E = count_wp.shape[0]
    dev = str(out_wp.device)
    wp.launch(_fill_i32_k, dim=E, inputs=[out_wp, 0], device=dev)
    wp.launch(_self_intersections_by_i_k, dim=E * n_max,
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


def _arc_resample_selected_inplace(
    points_wp, active_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp, out_wp, dev
):
    E = points_wp.shape[0] // M
    wp.launch(_arc_scan_selected_k, dim=E,
              inputs=[points_wp, active_wp, M, num, real_wp, seg_wp, s_wp,
                      count_r_wp, count_out_wp], device=dev)
    wp.launch(_arc_lookup_selected_k, dim=E * num,
              inputs=[real_wp, seg_wp, s_wp, count_r_wp, active_wp, M, num, out_wp],
              device=dev)


def arc_length_resample_selected_inplace(
    points_wp: "wp.array",
    active_wp: "wp.array",
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
    """Selected NaN-aware arc-length resample. Inactive envs leave output untouched."""
    _init()
    _arc_resample_selected_inplace(
        points_wp, active_wp, M, num, real_wp, seg_wp, s_wp,
        count_r_wp, count_out_wp, out_wp, dev,
    )
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


def _run_pipeline(
    config,
    seed_buf_wp: wp.array,
    out: "Track",
    scratch: "_Scratch",
    generator_spec=None,
) -> "Track":
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
    if generator_spec is None:
        generator_spec = getattr(scratch, "generator_spec", None)
    if generator_spec is None:
        from . import generator_registry
        generator_spec = generator_registry.get(config.generator)

    n_max = int(config.N_max)
    gen = scratch.gen        # generator-private scratch
    relax = scratch.relax    # RelaxScratch — band/L0 + XPBD buffers

    # 1. Generate centerline in-place into the orchestrator-owned output buffers.
    generate = generator_spec.generate
    generate(seed_buf_wp, config,
             out_centerline=scratch.gen_centerline,
             out_valid_wp=scratch.gen_valid,
             scratch=gen)

    # 2. Constant-spacing resample (gen centerline -> bridge buffers).
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
        # 3. Relaxation setup: band / L0 (the relax module owns its own setup).
        warp_relax.band_l0_inplace(
            scratch.cs_center, n_max, relax.band, relax.L0,
            scratch.count, config, capturing=_CAPTURING,
        )

        # 4. XPBD relaxation in-place. Thread the pipeline's capture state in explicitly
        # so warp_relax decides its host-sync without reaching back into this module.
        warp_relax.xpbd_solve_inplace(
            scratch.cs_center, relax.relaxed, relax.xpbd_db,
            relax.band, relax.L0, scratch.count, n_max, config,
            capturing=_CAPTURING,
            sep_cache_idx_wp=relax.sep_cache_idx,
            sep_cache_count_wp=relax.sep_cache_count,
            sep_cache_overflow_wp=relax.sep_cache_overflow,
            cheb_prev_wp=relax.cheb_prev,
        )
        relax_out = relax.relaxed
    else:
        relax_out = scratch.cs_center

    # 5. Inflate (resample_uniform + frame + offset + validity) in-place. inflate_warp
    # reads the inflate group (kappa/w/area_a/area_b) + the bridge buffers (cs_seg/cs_s/
    # count) off the composite scratch.
    return inflate_warp(
        relax_out, config, out=out, valid=scratch.gen_valid,
        count=scratch.count, scratch=scratch, seeds=seed_buf_wp,
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
    z_wp: "wp.array",
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
        z_wp:         [E*n_max] float32 per-point altitude (flat); read only when
                      config.z_valid_grade > 0 (the grade gate), else ignored.
        out_valid:    [E] int32 wp.array to write into (e.g. out.valid).
        config:       TrackGenConfig (uses half_width, turning_tol, w_floor, relax_tol,
                      z_valid_grade).
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
            z_wp, float(config.z_valid_grade),
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


def _winding(
    center_wp: "wp.array",
    out_winding: "wp.array",
    count_wp: "wp.array",
):
    """In-place: writes per-env signed loop winding (+1 CCW / -1 CW / 0 degenerate)
    into out_winding (wp.array [E] float32) from the sign of the centerline's signed
    area. All args are wp.array; nothing allocated. n_max inferred from
    center_wp.shape[0] and count_wp.shape[0].

    Fixed launch (no host branch) -> safe inside a CUDA graph capture region.
    """
    _init()
    E = count_wp.shape[0]
    n_max = center_wp.shape[0] // E
    wp.launch(_winding_k, dim=E,
              inputs=[center_wp, n_max, count_wp, out_winding],
              device=str(out_winding.device))
    _sync(out_winding.device)


class GenScratch:
    """Pre-allocated generation buffers for the bezier generator's PRIVATE working scratch.

    The generation OUTPUT buffers (gen_centerline, gen_valid) are owned by the orchestrator
    (_Scratch) and passed into generate_centerline_warp; they are NOT part of this class.

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
    gen_arc_real:  [E*P*npseg] vec2f — arc-resample compacted real points scratch.
    gen_arc_seg:   [E*P*npseg] float32 — arc-resample per-segment length scratch.
    gen_arc_s:     [E*(P*npseg+1)] float32 — arc-resample cumulative arc-length scratch.
    gen_arc_cr:    [E] int32 — arc-resample real-point-count scratch.
    gen_arc_co:    [E] int32 — arc-resample output-count scratch (also used as count=N
                   input to self_intersections_inplace after the Bezier resample).
    gen_style_rad:   [E] float32 — per-env rad draw (method #1; only written/read when
                     config.style_sampling=True, else None/untouched). Optional.
    gen_style_scale: [E] float32 — per-env scale draw (method #1). Optional.
    gen_style_clamp: [E] float32 — per-env handle_clamp_frac draw (method #1). Optional.
    """

    __slots__ = (
        "gen_count", "gen_corners", "gen_ordered", "gen_used", "gen_keys",
        "gen_tan", "gen_scale", "gen_dense", "gen_poly",
        "gen_rs", "gen_crossers",
        "gen_arc_real", "gen_arc_seg", "gen_arc_s", "gen_arc_cr", "gen_arc_co",
        "gen_style_rad", "gen_style_scale", "gen_style_clamp",
    )

    def __init__(
        self,
        gen_count: "wp.array",
        gen_corners: "wp.array",
        gen_ordered: "wp.array",
        gen_used: "wp.array",
        gen_keys: "wp.array",
        gen_tan: "wp.array",
        gen_scale: "wp.array",
        gen_dense: "wp.array",
        gen_poly: "wp.array",
        gen_rs: "wp.array",
        gen_crossers: "wp.array",
        gen_arc_real: "wp.array",
        gen_arc_seg: "wp.array",
        gen_arc_s: "wp.array",
        gen_arc_cr: "wp.array",
        gen_arc_co: "wp.array",
        gen_style_rad: "wp.array | None" = None,
        gen_style_scale: "wp.array | None" = None,
        gen_style_clamp: "wp.array | None" = None,
    ) -> None:
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
        self.gen_arc_real = gen_arc_real
        self.gen_arc_seg = gen_arc_seg
        self.gen_arc_s = gen_arc_s
        self.gen_arc_cr = gen_arc_cr
        self.gen_arc_co = gen_arc_co
        self.gen_style_rad = gen_style_rad
        self.gen_style_scale = gen_style_scale
        self.gen_style_clamp = gen_style_clamp


class RelaxScratch:
    """Pre-allocated relaxation buffers (band/L0 setup + XPBD solve; warp_relax stage).

    relaxed:  [E*N_max] vec2f — xpbd_solve output / resample_uniform input on the generate
              path (written by xpbd_solve, read-then-overwritten by resample_uniform which
              writes directly into out.center).
    band:     [E] int32 — band_l0_inplace output (excluded-neighbour half-window).
    L0:       [E] float32 — band_l0_inplace output (rest segment length per env).
    xpbd_db:  [E*N_max] vec2f — xpbd_solve position ping-pong scratch.
    cheb_prev: [E*N_max] vec2f — xpbd_solve previous-iterate buffer for Chebyshev
              acceleration; None when relax_accel == "none".
    sep_cache_idx/count/overflow: optional fixed-size separation candidate cache.
    """

    __slots__ = (
        "relaxed", "band", "L0", "xpbd_db", "cheb_prev",
        "sep_cache_idx", "sep_cache_count", "sep_cache_overflow",
    )

    def __init__(
        self,
        relaxed: "wp.array",
        band: "wp.array",
        L0: "wp.array",
        xpbd_db: "wp.array",
        cheb_prev: "wp.array | None" = None,
        sep_cache_idx: "wp.array | None" = None,
        sep_cache_count: "wp.array | None" = None,
        sep_cache_overflow: "wp.array | None" = None,
    ) -> None:
        self.relaxed = relaxed
        self.band = band
        self.L0 = L0
        self.xpbd_db = xpbd_db
        self.cheb_prev = cheb_prev
        self.sep_cache_idx = sep_cache_idx
        self.sep_cache_count = sep_cache_count
        self.sep_cache_overflow = sep_cache_overflow


class InflateScratch:
    """Pre-allocated inflation buffers (offset / frame-curvature / validity stages).

    area_a/area_b: [E] float32 accumulators for the offset shoelace kernel.
    kappa:         [E*N_max] float32 Menger curvature scratch for frame_curvature
                   (kappa is computed by the kernel but unused by the pipeline).
    w:             [E*N_max] float32 per-point half-width buffer for validity_inplace.
    out2/ctr2/inn2/tan2/nrm2: [E*N_max] vec2f staging buffers the 2D pipeline
                   kernels write; the final lift kernel copies them into the
                   public vec3f Track buffers with z = 0. None only on legacy
                   constructions that never reach inflate_warp.
    z:             [E*N_max] float32 per-point altitude scratch for the 2.5D
                   elevation stage (written by warp_zprofile.apply_z_profile on
                   the non-flat path; zero-filled at alloc so the flat path and
                   the grade-validity check read zeros). None on legacy
                   constructions that never reach inflate_warp.
    knot_cum/knot_count/knot_z: knot-stage scratch for the knot-based elevation
                   profiles ("uniform"/"random_walk"), from
                   warp_zprofile.alloc_knot_scratch: knot_cum/knot_z are
                   [E*z_control_points] float32 (knot arc positions and knot
                   altitudes), knot_count is [E] int32. None on the flat/noise
                   paths and on legacy constructions that never reach
                   inflate_warp.
    """

    __slots__ = ("area_a", "area_b", "kappa", "w",
                 "out2", "ctr2", "inn2", "tan2", "nrm2", "z",
                 "knot_cum", "knot_count", "knot_z")

    def __init__(
        self,
        area_a: "wp.array",
        area_b: "wp.array",
        kappa: "wp.array",
        w: "wp.array",
        out2: "wp.array | None" = None,
        ctr2: "wp.array | None" = None,
        inn2: "wp.array | None" = None,
        tan2: "wp.array | None" = None,
        nrm2: "wp.array | None" = None,
        z: "wp.array | None" = None,
        knot_cum: "wp.array | None" = None,
        knot_count: "wp.array | None" = None,
        knot_z: "wp.array | None" = None,
    ) -> None:
        self.area_a = area_a
        self.area_b = area_b
        self.kappa = kappa
        self.w = w
        self.out2 = out2
        self.ctr2 = ctr2
        self.inn2 = inn2
        self.tan2 = tan2
        self.nrm2 = nrm2
        self.z = z
        self.knot_cum = knot_cum
        self.knot_count = knot_count
        self.knot_z = knot_z


class _Scratch:
    """Per-concern scratch groups for the in-place pipeline, composed in one holder.

    Owned by TrackGenerator; threaded into each stage on the owned path so the runtime
    generate() path makes zero per-call allocations. The buffers are grouped by lifetime/
    concern so each stage receives only the group it owns:

    .generator_spec — GeneratorSpec resolved when this scratch was allocated.
    .gen      — GenScratch:     generation intermediates (warp_generate).
    .relax    — RelaxScratch:   band/L0 + XPBD buffers (warp_relax).
    .inflate  — InflateScratch: offset/frame/validity scratch.

    Plus the BRIDGE buffers that span stages (the resample output threaded gen -> relax ->
    inflate), held directly on the holder:
    gen_centerline: [E*num_points] vec2f — generation output centerline (orchestrator-owned;
                    written by the generator, read by resample_constant_spacing).
    gen_valid:      [E] int32 — generation validity output (orchestrator-owned; written by
                    the generator, read by inflate_warp).
    cs_center: [E*N_max] vec2f — constant-spacing resampled centerline output.
    cs_seg:    [E*N_max] float32 — scan scratch shared by resample_constant_spacing and
               resample_uniform (sequential stages, safe to alias).
    cs_s:      [E*(N_max+1)] float32 — cumulative arc-length scratch (same sharing).
    count:     [E] int32 — real-point-count output of resample_constant_spacing; also
               threaded through relax -> inflate as Track.count.

    For convenience, attribute access falls through to the sub-groups: ``scratch.relaxed``
    resolves to ``scratch.relax.relaxed``, ``scratch.kappa`` to ``scratch.inflate.kappa``,
    etc., so flat-name call sites keep working while the grouping is type-enforced.
    gen_centerline and gen_valid are direct slots (orchestrator-owned); they do NOT fall
    through to scratch.gen.
    """

    __slots__ = (
        "generator_spec", "gen", "relax", "inflate",
        "gen_centerline", "gen_valid",
        "cs_center", "cs_seg", "cs_s", "count",
    )

    def __init__(
        self,
        inflate: "InflateScratch",
        generator_spec=None,
        gen: "GenScratch | None" = None,
        relax: "RelaxScratch | None" = None,
        gen_centerline: "wp.array | None" = None,
        gen_valid: "wp.array | None" = None,
        cs_center: "wp.array | None" = None,
        cs_seg: "wp.array | None" = None,
        cs_s: "wp.array | None" = None,
        count: "wp.array | None" = None,
    ) -> None:
        self.generator_spec = generator_spec
        self.gen = gen
        self.relax = relax
        self.inflate = inflate
        self.gen_centerline = gen_centerline
        self.gen_valid = gen_valid
        self.cs_center = cs_center
        self.cs_seg = cs_seg
        self.cs_s = cs_s
        self.count = count

    def __getattr__(self, name):
        # Fall through to the per-concern sub-groups so flat-name accesses
        # (scratch.relaxed, scratch.kappa, ...) keep working.
        # __getattr__ runs only for names not found via __slots__, so the bridge fields
        # (gen_centerline, gen_valid, cs_center, cs_seg, cs_s, count) and the sub-group
        # handles above are never routed here.
        for group in (self.gen, self.relax, self.inflate):
            if group is None:
                continue
            if name in getattr(group, "__slots__", ()):
                return getattr(group, name)
            attrs = getattr(group, "__dict__", None)
            if attrs is not None and name in attrs:
                return getattr(group, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")


def _fill_knot_scratch(stag, config, num_envs, device) -> None:
    """Attach knot-stage scratch to ``stag`` when the profile is knot-based.

    Construction-time only (config-static branch): the knot buffers must exist
    before capture so the stage-4b dispatch stays allocation-free. A no-op for
    the "flat"/"noise" profiles, which never touch the knot stage.
    """
    if str(getattr(config, "z_profile", "flat")) in ("uniform", "random_walk"):
        stag.knot_cum, stag.knot_count, stag.knot_z = \
            warp_zprofile.alloc_knot_scratch(
                num_envs, int(config.z_control_points), device)


def _inflate_warp_alloc(config, generator_spec=None):
    """Allocate a Track with pre-sized wp.array buffers for TrackGenerator.__init__.

    Sizes: outer/center/inner/tangent/normal are [E*N_max] vec3f (z = 0, lifted
    from the internal 2D pipeline); arclen is
    [E*N_max] float32; length/valid/count are [E] float32/int32/int32.
    These are the flat storage shapes — reshape via wp bridge at the boundary.

    Also allocates the composite _Scratch holder: a GenScratch, RelaxScratch and
    InflateScratch group plus the bridge buffers (cs_center/cs_seg/cs_s/count), all sized
    for the owned generate path so every stage runs zero-alloc.

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
        outer=wp.empty(flat, dtype=wp.vec3f, device=dev),
        center=wp.empty(flat, dtype=wp.vec3f, device=dev),
        inner=wp.empty(flat, dtype=wp.vec3f, device=dev),
        tangent=wp.empty(flat, dtype=wp.vec3f, device=dev),
        normal=wp.empty(flat, dtype=wp.vec3f, device=dev),
        arclen=wp.empty(flat, dtype=wp.float32, device=dev),
        length=wp.empty(E, dtype=wp.float32, device=dev),
        valid=wp.empty(E, dtype=wp.int32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        winding=wp.empty(E, dtype=wp.float32, device=dev),
    )
    if generator_spec is None:
        from . import generator_registry
        generator_spec = generator_registry.get(config.generator)
    gen = generator_spec.alloc_scratch(config)
    N_gen = int(config.num_points)
    gen_centerline = wp.empty(E * N_gen, dtype=wp.vec2f, device=dev)
    gen_valid = wp.empty(E, dtype=wp.int32, device=dev)
    # XPBD cached-contacts perf buffers (from main): sized by config.relax_sep_cache_slots,
    # threaded into RelaxScratch below; None when separation caching is disabled (slots == 0).
    sep_cache_slots = max(0, int(getattr(config, "relax_sep_cache_slots", 0)))
    if sep_cache_slots > 0:
        sep_cache_idx = wp.empty(flat * sep_cache_slots, dtype=wp.int32, device=dev)
        sep_cache_count = wp.empty(flat, dtype=wp.int32, device=dev)
        sep_cache_overflow = wp.empty(1, dtype=wp.int32, device=dev)
    else:
        sep_cache_idx = None
        sep_cache_count = None
        sep_cache_overflow = None
    # Chebyshev previous-iterate buffer: pre-allocated here so the capture-path solve
    # stays zero-per-call-alloc; None when acceleration is disabled. Contents never
    # matter (xpbd_solve never reads it before writing it), so wp.empty suffices.
    if str(getattr(config, "relax_accel", "none")) == "chebyshev":
        cheb_prev = wp.empty(flat, dtype=wp.vec2f, device=dev)
    else:
        cheb_prev = None
    relax = RelaxScratch(
        relaxed=wp.empty(flat, dtype=wp.vec2f, device=dev),
        band=wp.empty(E, dtype=wp.int32, device=dev),
        L0=wp.empty(E, dtype=wp.float32, device=dev),
        xpbd_db=wp.empty(flat, dtype=wp.vec2f, device=dev),
        cheb_prev=cheb_prev,
        sep_cache_idx=sep_cache_idx,
        sep_cache_count=sep_cache_count,
        sep_cache_overflow=sep_cache_overflow,
    )
    inflate = InflateScratch(
        area_a=wp.zeros(E, dtype=wp.float32, device=dev),
        area_b=wp.zeros(E, dtype=wp.float32, device=dev),
        kappa=wp.empty(flat, dtype=wp.float32, device=dev),
        w=wp.empty(flat, dtype=wp.float32, device=dev),
        out2=wp.empty(flat, dtype=wp.vec2f, device=dev),
        ctr2=wp.empty(flat, dtype=wp.vec2f, device=dev),
        inn2=wp.empty(flat, dtype=wp.vec2f, device=dev),
        tan2=wp.empty(flat, dtype=wp.vec2f, device=dev),
        nrm2=wp.empty(flat, dtype=wp.vec2f, device=dev),
        # Zero-filled: the flat path never writes z (grade check reads zeros).
        z=wp.zeros(flat, dtype=wp.float32, device=dev),
    )
    _fill_knot_scratch(inflate, config, E, dev)
    scratch = _Scratch(
        inflate=inflate, generator_spec=generator_spec, gen=gen, relax=relax,
        # orchestrator-owned generation output buffers
        gen_centerline=gen_centerline,
        gen_valid=gen_valid,
        # bridge buffers threaded gen -> relax -> inflate
        cs_center=wp.empty(flat, dtype=wp.vec2f, device=dev),
        # The resample scan scratch is indexed by the INPUT count N (= num_points when fed the
        # generation buffer), not the N_max output. Size it to max(N_max, num_points) so a
        # config with num_points > N_max cannot overrun it (out-of-bounds GPU writes).
        cs_seg=wp.empty(E * max(n_max, N_gen), dtype=wp.float32, device=dev),
        cs_s=wp.empty(E * (max(n_max, N_gen) + 1), dtype=wp.float32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
    )
    return track, scratch


@wp.kernel
def _lift_track_k(
    out2: wp.array(dtype=wp.vec2f), ctr2: wp.array(dtype=wp.vec2f),
    inn2: wp.array(dtype=wp.vec2f), tan2: wp.array(dtype=wp.vec2f),
    nrm2: wp.array(dtype=wp.vec2f),
    z_base: float,
    outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
    inner: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
    normal: wp.array(dtype=wp.vec3f),
):
    # Flat (constant-altitude) lift: every point gets the same z_base. Only
    # positions carry z; tangent/normal z stays 0.0. With z_base=0.0 (the flat
    # default) this writes 0.0 -> bit-identical to the legacy path.
    t = wp.tid()
    outer[t] = wp.vec3f(out2[t][0], out2[t][1], z_base)
    center[t] = wp.vec3f(ctr2[t][0], ctr2[t][1], z_base)
    inner[t] = wp.vec3f(inn2[t][0], inn2[t][1], z_base)
    tangent[t] = wp.vec3f(tan2[t][0], tan2[t][1], 0.0)
    normal[t] = wp.vec3f(nrm2[t][0], nrm2[t][1], 0.0)


@wp.kernel
def _lift_track_zvar_k(
    out2: wp.array(dtype=wp.vec2f), ctr2: wp.array(dtype=wp.vec2f),
    inn2: wp.array(dtype=wp.vec2f), nrm2: wp.array(dtype=wp.vec2f),
    z: wp.array(dtype=wp.float32),
    outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
    inner: wp.array(dtype=wp.vec3f), normal: wp.array(dtype=wp.vec3f),
):
    # Level cross-sections: all three polylines share the centerline z.
    # tangent is NOT written here -- _track_frames3_k recomputes it in 3D.
    t = wp.tid()
    zt = z[t]
    outer[t] = wp.vec3f(out2[t][0], out2[t][1], zt)
    center[t] = wp.vec3f(ctr2[t][0], ctr2[t][1], zt)
    inner[t] = wp.vec3f(inn2[t][0], inn2[t][1], zt)
    normal[t] = wp.vec3f(nrm2[t][0], nrm2[t][1], 0.0)


@wp.kernel
def _track_frames3_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    tangent: wp.array(dtype=wp.vec3f),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
):
    # Per-env serial recompute of tangent (3D central diff) + true 3D arc
    # length. Same work distribution as checkpoints/props per-env scans.
    # Mirrors course_line.py's _line_frames_k: real slots [0, m) get the
    # central-diff tangent + cumulative arclen; padding slots [m, n_max) are
    # NaN-filled (per the Track.tangent/arclen NaN-padding contract in
    # types.py). Degenerate envs (m < 3) NaN-fill the whole row and zero length.
    e = wp.tid()
    base = e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        length[e] = 0.0
        for i in range(n_max):
            tangent[base + i] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            arclen[base + i] = wp.nan
        return
    acc = float(0.0)
    for i in range(n_max):
        if i < m:
            if i > 0:
                acc = acc + wp.length(center[base + i] - center[base + i - 1])
            arclen[base + i] = acc
            prev = ((i - 1) % m + m) % m
            nxt = (i + 1) % m
            d = center[base + nxt] - center[base + prev]
            l = wp.length(d)
            if l > 1.0e-12:
                tangent[base + i] = d / l
            else:
                tangent[base + i] = wp.vec3f(0.0, 0.0, 0.0)
        else:
            tangent[base + i] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            arclen[base + i] = wp.nan
    length[e] = acc + wp.length(center[base] - center[base + m - 1])


def inflate_warp(center, config, out=None,
                 valid: "wp.array | None" = None,
                 count: "wp.array | None" = None,
                 scratch: "_Scratch | None" = None,
                 seeds: "wp.array | None" = None):
    """Pure-Warp inflation: resample -> frame -> offset -> z-profile -> validity -> lift.

    Writes results into out's wp.array Track buffers (or allocates a fresh Track).
    All inputs are wp.arrays; center is a flat [E*n_max] vec2f wp.array.

    The 2D stages operate on the vec2f staging buffers (scratch.inflate.ctr2/
    tan2/nrm2/out2/inn2); a final lift kernel copies them into the public vec3f
    Track buffers. On the flat default (``config.z_profile == "flat"``) the lift
    writes a constant z (``config.z_base``, 0.0 by default) and the arclen/length/
    tangent tables stay the byte-identical 2D values from stages 1-5. On a non-flat
    profile the per-point altitude ``scratch.inflate.z`` is filled by
    ``warp_zprofile.apply_z_profile`` (from the 2D ``(cum, perim)`` tables still in
    out.arclen/out.length), the three polylines are lifted level to that z, and
    ``_track_frames3_k`` recomputes the tangent + true 3D arclen/length (overwriting
    the 2D tables). A non-flat profile REQUIRES ``seeds`` (the standalone path is
    flat-only otherwise).
    Scratch (area_a/area_b/kappa/w/z + staging) is passed in via scratch (zero
    allocation on owned path) or allocated here (the out=None / standalone path).

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
        seeds:    [E] int32 wp.array per-env base seeds for the z profiler. Required
                  when config.z_profile != "flat"; ignored on the flat path.

    Returns:
        track_gen.types.Track with wp.array fields.

    Raises:
        ValueError: if config.z_profile != "flat" and seeds is None (the standalone
            inflate path is flat-only unless seeds are provided).
    """
    from .types import Track  # local import: keep warp_pipeline free of oracle modules

    _init()

    # A non-flat elevation profile needs per-env seeds to drive the profiler; the
    # standalone inflate path (seeds omitted) supports only the flat profile.
    _z_profile = str(getattr(config, "z_profile", "flat"))
    if _z_profile != "flat" and seeds is None:
        raise ValueError(
            "z_profile != 'flat' requires seeds= (the standalone inflate path is "
            "flat-only otherwise)"
        )

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
            outer=wp.empty(flat, dtype=wp.vec3f, device=dev),
            center=wp.empty(flat, dtype=wp.vec3f, device=dev),
            inner=wp.empty(flat, dtype=wp.vec3f, device=dev),
            tangent=wp.empty(flat, dtype=wp.vec3f, device=dev),
            normal=wp.empty(flat, dtype=wp.vec3f, device=dev),
            arclen=wp.empty(flat, dtype=wp.float32, device=dev),
            length=wp.empty(E, dtype=wp.float32, device=dev),
            valid=wp.empty(E, dtype=wp.int32, device=dev),
            count=wp.empty(E, dtype=wp.int32, device=dev),
            winding=wp.empty(E, dtype=wp.float32, device=dev),
        )
        if scratch is None:
            # Standalone path: only the inflate group is needed (no gen/relax buffers);
            # cs_seg/cs_s left None so inflate_warp allocates its own resample scan scratch.
            scratch = _Scratch(
                inflate=InflateScratch(
                    area_a=wp.zeros(E, dtype=wp.float32, device=dev),
                    area_b=wp.zeros(E, dtype=wp.float32, device=dev),
                    kappa=wp.empty(flat, dtype=wp.float32, device=dev),
                    w=wp.empty(flat, dtype=wp.float32, device=dev),
                    out2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    ctr2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    inn2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    tan2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    nrm2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    z=wp.zeros(flat, dtype=wp.float32, device=dev),
                ),
            )
    else:
        # Owned path: the caller MUST pass pre-allocated scratch.
        assert scratch is not None, (
            "inflate_warp(out=...) requires a pre-allocated scratch=_Scratch(...) "
            "(zero-allocation contract); pass scratch alongside out."
        )

    # The 2D pipeline kernels write the vec2f staging buffers; the lift at the
    # end copies them into the public vec3f Track arrays with z = 0.
    stag = scratch.inflate
    if stag.ctr2 is None:
        stag.out2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.ctr2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.inn2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.tan2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.nrm2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
    if stag.z is None:
        stag.z = wp.zeros(flat, dtype=wp.float32, device=dev)
    if stag.knot_cum is None:
        _fill_knot_scratch(stag, config, E, dev)

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

        # 1. resample_uniform: writes into the vec2f center staging buffer.
        resample_uniform(center, stag.ctr2, N, cnt_wp,
                         seg_wp=rs_seg_wp, s_wp=rs_s_wp, device=dev)
        _sync(dev)

        # Per-point half-width kernel-filled into scratch.w.
        wp.launch(_fill_f32_k, dim=flat, inputs=[scratch.w, hw], device=dev)

        # Track.count == N for all envs.
        wp.launch(_fill_i32_k, dim=E, inputs=[out.count, N], device=dev)

    else:
        # --- Constant-spacing path: variable count per env, NaN-padded output ---
        cnt_wp = count  # wp.int32 array

        # 1. resample_uniform: writes into the vec2f center staging buffer.
        resample_uniform(center, stag.ctr2, n_max, cnt_wp,
                         seg_wp=rs_seg_wp, s_wp=rs_s_wp, device=dev)
        _sync(dev)

        # Per-point half-width kernel-filled into scratch.w.
        wp.launch(_fill_f32_k, dim=flat, inputs=[scratch.w, hw], device=dev)

        # Track.count == per-env real point count.
        wp.copy(out.count, cnt_wp)

    # 2. frame + curvature — into the vec2f tangent/normal staging buffers.
    frame_curvature(stag.ctr2, stag.tan2, stag.nrm2, scratch.kappa, cnt_wp)

    # 3. cumulative arc length + total length — in-place into out.arclen/out.length.
    _arclength(stag.ctr2, out.arclen, out.length, cnt_wp)

    # 3b. signed loop winding (+1 CCW / -1 CW / 0 degenerate) — in-place into out.winding.
    _winding(stag.ctr2, out.winding, cnt_wp)

    # 4. offset to outer/inner borders — into the vec2f outer/inner staging buffers.
    offset(stag.ctr2, stag.nrm2, hw, stag.out2, stag.inn2,
           scratch.area_a, scratch.area_b, cnt_wp)

    # 4b. Elevation (2.5D): fill the per-point altitude from the configured profile.
    # CRITICAL ordering: out.arclen/out.length still hold the 2D (cum, perim) tables
    # from stage 3 here — the profiler consumes them as its plan-view arc
    # parameterization, BEFORE _track_frames3_k (stage 6, non-flat) overwrites them
    # with 3D values. Skipped on the flat path (scratch.z stays zero-filled), which
    # keeps the arclen/length/tangent tables byte-identical to the legacy 2D path.
    #
    # uniform/random_walk decide altitude at z_control_points arc-spaced KNOTS and
    # interpolate with a periodic monotone cubic, so smoothness is set by the knot
    # count rather than the resampled point count. noise is analytically smooth
    # already (its harmonics band-limit it), so it stays per-point.
    _is_flat = _z_profile == "flat"
    if _z_profile in ("uniform", "random_walk"):
        warp_zprofile.apply_z_profile_knots(
            config, seeds, cnt_wp, n_max, out.arclen, out.length,
            stag.knot_cum, stag.knot_count, stag.knot_z, stag.z)
    elif not _is_flat:
        warp_zprofile.apply_z_profile(
            config, seeds, cnt_wp, n_max, out.arclen, out.length, stag.z)

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

    # validity_inplace writes into out.valid directly. scratch.z drives the grade
    # gate; on the flat path z is zero-filled and z_valid_grade defaults to 0 -> the
    # grade branch is a no-op and validity is byte-identical to the legacy path.
    validity_inplace(stag.ctr2, scratch.w, cnt_wp, gv_wp,
                     stag.out2, stag.inn2, has_border, n_max, stag.z,
                     out.valid, config)

    # 6. Lift the vec2f staging buffers into the public vec3f Track arrays.
    if _is_flat:
        # Flat: constant z = z_base (0.0 by default). arclen/length/tangent keep the
        # 2D values from stages 2-3 (a constant z-offset changes neither). NaN xy
        # padding lifts to NaN-xy.
        wp.launch(_lift_track_k, dim=flat,
                  inputs=[stag.out2, stag.ctr2, stag.inn2, stag.tan2, stag.nrm2,
                          float(config.z_base),
                          out.outer, out.center, out.inner, out.tangent,
                          out.normal],
                  device=dev)
    else:
        # Non-flat: level cross-sections share the per-point centerline z, then
        # recompute tangent + true 3D arclen/length (overwriting the 2D tables).
        wp.launch(_lift_track_zvar_k, dim=flat,
                  inputs=[stag.out2, stag.ctr2, stag.inn2, stag.nrm2, stag.z,
                          out.outer, out.center, out.inner, out.normal],
                  device=dev)
        wp.launch(_track_frames3_k, dim=E,
                  inputs=[out.center, cnt_wp, n_max,
                          out.tangent, out.arclen, out.length],
                  device=dev)
    _sync(dev)

    return out


# Generation lives in warp_generate (single-pass corner sampling -> Bezier assemble ->
# arc-resample -> polygon fallback). warp_generate reuses this module's shared low-level
# primitives (_init/_sync/_fill_i32_k/_arc_resample_inplace/self_intersections_inplace),
# so the two modules reference each other. Generation symbols are re-exported here LAZILY
# (PEP 562 module __getattr__): any name not defined on this module is resolved from
# warp_generate on first attribute access, so warp_pipeline.<gen-name> keeps working for
# the orchestrator and the test suite while the import is deferred past module load (which
# avoids any module-load import cycle regardless of which module is imported first).
def __getattr__(name):
    if not name.startswith("__"):
        from . import warp_generate
        try:
            return getattr(warp_generate, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
