"""Frozen copies of the HEAD-1077b18 kernels that ``track_gen.ops`` replaces (roadmap PR7).

Generated from ``git show 1077b18:<file>``: every body is VERBATIM; the only edit is the
rename ``<name>`` -> ``_<origin><name>`` (origin = pipe | bezier | hull | polar | voronoi |
ckpt | rep | gate) applied to definitions and to the calls between them, so the copies
coexist in one module. ``tests/test_ops.py`` checks each public op bitwise against
these, which keeps the byte-identity contract checkable after the ``_src`` originals are
deleted. Do not edit the bodies.
"""
# ruff: noqa
# fmt: off
from __future__ import annotations

import warp as wp


# ------------------------------------------------------------------------------------
# pipe: track_gen/_src/warp_pipeline.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_pipeline.py:163
@wp.func
def _pipe_ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
    # Returns (q.y-o.y)*(p.x-o.x) - (p.y-o.y)*(q.x-o.x)
    return (qy - oy) * (px - ox) - (py - oy) * (qx - ox)


# track_gen/_src/warp_pipeline.py:168
@wp.func
def _pipe_nan0(x: float) -> float:
    # NaN/inf -> 0.0 guard (nan_to_num equivalent) for the border self-intersection
    # check. wp.where(cond, if_true, if_false) is the Warp 1.14 non-deprecated select
    # primitive. A NO-OP for finite inputs.
    return wp.where(wp.isfinite(x), x, 0.0)


# track_gen/_src/warp_pipeline.py:175
@wp.func
def _pipe_self_intersections_func(poly: wp.array(dtype=wp.vec2f), base: int, N: int) -> int:
    # Proper-crossing double-loop count for the env whose points start at `base`.
    # Each coordinate is read through _pipe_nan0 (NaN->0 guard), which is a no-op on finite
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
            aix = _pipe_nan0(Ai[0]); aiy = _pipe_nan0(Ai[1])
            bix = _pipe_nan0(Bi[0]); biy = _pipe_nan0(Bi[1])
            ajx = _pipe_nan0(Aj[0]); ajy = _pipe_nan0(Aj[1])
            bjx = _pipe_nan0(Bj[0]); bjy = _pipe_nan0(Bj[1])
            d1 = _pipe_ccw(ajx, ajy, bjx, bjy, aix, aiy)
            d2 = _pipe_ccw(ajx, ajy, bjx, bjy, bix, biy)
            d3 = _pipe_ccw(aix, aiy, bix, biy, ajx, ajy)
            d4 = _pipe_ccw(aix, aiy, bix, biy, bjx, bjy)
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


# track_gen/_src/warp_pipeline.py:214
@wp.kernel
def _pipe_self_intersections_by_i_k(
    poly: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    # One thread per segment start i. This preserves the same pair predicate as
    # _pipe_self_intersections_func, but avoids serializing every O(N^2) pair check onto
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
    aix = _pipe_nan0(Ai[0]); aiy = _pipe_nan0(Ai[1])
    bix = _pipe_nan0(Bi[0]); biy = _pipe_nan0(Bi[1])

    for j in range(i + 1, N):
        diff = j - i
        circ_dist = wp.min(diff, N - diff)
        if circ_dist <= 1:
            continue
        Aj = poly[base + j]
        Bj = poly[base + (j + 1) % N]
        ajx = _pipe_nan0(Aj[0]); ajy = _pipe_nan0(Aj[1])
        bjx = _pipe_nan0(Bj[0]); bjy = _pipe_nan0(Bj[1])
        d1 = _pipe_ccw(ajx, ajy, bjx, bjy, aix, aiy)
        d2 = _pipe_ccw(ajx, ajy, bjx, bjy, bix, biy)
        d3 = _pipe_ccw(aix, aiy, bix, biy, ajx, ajy)
        d4 = _pipe_ccw(aix, aiy, bix, biy, bjx, bjy)
        lj2 = (bjx - ajx) * (bjx - ajx) + (bjy - ajy) * (bjy - ajy)
        li2 = (bix - aix) * (bix - aix) + (biy - aiy) * (biy - aiy)
        ej = float(1.0e-3) * lj2
        ei = float(1.0e-3) * li2
        seg_ij = (d1 > ej and d2 < -ej) or (d1 < -ej and d2 > ej)
        seg_ji = (d3 > ei and d4 < -ei) or (d3 < -ei and d4 > ei)
        if seg_ij and seg_ji:
            wp.atomic_add(out, e, int(1))


# track_gen/_src/warp_pipeline.py:259
@wp.func
def _pipe_thickness_func(pts: wp.array(dtype=wp.vec2f), base: int, N: int, band: int) -> float:
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


# track_gen/_src/warp_pipeline.py:294
@wp.kernel
def _pipe_thickness_k(
    pts: wp.array(dtype=wp.vec2f),
    band: wp.array(dtype=wp.int32),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Delegates to _pipe_thickness_func with this env's band.
    e = wp.tid()
    out[e] = _pipe_thickness_func(pts, e * n_max, count[e], band[e])


# track_gen/_src/warp_pipeline.py:306
@wp.kernel
def _pipe_resample_scan_k(
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


# track_gen/_src/warp_pipeline.py:333
@wp.kernel
def _pipe_resample_lookup_k(
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


# track_gen/_src/warp_pipeline.py:368
@wp.kernel
def _pipe_cs_scan_k(c: wp.array(dtype=wp.vec2f), N: int, spacing: wp.float32, n_max: int,
               seg: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32),
               count: wp.array(dtype=wp.int32), truncated: wp.array(dtype=wp.int32)):
    # One thread per env e. Closed-loop seg lengths + cumulative arc s (len N+1),
    # then count = floor(total/spacing)+1, capped so target (count-1)*spacing < total
    # and to n_max. Mirrors geometry._resample_one's spacing branch. truncated[e] records
    # whether the UNCLAMPED count exceeded n_max (the loop's tail was dropped); the
    # validity gate fails such envs closed.
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
        truncated[e] = 0
        return
    k = int(wp.floor(total / spacing)) + 1
    while k > 1 and wp.float32(k - 1) * spacing >= total:
        k = k - 1
    count[e] = wp.min(wp.max(k, 1), n_max)
    truncated[e] = wp.where(k > n_max, int(1), int(0))


# track_gen/_src/warp_pipeline.py:399
@wp.kernel
def _pipe_cs_lookup_k(c: wp.array(dtype=wp.vec2f), seg: wp.array(dtype=wp.float32),
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


# track_gen/_src/warp_pipeline.py:423
@wp.kernel
def _pipe_arc_scan_k(
    dense: wp.array(dtype=wp.vec2f),
    M: int,
    num: int,
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
):
    # One thread per env e. NaN-aware generalization of _pipe_resample_scan_k: first
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


# track_gen/_src/warp_pipeline.py:468
@wp.kernel
def _pipe_arc_lookup_k(
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
    # real-loop arc length, then lerp. Mirrors _pipe_resample_lookup_k with variable R and
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


# track_gen/_src/warp_pipeline.py:510
@wp.kernel
def _pipe_arc_scan_selected_k(
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
    # Same as _pipe_arc_scan_k for active envs. Inactive envs write only their counts,
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


# track_gen/_src/warp_pipeline.py:553
@wp.kernel
def _pipe_arc_lookup_selected_k(
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


# track_gen/_src/warp_pipeline.py:590
@wp.func
def _pipe_turning_func(c: wp.array(dtype=wp.vec2f), base: int, N: int) -> float:
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


# track_gen/_src/warp_pipeline.py:609
@wp.kernel
def _pipe_turning_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Delegates to _pipe_turning_func over [e*n_max, e*n_max+count[e]).
    e = wp.tid()
    out[e] = _pipe_turning_func(c, e * n_max, count[e])


# track_gen/_src/warp_pipeline.py:767
@wp.kernel
def _pipe_fill_f32_k(arr: wp.array(dtype=wp.float32), v: float):
    # One thread per element: constant float fill.
    arr[wp.tid()] = v


# track_gen/_src/warp_pipeline.py:772
@wp.kernel
def _pipe_fill_i32_k(arr: wp.array(dtype=wp.int32), v: int):
    # One thread per element: constant int fill.
    arr[wp.tid()] = v


# ------------------------------------------------------------------------------------
# bezier: track_gen/_src/warp_generate.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_generate.py:34
@wp.kernel
def _bezier_corner_count_sample_k(
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


# track_gen/_src/warp_generate.py:84
@wp.kernel
def _bezier_corner_sample_k(
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


# track_gen/_src/warp_generate.py:136
@wp.kernel
def _bezier_corner_sample_style_k(
    seeds: wp.array(dtype=wp.int32),
    attempt: int,
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: wp.array(dtype=wp.float32),
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Per-env-scale variant of _bezier_corner_sample_k (method #1). BIT-IDENTICAL to it in every
    # respect except `scale` is read per env from a device array (scale[e]) instead of a
    # broadcast scalar. The RNG seeding/draw order is unchanged, so for a UNIFORM per-env
    # scale array equal to the config scalar this reproduces _bezier_corner_sample_k exactly.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 9781 + attempt)
    base = e * P
    sc = scale[e]
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
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
        nx = wp.randf(state) - 0.5
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * sc,
                                 (y * cell_size + ny) * sc)


# track_gen/_src/warp_generate.py:175
@wp.kernel
def _bezier_ccw_sort_k(
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


# track_gen/_src/warp_generate.py:221
@wp.func
def _bezier_safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    # Mirrors geometry.safe_normalize: v / clamp_min(||v||, 1e-8). The wp.max
    # floors finite lengths at 1e-8 and a NaN vector divides to (nan, nan)
    # so NaN propagates to BOTH components (matching oracle behavior on pruned corners).
    return v / wp.max(wp.length(v), 1.0e-8)


# ------------------------------------------------------------------------------------
# hull: track_gen/_src/warp_generate_hull.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_generate_hull.py:40
@wp.kernel
def _hull_point_count_sample_k(
    seeds: wp.array(dtype=wp.int32),
    min_num: int,
    max_num: int,
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. Per-env point count in [min_num, max_num] inclusive from a
    # single uniform draw. A multiplier (5119) distinct from the position stream's keeps
    # the count and position RNG streams decorrelated.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 5119 + 1)
    span = max_num - min_num + 1
    count = min_num + int(wp.randf(state) * float(span))
    out[e] = wp.min(count, max_num)


# track_gen/_src/warp_generate_hull.py:57
@wp.kernel
def _hull_point_sample_k(
    seeds: wp.array(dtype=wp.int32),
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: float,
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Samples P points on a num_cells x num_cells grid with a
    # bounded duplicate-cell rejection (mirrors the bezier corner sampler's spread, same
    # grid placement * scale so the coordinate range matches bezier). 7919 multiplier is
    # distinct from bezier's 9781 so the two generators draw different point sets.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 7919 + 1)
    base = e * P
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
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
        nx = wp.randf(state) - 0.5
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * scale,
                                 (y * cell_size + ny) * scale)


# track_gen/_src/warp_generate_hull.py:95
@wp.kernel
def _hull_angle_sort_k(
    points: wp.array(dtype=wp.vec2f),
    P: int,
    count: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Orders this env's FIRST m = count[e] points ascending by the
    # centroid-relative angle atan2(dx, dy) (X first, matching the bezier ccw-sort), about
    # the centroid of those m points. Rows [m, P) are written NaN. Insertion sort reads
    # only behind its write frontier, so uninitialised keys/out scratch is never consumed.
    e = wp.tid()
    base = e * P
    m = count[e]
    if m < 1:
        m = 1
    if m > P:
        m = P

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


# track_gen/_src/warp_generate_hull.py:202
@wp.func
def _hull_aug_vertex(c: wp.array(dtype=wp.vec2f), b: int, i: int, cnt: int) -> wp.vec2f:
    # Wrap index i into [0, cnt) (the augmented loop is closed) and read that vertex.
    return c[b + (i % cnt)]


# track_gen/_src/warp_generate_hull.py:208
@wp.kernel
def _hull_catmull_rom_k(
    aug: wp.array(dtype=wp.vec2f),
    aug_count: wp.array(dtype=wp.int32),
    AP: int,
    npseg: int,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per dense sample t (dim = E * AP * npseg). Decodes (e, segment i, sample
    # s), evaluates the closed Catmull-Rom spline of segment i (from vertex i to vertex
    # i+1) at u = s/(npseg-1) using the four control vertices i-1, i, i+1, i+2 (indices
    # wrapped mod cnt = aug_count[e]). Segments i >= cnt write NaN and drop out via the
    # arc-resample. AP = 2*P is the augmented-loop stride per env.
    t = wp.tid()
    per_env = AP * npseg
    e = t // per_env
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * AP
    cnt = aug_count[e]

    if i >= cnt or cnt < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    p0 = _hull_aug_vertex(aug, b, i - 1 + cnt, cnt)
    p1 = _hull_aug_vertex(aug, b, i, cnt)
    p2 = _hull_aug_vertex(aug, b, i + 1, cnt)
    p3 = _hull_aug_vertex(aug, b, i + 2, cnt)

    u = float(s) / float(npseg - 1)
    u2 = u * u
    u3 = u2 * u
    # Uniform Catmull-Rom basis (tension 0.5), passes through p1 (u=0) and p2 (u=1).
    out[t] = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


# track_gen/_src/warp_generate_hull.py:251
@wp.kernel
def _hull_assemble_polygon_selected_k(
    aug: wp.array(dtype=wp.vec2f),
    aug_count: wp.array(dtype=wp.int32),
    AP: int,
    npseg: int,
    active: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Straight-piece fallback over the augmented hull loop, only for envs whose smooth
    # Catmull-Rom resample self-crossed. Inactive rows are left untouched because the
    # selected arc-resampler skips them.
    t = wp.tid()
    per_env = AP * npseg
    e = t // per_env
    if active[e] <= 0:
        return

    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * AP
    cnt = aug_count[e]
    cm = wp.max(cnt, 1)

    if i >= cnt or cnt < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    c0 = aug[b + i]
    c1 = aug[b + (i + 1) % cm]
    u = float(s) / float(npseg - 1)
    out[t] = (1.0 - u) * c0 + u * c1


# ------------------------------------------------------------------------------------
# polar: track_gen/_src/warp_generate_polar.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_generate_polar.py:35
@wp.func
def _polar_catmull_rom(p0: wp.vec2f, p1: wp.vec2f, p2: wp.vec2f, p3: wp.vec2f, u: float) -> wp.vec2f:
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


# track_gen/_src/warp_generate_polar.py:72
@wp.func
def _polar_control_at(controls: wp.array(dtype=wp.vec2f), b: int, i: int, K: int) -> wp.vec2f:
    return controls[b + (i % K)]


# track_gen/_src/warp_generate_polar.py:77
@wp.kernel
def _polar_polar_spline_dense_k(
    controls: wp.array(dtype=wp.vec2f),
    K: int,
    samples_per_segment: int,
    dense: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Evaluate a closed uniform Catmull-Rom spline through the polar
    # controls. Output stride per env is K * samples_per_segment.
    e = wp.tid()
    cb = e * K
    M = K * samples_per_segment
    db = e * M

    for i in range(K):
        p0 = _polar_control_at(controls, cb, i - 1 + K, K)
        p1 = _polar_control_at(controls, cb, i, K)
        p2 = _polar_control_at(controls, cb, i + 1, K)
        p3 = _polar_control_at(controls, cb, i + 2, K)
        for s in range(samples_per_segment):
            # Endpoint-excluded within each segment; the next segment starts at p2. The
            # final dense point closes back to the first through arc-resample's wrap edge.
            u = float(s) / float(samples_per_segment)
            dense[db + i * samples_per_segment + s] = _polar_catmull_rom(p0, p1, p2, p3, u)


# track_gen/_src/warp_generate_polar.py:103
@wp.kernel
def _polar_normalize_centerline_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
):
    # One thread per env e. Center by bbox and scale so each env's longest bbox dimension
    # exactly matches target_extent after arc resampling.
    e = wp.tid()
    b = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(N):
        p = points[b + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(N):
        p = points[b + i]
        points[b + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


# ------------------------------------------------------------------------------------
# voronoi: track_gen/_src/warp_generate_voronoi.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_generate_voronoi.py:47
@wp.func
def _voronoi_catmull_rom(p0: wp.vec2f, p1: wp.vec2f, p2: wp.vec2f, p3: wp.vec2f, u: float) -> wp.vec2f:
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


# track_gen/_src/warp_generate_voronoi.py:194
@wp.func
def _voronoi_aug_at(aug: wp.array(dtype=wp.vec2f), base: int, i: int, count: int) -> wp.vec2f:
    return aug[base + (i % count)]


# track_gen/_src/warp_generate_voronoi.py:199
@wp.kernel
def _voronoi_catmull_rom_dense_k(
    aug: wp.array(dtype=wp.vec2f),
    AP: int,
    npseg: int,
    dense: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    per_env = AP * npseg
    e = t // per_env
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    base = e * AP
    p0 = _voronoi_aug_at(aug, base, i - 1 + AP, AP)
    p1 = _voronoi_aug_at(aug, base, i, AP)
    p2 = _voronoi_aug_at(aug, base, i + 1, AP)
    p3 = _voronoi_aug_at(aug, base, i + 2, AP)
    u = float(s) / float(npseg)
    dense[t] = _voronoi_catmull_rom(p0, p1, p2, p3, u)


# track_gen/_src/warp_generate_voronoi.py:221
@wp.kernel
def _voronoi_assemble_polygon_selected_k(
    aug: wp.array(dtype=wp.vec2f),
    AP: int,
    npseg: int,
    active: wp.array(dtype=wp.int32),
    dense: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    per_env = AP * npseg
    e = t // per_env
    if active[e] <= 0:
        return
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    base = e * AP
    p0 = aug[base + i]
    p1 = aug[base + (i + 1) % AP]
    u = float(s) / float(npseg)
    dense[t] = (1.0 - u) * p0 + u * p1


# track_gen/_src/warp_generate_voronoi.py:244
@wp.kernel
def _voronoi_normalize_centerline_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
):
    e = wp.tid()
    base = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(N):
        p = points[base + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(N):
        p = points[base + i]
        points[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


# ------------------------------------------------------------------------------------
# ckpt: track_gen/_src/warp_generate_checkpoint.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_generate_checkpoint.py:278
@wp.kernel
def _ckpt_normalize_centerline_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
):
    # One thread per env e. Center by bbox and scale so each env's longest bbox dimension matches
    # target_extent (identical to warp_generate_polar._normalize_centerline_k).
    e = wp.tid()
    b = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)
    for i in range(N):
        p = points[b + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)
    for i in range(N):
        p = points[b + i]
        points[b + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


# ------------------------------------------------------------------------------------
# rep: track_gen/_src/warp_generate_repulsive.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_generate_repulsive.py:250
@wp.func
def _rep_safe_dir(v: wp.vec2f) -> wp.vec2f:
    # safe_normalize with the oracle's 1e-8 floor (matches geometry.safe_normalize).
    return v / wp.max(wp.length(v), float(1.0e-8))


# ------------------------------------------------------------------------------------
# gate: track_gen/_src/warp_gate.py @ 1077b18
# ------------------------------------------------------------------------------------


# track_gen/_src/warp_gate.py:24
@wp.func
def _gate_safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    return v / wp.max(wp.length(v), 1.0e-8)


# track_gen/_src/warp_gate.py:29
@wp.func
def _gate_cross2(a: wp.vec2f, b: wp.vec2f) -> float:
    return a[0] * b[1] - a[1] * b[0]


# track_gen/_src/warp_gate.py:34
@wp.func
def _gate_proper_segment_intersection(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f) -> int:
    ab = b - a
    cd = d - c
    ac = c - a
    ad = d - a
    ca = a - c
    cb = b - c
    o1 = _gate_cross2(ab, ac)
    o2 = _gate_cross2(ab, ad)
    o3 = _gate_cross2(cd, ca)
    o4 = _gate_cross2(cd, cb)
    hit_ab = (o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)
    hit_cd = (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    if hit_ab and hit_cd:
        return int(1)
    return int(0)


# track_gen/_src/warp_gate.py:53
@wp.kernel
def _gate_fill_vec2_k(arr: wp.array(dtype=wp.vec2f), value: wp.vec2f):
    arr[wp.tid()] = value


# track_gen/_src/warp_gate.py:58
@wp.kernel
def _gate_fill_vec3_k(arr: wp.array(dtype=wp.vec3f), value: wp.vec3f):
    arr[wp.tid()] = value


# track_gen/_src/warp_gate.py:63
@wp.kernel
def _gate_fill_quat_k(arr: wp.array(dtype=wp.quatf), value: wp.quatf):
    arr[wp.tid()] = value


# track_gen/_src/warp_gate.py:68
@wp.kernel
def _gate_fill_f32_k(arr: wp.array(dtype=wp.float32), value: float):
    arr[wp.tid()] = value


# track_gen/_src/warp_gate.py:73
@wp.kernel
def _gate_fill_i32_k(arr: wp.array(dtype=wp.int32), value: int):
    arr[wp.tid()] = value


# track_gen/_src/warp_gate.py:115
@wp.kernel
def _gate_order_ccw_k(
    src: wp.array(dtype=wp.vec2f),
    src_stride: int,
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    keys: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.vec2f),
):
    e = wp.tid()
    src_base = e * src_stride
    dst_base = e * max_gates
    m = count[e]
    if m < 0:
        m = 0
    if m > src_stride:
        m = src_stride
    if m > max_gates:
        m = max_gates

    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(max_gates):
        if i < m:
            p = src[src_base + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])

    cx = wp.float32(0.0)
    cy = wp.float32(0.0)
    if m > 0:
        cx = wp.float32(sx / wp.float64(m))
        cy = wp.float32(sy / wp.float64(m))

    for i in range(max_gates):
        if i < m:
            p = src[src_base + i]
            key = wp.atan2(p[0] - cx, p[1] - cy)
            j = i - 1
            while j >= 0 and keys[dst_base + j] > key:
                keys[dst_base + j + 1] = keys[dst_base + j]
                dst[dst_base + j + 1] = dst[dst_base + j]
                j = j - 1
            keys[dst_base + j + 1] = key
            dst[dst_base + j + 1] = p
        else:
            dst[dst_base + i] = wp.vec2f(wp.nan, wp.nan)


# track_gen/_src/warp_gate.py:217
@wp.kernel
def _gate_normalize_positions_k(
    position: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_extent: float,
):
    e = wp.tid()
    base = e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if cnt > max_gates:
        cnt = max_gates

    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            min_x = wp.min(min_x, p[0])
            max_x = wp.max(max_x, p[0])
            min_y = wp.min(min_y, p[1])
            max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            position[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)
        else:
            position[base + i] = wp.vec2f(wp.nan, wp.nan)
