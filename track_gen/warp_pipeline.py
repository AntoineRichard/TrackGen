# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Warp track-generation pipeline kernels.

Every pipeline stage (generation, resample, relax, inflate) is expressed as Warp
kernels that run on BOTH the Warp ``cpu`` device (tests/CI, GPU-free) and ``cuda``
(production), with torch acting only as the array container (``wp.from_torch`` at the
boundary). The whole pipeline is graph-capturable on CUDA. During the port each kernel
is verified ``allclose`` against the equivalent torch function (the oracle).

Convention: one thread per output element; flat arrays ``[E*N]`` of ``wp.vec2f`` and
``[E]`` per-env scalars; env index ``e = tid // N``; launch with
``device=str(tensor.device)``.
"""
from __future__ import annotations

import math

import torch

from . import warp_relax  # pure-Warp XPBD solve (cpu+cuda); part of the pure-Warp impl

try:
    import warp as wp
    _HAVE_WARP = True
except Exception:  # warp is an optional extra
    _HAVE_WARP = False

_INITED = False

# True only inside generate_tracks_warp_graph's capture/warmup region. While set, every
# wrapper's _sync() is a no-op (host-blocking sync is ILLEGAL during CUDA graph capture)
# and warp_relax.xpbd_solve skips its own wp.synchronize(). The public eager path never
# sets it, so eager behaviour is unchanged. Module-global because the whole pipeline (and
# warp_relax) must agree, and the captured region is single-threaded/serial by construction.
_CAPTURING = False


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch / wp.from_torch."""
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


def _mean_seg_len_torch(center: torch.Tensor) -> torch.Tensor:
    """Mean closed-loop segment length per env (perimeter / N). center [E, N, 2] -> [E].

    Reproduces geometry.mean_seg_len (= geometry.perimeter / N) with the identical
    roll-and-norm formula, but without importing geometry at runtime (warp_pipeline
    must stay free of the torch oracle modules). Reused by the validity / inflate band
    derivation; the caller applies any clamp_min itself, matching the oracle's call site.
    """
    seg = torch.roll(center, -1, dims=1) - center
    return torch.linalg.norm(seg, dim=-1).sum(dim=1) / center.shape[1]


if _HAVE_WARP:

    @wp.kernel
    def _offset_build_k(
        center: wp.array(dtype=wp.vec2f),
        Nrm: wp.array(dtype=wp.vec2f),
        half_width: float,
        N: int,
        area_a: wp.array(dtype=wp.float32),
        area_b: wp.array(dtype=wp.float32),
    ):
        # One thread per point t.  e = env index, i = point-within-env index.
        # Accumulates the per-env signed shoelace cross-product terms for
        # candidate polygons a (center + hw*Nrm) and b (center - hw*Nrm) via
        # atomic adds.  area_a/area_b must be zero-initialised before launch.
        t = wp.tid()
        e = t // N
        i = t % N
        b_base = e * N

        # Offset points at this thread's index.
        ct = center[t]
        nt = Nrm[t]
        at = ct + half_width * nt
        bt = ct - half_width * nt

        # Offset points at the NEXT index (wraps within env).
        next_idx = b_base + (i + 1) % N
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
        N: int,
        area_a: wp.array(dtype=wp.float32),
        area_b: wp.array(dtype=wp.float32),
        outer: wp.array(dtype=wp.vec2f),
        inner: wp.array(dtype=wp.vec2f),
    ):
        # One thread per point t.  Recompute a[t], b[t] (cheap), then assign
        # outer/inner based on which candidate has the larger |signed area|.
        t = wp.tid()
        e = t // N
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
    def _double_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        # Smoke-test kernel: out[i] = 2*x[i]. Exercises the Warp cpu/cuda launch path
        # (used only by the scaffolding smoke test, _smoke_double).
        i = wp.tid()
        out[i] = 2.0 * x[i]

    @wp.kernel
    def _frame_k(c: wp.array(dtype=wp.vec2f), N: int,
                 T: wp.array(dtype=wp.vec2f), Nrm: wp.array(dtype=wp.vec2f),
                 kappa: wp.array(dtype=wp.float32)):
        # Per closed-loop point: central-difference unit tangent, left-normal, and
        # non-negative Menger curvature. Matches geometry.tangents_normals + menger_curvature.
        t = wp.tid()
        e = t // N
        i = t % N
        b = e * N
        xp = c[b + ((i + N - 1) % N)]
        xc = c[t]
        xn = c[b + ((i + 1) % N)]
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
        # NaN/inf -> 0.0 guard, reproducing torch.nan_to_num(..., nan=0.0) for the border
        # self-intersection check. wp.where(cond, if_true, if_false) is the Warp 1.14
        # non-deprecated select primitive. A NO-OP for finite inputs, so the
        # self_intersections oracle (torch.equal) test is unaffected.
        return wp.where(wp.isfinite(x), x, 0.0)

    @wp.func
    def _self_intersections_func(poly: wp.array(dtype=wp.vec2f), base: int, N: int) -> int:
        # Proper-crossing double-loop count for the env whose points start at `base`.
        # Each coordinate is read through _nan0 (NaN->0 guard), which is a no-op on finite
        # inputs (so the self_intersections torch.equal test stays exact) and reproduces
        # the validity border check's torch.nan_to_num(outer/inner, nan=0.0).
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
                seg_ij = (d1 > 0.0) != (d2 > 0.0)
                seg_ji = (d3 > 0.0) != (d4 > 0.0)
                if seg_ij and seg_ji:
                    count = count + 1
        return count

    @wp.kernel
    def _self_intersections_k(
        poly: wp.array(dtype=wp.vec2f),
        N: int,
        out: wp.array(dtype=wp.int32),
    ):
        # One thread per env e. Delegates to _self_intersections_func over [e*N, e*N+N).
        e = wp.tid()
        out[e] = _self_intersections_func(poly, e * N, N)

    @wp.kernel
    def _sep_min_k(
        pts: wp.array(dtype=wp.vec2f),
        band: wp.array(dtype=wp.int32),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. Min distance over pairs with circ_dist > band[e].
        e = wp.tid()
        b = band[e]
        sep_min = float(1.0e30)
        for i in range(N):
            for j in range(i + 1, N):
                diff = j - i
                circ_dist = wp.min(diff, N - diff)
                if circ_dist > b:
                    pi = pts[e * N + i]
                    pj = pts[e * N + j]
                    d = wp.length(pi - pj)
                    sep_min = wp.min(sep_min, d)
        out[e] = sep_min

    @wp.kernel
    def _curvrad_min_k(
        pts: wp.array(dtype=wp.vec2f),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. 1 / max Menger curvature.
        e = wp.tid()
        kappa_max = float(0.0)
        for i in range(N):
            xp = pts[e * N + (i + N - 1) % N]
            xc = pts[e * N + i]
            xn = pts[e * N + (i + 1) % N]
            a = xc - xp
            bb = xn - xc
            cc = xn - xp
            cross = a[0] * bb[1] - a[1] * bb[0]
            area = 0.5 * wp.abs(cross)
            denom = wp.max(wp.length(a) * wp.length(bb) * wp.length(cc), float(1.0e-12))
            kappa = 4.0 * area / denom
            kappa_max = wp.max(kappa_max, kappa)
        out[e] = 1.0 / wp.max(kappa_max, float(1.0e-12))

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
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. Delegates to _thickness_func with this env's band.
        e = wp.tid()
        out[e] = _thickness_func(pts, e * N, N, band[e])

    @wp.kernel
    def _resample_scan_k(
        c: wp.array(dtype=wp.vec2f),
        N: int,
        seg: wp.array(dtype=wp.float32),
        s: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. Pure Warp: segment lengths seg[e*N+i]=|c[i+1]-c[i]|
        # (i+1 wraps) and the cumulative arc length s[e*(N+1)+0]=0, s[..+i+1]=s[..+i]+seg[i].
        # The running sum is accumulated in float64 to limit drift vs the torch oracle's
        # cumsum (the residual ~1e-4 is float32 sqrt rounding, geometrically negligible).
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

    @wp.kernel
    def _resample_lookup_k(
        c: wp.array(dtype=wp.vec2f),
        seg: wp.array(dtype=wp.float32),
        s: wp.array(dtype=wp.float32),
        N: int,
        out: wp.array(dtype=wp.vec2f),
    ):
        # One thread per output point t; e = env index, k = point-within-env index.
        # Finds the segment containing target arc-length tk via linear scan matching
        # searchsorted(right=False).clamp(max=N-1), then lerps.
        t = wp.tid()
        e = t // N
        k = t % N
        eb = e * N
        es = e * (N + 1)

        total = s[es + N]
        tk = float(k) * total / float(N)

        # Linear scan: first j with s[es+j+1] >= tk, clamped to N-1.
        idx = int(0)
        while idx < N - 1 and s[es + idx + 1] < tk:
            idx = idx + 1

        s0 = s[es + idx]
        segl = wp.max(seg[eb + idx], float(1.0e-12))
        frac = wp.clamp((tk - s0) / segl, float(0.0), float(1.0))

        # p0 = c[eb+idx]; p1 = c[eb+(idx+1)%N]
        p0 = c[eb + idx]
        p1 = c[eb + (idx + 1) % N]
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
        if total <= 0.0:
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
        # Public arc-resample count (folds in the wrapper's torch.where): R>=2 -> num, else 0.
        count_out[e] = wp.where(r >= 2, num, 0)

        # R >= 2: closed-loop arc length over real_pts[0..R-1]. seg[j] = |real[(j+1)%R]
        # - real[j]| (j=R-1 is the wrap |real[0]-real[R-1]|); s[0]=0, s[j+1]=s[j]+seg[j].
        # Accumulate in float64 to limit drift vs the torch oracle's cumsum.
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
        # edge gives atan2(0,0)=0, matching the torch safe_normalize-then-atan2 case).
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
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. Delegates to _turning_func over [e*N, e*N+N).
        e = wp.tid()
        out[e] = _turning_func(c, e * N, N)

    @wp.kernel
    def _validity_k(
        center: wp.array(dtype=wp.vec2f),
        w: wp.array(dtype=wp.float32),
        count: wp.array(dtype=wp.int32),
        gen_valid: wp.array(dtype=wp.int32),
        outer: wp.array(dtype=wp.vec2f),
        inner: wp.array(dtype=wp.vec2f),
        has_border: int,
        N: int,
        half_width: float,
        turning_tol: float,
        w_floor: float,
        relax_tol: float,
        out: wp.array(dtype=wp.int32),
    ):
        # One thread per env e. Fuses inflation._validity_stage entirely in-kernel:
        # gen_valid AND turning AND width-floor AND no-NaN AND thickness AND border-simple.
        # All sub-results are 0/1 int flags (Warp can't AND Python bools in dynamic loops).
        e = wp.tid()
        base = e * N

        # --- turning ---
        turn = _turning_func(center, base, N)
        turn_ok = int(0)
        if wp.abs(wp.abs(turn) - 2.0 * wp.pi) <= turning_tol:
            turn_ok = int(1)

        # --- real-point mask (i < count[e]): width floor + no NaN over real points ---
        cnt = count[e]
        w_ok = int(1)
        no_nan = int(1)
        for i in range(N):
            if i < cnt:
                if not (w[base + i] > w_floor):
                    w_ok = int(0)
                ci = center[base + i]
                if not (wp.isfinite(ci[0]) and wp.isfinite(ci[1])):
                    no_nan = int(0)

        # --- band = round(2*hw / (perimeter/N)).clamp_min(1); perimeter = closed-loop sum ---
        peri = float(0.0)
        for i in range(N):
            peri += wp.length(center[base + (i + 1) % N] - center[base + i])
        L0 = wp.max(peri / float(N), float(1.0e-9))
        band = wp.max(int(wp.round(2.0 * half_width / L0)), 1)

        # --- thickness gate ---
        th = _thickness_func(center, base, N, band)
        th_ok = int(0)
        if th >= (1.0 - relax_tol) * half_width:
            th_ok = int(1)

        # --- border self-intersection gate (skipped when has_border == 0) ---
        border_ok = int(1)
        if has_border == 1:
            cross = _self_intersections_func(outer, base, N) + _self_intersections_func(inner, base, N)
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
        N: int,
        arclen: wp.array(dtype=wp.float32),
        length: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. FIXED-mode arc length (count == N, all real):
        # seg_len[i] = |c[(i+1)%N] - c[i]| (i=N-1 is the wrap segment),
        # arclen[i] = sum_{j<i} seg_len[j] (arclen[0]=0; wrap NOT in any arclen entry),
        # length    = sum_i seg_len[i] (full closed perimeter, wrap INCLUDED).
        # Running sum is accumulated in float64 to limit drift vs the torch oracle's
        # cumsum (residual ~1e-3 over N segments is float32 sqrt rounding).
        e = wp.tid()
        b = e * N
        acc = wp.float64(0.0)
        for i in range(N):
            arclen[b + i] = wp.float32(acc)            # arc length BEFORE segment i
            d = c[b + (i + 1) % N] - c[b + i]
            acc = acc + wp.float64(wp.length(d))       # add segment i (i=N-1 is the wrap)
        length[e] = wp.float32(acc)

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
        # ACCEPTED RNG REDESIGN (does NOT match the torch _sample_corner_points
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
        keys: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.vec2f),
    ):
        # One thread per env e. Orders this env's P corners ascending by the
        # centroid-relative angle key = atan2(dx, dy) (X FIRST, matching the
        # geometry.ccw_sort quirk). The output `out` and scratch `keys` buffers
        # double as the sorted-prefix arrays for an in-place insertion sort; the
        # original corner is always read from the read-only `points` input.
        e = wp.tid()
        base = e * P

        # Centroid (sums accumulated in float64 to match torch.mean closely and
        # keep the angular ordering robust at the ULP level).
        sx = wp.float64(0.0)
        sy = wp.float64(0.0)
        for i in range(P):
            p = points[base + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])
        cx = wp.float32(sx / wp.float64(P))
        cy = wp.float32(sy / wp.float64(P))

        # Stable insertion sort (strict `>` -> stable for distinct keys).
        for c in range(P):
            p = points[base + c]
            key = wp.atan2(p[0] - cx, p[1] - cy)   # X first!
            j = c - 1
            while j >= 0 and keys[base + j] > key:
                keys[base + j + 1] = keys[base + j]
                out[base + j + 1] = out[base + j]
                j = j - 1
            keys[base + j + 1] = key
            out[base + j + 1] = p

    @wp.func
    def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
        # Mirrors geometry.safe_normalize: v / clamp_min(||v||, 1e-8). The wp.max
        # floors finite lengths at 1e-8 exactly like torch.clamp_min, and a NaN
        # vector divides to (nan, nan) (NaN/eps = nan) so NaN propagates to BOTH
        # components, matching the torch oracle bit-for-bit on pruned corners.
        return v / wp.max(wp.length(v), 1.0e-8)

    @wp.func
    def _pruned_corner(c: wp.array(dtype=wp.vec2f), b: int, i: int, cnt: int) -> wp.vec2f:
        # Folds in _prune_corners' NaN step: corner i of the env at base b is real iff
        # i < cnt; rows i >= cnt are replaced by (nan, nan), reproducing the torch
        # where(arange(P) < count, corners, nan) prune EXACTLY (same NaN positions). The
        # downstream safe_normalize/bezier NaN-propagation then matches the oracle.
        ci = c[b + i]
        return wp.where(i < cnt, ci, wp.vec2f(wp.nan, wp.nan))

    @wp.kernel
    def _vertex_tangents_k(
        c: wp.array(dtype=wp.vec2f),
        count: wp.array(dtype=wp.int32),
        P: int,
        p: float,
        tangents: wp.array(dtype=wp.vec2f),
    ):
        # One thread per corner t (dim = E*P); e = env index, i = corner-within-env.
        # Mirrors geometry.vertex_tangents: u_out_i = dir(i -> i+1), u_in_i = dir(i-1 -> i)
        # (== roll(u_out, +1)); tangent_i = safe_normalize(p*u_out + (1-p)*u_in). The
        # count->NaN prune is folded in-kernel (corner i is NaN iff i >= count[e]); NaN at
        # any pruned corner propagates the same way the torch oracle's safe_normalize does.
        t = wp.tid()
        e = t // P
        i = t % P
        b = e * P
        cnt = count[e]
        c_i = _pruned_corner(c, b, i, cnt)
        c_next = _pruned_corner(c, b, (i + 1) % P, cnt)
        c_prev = _pruned_corner(c, b, (i + P - 1) % P, cnt)
        u_out = _safe_normalize2(c_next - c_i)
        u_in = _safe_normalize2(c_i - c_prev)
        blended = p * u_out + (1.0 - p) * u_in
        tangents[t] = _safe_normalize2(blended)

    @wp.kernel
    def _assemble_k(
        c: wp.array(dtype=wp.vec2f),
        count: wp.array(dtype=wp.int32),
        tangents: wp.array(dtype=wp.vec2f),
        P: int,
        npseg: int,
        rad: float,
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

        c0 = _pruned_corner(c, b, i, cnt)
        c1 = _pruned_corner(c, b, (i + 1) % P, cnt)
        t0 = tangents[b + i]
        t1 = tangents[b + (i + 1) % P]

        chord = wp.length(c1 - c0)
        handle = rad * chord
        p1 = c0 + t0 * handle    # leave c0 along its tangent
        p2 = c1 - t1 * handle    # arrive at c1 along its tangent

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
        # ACCEPTED RNG REDESIGN (does NOT match the torch oracle's per-env corner-count
        # draw bit-for-bit; validated by range/reproducibility only). One thread per env e.
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
    def _fill_vec2_k(arr: wp.array(dtype=wp.vec2f), vx: float, vy: float):
        # One thread per element: constant vec2 fill (pure-Warp torch.full replacement).
        arr[wp.tid()] = wp.vec2f(vx, vy)

    @wp.kernel
    def _fill_f32_k(arr: wp.array(dtype=wp.float32), v: float):
        # One thread per element: constant float fill (pure-Warp torch.full replacement).
        arr[wp.tid()] = v

    @wp.kernel
    def _fill_i32_k(arr: wp.array(dtype=wp.int32), v: int):
        # One thread per element: constant int fill (pure-Warp torch.full/zeros/ones).
        arr[wp.tid()] = v

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
        # select saw the OLD valid). Folds the torch `valid = valid | accept`.
        e = wp.tid()
        if accept[e] != 0:
            valid[e] = 1

    @wp.kernel
    def _band_l0_k(
        center: wp.array(dtype=wp.vec2f),
        N: int,
        two_hw: float,
        band_out: wp.array(dtype=wp.int32),
        l0_out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. L0 = perimeter/N (closed-loop mean segment length; mirrors
        # _mean_seg_len_torch). band = round(2*hw / L0).clamp_min(1); the isfinite guard on
        # 2*hw/L0 reproduces the torch nan_to_num(nan=1.0,posinf=1.0,neginf=1.0).round()
        # .long().clamp_min(1) for invalid (NaN-centerline) envs -> band 1. L0 itself may stay
        # NaN for invalid envs (that flows untouched into xpbd, which propagates the NaN).
        e = wp.tid()
        base = e * N
        peri = float(0.0)
        for i in range(N):
            peri += wp.length(center[base + (i + 1) % N] - center[base + i])
        l0 = peri / float(N)
        l0_out[e] = l0
        bf = two_hw / wp.max(l0, float(1.0e-9))
        band_out[e] = wp.where(wp.isfinite(bf), wp.max(int(wp.round(bf)), 1), 1)


def corner_count_sample(seeds: torch.Tensor, attempt: int, config) -> torch.Tensor:
    """Sample a per-env corner COUNT in [min_num_points, max_num_points] via Warp RNG.

    ACCEPTED RNG REDESIGN: pure-Warp replacement for the torch oracle's per-env corner
    count draw in BezierCenterlineGenerator.generate. It does NOT reproduce the oracle
    bit-for-bit (the legacy PerEnvSeededRNG is retired from the pure-Warp path); it is
    validated by range and reproducibility only. One thread per env (see
    _corner_count_sample_k): state = rand_init(seed*6151 + attempt) -- a DISTINCT seed
    multiplier from corner_sample's 9781 so corner counts and corner positions are
    uncorrelated -- then one randf maps to an inclusive integer in the closed range.

    Args:
        seeds:   [E] int per-env base seed (narrowed to int32 before the seed mix).
        attempt: int retry counter (mixed into the seed for attempt-to-attempt variety).
        config:  TrackGenConfig (uses min_num_points, max_num_points).

    Returns:
        [E] long corner count per env, every value in [min_num_points, max_num_points].
        Pure Warp (cpu+cuda); reproducible per (seeds, attempt, config) within a device.
    """
    _init()
    E = seeds.shape[0]
    dev = str(seeds.device)
    min_num = int(config.min_num_points)
    max_num = int(config.max_num_points)

    seeds_i32 = seeds.to(torch.int32).contiguous()
    out_t = torch.empty(E, device=seeds.device, dtype=torch.int32)
    wp.launch(
        _corner_count_sample_k, dim=E,
        inputs=[wp.from_torch(seeds_i32, dtype=wp.int32), int(attempt),
                min_num, max_num, wp.from_torch(out_t, dtype=wp.int32)],
        device=dev,
    )
    _sync(seeds.device)
    return out_t.long()


def generate_centerline_warp(seeds: torch.Tensor, config):
    """Static, fixed-iteration, masked accept-first-valid centerline generation.

    Pure-Warp drop-in for BezierCenterlineGenerator.generate with the downstream final
    arc-length resample to ``num_points`` FUSED in (returns the 256-resampled centerline
    directly). Composes the verified wrappers in the oracle's order: per attempt sample
    corners -> ccw_sort -> sample a per-env corner count -> assemble dense (NaN-pruned) ->
    gate (angle & turn & finite & simple) -> resample the gated dense to num_points.

    The loop runs a FIXED ``config.max_regen_iters`` times for ALL envs (no host
    data-branching, no early exit on valid.all()) so the whole thing is graph-capturable.
    Each env keeps its FIRST accepted candidate: ``take = accept & ~valid`` selects only
    envs newly accepted this attempt, then ``valid |= accept``; later attempts recompute
    but never overwrite an already-valid env. The stored centerline is the SAME
    arc_length_resample_warp(dense, num_points) that ``gates`` checked simple_ok on, so
    every valid env's centerline is simple by construction.

    Args:
        seeds:  [E] int per-env base seed.
        config: TrackGenConfig (uses max_regen_iters, num_points, and the fields the
                composed wrappers read: max_num_points, min/max_num_points, min_angle,
                turning_tol, rad, edgy, num_points_per_segment, min_point_distance, scale).

    Returns:
        (centerline [E, num_points, 2] float32, valid [E] bool). Invalid envs keep an
        all-NaN centerline row. Pure Warp + torch glue (cpu+cuda); no oracle-module imports.
    """
    _init()
    E = seeds.shape[0]
    N = int(config.num_points)
    dev = str(seeds.device)

    # Persistent buffers updated IN PLACE across the fixed loop (this in-place pattern is
    # also what makes the later CUDA graph capture work). torch.empty is just an I/O alloc;
    # the kernel fills below (NaN centerline, valid=0) replace torch.full/torch.zeros so no
    # torch compute touches them.
    centerline = torch.empty(E * N, 2, device=seeds.device, dtype=torch.float32)
    valid = torch.empty(E, device=seeds.device, dtype=torch.int32)
    cl_w = wp.from_torch(centerline, dtype=wp.vec2f)
    valid_w = wp.from_torch(valid, dtype=wp.int32)
    nan = float("nan")
    wp.launch(_fill_vec2_k, dim=E * N, inputs=[cl_w, nan, nan], device=dev)
    wp.launch(_fill_i32_k, dim=E, inputs=[valid_w, 0], device=dev)

    for k in range(int(config.max_regen_iters)):
        corners = ccw_sort(corner_sample(seeds, k, config))   # [E, P, 2]
        count = corner_count_sample(seeds, k, config)         # [E]
        dense = assemble(corners, count, config)              # [E, M, 2] (NaN-pruned)
        accept = gates(corners, dense, count, config)         # [E] bool
        rs, _ = arc_length_resample_warp(dense, N)            # [E, N, 2] (the gated centerline)
        accept_w = wp.from_torch(accept.to(torch.int32).contiguous(), dtype=wp.int32)
        rs_w = wp.from_torch(rs.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
        # accept-FIRST-valid: select reads the OLD valid (take = accept & ~valid), then
        # _or_update_k does valid |= accept AFTER so the select saw the pre-update flag.
        wp.launch(_select_first_valid_k, dim=E * N,
                  inputs=[accept_w, valid_w, rs_w, cl_w, N], device=dev)
        wp.launch(_or_update_k, dim=E, inputs=[accept_w, valid_w], device=dev)

    _sync(seeds.device)
    return centerline.view(E, N, 2), valid.bool()


def generate_tracks_warp(config, seeds: torch.Tensor):
    """End-to-end pure-Warp track generation: a drop-in for TrackGenerator.generate.

    Composes the verified pure-Warp stages in the torch oracle's order
    (TrackGenerator.generate -> _resample_stage -> relaxation.relax -> inflate):

      1. ``generate_centerline_warp`` -> [E, N, 2] centerline already arc-length resampled
         to ``num_points`` (the oracle's dense->N resample is fused in) plus the [E] bool
         generation flag. Never-accepted envs carry an all-NaN centerline row.
      2. Relax: band = round(2*half_width / mean_seg_len).clamp_min(1) and rest length
         L0 = perimeter/N (mean_seg_len), then ``warp_relax.xpbd_solve`` (a fused pure-Warp
         XPBD solve that runs on cpu AND cuda; it is called DIRECTLY, not via the torch
         relaxation.relax which only takes the Warp path on cuda). The band is guarded with
         nan_to_num so an invalid env's NaN mean_seg_len still yields a valid int band (the
         kernel must not choke); the NaN otherwise flows untouched through relax + resample.
      3. ``resample_uniform`` re-uniformizes (matches relaxation._relax_xpbd's final resample).
      4. ``inflate_warp`` builds the Track, with the generation flag passed as ``valid``.

    Invalid (never-accepted) envs propagate NaN through relax + resample, so inflate_warp's
    validity gate marks them invalid (no_nan=False AND gen_valid=False). This is the intended
    static-batch behaviour: a single fixed-size launch, no per-env host branching, fully
    graph-capturable on cuda. Validated by YIELD / WIDTH / SHAPE aggregates (Warp RNG produces
    different tracks than the torch oracle, so there is no per-env allclose).

    Only the default relaxation is ported: ``relax_solver`` must be "xpbd" and
    ``smooth_finish`` must be False (asserted); ``relax_band`` is honored if set.

    Args:
        config: TrackGenConfig (output_mode must be "fixed"; relax_solver="xpbd",
                smooth_finish=False; uses num_points, half_width, relax_band and the
                fields the composed stages read).
        seeds:  [E] int per-env base seed (the only per-env input).

    Returns:
        track_gen.types.Track with all fields shaped per inflate_warp. Pure Warp + torch glue
        (cpu+cuda); no oracle-module imports.
    """
    # Only the default (XPBD, no finisher) relaxation is ported to pure Warp. Fail loudly
    # rather than silently diverge from the torch oracle if the facade passes other knobs.
    assert config.relax_solver == "xpbd", \
        f"generate_tracks_warp only supports relax_solver='xpbd', got {config.relax_solver!r}"
    assert not config.smooth_finish, \
        "generate_tracks_warp does not implement the smooth_finish (tp_sobolev) pass"

    N = int(config.num_points)
    hw = float(config.half_width)

    centerline, gen_valid = generate_centerline_warp(seeds, config)   # [E, N, 2], [E] bool
    E = centerline.shape[0]
    dev = str(centerline.device)

    # band = round(2*hw / (perimeter/N)).clamp_min(1) and rest length L0 = perimeter/N,
    # BOTH computed in one pure-Warp kernel (folds _mean_seg_len_torch + the nan_to_num /
    # round / long / clamp_min torch glue). L0 stays NaN for invalid envs and flows untouched
    # through xpbd; the band's isfinite guard yields a valid int band there.
    band = torch.empty(E, device=centerline.device, dtype=torch.int32)
    L0 = torch.empty(E, device=centerline.device, dtype=torch.float32)
    cl_w = wp.from_torch(centerline.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    band_w = wp.from_torch(band, dtype=wp.int32)
    l0_w = wp.from_torch(L0, dtype=wp.float32)
    wp.launch(_band_l0_k, dim=E, inputs=[cl_w, N, 2.0 * hw, band_w, l0_w], device=dev)

    if config.relax_band is not None:
        # Honor an explicit per-track band override exactly like relaxation._band: overwrite
        # band with the config constant (L0 from _band_l0_k above is still the xpbd rest
        # length). Host branch on a CONFIG value (not tensor data) -> capture-safe.
        wp.launch(_fill_i32_k, dim=E, inputs=[band_w, int(config.relax_band)], device=dev)

    _sync(centerline.device)
    relaxed = warp_relax.xpbd_solve(centerline, band, L0, config)     # pure Warp (cpu+cuda)
    relaxed = resample_uniform(relaxed, N)                            # final re-uniform
    return inflate_warp(relaxed, config, valid=gen_valid)


def assemble(corners: torch.Tensor, count: torch.Tensor, config) -> torch.Tensor:
    """Build the closed dense Bezier centerline from ccw-ordered corners.

    Pure-Warp drop-in for BezierCenterlineGenerator._assemble_centerline (with the
    _prune_corners NaN step folded in): corner row i of env e is kept iff i < count[e],
    else replaced by NaN. Then per-corner blended unit tangents are computed and each of
    the P closed segments is sampled as a cubic Bezier (npseg samples), yielding a dense
    ``[E, P*npseg, 2]`` polyline. NaN from pruned corners lands in the SAME positions as
    the oracle. Pure Warp (cpu+cuda); allclose to the oracle to atol=1e-4 (float32 sqrt
    drift in safe_normalize).

    Args:
        corners: [E, P, 2] float32 ccw-ordered corners (P == config.max_num_points).
        count:   [E] int real-corner count per env; rows >= count are pruned to NaN.
        config:  TrackGenConfig (uses edgy, rad, max_num_points, num_points_per_segment).

    Returns:
        [E, P * num_points_per_segment, 2] float32 dense closed centerline (NaN where pruned).
    """
    _init()
    E, P, _ = corners.shape
    assert P == int(config.max_num_points), "corners P must equal config.max_num_points"
    npseg = int(config.num_points_per_segment)
    # u = s/(npseg-1) below needs npseg >= 2 (one sample per segment is non-physical
    # and would divide by zero, unlike the oracle's linspace which tolerates it).
    assert npseg >= 2, "num_points_per_segment must be >= 2"
    dev = str(corners.device)

    # edgy -> vertex_tangents blend weight (default edgy=0 -> p=0.5).
    p = math.atan(config.edgy) / math.pi + 0.5

    # RAW corners + count fed straight to the kernels; the _prune_corners NaN step
    # (corner i -> NaN iff i >= count[e]) is folded in-kernel via _pruned_corner.
    cf = wp.from_torch(corners.reshape(E * P, 2).contiguous(), dtype=wp.vec2f)
    cnt = wp.from_torch(count.to(torch.int32).contiguous(), dtype=wp.int32)
    tan_t = torch.empty(E * P, 2, device=corners.device, dtype=torch.float32)
    wp_tan = wp.from_torch(tan_t, dtype=wp.vec2f)
    out_t = torch.empty(E * P * npseg, 2, device=corners.device, dtype=torch.float32)
    wp_out = wp.from_torch(out_t, dtype=wp.vec2f)

    wp.launch(_vertex_tangents_k, dim=E * P, inputs=[cf, cnt, P, float(p), wp_tan], device=dev)
    wp.launch(_assemble_k, dim=E * P * npseg,
              inputs=[cf, cnt, wp_tan, P, npseg, float(config.rad), wp_out], device=dev)
    _sync(corners.device)
    return out_t.view(E, P * npseg, 2)


def offset(center: torch.Tensor, Nrm: torch.Tensor, half_width: float):
    """Constant-width offset of a closed-loop centerline, matching inflation._offset_stage.

    Computes a = center + half_width*Nrm and b = center - half_width*Nrm per point,
    then assigns outer = whichever candidate has larger |shoelace area|, inner = the
    other.  Pure Warp (cpu+cuda); allclose to the torch oracle to atol=1e-5.

    Args:
        center:     [E, N, 2] float32 closed-loop points (finite; no NaN assumed).
        Nrm:        [E, N, 2] float32 unit left-normals.
        half_width: Python float, constant for all points/envs.

    Returns:
        outer: [E, N, 2], inner: [E, N, 2]
    """
    _init()
    E, N, _ = center.shape
    dev = str(center.device)
    flat = E * N

    cf = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    nf = wp.from_torch(Nrm.reshape(flat, 2).contiguous(), dtype=wp.vec2f)

    # Per-env accumulators (zero-initialised so atomic_add works correctly).
    area_a = torch.zeros(E, device=center.device, dtype=torch.float32)
    area_b = torch.zeros(E, device=center.device, dtype=torch.float32)
    out_t = torch.empty(flat, 2, device=center.device, dtype=torch.float32)
    inn_t = torch.empty(flat, 2, device=center.device, dtype=torch.float32)

    wa = wp.from_torch(area_a, dtype=wp.float32)
    wb = wp.from_torch(area_b, dtype=wp.float32)
    wo = wp.from_torch(out_t, dtype=wp.vec2f)
    wi = wp.from_torch(inn_t, dtype=wp.vec2f)

    wp.launch(_offset_build_k, dim=flat,
              inputs=[cf, nf, float(half_width), N, wa, wb],
              device=dev)
    wp.launch(_offset_assign_k, dim=flat,
              inputs=[cf, nf, float(half_width), N, wa, wb, wo, wi],
              device=dev)
    _sync(center.device)
    return out_t.view(E, N, 2), inn_t.view(E, N, 2)


def _smoke_double(x: torch.Tensor) -> torch.Tensor:
    """Smoke test: 2*x via a Warp kernel on x's device (cpu or cuda)."""
    _init()
    out = torch.empty_like(x)
    wp.launch(
        _double_k,
        dim=x.shape[0],
        inputs=[wp.from_torch(x.contiguous(), dtype=wp.float32),
                wp.from_torch(out, dtype=wp.float32)],
        device=str(x.device),
    )
    _sync(x.device)
    return out


def frame_curvature(center: torch.Tensor):
    """Per-point unit tangent, left-normal, and Menger curvature on a closed loop.
    center [E, N, 2] -> (T [E,N,2], Nrm [E,N,2], kappa [E,N]). Pure Warp (cpu+cuda)."""
    _init()
    E, N, _ = center.shape
    dev = str(center.device)
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    T = torch.empty(E * N, 2, device=center.device, dtype=torch.float32)
    Nrm = torch.empty_like(T)
    kap = torch.empty(E * N, device=center.device, dtype=torch.float32)
    wp.launch(_frame_k, dim=E * N,
              inputs=[cf, N, wp.from_torch(T, dtype=wp.vec2f),
                      wp.from_torch(Nrm, dtype=wp.vec2f), wp.from_torch(kap, dtype=wp.float32)],
              device=dev)
    _sync(center.device)
    return T.view(E, N, 2), Nrm.view(E, N, 2), kap.view(E, N)


def self_intersections(poly: torch.Tensor) -> torch.Tensor:
    """Count proper self-crossings of each closed polyline. poly [E, N, 2] -> [E] long.

    Matches geometry.self_intersections exactly (torch.equal). Pure Warp (cpu+cuda).
    One thread per env; O(N^2) loop over edge pairs inside the kernel.
    """
    _init()
    E, N, _ = poly.shape
    dev = str(poly.device)

    flat = poly.reshape(E * N, 2).contiguous()
    wp_poly = wp.from_torch(flat, dtype=wp.vec2f)

    out_t = torch.zeros(E, device=poly.device, dtype=torch.int32)
    wp.launch(_self_intersections_k, dim=E,
              inputs=[wp_poly, N, wp.from_torch(out_t, dtype=wp.int32)],
              device=dev)
    _sync(poly.device)
    return out_t.long()


def separation_min(points: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Min Euclidean distance over pairs with circ-index-dist > band. [E] float32.

    points [E, N, 2]; band [E] int. Returns +inf when no valid pair exists.
    Pure Warp (cpu+cuda). Matches geometry.separation_min to allclose(atol=1e-4).
    """
    _init()
    E, N, _ = points.shape
    dev = str(points.device)

    flat = points.reshape(E * N, 2).contiguous()
    wp_pts = wp.from_torch(flat, dtype=wp.vec2f)
    band_i32 = band.to(torch.int32).contiguous()
    wp_band = wp.from_torch(band_i32, dtype=wp.int32)

    out_t = torch.empty(E, device=points.device, dtype=torch.float32)
    wp.launch(_sep_min_k, dim=E,
              inputs=[wp_pts, wp_band, N, wp.from_torch(out_t, dtype=wp.float32)],
              device=dev)
    _sync(points.device)
    # Replace sentinel (no valid pair) with actual +inf to match torch oracle.
    out_t[out_t >= 1.0e29] = float("inf")
    return out_t


def curvature_radius_min(points: torch.Tensor) -> torch.Tensor:
    """1 / max Menger curvature over the loop. points [E, N, 2] -> [E] float32.

    Pure Warp (cpu+cuda). Matches geometry.curvature_radius_min to allclose(atol=1e-4).
    """
    _init()
    E, N, _ = points.shape
    dev = str(points.device)

    flat = points.reshape(E * N, 2).contiguous()
    wp_pts = wp.from_torch(flat, dtype=wp.vec2f)

    out_t = torch.empty(E, device=points.device, dtype=torch.float32)
    wp.launch(_curvrad_min_k, dim=E,
              inputs=[wp_pts, N, wp.from_torch(out_t, dtype=wp.float32)],
              device=dev)
    _sync(points.device)
    return out_t


def thickness(points: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Discrete curve thickness = min(curvature_radius_min, 0.5*separation_min). [E] float32.

    points [E, N, 2] float32; band [E] int (per-env exclusion window).
    Matches geometry.thickness to allclose(atol=1e-4). Pure Warp (cpu+cuda).
    """
    _init()
    E, N, _ = points.shape
    dev = str(points.device)

    flat = points.reshape(E * N, 2).contiguous()
    wp_pts = wp.from_torch(flat, dtype=wp.vec2f)
    band_i32 = band.to(torch.int32).contiguous()
    wp_band = wp.from_torch(band_i32, dtype=wp.int32)

    out_t = torch.empty(E, device=points.device, dtype=torch.float32)
    wp.launch(_thickness_k, dim=E,
              inputs=[wp_pts, wp_band, N, wp.from_torch(out_t, dtype=wp.float32)],
              device=dev)
    _sync(points.device)
    return out_t


def resample_uniform(center: torch.Tensor, n: int) -> torch.Tensor:
    """Arc-length-uniform resample of each closed loop to n points.

    Matches track_gen.relaxation._resample_uniform within FP tolerance (~1e-4; the
    Warp float32 sqrt vs torch's differs by rounding, geometrically negligible).
    Two Warp kernels: scan (one thread per env, builds seg+cumulative s from points)
    then lookup (one thread per output point, linear-scan searchsorted + lerp).
    center [E, N, 2] float32 -> [E, n, 2] float32.  n must equal N for now.
    Pure Warp (cpu+cuda), no torch compute.
    """
    _init()
    assert n == center.shape[1], "resample_uniform: n must equal N (input point count)"
    E, N, _ = center.shape
    dev = str(center.device)
    flat = E * N

    cf = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    seg_t = torch.empty(flat, device=center.device, dtype=torch.float32)
    s_t = torch.empty(E * (N + 1), device=center.device, dtype=torch.float32)
    out_t = torch.empty(flat, 2, device=center.device, dtype=torch.float32)
    wp_seg = wp.from_torch(seg_t, dtype=wp.float32)
    wp_s = wp.from_torch(s_t, dtype=wp.float32)
    wp_out = wp.from_torch(out_t, dtype=wp.vec2f)

    wp.launch(_resample_scan_k, dim=E, inputs=[cf, N, wp_seg, wp_s], device=dev)
    wp.launch(_resample_lookup_k, dim=flat, inputs=[cf, wp_seg, wp_s, N, wp_out], device=dev)
    _sync(center.device)
    return out_t.view(E, n, 2)


def resample_constant_spacing(center: torch.Tensor, spacing: float, n_max: int):
    """Arc-length resample each fully-real closed loop to constant `spacing`, padded to
    n_max with NaN. Returns (out [E, n_max, 2], count [E] long). Matches
    geometry.arc_length_resample(points, spacing=spacing, n_max=n_max). Pure Warp (cpu+cuda)."""
    _init()
    E, N, _ = center.shape
    dev = str(center.device)
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    seg = torch.empty(E * N, device=center.device, dtype=torch.float32)
    s = torch.empty(E * (N + 1), device=center.device, dtype=torch.float32)
    cnt = torch.empty(E, device=center.device, dtype=torch.int32)
    out = torch.empty(E * n_max, 2, device=center.device, dtype=torch.float32)
    wp.launch(_cs_scan_k, dim=E, inputs=[cf, N, float(spacing), n_max,
              wp.from_torch(seg, dtype=wp.float32), wp.from_torch(s, dtype=wp.float32),
              wp.from_torch(cnt, dtype=wp.int32)], device=dev)
    wp.launch(_cs_lookup_k, dim=E * n_max, inputs=[cf, wp.from_torch(seg, dtype=wp.float32),
              wp.from_torch(s, dtype=wp.float32), N, float(spacing), n_max,
              wp.from_torch(cnt, dtype=wp.int32), wp.from_torch(out, dtype=wp.vec2f)], device=dev)
    _sync(center.device)
    return out.view(E, n_max, 2), cnt.long()


def arc_length_resample_warp(points: torch.Tensor, num: int):
    """NaN-aware arc-length-uniform resample, a drop-in for geometry.arc_length_resample.

    FIXED mode (``num`` given, no spacing/valid_mask): per env, DROP every non-finite
    point (interior NaN too) compacting the rest IN ORDER (R real points), close that
    real loop, and emit exactly ``num`` arc-uniform points. Envs with R < 2 yield an
    all-NaN row and count 0; envs with R >= 2 get count ``num``. Generalizes
    resample_uniform (variable R per env, NaN dropping, output count != input count).

    Two Warp kernels mirroring _resample_scan_k / _resample_lookup_k: _arc_scan_k (one
    thread per env) compacts real points and builds the closed-loop float64-accumulated
    arc length; _arc_lookup_k (one thread per output point) does a linear-scan
    searchsorted + lerp. Pure Warp (cpu+cuda), no torch compute on the geometry.

    Args:
        points: [E, M, 2] float32 dense loops (may contain NaN padding / interior NaN).
        num: fixed output point count per env.

    Returns:
        (resampled [E, num, 2] float32, count [E] long). Matches the torch oracle:
        count EXACT; positions allclose to atol~5e-4 (float32-cumsum vs float64 drift).
    """
    _init()
    E, M, _ = points.shape
    dev = str(points.device)
    device = points.device

    df = wp.from_torch(points.reshape(E * M, 2).contiguous(), dtype=wp.vec2f)

    # Scratch: compacted real points, per-real-segment lengths, cumulative arc length,
    # the real-point count R per env (read by the lookup to gate NaN / clamp idx), and the
    # public per-env output count written directly by the scan kernel (R>=2 -> num, else 0).
    real_t = torch.empty(E * M, 2, device=device, dtype=torch.float32)
    seg_t = torch.empty(E * M, device=device, dtype=torch.float32)
    s_t = torch.empty(E * (M + 1), device=device, dtype=torch.float32)
    count_r_t = torch.empty(E, device=device, dtype=torch.int32)
    count_out_t = torch.empty(E, device=device, dtype=torch.int32)
    out_t = torch.empty(E * num, 2, device=device, dtype=torch.float32)

    wp_real = wp.from_torch(real_t, dtype=wp.vec2f)
    wp_seg = wp.from_torch(seg_t, dtype=wp.float32)
    wp_s = wp.from_torch(s_t, dtype=wp.float32)
    wp_count_r = wp.from_torch(count_r_t, dtype=wp.int32)
    wp_count_out = wp.from_torch(count_out_t, dtype=wp.int32)
    wp_out = wp.from_torch(out_t, dtype=wp.vec2f)

    wp.launch(_arc_scan_k, dim=E,
              inputs=[df, M, num, wp_real, wp_seg, wp_s, wp_count_r, wp_count_out], device=dev)
    wp.launch(_arc_lookup_k, dim=E * num,
              inputs=[wp_real, wp_seg, wp_s, wp_count_r, M, num, wp_out], device=dev)
    _sync(points.device)

    # Public count written in-kernel (R >= 2 -> num, R < 2 -> 0); .long() is an I/O dtype view.
    return out_t.view(E, num, 2), count_out_t.long()


def turning_number(center: torch.Tensor) -> torch.Tensor:
    """Signed total turning of each closed polygon, in radians. center [E, N, 2] -> [E] float32.

    +/-2*pi for a simple loop (sign = orientation); ~0 for a figure-eight whose lobes
    wind in opposite directions. Matches geometry.turning_number to allclose(atol=1e-4).
    Pure Warp (cpu+cuda). One thread per env; O(N) loop over edge-angle deltas.
    """
    _init()
    E, N, _ = center.shape
    dev = str(center.device)

    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    out_t = torch.empty(E, device=center.device, dtype=torch.float32)
    wp.launch(_turning_k, dim=E,
              inputs=[cf, N, wp.from_torch(out_t, dtype=wp.float32)],
              device=dev)
    _sync(center.device)
    return out_t


def gates(corners: torch.Tensor, dense: torch.Tensor, count: torch.Tensor,
          config) -> torch.Tensor:
    """Per-env accept mask, a drop-in for the gate conjunction in generate().

    Reproduces ``ok = angle_ok & turn_ok & finite_ok & simple_ok`` from
    BezierCenterlineGenerator.generate entirely in Warp kernels (the wrapper does ZERO
    torch compute -- only wp.from_torch I/O + dtype views + launches):

      1. ANGLE: _corner_angles_gate_k (one thread per env) takes the RAW corners + count
         and folds in the _prune_corners NaN step (corner i real iff i < count[e] AND
         both components finite), then checks that every CONSTRAINED corner (it and both
         circular neighbours real) has interior angle > min_angle. Unconstrained corners
         are skipped, matching the oracle's nan_to_num(angle, 0) | ~constrained.
      2. TURN + FINITE inputs: arc_length_resample_warp(dense, num_points_per_segment) ->
         turning_number (turn) plus the resample count (cnt_turn).
      3. SIMPLE input: arc_length_resample_warp(dense, num_points) -> self_intersections
         (cross).
      4. COMBINE: _gates_combine_k (one thread per env) computes turn_ok / finite_ok /
         simple_ok from turn/cnt_turn/cross and ANDs them with angle_ok into the result.

    The TURN resample is to num_points_per_segment (30) and the SIMPLE resample is to
    num_points (256) -- different sizes, exactly as the oracle. Equals the oracle's accept
    mask (torch.equal) for cases built clear of the thresholds; the ~5e-4 resample drift is
    geometrically negligible. Pure Warp (cpu+cuda); no oracle-module imports.

    Args:
        corners: [E, P, 2] float32 RAW ccw-sorted corners (pre-prune; P == count's domain).
        dense:   [E, M, 2] float32 assembled dense centerline (already NaN where pruned).
        count:   [E] int real-corner count per env (rows >= count are pruned to NaN).
        config:  TrackGenConfig (uses min_angle, turning_tol, num_points,
                 num_points_per_segment).

    Returns:
        [E] bool accept mask.
    """
    _init()
    E, P, _ = corners.shape
    dev = str(corners.device)
    device = corners.device

    # --- ANGLE gate (Warp kernel over RAW corners + count; prune folded in-kernel) ---
    cf = wp.from_torch(corners.reshape(E * P, 2).contiguous(), dtype=wp.vec2f)
    cnt = wp.from_torch(count.to(torch.int32).contiguous(), dtype=wp.int32)
    angle_ok_t = torch.empty(E, device=device, dtype=torch.int32)
    wp.launch(_corner_angles_gate_k, dim=E,
              inputs=[cf, cnt, P, float(config.min_angle),
                      wp.from_torch(angle_ok_t, dtype=wp.int32)],
              device=dev)

    # --- TURN + FINITE inputs (resample to num_points_per_segment, then turning number) ---
    rs_turn, cnt_turn = arc_length_resample_warp(dense, int(config.num_points_per_segment))
    turn = turning_number(rs_turn)

    # --- SIMPLE input (resample to num_points, then count self-crossings) ---
    rs_simple, _ = arc_length_resample_warp(dense, int(config.num_points))
    cross = self_intersections(rs_simple)

    # --- combine all gates in one kernel (no torch where/&/comparisons) ---
    out_t = torch.empty(E, device=device, dtype=torch.int32)
    wp.launch(
        _gates_combine_k, dim=E,
        inputs=[wp.from_torch(angle_ok_t, dtype=wp.int32),
                wp.from_torch(turn.contiguous(), dtype=wp.float32),
                wp.from_torch(cnt_turn.to(torch.int32).contiguous(), dtype=wp.int32),
                wp.from_torch(cross.to(torch.int32).contiguous(), dtype=wp.int32),
                float(config.turning_tol),
                wp.from_torch(out_t, dtype=wp.int32)],
        device=dev,
    )
    _sync(device)
    return out_t.bool()


def validity(center: torch.Tensor, w: torch.Tensor, count: torch.Tensor,
             gen_valid: torch.Tensor, config, outer: torch.Tensor | None = None,
             inner: torch.Tensor | None = None) -> torch.Tensor:
    """Per-track validity gate, a drop-in replacement for inflation._validity_stage.

    Combines: generation flag AND closed-loop turning AND width floor AND no-NaN AND
    thickness >= (1-relax_tol)*half_width AND zero border self-intersections. The heavy
    geometry runs through the verified Warp wrappers (turning_number, thickness,
    self_intersections); the real-point mask, width floor, NaN, band derivation and the
    boolean combine are light torch glue at the boundary, matching the established wrapper
    pattern. Equals inflation._validity_stage exactly (torch.equal on the bool output).

    Args:
        center:    [E, N, 2] float32 resampled centerline.
        w:         [E, N]    half-width per point.
        count:     [E]       int real-point count per env (fixed mode -> N).
        gen_valid: [E]       bool generation flag.
        config:    TrackGenConfig (uses turning_tol, w_floor, half_width, relax_tol).
        outer:     [E, N, 2] outer border polygon, or None to skip the border check.
        inner:     [E, N, 2] inner border polygon, or None to skip the border check.

    Returns:
        [E] bool validity. When either border is None the border check is skipped
        (border_ok all True), matching the oracle's outer/inner defaults.

    Pure Warp: a single ``_validity_k`` launch (one thread per env) does ALL the per-env
    logic (turning, real-mask width floor + no-NaN, in-kernel band derivation, thickness,
    and border self-intersection combine). The wrapper does ZERO torch compute -- only
    wp.from_torch I/O wrapping, dtype conversions for I/O (.to(int32) on inputs, .bool()
    on the output), and the dummy-array handling for the None border case.
    """
    _init()
    E, N = w.shape
    device = w.device
    dev = str(device)
    flat = E * N

    cf = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    wf = wp.from_torch(w.reshape(flat).contiguous().to(torch.float32), dtype=wp.float32)
    cnt = wp.from_torch(count.to(torch.int32).contiguous(), dtype=wp.int32)
    gv = wp.from_torch(gen_valid.to(torch.int32).contiguous(), dtype=wp.int32)

    if outer is None or inner is None:
        # No border: pass center as a dummy non-empty vec2f array and disable the check.
        has_border = 0
        ob = cf
        ib = cf
    else:
        # The kernel NaN->0 guards outer/inner internally (== oracle nan_to_num(nan=0.0)).
        has_border = 1
        ob = wp.from_torch(outer.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
        ib = wp.from_torch(inner.reshape(flat, 2).contiguous(), dtype=wp.vec2f)

    out_t = torch.empty(E, device=device, dtype=torch.int32)
    wp.launch(
        _validity_k, dim=E,
        inputs=[cf, wf, cnt, gv, ob, ib, int(has_border), N,
                float(config.half_width), float(config.turning_tol),
                float(config.w_floor), float(config.relax_tol),
                wp.from_torch(out_t, dtype=wp.int32)],
        device=dev,
    )
    _sync(device)
    return out_t.bool()


def _arclength(center: torch.Tensor):
    """Cumulative arc length [E, N] (0 at index 0) and closed-loop total length [E].

    FIXED-mode only (all points real, the wrap segment closes the loop). Reproduces
    inflation._arclength under that assumption via a single Warp kernel: one thread per
    env, float64-accumulated running sum. arclen[i] is the length before segment i; the
    total length includes the wrap segment (last point -> point 0). Pure Warp (cpu+cuda);
    allclose to the torch oracle to atol~1e-3 (float32-cumsum vs float64 drift over N segs).

    Args:
        center: [E, N, 2] float32 closed-loop points (finite; no NaN assumed).

    Returns:
        arclen: [E, N] float32 cumulative arc length, length: [E] float32 perimeter.
    """
    _init()
    E, N, _ = center.shape
    dev = str(center.device)

    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    arclen_t = torch.empty(E * N, device=center.device, dtype=torch.float32)
    length_t = torch.empty(E, device=center.device, dtype=torch.float32)
    wp.launch(_arclength_k, dim=E,
              inputs=[cf, N, wp.from_torch(arclen_t, dtype=wp.float32),
                      wp.from_torch(length_t, dtype=wp.float32)],
              device=dev)
    _sync(center.device)
    return arclen_t.view(E, N), length_t


def corner_sample(seeds: torch.Tensor, attempt: int, config) -> torch.Tensor:
    """Sample max_num_points grid-corner points per env with Warp's built-in RNG.

    ACCEPTED RNG REDESIGN: this is the pure-Warp replacement for the torch oracle
    generators.BezierCenterlineGenerator._sample_corner_points. It does NOT reproduce
    the oracle bit-for-bit (the legacy PerEnvSeededRNG is retired from the pure-Warp
    path); it is validated by structural properties only. The construction matches the
    oracle's GEOMETRY, though: each corner picks a grid cell in [0, num_cells**2),
    derives cell coords (x = cell % num_cells, y = cell // num_cells), adds per-corner
    noise in [-0.5, 0.5), and scales by (cell_size, scale). Distinct cells are preferred
    via bounded duplicate rejection, echoing the oracle's no-replacement top-k subset.

    Seeding/draw order (see _corner_sample_k): state = rand_init(seed*9781 + attempt);
    per corner, one cell-selection randf per duplicate-rejection try, then two noise
    randfs (nx, ny). Reproducible per (seeds, attempt, config).

    Args:
        seeds:   [E] int per-env base seed. Interpreted mod 2**32 (narrowed to int32
                 before the seed mix), so very large seeds may alias.
        attempt: int retry counter (mixed into the seed for attempt-to-attempt variety).
        config:  TrackGenConfig (uses max_num_points, min_point_distance, scale).

    Returns:
        [E, max_num_points, 2] float32 corner points in scaled grid coordinates.
        Pure Warp (cpu+cuda).
    """
    return _corner_sample_raw(seeds, attempt, config)[0]


def _corner_sample_raw(seeds: torch.Tensor, attempt: int, config):
    """corner_sample internals, also returning the chosen grid cells for inspection/tests.

    Returns:
        (corners [E, max_num_points, 2] float32, cells [E, max_num_points] int32).
        ``cells[e, c]`` is the grid-cell index in [0, num_cells**2) chosen for corner c
        of env e (after duplicate rejection) — the only directly-observable record of the
        cell-selection RNG, since the additive per-corner noise makes cells unrecoverable
        from the scaled positions.
    """
    _init()
    E = seeds.shape[0]
    dev = str(seeds.device)
    P = int(config.max_num_points)
    num_cells = int(1.0 / (config.min_point_distance * 2))
    nc2 = num_cells * num_cells
    cell_size = config.min_point_distance * 2.0

    seeds_i32 = seeds.to(torch.int32).contiguous()
    # Scratch dedup buffer doubling as the output cell record:
    # used[e*P + c] = cell chosen for corner c of env e (-1 = unset).
    used_t = torch.full((E * P,), -1, device=seeds.device, dtype=torch.int32)
    out_t = torch.empty(E * P, 2, device=seeds.device, dtype=torch.float32)

    wp.launch(
        _corner_sample_k, dim=E,
        inputs=[wp.from_torch(seeds_i32, dtype=wp.int32), int(attempt),
                num_cells, nc2, float(cell_size), float(config.scale), P,
                wp.from_torch(used_t, dtype=wp.int32),
                wp.from_torch(out_t, dtype=wp.vec2f)],
        device=dev,
    )
    _sync(seeds.device)
    return out_t.view(E, P, 2), used_t.view(E, P)


def ccw_sort(points: torch.Tensor) -> torch.Tensor:
    """Order each env's points angularly around their centroid. [E, P, 2] -> [E, P, 2].

    Matches geometry.ccw_sort for DISTINCT angle keys (torch.equal): the output is the
    input points permuted along P (no value arithmetic), sorted ascending by the
    centroid-relative angle key atan2(dx, dy) (the X-component FIRST argument is an
    intentional quirk of the original generator, preserved here). The sort is stable
    (strict ``>``); torch.argsort is non-stable, so on exactly-equal fp32 keys (a
    measure-zero tie) the two may order the tied points differently. Pure Warp
    (cpu+cuda); one thread per env, in-place insertion sort.
    """
    _init()
    E, P, _ = points.shape
    dev = str(points.device)
    flat = E * P

    pf = wp.from_torch(points.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    # keys_t / out_t are intentionally uninitialised: the insertion sort only ever reads
    # slots strictly behind its write frontier (index < c), so no garbage is consumed.
    keys_t = torch.empty(flat, device=points.device, dtype=torch.float32)
    out_t = torch.empty(flat, 2, device=points.device, dtype=torch.float32)
    wp.launch(_ccw_sort_k, dim=E,
              inputs=[pf, P, wp.from_torch(keys_t, dtype=wp.float32),
                      wp.from_torch(out_t, dtype=wp.vec2f)],
              device=dev)
    _sync(points.device)
    return out_t.view(E, P, 2)


def inflate_warp(center: torch.Tensor, config, valid: torch.Tensor | None = None):
    """Pure-Warp drop-in for inflation.inflate on a CLEAN, fixed-N centerline.

    Composes the verified Warp wrappers in the same order as inflation.inflate:
    resample -> frame+curvature -> constant width -> offset -> validity -> arclength ->
    assemble Track. Requires no NaN, ``center.shape[1] == config.num_points`` and
    ``config.output_mode == "fixed"`` (so count == N everywhere and the masked-resample /
    NaN-padding branches of the oracle collapse to the simple closed-loop case).

    Args:
        center: [E, N, 2] float32 centerline, N == config.num_points, no NaN.
        config: TrackGenConfig (output_mode must be "fixed").
        valid:  [E] bool generation flag; defaults to all-True.

    Returns:
        track_gen.types.Track with center/outer/inner/tangent/normal/arclen/length/
        valid/count. Equals inflation.inflate within FP tolerance (positions/frame
        ~1e-4, arclen/length ~1e-3; valid/count exact).
    """
    from .types import Track  # local import: keep warp_pipeline free of oracle modules

    assert config.output_mode == "fixed", "inflate_warp supports output_mode='fixed' only"
    assert center.shape[1] == config.num_points, "center N must equal config.num_points"
    _init()
    E, N, _ = center.shape
    dev = str(center.device)
    hw = float(config.half_width)

    # 1. arc-length-uniform resample; fixed mode -> every env keeps all N points.
    rs = resample_uniform(center, config.num_points)

    # Constant per-track count (== N) and per-point half-width (== hw), plus the default
    # all-valid generation flag, ALL kernel-filled (pure-Warp torch.full/ones replacements).
    # torch.empty is just an I/O alloc; validity/offset/the Track consume these as before.
    count_i32 = torch.empty(E, device=center.device, dtype=torch.int32)
    w = torch.empty(E * N, device=center.device, dtype=torch.float32)
    wp.launch(_fill_i32_k, dim=E, inputs=[wp.from_torch(count_i32, dtype=wp.int32), N], device=dev)
    wp.launch(_fill_f32_k, dim=E * N, inputs=[wp.from_torch(w, dtype=wp.float32), hw], device=dev)
    count = count_i32.long()                       # Track stores count as long (I/O dtype view)
    w = w.view(E, N)                               # validity indexes w as [E, N]

    # 2. frame + curvature (kappa unused, like inflation._width_stage).
    T, Nrm, _kappa = frame_curvature(rs)

    # 4. offset to outer/inner borders.
    outer, inner = offset(rs, Nrm, hw)

    # 5. per-track validity gate. Default gen flag (all valid) is a kernel int32 fill;
    # otherwise the passed bool `valid` is consumed directly (validity .to(int32) on it).
    if valid is not None:
        gen_valid = valid
    else:
        gen_valid_i32 = torch.empty(E, device=center.device, dtype=torch.int32)
        wp.launch(_fill_i32_k, dim=E, inputs=[wp.from_torch(gen_valid_i32, dtype=wp.int32), 1], device=dev)
        gen_valid = gen_valid_i32                  # validity .to(int32) is a no-op view here
    valid_out = validity(rs, w, count, gen_valid, config, outer, inner)

    # 6. cumulative arc length + total length.
    arclen, length = _arclength(rs)

    return Track(outer=outer, center=rs, inner=inner, tangent=T, normal=Nrm,
                 arclen=arclen, length=length, valid=valid_out, count=count)


class CapturedTracks:
    """A captured CUDA graph of the whole ``generate_tracks_warp`` pipeline + its replay.

    Built by :func:`generate_tracks_warp_graph`. Holds the static seed input buffer, the
    static Track output buffers (the device addresses baked into the graph), and the
    captured :class:`torch.cuda.CUDAGraph`. :meth:`replay` copies new seeds into the seed
    buffer, replays the graph (re-running EVERY stage on the GPU with the new seed contents),
    and returns a Track viewing the output buffers (cloned so successive replays don't alias).
    """

    def __init__(self, graph, seeds_buf: torch.Tensor, track, config):
        """Store the captured graph plus its static seed-input and Track-output buffers."""
        self._graph = graph
        self._seeds_buf = seeds_buf      # static [E] input buffer; copy_ new seeds before replay
        self._track = track              # Track whose tensors are the static graph outputs
        self._config = config

    def replay(self, new_seeds: torch.Tensor):
        """Replay the captured pipeline with ``new_seeds`` -> a fresh Track (cloned outputs)."""
        from .types import Track  # local import: keep warp_pipeline free of oracle modules

        if new_seeds.shape != self._seeds_buf.shape:
            raise ValueError(
                f"replay seeds shape {tuple(new_seeds.shape)} != captured "
                f"{tuple(self._seeds_buf.shape)}"
            )
        # Push the new per-env seeds into the static buffer the graph reads, then replay.
        self._seeds_buf.copy_(new_seeds.to(self._seeds_buf.dtype))
        self._graph.replay()
        # Clone the static outputs so the returned Track is stable across the next replay.
        t = self._track
        return Track(
            outer=t.outer.clone(), center=t.center.clone(), inner=t.inner.clone(),
            tangent=t.tangent.clone(), normal=t.normal.clone(), arclen=t.arclen.clone(),
            length=t.length.clone(), valid=t.valid.clone(), count=t.count.clone(),
        )


def generate_tracks_warp_graph(config, seeds_template: torch.Tensor) -> CapturedTracks:
    """Capture the ENTIRE ``generate_tracks_warp`` pipeline as ONE CUDA graph.

    The whole eager pipeline -- generation (fixed-iter regen loop), the band/L0 torch glue,
    the pure-Warp XPBD relax loop, the final resample, and inflate (with its torch validity
    glue) -- is captured into a single :class:`torch.cuda.CUDAGraph` and returned wrapped in a
    :class:`CapturedTracks` whose ``.replay(new_seeds)`` re-runs the graph with new seeds.

    Mechanism (Warp 1.14 + torch 2.6):
      * ``torch.cuda.graph`` is stream-level: it captures ALL CUDA work submitted to its
        internal capture stream. torch ops land there automatically; to make Warp's kernel
        launches land there too we wrap that stream as a ``wp.Stream(device, cuda_stream=...)``
        and enter ``wp.ScopedStream`` so ``wp.launch`` submits to the SAME capturing stream.
        Result: torch glue + Warp kernels are unified in one native graph.
      * Host-blocking syncs are illegal during capture. The module global ``_CAPTURING`` is
        set for the warmup + capture region so every wrapper's ``_sync`` and
        ``warp_relax.xpbd_solve``'s ``wp.synchronize`` become no-ops (the graph records the
        stream ordering, so they are unnecessary anyway).
      * ``torch.cuda.graph`` requires warmup on a side stream before capture; we run the
        pipeline a few times there (also under ``_CAPTURING`` + the routed Warp stream) so
        all kernels/modules are loaded and torch's graph memory pool is primed.
      * Static buffers: the [E] ``seeds_buf`` is allocated once; ``replay`` copies new seeds
        into it. The fresh per-call ``torch.empty`` scratch inside the wrappers is served
        from torch's private graph memory pool and its addresses are baked into the graph,
        so replay reuses them. The captured Track tensors are the graph's static outputs.

    The Warp RNG is deterministic in the seed buffer contents, so replay with the SAME seeds
    reproduces the eager result and replay with NEW seeds equals ``generate_tracks_warp(config,
    new_seeds)`` (positions allclose ~1e-4; valid/count exact).

    Args:
        config:         TrackGenConfig (same constraints as generate_tracks_warp: output_mode
                        "fixed", relax_solver "xpbd", smooth_finish False).
        seeds_template: [E] int CUDA tensor; its SHAPE/DTYPE/DEVICE fix the captured batch.
                        Its values seed the warmup but are otherwise irrelevant (replay
                        overwrites the seed buffer).

    Returns:
        CapturedTracks. CUDA only (graph capture needs a GPU).
    """
    global _CAPTURING
    assert seeds_template.is_cuda, "generate_tracks_warp_graph requires CUDA seeds"
    _init()

    dev = str(seeds_template.device)
    wp_dev = wp.get_device(dev)

    # Static input buffer the graph reads; replay copies new seeds into it.
    seeds_buf = torch.empty_like(seeds_template)
    seeds_buf.copy_(seeds_template)

    def _run():
        # Sync-free pipeline (guaranteed by _CAPTURING being set around every call site).
        return generate_tracks_warp(config, seeds_buf)

    # ---- Warmup on a side stream (torch.cuda.graph requirement) ----
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    _CAPTURING = True
    try:
        with torch.cuda.stream(side):
            wp_side = wp.Stream(wp_dev, cuda_stream=side.cuda_stream)
            with wp.ScopedStream(wp_side, sync_enter=False, sync_exit=False):
                for _ in range(3):
                    _run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()  # OUTSIDE capture: fine, ensures warmup finished

        # ---- Capture: route Warp onto torch's internal capture stream ----
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cap_stream = torch.cuda.current_stream()
            wp_cap = wp.Stream(wp_dev, cuda_stream=cap_stream.cuda_stream)
            with wp.ScopedStream(wp_cap, sync_enter=False, sync_exit=False):
                track = _run()
    finally:
        _CAPTURING = False

    return CapturedTracks(graph, seeds_buf, track, config)
