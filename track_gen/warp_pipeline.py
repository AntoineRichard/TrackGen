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

try:
    import warp as wp
    _HAVE_WARP = True
except Exception:  # warp is an optional extra
    _HAVE_WARP = False

_INITED = False


def _init() -> None:
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
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

    @wp.kernel
    def _self_intersections_k(
        poly: wp.array(dtype=wp.vec2f),
        N: int,
        out: wp.array(dtype=wp.int32),
    ):
        # One thread per env e. Loops all unique pairs (i,j) with j > i,
        # skips if circular index distance <= 1, counts proper crossings.
        e = wp.tid()
        count = int(0)
        for i in range(N):
            for j in range(i + 1, N):
                diff = j - i
                circ_dist = wp.min(diff, N - diff)
                if circ_dist <= 1:
                    continue
                Ai = poly[e * N + i]
                Bi = poly[e * N + (i + 1) % N]
                Aj = poly[e * N + j]
                Bj = poly[e * N + (j + 1) % N]
                d1 = _ccw(Aj[0], Aj[1], Bj[0], Bj[1], Ai[0], Ai[1])
                d2 = _ccw(Aj[0], Aj[1], Bj[0], Bj[1], Bi[0], Bi[1])
                d3 = _ccw(Ai[0], Ai[1], Bi[0], Bi[1], Aj[0], Aj[1])
                d4 = _ccw(Ai[0], Ai[1], Bi[0], Bi[1], Bj[0], Bj[1])
                seg_ij = (d1 > 0.0) != (d2 > 0.0)
                seg_ji = (d3 > 0.0) != (d4 > 0.0)
                if seg_ij and seg_ji:
                    count = count + 1
        out[e] = count

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

    @wp.kernel
    def _thickness_k(
        pts: wp.array(dtype=wp.vec2f),
        band: wp.array(dtype=wp.int32),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. thickness = min(rad_min, 0.5 * sep_min).
        e = wp.tid()
        b = band[e]

        # --- sep_min: min dist over pairs with circ_dist > band ---
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

        # --- rad_min: 1 / max Menger curvature ---
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
        rad_min = 1.0 / wp.max(kappa_max, float(1.0e-12))

        out[e] = wp.min(rad_min, 0.5 * sep_min)

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
    def _arc_scan_k(
        dense: wp.array(dtype=wp.vec2f),
        M: int,
        real_pts: wp.array(dtype=wp.vec2f),
        seg: wp.array(dtype=wp.float32),
        s: wp.array(dtype=wp.float32),
        count_r: wp.array(dtype=wp.int32),
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

    @wp.kernel
    def _turning_k(
        c: wp.array(dtype=wp.vec2f),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e. Signed total turning of the closed polygon.
        # Edge angle theta_i = atan2(d_i.y, d_i.x) for raw edge d_i = c[(i+1)%N] - c[i]
        # (atan2 is scale-invariant, so no normalization is needed; the zero-length
        # edge gives atan2(0,0)=0, matching the torch safe_normalize-then-atan2 case).
        # Per edge, dtheta = theta_i - theta_{i-1} wrapped into (-pi, pi] via
        # atan2(sin, cos); the sum over all edges is the turning number.
        e = wp.tid()
        b = e * N
        total = float(0.0)
        for i in range(N):
            di = c[b + (i + 1) % N] - c[b + i]
            ip = (i + N - 1) % N
            dp = c[b + (ip + 1) % N] - c[b + ip]
            theta_i = wp.atan2(di[1], di[0])
            theta_prev = wp.atan2(dp[1], dp[0])
            dth = theta_i - theta_prev
            total = total + wp.atan2(wp.sin(dth), wp.cos(dth))
        out[e] = total

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

    @wp.kernel
    def _vertex_tangents_k(
        c: wp.array(dtype=wp.vec2f),
        P: int,
        p: float,
        tangents: wp.array(dtype=wp.vec2f),
    ):
        # One thread per corner t (dim = E*P); e = env index, i = corner-within-env.
        # Mirrors geometry.vertex_tangents: u_out_i = dir(i -> i+1), u_in_i = dir(i-1 -> i)
        # (== roll(u_out, +1)); tangent_i = safe_normalize(p*u_out + (1-p)*u_in). NaN at any
        # pruned (NaN) corner propagates the same way the torch oracle's safe_normalize does.
        t = wp.tid()
        e = t // P
        i = t % P
        b = e * P
        c_i = c[t]
        c_next = c[b + (i + 1) % P]
        c_prev = c[b + (i + P - 1) % P]
        u_out = _safe_normalize2(c_next - c_i)
        u_in = _safe_normalize2(c_i - c_prev)
        blended = p * u_out + (1.0 - p) * u_in
        tangents[t] = _safe_normalize2(blended)

    @wp.kernel
    def _assemble_k(
        c: wp.array(dtype=wp.vec2f),
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
        # corner tangents. NaN corners/tangents propagate into the output as in the oracle.
        t = wp.tid()
        per_env = P * npseg
        e = t // per_env
        rem = t % per_env
        i = rem // npseg
        s = rem % npseg
        b = e * P

        c0 = c[b + i]
        c1 = c[b + (i + 1) % P]
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

    # Prune mask (folds in _prune_corners' NaN step): corner i is real iff i < count[e].
    row = torch.arange(P, device=corners.device)
    keep = (row < count.to(corners.device)[:, None]).unsqueeze(-1)        # [E, P, 1]
    pruned = torch.where(keep, corners, torch.full_like(corners, float("nan")))

    # edgy -> vertex_tangents blend weight (default edgy=0 -> p=0.5).
    p = math.atan(config.edgy) / math.pi + 0.5

    cf = wp.from_torch(pruned.reshape(E * P, 2).contiguous(), dtype=wp.vec2f)
    tan_t = torch.empty(E * P, 2, device=corners.device, dtype=torch.float32)
    wp_tan = wp.from_torch(tan_t, dtype=wp.vec2f)
    out_t = torch.empty(E * P * npseg, 2, device=corners.device, dtype=torch.float32)
    wp_out = wp.from_torch(out_t, dtype=wp.vec2f)

    wp.launch(_vertex_tangents_k, dim=E * P, inputs=[cf, P, float(p), wp_tan], device=dev)
    wp.launch(_assemble_k, dim=E * P * npseg,
              inputs=[cf, wp_tan, P, npseg, float(config.rad), wp_out], device=dev)
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
    # and the real-point count R per env (read by the lookup to gate NaN / clamp idx).
    real_t = torch.empty(E * M, 2, device=device, dtype=torch.float32)
    seg_t = torch.empty(E * M, device=device, dtype=torch.float32)
    s_t = torch.empty(E * (M + 1), device=device, dtype=torch.float32)
    count_r_t = torch.empty(E, device=device, dtype=torch.int32)
    out_t = torch.empty(E * num, 2, device=device, dtype=torch.float32)

    wp_real = wp.from_torch(real_t, dtype=wp.vec2f)
    wp_seg = wp.from_torch(seg_t, dtype=wp.float32)
    wp_s = wp.from_torch(s_t, dtype=wp.float32)
    wp_count_r = wp.from_torch(count_r_t, dtype=wp.int32)
    wp_out = wp.from_torch(out_t, dtype=wp.vec2f)

    wp.launch(_arc_scan_k, dim=E, inputs=[df, M, wp_real, wp_seg, wp_s, wp_count_r], device=dev)
    wp.launch(_arc_lookup_k, dim=E * num,
              inputs=[wp_real, wp_seg, wp_s, wp_count_r, M, num, wp_out], device=dev)
    _sync(points.device)

    # Public count matches the oracle: R >= 2 -> num, R < 2 -> 0.
    count = torch.where(count_r_t >= 2, num, 0).long()
    return out_t.view(E, num, 2), count


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
    """
    E, N = w.shape
    device = w.device

    # Real-point mask: slot j is real iff j < count[env] (fixed mode -> all True).
    idx = torch.arange(N, device=device).unsqueeze(0)        # [1, N]
    real = idx < count.unsqueeze(1)                          # [E, N]

    turning = turning_number(center)
    turn_ok = (turning.abs() - 2.0 * math.pi).abs() <= float(config.turning_tol)

    w_ok = torch.where(real, w > float(config.w_floor), torch.ones_like(real)).all(dim=1)

    nan_per_point = torch.isnan(center).any(dim=-1)
    no_nan = ~(nan_per_point & real).any(dim=1)

    # band = round(D / mean_seg_len).long().clamp_min(1); mean_seg_len = perimeter / N.
    D = 2.0 * float(config.half_width)
    L0 = _mean_seg_len_torch(center).clamp_min(1e-9)
    band = (D / L0).round().long().clamp_min(1)              # [E]
    th = thickness(center, band)
    th_ok = th >= (1.0 - float(config.relax_tol)) * float(config.half_width)

    if outer is None or inner is None:
        border_ok = torch.ones(E, dtype=torch.bool, device=device)
    else:
        crossings = self_intersections(torch.nan_to_num(outer, nan=0.0)) + \
                    self_intersections(torch.nan_to_num(inner, nan=0.0))
        border_ok = crossings == 0

    return gen_valid.to(torch.bool) & turn_ok & w_ok & no_nan & th_ok & border_ok


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
    E, N, _ = center.shape
    hw = float(config.half_width)

    # 1. arc-length-uniform resample; fixed mode -> every env keeps all N points.
    rs = resample_uniform(center, config.num_points)
    count = torch.full((E,), N, dtype=torch.long, device=center.device)

    # 2. frame + curvature (kappa unused, like inflation._width_stage).
    T, Nrm, _kappa = frame_curvature(rs)

    # 3. constant half-width per point.
    w = torch.full((E, N), hw, device=center.device, dtype=rs.dtype)

    # 4. offset to outer/inner borders.
    outer, inner = offset(rs, Nrm, hw)

    # 5. per-track validity gate.
    gen_valid = valid if valid is not None else torch.ones(E, dtype=torch.bool, device=center.device)
    valid_out = validity(rs, w, count, gen_valid, config, outer, inner)

    # 6. cumulative arc length + total length.
    arclen, length = _arclength(rs)

    return Track(outer=outer, center=rs, inner=inner, tangent=T, normal=Nrm,
                 arclen=arclen, length=length, valid=valid_out, count=count)
