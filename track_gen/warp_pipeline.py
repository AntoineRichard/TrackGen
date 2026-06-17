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
