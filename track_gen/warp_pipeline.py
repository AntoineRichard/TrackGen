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
