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
