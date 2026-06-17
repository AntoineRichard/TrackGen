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
