# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional NVIDIA Warp acceleration for the XPBD separation constraint.

The torch separation builds the full ``[E, N, N, 2]`` pairwise tensor every sweep
(~GB-scale, the measured ~100% hot-spot of the GPU solve). This module computes the
same per-bead displacement in a single FUSED Warp kernel — each bead loops its
neighbours and accumulates the push with NO ``[E, N, N]`` materialization — which is
~2-3 orders of magnitude faster on CUDA while staying numerically equivalent.

Import is always safe: if Warp is absent (the optional extra) the module simply
reports unavailable and the caller falls back to the pure-torch separation. Warp is
used only on CUDA tensors; CPU stays pure torch (and CPU-testable).
"""
from __future__ import annotations

import torch

try:
    import warp as wp
    _HAVE_WARP = True
except Exception:  # warp is an optional extra
    _HAVE_WARP = False

_INITED = False


if _HAVE_WARP:

    @wp.kernel
    def _sep_kernel(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32),
                    N: int, target: wp.float32, out: wp.array(dtype=wp.vec2f)):
        # One thread per bead (flat index over E*N). e = env, i = bead within env.
        t = wp.tid()
        e = t // N
        i = t % N
        xi = center[t]
        disp = wp.vec2f(0.0, 0.0)
        cnt = int(0)
        base = e * N
        for j in range(N):
            d = wp.abs(i - j)
            circ = wp.min(d, N - d)               # circular index distance
            if circ > band[e]:                    # non-adjacent pair only
                diff = xi - center[base + j]
                dist = wp.max(wp.length(diff), 1.0e-9)
                pen = target - dist
                if pen > 0.0:                     # closer than D*(1+margin) -> push apart
                    disp = disp + (0.5 * pen / dist) * diff
                    cnt += 1
        if cnt > 0:
            out[t] = disp / wp.float32(cnt)       # Jacobi average by violated-pair count
        else:
            out[t] = wp.vec2f(0.0, 0.0)


def warp_available(device) -> bool:
    """True iff Warp is importable and the tensors live on CUDA."""
    return _HAVE_WARP and "cuda" in str(device)


def should_use(device, config) -> bool:
    """Resolve config.relax_use_warp: None -> auto (Warp on CUDA), else the explicit bool."""
    flag = getattr(config, "relax_use_warp", None)
    if flag is None:
        return warp_available(device)
    return bool(flag) and warp_available(device)


def separation_disp(center: torch.Tensor, band: torch.Tensor, target: float) -> torch.Tensor:
    """Fused Warp separation. Numerically matches relaxation._separation_disp's result.

    Args:
        center: [E, N, 2] float32 CUDA tensor.
        band:   [E] integer tensor (excluded-neighbour index half-window).
        target: D * (1 + margin) separation distance.
    Returns:
        [E, N, 2] per-bead displacement.
    """
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True
    E, N, _ = center.shape
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    bw = wp.from_torch(band.to(torch.int32).contiguous(), dtype=wp.int32)
    out_t = torch.empty(E * N, 2, device=center.device, dtype=torch.float32)
    ow = wp.from_torch(out_t, dtype=wp.vec2f)
    wp.launch(_sep_kernel, dim=E * N, inputs=[cf, bw, N, float(target), ow], device=str(center.device))
    torch.cuda.synchronize()  # order Warp's write before torch reads (graph capture removes this later)
    return out_t.view(E, N, 2)
