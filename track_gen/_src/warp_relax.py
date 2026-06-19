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

    @wp.kernel
    def _disp_kernel(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32),
                     L0: wp.array(dtype=wp.float32), target: wp.float32, R_min: wp.float32,
                     sr: wp.float32, pr: wp.float32, br: wp.float32,
                     n_max: int, count: wp.array(dtype=wp.int32),
                     out: wp.array(dtype=wp.vec2f)):
        # Full fused XPBD sweep per bead: separation + spacing + bending, Jacobi (reads
        # only `center`, writes only out[t]) so the companion _apply_kernel can update
        # positions race-free. Matches the torch _separation_disp/_spacing_disp/_bending_disp.
        # count[e] is the number of real (non-padding) beads in env e; n_max is the buffer
        # stride. Padding beads (i >= count[e]) receive disp=0 so NaN positions stay NaN.
        t = wp.tid()
        e = t // n_max
        i = t % n_max
        b = e * n_max
        # --- guard: padding bead → zero displacement (NaN center stays NaN after apply) ---
        if i >= count[e]:
            out[t] = wp.vec2f(0.0, 0.0)
            return
        xi = center[t]
        ne = count[e]           # number of real beads in this env
        # --- separation ---
        sep = wp.vec2f(0.0, 0.0)
        cnt = int(0)
        for j in range(ne):
            dd = wp.abs(i - j)
            circ = wp.min(dd, ne - dd)
            if circ > band[e]:
                diff = xi - center[b + j]
                dist = wp.max(wp.length(diff), 1.0e-9)
                pen = target - dist
                if pen > 0.0:
                    sep = sep + (0.5 * pen / dist) * diff
                    cnt += 1
        if cnt > 0:
            sep = sep / wp.float32(cnt)
        # --- spacing (edges i and i-1 toward rest length L0[e]) ---
        xn = center[b + ((i + 1) % ne)]
        xp = center[b + ((i + ne - 1) % ne)]
        dn = xn - xi
        ln = wp.max(wp.length(dn), 1.0e-9)
        dp = xi - xp
        lp = wp.max(wp.length(dp), 1.0e-9)
        spc = 0.25 * (((ln - L0[e]) / ln) * dn - ((lp - L0[e]) / lp) * dp)
        # --- bending (push apex toward neighbour-midpoint if radius < R_min, flip-clamped) ---
        a = xi - xp
        bb = xn - xi
        la = wp.length(a)
        lb = wp.length(bb)
        lc = wp.length(xn - xp)
        denom = wp.max(la * lb * lc, 1.0e-12)
        cross = a[0] * bb[1] - a[1] * bb[0]
        area = 0.5 * wp.abs(cross)
        kappa = 4.0 * area / denom
        radius = 1.0 / wp.max(kappa, 1.0e-12)
        mid = 0.5 * (xp + xn)
        toward = mid - xi
        deficit = wp.max((R_min - radius) / R_min, 0.0)
        bscale = wp.min(br * deficit, 1.0)            # clamp: never pass the chord midpoint
        out[t] = sr * sep + pr * spc + bscale * toward

    @wp.kernel
    def _apply_kernel(center: wp.array(dtype=wp.vec2f), disp: wp.array(dtype=wp.vec2f)):
        # One thread per bead: in-place XPBD position update center[t] += disp[t]
        # (the apply half of the double-buffered disp/apply sweep).
        t = wp.tid()
        center[t] = center[t] + disp[t]


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


def xpbd_solve(center0: torch.Tensor, band: torch.Tensor, L0: torch.Tensor, config,
               count: torch.Tensor | None = None) -> torch.Tensor:
    """Full fixed-iteration XPBD solve in fused Warp kernels (separation + spacing +
    bending per sweep, double-buffered). Pure Warp loop — no torch ops, no per-iter
    sync, O(E*N) memory (no chunking). Numerically matches the torch _relax_xpbd sweep.

    Args:
        center0: [E, N, 2] float32 centerline (may be NaN-padded when count is given).
        band:    [E] integer excluded-neighbour index half-window.
        L0:      [E] per-track rest segment length (perimeter/count[e]).
        config:  TrackGenConfig (half_width, relax_margin, relax_iters, relax_*_relax).
        count:   Optional [E] int32 tensor of real bead counts per track. When None,
                 defaults to torch.full((E,), N) — parity path, bit-identical to the
                 fixed-N behaviour (the existing tests in test_warp_relax.py verify this).
    Returns:
        [E, N, 2] relaxed centerline; padding slots (i >= count[e]) remain NaN.
    """
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True
    E, N, _ = center0.shape
    # Parity path: count=None → every env has exactly N real beads (fixed-N mode).
    n_max = N
    if count is None:
        count = torch.full((E,), N, dtype=torch.int32, device=center0.device)
    hw = float(config.half_width)
    margin = float(config.relax_margin)
    target = 2.0 * hw * (1.0 + margin)
    R_min = hw * (1.0 + margin)
    sr = float(config.relax_sep_relax)
    pr = float(config.relax_spc_relax)
    br = float(config.relax_bend_relax)
    dev = str(center0.device)

    cb = center0.reshape(E * n_max, 2).contiguous().clone()   # working buffer, updated in place
    db = torch.empty_like(cb)
    cw = wp.from_torch(cb, dtype=wp.vec2f)
    dw = wp.from_torch(db, dtype=wp.vec2f)
    bw = wp.from_torch(band.to(torch.int32).contiguous(), dtype=wp.int32)
    lw = wp.from_torch(L0.to(torch.float32).contiguous(), dtype=wp.float32)
    cntw = wp.from_torch(count.to(torch.int32).contiguous(), dtype=wp.int32)
    for _ in range(int(config.relax_iters)):
        wp.launch(_disp_kernel, dim=E * n_max,
                  inputs=[cw, bw, lw, target, R_min, sr, pr, br, n_max, cntw, dw], device=dev)
        wp.launch(_apply_kernel, dim=E * n_max, inputs=[cw, dw], device=dev)
    # Host-blocking sync is ILLEGAL during CUDA graph capture; warp_pipeline sets _CAPTURING
    # while it captures the whole pipeline (this solve included) into one graph, where stream
    # ordering already serializes the launches. Skip the sync there; keep it on the eager path
    # so the caller's torch read sees the Warp write.
    from . import warp_pipeline  # local import avoids an import cycle at module load
    if not warp_pipeline._CAPTURING:
        wp.synchronize()
    return cb.view(E, n_max, 2)
