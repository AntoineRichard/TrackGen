"""Fused NVIDIA Warp XPBD relaxation kernels.

``xpbd_solve`` runs the full fixed-iteration XPBD solve (separation + spacing +
bending, double-buffered) as fused Warp kernels on BOTH the Warp ``cpu`` device and
``cuda`` — this is the pipeline's relaxation stage.

``separation_disp`` is a standalone fused separation kernel: each bead loops its
neighbours and accumulates the push with NO ``[E, N, N, 2]`` materialization,
~2-3 orders of magnitude faster on CUDA while staying numerically equivalent.
"""
from __future__ import annotations

import warp as wp

_INITED = False



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
    # positions race-free.
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


def xpbd_solve_inplace(
    center_wp: "wp.array",
    relaxed_wp: "wp.array",
    db_wp: "wp.array",
    band_wp: "wp.array",
    l0_wp: "wp.array",
    count_wp: "wp.array",
    n_max: int,
    config,
) -> None:
    """Full fixed-iteration XPBD solve — strict in-place, zero per-call allocation.

    Reads ``center_wp`` (the input centerline), writes the relaxed result into
    ``relaxed_wp``.  Uses ``db_wp`` as the displacement double-buffer.  All arrays
    are pre-allocated ``wp.array`` buffers owned by the caller (e.g. ``_Scratch``).

    Args:
        center_wp:  [E*n_max] wp.vec2f flat input centerline (NaN-padded beyond count[e]).
        relaxed_wp: [E*n_max] wp.vec2f flat output buffer (written in-place).
        db_wp:      [E*n_max] wp.vec2f flat displacement scratch (double-buffer).
        band_wp:    [E] wp.int32 excluded-neighbour half-window per env.
        l0_wp:      [E] wp.float32 rest segment length per env.
        count_wp:   [E] wp.int32 real bead count per env.
        n_max:      int buffer stride (== total points per env slot).
        config:     TrackGenConfig (half_width, relax_margin, relax_iters, relax_*_relax).
    """
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True

    E = count_wp.shape[0]
    hw = float(config.half_width)
    margin = float(config.relax_margin)
    target = 2.0 * hw * (1.0 + margin)
    R_min = hw * (1.0 + margin)
    sr = float(config.relax_sep_relax)
    pr = float(config.relax_spc_relax)
    br = float(config.relax_bend_relax)
    dev = center_wp.device

    # Copy input into the working buffer (relaxed_wp starts as a copy of center_wp).
    wp.copy(relaxed_wp, center_wp)

    for _ in range(int(config.relax_iters)):
        wp.launch(_disp_kernel, dim=E * n_max,
                  inputs=[relaxed_wp, band_wp, l0_wp, target, R_min, sr, pr, br,
                          n_max, count_wp, db_wp], device=dev)
        wp.launch(_apply_kernel, dim=E * n_max,
                  inputs=[relaxed_wp, db_wp], device=dev)

    from . import warp_pipeline  # local import avoids an import cycle at module load
    if not warp_pipeline._CAPTURING:
        wp.synchronize()


