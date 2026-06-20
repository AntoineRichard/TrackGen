"""Fused NVIDIA Warp XPBD relaxation: setup + solve.

This module owns the whole relaxation concern. ``band_l0_inplace`` precomputes the
per-env separation band and rest length (the relaxation SETUP) from the centerline;
``xpbd_solve_inplace`` then runs the full fixed-iteration XPBD solve (separation +
spacing + bending, double-buffered). Both run as fused Warp kernels on BOTH the Warp
``cpu`` device and ``cuda``, are strictly in-place and zero-alloc per call (all buffers
are pre-allocated by the caller), and import only ``warp`` — never the pipeline.
"""
from __future__ import annotations

import warp as wp

_INITED = False


@wp.func
def _separation_band(l0: float, two_hw: float) -> int:
    # Excluded-neighbour half-window for the XPBD separation pass and the thickness
    # gate: round(2*half_width / L0).clamp_min(1), where L0 is the mean segment length.
    # The divisor is floored at 1e-9 to bound the ratio, and the isfinite guard maps a
    # NaN/inf ratio (invalid NaN-centerline envs) to band 1. Shared by _band_l0_k (here)
    # and the pipeline's _validity_k so the band definition lives in exactly one place.
    bf = two_hw / wp.max(l0, float(1.0e-9))
    return wp.where(wp.isfinite(bf), wp.max(int(wp.round(bf)), 1), 1)

@wp.kernel
def _band_l0_k(
    center: wp.array(dtype=wp.vec2f),
    n_max: int,
    two_hw: float,
    band_out: wp.array(dtype=wp.int32),
    l0_out: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
):
    # One thread per env e. Count-aware: loop over count[e] real points, base e*n_max,
    # wrap index (i+1)%count[e]. L0 = perimeter/count[e] (mean segment length). The band
    # is _separation_band(L0, 2*hw); L0 itself may stay NaN for invalid (NaN-centerline)
    # envs (that flows untouched into xpbd, which propagates the NaN), while the band's
    # isfinite guard maps such envs to band 1.
    # PARITY: when count[e]==n_max for all e, produces identical output to the former
    # fixed-N kernel (same loop bounds, same formula).
    e = wp.tid()
    base = e * n_max
    cn = count[e]
    peri = float(0.0)
    for i in range(cn):
        peri += wp.length(center[base + (i + 1) % cn] - center[base + i])
    l0 = peri / float(cn)
    l0_out[e] = l0
    band_out[e] = _separation_band(l0, two_hw)

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
    capturing: bool = False,
) -> None:
    """Full fixed-iteration XPBD solve — strict in-place, zero per-call allocation.

    Reads ``center_wp`` (the input centerline), writes the relaxed result into
    ``relaxed_wp``.  Uses ``db_wp`` as the displacement double-buffer.  All arrays
    are pre-allocated ``wp.array`` buffers owned by the caller (e.g. the relax scratch).

    Args:
        center_wp:  [E*n_max] wp.vec2f flat input centerline (NaN-padded beyond count[e]).
        relaxed_wp: [E*n_max] wp.vec2f flat output buffer (written in-place).
        db_wp:      [E*n_max] wp.vec2f flat displacement scratch (double-buffer).
        band_wp:    [E] wp.int32 excluded-neighbour half-window per env.
        l0_wp:      [E] wp.float32 rest segment length per env.
        count_wp:   [E] wp.int32 real bead count per env.
        n_max:      int buffer stride (== total points per env slot).
        config:     TrackGenConfig (half_width, relax_margin, relax_iters, relax_*_relax).
        capturing:  True while a CUDA graph is being captured. The final host-blocking
                    wp.synchronize() is then skipped (it is ILLEGAL during capture and
                    unnecessary on replay, where the graph records stream ordering). The
                    orchestrator passes its capture state in explicitly so this module
                    never reaches back into the pipeline for it.
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
    dev = str(center_wp.device)

    # Copy input into the working buffer (relaxed_wp starts as a copy of center_wp).
    wp.copy(relaxed_wp, center_wp)

    for _ in range(int(config.relax_iters)):
        wp.launch(_disp_kernel, dim=E * n_max,
                  inputs=[relaxed_wp, band_wp, l0_wp, target, R_min, sr, pr, br,
                          n_max, count_wp, db_wp], device=dev)
        wp.launch(_apply_kernel, dim=E * n_max,
                  inputs=[relaxed_wp, db_wp], device=dev)

    # Skip the host-blocking sync during CUDA graph capture (illegal there; the graph
    # records stream ordering, so it is unnecessary on replay too). The caller passes
    # its capture state in -- this module never reads the pipeline's _CAPTURING global.
    if not capturing:
        wp.synchronize()


@wp.kernel
def _band_fill_k(arr: wp.array(dtype=wp.int32), v: int):
    # One thread per element: constant int fill (relax_band override).
    arr[wp.tid()] = v


def band_l0_inplace(
    center_wp: "wp.array",
    n_max: int,
    band_wp: "wp.array",
    l0_wp: "wp.array",
    count_wp: "wp.array",
    config,
    capturing: bool = False,
) -> None:
    """Relaxation setup — precompute the per-env separation band + rest length L0.

    Writes ``band_wp`` (excluded-neighbour half-window) and ``l0_wp`` (mean segment
    length) in place from the centerline. ``config.relax_band`` (when not None) overrides
    every env's band with that constant. Strict in-place, zero per-call allocation.

    Args:
        center_wp:  [E*n_max] wp.vec2f flat centerline (NaN-padded beyond count[e]).
        n_max:      int buffer stride (== total points per env slot).
        band_wp:    [E] wp.int32 output — separation band per env.
        l0_wp:      [E] wp.float32 output — rest segment length per env.
        count_wp:   [E] wp.int32 real bead count per env.
        config:     TrackGenConfig (uses half_width, relax_band).
        capturing:  True while a CUDA graph is being captured -> skip the host sync.
    """
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True

    E = count_wp.shape[0]
    dev = str(center_wp.device)
    two_hw = 2.0 * float(config.half_width)
    wp.launch(_band_l0_k, dim=E,
              inputs=[center_wp, n_max, two_hw, band_wp, l0_wp, count_wp], device=dev)
    if config.relax_band is not None:
        wp.launch(_band_fill_k, dim=E, inputs=[band_wp, int(config.relax_band)], device=dev)
    if not capturing and "cuda" in dev:
        wp.synchronize()


