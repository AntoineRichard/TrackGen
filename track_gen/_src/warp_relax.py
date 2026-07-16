"""NVIDIA Warp XPBD relaxation: setup + solve.

This module owns the whole relaxation concern. ``band_l0_inplace`` precomputes the
per-env separation band and rest length (the relaxation SETUP) from the centerline;
``xpbd_solve_inplace`` then runs the full fixed-iteration XPBD solve (separation +
spacing + bending, double-buffered). Separation can run in the dense baseline mode or in
an optional cached broadphase/narrowphase mode: rebuild candidate pairs every K sweeps,
then apply exact separation against cached candidates every sweep.

The Jacobi sweeps can optionally be accelerated with the Chebyshev semi-iterative
method (Macklin et al., "Unified Particle Physics for Real-Time Applications", 2014),
selected by ``config.relax_accel == "chebyshev"``: after a warmup of plain Jacobi
sweeps (``relax_cheby_start``), each subsequent sweep launches a small blend kernel
that combines the Jacobi update with the current and previous iterates using a
host-precomputed omega recurrence, cutting the sweeps needed for a given yield
roughly 3x. The schedule is data-independent and the launch sequence fixed, so the
accelerated solve stays CUDA-graph-capture safe.

After the main sweep loop an optional post-solve smoothing tail runs: a few shrink-free
Taubin passes (``relax_smooth_passes``) followed by spacing-only polish sweeps
(``relax_smooth_spacing_iters``), removing sub-R_min curvature noise the bending guard's
deadband cannot see while keeping bead uniformity. It is a fixed launch sequence too.

Both paths run as Warp kernels on BOTH the Warp ``cpu`` device and ``cuda``, are
strictly in-place and zero-alloc per call (all buffers are pre-allocated by the
caller; see ``cheb_prev_wp``), and import only ``warp`` — never the pipeline.
"""
from __future__ import annotations

import warp as wp

_INITED = False

# Post-solve Taubin smoothing tail: one pass = two half-steps, x += LAMBDA*L(x) then
# x += MU*L(x), with L the circular Laplacian. mu < -lambda is what makes the pass
# shrink-free (a plain Laplacian, mu=0, would shrink corners through the validity gate,
# which was measured to collapse yield). Fixed constants — deliberately not config.
_TAUBIN_LAMBDA = 0.5
_TAUBIN_MU = -0.53


def _cheby_schedule(n_iters: int, rho: float, start: int) -> list:
    """Host-precomputed per-sweep omega for Chebyshev semi-iterative acceleration of
    Jacobi (Macklin et al. 2014). ``rho`` is the spectral-radius estimate of the
    Jacobi iteration; ``start`` is the delayed-start sweep index (earlier sweeps run
    plain Jacobi so the separation contact set settles before acceleration engages).
    Entries below ``start`` are placeholders (1.0) — the solve loop skips the blend
    launch entirely for those sweeps. Recurrence: omega_1 = 1, omega_2 = 2/(2-rho^2),
    omega_{k+1} = 4/(4 - rho^2*omega_k)."""
    r2 = rho * rho
    omegas = [1.0] * n_iters
    omega = 1.0
    for k in range(start, n_iters):
        j = k - start
        if j == 0:
            omega = 1.0
        elif j == 1:
            omega = 2.0 / (2.0 - r2)
        else:
            omega = 4.0 / (4.0 - r2 * omega)
        omegas[k] = omega
    return omegas


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
def _step_kernel(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32),
                 L0: wp.array(dtype=wp.float32), target: wp.float32, R_min: wp.float32,
                 sr: wp.float32, pr: wp.float32, br: wp.float32, do_sep: int,
                 n_max: int, count: wp.array(dtype=wp.int32),
                 out: wp.array(dtype=wp.vec2f)):
    # Dense/cadenced XPBD sweep per bead: separation + spacing + bending. This keeps
    # Jacobi semantics by reading only `center` and writing updated positions to `out`.
    # count[e] is the number of real (non-padding) beads in env e; n_max is the buffer
    # stride. Padding beads copy through so NaN-padded tails stay NaN with odd iters.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    if i >= count[e]:
        out[t] = center[t]
        return
    xi = center[t]
    ne = count[e]           # number of real beads in this env
    band_e = band[e]
    l0_e = L0[e]
    target2 = target * target
    # --- separation ---
    sep = wp.vec2f(0.0, 0.0)
    cnt = int(0)
    if do_sep != 0:
        for j in range(ne):
            dd = wp.abs(i - j)
            circ = wp.min(dd, ne - dd)
            if circ > band_e:
                diff = xi - center[b + j]
                adx = wp.abs(diff[0])
                ady = wp.abs(diff[1])
                if adx < target and ady < target:
                    dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                    if dist2 < target2:
                        dist = wp.max(wp.sqrt(dist2), 1.0e-9)
                        pen = target - dist
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
    spc = 0.25 * (((ln - l0_e) / ln) * dn - ((lp - l0_e) / lp) * dp)
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
    step = sr * sep + pr * spc + bscale * toward
    out[t] = xi + step


@wp.kernel
def _build_sep_cache_kernel(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32),
                            radius: wp.float32, n_max: int, cache_slots: int,
                            count: wp.array(dtype=wp.int32),
                            cache_count: wp.array(dtype=wp.int32),
                            cache_idx: wp.array(dtype=wp.int32),
                            overflow: wp.array(dtype=wp.int32)):
    # Broadphase refresh: one thread per bead builds a directed fixed-slot list of
    # non-neighbour beads inside radius = target*(1 + skin). This intentionally stores
    # candidate indices only. The cached XPBD step recomputes the current exact distance
    # every sweep and applies separation only when dist < target.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    if i >= count[e]:
        cache_count[t] = 0
        return

    xi = center[t]
    ne = count[e]
    band_e = band[e]
    radius2 = radius * radius
    n_cached = int(0)
    did_overflow = int(0)

    for j in range(ne):
        dd = wp.abs(i - j)
        circ = wp.min(dd, ne - dd)
        if circ > band_e:
            diff = xi - center[b + j]
            adx = wp.abs(diff[0])
            ady = wp.abs(diff[1])
            if adx < radius and ady < radius:
                dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                if dist2 < radius2:
                    if n_cached < cache_slots:
                        cache_idx[t * cache_slots + n_cached] = j
                        n_cached += 1
                    else:
                        did_overflow = 1

    cache_count[t] = n_cached
    if did_overflow != 0:
        wp.atomic_add(overflow, 0, 1)


@wp.kernel
def _step_cached_kernel(center: wp.array(dtype=wp.vec2f), cache_count: wp.array(dtype=wp.int32),
                        cache_idx: wp.array(dtype=wp.int32), L0: wp.array(dtype=wp.float32),
                        target: wp.float32, R_min: wp.float32, sr: wp.float32,
                        pr: wp.float32, br: wp.float32, n_max: int, cache_slots: int,
                        count: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
    # Narrowphase cached sweep. The broadphase candidate list may be stale, so every
    # cached pair is re-tested against the exact current target before applying any push.
    # Spacing and bending are identical to the dense step kernel.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    if i >= count[e]:
        out[t] = center[t]
        return

    xi = center[t]
    ne = count[e]
    l0_e = L0[e]
    target2 = target * target

    sep = wp.vec2f(0.0, 0.0)
    cnt = int(0)
    n_cached = cache_count[t]
    for slot in range(cache_slots):
        if slot < n_cached:
            j = cache_idx[t * cache_slots + slot]
            diff = xi - center[b + j]
            adx = wp.abs(diff[0])
            ady = wp.abs(diff[1])
            if adx < target and ady < target:
                dist2 = diff[0] * diff[0] + diff[1] * diff[1]
                if dist2 < target2:
                    dist = wp.max(wp.sqrt(dist2), 1.0e-9)
                    pen = target - dist
                    sep = sep + (0.5 * pen / dist) * diff
                    cnt += 1
    if cnt > 0:
        sep = sep / wp.float32(cnt)

    xn = center[b + ((i + 1) % ne)]
    xp = center[b + ((i + ne - 1) % ne)]
    dn = xn - xi
    ln = wp.max(wp.length(dn), 1.0e-9)
    dp = xi - xp
    lp = wp.max(wp.length(dp), 1.0e-9)
    spc = 0.25 * (((ln - l0_e) / ln) * dn - ((lp - l0_e) / lp) * dp)

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
    bscale = wp.min(br * deficit, 1.0)
    step = sr * sep + pr * spc + bscale * toward
    out[t] = xi + step


@wp.kernel
def _accel_blend_kernel(x_hat: wp.array(dtype=wp.vec2f), x_cur: wp.array(dtype=wp.vec2f),
                        x_prev: wp.array(dtype=wp.vec2f), omega: wp.float32,
                        gamma: wp.float32, out: wp.array(dtype=wp.vec2f)):
    # Chebyshev semi-iterative blend, launched AFTER a Jacobi sweep (x_hat = Jacobi(x_cur))
    # by both the dense and cached step paths so acceleration is path-agnostic. Semantics:
    #   x_new = omega * (gamma*(x_hat - x_cur) + (x_cur - x_prev)) + x_prev
    # with omega from the host-precomputed schedule and gamma the constant
    # under-relaxation factor. On the FIRST accelerated sweep the caller passes
    # x_prev aliasing x_cur, so (x_cur - x_prev) cancels exactly and the result is
    # independent of the previous-iterate buffer's contents (safe on CUDA graph
    # replay, where buffer contents persist between replays).
    # In-place safe: reads and writes only element t (out may alias x_hat). NaN-padded
    # beads: x_hat == x_cur (step copies padding through) -> NaN propagates unchanged.
    t = wp.tid()
    h = x_hat[t]
    c = x_cur[t]
    p = x_prev[t]
    out[t] = omega * (gamma * (h - c) + (c - p)) + p


@wp.kernel
def _taubin_kernel(center: wp.array(dtype=wp.vec2f), factor: wp.float32,
                   n_max: int, count: wp.array(dtype=wp.int32),
                   out: wp.array(dtype=wp.vec2f)):
    # One thread per bead: a single Taubin half-step. Jacobi semantics (read `center`,
    # write `out`). count[e] is the number of real beads in env e; n_max is the buffer
    # stride. Padding beads (i >= count[e]) copy through so NaN-padded tails stay NaN.
    # Real beads apply x_i + factor * (0.5*(x_prev + x_next) - x_i) with circular indexing
    # over count[e]; the caller launches this with LAMBDA then MU for a shrink-free pass.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    b = e * n_max
    ne = count[e]
    if i >= ne:
        out[t] = center[t]
        return
    xi = center[t]
    xn = center[b + ((i + 1) % ne)]
    xp = center[b + ((i + ne - 1) % ne)]
    lap = 0.5 * (xp + xn) - xi
    out[t] = xi + factor * lap


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
    sep_cache_idx_wp: "wp.array | None" = None,
    sep_cache_count_wp: "wp.array | None" = None,
    sep_cache_overflow_wp: "wp.array | None" = None,
    cheb_prev_wp: "wp.array | None" = None,
) -> None:
    """Full fixed-iteration XPBD solve — strict in-place, zero per-call allocation.

    Reads ``center_wp`` (the input centerline), writes the relaxed result into
    ``relaxed_wp``.  Uses ``db_wp`` as the second position buffer.  All arrays
    are pre-allocated ``wp.array`` buffers owned by the caller (e.g. the relax scratch).

    Args:
        center_wp:  [E*n_max] wp.vec2f flat input centerline (NaN-padded beyond count[e]).
        relaxed_wp: [E*n_max] wp.vec2f flat output buffer (written in-place).
        db_wp:      [E*n_max] wp.vec2f flat position scratch (ping-pong buffer).
        band_wp:    [E] wp.int32 excluded-neighbour half-window per env.
        l0_wp:      [E] wp.float32 rest segment length per env.
        count_wp:   [E] wp.int32 real bead count per env.
        n_max:      int buffer stride (== total points per env slot).
        config:     TrackGenConfig (half_width, relax_margin, relax_iters, relax_*_relax,
                    relax_sep_every, relax_sep_cache_slots, relax_sep_cache_skin).
        capturing:  True while a CUDA graph is being captured. The final host-blocking
                    wp.synchronize() is then skipped (it is ILLEGAL during capture and
                    unnecessary on replay, where the graph records stream ordering). The
                    orchestrator passes its capture state in explicitly so this module
                    never reaches back into the pipeline for it.
        sep_cache_idx_wp: [E*n_max*relax_sep_cache_slots] int32 candidate indices. Required
                    only when relax_sep_cache_slots > 0 and relax_sep_every > 1.
        sep_cache_count_wp: [E*n_max] int32 candidate count per bead for cached separation.
        sep_cache_overflow_wp: [1] int32 overflow counter incremented when a bead has more
                    broadphase candidates than relax_sep_cache_slots on the latest refresh.
        cheb_prev_wp: [E*n_max] wp.vec2f previous-iterate buffer for Chebyshev
                    acceleration. Required (pre-allocated by the caller) when
                    config.relax_accel == "chebyshev" and a CUDA graph is being
                    captured; outside capture a missing buffer is allocated here as a
                    convenience for tests/standalone use. Never read before being
                    written within a solve, so its initial contents are irrelevant
                    (graph-replay safe). Unused when relax_accel == "none".
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
    sep_every = max(1, int(getattr(config, "relax_sep_every", 1)))
    cache_slots = max(0, int(getattr(config, "relax_sep_cache_slots", 0)))
    cache_skin = max(0.0, float(getattr(config, "relax_sep_cache_skin", 0.0)))
    smooth_passes = max(0, int(getattr(config, "relax_smooth_passes", 0)))
    smooth_spacing_iters = max(0, int(getattr(config, "relax_smooth_spacing_iters", 0)))
    # Cached mode means: broadphase refresh every sep_every sweeps, exact narrowphase
    # separation every sweep. Without cache buffers, sep_every > 1 is the naive skip path.
    use_cache = cache_slots > 0 and sep_every > 1
    dev = str(center_wp.device)

    if use_cache and (
        sep_cache_idx_wp is None
        or sep_cache_count_wp is None
        or sep_cache_overflow_wp is None
    ):
        raise ValueError("relax_sep_cache_slots > 0 requires pre-allocated cache buffers")

    n_iters = int(config.relax_iters)

    # --- Chebyshev acceleration setup ---------------------------------------
    # Host-precomputed, data-independent omega schedule -> fixed launch sequence, no
    # host sync or data-dependent branching inside the loop (graph-capture safe).
    use_cheby = str(getattr(config, "relax_accel", "none")) == "chebyshev"
    if use_cheby:
        gamma = float(getattr(config, "relax_cheby_gamma", 0.9))
        cheby_start = max(1, int(getattr(config, "relax_cheby_start", 8)))
        omegas = _cheby_schedule(
            n_iters, float(getattr(config, "relax_cheby_rho", 0.98)), cheby_start)
        if cheb_prev_wp is None:
            if capturing:
                # Allocation is illegal during CUDA graph capture; the capture path
                # must pre-allocate the buffer (the pipeline's RelaxScratch does).
                raise ValueError(
                    "relax_accel='chebyshev' requires a pre-allocated cheb_prev_wp "
                    "buffer during CUDA graph capture")
            # Convenience path for tests/standalone use only.
            cheb_prev_wp = wp.zeros_like(relaxed_wp)
        prev_wp = cheb_prev_wp

    # Copy input into the first position buffer, then ping-pong full position states.
    wp.copy(relaxed_wp, center_wp)
    read_wp = relaxed_wp    # x_cur
    write_wp = db_wp        # x_hat (Jacobi output; blended in place to x_new)

    for step_i in range(n_iters):
        if use_cache:
            if step_i % sep_every == 0:
                cache_radius = target * (1.0 + cache_skin)
                wp.launch(_band_fill_k, dim=1, inputs=[sep_cache_overflow_wp, 0], device=dev)
                wp.launch(_build_sep_cache_kernel, dim=E * n_max,
                          inputs=[read_wp, band_wp, cache_radius, n_max, cache_slots,
                                  count_wp, sep_cache_count_wp, sep_cache_idx_wp,
                                  sep_cache_overflow_wp], device=dev)
            wp.launch(_step_cached_kernel, dim=E * n_max,
                      inputs=[read_wp, sep_cache_count_wp, sep_cache_idx_wp, l0_wp,
                              target, R_min, sr, pr, br, n_max, cache_slots, count_wp,
                              write_wp], device=dev)
        else:
            do_sep = 1 if step_i % sep_every == 0 else 0
            wp.launch(_step_kernel, dim=E * n_max,
                      inputs=[read_wp, band_wp, l0_wp, target, R_min, sr, pr, br, do_sep,
                              n_max, count_wp, write_wp], device=dev)

        if use_cheby and step_i >= cheby_start:
            # Blend x_new = f(x_hat=write_wp, x_cur=read_wp, x_prev) in place into
            # write_wp. Pointer dance:
            #   * Sweeps < cheby_start run plain ping-pong below (no blend launch —
            #     an omega=1 identity blend is NOT bit-exact in float arithmetic, and
            #     skipping it also saves launches); prev_wp stays out of rotation.
            #   * On the TRANSITION sweep (step_i == cheby_start) the true previous
            #     iterate is not tracked yet, so x_prev aliases x_cur (read_wp):
            #     (x_cur - x_prev) cancels exactly and omega == 1 there, making the
            #     result independent of prev_wp's contents — prev_wp is therefore
            #     never read before being written (CUDA graph replay safe).
            #   * Then 3-cycle: new x_cur = write_wp (x_new), new x_prev = read_wp
            #     (old x_cur), and the freed prev_wp becomes the next sweep's Jacobi
            #     output buffer (fully overwritten by the step kernel before the
            #     blend reads it as x_hat).
            x_prev = read_wp if step_i == cheby_start else prev_wp
            wp.launch(_accel_blend_kernel, dim=E * n_max,
                      inputs=[write_wp, read_wp, x_prev,
                              wp.float32(omegas[step_i]), wp.float32(gamma),
                              write_wp], device=dev)
            read_wp, write_wp, prev_wp = write_wp, prev_wp, read_wp
        else:
            read_wp, write_wp = write_wp, read_wp

    # --- Post-solve smoothing tail ------------------------------------------
    # Continue the read/write ping-pong from wherever the main loop left it (the
    # Chebyshev prev buffer is not touched here; read_wp/write_wp are always two
    # distinct buffers). Taubin shrink-free smoothing (LAMBDA then MU per pass)
    # removes sub-R_min curvature noise the bending guard's deadband cannot see;
    # the spacing-only polish (do_sep=0, sr=0, br=0) then restores bead uniformity.
    # Fixed, data-independent launch sequence with no per-call allocation -> graph
    # capture safe. Each launch is one swap; the final copy-back below handles any
    # parity of total swaps.
    for _ in range(smooth_passes):
        wp.launch(_taubin_kernel, dim=E * n_max,
                  inputs=[read_wp, wp.float32(_TAUBIN_LAMBDA), n_max, count_wp, write_wp],
                  device=dev)
        read_wp, write_wp = write_wp, read_wp
        wp.launch(_taubin_kernel, dim=E * n_max,
                  inputs=[read_wp, wp.float32(_TAUBIN_MU), n_max, count_wp, write_wp],
                  device=dev)
        read_wp, write_wp = write_wp, read_wp
    for _ in range(smooth_spacing_iters):
        wp.launch(_step_kernel, dim=E * n_max,
                  inputs=[read_wp, band_wp, l0_wp, target, R_min, 0.0, pr, 0.0, 0,
                          n_max, count_wp, write_wp], device=dev)
        read_wp, write_wp = write_wp, read_wp

    if read_wp is not relaxed_wp:
        wp.copy(relaxed_wp, read_wp)

    # Skip the host-blocking sync during CUDA graph capture (illegal there; the graph
    # records stream ordering, so it is unnecessary on replay too). The caller passes
    # its capture state in -- this module never reads the pipeline's _CAPTURING global.
    # Gate on cuda like band_l0_inplace: on the cpu device kernels are eager, so a global
    # wp.synchronize() is pure overhead.
    if not capturing and "cuda" in dev:
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


