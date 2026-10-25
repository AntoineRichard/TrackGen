"""Private pre-0.3 scalar numeric reference; never imported by the library."""
from __future__ import annotations
from track_gen._src import warp_generate_repulsive as _implementation
from track_gen._src.runtime import _init
from track_gen.exec.context import resolve as _resolve_ctx

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

def _generate(seeds_wp: wp.array, config,
                            out_centerline: wp.array, out_valid_wp: wp.array,
                            scratch, ctx: "LaunchContext | None" = None) -> None:
    """Grow E closed centerlines from small circles with the pure-Warp TP-Sobolev flow and
    write the final ``[E*num_points]`` closed loops into ``out_centerline`` (valid=1 for all
    envs; the shared downstream gate decides real validity).

    Fixed host schedule (stage transitions and resample/stall-check cadences depend only on
    the loop counter), absorbing per-env freeze. The growth loop is a
    ``ctx.bounded_loop`` (``active`` = any env not frozen, refreshed by every stall check,
    checked every stall window from the first window at the final stage), so the call is
    recordable into any CUDA graph (``capture=CaptureClass.BOUNDED``): eagerly on CPU it
    stops at the first window after every env froze (the one sanctioned host poll), on
    CUDA it never host-syncs, and every lowering writes the same bytes. The gradient is
    hand-written analytic adjoints (per-vertex gather, no atomics), so the flow is
    byte-deterministic per device. Ported from the validated spike ``grow_warp.grow_warp``
    (with the tape replaced). The registered generator is the :class:`Repulsive` Stage,
    which runs the same private core with its per-env ``size`` (roadmap PR13d); this
    frozen entry point has none and refuses a config whose ``size`` is not 1.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_repulsive_warp requires scratch"
    _generate_repulsive(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)

def _generate_repulsive(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                        size) -> None:
    """The repulsive generator's launches on ``ctx`` (the private core of
    :func:`generate_repulsive_warp` and the :class:`Repulsive` Stage). ``size``: ``None``
    (the frozen entry point: the final copy is a ``ctx.copy``) or the Stage's ``[E]``
    float32 ``size`` (roadmap PR13d: the final copy is :func:`_copy_sized_k`, an exact
    scaling of the grown loop: its growth does not see the size)."""
    s = scratch
    E = s.E
    dev = str(out_centerline.device)

    alpha = float(config.repulsive_alpha)
    beta = float(config.repulsive_beta)
    tau = float(config.repulsive_tau)
    growth = float(config.repulsive_ratchet_rate)
    w_len = float(config.repulsive_w_len)
    resample_every = int(config.repulsive_resample_every)
    stall_window = int(config.repulsive_stall_window)
    stall_tol = float(config.repulsive_stall_area_tol)
    deac_i = int(bool(config.repulsive_deactivate_obstacles))
    eps = 1e-4
    p_exp = -(beta - alpha) / 2.0
    n_wall = s.n_wall
    M = s.M
    stages = s.stages
    stage_starts = s.stage_starts
    n_iters = s.n_iters

    center = s.center
    # Periodic-resample scratch (seg/s only; the [0:E*Ncur] prefix is used per stage).
    rs_scratch = ops.ArcScratch(None, s.rs_seg, s.rs_s, None)

    # The static schedule: the stages upsampled at the top of each iteration, the stage
    # resolution of each iteration, and the transitions the budget never reached.
    ups: "list[list[int]]" = [[] for _ in range(n_iters)]
    ncur = [0] * n_iters
    idx = 0
    for it in range(n_iters):
        while idx + 1 < len(stages) and it >= stage_starts[idx + 1]:
            idx += 1
            ups[it].append(idx)
        ncur[it] = stages[idx]
    tail_ups = list(range(idx + 1, len(stages)))

    # 1. Seed-driven obstacle layout + per-env target length.
    _sample_obstacles_inplace(seeds_wp, config, s.obs_pts, s.obs_mw, dev, ctx=ctx)
    grow_lo = float(config.repulsive_grow_mult_min)
    grow_hi = float(config.repulsive_grow_mult_max)
    ctx.launch(_seed_lfinal_k, E,
               [seeds_wp, s.r_init, grow_lo, grow_hi,
                s.L_final, s.L_init, s.L_target, s.reached, s.frozen, s.area_prev])
    # The loop flag: a check can only ever read a value written in THIS generation.
    ops.fill(ctx, s.active, 1)

    # 2. Initial circle at the coarsest stage prefix.
    N0 = stages[0]
    ctx.launch(_init_circle_k, (E, N0), [center, N0, s.r_init])

    def upsample(c, i):
        # Advance one coarse-to-fine stage: arc-length resample center stages[i-1] ->
        # stages[i]. The seg/s scan scratch aliases the periodic-resample rs_seg/rs_s
        # (never live at once).
        Nold, Nnew = stages[i - 1], stages[i]
        ops.arc_resample(
            c, center[0:E * Nold], Nold, Nnew,
            ops.ArcScratch(s.arc_real[0:E * Nold], s.rs_seg[0:E * Nold],
                           s.rs_s[0:E * (Nold + 1)], s.arc_cr),
            s.rs_out[0:E * Nnew], s.arc_co)
        c.copy(center, s.rs_out, count=E * Nnew)

    def body(c, it):
        # --- coarse-to-fine stage transition (upsample center N -> next N) ---
        for i in ups[it]:
            upsample(c, i)
        Ncur = ncur[it]
        h_wp = s.h_by_n[Ncur]

        # 1. ratchet target + deactivation flag
        c.launch(_ratchet_k, E, [s.L_target, s.L_final, growth, s.reached])

        # 2. energy gradient via hand-written analytic adjoints (all 3 terms -> s.grad).
        #    Per-vertex gather, no atomics -> byte-deterministic per device. Per-vertex
        #    tangents/weights are precomputed once (O(E*N)) and shared by both pair kernels.
        c.launch(_tangent_weight_k, (E, Ncur), [center, Ncur, s.frozen, s.tang, s.wdual])
        c.launch(_tp_prepass_k, (E, Ncur),
                 [center, Ncur, alpha, beta, eps, s.frozen,
                  s.tang, s.wdual, s.wcoef, s.btan])
        c.launch(_len_coef_k, E,
                 [center, Ncur, s.L_target, s.L_init, w_len, s.frozen,
                  s.len_coef, s.peri_e])
        c.launch(_grad_gather_k, (E, Ncur),
                 [center, Ncur, alpha, beta, eps, s.tang, s.wdual, s.wcoef, s.btan,
                  s.obs_pts, s.obs_mw, M, p_exp, s.reached,
                  n_wall, deac_i, s.len_coef, s.frozen, s.grad])

        # 3. Sobolev precondition g = A^{-1} grad
        c.launch(_conv_k, (E, Ncur), [s.grad, h_wp, Ncur, s.frozen, s.g])
        # 4. length-gradient Sobolev-orthogonal projection
        c.launch(_length_grad_k, (E, Ncur), [center, Ncur, s.lg])
        c.launch(_conv_k, (E, Ncur), [s.lg, h_wp, Ncur, s.frozen, s.ainv_lg])
        c.launch(_numden_k, E, [s.g, s.lg, s.ainv_lg, Ncur, s.num, s.den])
        c.launch(_project_k, (E, Ncur), [s.g, s.ainv_lg, Ncur, s.num, s.den])
        # 5. barycenter pin + normalized step
        c.launch(_gmean_k, E, [s.g, Ncur, s.gmean])
        c.launch(_gmax_msl_k, E, [s.g, s.gmean, s.peri_e, Ncur, s.gmax, s.msl])
        c.launch(_step_k, (E, Ncur),
                 [center, s.g, s.gmean, s.msl, s.gmax, tau, Ncur, s.frozen])
        # 6. hard rescale to the ratcheted target
        c.launch(_perim_bc_k, E, [center, Ncur, s.cur_len, s.bc])
        c.launch(_rescale_k, (E, Ncur), [center, s.bc, s.cur_len, s.L_target, Ncur, s.frozen])

        # 7. periodic arc-length resample (pure Warp). Frozen envs are left untouched
        #    (absorbing freeze); every other kernel above already skips them.
        if (it + 1) % resample_every == 0:
            ops.fill(c, s.count, Ncur)
            ops.resample_uniform(c, center[0:E * Ncur], Ncur, rs_scratch,
                                 s.rs_out[0:E * Ncur], count=s.count)
            c.launch(_copy_unfrozen_k, (E, Ncur), [center, s.rs_out, Ncur, s.frozen])

        # 8. stall-stop: freeze converged envs (device-side), then refresh the loop flag.
        if Ncur == stages[-1] and (it + 1) % stall_window == 0:
            c.launch(_freeze_update_k, E,
                     [center, Ncur, s.reached, stall_tol, s.area_prev, s.frozen, s.md_out])
            ops.fill(c, s.active, 0)
            c.launch(_active_k, E, [s.frozen, s.active])

    # Windows of stall_window iterations from the first window start at the final stage:
    # the iteration before each window is a stall check (it refreshes `active`), no window
    # holds a stage transition, and once every env is frozen an iteration changes nothing
    # read below (frozen envs return early), so every lowering writes the same bytes.
    ctx.bounded_loop(n_iters, body, active=s.active,
                     first_check=_first_check(stage_starts, stall_window),
                     check_every=stall_window)

    # Defense in depth: guarantee the centerline reaches the final stage resolution
    # (stages[-1] == num_points) even if the iteration budget was too small to trigger every
    # coarse-to-fine transition on schedule. Otherwise the tail resample/copy below would pack
    # stride-Ncur data into the stride-Nmax out_centerline (silent mis-strided garbage). Config
    # validation already forbids n_iters<=0; this keeps the stride invariant unconditional.
    for i in tail_ups:
        upsample(ctx, i)
    Ncur = stages[-1]

    # final periodic resample at Nmax, then hand the closed loop to the standard tail.
    ops.fill(ctx, s.count, Ncur)
    ops.resample_uniform(ctx, center[0:E * Ncur], Ncur, rs_scratch, s.rs_out[0:E * Ncur],
                         count=s.count)
    if size is None:
        ctx.copy(out_centerline, s.rs_out, count=E * Ncur)
    else:
        ctx.launch(_copy_sized_k, E * Ncur, [s.rs_out, Ncur, size, out_centerline])
    ops.fill(ctx, out_valid_wp, 1)


def _alloc_scratch(config):
    return _implementation._alloc_repulsive_scratch(config)
