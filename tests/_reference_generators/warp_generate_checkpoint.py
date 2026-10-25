"""Private pre-0.3 scalar numeric reference; never imported by the library."""
from __future__ import annotations
from track_gen._src import warp_generate_checkpoint as _implementation
from track_gen._src.runtime import _init
from track_gen.exec.context import resolve as _resolve_ctx

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

@wp.kernel
def _sample_checkpoints_k(
    seeds: wp.array(dtype=wp.int32),
    K: int,
    C: int,
    radius_min_frac: float,
    angle_jitter: float,
    base_radius: float,
    checkpoints: wp.array(dtype=wp.vec2f),
):
    # One thread per (env e, candidate k) slot. Builds C checkpoints monotone in angle:
    # angle c = 2*pi*c/C + jitter (jitter in +/- angle_jitter * slot, slot = 2*pi/C),
    # radius ~ U(radius_min_frac*R, R), R = base_radius. Stride per slot is C.
    slot_id = wp.tid()
    e = slot_id // K
    k = slot_id % K
    state = wp.rand_init(seeds[e] * _CHECKPOINT_SALT + k * _CAND_SALT + 1)
    b = slot_id * C
    two_pi = 2.0 * wp.pi
    slot = two_pi / float(C)
    rmin = radius_min_frac * base_radius

    for c in range(C):
        jitter = (2.0 * wp.randf(state) - 1.0) * angle_jitter * slot
        ang = float(c) * slot + jitter
        r = rmin + wp.randf(state) * (base_radius - rmin)
        checkpoints[b + c] = wp.vec2f(r * wp.cos(ang), r * wp.sin(ang))

@wp.kernel
def _steer_k(
    checkpoints: wp.array(dtype=wp.vec2f),
    K: int,
    C: int,
    N: int,
    turn_rate: float,
    steer_gain: float,
    lookahead: float,
    path: wp.array(dtype=wp.vec2f),
):
    # One thread per (env, candidate) slot. Walks a fixed N-step bounded-turn path that chases
    # the C checkpoints once around, writing the OPEN path into path[slot*N .. slot*N+N).
    #
    # Step length dl = ring_perimeter / N (a reduction over the C checkpoints) pins the walk to
    # ~one lap. Start at checkpoint 0 heading toward 1; each step: bearing to the current target,
    # err = wrap(bearing - theta), dtheta = clamp(steer_gain*err, +/- turn_rate), theta += dtheta,
    # advance position by dl*(cos,sin). The target index advances when within lookahead*R of it
    # (a per-step int-counter increment — capturable, no data-dependent host branch).
    slot_id = wp.tid()
    cb = slot_id * C
    pb = slot_id * N
    two_pi = 2.0 * wp.pi

    # Ring perimeter (closed C-gon over the checkpoints) -> dl. Fixed reduction over C.
    perim = float(0.0)
    for c in range(C):
        d = checkpoints[cb + (c + 1) % C] - checkpoints[cb + c]
        perim += wp.length(d)
    dl = perim / float(N)

    # Start at checkpoint 0, heading toward checkpoint 1 (deterministic, capturable).
    p = checkpoints[cb + 0]
    d01 = checkpoints[cb + 1] - checkpoints[cb + 0]
    theta = wp.atan2(d01[1], d01[0])
    tgt = int(1)  # index of current target checkpoint

    for i in range(N):
        path[pb + i] = p
        # Advance target if close enough (chase the NEXT checkpoint). Wrap the ring.
        target = checkpoints[cb + (tgt % C)]
        if wp.length(target - p) < lookahead:
            tgt = tgt + 1
            target = checkpoints[cb + (tgt % C)]
        # Proportional bounded steering toward the target bearing.
        bearing = wp.atan2(target[1] - p[1], target[0] - p[0])
        err = wp.atan2(wp.sin(bearing - theta), wp.cos(bearing - theta))  # wrap to (-pi, pi]
        dtheta = wp.clamp(steer_gain * err, -turn_rate, turn_rate)
        theta = theta + dtheta
        p = p + dl * wp.vec2f(wp.cos(theta), wp.sin(theta))

def sample_checkpoint_rings(ctx: "LaunchContext", seeds: wp.array, K: int, C: int,
                            radius_min_frac: float, angle_jitter: float, out: wp.array, *,
                            base_radius: float = BASE_RADIUS) -> None:
    """Draw ``K`` candidate rings of ``C`` checkpoints per env into ``out``.

    Slot ``e*K + k`` (env ``e``, candidate ``k``) gets ``C`` checkpoints monotone in
    angle: checkpoint ``c`` at angle ``2*pi*c/C`` plus a jitter of up to
    ``angle_jitter`` of the slot ``2*pi/C``, radius uniform in
    ``[radius_min_frac*base_radius, base_radius)``, from
    ``wp.rand_init(seeds[e] * 4099 + k * 2741 + 1)``. ``out`` is ``[E*K*C]`` ``vec2f``.
    One launch (``seeds.shape[0] * K`` threads); no allocation, no host sync, no
    validation.
    """
    K = int(K)
    ctx.launch(_sample_checkpoints_k, seeds.shape[0] * K,
               [seeds, K, int(C), float(radius_min_frac), float(angle_jitter),
                float(base_radius), out])

def _generate(seeds_wp: wp.array, config,
                             out_centerline: wp.array, out_valid_wp: wp.array,
                             scratch, ctx: "LaunchContext | None" = None) -> None:
    """Checkpoint-steering centerline generation — in-place owned path.

    Pure-Warp: sample checkpoints -> steer (fixed-N) -> close_heading_ramp -> best-of-K select
    -> (opt-in) single-crossing clip -> normalize -> write the chosen centerline into
    out_centerline. Marks all envs valid (the generation gate is always True; inflate does the
    real validity gate, as for bezier/polar/hull).

    The legacy entry point (frozen signature; the knobs of ``config`` as host scalars):
    the registered ``"checkpoint"`` generator is the :class:`Checkpoint` Stage on the same
    private core (roadmap PR13c).

    Args:
        seeds_wp:       [E] int32 wp.array per-env base seeds.
        config:         TrackGenConfig.
        out_centerline: [E*num_points] vec2f wp.array — written in-place with the centerline.
        out_valid_wp:   [E] int32 wp.array — filled with 1 (all valid at generation stage).
        scratch:        a CheckpointScratch from checkpoint_alloc_scratch.
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).

    K > 1 / K == 1 and clip_fallback are capture-time Python branches (no per-env host control
    flow), so the captured graph stays fixed and allocation-free.
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_checkpoint_warp requires scratch"
    _enqueue_checkpoint(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)

def _enqueue_checkpoint(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, dyn) -> None:
    """Private numeric reference of the pre-0.3 scalar path."""
    E = int(config.num_envs)
    N = int(config.num_points)
    C = int(config.checkpoint_count)
    K = max(int(config.checkpoint_best_of_k), 1)
    EK = E * K
    steer_gain = float(config.checkpoint_steer_gain)
    lookahead = float(config.checkpoint_lookahead_frac) * BASE_RADIUS
    target_extent = float(config.scale) * TARGET_EXTENT
    radius_min_frac = float(config.checkpoint_radius_min_frac)
    angle_jitter = float(config.checkpoint_angle_jitter)
    turn_rate = float(config.checkpoint_turn_rate)
    sample_checkpoint_rings(ctx, seeds_wp, K, C, radius_min_frac, angle_jitter, scratch.checkpoints)
    ctx.launch(_steer_k, EK, [scratch.checkpoints, K, C, N, turn_rate, steer_gain, lookahead, scratch.path])
    ctx.launch(_close_heading_ramp_k, EK, [scratch.path, N, scratch.cand])
    if K == 1:
        ctx.launch(_copy_single_k, E * N, [scratch.cand, N, out_centerline])
    else:
        ops.fill(ctx, scratch.cand_cnt, N)
        ops.self_intersections(ctx, scratch.cand, N, scratch.cross, count=scratch.cand_cnt)
        ctx.launch(_select_best_k, E, [scratch.cand, scratch.cross, K, N, out_centerline])
    if bool(getattr(config, 'checkpoint_clip_fallback', False)):
        assert scratch.clip_dense is not None, 'checkpoint_clip_fallback=True requires scratch allocated with the same flag; reallocate the generator after changing checkpoint_clip_fallback.'
        M = N + 2
        ctx.launch(_clip_assemble_k, E, [out_centerline, N, M, scratch.clip_dense])
        ops.arc_resample(ctx, scratch.clip_dense, M, N, ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr), out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def _alloc_scratch(config):
    return _implementation._alloc_checkpoint_scratch(config)
