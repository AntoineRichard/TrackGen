"""Independent scalar numeric reference with current stream initialization."""
from __future__ import annotations
from track_gen._src import warp_generate as _implementation
from track_gen._src.runtime import _init
from track_gen.exec.context import resolve as _resolve_ctx

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

@wp.kernel
def _assemble_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    tangents: wp.array(dtype=wp.vec2f),
    scale: wp.array(dtype=wp.float32),
    P: int,
    npseg: int,
    rad: float,
    clamp_frac: float,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per dense sample t (dim = E*P*npseg). Decodes (e, segment i, sample s),
    # rebuilds the cubic Bezier of segment i (corner i -> corner (i+1)%P) and evaluates
    # it at parameter u = s/(npseg-1) with the degree-3 Bernstein basis. Mirrors
    # BezierCenterlineGenerator._segment + _cubic_bezier: handle = rad*chord along the
    # corner tangents. The count->NaN prune is folded in-kernel (corner i is NaN iff
    # i >= count[e]); NaN corners/tangents propagate into the output as in the oracle.
    t = wp.tid()
    per_env = P * npseg
    e = t // per_env
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * P
    cnt = count[e]
    cm = wp.max(cnt, 1)                       # guard %0 for degenerate (cnt==0) envs

    c0 = _pruned_corner(c, b, i, cnt)
    # Closing segment i == cnt-1 wraps to corner 0 -> a real cubic Bezier (not the old
    # straight chord). Segments i >= cnt keep c0 = NaN and drop out via the resample.
    inext = (i + 1) % cm
    c1 = _pruned_corner(c, b, inext, cnt)
    t0 = tangents[b + i]
    t1 = tangents[b + inext]

    chord = wp.length(c1 - c0)
    # F2: clamp each end's handle by clamp_frac * (that corner's shorter incident edge),
    # so a long handle can't overshoot past a nearby corner and self-cross.
    h0 = wp.min(rad * chord, clamp_frac * scale[b + i])
    h1 = wp.min(rad * chord, clamp_frac * scale[b + inext])
    p1 = c0 + t0 * h0    # leave c0 along its tangent
    p2 = c1 - t1 * h1    # arrive at c1 along its tangent

    u = float(s) / float(npseg - 1)
    omu = 1.0 - u
    b0 = omu * omu * omu          # (1-u)^3
    b1 = 3.0 * u * omu * omu      # 3u(1-u)^2
    b2 = 3.0 * u * u * omu        # 3u^2(1-u)
    b3 = u * u * u                # u^3
    out[t] = b0 * c0 + b1 * p1 + b2 * p2 + b3 * c1

def corner_count_sample_inplace(seeds_wp, attempt, config, out_count,
                                ctx: "LaunchContext | None" = None):
    """In-place: writes per-env corner counts into out_count ([E] int32 wp.array). Zero alloc.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_count.device))
    # Legacy entry point (kept for out-of-tree plugins): ops.sample_counts keyed
    # seed * CORNER_COUNT_SALT + attempt, without the public op's validation.
    _sample_counts_launch(ctx, seeds_wp, CORNER_COUNT_SALT, int(attempt),
                          int(config.min_num_points), int(config.max_num_points), out_count, keying="hashed")
    ctx.sync()

def corner_sample_inplace(seeds_wp, attempt, config, out_corners, used_scratch,
                          scale_per_env=None, ctx: "LaunchContext | None" = None):
    """In-place: writes [E*P] vec2f into out_corners. used_scratch is [E*P] int32 scratch.
    Zero alloc.

    scale_per_env: optional [E] float32 wp.array of per-env scale (method #1). When None the
    config.scale scalar is broadcast to every env; when provided, scale[e] is read per env
    (bit-identical to the scalar form for a uniform array).
    ctx: optional track_gen.exec.LaunchContext; default resolves the
        device's stream (see track_gen.exec).

    Legacy entry point (kept for out-of-tree plugins): ops.sample_grid_points keyed
    seed * CORNER_SALT + attempt on the ops.grid_cells(config.min_point_distance) grid,
    without the public op's validation.
    """
    _init()
    P = int(config.max_num_points)
    # The grid of ops.grid_cells, computed inline without its validation (as at 1077b18).
    num_cells = int(1.0 / (config.min_point_distance * 2))
    cell_size = config.min_point_distance * 2.0
    ctx = _resolve_ctx(ctx, str(out_corners.device))
    # The kernel is chosen by whether a per-env scale was given, not by its type: as at
    # 1077b18, any [E] array Warp's launch accepts (e.g. a numpy float32 array) works.
    per_env = scale_per_env is not None
    scale = scale_per_env if per_env else float(config.scale)
    _sample_grid_points_launch(ctx, seeds_wp, CORNER_SALT, int(attempt), P, num_cells,
                               float(cell_size), scale, used_scratch, out_corners,
                               per_env=per_env, keying="hashed")
    ctx.sync()

def ccw_sort_inplace(corners_wp, count_wp, keys_scratch, out_wp, P,
                     ctx: "LaunchContext | None" = None):
    """In-place: writes sorted corners into out_wp ([E*P] vec2f). keys_scratch is [E*P] float32.
    Zero alloc.

    Legacy entry point (kept for out-of-tree plugins): ops.angle_sort with the historical
    ``count < 1 -> 1`` floor (the public op treats ``count <= 0`` as an empty row).

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_wp.device))
    _angle_sort_launch(ctx, corners_wp, P, count_wp, P, keys_scratch, out_wp, m_floor=1)
    ctx.sync()

def assemble_inplace(corners_wp, count_wp, config, tan_scratch, scale_scratch, out_wp, rad_per_env=None, clamp_per_env=None, ctx: 'LaunchContext | None'=None):
    """In-place: writes dense [E*P*npseg] vec2f into out_wp. Zero alloc.
    tan_scratch: [E*P] vec2f; scale_scratch: [E*P] float32.

    rad_per_env / clamp_per_env: optional [E] float32 wp.arrays of per-env rad /
    handle_clamp_frac (method #1). When BOTH None the config scalars are broadcast via the
    original _assemble_k (DEFAULT path, byte-for-byte unchanged). When provided, the
    bit-identical _assemble_style_k reads rad[e]/clamp_frac[e] per env instead. (Both are
    threaded together: the style path always supplies both arrays.)
    ctx: optional track_gen.exec.LaunchContext; default resolves the
        device's stream (see track_gen.exec).
    """
    _init()
    E = count_wp.shape[0]
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    assert npseg >= 2
    p = math.atan(config.edgy) / math.pi + 0.5
    clamp_frac = float(getattr(config, 'handle_clamp_frac', 1000000000.0))
    dev = str(out_wp.device)
    ctx = _resolve_ctx(ctx, dev)
    ctx.launch(_vertex_tangents_k, E * P, [corners_wp, count_wp, P, float(p), tan_scratch, scale_scratch])
    ctx.launch(_assemble_k, E * P * npseg, [corners_wp, count_wp, tan_scratch, scale_scratch, P, npseg, float(config.rad), clamp_frac, out_wp])
    ctx.sync()

def _generate(seeds_wp: wp.array, config,
                              out_centerline: wp.array, out_valid_wp: wp.array,
                              scratch, ctx: "LaunchContext | None" = None) -> None:
    """Single-pass centerline generation — in-place owned path only.

    Pure-Warp: sample corners -> sort ccw -> assemble Bezier -> arc-resample -> polygon
    fallback (if self-crossing) -> write chosen centerline into out_centerline.
    Marks all envs valid (generation gate is always True; inflate does the real gate).

    The legacy entry point (frozen signature; host scalars from ``config``, the scalar
    kernels): the registered ``"bezier"`` generator is the :class:`Bezier` Stage on the
    same private core (roadmap PR13c).

    Args:
        seeds_wp:      [E] int32 wp.array per-env base seeds.
        config:        TrackGenConfig.
        out_centerline:[E*N] vec2f wp.array — written in-place with the chosen centerline.
        out_valid_wp:  [E] int32 wp.array — filled with 1 (all valid at generation stage).
        scratch:       the generation buffer group exposing the gen_* arrays — a
                       GenScratch (the composite _Scratch's ``.gen``).
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_centerline_warp requires scratch"
    _enqueue_bezier(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)

def _enqueue_bezier(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, dyn) -> None:
    """Private numeric reference of the pre-0.3 scalar path."""
    E = scratch.gen_count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M = P * npseg
    arc = ops.ArcScratch(scratch.gen_arc_real, scratch.gen_arc_seg, scratch.gen_arc_s, scratch.gen_arc_cr)
    ops.sample_counts(ctx, seeds_wp, CORNER_COUNT_SALT, 0, int(config.min_num_points), int(config.max_num_points), scratch.gen_count, keying="hashed")
    num_cells, cell = ops.grid_cells(config.min_point_distance)
    ops.sample_grid_points(ctx, seeds_wp, CORNER_SALT, 0, P, num_cells, float(cell), float(config.scale), scratch.gen_used, scratch.gen_corners, keying="hashed")
    _angle_sort_launch(ctx, scratch.gen_corners, P, scratch.gen_count, P, scratch.gen_keys, scratch.gen_ordered, m_floor=1)
    assemble_inplace(scratch.gen_ordered, scratch.gen_count, config, scratch.gen_tan, scratch.gen_scale, scratch.gen_dense, rad_per_env=None, clamp_per_env=None, ctx=ctx)
    ops.arc_resample(ctx, scratch.gen_dense, M, N, arc, scratch.gen_rs, scratch.gen_arc_co)
    ops.self_intersections(ctx, scratch.gen_rs, N, scratch.gen_crossers, count=scratch.gen_arc_co)
    ctx.launch(_assemble_polygon_selected_k, E * M, [scratch.gen_ordered, scratch.gen_count, P, npseg, scratch.gen_crossers, scratch.gen_poly])
    ops.arc_resample(ctx, scratch.gen_poly, M, N, arc, out_centerline, scratch.gen_arc_co, active=scratch.gen_crossers)
    ctx.launch(_select_vec2_k, E * N, [scratch.gen_rs, out_centerline, scratch.gen_crossers, N, out_centerline])
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def _alloc_scratch(config):
    return _implementation._alloc_bezier_scratch(config)
