"""Independent scalar numeric reference with current stream initialization."""
from __future__ import annotations
from track_gen._src import warp_generate_hull as _implementation
from track_gen._src.runtime import _init
from track_gen.exec.context import resolve as _resolve_ctx

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

@wp.kernel
def _midpoint_displace_k(
    sorted_pts: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    seeds: wp.array(dtype=wp.int32),
    P: int,
    disp: float,
    out_aug: wp.array(dtype=wp.vec2f),
    out_count: wp.array(dtype=wp.int32),
):
    # One thread per env e. Builds the augmented loop with 2m vertices interleaving the m
    # angle-sorted vertices and m displaced edge-midpoints:
    #   out_aug[base + 2*i]   = sorted[i]
    #   out_aug[base + 2*i+1] = midpoint(sorted[i], sorted[(i+1)%m]) + r_i * radial_dir
    # where radial_dir points from the centroid through the midpoint and r_i is a random
    # per-edge signed amount in [-disp, +disp] * (centroid->midpoint distance). Positive
    # r_i bulges the lobe OUT, negative pinches it IN — this is the racing-shape variety.
    # Rows >= 2m are written NaN. The stride for the augmented loop is 2P.
    e = wp.tid()
    sbase = e * P
    abase = e * (2 * P)
    m = count[e]
    if m < 1:
        m = 1
    if m > P:
        m = P

    # Centroid of the m real sorted vertices (for the radial displacement direction).
    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(P):
        if i < m:
            p = sorted_pts[sbase + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])
    cx = wp.float32(sx / wp.float64(m))
    cy = wp.float32(sy / wp.float64(m))
    centroid = wp.vec2f(cx, cy)

    # Per-edge displacement RNG (distinct multiplier 3083 from the count/position streams).
    state = _rng.init(seeds[e], wp.uint32(_DISPLACE_SALT))

    for i in range(P):
        if i < m:
            v0 = sorted_pts[sbase + i]
            v1 = sorted_pts[sbase + (i + 1) % m]
            mid = 0.5 * (v0 + v1)
            radial = mid - centroid
            rlen = wp.length(radial)
            rdir = radial / wp.max(rlen, 1.0e-8)
            # Signed per-edge amount in [-disp, +disp], scaled by the centroid->midpoint
            # distance so the displacement is proportional to the local track radius.
            amt = (wp.randf(state) * 2.0 - 1.0) * disp * rlen
            out_aug[abase + 2 * i] = v0
            out_aug[abase + 2 * i + 1] = mid + amt * rdir
        else:
            out_aug[abase + 2 * i] = wp.vec2f(wp.nan, wp.nan)
            out_aug[abase + 2 * i + 1] = wp.vec2f(wp.nan, wp.nan)

    out_count[e] = 2 * m

def point_count_sample_inplace(seeds_wp, config, out_count,
                               ctx: "LaunchContext | None" = None):
    """In-place: per-env point count in [min_num_points, max_num_points]. Zero alloc.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_count.device))
    # Legacy entry point (kept for out-of-tree plugins): ops.sample_counts keyed
    # seed * POINT_COUNT_SALT + POINT_SALT_OFFSET, without the public op's validation.
    _sample_counts_launch(ctx, seeds_wp, POINT_COUNT_SALT, POINT_SALT_OFFSET,
                          int(config.min_num_points), int(config.max_num_points), out_count)
    ctx.sync()

def point_sample_inplace(seeds_wp, config, out_points, used_scratch,
                         ctx: "LaunchContext | None" = None):
    """In-place: writes [E*P] vec2f random points into out_points. Zero alloc.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    P = int(config.max_num_points)
    # The grid of ops.grid_cells, computed inline without its validation (as at 1077b18).
    num_cells = int(1.0 / (config.min_point_distance * 2))
    cell_size = config.min_point_distance * 2.0
    ctx = _resolve_ctx(ctx, str(out_points.device))
    # Legacy entry point (kept for out-of-tree plugins): ops.sample_grid_points keyed
    # seed * POINT_SALT + POINT_SALT_OFFSET, without the public op's validation.
    _sample_grid_points_launch(ctx, seeds_wp, POINT_SALT, POINT_SALT_OFFSET, P, num_cells,
                               float(cell_size), float(config.scale), used_scratch,
                               out_points)
    ctx.sync()

def _generate(seeds_wp: wp.array, config,
                       out_centerline: wp.array, out_valid_wp: wp.array,
                       scratch, ctx: "LaunchContext | None" = None) -> None:
    """Convex-hull + midpoint-displacement centerline generation — in-place owned path.

    Pure-Warp: sample points -> angle-sort -> midpoint-displace -> closed Catmull-Rom dense
    -> arc-resample -> polygon fallback for self-crossers -> write the chosen centerline into
    out_centerline. Marks all envs valid (generation gate is always True; inflate does the
    real validity gate, as for bezier).

    The legacy entry point (frozen signature; ``config.hull_displacement`` as a host
    scalar): the registered ``"hull"`` generator is the :class:`Hull` Stage on the same
    private core (roadmap PR13c).

    Args:
        seeds_wp:      [E] int32 wp.array per-env base seeds.
        config:        TrackGenConfig.
        out_centerline:[E*N] vec2f wp.array — written in-place with the generated centerline.
        out_valid_wp:  [E] int32 wp.array — filled with 1 (all valid at generation stage).
        scratch:       a HullScratch from hull_alloc_scratch.
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_hull_warp requires scratch"
    _enqueue_hull(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)

def _enqueue_hull(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, dyn) -> None:
    """Private numeric reference of the pre-0.3 scalar path."""
    E = scratch.count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    AP = 2 * P
    M = AP * npseg
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)
    ops.sample_counts(ctx, seeds_wp, POINT_COUNT_SALT, POINT_SALT_OFFSET, int(config.min_num_points), int(config.max_num_points), scratch.count)
    num_cells, cell = ops.grid_cells(config.min_point_distance)
    ops.sample_grid_points(ctx, seeds_wp, POINT_SALT, POINT_SALT_OFFSET, P, num_cells, float(cell), float(config.scale), scratch.used, scratch.points)
    _angle_sort_launch(ctx, scratch.points, P, scratch.count, P, scratch.keys, scratch.sorted, m_floor=1)
    disp = float(getattr(config, 'hull_displacement', 0.15))
    ctx.launch(_midpoint_displace_k, E, [scratch.sorted, scratch.count, seeds_wp, P, disp, scratch.aug, scratch.aug_count])
    ops.closed_catmull_rom(ctx, scratch.aug, AP, npseg, scratch.dense, count=scratch.aug_count, endpoint=True)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.self_intersections(ctx, out_centerline, N, scratch.crossers, count=scratch.arc_co)
    ops.closed_polyline(ctx, scratch.aug, AP, npseg, scratch.dense, count=scratch.aug_count, endpoint=True, active=scratch.crossers)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co, active=scratch.crossers)
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def _alloc_scratch(config):
    return _implementation._alloc_hull_scratch(config)
