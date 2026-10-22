"""Pure-Warp centerline generator: convex-hull + midpoint displacement.

Method (registered as ``"hull"``):

1. Sample P random points per env (Warp RNG, same grid-cell scheme as the bezier
   generator so the coordinate range matches bezier exactly).
2. Build a hull-like base loop by sorting the first ``m = count[e]`` points ASCENDING by
   their centroid-relative angle (the cheap fixed-shape stand-in for a general convex
   hull — a true hull algorithm is awkward in fixed-shape Warp): the ``ops.angle_sort``
   kernel with the legacy one-point count floor, the same sort the bezier generator uses.
3. Insert ONE midpoint-displacement layer: for each edge of the angle-sorted loop, add its
   midpoint displaced along the radial direction (from the centroid) by a random per-edge
   amount in/out. The interleaved ``[v0, mid01, v1, mid12, ...]`` loop has ``2m`` vertices.
   This radial displacement is what gives the method its racing-shape variety — it is what
   makes the result DIFFERENT from a plain angle-sort (which would collapse toward bezier).
4. Smooth the augmented ``2m``-vertex loop with closed Catmull-Rom segments into a dense
   polyline (``npseg`` samples per segment) and arc-resample to ``N = config.num_points``.
5. If that smooth resample self-crosses, fall back to the augmented polygonal loop for
   that env only; XPBD can re-round the straight fallback downstream.

Coordinates use the SAME grid placement * ``config.scale`` as the bezier generator, so the
downstream constant-spacing / relax / inflate stages behave identically.

Hard invariants (shared with the rest of ``track_gen/_src``): Warp-native + torch-free,
ZERO per-call allocation (all buffers come from ``alloc_scratch``), CUDA-graph-capturable
(fixed bounds, no host sync, no per-env Python branching), deterministic via Warp RNG.

Convention (shared with ``warp_pipeline``): one thread per output element; flat arrays
``[E*N]`` of ``wp.vec2f`` and ``[E]`` per-env scalars; env index ``e = tid // N``.

Every step except the midpoint displacement is a ``track_gen.ops`` primitive (sampling,
angle sort, closed Catmull-Rom, arc resample, self-intersections, straight-polyline
fallback, fill); the angle sort runs the ``ops.angle_sort`` kernel with the legacy
one-point count floor.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.geometry import _angle_sort_launch
from ..ops.sampling import _sample_counts_launch, _sample_grid_points_launch
from .runtime import _init

# RNG keying of the point streams (``wp.rand_init(seed * salt + POINT_SALT_OFFSET)``, see
# :func:`track_gen.ops.sample_counts`). The salts differ from bezier's (6151 / 9781) so the
# two generators draw different point sets, and from each other so the count and position
# streams stay decorrelated.
POINT_COUNT_SALT = 5119
POINT_SALT = 7919
POINT_SALT_OFFSET = 1


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
    state = wp.rand_init(seeds[e] * 3083 + 1)

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


class HullScratch:
    """Pre-allocated private working scratch for the ``"hull"`` generator.

    points:    [E*P] vec2f — raw sampled points.
    count:     [E] int32 — per-env sampled point count m.
    used:      [E*P] int32 — dedup scratch for the grid sampler.
    sorted:    [E*P] vec2f — angle-sorted base loop (NaN-padded beyond m).
    keys:      [E*P] float32 — angle-sort key scratch.
    aug:       [E*2P] vec2f — augmented loop (orig + displaced midpoints; NaN beyond 2m).
    aug_count: [E] int32 — augmented loop length (2m) per env.
    dense:     [E*2P*npseg] vec2f — Catmull-Rom dense polyline, then selected polygon
                 fallback dense polyline for crossing rows.
    crossers:  [E] int32 — self-intersection counts for smooth hull resamples.
    arc_real:  [E*2P*npseg] vec2f — arc-resample compacted-real scratch.
    arc_seg:   [E*2P*npseg] float32 — arc-resample per-segment length scratch.
    arc_s:     [E*(2P*npseg+1)] float32 — arc-resample cumulative arc-length scratch.
    arc_cr:    [E] int32 — arc-resample real-point-count scratch.
    arc_co:    [E] int32 — arc-resample output-count scratch.
    """

    __slots__ = (
        "points", "count", "used", "sorted", "keys", "aug", "aug_count",
        "dense", "crossers", "arc_real", "arc_seg", "arc_s", "arc_cr", "arc_co",
    )

    def __init__(self, points, count, used, sorted, keys, aug, aug_count,
                 dense, crossers, arc_real, arc_seg, arc_s, arc_cr, arc_co):
        self.points = points
        self.count = count
        self.used = used
        self.sorted = sorted
        self.keys = keys
        self.aug = aug
        self.aug_count = aug_count
        self.dense = dense
        self.crossers = crossers
        self.arc_real = arc_real
        self.arc_seg = arc_seg
        self.arc_s = arc_s
        self.arc_cr = arc_cr
        self.arc_co = arc_co


def hull_alloc_scratch(config):
    """Allocate the hull generator's PRIVATE working scratch (one alloc per generator).

    The augmented loop has up to 2P vertices (P originals + P displaced midpoints), so the
    augmented / dense / arc-resample buffers are sized for AP = 2P, not P.
    """
    _init()
    E = int(config.num_envs)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    AP = 2 * P
    M_dense = AP * npseg
    dev = str(config.device)
    return HullScratch(
        points=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        used=wp.empty(E * P, dtype=wp.int32, device=dev),
        sorted=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        keys=wp.empty(E * P, dtype=wp.float32, device=dev),
        aug=wp.empty(E * AP, dtype=wp.vec2f, device=dev),
        aug_count=wp.empty(E, dtype=wp.int32, device=dev),
        dense=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        crossers=wp.empty(E, dtype=wp.int32, device=dev),
        arc_real=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        arc_seg=wp.empty(E * M_dense, dtype=wp.float32, device=dev),
        arc_s=wp.empty(E * (M_dense + 1), dtype=wp.float32, device=dev),
        arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        arc_co=wp.empty(E, dtype=wp.int32, device=dev),
    )


def generate_hull_warp(seeds_wp: wp.array, config,
                       out_centerline: wp.array, out_valid_wp: wp.array,
                       scratch, ctx: "LaunchContext | None" = None) -> None:
    """Convex-hull + midpoint-displacement centerline generation — in-place owned path.

    Pure-Warp: sample points -> angle-sort -> midpoint-displace -> closed Catmull-Rom dense
    -> arc-resample -> polygon fallback for self-crossers -> write the chosen centerline into
    out_centerline. Marks all envs valid (generation gate is always True; inflate does the
    real validity gate, as for bezier).

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

    E = scratch.count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    AP = 2 * P
    M = AP * npseg  # dense points per env
    disp = float(getattr(config, "hull_displacement", 0.15))
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    # Step 1: sample point count + positions.
    ops.sample_counts(ctx, seeds_wp, POINT_COUNT_SALT, POINT_SALT_OFFSET,
                      int(config.min_num_points), int(config.max_num_points), scratch.count)
    num_cells, cell = ops.grid_cells(config.min_point_distance)
    ops.sample_grid_points(ctx, seeds_wp, POINT_SALT, POINT_SALT_OFFSET, P, num_cells,
                           float(cell), float(config.scale), scratch.used, scratch.points)

    # Step 2: angle-sort the first m points about their centroid. This is ops.angle_sort's
    # kernel with the legacy count floor of 1 (a count < 1 sorts the first point), so the
    # output does not depend on TrackGenConfig's min_num_points >= 2 check (a config
    # mutated after construction is not re-validated).
    _angle_sort_launch(ctx, scratch.points, P, scratch.count, P, scratch.keys,
                       scratch.sorted, m_floor=1)

    # Step 3: midpoint-displacement layer -> augmented 2m-vertex loop.
    ctx.launch(_midpoint_displace_k, E,
               [scratch.sorted, scratch.count, seeds_wp, P, disp,
                scratch.aug, scratch.aug_count])

    # Step 4: closed Catmull-Rom smoothing -> dense polyline over the 2m-vertex loop
    # (endpoint-inclusive segments, u = s/(npseg-1); segments >= 2m are NaN and drop out
    # of the arc-resample).
    ops.closed_catmull_rom(ctx, scratch.aug, AP, npseg, scratch.dense,
                           count=scratch.aug_count, endpoint=True)

    # Step 5: arc-resample the smooth dense polyline -> out_centerline (N points per env).
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)

    # Step 6: self-intersections of the smooth resample -> crossers. scratch.arc_co was
    # written by the resample: R>=2 -> N, R<2 -> 0. Pass directly as count.
    ops.self_intersections(ctx, out_centerline, N, scratch.crossers, count=scratch.arc_co)

    # Step 7: assemble and resample the augmented polygon fallback only for crossing envs.
    # The selected resampler leaves inactive out_centerline rows untouched, so non-crossers
    # keep their Catmull-Rom result and crossers are overwritten in place.
    ops.closed_polyline(ctx, scratch.aug, AP, npseg, scratch.dense, count=scratch.aug_count,
                        endpoint=True, active=scratch.crossers)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co,
                     active=scratch.crossers)

    # Step 8: mark all envs valid (gen gate is always True; inflate does the real gate).
    ops.fill(ctx, out_valid_wp, 1)

    ctx.sync()


def register(reg) -> None:
    """Register the built-in ``"hull"`` track generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .generator_registry import GeneratorSpec

    reg.add(GeneratorSpec(
        name="hull",
        alloc_scratch=hull_alloc_scratch,
        generate=generate_hull_warp,
        accepts_ctx=True,
    ))
