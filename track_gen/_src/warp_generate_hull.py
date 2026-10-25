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

Roadmap PR13c: the registered ``"hull"`` generator is the :class:`Hull` Stage, whose
``hull_displacement`` is dynamic (a per-env array, :func:`_midpoint_displace_dyn_k`);
the private :func:`_enqueue_hull` core launches its per-env array-reading kernel.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.geometry import _angle_sort_launch
from ..ops.sampling import _sample_counts_launch, _sample_grid_points_launch
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import HullParams, _declared_bounds
from .builtin_stage import check_dyn, default_alloc, run_view
from .runtime import _init

# RNG keying of the point streams (``wp.rand_init(seed * salt + POINT_SALT_OFFSET)``, see
# :func:`track_gen.ops.sample_counts`). The salts differ from bezier's (6151 / 9781) so the
# two generators draw different point sets, and from each other so the count and position
# streams stay decorrelated.
POINT_COUNT_SALT = 5119
POINT_SALT = 7919
POINT_SALT_OFFSET = 1




# The declared bounds of HullParams.hull_displacement (the twin's clamp, float32).
_DISP_LO = wp.constant(wp.float32(_declared_bounds(HullParams, "hull_displacement")[0]))
_DISP_HI = wp.constant(wp.float32(_declared_bounds(HullParams, "hull_displacement")[1]))


@wp.kernel
def _midpoint_displace_dyn_k(
    sorted_pts: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    seeds: wp.array(dtype=wp.int32),
    P: int,
    disp_e: wp.array(dtype=wp.float32),
    out_aug: wp.array(dtype=wp.vec2f),
    out_count: wp.array(dtype=wp.int32),
):
    # _midpoint_displace_k with the displacement read per env (roadmap PR13c): loaded once,
    # clamped to the declared [0, 1] by comparisons (NaN and every in-range value, signed
    # zeros included, pass bit for bit), then the scalar kernel's body verbatim.
    e = wp.tid()
    disp = disp_e[e]
    disp = wp.where(disp < _DISP_LO, _DISP_LO, disp)
    disp = wp.where(disp > _DISP_HI, _DISP_HI, disp)
    sbase = e * P
    abase = e * (2 * P)
    m = count[e]
    if m < 1:
        m = 1
    if m > P:
        m = P
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
    state = wp.rand_init(seeds[e] * 3083 + 1)
    for i in range(P):
        if i < m:
            v0 = sorted_pts[sbase + i]
            v1 = sorted_pts[sbase + (i + 1) % m]
            mid = 0.5 * (v0 + v1)
            radial = mid - centroid
            rlen = wp.length(radial)
            rdir = radial / wp.max(rlen, 1.0e-8)
            amt = (wp.randf(state) * 2.0 - 1.0) * disp * rlen
            out_aug[abase + 2 * i] = v0
            out_aug[abase + 2 * i + 1] = mid + amt * rdir
        else:
            out_aug[abase + 2 * i] = wp.vec2f(wp.nan, wp.nan)
            out_aug[abase + 2 * i + 1] = wp.vec2f(wp.nan, wp.nan)
    out_count[e] = 2 * m






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


#: The dynamic fields of the Hull Stage (roadmap PR13c).
_DYNAMIC = ("min_num_points", "hull_displacement", "size")




def _alloc_hull_scratch(config, alloc=None):
    """the private scratch allocator with every buffer from ``alloc(name, dtype, n)``
    (named after its :class:`HullScratch` slot; default ``wp.empty`` on
    ``config.device``). The Stage passes its ``StageBuilder.scratch``."""
    _init()
    alloc = default_alloc(config) if alloc is None else alloc
    E = int(config.num_envs)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    AP = 2 * P
    M_dense = AP * npseg
    return HullScratch(
        points=alloc("points", wp.vec2f, E * P),
        count=alloc("count", wp.int32, E),
        used=alloc("used", wp.int32, E * P),
        sorted=alloc("sorted", wp.vec2f, E * P),
        keys=alloc("keys", wp.float32, E * P),
        aug=alloc("aug", wp.vec2f, E * AP),
        aug_count=alloc("aug_count", wp.int32, E),
        dense=alloc("dense", wp.vec2f, E * M_dense),
        crossers=alloc("crossers", wp.int32, E),
        arc_real=alloc("arc_real", wp.vec2f, E * M_dense),
        arc_seg=alloc("arc_seg", wp.float32, E * M_dense),
        arc_s=alloc("arc_s", wp.float32, E * (M_dense + 1)),
        arc_cr=alloc("arc_cr", wp.int32, E),
        arc_co=alloc("arc_co", wp.int32, E),
    )




def _enqueue_hull(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                  dyn) -> None:
    """Enqueue the generator with resolved per-environment parameter arrays."""
    check_dyn(dyn, _DYNAMIC, "hull")
    E = scratch.count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    AP = 2 * P
    M = AP * npseg  # dense points per env
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    # Step 1: sample point count + positions.
    _sample_counts_launch(ctx, seeds_wp, POINT_COUNT_SALT, POINT_SALT_OFFSET,
                          dyn["min_num_points"], int(config.max_num_points),
                          scratch.count)
    num_cells, cell = ops.grid_cells(config.min_point_distance)
    ops.sample_grid_points(ctx, seeds_wp, POINT_SALT, POINT_SALT_OFFSET, P, num_cells,
                           float(cell), float(config.scale), scratch.used,
                           scratch.points, size=dyn["size"])

    # Step 2: angle-sort the first m points about their centroid. This is ops.angle_sort's
    # kernel with the legacy count floor of 1 (a count < 1 sorts the first point), so the
    # output does not depend on TrackGenConfig's min_num_points >= 2 check (a config
    # mutated after construction is not re-validated).
    _angle_sort_launch(ctx, scratch.points, P, scratch.count, P, scratch.keys,
                       scratch.sorted, m_floor=1)

    # Step 3: midpoint-displacement layer -> augmented 2m-vertex loop.
    ctx.launch(_midpoint_displace_dyn_k, E,
               [scratch.sorted, scratch.count, seeds_wp, P, dyn["hull_displacement"],
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


class Hull(Stage):
    """``"hull"``: the convex-hull + midpoint-displacement track generator
    as a ``kind="generator"`` Stage.
    ``hull_displacement`` is dynamic: a per-env array read by
    :func:`_midpoint_displace_dyn_k`, clamped to ``[0, 1]`` at use. Roadmap PR13d:
    ``min_num_points`` (int32) and ``size`` too."""

    kind = "generator"
    name = "hull"
    output = "centerline"
    capture = CaptureClass.STATIC
    Params = HullParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}

    def build(self, b, p):
        cfg = run_view(p, b)
        scratch = _alloc_hull_scratch(cfg, b.scratch)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        dyn = {k: b.param(k) for k in _DYNAMIC}
        seed = b.seed

        def enqueue(ctx):
            _enqueue_hull(seed, cfg, xy, valid, scratch, ctx, dyn)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"hull"`` track generator (the :class:`Hull` Stage) into
    ``reg`` (called by ``track_gen._src.builtin_plugins``; importing this module
    registers nothing)."""
    reg.add(Hull)
