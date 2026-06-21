"""Pure-Warp first-stage centerline generator: convex-hull + midpoint displacement.

Method (registered as ``"hull"``):

1. Sample P random points per env (Warp RNG, same grid-cell scheme as the bezier
   generator so the coordinate range matches bezier exactly).
2. Build a hull-like base loop by sorting the first ``m = count[e]`` points ASCENDING by
   their centroid-relative angle (the cheap fixed-shape stand-in for a general convex
   hull — a true hull algorithm is awkward in fixed-shape Warp). This reuses the bezier
   ccw-sort pattern.
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
"""
from __future__ import annotations

import warp as wp

from . import warp_pipeline as _pipe


@wp.kernel
def _point_count_sample_k(
    seeds: wp.array(dtype=wp.int32),
    min_num: int,
    max_num: int,
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. Per-env point count in [min_num, max_num] inclusive from a
    # single uniform draw. A multiplier (5119) distinct from the position stream's keeps
    # the count and position RNG streams decorrelated.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 5119 + 1)
    span = max_num - min_num + 1
    count = min_num + int(wp.randf(state) * float(span))
    out[e] = wp.min(count, max_num)


@wp.kernel
def _point_sample_k(
    seeds: wp.array(dtype=wp.int32),
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: float,
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Samples P points on a num_cells x num_cells grid with a
    # bounded duplicate-cell rejection (mirrors the bezier corner sampler's spread, same
    # grid placement * scale so the coordinate range matches bezier). 7919 multiplier is
    # distinct from bezier's 9781 so the two generators draw different point sets.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 7919 + 1)
    base = e * P
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        for _retry in range(8):
            dup = int(0)
            for k in range(c):
                if used[base + k] == cell:
                    dup = int(1)
            if dup == 0:
                break
            cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        used[base + c] = cell

        x = float(cell % num_cells)
        y = float(cell // num_cells)
        nx = wp.randf(state) - 0.5
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * scale,
                                 (y * cell_size + ny) * scale)


@wp.kernel
def _angle_sort_k(
    points: wp.array(dtype=wp.vec2f),
    P: int,
    count: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Orders this env's FIRST m = count[e] points ascending by the
    # centroid-relative angle atan2(dx, dy) (X first, matching the bezier ccw-sort), about
    # the centroid of those m points. Rows [m, P) are written NaN. Insertion sort reads
    # only behind its write frontier, so uninitialised keys/out scratch is never consumed.
    e = wp.tid()
    base = e * P
    m = count[e]
    if m < 1:
        m = 1
    if m > P:
        m = P

    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(P):
        if i < m:
            p = points[base + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])
    cx = wp.float32(sx / wp.float64(m))
    cy = wp.float32(sy / wp.float64(m))

    for c in range(P):
        if c < m:
            p = points[base + c]
            key = wp.atan2(p[0] - cx, p[1] - cy)   # X first!
            j = c - 1
            while j >= 0 and keys[base + j] > key:
                keys[base + j + 1] = keys[base + j]
                out[base + j + 1] = out[base + j]
                j = j - 1
            keys[base + j + 1] = key
            out[base + j + 1] = p
        else:
            out[base + c] = wp.vec2f(wp.nan, wp.nan)


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


@wp.func
def _aug_vertex(c: wp.array(dtype=wp.vec2f), b: int, i: int, cnt: int) -> wp.vec2f:
    # Wrap index i into [0, cnt) (the augmented loop is closed) and read that vertex.
    return c[b + (i % cnt)]


@wp.kernel
def _catmull_rom_k(
    aug: wp.array(dtype=wp.vec2f),
    aug_count: wp.array(dtype=wp.int32),
    AP: int,
    npseg: int,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per dense sample t (dim = E * AP * npseg). Decodes (e, segment i, sample
    # s), evaluates the closed Catmull-Rom spline of segment i (from vertex i to vertex
    # i+1) at u = s/(npseg-1) using the four control vertices i-1, i, i+1, i+2 (indices
    # wrapped mod cnt = aug_count[e]). Segments i >= cnt write NaN and drop out via the
    # arc-resample. AP = 2*P is the augmented-loop stride per env.
    t = wp.tid()
    per_env = AP * npseg
    e = t // per_env
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * AP
    cnt = aug_count[e]

    if i >= cnt or cnt < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    p0 = _aug_vertex(aug, b, i - 1 + cnt, cnt)
    p1 = _aug_vertex(aug, b, i, cnt)
    p2 = _aug_vertex(aug, b, i + 1, cnt)
    p3 = _aug_vertex(aug, b, i + 2, cnt)

    u = float(s) / float(npseg - 1)
    u2 = u * u
    u3 = u2 * u
    # Uniform Catmull-Rom basis (tension 0.5), passes through p1 (u=0) and p2 (u=1).
    out[t] = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


@wp.kernel
def _assemble_polygon_selected_k(
    aug: wp.array(dtype=wp.vec2f),
    aug_count: wp.array(dtype=wp.int32),
    AP: int,
    npseg: int,
    active: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Straight-piece fallback over the augmented hull loop, only for envs whose smooth
    # Catmull-Rom resample self-crossed. Inactive rows are left untouched because the
    # selected arc-resampler skips them.
    t = wp.tid()
    per_env = AP * npseg
    e = t // per_env
    if active[e] <= 0:
        return

    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * AP
    cnt = aug_count[e]
    cm = wp.max(cnt, 1)

    if i >= cnt or cnt < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    c0 = aug[b + i]
    c1 = aug[b + (i + 1) % cm]
    u = float(s) / float(npseg - 1)
    out[t] = (1.0 - u) * c0 + u * c1


def point_count_sample_inplace(seeds_wp, config, out_count):
    """In-place: per-env point count in [min_num_points, max_num_points]. Zero alloc."""
    _pipe._init()
    E = out_count.shape[0]
    wp.launch(_point_count_sample_k, dim=E,
              inputs=[seeds_wp, int(config.min_num_points), int(config.max_num_points),
                      out_count],
              device=str(out_count.device))
    _pipe._sync(out_count.device)


def point_sample_inplace(seeds_wp, config, out_points, used_scratch):
    """In-place: writes [E*P] vec2f random points into out_points. Zero alloc."""
    _pipe._init()
    E = seeds_wp.shape[0]
    P = int(config.max_num_points)
    num_cells = int(1.0 / (config.min_point_distance * 2))
    nc2 = num_cells * num_cells
    cell_size = config.min_point_distance * 2.0
    dev = str(out_points.device)
    wp.launch(_pipe._fill_i32_k, dim=E * P, inputs=[used_scratch, -1], device=dev)
    wp.launch(_point_sample_k, dim=E,
              inputs=[seeds_wp, num_cells, nc2, float(cell_size), float(config.scale),
                      P, used_scratch, out_points],
              device=dev)
    _pipe._sync(out_points.device)


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
    _pipe._init()
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
                       scratch) -> None:
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
    """
    _pipe._init()
    assert scratch is not None, "generate_hull_warp requires scratch"

    E = scratch.count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    AP = 2 * P
    M = AP * npseg  # dense points per env
    disp = float(getattr(config, "hull_displacement", 0.15))
    dev = str(out_centerline.device)

    # Step 1: sample point count + positions.
    point_count_sample_inplace(seeds_wp, config, scratch.count)
    point_sample_inplace(seeds_wp, config, scratch.points, scratch.used)

    # Step 2: angle-sort the first m points about their centroid.
    wp.launch(_angle_sort_k, dim=E,
              inputs=[scratch.points, P, scratch.count, scratch.keys, scratch.sorted],
              device=dev)

    # Step 3: midpoint-displacement layer -> augmented 2m-vertex loop.
    wp.launch(_midpoint_displace_k, dim=E,
              inputs=[scratch.sorted, scratch.count, seeds_wp, P, disp,
                      scratch.aug, scratch.aug_count],
              device=dev)

    # Step 4: closed Catmull-Rom smoothing -> dense polyline.
    wp.launch(_catmull_rom_k, dim=E * M,
              inputs=[scratch.aug, scratch.aug_count, AP, npseg, scratch.dense],
              device=dev)

    # Step 5: arc-resample the smooth dense polyline -> out_centerline (N points per env).
    _pipe._arc_resample_inplace(scratch.dense, M, N,
                                scratch.arc_real, scratch.arc_seg, scratch.arc_s,
                                scratch.arc_cr, scratch.arc_co, out_centerline, dev)

    # Step 6: self-intersections of the smooth resample -> crossers. scratch.arc_co was
    # written by _arc_scan_k: R>=2 -> N, R<2 -> 0. Pass directly as count.
    _pipe.self_intersections_inplace(out_centerline, scratch.arc_co, scratch.crossers, N)

    # Step 7: assemble and resample the augmented polygon fallback only for crossing envs.
    # The selected resampler leaves inactive out_centerline rows untouched, so non-crossers
    # keep their Catmull-Rom result and crossers are overwritten in place.
    wp.launch(_assemble_polygon_selected_k, dim=E * M,
              inputs=[scratch.aug, scratch.aug_count, AP, npseg, scratch.crossers,
                      scratch.dense],
              device=dev)
    _pipe._arc_resample_selected_inplace(
        scratch.dense, scratch.crossers, M, N,
        scratch.arc_real, scratch.arc_seg, scratch.arc_s,
        scratch.arc_cr, scratch.arc_co, out_centerline, dev,
    )

    # Step 8: mark all envs valid (gen gate is always True; inflate does the real gate).
    wp.launch(_pipe._fill_i32_k, dim=E, inputs=[out_valid_wp, 1], device=dev)

    _pipe._sync(dev)


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="hull",
    alloc_scratch=hull_alloc_scratch,
    generate=generate_hull_warp,
))
