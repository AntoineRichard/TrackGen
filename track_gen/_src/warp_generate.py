"""Pure-Warp single-pass centerline generation.

Owns the generation concern of the pipeline: sample corners (Warp-RNG) -> ccw-sort ->
assemble cubic-Bezier dense centerline -> arc-resample -> polygon fallback for any
self-crosser -> write the chosen centerline through the Bezier Stage. The result is fed
into resample/relax/inflate by the orchestrator in ``warp_pipeline``.

Generation is single-pass: the generation gate is always True (inflate does the real
validity gate), and any track that still self-crosses falls back to its corner polygon
(``handle_clamp_frac=0``), which XPBD re-rounds.

Convention (shared with ``warp_pipeline``): one thread per output element; flat arrays
``[E*N]`` of ``wp.vec2f`` and ``[E]`` per-env scalars; env index ``e = tid // N``;
launch with ``device=str(tensor.device)``.

The shared steps (count / grid-point sampling, angle sort, NaN-aware arc resample,
self-intersection count, fills) are the public ``track_gen.ops`` primitives, the same
building blocks a plugin generator uses; Bezier-specific vertex tangents, cubic assembly
and its polygon fallback live here.
"""
from __future__ import annotations

import math

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs
from ..ops.geometry import _angle_sort_launch
from ..ops.sampling import _sample_counts_launch, _sample_grid_points_launch
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import BezierParams, _declared_bounds
from .builtin_stage import check_dyn, default_alloc, run_view
from .runtime import _init

# RNG keying of the corner streams (``wp.rand_init(seed * salt + attempt)``, see
# :func:`track_gen.ops.sample_counts`). The count and position salts differ so the two
# streams stay decorrelated; the built-in generator always uses attempt 0.
CORNER_COUNT_SALT = 6151
CORNER_SALT = 9781





@wp.func
def _pruned_corner(c: wp.array(dtype=wp.vec2f), b: int, i: int, cnt: int) -> wp.vec2f:
    # Folds in _prune_corners' NaN step: corner i of the env at base b is real iff
    # i < cnt; rows i >= cnt are replaced by (nan, nan) (same NaN positions as the
    # oracle's where(arange(P) < count, corners, nan) prune).
    ci = c[b + i]
    return wp.where(i < cnt, ci, wp.vec2f(wp.nan, wp.nan))

@wp.kernel
def _vertex_tangents_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    P: int,
    p: float,
    tangents: wp.array(dtype=wp.vec2f),
    scale: wp.array(dtype=wp.float32),
):
    # One thread per corner t (dim = E*P); e = env index, i = corner-within-env.
    # Mirrors geometry.vertex_tangents: u_out_i = dir(i -> i+1), u_in_i = dir(i-1 -> i)
    # (== roll(u_out, +1)); tangent_i = safe_normalize(p*u_out + (1-p)*u_in). The
    # count->NaN prune is folded in-kernel (corner i is NaN iff i >= count[e]); NaN at
    # any pruned corner propagates the same way the oracle's safe_normalize does.
    t = wp.tid()
    e = t // P
    i = t % P
    b = e * P
    cnt = count[e]
    cm = wp.max(cnt, 1)                       # guard %0 for degenerate (cnt==0) envs
    c_i = _pruned_corner(c, b, i, cnt)
    # Wrap mod cnt (not P): a real corner's circular neighbours are real corners, so the
    # closing edge (corner cnt-1 -> corner 0) is a genuine segment. (Mod-P-with-NaN would
    # poison the seam tangents and drop the first/last corner + close with a straight chord.)
    c_next = _pruned_corner(c, b, (i + 1) % cm, cnt)
    c_prev = _pruned_corner(c, b, (i + cm - 1) % cm, cnt)
    u_out = funcs.safe_normalize2(c_next - c_i)
    u_in = funcs.safe_normalize2(c_i - c_prev)
    blended = p * u_out + (1.0 - p) * u_in
    tangents[t] = funcs.safe_normalize2(blended)
    # Per-corner scale = shorter incident edge; the adaptive handle clamp caps handles by it.
    scale[t] = wp.min(wp.length(c_next - c_i), wp.length(c_i - c_prev))



@wp.kernel
def _assemble_dyn_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    tangents: wp.array(dtype=wp.vec2f),
    scale: wp.array(dtype=wp.float32),
    P: int,
    npseg: int,
    rad: wp.array(dtype=wp.float32),
    clamp_frac: wp.array(dtype=wp.float32),
    rad_lo: float,
    rad_hi: float,
    clamp_lo: float,
    clamp_hi: float,
    out: wp.array(dtype=wp.vec2f),
):
    # Each per-env knob is loaded once and clamped to [lo, hi] by COMPARISONS (x < lo / x > hi
    # are false for NaN and for every in-range value, signed zeros included, so an
    # in-range operand passes bit for bit; no fminf / fmaxf).
    t = wp.tid()
    per_env = P * npseg
    e = t // per_env
    rad_e = rad[e]
    rad_e = wp.where(rad_e < rad_lo, rad_lo, rad_e)
    rad_e = wp.where(rad_e > rad_hi, rad_hi, rad_e)
    cf_e = clamp_frac[e]
    cf_e = wp.where(cf_e < clamp_lo, clamp_lo, cf_e)
    cf_e = wp.where(cf_e > clamp_hi, clamp_hi, cf_e)
    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * P
    cnt = count[e]
    cm = wp.max(cnt, 1)
    c0 = _pruned_corner(c, b, i, cnt)
    inext = (i + 1) % cm
    c1 = _pruned_corner(c, b, inext, cnt)
    t0 = tangents[b + i]
    t1 = tangents[b + inext]
    chord = wp.length(c1 - c0)
    h0 = wp.min(rad_e * chord, cf_e * scale[b + i])
    h1 = wp.min(rad_e * chord, cf_e * scale[b + inext])
    p1 = c0 + t0 * h0
    p2 = c1 - t1 * h1
    u = float(s) / float(npseg - 1)
    omu = 1.0 - u
    b0 = omu * omu * omu
    b1 = 3.0 * u * omu * omu
    b2 = 3.0 * u * u * omu
    b3 = u * u * u
    out[t] = b0 * c0 + b1 * p1 + b2 * p2 + b3 * c1

@wp.kernel
def _assemble_polygon_selected_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    P: int,
    npseg: int,
    active: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Polygon fallback equivalent to _assemble_k with handle_clamp_frac=0.0, but
    # only for envs whose Bezier resample self-crossed. Non-active rows are left
    # untouched because _select_vec2_k never reads their polygon fallback slots.
    t = wp.tid()
    per_env = P * npseg
    e = t // per_env
    if active[e] <= 0:
        return

    rem = t % per_env
    i = rem // npseg
    s = rem % npseg
    b = e * P
    cnt = count[e]
    cm = wp.max(cnt, 1)

    c0 = _pruned_corner(c, b, i, cnt)
    c1 = _pruned_corner(c, b, (i + 1) % cm, cnt)

    u = float(s) / float(npseg - 1)
    omu = 1.0 - u
    b0 = omu * omu * omu
    b1 = 3.0 * u * omu * omu
    b2 = 3.0 * u * u * omu
    b3 = u * u * u
    out[t] = (b0 + b1) * c0 + (b2 + b3) * c1

@wp.kernel
def _corner_angles_gate_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    P: int,
    min_angle: float,
    ok: wp.array(dtype=wp.int32),
):
    # One thread per env e. Reproduces generate()'s ANGLE gate over this env's P RAW
    # corners with the _prune_corners NaN step folded in: corner i is REAL iff
    # i < count[e] AND both components finite. angle_ok = ((angle > min_angle) |
    # ~constrained) over all corners; a corner is "constrained" only when i and BOTH
    # its circular neighbours i-1, i+1 are real; unconstrained corners are skipped
    # (mirrors the oracle's nan_to_num(angle, 0) | ~constrained passing them). For each
    # constrained corner: interior angle = pi - acos(clamp(dot(u_in, u_out))),
    # u_in = safe_normalize(c_i - c_prev), u_out = safe_normalize(c_next - c_i), with the
    # same [-1+1e-7, 1-1e-7] cos clamp. If any constrained corner fails (not > min_angle),
    # the env's flag is 0.
    e = wp.tid()
    b = e * P
    cnt = count[e]
    flag = int(1)
    for i in range(P):
        ip = (i + P - 1) % P
        inx = (i + 1) % P
        ci = _pruned_corner(c, b, i, cnt)
        cp = _pruned_corner(c, b, ip, cnt)
        cn = _pruned_corner(c, b, inx, cnt)
        real_i = wp.isfinite(ci[0]) and wp.isfinite(ci[1])
        real_p = wp.isfinite(cp[0]) and wp.isfinite(cp[1])
        real_n = wp.isfinite(cn[0]) and wp.isfinite(cn[1])
        if real_i and real_p and real_n:
            u_in = funcs.safe_normalize2(ci - cp)
            u_out = funcs.safe_normalize2(cn - ci)
            cos = wp.clamp(wp.dot(u_in, u_out), -1.0 + 1.0e-7, 1.0 - 1.0e-7)
            angle = wp.pi - wp.acos(cos)
            if not (angle > min_angle):
                flag = int(0)
    ok[e] = flag

@wp.kernel
def _gates_combine_k(
    angle_ok: wp.array(dtype=wp.int32),
    turn: wp.array(dtype=wp.float32),
    cnt_turn: wp.array(dtype=wp.int32),
    cross_simple: wp.array(dtype=wp.int32),
    turning_tol: float,
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. Fuses generate()'s gate conjunction:
    #   turn_ok   = |(|turn| - 2*pi)| <= turning_tol
    #   finite_ok = (cnt_turn >= 2) and isfinite(turn)
    #   simple_ok = (cross_simple == 0)
    #   out       = angle_ok & turn_ok & finite_ok & simple_ok   (int 0/1 flags)
    # cnt_turn is the [E] count returned by arc_length_resample_warp(dense, npseg).
    e = wp.tid()
    tu = turn[e]
    turn_ok = int(0)
    if wp.abs(wp.abs(tu) - 2.0 * wp.pi) <= turning_tol:
        turn_ok = int(1)
    finite_ok = int(0)
    if cnt_turn[e] >= 2 and wp.isfinite(tu):
        finite_ok = int(1)
    simple_ok = int(0)
    if cross_simple[e] == 0:
        simple_ok = int(1)
    out[e] = angle_ok[e] & turn_ok & finite_ok & simple_ok

@wp.kernel
def _select_vec2_k(rs: wp.array(dtype=wp.vec2f), rs_poly: wp.array(dtype=wp.vec2f),
                   crossers: wp.array(dtype=wp.int32), N: int, out: wp.array(dtype=wp.vec2f)):
    # One thread per point t (dim=E*N). e = env index.
    # Selects rs_poly[t] if crossers[e] > 0, else rs[t] (polygon fallback for self-crossers).
    t = wp.tid()
    e = t // N
    if crossers[e] > 0:
        out[t] = rs_poly[t]
    else:
        out[t] = rs[t]

@wp.kernel
def _select_first_valid_k(
    accept: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    rs: wp.array(dtype=wp.vec2f),
    centerline: wp.array(dtype=wp.vec2f),
    N: int,
):
    # One thread per POINT t (dim = E*N); e = env index. Accept-FIRST-valid take:
    # write the freshly-resampled rs into centerline ONLY for envs newly accepted this
    # attempt (accept[e]==1 AND the OLD valid[e]==0). Reads valid but never writes it
    # (no race); _or_update_k updates valid AFTER so the take here saw the old flag.
    t = wp.tid()
    e = t // N
    if accept[e] == 1 and valid[e] == 0:
        centerline[t] = rs[t]

@wp.kernel
def _or_update_k(accept: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32)):
    # One thread per env e. valid |= accept (run AFTER _select_first_valid_k so the
    # select saw the OLD valid).
    e = wp.tid()
    if accept[e] != 0:
        valid[e] = 1









#: The dynamic fields of the Bezier Stage (roadmap PR13c / PR13d), in Params order.
_DYNAMIC = ("min_num_points", "rad", "handle_clamp_frac", "size")
_RAD_BOUNDS = _declared_bounds(BezierParams, "rad")
_CLAMP_BOUNDS = _declared_bounds(BezierParams, "handle_clamp_frac")




def _enqueue_bezier(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                    dyn) -> None:
    """Enqueue the generator with resolved per-environment parameter arrays."""
    check_dyn(dyn, _DYNAMIC, "bezier")
    E = scratch.gen_count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M = P * npseg  # dense points per env
    arc = ops.ArcScratch(scratch.gen_arc_real, scratch.gen_arc_seg, scratch.gen_arc_s,
                         scratch.gen_arc_cr)

    rad_pe, clamp_pe = dyn["rad"], dyn["handle_clamp_frac"]
    rad_bounds, clamp_bounds = _RAD_BOUNDS, _CLAMP_BOUNDS

    # Step 1-3: sample the corner count and the grid corners, sort the first count[e]
    # corners by angle about their centroid. The sort is ops.angle_sort's kernel with the
    # legacy count floor of 1 (a count < 1 sorts the first corner), so the output does not
    # depend on TrackGenConfig's min_num_points >= 2 check (a config mutated after
    # construction is not re-validated).
    _sample_counts_launch(ctx, seeds_wp, CORNER_COUNT_SALT, 0, dyn["min_num_points"],
                          int(config.max_num_points), scratch.gen_count)
    num_cells, cell = ops.grid_cells(config.min_point_distance)
    ops.sample_grid_points(ctx, seeds_wp, CORNER_SALT, 0, P, num_cells, float(cell),
                           float(config.scale),
                           scratch.gen_used, scratch.gen_corners, size=dyn["size"])
    _angle_sort_launch(ctx, scratch.gen_corners, P, scratch.gen_count, P, scratch.gen_keys,
                       scratch.gen_ordered, m_floor=1)

    # Step 4: assemble Bezier dense -> gen_dense
    p = math.atan(config.edgy) / math.pi + 0.5
    ctx.launch(_vertex_tangents_k, E * P,
               [scratch.gen_ordered, scratch.gen_count, P, float(p), scratch.gen_tan,
                scratch.gen_scale])
    ctx.launch(_assemble_dyn_k, E * P * npseg,
               [scratch.gen_ordered, scratch.gen_count, scratch.gen_tan,
                scratch.gen_scale, P, npseg, rad_pe, clamp_pe, rad_bounds[0],
                rad_bounds[1], clamp_bounds[0], clamp_bounds[1], scratch.gen_dense])

    # Step 5: arc-resample Bezier dense -> gen_rs (N points per env)
    ops.arc_resample(ctx, scratch.gen_dense, M, N, arc, scratch.gen_rs, scratch.gen_arc_co)

    # Step 6: self-intersections of the Bezier resample -> gen_crossers.
    # gen_arc_co was written by the resample: R>=2 -> N, R<2 -> 0. Pass directly as count.
    ops.self_intersections(ctx, scratch.gen_rs, N, scratch.gen_crossers,
                           count=scratch.gen_arc_co)

    # Step 7: assemble and resample the polygon fallback only for self-crossing envs.
    # Most Bezier centerlines are already simple, so non-crossers skip this fallback work.
    # (The Bernstein-form polygon kernel stays here: ops.closed_polyline's lerp rounds
    # differently in fp32.)
    ctx.launch(_assemble_polygon_selected_k, E * M,
               [scratch.gen_ordered, scratch.gen_count, P, npseg,
                scratch.gen_crossers, scratch.gen_poly])
    ops.arc_resample(ctx, scratch.gen_poly, M, N, arc, out_centerline, scratch.gen_arc_co,
                     active=scratch.gen_crossers)

    # Step 8: select: rs if no crossings, rs_poly (= out_centerline temporarily) if crossings.
    # out_centerline aliases rs_poly arg: safe per-thread (each thread reads its own slot
    # then writes it for the crossers>0 branch; reads gen_rs for the else branch).
    ctx.launch(_select_vec2_k, E * N,
               [scratch.gen_rs, out_centerline, scratch.gen_crossers, N, out_centerline])

    # Step 9: mark all envs valid (gen gate is always True; inflate does the real gate).
    ops.fill(ctx, out_valid_wp, 1)

    ctx.sync()


class GenScratch:
    """Pre-allocated generation buffers for the bezier generator's PRIVATE working scratch.

    The generation OUTPUT buffers (gen_centerline, gen_valid) are owned by the orchestrator
    (_Scratch) and passed into the private enqueue function; they are not part of this class.

    gen_count:     [E] int32 — per-env corner count (ops.sample_counts, salt
                   CORNER_COUNT_SALT).
    gen_corners:   [E*P] vec2f — raw corners (ops.sample_grid_points, salt CORNER_SALT;
                   P=max_num_points).
    gen_ordered:   [E*P] vec2f — angle-sorted corners (the ops.angle_sort kernel with the
                   legacy count floor of 1).
    gen_used:      [E*P] int32 — cell scratch of ops.sample_grid_points.
    gen_keys:      [E*P] float32 — sort-key scratch of the angle sort.
    gen_tan:       [E*P] vec2f — vertex tangents scratch for assemble_inplace.
    gen_scale:     [E*P] float32 — vertex scale scratch for assemble_inplace.
    gen_dense:     [E*P*npseg] vec2f — Bezier assembled dense centerline.
    gen_poly:      [E*P*npseg] vec2f — polygon assembled dense centerline (the
                   self-crossing fallback, handle_clamp_frac=0).
    gen_rs:        [E*num_points] vec2f — Bezier arc-resampled N-point centerline.
    gen_crossers:  [E] int32 — self-intersection counts (fallback select input).
    gen_arc_real:  [E*P*npseg] vec2f — ops.ArcScratch.real of ops.arc_resample.
    gen_arc_seg:   [E*P*npseg] float32 — ops.ArcScratch.seg.
    gen_arc_s:     [E*(P*npseg+1)] float32 — ops.ArcScratch.s.
    gen_arc_cr:    [E] int32 — ops.ArcScratch.count_real.
    gen_arc_co:    [E] int32 — ops.arc_resample's count_out (also the count=N input of
                   ops.self_intersections after the Bezier resample).
    """

    __slots__ = (
        "gen_count", "gen_corners", "gen_ordered", "gen_used", "gen_keys",
        "gen_tan", "gen_scale", "gen_dense", "gen_poly",
        "gen_rs", "gen_crossers",
        "gen_arc_real", "gen_arc_seg", "gen_arc_s", "gen_arc_cr", "gen_arc_co",
    )

    def __init__(
        self,
        gen_count: "wp.array",
        gen_corners: "wp.array",
        gen_ordered: "wp.array",
        gen_used: "wp.array",
        gen_keys: "wp.array",
        gen_tan: "wp.array",
        gen_scale: "wp.array",
        gen_dense: "wp.array",
        gen_poly: "wp.array",
        gen_rs: "wp.array",
        gen_crossers: "wp.array",
        gen_arc_real: "wp.array",
        gen_arc_seg: "wp.array",
        gen_arc_s: "wp.array",
        gen_arc_cr: "wp.array",
        gen_arc_co: "wp.array",
    ) -> None:
        self.gen_count = gen_count
        self.gen_corners = gen_corners
        self.gen_ordered = gen_ordered
        self.gen_used = gen_used
        self.gen_keys = gen_keys
        self.gen_tan = gen_tan
        self.gen_scale = gen_scale
        self.gen_dense = gen_dense
        self.gen_poly = gen_poly
        self.gen_rs = gen_rs
        self.gen_crossers = gen_crossers
        self.gen_arc_real = gen_arc_real
        self.gen_arc_seg = gen_arc_seg
        self.gen_arc_s = gen_arc_s
        self.gen_arc_cr = gen_arc_cr
        self.gen_arc_co = gen_arc_co




def _alloc_bezier_scratch(config, alloc=None):
    """the private scratch allocator with every buffer from ``alloc(name, dtype, n)``
    (named after its :class:`GenScratch` slot; default ``wp.empty`` on
    ``config.device``). The Stage passes its ``StageBuilder.scratch``."""
    _init()
    alloc = default_alloc(config) if alloc is None else alloc
    E = int(config.num_envs)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M_dense = P * npseg
    N_gen = int(config.num_points)
    return GenScratch(
        gen_count=alloc("gen_count", wp.int32, E),
        gen_corners=alloc("gen_corners", wp.vec2f, E * P),
        gen_ordered=alloc("gen_ordered", wp.vec2f, E * P),
        gen_used=alloc("gen_used", wp.int32, E * P),
        gen_keys=alloc("gen_keys", wp.float32, E * P),
        gen_tan=alloc("gen_tan", wp.vec2f, E * P),
        gen_scale=alloc("gen_scale", wp.float32, E * P),
        gen_dense=alloc("gen_dense", wp.vec2f, E * M_dense),
        gen_poly=alloc("gen_poly", wp.vec2f, E * M_dense),
        gen_rs=alloc("gen_rs", wp.vec2f, E * N_gen),
        gen_crossers=alloc("gen_crossers", wp.int32, E),
        gen_arc_real=alloc("gen_arc_real", wp.vec2f, E * M_dense),
        gen_arc_seg=alloc("gen_arc_seg", wp.float32, E * M_dense),
        gen_arc_s=alloc("gen_arc_s", wp.float32, E * (M_dense + 1)),
        gen_arc_cr=alloc("gen_arc_cr", wp.int32, E),
        gen_arc_co=alloc("gen_arc_co", wp.int32, E),
    )


class Bezier(Stage):
    """``"bezier"``: the corner-sampled cubic-Bezier track generator
    as a ``kind="generator"`` Stage.
    ``rad`` / ``handle_clamp_frac`` are dynamic: per-env arrays read by
    :func:`_assemble_dyn_k`, clamped to their declared bounds at use. Roadmap PR13d:
    ``min_num_points`` (int32) and ``size`` are dynamic too."""

    kind = "generator"
    name = "bezier"
    output = "centerline"
    capture = CaptureClass.STATIC
    Params = BezierParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}

    def build(self, b, p):
        cfg = run_view(p, b)
        scratch = _alloc_bezier_scratch(cfg, b.scratch)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        dyn = {k: b.param(k) for k in _DYNAMIC}
        seed = b.seed

        def enqueue(ctx):
            _enqueue_bezier(seed, cfg, xy, valid, scratch, ctx, dyn)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"bezier"`` track generator (the :class:`Bezier` Stage)
    into ``reg`` (called by ``track_gen._src.builtin_plugins``; importing this module
    registers nothing)."""
    reg.add(Bezier)
