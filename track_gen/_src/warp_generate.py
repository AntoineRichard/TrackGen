"""Pure-Warp single-pass centerline generation.

Owns the generation concern of the pipeline: sample corners (Warp-RNG) -> ccw-sort ->
assemble cubic-Bezier dense centerline -> arc-resample -> polygon fallback for any
self-crosser -> write the chosen centerline. ``generate_centerline_warp`` is the entry
point; everything else here is a generation-only kernel or helper. The result is fed
into resample/relax/inflate by the orchestrator in ``warp_pipeline``.

Generation is single-pass: the generation gate is always True (inflate does the real
validity gate), and any track that still self-crosses falls back to its corner polygon
(``handle_clamp_frac=0``), which XPBD re-rounds.

Convention (shared with ``warp_pipeline``): one thread per output element; flat arrays
``[E*N]`` of ``wp.vec2f`` and ``[E]`` per-env scalars; env index ``e = tid // N``;
launch with ``device=str(tensor.device)``.

The shared steps (count / grid-point sampling, angle sort, NaN-aware arc resample,
self-intersection count, fills) are the public ``track_gen.ops`` primitives, the same
building blocks a plugin generator uses; only the Bezier-specific kernels (style draw,
vertex tangents, cubic assembly and its polygon fallback) live here.
"""
from __future__ import annotations

import math

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs
from ..ops.geometry import _angle_sort_launch
from ..ops.sampling import _sample_counts_launch, _sample_grid_points_launch
from .runtime import _init

# RNG keying of the corner streams (``wp.rand_init(seed * salt + attempt)``, see
# :func:`track_gen.ops.sample_counts`). The count and position salts differ so the two
# streams stay decorrelated; the built-in generator always uses attempt 0.
CORNER_COUNT_SALT = 6151
CORNER_SALT = 9781

# Legacy private name (also reachable as warp_pipeline._safe_normalize2): the same
# arithmetic as ops.funcs.safe_normalize2.
_safe_normalize2 = funcs.safe_normalize2



@wp.kernel
def _style_sample_k(
    seeds: wp.array(dtype=wp.int32),
    rad_lo: float, rad_hi: float,
    scale_lo: float, scale_hi: float,
    clamp_lo: float, clamp_hi: float,
    rad_out: wp.array(dtype=wp.float32),
    scale_out: wp.array(dtype=wp.float32),
    clamp_out: wp.array(dtype=wp.float32),
):
    # Per-env style draw for method #1 (per-env style randomization). One thread per env e.
    #
    # Seeding: state = wp.rand_init(seeds[e] * 2741). The 2741 multiplier is DISTINCT from
    # corner_count's 6151 and corner_sample's 9781, so the style stream is decorrelated from
    # both the count and the corner-position streams for the same seed. No `attempt` term:
    # generation is single-pass, so each env's style is a pure function of its seed.
    #
    # Draw ORDER (fixed): rad, then scale, then handle_clamp_frac, each a single uniform
    # randf mapped to [lo, hi). A "disabled" knob is encoded by lo == hi (the wrapper passes
    # lo == hi == scalar when its *_range is None), which collapses to the constant lo.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 2741)
    rad_out[e] = rad_lo + wp.randf(state) * (rad_hi - rad_lo)
    scale_out[e] = scale_lo + wp.randf(state) * (scale_hi - scale_lo)
    clamp_out[e] = clamp_lo + wp.randf(state) * (clamp_hi - clamp_lo)

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

@wp.kernel
def _assemble_style_k(
    c: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    tangents: wp.array(dtype=wp.vec2f),
    scale: wp.array(dtype=wp.float32),
    P: int,
    npseg: int,
    rad: wp.array(dtype=wp.float32),
    clamp_frac: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.vec2f),
):
    # Per-env (rad, clamp_frac) variant of _assemble_k (method #1). BIT-IDENTICAL to it
    # except `rad` and `clamp_frac` are read per env from device arrays (rad[e],
    # clamp_frac[e]) instead of broadcast scalars. For uniform per-env arrays equal to the
    # config scalars this reproduces _assemble_k exactly.
    t = wp.tid()
    per_env = P * npseg
    e = t // per_env
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

    rad_e = rad[e]
    cf_e = clamp_frac[e]
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
                          int(config.min_num_points), int(config.max_num_points), out_count)
    ctx.sync()

def style_sample_inplace(seeds_wp, config, rad_out, scale_out, clamp_out,
                         ctx: "LaunchContext | None" = None):
    """In-place per-env style draw (method #1). Writes [E] float32 per-env rad/scale/
    handle_clamp_frac into rad_out/scale_out/clamp_out. Zero alloc.

    Each *_range that is None collapses to the corresponding config scalar (lo == hi), so
    that knob is held constant across envs. Only called when config.style_sampling=True.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    E = seeds_wp.shape[0]
    dev = str(rad_out.device)
    ctx = _resolve_ctx(ctx, dev)
    rad_lo, rad_hi = _range_or_scalar(config.rad_range, float(config.rad))
    scale_lo, scale_hi = _range_or_scalar(config.scale_range, float(config.scale))
    clamp_default = float(getattr(config, "handle_clamp_frac", 1.0e9))
    clamp_lo, clamp_hi = _range_or_scalar(config.handle_clamp_frac_range, clamp_default)
    ctx.launch(_style_sample_k, E,
               [seeds_wp, rad_lo, rad_hi, scale_lo, scale_hi, clamp_lo, clamp_hi,
                rad_out, scale_out, clamp_out])
    ctx.sync()


def _range_or_scalar(rng, scalar):
    """Map an optional (lo, hi) range to a (lo, hi) tuple, or (scalar, scalar) if None."""
    if rng is None:
        return float(scalar), float(scalar)
    return float(rng[0]), float(rng[1])


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
                               per_env=per_env)
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

def assemble_inplace(corners_wp, count_wp, config, tan_scratch, scale_scratch, out_wp,
                     rad_per_env=None, clamp_per_env=None,
                     ctx: "LaunchContext | None" = None):
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
    clamp_frac = float(getattr(config, "handle_clamp_frac", 1.0e9))
    dev = str(out_wp.device)
    ctx = _resolve_ctx(ctx, dev)
    ctx.launch(_vertex_tangents_k, E * P,
               [corners_wp, count_wp, P, float(p), tan_scratch, scale_scratch])
    if rad_per_env is None and clamp_per_env is None:
        ctx.launch(_assemble_k, E * P * npseg,
                   [corners_wp, count_wp, tan_scratch, scale_scratch,
                    P, npseg, float(config.rad), clamp_frac, out_wp])
    else:
        ctx.launch(_assemble_style_k, E * P * npseg,
                   [corners_wp, count_wp, tan_scratch, scale_scratch,
                    P, npseg, rad_per_env, clamp_per_env, out_wp])
    ctx.sync()

def generate_centerline_warp(seeds_wp: wp.array, config,
                              out_centerline: wp.array, out_valid_wp: wp.array,
                              scratch, ctx: "LaunchContext | None" = None) -> None:
    """Single-pass centerline generation — in-place owned path only.

    Pure-Warp: sample corners -> sort ccw -> assemble Bezier -> arc-resample -> polygon
    fallback (if self-crossing) -> write chosen centerline into out_centerline.
    Marks all envs valid (generation gate is always True; inflate does the real gate).

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
    E = scratch.gen_count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M = P * npseg  # dense points per env
    arc = ops.ArcScratch(scratch.gen_arc_real, scratch.gen_arc_seg, scratch.gen_arc_s,
                         scratch.gen_arc_cr)

    # Step 0 (opt-in, method #1): per-env style draw. The flag is a Python bool resolved at
    # CUDA-graph capture time, so it selects which kernels the captured graph contains; the
    # per-env style values themselves are device arrays (capturable). When OFF, the style
    # arrays are never touched and the DEFAULT scalar kernels run (byte-for-byte unchanged).
    # gen_style_* are present on the GenScratch group only when allocated for style.
    style = bool(getattr(config, "style_sampling", False))
    scale_pe = rad_pe = clamp_pe = None
    if style:
        style_sample_inplace(seeds_wp, config,
                             scratch.gen_style_rad, scratch.gen_style_scale,
                             scratch.gen_style_clamp, ctx=ctx)
        scale_pe = scratch.gen_style_scale
        rad_pe = scratch.gen_style_rad
        clamp_pe = scratch.gen_style_clamp

    # Step 1-3: sample the corner count and the grid corners, sort the first count[e]
    # corners by angle about their centroid. The sort is ops.angle_sort's kernel with the
    # legacy count floor of 1 (a count < 1 sorts the first corner), so the output does not
    # depend on TrackGenConfig's min_num_points >= 2 check (a config mutated after
    # construction is not re-validated).
    ops.sample_counts(ctx, seeds_wp, CORNER_COUNT_SALT, 0, int(config.min_num_points),
                      int(config.max_num_points), scratch.gen_count)
    num_cells, cell = ops.grid_cells(config.min_point_distance)
    ops.sample_grid_points(ctx, seeds_wp, CORNER_SALT, 0, P, num_cells, float(cell),
                           scale_pe if style else float(config.scale),
                           scratch.gen_used, scratch.gen_corners)
    _angle_sort_launch(ctx, scratch.gen_corners, P, scratch.gen_count, P, scratch.gen_keys,
                       scratch.gen_ordered, m_floor=1)

    # Step 4: assemble Bezier dense -> gen_dense
    assemble_inplace(scratch.gen_ordered, scratch.gen_count, config,
                     scratch.gen_tan, scratch.gen_scale, scratch.gen_dense,
                     rad_per_env=rad_pe, clamp_per_env=clamp_pe, ctx=ctx)

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
    (_Scratch) and passed into generate_centerline_warp; they are NOT part of this class.

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
    gen_style_rad:   [E] float32 — per-env rad draw (method #1; only written/read when
                     config.style_sampling=True, else None/untouched). Optional.
    gen_style_scale: [E] float32 — per-env scale draw (method #1). Optional.
    gen_style_clamp: [E] float32 — per-env handle_clamp_frac draw (method #1). Optional.
    """

    __slots__ = (
        "gen_count", "gen_corners", "gen_ordered", "gen_used", "gen_keys",
        "gen_tan", "gen_scale", "gen_dense", "gen_poly",
        "gen_rs", "gen_crossers",
        "gen_arc_real", "gen_arc_seg", "gen_arc_s", "gen_arc_cr", "gen_arc_co",
        "gen_style_rad", "gen_style_scale", "gen_style_clamp",
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
        gen_style_rad: "wp.array | None" = None,
        gen_style_scale: "wp.array | None" = None,
        gen_style_clamp: "wp.array | None" = None,
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
        self.gen_style_rad = gen_style_rad
        self.gen_style_scale = gen_style_scale
        self.gen_style_clamp = gen_style_clamp


def bezier_alloc_scratch(config):
    """Allocate the bezier generator's PRIVATE working scratch (one alloc per generator).

    The generation OUTPUT buffers (gen_centerline, gen_valid) are owned by the orchestrator
    and passed to generate_centerline_warp; they are not allocated here.
    """
    _init()
    E = int(config.num_envs)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M_dense = P * npseg
    N_gen = int(config.num_points)
    dev = str(config.device)
    # Per-env style buffers (method #1): allocated ONLY when opt-in style sampling is on, so
    # the default path's allocation footprint is unchanged. Each is [E] float32.
    style = bool(getattr(config, "style_sampling", False))
    gen_style_rad = wp.empty(E, dtype=wp.float32, device=dev) if style else None
    gen_style_scale = wp.empty(E, dtype=wp.float32, device=dev) if style else None
    gen_style_clamp = wp.empty(E, dtype=wp.float32, device=dev) if style else None
    return GenScratch(
        gen_count=wp.empty(E, dtype=wp.int32, device=dev),
        gen_corners=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_ordered=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_used=wp.empty(E * P, dtype=wp.int32, device=dev),
        gen_keys=wp.empty(E * P, dtype=wp.float32, device=dev),
        gen_tan=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_scale=wp.empty(E * P, dtype=wp.float32, device=dev),
        gen_dense=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_poly=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_rs=wp.empty(E * N_gen, dtype=wp.vec2f, device=dev),
        gen_crossers=wp.empty(E, dtype=wp.int32, device=dev),
        gen_arc_real=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_arc_seg=wp.empty(E * M_dense, dtype=wp.float32, device=dev),
        gen_arc_s=wp.empty(E * (M_dense + 1), dtype=wp.float32, device=dev),
        gen_arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        gen_arc_co=wp.empty(E, dtype=wp.int32, device=dev),
        gen_style_rad=gen_style_rad,
        gen_style_scale=gen_style_scale,
        gen_style_clamp=gen_style_clamp,
    )


def register(reg) -> None:
    """Register the built-in ``"bezier"`` track generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .generator_registry import GeneratorSpec

    reg.add(GeneratorSpec(
        name="bezier",
        alloc_scratch=bezier_alloc_scratch,
        generate=generate_centerline_warp,
        accepts_ctx=True,
    ))
