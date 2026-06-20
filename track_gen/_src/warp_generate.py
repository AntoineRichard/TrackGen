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

Low-level primitives shared with the rest of the pipeline (``_init``, ``_sync``,
``_fill_i32_k``, ``_arc_resample_inplace``, ``self_intersections_inplace``) are reused
from ``warp_pipeline`` via the ``_pipe`` module reference; they are only ever called
from the Python wrappers here (never inside a Warp kernel at decoration time), so there
is no module-load import cycle.
"""
from __future__ import annotations

import dataclasses
import math

import warp as wp

from . import warp_pipeline as _pipe


@wp.kernel
def _corner_count_sample_k(
    seeds: wp.array(dtype=wp.int32),
    attempt: int,
    min_num: int,
    max_num: int,
    out: wp.array(dtype=wp.int32),
):
    # ACCEPTED RNG REDESIGN (does NOT match the oracle's per-env corner-count draw
    # bit-for-bit; validated by range/reproducibility only). One thread per env e.
    #
    # Seeding: state = wp.rand_init(seeds[e] * 6151 + attempt). The 6151 multiplier is
    # DISTINCT from corner_sample's 9781 so the count stream and the corner-position
    # stream stay decorrelated (different rand_init states for the same (seed, attempt)).
    #
    # Draw: a single uniform randf in [0, 1) maps to an inclusive integer count in
    # [min_num, max_num] via floor(randf * range) where range = max_num - min_num + 1,
    # then clamp to max_num to fold the measure-zero randf == 1.0 edge back in range.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 6151 + attempt)
    span = max_num - min_num + 1
    count = min_num + int(wp.randf(state) * float(span))
    out[e] = wp.min(count, max_num)

@wp.kernel
def _corner_sample_k(
    seeds: wp.array(dtype=wp.int32),
    attempt: int,
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: float,
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # ACCEPTED RNG REDESIGN (does NOT match the oracle _sample_corner_points
    # bit-for-bit; validated by structural properties only). One thread per env e.
    #
    # Seeding: state = wp.rand_init(seeds[e] * 9781 + attempt) -> reproducible per
    # (env, attempt). 9781 is a large odd multiplier so distinct env seeds map to
    # well-separated rand_init states.
    #
    # Draw ORDER per corner c (fixed so a corner's noise is deterministic given its
    # retries): for each duplicate-rejection retry draw ONE cell-selection randf,
    # then once a cell is accepted draw the two noise randfs (nx, ny) in that order.
    #
    # Dedup: a corner's cell is redrawn (up to 8 tries) if it collides with any cell
    # already chosen for an EARLIER corner of this env, preserving the distinct-cell
    # spread the oracle's top-k subset gave. The per-thread chosen-cell history lives
    # in the scratch buffer used[e*P + 0 .. e*P + c] (pre-filled with -1 by the
    # wrapper); after the retry budget we accept whatever cell we have.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * 9781 + attempt)
    base = e * P
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        # Bounded duplicate rejection against earlier corners of this env.
        # dup is an int flag (Warp can't mutate a Python bool in a dynamic loop).
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
        nx = wp.randf(state) - 0.5            # [0,1) -> [-0.5, 0.5)
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * scale,
                                 (y * cell_size + ny) * scale)

@wp.kernel
def _ccw_sort_k(
    points: wp.array(dtype=wp.vec2f),
    P: int,
    count: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Orders this env's FIRST m = count[e] corners ascending by
    # the centroid-relative angle key = atan2(dx, dy) (X FIRST), about the centroid of
    # those m corners; rows [m, P) are written NaN (the pruned tail). m == P sorts all P
    # slots with no NaN tail. The insertion sort reads only slots behind
    # its write frontier, so the uninitialised keys/out scratch is never consumed.
    e = wp.tid()
    base = e * P
    m = count[e]
    if m < 1:
        m = 1
    if m > P:
        m = P

    # Centroid over the first m corners (float64 for precision).
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

@wp.func
def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    # Mirrors geometry.safe_normalize: v / clamp_min(||v||, 1e-8). The wp.max
    # floors finite lengths at 1e-8 and a NaN vector divides to (nan, nan)
    # so NaN propagates to BOTH components (matching oracle behavior on pruned corners).
    return v / wp.max(wp.length(v), 1.0e-8)

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
    u_out = _safe_normalize2(c_next - c_i)
    u_in = _safe_normalize2(c_i - c_prev)
    blended = p * u_out + (1.0 - p) * u_in
    tangents[t] = _safe_normalize2(blended)
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
            u_in = _safe_normalize2(ci - cp)
            u_out = _safe_normalize2(cn - ci)
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

def corner_count_sample_inplace(seeds_wp, attempt, config, out_count):
    """In-place: writes per-env corner counts into out_count ([E] int32 wp.array). Zero alloc."""
    _pipe._init()
    E = out_count.shape[0]
    wp.launch(_corner_count_sample_k, dim=E,
              inputs=[seeds_wp, int(attempt), int(config.min_num_points),
                      int(config.max_num_points), out_count],
              device=str(out_count.device))
    _pipe._sync(out_count.device)

def corner_sample_inplace(seeds_wp, attempt, config, out_corners, used_scratch):
    """In-place: writes [E*P] vec2f into out_corners. used_scratch is [E*P] int32 scratch.
    Zero alloc."""
    _pipe._init()
    E = seeds_wp.shape[0]
    P = int(config.max_num_points)
    num_cells = int(1.0 / (config.min_point_distance * 2))
    nc2 = num_cells * num_cells
    cell_size = config.min_point_distance * 2.0
    dev = str(out_corners.device)
    # Fill scratch with -1 (dedup init)
    wp.launch(_pipe._fill_i32_k, dim=E * P, inputs=[used_scratch, -1], device=dev)
    wp.launch(_corner_sample_k, dim=E,
              inputs=[seeds_wp, int(attempt), num_cells, nc2, float(cell_size),
                      float(config.scale), P, used_scratch, out_corners],
              device=dev)
    _pipe._sync(out_corners.device)

def ccw_sort_inplace(corners_wp, count_wp, keys_scratch, out_wp, P):
    """In-place: writes sorted corners into out_wp ([E*P] vec2f). keys_scratch is [E*P] float32.
    Zero alloc."""
    _pipe._init()
    dev = str(out_wp.device)
    E = count_wp.shape[0]
    wp.launch(_ccw_sort_k, dim=E,
              inputs=[corners_wp, P, count_wp, keys_scratch, out_wp],
              device=dev)
    _pipe._sync(out_wp.device)

def assemble_inplace(corners_wp, count_wp, config, tan_scratch, scale_scratch, out_wp):
    """In-place: writes dense [E*P*npseg] vec2f into out_wp. Zero alloc.
    tan_scratch: [E*P] vec2f; scale_scratch: [E*P] float32."""
    _pipe._init()
    E = count_wp.shape[0]
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    assert npseg >= 2
    p = math.atan(config.edgy) / math.pi + 0.5
    clamp_frac = float(getattr(config, "handle_clamp_frac", 1.0e9))
    dev = str(out_wp.device)
    wp.launch(_vertex_tangents_k, dim=E * P,
              inputs=[corners_wp, count_wp, P, float(p), tan_scratch, scale_scratch],
              device=dev)
    wp.launch(_assemble_k, dim=E * P * npseg,
              inputs=[corners_wp, count_wp, tan_scratch, scale_scratch,
                      P, npseg, float(config.rad), clamp_frac, out_wp],
              device=dev)
    _pipe._sync(out_wp.device)

def generate_centerline_warp(seeds_wp: wp.array, config,
                              out_centerline: wp.array, out_valid_wp: wp.array,
                              scratch: "_Scratch") -> None:
    """Single-pass centerline generation — in-place owned path only.

    Pure-Warp: sample corners -> sort ccw -> assemble Bezier -> arc-resample -> polygon
    fallback (if self-crossing) -> write chosen centerline into out_centerline.
    Marks all envs valid (generation gate is always True; inflate does the real gate).

    Args:
        seeds_wp:      [E] int32 wp.array per-env base seeds.
        config:        TrackGenConfig.
        out_centerline:[E*N] vec2f wp.array — written in-place with the chosen centerline.
        out_valid_wp:  [E] int32 wp.array — filled with 1 (all valid at generation stage).
        scratch:       _Scratch with generation intermediates.
    """
    _pipe._init()

    assert scratch is not None, "generate_centerline_warp requires scratch"
    E = scratch.gen_count.shape[0]
    N = int(config.num_points)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M = P * npseg  # dense points per env
    dev = str(out_centerline.device)

    # Step 1-3: sample corners, sort
    corner_count_sample_inplace(seeds_wp, 0, config, scratch.gen_count)
    corner_sample_inplace(seeds_wp, 0, config, scratch.gen_corners, scratch.gen_used)
    ccw_sort_inplace(scratch.gen_corners, scratch.gen_count, scratch.gen_keys,
                     scratch.gen_ordered, P)

    # Step 4: assemble Bezier dense -> gen_dense
    assemble_inplace(scratch.gen_ordered, scratch.gen_count, config,
                     scratch.gen_tan, scratch.gen_scale, scratch.gen_dense)

    # Step 5: arc-resample Bezier dense -> gen_rs (N points per env)
    _pipe._arc_resample_inplace(scratch.gen_dense, M, N,
                          scratch.gen_arc_real, scratch.gen_arc_seg,
                          scratch.gen_arc_s, scratch.gen_arc_cr,
                          scratch.gen_arc_co, scratch.gen_rs, dev)

    # Step 6: assemble polygon dense -> gen_poly (handle_clamp_frac=0)
    cfg_poly = dataclasses.replace(config, handle_clamp_frac=0.0)
    assemble_inplace(scratch.gen_ordered, scratch.gen_count, cfg_poly,
                     scratch.gen_tan, scratch.gen_scale, scratch.gen_poly)

    # Step 7: arc-resample polygon dense -> out_centerline (temporarily holds rs_poly)
    _pipe._arc_resample_inplace(scratch.gen_poly, M, N,
                          scratch.gen_arc_real, scratch.gen_arc_seg,
                          scratch.gen_arc_s, scratch.gen_arc_cr,
                          scratch.gen_arc_co, out_centerline, dev)

    # Step 8: self-intersections of the Bezier resample -> gen_crossers
    # gen_arc_co was written by _arc_scan_k: R>=2 -> N, R<2 -> 0. Pass directly as count.
    _pipe.self_intersections_inplace(scratch.gen_rs, scratch.gen_arc_co,
                               scratch.gen_crossers, N)

    # Step 9: select: rs if no crossings, rs_poly (= out_centerline temporarily) if crossings.
    # out_centerline aliases rs_poly arg: safe per-thread (each thread reads its own slot
    # then writes it for the crossers>0 branch; reads gen_rs for the else branch).
    wp.launch(_select_vec2_k, dim=E * N,
              inputs=[scratch.gen_rs, out_centerline, scratch.gen_crossers, N, out_centerline],
              device=dev)

    # Step 10: mark all envs valid (gen gate is always True; inflate does the real gate).
    wp.launch(_pipe._fill_i32_k, dim=E, inputs=[out_valid_wp, 1], device=dev)

    _pipe._sync(dev)
