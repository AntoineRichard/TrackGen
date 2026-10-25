"""The kernels of the gate pipeline (the stages live in
:mod:`track_gen.stages.gates`).

This module owns the gate kernels the stages launch (ordering, sphere relaxation, lift,
tangents, frames, validity) and a few standalone wrappers: allocation
(:func:`alloc_gate_sequence`), :func:`normalize_positions`, :func:`relax_gate_spheres`,
:func:`tangents_from_positions` and :func:`finalize_gate_sequence` (frames + validity
on a hand-built ``GateSequence``). Native gate generators are ``gate_generator``
Stages in separate modules; ordering, normalization and everything after them are
stages of the gate :class:`~track_gen.pipeline.Program`.

Roadmap PR13d: ``gate_frames.default`` launches :func:`_finalize_frame_dyn_k` (its
``gate_width`` per env) and ``validity.gate_default`` :func:`_finalize_validity_bars_k`
(the bar-crossing check per env, on the envs with a non-zero bar); the PR11 kernels
stay for :func:`finalize_gate_sequence` (one static width) and the frozen references.

Deprecated (until 1.0), for legacy ``GateGeneratorSpec`` plugins written to the 0.1 ABI:
:func:`alloc_order_scratch`, :func:`order_points`, :func:`finish_ordered_gates` and the
``_GateStaging`` name (now the legacy adapter's staging view,
``track_gen.stages.gates._GateStaging``).
"""
from __future__ import annotations


import warp as wp

from ..ops import rng as _rng

from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs as _funcs
from ..ops.fill import fill as _ops_fill
from ..ops.geometry import _normalize_extent_launch  # noqa: F401
from .collision_geom import _frame_quat, _safe_normalize3
# Shared runtime helpers. _sync is re-exported for importers only: library code
# syncs through ctx.sync() (a debug-only no-op unless TRACKGEN_DEBUG_SYNC=1).
from .runtime import _init as _initialize
from .types import GateGenConfig, GateSequence

# The random_pairs order's salt (order.random_pairs).
_ORDER_SALT = 0x977A7D68


# Device helpers and fills now live in track_gen.ops (PR7); the historical names are
# kept as aliases for legacy importers (the validity kernel below still calls
# _proper_segment_intersection). Fill kernels _fill_vec2_k, _fill_vec3_k, _fill_quat_k,
# _fill_f32_k, _fill_i32_k are imported above; this module fills through ops.fill.
_cross2 = _funcs.cross2
_proper_segment_intersection = _funcs.segments_cross


@wp.kernel
def _lift_positions_k(
    pos2: wp.array(dtype=wp.vec2f),
    z: wp.array(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3f),
):
    t = wp.tid()
    p = pos2[t]
    position[t] = wp.vec3f(p[0], p[1], z[t])


@wp.kernel
def _order_raw_k(
    src: wp.array(dtype=wp.vec2f),
    src_stride: int,
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    dst: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // max_gates
    i = t % max_gates
    src_base = e * src_stride
    dst_base = e * max_gates
    m = count[e]
    if m < 0:
        m = 0
    if m > src_stride:
        m = src_stride
    if m > max_gates:
        m = max_gates
    if i < m:
        dst[dst_base + i] = src[src_base + i]
    else:
        dst[dst_base + i] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _order_random_pairs_k(
    seeds: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.vec2f),
    src_stride: int,
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    salt: wp.uint32,
    keys: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.vec2f),
):
    e = wp.tid()
    src_base = e * src_stride
    dst_base = e * max_gates
    m = count[e]
    if m < 0:
        m = 0
    if m > src_stride:
        m = src_stride
    if m > max_gates:
        m = max_gates

    pair_count = (m + 1) // 2
    for pair in range(max_gates):
        if pair < pair_count:
            state = _rng.init_at(seeds[e], salt, pair)
            keys[dst_base + pair] = wp.randf(state) + float(pair) * 1.0e-6

    out_i = int(0)
    for _unit in range(max_gates):
        if _unit < pair_count:
            best_pair = int(0)
            best_key = float(1.0e30)
            for pair in range(max_gates):
                if pair < pair_count:
                    key = keys[dst_base + pair]
                    if key < best_key:
                        best_key = key
                        best_pair = pair

            src_i = best_pair * 2
            dst[dst_base + out_i] = src[src_base + src_i]
            out_i = out_i + 1
            if src_i + 1 < m:
                dst[dst_base + out_i] = src[src_base + src_i + 1]
                out_i = out_i + 1
            keys[dst_base + best_pair] = float(1.0e30)

    for i in range(max_gates):
        if i >= m:
            dst[dst_base + i] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _relax_gate_spheres_k(
    position: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_distance: float,
    iterations: int,
    src: wp.array(dtype=wp.vec2f),
):
    # ``src`` is the input version (gate_relax.spheres rewrites gates.xy into a new
    # buffer): the env's row is copied first, before the early return, so the output
    # version is always the input relaxed (or unchanged). ``src`` may be ``position``.
    e = wp.tid()
    row = e * max_gates
    for k in range(max_gates):
        position[row + k] = src[row + k]
    if target_distance <= 0.0 or iterations <= 0:
        return

    base = e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if cnt > max_gates:
        cnt = max_gates

    eps = float(1.0e-8)
    for it in range(iterations):
        moved = int(0)
        for i in range(max_gates):
            if i < cnt:
                pi = position[base + i]
                for j in range(i + 1, max_gates):
                    if j < cnt:
                        pj = position[base + j]
                        d = pj - pi
                        dist = wp.length(d)
                        if dist + eps < target_distance:
                            n = wp.vec2f(0.0, 0.0)
                            if dist > eps:
                                n = d / dist
                            else:
                                angle = (
                                    float(i + 1) * 12.9898
                                    + float(j + 1) * 78.233
                                    + float(it + 1) * 37.719
                                    + float(e + 1) * 19.371
                                )
                                n = wp.vec2f(wp.cos(angle), wp.sin(angle))
                            correction = 0.5 * (target_distance - dist) * n
                            pi = pi - correction
                            pj = pj + correction
                            position[base + i] = pi
                            position[base + j] = pj
                            moved = int(1)
        if moved == 0:
            break


@wp.kernel
def _tangents_from_positions_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    e = t // max_gates
    i = t % max_gates
    base = e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if cnt > max_gates:
        cnt = max_gates

    if i >= cnt:
        tangent[t] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        return

    if cnt < 2:
        tangent[t] = wp.vec3f(0.0, 0.0, 0.0)
        return

    if cnt == 2:
        p0 = position[base]
        p1 = position[base + 1]
        if i == 0:
            tangent[t] = p1 - p0
        else:
            tangent[t] = p0 - p1
        return

    prev_i = (i + cnt - 1) % cnt
    next_i = (i + 1) % cnt
    tangent[t] = position[base + next_i] - position[base + prev_i]


@wp.kernel
def _finalize_frame_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    forward: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_size: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    gate_width: float,
    align_full: int,
    fallbacks: wp.array(dtype=wp.int32),
    position_in: wp.array(dtype=wp.vec3f),
    tangent_in: wp.array(dtype=wp.vec3f),
):
    # position_in / tangent_in: the input versions (gate_frames rewrites gates.position
    # and gates.tangent into new buffers); they may be position / tangent themselves.
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if i >= cnt or i >= max_gates:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        position[t] = nan3
        tangent[t] = nan3
        forward[t] = nan3
        orientation[t] = wp.quatf(wp.nan, wp.nan, wp.nan, wp.nan)
        half_size[t] = wp.nan
        left[t] = nan3
        right[t] = nan3
        return
    p = position_in[t]
    tan = _safe_normalize3(tangent_in[t])
    fwd = tan
    if align_full == 0:
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
    fell = int(0)
    horiz2 = fwd[0] * fwd[0] + fwd[1] * fwd[1]
    if horiz2 < 1.0e-10:
        # Near-vertical (full_tangent on a steep segment, or degenerate
        # tangent): fall back to the horizontal tangent direction, then +x.
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
        if fwd[0] * fwd[0] + fwd[1] * fwd[1] < 1.0e-10:
            fwd = wp.vec3f(1.0, 0.0, 0.0)
        wp.atomic_add(fallbacks, e, 1)
        fell = int(1)
    fwd = _safe_normalize3(fwd)
    q = _frame_quat(fwd)
    hs = 0.5 * gate_width
    la = wp.quat_rotate(q, wp.vec3f(0.0, 1.0, 0.0))
    if align_full == 0 and tan[2] == 0.0:
        # Planar tangent, yaw-only frame: the left axis is analytic. Using it
        # directly (instead of the quat round-trip, which is only equal to
        # within rounding) keeps left/right bit-identical to the legacy 2D
        # path: left = p + hs * (-tan.y, tan.x, 0). This also reproduces the
        # legacy degenerate-tangent result (tan == 0 -> left == right == p).
        la = wp.vec3f(-tan[1], tan[0], 0.0)
    # Pose forward (physical gate-plane normal). VERBATIM tan — never a
    # re-normalization of it — whenever the pose forward IS the tangent
    # (full_tangent, or planar yaw_only), so progress plane normals stay
    # bit-identical to the tangent on those paths.
    fw = fwd
    if fell == 0:
        if align_full == 1:
            fw = tan
        elif tan[2] == 0.0:
            fw = tan
    position[t] = p
    tangent[t] = tan
    forward[t] = fw
    orientation[t] = q
    half_size[t] = hs
    left[t] = p + hs * la
    right[t] = p - hs * la


_C_WIDTH_LO = wp.constant(wp.float32(0.0))


@wp.kernel
def _finalize_frame_dyn_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    forward: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_size: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    gate_width_e: wp.array(dtype=wp.float32),
    align_full: int,
    fallbacks: wp.array(dtype=wp.int32),
    position_in: wp.array(dtype=wp.vec3f),
    tangent_in: wp.array(dtype=wp.vec3f),
):
    # Roadmap PR13d: _finalize_frame_k with gate_width read per env (gate_frames.default's
    # dynamic bar width, clamped to [0, inf) by a comparison: a NaN passes), then the
    # scalar body verbatim.
    t = wp.tid()
    e = t // max_gates
    gate_width = gate_width_e[e]
    gate_width = wp.where(gate_width < _C_WIDTH_LO, _C_WIDTH_LO, gate_width)
    i = t - e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if i >= cnt or i >= max_gates:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        position[t] = nan3
        tangent[t] = nan3
        forward[t] = nan3
        orientation[t] = wp.quatf(wp.nan, wp.nan, wp.nan, wp.nan)
        half_size[t] = wp.nan
        left[t] = nan3
        right[t] = nan3
        return
    p = position_in[t]
    tan = _safe_normalize3(tangent_in[t])
    fwd = tan
    if align_full == 0:
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
    fell = int(0)
    horiz2 = fwd[0] * fwd[0] + fwd[1] * fwd[1]
    if horiz2 < 1.0e-10:
        # Near-vertical (full_tangent on a steep segment, or degenerate
        # tangent): fall back to the horizontal tangent direction, then +x.
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
        if fwd[0] * fwd[0] + fwd[1] * fwd[1] < 1.0e-10:
            fwd = wp.vec3f(1.0, 0.0, 0.0)
        wp.atomic_add(fallbacks, e, 1)
        fell = int(1)
    fwd = _safe_normalize3(fwd)
    q = _frame_quat(fwd)
    hs = 0.5 * gate_width
    la = wp.quat_rotate(q, wp.vec3f(0.0, 1.0, 0.0))
    if align_full == 0 and tan[2] == 0.0:
        # Planar tangent, yaw-only frame: the left axis is analytic. Using it
        # directly (instead of the quat round-trip, which is only equal to
        # within rounding) keeps left/right bit-identical to the legacy 2D
        # path: left = p + hs * (-tan.y, tan.x, 0). This also reproduces the
        # legacy degenerate-tangent result (tan == 0 -> left == right == p).
        la = wp.vec3f(-tan[1], tan[0], 0.0)
    # Pose forward (physical gate-plane normal). VERBATIM tan — never a
    # re-normalization of it — whenever the pose forward IS the tangent
    # (full_tangent, or planar yaw_only), so progress plane normals stay
    # bit-identical to the tangent on those paths.
    fw = fwd
    if fell == 0:
        if align_full == 1:
            fw = tan
        elif tan[2] == 0.0:
            fw = tan
    position[t] = p
    tangent[t] = tan
    forward[t] = fw
    orientation[t] = q
    half_size[t] = hs
    left[t] = p + hs * la
    right[t] = p - hs * la


@wp.kernel
def _finalize_validity_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    forward: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_size: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    min_gates: int,
    center_distance: float,
    gate_width: float,
    z_valid_grade: float,
    valid: wp.array(dtype=wp.int32),
    fail: wp.array(dtype=wp.int32),
):
    # ``fail``: one bit per failed check (GATE_FAIL_BITS: count 0, finite 1, tangent 2,
    # spacing 3, crossing 4, grade 5), so valid[e] == (fail[e] == 0). ``gate_width`` is
    # read as a toggle only (> 0: run the crossing check).
    e = wp.tid()
    cnt = count[e]
    base = e * max_gates
    ok = int(1)
    f = int(0)

    if cnt < min_gates or cnt > max_gates:
        ok = int(0)
        f = f | 1

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            t = tangent[base + i]
            fw = forward[base + i]
            q = orientation[base + i]
            hs = half_size[base + i]
            li = left[base + i]
            ri = right[base + i]
            fields_finite = (
                wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]) and
                wp.isfinite(t[0]) and wp.isfinite(t[1]) and wp.isfinite(t[2]) and
                wp.isfinite(fw[0]) and wp.isfinite(fw[1]) and wp.isfinite(fw[2]) and
                wp.isfinite(q[0]) and wp.isfinite(q[1]) and
                wp.isfinite(q[2]) and wp.isfinite(q[3]) and
                wp.isfinite(hs) and
                wp.isfinite(li[0]) and wp.isfinite(li[1]) and wp.isfinite(li[2]) and
                wp.isfinite(ri[0]) and wp.isfinite(ri[1]) and wp.isfinite(ri[2])
            )
            if not fields_finite:
                ok = int(0)
                f = f | 2
            # Full 3D tangent norm: with gate_align="full_tangent" a legitimate
            # near-vertical tangent has a tiny XY norm, so an XY-only length
            # check would wrongly flag it. The frame kernel's own near-vertical
            # fallback handles orientation; here we only reject a genuinely
            # degenerate (zero-length) tangent.
            tangent_len2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
            if tangent_len2 <= 1.0e-12:
                ok = int(0)
                f = f | 4

    # CRITICAL for golden parity: pairwise min-distance stays an XY check and
    # the crossing check runs the 2D proper-intersection test on projected
    # endpoints — identical decisions to the legacy vec2f pipeline.
    min_d2 = center_distance * center_distance
    min_d2_slop = 1.0e-6 * wp.max(float(1.0), min_d2)
    for i in range(max_gates):
        if i < cnt:
            pi = position[base + i]
            for j in range(i + 1, max_gates):
                if j < cnt:
                    pj = position[base + j]
                    d = pj - pi
                    d2 = d[0] * d[0] + d[1] * d[1]
                    if d2 + min_d2_slop < min_d2:
                        ok = int(0)
                        f = f | 8

    if gate_width > 0.0:
        for i in range(max_gates):
            if i < cnt:
                li = left[base + i]
                ri = right[base + i]
                for j in range(i + 1, max_gates):
                    if j < cnt:
                        lj = left[base + j]
                        rj = right[base + j]
                        if _proper_segment_intersection(
                                wp.vec2f(li[0], li[1]), wp.vec2f(ri[0], ri[1]),
                                wp.vec2f(lj[0], lj[1]), wp.vec2f(rj[0], rj[1])) != 0:
                            ok = int(0)
                            f = f | 16

    # Grade check: reject any course whose |dz|/ds over a plan-view chord
    # exceeds z_valid_grade. Loops with wraparound (j = (i+1) % cnt) so the
    # closing chord (gate cnt-1 -> gate 0) is checked too. The wrap is also bounded
    # by the slot capacity: a count above max_gates (an over-reporting plugin; the
    # count bit is set above) never reads the next env's row. For cnt <= max_gates
    # this is the same arithmetic as j == cnt.
    if z_valid_grade > 0.0:
        for i in range(max_gates):
            if i < cnt and cnt >= 2:
                j = i + 1
                if j >= cnt or j >= max_gates:
                    j = 0
                pi = position[base + i]
                pj = position[base + j]
                dxy = wp.sqrt((pj[0] - pi[0]) * (pj[0] - pi[0]) +
                              (pj[1] - pi[1]) * (pj[1] - pi[1]))
                if wp.abs(pj[2] - pi[2]) > z_valid_grade * wp.max(dxy, 1.0e-9):
                    ok = int(0)
                    f = f | 32

    valid[e] = ok
    fail[e] = f


@wp.kernel
def _finalize_validity_bars_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    forward: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_size: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    min_gates: int,
    center_distance: float,
    z_valid_grade: float,
    valid: wp.array(dtype=wp.int32),
    fail: wp.array(dtype=wp.int32),
):
    # Roadmap PR13d: _finalize_validity_k without the static crossing toggle (its
    # gate_width argument): the bar-crossing check runs PER ENV, for an env with a real
    # gate of non-zero half_size (a NaN counts: conservative). Exact against the toggle-on
    # kernel: a zero half-size bar is degenerate (left == right == position) and never
    # properly crosses, so skipping the loop changes no decision; at gate_width 0 (every
    # half_size 0) it decides as the toggle-off kernel. Every other check verbatim.
    e = wp.tid()
    cnt = count[e]
    base = e * max_gates
    ok = int(1)
    f = int(0)
    bars = int(0)

    if cnt < min_gates or cnt > max_gates:
        ok = int(0)
        f = f | 1

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            t = tangent[base + i]
            fw = forward[base + i]
            q = orientation[base + i]
            hs = half_size[base + i]
            li = left[base + i]
            ri = right[base + i]
            if hs != 0.0:
                bars = int(1)
            fields_finite = (
                wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]) and
                wp.isfinite(t[0]) and wp.isfinite(t[1]) and wp.isfinite(t[2]) and
                wp.isfinite(fw[0]) and wp.isfinite(fw[1]) and wp.isfinite(fw[2]) and
                wp.isfinite(q[0]) and wp.isfinite(q[1]) and
                wp.isfinite(q[2]) and wp.isfinite(q[3]) and
                wp.isfinite(hs) and
                wp.isfinite(li[0]) and wp.isfinite(li[1]) and wp.isfinite(li[2]) and
                wp.isfinite(ri[0]) and wp.isfinite(ri[1]) and wp.isfinite(ri[2])
            )
            if not fields_finite:
                ok = int(0)
                f = f | 2
            # Full 3D tangent norm: with gate_align="full_tangent" a legitimate
            # near-vertical tangent has a tiny XY norm, so an XY-only length
            # check would wrongly flag it. The frame kernel's own near-vertical
            # fallback handles orientation; here we only reject a genuinely
            # degenerate (zero-length) tangent.
            tangent_len2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
            if tangent_len2 <= 1.0e-12:
                ok = int(0)
                f = f | 4

    # CRITICAL for golden parity: pairwise min-distance stays an XY check and
    # the crossing check runs the 2D proper-intersection test on projected
    # endpoints — identical decisions to the legacy vec2f pipeline.
    min_d2 = center_distance * center_distance
    min_d2_slop = 1.0e-6 * wp.max(float(1.0), min_d2)
    for i in range(max_gates):
        if i < cnt:
            pi = position[base + i]
            for j in range(i + 1, max_gates):
                if j < cnt:
                    pj = position[base + j]
                    d = pj - pi
                    d2 = d[0] * d[0] + d[1] * d[1]
                    if d2 + min_d2_slop < min_d2:
                        ok = int(0)
                        f = f | 8

    if bars != 0:
        for i in range(max_gates):
            if i < cnt:
                li = left[base + i]
                ri = right[base + i]
                for j in range(i + 1, max_gates):
                    if j < cnt:
                        lj = left[base + j]
                        rj = right[base + j]
                        if _proper_segment_intersection(
                                wp.vec2f(li[0], li[1]), wp.vec2f(ri[0], ri[1]),
                                wp.vec2f(lj[0], lj[1]), wp.vec2f(rj[0], rj[1])) != 0:
                            ok = int(0)
                            f = f | 16

    # Grade check: reject any course whose |dz|/ds over a plan-view chord
    # exceeds z_valid_grade. Loops with wraparound (j = (i+1) % cnt) so the
    # closing chord (gate cnt-1 -> gate 0) is checked too. The wrap is also bounded
    # by the slot capacity: a count above max_gates (an over-reporting plugin; the
    # count bit is set above) never reads the next env's row. For cnt <= max_gates
    # this is the same arithmetic as j == cnt.
    if z_valid_grade > 0.0:
        for i in range(max_gates):
            if i < cnt and cnt >= 2:
                j = i + 1
                if j >= cnt or j >= max_gates:
                    j = 0
                pi = position[base + i]
                pj = position[base + j]
                dxy = wp.sqrt((pj[0] - pi[0]) * (pj[0] - pi[0]) +
                              (pj[1] - pi[1]) * (pj[1] - pi[1]))
                if wp.abs(pj[2] - pi[2]) > z_valid_grade * wp.max(dxy, 1.0e-9):
                    ok = int(0)
                    f = f | 32

    valid[e] = ok
    fail[e] = f


def alloc_gate_sequence(config: GateGenConfig) -> GateSequence:
    """Allocate a fixed-stride gate sequence on ``config.device``: NaN per-gate fields,
    zero ``valid`` / ``count`` and a zero ``fail_mask`` (the buffer
    :func:`finalize_gate_sequence` writes the fail bits into by default)."""
    _initialize()
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(config.device)
    flat = E * G
    from .types import Layout

    gates = GateSequence(
        position=wp.empty(flat, dtype=wp.vec3f, device=dev),
        tangent=wp.empty(flat, dtype=wp.vec3f, device=dev),
        forward=wp.empty(flat, dtype=wp.vec3f, device=dev),
        orientation=wp.empty(flat, dtype=wp.quatf, device=dev),
        half_size=wp.empty(flat, dtype=wp.float32, device=dev),
        left=wp.empty(flat, dtype=wp.vec3f, device=dev),
        right=wp.empty(flat, dtype=wp.vec3f, device=dev),
        valid=wp.empty(E, dtype=wp.int32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        generation=wp.zeros(E, dtype=wp.int32, device=dev),
        fail_mask=wp.empty(E, dtype=wp.int32, device=dev),
        layout=Layout(E, G, dev),
    )
    nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
    nan_q = wp.quatf(wp.nan, wp.nan, wp.nan, wp.nan)
    # Construction-time init: still stream-explicit (resolved once for this call).
    ctx = LaunchContext.for_call(dev)
    _ops_fill(ctx, gates.position, nan3)
    _ops_fill(ctx, gates.tangent, nan3)
    _ops_fill(ctx, gates.forward, nan3)
    _ops_fill(ctx, gates.orientation, nan_q)
    _ops_fill(ctx, gates.half_size, wp.nan)
    _ops_fill(ctx, gates.left, nan3)
    _ops_fill(ctx, gates.right, nan3)
    _ops_fill(ctx, gates.valid, 0)
    _ops_fill(ctx, gates.count, 0)
    _ops_fill(ctx, gates.fail_mask, 0)
    ctx.sync()
    return gates


def normalize_positions(
    position: wp.array,
    max_gates: int,
    count: wp.array,
    target_extent: float,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Center and scale each env's real gate positions by bbox extent, then NaN-pad.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _initialize()
    dev = str(position.device)
    ctx = _resolve_ctx(ctx, dev)
    # track_gen.ops.normalize_extent(count=...) (the former _normalize_positions_k).
    _normalize_extent_launch(ctx, position, int(max_gates), float(target_extent),
                             count.shape[0], count=count)
    ctx.sync()


def relax_gate_spheres(
    position: wp.array,
    max_gates: int,
    count: wp.array,
    target_distance: float,
    iterations: int,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Iteratively separate overlapping gate center spheres/disks in-place.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _initialize()
    if float(target_distance) <= 0.0 or int(iterations) <= 0:
        return
    dev = str(position.device)
    ctx = _resolve_ctx(ctx, dev)
    ctx.launch(
        _relax_gate_spheres_k,
        count.shape[0],
        [position, int(max_gates), count, float(target_distance), int(iterations),
         position],
    )
    ctx.sync()


def tangents_from_positions(
    position: wp.array,
    tangent: wp.array,
    max_gates: int,
    count: wp.array,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Write raw central-difference tangents over each env's real count.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _initialize()
    dev = str(tangent.device)
    ctx = _resolve_ctx(ctx, dev)
    G = int(max_gates)
    ctx.launch(
        _tangents_from_positions_k,
        count.shape[0] * G,
        [position, tangent, G, count],
    )
    ctx.sync()


def _gate_center_distance(config) -> float:
    """``2 * gate_radius`` of a ``GateGenConfig`` or a Params with ``gate_radius``."""
    return 2.0 * float(config.gate_radius)


def finalize_gate_sequence(
    gates: GateSequence,
    config: GateGenConfig,
    fallbacks: "wp.array | None" = None,
    ctx: "LaunchContext | None" = None,
    fail: "wp.array | None" = None,
) -> None:
    """Normalize tangents, derive gate frames/endpoints, NaN-pad, and validate (in place
    on a hand-built ``GateSequence``: the ``gate_frames.default`` and
    ``validity.gate_default`` launches).

    ``fallbacks`` is an ``[E]`` int32 scratch counting per-env frame fallbacks
    (near-vertical/degenerate tangents); it is zeroed here each run. ``fail`` receives
    the per-env fail bits (``track_gen.pipeline.GATE_FAIL_BITS``); ``None`` writes them
    into ``gates.fail_mask`` (allocated by :func:`alloc_gate_sequence`). ``None``
    (convenience/test path) allocates a throwaway buffer for ``fallbacks``, and for
    ``fail`` when ``gates.fail_mask`` is ``None`` too — pass a persistent ``fallbacks``
    (and a ``GateSequence`` with a ``fail_mask``, or ``fail``) on capturable paths: the
    call then allocates nothing.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _initialize()
    E = gates.count.shape[0]
    G = int(config.max_gates)
    dev = str(gates.position.device)
    ctx = _resolve_ctx(ctx, dev)
    min_center_distance = _gate_center_distance(config)
    align_full = int(config.gate_align == "full_tangent")
    if fallbacks is None:
        fallbacks = wp.empty(E, dtype=wp.int32, device=dev)
    if fail is None:
        fail = getattr(gates, "fail_mask", None)
    if fail is None:
        fail = wp.empty(E, dtype=wp.int32, device=dev)
    _ops_fill(ctx, fallbacks, 0)
    ctx.launch(
        _finalize_frame_k,
        E * G,
        [
            gates.position,
            gates.tangent,
            gates.forward,
            gates.orientation,
            gates.half_size,
            gates.left,
            gates.right,
            gates.count,
            G,
            float(config.gate_width),
            align_full,  # yaw_only (0) planar frames vs full_tangent (1)
            fallbacks,
            gates.position,
            gates.tangent,
        ],
    )
    ctx.launch(
        _finalize_validity_k,
        E,
        [
            gates.position,
            gates.tangent,
            gates.forward,
            gates.orientation,
            gates.half_size,
            gates.left,
            gates.right,
            gates.count,
            G,
            int(config.min_gates),
            float(min_center_distance),
            float(config.gate_width),
            float(config.z_valid_grade),
            gates.valid,
            fail,
        ],
    )
    ctx.sync()
