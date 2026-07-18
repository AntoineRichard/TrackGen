"""Common pure-Warp helpers for native gate sequence generation.

This module owns only shared gate-buffer utilities: allocation, deterministic point
ordering, count-aware normalization/tangents, and final frame/validity computation.
Native gate generators plug into these helpers but live in separate modules.
"""
from __future__ import annotations

import warp as wp

from .collision_geom import _frame_quat, _safe_normalize3
from .types import GateGenConfig, GateSequence

_INITED = False
_CAPTURING = False
_ORDER_SALT = 7919


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


@wp.func
def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    return v / wp.max(wp.length(v), 1.0e-8)


@wp.func
def _cross2(a: wp.vec2f, b: wp.vec2f) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _proper_segment_intersection(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f) -> int:
    ab = b - a
    cd = d - c
    ac = c - a
    ad = d - a
    ca = a - c
    cb = b - c
    o1 = _cross2(ab, ac)
    o2 = _cross2(ab, ad)
    o3 = _cross2(cd, ca)
    o4 = _cross2(cd, cb)
    hit_ab = (o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)
    hit_cd = (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    if hit_ab and hit_cd:
        return int(1)
    return int(0)


@wp.kernel
def _fill_vec2_k(arr: wp.array(dtype=wp.vec2f), value: wp.vec2f):
    arr[wp.tid()] = value


@wp.kernel
def _fill_vec3_k(arr: wp.array(dtype=wp.vec3f), value: wp.vec3f):
    arr[wp.tid()] = value


@wp.kernel
def _fill_quat_k(arr: wp.array(dtype=wp.quatf), value: wp.quatf):
    arr[wp.tid()] = value


@wp.kernel
def _fill_f32_k(arr: wp.array(dtype=wp.float32), value: float):
    arr[wp.tid()] = value


@wp.kernel
def _fill_i32_k(arr: wp.array(dtype=wp.int32), value: int):
    arr[wp.tid()] = value


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
def _order_ccw_k(
    src: wp.array(dtype=wp.vec2f),
    src_stride: int,
    count: wp.array(dtype=wp.int32),
    max_gates: int,
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

    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(max_gates):
        if i < m:
            p = src[src_base + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])

    cx = wp.float32(0.0)
    cy = wp.float32(0.0)
    if m > 0:
        cx = wp.float32(sx / wp.float64(m))
        cy = wp.float32(sy / wp.float64(m))

    for i in range(max_gates):
        if i < m:
            p = src[src_base + i]
            key = wp.atan2(p[0] - cx, p[1] - cy)
            j = i - 1
            while j >= 0 and keys[dst_base + j] > key:
                keys[dst_base + j + 1] = keys[dst_base + j]
                dst[dst_base + j + 1] = dst[dst_base + j]
                j = j - 1
            keys[dst_base + j + 1] = key
            dst[dst_base + j + 1] = p
        else:
            dst[dst_base + i] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _order_random_pairs_k(
    seeds: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.vec2f),
    src_stride: int,
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    salt: int,
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
            state = wp.rand_init(seeds[e] * 3187 + salt + pair * 104729)
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
def _normalize_positions_k(
    position: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_extent: float,
):
    e = wp.tid()
    base = e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if cnt > max_gates:
        cnt = max_gates

    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            min_x = wp.min(min_x, p[0])
            max_x = wp.max(max_x, p[0])
            min_y = wp.min(min_y, p[1])
            max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            position[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)
        else:
            position[base + i] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _relax_gate_spheres_k(
    position: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_distance: float,
    iterations: int,
):
    e = wp.tid()
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
):
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
    p = position[t]
    tan = _safe_normalize3(tangent[t])
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
):
    e = wp.tid()
    cnt = count[e]
    base = e * max_gates
    ok = int(1)

    if cnt < min_gates or cnt > max_gates:
        ok = int(0)

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
            # Full 3D tangent norm: with gate_align="full_tangent" a legitimate
            # near-vertical tangent has a tiny XY norm, so an XY-only length
            # check would wrongly flag it. The frame kernel's own near-vertical
            # fallback handles orientation; here we only reject a genuinely
            # degenerate (zero-length) tangent.
            tangent_len2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
            if tangent_len2 <= 1.0e-12:
                ok = int(0)

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

    # Grade check: reject any course whose |dz|/ds over a plan-view chord
    # exceeds z_valid_grade. Loops with wraparound (j = (i+1) % cnt) so the
    # closing chord (gate cnt-1 -> gate 0) is checked too.
    if z_valid_grade > 0.0:
        for i in range(max_gates):
            if i < cnt and cnt >= 2:
                j = i + 1
                if j == cnt:
                    j = 0
                pi = position[base + i]
                pj = position[base + j]
                dxy = wp.sqrt((pj[0] - pi[0]) * (pj[0] - pi[0]) +
                              (pj[1] - pi[1]) * (pj[1] - pi[1]))
                if wp.abs(pj[2] - pi[2]) > z_valid_grade * wp.max(dxy, 1.0e-9):
                    ok = int(0)

    valid[e] = ok


def alloc_order_scratch(config: GateGenConfig):
    """Allocate common per-env count and ordering-key scratch for gate generators."""
    _init()
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(config.device)
    return (
        wp.empty(E, dtype=wp.int32, device=dev),
        wp.empty(E * G, dtype=wp.float32, device=dev),
    )


def alloc_gate_sequence(config: GateGenConfig) -> GateSequence:
    """Allocate a fixed-stride gate sequence on ``config.device``."""
    _init()
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(config.device)
    flat = E * G
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
    )
    nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
    nan_q = wp.quatf(wp.nan, wp.nan, wp.nan, wp.nan)
    wp.launch(_fill_vec3_k, dim=flat, inputs=[gates.position, nan3], device=dev)
    wp.launch(_fill_vec3_k, dim=flat, inputs=[gates.tangent, nan3], device=dev)
    wp.launch(_fill_vec3_k, dim=flat, inputs=[gates.forward, nan3], device=dev)
    wp.launch(_fill_quat_k, dim=flat, inputs=[gates.orientation, nan_q], device=dev)
    wp.launch(_fill_f32_k, dim=flat, inputs=[gates.half_size, wp.nan], device=dev)
    wp.launch(_fill_vec3_k, dim=flat, inputs=[gates.left, nan3], device=dev)
    wp.launch(_fill_vec3_k, dim=flat, inputs=[gates.right, nan3], device=dev)
    wp.launch(_fill_i32_k, dim=E, inputs=[gates.valid, 0], device=dev)
    wp.launch(_fill_i32_k, dim=E, inputs=[gates.count, 0], device=dev)
    _sync(dev)
    return gates


def order_points(
    seeds_wp: wp.array,
    src: wp.array,
    src_stride: int,
    count: wp.array,
    max_gates: int,
    ordering: str,
    keys: wp.array,
    dst: wp.array,
) -> None:
    """Order source points into a fixed gate buffer and NaN-pad inactive slots."""
    _init()
    E = count.shape[0]
    dev = str(dst.device)
    src_stride_i = int(src_stride)
    max_gates_i = int(max_gates)
    if ordering == "raw":
        wp.launch(
            _order_raw_k,
            dim=E * max_gates_i,
            inputs=[src, src_stride_i, count, max_gates_i, dst],
            device=dev,
        )
    elif ordering == "ccw":
        wp.launch(
            _order_ccw_k,
            dim=E,
            inputs=[src, src_stride_i, count, max_gates_i, keys, dst],
            device=dev,
        )
    elif ordering == "random_pairs":
        wp.launch(
            _order_random_pairs_k,
            dim=E,
            inputs=[seeds_wp, src, src_stride_i, count, max_gates_i, int(_ORDER_SALT), keys, dst],
            device=dev,
        )
    else:
        raise ValueError(f"unsupported gate ordering {ordering!r}")
    _sync(dev)


def finish_ordered_gates(
    seeds_wp: wp.array,
    src: wp.array,
    src_stride: int,
    count: wp.array,
    max_gates: int,
    ordering: str,
    keys: wp.array,
    out,
    normalize_extent: float | None = None,
) -> None:
    """Order gates into the 2D staging buffer, optionally normalize, copy count.

    ``out`` is the vec2f staging view handed to generators (``_GateStaging``):
    ``out.position`` is the flat ``[E * max_gates]`` vec2f scratch the 2D
    ordering/normalization kernels operate on. Tangents are no longer computed
    here — the pipeline lifts the staged positions to the public vec3f buffers
    and derives tangents from those (see ``_run_gate_pipeline``).
    """
    G = int(max_gates)
    order_points(seeds_wp, src, int(src_stride), count, G, ordering, keys, out.position)
    if normalize_extent is not None:
        normalize_positions(out.position, G, count, float(normalize_extent))
    wp.copy(out.count, count)


def normalize_positions(
    position: wp.array,
    max_gates: int,
    count: wp.array,
    target_extent: float,
) -> None:
    """Center and scale each env's real gate positions by bbox extent, then NaN-pad."""
    _init()
    dev = str(position.device)
    wp.launch(
        _normalize_positions_k,
        dim=count.shape[0],
        inputs=[position, int(max_gates), count, float(target_extent)],
        device=dev,
    )
    _sync(dev)


def relax_gate_spheres(
    position: wp.array,
    max_gates: int,
    count: wp.array,
    target_distance: float,
    iterations: int,
) -> None:
    """Iteratively separate overlapping gate center spheres/disks in-place."""
    _init()
    if float(target_distance) <= 0.0 or int(iterations) <= 0:
        return
    dev = str(position.device)
    wp.launch(
        _relax_gate_spheres_k,
        dim=count.shape[0],
        inputs=[position, int(max_gates), count, float(target_distance), int(iterations)],
        device=dev,
    )
    _sync(dev)


def tangents_from_positions(
    position: wp.array,
    tangent: wp.array,
    max_gates: int,
    count: wp.array,
) -> None:
    """Write raw central-difference tangents over each env's real count."""
    _init()
    dev = str(tangent.device)
    G = int(max_gates)
    wp.launch(
        _tangents_from_positions_k,
        dim=count.shape[0] * G,
        inputs=[position, tangent, G, count],
        device=dev,
    )
    _sync(dev)


def _gate_center_distance(config: GateGenConfig) -> float:
    return 2.0 * float(config.gate_radius)


def finalize_gate_sequence(
    gates: GateSequence,
    config: GateGenConfig,
    fallbacks: "wp.array | None" = None,
) -> None:
    """Normalize tangents, derive gate frames/endpoints, NaN-pad, and validate.

    ``fallbacks`` is an ``[E]`` int32 scratch counting per-env frame fallbacks
    (near-vertical/degenerate tangents); it is zeroed here each run. ``None``
    (convenience/test path) allocates a throwaway buffer — pass a persistent
    buffer on capturable paths.
    """
    _init()
    E = gates.count.shape[0]
    G = int(config.max_gates)
    dev = str(gates.position.device)
    min_center_distance = _gate_center_distance(config)
    align_full = int(config.gate_align == "full_tangent")
    if fallbacks is None:
        fallbacks = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(_fill_i32_k, dim=E, inputs=[fallbacks, 0], device=dev)
    wp.launch(
        _finalize_frame_k,
        dim=E * G,
        inputs=[
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
        ],
        device=dev,
    )
    wp.launch(
        _finalize_validity_k,
        dim=E,
        inputs=[
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
        ],
        device=dev,
    )
    _sync(dev)


class _GateStaging:
    """Vec2f staging view handed to generators.

    ``position`` aliases the pipeline-owned ``[E * max_gates]`` vec2f scratch
    that the 2D ordering/normalization/relaxation kernels operate on;
    ``count`` aliases the public ``GateSequence.count``. The pipeline lifts
    the staged positions into the public vec3f arrays afterwards.
    """

    __slots__ = ("position", "count")

    def __init__(self, position: wp.array, count: wp.array) -> None:
        self.position = position
        self.count = count


def _gate_warp_alloc(config: GateGenConfig, generator_spec):
    """Allocate facade-owned gate output plus pipeline + generator scratch.

    Returns ``(gates, (gen_scratch, pos2, z_cum, z, fallbacks))``: ``pos2`` is
    the vec2f staging buffer the 2D kernels write, ``z_cum`` the per-gate
    cumulative plan-view arc length scratch the Z profiler needs, ``z`` the
    per-gate elevation profile written by the Z profiler each run, ``fallbacks``
    the per-env frame-fallback counters.
    """
    from . import warp_zprofile

    gates = alloc_gate_sequence(config)
    gen_scratch = generator_spec.alloc_scratch(config)
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(config.device)
    pos2 = wp.empty(E * G, dtype=wp.vec2f, device=dev)
    z_cum, z = warp_zprofile.alloc_z_scratch(config)
    fallbacks = wp.zeros(E, dtype=wp.int32, device=dev)
    return gates, (gen_scratch, pos2, z_cum, z, fallbacks)


def _run_gate_pipeline(
    config: GateGenConfig,
    seed_buf_wp: wp.array,
    out: GateSequence,
    scratch,
    generator_spec,
) -> GateSequence:
    """Run a native gate generator into ``out`` and finalize the common fields.

    The generator and the shared 2D kernels (ordering, normalization, sphere
    relaxation) operate on the vec2f ``pos2`` staging buffer; the staged
    positions are then lifted to the public vec3f ``out.position`` (z from the
    ``z`` scratch — all zeros until a z-profile stage writes it), tangents are
    derived from the lifted positions, and the frame/validity finalization
    runs on the vec3f arrays.
    """
    from . import warp_zprofile

    _init()
    gen_scratch, pos2, z_cum, z, fallbacks = scratch
    G = int(config.max_gates)
    dev = str(out.position.device)
    staging = _GateStaging(pos2, out.count)
    generator_spec.generate(seed_buf_wp, config, staging, gen_scratch)
    min_center_distance = _gate_center_distance(config)
    solve_iters = int(getattr(config, "gate_solve_iters", 0))
    if solve_iters > 0 and min_center_distance > 0.0:
        relax_gate_spheres(pos2, G, out.count, min_center_distance, solve_iters)
    # Z profile runs on the final ordered/relaxed 2D anchors, before the lift.
    # It writes every real slot and zeroes padding, so the z scratch is fully
    # overwritten each run (capture-safe: launches only, no alloc/sync/branch).
    warp_zprofile.apply_z_profile(config, seed_buf_wp, pos2, out.count, z_cum, z)
    wp.launch(
        _lift_positions_k,
        dim=int(pos2.shape[0]),
        inputs=[pos2, z, out.position],
        device=dev,
    )
    tangents_from_positions(out.position, out.tangent, G, out.count)
    finalize_gate_sequence(out, config, fallbacks=fallbacks)
    return out
