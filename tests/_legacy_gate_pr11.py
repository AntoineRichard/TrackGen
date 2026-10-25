"""Independent numeric composition of the PR11 gate pipeline, using current RNG streams.

PR11 decomposed ``warp_gate._run_gate_pipeline`` into the stages of a gate
:class:`~track_gen.pipeline.Program` and deleted the Python glue it ran. This module keeps
a copy of ONLY the HEAD code PR11 edits or deletes:

- the three kernels PR11 extends with trailing arrays (``_relax_gate_spheres_k``,
  ``_finalize_frame_k``, ``_finalize_validity_k``), copied verbatim;
- the glue PR11 deletes: the five built-in ``generate_*_gates`` bodies (and their scratch
  sizes), ``finish_ordered_gates`` / ``order_points``, ``_gate_warp_alloc``,
  ``_run_gate_pipeline`` and the ``finalize_gate_sequence`` launch sequence.

Every kernel and helper PR11 leaves unchanged is imported LIVE (``_order_raw_k``,
``_order_random_pairs_k``, ``_lift_positions_k``, ``_tangents_from_positions_k``,
``_safe_normalize3``, ``_frame_quat``, ``warp_zprofile``, ``track_gen.ops`` and the
samplers). A later drift in one of them is caught by ``test_reference_is_frozen`` (the
digests of :mod:`tests.tools.capture_pr11_gate_stages`, captured from the live dd181f0
path) and by the PR7 fingerprints.

:func:`run_reference` returns the array at every stage boundary under the channel name
the Program gives it (the taps of the capture tool).
"""
from __future__ import annotations

import types

import numpy as np
import warp as wp

from track_gen import ops
from tests._reference_generators import warp_zprofile
from track_gen._src.collision_geom import _frame_quat, _safe_normalize3
from track_gen._src.warp_gate import (
    _lift_positions_k,
    _order_random_pairs_k,
    _order_raw_k,
    _tangents_from_positions_k,
)
from track_gen.exec.context import LaunchContext
from track_gen.ops import funcs as _funcs
from track_gen.ops.constants import TARGET_EXTENT
from track_gen.ops.geometry import _angle_sort_launch, _normalize_extent_launch

_ORDER_SALT = 0x977A7D68
_proper_segment_intersection = _funcs.segments_cross

FINAL_FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left",
                "right", "valid", "count")


# ---------------------------------------------------------------------------------------
# The three kernels PR11 edits (verbatim dd181f0)
# ---------------------------------------------------------------------------------------

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
        la = wp.vec3f(-tan[1], tan[0], 0.0)
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
            tangent_len2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
            if tangent_len2 <= 1.0e-12:
                ok = int(0)

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


# ---------------------------------------------------------------------------------------
# The deleted glue (dd181f0), with taps
# ---------------------------------------------------------------------------------------

class _Taps(dict):
    def take(self, name: str, arr) -> None:
        self[name] = arr.numpy().copy()


def _anchors(ctx, cfg, seeds):
    """The five ``generate_*_gates`` bodies up to ``finish_ordered_gates``: returns
    ``(src, src_stride, count, normalize_extent)``."""
    from track_gen._src import warp_generate, warp_generate_hull as hull
    from tests._reference_generators.warp_generate_checkpoint import sample_checkpoint_rings
    from tests._reference_generators.warp_generate_polar import (polar_jitters, polar_num_knots,
                                                    sample_polar_controls)
    from tests._reference_generators.warp_generate_voronoi import (BOX_SIZE, sample_voronoi_sites,
                                                      select_voronoi_anchors,
                                                      voronoi_layout_mode)

    E, dev, name = int(cfg.num_envs), str(cfg.device), cfg.generator
    if name in ("bezier", "hull"):
        P = int(cfg.max_num_points)
        count = wp.zeros(E, dtype=wp.int32, device=dev)
        points = wp.zeros(E * P, dtype=wp.vec2f, device=dev)
        used = wp.zeros(E * P, dtype=wp.int32, device=dev)
        num_cells, cell_size = ops.grid_cells(cfg.min_point_distance)
        if name == "bezier":
            ops.sample_counts(ctx, seeds, warp_generate.CORNER_COUNT_SALT, 0,
                              int(cfg.min_num_points), int(cfg.max_num_points), count, keying="hashed")
            ops.sample_grid_points(ctx, seeds, warp_generate.CORNER_SALT, 0, P, num_cells,
                                   float(cell_size), float(cfg.scale), used, points, keying="hashed")
        else:
            ops.sample_counts(ctx, seeds, hull.POINT_COUNT_SALT, hull.POINT_SALT_OFFSET,
                              int(cfg.min_num_points), int(cfg.max_num_points), count, keying="hashed")
            ops.sample_grid_points(ctx, seeds, hull.POINT_SALT, hull.POINT_SALT_OFFSET, P,
                                   num_cells, float(cell_size), float(cfg.scale), used,
                                   points, keying="hashed")
        return points, P, count, None
    target_extent = float(cfg.scale) * TARGET_EXTENT
    count = wp.zeros(E, dtype=wp.int32, device=dev)
    if name == "polar":
        K = polar_num_knots(cfg)
        controls = wp.zeros(E * K, dtype=wp.vec2f, device=dev)
        radial_jitter, angular_jitter = polar_jitters(cfg)
        ops.fill(ctx, count, K)
        sample_polar_controls(ctx, seeds, K, radial_jitter, angular_jitter, controls)
        return controls, K, count, target_extent
    if name == "checkpoint":
        C = int(cfg.checkpoint_count)
        checkpoints = wp.zeros(E * C, dtype=wp.vec2f, device=dev)
        ops.fill(ctx, count, C)
        sample_checkpoint_rings(ctx, seeds, 1, C, float(cfg.checkpoint_radius_min_frac),
                                float(cfg.checkpoint_angle_jitter), checkpoints)
        return checkpoints, C, count, target_extent
    if name == "voronoi":
        S = int(cfg.voronoi_num_sites)
        K = int(cfg.voronoi_control_points)
        sites = wp.zeros(E * S, dtype=wp.vec2f, device=dev)
        used = wp.zeros(E * S, dtype=wp.int32, device=dev)
        selected = wp.zeros(E * K, dtype=wp.vec2f, device=dev)
        layout_mode = voronoi_layout_mode(getattr(cfg, "voronoi_site_layout", "void_ring"))
        ops.fill(ctx, count, K)
        sample_voronoi_sites(ctx, seeds, S, layout_mode, BOX_SIZE, sites)
        select_voronoi_anchors(ctx, seeds, sites, S, K, layout_mode, BOX_SIZE,
                               float(getattr(cfg, "voronoi_radial_variation", 0.62)),
                               float(getattr(cfg, "voronoi_angular_jitter", 0.08)),
                               selected, used)
        return selected, K, count, target_extent
    raise ValueError(f"no frozen reference for gate generator {name!r}")


def _order_points(ctx, seeds, src, src_stride, count, G, ordering, keys, dst) -> None:
    E = count.shape[0]
    if ordering == "raw":
        ctx.launch(_order_raw_k, E * G, [src, int(src_stride), count, G, dst])
    elif ordering == "ccw":
        _angle_sort_launch(ctx, src, int(src_stride), count, G, keys, dst, 0)
    elif ordering == "random_pairs":
        ctx.launch(_order_random_pairs_k, E,
                   [seeds, src, int(src_stride), count, G, wp.uint32(_ORDER_SALT), keys, dst])
    else:
        raise ValueError(f"unsupported gate ordering {ordering!r}")


def _finish(ctx, taps, cfg, seeds, generate_staging):
    """``_run_gate_pipeline`` after the generator (relax -> z -> lift -> tangents ->
    finalize), on the staging buffers ``(pos2, count)``."""
    E, G, dev = int(cfg.num_envs), int(cfg.max_gates), str(cfg.device)
    pos2, count = generate_staging
    gates = types.SimpleNamespace(
        position=wp.zeros(E * G, dtype=wp.vec3f, device=dev),
        tangent=wp.zeros(E * G, dtype=wp.vec3f, device=dev),
        forward=wp.zeros(E * G, dtype=wp.vec3f, device=dev),
        orientation=wp.zeros(E * G, dtype=wp.quatf, device=dev),
        half_size=wp.zeros(E * G, dtype=wp.float32, device=dev),
        left=wp.zeros(E * G, dtype=wp.vec3f, device=dev),
        right=wp.zeros(E * G, dtype=wp.vec3f, device=dev),
        valid=wp.zeros(E, dtype=wp.int32, device=dev),
        count=count)
    cum, perim, z = warp_zprofile.alloc_z_scratch(E, G, dev)
    fallbacks = wp.zeros(E, dtype=wp.int32, device=dev)
    min_center_distance = 2.0 * float(cfg.gate_radius)
    solve_iters = int(getattr(cfg, "gate_solve_iters", 0))
    if solve_iters > 0 and min_center_distance > 0.0:
        ctx.launch(_relax_gate_spheres_k, E,
                   [pos2, G, count, float(min_center_distance), solve_iters])
        taps.take("gates.xy@gate_relax", pos2)
    if cfg.z_profile != "flat":
        warp_zprofile.gate_cum_perim(pos2, count, G, cum, perim, ctx=ctx)
        taps.take("gates.cum", cum)
        taps.take("gates.perim", perim)
    warp_zprofile._apply(cfg, seeds, count, G, cum, perim, z, ctx=ctx)
    taps.take("gates.z", z)
    ctx.launch(_lift_positions_k, int(pos2.shape[0]), [pos2, z, gates.position])
    taps.take("gates.position@lift", gates.position)
    ctx.launch(_tangents_from_positions_k, E * G, [gates.position, gates.tangent, G, count])
    taps.take("gates.tangent@tangents", gates.tangent)
    align_full = int(cfg.gate_align == "full_tangent")
    ops.fill(ctx, fallbacks, 0)
    ctx.launch(_finalize_frame_k, E * G,
               [gates.position, gates.tangent, gates.forward, gates.orientation,
                gates.half_size, gates.left, gates.right, gates.count, G,
                float(cfg.gate_width), align_full, fallbacks])
    ctx.launch(_finalize_validity_k, E,
               [gates.position, gates.tangent, gates.forward, gates.orientation,
                gates.half_size, gates.left, gates.right, gates.count, G,
                int(cfg.min_gates), float(min_center_distance), float(cfg.gate_width),
                float(cfg.z_valid_grade), gates.valid])
    for f in FINAL_FIELDS:
        taps.take(f, getattr(gates, f))
    taps.take("fallbacks", fallbacks)
    return taps


def _seed_array(seeds, dev: str):
    return wp.array(np.asarray(seeds, dtype=np.int32), dtype=wp.int32, device=dev)


def run_reference(cfg, seeds) -> dict:
    """The dd181f0 gate pipeline of a built-in generator for ``cfg`` (a
    ``GateGenConfig`` with flat fields) at the EFFECTIVE per-env ``seeds``: ``tap ->
    array`` at every stage boundary."""
    dev = str(cfg.device)
    ctx = LaunchContext.for_call(dev)
    seeds = _seed_array(seeds, dev)
    E, G = int(cfg.num_envs), int(cfg.max_gates)
    taps = _Taps()
    src, src_stride, count, normalize_extent = _anchors(ctx, cfg, seeds)
    taps.take("anchors.xy", src)
    taps.take("anchors.count", count)
    pos2 = wp.zeros(E * G, dtype=wp.vec2f, device=dev)
    out_count = wp.zeros(E, dtype=wp.int32, device=dev)
    keys = wp.zeros(E * G, dtype=wp.float32, device=dev)
    _order_points(ctx, seeds, src, src_stride, count, G, str(cfg.gate_ordering), keys,
                  pos2)
    taps.take("gates.xy@order", pos2)
    if normalize_extent is not None:
        _normalize_extent_launch(ctx, pos2, G, float(normalize_extent), E, count=count)
        taps.take("gates.xy@normalize", pos2)
    ctx.copy(out_count, count)
    taps.take("gates.count", out_count)
    return dict(_finish(ctx, taps, cfg, seeds, (pos2, out_count)))


def run_reference_legacy(cfg, spec, seeds) -> dict:
    """The frozen ordered-gate toy reference, independent of its modern Stage."""
    dev = str(cfg.device)
    ctx = LaunchContext.for_call(dev)
    seeds = _seed_array(seeds, dev)
    E, G = int(cfg.num_envs), int(cfg.max_gates)
    taps = _Taps()
    pos2 = wp.zeros(E * G, dtype=wp.vec2f, device=dev)
    count = wp.zeros(E, dtype=wp.int32, device=dev)
    staging = types.SimpleNamespace(position=pos2, count=count)
    from tests._plugin_toys import toy_ring_generate

    if spec.name == "toy.ring":
        toy_ring_generate(seeds, cfg, staging, None, ctx=ctx)
    taps.take("gates.xy@generator", pos2)
    out = dict(_finish(ctx, taps, cfg, seeds, (pos2, count)))
    if spec.name == "tests.zero-count":
        out = {k: v for k, v in out.items() if k in FINAL_FIELDS or k == "fallbacks"}
    return out
