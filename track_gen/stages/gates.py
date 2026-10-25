"""The built-in stages of the gate pipeline (roadmap PR11).

A gate :class:`~track_gen.pipeline.Program` (``BatchSpec.max_gates`` set, e.g.
:func:`track_gen.pipeline.presets.gates`) runs, by stage id::

    generator -> order -> [normalize] -> [carry_*] -> gate_relax -> [gate_arclen]
      -> [elevation] -> lift -> [tangents] -> gate_frames -> validity

Roadmap PR13d: ``normalize.extent`` has Params (its dynamic ``size``),
``gate_frames.default``'s ``gate_width`` is dynamic and ``validity.gate_default`` checks
bar crossings per env (its twins: ``_normalize_extent_to_sized_k``,
``warp_gate._finalize_frame_dyn_k``, ``warp_gate._finalize_validity_bars_k``; the
default outputs are byte-identical).

Each stage wraps the 0.2.0.dev3 ``warp_gate._run_gate_pipeline`` launches verbatim (the
same kernels, dims and scalars; three kernels gained trailing input / fail arrays,
``_finalize_validity_k``'s grade-check wrap is also bounded by ``max_gates`` (the same
arithmetic for every count <= max_gates), and the extent normalization runs the same arithmetic out of place,
``_normalize_extent_to_k``), so the default Program is byte-identical to it. The generator is an ``anchors`` Stage
(``kind="gate_generator"``, ``output="anchors"``: ``anchors.xy`` / ``anchors.count``,
unordered or generator-ordered), or a centerline generator followed by ``stations``.

The ``order``, ``stations`` and ``validity`` stages (:data:`REGISTRY_STAGES`) are
registered in :data:`track_gen.plugin.registry`; ``normalize``, ``carry``,
``gate_relax``, ``gate_arclen``, ``lift.gates``, ``tangents`` and ``gate_frames``
(:data:`CATALOGUE_STAGES`) are built-in only. The ``elevation`` stages are the track
ones (:mod:`track_gen.stages.track`), their ports rewired to ``gates.*`` by the planner.
"""
from __future__ import annotations

import types
from typing import Callable

import warp as wp

from .._src import warp_gate as _wg
from .._src import warp_zprofile as _zprof
from .._src.collision_geom import _frame_quat, _safe_normalize3
from ..ops.constants import SIZE_BOUNDS, TARGET_EXTENT
from ..ops.fill import fill as _ops_fill
from ..ops.geometry import _angle_sort_launch
from ..ops.resample import _pad
from ..plugin.params import NoParams
from ..plugin.stage import CaptureClass, Stage
from . import params as _p

__all__ = ["CATALOGUE_STAGES", "GATE_EXPORTS", "REGISTRY_STAGES"]

#: GateSequence field -> the channels it may come from (the FINAL version of the first
#: one a stage writes is the field's array). ``valid`` is ``validity.gate_default``'s
#: own flag when that stage is also the last writer of ``valid.fail``, else the commit
#: derives it as ``fail_mask == 0``.
GATE_EXPORTS = types.MappingProxyType({
    "position": ("gates.position",),
    "tangent": ("gates.tangent",),
    "forward": ("gates.forward",),
    "orientation": ("gates.orientation",),
    "half_size": ("gates.half_size",),
    "left": ("gates.left",),
    "right": ("gates.right",),
    "count": ("gates.count",),
    "valid": ("gates.valid", "valid.fail"),
    "fail_mask": ("valid.fail",),
})


def _alloc_gate_sequence(layout):
    """The gate pipeline's result :class:`~track_gen.GateSequence`, allocated as
    0.2.0.dev3 did (NaN-filled per-gate fields, zero ``count`` / ``valid`` /
    ``generation``), plus the zeroed ``fail_mask`` (``alloc_gate_sequence``'s).
    ``layout.stride`` is ``max_gates``."""
    E, G, dev = int(layout.num_envs), int(layout.stride), str(layout.device)
    gates = _wg.alloc_gate_sequence(types.SimpleNamespace(num_envs=E, max_gates=G,
                                                          device=dev))
    gates.generation = wp.zeros(E, dtype=wp.int32, device=dev)
    return gates


def _orderings_of(spec, registry) -> "list[str]":
    """The order names a gate generator supports, sorted: a Stage's whitelist
    (``orderings``) or every registered order but its ``excluded_orderings``;
    every registered order except its exclusions."""
    if spec.orderings is not None:
        return sorted(spec.orderings)
    return sorted(set(registry.available("order")) - set(spec.excluded_orderings))


def _supported_orderings(registry, name: str) -> "list[str]":
    """The orders gate generator ``name`` supports (sorted; see :func:`_orderings_of`).
    With only the built-in orders registered: ``['ccw', 'random_pairs']`` for bezier /
    hull, ``['ccw', 'raw']`` for polar / checkpoint / voronoi."""
    return _orderings_of(registry.get("gate_generator", name), registry)


def _preferred(spec) -> dict:
    """A gate generator Stage's ``preferred`` mapping."""
    return dict(spec.preferred)


def _default_order(spec, registry) -> str:
    """The default order of gate generator ``spec`` (one rule for ``presets.gates``,
    ``GateGenConfig.gate_ordering=None``, the conformance kit and a legacy spec in a
    PipelineSpec): its ``preferred["order"]`` when it accepts it, else ``"ccw"`` when it
    accepts it (or accepts nothing), else the first order it accepts (sorted). ``"ccw"``
    for the five built-ins."""
    accepted = _orderings_of(spec, registry)
    preferred = _preferred(spec).get("order")
    if preferred in accepted:
        return preferred
    if "ccw" in accepted or not accepted:
        return "ccw"
    return accepted[0]


class _Builtin(Stage):
    """Base of the built-in gate stages (STATIC, no ``output``)."""

    capture = CaptureClass.STATIC
    output = None


# ---------------------------------------------------------------------------------------
# order: anchors -> gate slots (registry kind; the 0.2.0.dev3 kernels verbatim)
# ---------------------------------------------------------------------------------------

_ORDER_READS = {"anchors": "anchors.xy", "anchor_count": "anchors.count"}
_ORDER_WRITES = {"xy": "gates.xy", "count": "gates.count"}


class _Order(_Builtin):
    kind = "order"
    reads = _ORDER_READS
    writes = _ORDER_WRITES

    def _io(self, b):
        src, acount = b.read("anchors"), b.read("anchor_count")
        dst = b.write("xy", wp.vec2f, per="gates")
        count = b.write("count", wp.int32)
        return src, acount, dst, count, b.stride_of("anchors"), b.stride_of("xy")


class OrderRaw(_Order):
    """``order.raw``: the anchors in their emitted order (index preserving: the only
    order that keeps provided per-anchor attributes attached)."""

    name = "raw"

    def build(self, b, p):
        src, acount, dst, count, A, G = self._io(b)
        n = b.E * G

        def enqueue(ctx):
            ctx.launch(_wg._order_raw_k, n, [src, A, acount, G, dst])
            ctx.copy(count, acount)
        return enqueue


class OrderCcw(_Order):
    """``order.ccw``: an angle sort around the env's anchor centroid (clockwise in a
    y-up frame; the name ``ccw`` is historical)."""

    name = "ccw"

    def build(self, b, p):
        src, acount, dst, count, A, G = self._io(b)
        keys = b.scratch("keys", wp.float32, b.E * G)

        def enqueue(ctx):
            # track_gen.ops angle sort: m_floor=0 -> count <= 0 gives an all-NaN row.
            _angle_sort_launch(ctx, src, A, acount, G, keys, dst, 0)
            ctx.copy(count, acount)
        return enqueue


class OrderRandomPairs(_Order):
    """``order.random_pairs``: consecutive anchor pairs in a seeded random order,
    using the fixed ``order.random_pairs/pairs`` namespace and pair indices."""

    name = "random_pairs"
    reads = {**_ORDER_READS, "seed": "seed"}

    def build(self, b, p):
        src, acount, dst, count, A, G = self._io(b)
        seed = b.read("seed")
        keys = b.scratch("keys", wp.float32, b.E * G)
        E = b.E

        def enqueue(ctx):
            ctx.launch(_wg._order_random_pairs_k, E,
                       [seed, src, A, acount, G, wp.uint32(_wg._ORDER_SALT), keys, dst])
            ctx.copy(count, acount)
        return enqueue


# ---------------------------------------------------------------------------------------
# normalize / carry (catalogue)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _normalize_extent_to_k(
    src: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_extent: float,
    dst: wp.array(dtype=wp.vec2f),
):
    # track_gen.ops.geometry._normalize_extent_count_k out of place (the same per-env
    # bbox, centre and scale arithmetic, NaN padding included): reads src, writes dst.
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
            p = src[base + i]
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
            p = src[base + i]
            dst[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)
        else:
            dst[base + i] = wp.vec2f(wp.nan, wp.nan)


_C_SIZE_LO = wp.constant(wp.float32(SIZE_BOUNDS[0]))
_C_SIZE_HI = wp.constant(wp.float32(SIZE_BOUNDS[1]))


@wp.kernel
def _normalize_extent_to_sized_k(
    src: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_extent: float,
    dst: wp.array(dtype=wp.vec2f),
    size_e: wp.array(dtype=wp.float32),
):
    # Roadmap PR13d: _normalize_extent_to_k to the extent target_extent * size[e] (the
    # per-env size, clamped to SIZE_BOUNDS by comparisons, folded into the finite scalar
    # target before the division; the per-gate arithmetic verbatim).
    e = wp.tid()
    size = size_e[e]
    size = wp.where(size < _C_SIZE_LO, _C_SIZE_LO, size)
    size = wp.where(size > _C_SIZE_HI, _C_SIZE_HI, size)
    target_extent = target_extent * size
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
            p = src[base + i]
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
            p = src[base + i]
            dst[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)
        else:
            dst[base + i] = wp.vec2f(wp.nan, wp.nan)


class NormalizeExtent(_Builtin):
    """``normalize.extent``: centre each env's gates and scale them to the extent
    ``BatchSpec.scale * TARGET_EXTENT * size[e]`` (a rewrite of ``gates.xy``, one
    out-of-place launch); the gate preset adds it after the order of a generator whose
    ``preferred["normalize"]`` is ``"extent"`` (polar, checkpoint, voronoi). Roadmap
    PR13d: its dynamic ``size`` (``normalize.size``, default 1: unchanged bytes) is the
    course size of those gate generators."""

    kind = "normalize"
    name = "extent"
    Params = _p.NormalizeExtentParams
    reads = {"xy": "gates.xy", "count": "gates.count"}
    writes = {"xy": "gates.xy"}

    def build(self, b, p):
        x_in, count = b.read("xy"), b.read("count")
        x_out = b.write("xy", wp.vec2f, per="gates")
        G, E = int(b.layout.stride), b.E
        target = float(b.scale) * TARGET_EXTENT
        size = b.param("size")

        def enqueue(ctx):
            ctx.launch(_normalize_extent_to_sized_k, E,
                       [x_in, G, count, float(target), x_out, size])
        return enqueue


@wp.kernel
def _carry_f32_k(src: wp.array(dtype=wp.float32), A: int,
                 count: wp.array(dtype=wp.int32), G: int, pad: float,
                 dst: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = count[e]
    m = wp.max(0, wp.min(m, wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = pad


@wp.kernel
def _carry_vec3_k(src: wp.array(dtype=wp.vec3f), A: int,
                  count: wp.array(dtype=wp.int32), G: int,
                  dst: wp.array(dtype=wp.vec3f)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = count[e]
    m = wp.max(0, wp.min(m, wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = wp.vec3f(wp.nan, wp.nan, wp.nan)


class _Carry(_Builtin):
    """``carry.<attr>``: a provided per-anchor attribute copied to the gate slots BY
    INDEX (the order must be ``order.raw``); padding ``pad``."""

    kind = "carry"
    _dtype = wp.float32
    _pad = 0.0

    def build(self, b, p):
        src, count = b.read("src"), b.read("count")
        dst = b.write("dst", self._dtype, per="gates")
        A, G = b.stride_of("src"), int(b.layout.stride)
        n = b.E * G
        if self._dtype is wp.vec3f:
            def enqueue(ctx):
                ctx.launch(_carry_vec3_k, n, [src, A, count, G, dst])
        else:
            pad = float(self._pad)

            def enqueue(ctx):
                ctx.launch(_carry_f32_k, n, [src, A, count, G, pad, dst])
        return enqueue


class CarryZ(_Carry):
    """``carry.z``: ``anchors.z`` -> ``gates.z`` (padding 0, like the elevations)."""

    name = "z"
    reads = {"src": "anchors.z", "count": "gates.count"}
    writes = {"dst": "gates.z"}


class CarryTangent(_Carry):
    """``carry.tangent``: ``anchors.tangent`` -> ``gates.tangent`` (padding NaN)."""

    name = "tangent"
    reads = {"src": "anchors.tangent", "count": "gates.count"}
    writes = {"dst": "gates.tangent"}
    _dtype = wp.vec3f


class CarryHalfSize(_Carry):
    """``carry.half_size``: ``anchors.half_size`` -> ``gates.half_size`` (padding
    NaN)."""

    name = "half_size"
    reads = {"src": "anchors.half_size", "count": "gates.count"}
    writes = {"dst": "gates.half_size"}
    _pad = float("nan")


class CarryRoll(_Carry):
    """``carry.roll``: ``anchors.roll`` -> ``gates.roll`` (padding 0)."""

    name = "roll"
    reads = {"src": "anchors.roll", "count": "gates.count"}
    writes = {"dst": "gates.roll"}


#: provided attribute -> its carry stage class.
CARRY_STAGES = {"z": CarryZ, "tangent": CarryTangent, "half_size": CarryHalfSize,
                "roll": CarryRoll}


# ---------------------------------------------------------------------------------------
# gate_relax / gate_arclen / lift / tangents (catalogue)
# ---------------------------------------------------------------------------------------

class GateRelaxSpheres(_Builtin):
    """``gate_relax.spheres``: the sphere relaxation (plan view), a rewrite of
    ``gates.xy`` (the kernel copies its input row first, then relaxes in place)."""

    kind = "gate_relax"
    name = "spheres"
    Params = _p.GateRelaxParams
    reads = {"xy": "gates.xy", "count": "gates.count"}
    writes = {"xy": "gates.xy"}

    def build(self, b, p):
        src, count = b.read("xy"), b.read("count")
        dst = b.write("xy", wp.vec2f, per="gates")
        G, E = int(b.layout.stride), b.E
        distance = float(_wg._gate_center_distance(p))
        iters = int(p.gate_solve_iters)

        def enqueue(ctx):
            ctx.launch(_wg._relax_gate_spheres_k, E, [dst, G, count, distance, iters, src])
        return enqueue


class GateRelaxNone(_Builtin):
    """``gate_relax.none``: no sphere solve (``gate_solve_iters == 0``, or a
    ``gate_radius`` whose float32 ``2 * r`` is 0)."""

    kind = "gate_relax"
    name = "none"

    def build(self, b, p):
        def enqueue(ctx):
            pass
        return enqueue


class GateArclen(_Builtin):
    """``gate_arclen.default``: the plan-view cumulative chord length of the gates and
    the closed-loop perimeter (``warp_zprofile.gate_cum_perim``), for the elevation."""

    kind = "gate_arclen"
    name = "default"
    reads = {"xy": "gates.xy", "count": "gates.count"}
    writes = {"cum": "gates.cum", "perim": "gates.perim"}

    def build(self, b, p):
        xy, count = b.read("xy"), b.read("count")
        cum = b.write("cum", wp.float32, per="gates")
        perim = b.write("perim", wp.float32)
        G = int(b.layout.stride)

        def enqueue(ctx):
            _zprof.gate_cum_perim(xy, count, G, cum, perim, ctx=ctx)
        return enqueue


class LiftGates(_Builtin):
    """``lift.gates``: ``gates.position = (gates.xy, gates.z)``."""

    kind = "lift"
    name = "gates"
    reads = {"xy": "gates.xy", "z": "gates.z"}
    writes = {"position": "gates.position"}

    def build(self, b, p):
        xy, z = b.read("xy"), b.read("z")
        position = b.write("position", wp.vec3f, per="gates")
        n = b.E * int(b.layout.stride)

        def enqueue(ctx):
            ctx.launch(_wg._lift_positions_k, n, [xy, z, position])
        return enqueue


class TangentsCentral(_Builtin):
    """``tangents.central``: raw central-difference tangents of the lifted gates
    (closed loop; normalized by the frames stage)."""

    kind = "tangents"
    name = "central"
    reads = {"position": "gates.position", "count": "gates.count"}
    writes = {"tangent": "gates.tangent"}

    def build(self, b, p):
        position, count = b.read("position"), b.read("count")
        tangent = b.write("tangent", wp.vec3f, per="gates")
        G = int(b.layout.stride)
        n = b.E * G

        def enqueue(ctx):
            ctx.launch(_wg._tangents_from_positions_k, n, [position, tangent, G, count])
        return enqueue


# ---------------------------------------------------------------------------------------
# gate_frames (catalogue)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _finalize_frame_posed_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    forward: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_size: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    align_full: int,
    fallbacks: wp.array(dtype=wp.int32),
    position_in: wp.array(dtype=wp.vec3f),
    tangent_in: wp.array(dtype=wp.vec3f),
    half_size_in: wp.array(dtype=wp.float32),
    roll: wp.array(dtype=wp.float32),
):
    # The frame of _finalize_frame_k, then the provided half-size and a roll about the
    # frame's forward (local x) axis, applied to the orientation and the left axis.
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
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
        if fwd[0] * fwd[0] + fwd[1] * fwd[1] < 1.0e-10:
            fwd = wp.vec3f(1.0, 0.0, 0.0)
        wp.atomic_add(fallbacks, e, 1)
        fell = int(1)
    fwd = _safe_normalize3(fwd)
    q0 = _frame_quat(fwd)
    r = roll[t]
    q = q0 * wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), r)
    la = wp.quat_rotate(q, wp.vec3f(0.0, 1.0, 0.0))
    fw = fwd
    if fell == 0:
        if align_full == 1:
            fw = tan
        elif tan[2] == 0.0:
            fw = tan
    hs = half_size_in[t]
    position[t] = p
    tangent[t] = tan
    forward[t] = fw
    orientation[t] = q
    half_size[t] = hs
    left[t] = p + hs * la
    right[t] = p - hs * la


_FRAME_READS = {"position": "gates.position", "tangent": "gates.tangent",
                "count": "gates.count"}
_FRAME_WRITES = {"position": "gates.position", "tangent": "gates.tangent",
                 "forward": "gates.forward", "orientation": "gates.orientation",
                 "half_size": "gates.half_size", "left": "gates.left",
                 "right": "gates.right"}


class _Frames(_Builtin):
    kind = "gate_frames"
    reads = _FRAME_READS
    writes = _FRAME_WRITES

    def _io(self, b):
        pos_in, tan_in, count = b.read("position"), b.read("tangent"), b.read("count")
        out = [b.write(port, None, per="gates") for port in
               ("position", "tangent", "forward", "orientation", "half_size", "left",
                "right")]
        fallbacks = b.scratch("fallbacks", wp.int32, b.E)
        return pos_in, tan_in, count, out, fallbacks


class GateFramesDefault(_Frames):
    """``gate_frames.default``: unit tangent, pose forward, orientation (yaw only or
    full tangent), ``half_size = 0.5 * gate_width`` and the bar endpoints; NaN padding
    (rewrites ``gates.position`` / ``gates.tangent``). Counts frame fallbacks per env
    into the ``fallbacks`` scratch (``GateGenerator.frame_fallbacks``). Roadmap PR13d:
    ``gate_width`` is dynamic (per env, clamped to ``>= 0`` at use:
    ``_finalize_frame_dyn_k``)."""

    name = "default"
    Params = _p.GateFramesParams

    def build(self, b, p):
        pos_in, tan_in, count, out, fallbacks = self._io(b)
        G, E = int(b.layout.stride), b.E
        width = b.param("gate_width")
        align_full = int(p.gate_align == "full_tangent")

        def enqueue(ctx):
            _ops_fill(ctx, fallbacks, 0)
            ctx.launch(_wg._finalize_frame_dyn_k, E * G,
                       [*out, count, G, width, align_full, fallbacks, pos_in, tan_in])
        return enqueue


class GateFramesPosed(_Frames):
    """``gate_frames.posed``: the frames of a generator that provides per-gate
    ``half_size`` and ``roll`` (``gates.half_size`` is rewritten from its carried
    version, the roll turns the frame about its forward axis)."""

    name = "posed"
    Params = _p.GateFramesPosedParams
    reads = {**_FRAME_READS, "half_size": "gates.half_size", "roll": "gates.roll"}

    def build(self, b, p):
        hs_in, roll = b.read("half_size"), b.read("roll")
        pos_in, tan_in, count, out, fallbacks = self._io(b)
        G, E = int(b.layout.stride), b.E
        align_full = int(p.gate_align == "full_tangent")

        def enqueue(ctx):
            _ops_fill(ctx, fallbacks, 0)
            ctx.launch(_finalize_frame_posed_k, E * G,
                       [*out, count, G, align_full, fallbacks, pos_in, tan_in, hs_in,
                        roll])
        return enqueue


# ---------------------------------------------------------------------------------------
# validity.gate_default (registry)
# ---------------------------------------------------------------------------------------

class GateValidityDefault(_Builtin):
    """``validity.gate_default``: the fused per-env gate check (one launch): its 0/1
    flag into ``gates.valid`` and one ``valid.fail`` bit per failed check
    (``GATE_FAIL_BITS``: count, finite, tangent, spacing, crossing, grade). It starts
    ``valid.fail``, so it comes before any plugin check.

    Roadmap PR13d: the bar-crossing check runs per env, for every env with a real gate of
    non-zero ``half_size`` (``_finalize_validity_bars_k``: the bars of a run-time
    ``gate_frames.gate_width`` and posed gates alike), deciding exactly as the PR11
    kernel with the crossing check on (a zero-width bar never crosses)."""

    kind = "validity"
    name = "gate_default"
    Params = _p.GateValidityParams
    reads = {"position": "gates.position", "tangent": "gates.tangent",
             "forward": "gates.forward", "orientation": "gates.orientation",
             "half_size": "gates.half_size", "left": "gates.left", "right": "gates.right",
             "count": "gates.count"}
    writes = {"valid": "gates.valid", "fail": "valid.fail"}

    def build(self, b, p):
        from .channels import GATE_FAIL_BITS  # noqa: PLC0415

        ins = [b.read(port) for port in ("position", "tangent", "forward", "orientation",
                                         "half_size", "left", "right", "count")]
        valid = b.write("valid", wp.int32)
        fail = b.write("fail", wp.int32)
        b._reserve_fail_bits(GATE_FAIL_BITS)
        G, E = int(b.layout.stride), b.E
        min_gates = int(p.min_gates)
        distance = float(_wg._gate_center_distance(p))
        grade = float(p.z_valid_grade)
        # The kernel checks the bars of every env that has one.

        def enqueue(ctx):
            ctx.launch(_wg._finalize_validity_bars_k, E,
                       [*ins, G, min_gates, distance, grade, valid, fail])
        return enqueue


# ---------------------------------------------------------------------------------------
# stations (registry): anchors on a centerline loop
# ---------------------------------------------------------------------------------------

@wp.kernel
def _stations_k(xy: wp.array(dtype=wp.vec2f), stride: int,
                count: wp.array(dtype=wp.int32), use_count: int,
                valid: wp.array(dtype=wp.int32),
                truncated: wp.array(dtype=wp.int32), use_truncated: int, A: int,
                out: wp.array(dtype=wp.vec2f), out_count: wp.array(dtype=wp.int32)):
    # One thread per env: A stations at equal plan-view arc length along the closed
    # loop (station 0 is point 0). An invalid, truncated, degenerate (< 2 points) or
    # zero / non-finite-length loop yields count 0 and a NaN row (fails closed).
    e = wp.tid()
    m = stride
    if use_count != 0:
        m = wp.max(0, wp.min(count[e], stride))
    base = e * stride
    L = float(0.0)
    for i in range(m):
        j = i + 1
        if j == m:
            j = 0
        L = L + wp.length(xy[base + j] - xy[base + i])
    ok = int(1)
    if valid[e] == 0 or m < 2:
        ok = int(0)
    if use_truncated != 0:
        if truncated[e] != 0:
            ok = int(0)
    if not (L > 0.0):
        ok = int(0)
    obase = e * A
    if ok == 0:
        out_count[e] = 0
        for k in range(A):
            out[obase + k] = wp.vec2f(wp.nan, wp.nan)
        return
    seg = int(0)
    seg_start = float(0.0)
    p0 = xy[base]
    p1 = xy[base + 1]
    seg_len = wp.length(p1 - p0)
    for k in range(A):
        s = L * float(k) / float(A)
        while seg < m - 1 and seg_start + seg_len < s:
            seg_start = seg_start + seg_len
            seg = seg + 1
            nxt = seg + 1
            if nxt == m:
                nxt = 0
            p0 = xy[base + seg]
            p1 = xy[base + nxt]
            seg_len = wp.length(p1 - p0)
        u = float(0.0)
        if seg_len > 0.0:
            u = wp.clamp((s - seg_start) / seg_len, 0.0, 1.0)
        out[obase + k] = p0 + u * (p1 - p0)
    out_count[e] = A


@wp.func
def _seg3(p0: wp.vec2f, p1: wp.vec2f, z0: float, z1: float) -> float:
    d = p1 - p0
    dz = z1 - z0
    return wp.sqrt(d[0] * d[0] + d[1] * d[1] + dz * dz)


@wp.kernel
def _stations_ex_k(xy: wp.array(dtype=wp.vec2f), z: wp.array(dtype=wp.float32),
                   use_z: int, stride: int, count: wp.array(dtype=wp.int32),
                   use_count: int, valid: wp.array(dtype=wp.int32),
                   truncated: wp.array(dtype=wp.int32), use_truncated: int, A: int,
                   out: wp.array(dtype=wp.vec2f), out_count: wp.array(dtype=wp.int32),
                   i0: wp.array(dtype=wp.int32), i1: wp.array(dtype=wp.int32),
                   frac: wp.array(dtype=wp.float32)):
    # _stations_k (roadmap PR13a) that also writes the station tables (the GLOBAL source
    # indices of the segment ends and the fraction along it) for the carries. use_z == 0
    # is _stations_k's plan-view code verbatim (byte-equal anchors); use_z != 0 spaces
    # the stations in 3D arc length (segment length sqrt(dx^2 + dy^2 + dz^2), the
    # perimeter, station and segment positions in float64). Every exit path writes the
    # tables: the ok == 0 path the sentinels (i0 = i1 = -1, frac = NaN).
    e = wp.tid()
    m = stride
    if use_count != 0:
        m = wp.max(0, wp.min(count[e], stride))
    base = e * stride
    L = float(0.0)
    L64 = wp.float64(0.0)
    for i in range(m):
        j = i + 1
        if j == m:
            j = 0
        if use_z == 0:
            L = L + wp.length(xy[base + j] - xy[base + i])
        else:
            L64 = L64 + wp.float64(_seg3(xy[base + i], xy[base + j], z[base + i],
                                         z[base + j]))
    if use_z != 0:
        L = wp.float32(L64)
    ok = int(1)
    if valid[e] == 0 or m < 2:
        ok = int(0)
    if use_truncated != 0:
        if truncated[e] != 0:
            ok = int(0)
    if not (L > 0.0):
        ok = int(0)
    obase = e * A
    if ok == 0:
        out_count[e] = 0
        for k in range(A):
            out[obase + k] = wp.vec2f(wp.nan, wp.nan)
            i0[obase + k] = -1
            i1[obase + k] = -1
            frac[obase + k] = wp.nan
        return
    seg = int(0)
    nxt = int(1)
    if nxt == m:
        nxt = 0
    p0 = xy[base]
    p1 = xy[base + nxt]
    if use_z == 0:
        seg_start = float(0.0)
        seg_len = wp.length(p1 - p0)
        for k in range(A):
            s = L * float(k) / float(A)
            while seg < m - 1 and seg_start + seg_len < s:
                seg_start = seg_start + seg_len
                seg = seg + 1
                nxt = seg + 1
                if nxt == m:
                    nxt = 0
                p0 = xy[base + seg]
                p1 = xy[base + nxt]
                seg_len = wp.length(p1 - p0)
            u = float(0.0)
            if seg_len > 0.0:
                u = wp.clamp((s - seg_start) / seg_len, 0.0, 1.0)
            out[obase + k] = p0 + u * (p1 - p0)
            i0[obase + k] = base + seg
            i1[obase + k] = base + nxt
            frac[obase + k] = u
    else:
        ss = wp.float64(0.0)
        sl = wp.float64(_seg3(p0, p1, z[base], z[base + nxt]))
        for k in range(A):
            s64 = L64 * wp.float64(k) / wp.float64(A)
            while seg < m - 1 and ss + sl < s64:
                ss = ss + sl
                seg = seg + 1
                nxt = seg + 1
                if nxt == m:
                    nxt = 0
                p0 = xy[base + seg]
                p1 = xy[base + nxt]
                sl = wp.float64(_seg3(p0, p1, z[base + seg], z[base + nxt]))
            u = float(0.0)
            if sl > wp.float64(0.0):
                u = wp.float32(wp.clamp((s64 - ss) / sl, wp.float64(0.0),
                                        wp.float64(1.0)))
            out[obase + k] = p0 + u * (p1 - p0)
            i0[obase + k] = base + seg
            i1[obase + k] = base + nxt
            frac[obase + k] = u
    out_count[e] = A


class _Stations(_Builtin):
    kind = "stations"
    Params = _p.StationsParams
    writes = {"anchors": "anchors.xy", "anchor_count": "anchors.count"}
    #: (source domain, destination, which families): carried to anchors only when a
    #: later stage reads them (carry.z, carry.attributes, an explicit read).
    _carries = (None, "anchors", "read")

    def capacity(self, params, layout):
        return {"gates": int(params.gates), "min_gates": 0}

    def _build(self, b, p, *, xy, count, use_count, valid, truncated, use_truncated):
        from ..stages.track import _carry_ops  # noqa: PLC0415

        out = b.write("anchors", wp.vec2f, per="anchors")
        out_count = b.write("anchor_count", wp.int32)
        stride, A, E = b.stride_of("xy"), int(p.gates), b.E
        xyz = str(p.spacing_metric) == "xyz"
        carried = b._carried()
        if not carried and not xyz:
            def enqueue(ctx):
                ctx.launch(_stations_k, E, [xy, stride, count, use_count, valid,
                                            truncated, use_truncated, A, out, out_count])
            return enqueue
        src_dom = self._carries[0]
        z = b.read(f"~{src_dom}.z") if xyz else None
        use_z = 1 if xyz else 0
        from ..stages.track import _tables  # noqa: PLC0415

        tables = _tables(b, E * A, int(xy.shape[0]))
        ops = _carry_ops(b, carried, tables, "anchors")

        def enqueue(ctx):
            ctx.launch(_stations_ex_k, E,
                       [xy, z, use_z, stride, count, use_count, valid, truncated,
                        use_truncated, A, out, out_count, tables.i0, tables.i1,
                        tables.frac])
            for op in ops:
                op(ctx)
        return enqueue


class StationsRaw(_Stations):
    """``stations.raw``: ``gates`` anchors at equal arc length on the generator's raw
    loop (``raw.xy``, every point real), in loop order (default order ``raw``). It
    carries the generator's raw attributes that a later stage reads to
    ``anchors.<attr>`` (``linear`` / ``nearest`` / ... through its tables); with
    ``spacing_metric="xyz"`` it spaces them in 3D, which needs ``raw.z``."""

    name = "raw"
    reads = {"xy": "raw.xy", "valid": "gen.valid"}
    _carries = ("raw", "anchors", "read")

    def build(self, b, p):
        xy, valid = b.read("xy"), b.read("valid")
        return self._build(b, p, xy=xy, count=valid, use_count=0, valid=valid,
                           truncated=valid, use_truncated=0)


class StationsRelaxed(_Stations):
    """``stations.relaxed``: ``gates`` anchors at equal arc length on the relaxed
    centerline (``cl.xy`` / ``cl.count``, after the track pipeline's resample and relax
    stages); a truncated loop yields no station. It carries the centerline families a
    later stage reads (``cl.z``, the images of the generator's raw attributes; not
    ``cl.half_width``) to ``anchors.<attr>``; ``spacing_metric="xyz"`` needs ``cl.z``
    (a generator that provides z, or an elevation on the relaxed centerline)."""

    name = "relaxed"
    reads = {"xy": "cl.xy", "count": "cl.count", "valid": "gen.valid",
             "truncated": "cl.truncated"}
    _carries = ("cl", "anchors", "read")

    def build(self, b, p):
        xy, count = b.read("xy"), b.read("count")
        valid, truncated = b.read("valid"), b.read("truncated")
        return self._build(b, p, xy=xy, count=count, use_count=1, valid=valid,
                           truncated=truncated, use_truncated=1)


# carry.attributes: anchors -> gate slots by index (order.raw), padding past the gate count
# (NaN / -1). One CONCRETE kernel per dtype (no generic ``Any`` + ``wp.overload`` kernel:
# Warp 1.18's CPU graph recorder keys a kernel by ``kernel.key``, shared by every overload).
@wp.kernel
def _carry_index_f32_k(src: wp.array(dtype=wp.float32), A: int,
                       count: wp.array(dtype=wp.int32), G: int,
                       dst: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = count[e]
    m = wp.max(0, wp.min(m, wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = _pad(dst[t])


@wp.kernel
def _carry_index_i32_k(src: wp.array(dtype=wp.int32), A: int,
                       count: wp.array(dtype=wp.int32), G: int,
                       dst: wp.array(dtype=wp.int32)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = count[e]
    m = wp.max(0, wp.min(m, wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = _pad(dst[t])


@wp.kernel
def _carry_index_vec2f_k(src: wp.array(dtype=wp.vec2f), A: int,
                         count: wp.array(dtype=wp.int32), G: int,
                         dst: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = count[e]
    m = wp.max(0, wp.min(m, wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = _pad(dst[t])


@wp.kernel
def _carry_index_vec3f_k(src: wp.array(dtype=wp.vec3f), A: int,
                         count: wp.array(dtype=wp.int32), G: int,
                         dst: wp.array(dtype=wp.vec3f)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = count[e]
    m = wp.max(0, wp.min(m, wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = _pad(dst[t])


_CARRY_INDEX_KERNELS = {wp.float32: _carry_index_f32_k, wp.int32: _carry_index_i32_k,
                        wp.vec2f: _carry_index_vec2f_k, wp.vec3f: _carry_index_vec3f_k}


class CarryAttributes(_Builtin):
    """``carry.attributes``: every per-anchor plugin attribute (``anchors.<pkg>.<name>``,
    a gate generator's or carried by the stations) copied to the gate slots BY INDEX
    (``gates.<pkg>.<name>``; the order must be ``order.raw``), padding NaN / -1. One
    launch per attribute (implicit ports)."""

    kind = "carry"
    name = "attributes"
    reads = {"count": "gates.count"}
    _carries = ("anchors", "gates", "index")

    def build(self, b, p):
        count = b.read("count")
        G, E = int(b.layout.stride), b.E
        carries = []
        for c in b._carried():
            src = b.read(c.src_port)
            dst = b.write(c.dst_port, c.dtype, per="gates")
            carries.append((_CARRY_INDEX_KERNELS[src.dtype], src, dst))
        A = b._plan.length("anchors") // E if carries else 0

        def enqueue(ctx):
            for kernel, src, dst in carries:
                ctx.launch(kernel, E * G, [src, A, count, G, dst])
        return enqueue


# The kinds open to plugins: registered in track_gen.plugin.registry.
REGISTRY_STAGES = (OrderRaw, OrderCcw, OrderRandomPairs, StationsRaw, StationsRelaxed,
                   GateValidityDefault)
# The built-in-only kinds: the private catalogue.
CATALOGUE_STAGES = (NormalizeExtent, CarryZ, CarryTangent, CarryHalfSize, CarryRoll,
                    CarryAttributes, GateRelaxSpheres, GateRelaxNone, GateArclen,
                    LiftGates, TangentsCentral, GateFramesDefault, GateFramesPosed)
