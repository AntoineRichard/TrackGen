"""The built-in components (roadmap PR14): one thin adapter per legacy utility.

Each adapter validates its configuration in ``__init__`` (the legacy constructor's
checks, early), builds its legacy utility in :meth:`~track_gen.tools.Component.attach`
(on ``cc.ctx``, with explicit capacities), exposes it as ``adapter.utility`` and its
preallocated result as ``adapter.result``, ends ``attach`` with the cold
``on_generate`` and the module loads, and drives the utility's hot methods with
``ctx=``. The legacy classes keep every name (they ARE the 0.x API).
"""
from __future__ import annotations

import math
import types
import warnings

import warp as wp

from .._src.checkpoints import (CheckpointSampler, CheckpointSet,
                                _readback_max_checkpoints)
from .._src.collision import CollisionChecker
from .._src.collision_discs import DiscChecker
from .._src.collision_frames import FrameChecker
from .._src.course_line import CourseLine
from .._src.heightfield import HeightFieldBaker
from .._src.localize import TrackLocalizer
from .._src.progress import ProgressTracker
from .._src.props import PropSampler
from ..results import (HasBand, HasCenterline, HasCheckpoints, HasGateFrames, HasGates,
                       HasGeneration, HasValidity)
from ..exec import _compat
from ._component import Component, InputSpec, _user_stacklevel

__all__ = ["BandCollision", "CenterlineFromGates", "Checkpoints", "DiscCollision",
           "FrameCollision", "GateCheckpoints", "HeightFieldBake", "Localizer", "Progress",
           "Props"]


def _load_modules(device, *modules: str) -> None:
    """Load (compile) the Warp modules a component's hot methods launch from, so its
    first hot call can be recorded into a capture with no eager call before."""
    for m in modules:
        wp.load_module(m, device=device)


def _resolve_size_hi(size_hi, cc):
    """``size_hi`` as given, or (``"auto"``) the producer's declared bound
    ``cc.bounds["size_hi"]`` (``None`` when it declares none)."""
    if size_hi != "auto":
        return size_hi
    return dict(getattr(cc, "bounds", None) or {}).get("size_hi")


def _boxes(max_boxes: int) -> "types.MappingProxyType":
    """The oriented-box inputs (``[E * max_boxes]``); ``box_position`` defaults to
    ``position`` with one box per env (the agent is the box)."""
    return types.MappingProxyType({
        "box_position": InputSpec(wp.vec3f, max_boxes,
                                  default="position" if max_boxes == 1 else None,
                                  doc="box centers (NaN: an inactive slot)"),
        "orientation": InputSpec(wp.quatf, max_boxes, doc="box orientations (yaw used)"),
        "half_extents": InputSpec(wp.vec2f, max_boxes, doc="planar box half sizes"),
    })


_POSITION = types.MappingProxyType({
    "position": InputSpec(wp.vec3f, doc="one agent position per env (NaN: paused)")})


def _capacity(owner: str, what: str, given, spacing: float, cc) -> int:
    """An explicit capacity (``>= 3``), or the bound ``max(3, ceil(max_loop_length /
    spacing))`` from the producer's plan."""
    if given is not None:
        return int(given)
    L = cc.capacity.get("max_loop_length")
    if L is None:
        raise ValueError(
            f"{owner}: {what}=None needs the producer's capacity 'max_loop_length' "
            f"(attach through ComponentContext.for_producer(generator)), which this "
            f"producer does not declare (an elevation without declared z ranges, a "
            f"variable width, a plugin geometry stage, a gate pipeline, or a hand-built "
            f"source): pass {what} explicitly, e.g. ceil(1.5 * max loop length / spacing)")
    return max(3, math.ceil(float(L) / spacing))


def _check_resolution(owner: str, resolution: int, size_hi, reference_resolution: int,
                      bytes_per_texel: int, E: int) -> None:
    """Warn when a per-env grid of ``resolution`` texels per side loses the accuracy of
    ``reference_resolution`` at size 1 at the largest declared size ``size_hi`` (the
    grid spans each env's bounding box, so its cells grow with the size)."""
    if size_hi is None or not float(size_hi) > 1.0:
        return
    need = math.ceil(int(reference_resolution) * float(size_hi))
    if int(resolution) < need:
        mib = E * need * need * bytes_per_texel / 2 ** 20
        warnings.warn(
            f"{owner}: resolution {resolution} does not keep the accuracy of "
            f"{reference_resolution} at size 1 up to size_hi={size_hi}: the grid spans "
            f"each env's bounding box, so its cells grow with the size; resolution >= "
            f"{need} (~{mib:.0f} MiB for {E} envs) keeps it; pass a lower "
            f"reference_resolution to accept the coarser grid"
            + (", or use method='segments'" if owner == "BandCollision" else ""),
            UserWarning, stacklevel=_user_stacklevel())


def _check_size_hi(size_hi, reference_resolution) -> None:
    if isinstance(size_hi, str):
        if size_hi != "auto":
            raise ValueError(f"size_hi must be > 0, None or 'auto', got {size_hi!r}")
    elif size_hi is not None and not float(size_hi) > 0.0:
        raise ValueError(f"size_hi must be > 0 (or None), got {size_hi!r}")
    if isinstance(reference_resolution, bool) or int(reference_resolution) < 1:
        raise ValueError(f"reference_resolution must be an int >= 1, got "
                         f"{reference_resolution!r}")


# ---------------------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------------------

class BandCollision(Component):
    """Out-of-bounds boxes against the road band (wraps
    :class:`~track_gen.collision.CollisionChecker`).

    Read target: the producer's result (``HasBand``). Inputs: ``box_position``
    (``default="position"`` when ``max_boxes == 1``), ``orientation``, ``half_extents``
    (``[E * max_boxes]``). ``on_generate``: the SDF bake (``method="sdf"``); ``step``:
    ``query`` -> ``BoxContact``. With ``size_hi`` (the largest size a size curriculum
    will reach) an ``"sdf"`` grid below ``ceil(reference_resolution * size_hi)`` texels
    per side warns at attach. ``size_hi="auto"`` (the default, roadmap PR15) reads the
    producer's declared bound (``cc.bounds["size_hi"]``: a size randomizer's
    construction ``hi``, else the construction ``size``); ``None`` declares none.
    """

    requires = (HasBand,)

    def __init__(self, method: str = "segments", *, max_boxes: int = 1,
                 sdf_resolution: int = 128, sdf_padding: "float | None" = None,
                 size_hi: "float | str | None" = "auto", reference_resolution: int = 128,
                 name: str = "collision") -> None:
        if method not in ("segments", "sdf"):
            raise ValueError(f"method must be one of {{'segments', 'sdf'}}, got {method!r}")
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if int(sdf_resolution) < 8:
            raise ValueError(f"sdf_resolution must be >= 8, got {sdf_resolution!r}")
        if sdf_padding is not None and float(sdf_padding) <= 0.0:
            raise ValueError(f"sdf_padding must be > 0 (or None for auto), got "
                             f"{sdf_padding!r}")
        _check_size_hi(size_hi, reference_resolution)
        self.method = method
        self.max_boxes = int(max_boxes)
        self.sdf_resolution = int(sdf_resolution)
        self.sdf_padding = sdf_padding
        self.size_hi = size_hi
        self.reference_resolution = int(reference_resolution)
        self.name = name
        self.inputs = _boxes(self.max_boxes)

    def attach(self, source, cc) -> None:
        if self.method == "sdf":
            _check_resolution("BandCollision", self.sdf_resolution,
                              _resolve_size_hi(self.size_hi, cc),
                              self.reference_resolution, 5, self.layout.num_envs)
        with cc.ctx.scope():   # the sdf constructor's own bake IS the cold on_generate
            self.utility = CollisionChecker(source, self.max_boxes, self.method,
                                            sdf_resolution=self.sdf_resolution,
                                            sdf_padding=self.sdf_padding)
        self.result = self.utility.result
        _load_modules(self.layout.device, "track_gen._src.collision",
                      "track_gen._src.collision_sdf")

    def on_bind(self, box_position, orientation, half_extents) -> None:
        self.utility.bind_inputs(box_position, orientation, half_extents)

    def on_generate(self, ctx, mask) -> None:
        if self.method == "sdf":
            self.utility.bake(ctx=ctx)

    def step(self, ctx):
        return self.utility.query(ctx=ctx)


@wp.kernel
def _gate_posts_k(left: wp.array(dtype=wp.vec3f), right: wp.array(dtype=wp.vec3f),
                  count: wp.array(dtype=wp.int32), G: int,
                  posts: wp.array(dtype=wp.vec3f), post_count: wp.array(dtype=wp.int32)):
    # One thread per gate slot: interleave the left / right posts (NaN padding carries
    # over); slot 0 of each env writes its post count, 2 * min(count, G).
    i = wp.tid()
    posts[2 * i] = left[i]
    posts[2 * i + 1] = right[i]
    e = i // G
    if i - e * G == 0:
        n = count[e]
        if n > G:
            n = G
        if n < 0:
            n = 0
        post_count[e] = 2 * n


class DiscCollision(Component):
    """Boxes against discs (wraps :class:`~track_gen.collision.DiscChecker`).

    Posts mode (``discs=None``): the read target is the producer's gates
    (``HasGateFrames``): ``on_generate`` interleaves their ``left`` / ``right`` posts into
    the owned ``posts`` (``[2 * E * max_gates]``) and writes ``post_count`` (``2 *
    count``); the gates' ``valid`` makes an invalid env hit nothing. Raw mode (``discs=``
    a ``[E * D]`` vec3f array, optional ``count`` / ``valid`` ``[E]`` int32): no read
    target (``requires=()``, ``topology="any"``), the arrays are aliased. Inputs as
    :class:`BandCollision`; ``step``: ``query`` -> ``DiscContact``.
    """

    def __init__(self, radius: float, *, max_boxes: int = 1,
                 discs: "wp.array | None" = None, count: "wp.array | None" = None,
                 valid: "wp.array | None" = None, name: str = "collision") -> None:
        if not (float(radius) > 0.0):
            raise ValueError(f"radius must be > 0, got {radius!r}")
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if discs is None and (count is not None or valid is not None):
            raise ValueError("count= / valid= apply to raw mode (discs=) only: posts mode "
                             "reads the gates' count and valid")
        self.radius = float(radius)
        self.max_boxes = int(max_boxes)
        self.name = name
        self.inputs = _boxes(self.max_boxes)
        self.discs, self.count, self.valid = discs, count, valid
        self.posts: "wp.array | None" = None
        self.post_count: "wp.array | None" = None
        if discs is None:
            self.requires = (HasGateFrames,)
        else:
            self.requires = ()
            self.topology = "any"

    def attach(self, source, cc) -> None:
        lay = self.layout
        E = int(lay.num_envs)
        with cc.ctx.scope():
            if self.discs is None:
                G = int(lay.stride)
                self._G = G
                self._gates = (source.left, source.right, source.count)
                self.posts = wp.zeros(2 * E * G, dtype=wp.vec3f, device=lay.device)
                self.post_count = wp.zeros(E, dtype=wp.int32, device=lay.device)
                self.utility = DiscChecker(self.posts, radius=self.radius,
                                           max_boxes=self.max_boxes, num_envs=E,
                                           count=self.post_count, valid=source.valid)
            else:
                self.utility = DiscChecker(self.discs, radius=self.radius,
                                           max_boxes=self.max_boxes, num_envs=E,
                                           count=self.count, valid=self.valid)
        self.result = self.utility.result
        self.on_generate(cc.ctx, None)
        _load_modules(lay.device, "track_gen._src.collision_discs", __name__)

    def on_bind(self, box_position, orientation, half_extents) -> None:
        self.utility.bind_inputs(box_position, orientation, half_extents)

    def on_generate(self, ctx, mask) -> None:
        if self.posts is None:
            return
        left, right, count = self._gates
        ctx.launch(_gate_posts_k, int(left.shape[0]),
                   [left, right, count, self._G, self.posts, self.post_count])

    def step(self, ctx):
        return self.utility.query(ctx=ctx)


class FrameCollision(Component):
    """A sphere against the gate frames around the current target (wraps
    ``FrameChecker``; ``num_envs`` comes from the layout).

    Read target: the producer's gates (``HasGateFrames``); ``after=(progress,)``: the
    query windows on that component's ``result.next_checkpoint``. Input: ``position``.
    ``step``: ``query`` -> ``FrameContact``.
    """

    requires = (HasGateFrames,)

    def __init__(self, radius: float, frame_thickness: float, frame_depth: float, *,
                 window: int = 4, progress: str = "progress",
                 name: str = "collision") -> None:
        for what, v in (("radius", radius), ("frame_thickness", frame_thickness),
                        ("frame_depth", frame_depth)):
            if not (float(v) > 0.0):
                raise ValueError(f"{what} must be > 0, got {v!r}")
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        self.radius = float(radius)
        self.frame_thickness = float(frame_thickness)
        self.frame_depth = float(frame_depth)
        self.window = int(window)
        self.progress = progress
        self.after = (progress,)
        self.name = name
        self.inputs = _POSITION

    def attach(self, source, cc) -> None:
        with cc.ctx.scope():
            self.utility = FrameChecker(source, int(self.layout.num_envs),
                                        radius=self.radius,
                                        frame_thickness=self.frame_thickness,
                                        frame_depth=self.frame_depth, window=self.window)
        self.utility.bind_window(cc.component(self.progress).result.next_checkpoint)
        self.result = self.utility.result
        _load_modules(self.layout.device, "track_gen._src.collision_frames")

    def on_bind(self, position) -> None:
        self.utility.bind_inputs(position)

    def step(self, ctx):
        return self.utility.query(ctx=ctx)


# ---------------------------------------------------------------------------------------
# Samplers, bakes and lines (per generation)
# ---------------------------------------------------------------------------------------

class Props(Component):
    """Boundary prop poses (wraps :class:`~track_gen.props.PropSampler`).

    Read target: the producer's result (``HasBand``). Capacity: ``max_props``, or
    ``max(3, ceil(max_loop_length / spacing))`` from the producer's plan (``ValueError``
    without it). ``on_generate``: ``sample``; ``step``: the ``PropSet``.
    """

    requires = (HasBand,)

    def __init__(self, spacing: float, *, boundary: str = "outer", mode: str = "points",
                 max_props: "int | None" = None, name: str = "props") -> None:
        if not (float(spacing) > 0.0):
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if boundary not in ("inner", "outer"):
            raise ValueError(f"boundary must be one of {{'inner', 'outer'}}, got "
                             f"{boundary!r}")
        if mode not in ("points", "segments"):
            raise ValueError(f"mode must be one of {{'points', 'segments'}}, got {mode!r}")
        if max_props is not None and int(max_props) < 3:
            raise ValueError(f"max_props must be >= 3 (or None for the plan bound), got "
                             f"{max_props!r}")
        self.spacing = float(spacing)
        self.boundary = boundary
        self.mode = mode
        self.max_props = max_props
        self.name = name

    def attach(self, source, cc) -> None:
        M = _capacity("Props", "max_props", self.max_props, self.spacing, cc)
        with cc.ctx.scope():
            self.utility = PropSampler(source, self.spacing, boundary=self.boundary,
                                       mode=self.mode, max_props=M)
        self.result = self.utility.result
        self.on_generate(cc.ctx, None)

    def on_generate(self, ctx, mask) -> None:
        self.utility.sample(ctx=ctx)


#: :attr:`Checkpoints.defaulted` values: how a capacity the component chose was found.
_DEFAULTED_PLAN = "the plan bound ceil(max_loop_length / checkpoint_spacing)"
_DEFAULTED_READBACK = "read back from the construction batch"


class Checkpoints(Component):
    """Track checkpoints (wraps :class:`~track_gen.checkpoints.CheckpointSampler`).

    Read target: the producer's result (``HasBand``; ``generation`` read when present).
    Capacity: ``max_checkpoints`` (an int ``>= 3``); ``None``: ``max(3,
    ceil(max_loop_length / spacing))`` from the producer's plan (``ValueError`` without
    it); ``"auto"`` (roadmap PR15): that plan bound, else, when the plan declares none
    (an elevated config, a variable width, a plugin geometry stage), the readback of the
    batch it attaches to (``ceil(1.5 * max valid centerline perimeter / spacing)``, one
    cold host read; the ``CheckpointSampler(max_checkpoints=None)`` default).
    :attr:`defaulted` says how a capacity it chose itself was found (``None`` for an
    explicit int); the sampler's ``max_checkpoints`` is the capacity used.
    ``on_generate``: ``sample`` (the set's ``valid`` / ``generation`` are written with
    its rows); ``step``: the ``CheckpointSet``.
    """

    requires = (HasBand,)
    optional = (HasGeneration,)

    def __init__(self, spacing: float, *, max_checkpoints: "int | str | None" = None,
                 name: str = "checkpoints") -> None:
        if not (float(spacing) > 0.0):
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if isinstance(max_checkpoints, str):
            if max_checkpoints != "auto":
                raise ValueError(f"max_checkpoints must be an int >= 3, None (the plan "
                                 f"bound) or 'auto', got {max_checkpoints!r}")
        elif max_checkpoints is not None and int(max_checkpoints) < 3:
            raise ValueError(f"max_checkpoints must be >= 3 (or None for the plan bound), "
                             f"got {max_checkpoints!r}")
        self.spacing = float(spacing)
        self.max_checkpoints = max_checkpoints
        self.name = name
        #: How the capacity was chosen when the component chose it (``max_checkpoints``
        #: ``None`` / ``"auto"``): ``"the plan bound ..."`` or ``"read back from the
        #: construction batch"``; ``None`` for an explicit capacity (set by ``attach``).
        self.defaulted: "str | None" = None

    def attach(self, source, cc) -> None:
        given = self.max_checkpoints
        auto = isinstance(given, str)
        if auto and cc.capacity.get("max_loop_length") is None:
            lay = self.layout
            if cc.ctx.stream is not None:      # the batch is complete on cc.ctx's stream
                _compat.stream_synchronize(cc.ctx.stream)
            M = _readback_max_checkpoints(source, int(lay.num_envs), int(lay.stride),
                                          self.spacing)
            self.defaulted = _DEFAULTED_READBACK
        else:
            M = _capacity("Checkpoints", "max_checkpoints", None if auto else given,
                          self.spacing, cc)
            self.defaulted = _DEFAULTED_PLAN if auto or given is None else None
        with cc.ctx.scope():
            self.utility = CheckpointSampler(source, self.spacing, max_checkpoints=M)
        self.result = self.utility.result
        self.on_generate(cc.ctx, None)

    def on_generate(self, ctx, mask) -> None:
        self.utility.sample(ctx=ctx)


class GateCheckpoints(Component):
    """The gates as checkpoints (``CheckpointSet.from_gates``: a zero-copy view;
    ``utility`` is ``None``). Read target: the producer's gates (``HasGateFrames``;
    ``generation`` read when present). Every array of its result aliases the gates."""

    requires = (HasGateFrames,)
    optional = (HasGeneration,)

    def __init__(self, *, name: str = "checkpoints") -> None:
        self.name = name

    def attach(self, source, cc) -> None:
        self.result = CheckpointSet.from_gates(source)


class CenterlineFromGates(Component):
    """A 3D centerline through the gates (wraps ``CourseLine``; the roadmap's
    ``centerline_from_gates``): its result is the ``Centerline``.

    Read target: the producer's gates (``HasGates``; ``generation`` read when present,
    copied into the line's own). ``on_generate``: ``refresh``.
    """

    requires = (HasGates,)
    optional = (HasGeneration,)

    def __init__(self, samples_per_gate: int = 8, *, name: str = "centerline") -> None:
        if int(samples_per_gate) < 2:
            raise ValueError(f"samples_per_gate must be >= 2, got {samples_per_gate!r}")
        self.samples_per_gate = int(samples_per_gate)
        self.name = name

    def attach(self, source, cc) -> None:
        with cc.ctx.scope():
            self.utility = CourseLine(source, self.samples_per_gate)
        self.result = self.utility.centerline
        self.on_generate(cc.ctx, None)

    def on_generate(self, ctx, mask) -> None:
        self.utility.refresh(ctx=ctx)


class HeightFieldBake(Component):
    """A per-env road heightfield (wraps :class:`~track_gen.heightfield.HeightFieldBaker`).

    Read target: the producer's result (``HasBand``). ``on_generate``: ``bake``. With
    ``size_hi`` a grid below ``ceil(reference_resolution * size_hi)`` texels per side
    warns at attach; ``size_hi="auto"`` (the default, roadmap PR15) reads the
    producer's declared bound (``cc.bounds["size_hi"]``), ``None`` declares none.
    """

    requires = (HasBand,)

    def __init__(self, resolution: int, *, padding: "float | None" = None,
                 size_hi: "float | str | None" = "auto", reference_resolution: int = 128,
                 name: str = "heightfield") -> None:
        if int(resolution) < 8:
            raise ValueError(f"resolution must be >= 8, got {resolution!r}")
        if padding is not None and not (float(padding) > 0.0):
            raise ValueError(f"padding must be > 0 (or None for auto), got {padding!r}")
        _check_size_hi(size_hi, reference_resolution)
        self.resolution = int(resolution)
        self.padding = padding
        self.size_hi = size_hi
        self.reference_resolution = int(reference_resolution)
        self.name = name

    def attach(self, source, cc) -> None:
        _check_resolution("HeightFieldBake", self.resolution,
                          _resolve_size_hi(self.size_hi, cc),
                          self.reference_resolution, 4, self.layout.num_envs)
        with cc.ctx.scope():
            self.utility = HeightFieldBaker(source, self.resolution, self.padding)
        self.result = self.utility.result
        self.on_generate(cc.ctx, None)

    def on_generate(self, ctx, mask) -> None:
        self.utility.bake(ctx=ctx)


# ---------------------------------------------------------------------------------------
# Stateful per-step components
# ---------------------------------------------------------------------------------------

class Progress(Component):
    """Ordered progress through checkpoints (wraps
    :class:`~track_gen.progress.ProgressTracker`; stateful).

    Read target: the ``result`` of the component named ``source`` (default
    ``"checkpoints"``; ``checkpoints=`` is an alias kept for 0.x) (``HasCheckpoints``;
    its ``valid`` / ``generation`` read when present: invalid envs frozen, self-reset on a
    new generation). Input: ``position``. ``step``: ``update`` -> ``ProgressEvents``;
    ``reset``: clears the masked envs' state.
    """

    stateful = True
    requires = (HasCheckpoints,)
    optional = (HasValidity, HasGeneration)

    def __init__(self, *, source: "str | None" = None, checkpoints: "str | None" = None,
                 name: str = "progress") -> None:
        if source is not None and checkpoints is not None and source != checkpoints:
            raise TypeError(f"Progress: source={source!r} and its alias "
                            f"checkpoints={checkpoints!r} disagree: pass source= only")
        self.source = source if source is not None else \
            (checkpoints if checkpoints is not None else "checkpoints")
        self.name = name
        self.inputs = _POSITION

    def attach(self, source, cc) -> None:
        with cc.ctx.scope():   # the self-reset state copies the set's generation on cc.ctx
            self.utility = ProgressTracker(source)
        self.result = self.utility.result
        _load_modules(self.layout.device, "track_gen._src.progress")

    def on_bind(self, position) -> None:
        self.utility.bind(position)

    def step(self, ctx):
        return self.utility.update(ctx=ctx)

    def reset(self, ctx, mask) -> None:
        self.utility.reset(mask, ctx=ctx)


class Localizer(Component):
    """Track-frame localization (wraps :class:`~track_gen.localize.TrackLocalizer`;
    stateful in both modes: the warm memory and the self-reset state are per env).

    Read target: the producer's result, or the ``result`` of the component named
    ``source`` (``HasCenterline``; ``valid`` / ``generation`` read when present).
    Input: ``position``. ``step``: ``query`` -> ``TrackFrame``; ``reset``: drops the
    masked envs' warm memory.
    """

    stateful = True
    requires = (HasCenterline,)
    optional = (HasValidity, HasGeneration)

    def __init__(self, *, warm_window: "int | None" = None, source: "str | None" = None,
                 name: str = "localizer") -> None:
        if warm_window is not None and int(warm_window) < 1:
            raise ValueError(f"warm_window must be >= 1 (or None for full scans), got "
                             f"{warm_window!r}")
        self.warm_window = None if warm_window is None else int(warm_window)
        self.source = source
        self.name = name
        self.inputs = _POSITION

    def attach(self, source, cc) -> None:
        with cc.ctx.scope():
            self.utility = TrackLocalizer(source, warm_window=self.warm_window)
        self.result = self.utility.result
        _load_modules(self.layout.device, "track_gen._src.localize")

    def on_bind(self, position) -> None:
        self.utility.bind(position)

    def step(self, ctx):
        return self.utility.query(ctx=ctx)

    def reset(self, ctx, mask) -> None:
        self.utility.reset(mask, ctx=ctx)


# ``utility``: the wrapped legacy utility (set by ``attach``; ``None`` for
# ``GateCheckpoints``). An adapter attribute, not part of the component contract.
for _cls in (BandCollision, CenterlineFromGates, Checkpoints, DiscCollision,
             FrameCollision, GateCheckpoints, HeightFieldBake, Localizer, Progress, Props):
    _cls.utility = None
del _cls
