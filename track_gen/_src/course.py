"""Course: the composition root of a generator and its runtime components.

A ``Course`` drives one producer (a ``TrackGenerator``, a ``GateGenerator``, a built
``Program``, or any object with the :class:`track_gen.pipeline.Producer` surface) and an
ordered list of components (:mod:`track_gen.tools`: checkpoints, progress, collision,
localization, bakes, your own), and owns the orchestration invariants that are otherwise
the caller's burden (roadmap PR15). Three front doors build one:

- ``Course(generator, components)``: any producer, any components (sorted by their
  ``after`` / ``source`` dependencies);
- the presets ``Course.track(source, ...)`` / ``Course.gates(source, ...)``: the 0.x
  component sets over a config, a ``PipelineSpec`` or a built generator;
- ``Course.from_config(CourseConfig)`` (and the 0.x ``Course(CourseConfig)``): the 0.x
  course, byte for byte and launch for launch:

  - ``mode="track"``: TrackGenerator -> ``Checkpoints`` (``CheckpointSampler``) ->
    ``Progress`` (``ProgressTracker``) -> out-of-bounds ``BandCollision``
    (``CollisionChecker``: ``"segments"`` / ``"sdf"`` / none) -> optional
    ``HeightFieldBake``;
  - ``mode="gates"``: GateGenerator -> ``GateCheckpoints`` (``CheckpointSet.from_gates``)
    -> ``CenterlineFromGates`` (``CourseLine``: the 3D spline centerline through the
    gates) -> ``Progress`` -> optional gate-post ``DiscCollision`` / ``FrameCollision``
    -> ``Localizer`` (``TrackLocalizer`` on the line: ``step()`` returns the per-env
    :class:`TrackFrame`).

Lifecycle: construct (the producer's first batch, every component attached, one warm-up
regeneration) -> ``bind()`` (stable sim buffers, routed to the components that declare
them; required for ``step()`` when one declares a required input) -> per-step
``step()`` / per-env ``reset(mask)``, with ``regenerate()`` / ``generate(seeds=...)``
whenever new courses are wanted. Per-env new courses: ``regenerate(mask=done,
advance=True)`` advances the epoch of the masked envs on device (see
:mod:`track_gen.seeds`), regenerates, refreshes every component (``on_generate`` with the
mask) and resets the masked envs' runtime state; the other envs keep their course and
state. The pipelines are fixed-shape, so every env is recomputed, and the unmasked ones
(which keep the effective seed their course was built from) are recomputed to identical
values.

CUDA graphs: regeneration is ONE body (producer + every component's ``on_generate`` in
execution order + every component's ``reset`` in reverse order: full, or masked for
``regenerate(mask=...)``). Inside a caller's capture ``regenerate()`` / ``generate()``
record it into the caller's graph next to ``step()`` / ``reset()`` (an RL auto-reset loop
``[physics] -> step() -> done mask -> regenerate(mask=done, advance=True)`` replays
entirely on device); outside a capture the course replays its own graph
(:attr:`Course.graph`, one more per ``(advance, mask)`` variant) on CUDA. A masked
regeneration is skipped on device when no env is masked, where CUDA supports it (see
:meth:`Course.regenerate`). ``step()``/``reset()`` are never self-captured: they are
capture-ready for the caller's sim graph as-is (no component host-synchronizes; the
former ``track_gen.set_capturing`` is a deprecated no-op). Every launch of one call goes
to ONE stream: ``stream=`` when given, else an enclosing ``torch.cuda.graph`` capture
stream, else the course's default stream, else Warp's current stream (see
:mod:`track_gen.exec`).

An env with ``valid[e] == 0`` on ``course.result`` reads like a poisoned one in every
built-in component (``oob = 1`` with NaN geometry, no post hits, frozen progress with
``dist_to_next = 0``, NaN localization, ``checkpoints.count = 0``): gate rewards and
observations on :attr:`StepResult.valid` and give such envs a new course
(``regenerate(mask=done, advance=True)``).
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import numbers
import sys
import types
import warnings
import weakref
from dataclasses import dataclass

import warp as wp

from ..exec import _compat
from ..exec.context import (_AMBIENT, _RESOLVED_BLOCKED, LaunchContext, _host_blocked,
                            _rebind_close, _rebind_open, _torch_capturing,
                            resolve as _resolve_ctx)
from ..exec.errors import CaptureError
from ..exec.runner import GraphRunner, _check_default_stream, ensure_not_capturing
from ..ops.reduce import any_set
from .. import tools as _tools
from ..tools._component import _order, _per_step, _route, _user_stacklevel
from .checkpoints import CheckpointSampler, CheckpointSet  # noqa: F401 - 0.x names
from .collision import BoxContact, CollisionChecker
from .collision_discs import DiscChecker, DiscContact
from .collision_frames import FrameChecker, FrameContact
from .course_line import CourseLine
from .gate_generator import GateGenerator
from .heightfield import HeightFieldBaker
from .localize import TrackFrame, TrackLocalizer
from .progress import ProgressEvents, ProgressTracker
from .rng_utils import PerEnvSeededRNG
from .runtime import _check_arr, _init, set_capturing
from .track_generator import TrackGenerator
from .types import GateGenConfig, GateSequence, Track, TrackGenConfig


@wp.kernel
def _seed_range_k(base: wp.int32, out: wp.array(dtype=wp.int32)):
    # out[e] = base + e with int32 wrap-around: PerEnvSeededRNG's int expansion
    # (seed + arange(E), cast to int32). Unsigned add: no signed-overflow UB.
    e = wp.tid()
    out[e] = wp.int32(wp.uint32(base) + wp.uint32(e))


#: The smallest positive float32 (2**-149): Course gates mode's floor of
#: ``gate_frames.gate_width``.
_F32_MIN_POSITIVE = 2.0 ** -149


def _wrap_int32(value: int) -> int:
    """``value`` reduced to the int32 range with two's-complement wrap-around."""
    return ((int(value) + 2**31) % 2**32) - 2**31


@dataclass(frozen=True)
class CourseConfig:
    """Configuration for :class:`Course`. Strict option applicability:
    inapplicable options raise instead of being silently ignored.

    Attributes
    ----------
    mode : str
        ``"track"`` or ``"gates"``.
    gen : TrackGenConfig or GateGenConfig
        Generator config; its type must match ``mode``. ``num_envs`` and
        ``device`` are taken from here. Gates mode requires
        ``gen.gate_width > 0`` (a width-0 gate can never be crossed). The
        width is per env since 0.2.0.dev5 (``gate_frames.gate_width``): a
        scalar ``set`` of a value ``<= 0`` is refused, but an ``[E]`` array
        written after construction is not range-checked (a value ``<= 0`` is
        used as 0): keep it ``> 0``, or those envs' gates can never be
        passed.
    seeds : int or wp.array
        Initial per-env RNG seeding (forwarded to ``PerEnvSeededRNG``).
    collision : str or None
        Track mode only: ``"segments"``, ``"sdf"``, or ``None`` (no
        out-of-bounds checking — progress-only bundles are legal).
    sdf_resolution : int or None
        Track mode with ``collision="sdf"`` only; ``None`` -> 128. The
        facade always uses ``CollisionChecker``'s AUTO SDF padding
        (``sdf_padding=None``, i.e. 10% of each env's larger AABB extent);
        ``sdf_padding`` itself is not exposed here.
    heightfield_resolution : int or None
        Track mode only; when set (``>= 8``), builds a
        :class:`HeightFieldBaker` exposed as ``course.heightfield`` and
        re-baked on every ``generate()`` (nearest-edge road-surface grid for
        external physics solvers). Uses the baker's AUTO padding (10% of each
        env's larger AABB extent). ``None`` (default) means no heightfield.
    post_radius : float
        Gates mode only: > 0 enables ``DiscChecker`` gate-post collision.
    frame_collision : bool
        Gates mode only: ``True`` enables ``FrameChecker`` sphere-vs-gate-frame
        collision (each square gate = four thin oriented frame members).
        Mutually exclusive with ``post_radius > 0`` (choose discs OR frames);
        requires ``frame_thickness``, ``frame_depth`` and ``agent_radius`` all
        > 0.
    frame_thickness : float
        Gates mode with ``frame_collision`` only (required, > 0): thickness of
        the frame members (their extent across the opening plane).
    frame_depth : float
        Gates mode with ``frame_collision`` only (required, > 0): depth of the
        frame members along the gate-forward axis.
    agent_radius : float
        Gates mode with ``frame_collision`` only (required, > 0): the agent
        sphere radius for frame collision (posts are exclusive, so this cannot
        ride ``post_radius``).
    checkpoint_spacing : float or None
        Track mode only (required there): centerline checkpoint spacing.
    max_checkpoints : int or None
        Track mode only: the checkpoint capacity per env (``>= 3``), forwarded to
        ``CheckpointSampler``. ``None`` (default; roadmap PR14): the plan's
        size-independent bound ``max(3, ceil(max_loop_length / checkpoint_spacing))``
        with ``max_loop_length = 1.5 * N_max * spacing + dZ``
        (:meth:`track_gen.tools.ComponentContext.for_producer`; 70 at the
        ``TrackGenConfig`` defaults and ``checkpoint_spacing=0.5``), or, when the plan
        declares none (an elevation whose ``z_min`` / ``z_max`` declare no range, a
        variable width, a plugin geometry stage), the construction batch's readback as
        before (``ceil(1.5 * max valid centerline perimeter / checkpoint_spacing)``).
        A defaulted capacity that clips a valid env of the first batch warns
        (``UserWarning``) at construction. To keep the 0.2.0.dev5 shape, pass the
        capacity it printed (``course.checkpoints.position.shape[0] // num_envs``).
    max_boxes : int
        Collision query stride (boxes per env). Must be >= 1.
    samples_per_gate : int
        Gates mode only: centerline samples per gate for the
        :class:`CourseLine` spline (>= 2). Default 8.
    localize_window : int or None
        Gates mode only: forwarded as ``TrackLocalizer(warm_window=...)``;
        ``None`` (default) means full cold scans every query.
    """

    mode: str
    gen: "TrackGenConfig | GateGenConfig"
    seeds: "int | wp.array" = 0
    collision: "str | None" = None
    sdf_resolution: "int | None" = None
    heightfield_resolution: "int | None" = None
    post_radius: float = 0.0
    frame_collision: bool = False
    frame_thickness: float = 0.0
    frame_depth: float = 0.0
    agent_radius: float = 0.0
    checkpoint_spacing: "float | None" = None
    max_checkpoints: "int | None" = None
    max_boxes: int = 1
    samples_per_gate: int = 8
    localize_window: "int | None" = None

    def __post_init__(self):
        if self.mode not in ("track", "gates"):
            raise ValueError(
                f"mode must be 'track' or 'gates', got {self.mode!r}")
        if int(self.max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {self.max_boxes!r}")
        if self.mode == "track":
            if not isinstance(self.gen, TrackGenConfig):
                raise ValueError(
                    "mode='track' requires gen to be a TrackGenConfig, got "
                    f"{type(self.gen).__name__}")
            if float(self.post_radius) != 0.0:
                raise ValueError(
                    "post_radius applies to gates mode only (got "
                    f"{self.post_radius!r})")
            if self.frame_collision:
                raise ValueError(
                    "frame_collision applies to gates mode only (got "
                    f"{self.frame_collision!r})")
            for _name in ("frame_thickness", "frame_depth", "agent_radius"):
                if float(getattr(self, _name)) != 0.0:
                    raise ValueError(
                        f"{_name} applies to gates mode only (got "
                        f"{getattr(self, _name)!r})")
            if self.collision not in (None, "segments", "sdf"):
                raise ValueError(
                    "collision must be one of {None, 'segments', 'sdf'}, got "
                    f"{self.collision!r}")
            if self.checkpoint_spacing is None \
                    or not (float(self.checkpoint_spacing) > 0.0):
                raise ValueError(
                    "track mode requires checkpoint_spacing > 0, got "
                    f"{self.checkpoint_spacing!r}")
            if self.sdf_resolution is not None:
                if self.collision != "sdf":
                    raise ValueError(
                        "sdf_resolution applies only with collision='sdf'")
                if int(self.sdf_resolution) < 8:
                    raise ValueError(
                        f"sdf_resolution must be >= 8, got {self.sdf_resolution!r}")
            if self.heightfield_resolution is not None \
                    and int(self.heightfield_resolution) < 8:
                raise ValueError(
                    "heightfield_resolution must be >= 8, got "
                    f"{self.heightfield_resolution!r}")
            if self.max_checkpoints is not None and int(self.max_checkpoints) < 3:
                raise ValueError(
                    f"max_checkpoints must be >= 3, got {self.max_checkpoints!r}")
            if int(self.samples_per_gate) != 8:
                raise ValueError(
                    "samples_per_gate is a gates-mode option (got "
                    f"{self.samples_per_gate!r})")
            if self.localize_window is not None:
                raise ValueError(
                    "localize_window is a gates-mode option (got "
                    f"{self.localize_window!r})")
        else:
            if not isinstance(self.gen, GateGenConfig):
                raise ValueError(
                    "mode='gates' requires gen to be a GateGenConfig, got "
                    f"{type(self.gen).__name__}")
            if self.collision is not None:
                raise ValueError(
                    "collision is a track-mode option; gates mode uses "
                    "post_radius (got collision="
                    f"{self.collision!r})")
            if self.sdf_resolution is not None:
                raise ValueError("sdf_resolution is a track-mode option")
            if self.heightfield_resolution is not None:
                raise ValueError(
                    "heightfield_resolution is a track-mode option")
            if self.checkpoint_spacing is not None:
                raise ValueError(
                    "checkpoint_spacing is a track-mode option; gates mode "
                    "uses the gates themselves as checkpoints")
            if self.max_checkpoints is not None:
                raise ValueError("max_checkpoints is a track-mode option")
            if not (float(self.post_radius) >= 0.0):
                raise ValueError(
                    f"post_radius must be >= 0, got {self.post_radius!r}")
            if self.frame_collision and float(self.post_radius) > 0.0:
                raise ValueError(
                    "frame_collision and post_radius > 0 are mutually "
                    "exclusive: choose gate-post discs OR gate frames (got "
                    f"post_radius={self.post_radius!r})")
            if self.frame_collision:
                for _name in ("frame_thickness", "frame_depth", "agent_radius"):
                    if not (float(getattr(self, _name)) > 0.0):
                        raise ValueError(
                            f"frame_collision requires {_name} > 0, got "
                            f"{getattr(self, _name)!r}")
            else:
                for _name in ("frame_thickness", "frame_depth", "agent_radius"):
                    if float(getattr(self, _name)) != 0.0:
                        raise ValueError(
                            f"{_name} requires frame_collision=True (got "
                            f"{getattr(self, _name)!r})")
            if not (float(self.gen.gate_width) > 0.0):
                raise ValueError(
                    "gates mode requires gen.gate_width > 0: a width-0 gate "
                    "has a degenerate crossing segment and can never be "
                    "passed")
            if int(self.samples_per_gate) < 2:
                raise ValueError(
                    "samples_per_gate must be >= 2, got "
                    f"{self.samples_per_gate!r}")
            if self.localize_window is not None \
                    and int(self.localize_window) < 1:
                raise ValueError(
                    "localize_window must be >= 1 (or None for full scans), "
                    f"got {self.localize_window!r}")
        # max_boxes is the collision-query stride; without a collision checker
        # it is a dead option (track: collision=None; gates: post_radius==0).
        if int(self.max_boxes) > 1:
            has_checker = (self.mode == "track" and self.collision is not None) \
                or (self.mode == "gates" and float(self.post_radius) > 0.0)
            if not has_checker:
                raise ValueError(
                    "max_boxes > 1 is a collision-query stride but this course "
                    "has no box/disc collision checker (track: set collision; "
                    "gates: set post_radius > 0 — frame_collision uses sphere "
                    f"binding and ignores max_boxes), got max_boxes={self.max_boxes!r}")


# ---------------------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------------------

#: component name -> the 0.x StepResult field it fills
_LEGACY_FIELDS = {"progress": "events", "collision": "contacts", "localizer": "frame"}
#: the 0.x StepResult field -> the component name of its entry
_FIELD_ENTRIES = {f: k for k, f in _LEGACY_FIELDS.items()}


def _clone_entry(name: str, obj):
    """A deep copy of one StepResult entry (cold: allocates)."""
    if isinstance(obj, wp.array):
        return wp.clone(obj)
    clone = getattr(obj, "clone", None)
    if callable(clone):
        return clone()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        new = copy.copy(obj)
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, wp.array):
                object.__setattr__(new, f.name, wp.clone(v))
        return new
    raise TypeError(f"StepResult.clone(): the result of component {name!r} "
                    f"({type(obj).__name__}) has no clone() and is neither a wp.array nor "
                    f"a dataclass")


@dataclass
class StepResult:
    """Per-step bundle returned by :meth:`Course.step`.

    Holds the components' preallocated in-place results: the SAME ``StepResult`` (and the
    same underlying buffers) is returned on every ``step()``; use :meth:`clone` for
    snapshots.

    Since roadmap PR15 it also gives read-only mapping access to the result of every
    per-step component (one that overrides ``Component.step``), by component name, in
    execution order: ``res["progress"]``, ``res.get(name)``, ``keys()``, ``values()``,
    ``items()``, ``name in res``, ``dict(res)``. Per-generation components (checkpoints,
    the course line, bakes, props) are read through ``course.components[name].result``.
    It is not a ``collections.abc.Mapping``: it stays truthy, ``iter(res)`` raises
    ``TypeError`` and dataclass utilities see the four 0.x fields only
    (``dataclasses.replace`` rebuilds the entries from them; the ``repr`` is the
    dataclass one). Assigning a 0.x field (``res.events = x``, as tree utilities that
    copy a dataclass and set its fields do) updates its entry too (``None`` removes it),
    without touching the instance it was copied from.

    Attributes
    ----------
    events : ProgressEvents
        Progress events for this step: ``res["progress"]`` (``None`` on a course without
        a component named ``progress``).
    contacts : BoxContact or DiscContact or FrameContact or None
        Collision result: ``res["collision"]`` (``None`` when the course has no
        component named ``collision``).
    frame : TrackFrame or None
        Localization frame: ``res["localizer"]`` (``None`` without a component named
        ``localizer``: the 0.x track mode).
    valid : wp.array or None
        ``[E]`` int32 validity of each env's course (roadmap PR14): an alias of
        ``course.result.valid`` (``None`` only on a hand-built instance, or for a
        producer whose result has no ``valid``). The other fields of an env with
        ``valid[e] == 0`` hold the invalid-env sentinels (see :class:`Course`): gate
        rewards and observations on it with a select (``where(valid, x, 0)``, never a
        multiply: ``NaN * 0`` is NaN) and end such episodes. It is producer state and
        live: never write into it (exclude it from code that zeroes or poisons the step
        outputs; ``dataclasses.fields`` lists it, the mapping entries do not), and read it
        before a ``regenerate`` in the same iteration, which rewrites it for the
        regenerated envs (or ``clone()`` the result).
    """

    events: ProgressEvents
    contacts: "BoxContact | DiscContact | FrameContact | None"
    frame: "TrackFrame | None"
    valid: "wp.array | None" = None

    #: Not iterable, as in 0.x (``keys()`` / ``items()`` list the entries).
    __iter__ = None

    def __post_init__(self) -> None:
        # The entries live in a plain attribute, not a dataclass field: fields(), asdict()
        # and == are the 0.x ones. A hand-built instance gets the entries of its fields.
        self._items = {k: getattr(self, f) for k, f in _LEGACY_FIELDS.items()
                       if getattr(self, f) is not None}

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        entry = _FIELD_ENTRIES.get(name)
        items = self.__dict__.get("_items")
        if entry is None or items is None:
            return
        # Copy on write: a copy.copy() / deepcopy-with-memo of this instance shares (or
        # copied) the entries; reassigning a 0.x field keeps that field and its entry in
        # sync here only. Never on the hot path (step() assigns nothing).
        items = dict(items)
        if value is None:
            items.pop(entry, None)
        else:
            items[entry] = value
        object.__setattr__(self, "_items", items)

    @classmethod
    def _of(cls, items: dict, valid) -> "StepResult":
        """The StepResult of a course: ``items`` = ``name -> result`` of its per-step
        components, in execution order; the legacy fields from the names."""
        sr = cls(*(items.get(k) for k in _LEGACY_FIELDS), valid)
        sr._items = dict(items)
        return sr

    def __getitem__(self, name: str):
        """The result of the per-step component ``name``.

        Raises:
            KeyError: no per-step component has that name (the message lists them).
        """
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"no component {name!r} in this StepResult (per-step "
                           f"components: {list(self._items)}; per-generation ones: "
                           f"course.components[name].result)") from None

    def get(self, name: str, default=None):
        """The result of the per-step component ``name``, else ``default``."""
        return self._items.get(name, default)

    def keys(self):
        """The names of the per-step components, in execution order."""
        return self._items.keys()

    def values(self):
        """Their results, in execution order."""
        return self._items.values()

    def items(self):
        """``(name, result)`` pairs, in execution order."""
        return self._items.items()

    def __contains__(self, name) -> bool:
        return name in self._items

    def clone(self) -> "StepResult":
        """Deep-copy every entry (and ``valid``): a ``wp.array`` with ``wp.clone``, an
        object with a ``clone()`` method with it, a dataclass as a shallow copy whose
        ``wp.array`` fields are cloned (cold: allocates).

        Raises:
            TypeError: an entry is none of these (the message names its component).
        """
        items = {k: _clone_entry(k, v) for k, v in self._items.items()}
        return type(self)._of(items, None if self.valid is None else wp.clone(self.valid))


# ---------------------------------------------------------------------------------------
# The producer surface
# ---------------------------------------------------------------------------------------

#: The hot / cold methods a component must define (``tools.Component`` has them all).
_HOT_METHODS = ("attach", "bind", "on_generate", "step", "reset")
#: The 0.x bind names: accepted on every course, ignored when no component declares them.
_LEGACY_INPUTS = ("position", "orientation", "half_extents", "box_position")


class _Required:
    """``bind()``'s ``position`` default: an omitted ``position`` is the 0.x
    ``TypeError`` on a config course, "not given" on a composed one."""

    def __repr__(self) -> str:
        return "<required on a CourseConfig course>"


_MISSING = _Required()

#: id(producer) -> (weakref(producer), weakref(the Course driving it)): a newer Course
#: over the same producer (by identity) supersedes the older one (one course per
#: producer). An entry is dropped when its producer is collected.
_DRIVERS: dict = {}


def _drop_driver(key: int, pref) -> None:
    entry = _DRIVERS.get(key)
    if entry is not None and entry[0] is pref:
        del _DRIVERS[key]


def _take_over(prod, course) -> None:
    """Register ``course`` as the driver of ``prod``; the course driving it before (if
    alive) is superseded. A producer that is not weak-referenceable is not tracked."""
    key = id(prod)
    entry = _DRIVERS.get(key)
    if entry is not None and entry[0]() is not prod:     # a collected producer's id
        entry = None
    if entry is None:
        try:
            pref = weakref.ref(prod)
        except TypeError:
            return
        weakref.finalize(prod, _drop_driver, key, pref)
    else:
        pref = entry[0]
        old = entry[1]()
        if old is not None and old is not course:
            old._superseded = True
    _DRIVERS[key] = (pref, weakref.ref(course))


def _pipeline_type(module: str, name: str):
    """``track_gen.pipeline.<module>.<name>`` when that module is loaded, else ``None``
    (``import track_gen`` never imports the pipeline package: an object can only be
    one of its types once it is loaded)."""
    mod = sys.modules.get(f"track_gen.pipeline.{module}")
    return None if mod is None else getattr(mod, name, None)


def _is_program(obj) -> bool:
    cls = _pipeline_type("program", "Program")
    return cls is not None and isinstance(obj, cls)


def _is_spec(obj) -> bool:
    cls = _pipeline_type("spec", "PipelineSpec")
    return cls is not None and isinstance(obj, cls)


def _program_of(obj):
    """The ``Program`` of a facade (its ``program``), ``obj`` itself when it is one, else
    ``None`` (an unrelated attribute named ``program`` is not unwrapped)."""
    if _is_program(obj):
        return obj
    prog = getattr(obj, "program", None)
    return prog if _is_program(prog) else None


class _Producer:
    """The Producer surface of the object a course drives (private): the documented
    members (``result``, ``layout``, ``enqueue``; optional ``seeds``, ``capture_class``,
    ``prepared`` / ``prepare()``, ``default_stream``); a ``Program``'s specifics
    (``capturable``, ``legacy_stream``, ``host_stage_ids``, ``runner.name``) are read
    here only. A facade's ``program`` is unwrapped when it is a ``Program``; its ``rng``
    is used when it is a ``PerEnvSeededRNG`` (an unrelated attribute of either name is
    ignored)."""

    def __init__(self, generator) -> None:
        prog = _program_of(generator)
        prod = generator if prog is None else prog
        missing = [m for m in ("result", "layout", "enqueue") if not hasattr(prod, m)]
        if missing or not callable(getattr(prod, "enqueue", None)):
            raise TypeError(
                f"Course: {type(generator).__name__} lacks the Producer members "
                f"{missing or ['enqueue']} (result, layout, enqueue(ctx, *, advance, mask)"
                f"; see track_gen.pipeline.Producer)")
        self.obj = prod
        if _is_program(prod):
            self.capturable = bool(prod.capturable)
            self.legacy_stream = bool(prod.legacy_stream)
            self.host_ids = tuple(prod.host_stage_ids)
            self.name = str(prod.runner.name)
        else:
            cc = getattr(prod, "capture_class", None)
            host = False
            if cc is not None:
                from ..plugin.stage import CaptureClass  # noqa: PLC0415 - keep import cheap
                host = CaptureClass(cc) is CaptureClass.HOST
            self.capturable = not host
            self.legacy_stream = False
            self.host_ids = ()
            self.name = type(prod).__name__
        self.default_stream = getattr(prod, "default_stream", None)
        rng = getattr(generator, "rng", None)
        self.rng = rng if isinstance(rng, PerEnvSeededRNG) else None

    @property
    def prepared(self) -> bool:
        return bool(getattr(self.obj, "prepared", True))

    def prepare(self) -> None:
        """An unprepared producer's own warm-up (a ``Program`` built with
        ``prepare=False``)."""
        if not self.prepared and callable(getattr(self.obj, "prepare", None)):
            self.obj.prepare()


def _check_components(components) -> None:
    """Refuse an attached component (``ValueError``) and one lacking a hot method
    (``TypeError``), before anything is built."""
    for c in components:
        name = getattr(c, "name", None)
        if getattr(c, "layout", None) is not None:
            raise ValueError(f"Course: component {name!r} is already attached (to another "
                             f"Course or ComponentContext); build a new instance")
        miss = [m for m in _HOT_METHODS if not callable(getattr(c, m, None))]
        if miss:
            raise TypeError(f"Course: component {name!r} ({type(c).__name__}) lacks "
                            f"{miss}: a component defines attach, bind, on_generate, step "
                            f"and reset (subclass track_gen.tools.Component for the "
                            f"defaults)")


# ---------------------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------------------

class Course:
    """The composition root: one producer and its runtime components (roadmap PR15).

    Three front doors (all run ``__init__``, so a subclass initializes normally):

    - ``Course(generator, components=(), *, stream=None, prepare=True)``: ``generator``
      is any producer (a ``TrackGenerator``, a ``GateGenerator``, a built
      ``track_gen.pipeline.Program``, or an object with the
      :class:`track_gen.pipeline.Producer` surface); ``components`` are
      :mod:`track_gen.tools` components (built-in adapters or your own), sorted by their
      ``after`` / ``source`` dependencies (stable: an ordered list is kept as given);
    - the presets :meth:`track` / :meth:`gates` (the 0.x component sets over a config, a
      ``PipelineSpec`` or a built generator, plus ``extra`` components);
    - :meth:`from_config` (``Course(cfg)`` and ``Course(config=cfg)``, the 0.x forms, are
      its aliases): the 0.x course, byte for byte and launch for launch.

    Lifecycle: construct -> :meth:`bind` (stable sim buffers; required for :meth:`step`
    when a component declares a required input) -> per-step :meth:`step` / per-env
    :meth:`reset`, plus :meth:`regenerate` / :meth:`generate` for new courses.
    Construction attaches every component to the producer's current batch (their
    auto-derivations may read it back on the host once), then, with ``prepare=True``,
    runs one real warm-up regeneration; all of it must happen outside any CUDA graph
    capture. The 0.x sub-tool attributes are set from the components named
    ``progress``, ``checkpoints``, ``collision``, ``heightfield``, ``centerline`` and
    ``localizer`` (``None`` when absent): ``progress``, ``checkpoints`` (the checkpoint
    component's result), ``checkpoint_sampler``, ``collision``, ``heightfield``,
    ``course_line``, ``localizer``; with ``generator``, ``rng``, ``result``,
    :attr:`components`, :attr:`inputs`, :attr:`seeds`.

    Regeneration recomputes the whole batch (fixed-batch pipelines); per-env NEW courses
    come from :meth:`regenerate`'s ``mask`` + ``advance`` (only the masked envs change
    and are reset), per-env respawns on the SAME course from :meth:`reset`'s mask.

    Execution order: ``attach``, ``on_generate`` and ``step`` in the sorted order,
    ``reset`` in reverse order (a component is reset before the components it reads;
    resets read only their own state and the mask). ``on_generate`` and ``reset`` run in
    the masked regeneration body, under a CUDA conditional node where the capture allows
    one: they read no bound input and emit only kernels, fills and device-to-device
    copies on the call's stream (``track_gen.testing.check_component`` checks both).

    One course per producer: building a Course over a producer that a live Course drives
    makes the older course raise ``RuntimeError`` on its next ``step`` / ``reset`` /
    ``regenerate`` / ``enqueue_generate`` / ``generate`` (its components would read a
    source the newer course regenerates). Regenerate through the course: calling the
    producer directly regenerates the source without refreshing the components.

    Lifetime: the course holds the producer, every component and the
    :class:`StepResult`; recording a call into a Warp-registered CUDA capture anchors the
    course on that graph, with the arrays bound when ``step()`` was recorded and the
    ``mask`` / ``seeds`` arrays the recorded call reads (dropping every reference, or
    rebinding, while the graph lives keeps every buffer it reads). A CPU graph
    (``wp.ScopedCapture("cpu")``) retains the arrays its recorded launches touch itself.
    Under a plain ``torch.cuda.graph``, keep the course and those arrays alive
    (``track_gen.interop.torch.retain(g, course, ...)``).

    Invalid envs (``valid[e] == 0`` on :attr:`result`; roadmap PR14): every built-in
    component answers as for a poisoned env (``course.checkpoints.count = 0`` with NaN
    slots, ``oob = 1`` with NaN geometry, no post hit, frozen progress with
    ``dist_to_next = 0``, NaN ``s`` / ``n``, a NaN heightfield with the box ``(-1, -1)-(1,
    1)``). :attr:`StepResult.valid` carries the mask: set ``done |= 1 - valid`` and give
    those envs a NEW course with ``regenerate(mask=done, advance=True)`` (``reset`` would
    respawn them on the same invalid course).

    Args:
        generator: The producer (see above), or a ``CourseConfig`` (the 0.x form, then
            ``components`` / ``stream`` / ``prepare=False`` raise ``TypeError``).
        components: The components (unattached instances; each belongs to one course).
        stream: The course's default stream: calls that pass no ``stream=`` (and run
            outside an ambient scope) run on it. Default: the producer's
            (``TrackGenerator(..., stream=s)``), else resolved per call. Such a course
            refuses (``CaptureError``) a call without ``stream=`` while another stream
            is capturing: pass the capturing stream (or ``"torch"``) when recording.
        prepare: Run one warm-up regeneration at construction (loads every module the
            regeneration body uses). ``False`` defers it: :meth:`prepare` runs it, as
            does the first regeneration call outside a capture; recording one first
            raises ``CaptureError``.
        config: The 0.x keyword (``Course(config=cfg)``).

    Raises:
        TypeError: a config or spec given to the generic form (use the presets), a
            ``CourseConfig`` with components / ``stream`` / ``prepare=False``, a producer
            missing a required member, a component missing a method.
        ValueError: duplicate component names, a dependency that is not a component, a
            cycle, an attached component, an input declared with two specs or by a
            component and the generator.
        track_gen.exec.CaptureError: a capture is open (before anything is allocated).
    """

    def __init__(self, generator=None, components=(), *, stream=None, prepare: bool = True,
                 config: "CourseConfig | None" = None) -> None:
        _init()
        if config is not None:
            if generator is not None:
                raise TypeError("Course: pass a generator (or a CourseConfig) or config=, "
                                "not both")
            generator = config
        if hasattr(components, "attach") or isinstance(components, (str, bytes)):
            raise TypeError("Course: components must be a sequence of components, got "
                            f"{type(components).__name__} (pass [component])")
        comps = list(components)
        if isinstance(generator, CourseConfig):
            if comps or stream is not None or not prepare:
                raise TypeError("Course(CourseConfig) takes no components / stream / "
                                "prepare: use Course(generator, components, ...) or the "
                                "presets Course.track / Course.gates")
            self._init_from_config(generator)
            return
        if generator is None:
            raise TypeError("Course() needs a generator (a TrackGenerator, GateGenerator, "
                            "Program or Producer) or a CourseConfig")
        if isinstance(generator, (TrackGenConfig, GateGenConfig)) or _is_spec(generator):
            raise TypeError(f"Course(generator, components) takes a built generator, got a "
                            f"{type(generator).__name__}: use Course.track(...) / "
                            f"Course.gates(...) for a config or a spec, or "
                            f"Course.from_config(CourseConfig)")
        self._cfg = None
        self._setup(generator, comps, stream=stream, prepare=prepare)

    # -- front doors -----------------------------------------------------------------
    @classmethod
    def from_config(cls, config: "CourseConfig") -> "Course":
        """The 0.x course of a :class:`CourseConfig` (``Course(config)`` is an alias):
        the generator, components, outputs and launches it had before roadmap PR15
        (0.2.0.dev6 at PR14); relative to 0.2.0.dev5, see the CHANGELOG migration items
        1-8 (invalid envs, the default checkpoint capacity).

        Raises:
            TypeError: ``config`` is not a ``CourseConfig``.
        """
        if not isinstance(config, CourseConfig):
            raise TypeError(f"Course.from_config takes a CourseConfig, got "
                            f"{type(config).__name__}")
        return cls(config)

    @classmethod
    def track(cls, source, *, checkpoint_spacing: float,
              collision: "str | object | None" = None,
              max_checkpoints: "int | None" = None, extra=(), seeds=None,
              stream: "wp.Stream | None" = None, prepare: bool = True) -> "Course":
        """A track course: ``[Checkpoints, Progress, collision] + extra``.

        Args:
            source: A ``TrackGenConfig`` or a track ``PipelineSpec`` (the preset builds
                ``TrackGenerator(source, seeds=seeds, stream=stream)``), or a built
                ``TrackGenerator`` / track ``Program`` (used as is: ``seeds=`` /
                ``stream=`` raise ``TypeError``).
            checkpoint_spacing: Centerline checkpoint spacing (a number ``> 0``).
            collision: ``None`` (default, as ``CourseConfig.collision``), ``"segments"``
                / ``"sdf"`` (a ``tools.BandCollision`` of that method), or a component
                named ``"collision"`` (it fills ``StepResult.contacts`` and
                ``course.collision``; e.g. ``tools.BandCollision("sdf",
                sdf_resolution=256, max_boxes=2)``).
            max_checkpoints: The checkpoint capacity per env (an int ``>= 3``); ``None``
                or ``"auto"``: the plan bound, else the readback of the construction batch
                (``tools.Checkpoints(max_checkpoints="auto")``), with a ``UserWarning``
                when it clips a valid env of the first batch.
            extra: More components, a sequence (sorted with the preset's by their
                dependencies).
            seeds: The generator's seeds (anything ``TrackGenerator(seeds=)`` takes); a
                ``PerEnvSeededRNG`` becomes ``course.rng`` (``generate(seeds=)`` then
                reseeds through it, as 0.x).
            stream: The default stream of the generator and the course.
            prepare: See :class:`Course`.

        Raises:
            ValueError: a bad spacing / collision / capacity, a ``collision=`` component
                not named ``"collision"``, an ``extra`` that clashes with the preset's
                names or dependencies (all before anything is built).
            TypeError: a source of the other family or of another type, ``seeds=`` /
                ``stream=`` with a built generator, a single component as ``extra``.
        """
        if isinstance(checkpoint_spacing, bool) or \
                not isinstance(checkpoint_spacing, numbers.Real) or \
                not float(checkpoint_spacing) > 0.0:
            raise ValueError(f"checkpoint_spacing must be a number > 0, got "
                             f"{checkpoint_spacing!r}")
        if not (collision is None or hasattr(collision, "attach") or
                (isinstance(collision, str) and collision in ("segments", "sdf"))):
            raise ValueError(f"collision must be 'segments', 'sdf', None or a component "
                             f"named 'collision', got {collision!r}")
        _check_slot("track", collision)
        if isinstance(max_checkpoints, str) and max_checkpoints == "auto":
            max_checkpoints = None
        if max_checkpoints is not None and _int_at_least(max_checkpoints, 3) is None:
            raise ValueError(f"max_checkpoints must be an int >= 3, 'auto' or None, got "
                             f"{max_checkpoints!r}")
        extra = _extra_list("track", extra)
        built = _built(source, "track")
        if built is not None and (seeds is not None or stream is not None):
            raise TypeError("Course.track: seeds= / stream= configure the generator the "
                            "preset builds; a built generator already has them")
        own = [_Named("checkpoints"), _Named("progress", source="checkpoints")]
        if isinstance(collision, str):
            own.append(_Named("collision"))
        elif collision is not None:
            own.append(collision)
        _precheck(own, extra)
        gen = built if built is not None else \
            TrackGenerator(source, seeds=seeds, stream=stream)
        comps = [_tools.Checkpoints(checkpoint_spacing,
                                    max_checkpoints="auto" if max_checkpoints is None
                                    else int(max_checkpoints)),
                 _tools.Progress()]
        if isinstance(collision, str):
            comps.append(_tools.BandCollision(collision))
        elif collision is not None:
            comps.append(collision)
        return cls(gen, comps + extra, stream=stream, prepare=prepare)

    @classmethod
    def gates(cls, source, *, collision: "float | object | None" = None,
              samples_per_gate: int = 8, localize_window: "int | None" = None, extra=(),
              seeds=None, stream: "wp.Stream | None" = None,
              prepare: bool = True) -> "Course":
        """A gate course: ``[GateCheckpoints, CenterlineFromGates, Progress, collision,
        Localizer] + extra`` (the localizer on the gates' centerline).

        Args:
            source: A ``GateGenConfig`` or a gate ``PipelineSpec`` (the preset builds
                ``GateGenerator(source, seeds=seeds, stream=stream)``), or a built
                ``GateGenerator`` / gate ``Program`` (used as is: ``seeds=`` /
                ``stream=`` raise ``TypeError``). Its construction
                ``gate_frames.gate_width`` must be ``> 0``, or, when a randomizer draws
                the width (``with_randomize({"gate_frames.gate_width": (lo, hi)})``),
                that randomizer's construction ``lo`` (checked before anything is
                built); the preset then refuses a later scalar ``set`` of a width
                ``<= 0`` on the generator it uses (a given one included).
            collision: ``None``, a post radius ``> 0`` (a ``tools.DiscCollision``), or a
                component named ``"collision"`` (e.g. ``tools.FrameCollision(radius,
                thickness, depth)``, ``tools.DiscCollision(r, max_boxes=2)``).
            samples_per_gate: Centerline samples per gate (``>= 2``).
            localize_window: The localizer's ``warm_window`` (``None``: full scans).
            extra: More components, a sequence (sorted with the preset's by their
                dependencies).
            seeds: As :meth:`track`.
            stream: As :meth:`track`.
            prepare: As :meth:`track`.

        Raises:
            ValueError: a bad radius, a ``collision=`` component not named
                ``"collision"``, ``gate_width <= 0`` (or a width randomizer's ``lo <=
                0``), an ``extra`` that clashes (all before anything is built).
            TypeError: a source of the other family or of another type, ``seeds=`` /
                ``stream=`` with a built generator, a single component as ``extra``.
        """
        radius = isinstance(collision, numbers.Real) and not isinstance(collision, bool)
        if radius and not float(collision) > 0.0:
            raise ValueError(f"collision (a post radius) must be > 0, got {collision!r}")
        if not radius and not (collision is None or hasattr(collision, "attach")):
            raise ValueError(f"collision must be a post radius > 0, None or a component "
                             f"named 'collision', got {collision!r}")
        if not radius:
            _check_slot("gates", collision)
        if int(samples_per_gate) < 2:
            raise ValueError(f"samples_per_gate must be >= 2, got {samples_per_gate!r}")
        if localize_window is not None and int(localize_window) < 1:
            raise ValueError(f"localize_window must be >= 1 (or None for full scans), got "
                             f"{localize_window!r}")
        extra = _extra_list("gates", extra)
        built = _built(source, "gates")
        if built is not None and (seeds is not None or stream is not None):
            raise TypeError("Course.gates: seeds= / stream= configure the generator the "
                            "preset builds; a built generator already has them")
        _gate_width_check(source if built is None else built)
        own = [_Named("checkpoints"), _Named("centerline"),
               _Named("progress", source="checkpoints"),
               _Named("localizer", source="centerline")]
        if radius:
            own.insert(3, _Named("collision"))
        elif collision is not None:
            own.insert(3, collision)
        _precheck(own, extra)
        gen = built if built is not None else \
            GateGenerator(source, seeds=seeds, stream=stream)
        _narrow_gate_width(gen)
        comps = [_tools.GateCheckpoints(), _tools.CenterlineFromGates(samples_per_gate),
                 _tools.Progress()]
        if radius:
            comps.append(_tools.DiscCollision(float(collision)))
        elif collision is not None:
            comps.append(collision)
        comps.append(_tools.Localizer(warm_window=localize_window, source="centerline"))
        return cls(gen, comps + extra, stream=stream, prepare=prepare)

    def _init_from_config(self, config: CourseConfig) -> None:
        """The 0.x construction (the same allocations, attach order, module loads and
        warm-up, byte for byte and launch for launch)."""
        self._cfg = config
        E = int(config.gen.num_envs)
        # Canonicalize via Warp: "cuda" -> "cuda:0" so the string matches
        # str(arr.device) in _check_arr (bind/seed validation would otherwise
        # reject cuda:0 buffers when the config used the "cuda" alias).
        device = wp.get_device(config.gen.device)
        self._E, self._device = E, str(device)
        if isinstance(config.seeds, wp.array):
            self._validate_seed_array(config.seeds)
        label = f"course[{config.gen.generator}]"
        # Refuse a capture BEFORE the first allocation (the rng's).
        ensure_not_capturing(device, label=label)
        rng = PerEnvSeededRNG(seeds=config.seeds, num_envs=E, device=self._device)
        # The generator prepares itself (one real generation), so its result already
        # holds the batch for the initial seeds.
        if config.mode == "track":
            gen = TrackGenerator(config.gen, rng)
        else:
            gen = GateGenerator(config.gen, rng)
            # Roadmap PR13d: gate_frames.gate_width is per env; keep the construction
            # check's invariant for a scalar set() (an array is not range-checked: a
            # value <= 0 is used as 0, and those envs' gates can never be passed).
            _narrow_gate_width(gen)
        self._setup(gen, _components_for(config), stream=None, prepare=True, label=label,
                    rng=rng)

    def _setup(self, generator, components, *, stream, prepare: bool, label=None,
               rng=None) -> None:
        """The one construction path of every front door."""
        pinfo = _Producer(generator)
        prod = pinfo.obj
        comps = list(components)
        _check_components(comps)
        ordered = _order(comps)
        gen_inputs = frozenset(n for n in (getattr(prod, "input_info", None) or {})
                               if n != "regen_mask")
        clash = sorted({k for c in ordered for k in dict(getattr(c, "inputs", {}) or {})
                        if k in gen_inputs})
        if clash:
            raise ValueError(f"Course: input {clash[0]!r} is declared by the generator and "
                             f"by a component: rename the component's input")
        lay = prod.layout
        dev = wp.get_device(lay.device)
        if label is None:
            label = f"course[{pinfo.name}]"
        default_stream = stream if stream is not None else pinfo.default_stream
        _check_default_stream(dev, default_stream)     # TypeError / ValueError, early
        # Refuse a capture BEFORE any work: the producer's own warm-up, the first
        # allocation.
        ensure_not_capturing(dev, default_stream=default_stream, label=label)
        pinfo.prepare()               # an unprepared Program: its own warm-up first
        self.generator = generator
        self.rng = rng if rng is not None else pinfo.rng
        self._producer = prod
        self.result: "Track | GateSequence" = prod.result
        self._E = int(lay.num_envs)
        self._device = str(dev)
        self._is_cuda = dev.is_cuda
        self._gen_inputs = gen_inputs
        self.collision: "CollisionChecker | DiscChecker | FrameChecker | None" = None
        self.course_line: "CourseLine | None" = None
        self.localizer: "TrackLocalizer | None" = None
        self.heightfield: "HeightFieldBaker | None" = None
        self.checkpoints: "CheckpointSet | None" = None
        self.checkpoint_sampler: "CheckpointSampler | None" = None
        self.progress: "ProgressTracker | None" = None
        self._posts: "wp.array | None" = None
        self._bind_args: "dict | None" = None
        self._superseded = False
        self._reset_all_mask = wp.full(self._E, 1, dtype=wp.int32, device=self._device)
        # Persistent [E] int32 buffer for generate(seeds=int): the int is expanded
        # device-side into it, so a reseed never allocates.
        self._reseed_buf = wp.empty(self._E, dtype=wp.int32, device=self._device)
        # Persistent device inputs (see `inputs`) and the [1] int32 "any env masked" flag
        # of a masked regeneration (written on device).
        self._regen_mask = wp.zeros(self._E, dtype=wp.int32, device=self._device)
        self._regen_flag = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._reset_mask = wp.zeros(self._E, dtype=wp.int32, device=self._device)
        # The private all-zero mask of a deferred warm-up (prepare=False only).
        self._zero_mask = None if prepare else \
            wp.zeros(self._E, dtype=wp.int32, device=self._device)
        self._inputs = types.MappingProxyType({"regen_mask": self._regen_mask,
                                               "reset_mask": self._reset_mask})
        # A Course over a pipeline with HOST stages runs eagerly (it does not segment)
        # and refuses a caller's capture naming them.
        self._runner = GraphRunner(
            self._device, capturable=pinfo.capturable, legacy_stream=pinfo.legacy_stream,
            default_stream=default_stream, owner=self, name=label,
            _host_ids=pinfo.host_ids)
        # The cold attach phase (allocation + readback-based sizing; never inside a
        # capture) runs on the stream where allocations happen (Warp's current one: every
        # attach allocates and zero-fills there, then launches its first on_generate on
        # cc.ctx), after the producer's work on the course stream; the course stream then
        # waits for it. A config course has no default stream: one stream, no split,
        # unless it is built inside an ambient LaunchContext.scope() on another stream
        # (then the attach phase runs on Warp's current stream, ordered after and before
        # that stream, as for a stream= course).
        ctx = self._runner.context()
        cur = LaunchContext.current(dev) if dev.is_cuda else ctx
        split = dev.is_cuda and ctx.stream is not None and \
            int(ctx.stream.cuda_stream or 0) != int(cur.stream.cuda_stream or 0)
        pstream = pinfo.default_stream
        if dev.is_cuda and ctx.stream is not None and isinstance(pstream, wp.Stream) and \
                int(pstream.cuda_stream or 0) != int(ctx.stream.cuda_stream or 0):
            # A course on another stream than the producer's own default stream: the
            # producer's pending work there (e.g. gen.generate(sync=False)) comes first,
            # for the attach phase and the warm-up alike.
            ctx.stream.wait_stream(pstream)
        if split:
            cur.stream.wait_stream(ctx.stream)
        actx = cur if split else ctx
        with actx.scope():
            if getattr(prod, "spec", None) is not None or \
                    getattr(prod, "capacity", None) is not None or \
                    getattr(prod, "bounds", None) is not None:
                cc = _tools.ComponentContext.for_producer(prod, ctx=actx)
            else:
                cc = _tools.ComponentContext(lay, ctx=actx)
            for comp in ordered:
                cc.add(comp, self.result)
            self._components = ordered
            self._by_name = {c.name: c for c in ordered}
            self._components_view = types.MappingProxyType(self._by_name)
            self._set_legacy_attributes()
            self._needs_bind = any(s.required for c in ordered
                                   for s in dict(getattr(c, "inputs", {}) or {}).values())
            # Load the reseed kernel's module now (writes only the scratch buffer), so a
            # first generate(seeds=int) never loads a module inside a caller's capture.
            actx.launch(_seed_range_k, self._E, [0, self._reseed_buf])
            # Same for the masked regeneration's reduction (writes only the flag; the
            # mask is all zeros).
            any_set(actx, self._regen_mask, self._regen_flag)
        if split:
            ctx.stream.wait_stream(cur.stream)
        self._step_result = StepResult._of(
            {c.name: c.result for c in ordered if _per_step(c)},
            getattr(self.result, "valid", None))
        if prepare:
            # One real warm-up regeneration (producer + every component) outside any
            # capture: loads every module the regeneration body uses.
            self._runner.prepare(self._enqueue_regenerate)
        self._warn_clipped_default()
        # One course per producer: this course takes it over. The last statement, so a
        # construction that raised (a warning turned into an error included) supersedes
        # nothing.
        _take_over(prod, self)

    def _set_legacy_attributes(self) -> None:
        """The 0.x sub-tool attributes, from the components of their names."""
        by = self._by_name

        def util(name):
            c = by.get(name)
            return None if c is None else getattr(c, "utility", None)

        self.progress = util("progress")
        self.checkpoints = by["checkpoints"].result if "checkpoints" in by else None
        self.checkpoint_sampler = util("checkpoints")   # None in gates mode
        self.collision = util("collision")
        coll = by.get("collision")
        self._posts = None if coll is None else getattr(coll, "posts", None)
        self.heightfield = util("heightfield")
        self.course_line = util("centerline")
        self.localizer = util("localizer")

    def _warn_clipped_default(self) -> None:
        """After construction: warn once per checkpoint component whose capacity it chose
        itself (``defaulted``) when that capacity clips a valid env of the first batch
        (one cold readback of two ``[E]`` int32 arrays; never with an explicit
        ``max_checkpoints``)."""
        hits = [c for c in self._components if getattr(c, "defaulted", None) and
                getattr(getattr(c, "utility", None), "truncated", None) is not None]
        if not hits:
            return
        ctx = self._runner.context()
        if ctx.stream is not None:
            _compat.stream_synchronize(ctx.stream)
        va = self.result.valid.numpy()
        for c in hits:
            tr = c.utility.truncated.numpy()
            n = int(((tr != 0) & (va != 0)).sum())
            if not n:
                continue
            where = "checkpoint_sampler.truncated" if c.name == "checkpoints" else \
                f"components[{c.name!r}].utility.truncated"
            fix = "CourseConfig.max_checkpoints" if self._cfg is not None else \
                "max_checkpoints"
            warnings.warn(
                f"Course: the default max_checkpoints={c.utility.max_checkpoints} "
                f"({c.defaulted}) clips the checkpoint ring of {n} valid env(s) of the "
                f"first batch ({where}); pass a larger {fix}", UserWarning,
                stacklevel=_user_stacklevel())

    # -- state -----------------------------------------------------------------------
    @property
    def graph(self) -> "wp.Graph | None":
        """The course's own regeneration graph (producer + every component's refresh as
        ONE graph) for a plain :meth:`regenerate` / :meth:`generate`: captured on the
        first such CUDA call made outside any capture, replayed by later ones. ``None``
        on CPU, for a producer with HOST stages, and before that call. Each ``(advance,
        mask)`` variant of :meth:`regenerate` caches one more graph."""
        return self._runner.graph

    @property
    def seeds(self):
        """The producer's :class:`~track_gen.seeds.SeedState` (the per-env seeds, bound
        by pointer or owned, plus the per-env epoch); ``None`` for a producer without
        one."""
        return getattr(self._producer, "seeds", None)

    @property
    def inputs(self) -> "types.MappingProxyType":
        """Persistent device inputs (stable pointers, allocated at construction):

        - ``"regen_mask"``: ``[E]`` int32, zero-initialised. Write the envs to regenerate
          from your own kernels (e.g. ``done`` flags inside your sim graph) and pass it as
          ``regenerate(mask=course.inputs["regen_mask"], advance=True)``. Its pointer
          never changes, so the course's self-graph for it is captured once.
        - ``"reset_mask"``: ``[E]`` int32, zero-initialised (roadmap PR15): the envs to
          respawn on their current course, passed as
          ``reset(course.inputs["reset_mask"])``.

        The generator's own inputs stay at ``course.generator.inputs``: write them in
        place (they are read by every regeneration and every replay).
        """
        return self._inputs

    @property
    def components(self) -> "types.MappingProxyType":
        """The components, ``name -> component``, in execution order (read-only)."""
        return self._components_view

    # -- validation ------------------------------------------------------------------
    def _check_arr(self, name: str, arr, shape: tuple, dtype) -> None:
        """Validate a bound/seed wp.array's shape, dtype, and device (thin wrapper over
        :func:`runtime._check_arr`, pinned to this course's device)."""
        _check_arr(name, arr, shape, dtype, self._device)

    def _validate_seed_array(self, seeds) -> None:
        """A wp.array seeds must be ``[E]`` int32 on the course device.

        ``set_seeds_warp`` launches with ``dim=len(seeds)`` against the ``[E]`` state
        arrays: a longer array corrupts device memory past the states, a shorter one only
        partially reseeds.
        """
        self._check_arr("seeds", seeds, (self._E,), wp.int32)

    def _check_mask(self, mask, name: str = "mask") -> None:
        """A contiguous ``[E]`` int32 mask on the course device (the ``SeedState``
        messages; checked by the course, so a producer without seeds works)."""
        _check_arr(name, mask, (self._E,), wp.int32, self._device)
        if not mask.is_contiguous:
            raise ValueError(f"{name} must be a contiguous array (got strides "
                             f"{tuple(mask.strides)}); pass a contiguous copy")

    def _needs_boxes(self) -> bool:
        if self._cfg is None:
            return any("orientation" in dict(getattr(c, "inputs", {}) or {})
                       for c in self._components)
        # Frame collision binds a sphere (position only) — no oriented-box
        # yaw/half_extents buffers — so it is deliberately excluded here.
        cfg = self._cfg
        if cfg.mode == "gates" and cfg.frame_collision:
            return False
        return (cfg.mode == "track" and cfg.collision is not None) or \
               (cfg.mode == "gates" and cfg.post_radius > 0.0)

    def _validate_bind_args(self, a: dict) -> None:
        """Eagerly validate bound buffers (config course) — E, max_boxes, device are all
        known at construction, so shape/dtype/device errors surface at ``bind()`` rather
        than at the first ``step()``."""
        E = self._E
        self._check_arr("position", a["position"], (E,), wp.vec3f)
        if self._needs_boxes():
            nb = (E * self._cfg.max_boxes,)
            self._check_arr("orientation", a["orientation"], nb, wp.quatf)
            self._check_arr("half_extents", a["half_extents"], nb, wp.vec2f)
            if self._cfg.max_boxes > 1:
                self._check_arr("box_position", a["box_position"], nb, wp.vec3f)

    def _check_owner(self) -> None:
        if self._superseded:
            raise RuntimeError(
                "this Course's generator is now driven by another Course built later over "
                "the same generator: its components would read a source that course "
                "regenerates (stale checkpoints and bakes); use the newer course, or build "
                "one generator per course")

    # -- binding ---------------------------------------------------------------------
    def bind(self, position: "wp.array" = _MISSING, orientation: "wp.array | None" = None,
             half_extents: "wp.array | None" = None,
             box_position: "wp.array | None" = None, **arrays: "wp.array") -> None:
        """Bind stable sim buffers to the components that declare them (required before
        :meth:`step` when one declares a required input).

        Every argument is routed by name to every component whose ``inputs`` declare it
        (``Progress``, ``Localizer`` and your own all take ``position``), or that names it
        as an input's ``default`` (``box_position`` defaults to ``position`` with one box
        per env; an explicitly given name wins over the default). The four 0.x names
        (``position``, ``orientation``, ``half_extents``, ``box_position``) are accepted on
        every course and ignored when no component declares them.

        ``position`` is the ``[E]`` vec3f agent-position buffer. With a box collision
        component, ``orientation`` (``[E * max_boxes]`` quatf box poses) and
        ``half_extents`` (``[E * max_boxes]`` vec2f, planar boxes) are required too; with
        ``max_boxes == 1`` the same ``position`` buffer serves as the box positions,
        otherwise pass a separate ``box_position`` ``[E * max_boxes]`` vec3f buffer. May be
        called any time after construction (before or after a regeneration).

        A bind is complete and atomic: every route is validated before anything is
        rebound, and a rebind replaces every binding; when a component refuses
        (``on_bind`` raising part-way), the components already rebound by the call and
        the one that refused are bound again to their previous arrays, their
        rebind-after-capture probes are restored, and the error propagates (every
        previous binding stays in place, and a later rebind after a capture still
        warns).

        Do NOT rebind after capturing ``step()`` into a sim graph: the captured graph
        replays against the buffer pointers live at capture time, so a later rebind
        leaves it reading the old buffers (silently divergent results) — keep writing into
        the SAME bound buffers and rebind only before (re)capturing. Since roadmap PR14
        such a rebind (other arrays, after ``step()`` ran while its stream or a CPU graph
        capture was capturing) warns once,
        :class:`~track_gen.exec.RebindAfterCaptureWarning`; rebinding the same arrays is
        silent. The recipe "rebind, then record again" warns
        once too: filter the category when you re-record on purpose. The course's own
        regeneration graphs never read bound inputs, so they stay valid.

        The generator's inputs (``course.generator.inputs``, e.g. ``"centerline.xy"`` of
        ``generator="external"``) are not bound here: write them in place.

        Raises:
            TypeError: an unknown name, a generator input name (with the in-place
                recipe), a missing required input; on a config course, ``bind()``
                without ``position``.
            ValueError: an array of the wrong shape, dtype or device (``position=None``
                on a config course: the 0.x message).
            RuntimeError: a config course with a box collision checker bound without
                ``orientation`` / ``half_extents`` (or ``box_position`` with ``max_boxes
                > 1``).
        """
        if position is _MISSING:
            if self._cfg is not None:
                raise TypeError("Course.bind() missing 1 required argument: 'position'")
            position = None
        token, hits = _rebind_open()
        ok = False
        try:
            given = {k: v for k, v in zip(_LEGACY_INPUTS, (position, orientation,
                                                           half_extents, box_position))
                     if v is not None}
            given.update({k: v for k, v in arrays.items() if v is not None})
            self._bind(given)
            ok = True
        finally:
            if not ok:
                hits.clear()         # a refused bind changed nothing: no warning
            _rebind_close(token, hits, stacklevel=2)

    def _bind(self, given: dict) -> None:
        if self._cfg is not None:
            needs_boxes = self._needs_boxes()
            if needs_boxes and (given.get("orientation") is None or
                                given.get("half_extents") is None):
                raise RuntimeError(
                    "this course has a collision checker: bind orientation and "
                    "half_extents as well")
            if needs_boxes and self._cfg.max_boxes > 1 and \
                    given.get("box_position") is None:
                raise RuntimeError(
                    "max_boxes > 1: bind a separate box_position "
                    "[E*max_boxes] buffer")
            self._validate_bind_args({k: given.get(k) for k in _LEGACY_INPUTS})
        declared: set = set()
        for c in self._components:
            ins = dict(getattr(c, "inputs", {}) or {})
            declared |= set(ins) | {s.default for s in ins.values()
                                    if s.default is not None}
        gen = sorted(k for k in given if k in self._gen_inputs and k not in declared)
        if gen:
            raise TypeError(
                f"Course.bind: {gen} {'is an input' if len(gen) == 1 else 'are inputs'} of "
                f"the generator, not of a component: write "
                f"course.generator.inputs[{gen[0]!r}] in place (a stable array, read by "
                f"every regeneration and every replay), e.g. wp.copy into it on the "
                f"regeneration's stream; binding a generator input to your own array "
                f"arrives with build(inputs=)")
        unknown = sorted(set(given) - declared - set(_LEGACY_INPUTS))
        if unknown:
            raise TypeError(f"Course.bind: unknown inputs {unknown} (component inputs: "
                            f"{sorted(declared)})")
        plan = []
        for c in self._components:
            ins = dict(getattr(c, "inputs", {}) or {})
            if not ins:
                continue
            routes = set(ins) | {s.default for s in ins.values() if s.default is not None}
            sub = {k: v for k, v in given.items() if k in routes}
            _route(c, sub)                      # validate every route before binding
            plan.append((c, routes, sub))
        # The rebind-after-capture probes (roadmap PR14) of every component and its
        # utility: a rebind clears them; a refused bind restores them with the arrays.
        armed = [(o, o._recorded) for c, _, _ in plan
                 for o in (c, getattr(c, "utility", None))
                 if o is not None and hasattr(o, "_recorded")]
        done = 0
        try:
            for c, _, sub in plan:
                c.bind(**sub)
                done += 1
        except BaseException:
            prev = self._bind_args
            if prev is not None:
                # Restore what this call rebound, the component that refused included
                # (its on_bind may have pinned an array before raising).
                for c, routes, _ in plan[:done + 1]:
                    try:
                        c.bind(**{k: v for k, v in prev.items() if k in routes})
                    except Exception:   # noqa: BLE001 - the original error propagates
                        pass
            for o, flag in armed:
                o._recorded = flag
            raise
        self._bind_args = given

    def _apply_bind(self) -> None:
        """Route the bound arrays to every component again (0.x private name)."""
        if self._bind_args is not None:
            self._bind(self._bind_args)

    # -- the regeneration body ---------------------------------------------------------
    def _enqueue_regenerate(self, ctx: LaunchContext, *, advance: bool = False,
                            mask: "wp.array | None" = None) -> None:
        """The ONE regeneration body, all on ``ctx``.

        ``mask is None``: [advance every env] -> the producer's generation (seed prologue
        included) -> every component's ``on_generate(ctx, None)`` in execution order ->
        every component's ``reset`` of every env in reverse order. With a ``mask``:
        ``any_set(mask)`` into a device flag, then, skipped on device when no env is
        masked where CUDA supports it (:meth:`LaunchContext.skip_unless`), [advance the
        masked envs] -> the producer's masked generation (its masked commit) ->
        ``on_generate(ctx, mask)`` -> ``reset(ctx, mask)`` in reverse order.

        Run by ``prepare`` (construction), by the eager path, by the self-graph capture
        and by a caller's recording alike. Zero allocation, host-sync free.
        """
        if mask is None:
            self._producer.enqueue(ctx, advance=advance)
            self._refresh(ctx)
            return
        any_set(ctx, mask, self._regen_flag)
        ran = []

        def body(c, _advance=advance, _mask=mask) -> None:
            self._enqueue_regenerate_masked(c, advance=_advance, mask=_mask)
            ran.append(True)

        try:
            ctx.skip_unless(self._regen_flag, body)
        except CaptureError:
            raise
        except RuntimeError as exc:
            if not ran:
                raise
            # The body returned: Warp refused it as a CUDA conditional-node body.
            raise CaptureError(
                f"{self._runner.name!r}: the masked regeneration body could not be "
                f"recorded under a CUDA conditional node: a component's on_generate or "
                f"reset uses an operation a conditional body cannot hold (a copy to or "
                f"from pageable host memory, an event, another stream). Components in "
                f"the body: {[c.name for c in self._components]}. Check each with "
                f"track_gen.testing.check_component (capture.caller); "
                f"track_gen.exec.unconditional() records the body without the node."
            ) from exc

    def _enqueue_regenerate_masked(self, ctx: LaunchContext, *, advance: bool,
                                   mask: wp.array) -> None:
        """Masked regeneration body (an exact no-op when every ``mask[e]`` is 0: nothing
        advances, no effective seed is re-derived, so every env is recomputed from the
        effective seed its current course was built from, the producer's commit restores
        every row, every component recomputes from the unchanged result, and nothing is
        reset)."""
        self._producer.enqueue(ctx, advance=advance, mask=mask)
        self._refresh_geometry(ctx, mask)
        self._reset_runtime(ctx, mask)

    def _refresh(self, ctx: LaunchContext) -> None:
        """Post-generation coherence: every component's ``on_generate`` + a full reset.

        Args:
            ctx: the regeneration's launch context; every component runs on
                ``ctx.stream``.
        """
        self._refresh_geometry(ctx)
        self._reset_runtime(ctx, self._reset_all_mask)

    def _refresh_geometry(self, ctx: LaunchContext, mask=None) -> None:
        """Rebuild every buffer DERIVED from the course geometry (checkpoint resample,
        course line, sdf / heightfield bake, posts, your components' per-generation
        work): each component's ``on_generate``, in execution order, on ``ctx.stream``.
        The built-ins recompute the whole batch: an env whose geometry did not change
        gets bitwise-identical derived buffers."""
        for comp in self._components:
            comp.on_generate(ctx, mask)

    def _reset_runtime(self, ctx: LaunchContext, mask: wp.array) -> None:
        """Reset every component's per-env runtime state where ``mask[e] != 0``, in
        reverse execution order (the localizer's warm-start memory, then progress, for
        the 0.x courses), on ``ctx.stream``. A stateless component emits nothing."""
        for comp in reversed(self._components):
            comp.reset(ctx, mask)

    def _regen_call(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one regeneration; validates ``mask`` host-side,
        before any launch."""
        if mask is not None:
            self._check_mask(mask)       # [E] int32, contiguous, course device
        advance = bool(advance)
        if mask is None and not advance:
            return self._enqueue_regenerate, None
        key = (advance, 0 if mask is None else int(mask.ptr))
        return functools.partial(self._enqueue_regenerate, advance=advance,
                                 mask=mask), key

    def _ctx(self, stream) -> LaunchContext:
        """The call's context. A course with a default stream refuses a call without
        ``stream=`` (and outside an ambient scope) while another stream is capturing
        (Warp's current stream, not by a library self-capture, or torch's in this
        thread): the call would run eagerly on its own stream, outside the caller's
        graph."""
        ctx = self._runner.context(stream)
        ds = self._runner.default_stream
        if stream is None and ds is not None and ctx.stream is ds and \
                _AMBIENT.get() is None and not ctx.capturing:
            dev = ctx.device
            cur = wp.get_stream(dev)
            foreign = cur.is_capturing and \
                int(cur.cuda_stream or 0) != int(ds.cuda_stream or 0) and \
                not _compat.library_capture(cur)
            if foreign or _torch_capturing(dev) is not None:
                raise CaptureError(
                    f"{self._runner.name!r} was built with stream= (its default stream) "
                    f"and is called while another stream is capturing: the call would run "
                    f"eagerly on its own stream, outside your graph. Pass stream= (the "
                    f"capturing stream, or 'torch') when recording.")
        return ctx

    def _ready(self, ctx: LaunchContext) -> None:
        """Prepare lazily (``prepare=False``) outside a capture; refuse inside one."""
        if self._runner.prepared:
            return
        if _host_blocked(ctx):
            raise CaptureError(
                f"{self._runner.name!r} was built with prepare=False and is being recorded "
                f"into a capture: call course.prepare() outside the capture first (its "
                f"warm-up loads every module the regeneration body uses)")
        self.prepare()

    def prepare(self) -> None:
        """Load every module the regeneration bodies use, outside any capture, and wait
        for it: a no-op once prepared (construction with ``prepare=True`` included).

        For a course built with ``prepare=False``: one real masked regeneration on a
        private all-zero mask (an exact identity: no env's course or runtime state
        changes), synchronized. The first regeneration call outside a capture runs it
        too; recording one before it raises ``CaptureError``.

        Raises:
            track_gen.exec.CaptureError: a capture is open.
        """
        if self._runner.prepared:
            return
        self._runner.prepare(functools.partial(self._enqueue_regenerate, advance=True,
                                               mask=self._zero_mask))

    def regenerate(self, *, mask: "wp.array | None" = None, advance: bool = False,
                   stream=None, sync: bool = True):
        """Regenerate courses plus a coherent refresh of every component; returns
        :attr:`result`.

        ``mask=None`` (whole batch): the producer's generation from the current seeds
        (after advancing EVERY env's epoch when ``advance=True``), then every component's
        ``on_generate`` (checkpoint resample / sdf and heightfield bake / posts and line
        rebuild as applicable, your components' per-generation work) and a FULL reset of
        every component (progress, localizer, ...) — every course changed (or, without
        ``advance``, was rebuilt identically).

        ``mask`` given (``[E]`` int32 on the course device, read on device): envs with
        ``mask[e] != 0`` advance their epoch (when ``advance=True``) and get the course
        of their live ``(seed, epoch)``, and their runtime state is reset (every
        component's ``reset(ctx, mask)``, in reverse order); envs with ``mask[e] == 0``
        keep their course and runtime state. The pipeline is fixed-shape, so it still
        recomputes every env: unmasked envs keep the effective seed their current course
        was built from and are recomputed to identical values, even if their seed or
        epoch changed since (``rng.set_seeds_warp``, ``seeds.reset_epoch()``, a write to
        the seed array): such a change reaches an env at its next masked regeneration,
        or every env at the next one without a mask. The generator's commit also
        restores an unmasked env's ``Track`` rows (on a gate course its
        ``GateSequence`` rows) bit for bit after the recompute
        (``track_gen.stages.commit``). Without ``advance`` a mask respawns the masked
        envs (as :meth:`reset`) on the course of their current ``(seed, epoch)``: the
        same one unless their seed was rewritten.

        Conditional skip: the masked body runs after a device-side ``any_set(mask)``
        under :meth:`track_gen.exec.LaunchContext.skip_unless`. Inside a Warp-registered
        capture (Warp ``ScopedCapture``, the course's own self-graph,
        ``capture_begin(external=True)``, or
        ``track_gen.interop.torch.registered(conditional=True)`` under
        ``torch.cuda.graph``) with CUDA Toolkit and driver >= 12.4, it is skipped on
        device when no env is masked; the caller's graph then holds a conditional node,
        so it can no longer be used as a CUDA child graph, and ``wp.capture_if`` ends and
        restarts the capture: join every side stream before this call (or wrap it in
        ``track_gen.exec.unconditional()``). Every component's ``on_generate`` and
        ``reset`` run inside that conditional body (kernels, fills and device-to-device
        copies only; a refused body raises ``CaptureError`` naming the components).
        Elsewhere it runs unconditionally, which is correct because an all-zero mask makes
        it a no-op: nothing advances, nothing is reset, and every env is recomputed from
        the effective seed its current course was built from (the producer's commit
        restores every row; the built-in components recompute from the unchanged result).

        Execution (stream resolved per :mod:`track_gen.exec`):

        - inside a capture on the resolved stream: RECORDED into the caller's graph (e.g.
          next to :meth:`step` / :meth:`reset`); each replay regenerates from the seeds,
          epochs and mask at replay time (and, with ``advance``, advances again);
        - outside a capture on CUDA: replay of a self-graph (producer + refresh as ONE
          graph, captured on the first call of each ``(advance, mask)`` variant;
          :attr:`graph` is the plain one);
        - on CPU, or with a producer that has HOST stages: eager (a Course does not
          segment).

        A course built with ``prepare=False`` prepares first (see :meth:`prepare`).

        Args:
            mask: optional contiguous ``[E]`` int32 regeneration mask. Pass a STABLE
                buffer (e.g. ``inputs["regen_mask"]``): outside a capture each distinct
                mask pointer captures, and caches, one more self-graph (up to a bound,
                past which new pointers run eagerly).
            advance: advance the epoch (of every env, or of the masked envs) on device
                first, so those envs get new courses. Default ``False`` (rebuild from
                unchanged seeds and epochs).
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until the regeneration finished on the resolved stream
                (default ``True``; never while capturing, nor while torch is capturing in
                the calling thread).

        Raises:
            ValueError: ``mask`` is not a contiguous ``[E]`` int32 array on the course
                device (before any launch).
            RuntimeError: a newer Course drives this course's producer.
            track_gen.exec.NotCapturableError: recording a course whose producer has HOST
                stages (before any launch).
            track_gen.exec.CaptureError: recording a course that was never prepared; a
                ``stream=`` course called without ``stream=`` under another stream's
                capture; a component's ``on_generate`` / ``reset`` refused in a
                conditional body.

        In track mode, also check ``course.checkpoint_sampler.truncated`` afterwards: a
        per-env ``[E]`` int32 flag (1 if ``max_checkpoints`` clipped that env's
        checkpoint ring), mirroring ``PropSet.truncated`` for prop rings.
        """
        self._check_owner()
        body, key = self._regen_call(advance, mask)
        ctx = self._ctx(stream)
        self._ready(ctx)
        rctx = self._runner.run(body, stream=stream, sync=sync, key=key)
        if mask is not None:
            _compat.anchor(rctx.stream, mask)     # recorded: the graph reads it
        return self.result

    def enqueue_generate(self, *, stream=None, mask: "wp.array | None" = None,
                         advance: bool = False):
        """Enqueue one regeneration (as :meth:`regenerate`, same ``mask`` / ``advance``)
        without self-graphing or blocking: recorded into the caller's capture when the
        resolved stream is capturing, otherwise eager on that stream. Returns
        :attr:`result` (stream-ordered)."""
        self._check_owner()
        body, _ = self._regen_call(advance, mask)
        ctx = self._ctx(stream)
        self._ready(ctx)
        rctx = self._runner.run(body, stream=stream, policy="eager")
        if mask is not None:
            _compat.anchor(rctx.stream, mask)     # recorded: the graph reads it
        return self.result

    def generate(self, seeds: "int | wp.array | None" = None, *, stream=None,
                 sync: bool = True):
        """Optional reseed, then :meth:`regenerate`. Returns :attr:`result`.

        ``seeds`` int ``k`` reseeds every env with ``k + e`` (the same expansion as
        constructing a fresh ``PerEnvSeededRNG(seeds=k)``), expanded device-side into a
        persistent buffer: no allocation. A ``wp.array`` must be ``[E]`` int32 on the
        course device. The reseed and the regeneration run on the SAME resolved stream,
        so this whole call can be recorded into a caller's capture; note that an int
        ``k`` is then BAKED into the caller's graph (every replay reseeds with ``k``),
        while a ``wp.array`` is read at replay time (write new seeds into it to vary
        replays).

        Determinism contract: the generators are deterministic under an unchanged
        ``(seed, epoch)``. Calling ``generate()`` again WITHOUT ``seeds=`` reproduces the
        identical batch (plus a full reset); pass ``seeds=`` to reseed and get new
        courses. A reseed also zeroes every env's epoch (on device, in the same unit),
        so ``generate(seeds=s)`` reproduces a fresh course built with ``seeds=s``
        whatever was advanced before.

        Who writes the seeds follows their owner's rule: with :attr:`rng` (a config
        course, or a generator built with a ``PerEnvSeededRNG``) the reseed goes through
        ``rng.set_seeds_warp``, which overwrites ``rng.seeds_warp`` in place: when
        ``CourseConfig.seeds`` was a ``wp.array``, that IS your array (the rng binds it
        without a copy), so ``generate(seeds=...)`` rewrites it. Without an rng, owned
        seeds are set with ``SeedState.set`` (a copy, epochs zeroed); seeds bound to your
        array are never written (``RuntimeError``: write the array, then
        ``regenerate()``).

        Args:
            seeds: ``None``, an int, or an ``[E]`` int32 ``wp.array``.
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until done (default ``True``; never while capturing).

        Raises:
            TypeError: the producer has no ``SeedState`` (before any launch).
            RuntimeError: seeds bound to a caller array with no rng; a newer Course drives
                this course's producer.
        """
        if seeds is None:
            return self.regenerate(stream=stream, sync=sync)
        self._check_owner()
        if isinstance(seeds, wp.array):
            self._validate_seed_array(seeds)
        state = self.seeds
        if state is None:
            # Before any launch, with or without an rng (the reseed zeroes the epochs).
            raise TypeError("Course.generate(seeds=...): the generator has no "
                            "SeedState to reseed")
        if self.rng is None:
            if state.bound:
                raise RuntimeError(
                    "Course.generate(seeds=...): the generator's seeds are bound to a "
                    "caller array, which the course never writes (the SeedState rule): "
                    "write that array, then regenerate(); or build the generator with "
                    "seeds=PerEnvSeededRNG(...) for the 0.x reseed-in-place behaviour")
        ctx = self._ctx(stream)
        # The reseed and the regeneration form one unit: same stream, and the same
        # serialization as the regeneration itself (see GraphRunner.serial).
        with self._runner.serial(ctx):
            # Raise (before the reseed launches) where regenerate() would.
            self._runner.check_record(ctx)
            self._ready(ctx)
            if self.rng is not None:
                if isinstance(seeds, wp.array):
                    self.rng.set_seeds_warp(seeds, None, ctx=ctx)
                else:
                    ctx.launch(_seed_range_k, self._E,
                               [_wrap_int32(int(seeds)), self._reseed_buf])
                    self.rng.set_seeds_warp(self._reseed_buf, None, ctx=ctx)
                state.reset_epoch(ctx=ctx)
            else:
                if not isinstance(seeds, wp.array):
                    ctx.launch(_seed_range_k, self._E,
                               [_wrap_int32(int(seeds)), self._reseed_buf])
                    seeds = self._reseed_buf
                state.set(seeds, ctx=ctx)
            self._runner.run(self._enqueue_regenerate, stream=ctx.stream, sync=sync)
        if isinstance(seeds, wp.array):
            _compat.anchor(ctx.stream, seeds)     # recorded: the graph reads it
        return self.result

    # -- per-step ----------------------------------------------------------------------
    def step(self, stream=None) -> StepResult:
        """Every component's ``step`` (progress update, collision query, localization,
        your components), in execution order, on the bound buffers.

        Returns the same :class:`StepResult` instance every call (its entries: the
        per-step components' results).

        Args:
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is
                resolved per track_gen.exec (an enclosing torch.cuda.graph capture is
                adopted, else the course's default stream, else Warp's current stream).
                Every component launch of this call goes to that one stream.

        Raises:
            RuntimeError: a component declares a required input and :meth:`bind` was
                not called; a newer Course drives this course's producer.
            track_gen.exec.CaptureError: a ``stream=`` course called without ``stream=``
                under another stream's capture.
        """
        if self._superseded:
            self._check_owner()
        if self._needs_bind and self._bind_args is None:
            raise RuntimeError("call bind() before step()")
        ctx = self._ctx(stream)
        # At most one capture-state query for the whole step, made by the first unarmed
        # utility's rebind probe and shared (track_gen.exec.context._probe_recorded).
        token = _RESOLVED_BLOCKED.set([ctx, None])
        try:
            for comp in self._components:
                comp.step(ctx)
        finally:
            _RESOLVED_BLOCKED.reset(token)
        # The course and the binding this graph reads (a later rebind replaces the dict).
        _compat.anchor(ctx.stream, self, self._bind_args)
        return self._step_result

    def reset(self, mask: "wp.array | None" = None, stream=None) -> None:
        """Per-env respawn on the SAME course: every component's ``reset`` (progress
        state, the localizer's warm-start memory, your stateful components), in reverse
        execution order, where ``mask[e] != 0``. Collision, checkpoints and bakes derive
        from the course geometry and are unaffected by respawns.

        Args:
            mask: ``[E]`` int32 respawn mask (e.g. ``inputs["reset_mask"]``); ``None``:
                every env.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is
                resolved per track_gen.exec (an enclosing torch.cuda.graph capture is
                adopted, else the course's default stream, else Warp's current stream).
                Every component launch of this call goes to that one stream.
        """
        if self._superseded:
            self._check_owner()
        ctx = self._ctx(stream)
        self._reset_runtime(ctx, self._reset_all_mask if mask is None else mask)
        _compat.anchor(ctx.stream, self, mask)

    def _fill_posts(self, ctx=None) -> None:
        """Interleave the gate left/right posts into the disc buffer (the post collision
        component's ``on_generate``).

        Args:
            ctx: optional track_gen.exec.LaunchContext; default resolves the device's
                stream (see track_gen.exec).
        """
        ctx = _resolve_ctx(ctx, self._device)
        self._by_name["collision"].on_generate(ctx, None)

    set_capturing = staticmethod(set_capturing)


# ---------------------------------------------------------------------------------------
# The 0.x component sets and the preset helpers
# ---------------------------------------------------------------------------------------

def _components_for(cfg: CourseConfig) -> list:
    """The ordered components of a :class:`CourseConfig` (0.x order: ``on_generate`` /
    ``step`` run in this order, step: progress, collision, localizer). A defaulted
    checkpoint capacity is ``"auto"`` (the plan bound, else the construction readback);
    the bakes declare no ``size_hi`` (no resolution warning on the 0.x path)."""
    if cfg.mode == "track":
        M = cfg.max_checkpoints
        comps = [_tools.Checkpoints(cfg.checkpoint_spacing,
                                    max_checkpoints="auto" if M is None else M),
                 _tools.Progress()]
        if cfg.collision is not None:
            comps.append(_tools.BandCollision(cfg.collision, max_boxes=cfg.max_boxes,
                                              sdf_resolution=cfg.sdf_resolution or 128,
                                              size_hi=None))
        if cfg.heightfield_resolution is not None:
            comps.append(_tools.HeightFieldBake(cfg.heightfield_resolution, size_hi=None))
        return comps
    comps = [_tools.GateCheckpoints(), _tools.CenterlineFromGates(cfg.samples_per_gate),
             _tools.Progress()]
    if cfg.post_radius > 0.0:
        comps.append(_tools.DiscCollision(cfg.post_radius, max_boxes=cfg.max_boxes))
    elif cfg.frame_collision:
        comps.append(_tools.FrameCollision(cfg.agent_radius, cfg.frame_thickness,
                                           cfg.frame_depth))
    comps.append(_tools.Localizer(warm_window=cfg.localize_window, source="centerline"))
    return comps


class _Named:
    """A placeholder of a preset's own component (name and dependencies only), sorted
    with ``extra`` before anything is built."""

    layout = None

    def __init__(self, name: str, *, source: "str | None" = None, after=()) -> None:
        self.name, self.source, self.after = name, source, tuple(after)


def _precheck(own: list, extra: list) -> None:
    """The preset's checks of ``extra`` before anything is built: attached components,
    missing methods, duplicate names, unknown dependencies, cycles."""
    _check_components([c for c in own if not isinstance(c, _Named)] + list(extra))
    _order(list(own) + list(extra))


def _int_at_least(value, lo: int) -> "int | None":
    """``int(value)`` when it is an integer ``>= lo`` (not a bool / str), else ``None``."""
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v >= lo and v == value else None


def _extra_list(preset: str, extra) -> list:
    """A preset's ``extra`` as a list; a single component is a ``TypeError`` (as for
    ``Course(generator, components)``)."""
    if hasattr(extra, "attach") or isinstance(extra, (str, bytes)):
        raise TypeError(f"Course.{preset}: extra must be a sequence of components, got "
                        f"{type(extra).__name__} (pass [component])")
    return list(extra)


def _check_slot(preset: str, collision) -> None:
    """A component in the ``collision=`` slot must be named ``"collision"``."""
    if collision is None or isinstance(collision, str):
        return
    if getattr(collision, "name", None) != "collision":
        raise ValueError(f"Course.{preset}: the collision= component must be named "
                         f"'collision' (it fills StepResult.contacts and "
                         f"course.collision), got {getattr(collision, 'name', None)!r}; "
                         f"pass a collision under another name in extra=")


def _family_of_spec(spec) -> str:
    return "gates" if getattr(spec.batch, "max_gates", None) is not None else "track"


def _built(source, family: str):
    """``source`` when it is a built generator / ``Program`` of ``family`` (``"track"`` /
    ``"gates"``); ``None`` for a config or a spec of that family. ``TypeError`` for the
    other family or another type."""
    other = "Course.gates" if family == "track" else "Course.track"
    prog = _program_of(source)
    if prog is not None:
        kind = _family_of_spec(prog.spec)
        if kind != family:
            raise TypeError(f"Course.{family}: {type(source).__name__} produces "
                            f"{'gates' if kind == 'gates' else 'tracks'}: use {other}")
        return source
    if _is_spec(source):
        if _family_of_spec(source) != family:
            raise TypeError(f"Course.{family}: this PipelineSpec is a "
                            f"{_family_of_spec(source)} pipeline: use {other}")
        return None
    want = TrackGenConfig if family == "track" else GateGenConfig
    if isinstance(source, want):
        return None
    if isinstance(source, (TrackGenConfig, GateGenConfig)):
        raise TypeError(f"Course.{family}: a {type(source).__name__} configures the other "
                        f"family: use {other}")
    raise TypeError(f"Course.{family} takes a {want.__name__}, a PipelineSpec, or a built "
                    f"generator / Program, got {type(source).__name__}"
                    + (" (use Course.from_config)" if isinstance(source, CourseConfig)
                       else ""))


def _param(params, name: str):
    """``name`` of a stage's params: a resolved ``Params`` (attribute) or a spec's
    mapping (key); ``None`` when absent."""
    v = getattr(params, name, None)
    if v is None and hasattr(params, "get"):
        v = params.get(name)
    return v


def _stage_refs(source) -> list:
    """The stage refs of a spec, or of a built generator / Program's spec (``[]`` for a
    config)."""
    prog = _program_of(source)
    spec = getattr(prog, "spec", None) if prog is not None else source
    return list(getattr(spec, "stages", None) or ())


def _gate_width_value(source) -> "float | None":
    """The construction ``gate_frames.gate_width`` of a ``GateGenConfig``, a spec or a
    built generator / Program, read before anything is built (``None``: none)."""
    if isinstance(source, GateGenConfig):
        return float(source.gate_width)
    prog = _program_of(source)
    if prog is not None:
        frames = prog.params["gate_frames"] if "gate_frames" in prog.params else None
        if frames is None or "gate_width" not in frames:
            return None
        w = getattr(frames.params, "gate_width", None)
        return None if w is None else float(w)
    for ref in getattr(source, "stages", ()) or ():
        if ref.kind == "gate_frames":
            w = _param(ref.params, "gate_width")
            return None if w is None else float(w)
    return None


def _gate_width_check(source) -> None:
    """``gate_frames.gate_width > 0`` before anything is built: when a randomizer draws
    the width every generation (roadmap PR13d), its construction ``lo`` must be ``> 0``
    (the construction width is never used); otherwise the construction width."""
    for ref in _stage_refs(source):
        p = ref.params
        if ref.kind == "randomize" and _param(p, "stage") == "gate_frames" and \
                _param(p, "field") == "gate_width":
            lo = _param(p, "lo")
            if lo is not None and not float(lo) > 0.0:
                raise ValueError(
                    f"Course.gates requires the gate_frames.gate_width randomizer's lo > 0, "
                    f"got lo={lo!r}: a width-0 gate has a degenerate crossing segment and "
                    f"can never be passed")
            return
    w = _gate_width_value(source)
    if w is not None and not w > 0.0:
        raise ValueError("Course.gates requires gate_frames.gate_width > 0: a width-0 gate "
                         "has a degenerate crossing segment and can never be passed (pass "
                         "gate_width= to the config or presets.gates, or randomize it with "
                         "lo > 0)")


def _narrow_gate_width(gen) -> None:
    """Refuse a later scalar ``set`` of ``gate_frames.gate_width <= 0`` (roadmap PR13d:
    the width is per env; an array written later is not range-checked: a value ``<= 0``
    is used as 0, and those envs' gates can never be passed)."""
    params = getattr(gen, "params", None)
    frames = params.get("gate_frames") if params is not None and \
        hasattr(params, "get") else None
    if frames is not None and "gate_width" in frames:
        frames._narrow("gate_width", _F32_MIN_POSITIVE, None,
                       "> 0 in Course gates mode (a width-0 gate has a "
                       "degenerate crossing segment and can never be passed)")
