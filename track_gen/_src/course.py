"""Unified course facade: generation + collision + progress in one object.

``Course`` bundles the runtime utilities per mode and owns the orchestration
invariants that are otherwise the caller's burden:

- ``mode="track"``: TrackGenerator -> out-of-bounds ``CollisionChecker``
  (``"segments"`` / ``"sdf"`` / ``None``) -> ``CheckpointSampler`` ->
  ``ProgressTracker``.
- ``mode="gates"``: GateGenerator -> ``CheckpointSet.from_gates`` ->
  ``ProgressTracker``, plus a ``CourseLine`` (3D spline centerline through
  the gates) with a ``TrackLocalizer`` bound to it (``step()`` returns the
  per-env :class:`TrackFrame`); optional ``DiscChecker`` gate-post collision
  (``post_radius > 0``), with the posts array and the line rebuilt
  device-side on every regeneration.

Lifecycle: construct (generates the first batch and builds every sub-tool)
-> ``bind()`` (stable sim buffers, required for ``step()``) -> per-step
``step()`` / per-env ``reset(mask)``, with ``regenerate()`` /
``generate(seeds=...)`` whenever new courses are wanted. Per-env new courses:
``regenerate(mask=done, advance=True)`` advances the epoch of the masked envs
on device (see :mod:`track_gen.seeds`), regenerates, refreshes the derived
sub-tools and resets the masked envs' runtime state (progress, localizer); the
other envs keep their course and state. The pipelines are fixed-shape, so
every env is recomputed, and the unmasked ones (which keep the effective seed
their course was built from) are recomputed to identical values.

CUDA graphs: regeneration is ONE body (generator pipeline + coherent
refresh: checkpoint resample / sdf + heightfield bake / posts / line, then a
progress reset: full, or masked for ``regenerate(mask=...)``). Inside a
caller's capture ``regenerate()`` / ``generate()`` record it into the
caller's graph next to ``step()`` / ``reset()`` (an RL auto-reset loop
``[physics] -> step() -> done mask -> regenerate(mask=done, advance=True)``
replays entirely on device); outside a capture the course replays its own
graph (:attr:`Course.graph`, one more per ``(advance, mask)`` variant) on
CUDA. A masked regeneration is skipped on device when no env is masked,
where CUDA supports it (see :meth:`Course.regenerate`).
``step()``/``reset()`` are never
self-captured: they are capture-ready for the caller's sim graph as-is (no
utility host-synchronizes; the former ``track_gen.set_capturing`` is a
deprecated no-op). Every launch of one call goes to ONE stream: ``stream=``
when given, else an enclosing ``torch.cuda.graph`` capture stream, else
Warp's current stream (see :mod:`track_gen.exec`).

Since roadmap PR14 the sub-tools are built and driven through the component adapters
of :mod:`track_gen.tools` (``course._components``, private; the public attributes keep
their types), and an env with ``valid[e] == 0`` on ``course.result`` reads like a
poisoned one in every sub-tool (``oob = 1`` with NaN geometry, no post hits, frozen
progress with ``dist_to_next = 0``, NaN localization, ``checkpoints.count = 0``):
gate rewards and observations on :attr:`StepResult.valid` and give such envs a new
course (``regenerate(mask=done, advance=True)``).
"""
from __future__ import annotations

import functools
import math
import types
import warnings
from dataclasses import dataclass

import warp as wp

from ..exec.context import (LaunchContext, _rebind_close, _rebind_open,
                            resolve as _resolve_ctx)
from ..exec.runner import GraphRunner, ensure_not_capturing
from ..ops.reduce import any_set
from .. import tools as _tools
from .checkpoints import CheckpointSampler, CheckpointSet, _readback_max_checkpoints
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


@dataclass
class StepResult:
    """Per-step bundle returned by :meth:`Course.step`.

    Holds the sub-tools' preallocated in-place result instances — the SAME
    ``StepResult`` (and the same underlying buffers) is returned on every
    ``step()``; use ``clone()`` for snapshots.

    Attributes
    ----------
    events : ProgressEvents
        Progress events for this step.
    contacts : BoxContact or DiscContact or FrameContact or None
        Collision result (``None`` when the course has no collision checker).
    frame : TrackFrame or None
        Localization frame on the gates-mode :class:`CourseLine` (``None``
        in track mode).
    valid : wp.array or None
        ``[E]`` int32 validity of each env's course (roadmap PR14): an alias of
        ``course.result.valid`` (``None`` only on a hand-built instance). The other
        fields of an env with ``valid[e] == 0`` hold the invalid-env sentinels (see
        :class:`Course`): gate rewards and observations on it with a select
        (``where(valid, x, 0)``, never a multiply: ``NaN * 0`` is NaN) and end such
        episodes. It is producer state and live: never write into it (exclude it from
        code that zeroes or poisons the step outputs; ``dataclasses.fields`` lists it),
        and read it before a ``regenerate`` in the same iteration, which rewrites it
        for the regenerated envs (or ``clone()`` the result).
    """

    events: ProgressEvents
    contacts: "BoxContact | DiscContact | FrameContact | None"
    frame: "TrackFrame | None"
    valid: "wp.array | None" = None

    def clone(self) -> "StepResult":
        """Deep-copy the sub-results (and ``valid``)."""
        return StepResult(
            events=self.events.clone(),
            contacts=None if self.contacts is None else self.contacts.clone(),
            frame=None if self.frame is None else self.frame.clone(),
            valid=None if self.valid is None else wp.clone(self.valid),
        )


class Course:
    """One object bundling generation, collision, and progress per mode.

    Lifecycle: construct -> :meth:`bind` (stable sim buffers; required for
    :meth:`step`) -> per-step :meth:`step` / per-env :meth:`reset`, plus
    whole-batch :meth:`regenerate` / :meth:`generate` for new courses.
    Construction generates the first batch and builds every sub-tool (their
    auto-derivations need a real batch, read back on the host once), then
    runs one real warm-up regeneration; all of it must happen outside any
    CUDA graph capture. The sub-tools are reachable as attributes right
    after construction (``generator``, ``rng``, ``result``, ``collision``,
    ``checkpoints``, ``checkpoint_sampler``, ``progress``, an optional
    track-mode ``heightfield`` (when ``heightfield_resolution`` is set), and
    — gates mode — ``course_line``, ``localizer``).

    Regeneration recomputes the whole batch (fixed-batch pipelines); per-env
    NEW courses come from :meth:`regenerate`'s ``mask`` + ``advance`` (only
    the masked envs change and are reset), per-env respawns on the SAME
    course from :meth:`reset`'s mask.

    Invalid envs (``valid[e] == 0`` on :attr:`result`; roadmap PR14): every sub-tool
    answers as for a poisoned env (``course.checkpoints.count = 0`` with NaN slots,
    ``oob = 1`` with NaN geometry, no post hit, frozen progress with ``dist_to_next =
    0``, NaN ``s`` / ``n``, a NaN heightfield with the box ``(-1, -1)-(1, 1)``).
    :attr:`StepResult.valid` carries the mask: set ``done |= 1 - valid`` and give those
    envs a NEW course with ``regenerate(mask=done, advance=True)`` (``reset`` would
    respawn them on the same invalid course).

    The sub-tools are built and driven through the adapters of :mod:`track_gen.tools`
    (``_components``, private; ``Course(generator, components)`` is roadmap PR15).
    """

    def __init__(self, config: CourseConfig) -> None:
        _init()
        self._cfg = config
        self._E = int(config.gen.num_envs)
        # Canonicalize via Warp: "cuda" -> "cuda:0" so the string matches
        # str(arr.device) in _check_arr (bind/seed validation would otherwise
        # reject cuda:0 buffers when the config used the "cuda" alias).
        device = wp.get_device(config.gen.device)
        self._device = str(device)
        self._is_cuda = device.is_cuda
        if isinstance(config.seeds, wp.array):
            self._validate_seed_array(config.seeds)
        # Refuse a capture BEFORE the first allocation (the rng's).
        ensure_not_capturing(device, label=f"course[{config.gen.generator}]")
        self.rng = PerEnvSeededRNG(seeds=config.seeds, num_envs=self._E,
                                   device=self._device)
        # The generator prepares itself (one real generation), so its result
        # already holds the batch for the initial seeds.
        if config.mode == "track":
            self.generator = TrackGenerator(config.gen, self.rng)
            self.result: "Track | GateSequence" = self.generator._track
        else:
            self.generator = GateGenerator(config.gen, self.rng)
            self.result = self.generator._gate_sequence
            # Roadmap PR13d: gate_frames.gate_width is per env; keep the construction
            # check's invariant for a scalar set() (an array is not range-checked: a
            # value <= 0 is used as 0, and those envs' gates can never be passed).
            frames = self.generator.params.get("gate_frames")
            if frames is not None and "gate_width" in frames:
                frames._narrow("gate_width", _F32_MIN_POSITIVE, None,
                               "> 0 in Course gates mode (a width-0 gate has a "
                               "degenerate crossing segment and can never be passed)")

        self.collision: "CollisionChecker | DiscChecker | FrameChecker | None" \
            = None
        self.course_line: "CourseLine | None" = None
        self.localizer: "TrackLocalizer | None" = None
        self.heightfield: "HeightFieldBaker | None" = None
        self.checkpoints: "CheckpointSet | None" = None
        self.checkpoint_sampler: "CheckpointSampler | None" = None
        self.progress: "ProgressTracker | None" = None
        self._posts: "wp.array | None" = None
        self._bind_args: "dict | None" = None
        self._step_result: "StepResult | None" = None
        self._reset_all_mask = wp.full(self._E, 1, dtype=wp.int32,
                                       device=self._device)
        # Persistent [E] int32 buffer for generate(seeds=int): the int is
        # expanded device-side into it, so a reseed never allocates.
        self._reseed_buf = wp.empty(self._E, dtype=wp.int32,
                                    device=self._device)
        # Persistent device input (see `inputs`) and the [1] int32 "any env
        # masked" flag of a masked regeneration (written on device).
        self._regen_mask = wp.zeros(self._E, dtype=wp.int32, device=self._device)
        self._regen_flag = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._inputs = types.MappingProxyType({"regen_mask": self._regen_mask})

        gen_runner = self.generator._runner
        # A Course over a pipeline with HOST stages runs eagerly (it does not segment)
        # and refuses a caller's capture naming them.
        self._runner = GraphRunner(
            self._device,
            capturable=gen_runner.capturable,
            legacy_stream=gen_runner.legacy_stream,
            owner=self,
            name=f"course[{config.gen.generator}]",
            _host_ids=gen_runner._host_ids,
        )
        # Sub-tool construction (allocation + readback-based sizing) happens
        # here, cold, never inside a capture. The generator's construction
        # already blocked until its warm-up pass finished, so the readbacks see
        # the first batch.
        ctx = self._runner.context()
        with ctx.scope():
            self._build_subtools(ctx)
            # Load the reseed kernel's module now (writes only the scratch
            # buffer), so a first generate(seeds=int) never loads a module
            # inside a caller's capture.
            ctx.launch(_seed_range_k, self._E, [0, self._reseed_buf])
            # Same for the masked regeneration's reduction (writes only the
            # flag; the mask is all zeros).
            any_set(ctx, self._regen_mask, self._regen_flag)
        # One real warm-up regeneration (generator + refresh) outside any
        # capture: loads every module the regeneration body uses.
        self._runner.prepare(self._enqueue_regenerate)
        self._warn_clipped_default()

    @property
    def graph(self) -> "wp.Graph | None":
        """The course's own regeneration graph (generator pipeline + refresh
        as ONE graph) for a plain :meth:`regenerate` / :meth:`generate`:
        captured on the first such CUDA call made outside any capture,
        replayed by later ones. ``None`` on CPU, for a generator pipeline
        with HOST stages, and before that call. Each ``(advance, mask)`` variant of
        :meth:`regenerate` caches one more graph."""
        return self._runner.graph

    @property
    def seeds(self):
        """The generator's :class:`~track_gen.seeds.SeedState` (the rng's
        seeds bound by pointer, plus the per-env epoch)."""
        return self.generator.seeds

    @property
    def inputs(self) -> "types.MappingProxyType":
        """Persistent device inputs (stable pointers, allocated at
        construction):

        - ``"regen_mask"``: ``[E]`` int32, zero-initialised. Write the envs
          to regenerate from your own kernels (e.g. ``done`` flags inside your
          sim graph) and pass it as ``regenerate(mask=course.inputs
          ["regen_mask"], advance=True)``. Its pointer never changes, so the
          course's self-graph for it is captured once.
        """
        return self._inputs

    # -- validation ------------------------------------------------------

    def _check_arr(self, name: str, arr, shape: tuple, dtype) -> None:
        """Validate a bound/seed wp.array's shape, dtype, and device
        (thin wrapper over :func:`runtime._check_arr`, pinned to this
        course's device)."""
        _check_arr(name, arr, shape, dtype, self._device)

    def _validate_seed_array(self, seeds) -> None:
        """A wp.array seeds must be ``[E]`` int32 on the course device.

        ``set_seeds_warp`` launches with ``dim=len(seeds)`` against the ``[E]``
        state arrays: a longer array corrupts device memory past the states, a
        shorter one only partially reseeds.
        """
        self._check_arr("seeds", seeds, (self._E,), wp.int32)

    def _needs_boxes(self) -> bool:
        # Frame collision binds a sphere (position only) — no oriented-box
        # yaw/half_extents buffers — so it is deliberately excluded here.
        cfg = self._cfg
        if cfg.mode == "gates" and cfg.frame_collision:
            return False
        return (cfg.mode == "track" and cfg.collision is not None) or \
               (cfg.mode == "gates" and cfg.post_radius > 0.0)

    def _validate_bind_args(self, a: dict) -> None:
        """Eagerly validate bound buffers — E, max_boxes, device are all
        known at construction, so shape/dtype/device errors surface at
        ``bind()`` rather than at the first ``step()``."""
        E = self._E
        self._check_arr("position", a["position"], (E,), wp.vec3f)
        if self._needs_boxes():
            nb = (E * self._cfg.max_boxes,)
            self._check_arr("orientation", a["orientation"], nb, wp.quatf)
            self._check_arr("half_extents", a["half_extents"], nb, wp.vec2f)
            if self._cfg.max_boxes > 1:
                self._check_arr("box_position", a["box_position"], nb, wp.vec3f)

    # -- binding ---------------------------------------------------------

    def bind(self, position: wp.array, orientation: "wp.array | None" = None,
             half_extents: "wp.array | None" = None,
             box_position: "wp.array | None" = None) -> None:
        """Bind stable sim buffers (required before :meth:`step`).

        ``position`` is the ``[E]`` vec3f agent-position buffer driving
        progress. When a box-collision checker is enabled, ``orientation``
        (``[E * max_boxes]`` quatf box poses) and
        ``half_extents`` (``[E * max_boxes]`` vec2f, planar boxes) are
        required too; with
        ``max_boxes == 1`` the same ``position`` buffer serves as the box
        positions, otherwise pass a separate ``box_position``
        ``[E * max_boxes]`` vec3f buffer. May be called any time after
        construction (before or after a regeneration); rebinding replaces the
        previous binding.

        Do NOT rebind after capturing ``step()`` into a sim graph: the
        captured graph replays against the buffer pointers live at capture
        time, so a later rebind leaves it reading the old buffers (silently
        divergent results) — keep writing into the SAME bound buffers and
        rebind only before (re)capturing. Since roadmap PR14 such a rebind
        (other arrays, after ``step()`` ran while its stream or a CPU graph
        capture was capturing) warns once,
        :class:`~track_gen.exec.RebindAfterCaptureWarning`; rebinding the same
        arrays is silent. The recipe "rebind, then record again" warns once
        too: filter the category when you re-record on purpose.
        """
        token, hits = _rebind_open()
        try:
            self._bind(position, orientation, half_extents, box_position)
        finally:
            _rebind_close(token, hits, stacklevel=2)

    def _bind(self, position, orientation, half_extents, box_position) -> None:
        needs_boxes = self._needs_boxes()
        if needs_boxes and (orientation is None or half_extents is None):
            raise RuntimeError(
                "this course has a collision checker: bind orientation and "
                "half_extents as well")
        if needs_boxes and self._cfg.max_boxes > 1 and box_position is None:
            raise RuntimeError(
                "max_boxes > 1: bind a separate box_position "
                "[E*max_boxes] buffer")
        args = {"position": position, "orientation": orientation,
                "half_extents": half_extents,
                "box_position": box_position}
        self._validate_bind_args(args)
        self._bind_args = args
        if self.progress is not None:
            self._apply_bind()

    def _apply_bind(self) -> None:
        """Route the bound arrays to every component by its declared inputs (a name that
        is only an input's ``default`` is routed by ``Component.bind``: ``box_position``
        defaults to ``position`` with one box per env)."""
        a = self._bind_args
        if a is None:
            return
        given = {k: v for k, v in a.items() if v is not None}
        for comp in self._components:
            if not comp.inputs:
                continue
            routes = set(comp.inputs) | {s.default for s in comp.inputs.values()
                                         if s.default is not None}
            comp.bind(**{k: v for k, v in given.items() if k in routes})

    # -- generation + refresh --------------------------------------------

    def _enqueue_regenerate(self, ctx: LaunchContext, *, advance: bool = False,
                            mask: "wp.array | None" = None) -> None:
        """The ONE regeneration body, all on ``ctx``.

        ``mask is None``: [advance every env] -> generator pipeline (seed
        prologue included) -> the coherent downstream refresh with a FULL
        runtime reset. With a ``mask``: ``any_set(mask)`` into a device flag,
        then, skipped on device when no env is masked where CUDA supports it
        (:meth:`LaunchContext.skip_unless`), [advance the masked envs] ->
        pipeline -> refresh -> runtime reset of the masked envs only.

        Run by ``prepare`` (construction), by the eager path, by the
        self-graph capture and by a caller's recording alike. Zero
        allocation, host-sync free.
        """
        if mask is None:
            self.generator._enqueue(ctx, advance=advance)
            self._refresh(ctx)
            return
        any_set(ctx, mask, self._regen_flag)
        ctx.skip_unless(self._regen_flag, functools.partial(
            self._enqueue_regenerate_masked, advance=advance, mask=mask))

    def _enqueue_regenerate_masked(self, ctx: LaunchContext, *, advance: bool,
                                   mask: wp.array) -> None:
        """Masked regeneration body (an exact no-op when every ``mask[e]`` is
        0: nothing advances, no effective seed is re-derived, so every env is
        recomputed from the effective seed its current course was built from,
        and nothing is reset)."""
        self.generator._enqueue(ctx, advance=advance, mask=mask)
        self._refresh_geometry(ctx, mask)
        self._reset_runtime(ctx, mask)

    def _regen_call(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one regeneration; validates ``mask``
        host-side, before any launch."""
        if mask is not None:
            self.seeds.check_mask(mask)   # [E] int32, contiguous, course device
        advance = bool(advance)
        if mask is None and not advance:
            return self._enqueue_regenerate, None
        key = (advance, 0 if mask is None else int(mask.ptr))
        return functools.partial(self._enqueue_regenerate, advance=advance,
                                 mask=mask), key

    def regenerate(self, *, mask: "wp.array | None" = None, advance: bool = False,
                   stream=None, sync: bool = True):
        """Regenerate courses plus a coherent downstream refresh; returns
        :attr:`result`.

        ``mask=None`` (whole batch): the generator pipeline from the current
        seeds (after advancing EVERY env's epoch when ``advance=True``), then
        checkpoint resample / sdf and heightfield bake / posts and line
        rebuild as applicable, and a FULL progress (and localizer) reset —
        every course changed (or, without ``advance``, was rebuilt
        identically).

        ``mask`` given (``[E]`` int32 on the course device, read on device):
        envs with ``mask[e] != 0`` advance their epoch (when ``advance=True``)
        and get the course of their live ``(seed, epoch)``, and their runtime
        state is reset (``progress.reset(mask)``, ``localizer.reset(mask)``);
        envs with ``mask[e] == 0`` keep their course and runtime state. The
        pipeline is fixed-shape, so it still recomputes every env: unmasked
        envs keep the effective seed their current course was built from and
        are recomputed to identical values, even if their seed or epoch changed
        since (``rng.set_seeds_warp``, ``seeds.reset_epoch()``, a write to the
        seed array): such a change reaches an env at its next masked
        regeneration, or every env at the next one without a mask. On a track
        course the generator's commit also restores an unmasked env's ``Track``
        rows bit for bit after the recompute (``track_gen.stages.commit``; on a
        gate course its ``GateSequence`` rows, since roadmap PR11). Without ``advance`` a
        mask respawns the masked envs (as :meth:`reset`) on the course of their
        current ``(seed, epoch)``: the same one unless their seed was
        rewritten.

        Conditional skip: the masked body runs after a device-side
        ``any_set(mask)`` under :meth:`track_gen.exec.LaunchContext.skip_unless`.
        Inside a Warp-registered capture (Warp ``ScopedCapture``, the course's
        own self-graph, ``capture_begin(external=True)``, or
        ``track_gen.interop.torch.registered(conditional=True)`` under
        ``torch.cuda.graph``) with CUDA Toolkit and driver >= 12.4, it is
        skipped on device when no env is masked; the caller's graph then holds
        a conditional node, so it can no longer be used as a CUDA child graph.
        Elsewhere it runs unconditionally, which is correct because an all-zero
        mask makes it a no-op: nothing advances, nothing is reset, and every
        env is recomputed from the effective seed its current course was built
        from (bitwise identical as long as the pipeline is deterministic; its
        only float atomics, the offset stage's shoelace areas, feed a
        comparison, not an output).

        Execution (stream resolved per :mod:`track_gen.exec`):

        - inside a capture on the resolved stream: RECORDED into the
          caller's graph (e.g. next to :meth:`step` / :meth:`reset`); each
          replay regenerates from the seeds, epochs and mask at replay time
          (and, with ``advance``, advances again);
        - outside a capture on CUDA: replay of a self-graph (generator +
          refresh as ONE graph, captured on the first call of each
          ``(advance, mask)`` variant; :attr:`graph` is the plain one);
        - on CPU, or with a generator pipeline that has HOST stages (a
          ``capturable=False`` generator): eager (a Course does not segment).

        Args:
            mask: optional contiguous ``[E]`` int32 regeneration mask. Pass a
                STABLE buffer (e.g. ``inputs["regen_mask"]``): outside a capture
                each distinct mask pointer captures, and caches, one more
                self-graph (up to a bound, past which new pointers run
                eagerly).
            advance: advance the epoch (of every env, or of the masked envs)
                on device first, so those envs get new courses. Default
                ``False`` (rebuild from unchanged seeds and epochs).
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until the regeneration finished on the
                resolved stream (default ``True``; never while capturing, nor
                while torch is capturing in the calling thread).

        Raises:
            ValueError: ``mask`` is not a contiguous ``[E]`` int32 array on the
                course device (before any launch).
            track_gen.exec.NotCapturableError: recording a course whose
                generator pipeline has HOST stages (before any launch).

        In track mode, also check ``course.checkpoint_sampler.truncated``
        afterwards: a per-env ``[E]`` int32 flag (1 if ``max_checkpoints``
        clipped that env's checkpoint ring), mirroring ``PropSet.truncated``
        for prop rings.
        """
        body, key = self._regen_call(advance, mask)
        self._runner.run(body, stream=stream, sync=sync, key=key)
        return self.result

    def enqueue_generate(self, *, stream=None, mask: "wp.array | None" = None,
                         advance: bool = False):
        """Enqueue one regeneration (as :meth:`regenerate`, same ``mask`` /
        ``advance``) without self-graphing or blocking: recorded into the
        caller's capture when the resolved stream is capturing, otherwise
        eager on that stream. Returns :attr:`result` (stream-ordered)."""
        body, _ = self._regen_call(advance, mask)
        self._runner.run(body, stream=stream, policy="eager")
        return self.result

    def generate(self, seeds: "int | wp.array | None" = None, *, stream=None,
                 sync: bool = True):
        """Optional reseed, then :meth:`regenerate`. Returns :attr:`result`.

        ``seeds`` int ``k`` reseeds every env with ``k + e`` (the same
        expansion as constructing a fresh ``PerEnvSeededRNG(seeds=k)``),
        expanded device-side into a persistent buffer: no allocation. A
        ``wp.array`` must be ``[E]`` int32 on the course device. The reseed
        and the regeneration run on the SAME resolved stream, so this whole
        call can be recorded into a caller's capture; note that an int ``k``
        is then BAKED into the caller's graph (every replay reseeds with
        ``k``), while a ``wp.array`` is read at replay time (write new seeds
        into it to vary replays).

        Determinism contract: the generators are deterministic under an
        unchanged ``(seed, epoch)``. Calling ``generate()`` again WITHOUT
        ``seeds=`` reproduces the identical batch (plus a full progress reset);
        pass ``seeds=`` to reseed and get new courses. A reseed also zeroes
        every env's epoch (on device, in the same unit), so
        ``generate(seeds=s)`` reproduces a fresh ``Course`` built with
        ``seeds=s`` whatever was advanced before.

        The reseed goes through ``course.rng.set_seeds_warp``, which overwrites
        ``course.rng.seeds_warp`` in place: when ``CourseConfig.seeds`` was a
        ``wp.array``, that IS your array (the rng binds it without a copy), so
        ``generate(seeds=...)`` rewrites it.

        Args:
            seeds: ``None``, an int, or an ``[E]`` int32 ``wp.array``.
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until done (default ``True``; never while
                capturing).
        """
        if seeds is None:
            return self.regenerate(stream=stream, sync=sync)
        if isinstance(seeds, wp.array):
            self._validate_seed_array(seeds)
        ctx = self._runner.context(stream)
        # The reseed and the regeneration form one unit: same stream, and the
        # same serialization as the regeneration itself (see GraphRunner.serial).
        with self._runner.serial(ctx):
            # Raise (before the reseed launches) where regenerate() would.
            self._runner.check_record(ctx)
            if isinstance(seeds, wp.array):
                self.rng.set_seeds_warp(seeds, None, ctx=ctx)
            else:
                ctx.launch(_seed_range_k, self._E,
                           [_wrap_int32(int(seeds)), self._reseed_buf])
                self.rng.set_seeds_warp(self._reseed_buf, None, ctx=ctx)
            self.generator.seeds.reset_epoch(ctx=ctx)
            self._runner.run(self._enqueue_regenerate, stream=ctx.stream, sync=sync)
        return self.result

    def _build_subtools(self, ctx: LaunchContext) -> None:
        """Build the sub-tools: the ordered component adapters of
        :func:`_components_for`, attached through a
        :class:`~track_gen.tools.ComponentContext` of the generator (its plan's static
        capacities) on the construction context (construction only, never inside a
        capture); the legacy attributes are set from them.

        Args:
            ctx: the construction launch context (the stream the generator's
                warm-up pass ran on).
        """
        cfg = self._cfg
        cc = _tools.ComponentContext.for_producer(self.generator, ctx=ctx)
        M = cfg.max_checkpoints
        self._cp_default = None
        if cfg.mode == "track" and M is None:
            L = cc.capacity.get("max_loop_length")
            if L is not None:
                M = max(3, math.ceil(float(L) / float(cfg.checkpoint_spacing)))
                self._cp_default = "the plan bound ceil(max_loop_length / " \
                    "checkpoint_spacing)"
            else:   # the plan declares no bound: read the first batch back (as before)
                lay = self.result.layout
                M = _readback_max_checkpoints(self.result, int(lay.num_envs),
                                              int(lay.stride),
                                              float(cfg.checkpoint_spacing))
                self._cp_default = "read back from the construction batch"
        self._components = []
        for comp in _components_for(cfg, M):
            cc.add(comp, self.result)
            self._components.append(comp)
        by = {c.name: c for c in self._components}
        self._by_name = by
        self.progress = by["progress"].utility
        self.checkpoints = by["checkpoints"].result
        self.checkpoint_sampler = by["checkpoints"].utility   # None in gates mode
        coll = by.get("collision")
        self.collision = None if coll is None else coll.utility
        self._posts = None if coll is None else getattr(coll, "posts", None)
        self.heightfield = by["heightfield"].utility if "heightfield" in by else None
        self.course_line = by["centerline"].utility if "centerline" in by else None
        self.localizer = by["localizer"].utility if "localizer" in by else None
        # Runtime resets in the 0.x order: the localizer's warm memory, then progress.
        self._reset_order = [by[n] for n in ("localizer", "progress") if n in by]
        self._apply_bind()

    def _warn_clipped_default(self) -> None:
        """After the warm-up regeneration (synced): warn once when a DEFAULTED
        checkpoint capacity clips a valid env of the first batch (one cold readback of
        two ``[E]`` int32 arrays; never with an explicit ``max_checkpoints``)."""
        if self._cp_default is None:
            return
        tr = self.checkpoint_sampler.truncated.numpy()
        va = self.result.valid.numpy()
        n = int(((tr != 0) & (va != 0)).sum())
        if n:
            warnings.warn(
                f"Course: the default max_checkpoints="
                f"{self.checkpoint_sampler.max_checkpoints} ({self._cp_default}) clips "
                f"the checkpoint ring of {n} valid env(s) of the first batch "
                f"(checkpoint_sampler.truncated); pass a larger CourseConfig."
                f"max_checkpoints", UserWarning, stacklevel=_tools._component.
                _user_stacklevel())

    def _fill_posts(self, ctx=None) -> None:
        """Interleave the gate left/right posts into the disc buffer (the post
        collision adapter's ``on_generate``).

        Args:
            ctx: optional track_gen.exec.LaunchContext; default resolves the
                device's stream (see track_gen.exec).
        """
        ctx = _resolve_ctx(ctx, self._device)
        self._by_name["collision"].on_generate(ctx, None)

    def _refresh(self, ctx: LaunchContext) -> None:
        """Post-generation coherence: resample/bake/posts + full reset.

        Args:
            ctx: the regeneration's launch context; every sub-tool runs on
                ``ctx.stream``.
        """
        self._refresh_geometry(ctx)
        self._reset_runtime(ctx, self._reset_all_mask)

    def _refresh_geometry(self, ctx: LaunchContext, mask=None) -> None:
        """Rebuild every sub-tool buffer DERIVED from the course geometry
        (checkpoint resample, course line, sdf / heightfield bake, posts): each
        component's ``on_generate``, in list order, on ``ctx.stream``. Each recomputes
        the whole batch: an env whose geometry did not change gets bitwise-identical
        derived buffers."""
        for comp in self._components:
            comp.on_generate(ctx, mask)

    def _reset_runtime(self, ctx: LaunchContext, mask: wp.array) -> None:
        """Reset every stateful sub-tool's per-env runtime state where
        ``mask[e] != 0`` (the localizer's warm-start memory, then progress), on
        ``ctx.stream``. The other sub-tools derive from the geometry."""
        for comp in self._reset_order:
            comp.reset(ctx, mask)

    # -- per-step ----------------------------------------------------------

    def step(self, stream=None) -> StepResult:
        """Progress update + collision query on the bound buffers.

        Args:
            stream: optional wp.Stream (or 'torch') to run on; by default the
                stream is resolved per track_gen.exec (an enclosing
                torch.cuda.graph capture is adopted, else Warp's current
                stream).
                Every sub-tool launch of this call goes to that one stream.
        """
        if self.progress is None:
            raise RuntimeError("call generate() before step()")
        if self._bind_args is None:
            raise RuntimeError("call bind() before step()")
        ctx = LaunchContext.for_call(self._device, stream)
        for comp in self._components:     # progress, collision, localizer (list order)
            comp.step(ctx)
        if self._step_result is None:
            by = self._by_name
            self._step_result = StepResult(
                events=by["progress"].result,
                contacts=by["collision"].result if "collision" in by else None,
                frame=by["localizer"].result if "localizer" in by else None,
                valid=self.result.valid)
        return self._step_result

    def reset(self, mask: wp.array, stream=None) -> None:
        """Per-env respawn on the SAME course: clears progress state (and,
        in gates mode, the localizer's warm-start memory) where
        ``mask[e] == 1``. Collision and checkpoints derive from the course
        geometry and are unaffected by respawns.

        Args:
            mask: ``[E]`` int32 respawn mask.
            stream: optional wp.Stream (or 'torch') to run on; by default the
                stream is resolved per track_gen.exec (an enclosing
                torch.cuda.graph capture is adopted, else Warp's current
                stream).
                Every sub-tool launch of this call goes to that one stream.
        """
        if self.progress is None:
            raise RuntimeError("call generate() before reset()")
        self._reset_runtime(LaunchContext.for_call(self._device, stream), mask)

    set_capturing = staticmethod(set_capturing)


def _components_for(cfg: CourseConfig, max_checkpoints: "int | None") -> list:
    """The ordered component adapters of a :class:`CourseConfig` (the roadmap PR15
    seam). ``on_generate`` and ``step`` run in this order (step: progress, collision,
    localizer, as in 0.x)."""
    if cfg.mode == "track":
        comps = [_tools.Checkpoints(cfg.checkpoint_spacing,
                                    max_checkpoints=max_checkpoints),
                 _tools.Progress()]
        if cfg.collision is not None:
            comps.append(_tools.BandCollision(cfg.collision, max_boxes=cfg.max_boxes,
                                              sdf_resolution=cfg.sdf_resolution or 128))
        if cfg.heightfield_resolution is not None:
            comps.append(_tools.HeightFieldBake(cfg.heightfield_resolution))
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
