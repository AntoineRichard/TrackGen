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
``step()`` / per-env ``reset(mask)``, with whole-batch ``regenerate()`` /
``generate(seeds=...)`` whenever new courses are wanted. Whole-batch
generation is a generator constraint (the pipelines are fixed-batch);
per-env control lives in ``reset(mask)``.

CUDA graphs: regeneration is ONE body (generator pipeline + coherent
refresh: checkpoint resample / sdf + heightfield bake / posts / line, then a
full progress reset). Inside a caller's capture ``regenerate()`` /
``generate()`` record it into the caller's graph next to ``step()`` /
``reset()``; outside a capture the course replays its own graph
(:attr:`Course.graph`) on CUDA. ``step()``/``reset()`` are never
self-captured: they are capture-ready for the caller's sim graph as-is (no
utility host-synchronizes; the former ``track_gen.set_capturing`` is a
deprecated no-op). Every launch of one call goes to ONE stream: ``stream=``
when given, else an enclosing ``torch.cuda.graph`` capture stream, else
Warp's current stream (see :mod:`track_gen.exec`).

Results are undefined for envs with ``valid[e] == 0`` on ``course.result``.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..exec.runner import GraphRunner, ensure_not_capturing
from .checkpoints import CheckpointSampler, CheckpointSet
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
def _interleave_posts_k(
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    posts: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()             # dim = E * max_gates; NaN padding carries over
    posts[2 * i] = left[i]
    posts[2 * i + 1] = right[i]


@wp.kernel
def _seed_range_k(base: wp.int32, out: wp.array(dtype=wp.int32)):
    # out[e] = base + e with int32 wrap-around: PerEnvSeededRNG's int expansion
    # (seed + arange(E), cast to int32). Unsigned add: no signed-overflow UB.
    e = wp.tid()
    out[e] = wp.int32(wp.uint32(base) + wp.uint32(e))


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
        ``gen.gate_width > 0`` (a width-0 gate can never be crossed).
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
        Track mode only: forwarded to ``CheckpointSampler``.
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
    """

    events: ProgressEvents
    contacts: "BoxContact | DiscContact | FrameContact | None"
    frame: "TrackFrame | None"

    def clone(self) -> "StepResult":
        """Deep-copy the sub-results."""
        return StepResult(
            events=self.events.clone(),
            contacts=None if self.contacts is None else self.contacts.clone(),
            frame=None if self.frame is None else self.frame.clone(),
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

    Regeneration is whole-batch by generator design (fixed-batch pipelines);
    per-env respawn control is :meth:`reset`'s mask. Results are undefined
    for envs with ``valid[e] == 0`` on :attr:`result`.
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

        gen_runner = self.generator._runner
        self._runner = GraphRunner(
            self._device,
            capturable=gen_runner.capturable,
            legacy_stream=gen_runner.legacy_stream,
            owner=self,
            name=f"course[{config.gen.generator}]",
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
        # One real warm-up regeneration (generator + refresh) outside any
        # capture: loads every module the regeneration body uses.
        self._runner.prepare(self._enqueue_regenerate)

    @property
    def graph(self) -> "wp.Graph | None":
        """The course's own regeneration graph (generator pipeline + refresh
        as ONE graph): captured on the first CUDA :meth:`regenerate` /
        :meth:`generate` made outside any capture, replayed by later ones.
        ``None`` on CPU, for a non-capturable generator, and before that
        call."""
        return self._runner.graph

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
        rebind only before (re)capturing.
        """
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
        a = self._bind_args
        if a is None:
            return
        self.progress.bind(a["position"])
        if self.localizer is not None:
            self.localizer.bind(a["position"])
        if isinstance(self.collision, FrameChecker):
            # Sphere-vs-frame: bind the agent position and window the query on
            # the tracker's live target-checkpoint state (read-only alias).
            self.collision.bind_inputs(a["position"])
            self.collision.bind_window(self.progress.next_checkpoint_state)
        elif self.collision is not None:
            box_pos = a["box_position"] if a["box_position"] is not None \
                else a["position"]
            self.collision.bind_inputs(box_pos, a["orientation"],
                                       a["half_extents"])

    # -- generation + refresh --------------------------------------------

    def _enqueue_regenerate(self, ctx: LaunchContext) -> None:
        """The ONE regeneration body: generator pipeline (seed refresh
        included) then the coherent downstream refresh, all on ``ctx``.

        Run by ``prepare`` (construction), by the eager path, by the
        self-graph capture and by a caller's recording alike. Zero
        allocation, host-sync free.
        """
        self.generator._enqueue(ctx)
        self._refresh(ctx)

    def regenerate(self, *, stream=None, sync: bool = True):
        """Whole-batch regeneration from the rng's CURRENT seeds plus a
        coherent downstream refresh; returns :attr:`result`.

        The body is the generator pipeline, then checkpoint resample / sdf
        and heightfield bake / posts and line rebuild as applicable, and a
        FULL progress (and localizer) reset — every course changed.

        Execution (stream resolved per :mod:`track_gen.exec`):

        - inside a capture on the resolved stream: RECORDED into the
          caller's graph (e.g. next to :meth:`step` / :meth:`reset`); each
          replay regenerates from the rng's seeds at replay time;
        - outside a capture on CUDA: replay of :attr:`graph` (generator +
          refresh as ONE graph, captured on the first such call);
        - on CPU, or with a ``capturable=False`` generator: eager.

        Args:
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until the regeneration finished on the
                resolved stream (default ``True``; never while capturing, nor
                while torch is capturing in the calling thread).

        Raises:
            track_gen.exec.NotCapturableError: recording a course whose
                generator is ``capturable=False`` (before any launch).

        In track mode, also check ``course.checkpoint_sampler.truncated``
        afterwards: a per-env ``[E]`` int32 flag (1 if ``max_checkpoints``
        clipped that env's checkpoint ring), mirroring ``PropSet.truncated``
        for prop rings.
        """
        self._runner.run(self._enqueue_regenerate, stream=stream, sync=sync)
        return self.result

    def enqueue_generate(self, *, stream=None):
        """Enqueue one regeneration (as :meth:`regenerate`) without
        self-graphing or blocking: recorded into the caller's capture when
        the resolved stream is capturing, otherwise eager on that stream.
        Returns :attr:`result` (stream-ordered)."""
        self._runner.run(self._enqueue_regenerate, stream=stream, policy="eager")
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
        unchanged RNG. Calling ``generate()`` again WITHOUT ``seeds=``
        reproduces the identical batch (plus a full progress reset); pass
        ``seeds=`` to advance the RNG and get new courses.

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
            self._runner.run(self._enqueue_regenerate, stream=ctx.stream, sync=sync)
        return self.result

    def _build_subtools(self, ctx: LaunchContext) -> None:
        """Construct the sub-tools from the first batch (construction only:
        allocation + readback-based sizing, never inside a capture).

        Args:
            ctx: the construction launch context (the stream the generator's
                warm-up pass ran on).
        """
        cfg = self._cfg
        if cfg.mode == "track":
            self.checkpoint_sampler = CheckpointSampler(
                self.result, cfg.checkpoint_spacing,
                max_checkpoints=cfg.max_checkpoints)
            self.checkpoints = self.checkpoint_sampler.sample(stream=ctx.stream)
            if cfg.collision == "segments":
                self.collision = CollisionChecker(
                    self.result, max_boxes=cfg.max_boxes, method="segments")
            elif cfg.collision == "sdf":
                self.collision = CollisionChecker(
                    self.result, max_boxes=cfg.max_boxes, method="sdf",
                    sdf_resolution=cfg.sdf_resolution or 128)
            if cfg.heightfield_resolution is not None:
                self.heightfield = HeightFieldBaker(
                    self.result, cfg.heightfield_resolution)
        else:
            self.checkpoints = CheckpointSet.from_gates(self.result)
            self.course_line = CourseLine(self.result, cfg.samples_per_gate)
            self.localizer = TrackLocalizer(self.course_line.track,
                                            warm_window=cfg.localize_window)
            if cfg.post_radius > 0.0:
                n_slots = int(self.result.position.shape[0])  # E * max_gates
                self._posts = wp.zeros(2 * n_slots, dtype=wp.vec3f,
                                       device=self._device)
                self._fill_posts(ctx=ctx)
                self.collision = DiscChecker(
                    self._posts, radius=cfg.post_radius,
                    max_boxes=cfg.max_boxes, num_envs=self._E)
            elif cfg.frame_collision:
                self.collision = FrameChecker(
                    self.result, num_envs=self._E, radius=cfg.agent_radius,
                    frame_thickness=cfg.frame_thickness,
                    frame_depth=cfg.frame_depth)
        self.progress = ProgressTracker(self.checkpoints)
        self._apply_bind()

    def _fill_posts(self, ctx=None) -> None:
        """Interleave the gate left/right posts into the disc buffer.

        Args:
            ctx: optional track_gen.exec.LaunchContext; default resolves the
                device's stream (see track_gen.exec).
        """
        ctx = _resolve_ctx(ctx, self._device)
        seq = self.result
        ctx.launch(_interleave_posts_k, int(seq.position.shape[0]),
                   [seq.left, seq.right, self._posts])

    def _refresh(self, ctx: LaunchContext) -> None:
        """Post-generation coherence: resample/bake/posts + full reset.

        Args:
            ctx: the regeneration's launch context; every sub-tool runs on
                ``ctx.stream``.
        """
        s = ctx.stream
        if self.checkpoint_sampler is not None:
            self.checkpoint_sampler.sample(stream=s)
        if self.course_line is not None:
            self.course_line.refresh(stream=s)  # before the resets: the line
                                                 # must be current before any
                                                 # consumer query
        if isinstance(self.collision, CollisionChecker) \
                and self.collision._method == "sdf":
            self.collision.bake(stream=s)  # re-baked on every regeneration
                                            # (part of the regeneration graph),
                                            # never per step
        if self.heightfield is not None:
            self.heightfield.bake(stream=s)  # nearest-edge road surface; part
                                              # of the regeneration graph
                                              # alongside the sdf bake
        if self._posts is not None:
            self._fill_posts(ctx=ctx)
        if self.localizer is not None:
            self.localizer.reset(self._reset_all_mask, stream=s)
        self.progress.reset(self._reset_all_mask, stream=s)

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
        s = LaunchContext.for_call(self._device, stream).stream
        events = self.progress.update(stream=s)
        contacts = self.collision.query(stream=s) \
            if self.collision is not None else None
        frame = self.localizer.query(stream=s) \
            if self.localizer is not None else None
        if self._step_result is None:
            self._step_result = StepResult(events=events, contacts=contacts,
                                           frame=frame)
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
        s = LaunchContext.for_call(self._device, stream).stream
        self.progress.reset(mask, stream=s)
        if self.localizer is not None:
            self.localizer.reset(mask, stream=s)

    set_capturing = staticmethod(set_capturing)
