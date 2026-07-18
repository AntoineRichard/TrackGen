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

Lifecycle: construct -> ``bind()`` (stable sim buffers, required) ->
``generate()`` (whole batch: generator pipeline + coherent refresh + full
progress reset) -> per-step ``step()`` / per-env ``reset(mask)``. Whole-batch
generation is a generator constraint (the pipelines are fixed-batch captured
graphs); per-env control lives in ``reset(mask)``.

CUDA graphs: the generator keeps its own pipeline graph (Graph A); the
facade captures the refresh sequence into its own graph on the first cuda
``generate()`` (Graph B) and replays it afterwards. ``step()``/``reset()``
are NOT auto-captured — they are capture-ready for the caller's sim graph;
``track_gen.set_capturing(True)`` flips the ONE shared capture flag used by
collision, props, checkpoints, progress, and this facade, all at once.

Results are undefined for envs with ``valid[e] == 0`` on ``course.result``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import runtime
from .checkpoints import CheckpointSampler, CheckpointSet
from .collision import BoxContact, CollisionChecker
from .collision_discs import DiscChecker, DiscContact
from .collision_frames import FrameChecker, FrameContact
from .course_line import CourseLine
from .gate_generator import GateGenerator
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
                    "has no collision checker (track: set collision; gates: set "
                    f"post_radius > 0), got max_boxes={self.max_boxes!r}")


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
    :meth:`step`) -> :meth:`generate` (whole batch) -> per-step :meth:`step`
    / per-env :meth:`reset`. Sub-tools are constructed after the FIRST
    ``generate()`` (their auto-derivations need a real batch) and are
    reachable as attributes (``generator``, ``rng``, ``result``,
    ``collision``, ``checkpoints``, ``checkpoint_sampler``, ``progress``,
    and — gates mode — ``course_line``, ``localizer``).

    ``generate()`` is whole-batch by generator design (fixed-batch captured
    pipelines); per-env respawn control is :meth:`reset`'s mask. Results are
    undefined for envs with ``valid[e] == 0`` on :attr:`result`.
    """

    def __init__(self, config: CourseConfig) -> None:
        _init()
        self._cfg = config
        self._E = int(config.gen.num_envs)
        # Canonicalize via Warp: "cuda" -> "cuda:0" so the string matches
        # str(arr.device) in _check_arr (bind/seed validation would otherwise
        # reject cuda:0 buffers when the config used the "cuda" alias).
        self._device = str(wp.get_device(config.gen.device))
        self._is_cuda = "cuda" in self._device
        if isinstance(config.seeds, wp.array):
            self._validate_seed_array(config.seeds)
        self.rng = PerEnvSeededRNG(seeds=config.seeds, num_envs=self._E,
                                   device=self._device)
        if config.mode == "track":
            self.generator = TrackGenerator(config.gen, self.rng)
        else:
            self.generator = GateGenerator(config.gen, self.rng)

        self.result: "Track | GateSequence | None" = None
        self.collision: "CollisionChecker | DiscChecker | FrameChecker | None" \
            = None
        self.course_line: "CourseLine | None" = None
        self.localizer: "TrackLocalizer | None" = None
        self.checkpoints: "CheckpointSet | None" = None
        self.checkpoint_sampler: "CheckpointSampler | None" = None
        self.progress: "ProgressTracker | None" = None
        self._posts: "wp.array | None" = None
        self._bind_args: "dict | None" = None
        self._step_result: "StepResult | None" = None
        self._refresh_graph = None
        self._reset_all_mask = wp.full(self._E, 1, dtype=wp.int32,
                                       device=self._device)

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
        ``[E * max_boxes]`` vec3f buffer. May be called before or after the
        first
        ``generate()``; rebinding replaces the previous binding.

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

    def generate(self, seeds: "int | wp.array | None" = None):
        """Whole-batch (re)generation plus a coherent downstream refresh.

        Optional reseed, generator pipeline (its own captured graph on
        cuda), then: checkpoint resample / sdf bake / posts rebuild as
        applicable, and a FULL progress reset (every course changed). On
        cuda the refresh is captured once into a facade-owned graph and
        replayed on later calls. Returns :attr:`result`.

        Determinism contract: the generators are deterministic under an
        unchanged RNG. Calling ``generate()`` again WITHOUT ``seeds=``
        reproduces the identical batch (plus a full progress reset); pass
        ``seeds=`` to advance the RNG and get new courses.

        In track mode, also check ``course.checkpoint_sampler.truncated``
        after ``generate()``: a per-env ``[E]`` int32 flag (1 if
        ``max_checkpoints`` clipped that env's checkpoint ring), mirroring
        ``PropSet.truncated`` for prop rings.
        """
        if seeds is not None:
            # Reseed under the lock: seed-array construction and the device
            # copy in set_seeds_warp ride the shared stream a concurrent
            # thread may be capturing.
            with runtime._CAPTURE_LOCK:
                if isinstance(seeds, wp.array):
                    self._validate_seed_array(seeds)
                    self.rng.set_seeds_warp(seeds, None)
                else:
                    # Mirror PerEnvSeededRNG's int expansion (seed + arange) so
                    # reseeding via int matches constructing a fresh RNG with it.
                    seed_arr = wp.array(int(seeds) + np.arange(self._E),
                                        dtype=wp.int32, device=self._device)
                    self.rng.set_seeds_warp(seed_arr, None)
        # NOTE: generator.generate() takes runtime._CAPTURE_LOCK internally, so it must
        # stay OUTSIDE the locked regions (the lock is not reentrant).
        self.result = self.generator.generate()
        if self.progress is None:
            # LOCK CONTRACT: every remaining device operation in generate() —
            # subtool construction (allocations + eager launches), eager
            # refresh, warmup, capture, replay — holds _CAPTURE_LOCK, so a
            # concurrent thread's capture never records our allocations or
            # async frees. Only generator.generate() (self-locking) is outside.
            with runtime._CAPTURE_LOCK:
                self._build_subtools()
                self._refresh()  # eager: cpu every-call path is below; on cuda
                                  # this is call 1 of 3 (see capture block)
        elif self._refresh_graph is not None:
            with runtime._CAPTURE_LOCK:
                wp.capture_launch(self._refresh_graph)
                wp.synchronize()
        else:
            self._refresh()
        if self._is_cuda and self._refresh_graph is None:
            with runtime._CAPTURE_LOCK:
                prev = runtime._CAPTURING      # save/restore (generators' idiom;
                set_capturing(True)             # flag mutations stay under the lock)
                try:
                    self._refresh()  # warmup, sync-free (call 2 of 3: runs for
                                      # real, priming any lazy kernel/module
                                      # init before we start recording below)
                    wp.synchronize()
                    with wp.ScopedCapture(device=self._device) as cap:
                        self._refresh()  # call 3 of 3: RECORDED into the graph,
                                      # not executed here; replayed just below
                    self._refresh_graph = cap.graph
                finally:
                    set_capturing(prev)
                wp.capture_launch(self._refresh_graph)
                wp.synchronize()
        return self.result

    def _build_subtools(self) -> None:
        cfg = self._cfg
        if cfg.mode == "track":
            self.checkpoint_sampler = CheckpointSampler(
                self.result, cfg.checkpoint_spacing,
                max_checkpoints=cfg.max_checkpoints)
            self.checkpoints = self.checkpoint_sampler.sample()
            if cfg.collision == "segments":
                self.collision = CollisionChecker(
                    self.result, max_boxes=cfg.max_boxes, method="segments")
            elif cfg.collision == "sdf":
                self.collision = CollisionChecker(
                    self.result, max_boxes=cfg.max_boxes, method="sdf",
                    sdf_resolution=cfg.sdf_resolution or 128)
        else:
            self.checkpoints = CheckpointSet.from_gates(self.result)
            self.course_line = CourseLine(self.result, cfg.samples_per_gate)
            self.localizer = TrackLocalizer(self.course_line.track,
                                            warm_window=cfg.localize_window)
            if cfg.post_radius > 0.0:
                n_slots = int(self.result.position.shape[0])  # E * max_gates
                self._posts = wp.zeros(2 * n_slots, dtype=wp.vec3f,
                                       device=self._device)
                self._fill_posts()
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

    def _fill_posts(self) -> None:
        seq = self.result
        wp.launch(_interleave_posts_k, dim=int(seq.position.shape[0]),
                  inputs=[seq.left, seq.right, self._posts],
                  device=self._device)

    def _refresh(self) -> None:
        """Post-generation coherence: resample/bake/posts + full reset."""
        if self.checkpoint_sampler is not None:
            self.checkpoint_sampler.sample()
        if self.course_line is not None:
            self.course_line.refresh()  # before the resets: the line must be
                                         # current before any consumer query
        if isinstance(self.collision, CollisionChecker) \
                and self.collision._method == "sdf":
            self.collision.bake()  # re-baked on every _refresh() call, incl.
                                    # both real warmup passes before the first
                                    # cuda capture: a one-time construction
                                    # cost, not a per-step one
        if self._posts is not None:
            self._fill_posts()
        if self.localizer is not None:
            self.localizer.reset(self._reset_all_mask)
        self.progress.reset(self._reset_all_mask)

    # -- per-step ----------------------------------------------------------

    def step(self) -> StepResult:
        """Progress update + collision query on the bound buffers."""
        if self.progress is None:
            raise RuntimeError("call generate() before step()")
        if self._bind_args is None:
            raise RuntimeError("call bind() before step()")
        events = self.progress.update()
        contacts = self.collision.query() if self.collision is not None else None
        frame = self.localizer.query() if self.localizer is not None else None
        if self._step_result is None:
            self._step_result = StepResult(events=events, contacts=contacts,
                                           frame=frame)
        return self._step_result

    def reset(self, mask: wp.array) -> None:
        """Per-env respawn on the SAME course: clears progress state (and,
        in gates mode, the localizer's warm-start memory) where
        ``mask[e] == 1``. Collision and checkpoints derive from the course
        geometry and are unaffected by respawns."""
        if self.progress is None:
            raise RuntimeError("call generate() before reset()")
        self.progress.reset(mask)
        if self.localizer is not None:
            self.localizer.reset(mask)

    set_capturing = staticmethod(set_capturing)
