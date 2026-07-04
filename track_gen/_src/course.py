"""Unified course facade: generation + collision + progress in one object.

``Course`` bundles the runtime utilities per mode and owns the orchestration
invariants that are otherwise the caller's burden:

- ``mode="track"``: TrackGenerator -> out-of-bounds ``CollisionChecker``
  (``"segments"`` / ``"sdf"`` / ``None``) -> ``CheckpointSampler`` ->
  ``ProgressTracker``.
- ``mode="gates"``: GateGenerator -> ``CheckpointSet.from_gates`` ->
  ``ProgressTracker``; optional ``DiscChecker`` gate-post collision
  (``post_radius > 0``), with the posts array rebuilt device-side on every
  regeneration.

Lifecycle: construct -> ``bind()`` (stable sim buffers, required) ->
``generate()`` (whole batch: generator pipeline + coherent refresh + full
progress reset) -> per-step ``step()`` / per-env ``reset(mask)``. Whole-batch
generation is a generator constraint (the pipelines are fixed-batch captured
graphs); per-env control lives in ``reset(mask)``.

CUDA graphs: the generator keeps its own pipeline graph (Graph A); the
facade captures the refresh sequence into its own graph on the first cuda
``generate()`` (Graph B) and replays it afterwards. ``step()``/``reset()``
are NOT auto-captured — they are capture-ready for the caller's sim graph;
``set_capturing(True)`` flips the facade's and every sub-module's
``_CAPTURING`` flag in one call.

Results are undefined for envs with ``valid[e] == 0`` on ``course.result``.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from . import runtime
from .checkpoints import CheckpointSampler, CheckpointSet
from .collision import BoxContact, CollisionChecker
from .collision_discs import DiscChecker, DiscContact
from .gate_generator import GateGenerator
from .progress import ProgressEvents, ProgressTracker
from .rng_utils import PerEnvSeededRNG
from .runtime import _check_arr, _init, set_capturing
from .track_generator import TrackGenerator
from .types import GateGenConfig, GateSequence, Track, TrackGenConfig


@wp.kernel
def _interleave_posts_k(
    left: wp.array(dtype=wp.vec2f),
    right: wp.array(dtype=wp.vec2f),
    posts: wp.array(dtype=wp.vec2f),
):
    i = wp.tid()             # dim = E * max_gates; NaN padding carries over
    posts[2 * i] = left[i]
    posts[2 * i + 1] = right[i]


@dataclass
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
        Track mode with ``collision="sdf"`` only; ``None`` -> 128.
    post_radius : float
        Gates mode only: > 0 enables ``DiscChecker`` gate-post collision.
    checkpoint_spacing : float or None
        Track mode only (required there): centerline checkpoint spacing.
    max_checkpoints : int or None
        Track mode only: forwarded to ``CheckpointSampler``.
    max_boxes : int
        Collision query stride (boxes per env). Must be >= 1.
    """

    mode: str
    gen: "TrackGenConfig | GateGenConfig"
    seeds: "int | wp.array" = 0
    collision: "str | None" = None
    sdf_resolution: "int | None" = None
    post_radius: float = 0.0
    checkpoint_spacing: "float | None" = None
    max_checkpoints: "int | None" = None
    max_boxes: int = 1

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
            if not (float(self.gen.gate_width) > 0.0):
                raise ValueError(
                    "gates mode requires gen.gate_width > 0: a width-0 gate "
                    "has a degenerate crossing segment and can never be "
                    "passed")


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
    contacts : BoxContact or DiscContact or None
        Collision result (``None`` when the course has no collision checker).
    """

    events: ProgressEvents
    contacts: "BoxContact | DiscContact | None"

    def clone(self) -> "StepResult":
        """Deep-copy both sub-results."""
        return StepResult(
            events=self.events.clone(),
            contacts=None if self.contacts is None else self.contacts.clone(),
        )


class Course:
    """One object bundling generation, collision, and progress per mode.

    Lifecycle: construct -> :meth:`bind` (stable sim buffers; required for
    :meth:`step`) -> :meth:`generate` (whole batch) -> per-step :meth:`step`
    / per-env :meth:`reset`. Sub-tools are constructed after the FIRST
    ``generate()`` (their auto-derivations need a real batch) and are
    reachable as attributes (``generator``, ``rng``, ``result``,
    ``collision``, ``checkpoints``, ``checkpoint_sampler``, ``progress``).

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
        self.collision: "CollisionChecker | DiscChecker | None" = None
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
        cfg = self._cfg
        return (cfg.mode == "track" and cfg.collision is not None) or \
               (cfg.mode == "gates" and cfg.post_radius > 0.0)

    def _validate_bind_args(self, a: dict) -> None:
        """Eagerly validate bound buffers — E, max_boxes, device are all
        known at construction, so shape/dtype/device errors surface at
        ``bind()`` rather than at the first ``step()``."""
        E = self._E
        self._check_arr("position", a["position"], (E,), wp.vec2f)
        if self._needs_boxes():
            nb = (E * self._cfg.max_boxes,)
            self._check_arr("yaw", a["yaw"], nb, wp.float32)
            self._check_arr("half_extents", a["half_extents"], nb, wp.vec2f)
            if self._cfg.max_boxes > 1:
                self._check_arr("box_position", a["box_position"], nb, wp.vec2f)

    # -- binding ---------------------------------------------------------

    def bind(self, position: wp.array, yaw: "wp.array | None" = None,
             half_extents: "wp.array | None" = None,
             box_position: "wp.array | None" = None) -> None:
        """Bind stable sim buffers (required before :meth:`step`).

        ``position`` is the ``[E]`` vec2f agent-position buffer driving
        progress. When a box-collision checker is enabled, ``yaw`` and
        ``half_extents`` (``[E * max_boxes]``) are required too; with
        ``max_boxes == 1`` the same ``position`` buffer serves as the box
        positions, otherwise pass a separate ``box_position``
        ``[E * max_boxes]`` buffer. May be called before or after the first
        ``generate()``; rebinding replaces the previous binding.

        Do NOT rebind after capturing ``step()`` into a sim graph: the
        captured graph replays against the buffer pointers live at capture
        time, so a later rebind leaves it reading the old buffers (silently
        divergent results) — keep writing into the SAME bound buffers and
        rebind only before (re)capturing.
        """
        needs_boxes = self._needs_boxes()
        if needs_boxes and (yaw is None or half_extents is None):
            raise RuntimeError(
                "this course has a collision checker: bind yaw and "
                "half_extents as well")
        if needs_boxes and self._cfg.max_boxes > 1 and box_position is None:
            raise RuntimeError(
                "max_boxes > 1: bind a separate box_position "
                "[E*max_boxes] buffer")
        args = {"position": position, "yaw": yaw,
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
        if self.collision is not None:
            box_pos = a["box_position"] if a["box_position"] is not None \
                else a["position"]
            self.collision.bind_inputs(box_pos, a["yaw"], a["half_extents"])

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
        """
        if seeds is not None:
            if isinstance(seeds, wp.array):
                self._validate_seed_array(seeds)
                self.rng.set_seeds_warp(seeds, None)
            else:
                tmp = PerEnvSeededRNG(seeds=int(seeds), num_envs=self._E,
                                      device=self._device)
                self.rng.set_seeds_warp(tmp.seeds_warp, None)
        self.result = self.generator.generate()
        if self.progress is None:
            self._build_subtools()
            self._refresh()          # eager warmup (also the cpu path)
        elif self._refresh_graph is not None:
            wp.capture_launch(self._refresh_graph)
            wp.synchronize()
        else:
            self._refresh()
        if self._is_cuda and self._refresh_graph is None:
            set_capturing(True)
            try:
                self._refresh()      # warmup, sync-free
                wp.synchronize()
                with wp.ScopedCapture(device=self._device) as cap:
                    self._refresh()
                self._refresh_graph = cap.graph
            finally:
                set_capturing(False)
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
            if cfg.post_radius > 0.0:
                n_slots = int(self.result.position.shape[0])  # E * max_gates
                self._posts = wp.zeros(2 * n_slots, dtype=wp.vec2f,
                                       device=self._device)
                self._fill_posts()
                self.collision = DiscChecker(
                    self._posts, radius=cfg.post_radius,
                    max_boxes=cfg.max_boxes, num_envs=self._E)
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
        if isinstance(self.collision, CollisionChecker) \
                and self.collision._method == "sdf":
            self.collision.bake()
        if self._posts is not None:
            self._fill_posts()
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
        if self._step_result is None:
            self._step_result = StepResult(events=events, contacts=contacts)
        return self._step_result

    def reset(self, mask: wp.array) -> None:
        """Per-env respawn on the SAME course: clears progress state where
        ``mask[e] == 1``. Collision and checkpoints derive from the course
        geometry and are unaffected by respawns."""
        if self.progress is None:
            raise RuntimeError("call generate() before reset()")
        self.progress.reset(mask)

    set_capturing = staticmethod(set_capturing)
