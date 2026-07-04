# Course Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `track_gen.course` — one object bundling generation + collision + checkpoints/progress per mode (`track`, `gates`): construct → `bind()` → `generate()` → per-step `step()` / per-env `reset(mask)`, with a facade-owned CUDA refresh graph (Graph B) and a one-switch capture helper.

**Architecture:** `Course` wires the existing utilities and owns the orchestration invariants (rebake/resample/posts-rebuild + full progress reset after every whole-batch `generate()`; per-env `reset(mask)` for respawns). Sub-tools are built lazily after the first `generate()` (auto-derivations need a real batch) and stay reachable as attributes. Binding is uniform: this plan first adds `ProgressTracker.bind(position)` and `DiscChecker.bind_inputs(...)` so all tools support post-construction (re)binding like `CollisionChecker`.

**Tech Stack:** Python ≥ 3.10, warp-lang ≥ 1.14, numpy. Tests: pytest (+ torch in the cuda test).

**Spec:** `docs/superpowers/specs/2026-07-04-course-facade-design.md`

## Global Constraints

- Runtime deps numpy + warp-lang only; numpy only in host-side setup (never in `step()`/`reset()`). Everything via `wp.launch(..., inputs=[...])`.
- `step()`/`reset()` allocation-free and host-sync-free under capture; the facade module has `_INITED/_CAPTURING/_init/_sync` like its siblings, plus `set_capturing(flag)` that toggles the facade AND all sub-module flags (`collision`, `collision_discs`, `checkpoints`, `progress`) in one call.
- Whole-batch `generate()` / per-env `reset(mask)` split (generator fixed-batch constraint, documented). `generate()` sequence: optional reseed → generator pipeline (Graph A, untouched) → refresh (checkpoint `sample()` / sdf `bake()` / posts rebuild as applicable) → FULL progress reset via a persistent all-ones mask buffer. First cuda `generate()` captures the refresh into facade-owned Graph B after eager warmup; later calls replay it. cpu runs eagerly.
- Strict config applicability (ValueError, no silent ignores): `checkpoint_spacing`/`max_checkpoints`/`collision`/`sdf_resolution` are track-mode options; `post_radius` is gates-mode; inapplicable fields default to `None`/`0.0` sentinels so common constructions never trip. NaN-proof numeric validation (`not (x > 0)`).
- The facade is bound-mode only: `step()` without `bind()` → `RuntimeError` naming bind; `step()`/`reset()` before the first `generate()` → `RuntimeError("... generate() ...")`.
- `StepResult` holds the sub-tools' in-place result instances (same instance every `step()`; `clone()` deep-copies).
- Gates mode requires `gen.gate_width > 0` (a width-0 gate can never be crossed — the facade exists to make progress work, so this is a construction-time `ValueError`, a spec-compatible tightening).
- Every pytest run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest …`
- Work on branch `feature/course-facade` off main (Task 1 Step 0).

---

### Task 1: Uniform binding — `ProgressTracker.bind()` and `DiscChecker.bind_inputs()`

**Files:**
- Modify: `track_gen/_src/progress.py` (add `bind` method; ctor delegates to it)
- Modify: `track_gen/_src/collision_discs.py` (add `bind_inputs` method; ctor delegates)
- Test: `tests/test_progress.py`, `tests/test_collision_discs.py` (append)

**Interfaces:**
- Consumes: existing `ProgressTracker._validate_position`, `DiscChecker._validate_inputs`.
- Produces (Task 3 relies on): `ProgressTracker.bind(position: wp.array) -> None` (validates, sets `_bound_pos`; rebinding replaces); `DiscChecker.bind_inputs(position, yaw, half_extents) -> None` (validates, sets `_bound`; rebinding replaces). Constructor binding behavior unchanged.

- [ ] **Step 0: Create the feature branch**

```bash
git checkout -b feature/course-facade
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_progress.py`:

```python
def test_bind_after_construction_and_rebind():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    buf = wp.zeros(E, dtype=wp.vec2f, device="cpu")
    tracker.bind(buf)
    wp.copy(buf, _pos(-22.5))
    tracker.update()                     # bound mode now works
    wp.copy(buf, _pos(22.5))
    ev = tracker.update()
    assert int(ev.passed.numpy()[0]) == 1
    buf2 = wp.zeros(E, dtype=wp.vec2f, device="cpu")
    tracker.bind(buf2)                   # rebinding replaces
    wp.copy(buf2, _pos(67.5))
    tracker.update()                     # reads buf2, no error
    with pytest.raises(ValueError, match="position"):
        tracker.bind(wp.zeros(E + 1, dtype=wp.vec2f, device="cpu"))
```

Append to `tests/test_collision_discs.py`:

```python
def test_bind_inputs_after_construction():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.12, 0.0]])
    pos, yaw, he = _boxes(1, 1, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    checker = DiscChecker(discs, radius=0.03, max_boxes=1, num_envs=1)
    checker.bind_inputs(pos, yaw, he)
    res = checker.query()
    assert int(res.hit.numpy()[0]) == 1
    with pytest.raises(ValueError, match="bound"):
        checker.query(pos, yaw, he)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress.py::test_bind_after_construction_and_rebind tests/test_collision_discs.py::test_bind_inputs_after_construction -v`
Expected: FAIL — `AttributeError` (`bind` / `bind_inputs` missing)

- [ ] **Step 3: Implement**

In `track_gen/_src/progress.py`, inside `ProgressTracker`, replace the
constructor's binding block

```python
        self._bound_pos: "wp.array | None" = None
        if position is not None:
            self._validate_position(position)
            self._bound_pos = position
```

with

```python
        self._bound_pos: "wp.array | None" = None
        if position is not None:
            self.bind(position)
```

and add after `_validate_position`:

```python
    def bind(self, position: wp.array) -> None:
        """Bind (or rebind) a stable ``[E]`` vec2f position buffer.

        After binding, ``update()`` takes no arguments and reads the buffer
        in place. Validation happens here, once; the array must keep the
        same ``.ptr`` for the binding's lifetime (CUDA-graph contract).
        """
        self._validate_position(position)
        self._bound_pos = position
```

In `track_gen/_src/collision_discs.py`, inside `DiscChecker`, replace the
constructor's binding block

```python
        self._bound = None
        if position is not None or yaw is not None or half_extents is not None:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "bind all of position/yaw/half_extents or none")
            self._validate_inputs(position, yaw, half_extents)
            self._bound = (position, yaw, half_extents)
```

with

```python
        self._bound = None
        if position is not None or yaw is not None or half_extents is not None:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "bind all of position/yaw/half_extents or none")
            self.bind_inputs(position, yaw, half_extents)
```

and add after `_validate_inputs`:

```python
    def bind_inputs(self, position: wp.array, yaw: wp.array,
                    half_extents: wp.array) -> None:
        """Bind (or rebind) stable per-step input buffers (validated once).

        After binding, ``query()`` takes no arguments and reads these arrays
        in place; same-``.ptr`` rule applies under CUDA-graph capture.
        """
        self._validate_inputs(position, yaw, half_extents)
        self._bound = (position, yaw, half_extents)
```

- [ ] **Step 4: Run the affected files**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress.py tests/test_collision_discs.py -q`
Expected: all PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/progress.py track_gen/_src/collision_discs.py tests/test_progress.py tests/test_collision_discs.py
git commit -m "feat: uniform post-construction binding — ProgressTracker.bind, DiscChecker.bind_inputs"
```

---

### Task 2: `CourseConfig` + `StepResult` + validation matrix

**Files:**
- Create: `track_gen/_src/course.py` (module skeleton: docstring, `_INITED/_CAPTURING/_init/_sync`, `set_capturing`, `CourseConfig`, `StepResult`)
- Test: `tests/test_course_config.py`

**Interfaces:**
- Consumes: `TrackGenConfig`, `GateGenConfig` from `types`; sub-module `_CAPTURING` flags.
- Produces (Tasks 3–6 rely on):
  - `CourseConfig(mode, gen, seeds=0, collision=None, sdf_resolution=None, post_radius=0.0, checkpoint_spacing=None, max_checkpoints=None, max_boxes=1)` with the validation rules below.
  - `StepResult` dataclass: `events: ProgressEvents`, `contacts` (BoxContact | DiscContact | None); `clone()`.
  - `set_capturing(flag: bool)` module function toggling `course`, `collision`, `collision_discs`, `checkpoints`, `progress` module flags.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_course_config.py`:

```python
"""CourseConfig validation matrix + set_capturing flag propagation."""
from __future__ import annotations

import pytest

from track_gen import GateGenConfig, TrackGenConfig


def _track_cfg(**kw):
    from track_gen.course import CourseConfig
    base = dict(mode="track", gen=TrackGenConfig(num_envs=2, device="cpu"),
                checkpoint_spacing=0.6)
    base.update(kw)
    return CourseConfig(**base)


def _gates_cfg(**kw):
    from track_gen.course import CourseConfig
    base = dict(mode="gates",
                gen=GateGenConfig(num_envs=2, device="cpu", gate_width=0.1))
    base.update(kw)
    return CourseConfig(**base)


def test_valid_constructions():
    _track_cfg()                                   # progress-only track
    _track_cfg(collision="segments")
    _track_cfg(collision="sdf", sdf_resolution=64)
    _track_cfg(collision="sdf")                    # sdf_resolution defaults to 128
    _gates_cfg()                                   # progress-only gates
    _gates_cfg(post_radius=0.02)


def test_mode_and_gen_type_agreement():
    from track_gen.course import CourseConfig
    with pytest.raises(ValueError, match="mode"):
        CourseConfig(mode="drone", gen=TrackGenConfig(num_envs=1, device="cpu"))
    with pytest.raises(ValueError, match="TrackGenConfig"):
        CourseConfig(mode="track",
                     gen=GateGenConfig(num_envs=1, device="cpu"),
                     checkpoint_spacing=0.5)
    with pytest.raises(ValueError, match="GateGenConfig"):
        CourseConfig(mode="gates", gen=TrackGenConfig(num_envs=1, device="cpu"))


def test_inapplicable_options_raise():
    with pytest.raises(ValueError, match="post_radius"):
        _track_cfg(post_radius=0.02)
    with pytest.raises(ValueError, match="collision"):
        _gates_cfg(collision="segments")
    with pytest.raises(ValueError, match="sdf_resolution"):
        _gates_cfg(sdf_resolution=64)
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        _gates_cfg(checkpoint_spacing=0.5)
    with pytest.raises(ValueError, match="max_checkpoints"):
        _gates_cfg(max_checkpoints=32)
    with pytest.raises(ValueError, match="sdf_resolution"):
        _track_cfg(collision="segments", sdf_resolution=64)  # sdf-only knob


def test_numeric_validation():
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        _track_cfg(checkpoint_spacing=0.0)
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        _track_cfg(checkpoint_spacing=float("nan"))
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        from track_gen.course import CourseConfig
        CourseConfig(mode="track", gen=TrackGenConfig(num_envs=1, device="cpu"))
    with pytest.raises(ValueError, match="max_boxes"):
        _track_cfg(max_boxes=0)
    with pytest.raises(ValueError, match="collision"):
        _track_cfg(collision="bvh")
    with pytest.raises(ValueError, match="post_radius"):
        _gates_cfg(post_radius=float("nan"))
    with pytest.raises(ValueError, match="gate_width"):
        from track_gen.course import CourseConfig
        CourseConfig(mode="gates",
                     gen=GateGenConfig(num_envs=1, device="cpu", gate_width=0.0))


def test_set_capturing_propagates():
    from track_gen._src import checkpoints as cps_mod
    from track_gen._src import collision as col_mod
    from track_gen._src import collision_discs as discs_mod
    from track_gen._src import course as course_mod
    from track_gen._src import progress as prog_mod
    from track_gen.course import set_capturing
    set_capturing(True)
    try:
        assert course_mod._CAPTURING and col_mod._CAPTURING \
            and discs_mod._CAPTURING and cps_mod._CAPTURING \
            and prog_mod._CAPTURING
    finally:
        set_capturing(False)
    assert not (course_mod._CAPTURING or col_mod._CAPTURING
                or discs_mod._CAPTURING or cps_mod._CAPTURING
                or prog_mod._CAPTURING)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_config.py -v`
Expected: FAIL — `ModuleNotFoundError: track_gen.course`

- [ ] **Step 3: Implement the module skeleton in `track_gen/_src/course.py`**

```python
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

import numpy as np
import warp as wp

from . import checkpoints as _cps_mod
from . import collision as _col_mod
from . import collision_discs as _discs_mod
from . import progress as _prog_mod
from .checkpoints import CheckpointSampler, CheckpointSet
from .collision import BoxContact, CollisionChecker
from .collision_discs import DiscChecker, DiscContact
from .gate_generator import GateGenerator
from .progress import ProgressEvents, ProgressTracker
from .rng_utils import PerEnvSeededRNG
from .track_generator import TrackGenerator
from .types import GateGenConfig, GateSequence, Track, TrackGenConfig

_INITED = False
_CAPTURING = False


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


def set_capturing(flag: bool) -> None:
    """Toggle the capture flag on the facade AND all sub-tool modules.

    One switch for user-side CUDA graph captures of ``step()``/``reset()``:
    while ``True``, no utility performs a host synchronize.
    """
    global _CAPTURING
    _CAPTURING = bool(flag)
    _col_mod._CAPTURING = bool(flag)
    _discs_mod._CAPTURING = bool(flag)
    _cps_mod._CAPTURING = bool(flag)
    _prog_mod._CAPTURING = bool(flag)


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
```

- [ ] **Step 4: Create a minimal public shim so the tests import (extended in Task 3)**

Create `track_gen/course.py`:

```python
"""Public course facade: generation + collision + progress in one object.

See :class:`Course` for the lifecycle (bind -> generate -> step/reset) and
``set_capturing`` for one-switch CUDA-graph capture of the step path.
"""
from ._src.course import CourseConfig, StepResult, set_capturing

__all__ = ["CourseConfig", "StepResult", "set_capturing"]
```

(`Course` is added to this shim and to `track_gen/__init__.py` in Task 3.)

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_config.py -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/course.py track_gen/course.py tests/test_course_config.py
git commit -m "feat: CourseConfig validation matrix, StepResult, set_capturing"
```

---

### Task 3: `Course` — track mode end-to-end (lifecycle, refresh, step, reset)

**Files:**
- Modify: `track_gen/_src/course.py` (append `Course`)
- Modify: `track_gen/course.py` (export `Course`), `track_gen/__init__.py` (add `from . import course`, `__all__ += ["course"]`), `tests/test_public_api.py` (add `"course"`)
- Test: `tests/test_course_track.py`

**Interfaces:**
- Consumes: Tasks 1–2; `TrackGenerator`, `CheckpointSampler`, `CollisionChecker` (+ `bind_inputs`), `ProgressTracker` (+ `bind`), `PerEnvSeededRNG`.
- Produces (Tasks 4–6 rely on): `Course(config)` with methods `bind(position, yaw=None, half_extents=None, box_position=None)`, `generate(seeds=None) -> Track | GateSequence`, `step() -> StepResult`, `reset(mask)`, staticmethod `set_capturing(flag)`; attributes `generator`, `rng`, `result`, `collision`, `checkpoints`, `checkpoint_sampler`, `progress`; internals `_refresh()`, `_refresh_graph`, `_reset_all_mask`, `_posts` (None in track mode), `_step_result`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_course_track.py`:

```python
"""Track-mode Course facade: lifecycle, refresh coherence, step/reset."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import TrackGenConfig

E = 4
SPACING = 0.6


def _course(collision="segments", **kw):
    from track_gen.course import Course, CourseConfig
    cfg = CourseConfig(mode="track",
                       gen=TrackGenConfig(num_envs=E, device="cpu"),
                       seeds=123, collision=collision,
                       checkpoint_spacing=SPACING, max_checkpoints=64, **kw)
    return Course(cfg)


def _buffers():
    pos = wp.zeros(E, dtype=wp.vec2f, device="cpu")
    yaw = wp.zeros(E, dtype=wp.float32, device="cpu")
    he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device="cpu")
    return pos, yaw, he


def _drive(course, pos_buf, n_steps=40):
    """Walk each valid env along its own centerline; returns final events."""
    track = course.result
    n_max = track.outer.shape[0] // E
    center = np.nan_to_num(track.center.numpy().reshape(E, n_max, 2), nan=0.0)
    counts = track.count.numpy()
    ev = None
    for s in range(n_steps):
        step_pos = np.zeros((E, 2), np.float32)
        for e in range(E):
            m = max(int(counts[e]), 1)
            step_pos[e] = center[e, (s * 3) % m]
        wp.copy(pos_buf, wp.array(step_pos, dtype=wp.vec2f, device="cpu"))
        ev = course.step().events
    return ev


def test_lifecycle_errors():
    from track_gen.course import Course, CourseConfig
    course = _course()
    mask = wp.zeros(E, dtype=wp.int32, device="cpu")
    with pytest.raises(RuntimeError, match="generate"):
        course.step()
    with pytest.raises(RuntimeError, match="generate"):
        course.reset(mask)
    course.generate()
    with pytest.raises(RuntimeError, match="bind"):
        course.step()


def test_import_surface():
    import track_gen
    from track_gen.course import Course, CourseConfig, StepResult  # noqa: F401
    assert "course" in track_gen.__all__


def test_end_to_end_generate_step_reset():
    course = _course()
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    track = course.generate()
    assert track is course.result
    assert course.progress is not None and course.collision is not None
    assert course.checkpoints is course.checkpoint_sampler._set

    ev = _drive(course, pos)
    prog = ev.progress.numpy()
    valid = track.valid.numpy().astype(bool)
    assert prog[valid].sum() > 0, "driving the centerline must pass checkpoints"

    res = course.step()
    assert res is course.step()          # same StepResult instance
    assert res.contacts is not None
    # Boxes on the centerline are inside the band.
    oob = res.contacts.oob.numpy()
    assert not oob[valid].any()

    # Per-env reset: only masked envs are cleared.
    before = course.progress._progress.numpy().copy()
    mask_np = np.zeros(E, np.int32)
    victim = int(np.argmax(before * valid))
    mask_np[victim] = 1
    course.reset(wp.array(mask_np, dtype=wp.int32, device="cpu"))
    after = course.progress._progress.numpy()
    assert after[victim] == 0
    keep = [e for e in range(E) if e != victim]
    np.testing.assert_array_equal(after[keep], before[keep])


def test_regenerate_refreshes_everything():
    course = _course()
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    course.generate()
    counts1 = course.checkpoints.count.numpy().copy()
    _drive(course, pos, n_steps=20)
    assert course.progress._progress.numpy().sum() > 0

    track2 = course.generate(seeds=999)          # new courses for everyone
    assert track2 is course.result               # in-place fixed-batch contract
    counts2 = course.checkpoints.count.numpy()
    assert (counts1 != counts2).any(), "new geometry should change checkpoint counts"
    assert course.progress._progress.numpy().sum() == 0   # full reset
    assert np.isnan(course.progress._prev_pos.numpy()).all()


def test_sdf_mode_rebakes_on_regenerate():
    course = _course(collision="sdf", sdf_resolution=64)
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    track = course.generate()
    valid = track.valid.numpy().astype(bool)
    e = int(np.argmax(valid))
    n_max = track.outer.shape[0] // E

    def probe(kind):
        center = np.nan_to_num(track.center.numpy().reshape(E, n_max, 2), nan=0.0)
        p = np.zeros((E, 2), np.float32)
        p[e] = center[e, 0] if kind == "inside" else np.array([50.0, 50.0])
        wp.copy(pos, wp.array(p, dtype=wp.vec2f, device="cpu"))
        return int(course.step().contacts.oob.numpy()[e])

    assert probe("inside") == 0
    assert probe("far") == 1
    course.generate(seeds=777)
    # Fresh bake: the NEW track's centerline reads inside, far still outside.
    assert probe("inside") == 0
    assert probe("far") == 1


def test_progress_only_bundle():
    course = _course(collision=None)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    course.generate()
    res = course.step()
    assert res.contacts is None
    assert course.collision is None


def test_facade_matches_manual_wiring():
    from track_gen.progress import ProgressTracker
    course = _course(collision="segments")
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    track = course.generate()
    # Twin tracker on the SAME checkpoint set, driven with the same buffer.
    twin = ProgressTracker(course.checkpoints, position=pos)
    all_mask = wp.array(np.ones(E, np.int32), dtype=wp.int32, device="cpu")
    course.reset(all_mask)
    n_max = track.outer.shape[0] // E
    center = np.nan_to_num(track.center.numpy().reshape(E, n_max, 2), nan=0.0)
    counts = track.count.numpy()
    for s in range(25):
        step_pos = np.zeros((E, 2), np.float32)
        for e in range(E):
            m = max(int(counts[e]), 1)
            step_pos[e] = center[e, (s * 3) % m]
        wp.copy(pos, wp.array(step_pos, dtype=wp.vec2f, device="cpu"))
        ev_f = course.step().events
        ev_t = twin.update()
        np.testing.assert_array_equal(ev_f.passed.numpy(), ev_t.passed.numpy())
        np.testing.assert_array_equal(ev_f.next_checkpoint.numpy(),
                                      ev_t.next_checkpoint.numpy())
        np.testing.assert_array_equal(ev_f.progress.numpy(), ev_t.progress.numpy())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_track.py -v`
Expected: FAIL — `ImportError: cannot import name 'Course'`

- [ ] **Step 3: Append `Course` to `track_gen/_src/course.py`**

```python
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
        self._device = str(config.gen.device)
        self._is_cuda = "cuda" in self._device
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
        """
        needs_boxes = (self._cfg.mode == "track" and self._cfg.collision
                       is not None) or \
                      (self._cfg.mode == "gates" and self._cfg.post_radius > 0.0)
        if needs_boxes and (yaw is None or half_extents is None):
            raise RuntimeError(
                "this course has a collision checker: bind yaw and "
                "half_extents as well")
        if needs_boxes and self._cfg.max_boxes > 1 and box_position is None:
            raise RuntimeError(
                "max_boxes > 1: bind a separate box_position "
                "[E*max_boxes] buffer")
        self._bind_args = {"position": position, "yaw": yaw,
                           "half_extents": half_extents,
                           "box_position": box_position}
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
        """
        if seeds is not None:
            if isinstance(seeds, wp.array):
                self.rng.set_seeds_warp(seeds, None)
            else:
                tmp = PerEnvSeededRNG(seeds=int(seeds), num_envs=self._E,
                                      device=self._device)
                self.rng.set_seeds_warp(tmp.seeds_warp, None)
        first = self.result is None
        self.result = self.generator.generate()
        if first:
            self._build_subtools()
            self._refresh()          # eager warmup (also the cpu path)
            if self._is_cuda:
                set_capturing(True)
                try:
                    self._refresh()  # second warmup, sync-free
                    wp.synchronize()
                    with wp.ScopedCapture(device=self._device) as cap:
                        self._refresh()
                    self._refresh_graph = cap.graph
                finally:
                    set_capturing(False)
                wp.capture_launch(self._refresh_graph)
                wp.synchronize()
        else:
            if self._refresh_graph is not None:
                wp.capture_launch(self._refresh_graph)
                wp.synchronize()
            else:
                self._refresh()
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
```

And add the posts kernel above the `CourseConfig` class:

```python
@wp.kernel
def _interleave_posts_k(
    left: wp.array(dtype=wp.vec2f),
    right: wp.array(dtype=wp.vec2f),
    posts: wp.array(dtype=wp.vec2f),
):
    i = wp.tid()             # dim = E * max_gates; NaN padding carries over
    posts[2 * i] = left[i]
    posts[2 * i + 1] = right[i]
```

- [ ] **Step 4: Wire the public surface**

`track_gen/course.py`:

```python
from ._src.course import Course, CourseConfig, StepResult, set_capturing

__all__ = ["Course", "CourseConfig", "StepResult", "set_capturing"]
```

`track_gen/__init__.py`: add `from . import course` after `from . import
progress`; add `"course"` to `__all__`. `tests/test_public_api.py`: add
`"course"` to the curated set.

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_track.py tests/test_course_config.py tests/test_public_api.py -v`
Expected: all PASS (generation on cpu takes a few seconds per construct)

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/course.py track_gen/course.py track_gen/__init__.py tests/test_public_api.py tests/test_course_track.py
git commit -m "feat: Course facade — track mode end-to-end (generate/refresh/step/reset)"
```

---

### Task 4: Gates mode end-to-end

**Files:**
- Test: `tests/test_course_gates.py`
- Modify (only if a test exposes a facade bug): `track_gen/_src/course.py`

**Interfaces:**
- Consumes: Task 3 `Course` (gates branch already implemented there).
- Produces: verified gates-mode semantics for Tasks 5–6.

- [ ] **Step 1: Write the tests**

Create `tests/test_course_gates.py`:

```python
"""Gates-mode Course facade: from_gates progress + post collision."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import GateGenConfig

E = 4


def _course(post_radius=0.02, **kw):
    from track_gen.course import Course, CourseConfig
    cfg = CourseConfig(mode="gates",
                       gen=GateGenConfig(num_envs=E, device="cpu",
                                         gate_width=0.1),
                       seeds=21, post_radius=post_radius, **kw)
    return Course(cfg)


def _buffers():
    pos = wp.zeros(E, dtype=wp.vec2f, device="cpu")
    yaw = wp.zeros(E, dtype=wp.float32, device="cpu")
    he = wp.array(np.full((E, 2), 0.01, np.float32), dtype=wp.vec2f, device="cpu")
    return pos, yaw, he


def test_gate_pass_through_facade():
    course = _course()
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    seq = course.generate()
    valid = seq.valid.numpy().astype(bool)
    e = int(np.argmax(valid))
    G = seq.position.shape[0] // E
    g0 = seq.position.numpy().reshape(E, G, 2)[e, 0]
    t0 = seq.tangent.numpy().reshape(E, G, 2)[e, 0]

    def put(p):
        arr = np.zeros((E, 2), np.float32)
        arr[e] = p
        wp.copy(pos, wp.array(arr, dtype=wp.vec2f, device="cpu"))

    put(g0 - 0.2 * t0)
    course.step()
    put(g0 + 0.2 * t0)
    ev = course.step().events
    assert int(ev.passed.numpy()[e]) == 1
    assert int(ev.checkpoint_passed.numpy()[e]) == 0
    assert int(ev.next_checkpoint.numpy()[e]) == 1


def test_post_collision_and_rebuild_on_regenerate():
    course = _course()
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    seq = course.generate()
    valid = seq.valid.numpy().astype(bool)
    e = int(np.argmax(valid))
    G = seq.position.shape[0] // E
    left0 = seq.left.numpy().reshape(E, G, 2)[e, 0]

    arr = np.zeros((E, 2), np.float32)
    arr[e] = left0
    wp.copy(pos, wp.array(arr, dtype=wp.vec2f, device="cpu"))
    res = course.step()
    assert int(res.contacts.hit.numpy()[e]) == 1
    assert int(res.contacts.disc.numpy()[e]) == 0      # gate 0 left post

    course.generate(seeds=888)                          # posts must follow
    seq2 = course.result
    left0b = seq2.left.numpy().reshape(E, G, 2)
    e2 = int(np.argmax(seq2.valid.numpy()))
    arr = np.zeros((E, 2), np.float32)
    arr[e2] = left0b[e2, 0]
    wp.copy(pos, wp.array(arr, dtype=wp.vec2f, device="cpu"))
    res = course.step()
    assert int(res.contacts.hit.numpy()[e2]) == 1       # NEW gate's post hits
    assert int(course.progress._progress.numpy().sum()) == 0  # full reset


def test_progress_only_gates_bundle():
    course = _course(post_radius=0.0)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    course.generate()
    res = course.step()
    assert res.contacts is None and course.collision is None


def test_checkpoints_alias_gates_zero_copy():
    course = _course()
    pos, yaw, he = _buffers()
    course.bind(position=pos, yaw=yaw, half_extents=he)
    seq = course.generate()
    assert course.checkpoints.position.ptr == seq.position.ptr
    assert course.checkpoints.count.ptr == seq.count.ptr
```

- [ ] **Step 2: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_gates.py -v`
Expected: 4 PASS. A failure most likely means a facade wiring bug (posts
rebuild ordering in `_refresh`, or gate-mode `bind` box-position reuse) —
fix `track_gen/_src/course.py` minimally.

- [ ] **Step 3: Run the full suite for regressions**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_course_gates.py track_gen/_src/course.py
git commit -m "test: gates-mode Course facade — pass-through, posts rebuild, zero-copy"
```

---

### Task 5: CUDA — Graph B replay + user-side step capture

**Files:**
- Test: `tests/test_course_cuda.py`

**Interfaces:**
- Consumes: `Course`, `set_capturing`; determinism of generators under equal seeds.

- [ ] **Step 1: Write the tests**

Create `tests/test_course_cuda.py`:

```python
"""CUDA-only: facade Graph B refresh replay + user-captured step()."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda"),
]

import warp as wp  # noqa: E402
from track_gen import TrackGenConfig  # noqa: E402
from track_gen.course import Course, CourseConfig, set_capturing  # noqa: E402

DEV = "cuda:0"
E = 8


def _cfg(seeds=5):
    return CourseConfig(mode="track",
                        gen=TrackGenConfig(num_envs=E, device=DEV),
                        seeds=seeds, collision="segments",
                        checkpoint_spacing=0.6, max_checkpoints=64)


def _bound_course(seeds=5):
    course = Course(_cfg(seeds))
    pos = wp.zeros(E, dtype=wp.vec2f, device=DEV)
    yaw = wp.zeros(E, dtype=wp.float32, device=DEV)
    he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
    course.bind(position=pos, yaw=yaw, half_extents=he)
    return course, pos


def test_graph_b_refresh_replay_recomputes():
    from track_gen.checkpoints import CheckpointSampler
    course, _ = _bound_course()
    course.generate()
    assert course._refresh_graph is not None      # captured on first generate
    course.generate(seeds=901)                    # replay path
    # Poisoned-replay proof: trash the checkpoint buffers and progress state,
    # regenerate with new seeds; the replayed refresh must recompute both.
    course.checkpoints.position.fill_(12345.0)
    course.progress._progress.fill_(-7)
    course.generate(seeds=902)
    ref = CheckpointSampler(course.result, 0.6, max_checkpoints=64).sample()
    np.testing.assert_allclose(course.checkpoints.position.numpy(),
                               ref.position.numpy(), rtol=1e-5, equal_nan=True)
    assert (course.progress._progress.numpy() == 0).all()


def test_user_captured_step_matches_eager_twin():
    course_c, pos_c = _bound_course(seeds=5)
    course_e, pos_e = _bound_course(seeds=5)      # identical seeds -> same tracks
    course_c.generate()
    course_e.generate()

    set_capturing(True)
    try:
        course_c.step()                            # warmup
        wp.synchronize()
        all_mask = wp.full(E, 1, dtype=wp.int32, device=DEV)
        course_c.reset(all_mask)
        course_e.reset(wp.full(E, 1, dtype=wp.int32, device=DEV))
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            course_c.step()
    finally:
        set_capturing(False)

    n_max = course_c.result.outer.shape[0] // E
    center = np.nan_to_num(
        course_c.result.center.numpy().reshape(E, n_max, 2), nan=0.0)
    counts = course_c.result.count.numpy()
    for s in range(12):
        step_pos = np.zeros((E, 2), np.float32)
        for e in range(E):
            m = max(int(counts[e]), 1)
            step_pos[e] = center[e, (s * 4) % m]
        arr = wp.array(step_pos, dtype=wp.vec2f, device=DEV)
        wp.copy(pos_c, arr)
        wp.copy(pos_e, arr)
        course_c._step_result.events.passed.fill_(-7)   # poison
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ev_e = course_e.step().events
        np.testing.assert_array_equal(
            course_c._step_result.events.passed.numpy(), ev_e.passed.numpy())
        np.testing.assert_array_equal(
            course_c._step_result.events.progress.numpy(),
            ev_e.progress.numpy())
        np.testing.assert_array_equal(
            course_c._step_result.contacts.oob.numpy(),
            course_e._step_result.contacts.oob.numpy())
```

- [ ] **Step 2: Run (GPU present — must RUN and PASS)**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_cuda.py -v`
Expected: 2 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_course_cuda.py
git commit -m "test: Course facade CUDA — Graph B refresh replay and user-captured step"
```

---

### Task 6: Docs + final verification

**Files:**
- Modify: `docs/reference/api.rst` (Course facade section at end), `docs/tutorials/runtime-utilities.rst` (closing "Putting it together" section)

**Interfaces:**
- Consumes: everything shipped in Tasks 1–5.

- [ ] **Step 1: api.rst — append at end of file**

```rst
Course facade
-------------

One object bundling generation, collision, and progress per mode; see the
:doc:`runtime utilities tutorial </tutorials/runtime-utilities>` for the
lifecycle.

.. automodule:: track_gen.course
   :no-members:

.. autoclass:: track_gen.course.CourseConfig
   :no-members:

.. autoclass:: track_gen.course.Course
   :members:

.. autoclass:: track_gen.course.StepResult
   :no-members:

   .. automethod:: clone

.. autofunction:: track_gen.course.set_capturing
```

- [ ] **Step 2: tutorial — append at end of `docs/tutorials/runtime-utilities.rst`**

```rst
Putting it together: the Course facade
--------------------------------------

Everything above can be wired by hand — or bundled by
``track_gen.course.Course``, which owns the orchestration invariants
(rebake/resample/posts rebuild and a full progress reset on every
regeneration; per-env respawns via a mask):

.. code-block:: python

   from track_gen import TrackGenConfig
   from track_gen.course import Course, CourseConfig

   course = Course(CourseConfig(
       mode="track",
       gen=TrackGenConfig(num_envs=E, device="cuda"),
       seeds=42,
       collision="segments",          # or "sdf" / None
       checkpoint_spacing=0.6,
   ))
   course.bind(position=robot_pos, yaw=robot_yaw, half_extents=robot_he)

   track = course.generate()          # whole batch + coherent refresh
   for _ in range(steps):
       sim.step()                     # writes the bound buffers in place
       res = course.step()            # events + contacts, no args
       course.reset(done_mask)        # respawn finished envs on the same course
   course.generate(seeds=next_seed)   # new courses for everyone

``generate()`` is whole-batch (the generator pipelines are fixed-batch
captured graphs); per-env control is ``reset(mask)``. In gates mode the same
object wraps ``GateGenerator`` + gate progress + optional ``DiscChecker``
posts (``post_radius > 0``). The underlying tools stay reachable
(``course.collision``, ``course.progress``, ``course.checkpoints``) and
``track_gen.course.set_capturing(True)`` flips every utility's capture flag
at once when you capture ``step()`` into your own sim graph.
```

- [ ] **Step 3: Build docs and run the complete suite**

Run: `python3 -m sphinx -b html docs /tmp/claude-1000/-home-antoine-Documents-track-gen/a3819d36-c82d-4063-bcba-b7abbecf061d/scratchpad/docs-build -q 2>&1 | tail -3`
Expected: clean (no new warnings).

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all PASS (cuda tests run on this machine)

- [ ] **Step 4: Commit**

```bash
git add docs/reference/api.rst docs/tutorials/runtime-utilities.rst
git commit -m "docs: Course facade — API reference and tutorial closing section"
```

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** uniform binding additions (Task 1 — spec's "binding is applied to the sub-tools when they exist; rebinding replaces"), CourseConfig strict validation incl. sentinels + gates gate_width>0 tightening (Task 2), Course lifecycle / deferred build / whole-batch generate + Graph B / per-env reset / StepResult same-instance / sub-tool attributes (Task 3), gates mode + device-side posts rebuild (Tasks 3–4), CUDA Graph B replay + user-side capture via set_capturing (Task 5), api.rst + tutorial (Task 6). Out-of-scope list respected (no per-env regen, no auto-captured step, no mixed mode, no props integration).
- **Type consistency:** `CourseConfig` fields and `Course` attribute names identical across Tasks 2–6; `bind(position, yaw, half_extents, box_position)` consistent between Task 3 code and Task 4–5 tests; `set_capturing` exported at module level and as a staticmethod.
- **Known deviations documented inline:** the Task 5 walrus-typo correction note; `_refresh` reads `self.collision._method` (private sibling access within the package — consistent with how tests already probe it; acceptable internal coupling, reviewer may weigh in).
