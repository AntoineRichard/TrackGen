# Course Facade — Design

**Date:** 2026-07-04
**Status:** Approved
**Module:** `track_gen.course`

## Goal

One object that bundles generation, collision, and checkpoint/progress per
main mode, so a sim integrates the whole track_gen runtime with four calls:
construct → `bind()` → `generate()` → per-step `step()` / per-env
`reset(mask)`. The facade wires the existing utilities (it adds no new
geometry) and owns the orchestration invariants that are currently the
caller's burden: rebake/resample after regeneration, full progress reset when
courses change, posts rebuilt from regenerated gates.

## Modes

- **`mode="track"`**: `TrackGenerator` (+ `PerEnvSeededRNG`) →
  out-of-bounds `CollisionChecker` (`collision="segments" | "sdf" | None`) →
  `CheckpointSampler(checkpoint_spacing)` → `ProgressTracker`.
- **`mode="gates"`**: `GateGenerator` → `CheckpointSet.from_gates` →
  `ProgressTracker`; optional `DiscChecker` gate-post collision enabled by
  `post_radius > 0`. The posts array (`[E * 2G]`, interleaved left/right) is
  facade-owned and rebuilt by a Warp kernel during refresh (device-side, so
  the refresh sequence stays graph-capturable; the tutorial's host-side
  interleave recipe remains valid for standalone use).

## Public API

```python
from track_gen.course import Course, CourseConfig, StepResult

course = Course(CourseConfig(
    mode="track",
    gen=TrackGenConfig(num_envs=E, device="cuda"),
    seeds=42,
    collision="segments",        # track mode: "segments" | "sdf" | None
    sdf_resolution=128,          # sdf only
    post_radius=0.0,             # gates mode: > 0 enables DiscChecker
    checkpoint_spacing=0.6,      # track mode only
    max_checkpoints=None,        # track mode; None = CheckpointSampler auto
    max_boxes=1,                 # collision query stride
))
course.bind(position=pos_buf, yaw=yaw_buf, half_extents=he_buf)
track = course.generate()        # whole batch; returns Track / GateSequence
res = course.step()              # StepResult(events, contacts)
course.reset(done_mask)          # per-env respawn on the SAME course
```

### `CourseConfig`

Dataclass; `__post_init__` validates:

- `mode in {"track", "gates"}`; `gen` type must match the mode
  (`TrackGenConfig` ↔ track, `GateGenConfig` ↔ gates) — `ValueError`
  otherwise.
- `seeds`: int or wp.array, forwarded to `PerEnvSeededRNG(seeds, num_envs,
  device)` (both taken from `gen`).
- Applicability is strict, not silently ignored: `checkpoint_spacing` /
  `max_checkpoints` / `collision` / `sdf_resolution` set in gates mode →
  `ValueError`; `post_radius > 0` in track mode → `ValueError`.
  Inapplicable fields default to sentinels (`None` / `0.0`) so the common
  construction never trips.
- track mode requires `checkpoint_spacing > 0` (NaN-proof); `collision=None`
  disables OOB checks (progress-only bundles are legal).
- `max_boxes >= 1`; box-collision options require max_boxes agreement with
  the bound buffers at `bind()` time.

### `Course`

- `__init__(config)`: builds the generator + rng eagerly; defers
  collision/checkpoints/progress construction until after the FIRST
  `generate()` (CheckpointSampler's auto `max_checkpoints` derivation needs a
  real batch, and `CollisionChecker(sdf)` bakes at construction). Before the
  first `generate()`, `step()`/`reset()` raise `RuntimeError("call
  generate() first")`.
- `bind(position, yaw=None, half_extents=None)`: REQUIRED before `step()`
  (the facade is bound-mode only — its purpose is the stable-buffer sim
  integration; per-call arrays remain available on the underlying tools).
  `position` is `[E]` vec2f for progress; `yaw`/`half_extents` (+ a
  `[E*max_boxes]` position view) are required iff a box-vs-something checker
  is enabled. For `max_boxes == 1` the SAME `[E]` position buffer serves both
  progress and collision (documented; stride check enforces it). For
  `max_boxes > 1`, `bind` takes an additional `box_position` `[E*max_boxes]`
  buffer and `position` remains the `[E]` agent buffer driving progress.
  May be called before or after the first `generate()`; binding is applied
  to the sub-tools when they exist (rebinding replaces it).
- `generate(seeds=None) -> Track | GateSequence`: whole-batch (generator
  fixed-batch constraint, documented). Sequence: optional
  `rng.set_seeds_warp(seeds)` → generator `generate()` (its own pipeline
  graph on cuda) → **refresh**: sdf `bake()` (if sdf), checkpoint `sample()`
  (track mode), posts rebuild (gates mode + posts), then FULL progress reset
  (persistent all-ones mask buffer) — every course changed, so all progress
  state is invalid. First call constructs the deferred sub-tools and, on
  cuda, captures the refresh sequence into the facade-owned Graph B;
  subsequent calls replay it. On cpu everything runs eagerly.
- `step() -> StepResult`: `progress.update()` + collision `query()` (when
  enabled), both in bound mode; returns the facade's `StepResult`. No
  allocation, no host sync under capture (module `_CAPTURING` flag mirrors
  the sub-modules'; setting the facade flag sets the sub-tools' flags too via
  a small helper so a user capture needs ONE switch).
- `reset(mask)`: forwards to `progress.reset(mask)` — per-env respawn on the
  same course. Collision/checkpoints are course-derived and unaffected by
  respawns; nothing else to reset.
- Sub-tool access: `course.generator`, `course.rng`, `course.collision`
  (`CollisionChecker | DiscChecker | None`), `course.checkpoints`
  (`CheckpointSet`), `course.checkpoint_sampler` (track mode, else `None`),
  `course.progress`, `course.result` (the live `Track`/`GateSequence`, None
  before first generate).

### `StepResult`

Dataclass: `events: ProgressEvents`, `contacts: BoxContact | DiscContact |
None`. Holds the sub-tools' in-place instances (same-instance contract
cascades: the same `StepResult` is returned every `step()`); `clone()`
deep-copies both.

## Graphs

- **Graph A** (existing, untouched): the generator's internal pipeline
  capture.
- **Graph B** (facade-owned, cuda only): the refresh sequence — bake and/or
  resample and/or posts rebuild + full progress reset. Captured on the first
  `generate()` (after eager warmup runs, following the TrackGenerator warmup
  pattern), replayed on subsequent ones. All refresh inputs/outputs are
  stable buffers, so capture is sound.
- `step()`/`reset()` are NOT auto-captured: they are capture-ready for the
  user's own sim graph. The facade exposes `track_gen.course._CAPTURING`
  plus a `Course.set_capturing(flag)` helper that toggles the facade's and
  all sub-modules' flags at once (one switch for user captures).

## Error handling

- Mode/config mismatches and inapplicable options: `ValueError` at
  `CourseConfig` construction.
- `step()`/`reset()` before first `generate()`: `RuntimeError`.
- `step()` without `bind()`: `RuntimeError` naming `bind`.
- `bind()` validation reuses the sub-tools' validators (shape/dtype/device,
  stride agreement with `max_boxes`).
- Invalid envs: unchanged library contract (results undefined; callers gate
  on `valid` from `course.result`).

## Files

- `track_gen/_src/course.py` — `CourseConfig`, `StepResult`, `Course`, the
  posts-rebuild kernel, `set_capturing` helper.
- `track_gen/course.py` — public shim.
- `track_gen/__init__.py` + curated-surface test: add `course`.
- Docs: api.rst "Course facade" section; runtime-utilities tutorial gains a
  closing "Putting it together" section showing the facade replacing the
  manual wiring (both forms kept).

## Testing

- **End-to-end, both modes, cpu**: construct → bind → generate → step loop
  (events/contacts sane vs direct sub-tool calls on the same buffers) →
  partial `reset(mask)` (progress zeroed exactly where masked, elsewhere
  intact) → `generate()` again (checkpoint counts change with new geometry;
  sdf probe box flips OOB state after regen proving rebake; posts follow the
  new gates; progress fully reset).
- **Deferred-construction contract**: step/reset before generate raise;
  bind-before-generate works; step without bind raises.
- **Config validation matrix**: every inapplicable-option combination.
- **Equivalence**: facade `step()` results identical to manually-wired
  sub-tools driven with the same buffers and sequence (both modes).
- **CUDA**: Graph B replay correctness (regenerate twice; poison refreshed
  buffers between capture and replay); user-side capture of
  `step()` + `reset(mask)` via `set_capturing` with poisoned replay.
- Docs build clean; asset/tutorial references intact.

## Out of scope (YAGNI)

- Per-env regeneration (blocked by the generators' fixed-batch design).
- Auto-capturing `step()` into a facade-owned graph.
- Mixed mode (track + gates on the same envs).
- Props/rendering integration (PropSampler stays standalone; it has no
  per-step role).
- Multi-agent strides for progress (one agent per env, as in progress).
