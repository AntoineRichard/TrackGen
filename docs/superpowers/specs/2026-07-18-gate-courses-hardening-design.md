# Gate Courses Follow-Up Hardening — Design

**Date:** 2026-07-18
**Status:** Approved
**Modules:** `track_gen._src.types`, `_src.warp_gate`, `_src.checkpoints`, `_src.course`, `_src.collision_frames`, `tests/`, `docs/`

## Goal

Close out the triaged follow-ups from the 3D gate courses merge (commits
`2a10742..3e5944e`; triage list at the end of `.superpowers/sdd/progress.md`).
Two behavior changes with regression gates, then tests, docs, and polish.
All nine remaining items are covered; nothing else from that list stays open.

## 1. Pass-plane alignment (behavior change)

Problem (final-review Minor 2): progress pass detection uses the 3D spline
tangent as the crossing-plane normal, while `yaw_only` gates are physically
upright — on sloped yaw-only courses the two disagree in a band of width
about `half_size * (1 - cos(pitch))`.

Fix: pass detection uses the gate's PHYSICAL plane in both alignment modes.

- **`GateSequence.forward`** — new field, `[E * max_gates] wp.vec3f`,
  NaN-padded: the pose's x-axis. Written in `_finalize_frame_k`
  **analytically** (not via quat rotation, for planar bit-exactness):
  `full_tangent` → the unit tangent; `yaw_only` with `tangent.z == 0` →
  `forward = tangent` VERBATIM (no re-normalization — dividing an
  already-unit vector by its ~1.0 length can perturb bits; this branch
  mirrors the kernel's existing analytic planar left-axis branch);
  `yaw_only` with `tangent.z != 0` → `normalize(tangent.x, tangent.y, 0)`;
  near-vertical fallback → the fallback chain's forward. Planar behavior
  and the goldens are therefore bit-unaffected (the golden npz does not
  include the field; it is NOT regenerated).
- **`CheckpointSet.from_gates`** aliases `forward` into its `tangent` slot
  (the documented "unit forward travel direction" — now the physical gate
  normal). The track-sampler producer is unchanged. `clone()`, dtype
  validation, allocation (NaN fill), and docstrings updated accordingly.
- Consequence: `_plane_pass` (unchanged) now receives the upright plane for
  yaw-only gates, agreeing exactly with `FrameChecker`'s boxes;
  `dist_to_next` and wrong-way semantics follow the same plane.
- Docs: the `gates-3d.rst` yaw-only caveat collapses to one sentence: pass
  detection uses the gate's physical plane in both modes.

Regression gate: a steeply sloped yaw-only course fixture where a
trajectory crosses inside the UPRIGHT opening but outside the
tangent-tilted plane's opening — asserts it now registers a pass — plus the
mirror case (inside tilted, outside upright → no pass). Planar golden test
and full suite stay green.

## 2. Capture-race fix (behavior change)

Problem (Task 10 root cause): in `Course.generate()` the first-call work —
reseed `wp.array` construction, `_build_subtools()` (subtool allocations +
eager launches), and the CPU-path eager `_refresh` — runs OUTSIDE
`runtime._CAPTURE_LOCK`. Two threads constructing fresh Courses and calling
their first `generate()` concurrently on cuda flake ~25% (CUDA-700 in
async frees racing another thread's capture).

Fix: move that first-call work under the lock. Ordering constraint: the
lock is NOT reentrant and `generator.generate()` takes it internally, so
the sequence is — `generator.generate()` (self-locked, outside) → acquire
`_CAPTURE_LOCK` → reseed-array creation, `_build_subtools()`, eager
`_refresh` / warmup / capture / replay exactly as today → release. The
cuda warmup/capture region already holds the lock; this closes the
construction gap. A short comment at the lock site documents the coverage
contract (replacing the planned TODO). The reseed path runs on every
`generate(seeds=...)`, not only the first call — its array construction
moves under the lock uniformly.

Proof: the previously-flaky test shape — two threads, fresh gates-mode
Courses (`frame_collision=True`), concurrent FIRST `generate()` + `step()`
— is added to `tests/test_generate_concurrent_cuda.py` and stress-run
(~10 repeats) during implementation; it must be deterministically green.
The existing serialized-warmup test is renamed from "capture" to "replay"
wording (name + docstring + commit message accuracy), per final review.

## 3. Tests, docs, polish

Tests:
- **`frame_fallbacks` behavioral test**: drive a near-vertical
  `full_tangent` gate through `finalize_gate_sequence` and assert the
  per-env counter reads 1; a planar course asserts 0.
- **Progress-oracle independence fixtures**: ~6 hand-computed crossing
  cases (forward pass, backward crossing, edge-touch on the plane,
  crossing outside the u-opening, v-half-bounded rejection, NaN pause)
  asserted against LITERAL expected event values — not against
  `tests/_progress_oracle.py`, which mirrors the kernel.
- **Closing-chord grade**: verify the existing Task 4 test covers the
  wraparound chord directly; extend only if coverage is indirect.

Docs:
- `gates-3d.rst`: constant-parameter sampling note (CourseLine samples
  uniform Catmull-Rom parameters per gate segment, not constant 3D
  arc-length; `arclen` is nevertheless true arc length) + the Section 1
  caveat simplification. Plan deviation list gains the sampling item
  (docs/superpowers/plans/2026-07-17-3d-gate-courses.md, self-review
  section).
- `CourseConfig` `max_boxes > 1` error message mentions frame collision
  ("frame_collision uses sphere binding; max_boxes applies to box/disc
  checkers").

Polish:
- `FrameChecker` NaN guard switches from `p[0] != p[0]` to `_is_nan3`
  (convention consistency; no behavior change).

## Non-goals

- No changes to the 2D pipeline, the goldens npz, or `Track` semantics.
- No corridor-tube work (next stage, unchanged).
- `step()`-time locking (user-side concurrency during another thread's
  capture) stays out of scope — the lock contract covers generate-time
  work only.

## Acceptance

Full suite green (cpu + cuda) including the new concurrent-construction
stress test; `tests/test_golden_migration.py` exact; `sphinx -W` clean;
every item on the `.superpowers/sdd/progress.md` follow-up list either
closed by this work or explicitly listed here as out of scope (only the
residual `step()`-time concurrency note remains).
