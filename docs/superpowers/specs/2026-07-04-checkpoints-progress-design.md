# Checkpoints, Progress Tracking & Disc Collision — Design

**Date:** 2026-07-04
**Status:** Approved
**Modules:** `track_gen.checkpoints`, `track_gen.progress`, `track_gen.collision` (extension)

## Goal

Three modular, Warp-native utilities that together provide "did the agent
advance along the course?" signals for BOTH gate sequences and tracks, plus
physical post/obstacle collision:

1. **`track_gen.checkpoints`** — a shared ordered-checkpoint abstraction
   (`CheckpointSet`) sourced from either a `GateSequence` (zero-copy) or a
   subsampled track centerline (`CheckpointSampler`, user-set spacing).
2. **`track_gen.progress`** — a stateful, source-agnostic `ProgressTracker`:
   pass-through detection, ordered progress/laps, wrong-way/wrong-checkpoint
   events, and `dist_to_next` for delta-distance rewards.
3. **`track_gen.collision.DiscChecker`** — oriented-box vs disc-obstacle
   collision (gate posts, physical cones, any point obstacles), in the
   existing collision family.

Plus a **documentation deliverable**: a narrative tutorial page with
deterministic figures explaining the whole utility family.

## Why this decomposition

Progress logic is identical for drone-racing gates and car-racing tracks:
the reward is typically the negative delta of the distance to the next goal,
plus discrete pass events. `Track` is index-aligned (`inner[i]`, `center[i]`,
`outer[i]` share a cross-section), so a subsampled centerline checkpoint
naturally carries a physical crossing segment `inner ↔ outer` — a virtual
gate. One tracker consumes both. Post collision is a collision-family
concern, not a progress concern, and generalizes to any disc obstacles.

## Requirements (shared)

- Warp-first: pure kernels; per-step methods (`update`, `query`) are
  allocation-free and host-sync-free under graph capture (module
  `_CAPTURING` + `_sync` pattern). warp-lang >= 1.14.
- One agent per env for progress ([E] state); collision `DiscChecker` is
  batched like `CollisionChecker` ([E * max_boxes] queries).
- Flat NaN-padded layouts, in-place result reuse + `clone()`, results
  undefined for `valid[e] == 0` envs — all per library convention.
- Runtime deps numpy + warp-lang only (numpy at construction time only).

### Stable-buffer / input-binding contract (graph capture)

The utilities integrate with a sim that owns persistent state arrays (e.g.
the robots' pose buffer). Two rules, uniform across the family:

1. **Everything the tool owns is preallocated with stable pointers**:
   internal state, scratch, and result buffers are allocated once in
   `__init__` and only ever written in place. (Already the convention;
   stated here as a hard requirement so CUDA graphs can bake addresses.)
2. **Per-step inputs can be BOUND once instead of passed per call.** Each
   per-step method gets a bind-once mode: the tool latches onto the user's
   stable wp.array(s) (same `.ptr` for the lifetime of the binding) and
   reads them in place on every call — no per-call marshalling, no per-call
   validation in the hot path:
   - `ProgressTracker(checkpoints, position=None)` — when `position` is
     bound at construction, `update()` takes no arguments and reads the
     bound buffer; `tracker.update(pos)` (per-call mode) remains available
     when unbound. Calling `update()` unbound, or `update(pos)` while
     bound, raises `ValueError`.
   - `DiscChecker(..., position=None, yaw=None, half_extents=None)` — all
     three bound together or not at all; `query()` no-arg form when bound.
   - **Retrofit** (same commit family): `CollisionChecker.bind_inputs(
     position, yaw, half_extents)` adds the identical bound mode to the
     existing box checker (`query()` no-arg form). Passing arrays per call
     stays supported and unchanged.
   - Shape/dtype/device validation happens once at bind time (construction
     or `bind_inputs`), never inside the bound-mode hot path.
   - Docs state the capture rule explicitly: under graph capture, per-call
     mode requires the SAME arrays (identical `.ptr`) at capture and every
     replay; bound mode makes this structural — bind once, capture, replay.

## Unit 1: `track_gen.checkpoints`

### `CheckpointSet`

Dataclass of `[E * M]` wp.arrays + `[E]` count. The consumer contract for
`ProgressTracker` (and anything else that wants ordered course goals):

| field      | type  | meaning                                          |
|------------|-------|--------------------------------------------------|
| `position` | vec2f | checkpoint center                                |
| `left`     | vec2f | crossing-segment endpoint (gate left / inner)    |
| `right`    | vec2f | crossing-segment endpoint (gate right / outer)   |
| `tangent`  | vec2f | unit forward (travel) direction                  |
| `count`    | int32 | `[E]` real checkpoints per env                   |

Slots `>= count[e]` are NaN-padded. Checkpoint order is index order.

- **`CheckpointSet.from_gates(seq: GateSequence) -> CheckpointSet`** —
  zero-copy: the returned set ALIASES `seq.position/left/right/tangent/count`
  (no kernel, no allocation; `M = max_gates`). Regenerated gates are seen
  automatically; progress state must be reset by the caller after a regen.
  Note: with `gate_width=0`, `left == right == position` — the crossing
  segment degenerates and pass-through cannot trigger; construction warns in
  the docstring (not a runtime check — the values are device data).
- **`CheckpointSampler(track, spacing, max_checkpoints=None)`** — facade in
  the `PropSampler` mold. `.sample() -> CheckpointSet` refreshes the owned
  buffers in place from the CURRENT track batch:
  - Centerline resample at arc spacing with the props snap rule:
    `n[e] = clamp(round(perimeter_e / spacing), 3, max_checkpoints)`,
    effective step `perimeter_e / n[e]` (perimeter = closed centerline
    polyline length over real points).
  - Checkpoint `k` at arc `k*step`: `position` on the centerline,
    `left`/`right` = `inner`/`outer` interpolated at the same polyline
    segment and parameter (the road cross-section, valid because the three
    polylines are index-aligned), `tangent` = unit direction of the
    containing centerline segment.
  - `max_checkpoints=None` derives `max(3, ceil(1.5 * max valid-env
    centerline perimeter / spacing))` (host readback at construction only);
    ValueError if no valid env.
  - Spacing is expected to be coarse ("a small number of checkpoints,
    similar in count to gates"); nothing enforces this, docs recommend it.
  - Degenerate env (track `count[e] < 3`): checkpoint `count = 0`, all NaN.
  - Validation mirrors `PropSampler`: `spacing` must satisfy
    `not (spacing > 0) -> ValueError` (NaN-proof), `max_checkpoints >= 3`
    when explicit, batch-layout sanity.

`CheckpointSet` itself carries only the five consumer-contract fields —
`truncated` and `step` live on the `CheckpointSampler` (`[E]` buffers,
refreshed by `sample()`), because they are producer diagnostics, not part of
the consumer contract (`from_gates` sets have no such notion).

## Unit 2: `track_gen.progress`

### `ProgressTracker`

```python
tracker = ProgressTracker(checkpoint_set)
events = tracker.update(position)   # [E] vec2f, per-call mode
tracker.reset(mask)                 # [E] int32, per-env episodic reset

# Bound mode: latch onto the sim's stable pose buffer once, then no-arg updates
tracker = ProgressTracker(checkpoint_set, position=robot_pos)  # stable wp.array
events = tracker.update()           # reads robot_pos in place every step
```

Owns device state `[E]`: `prev_pos` (vec2f, NaN = "no motion yet"),
`next_checkpoint` (int32), `laps` (int32), `progress` (int32).

**`update(position)`** — one fused kernel, thread per env:

- Skip inert: if `prev_pos` is NaN (first step after construction/reset),
  write `prev_pos = position`, emit no events (all zero/-1), still emit
  `next_checkpoint`, `laps`, `progress`, and `dist_to_next`.
- **Pass-through**: swept segment `prev_pos -> position` properly intersects
  the target's crossing segment `left[g] <-> right[g]`
  (`g = next_checkpoint[e]`) AND `dot(position - prev_pos, tangent[g]) > 0`.
  On pass: `passed = 1`, `checkpoint_passed = g`,
  `next_checkpoint = (g+1) % count[e]`, `progress += 1`, `laps += 1` when
  wrapping to 0. At most one advance per `update()`; the wrong-checkpoint
  scan (below) always runs against the ORIGINAL target's index set, so a
  step that jumps two checkpoints advances one AND reports the second as
  `wrong_checkpoint` in the same update (documented: call at the physics
  rate to avoid multi-gate jumps).
- **Wrong-way**: backward crossing of the target (intersects, negative dot)
  -> `wrong_way = 1`, no advance.
- **Wrong-checkpoint**: scan all other real checkpoints; if the swept
  segment crosses any non-target crossing segment (either direction),
  `wrong_checkpoint = that index` (first hit in index order), else -1.
- **`dist_to_next`**: `|position - position[next_checkpoint]|` after any
  advance this step. The delta-distance reward is
  `r_t = dist_to_next[t-1] - dist_to_next[t]` computed by the caller
  (documented with an example; the tracker deliberately does not difference
  it, so reward shaping stays in user land).
- `count[e] < 1`: all events inert (0 / -1 / NaN dist).

**`reset(mask)`** — kernel: where `mask[e] == 1`, set
`next_checkpoint = 0`, `laps = 0`, `progress = 0`, `prev_pos = NaN`.
The NaN sentinel guarantees the first post-reset `update()` cannot emit a
spurious crossing (teleport-safe). Callers MUST reset after regenerating the
bound gates/track (state refers to the old course; documented, mirrors the
sdf `bake()` contract).

### `ProgressEvents`

`[E]` wp.arrays, in-place reuse + `clone()`: `passed` (int32 0/1),
`checkpoint_passed` (int32 idx/-1), `next_checkpoint` (int32), `laps`
(int32), `progress` (int32 total passes), `wrong_way` (int32 0/1),
`wrong_checkpoint` (int32 idx/-1), `dist_to_next` (float32).

## Unit 3: `track_gen.collision.DiscChecker`

```python
posts = wp.array(...)  # [E * D] vec2f disc centers (e.g. gates.left/right interleaved)
checker = DiscChecker(discs=posts, radius=0.015, max_boxes=B, count=None)
result = checker.query(position, yaw, half_extents)   # per-call mode

# Bound mode: latch onto the sim's stable pose buffers once
checker = DiscChecker(discs=posts, radius=0.015, max_boxes=B,
                      position=pos_buf, yaw=yaw_buf, half_extents=he_buf)
result = checker.query()
```

- Binds a flat `[E * D]` vec2f disc-center array (aliased, so regenerated
  buffers are seen automatically), scalar `radius > 0`, and optional
  `count [E]` int32 array of real discs per env (`None` = NaN-marked: slots
  with NaN centers are skipped — matches how gate arrays are padded).
- `query(position, yaw, half_extents)` — same `[E * max_boxes]` box inputs
  and validation as `CollisionChecker.query`. Thread per box, loop over the
  env's discs: hit iff distance(disc center, solid OBB) <= radius (reuses
  `_point_to_local_box_dist`/`_rot2` from `collision_geom`).
- `DiscContact` result `[E * max_boxes]`: `hit` (int32 0/1), `disc` (int32
  index of deepest-penetration disc, -1 none), `depth` (float32, >= 0
  penetration; 0 when no hit), `nearest` (vec2f closest point on that
  disc's boundary to the box), all NaN/-1/0 for inactive (NaN-position)
  boxes. In-place reuse + `clone()`.
- Gate-post recipe (documented + tested, no dedicated helper — YAGNI): build
  the `[E * 2G]` disc array by interleaving `gates.left`/`gates.right`; the
  gate arrays' NaN padding carries over, so the `count=None` NaN-skip mode
  just works. The tutorial shows the two-line recipe.
- Lives in `track_gen/_src/collision.py`'s family: implementation in
  `track_gen/_src/collision_discs.py`, re-exported from
  `track_gen.collision` (public shim gains `DiscChecker`, `DiscContact`).

## Documentation deliverable

- **Tutorial page `docs/tutorials/runtime-utilities.rst`** — narrative
  walkthrough of the utility family: out-of-bounds collision (backend
  choice), props instancing, checkpoints from gates vs tracks, progress
  tracking + the `-delta dist_to_next` reward pattern (with a code snippet),
  disc obstacles for posts/cones. Added to the tutorials toctree.
- **Figures** (extend `viz/render_utility_assets.py`; fixed seeds, cpu,
  committed to `docs/assets/`, smoke-tested):
  1. `checkpoints-overview.png` — a track with subsampled centerline
     checkpoints and their inner<->outer crossing segments drawn as virtual
     gates, beside a gate-sequence-sourced CheckpointSet.
  2. `progress-tracking.png` — scripted agent trajectory threading
     checkpoints: path colored by progress, passed checkpoints marked,
     current target highlighted, inset `dist_to_next`-vs-step sawtooth.
  3. `disc-collision.png` — gate posts as discs, agent boxes colored
     hit/miss with penetration annotations.
- API reference: three new sections (Checkpoints, Progress, extended
  Collision) in `docs/reference/api.rst`.

## File layout

- `track_gen/_src/checkpoints.py` — `CheckpointSet`, `CheckpointSampler`,
  resample kernels (scan reuses the props scan pattern on the centerline).
- `track_gen/_src/progress.py` — `ProgressTracker`, `ProgressEvents`,
  fused update kernel + reset kernel.
- `track_gen/_src/collision_discs.py` — `DiscChecker`, `DiscContact`,
  query kernel.
- Shared geometry additions to `track_gen/_src/collision_geom.py`: proper
  segment-segment intersection with orientation (for pass-through).
- Public shims: `track_gen/checkpoints.py`, `track_gen/progress.py`;
  `track_gen/collision.py` re-exports the disc classes.
- `track_gen/__init__.py` + curated-surface test: add `checkpoints`,
  `progress`.

## Testing

- **Checkpoints**: annulus analytics (positions on centerline circle,
  left/right on inner/outer circles, tangents perpendicular to radius, snap
  counts); `from_gates` aliasing (mutate gate buffers via wp.copy -> set
  sees it, zero-copy identity of `.ptr`); degenerate/truncation/derivation;
  numpy oracle on generated tracks (reuses the props oracle machinery for
  the centerline + independent left/right interpolation check).
- **Progress**: hand-built square course with a scripted path — exact
  expected event sequence (passes, laps, progress, dist_to_next values);
  wrong-way path; wrong-checkpoint (skip-a-gate) path; reset-mask semantics
  (no spurious crossing after teleport-reset); count<1 inert; oracle random
  walk vs numpy reference on generated gates AND on track checkpoints (the
  same tracker on both sources).
- **DiscChecker**: analytic disc/box cases (face hit, corner hit, graze at
  exactly radius, deepest-disc argmax, NaN discs skipped, NaN boxes inert);
  gate-post recipe end-to-end (build posts from a generated GateSequence,
  hit a post, verify index maps back to the gate).
- **Input binding**: bound-mode equivalence (bound `update()`/`query()`
  results identical to per-call mode on the same data); bind-time
  validation errors; mode-misuse errors (no-arg while unbound, arg while
  bound); writing new poses into the bound buffer between calls is
  reflected without re-binding; `CollisionChecker.bind_inputs` retrofit.
- **CUDA**: graph capture for `update()`, `reset()`, `sample()`, and disc
  `query()` with poisoned-buffer replay; the progress + disc capture tests
  run in BOUND mode (the intended graph pattern: sim writes the pose
  buffer, replays the captured update); cuda-marked.
- **Docs**: asset-renderer smoke tests for the three figures; sphinx build
  clean; tutorial page in toctree.

## Out of scope (YAGNI)

- Multi-agent per env progress (add an agent stride later if needed).
- Multiple simultaneous advances per step (document single-advance).
- Non-disc obstacle shapes; disc-disc or box-box collision.
- Reward computation itself (the tracker emits `dist_to_next`; deltas are
  the caller's).
- Automatic reset detection after course regeneration.
