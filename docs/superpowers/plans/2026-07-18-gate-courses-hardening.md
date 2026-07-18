# Gate Courses Follow-Up Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the triaged follow-ups from the 3D gate courses merge: align yaw-only pass detection with the physical gate plane, close the first-call capture-race window, and land the outstanding tests, docs, and polish.

**Architecture:** Two behavior changes land first, each with its own regression gate — a new `GateSequence.forward` field consumed by `CheckpointSet.from_gates` (pass-plane alignment), and `_CAPTURE_LOCK` coverage extended over `Course.generate()`'s first-call construction (race fix, proven by the previously-flaky concurrent-construction test). Tests and docs/polish follow as mechanical tasks.

**Tech Stack:** Python, NVIDIA Warp (warp-lang >= 1.14), pytest, Sphinx. No new dependencies.

**Execution model policy (user-mandated):** implementer and fixer subagents run on Opus (integration/judgment) or Sonnet (transcription/mechanical) ONLY; Fable is used exclusively for reviews. Suggested: Task 1 Opus, Task 2 Opus, Task 3 Sonnet, Task 4 Sonnet.

## Global Constraints

- Work on a new branch `feat/gate-hardening` off `main` (`d8aec13` or later).
- The golden gate is inviolable: `pytest tests/test_golden_migration.py -v` must pass with EXACT XY equality after every task; `tests/goldens/pre_vec3f.npz` is frozen — never regenerated or extended.
- Capturable paths (anything reachable from `_run_gate_pipeline` or `Course._refresh`) stay allocation-free, host-sync-free, free of host branches on device data. `runtime._CAPTURE_LOCK` is NOT reentrant; `GateGenerator.generate()` / `TrackGenerator.generate()` take it internally.
- NaN-padding convention: any NaN component marks a padded slot; new arrays follow it.
- This machine has a GPU — run `pytest tests/ -q` (cpu+cuda together, currently 595 tests) per task, plus `python -m sphinx -W -b html docs /tmp/sphinx_check` for any docs-touching task.
- Commits: conventional style; GPG signing is broken in this session — use `git commit --no-gpg-sign`. Append to every commit message:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ABYQDYMWzWSk9H6aJ1x8p
```

---

### Task 1: Pass-plane alignment (`GateSequence.forward`)

**Files:**
- Modify: `track_gen/_src/types.py` (GateSequence dataclass: field, docstring, `clone()`)
- Modify: `track_gen/_src/warp_gate.py` (`alloc_gate_sequence`, `_finalize_frame_k` ~360-416, its launch site in `finalize_gate_sequence`, `_finalize_validity_k` finite checks)
- Modify: `track_gen/_src/checkpoints.py:72-81` (`from_gates`)
- Modify: `docs/gates-3d.rst` (one sentence, see Step 6)
- Test: `tests/test_progress_gate_plane.py` (new), plus additions to `tests/test_types.py`

**Interfaces:**
- Produces: `GateSequence.forward` — flat `[E * max_gates] wp.vec3f`, NaN-padded: the gate pose's x-axis (physical plane normal). `CheckpointSet.from_gates` now aliases `tangent=seq.forward`. Task 3's hand-computed fixtures and any consumer of gate checkpoints rely on this: the crossing-plane normal IS the pose forward in both alignment modes.
- Bit-exactness contract: for `full_tangent`, and for `yaw_only` with a planar tangent (`tangent.z == 0`), `forward` equals the stored unit `tangent` VERBATIM — no re-normalization (dividing an already-unit vector by its ~1.0 length can perturb bits). Only sloped `yaw_only` gates get `normalize(tangent.x, tangent.y, 0)`; near-vertical fallback gates get the fallback chain's forward.

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_progress_gate_plane.py`:

```python
"""Pass detection uses the gate's PHYSICAL plane (pose forward), not the
3D spline tangent. On sloped yaw-only gates the two differ; these fixtures
pin the aligned semantics with hand-computed geometry."""
import numpy as np
import warp as wp

from track_gen._src.checkpoints import CheckpointSet
from track_gen._src.progress import ProgressTracker
from track_gen._src.types import GateSequence

E, G = 1, 1
SQ2 = np.float32(1.0 / np.sqrt(2.0))


def _sloped_yaw_only_gate():
    """One gate at origin: spline tangent pitched 45 deg ((s2,0,s2)), pose
    yaw-only upright (forward +x, identity quat), half_size 1, posts at
    y = +/-1."""
    dev = "cpu"
    return GateSequence(
        position=wp.array(np.array([[0, 0, 0]], np.float32), dtype=wp.vec3f, device=dev),
        tangent=wp.array(np.array([[SQ2, 0, SQ2]], np.float32), dtype=wp.vec3f, device=dev),
        forward=wp.array(np.array([[1, 0, 0]], np.float32), dtype=wp.vec3f, device=dev),
        orientation=wp.array(np.array([[0, 0, 0, 1]], np.float32), dtype=wp.quatf, device=dev),
        half_size=wp.array(np.array([1.0], np.float32), dtype=wp.float32, device=dev),
        left=wp.array(np.array([[0, 1, 0]], np.float32), dtype=wp.vec3f, device=dev),
        right=wp.array(np.array([[0, -1, 0]], np.float32), dtype=wp.vec3f, device=dev),
        valid=wp.array(np.array([1], np.int32), device=dev),
        count=wp.array(np.array([1], np.int32), device=dev),
    )


def _step(tracker, pos, xyz):
    p = pos.numpy()
    p[0] = xyz
    wp.copy(pos, wp.array(p, dtype=wp.vec3f, device="cpu"))
    return tracker.update()


def test_upright_crossing_passes_despite_tilted_tangent():
    # prev (-0.5, 0, 0.9) -> pos (0.5, 0, 0.9): crosses the UPRIGHT plane
    # x = 0 at (0, 0, 0.9), inside the opening (|u|=0, |v|=0.9 <= 1).
    # Against the tilted tangent plane (normal (s2,0,s2)) both endpoints
    # are on the positive side (d = (x+z)/sqrt(2): 0.283 and 0.990) — the
    # OLD semantics saw no crossing at all.
    seq = _sloped_yaw_only_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 0.0, 0.9])          # arms prev_pos
    ev = _step(tr, pos, [0.5, 0.0, 0.9])
    assert int(ev.passed.numpy()[0]) == 1
    assert int(ev.checkpoint_passed.numpy()[0]) == 0


def test_tilted_only_crossing_no_longer_passes():
    # prev (0.2, 0, -0.5) -> pos (0.8, 0, -0.5): x stays positive so the
    # upright plane x = 0 is never crossed; but x + z changes sign
    # (-0.3 -> 0.3), i.e. the OLD tilted-tangent plane WAS crossed at
    # (0.5, 0, -0.5) with |u|=0, old-v = 0.707 <= 1 — the old semantics
    # counted a pass here. Aligned semantics: no event of any kind.
    seq = _sloped_yaw_only_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [0.2, 0.0, -0.5])
    ev = _step(tr, pos, [0.8, 0.0, -0.5])
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
```

Also add to `tests/test_types.py` (in the GateSequence-fields test area): assert `GateSequence` has a `forward` field and that `clone()` deep-copies it (`clone.forward.ptr != seq.forward.ptr`), following the file's existing pattern for `orientation`/`half_size`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_progress_gate_plane.py -v`
Expected: FAIL at construction — `GateSequence.__init__() got an unexpected keyword argument 'forward'`.

- [ ] **Step 3: Implement**

1. `types.py` — add `forward: wp.array` to `GateSequence` between `tangent` and `orientation` (field, numpydoc entry: "flat `[E * max_gates]` `vec3f` unit pose forward (x-axis of `orientation`): the physical gate-plane normal used for pass detection; equals `tangent` for `full_tangent` and for planar `yaw_only` gates, the horizontal projection of `tangent` for sloped `yaw_only` gates; NaN-padded"), and `forward=wp.clone(self.forward)` in `clone()`.
2. `warp_gate.py` `alloc_gate_sequence` — allocate `forward=wp.empty(flat, dtype=wp.vec3f, device=dev)` and NaN-fill it alongside the other vec3 fields.
3. `warp_gate.py` `_finalize_frame_k` — add `forward: wp.array(dtype=wp.vec3f)` parameter (after `tangent`); NaN it in the padding branch; compute it in the body. The body currently reads (lines 388-416); change to:

```python
    p = position[t]
    tan = _safe_normalize3(tangent[t])
    fwd = tan
    if align_full == 0:
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
    fell = int(0)
    horiz2 = fwd[0] * fwd[0] + fwd[1] * fwd[1]
    if horiz2 < 1.0e-10:
        # Near-vertical (full_tangent on a steep segment, or degenerate
        # tangent): fall back to the horizontal tangent direction, then +x.
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
        if fwd[0] * fwd[0] + fwd[1] * fwd[1] < 1.0e-10:
            fwd = wp.vec3f(1.0, 0.0, 0.0)
        wp.atomic_add(fallbacks, e, 1)
        fell = int(1)
    fwd = _safe_normalize3(fwd)
    q = _frame_quat(fwd)
    hs = 0.5 * gate_width
    la = wp.quat_rotate(q, wp.vec3f(0.0, 1.0, 0.0))
    if align_full == 0 and tan[2] == 0.0:
        # Planar tangent, yaw-only frame: the left axis is analytic. Using it
        # directly (instead of the quat round-trip, which is only equal to
        # within rounding) keeps left/right bit-identical to the legacy 2D
        # path: left = p + hs * (-tan.y, tan.x, 0). This also reproduces the
        # legacy degenerate-tangent result (tan == 0 -> left == right == p).
        la = wp.vec3f(-tan[1], tan[0], 0.0)
    # Pose forward (physical gate-plane normal). VERBATIM tan — never a
    # re-normalization of it — whenever the pose forward IS the tangent
    # (full_tangent, or planar yaw_only), so progress plane normals stay
    # bit-identical to the tangent on those paths.
    fw = fwd
    if fell == 0:
        if align_full == 1:
            fw = tan
        elif tan[2] == 0.0:
            fw = tan
    tangent[t] = tan
    forward[t] = fw
    orientation[t] = q
    half_size[t] = hs
    left[t] = p + hs * la
    right[t] = p - hs * la
```

4. Update the `_finalize_frame_k` launch in `finalize_gate_sequence` to pass `gates.forward`, and extend `_finalize_validity_k`'s finite-field checks to include the `forward` components (same pattern as `tangent`).
5. `checkpoints.py` `from_gates` — alias the pose forward:

```python
        return cls(position=seq.position, left=seq.left, right=seq.right,
                   tangent=seq.forward, up_half=seq.half_size, count=seq.count)
```

and update its docstring line to "``forward`` (as ``tangent``: the physical gate-plane normal)". Update the `CheckpointSet.tangent` attribute docstring's gate clause similarly.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_progress_gate_plane.py tests/test_types.py tests/test_gate_3d.py tests/test_warp_gate.py tests/test_gate_generator.py tests/test_course_gates_3d.py tests/test_golden_migration.py -q` then `pytest tests/ -q`
Expected: all PASS (goldens exact — planar `forward == tangent` verbatim; full suite currently 595 + your new tests).

- [ ] **Step 5: One-sentence doc**

In `docs/gates-3d.rst`, in the runtime/pass-detection prose (search for the paragraph describing plane-crossing detection), add: "Pass detection uses the gate's *physical* plane — the pose forward — in both alignment modes, so a ``yaw_only`` gate on a sloped course is crossed exactly where its upright frame stands." Build docs: `python -m sphinx -W -b html docs /tmp/sphinx_check` → must succeed.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/types.py track_gen/_src/warp_gate.py track_gen/_src/checkpoints.py docs/gates-3d.rst tests/test_progress_gate_plane.py tests/test_types.py
git commit --no-gpg-sign -m "feat!: gate pass detection uses the pose forward (physical plane) in both alignment modes"
```

---

### Task 2: Capture-lock extension over first-call construction

**Files:**
- Modify: `track_gen/_src/course.py:431-499` (`generate()`)
- Modify: `tests/test_generate_concurrent_cuda.py` (rename existing gates test; add fresh-construction test)

**Interfaces:**
- Consumes: `runtime._CAPTURE_LOCK` (non-reentrant; `generator.generate()` takes it internally — MUST stay outside any region this task locks).
- Produces: contract, documented at the lock site: ALL device work in `Course.generate()` except `generator.generate()` itself runs under `_CAPTURE_LOCK` — reseed array construction/copy, `_build_subtools()` allocations + eager launches, and every `_refresh` (eager, warmup, capture, replay).

- [ ] **Step 1: Write the failing (flaky-exposing) test**

Add to `tests/test_generate_concurrent_cuda.py` (same markers/thread-harness pattern as the existing tests in the file — reuse its helpers; keep the existing tests byte-identical except the rename in Step 4):

```python
def test_concurrent_gates_3d_fresh_construction() -> None:
    """Two threads each build a FRESH gates-mode 3D Course and run their
    FIRST generate() + step() concurrently — construction, subtool
    allocation, warmup, capture, and replay all racing. This was ~25%
    CUDA-700 before generate()'s first-call work moved under
    runtime._CAPTURE_LOCK; it must now be deterministically green."""
    errors: list = []

    def worker(seed: int) -> None:
        try:
            course, pos = _make_gates_3d_course(seed)   # existing helper: builds config+Course+bind, NO generate
            course.generate()
            course.step()
        except Exception as exc:  # noqa: BLE001 — capture for main-thread assert
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(11,))
    t2 = threading.Thread(target=worker, args=(37,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors, errors
```

If the file's existing gates helper performs generate()/warmup internally, split it so a construction-only variant exists (keep the serialized-warmup helper for the replay test). Match the file's existing cuda/slow markers and any device-guard skips exactly.

- [ ] **Step 2: Run to verify it exposes the race**

Run: `for i in 1 2 3 4 5 6 7 8; do pytest tests/test_generate_concurrent_cuda.py::test_concurrent_gates_3d_fresh_construction -q || echo "FLAKED run $i"; done`
Expected: at least one FLAKED run (CUDA-700/allocation error) against the unfixed code. If 8 runs stay green, raise to 15 before concluding; record the observed rate in your report either way.

- [ ] **Step 3: Implement the lock extension**

Restructure `Course.generate()` (current body at `course.py:431-499`). The reseed block, `_build_subtools()`, and the cpu-path eager first `_refresh()` move under the lock; `generator.generate()` stays outside; the already-locked cuda regions are unchanged:

```python
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
```

The follow-on cuda capture block (`if self._is_cuda and self._refresh_graph is None:` with warmup/ScopedCapture/replay) is byte-unchanged. Note the diff: today `_build_subtools()` runs unlocked and cuda-vs-cpu branches around the first `_refresh()`; afterwards both are under one locked region and the `_is_cuda` distinction for that first refresh disappears (locking the cpu path too is harmless and uncontended). Update the surrounding comments exactly as shown — they ARE the documented coverage contract the spec requires.

- [ ] **Step 4: Rename the replay test**

In the same file: `test_concurrent_gates_3d_courses` → `test_concurrent_gates_3d_replay`; adjust its docstring's first line to say it exercises the REPLAY path (construction/warmup serialized by design — that scenario is now covered by `test_concurrent_gates_3d_fresh_construction`). No assertion changes.

- [ ] **Step 5: Verify the fix**

Run: `for i in $(seq 1 10); do pytest tests/test_generate_concurrent_cuda.py -q || echo "FLAKED run $i"; done`
Expected: 10/10 green, zero FLAKED lines. Then `pytest tests/ -q` and `pytest tests/test_golden_migration.py -q` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/course.py tests/test_generate_concurrent_cuda.py
git commit --no-gpg-sign -m "fix: Course.generate() first-call construction and reseed run under _CAPTURE_LOCK"
```

---

### Task 3: Test closeout (fallback counter, oracle independence, closing chord)

**Files:**
- Test: `tests/test_gate_3d.py` (frame_fallbacks additions; closing-chord check)
- Test: `tests/test_progress_handcomputed.py` (new)

**Interfaces:**
- Consumes: `finalize_gate_sequence(gates, config, ...)` in `track_gen/_src/warp_gate.py` (read its CURRENT signature first — it gained align/fallbacks plumbing in the 3D work) and `GateGenerator.frame_fallbacks`; `GateSequence.forward` from Task 1; `ProgressTracker`/`CheckpointSet` as in Task 1's test.

- [ ] **Step 1: frame_fallbacks behavioral test**

Add to `tests/test_gate_3d.py`:

```python
def test_frame_fallback_counter_increments_on_near_vertical():
    """A near-vertical full_tangent gate must engage the +x fallback and
    count it; a planar course must count zero."""
    import warp as wp
    from track_gen._src import warp_gate
    from track_gen._src.collision_geom import _safe_normalize3  # noqa: F401 (kernel dep)

    E1, G1 = 1, 2
    dev = "cpu"
    nan3 = np.full(3, np.nan, np.float32)
    pos = wp.array(np.array([[0, 0, 0], [1, 0, 0]], np.float32), dtype=wp.vec3f, device=dev)
    # gate 0 tangent is (numerically) vertical; gate 1 planar
    tan = wp.array(np.array([[1e-8, 0, 1], [1, 0, 0]], np.float32), dtype=wp.vec3f, device=dev)
    forward = wp.array(np.tile(nan3, (G1, 1)), dtype=wp.vec3f, device=dev)
    quat = wp.array(np.tile(np.full(4, np.nan, np.float32), (G1, 1)), dtype=wp.quatf, device=dev)
    hs = wp.zeros(G1, dtype=wp.float32, device=dev)
    left = wp.array(np.tile(nan3, (G1, 1)), dtype=wp.vec3f, device=dev)
    right = wp.array(np.tile(nan3, (G1, 1)), dtype=wp.vec3f, device=dev)
    count = wp.array(np.array([2], np.int32), device=dev)
    fallbacks = wp.zeros(E1, dtype=wp.int32, device=dev)
    wp.launch(warp_gate._finalize_frame_k, dim=E1 * G1,
              inputs=[pos, tan, forward, quat, hs, left, right, count, G1,
                      0.1, 1, fallbacks], device=dev)
    assert int(fallbacks.numpy()[0]) == 1  # exactly the vertical gate
    fwd0 = forward.numpy()[0]
    np.testing.assert_allclose(fwd0, [1.0, 0.0, 0.0], atol=1e-6)  # +x fallback


def test_frame_fallback_counter_zero_on_planar_generation():
    _, seq, cfg = _gen(gate_align="full_tangent")  # existing planar helper
    gen, _, _ = _gen(gate_align="full_tangent"), None, None
    g = _gen(gate_align="full_tangent")[0]
    assert int(g.frame_fallbacks.numpy().sum()) == 0
```

Adjust the `_finalize_frame_k` launch's input order to the kernel's CURRENT signature if it differs (parameter order after Task 1: position, tangent, forward, orientation, half_size, left, right, count, max_gates, gate_width, align_full, fallbacks). The second test should use the file's existing `_gen` helper idiomatically — read it and call it the way sibling tests do (the sketch above shows intent; write it cleanly).

- [ ] **Step 2: Hand-computed progress fixtures**

Create `tests/test_progress_handcomputed.py` — six literal fixtures against `ProgressTracker` on the same single-gate synthetic from Task 1 (gate at origin, forward +x, posts y=±1, `up_half`=1, but with a PLANAR tangent (1,0,0) so old/new semantics agree and these pin the base contract independently of `tests/_progress_oracle.py`):

```python
"""Hand-computed progress fixtures. These assert LITERAL expected events —
deliberately not routed through tests/_progress_oracle.py, which mirrors
the kernel's _plane_pass and therefore cannot catch a shared bug."""
```

Fixture table (each its own test; two `update()` calls — arm then move — asserting `(passed, wrong_way, wrong_checkpoint, dist_to_next)` literally):

| case | prev → pos | expected |
|---|---|---|
| forward pass | (-0.5,0,0) → (0.5,0,0) | passed=1, dist=0.5 (to same gate, n=1 wraps to itself) |
| backward crossing | (0.5,0,0) → (-0.5,0,0) | wrong_way=1, passed=0 |
| edge touch (lands ON plane) | (-0.5,0,0) → (0,0,0) | passed=1 (d0<0, d1>=0 counts) |
| outside u-opening | (-0.5,1.5,0) → (0.5,1.5,0) | passed=0, wrong_way=0 |
| outside v-half | (-0.5,0,1.5) → (0.5,0,1.5) | passed=0 (v=1.5 > up_half=1) |
| NaN pause | (-0.5,0,0) → (nan,nan,nan) | passed=0, dist NaN |

Compute each `dist_to_next` by hand (Euclidean from the post-move position to the gate center after any advance) and assert with `atol=1e-6`.

- [ ] **Step 3: Closing-chord grade coverage check**

Run: `grep -n "closing\|wrap\|% cnt\|j = 0" tests/test_gate_3d.py`
If a test already violates the grade ONLY on the closing chord (gate n-1 → 0) and asserts invalidity, note it in the report and stop. If coverage is indirect, add a direct kernel-level case: a 4-gate square course, all z equal except gate 3 (z=5), `z_valid_grade=0.5` — the steep chords are 2→3 AND the closing 3→0; then a variant where only the closing chord is steep (z = [0,0,0,5] with gates ordered so 3→0 is short): assert `valid == 0`, and `valid == 1` with `z_valid_grade=0.0`.

- [ ] **Step 4: Run + commit**

Run: `pytest tests/test_gate_3d.py tests/test_progress_handcomputed.py -q` then `pytest tests/ -q`
Expected: all PASS.

```bash
git add tests/test_gate_3d.py tests/test_progress_handcomputed.py
git commit --no-gpg-sign -m "test: frame-fallback counter, hand-computed progress fixtures, closing-chord grade"
```

---

### Task 4: Docs + polish closeout

**Files:**
- Modify: `docs/gates-3d.rst` (sampling note)
- Modify: `docs/superpowers/plans/2026-07-17-3d-gate-courses.md` (deviation-list addendum)
- Modify: `track_gen/_src/course.py:245-247` (max_boxes message)
- Modify: `track_gen/_src/collision_frames.py:72` (NaN guard)
- Test: existing suites only

**Interfaces:** none new.

- [ ] **Step 1: Sampling note**

In `docs/gates-3d.rst`, in the CourseLine/centerline section, add: "The centerline is sampled at ``samples_per_gate`` uniform *spline-parameter* steps per gate segment — not constant 3D arc-length (samples bunch slightly where gates are close together). ``arclen`` is nevertheless the true cumulative arc length of the sampled polyline, so localization and progress are unaffected."

- [ ] **Step 2: Plan deviation addendum**

In `docs/superpowers/plans/2026-07-17-3d-gate-courses.md`, "Spec deviations" list in the self-review section, append: "(6) `CourseLine` samples uniform Catmull-Rom parameters per gate segment rather than the spec's 'constant 3D arc-length' resample; `arclen` still stores true arc length, and no consumer assumes uniform spacing."

- [ ] **Step 3: max_boxes message**

`course.py:245-247` — replace the message with:

```python
                    "max_boxes > 1 is a collision-query stride but this course "
                    "has no box/disc collision checker (track: set collision; "
                    "gates: set post_radius > 0 — frame_collision uses sphere "
                    f"binding and ignores max_boxes), got max_boxes={self.max_boxes!r}")
```

- [ ] **Step 4: NaN guard**

`collision_frames.py:72` — replace `if p[0] != p[0]:` with `if _is_nan3(p) == 1:` and add `_is_nan3` to the module's `collision_geom` import line.

- [ ] **Step 5: Run everything + commit**

Run: `pytest tests/ -q` (full), `pytest tests/test_golden_migration.py -v`, `python -m sphinx -W -b html docs /tmp/sphinx_check`
Expected: all green, sphinx zero warnings.

```bash
git add docs/gates-3d.rst docs/superpowers/plans/2026-07-17-3d-gate-courses.md track_gen/_src/course.py track_gen/_src/collision_frames.py
git commit --no-gpg-sign -m "docs+polish: sampling note, deviation addendum, max_boxes message, FrameChecker NaN guard"
```

---

## Plan Self-Review Notes (resolved)

- **Spec coverage:** §1 pass-plane → Task 1 (forward field, from_gates, bit-exactness branch, regression pair, doc sentence); §2 race fix → Task 2 (lock extension, contract comment, fresh-construction test + rename, stress runs); §3 tests → Task 3; §3 docs/polish → Task 4. Acceptance criteria = Task 4 Step 5 plus per-task gates. All nine triage items covered; step()-time locking explicitly out of scope per the spec.
- **Type consistency:** `GateSequence.forward` ordering (between `tangent` and `orientation`) is used consistently in Task 1's construction, Task 3's kernel launch input order, and the docstring; `from_gates` alias name stays `tangent` on `CheckpointSet` (consumer contract unchanged).
- **Known softness, deliberate:** Task 2 Step 2 may not flake in 8 runs (it's ~25% per run pair); the step tells the implementer to escalate to 15 and record the rate rather than fabricate certainty. Task 3 Step 1's second test sketch is marked as intent — the implementer must adapt to the existing `_gen` helper rather than paste.
