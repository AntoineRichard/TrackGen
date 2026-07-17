# 3D Gate Courses (2.5D Lift) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batched 3D drone gate courses: the existing 2D pipeline generates the layout, a pluggable Z profiler assigns altitudes, gates get full 3D poses, and runtime (pass detection, localization, gate-frame collision) works in 3D — with the entire public data model moved to `vec3f` in one migration and 2D behavior preserved bit-for-bit.

**Architecture:** Approach A from the spec (`docs/superpowers/specs/2026-07-17-3d-gate-courses-design.md`): unified vec3f public types (`GateSequence`, `Track`), internals stay vec2f with lift kernels at the pipeline boundary. The existing `Course` `mode="gates"` path is extended (NOT a new mode — the spec's "gate_course" maps onto it). Gate tangents already come from knot central differences (`warp_gate._tangents_from_positions_k`), which IS the Catmull-Rom knot tangent, so 3D gate tangents are a dtype lift, and the spline only matters for the resampled localization centerline.

**Tech Stack:** Python, NVIDIA Warp (warp-lang >= 1.14), NumPy, pytest. No new dependencies.

## Global Constraints

- All geometry arrays are flat `[E * stride]` wp.arrays, NaN-padded past `count[e]`; per-env scalars are `[E]` int32/float32. Env index in kernels: `e = tid // stride`.
- Fixed shapes forever after construction; `generate()`/`step()`/`query()` are allocation-free and host-sync-free under capture. Follow the `_CAPTURING` / `_sync(device)` idiom of each module (`runtime._sync`); never add a host readback inside a capturable path.
- All CUDA work in `generate()` stays behind `runtime._CAPTURE_LOCK` (see `gate_generator.py:159` and memory note on the CUDA-700 race). Do not restructure that locking.
- In-kernel RNG idiom: `state = wp.rand_init(seeds[e] * <DISTINCT PRIME> + salt)` then `wp.randf(state)`. Every new kernel must use a prime multiplier not already used in the codebase (grep `rand_init` first). Existing primes include 3187, 6151, 2741, 9781, 104729.
- Warp quats are `wp.quatf(x, y, z, w)`; `wp.mat33` is row-major constructed.
- Frame conventions (match existing 2D code): forward = tangent; LEFT axis for a horizontal forward `t` is `(-t.y, t.x, 0)`, i.e. `up_world × forward`; gate `left = position + half_size * left_axis`, `right = position - half_size * left_axis`. Roll is always zero.
- The whole suite must pass on cpu; run cuda too where available: `pytest tests/ -x -q` and `pytest tests/ -x -q -k cuda`.
- Commit after every green task. Do not rewrite history (repo policy: squash deferred to release).
- Docstrings are load-bearing in this repo (Sphinx-rendered). Update every docstring whose described shapes/dtypes change. The pervasive reshape idiom changes from `view(E, N, 2)` to `view(E, N, 3)`.

---

### Task 1: Pre-migration golden capture

Freeze the current 2D outputs so the vec3f migration can prove bit-identical XY behavior.

**Files:**
- Create: `tests/tools/capture_goldens.py`
- Create: `tests/goldens/pre_vec3f.npz` (generated, committed)
- Create: `tests/test_golden_migration.py`

**Interfaces:**
- Produces: `tests/goldens/pre_vec3f.npz` with keys `track/<gen>/<field>` and `gates/<gen>/<field>` for all five generators (`bezier`, `hull`, `polar`, `voronoi`, `checkpoint`), cpu, `num_envs=8`, `seeds=1234`. Task 2's final step rewrites the comparison in `test_golden_migration.py` to XY-of-vec3f.

- [ ] **Step 1: Write the capture script**

```python
"""Capture pre-vec3f golden outputs (cpu, fixed seeds) for the migration regression.

Run once BEFORE the vec3f migration; the .npz is committed and never regenerated.
"""
import numpy as np

from track_gen._src.gate_generator import GateGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import GateGenConfig, TrackGenConfig

GOLDEN = "tests/goldens/pre_vec3f.npz"
GENERATORS = ("bezier", "hull", "polar", "voronoi", "checkpoint")
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen",
                "length", "valid", "count", "winding")
GATE_FIELDS = ("position", "tangent", "left", "right", "valid", "count")


def capture() -> dict:
    out = {}
    for gen in GENERATORS:
        cfg = TrackGenConfig(generator=gen, device="cpu", num_envs=8)
        rng = PerEnvSeededRNG(seeds=1234, num_envs=8, device="cpu")
        track = TrackGenerator(cfg, rng).generate()
        for f in TRACK_FIELDS:
            out[f"track/{gen}/{f}"] = getattr(track, f).numpy().copy()
        gcfg = GateGenConfig(generator=gen, device="cpu", num_envs=8,
                             gate_width=0.05)
        grng = PerEnvSeededRNG(seeds=1234, num_envs=8, device="cpu")
        seq = GateGenerator(gcfg, grng).generate()
        for f in GATE_FIELDS:
            out[f"gates/{gen}/{f}"] = getattr(seq, f).numpy().copy()
    return out


if __name__ == "__main__":
    np.savez_compressed(GOLDEN, **capture())
    print(f"wrote {GOLDEN}")
```

If a `TrackGenConfig`/`GateGenConfig` default raises for some generator (e.g. a generator-specific floor), copy the minimal working config from `tests/test_generators.py` / `tests/test_gate_generator.py` for that generator instead — the goldens must cover all five generators for both pipelines.

- [ ] **Step 2: Generate and inspect the goldens**

First create `tests/goldens/` and, if `tests/` uses package imports, an empty `tests/tools/__init__.py` so `from tests.tools.capture_goldens import ...` resolves under pytest.

Run: `python tests/tools/capture_goldens.py && python -c "import numpy as np; d=np.load('tests/goldens/pre_vec3f.npz'); print(len(d.files), 'arrays'); print(sorted(d.files)[:5])"`
Expected: `80 arrays` (5 generators × 10 track fields + 5 × 6 gate fields); no exception.

- [ ] **Step 3: Write the (currently trivial) regression test**

```python
"""Golden regression: the generation pipelines reproduce the frozen pre-vec3f batch."""
import numpy as np
import pytest

from tests.tools.capture_goldens import GOLDEN, capture


def test_pipelines_match_pre_vec3f_goldens():
    golden = np.load(GOLDEN)
    fresh = capture()
    for key in golden.files:
        np.testing.assert_allclose(
            fresh[key], golden[key], rtol=0.0, atol=0.0, equal_nan=True,
            err_msg=key)
```

- [ ] **Step 4: Run it**

Run: `pytest tests/test_golden_migration.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add tests/tools/capture_goldens.py tests/goldens/pre_vec3f.npz tests/test_golden_migration.py
git commit -m "test: freeze pre-vec3f golden outputs for the 3D migration"
```

---

### Task 2: The vec3f migration (atomic)

Move the public data model to vec3f with z ≡ 0 and identical XY behavior. This task is deliberately atomic: the arrays are a shared contract between producers and consumers, so no intermediate state can keep the suite green. Work through the steps in order; run the full suite and the goldens only at the end; ONE commit. The regression anchor (Task 1 goldens) is the review gate.

**Files:**
- Modify: `track_gen/_src/types.py` (GateSequence ~999-1058, Track ~1061-1158)
- Modify: `track_gen/_src/warp_gate.py`, `track_gen/_src/gate_generator.py`
- Modify: `track_gen/_src/warp_pipeline.py` (lift at pipeline end; internal kernels untouched)
- Modify: `track_gen/_src/collision_geom.py` (new 3D helpers; existing 2D helpers stay)
- Modify: `track_gen/_src/checkpoints.py`, `track_gen/_src/progress.py`, `track_gen/_src/localize.py`
- Modify: `track_gen/_src/collision.py`, `track_gen/_src/collision_sdf.py`, `track_gen/_src/collision_discs.py`
- Modify: `track_gen/_src/props.py` (input dtypes only; PropSet output stays 2D by spec)
- Modify: `track_gen/_src/course.py`
- Modify: `viz/plot_tracks.py`, `viz/render_utility_assets.py` (reshape 2→3, plot xy)
- Modify: every test under `tests/` that constructs vec2f buffers or reshapes `(..., 2)`
- Modify: `tests/test_golden_migration.py`, `tests/tools/capture_goldens.py`

**Interfaces:**
- Produces (public contract for all later tasks):
  - `GateSequence`: `position/tangent/left/right` `[E*G] wp.vec3f`; NEW `orientation` `[E*G] wp.quatf`; NEW `half_size` `[E*G] wp.float32`; `normal` REMOVED; `valid/count` unchanged. `clone()` covers the new fields.
  - `Track`: `outer/center/inner/tangent/normal` `[E*N] wp.vec3f` (z=0); scalars unchanged.
  - `CheckpointSet`: `position/left/right/tangent` vec3f; NEW `up_half` `[E*M] wp.float32` (vertical half-opening; `_BIG` = unbounded for track cross-sections; aliases `GateSequence.half_size` for gates).
  - `TrackFrame`: `s`, `n` (right offset), NEW `n_up` (up offset), `segment`.
  - `ProgressTracker`/`TrackLocalizer`/`Course`: bound `position` buffers are `[E] wp.vec3f`.
  - `Course.bind(position, orientation=None, half_extents=None, box_position=None)`: `orientation` `[E*max_boxes] wp.quatf` replaces `yaw`; `box_position` vec3f; `half_extents` stays vec2f (planar boxes).
  - New wp.funcs in `collision_geom.py`: `_safe_normalize3(v: wp.vec3f) -> wp.vec3f`, `_is_nan3(v: wp.vec3f) -> int`, `_quat_yaw(q: wp.quatf) -> float`, `_yaw_quat(yaw: float) -> wp.quatf`, `_frame_quat(fwd: wp.vec3f) -> wp.quatf`, `_plane_pass(...) -> int` (below).

- [ ] **Step 1: Shared 3D helpers in `collision_geom.py`**

Add (keeping every existing 2D helper untouched):

```python
@wp.func
def _safe_normalize3(v: wp.vec3f) -> wp.vec3f:
    l = wp.length(v)
    if l < 1.0e-12:
        return wp.vec3f(0.0, 0.0, 0.0)
    return v / l


@wp.func
def _is_nan3(v: wp.vec3f) -> int:
    if v[0] == v[0] and v[1] == v[1] and v[2] == v[2]:
        return 0
    return 1


@wp.func
def _yaw_quat(yaw: float) -> wp.quatf:
    h = 0.5 * yaw
    return wp.quatf(0.0, 0.0, wp.sin(h), wp.cos(h))


@wp.func
def _quat_yaw(q: wp.quatf) -> float:
    return wp.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                    1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


@wp.func
def _frame_quat(fwd: wp.vec3f) -> wp.quatf:
    """Roll-free frame quat: x=forward, y=left(=up_world x fwd), z=up.

    Caller guarantees fwd is unit and not near-vertical (fallback is the
    caller's job — see _finalize_frame_k).
    """
    up_w = wp.vec3f(0.0, 0.0, 1.0)
    left = wp.normalize(wp.cross(up_w, fwd))
    up = wp.cross(fwd, left)
    return wp.quat_from_matrix(wp.mat33(
        fwd[0], left[0], up[0],
        fwd[1], left[1], up[1],
        fwd[2], left[2], up[2]))


@wp.func
def _plane_pass(prev: wp.vec3f, pos: wp.vec3f, fwd: wp.vec3f,
                l: wp.vec3f, r: wp.vec3f, v_half: float) -> int:
    """Swept segment vs gate plane, bounded opening. +1 forward pass,
    -1 backward crossing inside the opening, 0 otherwise.

    u axis spans left->right (u_half from the endpoints); v axis =
    u_axis x fwd (up for roll-free frames); v bounded by v_half
    (checkpoints from track cross-sections pass _BIG = unbounded)."""
    mid = 0.5 * (l + r)
    d0 = wp.dot(prev - mid, fwd)
    d1 = wp.dot(pos - mid, fwd)
    crossing = int(0)
    if d0 < 0.0 and d1 >= 0.0:
        crossing = 1
    if d0 > 0.0 and d1 <= 0.0:
        crossing = -1
    if crossing == 0:
        return 0
    t = d0 / (d0 - d1)
    pi = prev + (pos - prev) * t
    u_axis = r - l
    u_len = wp.length(u_axis)
    if u_len < 1.0e-12:
        return 0
    u_axis = u_axis / u_len
    u = wp.dot(pi - mid, u_axis)
    v = wp.dot(pi - mid, _safe_normalize3(wp.cross(u_axis, fwd)))
    if wp.abs(u) <= 0.5 * u_len and wp.abs(v) <= v_half:
        return crossing
    return 0
```

- [ ] **Step 2: `types.py` — GateSequence and Track**

`GateSequence`: change field docs to vec3f; delete `normal` (field, docstring, `clone()` line); add `orientation: wp.array` and `half_size: wp.array` (with docstrings: quatf gate pose, x=forward/y=left/z=up, roll-free; float32 square-opening half-extent = `0.5 * gate_width`) and their `clone()` lines. `Track`: docstrings to vec3f, note z=0 from the 2D pipeline; correct the `normal` doc (still the planar left-normal, now `(-t.y, t.x, 0)`). Update the module-level reshape examples to `view(E, ..., 3)`.

- [ ] **Step 3: `warp_gate.py` — gate pipeline lift**

The 2D anchor/order/relax kernels keep operating on vec2f scratch. Changes:

1. `alloc_gate_sequence` (`warp_gate.py:452`): public arrays become `wp.vec3f` (fill with `wp.vec3f(nan,nan,nan)` via a new `_fill_vec3_k`), plus `orientation` (`wp.quatf`, NaN-filled via `_fill_quat_k`) and `half_size` (`wp.float32`, NaN-filled). Delete the `normal` allocation.
2. `_gate_warp_alloc` (`warp_gate.py:647`): add vec2f scratch `pos2` `[E*G]` and float32 scratch `z` `[E*G]` (zero-filled; the Phase B z-profile stage writes it) to the scratch tuple. The generators and `order_points`/`relax_gate_spheres` now write/read `pos2` instead of `gates.position` (mechanical: `finish_ordered_gates` at `warp_gate.py:522` orders into `pos2`).
3. New lift kernel:

```python
@wp.kernel
def _lift_positions_k(
    pos2: wp.array(dtype=wp.vec2f),
    z: wp.array(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3f),
):
    t = wp.tid()
    p = pos2[t]
    position[t] = wp.vec3f(p[0], p[1], z[t])
```

4. `_tangents_from_positions_k` (`warp_gate.py:294`): dtype swap vec2f→vec3f (`position`, `tangent` params; NaN literal becomes `wp.vec3f(wp.nan, wp.nan, wp.nan)`, zero becomes 3D). The central-difference math is dimension-agnostic — no logic change. It now runs on the lifted `gates.position`.
5. `_finalize_frame_k` (`warp_gate.py:333`) rewrite:

```python
@wp.kernel
def _finalize_frame_k(
    position: wp.array(dtype=wp.vec3f),
    tangent: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_size: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.vec3f),
    right: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    gate_width: float,
    align_full: int,
    fallbacks: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if i >= cnt or i >= max_gates:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        position[t] = nan3
        tangent[t] = nan3
        orientation[t] = wp.quatf(wp.nan, wp.nan, wp.nan, wp.nan)
        half_size[t] = wp.nan
        left[t] = nan3
        right[t] = nan3
        return
    p = position[t]
    tan = _safe_normalize3(tangent[t])
    fwd = tan
    if align_full == 0:
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
    horiz2 = fwd[0] * fwd[0] + fwd[1] * fwd[1]
    if horiz2 < 1.0e-10:
        # Near-vertical (full_tangent on a steep segment, or degenerate
        # tangent): fall back to the horizontal tangent direction, then +x.
        fwd = wp.vec3f(tan[0], tan[1], 0.0)
        if fwd[0] * fwd[0] + fwd[1] * fwd[1] < 1.0e-10:
            fwd = wp.vec3f(1.0, 0.0, 0.0)
        wp.atomic_add(fallbacks, e, 1)
    fwd = _safe_normalize3(fwd)
    q = _frame_quat(fwd)
    hs = 0.5 * gate_width
    la = wp.quat_rotate(q, wp.vec3f(0.0, 1.0, 0.0))
    tangent[t] = tan
    orientation[t] = q
    half_size[t] = hs
    left[t] = p + hs * la
    right[t] = p - hs * la
```

In this task `align_full` is hard-wired to `0` at the call site in `finalize_gate_sequence` (`warp_gate.py:604`) and `fallbacks` is a new `[E]` int32 scratch zeroed each run (exposed later as `GateGenerator.frame_fallbacks`). For planar tangents, `align_full=0` reproduces the old left/right exactly: `quat_rotate(yaw-frame, +y) == (-t.y, t.x, 0)`.

6. `_finalize_validity_k` (`warp_gate.py:371`): params to vec3f (drop the `normal` param; add `orientation`/`half_size` finite checks). CRITICAL for golden parity: the pairwise min-distance check stays on XY (`d[0]*d[0] + d[1]*d[1]`) and the crossing check calls the existing 2D `_proper_segment_intersection` on `wp.vec2f(li[0], li[1])` etc. — identical decisions to today.
7. Import the new helpers from `collision_geom` (`_safe_normalize3`, `_frame_quat`).

- [ ] **Step 4: `warp_pipeline.py` — track lift**

The pipeline currently writes `Track.outer/center/inner/tangent/normal` (vec2f) directly. Rename those five buffers in the pipeline's allocation to internal vec2f scratch (`_out2`, `_ctr2`, `_inn2`, `_tan2`, `_nrm2` — same shapes), keep EVERY kernel launch pointing at the scratch (a mechanical rename), and append one lift launch at the end of `_run_pipeline` (`warp_pipeline.py:868-947`), after `_validity_k`:

```python
@wp.kernel
def _lift_track_k(
    out2: wp.array(dtype=wp.vec2f), ctr2: wp.array(dtype=wp.vec2f),
    inn2: wp.array(dtype=wp.vec2f), tan2: wp.array(dtype=wp.vec2f),
    nrm2: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
    inner: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
    normal: wp.array(dtype=wp.vec3f),
):
    t = wp.tid()
    outer[t] = wp.vec3f(out2[t][0], out2[t][1], 0.0)
    center[t] = wp.vec3f(ctr2[t][0], ctr2[t][1], 0.0)
    inner[t] = wp.vec3f(inn2[t][0], inn2[t][1], 0.0)
    tangent[t] = wp.vec3f(tan2[t][0], tan2[t][1], 0.0)
    normal[t] = wp.vec3f(nrm2[t][0], nrm2[t][1], 0.0)
```

NaN xy lifts to NaN-xy with z=0.0; the NaN-padding convention for vec3f is "any NaN component" — `_is_nan3` above and all consumers' finite checks treat it so. The `Track` allocation site switches those five fields to vec3f. Nothing else in this 1700-line file changes.

- [ ] **Step 5: `checkpoints.py`**

- `CheckpointSet`: fields to vec3f docs; add `up_half: wp.array` field + docstring + `clone()` line. `from_gates` (`checkpoints.py:67`): add `up_half=seq.half_size` (`normal` was never used here).
- `_seg_at_arc` unchanged. `_place_checkpoints_k` (`checkpoints.py:117`): dtype swap on the six polyline/out params (vec2f→vec3f), NaN literal to `wp.vec3f`, `_safe_normalize2` → `_safe_normalize3`; add `out_up_half: wp.array(dtype=wp.float32)` written to `_BIG` for real slots and `wp.nan` for padding (import `_BIG` from `runtime`).
- `CheckpointSampler.__init__`: allocate the set's arrays as vec3f, plus `up_half=wp.zeros(n, dtype=wp.float32, device=dev)`.
- `_derive_max_checkpoints` (`checkpoints.py:247`): `reshape(E, n_max, 2)` → `reshape(E, n_max, 3)` (perimeter math is dimension-agnostic).

- [ ] **Step 6: `progress.py` — plane-crossing kernel**

`_progress_update_k` (`progress.py:93`) becomes 3D: `cp_position/cp_left/cp_right/cp_tangent/position/prev_pos` to vec3f, add `cp_up_half: wp.array(dtype=wp.float32)` after `cp_tangent`. Replace the crossing block (`progress.py:143-156`) with:

```python
    prev = prev_pos[e]
    if _is_nan3(prev) == 0:
        c = _plane_pass(prev, pos, cp_tangent[base + g],
                        cp_left[base + g], cp_right[base + g],
                        cp_up_half[base + g])
        if c == 1:
            passed = int(1)
            cp_passed = g
        elif c == -1:
            wway = int(1)
        for i in range(n):
            if i != g and wcp == -1:
                if _plane_pass(prev, pos, cp_tangent[base + i],
                               cp_left[base + i], cp_right[base + i],
                               cp_up_half[base + i]) != 0:
                    wcp = i
```

(The forward-direction dot test is now inside `_plane_pass` via the sign of the crossing.) `_progress_reset_k`: NaN sentinel to `wp.vec3f(wp.nan, wp.nan, wp.nan)`. `ProgressTracker.__init__`: dtype validation loops expect vec3f (and validate `checkpoints.up_half` as `[stride]` float32); `_prev_pos` allocation `np.full((E, 3), ...)`; `_validate_position` expects `wp.vec3f`. Semantics note for the docstring: pass detection is now plane-crossing within the opening (planar behavior unchanged for planar motion — the u-extent equals the old segment test; `up_half=_BIG` keeps track cross-sections vertically unbounded).

- [ ] **Step 7: `localize.py`**

`TrackFrame`: add `n_up: wp.array` field + docstring ("signed vertical offset in the roll-free frame at the foot point; equals `position.z` for planar tracks") + `clone()` line. `_localize_k` (`localize.py:92`): `center/position` to vec3f, `_is_nan2`→`_is_nan3`, all point math is dimension-agnostic; replace the offset block (`localize.py:167-177`) with:

```python
    t = _safe_normalize3(ab)
    right_hat = wp.cross(t, wp.vec3f(0.0, 0.0, 1.0))   # (t.y, -t.x, 0)
    rl = wp.length(right_hat)
    if rl < 1.0e-6:
        right_hat = wp.vec3f(1.0, 0.0, 0.0)            # vertical segment guard
    else:
        right_hat = right_hat / rl
    up_hat = wp.cross(right_hat, t)

    s = seg_start + best_u * (seg_end - seg_start)
    if s >= length[e]:
        s = s - length[e]
    out_s[e] = s
    out_n[e] = wp.dot(p - q, right_hat)
    out_n_up[e] = wp.dot(p - q, up_hat)
```

Add `out_n_up` param + NaN write in the degenerate branch; allocate `n_up` in `TrackLocalizer.__init__`; `_validate_position` expects vec3f. `curvature()`/`speed_profile()` kernels (`localize.py:340,460`): dtype swap to vec3f — Menger curvature via cross-product norm: replace the scalar 2D cross with `wp.length(wp.cross(b - a, c - a))` (identical value for z=0).

- [ ] **Step 8: collision trio (planar compute preserved)**

- `collision.py`: kernels reading `Track.outer/inner` and bound `box_position` switch those params to vec3f and immediately project: `p2 = wp.vec2f(p[0], p[1])` at the top of each kernel body — every downstream expression unchanged. `bind_inputs(box_position, orientation, half_extents)`: `orientation` `[E*max_boxes]` quatf; kernels compute `yaw = _quat_yaw(orientation[b])` where they read the old float. Validation messages updated.
- `collision_sdf.py`: bake kernel reads vec3f boundaries → xy projection; the SDF grid, lookups and query stay 2D byte-identical.
- `collision_discs.py`: the discs (posts) input array becomes vec3f; query kernel projects to xy.
- Half-extents stay `wp.vec2f` everywhere (planar boxes; spec keeps 2D OOB semantics).

- [ ] **Step 9: `props.py`**

`_scan_boundary_k` and the placement kernels take vec3f polylines and project to xy for pose output (`PropSet` stays the documented 2D pose format — consumer lifts to 3D per the props spec). Arc lengths now computed on xy (identical for z=0): use `wp.length(wp.vec2f(d[0], d[1]))`.

- [ ] **Step 10: `course.py`**

- `_interleave_posts_k` (`course.py:48`): vec2f→vec3f params; `_posts` allocation to vec3f.
- `bind()` (`course.py:283`): signature `bind(self, position, orientation=None, half_extents=None, box_position=None)`; `_validate_bind_args`: `position` `[E]` vec3f, `orientation` `[E*max_boxes]` quatf, `box_position` vec3f; error strings mention `orientation` instead of `yaw`; `_apply_bind` passes `a["orientation"]` through to `collision.bind_inputs`.
- Module/class docstrings: reshape idiom and the bind contract.

- [ ] **Step 11: viz + tests sweep**

Find every remaining 2D assumption:

Run: `grep -rn "view(E\|reshape(E\|, 2)\|vec2f" viz/ tests/ | grep -v goldens`

- `viz/plot_tracks.py:95-97`, `viz/render_utility_assets.py:60-62`: reshape `(E, N, 3)`, plot `[..., 0], [..., 1]`.
- Tests: every constructed position buffer becomes vec3f with z=0 (e.g. `wp.array(np.array([[x, y, 0.0]], np.float32), dtype=wp.vec3f, ...)`); every reshape gains the third component; `test_types.py` gains assertions that `GateSequence` has `orientation`/`half_size` and no `normal`; progress/checkpoint tests updated for `up_half`. Do NOT weaken any assertion — only re-dimension inputs and expected values (expected z is always 0.0, expected `n_up` is `z` of the query point).

- [ ] **Step 12: goldens comparison update**

In `capture_goldens.py`, add module constants `VEC3_TRACK = ("outer", "center", "inner", "tangent", "normal")`, `VEC3_GATES = ("position", "tangent", "left", "right")`, and in `test_golden_migration.py` compare per key: for vec3 fields `fresh[key][:, :2]` vs golden AND `fresh[key][:, 2]` all-zero-or-NaN (NaN exactly where golden xy is NaN); everything else exact as before. (`GATE_FIELDS` never included `normal`, so nothing to drop there; the golden npz stays byte-identical to Task 1's.)

- [ ] **Step 13: Full suite + goldens**

Run: `pytest tests/ -q` then `pytest tests/test_golden_migration.py -v`
Expected: everything PASSES. The golden test proves bit-identical XY across both pipelines and all five generators.

- [ ] **Step 14: Commit**

```bash
git add -A track_gen/ viz/ tests/
git commit -m "feat!: unified vec3f data model (z=0), gate orientation quats, plane-crossing progress

BREAKING: GateSequence/Track geometry is vec3f; GateSequence.normal removed,
orientation (quatf) + half_size added; CheckpointSet gains up_half; TrackFrame
gains n_up; bound position buffers are [E] vec3f; Course.bind takes orientation
quaternions instead of yaw. XY behavior is golden-verified bit-identical."
```

---

### Task 3: Z profiler module

Pluggable altitude generation, not yet wired into the pipeline.

**Files:**
- Create: `track_gen/_src/warp_zprofile.py`
- Modify: `track_gen/_src/types.py` (GateGenConfig fields + validation, ~line 876)
- Test: `tests/test_zprofile.py`

**Interfaces:**
- Consumes: ordered 2D gate anchors `pos2` `[E*G] vec2f` scratch, `count [E] int32`, seed buffer `[E] int32` (same buffer the gate pipeline already threads).
- Produces: `apply_z_profile(config, seeds_wp, pos2, count, cum_scratch, z)` — fills `z` `[E*G] float32` (0 past `count[e]`); `alloc_z_scratch(config) -> (cum, z)` both `[E*G] float32`. New `GateGenConfig` fields: `z_profile: str = "flat"`, `z_base: float = 0.0`, `z_min: float = 0.0`, `z_max: float = 0.0`, `z_max_step: float = 0.0` (a GRADE: max |dz| per unit plan-view arc length), `z_noise_amplitude: float = 0.0`, `z_noise_harmonics: int = 3`, `z_valid_grade: float = 0.0` (0 disables the validity check; used in Task 4).

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
import warp as wp

from track_gen._src.types import GateGenConfig
from track_gen._src import warp_zprofile

E, G = 8, 16


def _ring(seed=0):
    """Ordered ring anchors + counts, plausible gate layout."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(6, G + 1, size=E).astype(np.int32)
    pos = np.full((E * G, 2), np.nan, np.float32)
    for e in range(E):
        n = counts[e]
        ang = np.sort(rng.uniform(0, 2 * np.pi, n)).astype(np.float32)
        r = 1.0 + 0.2 * rng.standard_normal(n).astype(np.float32)
        pos[e * G:e * G + n, 0] = r * np.cos(ang)
        pos[e * G:e * G + n, 1] = r * np.sin(ang)
    return pos, counts


def _run(profile, **kw):
    cfg = GateGenConfig(device="cpu", num_envs=E, max_gates=G, gate_width=0.05,
                        z_profile=profile, **kw)
    pos_np, counts = _ring()
    pos2 = wp.array(pos_np, dtype=wp.vec2f, device="cpu")
    count = wp.array(counts, dtype=wp.int32, device="cpu")
    seeds = wp.array(np.arange(E, dtype=np.int32) + 7, dtype=wp.int32,
                     device="cpu")
    cum, z = warp_zprofile.alloc_z_scratch(cfg)
    warp_zprofile.apply_z_profile(cfg, seeds, pos2, count, cum, z)
    return z.numpy().reshape(E, G), counts, pos_np.reshape(E, G, 2)


def test_flat_is_base():
    z, counts, _ = _run("flat", z_base=1.5)
    for e in range(E):
        assert np.allclose(z[e, :counts[e]], 1.5)


def test_uniform_bounds_and_determinism():
    z1, counts, _ = _run("uniform", z_min=1.0, z_max=3.0)
    z2, _, _ = _run("uniform", z_min=1.0, z_max=3.0)
    np.testing.assert_array_equal(z1, z2)
    for e in range(E):
        zz = z1[e, :counts[e]]
        assert (zz >= 1.0).all() and (zz <= 3.0).all()
        assert zz.std() > 0.0


def test_walk_bounds_closure_and_grade():
    z, counts, pos = _run("random_walk", z_base=2.0, z_min=0.5, z_max=3.5,
                          z_max_step=0.3)
    for e in range(E):
        n = counts[e]
        zz = z[e, :n]
        assert (zz >= 0.5 - 1e-5).all() and (zz <= 3.5 + 1e-5).all()
        # closure: bridge pulls the walk back near its start
        assert abs(zz[0] - zz[-1]) < 1.0
        # grade cap (bridge adds at most drift/perimeter; allow 2x slack)
        p = pos[e, :n]
        ds = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)[: n - 1]
        grade = np.abs(np.diff(zz)) / np.maximum(ds, 1e-9)
        assert (grade <= 2.0 * 0.3 + 1e-4).all()


def test_noise_bounds_and_periodicity_shape():
    z, counts, _ = _run("noise", z_base=2.0, z_noise_amplitude=0.5,
                        z_noise_harmonics=3, z_min=1.0, z_max=3.0)
    for e in range(E):
        zz = z[e, :counts[e]]
        assert (zz >= 1.0 - 1e-5).all() and (zz <= 3.0 + 1e-5).all()
        assert zz.std() > 0.0


def test_config_validation():
    with pytest.raises(ValueError):
        GateGenConfig(device="cpu", num_envs=1, z_profile="bogus")
    with pytest.raises(ValueError):
        GateGenConfig(device="cpu", num_envs=1, z_profile="uniform",
                      z_min=2.0, z_max=1.0)
    with pytest.raises(ValueError):
        GateGenConfig(device="cpu", num_envs=1, z_profile="random_walk",
                      z_max_step=-0.1)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zprofile.py -x -q`
Expected: FAIL — `module 'track_gen._src' has no attribute 'warp_zprofile'` / unknown config field.

- [ ] **Step 3: Config fields + validation**

Add the fields listed in Interfaces to `GateGenConfig` with docstrings mirroring the existing style, and to `__post_init__`-style validation (GateGenConfig validates in the same place its other fields do — follow the existing pattern): `z_profile in ("flat", "uniform", "random_walk", "noise")`; `z_min <= z_max`; `z_max_step >= 0`; `z_noise_amplitude >= 0`; `z_noise_harmonics >= 1`; `z_valid_grade >= 0`.

- [ ] **Step 4: Implement `warp_zprofile.py`**

```python
"""Pluggable per-gate altitude (Z) profiles for gate courses.

The Z profiler is the vertical counterpart of the XY generator: it runs on
the ordered 2D anchors AFTER ordering/relaxation and BEFORE the 3D lift.
All kernels are fixed-shape, allocation-free and capture-safe. Profiles are
closed-loop consistent: flat trivially, random_walk via a Brownian-bridge
drift subtraction, noise via periodic harmonics in normalized arc length.
Padding slots get z = 0 (the lift NaNs them via pos2 anyway).
"""
import warp as wp

from .types import GateGenConfig

_P_UNIFORM = 15679
_P_WALK = 15683
_P_NOISE = 15731
_TWO_PI = 6.2831853071795864


@wp.kernel
def _cum_chords_k(
    pos2: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    cum: wp.array(dtype=wp.float32),
):
    # cum[base+i] = plan-view arc length from gate 0 to gate i; the closing
    # chord (n-1 -> 0) is NOT included in cum but callers can recover the
    # perimeter as cum[n-1] + |p0 - p_{n-1}|.
    e = wp.tid()
    base = e * max_gates
    n = count[e]
    if n > max_gates:
        n = max_gates
    acc = float(0.0)
    for i in range(max_gates):
        if i < n:
            if i > 0:
                acc = acc + wp.length(pos2[base + i] - pos2[base + i - 1])
            cum[base + i] = acc
        else:
            cum[base + i] = 0.0


@wp.kernel
def _z_flat_k(count: wp.array(dtype=wp.int32), max_gates: int, z_base: float,
              z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    if i < count[e]:
        z[t] = z_base
    else:
        z[t] = 0.0


@wp.kernel
def _z_uniform_k(seeds: wp.array(dtype=wp.int32),
                 count: wp.array(dtype=wp.int32), max_gates: int,
                 z_min: float, z_max: float,
                 z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    if i >= count[e]:
        z[t] = 0.0
        return
    state = wp.rand_init(seeds[e] * _P_UNIFORM + i)
    z[t] = z_min + wp.randf(state) * (z_max - z_min)


@wp.kernel
def _z_walk_k(seeds: wp.array(dtype=wp.int32),
              count: wp.array(dtype=wp.int32), max_gates: int,
              cum: wp.array(dtype=wp.float32),
              pos2: wp.array(dtype=wp.vec2f),
              z_base: float, z_min: float, z_max: float, max_grade: float,
              z: wp.array(dtype=wp.float32)):
    e = wp.tid()
    base = e * max_gates
    n = count[e]
    if n > max_gates:
        n = max_gates
    if n < 1:
        return
    state = wp.rand_init(seeds[e] * _P_WALK)
    acc = float(0.0)
    for i in range(n):
        if i > 0:
            ds = cum[base + i] - cum[base + i - 1]
            acc = acc + (2.0 * wp.randf(state) - 1.0) * max_grade * ds
        z[base + i] = acc
    # Brownian bridge: subtract the linear drift so the closing step
    # (n-1 -> 0) carries no accumulated offset, then rebase and clamp.
    perim = cum[base + n - 1] + wp.length(pos2[base] - pos2[base + n - 1])
    drift = acc
    for i in range(n):
        frac = 0.0
        if perim > 1.0e-9:
            frac = cum[base + i] / perim
        z[base + i] = wp.clamp(z_base + z[base + i] - drift * frac,
                               z_min, z_max)
    for i in range(n, max_gates):
        z[base + i] = 0.0


@wp.kernel
def _z_noise_k(seeds: wp.array(dtype=wp.int32),
               count: wp.array(dtype=wp.int32), max_gates: int,
               cum: wp.array(dtype=wp.float32),
               pos2: wp.array(dtype=wp.vec2f),
               z_base: float, z_min: float, z_max: float,
               amplitude: float, harmonics: int,
               z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    n = count[e]
    if n > max_gates:
        n = max_gates
    if i >= n:
        z[t] = 0.0
        return
    base = e * max_gates
    perim = cum[base + n - 1] + wp.length(pos2[base] - pos2[base + n - 1])
    frac = 0.0
    if perim > 1.0e-9:
        frac = cum[base + i] / perim
    acc = float(0.0)
    norm = float(0.0)
    for k in range(harmonics):
        state = wp.rand_init(seeds[e] * _P_NOISE + k)
        a = wp.randf(state)                       # harmonic amplitude in [0,1)
        phase = wp.randf(state) * _TWO_PI
        w = 1.0 / float(k + 1)                    # 1/f-ish spectrum
        acc = acc + a * w * wp.sin(_TWO_PI * float(k + 1) * frac + phase)
        norm = norm + w
    zz = z_base + amplitude * acc / wp.max(norm, 1.0e-9)
    z[t] = wp.clamp(zz, z_min, z_max)


def alloc_z_scratch(config: GateGenConfig):
    """(cum, z) float32 scratch, both [E * max_gates], zero-initialized."""
    E, G = int(config.num_envs), int(config.max_gates)
    dev = str(config.device)
    return (wp.zeros(E * G, dtype=wp.float32, device=dev),
            wp.zeros(E * G, dtype=wp.float32, device=dev))


def apply_z_profile(config: GateGenConfig, seeds_wp: wp.array,
                    pos2: wp.array, count: wp.array,
                    cum: wp.array, z: wp.array) -> None:
    """Fill z [E*G] from the configured profile. Capture-safe, no sync."""
    E, G = int(config.num_envs), int(config.max_gates)
    dev = str(config.device)
    profile = config.z_profile
    if profile != "flat":
        wp.launch(_cum_chords_k, dim=E, inputs=[pos2, count, G, cum],
                  device=dev)
    if profile == "flat":
        wp.launch(_z_flat_k, dim=E * G,
                  inputs=[count, G, float(config.z_base), z], device=dev)
    elif profile == "uniform":
        wp.launch(_z_uniform_k, dim=E * G,
                  inputs=[seeds_wp, count, G, float(config.z_min),
                          float(config.z_max), z], device=dev)
    elif profile == "random_walk":
        wp.launch(_z_walk_k, dim=E,
                  inputs=[seeds_wp, count, G, cum, pos2,
                          float(config.z_base), float(config.z_min),
                          float(config.z_max), float(config.z_max_step), z],
                  device=dev)
    else:  # "noise" — config validation guarantees membership
        wp.launch(_z_noise_k, dim=E * G,
                  inputs=[seeds_wp, count, G, cum, pos2,
                          float(config.z_base), float(config.z_min),
                          float(config.z_max),
                          float(config.z_noise_amplitude),
                          int(config.z_noise_harmonics), z], device=dev)
```

Before finalizing, run `grep -rn "15679\|15683\|15731" track_gen/` — if any prime is taken, pick fresh ones.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_zprofile.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/warp_zprofile.py track_gen/_src/types.py tests/test_zprofile.py
git commit -m "feat: pluggable Z profilers (flat/uniform/random_walk/noise) for gate courses"
```

---

### Task 4: Wire Z + alignment modes + grade validity into the gate pipeline

**Files:**
- Modify: `track_gen/_src/warp_gate.py` (`_gate_warp_alloc`, `_run_gate_pipeline`, `finalize_gate_sequence`, `_finalize_validity_k`)
- Modify: `track_gen/_src/gate_generator.py` (expose `frame_fallbacks`)
- Modify: `track_gen/_src/types.py` (`gate_align: str = "yaw_only"` + validation: one of `"yaw_only"`, `"full_tangent"`)
- Test: `tests/test_gate_3d.py`

**Interfaces:**
- Consumes: `warp_zprofile.alloc_z_scratch` / `apply_z_profile` (Task 3), `_finalize_frame_k(..., align_full, fallbacks)` (Task 2).
- Produces: `GateGenerator.generate()` now returns gates with `position.z` from the configured profile, `orientation` per `gate_align`, and validity extended by the grade check; `GateGenerator.frame_fallbacks` `[E] int32`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
import warp as wp

from track_gen._src.gate_generator import GateGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.types import GateGenConfig

E = 8


def _gen(**kw):
    cfg = GateGenConfig(device="cpu", num_envs=E, gate_width=0.05, **kw)
    rng = PerEnvSeededRNG(seeds=42, num_envs=E, device="cpu")
    g = GateGenerator(cfg, rng)
    return g, g.generate(), cfg


def _valid_gates(seq, cfg, e):
    n = int(seq.count.numpy()[e])
    G = int(cfg.max_gates)
    sl = slice(e * G, e * G + n)
    return (seq.position.numpy()[sl], seq.tangent.numpy()[sl],
            seq.orientation.numpy()[sl], seq.left.numpy()[sl],
            seq.right.numpy()[sl])


def test_z_profile_reaches_positions():
    _, seq, cfg = _gen(z_profile="uniform", z_min=1.0, z_max=2.0)
    valid = seq.valid.numpy()
    assert valid.any()
    for e in np.flatnonzero(valid):
        p, _, _, _, _ = _valid_gates(seq, cfg, e)
        assert (p[:, 2] >= 1.0 - 1e-5).all() and (p[:, 2] <= 2.0 + 1e-5).all()
        assert p[:, 2].std() > 0.0


def test_yaw_only_gates_stay_upright():
    _, seq, cfg = _gen(z_profile="uniform", z_min=0.5, z_max=3.0,
                       gate_align="yaw_only")
    for e in np.flatnonzero(seq.valid.numpy()):
        p, _, _, l, r = _valid_gates(seq, cfg, e)
        # posts horizontal: left/right at the same altitude as the center
        np.testing.assert_allclose(l[:, 2], p[:, 2], atol=1e-5)
        np.testing.assert_allclose(r[:, 2], p[:, 2], atol=1e-5)


def test_full_tangent_follows_slope():
    _, seq, cfg = _gen(z_profile="random_walk", z_base=1.5, z_min=0.5,
                       z_max=3.0, z_max_step=0.5, gate_align="full_tangent")
    saw_tilt = False
    for e in np.flatnonzero(seq.valid.numpy()):
        p, t, q, _, _ = _valid_gates(seq, cfg, e)
        # orientation x-axis == unit tangent (full alignment)
        for i in range(len(p)):
            x, y, z, w = q[i]
            fwd = np.array([
                1 - 2 * (y * y + z * z),
                2 * (x * y + w * z),
                2 * (x * z - w * y)])
            np.testing.assert_allclose(fwd, t[i], atol=1e-4)
            if abs(t[i][2]) > 1e-3:
                saw_tilt = True
    assert saw_tilt


def test_grade_validity_flags_steep_uniform():
    _, seq_off, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0)
    _, seq_on, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0,
                        z_valid_grade=0.5)
    # absurd z range: with the check on, strictly fewer (realistically zero)
    # envs stay valid
    assert seq_on.valid.numpy().sum() < max(1, seq_off.valid.numpy().sum())


def test_flat_default_matches_2d_goldens():
    # z_profile default is "flat" with z_base 0: Task 1 goldens still hold
    # (also covered by test_golden_migration, asserted here for locality).
    _, seq, cfg = _gen()
    for e in np.flatnonzero(seq.valid.numpy()):
        p, _, _, _, _ = _valid_gates(seq, cfg, e)
        np.testing.assert_allclose(p[:, 2], 0.0, atol=0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_gate_3d.py -x -q`
Expected: FAIL (`gate_align` unknown field, z all zero).

- [ ] **Step 3: Wire the stage**

In `_gate_warp_alloc`: also allocate `(z_cum, z)` via `warp_zprofile.alloc_z_scratch` and a `frame_fallbacks = wp.zeros(E, dtype=wp.int32, device=dev)`; return them in the scratch tuple. In `_run_gate_pipeline` (`warp_gate.py:654`), after ordering/relaxation produce final `pos2` + `count` and BEFORE the lift: zero `fallbacks` (`fallbacks.zero_()` is not capture-legal — use a `_fill_i32_k` launch), call `warp_zprofile.apply_z_profile(config, seed_buf_wp, pos2, count, z_cum, z)`, then `_lift_positions_k(pos2, z, gates.position)`, then the vec3f tangent kernel, then `_finalize_frame_k` with `align_full=int(config.gate_align == "full_tangent")` and `fallbacks`. In `_finalize_validity_k`, add params `z_valid_grade: float` and the check (after the pairwise loop; uses positions' xy chords):

```python
    if z_valid_grade > 0.0:
        for i in range(max_gates):
            if i < cnt and cnt >= 2:
                j = i + 1
                if j == cnt:
                    j = 0
                pi = position[base + i]
                pj = position[base + j]
                dxy = wp.sqrt((pj[0] - pi[0]) * (pj[0] - pi[0]) +
                              (pj[1] - pi[1]) * (pj[1] - pi[1]))
                if wp.abs(pj[2] - pi[2]) > z_valid_grade * wp.max(dxy, 1.0e-9):
                    ok = int(0)
```

`GateGenerator` exposes `self.frame_fallbacks` (the scratch array; document as debug-only). CRITICAL: all launches sit inside the captured pipeline — no allocation, no sync, no host branch on device data.

- [ ] **Step 4: Run tests + full suite**

Run: `pytest tests/test_gate_3d.py tests/test_gate_generator.py tests/test_warp_gate.py tests/test_golden_migration.py -q` then `pytest tests/ -q`
Expected: all PASS (goldens still exact: default profile is flat/0).

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/ tests/test_gate_3d.py
git commit -m "feat: Z profile + gate_align wired into the gate pipeline, grade validity"
```

---

### Task 5: Periodic spline course line (3D centerline for gates mode)

**Files:**
- Create: `track_gen/_src/course_line.py`
- Test: `tests/test_course_line.py`

**Interfaces:**
- Consumes: a `GateSequence` (vec3f, Task 2/4).
- Produces: `class CourseLine` — `__init__(self, seq: GateSequence, samples_per_gate: int = 8)`; attribute `track: Track` (a REAL `Track` instance: `center/tangent` vec3f filled from the closed Catmull-Rom through the gate centers, `arclen/length/count/valid` filled, `outer/inner/normal` NaN-filled, `winding` 0) with `N_max = samples_per_gate * max_gates`; method `refresh() -> None` (kernel launches only, capture-safe, call after every regeneration). Consumed by `TrackLocalizer` (Task 6), which only reads `center/arclen/length/count/valid`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import warp as wp

from track_gen._src.course_line import CourseLine
from track_gen._src.gate_generator import GateGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.types import GateGenConfig

E = 4


def _seq(**kw):
    cfg = GateGenConfig(device="cpu", num_envs=E, gate_width=0.05, **kw)
    rng = PerEnvSeededRNG(seeds=3, num_envs=E, device="cpu")
    return GateGenerator(cfg, rng).generate(), cfg


def test_interpolates_gate_anchors():
    seq, cfg = _seq(z_profile="random_walk", z_base=1.0, z_min=0.2,
                    z_max=2.0, z_max_step=0.4)
    line = CourseLine(seq, samples_per_gate=8)
    line.refresh()
    G, spg = int(cfg.max_gates), 8
    n_max = G * spg
    ctr = line.track.center.numpy().reshape(E, n_max, 3)
    gp = seq.position.numpy().reshape(E, G, 3)
    for e in np.flatnonzero(seq.valid.numpy()):
        n = int(seq.count.numpy()[e])
        for i in range(n):
            # sample j = i*spg sits exactly on gate i (CR interpolates knots)
            np.testing.assert_allclose(ctr[e, i * spg], gp[e, i], atol=1e-5)


def test_arclen_monotone_and_closed():
    seq, cfg = _seq(z_profile="uniform", z_min=0.5, z_max=1.5)
    line = CourseLine(seq, samples_per_gate=8)
    line.refresh()
    n_max = int(cfg.max_gates) * 8
    arc = line.track.arclen.numpy().reshape(E, n_max)
    length = line.track.length.numpy()
    cnt = line.track.count.numpy()
    for e in np.flatnonzero(seq.valid.numpy()):
        m = int(cnt[e])
        a = arc[e, :m]
        assert (np.diff(a) > 0).all()
        assert length[e] > a[-1] > 0.0


def test_refresh_tracks_regeneration():
    seq, cfg = _seq(z_profile="uniform", z_min=0.5, z_max=1.5)
    line = CourseLine(seq, samples_per_gate=4)
    line.refresh()
    before = line.track.center.numpy().copy()
    # GateGenerator overwrites in place on regenerate; rerun + refresh
    # (fresh rng advance via new seeds)
    rng2 = PerEnvSeededRNG(seeds=99, num_envs=E, device="cpu")
    seq2 = GateGenerator(cfg, rng2).generate()
    line2 = CourseLine(seq2, samples_per_gate=4)
    line2.refresh()
    assert not np.allclose(before, line2.track.center.numpy(), equal_nan=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_course_line.py -x -q`
Expected: FAIL — no module `course_line`.

- [ ] **Step 3: Implement**

```python
"""Resampled 3D centerline through a gate sequence (closed Catmull-Rom).

``CourseLine`` gives gates mode a Track-shaped centerline so the standard
Track consumers (``TrackLocalizer``; later the corridor tube) work unchanged:
``center``/``tangent``/``arclen``/``length``/``count``/``valid`` are real,
``outer``/``inner``/``normal`` are NaN (no road band), ``winding`` is 0.

count[e] = samples_per_gate * gate_count[e]; N_max = samples_per_gate *
max_gates. ``refresh()`` is two kernel launches, allocation-free and
capture-safe; call it after every gate regeneration (the facade does).
"""
import warp as wp

from .runtime import _init, _sync
from .types import GateSequence, Track


@wp.func
def _cr_point(p0: wp.vec3f, p1: wp.vec3f, p2: wp.vec3f, p3: wp.vec3f,
              t: float) -> wp.vec3f:
    # uniform Catmull-Rom (tangent at knot i = 0.5*(p_{i+1} - p_{i-1}),
    # matching the gate pipeline's central-difference gate tangents)
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2.0 * p1) + (p2 - p0) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)


@wp.kernel
def _sample_line_k(
    gate_pos: wp.array(dtype=wp.vec3f),
    gate_count: wp.array(dtype=wp.int32),
    gate_valid: wp.array(dtype=wp.int32),
    max_gates: int,
    spg: int,
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    n_max = max_gates * spg
    e = tid // n_max
    j = tid - e * n_max
    n = gate_count[e]
    if n > max_gates:
        n = max_gates
    m = n * spg
    if j == 0:
        count[e] = m
        valid[e] = gate_valid[e]
    if j >= m or n < 3 or gate_valid[e] == 0:
        center[tid] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        return
    gbase = e * max_gates
    i = j // spg
    t = float(j - i * spg) / float(spg)
    i0 = ((i - 1) % n + n) % n
    i2 = (i + 1) % n
    i3 = (i + 2) % n
    center[tid] = _cr_point(gate_pos[gbase + i0], gate_pos[gbase + i],
                            gate_pos[gbase + i2], gate_pos[gbase + i3], t)


@wp.kernel
def _line_frames_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    tangent: wp.array(dtype=wp.vec3f),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    base = e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        length[e] = 0.0
        for i in range(n_max):
            tangent[base + i] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            arclen[base + i] = wp.nan
        return
    acc = float(0.0)
    for i in range(n_max):
        if i < m:
            if i > 0:
                acc = acc + wp.length(center[base + i] - center[base + i - 1])
            arclen[base + i] = acc
            prev = ((i - 1) % m + m) % m
            nxt = (i + 1) % m
            d = center[base + nxt] - center[base + prev]
            l = wp.length(d)
            if l > 1.0e-12:
                tangent[base + i] = d / l
            else:
                tangent[base + i] = wp.vec3f(0.0, 0.0, 0.0)
        else:
            tangent[base + i] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            arclen[base + i] = wp.nan
    length[e] = acc + wp.length(center[base] - center[base + m - 1])


class CourseLine:
    """See module docstring."""

    def __init__(self, seq: GateSequence, samples_per_gate: int = 8) -> None:
        _init()
        if int(samples_per_gate) < 2:
            raise ValueError(
                f"samples_per_gate must be >= 2, got {samples_per_gate!r}")
        E = int(seq.count.shape[0])
        stride = int(seq.position.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"gate batch layout invalid: {stride} slots for {E} envs")
        self._seq = seq
        self._E = E
        self._G = stride // E
        self._spg = int(samples_per_gate)
        self._n_max = self._G * self._spg
        self._device = str(seq.position.device)
        dev = self._device
        n = E * self._n_max
        nan3 = float("nan")
        self.track = Track(
            outer=wp.full(n, wp.vec3f(nan3, nan3, nan3), device=dev),
            center=wp.zeros(n, dtype=wp.vec3f, device=dev),
            inner=wp.full(n, wp.vec3f(nan3, nan3, nan3), device=dev),
            tangent=wp.zeros(n, dtype=wp.vec3f, device=dev),
            normal=wp.full(n, wp.vec3f(nan3, nan3, nan3), device=dev),
            arclen=wp.zeros(n, dtype=wp.float32, device=dev),
            length=wp.zeros(E, dtype=wp.float32, device=dev),
            valid=wp.zeros(E, dtype=wp.int32, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
            winding=wp.zeros(E, dtype=wp.float32, device=dev),
        )

    def refresh(self) -> None:
        """Resample from the CURRENT gate batch (capture-safe)."""
        s, t = self._seq, self.track
        wp.launch(_sample_line_k, dim=self._E * self._n_max,
                  inputs=[s.position, s.count, s.valid, self._G, self._spg,
                          t.center, t.count, t.valid],
                  device=self._device)
        wp.launch(_line_frames_k, dim=self._E,
                  inputs=[t.center, t.count, self._n_max,
                          t.tangent, t.arclen, t.length],
                  device=self._device)
        _sync(self._device)
```

(If `wp.full` with a vec3f value is awkward in the installed Warp version, allocate with `wp.zeros` and NaN-fill once with `_fill_vec3_k` from Task 2.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_course_line.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/course_line.py tests/test_course_line.py
git commit -m "feat: CourseLine — periodic Catmull-Rom 3D centerline through gate sequences"
```

---

### Task 6: Course facade — localizer in gates mode

**Files:**
- Modify: `track_gen/_src/course.py`
- Test: `tests/test_course_gates_3d.py`

**Interfaces:**
- Consumes: `CourseLine` (Task 5), `TrackLocalizer` (3D since Task 2).
- Produces: in gates mode, `Course` builds `self.course_line: CourseLine` and `self.localizer: TrackLocalizer` after the first `generate()`; `course_line.refresh()` joins `_refresh()` (before the progress reset); `bind()`'s position buffer is shared with the localizer (`localizer.bind(position)` in `_apply_bind`). New `CourseConfig` fields: `samples_per_gate: int = 8`, `localize_window: "int | None" = None` (forwarded as `TrackLocalizer(warm_window=...)`; track-mode-forbidden validation mirrors the existing style: both raise in track mode). `step()` also calls `self.localizer.query()` in gates mode and `StepResult` gains `frame: "TrackFrame | None"`. `reset(mask)` forwards to `localizer.reset(mask)` too.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import warp as wp

from track_gen._src.course import Course, CourseConfig
from track_gen._src.types import GateGenConfig

E = 4


def _course(**kw):
    gcfg = GateGenConfig(device="cpu", num_envs=E, gate_width=0.2,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    c = Course(CourseConfig(mode="gates", gen=gcfg, seeds=11, **kw))
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    c.bind(pos)
    c.generate()
    return c, pos


def test_gates_course_has_localizer_and_line():
    c, _ = _course()
    assert c.course_line is not None
    assert c.localizer is not None
    assert int(c.course_line.track.valid.numpy().sum()) \
        == int(c.result.valid.numpy().sum())


def test_step_localizes_near_first_gate():
    c, pos = _course()
    e = int(np.flatnonzero(c.result.valid.numpy())[0])
    G = c.result.position.shape[0] // E
    gate0 = c.result.position.numpy()[e * G]
    p = pos.numpy()
    p[e] = gate0 + np.array([0.0, 0.0, 0.1], np.float32)
    wp.copy(pos, wp.array(p, dtype=wp.vec3f, device="cpu"))
    res = c.step()
    frame = res.frame
    assert frame is not None
    # foot point at gate 0 => s near 0 (mod length), n_up near +0.1
    L = float(c.course_line.track.length.numpy()[e])
    s = float(frame.s.numpy()[e])
    assert min(s, L - s) < 0.2 * L
    assert abs(float(frame.n_up.numpy()[e]) - 0.1) < 0.05


def test_track_mode_rejects_gate_options():
    from track_gen._src.types import TrackGenConfig
    import pytest
    tcfg = TrackGenConfig(device="cpu", num_envs=E)
    with pytest.raises(ValueError):
        CourseConfig(mode="track", gen=tcfg, checkpoint_spacing=0.1,
                     samples_per_gate=4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_course_gates_3d.py -x -q`
Expected: FAIL — unknown `CourseConfig` fields / missing attributes.

- [ ] **Step 3: Implement**

- `CourseConfig`: add the two fields; validation: track mode raises on non-default `samples_per_gate`/`localize_window` ("gates-mode options", existing message style); gates mode requires `samples_per_gate >= 2`.
- `Course.__init__`: `self.course_line = None`, `self.localizer = None`.
- `_build_subtools` gates branch: after `CheckpointSet.from_gates`, build `self.course_line = CourseLine(self.result, cfg.samples_per_gate)` and `self.localizer = TrackLocalizer(self.course_line.track, warm_window=cfg.localize_window)`.
- `_refresh`: gates mode calls `self.course_line.refresh()` FIRST (before the progress reset; order matters — checkpoints alias gates so nothing else needs refreshing), and `self.localizer.reset(self._reset_all_mask)`.
- `_apply_bind`: also `self.localizer.bind(a["position"])` when the localizer exists.
- `step()`: `frame = self.localizer.query() if self.localizer is not None else None`; `StepResult` gains `frame` field (+ `clone()`); construct once as before.
- `reset(mask)`: forward to `self.localizer.reset(mask)` when present.
- Update the module docstring's gates-mode line.

- [ ] **Step 4: Run tests + full suite**

Run: `pytest tests/test_course_gates_3d.py tests/test_course_track.py -q && pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/course.py tests/test_course_gates_3d.py
git commit -m "feat: gates-mode Course builds a CourseLine + 3D localizer, step() returns the track frame"
```

---

### Task 7: Gate-frame collision (square frames vs drone sphere)

**Files:**
- Create: `track_gen/_src/collision_frames.py`
- Modify: `track_gen/_src/course.py` (wiring + config)
- Test: `tests/test_collision_frames.py`

**Interfaces:**
- Consumes: `GateSequence` (`position`, `orientation`, `half_size`, `count`), `ProgressTracker._next` (aliased via a new read-only accessor `ProgressTracker.next_checkpoint_state` returning the `[E] int32` state array — add a one-line property in `progress.py`).
- Produces: `class FrameChecker` — `__init__(self, seq, num_envs, radius, frame_thickness, frame_depth, window=4)`; `bind_inputs(position: wp.array)` (`[E]` vec3f) and `bind_window(next_cp: wp.array)`; `query() -> FrameContact` where `FrameContact` has `hit` `[E] int32` and `depth` `[E] float32` (max penetration, 0 when no hit; preallocated, overwritten per query; `clone()`). `CourseConfig` gains `frame_collision: bool = False`, `frame_thickness: float = 0.0`, `frame_depth: float = 0.0` (gates-mode-only; mutually exclusive with `post_radius > 0`; `frame_collision=True` requires both > 0 and uses `max_boxes == 1`); the drone radius rides the existing convention: reuse `post_radius`? NO — posts are exclusive; add `agent_radius: float = 0.0` (required > 0 with `frame_collision`). In gates mode with `frame_collision`, `Course` wires `self.collision = FrameChecker(...)`, binds the position buffer, binds the window to `self.progress.next_checkpoint_state`, and `StepResult.contacts` carries the `FrameContact`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import warp as wp

from track_gen._src.collision_frames import FrameChecker
from track_gen._src.types import GateSequence

E, G = 2, 4


def _square_gate_seq():
    """One valid env with a single gate at origin, facing +x, half_size 1."""
    n = E * G
    nan3 = np.full(3, np.nan, np.float32)
    pos = np.tile(nan3, (n, 1)); tan = pos.copy()
    left = pos.copy(); right = pos.copy()
    quat = np.tile(np.full(4, np.nan, np.float32), (n, 1))
    hs = np.full(n, np.nan, np.float32)
    pos[0] = (0, 0, 0); tan[0] = (1, 0, 0)
    quat[0] = (0, 0, 0, 1)              # identity: fwd=+x, left=+y, up=+z
    hs[0] = 1.0
    left[0] = (0, 1, 0); right[0] = (0, -1, 0)
    dev = "cpu"
    return GateSequence(
        position=wp.array(pos, dtype=wp.vec3f, device=dev),
        tangent=wp.array(tan, dtype=wp.vec3f, device=dev),
        orientation=wp.array(quat, dtype=wp.quatf, device=dev),
        half_size=wp.array(hs, dtype=wp.float32, device=dev),
        left=wp.array(left, dtype=wp.vec3f, device=dev),
        right=wp.array(right, dtype=wp.vec3f, device=dev),
        valid=wp.array(np.array([1, 0], np.int32), device=dev),
        count=wp.array(np.array([1, 0], np.int32), device=dev),
    )


def _query_at(p):
    seq = _square_gate_seq()
    chk = FrameChecker(seq, num_envs=E, radius=0.1, frame_thickness=0.1,
                       frame_depth=0.1)
    pos = wp.array(np.array([p, [0, 0, 0]], np.float32), dtype=wp.vec3f,
                   device="cpu")
    chk.bind_inputs(pos)
    chk.bind_window(wp.zeros(E, dtype=wp.int32, device="cpu"))
    return chk.query().hit.numpy()[0]


def test_through_opening_no_hit():
    assert _query_at([0.0, 0.0, 0.0]) == 0      # dead center
    assert _query_at([0.0, 0.5, 0.5]) == 0      # inside opening


def test_post_and_bar_hits():
    assert _query_at([0.0, 1.05, 0.0]) == 1     # left post
    assert _query_at([0.0, -1.05, 0.0]) == 1    # right post
    assert _query_at([0.0, 0.0, 1.05]) == 1     # top bar
    assert _query_at([0.0, 0.0, -1.05]) == 1    # bottom bar


def test_far_away_no_hit():
    assert _query_at([5.0, 5.0, 5.0]) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_collision_frames.py -x -q`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `collision_frames.py`**

```python
"""Sphere-vs-gate-frame collision for 3D gate courses.

Each square gate is four thin oriented boxes in the gate's local frame
(x=forward/depth, y=left, z=up): posts at y = +/-(hs + t/2) spanning the
opening height plus corners, bars at z = +/-(hs + t/2). The agent is a
sphere; a hit is sphere-vs-box penetration against any frame member of the
gates inside a fixed index window around the CURRENT target checkpoint
(prev, target, and the next ``window - 2`` gates) — fixed-size, so the
query is one kernel, allocation-free and capture-safe.
"""
from dataclasses import dataclass

import warp as wp

from .runtime import _check_arr, _init, _sync
from .types import GateSequence


@dataclass
class FrameContact:
    """Per-env frame-collision result; overwritten in place per query.

    hit : [E] int32 — 1 iff the sphere penetrates any frame box this query.
    depth : [E] float32 — max penetration depth, 0.0 when no hit.
    """

    hit: wp.array
    depth: wp.array

    def clone(self) -> "FrameContact":
        return FrameContact(hit=wp.clone(self.hit), depth=wp.clone(self.depth))


@wp.func
def _sd_box(p: wp.vec3f, half: wp.vec3f) -> float:
    # signed distance point -> origin-centered AABB
    q = wp.vec3f(wp.abs(p[0]) - half[0], wp.abs(p[1]) - half[1],
                 wp.abs(p[2]) - half[2])
    outside = wp.length(wp.vec3f(wp.max(q[0], 0.0), wp.max(q[1], 0.0),
                                 wp.max(q[2], 0.0)))
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    return outside + inside


@wp.kernel
def _frame_query_k(
    gate_pos: wp.array(dtype=wp.vec3f),
    gate_quat: wp.array(dtype=wp.quatf),
    gate_hs: wp.array(dtype=wp.float32),
    gate_count: wp.array(dtype=wp.int32),
    gate_valid: wp.array(dtype=wp.int32),
    max_gates: int,
    next_cp: wp.array(dtype=wp.int32),
    window: int,
    position: wp.array(dtype=wp.vec3f),
    radius: float,
    thick: float,
    depth_x: float,
    out_hit: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    out_hit[e] = 0
    out_depth[e] = 0.0
    n = gate_count[e]
    if n > max_gates:
        n = max_gates
    if gate_valid[e] == 0 or n < 1:
        return
    base = e * max_gates
    p = position[e]
    if p[0] != p[0]:
        return
    g0 = next_cp[e] - 1
    worst = float(0.0)
    hit = int(0)
    for k in range(window):
        gi = ((g0 + k) % n + n) % n
        c = gate_pos[base + gi]
        q = gate_quat[base + gi]
        hs = gate_hs[base + gi]
        if c[0] != c[0] or hs != hs:
            continue
        lp = wp.quat_rotate_inv(q, p - c)
        hd = 0.5 * depth_x
        ht = 0.5 * thick
        span = hs + thick          # posts cover the corners
        # posts: centers (0, +/-(hs+ht), 0), half (hd, ht, span)
        # bars:  centers (0, 0, +/-(hs+ht)), half (hd, span, ht)
        for m in range(4):
            off = wp.vec3f(0.0, 0.0, 0.0)
            half = wp.vec3f(hd, ht, span)
            if m == 0:
                off = wp.vec3f(0.0, hs + ht, 0.0)
            elif m == 1:
                off = wp.vec3f(0.0, -(hs + ht), 0.0)
            elif m == 2:
                off = wp.vec3f(0.0, 0.0, hs + ht)
                half = wp.vec3f(hd, span, ht)
            else:
                off = wp.vec3f(0.0, 0.0, -(hs + ht))
                half = wp.vec3f(hd, span, ht)
            d = _sd_box(lp - off, half) - radius
            if d < 0.0:
                hit = int(1)
                if -d > worst:
                    worst = -d
    out_hit[e] = hit
    out_depth[e] = worst


class FrameChecker:
    """See module docstring. Mirrors DiscChecker's bind/query lifecycle."""

    def __init__(self, seq: GateSequence, num_envs: int, radius: float,
                 frame_thickness: float, frame_depth: float,
                 window: int = 4) -> None:
        _init()
        for name, v in (("radius", radius), ("frame_thickness", frame_thickness),
                        ("frame_depth", frame_depth)):
            if not (float(v) > 0.0):
                raise ValueError(f"{name} must be > 0, got {v!r}")
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        E = int(num_envs)
        stride = int(seq.position.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"gate batch layout invalid: {stride} slots for {E} envs")
        self._seq = seq
        self._E = E
        self._G = stride // E
        self._radius = float(radius)
        self._thick = float(frame_thickness)
        self._depth = float(frame_depth)
        self._window = int(window)
        self._device = str(seq.position.device)
        self._pos: "wp.array | None" = None
        self._next: "wp.array | None" = None
        self._contact = FrameContact(
            hit=wp.zeros(E, dtype=wp.int32, device=self._device),
            depth=wp.zeros(E, dtype=wp.float32, device=self._device))

    def bind_inputs(self, position: wp.array) -> None:
        _check_arr("position", position, (self._E,), wp.vec3f, self._device)
        self._pos = position

    def bind_window(self, next_cp: wp.array) -> None:
        """Bind the [E] int32 current-target buffer (ProgressTracker state)."""
        _check_arr("next_cp", next_cp, (self._E,), wp.int32, self._device)
        self._next = next_cp

    def query(self) -> FrameContact:
        if self._pos is None or self._next is None:
            raise RuntimeError("call bind_inputs() and bind_window() first")
        s = self._seq
        wp.launch(_frame_query_k, dim=self._E,
                  inputs=[s.position, s.orientation, s.half_size, s.count,
                          s.valid, self._G, self._next, self._window,
                          self._pos, self._radius, self._thick, self._depth,
                          self._contact.hit, self._contact.depth],
                  device=self._device)
        _sync(self._device)
        return self._contact
```

- [ ] **Step 4: Course wiring**

`CourseConfig`: add `frame_collision: bool = False`, `frame_thickness: float = 0.0`, `frame_depth: float = 0.0`, `agent_radius: float = 0.0`. Validation (gates branch): `frame_collision` with `post_radius > 0` raises ("choose gate-post discs OR gate frames"); `frame_collision=True` requires `frame_thickness > 0`, `frame_depth > 0`, `agent_radius > 0`; all four raise in track mode (existing message style). In `_build_subtools` gates branch: when `cfg.frame_collision`, `self.collision = FrameChecker(self.result, num_envs=self._E, radius=cfg.agent_radius, frame_thickness=cfg.frame_thickness, frame_depth=cfg.frame_depth)` and in `_apply_bind`: `self.collision.bind_inputs(a["position"])` plus `self.collision.bind_window(self.progress.next_checkpoint_state)` (add to `progress.py`: `@property def next_checkpoint_state(self): return self._next` with a docstring marking it read-only state aliasing). `_needs_boxes()` must NOT report frame collision (no box binding needed): keep its current expression and exclude `frame_collision` explicitly. Add a course-level test in `tests/test_course_gates_3d.py`: build with `frame_collision=True, agent_radius=0.05, frame_thickness=0.05, frame_depth=0.05`, place a drone on a valid env's first-gate post, `step()`, assert `contacts.hit` for that env.

- [ ] **Step 5: Run tests + full suite**

Run: `pytest tests/test_collision_frames.py tests/test_course_gates_3d.py -q && pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/collision_frames.py track_gen/_src/course.py track_gen/_src/progress.py tests/
git commit -m "feat: sphere-vs-gate-frame collision with windowed queries, Course wiring"
```

---

### Task 8: Gate-course viz

**Files:**
- Create: `viz/plot_gate_courses.py`
- Test: none (viz scripts are exercised manually; keep it import-clean)

**Interfaces:**
- Consumes: `GateGenerator` + `CourseLine`.
- Produces: `python viz/plot_gate_courses.py --out gate_courses.png` renders a grid of valid envs, two panels each: plan view (XY: centerline + gate segments left-right) and elevation profile (arclen vs z of the course line, gate altitudes as markers). Follow the CLI/style conventions of `viz/plot_tracks.py` (argparse, matplotlib, no seaborn, save-not-show).

- [ ] **Step 1: Implement** — mirror `viz/plot_tracks.py`'s structure: build a cpu `GateGenConfig` (`z_profile="random_walk"`, `z_base=1.5`, `z_min=0.5`, `z_max=2.5`, `z_max_step=0.4`, `gate_align="full_tangent"`, `gate_width=0.2`), generate, build `CourseLine(seq, 8)`, `refresh()`, and for each of the first N valid envs draw: left panel `center[:, 0], center[:, 1]` plus per-gate `[left, right]` xy segments; right panel `arclen[:m]` vs `center[:m, 2]` plus gate markers at their arc positions (`arclen[i * spg]`). Label axes; `--envs`, `--seed`, `--out` flags.

- [ ] **Step 2: Verify**

Run: `python viz/plot_gate_courses.py --out /tmp/gate_courses.png && python -c "from PIL import Image; im=Image.open('/tmp/gate_courses.png'); print(im.size)"`
Expected: a size printout; open the PNG and eyeball: smooth closed plan curves through the gates, elevation within [0.5, 2.5].

- [ ] **Step 3: Commit**

```bash
git add viz/plot_gate_courses.py
git commit -m "viz: gate-course plan + elevation plots"
```

---

### Task 9: Docs + migration notes

**Files:**
- Create: `docs/gates-3d.rst` (placed in the same toctree as the existing gates tutorial — locate it: `grep -rn "gates" docs/index.rst docs/*/index.rst`)
- Modify: `README.md`, `docs/relaxation/gates.rst`, every doc page showing `(…, 2)` reshapes

**Interfaces:** none (docs only).

- [ ] **Step 1: Sweep the reshape idiom**

Run: `grep -rn "view(E\|, 2)\|vec2f\|yaw" README.md docs/ --include=*.rst --include=*.md | grep -v superpowers | grep -v related-work`

Update every hit that describes the public API: reshapes to `(…, 3)`, `GateSequence.normal` references removed in favor of `orientation`/`half_size`, `Course.bind` yaw→orientation. Leave `docs/related-work/` and `docs/superpowers/` untouched (historical).

- [ ] **Step 2: Write `docs/gates-3d.rst`** — sections: why 2.5D (layout from the proven 2D generators, Z as a first-class profiler); the four `z_profile`s with their config knobs and one rendered figure (`viz/plot_gate_courses.py` output, committed under `docs/_static/gate-courses.png`); `gate_align` modes; runtime parity (pass detection semantics, `TrackFrame.s/n/n_up`, `frame_collision`); validity (`z_valid_grade`, and the spec's two explicit v1 limitations: plan-view crossings still rejected, spline XY drift is unchecked-by-design); migration box listing the breaking changes verbatim from Task 2's commit message. Add to the toctree next to the gates tutorial.

- [ ] **Step 3: Build docs if the repo has a docs build** (check `docs/Makefile` / `pyproject`): `make -C docs html` or the project's equivalent; fix warnings introduced by the new page.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: 3D gate courses page, vec3f migration notes across the docs"
```

---

### Task 10: Capture safety + final verification

**Files:**
- Modify: `tests/test_generate_concurrent_cuda.py`
- Test: full matrix

**Interfaces:** none new.

- [ ] **Step 1: Extend the concurrency test** — add a gates-mode 3D variant mirroring the existing track-mode case: two threads, each building a gates-mode `Course` (`z_profile="uniform"`, `frame_collision=True`, `agent_radius=0.05`, `frame_thickness=0.05`, `frame_depth=0.05`), concurrent `generate()` + `step()` on cuda; assert no CUDA errors and both batches valid-count > 0. Keep the `_CAPTURE_LOCK` regression assertions of the existing test untouched.

- [ ] **Step 2: Full matrix**

Run: `pytest tests/ -q` (cpu) and, on the CUDA machine, `pytest tests/ -q` again (the suite's cuda-parametrized tests) plus `pytest tests/test_generate_concurrent_cuda.py tests/test_warp_graph.py -v`
Expected: all PASS, including `test_golden_migration` (the standing proof that 2D collision and generation behavior survived — same OOB verdicts, same geometry).

- [ ] **Step 3: Commit**

```bash
git add tests/test_generate_concurrent_cuda.py
git commit -m "test: gates-mode 3D concurrent capture coverage"
```

---

## Plan Self-Review Notes (resolved)

- **Spec coverage:** data model → Task 2; Z profilers → Task 3; pipeline lift/spline/poses → Tasks 4–5; runtime parity (progress, localize, frame collision, Course) → Tasks 2, 6, 7; 2D collision functional guarantee → Task 1 goldens + Task 2 Step 8 + Task 10; validity → Tasks 2 (unchanged 2D checks), 4 (grade); viz/docs → Tasks 8–9; capture safety → Task 10.
- **Spec deviations (deliberate, spot-checked against the code):** (1) `Course` mode is the EXISTING `"gates"`, not a new `"gate_course"`. (2) Z/alignment config lands on `GateGenConfig` only — `TrackGenConfig` is untouched until the tube stage. (3) `gate_half_size` is not a new knob: `half_size = 0.5 * gate_width` (existing knob, no duplication). (4) The spec's "min 3D gate spacing" is subsumed: the existing pairwise XY min-distance check is strictly stronger than any 3D-distance floor with the same threshold. (5) The near-vertical fallback uses the horizontal tangent direction (then +x), not the neighbor's yaw — same effect, no cross-thread gate dependency in the kernel.
- **Type consistency:** `up_half` (CheckpointSet) vs `half_size` (GateSequence) — aliased in `from_gates`, `_BIG` for track cross-sections; `_plane_pass` consumes `v_half` scalar per checkpoint. Position buffers `[E] vec3f` everywhere (`ProgressTracker`, `TrackLocalizer`, `Course.bind`, `FrameChecker.bind_inputs`).
