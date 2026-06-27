# Code-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Critical and Important findings from the 2026-06-27 full code review (and a few cheap high-value suggestions), hardening config validation and the RNG kernels so reachable crashes/corruption/correctness bugs fail loudly or are eliminated.

**Architecture:** Mostly additive validation + small kernel corrections. The biggest win is one config-validation pass on `TrackGenConfig.__post_init__` that consolidates four Important findings (it currently lacks guards its twin `GateGenConfig` already has). The RNG fixes correct an indexing-stride bug across kernels, a state double-buffer init bug, a dispatcher type gap, and a quaternion seed-reuse, backed by the repo's first RNG test file.

**Tech Stack:** Python, NVIDIA Warp, numpy, pytest. Run tests with `env -u PYTHONPATH .venv/bin/python -m pytest`. Branch: `code-review-fixes` (already checked out).

Source spec: `docs/code-review-2026-06-27.md`.

---

## Context notes (verified during planning)
- `TrackGenConfig` defaults all pass the new guards (`min_num_points=9`, `max_num_points=13`, `num_points_per_segment=30`, `min_point_distance=0.05`, `num_points=256`, `N_max=384`, `half_width=0.1`, `num_envs=1`). Existing tests size `N_max >= num_points` (e.g. `1024/1100`, `64/128`), so none regress.
- The track explorer can already reach `num_points(256) > N_max` when the `N_max` slider is dropped below 256 — today that path risks out-of-bounds GPU writes; the new guard converts it to a clean `ValueError` the UI already catches.
- There is currently **no** RNG test file; the 3D stride bug went undetected. `quaternion` is unused in the runtime (only `_experimental` touches this RNG), so Task 5 is correctness hygiene, not a hot path.
- `PerEnvSeededRNG` exposes `sample_uniform_warp(low, high, shape, ids=None)`, `sample_normal_warp`, `sample_quaternion_warp(shape, ids=None)`, etc. Output shape is `(num_envs,) + shape`. `states_warp` is the public state array.

---

### Task 1: Harden `TrackGenConfig.__post_init__`

Consolidates Important findings: missing `min_point_distance` / `min_num_points` / `max_num_points` / `num_points_per_segment` guards, missing `num_points <= N_max` invariant (OOB-write risk), and missing `half_width` / `spacing` / `num_envs` positivity.

**Files:**
- Modify: `track_gen/_src/types.py` (`TrackGenConfig.__post_init__`, starts at line 218)
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_types.py`:

```python
def test_track_config_validates_sampler_and_buffer_invariants():
    from track_gen._src.types import TrackGenConfig
    import pytest
    with pytest.raises(ValueError, match="num_envs"):
        TrackGenConfig(num_envs=0)
    with pytest.raises(ValueError, match="half_width"):
        TrackGenConfig(half_width=0.0)
    with pytest.raises(ValueError, match="min_num_points"):
        TrackGenConfig(min_num_points=1)
    with pytest.raises(ValueError, match="max_num_points"):
        TrackGenConfig(min_num_points=13, max_num_points=9)
    with pytest.raises(ValueError, match="min_point_distance"):
        TrackGenConfig(min_point_distance=0.0)
    with pytest.raises(ValueError, match="min_point_distance"):
        TrackGenConfig(min_point_distance=0.6)  # > 0.5 -> num_cells == 0 -> div-by-zero
    with pytest.raises(ValueError, match="num_points_per_segment"):
        TrackGenConfig(num_points_per_segment=1)
    with pytest.raises(ValueError, match="num_points must be <= N_max"):
        TrackGenConfig(num_points=512, N_max=384)  # would overrun the N_max-sized scan scratch
    with pytest.raises(ValueError, match="spacing"):
        TrackGenConfig(spacing=0.0)


def test_track_config_defaults_still_valid():
    from track_gen._src.types import TrackGenConfig
    cfg = TrackGenConfig()  # must not raise
    assert cfg.num_points <= cfg.N_max
    assert cfg.spacing == 0.6 * cfg.half_width
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_types.py::test_track_config_validates_sampler_and_buffer_invariants -v`
Expected: FAIL — `TrackGenConfig(num_envs=0)` etc. construct without raising (the test's first `pytest.raises` fails).

- [ ] **Step 3: Add the guards**

In `track_gen/_src/types.py`, insert this block at the **top** of `TrackGenConfig.__post_init__` (immediately after the `def __post_init__(self):` line, before the existing `if int(self.voronoi_control_points) < 6:` check):

```python
        if int(self.num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs!r}")
        if float(self.half_width) <= 0.0:
            raise ValueError(f"half_width must be > 0, got {self.half_width!r}")
        # Point-family sampler inputs (shared bezier/hull corner sampler). min_point_distance
        # is a divisor: num_cells = int(1/(min_point_distance*2)) in warp_generate.py, so
        # values <= 0 or > 0.5 drive a divide-by-zero in the grid kernels; min/max_num_points
        # feed wp.randi and must be a non-inverted range. GateGenConfig already guards these.
        if int(self.min_num_points) < 2:
            raise ValueError(f"min_num_points must be >= 2, got {self.min_num_points!r}")
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}")
        if not (0.0 < float(self.min_point_distance) <= 0.5):
            raise ValueError(
                f"min_point_distance must be in (0, 0.5], got {self.min_point_distance!r}")
        if int(self.num_points_per_segment) < 2:
            raise ValueError(
                f"num_points_per_segment must be >= 2, got {self.num_points_per_segment!r}")
        # The constant-spacing scan scratch is sized to N_max but fed the num_points-strided
        # generation buffer; num_points > N_max would write past it (OOB GPU memory).
        if int(self.num_points) > int(self.N_max):
            raise ValueError(
                f"num_points must be <= N_max (the resample scratch is sized to N_max); "
                f"got num_points={self.num_points!r} > N_max={self.N_max!r}")
```

Then, at the **end** of `__post_init__`, replace the trailing spacing block:

```python
        if self.spacing is None:
            self.spacing = 0.6 * self.half_width
```

with:

```python
        if self.spacing is None:
            self.spacing = 0.6 * self.half_width
        if float(self.spacing) <= 0.0:
            raise ValueError(f"spacing must be > 0, got {self.spacing!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_types.py -q`
Expected: PASS (new tests + existing type tests).

Then the full suite: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green (314 passed baseline + new tests; no regressions).

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/types.py tests/test_types.py
git commit -m "fix: validate TrackGenConfig sampler inputs and num_points<=N_max"
```

---

### Task 2: Fix the RNG 3D intra-block stride bug (Critical)

All 3D kernels linearize the `(j,k)` block with `j*shape[1]+k` (stride = first trailing dim) instead of `j*shape[2]+k` (row-major stride = last dim). With unequal trailing dims this collides/overlaps RNG state → duplicated or correlated draws. 12 occurrences (10 distribution kernels + the 2 quaternion lines).

**Files:**
- Modify: `track_gen/_src/rng_kernels.py` (12 lines: 109, 241, 401, 464, 657, 725, 916, 990, 1212, 1287, 1489, 1490)
- Test: `tests/test_rng.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_rng.py`:

```python
import warp as wp
from track_gen._src.rng_utils import PerEnvSeededRNG


def _rng(num_envs=4, seed=0):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device="cpu")


def test_uniform_3d_block_has_no_index_collisions():
    # A (3,5) block has 15 distinct (j,k) cells. With the correct row-major stride
    # (j*shape[2]+k) each cell seeds a distinct PCG state -> 15 distinct floats per env.
    # The stride bug (j*shape[1]+k) collides cells (e.g. (1,0) and (0,3) both map to 3),
    # producing duplicate values within a single draw.
    rng = _rng(num_envs=2, seed=7)
    out = wp.to_torch(rng.sample_uniform_warp(0.0, 1.0, (3, 5)))  # (2,3,5)
    for e in range(out.shape[0]):
        flat = out[e].reshape(-1)
        assert flat.unique().numel() == flat.numel(), "duplicate RNG values within a 3D draw"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py::test_uniform_3d_block_has_no_index_collisions -v`
Expected: FAIL — duplicate values present (collided indices yield identical `randf`).

- [ ] **Step 3: Fix the stride in all 3D kernels**

In `track_gen/_src/rng_kernels.py`, replace every occurrence of `j * shape[1] + k` with `j * shape[2] + k` (12 occurrences). This is a safe global replace — the pattern appears only in the 3D kernels' state-index expression.

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py -v`
Expected: PASS.

Then full suite: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green (no regression — the generators use their own per-kernel `wp.rand_init`, not these 3D paths).

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/rng_kernels.py tests/test_rng.py
git commit -m "fix: correct 3D RNG kernel intra-block stride (shape[2], not shape[1])"
```

---

### Task 3: Fix RNG state double-buffer init (Important)

`_new_states` is `wp.zeros` and never mirrored from `_states`, while every sampler does a whole-array `wp.copy(states, new_states)`. A first partial-`ids` sample writes `new_states` only for selected envs, so the unconditional copy zeroes every non-selected env's state — destroying seed diversity.

**Files:**
- Modify: `track_gen/_src/rng_utils.py` (after line 38, inside `__init__`)
- Test: `tests/test_rng.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rng.py`:

```python
def test_partial_ids_sample_preserves_untouched_env_states():
    import numpy as np
    rng = _rng(num_envs=4, seed=11)
    before = wp.to_torch(rng.states_warp).clone()
    ids = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    rng.sample_uniform_warp(0.0, 1.0, (1,), ids=ids)  # touch only env 0
    after = wp.to_torch(rng.states_warp)
    # Envs 1..3 were not sampled; their states must be unchanged (not zeroed).
    assert (after[1:] == before[1:]).all(), "partial-ids sample corrupted untouched env states"
    assert (after[1:] != 0).any(), "untouched env states were zeroed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py::test_partial_ids_sample_preserves_untouched_env_states -v`
Expected: FAIL — envs 1..3 states become 0 after the partial-ids sample.

- [ ] **Step 3: Mirror the state buffer at construction**

In `track_gen/_src/rng_utils.py`, in `__init__`, immediately after `self.set_seeds_warp(self._seeds, None)` (line 38), add:

```python
        # Mirror states into the double-buffer so a first PARTIAL-ids sample (which only
        # writes new_states for selected envs, then wp.copy's the WHOLE array back) cannot
        # zero the untouched envs' seed-derived states.
        wp.copy(self._new_states, self._states)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py -q`
Expected: PASS.

Then full suite: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/rng_utils.py tests/test_rng.py
git commit -m "fix: mirror RNG state double-buffer so partial-ids sampling can't zero envs"
```

---

### Task 4: Accept integer bounds in `uniform()` / `normal()` (Important)

`uniform()` and `normal()` branch only on `wp.array` vs `float`; passing plain Python ints (`uniform(0, 1, ...)`) matches neither branch and raises `UnboundLocalError`. `integer()` already accepts ints — this is an API inconsistency.

**Files:**
- Modify: `track_gen/_src/rng_kernels.py` (`uniform` at line 316, `normal` at ~line 1413 — the `elif isinstance(low, float) or isinstance(high, float):` lines)
- Test: `tests/test_rng.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rng.py`:

```python
def test_uniform_and_normal_accept_python_int_bounds():
    rng = _rng(num_envs=3, seed=3)
    u = wp.to_torch(rng.sample_uniform_warp(0, 1, (2,)))  # int bounds, must not raise
    assert u.shape == (3, 2)
    assert (u >= 0.0).all() and (u < 1.0).all()
    n = wp.to_torch(rng.sample_normal_warp(0, 1, (2,)))  # int mean/std, must not raise
    assert n.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py::test_uniform_and_normal_accept_python_int_bounds -v`
Expected: FAIL with `UnboundLocalError` (or similar) — `output` never assigned for int bounds.

- [ ] **Step 3: Broaden the scalar branch to accept ints**

In `track_gen/_src/rng_kernels.py`, in `uniform()`, change:

```python
    elif isinstance(low, float) or isinstance(high, float):
        if isinstance(low, wp.array) or isinstance(high, wp.array):
            raise ValueError("The low value must be a tensor if the high value is a tensor.")
        output = uniform_single(low, high, states, new_states, ids, shape, device=device)
```

to:

```python
    elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
        output = uniform_single(float(low), float(high), states, new_states, ids, shape, device=device)
    else:
        raise TypeError(
            f"low and high must both be float/int scalars or both wp.array, "
            f"got {type(low).__name__} and {type(high).__name__}")
```

In `normal()` (same structure), change the corresponding `elif isinstance(mean, float) or isinstance(std, float):` branch to:

```python
    elif isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        output = normal_single(float(mean), float(std), states, new_states, ids, shape, device=device)
    else:
        raise TypeError(
            f"mean and std must both be float/int scalars or both wp.array, "
            f"got {type(mean).__name__} and {type(std).__name__}")
```

(Note: verify the scalar helper names `uniform_single` / `normal_single` by reading the two functions before editing; use whatever the existing `float`-branch calls.)

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py -q`
Expected: PASS.

Then full suite: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/rng_kernels.py tests/test_rng.py
git commit -m "fix: accept integer bounds in uniform()/normal() RNG dispatchers"
```

---

### Task 5: Decorrelate quaternion axis and angle seeds (Important)

The quaternion kernels read the SAME state value for both `wp.sample_unit_sphere` (axis) and `wp.randf` (angle), so the angle is a deterministic function of the axis. Add a large odd decorrelation salt to the angle's seed in the 1D/2D/3D kernels. (The stride is already fixed by Task 2.) `quaternion` is unused in the runtime, so this is correctness hygiene.

**Files:**
- Modify: `track_gen/_src/rng_kernels.py` (`rand_quaternion_1D` line 1442, `rand_quaternion_2D` line 1465, `rand_quaternion_3D` line 1490)
- Test: `tests/test_rng.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rng.py`:

```python
def test_quaternion_draws_are_not_all_identical_within_a_block():
    # Axis and angle previously shared a seed, making the angle a function of the axis;
    # combined with any index collision this produced repeated/over-correlated quats.
    # After decorrelation, a (6,) block of quats for one env should not be all-equal.
    rng = _rng(num_envs=2, seed=5)
    q = wp.to_torch(rng.sample_quaternion_warp((6,)))  # (2,6,4)
    for e in range(q.shape[0]):
        block = q[e]
        first = block[0]
        assert not bool((block == first).all()), "quaternion block is degenerate/identical"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py::test_quaternion_draws_are_not_all_identical_within_a_block -v`
Expected: this test guards against degeneracy; it may already pass for distinct indices. Treat it as a regression guard. If it passes pre-fix, still apply Step 3 (the correlation defect is real even when values differ) and keep the test as a guard.

- [ ] **Step 3: Salt the angle seed**

In `track_gen/_src/rng_kernels.py`:

`rand_quaternion_1D` (line 1442) — change:
```python
    angle = wp.randf(states[ids[tid]], 0.0, 2.0 * 4.0 * wp.atan(1.0))
```
to:
```python
    angle = wp.randf(states[ids[tid]] + wp.uint32(2654435761), 0.0, 2.0 * 4.0 * wp.atan(1.0))
```

`rand_quaternion_2D` (line 1465) — change:
```python
    angle = wp.randf(states[ids[i]] + wp.uint32(j), 0.0, 2.0 * 4.0 * wp.atan(1.0))
```
to:
```python
    angle = wp.randf(states[ids[i]] + wp.uint32(j) + wp.uint32(2654435761), 0.0, 2.0 * 4.0 * wp.atan(1.0))
```

`rand_quaternion_3D` (line 1490, after Task 2 made it `shape[2]`) — change:
```python
    angle = wp.randf(states[ids[i]] + wp.uint32(j * shape[2] + k), 0.0, 2.0 * 4.0 * wp.atan(1.0))
```
to:
```python
    angle = wp.randf(states[ids[i]] + wp.uint32(j * shape[2] + k) + wp.uint32(2654435761), 0.0, 2.0 * 4.0 * wp.atan(1.0))
```

(`2654435761` is Knuth's multiplicative hash constant — a large odd salt; it gives the angle an independent PCG seed from the axis. Add a one-line comment at the first use noting why.)

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_rng.py -q`
Expected: PASS.

Then full suite: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/rng_kernels.py tests/test_rng.py
git commit -m "fix: decorrelate quaternion axis/angle seeds with a salt"
```

---

### Task 6: Make the `TrackGenerator` determinism contract explicit (Important)

`generate()` re-copies the RNG's fixed seeds every call, so repeated calls return the IDENTICAL batch — but the docstrings imply fresh randomness. Make the contract explicit (the conservative fix; do not silently change RNG behavior the oracle tests depend on) and pin it with a test.

**Files:**
- Modify: `track_gen/_src/track_generator.py` (generate() docstring ~line 123-124; seed-refresh comment line 153-154)
- Test: `tests/test_track_generator_facade.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_track_generator_facade.py` (it already imports `TrackGenConfig`, `TrackGenerator`, `PerEnvSeededRNG` — reuse those; mirror an existing test's construction):

```python
def test_repeated_generate_is_deterministic_for_a_fixed_rng():
    import warp as wp
    E, N_max = 4, 128
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=64, N_max=N_max, device="cpu")
    rng = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, rng)
    a = wp.to_torch(gen.generate().center).clone()
    b = wp.to_torch(gen.generate().center).clone()
    # Documented contract: output is deterministic for a fixed rng state (same instance,
    # buffers reused). Callers vary the batch by reseeding the rng between calls.
    assert wp.types.warp.config  # noop guard to keep import explicit
    import torch
    assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
```

(If `test_track_generator_facade.py` lacks the `TrackGenConfig`/`TrackGenerator`/`PerEnvSeededRNG` imports at module scope, copy the import line from the top of that file's existing tests.)

- [ ] **Step 2: Run test to verify it passes (documents current behavior)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_track_generator_facade.py::test_repeated_generate_is_deterministic_for_a_fixed_rng -v`
Expected: PASS — this test pins the real (deterministic) behavior. Remove the stray `assert wp.types.warp.config` line if it errors; it is only a reminder to keep the import — the meaningful assertion is `torch.equal`.

- [ ] **Step 3: Fix the docstrings to state the contract**

In `track_gen/_src/track_generator.py`, replace the docstring paragraph at lines 122-124:

```python
        Writes results into ``self._track`` in place and returns the SAME instance every
        call (stable ``.ptr`` pointers). Use ``Track.clone()`` to obtain an independent copy.
```

with:

```python
        Writes results into ``self._track`` in place and returns the SAME instance every
        call (stable ``.ptr`` pointers). Use ``Track.clone()`` to obtain an independent copy.

        Determinism: ``generate()`` re-copies the rng's CURRENT seeds each call, so repeated
        calls with an unchanged rng return the IDENTICAL batch. To vary the batch between
        calls, reseed the rng first (e.g. ``rng.set_seeds_warp(new_seeds, None)``).
```

And clarify the comment at line 153:

```python
        # Refresh the seed buffer in place from the rng (zero allocation: wp.copy).
        # rng.seeds_warp is a wp.array [num_envs] int32, matching the fixed batch.
```

to:

```python
        # Refresh the seed buffer in place from the rng's CURRENT seeds (zero allocation:
        # wp.copy). The rng holds fixed seeds unless reseeded, so back-to-back generate()
        # calls are deterministic; reseed the rng to vary the batch. seeds_warp is [E] int32.
```

- [ ] **Step 4: Run the test + full suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_track_generator_facade.py -q`
Expected: PASS.

Then: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/track_generator.py tests/test_track_generator_facade.py
git commit -m "docs: make TrackGenerator deterministic-batch contract explicit + test"
```

---

### Task 7: Fix the explorer thickness-stat NaN poisoning (Important)

`_stats` computes the half-width median over ALL envs; a single `count==0` (degenerate) env NaN-pads index 0, and `torch.median` returns NaN if any element is NaN, collapsing `band` to 1 and corrupting `mean_thickness` for the whole batch — exactly on the low-yield configs the explorer exists to diagnose. Extract a small pure helper restricted to valid envs and test it directly.

**Files:**
- Modify: `viz/param_explorer.py` (`_stats`, line 293-294)
- Test: `tests/test_param_explorer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_param_explorer.py`:

```python
def test_estimate_half_width_ignores_invalid_nan_envs():
    import torch
    # env 1 is invalid: its index-0 row is NaN (count==0 -> fully NaN-padded). A median over
    # all envs would be NaN-poisoned; restricting to valid envs must yield a finite width.
    outer = torch.zeros(3, 4, 2)
    center = torch.zeros(3, 4, 2)
    outer[:, 0, 0] = torch.tensor([1.0, float("nan"), 1.0])   # half-width 1.0 for valid envs
    center[:, 0, 0] = 0.0
    valid = torch.tensor([True, False, True])
    hw = px._estimate_half_width(outer, center, valid)
    assert torch.isfinite(torch.tensor(hw))
    assert abs(hw - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py::test_estimate_half_width_ignores_invalid_nan_envs -v`
Expected: FAIL — `px._estimate_half_width` does not exist yet.

- [ ] **Step 3: Extract the helper and use valid envs**

In `viz/param_explorer.py`, add a module-level helper (place it just above `def _stats(track)` at line 281):

```python
def _estimate_half_width(outer, center, valid) -> float:
    """Median half-width over VALID envs only. Index 0 is real (count>=1) for valid envs;
    including invalid (count==0, fully NaN-padded) envs would NaN-poison the median."""
    import torch
    d = torch.linalg.norm(outer[valid, 0] - center[valid, 0], dim=-1)
    return float(d.median())
```

Then in `_stats`, replace lines 293-294:

```python
    # half-width from the first REAL point of each env (index 0 is always real / non-NaN).
    hw = float(torch.linalg.norm(t.outer[:, 0] - t.center[:, 0], dim=-1).median())
```

with:

```python
    # half-width from the first REAL point of each VALID env (invalid envs have count==0
    # and are fully NaN-padded, which would NaN-poison the median). n > 0 here, so non-empty.
    hw = _estimate_half_width(t.outer, t.center, valid)
```

(`valid` is already a boolean mask in `_stats` — `t.valid` is built with `.bool()`.)

- [ ] **Step 4: Run the test + full suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py -q`
Expected: PASS.

Then: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "fix: compute explorer half-width over valid envs to avoid NaN-poisoned thickness"
```

---

### Task 8 (optional cleanup): Cheap, low-risk suggestions

Bundle three safe Suggestions from the review. Skip if time-boxed to Critical/Important only.

**Files:**
- Modify: `track_gen/_src/generator_registry.py` (`register`, lines 41-43)
- Modify: `track_gen/_src/rng_utils.py` (remove dead `get_offset`, line ~58)
- Modify: `track_gen/_src/warp_generate_voronoi.py` (document `"ring"` layout, line ~70)
- Test: `tests/test_rng.py` (no new test needed for dead-code removal; rely on full suite)

- [ ] **Step 1: Cross-module duplicate detection in the centerline registry**

Mirror the gate registry. In `generator_registry.py`, replace `register`:

```python
def register(spec: GeneratorSpec) -> None:
    GENERATORS[spec.name] = spec
```

with:

```python
def register(spec: GeneratorSpec) -> None:
    existing = GENERATORS.get(spec.name)
    if existing is not None and getattr(existing.generate, "__module__", None) != getattr(
        spec.generate, "__module__", None
    ):
        raise ValueError(
            f"generator {spec.name!r} is already registered by "
            f"{getattr(existing.generate, '__module__', None)!r}")
    GENERATORS[spec.name] = spec
```

- [ ] **Step 2: Remove dead `get_offset`**

In `rng_utils.py`, delete the unused `get_offset` static method (verified no callers: `grep -rn "get_offset" track_gen tests viz` returns only its definition).

- [ ] **Step 3: Document the voronoi `"ring"` layout**

In `warp_generate_voronoi.py`, add a comment at the `_LAYOUT_RING` branch noting it is the uniform-box-fill baseline (only `"void_ring"` samples an annulus), so the name is not mistaken for a ring distribution.

- [ ] **Step 4: Run the full suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: green (reload-safety tests in `test_*registry*` still pass; the new dup guard only triggers on cross-module same-name).

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/generator_registry.py track_gen/_src/rng_utils.py track_gen/_src/warp_generate_voronoi.py
git commit -m "chore: registry dup detection, drop dead get_offset, clarify voronoi ring layout"
```

---

## Deferred (not in this plan)
These review findings are intentionally out of scope here — larger or lower-value — and are tracked in `docs/code-review-2026-06-27.md`:
- Quaternion non-uniform-over-SO(3) distribution (changing the distribution could break any future consumer; the unused path makes it low value).
- `warp_relax` `sep_cache_overflow` surfacing, host-sync consistency, separation-band upper clamp.
- `num_points_per_segment`/`min_point_distance` defensive in-kernel clamps (config validation in Task 1 already closes the reachable paths).
- Pipeline atomic-area label nondeterminism, `count=None` divisibility assert.
- Gate `min_gates` vs `min_num_points` feasibility, rng/env-count construction check.
- Duplicated `_normalize_centerline_k`, voronoi dead `cluster>5` guard, checkpoint clip-fallback alloc/generate flag assert.

---

## Self-review notes
- **Spec coverage:** Critical RNG stride → Task 2. Important: TrackGenConfig validation (4 findings) → Task 1; RNG quaternion → Task 5; RNG state buffer → Task 3; RNG int-bounds → Task 4; TrackGenerator determinism → Task 6; explorer thickness → Task 7; `num_points<=N_max` OOB → Task 1. Cheap suggestions → Task 8. All Critical+Important covered.
- **Placeholders:** none — every code step shows full old/new text. Task 4 flags one verify-before-edit (scalar helper names `uniform_single`/`normal_single`).
- **Consistency:** new symbols `_estimate_half_width` (Task 7) and the `2654435761` salt (Task 5) are used consistently; test API matches `PerEnvSeededRNG.sample_*_warp` signatures verified during planning.
- **Risk note:** Task 1 makes `num_points > N_max` (and other invalid configs) raise where they previously corrupted memory or failed deep — desired. Confirm no committed test/config relies on `num_points > N_max` (verified: none do).
