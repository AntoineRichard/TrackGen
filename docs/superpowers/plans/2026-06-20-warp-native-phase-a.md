# Warp-native Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the shipped `track_gen` runtime fully Warp-native (zero torch in the import graph, public API, and RNG), with persistent in-place output buffers — the eager (CPU-testable) half of the spec. The CUDA-graph auto-capture is Phase B.

**Architecture:** Phase A is a handful of large, interdependent coherent changes, so it is decomposed into **three milestones**, each ending green on the Warp `cpu` device. This document fully details **Milestone 1 (RNG de-torch)** — the most independent, lowest-risk piece. **Milestones 2 and 3 are scoped below and will be detailed just-in-time** (the in-place buffer-ownership restructure in M2 is best planned against the post-M1 code).

**Tech Stack:** Python ≥ 3.10, NVIDIA Warp (`warp-lang`), numpy, pytest. torch becomes a dev-only dep (M3).

## Global Constraints (from the spec — bind every task)

- **GPU-first buffers:** generator pre-allocates output + scratch `wp.array`s once; `generate()` writes **in place**; **stable pointers**; returns the **same `Track`** instance; **zero per-call allocation**. (M2)
- **No readback in `track_gen/`:** no `wp.to_numpy`/`.numpy()`/host copies on any hot path. Boundary serves `wp.array`; consumers convert.
- **numpy is init-only host glue**, owns no real op; the numpy-RNG path is removed.
- **Public API** (after Phase A) = `TrackGenerator`, `TrackGenConfig`, `Track`, `PerEnvSeededRNG`, `__version__`. (M2)
- **Oracle stays torch** (`tests/_oracle/`); tests `wp.to_torch` the pipeline output before comparing. (M2)
- **Verification gate:** `.venv/bin/python -m pytest -q` green on the Warp `cpu` device after every milestone.
- **GPG signing fails in this env** → every commit uses `--no-gpg-sign`.

## Milestone overview

- **M1 — RNG de-torch (this doc, detailed).** `rng_utils.py` becomes torch-free: drop all
  `*_torch` methods, the host numpy-RNG path, the `*_numpy` samplers, and `sample_unique_integers*`
  (unused, numpy-RNG-backed). Keep the `*_warp` samplers + `seeds_warp`/`states_warp`/`set_seeds_warp`;
  numpy survives only in `__init__`. Update the consumers (oracle, Fourier, 7 `set_seeds` callers).
  torch is still a dep (the pipeline uses it) — M1 just shrinks the surface and frees `rng_utils`.
- **M2 — runtime de-torch + buffer ownership + API (detailed after M1).** `Track`→`wp.array`;
  `TrackGenerator` owns pre-allocated buffers, `generate()` writes in place and returns the same
  `Track`; remove free `generate_tracks_warp`/`_graph` + `CapturedTracks`, fold stages into private
  helpers; swap all `torch.empty/full/zeros`→`wp.array`, `torch.where`→a wp select kernel, delete
  dead `_mean_seg_len_torch`; migrate ~40 test files to the `wp.to_torch` boundary; add the
  stable-`.ptr` test. Gate: `pytest -q` green on `cpu`.
- **M3 — deps + torch-free guards (detailed after M2).** `pyproject`: core = `["numpy","warp-lang"]`,
  `dev += ["scipy","torch"]`; guard tests: `import track_gen` is torch-free (subprocess), numpy
  appears only in `__init__`, no `wp.to_numpy` in `track_gen/`. Gate: `pytest -q` green; import works
  with torch absent.

---

## Milestone 1 — RNG de-torch

`PerEnvSeededRNG` (`track_gen/_src/rng_utils.py`) currently exposes `_warp`/`_torch`/`_numpy`
variants for 7 distributions plus seeds/states/set_seeds in three flavors and a host numpy-RNG
(`np.random.default_rng`). The Warp pipeline uses only the `_warp` samplers. M1 strips everything
else and updates the (dev-side) consumers to convert via `wp.to_torch`.

### Task 1: Confirm the to-delete surface is unused outside rng_utils

**Files:** none (read-only verification that drives Task 2's deletions).

**Interfaces:**
- Produces: a confirmed list of methods safe to delete (no consumers) vs. methods with consumers that Task 3 must rewrite.

- [ ] **Step 1: Confirm `sample_unique_integers*` has no consumers**

Run:
```bash
cd /home/antoiner/Documents/TrackGen
grep -rn "sample_unique_integers" track_gen/ tests/ benchmarks/ viz/ | grep -v "rng_utils.py"
```
Expected: **no output** (only the definitions in `rng_utils.py` exist). If any consumer appears, STOP and report — the deletion plan must account for it.

- [ ] **Step 2: Confirm the numpy-RNG internals have no external consumers**

Run:
```bash
grep -rn "initialize_numpy_rng\|set_numpy_rng_seeds\|_numpy_rngs\|sample_[a-z_]*_numpy\|seeds_numpy\|states_numpy" track_gen/ tests/ benchmarks/ viz/ | grep -v "rng_utils.py"
```
Expected: **no output**. If any appear, STOP and report.

- [ ] **Step 3: Record the consumer list for Task 3 (must match)**

Run:
```bash
grep -rn "sample_[a-z_]*_torch\|seeds_torch\|states_torch\|\.set_seeds(" track_gen/ tests/ benchmarks/ viz/ | grep -v "rng_utils.py"
```
Expected exactly these consumers (Task 3 rewrites all of them):
- `track_gen/_experimental/fourier.py` — `sample_normal_torch` ×2
- `tests/_oracle/generators.py` — `sample_uniform_torch` ×2, `sample_integer_torch` ×1
- `tests/test_warp_pipeline_e2e.py`, `tests/test_generator_simplicity_gate.py`,
  `tests/test_end_to_end_relaxation.py`, `tests/test_generators.py` (×2),
  `benchmarks/benchmark_relaxation.py`, `viz/plot_tracks.py`, `viz/plot_ablations.py` (×2) — `.set_seeds(`

### Task 2: Strip `rng_utils.py` to the Warp-native surface

**Files:**
- Modify: `track_gen/_src/rng_utils.py`

**Interfaces:**
- Consumes: Task 1's confirmation.
- Produces: a torch-free `PerEnvSeededRNG` exposing **only**: `seeds_warp`, `states_warp` (properties),
  `set_seeds_warp(seeds: wp.array, ids: wp.array | None)`, and the `_warp` samplers
  `sample_uniform_warp` / `sample_sign_warp` / `sample_integer_warp` / `sample_normal_warp` /
  `sample_poisson_warp` / `sample_quaternion_warp`. `__init__` keeps its numpy use
  (`wp.array(np.arange(...))` / `wp.array(np.ones(...)*seeds)`) — numpy is init-only.

- [ ] **Step 1: Remove `from torch import …` / `import torch`**

Delete the torch import at the top of `rng_utils.py` (it is only used by the methods removed below).

- [ ] **Step 2: Delete the torch-facing methods**

Remove these method/property defs in full: `seeds_torch`, `states_torch`, `set_seeds`,
`sample_uniform_torch`, `sample_sign_torch`, `sample_integer_torch`, `sample_normal_torch`,
`sample_poisson_torch`, `sample_quaternion_torch`, `sample_unique_integers_torch`.

- [ ] **Step 3: Delete the host numpy-RNG path + numpy samplers**

Remove in full: `initialize_numpy_rng`, `set_numpy_rng_seeds`, `set_seeds_numpy`,
`seeds_numpy`, `states_numpy`, and every `sample_*_numpy`
(`sample_uniform_numpy`, `sample_sign_numpy`, `sample_integer_numpy`, `sample_normal_numpy`,
`sample_poisson_numpy`, `sample_quaternion_numpy`, `sample_unique_integers_numpy`) and
`sample_unique_integers_warp` (it is numpy-RNG-backed and unused). Delete the now-dead
`self._use_numpy_rng` / `self._numpy_rng_is_initialized` flags from `__init__` and any
`if self._use_numpy_rng:` branch in `set_seeds_warp`.

- [ ] **Step 4: Verify no torch/numpy-RNG residue remains**

Run:
```bash
cd /home/antoiner/Documents/TrackGen
grep -nE "torch|_numpy_rng|np\.random|sample_[a-z_]*_(torch|numpy)" track_gen/_src/rng_utils.py
```
Expected: **no output** (numpy survives only as `np.arange`/`np.ones` in `__init__`, which this
pattern does not match; if `np.arange`/`np.ones` lines appear, that's fine — re-run with the exact
pattern above which excludes them). Then:
```bash
.venv/bin/python -c "import warp as wp; wp.init(); from track_gen._src.rng_utils import PerEnvSeededRNG; import torch, sys; import track_gen._src.rng_utils as m; assert 'torch' not in [getattr(o,'__name__',None) for o in vars(m).values()], 'torch leaked into rng_utils'; print('rng_utils torch-free OK')"
```
Expected: `rng_utils torch-free OK`.

### Task 3: Update the RNG consumers (must land with Task 2 for green)

**Files:**
- Modify: `track_gen/_experimental/fourier.py`, `tests/_oracle/generators.py`,
  `tests/test_warp_pipeline_e2e.py`, `tests/test_generator_simplicity_gate.py`,
  `tests/test_end_to_end_relaxation.py`, `tests/test_generators.py`,
  `benchmarks/benchmark_relaxation.py`, `viz/plot_tracks.py`, `viz/plot_ablations.py`

**Interfaces:**
- Consumes: the Warp-native RNG surface from Task 2.

- [ ] **Step 1: Sampler calls → `wp.to_torch(*_warp(...))`**

In `track_gen/_experimental/fourier.py` (a torch module — it may use torch freely):
```python
import warp as wp
# was: a = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)
a = wp.to_torch(self.rng.sample_normal_warp(0.0, 1.0, (self.K, 2), ids=ids))
b = wp.to_torch(self.rng.sample_normal_warp(0.0, 1.0, (self.K, 2), ids=ids))
```
In `tests/_oracle/generators.py`:
```python
import warp as wp
# u = self.rng.sample_uniform_torch(0.0, 1.0, (n,), ids=ids)
u = wp.to_torch(self.rng.sample_uniform_warp(0.0, 1.0, (n,), ids=ids))
# noise = self.rng.sample_uniform_torch(-0.5, 0.5, (self.config.max_num_points, 2), ids=ids)
noise = wp.to_torch(self.rng.sample_uniform_warp(-0.5, 0.5, (self.config.max_num_points, 2), ids=ids))
# count = self.rng.sample_integer_torch(self.config.min_num_points, self.config.max_num_points + 1, (1,), ids=ids)
count = wp.to_torch(self.rng.sample_integer_warp(self.config.min_num_points, self.config.max_num_points + 1, (1,), ids=ids))
```
(These produce identical values — the `_torch` samplers were `wp.to_torch` of the `_warp` ones.)

- [ ] **Step 2: `rng.set_seeds(torch_seeds, ids=torch_ids)` → `rng.set_seeds_warp(wp_seeds, ids=wp_ids)`**

In each of the 7 `set_seeds` call sites, build the seeds/ids as `wp.array(..., dtype=wp.int32)`
(or `wp.from_torch` an existing int32 tensor) and call `set_seeds_warp`. Example
(`tests/test_end_to_end_relaxation.py:15`):
```python
import warp as wp
# was: rng.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
rng.set_seeds_warp(wp.from_torch(seeds.to(torch.int32)),
                   ids=wp.array(list(range(E)), dtype=wp.int32, device=str(seeds.device)))
```
Apply the analogous change at the other six sites (the `ids=` they pass is always
`torch.arange(...)`/an int32 id tensor → `wp.array(range, dtype=wp.int32)` or
`wp.from_torch(ids.to(torch.int32))`). Tests/benchmarks/viz are dev-side and may import torch.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, same count as the pre-M1 baseline (no tests added/removed in M1 — only RNG-call
rewrites; values are identical so all oracle-parity assertions still hold).

- [ ] **Step 4: Commit**

```bash
cd /home/antoiner/Documents/TrackGen
git add -A
git status   # confirm only rng_utils + the 9 consumer files changed
git commit --no-gpg-sign -m "refactor(rng): strip PerEnvSeededRNG to the Warp-native surface; drop torch + numpy-RNG paths"
```

---

## Self-Review

- **Spec coverage (M1 slice):** spec §4 (RNG de-torch + numpy-RNG removal) → Tasks 1–3. The broader
  spec (§1 Track, §2 alloc, §3 ops, §5 generator/graph, §6 deps, §7 tests, §8 fourier, §9 facade)
  is M2/M3 — explicitly deferred and scoped in the milestone overview, to be detailed just-in-time.
- **Placeholders:** none in M1; M2/M3 are intentionally overview-level (will be separate detailed plans).
- **Consistency:** the kept RNG surface in Task 2's Interfaces matches the calls Task 3 makes
  (`sample_*_warp`, `set_seeds_warp`). `wp.to_torch` is used only on the dev/test/experimental side.

## Note on baseline

Before Task 1, capture the green baseline: `.venv/bin/python -m pytest -q` (record the pass count).
M1 must match it exactly.
