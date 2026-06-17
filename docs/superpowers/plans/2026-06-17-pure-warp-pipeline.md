# Pure-Warp End-to-End Track Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-express the whole track-gen pipeline (generation → resample → relax → inflate) in pure NVIDIA Warp kernels, captured as one end-to-end CUDA graph, runnable on Warp's CPU device (tests) and CUDA (prod), with torch only as the I/O container — then delete the torch compute.

**Architecture:** Each stage becomes a Warp kernel over flat arrays (`[E*N]` `vec2f`, `[E]` scalars; env index = `tid // N`). The existing torch functions are the **verification oracle** — every Warp kernel is asserted `allclose` against its torch counterpart on both `cpu` and `cuda` Warp devices before it replaces it. Generation's regen loop is made **static** (fixed `K` attempts, masked accept-first-valid) so the pipeline is graph-capturable. RNG uses Warp's built-in per-thread generator seeded per (env, attempt).

**Tech Stack:** NVIDIA Warp 1.14 (`wp.kernel`, `wp.from_torch`, `wp.rand_init`/`wp.randf`/`wp.randn`, `wp.HashGrid` not needed, `wp.capture_begin/end`), PyTorch (I/O container + oracle during port), pytest. Spec: `docs/superpowers/specs/2026-06-17-pure-warp-pipeline-design.md`. Reference Warp code: `track_gen/warp_relax.py` + `docs/superpowers/spikes/2026-06-17-warp-xpbd/`.

**Env:** `.venv/bin/python` (CUDA torch 2.6 + warp 1.14). GPU tests need CUDA; geometry/inflation kernels also run on the Warp **cpu** device so most tests work GPU-free. Run on the GPU **serially** (one process; the card is 16 GB and shared).

**Conventions for every Warp kernel:** define inside `if _HAVE_WARP:` guard in `track_gen/warp_pipeline.py`; one thread per output element; read torch tensors via `wp.from_torch(t.reshape(...).contiguous(), dtype=wp.vec2f|wp.float32|wp.int32)`; the public wrapper does `wp.init()` once, launches with `device=str(tensor.device)`, and returns torch views. Each wrapper takes/returns torch tensors so it's a drop-in for the torch oracle.

---

## Phase 0 — Scaffolding + Warp-CPU sanity

### Task 1: `warp_pipeline.py` skeleton + Warp-CPU smoke test

**Files:** Create `track_gen/warp_pipeline.py`; Test `tests/test_warp_pipeline_smoke.py`.

- [ ] **Step 1: Failing test** — `tests/test_warp_pipeline_smoke.py`:
```python
import pytest, torch
pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl


def test_warp_runs_on_cpu_device():
    # The pure-Warp pipeline must run on the Warp CPU device so CI works GPU-free.
    out = wpl._smoke_double(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_warp_runs_on_cuda_device():
    out = wpl._smoke_double(torch.tensor([1.0, 2.0, 3.0], device="cuda"))
    assert torch.allclose(out.cpu(), torch.tensor([2.0, 4.0, 6.0]))
```
- [ ] **Step 2: Run, expect fail** — `.venv/bin/python -m pytest tests/test_warp_pipeline_smoke.py -q` → ImportError/AttributeError.
- [ ] **Step 3: Implement** — `track_gen/warp_pipeline.py`:
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-Warp track-generation pipeline kernels (run on Warp cpu + cuda)."""
from __future__ import annotations
import torch
try:
    import warp as wp
    _HAVE_WARP = True
except Exception:
    _HAVE_WARP = False
_INITED = False

def _init():
    global _INITED
    if not _INITED:
        wp.init(); _INITED = True

if _HAVE_WARP:
    @wp.kernel
    def _double_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = 2.0 * x[i]

def _smoke_double(x: torch.Tensor) -> torch.Tensor:
    _init()
    out = torch.empty_like(x)
    wp.launch(_double_k, dim=x.shape[0],
              inputs=[wp.from_torch(x.contiguous(), dtype=wp.float32),
                      wp.from_torch(out, dtype=wp.float32)],
              device=str(x.device))
    (wp.synchronize() if "cuda" in str(x.device) else None)
    return out
```
- [ ] **Step 4: Run, expect pass** (both cpu + cuda tests).
- [ ] **Step 5: Commit** — `git add track_gen/warp_pipeline.py tests/test_warp_pipeline_smoke.py && git commit -m "warp_pipeline: scaffolding + warp cpu/cuda smoke"`.

---

## Phase 1 — Inflation/geometry kernels (clean torch oracles, no RNG)

> Each task ports one torch function to a Warp kernel and asserts `allclose` to the torch oracle on cpu (and cuda when available). Helper for tests: `_dev_params = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])`, parametrize.

### Task 2: `resample_uniform` kernel

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_resample.py`.

Oracle: `track_gen.relaxation._resample_uniform(center[E,N,2], n)` (already batched torch). Port the per-bead arc-length lookup into a kernel: precompute per-env cumulative arc length `s` and segment lengths in a first kernel (or in torch as the oracle does, then a kernel for the lookup). Keep N fixed (input N == output N for the relax path).

- [ ] **Step 1: Failing test** — random closed loops, assert kernel `allclose` to `_resample_uniform` (atol 1e-5) on each device:
```python
import math, pytest, torch
pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from track_gen.relaxation import _resample_uniform
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.parametrize("dev", DEVS)
def test_resample_matches_torch(dev):
    torch.manual_seed(0)
    c = (torch.randn(12, 200, 2) * torch.linspace(0.3, 2.0, 12).view(12,1,1)).to(dev)
    got = wpl.resample_uniform(c, 200)
    ref = _resample_uniform(c, 200)
    assert torch.allclose(got, ref, atol=1e-4)
```
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** the kernel + `resample_uniform(center, n)` wrapper. Per env e, per output point k: target arc length `tk = k * total_e / n`; binary-search the cumulative-length array for the bracketing segment; lerp. Compute per-env segment lengths + cumulative `s` either in a small prefix kernel or reuse torch for the prefix-sum and a kernel only for the lookup (prefix-sum stays as the oracle's torch until Task 13 fuses it). Provide the full kernel + wrapper code in-task, mirroring `_resample_uniform`'s math (closed loop: seg i = x[i+1]-x[i] wrapping, targets `arange(n)*total/n`, clamp segment index to N-1).
- [ ] **Step 4: Run, expect pass (cpu+cuda).**
- [ ] **Step 5: Commit** — `"warp_pipeline: resample_uniform kernel (allclose torch)"`.

### Task 3: `frame_curvature` kernel (tangent, left-normal, Menger curvature)

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_frame.py`.
Oracle: `geometry.tangents_normals(center)` + `geometry.menger_curvature(center)`.

- [ ] **Step 1: Failing test** — circle radius r: assert `||T||=1`, `T·N=0`, curvature ≈ 1/r, and `allclose` to the torch oracles on a random loop (cpu+cuda).
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** `frame_curvature_k` (per point i: `T = normalize(x[i+1]-x[i-1])`, `N=(-T.y,T.x)`; Menger kappa from triple (i-1,i,i+1) = `4*area/(|a||b||c|)` clamp 1e-12) + wrapper `frame_curvature(center) -> (T, Nrm, kappa)`. Full code in-task from the geometry oracle.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** — `"warp_pipeline: frame+curvature kernel"`.

### Task 4: `offset` kernel (constant-width borders + outer/inner by area)

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_offset.py`.
Oracle: `inflation._offset_stage(center, Nrm, w)` with constant `w=half_width`.

- [ ] **Step 1: Failing test** — circle: outer area > center area > inner; `allclose` to `_offset_stage` (feed the same `Nrm` from Task 3) on cpu+cuda.
- [ ] **Step 2–4:** Implement `offset_k` (a = center + w*N, b = center - w*N) + a per-env shoelace-area reduction kernel (or reuse `geometry.polygon_area` torch as oracle and a kernel for the assign) to pick outer=larger-|area|. Wrapper `offset(center, Nrm, half_width) -> (outer, inner)`. Verify pass.
- [ ] **Step 5: Commit** — `"warp_pipeline: constant-width offset kernel"`.

### Task 5: `self_intersections` + `thickness` kernels

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_geom_gates.py`.
Oracle: `geometry.self_intersections`, `geometry.thickness`, `geometry.separation_min`, `geometry.curvature_radius_min`.

- [ ] **Step 1: Failing test** — figure-eight >0 crossings, circle ==0; thickness of a circle ≈ radius (large band); `allclose`/equal to torch oracles (cpu+cuda).
- [ ] **Step 2–4:** Implement `self_intersections_k` (per env: double loop over edges, proper-crossing orientation test, exclude circ-index ≤1, count once) returning `[E]` counts; `separation_min_k` / curvature-radius reduction → `thickness`. Wrappers match the torch signatures. Verify.
- [ ] **Step 5: Commit** — `"warp_pipeline: self_intersections + thickness kernels"`.

### Task 6: `validity` kernel

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_validity.py`.
Oracle: `inflation._validity_stage(center, w, count, gen_valid, config, outer, inner)`.

- [ ] **Step 1: Failing test** — clean circle valid; figure-eight invalid; a folded-border ellipse invalid; equal to the torch oracle on the same inputs (cpu+cuda).
- [ ] **Step 2–4:** Implement `validity` combining: gen_valid AND turning≈2π AND w>w_floor AND no-NaN AND thickness≥(1-tol)·hw AND border self-intersections==0 (reuse Task 5 kernels + a turning-number kernel). Verify.
- [ ] **Step 5: Commit** — `"warp_pipeline: validity kernel"`.

### Task 7: `inflate_warp(center, config) -> Track`

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_inflate.py`.

- [ ] **Step 1: Failing test** — on circle/ellipse synthetic centerlines, `inflate_warp` produces a `Track` whose fields are `allclose` to `inflation.inflate` (same `Track` dataclass), valid flags equal (cpu+cuda).
- [ ] **Step 2–4:** Compose Tasks 2–6 + an arclength/length kernel into `inflate_warp` returning the existing `types.Track`. Verify against `inflation.inflate`.
- [ ] **Step 5: Commit** — `"warp_pipeline: inflate_warp == torch inflate"`.

---

## Phase 2 — Generation kernels

### Task 8: Warp RNG + `corner_sample` kernel

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_corner_sample.py`.

Design note (redesign, RNG break accepted): per (env e, attempt k), seed `wp.rand_init(base_seed[e] * 9781 + k)`; sample `max_num_points` corner positions in the scaled box. To preserve diversity + rough min-spacing without the grid-topk trick, sample on the `num_cells` grid: pick a random cell per corner with rejection of duplicates (bounded retries), then add per-corner uniform noise in [-0.5,0.5) and scale — matching `_sample_corner_points`'s coordinate construction, only the cell-selection differs.

- [ ] **Step 1: Failing test** — shape `[E, max_num_points, 2]`; finite; within the scaled box; reproducible for a fixed seed; two different env seeds differ (cpu+cuda).
- [ ] **Step 2–4:** Implement `corner_sample_k` + wrapper `corner_sample(seeds, attempt, config) -> corners[E,P,2]`. Verify the structural properties (not bit-equality vs torch — RNG differs by design).
- [ ] **Step 5: Commit** — `"warp_pipeline: corner_sample kernel (warp RNG)"`.

### Task 9: `ccw_sort` kernel

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_ccw_sort.py`.
Oracle: `geometry.ccw_sort` (orders points by `atan2(dx,dy)` around centroid).

- [ ] **Step 1: Failing test** — feed a fixed scrambled point set, assert the Warp `ccw_sort` returns the SAME ordering as `geometry.ccw_sort` (cpu+cuda).
- [ ] **Step 2–4:** Implement `ccw_sort_k`: one thread per env, compute centroid, key = `atan2(dx,dy)` per corner, insertion-sort the ≤P corners by key (stable, matching torch argsort), write reordered corners. Verify exact ordering match.
- [ ] **Step 5: Commit** — `"warp_pipeline: ccw_sort kernel"`.

### Task 10: `assemble` kernel (vertex tangents + cubic Bézier)

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_assemble.py`.
Oracle: `generators.vertex_tangents` + `_segment`/`_cubic_bezier` (dense centerline from corners).

- [ ] **Step 1: Failing test** — feed fixed corners (+ count for NaN-pruning), assert the Warp dense centerline `allclose` to the torch assemble (handling NaN-pruned tails identically) on cpu+cuda.
- [ ] **Step 2–4:** Implement `assemble_k`: per dense sample (env, segment i, sample s): blend unit tangents `t = normalize(p*u_out + (1-p)*u_in)` at the two corners, handles at `rad*chord`, eval cubic Bernstein at `s/(npseg-1)`. NaN-propagate for pruned corners. Wrapper `assemble(corners, count, config) -> dense[E,M,2]`. Verify.
- [ ] **Step 5: Commit** — `"warp_pipeline: assemble (tangents+cubic) kernel"`.

### Task 11: `gates` kernel (angle, turning, simplicity)

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_gates.py`.
Oracle: `generators._corner_angles`/min-angle test, `geometry.turning_number`, simplicity via Task 5 `self_intersections` on a 256-resample.

- [ ] **Step 1: Failing test** — fixed dense candidates: assert the Warp per-env `accept` mask equals the torch gate result (angle_ok AND turn_ok AND finite_ok AND simple_ok) on cpu+cuda.
- [ ] **Step 2–4:** Implement `gates_k` returning `[E]` bool accept, reusing turning + self_intersections (on a resampled-256 loop via Task 2). Verify equal to the torch conjunction.
- [ ] **Step 5: Commit** — `"warp_pipeline: gates kernel"`.

### Task 12: static regen → `generate_centerline_warp(seeds, config)`

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_generate.py`.

- [ ] **Step 1: Failing test** — `generate_centerline_warp(seeds[E], config)` returns `(centerline[E,M,2], valid[E])`; valid yield ≥ 0.95 at E=256 scale=1; every valid centerline is simple (256-resample self_intersections==0); reproducible (cpu+cuda).
- [ ] **Step 2–4:** Implement the static regen driver: for `k in range(max_regen_iters)`: corner_sample(attempt k) → ccw_sort → assemble → gates; for envs not yet accepted, store the candidate + mark accepted (masked accept-first-valid). All fixed-iteration, no host branching on data. Verify yield matches the torch generator (~100%).
- [ ] **Step 5: Commit** — `"warp_pipeline: static-regen generate_centerline_warp"`.

---

## Phase 3 — Assembly, graph capture, integration

### Task 13: `generate_tracks_warp(config, seeds) -> Track` (no graph yet)

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_pipeline_e2e.py`.

- [ ] **Step 1: Failing test** — end-to-end Warp pipeline (gen → resample → relax → inflate) at E=64, half_width=0.03: validity yield ≥ 0.9 and constant width, matching the torch pipeline's yield within a few % (cpu+cuda).
- [ ] **Step 2–4:** Compose `generate_centerline_warp` → `resample_uniform` → `warp_relax.xpbd_solve` → `inflate_warp`. Verify yield/shape vs the torch `TrackGenerator.generate`.
- [ ] **Step 5: Commit** — `"warp_pipeline: end-to-end generate_tracks_warp"`.

### Task 14: End-to-end CUDA graph capture

**Files:** Modify `track_gen/warp_pipeline.py`; Test `tests/test_warp_graph.py` (cuda-only).

- [ ] **Step 1: Failing test** (cuda-only) — capture the pipeline once into a graph; replaying it with new seeds copied into fixed buffers yields the SAME `Track` as the non-graph path (allclose); timing recorded.
- [ ] **Step 2–4:** Restructure `generate_tracks_warp` to use fixed pre-allocated buffers (seeds, corners, dense, center, disp, Track fields) and `wp.capture_begin/end` over the static-regen + resample + relax + inflate region. Per call: copy seeds/params into buffers, `wp.capture_launch`, read Track. Verify replay==non-graph + record E=8192 time.
- [ ] **Step 5: Commit** — `"warp_pipeline: end-to-end CUDA graph capture"`.

### Task 15: Facade integration, test migration, remove torch compute

**Files:** Modify `track_gen/track_generator.py`, `track_gen/__init__.py`, `benchmarks/benchmark_relaxation.py`; remove/trim `geometry.py`/`generators.py`/`inflation.py` torch compute; update tests.

- [ ] **Step 1:** Point `TrackGenerator.generate` at `warp_pipeline.generate_tracks_warp`; keep the same `Track` return + public API.
- [ ] **Step 2:** Migrate the test suite: tests that exercised torch stages now target the Warp pipeline (the per-kernel oracle tests from Phases 1–2 already pin equivalence). Keep the geometry primitive tests only for primitives still used.
- [ ] **Step 3:** Remove the torch compute pipeline (the `_*_stage` functions in `inflation.py`, the torch generation internals in `generators.py`, torch-only geometry no longer referenced) — torch remains only as the array container at the boundary. Run full suite (cpu) + cuda tests.
- [ ] **Step 4:** Update `benchmarks/benchmark_relaxation.py` (or add `benchmarks/benchmark_pipeline.py`) to drive the Warp pipeline end-to-end at E=8192 (gen+relax+inflate), GPU + CPU, reporting validity / wall-clock / peak memory.
- [ ] **Step 5: Commit** — `"Pure-Warp pipeline is the implementation; remove torch compute"`.

---

## Self-Review

**Spec coverage:** gen kernels → Tasks 8–12; resample → Task 2; relax → reused (done); inflate kernels → Tasks 3–7; static regen → Task 12; end-to-end + graph → Tasks 13–14; warp-cpu test path → Task 1 + per-task cpu params; torch-as-oracle-then-remove → Phases 1–2 oracle asserts + Task 15 removal; RNG (warp built-in) → Task 8. All spec §3.1 kernels + §4 verification + §7 done-criteria mapped.

**Placeholder scan:** Tasks 2–11 give the oracle + the verification test + the kernel algorithm; the genuinely mechanical kernels (resample/frame/offset/self-intersection/validity) are fully specified by their torch oracles (which the executor reads and ports line-for-line, asserting allclose) — the verification test is the exact contract, so "port function X to a kernel, assert allclose" is a complete, non-ambiguous instruction here even where the literal Warp body is written during the task. The redesigned corner sampling (Task 8) is the one stage validated by properties (yield/diversity), not bit-equality, and says so explicitly. No TBD/TODO.

**Type/name consistency:** wrappers mirror the torch oracle signatures and return the existing `types.Track`; kernel naming `*_k`, wrappers without suffix; `generate_centerline_warp`/`generate_tracks_warp`/`inflate_warp` used consistently across Tasks 7/12/13/15; flat `[E*N]`/`[E]` layout + `tid//N` convention stated once in the header and reused.

**Note on granularity:** several tasks bundle Steps 2–4 (the per-kernel write/run/verify loop) because the verification test from Step 1 is the precise gate; the executor writes the kernel by porting the named torch oracle until the allclose test passes. This is intentional for a port whose oracle fully defines correctness.
