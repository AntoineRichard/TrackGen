# Warp-native runtime: zero torch on the hot path

**Date:** 2026-06-20
**Status:** Design — pending spec review
**Builds on:** the `chore/warp-core-dependency` work (warp-lang is a core dep; `__init__` eager; `_HAVE_WARP` guards purged). That must be merged first.

## Goal

Make the **shipped runtime** of `track_gen` fully NVIDIA-Warp-native — **zero torch** in the
import graph of `track_gen/_src`, the public API, and the RNG. Torch becomes a **dev/test
dependency** (the torch oracle and the experimental Fourier generator keep using it). The
Warp kernels are already torch-agnostic; torch today is only the transport/boundary layer
(buffer allocations, the `Track` type, the RNG's torch methods, and the CUDA-graph capture).

This is the endgame of the original "only Warp on the hot path" goal: the hot path becomes
Warp end-to-end, with consumers converting to torch themselves via `wp.to_torch` (zero-copy,
same-device) at the boundary.

## Core principles (GPU-first, non-negotiable)

1. **Pre-allocate once; stable pointers; update in place.** The generator owns its output
   buffers (the `Track` `wp.array`s). They are allocated **once** at construction (sized
   `[E, N_max]`) and the **same device pointers persist for the object's lifetime**. Each
   `generate()`/`replay()` writes results into those buffers **in place** — it never
   re-allocates and never swaps the pointer. A consumer therefore calls `wp.to_torch(track.center)`
   **once**; because the torch view shares the underlying memory, it keeps reflecting every
   subsequent run. Intermediate per-stage scratch is likewise pre-allocated once and reused —
   **zero allocation in the per-call hot path**.
   - *API consequence:* outputs are owned by a **stateful object** (`TrackGenerator`,
     `CapturedTracks`); `generate()`/`replay()` return the **same** `Track` instance each call.
2. **No readback in our code.** `track_gen` **serves `wp.array`** and stops there — consumers
   convert to torch/numpy/whatever on their side. **No `wp.to_numpy`** anywhere in shipped
   `track_gen/` (only with an explicit, justified exception). No `.numpy()`, no host copies on
   the hot path.
3. **numpy is init-only glue, owns no real operation.** numpy may appear **only** in one-time
   construction (e.g. `wp.array(np.arange(num_envs))` to seed an index/seed buffer), executed
   once per object — never inside `generate()`/`replay()` and never as a compute step. The
   vestigial host numpy-RNG path is removed entirely (see §4).

## Non-goals

- **Not** removing torch from the repo. The torch oracle (`tests/_oracle/`) stays torch —
  it is the validation baseline. `track_gen/_experimental/fourier.py` stays torch (private,
  unsupported, **left where it is**). Both rely on torch as a **dev** dependency.
- **Not** changing pipeline numerics, kernel logic, or generation behavior.
- **Not** porting Fourier to Warp.

## Current state (from the audit)

- Warp kernels: already torch-free.
- Torch coupling (≈130 sites: 98 trivial / 17 moderate / 14 hard) is all boundary/transport:
  - ~25–30 buffer allocations (`torch.empty/full/zeros`) handed to `wp.from_torch`.
  - A few real ops: `torch.where`×3, and the **dead** `_mean_seg_len_torch` (`roll`/`linalg.norm`).
  - `Track` dataclass fields are `torch.Tensor` (the public output type).
  - `PerEnvSeededRNG` exposes torch methods (`seeds_torch`, `sample_*_torch`) used by the
    oracle + Fourier; the **pipeline already uses the `sample_*_warp` path**.
  - `generate_tracks_warp_graph` captures with `torch.cuda.CUDAGraph` + `wp.ScopedStream`
    **because torch glue is interleaved** with Warp launches on a shared stream.

## Target design

### 1. `Track` → `wp.array`
`track_gen/_src/types.py`: the `Track` fields (`outer/center/inner/tangent/normal` `[E,N,2]`,
`arclen` `[E,N]`, `length/valid/count` `[E]`) change from `torch.Tensor` to `wp.array`
(device-resident: `wp.vec2f` for the `[…,2]` fields, `wp.float32`/`wp.int32`/`wp.bool` for the
rest). `TrackGenConfig` is **unchanged** (pure scalars, no tensors, no torch import).
Consumers recover torch with `wp.to_torch(track.center)` (zero-copy on the same device).
`types.py` drops `from torch import Tensor`.

These buffers are **owned and pre-allocated by the generator** (Principle 1): the `Track`
returned by `generate()` is the same instance every call, its `wp.array`s written in place.
`Track` is a plain holder of persistent `wp.array`s — no per-call construction.

### 2. Allocation layer → Warp (pre-allocated once)
Every `torch.empty/full/zeros/empty_like` buffer in `warp_pipeline.py` / `warp_relax.py`
becomes a `wp.array` (the `torch.empty(...)` + `wp.from_torch` pair collapses to a direct
device `wp.array`). **All such buffers — outputs AND per-stage scratch, including the per-track
`count` array (device, kernel-consumed) — are pre-allocated once by the owning object and reused
in place** (Principle 1); none are allocated per `generate()` call. `torch.int/float/bool` dtype
constants become `wp.int32`/`wp.float32`/`wp.bool`; `torch.cuda.synchronize` → `wp.synchronize`.
numpy is **not** used here (device data is `wp.array`); it appears only in one-time index/seed
setup (§4). This is the bulk: trivial-but-high-volume (~25–30 sites) plus the lift-allocations-
to-construction restructure.

### 3. Real torch ops → Warp / numpy / deletion
- `_mean_seg_len_torch` (dead on the Warp path) → **delete** (and the `import torch` it forces).
- `torch.where(...)` selects (the Fix-B polygon-fallback select, ×3) → a small Warp select
  kernel or a numpy host select on the already-computed arrays.
- Scattered `.long()`/`.bool()`/`.view()`/`.contiguous()` → `wp` dtype/shape equivalents or
  `wp.to_numpy(...).astype(...)` for host-side counts.

### 4. `PerEnvSeededRNG` → torch-free, warp-only sampling
`rng_utils.py` drops `from torch import …` and the torch surface (`seeds_torch`,
`states_torch`, `sample_*_torch`, the torch branch of `set_seeds`) **and the vestigial host
numpy-RNG path** (`_numpy_rngs`, `np.random.default_rng`, `initialize_numpy_rng`,
`sample_*_numpy`, the numpy branch of `set_seeds`). What remains: the **warp-native** samplers
(`sample_*_warp`) the pipeline already uses, plus `seeds_warp`/`states_warp`. numpy survives in
this file **only** as one-time construction glue — `wp.array(np.arange(num_envs))` /
`wp.array(np.ones(num_envs)*seeds)` in `__init__`, executed once (Principle 3). **Callers that
needed torch samples** — the oracle generators (`tests/_oracle/generators.py`) and
`_experimental/fourier.py` — switch to `wp.to_torch(rng.sample_*_warp(...))` at their (dev-side)
call sites; the values are identical (the torch samplers were already `wp.to_torch` of the warp
ones). The plan verifies the warp samplers exist for every sampler the oracle/Fourier needs.

### 5. One stateful `TrackGenerator` with automatic graph capture
The eager and graph paths **unify** into the single stateful generator. The free
`generate_tracks_warp` / `generate_tracks_warp_graph` functions and the `CapturedTracks` class
are **removed** (folded into `TrackGenerator` as private helpers).

`TrackGenerator(config, rng)` pre-allocates, once: the output `Track` `wp.array`s, all per-stage
scratch, and a static `wp.array` **seed buffer**. `self._graph = None`.

`generate()` (no batch-size arg — operates on the fixed configured batch `E = config.num_envs`):
1. Refresh the seed buffer **in place** from `rng` (current per-env seeds).
2. **CUDA + `self._graph is None`:** warm up (run the pipeline a few times on a side stream),
   then capture it with Warp-native capture (`wp.ScopedCapture(device)` → `wp.Graph` stored on
   `self._graph`), then launch it. (No `torch.cuda.CUDAGraph`, no stream-routing — the pipeline
   is pure Warp now.)
3. **CUDA + graph exists:** `wp.capture_launch(self._graph)` — pure replay, all buffers updated
   in place.
4. **Warp `cpu` device (no CUDA graphs):** run the kernels eagerly in place (no capture).
5. Return the persistent `Track` (same instance every call; buffers rewritten in place).

Re-running with the same `rng` seeds reproduces the same tracks (deterministic); reseeding the
rng before `generate()` yields new tracks, written into the same buffers. **The graph rework is
the hard, CUDA-only piece** (Phase B); Phase A delivers steps 1/4/5 (always-eager).

### 6. Dependencies (`pyproject.toml`)
```toml
dependencies = ["numpy", "warp-lang"]            # torch + scipy removed from core
[project.optional-dependencies]
dev = ["pytest", "matplotlib", "scipy", "torch"]  # oracle + experimental + test boundary
ui  = ["gradio"]
```
The shipped wheel no longer depends on torch. `import track_gen` works without torch installed.

### 7. Test strategy (chosen: convert at the boundary)
The torch oracle **stays torch** (the reference). Each of the ~40 test files that build torch
inputs / compare torch outputs is updated to: build inputs as `wp.array`/numpy, and wrap the
pipeline's `wp.array` output with `wp.to_torch(...)` before the existing `torch.allclose` /
`torch.equal` assertions against the oracle. Mechanical, broad, preserves the cross-check.
A new guard test asserts `import track_gen` pulls **no** torch (subprocess; mirrors the warp
guard test we removed earlier, inverted target).

### 8. `_experimental/fourier.py`
**Left in place** (torch, private, unsupported). It imports torch (a dev dep). Its RNG calls
switch to `wp.to_torch(rng.sample_*_warp(...))` since the torch samplers are gone. Not imported
by `__init__` or the hot path, so it never pulls torch into the runtime import graph.

### 9. Facade / benchmarks / viz
- `track_generator.py` **becomes the whole public entry point** (§5): it owns the buffers,
  seed buffer, and (Phase B) the graph; `generate()` updates in place and returns the persistent
  `wp.array`-`Track`. The pipeline kernels (formerly `warp_pipeline.generate_tracks_warp`) move
  to private stage helpers it drives. Public `__all__` = `TrackGenerator`, `TrackGenConfig`,
  `Track`, `PerEnvSeededRNG`, `__version__` (the two `generate_tracks_warp*` names are gone).
- `benchmarks/`, `viz/` are **consumers** (outside the runtime): they construct a
  `TrackGenerator`, call `generate()`, and convert the `wp.array` outputs on **their** side
  (`wp.to_torch`/`wp.to_numpy`) for metrics/plotting. That readback is consumer-side and allowed
  (it is not in `track_gen/`).

## Phasing

- **Phase A — eager in-place path (CPU-testable), the bulk:** `Track`→`wp.array`; lift all
  buffers (output + scratch + seed) to one-time allocation owned by `TrackGenerator`;
  `generate()` runs the pipeline eagerly **in place** and returns the persistent `Track`;
  remove the free `generate_tracks_warp`/`_graph` functions + `CapturedTracks` and slim `__all__`;
  real-op replacements; RNG de-torch + numpy-RNG removal; `pyproject`; oracle/Fourier RNG-call
  updates; ~40 test-file boundary conversions; stable-pointer + torch-free-import + no-readback
  guards. Gate: `pytest -q` green on the Warp `cpu` device.
- **Phase B — automatic graph capture (CUDA-only):** add the capture-on-first-call / replay
  branch to `generate()` (steps 2–3 of §5) using Warp-native `wp.ScopedCapture`. Gate: graph
  capture+replay equals the eager result on `cuda` (guarded by `torch.cuda.is_available()`), and
  a second `generate()` is a pure replay (no re-capture, buffers in place).

Each phase is its own implementation plan.

## Risks

- **Graph rework (Phase B)** is the real risk: Warp graph capture/replay semantics + static
  buffers must be gotten right, and it only validates on CUDA. Mitigation: phase it last,
  behind the green eager path; test on `cuda:0`.
- **Test-migration breadth (~40 files):** mechanical but the largest churn; a wrong boundary
  conversion is caught by `allclose` against the oracle.
- **`wp.array` ergonomics in tests:** indexing/reshaping differs from torch; the boundary
  `wp.to_torch` keeps assertions in torch to minimize churn.
- **`wp.where`/select availability:** if no clean Warp builtin, use a tiny select kernel or a
  host numpy select — both acceptable on this path.

## Verification

- `pytest -q` green before (baseline) and after Phase A on Warp `cpu`.
- `python -c "import sys, track_gen; assert 'torch' not in sys.modules"` (fresh interpreter) —
  the runtime is torch-free.
- `pip install -e .` with **torch absent** still imports `track_gen` and runs a CPU generation.
- **`grep -rn "wp.to_numpy\|\.numpy()\|np\." track_gen/` shows numpy only in one-time `__init__`
  setup** — zero numpy and zero readback in any `generate()`/`replay()`/hot-path code path; the
  numpy-RNG path is gone.
- **Stable pointers:** a test calls `generate()` twice and asserts the `Track` `wp.array`s have
  the **same `.ptr`** both calls (buffers reused in place, not re-allocated), and that a
  `wp.to_torch(track.center)` taken after the first call reflects the second call's values.
- Phase B: graph capture/replay equals eager output (positions allclose ~1e-4; valid/count
  exact) on `cuda`.

## Open items resolved

- Track type → `wp.array` (not numpy: numpy would force host copies for CUDA outputs).
- Output + scratch buffers → **pre-allocated once, stable pointers, updated in place**; owned
  by a stateful object; `generate()`/`replay()` return the same `Track`. Zero per-call alloc.
- numpy → **init-only host glue, owns no real op**; numpy-RNG path removed; **no `wp.to_numpy`
  / readback in `track_gen/`** (the boundary serves `wp.array`; consumers convert).
- API → one stateful `TrackGenerator` that **auto-captures** a Warp graph on the first CUDA
  `generate()` and replays it thereafter (eager in place on `cpu`); free
  `generate_tracks_warp`/`_graph` + `CapturedTracks` removed; `__all__` =
  `TrackGenerator`, `TrackGenConfig`, `Track`, `PerEnvSeededRNG`, `__version__`.
- Fourier → left in `track_gen/_experimental/` (torch, dev-only-importable).
- Test validation → `wp.to_torch` at the boundary; oracle stays torch.
- torch → `dev` dependency; scipy → `dev`; core = `numpy`, `warp-lang`.
