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

### 2. Allocation layer → Warp / numpy
Every `torch.empty/full/zeros/empty_like` buffer in `warp_pipeline.py` / `warp_relax.py`
becomes a `wp.zeros`/`wp.empty` device array (replacing the `torch.empty(...)` + `wp.from_torch`
pair with a direct `wp.array`), or `numpy.full` for host-side scalar/count arrays. `torch.int/
float/bool` dtype constants become `wp.int32`/`wp.float32`/`wp.bool` (or `np.*`). `torch.cuda.
synchronize` → `wp.synchronize`. This is the bulk: trivial but high-volume (~25–30 sites).

### 3. Real torch ops → Warp / numpy / deletion
- `_mean_seg_len_torch` (dead on the Warp path) → **delete** (and the `import torch` it forces).
- `torch.where(...)` selects (the Fix-B polygon-fallback select, ×3) → a small Warp select
  kernel or a numpy host select on the already-computed arrays.
- Scattered `.long()`/`.bool()`/`.view()`/`.contiguous()` → `wp` dtype/shape equivalents or
  `wp.to_numpy(...).astype(...)` for host-side counts.

### 4. `PerEnvSeededRNG` → torch-free
`rng_utils.py` drops `from torch import …` and the torch-facing surface
(`seeds_torch`, `states_torch`, `sample_*_torch`, the torch branch of `set_seeds`). It keeps
the warp-native (`seeds_warp`, `sample_*_warp`) + numpy methods. The pipeline already uses the
warp path, so the runtime is unaffected. **Callers that needed torch samples** — the oracle
generators (`tests/_oracle/generators.py`) and `_experimental/fourier.py` — wrap the warp/numpy
samplers with `wp.to_torch(...)` at their call sites (they are dev/test-side, torch-allowed).

### 5. CUDA-graph path → Warp-native capture
With the glue gone, `generate_tracks_warp_graph` no longer needs `torch.cuda.CUDAGraph` +
`wp.ScopedStream` stream-routing. It uses Warp's native capture (`wp.capture_begin(device)` /
`wp.capture_end()` → a `wp.Graph`, replayed with `wp.capture_launch`). `CapturedTracks` holds a
static `wp.array` seed buffer; `replay(new_seeds)` copies into it (`seed_buf.assign(...)`) and
`wp.capture_launch`es; the captured output is a static `wp.array` `Track`. This removes the
torch warmup/side-stream dance entirely. **CUDA-only; the hard, separately-phased piece.**

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
- `track_generator.py`: `_resolve_ids`/`_seeds_for` produce `wp.array` (or accept `wp.array`);
  `generate` returns the `wp.array`-`Track`.
- `benchmarks/`, `viz/`: seeds via `wp.array`/`wp.arange`; outputs read with `wp.to_numpy` /
  `wp.to_torch` for metrics/plotting. Import lines + boundary conversions only.

## Phasing

- **Phase A — eager path (CPU-testable), the bulk:** Track→`wp.array`, allocation layer,
  real-op replacements, RNG de-torch, facade, `pyproject`, oracle/fourier RNG-call updates,
  and the ~40 test-file boundary conversions. Gate: `pytest -q` green on the Warp `cpu` device.
- **Phase B — graph path (CUDA-only):** port `generate_tracks_warp_graph` + `CapturedTracks`
  to Warp-native capture. Gate: the graph tests pass on `cuda` (`torch.cuda.is_available()`).

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
- Phase B: graph capture/replay equals eager output (positions allclose ~1e-4; valid/count
  exact) on `cuda`.

## Open items resolved

- Track type → `wp.array` (not numpy: numpy would force host copies for CUDA outputs).
- Fourier → left in `track_gen/_experimental/` (torch, dev-only-importable).
- Test validation → `wp.to_torch` at the boundary; oracle stays torch.
- torch → `dev` dependency; scipy → `dev`; core = `numpy`, `warp-lang`.
