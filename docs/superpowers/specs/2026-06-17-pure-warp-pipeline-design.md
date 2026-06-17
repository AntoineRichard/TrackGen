# Pure-Warp End-to-End Track Generation Pipeline — Design

**Date:** 2026-06-17
**Status:** Proposed (pre-implementation)
**Builds on:** the merged relaxation feature + the fused-Warp XPBD solve (`track_gen/warp_relax.py`).
**Scope:** Re-express the *entire* track-generation pipeline — generation → resample → relax → inflate — in **pure NVIDIA Warp kernels**, captured as a single **end-to-end CUDA graph**, runnable on Warp's **CPU device** (tests/CI) and **CUDA** (production), with **torch reduced to the I/O container only** (`wp.from_torch`/`wp.to_torch` at the boundary). The existing torch implementation is used as a **verification oracle** for each kernel and then **removed**, leaving a single, no-mix Warp implementation.

---

## 1. Goal & context

Today the pipeline is a torch/Warp **mix**: generation is torch (+ Warp RNG), geometry/inflation are torch, and the relaxation is fused Warp on CUDA but torch on CPU. This works (E=8192 relax ~0.32 s; gen ~5 s on GPU) but is architecturally untidy and not graph-capturable end-to-end. Two motivations, per the owner:

1. **No torch/Warp mixing.** One implementation, not two paths.
2. **End-to-end graphing.** The whole pipeline as one replayable CUDA graph → minimal launch overhead, GPU-resident, deployable.

A key enabler: **Warp kernels also run on the CPU device.** So a pure-Warp pipeline yields *both* CUDA and a CPU path (for tests) from one codebase — torch becomes just the array container. This is strictly tidier than the current two-path design.

Performance is a secondary motivation (the relax is already ~915× faster; gen-on-GPU already works — the earlier "crash" was a caller device-mismatch, not a Warp wall, and gen runs at ~5 s/8192). The primary win is **architectural unification + a single graph**; making gen fast (the leftover per-env `arc_length_resample` loop) comes for free in the Warp port.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| Target | **Pure Warp** for gen + resample + relax + inflate; torch only as the I/O container. Runs on Warp `cpu` and `cuda`. |
| Torch impl | **Verification oracle, then removed.** Each Warp kernel is checked `allclose` against the current torch function before it replaces it; once the Warp pipeline matches end-to-end, the torch compute is deleted. |
| Graph | **Single end-to-end CUDA graph**: gen(static regen) → resample → relax → inflate, captured once, replayed. |
| RNG | **Warp built-in per-thread RNG** (`wp.rand_init(seed)`, `wp.randf`/`wp.randn`), seeded per (env, attempt). The legacy `rng_kernels`/`PerEnvSeededRNG` state machine is retired from the pipeline. (The design already accepted a non-bit-compatible RNG change.) |
| Regen loop | **Static, capturable**: fixed `max_regen_iters` *attempts*; each env keeps its **first valid** candidate (masked); already-valid envs ignore later attempts. No data-dependent control flow. |
| Data layout | Flat Warp arrays: corners `[E*P]` vec2, dense/centerline `[E*M]`/`[E*N]` vec2, per-env scalars `[E]`. One thread per element; env index = `tid // stride`. |
| Determinism | Per-(env, attempt) seeding → bit-reproducible run-to-run on a given device. |

## 3. Architecture — kernels & data flow

```
seeds[E] ─► [GEN: static regen, K attempts]
              corner_sample → ccw_sort → prune(count) → tangents+cubic → dense[E*M]
              gates(angle, turning, simplicity) → accept-first-valid mask
            ─► centerline[E*M] (+ valid[E])
              │  arc_length resample (Warp)
              ▼
            center[E*N]
              │  relax (fused Warp xpbd — DONE)  [disp_k + apply_k] × iters
              ▼
            relaxed[E*N]
              │  inflate (Warp): frame+curvature → constant-width offset → validity
              ▼
            Track(outer,center,inner,tangent,normal,arclen,length,valid,count)[E,N,*]
```

Whole region (gen→inflate) captured in one CUDA graph; per call: write `seeds`/params into fixed buffers, replay, read `Track` buffers.

### 3.1 Kernels to author (each verified vs its torch oracle)

| Stage | Warp kernel(s) | Torch oracle (current) | Notes / hard parts |
|---|---|---|---|
| Corner sample | `corner_sample_k` | `_sample_cell_indices`/`_sample_corner_points` | Per-env grid-cell top-k of n uniforms → reproduce via a per-env size-k selection, OR redesign (RNG break accepted). |
| ccw sort | `ccw_sort_k` | `geometry.ccw_sort` | Per-env angular sort of P≤max_num_points corners → single-thread insertion sort per env (small P). |
| Prune | folded into assemble via `count[E]` mask | `_prune_corners` | Sample per-env count; mark tail invalid. |
| Tangents + cubic | `assemble_k` | `vertex_tangents` + `_cubic_bezier`/`_segment` | Per dense-sample thread: blend tangents, eval cubic Bernstein. |
| Gates | `gates_k` (+ `self_intersections` in Warp) | `_corner_angles`, `turning_number`, `geometry.self_intersections` | Simplicity = O(N²) self-intersection per attempt (per-env double loop). Turning = O(N). |
| Static regen | host loop of K attempts over the above, masked accept-first-valid | bounded regen `while` loop | Fixed K → capturable; advance RNG per attempt. |
| Resample | `resample_k` | `geometry.arc_length_resample` / `relaxation._resample_uniform` | Per-env arc-length cumsum + searchsorted-equivalent in-kernel (handle NaN/ragged real counts). |
| Relax | `_disp_kernel`+`_apply_kernel` | **done** (`warp_relax`) | Already verified vs torch (4e-9). |
| Inflate | `frame_k`, `width_k`, `offset_k`, `validity_k` | `inflation._frame_curvature_stage`/`_width_stage`/`_offset_stage`/`_validity_stage` | Constant width; validity = thickness + border self-intersection + turning. |

### 3.2 Module structure
- `track_gen/warp_pipeline.py` — all pipeline kernels + the graph-captured `generate_tracks_warp(config, seeds) -> Track`.
- `track_gen/warp_relax.py` — existing fused relax kernels (reused).
- Torch modules (`geometry.py`, `generators.py`, `inflation.py`) — kept ONLY as the test oracle during the port; deleted (or reduced to nothing) once Warp is verified and the suite migrated.

## 4. Verification strategy

The torch pipeline is a complete, validated oracle. For **each** Warp kernel:
1. Generate random/representative inputs (on CPU + CUDA).
2. Run the torch function and the Warp kernel.
3. Assert `allclose` (atol ~1e-5; the RNG-dependent corner sampling is compared on *fixed* inputs, i.e. feed identical corners to both `ccw_sort`s, etc. — RNG itself is validated separately for distribution, not bit-equality).
4. Warp kernels run on **both** `cpu` and `cuda` devices in tests (Warp CPU device), so CI without a GPU still exercises them.

End-to-end: the Warp pipeline's **validity yield and shape statistics** must match the torch pipeline's (≥ the ~98% relaxed-valid baseline), on a fixed seed set. The existing test suite is the regression gate; tests migrate from torch-function calls to Warp-pipeline calls as stages are replaced.

## 5. Risks & open questions

- **Corner-sample fidelity.** Exact grid-top-k reproduction in a per-env kernel is fiddly; a redesign (direct sampling + min-spacing handled by the angle gate) is acceptable given the accepted RNG break, but must still yield diverse, gate-passing tracks. Validate yield, not bit-equality.
- **`ccw_sort` in-kernel.** Per-env insertion sort of ≤~13 corners is simple; confirm stability vs the torch `argsort(atan2(dx,dy))` ordering (ordering convention must match so downstream geometry matches the oracle).
- **Static-regen yield/cost.** Fixed K attempts for *all* envs (vs dynamic re-draw of only failures) costs ~K× generation work but is capturable. K≈ current `max_regen_iters` (~20) should match the ~100% simple-yield; measure. Generation is cheap per attempt, so K× is acceptable.
- **Warp self-intersection (simplicity gate).** O(N²) per attempt per env; at N=256 that's the same dense cost the relax separation already pays comfortably. Fine.
- **Resample in-kernel with NaN/ragged counts.** The trickiest geometry kernel (variable real-point count per env, wrap closure); port carefully against the torch oracle.
- **Graph capture of the whole pipeline.** Requires fully static shapes/buffers (fixed E, N, K, iters) and no host-side branching inside the captured region — the static regen + fixed buffers satisfy this; confirm Warp graph capture spans all stages on one stream.
- **CPU-device parity.** Warp CPU kernels must produce results matching CUDA (within fp tolerance) so tests are meaningful.

## 6. Out of scope (for this port)
- Changing the *algorithm* (constraints, thickness criterion, validity definition) — this is a faithful re-expression in Warp, verified against the torch oracle.
- Hash-grid separation (characterized; only wins at N≥~1024; not the default).
- Multi-GPU / distributed.

## 7. Definition of done
- `generate_tracks_warp(config, seeds)` produces a `Track` on `cuda` and `cpu` (Warp devices), end-to-end CUDA-graph-captured on CUDA.
- Validity yield + shape stats match the torch oracle (≥98% relaxed-valid) on a fixed seed set; per-kernel `allclose` tests pass on cpu+cuda.
- The torch compute pipeline is removed; torch remains only as the array container. Full suite green; benchmark updated to the Warp pipeline.
