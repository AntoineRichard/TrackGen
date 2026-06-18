# Architecture — the pure-Warp track-generation pipeline

This document explains how `track_gen` turns a batch of per-environment seeds into a
batch of `Track`s. The whole pipeline lives in
[`track_gen/warp_pipeline.py`](../track_gen/warp_pipeline.py) (plus the fused relaxation
solve in [`track_gen/warp_relax.py`](../track_gen/warp_relax.py)) and is written entirely
in [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels.

## Goals

1. **No torch/Warp mixing.** One implementation. Every pipeline stage is a Warp kernel;
   PyTorch is only the array container at the boundary (`wp.from_torch` in, torch views
   out). The pipeline imports no torch compute module at runtime.
2. **One codebase, two devices.** Warp kernels run on the Warp **`cpu`** device (GPU-free,
   for tests/CI) and on **`cuda`** (production). The same code path serves both.
3. **A single replayable CUDA graph.** The entire pipeline is static (fixed shapes, fixed
   iteration counts, no host-side branching on tensor data), so it captures into one CUDA
   graph that can be replayed with new seeds.

## Data flow

```
seeds[E]
  │  GENERATION  (static regen: a fixed K = max_regen_iters attempts, accept-first-valid;
  │    corner_sample ─► ccw_sort ─► assemble (Bézier) ─► gates ─► arc-length resample [FUSED])
  ▼
centerline[E, N, 2]              (= the gated arc-length resample to num_points; NaN rows for
  │                                 envs that never passed) + valid[E]
  │  RELAX      (fused XPBD: separation + spacing + bending, fixed iters, double-buffered)
  ▼
relaxed[E, N, 2]
  │  INFLATE    resample_uniform (re-uniformize) ─► frame+curvature ─► constant-width offset
  │             ─► validity ─► arclength
  ▼
Track(outer, center, inner, tangent, normal, arclen, length, valid, count)
```

## Array layout & kernel conventions

- Points are flat `wp.array(dtype=wp.vec2f)` of length `E*N` (or `E*M`, `E*P`). Per-env
  scalars are `[E]` arrays.
- **One thread per output element.** A kernel launched with `dim=E*N` decodes its
  environment as `e = tid // N` and its within-env index as `i = tid % N`. Reductions
  (per-env min/sum/count) use `dim=E`, one thread looping that env's `N` points.
- **Count-aware buffers.** Post-generation, `n_max` is the buffer stride and `count[e]`
  is env `e`'s real-point count; padding slots `i ≥ count[e]` hold `wp.nan`. Count-aware
  kernels loop `range(count[e])`, base their reads at `e*n_max`, wrap neighbours with
  `% count[e]`, and guard `i ≥ count[e]`. The **parity invariant**: when `count[e] == N_max`
  for all envs, every count-aware kernel is bit-identical to the fixed-`N` kernel — this is
  what protects fixed mode and the existing tests.
- Public wrappers do: `_init()` (idempotent `wp.init`) → `wp.from_torch(t.reshape(...).contiguous(), dtype=...)`
  → `wp.launch(kernel, dim=..., device=str(tensor.device))` → `_sync(device)` → return torch views.
- **In-kernel idioms** (so there is no torch glue to break graph capture): boolean
  reductions use `int` 0/1 flags (Warp can't fold Python `bool`s in dynamic loops); NaN is
  `wp.nan`; conditional selects use `wp.where`; floating accumulations that must track a
  torch `cumsum` are done in `wp.float64` then cast.
- **Shared `@wp.func` helpers** keep the heavy geometry DRY across kernels:
  `_safe_normalize2` (= `v / max(‖v‖, 1e-8)`), `_nan0` (NaN/inf → 0), `_pruned_corner`
  (returns NaN for `i ≥ count`), `_thickness_func`, `_self_intersections_func`,
  `_turning_func`. The standalone kernels (`_thickness_k`, `_self_intersections_k`,
  `_turning_k`) and the fused `_validity_k`/`gates` all call the same helpers.

## Stages

### Generation — `generate_centerline_warp(seeds, config)`

A **static, fixed-iteration, masked accept-first-valid** loop (no early exit, no host
branching on tensor data — so it is graph-capturable). For each of `K = max_regen_iters`
attempts:

| step | wrapper / kernels | what it does |
|---|---|---|
| sample corners | `corner_sample` / `_corner_sample_k` | per `(env, attempt)` seed `wp.rand_init(seed*9781 + attempt)`; pick `max_num_points` grid cells (bounded duplicate-rejection) + per-corner noise; matches the torch generator's coordinate construction (RNG redesign — validated by properties, not bit-equality) |
| sample count | `corner_count_sample` / `_corner_count_sample_k` | per-env corner count in `[min,max]_num_points` (distinct RNG stream, `*6151`) |
| order | `ccw_sort` / `_ccw_sort_k` | per-env insertion sort by `atan2(dx, dy)` around the centroid → a simple polygon |
| build | `assemble` / `_vertex_tangents_k` + `_assemble_k` | blend unit vertex tangents (`p·u_out + (1−p)·u_in`), then a cubic Bézier per edge (handles at `rad·chord`); the `count`→NaN prune is folded in |
| gate | `gates` / `_corner_angles_gate_k` + `_gates_combine_k` | accept iff min corner-angle ok **and** turning ≈ 2π **and** finite **and** simple (self-intersection-free) on the 256-point resample |
| accept | `_select_first_valid_k` + `_or_update_k` | `take = accept & ¬valid`; copy that attempt's centerline in place; `valid |= accept` |

Output: `centerline[E, N, 2]` (the gated 256-resample; all-NaN for never-accepted envs)
and `valid[E]`. Generation yield is ~100% at the default config.

### Resample — `arc_length_resample_warp` and `resample_uniform`

Two arc-length resamplers:

- **`arc_length_resample_warp(points[E,M,2], num)`** — the general, **NaN-aware** resampler
  (kernels `_arc_scan_k` + `_arc_lookup_k`). It compacts the finite points per env (drops
  NaN, in order), builds the closed-loop cumulative arc length in `float64`, and looks up
  `num` arc-uniform targets (searchsorted + lerp). Envs with `< 2` real points yield an
  all-NaN row and `count 0`. Used by `gates` (dense→256 and dense→30) and, fused into the
  generator output, as the dense→`num_points` resample.
- **`resample_uniform(center[E,N,2], n, count=None)`** — the count-aware `N→N` re-uniformizer
  (`_resample_scan_k` + `_resample_lookup_k`), used after relax (and inside `inflate_warp`).
  With `count=None` all `E*N` points are real; with `count` it re-uniformizes each env's
  `count[e]` real points (NaN-padded past `count[e]`).
- **`resample_constant_spacing(center[E,N,2], spacing, N_max)`** — the count-aware
  resampler: from a fixed source it picks a per-track `count = floor(perimeter/spacing)+1`
  (decremented while `(count-1)·spacing ≥ perimeter`, capped at `N_max`),
  lays the arc-uniform points into an `[E, N_max, 2]` buffer NaN-padded past `count[e]`,
  and matches the `geometry.arc_length_resample(spacing=)` oracle. Selected by
  `output_mode="constant_spacing"`.

### Relax — `warp_relax.xpbd_solve(center, band, L0, config)`

A fixed-iteration XPBD solve, **fully fused** (kernels `_disp_kernel` + `_apply_kernel`,
double-buffered) so there is no `[E,N,N]` materialization and no per-iteration sync. Each
sweep applies, per bead: a Jacobi-averaged **separation** push (non-adjacent pairs closer
than `D·(1+margin)`), an **edge-spacing** correction toward rest length `L0`, and a
flip-clamped **bending** push when the local radius is below `R_min`. It runs on cpu and
cuda (it syncs with `wp.synchronize`, not `torch.cuda`), and reshapes the centerline so a
constant-width inflation becomes valid (thickness ≥ half_width). `generate_tracks_warp`
derives `band` and `L0` from `mean_seg_len` via `_band_l0_k`. The sweep is count-aware in
both modes — it operates over each env's `count[e]` real points (fixed mode is just
`count[e] == N`).

### Inflate — `inflate_warp(center, config, valid=None, count=None)`

Composes: `resample_uniform` → `frame_curvature` (`_frame_k`: central-difference unit
tangent, left-normal, Menger curvature) → constant half-width → `offset`
(`_offset_build_k` + `_offset_assign_k`: ±`w` along the normal; outer = larger-|area|
candidate) → `validity` (`_validity_k`) → an arc-length kernel (`_arclength_k`) → a `Track`.

`_validity_k` is a single per-env kernel that combines: the generation flag, closed-loop
turning ≈ 2π, a width floor, no-NaN, thickness ≥ `(1−relax_tol)·half_width`, and zero
border self-intersections. The offset, validity, and arclength stages are count-aware in
both modes — they operate over each env's `count[e]` real points (fixed mode is `count[e] == N`).

## Output modes / constant spacing

`output_mode="fixed"` (the default) gives every track `num_points` points. The catch: a
fixed 256 **over-resolves** the centerline relative to its half-width, so the slow Jacobi
XPBD solve **under-converges** under the fixed iteration count → jagged tracks whose 1 m
road self-overlaps. `output_mode="constant_spacing"` (`spacing`, `N_max`) instead relaxes
each track at a constant arc spacing (per-track `count[e] = floor(perimeter/spacing)+1`,
decremented while `(count-1)·spacing ≥ perimeter`, capped at `N_max`, NaN-padded); at a
width-appropriate spacing the same solve converges → smooth, valid tracks. The library
default is `spacing=0.1` (equal to the default `half_width`); a good rule of thumb is
`spacing ≈ 0.6·half_width` — the value the parameter explorer defaults to.

## Parameter explorer

[`viz/param_explorer.py`](../viz/param_explorer.py) is an interactive
[Gradio](https://www.gradio.app/) UI for *seeing* how the config affects generation: sliders
for the regime / shape / resolution / relaxation knobs drive the real `generate_tracks_warp`,
rendering a paged grid of tracks plus the valid-yield and quality stats over a full batch
(so the yield numbers are statistically meaningful). It builds on the same pure core
(`build_config` → `generate_tracks_warp` → `draw_track`) and defaults to `constant_spacing`.
Launch with `.venv/bin/python -m viz.param_explorer` (needs the optional `ui` extra); the
README has the control walkthrough. Note: the explorer's default is `constant_spacing`, whereas the
library `TrackGenConfig` default remains `fixed` (the stable, parity-tested baseline).

## Torch as the test oracle

The original torch implementation is **retained, but only as the verification oracle**:
`geometry.py`, `inflation.py`, `generators.py`, and the torch `relaxation.py` backends are
warp-free and are **not** imported by the runtime pipeline. Every Warp kernel has a test
(`tests/test_warp_*.py`) asserting it matches its torch counterpart on both `cpu` and
`cuda` (`torch.equal` for integer/boolean results; `allclose` at ~1e-4 for float results —
Warp's float32 `sqrt`/`length` differs from torch by ~ULP, which is geometrically
negligible and an accepted tolerance; the corner-sampling RNG is validated by structural
properties, not bit-equality, since it is a deliberate redesign).

## End-to-end CUDA graph — `generate_tracks_warp_graph`

Because the pipeline is pure Warp and sync-free, the **whole** thing captures as one CUDA
graph:

- CUDA graph capture is **stream-level**. `torch.cuda.graph` captures all CUDA work on its
  internal capture stream; Warp's launches are routed onto that same stream via
  `wp.ScopedStream`, so torch boundary ops *and* every Warp kernel land in one native graph.
- A module global `_CAPTURING` makes every wrapper's `_sync` (and `warp_relax`'s
  `wp.synchronize`) a no-op during capture — host-blocking syncs are illegal mid-capture,
  and the graph records stream ordering anyway.
- The `[E]` seed buffer is static; `CapturedTracks.replay(new_seeds)` copies new seeds into
  it and replays, re-running every stage on the GPU off the buffer contents. Replay yields
  a `Track` matching the eager `generate_tracks_warp` (positions allclose; `valid`/`count`
  exact).
- `output_mode="constant_spacing"` captures too: the per-track `count[e]` is device-side
  data the count-aware kernels read at runtime, so all launch dims stay static via `N_max`
  and nothing branches on tensor data on the host.

At large batches the pipeline is compute-bound (relaxation dominates), so graph replay is
~the same wall-clock as the eager call; the graph's value is a single, GPU-resident,
deployable replayable unit, not a speedup.

## Determinism, yield, FP tolerance

- **Determinism.** Warp's per-`(env, attempt)` RNG is deterministic, so a given seed buffer
  reproduces the same tracks run-to-run on a device. The `cpu` and `cuda` RNG streams may
  differ — each device is internally reproducible; cross-device yields are compared
  statistically, not per-env.
- **Yield.** Relaxed-valid yield is ≈ 0.975–0.98 at the default config (E ≥ 512), on par
  with the torch baseline. The fixed-mode residual loss — and the ≈ 0.68 yield in the
  tight-width / fat-band regime — is largely slow-Jacobi **under-convergence** from
  over-resolution, *not* genuinely un-relaxable geometry: at a fixed 256 points the centerline
  is over-resolved relative to its half-width, so the fixed iteration count cannot drive the
  Jacobi solve to convergence. `output_mode="constant_spacing"` (relaxing at ~0.6×half_width
  spacing) makes the relaxation essentially **lossless** — every *generation*-valid track stays
  valid after relax. At the default `max_regen_iters=10` (both modes, identical generation) the
  E=8192 yield goes **0.684 → 0.999**, while running *faster* at equal regen (~0.55 vs ~0.79
  s/8192 — the solve runs on ~145 nodes/track, not 256) and remaining graph-capturable. With
  relaxation no longer the bottleneck, the residual ceiling is now **generation/regen**:
  final-valid ≈ generation-valid (~0.52 at `max_regen_iters=1`, ~0.999 at 10) — whereas in
  fixed mode regen could not move the relaxation-bound yield (flat 0.684 at regen 10/20/40).
- **FP tolerance & hard thresholds.** Validity gates (`th_ok`, `turn_ok`) are hard
  comparisons; near a decision boundary the accepted ~1e-4 Warp-vs-torch drift can flip a
  single env's bool. Tests keep their inputs away from those boundaries; the end-to-end
  yield comparison uses an aggregate tolerance, not per-env equality.
