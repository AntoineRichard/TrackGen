# Architecture — the pure-Warp track-generation pipeline

This document explains how `track_gen` turns a batch of per-environment seeds into a
batch of `Track`s. The whole pipeline lives in
[`track_gen/_src/warp_pipeline.py`](../track_gen/_src/warp_pipeline.py) (plus the
relaxation solve in [`track_gen/_src/warp_relax.py`](../track_gen/_src/warp_relax.py)) and
is written entirely in [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels.

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
  │  GENERATION  (registered phase-1 generator selected by config.generator; single pass,
  │    fixed scratch, no host retry loop; e.g. bezier / polar / hull / voronoi)
  ▼
centerline[E, num_points, 2]     (every env real, ~always simple) + valid[E] (all True —
  │                                 final validity is decided post-relax by INFLATE)
  │  RESAMPLE   resample_constant_spacing → per-track count[e] = ⌊perimeter/spacing⌋+1, capped at N_max
  ▼
spaced[E, N_max, 2]              (NaN-padded past each track's count[e])
  │  RELAX      (XPBD: separation + spacing + bending, fixed iters, double-buffered, count-aware)
  ▼
relaxed[E, N_max, 2]
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
  what protects the fixed-`N` parity path the per-kernel oracle tests exercise.
- Public wrappers do: `_init()` (idempotent `wp.init`) → `wp.from_torch(t.reshape(...).contiguous(), dtype=...)`
  → `wp.launch(kernel, dim=..., device=str(tensor.device))` → `_sync(device)` → return torch views.
- **In-kernel idioms** (so there is no torch glue to break graph capture): boolean
  reductions use `int` 0/1 flags (Warp can't fold Python `bool`s in dynamic loops); NaN is
  `wp.nan`; conditional selects use `wp.where`; floating accumulations that must track a
  torch `cumsum` are done in `wp.float64` then cast.
- **Shared `@wp.func` helpers** keep the heavy geometry DRY across kernels:
  `_safe_normalize2` (= `v / max(‖v‖, 1e-8)`), `_nan0` (NaN/inf → 0), `_pruned_corner`
  (returns NaN for `i ≥ count`), `_thickness_func`, `_self_intersections_func`,
  `_turning_func`. The standalone kernels (`_thickness_k`, `_self_intersections_by_i_k`,
  `_turning_k`) and the fused `_validity_k`/`gates` all call the same helpers.

## Stages

### Generation — registered first-stage generators

The first stage is selected by `TrackGenConfig.generator` through
`track_gen._src.generator_registry`. Every registered generator implements the same
`GeneratorSpec` contract: allocate private fixed-shape scratch once, then write an
`[E*num_points]` closed centerline and `[E]` generation-valid flag in pure Warp. The common
runtime stays graph-capturable because each generator has static launch dimensions and no
host-side retry loop conditioned on generated data.

Registered generators:

- `bezier`: corner sampling -> prune-then-sort -> closed Bezier assembly with polygon fallback.
- `polar`: fixed-count polar control knots -> periodic Catmull-Rom -> bbox normalization.
- `hull`: angle-sorted random points -> midpoint displacement -> Catmull-Rom with polygon fallback.
- `voronoi`: fixed site field -> angular anchor targets snapped to nearby unused sites ->
  graph-cycle smoothing with polygon fallback. This is the runtime-safe distillation of the
  Voronoi/random-geometric spike; exact Voronoi/Delaunay ridge traversal stays out of the
  hot path because dynamic cycle extraction is not CUDA-graph friendly.

#### Bezier generator details

A **single pass**: one corner draw per env, no regen loop and no generation gate. The whole
thing is static (no early exit, no host branching on tensor data) so it stays graph-capturable.
The steps:

| step | wrapper / kernels | what it does |
|---|---|---|
| sample corners | `corner_sample` / `_corner_sample_k` | per-env seed `wp.rand_init(seed*9781)`; pick `max_num_points` grid cells (bounded duplicate-rejection) + per-corner noise; matches the torch generator's coordinate construction (RNG redesign — validated by properties, not bit-equality) |
| sample count | `corner_count_sample` / `_corner_count_sample_k` | per-env corner count in `[min,max]_num_points` (distinct RNG stream, `*6151`) |
| order (prune-then-sort) | `ccw_sort(raw, count)` / `_ccw_sort_k` | sort **only the first `count[e]`** corners by `atan2` around **their own** centroid (NaN tail untouched) → an angularly-monotone, star-shaped polygon. The old sort-all-then-keep-`count` ordered a partial wedge about the wrong (all-corner) centroid and produced figure-eight (winding-0) loops; prune-then-sort eliminates them |
| build | `assemble` / `_vertex_tangents_k` + `_assemble_k` | blend unit vertex tangents (`p·u_out + (1−p)·u_in`), then a cubic Bézier per **closed** edge (segments wrap `mod count[e]`, so the closing edge is a real Bézier rather than a dropped straight chord — **F1**); each handle is `rad·chord` but **clamped to `handle_clamp_frac · shorter-incident-edge`** so a long handle can't overshoot a nearby corner into a self-crossing (**F2**); the `count`→NaN prune is folded in |
| resample | `arc_length_resample_warp(dense, num_points)` | the dense Bézier → `num_points` arc-uniform points (fused into the generator output) |
| de-cross (Fix B) | `self_intersections` + selected polygon fallback + `_select_vec2_k` | the few tracks whose Bézier centerline still self-crosses fall back to their **corner polygon** (straight pieces), which the angle-sorted ordering makes provably simple; the downstream XPBD relax re-rounds the straightened corners. The fallback assemble/resample runs only for crossing envs, while the final device-side select keeps the stage graph-capturable |

Output: `centerline[E, num_points, 2]` (every env real — no NaN rows) and `valid[E]`, which is
**all True**: there is no generation gate. Final validity (turning / width / thickness / optional
border-crossing) is decided **after relaxation** by `inflate_warp`. Generation produces a simple
closed loop for ≈100% of envs at the default config.

### Resample — `arc_length_resample_warp` and `resample_uniform`

Two arc-length resamplers:

- **`arc_length_resample_warp(points[E,M,2], num)`** — the general, **NaN-aware** resampler
  (kernels `_arc_scan_k` + `_arc_lookup_k`). It compacts the finite points per env (drops
  NaN, in order), builds the closed-loop cumulative arc length in `float64`, and looks up
  `num` arc-uniform targets (searchsorted + lerp). Envs with `< 2` real points yield an
  all-NaN row and `count 0`. Fused into the generator output as the dense→`num_points` resample
  and reused by the selected polygon-fallback de-cross path; also used by the standalone `gates` parity
  wrapper (dense→`num_points` and dense→`num_points_per_segment`), which the torch oracle tests
  exercise but the single-pass generator no longer calls.
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

### Relax — `warp_relax.xpbd_solve_inplace(center, band, L0, config)`

A fixed-iteration XPBD solve, double-buffered, with no `[E,N,N]` materialization and no
per-iteration host sync. Each sweep applies, per bead: a Jacobi-averaged **separation**
push (non-adjacent pairs closer than `target = 2*half_width*(1+relax_margin)`), an
**edge-spacing** correction toward rest length `L0`, and a flip-clamped **bending** push
when the local radius is below `R_min`. It runs on cpu and cuda (it syncs with
`wp.synchronize`, not `torch.cuda`), and reshapes the centerline so a constant-width
inflation becomes valid (thickness >= half_width). `generate_tracks_warp` derives `band`
and `L0` from `mean_seg_len` via `_band_l0_k`. The sweep is count-aware: it operates over
each env's `count[e]` real points (the fixed-`N` parity path is `count[e] == N`).

The separation term has two execution modes. The baseline `_step_kernel` performs the dense
`O(count[e]^2)` separation scan whenever `relax_sep_every` says to do so. With
`relax_sep_cache_slots == 0`, `relax_sep_every > 1` is a naive skip cadence: spacing and
bending still run every sweep, but separation is absent between dense scans. With
`relax_sep_cache_slots > 0`, the solver uses a fixed-size broadphase cache: `_build_sep_cache_kernel`
rebuilds candidate bead indices every `relax_sep_every` sweeps using radius
`target*(1+relax_sep_cache_skin)`, and `_step_cached_kernel` runs the exact narrowphase
`dist < target` test plus separation push on cached candidates every sweep. Cache buffers are
preallocated in `RelaxScratch`, so this path remains CUDA-graph-capturable.

### Inflate — `inflate_warp(center, config, valid=None, count=None)`

Composes: `resample_uniform` → `frame_curvature` (`_frame_k`: central-difference unit
tangent, left-normal, Menger curvature) → constant half-width → `offset`
(`_offset_build_k` + `_offset_assign_k`: ±`w` along the normal; outer = larger-|area|
candidate) → `validity` (`_validity_k`) → an arc-length kernel (`_arclength_k`) → a `Track`.

`_validity_k` is a single per-env kernel that combines: the generation flag (all True now that
generation no longer gates — so validity is purely geometric), closed-loop turning ≈ 2π, a
width floor, no-NaN, thickness ≥ `(1−relax_tol)·half_width`, and — only when
`validity_border_check` is set (**default off**, as it is redundant with the thickness/separation
gate: a crossing or fat-band overlap drives `separation_min → 0 → thickness < half_width →
invalid` anyway) — zero border self-intersections. The offset, validity, and arclength stages
are count-aware — they operate over each env's `count[e]` real points (the fixed-`N` parity path
is `count[e] == N`).

## Output mode — constant spacing

`constant_spacing` is the **only** output mode (the dataclass enforces it; any other
`output_mode` raises in `__post_init__`). Each track is relaxed and emitted at a constant arc
spacing: a per-track `count[e] = floor(perimeter/spacing)+1` (decremented while
`(count-1)·spacing ≥ perimeter`, capped at `N_max`, NaN-padded past `count[e]`).

The legacy `fixed` mode — every track padded to a constant point *count* (`num_points`) — was
**dropped**. A fixed 256 points **over-resolves** the centerline relative to its half-width
(segment ≈ 0.2 m ≪ a 0.5 m half-width), so the slow Jacobi XPBD solve **under-converges** under
the fixed iteration count → jagged tracks whose road self-overlaps. Relaxing at a
width-appropriate spacing instead lets the same solve converge → smooth, valid tracks on fewer
nodes/track (so it is also faster). `num_points` survives only as the intermediate
dense-resample resolution *before* the constant-spacing step.

`spacing` defaults to `None`, which auto-couples to `0.6·half_width` (the relax-friendly rule of
thumb) — a fixed spacing default would be wrong as `half_width` varies. Set it explicitly to
override. **Size `N_max ≥ max(perimeter)/spacing + 1`**: a track whose true count exceeds `N_max`
is silently truncated (its closing segment then spans the gap) and fails validity — the fat-band
default (`half_width=0.5`, `spacing=0.30`, `N_max=384`) leaves ample headroom (mean ≈ 160, max ≈
270 points/track).

## Parameter explorer

[`viz/param_explorer.py`](../viz/param_explorer.py) is an interactive
[Gradio](https://www.gradio.app/) UI for *seeing* how the config affects generation: sliders
for the generator method, regime / shape / generator-specific / resolution / relaxation knobs
drive the real `TrackGenerator` pipeline, rendering a paged grid of tracks plus the valid-yield
and quality stats over a full batch (so the yield numbers are statistically meaningful). It
builds on the same pure core (`build_config` → `TrackGenerator.generate` → `draw_track`). It opens on the high-yield fat-band
regime — `half_width=0.5`, `scale=10`, `spacing=0.30`, `N_max=384`, XPBD `150` iters,
`rad=0.4`/`handle_clamp_frac=0.4` (clamp == rad, so it only trims overshoot corners rather
than binding every segment) — the config that relaxes to ≈ 99.9% valid at the default batch.
Launch with `.venv/bin/python -m viz.param_explorer` (needs the optional `ui` extra); the
README has the control walkthrough.

## Torch as the test oracle

The original torch implementation is **retained, but only as the verification oracle** and
lives under `tests/_oracle/` (importable by tests as `tests._oracle.*`); it is **not**
shipped as part of the `track_gen` package. The modules `tests._oracle.geometry`,
`tests._oracle.inflation`, `tests._oracle.generators`, and `tests._oracle.relaxation` are
warp-free and are **not** imported by the runtime pipeline. Every Warp kernel has a test
(`tests/test_warp_*.py`) asserting it matches its torch counterpart on both `cpu` and
`cuda` (`torch.equal` for integer/boolean results; `allclose` at ~1e-4 for float results —
Warp's float32 `sqrt`/`length` differs from torch by ~ULP, which is geometrically
negligible and an accepted tolerance; the corner-sampling RNG is validated by structural
properties, not bit-equality, since it is a deliberate redesign).

The Fourier generator lives in `track_gen._experimental.fourier` and is **unsupported** —
it is self-contained, not on the Warp pipeline, and receives no compatibility guarantees.

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

- **Determinism.** Warp's per-env RNG is deterministic, so a given seed buffer reproduces the
  same tracks run-to-run on a device. The `cpu` and `cuda` RNG streams may differ — each device
  is internally reproducible; cross-device yields are compared statistically, not per-env.
- **Yield.** Relaxed-valid yield is ≈ **0.999** end-to-end (E ≥ 2048): ≈ 0.9991 at the fat-band
  default (`half_width=0.5`, `scale=10`, `spacing=0.30`, `N_max=384`), ≈ 0.9955 at the library
  default config, ≈ 0.9998 in the thin (`half_width=0.03`) regime — all measured at E=8192. Two
  changes got it there:
  - **Constant spacing made relaxation lossless.** The old `fixed`-256 ceiling (≈ 0.68 in the
    fat-band regime) was slow-Jacobi **under-convergence** from over-resolution, *not*
    un-relaxable geometry: at 256 points the centerline is over-resolved relative to its
    half-width, so the fixed iteration count can't drive the Jacobi solve to convergence.
    Relaxing at ~0.6×half_width spacing (≈ 145–160 nodes/track, not 256) lifted that same regime
    **0.684 → 0.999** — and runs *faster* (fewer nodes), while staying graph-capturable.
  - **Single-pass generation + Fix B replaced the regen loop.** With relaxation lossless, the
    residual was a small fraction of *generation* self-crossers. Rather than a fixed
    `max_regen_iters` accept-first-valid loop, generation now takes **one** corner draw per env
    and routes any track whose Bézier centerline self-crosses to its (provably simple) corner
    polygon, which XPBD re-rounds — rescuing essentially every self-crosser (→ ≈ 0.999 at E=8192).
    `max_regen_iters` is therefore **vestigial** on the Warp path: it remains a `TrackGenConfig`
    field for the torch oracle but is ignored by `generate_tracks_warp`.
- **FP tolerance & hard thresholds.** Validity gates (`th_ok`, `turn_ok`) are hard
  comparisons; near a decision boundary the accepted ~1e-4 Warp-vs-torch drift can flip a
  single env's bool. Tests keep their inputs away from those boundaries; the end-to-end
  yield comparison uses an aggregate tolerance, not per-env equality.
