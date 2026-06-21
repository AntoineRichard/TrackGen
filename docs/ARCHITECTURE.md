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
  │  FIRST-STAGE GENERATION  (registered config.generator: "bezier", "hull", or "polar";
  │    single pass, generator-private scratch, no regen loop, no generation gate)
  ▼
centerline[E, num_points, 2]     (every env real) + valid[E] (all True —
  │                                 final geometric validity is decided post-relax by INFLATE)
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

### Generation — registered first-stage centerline methods

The first stage is now pluggable. `TrackGenerator.__init__` resolves
`config.generator` through `track_gen._src.generator_registry`, allocates that generator's
private scratch once, and `_run_pipeline` calls the resolved `GeneratorSpec.generate`
with orchestrator-owned `out_centerline` and `out_valid_wp` buffers. The production
runtime registry currently exposes:

| generator | module | representation | repair path |
|---|---|---|---|
| `"bezier"` | `warp_generate.py` | sampled grid corners → angle-sorted closed cubic Bezier | selected corner-polygon fallback for Bezier self-crossers |
| `"hull"` | `warp_generate_hull.py` | angle-sorted point loop → displaced midpoints → closed Catmull-Rom | selected augmented-polygon fallback for Catmull-Rom self-crossers |
| `"polar"` | `warp_generate_polar.py` | sorted polar control knots → periodic Catmull-Rom spline | no generator-local fallback; downstream relaxation/inflation gates final validity |

Every registered runtime generator follows the same hard contract:

- `alloc_scratch(config)` allocates fixed-shape, generator-private Warp buffers once.
- `generate(seeds_wp, config, out_centerline, out_valid_wp, scratch)` writes an
  `[E*num_points]` closed centerline into the supplied output buffer and writes `[E]`
  generation flags.
- The hot path is pure Warp, zero allocation, deterministic in `(seed, config)`, and
  graph-capturable: no host-side retry loop, no host branch on generated tensor data, and
  no per-env Python branching.
- `out_valid_wp` is filled with `1` by the current runtime generators. It is a stage flag,
  not the final quality decision. Turning, thickness, NaNs, width floor, and optional
  border intersections are judged later by `inflate_warp` after constant-spacing and PBD
  relaxation.

#### `generator="bezier"` — corner-sort Bezier family

This is the original first-stage method, now wrapped as a `GeneratorSpec` and extended
with optional per-env style sampling. It is still a **single-pass** draw: one count draw,
one corner-position draw, no accept/retry loop, then device-side rescue for the few smooth
curves that cross themselves.

1. **Optional style draw.** When `style_sampling=False` (the default), the legacy scalar
   kernels consume `rad`, `scale`, and `handle_clamp_frac` exactly as before. When
   `style_sampling=True`, `_style_sample_k` draws per-env `rad`, `scale`, and
   `handle_clamp_frac` from `*_range` fields into `[E]` device arrays using the seed salt
   `2741`. A missing range collapses to the scalar value, so individual knobs can stay
   fixed. The Python branch selecting scalar vs style kernels is resolved at CUDA graph
   capture time; the sampled values themselves are device data.
2. **Corner count and positions.** `_corner_count_sample_k` draws
   `count[e] in [min_num_points, max_num_points]` from a stream salted by `6151`.
   `_corner_sample_k` (or `_corner_sample_style_k`) draws `max_num_points` grid cells from
   a `num_cells x num_cells` grid derived from `min_point_distance`, with bounded
   duplicate-cell rejection and per-corner jitter. The position stream uses salt `9781`;
   style sampling only changes the per-env scale multiplier.
3. **Prune-then-sort.** `_ccw_sort_k` sorts only the first `count[e]` corners by
   `atan2(dx, dy)` around the centroid of those same kept corners and writes NaN past the
   count. Sorting after pruning matters: sorting all `P` points and then keeping a prefix
   produced partial angular wedges around the wrong centroid, which could close as
   figure-eight loops.
4. **Closed Bezier assembly.** `_vertex_tangents_k` blends incoming/outgoing unit edge
   directions using `edgy` (`p = atan(edgy)/pi + 0.5`). `_assemble_k` emits
   `num_points_per_segment` samples for every closed edge, wrapping the final corner back
   to the first. The handle length starts as `rad * chord`, then is clamped to
   `handle_clamp_frac * min(adjacent edge lengths)` so a handle cannot overshoot a nearby
   corner. The style variant reads `rad[e]` and `clamp[e]` from the sampled arrays.
5. **Dense-to-N resample and de-cross.** `_arc_resample_inplace` arc-resamples the dense
   Bezier loop to `num_points`. `self_intersections_inplace` counts proper crossings on
   that N-point loop. For crossing envs only, `_assemble_polygon_selected_k` emits the
   straight corner polygon and `_arc_resample_selected_inplace` overwrites those rows.
   `_select_vec2_k` then chooses Bezier rows for non-crossers and polygon rows for
   crossers. XPBD later re-rounds the polygon fallback.

The important knobs are `min_num_points`, `max_num_points`, `num_points_per_segment`,
`min_point_distance`, `rad`, `edgy`, `scale`, `handle_clamp_frac`, and the opt-in
`style_sampling` ranges. The method remains mostly star-shaped around the sampled-corner
centroid; style sampling broadens the family without changing the representation.

#### `generator="hull"` — angle-sort hull plus midpoint displacement

This method is the method-2 production port from `docs/pre-relaxation-generator-methods.md`.
It deliberately avoids a true dynamic convex-hull algorithm because that is awkward in a
fixed-shape Warp graph. Instead, it uses the same angle-sort pattern as Bezier as a cheap,
static hull-like base, then adds a radial midpoint-displacement layer for more racing-shape
variety.

1. **Point count and positions.** `_point_count_sample_k` draws the base count with salt
   `5119`. `_point_sample_k` draws `P=max_num_points` grid-jittered points with bounded
   duplicate-cell rejection, using the same coordinate construction and `scale` convention
   as Bezier but a different position salt (`7919`).
2. **Hull-like ordering.** `_angle_sort_k` sorts the first `m=count[e]` points by angle
   around their own centroid and writes NaN beyond `m`. This is not an exact convex hull;
   it is a fixed-bound stand-in that gives a simple ordered base loop in the same static
   execution style as the rest of the pipeline.
3. **Midpoint displacement.** `_midpoint_displace_k` interleaves every sorted vertex with
   one displaced edge midpoint, producing an augmented loop of `2m` vertices. Each midpoint
   is moved along the radial direction from the centroid through the midpoint by a signed
   random amount in `[-hull_displacement, +hull_displacement] * distance(centroid, midpoint)`
   using salt `3083`. Positive values bulge a lobe outward; negative values pinch it inward.
4. **Closed Catmull-Rom smoothing.** `_catmull_rom_k` evaluates a closed uniform
   Catmull-Rom spline over the augmented vertices, with `2P * num_points_per_segment`
   dense slots per env. Segments beyond the real augmented count write NaN and are ignored
   by the arc resampler.
5. **Dense-to-N resample and fallback.** The smooth dense loop is arc-resampled directly
   into `out_centerline`, checked for self-intersections, and crossing envs are overwritten
   by an arc-resampled straight augmented polygon. Non-crossing rows are left untouched by
   the selected resampler.

The main knob is `hull_displacement`. At `0`, the method collapses toward a plain
angle-sorted loop. Larger values create stronger lobes, pinches, and straights, but can
increase the downstream relaxation and thickness burden.

#### `generator="polar"` — periodic polar control-knot spline

The polar generator is the method-3 production port. It starts from a closed radial
representation instead of sampled Cartesian corners, so it is smooth and centered by
construction and is useful as a low-burden contrast to the corner/polygon families.

1. **Control knots.** `_polar_controls_k` draws `K=polar_num_knots` fixed-order polar
   controls from salt `7919 + 17`. Each knot `i` starts at angle `2*pi*i/K`, receives
   bounded angular jitter, and receives radial jitter around `_BASE_RADIUS`. The radial
   jitter is clamped so radii stay positive; angular jitter is clamped below half a cell
   so the index order remains the sorted angular order without a runtime sort.
2. **Periodic spline.** `_polar_spline_dense_k` evaluates a closed uniform Catmull-Rom
   spline through the `K` controls. Samples are endpoint-excluded within each segment;
   the arc resampler closes the final segment back to the first point.
3. **Dense-to-N resample.** `_arc_resample_inplace` converts the dense periodic spline to
   `num_points` arc-uniform points.
4. **Normalization.** `_normalize_centerline_k` centers each env by its bounding box and
   isotropically rescales the longest bbox dimension to `config.scale * 1.44`. The `1.44`
   constant matches the Bezier baseline's typical longest bbox extent at `scale=1`, so
   `half_width`, constant spacing, and relaxation see comparable coordinate ranges across
   generators.
5. **Generation flag.** The generator writes `out_valid_wp=1` for every env. It does not
   run a local polygon fallback; any rare bad geometry is handled by the common
   post-relax validity gate.

The knobs are `polar_num_knots`, `polar_radial_jitter`, and `polar_angular_jitter`. The
implementation intentionally uses random radial knots rather than the old low-pass Fourier
function path, because the latter tended to collapse toward high-compactness near-circles.

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

### Relax — PBD/XPBD bead-chain relaxation

Relaxation lives in `track_gen/_src/warp_relax.py` and is the only production relax backend
used by `TrackGenerator` (`relax_solver="xpbd"`, `smooth_finish=False`). The implementation
is a PBD/XPBD-style position projection over the constant-spacing centerline beads: it does
not allocate dense `[E,N,N]` tensors, does not run an optimizer, and does not store a
per-constraint Lagrange multiplier history. Instead, every fixed sweep reads one complete
position buffer, computes local position corrections, writes a second buffer, and swaps the
two buffers. That gives Jacobi semantics and keeps the solve graph-capturable.

Setup happens before the sweep in `band_l0_inplace`:

- `L0[e] = perimeter(center[e]) / count[e]`, the rest edge length for the constant-spacing
  bead chain.
- `band[e] = round(2*half_width / L0[e]).clamp_min(1)` unless `config.relax_band` overrides
  it. The band excludes immediate geometric neighbours from the separation scan, so the
  road does not try to push apart points that are adjacent along the same local segment.
- `target = 2*half_width*(1+relax_margin)` is the non-local separation distance. It is a
  slightly inflated road diameter, so relaxation leaves margin for the later thickness gate.
- `R_min = half_width*(1+relax_margin)` is the local curvature-radius target used by the
  bending correction.

Each sweep applies three corrections per real bead `i`:

1. **Non-local separation.** For every bead `j` with circular index distance greater than
   `band[e]`, if the current Euclidean distance is below `target`, bead `i` receives a push
   along `xi - xj` of `0.5 * (target - dist) / dist`. The pushes are averaged over all
   colliding candidates for that bead, then scaled by `relax_sep_relax`. This is the term
   that opens self-approaches and makes enough room for a constant-width road.
2. **Edge spacing.** The two incident edges `(i-1,i)` and `(i,i+1)` are corrected toward
   rest length `L0[e]`. The implementation uses the local formula from `_step_kernel`,
   scaled by `relax_spc_relax`, so the relaxed loop keeps near-constant bead spacing
   instead of stretching into a few long chords.
3. **Bending / radius guard.** The local Menger curvature through `(i-1,i,i+1)` gives a
   radius estimate. If `radius < R_min`, the bead is pushed toward the midpoint of its
   neighbours. The scale is `relax_bend_relax * (R_min - radius) / R_min`, clamped to `1`,
   so the bead never passes the chord midpoint in a single sweep. This removes jagged
   under-radius corners introduced by generation or by separation pushes.

The solver is count-aware. Kernels launch over the static stride `N_max`, but threads with
`i >= count[e]` copy through the NaN-padded tail. When every `count[e] == N_max`, the same
kernels reduce to the old fixed-N parity path.

Separation has two execution modes:

- **Dense / cadenced mode.** `_step_kernel` scans all non-band neighbours in
  `O(count[e]^2)` whenever `step_i % relax_sep_every == 0`; spacing and bending still run
  every sweep. If `relax_sep_cache_slots == 0` and `relax_sep_every > 1`, this is a naive
  skip cadence: there is no separation force between dense scans.
- **Cached broadphase mode.** When `relax_sep_cache_slots > 0` and `relax_sep_every > 1`,
  `_build_sep_cache_kernel` refreshes a fixed-slot directed candidate list every
  `relax_sep_every` sweeps using radius `target*(1+relax_sep_cache_skin)`. Then
  `_step_cached_kernel` runs every sweep, re-testing each cached candidate with the exact
  current `dist < target` narrowphase before applying the separation push. Cache arrays live
  in `RelaxScratch`, including an overflow counter for beads whose candidate list exceeded
  the configured slot count, so the mode remains allocation-free and CUDA-graph-capturable.

If `relax_enable=False`, `_run_pipeline` bypasses `band_l0_inplace` and `xpbd_solve_inplace`
and inflates the constant-spacing centerline directly. Otherwise the relaxed output is
re-uniformized by `inflate_warp` before frame, offset, validity, and arclength are computed.

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
for the regime / shape / resolution / relaxation knobs drive the real `TrackGenerator.generate`,
rendering a paged grid of tracks plus the valid-yield and quality stats over a full batch
(so the yield numbers are statistically meaningful). It builds on the same pure core
(`build_config` → `PerEnvSeededRNG` + `TrackGenerator.generate` → `draw_track`). It opens on the high-yield fat-band
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

## End-to-end CUDA graph — `TrackGenerator.generate`

`TrackGenerator` owns the production graph-captured path. Construction resolves the selected
generator, pre-allocates the persistent `Track`, all per-stage scratch groups, and the `[E]`
seed buffer. `generate()` always returns that same `Track` instance with stable `wp.array`
pointers; callers that need a snapshot use `Track.clone()`.

On the Warp `cpu` device, every `generate()` call runs `_run_pipeline` eagerly. On `cuda`,
the first call compiles/warms the kernels, captures `_run_pipeline` with `wp.ScopedCapture`,
stores the resulting `wp.Graph`, and then launches it. Subsequent calls copy the current
`rng.seeds_warp` values into the pre-allocated seed buffer and replay the stored graph with
`wp.capture_launch`.

The capture works because every stage is pure Warp and fixed-shape:

- A module global `_CAPTURING` makes every wrapper's `_sync` and `warp_relax`'s final
  `wp.synchronize` a no-op during capture. Host-blocking syncs are illegal inside capture,
  and the graph records stream ordering.
- The seed buffer address is stable; replay reuses the same buffer and reads the new seed
  contents on device.
- `output_mode="constant_spacing"` captures too: per-track `count[e]` is device-side data,
  and count-aware kernels keep static launch dimensions via `N_max`.
- Generator selection and `relax_enable` are Python branches resolved before capture. The
  captured graph is fixed for that `TrackGenerator`'s config.

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
  - **Single-pass first-stage generation replaced the regen loop.** With relaxation lossless,
    the default Bezier residual was a small fraction of smooth-centerline self-crossers.
    Rather than a fixed `max_regen_iters` accept-first-valid loop, the Bezier generator now
    takes one corner draw per env and routes any track whose smooth Bezier centerline
    self-crosses to its provably simple corner polygon, which XPBD re-rounds. The `hull`
    generator follows the same selected-polygon rescue pattern for its Catmull-Rom
    self-crossers, while `polar` emits a closed radial spline and relies on the common
    post-relax validity gate. `max_regen_iters` is therefore **vestigial** on the Warp path:
    it remains a `TrackGenConfig` field for the torch oracle but is ignored by `_run_pipeline`.
- **FP tolerance & hard thresholds.** Validity gates (`th_ok`, `turn_ok`) are hard
  comparisons; near a decision boundary the accepted ~1e-4 Warp-vs-torch drift can flip a
  single env's bool. Tests keep their inputs away from those boundaries; the end-to-end
  yield comparison uses an aggregate tolerance, not per-env equality.
