# First-Stage Generator Contract

A generator produces the initial closed centerline that the pipeline then resamples,
relaxes (XPBD), and inflates. To add one, implement two callables and register a
`GeneratorSpec` (see `track_gen/_src/generator_registry.py`).

## What you implement

- `alloc_scratch(config) -> scratch` — allocate your generator's PRIVATE working buffers
  ONCE. Fixed shapes derived from `config` (e.g. `num_envs`, `max_num_points`,
  `num_points_per_segment`, `num_points`), all on `str(config.device)`. Return any object
  exposing the buffers your `generate` uses. Do NOT allocate the output centerline/valid
  here — the orchestrator owns those.
- `generate(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None`:
  - `seeds_wp`: `[E]` int32 wp.array, one base seed per env.
  - `out_centerline`: `[E*num_points]` `wp.vec2f` — write a CLOSED centerline of
    `config.num_points` points per env, in place.
  - `out_valid_wp`: `[E]` int32 — write 1 for envs that produced a usable centerline.
  - `scratch`: the object your `alloc_scratch` returned.

## Current standard generators

The standard runtime generators are `bezier`, `polar`, `hull`, and `voronoi`. The Voronoi
method is implemented as a fixed-budget site-field / graph-cycle generator rather than exact
Voronoi ridge walking; exact Delaunay/Voronoi construction remains an offline diagnostic until
it can satisfy the same fixed-shape Warp contract.

## Hard rules

- Pure Warp kernels (`wp.launch`), one env per row. NO torch in `track_gen/_src`.
- Zero dynamic allocation inside `generate` (all buffers come from `alloc_scratch`).
- CUDA-graph capturable: no host sync, no host-side retry loop conditioned on generated
  data, no per-env Python branching inside `generate`.
- Fixed bounds for every loop/buffer (graph capture needs static shapes).
- Deterministic in `(per-env seed, config)`; use the Warp RNG (`track_gen._src.rng_*`).
  cpu vs cuda RNG may differ (as elsewhere).

## What you do NOT have to guarantee

- A simple (non-self-intersecting) loop is preferred but not required — XPBD repairs
  thickness and the polygon fallback handles self-crossings. Output must be finite (no NaN)
  for valid envs.

## How a generator is judged

Run `benchmarks/compare_generators.py` (see docs/generator-baseline.md). Generators are
characterized, never gated: a method that scores worse on yield but better on speed or
style stays selectable.
