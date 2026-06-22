# Voronoi Standard Generator Promotion - 2026-06-22

This note records the cleanup pass that promoted Voronoi from an experimental spike into a standard first-phase generator available through `generator="voronoi"`.

## Runtime Shape

- Added the Warp-native generator implementation in `track_gen/_src/warp_generate_voronoi.py`.
- Registered `voronoi` through the normal generator registry import path.
- Added `TrackParams` fields for the bounded Voronoi controls: `voronoi_num_sites`, `voronoi_site_layout`, `voronoi_control_points`, `voronoi_radial_variation`, and `voronoi_angular_jitter`.
- Wired the generator into the Gradio parameter explorer so it can be selected alongside the other standard first-phase generators.
- Documented the generator in the architecture, generator contract, baseline, and pre-relaxation generator-method notes.

The production path intentionally uses fixed-shape arrays and bounded kernels. It does not port general Voronoi/Delaunay construction, dynamic cycle-basis extraction, or unbounded graph walking into the runtime path.

## Distilled Sweep Result

The useful result from the Voronoi spike was not exact Voronoi ridge walking. The stable runtime idea is a density-biased site cloud, angular anchor selection, bounded smoothing, and deterministic resampling into the standard closed-centerline contract.

The promoted defaults use the `void_ring` layout because it gives less oval, more varied loop shapes while keeping deterministic finite output and avoiding dynamic topology in the generator.

## Verification

Focused generator/UI/type/registry run:

```bash
/home/antoiner/Documents/TrackGen/.venv/bin/python -m pytest \
  tests/test_generate_voronoi.py \
  tests/test_param_explorer.py \
  tests/test_types.py \
  tests/test_generator_registry.py \
  -q
```

Result: `29 passed in 4.00s`.

Full suite:

```bash
/home/antoiner/Documents/TrackGen/.venv/bin/python -m pytest -q
```

Result: `297 passed in 32.03s`.

Benchmark spot check:

```bash
/home/antoiner/Documents/TrackGen/.venv/bin/python -m benchmarks.compare_generators \
  --generators voronoi \
  --E 512 \
  --seed 0
```

Observed summary:

- `yield`: `1.0`
- `pre_relax_self_intersection_rate`: `0.0`
- `mean_compactness`: `0.7335`
- `shape_variety_pass`: `1.0`

Generated preview artifact:

- `viz/out/voronoi_standard_grid.png`

## Outcome

Voronoi is now treated as a standard, non-experimental first-phase generator. Future changes should extend the production `voronoi` path directly and keep the public contract aligned with `TrackParams`, the generator registry, Gradio controls, and the generator documentation.
