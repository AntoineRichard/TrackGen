# Generator Baseline Metrics

Reference metrics for the registered first-stage generators, produced by
`benchmarks/compare_generators.py`. New methods are reported against this table; it
characterizes tradeoffs (quality / diversity / speed) and never gates which generators
ship — every registered generator stays selectable via `config.generator`.

Suite: seed base 0, E=4096, default `TrackGenConfig` (cpu). Regenerate with
`.venv/bin/python -m benchmarks.compare_generators --E 4096 --seed 0`.

| generator | yield | pre_relax_self_intersection_rate | xpbd_displacement | mean_length | mean_compactness | peak_curvature | lap_time | gen_ms_per_call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bezier | 0.9937 | 0.006348 | 0.0484 | 5.061 | 0.4415 | 8.79 | 10.14 | 1.41e+04 |

Sanity row for the new standard Voronoi generator, generated with `--generators voronoi --E 512 --seed 0` on CPU:

| generator | yield | pre_relax_self_intersection_rate | xpbd_displacement | mean_length | mean_compactness | compactness_p50 | compactness_degenerate_rate | shape_variety_pass | mean_chicanes | straight_frac | peak_curvature | lap_time | gen_ms_per_call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| voronoi | 1 | 0 | 0.01315 | 4.27 | 0.7335 | 0.7318 | 0 | 1 | 13.88 | 0.2185 | 8.205 | 7.849 | 860.4 |
