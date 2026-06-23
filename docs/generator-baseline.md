# Generator Baseline Metrics

Reference metrics for the registered first-stage generators, produced by
`benchmarks/compare_generators.py`. New methods are reported against this table; it
characterizes tradeoffs (quality / diversity / speed) and never gates which generators
ship — every registered generator stays selectable via `config.generator`.

Suite: seed base 0, E=512, default `TrackGenConfig` on the Warp `cpu` device. Regenerate
with `.venv/bin/python -m benchmarks.compare_generators --E 512 --seed 0`. Use a larger
`E` for release-grade timing; CPU timings below are machine-dependent and intended for
relative comparison only.

| generator | yield | pre_relax_self_intersection_rate | xpbd_displacement | mean_length | mean_compactness | compactness_p50 | compactness_degenerate_rate | shape_variety_pass | mean_chicanes | straight_frac | peak_curvature | lap_time | gen_ms_per_call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bezier | 0.9922 | 0.005859 | 0.05008 | 5.056 | 0.4387 | 0.4238 | 0 | 1 | 13.21 | 0.2295 | 8.762 | 10.16 | 794 |
| checkpoint | 0.9766 | 0.001953 | 0.03541 | 4.224 | 0.6142 | 0.6225 | 0 | 1 | 14.8 | 0.1991 | 8.577 | 8.549 | 1125 |
| hull | 0.9941 | 0.007812 | 0.06167 | 4.958 | 0.4254 | 0.4111 | 0 | 1 | 14.13 | 0.1919 | 8.908 | 10.52 | 839.5 |
| polar | 1 | 0 | 0.0271 | 4.736 | 0.5586 | 0.5524 | 0 | 1 | 12.72 | 0.1978 | 8.44 | 9.488 | 568 |
| voronoi | 1 | 0 | 0.01315 | 4.27 | 0.7335 | 0.7318 | 0 | 1 | 13.88 | 0.2185 | 8.205 | 7.849 | 736.9 |
