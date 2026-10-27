# Repulsive solver repair: evidence record

This internal record captures the 2026-10-25 investigation and repair after
`v0.3.0rc2`. It is excluded from the public documentation build. The binding
requirement is permanent clearance of every sampled filled disc and the outer
circular wall by the entire finished road band, after the shared tail. A disc in
the global loop hole is allowed. Invalid output follows the existing poison/keep
policy; no new public result fields are introduced.

- [Upstream provenance and port differences](upstream-provenance.md) separates
  historical prototype behavior from the present contract.
- [Acceptance energy and length precision](energy-and-length-precision.md)
  records genuine descent, false FP32 acceptance and the retraction floor.
- [Projected metric precision](projected-metric-precision.md) records the
  cancellation counterexample and the residual quadratic/error bound.
- [Finished-band certification](finished-band-certification.md) records the
  folded-cell gap found in the new certifier during this repair and its
  conservative convex-hull replacement.
- [Geometric arithmetic precision](geometric-arithmetic-precision.md) records
  scaled small-disc and near-wall failures and the float64 certificate repair.

The chronological distinction matters. The rc2 solver used a half-order fixed
metric, normalized steps, total-length rescaling and area-only stopping. During
its replacement, separate CPU audits found scalar-energy precision loss and
projected-metric cancellation. An adversarial review also found a folded-strip
coverage gap in the **new** final-band certifier; this was not a demonstrated rc2
seed reproduction. The rejected per-edge constraint experiment is already
archived [in the sibling record](../per-edge-projection/README.md) and is not
production code.

A finite feasible raw curve reaching its actual target can be valid at the
bounded iteration budget without optimizer stationarity. Frozen is an optional
absorbing early stop requiring small motion, projected stationarity and area
change. A rejected or quantized no-op alone does not establish convergence.
Full final-road validity remains mandatory. The [migration guide](../../../upgrading/from-0.2.rst)
records the changed outputs, 2.0–3.0 growth defaults, removed knobs and size behavior.

## Portable verification and limits

The audit source snapshots and JSON/NPZ coordinate datasets were scratch-only
artifacts and are **not committed**. The notes preserve their source hashes and
measured conclusions, not dependencies on the original machine. Repository tests
provide portable synthetic fixtures and independent numerical/geometry checks:

```sh
PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  -m 'not cuda' \
  tests/test_repulsive_tangent_energy.py tests/test_repulsive_barrier_energy.py \
  tests/test_repulsive_armijo_precision.py tests/test_repulsive_metric_precision.py \
  tests/test_repulsive_solver_clearance.py tests/test_repulsive_final_clearance.py \
  tests/test_repulsive_clearance_pipeline.py tests/test_repulsive_bounded.py
```

Use a configured project test environment and a fresh checkout of the repair.
These files include CUDA-marked cases, but their presence does not establish a
GPU pass. The numerical and adversarial evidence summarized here was obtained on
CPU (Warp 1.18.0 where recorded), with CUDA cases skipped. The final 16-seed proof,
coordinated CUDA/capture validation, broader release validation and downstream
racing validation were still pending when this record was written. No population
yield, runtime or universal seed-success claim follows from these local checks.
