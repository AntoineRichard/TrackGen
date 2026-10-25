# Acceptance energy and length precision

CPU audit, Warp 1.18.0, of runtime `track_gen/_src/warp_generate_repulsive.py`
with SHA-256:

```text
78d8d31a848cf3595188199bd06806d49d671c2f5c9fbca62b60ebd2f680f443
```

The source did not change during the recorded circle run. An immutable source
copy, kernel-interception probe and JSON/NPZ states were kept as scratch-only
audit artifacts; they are not committed. The probe compared scalar FP32 objective
reduction, a double sum of the same FP32 primitive terms, and an independent
NumPy FP64 evaluation of the same TP/ring/wall/segment-barrier model.

## Separate genuine descent from false acceptance

Short-target fixtures used seeds 7/23/41, tau 0.4, growth 3.0–3.5 and `keep`.
The circle fixture used seed 7, N64/stages `(64,)`, the same growth/tau, one disc
slot with radius fractions 0.8–0.9 (empty placement band, inactive force), active
wall radius 1.716015 and reserved margin 0.1.

| Observation | Scalar FP32 ΔE | Double sum of FP32 primitives | Independent FP64 ΔE |
| --- | ---: | ---: | ---: |
| Seed 23, iteration 210 | −0.0237732 | −0.0232591 | −0.0232587 |
| Circle, iteration 95 | −3.05176e−5 | +1.37525e−4 | +1.37508e−4 |
| Circle, iteration 210, rejected trial | +1.52588e−4 | +8.89557e−5 | +8.67472e−5 |

All 10 late accepted steps of seeds 23/41 passed Armijo in FP64, with ΔE about
−0.023 to −0.258. Reduction error reached 0.00216, while local primitive error was
at most 1.74e−6: precision noise did not explain their substantial ongoing motion.
Seed 7's late trials crossed a barrier and were correctly rejected.

The circle had **35 false Armijo acceptances among 94 accepted steps**. Maximum
scalar-reduction delta error was 3.19e−4, versus local primitive error 2.09e−6.
This supports double objective accumulation without claiming that all stored
FP32 geometry or primitive error disappears.

## Retraction has a different precision floor

At the circle's iteration 210, independent ΔE was +8.67472e−5 and
`gradient · displacement` was +8.67347e−5. The stored candidate perimeter drifted
by +8.53788e−7 (1.03e−7 relative). With length multiplier about 104.8625,
`lambda * ΔL = +8.95304e−5`; subtracting this first-order constraint contribution
left about −2.78e−6 of shape improvement. Tiny retraction residuals can dominate
near-stationary constrained energy, independently of scalar reduction error.
This motivated double perimeter/retraction arithmetic. It does not make rejected
searches evidence of convergence.

## Portable regressions and recorded outcome

[tests/test_repulsive_armijo_precision.py](../../../../tests/test_repulsive_armijo_precision.py)
constructs N64 circles and a 96-point wall analytically, at phases 0 and 0.173.
Its second candidate is a deterministic `2^-22` radial expansion; no RNG or saved
coordinate dataset is needed. It checks absolute energy and nearby ΔE against an
independent FP64 formula at `1e-5` absolute tolerance.

The pre-fix red run had 4 CPU failures (4 CUDA cases deselected): maximum energy
error 5.91e−4; measured candidate changes 0.000244140625 and 0.000289916992 versus
FP64 0.000208233133 and 0.000209424392. After the precision change, the unchanged
regression recorded **4 CPU passed, 4 CUDA skipped**. No tolerance or stopping
threshold was relaxed. Perimeter, exact satisfied-length no-op and false quantized
progress tests are in
[tests/test_repulsive_solver_clearance.py](../../../../tests/test_repulsive_solver_clearance.py).
These are focused historical CPU results, not final generator/GPU certification.
