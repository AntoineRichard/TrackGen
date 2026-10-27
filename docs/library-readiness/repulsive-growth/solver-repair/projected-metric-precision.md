# Projected metric precision

Read-only follow-up reviewing double field storage and residual-quadratic
calculation. The runtime snapshot SHA-256 was:

```text
8d93bcfe870e6f8113555616d2582f8e8ce453bff04ec9fb9b042a21b088518f
```

This review did not run generation or GPU experiments. Its tiny independent
counterexample used NumPy linear algebra; the original script/measurements were
scratch-only and are not committed. The source snapshot identifies the reviewed
implementation and is not a claim that later source is identical.

## Stable same-model quadratic

Let `f` be the stored gradient, `l` the stored length gradient, `H` the actual
stored positive metric annihilating translation, `a = Hf`, `b = Hl`, and
`lambda = (l·a)/(l·b)` when the denominator is not clamped. In exact arithmetic:

```text
p = a - lambda*b
l·p = 0
r = P0(f - lambda*l)         # P0 subtracts the mean
p = Hr
q = f·p = r·p = r·Hr >= 0
```

Keeping convolutions, the projected field and its mean in double removes a
float32 subtraction floor, but `f·p` can still cancel a large removed normal
component against a tiny surviving field. Evaluate `r·p` in double. Mean removal
is required because barycenter pinning also removes translation, including a
possibly large obstacle force. For the PSD identity, a computed finite lambda
may be treated as fixed data; ratio accuracy/denominator floors remain separate
constraint-tangency concerns.

For a tiny independent SPD example, use `H = [[.8,.2],[.2,.3]]` (eigenvalues about
.229844 and .870156), `l = float32([.14159,-.27182])` and
`f = float32(127*l)`, then promote both before metric/projection operations:

| Quantity | Value |
| --- | ---: |
| `f·p` | −3.7380529579892144e−14 |
| `r·p` | +5.035469192941457e−15 |
| recomputed `r·Hr` | +5.035469136911603e−15 |
| `l·p` | −3.3398424406942337e−16 |

The tiny tangency residual times the large normal coefficient reverses the sign
of the full-gradient quadratic. Residual evaluation fixes this counterexample
without changing the model or accepting material negative curvature.

## Forward-error bound and limits

For `p_hat = p + delta_p`, `r_hat = r + delta_r`, norm bounds `B_p`, `B_r`, and
standard summation factor `gamma_k = k*u64/(1-k*u64)`, the reviewed bound is:

```text
input_norm = norm(f) + abs(lambda)*norm(l)
B_p = (4*gamma_(3N) + 4*gamma_N + 8*u64) * norm(h,1) * input_norm
B_r = (4*u64 + 2*gamma_N) * input_norm
B_q = gamma_(2N) * sum(abs(r_hat_i * p_hat_i))
      + norm(r_hat) * B_p
      + B_r * (norm(p_hat) + B_p)
```

It covers double difference-form convolution, projection subtraction and mean
removal under ordinary finite double arithmetic for supported normalized rows.
Crucially, field error scales with the **preprojection** input magnitude, not just
the tiny residual. FP32 gradient modeling error is separate from this algebraic
PSD calculation.

Illustrative convolution norm-bound factors were 3.60e−14 for N64, s=5/3 and
5.77e−13 for N1024, s=1.95; smallest nonzero filter eigenvalues were 4.325e−5 and
1.580e−10. Subtracting large convolutions can leave unresolved high-mode fields
even in double. Applying H directly to a double residual is an algebraically
equivalent possible accuracy improvement, not an adopted change requested by
this audit.

With imperfect tangency, `f·p = r·p + lambda*(l·p) + translation leakage`.
A residual metric norm therefore does not alone certify actual trial descent;
retraction and objective/geometry acceptance still apply. Material negative q
outside `B_q` must fail. Arithmetic uncertainty may support numerical stationarity
only with small unrounded field/motion, actual target attainment and independent
geometry checks. Rejected or quantized no-ops alone cannot establish convergence.

Portable coverage lives in
[tests/test_repulsive_metric_precision.py](../../../../tests/test_repulsive_metric_precision.py)
(actual stored spectra, high modes, translation/frozen rows, unresolved metric
rejection) and
[tests/test_repulsive_solver_clearance.py](../../../../tests/test_repulsive_solver_clearance.py)
(projected residual versus a large normal component, tau-independent
stationarity, false progress and legal near-circle integration). The synthetic
metric diagnostic alone does not verify generation, full convolution/retraction
behavior, or GPU execution; the lead owns those coordinated checks.
