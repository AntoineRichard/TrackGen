# Independent Repulsive roughness audit

2026-10-25, macOS arm64, CPU only. This audit did not edit any repository source, index, branch, or GitHub state, and did not use a GPU. Numerical variants are process-local monkeypatches in scratch scripts. The solver implementation was developed separately; these findings were communicated during that work.

## Result

The raw jaggedness is a genuine numerical/algorithmic problem. It is not adequately explained as expected polygonal appearance, and the evidence does not support blaming the analytic gradient. There are **two independently demonstrated method mismatches** plus an unchecked update/stopping policy:

1. **The ring Sobolev filter has half the intended differential order.** Correcting only this factor substantially reduces raw turns on all three tested seeds.
2. **The vertex-tangent energy underpenalizes the grid-scale zigzag mode relative to the paper's edge quadrature.** Its gradient is correct for that surrogate, but that does not make the surrogate identical to the paper's discrete energy.
3. **A fixed normalized step is accepted without energy/feasibility checks, area stall is treated as convergence, and resampling can sharply change the objective.** Backtracking is necessary but, with the old metric/objective, was insufficient to remove the roughness.

These findings explain why the user's intuition is justified. They do not establish that every fold or transient corner is erroneous: growth under confinement legitimately buckles and folds. The measured reversals, objective-order mismatch and large excess turns are the stronger evidence.

## Baseline integrity and reproduction

Baseline runtime corresponds to immutable RC2 commit `a42759a3f999a8668a200cddc3f04e0508478dd1`; the original audit read docs head `4aec44a8f30d7ef998bb37ce6cb86e6ad2d12bd2`. Existing unrelated documentation changes in that checkout were not audit edits. Source hashes were checked before and after the experiments:

```text
track_gen/_src/warp_generate_repulsive.py
f87c497dbd86aedfdd4f8987f2feda546f78fb02ca165d274fadb866d0b41570
tests/test_generate_repulsive.py
baaf1e644be09466bece4172bd3f98de56c64b915fe1d710f1d0e69ef6edcb98
```

Python 3.12.14, Warp 1.18.0, NumPy 2.5.4, macOS 26.7.1. `roughness-audit-diagnostics.json` preserves the original environment, source hashes, scalar diagnostics and compact variant summaries. A fresh baseline rerun was byte-identical to the prior investigation's raw and finished centerline arrays:

```text
raw float32 array SHA256:
1f9ee92f93a51c873fdd760e2d9612f524a0ed5b6900cb2e43265cb2b80af235
finished centerline SHA256:
52b92b844e1dead204ec6ccd386de2c26f61f08923804e2d965a4dbc32be33ba
```

The portable tool is `roughness_audit.py` in this directory, alongside the previously preserved `reproduce.py`/`metrics.py` capture helpers. It requires Python and NumPy for synthetic calculations, plus Warp for original-kernel state checks and end-to-end variants. It does not require the test suite's pytest/torch imports: the small original analytic-kernel harness is included directly.

Run the tool from the records checkout, pointing `--repo` at an **immutable RC2 export**. This prevents a future corrected checkout or concurrent implementation edits from silently becoming the audit baseline. For example, from the repository root:

```sh
mkdir -p /tmp/trackgen-rc2-audit-source
git archive a42759a3f999a8668a200cddc3f04e0508478dd1 | tar -x -C /tmp/trackgen-rc2-audit-source
python docs/library-readiness/repulsive-growth/roughness_audit.py smoke \
  --repo /tmp/trackgen-rc2-audit-source --out-dir /tmp/trackgen-roughness-audit
MPLCONFIGDIR=/tmp/trackgen-repulsive-mpl WARP_CACHE_PATH=/tmp/trackgen-repulsive-warp \
  python docs/library-readiness/repulsive-growth/roughness_audit.py capture \
  --repo /tmp/trackgen-rc2-audit-source --out-dir /tmp/trackgen-roughness-audit
MPLCONFIGDIR=/tmp/trackgen-repulsive-mpl WARP_CACHE_PATH=/tmp/trackgen-repulsive-warp \
  python docs/library-readiness/repulsive-growth/roughness_audit.py states \
  --repo /tmp/trackgen-rc2-audit-source --out-dir /tmp/trackgen-roughness-audit
```

`capture` creates `OUT/captures/baseline.json` and `.npz`; later commands read them. Existing compatible captures can instead be supplied with `--states-dir`. Commands `modes`, `paper_discretization`, `spectra`, `cancellation`, `weighted_obstacles`, and `stage_jumps` reproduce the corresponding isolated checks. `smoke` is a small deterministic NumPy check and imports no runtime or Warp.

For an individual end-to-end comparison, retain the same explicit `--repo` and `--out-dir` arguments and use `filter --variant order --seed 7`, `filter --variant order --seed 23`, `filter --variant order --seed 41`, or `filter --variant order --seed 7 --keep`. Other recorded isolated variants are `filter --variant normalized_dc0 --seed 7 --keep`, `filter --variant normalized_dc0 --seed 23 --keep`, `backtrack --variant base --seed 7`, and `backtrack --variant normalized_dc0 --seed 7 --keep`. `--keep` retains inner-disc forces. `filter --variant order` changes only the exponent from5/6 to5/3. No values were selected by a broad tuning sweep.

Both source-file hashes above are checked before dispatch and again after execution. Runtime imports must resolve inside `--repo`; an archived tree needs no Git metadata. `--allow-source-difference` explicitly opts into auditing another revision, in which case the RC2 reference identities no longer establish baseline equality. Numerical variants are process-local monkeypatches, never source edits. Scalar output goes under `--out-dir`, and generated trajectory JSON/NPZ pairs go under its `captures/` directory. No trajectory datasets or large backtracking event logs are committed. Every capture checks ordinary-versus-observed raw array byte equality under the same process-local variant; its array hashes distinguish raw geometry from the returned road.

The preservation step syntax-checked this portable tool, checked its CLI, and ran only its deterministic `smoke` command. It did not repeat the original experiments or use a GPU. The preserved scalar diagnostics come from the original audit; the new normalized mode-spectrum columns are algebraic transformations of the recorded Hessian values, not additional trajectory runs. End-to-end process-local variants were not rerun while making the tool portable.

## 1. Filter order: confirmed mismatch and causal experiment

Production `_alloc_repulsive_scratch` uses `s_exp=(beta-1)/(2*alpha)` and `_sobolev_circulant_row` applies `1/(lambda_k**s_exp + 1e-3)`, where `lambda_k=2-2*cos(2*pi*k/N)` is a Laplacian eigenvalue. A Laplacian already has differential order two. Therefore the implemented operator has order `(beta-1)/alpha`, rather than `2*(beta-1)/alpha`.

The primary paper defines `s=(beta-1)/alpha` and an inner-product operator of order `2s` (§4.2.2, Appendix A). Its timestep method uses backtracking with constraint projection and a Sobolev-gradient stopping tolerance (§5.4). Its discrete energy uses edge tangents and disjoint-edge trapezoidal quadrature (§5.1). Its obstacle potential integrates over both the curve and obstacle (§8.1). These distinctions matter when assessing fidelity. [Yu, Schumacher and Crane, Repulsive Curves](https://arxiv.org/html/2006.07859).

The earlier spike `docs/superpowers/spikes/2026-06-17-relaxation-bakeoff/relax_tpsobolev.py:21` explicitly claims `2s=(beta-1)/alpha`, showing the likely origin of the factor-of-two mistake. It is documented as a surrogate; the incorrect differential order is separate from the deliberate use of a ring surrogate.

Only doubling the exponent, keeping epsilon, tau, growth, resampling and stopping unchanged, gives:

| Seed | Baseline raw max / p95 turn | Exponent-only raw max / p95 turn | Baseline / corrected max turn over all iterations |
|---|---:|---:|---:|
| 7 | 114.97° / 47.76° | 31.09° / 15.59° | 175.43° / 53.81° |
| 23 | 100.26° / 33.79° | 35.02° / 17.72° | 136.03° / 61.98° |
| 41 | 91.02° / 52.77° | 40.76° / 19.64° | 179.97° / 47.22° |

Seed 7 freezes at 176 rather than 192 and finishes with raw perimeter 12.50455 rather than 12.22229 (target 12.51645). Wall centerline clearance remains +0.61056. Default obstacle deactivation still permits disc penetration. This is a strong isolated causal effect, not a complete fix or a three-seed guarantee.

A finite-difference fixed-perimeter radial-mode test on the actual vertex objective independently supports the order diagnosis. At N=256 around a unit circle, normalize `lambda` by `lambda_1`. For modes k=2,4,8, the constrained Hessian eigenvalue divided by `lambda**(5/6)+eps` is 0.1733, 0.6927, 1.9162. With exponent 5/3 it is 0.05460, 0.06878, 0.06004. The corrected exponent substantially flattens the low/moderate-frequency stiffness, as intended by preconditioning. This is an empirical local spectrum, not a proof of the full nonlinear metric.

### Resolution and precision require care

Fixed epsilon plus unnormalized eigenvalues changes the filter shape with N. At N=256 and exponent 5/3, several low modes fall into the epsilon plateau. A dimensionless `lambda/lambda_1` with DC removed avoids this resolution coupling; coordinate length factors common to the metric cancel under normalized stepping.

However, **do not normalize while keeping the DC gain at 1000**. Casting the real-space row to float32 then creates 19 negative nonzero spectral eigenvalues at N=256; minimum −1.36556e−6 versus intended positive minimum 4.26249e−7. Setting spectral DC to zero before `irfft` gives all positive nonzero modes, minimum 4.22679e−7, with maximum relative spectral error 1.09%. The existing unnormalized baseline row remains positive and has maximum relative spectral error 2.24e−5. This precision hazard belongs to the proposed normalization, not the existing baseline bug.

Normalized/corrected/DC-zero with retained disc forces gives seed 7 max/p95 27.30°/17.18°, but seed 23 **escapes the domain**, wall gap −0.34534. Thus clean spectral math does not replace feasibility enforcement. Seed 7's common road tail also moves its road band into several discs despite positive raw centerline gaps; the finished-road contract still needs its own check.

## 2. Correct gradient, weaker discrete objective

The existing `test_analytic_gradient_matches_reference` covers the full TP pair, tangent and dual-weight couplings, obstacle force, and length penalty on five randomized N32/64 states. It is not merely an obstacle self-term test. The CPU test was rerun: **1 passed, 1 CUDA skip**.

The independent `states` check tests both TP-only and full gradients against float64 directional central differences on actual baseline iterations 1,47,128,176,191, including severe roughness. Maximum relative directional errors respectively were 2.67e−5, 3.97e−6, 1.45e−6, 2.30e−6, 7.20e−7. These are consistent with float32 evaluation. No wrong sign or missing ordinary gradient coupling was found.

The energy itself is a different discretization. Centered tangents `normalize(x[i+1]-x[i-1])` exactly cancel an alternating coordinate perturbation before normalization. The exclusion of circular index distances <=2 additionally removes the closest interactions. A fixed-perimeter mode experiment compares the current energy with an independent implementation of the paper's edge-tangent trapezoidal quadrature (same alpha,beta and epsilon; only tangent/quadrature/exclusion choices change):

| N256 radial mode | Current vertex constrained Hessian eigenvalue | Edge-quadrature constrained Hessian eigenvalue |
|---|---:|---:|
| k=2 | 0.5503 | 0.6037 |
| k=4 | 6.979 | 8.075 |
| k=8 | 61.16 | 79.45 |
| k=32 | 2352.84 | 6196.91 |
| k=64 | 4882.68 | 42297.06 |
| k=128 (Nyquist) | 978.78 | 119095.56 |

The current stiffness drops fivefold from k64 to k128 while the edge-quadrature stiffness rises. The Nyquist penalty is about 122 times smaller in the current discretization. All tested circle eigenvalues remain positive: this experiment does **not** prove that circle wiggles lower energy or that this alone creates the actual folds. It proves a substantial high-frequency regularization difference that a passing gradient test cannot detect.

Implementing and validating edge quadrature would be the more faithful objective, with new analytic adjoints and higher per-pair work. Correcting the filter and acceptance is a smaller justified fix, provided the remaining vertex/ring approximation is stated honestly and tested for roughness rather than advertised as the full paper method.

## 3. Length projection, objective scaling and obstacle derivative

The length-gradient sign is correct (`u_back-u_forward`), and the Sobolev projection formula is correct for the SPD ring metric. Translation removal commutes with this projection in exact arithmetic because the length gradient sums to zero and a circulant maps constants to constants. The convolution orientation and `irfft` normalization are correct; there is no extra missing N factor.

The existing length penalty is mathematically annihilated by the subsequent projection. If `l=grad L`, its contribution is `c*l`; after preconditioning and projection:

```text
P_A A^-1(c*l) = c*A^-1*l - [c*<l,A^-1*l>/<l,A^-1*l>]*A^-1*l = 0.
```

On the five recorded states with a 1.2% target change, changing only the length penalty alters the projected direction by 7.5e−9 to 2.2e−7 relative, attributable to finite precision. A length-only gradient projects to norm ~3e−15…7e−14 in float64. Seeded outputs can still diverge over long trajectories from those rounding differences; that does not make this a mathematically active shape regularizer. Omitting it from a fixed-length shape Armijo objective is appropriate.

Current obstacle energy lacks curve dual length: `Eobs=sum_i O_i` versus TP's `w_i*w_j`. Refinement approximately doubles obstacle importance relative to TP. Scaling all coordinates and obstacle geometry by c scales TP energy as `c^(alpha-beta+2)`, but the current obstacle energy as `c^(alpha-beta+1)`, ignoring fixed epsilons. This changes the intended system as resolution or world scale changes.

For `Eobs=sum_i w_i O_i`, the implementation agent's proposed derivative is correct:

```text
grad_v Eobs = w_v * grad O_v
 + 0.5*u_back*(O_prev+O_v)
 - 0.5*u_forward*(O_v+O_next).
```

Independent float64 directional finite differences at random N32/64 and actual rough N256 verify it with maximum relative error 1.42e−9 / 1.78e−9 / 4.89e−9. Scaling geometry and obstacle masses by 3 gives energy ratio ~1/3 at default exponents, matching TP. Fixed epsilons introduce small departures; they should scale consistently if strong scale invariance is claimed. This weighting correction changes the intended discrete objective, not a faulty derivative of the existing objective.

## 4. Steps, resampling and stopping

At actual rough iteration 47, the current projected direction is a descent direction (`grad·d=-51.97` for a unit tau parameter), yet after retraction to the same perimeter tau=.4 raises total energy by +0.235; tau=.2 decreases it by −1.256. This directly demonstrates finite-step overshoot. Other sampled late states can decrease total energy while their maximum turn increases. Global objective decrease alone does not establish smoothness or stationarity.

The scratch `backtrack` variant tries eight steps `.4/2**j`, accepts an Armijo decrease at a common ratcheted target, and keeps the metric otherwise unchanged. Seed 7 still finishes rougher, max/p95 137.77°/57.97°. This is intentionally an isolation test, not the recommended final algorithm: forced growth is compared to a radially rescaled baseline, which may itself be infeasible. It shows why an Armijo-only patch is insufficient. Correct metric + normalized/DC-zero + retained obstacles + backtracking gives 27.56°/16.87°, still with finished-road disc overlap.

The existing last-window default displacement cosine −0.7628 was reproduced from exact saved coordinates. Corrected exponent-only seed 7 still has −0.3207. Fixed normalized stepping need not converge even after visual quality improves.

Actual resampling operations are not harmless reindexings of a smooth curve. Exact intercepted boundary energies:

| Operation | TP before → after | Obstacle before → after |
|---|---:|---:|
| Same-N resample, iteration 50, N64 | 49.26 → 98.94 | 306.68 → 346.11 |
| Stage 64→128, iteration 60 | 64.62 → 1976.78 | 328.20 → 702.23 |
| Stage 128→256, iteration 118 | 244.12 → 612.21 | 857.73 → 1793.29 |
| Same-N resample, iteration 175, N256 | 320.67 → 315.10 | 994.35 → 1025.25 |

These are recorded around the original kernels. The arc interpolation is linear. At stage changes the objectives have different quadratures/resolutions, so the ratios are not themselves an Armijo failure; they show that the optimization problem changes abruptly and convergence histories must reset. At fixed N, resampling may also increase the objective and shorten perimeter.

Default freeze at relative area change 0.04474 <0.05 and fixed iteration 192 is not a stationarity certificate. Tightening area tolerance and increasing the bound alone was already measured to worsen raw turns. The absorbing freeze and fixed host schedule are correctly implemented; the criterion is weak, not the masking/capture mechanism. All ordinary/default GPU-independent math issues already reproduce on CPU, so GPU noise is not a necessary explanation. CPU/CUDA trajectories can differ; no cross-device robustness claim was tested here.

## Recommended update and convergence policy

1. Use the corrected metric order, with a dimensionless spectral scale and DC removed before float32 row construction. Verify positive nonzero Fourier modes of the realized row for supported N/exponents; compare actual convolution against the spectral action on representative modes.
2. At the current accepted perimeter, evaluate a projected shape direction and attempt a fixed maximum number of backtracking candidates. Test the energy of the **retracted candidate**, with finite values and all geometric feasibility predicates. Use Armijo based on the derivative of the same objective; keep the old accepted shape if all trials fail. Finite bounds and per-env accepted flags fit existing capture semantics without host polls or per-step allocation.
3. Perform forced growth as a separate feasible constraint update; do not claim it is an energy descent step. Backtrack its growth fraction if needed and retain the last feasible state. A target ratchet reaching its upper bound is not proof that the actual curve reached that perimeter.
4. Check feasibility again after periodic reparameterization, stage transition, final resampling and the common road tail. An accepted shape step cannot protect geometry changed later. Reset movement/energy history on resolution/reparameterization events and on active-obstacle-set changes.
5. Freeze only after the actual target condition plus several stable accepted checks. Observe shape displacement (relative to mean edge), energy change and a meaningful projected-gradient or constrained-stationarity residual. If all candidate steps are rejected, distinguish blocked progress from convergence; zero accepted movement alone is not sufficient. Keep bounded termination separate from output geometric validity.
6. Correct the obstacle curve weighting with its complete derivative if resolution/scale consistency is part of this fix. The derivative formula above has an independent finite-difference check. Separately assess edge-quadrature energy if grid-scale roughness remains; adding cosmetic smoothing would conceal rather than resolve that discrepancy.

This audit does not prescribe a universal tau, barrier margin, growth budget or convergence tolerance. Those interact with finite road width and final-tail displacement and must be validated with the implementation's hard-clearance contract. The reported filter-only variants deliberately demonstrate why selecting the smoothest-looking seed or tuning alone is unsafe.

## Scope and remaining uncertainty

- Three explicit seeds support the exponent-only causal comparison; they are not a yield study.
- The independent FD checks establish the tested derivatives on tested states. They do not cover degenerate zero-length centered stencils, exact absolute-value kinks, arbitrary accepted exponent values, or every Warp/device precision path.
- The ring metric remains a surrogate for the geometry-dependent Sobolev–Slobodeckij metric, even after correcting its exponent.
- The edge-quadrature test measures a local spectrum. A complete production replacement needs its own adjoint, capture, performance and output-quality validation.
- The separately developed hard finished-road clearance implementation and GPU validation were not modified or certified by this read-only audit.
