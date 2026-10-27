# Repulsive repair review and integration gates

Fresh independent review covered commit `3e3000233dca68ad9cd1e6721b943742039ef1b2`
against immutable rc2 `a42759a3f999a8668a200cddc3f04e0508478dd1`. It inspected
all executable changes, tests, current contract docs and archival isolation. The
reviewer independently ran 32 focused CPU tests, checked source/artifact hashes
and recomputed stored raw perimeters. Root separately ran the combined 120-test
CPU focused suite. These are scoped implementation checks, not full release gates.

## P1: effective runtime size

The reviewer demonstrated a real validity bypass: size arrays are clamped at use
to the shared bounds, but the metadata publisher retained the raw value. With
seed 7, N256, growth 3–3.5, half-width 1, spacing .6 and size 10, array values
10 and 100 produced identical raw and finished-road bytes. The first correctly
failed disc clearance (mask 256), while the second incorrectly reported valid.
A small N16 lower-bound case also changed a correct valid result into wall
failure when the array was .01 instead of effective .1. Root independently
reproduced both public cases before editing the implementation.

The fix publishes the same comparison-clamped size as the geometry copy, using
shared SIZE_BOUNDS and preserving NaN propagation. Four correct regression cases
failed before the fix; five CPU cases pass afterward. The tests cover metadata,
finite out-of-range values, both infinities, NaN invalidity, actual public
geometry/diagnostics, matched masked epochs and capture replay without allocation
or synchronization. All four explicit CUDA cases passed in the first coordinated focused run.
That run exposed separate precision/yield integration gates, so it does not
clear the whole CUDA lane.
The first draft of the masked test incorrectly omitted advance=True; that test
error was corrected before the recorded final RED run.

Post-fix clearance-helper SHA-256: `e0f8d9b70fae0112da5f2696b34a8bf7b78c6bb3ef82c2c9de6e504547fae4e6`.
Portable regression: `tests/test_repulsive_effective_size.py`. No other actionable
source finding or deferred minor was reported. This is the single review fix pass;
its regression and whole-suite validation establish the correction.

## Disposition of declined review scope

- CUDA conditional execution, nested-array pointers and full GPU behavior remain
  required exact-archive integration checks owned by root.
- Full CPU, cross-platform CI and new Repulsive fingerprints/statistics remain
  required gates. Other generator references must not be refreshed.
- Universal convergence/yield/smoothness is not promised by bounded stochastic
  construction; finite feasible target-reaching geometry and mandatory final
  checks define success. Actual invalid rows remain invalid.
- The rejected per-edge experiment stays excluded and unimported. Its complete
  historical proof is not an adoption requirement.
- Local float32 energy primitives retain a measured finite-precision floor;
  double reductions/projection and geometric certificates do not imply exact
  arbitrary-precision optimization.
- Malicious plugins or external writes corrupting private memory are unsupported;
  ordinary public parameter updates, rewiring, masks and output views are covered.
- Unchanged elevation and generic alias/schema behavior remain covered by the
  required broader suite rather than a new independent algorithm proof.
- PR4 auto-spacing finding 4 stays deferred for the user's answer.
- The separately parked PR5 docs require reconciliation and Repulsive rerendering
  before integration; this minimal runtime documentation is not that visual pass.
- Downstream racing acceptance and explicit final-release approval remain with
  the user. No merge to main, final release or site deployment is authorized.

## Recorded implementation rulings

Geometric validity is separate from optional early-stop stationarity; root
corrected the earlier internal frozen-as-valid assumption. Permanent full-width
clearance motivated the 2–3 growth defaults and size-dependent reservation.
Per-edge projection was rejected on controlled generator outcomes. Convex-hull
cell coverage deliberately permits conservative rejection of unused concave
pockets to protect physical folded strips. Precision and parameter removals are
Repulsive-specific breaks. These decisions and their evidence are in the current
design, plan, migration guide and excluded solver-repair record.


## Integration tests and GPU performance gate

The first full CPU run completed with 3624 passes and four failures caused by
stale integration assumptions: requiring ordinary defaults to freeze, using the
frozen Repulsive tail as an epoch-zero oracle, omitting a construction-only
planner check from wrapper coverage, and asserting the old growth defaults.
Commit `0d14a23` corrects those tests without changing executable runtime code.
Focused checks passed: 141 defaults/wrapper tests, 163 epoch tests, one complete
CPU conformance case, and 28 tangent/reference tests. The conformance workload
uses supported fixed 1.1 growth and one disc; both fixed eight-row batches
genuinely freeze and skip a window, retaining all original assertions.

The first CUDA default batch returned 53/64 valid roads. Every valid road
passed the independent float64 hull/disc and circular-wall reference; minimum
gaps were .002527 and .193317. The rc2 comparison flagged all 64 valid but
only one road passed the same physical full-band check (its analytic discs
were reconstructed from opposite force-ring samples). The obsolete 58/64
flag gate was replaced by an explicit 75% fixed-batch yield floor plus an
independent check of every returned road. This is not a universal yield claim.
The unchanged compactness gate also passed on the measured batch, median .546.

A separate fresh independent review of only `0d14a23` found no actionable
issues. It checked the conservative hull reference, retained direct-seed
comparison, nonvacuous absorbing check and construction-only exemption. It
accepted the mesh/force-scaled fp32 translation-balance tolerance while noting
that gamma is a practical regression tolerance, not a rigorous error bound
for every intermediate derivative operation. The independent component-wise
finite-difference and energy checks retain their original tolerances.

The warm GPU generation cost of 188.255 seconds is a release blocker. A
three-resolution kernel profile isolates serial full-energy evaluation and
line-search work. Launch-only changes preserved bytes but improved these
kernels by only about 10%; parallel evaluation/search is being implemented
and must receive its own focused proof and full integration validation.
The c017fd0 candidate snapshot was deliberately stopped after this discovery;
no passing complete snapshot comparison or full GPU suite is claimed.


## Parallel execution review

The performance fix at `58de72f` received a fresh independent review against
`0d14a23`, with no actionable findings. It covered the complete objective,
FP64 association budget, swept geometry, masks and frozen/no-op/failure
paths, first-accepted trial retention, in-place writes, preallocation, stage
scratch naming/lifetimes and live context-driven wiring. The six sequential
oracle functions moved into tests are AST-identical to their prior bodies.

Exact-archive isolated helpers passed 21 CUDA tests; the wired selection
passed 102. The complete 64-row run retained all geometry and diagnostic
bytes from the corrected serial reference, reducing warm time from 188.255
to 3.464 seconds. The 1,024-row run measured 12.679 seconds and retained
its first 64 raw/valid rows byte-for-byte. These scoped results do not
substitute for the pending full validation lanes and snapshot comparisons.
