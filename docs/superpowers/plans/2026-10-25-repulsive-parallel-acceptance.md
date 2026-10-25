# Parallel Repulsive objective and bounded proposals

The repaired generator satisfies hard clearance but the exact GPU default batch
E64/N256 takes about188 seconds per warm graph, versus0.183 seconds for rc2.
Correctness remains binding: permanent filled-disc/domain certificates, actual
length targets, strict accepted energy decrease, true stopping diagnostics,
unchanged RNG and other generators, no hot allocation or host synchronization.

## Diagnosis and rejected first experiment

Root profiled the actual scalar body at N64/N128/N256 on GB10. At N256, base
energy takes0.10696 seconds, shape acceptance0.59651 seconds, growth0.01573
seconds, and final resample0.21999 seconds. The parallel gradient is0.002565
seconds. Heavy kernels launch one thread per environment; E64 with the default
block256 gives one CUDA block. The default base-energy schedule alone evaluates
447,492,096 directed edge pairs before candidate retries.

External-only block-size256/32/8/1 experiments preserve every final array byte.
Block1 improves N256 base energy only to0.09293 seconds and acceptance to0.53747
seconds. Launch tuning does not remove the long serial per-environment chain.
No tuning change was integrated.

## Accepted implementation scope

1. Add an isolated `repulsive_parallel_energy.py` with fixed `(env, owner-edge)`
   work, existing FP32 local primitives, deterministic FP64 row partials and a
   fixed row-order FP64 reduction. Four E*N double arrays use512KiB at E64/N256.
   No atomic sums. A cold immutable model holds constants and existing arrays.
2. Add isolated `repulsive_parallel_proposals.py` with preallocated masks,
   scalar O(N) perimeter/centroid preparation, per-edge double hard certificates,
   and fixed masked24 shape/16 growth trials. Keep the exact coordinate formula,
   fractions, first acceptable trial, strict Armijo and no-op/frozen/failure paths.
   Accepted or finished rows stop expensive work through device masks. No nested
   conditional feature or new execution-layer API is needed.
3. Compare helpers to the current scalar kernels before wiring. Main generator
   source remains unchanged until root approves. After approval, move unused
   scalar reference kernels to tests rather than ship a second runtime solver.
4. Root owns exact commits/archives and all GPU validation. First profile the
   isolated energy helper before full wiring; then profile candidate objective,
   acceptance, resampling and growth. A source performance claim requires a warm
   whole-generator measurement, not just a microbenchmark.

## Tradeoffs and numerical contract

A full parallel pair table plus original serial fold could preserve addition
order, but costs O(E*N*N + E*N*M) scratch (about48MiB at E64/N256). Root selected
O(E*N) row partials for racing-scale batches. This regroups FP64 additions and
may change Repulsive trajectories; no old Repulsive byte promise is made.

Both reductions obey the standard gamma_K times absolute-term magnitude bound.
Reference comparisons allow only the resulting double association budget, not a
new FP32 reduction floor. Existing independent near-stationary energy/sign tests
retain their original 1e-5 absolute requirement. Hard certificates and acceptance
inequalities are not weakened. Ordinary/capture/conditional execution of the
new implementation must still produce identical bytes on each device.

## Isolated validation progress

The missing parallel-energy feature first failed its N16 CPU test as intended.
The implemented helper now passes9 CPU tests;9 explicit CUDA cases await root.
Coverage includes N16/64/256, fractional proposals, admissible nondefault
exponents, derived summation budgets against the scalar objective, independent
FP64 near-stationary circle energy/sign, real-disc nonfinite rejection, masked
and zero-selected calls, poisoned partials, CPU capture replay, and no-allocation/
no-host-sync guards. The main numerical body remains SHA49990835.

The proposal helper and its tests are delegated to the child with exclusive new
file ownership. Real-disc per-environment offsets, swept certificates, special
paths and captured poisoned reuse are required before main wiring.

## Integrated validation and source hold

Root committed the isolated energy helper in `5fec7b3`. All 9 CUDA energy tests
passed without skips. Warm E64/N256 energy timing improved from 0.105961 seconds
to 0.000580158 seconds (182.64x); maximum energy difference was 5.86e-13 against
a derived 1.08e-9 association budget. E1 improved 200.54x. This is an isolated
energy result, not a whole-generator performance claim.

Root committed the proposal helper in `38a35eb`; the combined helpers passed 21
CUDA cases without skips. Before main wiring, an external diagnostic compared
cloned actual states from seeds 7/23/41: shape/growth at iterations 1,47,95,128,176,
189 and every resample (20 operation snapshots across N64/128/256). All compared
center/proposal/window/blocked/accepted bytes matched the scalar implementation,
with maximum difference 0. Ordinary final validity remained 1/1/1. Reproducer and
actual states: /private/tmp/trackgen-compare-parallel-proposals.py and
/private/tmp/trackgen-repulsive-parallel-real-states/.

Approved main wiring is committed as `58de72f`, runtime SHA
0028dc6ce25a42a54ceb53c8006d3cc3068736f17c0fda2f904eb3ce556b6c5e.
The cold model/workspace are allocated through StageBuilder, and the base
objective plus shape/growth/resample paths use the parallel helpers. The old
scalar operation-order oracle now lives only in
tests/_repulsive_serial_reference.py; production contains one implementation.
The canonical proposal coordinate formula is shared. No parameter, tolerance,
RNG, port, public result or other-generator behavior was intentionally changed.

Exact wired CPU checks completed:

- 52 unit/oracle/precision cases passed (33 CUDA cases skipped locally).
- 3 public integration cases passed: valid target-reaching seeds 7/23/41,
 no host synchronization, and actual CPU graph replay.
- 13 bounded cases passed, 1 slow case deselected: poll/unroll parity, stale flags,
 call-time cadence, mixed absorption, batch identity and replay.
- 120 scoped generation/clearance/effective-size/difficulty-size cases passed,
 75 deselected, in 119.40 seconds.
- 4 live Armijo precision cases passed (4 CUDA skips) after removing the obsolete
 duplicate total column; the original 1e-5 and sign assertions remain unchanged.

All source edits are held for root's fresh review, exact-archive whole-generator
GPU timing, capture/conditional validation and independent full-band audit.
No full-generator speedup, new yield baseline or release claim is made yet.
The implementer ran no GPU job and made no commit, push or tag.


## Held live GPU result and performance ruling

Exact commit `58de72f` passed 102 focused CUDA tests, with 269 CPU cases
deselected and no CUDA skips. The complete 64-row default batch took 7.586s
to construct and 3.464s for warm generation, versus 188.255s for the corrected
serial implementation. Every saved raw/band/count/valid/failure/barrier/gap
array was byte-identical to that held serial reference. All 53 reported-valid
roads independently cleared every disc and the wall.

The 1,024-row default batch took 12.599s to construct and 12.679s warm; 843
rows reported valid. Its first 64 raw curves and validity values were byte-
identical to the 64-row run. This is batching/performance evidence, not a new
independent physical audit of all 1,024 rows. The stronger objective and hard
certificates remain materially more expensive than rc2's 0.183s at E64.
Freeze this reviewed implementation for the required broader gates and expose
these actual costs for downstream racing validation; no latency guarantee or
universal convergence/yield claim is made.

A fresh independent review of `0d14a23..58de72f` found no actionable issues
in helpers, wiring, precision, masks, in-place writes, capture or scratch
lifetimes. All six moved scalar oracle bodies/decorators were AST-identical
to the baseline. The independent precision thresholds and sign checks remain
unchanged. Required full CPU/CUDA/full-GPU, cross-platform CI, non-Repulsive
byte guards and the next release candidate remain pending.

Mac RP fingerprints and every RP statistical summary value were unchanged
by parallelization; the reference revision records the held live source. No
other generator reference or statistical tolerance is refreshed.
