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
retain their original1e-5 absolute requirement. Hard certificates and acceptance
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
