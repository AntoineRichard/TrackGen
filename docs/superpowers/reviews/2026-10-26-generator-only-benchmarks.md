# Generator-only benchmark boundary review

The user's correction is binding: the chooser compares generator construction;
shared XPBD and other road processing do not belong in that comparison.

## Frozen measurement operation

Runner source: `c920101ed61ed02a96fe248ceab2a10e5eba8c68`.
Runner SHA256: `b6c3bb4f266741f10f6a61f23e5958b4d8fade28ad43cfee40e98fadb3deab6a`.
Runtime library and package remain identical to immutable `v0.3.0rc3`.

The benchmark constructs the ordinary facade outside timing, preserving resolved
parameters and Repulsive's existing width reservation. The benchmark body then
calls only seed advance, effective key derivation, the generator parameter latch
and the already-bound generator slot. Its `_SlotEnqueue` retains BOUNDED execution
and conditional graph behavior. No shared road stage or result commit runs.

A generator-only public PipelineSpec is intentionally invalid: the planner requires
complete Track exports and Repulsive requires its full-road validity contract.
Using its existing bound slot is confined to the maintenance benchmark; the public
API and production validity contract are unchanged. Merely disabling relaxation
would still include other shared work and would not establish this boundary.

All families produce 256 raw XY samples at size/scale 1 with default construction
parameters. Repulsive retains half-width 0.03 for its own barrier reservation.
CPU batches contain 64 attempts; both GPU batches contain 4096. Generator-native
flags are diagnostics, not final road validity. The table reports time and attempted
centerlines/s, without a final-validity percentage.

## Correctness evidence before production

- RED: nine expected failures identified the missing raw body/guards and obsolete
  finished-road yield accounting. Eight unchanged tests passed.
- Final focused local tests: 17 passed, one CUDA test deselected, in 13.16 s.
- Independent static review: no actionable findings. The timing boundary, guards,
  raw/public parity, BOUNDED context, process isolation and metadata were checked.
- Both measured GPU hosts passed all 18 runner tests with required CUDA available:
  GB10 in 20.02 s and RTX 5090 in 23.80 s. No CUDA skips.
- Both hosts passed real four-environment Polar and Repulsive graph pilots. Capture
  did not execute; replay advanced every epoch/effective key, overwrote raw storage
  and changed accepted finite raw geometry. These pilot timings are not published.
- An additional correctness receipt on each GPU captured all six raw generators,
  then compared two advancing epochs against independent public-pipeline twins.
  All raw XY and generator-flag bytes matched. All public Track arrays, generation
  metadata and common centerline buffers remained unchanged by the raw operation.

The proof calls poison raw output and intrinsic flags. All-rejected changing output
cannot establish the geometry proof. Three proof calls precede pure warm replays;
expensive readbacks/hashing remain outside timing. Warm replay lasts at least one
second. Five timed calls follow an epoch reset, at explicit epochs 1 through 5.
CPU uses completed-call wall time; GPUs use events around actual graph replay.

## Production and publication

Production uses an archive of the exact committed source, one sequential job per
physical machine. No local tests or rendering overlap the Mac measurement; no GPU
tests overlap that host's measurement. Supervisors retain timeout phase and kill
worker descendants. Raw records, settings, epochs, guards and hardware/software
metadata must pass integrity review before the table is published. All 18 records
must be newly measured; no timing is inferred by subtracting shared work.

## Final validation and publication

- Required full CPU lane, including executable documentation snippets: 3,770 passed,
  four optional UI skips, 837 deselected, 640.42 s; exit 0.
- All three production supervisors exited 0. All 18 cells and all 90 timed samples
  passed exact-source/configuration/protocol/seed/epoch/warm-up/guard/unit checks.
- An independent reviewer approved all 18 original records, raw hashes, sample
  arithmetic, hardware/provider metadata, privacy and the displayed table. No
  actionable findings remain. The ms/seconds wording was corrected.
- The publication retains every original record unchanged, with raw-file hashes.
  The generator-only JSON is 113,011 bytes, SHA256
  `12acf54f6b51bc2ee0ff56c27b08648ebd73bde9fbba82d10bebbe33e60f0806`.
- Mac Repulsive's measured median is displayed as 144 s per 64-candidate batch and
  0.444 attempted centerlines/s. Its own growth solver remains inside the boundary.
- Strict Sphinx `-W --keep-going -E` passed with the default 82-pair registry.
  Desktop/mobile browser checks passed in both themes, including all 18 cells,
  the downloaded schema-2 records, delivery matching the committed source data,
  readable cells, contained table scrolling and no console/network/page errors.
  The final presentation pass shortened family links to their names; measurements
  and JSON bytes remained unchanged. The repeated strict build and all four browser
  cases passed for the final concise labels.
- Runtime, RNG/defaults, package metadata and numerical goldens remain identical to
  rc3. All media and per-asset capture provenance remain unchanged from `3ba86e3`.

The preview is refreshed only after production finishes. The rc3 tag stays immutable;
this documentation/tooling follow-up does not authorize a final release or deployment.
