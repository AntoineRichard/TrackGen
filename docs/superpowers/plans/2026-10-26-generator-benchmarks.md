# Generator comparison benchmarks

> For agentic workers: use subagent-driven development for the timing runner and independent review. The docs remain on the existing Sphinx site.

**Goal:** add measured 64-candidate CPU and 4,096-candidate GPU generation costs to Choose a generator for the Mac CPU, Spark GB10 GPU and workstation GPU.

**Scope:** benchmark/documentation tools only. The library, defaults, numerical references and immutable rc3 tag remain unchanged. No final release or deployment is authorized.

## Protocol and design

- Measure the complete public finished-road pipeline, rather than raw centerline generation: that is what Racing consumes. Preserve each family's default generator settings.
- Freeze the quickstart road workload across all six families and hosts: E=64 on CPU / 4096 on GPU, raw points=256, N_max=512, half-width=.03, spacing=.018, size/scale=1, XPBD=50, flat elevation, default validity, poison invalid output. Serialize the full resolved config.
- CPU: eager completed-call wall time from perf_counter_ns. GPU: explicit Warp ScopedCapture of generate(advance=True, sync=False), then graph replay with timing-enabled Warp events on the same stream. Save GPU wall latency separately. Exclude initialization, constructor, capture, first replay and warm-up from steady times.
- Warm for at least three full calls (first plus two) and at least one second. Reset epochs after warm-up, then time five advancing epochs 1..5. Use the explicitly materialized int32 seeds 0..E-1 for each batch/family.
- Before timing, verify capture did not execute, replay advances every epoch once, generation matches epochs, and poisoned result storage is overwritten. Verify finite valid geometry changes across epochs; do not treat all-invalid poison as proof of a full replay.
- Read valid/count/generation/fail flags only after each completed timing interval; include every candidate in yield. Never refill to 4096 valid roads.
- Publish median ms/batch, attempted tracks/s and valid percentage in one compact six-row table on generators/overview.rst. Label CPU wall time and GPU device-event time; these are measured systems/execution modes, not an isolated architecture ratio. Link tutorial choice and detailed benchmark methodology.
- Save individual timing/yield samples, ranges, exact config/seeds/epochs, phase costs, protocol/source hashes, source revision and software/hardware metadata in downloadable JSON. Never publish host aliases/addresses, usernames, executable paths, device UUIDs or private logs.
- Run one generator per subprocess, sequentially on each physical host. Maintain progress phase outside timing and stop/reap the process group on timeout. Never silently shrink E, capacity, iteration budget or work to fill a cell.
- The user explicitly reduced the CPU batch to64. Apply it to all six CPU families; GPUs remain4096. Label batch sizes in every table header; throughput is batch-dependent. Use3600s/CPU-case and1800s/GPU-case process budgets. A construction timeout is not a warm-latency lower bound.
- Workstation is currently not reachable; collect its real hardware/runtime when the user boots it. Use Warp1.18.0 consistently and report Python/NumPy/driver differences.

## Implementation tasks

1. Add benchmarks/benchmark_generators.py and meaningful tests/test_generator_benchmark.py. Interfaces: run_case(generator, *, envs, device, repeats, warmup_seconds, source_revision, label) -> JSON-safe result; a CLI supervisor emits per-family progress/results and a complete manifest, supports explicit runtime budget, and isolates child failures. Tests must exercise real small CPU generation, epoch/recomputation guards, aggregation units/invalid denominator, and timeout phase preservation. CUDA test covers actual graph replay; no mocked timing claims.
2. Independently review the runner, pilot tiny CPU and GPU cases, then commit its validated source and freeze the protocol/source hash before production measurements. Export committed source to remote machines.
3. Root schedules the Mac and GPUs; no concurrent work on a measured physical device. Run all six CPU families at64 and all GPU families at4096. Preserve each actual batch size in the data and table; never extrapolate one to the other. Preserve raw logs externally; audit result hashes/config/epochs before accepting records.
4. Add downloadable current-only data and measured table to generators/overview.rst, update generators/benchmarks.rst methodology, and link from tutorials/choosing-a-generator.rst. Preserve quality metric explanations and internal historical records; never reuse old pre-0.3 timing files.
5. Verify small runner tests including actual CUDA replay, complete required CPU lane and strict docs, numeric table-to-data integrity, no runtime/golden diff and independent final review. Only push validated changes. Keep final release/deployment gated on the user's separate approval.

## Review focus

- No-op/frozen graph replay, missing epoch advance, or host work accidentally included in device-event time.
- A failed/poisoned row excluded from yield denominator, or changing resolution/budget for an expensive family.
- Timing units or cold phases mislabeled as steady generation.
- Timeout killing a worker but leaving descendants, partial results or a misleading latency cell.
- Metadata or diagnostics leaking machine addresses/usernames, and stale or mixed protocol revisions across hosts.
