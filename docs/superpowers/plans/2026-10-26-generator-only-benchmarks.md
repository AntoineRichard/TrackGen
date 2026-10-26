# Generator-only benchmark correction implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent harness, documentation and review work below.

**Goal:** Compare each generator's own construction cost without the shared road pipeline, as requested by the user after inspecting the XPBD-inclusive table.

**Architecture:** A maintenance-only body invokes the existing bound generator slot of an ordinary TrackGenerator. It advances epochs, derives effective seeds, latches generator parameters and executes only the generator slot. The full facade is constructed outside timing to preserve its resolved parameters and Repulsive width reservation. No library changes are required.

**Tech stack:** Existing Python supervisor, Warp CPU/CUDA graph replay, pytest, Sphinx, static JSON measurements.

**Spec:** The user's 26 October correction excludes XPBD from the generator comparison. This plan makes the boundary generator-only rather than merely disabling XPBD while retaining other shared road work.

## Constraints and review focus

- CPU batches remain 64; both GPU batches remain 4096. All six families produce 256 raw XY points with default construction parameters, size/scale 1. Repulsive retains its existing 0.03 width reservation.
- Include seed preparation and generator-owned sampling, splines, crossing fallback, resampling and iterative growth. Exclude shared constant-spacing resampling, XPBD/tails, elevation, borders, final validity and commit.
- Never present generator-native status as finished-road validity. The table contains time and attempted centerlines/s; all attempts remain in the denominator.
- Five advancing measured epochs follow at least three warm calls and at least one second of warm replay. CPU uses completed-call wall time; GPUs use events around actual captured replay, with wall time retained separately. No measurement includes diagnostics, initialization, construction, compilation or capture.
- A generator-only PipelineSpec is invalid by design; retain the already-bound slot and its BOUNDED context. Do not replace runtime stages or weaken Repulsive's contracts.
- Prove raw-buffer parity with the ordinary full pipeline at matching seeds/epochs for every family; prove common stages and final Track storage remain untouched. Prove actual CUDA capture/replay and sentinel overwrite.
- No runtime, RNG, defaults, package metadata, golden or immutable rc3 changes. Preserve all media and capture provenance.
- Source-first measurements: commit reviewed/tested runner before archiving to both GPU hosts. Do not time from a dirty or different source tree, reuse full-road timings or infer raw time by subtraction.
- One production job per physical machine. No local tests/rendering during Mac timing, no GPU tests during that host's timing. Only root schedules production work.
- Retain source/protocol hashes, exact settings, all five timing samples, actual hardware/software and diagnostics. Never commit host addresses or private paths.

## Work

- [x] Harness: write failing boundary/parity/accounting/replay tests, then adapt benchmarks/benchmark_generators.py and tests/test_generator_benchmark.py to schema/protocol version 2. Preserve process isolation, timeout phase reporting and explicit no-fallback CUDA selection. Limit expensive warm-up proof readbacks to the proof calls, then perform pure warm replays.
- [x] Independent harness review; targeted CPU tests and real CPU/CUDA pilots on the committed source. Export exact source archives for Spark GB10 and workstation RTX 5090.
- [x] Measure all 18 cells: Mac CPU64, GB10 CUDA4096 and RTX5090 CUDA4096. Require successful protocol/seed/epoch/geometry guards; retain honest error outcomes rather than fabricating results.
- [x] Documentation: update methodology and choosing tutorial; replace the overview table and public JSON with verified generator-only measurements. Remove finished-road validity percentages. Keep quality diagnostics clearly separate and internal historical records outside the site.
- [x] Verify all data against raw records, table units/formatting and hardware specifications. Run required CPU lane and snippets, strict docs with all82media pairs, browser desktop/mobile in both themes and independent final review. Check source/package/goldens/media remain unchanged.
- [ ] Commit and push validated work to v3, refresh local preview, update draft PR4 and report actual CI status. Leave rc3 immutable and wait for Racing/final-release approval.
