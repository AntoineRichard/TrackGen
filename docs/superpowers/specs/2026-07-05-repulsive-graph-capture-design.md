# Repulsive Generator — CUDA Graph Capture Design

**Date:** 2026-07-05
**Status:** Implemented
**Depends on:** the repulsive-growth generator (2026-07-05-repulsive-generator-design.md),
the pluggable generator framework + `GeneratorSpec.capturable` flag (2026-06-21), the
pipeline-level auto-capture path in `TrackGenerator` (2026-06-18).

---

## 1. Goal

Make `generator="repulsive"` participate in the pipeline-level CUDA graph like the other five
generators (`capturable=True`), removing its host-driven, eager-only special case. The historical
interior blocker (a per-iteration `wp.Tape`) is already gone — replaced by hand-written analytic
adjoints — so what remained was the host-side loop control: coarse-to-fine stage transitions, the
fixed-cadence periodic resample, and the **area-stall early exit** (a per-window host readback of
`frozen.sum() >= E`), the last of which is a host branch on device data and thus illegal inside a
capture region.

## 2. API investigation (Warp 1.14, RTX 4090, driver 13.0)

`wp.is_conditional_graph_supported()` → **True** (CUDA Toolkit 12.9 bundled with Warp, driver 13.0;
CUDA 12.4+ is the documented floor for conditional graph nodes). Both `wp.capture_while(condition,
body)` and `wp.capture_if(condition, on_true, ...)` exist and take a device `int` array whose first
element is the condition. Throwaway prototypes confirmed the two properties this design needs:

1. `capture_while` nested inside `wp.ScopedCapture` records a conditional while-node and replays
   byte-identically to an eager host-driven loop (data-dependent iteration count reproduced).
2. `capture_if` nested **inside** a `capture_while` body, with a multi-launch conditional body
   (mimicking the 2-launch periodic resample), also records and replays byte-identically.

So architecture **B (fully capturable generator)** is feasible. It is the one adopted.

## 3. Architecture (B, with a preserved eager path)

The growth loop splits into two phases:

- **Phase 1 — coarse stages `[0, final_start)`** (`final_start = min(stage_starts[-1], n_iters)`):
  stage transitions + fixed-cadence periodic resample, no stall/freeze (freezing only ever happens
  at the final stage). Every iteration index and every resample point is host-known, so this phase
  is a plain Python loop that **unrolls at capture time** into a fixed launch sequence — identical
  on the eager and captured paths.

- **Phase 2 — final stage `[final_start, n_iters)`** with the area-stall early exit:
  - **Eager** (`_pipe._CAPTURING` is False: the Warp `cpu` device always, and the pre-capture
    reference call on `cuda`): the **original** Python loop with the host `frozen.sum() >= E`
    readback and `break`, byte-for-byte **unchanged**.
  - **Captured** (`_pipe._CAPTURING` is True, inside `ScopedCapture`): the same early exit expressed
    **device-side**. `wp.capture_while(keep, body)` loops on a device `keep` flag; the body runs one
    growth iteration, then two `wp.capture_if` branches (keyed off a device iteration counter
    `it_dev` via tiny single-thread flag kernels) run the periodic resample at cadence
    `resample_every` and the stall freeze at cadence `stall_window`, exactly as the eager cadence
    tests `(it + 1) % cadence == 0`. `_set_active_k` (a `frozen.sum() < E` reduction) runs inside
    the stall branch — the device form of the host readback — and `_advance_keep_k` increments
    `it_dev` and recomputes `keep = active AND it_dev < n_iters` (the two conditions the eager loop
    enforces via `if all frozen: break` and its `range(..., n_iters)` bound).

Both paths execute the **same kernels in the same order for the same number of iterations**, so the
eager result and the captured replay are **bit-identical** (verified: E=64 seed-11 and E=512 seed-0
both `array_equal`). The generator flips to `GeneratorSpec(capturable=True)` and joins the
pipeline-level graph; `test_warp_graph.py` auto-enrolls it and its replay==eager parity assertion
passes byte-exactly.

New device scratch (single `int32` scalars, allocated once): `it_dev`, `active`, `keep`,
`cond_res`, `cond_stall`. New single-thread control kernels: `_init_stall_state_k`, `_flags_k`,
`_set_active_k`, `_advance_keep_k`. No per-call allocation; the arc-length/uniform resamples keep
fixed launch dims (their `_sync` is already gated on `_pipe._CAPTURING`).

## 4. Why not the simpler variants

- **Drop the early exit entirely (fixed-topology unroll, no conditional nodes).** Simplest, but the
  captured path then runs the *full* `n_iters` while the eager path early-exits, so to keep parity
  the eager path must also run the full budget — which (a) changes the Warp `cpu` behaviour
  (violating "cpu unchanged"), (b) adds ~26 % mostly-frozen tail iterations that were a measured
  **regression** (~8 % slower than the early-exit original at E=64), and (c) shifts the seed-11 gate
  from 0.1509 to 0.1544 (still in band, but a needless change). Rejected.
- **Make the periodic resample frozen-aware** so full-budget == early-exit byte-identically. Keeps
  the eager early exit, but reparameterizing converged envs differently flipped a borderline env:
  seed-11 dropped to **63/64** @ 0.1475. Rejected (the gate must stay 64/64).

Architecture B (device-side `capture_while`) is the only option that keeps the eager `cpu` path
unchanged **and** the seed-11 gate at exactly 64/64 @ 0.1509 **and** passes replay==eager parity.

## 5. Measured reality — capture is roughly wall-clock-neutral here

The spec that deferred this work called the loop "heavily launch-bound at small/mid E" and named
graph capture "the binding lever". Direct measurement on the RTX 4090 does **not** bear that out:

| E | host launch-issue time | GPU compute (issue+drain) | eager wall |
|---|---|---|---|
| 64 | 86.7 ms | ~107 ms | ~113 ms |
| 1024 | 271 ms | ~313 ms | ~305 ms |

The host launch issue **overlaps** the O(N²) tangent-point / Sobolev kernels and is *smaller* than
GPU compute, so the loop is **GPU-compute/latency-bound**, not host-launch-issue-bound. (The
"110 W of 450 W" symptom the spec cited is low *occupancy* at small E — latency-bound — which graph
capture does not fix.) Controlled same-session eager-vs-captured comparison: **1.13× at E=64, 1.06×
at E=512, ~neutral (0.94–1.02×) at E≥1024** — within the spec's own ±1.5× throttle band. The small
small-E win is the removed per-window host syncs, not launch-issue elimination. This corroborates
the RESPA spike's 1.3× negative result.

**So the value of `capturable=True` here is architectural, not wall-clock:** the generator no longer
needs a special eager branch or per-window host syncs, it joins the single replayable pipeline graph
like the others, and the host thread is freed across the whole captured pipeline (~87 ms/call at
E=64) for CPU/GPU overlap in an RL loop — the actual point of the pipeline graph. Byte-determinism
per device and the seed-11 quality gate are preserved exactly.

## 6. Verification

- `captured == eager == pre-change output`, byte-identical (E=64 seed-11, E=512 seed-0).
- Determinism: two captured runs byte-identical (per device); CPU + CUDA determinism tests green,
  run twice.
- Seed-11 gate: **64/64 @ median compactness 0.1509** (unchanged).
- `test_warp_graph.py::test_autocapture_replay_matches_eager[repulsive]` (auto-enrolled) passes.
- Full suite: 562 passed, 3 skipped (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, CUDA not filtered).
