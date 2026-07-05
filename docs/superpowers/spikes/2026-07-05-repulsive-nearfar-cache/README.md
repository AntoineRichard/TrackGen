# Repulsive TP Gradient — Near/Far RESPA Split Spike — 2026-07-05

Tests a **near/far-field split** (RESPA / multiple-timestepping) of the tangent-point (TP)
gradient in the production repulsive-growth generator
(`track_gen/_src/warp_generate_repulsive.py`), the all-pairs `O(N²)` gather that the
phase-1 spike flagged as the dominant compute at `E ≥ 1024`.

**Bottom line: quality and determinism are green, but the perf payoff is only ~1.3× at
E=8192 (≈1.1× at E≤1024), well short of the 2–4× target — because env-freezing already
elides most of the full-`N` final-stage work the split targets, and the coarse stages
(~27% of wall-clock) lie outside the split. Recommendation: DON'T SHIP (see below).**

This spike does **not** modify `track_gen/_src/`. It reuses the production kernels verbatim
where possible (`nearfar.py` imports `warp_generate_repulsive`) and adds the near/far
machinery + a growth-loop clone whose only change is the gradient assembly.

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 \
  docs/superpowers/spikes/2026-07-05-repulsive-nearfar-cache/run_spike.py <cmd>   # cuda
# cmd in {smoke, frontier, perf, determ, stages}
```

## Artifacts

- `nearfar.py` — the near/far kernels (candidate broadphase, near prepass/gather, TP-only
  full gather, obstacle+length gather, refresh/frozen combine) + `make_generate(K, cutoff,
  maxnbr)`, a near-clone of `rep.generate_repulsive_warp` whose gradient step is the split.
- `run_spike.py` — the experiment driver (frontier / perf / determinism / stage breakdown).
- `_profile.py` — per-kernel cost attribution at E=8192, N=256.
- `_baseline.py` — confirms the production ground truth (64/64 @ 0.15093).

## The split (as implemented)

The total per-iter gradient is `g = g_TP + g_obs + g_len`. Only `g_TP` is split:

- **NEAR** — TP pairs (circ>2) whose partner is inside a Euclidean cutoff radius, from a
  **fixed-slot per-vertex candidate list** (ascending-j order → deterministic, no atomics)
  rebuilt every `K` iters. The cutoff is `(cutoff_beads + 2·τ·K)` mean-segment-lengths,
  applied **per-env on-device** from the current perimeter — the `2·τ·K` margin covers the
  most a pair can approach in a K-window (steps are capped at ~`τ·msl`/iter), so no pair
  crosses the eval horizon between refreshes (same safety argument as `warp_relax`'s cache).
  Recomputed **exactly every iteration** against the (stale) cached partner set.
- **FAR** — everything else. On a refresh iter `g_far = g_full_tp − g_near_tp` (exact), held
  **FROZEN** in between. No truncation of the physics — the global packing pressure is
  preserved, just temporally coarsened.

The partition is exact **by construction**: `g_far` is the full TP minus the near TP at
refresh time, so no pair is dropped or double-counted regardless of cutoff/margin. The
cutoff/margin govern only *accuracy between refreshes*.

- **K=1 and the coarse stages fall back to the exact production combined gather**
  (`rep._grad_gather_k`) — so **K=1 reduces byte-for-byte to current behavior** (verified:
  `smoke` prints 64/64 @ 0.15093, identical to `_baseline.py`). The split is gated to the
  **final `N=256` stage** (`split_final_only=True`) since the coarse `O(N²)` is 16×/4× cheaper.
- **Obstacle + length are kept exact every iter** (O(N·M), M=240, and O(N)) — correctness-
  critical for domain confinement and, per profiling below, a small cost.

A per-vertex Euclidean-radius candidate cache (not an along-strand band) is essential: the
strong short-range TP forces are between geometrically-close strands that are **far apart
along the curve** (packed folds). An along-strand band would freeze exactly the fold-fold
repulsion that prevents self-crossings.

## Experiment 1 — K / cutoff quality frontier (E=64, seed 11, through the tail)

`maxnbr` = fixed candidate slots/vertex (heuristic: 64 / 112 / 200 for cutoff 8 / 16 / 32).
`overflow` = TP pairs dropped past `maxnbr`, summed over all refresh launches (diagnostic;
dropped near pairs fall into the frozen far field — still exact at each refresh).

| K | cutoff | maxnbr | yield | compactness | overflow | in band & ≥63/64 |
|---|---|---|---|---|---|---|
| **1** | — | — | **64/64** | **0.15093** | — | ground truth |
| 2 | 8 | 64 | 64/64 | 0.15112 | 23 k | ✅ |
| 4 | 8 | 64 | 64/64 | 0.15073 | 406 k | ✅ |
| **8** | **8** | **64** | **64/64** | **0.15066** | 2.1 M | ✅ **best (perf+quality)** |
| 16 | 8 | 64 | 60/64 | 0.15471 | 4.4 M | ❌ |
| 2 | 16 | 112 | 64/64 | 0.15110 | 5.0 M | ✅ |
| 4 | 16 | 112 | 63/64 | 0.15059 | 4.4 M | ✅ |
| 8 | 16 | 112 | 63/64 | 0.15213 | 4.8 M | ✅ |
| 16 | 16 | 112 | 63/64 | 0.15380 | 5.7 M | ✅ |
| 2 | 32 | 200 | 64/64 | 0.15096 | 13.3 M | ✅ |
| 4 | 32 | 200 | 64/64 | 0.15092 | 8.3 M | ✅ |
| 8 | 32 | 200 | 64/64 | 0.15161 | 5.5 M | ✅ |
| 16 | 32 | 200 | 62/64 | 0.15150 | 3.8 M | ❌ |

**Quality is remarkably robust.** Every tested config stays inside the 0.146–0.155
compactness band. Yield holds **64/64 up to K=8** at cutoff 8 and 32; K=16 is the first cliff
(60–63/64). The far-field freeze does **not** visibly degrade the tracks at any K ≤ 8 — the
far field really is smooth enough to temporally coarsen. The **frontier winner is K=8 /
cutoff=8 / maxnbr=64** (64/64 @ 0.15066): the largest K holding full yield at the smallest
cutoff, i.e. the best perf ceiling (`N/maxnbr = 4×`).

The heavy `overflow` at maxnbr=64 (a radius-8-spacing neighborhood routinely holds >64 pairs
in dense folds) does **not** hurt yield — truncated near pairs get the frozen-far treatment
and it re-anchors every K iters. Truncation is deterministic (first maxnbr in ascending-j
order), so byte-determinism is preserved.

## Experiment 2 — Performance (RTX 4090, growth-only, post-warmup)

Alternating A/B (min-of-5) to control clock/thermal drift. **K=8 / cutoff=8 / maxnbr=64.**

| E | baseline K=1 | split | speedup |
|---|---|---|---|
| 64   | 0.093 s | 0.076 s | 1.1–1.2× |
| 1024 | 0.236 s | 0.215 s | **1.10×** |
| 8192 | 1.25 s  | 0.94 s  | **1.34×** |

Speedup **grows with batch** (more compute-bound) but tops out ~1.3×. **This is well below
the 2–4× the split was expected to deliver.** Two measurements explain exactly why.

### Where the time goes — stage breakdown (`run_spike.py stages`)

| E | config | coarse (N=64,128) | final+settle (N=256) | final % |
|---|---|---|---|---|
| 8192 | K=1 baseline | 339 ms | 916 ms | 73% |
| 8192 | K=8 cut=8 | 346 ms (untouched) | **598 ms (1.53×)** | 63% |
| 1024 | K=1 baseline | 69 ms | 174 ms | 72% |
| 1024 | K=8 cut=8 | 66 ms (untouched) | 147 ms (1.18×) | 69% |

The split delivers **1.5× on its target final+settle phase** at E=8192 — real, but the coarse
stages are ~27% of wall-clock and untouched, so the Amdahl ceiling for `split_final_only` is
`1/0.27 ≈ 3.7×` even with an *infinite* final-phase speedup. 1.5× on 73% → **1.33× overall**.

### Why only 1.5× on the final phase (not the profiled 2.7×) — per-kernel attribution

`_profile.py` at E=8192, N=256, **all envs active** (ms/call):

| kernel group | ms | notes |
|---|---|---|
| TP prepass (full) + gather (full) | 10.7 + 13.1 = **23.9** | the O(N²) target — dominant |
| TP prepass (near) + gather (near) | 1.4 + 1.3 = **2.7** | **9× cheaper** than full ✅ |
| candidate build | **7.2** | O(N²) scan; **refresh only** (1/K) |
| obstacle+length (O(N·M)) | 0.74 | small floor |
| Sobolev `_conv_k` ×2 | 3.0 | small floor (untouched) |

So on the *fully-active* final iters the model predicts **2.67×** at K=8. The **1.5×** actually
measured on the final phase is diluted by:

1. **Env-freezing.** During settle, envs freeze progressively and *already skip* their full-`N`
   TP work (the production `frozen[e]` guard). The split's target — full-batch N=256 TP — only
   exists for the pre-freeze part of the final stage; the rest is already cheap. This is the
   dominant dilution and the honest surprise: **the stall-stop lever from phase-1 pre-empts
   most of what the near/far split was going to save.**
2. **The ~10 small optimizer-tail kernels** per iter (ratchet, tangent/weight, length-grad,
   numden, project, gmean, gmax/msl, step, perim/bc, rescale) are unchanged and add a fixed
   per-iter cost that dilutes the ratio — and the loop is **partly launch/issue-bound** (phase-1
   measured ~110 W at "100% util"), so cutting TP FLOPs helps less than a FLOP model predicts.
   E=64 barely moves (1.1×, pure launch-bound); the gain only appears as E grows the compute.
3. **cand_build's 7.2 ms refresh tax** (≈0.9 ms/iter amortized at K=8).

## Experiment 3 — Determinism

`run_spike.py determ` (K=8 / cutoff=8, E=64, two runs on cuda): **byte-identical = True.**
The split stays gather-form (one thread per vertex, register accumulate), the candidate list
is built in ascending-j order with **no atomics feeding the gradient** (the only atomic is the
diagnostic overflow *counter*, which is order-independent integer sum), and the on-device
per-env cutoff removed the host readback that an earlier draft used. Byte-determinism per
device is preserved.

## Memory cost

The candidate buffer is `E · N_max · maxnbr · 4 B`. At E=8192 / maxnbr=64 that is **512 MB** —
on top of the phase-1 baseline's ~590 MiB peak, i.e. the split **nearly doubles peak GPU
memory**. Larger cutoffs are worse (maxnbr 112/200 → 0.9/1.6 GB).

## Recommendation: DON'T SHIP (yet)

The technique is **correct, safe, and byte-deterministic** — quality holds 64/64 with the
far-field freeze up to K=8, so this is a *negative perf result on a sound method*, not a broken
one. But it should **not** ship as a `repulsive_*` knob, because:

1. **The payoff is ~1.3× at E=8192 and ~1.1× at E≤1024** — far short of 2–4×, and near-zero at
   the E=64 default. Not worth a user-facing knob.
2. **The stall-stop lever already captured most of the win.** Freezing elides the full-`N` TP
   the split targets, so the two levers are largely redundant; the split only re-accelerates the
   *pre-freeze* slice of the final stage (1.5× on 73% of wall-clock).
3. **~2× peak memory** for the candidate cache (512 MB at E=8192).
4. **Non-trivial complexity + a real footgun**: 6 new kernels, a candidate cache whose
   `id(scratch)` key can be reused after GC (this bit us — an undersized buffer caused an
   intermittent illegal-memory-access at E=8192 until the cache validated E/N_max).
5. **Phase-1 already named the more binding lever: CUDA graph capture.** At the ~110 W
   issue-bound operating point the bottleneck is per-iter launch overhead across *all* stages
   (including the 27% coarse fraction the near/far split cannot touch), not TP FLOPs. Graph
   capture attacks that directly and compounds with freezing; the near/far split does neither.

**When it *would* pay off (future work):** a config that **disables freezing or runs a long
wall-free settle** (the phase-1 "longer wall-free relax remains untested" resume point) would
make final+settle a much larger, non-shrinking fraction and let the 1.5–2.7× land closer to the
profiled ceiling. If such a regime is ever wanted, this spike's kernels are the ready
implementation. Absent that, prioritize **graph capture** (per-stage graphs) over this split.
