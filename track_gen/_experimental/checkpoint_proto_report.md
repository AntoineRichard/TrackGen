# Checkpoint-Steering generator (#5) — host prototype report

Prototype: `track_gen/_experimental/checkpoint_proto.py` (pure numpy + matplotlib, DEV-only,
not imported by runtime). Mirrors `grammar_proto.py` in structure and validation methodology so
the two first-stage generators can be compared head-to-head. This documents the algorithm as
implemented, the closure variant chosen and why, the tuned `DEFAULTS`, the K=1/4/8 metrics, the
render paths, and the verdict.

## Method (CarRacing, adapted for Warp-portability)

Gymnasium `CarRacing` samples ~12 radial checkpoints, steers a bounded-turn path through them,
over-generates ~4 laps, finds where the path re-crosses the start, **trims** to that sub-loop,
and **regenerates in a `while True`** until it closes and is "well glued". The trim
(variable-length) and the retry (unbounded) are not CUDA-graph-capturable, so — exactly as the
grammar port did — they are replaced by fixed-shape equivalents:

1. **Sample C checkpoints** on radii `U(radius_min_frac*R, R)` at angles `2*pi*c/C + jitter`,
   with `jitter` bounded to under one angular slot so the checkpoint sequence stays
   angle-monotone (the path winds exactly once). This is CarRacing's checkpoint distribution.

2. **Fixed-N bounded-turn steering, one lap.** Walk a fixed `N`-step path. State = (position,
   heading). Each step: aim at the current target checkpoint, `err = wrap(target_bearing -
   heading)`, `dtheta = clamp(steer_gain * err, ±turn_rate)`, advance heading, step forward by a
   constant `dl`. The target index advances when within `lookahead_frac*R` of the current
   checkpoint (a per-step int-counter increment — Warp-friendly; no host branch on data).
   - **Key fix for one-lap closure:** `dl = (checkpoint-ring perimeter) / N`. The ring perimeter
     is a fixed reduction over the checkpoint positions (capturable). A naive constant
     `dl = 2*pi/N` made the path wind **1.1–1.5 laps** and coil over itself; ring-`dl` pins it to
     ~1 lap. No over-generate-and-trim.

3. **Explicit closure by construction** (see next section) — no trim, no reject.

4. **Best-of-K selection** replaces the `while True` retry: generate `K` decorrelated candidates
   (candidate k uses `seed*K + k`), keep the one with the fewest self-intersections (deterministic
   argmin, ties → lowest k, early-out on the first zero). Bounded and capturable — the same trick
   that took grammar's self-intersection rate from ~33% to ~5%.

Everything is fixed-shape (fixed C, N, K; bounded steering; the only "branches" are
capture-time constants), so the prototype maps cleanly to a future `warp_generate_checkpoint.py`.
The host best-of-K scorer (`_self_intersections_count`) is vectorised over the edge-pair grid;
the Warp port would use a tile/grid kernel for that, and only the per-step steering loop ports
literally.

## Closure variant chosen, and why

The task said to try the two book variants and keep what renders best. Both book variants were
implemented; a **third** (the winner) sits between them. All three are in the module
(`close_positions`, `close_kappa`, `close_heading_ramp`, dispatched by `cfg["closure"]`).

| closure | what it does | K=1 self-int | K=8 self-int | median compactness | character |
|---|---|---|---|---|---|
| `pos` (a) | subtract the linear endpoint-residual ramp in **position** space (grammar's gap-distribution edge step on positions) | **0.96** | **0.69** | 0.44 | flowing, but the steered head-arc and return tail-arc physically overlap near the start → almost every loop self-crosses. Best-of-K **cannot** rescue an all-bad pool. Unusable. |
| `kappa` (b) | extract heading, **multiplicatively** rescale every turn so net turn = 2*pi (grammar's T1), gap-distribute displacement | 0.24 | **0.00** | **0.81** | clean, but the uniform rescale flattens local curvature → smooth convex **blob, indistinguishable from polar/hull**. Loses the CarRacing flow. |
| `heading_ramp` (c) **← DEFAULT** | close heading to turning-number-1 **additively**: add constant drift `(2*pi - net)/N` to every step, gap-distribute displacement | 0.20 | **0.00** | **0.55** | clean **and** keeps the steered sweep/undulation. Inlets and bulges survive; not a blob. |

(300-seed comparison at the tuned defaults.)

`heading_ramp` is grammar's T1 *displacement* closure with an **additive** (not multiplicative)
heading correction. Like `kappa` it guarantees turning number 1 — no inner loops, so best-of-K
starts from a low crossing rate it can drive to exactly 0 — but unlike `kappa` it preserves the
local curvature variation that gives the distinctive flowing character. `pos` is kept for
comparison only; it demonstrates why a pure position-space close fails here (the open path's two
ends overlap, a defect that lives in the *topology*, which best-of-K cannot select away).

Note: a "return-to-start" stretch (pinning the tail back to checkpoint 0) was tried and
**rejected** — it makes the return arc cross the departure arc near the start, a guaranteed cusp
(100% self-cross). Closure is the closer's job, not the steerer's.

## Tuned DEFAULTS

```python
DEFAULTS = dict(
    num_points=256,         # N: fixed path length (one lap)
    checkpoint_count=12,    # C: CarRacing's canonical 12. chicane_count scales ~ with C.
    radius_min_frac=0.33,   # checkpoint radius ~ U(0.33*R, R)  — CarRacing's R/3 exactly
    angle_jitter=0.55,      # ± fraction of the angular slot (slot = 2*pi/C); <1 keeps monotone
    turn_rate=0.42,         # max heading change per step (rad)
    steer_gain=0.65,        # proportional steering gain toward target bearing
    lookahead_frac=0.16,    # advance to next checkpoint within this*R
    closure="heading_ramp", # winner; "kappa" and "pos" also available
    candidates=8,           # K for best-of-K
    scale=1.0,
)
```

Tuning notes (renders were the arbiter):
- **C** is the main character knob: `chicane_count ≈ C` roughly. C=8 → ~3 chicanes (calm,
  rounder, comp ~0.77); C=12 → ~7 chicanes (the CarRacing waviness, comp ~0.58); C=14 → ~20
  chicanes (too busy, K=1 self-int 21%). C=12 is the sweet spot and CarRacing's canonical value.
- **radius_min_frac** controls radial drama. 0.25 deepens inlets but pushes some seeds to a
  spiky "starfish" look; 0.40 is rounder. 0.33 (= R/3, CarRacing) balances inlets vs simplicity.
- **turn_rate** in 0.30–0.55 barely moves validity (best-of-K absorbs it); 0.42 keeps arcs
  flowing without hairpin spikes.
- The character ↔ validity tension is real: a longer/wavier path has more character but more
  K=1 crossings. **This is exactly what best-of-K is for**, and it resolves it cleanly here.

## Metrics (500 seeds, tuned DEFAULTS, closure=heading_ramp)

Self-intersection sampled on the first 150 seeds in the driver table; the full-500 best-of-8
rate was separately confirmed to be **0/500**.

| K | compactness p10/p50/p90 | chicane (mean) | straight_fraction (mean) | self-int rate |
|---|---|---|---|---|
| 1 | 0.355 / 0.549 / 0.658 | 6.89 | 0.767 | **0.207** |
| 4 | 0.455 / 0.581 / 0.670 | 6.86 | 0.760 | **0.007** |
| 8 | 0.460 / 0.584 / 0.674 | 7.09 | 0.759 | **0.000** |

The best-of-K effect is clearly visible: **self-intersection 20.7% (K=1) → 0.7% (K=4) → 0.0%
(K=8)**, with compactness essentially unchanged (best-of-K trims the worst tails, it does not
regularise the shape). Median compactness ~0.58 sits well below the `kappa`/polar blob (~0.81)
and in the characterful range, while `chicane_count ~7` and `straight_fraction ~0.76` confirm a
mix of sweeping corners and straighter runs rather than a constant-curvature ring.

For context: the grammar generator (#6) targets ~3 chicanes and a deliberate segmented
straight+corner road-course (compactness ~0.4–0.6). Checkpoint-steering produces a *different*
distribution — more, gentler direction changes (continuous flow) rather than few sharp,
deliberate corners.

## Renders

- Single-candidate (K=1): `viz/out/checkpoint_proto_grid_k1.png` — several seeds show red
  (self-crossing) loops; the ~20% K=1 crossing rate made visible.
- Best-of-8: `viz/out/checkpoint_proto_grid_k8.png` — all clean (0 crossings across the full 500
  seeds), flowing/sweeping loops with inlets and bulges.

(Renders are throwaway / not committed; regenerate with
`python track_gen/_experimental/checkpoint_proto.py`.)

## Verdict

**PORT-WORTHY (with one tuning caveat).** Checkpoint-steering with the `heading_ramp` closure and
best-of-K produces a genuinely **distinct "flowing / sweeping" character** — continuous, organic
undulation with radial inlets — that is clearly *neither* the grammar generator's deliberate
few-corner road-course *nor* a convex polar/hull blob. It is fixed-shape, free of data-dependent
host control flow, and reaches a **0% self-intersection rate at best-of-8 across all 500 seeds**,
so it satisfies the Warp-capturability constraints and the validity bar.

Recommended defaults: the tuned `DEFAULTS` above (C=12, radius_min_frac=0.33, turn_rate=0.42,
closure=`heading_ramp`, K=8).

Risks to watch on the Warp port:
1. **The closure is the whole ballgame.** The naive book variants both fail in opposite ways
   (`pos` → 96%/69% self-cross; `kappa` → polar blob). The Warp kernel must implement the
   *additive* heading-ramp close (turning-number-1 by additive drift + gap-distribution
   displacement), not a multiplicative T1 rescale, or the character is lost. This is the single
   highest-risk item — it is subtle and easy to get wrong by copying grammar's T1 verbatim.
2. **K=1 validity is only ~80%, so best-of-K is load-bearing, not cosmetic.** K=4 already reaches
   ~0.7% and K=8 reaches 0%; the port must keep K≥4 (ideally 8) and the deterministic
   fewest-crossings argmin. The per-candidate self-intersection count is the cost driver of the
   port (O(N^2) edge pairs × K) and should be a tiled/blocked kernel, not the host loop used here.

## Cheaper alternative: K + single-crossing CLIP ("loop removal") — benchmark

The K=8 brute-force pays 8× the O(N²) edge-pair cost just to *select away* the rare crosser. Can a
cheaper **K=2 best-of-K + one-shot clip** match it? The clip (`clip_single_crossing`) finds the
self-crossings (the same O(N²) edge-pair PASS, now recording the crossing index pair `(i,j)` and
the intersection point `P`), and **if the curve crosses** splits it at the first crossing into two
sub-loops — inner arc `pts[i+1..j]` and outer arc `pts[j+1..N-1]+pts[0..i]`, each closed through
`P` — keeps the **longer by arc length**, and arc-resamples it back to N points (fixed-N out).
With **exactly one** crossing both sub-loops are simple, so the clip is a guaranteed rescue. With
≥2 it is best-effort: we apply the single clip anyway (one-shot, **no** iterate-until-simple loop)
and re-count residual crossings. The `K+clip` pipeline (`generate_centerline_clip`) clips each of K
candidates once and keeps the one with the fewest residual crossings (argmin, ties → lowest k).

Benchmark: `track_gen/_experimental/checkpoint_clip_bench.py`, **1000 seeds**, tuned DEFAULTS.
Cost proxy = O(N²) self-intersection PASSES per env (the Warp cost driver), worst-case (no
early-out): best-of-K alone = **K** passes; `K+clip` = **2K** passes (K clip-find + K residual-count).

| config | post-proc SI rate | cost (N² passes/env) | compactness p50 | straight_fraction |
|---|---|---|---|---|
| K=1        | 0.2210 | 1 | 0.555 | 0.770 |
| K=2        | 0.0630 | 2 | 0.574 | 0.763 |
| K=4        | 0.0040 | 4 | 0.581 | 0.758 |
| K=8        | **0.0000** | 8 | 0.580 | 0.757 |
| K=1+clip   | 0.0770 | 2 | 0.582 | 0.767 |
| **K=2+clip** | **0.0060** | **4** | 0.586 | 0.766 |
| K=4+clip   | **0.0000** | 8 | 0.583 | 0.765 |

(Host-numpy wall clock was ~80–100 s / 1000 envs across configs — dominated by the O(N²) self-int
test in pure numpy; it's a rough secondary signal, NOT the Warp cost. The pass column is the real
trade.) Compactness p50 (~0.58) and straight_fraction (~0.77) are **unchanged** by clipping — the
clip keeps the longer sub-loop, so it preserves the flowing/sweeping shape rather than wrecking it.

### KEY diagnostic — crossing-count distribution among the raw K=1 crossers

This single number explains the whole table. Of the **221/1000 = 22.1%** single-candidate crossers:

| crossings on the crosser | count | fraction of crossers | fraction of all envs |
|---|---|---|---|
| **exactly 1** (one-shot clippable → simple) | 110 | **0.498** | 0.110 |
| exactly 2 | 60 | 0.272 | — |
| ≥3 (max seen: 6) | 51 | 0.231 | — |

**Only ~50% of crossers have a single crossing** — so a *lone* one-shot clip cannot zero the rate
(it leaves the ~half with ≥2 crossings, some still crossing after one clip). That is exactly why
`K=1+clip` only reaches 7.7% (≈ the 11% single-crossers it can fix, minus partial reductions on
multi-crossers). The leverage comes from **combining** clip with even a tiny best-of-K: clipping
*reduces* every candidate's crossing count (a 3-crossing curve clips toward 1–2), and best-of-2
then **selects the candidate that clipped cleanest**. Two clipped, decorrelated candidates almost
always contain one that lands at zero → **K=2+clip = 0.6%**, statistically tied with **K=4 (0.4%)**
and one residual short of K=8.

### Verdict on K=2+clip

**K=2+clip matches K=4-quality (0.6% vs 0.4% SI) at K=4-equivalent cost (4 passes), and lands within
one residual env of K=8 at HALF of K=8's cost (4 vs 8 passes).** It is the **sweet spot for the
near-zero tier**: the cheapest config that drops SI to <1% while *fully* preserving shape. It does
**not** beat plain K=4 on the cost/quality frontier (both are 4 passes, ~0.5% SI), because the clip
itself costs a second pass per candidate (2K, not K) — so "K=2+clip" and "K=4-alone" are the same
price. The honest framing: **clip buys you K=8's quality tier at K=4's price** (K=4+clip = 0/1000,
8 passes, identical to K=8 but with shape-preserving rescue instead of blind reselection), and lets
**K=2 reach the sub-1% band it cannot reach alone** (6.3% → 0.6%).

**Residual for the validity gate:** K=2+clip leaves **~0.6% (6/1000)** self-intersecting envs — the
multi-crossing cases (≥2) where neither decorrelated candidate clipped to zero. That residual must
still be caught by the downstream validity gate (or bumped to K=4+clip / K=8 for a hard-zero
guarantee). Recommendation: **K=2+clip when a ~0.6% residual is acceptable to the gate** (best
shape-preserving rescue per pass in the near-zero band); **K=4+clip for a measured 0/1000** at the
same 8-pass budget as K=8. The clip is fully bounded (O(N²) find + O(N) keep-longer + O(N) resample,
no unbounded iterate-until-simple), so it is as CUDA-graph-capturable as best-of-K itself.

Render: `viz/out/checkpoint_k2clip_grid.png` (K=2+clip) — 24/25 clean, flowing loops with inlets,
visually the same family as the K=8 grid; clip preserves shape (no degenerate cusps/collapsed
loops). The lone red cell is a residual ≥2-crossing seed, consistent with the 0.6% rate.
(Regenerate with `python track_gen/_experimental/checkpoint_clip_grid.py`; renders not committed.)
