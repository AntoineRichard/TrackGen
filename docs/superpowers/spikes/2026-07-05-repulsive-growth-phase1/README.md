# Repulsive-Growth Phase-1 Generator Spike — 2026-07-05

Tests the generation strategy of Henrich et al., *"Generating Race Tracks With
Repulsive Curves"* (IEEE 10645670; built on Yu/Schumacher/Crane, *Repulsive
Curves*, SIGGRAPH 2021) as a **phase-1 centerline generator** for `track_gen`.

The idea: start each env from a small circle (embedded by construction), ratchet
a per-env target length upward while a tangent-point (TP) energy keeps the curve
self-avoiding, confine growth to a disc domain seeded with random disc obstacles
(paper formulation: wall + obstacles are point rings with plain inverse-power
repulsion `p = beta - alpha`, wall weight `1.0`, inner discs `0.25`). The grown
centerlines are resampled to the runtime's **constant spacing `0.6·hw`** and then
run the **standard tail** (oracle XPBD relax → constant-width inflate) and are
scored against the runtime `bezier` generator.

This inverts the two documented TP-Sobolev failures (2026-06-17 bake-off,
2026-06-18 finisher): TP is never asked to fix a jagged curvature-limited curve
or untangle anything — the curve is embedded at every step and TP works in its
native separation-limited regime.

Throwaway spike. Batched torch over `E=64` envs, `N=256` points, dense `O(N^2)`
pairs. Deterministic (fixed seeds, no wall-clock in the math). Run on an
RTX 4090 via a dedicated CUDA venv (`.venv-gpu`):

```bash
.venv-gpu/bin/python docs/superpowers/spikes/2026-07-05-repulsive-growth-phase1/grow_tp.py --device cuda
```

## Artifacts

- `grow_tp.py`: the whole spike — obstacle sampling, growth flow, standard tail,
  validity/diversity scoring, figures.
- `growth-snapshots.png`: 4 envs × iteration snapshots — the circle growing and
  buckling into folds around obstacles.
- `grown-grid.png`: 16 grown-then-tailed tracks with constant-width inflation
  (green = valid, red = invalid).
- `baseline-grid.png`: the runtime `bezier` baseline for the same seeds.
- `pre-post-xpbd.png`: 16 envs, pre-XPBD (orange) and post-XPBD (green/red)
  centerlines overlaid, with the post-tail inflated borders, to see what the
  XPBD tail fixed (or failed to fix).

## Main Findings

The strategy **works as a generator** — repulsive growth produces genuine,
smooth, track-like closed loops that weave around obstacles — and once the tail
is **runtime-matched** (below) it clears the **fixed** half-width validity bar
(`hw = 0.1`) at **100% yield** while folding into rich multi-lobe circuits.

Final tuned config (defaults in `grow_tp.py`, after the **paper-recipe pass**
below): `r_dom_frac=0.35`, `dom_init_ratio=4`, `grow_mult=[4.5, 5.5]·init` (the
per-env target is now a multiple of the **init-circle** perimeter, reference
style, not of `P_ref`), `growth=0.008`, `tau=0.4`, `w_len=30`, `alpha=3`,
`beta=6`, obstacle `k_range=(8, 12)` (~9.9/env) at `r_frac=(0.02, 0.045)` of
`r_dom` (their 0.025 ratio), `disc_clearance=0`, inner discs **deactivated at
target length** (wall kept), 382 growth iters, and the tail fed a **per-env
constant-spacing 0.06 (= 0.6·hw) resample** (see "The spacing-mismatch fix").

| metric | baseline `bezier` | repulsive growth + tail (new) | (old defaults) |
|---|---|---|---|
| valid yield | 64/64 (100%) | **64/64 (100%)** post-tail (0/64 pre-tail) | 64/64 (8/64 pre-tail) |
| perimeter | 5.05 ± 0.88 | 13.20 ± 0.79 | 6.34 ± 0.59 |
| compactness (4πA/P²) | 0.425 ± 0.105 | **0.146 ± 0.012** (far foldier) | 0.260 ± 0.034 |
| max Menger curvature (median) | 24.8 | 8.3 | 8.0 |
| wall-clock (E=64) | 0.002 s | ~2.5 s (~1.0 s grow @ 2.6 ms/iter + ~1.5 s tail) | ~3.2 s |
| XPBD tail displacement (median) | — | 0.016 | 0.020 |

The perimeter jump (6.3 → 13.2) is deliberate: the target is now ~5× the init
circle in an `outerRadius = 4·initRadius` domain, so the curve **overfills the
domain circumference by ~1.28×** and must buckle into folds to fit. Compactness
halves again (0.260 → 0.146) — the tracks go from 2–4 gentle lobes to dense
7–9-fold serpentine circuits, the paper's aesthetic. See `grown-grid.png`.

Reading the numbers:

- **Yield 100%.** All 64 tracks are legitimate dogbone / S / Y / switchback
  circuits threading between the discs with clean constant-width inflation. This
  is up from the pre-fix 32/64 (50%) — the headline result of this pass.
- **Longer, foldier than the baseline.** With `grow_mult = [4.5, 5.5]·init` the
  curves *overfill the domain circumference* by ~1.28× (perimeter 13.2, well past
  the baseline's 5.05) and buckle into dense serpentine circuits (compactness
  0.146 vs 0.425 — far more elongated, 7–9 folds vs the baseline's 1–2 lobes).
- **Smoother, tighter-spread diversity.** Grown tracks have no tight hairpins
  (max curvature 8.0 vs 24.8) and every std is smaller than the baseline's:
  repulsive growth converges toward a family of smooth multi-fold loops rather
  than reproducing the baseline's spread. Shape *richness* (folds per track) is
  now high; shape *variance* is still below baseline.
- **~1600× slower** than the trivial `bezier` baseline, but still ~3 s for 64
  envs on GPU. Growth is the dense `O(N²)` TP + `O(N·M)` obstacle sums per iter;
  the per-env NaN-padded tail (bucketed `relax` + per-env validity/diversity
  loops, below) adds ~1.5 s over the old single dense batch — the price of
  matching the runtime's per-env-count contract.
- **The tail opens tight folds** (displacement 0.016): at the paper-recipe
  overfill the pre-tail folds pinch *below* `hw` (median thickness 0.047, 46/64
  with inflation-border crossings, 0/64 pre-tail valid), so the tail does more
  than polish — but `pre-post-xpbd.png` shows the grown (orange) and post-XPBD
  (green) *centerlines* nearly coincident: the tail gently opens sub-`hw` pinches,
  it does **not** untangle gross tangles. The centerlines were already
  well-formed dense mazes; only the lane width needed the final `hw` of room.

### Matching the paper's recipe (2026-07-05 pass)

The previous pass reached 64/64 but the tracks were visually *underwhelming* vs
the source paper — too few folds, and the curves kept an exaggerated distance
from the obstacles. We diffed our layout against the reference Unity
implementation (`racetrack-generation`, `EnergyCurve.cs` /
`MapGenToolBase.cs`) and closed the gaps:

| knob | reference (`EnergyCurve.cs`) | old spike | **new default** |
|---|---|---|---|
| inner-obstacle count | `numObstacles = 10` | 2–5 (3.6/env) | 8–12 (**9.9/env**) |
| obstacle / domain radius | `1 / 40 = 0.025` | 0.06–0.14 (2.5–5.5× bigger) | **0.02–0.045** |
| obstacle placement | evenly-spaced **angles** + random radius | 2D rejection sampling | **angular-stratified** (mirrors ref) |
| domain / init radius | `outerRadius = 4·innerRadius` | ~6 | **4** |
| target length | `lengthScale = 6` × init perimeter | ~3–4× init (`1.1–1.5·P_ref`) | **4.5–5.5× init** |
| repulsion-ring clearance | 0 (rings at the physical radius) | `0.6·hw` (+25–55% on the drawn disc) | **0** |
| deactivate after target | yes — **whole list, wall included** | none | **inner discs only (wall kept)** |

What each change did, one knob at a time (seed 11, E=64, keep-wall deactivation):

- **Growth target is the dominant fold lever (the user's suspicion was right —
  we were badly under-growing).** The old target (`1.1–1.5·P_ref`) is only
  ~0.6× the domain circumference — *under*-filled, so the curve never had to
  fold. Re-expressing the target as the reference's `grow_mult × init-circle`
  perimeter and pushing it up:

  | `grow_mult` | overfill (target / wall circ.) | fill fraction | post-tail yield | compactness | tail disp. |
  |---|---|---|---|---|---|
  | 4.5–5.5 **(default)** | 1.28 | 30% | **64/64** | 0.146 | 0.016 |
  | 5.0–6.0 | 1.40 | 33% | 63/64 | 0.133 | 0.023 |
  | 6.0–7.0 | 1.65 | 38% | 63/64 | 0.104 | 0.028 |

  Yield is a *soft* frontier, not a cliff — even overfill 1.65 holds 63/64 (the
  one failure is always a single genuine self-crossing born during growth).
  Higher overfill = denser mazes at the cost of ~1 env and more tail rescue.
  Default `4.5–5.5` keeps the full 64/64 while already tripling the fold count
  (compactness 0.260 → 0.146). The startup line prints overfill + fill fraction;
  fill stays ~30% even at `grow_mult 6–7`, so **capacity is never the binding
  constraint — yield is.** (Reference overfill is ~1.5; `grow_mult 6` reproduces
  it exactly, at 63/64.)

- **Smaller, busier, angular-stratified obstacles remove the "obstacles look
  bigger than they are" effect.** Matching the reference's 0.025 obstacle/domain
  ratio (down from 0.06–0.14) and its 10-per-env angular placement (up from 2–5
  rejection-sampled) gives a uniform obstacle field the curve threads *through*
  rather than a few big discs it detours around. Combined with **zero disc
  clearance** (rings now sit at the exact physical radius — the old `0.6·hw`
  inflated each drawn disc by a quarter to a half of its radius), the grown
  curve passes right up against the discs instead of holding a fat halo off them.

- **Deactivating the inner discs at target length (keep the wall) closes the
  halos.** With obstacles live through the whole run (`--no-deac`) the curve
  clusters asymmetrically and leaves a clear ring of empty space around every
  disc — the exact "exaggerated distance" the user flagged. Zeroing the inner
  disc weights once an env hits its length target (mirroring the reference's
  `deacObsAfterScaling` → `obstacles.ForEach(Disable)`) lets the settle phase
  (pure TP + length) relax those halos shut and fill the domain uniformly. It
  costs ~1 env of yield vs `--no-deac` at equal geometry (63 vs 64 at
  `grow_mult 5–6`), which is why the *default* pairs deactivation with the safer
  `grow_mult 4.5–5.5` to keep 64/64.

- **We must NOT deactivate the wall (the reference does).** Their code drops the
  *entire* obstacle list — wall included — because it stops the flow on stall
  (`numStuckIterations ≥ 3 && TargetLengthReached`). This spike runs a fixed
  iteration budget, so with the wall gone the pure-TP-at-fixed-length settle has
  hundreds of free iterations and **unfolds every curve back into a circle**
  (compactness → 0.999, `--deac-wall` reproduces this). Keeping the wall
  preserves the confinement that holds the folds. This is the one place we
  deliberately diverge from the reference, forced by the fixed-iteration design.

### The spacing-mismatch fix (what fixed the 50% → 100% jump)

The previous pass hit a **50% ceiling** it blamed on TP's sub-`hw` equilibrium
thickness. The real culprit was a **tail mis-tuning**: the runtime pipeline
resamples every centerline to **constant spacing `0.6·hw = 0.06`** before XPBD
(`warp_pipeline.py` calls `resample_constant_spacing(..., config.spacing)`, and
`config.spacing` defaults to `0.6·half_width`), so the Jacobi solver's per-track
rest length `L0` and its exclusion band are *calibrated for that spacing*. The
spike instead fed the tail the raw `N = 256` grown curve — spacing ≈ `P/256`
≈ `0.021`, ~3× finer. At over-fine spacing the exclusion band spans ~10 beads and
the tiny rest length makes the separation constraint over-correct into
**high-frequency sawtooth** on any pinched fold (visible in the old
`pre-post-xpbd.png` as red zigzag on envs 2/6/7/8/10).

**The fix:** resample the grown curve to **per-env constant spacing `0.6·hw`**
(NaN-padded `[E, n_max, 2]` with per-env counts, via the oracle's own
`G.arc_length_resample(grown, spacing=0.6·hw)` — the torch mirror of the runtime
Warp resampler) *before* the tail. Growth itself stays at `N = 256` (the TP flow
needs the resolution); only the tail input is coarsened. This single change lifts
post-tail yield **32/64 → 64/64** and eliminates the sawtooth entirely
(`pre-post-xpbd.png` is now all-smooth green). It also *inverts* the previous
pass's central conclusion: the roomy `r_dom_frac 0.40` domain was chosen only
because the mis-spaced tail *could not* open pinched folds. With the calibrated
tail, a **tighter `0.35` domain gives richer folds at the same 100% yield** — so
the "roomy domain softens the fold aesthetic" trade-off is gone.

**What the fix did *not* need:** none of the previously-hypothesised
thickness-gap levers (grow at coarser `N`, add a self-inflation energy term,
domain-aware tail). The equilibrium-thickness framing was a red herring; the tail
was simply being run out of calibration.

**Tail implementation shipped.** This spike runs the **torch oracle `relax`**
(the validated reference the runtime Warp XPBD is `allclose` to), **bucketed by
per-env bead count** so equal-count envs relax as one dense batch, then written
back NaN-padded. The alternative — wiring the *actual* runtime Warp tail
(`resample_constant_spacing → warp_relax.xpbd_solve → inflate`) — is a fidelity
win but needs hand-built `_Scratch` wp.array buffers, too gnarly for a throwaway
spike; the oracle is numerically equivalent and far simpler here.

### How far can the domain tighten?

At the shipped `r_dom_frac = 0.35` the tail is in polish mode (pre-tail 8/64,
displacement 0.020). Tightening further to `0.30` still yields **64/64** and
slightly richer folds (compactness 0.245), but pushes the pre-tail curves to
**0/64 valid with 63/64 self-crossed** — the tail is then doing full *rescue*,
untangling incipient crossings (displacement 0.034). It worked for all 64 here,
but that is the regime the 2026-06-17/18 studies flagged XPBD as fragile in, so
`0.35` is shipped as the default (rich folds, tail still polishing) with `0.30`
available via `--r-dom-frac` for the foldier-but-riskier aesthetic.

## Failure modes observed

First-attempt diagnoses (from the initial 0/64 run, fixed in `grow()` /
`sample_obstacles` before this pass — recorded because they are part of the
record):

1. **Zero obstacles placed, silently.** Rejection-sampling was unsatisfiable —
   `r_init` was too large relative to the domain, so every candidate disc was
   rejected and the "random obstacle" layouts were empty. Fixed by re-deriving
   `r_init = P_ref / (2π·2.75)` and checking the rejection inequality is
   satisfiable given `c_frac·r_dom`.
2. **Domain undersized.** `r_dom` circumference was smaller than the target curve
   perimeter, so the wall ring crushed the curve inward. Fixed by sizing
   `r_dom = r_dom_frac · P_ref` and (this pass) tuning `r_dom_frac` up to 0.40.
3. **Length dumped into high-frequency sawtooth.** A wide thickness-style TP
   pair-exclusion band (±26 pts) hid sub-band wavelengths from the TP kernel, so
   a hard uniform length-rescale poured the ratcheted length into invisible
   high-frequency wiggle. Fixed with a constant ±2 exclusion (the TP wedge factor
   discounts along-curve neighbours itself).

Failure modes hit *during this tuning pass* (diagnosed with the instrumented
probes, not blind-tuned):

4. **Clearance-vs-margin overlap → energy blowup.** The repulsion rings sit at
   `r + clearance` (to leave the grown curve inflation room), but the rejection
   test only cleared the *physical* radius `r`. With `clearance (0.06) >
   0.05·r_dom (0.042)` the inner rings overlapped the start circle, so the `p=3`
   obstacle energy hit **31 million** at iteration 0 and instantly shredded the
   curve. Fixed: rejection now requires `rad ≥ r_init + r + clearance + 0.05·r_dom`.
5. **Soft length penalty is hopelessly outgunned → curve collapses.** The
   original design used a soft penalty `w_len·((L−L_target)/L_init)²`. Per-term
   preconditioned gradient probes showed the wall/obstacle gradient exceeds the
   length gradient by **10³–10⁵×** (iter 0: obstacle 6.8e5 vs length 3.2). The
   curve shrinks *away* from obstacles until TP self-repulsion balances them at a
   tiny equilibrium (P ≈ 0.7); the length term never participates. To counteract
   the obstacle gradient the penalty would need `w_len ≈ 3000+`, which would
   overpower TP and self-collide. **`w_len` tuning alone cannot work.** Fixed by
   switching `grow()` to the paper's actual method: a **hard** length constraint
   via Sobolev-orthogonal projection + rescale-to-target (the oracle `_tp_flow`
   step), which — crucially — does *not* reproduce failure mode 3, because the
   projection keeps the enforced growth in low modes. `w_len` is kept as a small
   inert regularizer / live knob.
6. **Overfill → folds pinch to zero width.** With the hard constraint and a tight
   domain (`r_dom_frac 0.17`), the curve grows to the full target but the folds
   pinch together (thickness 0.010, 8/8 crossings) — the target perimeter is ~94%
   of the domain circumference, leaving no room for hw-wide lanes. Mid-growth
   snapshots (P ≈ 3.7) looked great; the *final* target overfilled. Mitigated by
   the roomy domain.
7. **Residual pinches & tail sawtooth (the old 32 invalid) — FIXED this pass.**
   Originally two flavours: *near-miss pinches* (one spot dips under hw) and
   *tail-generated sawtooth* on the tightest pre-tail curves. Both were symptoms
   of the **spacing mismatch**: the tail was fed the `N=256` grown curve at
   ~3× the runtime's calibrated spacing, so XPBD's separation constraint
   over-corrected into high-frequency zigzag instead of gently opening folds.
   Resampling the tail input to per-env constant spacing `0.6·hw` (Main Findings →
   "The spacing-mismatch fix") eliminated the sawtooth and cleared the near-miss
   pinches too — post-tail yield 32/64 → 64/64. This was *not* a domain-awareness
   problem as originally guessed; it was a calibration mismatch.

## Resume Points

- **Reference mechanisms we still don't replicate.** From `EnergyCurve.cs` /
  `MapGenToolBase.cs`, beyond the layout knobs now matched (obstacle recipe,
  geometry ratios, deactivation): (1) **stop-on-stall** — they end the flow when
  `numStuckIterations ≥ 3 && TargetLengthReached`; we run a fixed budget, which
  is *why* we can't deactivate the wall (see the paper-recipe section). Porting a
  stall detector would let us mirror their full wall-inclusive deactivation
  without circle-ifying. (2) **Exact line search + backprojection** — they choose
  the step by line search and correct constraint drift by backprojection each
  iter; we take a fixed `tau·mean_seg_len/‖g‖` step and hard-rescale to the
  target. (3) **Curve subdivision** — they subdivide when the average edge length
  doubles (`subdivideLimit = 2`); we grow at fixed `N = 256` + periodic resample.
  (4) Their inner-obstacle radial range runs all the way to the wall
  (`outerRadius`); we cap at `c_frac = 0.9·r_dom`. None of these blocked the
  visual match, but a faithful Warp port would want (1)–(3).
- **BVH far-field, keep it all in Warp.** The dense `O(N²)` TP pairs and `O(N·M)`
  obstacle sums are trivial at E=64/N=256 but won't scale. A production port
  should evaluate the TP (and obstacle) far-field with `wp.Bvh` — per-edge AABBs,
  refit per iteration — instead of dense pairs, keeping the whole growth loop in
  Warp. This is the same near/far split the *Repulsive Curves* paper uses
  (Barnes–Hut); the BVH gives it on GPU.
- **Torch prototype vs the pure-Warp generator contract.** The phase-1 generator
  contract (`track_gen/_src/generator_registry.py`) is pure-Warp and
  graph-capturable. This spike is torch (autograd on the TP/obstacle energy). A
  real integration would either (a) port the constrained flow to hand-written
  Warp gradients + BVH so it stays graph-capturable, or (b) integrate eagerly
  outside the capture, exactly like the `tp_sobolev_checkpointed` idea in the
  2026-06-23 spike — capture phase 0/inflation, run the torch growth phase
  eagerly between them.
- **Obstacles as gameplay content.** The per-env disc layouts are already random
  gameplay-relevant geometry. Export them as props / `DiscChecker` obstacles so
  the phase-1 *layout* becomes actual track content (the curve is guaranteed to
  weave around them, not just avoid them abstractly).
- **Thickness gap — resolved, not a gap.** The previous pass thought the yield
  ceiling was the TP-thickness / `hw` scale mismatch and listed levers (coarser-N
  growth, a self-inflation energy term, a domain-aware tail). None were needed:
  the ceiling was a **tail spacing mis-calibration** (Main Findings). At the
  runtime-matched spacing yield is 100% and the domain has *tightened* (0.40 →
  0.35) for richer folds, the opposite of what the "thickness gap" framing
  predicted. The equilibrium-thickness concern was a red herring.
- **Ship the real Warp tail for full fidelity.** This spike's tail is the torch
  oracle `relax` bucketed by per-env count (numerically equivalent to the runtime
  Warp XPBD). A follow-up should wire the actual runtime tail
  (`warp_pipeline.resample_constant_spacing → warp_relax.xpbd_solve →
  inflate_warp`) so the spike exercises the exact production path end-to-end —
  worthwhile once the growth phase itself is ported to Warp (BVH point above).
- **Push the tighter-domain / rescue regime.** `r_dom_frac 0.30` yields 64/64 with
  the tail *untangling* self-crossed pre-tail curves (0/64 pre-tail valid). If
  that holds across seeds it reopens the paper's tight-domain "many folds"
  aesthetic — but it is exactly the XPBD-on-crossings regime the 2026-06-17/18
  studies flagged as fragile. Worth a seed sweep before trusting it as a default.
