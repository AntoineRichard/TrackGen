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
- `grow_warp.py`: **pure-Warp port of the growth phase** (see "Warp port" below).
  Reuses `grow_tp`'s obstacle sampling + tail + scoring verbatim; the growth loop
  itself is Warp kernels only, no torch. Runs torch-vs-Warp parity at `E=64` and a
  perf sweep at `E=64/1024/8192`.
- `warp-parity-grid.png`: torch (top row) vs pure-Warp (bottom row) grown-then-tailed
  tracks for the same seeds — qualitatively identical serpentine circuits.
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

## Warp port (2026-07-05) — the growth phase runs pure-Warp

`grow_warp.py` ports the whole `grow_tp.grow()` flow to NVIDIA Warp kernels (flat
`[E*N]` `vec2f` buffers, `env = tid // N`, no host sync in the loop — the
`warp_relax` / `warp_pipeline` conventions). Only the growth loop is ported; the
obstacle layout sampling + sizing and the XPBD tail / validity / diversity scoring
are imported verbatim from `grow_tp` (same `SEED`, so layouts and `L_final` are
byte-identical to the torch run). This de-risks a future production `repulsive`
generator: the numerically-heavy inner loop is now proven fast and batched in Warp.

**How the pieces map to Warp:**

- **TP + obstacle + length-penalty gradient — `wp.Tape`, not hand-derived.** The
  total energy (tangent-point pairs + inverse-power obstacle rings + the small
  length-ratchet penalty) is one differentiable Warp kernel set that accumulates a
  scalar via `atomic_add`; the gradient comes from `wp.Tape` (Warp's own reverse-mode
  autodiff). This was the design's suggested starting point and it **just worked** —
  a `wp.Tape` grad matches `torch.autograd.grad` on the identical energy to **~2e-7
  relative** (smoke test), so there was no reason to hand-derive the (messy, `w`- and
  `T`-dependent) analytic TP gradient. Dense `O(N²)` pairs per env with the paper's
  constant `±2` circular exclusion, exactly like torch.
- **Fractional-Sobolev preconditioner WITHOUT FFT.** Warp has no FFT, so the inverse
  ring-Laplacian filter `1/(λ_k^s + ε)` is turned into its **real-space circulant
  first row** once on the host (`numpy.fft.irfft` of the spectral filter), uploaded as
  a `[N]` `wp.array`, and applied as an `O(N²)` **circular-convolution kernel**
  `A^{-1}g[i] = Σ_j h[(i−j) mod N]·g[j]`. Same cost class as the TP energy. Verified
  against torch's `rfft` version on random input: **max abs diff 1.2e-4** on a
  ~150-scale signal (**~8e-7 relative**).
- **Per-env reductions** (perimeter, barycenter, `gmax`, `⟨g,lg⟩`, `⟨lg,A⁻¹lg⟩`,
  mean segment length) are small dedicated one-thread-per-env kernels serial over
  `N=256`, `warp_relax`-style.
- **Resample** reuses `warp_pipeline.resample_uniform` (the fixed-`N` sibling of the
  NaN-aware arc-length resampler) with `count=N` — zero-alloc, sync-free.
- **Ratchet + deactivation state** (`L_target`, `reached`) live in device arrays; the
  ratchet is a one-line kernel and the inner-disc deactivation is a data-driven branch
  *inside* the obstacle kernel (`reached[e]`), never host logic.

**Parity at `E=64` (same seed as torch):**

| metric | torch `grow_tp` | pure-Warp `grow_warp` |
|---|---|---|
| post-tail valid yield | 64/64 | **64/64** |
| compactness (4πA/P²) | 0.146 ± 0.012 | **0.146 ± 0.012** |
| max Menger curvature (median) | 8.3 | 8.3 |
| perimeter | 13.20 ± 0.79 | 13.20 ± 0.78 |

Yield and every diversity statistic match to the printed precision, and
`warp-parity-grid.png` shows the two families are the same serpentine circuits
threading the same obstacle fields. **Exact float parity is not expected and not
achieved**: the *per-bead* pre-tail centerline displacement between torch and Warp is
~0.06 (median), because the growth flow is mildly chaotic — accumulated float / op-order
differences over ~382 iters, compounded by the periodic arc-length **re-parameterization**
(every 25 iters, which shifts which bead lands where), drift the point *positions* while
leaving the *shape* statistics identical. This is exactly the "qualitative + yield parity,
not float parity" bar the task set.

**Performance** (RTX 4090, post-warmup, module load excluded, default config, ~382 iters):

| E | iters | total | ms/iter | peak GPU |
|---|---|---|---|---|
| 64 | 382 | 0.53 s | 1.39 ms | 4.6 MiB |
| 1024 | 383 | 3.49 s | 9.10 ms | 73.7 MiB |
| 8192 | 383 | 24.5 s | 63.9 ms | 589.8 MiB |

Torch growth at `E=64` is 0.96 s (2.52 ms/iter), so Warp is **~1.8× faster at E=64** —
modest, and honestly so: at small `E` the loop is **host-launch-bound**, not
compute-bound. Each iteration issues ~13 tiny kernel launches **plus a fresh `wp.Tape`
record + `tape.backward`** (which itself expands to a sequence of adjoint launches), and
those tiny kernels don't fill a 4090. The Warp advantage grows with batch: `E=64→1024` is
16× the envs for only 6.5× the time (launch overhead amortizes), and `1024→8192` is 8× envs
for ~7× time (now compute-bound, dense `O(N²)` scaling). Memory stays tiny (590 MiB at
`E=8192`). The real payoff of the pure-Warp loop is not the 1.8× — it is that it is the
prerequisite for **CUDA graph capture**, which would erase precisely the per-iter launch
overhead that dominates the small-`E` regime.

### Graph-capture feasibility (not implemented — assessed)

The task asked what blocks CUDA graph capture of this loop. Assessment (capture **not**
implemented, per instructions):

- **No host syncs in the loop** — confirmed. All state kernels are async; the resample
  uses the fixed-`N` `resample_uniform` path (no `count` readback, unlike
  `resample_constant_spacing`'s truncation-warning readback); the iteration count `n_iters`
  is computed **once before** the loop from `L_final.max()`. Nothing reads back mid-loop.
- **The one real blocker is the per-iteration `wp.Tape`.** Creating `wp.Tape()` and calling
  `tape.backward()` every iteration both *allocates* (adjoint buffers / launch records) and
  does host-side recording — illegal inside a capture region. **But the launch topology is
  iteration-invariant here** (same dims, same arrays, fixed `±2` exclusion, static
  deactivation branch that lives *inside* the kernel), so the fix is standard: **record ONE
  persistent tape before capture and replay its forward+backward inside the graph**. Warp
  supports capturing tape replay. After that, the loop body is a fixed sequence of ~13
  forward/adjoint kernels + 2 resample kernels — fully capturable.
- **Everything else is already capture-clean**: all buffers pre-allocated (zero alloc in the
  loop except the tape), the obstacle deactivation is a data-driven in-kernel branch (not host
  control flow), and the resample cadence is a *static* `(it+1)%25` schedule (unrollable into
  the captured graph since `n_iters` is fixed).
- **The alternative that makes capture trivial**: hand-write the analytic adjoint kernels and
  drop `wp.Tape` entirely. Not needed for the spike (the tape gave exact-parity gradients for
  free), but a production port that wants a single captured graph would either persist one tape
  or go analytic. This is the concrete integration decision the port surfaces.

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
- **Growth phase is now ported to pure Warp — DONE (`grow_warp.py`).** The
  numerically-heavy loop runs as Warp kernels with `wp.Tape` gradients, at parity
  with torch (64/64, identical diversity) and ~1.8× faster at `E=64` / linear-scaling
  to `E=8192` (63.9 ms/iter, 590 MiB). See the "Warp port" section. Remaining work to
  make it a production generator: **(a)** make it graph-capturable by persisting one
  `wp.Tape` (or hand-writing analytic adjoints) — the *only* capture blocker is the
  per-iter tape (assessed in "Graph-capture feasibility"); **(b)** wire it behind the
  `generator_registry.py` pure-Warp contract; **(c)** the BVH far-field below for scale.
  The old torch-eager-outside-capture fallback (option b in the 2026-06-23
  `tp_sobolev_checkpointed` idea) is no longer the only path — the loop is already pure
  Warp.
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
