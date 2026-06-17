# Constant-Width Track Generation via Centerline Relaxation — Design

**Date:** 2026-06-17
**Status:** Proposed (pre-implementation)
**Supersedes the inflation strategy of:** `2026-06-16-batched-track-inflation-design.md`
**Scope:** Replace the failed "shrink the width to rescue a fixed centerline" inflation strategy with a **bead-chain relaxation** stage that reshapes the centerline until a **constant** track width fits everywhere with no self-intersection. Ship **three selectable, batched, GPU+CPU relaxation backends** (XPBD projection — default; energy-gradient; tangent-point/Sobolev) behind one interface, an optional TP-Sobolev smoothing finisher, a **benchmark harness sized for 8192 tracks**, and the package/validity fixes the spike exposed.

---

## 1. Goal & context

`track_gen` is a GPU-batched closed-curve race-track generator for Isaac Lab RL environments. The prior design (`2026-06-16-...`) added an **inflation** stage that holds the generated centerline fixed and derives a *variable* half-width clamped by curvature (`w = min(w_max, α/κ)`) and optional self-distance. This **does not work**, and a measured bake-off (Appendix A) shows why:

- At `half_width = 0.03`, `scale = 1.0`, **2/64 tracks (3%) are valid.** The Bézier centerlines have curvature radii ~0.005–0.03 — far below the requested width — so the curvature clamp collapses the width to ~10% of target, producing useless near-zero-width ribbons.
- **The validity gate is blind to border crossings.** It checks turning-number + width-floor + NaN, none of which detect that inner/outer borders self-cross. It reported 100% valid while 59/64 tracks had crossing borders (173 crossings total). A silent failure.
- Some generated centerlines **self-intersect**, and the weak turning-number gate (`turning_tol=0.35`) lets them through.

**Root cause:** the strategy tries to fit a constant downstream constraint (a width) to a fixed, over-sharp, sometimes-self-intersecting upstream shape. The fix is to **reshape the centerline** so the width fits — the user's "rope/chain relaxation" idea.

### The unifying principle: curve thickness

The property we need — *"consecutive points stay at fixed spacing; non-consecutive points never come closer than the track width"* — is the **thickness** (a.k.a. reach / normal injectivity radius; Gonzalez–Maddocks ropelength theory) of a curve:

> `thickness(γ) = min( min-radius-of-curvature, ½ · doubly-critical-self-distance )`

A single guarantee — **thickness ≥ `half_width`** — simultaneously prevents (a) the inner border folding at a sharp corner (local curvature) and (b) distant strands overlapping (global self-approach). If we relax the centerline until `thickness ≥ half_width`, then a **constant-width** offset `± half_width` is valid by construction: no clamping, no rejection.

### What the bake-off taught us (drives the design)

Three solvers were prototyped on an identical, fair harness (Appendix A). Decisive findings:

1. **The binding constraint here is LOCAL CURVATURE, not separation.** 59/64 tracks are curvature-limited at init; only 5 are separation-limited. The dominant relaxation lever is **corner-rounding (bending)**, with pairwise separation secondary.
2. **XPBD projection wins** as the default: 64/64 valid (hard guarantee), near-zero shape distortion (median per-point displacement 0.0023 ≈ 0.08·`half_width`), converges by ~100–150 sweeps, deterministic, hash-grid-scalable.
3. **Pure tangent-point/Sobolev energy is the wrong tool for this *regime* (but kept as a backend):** it's a *repulsion* energy, so it can only fix a sharp corner by inflating the whole loop toward a circle (72% valid, 1.4–5.4× area growth — tracks visibly balloon into circles). It would shine when the binding constraint is genuine distant-strand collision (figure-eights, dense packing) — hence we ship it as a selectable backend for benchmarking and other regimes, not the default.
4. **The hybrid (any feasible curve → a few tangent-point/Sobolev steps) is the only config both 100%-valid AND smoother than pure XPBD** (clearance coefficient-of-variation 0.025 vs 0.040), at ~20% extra wall-clock and a modest shape budget.

All three are now shipped as **selectable backends** so they can be benchmarked head-to-head at scale (8192 tracks); XPBD is the default until the at-scale benchmark says otherwise.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Success criterion | **Validity-first, constant width.** Relax until `thickness ≥ half_width` everywhere; let shapes round out where they must; aim ~100% usable tracks. |
| Solver backends | **Three, selectable via `relax_solver`:** `xpbd` (default), `energy` (Adam soft-penalty), `tp_sobolev` (tangent-point + fractional-Sobolev flow). One shared interface; all batched, device-agnostic (GPU+CPU), env-chunked. |
| Default | **`xpbd`** (bake-off winner). |
| Optional finisher | **Tangent-point/Sobolev smoothing**, `smooth_finish` flag (default **off**), applied after the chosen backend; evens clearance / smooths curvature. |
| Width model | **Constant** `half_width` (curvature & self-distance width-clamps removed from inflation). |
| Validity | Real check: `thickness ≥ half_width` **AND zero border self-intersections** AND turning ≈ 2π AND no-NaN. Replaces the border-blind gate. |
| Generator | Add a **real simplicity gate** (reject self-intersecting centerlines) — relaxation by repulsion cannot untangle a figure-eight, so the init must be simple. |
| Benchmark | **`benchmark_relaxation.py`** runs all three backends at **E=8192** on **GPU (primary) + CPU (fallback)**, reporting validity yield, wall-clock, **peak GPU memory**, and shape-quality metrics. |
| Scale | **Env-chunking** of the dense `O(E·N²)` term (config `relax_chunk_size`) so 8192×256 fits in GPU memory; banded/hash-grid neighbor query is documented future work. |
| Packaging | **Proper installable package** (`pyproject.toml`, `track_gen/` package dir) — fixes `types.py` shadowing stdlib `types` and the `TrackGen`-vs-`track_gen` import break. |

## 3. Architecture

```
TrackGen/                              # repo root
├── pyproject.toml                     # NEW: name=track_gen, deps, optional warp, dev extras
├── track_gen/                         # NEW package dir (modules moved here)
│   ├── __init__.py                    # public API (+ relax, relaxation backends)
│   ├── rng_kernels.py / rng_utils.py  # unchanged (Warp RNG)
│   ├── geometry.py                    # primitives (+ thickness / self_intersections / separation_min)
│   ├── generators.py                  # Bezier + Fourier (+ real simplicity gate)
│   ├── relaxation.py                  # NEW: relax() dispatcher + 3 backends + TP/Sobolev finisher
│   ├── inflation.py                   # constant-width offset + real validity gate
│   ├── types.py                       # TrackGenConfig, Track (no stdlib shadow under the package)
│   └── track_generator.py             # facade: generate -> resample -> relax -> inflate
├── benchmarks/
│   └── benchmark_relaxation.py        # NEW: 8192-track backend benchmark (GPU + CPU)
├── tests/                             # CPU-testable suite (conftest path-hack retired)
└── viz/
```

**Pipeline (data flow):**

```
PerEnvSeededRNG ─► generator.generate(ids) ─► Centerline[E,M,2] + valid (SIMPLE by construction)
                                                   │  arc-length resample to N (uniform spacing)
                                                   ▼
            relax(center, config)  ── dispatch on config.relax_solver ──►  reshaped centerline
              {xpbd | energy | tp_sobolev}    [+ optional smooth_finish]    (thickness ≥ half_width)
                                                   ▼
                          inflate(center, config) ─► Track (constant ± half_width, real validity)
```

## 4. Relaxation stage (`relaxation.py`)

Pure batched torch, device-agnostic (CPU+GPU), CPU-testable, **no RNG** (deterministic). Reference spikes: `docs/superpowers/spikes/2026-06-17-relaxation-bakeoff/{relax_xpbd,relax_energy,relax_tpsobolev}.py`.

### 4.0 Backend interface & dispatcher

```python
def relax(center: Tensor[E, N, 2], config) -> Tensor[E, N, 2]:
    """Reshape a closed, arc-length-uniform centerline so thickness >= half_width.
    Dispatches on config.relax_solver in {"xpbd","energy","tp_sobolev"}; runs the
    optional smooth_finish pass; returns the relaxed centerline (same N).
    Validity is decided downstream in inflation."""
```

Each backend implements `_relax_<name>(center, config) -> Tensor[E,N,2]` with the **same signature and contract** (closed loop, same N, device preserved). The dispatcher applies env-chunking (§4.6) and the finisher (§4.5) uniformly so backends only implement the core sweep/step.

### 4.1 Shared model & constants
- Closed bead-chain `x: [E, N, 2]`, indices wrap. Init = the arc-length-resampled centerline (uniform spacing, so `band` is well-defined).
- `D = 2 · half_width` (centerline-to-centerline clearance so the two `half_width` borders don't touch); `R_min = half_width` (min radius of curvature so the inner border doesn't fold).
- `L0[e] = perimeter(center_init)[e] / N` (per-track rest spacing, fixed from init).
- `band[e] = round(D / L0[e])` (excluded index half-window; default derived, config-overridable). Pairs within `band` indices are legitimately close (consecutive beads) and excluded; pairs beyond `band` that are Euclidean-close signal a too-sharp corner or a self-approach.
- Shared "thickness ≥ target" early-stop check: `target = (1 − relax_tol)·half_width`.

### 4.2 Backend `xpbd` (default) — position-based constraint projection
Reference: `relax_xpbd.py` (64/64). Per sweep, three constraints summed into one per-bead displacement:
1. **Bending (dominant).** Per-bead Menger radius `r_i`; if `r_i < R_min`, pull apex toward neighbours' midpoint by `clamp((R_min−r_i)/R_min,0,1)·(mid−apex)`. **Corner-flip clamp:** the applied step (after `bend_relax`) is scaled so its length never exceeds `|apex−mid|` — a bead can't cross the chord midpoint and flip the corner. This clamp is what makes strong bending (`bend_relax≈1.5`) stable.
2. **Separation.** Non-adjacent pairs (`circ_index_dist > band`) closer than `D·(1+margin)` pushed symmetrically apart, **Jacobi-averaged by per-bead violated-pair count**, under-relaxed (`sep_relax`).
3. **Spacing / inextensibility.** Each edge toward `L0`, halved (2 edges/bead), under-relaxed (`spc_relax`).

`x ← x + sep_relax·sep + spc_relax·spc + clamp(bend_relax·bend)`. Hard guarantee, deterministic, minimal distortion.

### 4.3 Backend `energy` — differentiable energy minimization
Reference: `relax_energy.py` (62/64). Bead positions are optimization variables (`requires_grad`); minimize, with **Adam** (or L-BFGS), batched over envs:
`E = w_sep·Σ relu(D−d_ij)² + w_len·Σ(‖edge‖−L0)² + w_bend·Σ‖x_{i+1}−2x_i+x_{i-1}‖² + w_anchor·Σ‖x−x0‖²` (pairwise term masked by `band`). **Soft** constraints → only approximately feasible (residual hairpin crossings; no hard guarantee), hyperparameter-sensitive (lr, weights). Shipped for benchmarking and the differentiability it offers (could later backprop through generation). Per-track early stop on the thickness target.

### 4.4 Backend `tp_sobolev` — tangent-point + fractional-Sobolev flow (standalone)
Reference: `relax_tpsobolev.py`. Gradient flow on the tangent-point energy (kernel `|(y−x)⊥T_x|^α / |y−x|^β`, `α=2,β=4.5`) **preconditioned** by a fractional graph-Laplacian of order `s=(β−1)/(2α)=0.875`. On arc-length-uniform points the ring Laplacian is **circulant** → the preconditioner solve is a per-mode FFT (`O(E·N·logN)`, <1% of step; the O(N²) energy dominates). Length held ~fixed (gradient projected against length gradient + barycenter pin); per-track early stop. **Caveat (documented):** in the curvature-limited regime it over-rounds toward circles (72% valid, large area growth) — *not* the default; available as a backend for benchmarking and separation-limited regimes.

### 4.5 Optional finisher (`smooth_finish`, default off)
After the chosen backend, run `smooth_finish_iters` (≈8) tangent-point/Sobolev steps (§4.4 machinery) on the (ideally feasible) curve to even out clearance / smooth curvature. Must not break feasibility when warm-started from an `xpbd`-feasible curve (bake-off: stays 64/64, clearance CV 0.040 → 0.025). When the upstream backend left a track infeasible, the finisher is best-effort and the validity gate still decides.

### 4.6 Env-chunking for scale (the 8192 requirement)
The dense separation/energy term is `[E,N,N]`; at `E=8192, N=256` that is ~2 GB/tensor. The dispatcher processes envs in chunks of `relax_chunk_size` and concatenates — results are identical to unchunked (envs are independent). Default is `None` (no chunking, simplest/correct for small E); the benchmark (§9) sweeps `relax_chunk_size` and documents a recommended value for large E. This bounds peak GPU memory for the benchmark and production at scale. (Banded/neighbor-list separation that drops the `N²` term is documented future work.)

### 4.7 Iteration control & defaults
- **Per-track early stop:** each sweep/step, compute `thickness`; freeze tracks at `thickness ≥ target`. Stop when all frozen or the per-backend max iters reached. Saves compute and prevents the over-iteration regression seen at 200 XPBD sweeps.
- Final arc-length resample to `N` uniform points.
- Defaults (from ablation): xpbd `relax_iters=150` (knee ≈100), `sep_relax=1.0`, `spc_relax=1.0`, `bend_relax=1.5`, `margin=0.15`; energy `energy_steps=800`, `energy_lr=3e-3`, `w_sep=80, w_len=8, w_bend=1, w_anchor=0.01`; tp `tp_iters=100, tp_tau=0.7`; finisher `smooth_finish_iters=8, smooth_finish_tau=0.2`; `relax_tol=0.02`.

## 5. Generator change (`generators.py`)

Add a **real simplicity gate** to the Bézier bounded-regen loop: accept a candidate centerline only if its closed polyline has **zero self-intersections** (vectorized segment-crossing test, batched `[E,N,N]`), alongside the existing min-angle / turning gates (and tighten `turning_tol`). Relaxation by repulsion cannot untangle a self-crossing init, so the generator must hand the relaxer a simple loop. Fourier (smoother at low K) remains an alternative init.

## 6. Inflation change (`inflation.py`)

- **Width → constant.** Remove the curvature clamp and self-distance clamp; `w = half_width` everywhere.
- **Validity → real.** `valid = (thickness ≥ (1−tol)·half_width) AND (border self-intersections == 0) AND (|turning| ≈ 2π) AND no-NaN`. The border self-intersection test (inner ∪ outer) fixes the silent-failure bug. Failing tracks → `valid=False`, never crash.
- Tangent/normal/arclen/length assembly unchanged.

## 7. Geometry additions (`geometry.py`)

Promote bake-off primitives to library functions (pure batched torch, CPU-testable): `self_intersections(poly) -> [E]`, `separation_min(center, band) -> [E]`, `curvature_radius_min(center) -> [E]`, `thickness(center, band) -> [E]`.

## 8. Config additions (`TrackGenConfig`)

```
# --- Relaxation: backend selection + scale ---
relax_enable: bool = True
relax_solver: str = "xpbd"           # {"xpbd","energy","tp_sobolev"}
relax_chunk_size: int | None = None  # env-chunk the dense [E,N,N] term (None = no chunk)
relax_tol: float = 0.02              # target = (1 - tol) * half_width
relax_band: int | None = None        # None => round(D / L0) per track
# xpbd
relax_iters: int = 150
relax_sep_relax: float = 1.0
relax_spc_relax: float = 1.0
relax_bend_relax: float = 1.5
relax_margin: float = 0.15
# energy
energy_steps: int = 800
energy_lr: float = 3e-3
energy_w_sep: float = 80.0
energy_w_len: float = 8.0
energy_w_bend: float = 1.0
energy_w_anchor: float = 0.01
# tp_sobolev (standalone backend + finisher share tp_alpha/tp_beta)
tp_iters: int = 100
tp_tau: float = 0.7
tp_alpha: float = 2.0
tp_beta: float = 4.5
# optional finisher
smooth_finish: bool = False
smooth_finish_iters: int = 8
smooth_finish_tau: float = 0.2
```
Removed/deprecated: `alpha` (curvature width-clamp), `clamp_self_distance`, `self_distance_*` (superseded by relaxation).

## 9. Benchmark harness (`benchmarks/benchmark_relaxation.py`)

Standalone script (not a unit test). Generates a fixed batch of **E=8192** simple Bézier centerlines (seeded; cached to disk so runs are comparable), then for each backend in `{xpbd, energy, tp_sobolev}` (and optionally `+smooth_finish`) runs the relax→inflate pipeline and reports a table:

- **Validity yield:** % valid (thickness ≥ target AND zero border/centerline crossings).
- **Wall-clock:** end-to-end relax time (and per-iteration), warm-start excluded; report median of N repeats.
- **Peak GPU memory:** `torch.cuda.max_memory_allocated` per backend (reset between runs); sweep `relax_chunk_size` to show the memory/time tradeoff.
- **Shape quality:** median per-point displacement, clearance CV (std/mean of half-clearance), max-curvature distribution.
- **Devices:** GPU (primary) and CPU (fallback/CI); auto-detect CUDA, fall back cleanly. Optional summary figures saved under `benchmarks/out/`.

The 8192×256 dense term (~2 GB/tensor) is the reason `relax_chunk_size` exists; the benchmark documents the chosen default by sweeping it.

## 10. Error handling, robustness, determinism

- Relaxation is RNG-free → bit-reproducible (xpbd/tp_sobolev); the `energy` backend is deterministic given a fixed init and seeds (no stochastic sampling in the optimizer).
- Corner-flip clamp + Jacobi under-relaxation prevent XPBD blow-ups; per-track early stop bounds work and prevents over-iteration drift; soft `energy` residual failures are caught by the validity gate, never crash.
- `safe_normalize`/`clamp_min` floors on every division; non-converged tracks → best-effort geometry + `valid=False`.
- Env-chunking is numerically identical to unchunked (independent envs).
- Generator simplicity gate runs inside the existing bounded regen loop (no new hang risk).

## 11. Packaging (`pyproject.toml` + `track_gen/` dir)

Move modules into `track_gen/`. Once `types.py` is `track_gen/types.py` it's imported as `track_gen.types` and **no longer shadows stdlib `types`** (the shadow only occurred because the package dir was a top-level `sys.path` entry containing `types.py`). `pyproject.toml`: `name="track_gen"`; runtime deps `torch`, `scipy`, `numpy`; optional extra `warp=["warp-lang"]` (RNG/generation only — geometry/relaxation/inflation are warp-free); dev extra `pytest`, `matplotlib`. `uv pip install -e .` makes `import track_gen` work from anywhere, retiring the `conftest.py` path hack.

## 12. Testing strategy (TDD)

Most tests run on CPU with no GPU/Warp (relaxer, inflation, geometry are warp-free).

- **`geometry.py`:** `self_intersections` (figure-eight > 0, convex loop == 0); `thickness` of a radius-`r` circle == `r`; `separation_min`/`curvature_radius_min` on known shapes.
- **Each backend (synthetic CPU centerlines):** a sharp-cornered polygon relaxes to `min-radius ≥ half_width`; a deliberate near-touch relaxes to `separation ≥ 2·half_width`; xpbd corner-flip clamp prevents flip on a hairpin; determinism (identical output on repeat); per-track early stop freezes converged tracks; pathological input left best-effort (no raise/NaN).
- **Dispatcher + chunking:** `relax_solver` selects the right backend; **chunked output == unchunked output** (independent envs) within float tolerance.
- **Regression bake-off (harness as test):** on fixed simple Bézier centerlines, `xpbd` `relax → inflate` yields **100% valid** with median displacement below a bound; `energy`/`tp_sobolev` meet their documented yields (so a regression that silently breaks a backend is caught).
- **Validity gate:** a deliberately self-crossing border input is flagged `valid=False` (regression for the silent-failure bug).
- **Generator simplicity gate:** a self-intersecting candidate is rejected.
- **Finisher:** `smooth_finish=True` after `xpbd` keeps 100% valid and reduces clearance CV vs off.
- **Benchmark smoke test:** `benchmark_relaxation.py` runs at small E on CPU (CI), all three backends, producing the metrics table without error; GPU/peak-memory path guarded by CUDA availability.
- **Packaging:** `import track_gen` works without the symlink; integration test runs from repo root with no stdlib `types` shadow; full-suite regression gate.

## 13. Out of scope (YAGNI)

- Banded/hash-grid neighbor-list optimization of the `O(N²)` term (env-chunking ships; the algorithmic drop of `N²` is future work).
- Untangling self-intersecting inits (gated out at generation instead).
- Per-env width/`rad`/`edgy` sampling, open tracks, banking/3D, USD/Isaac asset wiring.

---

## Appendix A — Bake-off data (2026-06-17)

Identical harness (`docs/superpowers/spikes/2026-06-17-relaxation-bakeoff/common.py`), 64 simple Bézier centerlines, `scale=1.0`, `N=256`, `half_width=0.03`. Validity = `thickness ≥ 0.98·half_width` AND zero border/centerline crossings. Baseline (raw inflation) = **2/64 valid**, 173 border crossings over 59 tracks.

| Solver | Valid | Shape change (disp med) | Clearance CV ↓ | Converge | GPU scale | Guarantee |
|---|---|---|---|---|---|---|
| **xpbd** | **64/64** | **0.0023** | 0.040 | ~100–150 it, 36s CPU | hash-grid O(N·k) | **hard** |
| energy (Adam) | 62/64 | 0.0049 | — | ~800 it, 52s | dense O(N²) | soft (2 hairpins cross) |
| tp_sobolev | 46/64 | 0.067 (area 1.4–5.4×) | 0.062 | med 8 it (stalls) | dense O(N²) | soft (over-rounds) |
| xpbd → tp_sobolev finisher | 64/64 | 0.016 | **0.025** | +1.5s on xpbd | dense finisher | hard |

Figures (`baseline.png`, `after_xpbd.png`, `after_energy.png`, `after_tpsobolev.png`, `after_hybrid.png`, `current_failure.png`) are in the spike dir. Scaling correction: the per-curve Sobolev solve is essentially free (FFT, <1% of step); the `O(N²)` pairwise energy is the real cost — so "thousands of small dense solves" is a non-issue, but the dense pairwise tensor at E=8192 is the memory driver (→ env-chunking).

## Appendix B — Reference: Repulsive Curves

Yu, Schumacher, Crane, "Repulsive Curves," SIGGRAPH 2021 — tangent-point energy + fractional-Sobolev preconditioning (the continuum, mesh-independent cousin of bead repulsion; the preconditioner de-stiffens the flow to converge in tens of iterations). Classic repulsive energies: Möbius energy (O'Hara), tangent-point energy. The `tp_sobolev` backend and finisher (§4.4–4.5) are a pragmatic batched approximation of this method.
