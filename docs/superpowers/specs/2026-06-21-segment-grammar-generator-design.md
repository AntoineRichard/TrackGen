# Segment-Grammar First-Stage Generator (#6) — Design Spec

**Date:** 2026-06-21
**Status:** Historical design (pre-implementation; catalog references below reflect 2026-06-21, not the current runtime registry)
**Depends on:** the pluggable generator framework (registry + `GeneratorSpec`, `docs/generator-contract.md`), the existing generators (`warp_generate.py` bezier, `warp_generate_hull.py`, `warp_generate_polar.py`), and the shape-variety gate (`tests/test_shape_variety.py` + `benchmarks/compare_generators.py` `compactness_p*`/`mean_chicanes`/`straight_frac`, `benchmarks/track_metrics.py` `chicane_count`/`straight_fraction`).
**Strategy input:** the deferred-methods investigation (its numpy prototype validated the curvature-integrator + closure approach and exposed the residual/self-intersection tradeoff). The doc section is "### 6. Segment Grammar / Road-Block Generator" in `docs/pre-relaxation-generator-methods.md`.

## Goal

Add a fourth first-stage generator, `"grammar"`, that builds a closed centerline from an explicit **racing-segment vocabulary** (straights, sweepers, hairpins, chicanes, S-bends, kinks, clothoid/spiral transitions). It is the first generator that produces *counted, deliberate* racing features — true sustained straights, localized hairpins, and chicanes — which bezier/hull/polar **structurally cannot** express (all three are star-shaped position/radius parameterizations). It must pass the shape-variety gate and add net-new character validated by renders, not just yield.

## Background / current state

- At time of writing, the catalog (`generator_registry.available()`) was `["bezier", "hull", "polar"]`; the current runtime registry also includes `checkpoint` and `voronoi`. The original three are star-shaped and cannot hold a true straight, a localized hairpin, or a real chicane — curvature is only an emergent side effect.
- The framework: each generator is a module exposing `alloc_scratch(config)` + `generate(seeds_wp, config, out_centerline, out_valid_wp, scratch)`, registered as a `GeneratorSpec` (one line in `generator_registry._ensure_loaded`). `warp_generate_polar.py` is the closest template (closed-by-construction, `valid=1`, a two-pass-over-N kernel: build raw points, then center + isotropic bbox-rescale).
- The shape-variety gate (just landed) is the acceptance check: median post-relax compactness must stay < 0.85, and the harness reports `chicane_count` / `straight_fraction` so feature presence is measurable.
- The investigation's numpy prototype established: heading closure is *linear* (DC-shift κ so `Σκ·ds = 2π`); displacement closure is solved warp-native by a *gap-distribution* pass (no host solve); and **characterful grammars self-intersect ~42% at m=1 with median residual ~1.3× extent**, so the grammar must be budgeted to keep the linear correction mild.

## Non-goals

- Not method #4's smooth Fourier-curvature variant — the prototype showed it is near-circular (compactness 0.99, the polar trap). The character lives in the *named-segment grammar*, which this spec ships.
- Not method #7 (chain-code/discrete) — a related but separate follow-on.
- No host-side iterative/nonlinear closure solve, no data-dependent retry loop (contract violation).
- No template library (the "template + perturbation" option was considered and rejected for this version in favor of budgeted sampling).

## Architecture overview

A **curvature-integrator** generator. Per env, all steps are fixed-bound loops over `S` (segment count) and `N` (samples) → CUDA-graph-capturable, zero per-call allocation:

1. Sample a fixed-length grammar of `S` segments from the vocabulary.
2. Apply the **budget + antisymmetry bias** (residual-taming).
3. Rasterize the segment sequence into a per-sample curvature profile `κ[i]` (N samples).
4. **Heading closure:** DC-shift κ so `Σκ·ds = 2π` (single winding).
5. Integrate: `θ = θ₀ + ds·cumsum(κ)`; `raw = ds·cumsum(cosθ, sinθ)`.
6. **Displacement closure (gap distribution):** measure the net residual `(dx,dy)` (the loop's failure to return to its start) and subtract `(i/N)·(dx,dy)` from point `i` — closes exactly, O(N), no solve.
7. Center + isotropic bbox-rescale to `scale·_BEZIER_EXTENT` (reuse polar's normalization); write N-point centerline + `valid=1`.

Self-intersecting residual cases are NOT prevented by construction; they ride the pipeline's existing polygon fallback + XPBD repair, exactly like bezier.

**Build order: prototype-first, then warp port** (the investigation's recommendation — the budget/antisymmetry params are empirical).

## Component 1 — Segment vocabulary

The fuller vocabulary reduces to **two κ-primitives** plus **named patterns**:

- **Constant-κ segment** `(length_frac, κ, sign)` — covers: straight (`κ≈0`), sweeper (low `|κ|`, long), corner (moderate), hairpin (high `|κ|`, short), kink (high `|κ|`, very short).
- **Linear-ramp-κ segment** `(length_frac, κ_start, κ_end, sign)` — covers clothoid-in (`0→k`), clothoid-out (`k→0`), spiral (`k₁→k₂`).
- **Named patterns** (emitted as fixed short sub-sequences of the primitives): **chicane** / **S-bend** = a pair of opposite-sign segments (optionally with ramp transitions); used so the antisymmetry bias can pair them.

A segment is a fixed-width record; `S` segments per env is a config bound. The vocabulary is data (magnitude/length/sign ranges per type), not branchy code — the rasterizer treats every segment uniformly as a `(κ_start, κ_end)` ramp over its sample span (constant = equal endpoints).

## Component 2 — Grammar sampling + residual-taming (the budget + antisymmetry bias)

Per env, from `wp.rand_init(seed * GRAMMAR_SALT)`:
- Draw `S` segment types (weighted by config), each with length-fraction, magnitude, and sign.
- **Straight quota:** force at least `grammar_straight_frac` of arc-length to be straights (κ≈0) — guarantees real straights appear and reduces net turning concentration.
- **Antisymmetry / pairing bias:** pair opposite-sign features (every strong + feature gets a matching − feature; chicanes/S-bends emitted as ± pairs) so net displacement is near zero *before* correction — this is the primary residual reducer.
- **Curvature-variance budget:** clamp the total signed-curvature magnitude/variance (`grammar_curvature_budget`) and cap hairpin span (`grammar_hairpin_max_frac`) so no single feature dominates the gap.

All clamps are branchless, fixed-shape, deterministic. Exact default values are tuned by the prototype against the residual/character tradeoff.

## Component 3 — Closure (warp-native, no solve)

- **Heading (exact, linear):** after rasterizing κ, subtract its mean and add `2π / (N·ds)` so `Σκ·ds = 2π` — the loop winds exactly once; seam tangent matches by construction. One scalar per env.
- **Displacement (gap distribution):** integrate to a raw open polyline, measure the endpoint residual `(dx,dy)`, then subtract `(i/N)·(dx,dy)` from point `i`. Standard discrete closed-polygon construction — always closes exactly, fixed-bound second pass, no iteration, no host sync. Slightly perturbs geometry where the residual was largest; the budget/antisymmetry bias keeps that perturbation mild, and XPBD owns final repair.

## Component 4 — Integration + normalization

`θ[i] = θ₀ + ds·prefix_sum(κ)[i]`; `t[i] = (cos θ[i], sin θ[i])`; `raw[i] = ds·prefix_sum(t)[i]`; then the gap-distribution close; then center at origin and isotropically rescale the longest bbox dimension to `config.scale·_BEZIER_EXTENT` (the same constant/pass `warp_generate_polar.py` uses, so half_width/spacing/relax see the same coordinate range). Write N points into `out_centerline`, `out_valid_wp = 1`.

## Component 5 — Config surface

New `TrackGenConfig` fields (defaults from the prototype), surfaced in the gradio explorer like polar's knobs:
- `grammar_segments: int` — `S`, the fixed segment count (a graph-capture bound).
- `grammar_straight_frac: float` — minimum arc-length fraction forced to straights.
- `grammar_curvature_budget: float` — total signed-curvature variance clamp.
- `grammar_chicane_bias: float` — strength of the opposite-sign pairing/antisymmetry.
- `grammar_hairpin_max_frac: float` — cap on any single high-κ feature's span.

(Names indicative; finalized in the plan. All `grammar_*`-prefixed so they're clearly this generator's.)

## Component 6 — Torch/numpy prototype (tuning; not shipped)

A `track_gen/_experimental/grammar_proto.py` (torch/numpy allowed there; never imported by the runtime) implementing Components 1–4 on host. Used to tune the budget/antisymmetry defaults against:
- **Shape-variety:** compactness percentiles (median well below 0.85), and `mean_chicanes`/`straight_fraction` clearly above bezier/hull/polar.
- **Closure health:** median residual / bbox-extent small (target well under the ~1.3× the unbudgeted prototype hit), and pre-relax self-intersection rate low enough that the polygon fallback + XPBD recover ≥ the catalog's yields.
- **Renders:** a seed grid where straights/hairpins/chicanes are visibly present and loops are neither collapsed nor kinked — the arbiter.

The prototype is a tuning artifact; it may be kept under `_experimental` or removed after the warp port. No permanent torch oracle is created (the warp generator is validated by the gate, per the framework's precedent).

## Component 7 — Warp port

Port the tuned Components 1–4 to `track_gen/_src/warp_generate_grammar.py`, mirroring `warp_generate_polar.py`:
- `GrammarScratch` (private buffers: per-env segment records `[E*S*...]`, the κ profile `[E*N]`, raw points `[E*N]`, prefix-sum scratch as needed) allocated once in `grammar_alloc_scratch(config)`.
- Warp kernels (one env per row, fixed-bound): a segment-sample kernel and a build-integrate-close-normalize kernel (split as the polar pattern suggests). No host sync, no per-env Python branching, deterministic.
- Register `GeneratorSpec(name="grammar", alloc_scratch=grammar_alloc_scratch, generate=generate_grammar_warp)` + one import line in `generator_registry._ensure_loaded`.

## Acceptance / validation

The generator is accepted only if, over a fixed seed suite:
1. **`tests/test_shape_variety.py` passes** for `"grammar"` (median compactness < 0.85).
2. **Feature presence:** `compare_generators` shows `mean_chicanes` and `straight_frac` for `"grammar"` clearly above bezier/hull/polar (it *makes* these features; they don't merely emerge).
3. **Yield:** post-relax `Track.valid` yield in the same healthy band as the other generators.
4. **Renders:** a seed grid confirms straights/hairpins/chicanes are visible and loops are not collapsed/kinked.
A perfect yield with no feature presence is an automatic reject (the polar lesson).

## Invariants

- Runtime `track_gen/_src/**` stays Warp-native and torch-free (the prototype's torch lives only in `_experimental`/dev).
- Zero per-call allocation (`GrammarScratch` pre-allocated); CUDA-graph-capturable (fixed bounds, no host sync/solve, no per-env Python branching); deterministic in `(seed, config)`.
- Registering `"grammar"` is additive: one new module + one `GeneratorSpec` + one registry import line + the config fields. `track_gen.__all__` unchanged. The full suite + the cuda-graph parity test + the shape-variety gate stay green.
- Commits `--no-gpg-sign`.

## Scope boundary & follow-on

- **This spec:** the `"grammar"` generator (vocabulary + budgeted grammar + κ-integrator + closure), its config fields, the tuning prototype, and the warp port — accepted via the shape-variety gate.
- **Follow-on (separate):** method #7 chain-code/discrete (a discrete cousin sharing the closure family); a richer template-based mode; exposing per-generator params for hull/polar in the explorer. The smooth-Fourier-curvature (#4) variant is explicitly *not* pursued (degenerate).
