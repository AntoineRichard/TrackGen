# Full Code Review — track_gen

**Date:** 2026-06-27
**Method:** 8 reviewers fanned out across the codebase, one report per subsystem (~10.5k lines reviewed).
**Scope:** `track_gen/_src/*`, `track_gen/__init__.py`, `viz/param_explorer.py`, `viz/plot_tracks.py`.

## Executive summary

The codebase is well-engineered: pure-Warp, zero-alloc, CUDA-graph-capture-safe kernels with rigorous NaN-padding/per-env-count discipline and unusually thorough docstrings. One **Critical** bug was found (an RNG 3D-kernel stride error that breaks sample independence). The dominant cross-cutting theme is a **validation asymmetry**: `GateGenConfig` validates its inputs thoroughly, but `TrackGenConfig` — which feeds the same samplers — omits the same guards, leaving reachable `ZeroDivisionError` / out-of-bounds-GPU-write / inverted-range paths from the public API.

| Severity | Count |
|---|---|
| Critical | 1 |
| Important | 10 |
| Suggestion | 19 |

### Highest-priority items (cross-cutting)
1. **[Critical] RNG 3D kernel stride bug** — `rng_kernels.py:109` (and 10 sibling kernels). Wrong intra-block linearization breaks independence; actively triggered by the Fourier path.
2. **[Important ×3, convergent] `TrackGenConfig` validation gap** — flagged independently by the core-types, pipeline, and point-generator reviewers: `min_point_distance` (divisor → `ZeroDivisionError`/int32 overflow), `num_points ≤ N_max` (→ **out-of-bounds GPU writes / memory corruption**), `num_points_per_segment ≥ 2`, `min_num_points`/`max_num_points` ordering, `half_width`/`spacing`/`num_envs > 0`. The fix is a single hardening pass on `TrackGenConfig.__post_init__` mirroring the guards `GateGenConfig` already has.
3. **[Important] RNG correctness cluster** — quaternion seed reuse (correlated rotations), state double-buffer corruption on first partial-`ids` sample, `UnboundLocalError` on integer bounds.
4. **[Important] `TrackGenerator` deterministic-batch footgun** — repeated `generate()` returns identical tracks while docstrings imply fresh randomness.
5. **[Important] Explorer thickness stat** — a single `count==0` env NaN-poisons the half-width median, corrupting `mean_thickness` for the whole batch exactly on the low-yield configs the explorer exists to diagnose.

---

## Section 1 — Core types, registries & public API
`types.py`, `generator_registry.py`, `gate_generator_registry.py`, `__init__.py`

Import-leaf core: config + result dataclasses with documented in-place-aliasing and `clone()` escape hatches, two lazy registries, eager public re-exports. Design is sound; main weakness is config validation asymmetry.

- **[Important] `types.py:219-271`** — `TrackGenConfig.__post_init__` omits the point-family sampler guards `GateGenConfig` has. `min_point_distance` is a divisor (`warp_generate.py:559`); `0.0` → opaque `ZeroDivisionError`, negative → negative cell count. `min_num_points`/`max_num_points` feed `wp.randi` and can be inverted. **Fix:** raise on `min_point_distance <= 0`, `min_num_points < 2`, `max_num_points < min_num_points`.
- **[Important] `types.py:140-141,270-271`** — `half_width`, auto-derived `spacing`, and `num_envs` unvalidated; non-positive `half_width` silently yields non-positive `spacing`. **Fix:** require `half_width > 0`, resolved `spacing > 0`, `num_envs >= 1`.
- **[Suggestion] `types.py:110-112`** — style `*_range` tuples unvalidated (inverted/short tuples flow into RNG). Validate length-2, `lo <= hi`.
- **[Suggestion] `types.py:79-85,145`** — documented checkpoint-steering bounds and `relax_solver` enum unenforced while siblings are validated.
- **[Suggestion] `generator_registry.py:41-43`** — `register` silently overwrites duplicates; the gate registry rejects cross-module collisions. Unify.
- **[Suggestion] `gate_generator_registry.py:62-65`** — `_ensure_loaded` silently skips modules lacking `register_specs`; make it loud.

**Strengths:** documented aliasing contract + torch-free `clone()`; frozen registry specs; cycle-free lazy loading; gate registry cross-module dup protection; thorough `GateGenConfig` validation with rationale comments; small explicit public surface.

## Section 2 — Track orchestration & relaxation
`track_generator.py`, `warp_relax.py`

Tidy fixed-batch facade + correct XPBD double-buffering and capture sequencing. Issues are a misleading determinism contract and dead diagnostics, not crashes.

- **[Important] `track_generator.py:153-155`** — `generate()` re-copies fixed seeds every call; repeated calls return the **identical** batch, while docstrings imply variation. **Fix:** advance seeds per call, or make the deterministic contract explicit.
- **[Suggestion] `warp_relax.py:168-169`** — `sep_cache_overflow` is incremented but never read; undersized cache silently drops collision candidates. Surface it or remove it.
- **[Suggestion] `warp_relax.py:339-340 vs 385-386`** — inconsistent host-sync: `xpbd_solve_inplace` syncs on CPU where `band_l0_inplace` correctly doesn't. Gate on `"cuda" in dev`.
- **[Suggestion] `track_generator.py:140`** — `isinstance(int)` rejects numpy ints, accepts `bool`. Use `numbers.Integral` excluding `bool`.
- **[Suggestion] `track_generator.py:164-174`** — capture forces `_CAPTURING=False` in `finally` instead of save/restore (gate code does it right).
- **[Suggestion] `warp_relax.py:27-28`** — separation band has no upper bound; tiny tracks silently disable all separation.

**Strengths:** zero-alloc seed refresh issued outside capture (subtle CUDA-graph detail done right); correct warmup→sync→capture sequencing; correct ping-pong parity incl. `relax_iters==0`; centralized separation-band `@wp.func`; consistent degenerate-input epsilon floors; accurate docstrings.

## Section 3 — Warp pipeline
`warp_pipeline.py`

Pure-Warp staging (generate→resample→relax→inflate) with rigorous NaN-padding and float64 arc-length scans. Main gap is a missing structural invariant.

- **[Important] `warp_pipeline.py:1463`** — shared scan scratch `cs_seg`/`cs_s` sized to `N_max`, but resample is fed the `num_points`-strided generation buffer. If `num_points > N_max`, writes index **past the scratch → OOB GPU writes / memory corruption**. No validation relates `num_points` to `N_max` (defaults 256<384 happen to be safe). **Fix:** assert `num_points <= N_max` in `__post_init__`, or size scratch to `max(num_points, N_max)`.
- **[Suggestion] `warp_pipeline.py:1088`** — N_max truncation warning fires spuriously when count legitimately equals `n_max`. Record a real overflow flag.
- **[Suggestion] `warp_pipeline.py:1517`** — `count=None` path infers `E` via integer division with no exactness check; mismatched length → silently wrong layout. Assert divisibility.
- **[Suggestion] `warp_pipeline.py:88`** — outer/inner assignment via order-nondeterministic atomic-float area sums can flip labels for near-equal-area degenerate tracks (which validity usually rejects). Comment or deterministic reduction.

**Strengths:** consistent NaN-padding producer/consumer contract; rigorous per-env count handling with float64 accumulators; numerically defended degenerate inputs; clean capture-safety/zero-alloc; serial & parallel self-intersection paths share one predicate.

## Section 4 — RNG
`rng_kernels.py`, `rng_utils.py`

Per-env-seeded RNG over Warp. Seed-derivation design is good, but several real defects in the kernels.

- **[Critical] `rng_kernels.py:109`** — all 3D kernels linearize the `(j,k)` block with the wrong stride `j*shape[1]+k` (should be `j*shape[2]+k`). With unequal trailing dims this overlaps/collides state → correlated or duplicated draws. **Actively triggered** by `_experimental/fourier.py:72-73` (shape `(K,2)`). Affects ~11 kernels. **Fix:** use `shape[2]`; add a `(3,5)`-shape independence test.
- **[Important] `rng_kernels.py:1441`** — quaternion kernels reuse the same seed for axis and angle → strongly correlated rotations (and axis-angle with uniform angle isn't uniform over SO(3)). Read state into a local and advance it.
- **[Important] `rng_utils.py:32`** — `_new_states` zero-initialized and never mirrored from `_states`; a first **partial-`ids`** sample does a whole-array `wp.copy(states, new_states)`, zeroing untouched envs (destroying seed diversity). **Fix:** `wp.copy(new_states, states)` after init; ideally copy back only touched ids.
- **[Important] `rng_kernels.py:303`** — `uniform()`/`normal()` raise `UnboundLocalError` when both bounds are plain Python ints (`integer()` handles it). Accept ints or raise a clear `TypeError`; also fix mismatched error strings.
- **[Suggestion] `rng_utils.py:25`** — no validation of `seeds.shape == num_envs`, `low<=high`, `std>=0`, `lam>0`; int64→int32 seed cast can wrap.
- **[Suggestion] `rng_kernels.py:132`** — `shape[0]==1` collapse is too broad (mishandles `(1,5)`: 1 sample drawn but offset advances by 5).
- **[Suggestion] `rng_utils.py:58`** — `get_offset` is dead code.

**Strengths:** deliberate `seed + arange` per-env derivation with documented rationale; construction auto-init; consistent dispatcher structure with race-free read-then-copy state update; justified poisson int cast; exactly-balanced sign sampling.

## Section 5 — Point & curve generators (bezier/hull/polar)
`warp_generate.py`, `warp_generate_hull.py`, `warp_generate_polar.py`

Consistent, well-documented sample→sort→assemble→resample→fallback pattern. Weaknesses are missing input bounds and uneven self-intersection fallback.

- **[Important] `warp_generate.py:559`** — `num_cells = int(1.0/(min_point_distance*2))`: for `min_point_distance > 0.5` → `num_cells==0` → integer mod/divide-by-zero in the grid kernels (GPU UB); for `< ~1.1e-5` → `nc2` int32 overflow → garbage indices. Reachable via public config. **Fix:** validate/clamp so `num_cells >= 1` and `nc2 < 2**31`.
- **[Important] `warp_generate_hull.py:237`** — `num_points_per_segment` unvalidated; hull `_catmull_rom_k` divides by `npseg-1` (→ div-by-zero at `npseg==1`). Bezier asserts `>=2`; hull/polar don't. **Fix:** validate `num_points_per_segment >= 2` in config (single source of truth).
- **[Suggestion] `warp_generate_polar.py:178`** — polar has no self-intersection detection/fallback (bezier & hull do) yet marks every env valid; lowers downstream yield. Add fallback or document reliance on the inflate gate.
- **[Suggestion] `warp_generate_hull.py:250`** — hull's straight-chord fallback isn't guaranteed simple (inward-displaced midpoints) and isn't re-tested; can still self-cross.
- **[Suggestion] `warp_generate.py:425`** — `_corner_angles_gate_k` wraps neighbours mod `P` not mod `cnt`, so closing-seam corners skip the angle gate (consumed by gates module/tests).

**Strengths:** decorrelated per-stream RNG salts with rationale; consistent NaN-pruning with correct mod-`cnt` seam closure; float64 centroid accumulation; provably-safe insertion-sort scratch usage; explicitly-reasoned safe aliasing; zero per-call alloc; polar's seam-free `[0,1)` sampling.

## Section 6 — Voronoi & checkpoint generators
`warp_generate_voronoi.py`, `warp_generate_checkpoint.py`

Pure-Warp, alloc-free, capture-safe; sound math; graceful degenerate handling. No Critical bug.

- **[Suggestion] `warp_generate_checkpoint.py:576`** — `checkpoint_clip_fallback` read independently at alloc vs generate time; flag flip → launches with `None` buffers → opaque error. Add an assertion tying them.
- **[Suggestion] `warp_generate_voronoi.py:70-103`** — layout mode `"ring"` actually falls through to uniform box fill; only `"void_ring"` samples an annulus. Rename or document.
- **[Suggestion] `types.py:76`** — `checkpoint_angle_jitter` has no upper bound; `>= ~1.0` scrambles ring order with no validation (siblings are validated).
- **[Suggestion] `warp_generate_voronoi.py:239` & `checkpoint.py:276`** — `_normalize_centerline_k` duplicated verbatim in 3 generators; hoist to `warp_pipeline`.
- **[Suggestion] `warp_generate_voronoi.py:78-80`** — dead `if cluster > 5` guard.
- **[Suggestion] `warp_generate_voronoi.py:155-170`** — at `S==K` anchor snap can self-cross while still flagged valid (relies on inflate gate).

**Strengths:** strong capture discipline; robust degenerate handling (unit-circle reconstruction, extent epsilon); deterministic best-of-K with documented tie-break and decorrelated salts; mathematically careful heading-ramp closure (turning number 1); provably-safe clip bounds; exceptional docstrings.

## Section 7 — Gate subsystem
`warp_gate.py`, `gate_generator.py`, `warp_generate_{gates,polar_gates,voronoi_gates,checkpoint_gates}.py`, `gate_generator_registry.py`

Clean, well-factored: shared pipeline in `warp_gate.py`, thin registry plugins per generator. Verified OOB-safe count arithmetic, no aliasing/races. No Critical bug; all gate tests pass.

- **[Suggestion] `gate_generator.py:38`** — feasibility check validates only the generator's *max* count vs `min_gates`; point generators can draw `< min_gates` (e.g. `min_gates=11, min_num_points=9`) → envs silently `valid=0`. Validate the *minimum* producible count or document `min_gates` as a validity floor.
- **[Suggestion] `gate_generator.py:100`** — no construction-time check that `rng` env count matches `config.num_envs`; mismatch surfaces later as a `wp.copy` size error.
- **[Suggestion] `gate_generator.py:107`** — process-wide `_CAPTURING` globals mutated during capture; concurrent generators would race (capture isn't concurrency-safe anyway). Document or thread the state.
- **[Suggestion] `warp_gate.py:434`** — gate-width collision uses strict proper-intersection only; collinear/endpoint-touching wide bars aren't flagged. Document or add collinear-overlap test.
- **[Suggestion] `warp_gate.py:347`** — `_finalize_frame_k` and `_finalize_validity_k` clamp `cnt` differently (defensive-only today). Clamp uniformly.

**Strengths:** clean separation of concerns; careful graph-capture correctness (zero-alloc, 3-pass warmup, both `_CAPTURING` flags); consistent NaN-padding + finiteness validation; defensive count clamping; no aliasing/races; float64 centroid; strong `GateGenConfig` validation + actionable errors + reload-idempotent registry.

## Section 8 — Visualization & param explorer UI
`viz/param_explorer.py`, `viz/plot_tracks.py`

Gradio explorer + headless PNG renderer over the real pipeline. UI-alive error handling and control/output sync guards are well done.

- **[Important] `param_explorer.py:294`** — `hw = ...median()` over **all** envs; a single `count==0` env NaN-pads index 0, and `torch.median` returns NaN if any element is NaN → `band` collapses to 1 → wrong `mean_thickness` for the whole batch, exactly on low-yield configs. **Fix:** restrict to `valid` envs (non-empty here); fix the stale comment.
- **[Suggestion] `param_explorer.py:296`** — `band` computed on the torch stream then read by a Warp kernel with no pre-launch sync (CUDA stream mismatch). Sync before launch or compute band in Warp.
- **[Suggestion] `param_explorer.py:659`** — `_collect` lacks the length guard `_collect_gate` has; `zip` silently truncates on drift. Add the guard / build-time assert.
- **[Suggestion] `param_explorer.py:239`** — `render_page` re-derives torch tensors per grid cell; gate renderer converts once. Convert once.
- **[Suggestion] `plot_tracks.py:161`** — `scale` goes ≤0 when `track_width_m >= box_m` (CLI misuse) with no warning. Validate.

**Strengths:** well-designed UI-alive error handling (ValueError vs unexpected); build-time control/output drift guards incl. the new `TRACK_MODE_SECTION_SIZES` single-source-of-truth; performance-aware `_gate_stats` (batched cdist vs per-env sync loop); correct device syncs at the heavy boundary; cleanly factored visibility; robust NaN-aware plotting.

---

## Recommended action order
1. **RNG 3D stride bug** (Critical) — fix the stride + add an unequal-dim independence test. Also the three RNG Important bugs (quaternion, partial-ids state copy, int-bounds dispatch).
2. **`TrackGenConfig`/pipeline hardening** (Important, convergent) — one pass adding: `num_points <= N_max`, `min_point_distance` bounds, `num_points_per_segment >= 2`, `min/max_num_points` ordering, `half_width/spacing/num_envs > 0`. Closes the OOB-write and ZeroDivisionError paths at once.
3. **`TrackGenerator` determinism** + **explorer thickness median** (Important) — small, high-value fixes.
4. Suggestions as cleanup (registry symmetry, dead code, layout-name clarity, duplicated normalize kernel, sync consistency).
