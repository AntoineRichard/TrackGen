# Batched Track Generation with Inflation — Design

**Date:** 2026-06-16
**Status:** Approved (pre-implementation)
**Scope:** Extend the GPU-batched closed-curve track generator into a fast, fully-batched pipeline that emits, per track, three aligned closed polylines — outer boundary, centerline, inner boundary — via variable-width track inflation, with self-intersection and border-collision handled rather than hoped-for.

---

## 1. Goal & context

`track_generator.py` is a GPU-batched closed-Bézier track generator (the `ccw_sort` / `get_bezier_curve` lineage), tensorized to spawn thousands of tracks in parallel for Isaac Lab RL environments. `rng_utils.py` (`PerEnvSeededRNG`) and `rng_kernels.py` provide per-env reproducible sampling via Warp kernels, with a numpy fallback for unique-integer sampling.

Today the code produces only a **centerline** and only weakly targets self-intersection (a vertex-angle rejection test on the control polygon). The two missing halves of the stated goal are:

1. **Inflation** — materialize left/right borders from the centerline. Absent entirely today.
2. **Robust validity** — the borders must not cross. There are *two* distinct failure modes the current code does not address:
   - **Local curvature:** where the centerline radius of curvature drops below the half-width, the inner border folds (a cusp).
   - **Global self-approach:** where two *distant* parts of the track pass within `2·half_width`, borders overlap even though curvature is low at both.

This design adds inflation, fixes the existing correctness/robustness issues, removes the numpy host-sync bottleneck, and introduces a second (Fourier) generator — all behind a clean modular split.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Generators | **Both** — Bézier (default) + truncated-Fourier — behind one shared `CenterlineGenerator` interface. |
| Width model | **Variable width**, clamped by curvature (`w = min(w_max, α/κ)`). No curvature-based rejection. |
| Global self-approach | **Optional** (config flag) additional clamp of width by self-distance. Off → cheap; on → inflation always valid by construction, zero rejection. |
| Output format | Fixed-`N`, arc-length-uniform, **index-aligned cross-sections** (`outer[i]/center[i]/inner[i]` share a normal), `[E, N, 2]` per boundary. |
| Variable point count | **Both, configurable.** Default = fixed-`N` dense boundaries. Optional constant-arc-length-spacing mode → variable real length per track, padded to `N_max` with `-1`/NaN + per-track `count`. Variable *corner* counts upstream are NaN-padded (also fixes the prune-collapse bug). |
| Extra per-track data | Per-point tangent + normal; arc-length (cumulative + total); per-track validity mask. (Half-width is **not** stored — recoverable as `‖outer − center‖`.) |
| Code structure | **Clean modular split** (see §3). |

## 3. Architecture

Five files. `rng_kernels.py` and `rng_utils.py` are untouched.

```
track_gen/
├── rng_kernels.py        # unchanged (Warp RNG)
├── rng_utils.py          # unchanged (PerEnvSeededRNG)
├── geometry.py           # pure batched-torch primitives, device-agnostic, CPU-testable
├── generators.py         # CenterlineGenerator protocol + Bezier + Fourier
├── inflation.py          # Centerline -> Track (resample, width, offset, assemble)
└── track_generator.py    # TrackGenConfig, Track, TrackGenerator facade (+ compat shim)
```

**Design rationale:** the existing single class already does too much; adding two generators, an inflation stage, and metadata to it would compound that. The pure-function `geometry.py` core is device-agnostic and unit-testable on CPU without GPU or Warp, which is where most of the test value lives.

### 3.1 Intermediate & result types

`Centerline` — the only thing every generator returns and the only thing inflation consumes:
- `points: [E, M_max, 2]` — closed, ordered, dense samples; shorter tracks NaN-padded to `M_max`.
- `valid: [E] bool` — generation-time validity (e.g. Bézier regen gave up for this env).

`Track` — the final result dataclass:
- `outer, center, inner: [E, N, 2]` — index-aligned cross-sections.
- `tangent, normal: [E, N, 2]` — unit tangent & unit left-normal along the centerline.
- `arclen: [E, N]` cumulative; `length: [E]` total.
- `valid: [E] bool`; `count: [E] int` (`== N` in fixed mode, `≤ N_max` in constant-spacing mode; trailing slots `-1`/NaN).
- Half-width is **not** stored; recover as `‖outer − center‖`.

### 3.2 `TrackGenConfig` (single dataclass, passed everywhere)

- `generator ∈ {"bezier", "fourier"}`; `device`; `num_envs`.
- **Bézier:** `min_num_points`, `max_num_points`, `num_points_per_segment`, `min_point_distance`, `min_angle`, `rad`, `edgy`, `scale` (each of `rad`/`edgy`/half-width optionally a per-env sampling range for diversity).
- **Fourier:** `num_harmonics K`, decay exponent `p` (default 2 → `1/k²`), `amplitude`, `scale`.
- **Width:** `half_width` (scalar or per-env range), curvature safety factor `α` (default ~0.9), `clamp_self_distance: bool`, `self_distance_margin`, `self_distance_band`, `self_distance_decimation` (~64–96).
- **Output:** `num_points N`, `output_mode ∈ {"fixed", "constant_spacing"}`, `spacing`, `N_max`.
- **Robustness:** `max_regen_iters`, `turning_tol`.

## 4. Data flow

```
PerEnvSeededRNG ──► CenterlineGenerator.generate(ids) ──► Centerline[E,M_max,2] + valid
                                                              │
                                                              ▼
                                  inflate(centerline, config) ──► Track
```

## 5. Generation stage (`generators.py`)

Both generators implement `CenterlineGenerator.generate(ids) -> Centerline`; inflation never knows which ran.

### 5.1 Bézier (`BezierCenterlineGenerator`) — current pipeline, repaired

1. **On-GPU cell sampling** (removes the numpy host-sync). `u = rng.sample_uniform_torch(0, 1, (num_cells², ), ids)`; `cell_idxs = u.topk(max_num_points).indices`. The index set of the *k* largest of *n* i.i.d. continuous uniforms is exactly a uniform *k*-subset without replacement — distributionally identical to the old `rng.choice(..., replace=False)`, but fully device-resident and per-env seeded. Cells → grid `x, y` + uniform noise → scale.
2. **`ccw_sort`** around the centroid → a provably simple control polygon (each edge lives in a disjoint angular wedge, so non-adjacent edges cannot cross).
3. **Variable corners, NaN-padded.** Pruned corners become a NaN sentinel — *not* collapsed onto the first point — so no zero-length segments reach resampling.
4. **Vector-space tangents** replace the angle blend, the two "broken roll" patches, and the `+π` wraparound hack: `t = safe_normalize(p·u_out + (1−p)·u_in)`, where `u_in/u_out` are unit edge directions and `p = atan(edgy)/π + 0.5`. No atan2 wraparound; no cusp-inducing wrong-way tangents.
5. **Cubic Bézier** per consecutive pair: handles of length `rad·chord` along those tangents, `num_points_per_segment` samples each, concatenated into a closed dense centerline.
6. **Bounded iterative regen** (replaces unbounded recursion + per-level `.clone()`): a `while` loop regenerates only the envs failing the clamped-`arccos` `min_angle` test **and** the O(N) turning-number gate (`|Σ signed turn| ≈ 2π`), up to `max_regen_iters`; survivors that still fail are marked `valid=False`.

### 5.2 Fourier (`FourierCenterlineGenerator`) — guaranteed-smooth alternative

- Sample harmonic coefficients `a_k, b_k ∈ ℝ² ~ N(0, (amp / k^p)²)`, `k = 1..K`, via `rng.sample_normal_torch` (per-env seeded). Decay `p = 2` bounds curvature by construction; low `K` makes self-intersection rare-to-impossible.
- Evaluate `c(t) = c0 + Σ_k a_k cos(kt) + b_k sin(kt)` on a dense `t ∈ [0, 2π)` grid → closed, C∞ centerline. Mean-center and scale the bounding box to `scale`.
- `M` fixed, no padding, `valid` all-true; the same turning-number gate runs as a cheap safety net for the rare low-`K` self-crossing.

Both return `Centerline(points[E, M_max, 2], valid[E])`.

## 6. Inflation stage (`inflation.py`)

`inflate(centerline, config) -> Track`. Six batched steps:

1. **Masked arc-length resample.** Segment lengths between consecutive *real* (non-NaN) points, including the closing wrap segment → cumulative `s`, total `L` per env. Targets: `linspace(0, L, N, endpoint=False)` (fixed mode) or `arange(0, L, spacing)` padded to `N_max` (constant-spacing mode). Batched `searchsorted(s, targets)` + lerp between bracketing real points → arc-length-uniform centerline. Removes Bézier-`t` corner bunching that makes naive offsets noisy.
2. **Tangents & normals.** Central difference on the resampled closed loop → `safe_normalize` → unit `T`; left-normal `Nrm = [−T_y, T_x]`. Orthonormal by construction.
3. **Curvature `κ`.** Finite-difference curvature on triples of the now-uniform points (uniform spacing → numerically stable). `κ ≥ 0`; straights → `κ ≈ 0`.
4. **Width.**
   - **Curvature clamp:** `w_curv = min(w_max, α/κ)`, computed as `where(κ > eps, α/κ, w_max)`. The inward offset is cusp-free iff `w·κ < 1`; `α < 1` (default ~0.9) is the safety fraction, so `w·κ ≤ α < 1` always — the inner border never folds at a corner.
   - **Self-distance clamp (optional, `clamp_self_distance`):** `d_i =` nearest distance from point `i` to any non-adjacent point (decimate to ~64–96 points, `cdist`, mask a `±band` index window — accounting for closed-loop wraparound — then interpolate `d` back to `N`). `w_self = ½·(d_i − margin)`. Two borders collide when each offsets ≥ `d/2`; this prevents it.
   - `w = min(w_max, α/κ, [w_self])`, floored at 0. Per-point `w: [E, N]`.
5. **Offset.** `outer = center + w·Nrm`; `inner = center − w·Nrm`. Aligned cross-sections.
6. **Assemble + validity.** `valid = gen_valid ∧ (|turning| ≈ 2π) ∧ (w > w_floor everywhere) ∧ no-NaN`. A track that pinches to ~0 width is flagged, not crashed. Fill `arclen`, `length`, `count`; build `Track`.

**Output modes** drop out of step 1: **fixed** → dense `N`, `count = N`; **constant_spacing** → real length varies, padded to `N_max` with `-1`/NaN + `count`.

## 7. Facade (`track_generator.py`)

- `TrackGenConfig` and `Track` dataclasses.
- `TrackGenerator(config, rng)` with `generate(num_or_ids) -> Track` — instantiates the configured generator, runs it, runs `inflate`, returns `Track`.
- A thin backward-compatibility shim mapping the old `generate_tracks(num_tracks)` entry point onto the new pipeline (centerline-only return) for any existing caller.

## 8. Error handling & robustness

- Clamp `arccos` input to `[−1+eps, 1−eps]` — fixes the current `NaN > min_angle == False` silent-regeneration trap.
- `safe_normalize` (eps floor) wherever a direction is normalized.
- `κ → 0` handled by `where(κ > eps, α/κ, w_max)` — no divide-by-zero.
- Bounded `while` regen with `max_regen_iters`, then flag remainder `valid=False` — no recursion-depth blowup, no hang on a bad parameter combo.
- Self-distance band mask accounts for closed-loop wraparound (neighbors excluded on both sides).
- Invalid/degenerate tracks → best-effort geometry + `valid=False`; never raise in the hot path.
- Per-env determinism preserved (all randomness via `PerEnvSeededRNG`). **Intentional break:** on-GPU top-k sampling yields different *values* than the old numpy `choice` path for a given seed — still per-env reproducible, just not bit-identical to the old code.

## 9. Testing strategy (TDD)

Most of the suite runs anywhere — no GPU or Warp required.

- **`geometry.py` unit tests, pure torch on CPU:** resample spacing uniform within tol + NaN-padding handled; `‖T‖ = 1` and `T·Nrm = 0`; curvature of a radius-`r` circle = `1/r`, straight line ≈ 0; turning number of a convex polygon = `2π`, figure-eight ≈ 0; `nearest_nonadjacent_distance` + band mask on a known config; `ccw_sort` orders a scramble.
- **Inflation invariants, CPU synthetic centerlines (circle / ellipse / near-touch):** offset sign & magnitude correct; `w ≤ w_max`; `w·κ < 1` everywhere (no fold); self-distance clamp yields no border overlap on a deliberate near-touch; `valid` flags a deliberately self-crossing input.
- **Generator integration tests, guarded by Warp availability (Warp CPU device if no GPU):** same seed → same track; env-`i` independent of env-`j`; variable-corner padding shapes; both generators emit valid closed loops.
- **Dev-only matplotlib smoke script** (not a unit test) to eyeball a grid of tracks + borders.

## 10. Out of scope (YAGNI)

- Constant-width tracks / curvature-based rejection (explicitly rejected in favor of variable width).
- Non-closed (open) tracks.
- Banking, elevation, or any 3D track geometry — this is 2D centerline + 2D borders.
- Collision meshes / USD export / Isaac Lab asset wiring — the pipeline emits tensors; downstream consumes them.

## Appendix — priority bundle from the source analysis (for implementation ordering)

1. On-GPU top-k sampling (kills numpy host-sync) — enables "fast batched."
2. Vector-space tangent blending — deletes atan2 hacks, prerequisite for clean offsets.
3. Arc-length resample — prerequisite for fixed-`N` aligned output.
4. Curvature-clamped variable width (+ optional self-distance clamp) — the inflation core.
5. Turning-number gate + bounded regen — cheap simplicity guarantee without hangs.
6. Fourier generator — guaranteed-smooth alternative behind the shared interface.
7. Correctness fixes (clamp arccos, NaN-sentinel padding, per-env `rad`/`edgy`/width sampling).
