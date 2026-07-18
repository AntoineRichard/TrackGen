# Knot-Based Track Elevation — Design

**Date:** 2026-07-18
**Status:** Approved
**Modules:** `track_gen._src.types`, `_src.warp_zprofile`, `_src.warp_pipeline`, `docs/`, `viz/`

## Problem

The 2.5D track elevation stage (merged 2026-07-18, commits `c79f186..7fb6668`)
runs the Z profiler over the **resampled centerline**, so it makes one
independent altitude decision per point. On a default track that is ~94 points
at ~6 cm spacing, which turns profiles designed for ~12 sparse gate anchors into
high-frequency noise. Measured on one valid env (94 points, 1.57 m extent):

| profile | relief | direction changes |
|---|---|---|
| `uniform` | 25% of extent | 72 / 92 |
| `random_walk` | 7% of extent | 52 / 92 |
| `noise` | 18% of extent | 4 / 92 |

`uniform` is per-point vertical jitter; `random_walk` is a washboard. Only
`noise` produces a drivable surface, because its harmonics band-limit it. Worse,
the coupling is backwards: raising the resample resolution makes the road
*bumpier*, when it should only make the same road smoother-sampled.

## Goal

Decide altitude at a small, configurable number of **control knots** and
interpolate smoothly between them — the way gate courses already work, where the
~12 gates *are* the knots. Hill count becomes independent of resample density
and of the generator's layout.

## 1. The knot stage

**Config.** `TrackGenConfig` gains `z_control_points: int = 10`, validated
`>= 3` (a closed loop needs three knots to be non-degenerate). It applies to
`uniform` and `random_walk`; it is inert for `flat` and `noise`, and the
docstring says so. `GateGenConfig` does NOT gain it — gates are already knots.

**Knot sampling reuses the existing profiler.** Knots sit at uniform arc
fractions, so knot `k` has cumulative arc `k * perim / K` — an analytic table.
Because Task 1 generalized `apply_z_profile` to take `(count, stride, cum,
perim)`, knot sampling is that same function called with `stride = K` and a
knot-sized buffer. No new sampling kernels: `uniform` draws K i.i.d. heights,
`random_walk` runs its Brownian-bridge walk over K steps with `z_max_step`
applied per knot interval (`ds = perim / K`, so the cap now bounds a real
stretch of road rather than a 6 cm hop). Both remain clamped to
`[z_min, z_max]` at the knots.

**Interpolation: periodic monotone cubic.** One new kernel evaluates a cubic
Hermite with Fritsch–Carlson-limited tangents at each resampled point's arc
fraction `arclen[i] / perim`, writing the per-point `z` buffer the lift already
consumes. Knot spacing is uniform, so the limiter simplifies: with secants
`d_k = (z_{k+1} - z_k) / h`, the tangent at knot `k` is `0` when
`d_{k-1} * d_k <= 0` (a local extremum — this is what kills overshoot), else the
average of the neighbouring secants magnitude-limited to
`3 * min(|d_{k-1}|, |d_k|)`. Wraparound indexing makes it periodic, so the loop
closes with matching value and slope.

**Guarantees, stated honestly:**

- **Bounds are exact.** Monotone limiting keeps the interpolant inside the
  interval between adjacent knot values, so per-point `z` never leaves
  `[z_min, z_max]`. No post-interpolation clamp is needed, and none is added.
- **Grade is bounded but not by `z_max_step`.** A PCHIP segment's slope can
  reach up to **3× its knot-to-knot secant** in the worst case. So
  `z_max_step` shapes the walk *at the knots*, and `z_valid_grade` — which
  already runs on the final per-point data — remains the mechanism that gates
  realized steepness. Both are documented as such; neither is over-claimed.

## 2. Wiring and the XPBD invariant

Dispatch inside `inflate_warp`'s existing non-flat branch:

- `flat` → unchanged legacy path (byte-identical; goldens gate it).
- `noise` → analytic per-point exactly as today (`stride = n_max`, cum =
  `out.arclen`). Sampling it at knots then re-splining would only add error to
  the one profile that already behaves.
- `uniform` / `random_walk` → knot cum table → `apply_z_profile` (stride `K`)
  → monotone eval → per-point `z`.

Stage position is unchanged: still between the 2D validity inputs and the lift,
strictly downstream of relaxation. New scratch is one `[E * K] float32` knot
buffer plus its `[E * K]` cum table and `[E]` count, all allocated in
`_inflate_warp_alloc` — the hot path stays allocation-free, sync-free, and
config-static, as required for graph capture.

**Invariant made explicit:** *XPBD solves in 2D; elevation is applied strictly
after relaxation.* This is already true structurally (`warp_relax.py` contains
no `vec3f`; relaxation completes in `_run_pipeline` before `inflate_warp` runs),
and this design adds a regression test that pins it: same seed, flat config vs
hilly config, assert the XY components of `center`, `outer`, and `inner` are
bit-identical. A future refactor cannot silently reorder the stages.

The consequence of that invariant is unchanged and remains a documented v1
limitation: separation is enforced in plan view, so two track sections passing
close in XY are pushed apart regardless of their heights — overpasses and
figure-eights stay impossible until the solver becomes z-aware (the full-3D
stage).

## 3. Testing, docs, figures

**Tests** (cpu + cuda, batched over E with mixed validity):

- **Resample-density independence** — the defect as a regression gate:
  generate the same layout at two point densities differing by ~2× and assert
  (a) the elevation's direction-change count stays `<= z_control_points` in
  both, and (b) the two counts differ by no more than 2 — i.e. the count
  tracks `K`, not the point count. Under today's per-point profiling the
  denser run would roughly double it.
- **No overshoot** — `max(z)` and `min(z)` over real points equal the max and
  min of the sampled knot values within 1e-5, and `[z_min, z_max]` holds.
- **Closure** — the loop closes smoothly: `|z[0] - z[last]|` is at most one
  knot interval's worth of change, `z_max_step * perim / K` (plus 1e-5), so
  the closing segment carries no accumulated seam.
- **Determinism** per seed; padding slots stay NaN per the `Track` contract;
  `z_control_points` validation (`< 3` raises).
- **XPBD invariance** — the flat-vs-hilly bit-identical-XY test above.
- **Unaffected paths** — goldens exact, gate suites unchanged, `noise` output
  unchanged from its current values.

**Docs and figures:**

- `docs/tracks-25d.rst`: add the `z_control_points` row and state which
  profiles it governs; replace the neutral profile descriptions with actual
  guidance (what each profile is *for* on a road); document the bounds
  guarantee and the 3×-secant grade caveat.
- Re-shoot `docs/_static/tracks-25d.png` with a profile that now looks like a
  road.
- Add the missing **profile-comparison figure**: one row per profile, same
  seed, so the four options are visually distinguishable — the gap that
  prompted this work.

## Non-goals

- No change to gates (already knot-based), to `flat`, or to `noise` values.
- No banking, no tube SDF, no z-aware XPBD — the full-3D stage owns those.
- No deliberate high-frequency "road roughness" layer. If wanted later, that is
  a separate, designed feature layered on top of the smooth profile — not the
  accidental noise this design removes.
- The heightfield's off-road medial-axis discontinuities (measured: a 0.246 m
  jump between adjacent texels against 0.275 m of relief) are a real, separate
  finding and are NOT addressed here; they need their own decision (document
  the limitation, or blend across the nearest few segments).
