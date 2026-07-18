# 2.5D Tracks — Design

**Date:** 2026-07-18
**Status:** Approved
**Modules:** `track_gen._src.types`, `_src.warp_zprofile`, `_src.warp_pipeline`, `_src.checkpoints`, `_src.props`, `_src.localize` (docs only), `_src.course`, new `_src.heightfield`

## Goal

Bring the Z profiler to road tracks: the 2D pipeline lays out the track as
today, a per-centerline-point elevation stage lifts the band into 2.5D
(level cross-sections), distances become true 3D, and a baked per-env
**heightfield** lets an external physics solver collide against the
non-flat ground. Explicitly NOT included: the corridor/tube SDF (reserved
for full-3D) — out-of-bounds collision stays the existing plan-view
band/SDF machinery, which remains exactly valid because cross-sections are
level (the band's XY projection is unchanged by the lift).

## 1. Config and the elevation stage

- **`TrackGenConfig`** gains the same eight knobs as `GateGenConfig`:
  `z_profile` (`"flat"` default | `"uniform"` | `"random_walk"` |
  `"noise"`), `z_base`, `z_min`, `z_max`, `z_max_step` (grade: max |dz|
  per unit plan-view arc length), `z_noise_amplitude`,
  `z_noise_harmonics`, `z_valid_grade` (0 disables). Validation moves to a
  shared helper used by both configs so the two field sets cannot drift.
- **`warp_zprofile` generalization**: the profile kernels take
  `(cum, perim)` directly instead of deriving chords from `pos2`. The gate
  path keeps computing `cum`/`perim` from anchors as today; the track path
  passes the resampler's 2D `arclen` scratch and 2D `length` — no new
  scratch, no behavior change for gates (goldens and existing zprofile
  tests unaffected).
- **Elevation stage** at the pipeline's existing lift boundary
  (`_lift_track_k` in `warp_pipeline`), inside the captured graph,
  allocation-free:
  - `z_profile == "flat"`: byte-identical legacy path — z = `z_base`
    (default 0) on all three polylines, `tangent`/`arclen`/`length`
    written from the 2D scratch verbatim. The frozen goldens (which store
    `arclen`) stay exact.
  - non-flat: evaluate z(s) per centerline index from the profiler;
    write the SAME z to `center[i]`, `outer[i]`, `inner[i]` (level
    cross-sections — the road is a lifted ribbon, no banking); recompute
    `tangent` as normalized 3D central differences (`normal` stays the
    planar left-normal); recompute `arclen`/`length` as TRUE 3D
    cumulative distances.
  - The profile-selection branch is config-static (capture-stable).
- **Validity**: `_validity_k` gains the wraparound grade check (same form
  as the gate kernel's: flag when `|dz| > z_valid_grade * max(dxy, 1e-9)`
  on any consecutive centerline pair including the closing segment),
  gated on `z_valid_grade > 0`. All existing 2D checks run unchanged on
  the plan view, before the lift.
- Closure guarantees carry over unchanged: random_walk subtracts its
  Brownian-bridge drift over the loop; noise is periodic in s/perimeter.

## 2. Consumer semantics

- **Distances are true 3D everywhere** (user decision): `arclen`,
  `length`, localizer `s`, and the checkpoint/prop spacing scans all
  measure 3D distance on lifted tracks. The scan kernels
  (`props._scan_boundary_k` path and its checkpoint use) switch from
  xy-projected to 3D segment length — one line each; flat tracks are
  numerically identical (z = 0 contributes nothing to the length sum).
- **Collision untouched**: segments/SDF/discs keep computing in the
  plane. Correct by construction with level cross-sections.
- **Checkpoints**: cross-sections lift automatically (`left`/`right`
  interpolate the lifted `inner`/`outer` at the same index); crossing
  planes use the (now mildly pitched) 3D segment tangent with
  `up_half = _BIG` (vertically unbounded) as today.
- **Localization**: unchanged code; on lifted tracks `n_up` is the height
  above the road surface at the foot point (roll-free frame).
- **`curvature()` / `speed_profile()`**: now measure 3D bending on lifted
  tracks (vertical crests reduce the speed limit — physically right).
  Docstring notes added; flat tracks unchanged.
- **Props**: `PropSet.position` becomes `wp.vec3f`, sampled from the
  lifted boundary polylines so cones/walls sit ON the road. `tangent`
  (vec2f), `yaw`, and `length` stay planar — pose roll/pitch remains the
  consumer's business. Breaking change, pre-1.0 accepted; docs and
  reshape examples updated.
- **`Course`**: track-mode wiring unchanged except the optional
  heightfield below.

## 3. Heightfield baker

New `track_gen/_src/heightfield.py`, mirroring the SDF baker's shape and
lifecycle:

- `HeightFieldBaker(track, resolution, padding=None)` — per-env square
  grid, flat `[E * res * res] float32` heights, plus per-env `origin`
  (`[E] vec2f`) and `cell_size` (`[E] float32`) derived from the
  plan-view band AABB + padding (AUTO padding rule mirrors the SDF
  baker's 10%). `bake()` is allocation-free, capture-safe (`_sync`
  idiom), and re-baked on regeneration.
- **Bake semantics (nearest-edge continuation)**: each pixel takes the
  road z of the NEAREST centerline cross-section by plan-view distance —
  on-road pixels get the road surface, off-road pixels get the nearest
  edge's z continued outward. The surface is continuous (no cliffs at
  road edges); the road sits flush in its terrain. Flat tracks bake a
  constant `z_base` sheet. Invalid envs bake NaN.
- **`CourseConfig.heightfield_resolution: int | None = None`**
  (track-mode-only; validation in the existing strict style; >= 8 like
  `sdf_resolution`): when set, `Course` constructs the baker in
  `_build_subtools` (under the capture lock, per the current lock
  contract) and re-bakes in `_refresh`; exposed as `course.heightfield`.
- The consumer binds the grid to their solver's heightfield primitive;
  the docs show the `[E, res, res]` reshape and the world mapping
  (`world_xy = origin[e] + (i, j) * cell_size[e]`, z = grid value).

## Non-goals

- Corridor/tube SDF and any 3D out-of-bounds volume — full-3D stage.
- Banking / rolled cross-sections; per-point roll storage.
- Terrain independent of the track (the heightfield is derived from the
  road, not a landscape generator).
- Prop pose roll/pitch (positions gain z; orientation stays planar).

## Testing

- **Golden anchor**: flat default → all track fields byte-identical to
  the frozen goldens (the legacy-path guarantee).
- Profiler-on-tracks: bounds, closure (|z_first − z_last| small), grade
  cap, determinism per seed, padding NaN.
- 3D distances: sum of 3D chords == `length` (tolerance); scan spacing
  matches 3D arc distance; checkpoint/prop z equals the road z at their
  arc positions.
- Validity: grade check flags a steep closing segment (isolating
  geometry, as in the gates test) and is disabled at 0.
- Heightfield: pixel under a centerline point == that point's z;
  off-road pixel == nearest cross-section's z; continuity across the
  road edge; flat track bakes constant sheet; invalid env bakes NaN;
  capture safety (bake inside the refresh graph on cuda).
- Full suite + goldens + `sphinx -W` green throughout.

## Docs

- New "2.5D tracks" section/page beside `gates-3d.rst`: profiles on
  tracks, level cross-sections, 3D distance semantics, the
  curvature/speed_profile note, `PropSet` z, heightfield usage with a
  rendered figure (plan + elevation + heightfield panel from a new viz
  helper or an extension of `plot_tracks.py`).
- Changelog/migration note: `PropSet.position` vec3f; new config knobs.
