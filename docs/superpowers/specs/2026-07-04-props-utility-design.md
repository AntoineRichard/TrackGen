# Boundary Prop Sampling Utility — Design

**Date:** 2026-07-04
**Status:** Approved
**Module:** `track_gen.props`

## Goal

A Warp-native, GPU-batched utility that resamples track boundaries at a
user-chosen spacing and emits instancing poses for rendering-only props —
cone lines along a boundary, wall pieces tiling it. Complementary to
`track_gen.collision` (these props are NOT colliders); third sibling in the
public utility-namespace pattern (`track_gen.collision`, `track_gen.props`, …).

## Requirements

- **Batched**: operates on a `Track` batch (E envs), flat `[E * max_props]`
  output layout, NaN-padded past `count[e]`, matching library conventions.
- **User-set spacing** in world units, snapped per env so the closed ring has
  no seam gap or doubled prop.
- **Two modes**: `"points"` (pose per sample — cones, poles, markers) and
  `"segments"` (pose per span between consecutive samples — walls).
- **Curves**: the actual boundary polylines, `boundary="outer"` or `"inner"`.
  No lateral offset, no centerline placement (YAGNI — decided).
- **2D pose format**: `position` vec2f, unit `tangent` vec2f, `yaw` float32;
  `length` float32 per prop. Lifting to 3D transforms is the consumer's job.
- **Warp-first**: pure Warp kernels; `sample()` is allocation-free and
  host-sync-free under graph capture (module `_CAPTURING` flag, `_sync`
  helper — same pattern as `collision.py`). warp-lang >= 1.14.

## Public API

```python
from track_gen.props import PropSampler, PropSet

cones = PropSampler(track, spacing=0.08, boundary="outer", mode="points")
walls = PropSampler(track, spacing=0.15, boundary="inner", mode="segments")

props = cones.sample()   # PropSet, flat [E * max_props] wp.arrays
```

### `PropSampler`

Constructor:
`PropSampler(track, spacing, boundary="outer", mode="points", max_props=None)`.

- Binds a `Track` instance. Because `TrackGenerator.generate()` overwrites the
  same buffers in place, the sampler reads the CURRENT batch on every
  `sample()` — no rebind step (same aliasing contract as `CollisionChecker`
  with `method="segments"`).
- One sampler = one curve + one mode + one spacing. Fixed shapes make
  `sample()` CUDA-graph capturable. Both boundaries → construct two samplers.
- `max_props` (output slots per env): explicit int, or `None` to derive from
  the batch bound at construction: `ceil(1.5 * max_perimeter / spacing)`,
  where `max_perimeter` is the largest boundary perimeter over the batch's
  valid envs (a host-side one-time readback at construction — never in
  `sample()`). If a later regeneration produces a boundary needing more
  props, the ring is truncated and flagged (`truncated[e] = 1`).
- Validation (`ValueError`): `spacing > 0`, `boundary in {"inner", "outer"}`,
  `mode in {"points", "segments"}`, `max_props >= 3` when explicit, track
  batch layout sane (E >= 1, flat stride divisible). If `max_props=None` and
  the bound batch has no valid env to derive from, raise `ValueError` asking
  for an explicit `max_props`.

### `PropSet`

Dataclass of preallocated Warp arrays; `sample()` returns the SAME instance
every call and overwrites it in place (`clone()` for snapshots — the
`Track`/`GateSequence`/`BoxContact` contract).

| field       | shape             | type    | meaning                                            |
|-------------|-------------------|---------|----------------------------------------------------|
| `position`  | `[E * max_props]` | vec2f   | prop pose position (on-curve point, or chord midpoint in segments mode) |
| `tangent`   | `[E * max_props]` | vec2f   | unit direction (curve tangent at sample, or chord direction) |
| `yaw`       | `[E * max_props]` | float32 | `atan2(tangent.y, tangent.x)`                      |
| `length`    | `[E * max_props]` | float32 | points: effective arc step; segments: chord length |
| `count`     | `[E]`             | int32   | real prop count per env; slots `>= count[e]` NaN   |
| `truncated` | `[E]`             | int32   | 1 if `max_props` clipped this env's ring           |
| `step`      | `[E]`             | float32 | effective arc spacing `perimeter / count`          |

## Semantics

- **Perimeter** is the boundary polyline's own closed-loop length over real
  points `0 .. count[e]-1` (not the centerline `arclen`).
- **Snap spacing**: `n[e] = clamp(round(perimeter_e / spacing), 3, max_props)`;
  effective step `perimeter_e / n[e]`. Effective spacing deviates from the
  request by at most ~half a step; the ring closes exactly.
- **points mode**: sample `k` sits at arc position `k * step` along the
  polyline. `position` on the curve; `tangent` = unit direction of the
  polyline segment containing the sample; `length` = step.
- **segments mode**: prop `k` spans on-curve samples `k → k+1 (mod n)`.
  `position` = chord midpoint, `tangent`/`yaw` = chord direction, `length` =
  chord length — instancing a unit-length wall scaled by `length` tiles the
  boundary with only corner cracks on curves (documented; reduce spacing for
  tighter walls). All `n` spans are emitted (closed ring).
- **Degenerate/invalid envs** (`count[e] < 3` real boundary points):
  `count = 0`, `truncated = 0`, `step = NaN`, all per-prop fields NaN.
  Results for envs with `valid[e] == 0` are undefined (callers gate on
  `valid`, as everywhere in the library).
- Winding/orientation: tangents follow the boundary point order (same
  winding as the generator emits); consumers wanting outward-facing props
  rotate by ±90° downstream.

## Kernels

`track_gen/_src/props.py`, two kernels, same scan+lookup idiom as the
pipeline resampler but self-contained:

1. **Scan** (thread per env): walk the boundary polyline once; write
   cumulative arc length into a preallocated `[E * N_max]` float32 scratch;
   emit `perimeter`, `count`, `step`, `truncated`.
2. **Place** (thread per `(e, slot)`): slots `>= count[e]` write NaN and
   return. Otherwise compute the slot's target arc position(s)
   (`k*step`, and `(k+1)*step mod perimeter` for segments mode), binary-search
   the cumulative table, lerp position(s) on the polyline, emit
   position/tangent/yaw/length.

Shared helper (`_safe_normalize2`) comes from
`collision_geom.py`. Scratch and outputs preallocated in `__init__`;
`sample()` is two launches + `_sync`.

## File layout

- `track_gen/_src/props.py` — kernels + `PropSampler` + `PropSet`.
- `track_gen/props.py` — public shim re-exporting the two names.
- `track_gen/__init__.py` — `from . import props`, extend `__all__`
  (+ update the curated-surface test `tests/test_public_api.py`).
- `docs/reference/api.rst` — "Boundary props" section beside the collision
  section.

## Testing

- **Analytic (annulus fixture reuse)**: circle radius r → `n =
  round(perimeter/spacing)` with polygon perimeter, positions on the circle
  (|p| = r within polyline tolerance), uniform angular gaps, points-mode
  tangents ⟂ radius, segments-mode chord length `≈ 2 r sin(π/n)`, ring
  closure (slot n−1 connects to slot 0).
- **Oracle**: small numpy reference (cumulative arc + interp) compared on
  real generated tracks (cpu, valid envs).
- **Contract**: constructor validation errors, `max_props=None` derivation,
  truncation flag with tiny explicit `max_props`, in-place reuse + `clone()`,
  buffer aliasing after `wp.copy` into the bound Track, per-env count
  variation, degenerate env → count 0 + NaN.
- **CUDA**: graph capture test for `sample()` with poisoned-buffer replay
  (assert the replay recomputes, not stale buffers), cuda-marked.
- Suite runs with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (repo convention).

## Out of scope (YAGNI)

- Lateral offsets / centerline placement.
- 3D poses/quaternions (consumer lifts 2D → 3D).
- Jitter/randomized placement, per-prop variation.
- Density by count instead of spacing (derive: `spacing = perimeter/n`).
- Collision for props — explicitly rendering-only, per the request.
