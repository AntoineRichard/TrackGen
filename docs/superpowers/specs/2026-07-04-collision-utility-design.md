# Collision / Out-of-Bounds Utility — Design

**Date:** 2026-07-04
**Status:** Approved
**Module:** `track_gen.collision`

## Goal

A Warp-native, GPU-accelerated utility that determines whether oriented boxes have
left the drivable band of a generated track (out-of-bounds detection), returning full
contact information per box. This is the first entry in a family of query utilities;
it establishes the public-namespace pattern for the ones that follow.

## Requirements

- **Batched**: operates on a `Track` batch (E envs), with **many boxes per env**
  (`max_boxes = B`, fixed stride, flat `[E * B]` layout like the rest of the codebase).
- **Oriented boxes of varying sizes**: each box is (center `vec2f`, yaw `float32`,
  half-extents `vec2f`). Per-box sizes are independent.
- **Full contact info** per box: OOB flag, signed clearance, nearest boundary point,
  boundary normal, and which boundary (inner/outer) is nearest.
- **OOB rule**: a box is out of bounds as soon as **any part of it** overlaps the inner
  hole or exits the outer boundary ("any part crosses").
- **Warp-friendly / GPU-accelerated is a hard requirement**: pure Warp kernels, no host
  sync inside `query()`, CUDA-graph capturable, fixed shapes.
- Two backends from day one: exact **segments** scan (default) and baked **sdf** grid.

## Public API

```python
import track_gen
from track_gen.collision import CollisionChecker, BoxContact

track = generator.generate()
checker = CollisionChecker(track, max_boxes=8, method="segments")  # or "sdf"

contact = checker.query(position, yaw, half_extents)
# position:     wp.array [E*B] vec2f  — box centers
# yaw:          wp.array [E*B] float32 — box orientations (radians)
# half_extents: wp.array [E*B] vec2f  — per-box half sizes
```

### `CollisionChecker`

Constructor: `CollisionChecker(track, max_boxes, method="segments", sdf_resolution=128,
sdf_padding=None)`.

- Binds to a `Track` instance. Because `TrackGenerator.generate()` overwrites the same
  `Track` buffers in place, the **segments** backend automatically reads fresh track
  data after each regeneration — no rebind step.
- The **sdf** backend requires an explicit `checker.bake()` after each `generate()`.
  `bake()` is itself a Warp kernel launch (graph-capturable). The initial bake runs at
  construction when `method="sdf"`.
- Construction validates: `max_boxes >= 1`, `method in {"segments", "sdf"}`,
  `sdf_resolution >= 8`, `sdf_padding` (if given) `> 0`. Per-env grid bounds are the
  track AABB expanded by `sdf_padding` on every side; `sdf_padding=None` defaults to
  10% of the AABB's larger extent. Callers whose boxes may stray further than the
  padding should pass an explicit value; sdf query samples outside the grid clamp to
  the edge texels (they stay negative there, so far-out boxes still read OOB).
- `query()` performs cheap host-side shape/dtype validation on its inputs, launches
  the backend kernel, and returns the checker's preallocated `BoxContact`.

### `BoxContact`

Dataclass of preallocated Warp arrays, all flat `[E * max_boxes]`:

| field      | type    | meaning                                                        |
|------------|---------|----------------------------------------------------------------|
| `oob`      | int32   | 1 if the box crosses a boundary or lies outside the band       |
| `distance` | float32 | signed clearance: + = margin to nearest boundary, − = penetration |
| `nearest`  | vec2f   | nearest point on the boundary polylines                        |
| `normal`   | vec2f   | boundary normal at `nearest`, pointing **into** the drivable band |
| `boundary` | int32   | which boundary is nearest: 0 = inner, 1 = outer                |

Same in-place contract as `Track` / `GateSequence`: `query()` returns the **same**
`BoxContact` instance every call and overwrites its buffers; `clone()` returns a
fully-owned deep copy. This keeps `query()` allocation-free and graph-capturable.

### Inactive slots and invalid envs

- A NaN box position marks an inactive slot: outputs are NaN and `oob = 0`.
- Results are **undefined** for envs with `valid[e] == 0` (callers gate on `valid`,
  as everywhere else in the library).

## Semantics

- **Drivable band** = inside the outer loop ∧ outside the inner loop, using real
  points `0 .. count[e]-1` as closed polygons (NaN-padded tail excluded).
- **OOB flag (exact)**: any box corner outside the band **∨** any boundary segment
  intersects the box. For a convex box this is exact: if the outside region touches
  the box interior without any corner being outside, the boundary must cross the box.
- **distance**:
  - Not OOB: minimum OBB↔segment distance over all boundary segments (exact, > 0).
  - OOB: −(deepest corner penetration), i.e. minus the max distance from any outside
    corner to the boundary. In the rare "thin peninsula pokes through a box edge while
    all four corners remain inside" case this evaluates to 0⁻ (flag still correct);
    documented approximation.
- **nearest / normal / boundary**: from the argmin segment. The normal is the segment
  perpendicular, sign-oriented toward the index-aligned centerline point
  (`Track.center[i]`), so it always points into the band.

## Backends

### `segments` (default — exact)

One thread per box (`E*B` threads). Each thread loops over its env's boundary
segments (inner + outer, ≤ 2·`count[e]` ≤ 2·`N_max`):

- crossing-number test per box corner → inside/outside band classification;
- OBB↔segment distance (segment transformed into box frame) with argmin tracking
  → clearance, nearest point, normal, boundary id;
- segment↔box overlap test → exact "crosses" detection.

Cost O(E·B·N) per query — at E=4096, B=8, N≤768 that is ~25M segment tests, trivial
on GPU and consistent with the codebase's dense-scan idiom (XPBD does O(N²)).
No precompute, no memory overhead, no staleness.

### `sdf` (baked grid — O(1) queries, approximate near boundaries)

- **Bake** (thread per texel, per env): brute-force signed distance to the band over a
  per-env padded-AABB grid `[R, R]` (`R = sdf_resolution`), sign positive inside the
  band. A parallel int8 grid stores the nearest-boundary id (0 inner / 1 outer).
  Bake cost O(E·R²·N) — GPU-oriented; CPU bakes are slow and only for small tests.
- **Query** (thread per box): bilinear sample of φ at the 4 corners + center;
  `oob = min corner φ < 0`; `distance = min corner φ`; `normal` from the central-
  difference gradient at the argmin sample; `nearest = p − φ·∇φ`; `boundary` from the
  id grid.
- Error is bounded by grid resolution near boundaries; sub-cell peninsula features can
  be missed. Memory ≈ `E · R² · 5` bytes (~330 MB at E=4096, R=128) — the trade-off is
  documented so users pick the backend deliberately.

## File layout

- `track_gen/_src/collision.py` — `BoxContact`, `CollisionChecker`, segment kernels.
- `track_gen/_src/collision_sdf.py` — SDF bake + query kernels.
- `track_gen/collision.py` — public shim: module docstring + re-export of
  `CollisionChecker`, `BoxContact`.
- `track_gen/__init__.py` — add `from . import collision`, extend `__all__`.
- Future utilities become sibling public modules (`track_gen.raycast`, …), keeping the
  top-level namespace flat and discoverable without a `utils` grab-bag.

## Error handling

- Constructor raises `ValueError` on invalid `method`, `max_boxes < 1`,
  `sdf_resolution < 8`, or non-positive `sdf_padding`.
- `query()` raises `ValueError` on shape/dtype mismatches (host-side check on array
  metadata only — no device sync).
- No silent fallbacks: an sdf checker never silently degrades to segments.

## Testing

- **Analytic**: hand-built annulus `Track` (concentric circles) where every output has
  a closed-form answer — boxes fully inside, crossing outer, inside the hole,
  straddling inner, rotated 45°, per-box mixed sizes.
- **Oracle**: property tests comparing against a small numpy reference
  (point-in-polygon + segment distances) on real generated tracks, `device="cpu"`,
  small E.
- **Backend agreement**: sdf results within grid-resolution tolerance of segments on
  the same inputs.
- **Contract**: NaN-slot behavior, per-env `count` variation, in-place buffer reuse +
  `clone()`, validation errors.
- **CUDA**: graph-capture smoke test for `query()` and `bake()` (`cuda` marker).
- Tests run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (repo convention).

## Out of scope (YAGNI)

- Continuous/swept collision between steps.
- Non-box shapes (circles, capsules, polygons) — future utilities can share the
  namespace pattern.
- Box↔box collision; gates-vs-box checks.
- Automatic staleness detection for the sdf bake.
