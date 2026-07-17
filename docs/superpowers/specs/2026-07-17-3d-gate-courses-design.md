# 3D Gate Courses (2.5D Lift) — Design

**Date:** 2026-07-17
**Status:** Approved
**Modules:** `track_gen._src.types`, `_src.gate_generator`, `_src.progress`, `_src.checkpoints`, `_src.localize`, `_src.course`, new `_src.z_profile`, new `_src.gate_collision`

## Goal

Let users generate batched **3D gate sequences for drones** while keeping the
proven 2D pipeline as the layout engine. The 2D generators produce the
plan-view layout exactly as today; a new post-stage assigns per-gate
altitudes (a pluggable **Z profiler**), fits a periodic 3D spline through the
gate anchors, and derives full 3D gate poses from the fit. Runtime gains full
parity in 3D: gate-pass detection, progress/rewards, localization, and
gate-frame collision. "2.5D in practice" — and deliberately staged so a later
native-3D pipeline (3D XPBD over gate chains, the opening described in
`docs/related-work/prior-art.rst`) slots in behind the same data model.

## Architecture decision (Approach A — unified vec3f)

One public data model: `GateSequence` and `Track` move to `wp.vec3f` in a
**single migration**, with z = 0 for the 2D path. Pre-1.0 breaking is
accepted. Internals stay `vec2f`: no generator, resampler, XPBD, or inflation
kernel is rewritten — a lift kernel writes into the vec3f public arrays at
the pipeline boundary. Runtime consumers are written once against the vec3f
types; 2D is the z = 0 special case.

Rejected alternatives: a parallel `GateSequence3D` module (permanent
duplication of progress/localize logic for risk-avoidance we don't need
pre-1.0) and native 3D generation now (abandons the six proven 2D generators
as the layout source and needs parallel-transport frames from day one; it is
the v2 destination this design lines up with, not the first step).

## Data model

All arrays stay flat, fixed-shape, NaN-padded, CUDA-graph capturable.

**`GateSequence`** (`types.py`):
- `position`, `tangent`, `left`, `right`: `vec2f` → `wp.vec3f`, flat
  `[E * max_gates]`.
- New `orientation` `[E * max_gates] wp.quatf` — full gate pose. Yaw-only
  quaternions in 2D mode, so the field is never garbage.
- New `half_size` `[E * max_gates] float32` — square opening half-extent.
  Constant from config in v1; per-gate so it can vary later.
- `left`/`right` remain the post positions, now **derived** from
  `orientation` + `half_size`.
- `normal` (2D perpendicular) is **removed** — superseded by `orientation`.

**`Track`** (`types.py`): `outer`/`center`/`inner`/`tangent`/`normal` become
`wp.vec3f` (z = 0 from the 2D pipeline) in the same migration, so consumers
change reshapes `(…, 2) → (…, 3)` exactly once and the later tube stage is
purely additive. `winding`, `arclen`, `valid`/`count` semantics unchanged
(computed in plan view internally). In gate-course mode, `center`/`tangent`
carry the resampled **3D** course and `outer`/`inner` are NaN (no band).

Memory cost: +50% on geometry arrays — negligible next to SDF/RNG buffers.

## Generation stage (the 2.5D lift)

Appended after the existing pipeline; all batched Warp kernels, no host sync,
fed by the existing per-env seed streams.

1. **2D pipeline runs unchanged** → validated layout + gate anchors in the
   plane, exactly as today (including the full 2D validity pass).
2. **Z profiler** — first-class and pluggable, the vertical counterpart of
   the XY generator (`z_profile` enum + params, mirroring generator
   selection):
   - `flat` (default): z ≡ `z_base`. Exact current behavior.
   - `uniform`: i.i.d. per gate in `[z_min, z_max]`.
   - `random_walk`: bounded steps clamped to `[z_min, z_max]`; cumulative
     drift subtracted (Brownian-bridge style) so the closed loop closes.
     `z_max_step` is a **grade** (max |Δz| per unit of plan-view arc-length
     between consecutive gates), so it scales with gate spacing and doubles
     as the difficulty knob.
   - `noise`: a few random Fourier harmonics in arc-length — periodic by
     construction.
3. **Periodic 3D spline fit** — closed Catmull-Rom through the 3D gate
   anchors, resampled at constant 3D arc-length into the standard `Track`
   arrays (`center`, `tangent`, `arclen`, `length` become true 3D
   quantities). Downstream runtime consumes `Track` as always.
4. **Gate pose kernel** — tangent at each gate's spline parameter →
   quaternion, per `gate_align` mode:
   - `yaw_only`: forward = horizontal projection of the tangent; gate stays
     upright.
   - `full_tangent`: forward = tangent; right = `normalize(up_world ×
     forward)`, up = `forward × right` (roll always zero). Epsilon fallback
     to the neighboring gate's yaw for near-vertical tangents (counted into a
     debug stat, never raises).
   - Posts `left`/`right` derived from pose + `half_size`.

## Runtime consumers

- **Gate passing / progress** (`progress.py`, `checkpoints.py`): a pass is a
  step segment (prev → current position) crossing the gate plane in the
  forward direction with the intersection inside the opening
  (`|u| ≤ half_size`, `|v| ≤ half_size` in the gate frame). `dist_to_next`
  and rewards become 3D Euclidean. Ordering, laps, reset unchanged.
- **Localization** (`localize.py`): windowed nearest-point search on the 3D
  centerline, returning `s` plus a 2D lateral offset `(n_right, n_up)` in
  the roll-free frame at the foot point. The 2D path yields `n_up = z`
  trivially — one implementation serves both.
- **Gate-frame collision** (new `gate_collision.py`; replaces the band/SDF
  for gate courses): each square gate is 4 thin oriented boxes (2 posts +
  top/bottom bars; thickness and depth from config). The drone binds as a
  sphere (`position` vec3f + `radius`); collision when min sphere-vs-box
  distance < 0. Tested only against a fixed-size index window around the
  drone's current checkpoint — capture-safe and cheap.
- **`Course` facade**: constructor gains `mode: "track" | "gate_course"`.
  Gate-course mode wires progress + localize + gate collision and skips the
  band SDF entirely. `bind()` takes `orientation` (quaternion) instead of
  `yaw`; 2D consumers pass a yaw-only quaternion.
- **2D track collision stays fully functional.** In `track` mode the
  band/SDF/box/disc collision path is wired exactly as today. Its kernels
  keep computing in the plane: they read the xy components of the vec3f
  `Track` buffers (or internal vec2f scratch copies where cheaper) and the
  baked 2D SDF is unchanged. The only contract change is `bind()` accepting
  a quaternion, from which the 2D path extracts yaw. Behavior parity is
  gated by regression tests (see Testing).

## Config surface

Additions to `TrackGenConfig` (flat-dataclass style, eagerly validated —
`z_min ≤ z_max`, positive sizes, etc.):

- `z_profile`: `"flat"` (default) | `"uniform"` | `"random_walk"` | `"noise"`
- `z_base`, `z_min`, `z_max`, `z_max_step`, `z_noise_amplitude`,
  `z_noise_harmonics`
- `gate_align`: `"yaw_only"` | `"full_tangent"`
- `gate_half_size`, `gate_frame_thickness`, `gate_frame_depth`

`mode` lives on `Course`, not the config — it selects which consumers get
wired, not how geometry is generated.

## Validity and error handling

**XY is generated and validated completely independently of Z.** The Z
profiler runs after the 2D pipeline has finished, including its validity
pass, so the existing self-intersection, turning-number, and thickness checks
run unchanged on the same 2D centerline they check today.

Consequences, accepted for v1:

- **Conservative by design**: a plan-view crossing is rejected even when Z
  separation would make it a good drone course (figure-eight overpass). V1
  only flies courses whose layout was already a valid 2D track.
- **The 3D spline refit may drift slightly off the checked centerline in XY**
  (it interpolates gate anchors, not every centerline point). Harmless for a
  gate course — there is no road band to self-intersect; gate proximity is
  covered by the new checks below.

New gate-course checks, same `valid`/`count` + reseed convention as today:

- **Max segment slope** between consecutive gates (catches steep `uniform`
  draws; threshold configurable).
- **Min 3D gate spacing.**

No host-side branching under graph capture; degenerate configs rejected at
construction; near-vertical pose fallbacks counted, never raised.

## Testing

Pytest, batched over E envs with mixed validity, CPU + CUDA:

- **Regression anchor**: with `z_profile="flat"`, the vec3f pipeline
  produces z ≡ 0 and XY **bit-identical to pre-migration goldens** — the
  proof the 2D behavior survived the dtype break.
- **Z profilers**: bounds, loop closure (|z_first − z_last| ≈ 0), slope cap,
  per-seed determinism.
- **Spline + poses**: interpolates anchors, periodic; `yaw_only` upright;
  `full_tangent` matches tangent; near-vertical fallback engages, no NaNs.
- **Runtime**: pass through opening = pass; outside opening or backward
  crossing = no pass; localize round-trip (synthesize a point at known
  `(s, n_right, n_up)`, recover it); sphere hits posts/bars but flies clean
  through the opening.
- **Capture safety**: extend `tests/test_generate_concurrent_cuda.py` to
  gate-course mode (preserves the `_CAPTURE_LOCK` regression coverage).
- **2D collision regression**: the entire existing collision suite
  (`collision.py`, `collision_sdf.py`, `collision_geom.py`,
  `collision_discs.py`) must pass unmodified in behavior after the vec3f
  migration — same OOB verdicts on the same seeds as pre-migration goldens.

## Viz and docs

- Small gate-course path in `viz/`: plan view + elevation profile side by
  side, gates drawn as oriented squares. No full 3D renderer in v1.
- New "3D gate courses" docs page beside the gates tutorial; sweep every
  `(…, 2)` reshape in code/tests/docs to `(…, 3)`; changelog migration note
  (`normal` removed, `orientation`/`half_size` added, `bind()` takes a
  quaternion).
- One branch, mechanical consumer sweep; no interim history rewrites.

## Non-goals / future work

- **Corridor tube collision** around the 3D centerline (the road-track OOB
  analogue) — next stage after gates.
- **Plan-crossing courses**: allow XY crossings when Z clearance at the
  crossing exceeds a threshold. Changes what "valid layout" means; belongs
  with the native-3D work.
- **3D course self-clearance** (no two non-adjacent segments within a 3D
  clearance): the XPBD pairwise-separation machinery ported to vec3f, if a
  stronger guarantee is ever needed.
- **Native 3D generation + 3D XPBD** over gate chains (curvature, clearance,
  slope/bank, visibility constraints) — the v2 this design's data model is
  shaped for.
- Banking/roll on gates, per-gate variable `half_size` sources, terrain
  awareness.
