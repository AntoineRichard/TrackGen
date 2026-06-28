# README gate documentation + collision-phase imagery — design

**Date:** 2026-06-28
**Status:** Approved (brainstorming), pending spec review

## Problem

The README already has a "Gate sequence generation" section (added at merge `5a3e6aa`),
but it has drifted incomplete relative to the current gate code:

1. No `GateSequence` result table — the `Track` result gets a full field table, gates do
   not. The example only reads `position`/`tangent`/`valid`, omitting `normal`, `left`,
   `right`, `count`, and the same-instance aliasing / `clone()` caveat.
2. `gate_width` and `min_gates`/`max_gates` are undocumented.
3. The "Gates" tab in the Parameter explorer UI is undocumented — that section covers only
   track controls.
4. No gate imagery — the Track section has rendered strips/grids; gates have none. There is
   also no gate-rendering path in `viz/render_readme_assets.py`.

## Goal

Fill these four gaps. Documentation-only **except** for a new, self-contained gate asset
renderer. The new image must visibly demonstrate that the **phase-2 gate collision solve**
runs (not just show a final, already-separated sequence).

Non-goals: no changes to `track_gen` core, no changes to the explorer UI, no shared
drawing-helper extraction between the explorer and the asset renderer.

## Ground-truth facts (verified against current code)

- **`gate_width` is the full gate opening.** In `warp_gate` (`_finalize_endpoints_k`):
  `hw = 0.5 * gate_width`, `left = p + hw·n`, `right = p - hw·n`. Default `0.0` collapses
  `left == right == position` (point gates). When `gate_width > 0`, `_finalize_validity_k`
  additionally rejects a sequence whose gate **bars** (left–right segments) cross.
- **`max_gates` / `min_gates` validation** (`GateGenerator.__init__`): the chosen generator
  reports a reachable gate count `required_max_gates = generator_spec.max_gates(config)`.
  Construction raises if `required_max_gates < min_gates` (generator cannot produce that many)
  or if `required_max_gates > max_gates` (output buffer too small). `max_gates` sizes the
  fixed `[E*max_gates]` buffers.
- **`GateSequence` fields** (`types.py`): `position`, `tangent`, `normal`, `left`, `right`
  are flat `[E*max_gates]` `vec2f`; reshape via `wp.to_torch(...).view(E, max_gates, 2)`.
  `valid` is `[E]` int32 (0/1), `count` is `[E]` int32. Slots `i >= count[e]` are NaN-padded.
  `GateGenerator.generate()` returns the SAME instance every call; `GateSequence.clone()`
  gives an owned deep copy.
- **Registered gate generators**: `bezier` (default), `checkpoint`, `hull`, `polar`,
  `voronoi`. Ordering support is generator-specific: Bezier/Hull `{ccw, random_pairs}`;
  Polar/Voronoi/Checkpoint `{ccw, raw}`.
- **Raw-anchor inspection**: `gate_solve_iters=0` emits ordered, bbox-normalized anchors
  *before* the collision solve (the explorer exposes this as "show raw anchors"). The center
  spacing target is `2 * gate_radius`.
- **Asset renderer today** (`viz/render_readme_assets.py`): self-contained, public-API-only,
  CPU + fixed seeds; renders three **track-only** figures (`readme-generator-grid.png`,
  `readme-pipeline-stages.png`, `readme-generator-strip.png`). No gate path exists.

## Design

### 1. New gate asset renderer (the only code change)

Add to `viz/render_readme_assets.py`, matching its existing style (self-contained, imports
only from the public `track_gen` package + matplotlib/numpy, CPU, deterministic seeds):

- `GATE_GENERATORS = [("bezier","Bezier"), ("checkpoint","Checkpoint"), ("hull","Hull"),
  ("polar","Polar"), ("voronoi","Voronoi")]`.
- A slim, self-contained `_draw_gate_sequence(ax, ...)` (NOT imported from
  `viz.param_explorer`) that draws: gate centers, `gate_radius` circles, and — when drawing
  the solved row — unit tangents (quiver) and gate bars (`left`–`right`).
- `render_gate_assets()` producing `docs/assets/readme-gate-strip.png` as a **2×5 figure**,
  one column per generator:
  - **Top row — raw anchors**: build a `GateGenConfig` with `gate_solve_iters=0`; draw
    centers + radius circles. Some circles overlap (pre-collision state).
  - **Bottom row — collision-solved**: same generator, **same seed**, default
    `gate_solve_iters`, illustrative `gate_width > 0`; draw centers + radius circles +
    tangents + gate bars. Circles are separated/tangent (post-collision).
  - Per generator, pick a fixed seed and a representative env that is valid + finite in the
    solved config; render that same env index in both rows so the only difference between
    rows is the collision solve.
  - Choose `gate_radius` (and/or `scale`) for the asset so raw anchors actually overlap and
    the phase-2 separation is visible; this illustrative choice lives in the renderer.
  - Figure caption: *"Phase-2 gate collision solve separates overlapping gate spheres
    (top: raw anchors, `gate_solve_iters=0`; bottom: solved) to the `2·gate_radius` spacing
    target."* If 5 columns get cramped with circles, widen the figure (do not drop
    generators).
- Wire `render_gate_assets()` into `render_readme_assets()` so the existing
  `python -m viz.render_readme_assets` command regenerates the gate image alongside the
  track images, and append its path to the returned list.

### 2. `GateSequence` result table + aliasing note (README gate section)

After the existing gate example, add a field table parallel to the `Track` table:

| field | shape | meaning |
|---|---|---|
| `position` | `[E, G, 2]` | gate centers (`G = max_gates`) |
| `tangent`, `normal` | `[E, G, 2]` | unit tangent / left-normal at each gate |
| `left`, `right` | `[E, G, 2]` | gate endpoints (`center ± 0.5·gate_width·normal`) |
| `valid` | `[E]` bool | per-sequence validity |
| `count` | `[E]` int | real gates per env; slots `i >= count[e]` are NaN padding |

Add the same-instance aliasing sentence the Track section has: `generate()` reuses one
`GateSequence`; use `gates.clone()` for an independent snapshot.

### 3. `gate_width` + `min_gates`/`max_gates` prose (README gate section)

One short paragraph: `gate_width` is the **full** gate opening; endpoints sit at
`±0.5·gate_width` along the gate normal; default `0.0` ⇒ point gates
(`left == right == position`); `gate_width > 0` additionally invalidates a sequence whose
gate bars cross. `max_gates` sizes the fixed output buffer and must be ≥ the chosen
generator's reachable gate count; `min_gates` rejects configs the generator cannot satisfy
(both raised at `GateGenerator` construction).

### 4. Gates explorer tab (README "Parameter explorer (UI)" section)

A short bullet block describing the **Gates** tab control groups (mirroring the
`viz/param_explorer.py` "Gates" tab):

- **Gate generator** — method selector + ordering (choices follow the selected generator).
- **Gate layout** — `gate_width`, `scale`.
- **Gate collisions** — `gate_radius`, `gate_solve_iters`, and "show raw anchors" (sets
  `gate_solve_iters=0` to inspect pre-collision anchors). Center spacing target =
  `2·gate_radius`.
- **Generator-specific sampling** — point-family (Bezier/Hull), Polar, Voronoi, or
  Checkpoint controls, shown per selected generator.
- **Batch** — grid (n×n), seed, batch size; valid-yield + gate-count stats over the batch.

### 5. Embed the new image (README gate section)

Add `![…](docs/assets/readme-gate-strip.png)` with a caption that names the phase-2
collision behavior and notes `python -m viz.render_readme_assets` regenerates it.

## Testing / verification

- Run `python -m viz.render_readme_assets` on CPU; confirm it writes
  `docs/assets/readme-gate-strip.png` (plus the existing three) without error and that the
  top row shows overlapping circles while the bottom row shows separated/tangent circles.
- No `track_gen` core or explorer changes ⇒ no new unit tests; existing suite unaffected.
- Proofread README edits against the ground-truth facts above (shapes, `gate_width`
  semantics, ordering support, validation messages).

## Files touched

- `viz/render_readme_assets.py` — new gate renderer (code).
- `README.md` — items 2–5 (docs).
- `docs/assets/readme-gate-strip.png` — new committed asset (generated).
