# Track-tab generator-accurate parameter visibility

**Date:** 2026-06-27
**Status:** Design (approved approach, pending spec review)

## Problem

The Gates tab of `viz/param_explorer.py` already shows only the controls the selected
gate generator consumes, toggling them live when the generator dropdown changes. The
Tracks tab does **not**: it renders every generator-specific section at once (Bezier
controls, Polar, Voronoi, Checkpoint, hull_displacement, plus a mixed "Shape / sampling"
block), so a user on the `polar` generator still sees inert Bezier and Voronoi knobs.

Goal: make the Tracks tab expose only the parameters the selected phase-1 generator
actually uses, mirroring the Gates tab — and, while aligning the two tabs, clean up the
Gates tab so both follow the same convention.

## Scope

UI-only change to `viz/param_explorer.py` plus tests. No change to generation behavior,
the public API, or any `*_src` module. Control *values* still flow through the existing
`controls` / `_collect` / `build_config` path; only `visible=` state and section grouping
change on the track side, plus removal of dead controls on the gate side.

## Generator → parameter map (tracks)

Derived from the actual `config.<field>` reads in each generator module
(`warp_generate*.py`). Always-visible sections are generator-independent pipeline
parameters.

**Always visible** (every generator): the generator dropdown, Regime (`half_width`,
`scale`), Resolution (`spacing`, `n_max`), Relaxation (`relax_iters`, `sep`, `spc`,
`bend`, `margin`), PBD separation (`sep_every`, `sep_slots`, `sep_skin`), Batch
(`grid_n`, `seed`, `batch_size`).

**Generator-specific sections** (header + its controls toggle together):

| Section | Controls | Visible for |
|---|---|---|
| Sampling (point-family) | min corners, max corners, min_point_distance | bezier, hull |
| Curve smoothing | num_points_per_segment | bezier, hull, polar, voronoi |
| Bezier controls | rad, edgy, handle_clamp_frac | bezier |
| Hull controls | hull_displacement | hull |
| Polar knot spline | polar_knots, polar_radial, polar_angular | polar |
| Voronoi graph cycle | vor_sites, vor_layout, vor_control, vor_radial, vor_angular | voronoi |
| Checkpoint steering | checkpoint_* (8 controls) | checkpoint |

This is a "light regroup": `num_points_per_segment` moves out of today's mixed
"Shape / sampling" header into its own one-control "Curve smoothing" section (it is the
single knob used by 4 of 5 generators, so it cannot share the bezier/hull-only Sampling
header), and `hull_displacement` moves into its own Hull section. Control creation order
in the column otherwise stays as-is; only the section headers split so no header ever
renders with an orphaned control for a generator that ignores it.

## Mechanism (mirrors the gate tab)

1. Add a pure helper next to `gate_visible_sections`:

   ```python
   def track_visible_sections(generator: str) -> dict[str, bool]:
       name = str(generator)
       point = name in {"bezier", "hull"}
       return {
           "sampling": point,
           "smoothing": name in {"bezier", "hull", "polar", "voronoi"},
           "bezier": name == "bezier",
           "hull": name == "hull",
           "polar": name == "polar",
           "voronoi": name == "voronoi",
           "checkpoint": name == "checkpoint",
       }
   ```

2. At build time, initialize each generator-specific control and its section-header
   `gr.Markdown` with `visible=track_visible_sections(generator_default)[<section>]`.

3. Wire a `generator.change(_track_mode_update, [generator], track_mode_outputs)` handler
   that returns an ordered list of `gr.update(visible=...)` for exactly those components,
   structured like the existing `_gate_mode_update` (but with no ordering-dropdown update,
   since track generators have no per-generator ordering constraint).

4. `_track_mode_update` recomputes visibility only; it never changes control values, so
   `generate`/`_collect`/`build_config` are untouched.

## Gate-tab alignment ("match", option b — done properly)

The Gates tab carries five controls that **no** gate generator consumes — they are created
`visible=False` permanently and only feed `GateGenConfig` defaults:
`gate_num_points_per_segment`, `gate_rad`, `gate_edgy`, `gate_handle_clamp_frac`,
`gate_hull_displacement`. Remove them so the two tabs share one clean convention with no
dead controls:

- Delete the five `gr.Slider` definitions and drop them from `gate_controls` and from the
  `gate_mode_outputs` list + the matching `gr.update(...)` rows in `_gate_mode_update`.
- Remove their keys from `GATE_CONTROL_KEYS` and from `default_gate_params`.
- In `build_gate_config`, drop the `num_points_per_segment=`, `rad=`, `edgy=`,
  `handle_clamp_frac=`, and `hull_displacement=` arguments so `GateGenConfig` applies its
  defaults. These fields are inert on the gate path (only corner anchors are sampled; the
  Bezier/hull curve assembly never runs), so generated output is unchanged.
- Rename the gate "Point-family controls" header to **"Sampling (point-family)"** to match
  the track section name, so both tabs use identical section naming.

`gate_min_num_points`, `gate_max_num_points`, and `gate_min_point_distance` stay — the gate
corner sampler does consume them.

## Testing

In `tests/test_param_explorer.py`:

- Add `track_visible_sections` coverage: for each of the five generators assert the exact
  set of `True` sections (e.g. `polar` → only `smoothing` + `polar`; `checkpoint` → only
  `checkpoint`; `bezier` → `sampling` + `smoothing` + `bezier`).
- Update existing gate tests for the removed keys: assert the five dropped keys are absent
  from `GATE_CONTROL_KEYS`, and that `build_gate_config` still produces a valid
  `GateGenConfig` (with default inert fields) from a params dict lacking them.
- Confirm `len(gate_controls) == len(GATE_CONTROL_KEYS)` still holds after removal (the
  build-time guard already asserts this).

## Out of scope

- Removing the inert point-family fields from `GateGenConfig` itself (they remain for
  parity with `TrackGenConfig`; already documented as inert on the gate path).
- Any reordering of the always-visible pipeline sections.
- The track tab's per-generator ordering (tracks have none).
