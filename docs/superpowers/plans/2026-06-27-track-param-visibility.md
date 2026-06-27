# Track-tab generator-accurate parameter visibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the Tracks tab of `viz/param_explorer.py`, show only the controls the selected phase-1 generator consumes (toggling live on generator change), and remove the 5 permanently-hidden inert sliders from the Gates tab so both tabs share one clean convention.

**Architecture:** Pure UI change. Add a `track_visible_sections(generator)` helper mirroring `gate_visible_sections`, regroup the track generator-specific controls into clean per-generator sections, initialize each with `visible=`, and wire a `generator.change` handler that returns `gr.update(visible=...)`. Control values still flow through the untouched `controls`/`_collect`/`build_config` path. On the gate side, delete the inert controls and rename one header.

**Tech Stack:** Python, Gradio (`ui` extra), pytest. Run tests with `env -u PYTHONPATH .venv/bin/python -m pytest`.

Spec: `docs/superpowers/specs/2026-06-27-track-param-visibility-design.md`

---

## File structure

- Modify: `viz/param_explorer.py` — add `track_visible_sections`; regroup + toggle track controls; remove inert gate controls; rename gate header.
- Test: `tests/test_param_explorer.py` — add visibility coverage; assert dropped gate keys.

No other files change. No `track_gen/_src/*` change.

---

### Task 1: `track_visible_sections` helper

**Files:**
- Modify: `viz/param_explorer.py` (add function next to `gate_visible_sections`, after line 387)
- Test: `tests/test_param_explorer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_param_explorer.py`:

```python
def test_track_visible_sections_are_generator_specific():
    assert px.track_visible_sections("bezier") == {
        "sampling": True, "smoothing": True, "bezier": True, "hull": False,
        "polar": False, "voronoi": False, "checkpoint": False,
    }
    assert px.track_visible_sections("hull") == {
        "sampling": True, "smoothing": True, "bezier": False, "hull": True,
        "polar": False, "voronoi": False, "checkpoint": False,
    }
    assert px.track_visible_sections("polar") == {
        "sampling": False, "smoothing": True, "bezier": False, "hull": False,
        "polar": True, "voronoi": False, "checkpoint": False,
    }
    assert px.track_visible_sections("voronoi") == {
        "sampling": False, "smoothing": True, "bezier": False, "hull": False,
        "polar": False, "voronoi": True, "checkpoint": False,
    }
    assert px.track_visible_sections("checkpoint") == {
        "sampling": False, "smoothing": False, "bezier": False, "hull": False,
        "polar": False, "voronoi": False, "checkpoint": True,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py::test_track_visible_sections_are_generator_specific -v`
Expected: FAIL with `AttributeError: module 'viz.param_explorer' has no attribute 'track_visible_sections'`

- [ ] **Step 3: Write minimal implementation**

In `viz/param_explorer.py`, immediately after the `gate_visible_sections` function (after line 387), add:

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

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py::test_track_visible_sections_are_generator_specific -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "feat: add track_visible_sections helper for per-generator UI"
```

---

### Task 2: Regroup track controls into clean sections + toggle visibility

**Files:**
- Modify: `viz/param_explorer.py` — track tab build (lines ~674-729), event wiring (after `controls = [...]`, line ~773)
- Test: `tests/test_param_explorer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_param_explorer.py`:

```python
def _visible_by_label(app, label):
    """Return the visible prop of the FIRST app component with this exact label."""
    for c in app.config["components"]:
        props = c.get("props", {})
        if props.get("label") == label:
            return props.get("visible", True)
    raise AssertionError(f"no component labeled {label!r}")


def test_track_tab_shows_only_selected_generator_controls():
    pytest.importorskip("gradio")
    app = px.build_app()
    markdown = {
        c.get("props", {}).get("value")
        for c in app.config["components"]
        if c.get("type") == "markdown"
    }
    # New clean section headers exist; the old mixed header is gone.
    assert "### Sampling (point-family)" in markdown
    assert "### Curve smoothing" in markdown
    assert "### Hull controls" in markdown
    assert "### Shape / sampling" not in markdown

    # Default track generator is "polar": only smoothing + polar sections visible.
    # Use track-unique labels (gate tab reuses some bare names).
    assert _visible_by_label(app, "num_points_per_segment (generator smoothing samples)") is True
    assert _visible_by_label(app, "min corners") is False
    assert _visible_by_label(app, "rad (roundness)") is False
    assert _visible_by_label(app, "hull_displacement (hull midpoint displacement)") is False
    assert _visible_by_label(app, "checkpoint_count (radial waypoints)") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py::test_track_tab_shows_only_selected_generator_controls -v`
Expected: FAIL — `### Shape / sampling` is still in markdown (old header present), and the hidden-control assertions fail because controls are currently always visible.

- [ ] **Step 3a: Compute section visibility at build time**

In `viz/param_explorer.py`, in the track tab, after the `generator = gr.Dropdown(...)` line (line 677), insert:

```python
                        track_mode_visible = track_visible_sections(generator_default)
```

- [ ] **Step 3b: Replace the generator-specific control block**

Replace the entire block from `gr.Markdown("### Shape / sampling")` through the last checkpoint slider (current lines 681-729) with the following. Note `hull_displacement` moves into its own Hull section and `num_points_per_segment` into its own Curve-smoothing section; every generator-specific control and header gets a `visible=` flag:

```python
                        sampling_md = gr.Markdown("### Sampling (point-family)",
                                                  visible=track_mode_visible["sampling"])
                        min_np = gr.Slider(5, 20, value=defaults["min_num_points"], step=1,
                                           label="min corners", visible=track_mode_visible["sampling"])
                        max_np = gr.Slider(5, 20, value=defaults["max_num_points"], step=1,
                                           label="max corners", visible=track_mode_visible["sampling"])
                        min_dist = gr.Slider(0.02, 0.20, value=defaults["min_point_distance"], step=0.005,
                                             label="min_point_distance (sampling spread)",
                                             visible=track_mode_visible["sampling"])
                        smoothing_md = gr.Markdown("### Curve smoothing",
                                                   visible=track_mode_visible["smoothing"])
                        samples_per_seg = gr.Slider(8, 60, value=defaults["num_points_per_segment"], step=1,
                                                    label="num_points_per_segment (generator smoothing samples)",
                                                    visible=track_mode_visible["smoothing"])
                        bezier_md = gr.Markdown("### Bezier controls", visible=track_mode_visible["bezier"])
                        rad = gr.Slider(0.0, 0.6, value=defaults["rad"], step=0.01, label="rad (roundness)",
                                        visible=track_mode_visible["bezier"])
                        edgy = gr.Slider(0.0, 1.0, value=defaults["edgy"], step=0.05, label="edgy",
                                         visible=track_mode_visible["bezier"])
                        handle_clamp = gr.Slider(0.0, 1.0, value=defaults["handle_clamp_frac"], step=0.01,
                                                 label="handle_clamp_frac (overshoot<->roundness)",
                                                 visible=track_mode_visible["bezier"])
                        hull_md = gr.Markdown("### Hull controls", visible=track_mode_visible["hull"])
                        hull_disp = gr.Slider(0.0, 0.8, value=defaults["hull_displacement"], step=0.01,
                                              label="hull_displacement (hull midpoint displacement)",
                                              visible=track_mode_visible["hull"])
                        polar_md = gr.Markdown("### Polar knot spline", visible=track_mode_visible["polar"])
                        polar_knots = gr.Slider(4, 24, value=defaults["polar_num_knots"], step=1,
                                                label="polar knots", visible=track_mode_visible["polar"])
                        polar_radial = gr.Slider(0.0, 0.85, value=defaults["polar_radial_jitter"], step=0.01,
                                                 label="polar radial jitter", visible=track_mode_visible["polar"])
                        polar_angular = gr.Slider(0.0, 0.45, value=defaults["polar_angular_jitter"], step=0.01,
                                                  label="polar angular jitter", visible=track_mode_visible["polar"])
                        vor_md = gr.Markdown("### Voronoi graph cycle", visible=track_mode_visible["voronoi"])
                        vor_sites = gr.Slider(32, 512, value=defaults["voronoi_num_sites"], step=16,
                                              label="voronoi sites", visible=track_mode_visible["voronoi"])
                        vor_layout = gr.Dropdown(["void_ring", "ring", "clustered", "mixed"],
                                                 value=defaults["voronoi_site_layout"],
                                                 label="voronoi site layout", visible=track_mode_visible["voronoi"])
                        vor_control = gr.Slider(6, 32, value=defaults["voronoi_control_points"], step=1,
                                                label="voronoi control points", visible=track_mode_visible["voronoi"])
                        vor_radial = gr.Slider(0.0, 0.85, value=defaults["voronoi_radial_variation"], step=0.01,
                                               label="voronoi radial variation", visible=track_mode_visible["voronoi"])
                        vor_angular = gr.Slider(0.0, 0.25, value=defaults["voronoi_angular_jitter"], step=0.01,
                                                label="voronoi angular jitter", visible=track_mode_visible["voronoi"])
                        checkpoint_md = gr.Markdown("### Checkpoint steering", visible=track_mode_visible["checkpoint"])
                        checkpoint_count = gr.Slider(4, 24, value=defaults["checkpoint_count"], step=1,
                                                     label="checkpoint_count (radial waypoints)",
                                                     visible=track_mode_visible["checkpoint"])
                        checkpoint_radius_min_frac = gr.Slider(0.1, 0.9, value=defaults["checkpoint_radius_min_frac"],
                                                               step=0.01, label="checkpoint_radius_min_frac",
                                                               visible=track_mode_visible["checkpoint"])
                        checkpoint_angle_jitter = gr.Slider(0.0, 0.9, value=defaults["checkpoint_angle_jitter"],
                                                            step=0.01, label="checkpoint_angle_jitter",
                                                            visible=track_mode_visible["checkpoint"])
                        checkpoint_turn_rate = gr.Slider(0.1, 1.0, value=defaults["checkpoint_turn_rate"],
                                                         step=0.01, label="checkpoint_turn_rate",
                                                         visible=track_mode_visible["checkpoint"])
                        checkpoint_steer_gain = gr.Slider(0.1, 1.0, value=defaults["checkpoint_steer_gain"],
                                                          step=0.01, label="checkpoint_steer_gain",
                                                          visible=track_mode_visible["checkpoint"])
                        checkpoint_lookahead_frac = gr.Slider(0.05, 0.4, value=defaults["checkpoint_lookahead_frac"],
                                                              step=0.01, label="checkpoint_lookahead_frac",
                                                              visible=track_mode_visible["checkpoint"])
                        checkpoint_best_of_k = gr.Slider(1, 8, value=defaults["checkpoint_best_of_k"], step=1,
                                                         label="checkpoint_best_of_k (candidates)",
                                                         visible=track_mode_visible["checkpoint"])
                        checkpoint_clip_fallback = gr.Checkbox(value=defaults["checkpoint_clip_fallback"],
                                                               label="checkpoint_clip_fallback (single-crossing rescue)",
                                                               visible=track_mode_visible["checkpoint"])
```

- [ ] **Step 3c: Add the mode-update handler and wire it**

In `viz/param_explorer.py`, immediately after the `controls = [...]` list (after line 773) and before `def _generate(*vals):`, insert:

```python
                track_mode_outputs = [
                    sampling_md, min_np, max_np, min_dist,
                    smoothing_md, samples_per_seg,
                    bezier_md, rad, edgy, handle_clamp,
                    hull_md, hull_disp,
                    polar_md, polar_knots, polar_radial, polar_angular,
                    vor_md, vor_sites, vor_layout, vor_control, vor_radial, vor_angular,
                    checkpoint_md, checkpoint_count, checkpoint_radius_min_frac,
                    checkpoint_angle_jitter, checkpoint_turn_rate, checkpoint_steer_gain,
                    checkpoint_lookahead_frac, checkpoint_best_of_k, checkpoint_clip_fallback,
                ]

                def _track_mode_update(generator_name):
                    v = track_visible_sections(generator_name)
                    counts = [("sampling", 4), ("smoothing", 2), ("bezier", 4), ("hull", 2),
                              ("polar", 4), ("voronoi", 6), ("checkpoint", 9)]
                    updates = []
                    for key, n in counts:
                        updates.extend([gr.update(visible=v[key])] * n)
                    return updates

                generator.change(_track_mode_update, [generator], track_mode_outputs)
```

(The 31 components in `track_mode_outputs` line up with `sum(n for _, n in counts) == 31`: each group is one header markdown plus its controls.)

- [ ] **Step 4: Run the focused test, then the full suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py -v`
Expected: PASS (new test + all existing param-explorer tests, including `test_build_app_smoke`).

Then: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "feat: hide non-selected track generator controls in explorer"
```

---

### Task 3: Remove the 5 inert controls from the Gates tab + rename header

**Files:**
- Modify: `viz/param_explorer.py` — `GATE_CONTROL_KEYS` (lines 58-68), `default_gate_params` (lines 348-352), `build_gate_config` (lines 430-434), gate UI sliders (lines 858-867), `gate_controls` list (lines 926-929), gate "Point-family" header (line 848)
- Test: `tests/test_param_explorer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_param_explorer.py`:

```python
def test_gate_controls_drop_inert_point_family_knobs():
    dropped = ("gate_num_points_per_segment", "gate_rad", "gate_edgy",
               "gate_handle_clamp_frac", "gate_hull_displacement")
    for key in dropped:
        assert key not in px.GATE_CONTROL_KEYS
        assert key not in px.default_gate_params()


def test_build_gate_config_uses_defaults_for_dropped_knobs():
    from track_gen._src.types import GateGenConfig
    d = GateGenConfig()
    cfg = px.build_gate_config(_gate_params(gate_generator="bezier"))
    assert cfg.num_points_per_segment == d.num_points_per_segment
    assert cfg.rad == d.rad
    assert cfg.edgy == d.edgy
    assert cfg.handle_clamp_frac == d.handle_clamp_frac
    assert cfg.hull_displacement == d.hull_displacement
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py::test_gate_controls_drop_inert_point_family_knobs tests/test_param_explorer.py::test_build_gate_config_uses_defaults_for_dropped_knobs -v`
Expected: FAIL — keys still present in `GATE_CONTROL_KEYS`/`default_gate_params`; `build_gate_config` still reads `p["gate_rad"]` etc.

- [ ] **Step 3a: Drop keys from `GATE_CONTROL_KEYS`**

Replace lines 58-68 (`GATE_CONTROL_KEYS = [...]`) with:

```python
GATE_CONTROL_KEYS = [
    "gate_generator", "gate_ordering", "gate_width", "gate_radius", "gate_solve_iters",
    "gate_show_raw", "gate_scale", "gate_min_num_points", "gate_max_num_points",
    "gate_min_point_distance",
    "gate_polar_num_knots", "gate_polar_radial_jitter", "gate_polar_angular_jitter",
    "gate_voronoi_num_sites", "gate_voronoi_site_layout", "gate_voronoi_control_points",
    "gate_voronoi_radial_variation", "gate_voronoi_angular_jitter",
    "gate_checkpoint_count", "gate_checkpoint_radius_min_frac",
    "gate_checkpoint_angle_jitter", "gate_grid_n", "gate_seed", "gate_batch_size",
]
```

- [ ] **Step 3b: Drop keys from `default_gate_params`**

Remove these 5 lines from `default_gate_params` (lines 348-352):

```python
        "gate_num_points_per_segment": cfg.num_points_per_segment,
        "gate_rad": cfg.rad,
        "gate_edgy": cfg.edgy,
        "gate_handle_clamp_frac": cfg.handle_clamp_frac,
        "gate_hull_displacement": cfg.hull_displacement,
```

- [ ] **Step 3c: Drop the args from `build_gate_config`**

Remove these 5 lines from the `GateGenConfig(...)` call in `build_gate_config` (lines 430-434), so the inert fields fall back to `GateGenConfig` defaults:

```python
        num_points_per_segment=int(p["gate_num_points_per_segment"]),
        rad=float(p["gate_rad"]),
        edgy=float(p["gate_edgy"]),
        handle_clamp_frac=float(p["gate_handle_clamp_frac"]),
        hull_displacement=float(p["gate_hull_displacement"]),
```

- [ ] **Step 3d: Delete the 5 gate slider definitions**

Remove the 5 slider definitions (lines 858-867): `gate_samples_per_seg`, `gate_rad`, `gate_edgy`, `gate_handle_clamp`, `gate_hull_disp`.

- [ ] **Step 3e: Drop them from the `gate_controls` list**

In the `gate_controls = [...]` list (lines 926-929), remove `gate_samples_per_seg, gate_rad, gate_edgy, gate_handle_clamp, gate_hull_disp`. The relevant lines change from:

```python
                gate_controls = [gate_generator, gate_ordering, gate_width, gate_radius, gate_solve_iters,
                                 gate_show_raw, gate_scale, gate_min_np, gate_max_np,
                                 gate_min_point_distance, gate_samples_per_seg, gate_rad,
                                 gate_edgy, gate_handle_clamp, gate_hull_disp,
```

to:

```python
                gate_controls = [gate_generator, gate_ordering, gate_width, gate_radius, gate_solve_iters,
                                 gate_show_raw, gate_scale, gate_min_np, gate_max_np,
                                 gate_min_point_distance,
```

(Leave the remaining `gate_controls` entries — polar/voronoi/checkpoint/grid/seed/batch — unchanged.)

- [ ] **Step 3f: Rename the gate point-family header**

On line 848, change:

```python
                        gate_point_md = gr.Markdown("### Point-family controls", visible=gate_mode_visible["point"])
```

to:

```python
                        gate_point_md = gr.Markdown("### Sampling (point-family)", visible=gate_mode_visible["point"])
```

- [ ] **Step 4: Run the focused tests, then the full suite**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/test_param_explorer.py -v`
Expected: PASS — the new gate tests pass; `test_build_app_smoke` confirms `len(gate_controls) == len(GATE_CONTROL_KEYS)` still holds (both shrank by 5); `test_gate_app_labels_explain_units_and_collision_stage` still passes (it asserts none of the dropped labels).

Then: `env -u PYTHONPATH .venv/bin/python -m pytest -q`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "refactor: drop inert point-family controls from gate explorer tab"
```

---

## Self-review notes

- **Spec coverage:** §"Generator → parameter map" → Task 2 control grouping + Task 1 helper; §"Mechanism" → Task 1 (helper) + Task 2 (build-time `visible=` and `generator.change`); §"Gate-tab alignment" → Task 3 (all five sub-edits + header rename); §"Testing" → tests in Tasks 1-3. All covered.
- **Type/name consistency:** `track_visible_sections` keys (`sampling/smoothing/bezier/hull/polar/voronoi/checkpoint`) are identical in Task 1 impl, Task 1 test, and the Task 2 `track_mode_visible[...]` / `_track_mode_update` lookups. `track_mode_outputs` (31 entries) matches `sum(counts) == 31`.
- **No-behavior-change guards:** the track `controls`/`_collect` lists are untouched (visibility-only), and dropped gate fields fall back to `GateGenConfig` defaults that are inert on the gate path — so generated output is unchanged on both tabs. The build-time length guard catches any gate list/key mismatch.
