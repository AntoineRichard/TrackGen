"""Parameter-explorer core (build_config + render_grid) on CPU, plus a gradio app-builds smoke.

Output is always constant_spacing (the only supported mode): per-track boundary arrays are
[E, N_max, 2] NaN-padded with a per-env real-point count in ``track.count`` that VARIES per
env. Stats/assertions are count-aware (mean count over valid tracks; never count == N_max).
"""
import pytest
import torch

from viz import param_explorer as px
from track_gen._src.types import TrackGenConfig


def _params(**over):
    p = dict(half_width=0.5, scale=10.0, min_num_points=9, max_num_points=13,
             min_point_distance=0.05, num_points_per_segment=30, hull_displacement=0.15,
             rad=0.2, edgy=0.0, handle_clamp_frac=0.10,
             polar_num_knots=12, polar_radial_jitter=0.60, polar_angular_jitter=0.30,
             voronoi_num_sites=256, voronoi_site_layout="void_ring",
             voronoi_control_points=18, voronoi_radial_variation=0.62,
             voronoi_angular_jitter=0.08,
             num_points=256, spacing=0.30, n_max=384, relax_iters=40,
             relax_sep_relax=1.0, relax_spc_relax=1.0, relax_bend_relax=1.5,
             relax_margin=0.15, grid_n=3, seed=0, batch_size=16)
    p.update(over)
    return p


def _gate_params(**over):
    p = px.default_gate_params()
    p.update({"gate_grid_n": 2, "gate_batch_size": 4})
    p.update(over)
    return p


def test_default_params_favor_polar_knot_method():
    defaults = TrackGenConfig()
    params = px.default_params()
    cfg = px.build_config(params)
    assert cfg.generator == "polar"
    assert cfg.polar_num_knots == defaults.polar_num_knots
    assert cfg.polar_radial_jitter == defaults.polar_radial_jitter
    assert cfg.polar_angular_jitter == defaults.polar_angular_jitter


def test_build_config_maps_polar_controls():
    cfg = px.build_config(_params(
        generator="polar",
        polar_num_knots=16,
        polar_radial_jitter=0.72,
        polar_angular_jitter=0.22,
    ))
    assert cfg.generator == "polar"
    assert cfg.polar_num_knots == 16
    assert abs(cfg.polar_radial_jitter - 0.72) < 1e-9
    assert abs(cfg.polar_angular_jitter - 0.22) < 1e-9


def test_build_config_maps_and_clamps():
    cfg = px.build_config(_params(min_num_points=15, max_num_points=8))
    assert cfg.min_num_points <= cfg.max_num_points
    assert cfg.num_envs == 16  # batch_size=16 from _params default
    assert abs(cfg.half_width - 0.5) < 1e-9 and abs(cfg.scale - 10.0) < 1e-9
    assert abs(cfg.handle_clamp_frac - 0.10) < 1e-9  # the overshoot-clamp knob round-trips
    cfg2 = px.build_config(_params(spacing=0.3, n_max=384))
    assert cfg2.N_max == 384
    assert abs(cfg2.spacing - 0.3) < 1e-9


def test_build_config_auto_spacing():
    # spacing omitted -> config auto-couples it to 0.6*half_width (constant-spacing default).
    p = _params(half_width=0.5)
    p["spacing"] = None
    cfg = px.build_config(p)
    assert abs(cfg.spacing - 0.6 * 0.5) < 1e-9


def test_build_config_maps_hull_shape_knobs():
    cfg = px.build_config(_params(
        generator="hull",
        hull_displacement=0.42,
        min_point_distance=0.08,
        num_points_per_segment=24,
    ))
    assert cfg.generator == "hull"
    assert abs(cfg.hull_displacement - 0.42) < 1e-9
    assert abs(cfg.min_point_distance - 0.08) < 1e-9
    assert cfg.num_points_per_segment == 24


def test_build_config_maps_voronoi_shape_knobs():
    cfg = px.build_config(_params(
        generator="voronoi",
        voronoi_num_sites=512,
        voronoi_site_layout="clustered",
        voronoi_control_points=22,
        voronoi_radial_variation=0.70,
        voronoi_angular_jitter=0.12,
    ))
    assert cfg.generator == "voronoi"
    assert cfg.voronoi_num_sites == 512
    assert cfg.voronoi_site_layout == "clustered"
    assert cfg.voronoi_control_points == 22
    assert abs(cfg.voronoi_radial_variation - 0.70) < 1e-9
    assert abs(cfg.voronoi_angular_jitter - 0.12) < 1e-9


def test_build_config_maps_repulsive_shape_knobs():
    cfg = px.build_config(_params(
        generator="repulsive",
        repulsive_grow_mult_min=4.0,
        repulsive_grow_mult_max=6.0,
        repulsive_domain_frac=0.30,
        repulsive_domain_init_ratio=3.5,
        repulsive_obstacle_count_min=6,
        repulsive_obstacle_count_max=14,
        repulsive_obstacle_radius_min_frac=0.015,
        repulsive_obstacle_radius_max_frac=0.05,
        repulsive_ratchet_rate=0.010,
        repulsive_alpha=2.5,
        repulsive_beta=5.5,
        repulsive_tau=0.35,
        repulsive_w_len=25.0,
        repulsive_settle_iters=30,
        repulsive_resample_every=20,
        repulsive_stall_window=12,
        repulsive_stall_area_tol=0.04,
        repulsive_deactivate_obstacles=False,
    ))
    assert cfg.generator == "repulsive"
    assert abs(cfg.repulsive_grow_mult_min - 4.0) < 1e-9
    assert abs(cfg.repulsive_grow_mult_max - 6.0) < 1e-9
    assert abs(cfg.repulsive_domain_frac - 0.30) < 1e-9
    assert abs(cfg.repulsive_domain_init_ratio - 3.5) < 1e-9
    assert cfg.repulsive_obstacle_count_min == 6
    assert cfg.repulsive_obstacle_count_max == 14
    assert abs(cfg.repulsive_obstacle_radius_min_frac - 0.015) < 1e-9
    assert abs(cfg.repulsive_obstacle_radius_max_frac - 0.05) < 1e-9
    assert abs(cfg.repulsive_ratchet_rate - 0.010) < 1e-9
    assert abs(cfg.repulsive_alpha - 2.5) < 1e-9
    assert abs(cfg.repulsive_beta - 5.5) < 1e-9
    assert abs(cfg.repulsive_tau - 0.35) < 1e-9
    assert abs(cfg.repulsive_w_len - 25.0) < 1e-9
    assert cfg.repulsive_settle_iters == 30
    assert cfg.repulsive_resample_every == 20
    assert cfg.repulsive_stall_window == 12
    assert abs(cfg.repulsive_stall_area_tol - 0.04) < 1e-9
    assert cfg.repulsive_deactivate_obstacles is False


def test_build_gate_config_maps_solver_and_shape_knobs():
    cfg = px.build_gate_config(_gate_params(
        gate_generator="bezier",
        gate_ordering="random_pairs",
        gate_width=0.2,
        gate_radius=0.1,
        gate_solve_iters=11,
        gate_scale=2.5,
        gate_min_num_points=7,
        gate_max_num_points=12,
    ))
    assert cfg.generator == "bezier"
    assert cfg.gate_ordering == "random_pairs"
    assert cfg.min_gates == 7
    assert cfg.max_gates == 12
    assert abs(cfg.gate_width - 0.2) < 1e-9
    assert abs(cfg.gate_radius - 0.1) < 1e-9
    assert cfg.gate_solve_iters == 11
    assert abs(cfg.scale - 2.5) < 1e-9
    assert cfg.min_num_points == 7
    assert cfg.max_num_points == 12


def test_build_gate_config_derives_fixed_mode_gate_counts():
    cases = [
        ("polar", "gate_polar_num_knots", 14),
        ("voronoi", "gate_voronoi_control_points", 21),
        ("checkpoint", "gate_checkpoint_count", 9),
    ]
    for generator, key, count in cases:
        cfg = px.build_gate_config(_gate_params(gate_generator=generator, **{key: count}))
        assert cfg.min_gates == count
        assert cfg.max_gates == count


def test_build_gate_config_raw_toggle_disables_solver():
    cfg = px.build_gate_config(_gate_params(gate_show_raw=True, gate_solve_iters=13))
    assert cfg.gate_solve_iters == 0


def test_gate_supported_orderings_follow_generator_capabilities():
    assert px.gate_supported_orderings("bezier") == ["ccw", "random_pairs"]
    assert px.gate_supported_orderings("hull") == ["ccw", "random_pairs"]
    assert px.gate_supported_orderings("polar") == ["ccw", "raw"]
    assert px.gate_supported_orderings("voronoi") == ["ccw", "raw"]
    assert px.gate_supported_orderings("checkpoint") == ["ccw", "raw"]


def test_gate_visible_sections_are_generator_specific():
    assert px.gate_visible_sections("bezier") == {
        "point": True,
        "polar": False,
        "voronoi": False,
        "checkpoint": False,
    }
    assert px.gate_visible_sections("polar")["polar"] is True
    assert px.gate_visible_sections("voronoi")["voronoi"] is True
    assert px.gate_visible_sections("checkpoint")["checkpoint"] is True


def test_build_gate_config_clamps_unsupported_ordering():
    cfg = px.build_gate_config(_gate_params(gate_generator="checkpoint", gate_ordering="random_pairs"))
    assert cfg.generator == "checkpoint"
    assert cfg.gate_ordering == "ccw"


def test_render_gate_grid_runs():
    fig, stats = px.render_gate_grid(_gate_params(
        gate_generator="bezier",
        gate_ordering="ccw",
        gate_width=0.05,
        gate_radius=0.025,
    ))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 4
    assert 0.0 <= stats["yield"] <= 1.0
    assert stats["n_valid"] + stats["n_invalid"] == 4
    assert stats["target_center_distance"] >= 0.05
    import matplotlib.pyplot as plt
    plt.close(fig)


import matplotlib.figure


def test_render_grid_constant_spacing():
    fig, stats = px.render_grid(_params(grid_n=3, spacing=0.30, n_max=384))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 9                       # grid_n**2 cells
    assert 0.0 <= stats["yield"] <= 1.0
    # constant_spacing: count is a per-track real-point count averaged over valid tracks,
    # so it is variable (>= 1) and must NOT equal N_max.
    assert stats["count"] >= 1
    assert stats["count"] <= 384                    # capped by N_max
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_grid_constant_spacing_runs():
    fig, stats = px.render_grid(_params(grid_n=3, spacing=0.30, n_max=384))
    assert 0.0 <= stats["yield"] <= 1.0
    assert stats["count"] >= 1                       # variable per-track count (mean over valid)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_build_app_smoke():
    gr = pytest.importorskip("gradio")            # skip if the ui extra isn't installed
    app = px.build_app()
    assert isinstance(app, gr.Blocks)


def test_gate_app_labels_explain_units_and_collision_stage():
    pytest.importorskip("gradio")
    app = px.build_app()
    labels = {
        c.get("props", {}).get("label")
        for c in app.config["components"]
        if c.get("props", {}).get("label")
    }
    markdown = {
        c.get("props", {}).get("value")
        for c in app.config["components"]
        if c.get("type") == "markdown"
    }

    assert "### Gate Collisions" in markdown
    assert "Center spacing target = 2 * gate_radius." in markdown
    assert "gate_width [world units]" in labels
    assert "gate_radius [world units]" in labels
    assert "scale [x]" in labels
    assert "min_point_distance [pre-scale world units]" in labels
    assert "min gates" not in labels
    assert "max gates" not in labels
    assert "min sampled anchors" in labels
    assert "max sampled anchors" in labels
    assert "For Bezier/Hull gates, sampled anchors become gate centers." in markdown


def test_batch_and_pagination():
    import warp as wp
    p = _params(grid_n=3, batch_size=20, spacing=0.30, n_max=384)
    track = px.generate_batch(p)
    # Track fields are wp.array; center is flat [E*N_max] vec2f, count is [E] int32.
    center_t = wp.to_torch(track.center).view(20, 384, 3)[..., :2]
    count_t = wp.to_torch(track.count)
    assert center_t.shape[0] == 20                      # full batch generated
    assert center_t.shape[1] == 384                     # NaN-padded to N_max
    assert count_t.shape[0] == 20
    # constant_spacing output is NaN-padded: each env has count[e] real points then NaN pad.
    for e in range(20):
        c = int(count_t[e].item())
        assert 1 <= c <= 384
        finite = torch.isfinite(center_t[e]).all(dim=-1)
        # exactly the first count[e] points are real (finite); the rest are NaN padding.
        assert int(finite.sum().item()) == c
        assert bool(finite[:c].all().item())
        if c < 384:
            assert not bool(finite[c:].any().item())
    import matplotlib.figure
    f0 = px.render_page(track, 0, 3); assert isinstance(f0, matplotlib.figure.Figure) and len(f0.axes) == 9
    f1 = px.render_page(track, 1, 3); assert len(f1.axes) == 9   # page 2 (start=9)
    import matplotlib.pyplot as plt; plt.close(f0); plt.close(f1)
    st = px._stats(track)
    assert 0.0 <= st["yield"] <= 1.0                   # stats over all 20, not 9
    # mean count over valid tracks is variable (constant_spacing), never a fixed N.
    assert st["count"] >= 1
    assert px.n_pages(20, 3) == 3                       # ceil(20/9)


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


def test_track_mode_visibility_orders_flags_to_match_outputs():
    # The generator.change handler emits these flags positionally onto
    # track_mode_outputs, so order + count must be exact. Checking non-default
    # generators (the default build is polar) catches section/count drift that a
    # polar-only build-time assertion cannot.
    assert px.track_mode_visibility("bezier") == (
        [True] * 4 + [True] * 2 + [True] * 4 + [False] * 2
        + [False] * 4 + [False] * 6 + [False] * 9 + [False] * 19
    )
    assert px.track_mode_visibility("hull") == (
        [True] * 4 + [True] * 2 + [False] * 4 + [True] * 2
        + [False] * 4 + [False] * 6 + [False] * 9 + [False] * 19
    )
    assert px.track_mode_visibility("voronoi") == (
        [False] * 4 + [True] * 2 + [False] * 4 + [False] * 2
        + [False] * 4 + [True] * 6 + [False] * 9 + [False] * 19
    )
    assert px.track_mode_visibility("checkpoint") == (
        [False] * (4 + 2 + 4 + 2 + 4 + 6) + [True] * 9 + [False] * 19
    )
    assert px.track_mode_visibility("repulsive") == (
        [False] * (4 + 2 + 4 + 2 + 4 + 6 + 9) + [True] * 19
    )
    # Every generator yields the same fixed number of flags (one per output component).
    for gen in ("bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive"):
        assert len(px.track_mode_visibility(gen)) == 31 + 19


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


def test_track_visible_sections_are_generator_specific():
    assert px.track_visible_sections("bezier") == {
        "sampling": True, "smoothing": True, "bezier": True, "hull": False,
        "polar": False, "voronoi": False, "checkpoint": False, "repulsive": False,
    }
    assert px.track_visible_sections("hull") == {
        "sampling": True, "smoothing": True, "bezier": False, "hull": True,
        "polar": False, "voronoi": False, "checkpoint": False, "repulsive": False,
    }
    assert px.track_visible_sections("polar") == {
        "sampling": False, "smoothing": True, "bezier": False, "hull": False,
        "polar": True, "voronoi": False, "checkpoint": False, "repulsive": False,
    }
    assert px.track_visible_sections("voronoi") == {
        "sampling": False, "smoothing": True, "bezier": False, "hull": False,
        "polar": False, "voronoi": True, "checkpoint": False, "repulsive": False,
    }
    assert px.track_visible_sections("checkpoint") == {
        "sampling": False, "smoothing": False, "bezier": False, "hull": False,
        "polar": False, "voronoi": False, "checkpoint": True, "repulsive": False,
    }
    assert px.track_visible_sections("repulsive") == {
        "sampling": False, "smoothing": False, "bezier": False, "hull": False,
        "polar": False, "voronoi": False, "checkpoint": False, "repulsive": True,
    }


def test_estimate_half_width_ignores_invalid_nan_envs():
    import torch
    # env 1 is invalid: its index-0 row is NaN (count==0 -> fully NaN-padded). A median over
    # all envs would be NaN-poisoned; restricting to valid envs must yield a finite width.
    outer = torch.zeros(3, 4, 2)
    center = torch.zeros(3, 4, 2)
    outer[:, 0, 0] = torch.tensor([1.0, float("nan"), 1.0])  # half-width 1.0 for valid envs
    center[:, 0, 0] = 0.0
    valid = torch.tensor([True, False, True])
    hw = px._estimate_half_width(outer, center, valid)
    assert torch.isfinite(torch.tensor(hw))
    assert abs(hw - 1.0) < 1e-6


def test_collect_raises_on_control_count_mismatch():
    with pytest.raises(ValueError, match="track controls"):
        px._collect(1, 2, 3)  # far fewer than the expected key count


# ---------------------------------------------------------------------------------------
# Schema-driven plugin controls (roadmap PR8): registry.describe(...).params_schema
# ---------------------------------------------------------------------------------------

import dataclasses  # noqa: E402

from track_gen import plugin  # noqa: E402
from track_gen.plugin import registry  # noqa: E402
from tests._plugin_toys import (TOY_CIRCLE, TOY_RING, ToyWobble,  # noqa: E402
                                register_stage_toys)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RingParams(plugin.Params):
    gates: int = plugin.static(8, lo=3, hi=16, doc="ring gates")


class _ParamRing(TOY_RING):
    name = "toy.ring_p"
    Params = _RingParams


def _wobble_schema():
    registry.add(ToyWobble)
    return registry.describe("generator", "toy.wobble").params_schema


def _track_vals(**over):
    """Fixed Tracks-tab values in TRACK_CONTROL_KEYS order (the explorer defaults)."""
    p = px.default_params()
    p.update(grid_n=2, batch_size=4)
    p.update(over)
    return [p[k] for k in px.TRACK_CONTROL_KEYS]


def test_schema_controls_toy_wobble_sliders():
    specs = {s.field: s for s in px.schema_controls(_wobble_schema())}
    assert list(specs) == ["lobes", "amp"]
    lobes, amp = specs["lobes"], specs["amp"]
    assert (lobes.widget, lobes.minimum, lobes.maximum, lobes.step) == ("slider", 2, 12, 1)
    assert lobes.value == 5 and lobes.label == "lobes" and not lobes.dynamic
    assert (amp.widget, amp.minimum, amp.maximum) == ("slider", 0.0, 0.4)
    assert amp.step == pytest.approx(0.004)                    # (max - min) / 100
    assert amp.value == pytest.approx(0.15)
    assert amp.label == "amp (dynamic)" and amp.dynamic
    assert lobes.cast(6.0) == 6 and isinstance(lobes.cast(6.0), int)
    assert amp.cast(0.25) == pytest.approx(0.25)


def test_schema_controls_widget_rules():
    schema = {"properties": {
        "mode": {"type": "string", "enum": ["a", "b"], "default": "b"},
        "flag": {"type": "boolean", "default": True},
        "gain": {"type": "number", "default": 0.5, "minimum": 0.0},
        "iters": {"type": "integer", "default": 3},
        "knots": {"type": "integer", "default": 6, "minimum": 4, "maximum": 20,
                  "x-ui": {"label": "knot count", "step": 2}},
        "stages": {"type": "array", "items": {"type": "integer"}, "default": [16, 32]},
        "span": {"type": ["array", "null"], "default": None},
        "tag": {"type": "string", "default": "x"},
        "level": {"type": ["number", "null"], "default": None},
    }}
    specs = {s.field: s for s in px.schema_controls(schema)}
    assert {k: s.widget for k, s in specs.items()} == {
        "mode": "dropdown", "flag": "checkbox", "gain": "number", "iters": "number",
        "knots": "slider", "stages": "textbox", "span": "textbox", "tag": "textbox",
        "level": "number"}
    assert specs["mode"].choices == ("a", "b") and specs["mode"].value == "b"
    assert specs["knots"].label == "knot count" and specs["knots"].step == 2
    assert specs["stages"].value == "[16, 32]"                 # a JSON literal
    # cast: the component value back to the field value
    assert specs["mode"].cast("a") == "a"
    with pytest.raises(ValueError, match="not one of"):
        specs["mode"].cast("c")
    assert specs["flag"].cast(0) is False
    assert specs["iters"].cast(4.0) == 4
    assert specs["stages"].cast("[16, 32, 64]") == [16, 32, 64]  # JSON-shaped
    assert specs["span"].cast("null") is None and specs["span"].cast("") is None
    assert specs["span"].cast("[0.1, 0.3]") == [0.1, 0.3]
    assert specs["tag"].cast('"y"') == "y" and specs["tag"].cast("bare") == "bare"
    assert specs["level"].cast(None) is None
    with pytest.raises(ValueError, match="JSON literal"):
        specs["stages"].cast("[16, ")


@pytest.mark.parametrize("kind", ["generator", "gate_generator"])
def test_schema_controls_round_trip_builtin_params(kind):
    # The generic path handles every built-in schema too: the controls' initial values,
    # resolved as generator_params, give the Params defaults.
    from track_gen.plugin._registry import _params_class
    from track_gen.plugin.params import _coerce

    for name in registry.available(kind):
        info = registry.describe(kind, name)
        # GeneratorSpec.params, or a Stage's Params (the gate built-ins since PR11)
        params_cls = _params_class(registry.get(kind, name))
        specs = px.schema_controls(info.params_schema)
        assert specs, name
        values = {s.field: s.cast(s.value) for s in specs}
        assert _coerce(params_cls, values, what=name) == params_cls(), name


def test_plugin_generators_lists_non_builtins_with_fields():
    assert px.plugin_generators("generator") == []          # built-ins keep curated layout
    assert px.plugin_generators("gate_generator") == []

    class _NoFields(ToyWobble):
        name = "toy.nofields"
        Params = plugin.NoParams

    register_stage_toys(registry)
    registry.add(TOY_CIRCLE)                                 # legacy spec, no params=
    registry.add(_NoFields)                                  # Params without fields
    registry.add(TOY_RING)
    registry.add(_ParamRing)
    assert px.plugin_generators("generator") == [
        "toy.lobed", "toy.wobble", "toy.wobble_host", "toy.wobble_input"]
    assert px.plugin_generators("gate_generator") == ["toy.ring_p"]


def test_plugin_section_visibility_and_keys():
    registry.add(ToyWobble)
    registry.add(_ParamRing)
    sections = px.plugin_sections("generator")
    assert [(n, [s.field for s in specs]) for n, specs in sections] == [
        ("toy.wobble", ["lobes", "amp"])]
    # header + one flag per control, visible only for the selected generator
    assert px.plugin_section_visibility("toy.wobble", sections) == [True, True, True]
    assert px.plugin_section_visibility("polar", sections) == [False, False, False]
    keys = px.plugin_control_keys(sections)
    assert [(g, s.field) for g, s in keys] == [("toy.wobble", "lobes"), ("toy.wobble", "amp")]
    gate_sections = px.plugin_sections("gate_generator")
    assert px.plugin_section_visibility("toy.ring_p", gate_sections) == [True, True]


def test_collect_maps_selected_plugin_values_to_generator_params():
    registry.add(ToyWobble)
    registry.add(dataclasses.replace(TOY_CIRCLE, name="toy.circle_p", params=_RingParams))
    keys = px.plugin_control_keys(px.plugin_sections("generator"))
    assert [(g, s.field) for g, s in keys] == [
        ("toy.circle_p", "gates"), ("toy.wobble", "lobes"), ("toy.wobble", "amp")]
    extra = [9.0, 7.0, 0.2]
    p = px._collect(*_track_vals(generator="toy.wobble"), *extra, plugin_keys=keys)
    assert p["generator"] == "toy.wobble"
    assert p["generator_params"] == {"lobes": 7, "amp": pytest.approx(0.2)}
    assert isinstance(p["generator_params"]["lobes"], int)
    # a built-in keeps its flat fields: no generator_params
    p = px._collect(*_track_vals(generator="polar"), *extra, plugin_keys=keys)
    assert p["generator_params"] is None
    # the fixed controls map as before
    assert p["polar_num_knots"] == px.default_params()["polar_num_knots"]
    with pytest.raises(ValueError, match="track controls"):
        px._collect(*_track_vals(), plugin_keys=keys)        # plugin values missing


def test_build_config_passes_generator_params():
    cfg = px.build_config(_params(generator="toy.wobble",
                                  generator_params={"lobes": 4, "amp": 0.1}))
    assert cfg.generator == "toy.wobble"
    assert cfg.generator_params == {"lobes": 4, "amp": 0.1}
    assert px.build_config(_params(generator="polar")).generator_params is None


def _plugin_track(**gp):
    registry.add(ToyWobble)
    track = px.generate_batch(_params(generator="toy.wobble", batch_size=4, grid_n=2,
                                      generator_params=gp))
    import warp as wp
    return wp.to_torch(track.center).clone(), wp.to_torch(track.valid).clone()


def test_generate_batch_runs_plugin_stage_with_its_params():
    center, valid = _plugin_track(lobes=4, amp=0.0)
    assert valid.shape == (4,) and bool(valid.all())
    center2, _ = _plugin_track(lobes=4, amp=0.3)
    assert not torch.equal(torch.nan_to_num(center), torch.nan_to_num(center2)), \
        "the dynamic amp value from generator_params must reach the stage"


def test_render_grid_reports_plugin_param_errors():
    registry.add(ToyWobble)
    fig, stats = px.render_grid(_params(generator="toy.wobble", batch_size=4, grid_n=2,
                                        generator_params={"lobes": 99}))
    assert "lobes" in stats["error"]                          # lo/hi rule, at construction
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_gate_plugin_controls_reach_gate_config():
    registry.add(_ParamRing)
    keys = px.plugin_control_keys(px.plugin_sections("gate_generator"))
    base = _gate_params(gate_generator="toy.ring_p", gate_ordering="ccw")
    vals = [base[k] for k in px.GATE_CONTROL_KEYS]
    p = px._collect_gate(*vals, 6.0, plugin_keys=keys)
    assert p["gate_generator_params"] == {"gates": 6}
    cfg = px.build_gate_config(p)
    assert cfg.generator == "toy.ring_p" and cfg.generator_params == {"gates": 6}
    assert cfg.gate_ordering == "raw"                         # clamped to supported
    assert (cfg.min_gates, cfg.max_gates) == (2, max(2, cfg.max_num_points))  # fallback
    gates, used = px.generate_gate_batch(p)
    assert used.generator == "toy.ring_p"
    import warp as wp
    assert wp.to_torch(gates.count).tolist() == [8] * 4
    p_builtin = px._collect_gate(*[_gate_params()[k] for k in px.GATE_CONTROL_KEYS], 6.0,
                                 plugin_keys=keys)
    assert p_builtin["gate_generator_params"] is None
    with pytest.raises(ValueError, match="gate controls"):
        px._collect_gate(*vals, plugin_keys=keys)


def test_build_app_renders_plugin_controls_hidden():
    pytest.importorskip("gradio")
    registry.add(ToyWobble)
    app = px.build_app()
    # the default generator is polar: the toy's section exists but is hidden
    assert _visible_by_label(app, "lobes") is False
    assert _visible_by_label(app, "amp (dynamic)") is False
    markdown = {c.get("props", {}).get("value") for c in app.config["components"]
                if c.get("type") == "markdown"}
    assert "### `toy.wobble` parameters" in markdown
