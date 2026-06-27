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
    assert cfg.output_mode == "constant_spacing"  # the only supported mode
    assert abs(cfg.half_width - 0.5) < 1e-9 and abs(cfg.scale - 10.0) < 1e-9
    assert abs(cfg.handle_clamp_frac - 0.10) < 1e-9  # the overshoot-clamp knob round-trips
    cfg2 = px.build_config(_params(spacing=0.3, n_max=384))
    assert cfg2.output_mode == "constant_spacing" and cfg2.N_max == 384
    assert abs(cfg2.spacing - 0.3) < 1e-9


def test_build_config_auto_spacing():
    # spacing omitted -> config auto-couples it to 0.6*half_width (constant-spacing default).
    p = _params(half_width=0.5)
    p["spacing"] = None
    cfg = px.build_config(p)
    assert cfg.output_mode == "constant_spacing"
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
    center_t = wp.to_torch(track.center).view(20, 384, 2)
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
    assert cfg.num_points_per_segment == d.num_points_per_segment
    assert cfg.rad == d.rad
    assert cfg.edgy == d.edgy
    assert cfg.handle_clamp_frac == d.handle_clamp_frac
    assert cfg.hull_displacement == d.hull_displacement


def test_track_mode_visibility_orders_flags_to_match_outputs():
    # The generator.change handler emits these flags positionally onto
    # track_mode_outputs, so order + count must be exact. Checking non-default
    # generators (the default build is polar) catches section/count drift that a
    # polar-only build-time assertion cannot.
    assert px.track_mode_visibility("bezier") == (
        [True] * 4 + [True] * 2 + [True] * 4 + [False] * 2
        + [False] * 4 + [False] * 6 + [False] * 9
    )
    assert px.track_mode_visibility("hull") == (
        [True] * 4 + [True] * 2 + [False] * 4 + [True] * 2
        + [False] * 4 + [False] * 6 + [False] * 9
    )
    assert px.track_mode_visibility("voronoi") == (
        [False] * 4 + [True] * 2 + [False] * 4 + [False] * 2
        + [False] * 4 + [True] * 6 + [False] * 9
    )
    assert px.track_mode_visibility("checkpoint") == (
        [False] * (4 + 2 + 4 + 2 + 4 + 6) + [True] * 9
    )
    # Every generator yields the same fixed number of flags (one per output component).
    for gen in ("bezier", "hull", "polar", "voronoi", "checkpoint"):
        assert len(px.track_mode_visibility(gen)) == 31


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
