"""Parameter-explorer core (build_config + render_grid) on CPU, plus a gradio app-builds smoke.

Output is always constant_spacing (the only supported mode): per-track boundary arrays are
[E, N_max, 2] NaN-padded with a per-env real-point count in ``track.count`` that VARIES per
env. Stats/assertions are count-aware (mean count over valid tracks; never count == N_max).
"""
import pytest
import torch

from viz import param_explorer as px


def _params(**over):
    p = dict(half_width=0.5, scale=10.0, min_num_points=9, max_num_points=13,
             min_point_distance=0.05, num_points_per_segment=30, hull_displacement=0.15,
             rad=0.2, edgy=0.0, handle_clamp_frac=0.10, num_points=256,
             spacing=0.30, n_max=384, relax_iters=40, relax_sep_relax=1.0,
             relax_spc_relax=1.0, relax_bend_relax=1.5, relax_margin=0.15,
             grid_n=3, seed=0, batch_size=16)
    p.update(over)
    return p


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
