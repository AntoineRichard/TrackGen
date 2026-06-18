# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Parameter-explorer core (build_config + render_grid) on CPU, plus a gradio app-builds smoke."""
import pytest

from viz import param_explorer as px


def _params(**over):
    p = dict(half_width=0.5, scale=10.0, min_num_points=9, max_num_points=13, rad=0.2, edgy=0.0,
             output_mode="fixed", num_points=256, spacing=0.30, n_max=384,
             relax_iters=40, max_regen_iters=3, relax_sep_relax=1.0, relax_spc_relax=1.0,
             relax_bend_relax=1.5, relax_margin=0.15, grid_n=3, seed=0, batch_size=16)
    p.update(over)
    return p


def test_build_config_maps_and_clamps():
    cfg = px.build_config(_params(min_num_points=15, max_num_points=8))
    assert cfg.min_num_points <= cfg.max_num_points
    assert cfg.num_envs == 16  # batch_size=16 from _params default
    assert cfg.output_mode == "fixed"
    assert abs(cfg.half_width - 0.5) < 1e-9 and abs(cfg.scale - 10.0) < 1e-9
    cfg2 = px.build_config(_params(output_mode="constant_spacing", spacing=0.3, n_max=384))
    assert cfg2.output_mode == "constant_spacing" and cfg2.N_max == 384


import matplotlib.figure


def test_render_grid_fixed():
    fig, stats = px.render_grid(_params(grid_n=3, output_mode="fixed", num_points=128))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 9                       # grid_n**2 cells
    assert 0.0 <= stats["yield"] <= 1.0
    assert stats["count"] == 128                    # fixed mode: count == num_points
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_grid_constant_spacing_runs():
    fig, stats = px.render_grid(_params(grid_n=3, output_mode="constant_spacing",
                                        spacing=0.30, n_max=384))
    assert 0.0 <= stats["yield"] <= 1.0
    assert stats["count"] >= 1                       # variable per-track count (mean over valid)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_build_app_smoke():
    gr = pytest.importorskip("gradio")            # skip if the ui extra isn't installed
    app = px.build_app()
    assert isinstance(app, gr.Blocks)


def test_batch_and_pagination():
    p = _params(grid_n=3, batch_size=20, output_mode="fixed", num_points=128)
    track = px.generate_batch(p)
    assert track.center.shape[0] == 20                 # full batch generated
    import matplotlib.figure
    f0 = px.render_page(track, 0, 3); assert isinstance(f0, matplotlib.figure.Figure) and len(f0.axes) == 9
    f1 = px.render_page(track, 1, 3); assert len(f1.axes) == 9   # page 2 (start=9)
    import matplotlib.pyplot as plt; plt.close(f0); plt.close(f1)
    st = px._stats(track, "fixed", 128)
    assert 0.0 <= st["yield"] <= 1.0                   # stats over all 20, not 9
    assert px.n_pages(20, 3) == 3                       # ceil(20/9)
