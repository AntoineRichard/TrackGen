"""Smoke test for the README gate asset renderer."""
import pytest


@pytest.mark.slow
def test_render_gate_assets_writes_png(tmp_path):
    from viz.render_readme_assets import render_gate_assets

    path = render_gate_assets(output_dir=tmp_path)

    assert path.name == "readme-gate-strip.png"
    assert path.exists()
    assert path.stat().st_size > 1000  # a real, non-empty PNG


@pytest.mark.slow
def test_render_generator_panels_writes_pngs(tmp_path):
    from viz.render_readme_assets import render_generator_panels

    # Render the repulsive panel on a reduced budget (small stages/num_points) so this CPU
    # smoke test stays fast; the ~1000x-slower default config is exercised on CUDA elsewhere.
    # The committed PNGs are rendered with no overrides (full config), so this does not change
    # them. Other panels are untouched.
    paths = render_generator_panels(
        output_dir=tmp_path,
        config_overrides={"repulsive": dict(num_points=64,
                                            repulsive_stages=(16, 32, 64), N_max=384)},
    )
    names = {p.name for p in paths}
    assert names == {
        "generator-bezier.png", "generator-checkpoint.png", "generator-hull.png",
        "generator-polar.png", "generator-voronoi.png", "generator-repulsive.png",
    }
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000


@pytest.mark.slow
def test_render_utilities_overview_writes_png(tmp_path):
    from viz.render_utility_assets import render_utilities_overview

    path = render_utilities_overview(output_dir=tmp_path)

    assert path.name == "utilities-overview.png"
    assert path.exists()
    assert path.stat().st_size > 1000  # a real, non-empty PNG


@pytest.mark.slow
def test_render_course_assets_write_pngs(tmp_path):
    from viz.render_utility_assets import (render_checkpoints_overview,
                                           render_disc_collision,
                                           render_progress_tracking)

    names = {render_checkpoints_overview(output_dir=tmp_path).name,
             render_progress_tracking(output_dir=tmp_path).name,
             render_disc_collision(output_dir=tmp_path).name}
    assert names == {"checkpoints-overview.png", "progress-tracking.png",
                     "disc-collision.png"}
    for n in names:
        assert (tmp_path / n).stat().st_size > 1000
