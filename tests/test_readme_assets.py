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

    paths = render_generator_panels(output_dir=tmp_path)
    names = {p.name for p in paths}
    assert names == {
        "generator-bezier.png", "generator-checkpoint.png", "generator-hull.png",
        "generator-polar.png", "generator-voronoi.png",
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
