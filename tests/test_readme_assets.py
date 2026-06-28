"""Smoke test for the README gate asset renderer."""
import pytest


@pytest.mark.slow
def test_render_gate_assets_writes_png(tmp_path):
    from viz.render_readme_assets import render_gate_assets

    path = render_gate_assets(output_dir=tmp_path)

    assert path.name == "readme-gate-strip.png"
    assert path.exists()
    assert path.stat().st_size > 1000  # a real, non-empty PNG
