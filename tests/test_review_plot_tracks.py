"""Smoke coverage for the supported track-grid rendering entry point."""
import pytest


def test_single_track_grid_writes_an_image(tmp_path, monkeypatch):
    """Removed config fields and scalar subplot axes must not break a render."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    from viz import plot_tracks

    monkeypatch.setattr(plot_tracks, "OUT_DIR", str(tmp_path))
    paths = plot_tracks.render(
        images=1, rows=1, cols=1, device="cpu", seed=42, dpi=40,
    )
    assert paths == [str(tmp_path / "tracks_grid_00.png")]
    assert (tmp_path / "tracks_grid_00.png").stat().st_size > 1000
