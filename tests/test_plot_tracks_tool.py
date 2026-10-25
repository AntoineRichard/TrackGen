"""The maintained plot entry point uses the current configuration surface."""
import pytest


def test_small_plot_grid_uses_supported_config_fields(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from viz import plot_tracks
    monkeypatch.setattr(plot_tracks, "OUT_DIR", str(tmp_path))
    paths=plot_tracks.render(images=1,rows=1,cols=1,device="cpu",seed=42,dpi=40)
    assert len(paths)==1
    assert (tmp_path / "tracks_grid_00.png").stat().st_size > 1000
