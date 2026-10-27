"""The ablation figures consume current road and experimental centerline results."""
import pytest


@pytest.mark.parametrize("figure, filename", [
    ("figure1_rad_edgy", "fig1_bezier_rad_edgy.png"),
    ("figure3_bezier_vs_fourier", "fig3_bezier_vs_fourier.png"),
])
def test_ablation_figure_writes_an_image(tmp_path, monkeypatch, figure, filename):
    """Reject removed config fields and stale tensor assumptions through a real render."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    from viz import plot_ablations

    monkeypatch.setattr(plot_ablations, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(plot_ablations, "DEVICE", "cpu")
    monkeypatch.setattr(plot_ablations, "DPI", 35)
    assert getattr(plot_ablations, figure)() == str(tmp_path / filename)
    assert (tmp_path / filename).stat().st_size > 1000
