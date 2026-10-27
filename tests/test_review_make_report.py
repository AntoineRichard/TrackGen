"""The yield report remains executable with the current Warp result layout."""
import pytest


def test_report_builds_a_pdf_on_cpu(tmp_path, monkeypatch):
    """A real report render must consume per-env geometry and produce a complete PDF."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    from viz import make_report

    path = tmp_path / "report.pdf"
    monkeypatch.setattr(make_report, "OUT_PDF", str(path))
    monkeypatch.setattr(make_report, "DEVICE", "cpu")
    make_report.main()
    result = path.read_bytes()
    assert result.startswith(b"%PDF-")
    assert result.rstrip().endswith(b"%%EOF")
    assert len(result) > 10000


def test_constant_spacing_caption_fits_the_pdf_page(tmp_path, monkeypatch):
    """The narrow two-column report page must keep its full caption visible."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    from matplotlib.backends.backend_pdf import PdfPages
    from viz import make_report

    monkeypatch.setattr(make_report, "DEVICE", "cpu")
    with PdfPages(tmp_path / "caption.pdf") as pdf:
        class CheckedPages:
            def savefig(self, figure):
                figure.canvas.draw()
                bounds = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
                assert bounds.x0 >= 0
                assert bounds.x1 <= figure.bbox.x1
                pdf.savefig(figure)

        make_report.page_constant_spacing(CheckedPages())
