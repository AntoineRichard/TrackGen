from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SECTIONS = [
    "00-abstract.tex",
    "01-introduction.tex",
    "02-review-protocol.tex",
    "03-scope-definitions.tex",
    "04-adjacent-surveys.tex",
    "05-representations.tex",
    "06-generation-methods.tex",
    "07-domain-constraints.tex",
    "08-metrics.tex",
    "09-benchmark-protocol.tex",
    "10-reference-implementations.tex",
    "11-reporting-practices.tex",
    "12-open-problems.tex",
    "13-conclusion.tex",
]


def test_paper_scaffold_is_complete():
    required = [
        "README.md",
        "Makefile",
        "latexmkrc",
        "main.tex",
        "preamble.tex",
        "macros.tex",
        "references.bib",
    ]
    assert all((PAPER / name).is_file() for name in required)
    assert all((PAPER / "sections" / name).is_file() for name in SECTIONS)


def test_main_includes_every_section_once():
    text = (PAPER / "main.tex").read_text()
    for section in SECTIONS:
        stem = section.removesuffix(".tex")
        assert text.count(rf"\input{{sections/{stem}}}") == 1


def test_sources_contain_no_unresolved_markers():
    forbidden = ("TO" + "DO", "T" + "BD", "FIX" + "ME", "CITATION " + "NEEDED")
    sources = list(PAPER.rglob("*.tex")) + list(PAPER.rglob("*.bib"))
    for source in sources:
        text = source.read_text()
        assert not any(marker in text for marker in forbidden), source
