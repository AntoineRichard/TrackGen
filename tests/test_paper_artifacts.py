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

SECTION_CONTRACTS = {
    "00-abstract.tex": (r"\begin{abstract}", None),
    "01-introduction.tex": (r"\section{Introduction}", "sec:introduction"),
    "02-review-protocol.tex": (
        r"\section{Review Protocol}",
        "sec:review-protocol",
    ),
    "03-scope-definitions.tex": (
        r"\section{Scope, Definitions, and Boundaries}",
        "sec:scope-definitions",
    ),
    "04-adjacent-surveys.tex": (
        r"\section{Relationship to Existing Surveys}",
        "sec:adjacent-surveys",
    ),
    "05-representations.tex": (
        r"\section{Course Representations}",
        "sec:representations",
    ),
    "06-generation-methods.tex": (
        r"\section{Generation Methods}",
        "sec:generation-methods",
    ),
    "07-domain-constraints.tex": (
        r"\section{Domain Constraints}",
        "sec:domain-constraints",
    ),
    "08-metrics.tex": (
        r"\section{Metrics and Reporting Protocol}",
        "sec:metrics",
    ),
    "09-benchmark-protocol.tex": (
        r"\section{Benchmark Protocol}",
        "sec:benchmark-protocol",
    ),
    "10-reference-implementations.tex": (
        r"\section{Open Reference Implementations}",
        "sec:reference-implementations",
    ),
    "11-reporting-practices.tex": (
        r"\section{Reporting Practices in RL and Control}",
        "sec:reporting-practices",
    ),
    "12-open-problems.tex": (r"\section{Open Problems}", "sec:open-problems"),
    "13-conclusion.tex": (r"\section{Conclusion}", "sec:conclusion"),
}
REQUIRED_SUBSECTIONS = {
    "07-domain-constraints.tex": (
        r"\subsection{Ground}",
        r"\subsection{Aerial}",
        r"\subsection{Maritime}",
        r"\subsection{Cross-Domain}",
    ),
    "08-metrics.tex": (
        r"\subsection{Feasibility}",
        r"\subsection{Geometry}",
        r"\subsection{Difficulty}",
        r"\subsection{Diversity}",
        r"\subsection{Reproducibility}",
        r"\subsection{Simulation Feasibility}",
    ),
}


def test_paper_scaffold_is_complete():
    required = [
        ".gitignore",
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


def test_section_headings_and_labels_match_contract():
    assert list(SECTION_CONTRACTS) == SECTIONS
    for section, (heading, label) in SECTION_CONTRACTS.items():
        text = (PAPER / "sections" / section).read_text()
        lines = text.splitlines()
        assert lines[0] == heading, section
        assert text.count(heading) == 1, section

        if label is None:
            assert not any(line.startswith(r"\label{sec:") for line in lines), section
            continue

        expected_label = rf"\label{{{label}}}"
        assert lines[1] == expected_label, section
        assert text.count(expected_label) == 1, section
        assert text.count(r"\label{sec:") == 1, section


def test_required_subsections_match_contract():
    for section, expected in REQUIRED_SUBSECTIONS.items():
        text = (PAPER / "sections" / section).read_text()
        actual = tuple(
            line for line in text.splitlines() if line.startswith(r"\subsection{")
        )
        assert actual == expected, section


def test_sources_contain_no_unresolved_markers():
    forbidden = ("TO" + "DO", "T" + "BD", "FIX" + "ME", "CITATION " + "NEEDED")
    sources = list(PAPER.rglob("*.tex")) + list(PAPER.rglob("*.bib"))
    for source in sources:
        text = source.read_text().casefold()
        assert not any(marker.casefold() in text for marker in forbidden), source
