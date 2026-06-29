# Robot-Racing Course Generation Survey Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible evidence corpus and a compilable, evidence-backed LaTeX survey manuscript that defines the scope, taxonomy, metrics, benchmark protocol, and research agenda for track, gate, and course generation in robot racing.

**Architecture:** Treat structured research data as the source of truth: independent literature searches feed a reconciled candidate ledger, screened papers feed a coded evidence matrix, and scripts render manuscript tables from that matrix. The manuscript remains modular by section, while decision records capture taxonomy, difficulty-spectrum, benchmark, and simulator-feasibility choices before those choices become claims. TrackGen's current implementation and related-work notes seed part of the search and provide reference evidence, but they do not bound the corpus.

**Tech Stack:** LaTeX with latexmk, pdfLaTeX, BibTeX and natbib; Python 3.10 standard library; pytest; CSV and JSON research artifacts; TikZ/PGFPlots or Matplotlib for reproducible figures; Crossref, publisher pages, proceedings archives, and official project repositories for metadata verification.

---

## Scope And Phase Boundary

This plan produces the survey foundation and manuscript. It includes:

- a reproducible search and screening protocol;
- blind and corpus-aware literature discovery;
- a deduplicated, verified bibliography and coded evidence matrix;
- the representation and generator taxonomies;
- a candidate metric catalog and reporting checklist;
- a difficulty-spectrum decision based on evidence rather than fixed labels;
- a benchmark selection specification;
- a simulator/export compatibility analysis;
- a compilable survey manuscript with generated tables and figures;
- explicit specifications for the next benchmark-data and simulator-adapter projects.

This plan does not freeze release-scale training distributions, generate the final
10,000-100,000-course artifacts, implement new generator families, implement production
simulator adapters, or benchmark RL/control policies. Those are independent engineering
projects whose interfaces depend on the survey decisions produced here. After the metric
and export decisions are reviewed, create separate specs and implementation plans for:

1. the course-corpus and evaluation-suite release; and
2. stable serialization plus simulator/RL-framework adapters.

### Execution Granularity

The checklist names scientific milestones. During execution, expand every repeated loop
into one tracked action per query, candidate metadata check, full-text coding row, claim,
table, or review finding. Those are the 2-5 minute work units; never mark a milestone
complete from a partially reviewed batch.


## File Structure

Create or modify these files:

- Create paper/README.md: build instructions, corpus workflow, and artifact status.
- Create paper/.gitignore: ignore LaTeX build products while retaining source artifacts.
- Create paper/Makefile: deterministic validation, table generation, PDF build, and cleanup.
- Create paper/latexmkrc: isolated build directory and strict pdfLaTeX settings.
- Create paper/main.tex: manuscript entry point and section order.
- Create paper/preamble.tex: packages, bibliography style, colors, and cross-reference setup.
- Create paper/macros.tex: stable terminology macros used throughout the manuscript.
- Create paper/references.bib: verified bibliography entries only.
- Create paper/sections/00-abstract.tex through paper/sections/13-conclusion.tex: one responsibility per manuscript section.
- Create paper/data/README.md: field definitions, coding rules, and evidence conventions.
- Create paper/data/taxonomy.json: controlled vocabulary for coded fields.
- Create paper/data/search_queries.csv: frozen query families and search streams.
- Create paper/data/search_log.csv: one row for each executed search.
- Create paper/data/candidates.csv: discovery, screening, and metadata status for every source.
- Create paper/data/seed_coverage.csv: coverage of the existing TrackGen notes.
- Create paper/data/evidence.csv: normalized coding for every included source.
- Create paper/data/coding_primary.csv: deterministic reliability sample copied from the primary coding.
- Create paper/data/coding_reliability.csv: independent second coding of the same sample.
- Create paper/data/coding_reliability_summary.csv: generated agreement and kappa results.
- Create paper/data/claims.csv: manuscript claim-to-source ledger.
- Create paper/data/metrics.csv: candidate and recommended course-generation metrics.
- Create paper/data/simulators.csv: simulator and export compatibility evidence.
- Create paper/data/conflicts.csv: unresolved merge or coding conflicts.
- Create paper/data/agent_runs/*.csv: immutable outputs from independent discovery agents.
- Create paper/data/agent_runs/*.md: search logs, saturation evidence, and scope observations for each agent.
- Create paper/notes/survey-structure.md: structural analysis of high-impact IJRR and Science Robotics surveys.
- Create paper/decisions/0001-scope-and-unit-of-analysis.md through 0005-simulator-feasibility.md: reviewed scientific decisions.
- Create paper/scripts/validate_corpus.py: schema, referential-integrity, and bibliography checks.
- Create paper/scripts/coding_reliability.py: deterministic stratified sampling and inter-coder agreement.
- Create paper/scripts/merge_candidates.py: deterministic DOI/title deduplication.
- Create paper/scripts/render_tables.py: deterministic LaTeX table generation.
- Create paper/scripts/render_figure_data.py: generated corpus-count macros for reproducible figures.
- Create paper/scripts/check_tex_log.py: fail on unresolved citations, references, and labels.
- Create paper/tables/*.tex: generated comparison tables checked into version control.
- Create paper/figures/taxonomy.tex: source for the taxonomy figure.
- Create paper/figures/corpus-flow.tex: source for the search and screening flow.
- Create paper/figures/corpus-counts.tex: generated LaTeX macros for corpus-flow counts and search date.
- Create paper/figures/benchmark-pipeline.tex: source for benchmark selection.
- Create paper/reviews/citation-audit.md, scope-audit.md, and manuscript-audit.md: independent review findings and resolutions.
- Create tests/test_paper_artifacts.py: project-layout and source hygiene tests.
- Create tests/test_survey_corpus.py: validator and merge behavior tests.
- Create tests/test_survey_tables.py: deterministic table-rendering tests.
- Modify docs/related-work/state-of-the-art.rst: point readers to the structured survey corpus after reconciliation.
- Modify docs/related-work/prior-art.rst: distinguish the historical seed note from the reviewed survey corpus.
- Modify docs/index.rst: add the survey methodology/corpus entry after the artifact is usable.

## Task 1: Install LaTeX And Establish A Strict Paper Build

**Files:**

- Create: paper/.gitignore
- Create: paper/README.md
- Create: paper/Makefile
- Create: paper/latexmkrc
- Create: paper/main.tex
- Create: paper/preamble.tex
- Create: paper/macros.tex
- Create: paper/references.bib
- Create: paper/sections/00-abstract.tex
- Create: paper/sections/01-introduction.tex
- Create: paper/sections/02-review-protocol.tex
- Create: paper/sections/03-scope-definitions.tex
- Create: paper/sections/04-adjacent-surveys.tex
- Create: paper/sections/05-representations.tex
- Create: paper/sections/06-generation-methods.tex
- Create: paper/sections/07-domain-constraints.tex
- Create: paper/sections/08-metrics.tex
- Create: paper/sections/09-benchmark-protocol.tex
- Create: paper/sections/10-reference-implementations.tex
- Create: paper/sections/11-reporting-practices.tex
- Create: paper/sections/12-open-problems.tex
- Create: paper/sections/13-conclusion.tex
- Create: tests/test_paper_artifacts.py

- [ ] **Step 1: Install the smallest complete LaTeX toolchain**




Run:

~~~bash
sudo apt-get update
sudo apt-get install -y latexmk texlive-latex-base texlive-latex-recommended \
  texlive-latex-extra texlive-fonts-recommended texlive-pictures chktex
latexmk --version
pdflatex --version
bibtex --version
~~~

Expected: all three version commands exit 0. Do not install a journal class yet; keep the
content independent from venue formatting until IJRR versus Science Robotics is chosen.

- [ ] **Step 2: Write the failing artifact-layout test**

Create tests/test_paper_artifacts.py:

~~~python
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
~~~

- [ ] **Step 3: Run the artifact test and confirm the expected failure**




Run:

~~~bash
.venv/bin/python -m pytest -q tests/test_paper_artifacts.py
~~~

Expected: failure because paper/main.tex and the section files do not exist.

- [ ] **Step 4: Create the reproducible build files**

Create paper/.gitignore:

~~~gitignore
build/
*.synctex.gz
~~~

Create paper/latexmkrc:

~~~perl
$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex = 'bibtex %O %B';
~~~

Create paper/Makefile:

~~~make
PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all validate tables pdf lint check clean

all: check

validate:
	$(PYTHON) scripts/validate_corpus.py

tables: validate
	$(PYTHON) scripts/render_tables.py

pdf: tables
	$(LATEXMK) -pdf main.tex
	$(PYTHON) scripts/check_tex_log.py build/main.log

lint:
	chktex -q -n 8 -n 13 -n 24 main.tex

check: validate tables pdf lint

clean:
	$(LATEXMK) -C main.tex
~~~

Create paper/preamble.tex:

~~~tex
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,matrix}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{hyperref}
\usepackage[nameinlink,noabbrev]{cleveref}

\definecolor{groundcolor}{HTML}{2E6F9E}
\definecolor{aircolor}{HTML}{A45136}
\definecolor{watercolor}{HTML}{2B7A68}
\definecolor{neutralcolor}{HTML}{555555}

\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\emergencystretch}{2em}
\hypersetup{
  colorlinks=true,
  linkcolor=neutralcolor,
  citecolor=groundcolor,
  urlcolor=watercolor
}
~~~

Create paper/macros.tex:

~~~tex
\newcommand{\TrackGen}{\textsc{TrackGen}}
\newcommand{\CourseGen}{course generation}
\newcommand{\NotReported}{NR}
\newcommand{\OSS}{open-source software}
\newcommand{\simfeas}{simulation feasibility}
~~~

Create paper/main.tex:

~~~tex
\documentclass[10pt]{article}
\input{preamble}
\input{macros}

\title{Track, Gate, and Course Generation for Robot Racing:\\
Representations, Methods, Metrics, and Benchmarks}
\author{Author names withheld during drafting}
\date{}

\begin{document}
\maketitle
\input{sections/00-abstract}
\input{sections/01-introduction}
\input{sections/02-review-protocol}
\input{sections/03-scope-definitions}
\input{sections/04-adjacent-surveys}
\input{sections/05-representations}
\input{sections/06-generation-methods}
\input{sections/07-domain-constraints}
\input{sections/08-metrics}
\input{sections/09-benchmark-protocol}
\input{sections/10-reference-implementations}
\input{sections/11-reporting-practices}
\input{sections/12-open-problems}
\input{sections/13-conclusion}

\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
~~~

Create paper/references.bib as an empty file. It remains empty until Task 5 verifies the
first source.

- [ ] **Step 5: Create section contracts that compile without unsupported claims**

Use these exact headings and opening contracts:

| File | Heading and initial contract |
| --- | --- |
| 00-abstract.tex | An unnumbered abstract stating that the article studies generation of robot-racing courses across ground, aerial, and maritime domains. Do not include corpus counts before screening closes. |
| 01-introduction.tex | Section “Introduction”; explain why course distributions affect generalization, safety, reproducibility, and comparison. |
| 02-review-protocol.tex | Section “Review Protocol”; define discovery streams, screening, coding, and metadata verification. |
| 03-scope-definitions.tex | Section “Scope, Definitions, and Boundaries”; define course, representation, generator, training distribution, and evaluation suite. |
| 04-adjacent-surveys.tex | Section “Relationship to Existing Surveys”; reserve the comparison for generated table adjacent-surveys.tex. |
| 05-representations.tex | Section “Course Representations”; organize representation before method. |
| 06-generation-methods.tex | Section “Generation Methods”; distinguish synthesis, selection, mutation, replay, and repair. |
| 07-domain-constraints.tex | Section “Domain Constraints”; use separate ground, aerial, maritime, and cross-domain subsections. |
| 08-metrics.tex | Section “Metrics and Reporting Protocol”; separate feasibility, geometry, difficulty, diversity, reproducibility, and simulation feasibility. |
| 09-benchmark-protocol.tex | Section “Benchmark Protocol”; define training distributions and spectrum-based evaluation suites without hard-coding easy/medium/hard labels. |
| 10-reference-implementations.tex | Section “Open Reference Implementations”; map implementations to taxonomy coverage and state their limitations. |
| 11-reporting-practices.tex | Section “Reporting Practices in RL and Control”; audit what downstream papers disclose about course generation. |
| 12-open-problems.tex | Section “Open Problems”; organize gaps by scientific consequence and tractability. |
| 13-conclusion.tex | Section “Conclusion”; restate the minimum reporting and benchmark recommendations. |

Each file must contain its heading, the contract as a short declarative paragraph, and
the stable labels sec:introduction through sec:conclusion. The abstract file uses:

~~~tex
\begin{abstract}
Robot-racing research depends on tracks, gates, waypoints, and buoy courses, yet the
generation and reporting of these task distributions remain fragmented across vehicle
domains and research communities. This survey develops a common vocabulary for course
representations, generation methods, evaluation metrics, benchmark construction, and
simulation feasibility across ground, aerial, and maritime robots.
\end{abstract}
~~~

- [ ] **Step 6: Run the scaffold tests and a direct LaTeX build**




Run:

~~~bash
.venv/bin/python -m pytest -q tests/test_paper_artifacts.py
latexmk -cd -pdf paper/main.tex
~~~

Expected: tests pass and paper/build/main.pdf exists with no LaTeX build error.

- [ ] **Step 7: Commit the paper scaffold**

~~~bash
git add paper tests/test_paper_artifacts.py
git commit -m "docs: scaffold reproducible survey manuscript"
~~~

## Task 2: Define The Research Data Contract And Validator

**Files:**

- Create: paper/__init__.py
- Create: paper/data/README.md
- Create: paper/data/taxonomy.json
- Create: paper/data/search_log.csv
- Create: paper/data/candidates.csv
- Create: paper/data/seed_coverage.csv
- Create: paper/data/evidence.csv
- Create: paper/data/claims.csv
- Create: paper/data/metrics.csv
- Create: paper/data/simulators.csv
- Create: paper/data/conflicts.csv
- Create: paper/scripts/__init__.py
- Create: paper/scripts/validate_corpus.py
- Create: tests/test_survey_corpus.py

- [ ] **Step 1: Write validator tests for valid and invalid miniature corpora**

Create tests/test_survey_corpus.py with temporary CSV fixtures covering:

~~~python
import csv
import json
from pathlib import Path

import pytest

from paper.scripts.validate_corpus import (
    CorpusError,
    DEFAULT_TAXONOMY,
    HEADERS,
    validate_directory,
)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS[path.name])
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def blank_row(filename: str) -> dict[str, str]:
    return dict.fromkeys(HEADERS[filename], "")


def build_valid_fixture(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "taxonomy.json").write_text(
        json.dumps(DEFAULT_TAXONOMY, indent=2) + "\n"
    )

    candidate = blank_row("candidates.csv")
    candidate.update(
        candidate_id="C0001",
        cite_key="Sample2026Course",
        title="A Fictional Course Generator",
        authors="A. Author",
        year="2026",
        venue="Test Proceedings",
        doi="10.0000/example",
        url="https://example.invalid/paper",
        source_type="paper",
        discovery_stream="test",
        discovery_query="fictional fixture",
        discovery_agent="pytest",
        screening_status="included",
        metadata_status="verified",
        metadata_evidence="https://example.invalid/metadata",
    )
    evidence = blank_row("evidence.csv")
    evidence.update(
        cite_key="Sample2026Course",
        domain="ground",
        vehicle="car",
        course_object="closed_track",
        representation_family="parametric_curve",
        generator_family="stochastic_procedural",
        generation_role="geometry_synthesis",
        validity_strategy="rejection",
        code_status="not_found",
        evidence_locator="Section 3",
    )
    claim = blank_row("claims.csv")
    claim.update(
        claim_id="CL0001",
        section="introduction",
        claim_text="The fictional fixture creates courses.",
        cite_keys="Sample2026Course",
        evidence_status="direct",
    )
    search = blank_row("search_log.csv")
    search.update(
        search_id="S0001",
        search_date="2026-06-29",
        stream="test",
        agent="pytest",
        query="fictional fixture",
        search_surface="local",
        results_screened="1",
        candidates_added="1",
    )
    seed = blank_row("seed_coverage.csv")
    seed.update(
        source_path="fixture.rst",
        source_heading="Fixture",
        source_label="Fictional source",
        candidate_id="C0001",
        coverage_status="linked",
    )

    rows_by_file = {
        "search_log.csv": [search],
        "candidates.csv": [candidate],
        "seed_coverage.csv": [seed],
        "evidence.csv": [evidence],
        "claims.csv": [claim],
        "metrics.csv": [],
        "simulators.csv": [],
        "conflicts.csv": [],
    }
    for filename, rows in rows_by_file.items():
        write_rows(data / filename, rows)
    (tmp_path / "references.bib").write_text("")
    return data


def rewrite_rows(path: Path, rows: list[dict[str, str]]) -> None:
    write_rows(path, rows)


def test_included_source_requires_verified_metadata_and_evidence(tmp_path):
    validate_directory(build_valid_fixture(tmp_path))


def test_excluded_source_requires_reason(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "candidates.csv")
    rows[0]["screening_status"] = "excluded"
    rows[0]["exclusion_reason"] = ""
    rewrite_rows(fixture / "candidates.csv", rows)
    with pytest.raises(CorpusError, match="exclusion_reason"):
        validate_directory(fixture)


def test_evidence_must_reference_included_candidate(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "evidence.csv")
    rows[0]["cite_key"] = "missing2026"
    rewrite_rows(fixture / "evidence.csv", rows)
    with pytest.raises(CorpusError, match="missing2026"):
        validate_directory(fixture)


def test_duplicate_doi_is_rejected_after_normalization(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "candidates.csv")
    duplicate = dict(rows[0])
    duplicate["candidate_id"] = "C0002"
    duplicate["cite_key"] = "Sample2026CourseB"
    duplicate["doi"] = "https://doi.org/" + rows[0]["doi"].upper()
    rewrite_rows(fixture / "candidates.csv", rows + [duplicate])
    with pytest.raises(CorpusError, match="duplicate DOI"):
        validate_directory(fixture)
~~~


- [ ] **Step 2: Run the validator tests and confirm import failure**




Run:

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_corpus.py
~~~

Expected: collection fails because paper.scripts.validate_corpus does not exist.

- [ ] **Step 3: Create exact CSV schemas**

Use these headers:

~~~text
search_log.csv:
search_id,search_date,stream,agent,query,search_surface,results_screened,candidates_added,notes

candidates.csv:
candidate_id,cite_key,title,authors,year,venue,doi,url,source_type,discovery_stream,discovery_query,discovery_agent,screening_status,exclusion_reason,metadata_status,metadata_evidence

seed_coverage.csv:
source_path,source_heading,source_label,candidate_id,coverage_status,notes

evidence.csv:
cite_key,domain,vehicle,course_object,representation_family,generator_family,generation_role,validity_strategy,geometry_metrics,difficulty_metrics,diversity_metrics,training_distribution,evaluation_suite,simulator,export_format,code_status,asset_status,reproducibility_fields,evidence_locator,coding_notes

claims.csv:
claim_id,section,claim_text,cite_keys,evidence_status,reviewer_notes

metrics.csv:
metric_id,layer,name,definition,formula_or_procedure,units,direction,domain,requires_dynamics,minimum_reporting,cite_keys,limitations

simulators.csv:
system,cite_key,domain,input_representation,export_format,load_validation,coordinate_frame,units,collision_geometry,spawn_reset,rl_interface,oss_status,evidence_locator

conflicts.csv:
conflict_id,record_type,record_key,field,value_a,value_b,resolution,resolver,resolution_evidence
~~~

Store multiple values inside one cell with semicolons. Reserve the literal NR for facts
that the reviewed source does not report; never infer a negative from silence.

- [ ] **Step 4: Define controlled vocabulary in taxonomy.json**

Create JSON arrays with these initial values:

~~~json
{
  "domain": ["ground", "aerial", "maritime", "mixed", "adjacent"],
  "course_object": [
    "closed_track",
    "open_corridor",
    "gate_chain",
    "waypoint_sequence",
    "road_network",
    "buoy_course",
    "world_asset",
    "fixed_benchmark"
  ],
  "representation_family": [
    "segment_grammar",
    "tile_grid",
    "parametric_curve",
    "sampled_centerline",
    "centerline_plus_width",
    "boundary_pair",
    "gate_poses",
    "waypoint_graph",
    "occupancy_heightfield_mesh",
    "simulator_native",
    "hybrid"
  ],
  "generator_family": [
    "constructive",
    "stochastic_procedural",
    "search_evolutionary",
    "learned_generative",
    "environment_design",
    "human_designed",
    "repair_projection",
    "selection_replay"
  ],
  "generation_role": [
    "geometry_synthesis",
    "task_selection",
    "mutation",
    "repair",
    "serialization",
    "benchmark_only",
    "boundary_case"
  ],
  "validity_strategy": [
    "by_construction",
    "rejection",
    "penalty",
    "repair_projection",
    "constraint_solver",
    "simulation_validation",
    "not_reported"
  ],
  "screening_status": ["candidate", "included", "excluded", "boundary"],
  "metadata_status": ["unverified", "verified", "conflict"],
  "code_status": ["official_open", "unofficial_open", "closed", "not_found", "not_applicable"],
  "evidence_status": ["direct", "triangulated", "inferred", "unsupported"]
}
~~~

The codebook may split or merge values only through a recorded decision in Task 7.

- [ ] **Step 5: Implement validation**

Implement paper/scripts/validate_corpus.py with these public functions:

~~~python
from __future__ import annotations

import csv
import json
from pathlib import Path


class CorpusError(ValueError):
    pass


HEADERS = {
    "search_log.csv": (
        "search_id", "search_date", "stream", "agent", "query", "search_surface",
        "results_screened", "candidates_added", "notes",
    ),
    "candidates.csv": (
        "candidate_id", "cite_key", "title", "authors", "year", "venue", "doi",
        "url", "source_type", "discovery_stream", "discovery_query",
        "discovery_agent", "screening_status", "exclusion_reason",
        "metadata_status", "metadata_evidence",
    ),
    "seed_coverage.csv": (
        "source_path", "source_heading", "source_label", "candidate_id",
        "coverage_status", "notes",
    ),
    "evidence.csv": (
        "cite_key", "domain", "vehicle", "course_object", "representation_family",
        "generator_family", "generation_role", "validity_strategy",
        "geometry_metrics", "difficulty_metrics", "diversity_metrics",
        "training_distribution", "evaluation_suite", "simulator", "export_format",
        "code_status", "asset_status", "reproducibility_fields",
        "evidence_locator", "coding_notes",
    ),
    "claims.csv": (
        "claim_id", "section", "claim_text", "cite_keys", "evidence_status",
        "reviewer_notes",
    ),
    "metrics.csv": (
        "metric_id", "layer", "name", "definition", "formula_or_procedure", "units",
        "direction", "domain", "requires_dynamics", "minimum_reporting", "cite_keys",
        "limitations",
    ),
    "simulators.csv": (
        "system", "cite_key", "domain", "input_representation", "export_format",
        "load_validation", "coordinate_frame", "units", "collision_geometry",
        "spawn_reset", "rl_interface", "oss_status", "evidence_locator",
    ),
    "conflicts.csv": (
        "conflict_id", "record_type", "record_key", "field", "value_a", "value_b",
        "resolution", "resolver", "resolution_evidence",
    ),
}

DEFAULT_TAXONOMY = {
    "domain": ["ground", "aerial", "maritime", "mixed", "adjacent"],
    "course_object": [
        "closed_track", "open_corridor", "gate_chain", "waypoint_sequence",
        "road_network", "buoy_course", "world_asset", "fixed_benchmark",
    ],
    "representation_family": [
        "segment_grammar", "tile_grid", "parametric_curve", "sampled_centerline",
        "centerline_plus_width", "boundary_pair", "gate_poses", "waypoint_graph",
        "occupancy_heightfield_mesh", "simulator_native", "hybrid",
    ],
    "generator_family": [
        "constructive", "stochastic_procedural", "search_evolutionary",
        "learned_generative", "environment_design", "human_designed",
        "repair_projection", "selection_replay",
    ],
    "generation_role": [
        "geometry_synthesis", "task_selection", "mutation", "repair",
        "serialization", "benchmark_only", "boundary_case",
    ],
    "validity_strategy": [
        "by_construction", "rejection", "penalty", "repair_projection",
        "constraint_solver", "simulation_validation", "not_reported",
    ],
    "screening_status": ["candidate", "included", "excluded", "boundary"],
    "metadata_status": ["unverified", "verified", "conflict"],
    "code_status": [
        "official_open", "unofficial_open", "closed", "not_found",
        "not_applicable",
    ],
    "evidence_status": ["direct", "triangulated", "inferred", "unsupported"],
}

CONTROLLED_FIELDS = {
    "candidates.csv": {
        "screening_status": "screening_status",
        "metadata_status": "metadata_status",
    },
    "evidence.csv": {
        "domain": "domain",
        "course_object": "course_object",
        "representation_family": "representation_family",
        "generator_family": "generator_family",
        "generation_role": "generation_role",
        "validity_strategy": "validity_strategy",
        "code_status": "code_status",
    },
    "claims.csv": {"evidence_status": "evidence_status"},
}

FORBIDDEN_MARKERS = ("TO" + "DO", "T" + "BD", "FIX" + "ME", "CITATION " + "NEEDED")


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value.rstrip("/")


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise CorpusError(f"{path}: file is missing")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != required:
            raise CorpusError(f"{path}: headers {actual!r} != {required!r}")
        rows = list(reader)
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise CorpusError(f"{path}:{row_number}: malformed CSV row")
    return rows


def _require(
    filename: str,
    row_number: int,
    row: dict[str, str],
    field: str,
) -> str:
    value = row[field].strip()
    if not value:
        raise CorpusError(f"{filename}:{row_number}: {field} is required")
    return value


def _check_unique(
    filename: str,
    rows: list[dict[str, str]],
    field: str,
    label: str,
    normalizer=lambda value: value.strip(),
) -> None:
    seen: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        raw = row[field]
        if not raw.strip():
            continue
        value = normalizer(raw)
        if value in seen:
            raise CorpusError(
                f"{filename}:{row_number}: duplicate {label} {value!r}; "
                f"first seen on row {seen[value]}"
            )
        seen[value] = row_number


def _validate_controlled(
    filename: str,
    rows: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
) -> None:
    for field, vocabulary_name in CONTROLLED_FIELDS.get(filename, {}).items():
        allowed = set(taxonomy[vocabulary_name])
        for row_number, row in enumerate(rows, start=2):
            for value in split_values(row[field]):
                if value not in allowed:
                    raise CorpusError(
                        f"{filename}:{row_number}: {field}={value!r} "
                        f"is outside {vocabulary_name}"
                    )


def _validate_markers(filename: str, rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for field, value in row.items():
            for marker in FORBIDDEN_MARKERS:
                if marker in value.upper():
                    raise CorpusError(
                        f"{filename}:{row_number}: {field} contains {marker!r}"
                    )


def validate_directory(data_dir: Path) -> None:
    taxonomy_path = data_dir / "taxonomy.json"
    if not taxonomy_path.is_file():
        raise CorpusError(f"{taxonomy_path}: file is missing")
    taxonomy = json.loads(taxonomy_path.read_text())
    for name in DEFAULT_TAXONOMY:
        if name not in taxonomy or not isinstance(taxonomy[name], list):
            raise CorpusError(f"{taxonomy_path}: missing list {name!r}")
        if len(taxonomy[name]) != len(set(taxonomy[name])):
            raise CorpusError(f"{taxonomy_path}: duplicate value in {name!r}")

    tables = {
        filename: read_csv(data_dir / filename, header)
        for filename, header in HEADERS.items()
    }
    for filename, rows in tables.items():
        _validate_markers(filename, rows)
        _validate_controlled(filename, rows, taxonomy)

    candidates = tables["candidates.csv"]
    for row_number, row in enumerate(candidates, start=2):
        _require("candidates.csv", row_number, row, "candidate_id")
        _require("candidates.csv", row_number, row, "title")
        status = _require(
            "candidates.csv", row_number, row, "screening_status"
        )
        _require("candidates.csv", row_number, row, "metadata_status")
        if status in {"included", "boundary"}:
            _require("candidates.csv", row_number, row, "cite_key")
            if row["metadata_status"] != "verified":
                raise CorpusError(
                    f"candidates.csv:{row_number}: {status} source requires "
                    "metadata_status=verified"
                )
        if status == "excluded" and not row["exclusion_reason"].strip():
            raise CorpusError(
                f"candidates.csv:{row_number}: exclusion_reason is required"
            )

    _check_unique("candidates.csv", candidates, "candidate_id", "candidate_id")
    _check_unique("candidates.csv", candidates, "cite_key", "cite_key")
    _check_unique(
        "candidates.csv", candidates, "doi", "DOI", normalize_doi
    )
    _check_unique(
        "search_log.csv",
        tables["search_log.csv"],
        "search_id",
        "search_id",
    )
    _check_unique("claims.csv", tables["claims.csv"], "claim_id", "claim_id")
    _check_unique("metrics.csv", tables["metrics.csv"], "metric_id", "metric_id")
    _check_unique(
        "conflicts.csv",
        tables["conflicts.csv"],
        "conflict_id",
        "conflict_id",
    )

    by_id = {row["candidate_id"]: row for row in candidates}
    screened_keys = {
        row["cite_key"]
        for row in candidates
        if row["screening_status"] in {"included", "boundary"}
    }
    evidence_rows = tables["evidence.csv"]
    _check_unique("evidence.csv", evidence_rows, "cite_key", "cite_key")
    evidence_keys = {row["cite_key"] for row in evidence_rows}
    if evidence_keys != screened_keys:
        missing = sorted(screened_keys - evidence_keys)
        extra = sorted(evidence_keys - screened_keys)
        raise CorpusError(
            f"evidence.csv: cite_key mismatch; missing={missing}, extra={extra}"
        )

    for filename in ("claims.csv", "metrics.csv"):
        for row_number, row in enumerate(tables[filename], start=2):
            for cite_key in split_values(row["cite_keys"]):
                if cite_key not in screened_keys:
                    raise CorpusError(
                        f"{filename}:{row_number}: unknown cite_key {cite_key!r}"
                    )
    for row_number, row in enumerate(tables["simulators.csv"], start=2):
        cite_key = row["cite_key"].strip()
        if cite_key and cite_key not in screened_keys:
            raise CorpusError(
                f"simulators.csv:{row_number}: unknown cite_key {cite_key!r}"
            )

    for row_number, row in enumerate(tables["seed_coverage.csv"], start=2):
        status = _require(
            "seed_coverage.csv", row_number, row, "coverage_status"
        )
        if status not in {"unreviewed", "linked", "excluded"}:
            raise CorpusError(
                f"seed_coverage.csv:{row_number}: invalid coverage_status "
                f"{status!r}"
            )
        candidate_id = row["candidate_id"].strip()
        if status in {"linked", "excluded"} and candidate_id not in by_id:
            raise CorpusError(
                f"seed_coverage.csv:{row_number}: unknown candidate_id "
                f"{candidate_id!r}"
            )

    for row_number, row in enumerate(tables["conflicts.csv"], start=2):
        if row["resolution"].strip():
            _require("conflicts.csv", row_number, row, "resolver")
            _require("conflicts.csv", row_number, row, "resolution_evidence")


if __name__ == "__main__":
    validate_directory(Path(__file__).resolve().parents[1] / "data")
    print("survey corpus validation passed")
~~~


- [ ] **Step 6: Document coding semantics**

Create paper/data/README.md with:

- one subsection per CSV;
- the inclusion and exclusion rules from Task 6;
- the distinction between NR, not applicable, and a documented negative;
- DOI normalization rules;
- semicolon-list escaping rules;
- evidence locators formatted as page, section, figure, table, appendix, or official URL;
- the rule that code and asset status require an official repository or author page;
- the rule that candidates.csv records discovery while evidence.csv records verified coding.

- [ ] **Step 7: Run tests and validator**




Run:

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_corpus.py
python3 paper/scripts/validate_corpus.py
~~~

Expected: tests pass and the header-only production corpus validates.

- [ ] **Step 8: Commit the research data contract**

~~~bash
git add paper/data paper/scripts paper/__init__.py tests/test_survey_corpus.py
git commit -m "docs: define survey evidence data contract"
~~~

## Task 3: Freeze Search Streams And Bootstrap The Existing Corpus

**Files:**

- Create: paper/data/search_queries.csv
- Create: paper/notes/survey-structure.md
- Populate: paper/data/seed_coverage.csv
- Populate: paper/data/candidates.csv
- Populate: paper/data/search_log.csv

- [ ] **Step 1: Add the frozen query matrix**

Create paper/data/search_queries.csv with header:

~~~text
query_id,stream,domain,query,rationale
~~~

Add these query families as separate rows, preserving the quoted phrases:

~~~text
B-G-01,blind-ground,ground,"procedural race track generation robot reinforcement learning","Direct robot-racing geometry"
B-G-02,blind-ground,ground,"autonomous racing generated tracks benchmark","Racing benchmark distributions"
B-G-03,blind-ground,ground,"search based procedural track generation racing games","Search-based PCG lineage"
B-G-04,blind-ground,ground,"procedural road geometry generation autonomous vehicle testing","Adjacent road synthesis"
B-G-05,blind-ground,ground,"OpenDRIVE procedural road network generation","Portable road representations"
B-G-06,blind-ground,ground,"F1TENTH RoboRacer generated map track benchmark","Small-scale racing platforms"
B-AM-01,blind-aerial-maritime,aerial,"drone racing random gate course generation reinforcement learning","3D gate synthesis"
B-AM-02,blind-aerial-maritime,aerial,"quadrotor waypoint generator obstacle course curriculum","Aerial curriculum generation"
B-AM-03,blind-aerial-maritime,aerial,"drone racing track complexity metric gate visibility","Aerial difficulty metrics"
B-AM-04,blind-aerial-maritime,maritime,"autonomous surface vehicle buoy course generation","Maritime course synthesis"
B-AM-05,blind-aerial-maritime,maritime,"boat racing reinforcement learning waypoint course","Maritime RL courses"
B-AM-06,blind-aerial-maritime,maritime,"VRX RobotX buoy navigation benchmark course","Competition and simulator courses"
B-AM-07,blind-aerial-maritime,maritime,"USV simulator procedural environment generation","Maritime simulation"
B-AM-08,blind-aerial-maritime,mixed,"robot racing course generation benchmark","Cross-domain terminology"
A-G-01,aware-geometry-rl,ground,"papers citing procedural generation CarRacing UED Bezier tracks","Citation expansion from known RL tasks"
A-G-02,aware-geometry-rl,ground,"papers citing automatic track generation evolutionary computation","Citation expansion from racing PCG"
A-G-03,aware-geometry-rl,adjacent,"road scenario generation survey geometry validity","Adjacent survey coverage"
A-S-01,aware-simulation,mixed,"generated course export CARLA OpenDRIVE Isaac Sim AirSim Gazebo","Simulator portability"
A-S-02,aware-simulation,ground,"Gymnasium CarRacing track generator source format","RL interface evidence"
A-S-03,aware-simulation,aerial,"AirSim drone racing gate API track generation","Aerial simulator interface"
A-S-04,aware-simulation,maritime,"Gazebo VRX world buoy course format","Maritime simulator interface"
B-M-01,blind-ground,mixed,"legged mobile robot racing obstacle course generation","Adjacent agile-robot course design"
B-M-02,blind-aerial-maritime,mixed,"robot obstacle course procedural generation reinforcement learning","Cross-domain course distributions"
X-01,survey-exemplars,adjacent,"site:journals.sagepub.com IJRR survey robotics taxonomy benchmark","IJRR structural exemplars"
X-02,survey-exemplars,adjacent,"site:science.org/journal/scirobotics review robotics taxonomy benchmark","Science Robotics structural exemplars"
~~~

- [ ] **Step 2: Analyze high-impact survey structure before drafting**

Create paper/notes/survey-structure.md. Verify the current metadata and citation-count
snapshot date for these exemplars, using counts only to explain exemplar selection:

- IJRR: Reinforcement Learning in Robotics: A Survey;
- IJRR: Human Motion Trajectory Prediction: A Survey;
- IJRR: Dynamic Movement Primitives in Robotics: A Tutorial Survey;
- Science Robotics: Social Robots for Education: A Review;
- Science Robotics: Biohybrid Actuators for Robotics: A Review of Devices Actuated by Living Cells;
- Science Robotics: A Review of Collective Robotic Construction.

For each article, record the section sequence, where scope is bounded, how prior reviews
are distinguished, taxonomy depth, comparison-table axes, treatment of evaluation and
reproducibility, and how the research agenda is argued. End the note with an
adopt/adapt/reject matrix for this survey. The adopted structure must keep
representations, generation mechanisms, metrics, benchmarks, and open problems separate;
citation count must not be used as evidence that a structural choice is scientifically
correct.


- [ ] **Step 3: Inventory every named source in the bootstrap documents**

Read these files completely:

~~~text
docs/related-work/state-of-the-art.rst
docs/related-work/prior-art.rst
docs/generators/benchmarks.rst
docs/tutorials/gate-sequences.rst
docs/superpowers/specs/2026-06-23-gate-sequence-generation-design.md
~~~

For every named paper, benchmark, simulator, competition, software system, or standard,
add one seed_coverage.csv row. Set coverage_status to linked when a candidate row exists,
excluded only after screening records a reason, and unreviewed otherwise. Preserve the
source heading and visible source label so the inventory is auditable.

- [ ] **Step 4: Convert existing citations into candidate rows without treating them as verified**

Assign stable IDs C0001 onward. Copy titles and links from the seed notes, set:

~~~text
discovery_stream=bootstrap
discovery_agent=main
screening_status=candidate
metadata_status=unverified
~~~

Use one row per distinct work. A project paper and its official software repository share
one candidate row when the paper is the citable system description; retain both DOI and
official project URL where available.

- [ ] **Step 5: Record the bootstrap operation**

Add one search_log.csv row for each bootstrap file. The query field is the file path,
search_surface is local-corpus, results_screened is the number of named sources reviewed,
and candidates_added is the number of new candidate IDs.

- [ ] **Step 6: Validate and commit**




Run:

~~~bash
python3 paper/scripts/validate_corpus.py
git add paper/data paper/notes/survey-structure.md
git commit -m "docs: bootstrap survey corpus from related work"
~~~

Expected: validation passes; every seed_coverage row is linked, excluded, or explicitly
unreviewed, and no seed is silently omitted.

## Task 4: Run Independent Blind And Corpus-Aware Discovery

**Files:**

- Create: paper/data/agent_runs/blind-ground.csv
- Create: paper/data/agent_runs/blind-aerial-maritime.csv
- Create: paper/data/agent_runs/aware-geometry-rl.csv
- Create: paper/data/agent_runs/aware-simulation-benchmarks.csv
- Create: paper/data/agent_runs/*.md
- Populate: paper/data/search_log.csv

- [ ] **Step 1: Dispatch two blind agents in parallel**

Use superpowers:dispatching-parallel-agents. Do not give either blind agent repository
paths, known paper names, current taxonomy labels, or TrackGen implementation details.

Blind ground-agent prompt:

~~~text
Find primary research and official system papers on generating tracks, roads, circuits,
or course distributions used to train or evaluate racing and agile ground robots.
Search autonomous racing, robot learning, autonomous-vehicle testing, racing-game PCG
when geometry transfers to robotics, legged or mixed-terrain robot racing, and benchmark/competition course design. Include
methods that synthesize, mutate, repair, select, or serialize geometry. Distinguish fixed
track control papers from actual generation contributions. For every candidate report:
title, authors, year, venue, DOI or stable primary URL, exact generation role,
representation, validity method, metrics, simulator/export format, public code/assets,
and a page/section/official-URL evidence locator. Report absent facts as NR. Search until
two consecutive query refinements add less than five percent new in-scope candidates.
Write only your own CSV and Markdown report; do not edit a shared bibliography or paper.
~~~

Blind aerial/maritime-agent prompt:

~~~text
Find primary research and official system papers on generating gate chains, waypoint
courses, obstacle courses, buoy courses, waterways, or course distributions used to
train or evaluate aerial and maritime racing/agile robots. Search drone racing,
quadrotor RL/control, autonomous surface vessels, boat racing, RobotX/VRX-style tasks,
and simulators that define portable course formats. Distinguish randomized course
geometry from visual or dynamics randomization. For every candidate report: title,
authors, year, venue, DOI or stable primary URL, exact generation role, representation,
validity method, metrics, simulator/export format, public code/assets, and a
page/section/official-URL evidence locator. Report absent facts as NR. Search until two
consecutive query refinements add less than five percent new in-scope candidates. Write
only your own CSV and Markdown report; do not edit a shared bibliography or paper.
~~~

- [ ] **Step 2: Dispatch two corpus-aware agents in parallel**

Give both agents the approved design spec plus only the bootstrap files relevant to their
brief.

Corpus-aware geometry/RL prompt:

~~~text
Read the survey design, state-of-the-art.rst, prior-art.rst, and benchmarks.rst. Treat
them as seed maps, not scope boundaries. Verify their primary references, follow backward
and forward citation chains, search alternate terminology, and find missing work in
ground racing, road geometry, PCG, UED/curriculum learning, repair/projection, and
dynamics-aware difficulty. Mark which sources were already in the seed corpus and which
were newly discovered. Extract the complete candidate schema with direct evidence
locators, using NR for unreported fields. Search until two consecutive expansion rounds
add less than five percent new in-scope candidates. Write only aware-geometry-rl.csv and
its report.
~~~

Corpus-aware simulator/benchmark prompt:

~~~text
Read the survey design, gate-sequences.rst, the gate-sequence design spec, and the
existing related-work notes. Expand the corpus around aerial and maritime course
generation, competitions, benchmark sets, simulator map formats, and RL framework
interfaces. Investigate Gymnasium/CarRacing, F1TENTH/RoboRacer, CARLA/OpenDRIVE,
Isaac Lab/Isaac Sim, AirSim, Gazebo/VRX, and generic serialized course bundles through
official papers, documentation, or standards. Record actual input representation,
coordinate frames, units, collision geometry, spawn/reset behavior, load validation,
and OSS status when documented. Use NR otherwise. Write only
aware-simulation-benchmarks.csv and its report.
~~~

- [ ] **Step 3: Require a common output schema without letting agents share state**

Each agent CSV must use the candidates.csv columns plus the evidence.csv coding columns.
Each Markdown report must contain:

1. queries and databases searched;
2. inclusion and boundary judgments;
3. terminology not present in the supplied brief;
4. sparse or contradictory areas;
5. two-round saturation calculation;
6. high-priority primary sources requiring manual retrieval.

- [ ] **Step 4: Review agent outputs for primary-source quality**

Reject rows supported only by Wikipedia, generic search snippets, secondary blog posts,
or aggregator metadata. Such surfaces may identify a source, but the row's
metadata_evidence and evidence_locator must point to the primary paper, proceedings,
standard, official documentation, or official repository.

- [ ] **Step 5: Record and commit immutable agent outputs**




Run:

~~~bash
git add paper/data/agent_runs paper/data/search_log.csv
git commit -m "docs: add independent survey discovery runs"
~~~

## Task 5: Merge, Deduplicate, And Verify The Candidate Corpus

**Files:**

- Create: paper/scripts/merge_candidates.py
- Modify: tests/test_survey_corpus.py
- Populate: paper/data/candidates.csv
- Populate: paper/data/conflicts.csv
- Populate: paper/references.bib

- [ ] **Step 1: Write failing merge tests**

Add tests proving:

- DOI values deduplicate after removing doi.org and doi: prefixes and lowercasing;
- rows without a DOI deduplicate by Unicode-normalized lowercase title after punctuation
  and repeated whitespace are removed;
- merged discovery_stream, discovery_query, and discovery_agent values retain all unique
  origins in sorted semicolon order;
- conflicting title, year, or venue values create conflicts.csv rows instead of choosing
  silently;
- the existing stable candidate_id wins over an agent-local ID.

Use this public interface:

~~~python
from paper.scripts.merge_candidates import merge_candidate_files

merged, conflicts = merge_candidate_files(existing_path, agent_paths)
~~~

- [ ] **Step 2: Run tests and confirm the expected import failure**

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_corpus.py
~~~

Expected: failure because merge_candidates.py does not exist.

- [ ] **Step 3: Implement deterministic merging**

Implement:

~~~python
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

from paper.scripts.validate_corpus import HEADERS, normalize_doi, split_values


BIBLIOGRAPHIC_FIELDS = ("title", "authors", "year", "venue", "doi", "url")
PROVENANCE_FIELDS = (
    "discovery_stream",
    "discovery_query",
    "discovery_agent",
)


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def identity_key(row: dict[str, str]) -> tuple[str, str]:
    doi = normalize_doi(row["doi"])
    return ("doi", doi) if doi else ("title", normalize_title(row["title"]))


def _identity_keys(row: dict[str, str]) -> set[tuple[str, str]]:
    keys = {("title", normalize_title(row["title"]))}
    doi = normalize_doi(row["doi"])
    if doi:
        keys.add(("doi", doi))
    return {key for key in keys if key[1]}


def _read_agent_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(HEADERS["candidates.csv"]) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing candidate columns {sorted(missing)}")
        return [
            {field: row.get(field, "") for field in HEADERS["candidates.csv"]}
            for row in reader
        ]


def _union_values(left: str, right: str) -> str:
    return ";".join(sorted(set(split_values(left)) | set(split_values(right))))


def _next_candidate_number(rows: list[dict[str, str]]) -> int:
    numbers = [
        int(row["candidate_id"][1:])
        for row in rows
        if re.fullmatch(r"C\d{4,}", row["candidate_id"])
    ]
    return max(numbers, default=0) + 1


def merge_candidate_files(
    existing_path: Path,
    agent_paths: list[Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    existing = _read_agent_rows(existing_path)
    merged = [dict(row) for row in existing]
    lookup: dict[tuple[str, str], int] = {}
    for index, row in enumerate(merged):
        for key in _identity_keys(row):
            lookup[key] = index

    incoming_rows = [
        row
        for path in sorted(agent_paths)
        for row in _read_agent_rows(path)
    ]
    incoming_rows.sort(key=lambda row: (identity_key(row), row["title"]))
    next_number = _next_candidate_number(merged)
    conflicts: list[dict[str, str]] = []

    for incoming in incoming_rows:
        matching = sorted(
            {lookup[key] for key in _identity_keys(incoming) if key in lookup}
        )
        if len(matching) > 1:
            raise ValueError(
                f"{incoming['title']!r} bridges multiple existing candidates {matching}"
            )
        if not matching:
            record = dict(incoming)
            record["candidate_id"] = f"C{next_number:04d}"
            next_number += 1
            merged.append(record)
            index = len(merged) - 1
            for key in _identity_keys(record):
                lookup[key] = index
            continue

        index = matching[0]
        record = merged[index]
        for field in PROVENANCE_FIELDS:
            record[field] = _union_values(record[field], incoming[field])
        for field in BIBLIOGRAPHIC_FIELDS:
            current = record[field].strip()
            proposed = incoming[field].strip()
            if not current and proposed:
                record[field] = proposed
                continue
            if not proposed:
                continue
            equivalent = (
                normalize_doi(current) == normalize_doi(proposed)
                if field == "doi"
                else normalize_title(current) == normalize_title(proposed)
                if field == "title"
                else current == proposed
            )
            if equivalent:
                continue
            conflict = dict.fromkeys(HEADERS["conflicts.csv"], "")
            conflict.update(
                conflict_id=f"X{len(conflicts) + 1:04d}",
                record_type="candidate",
                record_key=record["candidate_id"],
                field=field,
                value_a=current,
                value_b=proposed,
            )
            conflicts.append(conflict)
        for field in HEADERS["candidates.csv"]:
            if field not in {"candidate_id", *PROVENANCE_FIELDS, *BIBLIOGRAPHIC_FIELDS}:
                if not record[field].strip() and incoming[field].strip():
                    record[field] = incoming[field]
        for key in _identity_keys(record):
            lookup[key] = index

    return sorted(merged, key=lambda row: row["candidate_id"]), conflicts


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    header = HEADERS[path.name]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--agent", type=Path, action="append", default=[])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    merged, conflicts = merge_candidate_files(args.existing, args.agent)
    print(f"merged={len(merged)} conflicts={len(conflicts)}")
    if args.write:
        _write_rows(args.existing, merged)
        _write_rows(args.existing.parent / "conflicts.csv", conflicts)


if __name__ == "__main__":
    main()
~~~


- [ ] **Step 4: Merge all four agent files**




Run:

~~~bash
python3 paper/scripts/merge_candidates.py \
  --existing paper/data/candidates.csv \
  --agent paper/data/agent_runs/blind-ground.csv \
  --agent paper/data/agent_runs/blind-aerial-maritime.csv \
  --agent paper/data/agent_runs/aware-geometry-rl.csv \
  --agent paper/data/agent_runs/aware-simulation-benchmarks.csv
python3 paper/scripts/merge_candidates.py \
  --existing paper/data/candidates.csv \
  --agent paper/data/agent_runs/blind-ground.csv \
  --agent paper/data/agent_runs/blind-aerial-maritime.csv \
  --agent paper/data/agent_runs/aware-geometry-rl.csv \
  --agent paper/data/agent_runs/aware-simulation-benchmarks.csv \
  --write
~~~

Expected: the dry-run reports counts by source stream, duplicate identity, and conflict
field; the write run assigns stable IDs and does not erase bootstrap provenance.

- [ ] **Step 5: Verify metadata source by source**

For each candidate likely to be included:

1. resolve the DOI or stable paper URL;
2. match title, complete author list, year, and venue against the publisher or proceedings;
3. prefer the version of record DOI while retaining an accessible preprint URL when useful;
4. record the metadata source in metadata_evidence;
5. set metadata_status to verified only after all bibliographic fields agree;
6. resolve every conflicts.csv row with a source URL and resolver name;
7. create a BibTeX key using FirstAuthorYearShortTitle, adding a lowercase suffix for collisions;
8. add a references.bib entry containing author, title, year, venue, DOI when available,
   URL when useful, and no unverified abstract or keyword fields.

For software, standards, and competitions without a paper, use the official documentation
as the citable source and set source_type accordingly.

- [ ] **Step 6: Validate bibliography correspondence**

Extend validate_corpus.py so every verified included or boundary cite_key has exactly one
BibTeX entry and every BibTeX entry maps to a candidate row. Parse BibTeX entry keys with
a conservative regular expression; do not attempt to rewrite BibTeX.

Add this implementation to validate_corpus.py and call
validate_bibliography(data_dir.parent / "references.bib", screened_keys) at the end of
validate_directory:

~~~python
BIB_KEY_PATTERN = re.compile(
    r"(?m)^@\w+\s*{\s*([^,\s]+)\s*,"
)


def validate_bibliography(path: Path, expected_keys: set[str]) -> None:
    if not path.is_file():
        raise CorpusError(f"{path}: bibliography is missing")
    keys = BIB_KEY_PATTERN.findall(path.read_text())
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise CorpusError(f"{path}: duplicate BibTeX keys {duplicates}")
    actual = set(keys)
    if actual != expected_keys:
        raise CorpusError(
            f"{path}: BibTeX mismatch; "
            f"missing={sorted(expected_keys - actual)}, "
            f"extra={sorted(actual - expected_keys)}"
        )
~~~

Import re and Counter at module scope. Update build_valid_fixture so references.bib
contains:

~~~bibtex
@article{Sample2026Course,
  author = {Author, A.},
  title = {A Fictional Course Generator},
  journal = {Test Proceedings},
  year = {2026},
  doi = {10.0000/example}
}
~~~





Run:

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_corpus.py
python3 paper/scripts/validate_corpus.py
~~~

Expected: all tests and production validation pass with no unresolved bibliographic
conflicts for screened-in sources.

- [ ] **Step 7: Commit the reconciled bibliography**

~~~bash
git add paper/scripts paper/data paper/references.bib tests/test_survey_corpus.py
git commit -m "docs: reconcile and verify survey bibliography"
~~~

## Task 6: Screen Sources And Code Evidence With Reliability Checks

**Files:**

- Modify: paper/data/README.md
- Populate: paper/data/candidates.csv
- Populate: paper/data/evidence.csv
- Create: paper/data/coding_primary.csv
- Create: paper/data/coding_reliability.csv
- Create: paper/data/coding_reliability_summary.csv
- Create: paper/scripts/coding_reliability.py
- Populate: paper/data/conflicts.csv
- Modify: paper/scripts/validate_corpus.py
- Modify: tests/test_survey_corpus.py

- [ ] **Step 1: Apply explicit screening rules**

Include a source when it satisfies at least one rule:

1. it synthesizes, selects, mutates, repairs, validates, or serializes course geometry or
   a course distribution for robot racing or an adjacent transferable domain;
2. it defines a fixed benchmark, competition course set, or simulator interface that
   materially constrains course representation or evaluation;
3. it defines a metric or dynamics model used to characterize generated courses;
4. it is an adjacent survey needed to establish the survey gap.

Mark a source boundary when it studies racing/control on fixed courses but contributes a
requirement, metric, dataset, or reporting practice used by this survey.

Exclude a source when it only optimizes a racing line inside a fixed corridor, only
randomizes appearance or dynamics, only generates traffic participants on fixed roads,
or mentions a course without enough detail to support a survey claim. Preserve the row
and record the specific reason.

- [ ] **Step 2: Code direct evidence rather than abstract-level impressions**

For every included and boundary source, inspect the full paper and official supplement
when available. Populate evidence.csv using:

- exact representation and generator-family values from taxonomy.json;
- generation_role to distinguish geometry synthesis from task selection and replay;
- validity_strategy based on the actual algorithm;
- reported metric names without translating them into the survey's preferred terms;
- distribution sizes and benchmark counts with units;
- simulator/export details only when explicitly documented;
- code and asset status from official URLs;
- evidence_locator with page/section/table/figure or official documentation anchor;
- coding_notes for ambiguity, multiple roles, or domain transfer assumptions.

- [ ] **Step 3: Run an independent reliability sample**

Select a deterministic 20 percent stratified sample using SHA-256 of cite_key, taking at
least two sources from each populated domain. A second agent codes the sample without
seeing coding_primary.csv. Store its rows in coding_reliability.csv.

Compute exact agreement for:

~~~text
domain
course_object
representation_family
generator_family
generation_role
validity_strategy
code_status
~~~

Implement paper/scripts/coding_reliability.py:

~~~python
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path


CORE_FIELDS = (
    "domain",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "code_status",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical(value: str) -> str:
    return ";".join(sorted(item.strip() for item in value.split(";") if item.strip()))


def select_reliability_sample(
    evidence: list[dict[str, str]],
    fraction: float = 0.20,
) -> list[dict[str, str]]:
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evidence:
        domains = [item.strip() for item in row["domain"].split(";") if item.strip()]
        by_domain[domains[0]].append(row)
    selected: dict[str, dict[str, str]] = {}
    for rows in by_domain.values():
        count = min(len(rows), max(2, math.ceil(fraction * len(rows))))
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                row["cite_key"].encode("utf-8")
            ).hexdigest(),
        )
        for row in ranked[:count]:
            selected[row["cite_key"]] = row
    return [selected[key] for key in sorted(selected)]


def cohens_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must have equal nonzero length")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[value] / len(left)) * (right_counts[value] / len(right))
        for value in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def compare_codings(
    primary: list[dict[str, str]],
    reliability: list[dict[str, str]],
) -> list[dict[str, str]]:
    left = {row["cite_key"]: row for row in primary}
    right = {row["cite_key"]: row for row in reliability}
    if set(left) != set(right):
        raise ValueError(
            f"coding samples differ: primary={sorted(left)}, reliability={sorted(right)}"
        )
    summary: list[dict[str, str]] = []
    keys = sorted(left)
    for field in CORE_FIELDS:
        values_left = [_canonical(left[key][field]) for key in keys]
        values_right = [_canonical(right[key][field]) for key in keys]
        agreement = sum(
            a == b for a, b in zip(values_left, values_right)
        ) / len(keys)
        categories = set(values_left) | set(values_right)
        kappa = (
            cohens_kappa(values_left, values_right)
            if len(categories) >= 2
            else 1.0
        )
        summary.append(
            {
                "field": field,
                "n": str(len(keys)),
                "agreement": f"{agreement:.6f}",
                "kappa": f"{kappa:.6f}",
                "passes": str(agreement >= 0.80).lower(),
            }
        )
    return summary


def _write(path: Path, rows: list[dict[str, str]], header: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        rows = select_reliability_sample(_read(args.evidence))
        _write(args.output, rows, list(rows[0]) if rows else ["cite_key"])
        return
    summary = compare_codings(_read(args.primary), _read(args.reliability))
    _write(
        args.output,
        summary,
        ["field", "n", "agreement", "kappa", "passes"],
    )


if __name__ == "__main__":
    main()
~~~

Prepare the sample with:

~~~bash
python3 paper/scripts/coding_reliability.py --prepare \
  --evidence paper/data/evidence.csv \
  --output paper/data/coding_primary.csv
~~~

After the second coding pass, compare with:

~~~bash
python3 paper/scripts/coding_reliability.py \
  --primary paper/data/coding_primary.csv \
  --reliability paper/data/coding_reliability.csv \
  --output paper/data/coding_reliability_summary.csv
~~~

Add unit tests for deterministic stratified selection, order-insensitive semicolon labels,
perfect agreement, one deliberate disagreement, sample-key mismatch, and the 0.80 pass
threshold.


Report per-field agreement and Cohen's kappa when both coders use at least two categories.
If exact agreement is below 0.80 for any core field, revise that field's codebook,
independently recode the reliability sample for that field, and retain the original
disagreement in conflicts.csv.

- [ ] **Step 4: Add reliability and coverage validation**

Add tests requiring:

- one evidence row for every included source;
- no evidence row for excluded sources;
- no unreviewed seed_coverage row after screening closes;
- no unsupported controlled-vocabulary value;
- every populated evidence field has an evidence_locator;
- coding reliability summary exists and all reviewed core fields meet the threshold or
  have a resolved codebook conflict.

- [ ] **Step 5: Run a targeted sparse-cell search**

Create a cross-tabulation of domain by representation_family and domain by
generator_family. For every empty cell that is scientifically plausible, run at least
one targeted query and log it. Mark a cell structurally inapplicable only with a written
reason in the coding report. Continue each high-priority stream until two consecutive
query refinements add less than five percent new in-scope sources.

- [ ] **Step 6: Validate and commit screened evidence**

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_corpus.py
python3 paper/scripts/validate_corpus.py
git add paper/data paper/scripts/validate_corpus.py paper/scripts/coding_reliability.py tests/test_survey_corpus.py
git commit -m "docs: screen and code survey evidence"
~~~

## Task 7: Review And Freeze Scientific Decisions

**Files:**

- Create: paper/decisions/0001-scope-and-unit-of-analysis.md
- Create: paper/decisions/0002-representation-taxonomy.md
- Create: paper/decisions/0003-generator-taxonomy.md
- Create: paper/decisions/0004-difficulty-spectrum.md
- Create: paper/decisions/0005-simulator-feasibility.md
- Populate: paper/data/metrics.csv
- Populate: paper/data/simulators.csv

- [ ] **Step 1: Write the scope and unit-of-analysis decision**

Record:

- course as the umbrella term for tracks, road corridors, gate chains, waypoint routes,
  buoy courses, and simulator world layouts when geometry controls the task;
- generator as any method that synthesizes, mutates, repairs, selects, or packages course
  instances, with generation_role preserving these distinctions;
- racing-line optimization, trajectory planning, control, perception, and policy learning
  as downstream areas included only when they impose requirements or supply metrics;
- fixed benchmark courses as benchmark evidence, not algorithmic generators.

List every evidence-backed exception and its cite_keys.

- [ ] **Step 2: Freeze representation and generator taxonomies**

For each category, record:

1. operational definition;
2. inclusion and exclusion examples;
3. at least two representative sources when the corpus permits;
4. known ambiguity and how multi-label cases are coded;
5. consequences for feasibility, batching, portability, and vehicle compatibility.

Update taxonomy.json and recode evidence.csv in the same commit if a category changes.

- [ ] **Step 3: Define difficulty as a spectrum, not three named bins**

In 0004-difficulty-spectrum.md, choose:

- domain-specific difficulty coordinates;
- which coordinates require a vehicle or dynamics model;
- normalization and direction for every coordinate;
- whether the paper recommends a scalar score, a Pareto/vector description, or both;
- how quantiles or operating points may be named without implying universal thresholds;
- how feasibility is separated from difficulty;
- how uncertainty and simulator failures are represented;
- how evaluation courses are sampled across the resulting spectrum.

Populate metrics.csv with equations or exact procedures, units, direction, domain,
dynamics requirements, primary citations, and limitations. Do not use easy, medium, or
hard as canonical scientific categories.

- [ ] **Step 4: Define simulation feasibility as an auditable layer**

In 0005-simulator-feasibility.md, distinguish:

- serialization success;
- simulator import success;
- geometric and collision validity after import;
- coordinate-frame and unit consistency;
- spawn and reset validity;
- deterministic task reconstruction;
- RL interface availability;
- throughput and batchability.

Populate simulators.csv from official evidence for Gymnasium/CarRacing,
F1TENTH/RoboRacer, CARLA/OpenDRIVE, Isaac Lab/Isaac Sim, AirSim, Gazebo/VRX, and generic
JSON/NPZ bundles. Use NR where behavior is not documented.

- [ ] **Step 5: Hold the scientific review checkpoint**

Present the five decisions plus:

- corpus counts by domain, representation, generator family, and screening status;
- independent versus bootstrap discoveries;
- unresolved sparse cells;
- metric candidates and their evidence strength;
- the proposed spectrum representation;
- simulator/export evidence gaps;
- claims that TrackGen can currently support and claims requiring follow-on work.

Obtain explicit user approval before drafting the benchmark recommendations as settled
survey conclusions. Record requested changes in the decision files and recode affected
evidence.

- [ ] **Step 6: Validate and commit the decisions**

~~~bash
python3 paper/scripts/validate_corpus.py
git add paper/decisions paper/data/taxonomy.json paper/data/metrics.csv \
  paper/data/simulators.csv paper/data/evidence.csv
git commit -m "docs: freeze survey taxonomy and metric decisions"
~~~

## Task 8: Generate Evidence-Backed Comparison Tables

**Files:**

- Create: paper/scripts/render_tables.py
- Create: paper/tables/adjacent-surveys.tex
- Create: paper/tables/method-comparison-ground.tex
- Create: paper/tables/method-comparison-aerial.tex
- Create: paper/tables/method-comparison-maritime.tex
- Create: paper/tables/metric-hierarchy.tex
- Create: paper/tables/simulator-compatibility.tex
- Create: paper/tables/reporting-audit.tex
- Create: tests/test_survey_tables.py

- [ ] **Step 1: Write failing rendering tests**

Test:

- LaTeX escaping of ampersand, percent, underscore, hash, braces, and backslash;
- deterministic row sorting by year then cite_key;
- long values converted to stable abbreviations defined in the table caption or legend;
- NR rendered through the NotReported macro;
- every rendered row includes a citation;
- running the renderer twice yields byte-identical output.

Use:

~~~python
from paper.scripts.render_tables import escape_latex, render_all
~~~

- [ ] **Step 2: Run tests and confirm import failure**

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_tables.py
~~~

Expected: collection fails because render_tables.py does not exist.

- [ ] **Step 3: Implement table rendering**

Implement:

~~~python
from __future__ import annotations

import csv
from pathlib import Path


TABLE_OUTPUTS = {
    "adjacent-surveys": "adjacent-surveys.tex",
    "method-ground": "method-comparison-ground.tex",
    "method-aerial": "method-comparison-aerial.tex",
    "method-maritime": "method-comparison-maritime.tex",
    "metrics": "metric-hierarchy.tex",
    "simulators": "simulator-compatibility.tex",
    "reporting": "reporting-audit.tex",
}

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(value: str) -> str:
    return "".join(LATEX_ESCAPES.get(character, character) for character in value)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _values(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}


def _cell(value: str) -> str:
    value = value.strip()
    return r"\NotReported{}" if not value or value == "NR" else escape_latex(value)


def _citation(cite_key: str) -> str:
    return rf"\citep{{{cite_key}}}"


def _write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def _longtable(
    caption: str,
    label: str,
    column_spec: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    lines = [
        rf"\begin{{longtable}}{{{column_spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines)


def _candidate_sort(row: dict[str, str]) -> tuple[int, str]:
    year = int(row["year"]) if row["year"].isdigit() else 9999
    return year, row["cite_key"]


def _method_rows(
    domain: str,
    evidence: list[dict[str, str]],
    candidates: dict[str, dict[str, str]],
) -> list[list[str]]:
    selected = [
        row for row in evidence
        if domain in _values(row["domain"])
    ]
    selected.sort(key=lambda row: _candidate_sort(candidates[row["cite_key"]]))
    return [
        [
            _cell(candidates[row["cite_key"]]["title"])
            + " "
            + _citation(row["cite_key"]),
            _cell(row["representation_family"]),
            _cell(row["generator_family"]),
            _cell(row["generation_role"]),
            _cell(row["validity_strategy"]),
            _cell(row["simulator"]),
            _cell(row["code_status"]),
        ]
        for row in selected
    ]


def _reporting_rows(evidence: list[dict[str, str]]) -> list[list[str]]:
    fields = (
        "training_distribution",
        "evaluation_suite",
        "geometry_metrics",
        "difficulty_metrics",
        "simulator",
        "code_status",
        "reproducibility_fields",
    )
    rows: list[list[str]] = []
    for domain in ("ground", "aerial", "maritime", "mixed", "adjacent"):
        selected = [row for row in evidence if domain in _values(row["domain"])]
        if not selected:
            continue
        values = [_cell(domain), str(len(selected))]
        for field in fields:
            reported = sum(
                bool(row[field].strip() and row[field].strip() != "NR")
                for row in selected
            )
            values.append(f"{reported}/{len(selected)}")
        rows.append(values)
    return rows


def render_all(data_dir: Path, output_dir: Path) -> list[Path]:
    candidate_rows = _read(data_dir / "candidates.csv")
    candidates = {
        row["cite_key"]: row
        for row in candidate_rows
        if row["screening_status"] in {"included", "boundary"}
    }
    evidence = _read(data_dir / "evidence.csv")
    metrics = _read(data_dir / "metrics.csv")
    simulators = _read(data_dir / "simulators.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    survey_candidates = sorted(
        (
            row for row in candidate_rows
            if row["screening_status"] in {"included", "boundary"}
            and row["source_type"] in {"survey", "review"}
        ),
        key=_candidate_sort,
    )
    evidence_by_key = {row["cite_key"]: row for row in evidence}
    adjacent_rows = [
        [
            _cell(row["title"]) + " " + _citation(row["cite_key"]),
            _cell(evidence_by_key[row["cite_key"]]["domain"]),
            _cell(evidence_by_key[row["cite_key"]]["course_object"]),
            _cell(evidence_by_key[row["cite_key"]]["representation_family"]),
            _cell(evidence_by_key[row["cite_key"]]["generator_family"]),
            _cell(evidence_by_key[row["cite_key"]]["simulator"]),
        ]
        for row in survey_candidates
    ]

    contents = {
        "adjacent-surveys": _longtable(
            "Coverage of adjacent surveys and reviews.",
            "tab:adjacent-surveys",
            r"@{}p{.24\linewidth}p{.10\linewidth}p{.14\linewidth}"
            r"p{.16\linewidth}p{.16\linewidth}p{.10\linewidth}@{}",
            ["Work", "Domain", "Course", "Representation", "Method", "Simulator"],
            adjacent_rows,
        ),
        "method-ground": _longtable(
            "Ground-domain course-generation methods.",
            "tab:methods-ground",
            r"@{}p{.22\linewidth}p{.13\linewidth}p{.13\linewidth}"
            r"p{.12\linewidth}p{.13\linewidth}p{.11\linewidth}p{.08\linewidth}@{}",
            ["Work", "Representation", "Family", "Role", "Validity", "Simulator", "Code"],
            _method_rows("ground", evidence, candidates),
        ),
        "method-aerial": _longtable(
            "Aerial-domain course-generation methods.",
            "tab:methods-aerial",
            r"@{}p{.22\linewidth}p{.13\linewidth}p{.13\linewidth}"
            r"p{.12\linewidth}p{.13\linewidth}p{.11\linewidth}p{.08\linewidth}@{}",
            ["Work", "Representation", "Family", "Role", "Validity", "Simulator", "Code"],
            _method_rows("aerial", evidence, candidates),
        ),
        "method-maritime": _longtable(
            "Maritime-domain course-generation methods.",
            "tab:methods-maritime",
            r"@{}p{.22\linewidth}p{.13\linewidth}p{.13\linewidth}"
            r"p{.12\linewidth}p{.13\linewidth}p{.11\linewidth}p{.08\linewidth}@{}",
            ["Work", "Representation", "Family", "Role", "Validity", "Simulator", "Code"],
            _method_rows("maritime", evidence, candidates),
        ),
        "metrics": _longtable(
            "Candidate metrics and their reporting requirements.",
            "tab:metric-hierarchy",
            r"@{}p{.12\linewidth}p{.15\linewidth}p{.25\linewidth}"
            r"p{.08\linewidth}p{.10\linewidth}p{.20\linewidth}@{}",
            ["Layer", "Metric", "Definition", "Units", "Domain", "Limitation"],
            [
                [
                    _cell(row["layer"]),
                    _cell(row["name"]),
                    _cell(row["definition"]),
                    _cell(row["units"]),
                    _cell(row["domain"]),
                    _cell(row["limitations"])
                    + (" " + _citation(row["cite_keys"].split(";")[0])
                       if row["cite_keys"].strip() else ""),
                ]
                for row in sorted(metrics, key=lambda row: row["metric_id"])
            ],
        ),
        "simulators": _longtable(
            "Documented simulator and export compatibility.",
            "tab:simulator-compatibility",
            r"@{}p{.13\linewidth}p{.12\linewidth}p{.15\linewidth}"
            r"p{.12\linewidth}p{.12\linewidth}p{.12\linewidth}p{.12\linewidth}@{}",
            ["System", "Domain", "Input", "Export", "Import check", "Reset", "RL interface"],
            [
                [
                    _cell(row["system"]) + " " + _citation(row["cite_key"]),
                    _cell(row["domain"]),
                    _cell(row["input_representation"]),
                    _cell(row["export_format"]),
                    _cell(row["load_validation"]),
                    _cell(row["spawn_reset"]),
                    _cell(row["rl_interface"]),
                ]
                for row in sorted(simulators, key=lambda row: row["system"].lower())
            ],
        ),
        "reporting": _longtable(
            "Disclosure counts by domain; each cell is reported sources over the domain denominator.",
            "tab:reporting-audit",
            r"@{}lrrrrrrrr@{}",
            ["Domain", "N", "Train", "Eval", "Geometry", "Difficulty", "Simulator", "Code", "Repro."],
            _reporting_rows(evidence),
        ),
    }

    paths: list[Path] = []
    for table_name, filename in TABLE_OUTPUTS.items():
        path = output_dir / filename
        _write_if_changed(path, contents[table_name])
        paths.append(path)
    return paths


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    render_all(root / "data", root / "tables")
~~~


- [ ] **Step 4: Define exact table responsibilities**

- adjacent-surveys.tex compares scope coverage for geometry, generators, feasibility,
  vehicle difficulty, OSS, distributions, simulator export, and reproducibility.
- method-comparison-*.tex reports representation, generator role, validity, distribution,
  metrics, simulator, and code status; split by domain for legibility.
- metric-hierarchy.tex reports definition, domain, model requirement, direction, and
  limitation.
- simulator-compatibility.tex reports documented input format, import validation,
  coordinate frame, units, collision, reset, and RL interface.
- reporting-audit.tex aggregates disclosure rates by research community and field; the
  caption states the denominator and boundary-source treatment.

- [ ] **Step 5: Run tests and regenerate tables**

~~~bash
.venv/bin/python -m pytest -q tests/test_survey_tables.py
python3 paper/scripts/validate_corpus.py
python3 paper/scripts/render_tables.py
git diff --exit-code paper/tables
~~~

For the first generation, git diff is expected to show new files; stage them, rerun the
renderer, then require git diff --exit-code paper/tables on every later check.

- [ ] **Step 6: Commit generated tables and renderer**

~~~bash
git add paper/scripts/render_tables.py paper/tables tests/test_survey_tables.py
git commit -m "docs: generate survey comparison tables"
~~~

## Task 9: Draft The Evidence-Backed Manuscript

**Files:**

- Modify: paper/sections/00-abstract.tex
- Modify: paper/sections/01-introduction.tex
- Modify: paper/sections/02-review-protocol.tex
- Modify: paper/sections/03-scope-definitions.tex
- Modify: paper/sections/04-adjacent-surveys.tex
- Modify: paper/sections/05-representations.tex
- Modify: paper/sections/06-generation-methods.tex
- Modify: paper/sections/07-domain-constraints.tex
- Modify: paper/sections/08-metrics.tex
- Modify: paper/sections/09-benchmark-protocol.tex
- Modify: paper/sections/10-reference-implementations.tex
- Modify: paper/sections/11-reporting-practices.tex
- Modify: paper/sections/12-open-problems.tex
- Modify: paper/sections/13-conclusion.tex
- Populate: paper/data/claims.csv

- [ ] **Step 1: Draft the opening argument and review protocol**

Write Abstract, Introduction, Review Protocol, Scope/Definitions, and Relationship to
Existing Surveys in this order.

The Introduction must establish:

1. generated courses define a task distribution, not incidental scenery;
2. that distribution affects generalization, curriculum, safety margins, and sim-to-real
   claims;
3. current practice is fragmented across ground, aerial, maritime, PCG, testing, and UED;
4. the paper contributes common representations, method taxonomy, metrics, reporting,
   benchmark protocol, and OSS reference-implementation guidance.

The Review Protocol reports exact dates, search surfaces, streams, screening counts,
reliability results, and limitations. Do not describe the review as systematic unless
the completed process satisfies the corresponding reporting standard.

Add every externally verifiable sentence to claims.csv with direct cite_keys. Mark
evidence_status direct or triangulated; inferred claims must say that they are the
authors' synthesis in the manuscript.

- [ ] **Step 2: Draft representation before generation method**

Representations must be compared on:

- expressive dimension and topology;
- validity by construction;
- geometric continuity;
- vehicle/domain compatibility;
- simulator portability;
- batched generation and storage cost;
- ability to support mutation, repair, and difficulty measurement.

Generation Methods must distinguish geometry synthesis, task selection, mutation/replay,
and repair. Organize by mechanism rather than chronology and use the domain tables as
evidence, not as a replacement for synthesis.

- [ ] **Step 3: Draft domain constraints and metric hierarchy**

Use separate subsections for:

- ground vehicles: closure, width, curvature, friction, overtaking, map and boundary
  semantics, and dynamics-aware speed profiles;
- aerial vehicles: 3D gate pose, visibility, spacing, clearance, vertical motion,
  orientation, field of view, and obstacle placement;
- maritime vehicles: buoy semantics, waterways, current/wind, hydrodynamic turn radius,
  shoreline clearance, waypoint ambiguity, and simulator support;
- other agile robots: legged racing, hovercraft, and mixed-terrain courses when generation materially changes policy generalization;
- cross-domain invariants: finite geometry, ordering, clearance, reproducibility,
  coordinate frames, and deterministic reconstruction.

Metrics must state formula/procedure, units, desirable direction, invariances, domain,
dynamics model assumptions, computational cost, and failure modes. Keep feasibility,
difficulty, and diversity separate.

- [ ] **Step 4: Draft benchmark and reference-implementation recommendations**

Benchmark Protocol must specify:

1. large seeded training distributions per implemented generator family;
2. immutable generator version, config, seed list, rejection log, metrics, and splits;
3. feasible-first selection;
4. spectrum coverage using reviewed domain-specific coordinates;
5. diversity selection within spectrum regions;
6. held-out generator families or parameter regions for distribution shift;
7. a small smoke suite for expensive simulator tests;
8. simulator import and reset validation before policy evaluation;
9. uncertainty reporting when a scalar difficulty proxy is used.

State the default ambition of 10,000-100,000 training courses per generator and 100
evaluation courses per selected spectrum band or operating point per main domain as a
proposed protocol, not as completed artifacts. Explain that final bands derive from the
metric decision rather than universal labels.

Reference Implementations maps current TrackGen generators to taxonomy cells and
identifies uncovered segment, learned, maritime, and simulator-native families. Clearly
separate implemented capability, planned capability, and literature examples.

- [ ] **Step 5: Draft the reporting audit, open problems, and conclusion**

The reporting audit covers:

- generator name and version;
- full parameter/config distribution;
- seed policy;
- rejection and repair rate;
- train/validation/evaluation split;
- course metrics and difficulty controls;
- simulator and serialized format;
- fixed evaluation courses;
- code and asset availability.

Open Problems must tie each agenda item to an observed evidence gap and a testable
research question. Include dynamics-aware generation, sim-to-real distributions,
cross-simulator transfer, generator-curriculum co-design, recoverability, multi-agent
racing, maritime benchmarks, course cards, and generator-policy evaluation.

The Conclusion states a minimum reporting checklist and the role of OSS reference
implementations without claiming that TrackGen defines the field.

- [ ] **Step 6: Rewrite the abstract after all results are stable**

The final abstract contains:

- the precise scope;
- the corpus size and review dates;
- the two taxonomies;
- the main reporting/evaluation finding;
- the benchmark and OSS contribution;
- one sentence on open gaps.

Every number must be generated from validated data or traceable to a cited source.

- [ ] **Step 7: Validate claims and commit manuscript sections in coherent groups**

After each section group:

~~~bash
python3 paper/scripts/validate_corpus.py
latexmk -cd -pdf paper/main.tex
python3 paper/scripts/check_tex_log.py paper/build/main.log
~~~

Commit groups:

~~~bash
git add paper/sections paper/data/claims.csv
git commit -m "docs: draft survey framing and protocol"

git add paper/sections paper/data/claims.csv
git commit -m "docs: draft survey taxonomies and metrics"

git add paper/sections paper/data/claims.csv
git commit -m "docs: draft survey benchmark and research agenda"
~~~

Use only the relevant section files in each commit, even though the compact command above
shows the common staging set.

## Task 10: Build Reproducible Figures

**Files:**

- Create: paper/figures/corpus-flow.tex
- Create: paper/figures/corpus-counts.tex
- Create: paper/scripts/render_figure_data.py
- Create: paper/figures/taxonomy.tex
- Create: paper/figures/benchmark-pipeline.tex
- Modify: paper/sections/02-review-protocol.tex
- Modify: paper/sections/05-representations.tex
- Modify: paper/sections/06-generation-methods.tex
- Modify: paper/sections/09-benchmark-protocol.tex

- [ ] **Step 1: Create the corpus flow figure from validated counts**

Show discovery rows by stream, deduplicated candidates, screened sources, exclusions,
boundary sources, and included sources. Generate numeric labels from candidates.csv
rather than typing counts into the figure. The caption states the last search date and
explains that streams overlap.

Implement paper/scripts/render_figure_data.py:

~~~python
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


STATUS_MACROS = {
    "candidate": "CorpusCandidateCount",
    "included": "CorpusIncludedCount",
    "excluded": "CorpusExcludedCount",
    "boundary": "CorpusBoundaryCount",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render_counts(data_dir: Path, output: Path) -> None:
    candidates = _rows(data_dir / "candidates.csv")
    searches = _rows(data_dir / "search_log.csv")
    status_counts = Counter(row["screening_status"] for row in candidates)
    lines = [
        rf"\newcommand{{\CorpusDiscoveredCount}}{{{len(candidates)}}}",
        rf"\newcommand{{\CorpusLastSearchDate}}{{{max(row['search_date'] for row in searches)}}}",
    ]
    for status, macro in STATUS_MACROS.items():
        lines.append(
            rf"\newcommand{{\{macro}}}{{{status_counts.get(status, 0)}}}"
        )
    content = "\n".join(lines) + "\n"
    if not output.exists() or output.read_text() != content:
        output.write_text(content)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    render_counts(root / "data", root / "figures" / "corpus-counts.tex")
~~~

Add a test with a four-row candidate fixture that asserts exact macro values and
byte-identical output across two runs. Invoke this script from the Makefile tables target
after render_tables.py, and input corpus-counts.tex at the top of corpus-flow.tex.



- [ ] **Step 2: Create the taxonomy figure**

Use a compact matrix or layered TikZ diagram connecting:

~~~text
course object -> representation -> generation role/method -> validity strategy -> output format
~~~

Use ground, aerial, maritime, mixed, and adjacent as small domain annotations rather than
separate disconnected taxonomies. Ensure the figure remains legible at a single-column
width and in grayscale.

- [ ] **Step 3: Create the benchmark-selection pipeline**

Show:

~~~text
generator + versioned config + seeds
  -> raw course corpus
  -> feasibility and simulator validation
  -> metric computation
  -> difficulty-spectrum stratification
  -> diversity selection
  -> immutable training/evaluation splits
  -> policy evaluation with uncertainty
~~~

Represent simulator validation as a first-class gate, not a footnote.

- [ ] **Step 4: Add figures to the manuscript and verify accessibility**

Every figure requires:

- a caption that states the takeaway;
- all abbreviations expanded in the caption or nearby text;
- line styles or labels in addition to color;
- no font smaller than the manuscript footnote size;
- a text explanation that does not require seeing color.

- [ ] **Step 5: Build and visually inspect the PDF**

~~~bash
latexmk -cd -pdf paper/main.tex
python3 paper/scripts/check_tex_log.py paper/build/main.log
~~~

Inspect every figure at 100 percent and at a single-column print width. Resolve clipped
labels, unreadable nodes, and cross-reference placement before committing.

- [ ] **Step 6: Commit figures**

~~~bash
git add paper/figures paper/sections paper/scripts/render_figure_data.py paper/Makefile tests/test_survey_tables.py
git commit -m "docs: add survey taxonomy and benchmark figures"
~~~

## Task 11: Enforce Build, Citation, And Manuscript Quality Gates

**Files:**

- Create: paper/scripts/check_tex_log.py
- Modify: paper/Makefile
- Modify: tests/test_paper_artifacts.py
- Modify: tests/test_survey_corpus.py
- Modify: tests/test_survey_tables.py

- [ ] **Step 1: Write failing log-check tests**

Test that check_tex_log rejects logs containing:

~~~text
LaTeX Warning: Citation ... undefined
LaTeX Warning: Reference ... undefined
There were undefined references.
Label(s) may have changed. Rerun
multiply defined
Overfull \hbox
~~~

Test that it accepts a clean fixture and prints a one-line success message.

- [ ] **Step 2: Implement strict log checking**

Create paper/scripts/check_tex_log.py with:

~~~python
from __future__ import annotations

import re
import sys
from pathlib import Path


FAIL_PATTERNS = {
    "undefined citation": r"Citation .+ undefined",
    "undefined reference": r"Reference .+ undefined|There were undefined references",
    "unstable labels": r"Label\(s\) may have changed",
    "duplicate label": r"multiply defined",
    "overfull box": r"Overfull \\[hv]box",
}


def check_log(path: Path) -> None:
    lines = path.read_text(errors="replace").splitlines()
    failures: list[str] = []
    for label, pattern in FAIL_PATTERNS.items():
        regex = re.compile(pattern)
        for line_number, line in enumerate(lines, start=1):
            if regex.search(line):
                failures.append(f"{path}:{line_number}: {label}: {line.strip()}")
    if failures:
        raise ValueError("\n".join(failures))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_tex_log.py PATH", file=sys.stderr)
        return 2
    try:
        check_log(Path(argv[1]))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print("LaTeX log validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
~~~


- [ ] **Step 3: Add cross-artifact integrity checks**

Require:

- every cite command in TeX resolves to references.bib;
- every claims.csv cite_key resolves to an included or boundary source;
- every generated table citation resolves;
- every figure and table label is unique;
- every section is referenced by at least one label;
- generated tables are current;
- source files contain no unresolved markers;
- the PDF builds twice without changing generated tables.

- [ ] **Step 4: Run the complete quality gate**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_paper_artifacts.py \
  tests/test_survey_corpus.py \
  tests/test_survey_tables.py
make -C paper clean
make -C paper check
python3 paper/scripts/render_tables.py
git diff --exit-code paper/tables
~~~

Expected: all tests pass, the PDF builds, chktex reports no configured error, no
unresolved citation/reference/label or overfull box remains, and generated tables are
stable.

- [ ] **Step 5: Commit quality tooling**

~~~bash
git add paper/Makefile paper/scripts/check_tex_log.py tests
git commit -m "test: enforce survey artifact quality gates"
~~~

## Task 12: Independent Review, Documentation, And Follow-On Specifications

**Files:**

- Modify: docs/related-work/state-of-the-art.rst
- Modify: docs/related-work/prior-art.rst
- Modify: docs/index.rst
- Modify: paper/README.md
- Create: paper/reviews/citation-audit.md
- Create: paper/reviews/scope-audit.md
- Create: paper/reviews/manuscript-audit.md

- [ ] **Step 1: Run three independent review passes**

Citation audit prompt:

~~~text
Audit every quantitative, historical, priority, novelty, and comparative claim in the
manuscript. Check the cited primary source and evidence locator. Flag citation drift,
unsupported universals, wrong attribution, metadata errors, and claims that rely only on
secondary summaries. Do not rewrite prose; report section, sentence opening, severity,
and exact corrective action.
~~~

Scope audit prompt:

~~~text
Review the manuscript without using the TrackGen code as a definition of the field.
Look for missing terminology, vehicle domains, representation families, generation
roles, benchmark traditions, and simulator ecosystems. Pay special attention to maritime
work and methods found only by blind discovery. Report omissions, taxonomy forcing, and
places where reference implementations are mistaken for field boundaries.
~~~

Manuscript audit prompt:

~~~text
Review this as a candidate IJRR or Science Robotics survey. Evaluate the central thesis,
taxonomy durability, synthesis versus cataloguing, table usefulness, methodological
transparency, metric defensibility, benchmark actionability, and research-agenda
specificity. Flag repetitive prose, weak transitions, unsupported contribution claims,
and any figure or table that does not earn its space.
~~~

- [ ] **Step 2: Resolve findings with evidence**

Classify each finding as accepted, rejected with technical reason, or requiring user
decision. For accepted citation findings, update claims.csv and evidence.csv before
rewriting prose. For taxonomy findings, update the relevant decision record and rerun
reliability checks if coding categories change.

- [ ] **Step 3: Link repository documentation to the structured corpus**

Add a short note near the top of state-of-the-art.rst and prior-art.rst stating:

- these files are historical seed notes;
- the reviewed source ledger and coding live under paper/data;
- the manuscript lives under paper;
- new related work should enter candidates.csv and pass verification before becoming a
  survey claim.

Add a Related Work index entry in docs/index.rst and exact build commands in
paper/README.md.

- [ ] **Step 4: Write follow-on project briefs without freezing unsupported interfaces**

Add two “Next project” sections to paper/README.md:

Course corpus and benchmark release:

- consumes approved metrics.csv and 0004-difficulty-spectrum.md;
- generates 10,000-100,000 courses per selected generator/config distribution;
- publishes seeds, configs, rejection logs, metrics, splits, checksums, and course cards;
- produces difficulty-spectrum and diversity-coverage plots from released course metrics for inclusion in the manuscript;
- selects evaluation suites through feasibility, spectrum stratification, and diversity;
- requires a separate design spec and implementation plan.

Serialization and simulator adapters:

- consumes 0005-simulator-feasibility.md and simulators.csv;
- starts with a versioned generic JSON/NPZ schema;
- validates import, units, frames, collisions, and spawn/reset semantics;
- prioritizes adapters based on evidence and target experiments;
- requires a separate design spec and implementation plan.

- [ ] **Step 5: Run final verification from a clean paper build**

~~~bash
.venv/bin/python -m pytest -q
make -C paper clean
make -C paper check
git status --short
~~~

Expected: all project tests pass; the paper quality gate passes; git status contains only
the intended survey changes and any pre-existing user-owned untracked files.

- [ ] **Step 6: Commit documentation and review resolutions**

~~~bash
git add docs/related-work docs/index.rst paper
git commit -m "docs: complete survey foundation and review"
~~~

## Final Acceptance Checklist

- [ ] The blind agents received no current-corpus titles, paths, or taxonomy labels.
- [ ] Corpus-aware agents used all five bootstrap documents and expanded beyond them.
- [ ] Every named bootstrap source is linked to a screened candidate or a recorded exclusion.
- [ ] Every included source has verified metadata, a BibTeX entry, coded evidence, and a direct locator.
- [ ] Independent coding reaches the stated reliability threshold or records and resolves disagreement.
- [ ] Search streams meet the two-round saturation rule and sparse taxonomy cells receive targeted searches.
- [ ] Difficulty is represented as an evidence-backed spectrum, with feasibility kept separate.
- [ ] Simulation feasibility covers import, geometry, units, frames, collisions, reset, determinism, and RL access.
- [ ] Tables are generated from validated data and remain deterministic.
- [ ] Manuscript claims map to the claim ledger and primary sources.
- [ ] TrackGen is presented as reference implementation evidence, not the survey boundary.
- [ ] The benchmark protocol distinguishes proposed release targets from completed artifacts.
- [ ] The complete LaTeX build has no unresolved citation/reference/label or overfull-box failures.
- [ ] Independent citation, scope, and manuscript audits are resolved.
- [ ] Follow-on benchmark-data and simulator-adapter projects have explicit inputs and separate planning gates.
