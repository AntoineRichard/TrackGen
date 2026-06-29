from __future__ import annotations

import csv
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "paper" / "data"
AGENT_RUN_DIR = DATA_DIR / "agent_runs"
SEARCH_LOG_PATH = DATA_DIR / "search_log.csv"

COMMON_HEADER = (
    "candidate_id",
    "cite_key",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
    "discovery_stream",
    "discovery_query",
    "discovery_agent",
    "screening_status",
    "exclusion_reason",
    "metadata_status",
    "metadata_evidence",
    "domain",
    "vehicle",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "geometry_metrics",
    "difficulty_metrics",
    "diversity_metrics",
    "training_distribution",
    "evaluation_suite",
    "simulator",
    "export_format",
    "code_status",
    "asset_status",
    "reproducibility_fields",
    "evidence_locator",
    "coding_notes",
)

RUN_SPECS = {
    "blind-ground": {
        "stream": "blind-ground",
        "agent": "blind-ground",
        "rows": 45,
        "retained": 45,
        "excluded": 0,
    },
    "blind-aerial-maritime": {
        "stream": "blind-aerial-maritime",
        "agent": "blind-aerial-maritime",
        "rows": 32,
        "retained": 32,
        "excluded": 0,
    },
    "aware-geometry-rl": {
        "stream": "aware-geometry-rl",
        "agent": "aware-geometry-rl",
        "rows": 55,
        "retained": 51,
        "excluded": 4,
    },
    "aware-simulation-benchmarks": {
        "stream": "aware-simulation",
        "agent": "aware-simulation-benchmarks",
        "rows": 30,
        "retained": 30,
        "excluded": 0,
    },
}

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
INLINE_QUERY = re.compile(
    r"^-\s+" + chr(96) + r"([^\x60]+)" + chr(96) + r"\s*$"
)


@pytest.fixture
def agent_run_dir(tmp_path: Path) -> Path:
    return Path(shutil.copytree(AGENT_RUN_DIR, tmp_path / "agent_runs"))


def validate_agent_runs(path: Path) -> None:
    module = importlib.import_module("paper.scripts.validate_agent_runs")
    module.validate_agent_runs(path)


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        return tuple(reader.fieldnames or ()), list(reader)


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    header: tuple[str, ...] = COMMON_HEADER,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def mutate_row(
    data_dir: Path,
    slug: str,
    row_index: int,
    **updates: str,
) -> None:
    path = data_dir / f"{slug}.csv"
    _, rows = read_csv(path)
    rows[row_index].update(updates)
    write_csv(path, rows)


def extract_report_queries(path: Path) -> list[tuple[str, str]]:
    headings: list[tuple[int, str]] = []
    queries: list[tuple[str, str]] = []
    fenced_lines: list[str] = []
    fenced_section = ""
    capture_fence = False
    in_fence = False

    def query_context() -> bool:
        context = " ".join(text.casefold() for _, text in headings)
        return any(word in context for word in ("quer", "refinement", "saturation"))

    def section_name() -> str:
        return " / ".join(text for level, text in headings if level >= 2)

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING.match(line)
        if heading and not in_fence:
            level = len(heading.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading.group(2).strip()))
            continue

        if line.startswith(chr(96) * 3):
            if not in_fence:
                in_fence = True
                capture_fence = query_context()
                fenced_section = section_name()
                fenced_lines = []
            else:
                if capture_fence:
                    queries.extend(
                        (fenced_section, query)
                        for query in fenced_lines
                        if query.strip()
                    )
                in_fence = False
                capture_fence = False
                fenced_lines = []
            continue

        if in_fence:
            if capture_fence:
                fenced_lines.append(line)
            continue

        inline = INLINE_QUERY.fullmatch(line)
        if inline and query_context():
            queries.append((section_name(), inline.group(1)))

    return queries


def read_search_log() -> list[dict[str, str]]:
    with SEARCH_LOG_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_production_agent_runs_validate():
    validate_agent_runs(AGENT_RUN_DIR)


def test_cli_validates_default_agent_run_directory():
    result = subprocess.run(
        [sys.executable, "-m", "paper.scripts.validate_agent_runs"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "agent discovery validation passed"


@pytest.mark.parametrize("suffix", [".csv", ".md"])
def test_exact_run_pairs_are_required(agent_run_dir: Path, suffix: str):
    (agent_run_dir / f"blind-ground{suffix}").unlink()

    with pytest.raises(ValueError, match="expected exactly"):
        validate_agent_runs(agent_run_dir)


def test_unexpected_agent_run_file_is_rejected(agent_run_dir: Path):
    (agent_run_dir / "extra-run.csv").write_text(
        ",".join(COMMON_HEADER) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected.*extra-run.csv"):
        validate_agent_runs(agent_run_dir)


def test_csv_header_must_match_exactly(agent_run_dir: Path):
    path = agent_run_dir / "blind-ground.csv"
    _, rows = read_csv(path)
    wrong_header = COMMON_HEADER[:-1] + ("notes",)
    for row in rows:
        row["notes"] = row.pop("coding_notes")
    write_csv(path, rows, wrong_header)

    with pytest.raises(ValueError, match="header"):
        validate_agent_runs(agent_run_dir)


def test_csv_rows_must_have_uniform_width(agent_run_dir: Path):
    path = agent_run_dir / "blind-ground.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[1].append("unexpected")
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    with pytest.raises(ValueError, match="malformed|35 columns"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    ("slug", "row_index", "candidate_id"),
    [
        ("blind-ground", 0, "WRONG-001"),
        ("blind-aerial-maritime", 0, "BAAM-X001"),
        ("aware-geometry-rl", 0, "AGRL-0001"),
        ("aware-simulation-benchmarks", 0, "ASIM-0001"),
    ],
)
def test_candidate_ids_require_run_prefix_and_format(
    agent_run_dir: Path,
    slug: str,
    row_index: int,
    candidate_id: str,
):
    mutate_row(agent_run_dir, slug, row_index, candidate_id=candidate_id)

    with pytest.raises(ValueError, match="candidate_id"):
        validate_agent_runs(agent_run_dir)


def test_candidate_ids_must_be_unique(agent_run_dir: Path):
    _, rows = read_csv(agent_run_dir / "blind-ground.csv")
    mutate_row(
        agent_run_dir,
        "blind-ground",
        1,
        candidate_id=rows[0]["candidate_id"],
    )

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        validate_agent_runs(agent_run_dir)


def test_candidate_ids_must_be_sequential(agent_run_dir: Path):
    mutate_row(agent_run_dir, "blind-ground", 1, candidate_id="BG-099")

    with pytest.raises(ValueError, match="sequential"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("field", ["title", "metadata_evidence", "evidence_locator"])
def test_required_evidence_fields_must_be_nonempty(
    agent_run_dir: Path,
    field: str,
):
    mutate_row(agent_run_dir, "blind-ground", 0, **{field: ""})

    with pytest.raises(ValueError, match=field):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    ("slug", "field"),
    [
        ("blind-ground", "discovery_stream"),
        ("blind-ground", "discovery_agent"),
        ("blind-aerial-maritime", "discovery_stream"),
        ("blind-aerial-maritime", "discovery_agent"),
        ("aware-geometry-rl", "discovery_stream"),
        ("aware-geometry-rl", "discovery_agent"),
        ("aware-simulation-benchmarks", "discovery_stream"),
        ("aware-simulation-benchmarks", "discovery_agent"),
    ],
)
def test_discovery_stream_and_agent_follow_run_contract(
    agent_run_dir: Path,
    slug: str,
    field: str,
):
    mutate_row(agent_run_dir, slug, 0, **{field: "unexpected-run"})

    with pytest.raises(ValueError, match=field):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("field", ["screening_status", "metadata_status"])
def test_operational_status_values_are_controlled(
    agent_run_dir: Path,
    field: str,
):
    mutate_row(agent_run_dir, "aware-geometry-rl", 0, **{field: "maybe"})

    with pytest.raises(ValueError, match=field):
        validate_agent_runs(agent_run_dir)


def test_duplicate_doi_is_rejected_after_normalization(agent_run_dir: Path):
    path = agent_run_dir / "blind-ground.csv"
    _, rows = read_csv(path)
    source = next(row for row in rows if row["doi"] != "NR")
    target = next(row for row in rows if row is not source and row["doi"] != "NR")
    target["doi"] = "https://doi.org/" + source["doi"].upper()
    write_csv(path, rows)

    with pytest.raises(ValueError, match="duplicate DOI"):
        validate_agent_runs(agent_run_dir)


def test_duplicate_normalized_title_is_rejected(agent_run_dir: Path):
    path = agent_run_dir / "blind-ground.csv"
    _, rows = read_csv(path)
    rows[1]["title"] = re.sub(r"[^A-Za-z0-9]", " ", rows[0]["title"]).upper()
    write_csv(path, rows)

    with pytest.raises(ValueError, match="duplicate normalized title"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("value", ["ground;aerial", "ground;  aerial", "ground;"])
def test_list_fields_require_canonical_semicolon_spacing(
    agent_run_dir: Path,
    value: str,
):
    mutate_row(agent_run_dir, "blind-ground", 0, domain=value)

    with pytest.raises(ValueError, match="domain.*semicolon|domain.*list"):
        validate_agent_runs(agent_run_dir)


def test_list_fields_reject_nr_combined_with_another_value(
    agent_run_dir: Path,
):
    mutate_row(agent_run_dir, "blind-ground", 0, domain="NR; ground")

    with pytest.raises(ValueError, match="domain.*NR.*alone"):
        validate_agent_runs(agent_run_dir)


def test_blind_fields_do_not_use_taxonomy_vocabulary(agent_run_dir: Path):
    mutate_row(
        agent_run_dir,
        "blind-ground",
        0,
        domain="custom transfer domain",
        generator_family="custom generator description",
    )

    validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("field", ["code_status", "asset_status"])
@pytest.mark.parametrize("value", ["open source repository", "official_open; NR"])
def test_code_and_asset_statuses_reject_prose_or_lists(
    agent_run_dir: Path,
    field: str,
    value: str,
):
    mutate_row(agent_run_dir, "blind-ground", 0, **{field: value})

    with pytest.raises(ValueError, match=field):
        validate_agent_runs(agent_run_dir)


def test_all_scalar_code_and_asset_status_values_are_allowed(
    agent_run_dir: Path,
):
    mutate_row(
        agent_run_dir,
        "blind-ground",
        0,
        code_status="unofficial_open",
        asset_status="closed",
    )
    mutate_row(
        agent_run_dir,
        "blind-ground",
        1,
        code_status="not_applicable",
        asset_status="NR",
    )

    validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    ("category", "headings"),
    [
        ("search surfaces", ("## Search surfaces",)),
        ("queries", ("## Exact queries",)),
        ("boundary", ("## Inclusion and boundary judgments",)),
        ("terminology", ("## Terminology absent from the supplied brief",)),
        ("sparse", ("## Sparse and contradictory areas",)),
        ("saturation", ("## Saturation arithmetic",)),
        (
            "retrieval or limitations",
            ("## High-priority manual retrieval", "## Verification limitations"),
        ),
        (
            "validation",
            ("## Counts and validation", "## Verification limitations"),
        ),
    ],
)
def test_reports_require_each_documentation_section(
    agent_run_dir: Path,
    category: str,
    headings: tuple[str, ...],
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    for index, heading in enumerate(headings):
        assert heading in text
        text = text.replace(heading, f"## Documentation section {index}", 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=category):
        validate_agent_runs(agent_run_dir)


def test_report_heading_matching_accepts_clear_wording_variants(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "## Search surfaces": "## Discovery search channels",
        "## Exact queries": "## Executed search-query record",
        "## Inclusion and boundary judgments": "## Scope and screening decisions",
        "## Terminology absent from the supplied brief": "## Vocabulary encountered",
        "## Sparse and contradictory areas": "## Evidence gaps and conflicts",
        "## Saturation arithmetic": "## Stopping rule and saturation yield",
        "## High-priority manual retrieval": "## Retrieval gaps",
        "## Counts and validation": "## Schema validation checks",
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")

    validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    "surface",
    [
        "https://en.wikipedia.org/wiki/Racing",
        "https://www.google.com/search?q=track+generation",
        "https://medium.com/example/track-generation",
        "https://www.semanticscholar.org/paper/example",
    ],
)
@pytest.mark.parametrize("field", ["metadata_evidence", "evidence_locator"])
def test_secondary_evidence_surfaces_are_forbidden(
    agent_run_dir: Path,
    surface: str,
    field: str,
):
    mutate_row(agent_run_dir, "blind-ground", 0, **{field: surface})

    with pytest.raises(ValueError, match="forbidden secondary evidence"):
        validate_agent_runs(agent_run_dir)


def test_search_log_covers_every_report_query_in_order():
    log_rows = read_search_log()

    for slug, spec in RUN_SPECS.items():
        report_path = f"paper/data/agent_runs/{slug}.md"
        expected = extract_report_queries(ROOT / report_path)
        assert expected, report_path
        actual = [
            row
            for row in log_rows
            if report_path in row["notes"]
            and not row["query"].startswith("RUN-SUMMARY:")
        ]
        assert [row["query"] for row in actual] == [
            query for _, query in expected
        ]
        for row, (section, _) in zip(actual, expected, strict=True):
            assert row["search_date"] == "2026-06-29"
            assert row["stream"] == spec["stream"]
            assert row["agent"] == spec["agent"]
            assert row["search_surface"] == "mixed-primary-web"
            assert row["results_screened"] == "NR"
            assert row["candidates_added"] == "NR"
            assert section in row["notes"]
            assert re.search(
                r"counts? (?:was|were) not captured",
                row["notes"],
                re.IGNORECASE,
            )


def test_search_log_has_one_truthful_summary_per_run():
    log_rows = read_search_log()

    for slug, spec in RUN_SPECS.items():
        report_path = f"paper/data/agent_runs/{slug}.md"
        query = f"RUN-SUMMARY:{report_path}"
        summaries = [row for row in log_rows if row["query"] == query]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["search_date"] == "2026-06-29"
        assert summary["stream"] == spec["stream"]
        assert summary["agent"] == spec["agent"]
        assert summary["search_surface"] == "documented-agent-run"
        assert summary["results_screened"] == "NR"
        assert int(summary["candidates_added"]) == spec["rows"]
        notes = summary["notes"].casefold()
        assert f"{spec['retained']} retained" in notes
        assert f"{spec['excluded']} excluded" in notes
        assert "saturation" in notes
        assert "total screened-hit count was not captured" in notes


def test_search_log_summary_counts_match_output_csv_rows():
    log_rows = read_search_log()

    for slug in RUN_SPECS:
        _, output_rows = read_csv(AGENT_RUN_DIR / f"{slug}.csv")
        summary_query = f"RUN-SUMMARY:paper/data/agent_runs/{slug}.md"
        summary = next(row for row in log_rows if row["query"] == summary_query)
        assert int(summary["candidates_added"]) == len(output_rows)


def test_search_log_ids_are_unique_sequential_and_no_action_disappears():
    log_rows = read_search_log()
    ids = [row["search_id"] for row in log_rows]
    assert len(ids) == len(set(ids))
    assert ids == [f"S{number:04d}" for number in range(1, len(ids) + 1)]

    expected_task_rows = sum(
        len(extract_report_queries(AGENT_RUN_DIR / f"{slug}.md"))
        for slug in RUN_SPECS
    ) + len(RUN_SPECS)
    assert len(log_rows) - 5 == expected_task_rows


def test_make_validate_runs_both_corpus_validators():
    makefile = (ROOT / "paper" / "Makefile").read_text(encoding="utf-8")
    validate_target = makefile.split("validate:", 1)[1].split("\n\n", 1)[0]

    assert "validate_corpus.py" in validate_target
    assert "validate_agent_runs.py" in validate_target
