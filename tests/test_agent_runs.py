from __future__ import annotations

import csv
import importlib
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "paper" / "data"
AGENT_RUN_DIR = DATA_DIR / "agent_runs"
SEARCH_LOG_PATH = DATA_DIR / "search_log.csv"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
CANDIDATES_PATH = DATA_DIR / "candidates.csv"

SEARCH_LOG_HEADER = (
    "search_id",
    "search_date",
    "stream",
    "agent",
    "query",
    "search_surface",
    "results_screened",
    "candidates_added",
    "notes",
)

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
    data_dir = tmp_path / "data"
    run_dir = Path(shutil.copytree(AGENT_RUN_DIR, data_dir / "agent_runs"))
    shutil.copy2(SEARCH_LOG_PATH, data_dir / "search_log.csv")
    shutil.copy2(TAXONOMY_PATH, data_dir / "taxonomy.json")
    shutil.copy2(CANDIDATES_PATH, data_dir / "candidates.csv")
    complete_fixture_search_log(run_dir)
    return run_dir


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


def read_search_log(
    path: Path = SEARCH_LOG_PATH,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_search_log(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SEARCH_LOG_HEADER,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def complete_fixture_search_log(run_dir: Path) -> None:
    path = run_dir.parent / "search_log.csv"
    rows = read_search_log(path)
    remaining = Counter(
        (row["stream"], row["agent"], row["query"])
        for row in rows
        if row["search_surface"] == "mixed-primary-web"
    )
    next_id = max(int(row["search_id"][1:]) for row in rows) + 1

    for slug, spec in RUN_SPECS.items():
        report_path = f"paper/data/agent_runs/{slug}.md"
        for section, query in extract_report_queries(run_dir / f"{slug}.md"):
            key = (spec["stream"], spec["agent"], query)
            if remaining[key]:
                remaining[key] -= 1
                continue
            rows.append(
                {
                    "search_id": f"S{next_id:04d}",
                    "search_date": "2026-06-29",
                    "stream": spec["stream"],
                    "agent": spec["agent"],
                    "query": query,
                    "search_surface": "mixed-primary-web",
                    "results_screened": "NR",
                    "candidates_added": "NR",
                    "notes": (
                        f"Source: {report_path}; section: {section}. "
                        "Per-query screened-hit and candidate-add counts "
                        "were not captured."
                    ),
                }
            )
            next_id += 1

    write_search_log(path, rows)


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
        (
            "queries",
            (
                "## Exact queries",
                "### Exclusion-lead verification queries",
            ),
        ),
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
    log_path = agent_run_dir.parent / "search_log.csv"
    rows = read_search_log(log_path)
    for row in rows:
        if row["agent"] != "aware-geometry-rl":
            continue
        row["notes"] = row["notes"].replace(
            "section: Exact queries",
            "section: Executed search-query record",
        )
    write_search_log(log_path, rows)

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
        expected_queries = [query for _, query in expected]
        assert Counter(row["query"] for row in actual) == Counter(
            expected_queries
        )
        assert query_batches_follow_report_order(actual, expected_queries)
        for row in actual:
            assert row["search_date"] == "2026-06-29"
            assert row["stream"] == spec["stream"]
            assert row["agent"] == spec["agent"]
            assert row["search_surface"] == "mixed-primary-web"
            assert row["results_screened"] == "NR"
            assert row["candidates_added"] == "NR"
            sections = [
                section
                for section, query in expected
                if query == row["query"]
            ]
            assert any(section in row["notes"] for section in sections)
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


def fixture_search_log_path(run_dir: Path) -> Path:
    return run_dir.parent / "search_log.csv"


def exact_log_indexes(
    rows: list[dict[str, str]],
    slug: str,
) -> list[int]:
    spec = RUN_SPECS[slug]
    return [
        index
        for index, row in enumerate(rows)
        if row["stream"] == spec["stream"]
        and row["agent"] == spec["agent"]
        and row["search_surface"] == "mixed-primary-web"
    ]


def query_batches_follow_report_order(
    rows: list[dict[str, str]],
    expected_queries: list[str],
) -> bool:
    batches: list[list[str]] = []
    previous_id: int | None = None
    for row in rows:
        search_id = int(row["search_id"][1:])
        if previous_id is None or search_id != previous_id + 1:
            batches.append([])
        batches[-1].append(row["query"])
        previous_id = search_id

    for batch in batches:
        position = 0
        for query in batch:
            while (
                position < len(expected_queries)
                and expected_queries[position] != query
            ):
                position += 1
            if position == len(expected_queries):
                return False
            position += 1
    return True


def test_runtime_rejects_removed_report_query(agent_run_dir: Path):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    del rows[exact_log_indexes(rows, "blind-ground")[0]]
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="exact-query.*missing"):
        validate_agent_runs(agent_run_dir)


def test_runtime_rejects_extra_exact_query(agent_run_dir: Path):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    extra = dict(rows[exact_log_indexes(rows, "blind-ground")[0]])
    extra.update(search_id="S9999", query="undocumented extra query")
    rows.append(extra)
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="exact-query.*extra"):
        validate_agent_runs(agent_run_dir)


def test_runtime_rejects_reordered_query_batch(agent_run_dir: Path):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    first, second = exact_log_indexes(rows, "blind-ground")[:2]
    rows[first]["query"], rows[second]["query"] = (
        rows[second]["query"],
        rows[first]["query"],
    )
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="report order"):
        validate_agent_runs(agent_run_dir)


def test_runtime_preserves_duplicate_query_executions(agent_run_dir: Path):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    indexes = exact_log_indexes(rows, "aware-simulation-benchmarks")
    counts = Counter(rows[index]["query"] for index in indexes)
    duplicated_query = next(query for query, count in counts.items() if count > 1)
    del rows[next(index for index in indexes if rows[index]["query"] == duplicated_query)]
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="exact-query.*missing"):
        validate_agent_runs(agent_run_dir)


def test_runtime_rejects_query_attributed_to_wrong_agent(agent_run_dir: Path):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    index = exact_log_indexes(rows, "blind-ground")[0]
    rows[index]["agent"] = "different-agent"
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="exact-query.*missing"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_runtime_requires_exactly_one_run_summary(
    agent_run_dir: Path,
    mutation: str,
):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    summary_query = (
        "RUN-SUMMARY:paper/data/agent_runs/blind-ground.md"
    )
    index = next(
        index for index, row in enumerate(rows) if row["query"] == summary_query
    )
    if mutation == "missing":
        del rows[index]
    else:
        duplicate = dict(rows[index])
        duplicate["search_id"] = "S9999"
        rows.append(duplicate)
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="RUN-SUMMARY.*exactly one"):
        validate_agent_runs(agent_run_dir)


def test_runtime_summary_count_matches_csv_rows(agent_run_dir: Path):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    summary_query = (
        "RUN-SUMMARY:paper/data/agent_runs/blind-ground.md"
    )
    summary = next(row for row in rows if row["query"] == summary_query)
    summary["candidates_added"] = "44"
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="candidates_added.*CSV row count"):
        validate_agent_runs(agent_run_dir)


def test_candidate_discovery_query_must_resolve_to_query_ledgers(
    agent_run_dir: Path,
):
    mutate_row(
        agent_run_dir,
        "blind-ground",
        0,
        discovery_query="undocumented candidate discovery query",
    )

    with pytest.raises(ValueError, match="discovery_query.*query ledger"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    "field",
    [
        "domain",
        "course_object",
        "representation_family",
        "generator_family",
        "generation_role",
        "validity_strategy",
    ],
)
def test_aware_controlled_fields_use_taxonomy(
    agent_run_dir: Path,
    field: str,
):
    mutate_row(
        agent_run_dir,
        "aware-geometry-rl",
        0,
        **{field: "outside_taxonomy"},
    )

    with pytest.raises(ValueError, match=rf"{field}.*taxonomy"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    "field",
    [
        "domain",
        "course_object",
        "representation_family",
        "generator_family",
        "generation_role",
        "validity_strategy",
    ],
)
def test_aware_controlled_fields_allow_sole_nr(
    agent_run_dir: Path,
    field: str,
):
    mutate_row(agent_run_dir, "aware-geometry-rl", 0, **{field: "NR"})

    validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    "field",
    [
        "authors",
        "year",
        "doi",
        "url",
        "discovery_query",
        "domain",
        "coding_notes",
    ],
)
def test_factual_fields_use_nr_instead_of_blank(
    agent_run_dir: Path,
    field: str,
):
    mutate_row(agent_run_dir, "blind-ground", 0, **{field: ""})

    with pytest.raises(ValueError, match=rf"{field}.*NR rather than blank"):
        validate_agent_runs(agent_run_dir)


def test_cite_key_may_use_nr(agent_run_dir: Path):
    mutate_row(agent_run_dir, "blind-ground", 0, cite_key="NR")

    validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("reason", ["", "NR"])
def test_excluded_rows_require_specific_reason(
    agent_run_dir: Path,
    reason: str,
):
    _, rows = read_csv(agent_run_dir / "aware-geometry-rl.csv")
    row_index = next(
        index
        for index, row in enumerate(rows)
        if row["screening_status"] == "excluded"
    )
    mutate_row(
        agent_run_dir,
        "aware-geometry-rl",
        row_index,
        exclusion_reason=reason,
    )

    with pytest.raises(ValueError, match="exclusion_reason"):
        validate_agent_runs(agent_run_dir)


def test_aware_coding_notes_require_discovery_provenance(
    agent_run_dir: Path,
):
    mutate_row(
        agent_run_dir,
        "aware-simulation-benchmarks",
        0,
        coding_notes="Technical interpretation only.",
    )

    with pytest.raises(ValueError, match="coding_notes.*provenance"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    "locator",
    ["primary paper", "https:///missing-host"],
)
def test_evidence_locator_requires_precise_shape(
    agent_run_dir: Path,
    locator: str,
):
    mutate_row(
        agent_run_dir,
        "blind-ground",
        0,
        evidence_locator=locator,
    )

    with pytest.raises(ValueError, match="evidence_locator.*precise locator"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    "locator",
    [
        "p. 7",
        "Section 3.2",
        "Table 2",
        "Figure 4",
        "Appendix A",
        "source path: src/generator.py",
        "lines 10-20",
    ],
)
def test_evidence_locator_accepts_precise_non_url_markers(
    agent_run_dir: Path,
    locator: str,
):
    mutate_row(
        agent_run_dir,
        "blind-ground",
        0,
        evidence_locator=locator,
    )

    validate_agent_runs(agent_run_dir)


def fixture_summary(
    rows: list[dict[str, str]],
    slug: str,
) -> dict[str, str]:
    query = f"RUN-SUMMARY:paper/data/agent_runs/{slug}.md"
    return next(row for row in rows if row["query"] == query)


@pytest.mark.parametrize(
    ("slug", "row_index", "value"),
    [
        ("aware-geometry-rl", 0, "automatic"),
        ("aware-geometry-rl", 21, "asfault"),
        ("aware-geometry-rl", 0, "bootstrap seed C0009"),
        ("aware-simulation-benchmarks", 0, "CarRacing"),
    ],
)
def test_aware_discovery_query_requires_explicit_provenance_grammar(
    agent_run_dir: Path,
    slug: str,
    row_index: int,
    value: str,
):
    mutate_row(
        agent_run_dir,
        slug,
        row_index,
        discovery_query=value,
    )

    with pytest.raises(ValueError, match="provenance grammar"):
        validate_agent_runs(agent_run_dir)


def test_query_provenance_requires_an_exact_report_and_log_literal(
    agent_run_dir: Path,
):
    mutate_row(
        agent_run_dir,
        "aware-geometry-rl",
        11,
        discovery_query=(
            "query::procedural generation road paths driving simulation fabricated"
        ),
    )

    with pytest.raises(ValueError, match="query::.*exact-query"):
        validate_agent_runs(agent_run_dir)


def test_seed_provenance_requires_an_existing_bootstrap_candidate(
    agent_run_dir: Path,
):
    candidate_path = agent_run_dir.parent / "candidates.csv"
    header, candidates = read_csv(candidate_path)
    candidates = [
        row for row in candidates if row["candidate_id"] != "C0009"
    ]
    write_csv(candidate_path, candidates, header)

    with pytest.raises(ValueError, match="seed::C0009.*bootstrap candidate"):
        validate_agent_runs(agent_run_dir)


def test_seed_provenance_requires_matching_report_ledger_relationship(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    old = "| AGRL0001 | " + chr(96) + "seed::C0009" + chr(96) + " |"
    new = "| AGRL0001 | " + chr(96) + "seed::C0010" + chr(96) + " |"
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match="seed::C0009.*provenance ledger"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("value", ["citation::", "citation::not stable"])
def test_citation_provenance_requires_a_stable_source_identifier(
    agent_run_dir: Path,
    value: str,
):
    mutate_row(
        agent_run_dir,
        "aware-geometry-rl",
        3,
        discovery_query=value,
    )

    with pytest.raises(ValueError, match="citation::.*stable source identifier"):
        validate_agent_runs(agent_run_dir)


def test_citation_provenance_requires_matching_report_ledger_relationship(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    old = (
        "| AGRL0004 | "
        + chr(96)
        + "citation::10.1109/tciaig.2011.2163692"
        + chr(96)
        + " |"
    )
    new = (
        "| AGRL0004 | "
        + chr(96)
        + "citation::10.1145/3368089.3409730"
        + chr(96)
        + " |"
    )
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="citation::10.*provenance ledger",
    ):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    ("field", "old", "new", "message"),
    [
        ("results_screened", None, "0", "results_screened.*NR"),
        (
            "notes",
            "Source: paper/data/agent_runs/blind-ground.md",
            "Source: paper/data/agent_runs/aware-geometry-rl.md",
            "source path",
        ),
        ("notes", "45 retained", "44 retained", "retained"),
        ("notes", "0 excluded", "1 excluded", "excluded"),
        (
            "notes",
            "round=R2 added=1 denominator=45 "
            "cumulative_retained=45 percent=2.22%",
            "round=R2 added=9 denominator=45 "
            "cumulative_retained=45 percent=20.00%",
            "saturation",
        ),
        (
            "notes",
            "round=R3 added=0 denominator=45 "
            "cumulative_retained=45 percent=0.00%",
            "round=R3 added=1 denominator=45 "
            "cumulative_retained=45 percent=2.22%",
            "saturation",
        ),
        (
            "notes",
            "Total screened-hit count was not captured.",
            "Count was not captured.",
            "screened-hit count.*not captured",
        ),
    ],
)
def test_runtime_summary_rejects_incorrect_structured_semantics(
    agent_run_dir: Path,
    field: str,
    old: str | None,
    new: str,
    message: str,
):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    summary = fixture_summary(rows, "blind-ground")
    if field == "notes":
        assert old is not None and old in summary["notes"]
        summary["notes"] = summary["notes"].replace(old, new, 1)
    else:
        summary[field] = new
    write_search_log(path, rows)

    with pytest.raises(ValueError, match=message):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "round=R2 added=1 denominator=45 "
            "cumulative_retained=45 percent=2.22%",
            "round=R2 added=1 denominator=100 "
            "cumulative_retained=45 percent=1.00%",
        ),
        (
            "round=R3 added=0 denominator=45 "
            "cumulative_retained=45 percent=0.00%",
            "round=R3 added=0 denominator=100 "
            "cumulative_retained=45 percent=0.00%",
        ),
    ],
)
def test_runtime_summary_saturation_must_match_canonical_report(
    agent_run_dir: Path,
    old: str,
    new: str,
):
    path = agent_run_dir / "blind-ground.md"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match="final saturation.*canonical"):
        validate_agent_runs(agent_run_dir)


SaturationFixture = tuple[str, int, int, int, str]

GROUND_FINAL_SATURATION: tuple[SaturationFixture, ...] = (
    ("R2", 1, 45, 45, "2.22"),
    ("R3", 0, 45, 45, "0.00"),
)
SATURATION_SECTION_PATTERN = re.compile(
    r"\n## Canonical final-round record\n\n"
    + re.escape(chr(96) * 3)
    + r"final-saturation\n.*?\n"
    + re.escape(chr(96) * 3)
    + r"\n",
    re.DOTALL,
)


def render_saturation_section(
    records: tuple[SaturationFixture, ...],
    heading: str = "Canonical final-round record",
) -> str:
    fence = chr(96) * 3
    lines = "\n".join(
        f"round={round_id} added={added} denominator={denominator} "
        f"cumulative_retained={cumulative} percent={percent}%"
        for round_id, added, denominator, cumulative, percent in records
    )
    return (
        f"## {heading}\n\n"
        f"{fence}final-saturation\n{lines}\n{fence}\n"
    )


def replace_report_saturation(
    path: Path,
    records: tuple[SaturationFixture, ...],
    heading: str = "Canonical final-round record",
) -> None:
    text = path.read_text(encoding="utf-8")
    section = render_saturation_section(records, heading)
    if SATURATION_SECTION_PATTERN.search(text):
        text = SATURATION_SECTION_PATTERN.sub("\n" + section, text, count=1)
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def remove_report_saturation(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = SATURATION_SECTION_PATTERN.sub("\n", text, count=1)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def set_summary_saturation(
    summary: dict[str, str],
    records: tuple[SaturationFixture, ...],
) -> None:
    encoded = "; ".join(
        f"round={round_id} added={added} denominator={denominator} "
        f"cumulative_retained={cumulative} percent={percent}%"
        for round_id, added, denominator, cumulative, percent in records
    )
    replacement = f"final saturation: {encoded}. Total screened-hit"
    notes = re.sub(
        r"final saturation(?: arithmetic)?: .*?\. Total screened-hit",
        replacement,
        summary["notes"],
    )
    assert notes != summary["notes"]
    summary["notes"] = notes


def test_runtime_rejects_an_earlier_nonfinal_ratio(
    agent_run_dir: Path,
):
    report_path = agent_run_dir / "blind-ground.md"
    text = report_path.read_text(encoding="utf-8")
    log_path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(log_path)
    summary = fixture_summary(rows, "blind-ground")

    diagnostic = "Earlier nonfinal diagnostic ratio: 1/100 = 1.00%."
    report_path.write_text(
        text.rstrip() + "\n\n" + diagnostic + "\n",
        encoding="utf-8",
    )
    if "## Canonical final-round record" in text:
        records = (
            ("R2", 1, 100, 45, "1.00"),
            GROUND_FINAL_SATURATION[1],
        )
        set_summary_saturation(summary, records)
    else:
        summary["notes"] = summary["notes"].replace(
            "1/45 = 2.22%",
            "1/100 = 1.00%",
            1,
        )
    write_search_log(log_path, rows)

    with pytest.raises(ValueError, match="final saturation.*canonical"):
        validate_agent_runs(agent_run_dir)


def test_runtime_recomputes_final_saturation_percentage(
    agent_run_dir: Path,
):
    report_path = agent_run_dir / "blind-ground.md"
    text = report_path.read_text(encoding="utf-8")
    log_path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(log_path)
    summary = fixture_summary(rows, "blind-ground")

    if "## Canonical final-round record" in text:
        records = (
            ("R2", 1, 45, 45, "99.99"),
            GROUND_FINAL_SATURATION[1],
        )
        replace_report_saturation(report_path, records)
        set_summary_saturation(summary, records)
    else:
        report_path.write_text(
            text.replace("1/45 = 2.22%", "1/45 = 99.99%", 1),
            encoding="utf-8",
        )
        summary["notes"] = summary["notes"].replace(
            "1/45 = 2.22%",
            "1/45 = 99.99%",
            1,
        )
    write_search_log(log_path, rows)

    with pytest.raises(ValueError, match="percentage.*added.*denominator"):
        validate_agent_runs(agent_run_dir)


def test_runtime_allows_equal_arithmetic_for_distinct_zero_yield_rounds(
    agent_run_dir: Path,
):
    records = (
        ("R2", 0, 45, 45, "0.00"),
        ("R3", 0, 45, 45, "0.00"),
    )
    report_path = agent_run_dir / "blind-ground.md"
    text = report_path.read_text(encoding="utf-8")
    log_path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(log_path)
    summary = fixture_summary(rows, "blind-ground")

    if "## Canonical final-round record" in text:
        replace_report_saturation(report_path, records)
        set_summary_saturation(summary, records)
    else:
        summary["notes"] = summary["notes"].replace(
            "1/45 = 2.22%",
            "0/45 = 0.00%",
            1,
        )
    write_search_log(log_path, rows)

    validate_agent_runs(agent_run_dir)


def test_runtime_requires_one_canonical_final_round_record(
    agent_run_dir: Path,
):
    remove_report_saturation(agent_run_dir / "blind-ground.md")

    with pytest.raises(ValueError, match="canonical final-round record"):
        validate_agent_runs(agent_run_dir)


def test_runtime_requires_exactly_two_final_saturation_rounds(
    agent_run_dir: Path,
):
    records = GROUND_FINAL_SATURATION + (
        ("R4", 0, 45, 45, "0.00"),
    )
    replace_report_saturation(agent_run_dir / "blind-ground.md", records)

    with pytest.raises(ValueError, match="exactly two"):
        validate_agent_runs(agent_run_dir)


def test_runtime_requires_consecutive_final_round_identities(
    agent_run_dir: Path,
):
    records = (
        GROUND_FINAL_SATURATION[0],
        ("R4", 0, 45, 45, "0.00"),
    )
    report_path = agent_run_dir / "blind-ground.md"
    had_canonical = "## Canonical final-round record" in report_path.read_text(
        encoding="utf-8"
    )
    replace_report_saturation(report_path, records)
    if had_canonical:
        log_path = fixture_search_log_path(agent_run_dir)
        rows = read_search_log(log_path)
        set_summary_saturation(fixture_summary(rows, "blind-ground"), records)
        write_search_log(log_path, rows)

    with pytest.raises(ValueError, match="consecutive"):
        validate_agent_runs(agent_run_dir)


def test_runtime_ignores_saturation_record_under_wrong_heading(
    agent_run_dir: Path,
):
    path = agent_run_dir / "blind-ground.md"
    remove_report_saturation(path)
    replace_report_saturation(
        path,
        GROUND_FINAL_SATURATION,
        heading="Other machine-readable data",
    )

    with pytest.raises(ValueError, match="canonical final-round record"):
        validate_agent_runs(agent_run_dir)


def append_provenance_row(path: Path, row: str) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.rstrip() + "\n\n" + row + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "row",
    [
        (
            "| AGRL9999 | "
            + chr(96)
            + "seed::C0009"
            + chr(96)
            + " | stale seed relationship |"
        ),
        (
            "| AGRL9998 | "
            + chr(96)
            + "citation::10.1145/3680468"
            + chr(96)
            + " | stale citation relationship |"
        ),
    ],
)
def test_provenance_ledger_rejects_extra_stale_relationship(
    agent_run_dir: Path,
    row: str,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    append_provenance_row(path, row)

    with pytest.raises(ValueError, match="provenance ledger.*exactly"):
        validate_agent_runs(agent_run_dir)


def test_provenance_ledger_rejects_duplicate_candidate_mapping(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    append_provenance_row(
        path,
        (
            "| AGRL0001 | "
            + chr(96)
            + "seed::C0010"
            + chr(96)
            + " | duplicate candidate mapping |"
        ),
    )

    with pytest.raises(ValueError, match="one-to-one"):
        validate_agent_runs(agent_run_dir)


def test_provenance_ledger_rejects_missing_relationship(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    prefix = (
        "| AGRL0001 | "
        + chr(96)
        + "seed::C0009"
        + chr(96)
        + " |"
    )
    line = next(
        line for line in text.splitlines() if line.startswith(prefix)
    )
    path.write_text(text.replace(line + "\n", "", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance ledger.*exactly"):
        validate_agent_runs(agent_run_dir)


def test_provenance_ledger_rejects_mismatched_source(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    old = (
        "| AGRL0001 | "
        + chr(96)
        + "seed::C0009"
        + chr(96)
        + " |"
    )
    new = (
        "| AGRL0001 | "
        + chr(96)
        + "seed::C0010"
        + chr(96)
        + " |"
    )
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance ledger.*exactly"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize("value", [" ground", "ground "])
def test_list_coded_singletons_reject_outer_whitespace(
    agent_run_dir: Path,
    value: str,
):
    mutate_row(agent_run_dir, "blind-ground", 0, domain=value)

    with pytest.raises(ValueError, match="domain.*canonical whitespace"):
        validate_agent_runs(agent_run_dir)


def test_taxonomy_top_level_must_be_a_mapping(agent_run_dir: Path):
    path = agent_run_dir.parent / "taxonomy.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="taxonomy.*top-level.*mapping"):
        validate_agent_runs(agent_run_dir)


@pytest.mark.parametrize(
    ("target", "payload", "message"),
    [
        ("taxonomy.json", b"{not-json", "invalid taxonomy"),
        ("taxonomy.json", b"\xff", "invalid taxonomy"),
        ("agent_runs/blind-ground.csv", b"\xff", "invalid UTF-8"),
    ],
)
def test_malformed_structured_inputs_raise_domain_validation_errors(
    agent_run_dir: Path,
    target: str,
    payload: bytes,
    message: str,
):
    path = agent_run_dir.parent / target
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        validate_agent_runs(agent_run_dir)


def test_malformed_csv_raises_domain_validation_error(agent_run_dir: Path):
    path = agent_run_dir / "blind-ground.csv"
    path.write_text(
        path.read_text(encoding="utf-8") + '"unterminated\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV parse error"):
        validate_agent_runs(agent_run_dir)


def test_required_report_heading_inside_fence_does_not_count(
    agent_run_dir: Path,
):
    path = agent_run_dir / "aware-geometry-rl.md"
    text = path.read_text(encoding="utf-8")
    heading = "## Inclusion and boundary judgments"
    assert heading in text
    text = text.replace(heading, "## Scope documentation", 1)
    text += (
        "\n"
        + chr(96) * 3
        + "text\n"
        + heading
        + "\n"
        + chr(96) * 3
        + "\n"
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="boundary"):
        validate_agent_runs(agent_run_dir)


def test_runtime_exact_query_section_must_match_report_section(
    agent_run_dir: Path,
):
    path = fixture_search_log_path(agent_run_dir)
    rows = read_search_log(path)
    index = exact_log_indexes(rows, "blind-ground")[0]
    notes = rows[index]["notes"]
    rows[index]["notes"] = re.sub(
        r"section: .*?\. Per-query",
        "section: Wrong section. Per-query",
        notes,
    )
    assert rows[index]["notes"] != notes
    write_search_log(path, rows)

    with pytest.raises(ValueError, match="section.*report"):
        validate_agent_runs(agent_run_dir)


def test_simulator_nonexpansion_queries_use_the_exact_report_section():
    rows = read_search_log()
    queries = {
        "site:gymnasium.farama.org api env reset seed options "
        "terminated truncated official",
        "site:gymnasium.farama.org vectorize custom environment official",
        "site:pettingzoo.farama.org api parallel reset seed multi agent official",
    }
    matches = [row for row in rows if row["query"] in queries]

    assert len(matches) == 3
    assert all(
        "section: Exact queries by round / "
        "Non-expansion interface-verification queries."
        in row["notes"]
        for row in matches
    )


def test_geometry_corrective_queries_have_stable_ids_and_sections():
    rows = read_search_log()
    expected = [
        (
            "S0247",
            "autonomous vehicle fuzzing scenario generation road geometry fixed map",
        ),
        (
            "S0248",
            "safety-critical traffic scenario factory road generation fixed road",
        ),
        (
            "S0249",
            "procedural generation race track surroundings iterative level design",
        ),
    ]
    actual = [
        (row["search_id"], row["query"])
        for row in rows
        if row["search_id"] in {"S0247", "S0248", "S0249"}
    ]

    assert actual == expected
    for row in rows:
        if row["search_id"] not in {"S0247", "S0248", "S0249"}:
            continue
        assert row["stream"] == "aware-geometry-rl"
        assert row["agent"] == "aware-geometry-rl"
        assert row["results_screened"] == "NR"
        assert row["candidates_added"] == "NR"
        assert (
            "section: Exact queries / Exclusion-lead verification queries."
            in row["notes"]
        )
        assert "counts were not captured" in row["notes"]


def test_data_readme_allows_corrective_queries_after_one_summary():
    text = " ".join(
        (DATA_DIR / "README.md").read_text(encoding="utf-8").split()
    )

    assert "exactly one" in text and "RUN-SUMMARY" in text
    assert "corrective query rows may follow" in text
    assert "need not be physically last" in text


def test_production_search_log_has_final_task4_counts():
    rows = read_search_log()
    exact_counts = Counter(
        (row["stream"], row["agent"])
        for row in rows
        if row["search_surface"] == "mixed-primary-web"
    )

    assert len(rows) == 249
    assert exact_counts[("blind-ground", "blind-ground")] == 88
    assert exact_counts[
        ("blind-aerial-maritime", "blind-aerial-maritime")
    ] == 64
    assert exact_counts[("aware-geometry-rl", "aware-geometry-rl")] == 41
    assert exact_counts[
        ("aware-simulation", "aware-simulation-benchmarks")
    ] == 47
    assert sum(
        row["search_surface"] == "documented-agent-run"
        for row in rows
    ) == 4


def test_agent_reports_have_clean_terminal_whitespace():
    for path in sorted(AGENT_RUN_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[ \t]+$", text, re.MULTILINE), path
        assert text.endswith("\n") and not text.endswith("\n\n"), path


def test_make_validate_runs_both_corpus_validators():
    makefile = (ROOT / "paper" / "Makefile").read_text(encoding="utf-8")
    validate_target = makefile.split("validate:", 1)[1].split("\n\n", 1)[0]

    assert "validate_corpus.py" in validate_target
    assert "validate_agent_runs.py" in validate_target
