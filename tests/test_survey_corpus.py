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
