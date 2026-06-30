import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from paper.scripts.validate_corpus import (
    CorpusError,
    DEFAULT_TAXONOMY,
    HEADERS,
    split_values,
    validate_directory,
)

SEARCH_QUERY_HEADERS = (
    "query_id",
    "stream",
    "domain",
    "query",
    "rationale",
)

BIBLIOGRAPHY_HEADERS = (
    "candidate_id",
    "cite_key",
    "entry_type",
    "key_author",
    "authors",
    "author_kinds",
    "title",
    "year",
    "venue_field",
    "venue",
    "doi",
    "url",
)

CITATION_KEYS_HEADERS = ("candidate_id", "cite_key")

FIXTURE_BIBTEX = """@article{Sample2026Course,
  author = {A. Author},
  title = {A Fictional Course Generator},
  journal = {Test Proceedings},
  year = {2026},
  doi = {10.0000/example}
}
"""

ZETA_BIBTEX = """@article{Zeta2027Course,
  author = {A. Author},
  title = {A Later Course Generator},
  journal = {Test Proceedings},
  year = {2027},
  doi = {10.0000/zeta}
}
"""


def write_search_queries(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCH_QUERY_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def read_search_queries(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        if path.name == "bibliography.csv":
            headers = BIBLIOGRAPHY_HEADERS
        elif path.name == "citation_keys.csv":
            headers = CITATION_KEYS_HEADERS
        else:
            headers = HEADERS[path.name]
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def write_sequential_search_rows(path: Path, count: int = 4) -> None:
    rows = read_rows(path)
    template = rows[0]
    expanded = []
    for number in range(1, count + 1):
        row = dict(template)
        row.update(
            search_id=f"S{number:04d}",
            query=f"fictional fixture {number}",
        )
        expanded.append(row)
    write_rows(path, expanded)


def blank_row(filename: str) -> dict[str, str]:
    if filename == "bibliography.csv":
        headers = BIBLIOGRAPHY_HEADERS
    elif filename == "citation_keys.csv":
        headers = CITATION_KEYS_HEADERS
    else:
        headers = HEADERS[filename]


    return dict.fromkeys(headers, "")

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
        code_status="NR",
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
    bibliography = blank_row("bibliography.csv")
    bibliography.update(
        candidate_id="C0001",
        cite_key="Sample2026Course",
        entry_type="article",
        key_author="Author",
        authors="A. Author",
        author_kinds="personal",
        title="A Fictional Course Generator",
        year="2026",
        venue_field="journal",
        venue="Test Proceedings",
        doi="10.0000/example",
    )

    citation_key = blank_row("citation_keys.csv")
    citation_key.update(
        candidate_id="C0001",
        cite_key="Sample2026Course",
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
        "bibliography.csv": [bibliography],
        "citation_keys.csv": [citation_key],
    }
    for filename, rows in rows_by_file.items():
        write_rows(data / filename, rows)
    write_search_queries(
        data / "search_queries.csv",
        [
            {
                "query_id": "B-G-01",
                "stream": "blind-ground",
                "domain": "ground",
                "query": "fictional course generation",
                "rationale": "Exercise the query validator.",
            }
        ],
    )
    (tmp_path / "references.bib").write_text(FIXTURE_BIBTEX, encoding="utf-8")
    return data


def rewrite_rows(path: Path, rows: list[dict[str, str]]) -> None:
    write_rows(path, rows)


def append_citation_key(fixture: Path, candidate_id: str, cite_key: str) -> None:
    path = fixture / "citation_keys.csv"
    rows = read_rows(path)
    rows.append({"candidate_id": candidate_id, "cite_key": cite_key})
    rewrite_rows(path, rows)


def test_included_source_requires_verified_metadata_and_evidence(tmp_path):
    validate_directory(build_valid_fixture(tmp_path))


def test_committed_citation_key_ledger_has_canonical_prefix_and_appends():
    data_dir = Path(__file__).resolve().parents[1] / "paper" / "data"
    candidates = read_rows(data_dir / "candidates.csv")
    ledger_path = data_dir / "citation_keys.csv"
    ledger = read_rows(ledger_path)
    ledger_lines = ledger_path.read_bytes().splitlines(keepends=True)

    assert len(ledger) == 198
    assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == (
        "48d891587257f79b9c7cf97f90dd3ebd36bd0378e9ed8c100628afcfd6540e5f"
    )
    assert hashlib.sha256(b"".join(ledger_lines[:185])).hexdigest() == (
        "2370e0f9a105be3fcabd19913441f6f98c69c123bb73ca964e007bef994105c4"
    )
    assert len({row["candidate_id"] for row in ledger}) == 198
    assert len({row["cite_key"].casefold() for row in ledger}) == 198
    assert all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._/+\-]*", row["cite_key"])
        for row in ledger
    )

    appended_ids = [row["candidate_id"] for row in ledger[184:]]
    assert appended_ids == sorted(
        appended_ids,
        key=lambda candidate_id: (int(candidate_id[1:]), candidate_id),
    )
    ledger_by_id = {
        row["candidate_id"]: row["cite_key"] for row in ledger
    }
    assert {
        row["candidate_id"]: row["cite_key"]
        for row in candidates
        if row["cite_key"]
    } == ledger_by_id
    assert {
        candidate_id: ledger_by_id[candidate_id]
        for candidate_id in ("C0049", "C0082", "C0165", "C0188", "C0189")
    } == {
        "C0049": "DralligNodateProceduralRace",
        "C0082": "MIT2017MITRACECAR",
        "C0165": "Yu2025MasteringDiverse",
        "C0188": "IsaacLabNodateIsaacLab",
        "C0189": "TUMFTM2020LapTime",
    }


def test_citation_key_ledger_is_required(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture / "citation_keys.csv").unlink()

    with pytest.raises(
        CorpusError,
        match=r"citation_keys\.csv: file is missing",
    ):
        validate_directory(fixture)


def test_citation_key_ledger_requires_exact_header(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture / "citation_keys.csv").write_text(
        "candidate_id,key\nC0001,Sample2026Course\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match=r"citation_keys\.csv: headers"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"candidate_id": "", "cite_key": "Key"}], "candidate_id is required"),
        ([{"candidate_id": "C0001", "cite_key": ""}], "cite_key is required"),
        ([{"candidate_id": "C01", "cite_key": "Key"}], "at least four digits"),
        ([{"candidate_id": "C9999", "cite_key": "Key"}], "C9999.*does not exist"),
        (
            [
                {"candidate_id": "C0001", "cite_key": "Key"},
                {"candidate_id": "C0001", "cite_key": "OtherKey"},
            ],
            "duplicate candidate_id 'C0001'",
        ),
        (
            [
                {"candidate_id": "C0001", "cite_key": "Key"},
                {"candidate_id": "C0002", "cite_key": "key"},
            ],
            "duplicate cite_key 'key'",
        ),
        ([{"candidate_id": "C0001", "cite_key": "unsafe key"}], "BibTeX-safe"),
    ],
    ids=[
        "blank-id",
        "blank-key",
        "malformed-id",
        "orphan-id",
        "duplicate-id",
        "casefold-duplicate-key",
        "unsafe-key",
    ],
)
def test_citation_key_ledger_rejects_malformed_rows(tmp_path, rows, message):
    fixture = build_valid_fixture(tmp_path)
    write_rows(fixture / "citation_keys.csv", rows)

    with pytest.raises(CorpusError, match=message):
        validate_directory(fixture)


def test_active_candidate_key_must_equal_ledger(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    write_rows(
        fixture / "citation_keys.csv",
        [{"candidate_id": "C0001", "cite_key": "DifferentKey"}],
    )

    with pytest.raises(
        CorpusError,
        match=r"C0001.*cite_key.*does not match.*ledger",
    ):
        validate_directory(fixture)


def test_active_candidate_requires_ledger_assignment(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    write_rows(fixture / "citation_keys.csv", [])

    with pytest.raises(
        CorpusError,
        match=r"C0001.*missing.*citation key ledger",
    ):
        validate_directory(fixture)


def test_dormant_ledger_assignment_is_legal_and_stays_out_of_outputs(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidates = read_rows(fixture / "candidates.csv")
    candidates[0].update(
        cite_key="",
        screening_status="excluded",
        exclusion_reason="Out of scope",
    )
    rewrite_rows(fixture / "candidates.csv", candidates)
    rewrite_rows(fixture / "evidence.csv", [])
    claims = read_rows(fixture / "claims.csv")
    claims[0]["cite_keys"] = ""
    rewrite_rows(fixture / "claims.csv", claims)
    rewrite_rows(fixture / "bibliography.csv", [])
    (fixture.parent / "references.bib").write_text("", encoding="utf-8")

    validate_directory(fixture)

    assert read_rows(fixture / "citation_keys.csv") == [
        {"candidate_id": "C0001", "cite_key": "Sample2026Course"}
    ]


def test_bibliography_is_required(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture / "bibliography.csv").unlink()

    with pytest.raises(CorpusError, match=r"bibliography\.csv: file is missing"):
        validate_directory(fixture)


def test_bibliography_requires_exact_header(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    path.write_text(
        ",".join((*BIBLIOGRAPHY_HEADERS[:-1], "link")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match=r"bibliography\.csv: headers"):
        validate_directory(fixture)


def test_bibliography_requires_every_eligible_candidate(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    rewrite_rows(fixture / "bibliography.csv", [])

    with pytest.raises(
        CorpusError,
        match=r"bibliography\.csv: candidate_id mismatch.*missing=\['C0001'\]",
    ):
        validate_directory(fixture)


def test_bibliography_rejects_extra_candidate_row(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    extra = dict(rows[0])
    extra.update(candidate_id="C9999", cite_key="Extra2026Course")
    rewrite_rows(path, rows + [extra])

    with pytest.raises(
        CorpusError,
        match=r"bibliography\.csv:3: candidate_id='C9999'.*not eligible",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize("field", ["candidate_id", "cite_key"])
def test_bibliography_ids_and_keys_must_be_unique(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    duplicate = dict(rows[0])
    if field == "candidate_id":
        duplicate["cite_key"] = "Other2026Course"
    else:
        duplicate["candidate_id"] = "C9999"
    rewrite_rows(path, rows + [duplicate])

    with pytest.raises(CorpusError, match=rf"duplicate {field}"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cite_key", "Drifted2026Course"),
        ("title", "A Drifted Title"),
        ("authors", "Another Author"),
        ("year", "2025"),
        ("venue", "Another Venue"),
        ("doi", "10.0000/drift"),
    ],
)
def test_bibliography_candidate_fields_must_match(
    tmp_path, field, value
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0][field] = value
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: candidate_id='C0001'.*{field}.*candidates\.csv",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("screening_status", "metadata_status"),
    [("excluded", "verified"), ("candidate", "unverified")],
)
def test_bibliography_rejects_ineligible_candidate_leakage(
    tmp_path, screening_status, metadata_status
):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    candidates[0].update(
        cite_key="",
        screening_status=screening_status,
        metadata_status=metadata_status,
        exclusion_reason=("Out of scope" if screening_status == "excluded" else ""),
    )
    rewrite_rows(candidate_path, candidates)
    rewrite_rows(fixture / "evidence.csv", [])
    claims = read_rows(fixture / "claims.csv")
    claims[0]["cite_keys"] = ""
    rewrite_rows(fixture / "claims.csv", claims)

    with pytest.raises(
        CorpusError,
        match=r"bibliography\.csv:2: candidate_id='C0001'.*not eligible",
    ):
        validate_directory(fixture)


def test_verified_nonexcluded_candidate_requires_cite_key(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    candidates[0].update(screening_status="candidate", cite_key="")
    rewrite_rows(candidate_path, candidates)
    rewrite_rows(fixture / "evidence.csv", [])
    claims = read_rows(fixture / "claims.csv")
    claims[0]["cite_keys"] = ""
    rewrite_rows(fixture / "claims.csv", claims)

    with pytest.raises(
        CorpusError,
        match=r"candidates\.csv:2:.*cite_key is required.*verified.*non-excluded",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("screening_status", "metadata_status"),
    [("excluded", "verified"), ("candidate", "unverified")],
)
def test_ineligible_candidate_requires_blank_cite_key(
    tmp_path, screening_status, metadata_status
):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    candidates[0].update(
        screening_status=screening_status,
        metadata_status=metadata_status,
        exclusion_reason=("Out of scope" if screening_status == "excluded" else ""),
    )
    rewrite_rows(candidate_path, candidates)
    rewrite_rows(fixture / "evidence.csv", [])
    claims = read_rows(fixture / "claims.csv")
    claims[0]["cite_keys"] = ""
    rewrite_rows(fixture / "claims.csv", claims)

    with pytest.raises(
        CorpusError,
        match=r"candidates\.csv:2:.*cite_key must be blank.*ineligible",
    ):
        validate_directory(fixture)


def test_candidate_cite_keys_are_case_insensitively_unique(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    duplicate = dict(candidates[0])
    duplicate.update(
        candidate_id="C0002",
        cite_key="sample2026course",
        title="Another Course Generator",
        doi="10.0000/another",
        screening_status="candidate",
    )
    rewrite_rows(candidate_path, candidates + [duplicate])

    with pytest.raises(
        CorpusError,
        match=r"candidates\.csv:3: duplicate cite_key 'sample2026course'",
    ):
        validate_directory(fixture)


def test_bibliography_cite_keys_are_case_insensitively_unique(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    duplicate = dict(rows[0])
    duplicate.update(
        candidate_id="C9999",
        cite_key="sample2026course",
    )
    rewrite_rows(path, rows + [duplicate])

    with pytest.raises(
        CorpusError,
        match=r"bibliography\.csv:3: duplicate cite_key 'sample2026course'",
    ):
        validate_directory(fixture)



def test_bibliography_rows_require_canonical_order(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    second_candidate = dict(candidates[0])
    second_candidate.update(
        candidate_id="C0002",
        cite_key="Alpha2025Course",
        title="An Earlier Course Generator",
        year="2025",
        doi="10.0000/alpha",
        screening_status="candidate",
    )
    rewrite_rows(candidate_path, candidates + [second_candidate])
    append_citation_key(fixture, "C0002", "Alpha2025Course")

    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    second_row = dict(rows[0])
    second_row.update(
        candidate_id="C0002",
        cite_key="Alpha2025Course",
        title="An Earlier Course Generator",
        year="2025",
        doi="10.0000/alpha",
    )
    rewrite_rows(bibliography_path, rows + [second_row])

    with pytest.raises(
        CorpusError,
        match=r"bibliography\.csv: rows are not in canonical.*Alpha2025Course",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_id",
        "cite_key",
        "entry_type",
        "key_author",
        "authors",
        "author_kinds",
        "title",
    ],
)
def test_bibliography_requires_core_values(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0][field] = ""
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field} is required",
    ):
        validate_directory(fixture)


def test_bibliography_rejects_unsupported_entry_type(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0]["entry_type"] = "thesis"
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match="unsupported entry_type 'thesis'"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("entry_type", "expected_venue_field"),
    [
        ("article", "journal"),
        ("inproceedings", "booktitle"),
        ("misc", "howpublished"),
        ("techreport", "institution"),
        ("book", "publisher"),
    ],
)
def test_bibliography_entry_type_requires_exact_venue_field(
    tmp_path, entry_type, expected_venue_field
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0].update(entry_type=entry_type, venue_field="wrong")
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match=rf"{entry_type} requires venue_field='{expected_venue_field}'",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("venue", "venue_field", "message"),
    [
        ("Test Proceedings", "", "article requires venue_field='journal'"),
        ("", "journal", "venue_field must be empty when venue is empty"),
    ],
)
def test_bibliography_venue_and_field_must_be_paired(
    tmp_path, venue, venue_field, message
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0].update(venue=venue, venue_field=venue_field)
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match=message):
        validate_directory(fixture)


@pytest.mark.parametrize("field", ["year", "venue"])
def test_non_misc_bibliography_requires_year_and_venue(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    candidates[0][field] = ""
    rewrite_rows(candidate_path, candidates)

    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0][field] = ""
    if field == "venue":
        rows[0]["venue_field"] = ""
    rewrite_rows(bibliography_path, rows)

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field} is required for article",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize("field", ["year", "venue"])
@pytest.mark.parametrize("source_type", ["paper", "software paper"])
def test_misc_paper_like_bibliography_requires_year_and_venue(
    tmp_path, field, source_type
):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    candidates[0].update(source_type=source_type)
    candidates[0][field] = ""
    rewrite_rows(candidate_path, candidates)

    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0].update(entry_type="misc", venue_field="howpublished")
    rows[0][field] = ""
    if field == "venue":
        rows[0]["venue_field"] = ""
    rewrite_rows(bibliography_path, rows)

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field} is required for paper-like misc",
    ):
        validate_directory(fixture)


def test_misc_nonpaper_artifact_allows_missing_year_and_venue(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidate_path = fixture / "candidates.csv"
    candidates = read_rows(candidate_path)
    candidates[0].update(source_type="software", year="", venue="")
    rewrite_rows(candidate_path, candidates)

    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0].update(
        entry_type="misc",
        year="",
        venue_field="",
        venue="",
    )
    rewrite_rows(bibliography_path, rows)
    (fixture.parent / "references.bib").write_text(
        """@misc{Sample2026Course,
  author = {A. Author},
  title = {A Fictional Course Generator},
  doi = {10.0000/example}
}
""",
        encoding="utf-8",
    )

    validate_directory(fixture)


@pytest.mark.parametrize("field", ["authors", "key_author"])
def test_bibliography_rejects_nr_author_values(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0][field] = "NR"
    rewrite_rows(bibliography_path, rows)
    if field == "authors":
        candidates = read_rows(fixture / "candidates.csv")
        candidates[0]["authors"] = "NR"
        rewrite_rows(fixture / "candidates.csv", candidates)
        (fixture.parent / "references.bib").write_text(
            FIXTURE_BIBTEX.replace("author = {A. Author}", "author = {NR}"),
            encoding="utf-8",
        )

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field} cannot be NR",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("authors", "author_kinds", "field"),
    [
        ("A. Author;NR", "personal;personal", "authors"),
        (
            "A. Author;Second Author",
            "personal;NR",
            "author_kinds",
        ),
    ],
)
def test_bibliography_author_lists_reject_nr_tokens(
    tmp_path, authors, author_kinds, field
):
    fixture = build_valid_fixture(tmp_path)
    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0].update(authors=authors, author_kinds=author_kinds)
    rewrite_rows(bibliography_path, rows)

    candidates_path = fixture / "candidates.csv"
    candidates = read_rows(candidates_path)
    candidates[0]["authors"] = authors
    rewrite_rows(candidates_path, candidates)

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field}: NR must be the sole list sentinel",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    "marker",
    ["A. Author ET, AL.", "A. Author et-al", "A. Author Et.Al."],
)
@pytest.mark.parametrize("field", ["authors", "key_author"])
def test_bibliography_rejects_incomplete_author_markers(
    tmp_path, field, marker
):
    fixture = build_valid_fixture(tmp_path)
    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0][field] = marker
    rewrite_rows(bibliography_path, rows)
    if field == "authors":
        candidates = read_rows(fixture / "candidates.csv")
        candidates[0]["authors"] = marker
        rewrite_rows(fixture / "candidates.csv", candidates)
        (fixture.parent / "references.bib").write_text(
            FIXTURE_BIBTEX.replace(
                "author = {A. Author}",
                f"author = {{{marker}}}",
            ),
            encoding="utf-8",
        )

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field} contains incomplete author marker",
    ):
        validate_directory(fixture)


def test_bibliography_rejects_empty_canonical_author_token(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0].update(
        authors="A. Author; ;Second Author",
        author_kinds="personal;personal;personal",
    )
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match="authors contains an empty semicolon element",
    ):
        validate_directory(fixture)



@pytest.mark.parametrize(
    ("authors", "author_kinds", "message"),
    [
        (
            "A. Author;Second Author",
            "personal",
            "author_kinds must align one-to-one with authors",
        ),
        (
            "A. Author",
            "personal;",
            "author_kinds contains an empty semicolon element",
        ),
        ("A. Author", "organization", "invalid author kind 'organization'"),
    ],
)
def test_bibliography_author_kinds_align_with_authors(
    tmp_path, authors, author_kinds, message
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0].update(authors=authors, author_kinds=author_kinds)
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match=message):
        validate_directory(fixture)


def test_bibliography_year_must_be_four_digits_when_present(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0]["year"] = "20X6"
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match="year.*four digits"):
        validate_directory(fixture)



@pytest.mark.parametrize(
    "whitespace",
    [
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\u00a0",
        "\u1680",
        "\u2003",
        "\u2028",
        "\u202f",
        "\u3000",
    ],
    ids=[
        "space",
        "tab",
        "newline",
        "carriage-return",
        "vertical-tab",
        "form-feed",
        "no-break-space",
        "ogham-space-mark",
        "em-space",
        "line-separator",
        "narrow-no-break-space",
        "ideographic-space",
    ],
)
@pytest.mark.parametrize("filename", ["candidates.csv", "bibliography.csv"])
def test_canonical_urls_reject_unicode_whitespace(
    tmp_path, filename, whitespace
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / filename
    rows = read_rows(path)
    rows[0]["url"] = f"https://example.invalid/path{whitespace}segment"
    rewrite_rows(path, rows)

    message = (
        "carriage return"
        if filename == "bibliography.csv" and whitespace == "\r"
        else "whitespace"
    )
    with pytest.raises(
        CorpusError,
        match=rf"{re.escape(filename)}:2: url contains {message}",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    "doi",
    [
        "https://doi.org/10.0000/example",
        "10.0000/EXAMPLE",
        "10.12/example",
        "10.0000/example/",
    ],
)
def test_bibliography_doi_requires_canonical_form(tmp_path, doi):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0]["doi"] = doi
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match="doi.*canonical DOI"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    "url",
    [
        "example.invalid/paper",
        "ftp://example.invalid/paper",
        "https://user:secret@example.invalid/paper",
        "https://example.invalid:bad/paper",
    ],
)
def test_bibliography_url_requires_absolute_http_shape(tmp_path, url):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0]["url"] = url
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match="url.*absolute HTTP/HTTPS URL"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("doi", "url"),
    [
        ("10.0000/example", "https://doi.org/10.0000/other"),
        ("", "https://doi.org/10.0000/example"),
    ],
)
def test_bibliography_doi_url_must_match(tmp_path, doi, url):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0].update(doi=doi, url=url)
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match="DOI resolver URL.*does not match doi",
    ):
        validate_directory(fixture)


def test_bibliography_rejects_redundant_matching_doi_url(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    resolver = "https://doi.org/10.0000/example"
    bibliography_path = fixture / "bibliography.csv"
    rows = read_rows(bibliography_path)
    rows[0]["url"] = resolver
    rewrite_rows(bibliography_path, rows)
    rendered = FIXTURE_BIBTEX.replace(
        "  doi = {10.0000/example}\n",
        "  doi = {10.0000/example},\n"
        f"  url = {{{resolver}}}\n",
    )
    (fixture.parent / "references.bib").write_text(
        rendered,
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match=r"bibliography\.csv:2: redundant DOI resolver URL",
    ):
        validate_directory(fixture)



def test_bibliography_values_reject_surrounding_whitespace(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    rows[0]["key_author"] = " Author "
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match="key_author contains surrounding whitespace",
    ):
        validate_directory(fixture)

@pytest.mark.parametrize("field", BIBLIOGRAPHY_HEADERS)
def test_bibliography_values_reject_embedded_carriage_return(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "bibliography.csv"
    rows = read_rows(path)
    value = rows[0][field] or "value"
    rows[0][field] = f"{value}\rcontinued"
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError,
        match=rf"bibliography\.csv:2: {field} contains carriage return",
    ):
        validate_directory(fixture)



@pytest.mark.parametrize(
    ("line_ending", "name"),
    [(b"\r\n", "CRLF"), (b"\r", "lone-CR")],
)
def test_references_bib_requires_deterministic_lf_bytes(
    tmp_path, line_ending, name
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_bytes(
        FIXTURE_BIBTEX.encode("utf-8").replace(b"\n", line_ending)
    )

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*deterministic LF bytes",
    ):
        validate_directory(fixture)



def test_references_bib_is_required(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture.parent / "references.bib").unlink()

    with pytest.raises(CorpusError, match=r"references\.bib: file is missing"):
        validate_directory(fixture)


def test_references_bib_requires_utf8(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_bytes(b"\xff")

    with pytest.raises(
        CorpusError, match=r"references\.bib: invalid UTF-8"
    ) as error:
        validate_directory(fixture)

    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_references_bib_rejects_stale_bibliography_values(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidates = read_rows(fixture / "candidates.csv")
    candidates[0]["title"] = "An Updated Course Generator"
    rewrite_rows(fixture / "candidates.csv", candidates)
    bibliography = read_rows(fixture / "bibliography.csv")
    bibliography[0]["title"] = "An Updated Course Generator"
    rewrite_rows(fixture / "bibliography.csv", bibliography)

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*key='Sample2026Course'.*field 'title'",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_missing_entry(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture.parent / "references.bib").write_text("", encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib: entry mismatch.*missing=\['Sample2026Course'\]",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_extra_entry(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    extra = """@misc{Extra2026Course,
  author = {Extra Author},
  title = {Extra Course},
  howpublished = {Test Site},
  year = {2026},
  url = {https://example.invalid/extra}
}
"""
    path.write_text(FIXTURE_BIBTEX + "\n" + extra, encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib: entry mismatch.*extra=\['Extra2026Course'\]",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_duplicate_entry(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_text(
        FIXTURE_BIBTEX + "\n" + FIXTURE_BIBTEX,
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*duplicate entry key 'Sample2026Course'",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_duplicate_field(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    duplicate = FIXTURE_BIBTEX.replace(
        "  title = {A Fictional Course Generator},",
        "  author = {Another Author},\n"
        "  title = {A Fictional Course Generator},",
    )
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*key='Sample2026Course'.*duplicate field 'author'",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_unclosed_entry_brace(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_text(FIXTURE_BIBTEX[:-2], encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*key='Sample2026Course'.*closing brace",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_case_only_duplicate_entry_key(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    case_variant = FIXTURE_BIBTEX.replace(
        "@article{Sample2026Course,",
        "@article{sample2026course,",
    )
    path.write_text(
        FIXTURE_BIBTEX + "\n" + case_variant,
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*case-insensitive duplicate entry key 'sample2026course'",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_case_only_duplicate_field(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    duplicate = FIXTURE_BIBTEX.replace(
        "  title = {A Fictional Course Generator},",
        "  Author = {Another Author},\n"
        "  title = {A Fictional Course Generator},",
    )
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*key='Sample2026Course'.*case-insensitive duplicate field 'Author'",
    ):
        validate_directory(fixture)


@pytest.mark.parametrize(
    "payload",
    [
        "% leading comment\n" + FIXTURE_BIBTEX,
        FIXTURE_BIBTEX + "% trailing comment\n",
        FIXTURE_BIBTEX + "trailing garbage\n",
    ],
    ids=["leading-comment", "trailing-comment", "trailing-garbage"],
)
def test_references_bib_rejects_comments_and_trailing_garbage(
    tmp_path, payload
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*expected '@' to start a BibTeX entry",
    ):
        validate_directory(fixture)



@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("@article", "@book", "entry_type"),
        ("author = {A. Author}", "author = {Another Author}", "field 'author'"),
        (
            "title = {A Fictional Course Generator}",
            "title = {A Stale Course Generator}",
            "field 'title'",
        ),
        (
            "journal = {Test Proceedings}",
            "journal = {Other Proceedings}",
            "field 'journal'",
        ),
        ("year = {2026}", "year = {2025}", "field 'year'"),
        ("doi = {10.0000/example}", "doi = {10.0000/stale}", "field 'doi'"),
        ("journal =", "booktitle =", "field mismatch"),
    ],
)
def test_references_bib_rejects_entry_and_field_drift(
    tmp_path, old, new, message
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_text(FIXTURE_BIBTEX.replace(old, new), encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=rf"references\.bib:.*key='Sample2026Course'.*{message}",
    ):
        validate_directory(fixture)


def test_references_bib_rejects_extra_rendered_field(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    rendered = FIXTURE_BIBTEX.replace(
        "  doi = {10.0000/example}\n",
        "  doi = {10.0000/example},\n"
        "  url = {https://example.invalid/paper}\n",
    )
    path.write_text(rendered, encoding="utf-8")

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:.*key='Sample2026Course'.*field mismatch",
    ):
        validate_directory(fixture)


def test_references_bib_requires_deterministic_formatting(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture.parent / "references.bib"
    path.write_text(
        FIXTURE_BIBTEX.replace("  author =", "    author ="),
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match=r"references\.bib:2: does not match deterministic rendering",
    ):
        validate_directory(fixture)


def test_references_bib_entries_require_canonical_order(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    candidates = read_rows(fixture / "candidates.csv")
    second_candidate = dict(candidates[0])
    second_candidate.update(
        candidate_id="C0002",
        cite_key="Zeta2027Course",
        title="A Later Course Generator",
        year="2027",
        doi="10.0000/zeta",
        screening_status="candidate",
    )
    rewrite_rows(fixture / "candidates.csv", candidates + [second_candidate])
    append_citation_key(fixture, "C0002", "Zeta2027Course")

    bibliography = read_rows(fixture / "bibliography.csv")
    second_row = dict(bibliography[0])
    second_row.update(
        candidate_id="C0002",
        cite_key="Zeta2027Course",
        title="A Later Course Generator",
        year="2027",
        doi="10.0000/zeta",
    )
    rewrite_rows(
        fixture / "bibliography.csv",
        bibliography + [second_row],
    )
    (fixture.parent / "references.bib").write_text(
        ZETA_BIBTEX + "\n" + FIXTURE_BIBTEX,
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match=r"references\.bib: entries are not in canonical order.*Sample2026Course",
    ):
        validate_directory(fixture)


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
def test_search_queries_requires_exact_header(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture / "search_queries.csv").write_text(
        "query_id,stream,domain,query,reason\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match=r"search_queries\.csv: headers"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    "field", ["query_id", "stream", "domain", "query", "rationale"]
)
def test_search_queries_requires_nonempty_fields(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_queries.csv"
    rows = read_search_queries(path)
    rows[0][field] = ""
    write_search_queries(path, rows)

    with pytest.raises(CorpusError, match=rf"{field} is required"):
        validate_directory(fixture)


def test_search_query_id_must_be_unique(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_queries.csv"
    rows = read_search_queries(path)
    write_search_queries(path, rows + [dict(rows[0])])

    with pytest.raises(CorpusError, match="duplicate query_id"):
        validate_directory(fixture)


def test_search_query_stream_must_be_from_frozen_matrix(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_queries.csv"
    rows = read_search_queries(path)
    rows[0]["stream"] = "bootstrap"
    write_search_queries(path, rows)

    with pytest.raises(CorpusError, match="stream='bootstrap'.*frozen query matrix"):
        validate_directory(fixture)


def test_search_query_domain_must_be_in_taxonomy(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_queries.csv"
    rows = read_search_queries(path)
    rows[0]["domain"] = "space"
    write_search_queries(path, rows)

    with pytest.raises(CorpusError, match="domain='space'.*domain"):
        validate_directory(fixture)


@pytest.mark.parametrize("candidate_id", ["1", "C123", "c0001", "C0001x"])
def test_candidate_id_requires_c_and_at_least_four_digits(
    tmp_path, candidate_id
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "candidates.csv"
    rows = read_rows(path)
    rows[0]["candidate_id"] = candidate_id
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError, match="candidate_id.*C followed by at least four digits"
    ):
        validate_directory(fixture)


@pytest.mark.parametrize("search_date", ["2026-6-29", "2026-02-30", "not-a-date"])
def test_search_date_requires_iso_calendar_date(tmp_path, search_date):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0]["search_date"] = search_date
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match="search_date.*ISO date"):
        validate_directory(fixture)


@pytest.mark.parametrize("field", ["results_screened", "candidates_added"])
@pytest.mark.parametrize("value", ["-1", "1.5", "+1", "one"])
def test_search_counts_require_nonnegative_integers(tmp_path, field, value):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0][field] = value
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match=rf"{field}.*nonnegative integer"):
        validate_directory(fixture)

def test_search_ids_reject_a_gap_after_deleted_s0003(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    write_sequential_search_rows(path)
    rows = read_rows(path)
    del rows[2]
    write_rows(path, rows)

    with pytest.raises(CorpusError, match="search_id.*sequential"):
        validate_directory(fixture)


def test_search_ids_reject_reversed_row_order(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    write_sequential_search_rows(path)
    rows = read_rows(path)
    write_rows(path, list(reversed(rows)))

    with pytest.raises(CorpusError, match="search_id.*sequential"):
        validate_directory(fixture)


@pytest.mark.parametrize("search_id", ["S01", "S000A", "S-0001", "0001"])
def test_search_ids_reject_malformed_values(tmp_path, search_id):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    write_sequential_search_rows(path)
    rows = read_rows(path)
    rows[0]["search_id"] = search_id
    write_rows(path, rows)

    with pytest.raises(CorpusError, match="search_id.*sequential"):
        validate_directory(fixture)



def test_nonbootstrap_exact_query_accepts_documented_nr_counts(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0].update(
        results_screened="NR",
        candidates_added="NR",
        notes="Per-query screened-hit and candidate-add counts were not captured.",
    )
    rewrite_rows(path, rows)

    validate_directory(fixture)


def test_nonbootstrap_summary_accepts_documented_nr_screened_count(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0].update(
        query="RUN-SUMMARY:paper/data/agent_runs/test.md",
        search_surface="documented-agent-run",
        results_screened="NR",
        candidates_added="1",
        notes="Total screened-hit count was not captured.",
    )
    rewrite_rows(path, rows)

    validate_directory(fixture)


@pytest.mark.parametrize("field", ["results_screened", "candidates_added"])
def test_bootstrap_search_counts_reject_nr(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0].update(
        stream="bootstrap",
        query="fixture.rst",
        search_surface="local-corpus",
        notes="The count was not captured.",
    )
    rows[0][field] = "NR"
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match=rf"{field}.*nonnegative integer"):
        validate_directory(fixture)


@pytest.mark.parametrize("field", ["results_screened", "candidates_added"])
def test_nonbootstrap_nr_count_requires_explicit_uncaptured_note(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0][field] = "NR"
    rows[0]["notes"] = "Count unavailable."
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match=rf"{field}.*not captured"):
        validate_directory(fixture)


def test_local_corpus_bootstrap_count_must_equal_seed_mentions(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0].update(
        stream="bootstrap",
        query="fixture.rst",
        search_surface="local-corpus",
        results_screened="0",
    )
    rewrite_rows(path, rows)

    with pytest.raises(
        CorpusError, match="results_screened=0.*seed_coverage rows=1"
    ):
        validate_directory(fixture)


def test_local_corpus_bootstrap_count_accepts_matching_seed_mentions(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "search_log.csv"
    rows = read_rows(path)
    rows[0].update(
        stream="bootstrap",
        query="fixture.rst",
        search_surface="local-corpus",
        results_screened="1",
    )
    rewrite_rows(path, rows)

    validate_directory(fixture)



@pytest.mark.parametrize(
    ("filename", "field", "value"),
    [
        ("candidates.csv", "screening_status", "included;boundary"),
        ("candidates.csv", "metadata_status", "verified;unverified"),
        ("evidence.csv", "code_status", "not_found;closed"),
        ("claims.csv", "evidence_status", "direct;inferred"),
    ],
)
def test_scalar_controlled_fields_reject_semicolon_lists(
    tmp_path, filename, field, value
):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / filename)
    rows[0][field] = value
    rewrite_rows(fixture / filename, rows)

    with pytest.raises(CorpusError, match=rf"{field}.*exactly one"):
        validate_directory(fixture)


def test_controlled_fields_canonicalize_whitespace_before_relations(tmp_path):
    fixture = build_valid_fixture(tmp_path)

    candidates = read_rows(fixture / "candidates.csv")
    candidates[0]["screening_status"] = " included "
    candidates[0]["metadata_status"] = " verified "
    rewrite_rows(fixture / "candidates.csv", candidates)

    evidence = read_rows(fixture / "evidence.csv")
    evidence[0]["domain"] = " ground ; aerial "
    evidence[0]["code_status"] = " not_found "
    rewrite_rows(fixture / "evidence.csv", evidence)

    claims = read_rows(fixture / "claims.csv")
    claims[0]["evidence_status"] = " direct "
    rewrite_rows(fixture / "claims.csv", claims)

    validate_directory(fixture)


@pytest.mark.parametrize("value", ["ground;;aerial", "ground;"])
def test_multivalued_controlled_fields_reject_empty_elements(tmp_path, value):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "evidence.csv")
    rows[0]["domain"] = value
    rewrite_rows(fixture / "evidence.csv", rows)

    with pytest.raises(CorpusError, match="domain.*empty list element"):
        validate_directory(fixture)


@pytest.mark.parametrize("field", ["domain", "code_status"])
def test_evidence_controlled_fields_accept_nr_as_sole_value(tmp_path, field):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "evidence.csv")
    rows[0][field] = "NR"
    rewrite_rows(fixture / "evidence.csv", rows)

    validate_directory(fixture)


@pytest.mark.parametrize(
    ("field", "other"),
    [("domain", "ground"), ("code_status", "not_found")],
)
def test_evidence_controlled_fields_reject_nr_combined_with_value(
    tmp_path, field, other
):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "evidence.csv")
    rows[0][field] = f"NR;{other}"
    rewrite_rows(fixture / "evidence.csv", rows)

    with pytest.raises(CorpusError, match="NR must be used alone"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("filename", "field"),
    [
        ("candidates.csv", "screening_status"),
        ("candidates.csv", "metadata_status"),
        ("claims.csv", "evidence_status"),
    ],
)
def test_operational_statuses_reject_nr(tmp_path, filename, field):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / filename)
    rows[0][field] = "NR"
    rewrite_rows(fixture / filename, rows)

    with pytest.raises(CorpusError, match=rf"{field}=.*NR.*outside"):
        validate_directory(fixture)


REQUIRED_FIELDS_BY_FILE = {
    "search_log.csv": (
        "search_id",
        "search_date",
        "stream",
        "agent",
        "query",
        "search_surface",
        "results_screened",
        "candidates_added",
    ),
    "candidates.csv": (
        "candidate_id",
        "title",
        "screening_status",
        "metadata_status",
    ),
    "seed_coverage.csv": (
        "source_path",
        "source_heading",
        "source_label",
        "coverage_status",
    ),
    "evidence.csv": (
        "cite_key",
        "domain",
        "course_object",
        "representation_family",
        "generator_family",
        "generation_role",
        "validity_strategy",
        "code_status",
        "evidence_locator",
    ),
    "claims.csv": (
        "claim_id",
        "section",
        "claim_text",
        "evidence_status",
    ),
    "metrics.csv": (
        "metric_id",
        "layer",
        "name",
        "definition",
        "domain",
        "requires_dynamics",
        "minimum_reporting",
    ),
    "simulators.csv": ("system", "domain"),
    "conflicts.csv": (
        "conflict_id",
        "record_type",
        "record_key",
        "field",
        "value_a",
        "value_b",
    ),
}


def minimal_required_row(filename: str) -> dict[str, str]:
    row = blank_row(filename)
    values = {
        "metrics.csv": {
            "metric_id": "M0001",
            "layer": "geometry",
            "name": "Curvature",
            "definition": "A fictional metric.",
            "domain": "ground",
            "requires_dynamics": "no",
            "minimum_reporting": "definition",
        },
        "simulators.csv": {
            "system": "FixtureSim",
            "domain": "ground",
        },
        "conflicts.csv": {
            "conflict_id": "CF0001",
            "record_type": "candidate",
            "record_key": "C0001",
            "field": "title",
            "value_a": "Title A",
            "value_b": "Title B",
        },
    }
    row.update(values[filename])
    return row


def test_strict_csv_errors_are_wrapped_with_path_and_line(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "candidates.csv"
    path.write_text(
        ",".join(HEADERS[path.name]) + '\n"C0001,unterminated\n',
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusError,
        match=r"candidates\.csv:\d+: CSV parse error: unexpected end of data",
    ):
        validate_directory(fixture)


def test_entirely_blank_csv_row_is_rejected(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "claims.csv"
    rows = read_rows(path)
    rewrite_rows(path, rows + [blank_row(path.name)])

    with pytest.raises(CorpusError, match=r"claims\.csv:3: row is entirely blank"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("filename", "field"),
    [
        (filename, field)
        for filename, fields in REQUIRED_FIELDS_BY_FILE.items()
        for field in fields
    ],
)
def test_core_record_fields_are_required(tmp_path, filename, field):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / filename
    rows = read_rows(path)
    if not rows:
        rows = [minimal_required_row(filename)]
    rows[0][field] = ""
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match=rf"{field} is required"):
        validate_directory(fixture)


def test_included_source_requires_an_evidence_row(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    rewrite_rows(fixture / "evidence.csv", [])

    with pytest.raises(CorpusError, match=r"missing=\['Sample2026Course'\]"):
        validate_directory(fixture)


def test_included_source_requires_verified_metadata(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    rows = read_rows(fixture / "candidates.csv")
    rows[0]["metadata_status"] = "unverified"
    rewrite_rows(fixture / "candidates.csv", rows)

    with pytest.raises(CorpusError, match="requires metadata_status=verified"):
        validate_directory(fixture)


def test_production_taxonomy_matches_validator_default():
    taxonomy_path = (
        Path(__file__).resolve().parents[1] / "paper" / "data" / "taxonomy.json"
    )
    assert json.loads(taxonomy_path.read_text(encoding="utf-8")) == DEFAULT_TAXONOMY


def test_malformed_taxonomy_json_is_wrapped_with_path(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture / "taxonomy.json").write_text("{", encoding="utf-8")

    with pytest.raises(CorpusError, match=r"taxonomy\.json: invalid JSON"):
        validate_directory(fixture)


def test_taxonomy_requires_top_level_object(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    (fixture / "taxonomy.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(CorpusError, match=r"taxonomy\.json: top-level.*object"):
        validate_directory(fixture)


@pytest.mark.parametrize(
    ("domain_values", "message"),
    [
        ("ground", "domain.*must be a list"),
        (["ground", ""], "domain.*nonempty strings"),
        (["ground", 7], "domain.*nonempty strings"),
        (["ground", "ground"], "duplicate value.*domain"),
    ],
)
def test_taxonomy_requires_unique_nonempty_string_lists(
    tmp_path, domain_values, message
):
    fixture = build_valid_fixture(tmp_path)
    taxonomy = json.loads(json.dumps(DEFAULT_TAXONOMY))
    taxonomy["domain"] = domain_values
    (fixture / "taxonomy.json").write_text(
        json.dumps(taxonomy) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match=message):
        validate_directory(fixture)


def test_taxonomy_requires_every_vocabulary(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    taxonomy = json.loads(json.dumps(DEFAULT_TAXONOMY))
    del taxonomy["domain"]
    (fixture / "taxonomy.json").write_text(
        json.dumps(taxonomy) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="missing vocabulary 'domain'"):
        validate_directory(fixture)


def write_conflict(fixture: Path, **updates: str) -> None:
    row = minimal_required_row("conflicts.csv")
    row.update(updates)
    rewrite_rows(fixture / "conflicts.csv", [row])


@pytest.mark.parametrize(
    ("record_type", "record_key", "field"),
    [
        ("candidate", "C0001", "title"),
        ("evidence", "Sample2026Course", "domain"),
    ],
)
def test_conflict_target_resolves_supported_record(
    tmp_path, record_type, record_key, field
):
    fixture = build_valid_fixture(tmp_path)
    write_conflict(
        fixture,
        record_type=record_type,
        record_key=record_key,
        field=field,
    )

    validate_directory(fixture)


@pytest.mark.parametrize(
    ("record_type", "record_key", "field", "message"),
    [
        ("claim", "CL0001", "claim_text", "record_type='claim' is unsupported"),
        (
            "candidate",
            "missing",
            "title",
            "candidate record_key='missing' does not resolve",
        ),
        (
            "candidate",
            "C0001",
            "missing",
            "candidate field='missing' is not a column in candidates.csv",
        ),
        (
            "evidence",
            "missing",
            "domain",
            "evidence record_key='missing' does not resolve",
        ),
        (
            "evidence",
            "Sample2026Course",
            "missing",
            "evidence field='missing' is not a column in evidence.csv",
        ),
    ],
)
def test_conflict_target_must_resolve_to_supported_record_and_field(
    tmp_path, record_type, record_key, field, message
):
    fixture = build_valid_fixture(tmp_path)
    write_conflict(
        fixture,
        record_type=record_type,
        record_key=record_key,
        field=field,
    )

    with pytest.raises(CorpusError, match=message):
        validate_directory(fixture)


def test_resolved_conflict_requires_resolution_metadata(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    write_conflict(fixture, resolution="Use title A")

    with pytest.raises(CorpusError, match="resolver is required"):
        validate_directory(fixture)


def test_invalid_utf8_csv_is_wrapped_with_path_and_cause(tmp_path):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / "claims.csv"
    path.write_bytes(b"\xff")

    with pytest.raises(CorpusError, match=r"claims\.csv: invalid UTF-8") as error:
        validate_directory(fixture)

    assert isinstance(error.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize("filename", ["claims.csv", "metrics.csv"])
def test_declared_citation_lists_reject_trailing_empty_element(
    tmp_path, filename
):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / filename
    rows = read_rows(path)
    if not rows:
        rows = [minimal_required_row(filename)]
    rows[0]["cite_keys"] = "Sample2026Course;"
    rewrite_rows(path, rows)

    with pytest.raises(CorpusError, match="cite_keys contains an empty list element"):
        validate_directory(fixture)


@pytest.mark.parametrize("filename", ["claims.csv", "metrics.csv"])
def test_optional_declared_citation_lists_allow_blank(tmp_path, filename):
    fixture = build_valid_fixture(tmp_path)
    path = fixture / filename
    rows = read_rows(path)
    if not rows:
        rows = [minimal_required_row(filename)]
    rows[0]["cite_keys"] = ""
    rewrite_rows(path, rows)

    validate_directory(fixture)


def test_split_values_strips_whitespace_and_omits_empty_elements():
    assert split_values(" alpha ; ; beta; ") == ["alpha", "beta"]
