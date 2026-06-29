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
