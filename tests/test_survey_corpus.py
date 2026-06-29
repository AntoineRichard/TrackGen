import csv
import json
from pathlib import Path

import pytest

from paper.scripts.merge_candidates import (
    main as merge_candidates_main,
    merge_candidate_files,
)
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
        writer = csv.DictWriter(handle, fieldnames=HEADERS[path.name])
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


def merge_candidate_row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["candidates.csv"], "")
    row.update(
        candidate_id="agent-0001",
        title="A Candidate Track Generator",
        source_type="paper",
        discovery_stream="test-stream",
        discovery_query="test query",
        discovery_agent="pytest",
        screening_status="candidate",
        metadata_status="unverified",
    )
    row.update(updates)
    return row


def write_candidate_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS["candidates.csv"])
        writer.writeheader()
        writer.writerows(rows)


def build_merge_fixture(
    tmp_path: Path,
    existing_rows: list[dict[str, str]],
    *agent_groups: list[dict[str, str]],
) -> tuple[Path, list[Path]]:
    existing_path = tmp_path / "candidates.csv"
    write_candidate_rows(existing_path, existing_rows)
    agent_paths = []
    for index, rows in enumerate(agent_groups, start=1):
        path = tmp_path / f"agent-{index}.csv"
        write_candidate_rows(path, rows)
        agent_paths.append(path)
    return existing_path, agent_paths


def test_merge_deduplicates_doi_prefixes_case_and_trailing_slash(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0042",
        title="Canonical Track Generation",
        doi="doi:10.1000/ABC",
        discovery_stream="bootstrap",
    )
    incoming = merge_candidate_row(
        candidate_id="BG-009",
        title="Canonical Track Generation",
        doi="https://doi.org/10.1000/abc/",
        discovery_stream="blind-ground",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0042"]
    assert conflicts == []


def test_merge_deduplicates_nfkd_casefolded_punctuation_free_titles(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0010",
        title="Stra\u00dfe: G\u00e9n\u00e9ration\u2014de   Pistes!",
        doi="",
    )
    incoming = merge_candidate_row(
        title="STRASSE generation de pistes",
        doi="NR",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["candidate_id"] == "C0010"
    assert conflicts == []


def test_merge_canonicalizes_all_discovery_provenance_with_semicolon_space(
    tmp_path,
):
    existing = merge_candidate_row(
        candidate_id="C0001",
        doi="10.1000/provenance",
        discovery_stream="z-stream; m-stream",
        discovery_query="z query; m query",
        discovery_agent="z-agent; m-agent",
    )
    incoming = merge_candidate_row(
        doi="https://doi.org/10.1000/PROVENANCE",
        discovery_stream="a-stream; z-stream",
        discovery_query="a query; z query",
        discovery_agent="a-agent; z-agent",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["discovery_stream"] == "a-stream; m-stream; z-stream"
    assert merged[0]["discovery_query"] == "a query; m query; z query"
    assert merged[0]["discovery_agent"] == "a-agent; m-agent; z-agent"


@pytest.mark.parametrize(
    ("field", "current", "proposed"),
    [
        ("title", "Canonical Track", "A Different Track"),
        ("authors", "A. Author", "B. Author"),
        ("year", "2020", "2021"),
        ("venue", "Venue A", "Venue B"),
        ("doi", "10.1000/canonical", "10.1000/different"),
        ("url", "https://example.test/a", "https://example.test/b"),
        ("source_type", "paper", "software"),
    ],
)
def test_merge_preserves_existing_bibliography_and_records_conflicts(
    tmp_path, field, current, proposed
):
    existing = merge_candidate_row(
        candidate_id="C0008",
        title="Canonical Track",
        authors="A. Author",
        year="2020",
        venue="Venue A",
        doi="10.1000/canonical",
        url="https://example.test/a",
        source_type="paper",
    )
    incoming = dict(existing)
    incoming["candidate_id"] = "agent-conflict"
    incoming[field] = proposed
    existing[field] = current
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert merged[0][field] == current
    assert len(conflicts) == 1
    assert conflicts[0]["record_type"] == "candidate"
    assert conflicts[0]["record_key"] == "C0008"
    assert conflicts[0]["field"] == field
    assert conflicts[0]["value_a"] == current
    assert conflicts[0]["value_b"] == proposed


def test_merge_deduplicates_repeated_equivalent_conflicts(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        title="Canonical Track",
        authors="A. Author",
        doi="10.1000/repeated",
    )
    incoming_a = merge_candidate_row(
        candidate_id="agent-a",
        title="Canonical Track",
        authors="B. Author",
        doi="10.1000/repeated",
        discovery_agent="agent-a",
    )
    incoming_b = dict(incoming_a)
    incoming_b.update(candidate_id="agent-b", discovery_agent="agent-b")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming_a], [incoming_b]
    )

    _, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [(row["field"], row["value_b"]) for row in conflicts] == [
        ("authors", "B. Author")
    ]


def test_merge_treats_nr_bibliographic_cells_as_missing(tmp_path):
    incoming = merge_candidate_row(
        cite_key="AgentVerifiedKey",
        authors="NR",
        year=" nr ",
        venue="NR",
        doi="NR",
        url="NR",
        source_type="NR",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert conflicts == []
    assert merged[0]["title"] == "A Candidate Track Generator"
    for field in ("authors", "year", "venue", "doi", "url", "source_type"):
        assert merged[0][field] == ""


def test_multiple_nr_doi_rows_do_not_share_an_identity(tmp_path):
    first = merge_candidate_row(title="Alpha Track", doi="NR")
    second = merge_candidate_row(title="Beta Track", doi="nr")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [first, second]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == ["C0001", "C0002"]
    assert {row["title"] for row in merged} == {"Alpha Track", "Beta Track"}
    assert all(row["doi"] == "" for row in merged)
    assert conflicts == []


def test_generic_urls_do_not_deduplicate_different_titles(tmp_path):
    first = merge_candidate_row(
        title="Alpha Track",
        url="https://github.com/example/course-generator",
    )
    second = merge_candidate_row(
        title="Beta Track",
        url="https://github.com/example/course-generator",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [first, second]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 2


def test_arxiv_doi_and_url_bridge_title_variants(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0016",
        title="Original Preprint Title",
        doi="10.48550/arXiv.2401.01234v2",
        url="",
    )
    incoming = merge_candidate_row(
        title="Published Title Variant",
        doi="NR",
        url="https://arxiv.org/pdf/2401.01234v1.pdf?download=1",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["candidate_id"] == "C0016"
    assert [(row["field"], row["value_b"]) for row in conflicts] == [
        ("title", "Published Title Variant")
    ]


def test_merge_rejects_a_row_bridging_multiple_existing_identities(tmp_path):
    first = merge_candidate_row(
        candidate_id="C0001",
        title="Alpha Track",
        doi="10.1000/alpha",
    )
    second = merge_candidate_row(
        candidate_id="C0002",
        title="Beta Track",
        doi="10.1000/beta",
    )
    bridge = merge_candidate_row(
        title="Beta Track",
        doi="10.1000/alpha",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [first, second], [bridge]
    )

    with pytest.raises(
        ValueError,
        match=r"bridges multiple existing identities.*C0001.*C0002",
    ):
        merge_candidate_files(existing_path, agent_paths)


def test_new_candidates_cannot_inherit_agent_verification_or_cite_key(tmp_path):
    incoming = merge_candidate_row(
        cite_key="AgentClaimedVerifiedKey",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["cite_key"] == ""
    assert merged[0]["metadata_status"] == "unverified"


def test_existing_bootstrap_verification_fields_remain_stable(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        cite_key="",
        doi="10.1000/bootstrap",
        metadata_status="unverified",
    )
    incoming = merge_candidate_row(
        cite_key="IncomingKey",
        doi="10.1000/bootstrap",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["cite_key"] == ""
    assert merged[0]["metadata_status"] == "unverified"


def test_new_excluded_candidate_retains_specific_reason(tmp_path):
    incoming = merge_candidate_row(
        cite_key="ExcludedAgentKey",
        screening_status="excluded",
        exclusion_reason="Scenery only; no course geometry contribution.",
        metadata_status="verified",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["screening_status"] == "excluded"
    assert (
        merged[0]["exclusion_reason"]
        == "Scenery only; no course geometry contribution."
    )
    assert merged[0]["cite_key"] == ""
    assert merged[0]["metadata_status"] == "unverified"


@pytest.mark.parametrize("reason", ["", "NR", " nr "])
def test_new_excluded_candidate_requires_a_specific_reason(tmp_path, reason):
    incoming = merge_candidate_row(
        screening_status="excluded",
        exclusion_reason=reason,
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [], [incoming]
    )

    with pytest.raises(ValueError, match="excluded.*specific exclusion_reason"):
        merge_candidate_files(existing_path, agent_paths)


def test_next_stable_id_follows_maximum_without_filling_retired_gap(tmp_path):
    existing = [
        merge_candidate_row(candidate_id="C0071", title="Track 71"),
        merge_candidate_row(candidate_id="C0073", title="Track 73"),
        merge_candidate_row(candidate_id="C0100", title="Track 100"),
    ]
    incoming = merge_candidate_row(title="A Newly Discovered Track")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, existing, [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert [row["candidate_id"] for row in merged] == [
        "C0071",
        "C0073",
        "C0100",
        "C0101",
    ]
    assert "C0072" not in {row["candidate_id"] for row in merged}


def test_merge_preserves_all_metadata_evidence_origins(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0001",
        doi="10.1000/evidence",
        metadata_evidence="source-b; source-a",
    )
    incoming = merge_candidate_row(
        doi="10.1000/evidence",
        metadata_evidence="source-a; source-c",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [incoming]
    )

    merged, _ = merge_candidate_files(existing_path, agent_paths)

    assert merged[0]["metadata_evidence"] == "source-b; source-a; source-c"


def test_merge_csv_parse_errors_include_path_and_line(tmp_path):
    existing_path, _ = build_merge_fixture(tmp_path, [])
    bad_agent = tmp_path / "bad-agent.csv"
    bad_agent.write_text(
        ",".join(HEADERS["candidates.csv"])
        + '\n"agent-1,unterminated\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"bad-agent\.csv:\d+: CSV parse error: unexpected end of data",
    ):
        merge_candidate_files(existing_path, [bad_agent])


def test_merge_invalid_utf8_errors_include_path_and_cause(tmp_path):
    existing_path, _ = build_merge_fixture(tmp_path, [])
    bad_agent = tmp_path / "bad-utf8.csv"
    bad_agent.write_bytes(b"\xff")

    with pytest.raises(ValueError, match=r"bad-utf8\.csv: invalid UTF-8") as error:
        merge_candidate_files(existing_path, [bad_agent])

    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_merge_is_independent_of_agent_path_and_row_order(tmp_path):
    duplicate_a = merge_candidate_row(
        candidate_id="agent-a",
        title="Shared Track",
        authors="A. Author",
        discovery_agent="agent-a",
    )
    duplicate_b = merge_candidate_row(
        candidate_id="agent-b",
        title="Shared Track",
        authors="B. Author",
        discovery_agent="agent-b",
    )
    unique = merge_candidate_row(
        candidate_id="agent-c",
        title="Unique Track",
        discovery_agent="agent-c",
    )
    first_existing, first_agents = build_merge_fixture(
        tmp_path / "first",
        [],
        [duplicate_b, duplicate_a],
        [unique],
    )
    second_existing, second_agents = build_merge_fixture(
        tmp_path / "second",
        [],
        [duplicate_a, duplicate_b],
        [unique],
    )

    first_result = merge_candidate_files(first_existing, first_agents)
    second_result = merge_candidate_files(
        second_existing, list(reversed(second_agents))
    )

    assert first_result == second_result


def test_rerunning_merge_with_same_agent_inputs_is_idempotent(tmp_path):
    existing = merge_candidate_row(
        candidate_id="C0100",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/stable",
    )
    duplicate = merge_candidate_row(
        title="Stable Track",
        authors="Conflicting Author",
        doi="https://doi.org/10.1000/STABLE",
    )
    new = merge_candidate_row(title="New Track", doi="NR")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [duplicate, new]
    )

    first_merged, first_conflicts = merge_candidate_files(
        existing_path, agent_paths
    )
    write_candidate_rows(existing_path, first_merged)
    second_merged, second_conflicts = merge_candidate_files(
        existing_path, agent_paths
    )

    assert second_merged == first_merged
    assert second_conflicts == first_conflicts


def test_merge_cli_dry_run_reports_identity_stream_and_conflict_counts(
    tmp_path, capsys
):
    existing = merge_candidate_row(
        candidate_id="C0100",
        title="Stable Track",
        doi="10.1000/cli",
        discovery_stream="bootstrap",
    )
    duplicate = merge_candidate_row(
        title="Conflicting Published Title",
        doi="https://doi.org/10.1000/CLI",
        discovery_stream="blind-ground",
    )
    new = merge_candidate_row(
        title="New CLI Track",
        discovery_stream="aware-geometry-rl",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [duplicate, new]
    )
    original = existing_path.read_bytes()

    result = merge_candidates_main(
        [
            "--existing",
            str(existing_path),
            "--agent",
            str(agent_paths[0]),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "mode=dry-run" in output
    assert "merged_total=2" in output
    assert "new_count=1" in output
    assert "duplicate_matches[doi][blind-ground]=1" in output
    assert "conflicts[title]=1" in output
    assert "conflict_total=1" in output
    assert existing_path.read_bytes() == original
    assert not (tmp_path / "conflicts.csv").exists()


def test_merge_cli_write_has_exact_schemas_and_is_idempotent(tmp_path, capsys):
    existing = merge_candidate_row(
        candidate_id="C0100",
        title="Stable Track",
        authors="Original Author",
        doi="10.1000/write",
    )
    duplicate = merge_candidate_row(
        title="Stable Track",
        authors="Conflicting Author",
        doi="10.1000/write",
    )
    new = merge_candidate_row(title="New Written Track")
    existing_path, agent_paths = build_merge_fixture(
        tmp_path, [existing], [duplicate, new]
    )
    arguments = [
        "--existing",
        str(existing_path),
        "--agent",
        str(agent_paths[0]),
        "--write",
    ]

    assert merge_candidates_main(arguments) == 0
    capsys.readouterr()
    conflicts_path = tmp_path / "conflicts.csv"
    first_candidates = existing_path.read_bytes()
    first_conflicts = conflicts_path.read_bytes()
    with existing_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == HEADERS[
            "candidates.csv"
        ]
    with conflicts_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == HEADERS[
            "conflicts.csv"
        ]

    assert merge_candidates_main(arguments) == 0
    capsys.readouterr()

    assert existing_path.read_bytes() == first_candidates
    assert conflicts_path.read_bytes() == first_conflicts


def test_explicit_exclusion_wins_for_a_new_identity_seen_as_candidate(tmp_path):
    candidate = merge_candidate_row(
        candidate_id="blind-1",
        title="Overlapping Discovery",
        doi="10.1000/overlap",
        screening_status="candidate",
        exclusion_reason="",
    )
    excluded = merge_candidate_row(
        candidate_id="aware-1",
        title="Overlapping Discovery",
        doi="10.1000/overlap",
        screening_status="excluded",
        exclusion_reason="No course geometry contribution.",
    )
    existing_path, agent_paths = build_merge_fixture(
        tmp_path,
        [],
        [candidate],
        [excluded],
    )

    merged, conflicts = merge_candidate_files(existing_path, agent_paths)

    assert len(merged) == 1
    assert merged[0]["screening_status"] == "excluded"
    assert merged[0]["exclusion_reason"] == "No course geometry contribution."
    assert conflicts == []
