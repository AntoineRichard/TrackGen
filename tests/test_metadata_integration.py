from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

import paper.scripts.integrate_metadata as metadata_integration
from paper.scripts.integrate_metadata import (
    BIBLIOGRAPHY_HEADER,
    CONFLICT_RESULT_HEADER,
    METADATA_RESULT_HEADER,
    MetadataIntegrationError,
    integrate_metadata,
    main,
    write_integration_outputs,
)
from paper.scripts.prepare_metadata_batches import (
    MANIFEST_HEADER,
    build_manifest,
)
from paper.scripts.validate_corpus import HEADERS


def candidate_row(candidate_id: str, **values: str) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["candidates.csv"], "")
    row.update(
        candidate_id=candidate_id,
        title=f"Track Study {candidate_id}",
        discovery_stream="blind-ground",
        discovery_query=f"query::{candidate_id}",
        discovery_agent="discovery-agent",
        screening_status="candidate",
        metadata_status="unverified",
        metadata_evidence="discovery only",
    )
    row.update(values)
    return row


def conflict_row(
    conflict_id: str,
    candidate_id: str,
    field: str,
    **values: str,
) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["conflicts.csv"], "")
    row.update(
        conflict_id=conflict_id,
        record_type="candidate",
        record_key=candidate_id,
        field=field,
        value_a=f"old-{conflict_id}",
        value_b=f"new-{conflict_id}",
    )
    row.update(values)
    return row


def write_rows(
    path: Path,
    header: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=header,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def metadata_result_row(
    manifest_row: dict[str, str],
    candidate: dict[str, str],
    **values: str,
) -> dict[str, str]:
    candidate_id = candidate["candidate_id"]
    doi = f"10.1000/{candidate_id.casefold()}"
    publisher = f"publisher::https://publisher.example/{candidate_id}"
    doi_source = f"doi::https://doi.org/{doi}"
    row = dict.fromkeys(METADATA_RESULT_HEADER, "NR")
    row.update(
        candidate_id=candidate_id,
        input_sha256=manifest_row["input_sha256"],
        agent_id=manifest_row["batch_id"],
        verified_on="2026-06-30",
        metadata_status="verified",
        title=candidate["title"],
        authors="Alice Example",
        year="2025",
        venue="Journal of Generated Tracks",
        doi=doi,
        url=f"https://doi.org/{doi}",
        source_type="paper",
        title_evidence=publisher,
        authors_evidence=publisher,
        year_evidence=publisher,
        venue_evidence=publisher,
        doi_evidence=doi_source,
        url_evidence=doi_source,
        source_type_evidence=publisher,
        bib_entry_type="article",
        bib_venue_field="journal",
        bib_url=f"https://doi.org/{doi}",
        key_author="Example",
        author_kinds="personal",
    )
    row.update(values)
    return row


@dataclass
class IntegrationCase:
    candidates_path: Path
    conflicts_path: Path
    manifest_path: Path
    result_pairs: list[tuple[Path, Path]]
    metadata_rows: dict[str, list[dict[str, str]]]
    conflict_rows: dict[str, list[dict[str, str]]]
    candidates: list[dict[str, str]]
    conflicts: list[dict[str, str]]

    def write_results(self) -> None:
        for number, (metadata_path, conflict_path) in enumerate(
            self.result_pairs, start=1
        ):
            batch_id = f"metadata-{number:02d}"
            write_rows(
                metadata_path,
                METADATA_RESULT_HEADER,
                self.metadata_rows[batch_id],
            )
            write_rows(
                conflict_path,
                CONFLICT_RESULT_HEADER,
                self.conflict_rows[batch_id],
            )

    @property
    def all_metadata_rows(self) -> list[dict[str, str]]:
        return [
            row
            for batch_id in sorted(self.metadata_rows)
            for row in self.metadata_rows[batch_id]
        ]

    @property
    def all_conflict_rows(self) -> list[dict[str, str]]:
        return [
            row
            for batch_id in sorted(self.conflict_rows)
            for row in self.conflict_rows[batch_id]
        ]


def build_case(
    tmp_path: Path,
    candidates: list[dict[str, str]] | None = None,
    conflicts: list[dict[str, str]] | None = None,
) -> IntegrationCase:
    candidates = candidates or [candidate_row("C0001")]
    conflicts = conflicts or []
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    candidates_path = inputs / "candidates.csv"
    conflicts_path = inputs / "conflicts.csv"
    manifest_path = inputs / "metadata_manifest.csv"
    write_rows(candidates_path, HEADERS["candidates.csv"], candidates)
    write_rows(conflicts_path, HEADERS["conflicts.csv"], conflicts)
    manifest = build_manifest(candidates_path, conflicts_path)
    write_rows(manifest_path, MANIFEST_HEADER, manifest)

    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    manifest_by_id = {row["candidate_id"]: row for row in manifest}
    metadata_rows = {
        f"metadata-{number:02d}": [] for number in range(1, 7)
    }
    conflict_rows = {
        f"metadata-{number:02d}": [] for number in range(1, 7)
    }
    for manifest_row in manifest:
        candidate = candidates_by_id[manifest_row["candidate_id"]]
        metadata_rows[manifest_row["batch_id"]].append(
            metadata_result_row(manifest_row, candidate)
        )

    candidate_by_cite_key = {
        row["cite_key"]: row["candidate_id"]
        for row in candidates
        if row["cite_key"]
    }
    for conflict in conflicts:
        candidate_id = (
            conflict["record_key"]
            if conflict["record_type"] == "candidate"
            else candidate_by_cite_key[conflict["record_key"]]
        )
        manifest_row = manifest_by_id[candidate_id]
        metadata_row = next(
            row
            for row in metadata_rows[manifest_row["batch_id"]]
            if row["candidate_id"] == candidate_id
        )
        resolution = (
            metadata_row[conflict["field"]]
            if conflict["record_type"] == "candidate"
            and conflict["field"]
            in {"title", "authors", "year", "venue", "doi", "url", "source_type"}
            else "NR"
        )
        conflict_rows[manifest_row["batch_id"]].append(
            {
                "candidate_id": candidate_id,
                "input_sha256": manifest_row["input_sha256"],
                "agent_id": manifest_row["batch_id"],
                "input_conflict_id": conflict["conflict_id"],
                "field": conflict["field"],
                "value_a": conflict["value_a"],
                "value_b": conflict["value_b"],
                "resolution": resolution,
                "resolution_evidence": (
                    "publisher::https://publisher.example/conflict"
                    if resolution != "NR"
                    else "NR"
                ),
            }
        )

    results = tmp_path / "results"
    results.mkdir()
    result_pairs = [
        (
            results / f"metadata-{number:02d}.csv",
            results / f"metadata-{number:02d}-conflicts.csv",
        )
        for number in range(1, 7)
    ]
    case = IntegrationCase(
        candidates_path=candidates_path,
        conflicts_path=conflicts_path,
        manifest_path=manifest_path,
        result_pairs=result_pairs,
        metadata_rows=metadata_rows,
        conflict_rows=conflict_rows,
        candidates=candidates,
        conflicts=conflicts,
    )
    case.write_results()
    return case


def integrate(case: IntegrationCase):
    return integrate_metadata(
        case.candidates_path,
        case.conflicts_path,
        case.manifest_path,
        case.result_pairs,
    )


def result_row(case: IntegrationCase, candidate_id: str) -> dict[str, str]:
    return next(
        row
        for row in case.all_metadata_rows
        if row["candidate_id"] == candidate_id
    )


def conflict_result_row(
    case: IntegrationCase, conflict_id: str
) -> dict[str, str]:
    return next(
        row
        for row in case.all_conflict_rows
        if row["input_conflict_id"] == conflict_id
    )


def test_public_headers_are_exact_and_valid_results_integrate(tmp_path):
    assert METADATA_RESULT_HEADER == (
        "candidate_id",
        "input_sha256",
        "agent_id",
        "verified_on",
        "metadata_status",
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "url",
        "source_type",
        "title_evidence",
        "authors_evidence",
        "year_evidence",
        "venue_evidence",
        "doi_evidence",
        "url_evidence",
        "source_type_evidence",
        "bib_entry_type",
        "bib_venue_field",
        "bib_url",
        "key_author",
        "author_kinds",
        "notes",
    )
    assert CONFLICT_RESULT_HEADER == (
        "candidate_id",
        "input_sha256",
        "agent_id",
        "input_conflict_id",
        "field",
        "value_a",
        "value_b",
        "resolution",
        "resolution_evidence",
    )
    assert BIBLIOGRAPHY_HEADER == (
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
    case = build_case(tmp_path)

    result = integrate(case)

    assert [row["candidate_id"] for row in result.candidates] == ["C0001"]
    candidate = result.candidates[0]
    assert candidate["metadata_status"] == "verified"
    assert candidate["cite_key"] == "Example2025TrackStudy"
    assert candidate["metadata_evidence"].startswith("doi::https://doi.org/")
    assert result.bibliography == result.bibliography_rows
    assert result.bibtex == result.bibtex_text
    assert result.bibliography[0]["url"] == ""
    assert "@article{Example2025TrackStudy," in result.bibtex
    assert "  journal = {Journal of Generated Tracks}" in result.bibtex
    assert "  url =" not in result.bibtex


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stale_hash", "input_sha256"),
        ("impersonate", "agent_id"),
        ("missing", "missing"),
        ("extra", "extra"),
        ("duplicate", "duplicate"),
    ],
)
def test_candidate_result_assignment_is_exact(
    tmp_path, mutation, message
):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    if mutation == "stale_hash":
        row["input_sha256"] = "0" * 64
    elif mutation == "impersonate":
        row["agent_id"] = "metadata-06"
    elif mutation == "missing":
        case.metadata_rows[row["agent_id"]].remove(row)
    elif mutation == "extra":
        extra = dict(row, candidate_id="C9999")
        case.metadata_rows[row["agent_id"]].append(extra)
    else:
        case.metadata_rows[row["agent_id"]].append(dict(row))
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match=message):
        integrate(case)


def test_result_files_require_exact_headers_nonblank_cells_and_valid_csv(
    tmp_path,
):
    case = build_case(tmp_path)
    metadata_path = case.result_pairs[0][0]
    row = case.all_metadata_rows[0]
    row["notes"] = ""
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="NR|blank"):
        integrate(case)

    write_rows(metadata_path, tuple(reversed(METADATA_RESULT_HEADER)), [row])
    with pytest.raises(MetadataIntegrationError, match="headers"):
        integrate(case)

    metadata_path.write_text(
        ",".join(METADATA_RESULT_HEADER) + '\n"unterminated',
        encoding="utf-8",
    )
    with pytest.raises(MetadataIntegrationError, match="CSV parse"):
        integrate(case)


def test_exactly_six_distinct_result_pairs_are_required(tmp_path):
    case = build_case(tmp_path)

    with pytest.raises(MetadataIntegrationError, match="exactly 6"):
        integrate_metadata(
            case.candidates_path,
            case.conflicts_path,
            case.manifest_path,
            case.result_pairs[:-1],
        )
    with pytest.raises(MetadataIntegrationError, match="distinct"):
        integrate_metadata(
            case.candidates_path,
            case.conflicts_path,
            case.manifest_path,
            [*case.result_pairs[:-1], case.result_pairs[0]],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("duplicate", "duplicate"),
        ("unknown", "unknown"),
        ("wrong_values", "value_a|does not match"),
        ("wrong_hash", "input_sha256"),
    ],
)
def test_existing_conflict_results_are_exact(tmp_path, mutation, message):
    conflict = conflict_row(
        "XINPUT000001",
        "C0001",
        "title",
        value_a="Old title",
        value_b="Track Study C0001",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    row = conflict_result_row(case, conflict["conflict_id"])
    batch_rows = case.conflict_rows[row["agent_id"]]
    if mutation == "missing":
        batch_rows.remove(row)
    elif mutation == "duplicate":
        batch_rows.append(dict(row))
    elif mutation == "unknown":
        row["input_conflict_id"] = "XUNKNOWN"
    elif mutation == "wrong_values":
        row["value_a"] = "tampered"
    else:
        row["input_sha256"] = "f" * 64
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match=message):
        integrate(case)


def test_crossref_only_and_discovery_evidence_are_rejected(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    for field in (
        "title_evidence",
        "authors_evidence",
        "year_evidence",
        "venue_evidence",
        "doi_evidence",
        "url_evidence",
        "source_type_evidence",
    ):
        row[field] = "crossref::https://api.crossref.org/works/10.1000/test"
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="Crossref|crossref"):
        integrate(case)

    row["title_evidence"] = (
        "publisher::https://publisher.example/paper;"
        "semantic_scholar::https://semanticscholar.org/paper/example"
    )
    case.write_results()
    with pytest.raises(MetadataIntegrationError, match="discovery|aggregator"):
        integrate(case)


def test_published_paper_rejects_repository_only_core_evidence(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    repository = "official_repository::https://github.com/example/project"
    for field in (
        "title_evidence",
        "authors_evidence",
        "year_evidence",
        "venue_evidence",
        "source_type_evidence",
    ):
        row[field] = repository
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="publisher|proceedings"):
        integrate(case)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.crossref.org/works/10.1000/c0001",
        "https://api.crossref.org./works/10.1000/c0001",
        "https://doi.org/10.1000/c0001",
    ],
)
def test_evidence_kind_cannot_relabel_corroborating_host(tmp_path, url):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row["title_evidence"] = f"publisher::{url}"
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match="Crossref|crossref|DOI|doi|host",
    ):
        integrate(case)


def test_evidence_url_rejects_invalid_port(tmp_path):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["title_evidence"] = (
        "publisher::https://publisher.example:not-a-port/C0001"
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="port|URL|authority"):
        integrate(case)


def test_doi_is_normalized_before_candidate_and_bibliography_storage(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        doi="doi:10.1000/C0001",
        url="https://doi.org/10.1000/c0001",
        bib_url="https://DOI.ORG/10.1000/C0001",
    )
    case.write_results()

    result = integrate(case)

    assert result.candidates[0]["doi"] == "10.1000/c0001"
    assert result.bibliography[0]["doi"] == "10.1000/c0001"
    assert result.bibliography[0]["url"] == ""


def test_doi_conflict_resolution_is_stored_canonically(tmp_path):
    conflict = conflict_row(
        "XDOI",
        "C0001",
        "doi",
        value_a="10.1000/old",
        value_b="10.1000/c0001",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    metadata = result_row(case, "C0001")
    metadata.update(
        doi="doi:10.1000/C0001",
        url="https://doi.org/10.1000/c0001",
        bib_url="https://doi.org/10.1000/c0001",
    )
    resolution = conflict_result_row(case, "XDOI")
    resolution["resolution"] = "https://DOI.ORG/10.1000/C0001"
    case.write_results()

    result = integrate(case)

    assert result.candidates[0]["doi"] == "10.1000/c0001"
    assert result.conflicts[0]["resolution"] == "10.1000/c0001"


def test_distinct_evidenced_doi_resolver_url_is_not_omitted(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    alternate = "https://doi.org/10.1000/alternate"
    row["bib_url"] = alternate
    row["url_evidence"] += f";doi::{alternate}"
    case.write_results()

    result = integrate(case)

    assert result.bibliography[0]["url"] == alternate
    assert f"url = {{{alternate}}}" in result.bibtex


def test_malformed_populated_doi_is_rejected(tmp_path):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["doi"] = "not-a-doi"
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="DOI|doi"):
        integrate(case)


@pytest.mark.parametrize(
    "doi",
    [
        "ftp://doi.org/10.1000/c0001",
        "https://doi.org:99999/10.1000/c0001",
        "https://doi.org",
    ],
)
def test_doi_resolver_requires_http_valid_authority_and_path(tmp_path, doi):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["doi"] = doi
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match="DOI|doi|port|HTTP|path",
    ):
        integrate(case)


def test_paper_requires_a_canonical_doi_or_stable_url(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        doi="NR",
        doi_evidence="NR",
        url="NR",
        url_evidence="NR",
        bib_url="NR",
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="DOI|stable URL"):
        integrate(case)


def test_bibliography_url_rejects_unrelated_location(tmp_path):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["bib_url"] = (
        "https://unrelated.example/paper.pdf"
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="bib_url|bibliography URL"):
        integrate(case)


def test_bibliography_accepts_evidenced_accessible_preprint_url(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    preprint = "https://arxiv.org/abs/2501.00001"
    row["bib_url"] = preprint
    row["url_evidence"] += (
        f";preprint_repository::{preprint}"
    )
    case.write_results()

    result = integrate(case)

    assert result.bibliography[0]["url"] == preprint
    assert f"url = {{{preprint}}}" in result.bibtex


def test_preprint_only_entry_accepts_repository_evidence_for_core_fields(
    tmp_path,
):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    preprint_url = "https://arxiv.org/abs/2501.00001"
    repository = f"preprint_repository::{preprint_url}"
    doi = "10.48550/arxiv.2501.00001"
    row.update(
        venue="arXiv",
        doi=doi,
        url=preprint_url,
        source_type="preprint",
        title_evidence=repository,
        authors_evidence=repository,
        year_evidence=repository,
        venue_evidence=repository,
        doi_evidence=f"doi::https://doi.org/{doi}",
        url_evidence=repository,
        source_type_evidence=repository,
        bib_url=preprint_url,
    )
    case.write_results()

    result = integrate(case)

    assert result.candidates[0]["metadata_status"] == "verified"
    assert result.bibliography[0]["doi"] == doi
    assert result.bibliography[0]["url"] == preprint_url


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", " Track Study C0001"),
        ("authors", "Alice Example "),
        (
            "title_evidence",
            "publisher::https://publisher.example/C0001 ",
        ),
    ],
)
def test_result_cells_reject_surrounding_whitespace(tmp_path, field, value):
    case = build_case(tmp_path)
    case.all_metadata_rows[0][field] = value
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="whitespace"):
        integrate(case)


def test_nr_cannot_be_mixed_into_author_lists(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        authors="NR;Alice Example",
        author_kinds="personal;personal",
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="NR"):
        integrate(case)


def test_nr_cannot_be_mixed_into_evidence_lists(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row["title_evidence"] = (
        "NR;publisher::https://publisher.example/C0001"
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="NR"):
        integrate(case)


def test_nr_cannot_be_mixed_into_scalar_venue(tmp_path):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["venue"] = (
        "Journal of Generated Tracks;NR"
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="NR"):
        integrate(case)


def test_verified_official_undated_artifact_uses_nodate_corporate_key(
    tmp_path,
):
    case = build_case(
        tmp_path,
        candidates=[candidate_row("C0001", title="The Über Track Toolkit")],
    )
    row = case.all_metadata_rows[0]
    official = "official_repository::https://github.com/example/uber-track"
    row.update(
        authors="École Polytechnique Fédérale de Lausanne",
        year="NR",
        venue="NR",
        doi="NR",
        url="https://github.com/example/uber-track",
        source_type="software",
        title_evidence=official,
        authors_evidence=official,
        year_evidence="NR",
        venue_evidence="NR",
        doi_evidence="NR",
        url_evidence=official,
        source_type_evidence=official,
        bib_entry_type="misc",
        bib_venue_field="NR",
        bib_url="https://github.com/example/uber-track",
        key_author="ÉPFL",
        author_kinds="corporate",
        notes="The project has no publication year, venue, or DOI.",
    )
    case.write_results()

    result = integrate(case)

    assert result.candidates[0]["cite_key"] == "EPFLNodateUberTrack"
    assert result.bibliography[0]["year"] == ""
    assert result.bibliography[0]["venue_field"] == ""
    assert (
        "author = {{École Polytechnique Fédérale de Lausanne}}"
        in result.bibtex
    )


@pytest.mark.parametrize("notes", ["NR", "N/A", "Not available"])
def test_undated_artifact_requires_explanatory_notes(tmp_path, notes):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    official = "official_documentation::https://docs.example/tool"
    row.update(
        year="NR",
        venue="NR",
        doi="NR",
        source_type="documentation",
        title_evidence=official,
        authors_evidence=official,
        year_evidence="NR",
        venue_evidence="NR",
        doi_evidence="NR",
        url_evidence=official,
        source_type_evidence=official,
        bib_entry_type="misc",
        bib_venue_field="NR",
        bib_url="https://docs.example/tool",
        notes=notes,
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="notes"):
        integrate(case)


@pytest.mark.parametrize(
    ("authors", "author_kinds"),
    [
        ("One Author;Second Author", "personal"),
        ("One Author", "individual"),
        ("One Author et al.", "personal"),
    ],
)
def test_author_kinds_align_with_complete_authors(
    tmp_path, authors, author_kinds
):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row["authors"] = authors
    row["author_kinds"] = author_kinds
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="author"):
        integrate(case)


def test_immutable_candidate_columns_are_preserved(tmp_path):
    original = candidate_row(
        "C0001",
        discovery_stream="stream-b; stream-a",
        discovery_query="query-b; query-a",
        discovery_agent="agent-b; agent-a",
        screening_status="excluded",
        exclusion_reason="Out of scope",
    )
    case = build_case(tmp_path, candidates=[original])

    result = integrate(case)

    integrated = result.candidates[0]
    for field in (
        "candidate_id",
        "discovery_stream",
        "discovery_query",
        "discovery_agent",
        "screening_status",
        "exclusion_reason",
    ):
        assert integrated[field] == original[field]
    assert integrated["cite_key"] == ""
    assert result.bibliography == []
    assert result.bibtex == ""


def test_screening_and_evidence_conflicts_remain_outside_metadata(tmp_path):
    candidate = candidate_row("C0001", cite_key="Existing2024Key")
    conflicts = [
        conflict_row(
            "XSCREENING",
            "C0001",
            "screening_status",
            value_a="candidate",
            value_b="excluded",
        ),
        conflict_row(
            "XEVIDENCE",
            "Existing2024Key",
            "domain",
            record_type="evidence",
            value_a="ground",
            value_b="mixed",
        ),
    ]
    case = build_case(tmp_path, candidates=[candidate], conflicts=conflicts)

    result = integrate(case)

    assert result.candidates[0]["metadata_status"] == "verified"
    assert {row["resolution"] for row in result.conflicts} == {""}

    row = conflict_result_row(case, "XSCREENING")
    row["resolution"] = "excluded"
    row["resolution_evidence"] = (
        "official_documentation::https://example.org/screening"
    )
    case.write_results()
    with pytest.raises(MetadataIntegrationError, match="cannot resolve"):
        integrate(case)


def test_unresolved_bibliographic_conflict_requires_conflict_status(tmp_path):
    conflict = conflict_row(
        "XTITLE",
        "C0001",
        "title",
        value_a="Track Study C0001",
        value_b="Alternative title",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    resolution = conflict_result_row(case, "XTITLE")
    resolution["resolution"] = "NR"
    resolution["resolution_evidence"] = "NR"
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="metadata_status=conflict"):
        integrate(case)

    result_row(case, "C0001")["metadata_status"] = "conflict"
    case.write_results()
    result = integrate(case)
    assert result.candidates[0]["metadata_status"] == "conflict"
    assert result.candidates[0]["cite_key"] == ""


def test_conflict_status_requires_an_unresolved_bibliographic_conflict(
    tmp_path,
):
    case = build_case(tmp_path)
    result_row(case, "C0001")["metadata_status"] = "conflict"
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="iff|if and only if"):
        integrate(case)


def test_screening_only_conflict_does_not_justify_metadata_conflict_status(
    tmp_path,
):
    conflict = conflict_row(
        "XSCREENING",
        "C0001",
        "screening_status",
        value_a="candidate",
        value_b="included",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    result_row(case, "C0001")["metadata_status"] = "conflict"
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="iff|if and only if"):
        integrate(case)


@pytest.mark.parametrize(
    ("screening_status", "metadata_status"),
    [("included", "unverified"), ("boundary", "conflict")],
)
def test_included_and_boundary_candidates_must_finish_verified(
    tmp_path, screening_status, metadata_status
):
    conflicts = []
    if metadata_status == "conflict":
        conflicts.append(
            conflict_row(
                "XTITLE",
                "C0001",
                "title",
                value_a="Track Study C0001",
                value_b="Alternate Track Study",
            )
        )
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row("C0001", screening_status=screening_status)
        ],
        conflicts=conflicts,
    )
    metadata = result_row(case, "C0001")
    metadata["metadata_status"] = metadata_status
    if conflicts:
        resolution = conflict_result_row(case, "XTITLE")
        resolution["resolution"] = "NR"
        resolution["resolution_evidence"] = "NR"
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="verified|cite_key"):
        integrate(case)


def test_excluded_candidate_may_remain_unverified_and_keyless(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row(
                "C0001",
                screening_status="excluded",
                exclusion_reason="Out of scope",
            )
        ],
    )
    metadata = result_row(case, "C0001")
    metadata.update(metadata_status="unverified", verified_on="NR")
    case.write_results()

    result = integrate(case)

    assert result.candidates[0]["cite_key"] == ""
    assert result.bibliography == []


def test_bibliographic_conflict_resolution_updates_only_resolution_fields(
    tmp_path,
):
    conflict = conflict_row(
        "XTITLE",
        "C0001",
        "title",
        value_a="Old title",
        value_b="Track Study C0001",
    )
    case = build_case(tmp_path, conflicts=[conflict])

    result = integrate(case)

    output = result.conflicts[0]
    for field in (
        "conflict_id",
        "record_type",
        "record_key",
        "field",
        "value_a",
        "value_b",
    ):
        assert output[field] == conflict[field]
    assert output["resolution"] == "Track Study C0001"
    assert output["resolver"] == result_row(case, "C0001")["agent_id"]
    assert output["resolution_evidence"].startswith("publisher::")


def test_published_conflict_resolution_rejects_repository_evidence(tmp_path):
    conflict = conflict_row(
        "XTITLE",
        "C0001",
        "title",
        value_a="Old title",
        value_b="Track Study C0001",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    resolution = conflict_result_row(case, "XTITLE")
    resolution["resolution_evidence"] = (
        "official_repository::https://github.com/example/project"
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="publisher|proceedings"):
        integrate(case)


def test_published_conflict_resolution_accepts_proceedings_evidence(tmp_path):
    conflict = conflict_row(
        "XTITLE",
        "C0001",
        "title",
        value_a="Old title",
        value_b="Track Study C0001",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    resolution = conflict_result_row(case, "XTITLE")
    resolution["resolution_evidence"] = "proceedings::https://conf.example/paper"
    case.write_results()

    result = integrate(case)

    assert result.conflicts[0]["resolver"] == resolution["agent_id"]


def test_preprint_conflict_resolution_accepts_preprint_repository(tmp_path):
    conflict = conflict_row(
        "XTITLE",
        "C0001",
        "title",
        value_a="Old title",
        value_b="Track Study C0001",
    )
    case = build_case(tmp_path, conflicts=[conflict])
    metadata = result_row(case, "C0001")
    preprint_url = "https://arxiv.org/abs/2501.00001"
    repository = f"preprint_repository::{preprint_url}"
    doi = "10.48550/arxiv.2501.00001"
    metadata.update(
        venue="arXiv",
        doi=doi,
        url=preprint_url,
        source_type="preprint",
        title_evidence=repository,
        authors_evidence=repository,
        year_evidence=repository,
        venue_evidence=repository,
        doi_evidence=f"doi::https://doi.org/{doi}",
        url_evidence=repository,
        source_type_evidence=repository,
        bib_url=preprint_url,
    )
    resolution = conflict_result_row(case, "XTITLE")
    resolution["resolution_evidence"] = repository
    case.write_results()

    result = integrate(case)

    assert result.candidates[0]["metadata_status"] == "verified"
    assert result.conflicts[0]["resolution_evidence"] == repository


def test_new_disagreement_gets_stable_conflict_id(tmp_path):
    case = build_case(tmp_path)
    metadata = result_row(case, "C0001")
    metadata["metadata_status"] = "conflict"
    new_conflict = {
        "candidate_id": "C0001",
        "input_sha256": metadata["input_sha256"],
        "agent_id": metadata["agent_id"],
        "input_conflict_id": "NEW-local-1",
        "field": "venue",
        "value_a": "Venue A",
        "value_b": "Venue B",
        "resolution": "NR",
        "resolution_evidence": "NR",
    }
    case.conflict_rows[metadata["agent_id"]].append(new_conflict)
    case.write_results()

    first = integrate(case)
    case.conflict_rows[metadata["agent_id"]].reverse()
    case.write_results()
    second = integrate(case)

    assert first == second
    assert len(first.conflicts) == 1
    assert first.conflicts[0]["conflict_id"].startswith("X")
    assert len(first.conflicts[0]["conflict_id"]) == 13
    assert first.conflicts[0]["record_key"] == "C0001"
    assert first.conflicts[0]["resolution"] == ""


def _new_conflict_result(
    case: IntegrationCase,
    local_id: str,
    field: str,
    value_a: str,
    value_b: str,
) -> dict[str, str]:
    metadata = result_row(case, "C0001")
    return {
        "candidate_id": "C0001",
        "input_sha256": metadata["input_sha256"],
        "agent_id": metadata["agent_id"],
        "input_conflict_id": f"NEW-{local_id}",
        "field": field,
        "value_a": value_a,
        "value_b": value_b,
        "resolution": "NR",
        "resolution_evidence": "NR",
    }


@pytest.mark.parametrize(
    ("field", "value_a", "value_b"),
    [
        ("title", "Track: Study", "track study"),
        ("doi", "doi:10.1000/ABC", "https://doi.org/10.1000/abc"),
    ],
)
def test_new_conflict_rejects_semantically_equivalent_values(
    tmp_path, field, value_a, value_b
):
    case = build_case(tmp_path)
    metadata = result_row(case, "C0001")
    metadata["metadata_status"] = "conflict"
    case.conflict_rows[metadata["agent_id"]].append(
        _new_conflict_result(case, "equivalent", field, value_a, value_b)
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="equivalent|disagree"):
        integrate(case)


def test_new_conflict_rejects_semantic_duplicate_of_frozen_conflict(tmp_path):
    frozen = conflict_row(
        "XFROZEN",
        "C0001",
        "venue",
        value_a="Venue A",
        value_b="Venue B",
    )
    case = build_case(tmp_path, conflicts=[frozen])
    metadata = result_row(case, "C0001")
    metadata["metadata_status"] = "conflict"
    case.conflict_rows[metadata["agent_id"]].append(
        _new_conflict_result(
            case,
            "duplicate-frozen",
            "venue",
            "Venue B",
            "Venue A",
        )
    )
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="duplicate"):
        integrate(case)


def test_two_new_conflicts_are_deterministic_when_rows_are_reversed(tmp_path):
    case = build_case(tmp_path)
    metadata = result_row(case, "C0001")
    metadata["metadata_status"] = "conflict"
    batch_rows = case.conflict_rows[metadata["agent_id"]]
    batch_rows.extend(
        [
            _new_conflict_result(
                case,
                "venue",
                "venue",
                "Venue A",
                "Venue B",
            ),
            _new_conflict_result(
                case,
                "title",
                "title",
                "Title A",
                "Title B",
            ),
        ]
    )
    case.write_results()

    forward = integrate(case)
    batch_rows.reverse()
    case.write_results()
    reversed_result = integrate(case)

    assert reversed_result == forward
    assert len(forward.conflicts) == 2


def _set_fully_nonlatin_metadata(case: IntegrationCase) -> None:
    row = case.all_metadata_rows[0]
    publisher = "publisher::https://publisher.example/nonlatin"
    row.update(
        title="赛道生成方法",
        authors="轨道研究組",
        key_author="研究組",
        author_kinds="corporate",
        title_evidence=publisher,
        authors_evidence=publisher,
        year_evidence=publisher,
        venue_evidence=publisher,
        source_type_evidence=publisher,
    )
    case.write_results()


def test_existing_key_bypasses_nonlatin_base_generation(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row(
                "C0001",
                title="赛道生成方法",
                cite_key="Protected2025Key",
            )
        ],
    )
    _set_fully_nonlatin_metadata(case)

    result = integrate(case)

    assert result.candidates[0]["cite_key"] == "Protected2025Key"


def test_unkeyed_nonlatin_record_uses_stable_candidate_fallback(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[candidate_row("C0001", title="赛道生成方法")],
    )
    _set_fully_nonlatin_metadata(case)

    result = integrate(case)

    assert result.candidates[0]["cite_key"] == "Candidate00012025"


def test_collision_suffixes_all_unkeyed_members_by_candidate_id(tmp_path):
    candidates = [
        candidate_row("C0002", title="The Same Track Generator"),
        candidate_row("C0001", title="The Same Track Generator"),
    ]
    case = build_case(tmp_path, candidates=candidates)
    for row in case.all_metadata_rows:
        row.update(key_author="Smith", year="2024")
    case.write_results()

    result = integrate(case)

    assert {
        row["candidate_id"]: row["cite_key"] for row in result.candidates
    } == {
        "C0001": "Smith2024SameTracka",
        "C0002": "Smith2024SameTrackb",
    }


def test_existing_key_is_preserved_and_later_collision_is_suffixed(tmp_path):
    candidates = [
        candidate_row(
            "C0001",
            title="The Same Track Generator",
            cite_key="Smith2024SameTrack",
        ),
        candidate_row("C0002", title="The Same Track Generator"),
    ]
    case = build_case(tmp_path, candidates=candidates)
    for row in case.all_metadata_rows:
        row.update(key_author="Smith", year="2024")
    case.write_results()

    result = integrate(case)

    assert {
        row["candidate_id"]: row["cite_key"] for row in result.candidates
    } == {
        "C0001": "Smith2024SameTrack",
        "C0002": "Smith2024SameTracka",
    }


def test_existing_keys_must_be_case_insensitively_unique(tmp_path):
    candidates = [
        candidate_row("C0001", cite_key="ProtectedKey"),
        candidate_row("C0002", cite_key="protectedkey"),
    ]
    case = build_case(tmp_path, candidates=candidates)

    with pytest.raises(MetadataIntegrationError, match="case-insensitive|duplicate"):
        integrate(case)


def test_article_inproceedings_and_misc_render_fixed_fields(tmp_path):
    candidates = [
        candidate_row("C0001", title="Article Title"),
        candidate_row("C0002", title="Proceedings Title"),
        candidate_row("C0003", title="Tool Title"),
    ]
    case = build_case(tmp_path, candidates=candidates)
    article = result_row(case, "C0001")
    proceedings = result_row(case, "C0002")
    misc = result_row(case, "C0003")
    proceedings.update(
        bib_entry_type="inproceedings",
        bib_venue_field="booktitle",
        venue="TrackGen 2025",
        venue_evidence="proceedings::https://conf.example/trackgen",
    )
    official = "official_release::https://example.org/tool/releases/v1"
    misc.update(
        year="NR",
        venue="Version 1",
        doi="NR",
        url="https://example.org/tool/releases/v1",
        source_type="software",
        title_evidence=official,
        authors_evidence=official,
        year_evidence="NR",
        venue_evidence=official,
        doi_evidence="NR",
        url_evidence=official,
        source_type_evidence=official,
        bib_entry_type="misc",
        bib_venue_field="howpublished",
        bib_url="https://example.org/tool/releases/v1",
        notes="The software release has no publication year or DOI.",
    )
    case.write_results()

    result = integrate(case)

    assert result.bibtex.index("@article") < result.bibtex.index("@inproceedings")
    assert "  journal = {Journal of Generated Tracks}" in result.bibtex
    assert "  booktitle = {TrackGen 2025}" in result.bibtex
    assert "  howpublished = {Version 1}" in result.bibtex


def test_bibtex_escapes_special_characters_and_braces_corporate_authors(
    tmp_path,
):
    candidate = candidate_row("C0001", title="R&D_Tracks: 50% #1 {Edition}")
    case = build_case(tmp_path, candidates=[candidate])
    row = case.all_metadata_rows[0]
    row.update(
        authors="ACME & Sons;Jane_Doe",
        author_kinds="corporate;personal",
        key_author="ACME",
        venue="Tracks & Tests",
    )
    case.write_results()

    result = integrate(case)

    assert "author = {{ACME \\& Sons} and Jane\\_Doe}" in result.bibtex
    assert "title = {R\\&D\\_Tracks: 50\\% \\#1 \\{Edition\\}}" in result.bibtex
    assert "journal = {Tracks \\& Tests}" in result.bibtex


def test_results_are_deterministic_under_file_and_row_reordering(tmp_path):
    candidates = [candidate_row(f"C{number:04d}") for number in range(1, 9)]
    conflicts = [
        conflict_row(
            f"X{number:04d}",
            f"C{number:04d}",
            "title",
            value_a=f"Old {number}",
            value_b=f"Track Study C{number:04d}",
        )
        for number in range(1, 9)
    ]
    case = build_case(tmp_path, candidates=candidates, conflicts=conflicts)
    first = integrate(case)
    case.result_pairs.reverse()
    for rows in case.metadata_rows.values():
        rows.reverse()
    for rows in case.conflict_rows.values():
        rows.reverse()
    # Rewriting uses path names, while the API receives the reversed pair list.
    for metadata_path, conflict_path in case.result_pairs:
        batch_id = metadata_path.name.removesuffix(".csv")
        write_rows(
            metadata_path,
            METADATA_RESULT_HEADER,
            case.metadata_rows[batch_id],
        )
        write_rows(
            conflict_path,
            CONFLICT_RESULT_HEADER,
            case.conflict_rows[batch_id],
        )

    second = integrate(case)

    assert second == first


def output_paths(tmp_path: Path) -> dict[str, Path]:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return {
        "candidates_path": outputs / "candidates.csv",
        "conflicts_path": outputs / "conflicts.csv",
        "bibliography_path": outputs / "bibliography.csv",
        "bibtex_path": outputs / "references.bib",
    }


def test_writer_is_byte_idempotent_and_uses_exact_output_schemas(tmp_path):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs = output_paths(tmp_path)

    write_integration_outputs(result, **outputs)
    first = {name: path.read_bytes() for name, path in outputs.items()}
    write_integration_outputs(result, **outputs)

    assert {name: path.read_bytes() for name, path in outputs.items()} == first
    with outputs["candidates_path"].open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == HEADERS["candidates.csv"]
    with outputs["conflicts_path"].open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == HEADERS["conflicts.csv"]
    with outputs["bibliography_path"].open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == BIBLIOGRAPHY_HEADER
    assert not list((tmp_path / "outputs").glob(".*.tmp"))


def test_writer_rejects_input_aliases_without_optional_protection(tmp_path):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs = output_paths(tmp_path)
    outputs["candidates_path"] = case.candidates_path
    original = case.candidates_path.read_bytes()

    with pytest.raises(MetadataIntegrationError, match="alias|differ"):
        write_integration_outputs(result, **outputs)

    assert case.candidates_path.read_bytes() == original


def test_writer_rejects_empty_carried_input_identity(tmp_path):
    case = build_case(tmp_path)
    result = integrate(case)
    unsafe_result = metadata_integration.MetadataIntegrationResult(
        candidates=result.candidates,
        conflicts=result.conflicts,
        bibliography=result.bibliography,
        bibtex=result.bibtex,
        input_paths=(),
    )
    outputs = output_paths(tmp_path)
    supplemental = tmp_path / "unrelated-input.csv"
    supplemental.write_text("unrelated\n", encoding="utf-8")

    with pytest.raises(MetadataIntegrationError, match="input paths|required"):
        write_integration_outputs(
            unsafe_result, input_paths=[supplemental], **outputs
        )

    assert not any(path.exists() for path in outputs.values())


def test_writer_rejects_symlinked_input_and_output_aliases(tmp_path):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs = output_paths(tmp_path)
    input_alias = outputs["candidates_path"]
    input_alias.symlink_to(case.manifest_path)

    with pytest.raises(MetadataIntegrationError, match="alias|differ"):
        write_integration_outputs(result, **outputs)

    input_alias.unlink()
    outputs["candidates_path"].write_text("existing\n", encoding="utf-8")
    outputs["conflicts_path"].symlink_to(outputs["candidates_path"])
    with pytest.raises(MetadataIntegrationError, match="distinct|duplicate"):
        write_integration_outputs(result, **outputs)


def _sentinel_outputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, bytes]]:
    outputs = output_paths(tmp_path)
    sentinels = {}
    for name, path in outputs.items():
        sentinel = f"sentinel::{name}\n".encode()
        path.write_bytes(sentinel)
        sentinels[name] = sentinel
    return outputs, sentinels


@pytest.mark.parametrize("failure_point", ["stage", "fsync", "chmod"])
def test_writer_staging_failures_preserve_outputs_and_clean_temps(
    tmp_path, monkeypatch, failure_point
):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs, sentinels = _sentinel_outputs(tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "stage":
        real_stage = metadata_integration._stage_bytes
        calls = 0

        def fail_second_stage(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                fail()
            return real_stage(*args, **kwargs)

        monkeypatch.setattr(metadata_integration, "_stage_bytes", fail_second_stage)
    elif failure_point == "fsync":
        monkeypatch.setattr(metadata_integration.os, "fsync", fail)
    else:
        monkeypatch.setattr(metadata_integration.os, "chmod", fail)

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        write_integration_outputs(result, **outputs)

    assert {name: path.read_bytes() for name, path in outputs.items()} == sentinels
    assert not list((tmp_path / "outputs").glob(".*.tmp"))


def test_writer_rolls_back_when_replace_raises_after_committing(
    tmp_path, monkeypatch
):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs, sentinels = _sentinel_outputs(tmp_path)

    real_replace = Path.replace
    replacement_count = 0

    def fail_second_output(source: Path, target: Path):
        nonlocal replacement_count
        if source.name.endswith(".tmp") and not source.name.endswith(".bak.tmp"):
            replacement_count += 1
            if replacement_count == 2:
                real_replace(source, target)
                raise OSError("injected replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_output)

    with pytest.raises(OSError, match="injected replace failure"):
        write_integration_outputs(result, **outputs)

    assert {name: path.read_bytes() for name, path in outputs.items()} == sentinels
    assert not list((tmp_path / "outputs").glob(".*.tmp"))


def test_writer_retains_all_backups_and_reports_incomplete_rollback(
    tmp_path, monkeypatch
):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs, _sentinels = _sentinel_outputs(tmp_path)
    real_replace = Path.replace
    replacement_count = 0

    def fail_output_and_restore(source: Path, target: Path):
        nonlocal replacement_count
        if source.name.endswith(".restore.tmp"):
            raise OSError(f"injected restore failure for {target.name}")
        if source.name.endswith(".tmp") and not source.name.endswith(".bak.tmp"):
            replacement_count += 1
            if replacement_count == 2:
                real_replace(source, target)
                raise OSError("injected output replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_output_and_restore)

    with pytest.raises(MetadataIntegrationError) as raised:
        write_integration_outputs(result, **outputs)

    backups = sorted((tmp_path / "outputs").glob(".*.bak.tmp"))
    message = str(raised.value)
    assert len(backups) == 4
    assert "rollback incomplete" in message
    assert "injected restore failure" in message
    assert "injected restore failure for candidates.csv" in message
    assert "injected restore failure for conflicts.csv" in message
    assert all(str(path) in message for path in backups)
    assert not list((tmp_path / "outputs").glob(".*.restore.tmp"))
    assert not [
        path
        for path in (tmp_path / "outputs").glob(".*.tmp")
        if not path.name.endswith(".bak.tmp")
    ]


def _cli_inputs(case: IntegrationCase) -> list[str]:
    arguments = [
        "--candidates",
        str(case.candidates_path),
        "--conflicts",
        str(case.conflicts_path),
        "--manifest",
        str(case.manifest_path),
    ]
    for metadata_path, conflict_path in case.result_pairs:
        arguments.extend(["--metadata-result", str(metadata_path)])
        arguments.extend(["--conflict-result", str(conflict_path)])
    return arguments


def test_cli_rejects_missing_required_outputs(tmp_path):
    case = build_case(tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(_cli_inputs(case))

    assert raised.value.code == 2


def test_cli_output_path_failure_writes_nothing(tmp_path):
    case = build_case(tmp_path)
    outputs = output_paths(tmp_path)
    missing_output = tmp_path / "missing" / "candidates.csv"
    arguments = [
        *_cli_inputs(case),
        "--output-candidates",
        str(missing_output),
        "--output-conflicts",
        str(outputs["conflicts_path"]),
        "--output-bibliography",
        str(outputs["bibliography_path"]),
        "--output-bibtex",
        str(outputs["bibtex_path"]),
    ]

    with pytest.raises(MetadataIntegrationError, match="directory is missing"):
        main(arguments)

    assert not missing_output.exists()
    assert not any(path.exists() for path in outputs.values())


def test_cli_requires_explicit_outputs_and_writes_all_four(tmp_path):
    case = build_case(tmp_path)
    outputs = output_paths(tmp_path)
    arguments = [
        "--candidates",
        str(case.candidates_path),
        "--conflicts",
        str(case.conflicts_path),
        "--manifest",
        str(case.manifest_path),
    ]
    for metadata_path, conflict_path in case.result_pairs:
        arguments.extend(["--metadata-result", str(metadata_path)])
        arguments.extend(["--conflict-result", str(conflict_path)])
    arguments.extend(
        [
            "--output-candidates",
            str(outputs["candidates_path"]),
            "--output-conflicts",
            str(outputs["conflicts_path"]),
            "--output-bibliography",
            str(outputs["bibliography_path"]),
            "--output-bibtex",
            str(outputs["bibtex_path"]),
        ]
    )

    assert main(arguments) == 0
    assert all(path.is_file() for path in outputs.values())
