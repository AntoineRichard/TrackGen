from __future__ import annotations

import csv
import json
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
from paper.scripts.validate_corpus import (
    DEFAULT_TAXONOMY,
    HEADERS,
    validate_directory,
)


CITATION_KEYS_HEADER = ("candidate_id", "cite_key")


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


def write_citation_keys(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    write_rows(path, CITATION_KEYS_HEADER, rows)


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
    citation_keys_path: Path
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
    citation_keys_path = inputs / "citation_keys.csv"
    write_rows(candidates_path, HEADERS["candidates.csv"], candidates)
    write_rows(conflicts_path, HEADERS["conflicts.csv"], conflicts)
    manifest = build_manifest(candidates_path, conflicts_path)
    write_rows(manifest_path, MANIFEST_HEADER, manifest)
    write_citation_keys(
        citation_keys_path,
        [
            {"candidate_id": row["candidate_id"], "cite_key": row["cite_key"]}
            for row in candidates
            if row["cite_key"]
        ],
    )

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
        citation_keys_path=citation_keys_path,
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
        citation_keys_path=case.citation_keys_path,
        extend_citation_keys=True,
    )



def validate_integration_round_trip(tmp_path: Path, result) -> Path:
    root = tmp_path / "roundtrip"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "taxonomy.json").write_text(
        json.dumps(DEFAULT_TAXONOMY, indent=2) + "\n",
        encoding="utf-8",
    )
    generated_tables = {
        "candidates.csv",
        "conflicts.csv",
        "bibliography.csv",
        "citation_keys.csv",
    }
    for filename, header in HEADERS.items():
        if filename not in generated_tables:
            write_rows(data / filename, header, [])

    write_integration_outputs(
        result,
        candidates_path=data / "candidates.csv",
        conflicts_path=data / "conflicts.csv",
        bibliography_path=data / "bibliography.csv",
        bibtex_path=root / "references.bib",
        citation_keys_path=data / "citation_keys.csv",
    )
    validate_directory(data)
    return data


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


def test_strict_integration_rejects_missing_active_ledger_assignment(tmp_path):
    case = build_case(tmp_path)
    citation_keys_path = tmp_path / "citation_keys.csv"
    write_citation_keys(citation_keys_path, [])

    with pytest.raises(
        MetadataIntegrationError,
        match=r"candidate_id='C0001'.*citation key|citation key.*C0001",
    ):
        integrate_metadata(
            case.candidates_path,
            case.conflicts_path,
            case.manifest_path,
            case.result_pairs,
            citation_keys_path=citation_keys_path,
        )


def test_extension_appends_only_missing_active_keys_in_numeric_id_order(tmp_path):
    candidates = [
        candidate_row(
            "C0001",
            cite_key="DormantKey",
            screening_status="excluded",
            exclusion_reason="Out of scope",
        ),
        candidate_row("C0010", title="Zeta Course"),
        candidate_row("C0002", title="Alpha Course"),
    ]
    case = build_case(tmp_path, candidates=candidates)
    result = integrate(case)

    assert result.new_citation_keys == [
        {"candidate_id": "C0002", "cite_key": "Example2025AlphaCourse"},
        {"candidate_id": "C0010", "cite_key": "Example2025ZetaCourse"},
    ]
    assert result.citation_keys == [
        {"candidate_id": "C0001", "cite_key": "DormantKey"},
        *result.new_citation_keys,
    ]

    candidates_by_id = {
        row["candidate_id"]: row for row in result.candidates
    }
    assert candidates_by_id["C0001"]["cite_key"] == ""
    assert candidates_by_id["C0002"]["cite_key"] == "Example2025AlphaCourse"


def test_dormant_ledger_key_reserves_casefold_namespace(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row(
                "C0001",
                cite_key="example2025sametrack",
                screening_status="excluded",
                exclusion_reason="Out of scope",
            ),
            candidate_row("C0002", title="Same Track"),
        ],
    )

    result = integrate(case)
    candidates_by_id = {
        row["candidate_id"]: row for row in result.candidates
    }

    assert candidates_by_id["C0001"]["cite_key"] == ""
    assert candidates_by_id["C0002"]["cite_key"] == "Example2025SameTracka"
    assert result.new_citation_keys == [
        {"candidate_id": "C0002", "cite_key": "Example2025SameTracka"}
    ]


def test_collision_suffix_order_uses_numeric_candidate_ids(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row("C10000", title="Same Track"),
            candidate_row("C9999", title="Same Track"),
        ],
    )

    result = integrate(case)

    assert result.new_citation_keys == [
        {"candidate_id": "C9999", "cite_key": "Example2025SameTracka"},
        {"candidate_id": "C10000", "cite_key": "Example2025SameTrackb"},
    ]


def test_reinclusion_restores_exact_dormant_ledger_key(tmp_path):
    dormant_root = tmp_path / "dormant"
    dormant_root.mkdir()
    dormant_case = build_case(
        dormant_root,
        candidates=[
            candidate_row(
                "C0001",
                cite_key="StableKey",
                screening_status="excluded",
                exclusion_reason="Out of scope",
            )
        ],
    )
    dormant_result = integrate(dormant_case)

    active_root = tmp_path / "active"
    active_root.mkdir()
    active_case = build_case(
        active_root,
        candidates=[candidate_row("C0001")],
    )
    write_citation_keys(
        active_case.citation_keys_path,
        [{"candidate_id": "C0001", "cite_key": "StableKey"}],
    )
    active_result = integrate_metadata(
        active_case.candidates_path,
        active_case.conflicts_path,
        active_case.manifest_path,
        active_case.result_pairs,
        citation_keys_path=active_case.citation_keys_path,
    )

    assert dormant_result.candidates[0]["cite_key"] == ""
    assert active_result.candidates[0]["cite_key"] == "StableKey"
    assert active_result.new_citation_keys == []


def test_citation_key_ledger_requires_exact_header(tmp_path):
    case = build_case(tmp_path)
    case.citation_keys_path.write_text(
        "candidate_id,key\nC0001,Example2025TrackStudy\n",
        encoding="utf-8",
    )

    with pytest.raises(MetadataIntegrationError, match=r"headers.*cite_key"):
        integrate(case)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"candidate_id": "", "cite_key": "ExampleKey"}],
            "candidate_id.*required|candidate_id.*nonempty",
        ),
        (
            [{"candidate_id": "C0001", "cite_key": ""}],
            "cite_key.*required|cite_key.*nonempty",
        ),
        (
            [{"candidate_id": "C01", "cite_key": "ExampleKey"}],
            "candidate_id.*C01.*four digits|candidate_id.*format",
        ),
        (
            [{"candidate_id": "C9999", "cite_key": "ExampleKey"}],
            "C9999.*does not exist|unknown candidate_id.*C9999",
        ),
        (
            [
                {"candidate_id": "C0001", "cite_key": "ExampleKey"},
                {"candidate_id": "C0001", "cite_key": "OtherKey"},
            ],
            "duplicate candidate_id.*C0001",
        ),
        (
            [
                {"candidate_id": "C0001", "cite_key": "ExampleKey"},
                {"candidate_id": "C0002", "cite_key": "examplekey"},
            ],
            "case-insensitive|casefold|duplicate cite_key",
        ),
        (
            [{"candidate_id": "C0001", "cite_key": "unsafe key"}],
            "cite_key.*BibTeX-safe",
        ),
    ],
    ids=[
        "blank-candidate-id",
        "blank-cite-key",
        "malformed-candidate-id",
        "orphan-candidate-id",
        "duplicate-candidate-id",
        "casefold-duplicate-key",
        "unsafe-key",
    ],
)
def test_citation_key_ledger_rejects_malformed_rows(
    tmp_path, rows, message
):
    case = build_case(
        tmp_path,
        candidates=[candidate_row("C0001"), candidate_row("C0002")],
    )
    write_citation_keys(case.citation_keys_path, rows)

    with pytest.raises(MetadataIntegrationError, match=message):
        integrate(case)


def test_snapshot_cite_key_is_only_a_matching_ledger_assertion(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[candidate_row("C0001", cite_key="SnapshotKey")],
    )
    write_citation_keys(
        case.citation_keys_path,
        [{"candidate_id": "C0001", "cite_key": "LedgerKey"}],
    )

    with pytest.raises(
        MetadataIntegrationError,
        match=r"snapshot cite_key 'SnapshotKey'.*ledger",
    ):
        integrate(case)


def test_corrected_corpus_strict_replay_is_deterministic_and_preserves_keys():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "paper" / "data"
    runs_dir = data_dir / "metadata_runs"
    result_pairs = [
        (
            runs_dir / f"metadata-{number:02d}.csv",
            runs_dir / f"metadata-{number:02d}-conflicts.csv",
        )
        for number in range(1, 7)
    ]
    arguments = {
        "candidates_path": data_dir / "metadata_inputs" / "v1" / "candidates.csv",
        "conflicts_path": data_dir / "metadata_inputs" / "v1" / "conflicts.csv",
        "manifest_path": data_dir / "metadata_manifest.csv",
        "result_pairs": result_pairs,
        "citation_keys_path": data_dir / "citation_keys.csv",
    }

    first = integrate_metadata(**arguments)
    second = integrate_metadata(**arguments)
    with (data_dir / "citation_keys.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        committed_ledger = list(csv.DictReader(handle))

    assert second == first
    assert first.new_citation_keys == []
    assert first.citation_keys == committed_ledger
    assert len(first.bibliography) == 198
    assert {
        row["cite_key"] for row in first.bibliography
    } == {
        row["cite_key"] for row in committed_ledger
    }

    candidates_by_id = {
        row["candidate_id"]: row for row in first.candidates
    }
    assert {
        candidate_id: candidates_by_id[candidate_id]["cite_key"]
        for candidate_id in ("C0049", "C0082", "C0165", "C0188", "C0189")
    } == {
        "C0049": "DralligNodateProceduralRace",
        "C0082": "MIT2017MITRACECAR",
        "C0165": "Yu2025MasteringDiverse",
        "C0188": "IsaacLabNodateIsaacLab",
        "C0189": "TUMFTM2020LapTime",
    }
    assert {
        row["candidate_id"] for row in first.candidates if row["cite_key"]
    } == {
        row["candidate_id"] for row in committed_ledger
    }


def test_default_integration_outputs_validate_round_trip(tmp_path):
    result = integrate(build_case(tmp_path))

    data = validate_integration_round_trip(tmp_path, result)

    assert (data.parent / "references.bib").read_bytes() == (
        result.bibtex.encode("utf-8")
    )


def test_complex_rendered_values_validate_round_trip(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        title="Étude {Nested}\nTrack \\ Path, 100%",
        authors="ACME {R&D}, GmbH;Zoë D'Árc",
        author_kinds="corporate;personal",
        key_author="ACME",
        venue="Journál, Series",
    )
    case.write_results()

    result = integrate(case)
    validate_integration_round_trip(tmp_path, result)

    assert "author = {{ACME \\{R\\&D\\}, GmbH} and Zoë D'Árc}" in result.bibtex
    assert "title = {Étude \\{Nested\\}\nTrack \\textbackslash{} Path, 100\\%}" in (
        result.bibtex
    )

def test_multiline_lf_bibliography_values_validate_round_trip(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        title="Track\nStudy",
        authors="Alice\nExample;ACME\nLabs",
        author_kinds="personal;corporate",
        key_author="Example",
        venue="Journal\nSeries",
    )
    case.write_results()

    result = integrate(case)
    data = validate_integration_round_trip(tmp_path, result)

    references = (data.parent / "references.bib").read_bytes()
    assert b"\r" not in references
    assert references == result.bibtex.encode("utf-8")


@pytest.mark.parametrize("field", ["title", "authors", "venue", "key_author"])
def test_integrator_rejects_embedded_carriage_return_before_output(
    tmp_path, field
):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    value = row[field]
    row[field] = f"{value[:1]}\r{value[1:]}"
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match=rf"bibliography field {field!r} contains carriage return",
    ):
        integrate(case)


@pytest.mark.parametrize("field", BIBLIOGRAPHY_HEADER)
def test_render_bibtex_rejects_embedded_carriage_return(tmp_path, field):
    result = integrate(build_case(tmp_path))
    row = dict(result.bibliography[0])
    value = row[field] or "value"
    row[field] = f"{value}\rcontinued"

    with pytest.raises(
        MetadataIntegrationError,
        match=rf"bibliography field {field!r} contains carriage return",
    ):
        metadata_integration.render_bibtex([row])



def test_misc_paper_integration_outputs_validate_round_trip(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        bib_entry_type="misc",
        bib_venue_field="howpublished",
    )
    case.write_results()

    result = integrate(case)

    validate_integration_round_trip(tmp_path, result)



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
            citation_keys_path=case.citation_keys_path,
        )
    with pytest.raises(MetadataIntegrationError, match="distinct"):
        integrate_metadata(
            case.candidates_path,
            case.conflicts_path,
            case.manifest_path,
            [*case.result_pairs[:-1], case.result_pairs[0]],
            citation_keys_path=case.citation_keys_path,
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


def test_distinct_doi_resolver_bibliography_url_is_rejected(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    alternate = "https://doi.org/10.1000/alternate"
    row["bib_url"] = alternate
    row["url_evidence"] += f";doi::{alternate}"
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match="bib_url.*DOI|DOI.*bib_url|match.*doi",
    ):
        integrate(case)


def test_doi_resolver_bibliography_url_requires_candidate_doi(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    resolver = "https://doi.org/10.1000/alternate"
    row.update(
        doi="NR",
        doi_evidence="NR",
        url=resolver,
        url_evidence=f"doi::{resolver}",
        bib_url=resolver,
    )
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match="bib_url.*DOI|DOI.*bib_url|requires.*doi",
    ):
        integrate(case)


@pytest.mark.parametrize("field", ["url", "bib_url"])
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
def test_produced_urls_reject_unicode_whitespace(tmp_path, field, whitespace):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    malformed = f"https://publisher.example/path{whitespace}segment"
    row[field] = malformed
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="URL.*whitespace"):
        integrate(case)


def test_non_whitespace_unicode_url_validates_round_trip(tmp_path):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    url = "https://publisher.example/café"
    row.update(
        url=url,
        bib_url=url,
        url_evidence=f"publisher::{url}",
    )
    case.write_results()

    result = integrate(case)
    validate_integration_round_trip(tmp_path, result)

    assert result.bibliography[0]["url"] == url


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
    validate_integration_round_trip(tmp_path, result)

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


@pytest.mark.parametrize("field", ["year", "venue"])
def test_misc_paper_like_source_requires_year_and_venue(tmp_path, field):
    case = build_case(tmp_path)
    row = case.all_metadata_rows[0]
    row.update(
        bib_entry_type="misc",
        bib_venue_field="howpublished",
    )
    row[field] = "NR"
    row[f"{field}_evidence"] = "NR"
    if field == "venue":
        row["bib_venue_field"] = "NR"
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match=rf"paper-like {field}",
    ):
        integrate(case)


@pytest.mark.parametrize(
    "authors",
    ["One Author ET, AL.", "One Author et-al", "One Author Et.Al."],
)
def test_incomplete_author_markers_are_punctuation_robust(tmp_path, authors):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["authors"] = authors
    case.write_results()

    with pytest.raises(
        MetadataIntegrationError,
        match="authors.*complete|not et al",
    ):
        integrate(case)


@pytest.mark.parametrize("key_author", ["", "NR", "et al.", "ET, AL."])
def test_verified_key_author_must_be_complete(tmp_path, key_author):
    case = build_case(tmp_path)
    case.all_metadata_rows[0]["key_author"] = key_author
    case.write_results()

    with pytest.raises(MetadataIntegrationError, match="key_author"):
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


def test_existing_key_on_ineligible_candidate_is_dormant(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row(
                "C0001",
                cite_key="StaleKey",
                screening_status="excluded",
                exclusion_reason="Out of scope",
            )
        ],
    )

    result = integrate(case)

    assert result.candidates[0]["cite_key"] == ""
    assert result.bibliography == []
    assert result.citation_keys == [
        {"candidate_id": "C0001", "cite_key": "StaleKey"}
    ]


def test_casefold_collision_from_ineligible_key_is_rejected(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row("C0001", cite_key="ProtectedKey"),
            candidate_row(
                "C0002",
                cite_key="protectedkey",
                screening_status="excluded",
                exclusion_reason="Out of scope",
            ),
        ],
    )

    with pytest.raises(
        MetadataIntegrationError,
        match="existing cite_key.*not verified and non-excluded|case-insensitive",
    ):
        integrate(case)



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


def test_writer_preserves_existing_ledger_bytes_as_extension_prefix(tmp_path):
    case = build_case(
        tmp_path,
        candidates=[
            candidate_row(
                "C0001",
                cite_key="DormantKey",
                screening_status="excluded",
                exclusion_reason="Out of scope",
            ),
            candidate_row("C0002", title="Alpha Course"),
        ],
    )
    original_bytes = (
        b'candidate_id,cite_key\r\nC0001,"DormantKey"\r\n'
    )
    case.citation_keys_path.write_bytes(original_bytes)
    result = integrate(case)
    outputs = output_paths(tmp_path)
    output_citation_keys = outputs["candidates_path"].with_name(
        "citation_keys.csv"
    )

    write_integration_outputs(
        result,
        citation_keys_path=output_citation_keys,
        **outputs,
    )

    assert output_citation_keys.read_bytes() == (
        original_bytes + b"C0002,Example2025AlphaCourse\n"
    )


def test_mutation_during_manifest_validation_is_rejected(
    tmp_path,
    monkeypatch,
):
    case = build_case(tmp_path)
    original_validate = metadata_integration.validate_manifest_inputs

    def validate_then_mutate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        payload = case.candidates_path.read_bytes()
        assert b"discovery-agent" in payload
        case.candidates_path.write_bytes(
            payload.replace(b"discovery-agent", b"mutated-agent", 1)
        )
        return result

    monkeypatch.setattr(
        metadata_integration,
        "validate_manifest_inputs",
        validate_then_mutate,
    )

    with pytest.raises(
        MetadataIntegrationError,
        match=r"input .* changed since integration",
    ):
        integrate(case)


def test_writer_rejects_ledger_mutation_after_integration(tmp_path):
    case = build_case(tmp_path)
    result = integrate(case)
    case.citation_keys_path.write_text(
        "candidate_id,cite_key\nC0001,TamperedKey\n",
        encoding="utf-8",
    )
    outputs = output_paths(tmp_path)
    output_citation_keys = outputs["candidates_path"].with_name(
        "citation_keys.csv"
    )

    with pytest.raises(
        MetadataIntegrationError,
        match=r"input.*changed since integration|input mutation",
    ):
        write_integration_outputs(
            result,
            citation_keys_path=output_citation_keys,
            **outputs,
        )

    assert not any(path.exists() for path in outputs.values())
    assert not output_citation_keys.exists()


@pytest.mark.parametrize("alias_kind", ["ledger-input", "other-output"])
def test_writer_rejects_ledger_output_path_aliases(tmp_path, alias_kind):
    case = build_case(tmp_path)
    result = integrate(case)
    outputs = output_paths(tmp_path)
    output_citation_keys = (
        case.citation_keys_path
        if alias_kind == "ledger-input"
        else outputs["candidates_path"]
    )

    with pytest.raises(MetadataIntegrationError, match="alias|distinct|differ"):
        write_integration_outputs(
            result,
            citation_keys_path=output_citation_keys,
            **outputs,
        )


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
        citation_keys=result.citation_keys,
        new_citation_keys=result.new_citation_keys,
        citation_keys_bytes=result.citation_keys_bytes,
        input_fingerprints=(),
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
        "--citation-keys",
        str(case.citation_keys_path),
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
    write_citation_keys(
        case.citation_keys_path,
        [{"candidate_id": "C0001", "cite_key": "Example2025TrackStudy"}],
    )
    outputs = output_paths(tmp_path)
    arguments = _cli_inputs(case)
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


def _required_cli_outputs(outputs: dict[str, Path]) -> list[str]:
    return [
        "--output-candidates",
        str(outputs["candidates_path"]),
        "--output-conflicts",
        str(outputs["conflicts_path"]),
        "--output-bibliography",
        str(outputs["bibliography_path"]),
        "--output-bibtex",
        str(outputs["bibtex_path"]),
    ]


@pytest.mark.parametrize(
    ("extra_arguments", "message"),
    [
        (
            ["--extend-citation-keys"],
            "--output-citation-keys is required",
        ),
        (
            ["--output-citation-keys", "unused.csv"],
            "--output-citation-keys is forbidden",
        ),
    ],
    ids=["extension-missing-output", "strict-output-forbidden"],
)
def test_cli_couples_extension_and_ledger_output(
    tmp_path,
    capsys,
    extra_arguments,
    message,
):
    case = build_case(tmp_path)
    outputs = output_paths(tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(
            [
                *_cli_inputs(case),
                *_required_cli_outputs(outputs),
                *extra_arguments,
            ]
        )

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_extension_writes_all_outputs_and_appended_ledger(tmp_path):
    case = build_case(tmp_path)
    outputs = output_paths(tmp_path)
    output_citation_keys = outputs["candidates_path"].with_name(
        "citation_keys.csv"
    )

    assert (
        main(
            [
                *_cli_inputs(case),
                *_required_cli_outputs(outputs),
                "--extend-citation-keys",
                "--output-citation-keys",
                str(output_citation_keys),
            ]
        )
        == 0
    )
    assert all(path.is_file() for path in outputs.values())
    assert output_citation_keys.read_text(encoding="utf-8") == (
        "candidate_id,cite_key\n"
        "C0001,Example2025TrackStudy\n"
    )


def test_cli_requires_citation_key_ledger(tmp_path, capsys):
    case = build_case(tmp_path)
    outputs = output_paths(tmp_path)
    arguments = _cli_inputs(case)
    ledger_index = arguments.index("--citation-keys")
    del arguments[ledger_index : ledger_index + 2]

    with pytest.raises(SystemExit) as raised:
        main([*arguments, *_required_cli_outputs(outputs)])

    assert raised.value.code == 2
    assert "--citation-keys" in capsys.readouterr().err
