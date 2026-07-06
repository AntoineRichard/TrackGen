from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from paper.scripts.prepare_screening_batches import (
    CANDIDATE_HEADER,
    EVIDENCE_PACKET_HEADER,
    parse_evidence_packet_manifest,
)


AUDIT_HEADER = (
    "candidate_id",
    "title",
    "source_url",
    "evidence_version",
    "access_status",
    "local_archive_path_or_NR",
    "limitation_note",
)
QUEUE_HEADER = (
    "candidate_id",
    "title",
    "source_url",
    "raw_access_status",
    "action",
    "priority",
    "limitation_note",
)


def write_csv(path: Path, header: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for number in range(1, 203):
        candidate_id = f"C{number:04d}"
        row = dict.fromkeys(CANDIDATE_HEADER, "NR")
        row.update(
            candidate_id=candidate_id,
            cite_key=f"Key{number}",
            title=f"Scenario generation study {number}",
            authors="Example Author",
            year="2026",
            venue="Example venue",
            doi="NR",
            url=f"https://example.test/{candidate_id}",
            source_type="article",
            discovery_stream="test",
            discovery_query="test",
            discovery_agent="test",
            screening_status="candidate",
            exclusion_reason="NR",
            metadata_status="verified",
            metadata_evidence="test fixture",
        )
        rows.append(row)
    return rows


def audit_rows(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "title": candidate["title"],
                "source_url": candidate["url"],
                "evidence_version": "fixture-v1",
                "access_status": "metadata_only",
                "local_archive_path_or_NR": "NR",
                "limitation_note": "Fixture has no archived primary artifact.",
            }
        )
    return rows


def v7_row(candidate_id: str, payload: bytes, *, filename: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "artifact_id": "primary-report",
        "artifact_role": "primary-report",
        "source_url": f"https://example.test/{candidate_id}",
        "evidence_version": "fixture-v1",
        "evidence_retrieved_on": "2026-07-06",
        "access_status": "full_text",
        "evidence_archive_url": "NR",
        "evidence_sha256": hashlib.sha256(payload).hexdigest(),
        "local_filename": filename,
        "redistribution_status": "local-restricted",
        "retrieval_notes": "NR",
    }


def build_inputs(tmp_path: Path) -> dict[str, Path]:
    candidates = candidate_rows()
    candidates_path = tmp_path / "candidates.csv"
    write_csv(candidates_path, CANDIDATE_HEADER, candidates)

    audits_dir = tmp_path / "audits"
    audits_dir.mkdir()
    audits = audit_rows(candidates)
    by_id = {row["candidate_id"]: row for row in audits}
    by_id["C0002"]["access_status"] = "full_text_public_archive_corrupt"
    by_id["C0003"]["access_status"] = "related_local_full_text"
    by_id["C0004"]["access_status"] = "full_text_public"
    by_id["C0005"]["access_status"] = "official_evidence"
    write_csv(audits_dir / "audits.csv", AUDIT_HEADER, audits)

    archive = tmp_path / "source_archive"
    (archive / "C0001").mkdir(parents=True)
    (archive / "C0001" / "trusted.txt").write_bytes(b"trusted v7 evidence")
    (archive / "C0007").mkdir(parents=True)
    (archive / "C0007" / "mismatched.txt").write_bytes(b"actual archive bytes")

    manifest_path = tmp_path / "v7.csv"
    trusted = v7_row("C0001", b"trusted v7 evidence", filename="C0001/trusted.txt")
    mismatched = v7_row("C0007", b"declared but incorrect bytes", filename="C0007/mismatched.txt")
    write_csv(manifest_path, EVIDENCE_PACKET_HEADER, [trusted, mismatched])

    return {
        "candidates": candidates_path,
        "audits": audits_dir,
        "archive": archive,
        "v7": manifest_path,
        "manifest_output": tmp_path / "manifest.csv",
        "queue_output": tmp_path / "queue.csv",
        "report_output": tmp_path / "report.md",
    }


def normalize(paths: dict[str, Path]) -> None:
    from paper.scripts.normalize_main_evidence_audits import main

    assert (
        main(
            [
                "--candidates",
                str(paths["candidates"]),
                "--audits-dir",
                str(paths["audits"]),
                "--v7-manifest",
                str(paths["v7"]),
                "--source-archive",
                str(paths["archive"]),
                "--manifest-output",
                str(paths["manifest_output"]),
                "--queue-output",
                str(paths["queue_output"]),
                "--report-output",
                str(paths["report_output"]),
            ]
        )
        == 0
    )


def test_normalization_reuses_only_verified_v7_bytes_and_queues_every_provisional_candidate(
    tmp_path: Path,
) -> None:
    paths = build_inputs(tmp_path)

    normalize(paths)

    manifest_bytes = paths["manifest_output"].read_bytes()
    assert b"\r" not in manifest_bytes
    with paths["manifest_output"].open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert tuple(manifest[0]) == EVIDENCE_PACKET_HEADER
    assert [row["candidate_id"] for row in manifest] == [
        f"C{number:04d}" for number in range(1, 203)
    ]
    assert len(manifest) == 202
    by_id = {row["candidate_id"]: row for row in manifest}
    assert by_id["C0001"] == v7_row(
        "C0001", b"trusted v7 evidence", filename="C0001/trusted.txt"
    )
    assert by_id["C0002"]["evidence_sha256"] == "NR"
    assert by_id["C0002"]["local_filename"] == "NR"
    assert by_id["C0002"]["redistribution_status"] == "metadata-only"
    assert by_id["C0002"]["access_status"] == "abstract_only"
    assert by_id["C0007"]["evidence_sha256"] == "NR"
    parse_evidence_packet_manifest(
        manifest_bytes,
        allowed_candidate_ids={f"C{number:04d}" for number in range(1, 203)},
        source_archive=paths["archive"],
    )

    with paths["queue_output"].open(encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))
    assert tuple(queue[0]) == QUEUE_HEADER
    assert len(queue) == 201
    assert {row["candidate_id"] for row in queue} == set(by_id) - {"C0001"}
    actions = {row["candidate_id"]: row["action"] for row in queue}
    assert actions["C0002"] == "replace-corrupt-local"
    assert actions["C0003"] == "replace-mismatched-local"
    assert actions["C0004"] == "archive-public-full-text"
    assert actions["C0005"] == "archive-official-source"
    assert actions["C0006"] == "user-fetch-or-document-limitation"
    assert "Trusted byte-backed rows: 1." in paths["report_output"].read_text(encoding="utf-8")
    assert "Provisional metadata-only rows: 201." in paths["report_output"].read_text(encoding="utf-8")
    report = paths["report_output"].read_text(encoding="utf-8")
    assert "supplied source archive root" in report
    assert "source_archive/v7" not in report


def test_normalization_rejects_nonexact_audit_coverage(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    audit_path = paths["audits"] / "audits.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[:-1]
    write_csv(audit_path, AUDIT_HEADER, rows)

    from paper.scripts.normalize_main_evidence_audits import NormalizationError

    with pytest.raises(SystemExit, match="2"):
        normalize(paths)


def test_normalization_rejects_crlf_audit_csv(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    audit_path = paths["audits"] / "audits.csv"
    audit_path.write_bytes(audit_path.read_bytes().replace(b"\n", b"\r\n"))

    from paper.scripts.normalize_main_evidence_audits import (
        NormalizationError,
        normalize as normalize_rows,
    )

    with pytest.raises(NormalizationError, match="must use LF line endings"):
        normalize_rows(
            candidates_path=paths["candidates"],
            audits_dir=paths["audits"],
            v7_manifest_path=paths["v7"],
            source_archive=paths["archive"],
        )

EXPECTED_ACCESS_STATUS_ACTIONS = {
    "full_text": "user-fetch-or-document-limitation",
    "full_text_official": "archive-official-source",
    "full_text_public": "archive-public-full-text",
    "full_text_public_archive_corrupt": "replace-corrupt-local",
    "full_text_repository": "archive-public-full-text",
    "local-full-text": "user-fetch-or-document-limitation",
    "local-official-evidence": "archive-official-source",
    "local_full_text": "user-fetch-or-document-limitation",
    "metadata-only": "user-fetch-or-document-limitation",
    "metadata_only": "user-fetch-or-document-limitation",
    "metadata_only_official": "archive-official-source",
    "official-web-evidence": "archive-official-source",
    "official_artifact_only": "archive-official-source",
    "official_evidence": "archive-official-source",
    "public-full-text": "archive-public-full-text",
    "related_local_full_text": "replace-mismatched-local",
}


def test_access_status_mapping_is_complete_for_persisted_audits() -> None:
    from paper.scripts.normalize_main_evidence_audits import (
        ACCESS_STATUS_ACTIONS,
        NormalizationError,
        _action_for,
    )

    audit_dir = (
        Path(__file__).resolve().parents[1]
        / "paper/data/screening_work/v8/main_evidence_audits"
    )
    persisted_statuses: set[str] = set()
    for audit_path in sorted(audit_dir.glob("*.csv")):
        with audit_path.open(encoding="utf-8", newline="") as handle:
            persisted_statuses.update(
                row["access_status"] for row in csv.DictReader(handle)
            )

    assert persisted_statuses == set(EXPECTED_ACCESS_STATUS_ACTIONS)
    assert ACCESS_STATUS_ACTIONS == EXPECTED_ACCESS_STATUS_ACTIONS
    assert {status: _action_for(status) for status in persisted_statuses} == (
        EXPECTED_ACCESS_STATUS_ACTIONS
    )
    for status in ("not_public_full_text", "unofficial_full_text"):
        with pytest.raises(NormalizationError, match="unknown access_status"):
            _action_for(status)


@pytest.mark.parametrize("status", ("not_public_full_text", "unofficial_full_text"))
def test_normalization_rejects_unknown_status_on_trusted_row(
    tmp_path: Path, status: str
) -> None:
    paths = build_inputs(tmp_path)
    audit_path = paths["audits"] / "audits.csv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audits = list(csv.DictReader(handle))
    audits[0]["access_status"] = status
    write_csv(audit_path, AUDIT_HEADER, audits)

    from paper.scripts.normalize_main_evidence_audits import (
        NormalizationError,
        normalize as normalize_rows,
    )

    with pytest.raises(NormalizationError, match="unknown access_status"):
        normalize_rows(
            candidates_path=paths["candidates"],
            audits_dir=paths["audits"],
            v7_manifest_path=paths["v7"],
            source_archive=paths["archive"],
        )
