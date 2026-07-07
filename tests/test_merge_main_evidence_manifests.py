from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

import paper.scripts.merge_main_evidence_manifests as merge_module
from paper.scripts.merge_main_evidence_manifests import (
    ACQUISITION_QUEUE_HEADER,
    C0122_ARTIFACT_SHA256,
    C0122_LOCAL_FILENAME,
    EVIDENCE_PACKET_HEADER,
    MergeError,
    merge,
)
from paper.scripts.prepare_screening_batches import (
    CANDIDATE_HEADER,
    parse_evidence_packet_manifest,
)

def write_csv(path: Path, header: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            title=f"Evidence study {number}",
            authors="Fixture Author",
            year="2026",
            venue="Fixture venue",
            doi="",
            url=f"https://example.test/{candidate_id}",
            source_type="article",
            discovery_stream="test",
            discovery_query="test",
            discovery_agent="test",
            screening_status="candidate",
            exclusion_reason="",
            metadata_status="verified",
            metadata_evidence="fixture metadata",
        )
        if candidate_id == "C0122":
            row.update(
                title=(
                    "Automatically Generating Content for Testing Autonomous "
                    "Vehicles from User Descriptions"
                ),
                doi="10.1109/icse-nier66352.2025.00021",
                url="https://ieeexplore.ieee.org/document/11023959/",
            )
        rows.append(row)
    return rows

def provisional_row(candidate_id: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "artifact_id": "provisional-metadata",
        "artifact_role": "metadata-only",
        "source_url": f"https://example.test/{candidate_id}",
        "evidence_version": "fixture-v1",
        "evidence_retrieved_on": "2026-07-06",
        "access_status": "abstract_only",
        "evidence_archive_url": "NR",
        "evidence_sha256": "NR",
        "local_filename": "NR",
        "redistribution_status": "metadata-only",
        "retrieval_notes": (
            "attempted: doi_or_publisher=no fixture bytes | "
            "title_author=fixture metadata verified | "
            "scholarly_index_or_repository=no fixture artifact | "
            "official_page=fixture source endpoint checked; "
            "outcome: metadata-only record requires acquisition"
        ),
    }

def byte_row(candidate_id: str, payload: bytes, *, filename: str, access: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "artifact_id": "primary-pdf",
        "artifact_role": "primary-report",
        "source_url": f"https://example.test/{candidate_id}/artifact.pdf",
        "evidence_version": "fixture-v1",
        "evidence_retrieved_on": "2026-07-06",
        "access_status": access,
        "evidence_archive_url": "NR",
        "evidence_sha256": hashlib.sha256(payload).hexdigest(),
        "local_filename": filename,
        "redistribution_status": "local-restricted",
        "retrieval_notes": "Fixture bytes bind this candidate to the primary artifact.",
    }

def queue_row(candidate: dict[str, str]) -> dict[str, str]:
    return {
        "candidate_id": candidate["candidate_id"],
        "title": candidate["title"],
        "source_url": candidate["url"],
        "raw_access_status": "metadata_only",
        "action": "user-fetch-or-document-limitation",
        "priority": "high",
        "limitation_note": "Fixture has no archived primary artifact.",
    }

def build_inputs(tmp_path: Path) -> dict[str, Path]:
    candidates = candidate_rows()
    candidates_path = tmp_path / "candidates.csv"
    write_csv(candidates_path, CANDIDATE_HEADER, candidates)

    v7 = tmp_path / "source_archive" / "v7"
    v8 = tmp_path / "source_archive" / "v8"
    trusted_ids = ["C0001", *(f"C{number:04d}" for number in range(41, 68))]
    public_ids = [f"C{number:04d}" for number in range(2, 26)]
    official_ids = [f"C{number:04d}" for number in range(26, 41)]
    difficult_ids = [f"C{number:04d}" for number in range(68, 82)]
    trusted_payloads: dict[str, bytes] = {}
    for candidate_id in trusted_ids:
        payload = f"trusted v7 bytes {candidate_id}".encode("ascii")
        trusted_payloads[candidate_id] = payload
        (v7 / candidate_id).mkdir(parents=True)
        (v7 / candidate_id / "trusted.pdf").write_bytes(payload)
    public_rows = []
    for candidate_id in public_ids:
        payload = f"trusted public overlay bytes {candidate_id}".encode("ascii")
        (v8 / candidate_id).mkdir(parents=True)
        (v8 / candidate_id / "public.pdf").write_bytes(payload)
        public_rows.append(byte_row(candidate_id, payload, filename=f"{candidate_id}/public.pdf", access="full_text"))
    official_rows = []
    for candidate_id in official_ids:
        payload = f"trusted official overlay bytes {candidate_id}".encode("ascii")
        (v8 / candidate_id).mkdir(parents=True)
        (v8 / candidate_id / "official.pdf").write_bytes(payload)
        official_rows.append(byte_row(candidate_id, payload, filename=f"{candidate_id}/official.pdf", access="official_documentation"))
    difficult_rows = []
    for candidate_id in difficult_ids:
        payload = f"trusted difficult overlay bytes {candidate_id}".encode("ascii")
        (v8 / candidate_id).mkdir(parents=True)
        (v8 / candidate_id / "difficult.pdf").write_bytes(payload)
        difficult_rows.append(byte_row(candidate_id, payload, filename=f"{candidate_id}/difficult.pdf", access="full_text"))

    draft = [provisional_row(candidate["candidate_id"]) for candidate in candidates]
    for candidate_id in trusted_ids:
        draft[int(candidate_id[1:]) - 1] = byte_row(candidate_id, trusted_payloads[candidate_id], filename=f"{candidate_id}/trusted.pdf", access="full_text")
    draft_path = tmp_path / "draft.csv"
    write_csv(draft_path, EVIDENCE_PACKET_HEADER, draft)

    public_path = tmp_path / "high_public.csv"
    write_csv(public_path, EVIDENCE_PACKET_HEADER, public_rows)
    official_path = tmp_path / "high_official.csv"
    write_csv(official_path, EVIDENCE_PACKET_HEADER, official_rows)
    difficult_path = tmp_path / "high_difficult.csv"
    write_csv(difficult_path, EVIDENCE_PACKET_HEADER, difficult_rows)

    queue_path = tmp_path / "queue.csv"
    write_csv(queue_path, ACQUISITION_QUEUE_HEADER, [queue_row(row) for row in candidates if row["candidate_id"] not in trusted_ids])
    return {
        "candidates": candidates_path,
        "draft": draft_path,
        "public": public_path,
        "official": official_path,
        "difficult": difficult_path,
        "queue": queue_path,
        "v7": v7,
        "v8": v8,
        "supplied": Path(__file__).resolve().parents[1]
        / "paper/data/source_archive/provided/ICSE-NIER66352.2025.00021.pdf",
        "manifest": tmp_path / "output" / "manifest.csv",
        "remaining": tmp_path / "output" / "remaining.csv",
        "report": tmp_path / "output" / "report.md",
        "supplied_manifest": tmp_path / "output" / "supplied_manifest.csv",
        "supplied_summary": tmp_path / "output" / "supplied_summary.md",
    }

def run_merge(paths: dict[str, Path]) -> None:
    merge(
        candidates_path=paths["candidates"],
        draft_path=paths["draft"],
        high_public_path=paths["public"],
        high_official_path=paths["official"],
        high_difficult_path=paths["difficult"],
        acquisition_queue_path=paths["queue"],
        source_archive_v7=paths["v7"],
        source_archive_v8=paths["v8"],
        supplied_pdf_path=paths["supplied"],
        supplied_manifest_path=paths["supplied_manifest"],
        supplied_summary_path=paths["supplied_summary"],
        manifest_output_path=paths["manifest"],
        remaining_queue_output_path=paths["remaining"],
        report_output_path=paths["report"],
    )

def output_paths(paths: dict[str, Path]) -> tuple[Path, ...]:
    return (
        paths["supplied_manifest"],
        paths["supplied_summary"],
        paths["manifest"],
        paths["remaining"],
        paths["report"],
    )

def preexisting_output_bytes(paths: dict[str, Path]) -> dict[Path, bytes]:
    expected: dict[Path, bytes] = {}
    for number, path in enumerate(output_paths(paths), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"previous output {number}\n".encode("ascii")
        path.write_bytes(payload)
        expected[path] = payload
    return expected

def test_merge_produces_parser_valid_202_row_manifest_and_exact_queue_complement(
    tmp_path: Path,
) -> None:
    paths = build_inputs(tmp_path)

    run_merge(paths)

    manifest_bytes = paths["manifest"].read_bytes()
    assert b"\r" not in manifest_bytes
    with paths["manifest"].open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 202
    assert tuple(manifest[0]) == EVIDENCE_PACKET_HEADER
    assert [row["candidate_id"] for row in manifest] == [f"C{number:04d}" for number in range(1, 203)]
    assert sum(row["evidence_sha256"] != "NR" for row in manifest) == 82
    parse_evidence_packet_manifest(
        manifest_bytes,
        allowed_candidate_ids={f"C{number:04d}" for number in range(1, 203)},
        source_archive=paths["v8"],
    )

    with paths["remaining"].open(encoding="utf-8", newline="") as handle:
        remaining = list(csv.DictReader(handle))
    assert tuple(remaining[0]) == ACQUISITION_QUEUE_HEADER
    assert len(remaining) == 120
    assert {row["candidate_id"] for row in remaining} == {
        row["candidate_id"] for row in manifest if row["evidence_sha256"] == "NR"
    }

def test_merge_rejects_output_path_that_aliases_direct_input(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    original_draft = paths["draft"].read_bytes()
    paths["manifest"] = paths["draft"]

    with pytest.raises(MergeError, match="must not alias protected path"):
        run_merge(paths)

    assert paths["draft"].read_bytes() == original_draft

def test_merge_rejects_output_inside_source_archive_root(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    paths["manifest"] = paths["v7"].parent / "provided" / "published.csv"

    with pytest.raises(MergeError, match="must not alias protected path"):
        run_merge(paths)

def test_merge_rejects_output_symlink_that_aliases_input(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    original_draft = paths["draft"].read_bytes()
    paths["manifest"].parent.mkdir(parents=True)
    paths["manifest"].symlink_to(paths["draft"])

    with pytest.raises(MergeError, match="must not alias protected path"):
        run_merge(paths)

    assert paths["draft"].read_bytes() == original_draft

def test_merge_rejects_duplicate_resolved_output_path(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    paths["remaining"] = paths["manifest"]

    with pytest.raises(MergeError, match="output paths must be distinct"):
        run_merge(paths)

def test_late_staging_failure_preserves_all_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_inputs(tmp_path)
    original_outputs = preexisting_output_bytes(paths)
    real_stage = merge_module._write_staged_payload
    write_calls = 0

    def fail_later_stage(path: Path, payload: bytes) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 4:
            raise OSError("injected later staged-write failure")
        real_stage(path, payload)

    monkeypatch.setattr(merge_module, "_write_staged_payload", fail_later_stage)

    with pytest.raises(MergeError, match="unable to stage output payloads"):
        run_merge(paths)

    assert {path: path.read_bytes() for path in output_paths(paths)} == original_outputs

def test_staging_directory_fsync_failure_preserves_outputs_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_inputs(tmp_path)
    original_outputs = preexisting_output_bytes(paths)
    real_fsync_directory = merge_module._fsync_directory

    def fail_staging_directory_fsync(path: Path) -> None:
        if path.name.startswith(".merge-main-evidence."):
            raise OSError("injected staging-directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        merge_module, "_fsync_directory", fail_staging_directory_fsync
    )

    with pytest.raises(MergeError, match="unable to stage output payloads"):
        run_merge(paths)

    assert {path: path.read_bytes() for path in output_paths(paths)} == original_outputs
    for output_path in output_paths(paths):
        assert not list(output_path.parent.glob(".merge-main-evidence.*.tmp"))

def test_late_publication_failure_rolls_back_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_inputs(tmp_path)
    original_outputs = preexisting_output_bytes(paths)
    real_replace = merge_module.os.replace
    replace_calls = 0

    def fail_later_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("injected later publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(merge_module.os, "replace", fail_later_replace)

    with pytest.raises(MergeError, match="unable to publish staged output payloads"):
        run_merge(paths)

    assert {path: path.read_bytes() for path in output_paths(paths)} == original_outputs

def test_merge_rejects_duplicate_overlay_candidate_id(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    with paths["public"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    write_csv(paths["public"], EVIDENCE_PACKET_HEADER, rows + rows)

    with pytest.raises(MergeError, match="duplicate candidate_id"):
        run_merge(paths)

def test_merge_rejects_cross_overlay_candidate_id_collision(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    with paths["public"].open(encoding="utf-8", newline="") as handle:
        public_rows = list(csv.DictReader(handle))
    with paths["official"].open(encoding="utf-8", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    official_rows[0]["candidate_id"] = public_rows[0]["candidate_id"]
    write_csv(paths["official"], EVIDENCE_PACKET_HEADER, official_rows)

    with pytest.raises(MergeError, match="overlays: duplicate candidate_id 'C0002'"):
        run_merge(paths)

def test_merge_rejects_unknown_overlay_candidate_id(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    with paths["public"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["candidate_id"] = "C9999"
    write_csv(paths["public"], EVIDENCE_PACKET_HEADER, rows)

    with pytest.raises(MergeError, match="unknown candidate_id"):
        run_merge(paths)

def test_merge_rejects_hash_mismatched_v7_copy(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    (paths["v7"] / "C0001" / "trusted.pdf").write_bytes(b"tampered v7 bytes")

    with pytest.raises(MergeError, match="SHA-256 mismatch"):
        run_merge(paths)

def test_merge_refuses_mismatched_existing_destination(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    (paths["v8"] / "C0001").mkdir(parents=True)
    (paths["v8"] / "C0001" / "trusted.pdf").write_bytes(b"different destination bytes")

    with pytest.raises(MergeError, match="refusing overwrite"):
        run_merge(paths)

def test_merge_binds_supplied_c0122_with_declared_hash_and_provenance(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)

    run_merge(paths)

    with paths["manifest"].open(encoding="utf-8", newline="") as handle:
        by_id = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    supplied = by_id["C0122"]
    assert supplied["evidence_sha256"] == C0122_ARTIFACT_SHA256
    assert supplied["local_filename"] == C0122_LOCAL_FILENAME
    assert supplied["access_status"] == "full_text"
    assert supplied["redistribution_status"] == "local-restricted"
    assert "user supplied" in supplied["retrieval_notes"].casefold()
    assert hashlib.sha256((paths["v8"] / C0122_LOCAL_FILENAME).read_bytes()).hexdigest() == C0122_ARTIFACT_SHA256
    assert "does not grant redistribution rights" in paths["supplied_summary"].read_text(encoding="utf-8")

def test_merge_rejects_overlay_that_does_not_replace_a_provisional_draft_row(
    tmp_path: Path,
) -> None:
    paths = build_inputs(tmp_path)
    payload = b"unexpected trusted draft bytes"
    (paths["v7"] / "C0002").mkdir(parents=True)
    (paths["v7"] / "C0002" / "trusted.pdf").write_bytes(payload)
    with paths["draft"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1] = byte_row("C0002", payload, filename="C0002/trusted.pdf", access="full_text")
    write_csv(paths["draft"], EVIDENCE_PACKET_HEADER, rows)

    with pytest.raises(MergeError, match="must replace a provisional draft row"):
        run_merge(paths)

def test_merge_rejects_output_path_that_aliases_difficult_overlay(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    original_difficult = paths["difficult"].read_bytes()
    paths["manifest"] = paths["difficult"]

    with pytest.raises(MergeError, match="must not alias protected path"):
        run_merge(paths)

    assert paths["difficult"].read_bytes() == original_difficult

def test_merge_rejects_difficult_overlay_candidate_id_collision(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    with paths["public"].open(encoding="utf-8", newline="") as handle:
        public_rows = list(csv.DictReader(handle))
    with paths["difficult"].open(encoding="utf-8", newline="") as handle:
        difficult_rows = list(csv.DictReader(handle))
    difficult_rows[0]["candidate_id"] = public_rows[0]["candidate_id"]
    write_csv(paths["difficult"], EVIDENCE_PACKET_HEADER, difficult_rows)

    with pytest.raises(MergeError, match="overlays: duplicate candidate_id 'C0002'"):
        run_merge(paths)

def test_merge_requires_exact_14_row_difficult_overlay(tmp_path: Path) -> None:
    paths = build_inputs(tmp_path)
    with paths["difficult"].open(encoding="utf-8", newline="") as handle:
        difficult_rows = list(csv.DictReader(handle))
    write_csv(paths["difficult"], EVIDENCE_PACKET_HEADER, difficult_rows[:-1])

    with pytest.raises(MergeError, match="high-difficult overlay: expected 14 rows, found 13"):
        run_merge(paths)

def test_merge_rejects_difficult_overlay_c0122_before_copying_or_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_inputs(tmp_path)
    original_outputs = preexisting_output_bytes(paths)
    with paths["difficult"].open(encoding="utf-8", newline="") as handle:
        difficult_rows = list(csv.DictReader(handle))
    difficult_rows[0]["candidate_id"] = "C0122"
    write_csv(paths["difficult"], EVIDENCE_PACKET_HEADER, difficult_rows)

    def fail_copy(_: bytes, **__: object) -> None:
        pytest.fail("reserved-overlay preflight must run before artifact copying")

    monkeypatch.setattr(merge_module, "_copy_exclusive", fail_copy)

    with pytest.raises(MergeError, match="overlays: reserved candidate_id 'C0122'"):
        run_merge(paths)

    assert {path: path.read_bytes() for path in output_paths(paths)} == original_outputs
