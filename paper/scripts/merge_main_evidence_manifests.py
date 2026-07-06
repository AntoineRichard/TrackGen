from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

from paper.scripts.prepare_screening_batches import (
    CANDIDATE_HEADER,
    EVIDENCE_PACKET_HEADER,
    SnapshotError,
    parse_evidence_packet_manifest,
)


ACQUISITION_QUEUE_HEADER = (
    "candidate_id",
    "title",
    "source_url",
    "raw_access_status",
    "action",
    "priority",
    "limitation_note",
)
CANDIDATE_COUNT = 202
HIGH_PUBLIC_COUNT = 24
HIGH_OFFICIAL_COUNT = 15
BYTE_BACKED_COUNT = 68
PROVISIONAL_COUNT = 134
QUEUE_INPUT_COUNT = 174
C0122_CANDIDATE_ID = "C0122"
C0122_ARTIFACT_SHA256 = "8b5119b6edc8bce2bfb360a279a33b20e47fbca9d803a61a2e9cb18890cd020a"
C0122_LOCAL_FILENAME = (
    "C0122/automatically_generating_content_testing_autonomous_vehicles_icse_nier_2025.pdf"
)
_ACTION_ORDER = {
    "replace-mismatched-local": 0,
    "replace-corrupt-local": 1,
    "archive-public-full-text": 2,
    "archive-official-source": 3,
    "user-fetch-or-document-limitation": 4,
}
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class MergeError(ValueError):
    """A frozen evidence input or destination cannot be merged safely."""


def _canonical_csv_bytes(
    header: tuple[str, ...], rows: Iterable[dict[str, str]]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _read_canonical_csv(
    path: Path, header: tuple[str, ...], *, label: str
) -> list[dict[str, str]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MergeError(f"{label}: unable to read {path}") from exc
    if b"\r" in payload:
        raise MergeError(f"{label}: must use LF line endings")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MergeError(f"{label}: must be UTF-8") from exc
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        actual_header = tuple(next(reader))
    except StopIteration as exc:
        raise MergeError(f"{label}: must contain an exact header") from exc
    if actual_header != header:
        raise MergeError(f"{label}: header must be exactly {header!r}")
    rows: list[dict[str, str]] = []
    try:
        for row_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise MergeError(
                    f"{label}:{row_number}: must contain exactly {len(header)} fields"
                )
            if any(value != value.strip() for value in values):
                raise MergeError(f"{label}:{row_number}: fields must be trimmed")
            rows.append(dict(zip(header, values, strict=True)))
    except csv.Error as exc:
        raise MergeError(f"{label}: malformed CSV") from exc
    if payload != _canonical_csv_bytes(header, rows):
        raise MergeError(f"{label}: must use canonical UTF-8/LF CSV bytes")
    return rows


def _require_exact_candidate_rows(
    path: Path,
) -> tuple[list[dict[str, str]], set[str]]:
    rows = _read_canonical_csv(path, CANDIDATE_HEADER, label="candidates")
    if len(rows) != CANDIDATE_COUNT:
        raise MergeError(f"candidates: expected {CANDIDATE_COUNT} rows, found {len(rows)}")
    ids = [row["candidate_id"] for row in rows]
    if any(not candidate_id for candidate_id in ids) or len(set(ids)) != len(ids):
        raise MergeError("candidates: candidate_id values must be nonempty and unique")
    return rows, set(ids)


def _require_exact_manifest(
    path: Path,
    *,
    label: str,
    allowed_ids: set[str],
    expected_count: int | None = None,
) -> list[dict[str, str]]:
    rows = _read_canonical_csv(path, EVIDENCE_PACKET_HEADER, label=label)
    ids = [row["candidate_id"] for row in rows]
    unknown = sorted(set(ids) - allowed_ids, key=str.encode)
    if unknown:
        raise MergeError(f"{label}: unknown candidate_id {unknown[0]!r}")
    if len(set(ids)) != len(ids):
        duplicates = sorted(
            (candidate_id for candidate_id, count in Counter(ids).items() if count > 1),
            key=str.encode,
        )
        raise MergeError(f"{label}: duplicate candidate_id {duplicates[0]!r}")
    if expected_count is not None and len(rows) != expected_count:
        raise MergeError(f"{label}: expected {expected_count} rows, found {len(rows)}")
    return rows


def _relative_path(value: str, *, context: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        value == "NR"
        or "\\" in value
        or relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise MergeError(f"{context}: local_filename must be a normalized relative POSIX path")
    return relative


def _validated_artifact_bytes(
    row: dict[str, str], *, source_archive: Path, context: str
) -> bytes:
    declared_hash = row["evidence_sha256"]
    if len(declared_hash) != 64 or any(character not in "0123456789abcdef" for character in declared_hash):
        raise MergeError(f"{context}: evidence_sha256 must be lowercase 64-hex")
    relative = _relative_path(row["local_filename"], context=context)
    try:
        root = source_archive.resolve(strict=True)
        path = (root / Path(*relative.parts)).resolve(strict=True)
        path.relative_to(root)
        if not path.is_file():
            raise MergeError(f"{context}: local artifact is not a regular file")
        payload = path.read_bytes()
    except OSError as exc:
        raise MergeError(f"{context}: unable to read local artifact") from exc
    except ValueError as exc:
        raise MergeError(f"{context}: local artifact escapes source archive") from exc
    if hashlib.sha256(payload).hexdigest() != declared_hash:
        raise MergeError(f"{context}: local artifact SHA-256 mismatch")
    return payload


def _validate_overlays(
    rows: list[dict[str, str]], *, label: str, source_archive_v8: Path
) -> None:
    for row in rows:
        if row["evidence_sha256"] == "NR" or row["local_filename"] == "NR":
            raise MergeError(f"{label}: {row['candidate_id']} must have real hash-valid bytes")
        _validated_artifact_bytes(
            row,
            source_archive=source_archive_v8,
            context=f"{label} {row['candidate_id']}",
        )


def _copy_exclusive(
    payload: bytes,
    *,
    expected_hash: str,
    source_archive_v8: Path,
    local_filename: str,
    context: str,
) -> None:
    relative = _relative_path(local_filename, context=context)
    try:
        root = source_archive_v8.resolve(strict=True)
    except OSError as exc:
        raise MergeError(f"{context}: unable to resolve v8 source archive") from exc
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise MergeError(f"{context}: destination escapes source archive") from exc
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            existing = destination.resolve(strict=True)
            existing.relative_to(root)
            existing_payload = existing.read_bytes()
        except (OSError, ValueError) as exc:
            raise MergeError(f"{context}: unable to inspect existing destination") from exc
        if hashlib.sha256(existing_payload).hexdigest() != expected_hash:
            raise MergeError(f"{context}: refusing overwrite of mismatched existing destination")
        return
    except OSError as exc:
        raise MergeError(f"{context}: unable to create destination exclusively") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MergeError(f"{context}: unable to write destination") from exc
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
        raise MergeError(f"{context}: destination SHA-256 mismatch after copy")


def _validate_c0122_pdf(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MergeError("C0122: unable to read user-supplied PDF") from exc
    if hashlib.sha256(payload).hexdigest() != C0122_ARTIFACT_SHA256:
        raise MergeError("C0122: supplied PDF SHA-256 mismatch")
    try:
        info = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        text = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise MergeError("C0122: unable to verify supplied PDF identity") from exc
    required_info = (
        "Automatically Generating Content for Testing Autonomous Vehicles from User Descriptions",
        "10.1109/ICSE-NIER66352.2025.00021",
        "Pages:           5",
    )
    normalized_text = " ".join(text.split())
    if any(token not in info for token in required_info) or required_info[0] not in normalized_text:
        raise MergeError("C0122: supplied PDF is not the five-page ICSE-NIER paper")
    return payload


def _c0122_row() -> dict[str, str]:
    return {
        "candidate_id": C0122_CANDIDATE_ID,
        "artifact_id": "primary-pdf",
        "artifact_role": "primary-report",
        "source_url": "https://doi.org/10.1109/icse-nier66352.2025.00021",
        "evidence_version": "ICSE-NIER 2025 DOI 10.1109/icse-nier66352.2025.00021",
        "evidence_retrieved_on": "2026-07-06",
        "access_status": "full_text",
        "evidence_archive_url": "NR",
        "evidence_sha256": C0122_ARTIFACT_SHA256,
        "local_filename": C0122_LOCAL_FILENAME,
        "redistribution_status": "local-restricted",
        "retrieval_notes": (
            "User supplied five-page ICSE-NIER PDF; DOI, title, page count, and SHA-256 "
            "were verified locally. Local storage does not grant redistribution rights."
        ),
    }


def _read_queue(path: Path, *, candidate_ids: set[str]) -> list[dict[str, str]]:
    rows = _read_canonical_csv(path, ACQUISITION_QUEUE_HEADER, label="acquisition queue")
    if len(rows) != QUEUE_INPUT_COUNT:
        raise MergeError(f"acquisition queue: expected {QUEUE_INPUT_COUNT} rows, found {len(rows)}")
    ids = [row["candidate_id"] for row in rows]
    if set(ids) - candidate_ids:
        raise MergeError("acquisition queue: contains an unknown candidate_id")
    if len(set(ids)) != len(ids):
        raise MergeError("acquisition queue: candidate_id values must be unique")
    if any(row["action"] not in _ACTION_ORDER for row in rows):
        raise MergeError("acquisition queue: contains an unknown action")
    if any(row["priority"] not in _PRIORITY_ORDER for row in rows):
        raise MergeError("acquisition queue: contains an unknown priority")
    expected = sorted(
        rows,
        key=lambda row: (_PRIORITY_ORDER[row["priority"]], row["candidate_id"].encode()),
    )
    if rows != expected:
        raise MergeError("acquisition queue: must preserve canonical priority/candidate ordering")
    return rows


def _validate_final(
    rows: list[dict[str, str]], *, candidate_ids: set[str], source_archive_v8: Path
) -> bytes:
    if len(rows) != CANDIDATE_COUNT:
        raise MergeError(f"final manifest: expected {CANDIDATE_COUNT} rows, found {len(rows)}")
    ids = [row["candidate_id"] for row in rows]
    if set(ids) != candidate_ids or len(set(ids)) != CANDIDATE_COUNT:
        raise MergeError("final manifest: candidate coverage must be exact and unique")
    rows.sort(key=lambda row: (row["candidate_id"].encode(), row["artifact_id"].encode()))
    payload = _canonical_csv_bytes(EVIDENCE_PACKET_HEADER, rows)
    try:
        parse_evidence_packet_manifest(
            payload,
            allowed_candidate_ids=candidate_ids,
            source_archive=source_archive_v8,
        )
    except SnapshotError as exc:
        raise MergeError(f"final manifest: rejected by evidence parser: {exc}") from exc
    byte_count = sum(row["evidence_sha256"] != "NR" for row in rows)
    if byte_count != BYTE_BACKED_COUNT:
        raise MergeError(f"final manifest: expected {BYTE_BACKED_COUNT} byte-backed rows, found {byte_count}")
    if len(rows) - byte_count != PROVISIONAL_COUNT:
        raise MergeError("final manifest: provisional metadata-only count is not exact")
    return payload


def _report(
    rows: list[dict[str, str]],
    remaining: list[dict[str, str]],
    sources: dict[str, str],
) -> str:
    access = Counter(row["access_status"] for row in rows)
    redistribution = Counter(row["redistribution_status"] for row in rows)
    acquisition = Counter(sources[row["candidate_id"]] for row in rows)
    action_priority = Counter((row["action"], row["priority"]) for row in remaining)

    def table(counter: Counter[str], title: str) -> list[str]:
        lines = [f"## {title}", "", "| Value | Count |", "| --- | ---: |"]
        lines.extend(f"| {value} | {counter[value]} |" for value in sorted(counter, key=str.encode))
        return lines

    lines = [
        "# Main Evidence Packet",
        "",
        "## Coverage",
        "",
        f"- Final manifest rows: {len(rows)}/{CANDIDATE_COUNT}.",
        f"- Byte-backed rows: {sum(row['evidence_sha256'] != 'NR' for row in rows)}.",
        f"- Provisional metadata-only rows: {sum(row['evidence_sha256'] == 'NR' for row in rows)}.",
        f"- Remaining acquisition queue rows: {len(remaining)}.",
        "",
    ]
    lines.extend(table(access, "Access Status"))
    lines.extend([""])
    lines.extend(table(redistribution, "Redistribution Status"))
    lines.extend([""])
    lines.extend(table(acquisition, "Acquisition Source"))
    lines.extend(["", "## Remaining Action/Priority", "", "| Action | Priority | Count |", "| --- | --- | ---: |"])
    lines.extend(
        f"| {action} | {priority} | {action_priority[(action, priority)]} |"
        for action, priority in sorted(
            action_priority,
            key=lambda item: (_ACTION_ORDER[item[0]], _PRIORITY_ORDER[item[1]]),
        )
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Evidence bytes under `paper/data/source_archive/v8/` are deliberately untracked local artifacts; the manifest hashes bind the reviewed bytes but do not distribute them.",
            "- Public and official upstream endpoints can change or disappear. Their recorded URLs and versions are provenance, not a guarantee of future availability.",
            "- C0122 was supplied by the user and is stored locally as restricted evidence; that supply does not grant redistribution rights.",
            "",
        ]
    )
    return "\n".join(lines)


def _supplied_summary() -> str:
    return "\n".join(
        (
            "# Supplied C0122 Evidence",
            "",
            "The user supplied the five-page ICSE-NIER 2025 PDF for C0122. The merge tool verifies its title, DOI, page count, and SHA-256 before binding it as local-restricted full text.",
            "",
            "This local copy does not grant redistribution rights.",
            "",
        )
    )


def _resolved_path(path: Path, *, label: str) -> Path:
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise MergeError(f"{label}: unable to resolve path {path}") from exc


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return _resolved_path(left, label="path") == _resolved_path(right, label="path")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_output_paths(
    *,
    output_paths: tuple[Path, ...],
    input_paths: tuple[Path, ...],
    source_archive_roots: tuple[Path, ...],
) -> None:
    resolved_roots = tuple(
        _resolved_path(root, label="source archive") for root in source_archive_roots
    )
    for index, output in enumerate(output_paths):
        resolved_output = _resolved_path(output, label="output")
        if any(_paths_alias(output, other) for other in output_paths[:index]):
            raise MergeError(f"output paths must be distinct; duplicate {output}")
        if any(_paths_alias(output, input_path) for input_path in input_paths):
            raise MergeError(f"output {output} must not alias protected path")
        if any(_is_within(resolved_output, root) for root in resolved_roots):
            raise MergeError(f"output {output} must not alias protected path")


def _write_staged_payload(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("staged output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_output_payloads(
    output_payloads: tuple[tuple[Path, bytes], ...],
) -> None:
    staging_dirs: dict[Path, Path] = {}
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path] = {}
    original_outputs: dict[Path, bytes | None] = {}
    try:
        for destination, _ in output_payloads:
            destination.parent.mkdir(parents=True, exist_ok=True)
            original_outputs[destination] = (
                destination.read_bytes() if destination.exists() else None
            )
        for index, (destination, payload) in enumerate(output_payloads, start=1):
            staging_dir = staging_dirs.get(destination.parent)
            if staging_dir is None:
                staging_dir = Path(
                    tempfile.mkdtemp(
                        prefix=".merge-main-evidence.",
                        suffix=".tmp",
                        dir=destination.parent,
                    )
                )
                staging_dirs[destination.parent] = staging_dir
            staged_path = staging_dir / f"output-{index:02d}"
            _write_staged_payload(staged_path, payload)
            staged.append((destination, staged_path))
            original = original_outputs[destination]
            if original is not None:
                backup_path = staging_dir / f"backup-{index:02d}"
                _write_staged_payload(backup_path, original)
                backups[destination] = backup_path
        for staging_dir in staging_dirs.values():
            _fsync_directory(staging_dir)
    except (OSError, RuntimeError) as exc:
        for staging_dir in staging_dirs.values():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise MergeError("unable to stage output payloads") from exc

    replaced: list[Path] = []
    try:
        for destination, staged_path in staged:
            os.replace(staged_path, destination)
            replaced.append(destination)
        for parent in staging_dirs:
            _fsync_directory(parent)
    except OSError as exc:
        rollback_errors: list[str] = []
        for destination in reversed(replaced):
            try:
                backup_path = backups.get(destination)
                if backup_path is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup_path, destination)
            except OSError as rollback_error:
                rollback_errors.append(
                    f"{destination}: {type(rollback_error).__name__}: {rollback_error}"
                )
        for parent in staging_dirs:
            try:
                _fsync_directory(parent)
            except OSError as rollback_error:
                rollback_errors.append(
                    f"{parent}: {type(rollback_error).__name__}: {rollback_error}"
                )
        if rollback_errors:
            raise MergeError(
                "unable to publish staged output payloads; rollback incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise MergeError("unable to publish staged output payloads") from exc
    finally:
        for staging_dir in staging_dirs.values():
            shutil.rmtree(staging_dir, ignore_errors=True)


def merge(
    *,
    candidates_path: Path,
    draft_path: Path,
    high_public_path: Path,
    high_official_path: Path,
    acquisition_queue_path: Path,
    source_archive_v7: Path,
    source_archive_v8: Path,
    supplied_pdf_path: Path,
    supplied_manifest_path: Path,
    supplied_summary_path: Path,
    manifest_output_path: Path,
    remaining_queue_output_path: Path,
    report_output_path: Path,
) -> None:
    output_paths = (
        supplied_manifest_path,
        supplied_summary_path,
        manifest_output_path,
        remaining_queue_output_path,
        report_output_path,
    )
    _validate_output_paths(
        output_paths=output_paths,
        input_paths=(
            candidates_path,
            draft_path,
            high_public_path,
            high_official_path,
            acquisition_queue_path,
            supplied_pdf_path,
        ),
        source_archive_roots=(
            source_archive_v7.parent,
            source_archive_v7,
            source_archive_v8.parent,
            source_archive_v8,
        ),
    )
    _, candidate_ids = _require_exact_candidate_rows(candidates_path)
    draft = _require_exact_manifest(
        draft_path,
        label="draft manifest",
        allowed_ids=candidate_ids,
        expected_count=CANDIDATE_COUNT,
    )
    if {row["candidate_id"] for row in draft} != candidate_ids:
        raise MergeError("draft manifest: candidate coverage must be exact")
    high_public = _require_exact_manifest(
        high_public_path,
        label="high-public overlay",
        allowed_ids=candidate_ids,
        expected_count=HIGH_PUBLIC_COUNT,
    )
    high_official = _require_exact_manifest(
        high_official_path,
        label="high-official overlay",
        allowed_ids=candidate_ids,
        expected_count=HIGH_OFFICIAL_COUNT,
    )
    collision = set(row["candidate_id"] for row in high_public) & set(
        row["candidate_id"] for row in high_official
    )
    if collision:
        raise MergeError(f"overlays: duplicate candidate_id {sorted(collision, key=str.encode)[0]!r}")
    draft_by_id = {row["candidate_id"]: row for row in draft}
    for overlay in (*high_public, *high_official):
        draft_row = draft_by_id[overlay["candidate_id"]]
        if (
            draft_row["evidence_sha256"] != "NR"
            or draft_row["local_filename"] != "NR"
            or draft_row["redistribution_status"] != "metadata-only"
        ):
            raise MergeError(
                f"overlay {overlay['candidate_id']}: must replace a provisional draft row"
            )
    _validate_overlays(high_public, label="high-public overlay", source_archive_v8=source_archive_v8)
    _validate_overlays(high_official, label="high-official overlay", source_archive_v8=source_archive_v8)

    queue = _read_queue(acquisition_queue_path, candidate_ids=candidate_ids)
    draft_provisional = {row["candidate_id"] for row in draft if row["evidence_sha256"] == "NR"}
    if {row["candidate_id"] for row in queue} != draft_provisional:
        raise MergeError("acquisition queue: must exactly cover draft provisional metadata-only rows")

    copied: dict[str, str] = {}
    for row in draft:
        if row["evidence_sha256"] == "NR":
            continue
        payload = _validated_artifact_bytes(
            row,
            source_archive=source_archive_v7,
            context=f"draft {row['candidate_id']}",
        )
        _copy_exclusive(
            payload,
            expected_hash=row["evidence_sha256"],
            source_archive_v8=source_archive_v8,
            local_filename=row["local_filename"],
            context=f"draft {row['candidate_id']}",
        )
        copied[row["candidate_id"]] = "trusted-v7"

    supplied_payload = _validate_c0122_pdf(supplied_pdf_path)
    supplied = _c0122_row()
    _copy_exclusive(
        supplied_payload,
        expected_hash=C0122_ARTIFACT_SHA256,
        source_archive_v8=source_archive_v8,
        local_filename=C0122_LOCAL_FILENAME,
        context="C0122",
    )

    final_by_id = {row["candidate_id"]: dict(row) for row in draft}
    sources = {candidate_id: copied.get(candidate_id, "provisional-metadata") for candidate_id in candidate_ids}
    for row in high_public:
        final_by_id[row["candidate_id"]] = dict(row)
        sources[row["candidate_id"]] = "high-public"
    for row in high_official:
        final_by_id[row["candidate_id"]] = dict(row)
        sources[row["candidate_id"]] = "high-official"
    final_by_id[C0122_CANDIDATE_ID] = supplied
    sources[C0122_CANDIDATE_ID] = "user-supplied"
    final_rows = list(final_by_id.values())
    manifest_payload = _validate_final(
        final_rows,
        candidate_ids=candidate_ids,
        source_archive_v8=source_archive_v8,
    )
    remaining = [row for row in queue if final_by_id[row["candidate_id"]]["evidence_sha256"] == "NR"]
    if len(remaining) != PROVISIONAL_COUNT or {
        row["candidate_id"] for row in remaining
    } != {row["candidate_id"] for row in final_rows if row["evidence_sha256"] == "NR"}:
        raise MergeError("remaining queue: must exactly cover final provisional metadata-only rows")

    _publish_output_payloads(
        (
            (supplied_manifest_path, _canonical_csv_bytes(EVIDENCE_PACKET_HEADER, [supplied])),
            (supplied_summary_path, _supplied_summary().encode("utf-8")),
            (manifest_output_path, manifest_payload),
            (
                remaining_queue_output_path,
                _canonical_csv_bytes(ACQUISITION_QUEUE_HEADER, remaining),
            ),
            (report_output_path, _report(final_rows, remaining, sources).encode("utf-8")),
        )
    )


def _default_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2].joinpath(*parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge frozen main-evidence manifests into the v8 packet.")
    parser.add_argument("--candidates", type=Path, default=_default_path("paper", "data", "screening_inputs", "v8", "candidates.csv"))
    parser.add_argument("--draft", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_manifest_draft.csv"))
    parser.add_argument("--high-public", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_acquisitions", "high_public_manifest.csv"))
    parser.add_argument("--high-official", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_acquisitions", "high_official_manifest.csv"))
    parser.add_argument("--acquisition-queue", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_acquisition_queue.csv"))
    parser.add_argument("--source-archive-v7", type=Path, default=_default_path("paper", "data", "source_archive", "v7"))
    parser.add_argument("--source-archive-v8", type=Path, default=_default_path("paper", "data", "source_archive", "v8"))
    parser.add_argument("--supplied-pdf", type=Path, default=_default_path("paper", "data", "source_archive", "provided", "ICSE-NIER66352.2025.00021.pdf"))
    parser.add_argument("--supplied-manifest-output", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_acquisitions", "supplied_manifest.csv"))
    parser.add_argument("--supplied-summary-output", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_acquisitions", "supplied_summary.md"))
    parser.add_argument("--manifest-output", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_packet_manifest.csv"))
    parser.add_argument("--remaining-queue-output", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_remaining_queue.csv"))
    parser.add_argument("--report-output", type=Path, default=_default_path("paper", "data", "screening_work", "v8", "main_evidence_packet_report.md"))
    args = parser.parse_args(argv)
    try:
        merge(
            candidates_path=args.candidates,
            draft_path=args.draft,
            high_public_path=args.high_public,
            high_official_path=args.high_official,
            acquisition_queue_path=args.acquisition_queue,
            source_archive_v7=args.source_archive_v7,
            source_archive_v8=args.source_archive_v8,
            supplied_pdf_path=args.supplied_pdf,
            supplied_manifest_path=args.supplied_manifest_output,
            supplied_summary_path=args.supplied_summary_output,
            manifest_output_path=args.manifest_output,
            remaining_queue_output_path=args.remaining_queue_output,
            report_output_path=args.report_output,
        )
    except MergeError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
