from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

from paper.scripts.prepare_screening_batches import (
    CANDIDATE_HEADER,
    EVIDENCE_PACKET_HEADER,
    SnapshotError,
    _canonical_evidence_http_url,
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
NORMALIZATION_DATE = "2026-07-06"
REPLACEMENT_ACTIONS = {
    "related_local_full_text": "replace-mismatched-local",
    "full_text_public_archive_corrupt": "replace-corrupt-local",
}
ACTION_ORDER = (
    "replace-mismatched-local",
    "replace-corrupt-local",
    "archive-public-full-text",
    "archive-official-source",
    "user-fetch-or-document-limitation",
)
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
HIGH_PRIORITY_TERMS = (
    "generat",
    "procedur",
    "track",
    "course",
    "road",
    "route",
    "scenario",
    "environment",
    "world",
    "map",
    "terrain",
)
MEDIUM_PRIORITY_TERMS = (
    "simulat",
    "autonomous",
    "driving",
    "vehicle",
    "racing",
    "benchmark",
    "navigation",
)


class NormalizationError(ValueError):
    """The frozen main-evidence inputs cannot be normalized safely."""


def _read_csv(
    path: Path,
    header: tuple[str, ...],
    *,
    label: str,
    require_lf: bool = False,
) -> list[dict[str, str]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise NormalizationError(f"{label}: unable to read {path}") from exc
    if require_lf and b"\r" in payload:
        raise NormalizationError(f"{label}: must use LF line endings")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NormalizationError(f"{label}: must be UTF-8") from exc

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        actual_header = tuple(next(reader))
    except StopIteration as exc:
        raise NormalizationError(f"{label}: must contain an exact header") from exc
    if actual_header != header:
        raise NormalizationError(f"{label}: header must be exactly {header!r}")

    rows: list[dict[str, str]] = []
    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise NormalizationError(
                f"{label}:{row_number}: must contain exactly {len(header)} fields"
            )
        row = dict(zip(header, values, strict=True))
        if any(value != value.strip() for value in values):
            raise NormalizationError(f"{label}:{row_number}: fields must be trimmed")
        rows.append(row)
    return rows


def _canonical_csv_bytes(
    header: tuple[str, ...], rows: Iterable[dict[str, str]]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _candidate_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path, CANDIDATE_HEADER, label="candidates")
    if len(rows) != CANDIDATE_COUNT:
        raise NormalizationError(
            f"candidates: expected exactly {CANDIDATE_COUNT} rows, found {len(rows)}"
        )
    candidate_ids = [row["candidate_id"] for row in rows]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise NormalizationError("candidates: candidate_id must be nonempty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise NormalizationError("candidates: candidate_id values must be unique")
    return rows


def _audit_rows(audits_dir: Path, candidate_ids: set[str]) -> dict[str, dict[str, str]]:
    paths = sorted(audits_dir.glob("*.csv"), key=lambda path: path.name.encode("utf-8"))
    if not paths:
        raise NormalizationError("main evidence audits: no CSV files found")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(
            _read_csv(
                path,
                AUDIT_HEADER,
                label=f"audit {path.name}",
                require_lf=True,
            )
        )
    if len(rows) != CANDIDATE_COUNT:
        raise NormalizationError(
            f"main evidence audits: expected exactly {CANDIDATE_COUNT} audit rows, found {len(rows)}"
        )

    audits: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if not candidate_id:
            raise NormalizationError("main evidence audits: candidate_id must be nonempty")
        if candidate_id not in candidate_ids:
            raise NormalizationError(
                f"main evidence audits: unknown candidate_id {candidate_id!r}"
            )
        if candidate_id in audits:
            raise NormalizationError(
                f"main evidence audits: duplicate candidate_id {candidate_id!r}"
            )
        for field in ("title", "source_url", "evidence_version", "access_status", "limitation_note"):
            if not row[field]:
                raise NormalizationError(
                    f"main evidence audits: {candidate_id} has an empty {field}"
                )
        audits[candidate_id] = row
    if set(audits) != candidate_ids:
        missing = sorted(candidate_ids - set(audits), key=lambda value: value.encode("utf-8"))
        unexpected = sorted(set(audits) - candidate_ids, key=lambda value: value.encode("utf-8"))
        raise NormalizationError(
            "main evidence audits: candidate coverage is not exact; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return audits


def _v7_rows(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, str]]:
    rows = _read_csv(path, EVIDENCE_PACKET_HEADER, label="v7 evidence manifest")
    by_candidate: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id not in candidate_ids:
            raise NormalizationError(
                f"v7 evidence manifest: unknown candidate_id {candidate_id!r}"
            )
        if candidate_id in by_candidate:
            raise NormalizationError(
                "v7 evidence manifest: more than one row for "
                f"candidate_id {candidate_id!r}"
            )
        by_candidate[candidate_id] = row
    return by_candidate


def _declared_bytes_are_trustworthy(row: dict[str, str], source_archive: Path) -> bool:
    declared_hash = row["evidence_sha256"]
    filename = row["local_filename"]
    if declared_hash == "NR" or filename == "NR":
        return False
    if re.fullmatch(r"[0-9a-f]{64}", declared_hash) is None:
        return False
    relative = PurePosixPath(filename)
    if (
        "\\" in filename
        or relative.is_absolute()
        or relative.as_posix() != filename
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return False
    try:
        root = source_archive.resolve(strict=True)
        evidence_path = (root / Path(*relative.parts)).resolve(strict=True)
        evidence_path.relative_to(root)
        if not evidence_path.is_file():
            return False
        return hashlib.sha256(evidence_path.read_bytes()).hexdigest() == declared_hash
    except (OSError, ValueError):
        return False


def _action_for(raw_access_status: str) -> str:
    status = raw_access_status.casefold()
    if status in REPLACEMENT_ACTIONS:
        return REPLACEMENT_ACTIONS[status]
    if "public" in status and ("full" in status or "text" in status):
        return "archive-public-full-text"
    if "repository" in status and "full" in status:
        return "archive-public-full-text"
    if "official" in status:
        return "archive-official-source"
    return "user-fetch-or-document-limitation"


def _priority_for(candidate: dict[str, str], action: str) -> str:
    haystack = " ".join(
        (candidate["title"], candidate["source_type"], candidate["url"])
    ).casefold()
    if any(term in haystack for term in HIGH_PRIORITY_TERMS):
        return "high"
    if action.startswith("replace-") or any(
        term in haystack for term in MEDIUM_PRIORITY_TERMS
    ):
        return "medium"
    return "low"


def _limited_retrieval_notes(limitation_note: str) -> str:
    official_page = re.sub(r"[|;\r\n]+", " ", limitation_note).strip()
    if len(official_page) < 12 or not any(character.isalnum() for character in official_page):
        official_page = "audit notes no verified bytes"
    return (
        "attempted: doi_or_publisher=normalizer did not download bytes | "
        "title_author=metadata audit row inspected | "
        "scholarly_index_or_repository=no verified local artifact | "
        f"official_page={official_page}; "
        "outcome: metadata-only record requires acquisition"
    )


def _provisional_manifest_row(audit: dict[str, str]) -> dict[str, str]:
    source_url = _canonical_evidence_http_url(
        audit["source_url"], field="source_url", context=f"audit {audit['candidate_id']}"
    )
    return {
        "candidate_id": audit["candidate_id"],
        "artifact_id": "provisional-metadata",
        "artifact_role": "metadata-only",
        "source_url": source_url,
        "evidence_version": audit["evidence_version"],
        "evidence_retrieved_on": NORMALIZATION_DATE,
        "access_status": "abstract_only",
        "evidence_archive_url": "NR",
        "evidence_sha256": "NR",
        "local_filename": "NR",
        "redistribution_status": "metadata-only",
        "retrieval_notes": _limited_retrieval_notes(audit["limitation_note"]),
    }


def normalize(
    *,
    candidates_path: Path,
    audits_dir: Path,
    v7_manifest_path: Path,
    source_archive: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    candidates = _candidate_rows(candidates_path)
    candidate_ids = {row["candidate_id"] for row in candidates}
    audits = _audit_rows(audits_dir, candidate_ids)
    v7 = _v7_rows(v7_manifest_path, candidate_ids)

    manifest_rows: list[dict[str, str]] = []
    queue_rows: list[dict[str, str]] = []
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"].encode("utf-8")):
        candidate_id = candidate["candidate_id"]
        audit = audits[candidate_id]
        v7_row = v7.get(candidate_id)
        forced_replacement = audit["access_status"].casefold() in REPLACEMENT_ACTIONS
        if (
            v7_row is not None
            and not forced_replacement
            and _declared_bytes_are_trustworthy(v7_row, source_archive)
        ):
            manifest_rows.append(dict(v7_row))
            continue

        manifest_rows.append(_provisional_manifest_row(audit))
        action = _action_for(audit["access_status"])
        queue_rows.append(
            {
                "candidate_id": candidate_id,
                "title": candidate["title"],
                "source_url": audit["source_url"],
                "raw_access_status": audit["access_status"],
                "action": action,
                "priority": _priority_for(candidate, action),
                "limitation_note": audit["limitation_note"],
            }
        )

    manifest_rows.sort(
        key=lambda row: (row["candidate_id"].encode("utf-8"), row["artifact_id"].encode("utf-8"))
    )
    queue_rows.sort(
        key=lambda row: (
            PRIORITY_ORDER[row["priority"]],
            row["candidate_id"].encode("utf-8"),
        )
    )
    return manifest_rows, queue_rows


def _validate_outputs(
    manifest_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
    *,
    candidate_ids: set[str],
    source_archive: Path,
) -> bytes:
    if len(manifest_rows) != CANDIDATE_COUNT:
        raise NormalizationError("output manifest must contain exactly 202 rows")
    manifest_ids = [row["candidate_id"] for row in manifest_rows]
    if set(manifest_ids) != candidate_ids or len(set(manifest_ids)) != CANDIDATE_COUNT:
        raise NormalizationError("output manifest candidate coverage is not exact")
    manifest_bytes = _canonical_csv_bytes(EVIDENCE_PACKET_HEADER, manifest_rows)
    try:
        parse_evidence_packet_manifest(
            manifest_bytes,
            allowed_candidate_ids=candidate_ids,
            source_archive=source_archive,
        )
    except SnapshotError as exc:
        raise NormalizationError(f"output manifest rejected by parser: {exc}") from exc

    metadata_only_ids = {
        row["candidate_id"] for row in manifest_rows if row["evidence_sha256"] == "NR"
    }
    queue_ids = [row["candidate_id"] for row in queue_rows]
    if set(queue_ids) != metadata_only_ids or len(queue_ids) != len(set(queue_ids)):
        raise NormalizationError(
            "acquisition queue must cover every provisional metadata-only candidate exactly once"
        )
    if any(row["action"] not in ACTION_ORDER for row in queue_rows):
        raise NormalizationError("acquisition queue contains an unknown action")
    if any(row["priority"] not in PRIORITY_ORDER for row in queue_rows):
        raise NormalizationError("acquisition queue contains an unknown priority")
    if queue_rows != sorted(
        queue_rows,
        key=lambda row: (
            PRIORITY_ORDER[row["priority"]],
            row["candidate_id"].encode("utf-8"),
        ),
    ):
        raise NormalizationError("acquisition queue ordering is not deterministic")
    return manifest_bytes


def _report(manifest_rows: list[dict[str, str]], queue_rows: list[dict[str, str]]) -> str:
    trusted_count = sum(row["evidence_sha256"] != "NR" for row in manifest_rows)
    action_counts = Counter(row["action"] for row in queue_rows)
    lines = [
        "# Main Evidence Normalization",
        "",
        "## Coverage",
        "",
        f"- Frozen candidate coverage: {len(manifest_rows)}/{CANDIDATE_COUNT}.",
        f"- Trusted byte-backed rows: {trusted_count}.",
        f"- Provisional metadata-only rows: {len(manifest_rows) - trusted_count}.",
        "",
        "## Acquisition Queue",
        "",
        "Exact CSV header:",
        "",
        "```text",
        ",".join(ACQUISITION_QUEUE_HEADER),
        "```",
        "",
        "| Action | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {action} | {action_counts[action]} |" for action in ACTION_ORDER)
    lines.extend(
        [
            "",
            "Queue order is deterministic: high, medium, then low priority; ties use UTF-8 candidate_id order.",
            "High priority applies when title, source type, or source URL contains a generation, course, scenario, or environment term (including track, road, route, world, map, or terrain). Medium priority applies to replacement work or simulation/driving/vehicle/racing/benchmark/navigation evidence. All remaining candidates are low priority.",
            "",
            "## Limitations",
            "",
            "This normalization did not download public links or copy any source-archive bytes. A row is byte-backed only when its existing v7 local file is present beneath source_archive/v7 and its SHA-256 matches the v7 declaration. All other rows are deliberately metadata-only and require the separately listed acquisition action.",
            f"The provisional evidence_retrieved_on value ({NORMALIZATION_DATE}) records the deterministic normalization date, not a public-link download claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize v8 main evidence audits into a manifest and acquisition queue."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--audits-dir", type=Path, required=True)
    parser.add_argument("--v7-manifest", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        manifest_rows, queue_rows = normalize(
            candidates_path=args.candidates,
            audits_dir=args.audits_dir,
            v7_manifest_path=args.v7_manifest,
            source_archive=args.source_archive,
        )
        candidate_ids = {row["candidate_id"] for row in _candidate_rows(args.candidates)}
        manifest_bytes = _validate_outputs(
            manifest_rows,
            queue_rows,
            candidate_ids=candidate_ids,
            source_archive=args.source_archive,
        )
        _write_bytes(args.manifest_output, manifest_bytes)
        _write_bytes(
            args.queue_output,
            _canonical_csv_bytes(ACQUISITION_QUEUE_HEADER, queue_rows),
        )
        _write_bytes(args.report_output, _report(manifest_rows, queue_rows).encode("utf-8"))
    except NormalizationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
