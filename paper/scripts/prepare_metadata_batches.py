from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Sequence

if __package__:
    from .validate_corpus import (
        CANDIDATE_ID_PATTERN,
        DEFAULT_TAXONOMY,
        HEADERS,
        REQUIRED_FIELDS,
    )
else:
    from validate_corpus import (
        CANDIDATE_ID_PATTERN,
        DEFAULT_TAXONOMY,
        HEADERS,
        REQUIRED_FIELDS,
    )


class ManifestError(ValueError):
    pass


MANIFEST_VERSION = "1"
SUPPORTED_BATCH_COUNT = 6
MANIFEST_HEADER = (
    "manifest_version",
    "snapshot_sha256",
    "batch_id",
    "candidate_id",
    "input_sha256",
    "weight",
)
BIBLIOGRAPHIC_FIELDS = frozenset(
    {"title", "authors", "year", "venue", "doi", "url", "source_type"}
)
CANDIDATE_HEADER = HEADERS["candidates.csv"]
CONFLICT_HEADER = HEADERS["conflicts.csv"]
EVIDENCE_HEADER = HEADERS["evidence.csv"]
CANDIDATE_STATUS_VALUES = {
    field: frozenset(DEFAULT_TAXONOMY[field])
    for field in ("screening_status", "metadata_status")
}

CandidateRow = dict[str, str]


def _read_rows(path: Path, header: tuple[str, ...]) -> list[CandidateRow]:
    if not path.is_file():
        raise ManifestError(f"{path}: file is missing")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual = tuple(reader.fieldnames or ())
            if actual != header:
                raise ManifestError(
                    f"{path}: headers {actual!r} != {header!r}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise ManifestError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise ManifestError(
            f"{path}:{reader.line_num}: CSV parse error: {exc}"
        ) from exc
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ManifestError(f"{path}:{row_number}: malformed CSV row")
        if not any(value.strip() for value in row.values()):
            raise ManifestError(f"{path}:{row_number}: row is entirely blank")
    return rows


def _require_value(
    path: Path,
    row_number: int,
    row: CandidateRow,
    field: str,
) -> str:
    value = row[field].strip()
    if not value:
        raise ManifestError(f"{path}:{row_number}: {field} is required")
    return value


def _group_inputs(
    candidates_path: Path, conflicts_path: Path
) -> tuple[dict[str, CandidateRow], dict[str, list[CandidateRow]]]:
    candidate_rows = _read_rows(candidates_path, CANDIDATE_HEADER)
    conflict_rows = _read_rows(conflicts_path, CONFLICT_HEADER)
    if not candidate_rows:
        raise ManifestError("candidates.csv must contain at least one candidate")

    candidates: dict[str, CandidateRow] = {}
    candidates_by_cite_key: defaultdict[str, list[str]] = defaultdict(list)
    for row_number, row in enumerate(candidate_rows, start=2):
        required = {
            field: _require_value(candidates_path, row_number, row, field)
            for field in REQUIRED_FIELDS["candidates.csv"]
        }
        candidate_id = required["candidate_id"]
        if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
            raise ManifestError(
                f"{candidates_path}:{row_number}: "
                f"candidate_id={candidate_id!r} must be C followed by "
                "at least four digits"
            )
        for field, allowed in CANDIDATE_STATUS_VALUES.items():
            value = required[field]
            if value not in allowed:
                raise ManifestError(
                    f"{candidates_path}:{row_number}: {field}={value!r} "
                    f"is invalid; expected one of {sorted(allowed)}"
                )
        if candidate_id in candidates:
            raise ManifestError(
                f"{candidates_path}:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        candidates[candidate_id] = row
        cite_key = row["cite_key"].strip()
        if cite_key:
            candidates_by_cite_key[cite_key].append(candidate_id)

    grouped_conflicts: defaultdict[str, list[CandidateRow]] = defaultdict(list)
    conflict_ids: set[str] = set()
    for row_number, row in enumerate(conflict_rows, start=2):
        required = {
            field: _require_value(conflicts_path, row_number, row, field)
            for field in REQUIRED_FIELDS["conflicts.csv"]
        }
        conflict_id = required["conflict_id"]
        if conflict_id in conflict_ids:
            raise ManifestError(
                f"{conflicts_path}:{row_number}: duplicate conflict_id "
                f"{conflict_id!r}"
            )
        conflict_ids.add(conflict_id)

        record_type = required["record_type"]
        record_key = required["record_key"]
        if record_type == "candidate":
            if record_key not in candidates:
                raise ManifestError(
                    f"{conflicts_path}:{row_number}: candidate conflict "
                    f"record_key={record_key!r} is orphaned"
                )
            candidate_id = record_key
            target_filename = "candidates.csv"
            target_header = CANDIDATE_HEADER
        elif record_type == "evidence":
            matching_ids = candidates_by_cite_key.get(record_key, [])
            if len(matching_ids) != 1:
                raise ManifestError(
                    f"{conflicts_path}:{row_number}: evidence conflict "
                    f"record_key={record_key!r} does not identify one candidate"
                )
            candidate_id = matching_ids[0]
            target_filename = "evidence.csv"
            target_header = EVIDENCE_HEADER
        else:
            raise ManifestError(
                f"{conflicts_path}:{row_number}: unsupported record_type "
                f"{record_type!r}"
            )

        field = required["field"]
        if field not in target_header:
            raise ManifestError(
                f"{conflicts_path}:{row_number}: {record_type} "
                f"field={field!r} is not a column in {target_filename}"
            )
        if row["resolution"].strip():
            for required_field in ("resolver", "resolution_evidence"):
                _require_value(
                    conflicts_path,
                    row_number,
                    row,
                    required_field,
                )
        grouped_conflicts[candidate_id].append(row)
    return candidates, dict(grouped_conflicts)


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _input_sha256(
    candidate: CandidateRow, conflicts: list[CandidateRow]
) -> str:
    conflict_values = sorted(
        ([row[field] for field in CONFLICT_HEADER] for row in conflicts),
        key=tuple,
    )
    return _canonical_sha256(
        {
            "candidate": [candidate[field] for field in CANDIDATE_HEADER],
            "conflicts": conflict_values,
        }
    )


def _candidate_weight(conflicts: list[CandidateRow]) -> int:
    unresolved = sum(
        row["record_type"].strip() == "candidate"
        and row["field"].strip() in BIBLIOGRAPHIC_FIELDS
        and not row["resolution"].strip()
        for row in conflicts
    )
    return 1 + unresolved


def build_manifest(
    candidates_path: Path,
    conflicts_path: Path,
    batch_count: int = SUPPORTED_BATCH_COUNT,
) -> list[dict[str, str]]:
    """Build canonical manifest rows without modifying either input file.

    An empty candidate corpus is rejected because a header-only manifest cannot
    carry its required snapshot hash. Fewer than six candidates are supported;
    unused logical batches simply have no rows.
    """
    if batch_count != SUPPORTED_BATCH_COUNT:
        raise ManifestError(
            f"metadata verification supports exactly 6 batches, got {batch_count}"
        )
    candidates, conflicts_by_candidate = _group_inputs(
        Path(candidates_path), Path(conflicts_path)
    )
    records = [
        {
            "candidate_id": candidate_id,
            "input_sha256": _input_sha256(
                candidate, conflicts_by_candidate.get(candidate_id, [])
            ),
            "weight": _candidate_weight(
                conflicts_by_candidate.get(candidate_id, [])
            ),
        }
        for candidate_id, candidate in candidates.items()
    ]
    snapshot_sha256 = _canonical_sha256(
        {
            "manifest_version": MANIFEST_VERSION,
            "batch_count": batch_count,
            "candidates": [
                [record["candidate_id"], record["input_sha256"], record["weight"]]
                for record in sorted(
                    records, key=lambda record: record["candidate_id"]
                )
            ],
        }
    )

    capacity = math.ceil(len(records) / batch_count)
    batches = [
        {
            "batch_id": f"metadata-{number:02d}",
            "total_weight": 0,
            "records": [],
        }
        for number in range(1, batch_count + 1)
    ]
    ordered_records = sorted(
        records,
        key=lambda record: (
            -record["weight"],
            hashlib.sha256(record["candidate_id"].encode("utf-8")).hexdigest(),
            record["candidate_id"],
        ),
    )
    for record in ordered_records:
        eligible = [
            batch for batch in batches if len(batch["records"]) < capacity
        ]
        selected = min(
            eligible,
            key=lambda batch: (
                batch["total_weight"],
                len(batch["records"]),
                batch["batch_id"],
            ),
        )
        selected["records"].append(record)
        selected["total_weight"] += record["weight"]

    manifest = []
    for batch in batches:
        for record in sorted(
            batch["records"], key=lambda item: item["candidate_id"]
        ):
            manifest.append(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "snapshot_sha256": snapshot_sha256,
                    "batch_id": batch["batch_id"],
                    "candidate_id": record["candidate_id"],
                    "input_sha256": record["input_sha256"],
                    "weight": str(record["weight"]),
                }
            )
    return manifest


def validate_manifest_inputs(
    manifest_path: Path,
    candidates_path: Path,
    conflicts_path: Path,
) -> None:
    """Reject a manifest that is not the canonical build of its current inputs."""
    manifest_path = Path(manifest_path)
    actual = _read_rows(manifest_path, MANIFEST_HEADER)
    expected = build_manifest(Path(candidates_path), Path(conflicts_path))

    actual_ids = [row["candidate_id"] for row in actual]
    seen: set[str] = set()
    for row_number, candidate_id in enumerate(actual_ids, start=2):
        if candidate_id in seen:
            raise ManifestError(
                f"{manifest_path}:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        seen.add(candidate_id)

    expected_ids = [row["candidate_id"] for row in expected]
    expected_id_set = set(expected_ids)
    actual_id_set = set(actual_ids)
    if actual_id_set != expected_id_set:
        missing = sorted(expected_id_set - actual_id_set)
        extra = sorted(actual_id_set - expected_id_set)
        raise ManifestError(
            f"{manifest_path}: candidate_id mismatch; "
            f"missing={missing}, extra={extra}"
        )

    expected_by_id = {row["candidate_id"]: row for row in expected}
    for row in actual:
        candidate_id = row["candidate_id"]
        expected_row = expected_by_id[candidate_id]
        for field in (
            "manifest_version",
            "input_sha256",
            "weight",
            "batch_id",
            "snapshot_sha256",
        ):
            if row[field] != expected_row[field]:
                raise ManifestError(
                    f"{manifest_path}: candidate_id={candidate_id!r} "
                    f"{field} mismatch; expected {expected_row[field]!r}, "
                    f"found {row[field]!r}"
                )

    if actual_ids != expected_ids:
        raise ManifestError(f"{manifest_path}: rows are not in canonical order")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not path.parent.is_dir():
        raise ManifestError(f"{path.parent}: output directory is missing")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=MANIFEST_HEADER,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
        temporary_path.replace(path)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic metadata verification batches."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--conflicts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--batches",
        type=int,
        default=SUPPORTED_BATCH_COUNT,
        help="number of batches (only 6 is supported)",
    )
    parser.add_argument(
        "--refreeze",
        action="store_true",
        help="replace an existing manifest after validating current inputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if arguments.batches != SUPPORTED_BATCH_COUNT:
        raise ManifestError(
            "metadata verification supports exactly 6 batches, "
            f"got {arguments.batches}"
        )
    output = arguments.output
    output_resolved = output.resolve()
    for label, input_path in (
        ("candidates", arguments.candidates),
        ("conflicts", arguments.conflicts),
    ):
        if output_resolved == input_path.resolve():
            raise ManifestError(
                f"output path must differ from {label} input: {output}"
            )
    if output.exists() and not arguments.refreeze:
        validate_manifest_inputs(
            output,
            arguments.candidates,
            arguments.conflicts,
        )
        return 0
    rows = build_manifest(
        arguments.candidates,
        arguments.conflicts,
        batch_count=arguments.batches,
    )
    _write_manifest(output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
