from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass
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
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
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


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: file is missing") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise ManifestError(f"{path}: {label} must not be a symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ManifestError(f"{path}: {label} must be a regular file")
    return file_stat


def _require_real_directory(path: Path, label: str) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: directory is missing") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise ManifestError(f"{path}: {label} must not be a symlink")
    if not stat.S_ISDIR(file_stat.st_mode):
        raise ManifestError(f"{path}: {label} must be a real directory")
    return file_stat


def _paths_alias(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return first.resolve(strict=False) == second.resolve(strict=False)


def _reject_input_output_aliases(
    output: Path,
    candidates: Path,
    conflicts: Path,
) -> None:
    for label, input_path in (
        ("candidates", candidates),
        ("conflicts", conflicts),
    ):
        if _paths_alias(output, input_path):
            raise ManifestError(
                f"output path must differ from {label} input; "
                f"output aliases {label}: {output}"
            )


def _validate_manifest_paths(
    manifest_path: Path,
    candidates_path: Path,
    conflicts_path: Path,
) -> None:
    paths = (
        ("manifest", manifest_path),
        ("candidates", candidates_path),
        ("conflicts", conflicts_path),
    )
    identities = {
        label: (
            file_stat.st_dev,
            file_stat.st_ino,
        )
        for label, path in paths
        for file_stat in (_require_regular_file(path, label),)
    }
    for index, (label, path) in enumerate(paths):
        for other_label, other_path in paths[index + 1 :]:
            if (
                identities[label] == identities[other_label]
                or _paths_alias(path, other_path)
            ):
                raise ManifestError(
                    f"{path}: {label} aliases {other_label} "
                    f"at {other_path}"
                )


def _validate_snapshot_inputs(
    manifest_path: Path,
    snapshot_dir: Path,
) -> None:
    _require_real_directory(snapshot_dir, "snapshot directory")
    validate_manifest_inputs(
        manifest_path,
        snapshot_dir / "candidates.csv",
        snapshot_dir / "conflicts.csv",
    )


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
    candidates_path = Path(candidates_path)
    conflicts_path = Path(conflicts_path)
    _validate_manifest_paths(
        manifest_path,
        candidates_path,
        conflicts_path,
    )
    actual = _read_rows(manifest_path, MANIFEST_HEADER)
    expected = build_manifest(candidates_path, conflicts_path)

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


def _stage_manifest(
    path: Path, rows: list[dict[str, str]]
) -> Path:
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
        try:
            existing_stat = path.lstat()
        except FileNotFoundError:
            manifest_mode = 0o644
        else:
            manifest_mode = stat.S_IMODE(existing_stat.st_mode)
        os.chmod(temporary_path, manifest_mode)
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    temporary_path = _stage_manifest(path, rows)
    try:
        try:
            _rename_noreplace(temporary_path, path)
        except FileExistsError as exc:
            raise ManifestError(
                f"{path}: manifest already exists"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_snapshot_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o644)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


FileIdentity = tuple[int, int]


@dataclass(frozen=True)
class ManifestRecovery:
    directory: Path
    inode_backup: Path
    byte_backup: Path
    identity: FileIdentity
    sha256: str


def _path_identity(path: Path) -> FileIdentity | None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    return file_stat.st_dev, file_stat.st_ino


def _path_has_identity(path: Path, identity: FileIdentity) -> bool:
    return _path_identity(path) == identity


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise ManifestError(
            "atomic no-clobber snapshot publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source.absolute()),
        _AT_FDCWD,
        os.fsencode(destination.absolute()),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ENOSYS:
            raise ManifestError(
                "atomic no-clobber snapshot publication is unsupported"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )


def _rename_exchange(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise ManifestError(
            "atomic manifest exchange is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source.absolute()),
        _AT_FDCWD,
        os.fsencode(destination.absolute()),
        _RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ENOSYS:
            raise ManifestError(
                "atomic manifest exchange is unsupported"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )


def _remove_snapshot_directory(path: Path) -> None:
    for filename in ("candidates.csv", "conflicts.csv"):
        (path / filename).unlink(missing_ok=True)
    path.rmdir()


def _quarantine_snapshot_directory(
    path: Path, expected_identity: FileIdentity
) -> None:
    if not _path_has_identity(path, expected_identity):
        return
    quarantine_root = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.rollback.",
            suffix=".tmp",
            dir=path.parent,
        )
    )
    quarantine_path = quarantine_root / "snapshot"
    try:
        _rename_noreplace(path, quarantine_path)
    except BaseException:
        if not _path_has_identity(quarantine_path, expected_identity):
            quarantine_root.rmdir()
            raise
    _remove_snapshot_directory(quarantine_path)
    quarantine_root.rmdir()


def _annotate_exception(original: BaseException, message: str) -> None:
    add_note = getattr(original, "add_note", None)
    if callable(add_note):
        try:
            add_note(message)
            return
        except BaseException:
            pass
    try:
        notes = tuple(getattr(original, "__rollback_notes__", ()))
        setattr(original, "__rollback_notes__", (*notes, message))
    except BaseException:
        pass
    try:
        original.args = (*original.args, message)
    except BaseException:
        pass


def _record_rollback_error(
    original: BaseException,
    action: str,
    rollback_error: BaseException,
) -> None:
    _annotate_exception(
        original,
        f"rollback {action} failed: "
        f"{type(rollback_error).__name__}: {rollback_error}",
    )


def _path_matches_sha256(
    path: Path,
    identity: FileIdentity,
    expected_sha256: str,
) -> bool:
    try:
        file_stat = path.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            return False
        if (file_stat.st_dev, file_stat.st_ino) != identity:
            return False
        payload = path.read_bytes()
    except OSError:
        return False
    return (
        _path_has_identity(path, identity)
        and hashlib.sha256(payload).hexdigest() == expected_sha256
    )


def _manifest_matches_recovery(
    path: Path,
    recovery: ManifestRecovery,
) -> bool:
    return _path_matches_sha256(
        path,
        recovery.identity,
        recovery.sha256,
    )


def _cleanup_manifest_recovery(recovery: ManifestRecovery) -> None:
    recovery.inode_backup.unlink(missing_ok=True)
    recovery.byte_backup.unlink(missing_ok=True)
    recovery.directory.rmdir()


def _create_manifest_backup(manifest_path: Path) -> ManifestRecovery:
    manifest_stat = _require_regular_file(manifest_path, "manifest")
    manifest_identity = manifest_stat.st_dev, manifest_stat.st_ino
    manifest_payload = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    recovery_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{manifest_path.name}.recovery.",
            dir=manifest_path.parent,
        )
    )
    os.chmod(recovery_dir, 0o700)
    byte_backup = recovery_dir / "old-manifest.csv"
    inode_backup = recovery_dir / "old-manifest.inode"
    recovery = ManifestRecovery(
        directory=recovery_dir,
        inode_backup=inode_backup,
        byte_backup=byte_backup,
        identity=manifest_identity,
        sha256=manifest_sha256,
    )
    try:
        _write_snapshot_file(byte_backup, manifest_payload)
        os.chmod(byte_backup, stat.S_IMODE(manifest_stat.st_mode))
        os.link(
            manifest_path,
            inode_backup,
            follow_symlinks=False,
        )
        if (
            not _path_has_identity(inode_backup, manifest_identity)
            or not _manifest_matches_recovery(manifest_path, recovery)
        ):
            raise ManifestError(
                f"{manifest_path}: changed while creating rollback backup"
            )
    except BaseException as original:
        for label, path in (
            ("inode backup cleanup", inode_backup),
            ("byte backup cleanup", byte_backup),
        ):
            try:
                path.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                _record_rollback_error(original, label, cleanup_error)
        try:
            recovery_dir.rmdir()
        except BaseException as cleanup_error:
            _record_rollback_error(
                original, "recovery directory cleanup", cleanup_error
            )
        raise
    return recovery



def _attempt_rollback(
    original: BaseException,
    action: str,
    operation: Callable[[], None],
) -> None:
    try:
        operation()
    except BaseException as rollback_error:
        _record_rollback_error(original, action, rollback_error)


def _exchange_was_restored(
    manifest_path: Path,
    staged_manifest: Path,
    displaced_identity: FileIdentity,
    staged_manifest_identity: FileIdentity,
) -> bool:
    return (
        _path_has_identity(manifest_path, displaced_identity)
        and _path_has_identity(staged_manifest, staged_manifest_identity)
    )


def _exchange_was_restored_safely(
    original: BaseException,
    manifest_path: Path,
    staged_manifest: Path,
    displaced_identity: FileIdentity,
    staged_manifest_identity: FileIdentity,
) -> bool:
    try:
        return _exchange_was_restored(
            manifest_path,
            staged_manifest,
            displaced_identity,
            staged_manifest_identity,
        )
    except BaseException as rollback_error:
        _record_rollback_error(
            original,
            "manifest restoration",
            rollback_error,
        )
        return False


def _restore_manifest_exchange(
    original: BaseException,
    *,
    exchange_attempted: bool,
    manifest_path: Path,
    staged_manifest: Path,
    staged_manifest_identity: FileIdentity,
    staged_manifest_sha256: str,
    recovery: ManifestRecovery,
) -> bool:
    if not exchange_attempted:
        return True
    try:
        manifest_identity = _path_identity(manifest_path)
        displaced_identity = _path_identity(staged_manifest)
    except BaseException as rollback_error:
        _record_rollback_error(
            original,
            "manifest restoration",
            rollback_error,
        )
        return False
    if displaced_identity == staged_manifest_identity:
        try:
            manifest_unchanged = _manifest_matches_recovery(
                manifest_path, recovery
            )
        except BaseException as rollback_error:
            _record_rollback_error(
                original, "manifest restoration", rollback_error
            )
            return False
        if manifest_unchanged:
            return True
        _record_rollback_error(
            original,
            "manifest restoration",
            ManifestError("manifest changed before the exchange completed"),
        )
        return False
    if (
        manifest_identity == staged_manifest_identity
        and displaced_identity is not None
    ):
        if not _path_matches_sha256(
            manifest_path,
            staged_manifest_identity,
            staged_manifest_sha256,
        ):
            _record_rollback_error(
                original,
                "manifest restoration",
                ManifestError(
                    "published manifest content changed during rollback"
                ),
            )
            return False
        try:
            _rename_exchange(staged_manifest, manifest_path)
        except BaseException as rollback_error:
            if _exchange_was_restored_safely(
                original,
                manifest_path,
                staged_manifest,
                displaced_identity,
                staged_manifest_identity,
            ):
                return True
            _record_rollback_error(
                original,
                "manifest restoration",
                rollback_error,
            )
            return False
        if _exchange_was_restored_safely(
            original,
            manifest_path,
            staged_manifest,
            displaced_identity,
            staged_manifest_identity,
        ):
            return True
    _record_rollback_error(
        original,
        "manifest restoration",
        ManifestError("manifest ownership changed during rollback"),
    )
    return False


def _move_to_recovery(source: Path, destination: Path) -> None:
    source_identity = _path_identity(source)
    if source_identity is None:
        return
    try:
        _rename_noreplace(source, destination)
    except BaseException:
        if not _path_has_identity(destination, source_identity):
            raise


def _retain_recovery_state(
    original: BaseException,
    recovery: ManifestRecovery,
    *,
    manifest_path: Path,
    snapshot_dir: Path,
    snapshot_identity: FileIdentity | None,
    staged_manifest: Path | None,
) -> None:
    def retain_snapshot() -> None:
        if (
            snapshot_identity is not None
            and _path_has_identity(snapshot_dir, snapshot_identity)
        ):
            _move_to_recovery(
                snapshot_dir, recovery.directory / "snapshot"
            )

    _attempt_rollback(original, "snapshot recovery", retain_snapshot)

    def retain_swapped_manifest() -> None:
        if staged_manifest is not None and os.path.lexists(staged_manifest):
            _move_to_recovery(
                staged_manifest,
                recovery.directory / "swapped-manifest.csv",
            )

    _attempt_rollback(
        original,
        "swapped manifest recovery",
        retain_swapped_manifest,
    )

    def write_recovery_instructions() -> None:
        instructions = (
            f"Canonical manifest: {manifest_path}\n"
            "Pre-refreeze bytes: old-manifest.csv\n"
            "Pre-refreeze inode link: old-manifest.inode\n"
            "Published snapshot, when retained: snapshot/\n"
            "Swapped manifest, when retained: swapped-manifest.csv\n"
        ).encode("utf-8")
        _write_snapshot_file(
            recovery.directory / "RECOVERY.txt",
            instructions,
        )

    _attempt_rollback(
        original,
        "recovery instructions",
        write_recovery_instructions,
    )
    _annotate_exception(
        original,
        f"recovery retained at {recovery.directory}",
    )



def _publish_snapshot_set(
    *,
    candidates_path: Path,
    conflicts_path: Path,
    snapshot_dir: Path,
    manifest_path: Path,
    batch_count: int,
    replace_manifest: bool,
) -> None:
    if os.path.lexists(snapshot_dir):
        raise ManifestError(
            f"{snapshot_dir}: snapshot version already exists"
        )
    _require_real_directory(snapshot_dir.parent, "snapshot parent directory")

    if replace_manifest:
        _require_regular_file(manifest_path, "manifest")
    elif os.path.lexists(manifest_path):
        raise ManifestError(
            f"{manifest_path}: initial freeze manifest already exists"
        )

    payloads = {
        "candidates.csv": candidates_path.read_bytes(),
        "conflicts.csv": conflicts_path.read_bytes(),
    }
    staged_snapshot: Path | None = None
    staged_snapshot_identity: FileIdentity | None = None
    staged_manifest: Path | None = None
    staged_manifest_identity: FileIdentity | None = None
    staged_manifest_sha256: str | None = None
    manifest_backup: ManifestRecovery | None = None
    snapshot_publish_attempted = False
    manifest_publish_attempted = False
    manifest_exchange_attempted = False
    try:
        staged_snapshot = Path(
            tempfile.mkdtemp(
                prefix=f".{snapshot_dir.name}.",
                suffix=".tmp",
                dir=snapshot_dir.parent,
            )
        )
        staged_snapshot_identity = _path_identity(staged_snapshot)
        if staged_snapshot_identity is None:
            raise ManifestError("staged snapshot disappeared before publication")
        for filename, payload in payloads.items():
            _write_snapshot_file(staged_snapshot / filename, payload)
        rows = build_manifest(
            staged_snapshot / "candidates.csv",
            staged_snapshot / "conflicts.csv",
            batch_count=batch_count,
        )
        os.chmod(staged_snapshot, 0o755)

        staged_manifest = _stage_manifest(manifest_path, rows)
        staged_manifest_identity = _path_identity(staged_manifest)
        if staged_manifest_identity is None:
            raise ManifestError("staged manifest disappeared before publication")
        staged_manifest_sha256 = hashlib.sha256(
            staged_manifest.read_bytes()
        ).hexdigest()
        if replace_manifest:
            manifest_backup = _create_manifest_backup(manifest_path)

        snapshot_publish_attempted = True
        try:
            _rename_noreplace(staged_snapshot, snapshot_dir)
        except FileExistsError as exc:
            raise ManifestError(
                f"{snapshot_dir}: snapshot version already exists"
            ) from exc
        if not _path_has_identity(snapshot_dir, staged_snapshot_identity):
            raise ManifestError(
                "snapshot publication did not install the staged inode"
            )
        staged_snapshot = None

        manifest_publish_attempted = True
        if replace_manifest:
            if manifest_backup is None:
                raise ManifestError("refreeze recovery journal is missing")
            manifest_exchange_attempted = True
            _rename_exchange(staged_manifest, manifest_path)
            if (
                staged_manifest_sha256 is None
                or not _path_matches_sha256(
                    manifest_path,
                    staged_manifest_identity,
                    staged_manifest_sha256,
                )
                or not _manifest_matches_recovery(
                    staged_manifest, manifest_backup
                )
                or not _path_matches_sha256(
                    manifest_path,
                    staged_manifest_identity,
                    staged_manifest_sha256,
                )
            ):
                raise ManifestError(
                    f"{manifest_path}: manifest changed during refreeze"
                )
        else:
            try:
                _rename_noreplace(staged_manifest, manifest_path)
            except FileExistsError as exc:
                raise ManifestError(
                    f"{manifest_path}: manifest already exists"
                ) from exc
            if not _path_has_identity(
                manifest_path, staged_manifest_identity
            ):
                raise ManifestError(
                    "manifest publication did not install the staged inode"
                )
            staged_manifest = None
    except BaseException as original:
        restoration_complete = True
        if (
            replace_manifest
            and manifest_backup is not None
            and staged_manifest is not None
            and staged_manifest_identity is not None
        ):
            restoration_complete = _restore_manifest_exchange(
                original,
                exchange_attempted=manifest_exchange_attempted,
                manifest_path=manifest_path,
                staged_manifest=staged_manifest,
                staged_manifest_identity=staged_manifest_identity,
                staged_manifest_sha256=(
                    staged_manifest_sha256
                    if staged_manifest_sha256 is not None
                    else ""
                ),
                recovery=manifest_backup,
            )
        elif not replace_manifest:
            def remove_new_manifest() -> None:
                if (
                    manifest_publish_attempted
                    and staged_manifest_identity is not None
                    and _path_has_identity(
                        manifest_path, staged_manifest_identity
                    )
                ):
                    manifest_path.unlink()

            _attempt_rollback(
                original,
                "manifest restoration",
                remove_new_manifest,
            )

        recovery_required = (
            replace_manifest
            and manifest_backup is not None
            and not restoration_complete
        )
        if recovery_required:
            _retain_recovery_state(
                original,
                manifest_backup,
                manifest_path=manifest_path,
                snapshot_dir=snapshot_dir,
                snapshot_identity=staged_snapshot_identity,
                staged_manifest=staged_manifest,
            )
        else:
            def quarantine_snapshot() -> None:
                if (
                    snapshot_publish_attempted
                    and staged_snapshot_identity is not None
                    and _path_has_identity(
                        snapshot_dir, staged_snapshot_identity
                    )
                ):
                    _quarantine_snapshot_directory(
                        snapshot_dir,
                        staged_snapshot_identity,
                    )

            _attempt_rollback(
                original,
                "snapshot quarantine",
                quarantine_snapshot,
            )

            for label, cleanup_path, identity, cleanup in (
                (
                    "staged snapshot cleanup",
                    staged_snapshot,
                    staged_snapshot_identity,
                    _remove_snapshot_directory,
                ),
                (
                    "staged manifest cleanup",
                    staged_manifest,
                    staged_manifest_identity,
                    lambda path: path.unlink(missing_ok=True),
                ),
            ):
                def cleanup_staged_path() -> None:
                    if (
                        cleanup_path is not None
                        and identity is not None
                        and _path_has_identity(cleanup_path, identity)
                    ):
                        cleanup(cleanup_path)

                _attempt_rollback(
                    original,
                    label,
                    cleanup_staged_path,
                )

            if manifest_backup is not None:
                try:
                    _cleanup_manifest_recovery(manifest_backup)
                except BaseException as rollback_error:
                    _record_rollback_error(
                        original,
                        "manifest recovery cleanup",
                        rollback_error,
                    )
                    _annotate_exception(
                        original,
                        f"recovery retained at "
                        f"{manifest_backup.directory}",
                    )
        raise
    else:
        if replace_manifest and staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)
            staged_manifest = None
        if manifest_backup is not None:
            _cleanup_manifest_recovery(manifest_backup)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic metadata verification batches."
    )
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--conflicts", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
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
    output_exists = os.path.lexists(output)
    if arguments.refreeze:
        if arguments.snapshot_dir is None:
            raise ManifestError("--refreeze requires --snapshot-dir")
        try:
            _require_regular_file(output, "manifest")
        except ManifestError as exc:
            raise ManifestError(
                f"{output}: refreeze requires an existing regular "
                "non-symlink manifest"
            ) from exc
    if (
        arguments.snapshot_dir is not None
        and output_exists
        and not arguments.refreeze
    ):
        if (
            arguments.candidates is not None
            or arguments.conflicts is not None
        ):
            raise ManifestError(
                "initial snapshot freeze requires an absent manifest; "
                "omit direct inputs for ordinary snapshot validation"
            )
        _validate_snapshot_inputs(
            output,
            arguments.snapshot_dir,
        )
        return 0
    if arguments.candidates is None or arguments.conflicts is None:
        raise ManifestError(
            "--candidates and --conflicts are required to create or refreeze"
        )
    _require_regular_file(
        arguments.candidates,
        "candidates input",
    )
    _require_regular_file(
        arguments.conflicts,
        "conflicts input",
    )
    if _paths_alias(arguments.candidates, arguments.conflicts):
        raise ManifestError(
            f"{arguments.candidates}: candidates input aliases "
            f"conflicts input at {arguments.conflicts}"
        )
    _reject_input_output_aliases(
        output,
        arguments.candidates,
        arguments.conflicts,
    )
    if arguments.snapshot_dir is not None and (
        not output_exists or arguments.refreeze
    ):
        _publish_snapshot_set(
            candidates_path=arguments.candidates,
            conflicts_path=arguments.conflicts,
            snapshot_dir=arguments.snapshot_dir,
            manifest_path=output,
            batch_count=arguments.batches,
            replace_manifest=arguments.refreeze,
        )
        return 0
    if output_exists and not arguments.refreeze:
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
