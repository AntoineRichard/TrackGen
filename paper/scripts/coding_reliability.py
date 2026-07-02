from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import math
import os
import secrets
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CORE_FIELDS = (
    "survey_evidence_tier",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "code_status",
    "asset_status",
)
_SURVEY_EVIDENCE_TIERS = frozenset({"core", "supporting", "contextual"})


def _required_text(
    row: dict[str, str],
    field: str,
    *,
    row_number: int,
    source: str,
) -> str:
    if field not in row:
        raise ValueError(f"{source} row {row_number}: missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} row {row_number}: {field} must be nonblank")
    return value


def _required_cite_key(
    row: dict[str, str],
    *,
    row_number: int,
    source: str,
) -> str:
    cite_key = _required_text(
        row,
        "cite_key",
        row_number=row_number,
        source=source,
    )
    if cite_key != cite_key.strip():
        raise ValueError(
            f"{source} row {row_number}: cite_key must not contain "
            "surrounding whitespace"
        )
    return cite_key


def select_reliability_sample(
    evidence: list[dict[str, str]],
    fraction: float = 0.20,
) -> list[dict[str, str]]:
    try:
        valid_fraction = math.isfinite(fraction) and 0 < fraction <= 1
    except (TypeError, ValueError):
        valid_fraction = False
    if not valid_fraction:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    by_domain: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    seen_keys: set[str] = set()
    for row_number, row in enumerate(evidence, start=1):
        cite_key = _required_cite_key(
            row,
            row_number=row_number,
            source="evidence",
        )
        if cite_key in seen_keys:
            raise ValueError(
                f"evidence row {row_number}: duplicate cite_key {cite_key!r}"
            )
        seen_keys.add(cite_key)

        domain = _required_text(
            row,
            "domain",
            row_number=row_number,
            source="evidence",
        )
        labels = [label.strip() for label in domain.split(";") if label.strip()]
        if not labels:
            raise ValueError(f"evidence row {row_number}: domain must be nonblank")
        by_domain[labels[0]].append(row)

    selected: dict[str, dict[str, str]] = {}
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        count = min(len(rows), max(2, math.ceil(fraction * len(rows))))
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(row["cite_key"].encode("utf-8")).hexdigest(),
                row["cite_key"],
            ),
        )
        for row in ranked[:count]:
            selected.setdefault(row["cite_key"], row)
    return [selected[cite_key] for cite_key in sorted(selected)]


SELECTION_FIELDS = (
    "cite_key",
    "first_domain",
    "rank_sha256",
    "evidence_sha256",
)
PACKET_FIELDS = (
    "candidate_id",
    "cite_key",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
TEMPLATE_FIELDS = ("cite_key", *CORE_FIELDS)


def _selection_rows(
    selected: list[dict[str, str]],
    evidence_sha256: str,
) -> list[dict[str, str]]:
    return [
        {
            "cite_key": row["cite_key"],
            "first_domain": next(
                label.strip()
                for label in row["domain"].split(";")
                if label.strip()
            ),
            "rank_sha256": hashlib.sha256(
                row["cite_key"].encode("utf-8")
            ).hexdigest(),
            "evidence_sha256": evidence_sha256,
        }
        for row in selected
    ]


def _canonical(value: str) -> str:
    return ";".join(
        sorted(label.strip() for label in value.split(";") if label.strip())
    )


def _validate_survey_evidence_tier(
    value: str,
    *,
    row_number: int,
    source: str,
) -> None:
    if value not in _SURVEY_EVIDENCE_TIERS:
        raise ValueError(
            f"{source} row {row_number}: survey_evidence_tier must be exactly "
            "one of 'core', 'supporting', or 'contextual'"
        )


def cohens_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must have equal nonzero length")
    sample_size = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / sample_size
    left_counts = Counter(left)
    right_counts = Counter(right)
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[value] / sample_size) * (right_counts[value] / sample_size)
        for value in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _index_codings(
    rows: list[dict[str, str]],
    *,
    source: str,
) -> dict[str, dict[str, str]]:
    if not rows:
        raise ValueError(f"{source} coding sample must be nonempty")

    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        cite_key = _required_cite_key(
            row,
            row_number=row_number,
            source=source,
        )
        if cite_key in indexed:
            raise ValueError(
                f"{source} row {row_number}: duplicate cite_key {cite_key!r}"
            )
        for field in CORE_FIELDS:
            if field not in row:
                raise ValueError(
                    f"{source} row {row_number}: missing required field {field!r}"
                )
            value = row[field]
            if not isinstance(value, str):
                raise ValueError(
                    f"{source} row {row_number}: field {field!r} must be text"
                )
            if not _canonical(value):
                raise ValueError(
                    f"{source} row {row_number}: field {field!r} must have "
                    "a nonempty canonical value"
                )
            if field == "survey_evidence_tier":
                _validate_survey_evidence_tier(
                    value,
                    row_number=row_number,
                    source=source,
                )
        indexed[cite_key] = row
    return indexed


def compare_codings(
    primary: list[dict[str, str]],
    reliability: list[dict[str, str]],
) -> list[dict[str, str]]:
    left = _index_codings(primary, source="primary")
    right = _index_codings(reliability, source="reliability")
    if set(left) != set(right):
        raise ValueError(
            "coding samples differ: "
            f"primary={sorted(left)}, reliability={sorted(right)}"
        )

    keys = sorted(left)
    summary: list[dict[str, str]] = []
    for field in CORE_FIELDS:
        values_left = [_canonical(left[key][field]) for key in keys]
        values_right = [_canonical(right[key][field]) for key in keys]
        agreement = (
            sum(a == b for a, b in zip(values_left, values_right)) / len(keys)
        )
        left_categories = set(values_left)
        right_categories = set(values_right)
        kappa = (
            f"{cohens_kappa(values_left, values_right):.6f}"
            if len(left_categories) >= 2 and len(right_categories) >= 2
            else "NR"
        )
        summary.append(
            {
                "field": field,
                "n": str(len(keys)),
                "agreement": f"{agreement:.6f}",
                "kappa": kappa,
                "passes": str(agreement >= 0.80).lower(),
            }
        )
    return summary


SUMMARY_FIELDS = ("field", "n", "agreement", "kappa", "passes")


@dataclass(frozen=True)
class _InputSnapshot:
    path: Path
    parent_device: int
    parent_inode: int
    device: int
    inode: int
    sha256: str


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_fd(path: Path, *, create: bool = False) -> tuple[int, Path]:
    absolute_path = _absolute_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open("/", flags)
    try:
        for component in absolute_path.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except OSError as exc:
        os.close(directory_fd)
        raise ValueError(
            f"{path}: path components must be real directories: {exc}"
        ) from exc
    return directory_fd, absolute_path


def _read_regular_file(path: Path) -> tuple[bytes, _InputSnapshot]:
    absolute_path = _absolute_path(path)
    if not absolute_path.name:
        raise ValueError(f"{path}: CSV file is missing")

    parent_fd, _ = _open_directory_fd(absolute_path.parent)
    file_fd: int | None = None
    try:
        parent_status = os.fstat(parent_fd)
        try:
            file_fd = os.open(
                absolute_path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ValueError(
                f"{path}: CSV file is missing or unsafe: {exc}"
            ) from exc

        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{path}: CSV input must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise ValueError(f"{path}: input changed while it was being read")
        raw_bytes = b"".join(chunks)
        return raw_bytes, _InputSnapshot(
            path=absolute_path,
            parent_device=parent_status.st_dev,
            parent_inode=parent_status.st_ino,
            device=after.st_dev,
            inode=after.st_ino,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _read_csv_with_identity(
    path: Path,
) -> tuple[tuple[str, ...], list[dict[str, str]], _InputSnapshot]:
    raw_bytes, snapshot = _read_regular_file(path)
    reader: csv.DictReader | None = None
    try:
        with io.StringIO(
            raw_bytes.decode("utf-8"),
            newline="",
        ) as handle:
            reader = csv.DictReader(handle, strict=True)
            header = tuple(reader.fieldnames or ())
            if not header:
                raise ValueError(f"{path}: CSV header is missing")
            if any(field is None or not field.strip() for field in header):
                raise ValueError(f"{path}: CSV header contains a blank column name")
            duplicate_fields = sorted(
                field
                for field, count in Counter(header).items()
                if count > 1
            )
            if duplicate_fields:
                raise ValueError(
                    f"{path}: CSV header contains duplicate columns "
                    f"{duplicate_fields}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        line_number = getattr(reader, "line_num", "?")
        raise ValueError(
            f"{path}:{line_number}: CSV parse error: {exc}"
        ) from exc

    validated_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{row_number}: malformed CSV row")
        validated_rows.append(row)
    if not validated_rows:
        raise ValueError(f"{path}: CSV must contain at least one data row")
    return header, validated_rows, snapshot


def _require_columns(
    path: Path,
    header: tuple[str, ...],
    required: tuple[str, ...],
) -> None:
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(f"{path}: required columns are missing: {missing}")


def _require_exact_header(
    path: Path,
    header: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    if header != expected:
        raise ValueError(
            f"{path}: CSV header must be exactly {list(expected)}, "
            f"got {list(header)}"
        )


def _index_selection(
    path: Path,
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    snapshot_sha256: str | None = None
    for row_number, row in enumerate(rows, start=2):
        cite_key = _required_cite_key(
            row,
            row_number=row_number,
            source=str(path),
        )
        if cite_key in indexed:
            raise ValueError(
                f"{path}:{row_number}: duplicate cite_key {cite_key!r}"
            )
        first_domain = _required_text(
            row,
            "first_domain",
            row_number=row_number,
            source=str(path),
        )
        if first_domain != first_domain.strip() or ";" in first_domain:
            raise ValueError(
                f"{path}:{row_number}: first_domain must be one normalized label"
            )
        rank = _required_text(
            row,
            "rank_sha256",
            row_number=row_number,
            source=str(path),
        )
        expected_rank = hashlib.sha256(cite_key.encode("utf-8")).hexdigest()
        if rank != expected_rank:
            raise ValueError(
                f"{path}:{row_number}: rank_sha256 does not match cite_key"
            )
        evidence_sha256 = _required_text(
            row,
            "evidence_sha256",
            row_number=row_number,
            source=str(path),
        )
        if len(evidence_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in evidence_sha256
        ):
            raise ValueError(
                f"{path}:{row_number}: evidence_sha256 must be lowercase SHA-256"
            )
        if snapshot_sha256 is None:
            snapshot_sha256 = evidence_sha256
        elif evidence_sha256 != snapshot_sha256:
            raise ValueError(
                f"{path}:{row_number}: evidence_sha256 must match every row"
            )
        indexed[cite_key] = row
    return indexed


def _prepare_blind_rows(
    selection_path: Path,
    candidates_path: Path,
    *,
    include_snapshots: bool = False,
) -> (
    tuple[list[dict[str, str]], list[dict[str, str]]]
    | tuple[
        list[dict[str, str]],
        list[dict[str, str]],
        tuple[_InputSnapshot, ...],
    ]
):
    selection_header, selection_rows, selection_snapshot = (
        _read_csv_with_identity(selection_path)
    )
    _require_exact_header(selection_path, selection_header, SELECTION_FIELDS)
    selection = _index_selection(selection_path, selection_rows)

    candidates_header, candidates, candidates_snapshot = (
        _read_csv_with_identity(candidates_path)
    )
    _require_columns(candidates_path, candidates_header, PACKET_FIELDS)
    candidates_by_key: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(candidates, start=2):
        cite_key = row["cite_key"]
        if not cite_key:
            continue
        if cite_key != cite_key.strip():
            raise ValueError(
                f"{candidates_path}:{row_number}: cite_key must not contain "
                "surrounding whitespace"
            )
        if cite_key in candidates_by_key:
            raise ValueError(
                f"{candidates_path}:{row_number}: duplicate cite_key "
                f"{cite_key!r}"
            )
        candidates_by_key[cite_key] = row

    selected_candidates: list[dict[str, str]] = []
    for cite_key in sorted(selection):
        candidate = candidates_by_key.get(cite_key)
        if candidate is None:
            raise ValueError(
                f"{candidates_path}: selected cite_key {cite_key!r} is missing"
            )
        selected_candidates.append(candidate)

    packet = [
        {field: row[field] for field in PACKET_FIELDS}
        for row in selected_candidates
    ]
    template = [
        {"cite_key": row["cite_key"], **dict.fromkeys(CORE_FIELDS, "")}
        for row in selected_candidates
    ]
    if include_snapshots:
        return packet, template, (
            selection_snapshot,
            candidates_snapshot,
        )
    return packet, template


def _materialize_primary_rows(
    selection_path: Path,
    evidence_path: Path,
    *,
    include_snapshots: bool = False,
) -> (
    tuple[tuple[str, ...], list[dict[str, str]]]
    | tuple[
        tuple[str, ...],
        list[dict[str, str]],
        tuple[_InputSnapshot, ...],
    ]
):
    selection_header, selection_rows, selection_snapshot = (
        _read_csv_with_identity(selection_path)
    )
    _require_exact_header(selection_path, selection_header, SELECTION_FIELDS)
    selection = _index_selection(selection_path, selection_rows)

    evidence_header, evidence, evidence_snapshot = _read_csv_with_identity(
        evidence_path,
    )
    _require_columns(evidence_path, evidence_header, ("cite_key", "domain"))
    selected_evidence = select_reliability_sample(evidence)
    expected_selection = _selection_rows(
        selected_evidence,
        evidence_snapshot.sha256,
    )
    actual_selection = [
        selection[cite_key] for cite_key in sorted(selection)
    ]
    if actual_selection != expected_selection:
        raise ValueError(
            f"{selection_path}: selection does not exactly match the "
            f"deterministic sample for {evidence_path}"
        )
    if include_snapshots:
        return evidence_header, selected_evidence, (
            selection_snapshot,
            evidence_snapshot,
        )
    return evidence_header, selected_evidence


@dataclass(frozen=True)
class _DirectoryAnchor:
    path: Path
    fd: int
    device: int
    inode: int


@dataclass
class _OutputState:
    path: Path
    directory: _DirectoryAnchor
    name: str
    rows: list[dict[str, str]]
    header: tuple[str, ...]
    original_bytes: bytes | None
    original_mode: int | None
    original_identity: tuple[int, int] | None
    original_fd: int | None
    staged_name: str | None = None
    staged_created: bool = False
    staged_fd: int | None = None
    staged_identity: tuple[int, int] | None = None
    staged_path_identity: tuple[int, int] | None = None
    backup_name: str | None = None
    backup_identity: tuple[int, int] | None = None
    installed: bool = False


def _identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _directory_path_at(directory: _DirectoryAnchor) -> Path:
    path = Path(os.readlink(f"/proc/self/fd/{directory.fd}"))
    return path if path.is_absolute() else Path.cwd() / path


def _exception_diagnostic(error: BaseException) -> str:
    parts = [f"{type(error).__name__}: {error}"]
    for note in getattr(error, "__notes__", ()):
        if note not in parts:
            parts.append(note)
    for detail in getattr(error, "_trackgen_recovery_details", ()):
        if detail not in parts and detail not in parts[0]:
            parts.append(detail)
    return "\n".join(parts)


def _attach_exception_detail(error: BaseException, detail: str) -> None:
    try:
        if hasattr(error, "add_note"):
            error.add_note(detail)
            return
    except BaseException:
        pass

    try:
        details = list(getattr(error, "_trackgen_recovery_details", ()))
        details.append(detail)
        error._trackgen_recovery_details = tuple(details)
    except BaseException:
        details = [detail]

    try:
        original_args = getattr(error, "_trackgen_original_args", None)
        if original_args is None:
            original_args = error.args
            error._trackgen_original_args = original_args
        rendered_details = "\n".join(details)
        if isinstance(error, OSError) and error.errno is not None:
            original_strerror = getattr(
                error,
                "_trackgen_original_strerror",
                error.strerror,
            )
            error._trackgen_original_strerror = original_strerror
            error.strerror = f"{original_strerror}\n{rendered_details}"
        elif original_args and isinstance(original_args[0], str):
            error.args = (
                f"{original_args[0]}\n{rendered_details}",
                *original_args[1:],
            )
        else:
            error.args = (*original_args, f"recovery details: {rendered_details}")
    except BaseException:
        pass


class _CombinedCleanupError(RuntimeError):
    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        lines = [
            f"cleanup encountered {len(errors)} error(s):",
            *(
                f"[{index}] {_exception_diagnostic(error)}"
                for index, error in enumerate(errors, start=1)
            ),
        ]
        super().__init__("\n".join(lines))


def _entry_status(
    directory: _DirectoryAnchor,
    name: str,
) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_regular_entry_fd(
    directory: _DirectoryAnchor,
    name: str,
    expected: os.stat_result,
    *,
    description: str,
) -> int:
    file_fd = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory.fd,
    )
    try:
        status = os.fstat(file_fd)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"{description} must be a regular file")
        if _identity(status) != _identity(expected):
            raise ValueError(f"{description} changed during validation")
    except BaseException:
        os.close(file_fd)
        raise
    return file_fd


def _read_regular_entry(
    directory: _DirectoryAnchor,
    name: str,
    expected: os.stat_result,
    *,
    description: str,
) -> bytes:
    file_fd = _open_regular_entry_fd(
        directory,
        name,
        expected,
        description=description,
    )
    try:
        before = os.fstat(file_fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise ValueError(f"{description} changed during validation")
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _open_output_state(
    path: Path,
    rows: list[dict[str, str]],
    header: tuple[str, ...],
) -> _OutputState:
    absolute_path = _absolute_path(path)
    if not absolute_path.name:
        raise ValueError(f"{path}: output must name a regular file")

    directory_fd, parent_path = _open_directory_fd(
        absolute_path.parent,
        create=True,
    )
    parent_status = os.fstat(directory_fd)
    directory = _DirectoryAnchor(
        path=parent_path,
        fd=directory_fd,
        device=parent_status.st_dev,
        inode=parent_status.st_ino,
    )
    original_fd: int | None = None
    try:
        current = _entry_status(directory, absolute_path.name)
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise ValueError(
                f"{path}: existing output must be a regular file"
            )
        if current is None:
            original_bytes = None
        else:
            original_bytes = _read_regular_entry(
                directory,
                absolute_path.name,
                current,
                description=f"{path}: existing output",
            )
            original_fd = _open_regular_entry_fd(
                directory,
                absolute_path.name,
                current,
                description=f"{path}: existing output",
            )
        return _OutputState(
            path=absolute_path,
            directory=directory,
            name=absolute_path.name,
            rows=rows,
            header=header,
            original_bytes=original_bytes,
            original_mode=(
                None if current is None else stat.S_IMODE(current.st_mode)
            ),
            original_identity=(
                None if current is None else _identity(current)
            ),
            original_fd=original_fd,
        )
    except BaseException:
        if original_fd is not None:
            os.close(original_fd)
        os.close(directory_fd)
        raise


def _validate_transaction_aliases(
    states: list[_OutputState],
    input_snapshots: tuple[_InputSnapshot, ...],
) -> None:
    input_identities = {
        (snapshot.device, snapshot.inode): snapshot.path
        for snapshot in input_snapshots
    }
    seen_targets: dict[tuple[int, int, str], Path] = {}
    seen_outputs: dict[tuple[int, int], Path] = {}
    for state in states:
        target = (
            state.directory.device,
            state.directory.inode,
            state.name,
        )
        if target in seen_targets:
            raise ValueError(
                f"{seen_targets[target]} and {state.path} reference "
                "the same output"
            )
        seen_targets[target] = state.path

        if state.original_identity is None:
            continue
        if state.original_identity in input_identities:
            raise ValueError(
                f"{state.path} aliases input "
                f"{input_identities[state.original_identity]}"
            )
        if state.original_identity in seen_outputs:
            raise ValueError(
                f"{seen_outputs[state.original_identity]} and {state.path} "
                "reference the same output"
            )
        seen_outputs[state.original_identity] = state.path


def _revalidate_input(snapshot: _InputSnapshot) -> None:
    try:
        _raw_bytes, current = _read_regular_file(snapshot.path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"input {snapshot.path} changed before publication"
        ) from exc
    if (
        current.parent_device != snapshot.parent_device
        or current.parent_inode != snapshot.parent_inode
        or current.device != snapshot.device
        or current.inode != snapshot.inode
        or current.sha256 != snapshot.sha256
    ):
        raise ValueError(f"input {snapshot.path} changed before publication")


def _validate_independent_captures(
    primary: _InputSnapshot,
    reliability: _InputSnapshot,
) -> None:
    if (primary.device, primary.inode) == (
        reliability.device,
        reliability.inode,
    ):
        raise ValueError(
            "primary and reliability inputs are not independent: both "
            "captures reference the same file"
        )
    _revalidate_input(primary)
    _revalidate_input(reliability)


def _revalidate_output_parent(state: _OutputState) -> None:
    check_fd, _ = _open_directory_fd(state.directory.path)
    try:
        parent_status = os.fstat(check_fd)
        if (
            parent_status.st_dev != state.directory.device
            or parent_status.st_ino != state.directory.inode
        ):
            raise ValueError(
                f"{state.path}: output parent changed before publication"
            )
    finally:
        os.close(check_fd)


def _revalidate_output(state: _OutputState) -> None:
    _revalidate_output_parent(state)
    current = _entry_status(state.directory, state.name)
    if state.original_identity is None:
        if current is not None:
            raise ValueError(
                f"{state.path}: output changed before publication"
            )
        return
    if (
        current is None
        or not stat.S_ISREG(current.st_mode)
        or _identity(current) != state.original_identity
    ):
        raise ValueError(f"{state.path}: output changed before publication")
    current_bytes = _read_regular_entry(
        state.directory,
        state.name,
        current,
        description=f"{state.path}: output",
    )
    if (
        current_bytes != state.original_bytes
        or stat.S_IMODE(current.st_mode) != state.original_mode
    ):
        raise ValueError(f"{state.path}: output changed before publication")


def _revalidate_installed_output(state: _OutputState) -> None:
    _revalidate_output_parent(state)
    current = _entry_status(state.directory, state.name)
    if (
        current is None
        or not stat.S_ISREG(current.st_mode)
        or state.staged_identity is None
        or _identity(current) != state.staged_identity
    ):
        raise ValueError(
            f"{state.path}: installed output changed before commit"
        )


def _unused_entry_name(state: _OutputState, suffix: str) -> str:
    for _attempt in range(100):
        name = f".{state.name}.{secrets.token_hex(8)}.{suffix}"
        if _entry_status(state.directory, name) is None:
            return name
    raise OSError(f"{state.path}: unable to allocate transaction filename")


def _create_staged_file(state: _OutputState) -> None:
    assert state.staged_name is not None
    staged_fd = os.open(
        state.staged_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o600,
        dir_fd=state.directory.fd,
    )
    state.staged_created = True
    state.staged_fd = staged_fd


def _recover_staged_identity_from_fd(state: _OutputState) -> None:
    assert state.staged_fd is not None
    assert state.staged_name is not None
    descriptor_status = os.stat(state.staged_fd)
    staged_status = _entry_status(state.directory, state.staged_name)
    if (
        staged_status is not None
        and stat.S_ISREG(staged_status.st_mode)
        and _identity(staged_status) == _identity(descriptor_status)
    ):
        state.staged_identity = _identity(descriptor_status)
        state.staged_path_identity = state.staged_identity


def _stage_output(state: _OutputState) -> None:
    state.staged_name = _unused_entry_name(state, "tmp")
    state.staged_created = False
    state.staged_fd = None
    try:
        _create_staged_file(state)
        assert state.staged_fd is not None
        staged_status = os.fstat(state.staged_fd)
        state.staged_identity = _identity(staged_status)
        state.staged_path_identity = state.staged_identity
        handle = os.fdopen(
            os.dup(state.staged_fd),
            mode="w",
            encoding="utf-8",
            newline="",
        )
        with handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=state.header,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(state.rows)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), state.original_mode or 0o644)
    except BaseException:
        if state.staged_fd is not None:
            try:
                if (
                    state.staged_created
                    and state.staged_identity is None
                ):
                    _recover_staged_identity_from_fd(state)
            finally:
                os.close(state.staged_fd)
                state.staged_fd = None
        raise


# Linux provides the required no-replace and lossless-exchange semantics.
# There is no path-only fallback that preserves the same race guarantees.
_AT_EMPTY_PATH = 0x1000
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_LIBC = ctypes.CDLL(None, use_errno=True)


def _raise_syscall_error(operation: str, name: str) -> None:
    error_number = ctypes.get_errno() or errno.EIO
    raise OSError(
        error_number,
        f"{operation}: {os.strerror(error_number)}",
        name,
    )


def _link_fd_at(
    source_fd: int,
    directory: _DirectoryAnchor,
    target_name: str,
) -> None:
    linkat = getattr(_LIBC, "linkat", None)
    if linkat is None:
        raise OSError(errno.ENOSYS, "linkat is unavailable", target_name)
    ctypes.set_errno(0)
    result = linkat(
        ctypes.c_int(source_fd),
        ctypes.c_char_p(b""),
        ctypes.c_int(directory.fd),
        ctypes.c_char_p(os.fsencode(target_name)),
        ctypes.c_int(_AT_EMPTY_PATH),
    )
    if result != 0:
        _raise_syscall_error("linkat", target_name)


def _rename_at2(
    directory: _DirectoryAnchor,
    source_name: str,
    target_name: str,
    flags: int,
) -> None:
    renameat2 = getattr(_LIBC, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", target_name)
    ctypes.set_errno(0)
    result = renameat2(
        ctypes.c_int(directory.fd),
        ctypes.c_char_p(os.fsencode(source_name)),
        ctypes.c_int(directory.fd),
        ctypes.c_char_p(os.fsencode(target_name)),
        ctypes.c_uint(flags),
    )
    if result != 0:
        _raise_syscall_error("renameat2", target_name)


def _rename_noreplace_at(
    directory: _DirectoryAnchor,
    source_name: str,
    target_name: str,
) -> None:
    _rename_at2(
        directory,
        source_name,
        target_name,
        _RENAME_NOREPLACE,
    )


def _rename_exchange_at(
    directory: _DirectoryAnchor,
    source_name: str,
    target_name: str,
) -> None:
    _rename_at2(
        directory,
        source_name,
        target_name,
        _RENAME_EXCHANGE,
    )


def _preflight_rename_noreplace(state: _OutputState) -> None:
    assert state.staged_name is not None
    assert state.staged_identity is not None
    before = _entry_identity_at(state.directory, state.staged_name)
    if before != state.staged_identity:
        raise RuntimeError(
            f"{state.path}: staged output changed before "
            "RENAME_NOREPLACE preflight"
        )
    try:
        _rename_noreplace_at(
            state.directory,
            state.staged_name,
            state.staged_name,
        )
    except FileExistsError:
        pass
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            raise RuntimeError(
                f"{state.path}: RENAME_NOREPLACE is unavailable on the "
                "output filesystem"
            ) from exc
        raise
    else:
        raise RuntimeError(
            f"{state.path}: RENAME_NOREPLACE output-filesystem probe did "
            "not report EEXIST"
        )
    if _entry_identity_at(state.directory, state.staged_name) != before:
        raise RuntimeError(
            f"{state.path}: RENAME_NOREPLACE preflight changed the staged output"
        )


def _entry_has_identity(
    state: _OutputState,
    name: str,
    expected_identity: tuple[int, int] | None,
) -> bool:
    status = _entry_status(state.directory, name)
    return (
        status is not None
        and expected_identity is not None
        and _identity(status) == expected_identity
    )


def _entry_identity_at(
    directory: _DirectoryAnchor,
    name: str,
) -> tuple[int, int] | None:
    status = _entry_status(directory, name)
    return None if status is None else _identity(status)


def _exchange_expected(
    state: _OutputState,
    left_name: str,
    right_name: str,
    left_identity: tuple[int, int],
    right_identity: tuple[int, int],
    *,
    description: str,
) -> BaseException | None:
    operation_error: BaseException | None = None
    try:
        _rename_exchange_at(state.directory, left_name, right_name)
    except BaseException as exc:
        operation_error = exc

    left_after = _entry_identity_at(state.directory, left_name)
    right_after = _entry_identity_at(state.directory, right_name)
    if (left_after, right_after) == (right_identity, left_identity):
        return operation_error
    if (left_after, right_after) == (left_identity, right_identity):
        if operation_error is not None:
            raise operation_error
        raise RuntimeError(f"{state.path}: {description} did not complete")

    reverse_error: BaseException | None = None
    reverse_attempted = False
    if (
        left_after is not None
        and right_after is not None
        and (
            left_after == right_identity
            or right_after == left_identity
        )
    ):
        reverse_attempted = True
        try:
            _rename_exchange_at(state.directory, left_name, right_name)
        except BaseException as exc:
            reverse_error = exc

    if reverse_attempted:
        reversed_left = _entry_identity_at(state.directory, left_name)
        reversed_right = _entry_identity_at(state.directory, right_name)
        if (
            reverse_error is None
            and (reversed_left, reversed_right)
            != (right_after, left_after)
        ):
            reverse_error = RuntimeError(
                "reverse exchange did not restore the observed entries"
            )

    conflict = RuntimeError(
        f"{state.path}: {description} raced with another writer"
    )
    if operation_error is not None:
        _attach_exception_detail(
            conflict,
            f"exchange error: {type(operation_error).__name__}: "
            f"{operation_error}",
        )
    if reverse_error is not None:
        _attach_exception_detail(
            conflict,
            f"reverse exchange error: {type(reverse_error).__name__}: "
            f"{reverse_error}",
        )
    raise conflict from operation_error


def _create_backup(state: _OutputState) -> None:
    assert state.original_fd is not None
    assert state.original_identity is not None
    state.backup_name = _unused_entry_name(state, "bak")
    state.backup_identity = state.original_identity
    _link_fd_at(
        state.original_fd,
        state.directory,
        state.backup_name,
    )
    if not _entry_has_identity(
        state,
        state.backup_name,
        state.original_identity,
    ):
        raise RuntimeError(
            f"{state.path}: transaction backup changed during creation"
        )


def _publish_output(state: _OutputState) -> None:
    assert state.staged_fd is not None
    assert state.staged_identity is not None
    assert state.staged_name is not None
    if state.original_identity is None:
        operation_error: BaseException | None = None
        state.installed = True
        try:
            _link_fd_at(
                state.staged_fd,
                state.directory,
                state.name,
            )
        except BaseException as exc:
            operation_error = exc
        if not _entry_has_identity(
            state,
            state.name,
            state.staged_identity,
        ):
            if operation_error is not None:
                raise operation_error
            raise RuntimeError(
                f"{state.path}: output installation raced with another "
                "writer"
            )
        if operation_error is not None:
            raise operation_error
        return

    state.installed = True
    operation_error = _exchange_expected(
        state,
        state.staged_name,
        state.name,
        state.staged_identity,
        state.original_identity,
        description="output installation",
    )
    state.staged_path_identity = state.original_identity
    if operation_error is not None:
        raise operation_error


def _capture_identity_text(identity: tuple[int, int]) -> str:
    return f"(dev, ino)=({identity[0]}, {identity[1]})"


def _report_foreign_capture(
    state: _OutputState,
    quarantine_name: str,
    source_name: str,
    identity: tuple[int, int],
) -> None:
    """Attempt one lossless restoration, then report the observed race."""

    restoration_error: BaseException | None = None
    try:
        _rename_noreplace_at(
            state.directory,
            quarantine_name,
            source_name,
        )
    except BaseException as exc:
        restoration_error = exc

    source_identity = _entry_identity_at(state.directory, source_name)
    quarantine_identity = _entry_identity_at(
        state.directory,
        quarantine_name,
    )
    directory_path = _directory_path_at(state.directory)
    source_path = directory_path / source_name
    quarantine_path = directory_path / quarantine_name
    if (
        source_identity == identity
        and quarantine_identity is None
    ):
        race = RuntimeError(
            f"{source_path}: capture-then-classify captured foreign entry "
            f"{_capture_identity_text(identity)} and restored it without "
            "overwrite; cleanup raced with another writer"
        )
        if restoration_error is not None:
            _attach_exception_detail(
                race,
                "restoration syscall reported "
                f"{type(restoration_error).__name__}: {restoration_error}",
            )
        raise race from restoration_error

    if quarantine_identity == identity:
        recovery = RuntimeError(
            f"{source_path}: capture-then-classify captured foreign entry "
            f"{_capture_identity_text(identity)}; no-replace restoration "
            f"could not complete, so it remains at {quarantine_path}; "
            "the current source entry was preserved"
        )
        if restoration_error is not None:
            _attach_exception_detail(
                recovery,
                "restoration error: "
                f"{type(restoration_error).__name__}: {restoration_error}",
            )
        raise recovery from restoration_error

    conflict = RuntimeError(
        f"{source_path}: foreign entry {_capture_identity_text(identity)} "
        "changed again during capture classification"
    )
    if restoration_error is not None:
        _attach_exception_detail(
            conflict,
            "restoration error: "
            f"{type(restoration_error).__name__}: {restoration_error}",
        )
    raise conflict from restoration_error


def _capture_then_classify_entry(
    state: _OutputState,
    name: str,
    expected_identity: tuple[int, int] | None,
) -> Path | None:
    """Capture a cleanup source, then classify the captured inode."""

    current = _entry_status(state.directory, name)
    if current is None:
        return None
    if expected_identity is None or _identity(current) != expected_identity:
        raise RuntimeError(
            f"{state.path}: refusing to capture a known-foreign "
            f"transaction entry {name!r}"
        )

    for _attempt in range(100):
        quarantine_name = f".trackgen-retired-{secrets.token_hex(16)}"
        try:
            _rename_noreplace_at(
                state.directory,
                name,
                quarantine_name,
            )
        except FileExistsError:
            if _entry_identity_at(state.directory, name) == expected_identity:
                continue
            raise RuntimeError(
                f"{state.path}: transaction entry {name!r} changed before "
                "capture"
            )
        except FileNotFoundError:
            if _entry_status(state.directory, name) is None:
                return None
            raise

        source_after = _entry_identity_at(state.directory, name)
        captured_identity = _entry_identity_at(
            state.directory,
            quarantine_name,
        )
        quarantine_path = _directory_path_at(state.directory) / quarantine_name
        if captured_identity == expected_identity:
            if source_after is None:
                return quarantine_path
            raise RuntimeError(
                f"{state.path}: expected transaction entry was retained at "
                f"{quarantine_path}, but source {name!r} was refilled with "
                f"{_capture_identity_text(source_after)}"
            )
        if captured_identity is not None:
            _report_foreign_capture(
                state,
                quarantine_name,
                name,
                captured_identity,
            )
        raise RuntimeError(
            f"{state.path}: transaction entry {name!r} "
            "changed during capture classification"
        )
    raise OSError(
        f"{state.path}: unable to allocate a quarantine entry"
    )


def _relink_original_fd(state: _OutputState) -> None:
    assert state.original_fd is not None
    assert state.original_identity is not None
    operation_error: BaseException | None = None
    try:
        _link_fd_at(state.original_fd, state.directory, state.name)
    except BaseException as exc:
        operation_error = exc
    current_identity = _entry_identity_at(state.directory, state.name)
    if current_identity == state.original_identity:
        state.installed = False
        if operation_error is not None:
            raise operation_error
        return
    if operation_error is not None:
        raise operation_error
    if current_identity is not None:
        raise RuntimeError(
            f"{state.path}: exact-inode rollback raced with another writer"
        )
    raise RuntimeError(f"{state.path}: exact-inode rollback did not install output")


def _restore_new_output(state: _OutputState) -> None:
    assert state.staged_identity is not None
    _capture_then_classify_entry(
        state,
        state.name,
        state.staged_identity,
    )
    state.installed = False


def _restore_existing_output(state: _OutputState) -> None:
    assert state.original_identity is not None
    assert state.original_fd is not None
    assert state.staged_identity is not None
    current_identity = _entry_identity_at(state.directory, state.name)
    if current_identity == state.original_identity:
        state.installed = False
        return
    if current_identity is None:
        _relink_original_fd(state)
        return
    if current_identity != state.staged_identity:
        raise RuntimeError(
            f"{state.path}: refusing to replace an unrecognized output"
        )

    state.staged_path_identity = state.original_identity
    _capture_then_classify_entry(
        state,
        state.name,
        state.staged_identity,
    )
    _relink_original_fd(state)


def _restore_output(state: _OutputState) -> None:
    if not state.installed:
        return
    if state.original_identity is None:
        _restore_new_output(state)
    else:
        _restore_existing_output(state)


def _cleanup_artifacts(state: _OutputState) -> None:
    cleanup_errors: list[BaseException] = []
    artifacts = (
        ("staged_name", "staged_path_identity"),
        ("backup_name", "backup_identity"),
    )
    for name_field, identity_field in artifacts:
        name = getattr(state, name_field)
        if name is None:
            continue
        try:
            quarantine_path = _capture_then_classify_entry(
                state,
                name,
                getattr(state, identity_field),
            )
        except BaseException as exc:
            cleanup_errors.append(exc)
        else:
            if quarantine_path is None:
                setattr(state, name_field, None)
                setattr(state, identity_field, None)
            else:
                setattr(state, name_field, quarantine_path.name)
    if len(cleanup_errors) == 1:
        raise cleanup_errors[0]
    if cleanup_errors:
        raise _CombinedCleanupError(cleanup_errors)


def _cleanup_rollback_artifacts(state: _OutputState) -> None:
    _cleanup_artifacts(state)


def _cleanup_committed_artifacts(state: _OutputState) -> None:
    _cleanup_artifacts(state)


def _note_rollback_error(
    publication_error: BaseException,
    rollback_error: BaseException,
) -> None:
    _attach_exception_detail(
        publication_error,
        f"rollback error: {_exception_diagnostic(rollback_error)}",
    )


def _rollback_outputs(
    states: list[_OutputState],
    publication_error: BaseException,
) -> None:
    restored: list[_OutputState] = []
    for state in reversed(states):
        try:
            _restore_output(state)
        except BaseException as exc:
            _note_rollback_error(publication_error, exc)
        else:
            restored.append(state)
    for state in restored:
        try:
            _cleanup_rollback_artifacts(state)
        except BaseException as exc:
            _note_rollback_error(publication_error, exc)


def _atomic_write_csv(
    path: Path,
    rows: list[dict[str, str]],
    header: tuple[str, ...],
    *,
    input_snapshots: tuple[_InputSnapshot, ...] = (),
) -> None:
    _atomic_write_csvs(
        ((path, rows, header),),
        input_snapshots=input_snapshots,
    )


def _atomic_write_csvs(
    specifications: tuple[
        tuple[Path, list[dict[str, str]], tuple[str, ...]],
        ...,
    ],
    *,
    input_snapshots: tuple[_InputSnapshot, ...] = (),
) -> None:
    states: list[_OutputState] = []
    try:
        for path, rows, header in specifications:
            states.append(_open_output_state(path, rows, header))
        _validate_transaction_aliases(states, input_snapshots)

        try:
            for state in states:
                _stage_output(state)
            for state in states:
                _revalidate_output(state)
            for snapshot in input_snapshots:
                _revalidate_input(snapshot)

            for state in states:
                _preflight_rename_noreplace(state)
            for state in states:
                if state.original_identity is None:
                    continue
                _create_backup(state)

            for state in states:
                _publish_output(state)

            for state in states:
                _revalidate_installed_output(state)
            for snapshot in input_snapshots:
                _revalidate_input(snapshot)

            for state in states:
                _cleanup_committed_artifacts(state)
        except BaseException as publication_error:
            _rollback_outputs(states, publication_error)
            raise
    finally:
        for state in states:
            if state.staged_fd is not None:
                os.close(state.staged_fd)
            if state.original_fd is not None:
                os.close(state.original_fd)
            os.close(state.directory.fd)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and compare deterministic survey coding samples."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--select", action="store_true")
    mode.add_argument("--prepare-blind", action="store_true")
    mode.add_argument("--materialize-primary", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--packet-output", type=Path)
    parser.add_argument("--template-output", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _validate_mode(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    blind_arguments = (
        arguments.selection,
        arguments.candidates,
        arguments.packet_output,
        arguments.template_output,
    )
    if arguments.materialize_primary:
        if (
            arguments.evidence is None
            or arguments.selection is None
            or arguments.output is None
        ):
            parser.error(
                "--materialize-primary requires --selection, --evidence, "
                "and --output"
            )
        if (
            arguments.candidates is not None
            or arguments.packet_output is not None
            or arguments.template_output is not None
            or arguments.primary is not None
            or arguments.reliability is not None
        ):
            parser.error(
                "--materialize-primary cannot be combined with other "
                "mode arguments"
            )
        return

    if arguments.prepare or arguments.select:
        mode = "--prepare" if arguments.prepare else "--select"
        if (
            arguments.primary is not None
            or arguments.reliability is not None
            or any(value is not None for value in blind_arguments)
        ):
            parser.error(f"{mode} cannot be combined with other mode arguments")
        if arguments.evidence is None or arguments.output is None:
            parser.error(f"{mode} requires --evidence and --output")
        return

    if arguments.prepare_blind:
        if (
            arguments.evidence is not None
            or arguments.primary is not None
            or arguments.reliability is not None
            or arguments.output is not None
        ):
            parser.error(
                "--prepare-blind cannot be combined with other mode arguments"
            )
        required = {
            "--selection": arguments.selection,
            "--candidates": arguments.candidates,
            "--packet-output": arguments.packet_output,
            "--template-output": arguments.template_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"--prepare-blind requires {', '.join(missing)}")
        if arguments.packet_output == arguments.template_output:
            parser.error("--packet-output and --template-output must differ")
        return

    if arguments.evidence is not None or any(
        value is not None for value in blind_arguments
    ):
        parser.error("comparison mode cannot use preparation arguments")
    if (
        arguments.primary is None
        or arguments.reliability is None
        or arguments.output is None
    ):
        parser.error(
            "comparison mode requires --primary, --reliability, and --output"
        )


def _mode_paths(
    arguments: argparse.Namespace,
) -> tuple[tuple[str, Path], ...]:
    if arguments.materialize_primary:
        paths = (
            ("--selection", arguments.selection),
            ("--evidence", arguments.evidence),
            ("--output", arguments.output),
        )
    elif arguments.prepare or arguments.select:
        paths = (
            ("--evidence", arguments.evidence),
            ("--output", arguments.output),
        )
    elif arguments.prepare_blind:
        paths = (
            ("--selection", arguments.selection),
            ("--candidates", arguments.candidates),
            ("--packet-output", arguments.packet_output),
            ("--template-output", arguments.template_output),
        )
    else:
        # Same-file comparison fabricates perfect agreement. Distinct primary
        # and reliability inputs are an independence guard, not backward
        # compatibility.
        paths = (
            ("--primary", arguments.primary),
            ("--reliability", arguments.reliability),
            ("--output", arguments.output),
        )
    return tuple(
        (option, path)
        for option, path in paths
        if path is not None
    )


def _validate_distinct_paths(
    parser: argparse.ArgumentParser,
    paths: tuple[tuple[str, Path], ...],
) -> None:
    resolved: list[tuple[str, Path, Path]] = []
    for option, path in paths:
        try:
            canonical_path = path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            parser.error(f"{option} cannot be resolved: {exc}")

        for other_option, other_path, other_canonical in resolved:
            aliased = canonical_path == other_canonical
            if not aliased:
                try:
                    aliased = path.samefile(other_path)
                except OSError:
                    pass
            if aliased:
                parser.error(
                    f"{other_option} and {option} must reference distinct paths"
                )
        resolved.append((option, path, canonical_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    _validate_mode(parser, arguments)
    _validate_distinct_paths(parser, _mode_paths(arguments))

    if arguments.prepare_blind:
        selection_path = arguments.selection
        candidates_path = arguments.candidates
        packet_output = arguments.packet_output
        template_output = arguments.template_output
        assert selection_path is not None
        assert candidates_path is not None
        assert packet_output is not None
        assert template_output is not None
        packet, template, input_snapshots = _prepare_blind_rows(
            selection_path,
            candidates_path,
            include_snapshots=True,
        )
        _atomic_write_csvs(
            (
                (packet_output, packet, PACKET_FIELDS),
                (template_output, template, TEMPLATE_FIELDS),
            ),
            input_snapshots=input_snapshots,
        )
        return 0

    if arguments.materialize_primary:
        selection_path = arguments.selection
        evidence_path = arguments.evidence
        output_path = arguments.output
        assert selection_path is not None
        assert evidence_path is not None
        assert output_path is not None
        evidence_header, selected_evidence, input_snapshots = (
            _materialize_primary_rows(
                selection_path,
                evidence_path,
                include_snapshots=True,
            )
        )
        _atomic_write_csv(
            output_path,
            selected_evidence,
            evidence_header,
            input_snapshots=input_snapshots,
        )
        return 0

    output_path = arguments.output
    assert output_path is not None
    if arguments.prepare or arguments.select:
        evidence_path = arguments.evidence
        assert evidence_path is not None
        header, evidence, evidence_snapshot = _read_csv_with_identity(
            evidence_path
        )
        _require_columns(evidence_path, header, ("cite_key", "domain"))
        selected = select_reliability_sample(evidence)
        if arguments.select:
            rows = _selection_rows(selected, evidence_snapshot.sha256)
            output_header = SELECTION_FIELDS
        else:
            rows = selected
            output_header = header
        input_snapshots = (evidence_snapshot,)
    else:
        primary_path = arguments.primary
        reliability_path = arguments.reliability
        assert primary_path is not None
        assert reliability_path is not None
        primary_header, primary, primary_snapshot = _read_csv_with_identity(
            primary_path
        )
        (
            reliability_header,
            reliability,
            reliability_snapshot,
        ) = _read_csv_with_identity(reliability_path)
        _validate_independent_captures(
            primary_snapshot,
            reliability_snapshot,
        )
        required = ("cite_key", *CORE_FIELDS)
        _require_columns(primary_path, primary_header, required)
        _require_columns(reliability_path, reliability_header, required)
        rows = compare_codings(primary, reliability)
        output_header = SUMMARY_FIELDS
        input_snapshots = (primary_snapshot, reliability_snapshot)

    _atomic_write_csv(
        output_path,
        rows,
        output_header,
        input_snapshots=input_snapshots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
