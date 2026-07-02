from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import SplitResult, parse_qsl, urlsplit, urlunsplit

try:
    import paper.scripts.prepare_screening_batches as screening_batches
except ModuleNotFoundError:  # Direct execution from paper/scripts.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import paper.scripts.prepare_screening_batches as screening_batches

from paper.scripts.prepare_screening_batches import (
    MANIFEST_HEADER as COORDINATOR_MANIFEST_HEADER,
    PACKET_FILENAMES,
    SnapshotError,
    validate_snapshot as validate_coordinator_snapshot,
)


class ScreeningResultError(SnapshotError):
    """A result file, gate record, or immutable result snapshot is invalid."""


RESULT_HEADER = (
    "assignment_id",
    "phase",
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "batch_id",
    "coder_id",
    "screened_on",
    "screening_status",
    "criterion",
    "access_status",
    "source_urls",
    "evidence_version",
    "evidence_retrieved_on",
    "evidence_archive_url",
    "evidence_sha256",
    "screening_locator",
    "exclusion_reason",
    "notes",
)

CALIBRATION_DECISION_HEADER = (
    "decision_id",
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "candidate_ids_sha256",
    "assignment_ids_sha256",
    "status_agreement_numerator",
    "status_agreement_denominator",
    "status_agreement",
    "systematic_ambiguity",
    "decision",
    "decided_on",
    "decision_makers",
    "resolution_evidence",
)

PHASE_RESULT_MANIFEST_HEADER = (
    "manifest_version",
    "phase_result_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "reviewer_release_sha256",
    "phase",
    "batch_id",
    "coder_id",
    "result_filename",
    "result_file_sha256",
    "row_count",
)

CALIBRATION_DECISION_MANIFEST_HEADER = (
    "manifest_version",
    "calibration_decision_snapshot_sha256",
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "decision_id",
    "decision_file_sha256",
    "candidate_ids_file_sha256",
    "assignment_ids_file_sha256",
    "row_count",
)

MANIFEST_VERSION = "1"
PHASES = frozenset({"calibration", "main"})
BATCH_IDS = tuple(filename.removesuffix(".csv") for filename in PACKET_FILENAMES)
RESULT_FILENAMES = tuple(f"{batch_id}.csv" for batch_id in BATCH_IDS)
LEGACY_SCREENING_STATUSES = ("included", "boundary", "excluded")
SCREENING_STATUSES = frozenset(LEGACY_SCREENING_STATUSES)
INCLUSION_CRITERIA = (
    "include-1",
    "include-2",
    "include-3",
    "include-4",
)
EXCLUSION_CRITERIA = frozenset(
    {
        "exclude-fixed-racing-line",
        "exclude-appearance-dynamics",
        "exclude-traffic-only",
        "exclude-insufficient-detail",
        "exclude-out-of-scope",
    }
)
ACCESS_STATUSES = frozenset(
    {
        "full_text",
        "full_text_and_supplement",
        "official_documentation",
        "abstract_only",
    }
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}")
_POSITIVE_OR_ZERO = re.compile(r"0|[1-9][0-9]*")
_VERSION_PIN_PATH = re.compile(
    r"(?:/(?:web/)?[0-9]{8,14}(?:/|$)|"
    r"/(?:commit|releases?/tag|versions?)/[^/?#]+(?:/|$)|"
    r"(?<![0-9a-f])[0-9a-f]{7,64}(?![0-9a-f])|"
    r"(?:^|[/_.-])v?[0-9]+(?:\.[0-9]+)+(?:[/_.-]|$)|"
    r"(?:^|[/_.-])v[0-9]+(?:[/_.-]|$))",
    re.IGNORECASE,
)
_MUTABLE_REFERENCE_VALUES = frozenset(
    {"current", "default", "head", "home", "latest", "main", "master", "top"}
)
_LOCATOR_TOKEN = re.compile(
    r"(?:\bpages?\s+[A-Za-z0-9]|\bparagraphs?\s+[A-Za-z0-9]|"
    r"\bsentences?\s+[A-Za-z0-9]|\bpp?\.?\s*\d+|"
    r"\bsections?\s+[\"']?[A-Za-z0-9][^;,\s]*|"
    r"\u00a7+\s*[A-Za-z0-9]|"
    r"\b(?:tables?|figures?|fig\.?|algorithms?|chapters?|"
    r"supplements?|anchors?|commits?|lines?|positions?)\s+"
    r"[#A-Za-z0-9][A-Za-z0-9._:~/#-]*|"
    r"\bappendi(?:x|ces)\s+[A-Za-z0-9]|"
    r"#[A-Za-z0-9][A-Za-z0-9._:~-]{2,})",
    re.IGNORECASE,
)
_STABLE_HEADING_LOCATOR = re.compile(
    r"\b(?:tab|topics?|record|fields?|description|challenge|scenarios?|"
    r"results?\s+page|class|heading|statement)\b",
    re.IGNORECASE,
)
_ARXIV_VERSIONED_PATH = re.compile(
    r"/(?:abs|pdf)/[0-9]{4}\.[0-9]{4,5}v[1-9][0-9]*(?:\.pdf)?$",
    re.IGNORECASE,
)
_UUID_PATH = re.compile(
    r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}(?:/|$)",
    re.IGNORECASE,
)
_STABLE_FRAGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:~-]{2,}")

Row = dict[str, str]
FileIdentity = tuple[int, int]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(payload)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _raise_result_error(exc: BaseException) -> ScreeningResultError:
    return ScreeningResultError(str(exc))


def _normalize_os_errors(function):
    """Translate filesystem failures at public boundaries."""

    @wraps(function)
    def normalized(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as exc:
            raise _raise_result_error(exc) from exc

    return normalized


def _producer_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except SnapshotError as exc:
        raise _raise_result_error(exc) from exc


def _parse_csv(
    payload: bytes,
    label: str,
    header: tuple[str, ...],
    *,
    no_blank_cells: bool = True,
) -> list[Row]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ScreeningResultError(f"{label}: invalid UTF-8: {exc}") from exc
    if text.startswith("\ufeff"):
        raise ScreeningResultError(f"{label}: UTF-8 BOM is not allowed")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        actual_header = tuple(reader.fieldnames or ())
        if actual_header != header:
            raise ScreeningResultError(
                f"{label}: headers {actual_header!r} != {header!r}"
            )
        rows = list(reader)
    except csv.Error as exc:
        raise ScreeningResultError(
            f"{label}:{reader.line_num}: CSV parse error: {exc}"
        ) from exc
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ScreeningResultError(f"{label}:{row_number}: malformed CSV row")
        for field, value in row.items():
            if no_blank_cells and not value:
                raise ScreeningResultError(
                    f"{label}:{row_number}: {field} must not be blank"
                )
            if value != value.strip():
                raise ScreeningResultError(
                    f"{label}:{row_number}: {field} must be trimmed"
                )
            if "\x00" in value or "\r" in value or "\n" in value:
                raise ScreeningResultError(
                    f"{label}:{row_number}: {field} contains a control character"
                )
    return rows


def _csv_bytes(header: tuple[str, ...], rows: Iterable[Row]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=header,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _checksums(artifacts: dict[str, bytes]) -> bytes:
    return "".join(
        f"{_sha256(artifacts[name])}  {name}\n" for name in sorted(artifacts)
    ).encode("utf-8")


def parse_canonical_csv(
    payload: bytes,
    label: str,
    header: tuple[str, ...],
    *,
    no_blank_cells: bool = True,
) -> list[Row]:
    """Parse canonical UTF-8 CSV using the screening-result schema rules."""

    return _parse_csv(
        payload,
        label,
        header,
        no_blank_cells=no_blank_cells,
    )


def render_canonical_csv(
    header: tuple[str, ...],
    rows: Iterable[Row],
) -> bytes:
    """Render rows as canonical UTF-8 CSV."""

    return _csv_bytes(header, rows)


def render_sha256sums(artifacts: dict[str, bytes]) -> bytes:
    """Render the canonical SHA256SUMS payload for named artifacts."""

    return _checksums(artifacts)


def _identifier_preimage(
    values: Iterable[str],
    *,
    sort_utf8: bool,
) -> bytes:
    identifiers = list(values)
    if not identifiers:
        raise ScreeningResultError("identifier collection must not be empty")
    if any(not value or value != value.strip() for value in identifiers):
        raise ScreeningResultError(
            "identifier collection contains an invalid value"
        )
    if len(identifiers) != len(set(identifiers)):
        raise ScreeningResultError(
            "identifier collection contains duplicates"
        )
    if sort_utf8:
        identifiers.sort(key=lambda value: value.encode("utf-8"))
    return "".join(
        f"{value}\n" for value in identifiers
    ).encode("utf-8")


def sequence_ids_sha256(values: Iterable[str]) -> str:
    return _sha256(_identifier_preimage(values, sort_utf8=False))


def ordered_ids_sha256(values: Iterable[str]) -> str:
    return _sha256(_identifier_preimage(values, sort_utf8=True))


def canonical_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ScreeningResultError("agreement ratio is outside its valid range")
    value = Decimal(numerator) / Decimal(denominator)
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")


@dataclass(frozen=True)
class DirectoryFingerprint:
    path: Path
    identity: FileIdentity
    mode: int
    link_count: int
    entries_sha256: str
    expected_names: tuple[str, ...]

    @_normalize_os_errors
    def reattest(self) -> bool:
        try:
            parent_fd, parent_identity = screening_batches._open_directory_fd(
                self.path.parent, "snapshot fingerprint parent"
            )
        except SnapshotError as exc:
            raise _raise_result_error(exc) from exc
        root_fd: int | None = None
        try:
            root_fd = os.open(
                self.path.name,
                screening_batches._DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
            actual = screening_batches._attest_directory_fd(
                root_fd,
                set(self.expected_names),
                "snapshot fingerprint tree",
                self.mode,
            )
            if (
                actual.identity != self.identity
                or actual.link_count != self.link_count
                or actual.entries_sha256 != self.entries_sha256
            ):
                raise ScreeningResultError(
                    f"{self.path}: snapshot tree changed after capture"
                )
            screening_batches._recheck_directory_path(
                self.path.parent,
                parent_identity,
                "snapshot fingerprint parent",
            )
            if (
                screening_batches._identity_at(parent_fd, self.path.name)
                != self.identity
            ):
                raise ScreeningResultError(
                    f"{self.path}: snapshot tree changed after capture"
                )
            return True
        except SnapshotError as exc:
            raise _raise_result_error(exc) from exc
        finally:
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)


@dataclass(frozen=True)
class FileFingerprint:
    path: Path
    identity: FileIdentity
    sha256: str
    size: int
    mode: int
    link_count: int
    tree: DirectoryFingerprint | None = None

    @_normalize_os_errors
    def reattest(self) -> bool:
        read_file = _producer_call(
            screening_batches._read_regular_file,
            self.path,
            "immutable snapshot file",
        )
        if (
            read_file.identity != self.identity
            or _sha256(read_file.payload) != self.sha256
            or len(read_file.payload) != self.size
            or read_file.mode != self.mode
            or read_file.link_count != self.link_count
        ):
            raise ScreeningResultError(f"{self.path}: file changed after capture")
        if self.tree is not None:
            self.tree.reattest()
        return True


@dataclass(frozen=True)
class PhaseResultSnapshot:
    directory: Path
    phase: str
    rows: tuple[Row, ...]
    snapshot_sha256: str
    coordinator_snapshot_sha256: str
    protocol_sha256: str
    reviewer_release_sha256: str
    manifest: tuple[Row, ...]
    fingerprints: tuple[FileFingerprint, ...]


@dataclass(frozen=True)
class CalibrationDecisionSnapshot:
    directory: Path
    decision: Row
    snapshot_sha256: str
    coordinator_snapshot_sha256: str
    calibration_result_snapshot_sha256: str
    manifest: Row
    fingerprints: tuple[FileFingerprint, ...]


@dataclass(frozen=True)
class CoordinatorSnapshot:
    directory: Path
    payloads: dict[str, bytes]
    manifest: tuple[Row, ...]
    snapshot_sha256: str
    protocol_sha256: str
    allowed_inclusion_criteria: tuple[str, ...]
    allowed_screening_statuses: tuple[str, ...]
    calibration_candidate_ids: tuple[str, ...]
    tree: DirectoryFingerprint
    fingerprints: tuple[FileFingerprint, ...]


@dataclass(frozen=True)
class ReviewerReleaseSnapshot:
    directory: Path
    phase: str
    payloads: dict[str, bytes]
    manifest: Row
    snapshot_sha256: str
    root: DirectoryFingerprint
    packets_tree: DirectoryFingerprint
    fingerprints: tuple[FileFingerprint, ...]


# Compatibility aliases for callers and tests predating the public facade.
_Coordinator = CoordinatorSnapshot


@dataclass(frozen=True)
class CalibrationReleaseTuple:
    coordinator: CoordinatorSnapshot
    calibration_release: ReviewerReleaseSnapshot
    calibration: PhaseResultSnapshot
    decision: CalibrationDecisionSnapshot


@dataclass(frozen=True)
class CapturedInput:
    fingerprint: FileFingerprint
    payload: bytes


_CapturedInput = CapturedInput


def _capture_directory_fingerprint(
    path: Path,
    expected_names: Iterable[str],
    label: str,
) -> DirectoryFingerprint:
    directory = _absolute(Path(path))
    names = tuple(sorted(expected_names))
    try:
        parent_fd, parent_identity = screening_batches._open_directory_fd(
            directory.parent, f"{label} parent directory"
        )
    except SnapshotError as exc:
        raise _raise_result_error(exc) from exc
    root_fd: int | None = None
    try:
        try:
            root_stat = os.stat(
                directory.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ScreeningResultError(f"{directory}: {label} is missing") from exc
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ScreeningResultError(
                f"{directory}: {label} must be a real directory"
            )
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        root_fd = os.open(
            directory.name,
            screening_batches._DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
        opened = os.fstat(root_fd)
        if (opened.st_dev, opened.st_ino) != root_identity:
            raise ScreeningResultError(
                f"{directory}: {label} changed before capture"
            )
        attestation = screening_batches._attest_directory_fd(
            root_fd,
            set(names),
            label,
            0o755,
        )
        screening_batches._recheck_directory_path(
            directory.parent,
            parent_identity,
            f"{label} parent directory",
        )
        if (
            screening_batches._identity_at(parent_fd, directory.name)
            != root_identity
        ):
            raise ScreeningResultError(
                f"{directory}: {label} changed during capture"
            )
        return DirectoryFingerprint(
            path=directory,
            identity=attestation.identity,
            mode=attestation.mode,
            link_count=attestation.link_count,
            entries_sha256=attestation.entries_sha256,
            expected_names=names,
        )
    except SnapshotError as exc:
        raise _raise_result_error(exc) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _capture_coordinator(path: Path) -> _Coordinator:
    directory = _absolute(Path(path))
    tree = _capture_directory_fingerprint(
        directory,
        screening_batches.ROOT_FILENAMES,
        "coordinator snapshot",
    )
    payloads = _producer_call(validate_coordinator_snapshot, directory)
    taxonomy = json.loads(payloads["taxonomy.json"])
    criteria = taxonomy.get("screening_inclusion_criterion")
    allowed_inclusion_criteria = (
        INCLUSION_CRITERIA if criteria is None else tuple(criteria)
    )
    statuses = screening_batches._resolve_screening_result_statuses(
        taxonomy,
        strict_new=False,
    )
    allowed_screening_statuses = (
        LEGACY_SCREENING_STATUSES if statuses is None else statuses
    )
    tree.reattest()
    manifest = _parse_csv(
        payloads["manifest.csv"],
        "coordinator manifest.csv",
        COORDINATOR_MANIFEST_HEADER,
        no_blank_cells=False,
    )
    if not manifest:
        raise ScreeningResultError("coordinator manifest.csv must not be empty")
    snapshot_values = {row["snapshot_sha256"] for row in manifest}
    protocol_values = {row["protocol_sha256"] for row in manifest}
    if len(snapshot_values) != 1 or len(protocol_values) != 1:
        raise ScreeningResultError("coordinator manifest hashes are inconsistent")
    selection_rows = _parse_csv(
        payloads["calibration_selection.csv"],
        "coordinator calibration_selection.csv",
        screening_batches.CALIBRATION_SELECTION_HEADER,
    )
    calibration_candidate_ids = tuple(
        row["candidate_id"] for row in selection_rows
    )
    if len(calibration_candidate_ids) != screening_batches.CALIBRATION_CANDIDATE_COUNT:
        raise ScreeningResultError(
            "coordinator calibration selection must contain exactly 30 IDs"
        )
    fingerprints: list[FileFingerprint] = []
    for relative, payload in sorted(payloads.items()):
        captured = _capture_input(
            directory / relative, f"coordinator snapshot {relative}"
        )
        if captured.payload != payload:
            raise ScreeningResultError(
                f"{directory / relative}: coordinator changed during capture"
            )
        fingerprints.append(captured.fingerprint)
    tree.reattest()
    final_payloads = _producer_call(validate_coordinator_snapshot, directory)
    if _coordinator_payload_hashes(final_payloads) != _coordinator_payload_hashes(
        payloads
    ):
        raise ScreeningResultError("coordinator changed during capture")
    tree.reattest()
    return _Coordinator(
        directory=directory,
        payloads=payloads,
        manifest=tuple(manifest),
        snapshot_sha256=next(iter(snapshot_values)),
        allowed_screening_statuses=allowed_screening_statuses,
        protocol_sha256=next(iter(protocol_values)),
        allowed_inclusion_criteria=allowed_inclusion_criteria,
        calibration_candidate_ids=calibration_candidate_ids,
        tree=tree,
        fingerprints=tuple(fingerprints),
    )


def _coordinator_payload_hashes(payloads: dict[str, bytes]) -> dict[str, str]:
    return {name: _sha256(payload) for name, payload in payloads.items()}


def _valid_file_identity(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(type(component) is int for component in value)
    )


def _directory_fingerprint_signature(
    fingerprint: DirectoryFingerprint,
) -> tuple[object, ...]:
    if (
        type(fingerprint) is not DirectoryFingerprint
        or not isinstance(fingerprint.path, Path)
        or not _valid_file_identity(fingerprint.identity)
        or type(fingerprint.mode) is not int
        or type(fingerprint.link_count) is not int
        or type(fingerprint.entries_sha256) is not str
        or type(fingerprint.expected_names) is not tuple
        or any(type(name) is not str for name in fingerprint.expected_names)
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has an invalid tree fingerprint"
        )
    return (
        fingerprint.path,
        fingerprint.identity,
        fingerprint.mode,
        fingerprint.link_count,
        fingerprint.entries_sha256,
        fingerprint.expected_names,
    )


def _file_fingerprint_signature(
    fingerprint: FileFingerprint,
) -> tuple[object, ...]:
    if (
        type(fingerprint) is not FileFingerprint
        or not isinstance(fingerprint.path, Path)
        or not _valid_file_identity(fingerprint.identity)
        or type(fingerprint.sha256) is not str
        or type(fingerprint.size) is not int
        or type(fingerprint.mode) is not int
        or type(fingerprint.link_count) is not int
        or (
            fingerprint.tree is not None
            and type(fingerprint.tree) is not DirectoryFingerprint
        )
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has an invalid file fingerprint"
        )
    return (
        fingerprint.path,
        fingerprint.identity,
        fingerprint.sha256,
        fingerprint.size,
        fingerprint.mode,
        fingerprint.link_count,
        (
            None
            if fingerprint.tree is None
            else _directory_fingerprint_signature(fingerprint.tree)
        ),
    )


def _coordinator_state_signature(
    coordinator: CoordinatorSnapshot,
) -> tuple[object, ...]:
    if not isinstance(coordinator, CoordinatorSnapshot):
        raise ScreeningResultError(
            "coordinator must be a captured CoordinatorSnapshot"
        )
    if not isinstance(coordinator.directory, Path):
        raise ScreeningResultError(
            "caller-supplied coordinator has an invalid directory"
        )
    if type(coordinator.payloads) is not dict or any(
        type(name) is not str or type(payload) is not bytes
        for name, payload in coordinator.payloads.items()
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has an invalid payload mapping"
        )
    if type(coordinator.manifest) is not tuple or any(
        type(row) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in row.items()
        )
        for row in coordinator.manifest
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has an invalid manifest"
        )
    if (
        type(coordinator.allowed_inclusion_criteria) is not tuple
        or any(
            type(criterion) is not str
            for criterion in coordinator.allowed_inclusion_criteria
        )
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has invalid allowed inclusion criteria"
        )
    if (
        type(coordinator.allowed_screening_statuses) is not tuple
        or any(
            type(status) is not str
            for status in coordinator.allowed_screening_statuses
        )
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has invalid allowed screening statuses"
        )
    if (
        type(coordinator.calibration_candidate_ids) is not tuple
        or any(
            type(candidate_id) is not str
            for candidate_id in coordinator.calibration_candidate_ids
        )
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has invalid calibration candidate IDs"
        )
    if (
        type(coordinator.snapshot_sha256) is not str
        or type(coordinator.protocol_sha256) is not str
        or type(coordinator.fingerprints) is not tuple
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator has invalid scalar or fingerprint state"
        )

    payload_signature = tuple(
        (name, payload, _sha256(payload))
        for name, payload in sorted(
            coordinator.payloads.items(),
            key=lambda item: item[0].encode("utf-8"),
        )
    )
    manifest_signature = tuple(
        tuple(
            sorted(
                row.items(),
                key=lambda item: item[0].encode("utf-8"),
            )
        )
        for row in coordinator.manifest
    )
    return (
        coordinator.directory,
        payload_signature,
        manifest_signature,
        coordinator.snapshot_sha256,
        coordinator.protocol_sha256,
        coordinator.allowed_screening_statuses,
        coordinator.allowed_inclusion_criteria,
        coordinator.calibration_candidate_ids,
        _directory_fingerprint_signature(coordinator.tree),
        tuple(
            _file_fingerprint_signature(fingerprint)
            for fingerprint in coordinator.fingerprints
        ),
    )


def _recapture_coordinator(
    coordinator: CoordinatorSnapshot,
) -> CoordinatorSnapshot:
    if not isinstance(coordinator, CoordinatorSnapshot):
        raise ScreeningResultError(
            "coordinator must be a captured CoordinatorSnapshot"
        )
    fresh = _capture_coordinator(coordinator.directory)
    if _coordinator_state_signature(coordinator) != _coordinator_state_signature(
        fresh
    ):
        raise ScreeningResultError(
            "caller-supplied coordinator does not match authoritative disk capture"
        )
    return fresh


def _reattest_coordinator(
    coordinator: CoordinatorSnapshot,
) -> CoordinatorSnapshot:
    fresh = _recapture_coordinator(coordinator)
    _reattest_trusted_coordinator(fresh)
    return fresh


def _reattest_trusted_coordinator(coordinator: _Coordinator) -> None:
    expected = _coordinator_payload_hashes(coordinator.payloads)
    coordinator.tree.reattest()
    current = _producer_call(
        validate_coordinator_snapshot, coordinator.directory
    )
    if _coordinator_payload_hashes(current) != expected:
        raise ScreeningResultError("coordinator snapshot changed after capture")
    coordinator.tree.reattest()
    for fingerprint in coordinator.fingerprints:
        fingerprint.reattest()
    coordinator.tree.reattest()
    final = _producer_call(
        validate_coordinator_snapshot, coordinator.directory
    )
    if _coordinator_payload_hashes(final) != expected:
        raise ScreeningResultError("coordinator snapshot changed after capture")
    coordinator.tree.reattest()


def _expected_reviewer_release_manifest(
    coordinator: CoordinatorSnapshot,
    phase: str,
    *,
    calibration_result_snapshot_sha256: str = "NR",
    calibration_decision_snapshot_sha256: str = "NR",
) -> Row:
    if phase not in PHASES:
        raise ScreeningResultError(f"unsupported reviewer release phase {phase!r}")
    gate_hashes = (
        calibration_result_snapshot_sha256,
        calibration_decision_snapshot_sha256,
    )
    if phase == "calibration":
        if gate_hashes != ("NR", "NR"):
            raise ScreeningResultError(
                "calibration reviewer release must not declare gate snapshots"
            )
        assignment_count = screening_batches.CALIBRATION_ASSIGNMENT_COUNT
    else:
        if any(_LOWER_SHA256.fullmatch(value) is None for value in gate_hashes):
            raise ScreeningResultError(
                "main reviewer release requires authoritative calibration gate hashes"
            )
        assignment_count = screening_batches.MAIN_ASSIGNMENT_COUNT
    return {
        "manifest_version": screening_batches.MANIFEST_VERSION,
        "phase": phase,
        "coordinator_snapshot_sha256": coordinator.snapshot_sha256,
        "protocol_sha256": coordinator.protocol_sha256,
        "execution_profile_sha256": coordinator.manifest[0][
            "execution_profile_sha256"
        ],
        "prompt_template_sha256": coordinator.manifest[0][
            "prompt_template_sha256"
        ],
        "assignment_count": str(assignment_count),
        "calibration_result_snapshot_sha256": (
            calibration_result_snapshot_sha256
        ),
        "calibration_decision_snapshot_sha256": (
            calibration_decision_snapshot_sha256
        ),
    }


def _reviewer_release_snapshot_sha256(
    phase: str,
    payloads: dict[str, bytes],
) -> str:
    expected_paths = {
        "protocol.md",
        "execution_profile.json",
        "reviewer_prompt_template.md",
        "release_manifest.csv",
        "SHA256SUMS",
        *(f"packets/{filename}" for filename in PACKET_FILENAMES),
    }
    if set(payloads) != expected_paths:
        raise ScreeningResultError(
            "reviewer release payload set is incomplete for hashing"
        )
    return _canonical_sha256(
        {
            "files": [
                {"path": path, "sha256": _sha256(payloads[path])}
                for path in sorted(payloads, key=lambda value: value.encode("utf-8"))
            ],
            "manifest_version": MANIFEST_VERSION,
            "phase": phase,
        }
    )


def _capture_reviewer_release(
    path: Path,
    *,
    expected_manifest: Row,
    coordinator: CoordinatorSnapshot,
) -> ReviewerReleaseSnapshot:
    directory = _absolute(Path(path))
    if _paths_overlap(directory, coordinator.directory):
        raise ScreeningResultError(
            "reviewer release must be disjoint from the coordinator snapshot"
        )
    root = _capture_directory_fingerprint(
        directory,
        screening_batches.REVIEWER_RELEASE_ROOT_FILENAMES,
        "reviewer release snapshot",
    )
    packets_tree = _capture_directory_fingerprint(
        directory / "packets",
        PACKET_FILENAMES,
        "reviewer release packets",
    )
    payloads = _producer_call(
        screening_batches.validate_reviewer_release_snapshot,
        directory,
        expected_manifest=expected_manifest,
        coordinator_snapshot=coordinator.payloads,
    )
    manifest_rows = _parse_csv(
        payloads["release_manifest.csv"],
        "reviewer release manifest.csv",
        screening_batches.RELEASE_MANIFEST_HEADER,
    )
    if len(manifest_rows) != 1 or manifest_rows[0] != expected_manifest:
        raise ScreeningResultError(
            "reviewer release manifest does not match authoritative authorization"
        )

    relative_paths = (
        "protocol.md",
        "execution_profile.json",
        "reviewer_prompt_template.md",
        "release_manifest.csv",
        "SHA256SUMS",
        *(f"packets/{filename}" for filename in PACKET_FILENAMES),
    )
    coordinator_identities = {
        fingerprint.identity for fingerprint in coordinator.fingerprints
    }
    identities: set[FileIdentity] = set()
    fingerprints: list[FileFingerprint] = []
    for relative in relative_paths:
        captured = _capture_input(
            directory / relative,
            f"reviewer release {relative}",
        )
        if captured.payload != payloads[relative]:
            raise ScreeningResultError(
                f"{directory / relative}: reviewer release changed during capture"
            )
        if captured.fingerprint.identity in identities:
            raise ScreeningResultError(
                f"{directory / relative}: reviewer release file is aliased"
            )
        if captured.fingerprint.identity in coordinator_identities:
            raise ScreeningResultError(
                f"{directory / relative}: reviewer release aliases coordinator input"
            )
        identities.add(captured.fingerprint.identity)
        fingerprints.append(captured.fingerprint)

    root.reattest()
    packets_tree.reattest()
    final_payloads = _producer_call(
        screening_batches.validate_reviewer_release_snapshot,
        directory,
        expected_manifest=expected_manifest,
        coordinator_snapshot=coordinator.payloads,
    )
    if _coordinator_payload_hashes(final_payloads) != _coordinator_payload_hashes(
        payloads
    ):
        raise ScreeningResultError("reviewer release changed during capture")
    root.reattest()
    packets_tree.reattest()
    return ReviewerReleaseSnapshot(
        directory=directory,
        phase=expected_manifest["phase"],
        payloads=payloads,
        manifest=dict(expected_manifest),
        snapshot_sha256=_reviewer_release_snapshot_sha256(
            expected_manifest["phase"], payloads
        ),
        root=root,
        packets_tree=packets_tree,
        fingerprints=tuple(fingerprints),
    )


def _reattest_reviewer_release(
    release: ReviewerReleaseSnapshot,
    coordinator: CoordinatorSnapshot,
) -> None:
    expected_hashes = _coordinator_payload_hashes(release.payloads)
    release.root.reattest()
    release.packets_tree.reattest()
    current = _producer_call(
        screening_batches.validate_reviewer_release_snapshot,
        release.directory,
        expected_manifest=release.manifest,
        coordinator_snapshot=coordinator.payloads,
    )
    if _coordinator_payload_hashes(current) != expected_hashes:
        raise ScreeningResultError("reviewer release changed after capture")
    if (
        _reviewer_release_snapshot_sha256(release.phase, current)
        != release.snapshot_sha256
    ):
        raise ScreeningResultError("reviewer release digest changed after capture")
    for fingerprint in release.fingerprints:
        fingerprint.reattest()
    release.packets_tree.reattest()
    release.root.reattest()
    final = _producer_call(
        screening_batches.validate_reviewer_release_snapshot,
        release.directory,
        expected_manifest=release.manifest,
        coordinator_snapshot=coordinator.payloads,
    )
    if _coordinator_payload_hashes(final) != expected_hashes:
        raise ScreeningResultError("reviewer release changed after capture")
    release.packets_tree.reattest()
    release.root.reattest()


def _reattest_fingerprints(
    fingerprints: Sequence[FileFingerprint],
) -> None:
    for fingerprint in fingerprints:
        fingerprint.reattest()


def _coherent_trusted_final_attestation(
    fingerprint_groups: Sequence[Sequence[FileFingerprint]],
    coordinator: _Coordinator,
) -> None:
    """Bound mutations during validation with forward and reverse sweeps."""
    groups = tuple(tuple(group) for group in fingerprint_groups)
    for group in groups:
        _reattest_fingerprints(group)
    _reattest_trusted_coordinator(coordinator)
    _reattest_trusted_coordinator(coordinator)
    for group in reversed(groups):
        _reattest_fingerprints(group)


def _coherent_final_attestation(
    fingerprint_groups: Sequence[Sequence[FileFingerprint]],
    coordinator: CoordinatorSnapshot,
) -> None:
    """Recapture an untrusted coordinator before a coherent sweep."""
    fresh = _recapture_coordinator(coordinator)
    _coherent_trusted_final_attestation(fingerprint_groups, fresh)


def _capture_input(path: Path, label: str) -> _CapturedInput:
    read_file = _producer_call(
        screening_batches._read_regular_file, Path(path), label
    )
    return _CapturedInput(
        fingerprint=FileFingerprint(
            path=read_file.path,
            identity=read_file.identity,
            sha256=_sha256(read_file.payload),
            size=len(read_file.payload),
            mode=read_file.mode,
            link_count=read_file.link_count,
        ),
        payload=read_file.payload,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _path_forms(path: Path) -> tuple[Path, ...]:
    absolute = _absolute(Path(path))
    try:
        resolved = absolute.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ScreeningResultError(
            f"{path}: path could not be resolved for overlap validation"
        ) from exc
    return tuple(dict.fromkeys((absolute, resolved)))


def _reject_output_overlap(output_dir: Path, protected: Iterable[Path]) -> None:
    output_forms = _path_forms(Path(output_dir))
    for path in protected:
        protected_forms = _path_forms(Path(path))
        if any(
            _paths_overlap(output, candidate)
            for output in output_forms
            for candidate in protected_forms
        ):
            raise ScreeningResultError(
                f"{output_dir}: output aliases or overlaps immutable input {path}"
            )


@_normalize_os_errors
def capture_coordinator_snapshot(path: Path) -> CoordinatorSnapshot:
    """Capture and validate an immutable coordinator snapshot."""

    return _capture_coordinator(path)


@_normalize_os_errors
def reattest_coordinator_snapshot(
    coordinator: CoordinatorSnapshot,
) -> CoordinatorSnapshot:
    """Revalidate a previously captured coordinator snapshot."""

    return _reattest_coordinator(coordinator)


@_normalize_os_errors
def reattest_snapshot_set(
    coordinator: CoordinatorSnapshot,
    fingerprint_groups: Sequence[Sequence[FileFingerprint]],
) -> None:
    """Coherently reattest related snapshots against their coordinator."""

    return _coherent_final_attestation(fingerprint_groups, coordinator)


@_normalize_os_errors
def capture_input(path: Path, label: str) -> CapturedInput:
    """Capture one immutable regular-file input."""

    return _capture_input(path, label)


@_normalize_os_errors
def capture_flat_snapshot(
    snapshot_dir: Path,
    expected_filenames: Sequence[str],
) -> tuple[dict[str, bytes], tuple[FileFingerprint, ...]]:
    """Capture an exact, flat immutable snapshot directory."""

    return _capture_flat_snapshot(snapshot_dir, expected_filenames)


def paths_overlap(first: Path, second: Path) -> bool:
    """Compare absolute lexical and resolved forms for path overlap."""

    first_forms = _path_forms(Path(first))
    second_forms = _path_forms(Path(second))
    return any(
        _paths_overlap(first_form, second_form)
        for first_form in first_forms
        for second_form in second_forms
    )


def reject_output_overlap(
    output_dir: Path,
    protected: Iterable[Path],
) -> None:
    """Reject an output path that aliases or contains protected inputs."""

    return _reject_output_overlap(output_dir, protected)


def _capture_distinct_inputs(paths: Sequence[Path]) -> list[_CapturedInput]:
    lexical: dict[Path, Path] = {}
    captured: list[_CapturedInput] = []
    identities: dict[FileIdentity, Path] = {}
    for supplied in paths:
        absolute = _absolute(Path(supplied))
        if absolute in lexical:
            raise ScreeningResultError(
                f"{supplied}: result path aliases {lexical[absolute]}"
            )
        lexical[absolute] = Path(supplied)
        item = _capture_input(Path(supplied), "reviewer result")
        previous = identities.get(item.fingerprint.identity)
        if previous is not None:
            raise ScreeningResultError(
                f"{supplied}: result file aliases {previous}"
            )
        identities[item.fingerprint.identity] = Path(supplied)
        captured.append(item)
    return captured


def _validate_iso_date(value: str, *, field: str, context: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ScreeningResultError(
            f"{context}: {field} must be an ISO date in YYYY-MM-DD form"
        ) from exc
    if parsed.isoformat() != value:
        raise ScreeningResultError(
            f"{context}: {field} must be a canonical ISO date"
        )
    return parsed


def _canonical_http_url(value: str, *, field: str, context: str) -> str:
    if any(character.isspace() or ord(character) < 0x20 for character in value):
        raise ScreeningResultError(
            f"{context}: {field} must be a canonical HTTP(S) URL"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ScreeningResultError(
            f"{context}: {field} must be a valid HTTP(S) URL"
        ) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ScreeningResultError(
            f"{context}: {field} must be an absolute HTTP(S) URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ScreeningResultError(
            f"{context}: {field} must not contain URL credentials"
        )
    hostname = parsed.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None:
        default = (parsed.scheme == "http" and port == 80) or (
            parsed.scheme == "https" and port == 443
        )
        if not default:
            netloc = f"{netloc}:{port}"
    canonical = urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment)
    )
    if value != canonical:
        raise ScreeningResultError(
            f"{context}: {field} is not canonical; expected {canonical!r}"
        )
    return canonical


def _validate_source_urls(value: str, *, context: str) -> tuple[str, ...]:
    urls = value.split(";")
    if any(not url for url in urls):
        raise ScreeningResultError(f"{context}: source_urls contains an empty URL")
    canonical = [
        _canonical_http_url(url, field="source_urls", context=context)
        for url in urls
    ]
    if len(canonical) != len(set(canonical)):
        raise ScreeningResultError(f"{context}: source_urls contains a duplicate")
    if canonical != sorted(canonical, key=lambda url: url.encode("utf-8")):
        raise ScreeningResultError(
            f"{context}: source_urls must use canonical UTF-8 byte order"
        )
    return tuple(canonical)


def _validate_locator(value: str, *, context: str) -> None:
    if value.casefold() in {"abstract", "nr"} or value.startswith(("http://", "https://")):
        raise ScreeningResultError(
            f"{context}: screening_locator must identify precise evidence"
        )
    heading_clauses = [
        part.strip() for part in re.split(r"[;,>]", value) if part.strip()
    ]
    has_stable_heading_path = len(value) >= 20 and (
        len(heading_clauses) >= 2
        or _STABLE_HEADING_LOCATOR.search(value) is not None
    )
    if _LOCATOR_TOKEN.search(value) is None and not has_stable_heading_path:
        raise ScreeningResultError(
            f"{context}: screening_locator lacks a page, section, table, "
            "figure, algorithm, appendix, or stable anchor"
        )


def _has_persistent_document_identifier(parsed: SplitResult) -> bool:
    host = (parsed.hostname or "").casefold()
    path = parsed.path
    query = parsed.query
    if host == "doi.org" and path.startswith("/10."):
        return True
    if host == "arxiv.org" and _ARXIV_VERSIONED_PATH.fullmatch(path):
        return True
    if host == "proceedings.mlr.press" and re.match(r"/v[1-9][0-9]*/", path):
        return True
    if host == "openaccess.thecvf.com" and re.match(
        r"/content/[A-Za-z]+[0-9]{4}/.+\.pdf$", path
    ):
        return True
    if host == "docs.un.org" and path.startswith("/en/"):
        return True
    if host == "documents.un.org" and path == "/api/symbol/access":
        return any(
            key.casefold() == "s" and bool(value)
            for key, value in parse_qsl(query, keep_blank_values=True)
        )
    if host == "eur-lex.europa.eu" and "CELEX:" in query.upper():
        return True
    if host == "mediatum.ub.tum.de" and re.fullmatch(r"/[1-9][0-9]+", path):
        return True
    if host == "digitalcollection.zhaw.ch" and _UUID_PATH.search(path):
        return True
    if host == "robonation.org" and re.search(
        r"/20[0-9]{2}/(?:0[1-9]|1[0-2])/.+\.pdf$",
        path,
        re.IGNORECASE,
    ):
        return True
    if host == "cgl.ethz.ch" and re.search(
        r"/(?:19|20)[0-9]{2}/.+\.pdf$", path, re.IGNORECASE
    ):
        return True
    if host == "cogprints.org" and re.match(
        r"/[1-9][0-9]*/[1-9][0-9]*/.+\.pdf$", path, re.IGNORECASE
    ):
        return True
    if host == "raw.githubusercontent.com" and re.match(
        r"/mlresearch/v[1-9][0-9]*/main/assets/.+\.pdf$",
        path,
        re.IGNORECASE,
    ):
        return True
    return False


def _validate_version_pinned_archive_url(
    value: str,
    *,
    context: str,
    content_hash_pinned: bool = False,
) -> str:
    canonical = _canonical_http_url(
        value,
        field="evidence_archive_url",
        context=context,
    )
    parsed = urlsplit(canonical)
    pin_pairs = [
        (key.casefold(), query_value)
        for key, query_value in parse_qsl(
            parsed.query, keep_blank_values=True
        )
        if key.casefold()
        in {"commit", "revision", "sha", "timestamp", "version"}
    ]
    if any(
        query_value.casefold() in _MUTABLE_REFERENCE_VALUES
        for _, query_value in pin_pairs
    ):
        raise ScreeningResultError(
            f"{context}: evidence_archive_url uses a mutable version value"
        )
    persistent_document = _has_persistent_document_identifier(parsed)
    path_segments = {
        segment.casefold() for segment in parsed.path.split("/") if segment
    }
    if path_segments & _MUTABLE_REFERENCE_VALUES and not persistent_document:
        raise ScreeningResultError(
            f"{context}: evidence_archive_url uses a mutable path reference"
        )
    pinned_query = any(bool(query_value) for _, query_value in pin_pairs)
    pin_material = f"{parsed.path}#{parsed.fragment}"
    if (
        _VERSION_PIN_PATH.search(pin_material) is None
        and not pinned_query
        and not persistent_document
        and not content_hash_pinned
    ):
        raise ScreeningResultError(
            f"{context}: evidence_archive_url must be version-pinned"
        )
    return canonical


def _validate_url_fragment(
    url: str, locator: str, *, field: str, context: str
) -> None:
    fragment = urlsplit(url).fragment
    if not fragment:
        return
    if (
        fragment.casefold() in _MUTABLE_REFERENCE_VALUES
        or _STABLE_FRAGMENT.fullmatch(fragment) is None
    ):
        raise ScreeningResultError(
            f"{context}: {field} fragment is not a precise stable anchor"
        )
    github_lines = re.fullmatch(
        r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", fragment
    )
    represented_as_lines = False
    if github_lines is not None:
        start, end = github_lines.groups()
        rendered = start if end is None else f"{start}-{end}"
        represented_as_lines = (
            re.search(
                rf"\blines?\s+{re.escape(rendered)}\b",
                locator,
                re.IGNORECASE,
            )
            is not None
        )
        if not represented_as_lines and urlsplit(url).hostname == "github.com":
            represented_as_lines = (
                re.search(
                    r"\blines?\s+[1-9][0-9]*(?:-[1-9][0-9]*)?\b",
                    locator,
                    re.IGNORECASE,
                )
                is not None
            )
    if f"#{fragment}" not in locator and not represented_as_lines:
        raise ScreeningResultError(
            f"{context}: {field} fragment must appear in screening_locator"
        )


def _validate_result_decision(
    row: Row,
    *,
    context: str,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = LEGACY_SCREENING_STATUSES,
) -> None:
    status = row["screening_status"]
    criterion = row["criterion"]
    if status not in allowed_screening_statuses:
        raise ScreeningResultError(
            f"{context}: invalid screening_status {status!r}"
        )
    if status == "included" and criterion not in allowed_inclusion_criteria:
        raise ScreeningResultError(
            f"{context}: criterion {criterion!r} is invalid for included"
        )
    if status == "boundary" and criterion != "boundary":
        raise ScreeningResultError(
            f"{context}: criterion must be 'boundary' for boundary"
        )
    if status == "excluded" and criterion not in EXCLUSION_CRITERIA:
        raise ScreeningResultError(
            f"{context}: criterion {criterion!r} is invalid for excluded"
        )

    access = row["access_status"]
    if access not in ACCESS_STATUSES:
        raise ScreeningResultError(f"{context}: invalid access_status {access!r}")
    if status in {"included", "boundary"} and access == "abstract_only":
        raise ScreeningResultError(
            f"{context}: included and boundary decisions cannot use abstract_only"
        )

    screened = _validate_iso_date(
        row["screened_on"], field="screened_on", context=context
    )
    retrieved = _validate_iso_date(
        row["evidence_retrieved_on"],
        field="evidence_retrieved_on",
        context=context,
    )
    if retrieved > screened:
        raise ScreeningResultError(
            f"{context}: evidence_retrieved_on cannot follow screened_on"
        )
    source_urls = _validate_source_urls(row["source_urls"], context=context)
    if row["evidence_version"] == "NR" or len(row["evidence_version"]) < 3:
        raise ScreeningResultError(
            f"{context}: evidence_version must identify the inspected artifact"
        )
    archive_url = row["evidence_archive_url"]
    evidence_sha256 = row["evidence_sha256"]
    if (
        evidence_sha256 != "NR"
        and _LOWER_SHA256.fullmatch(evidence_sha256) is None
    ):
        raise ScreeningResultError(
            f"{context}: evidence_sha256 must be NR or lowercase 64-hex"
        )
    archive_canonical: str | None = None
    if archive_url != "NR":
        archive_canonical = _validate_version_pinned_archive_url(
            archive_url,
            context=context,
            content_hash_pinned=evidence_sha256 != "NR",
        )
    if (
        status in {"included", "boundary"}
        and access == "official_documentation"
        and archive_url == "NR"
        and evidence_sha256 == "NR"
    ):
        raise ScreeningResultError(
            f"{context}: official-documentation provenance requires a "
            "version-pinned archive URL or SHA-256"
        )
    locator = row["screening_locator"]
    _validate_locator(locator, context=context)
    for source_url in source_urls:
        _validate_url_fragment(
            source_url, locator, field="source_urls", context=context
        )
    if archive_canonical is not None:
        _validate_url_fragment(
            archive_canonical,
            locator,
            field="evidence_archive_url",
            context=context,
        )
    if access == "abstract_only":
        notes = row["notes"]
        limitation_terms = (
            "abstract",
            "access",
            "attempt",
            "exhausted",
            "full text",
            "limitation",
            "retriev",
            "unavailable",
        )
        if (
            notes == "NR"
            or len(notes) < 24
            or not any(term in notes.casefold() for term in limitation_terms)
        ):
            raise ScreeningResultError(
                f"{context}: abstract_only requires substantive notes "
                "explaining the evidence limitation"
            )

    reason = row["exclusion_reason"]
    if status in {"included", "boundary"}:
        if reason != "NR":
            raise ScreeningResultError(
                f"{context}: exclusion_reason must be NR for {status}"
            )
    elif (
        reason == "NR"
        or len(reason) < 24
        or reason.casefold() in {criterion.casefold(), "not relevant"}
    ):
        raise ScreeningResultError(
            f"{context}: exclusion_reason must be substantive and source-specific"
        )
    nr_fields = {
        "evidence_archive_url",
        "evidence_sha256",
        "exclusion_reason",
        "notes",
    }
    for field in RESULT_HEADER:
        if field not in nr_fields and row[field] == "NR":
            raise ScreeningResultError(f"{context}: {field} must not be NR")


def is_valid_identifier(value: object) -> bool:
    """Return whether value is a canonical screening identifier."""

    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def validate_iso_date(
    value: str,
    *,
    field: str,
    context: str,
) -> date:
    """Validate and return a canonical YYYY-MM-DD date."""

    return _validate_iso_date(value, field=field, context=context)


def validate_result_decision(row: Row, *, context: str) -> None:
    """Validate one screening decision and its evidence provenance."""

    return _validate_result_decision(row, context=context)


def _validate_phase_payloads(
    coordinator: _Coordinator | None,
    phase: str,
    payloads_by_batch: dict[str, bytes],
    *,
    reviewer_release_sha256: str,
    sealed_coordinator_hash: str | None = None,
    sealed_protocol_hash: str | None = None,
) -> tuple[list[Row], list[Row], str, str, str]:
    if phase not in PHASES:
        raise ScreeningResultError(f"unsupported screening phase {phase!r}")
    if _LOWER_SHA256.fullmatch(reviewer_release_sha256) is None:
        raise ScreeningResultError(
            "reviewer_release_sha256 must be a canonical lowercase SHA256"
        )
    if set(payloads_by_batch) != set(BATCH_IDS):
        raise ScreeningResultError(
            "phase result batches mismatch; "
            f"missing={sorted(set(BATCH_IDS) - set(payloads_by_batch))}, "
            f"extra={sorted(set(payloads_by_batch) - set(BATCH_IDS))}"
        )

    expected_by_assignment: dict[str, Row] = {}
    expected_by_batch: defaultdict[str, set[str]] = defaultdict(set)
    if coordinator is not None:
        for row in coordinator.manifest:
            if row["phase"] == phase:
                expected_by_assignment[row["assignment_id"]] = row
                expected_by_batch[row["batch_id"]].add(row["assignment_id"])
        coordinator_hash = coordinator.snapshot_sha256
        protocol_hash = coordinator.protocol_sha256
        allowed_inclusion_criteria = coordinator.allowed_inclusion_criteria
        allowed_screening_statuses = coordinator.allowed_screening_statuses
    else:
        coordinator_hash = sealed_coordinator_hash or ""
        protocol_hash = sealed_protocol_hash or ""
        allowed_inclusion_criteria = INCLUSION_CRITERIA
        allowed_screening_statuses = LEGACY_SCREENING_STATUSES
        if (
            _LOWER_SHA256.fullmatch(coordinator_hash) is None
            or _LOWER_SHA256.fullmatch(protocol_hash) is None
        ):
            raise ScreeningResultError(
                "standalone phase validation requires sealed coordinator and "
                "protocol hashes"
            )

    rows_by_assignment: dict[str, Row] = {}
    coder_by_candidate: defaultdict[str, set[str]] = defaultdict(set)
    manifest_rows: list[Row] = []
    for batch_id in BATCH_IDS:
        filename = f"{batch_id}.csv"
        payload = payloads_by_batch[batch_id]
        rows = _parse_csv(payload, filename, RESULT_HEADER)
        if not rows:
            raise ScreeningResultError(f"{filename}: result file must not be empty")
        assignments_in_file: list[str] = []
        for row_number, row in enumerate(rows, start=2):
            context = f"{filename}:{row_number}"
            if row["batch_id"] != batch_id:
                raise ScreeningResultError(
                    f"{context}: batch_id does not match result file"
                )
            if row["coder_id"] != batch_id:
                raise ScreeningResultError(
                    f"{context}: coder_id must equal preassigned reviewer role {batch_id}"
                )
            if row["phase"] != phase:
                raise ScreeningResultError(
                    f"{context}: phase does not match requested phase {phase}"
                )
            assignment_id = row["assignment_id"]
            if assignment_id in rows_by_assignment:
                raise ScreeningResultError(
                    f"{context}: duplicate assignment_id {assignment_id!r}"
                )
            if coordinator is not None:
                expected = expected_by_assignment.get(assignment_id)
                if expected is None:
                    raise ScreeningResultError(
                        f"{context}: assignment_id is not assigned to phase {phase}"
                    )
                for field in (
                    "phase",
                    "candidate_id",
                    "input_sha256",
                    "snapshot_sha256",
                    "batch_id",
                ):
                    if row[field] != expected[field]:
                        raise ScreeningResultError(
                            f"{context}: {field} does not match frozen assignment"
                        )
            elif row["snapshot_sha256"] != coordinator_hash:
                raise ScreeningResultError(
                    f"{context}: snapshot_sha256 does not match sealed manifest"
                )
            _validate_result_decision(
                row,
                context=context,
                allowed_inclusion_criteria=allowed_inclusion_criteria,
                allowed_screening_statuses=allowed_screening_statuses,
            )
            rows_by_assignment[assignment_id] = row
            assignments_in_file.append(assignment_id)
            coder_by_candidate[row["candidate_id"]].add(row["coder_id"])

        if coordinator is not None:
            actual = set(assignments_in_file)
            expected = expected_by_batch[batch_id]
            if len(actual) != len(assignments_in_file) or actual != expected:
                raise ScreeningResultError(
                    f"{filename}: assignment coverage mismatch; "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
        manifest_rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "phase_result_snapshot_sha256": "",
                "coordinator_snapshot_sha256": coordinator_hash,
                "protocol_sha256": protocol_hash,
                "reviewer_release_sha256": reviewer_release_sha256,
                "phase": phase,
                "batch_id": batch_id,
                "coder_id": batch_id,
                "result_filename": filename,
                "result_file_sha256": _sha256(payload),
                "row_count": str(len(rows)),
            }
        )

    if coordinator is not None and set(rows_by_assignment) != set(expected_by_assignment):
        raise ScreeningResultError("phase result assignment coverage is incomplete")
    candidate_counts = Counter(row["candidate_id"] for row in rows_by_assignment.values())
    expected_candidate_count = {
        "calibration": screening_batches.CALIBRATION_CANDIDATE_COUNT,
        "main": screening_batches.MAIN_CANDIDATE_COUNT,
    }[phase]
    expected_rating_count = {
        "calibration": screening_batches.CALIBRATION_ASSIGNMENT_COUNT,
        "main": screening_batches.MAIN_ASSIGNMENT_COUNT,
    }[phase]
    if (
        len(candidate_counts) != expected_candidate_count
        or len(rows_by_assignment) != expected_rating_count
    ):
        raise ScreeningResultError(
            f"{phase} result snapshot must contain exactly "
            f"{expected_candidate_count} candidates and "
            f"{expected_rating_count} ratings"
        )
    invalid_counts = {
        candidate_id: count
        for candidate_id, count in candidate_counts.items()
        if count != 2
    }
    if invalid_counts:
        raise ScreeningResultError(
            f"phase candidates must have exactly two ratings: {invalid_counts}"
        )
    if any(len(coders) != 2 for coders in coder_by_candidate.values()):
        raise ScreeningResultError(
            "each phase candidate must use two distinct preassigned reviewer roles"
        )

    snapshot_sha256 = _canonical_sha256(
        {
            "coordinator_snapshot_sha256": coordinator_hash,
            "manifest_version": MANIFEST_VERSION,
            "phase": phase,
            "protocol_sha256": protocol_hash,
            "reviewer_release_sha256": reviewer_release_sha256,
            "results": [
                {
                    "batch_id": row["batch_id"],
                    "coder_id": row["coder_id"],
                    "filename": row["result_filename"],
                    "row_count": row["row_count"],
                    "sha256": row["result_file_sha256"],
                }
                for row in manifest_rows
            ],
        }
    )
    for row in manifest_rows:
        row["phase_result_snapshot_sha256"] = snapshot_sha256
    ordered_rows = sorted(
        rows_by_assignment.values(),
        key=lambda row: (
            row["candidate_id"].encode("utf-8"),
            row["assignment_id"].encode("utf-8"),
        ),
    )
    return (
        ordered_rows,
        manifest_rows,
        snapshot_sha256,
        coordinator_hash,
        protocol_hash,
    )


_RELEASE_ASSIGNMENT_FIELDS = (
    "assignment_id",
    "phase",
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "batch_id",
)


def _validate_reviewer_release_assignments(
    result_rows: Sequence[Row],
    release_rows: Sequence[Row],
) -> None:
    def index(rows: Sequence[Row], label: str) -> dict[str, Row]:
        indexed: dict[str, Row] = {}
        for row in rows:
            try:
                assignment_id = row["assignment_id"]
            except (KeyError, TypeError) as exc:
                raise ScreeningResultError(
                    f"{label}: assignment_id is required"
                ) from exc
            if assignment_id in indexed:
                raise ScreeningResultError(
                    f"{label}: duplicate assignment_id {assignment_id!r}"
                )
            indexed[assignment_id] = row
        return indexed

    results_by_assignment = index(result_rows, "phase results")
    release_by_assignment = index(release_rows, "reviewer release")
    if set(results_by_assignment) != set(release_by_assignment):
        missing = sorted(set(release_by_assignment) - set(results_by_assignment))
        extra = sorted(set(results_by_assignment) - set(release_by_assignment))
        raise ScreeningResultError(
            "assignment_id coverage does not exactly match reviewer release; "
            f"missing={missing}, extra={extra}"
        )

    for assignment_id in sorted(
        results_by_assignment,
        key=lambda value: value.encode("utf-8"),
    ):
        result = results_by_assignment[assignment_id]
        released = release_by_assignment[assignment_id]
        for field in _RELEASE_ASSIGNMENT_FIELDS:
            if result.get(field) != released.get(field):
                raise ScreeningResultError(
                    f"assignment_id={assignment_id!r}: {field} does not "
                    "match authoritative reviewer release packet"
                )


def _result_payloads_from_inputs(
    captured: Sequence[_CapturedInput],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for item in captured:
        rows = _parse_csv(
            item.payload, str(item.fingerprint.path), RESULT_HEADER
        )
        if not rows:
            raise ScreeningResultError(
                f"{item.fingerprint.path}: result file must not be empty"
            )
        batches = {row["batch_id"] for row in rows}
        if len(batches) != 1:
            raise ScreeningResultError(
                f"{item.fingerprint.path}: result file must contain one batch"
            )
        batch_id = next(iter(batches))
        if batch_id in payloads:
            raise ScreeningResultError(f"duplicate result file for batch {batch_id}")
        payloads[batch_id] = item.payload
    return payloads


def _publish(
    output_dir: Path,
    artifacts: dict[str, bytes],
    *,
    post_publish_check: Callable[[], None] | None = None,
) -> None:
    try:
        screening_batches.publish_snapshot(
            Path(output_dir),
            artifacts,
            post_publish_check=post_publish_check,
        )
    except SnapshotError as exc:
        raise _raise_result_error(exc) from exc


@_normalize_os_errors
def seal_phase_results(
    coordinator_snapshot_dir: Path,
    phase: str,
    result_paths: Sequence[Path],
    output_dir: Path,
    *,
    reviewer_release_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path | None = None,
    calibration_result_snapshot_dir: Path | None = None,
    calibration_decision_snapshot_dir: Path | None = None,
) -> None:
    if phase not in PHASES:
        raise ScreeningResultError(f"unsupported screening phase {phase!r}")
    if len(result_paths) != len(BATCH_IDS):
        raise ScreeningResultError(
            f"phase sealing requires exactly six result paths, got {len(result_paths)}"
        )
    coordinator = _capture_coordinator(Path(coordinator_snapshot_dir))
    gate_inputs = (
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
    )
    authorization: CalibrationReleaseTuple | None = None
    if phase == "calibration":
        if any(value is not None for value in gate_inputs):
            raise ScreeningResultError(
                "calibration phase sealing does not accept calibration gate inputs"
            )
        expected_release_manifest = _expected_reviewer_release_manifest(
            coordinator,
            "calibration",
        )
    else:
        if any(value is None for value in gate_inputs):
            raise ScreeningResultError(
                "main phase sealing requires calibration reviewer release, "
                "result, and decision snapshots"
            )
        assert calibration_reviewer_release_snapshot_dir is not None
        assert calibration_result_snapshot_dir is not None
        assert calibration_decision_snapshot_dir is not None
        authorization = capture_calibration_release_tuple(
            coordinator.directory,
            calibration_reviewer_release_snapshot_dir,
            calibration_result_snapshot_dir,
            calibration_decision_snapshot_dir,
        )
        gate = authorization.decision.decision
        if (
            gate["decision"] != "release"
            or gate["systematic_ambiguity"] != "false"
            or Decimal(gate["status_agreement"]) < Decimal("0.80")
        ):
            raise ScreeningResultError(
                "main phase sealing requires a passing calibration release decision"
            )
        expected_release_manifest = _expected_reviewer_release_manifest(
            coordinator,
            "main",
            calibration_result_snapshot_sha256=(
                authorization.calibration.snapshot_sha256
            ),
            calibration_decision_snapshot_sha256=(
                authorization.decision.snapshot_sha256
            ),
        )

    release = _capture_reviewer_release(
        Path(reviewer_release_snapshot_dir),
        expected_manifest=expected_release_manifest,
        coordinator=coordinator,
    )
    protected = [
        coordinator.directory,
        release.directory,
        *(Path(path) for path in result_paths),
    ]
    if authorization is not None:
        protected.extend(
            (
                authorization.calibration_release.directory,
                authorization.calibration.directory,
                authorization.decision.directory,
            )
        )
    _reject_output_overlap(Path(output_dir), protected)
    immutable_directories = [coordinator.directory, release.directory]
    if authorization is not None:
        immutable_directories.extend(
            (
                authorization.calibration_release.directory,
                authorization.calibration.directory,
                authorization.decision.directory,
            )
        )
    if any(
        _paths_overlap(_absolute(Path(path)), immutable_directory)
        for path in result_paths
        for immutable_directory in immutable_directories
    ):
        raise ScreeningResultError(
            "reviewer result paths must be disjoint from immutable authorization snapshots"
        )

    captured = _capture_distinct_inputs([Path(path) for path in result_paths])
    payloads_by_batch = _result_payloads_from_inputs(captured)
    _, manifest, _, _, _ = _validate_phase_payloads(
        coordinator,
        phase,
        payloads_by_batch,
        reviewer_release_sha256=release.snapshot_sha256,
    )
    artifacts = {
        f"{batch_id}.csv": payloads_by_batch[batch_id] for batch_id in BATCH_IDS
    }
    artifacts["manifest.csv"] = _csv_bytes(PHASE_RESULT_MANIFEST_HEADER, manifest)
    artifacts["SHA256SUMS"] = _checksums(artifacts)

    _reattest_trusted_coordinator(coordinator)
    _reattest_reviewer_release(release, coordinator)
    if authorization is not None:
        reattest_calibration_release_tuple(authorization)
    for item in captured:
        item.fingerprint.reattest()

    def post_publish_check() -> None:
        published = validate_phase_result_snapshot(
            Path(output_dir),
            coordinator_snapshot_dir=coordinator.directory,
            reviewer_release_snapshot_dir=release.directory,
            calibration_reviewer_release_snapshot_dir=(
                authorization.calibration_release.directory
                if authorization is not None
                else None
            ),
            calibration_result_snapshot_dir=(
                authorization.calibration.directory
                if authorization is not None
                else None
            ),
            calibration_decision_snapshot_dir=(
                authorization.decision.directory
                if authorization is not None
                else None
            ),
        )
        groups: list[Sequence[FileFingerprint]] = [
            published.fingerprints,
            tuple(item.fingerprint for item in captured),
            release.fingerprints,
        ]
        if authorization is not None:
            groups.extend(
                (
                    authorization.calibration.fingerprints,
                    authorization.decision.fingerprints,
                    authorization.calibration_release.fingerprints,
                )
            )
        _reattest_reviewer_release(release, coordinator)
        _coherent_trusted_final_attestation(tuple(groups), coordinator)
        _reattest_reviewer_release(release, coordinator)
        if authorization is not None:
            reattest_calibration_release_tuple(authorization)

    _publish(
        Path(output_dir),
        artifacts,
        post_publish_check=post_publish_check,
    )


def _capture_flat_snapshot(
    snapshot_dir: Path, expected_filenames: Sequence[str]
) -> tuple[dict[str, bytes], tuple[FileFingerprint, ...]]:
    directory = _absolute(Path(snapshot_dir))
    try:
        screening_batches._validate_version_path(directory, "snapshot directory")
        parent_fd, parent_identity = screening_batches._open_directory_fd(
            directory.parent, "snapshot parent directory"
        )
    except SnapshotError as exc:
        raise _raise_result_error(exc) from exc
    root_fd: int | None = None
    try:
        try:
            root_stat = os.stat(
                directory.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError as exc:
            raise ScreeningResultError(f"{directory}: snapshot is missing") from exc
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            raise ScreeningResultError(f"{directory}: snapshot must be a real directory")
        root_fd = os.open(
            directory.name, screening_batches._DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd
        )
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        opened = os.fstat(root_fd)
        if (opened.st_dev, opened.st_ino) != root_identity:
            raise ScreeningResultError(f"{directory}: snapshot changed before read")

        required = set(expected_filenames)
        actual = set(os.listdir(root_fd))
        if actual != required:
            raise ScreeningResultError(
                f"{directory}: snapshot entries mismatch; "
                f"missing={sorted(required - actual)}, "
                f"extra={sorted(actual - required)}"
            )
        root_attestation = screening_batches._attest_directory_fd(
            root_fd, required, "result snapshot", 0o755
        )
        tree_fingerprint = DirectoryFingerprint(
            path=directory,
            identity=root_attestation.identity,
            mode=root_attestation.mode,
            link_count=root_attestation.link_count,
            entries_sha256=root_attestation.entries_sha256,
            expected_names=tuple(sorted(required)),
        )

        payloads: dict[str, bytes] = {}
        first_reads = {}
        identities: dict[FileIdentity, str] = {}
        fingerprints: list[FileFingerprint] = []
        for filename in expected_filenames:
            read_file = screening_batches._read_regular_file_at(
                root_fd, filename, f"{directory / filename}"
            )
            if read_file.mode != 0o644:
                raise ScreeningResultError(
                    f"{directory / filename}: mode {read_file.mode:#o} != 0o644"
                )
            if read_file.identity in identities:
                raise ScreeningResultError(
                    f"{directory / filename}: aliases {identities[read_file.identity]}"
                )
            identities[read_file.identity] = filename
            first_reads[filename] = read_file
            payloads[filename] = read_file.payload
            fingerprints.append(
                FileFingerprint(
                    path=directory / filename,
                    identity=read_file.identity,
                    sha256=_sha256(read_file.payload),
                    size=len(read_file.payload),
                    mode=read_file.mode,
                    link_count=read_file.link_count,
                    tree=tree_fingerprint,
                )
            )

        screening_batches._assert_directory_unchanged(
            root_fd, root_attestation, required, "result snapshot"
        )
        for filename in expected_filenames:
            second = screening_batches._read_regular_file_at(
                root_fd, filename, f"{directory / filename}"
            )
            first = first_reads[filename]
            if (
                second.identity != first.identity
                or second.mode != first.mode
                or second.link_count != first.link_count
                or _sha256(second.payload) != _sha256(first.payload)
            ):
                raise ScreeningResultError(
                    f"{directory / filename}: file changed after capture"
                )
        screening_batches._assert_directory_unchanged(
            root_fd, root_attestation, required, "result snapshot"
        )
        screening_batches._recheck_directory_path(
            directory.parent, parent_identity, "snapshot parent directory"
        )
        if screening_batches._identity_at(parent_fd, directory.name) != root_identity:
            raise ScreeningResultError(f"{directory}: snapshot changed after capture")
        return payloads, tuple(fingerprints)
    except SnapshotError as exc:
        raise _raise_result_error(exc) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


@_normalize_os_errors
def validate_phase_result_snapshot(
    snapshot_dir: Path,
    *,
    reviewer_release_snapshot_dir: Path,
    coordinator_snapshot_dir: Path | None = None,
    coordinator: CoordinatorSnapshot | None = None,
    calibration_reviewer_release_snapshot_dir: Path | None = None,
    calibration_result_snapshot_dir: Path | None = None,
    calibration_decision_snapshot_dir: Path | None = None,
) -> PhaseResultSnapshot:
    if (coordinator_snapshot_dir is None) == (coordinator is None):
        raise ScreeningResultError(
            "phase validation requires exactly one authoritative coordinator "
            "snapshot or capture"
        )
    if coordinator is None:
        assert coordinator_snapshot_dir is not None
        coordinator = _capture_coordinator(Path(coordinator_snapshot_dir))
    else:
        coordinator = _recapture_coordinator(coordinator)

    expected_files = (*RESULT_FILENAMES, "manifest.csv", "SHA256SUMS")
    payloads, fingerprints = _capture_flat_snapshot(
        Path(snapshot_dir), expected_files
    )
    checksum_inputs = {
        name: payload for name, payload in payloads.items() if name != "SHA256SUMS"
    }
    if payloads["SHA256SUMS"] != _checksums(checksum_inputs):
        raise ScreeningResultError("phase result snapshot checksum mismatch")
    manifest = _parse_csv(
        payloads["manifest.csv"],
        "phase result manifest.csv",
        PHASE_RESULT_MANIFEST_HEADER,
    )
    if len(manifest) != len(BATCH_IDS):
        raise ScreeningResultError("phase result manifest must contain six rows")
    phases = {row["phase"] for row in manifest}
    if len(phases) != 1:
        raise ScreeningResultError("phase result manifest has inconsistent phases")
    phase = next(iter(phases))
    coordinator_values = {
        row["coordinator_snapshot_sha256"] for row in manifest
    }
    protocol_values = {row["protocol_sha256"] for row in manifest}
    release_values = {row["reviewer_release_sha256"] for row in manifest}
    if (
        len(coordinator_values) != 1
        or len(protocol_values) != 1
        or len(release_values) != 1
    ):
        raise ScreeningResultError(
            "phase result manifest has inconsistent authorization bindings"
        )
    sealed_release_hash = next(iter(release_values))
    if _LOWER_SHA256.fullmatch(sealed_release_hash) is None:
        raise ScreeningResultError(
            "phase result manifest reviewer release digest is not canonical"
        )

    gate_inputs = (
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
    )
    authorization: CalibrationReleaseTuple | None = None
    if phase == "calibration":
        if any(value is not None for value in gate_inputs):
            raise ScreeningResultError(
                "calibration phase validation does not accept calibration gate inputs"
            )
        expected_release_manifest = _expected_reviewer_release_manifest(
            coordinator,
            "calibration",
        )
    elif phase == "main":
        if any(value is None for value in gate_inputs):
            raise ScreeningResultError(
                "main phase validation requires calibration reviewer release, "
                "result, and decision snapshots"
            )
        assert calibration_reviewer_release_snapshot_dir is not None
        assert calibration_result_snapshot_dir is not None
        assert calibration_decision_snapshot_dir is not None
        authorization = capture_calibration_release_tuple(
            coordinator.directory,
            calibration_reviewer_release_snapshot_dir,
            calibration_result_snapshot_dir,
            calibration_decision_snapshot_dir,
        )
        gate = authorization.decision.decision
        if (
            gate["decision"] != "release"
            or gate["systematic_ambiguity"] != "false"
            or Decimal(gate["status_agreement"]) < Decimal("0.80")
        ):
            raise ScreeningResultError(
                "main phase validation requires a passing calibration release decision"
            )
        expected_release_manifest = _expected_reviewer_release_manifest(
            coordinator,
            "main",
            calibration_result_snapshot_sha256=(
                authorization.calibration.snapshot_sha256
            ),
            calibration_decision_snapshot_sha256=(
                authorization.decision.snapshot_sha256
            ),
        )
    else:
        raise ScreeningResultError(f"unsupported screening phase {phase!r}")

    release = _capture_reviewer_release(
        Path(reviewer_release_snapshot_dir),
        expected_manifest=expected_release_manifest,
        coordinator=coordinator,
    )
    if release.snapshot_sha256 != sealed_release_hash:
        raise ScreeningResultError(
            "phase result manifest does not bind the authoritative reviewer release"
        )
    if _paths_overlap(_absolute(Path(snapshot_dir)), release.directory):
        raise ScreeningResultError(
            "phase result snapshot must be disjoint from reviewer release"
        )

    payloads_by_batch = {
        batch_id: payloads[f"{batch_id}.csv"] for batch_id in BATCH_IDS
    }
    rows, expected_manifest, snapshot_sha256, coordinator_hash, protocol_hash = (
        _validate_phase_payloads(
            coordinator,
            phase,
            payloads_by_batch,
            reviewer_release_sha256=release.snapshot_sha256,
            sealed_coordinator_hash=next(iter(coordinator_values)),
            sealed_protocol_hash=next(iter(protocol_values)),
        )
    )
    released_rows: list[Row] = []
    for batch_id in BATCH_IDS:
        released_rows.extend(
            _parse_csv(
                release.payloads[f"packets/{batch_id}.csv"],
                f"reviewer release packets/{batch_id}.csv",
                screening_batches.PACKET_HEADER,
                no_blank_cells=False,
            )
        )
    _validate_reviewer_release_assignments(rows, released_rows)

    if manifest != expected_manifest:
        raise ScreeningResultError(
            "phase result manifest does not match captured result bytes"
        )
    if payloads["manifest.csv"] != _csv_bytes(
        PHASE_RESULT_MANIFEST_HEADER, expected_manifest
    ):
        raise ScreeningResultError("phase result manifest is not canonical")
    captured = PhaseResultSnapshot(
        directory=_absolute(Path(snapshot_dir)),
        phase=phase,
        rows=tuple(rows),
        snapshot_sha256=snapshot_sha256,
        coordinator_snapshot_sha256=coordinator_hash,
        protocol_sha256=protocol_hash,
        reviewer_release_sha256=release.snapshot_sha256,
        manifest=tuple(manifest),
        fingerprints=fingerprints,
    )
    groups: list[Sequence[FileFingerprint]] = [
        captured.fingerprints,
        release.fingerprints,
    ]
    if authorization is not None:
        groups.extend(
            (
                authorization.calibration_release.fingerprints,
                authorization.calibration.fingerprints,
                authorization.decision.fingerprints,
            )
        )
    _reattest_reviewer_release(release, coordinator)
    _coherent_trusted_final_attestation(tuple(groups), coordinator)
    _reattest_reviewer_release(release, coordinator)
    if authorization is not None:
        reattest_calibration_release_tuple(authorization)
    return captured


def _calibration_statistics(rows: Sequence[Row]) -> tuple[int, int, list[str], list[str]]:
    by_candidate: defaultdict[str, list[Row]] = defaultdict(list)
    assignment_ids: list[str] = []
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)
        assignment_ids.append(row["assignment_id"])
    if len(by_candidate) != 30 or len(rows) != 60:
        raise ScreeningResultError(
            "calibration result snapshot must contain 30 candidates and 60 ratings"
        )
    for candidate_id, ratings in by_candidate.items():
        if len(ratings) != 2:
            raise ScreeningResultError(
                f"calibration candidate {candidate_id} must have exactly two ratings"
            )
    numerator = sum(
        ratings[0]["screening_status"] == ratings[1]["screening_status"]
        for ratings in by_candidate.values()
    )
    return numerator, 30, list(by_candidate), assignment_ids


def _decision_identifier_preimages(
    coordinator: _Coordinator,
    calibration: PhaseResultSnapshot,
) -> tuple[bytes, bytes]:
    _, _, result_candidate_ids, assignment_ids = _calibration_statistics(
        calibration.rows
    )
    if set(result_candidate_ids) != set(coordinator.calibration_candidate_ids):
        raise ScreeningResultError(
            "calibration result candidates do not match frozen selection"
        )
    return (
        _identifier_preimage(
            coordinator.calibration_candidate_ids,
            sort_utf8=False,
        ),
        _identifier_preimage(
            assignment_ids,
            sort_utf8=True,
        ),
    )


def _validate_calibration_decision(
    row: Row,
    coordinator: _Coordinator,
    calibration: PhaseResultSnapshot,
) -> None:
    (
        numerator,
        denominator,
        result_candidate_ids,
        assignment_ids,
    ) = _calibration_statistics(calibration.rows)
    if set(result_candidate_ids) != set(coordinator.calibration_candidate_ids):
        raise ScreeningResultError(
            "calibration result candidates do not match frozen selection"
        )
    expected = {
        "protocol_sha256": coordinator.protocol_sha256,
        "coordinator_snapshot_sha256": coordinator.snapshot_sha256,
        "calibration_result_snapshot_sha256": calibration.snapshot_sha256,
        "candidate_ids_sha256": sequence_ids_sha256(
            coordinator.calibration_candidate_ids
        ),
        "assignment_ids_sha256": ordered_ids_sha256(assignment_ids),
        "status_agreement_numerator": str(numerator),
        "status_agreement_denominator": str(denominator),
        "status_agreement": canonical_ratio(numerator, denominator),
    }
    for field, expected_value in expected.items():
        if row[field] != expected_value:
            raise ScreeningResultError(
                f"calibration decision {field} does not match derived value "
                f"{expected_value!r}"
            )
    if _IDENTIFIER.fullmatch(row["decision_id"]) is None:
        raise ScreeningResultError("calibration decision_id is invalid")
    if row["systematic_ambiguity"] not in {"true", "false"}:
        raise ScreeningResultError(
            "calibration systematic_ambiguity must be 'true' or 'false'"
        )
    if row["decision"] not in {"release", "revise"}:
        raise ScreeningResultError("calibration decision must be release or revise")
    release_allowed = (
        Fraction(numerator, denominator) >= Fraction(4, 5)
        and row["systematic_ambiguity"] == "false"
    )
    required_decision = "release" if release_allowed else "revise"
    if row["decision"] != required_decision:
        raise ScreeningResultError(
            f"calibration decision must be {required_decision!r} for the derived gate"
        )
    _validate_iso_date(
        row["decided_on"], field="decided_on", context="calibration decision"
    )
    makers = row["decision_makers"].split(";")
    if (
        row["decision_makers"] == "NR"
        or "accountable-author" not in makers
        or len(makers) != len(set(makers))
        or any(_IDENTIFIER.fullmatch(maker) is None for maker in makers)
        or makers != sorted(makers, key=lambda maker: maker.encode("utf-8"))
    ):
        raise ScreeningResultError(
            "calibration decision_makers must include accountable-author and "
            "contain distinct stable IDs in UTF-8 order"
        )
    evidence = row["resolution_evidence"]
    if evidence == "NR" or len(evidence) < 40 or " " not in evidence:
        raise ScreeningResultError(
            "calibration resolution_evidence must be substantive"
        )


def _decision_manifest(
    coordinator: _Coordinator,
    calibration: PhaseResultSnapshot,
    decision_payload: bytes,
    decision: Row,
    candidate_ids_payload: bytes | None = None,
    assignment_ids_payload: bytes | None = None,
) -> tuple[Row, str]:
    if candidate_ids_payload is None or assignment_ids_payload is None:
        candidate_ids_payload, assignment_ids_payload = (
            _decision_identifier_preimages(coordinator, calibration)
        )
    decision_hash = _sha256(decision_payload)
    candidate_ids_hash = _sha256(candidate_ids_payload)
    assignment_ids_hash = _sha256(assignment_ids_payload)
    snapshot_hash = _canonical_sha256(
        {
            "assignment_ids_file_sha256": assignment_ids_hash,
            "calibration_result_snapshot_sha256": calibration.snapshot_sha256,
            "candidate_ids_file_sha256": candidate_ids_hash,
            "coordinator_snapshot_sha256": coordinator.snapshot_sha256,
            "decision_file_sha256": decision_hash,
            "decision_id": decision["decision_id"],
            "manifest_version": MANIFEST_VERSION,
            "protocol_sha256": coordinator.protocol_sha256,
            "row_count": 1,
        }
    )
    return (
        {
            "manifest_version": MANIFEST_VERSION,
            "calibration_decision_snapshot_sha256": snapshot_hash,
            "protocol_sha256": coordinator.protocol_sha256,
            "coordinator_snapshot_sha256": coordinator.snapshot_sha256,
            "calibration_result_snapshot_sha256": calibration.snapshot_sha256,
            "decision_id": decision["decision_id"],
            "decision_file_sha256": decision_hash,
            "candidate_ids_file_sha256": candidate_ids_hash,
            "assignment_ids_file_sha256": assignment_ids_hash,
            "row_count": "1",
        },
        snapshot_hash,
    )


@_normalize_os_errors
def seal_calibration_decision(
    coordinator_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    decision_input_path: Path,
    output_dir: Path,
    *,
    calibration_reviewer_release_snapshot_dir: Path,
) -> None:
    coordinator = _capture_coordinator(Path(coordinator_snapshot_dir))
    calibration_release = _capture_reviewer_release(
        Path(calibration_reviewer_release_snapshot_dir),
        expected_manifest=_expected_reviewer_release_manifest(
            coordinator,
            "calibration",
        ),
        coordinator=coordinator,
    )
    calibration = validate_phase_result_snapshot(
        Path(calibration_result_snapshot_dir),
        coordinator=coordinator,
        reviewer_release_snapshot_dir=calibration_release.directory,
    )
    if calibration.phase != "calibration":
        raise ScreeningResultError(
            "calibration decision requires a calibration phase result snapshot"
        )
    _reject_output_overlap(
        Path(output_dir),
        [
            coordinator.directory,
            calibration_release.directory,
            calibration.directory,
            Path(decision_input_path),
        ],
    )
    decision_input = _capture_input(Path(decision_input_path), "calibration decision")
    if any(
        _paths_overlap(decision_input.fingerprint.path, directory)
        for directory in (
            coordinator.directory,
            calibration_release.directory,
            calibration.directory,
        )
    ):
        raise ScreeningResultError(
            "calibration decision input must be disjoint from immutable snapshots"
        )
    decisions = _parse_csv(
        decision_input.payload,
        str(decision_input.fingerprint.path),
        CALIBRATION_DECISION_HEADER,
    )
    if len(decisions) != 1:
        raise ScreeningResultError(
            "calibration decision input must contain exactly one row"
        )
    decision = decisions[0]
    _validate_calibration_decision(decision, coordinator, calibration)
    candidate_ids_payload, assignment_ids_payload = (
        _decision_identifier_preimages(coordinator, calibration)
    )
    manifest, _ = _decision_manifest(
        coordinator,
        calibration,
        decision_input.payload,
        decision,
        candidate_ids_payload,
        assignment_ids_payload,
    )
    artifacts = {
        "decision.csv": decision_input.payload,
        "candidate_ids.txt": candidate_ids_payload,
        "assignment_ids.txt": assignment_ids_payload,
        "manifest.csv": _csv_bytes(
            CALIBRATION_DECISION_MANIFEST_HEADER, [manifest]
        ),
    }
    artifacts["SHA256SUMS"] = _checksums(artifacts)

    _reattest_trusted_coordinator(coordinator)
    _reattest_reviewer_release(calibration_release, coordinator)
    for fingerprint in calibration.fingerprints:
        fingerprint.reattest()
    decision_input.fingerprint.reattest()

    def post_publish_check() -> None:
        published = validate_calibration_decision_snapshot(
            Path(output_dir),
            coordinator_snapshot_dir=coordinator.directory,
            calibration_reviewer_release_snapshot_dir=(
                calibration_release.directory
            ),
            calibration_result_snapshot_dir=calibration.directory,
        )
        _reattest_reviewer_release(calibration_release, coordinator)
        _coherent_trusted_final_attestation(
            (
                published.fingerprints,
                (decision_input.fingerprint,),
                calibration.fingerprints,
                calibration_release.fingerprints,
            ),
            coordinator,
        )
        _reattest_reviewer_release(calibration_release, coordinator)

    _publish(
        Path(output_dir),
        artifacts,
        post_publish_check=post_publish_check,
    )


def _validate_calibration_decision_snapshot(
    snapshot_dir: Path,
    coordinator: _Coordinator,
    calibration: PhaseResultSnapshot,
) -> CalibrationDecisionSnapshot:
    if calibration.phase != "calibration":
        raise ScreeningResultError(
            "calibration decision must bind a calibration result snapshot"
        )
    expected_files = (
        "decision.csv",
        "candidate_ids.txt",
        "assignment_ids.txt",
        "manifest.csv",
        "SHA256SUMS",
    )
    payloads, fingerprints = _capture_flat_snapshot(
        Path(snapshot_dir), expected_files
    )
    if payloads["SHA256SUMS"] != _checksums(
        {name: payload for name, payload in payloads.items() if name != "SHA256SUMS"}
    ):
        raise ScreeningResultError("calibration decision checksum mismatch")
    expected_candidate_ids, expected_assignment_ids = (
        _decision_identifier_preimages(coordinator, calibration)
    )
    if payloads["candidate_ids.txt"] != expected_candidate_ids:
        raise ScreeningResultError(
            "candidate_ids.txt does not match authoritative frozen selection"
        )
    if payloads["assignment_ids.txt"] != expected_assignment_ids:
        raise ScreeningResultError(
            "assignment_ids.txt does not match authoritative calibration results"
        )
    decisions = _parse_csv(
        payloads["decision.csv"],
        "calibration decision.csv",
        CALIBRATION_DECISION_HEADER,
    )
    if len(decisions) != 1:
        raise ScreeningResultError(
            "calibration decision snapshot must contain exactly one decision"
        )
    decision = decisions[0]
    _validate_calibration_decision(decision, coordinator, calibration)
    manifest_rows = _parse_csv(
        payloads["manifest.csv"],
        "calibration decision manifest.csv",
        CALIBRATION_DECISION_MANIFEST_HEADER,
    )
    if len(manifest_rows) != 1:
        raise ScreeningResultError(
            "calibration decision manifest must contain exactly one row"
        )
    expected_manifest, snapshot_hash = _decision_manifest(
        coordinator,
        calibration,
        payloads["decision.csv"],
        decision,
        payloads["candidate_ids.txt"],
        payloads["assignment_ids.txt"],
    )
    if manifest_rows[0] != expected_manifest or payloads["manifest.csv"] != _csv_bytes(
        CALIBRATION_DECISION_MANIFEST_HEADER, [expected_manifest]
    ):
        raise ScreeningResultError(
            "calibration decision manifest does not match captured decision"
        )
    captured = CalibrationDecisionSnapshot(
        directory=_absolute(Path(snapshot_dir)),
        decision=decision,
        snapshot_sha256=snapshot_hash,
        coordinator_snapshot_sha256=coordinator.snapshot_sha256,
        calibration_result_snapshot_sha256=calibration.snapshot_sha256,
        manifest=manifest_rows[0],
        fingerprints=fingerprints,
    )
    _coherent_trusted_final_attestation(
        (captured.fingerprints, calibration.fingerprints),
        coordinator,
    )
    return captured


@_normalize_os_errors
def validate_calibration_decision_snapshot(
    snapshot_dir: Path,
    *,
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
) -> CalibrationDecisionSnapshot:
    coordinator = _capture_coordinator(Path(coordinator_snapshot_dir))
    calibration_release = _capture_reviewer_release(
        Path(calibration_reviewer_release_snapshot_dir),
        expected_manifest=_expected_reviewer_release_manifest(
            coordinator,
            "calibration",
        ),
        coordinator=coordinator,
    )
    calibration = validate_phase_result_snapshot(
        Path(calibration_result_snapshot_dir),
        coordinator=coordinator,
        reviewer_release_snapshot_dir=calibration_release.directory,
    )
    captured = _validate_calibration_decision_snapshot(
        Path(snapshot_dir),
        coordinator,
        calibration,
    )
    _reattest_reviewer_release(calibration_release, coordinator)
    _coherent_trusted_final_attestation(
        (
            captured.fingerprints,
            calibration.fingerprints,
            calibration_release.fingerprints,
        ),
        coordinator,
    )
    _reattest_reviewer_release(calibration_release, coordinator)
    return captured


def _fingerprint_signature(
    fingerprints: Sequence[FileFingerprint],
) -> list[dict[str, object]]:
    return [
        {
            "identity": list(fingerprint.identity),
            "link_count": fingerprint.link_count,
            "mode": fingerprint.mode,
            "path": str(fingerprint.path),
            "sha256": fingerprint.sha256,
            "size": fingerprint.size,
        }
        for fingerprint in fingerprints
    ]


def _release_tuple_signature(captured: CalibrationReleaseTuple) -> str:
    coordinator = captured.coordinator
    calibration_release = captured.calibration_release
    calibration = captured.calibration
    decision = captured.decision
    return _canonical_sha256(
        {
            "calibration_release": {
                "fingerprints": _fingerprint_signature(
                    calibration_release.fingerprints
                ),
                "manifest": calibration_release.manifest,
                "payload_sha256": _coordinator_payload_hashes(
                    calibration_release.payloads
                ),
                "phase": calibration_release.phase,
                "snapshot_sha256": calibration_release.snapshot_sha256,
                "root": {
                    "entries_sha256": calibration_release.root.entries_sha256,
                    "identity": list(calibration_release.root.identity),
                    "link_count": calibration_release.root.link_count,
                    "mode": calibration_release.root.mode,
                    "path": str(calibration_release.root.path),
                },
                "packets_tree": {
                    "entries_sha256": (
                        calibration_release.packets_tree.entries_sha256
                    ),
                    "identity": list(
                        calibration_release.packets_tree.identity
                    ),
                    "link_count": (
                        calibration_release.packets_tree.link_count
                    ),
                    "mode": calibration_release.packets_tree.mode,
                    "path": str(calibration_release.packets_tree.path),
                },
            },
            "calibration": {
                "coordinator_snapshot_sha256": (
                    calibration.coordinator_snapshot_sha256
                ),
                "fingerprints": _fingerprint_signature(
                    calibration.fingerprints
                ),
                "manifest": list(calibration.manifest),
                "phase": calibration.phase,
                "protocol_sha256": calibration.protocol_sha256,
                "reviewer_release_sha256": (
                    calibration.reviewer_release_sha256
                ),
                "rows": list(calibration.rows),
                "snapshot_sha256": calibration.snapshot_sha256,
            },
            "coordinator": {
                "allowed_screening_statuses": list(
                    coordinator.allowed_screening_statuses
                ),
                "calibration_candidate_ids": list(
                    coordinator.calibration_candidate_ids
                ),
                "fingerprints": _fingerprint_signature(
                    coordinator.fingerprints
                ),
                "manifest": list(coordinator.manifest),
                "payload_sha256": _coordinator_payload_hashes(
                    coordinator.payloads
                ),
                "protocol_sha256": coordinator.protocol_sha256,
                "snapshot_sha256": coordinator.snapshot_sha256,
                "tree": {
                    "entries_sha256": coordinator.tree.entries_sha256,
                    "identity": list(coordinator.tree.identity),
                    "link_count": coordinator.tree.link_count,
                    "mode": coordinator.tree.mode,
                    "path": str(coordinator.tree.path),
                },
            },
            "decision": {
                "calibration_result_snapshot_sha256": (
                    decision.calibration_result_snapshot_sha256
                ),
                "coordinator_snapshot_sha256": (
                    decision.coordinator_snapshot_sha256
                ),
                "decision": decision.decision,
                "fingerprints": _fingerprint_signature(
                    decision.fingerprints
                ),
                "manifest": decision.manifest,
                "snapshot_sha256": decision.snapshot_sha256,
            },
        }
    )


@_normalize_os_errors
def capture_calibration_release_tuple(
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
) -> CalibrationReleaseTuple:
    coordinator = _capture_coordinator(Path(coordinator_snapshot_dir))
    calibration_release = _capture_reviewer_release(
        Path(calibration_reviewer_release_snapshot_dir),
        expected_manifest=_expected_reviewer_release_manifest(
            coordinator,
            "calibration",
        ),
        coordinator=coordinator,
    )
    calibration = validate_phase_result_snapshot(
        Path(calibration_result_snapshot_dir),
        coordinator=coordinator,
        reviewer_release_snapshot_dir=calibration_release.directory,
    )
    if calibration.phase != "calibration":
        raise ScreeningResultError(
            "calibration release tuple requires calibration results"
        )
    _reattest_reviewer_release(calibration_release, coordinator)
    _coherent_trusted_final_attestation(
        (calibration_release.fingerprints, calibration.fingerprints),
        coordinator,
    )
    decision = _validate_calibration_decision_snapshot(
        Path(calibration_decision_snapshot_dir),
        coordinator,
        calibration,
    )
    captured = CalibrationReleaseTuple(
        coordinator=coordinator,
        calibration_release=calibration_release,
        calibration=calibration,
        decision=decision,
    )
    _reattest_reviewer_release(calibration_release, coordinator)
    _coherent_trusted_final_attestation(
        (
            decision.fingerprints,
            calibration.fingerprints,
            calibration_release.fingerprints,
        ),
        coordinator,
    )
    _reattest_reviewer_release(calibration_release, coordinator)
    return captured


@_normalize_os_errors
def reattest_calibration_release_tuple(
    captured: CalibrationReleaseTuple,
) -> None:
    authoritative_coordinator = _recapture_coordinator(captured.coordinator)
    expected_signature = _release_tuple_signature(captured)
    _reattest_reviewer_release(
        captured.calibration_release,
        authoritative_coordinator,
    )
    _coherent_trusted_final_attestation(
        (
            captured.decision.fingerprints,
            captured.calibration.fingerprints,
            captured.calibration_release.fingerprints,
        ),
        authoritative_coordinator,
    )
    fresh = capture_calibration_release_tuple(
        authoritative_coordinator.directory,
        captured.calibration_release.directory,
        captured.calibration.directory,
        captured.decision.directory,
    )
    if _release_tuple_signature(fresh) != expected_signature:
        raise ScreeningResultError(
            "calibration release tuple changed after capture"
        )
    _reattest_reviewer_release(
        fresh.calibration_release,
        fresh.coordinator,
    )
    _coherent_trusted_final_attestation(
        (
            fresh.decision.fingerprints,
            fresh.calibration.fingerprints,
            fresh.calibration_release.fingerprints,
        ),
        fresh.coordinator,
    )
    _coherent_trusted_final_attestation(
        (
            captured.calibration_release.fingerprints,
            captured.calibration.fingerprints,
            captured.decision.fingerprints,
        ),
        authoritative_coordinator,
    )
    _reattest_reviewer_release(
        captured.calibration_release,
        authoritative_coordinator,
    )


@_normalize_os_errors
def _validate_role_result(
    reviewer_stage: Path,
    supplied_result_path: str,
) -> None:
    stage = _producer_call(
        screening_batches.validate_reviewer_stage_snapshot,
        reviewer_stage,
    )
    stage_manifest = _parse_csv(
        stage["stage_manifest.csv"],
        "stage_manifest.csv",
        screening_batches.STAGE_MANIFEST_HEADER,
    )
    if len(stage_manifest) != 1:
        raise ScreeningResultError("stage manifest must contain exactly one row")
    manifest = stage_manifest[0]
    configuration = json.loads(stage["execution_configuration.json"])
    configuration_version = configuration["configuration_version"]
    if configuration_version == "1":
        allowed_inclusion_criteria = INCLUSION_CRITERIA
    elif configuration_version == "2":
        allowed_inclusion_criteria = tuple(
            configuration["allowed_inclusion_criteria"]
        )
    else:
        raise ScreeningResultError(
            "unsupported execution configuration version"
        )
    allowed_screening_statuses = LEGACY_SCREENING_STATUSES
    if supplied_result_path != manifest["result_path"]:
        raise ScreeningResultError(
            "supplied result path does not match stage result path"
        )

    result_path = Path(supplied_result_path)
    payload = result_path.read_bytes()
    rows = _parse_csv(payload, str(result_path), RESULT_HEADER)
    if not rows:
        raise ScreeningResultError("result must contain at least one row")
    if payload != _csv_bytes(RESULT_HEADER, rows):
        raise ScreeningResultError("result must use canonical CSV bytes")

    packet_rows = screening_batches._read_csv_bytes(
        stage["packet.csv"],
        "packet.csv",
        screening_batches.PACKET_HEADER,
    )
    immutable_fields = (
        "assignment_id",
        "phase",
        "candidate_id",
        "input_sha256",
        "snapshot_sha256",
        "batch_id",
    )
    packet_assignments = [row["assignment_id"] for row in packet_rows]
    result_assignments = [row["assignment_id"] for row in rows]
    if result_assignments != packet_assignments:
        raise ScreeningResultError("result assignment coverage or order is invalid")
    for row_number, (result, packet) in enumerate(
        zip(rows, packet_rows),
        start=2,
    ):
        context = f"{result_path}:{row_number}"
        for field in immutable_fields:
            if result[field] != packet[field]:
                raise ScreeningResultError(
                    f"{context}: {field} does not match packet.csv"
                )
        if result["coder_id"] != manifest["role_id"]:
            raise ScreeningResultError(
                f"{context}: coder_id does not match stage role_id"
            )
        _validate_result_decision(
            result,
            context=context,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal immutable screening phase results and calibration gates."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--seal-phase", action="store_true")
    modes.add_argument("--seal-calibration-decision", action="store_true")
    modes.add_argument("--validate-role-result", action="store_true")
    parser.add_argument("--reviewer-stage", type=Path)
    parser.add_argument("--coordinator-snapshot", type=Path)
    parser.add_argument("--reviewer-release-snapshot", type=Path)
    parser.add_argument("--phase", choices=("calibration", "main"))
    parser.add_argument("--result", action="append")
    parser.add_argument("--calibration-reviewer-release-snapshot", type=Path)
    parser.add_argument("--calibration-result-snapshot", type=Path)
    parser.add_argument("--calibration-decision-snapshot", type=Path)
    parser.add_argument("--decision-input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _require_arguments(arguments, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(arguments, name) is None]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ScreeningResultError(f"missing required arguments: {options}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if arguments.validate_role_result:
        _require_arguments(arguments, ("reviewer_stage", "result"))
        if len(arguments.result) != 1:
            raise ScreeningResultError(
                "--validate-role-result requires exactly one --result"
            )
        forbidden = (
            "coordinator_snapshot",
            "reviewer_release_snapshot",
            "phase",
            "calibration_reviewer_release_snapshot",
            "calibration_result_snapshot",
            "calibration_decision_snapshot",
            "decision_input",
            "output_dir",
        )
        if any(getattr(arguments, name) is not None for name in forbidden):
            raise ScreeningResultError(
                "--validate-role-result only accepts --reviewer-stage and --result"
            )
        _validate_role_result(arguments.reviewer_stage, arguments.result[0])
        return 0

    if arguments.seal_phase:
        _require_arguments(
            arguments,
            (
                "coordinator_snapshot",
                "reviewer_release_snapshot",
                "phase",
                "result",
                "output_dir",
            ),
        )
        if arguments.reviewer_stage is not None:
            raise ScreeningResultError(
                "--seal-phase does not accept --reviewer-stage"
            )
        if arguments.decision_input is not None:
            raise ScreeningResultError(
                "--seal-phase does not accept --decision-input"
            )
        gate_names = (
            "calibration_reviewer_release_snapshot",
            "calibration_result_snapshot",
            "calibration_decision_snapshot",
        )
        if arguments.phase == "calibration":
            if any(getattr(arguments, name) is not None for name in gate_names):
                raise ScreeningResultError(
                    "calibration --seal-phase does not accept calibration gate snapshots"
                )
        else:
            _require_arguments(arguments, gate_names)
        seal_phase_results(
            arguments.coordinator_snapshot,
            arguments.phase,
            [Path(path) for path in arguments.result],
            arguments.output_dir,
            reviewer_release_snapshot_dir=(
                arguments.reviewer_release_snapshot
            ),
            calibration_reviewer_release_snapshot_dir=(
                arguments.calibration_reviewer_release_snapshot
            ),
            calibration_result_snapshot_dir=(
                arguments.calibration_result_snapshot
            ),
            calibration_decision_snapshot_dir=(
                arguments.calibration_decision_snapshot
            ),
        )
        return 0

    _require_arguments(
        arguments,
        (
            "coordinator_snapshot",
            "reviewer_release_snapshot",
            "calibration_result_snapshot",
            "decision_input",
            "output_dir",
        ),
    )
    if arguments.reviewer_stage is not None:
        raise ScreeningResultError(
            "--seal-calibration-decision does not accept --reviewer-stage"
        )
    if (
        arguments.phase is not None
        or arguments.result is not None
        or arguments.calibration_reviewer_release_snapshot is not None
        or arguments.calibration_decision_snapshot is not None
    ):
        raise ScreeningResultError(
            "--seal-calibration-decision does not accept phase or main-gate arguments"
        )
    seal_calibration_decision(
        arguments.coordinator_snapshot,
        arguments.calibration_result_snapshot,
        arguments.decision_input,
        arguments.output_dir,
        calibration_reviewer_release_snapshot_dir=(
            arguments.reviewer_release_snapshot
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
