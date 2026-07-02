from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import re
import stat
import unicodedata
import secrets
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class SnapshotError(ValueError):
    """The screening snapshot or one of its source inputs is invalid."""


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


MANIFEST_VERSION = "2"
BATCH_COUNT = 6
CANDIDATE_COUNT = 202
ASSIGNMENT_COUNT = 404
CALIBRATION_CANDIDATE_COUNT = 30
CALIBRATION_ASSIGNMENT_COUNT = 60
MAIN_CANDIDATE_COUNT = 172
MAIN_ASSIGNMENT_COUNT = 344
RANKING_SALT = "trackgen-screening-calibration-v1"
REVIEWER_PAIRS = (
    ("screening-01", "screening-02"),
    ("screening-01", "screening-03"),
    ("screening-01", "screening-04"),
    ("screening-02", "screening-05"),
    ("screening-02", "screening-06"),
    ("screening-03", "screening-05"),
    ("screening-04", "screening-06"),
    ("screening-01", "screening-05"),
    ("screening-01", "screening-06"),
    ("screening-02", "screening-03"),
    ("screening-02", "screening-04"),
    ("screening-03", "screening-04"),
    ("screening-03", "screening-06"),
    ("screening-04", "screening-05"),
    ("screening-05", "screening-06"),
)
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")
_CANDIDATE_ID_PATTERN = re.compile(r"C[0-9]{4,}")
_ASSIGNMENT_ID_PATTERN = re.compile(r"A-C[0-9]{4,}-0[1-6]")
_CITE_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._/+\-]*")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*")
_YEAR_PATTERN = re.compile(r"[0-9]{4}")
_DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}/[a-z0-9._;()/:+\-]+")
STABLE_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}"
)
_REQUESTED_MODEL_VERSION_PREFIX = "requested:"
_MAX_EXECUTION_PROFILE_BYTES = 64 * 1024
_MAX_EXECUTION_PROFILE_DEPTH = 16
_MAX_EXECUTION_PROFILE_NODES = 1024
_PROVIDER_LIMITATION_VALUE = "provider-not-exposed"
_PROVIDER_LIMITATION_KEYS = frozenset(
    {
        "backend_model_version",
        "decoding_parameters",
        "developer_instruction_bytes",
        "retrieval_cache_isolation",
        "system_instruction_bytes",
    }
)
_HOST_SECURITY_BOUNDARY = (
    "shared-same-user-host-no-acl-container-mount-guarantee"
)

CANDIDATE_HEADER = (
    "candidate_id",
    "cite_key",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
    "discovery_stream",
    "discovery_query",
    "discovery_agent",
    "screening_status",
    "exclusion_reason",
    "metadata_status",
    "metadata_evidence",
)
CONFLICT_HEADER = (
    "conflict_id",
    "record_type",
    "record_key",
    "field",
    "value_a",
    "value_b",
    "resolution",
    "resolver",
    "resolution_evidence",
)
BIBLIOGRAPHY_HEADER = (
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
CITATION_KEY_HEADER = ("candidate_id", "cite_key")
CALIBRATION_SELECTION_HEADER = ("candidate_id",)
EVIDENCE_HEADER = (
    "cite_key",
    "domain",
    "vehicle",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "geometry_metrics",
    "difficulty_metrics",
    "diversity_metrics",
    "training_distribution",
    "evaluation_suite",
    "simulator",
    "export_format",
    "code_status",
    "asset_status",
    "reproducibility_fields",
    "evidence_locator",
    "coding_notes",
)
MANIFEST_HEADER = (
    "manifest_version",
    "snapshot_sha256",
    "protocol_sha256",
    "execution_profile_sha256",
    "prompt_template_sha256",
    "assignment_id",
    "batch_id",
    "phase",
    "candidate_id",
    "cite_key",
    "input_sha256",
    "weight",
)
PACKET_HEADER = (
    "assignment_id",
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "batch_id",
    "phase",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
RELEASE_MANIFEST_HEADER = (
    "manifest_version",
    "phase",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "execution_profile_sha256",
    "prompt_template_sha256",
    "assignment_count",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
)
SCREENING_INCLUSION_CRITERION_KEY = "screening_inclusion_criterion"
CURRENT_INCLUSION_CRITERIA = ("include-relevant",)
SCREENING_RESULT_STATUS_KEY = "screening_result_status"
CURRENT_SCREENING_RESULT_STATUSES = ("included", "excluded")

RAW_FILENAMES = (
    "candidates.csv",
    "conflicts.csv",
    "bibliography.csv",
    "citation_keys.csv",
    "taxonomy.json",
    "protocol.md",
    "execution_profile.json",
    "reviewer_prompt_template.md",
)
ROOT_FILENAMES = frozenset(
    (
        *RAW_FILENAMES,
        "calibration_selection.csv",
        "manifest.csv",
        "SHA256SUMS",
        "packets",
    )
)
PACKET_FILENAMES = tuple(
    f"screening-{number:02d}.csv" for number in range(1, BATCH_COUNT + 1)
)
REVIEWER_RELEASE_ROOT_FILENAMES = frozenset(
    {
        "execution_profile.json",
        "protocol.md",
        "release_manifest.csv",
        "reviewer_prompt_template.md",
        "SHA256SUMS",
        "packets",
    }
)
STAGE_MANIFEST_HEADER = (
    "manifest_version",
    "stage_snapshot_sha256",
    "reviewer_release_sha256",
    "coordinator_snapshot_sha256",
    "phase",
    "task",
    "role_id",
    "protocol_sha256",
    "packet_sha256",
    "execution_profile_sha256",
    "prompt_template_sha256",
    "configuration_sha256",
    "prompt_sha256",
    "user_instruction_sha256",
    "stage_path",
    "result_path",
    "assignment_count",
)
REVIEWER_STAGE_ROOT_FILENAMES = frozenset(
    {
        "execution_configuration.json",
        "execution_profile.json",
        "packet.csv",
        "protocol.md",
        "reviewer_prompt.md",
        "reviewer_prompt_template.md",
        "stage_manifest.csv",
        "SHA256SUMS",
    }
)
_ROLE_PRIVATE_PARENT_PATTERN = re.compile(
    r"(screening-0[1-6])-[0-9a-f]{32}"
)
PACKET_FIELDS_FROM_CANDIDATE = (
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "title",
    "screening_status",
    "metadata_status",
)
REQUIRED_CONFLICT_FIELDS = (
    "conflict_id",
    "record_type",
    "record_key",
    "field",
    "value_a",
    "value_b",
)
REQUIRED_BIBLIOGRAPHY_FIELDS = (
    "candidate_id",
    "cite_key",
    "entry_type",
    "key_author",
    "authors",
    "author_kinds",
    "title",
)
SUPPORTED_ENTRY_TYPES = frozenset(
    {"article", "inproceedings", "misc", "techreport", "book"}
)
VENUE_FIELD_BY_ENTRY_TYPE = {
    "article": "journal",
    "inproceedings": "booktitle",
    "misc": "howpublished",
    "techreport": "institution",
    "book": "publisher",
}
AUTHOR_KINDS = frozenset({"personal", "corporate"})

SOURCE_CLASSES = (
    "standard-specification",
    "competition",
    "benchmark-dataset",
    "software",
    "scholarly",
    "official-other",
)
DECISION_HEADER = (
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
DECISION_MANIFEST_HEADER = (
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
DECISION_ROOT_FILENAMES = frozenset(
    {
        "decision.csv",
        "candidate_ids.txt",
        "assignment_ids.txt",
        "manifest.csv",
        "SHA256SUMS",
    }
)
HEX_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

Row = dict[str, str]
FileIdentity = tuple[int, int]


@dataclass(frozen=True)
class _ReadFile:
    path: Path
    identity: FileIdentity
    payload: bytes
    mode: int = 0
    link_count: int = 1


@dataclass(frozen=True)
class _DirectoryAttestation:
    identity: FileIdentity
    mode: int
    link_count: int
    entries_sha256: str


@dataclass(frozen=True)
class _SourceData:
    candidates: list[Row]
    conflicts_by_candidate: dict[str, list[Row]]
    bibliography_by_candidate: dict[str, Row]
    citation_keys_by_candidate: dict[str, Row]
    raw_hashes: dict[str, str]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise SnapshotError("canonical JSON serialization failed") from exc
    return (serialized + "\n").encode("utf-8")


def _validate_json_shape(value: object, *, label: str) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_EXECUTION_PROFILE_NODES:
            raise SnapshotError(f"{label} exceeds the JSON node limit")
        if depth > _MAX_EXECUTION_PROFILE_DEPTH:
            raise SnapshotError(f"{label} exceeds the JSON depth limit")
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                raise SnapshotError(f"{label} JSON object keys must be strings")
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def validate_execution_profile(payload: bytes) -> dict[str, object]:
    """Validate and return one canonical pre-freeze execution profile."""

    if not isinstance(payload, bytes):
        raise SnapshotError("execution profile must be supplied as bytes")
    if len(payload) > _MAX_EXECUTION_PROFILE_BYTES:
        raise SnapshotError("execution profile exceeds the byte limit")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant!r}")

    try:
        text = payload.decode("utf-8")
        profile = json.loads(text, parse_constant=reject_constant)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise SnapshotError("execution profile must be canonical UTF-8 JSON") from exc
    if not isinstance(profile, dict):
        raise SnapshotError("execution profile must be a JSON object")
    _validate_json_shape(profile, label="execution profile")
    if payload != _canonical_json_bytes(profile):
        raise SnapshotError("execution profile must use canonical JSON bytes")

    expected_fields = {
        "decoding_parameters",
        "developer_instruction",
        "model_identifier",
        "model_version",
        "profile_version",
        "provider",
        "provider_metadata_limitations",
        "retrieval_configuration",
        "runtime",
        "system_instruction",
        "tool_configuration",
    }
    if set(profile) != expected_fields:
        raise SnapshotError("execution profile fields do not match the closed schema")
    if profile["profile_version"] != "1":
        raise SnapshotError("execution profile_version must be '1'")
    for field in ("model_identifier", "provider", "runtime"):
        if (
            not isinstance(profile[field], str)
            or STABLE_IDENTIFIER_PATTERN.fullmatch(profile[field]) is None
        ):
            raise SnapshotError(
                f"execution profile {field} must be a stable identifier"
            )

    limitations = profile["provider_metadata_limitations"]
    if not isinstance(limitations, dict):
        raise SnapshotError(
            "execution profile provider_metadata_limitations must be an object"
        )
    unknown = set(limitations) - _PROVIDER_LIMITATION_KEYS
    if unknown:
        raise SnapshotError(
            "execution profile has unknown provider limitation keys: "
            f"{sorted(unknown)}"
        )
    if any(value != _PROVIDER_LIMITATION_VALUE for value in limitations.values()):
        raise SnapshotError(
            "execution profile provider limitations must use exact "
            f"value {_PROVIDER_LIMITATION_VALUE!r}"
        )

    model_version = profile["model_version"]
    if not isinstance(model_version, str):
        raise SnapshotError("execution profile model_version must be a string")
    requested = model_version.startswith(_REQUESTED_MODEL_VERSION_PREFIX)
    backend_limited = "backend_model_version" in limitations
    if requested != backend_limited:
        raise SnapshotError(
            "backend_model_version limitation must be present exactly when "
            "model_version uses requested:<alias-or-date>"
        )
    identifier = (
        model_version.removeprefix(_REQUESTED_MODEL_VERSION_PREFIX)
        if requested
        else model_version
    )
    if STABLE_IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise SnapshotError(
            "execution profile model_version identifier is invalid"
        )

    conditional_values = {
        "system_instruction_bytes": "system_instruction",
        "developer_instruction_bytes": "developer_instruction",
        "decoding_parameters": "decoding_parameters",
    }
    for limitation, field in conditional_values.items():
        unavailable = profile[field] is None
        if unavailable != (limitation in limitations):
            raise SnapshotError(
                f"{limitation} limitation does not match {field} availability"
            )
    for field in ("system_instruction", "developer_instruction"):
        value = profile[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise SnapshotError(f"execution profile {field} must be nonempty text")
    decoding = profile["decoding_parameters"]
    if decoding is not None and not isinstance(decoding, dict):
        raise SnapshotError(
            "execution profile decoding_parameters must be an object or null"
        )

    tool_configuration = profile["tool_configuration"]
    if not isinstance(tool_configuration, dict):
        raise SnapshotError("execution profile tool_configuration must be an object")
    expected_tool_fields = {
        "filesystem_policy",
        "fork_context",
        "host_security_boundary",
        "model",
        "reasoning_effort",
        "staging_isolation",
        "web_retrieval_policy",
    }
    if set(tool_configuration) != expected_tool_fields:
        raise SnapshotError("execution profile tool_configuration fields are invalid")
    if tool_configuration["model"] != profile["model_identifier"]:
        raise SnapshotError("execution profile spawn model must match model_identifier")
    if tool_configuration["reasoning_effort"] not in {
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        raise SnapshotError("execution profile reasoning_effort is invalid")
    fixed_tool_configuration = {
        "filesystem_policy": "immutable-stage-read-role-result-write",
        "fork_context": False,
        "host_security_boundary": _HOST_SECURITY_BOUNDARY,
        "staging_isolation": "procedural-role-private-path",
        "web_retrieval_policy": "public-only",
    }
    if any(
        tool_configuration[key] != value
        for key, value in fixed_tool_configuration.items()
    ):
        raise SnapshotError(
            "execution profile tool_configuration does not match the "
            "procedural shared-host contract"
        )
    retrieval_configuration = profile["retrieval_configuration"]
    if not isinstance(retrieval_configuration, dict):
        raise SnapshotError("execution profile retrieval_configuration must be an object")
    expected_retrieval = {
        "fresh_context": True,
        "provider_retrieval_cache_isolation": (
            _PROVIDER_LIMITATION_VALUE
            if "retrieval_cache_isolation" in limitations
            else "isolated"
        ),
        "public_retrieval_only": True,
        "ratings_supplied": False,
        "results_supplied": False,
        "shared_conversation_history": False,
        "shared_memory": False,
    }
    if retrieval_configuration != expected_retrieval:
        raise SnapshotError(
            "execution profile retrieval_configuration does not match the "
            "fresh public-only context contract"
        )
    return profile



_PROMPT_EXECUTION_PLACEHOLDERS = frozenset(
    {
        "PACKET_PATH",
        "PACKET_SHA256",
        "PROTOCOL_PATH",
        "PROTOCOL_SHA256",
        "ROLE_ID",
        "OUTPUT_PATH",
        "STAGE_PATH",
    }
)
_PROMPT_COMPLETION_PLACEHOLDERS = frozenset(
    {"OUTPUT_SHA256", "ROWS_WRITTEN"}
)


def validate_reviewer_prompt_template(payload: bytes) -> str:
    """Validate and return the exact pre-freeze reviewer prompt template."""

    if not isinstance(payload, bytes) or len(payload) > 64 * 1024:
        raise SnapshotError("reviewer prompt template is not bounded bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotError("reviewer prompt template must be UTF-8") from exc
    if not text.endswith("\n") or "\r" in text:
        raise SnapshotError(
            "reviewer prompt template must use LF and end with one newline"
        )
    placeholders = re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)
    expected = _PROMPT_EXECUTION_PLACEHOLDERS | _PROMPT_COMPLETION_PLACEHOLDERS
    if set(placeholders) != expected:
        raise SnapshotError(
            "reviewer prompt template placeholders do not match the closed schema"
        )
    if text.count("{{") != len(placeholders) or text.count("}}") != len(placeholders):
        raise SnapshotError("reviewer prompt template has malformed placeholders")
    return text

def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(payload)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_fd(path: Path, label: str) -> tuple[int, FileIdentity]:
    lexical = _absolute_lexical(path)
    descriptor = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in lexical.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise SnapshotError(
                        f"{path}: {label} contains a symlink or non-directory component"
                    ) from exc
                if exc.errno == errno.ENOENT:
                    raise SnapshotError(f"{path}: {label} is missing") from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        file_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(file_stat.st_mode):
            raise SnapshotError(f"{path}: {label} must be a real directory")
        return descriptor, (file_stat.st_dev, file_stat.st_ino)
    except BaseException:
        os.close(descriptor)
        raise


def _recheck_directory_path(
    path: Path, expected_identity: FileIdentity, label: str
) -> None:
    try:
        descriptor, identity = _open_directory_fd(path, label)
    except SnapshotError as exc:
        raise SnapshotError(
            f"{path}: {label} changed during operation"
        ) from exc
    os.close(descriptor)
    if identity != expected_identity:
        raise SnapshotError(f"{path}: {label} changed during operation")


def _read_regular_file_at(
    directory_fd: int, name: str, label: str
) -> _ReadFile:
    if not name or "/" in name or name in {".", ".."}:
        raise SnapshotError(f"{label}: invalid descriptor-relative filename")
    try:
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label}: file is missing") from exc
    if stat.S_ISLNK(path_stat.st_mode):
        raise SnapshotError(f"{label}: file must not be a symlink")
    if not stat.S_ISREG(path_stat.st_mode):
        raise SnapshotError(f"{label}: file must be a regular file")
    if path_stat.st_nlink != 1:
        raise SnapshotError(f"{label}: file must not have a hard link alias")

    try:
        descriptor = os.open(name, _FILE_READ_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SnapshotError(f"{label}: file must not be a symlink") from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError(f"{label}: file must be a regular file")
        if before.st_nlink != 1:
            raise SnapshotError(f"{label}: file must not have a hard link alias")
        if (before.st_dev, before.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise SnapshotError(f"{label}: file changed before read")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        final_path_stat = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False
        )
    except FileNotFoundError as exc:
        raise SnapshotError(f"{label}: file changed while being read") from exc
    if (
        _stat_fingerprint(before) != _stat_fingerprint(after)
        or _stat_fingerprint(after) != _stat_fingerprint(final_path_stat)
    ):
        raise SnapshotError(f"{label}: file changed while being read")
    return _ReadFile(
        path=Path(label),
        identity=(after.st_dev, after.st_ino),
        payload=b"".join(chunks),
        mode=stat.S_IMODE(after.st_mode),
        link_count=after.st_nlink,
    )


def _directory_entries_sha256(directory_fd: int) -> str:
    entries = []
    for name in sorted(os.listdir(directory_fd)):
        file_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        entries.append(
            [
                name,
                stat.S_IFMT(file_stat.st_mode),
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_nlink,
            ]
        )
    return _canonical_sha256(entries)


def _attest_directory_fd(
    directory_fd: int,
    expected_names: set[str] | frozenset[str],
    label: str,
    expected_mode: int,
) -> _DirectoryAttestation:
    file_stat = os.fstat(directory_fd)
    if not stat.S_ISDIR(file_stat.st_mode):
        raise SnapshotError(f"{label}: must be a real directory")
    mode = stat.S_IMODE(file_stat.st_mode)
    if mode != expected_mode:
        raise SnapshotError(
            f"{label}: directory mode {mode:#o} != {expected_mode:#o}"
        )
    actual_names = set(os.listdir(directory_fd))
    if actual_names != set(expected_names):
        missing = sorted(set(expected_names) - actual_names)
        extra = sorted(actual_names - set(expected_names))
        raise SnapshotError(
            f"{label}: entries mismatch; missing={missing}, extra={extra}"
        )
    return _DirectoryAttestation(
        identity=(file_stat.st_dev, file_stat.st_ino),
        mode=mode,
        link_count=file_stat.st_nlink,
        entries_sha256=_directory_entries_sha256(directory_fd),
    )


def _assert_directory_unchanged(
    directory_fd: int,
    expected: _DirectoryAttestation,
    expected_names: set[str] | frozenset[str],
    label: str,
) -> None:
    actual = _attest_directory_fd(
        directory_fd, expected_names, label, expected.mode
    )
    if actual != expected:
        raise SnapshotError(f"{label}: directory changed during operation")


def _require_no_symlink_components(
    path: Path,
    label: str,
    *,
    strict: bool,
) -> Path:
    lexical = _absolute_lexical(path)
    try:
        resolved = path.resolve(strict=strict)
    except FileNotFoundError as exc:
        raise SnapshotError(f"{path}: {label} is missing") from exc
    if resolved != lexical:
        raise SnapshotError(
            f"{path}: {label} must not contain a symlink path component"
        )
    return lexical


def _require_real_directory(path: Path, label: str) -> os.stat_result:
    lexical = _require_no_symlink_components(path, label, strict=True)
    try:
        file_stat = lexical.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"{path}: {label} is missing") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise SnapshotError(f"{path}: {label} must not be a symlink")
    if not stat.S_ISDIR(file_stat.st_mode):
        raise SnapshotError(f"{path}: {label} must be a real directory")
    return file_stat


def _stat_fingerprint(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_regular_file(path: Path, label: str) -> _ReadFile:
    lexical = _absolute_lexical(path)
    parent_fd, parent_identity = _open_directory_fd(
        lexical.parent, f"{label} parent directory"
    )
    try:
        read_file = _read_regular_file_at(parent_fd, lexical.name, str(path))
        _recheck_directory_path(
            lexical.parent,
            parent_identity,
            f"{label} parent directory",
        )
        final_stat = os.stat(
            lexical.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            (final_stat.st_dev, final_stat.st_ino) != read_file.identity
            or final_stat.st_nlink != read_file.link_count
        ):
            raise SnapshotError(f"{path}: {label} changed after read")
        return _ReadFile(
            path=lexical,
            identity=read_file.identity,
            payload=read_file.payload,
            mode=read_file.mode,
            link_count=read_file.link_count,
        )
    finally:
        os.close(parent_fd)


def _read_csv_bytes(
    payload: bytes,
    label: str,
    expected_header: tuple[str, ...],
) -> list[Row]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise SnapshotError(f"{label}: invalid UTF-8: {exc}") from exc
    handle = io.StringIO(text, newline="")
    reader = csv.DictReader(handle, strict=True)
    try:
        actual_header = tuple(reader.fieldnames or ())
        if actual_header != expected_header:
            raise SnapshotError(
                f"{label}: headers {actual_header!r} != {expected_header!r}"
            )
        rows = list(reader)
    except csv.Error as exc:
        raise SnapshotError(
            f"{label}:{reader.line_num}: CSV parse error: {exc}"
        ) from exc
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise SnapshotError(f"{label}:{row_number}: malformed CSV row")
        if not any(value.strip() for value in row.values()):
            raise SnapshotError(f"{label}:{row_number}: row is entirely blank")
        for field, value in row.items():
            if "\x00" in value:
                raise SnapshotError(
                    f"{label}:{row_number}: {field} contains a NUL byte"
                )
    return rows


def _required(
    label: str,
    row_number: int,
    row: Row,
    field: str,
) -> str:
    value = row[field].strip()
    if not value:
        raise SnapshotError(f"{label}:{row_number}: {field} is required")
    return value


def _require_trimmed(
    label: str,
    row_number: int,
    row: Row,
    fields: Sequence[str],
) -> None:
    for field in fields:
        if row[field] != row[field].strip():
            raise SnapshotError(
                f"{label}:{row_number}: {field} contains surrounding whitespace"
            )


def _resolve_inclusion_criteria(
    taxonomy: dict[str, list[str]],
    *,
    strict_new: bool,
) -> tuple[str, ...] | None:
    if SCREENING_INCLUSION_CRITERION_KEY not in taxonomy:
        if strict_new:
            raise SnapshotError(
                "taxonomy.json: taxonomy is missing "
                f"{SCREENING_INCLUSION_CRITERION_KEY!r}"
            )
        return None
    criteria = tuple(taxonomy[SCREENING_INCLUSION_CRITERION_KEY])
    if criteria != CURRENT_INCLUSION_CRITERIA:
        raise SnapshotError(
            "taxonomy.json: screening_inclusion_criterion must equal "
            "[\"include-relevant\"]"
        )
    return criteria


def _resolve_screening_result_statuses(
    taxonomy: dict[str, list[str]],
    *,
    strict_new: bool,
) -> tuple[str, ...] | None:
    if SCREENING_RESULT_STATUS_KEY not in taxonomy:
        if strict_new:
            raise SnapshotError(
                "taxonomy.json: taxonomy is missing "
                f"{SCREENING_RESULT_STATUS_KEY!r}"
            )
        return None
    statuses = taxonomy[SCREENING_RESULT_STATUS_KEY]
    if (
        not isinstance(statuses, list)
        or tuple(statuses) != CURRENT_SCREENING_RESULT_STATUSES
    ):
        raise SnapshotError(
            "taxonomy.json: screening_result_status must equal "
            "[\"included\", \"excluded\"]"
        )
    return tuple(statuses)

def _parse_taxonomy(
    payload: bytes,
    *,
    strict_new: bool,
) -> dict[str, list[str]]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"taxonomy.json: invalid taxonomy JSON: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise SnapshotError("taxonomy.json: taxonomy must be a nonempty object")
    _resolve_screening_result_statuses(value, strict_new=strict_new)
    for key, items in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
            or len(items) != len(set(items))
        ):
            raise SnapshotError(
                "taxonomy.json: taxonomy values must be nonempty unique "
                "string lists"
            )
    for required_key in ("screening_status", "metadata_status"):
        if required_key not in value:
            raise SnapshotError(
                f"taxonomy.json: taxonomy is missing {required_key!r}"
            )
    if "verified" not in value["metadata_status"]:
        raise SnapshotError(
            "taxonomy.json: metadata_status must contain 'verified'"
        )
    _resolve_inclusion_criteria(value, strict_new=strict_new)
    return value


def _validate_protocol(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise SnapshotError(f"protocol.md: invalid UTF-8: {exc}") from exc
    if not text.strip():
        raise SnapshotError("protocol.md: protocol must not be empty")
    if "\x00" in text:
        raise SnapshotError("protocol.md: protocol contains a NUL byte")


def _validate_candidates(
    rows: list[Row], taxonomy: dict[str, list[str]]
) -> tuple[dict[str, Row], dict[str, str]]:
    if len(rows) != CANDIDATE_COUNT:
        raise SnapshotError(
            "candidates.csv: must contain exactly "
            f"{CANDIDATE_COUNT} unique candidates, found {len(rows)} rows"
        )
    by_id: dict[str, Row] = {}
    cite_key_to_id: dict[str, str] = {}
    allowed_screening = set(taxonomy["screening_status"])
    for row_number, row in enumerate(rows, start=2):
        for field in REQUIRED_CANDIDATE_FIELDS:
            _required("candidates.csv", row_number, row, field)
        _require_trimmed(
            "candidates.csv",
            row_number,
            row,
            (
                "candidate_id",
                "cite_key",
                "title",
                "authors",
                "year",
                "venue",
                "doi",
                "url",
                "source_type",
                "screening_status",
                "metadata_status",
            ),
        )
        candidate_id = row["candidate_id"]
        if _CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
            raise SnapshotError(
                f"candidates.csv:{row_number}: candidate_id={candidate_id!r} "
                "must be C followed by at least four digits"
            )
        if candidate_id in by_id:
            raise SnapshotError(
                f"candidates.csv:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        if row["metadata_status"] != "verified":
            raise SnapshotError(
                f"candidates.csv:{row_number}: metadata_status must be "
                f"'verified', found {row['metadata_status']!r}"
            )
        if row["screening_status"] not in allowed_screening:
            raise SnapshotError(
                f"candidates.csv:{row_number}: screening_status="
                f"{row['screening_status']!r} is not in taxonomy"
            )
        cite_key = row["cite_key"]
        if cite_key:
            if _CITE_KEY_PATTERN.fullmatch(cite_key) is None:
                raise SnapshotError(
                    f"candidates.csv:{row_number}: cite_key={cite_key!r} "
                    "is not BibTeX-safe"
                )
            folded = cite_key.casefold()
            if folded in cite_key_to_id:
                raise SnapshotError(
                    f"candidates.csv:{row_number}: duplicate cite_key "
                    f"{cite_key!r}"
                )
            cite_key_to_id[folded] = candidate_id
        by_id[candidate_id] = row
    return by_id, cite_key_to_id


def _validate_citation_keys(
    rows: list[Row], candidates_by_id: dict[str, Row]
) -> dict[str, Row]:
    by_id: dict[str, Row] = {}
    seen_keys: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        _require_trimmed(
            "citation_keys.csv", row_number, row, CITATION_KEY_HEADER
        )
        candidate_id = _required(
            "citation_keys.csv", row_number, row, "candidate_id"
        )
        cite_key = _required(
            "citation_keys.csv", row_number, row, "cite_key"
        )
        if _CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
            raise SnapshotError(
                f"citation_keys.csv:{row_number}: invalid candidate_id "
                f"{candidate_id!r}"
            )
        if _CITE_KEY_PATTERN.fullmatch(cite_key) is None:
            raise SnapshotError(
                f"citation_keys.csv:{row_number}: invalid cite_key {cite_key!r}"
            )
        if candidate_id not in candidates_by_id:
            raise SnapshotError(
                f"citation_keys.csv:{row_number}: candidate_id="
                f"{candidate_id!r} does not exist in candidates.csv"
            )
        if candidate_id in by_id:
            raise SnapshotError(
                f"citation_keys.csv:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        folded = cite_key.casefold()
        if folded in seen_keys:
            raise SnapshotError(
                f"citation_keys.csv:{row_number}: duplicate cite_key "
                f"{cite_key!r}"
            )
        by_id[candidate_id] = row
        seen_keys[folded] = candidate_id
    return by_id


def _split_bibliography_list(
    value: str,
    *,
    field: str,
    row_number: int,
) -> list[str]:
    values = [item.strip() for item in value.split(";")]
    if any(not item for item in values):
        raise SnapshotError(
            f"bibliography.csv:{row_number}: {field} contains an empty "
            "semicolon element"
        )
    return values


def _validate_bibliography(
    rows: list[Row], candidates_by_id: dict[str, Row]
) -> dict[str, Row]:
    by_id: dict[str, Row] = {}
    seen_keys: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        _require_trimmed(
            "bibliography.csv", row_number, row, BIBLIOGRAPHY_HEADER
        )
        for field in REQUIRED_BIBLIOGRAPHY_FIELDS:
            _required("bibliography.csv", row_number, row, field)
        candidate_id = row["candidate_id"]
        cite_key = row["cite_key"]
        if candidate_id not in candidates_by_id:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: candidate_id="
                f"{candidate_id!r} does not exist in candidates.csv"
            )
        if candidate_id in by_id:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        folded = cite_key.casefold()
        if _CITE_KEY_PATTERN.fullmatch(cite_key) is None:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: cite_key={cite_key!r} "
                "is not BibTeX-safe"
            )
        if folded in seen_keys:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: duplicate cite_key "
                f"{cite_key!r}"
            )
        entry_type = row["entry_type"]
        if entry_type not in SUPPORTED_ENTRY_TYPES:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: unsupported entry_type "
                f"{entry_type!r}"
            )
        authors = _split_bibliography_list(
            row["authors"], field="authors", row_number=row_number
        )
        author_kinds = _split_bibliography_list(
            row["author_kinds"], field="author_kinds", row_number=row_number
        )
        if len(authors) != len(author_kinds):
            raise SnapshotError(
                f"bibliography.csv:{row_number}: author_kinds must align "
                "one-to-one with authors"
            )
        invalid_kinds = sorted(set(author_kinds) - AUTHOR_KINDS)
        if invalid_kinds:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: invalid author kind "
                f"{invalid_kinds[0]!r}"
            )
        venue = row["venue"]
        venue_field = row["venue_field"]
        expected_venue_field = VENUE_FIELD_BY_ENTRY_TYPE[entry_type]
        if venue and venue_field != expected_venue_field:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: {entry_type} requires "
                f"venue_field={expected_venue_field!r}"
            )
        if not venue and venue_field:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: venue_field must be empty "
                "when venue is empty"
            )
        year = row["year"]
        if year and _YEAR_PATTERN.fullmatch(year) is None:
            raise SnapshotError(
                f"bibliography.csv:{row_number}: year={year!r} must be "
                "four digits when present"
            )
        doi = row["doi"]
        if doi and (
            doi != doi.lower()
            or _DOI_PATTERN.fullmatch(doi) is None
        ):
            raise SnapshotError(
                f"bibliography.csv:{row_number}: doi={doi!r} is not canonical"
            )

        candidate = candidates_by_id[candidate_id]
        for field in (
            "cite_key",
            "title",
            "authors",
            "year",
            "venue",
            "doi",
        ):
            if row[field] != candidate[field]:
                raise SnapshotError(
                    f"bibliography.csv:{row_number}: candidate_id="
                    f"{candidate_id!r}: {field}={row[field]!r} does not "
                    f"match candidates.csv value {candidate[field]!r}"
                )
        if candidate["screening_status"] == "excluded":
            raise SnapshotError(
                f"bibliography.csv:{row_number}: excluded candidate "
                f"{candidate_id!r} must not have a bibliography row"
            )
        by_id[candidate_id] = row
        seen_keys[folded] = candidate_id

    expected = sorted(
        rows,
        key=lambda row: (
            row["cite_key"].casefold(),
            row["cite_key"],
            row["candidate_id"],
        ),
    )
    if rows != expected:
        raise SnapshotError(
            "bibliography.csv: rows are not in canonical cite_key order"
        )
    return by_id


def _validate_correspondence(
    candidates_by_id: dict[str, Row],
    bibliography_by_id: dict[str, Row],
    citation_keys_by_id: dict[str, Row],
) -> None:
    for candidate_id, candidate in candidates_by_id.items():
        candidate_key = candidate["cite_key"]
        ledger = citation_keys_by_id.get(candidate_id)
        bibliography = bibliography_by_id.get(candidate_id)
        if candidate_key:
            if ledger is None:
                raise SnapshotError(
                    f"candidates.csv: candidate_id={candidate_id!r} is "
                    "missing from citation_keys.csv"
                )
            if ledger["cite_key"] != candidate_key:
                raise SnapshotError(
                    f"candidates.csv: candidate_id={candidate_id!r} cite_key "
                    "does not match citation_keys.csv"
                )
        elif ledger is not None:
            raise SnapshotError(
                f"candidates.csv: candidate_id={candidate_id!r} has an "
                "empty cite_key but a citation_keys.csv row"
            )

        active = candidate["screening_status"] != "excluded"
        if active and not candidate_key:
            raise SnapshotError(
                f"candidates.csv: active candidate_id={candidate_id!r} "
                "has a genuinely unissued cite_key"
            )
        if active and bibliography is None:
            raise SnapshotError(
                f"candidates.csv: active candidate_id={candidate_id!r} is "
                "missing from bibliography.csv"
            )
        if not active and bibliography is not None:
            raise SnapshotError(
                f"candidates.csv: excluded candidate_id={candidate_id!r} "
                "must not appear in bibliography.csv"
            )
        if bibliography is not None:
            if ledger is None or bibliography["cite_key"] != ledger["cite_key"]:
                raise SnapshotError(
                    f"bibliography.csv: candidate_id={candidate_id!r} does "
                    "not match citation_keys.csv"
                )


def _validate_conflicts(
    rows: list[Row],
    candidates_by_id: dict[str, Row],
) -> dict[str, list[Row]]:
    by_cite_key: defaultdict[str, list[str]] = defaultdict(list)
    for candidate_id, candidate in candidates_by_id.items():
        if candidate["cite_key"]:
            by_cite_key[candidate["cite_key"]].append(candidate_id)

    grouped: defaultdict[str, list[Row]] = defaultdict(list)
    conflict_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        for field in REQUIRED_CONFLICT_FIELDS:
            _required("conflicts.csv", row_number, row, field)
        _require_trimmed(
            "conflicts.csv",
            row_number,
            row,
            ("conflict_id", "record_type", "record_key", "field"),
        )
        conflict_id = row["conflict_id"]
        if conflict_id in conflict_ids:
            raise SnapshotError(
                f"conflicts.csv:{row_number}: duplicate conflict_id "
                f"{conflict_id!r}"
            )
        conflict_ids.add(conflict_id)
        record_type = row["record_type"]
        record_key = row["record_key"]
        if record_type == "candidate":
            if record_key not in candidates_by_id:
                raise SnapshotError(
                    f"conflicts.csv:{row_number}: candidate record_key="
                    f"{record_key!r} does not resolve"
                )
            candidate_id = record_key
            target_header = CANDIDATE_HEADER
            target_name = "candidates.csv"
        elif record_type == "evidence":
            matching_ids = by_cite_key.get(record_key, [])
            if len(matching_ids) != 1:
                raise SnapshotError(
                    f"conflicts.csv:{row_number}: evidence record_key="
                    f"{record_key!r} does not resolve to one candidate"
                )
            candidate_id = matching_ids[0]
            target_header = EVIDENCE_HEADER
            target_name = "evidence.csv"
        else:
            raise SnapshotError(
                f"conflicts.csv:{row_number}: unsupported record_type "
                f"{record_type!r}"
            )
        if row["field"] not in target_header:
            raise SnapshotError(
                f"conflicts.csv:{row_number}: {record_type} field="
                f"{row['field']!r} is not a column in {target_name}"
            )
        if row["resolution"].strip():
            for required_field in ("resolver", "resolution_evidence"):
                _required(
                    "conflicts.csv", row_number, row, required_field
                )
        grouped[candidate_id].append(row)
    return dict(grouped)


def _load_source_data(
    raw_payloads: dict[str, bytes],
    *,
    strict_new: bool,
) -> _SourceData:
    taxonomy = _parse_taxonomy(
        raw_payloads["taxonomy.json"],
        strict_new=strict_new,
    )
    _validate_protocol(raw_payloads["protocol.md"])
    validate_execution_profile(raw_payloads["execution_profile.json"])
    validate_reviewer_prompt_template(
        raw_payloads["reviewer_prompt_template.md"]
    )
    candidates = _read_csv_bytes(
        raw_payloads["candidates.csv"],
        "candidates.csv",
        CANDIDATE_HEADER,
    )
    conflicts = _read_csv_bytes(
        raw_payloads["conflicts.csv"],
        "conflicts.csv",
        CONFLICT_HEADER,
    )
    bibliography = _read_csv_bytes(
        raw_payloads["bibliography.csv"],
        "bibliography.csv",
        BIBLIOGRAPHY_HEADER,
    )
    citation_keys = _read_csv_bytes(
        raw_payloads["citation_keys.csv"],
        "citation_keys.csv",
        CITATION_KEY_HEADER,
    )
    candidates_by_id, _ = _validate_candidates(candidates, taxonomy)
    citation_keys_by_id = _validate_citation_keys(
        citation_keys, candidates_by_id
    )
    bibliography_by_id = _validate_bibliography(
        bibliography, candidates_by_id
    )
    _validate_correspondence(
        candidates_by_id,
        bibliography_by_id,
        citation_keys_by_id,
    )
    conflicts_by_candidate = _validate_conflicts(
        conflicts, candidates_by_id
    )
    return _SourceData(
        candidates=candidates,
        conflicts_by_candidate=conflicts_by_candidate,
        bibliography_by_candidate=bibliography_by_id,
        citation_keys_by_candidate=citation_keys_by_id,
        raw_hashes={name: _sha256(raw_payloads[name]) for name in RAW_FILENAMES},
    )


def _candidate_input_sha256(
    candidate: Row,
    conflicts: list[Row],
    bibliography: Row | None,
    citation_key: Row | None,
    *,
    taxonomy_sha256: str,
    protocol_sha256: str,
) -> str:
    ordered_conflicts = sorted(
        conflicts,
        key=lambda row: tuple(row[field] for field in CONFLICT_HEADER),
    )
    return _canonical_sha256(
        {
            "bibliography": bibliography,
            "candidate": candidate,
            "citation_key": citation_key,
            "conflicts": ordered_conflicts,
            "protocol_sha256": protocol_sha256,
            "taxonomy_sha256": taxonomy_sha256,
        }
    )


def _packet_metadata(candidate: Row) -> dict[str, str]:
    return {
        field: candidate[field] if candidate[field] else "NR"
        for field in PACKET_FIELDS_FROM_CANDIDATE
    }


def _csv_bytes(
    header: tuple[str, ...], rows: list[Row]
) -> bytes:
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


def _normalize_metadata_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    pieces: list[str] = []
    separating = False
    for character in normalized:
        if character.isalnum():
            pieces.append(character)
            separating = False
        elif pieces and not separating:
            pieces.append(" ")
            separating = True
    return "".join(pieces).strip()


def _coarse_source_class(source_type: str) -> str:
    value = _normalize_metadata_token(source_type)
    if any(
        marker in value
        for marker in ("standard", "specification", "file format")
    ):
        return "standard-specification"
    if "competition" in value:
        return "competition"
    if any(marker in value for marker in ("benchmark", "dataset")):
        return "benchmark-dataset"
    if any(
        marker in value
        for marker in (
            "software",
            "repository",
            "simulator",
            "platform",
            "package",
            "game",
            "engine",
            "tool",
        )
    ):
        return "software"
    if any(
        marker in value
        for marker in (
            "article",
            "paper",
            "preprint",
            "chapter",
            "thesis",
            "report",
            "survey",
        )
    ):
        return "scholarly"
    return "official-other"


def _discovery_labels(candidate: Row) -> set[str]:
    labels: set[str] = set()
    for field, prefix in (
        ("discovery_stream", "stream:"),
        ("discovery_query", "query:"),
    ):
        for raw_label in candidate[field].split(";"):
            label = _normalize_metadata_token(raw_label)
            if label:
                labels.add(prefix + label)
    return labels


def _stable_candidate_rank(candidate_id: str) -> tuple[str, bytes]:
    payload = (
        RANKING_SALT.encode("utf-8")
        + b"\0"
        + candidate_id.encode("utf-8")
    )
    return _sha256(payload), candidate_id.encode("utf-8")


def _select_calibration_candidate_ids(
    candidates: Sequence[Row],
    target_size: int = CALIBRATION_CANDIDATE_COUNT,
) -> tuple[str, ...]:
    if target_size != CALIBRATION_CANDIDATE_COUNT:
        raise SnapshotError("calibration target size must be exactly 30")
    if len(candidates) != CANDIDATE_COUNT:
        raise SnapshotError(
            "calibration selection requires exactly "
            f"{CANDIDATE_COUNT} candidates, found {len(candidates)}"
        )

    grouped = {
        source_class: [
            candidate
            for candidate in candidates
            if _coarse_source_class(candidate["source_type"]) == source_class
        ]
        for source_class in SOURCE_CLASSES
    }
    populated = [
        source_class for source_class in SOURCE_CLASSES if grouped[source_class]
    ]
    quotas = {
        source_class: min(2, len(grouped[source_class]))
        for source_class in populated
    }
    remaining = target_size - sum(quotas.values())
    if remaining < 0:
        raise SnapshotError("calibration class minimums exceed target size")
    capacities = {
        source_class: len(grouped[source_class]) - quotas[source_class]
        for source_class in populated
    }
    capacity_total = sum(capacities.values())
    if remaining and capacity_total < remaining:
        raise SnapshotError("calibration target exceeds candidate capacity")

    remainders: list[tuple[int, int, str]] = []
    if remaining:
        for class_index, source_class in enumerate(SOURCE_CLASSES):
            if source_class not in quotas:
                continue
            increment, remainder = divmod(
                remaining * capacities[source_class], capacity_total
            )
            quotas[source_class] += increment
            remainders.append((remainder, -class_index, source_class))
        unallocated = target_size - sum(quotas.values())
        for _, _, source_class in sorted(remainders, reverse=True)[:unallocated]:
            quotas[source_class] += 1

    selected: set[str] = set()
    for source_class in SOURCE_CLASSES:
        quota = quotas.get(source_class, 0)
        pool = list(grouped[source_class])
        seen_labels: set[str] = set()
        for _ in range(quota):
            chosen = min(
                pool,
                key=lambda candidate: (
                    -len(
                        _discovery_labels(candidate) - seen_labels
                    ),
                    *_stable_candidate_rank(candidate["candidate_id"]),
                ),
            )
            pool.remove(chosen)
            selected.add(chosen["candidate_id"])
            seen_labels.update(_discovery_labels(chosen))
    if len(selected) != target_size:
        raise SnapshotError(
            f"calibration selection produced {len(selected)} records, "
            f"expected {target_size}"
        )
    return tuple(
        sorted(selected, key=_stable_candidate_rank)
    )


def _screening_rank(
    candidate_id: str, protocol_sha256: str | None = None
) -> tuple[str, bytes]:
    del protocol_sha256
    return _stable_candidate_rank(candidate_id)


def build_snapshot_artifacts(
    raw_payloads: dict[str, bytes],
    *,
    strict_new: bool = False,
) -> dict[str, bytes]:
    """Build every snapshot file in memory from exact raw source bytes."""
    if set(raw_payloads) != set(RAW_FILENAMES):
        missing = sorted(set(RAW_FILENAMES) - set(raw_payloads))
        extra = sorted(set(raw_payloads) - set(RAW_FILENAMES))
        raise SnapshotError(
            f"raw snapshot inputs mismatch; missing={missing}, extra={extra}"
        )
    source = _load_source_data(raw_payloads, strict_new=strict_new)
    calibration_ids = _select_calibration_candidate_ids(source.candidates)
    records: list[dict[str, object]] = []
    calibration_id_set = set(calibration_ids)
    for candidate in source.candidates:
        candidate_id = candidate["candidate_id"]
        input_sha256 = _candidate_input_sha256(
            candidate,
            source.conflicts_by_candidate.get(candidate_id, []),
            source.bibliography_by_candidate.get(candidate_id),
            source.citation_keys_by_candidate.get(candidate_id),
            taxonomy_sha256=source.raw_hashes["taxonomy.json"],
            protocol_sha256=source.raw_hashes["protocol.md"],
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "cite_key": candidate["cite_key"],
                "input_sha256": input_sha256,
                "weight": 1,
                "metadata": _packet_metadata(candidate),
            }
        )

    ranked = sorted(
        records,
        key=lambda record: _screening_rank(
            str(record["candidate_id"]),
            source.raw_hashes["protocol.md"],
        ),
    )
    assignments: list[dict[str, object]] = []
    for index, record in enumerate(ranked):
        reviewer_pair = REVIEWER_PAIRS[index % len(REVIEWER_PAIRS)]
        for batch_id in reviewer_pair:
            candidate_id = str(record["candidate_id"])
            assignments.append(
                {
                    **record,
                    "assignment_id": (
                        f"A-{candidate_id}-{batch_id.removeprefix('screening-')}"
                    ),
                    "batch_id": batch_id,
                    "phase": (
                        "calibration"
                        if candidate_id in calibration_id_set
                        else "main"
                    ),
                    "reviewer_pair": reviewer_pair,
                }
            )

    if len(records) != CANDIDATE_COUNT or len(assignments) != ASSIGNMENT_COUNT:
        raise SnapshotError(
            "canonical assignment derivation did not produce exactly "
            f"{CANDIDATE_COUNT} candidates and {ASSIGNMENT_COUNT} assignments"
        )
    calibration_assignment_count = sum(
        record["phase"] == "calibration" for record in assignments
    )
    if calibration_assignment_count != CALIBRATION_ASSIGNMENT_COUNT:
        raise SnapshotError(
            "canonical assignment derivation did not produce exactly "
            f"{CALIBRATION_ASSIGNMENT_COUNT} calibration assignments"
        )

    ordered = sorted(
        assignments,
        key=lambda record: (
            str(record["batch_id"]),
            str(record["candidate_id"]),
            str(record["assignment_id"]),
        ),
    )

    derivation_assignments = []
    for record in ordered:
        packet_input = {
            "assignment_id": record["assignment_id"],
            "candidate_id": record["candidate_id"],
            "input_sha256": record["input_sha256"],
            "batch_id": record["batch_id"],
            "phase": record["phase"],
            **record["metadata"],
        }
        derivation_assignments.append(
            {
                "assignment_id": record["assignment_id"],
                "batch_id": record["batch_id"],
                "phase": record["phase"],
                "candidate_id": record["candidate_id"],
                "cite_key": record["cite_key"],
                "input_sha256": record["input_sha256"],
                "reviewer_pair": list(record["reviewer_pair"]),
                "packet": packet_input,
                "weight": record["weight"],
            }
        )
    snapshot_sha256 = _canonical_sha256(
        {
            "assignments": derivation_assignments,
            "calibration_selection": list(calibration_ids),
            "manifest_version": MANIFEST_VERSION,
            "raw_files": [
                {"name": name, "sha256": source.raw_hashes[name]}
                for name in RAW_FILENAMES
            ],
        }
    )

    manifest_rows: list[Row] = []
    packet_rows: dict[str, list[Row]] = {
        filename: [] for filename in PACKET_FILENAMES
    }
    for record in ordered:
        batch_id = str(record["batch_id"])
        manifest_rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "snapshot_sha256": snapshot_sha256,
                "protocol_sha256": source.raw_hashes["protocol.md"],
                "execution_profile_sha256": source.raw_hashes[
                    "execution_profile.json"
                ],
                "prompt_template_sha256": source.raw_hashes[
                    "reviewer_prompt_template.md"
                ],
                "assignment_id": str(record["assignment_id"]),
                "batch_id": batch_id,
                "phase": str(record["phase"]),
                "candidate_id": str(record["candidate_id"]),
                "cite_key": str(record["cite_key"]),
                "input_sha256": str(record["input_sha256"]),
                "weight": str(record["weight"]),
            }
        )
        packet_rows[f"{batch_id}.csv"].append(
            {
                "assignment_id": str(record["assignment_id"]),
                "candidate_id": str(record["candidate_id"]),
                "input_sha256": str(record["input_sha256"]),
                "snapshot_sha256": snapshot_sha256,
                "batch_id": batch_id,
                "phase": str(record["phase"]),
                **record["metadata"],
            }
        )

    artifacts = dict(raw_payloads)
    artifacts["calibration_selection.csv"] = _csv_bytes(
        CALIBRATION_SELECTION_HEADER,
        [{"candidate_id": candidate_id} for candidate_id in calibration_ids],
    )
    artifacts["manifest.csv"] = _csv_bytes(MANIFEST_HEADER, manifest_rows)
    for filename in PACKET_FILENAMES:
        artifacts[f"packets/{filename}"] = _csv_bytes(
            PACKET_HEADER, packet_rows[filename]
        )
    checksum_paths = sorted(artifacts)
    artifacts["SHA256SUMS"] = "".join(
        f"{_sha256(artifacts[relative])}  {relative}\n"
        for relative in checksum_paths
    ).encode("utf-8")
    return artifacts


def _release_manifest_row(
    *,
    phase: str,
    coordinator_snapshot_sha256: str,
    protocol_sha256: str,
    execution_profile_sha256: str,
    prompt_template_sha256: str,
    calibration_result_snapshot_sha256: str = "NR",
    calibration_decision_snapshot_sha256: str = "NR",
) -> Row:
    if phase not in {"calibration", "main"}:
        raise SnapshotError(f"unsupported reviewer release phase {phase!r}")
    if (
        HEX_SHA256_PATTERN.fullmatch(coordinator_snapshot_sha256) is None
        or HEX_SHA256_PATTERN.fullmatch(protocol_sha256) is None
        or HEX_SHA256_PATTERN.fullmatch(execution_profile_sha256) is None
        or HEX_SHA256_PATTERN.fullmatch(prompt_template_sha256) is None
    ):
        raise SnapshotError(
            "reviewer release coordinator bindings must be canonical SHA256 "
            "values"
        )
    assignment_count = (
        CALIBRATION_ASSIGNMENT_COUNT
        if phase == "calibration"
        else MAIN_ASSIGNMENT_COUNT
    )
    gate_hashes = (
        calibration_result_snapshot_sha256,
        calibration_decision_snapshot_sha256,
    )
    if phase == "calibration":
        if gate_hashes != ("NR", "NR"):
            raise SnapshotError(
                "calibration release gate bindings must be explicit NR"
            )
    elif any(
        HEX_SHA256_PATTERN.fullmatch(value) is None for value in gate_hashes
    ):
        raise SnapshotError(
            "main release gate bindings must be canonical SHA256 values"
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "phase": phase,
        "coordinator_snapshot_sha256": coordinator_snapshot_sha256,
        "protocol_sha256": protocol_sha256,
        "execution_profile_sha256": execution_profile_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "assignment_count": str(assignment_count),
        "calibration_result_snapshot_sha256": (
            calibration_result_snapshot_sha256
        ),
        "calibration_decision_snapshot_sha256": (
            calibration_decision_snapshot_sha256
        ),
    }


def build_reviewer_release_artifacts(
    coordinator: dict[str, bytes],
    phase: str,
    *,
    calibration_result_snapshot_sha256: str = "NR",
    calibration_decision_snapshot_sha256: str = "NR",
) -> dict[str, bytes]:
    if phase not in {"calibration", "main"}:
        raise SnapshotError(f"unsupported reviewer release phase {phase!r}")
    manifest_rows = _validate_manifest_shape(coordinator["manifest.csv"])
    _validate_calibration_selection_shape(
        coordinator["calibration_selection.csv"],
        manifest_rows,
    )
    manifest_versions = {row["manifest_version"] for row in manifest_rows}
    coordinator_hashes = {row["snapshot_sha256"] for row in manifest_rows}
    protocol_hashes = {row["protocol_sha256"] for row in manifest_rows}
    profile_hashes = {
        row["execution_profile_sha256"] for row in manifest_rows
    }
    prompt_hashes = {row["prompt_template_sha256"] for row in manifest_rows}
    if (
        manifest_versions != {MANIFEST_VERSION}
        or len(coordinator_hashes) != 1
        or len(protocol_hashes) != 1
        or len(profile_hashes) != 1
        or len(prompt_hashes) != 1
    ):
        raise SnapshotError(
            "coordinator manifest release bindings are not coherent"
        )
    coordinator_snapshot_sha256 = next(iter(coordinator_hashes))
    protocol_sha256 = next(iter(protocol_hashes))
    execution_profile_sha256 = next(iter(profile_hashes))
    prompt_template_sha256 = next(iter(prompt_hashes))
    if _sha256(coordinator["protocol.md"]) != protocol_sha256:
        raise SnapshotError(
            "coordinator protocol does not match its manifest binding"
        )
    if (
        _sha256(coordinator["execution_profile.json"])
        != execution_profile_sha256
    ):
        raise SnapshotError(
            "coordinator execution profile does not match its manifest binding"
        )
    if (
        _sha256(coordinator["reviewer_prompt_template.md"])
        != prompt_template_sha256
    ):
        raise SnapshotError(
            "coordinator prompt template does not match its manifest binding"
        )
    release_manifest = _release_manifest_row(
        phase=phase,
        coordinator_snapshot_sha256=coordinator_snapshot_sha256,
        protocol_sha256=protocol_sha256,
        execution_profile_sha256=execution_profile_sha256,
        prompt_template_sha256=prompt_template_sha256,
        calibration_result_snapshot_sha256=(
            calibration_result_snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            calibration_decision_snapshot_sha256
        ),
    )
    expected_assignments = {
        row["assignment_id"] for row in manifest_rows if row["phase"] == phase
    }
    coordinator_packet_assignments = _manifest_packet_assignment_ids(
        manifest_rows
    )
    release_packet_assignments = _manifest_packet_assignment_ids(
        manifest_rows,
        phase=phase,
    )
    released_assignments: set[str] = set()
    artifacts = {
        "execution_profile.json": coordinator["execution_profile.json"],
        "protocol.md": coordinator["protocol.md"],
        "reviewer_prompt_template.md": coordinator[
            "reviewer_prompt_template.md"
        ],
        "release_manifest.csv": _csv_bytes(
            RELEASE_MANIFEST_HEADER, [release_manifest]
        ),
    }
    for filename in PACKET_FILENAMES:
        relative = f"packets/{filename}"
        batch_id = filename.removesuffix(".csv")
        payload = coordinator[relative]
        _validate_packet_shape(
            payload,
            relative,
            expected_batch_id=batch_id,
            expected_assignment_ids=coordinator_packet_assignments[filename],
        )
        rows = _read_csv_bytes(payload, relative, PACKET_HEADER)
        filtered = [row for row in rows if row["phase"] == phase]
        released_assignments.update(row["assignment_id"] for row in filtered)
        release_payload = _csv_bytes(PACKET_HEADER, filtered)
        _validate_packet_shape(
            release_payload,
            relative,
            expected_batch_id=batch_id,
            expected_assignment_ids=release_packet_assignments[filename],
        )
        artifacts[relative] = release_payload
    if released_assignments != expected_assignments:
        raise SnapshotError(
            f"{phase} reviewer release assignment coverage is not exact"
        )
    checksum_paths = sorted(artifacts)
    artifacts["SHA256SUMS"] = "".join(
        f"{_sha256(artifacts[relative])}  {relative}\n"
        for relative in checksum_paths
    ).encode("utf-8")
    return artifacts
def reviewer_release_snapshot_sha256(
    phase: str,
    payloads: dict[str, bytes],
) -> str:
    """Return the canonical digest of one exact reviewer release payload set."""

    expected_paths = {
        "execution_profile.json",
        "protocol.md",
        "release_manifest.csv",
        "reviewer_prompt_template.md",
        "SHA256SUMS",
        *(f"packets/{filename}" for filename in PACKET_FILENAMES),
    }
    if phase not in {"calibration", "main"}:
        raise SnapshotError(f"unsupported reviewer release phase {phase!r}")
    if set(payloads) != expected_paths:
        raise SnapshotError(
            "reviewer release payload set is incomplete for hashing"
        )
    return _canonical_sha256(
        {
            "files": [
                {"path": path, "sha256": _sha256(payloads[path])}
                for path in sorted(
                    payloads,
                    key=lambda value: value.encode("utf-8"),
                )
            ],
            "manifest_version": MANIFEST_VERSION,
            "phase": phase,
        }
    )


def _release_manifest_from_payloads(
    reviewer_release: dict[str, bytes],
) -> Row:
    try:
        payload = reviewer_release["release_manifest.csv"]
    except KeyError as exc:
        raise SnapshotError("reviewer release manifest is missing") from exc
    rows = _read_csv_bytes(
        payload,
        "release_manifest.csv",
        RELEASE_MANIFEST_HEADER,
    )
    if len(rows) != 1:
        raise SnapshotError(
            "release manifest must contain exactly one authorization row"
        )
    return _validate_release_manifest_payload(payload, rows[0])


def _reviewer_task(phase: str) -> str:
    if phase == "calibration":
        return "calibration-screening"
    if phase == "main":
        return "main-screening"
    raise SnapshotError(f"unsupported reviewer release phase {phase!r}")


def render_reviewer_prompt(
    template_payload: bytes,
    *,
    role_id: str,
    stage_path: Path,
    protocol_sha256: str,
    packet_sha256: str,
) -> bytes:
    """Render only the visible reviewer prompt; completion tokens stay literal."""

    if f"{role_id}.csv" not in PACKET_FILENAMES:
        raise SnapshotError(f"unsupported screening role {role_id!r}")
    if (
        HEX_SHA256_PATTERN.fullmatch(protocol_sha256) is None
        or HEX_SHA256_PATTERN.fullmatch(packet_sha256) is None
    ):
        raise SnapshotError("reviewer prompt hashes must be canonical SHA256")
    supplied_stage_path = Path(stage_path)
    canonical_stage_path = _absolute_lexical(supplied_stage_path)
    if (
        not supplied_stage_path.is_absolute()
        or supplied_stage_path != canonical_stage_path
        or canonical_stage_path.name != "v1"
    ):
        raise SnapshotError("reviewer stage path must be canonical and absolute")
    parent_match = _ROLE_PRIVATE_PARENT_PATTERN.fullmatch(
        canonical_stage_path.parent.name
    )
    if parent_match is None or parent_match.group(1) != role_id:
        raise SnapshotError("reviewer stage path does not match role_id")
    output_path = canonical_stage_path.parent / f"{role_id}-result.csv"
    text = validate_reviewer_prompt_template(template_payload)
    substitutions = {
        "ROLE_ID": role_id,
        "STAGE_PATH": str(canonical_stage_path),
        "PROTOCOL_PATH": str(canonical_stage_path / "protocol.md"),
        "PROTOCOL_SHA256": protocol_sha256,
        "PACKET_PATH": str(canonical_stage_path / "packet.csv"),
        "PACKET_SHA256": packet_sha256,
        "OUTPUT_PATH": str(output_path),
    }
    for placeholder in sorted(_PROMPT_EXECUTION_PLACEHOLDERS):
        text = text.replace(
            "{{" + placeholder + "}}",
            substitutions[placeholder],
        )
    if any(
        "{{" + placeholder + "}}" in text
        for placeholder in _PROMPT_EXECUTION_PLACEHOLDERS
    ):
        raise SnapshotError("reviewer prompt execution placeholder was not rendered")
    for placeholder in _PROMPT_COMPLETION_PLACEHOLDERS:
        if "{{" + placeholder + "}}" not in text:
            raise SnapshotError(
                "reviewer prompt completion placeholders must remain literal"
            )
    return text.encode("utf-8")


def _execution_configuration(
    *,
    release_manifest: Row,
    reviewer_release_sha256: str,
    role_id: str,
    stage_path: Path,
    profile: dict[str, object],
    packet_sha256: str,
    allowed_inclusion_criteria: tuple[str, ...] | None = None,
    allowed_screening_statuses: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if (
        allowed_screening_statuses is not None
        and allowed_screening_statuses != CURRENT_SCREENING_RESULT_STATUSES
    ):
        raise SnapshotError(
            "allowed screening statuses must equal "
            "[\"included\", \"excluded\"]"
        )
    supplied_stage_path = Path(stage_path)
    canonical_stage_path = _absolute_lexical(supplied_stage_path)
    if (
        not supplied_stage_path.is_absolute()
        or supplied_stage_path != canonical_stage_path
    ):
        raise SnapshotError("execution configuration stage path is not canonical")
    output_path = canonical_stage_path.parent / f"{role_id}-result.csv"
    phase = release_manifest["phase"]
    configuration = {
        "configuration_version": "1",
        "coordinator_snapshot_sha256": release_manifest[
            "coordinator_snapshot_sha256"
        ],
        "execution_profile": profile,
        "execution_profile_sha256": release_manifest[
            "execution_profile_sha256"
        ],
        "packet": {
            "path": str(canonical_stage_path / "packet.csv"),
            "sha256": packet_sha256,
        },
        "phase": phase,
        "prompt": {
            "path": str(canonical_stage_path / "reviewer_prompt.md"),
            "template_path": str(
                canonical_stage_path / "reviewer_prompt_template.md"
            ),
            "template_sha256": release_manifest[
                "prompt_template_sha256"
            ],
        },
        "protocol": {
            "path": str(canonical_stage_path / "protocol.md"),
            "sha256": release_manifest["protocol_sha256"],
        },
        "result": {"path": str(output_path)},
        "reviewer_release_sha256": reviewer_release_sha256,
        "role_id": role_id,
        "stage_path": str(canonical_stage_path),
        "task": _reviewer_task(phase),
        "user_instruction_delivery": (
            "exact-rendered-visible-prompt-bytes"
        ),
        "work_item_scope": "one-role-packet",
    }
    if allowed_inclusion_criteria is not None:
        configuration["configuration_version"] = "2"
        configuration["allowed_inclusion_criteria"] = list(
            allowed_inclusion_criteria
        )
    return configuration


def _stage_snapshot_sha256(
    *,
    artifacts: dict[str, bytes],
    reviewer_release_sha256: str,
    role_id: str,
) -> str:
    return _canonical_sha256(
        {
            "files": [
                {"path": path, "sha256": _sha256(artifacts[path])}
                for path in sorted(
                    artifacts,
                    key=lambda value: value.encode("utf-8"),
                )
            ],
            "manifest_version": MANIFEST_VERSION,
            "reviewer_release_sha256": reviewer_release_sha256,
            "role_id": role_id,
        }
    )


def build_reviewer_stage_artifacts(
    reviewer_release: dict[str, bytes],
    role_id: str,
    stage_path: Path,
    allowed_inclusion_criteria: tuple[str, ...] | None = None,
    allowed_screening_statuses: tuple[str, ...] | None = None,
) -> dict[str, bytes]:
    """Derive one role-only execution snapshot from a validated release."""

    release_manifest = _release_manifest_from_payloads(reviewer_release)
    phase = release_manifest["phase"]
    release_sha256 = reviewer_release_snapshot_sha256(
        phase,
        reviewer_release,
    )
    filename = f"{role_id}.csv"
    if filename not in PACKET_FILENAMES:
        raise SnapshotError(f"unsupported screening role {role_id!r}")
    packet_relative = f"packets/{filename}"
    try:
        packet = reviewer_release[packet_relative]
        protocol = reviewer_release["protocol.md"]
        profile_payload = reviewer_release["execution_profile.json"]
        template = reviewer_release["reviewer_prompt_template.md"]
    except KeyError as exc:
        raise SnapshotError("reviewer release is incomplete") from exc

    packet_rows = _read_csv_bytes(packet, packet_relative, PACKET_HEADER)
    if not packet_rows:
        raise SnapshotError(f"{packet_relative}: assigned packet must not be empty")
    if {row["phase"] for row in packet_rows} != {phase}:
        raise SnapshotError(f"{packet_relative}: phase does not match release")
    _validate_packet_shape(
        packet,
        packet_relative,
        expected_batch_id=role_id,
        expected_assignment_ids={
            row["assignment_id"] for row in packet_rows
        },
    )
    profile = validate_execution_profile(profile_payload)
    validate_reviewer_prompt_template(template)
    hash_bindings = {
        "protocol_sha256": _sha256(protocol),
        "execution_profile_sha256": _sha256(profile_payload),
        "prompt_template_sha256": _sha256(template),
    }
    for field, actual in hash_bindings.items():
        if release_manifest[field] != actual:
            raise SnapshotError(
                f"reviewer release {field} does not match staged source bytes"
            )

    packet_sha256 = _sha256(packet)
    configuration = _execution_configuration(
        release_manifest=release_manifest,
        reviewer_release_sha256=release_sha256,
        role_id=role_id,
        stage_path=stage_path,
        profile=profile,
        packet_sha256=packet_sha256,
        allowed_inclusion_criteria=allowed_inclusion_criteria,
        allowed_screening_statuses=allowed_screening_statuses,
    )
    configuration_payload = _canonical_json_bytes(configuration)
    prompt = render_reviewer_prompt(
        template,
        role_id=role_id,
        stage_path=stage_path,
        protocol_sha256=release_manifest["protocol_sha256"],
        packet_sha256=packet_sha256,
    )
    artifacts = {
        "execution_configuration.json": configuration_payload,
        "execution_profile.json": profile_payload,
        "packet.csv": packet,
        "protocol.md": protocol,
        "reviewer_prompt.md": prompt,
        "reviewer_prompt_template.md": template,
    }
    configuration_sha256 = _sha256(configuration_payload)
    prompt_sha256 = _sha256(prompt)
    stage_sha256 = _stage_snapshot_sha256(
        artifacts=artifacts,
        reviewer_release_sha256=release_sha256,
        role_id=role_id,
    )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "stage_snapshot_sha256": stage_sha256,
        "reviewer_release_sha256": release_sha256,
        "coordinator_snapshot_sha256": release_manifest[
            "coordinator_snapshot_sha256"
        ],
        "phase": phase,
        "task": _reviewer_task(phase),
        "role_id": role_id,
        "protocol_sha256": release_manifest["protocol_sha256"],
        "packet_sha256": packet_sha256,
        "execution_profile_sha256": release_manifest[
            "execution_profile_sha256"
        ],
        "prompt_template_sha256": release_manifest[
            "prompt_template_sha256"
        ],
        "configuration_sha256": configuration_sha256,
        "prompt_sha256": prompt_sha256,
        "user_instruction_sha256": prompt_sha256,
        "stage_path": str(stage_path),
        "result_path": str(Path(stage_path).parent / f"{role_id}-result.csv"),
        "assignment_count": str(len(packet_rows)),
    }
    artifacts["stage_manifest.csv"] = _csv_bytes(
        STAGE_MANIFEST_HEADER,
        [manifest],
    )
    checksum_paths = sorted(artifacts)
    artifacts["SHA256SUMS"] = "".join(
        f"{_sha256(artifacts[relative])}  {relative}\n"
        for relative in checksum_paths
    ).encode("utf-8")
    return artifacts


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise SnapshotError(
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
            raise SnapshotError(
                "atomic no-clobber snapshot publication is unsupported"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )


def _rename_noreplace_at(
    directory_fd: int, source_name: str, destination_name: str
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise SnapshotError(
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
        directory_fd,
        os.fsencode(source_name),
        directory_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ENOSYS:
            raise SnapshotError(
                "atomic no-clobber snapshot publication is unsupported"
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


# Keep rollback operational when publication-boundary hooks fail or race.
_cleanup_rename_noreplace_at = _rename_noreplace_at


def _write_snapshot_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    on_created: Callable[[FileIdentity], None] | None = None,
) -> FileIdentity:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    identity: FileIdentity | None = None
    try:
        file_stat = os.fstat(descriptor)
        identity = (file_stat.st_dev, file_stat.st_ino)
        if on_created is not None:
            on_created(identity)
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("snapshot write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert identity is not None
    return identity


def _write_snapshot_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
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


def _stage_artifacts(
    staging: int,
    artifacts: dict[str, bytes],
    identities: dict[str, FileIdentity],
) -> None:
    root_stat = os.fstat(staging)
    identities["."] = (root_stat.st_dev, root_stat.st_ino)
    has_packets = any(
        relative.startswith("packets/") for relative in artifacts
    )
    packets_fd: int | None = None
    if has_packets:
        os.mkdir("packets", mode=0o700, dir_fd=staging)
        packets_stat = os.stat(
            "packets", dir_fd=staging, follow_symlinks=False
        )
        identities["packets"] = (
            packets_stat.st_dev,
            packets_stat.st_ino,
        )
        packets_fd = os.open(
            "packets", _DIRECTORY_OPEN_FLAGS, dir_fd=staging
        )
    try:
        for relative in sorted(artifacts):
            payload = artifacts[relative]
            if relative.startswith("packets/"):
                if packets_fd is None:
                    raise SnapshotError(
                        "packet artifact has no staged packets directory"
                    )
                directory_fd = packets_fd
                name = relative.removeprefix("packets/")
            else:
                directory_fd = staging
                name = relative

            def record_created(
                identity: FileIdentity, relative: str = relative
            ) -> None:
                identities[relative] = identity

            identity = _write_snapshot_file_at(
                directory_fd,
                name,
                payload,
                on_created=record_created,
            )
            if identities.get(relative) != identity:
                raise SnapshotError(
                    f"staged {relative}: creation ownership was not recorded"
                )
        if packets_fd is not None:
            os.fchmod(packets_fd, 0o755)
            os.fsync(packets_fd)
        os.fchmod(staging, 0o755)
        os.fsync(staging)
    finally:
        if packets_fd is not None:
            os.close(packets_fd)


def _verify_staged_artifacts(
    staging: int,
    artifacts: dict[str, bytes],
    identities: dict[str, FileIdentity],
) -> None:
    has_packets = any(
        relative.startswith("packets/") for relative in artifacts
    )
    root_names = {
        relative.split("/", 1)[0] for relative in artifacts
    }
    root_attestation = _attest_directory_fd(
        staging, root_names, "staged snapshot", 0o755
    )
    if root_attestation.identity != identities["."]:
        raise SnapshotError("staged snapshot identity changed")

    packets_fd: int | None = None
    if has_packets:
        packets_fd = os.open(
            "packets", _DIRECTORY_OPEN_FLAGS, dir_fd=staging
        )
    try:
        if packets_fd is not None:
            packet_names = {
                relative.removeprefix("packets/")
                for relative in artifacts
                if relative.startswith("packets/")
            }
            packets_attestation = _attest_directory_fd(
                packets_fd, packet_names, "staged packets", 0o755
            )
            if packets_attestation.identity != identities["packets"]:
                raise SnapshotError("staged packets identity changed")
        for relative in sorted(artifacts):
            if relative.startswith("packets/"):
                if packets_fd is None:
                    raise SnapshotError(
                        "packet artifact has no staged packets directory"
                    )
                directory_fd = packets_fd
                name = relative.removeprefix("packets/")
            else:
                directory_fd = staging
                name = relative
            read_file = _read_regular_file_at(
                directory_fd, name, f"staged {relative}"
            )
            if read_file.identity != identities[relative]:
                raise SnapshotError(f"staged {relative}: identity changed")
            if read_file.link_count != 1:
                raise SnapshotError(
                    f"staged {relative}: hard link count changed"
                )
            if read_file.mode != 0o644:
                raise SnapshotError(
                    f"staged {relative}: mode {read_file.mode:#o} != 0o644"
                )
            if _sha256(read_file.payload) != _sha256(artifacts[relative]):
                raise SnapshotError(f"staged {relative}: hash changed")
        if packets_fd is not None:
            _assert_directory_unchanged(
                packets_fd,
                packets_attestation,
                packet_names,
                "staged packets",
            )
        _assert_directory_unchanged(
            staging,
            root_attestation,
            root_names,
            "staged snapshot",
        )
    finally:
        if packets_fd is not None:
            os.close(packets_fd)


def _create_staging_at(
    parent_fd: int,
    version_name: str,
    *,
    on_created: Callable[[str, int, FileIdentity], None] | None = None,
) -> tuple[str, int]:
    for _ in range(128):
        name = f".{version_name}.{secrets.token_hex(8)}.tmp"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(
                name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd
            )
        except BaseException:
            # The newly created name may have been replaced before open.
            # POSIX offers no conditional deletion-by-inode, so fail without
            # touching a pathname that could now belong to another writer.
            raise
        file_stat = os.fstat(descriptor)
        identity = (file_stat.st_dev, file_stat.st_ino)
        if on_created is not None:
            on_created(name, descriptor, identity)
        return name, descriptor
    raise SnapshotError("could not allocate a unique staging directory")


def _identity_at(directory_fd: int, name: str) -> FileIdentity | None:
    try:
        file_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return file_stat.st_dev, file_stat.st_ino


def _capture_identity_text(identity: FileIdentity) -> str:
    return f"(dev, ino)=({identity[0]}, {identity[1]})"


def _directory_path_at(directory_fd: int) -> Path:
    path = Path(os.readlink(f"/proc/self/fd/{directory_fd}"))
    return path if path.is_absolute() else Path.cwd() / path


def _identity_paths_at(
    directory_fd: int,
    identity: FileIdentity,
) -> tuple[list[Path], list[str]]:
    directory_path = _directory_path_at(directory_fd)
    discovered: list[Path] = []
    scan_errors: list[str] = []
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        return [], [f"parent scan failed: {type(exc).__name__}: {exc}"]
    for name in names:
        try:
            current_identity = _identity_at(directory_fd, name)
        except OSError as exc:
            scan_errors.append(
                f"could not inspect {directory_path / name}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if current_identity == identity:
            discovered.append(directory_path / name)
    return discovered, scan_errors


def _raise_moved_root_recovery(
    parent_fd: int,
    root_name: str,
    identity: FileIdentity,
) -> None:
    discovered, scan_errors = _identity_paths_at(parent_fd, identity)
    locations = (
        ", ".join(str(path) for path in discovered)
        if discovered
        else "no matching name discovered in the anchored parent"
    )
    recovery = SnapshotError(
        f"{root_name}: expected snapshot root {_capture_identity_text(identity)} "
        f"moved before capture; recovery location(s): {locations}; no entry "
        "was moved or overwritten"
    )
    for scan_error in scan_errors:
        _attach_exception_detail(recovery, scan_error)
    raise recovery


def _report_foreign_capture_at(
    directory_fd: int,
    quarantine_name: str,
    source_name: str,
    identity: FileIdentity,
) -> None:
    """Attempt one lossless restoration, then report the observed race."""

    restoration_error: BaseException | None = None
    try:
        _cleanup_rename_noreplace_at(
            directory_fd,
            quarantine_name,
            source_name,
        )
    except BaseException as exc:
        restoration_error = exc

    source_identity = _identity_at(directory_fd, source_name)
    quarantine_identity = _identity_at(directory_fd, quarantine_name)
    directory_path = _directory_path_at(directory_fd)
    source_path = directory_path / source_name
    quarantine_path = directory_path / quarantine_name
    if (
        source_identity == identity
        and quarantine_identity is None
    ):
        race = SnapshotError(
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
        recovery = SnapshotError(
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

    conflict = SnapshotError(
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


def _capture_then_classify_entry_at(
    directory_fd: int,
    name: str,
    expected_identity: FileIdentity,
) -> Path | None:
    """Capture a cleanup source, then classify the captured inode."""

    current_identity = _identity_at(directory_fd, name)
    if current_identity is None or current_identity != expected_identity:
        return None

    for _ in range(128):
        quarantine_name = f".trackgen-retired-{secrets.token_hex(16)}"
        try:
            _cleanup_rename_noreplace_at(
                directory_fd,
                name,
                quarantine_name,
            )
        except FileExistsError:
            if _identity_at(directory_fd, name) == expected_identity:
                continue
            raise SnapshotError(
                f"{name}: snapshot entry changed before capture"
            )
        except FileNotFoundError:
            if _identity_at(directory_fd, name) is None:
                return None
            raise

        source_identity = _identity_at(directory_fd, name)
        captured_identity = _identity_at(directory_fd, quarantine_name)
        quarantine_path = _directory_path_at(directory_fd) / quarantine_name
        if captured_identity == expected_identity:
            if source_identity is None:
                return quarantine_path
            raise SnapshotError(
                f"{name}: expected snapshot entry was retained at "
                f"{quarantine_path}, but the source was refilled with "
                f"{_capture_identity_text(source_identity)}"
            )
        if captured_identity is not None:
            _report_foreign_capture_at(
                directory_fd,
                quarantine_name,
                name,
                captured_identity,
            )
        raise SnapshotError(f"{name}: entry changed during capture classification")
    raise SnapshotError(
        f"{name}: could not allocate a quarantine cleanup name"
    )


def _capture_snapshot_root_at(
    parent_fd: int,
    root_name: str,
    identities: dict[str, FileIdentity],
    *,
    root_fd: int | None = None,
    alternate_names: Sequence[str] = (),
) -> None:
    """Capture the complete snapshot root once without traversing children."""

    root_identity = identities.get(".")
    if root_identity is None:
        return

    close_root_fd = False
    if root_fd is None:
        try:
            root_fd = os.open(
                root_name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return
        close_root_fd = True
    try:
        root_stat = os.fstat(root_fd)
        if (root_stat.st_dev, root_stat.st_ino) != root_identity:
            return
        quarantine_path = _capture_then_classify_entry_at(
            parent_fd,
            root_name,
            root_identity,
        )
        if quarantine_path is None:
            for alternate_name in alternate_names:
                if alternate_name == root_name:
                    continue
                quarantine_path = _capture_then_classify_entry_at(
                    parent_fd,
                    alternate_name,
                    root_identity,
                )
                if quarantine_path is not None:
                    return
            _raise_moved_root_recovery(parent_fd, root_name, root_identity)
    finally:
        if close_root_fd:
            os.close(root_fd)


def _capture_snapshot_transaction_root_at(
    parent_fd: int,
    output_name: str,
    staging_name: str | None,
    identities: dict[str, FileIdentity],
    *,
    root_fd: int | None,
) -> None:
    root_identity = identities.get(".")
    if root_identity is None:
        return
    alternate_names = () if staging_name is None else (staging_name,)
    _capture_snapshot_root_at(
        parent_fd,
        output_name,
        identities,
        root_fd=root_fd,
        alternate_names=alternate_names,
    )


def _validate_version_path(path: Path, label: str) -> None:
    if _VERSION_PATTERN.fullmatch(path.name) is None:
        raise SnapshotError(
            f"{path}: {label} must name an immutable version such as v1"
        )


def _read_freeze_inputs(
    *,
    candidates: Path,
    conflicts: Path,
    bibliography: Path,
    citation_keys: Path,
    taxonomy: Path,
    protocol: Path,
    execution_profile: Path,
    reviewer_prompt_template: Path,
    output_dir: Path,
) -> dict[str, bytes]:
    source_paths = {
        "candidates.csv": candidates,
        "conflicts.csv": conflicts,
        "bibliography.csv": bibliography,
        "citation_keys.csv": citation_keys,
        "taxonomy.json": taxonomy,
        "protocol.md": protocol,
        "execution_profile.json": execution_profile,
        "reviewer_prompt_template.md": reviewer_prompt_template,
    }
    read_files = {
        name: _read_regular_file(path, f"{name} input")
        for name, path in source_paths.items()
    }
    identities: dict[FileIdentity, str] = {}
    canonical_paths: dict[Path, str] = {}
    for name, read_file in read_files.items():
        if read_file.identity in identities:
            raise SnapshotError(
                f"{read_file.path}: {name} aliases {identities[read_file.identity]}"
            )
        if read_file.path in canonical_paths:
            raise SnapshotError(
                f"{read_file.path}: {name} aliases {canonical_paths[read_file.path]}"
            )
        identities[read_file.identity] = name
        canonical_paths[read_file.path] = name
    output_lexical = _absolute_lexical(output_dir)
    if output_lexical in canonical_paths:
        raise SnapshotError(
            f"{output_dir}: output path aliases {canonical_paths[output_lexical]}"
        )
    return {name: read_file.payload for name, read_file in read_files.items()}


def _publish_artifacts(
    output_dir: Path,
    artifacts: dict[str, bytes],
    *,
    post_publish_check: Callable[[], None] | None = None,
) -> None:
    output_dir = Path(output_dir)
    _validate_version_path(output_dir, "output directory")
    parent_path = _absolute_lexical(output_dir.parent)
    parent_fd, parent_identity = _open_directory_fd(
        parent_path, "output parent directory"
    )
    staging_name: str | None = None
    staging_fd: int | None = None
    identities: dict[str, FileIdentity] = {}
    try:
        if _identity_at(parent_fd, output_dir.name) is not None:
            raise SnapshotError(
                f"{output_dir}: snapshot version already exists"
            )
        def record_staging_creation(
            name: str, descriptor: int, identity: FileIdentity
        ) -> None:
            nonlocal staging_name, staging_fd
            staging_name = name
            staging_fd = descriptor
            identities["."] = identity

        created_name, created_fd = _create_staging_at(
            parent_fd,
            output_dir.name,
            on_created=record_staging_creation,
        )
        if created_name != staging_name or created_fd != staging_fd:
            raise SnapshotError("staging creation ownership was not recorded")
        _stage_artifacts(staging_fd, artifacts, identities)
        _verify_staged_artifacts(staging_fd, artifacts, identities)
        _recheck_directory_path(
            parent_path, parent_identity, "output parent directory"
        )
        _verify_staged_artifacts(staging_fd, artifacts, identities)
        try:
            _rename_noreplace_at(
                parent_fd, staging_name, output_dir.name
            )
        except FileExistsError as exc:
            raise SnapshotError(
                f"{output_dir}: snapshot version already exists"
            ) from exc
        _verify_staged_artifacts(staging_fd, artifacts, identities)
        if _identity_at(parent_fd, output_dir.name) != identities["."]:
            raise SnapshotError(
                "snapshot publication did not install the staged directory"
            )
        _recheck_directory_path(
            parent_path, parent_identity, "output parent directory"
        )
        if post_publish_check is not None:
            post_publish_check()
        _verify_staged_artifacts(staging_fd, artifacts, identities)
        if _identity_at(parent_fd, output_dir.name) != identities["."]:
            raise SnapshotError(
                "snapshot changed during post-publication validation"
            )
        _recheck_directory_path(
            parent_path, parent_identity, "output parent directory"
        )
        os.fsync(parent_fd)
        if _identity_at(parent_fd, output_dir.name) != identities["."]:
            raise SnapshotError(
                "snapshot changed during post-publication validation"
            )
    except BaseException as error:
        if identities:
            try:
                _capture_snapshot_transaction_root_at(
                    parent_fd,
                    output_dir.name,
                    staging_name,
                    identities,
                    root_fd=staging_fd,
                )
            except BaseException as cleanup_error:
                _attach_exception_detail(
                    error,
                    "snapshot rollback encountered "
                    f"{_exception_diagnostic(cleanup_error)}",
                )
        raise
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def publish_snapshot(
    output_dir: Path,
    artifacts: dict[str, bytes],
    *,
    post_publish_check: Callable[[], None] | None = None,
) -> None:
    """Publish an immutable snapshot through the hardened private publisher."""

    return _publish_artifacts(
        output_dir,
        artifacts,
        post_publish_check=post_publish_check,
    )


def freeze_snapshot(
    *,
    candidates: Path,
    conflicts: Path,
    bibliography: Path,
    citation_keys: Path,
    execution_profile: Path,
    reviewer_prompt_template: Path,
    taxonomy: Path,
    protocol: Path,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    _validate_version_path(output_dir, "output directory")
    _require_real_directory(output_dir.parent, "output parent directory")
    if os.path.lexists(output_dir):
        raise SnapshotError(
            f"{output_dir}: snapshot version already exists"
        )
    raw_payloads = _read_freeze_inputs(
        candidates=Path(candidates),
        conflicts=Path(conflicts),
        bibliography=Path(bibliography),
        citation_keys=Path(citation_keys),
        execution_profile=Path(execution_profile),
        reviewer_prompt_template=Path(reviewer_prompt_template),
        taxonomy=Path(taxonomy),
        protocol=Path(protocol),
        output_dir=output_dir,
    )
    artifacts = build_snapshot_artifacts(raw_payloads, strict_new=True)

    _publish_artifacts(output_dir, artifacts)


def _validate_exact_directory(
    path: Path,
    expected_names: set[str] | frozenset[str],
    label: str,
    expected_mode: int,
) -> FileIdentity:
    file_stat = _require_real_directory(path, label)
    actual_mode = stat.S_IMODE(file_stat.st_mode)
    if actual_mode != expected_mode:
        raise SnapshotError(
            f"{path}: {label} mode {actual_mode:#o} != {expected_mode:#o}"
        )
    actual_names = {entry.name for entry in os.scandir(path)}
    if actual_names != set(expected_names):
        missing = sorted(set(expected_names) - actual_names)
        extra = sorted(actual_names - set(expected_names))
        raise SnapshotError(
            f"{path}: {label} entries mismatch; missing={missing}, extra={extra}"
        )
    return file_stat.st_dev, file_stat.st_ino


def _validate_manifest_shape(payload: bytes) -> list[Row]:
    rows = _read_csv_bytes(payload, "manifest.csv", MANIFEST_HEADER)
    if len(rows) != ASSIGNMENT_COUNT:
        raise SnapshotError(
            "manifest.csv: must contain exactly "
            f"{ASSIGNMENT_COUNT} assignments, found {len(rows)}"
        )
    seen_assignments: set[str] = set()
    by_candidate: defaultdict[str, list[Row]] = defaultdict(list)
    valid_batches = set(filename.removesuffix(".csv") for filename in PACKET_FILENAMES)
    for row_number, row in enumerate(rows, start=2):
        assignment_id = row["assignment_id"]
        candidate_id = row["candidate_id"]
        batch_id = row["batch_id"]
        phase = row["phase"]
        if assignment_id in seen_assignments:
            raise SnapshotError(
                f"manifest.csv:{row_number}: duplicate assignment_id "
                f"{assignment_id!r}"
            )
        seen_assignments.add(assignment_id)
        expected_assignment_id = (
            f"A-{candidate_id}-{batch_id.removeprefix('screening-')}"
        )
        if (
            _ASSIGNMENT_ID_PATTERN.fullmatch(assignment_id) is None
            or assignment_id != expected_assignment_id
        ):
            raise SnapshotError(
                f"manifest.csv:{row_number}: assignment_id={assignment_id!r} "
                "does not match candidate and batch"
            )
        if batch_id not in valid_batches:
            raise SnapshotError(
                f"manifest.csv:{row_number}: invalid batch_id {batch_id!r}"
            )
        if phase not in {"calibration", "main"}:
            raise SnapshotError(
                f"manifest.csv:{row_number}: invalid phase {phase!r}"
            )
        if _POSITIVE_INTEGER_PATTERN.fullmatch(row["weight"]) is None:
            raise SnapshotError(
                f"manifest.csv:{row_number}: weight must be a canonical "
                "positive integer"
            )
        by_candidate[candidate_id].append(row)

    for candidate_id, candidate_rows in by_candidate.items():
        if len(candidate_rows) != 2:
            raise SnapshotError(
                f"manifest.csv: candidate_id={candidate_id!r} must have "
                "exactly two assignments"
            )
        if len({row["batch_id"] for row in candidate_rows}) != 2:
            raise SnapshotError(
                f"manifest.csv: candidate_id={candidate_id!r} assignments "
                "must use distinct batches"
            )
        if len({row["phase"] for row in candidate_rows}) != 1:
            raise SnapshotError(
                f"manifest.csv: candidate_id={candidate_id!r} assignments "
                "must share phase"
            )
        if len({row["input_sha256"] for row in candidate_rows}) != 1:
            raise SnapshotError(
                f"manifest.csv: candidate_id={candidate_id!r} assignments "
                "must share input_sha256"
            )


    if len(by_candidate) != CANDIDATE_COUNT:
        raise SnapshotError(
            "manifest.csv: must contain exactly "
            f"{CANDIDATE_COUNT} unique candidates, found {len(by_candidate)}"
        )
    assignment_counts = Counter(row["phase"] for row in rows)
    if assignment_counts != {
        "calibration": CALIBRATION_ASSIGNMENT_COUNT,
        "main": MAIN_ASSIGNMENT_COUNT,
    }:
        raise SnapshotError(
            "manifest.csv: phase assignment counts must be exactly "
            f"{CALIBRATION_ASSIGNMENT_COUNT} calibration and "
            f"{MAIN_ASSIGNMENT_COUNT} main, found {dict(assignment_counts)}"
        )
    candidate_counts = Counter(
        candidate_rows[0]["phase"]
        for candidate_rows in by_candidate.values()
    )
    if candidate_counts != {
        "calibration": CALIBRATION_CANDIDATE_COUNT,
        "main": MAIN_CANDIDATE_COUNT,
    }:
        raise SnapshotError(
            "manifest.csv: phase candidate counts must be exactly "
            f"{CALIBRATION_CANDIDATE_COUNT} calibration and "
            f"{MAIN_CANDIDATE_COUNT} main, found {dict(candidate_counts)}"
        )
    return rows


def _validate_calibration_selection_shape(
    payload: bytes,
    manifest_rows: Sequence[Row],
    *,
    expected_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    rows = _read_csv_bytes(
        payload,
        "calibration_selection.csv",
        CALIBRATION_SELECTION_HEADER,
    )
    candidate_ids = tuple(row["candidate_id"] for row in rows)
    if len(candidate_ids) != CALIBRATION_CANDIDATE_COUNT:
        raise SnapshotError(
            "calibration_selection.csv: must contain exactly "
            f"{CALIBRATION_CANDIDATE_COUNT} candidate IDs"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise SnapshotError(
            "calibration_selection.csv: candidate IDs must be unique"
        )
    if any(
        _CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None
        for candidate_id in candidate_ids
    ):
        raise SnapshotError(
            "calibration_selection.csv: invalid candidate_id"
        )
    canonical_ids = tuple(sorted(candidate_ids, key=_stable_candidate_rank))
    if candidate_ids != canonical_ids:
        raise SnapshotError(
            "calibration_selection.csv: candidate IDs are not in canonical order"
        )
    manifest_ids = {
        row["candidate_id"]
        for row in manifest_rows
        if row["phase"] == "calibration"
    }
    if set(candidate_ids) != manifest_ids:
        raise SnapshotError(
            "calibration_selection.csv: candidate IDs do not match "
            "calibration manifest rows"
        )
    if expected_ids is not None and candidate_ids != tuple(expected_ids):
        raise SnapshotError(
            "calibration_selection.csv: candidate IDs do not match "
            "metadata-derived selection"
        )
    return candidate_ids


def _manifest_packet_assignment_ids(
    manifest_rows: Sequence[Row],
    *,
    phase: str | None = None,
) -> dict[str, frozenset[str]]:
    if phase is not None and phase not in {"calibration", "main"}:
        raise SnapshotError(f"unsupported packet phase {phase!r}")
    assignments: dict[str, set[str]] = {
        filename: set() for filename in PACKET_FILENAMES
    }
    for row in manifest_rows:
        if phase is not None and row["phase"] != phase:
            continue
        filename = f"{row['batch_id']}.csv"
        if filename not in assignments:
            raise SnapshotError(
                f"manifest.csv: unknown packet filename for {row['batch_id']!r}"
            )
        assignments[filename].add(row["assignment_id"])
    return {
        filename: frozenset(assignment_ids)
        for filename, assignment_ids in assignments.items()
    }


def _validate_packet_shape(
    payload: bytes,
    filename: str,
    *,
    expected_batch_id: str,
    expected_assignment_ids: frozenset[str],
) -> None:
    rows = _read_csv_bytes(payload, filename, PACKET_HEADER)
    seen_assignments: set[str] = set()
    seen_candidates: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        assignment_id = row["assignment_id"]
        candidate_id = row["candidate_id"]
        if row["batch_id"] != expected_batch_id:
            raise SnapshotError(
                f"{filename}:{row_number}: batch_id={row['batch_id']!r} "
                f"does not match packet batch {expected_batch_id!r}"
            )
        if assignment_id in seen_assignments:
            raise SnapshotError(
                f"{filename}:{row_number}: duplicate assignment_id "
                f"{assignment_id!r}"
            )
        if candidate_id in seen_candidates:
            raise SnapshotError(
                f"{filename}:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        seen_assignments.add(assignment_id)
        seen_candidates.add(candidate_id)
        expected_assignment_id = (
            f"A-{candidate_id}-{row['batch_id'].removeprefix('screening-')}"
        )
        if assignment_id != expected_assignment_id:
            raise SnapshotError(
                f"{filename}:{row_number}: assignment_id={assignment_id!r} "
                "does not match candidate and batch"
            )
        if row["phase"] not in {"calibration", "main"}:
            raise SnapshotError(
                f"{filename}:{row_number}: invalid phase {row['phase']!r}"
            )
        for field, value in row.items():
            if not value:
                raise SnapshotError(
                    f"{filename}:{row_number}: {field} must not be blank"
                )
    if seen_assignments != set(expected_assignment_ids):
        missing = sorted(expected_assignment_ids - seen_assignments)
        extra = sorted(seen_assignments - expected_assignment_ids)
        raise SnapshotError(
            f"{filename}: assignment membership mismatch; "
            f"missing={missing}, extra={extra}"
        )


def validate_snapshot(snapshot_dir: Path) -> dict[str, bytes]:
    snapshot_dir = _absolute_lexical(Path(snapshot_dir))
    _validate_version_path(snapshot_dir, "snapshot directory")
    parent_fd, parent_identity = _open_directory_fd(
        snapshot_dir.parent, "snapshot parent directory"
    )
    root_fd: int | None = None
    packets_fd: int | None = None
    try:
        try:
            root_stat = os.stat(
                snapshot_dir.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise SnapshotError(
                f"{snapshot_dir}: snapshot directory is missing"
            ) from exc
        if stat.S_ISLNK(root_stat.st_mode):
            raise SnapshotError(
                f"{snapshot_dir}: snapshot directory must not be a symlink"
            )
        root_fd = os.open(
            snapshot_dir.name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        if (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino) != root_identity:
            raise SnapshotError(
                f"{snapshot_dir}: snapshot directory changed before read"
            )
        root_attestation = _attest_directory_fd(
            root_fd,
            ROOT_FILENAMES,
            "snapshot directory",
            0o755,
        )
        packets_fd = os.open(
            "packets", _DIRECTORY_OPEN_FLAGS, dir_fd=root_fd
        )
        packets_attestation = _attest_directory_fd(
            packets_fd,
            set(PACKET_FILENAMES),
            "packets directory",
            0o755,
        )

        relative_files = [
            *RAW_FILENAMES,
            "calibration_selection.csv",
            "manifest.csv",
            "SHA256SUMS",
            *(f"packets/{filename}" for filename in PACKET_FILENAMES),
        ]
        actual: dict[str, bytes] = {}
        first_reads: dict[str, _ReadFile] = {}
        identities: dict[FileIdentity, str] = {}
        for relative in relative_files:
            if relative.startswith("packets/"):
                directory_fd = packets_fd
                name = relative.removeprefix("packets/")
            else:
                directory_fd = root_fd
                name = relative
            read_file = _read_regular_file_at(
                directory_fd, name, relative
            )
            if read_file.mode != 0o644:
                raise SnapshotError(
                    f"{relative}: file mode {read_file.mode:#o} != 0o644"
                )
            if read_file.identity in identities:
                raise SnapshotError(
                    f"{relative}: file aliases {identities[read_file.identity]}"
                )
            identities[read_file.identity] = relative
            first_reads[relative] = read_file
            actual[relative] = read_file.payload

        manifest_rows = _validate_manifest_shape(actual["manifest.csv"])
        packet_assignments = _manifest_packet_assignment_ids(manifest_rows)
        for filename in PACKET_FILENAMES:
            _validate_packet_shape(
                actual[f"packets/{filename}"],
                f"packets/{filename}",
                expected_batch_id=filename.removesuffix(".csv"),
                expected_assignment_ids=packet_assignments[filename],
            )
        raw_payloads = {name: actual[name] for name in RAW_FILENAMES}
        source = _load_source_data(raw_payloads, strict_new=False)
        expected_selection = _select_calibration_candidate_ids(
            source.candidates
        )
        _validate_calibration_selection_shape(
            actual["calibration_selection.csv"],
            manifest_rows,
            expected_ids=expected_selection,
        )
        expected = build_snapshot_artifacts(raw_payloads, strict_new=False)
        for relative, expected_payload in expected.items():
            if actual[relative] != expected_payload:
                raise SnapshotError(
                    f"{snapshot_dir / relative}: content does not match "
                    "canonical snapshot derivation"
                )

        _assert_directory_unchanged(
            root_fd,
            root_attestation,
            ROOT_FILENAMES,
            "snapshot directory",
        )
        _assert_directory_unchanged(
            packets_fd,
            packets_attestation,
            set(PACKET_FILENAMES),
            "packets directory",
        )
        for relative in relative_files:
            if relative.startswith("packets/"):
                directory_fd = packets_fd
                name = relative.removeprefix("packets/")
            else:
                directory_fd = root_fd
                name = relative
            second = _read_regular_file_at(directory_fd, name, relative)
            first = first_reads[relative]
            if (
                second.identity != first.identity
                or second.link_count != first.link_count
                or second.mode != first.mode
                or _sha256(second.payload) != _sha256(first.payload)
            ):
                raise SnapshotError(f"{relative}: file changed after read")
        _assert_directory_unchanged(
            root_fd,
            root_attestation,
            ROOT_FILENAMES,
            "snapshot directory",
        )
        _assert_directory_unchanged(
            packets_fd,
            packets_attestation,
            set(PACKET_FILENAMES),
            "packets directory",
        )
        _recheck_directory_path(
            snapshot_dir.parent,
            parent_identity,
            "snapshot parent directory",
        )
        if _identity_at(parent_fd, snapshot_dir.name) != root_identity:
            raise SnapshotError(
                f"{snapshot_dir}: snapshot directory changed after read"
            )
        return actual
    finally:
        if packets_fd is not None:
            os.close(packets_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _validate_release_manifest_payload(
    payload: bytes,
    expected_manifest: Row,
) -> Row:
    rows = _read_csv_bytes(
        payload,
        "release_manifest.csv",
        RELEASE_MANIFEST_HEADER,
    )
    if len(rows) != 1:
        raise SnapshotError(
            "release manifest must contain exactly one authorization row"
        )
    row = rows[0]
    if payload != _csv_bytes(RELEASE_MANIFEST_HEADER, [row]):
        raise SnapshotError("release manifest is not canonical")
    if set(expected_manifest) != set(RELEASE_MANIFEST_HEADER):
        raise SnapshotError("expected release manifest schema is invalid")
    expected_row = {
        field: expected_manifest[field] for field in RELEASE_MANIFEST_HEADER
    }
    if row != expected_row:
        raise SnapshotError(
            "release manifest binding does not match expected authorization"
        )
    canonical_row = _release_manifest_row(
        phase=row["phase"],
        coordinator_snapshot_sha256=row["coordinator_snapshot_sha256"],
        protocol_sha256=row["protocol_sha256"],
        execution_profile_sha256=row["execution_profile_sha256"],
        prompt_template_sha256=row["prompt_template_sha256"],
        calibration_result_snapshot_sha256=(
            row["calibration_result_snapshot_sha256"]
        ),
        calibration_decision_snapshot_sha256=(
            row["calibration_decision_snapshot_sha256"]
        ),
    )
    if row != canonical_row:
        raise SnapshotError("release manifest fields are not canonical")
    return row


def validate_reviewer_release_snapshot(
    snapshot_dir: Path,
    *,
    expected_manifest: Row,
    coordinator_snapshot: dict[str, bytes],
) -> dict[str, bytes]:
    snapshot_dir = _absolute_lexical(Path(snapshot_dir))
    _validate_version_path(snapshot_dir, "reviewer release directory")
    parent_fd, parent_identity = _open_directory_fd(
        snapshot_dir.parent, "reviewer release parent directory"
    )
    root_fd: int | None = None
    packets_fd: int | None = None
    try:
        try:
            root_stat = os.stat(
                snapshot_dir.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise SnapshotError(
                f"{snapshot_dir}: reviewer release directory is missing"
            ) from exc
        if stat.S_ISLNK(root_stat.st_mode):
            raise SnapshotError(
                f"{snapshot_dir}: reviewer release must not be a symlink"
            )
        if not stat.S_ISDIR(root_stat.st_mode):
            raise SnapshotError(
                f"{snapshot_dir}: reviewer release must be a directory"
            )
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        root_fd = os.open(
            snapshot_dir.name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
        root_stat_after_open = os.fstat(root_fd)
        if (
            root_stat_after_open.st_dev,
            root_stat_after_open.st_ino,
        ) != root_identity:
            raise SnapshotError(
                f"{snapshot_dir}: reviewer release changed before read"
            )
        root_attestation = _attest_directory_fd(
            root_fd,
            REVIEWER_RELEASE_ROOT_FILENAMES,
            "reviewer release directory",
            0o755,
        )
        packets_fd = os.open(
            "packets",
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=root_fd,
        )
        packets_attestation = _attest_directory_fd(
            packets_fd,
            set(PACKET_FILENAMES),
            "reviewer release packets directory",
            0o755,
        )

        relative_files = [
            "execution_profile.json",
            "protocol.md",
            "reviewer_prompt_template.md",
            "release_manifest.csv",
            "SHA256SUMS",
            *(f"packets/{filename}" for filename in PACKET_FILENAMES),
        ]
        actual: dict[str, bytes] = {}
        first_reads: dict[str, _ReadFile] = {}
        identities: dict[FileIdentity, str] = {}
        for relative in relative_files:
            if relative.startswith("packets/"):
                directory_fd = packets_fd
                name = relative.removeprefix("packets/")
            else:
                directory_fd = root_fd
                name = relative
            read_file = _read_regular_file_at(
                directory_fd,
                name,
                f"reviewer release/{relative}",
            )
            if read_file.mode != 0o644:
                raise SnapshotError(
                    f"{relative}: file mode {read_file.mode:#o} != 0o644"
                )
            if read_file.identity in identities:
                raise SnapshotError(
                    f"{relative}: file aliases {identities[read_file.identity]}"
                )
            identities[read_file.identity] = relative
            first_reads[relative] = read_file
            actual[relative] = read_file.payload

        release_manifest = _validate_release_manifest_payload(
            actual["release_manifest.csv"],
            expected_manifest,
        )
        try:
            coordinator_manifest = coordinator_snapshot["manifest.csv"]
            coordinator_protocol = coordinator_snapshot["protocol.md"]
            coordinator_profile = coordinator_snapshot[
                "execution_profile.json"
            ]
            coordinator_prompt = coordinator_snapshot[
                "reviewer_prompt_template.md"
            ]
        except KeyError as exc:
            raise SnapshotError(
                "authoritative coordinator snapshot is incomplete"
            ) from exc
        coordinator_manifest_rows = _validate_manifest_shape(
            coordinator_manifest
        )
        if {
            row["snapshot_sha256"] for row in coordinator_manifest_rows
        } != {release_manifest["coordinator_snapshot_sha256"]}:
            raise SnapshotError(
                "release coordinator manifest snapshot binding does not match"
            )
        if {
            row["protocol_sha256"] for row in coordinator_manifest_rows
        } != {release_manifest["protocol_sha256"]}:
            raise SnapshotError(
                "release coordinator manifest protocol binding does not match"
            )
        if {
            row["execution_profile_sha256"]
            for row in coordinator_manifest_rows
        } != {release_manifest["execution_profile_sha256"]}:
            raise SnapshotError(
                "release coordinator execution profile binding does not match"
            )
        if {
            row["prompt_template_sha256"] for row in coordinator_manifest_rows
        } != {release_manifest["prompt_template_sha256"]}:
            raise SnapshotError(
                "release coordinator prompt template binding does not match"
            )
        coordinator_packet_assignments = _manifest_packet_assignment_ids(
            coordinator_manifest_rows
        )
        release_packet_assignments = _manifest_packet_assignment_ids(
            coordinator_manifest_rows,
            phase=release_manifest["phase"],
        )
        if actual["protocol.md"] != coordinator_protocol:
            raise SnapshotError(
                "protocol.md does not match authoritative coordinator "
                "snapshot derivation"
            )
        if (
            _sha256(actual["protocol.md"])
            != release_manifest["protocol_sha256"]
        ):
            raise SnapshotError(
                "release manifest protocol binding does not match protocol.md"
            )
        if actual["execution_profile.json"] != coordinator_profile:
            raise SnapshotError(
                "execution_profile.json does not match authoritative coordinator"
            )
        if (
            _sha256(actual["execution_profile.json"])
            != release_manifest["execution_profile_sha256"]
        ):
            raise SnapshotError("release execution profile binding does not match")
        validate_execution_profile(actual["execution_profile.json"])
        if actual["reviewer_prompt_template.md"] != coordinator_prompt:
            raise SnapshotError(
                "reviewer_prompt_template.md does not match authoritative coordinator"
            )
        if (
            _sha256(actual["reviewer_prompt_template.md"])
            != release_manifest["prompt_template_sha256"]
        ):
            raise SnapshotError("release prompt template binding does not match")
        validate_reviewer_prompt_template(actual["reviewer_prompt_template.md"])

        checksum_inputs = {
            name: payload
            for name, payload in actual.items()
            if name != "SHA256SUMS"
        }
        expected_checksums = "".join(
            f"{_sha256(checksum_inputs[name])}  {name}\n"
            for name in sorted(checksum_inputs)
        ).encode("utf-8")
        if actual["SHA256SUMS"] != expected_checksums:
            raise SnapshotError("reviewer release checksum mismatch")

        assignment_ids: set[str] = set()
        for filename in PACKET_FILENAMES:
            relative = f"packets/{filename}"
            payload = actual[relative]
            try:
                coordinator_payload = coordinator_snapshot[relative]
            except KeyError as exc:
                raise SnapshotError(
                    "authoritative coordinator snapshot is incomplete"
                ) from exc
            _validate_packet_shape(
                coordinator_payload,
                f"coordinator {relative}",
                expected_batch_id=filename.removesuffix(".csv"),
                expected_assignment_ids=(
                    coordinator_packet_assignments[filename]
                ),
            )
            coordinator_rows = _read_csv_bytes(
                coordinator_payload,
                f"coordinator {relative}",
                PACKET_HEADER,
            )
            expected_payload = _csv_bytes(
                PACKET_HEADER,
                [
                    row
                    for row in coordinator_rows
                    if row["phase"] == release_manifest["phase"]
                ],
            )
            if payload != expected_payload:
                raise SnapshotError(
                    f"{relative}: content does not match authoritative "
                    "coordinator snapshot derivation"
                )
            _validate_packet_shape(
                payload,
                relative,
                expected_batch_id=filename.removesuffix(".csv"),
                expected_assignment_ids=release_packet_assignments[filename],
            )
            rows = _read_csv_bytes(payload, relative, PACKET_HEADER)
            for row_number, row in enumerate(rows, start=2):
                if row["phase"] != release_manifest["phase"]:
                    raise SnapshotError(
                        f"{relative}:{row_number}: phase does not match "
                        "release manifest"
                    )
                if (
                    row["snapshot_sha256"]
                    != release_manifest["coordinator_snapshot_sha256"]
                ):
                    raise SnapshotError(
                        f"{relative}:{row_number}: coordinator binding does "
                        "not match release manifest"
                    )
                assignment_id = row["assignment_id"]
                if assignment_id in assignment_ids:
                    raise SnapshotError(
                        f"{relative}:{row_number}: duplicate released "
                        f"assignment_id {assignment_id!r}"
                    )
                assignment_ids.add(assignment_id)
        expected_count = int(release_manifest["assignment_count"])
        if len(assignment_ids) != expected_count:
            raise SnapshotError(
                "reviewer release assignment count does not match "
                "release manifest"
            )

        _assert_directory_unchanged(
            root_fd,
            root_attestation,
            REVIEWER_RELEASE_ROOT_FILENAMES,
            "reviewer release directory",
        )
        _assert_directory_unchanged(
            packets_fd,
            packets_attestation,
            set(PACKET_FILENAMES),
            "reviewer release packets directory",
        )
        for relative in relative_files:
            if relative.startswith("packets/"):
                directory_fd = packets_fd
                name = relative.removeprefix("packets/")
            else:
                directory_fd = root_fd
                name = relative
            second = _read_regular_file_at(
                directory_fd,
                name,
                f"reviewer release/{relative}",
            )
            first = first_reads[relative]
            if (
                second.identity != first.identity
                or second.link_count != first.link_count
                or second.mode != first.mode
                or second.payload != first.payload
            ):
                raise SnapshotError(
                    f"reviewer release/{relative}: file changed after read"
                )
        _assert_directory_unchanged(
            root_fd,
            root_attestation,
            REVIEWER_RELEASE_ROOT_FILENAMES,
            "reviewer release directory",
        )
        _assert_directory_unchanged(
            packets_fd,
            packets_attestation,
            set(PACKET_FILENAMES),
            "reviewer release packets directory",
        )
        _recheck_directory_path(
            snapshot_dir.parent,
            parent_identity,
            "reviewer release parent directory",
        )
        if _identity_at(parent_fd, snapshot_dir.name) != root_identity:
            raise SnapshotError(
                f"{snapshot_dir}: reviewer release changed after read"
            )
        return actual
    finally:
        if packets_fd is not None:
            os.close(packets_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_release_paths_disjoint(
    snapshot_dir: Path, output_dir: Path
) -> None:
    snapshot_lexical = _absolute_lexical(snapshot_dir)
    output_lexical = _absolute_lexical(output_dir)
    try:
        snapshot_resolved = snapshot_lexical.resolve(strict=False)
        output_resolved = output_lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SnapshotError(
            "coordinator snapshot and reviewer release paths could not be resolved"
        ) from exc
    snapshot_forms = {snapshot_lexical, snapshot_resolved}
    output_forms = {output_lexical, output_resolved}
    if any(
        _paths_overlap(snapshot_form, output_form)
        for snapshot_form in snapshot_forms
        for output_form in output_forms
    ):
        raise SnapshotError(
            "coordinator snapshot and reviewer release must be disjoint paths"
        )


def _read_exact_flat_snapshot(
    snapshot_dir: Path,
    *,
    label: str,
    expected_names: frozenset[str],
) -> dict[str, bytes]:
    snapshot_dir = _absolute_lexical(snapshot_dir)
    _validate_version_path(snapshot_dir, label)
    parent_fd, parent_identity = _open_directory_fd(
        snapshot_dir.parent, f"{label} parent directory"
    )
    root_fd: int | None = None
    try:
        try:
            root_stat = os.stat(
                snapshot_dir.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise SnapshotError(f"{snapshot_dir}: {label} is missing") from exc
        if stat.S_ISLNK(root_stat.st_mode):
            raise SnapshotError(f"{snapshot_dir}: {label} must not be a symlink")
        if not stat.S_ISDIR(root_stat.st_mode):
            raise SnapshotError(f"{snapshot_dir}: {label} must be a directory")
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        root_fd = os.open(
            snapshot_dir.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd
        )
        if (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino) != root_identity:
            raise SnapshotError(f"{snapshot_dir}: {label} changed before read")
        root_attestation = _attest_directory_fd(
            root_fd, expected_names, label, 0o755
        )
        if root_attestation.identity != root_identity:
            raise SnapshotError(f"{snapshot_dir}: {label} identity changed")

        first_reads = {
            name: _read_regular_file_at(root_fd, name, f"{label}/{name}")
            for name in sorted(expected_names)
        }
        for name, read_file in first_reads.items():
            if read_file.mode != 0o644:
                raise SnapshotError(
                    f"{label}/{name}: mode {read_file.mode:#o} != 0o644"
                )
            if read_file.link_count != 1:
                raise SnapshotError(f"{label}/{name}: hard link count changed")

        for name, first in first_reads.items():
            second = _read_regular_file_at(root_fd, name, f"{label}/{name}")
            if (
                second.identity != first.identity
                or second.link_count != first.link_count
                or second.mode != first.mode
                or second.payload != first.payload
            ):
                raise SnapshotError(f"{label}/{name}: file changed after read")
        _assert_directory_unchanged(
            root_fd, root_attestation, expected_names, label
        )
        _recheck_directory_path(
            snapshot_dir.parent,
            parent_identity,
            f"{label} parent directory",
        )
        if _identity_at(parent_fd, snapshot_dir.name) != root_identity:
            raise SnapshotError(f"{snapshot_dir}: {label} changed after read")
        return {
            name: read_file.payload for name, read_file in first_reads.items()
        }
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _parse_canonical_json_payload(
    payload: bytes,
    *,
    label: str,
    byte_limit: int = 128 * 1024,
) -> dict[str, object]:
    if not isinstance(payload, bytes) or len(payload) > byte_limit:
        raise SnapshotError(f"{label} exceeds the canonical JSON byte limit")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise SnapshotError(f"{label} must be canonical UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{label} must be a JSON object")
    _validate_json_shape(value, label=label)
    if payload != _canonical_json_bytes(value):
        raise SnapshotError(f"{label} must use canonical JSON bytes")
    return value


def validate_reviewer_stage_snapshot(
    snapshot_dir: Path,
) -> dict[str, bytes]:
    """Validate one exact role-only pre-execution staging snapshot."""

    snapshot_dir = _absolute_lexical(Path(snapshot_dir))
    actual = _read_exact_flat_snapshot(
        snapshot_dir,
        label="reviewer execution stage",
        expected_names=REVIEWER_STAGE_ROOT_FILENAMES,
    )
    parent_stat = _require_real_directory(
        snapshot_dir.parent,
        "role-private staging parent",
    )
    if stat.S_IMODE(parent_stat.st_mode) != 0o700:
        raise SnapshotError(
            "role-private staging parent mode must be exactly 0o700"
        )

    rows = _read_csv_bytes(
        actual["stage_manifest.csv"],
        "stage_manifest.csv",
        STAGE_MANIFEST_HEADER,
    )
    if len(rows) != 1:
        raise SnapshotError("stage manifest must contain exactly one row")
    manifest = rows[0]
    if actual["stage_manifest.csv"] != _csv_bytes(
        STAGE_MANIFEST_HEADER,
        [manifest],
    ):
        raise SnapshotError("stage manifest must be canonical")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise SnapshotError("stage manifest version is invalid")
    for field in (
        "stage_snapshot_sha256",
        "reviewer_release_sha256",
        "coordinator_snapshot_sha256",
        "protocol_sha256",
        "packet_sha256",
        "execution_profile_sha256",
        "prompt_template_sha256",
        "configuration_sha256",
        "prompt_sha256",
        "user_instruction_sha256",
    ):
        if HEX_SHA256_PATTERN.fullmatch(manifest[field]) is None:
            raise SnapshotError(f"stage manifest {field} is invalid")
    role_id = manifest["role_id"]
    parent_match = _ROLE_PRIVATE_PARENT_PATTERN.fullmatch(
        snapshot_dir.parent.name
    )
    if (
        f"{role_id}.csv" not in PACKET_FILENAMES
        or parent_match is None
        or parent_match.group(1) != role_id
    ):
        raise SnapshotError(
            "stage role does not match its random role-private parent"
        )
    phase = manifest["phase"]
    if manifest["task"] != _reviewer_task(phase):
        raise SnapshotError("stage task does not match its phase")
    result_path = str(snapshot_dir.parent / f"{role_id}-result.csv")
    if manifest["stage_path"] != str(snapshot_dir):
        raise SnapshotError("stage manifest path does not match its location")
    if manifest["result_path"] != result_path:
        raise SnapshotError("stage result path is not canonical and absolute")

    profile = validate_execution_profile(actual["execution_profile.json"])
    validate_reviewer_prompt_template(
        actual["reviewer_prompt_template.md"]
    )
    packet_rows = _read_csv_bytes(
        actual["packet.csv"],
        "packet.csv",
        PACKET_HEADER,
    )
    if not packet_rows:
        raise SnapshotError("staged packet must not be empty")
    if (
        {row["phase"] for row in packet_rows} != {phase}
        or {row["snapshot_sha256"] for row in packet_rows}
        != {manifest["coordinator_snapshot_sha256"]}
    ):
        raise SnapshotError(
            "staged packet phase or coordinator binding is invalid"
        )
    _validate_packet_shape(
        actual["packet.csv"],
        "packet.csv",
        expected_batch_id=role_id,
        expected_assignment_ids={
            row["assignment_id"] for row in packet_rows
        },
    )
    if manifest["assignment_count"] != str(len(packet_rows)):
        raise SnapshotError("stage assignment count is invalid")

    exact_hashes = {
        "protocol_sha256": _sha256(actual["protocol.md"]),
        "packet_sha256": _sha256(actual["packet.csv"]),
        "execution_profile_sha256": _sha256(
            actual["execution_profile.json"]
        ),
        "prompt_template_sha256": _sha256(
            actual["reviewer_prompt_template.md"]
        ),
        "configuration_sha256": _sha256(
            actual["execution_configuration.json"]
        ),
        "prompt_sha256": _sha256(actual["reviewer_prompt.md"]),
        "user_instruction_sha256": _sha256(
            actual["reviewer_prompt.md"]
        ),
    }
    for field, expected in exact_hashes.items():
        if manifest[field] != expected:
            raise SnapshotError(f"stage manifest {field} binding is invalid")

    configuration = _parse_canonical_json_payload(
        actual["execution_configuration.json"],
        label="execution_configuration.json",
    )
    configuration_version = configuration.get("configuration_version")
    if configuration_version == "1":
        allowed_inclusion_criteria = None
    elif configuration_version == "2":
        criteria = configuration.get("allowed_inclusion_criteria")
        taxonomy = (
            {SCREENING_INCLUSION_CRITERION_KEY: criteria}
            if isinstance(criteria, list)
            else {}
        )
        try:
            allowed_inclusion_criteria = _resolve_inclusion_criteria(
                taxonomy,
                strict_new=True,
            )
        except SnapshotError as exc:
            raise SnapshotError(
                "allowed inclusion criteria are invalid"
            ) from exc
    else:
        raise SnapshotError("execution configuration version is invalid")
    release_manifest = {
        "phase": phase,
        "coordinator_snapshot_sha256": manifest[
            "coordinator_snapshot_sha256"
        ],
        "protocol_sha256": manifest["protocol_sha256"],
        "execution_profile_sha256": manifest[
            "execution_profile_sha256"
        ],
        "prompt_template_sha256": manifest[
            "prompt_template_sha256"
        ],
    }
    expected_configuration = _execution_configuration(
        release_manifest=release_manifest,
        reviewer_release_sha256=manifest[
            "reviewer_release_sha256"
        ],
        role_id=role_id,
        stage_path=snapshot_dir,
        profile=profile,
        packet_sha256=manifest["packet_sha256"],
        allowed_inclusion_criteria=allowed_inclusion_criteria,
    )
    if configuration != expected_configuration:
        raise SnapshotError(
            "execution configuration does not match stage derivation"
        )
    expected_prompt = render_reviewer_prompt(
        actual["reviewer_prompt_template.md"],
        role_id=role_id,
        stage_path=snapshot_dir,
        protocol_sha256=manifest["protocol_sha256"],
        packet_sha256=manifest["packet_sha256"],
    )
    if actual["reviewer_prompt.md"] != expected_prompt:
        raise SnapshotError("rendered reviewer prompt does not match derivation")

    core_artifacts = {
        name: actual[name]
        for name in (
            "execution_configuration.json",
            "execution_profile.json",
            "packet.csv",
            "protocol.md",
            "reviewer_prompt.md",
            "reviewer_prompt_template.md",
        )
    }
    expected_stage_sha256 = _stage_snapshot_sha256(
        artifacts=core_artifacts,
        reviewer_release_sha256=manifest[
            "reviewer_release_sha256"
        ],
        role_id=role_id,
    )
    if manifest["stage_snapshot_sha256"] != expected_stage_sha256:
        raise SnapshotError("stage snapshot digest is invalid")

    checksum_inputs = {
        name: payload
        for name, payload in actual.items()
        if name != "SHA256SUMS"
    }
    expected_checksums = "".join(
        f"{_sha256(checksum_inputs[name])}  {name}\n"
        for name in sorted(checksum_inputs)
    ).encode("utf-8")
    if actual["SHA256SUMS"] != expected_checksums:
        raise SnapshotError("stage checksum manifest is invalid")
    return actual


def release_snapshot(
    snapshot_dir: Path,
    phase: str,
    output_dir: Path,
    calibration_decision_snapshot: Path | None = None,
    calibration_result_snapshot: Path | None = None,
    calibration_reviewer_release_snapshot: Path | None = None,
) -> None:
    snapshot_dir = Path(snapshot_dir)
    output_dir = Path(output_dir)
    _assert_release_paths_disjoint(snapshot_dir, output_dir)
    if phase == "calibration":
        if (
            calibration_reviewer_release_snapshot is not None
            or calibration_decision_snapshot is not None
            or calibration_result_snapshot is not None
        ):
            raise SnapshotError(
                "calibration release does not accept calibration gate snapshots"
            )
        if __package__:
            from .screening_results import (
                capture_coordinator_snapshot,
                reattest_coordinator_snapshot,
            )
        else:
            from screening_results import (
                capture_coordinator_snapshot,
                reattest_coordinator_snapshot,
            )

        captured_coordinator = capture_coordinator_snapshot(snapshot_dir)
        coordinator = captured_coordinator.payloads
        manifest_rows = _validate_manifest_shape(coordinator["manifest.csv"])
        coordinator_binding = manifest_rows[0]
        expected_manifest = _release_manifest_row(
            phase=phase,
            coordinator_snapshot_sha256=coordinator_binding["snapshot_sha256"],
            execution_profile_sha256=coordinator_binding[
                "execution_profile_sha256"
            ],
            prompt_template_sha256=coordinator_binding[
                "prompt_template_sha256"
            ],
            protocol_sha256=coordinator_binding["protocol_sha256"],
        )
        artifacts = build_reviewer_release_artifacts(
            coordinator,
            phase,
        )
        _validate_release_manifest_payload(
            artifacts["release_manifest.csv"],
            expected_manifest,
        )
        _assert_release_paths_disjoint(snapshot_dir, output_dir)
        reattest_coordinator_snapshot(captured_coordinator)

        def post_publish_check() -> None:
            _assert_release_paths_disjoint(snapshot_dir, output_dir)
            reattest_coordinator_snapshot(captured_coordinator)
            validate_reviewer_release_snapshot(
                output_dir,
                expected_manifest=expected_manifest,
                coordinator_snapshot=coordinator,
            )
            _assert_release_paths_disjoint(snapshot_dir, output_dir)
            reattest_coordinator_snapshot(captured_coordinator)

        _publish_artifacts(
            output_dir,
            artifacts,
            post_publish_check=post_publish_check,
        )
        return
    if phase != "main":
        raise SnapshotError(f"unsupported reviewer release phase {phase!r}")
    if (
        calibration_reviewer_release_snapshot is None
        or calibration_decision_snapshot is None
        or calibration_result_snapshot is None
    ):
        raise SnapshotError(
            "main release requires --calibration-reviewer-release-snapshot, "
            "--calibration-result-snapshot, and "
            "--calibration-decision-snapshot"
        )

    calibration_release_snapshot = Path(
        calibration_reviewer_release_snapshot
    )
    result_snapshot = Path(calibration_result_snapshot)
    decision_snapshot = Path(calibration_decision_snapshot)
    if __package__:
        from .screening_results import (
            _reject_output_overlap,
            capture_calibration_release_tuple,
            reattest_calibration_release_tuple,
        )
    else:
        from screening_results import (
            _reject_output_overlap,
            capture_calibration_release_tuple,
            reattest_calibration_release_tuple,
        )

    protected_paths = (
        snapshot_dir,
        calibration_release_snapshot,
        result_snapshot,
        decision_snapshot,
    )
    _reject_output_overlap(output_dir, protected_paths)
    captured = capture_calibration_release_tuple(
        snapshot_dir,
        calibration_release_snapshot,
        result_snapshot,
        decision_snapshot,
    )
    gate = captured.decision
    if gate.decision["decision"] != "release":
        raise SnapshotError("calibration gate decision is not release")
    if gate.decision["systematic_ambiguity"] != "false":
        raise SnapshotError("calibration gate has systematic ambiguity")
    if Decimal(gate.decision["status_agreement"]) < Decimal("0.80"):
        raise SnapshotError("calibration gate agreement is below 0.80")

    expected_manifest = _release_manifest_row(
        phase=phase,
        coordinator_snapshot_sha256=captured.coordinator.snapshot_sha256,
        execution_profile_sha256=captured.coordinator.manifest[0][
            "execution_profile_sha256"
        ],
        prompt_template_sha256=captured.coordinator.manifest[0][
            "prompt_template_sha256"
        ],
        protocol_sha256=captured.coordinator.protocol_sha256,
        calibration_result_snapshot_sha256=(
            captured.calibration.snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            captured.decision.snapshot_sha256
        ),
    )
    artifacts = build_reviewer_release_artifacts(
        captured.coordinator.payloads,
        phase,
        calibration_result_snapshot_sha256=(
            captured.calibration.snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            captured.decision.snapshot_sha256
        ),
    )
    _validate_release_manifest_payload(
        artifacts["release_manifest.csv"],
        expected_manifest,
    )
    _reject_output_overlap(output_dir, protected_paths)
    reattest_calibration_release_tuple(captured)

    def post_publish_check() -> None:
        _reject_output_overlap(output_dir, protected_paths)
        reattest_calibration_release_tuple(captured)
        validate_reviewer_release_snapshot(
            output_dir,
            expected_manifest=expected_manifest,
            coordinator_snapshot=captured.coordinator.payloads,
        )
        _reject_output_overlap(output_dir, protected_paths)
        reattest_calibration_release_tuple(captured)

    _publish_artifacts(
        output_dir,
        artifacts,
        post_publish_check=post_publish_check,
    )


def _assert_stage_paths_disjoint(
    coordinator_snapshot_dir: Path,
    reviewer_release_snapshot_dir: Path,
    staging_root: Path,
) -> None:
    sources = (
        _absolute_lexical(coordinator_snapshot_dir),
        _absolute_lexical(reviewer_release_snapshot_dir),
    )
    target = _absolute_lexical(staging_root)
    try:
        source_forms = [
            {source, source.resolve(strict=False)} for source in sources
        ]
        target_forms = {target, target.resolve(strict=False)}
    except (OSError, RuntimeError) as exc:
        raise SnapshotError("staging paths could not be resolved") from exc
    if any(
        _paths_overlap(source_form, target_form)
        for forms in source_forms
        for source_form in forms
        for target_form in target_forms
    ):
        raise SnapshotError(
            "staging root must be disjoint from coordinator and reviewer release"
        )


def _validated_release_for_staging(
    coordinator_snapshot_dir: Path,
    reviewer_release_snapshot_dir: Path,
) -> tuple[dict[str, bytes], dict[str, bytes], Row]:
    coordinator = validate_snapshot(coordinator_snapshot_dir)
    manifest_input = _read_regular_file(
        Path(reviewer_release_snapshot_dir) / "release_manifest.csv",
        "reviewer release manifest input",
    )
    manifest_rows = _read_csv_bytes(
        manifest_input.payload,
        "release_manifest.csv",
        RELEASE_MANIFEST_HEADER,
    )
    if len(manifest_rows) != 1:
        raise SnapshotError(
            "release manifest must contain exactly one authorization row"
        )
    expected_manifest = _validate_release_manifest_payload(
        manifest_input.payload,
        manifest_rows[0],
    )
    reviewer_release = validate_reviewer_release_snapshot(
        reviewer_release_snapshot_dir,
        expected_manifest=expected_manifest,
        coordinator_snapshot=coordinator,
    )
    return coordinator, reviewer_release, expected_manifest


def _create_role_private_parent(
    staging_root: Path,
    role_id: str,
) -> Path:
    staging_root = _absolute_lexical(staging_root)
    _require_real_directory(staging_root, "staging root")
    for _ in range(128):
        name = f"{role_id}-{secrets.token_hex(16)}"
        private_parent = staging_root / name
        try:
            private_parent.mkdir(mode=0o700)
        except FileExistsError:
            continue
        try:
            os.chmod(private_parent, 0o700)
            parent_stat = _require_real_directory(
                private_parent,
                "role-private staging parent",
            )
            if stat.S_IMODE(parent_stat.st_mode) != 0o700:
                raise SnapshotError(
                    "role-private staging parent mode must be exactly 0o700"
                )
            return private_parent
        except BaseException:
            # Preserve failed staging paths for inspection rather than deleting them.
            raise
    raise SnapshotError("could not allocate a random role-private staging path")


def stage_reviewer_execution(
    coordinator_snapshot_dir: Path,
    reviewer_release_snapshot_dir: Path,
    role_id: str,
    staging_root: Path,
) -> Path:
    """Validate a release and publish one procedural role-private stage."""

    if f"{role_id}.csv" not in PACKET_FILENAMES:
        raise SnapshotError(f"unsupported screening role {role_id!r}")
    coordinator_snapshot_dir = Path(coordinator_snapshot_dir)
    reviewer_release_snapshot_dir = Path(reviewer_release_snapshot_dir)
    staging_root = Path(staging_root)
    _assert_stage_paths_disjoint(
        coordinator_snapshot_dir,
        reviewer_release_snapshot_dir,
        staging_root,
    )
    coordinator, reviewer_release, expected_manifest = (
        _validated_release_for_staging(
            coordinator_snapshot_dir,
            reviewer_release_snapshot_dir,
        )
    )
    taxonomy = _parse_taxonomy(
        coordinator["taxonomy.json"],
        strict_new=False,
    )
    allowed_inclusion_criteria = _resolve_inclusion_criteria(
        taxonomy,
        strict_new=False,
    )
    allowed_screening_statuses = _resolve_screening_result_statuses(
        taxonomy,
        strict_new=False,
    )
    coordinator_hashes = {
        name: _sha256(payload) for name, payload in coordinator.items()
    }
    release_hashes = {
        name: _sha256(payload) for name, payload in reviewer_release.items()
    }
    private_parent = _create_role_private_parent(staging_root, role_id)
    output_dir = private_parent / "v1"

    def post_publish_check() -> None:
        _assert_stage_paths_disjoint(
            coordinator_snapshot_dir,
            reviewer_release_snapshot_dir,
            staging_root,
        )
        current_coordinator = validate_snapshot(coordinator_snapshot_dir)
        if {
            name: _sha256(payload)
            for name, payload in current_coordinator.items()
        } != coordinator_hashes:
            raise SnapshotError("coordinator changed during staging")
        current_release = validate_reviewer_release_snapshot(
            reviewer_release_snapshot_dir,
            expected_manifest=expected_manifest,
            coordinator_snapshot=current_coordinator,
        )
        if {
            name: _sha256(payload)
            for name, payload in current_release.items()
        } != release_hashes:
            raise SnapshotError("reviewer release changed during staging")
        validate_reviewer_stage_snapshot(output_dir)

    try:
        artifacts = build_reviewer_stage_artifacts(
            reviewer_release,
            role_id,
            output_dir,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        _publish_artifacts(
            output_dir,
            artifacts,
            post_publish_check=post_publish_check,
        )
    except BaseException:
        # Preserve failed staging paths for inspection rather than deleting them.
        raise
    return output_dir


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or validate immutable screening input batches."
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--stage-role", action="store_true")
    parser.add_argument("--phase", choices=("calibration", "main"))
    parser.add_argument("--reviewer-release-snapshot", type=Path)
    parser.add_argument("--role-id")
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--conflicts", type=Path)
    parser.add_argument("--bibliography", type=Path)
    parser.add_argument("--citation-keys", type=Path)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--execution-profile", type=Path)
    parser.add_argument("--reviewer-prompt-template", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument(
        "--calibration-reviewer-release-snapshot",
        type=Path,
    )
    parser.add_argument("--calibration-decision-snapshot", type=Path)
    parser.add_argument("--calibration-result-snapshot", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    direct_values = {
        "--candidates": arguments.candidates,
        "--conflicts": arguments.conflicts,
        "--bibliography": arguments.bibliography,
        "--citation-keys": arguments.citation_keys,
        "--taxonomy": arguments.taxonomy,
        "--protocol": arguments.protocol,
        "--execution-profile": arguments.execution_profile,
        "--reviewer-prompt-template": arguments.reviewer_prompt_template,
        "--output-dir": arguments.output_dir,
    }
    stage_values = {
        "--reviewer-release-snapshot": (
            arguments.reviewer_release_snapshot
        ),
        "--role-id": arguments.role_id,
        "--staging-root": arguments.staging_root,
    }
    selected_modes = sum(
        (arguments.freeze, arguments.release, arguments.stage_role)
    )
    if selected_modes > 1:
        raise SnapshotError(
            "--freeze, --release, and --stage-role are mutually exclusive"
        )
    if arguments.stage_role:
        required = {
            "--snapshot-dir": arguments.snapshot_dir,
            **stage_values,
        }
        missing = [
            name for name, value in required.items() if value is None
        ]
        if missing:
            raise SnapshotError(
                "--stage-role requires " + ", ".join(missing)
            )
        forbidden = [
            name for name, value in direct_values.items() if value is not None
        ]
        if arguments.phase is not None:
            forbidden.append("--phase")
        if (
            arguments.calibration_reviewer_release_snapshot is not None
            or arguments.calibration_decision_snapshot is not None
            or arguments.calibration_result_snapshot is not None
        ):
            forbidden.append("calibration gate snapshots")
        if forbidden:
            raise SnapshotError(
                "--stage-role does not accept " + ", ".join(forbidden)
            )
        output = stage_reviewer_execution(
            arguments.snapshot_dir,
            arguments.reviewer_release_snapshot,
            arguments.role_id,
            arguments.staging_root,
        )
        print(output)
        return 0
    if arguments.release:
        if arguments.snapshot_dir is None:
            raise SnapshotError("--release requires --snapshot-dir")
        if arguments.phase is None:
            raise SnapshotError("--release requires --phase")
        if arguments.output_dir is None:
            raise SnapshotError("--release requires --output-dir")
        forbidden = [
            name
            for name, value in direct_values.items()
            if name != "--output-dir" and value is not None
        ]
        if forbidden:
            raise SnapshotError(
                "--release does not accept " + ", ".join(forbidden)
            )
        release_snapshot(
            arguments.snapshot_dir,
            arguments.phase,
            arguments.output_dir,
            calibration_decision_snapshot=(
                arguments.calibration_decision_snapshot
            ),
            calibration_result_snapshot=(
                arguments.calibration_result_snapshot
            ),
            calibration_reviewer_release_snapshot=(
                arguments.calibration_reviewer_release_snapshot
            ),
        )
        return 0

    if arguments.freeze:
        if arguments.phase is not None:
            raise SnapshotError("--freeze does not accept --phase")
        if (
            arguments.calibration_reviewer_release_snapshot is not None
            or arguments.calibration_decision_snapshot is not None
            or arguments.calibration_result_snapshot is not None
        ):
            raise SnapshotError(
                "--freeze does not accept calibration gate snapshots"
            )
        if arguments.snapshot_dir is not None:
            raise SnapshotError(
                "--freeze uses --output-dir and cannot use --snapshot-dir"
            )
        missing = [name for name, value in direct_values.items() if value is None]
        if missing:
            raise SnapshotError(f"--freeze requires {', '.join(missing)}")
        freeze_snapshot(
            candidates=arguments.candidates,
            conflicts=arguments.conflicts,
            bibliography=arguments.bibliography,
            citation_keys=arguments.citation_keys,
            taxonomy=arguments.taxonomy,
            protocol=arguments.protocol,
            execution_profile=arguments.execution_profile,
            reviewer_prompt_template=(
                arguments.reviewer_prompt_template
            ),
            output_dir=arguments.output_dir,
        )
        return 0

    if arguments.phase is not None:
        raise SnapshotError("validation does not accept --phase")
    if (
        arguments.calibration_decision_snapshot is not None
        or arguments.calibration_result_snapshot is not None
    ):
        raise SnapshotError(
            "validation does not accept calibration gate snapshots"
        )
    if arguments.snapshot_dir is None:
        raise SnapshotError(
            "validation requires --snapshot-dir (or use --freeze)"
        )
    supplied_direct = [
        name for name, value in direct_values.items() if value is not None
    ]
    if supplied_direct:
        raise SnapshotError(
            "validation accepts only --snapshot-dir; unexpected "
            + ", ".join(supplied_direct)
        )
    validate_snapshot(arguments.snapshot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
