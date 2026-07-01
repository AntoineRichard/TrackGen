from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    import paper.scripts.prepare_screening_batches as screening_batches
    import paper.scripts.screening_agreement as screening_agreement
    import paper.scripts.screening_results as screening_results
except ModuleNotFoundError:  # Direct execution from paper/scripts.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import paper.scripts.prepare_screening_batches as screening_batches
    import paper.scripts.screening_agreement as screening_agreement
    import paper.scripts.screening_results as screening_results


class ScreeningIntegrationError(ValueError):
    """A screening snapshot binding, adjudication, or projection is invalid."""


MANIFEST_VERSION = "1"
SCREENING_RESULT_HEADER = screening_results.RESULT_HEADER
SCREENING_AGREEMENT_HEADER = screening_agreement.AGREEMENT_REPORT_HEADER

EXECUTION_REGISTER_HEADER = (
    "execution_id",
    "role_id",
    "role_type",
    "context_id",
    "task",
    "work_item_id",
    "model_identifier",
    "model_version",
    "configuration_sha256",
    "prompt_sha256",
    "provider",
    "runtime",
    "tool_configuration",
    "retrieval_configuration",
    "decoding_parameters",
    "system_instruction_sha256",
    "developer_instruction_sha256",
    "user_instruction_sha256",
    "cache_isolation_statement",
    "started_on",
    "completed_on",
    "result_file_sha256",
    "human_role",
    "training_calibration_exposure",
    "automated_actions",
)

ADJUDICATION_HEADER = (
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "primary_snapshot_sha256",
    "assignment_ids",
    "adjudicator_id",
    "reviewer_ids",
    "decided_on",
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
    "resolution_evidence",
    "resolved_conflict_ids",
    "notes",
)

ADJUDICATION_MANIFEST_HEADER = (
    "manifest_version",
    "adjudication_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
    "main_result_snapshot_sha256",
    "primary_snapshot_sha256",
    "adjudication_file_sha256",
    "execution_registry_sha256",
    "row_count",
    "execution_row_count",
)

SCREENING_DECISIONS_HEADER = (
    *SCREENING_RESULT_HEADER,
    "adjudicated",
    "final_screening_status",
    "final_criterion",
    "final_exclusion_reason",
    *(f"adjudication_{field}" for field in ADJUDICATION_HEADER),
)

AUTHOR_VERIFICATION_HEADER = (
    "candidate_id",
    "primary_snapshot_sha256",
    "adjudication_snapshot_sha256",
    "decision_sha256",
    "evidence_versions_sha256",
    "deciding_locators_sha256",
    "verified_by",
    "verified_role",
    "verified_on",
    "verification_status",
    "verification_evidence",
)

PROJECTION_MANIFEST_HEADER = (
    "manifest_version",
    "projection_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
    "main_result_snapshot_sha256",
    "primary_snapshot_sha256",
    "adjudication_snapshot_sha256",
    "execution_registry_sha256",
    "citation_key_ledger_sha256",
    "author_verification_sha256",
    "candidates_sha256",
    "citation_keys_sha256",
    "conflicts_sha256",
    "screening_decisions_sha256",
    "screening_agreement_sha256",
    "candidate_count",
    "decision_row_count",
    "agreement_row_count",
)

_ADJUDICATION_FILES = (
    "adjudications.csv",
    "execution_registry.csv",
    "manifest.csv",
    "SHA256SUMS",
)
_PROJECTION_FILES = (
    "candidates.csv",
    "citation_keys.csv",
    "conflicts.csv",
    "screening_decisions.csv",
    "screening_agreement.csv",
    "author_verification.csv",
    "manifest.csv",
    "SHA256SUMS",
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_CITE_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._/+\-]*")
_CACHE_ISOLATION_STATEMENT = (
    "Fresh context; no shared conversation history, memory, ratings, "
    "results, or retrieval cache."
)
_LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT = (
    "Fresh context; no shared conversation history, memory, ratings, or "
    "results were supplied; provider retrieval-cache isolation was not exposed."
)
_PROVIDER_METADATA_LIMITATION_VALUE = "provider-not-exposed"
_PROVIDER_METADATA_LIMITATION_KEYS = frozenset(
    {
        "backend_model_version",
        "decoding_parameters",
        "developer_instruction_bytes",
        "retrieval_cache_isolation",
        "system_instruction_bytes",
    }
)
_REQUESTED_MODEL_VERSION_PREFIX = "requested:"

Row = dict[str, str]


@dataclass(frozen=True)
class _ScreeningContext:
    coordinator: screening_results.CoordinatorSnapshot
    coordinator_dir: Path
    candidates: tuple[Row, ...]
    conflicts: tuple[Row, ...]
    calibration_release_dir: Path
    calibration_result_dir: Path
    calibration: screening_results.PhaseResultSnapshot
    calibration_decision: screening_results.CalibrationDecisionSnapshot
    calibration_decision_dir: Path
    main_release_dir: Path
    main: screening_results.PhaseResultSnapshot
    main_result_dir: Path
    rows: tuple[Row, ...]
    coordinator_snapshot_sha256: str
    protocol_sha256: str
    primary_snapshot_sha256: str


@dataclass(frozen=True)
class AdjudicationSnapshot:
    directory: Path
    rows: tuple[Row, ...]
    execution_registry: tuple[Row, ...]
    snapshot_sha256: str
    coordinator_snapshot_sha256: str
    protocol_sha256: str
    calibration_result_snapshot_sha256: str
    calibration_decision_snapshot_sha256: str
    main_result_snapshot_sha256: str
    primary_snapshot_sha256: str
    execution_registry_sha256: str
    manifest: Row
    fingerprints: tuple[screening_results.FileFingerprint, ...]


@dataclass(frozen=True)
class ScreeningIntegrationResult:
    candidates: tuple[Row, ...]
    citation_keys: tuple[Row, ...]
    conflicts: tuple[Row, ...]
    screening_decisions: tuple[Row, ...]
    screening_agreement: tuple[Row, ...]
    coordinator_snapshot_sha256: str
    protocol_sha256: str
    calibration_result_snapshot_sha256: str
    calibration_decision_snapshot_sha256: str
    main_result_snapshot_sha256: str
    primary_snapshot_sha256: str
    adjudication_snapshot_sha256: str
    execution_registry_sha256: str
    citation_key_ledger_sha256: str


@dataclass(frozen=True)
class ScreeningProjectionSnapshot:
    directory: Path
    snapshot_sha256: str
    coordinator_snapshot_sha256: str
    protocol_sha256: str
    calibration_result_snapshot_sha256: str
    calibration_decision_snapshot_sha256: str
    main_result_snapshot_sha256: str
    primary_snapshot_sha256: str
    adjudication_snapshot_sha256: str
    execution_registry_sha256: str
    citation_key_ledger_sha256: str
    author_verification_sha256: str
    candidate_count: int
    decision_row_count: int
    agreement_row_count: int
    manifest: Row
    fingerprints: tuple[screening_results.FileFingerprint, ...]


@dataclass(frozen=True)
class _CapturedIntegrationInputs:
    context: _ScreeningContext
    adjudication: AdjudicationSnapshot
    execution_register: screening_results.CapturedInput
    citation_key_ledger: screening_results.CapturedInput


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ScreeningIntegrationError:
        raise
    except (
        screening_batches.SnapshotError,
        screening_results.ScreeningResultError,
        screening_agreement.ScreeningAgreementError,
        OSError,
    ) as exc:
        raise ScreeningIntegrationError(str(exc)) from exc


def _canonical_snapshot_path(path: Path, label: str) -> Path:
    try:
        canonical = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ScreeningIntegrationError(
            f"{path}: {label} path could not be resolved after validation"
        ) from exc
    if not canonical.is_absolute():
        raise ScreeningIntegrationError(f"{path}: {label} path is not absolute")
    return canonical


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def combined_primary_snapshot_sha256(
    calibration_result_snapshot_sha256: str,
    main_result_snapshot_sha256: str,
) -> str:
    return _call(
        screening_agreement.combined_primary_snapshot_sha256,
        calibration_result_snapshot_sha256,
        main_result_snapshot_sha256,
    )


def _parse_csv(
    payload: bytes,
    label: str,
    header: tuple[str, ...],
    *,
    no_blank_cells: bool = True,
) -> list[Row]:
    return _call(
        screening_results.parse_canonical_csv,
        payload,
        label,
        header,
        no_blank_cells=no_blank_cells,
    )


def _csv_bytes(header: tuple[str, ...], rows: Sequence[Mapping[str, str]]) -> bytes:
    return _call(screening_results.render_canonical_csv, header, list(rows))


def _checksums(artifacts: dict[str, bytes]) -> bytes:
    return _call(screening_results.render_sha256sums, artifacts)


def _utf8(value: str) -> bytes:
    return value.encode("utf-8")


def _same_phase_snapshot(
    captured: screening_results.PhaseResultSnapshot,
    authoritative: screening_results.PhaseResultSnapshot,
) -> bool:
    return all(
        getattr(captured, field) == getattr(authoritative, field)
        for field in (
            "directory",
            "phase",
            "rows",
            "snapshot_sha256",
            "coordinator_snapshot_sha256",
            "protocol_sha256",
            "reviewer_release_sha256",
            "manifest",
            "fingerprints",
        )
    )


def _same_calibration_decision(
    captured: screening_results.CalibrationDecisionSnapshot,
    authoritative: screening_results.CalibrationDecisionSnapshot,
) -> bool:
    return all(
        getattr(captured, field) == getattr(authoritative, field)
        for field in (
            "directory",
            "decision",
            "snapshot_sha256",
            "coordinator_snapshot_sha256",
            "calibration_result_snapshot_sha256",
            "manifest",
            "fingerprints",
        )
    )


def _reattest_context(
    context: _ScreeningContext,
    *fingerprint_groups: Sequence[screening_results.FileFingerprint],
) -> None:
    _call(screening_results.reattest_coordinator_snapshot, context.coordinator)
    for captured, release_dir, result_dir in (
        (
            context.calibration,
            context.calibration_release_dir,
            context.calibration_result_dir,
        ),
        (
            context.main,
            context.main_release_dir,
            context.main_result_dir,
        ),
    ):
        validation_kwargs: dict[str, object] = {
            "coordinator": context.coordinator,
            "reviewer_release_snapshot_dir": release_dir,
        }
        if captured.phase == "main":
            validation_kwargs.update(
                {
                    "calibration_reviewer_release_snapshot_dir": (
                        context.calibration_release_dir
                    ),
                    "calibration_result_snapshot_dir": (
                        context.calibration_result_dir
                    ),
                    "calibration_decision_snapshot_dir": (
                        context.calibration_decision_dir
                    ),
                }
            )
        authoritative = _call(
            screening_results.validate_phase_result_snapshot,
            result_dir,
            **validation_kwargs,
        )
        if not _same_phase_snapshot(captured, authoritative):
            raise ScreeningIntegrationError(
                f"{captured.phase} result snapshot changed after capture"
            )
    authoritative_decision = _call(
        screening_results.validate_calibration_decision_snapshot,
        context.calibration_decision_dir,
        coordinator_snapshot_dir=context.coordinator_dir,
        calibration_reviewer_release_snapshot_dir=(
            context.calibration_release_dir
        ),
        calibration_result_snapshot_dir=context.calibration_result_dir,
    )
    if not _same_calibration_decision(
        context.calibration_decision, authoritative_decision
    ):
        raise ScreeningIntegrationError(
            "calibration decision snapshot changed after capture"
        )
    if authoritative_decision.decision["decision"] != "release":
        raise ScreeningIntegrationError(
            "main integration requires a passing calibration release decision"
        )
    _call(
        screening_results.reattest_snapshot_set,
        context.coordinator,
        (
            context.calibration.fingerprints,
            context.calibration_decision.fingerprints,
            context.main.fingerprints,
            *fingerprint_groups,
        ),
    )


def _load_context(
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
) -> _ScreeningContext:
    coordinator_dir = Path(coordinator_snapshot_dir)
    producer_payloads = _call(
        screening_batches.validate_snapshot, coordinator_dir
    )
    coordinator = _call(
        screening_results.capture_coordinator_snapshot, coordinator_dir
    )
    if {
        name: _sha256(payload)
        for name, payload in producer_payloads.items()
    } != {
        name: _sha256(payload)
        for name, payload in coordinator.payloads.items()
    }:
        raise ScreeningIntegrationError(
            "coordinator changed between authoritative validation and capture"
        )
    payloads = coordinator.payloads
    candidates = _parse_csv(
        payloads["candidates.csv"],
        "coordinator candidates.csv",
        screening_batches.CANDIDATE_HEADER,
        no_blank_cells=False,
    )
    conflicts = _parse_csv(
        payloads["conflicts.csv"],
        "coordinator conflicts.csv",
        screening_batches.CONFLICT_HEADER,
        no_blank_cells=False,
    )
    calibration_release_dir = Path(
        calibration_reviewer_release_snapshot_dir
    )
    main_release_dir = Path(main_reviewer_release_snapshot_dir)
    calibration = _call(
        screening_results.validate_phase_result_snapshot,
        Path(calibration_result_snapshot_dir),
        coordinator=coordinator,
        reviewer_release_snapshot_dir=calibration_release_dir,
    )
    calibration_decision = _call(
        screening_results.validate_calibration_decision_snapshot,
        Path(calibration_decision_snapshot_dir),
        coordinator_snapshot_dir=coordinator.directory,
        calibration_reviewer_release_snapshot_dir=calibration_release_dir,
        calibration_result_snapshot_dir=calibration.directory,
    )
    if calibration_decision.decision["decision"] != "release":
        raise ScreeningIntegrationError(
            "main integration requires a passing calibration release decision"
        )
    main = _call(
        screening_results.validate_phase_result_snapshot,
        Path(main_result_snapshot_dir),
        coordinator=coordinator,
        reviewer_release_snapshot_dir=main_release_dir,
        calibration_reviewer_release_snapshot_dir=(
            calibration_release_dir
        ),
        calibration_result_snapshot_dir=calibration.directory,
        calibration_decision_snapshot_dir=(
            calibration_decision.directory
        ),
    )
    if calibration.phase != "calibration":
        raise ScreeningIntegrationError(
            "calibration result snapshot must have phase 'calibration'"
        )
    if main.phase != "main":
        raise ScreeningIntegrationError(
            "main result snapshot must have phase 'main'"
        )
    calibration_ids = {row["candidate_id"] for row in calibration.rows}
    main_ids = {row["candidate_id"] for row in main.rows}
    candidate_ids = {row["candidate_id"] for row in candidates}
    selected_ids = set(coordinator.calibration_candidate_ids)
    if (
        calibration_ids != selected_ids
        or main_ids != candidate_ids - selected_ids
        or calibration_ids & main_ids
    ):
        raise ScreeningIntegrationError(
            "phase result snapshots do not match calibration_selection.csv"
        )
    for phase_snapshot in (calibration, main):
        if (
            phase_snapshot.coordinator_snapshot_sha256
            != coordinator.snapshot_sha256
            or phase_snapshot.protocol_sha256 != coordinator.protocol_sha256
        ):
            raise ScreeningIntegrationError(
                "phase result snapshots do not share the coordinator and protocol"
            )
    rows = tuple(
        sorted(
            (*calibration.rows, *main.rows),
            key=lambda row: (
                _utf8(row["candidate_id"]),
                _utf8(row["assignment_id"]),
            ),
        )
    )
    coordinator_dir = _canonical_snapshot_path(
        coordinator.directory,
        "coordinator snapshot",
    )
    calibration_release_dir = _canonical_snapshot_path(
        calibration_release_dir,
        "calibration reviewer release snapshot",
    )
    calibration_result_dir = _canonical_snapshot_path(
        calibration.directory,
        "calibration result snapshot",
    )
    calibration_decision_dir = _canonical_snapshot_path(
        calibration_decision.directory,
        "calibration decision snapshot",
    )
    main_release_dir = _canonical_snapshot_path(
        main_release_dir,
        "main reviewer release snapshot",
    )
    main_result_dir = _canonical_snapshot_path(
        main.directory,
        "main result snapshot",
    )

    context = _ScreeningContext(
        coordinator=coordinator,
        coordinator_dir=coordinator_dir,
        candidates=tuple(candidates),
        conflicts=tuple(conflicts),
        calibration_release_dir=calibration_release_dir,
        calibration_result_dir=calibration_result_dir,
        calibration=calibration,
        calibration_decision=calibration_decision,
        calibration_decision_dir=calibration_decision_dir,
        main_release_dir=main_release_dir,
        main=main,
        main_result_dir=main_result_dir,
        rows=rows,
        coordinator_snapshot_sha256=coordinator.snapshot_sha256,
        protocol_sha256=coordinator.protocol_sha256,
        primary_snapshot_sha256=combined_primary_snapshot_sha256(
            calibration.snapshot_sha256, main.snapshot_sha256
        ),
    )
    _reattest_context(context)
    return context


def _context_protected_paths(context: _ScreeningContext) -> tuple[Path, ...]:
    return (
        context.coordinator_dir,
        context.calibration_release_dir,
        context.calibration_result_dir,
        context.calibration_decision_dir,
        context.main_release_dir,
        context.main_result_dir,
    )


def _context_protected_fingerprints(
    context: _ScreeningContext,
) -> tuple[screening_results.FileFingerprint, ...]:
    return (
        *context.coordinator.fingerprints,
        *context.calibration.fingerprints,
        *context.calibration_decision.fingerprints,
        *context.main.fingerprints,
    )


def _ratings_by_candidate(context: _ScreeningContext) -> dict[str, tuple[Row, Row]]:
    grouped: defaultdict[str, list[Row]] = defaultdict(list)
    for row in context.rows:
        grouped[row["candidate_id"]].append(row)
    result: dict[str, tuple[Row, Row]] = {}
    for candidate_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: _utf8(row["assignment_id"]))
        result[candidate_id] = (ordered[0], ordered[1])
    return result


def _unresolved_screening_conflicts(
    context: _ScreeningContext,
) -> dict[str, tuple[Row, ...]]:
    grouped: defaultdict[str, list[Row]] = defaultdict(list)
    candidate_ids = {row["candidate_id"] for row in context.candidates}
    for row in context.conflicts:
        if (
            row["record_type"] == "candidate"
            and row["field"] == "screening_status"
            and row["resolution"] == ""
        ):
            if row["record_key"] not in candidate_ids:
                raise ScreeningIntegrationError(
                    f"unresolved conflict {row['conflict_id']!r} targets "
                    "an unknown candidate"
                )
            if row["resolver"]:
                raise ScreeningIntegrationError(
                    f"unresolved conflict {row['conflict_id']!r} has "
                    "resolution metadata"
                )
            grouped[row["record_key"]].append(row)
    return {
        candidate_id: tuple(
            sorted(rows, key=lambda row: _utf8(row["conflict_id"]))
        )
        for candidate_id, rows in grouped.items()
    }


def _normalize_exclusion_reason(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ScreeningIntegrationError(
            "excluded rating has an empty normalized exclusion reason"
        )
    return normalized


def _has_exclusion_reason_disagreement(
    pair: tuple[Row, Row],
) -> bool:
    first, second = pair
    return (
        first["screening_status"] == "excluded"
        and second["screening_status"] == "excluded"
        and first["criterion"] == second["criterion"]
        and _normalize_exclusion_reason(first["exclusion_reason"])
        != _normalize_exclusion_reason(second["exclusion_reason"])
    )


def _adjudication_trigger_ids(
    pair: tuple[Row, Row],
    *,
    has_unresolved_conflict: bool,
) -> tuple[str, ...]:
    first, second = pair
    triggers: list[str] = []
    if first["screening_status"] != second["screening_status"]:
        triggers.append("A1")
    if first["criterion"] != second["criterion"]:
        triggers.append("A2")
    if _has_exclusion_reason_disagreement(pair):
        triggers.append("A3")
    if has_unresolved_conflict:
        triggers.append("A4")
    return tuple(triggers)


def _required_adjudications(
    context: _ScreeningContext,
    ratings: dict[str, tuple[Row, Row]],
    unresolved: dict[str, tuple[Row, ...]],
) -> set[str]:
    del context
    required = set(unresolved)
    for candidate_id, pair in ratings.items():
        if _adjudication_trigger_ids(
            pair,
            has_unresolved_conflict=candidate_id in unresolved,
        ):
            required.add(candidate_id)
    return required


def _resolved_conflict_ids(
    row: Row,
    context: _ScreeningContext,
    unresolved: dict[str, tuple[Row, ...]],
    *,
    context_label: str,
) -> tuple[str, ...]:
    conflicts_by_id: dict[str, Row] = {}
    for conflict in context.conflicts:
        conflict_id = conflict["conflict_id"]
        if conflict_id in conflicts_by_id:
            raise ScreeningIntegrationError(
                f"coordinator has duplicate conflict_id {conflict_id!r}"
            )
        conflicts_by_id[conflict_id] = conflict
    allowed = {
        conflict["conflict_id"]
        for conflict in unresolved.get(row["candidate_id"], ())
    }

    value = row["resolved_conflict_ids"]
    identifiers = () if value == "NR" else tuple(value.split(";"))
    if value != "NR" and (
        any(not identifier or identifier == "NR" for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or list(identifiers) != sorted(identifiers, key=_utf8)
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: resolved_conflict_ids must be nonempty, unique, "
            "and in UTF-8 byte order"
        )
    if set(identifiers) != allowed:
        raise ScreeningIntegrationError(
            f"{context_label}: resolved_conflict_ids must exactly resolve all "
            f"unresolved screening conflicts; expected={sorted(allowed)}, "
            f"actual={sorted(identifiers)}"
        )

    for identifier in identifiers:
        conflict = conflicts_by_id.get(identifier)
        if conflict is None:
            raise ScreeningIntegrationError(
                f"{context_label}: unknown resolved conflict {identifier!r}"
            )
        if (
            conflict["record_type"] != "candidate"
            or conflict["record_key"] != row["candidate_id"]
            or conflict["field"] != "screening_status"
            or conflict["resolution"] != ""
            or conflict["resolver"] != ""
            or conflict["resolution_evidence"] != ""
        ):
            raise ScreeningIntegrationError(
                f"{context_label}: conflict {identifier!r} is not an "
                "unresolved screening conflict for this candidate"
            )
    return identifiers


def _is_substantive_source_fact(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).strip()
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    generic = {
        "source specific fact",
        "material evidence supports the final decision",
        "the source supports the final decision",
        "deciding source evidence",
    }
    return (
        normalized != "NR"
        and len(normalized) >= 48
        and len(words) >= 8
        and normalized.casefold() not in generic
    )


def _normalize_evidence_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _alphabetic_words(value: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", value, flags=re.UNICODE)


def _normalized_value_pattern(value: str) -> re.Pattern[str]:
    normalized = _normalize_evidence_text(value).casefold()
    return re.compile(
        rf"(?<!\w){re.escape(normalized)}(?!\w)",
        flags=re.UNICODE,
    )


def _contains_normalized_value(text: str, value: str) -> bool:
    return _normalized_value_pattern(value).search(text) is not None


def _validate_comparison_analysis(
    comparison: str,
    row: Row,
    pair: tuple[Row, Row],
    resolved_conflicts: tuple[Row, ...],
    trigger_ids: tuple[str, ...],
    deciding_fact: str,
    *,
    context_label: str,
) -> None:
    normalized = _normalize_evidence_text(comparison)
    folded = normalized.casefold()
    words = _alphabetic_words(normalized)
    folded_words = [word.casefold() for word in words]
    generic_phrases = (
        "complete token inventory",
        "token inventory",
        "token dump",
        "all listed values",
        "all required tokens",
        "all required values",
        "generic comparison",
        "generic rationale",
        "boilerplate rationale",
    )
    if (
        len(normalized) < 120
        or len(words) < 18
        or len(set(folded_words)) < 12
        or not _contains_normalized_value(folded, row["candidate_id"])
        or "whereas" not in folded_words
        or any(phrase in folded for phrase in generic_phrases)
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: comparison analysis must be candidate-specific, "
            "comparative, and substantive rather than a generic token inventory"
        )

    if "A1" in trigger_ids:
        statuses = {rating["screening_status"] for rating in pair}
        if any(
            not _contains_normalized_value(folded, status)
            for status in statuses
        ):
            raise ScreeningIntegrationError(
                f"{context_label}: A1 comparison analysis must include both "
                "differing raw statuses"
            )
    if "A2" in trigger_ids:
        criteria = {rating["criterion"] for rating in pair}
        if any(
            not _contains_normalized_value(folded, criterion)
            for criterion in criteria
        ):
            raise ScreeningIntegrationError(
                f"{context_label}: A2 comparison analysis must include both "
                "differing raw criteria"
            )
    if "A3" in trigger_ids:
        reason_words = [
            {
                word
                for word in _normalize_exclusion_reason(
                    rating["exclusion_reason"]
                ).split()
                if len(word) >= 5 and word.isalpha()
            }
            for rating in pair
        ]
        comparison_words = set(folded_words)
        for index in range(2):
            unique_words = reason_words[index] - reason_words[1 - index]
            if not unique_words or comparison_words.isdisjoint(unique_words):
                raise ScreeningIntegrationError(
                    f"{context_label}: A3 comparison analysis must include a "
                    "distinctive word from each raw exclusion reason"
                )
    if "A4" in trigger_ids:
        for conflict in resolved_conflicts:
            conflict_id = _normalize_evidence_text(
                conflict["conflict_id"]
            ).casefold()
            if not _contains_normalized_value(folded, conflict_id):
                raise ScreeningIntegrationError(
                    f"{context_label}: A4 comparison analysis must include "
                    f"resolved conflict {conflict['conflict_id']!r}"
                )

    known_values = {
        row["candidate_id"],
        row["input_sha256"],
        row["snapshot_sha256"],
        row["primary_snapshot_sha256"],
        row["assignment_ids"],
        row["adjudicator_id"],
        row["reviewer_ids"],
        row["screening_status"],
        row["criterion"],
        row["access_status"],
        row["source_urls"],
        row["evidence_archive_url"],
        row["evidence_sha256"],
        row["screening_locator"],
        row["exclusion_reason"],
        row["resolved_conflict_ids"],
        deciding_fact,
        "A1", "A2", "A3", "A4",
    }
    known_values.update(row["source_urls"].split(";"))
    known_values.update(row["assignment_ids"].split(";"))
    known_values.update(row["reviewer_ids"].split(";"))
    for rating in pair:
        known_values.update(
            (
                rating["candidate_id"],
                rating["assignment_id"],
                rating["batch_id"],
                rating["coder_id"],
                rating["screening_status"],
                rating["criterion"],
                rating["exclusion_reason"],
            )
        )
    for conflict in resolved_conflicts:
        known_values.update(
            (
                conflict["conflict_id"],
                conflict["field"],
                conflict["value_a"],
                conflict["value_b"],
            )
        )

    residual = folded
    normalized_known_values = {
        _normalize_evidence_text(value).casefold()
        for value in known_values
        if value and value != "NR"
    }
    for value in sorted(normalized_known_values, key=len, reverse=True):
        residual = _normalized_value_pattern(value).sub(" ", residual)
    residual_words = _alphabetic_words(residual)
    if (
        len(residual_words) < 10
        or len({word.casefold() for word in residual_words}) < 8
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: comparison analysis lacks independent "
            "alphabetic rationale after known decision tokens are removed"
        )


def _validate_adjudication_decision(
    row: Row,
    pair: tuple[Row, Row],
    resolved_conflicts: tuple[Row, ...],
    trigger_ids: tuple[str, ...],
    *,
    context_label: str,
) -> None:
    if (
        row["adjudicator_id"] == "NR"
        or not screening_results.is_valid_identifier(
            row["adjudicator_id"]
        )
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: adjudicator_id is invalid"
        )
    reviewer_ids = row["reviewer_ids"].split(";")
    if row["adjudicator_id"] in reviewer_ids:
        raise ScreeningIntegrationError(
            f"{context_label}: adjudicator must be distinct from both reviewers"
        )

    surrogate = {
        "assignment_id": pair[0]["assignment_id"],
        "phase": pair[0]["phase"],
        "candidate_id": row["candidate_id"],
        "input_sha256": row["input_sha256"],
        "snapshot_sha256": row["snapshot_sha256"],
        "batch_id": pair[0]["batch_id"],
        "coder_id": row["adjudicator_id"],
        "screened_on": row["decided_on"],
        "screening_status": row["screening_status"],
        "criterion": row["criterion"],
        "access_status": row["access_status"],
        "source_urls": row["source_urls"],
        "evidence_version": row["evidence_version"],
        "evidence_retrieved_on": row["evidence_retrieved_on"],
        "evidence_archive_url": row["evidence_archive_url"],
        "evidence_sha256": row["evidence_sha256"],
        "screening_locator": row["screening_locator"],
        "exclusion_reason": row["exclusion_reason"],
        "notes": row["notes"],
    }
    _call(
        screening_results.validate_result_decision,
        surrogate,
        context=context_label,
    )

    evidence_text = row["resolution_evidence"]
    if evidence_text == "NR" or len(evidence_text) < 120:
        raise ScreeningIntegrationError(
            f"{context_label}: resolution_evidence must be a substantive "
            "canonical JSON object"
        )
    evidence = _parse_canonical_json_object(
        evidence_text,
        field="resolution_evidence",
        context_label=context_label,
    )
    assert evidence is not None
    required_keys = {
        "schema_version",
        "raw_ratings",
        "controlling_rules",
        "raw_exclusion_reasons",
        "resolved_conflicts",
        "final_decision",
        "deciding_fact",
        "source_url",
        "deciding_locator",
        "comparison_analysis",
    }
    if set(evidence) != required_keys or evidence["schema_version"] != "1":
        raise ScreeningIntegrationError(
            f"{context_label}: resolution_evidence has an invalid schema"
        )

    expected_raw_ratings = [
        {
            "assignment_id": rating["assignment_id"],
            "criterion": rating["criterion"],
            "screening_status": rating["screening_status"],
        }
        for rating in pair
    ]
    expected_raw_reasons = (
        [
            {
                "assignment_id": rating["assignment_id"],
                "reason": rating["exclusion_reason"],
            }
            for rating in pair
        ]
        if "A3" in trigger_ids
        else []
    )
    expected_conflicts = [
        {
            "conflict_id": conflict["conflict_id"],
            "field": conflict["field"],
            "value_a": conflict["value_a"],
            "value_b": conflict["value_b"],
        }
        for conflict in resolved_conflicts
    ]
    exact_bindings = {
        "raw_ratings": expected_raw_ratings,
        "controlling_rules": list(trigger_ids),
        "raw_exclusion_reasons": expected_raw_reasons,
        "resolved_conflicts": expected_conflicts,
        "final_decision": {
            "criterion": row["criterion"],
            "screening_status": row["screening_status"],
        },
        "deciding_locator": row["screening_locator"],
    }
    for field, expected in exact_bindings.items():
        if evidence[field] != expected:
            raise ScreeningIntegrationError(
                f"{context_label}: resolution_evidence {field} must exactly "
                "match the authoritative screening data"
            )

    source_url = evidence["source_url"]
    if (
        not isinstance(source_url, str)
        or source_url not in row["source_urls"].split(";")
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: resolution_evidence source_url must exactly "
            "match one complete canonical source URL"
        )

    deciding_fact_object = evidence["deciding_fact"]
    if (
        not isinstance(deciding_fact_object, dict)
        or set(deciding_fact_object) != {"kind", "text"}
        or not isinstance(deciding_fact_object["kind"], str)
        or not isinstance(deciding_fact_object["text"], str)
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: resolution_evidence deciding_fact is invalid"
        )
    deciding_fact = deciding_fact_object["text"]
    if row["screening_status"] == "excluded":
        if deciding_fact_object != {
            "kind": "exclusion_reason",
            "text": row["exclusion_reason"],
        }:
            raise ScreeningIntegrationError(
                f"{context_label}: resolution_evidence deciding_fact must "
                "exactly bind the final exclusion reason"
            )
    elif (
        deciding_fact_object["kind"] != "transfer_source_fact"
        or not _is_substantive_source_fact(deciding_fact)
    ):
        raise ScreeningIntegrationError(
            f"{context_label}: resolution_evidence requires a substantive "
            "non-NR transfer/source fact"
        )

    comparison = evidence["comparison_analysis"]
    if not isinstance(comparison, str):
        raise ScreeningIntegrationError(
            f"{context_label}: resolution_evidence comparison_analysis "
            "must be a string"
        )
    _validate_comparison_analysis(
        comparison,
        row,
        pair,
        resolved_conflicts,
        trigger_ids,
        deciding_fact,
        context_label=context_label,
    )


def _validate_adjudication_rows(
    rows: Sequence[Row],
    context: _ScreeningContext,
) -> tuple[Row, ...]:
    ratings = _ratings_by_candidate(context)
    unresolved = _unresolved_screening_conflicts(context)
    required = _required_adjudications(context, ratings, unresolved)
    by_candidate: dict[str, Row] = {}
    for row_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"]
        if candidate_id in by_candidate:
            raise ScreeningIntegrationError(
                f"adjudications.csv:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        pair = ratings.get(candidate_id)
        if pair is None:
            raise ScreeningIntegrationError(
                f"adjudications.csv:{row_number}: unknown candidate_id "
                f"{candidate_id!r}"
            )
        assignment_ids = ";".join(
            rating["assignment_id"] for rating in pair
        )
        reviewer_ids = ";".join(rating["coder_id"] for rating in pair)
        expected = {
            "input_sha256": pair[0]["input_sha256"],
            "snapshot_sha256": context.coordinator_snapshot_sha256,
            "primary_snapshot_sha256": context.primary_snapshot_sha256,
            "assignment_ids": assignment_ids,
            "reviewer_ids": reviewer_ids,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ScreeningIntegrationError(
                    f"adjudications.csv:{row_number}: {field} must exactly "
                    "match the sealed ratings and conflicts"
                )
        resolved_conflict_ids = _resolved_conflict_ids(
            row,
            context,
            unresolved,
            context_label=f"adjudications.csv:{row_number}",
        )
        unresolved_by_id = {
            conflict["conflict_id"]: conflict
            for conflict in unresolved.get(candidate_id, ())
        }
        resolved_conflicts = tuple(
            unresolved_by_id[conflict_id]
            for conflict_id in resolved_conflict_ids
        )
        trigger_ids = _adjudication_trigger_ids(
            pair,
            has_unresolved_conflict=candidate_id in unresolved,
        )
        _validate_adjudication_decision(
            row,
            pair,
            resolved_conflicts,
            trigger_ids,
            context_label=f"adjudications.csv:{row_number}",
        )
        by_candidate[candidate_id] = dict(row)

    actual = set(by_candidate)
    if actual != required:
        raise ScreeningIntegrationError(
            "adjudication coverage mismatch; "
            f"missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )
    return tuple(
        by_candidate[candidate_id]
        for candidate_id in sorted(by_candidate, key=_utf8)
    )


def _execution_registry_sort_key(row: Mapping[str, str]) -> tuple[bytes, ...]:
    return (
        _utf8(row["task"]),
        _utf8(row["work_item_id"]),
        _utf8(row["role_id"]),
    )


def _parse_canonical_json_object(
    value: str,
    *,
    field: str,
    context_label: str,
    allow_nr: bool = False,
) -> dict[str, object] | None:
    if value == "NR" and allow_nr:
        return None

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant!r}")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ScreeningIntegrationError(
            f"{context_label}: {field} must be a canonical JSON object"
        ) from exc
    if not isinstance(parsed, dict) or value != canonical:
        raise ScreeningIntegrationError(
            f"{context_label}: {field} must be a canonical JSON object"
        )
    return parsed


def _validate_canonical_json_object(
    value: str,
    *,
    field: str,
    context_label: str,
    allow_nr: bool = False,
) -> None:
    _parse_canonical_json_object(
        value,
        field=field,
        context_label=context_label,
        allow_nr=allow_nr,
    )


def _provider_metadata_limitations(
    tool_configuration: Mapping[str, object],
    *,
    context_label: str,
) -> frozenset[str]:
    field = "provider_metadata_limitations"
    if field not in tool_configuration:
        return frozenset()
    limitations = tool_configuration[field]
    if not isinstance(limitations, dict) or not limitations:
        raise ScreeningIntegrationError(
            f"{context_label}: {field} must be a nonempty canonical JSON object"
        )
    unknown = set(limitations) - _PROVIDER_METADATA_LIMITATION_KEYS
    if unknown:
        raise ScreeningIntegrationError(
            f"{context_label}: {field} contains unknown keys: {sorted(unknown)}"
        )
    invalid_values = sorted(
        key
        for key, value in limitations.items()
        if value != _PROVIDER_METADATA_LIMITATION_VALUE
    )
    if invalid_values:
        raise ScreeningIntegrationError(
            f"{context_label}: {field} keys {invalid_values} must use exact "
            f"value {_PROVIDER_METADATA_LIMITATION_VALUE!r}"
        )
    return frozenset(limitations)


def _validate_execution_provenance(
    row: Row,
    *,
    context_label: str,
) -> None:
    role_type = row["role_type"]
    if role_type not in {"human", "automated", "hybrid"}:
        raise ScreeningIntegrationError(
            f"{context_label}: role_type must be human, automated, or hybrid"
        )
    automated_fields = (
        "model_identifier",
        "model_version",
        "configuration_sha256",
        "prompt_sha256",
        "provider",
        "runtime",
        "tool_configuration",
        "retrieval_configuration",
        "decoding_parameters",
        "system_instruction_sha256",
        "developer_instruction_sha256",
        "user_instruction_sha256",
        "cache_isolation_statement",
    )
    if role_type == "human":
        non_nr = [field for field in automated_fields if row[field] != "NR"]
        if non_nr:
            raise ScreeningIntegrationError(
                f"{context_label}: human-only role has automated provenance "
                f"in {non_nr}"
            )
    else:
        for field in ("model_identifier", "model_version", "provider", "runtime"):
            if (
                row[field] == "NR"
                or not screening_results.is_valid_identifier(row[field])
            ):
                raise ScreeningIntegrationError(
                    f"{context_label}: automated {field} is required"
                )
        for field in (
            "configuration_sha256",
            "prompt_sha256",
            "user_instruction_sha256",
        ):
            if _HASH_PATTERN.fullmatch(row[field]) is None:
                raise ScreeningIntegrationError(
                    f"{context_label}: automated {field} must be a SHA-256 digest"
                )
        tool_configuration = _parse_canonical_json_object(
            row["tool_configuration"],
            field="tool_configuration",
            context_label=context_label,
        )
        assert tool_configuration is not None
        limitations = _provider_metadata_limitations(
            tool_configuration,
            context_label=context_label,
        )
        model_version = row["model_version"]
        requested_model_version = model_version.startswith(
            _REQUESTED_MODEL_VERSION_PREFIX
        )
        backend_version_limited = "backend_model_version" in limitations
        if requested_model_version != backend_version_limited:
            raise ScreeningIntegrationError(
                f"{context_label}: provider_metadata_limitations "
                "'backend_model_version' must be present exactly when "
                "model_version uses requested:<alias-or-date>"
            )
        if requested_model_version and not screening_results.is_valid_identifier(
            model_version.removeprefix(_REQUESTED_MODEL_VERSION_PREFIX)
        ):
            raise ScreeningIntegrationError(
                f"{context_label}: provider_metadata_limitations "
                "'backend_model_version' requires model_version in exact "
                "requested:<alias-or-date> form"
            )
        conditional_hashes = {
            "system_instruction_sha256": "system_instruction_bytes",
            "developer_instruction_sha256": "developer_instruction_bytes",
        }
        for field, limitation in conditional_hashes.items():
            value_is_nr = row[field] == "NR"
            limitation_declared = limitation in limitations
            if value_is_nr != limitation_declared:
                raise ScreeningIntegrationError(
                    f"{context_label}: provider_metadata_limitations "
                    f"{limitation!r} must be present exactly when {field} is NR"
                )
            if not value_is_nr and _HASH_PATTERN.fullmatch(row[field]) is None:
                raise ScreeningIntegrationError(
                    f"{context_label}: automated {field} must be a SHA-256 digest"
                )
        _validate_canonical_json_object(
            row["retrieval_configuration"],
            field="retrieval_configuration",
            context_label=context_label,
        )
        decoding_limited = "decoding_parameters" in limitations
        if (row["decoding_parameters"] == "NR") != decoding_limited:
            raise ScreeningIntegrationError(
                f"{context_label}: provider_metadata_limitations "
                "'decoding_parameters' must be present exactly when "
                "decoding_parameters is NR"
            )
        if not decoding_limited:
            _validate_canonical_json_object(
                row["decoding_parameters"],
                field="decoding_parameters",
                context_label=context_label,
            )
        isolation = " ".join(
            unicodedata.normalize(
                "NFKC", row["cache_isolation_statement"]
            ).split()
        ).casefold()
        retrieval_cache_limited = "retrieval_cache_isolation" in limitations
        expected_statement = (
            _LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT
            if retrieval_cache_limited
            else _CACHE_ISOLATION_STATEMENT
        )
        expected_isolation = " ".join(
            unicodedata.normalize(
                "NFKC", expected_statement
            ).split()
        ).casefold()
        if isolation != expected_isolation:
            raise ScreeningIntegrationError(
                f"{context_label}: cache_isolation_statement and "
                "provider_metadata_limitations 'retrieval_cache_isolation' "
                "must select the matching authoritative declaration"
            )

    if role_type == "automated":
        for field in (
            "human_role",
            "training_calibration_exposure",
            "automated_actions",
        ):
            if row[field] != "NR":
                raise ScreeningIntegrationError(
                    f"{context_label}: {field} must be NR for an automated role"
                )
    else:
        if (
            row["human_role"] == "NR"
            or not screening_results.is_valid_identifier(row["human_role"])
        ):
            raise ScreeningIntegrationError(
                f"{context_label}: human_role is required"
            )
        exposure = unicodedata.normalize(
            "NFKC", row["training_calibration_exposure"]
        ).strip()
        if exposure == "NR" or len(exposure) < 16:
            raise ScreeningIntegrationError(
                f"{context_label}: training/calibration exposure is required"
            )
        actions = unicodedata.normalize(
            "NFKC", row["automated_actions"]
        ).strip()
        if actions == "NR" or not actions:
            raise ScreeningIntegrationError(
                f"{context_label}: automated_actions must be recorded"
            )
        if role_type == "hybrid" and actions.casefold() == "none":
            raise ScreeningIntegrationError(
                f"{context_label}: hybrid role must identify automated actions"
            )


def _validate_execution_registry_rows(
    rows: Sequence[Row],
    context: _ScreeningContext,
    adjudications: Sequence[Row],
) -> tuple[Row, ...]:
    result_hashes: dict[tuple[str, str], str] = {}
    for phase_snapshot in (context.calibration, context.main):
        for manifest_row in phase_snapshot.manifest:
            result_hashes[
                (phase_snapshot.phase, manifest_row["batch_id"])
            ] = manifest_row["result_file_sha256"]
    adjudication_hash = _sha256(_csv_bytes(ADJUDICATION_HEADER, adjudications))

    expected: dict[tuple[str, str], tuple[str, str]] = {}
    for rating in context.rows:
        task = (
            "calibration-screening"
            if rating["phase"] == "calibration"
            else "main-screening"
        )
        expected[(task, rating["assignment_id"])] = (
            rating["coder_id"],
            result_hashes[(rating["phase"], rating["batch_id"])],
        )
    for adjudication in adjudications:
        expected[("adjudication", adjudication["candidate_id"])] = (
            adjudication["adjudicator_id"],
            adjudication_hash,
        )

    by_work_item: dict[tuple[str, str], Row] = {}
    stable_by_role_task: defaultdict[
        tuple[str, str], set[tuple[str, ...]]
    ] = defaultdict(set)
    role_types: defaultdict[str, set[str]] = defaultdict(set)
    owners_by_identifier: dict[str, dict[str, tuple[str, str]]] = {
        "execution_id": {},
        "context_id": {},
    }
    valid_tasks = {
        "calibration-screening",
        "main-screening",
        "adjudication",
    }
    stable_fields = tuple(
        field
        for field in EXECUTION_REGISTER_HEADER
        if field != "work_item_id"
    )
    for row_number, source in enumerate(rows, start=2):
        row = dict(source)
        label = f"execution_registry.csv:{row_number}"
        for field in (
            "execution_id",
            "role_id",
            "context_id",
            "work_item_id",
        ):
            if not screening_results.is_valid_identifier(row[field]):
                raise ScreeningIntegrationError(
                    f"{label}: {field} is not a stable identifier"
                )
        if row["task"] not in valid_tasks:
            raise ScreeningIntegrationError(
                f"{label}: task is not a supported screening task"
            )
        _validate_execution_provenance(row, context_label=label)
        if _HASH_PATTERN.fullmatch(row["result_file_sha256"]) is None:
            raise ScreeningIntegrationError(
                f"{label}: result_file_sha256 must be a SHA-256 digest"
            )
        started = _call(
            screening_results.validate_iso_date,
            row["started_on"],
            field="started_on",
            context=label,
        )
        completed = _call(
            screening_results.validate_iso_date,
            row["completed_on"],
            field="completed_on",
            context=label,
        )
        if started > completed:
            raise ScreeningIntegrationError(
                f"{label}: started_on must not follow completed_on"
            )

        key = (row["task"], row["work_item_id"])
        if key in by_work_item:
            raise ScreeningIntegrationError(
                f"{label}: duplicate task/work_item_id {key!r}"
            )
        by_work_item[key] = row
        owner = (row["role_id"], row["task"])
        stable_by_role_task[owner].add(
            tuple(row[field] for field in stable_fields)
        )
        role_types[row["role_id"]].add(row["role_type"])
        for field in ("execution_id", "context_id"):
            previous = owners_by_identifier[field].get(row[field])
            if previous is not None and previous != owner:
                raise ScreeningIntegrationError(
                    f"{label}: {field} is shared by different role/task owners"
                )
            owners_by_identifier[field][row[field]] = owner

    actual = set(by_work_item)
    if actual != set(expected):
        raise ScreeningIntegrationError(
            "execution register coverage mismatch; "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    for key, (role_id, result_digest) in expected.items():
        row = by_work_item[key]
        if row["role_id"] != role_id:
            raise ScreeningIntegrationError(
                f"execution register role mismatch for {key!r}"
            )
        if row["result_file_sha256"] != result_digest:
            raise ScreeningIntegrationError(
                f"execution register result digest mismatch for {key!r}"
            )
    unstable = {
        owner: values
        for owner, values in stable_by_role_task.items()
        if len(values) != 1
    }
    if unstable:
        raise ScreeningIntegrationError(
            "execution register must use one stable execution, context, and "
            f"provenance record per role and task: {sorted(unstable)}"
        )
    inconsistent_role_types = {
        role_id: values
        for role_id, values in role_types.items()
        if len(values) != 1
    }
    if inconsistent_role_types:
        raise ScreeningIntegrationError(
            "execution register role types changed across tasks: "
            f"{sorted(inconsistent_role_types)}"
        )

    ratings = _ratings_by_candidate(context)
    for candidate_id, pair in ratings.items():
        task = (
            "calibration-screening"
            if pair[0]["phase"] == "calibration"
            else "main-screening"
        )
        reviewer_rows = [
            by_work_item[(task, rating["assignment_id"])]
            for rating in pair
        ]
        reviewer_contexts = {
            reviewer["context_id"] for reviewer in reviewer_rows
        }
        if len(reviewer_contexts) != 2:
            raise ScreeningIntegrationError(
                f"candidate_id={candidate_id!r}: reviewer execution "
                "contexts must be distinct"
            )
        human_identities = [
            reviewer["human_role"]
            for reviewer in reviewer_rows
            if reviewer["role_type"] in {"human", "hybrid"}
        ]
        if len(human_identities) != len(set(human_identities)):
            raise ScreeningIntegrationError(
                f"candidate_id={candidate_id!r}: human reviewer identities "
                "must be distinct"
            )

    for adjudication in adjudications:
        candidate_id = adjudication["candidate_id"]
        pair = ratings[candidate_id]
        reviewer_task = (
            "calibration-screening"
            if pair[0]["phase"] == "calibration"
            else "main-screening"
        )
        reviewer_rows = [
            by_work_item[(reviewer_task, rating["assignment_id"])]
            for rating in pair
        ]
        reviewer_contexts = {
            reviewer["context_id"] for reviewer in reviewer_rows
        }
        adjudicator_row = by_work_item[("adjudication", candidate_id)]
        if adjudicator_row["context_id"] in reviewer_contexts:
            raise ScreeningIntegrationError(
                f"candidate_id={candidate_id!r}: adjudicator context must "
                "be distinct from both reviewer contexts"
            )
        if adjudicator_row["role_type"] in {"human", "hybrid"}:
            reviewer_human_identities = {
                reviewer["human_role"]
                for reviewer in reviewer_rows
                if reviewer["role_type"] in {"human", "hybrid"}
            }
            if adjudicator_row["human_role"] in reviewer_human_identities:
                raise ScreeningIntegrationError(
                    f"candidate_id={candidate_id!r}: adjudicator human "
                    "identity must differ from both reviewer identities"
                )

    return tuple(sorted(by_work_item.values(), key=_execution_registry_sort_key))


def _canonical_execution_registry(
    payload: bytes,
    label: str,
    context: _ScreeningContext,
    adjudications: Sequence[Row],
) -> tuple[tuple[Row, ...], bytes]:
    parsed = _parse_csv(payload, label, EXECUTION_REGISTER_HEADER)
    rows = _validate_execution_registry_rows(parsed, context, adjudications)
    canonical = _csv_bytes(EXECUTION_REGISTER_HEADER, rows)
    if payload != canonical:
        raise ScreeningIntegrationError(
            f"{label}: execution register must be canonical UTF-8 CSV in "
            "task/work-item/role byte order"
        )
    return rows, canonical


def _adjudication_artifacts(
    rows: Sequence[Row],
    execution_registry: Sequence[Row],
    context: _ScreeningContext,
) -> tuple[dict[str, bytes], Row]:
    adjudication_payload = _csv_bytes(ADJUDICATION_HEADER, rows)
    execution_payload = _csv_bytes(
        EXECUTION_REGISTER_HEADER, execution_registry
    )
    adjudication_hash = _sha256(adjudication_payload)
    execution_hash = _sha256(execution_payload)
    binding = {
        "manifest_version": MANIFEST_VERSION,
        "coordinator_snapshot_sha256": context.coordinator_snapshot_sha256,
        "protocol_sha256": context.protocol_sha256,
        "calibration_result_snapshot_sha256": (
            context.calibration.snapshot_sha256
        ),
        "calibration_decision_snapshot_sha256": (
            context.calibration_decision.snapshot_sha256
        ),
        "main_result_snapshot_sha256": context.main.snapshot_sha256,
        "primary_snapshot_sha256": context.primary_snapshot_sha256,
        "adjudication_file_sha256": adjudication_hash,
        "execution_registry_sha256": execution_hash,
        "row_count": len(rows),
        "execution_row_count": len(execution_registry),
    }
    snapshot_hash = _canonical_sha256(binding)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "adjudication_snapshot_sha256": snapshot_hash,
        "coordinator_snapshot_sha256": context.coordinator_snapshot_sha256,
        "protocol_sha256": context.protocol_sha256,
        "calibration_result_snapshot_sha256": (
            context.calibration.snapshot_sha256
        ),
        "calibration_decision_snapshot_sha256": (
            context.calibration_decision.snapshot_sha256
        ),
        "main_result_snapshot_sha256": context.main.snapshot_sha256,
        "primary_snapshot_sha256": context.primary_snapshot_sha256,
        "adjudication_file_sha256": adjudication_hash,
        "execution_registry_sha256": execution_hash,
        "row_count": str(len(rows)),
        "execution_row_count": str(len(execution_registry)),
    }
    artifacts = {
        "adjudications.csv": adjudication_payload,
        "execution_registry.csv": execution_payload,
        "manifest.csv": _csv_bytes(ADJUDICATION_MANIFEST_HEADER, [manifest]),
    }
    artifacts["SHA256SUMS"] = _checksums(artifacts)
    return artifacts, manifest


def _publish_artifacts(
    output_dir: Path,
    artifacts: dict[str, bytes],
    *,
    post_publish_check: Callable[[], None] | None = None,
) -> None:
    _call(
        screening_batches.publish_snapshot,
        Path(output_dir),
        artifacts,
        post_publish_check=post_publish_check,
    )


def _capture_flat_exact(
    snapshot_dir: Path,
    expected_files: tuple[str, ...],
) -> tuple[dict[str, bytes], tuple[screening_results.FileFingerprint, ...]]:
    payloads, fingerprints = _call(
        screening_results.capture_flat_snapshot,
        Path(snapshot_dir),
        expected_files,
    )
    for fingerprint in fingerprints:
        _call(fingerprint.reattest)
    return payloads, fingerprints


def _validate_adjudication_snapshot_with_context(
    snapshot_dir: Path,
    context: _ScreeningContext,
    execution_register: screening_results.CapturedInput,
) -> AdjudicationSnapshot:
    snapshot_path = Path(snapshot_dir)
    context_paths = _context_protected_paths(context)
    context_fingerprints = _context_protected_fingerprints(context)
    _call(
        screening_results.reject_output_overlap,
        snapshot_path,
        (*context_paths, execution_register.fingerprint.path),
    )
    _reject_captured_input_aliases(
        (execution_register,),
        (*context_paths, snapshot_path),
        protected_fingerprints=context_fingerprints,
    )
    payloads, fingerprints = _capture_flat_exact(
        snapshot_path, _ADJUDICATION_FILES
    )
    _reject_captured_input_aliases(
        (execution_register,),
        (*context_paths, fingerprints[0].path.parent),
        protected_fingerprints=(
            *context_fingerprints,
            *fingerprints,
        ),
    )
    if payloads["SHA256SUMS"] != _checksums(
        {
            name: payload
            for name, payload in payloads.items()
            if name != "SHA256SUMS"
        }
    ):
        raise ScreeningIntegrationError(
            "adjudication snapshot checksum mismatch"
        )
    rows = _parse_csv(
        payloads["adjudications.csv"],
        "adjudications.csv",
        ADJUDICATION_HEADER,
    )
    validated = _validate_adjudication_rows(rows, context)
    adjudication_payload = _csv_bytes(ADJUDICATION_HEADER, validated)
    if payloads["adjudications.csv"] != adjudication_payload:
        raise ScreeningIntegrationError(
            "adjudications.csv is not in canonical candidate order"
        )
    registry, registry_payload = _canonical_execution_registry(
        payloads["execution_registry.csv"],
        "execution_registry.csv",
        context,
        validated,
    )
    if execution_register.payload != registry_payload:
        raise ScreeningIntegrationError(
            "sealed execution registry does not match the captured raw input"
        )
    expected_artifacts, expected_manifest = _adjudication_artifacts(
        validated, registry, context
    )
    if payloads != expected_artifacts:
        raise ScreeningIntegrationError(
            "adjudication snapshot does not match its canonical replay"
        )
    manifest_rows = _parse_csv(
        payloads["manifest.csv"],
        "adjudication manifest.csv",
        ADJUDICATION_MANIFEST_HEADER,
    )
    if len(manifest_rows) != 1 or manifest_rows[0] != expected_manifest:
        raise ScreeningIntegrationError(
            "adjudication manifest does not match sealed results"
        )
    return AdjudicationSnapshot(
        directory=fingerprints[0].path.parent,
        rows=validated,
        execution_registry=registry,
        snapshot_sha256=expected_manifest[
            "adjudication_snapshot_sha256"
        ],
        coordinator_snapshot_sha256=context.coordinator_snapshot_sha256,
        protocol_sha256=context.protocol_sha256,
        calibration_result_snapshot_sha256=(
            context.calibration.snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            context.calibration_decision.snapshot_sha256
        ),
        main_result_snapshot_sha256=context.main.snapshot_sha256,
        primary_snapshot_sha256=context.primary_snapshot_sha256,
        execution_registry_sha256=expected_manifest[
            "execution_registry_sha256"
        ],
        manifest=expected_manifest,
        fingerprints=fingerprints,
    )


def validate_adjudication_snapshot(
    snapshot_dir: Path,
    *,
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
    execution_register_path: Path,
) -> AdjudicationSnapshot:
    context = _load_context(
        coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
        main_reviewer_release_snapshot_dir,
        main_result_snapshot_dir,
    )
    execution_register = _call(
        screening_results.capture_input,
        Path(execution_register_path),
        "execution register",
    )
    snapshot = _validate_adjudication_snapshot_with_context(
        snapshot_dir, context, execution_register
    )
    _reattest_context(
        context,
        snapshot.fingerprints,
        (execution_register.fingerprint,),
    )
    return snapshot


def _reject_captured_input_aliases(
    captured: Sequence[screening_results.CapturedInput],
    protected: Sequence[Path],
    *,
    protected_fingerprints: Sequence[
        screening_results.FileFingerprint
    ] = (),
) -> None:
    protected_identities = {
        fingerprint.identity: fingerprint.path
        for fingerprint in protected_fingerprints
    }
    identities: dict[object, Path] = {}
    for item in captured:
        immutable = protected_identities.get(item.fingerprint.identity)
        if immutable is not None:
            raise ScreeningIntegrationError(
                f"{item.fingerprint.path}: input aliases an immutable "
                f"snapshot file {immutable}"
            )
        previous = identities.get(item.fingerprint.identity)
        if previous is not None:
            raise ScreeningIntegrationError(
                f"{item.fingerprint.path}: input aliases {previous}"
            )
        identities[item.fingerprint.identity] = item.fingerprint.path
        for path in protected:
            if _call(
                screening_results.paths_overlap,
                item.fingerprint.path,
                Path(path),
            ):
                raise ScreeningIntegrationError(
                    f"{item.fingerprint.path}: mutable input overlaps "
                    "an immutable snapshot"
                )


def seal_adjudication_results(
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
    adjudication_result_path: Path,
    execution_register_path: Path,
    output_dir: Path,
) -> None:
    context = _load_context(
        coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
        main_reviewer_release_snapshot_dir,
        main_result_snapshot_dir,
    )
    protected = _context_protected_paths(context)
    protected_fingerprints = _context_protected_fingerprints(context)
    _call(
        screening_results.reject_output_overlap,
        Path(output_dir),
        (*protected, Path(adjudication_result_path), Path(execution_register_path)),
    )
    adjudication_input = _call(
        screening_results.capture_input,
        Path(adjudication_result_path),
        "adjudication result",
    )
    execution_register = _call(
        screening_results.capture_input,
        Path(execution_register_path),
        "execution register",
    )
    _reject_captured_input_aliases(
        (adjudication_input, execution_register),
        protected,
        protected_fingerprints=protected_fingerprints,
    )
    rows = _parse_csv(
        adjudication_input.payload,
        str(adjudication_input.fingerprint.path),
        ADJUDICATION_HEADER,
    )
    validated = _validate_adjudication_rows(rows, context)
    canonical_adjudications = _csv_bytes(ADJUDICATION_HEADER, validated)
    if adjudication_input.payload != canonical_adjudications:
        raise ScreeningIntegrationError(
            "adjudication result input must be canonical UTF-8 CSV in "
            "candidate byte order"
        )
    registry, _ = _canonical_execution_registry(
        execution_register.payload,
        str(execution_register.fingerprint.path),
        context,
        validated,
    )
    artifacts, _ = _adjudication_artifacts(
        validated, registry, context
    )

    raw_fingerprints = (
        (adjudication_input.fingerprint,),
        (execution_register.fingerprint,),
    )

    def reattest_inputs() -> None:
        _reattest_context(context, *raw_fingerprints)

    def post_publish_check() -> None:
        reattest_inputs()
        published = _validate_adjudication_snapshot_with_context(
            Path(output_dir), context, execution_register
        )
        _reject_captured_input_aliases(
            (adjudication_input, execution_register),
            (*protected, published.directory),
            protected_fingerprints=(
                *protected_fingerprints,
                *published.fingerprints,
            ),
        )
        _reattest_context(
            context,
            published.fingerprints,
            *raw_fingerprints,
        )

    reattest_inputs()
    _publish_artifacts(
        Path(output_dir),
        artifacts,
        post_publish_check=post_publish_check,
    )


def _direct_final(pair: tuple[Row, Row]) -> tuple[str, str, str]:
    first, second = pair
    if (
        first["screening_status"] != second["screening_status"]
        or first["criterion"] != second["criterion"]
    ):
        raise ScreeningIntegrationError(
            f"candidate_id={first['candidate_id']!r}: disagreement lacks adjudication"
        )
    status = first["screening_status"]
    criterion = first["criterion"]
    if status != "excluded":
        return status, criterion, "NR"
    if (
        _normalize_exclusion_reason(first["exclusion_reason"])
        != _normalize_exclusion_reason(second["exclusion_reason"])
    ):
        raise ScreeningIntegrationError(
            f"candidate_id={first['candidate_id']!r}: exclusion reasons "
            "disagree without adjudication"
        )
    reason = min(
        (first["exclusion_reason"], second["exclusion_reason"]),
        key=_utf8,
    )
    return status, criterion, reason


def _same_adjudication_snapshot(
    captured: AdjudicationSnapshot,
    authoritative: AdjudicationSnapshot,
) -> bool:
    return captured == authoritative


def _capture_integration_inputs(
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
    adjudication_result_snapshot_dir: Path,
    execution_register_path: Path,
    citation_key_ledger_path: Path,
) -> _CapturedIntegrationInputs:
    context = _load_context(
        coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
        main_reviewer_release_snapshot_dir,
        main_result_snapshot_dir,
    )
    execution_register = _call(
        screening_results.capture_input,
        Path(execution_register_path),
        "execution register",
    )
    citation_key_ledger = _call(
        screening_results.capture_input,
        Path(citation_key_ledger_path),
        "citation key ledger",
    )
    context_paths = _context_protected_paths(context)
    context_fingerprints = _context_protected_fingerprints(context)
    adjudication_path = Path(adjudication_result_snapshot_dir)
    _reject_captured_input_aliases(
        (execution_register, citation_key_ledger),
        (*context_paths, adjudication_path),
        protected_fingerprints=context_fingerprints,
    )
    adjudication = _validate_adjudication_snapshot_with_context(
        adjudication_path,
        context,
        execution_register,
    )
    _call(
        screening_results.reject_output_overlap,
        adjudication.directory,
        (
            *context_paths,
            execution_register.fingerprint.path,
            citation_key_ledger.fingerprint.path,
        ),
    )
    _reject_captured_input_aliases(
        (execution_register, citation_key_ledger),
        (*context_paths, adjudication.directory),
        protected_fingerprints=(
            *context_fingerprints,
            *adjudication.fingerprints,
        ),
    )
    captured = _CapturedIntegrationInputs(
        context=context,
        adjudication=adjudication,
        execution_register=execution_register,
        citation_key_ledger=citation_key_ledger,
    )
    _reattest_integration_inputs(captured)
    return captured


def _reattest_integration_inputs(
    captured: _CapturedIntegrationInputs,
    *fingerprint_groups: Sequence[screening_results.FileFingerprint],
) -> None:
    authoritative = _validate_adjudication_snapshot_with_context(
        captured.adjudication.directory,
        captured.context,
        captured.execution_register,
    )
    if not _same_adjudication_snapshot(captured.adjudication, authoritative):
        raise ScreeningIntegrationError(
            "adjudication snapshot changed after initial capture"
        )
    _reattest_context(
        captured.context,
        captured.adjudication.fingerprints,
        (captured.execution_register.fingerprint,),
        (captured.citation_key_ledger.fingerprint,),
        *fingerprint_groups,
    )


def _validate_citation_key_ledger(
    captured: screening_results.CapturedInput,
    context: _ScreeningContext,
) -> tuple[tuple[Row, ...], dict[str, str]]:
    rows = _parse_csv(
        captured.payload,
        str(captured.fingerprint.path),
        screening_batches.CITATION_KEY_HEADER,
    )
    canonical = _csv_bytes(screening_batches.CITATION_KEY_HEADER, rows)
    if captured.payload != canonical:
        raise ScreeningIntegrationError(
            "citation key ledger must be canonical UTF-8 CSV"
        )
    baseline = _parse_csv(
        context.coordinator.payloads["citation_keys.csv"],
        "coordinator citation_keys.csv",
        screening_batches.CITATION_KEY_HEADER,
    )
    if rows[: len(baseline)] != baseline:
        raise ScreeningIntegrationError(
            "citation key ledger must preserve the coordinator ledger as an "
            "exact append-only prefix"
        )

    candidates = {row["candidate_id"]: row for row in context.candidates}
    by_candidate: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"]
        cite_key = row["cite_key"]
        if candidate_id not in candidates:
            raise ScreeningIntegrationError(
                f"citation_keys.csv:{row_number}: unknown candidate_id "
                f"{candidate_id!r}"
            )
        if _CITE_KEY_PATTERN.fullmatch(cite_key) is None:
            raise ScreeningIntegrationError(
                f"citation_keys.csv:{row_number}: invalid cite_key {cite_key!r}"
            )
        if candidate_id in by_candidate:
            raise ScreeningIntegrationError(
                f"citation_keys.csv:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}"
            )
        folded = cite_key.casefold()
        if folded in by_key:
            raise ScreeningIntegrationError(
                f"citation_keys.csv:{row_number}: duplicate cite_key "
                f"{cite_key!r}"
            )
        by_candidate[candidate_id] = cite_key
        by_key[folded] = candidate_id

    extras = rows[len(baseline) :]
    extra_ids = [row["candidate_id"] for row in extras]
    if extra_ids != sorted(extra_ids, key=_utf8):
        raise ScreeningIntegrationError(
            "new citation key assignments must be appended in candidate byte order"
        )
    for candidate_id in extra_ids:
        if candidates[candidate_id]["cite_key"]:
            raise ScreeningIntegrationError(
                f"candidate_id={candidate_id!r}: existing citation key cannot "
                "be reissued"
            )
    return tuple(rows), by_candidate


def _integrate_captured(
    captured: _CapturedIntegrationInputs,
) -> ScreeningIntegrationResult:
    context = captured.context
    adjudication = captured.adjudication
    adjudications = {
        row["candidate_id"]: row for row in adjudication.rows
    }
    ratings = _ratings_by_candidate(context)
    unresolved = _unresolved_screening_conflicts(context)
    citation_keys, citation_key_by_candidate = _validate_citation_key_ledger(
        captured.citation_key_ledger,
        context,
    )

    final: dict[str, tuple[str, str, str]] = {}
    integrated_candidates: list[Row] = []
    for original in context.candidates:
        candidate_id = original["candidate_id"]
        decision = adjudications.get(candidate_id)
        if decision is None:
            status, criterion, reason = _direct_final(
                ratings[candidate_id]
            )
        else:
            status = decision["screening_status"]
            criterion = decision["criterion"]
            reason = decision["exclusion_reason"]
        active_key = citation_key_by_candidate.get(candidate_id, "")
        if status in {"included", "boundary"} and not active_key:
            raise ScreeningIntegrationError(
                f"candidate_id={candidate_id!r}: included or boundary candidate "
                "requires an append-only audited citation key assignment"
            )
        candidate = dict(original)
        candidate["cite_key"] = (
            active_key if status in {"included", "boundary"} else ""
        )
        candidate["screening_status"] = status
        candidate["exclusion_reason"] = "" if reason == "NR" else reason
        integrated_candidates.append(candidate)
        final[candidate_id] = (status, criterion, reason)

    integrated_conflicts = [dict(row) for row in context.conflicts]
    conflict_indexes = {
        row["conflict_id"]: index
        for index, row in enumerate(integrated_conflicts)
    }
    for decision in adjudication.rows:
        conflict_ids = _resolved_conflict_ids(
            decision,
            context,
            unresolved,
            context_label=(
                f"adjudication candidate_id={decision['candidate_id']!r}"
            ),
        )
        for conflict_id in conflict_ids:
            target = integrated_conflicts[conflict_indexes[conflict_id]]
            target["resolution"] = decision["screening_status"]
            target["resolver"] = decision["adjudicator_id"]
            target["resolution_evidence"] = decision[
                "resolution_evidence"
            ]

    decisions: list[Row] = []
    for rating in context.rows:
        candidate_id = rating["candidate_id"]
        status, criterion, reason = final[candidate_id]
        adjudicated = adjudications.get(candidate_id)
        row = {
            **{field: rating[field] for field in SCREENING_RESULT_HEADER},
            "adjudicated": "yes" if adjudicated else "no",
            "final_screening_status": status,
            "final_criterion": criterion,
            "final_exclusion_reason": reason,
        }
        for field in ADJUDICATION_HEADER:
            row[f"adjudication_{field}"] = (
                adjudicated[field] if adjudicated else "NR"
            )
        if tuple(row) != SCREENING_DECISIONS_HEADER:
            raise ScreeningIntegrationError(
                "internal screening decision schema mismatch"
            )
        decisions.append(row)

    agreement = tuple(
        _call(
            screening_agreement.build_agreement_report,
            context.coordinator,
            context.calibration_release_dir,
            context.calibration,
            context.calibration_decision,
            context.main_release_dir,
            context.main,
        )
    )
    return ScreeningIntegrationResult(
        candidates=tuple(integrated_candidates),
        citation_keys=citation_keys,
        conflicts=tuple(integrated_conflicts),
        screening_decisions=tuple(decisions),
        screening_agreement=agreement,
        coordinator_snapshot_sha256=context.coordinator_snapshot_sha256,
        protocol_sha256=context.protocol_sha256,
        calibration_result_snapshot_sha256=(
            context.calibration.snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            context.calibration_decision.snapshot_sha256
        ),
        main_result_snapshot_sha256=context.main.snapshot_sha256,
        primary_snapshot_sha256=context.primary_snapshot_sha256,
        adjudication_snapshot_sha256=adjudication.snapshot_sha256,
        execution_registry_sha256=(
            adjudication.execution_registry_sha256
        ),
        citation_key_ledger_sha256=(
            captured.citation_key_ledger.fingerprint.sha256
        ),
    )


def integrate_screening(
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
    adjudication_result_snapshot_dir: Path,
    execution_register_path: Path,
    citation_key_ledger_path: Path,
) -> ScreeningIntegrationResult:
    captured = _capture_integration_inputs(
        coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
        main_reviewer_release_snapshot_dir,
        main_result_snapshot_dir,
        adjudication_result_snapshot_dir,
        execution_register_path,
        citation_key_ledger_path,
    )
    result = _integrate_captured(captured)
    _reattest_integration_inputs(captured)
    return result


def _author_decision_bindings(
    result: ScreeningIntegrationResult,
) -> dict[str, dict[str, str]]:
    decisions_by_candidate: defaultdict[str, list[Row]] = defaultdict(list)
    for row in result.screening_decisions:
        decisions_by_candidate[row["candidate_id"]].append(row)
    conflicts_by_candidate: defaultdict[str, list[Row]] = defaultdict(list)
    for row in result.conflicts:
        if row["record_type"] == "candidate":
            conflicts_by_candidate[row["record_key"]].append(row)

    bindings: dict[str, dict[str, str]] = {}
    for candidate in result.candidates:
        candidate_id = candidate["candidate_id"]
        decisions = sorted(
            decisions_by_candidate[candidate_id],
            key=lambda row: _utf8(row["assignment_id"]),
        )
        if len(decisions) != 2:
            raise ScreeningIntegrationError(
                f"candidate_id={candidate_id!r}: author binding requires "
                "exactly two final decision rows"
            )
        if decisions[0]["adjudicated"] == "yes":
            versions = [decisions[0]["adjudication_evidence_version"]]
            locators = [decisions[0]["adjudication_screening_locator"]]
        else:
            versions = sorted(
                {row["evidence_version"] for row in decisions},
                key=_utf8,
            )
            locators = sorted(
                {row["screening_locator"] for row in decisions},
                key=_utf8,
            )
        relevant_conflicts = sorted(
            conflicts_by_candidate.get(candidate_id, ()),
            key=lambda row: _utf8(row["conflict_id"]),
        )
        bindings[candidate_id] = {
            "decision_sha256": _canonical_sha256(
                {
                    "candidate": candidate,
                    "conflicts": relevant_conflicts,
                    "screening_decisions": decisions,
                }
            ),
            "evidence_versions_sha256": _canonical_sha256(versions),
            "deciding_locators_sha256": _canonical_sha256(locators),
        }
    return bindings


def _validate_author_verification(
    captured: screening_results.CapturedInput,
    result: ScreeningIntegrationResult,
) -> tuple[Row, ...]:
    rows = _parse_csv(
        captured.payload,
        str(captured.fingerprint.path),
        AUTHOR_VERIFICATION_HEADER,
    )
    canonical = _csv_bytes(AUTHOR_VERIFICATION_HEADER, rows)
    if captured.payload != canonical:
        raise ScreeningIntegrationError(
            "author verification must be canonical UTF-8 CSV"
        )
    bindings = _author_decision_bindings(result)
    expected_ids = sorted(bindings, key=_utf8)
    actual_ids = [row["candidate_id"] for row in rows]
    if actual_ids != expected_ids:
        raise ScreeningIntegrationError(
            "author verification must contain exactly one row for each of "
            "the 202 candidates in candidate byte order"
        )

    for row_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"]
        expected = {
            "primary_snapshot_sha256": result.primary_snapshot_sha256,
            "adjudication_snapshot_sha256": (
                result.adjudication_snapshot_sha256
            ),
            **bindings[candidate_id],
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ScreeningIntegrationError(
                    f"author_verification.csv:{row_number}: {field} does not "
                    "match the final decision binding"
                )
        if not screening_results.is_valid_identifier(row["verified_by"]):
            raise ScreeningIntegrationError(
                f"author_verification.csv:{row_number}: verified_by must be "
                "a stable author identifier"
            )
        if row["verified_role"] != "accountable-author":
            raise ScreeningIntegrationError(
                f"author_verification.csv:{row_number}: verified_role must be "
                "'accountable-author'"
            )
        _call(
            screening_results.validate_iso_date,
            row["verified_on"],
            field="verified_on",
            context=f"author_verification.csv:{row_number}",
        )
        if row["verification_status"] != "verified":
            raise ScreeningIntegrationError(
                f"author_verification.csv:{row_number}: verification_status "
                "must be 'verified'"
            )
        evidence = unicodedata.normalize(
            "NFKC", row["verification_evidence"]
        ).strip()
        if (
            evidence == "NR"
            or len(evidence) < 48
            or candidate_id.casefold() not in evidence.casefold()
            or len(_alphabetic_words(evidence)) < 8
        ):
            raise ScreeningIntegrationError(
                f"author_verification.csv:{row_number}: verification_evidence "
                "must be a substantive candidate-specific sign-off"
            )
    return tuple(rows)


def _projection_artifacts(
    result: ScreeningIntegrationResult,
    author_verification: Sequence[Row],
    author_verification_sha256: str,
) -> tuple[dict[str, bytes], Row]:
    data_artifacts = {
        "candidates.csv": _csv_bytes(
            screening_batches.CANDIDATE_HEADER, result.candidates
        ),
        "citation_keys.csv": _csv_bytes(
            screening_batches.CITATION_KEY_HEADER, result.citation_keys
        ),
        "conflicts.csv": _csv_bytes(
            screening_batches.CONFLICT_HEADER, result.conflicts
        ),
        "screening_decisions.csv": _csv_bytes(
            SCREENING_DECISIONS_HEADER, result.screening_decisions
        ),
        "screening_agreement.csv": _call(
            screening_agreement.render_agreement_csv,
            result.screening_agreement,
        ),
        "author_verification.csv": _csv_bytes(
            AUTHOR_VERIFICATION_HEADER, author_verification
        ),
    }
    if (
        _sha256(data_artifacts["author_verification.csv"])
        != author_verification_sha256
    ):
        raise ScreeningIntegrationError(
            "author verification digest changed during canonical projection"
        )
    binding = {
        "manifest_version": MANIFEST_VERSION,
        "coordinator_snapshot_sha256": (
            result.coordinator_snapshot_sha256
        ),
        "protocol_sha256": result.protocol_sha256,
        "calibration_result_snapshot_sha256": (
            result.calibration_result_snapshot_sha256
        ),
        "calibration_decision_snapshot_sha256": (
            result.calibration_decision_snapshot_sha256
        ),
        "main_result_snapshot_sha256": (
            result.main_result_snapshot_sha256
        ),
        "primary_snapshot_sha256": result.primary_snapshot_sha256,
        "adjudication_snapshot_sha256": (
            result.adjudication_snapshot_sha256
        ),
        "execution_registry_sha256": (
            result.execution_registry_sha256
        ),
        "citation_key_ledger_sha256": (
            result.citation_key_ledger_sha256
        ),
        "author_verification_sha256": author_verification_sha256,
        "outputs": {
            name: _sha256(payload)
            for name, payload in sorted(data_artifacts.items())
        },
        "candidate_count": len(result.candidates),
        "decision_row_count": len(result.screening_decisions),
        "agreement_row_count": len(result.screening_agreement),
    }
    snapshot_hash = _canonical_sha256(binding)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "projection_snapshot_sha256": snapshot_hash,
        "coordinator_snapshot_sha256": (
            result.coordinator_snapshot_sha256
        ),
        "protocol_sha256": result.protocol_sha256,
        "calibration_result_snapshot_sha256": (
            result.calibration_result_snapshot_sha256
        ),
        "calibration_decision_snapshot_sha256": (
            result.calibration_decision_snapshot_sha256
        ),
        "main_result_snapshot_sha256": (
            result.main_result_snapshot_sha256
        ),
        "primary_snapshot_sha256": result.primary_snapshot_sha256,
        "adjudication_snapshot_sha256": (
            result.adjudication_snapshot_sha256
        ),
        "execution_registry_sha256": (
            result.execution_registry_sha256
        ),
        "citation_key_ledger_sha256": (
            result.citation_key_ledger_sha256
        ),
        "author_verification_sha256": author_verification_sha256,
        "candidates_sha256": _sha256(data_artifacts["candidates.csv"]),
        "citation_keys_sha256": _sha256(
            data_artifacts["citation_keys.csv"]
        ),
        "conflicts_sha256": _sha256(data_artifacts["conflicts.csv"]),
        "screening_decisions_sha256": _sha256(
            data_artifacts["screening_decisions.csv"]
        ),
        "screening_agreement_sha256": _sha256(
            data_artifacts["screening_agreement.csv"]
        ),
        "candidate_count": str(len(result.candidates)),
        "decision_row_count": str(len(result.screening_decisions)),
        "agreement_row_count": str(len(result.screening_agreement)),
    }
    artifacts = dict(data_artifacts)
    artifacts["manifest.csv"] = _csv_bytes(
        PROJECTION_MANIFEST_HEADER, [manifest]
    )
    artifacts["SHA256SUMS"] = _checksums(artifacts)
    return artifacts, manifest


def _validate_projection_with_inputs(
    snapshot_dir: Path,
    captured: _CapturedIntegrationInputs,
    author_verification: screening_results.CapturedInput,
) -> ScreeningProjectionSnapshot:
    snapshot_path = Path(snapshot_dir)
    context_paths = _context_protected_paths(captured.context)
    immutable_paths = (*context_paths, captured.adjudication.directory)
    immutable_fingerprints = (
        *_context_protected_fingerprints(captured.context),
        *captured.adjudication.fingerprints,
    )
    mutable_inputs = (
        captured.execution_register,
        captured.citation_key_ledger,
        author_verification,
    )
    _call(
        screening_results.reject_output_overlap,
        snapshot_path,
        (
            *immutable_paths,
            *(item.fingerprint.path for item in mutable_inputs),
        ),
    )
    _reject_captured_input_aliases(
        mutable_inputs,
        (*immutable_paths, snapshot_path),
        protected_fingerprints=immutable_fingerprints,
    )
    payloads, fingerprints = _capture_flat_exact(
        snapshot_path, _PROJECTION_FILES
    )
    _reject_captured_input_aliases(
        mutable_inputs,
        (*immutable_paths, fingerprints[0].path.parent),
        protected_fingerprints=(
            *immutable_fingerprints,
            *fingerprints,
        ),
    )
    if payloads["SHA256SUMS"] != _checksums(
        {
            name: payload
            for name, payload in payloads.items()
            if name != "SHA256SUMS"
        }
    ):
        raise ScreeningIntegrationError(
            "screening projection checksum mismatch"
        )
    result = _integrate_captured(captured)
    author_rows = _validate_author_verification(
        author_verification,
        result,
    )
    expected, manifest = _projection_artifacts(
        result,
        author_rows,
        author_verification.fingerprint.sha256,
    )
    if payloads != expected:
        raise ScreeningIntegrationError(
            "screening projection does not match its canonical replay"
        )
    manifest_rows = _parse_csv(
        payloads["manifest.csv"],
        "screening projection manifest.csv",
        PROJECTION_MANIFEST_HEADER,
    )
    if len(manifest_rows) != 1 or manifest_rows[0] != manifest:
        raise ScreeningIntegrationError(
            "screening projection manifest is invalid"
        )
    snapshot = ScreeningProjectionSnapshot(
        directory=fingerprints[0].path.parent,
        snapshot_sha256=manifest["projection_snapshot_sha256"],
        coordinator_snapshot_sha256=(
            result.coordinator_snapshot_sha256
        ),
        protocol_sha256=result.protocol_sha256,
        calibration_result_snapshot_sha256=(
            result.calibration_result_snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            result.calibration_decision_snapshot_sha256
        ),
        main_result_snapshot_sha256=(
            result.main_result_snapshot_sha256
        ),
        primary_snapshot_sha256=result.primary_snapshot_sha256,
        adjudication_snapshot_sha256=(
            result.adjudication_snapshot_sha256
        ),
        execution_registry_sha256=(
            result.execution_registry_sha256
        ),
        citation_key_ledger_sha256=(
            result.citation_key_ledger_sha256
        ),
        author_verification_sha256=(
            author_verification.fingerprint.sha256
        ),
        candidate_count=len(result.candidates),
        decision_row_count=len(result.screening_decisions),
        agreement_row_count=len(result.screening_agreement),
        manifest=manifest,
        fingerprints=fingerprints,
    )
    _reattest_integration_inputs(
        captured,
        snapshot.fingerprints,
        (author_verification.fingerprint,),
    )
    return snapshot


def validate_screening_projection(
    snapshot_dir: Path,
    *,
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
    adjudication_result_snapshot_dir: Path,
    execution_register_path: Path,
    citation_key_ledger_path: Path,
    author_verification_path: Path,
) -> ScreeningProjectionSnapshot:
    captured = _capture_integration_inputs(
        coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
        main_reviewer_release_snapshot_dir,
        main_result_snapshot_dir,
        adjudication_result_snapshot_dir,
        execution_register_path,
        citation_key_ledger_path,
    )
    author_verification = _call(
        screening_results.capture_input,
        Path(author_verification_path),
        "author verification",
    )
    _reject_captured_input_aliases(
        (
            captured.execution_register,
            captured.citation_key_ledger,
            author_verification,
        ),
        (
            *_context_protected_paths(captured.context),
            captured.adjudication.directory,
        ),
        protected_fingerprints=(
            *_context_protected_fingerprints(captured.context),
            *captured.adjudication.fingerprints,
        ),
    )
    return _validate_projection_with_inputs(
        snapshot_dir,
        captured,
        author_verification,
    )


def seal_screening_projection(
    coordinator_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_decision_snapshot_dir: Path,
    main_reviewer_release_snapshot_dir: Path,
    main_result_snapshot_dir: Path,
    adjudication_result_snapshot_dir: Path,
    execution_register_path: Path,
    citation_key_ledger_path: Path,
    author_verification_path: Path,
    output_dir: Path,
) -> None:
    captured = _capture_integration_inputs(
        coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir,
        calibration_result_snapshot_dir,
        calibration_decision_snapshot_dir,
        main_reviewer_release_snapshot_dir,
        main_result_snapshot_dir,
        adjudication_result_snapshot_dir,
        execution_register_path,
        citation_key_ledger_path,
    )
    author_verification = _call(
        screening_results.capture_input,
        Path(author_verification_path),
        "author verification",
    )
    mutable_inputs = (
        captured.execution_register,
        captured.citation_key_ledger,
        author_verification,
    )
    immutable_paths = (
        *_context_protected_paths(captured.context),
        captured.adjudication.directory,
    )
    immutable_fingerprints = (
        *_context_protected_fingerprints(captured.context),
        *captured.adjudication.fingerprints,
    )
    _reject_captured_input_aliases(
        mutable_inputs,
        immutable_paths,
        protected_fingerprints=immutable_fingerprints,
    )
    _call(
        screening_results.reject_output_overlap,
        Path(output_dir),
        (
            *immutable_paths,
            *(item.fingerprint.path for item in mutable_inputs),
        ),
    )
    result = _integrate_captured(captured)
    author_rows = _validate_author_verification(
        author_verification,
        result,
    )
    _reattest_integration_inputs(
        captured,
        (author_verification.fingerprint,),
    )
    artifacts, _ = _projection_artifacts(
        result,
        author_rows,
        author_verification.fingerprint.sha256,
    )

    def post_publish_check() -> None:
        _reattest_integration_inputs(
            captured,
            (author_verification.fingerprint,),
        )
        published = _validate_projection_with_inputs(
            Path(output_dir),
            captured,
            author_verification,
        )
        _reattest_integration_inputs(
            captured,
            published.fingerprints,
            (author_verification.fingerprint,),
        )

    _publish_artifacts(
        Path(output_dir),
        artifacts,
        post_publish_check=post_publish_check,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal adjudications and immutable screening projections from "
            "validated coordinator and phase snapshots."
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--seal-adjudication", action="store_true")
    modes.add_argument("--seal-projection", action="store_true")
    parser.add_argument("--coordinator-snapshot", type=Path)
    parser.add_argument("--calibration-reviewer-release", type=Path)
    parser.add_argument("--calibration-result-snapshot", type=Path)
    parser.add_argument("--calibration-decision-snapshot", type=Path)
    parser.add_argument("--main-reviewer-release", type=Path)
    parser.add_argument("--main-result-snapshot", type=Path)
    parser.add_argument("--adjudication-result", type=Path)
    parser.add_argument("--adjudication-result-snapshot", type=Path)
    parser.add_argument("--execution-register", type=Path)
    parser.add_argument("--citation-key-ledger", type=Path)
    parser.add_argument("--author-verification", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _require(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    names: Sequence[str],
) -> None:
    missing = [
        "--" + name.replace("_", "-")
        for name in names
        if getattr(arguments, name) is None
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    common = (
        "coordinator_snapshot",
        "calibration_reviewer_release",
        "calibration_result_snapshot",
        "calibration_decision_snapshot",
        "main_reviewer_release",
        "main_result_snapshot",
        "execution_register",
        "output_dir",
    )
    if arguments.seal_adjudication:
        _require(parser, arguments, (*common, "adjudication_result"))
        if (
            arguments.adjudication_result_snapshot is not None
            or arguments.citation_key_ledger is not None
            or arguments.author_verification is not None
        ):
            parser.error(
                "--seal-adjudication does not accept projection inputs"
            )
        seal_adjudication_results(
            arguments.coordinator_snapshot,
            arguments.calibration_reviewer_release,
            arguments.calibration_result_snapshot,
            arguments.calibration_decision_snapshot,
            arguments.main_reviewer_release,
            arguments.main_result_snapshot,
            arguments.adjudication_result,
            arguments.execution_register,
            arguments.output_dir,
        )
        return 0

    _require(
        parser,
        arguments,
        (
            *common,
            "adjudication_result_snapshot",
            "citation_key_ledger",
            "author_verification",
        ),
    )
    if arguments.adjudication_result is not None:
        parser.error(
            "--seal-projection does not accept --adjudication-result"
        )
    seal_screening_projection(
        arguments.coordinator_snapshot,
        arguments.calibration_reviewer_release,
        arguments.calibration_result_snapshot,
        arguments.calibration_decision_snapshot,
        arguments.main_reviewer_release,
        arguments.main_result_snapshot,
        arguments.adjudication_result_snapshot,
        arguments.execution_register,
        arguments.citation_key_ledger,
        arguments.author_verification,
        arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
