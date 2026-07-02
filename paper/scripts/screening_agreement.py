"""Deterministic pre-adjudication screening agreement reports.

The public API requires a coordinator snapshot trust anchor and validated
phase-result snapshots. It revalidates both phases against that coordinator,
derives all provenance, and always performs 10,000 candidate-cluster bootstrap
replicates.

For a scope, replicate r, and draw j, the bootstrap index is the first eight
bytes of SHA-256 over screening-bootstrap-v1, NUL, the combined-primary hash,
NUL, scope, NUL, r, NUL, and j. It is interpreted as an unsigned big-endian
integer and reduced modulo the scope candidate count. Candidate order is
UTF-8 candidate_id byte order.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from . import screening_results
else:
    import screening_results


class ScreeningAgreementError(ValueError):
    """A trusted snapshot binding or agreement-report input is invalid."""


RESULT_HEADER = screening_results.RESULT_HEADER
REPORT_VERSION = "1"
BOOTSTRAP_ALGORITHM = "screening-bootstrap-v1"
PRODUCTION_BOOTSTRAP_REPLICATES = 10_000
STATUS_CATEGORIES = ("included", "boundary", "excluded")
INCLUSION_CRITERIA = tuple(screening_results.INCLUSION_CRITERIA)
EXCLUSION_CRITERIA = (
    "exclude-fixed-racing-line",
    "exclude-appearance-dynamics",
    "exclude-traffic-only",
    "exclude-insufficient-detail",
    "exclude-out-of-scope",
)
if set(EXCLUSION_CRITERIA) != set(screening_results.EXCLUSION_CRITERIA):
    raise RuntimeError("screening_results exclusion vocabulary changed")
CRITERION_CATEGORIES = (
    *INCLUSION_CRITERIA,
    "include-relevant",
    "boundary",
    *EXCLUSION_CRITERIA,
)
PROVENANCE_FIELDS = (
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "main_result_snapshot_sha256",
    "primary_result_snapshot_sha256",
)
BOOTSTRAP_METRICS = (
    "overall_exact_status_agreement",
    "exact_criterion_agreement",
    "krippendorff_alpha_nominal",
    "gwet_ac1_nominal",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _criterion_token(criterion: str) -> str:
    return criterion.replace("-", "_")


def _criterion_disagreement_field(left: str, right: str) -> str:
    if (
        left not in CRITERION_CATEGORIES
        or right not in CRITERION_CATEGORIES
        or left == right
    ):
        raise ScreeningAgreementError(
            "criterion disagreement fields require two different controlled criteria"
        )
    return (
        f"criterion_a_{_criterion_token(left)}"
        f"_criterion_b_{_criterion_token(right)}"
    )


CRITERION_DISAGREEMENT_FIELDS = tuple(
    _criterion_disagreement_field(left, right)
    for left in CRITERION_CATEGORIES
    for right in CRITERION_CATEGORIES
    if left != right
)
AGREEMENT_REPORT_HEADER = (
    "report_version",
    "scope",
    *PROVENANCE_FIELDS,
    "bootstrap_algorithm",
    "candidate_count",
    "rating_count",
    *(
        f"rating_a_{left}_rating_b_{right}"
        for left in STATUS_CATEGORIES
        for right in STATUS_CATEGORIES
    ),
    "overall_exact_status_agreement_count",
    "overall_exact_status_agreement_denominator",
    "overall_exact_status_agreement_rate",
    "exact_criterion_agreement_count",
    "exact_criterion_agreement_denominator",
    "exact_criterion_agreement_rate",
    *CRITERION_DISAGREEMENT_FIELDS,
    *(
        field
        for category in STATUS_CATEGORIES
        for field in (
            f"{category}_positive_numerator",
            f"{category}_positive_denominator",
            f"{category}_positive_agreement",
            f"{category}_negative_numerator",
            f"{category}_negative_denominator",
            f"{category}_negative_agreement",
        )
    ),
    "krippendorff_alpha_nominal_numerator",
    "krippendorff_alpha_nominal_denominator",
    "krippendorff_alpha_nominal",
    "gwet_ac1_nominal_numerator",
    "gwet_ac1_nominal_denominator",
    "gwet_ac1_nominal",
    *(
        field
        for metric in BOOTSTRAP_METRICS
        for field in (
            f"{metric}_bootstrap_replicates",
            f"{metric}_bootstrap_valid_replicates",
            f"{metric}_bootstrap_ci95_lower",
            f"{metric}_bootstrap_ci95_upper",
        )
    ),
)
AGREEMENT_HEADER = AGREEMENT_REPORT_HEADER


@dataclass(frozen=True)
class RatingPair:
    candidate_id: str
    status_a: str
    status_b: str
    criterion_a: str
    criterion_b: str


@dataclass(frozen=True)
class PointEstimates:
    matrix: dict[tuple[str, str], int]
    criterion_disagreements: dict[tuple[str, str], int]
    exact_status_count: int
    exact_status_rate: Fraction
    exact_criterion_count: int
    exact_criterion_rate: Fraction
    positive_components: dict[str, tuple[int, int]]
    negative_components: dict[str, tuple[int, int]]
    positive_agreement: dict[str, Fraction | None]
    negative_agreement: dict[str, Fraction | None]
    krippendorff_alpha: Fraction | None
    gwet_ac1: Fraction | None


@dataclass(frozen=True)
class BootstrapInterval:
    replicates: int
    valid_replicates: int
    lower: Fraction
    upper: Fraction


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ScreeningAgreementError(
            f"{field} must be a canonical lowercase SHA-256"
        )
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combined_primary_snapshot_sha256(
    calibration_snapshot_sha256: str,
    main_snapshot_sha256: str,
) -> str:
    calibration_hash = _validate_sha256(
        calibration_snapshot_sha256,
        "calibration_result_snapshot_sha256",
    )
    main_hash = _validate_sha256(
        main_snapshot_sha256,
        "main_result_snapshot_sha256",
    )
    return _canonical_sha256(
        {
            "calibration_result_snapshot_sha256": calibration_hash,
            "main_result_snapshot_sha256": main_hash,
        }
    )


# Compatibility alias for callers predating the public facade.
_combined_primary_snapshot_sha256 = combined_primary_snapshot_sha256


def _validate_status_criterion(
    status: str,
    criterion: str,
    *,
    candidate_id: str,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = (
        screening_results.LEGACY_SCREENING_STATUSES
    ),
) -> None:
    if status not in allowed_screening_statuses:
        raise ScreeningAgreementError(
            f"candidate_id={candidate_id!r}: invalid screening_status {status!r}"
        )
    if status == "included" and criterion not in allowed_inclusion_criteria:
        raise ScreeningAgreementError(
            f"candidate_id={candidate_id!r}: criterion {criterion!r} "
            "is invalid for included"
        )
    if status == "boundary" and criterion != "boundary":
        raise ScreeningAgreementError(
            f"candidate_id={candidate_id!r}: criterion must be 'boundary' "
            "for boundary"
        )
    if status == "excluded" and criterion not in EXCLUSION_CRITERIA:
        raise ScreeningAgreementError(
            f"candidate_id={candidate_id!r}: criterion {criterion!r} "
            "is invalid for excluded"
        )


def _fraction_or_none(numerator: int, denominator: int) -> Fraction | None:
    return None if denominator == 0 else Fraction(numerator, denominator)


def _point_estimates(
    pairs: Sequence[RatingPair],
    *,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = (
        screening_results.LEGACY_SCREENING_STATUSES
    ),
) -> PointEstimates:
    if not pairs:
        raise ScreeningAgreementError("agreement scope must contain candidates")
    matrix = {
        (left, right): 0
        for left in STATUS_CATEGORIES
        for right in STATUS_CATEGORIES
    }
    criterion_disagreements = {
        (left, right): 0
        for left in CRITERION_CATEGORIES
        for right in CRITERION_CATEGORIES
        if left != right
    }
    for pair in pairs:
        _validate_status_criterion(
            pair.status_a,
            pair.criterion_a,
            candidate_id=pair.candidate_id,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        _validate_status_criterion(
            pair.status_b,
            pair.criterion_b,
            candidate_id=pair.candidate_id,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        matrix[(pair.status_a, pair.status_b)] += 1
        if pair.criterion_a != pair.criterion_b:
            criterion_disagreements[(pair.criterion_a, pair.criterion_b)] += 1

    candidate_count = len(pairs)
    exact_status_count = sum(
        matrix[(category, category)] for category in STATUS_CATEGORIES
    )
    exact_criterion_count = sum(
        pair.criterion_a == pair.criterion_b for pair in pairs
    )
    positive_components: dict[str, tuple[int, int]] = {}
    negative_components: dict[str, tuple[int, int]] = {}
    positive_agreement: dict[str, Fraction | None] = {}
    negative_agreement: dict[str, Fraction | None] = {}
    for category in STATUS_CATEGORIES:
        true_positive = matrix[(category, category)]
        a_only = sum(
            matrix[(category, other)]
            for other in STATUS_CATEGORIES
            if other != category
        )
        b_only = sum(
            matrix[(other, category)]
            for other in STATUS_CATEGORIES
            if other != category
        )
        true_negative = candidate_count - true_positive - a_only - b_only
        positive = (2 * true_positive, 2 * true_positive + a_only + b_only)
        negative = (2 * true_negative, 2 * true_negative + a_only + b_only)
        positive_components[category] = positive
        negative_components[category] = negative
        positive_agreement[category] = _fraction_or_none(*positive)
        negative_agreement[category] = _fraction_or_none(*negative)

    total_ratings = 2 * candidate_count
    pooled = {
        category: sum(
            matrix[(category, other)] + matrix[(other, category)]
            for other in STATUS_CATEGORIES
        )
        for category in STATUS_CATEGORIES
    }
    observed_disagreement = Fraction(
        candidate_count - exact_status_count, candidate_count
    )
    expected_disagreement = sum(
        Fraction(
            count * (total_ratings - count),
            total_ratings * (total_ratings - 1),
        )
        for count in pooled.values()
    )
    alpha = (
        None
        if expected_disagreement == 0
        else 1 - observed_disagreement / expected_disagreement
    )
    expected_ac1 = sum(
        Fraction(count, total_ratings)
        * (1 - Fraction(count, total_ratings))
        / (len(STATUS_CATEGORIES) - 1)
        for count in pooled.values()
    )
    ac1_denominator = 1 - expected_ac1
    ac1 = (
        None
        if ac1_denominator == 0
        else (Fraction(exact_status_count, candidate_count) - expected_ac1)
        / ac1_denominator
    )
    return PointEstimates(
        matrix=matrix,
        criterion_disagreements=criterion_disagreements,
        exact_status_count=exact_status_count,
        exact_status_rate=Fraction(exact_status_count, candidate_count),
        exact_criterion_count=exact_criterion_count,
        exact_criterion_rate=Fraction(exact_criterion_count, candidate_count),
        positive_components=positive_components,
        negative_components=negative_components,
        positive_agreement=positive_agreement,
        negative_agreement=negative_agreement,
        krippendorff_alpha=alpha,
        gwet_ac1=ac1,
    )


def _format_fraction(value: Fraction) -> str:
    scaled, remainder = divmod(
        abs(value.numerator) * 1_000_000, value.denominator
    )
    if 2 * remainder >= value.denominator:
        scaled += 1
    sign = "-" if value.numerator < 0 and scaled else ""
    whole, fractional = divmod(scaled, 1_000_000)
    return f"{sign}{whole}.{fractional:06d}"


def _fraction_fields(prefix: str, value: Fraction | None) -> dict[str, str]:
    if value is None:
        return {
            f"{prefix}_numerator": "0",
            f"{prefix}_denominator": "0",
            prefix: "not_estimable",
        }
    return {
        f"{prefix}_numerator": str(value.numerator),
        f"{prefix}_denominator": str(value.denominator),
        prefix: _format_fraction(value),
    }


def _component_fields(
    prefix: str, components: tuple[int, int]
) -> dict[str, str]:
    numerator, denominator = components
    return {
        f"{prefix}_numerator": str(numerator),
        f"{prefix}_denominator": str(denominator),
        f"{prefix}_agreement": (
            "not_estimable"
            if denominator == 0
            else _format_fraction(Fraction(numerator, denominator))
        ),
    }


def _percentile_indices(valid_replicates: int) -> tuple[int, int]:
    """Return zero-based ceil(.025*m)-1 and ceil(.975*m)-1 indices."""

    if valid_replicates <= 0:
        raise ScreeningAgreementError("percentiles require valid replicates")

    def nearest_rank(numerator: int, denominator: int) -> int:
        rank = (valid_replicates * numerator + denominator - 1) // denominator
        return min(valid_replicates - 1, max(0, rank - 1))

    return nearest_rank(1, 40), nearest_rank(39, 40)


def _bootstrap_index(
    primary_snapshot_sha256: str,
    scope: str,
    *,
    replicate: int,
    draw: int,
    population_size: int,
) -> int:
    primary_hash = _validate_sha256(
        primary_snapshot_sha256, "primary_result_snapshot_sha256"
    )
    if scope not in {"calibration", "full_corpus"}:
        raise ScreeningAgreementError(f"invalid bootstrap scope {scope!r}")
    for value, field in ((replicate, "replicate"), (draw, "draw")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ScreeningAgreementError(
                f"bootstrap {field} must be a nonnegative integer"
            )
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size <= 0
    ):
        raise ScreeningAgreementError(
            "bootstrap population_size must be a positive integer"
        )
    payload = (
        BOOTSTRAP_ALGORITHM
        + "\0"
        + primary_hash
        + "\0"
        + scope
        + "\0"
        + str(replicate)
        + "\0"
        + str(draw)
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % population_size


def _bootstrap_intervals(
    pairs: Sequence[RatingPair],
    *,
    scope: str,
    primary_snapshot_sha256: str,
    replicates: int,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = (
        screening_results.LEGACY_SCREENING_STATUSES
    ),
) -> dict[str, BootstrapInterval]:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ScreeningAgreementError(
            "bootstrap_replicates must be a positive integer"
        )
    if scope not in {"calibration", "full_corpus"}:
        raise ScreeningAgreementError(f"invalid bootstrap scope {scope!r}")
    expected_count = 30 if scope == "calibration" else 202
    ordered = tuple(
        sorted(pairs, key=lambda pair: pair.candidate_id.encode("utf-8"))
    )
    if len(ordered) != expected_count:
        raise ScreeningAgreementError(
            f"{scope} bootstrap requires {expected_count} candidate pairs"
        )
    values: dict[str, list[Fraction]] = {
        metric: [] for metric in BOOTSTRAP_METRICS
    }
    for replicate in range(replicates):
        sample = tuple(
            ordered[
                _bootstrap_index(
                    primary_snapshot_sha256,
                    scope,
                    replicate=replicate,
                    draw=draw,
                    population_size=expected_count,
                )
            ]
            for draw in range(expected_count)
        )
        estimates = _point_estimates(
            sample,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        replicate_values = {
            "overall_exact_status_agreement": estimates.exact_status_rate,
            "exact_criterion_agreement": estimates.exact_criterion_rate,
            "krippendorff_alpha_nominal": estimates.krippendorff_alpha,
            "gwet_ac1_nominal": estimates.gwet_ac1,
        }
        for metric, estimate in replicate_values.items():
            if estimate is not None:
                values[metric].append(estimate)

    intervals: dict[str, BootstrapInterval] = {}
    for metric in BOOTSTRAP_METRICS:
        valid = sorted(values[metric])
        if not valid:
            raise ScreeningAgreementError(
                f"zero valid bootstrap replicates for {metric}"
            )
        lower_index, upper_index = _percentile_indices(len(valid))
        intervals[metric] = BootstrapInterval(
            replicates=replicates,
            valid_replicates=len(valid),
            lower=valid[lower_index],
            upper=valid[upper_index],
        )
    return intervals


def _normalize_result_row(
    source: Mapping[str, str], *, row_number: int, phase: str
) -> dict[str, str]:
    if not isinstance(source, Mapping):
        raise ScreeningAgreementError(
            f"{phase} row {row_number}: rating must be a mapping"
        )
    missing = set(RESULT_HEADER) - set(source)
    if missing:
        raise ScreeningAgreementError(
            f"{phase} row {row_number}: missing RESULT_HEADER fields "
            f"{sorted(missing)!r}"
        )
    row: dict[str, str] = {}
    for field in RESULT_HEADER:
        value = source[field]
        if not isinstance(value, str):
            raise ScreeningAgreementError(
                f"{phase} row {row_number}: {field} must be a string"
            )
        row[field] = value
    return row


def _phase_pairs(
    snapshot: screening_results.PhaseResultSnapshot,
    *,
    expected_phase: str,
    coordinator_snapshot_sha256: str,
    protocol_sha256: str,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = (
        screening_results.LEGACY_SCREENING_STATUSES
    ),
) -> tuple[list[RatingPair], set[str]]:
    if not isinstance(snapshot, screening_results.PhaseResultSnapshot):
        raise ScreeningAgreementError(
            f"{expected_phase} input must be a validated PhaseResultSnapshot"
        )
    if snapshot.phase != expected_phase:
        raise ScreeningAgreementError(
            f"expected {expected_phase} phase snapshot, found {snapshot.phase!r}"
        )
    _validate_sha256(
        snapshot.snapshot_sha256, f"{expected_phase} snapshot_sha256"
    )
    if snapshot.coordinator_snapshot_sha256 != coordinator_snapshot_sha256:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot coordinator binding "
            "does not match trusted coordinator"
        )
    if snapshot.protocol_sha256 != protocol_sha256:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot protocol binding "
            "does not match trusted protocol"
        )
    expected_ratings = 60 if expected_phase == "calibration" else 344
    expected_candidates = 30 if expected_phase == "calibration" else 172
    if len(snapshot.rows) != expected_ratings:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot requires {expected_ratings} ratings"
        )

    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    assignment_ids: set[str] = set()
    for row_number, source in enumerate(snapshot.rows, start=2):
        row = _normalize_result_row(
            source, row_number=row_number, phase=expected_phase
        )
        candidate_id = row["candidate_id"]
        if not candidate_id:
            raise ScreeningAgreementError(
                f"{expected_phase} row {row_number}: candidate_id is required"
            )
        if row["phase"] != expected_phase:
            raise ScreeningAgreementError(
                f"candidate_id={candidate_id!r}: rating phase "
                "does not match snapshot phase"
            )
        if row["snapshot_sha256"] != coordinator_snapshot_sha256:
            raise ScreeningAgreementError(
                f"candidate_id={candidate_id!r}: rating snapshot_sha256 "
                "does not match trusted coordinator"
            )
        _validate_sha256(
            row["input_sha256"],
            f"candidate_id={candidate_id!r} input_sha256",
        )
        _validate_status_criterion(
            row["screening_status"],
            row["criterion"],
            candidate_id=candidate_id,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        grouped[candidate_id].append(row)

    if len(grouped) != expected_candidates:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot requires "
            f"{expected_candidates} candidates"
        )
    pairs: list[RatingPair] = []
    for candidate_id in sorted(
        grouped, key=lambda value: value.encode("utf-8")
    ):
        ratings = sorted(
            grouped[candidate_id],
            key=lambda row: row["assignment_id"].encode("utf-8"),
        )
        if len(ratings) != 2:
            raise ScreeningAgreementError(
                f"candidate_id={candidate_id!r}: requires exactly two ratings"
            )
        if len({row["assignment_id"] for row in ratings}) != 2:
            raise ScreeningAgreementError(
                f"candidate_id={candidate_id!r}: requires "
                "two distinct assignment IDs"
            )
        if len({row["coder_id"] for row in ratings}) != 2:
            raise ScreeningAgreementError(
                f"candidate_id={candidate_id!r}: requires two distinct coder IDs"
            )
        if len({row["input_sha256"] for row in ratings}) != 1:
            raise ScreeningAgreementError(
                f"candidate_id={candidate_id!r}: ratings require "
                "matching input_sha256"
            )
        for rating in ratings:
            assignment_id = rating["assignment_id"]
            if assignment_id in assignment_ids:
                raise ScreeningAgreementError(
                    f"duplicate assignment_id {assignment_id!r}"
                )
            assignment_ids.add(assignment_id)
        pairs.append(
            RatingPair(
                candidate_id,
                ratings[0]["screening_status"],
                ratings[1]["screening_status"],
                ratings[0]["criterion"],
                ratings[1]["criterion"],
            )
        )
    return pairs, assignment_ids


def _validated_snapshot_pairs(
    calibration: screening_results.PhaseResultSnapshot,
    main: screening_results.PhaseResultSnapshot,
    *,
    coordinator_snapshot_sha256: str,
    protocol_sha256: str,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = (
        screening_results.LEGACY_SCREENING_STATUSES
    ),
) -> tuple[list[RatingPair], list[RatingPair], list[RatingPair]]:
    coordinator_hash = _validate_sha256(
        coordinator_snapshot_sha256, "coordinator_snapshot_sha256"
    )
    trusted_protocol_hash = _validate_sha256(
        protocol_sha256, "protocol_sha256"
    )
    calibration_pairs, calibration_assignments = _phase_pairs(
        calibration,
        expected_phase="calibration",
        coordinator_snapshot_sha256=coordinator_hash,
        protocol_sha256=trusted_protocol_hash,
        allowed_inclusion_criteria=allowed_inclusion_criteria,
        allowed_screening_statuses=allowed_screening_statuses,
    )
    main_pairs, main_assignments = _phase_pairs(
        main,
        expected_phase="main",
        coordinator_snapshot_sha256=coordinator_hash,
        protocol_sha256=trusted_protocol_hash,
        allowed_inclusion_criteria=allowed_inclusion_criteria,
        allowed_screening_statuses=allowed_screening_statuses,
    )
    if calibration.snapshot_sha256 == main.snapshot_sha256:
        raise ScreeningAgreementError(
            "calibration and main phase snapshot hashes must be distinct"
        )
    calibration_ids = {pair.candidate_id for pair in calibration_pairs}
    main_ids = {pair.candidate_id for pair in main_pairs}
    overlap = calibration_ids & main_ids
    if overlap:
        raise ScreeningAgreementError(
            f"candidate IDs occur in both phases: {sorted(overlap)!r}"
        )
    assignment_overlap = calibration_assignments & main_assignments
    if assignment_overlap:
        raise ScreeningAgreementError(
            f"assignment IDs occur in both phases: "
            f"{sorted(assignment_overlap)!r}"
        )
    full = sorted(
        (*calibration_pairs, *main_pairs),
        key=lambda pair: pair.candidate_id.encode("utf-8"),
    )
    if len(full) != 202:
        raise ScreeningAgreementError("full corpus requires 202 candidates")
    return calibration_pairs, main_pairs, full


def _scope_row(
    scope: str,
    pairs: Sequence[RatingPair],
    estimates: PointEstimates,
    provenance: Mapping[str, str],
    intervals: Mapping[str, BootstrapInterval],
) -> dict[str, str]:
    candidate_count = len(pairs)
    values: dict[str, str] = {
        "report_version": REPORT_VERSION,
        "scope": scope,
        **{field: provenance[field] for field in PROVENANCE_FIELDS},
        "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        "candidate_count": str(candidate_count),
        "rating_count": str(2 * candidate_count),
        **{
            f"rating_a_{left}_rating_b_{right}": str(
                estimates.matrix[(left, right)]
            )
            for left in STATUS_CATEGORIES
            for right in STATUS_CATEGORIES
        },
        "overall_exact_status_agreement_count": str(
            estimates.exact_status_count
        ),
        "overall_exact_status_agreement_denominator": str(candidate_count),
        "overall_exact_status_agreement_rate": _format_fraction(
            estimates.exact_status_rate
        ),
        "exact_criterion_agreement_count": str(
            estimates.exact_criterion_count
        ),
        "exact_criterion_agreement_denominator": str(candidate_count),
        "exact_criterion_agreement_rate": _format_fraction(
            estimates.exact_criterion_rate
        ),
        **{
            _criterion_disagreement_field(left, right): str(
                estimates.criterion_disagreements[(left, right)]
            )
            for left in CRITERION_CATEGORIES
            for right in CRITERION_CATEGORIES
            if left != right
        },
    }
    for category in STATUS_CATEGORIES:
        values.update(
            _component_fields(
                f"{category}_positive",
                estimates.positive_components[category],
            )
        )
        values.update(
            _component_fields(
                f"{category}_negative",
                estimates.negative_components[category],
            )
        )
    values.update(
        _fraction_fields(
            "krippendorff_alpha_nominal",
            estimates.krippendorff_alpha,
        )
    )
    values.update(
        _fraction_fields("gwet_ac1_nominal", estimates.gwet_ac1)
    )
    for metric in BOOTSTRAP_METRICS:
        interval = intervals[metric]
        values.update(
            {
                f"{metric}_bootstrap_replicates": str(interval.replicates),
                f"{metric}_bootstrap_valid_replicates": str(
                    interval.valid_replicates
                ),
                f"{metric}_bootstrap_ci95_lower": _format_fraction(
                    interval.lower
                ),
                f"{metric}_bootstrap_ci95_upper": _format_fraction(
                    interval.upper
                ),
            }
        )
    missing = set(AGREEMENT_REPORT_HEADER) - set(values)
    extra = set(values) - set(AGREEMENT_REPORT_HEADER)
    if missing or extra:
        raise ScreeningAgreementError(
            f"internal report schema mismatch: missing={sorted(missing)!r}, "
            f"extra={sorted(extra)!r}"
        )
    return {field: values[field] for field in AGREEMENT_REPORT_HEADER}


def _build_agreement_report(
    calibration: screening_results.PhaseResultSnapshot,
    main: screening_results.PhaseResultSnapshot,
    *,
    coordinator_snapshot_sha256: str,
    protocol_sha256: str,
    bootstrap_replicates: int,
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
    allowed_screening_statuses: tuple[str, ...] = (
        screening_results.LEGACY_SCREENING_STATUSES
    ),
) -> list[dict[str, str]]:
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise ScreeningAgreementError(
            "bootstrap_replicates must be a positive integer"
        )
    calibration_pairs, _, full_pairs = _validated_snapshot_pairs(
        calibration,
        main,
        coordinator_snapshot_sha256=coordinator_snapshot_sha256,
        protocol_sha256=protocol_sha256,
        allowed_inclusion_criteria=allowed_inclusion_criteria,
        allowed_screening_statuses=allowed_screening_statuses,
    )
    primary_hash = _combined_primary_snapshot_sha256(
        calibration.snapshot_sha256, main.snapshot_sha256
    )
    provenance = {
        "protocol_sha256": protocol_sha256,
        "coordinator_snapshot_sha256": coordinator_snapshot_sha256,
        "calibration_result_snapshot_sha256": calibration.snapshot_sha256,
        "main_result_snapshot_sha256": main.snapshot_sha256,
        "primary_result_snapshot_sha256": primary_hash,
    }
    report: list[dict[str, str]] = []
    for scope, pairs in (
        ("calibration", calibration_pairs),
        ("full_corpus", full_pairs),
    ):
        estimates = _point_estimates(
            pairs,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        intervals = _bootstrap_intervals(
            pairs,
            scope=scope,
            primary_snapshot_sha256=primary_hash,
            replicates=bootstrap_replicates,
            allowed_inclusion_criteria=allowed_inclusion_criteria,
            allowed_screening_statuses=allowed_screening_statuses,
        )
        report.append(
            _scope_row(scope, pairs, estimates, provenance, intervals)
        )
    return report


def _coordinator_snapshot_directory(
    coordinator_snapshot: Path | screening_results.CoordinatorSnapshot,
) -> Path:
    """Resolve a supported coordinator trust anchor to its directory."""

    if isinstance(coordinator_snapshot, screening_results.CoordinatorSnapshot):
        return coordinator_snapshot.directory
    if isinstance(coordinator_snapshot, Path):
        return coordinator_snapshot
    raise ScreeningAgreementError(
        "coordinator_snapshot must be a Path or CoordinatorSnapshot"
    )


def _coordinator_validation_arguments(
    coordinator_snapshot: Path | screening_results.CoordinatorSnapshot,
) -> dict[str, object]:
    if isinstance(coordinator_snapshot, screening_results.CoordinatorSnapshot):
        return {"coordinator": coordinator_snapshot}
    return {
        "coordinator_snapshot_dir": _coordinator_snapshot_directory(
            coordinator_snapshot
        )
    }


def _authoritative_coordinator_snapshot(
    coordinator_snapshot: Path | screening_results.CoordinatorSnapshot,
) -> screening_results.CoordinatorSnapshot:
    if isinstance(
        coordinator_snapshot,
        screening_results.CoordinatorSnapshot,
    ):
        capture = screening_results.reattest_coordinator_snapshot
    elif isinstance(coordinator_snapshot, Path):
        capture = screening_results.capture_coordinator_snapshot
    else:
        _coordinator_snapshot_directory(coordinator_snapshot)
        raise AssertionError("unreachable")

    try:
        return capture(coordinator_snapshot)
    except (screening_results.ScreeningResultError, OSError) as exc:
        raise ScreeningAgreementError(
            "coordinator snapshot failed authoritative revalidation: "
            f"{exc}"
        ) from exc


def _revalidate_phase_snapshot(
    snapshot: screening_results.PhaseResultSnapshot,
    *,
    expected_phase: str,
    coordinator_snapshot: Path | screening_results.CoordinatorSnapshot,
    reviewer_release_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path | None = None,
    calibration_result_snapshot_dir: Path | None = None,
    calibration_decision_snapshot_dir: Path | None = None,
) -> screening_results.PhaseResultSnapshot:
    """Reconstruct a phase snapshot against its complete release provenance."""

    if not isinstance(snapshot, screening_results.PhaseResultSnapshot):
        raise ScreeningAgreementError(
            f"{expected_phase} input must be a validated PhaseResultSnapshot"
        )
    validation_arguments = _coordinator_validation_arguments(
        coordinator_snapshot
    )
    validation_arguments["reviewer_release_snapshot_dir"] = (
        reviewer_release_snapshot_dir
    )
    for name, value in (
        (
            "calibration_reviewer_release_snapshot_dir",
            calibration_reviewer_release_snapshot_dir,
        ),
        (
            "calibration_result_snapshot_dir",
            calibration_result_snapshot_dir,
        ),
        (
            "calibration_decision_snapshot_dir",
            calibration_decision_snapshot_dir,
        ),
    ):
        if value is not None:
            validation_arguments[name] = value
    try:
        authoritative = screening_results.validate_phase_result_snapshot(
            snapshot.directory,
            **validation_arguments,
        )
    except (screening_results.ScreeningResultError, OSError) as exc:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot failed authoritative "
            f"revalidation: {exc}"
        ) from exc

    for field in (
        "directory",
        "phase",
        "rows",
        "snapshot_sha256",
        "coordinator_snapshot_sha256",
        "protocol_sha256",
        "reviewer_release_sha256",
        "manifest",
    ):
        if getattr(snapshot, field) != getattr(authoritative, field):
            raise ScreeningAgreementError(
                f"{expected_phase} snapshot {field} does not match "
                "authoritative capture"
            )
    if snapshot.fingerprints != authoritative.fingerprints:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot fingerprints do not match "
            "authoritative capture"
        )
    try:
        for fingerprint in (
            *snapshot.fingerprints,
            *authoritative.fingerprints,
        ):
            fingerprint.reattest()
    except (screening_results.ScreeningResultError, OSError) as exc:
        raise ScreeningAgreementError(
            f"{expected_phase} snapshot fingerprint reattestation failed: "
            f"{exc}"
        ) from exc
    if authoritative.phase != expected_phase:
        raise ScreeningAgreementError(
            f"expected {expected_phase} phase snapshot, "
            f"found {authoritative.phase!r}"
        )
    return authoritative


def _revalidate_calibration_decision_snapshot(
    snapshot: screening_results.CalibrationDecisionSnapshot,
    *,
    coordinator_snapshot: Path | screening_results.CoordinatorSnapshot,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration: screening_results.PhaseResultSnapshot,
) -> screening_results.CalibrationDecisionSnapshot:
    """Reconstruct the calibration gate against its release and result."""

    if not isinstance(
        snapshot, screening_results.CalibrationDecisionSnapshot
    ):
        raise ScreeningAgreementError(
            "calibration decision input must be a validated "
            "CalibrationDecisionSnapshot"
        )
    try:
        authoritative = (
            screening_results.validate_calibration_decision_snapshot(
                snapshot.directory,
                coordinator_snapshot_dir=_coordinator_snapshot_directory(
                    coordinator_snapshot
                ),
                calibration_reviewer_release_snapshot_dir=(
                    calibration_reviewer_release_snapshot_dir
                ),
                calibration_result_snapshot_dir=calibration.directory,
            )
        )
    except (screening_results.ScreeningResultError, OSError) as exc:
        raise ScreeningAgreementError(
            "calibration decision snapshot failed authoritative "
            f"revalidation: {exc}"
        ) from exc

    for field in (
        "directory",
        "decision",
        "snapshot_sha256",
        "coordinator_snapshot_sha256",
        "calibration_result_snapshot_sha256",
        "manifest",
    ):
        if getattr(snapshot, field) != getattr(authoritative, field):
            raise ScreeningAgreementError(
                f"calibration decision snapshot {field} does not match "
                "authoritative capture"
            )
    if snapshot.fingerprints != authoritative.fingerprints:
        raise ScreeningAgreementError(
            "calibration decision snapshot fingerprints do not match "
            "authoritative capture"
        )
    try:
        for fingerprint in (
            *snapshot.fingerprints,
            *authoritative.fingerprints,
        ):
            fingerprint.reattest()
    except (screening_results.ScreeningResultError, OSError) as exc:
        raise ScreeningAgreementError(
            "calibration decision snapshot fingerprint reattestation "
            f"failed: {exc}"
        ) from exc
    return authoritative


def build_agreement_report(
    coordinator_snapshot: Path | screening_results.CoordinatorSnapshot,
    calibration_reviewer_release_snapshot_dir: Path,
    calibration: screening_results.PhaseResultSnapshot,
    calibration_decision: screening_results.CalibrationDecisionSnapshot,
    main_reviewer_release_snapshot_dir: Path,
    main: screening_results.PhaseResultSnapshot,
) -> list[dict[str, str]]:
    """Build the report from the authoritative reviewer-release chain."""

    coordinator_snapshot = _authoritative_coordinator_snapshot(
        coordinator_snapshot
    )
    authoritative_calibration = _revalidate_phase_snapshot(
        calibration,
        expected_phase="calibration",
        coordinator_snapshot=coordinator_snapshot,
        reviewer_release_snapshot_dir=(
            calibration_reviewer_release_snapshot_dir
        ),
    )
    authoritative_decision = _revalidate_calibration_decision_snapshot(
        calibration_decision,
        coordinator_snapshot=coordinator_snapshot,
        calibration_reviewer_release_snapshot_dir=(
            calibration_reviewer_release_snapshot_dir
        ),
        calibration=authoritative_calibration,
    )
    authoritative_main = _revalidate_phase_snapshot(
        main,
        expected_phase="main",
        coordinator_snapshot=coordinator_snapshot,
        reviewer_release_snapshot_dir=main_reviewer_release_snapshot_dir,
        calibration_reviewer_release_snapshot_dir=(
            calibration_reviewer_release_snapshot_dir
        ),
        calibration_result_snapshot_dir=(
            authoritative_calibration.directory
        ),
        calibration_decision_snapshot_dir=authoritative_decision.directory,
    )
    if (
        authoritative_calibration.coordinator_snapshot_sha256
        != authoritative_main.coordinator_snapshot_sha256
    ):
        raise ScreeningAgreementError(
            "calibration and main coordinator bindings do not agree"
        )
    if (
        authoritative_calibration.protocol_sha256
        != authoritative_main.protocol_sha256
    ):
        raise ScreeningAgreementError(
            "calibration and main protocol bindings do not agree"
        )

    return _build_agreement_report(
        authoritative_calibration,
        authoritative_main,
        coordinator_snapshot_sha256=(
            authoritative_calibration.coordinator_snapshot_sha256
        ),
        protocol_sha256=authoritative_calibration.protocol_sha256,
        bootstrap_replicates=PRODUCTION_BOOTSTRAP_REPLICATES,
        allowed_inclusion_criteria=coordinator_snapshot.allowed_inclusion_criteria,
        allowed_screening_statuses=coordinator_snapshot.allowed_screening_statuses,
    )


def render_agreement_csv(
    rows: Sequence[Mapping[str, str]],
) -> bytes:
    """Render calibration and full-corpus rows as canonical UTF-8 CSV."""

    captured = list(rows)
    scopes = [
        row.get("scope") if isinstance(row, Mapping) else None
        for row in captured
    ]
    if len(captured) != 2 or scopes != ["calibration", "full_corpus"]:
        raise ScreeningAgreementError(
            "agreement CSV requires calibration then full_corpus rows"
        )
    expected = set(AGREEMENT_REPORT_HEADER)
    normalized: list[dict[str, str]] = []
    for row_number, row in enumerate(captured, start=2):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ScreeningAgreementError(
                f"report row {row_number} does not match "
                "AGREEMENT_REPORT_HEADER"
            )
        if any(
            not isinstance(row[field], str)
            for field in AGREEMENT_REPORT_HEADER
        ):
            raise ScreeningAgreementError(
                f"report row {row_number} contains a non-string value"
            )
        normalized.append(
            {field: row[field] for field in AGREEMENT_REPORT_HEADER}
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=AGREEMENT_REPORT_HEADER,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(normalized)
    return buffer.getvalue().encode("utf-8")


render_csv_bytes = render_agreement_csv
