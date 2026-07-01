"""Parse unsealed adjudication dossiers into normalized recommendations."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
import re
import sys

try:
    from paper.scripts import integrate_screening
except ModuleNotFoundError:  # Direct execution from paper/scripts.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paper.scripts import integrate_screening


STATUSES = frozenset({"included", "boundary", "excluded"})
CRITERIA = frozenset(
    {
        "include-1",
        "include-2",
        "include-3",
        "include-4",
        "boundary",
        "exclude-fixed-racing-line",
        "exclude-appearance-dynamics",
        "exclude-traffic-only",
        "exclude-insufficient-detail",
        "exclude-out-of-scope",
    }
)
ENTRY_HEADING = re.compile(r"^#{2,3}\s+(C\d{4})\b.*$", re.MULTILINE)
RECOMMENDATION = re.compile(
    r"`?(included|boundary|excluded)`?\s*(?:/|-)\s*`?([a-z0-9-]+)`?"
)


@dataclass(frozen=True)
class DraftRecommendation:
    candidate_id: str
    status: str
    criterion: str
    needs_accountable_author_review: bool
    draft_file: str


def parse_draft(path: Path) -> list[DraftRecommendation]:
    """Return one normalized recommendation for each dossier entry."""

    text = path.read_text(encoding="utf-8")
    headings = list(ENTRY_HEADING.finditer(text))
    recommendations: list[DraftRecommendation] = []
    seen: set[str] = set()

    for index, heading in enumerate(headings):
        candidate_id = heading.group(1)
        if candidate_id in seen:
            raise ValueError(f"{path}: duplicate candidate_id={candidate_id!r}")
        seen.add(candidate_id)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.start() : end]
        lines = [line for line in section.splitlines() if "Recommendation:" in line]
        if len(lines) != 1:
            raise ValueError(
                f"{path}: candidate_id={candidate_id!r} must have one recommendation"
            )
        match = RECOMMENDATION.search(lines[0])
        if match is None:
            raise ValueError(
                f"{path}: candidate_id={candidate_id!r} has an invalid recommendation"
            )
        status, criterion = match.groups()
        if status not in STATUSES or criterion not in CRITERIA:
            raise ValueError(
                f"{path}: candidate_id={candidate_id!r} has unsupported "
                f"status/criterion={status!r}/{criterion!r}"
            )
        recommendations.append(
            DraftRecommendation(
                candidate_id=candidate_id,
                status=status,
                criterion=criterion,
                needs_accountable_author_review=(
                    "NEEDS_ACCOUNTABLE_AUTHOR_REVIEW" in section
                ),
                draft_file=path.name,
            )
        )

    return recommendations


def build_rows(
    recommendations: list[DraftRecommendation],
    ratings: dict[str, tuple[dict[str, str], dict[str, str]]],
    unresolved: dict[str, tuple[dict[str, str], ...]],
    *,
    snapshot_sha256: str,
    primary_snapshot_sha256: str,
) -> list[dict[str, str]]:
    """Bind non-final recommendations to immutable screening facts."""

    by_candidate = {row.candidate_id: row for row in recommendations}
    if len(by_candidate) != len(recommendations):
        raise ValueError("draft recommendations contain duplicate candidate IDs")
    if set(by_candidate) != set(ratings):
        raise ValueError("draft recommendations do not cover the required candidates")

    rows: list[dict[str, str]] = []
    for candidate_id in sorted(by_candidate, key=str.encode):
        recommendation = by_candidate[candidate_id]
        pair = ratings[candidate_id]
        if pair[0]["input_sha256"] != pair[1]["input_sha256"]:
            raise ValueError(f"candidate_id={candidate_id!r} has inconsistent inputs")
        trigger_ids = integrate_screening._adjudication_trigger_ids(
            pair,
            has_unresolved_conflict=candidate_id in unresolved,
        )
        resolved_conflict_ids = tuple(
            row["conflict_id"] for row in unresolved.get(candidate_id, ())
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "draft_file": recommendation.draft_file,
                "draft_status": recommendation.status,
                "draft_criterion": recommendation.criterion,
                "needs_accountable_author_review": str(
                    recommendation.needs_accountable_author_review
                ).lower(),
                "input_sha256": pair[0]["input_sha256"],
                "snapshot_sha256": snapshot_sha256,
                "primary_snapshot_sha256": primary_snapshot_sha256,
                "assignment_ids": ";".join(
                    rating["assignment_id"] for rating in pair
                ),
                "reviewer_ids": ";".join(rating["coder_id"] for rating in pair),
                "raw_access_statuses": ";".join(
                    rating["access_status"] for rating in pair
                ),
                "trigger_ids": ";".join(trigger_ids) if trigger_ids else "NR",
                "resolved_conflict_ids": (
                    ";".join(resolved_conflict_ids)
                    if resolved_conflict_ids
                    else "NR"
                ),
                "conversion_status": (
                    "requires_accountable_author_review"
                    if recommendation.needs_accountable_author_review
                    else "ready_for_normalization"
                ),
            }
        )
    return rows


WORKSHEET_HEADER = (
    "candidate_id",
    "draft_file",
    "draft_status",
    "draft_criterion",
    "needs_accountable_author_review",
    "input_sha256",
    "snapshot_sha256",
    "primary_snapshot_sha256",
    "assignment_ids",
    "reviewer_ids",
    "raw_access_statuses",
    "trigger_ids",
    "resolved_conflict_ids",
    "conversion_status",
)


def render_workbook(rows: list[dict[str, str]]) -> str:
    """Render the non-final worksheet as canonical LF-delimited CSV."""

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=WORKSHEET_HEADER,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _load_recommendations(paths: list[Path]) -> list[DraftRecommendation]:
    rows: list[DraftRecommendation] = []
    for path in paths:
        rows.extend(parse_draft(path))
    return rows


def build_workbook_from_snapshots(
    *,
    drafts: list[Path],
    coordinator_snapshot: Path,
    calibration_reviewer_release: Path,
    calibration_result_snapshot: Path,
    calibration_decision_snapshot: Path,
    main_reviewer_release: Path,
    main_result_snapshot: Path,
) -> list[dict[str, str]]:
    """Build a review worksheet without creating an adjudication decision."""

    context = integrate_screening._load_context(
        coordinator_snapshot,
        calibration_reviewer_release,
        calibration_result_snapshot,
        calibration_decision_snapshot,
        main_reviewer_release,
        main_result_snapshot,
    )
    ratings = integrate_screening._ratings_by_candidate(context)
    unresolved = integrate_screening._unresolved_screening_conflicts(context)
    required = integrate_screening._required_adjudications(
        context, ratings, unresolved
    )
    required_ratings = {candidate_id: ratings[candidate_id] for candidate_id in required}
    required_unresolved = {
        candidate_id: unresolved[candidate_id]
        for candidate_id in required
        if candidate_id in unresolved
    }
    return build_rows(
        _load_recommendations(drafts),
        required_ratings,
        required_unresolved,
        snapshot_sha256=context.coordinator_snapshot_sha256,
        primary_snapshot_sha256=context.primary_snapshot_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-final adjudication reconciliation worksheet."
    )
    parser.add_argument("--draft", type=Path, action="append", required=True)
    parser.add_argument("--coordinator-snapshot", type=Path, required=True)
    parser.add_argument("--calibration-reviewer-release", type=Path, required=True)
    parser.add_argument("--calibration-result-snapshot", type=Path, required=True)
    parser.add_argument("--calibration-decision-snapshot", type=Path, required=True)
    parser.add_argument("--main-reviewer-release", type=Path, required=True)
    parser.add_argument("--main-result-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_workbook_from_snapshots(
        drafts=args.draft,
        coordinator_snapshot=args.coordinator_snapshot,
        calibration_reviewer_release=args.calibration_reviewer_release,
        calibration_result_snapshot=args.calibration_result_snapshot,
        calibration_decision_snapshot=args.calibration_decision_snapshot,
        main_reviewer_release=args.main_reviewer_release,
        main_result_snapshot=args.main_result_snapshot,
    )
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        handle.write(render_workbook(rows))


if __name__ == "__main__":
    main()
