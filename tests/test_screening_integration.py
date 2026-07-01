from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

import paper.scripts.integrate_screening as integration
import paper.scripts.prepare_screening_batches as screening_batches
import paper.scripts.screening_agreement as screening_agreement
import paper.scripts.screening_results as screening_results

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper" / "data" / "screening_protocol.md"
TAXONOMY = ROOT / "paper" / "data" / "taxonomy.json"
EXECUTION_PROFILE = (
    ROOT / "paper" / "data" / "screening_execution_profile.json"
)
REVIEWER_PROMPT = (
    ROOT / "paper" / "data" / "screening_reviewer_prompt.md"
)


EXECUTION_REGISTER_HEADER = (
    "execution_id",
    "role_id",
    "role_type",
    "context_id",
    "task",
    "stage_path",
    "stage_snapshot_sha256",
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

LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT = (
    "Fresh context; no shared conversation history, memory, ratings, or "
    "results were supplied; provider retrieval-cache isolation was not exposed."
)


def _csv_bytes(
    header: tuple[str, ...], rows: list[dict[str, str]]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=header,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_csv(
    path: Path, header: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_csv_bytes(header, rows))


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        return tuple(reader.fieldnames or ()), list(reader)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _combined_primary_hash(calibration_hash: str, main_hash: str) -> str:
    return _canonical_sha256(
        {
            "calibration_result_snapshot_sha256": calibration_hash,
            "main_result_snapshot_sha256": main_hash,
        }
    )


def _candidate(
    candidate_id: str, *, keyless: bool = False
) -> dict[str, str]:
    key = "" if keyless else f"Author2026{candidate_id}"
    return {
        "candidate_id": candidate_id,
        "cite_key": key,
        "title": f"Generated Course Study {candidate_id}",
        "authors": "Alex Author",
        "year": "2026",
        "venue": "Journal of Course Generation",
        "doi": f"10.1000/{candidate_id.casefold()}",
        "url": f"https://example.org/{candidate_id}",
        "source_type": "journal article",
        "discovery_stream": f"stream-{int(candidate_id[1:]) % 7}",
        "discovery_query": f"query-{int(candidate_id[1:]) % 13}",
        "discovery_agent": "discovery-worker",
        "screening_status": "excluded" if keyless else "candidate",
        "exclusion_reason": (
            "The earlier provisional screen considered this report out of scope."
            if keyless
            else ""
        ),
        "metadata_status": "verified",
        "metadata_evidence": f"publisher::https://example.org/{candidate_id}",
    }


def _conflict(
    conflict_id: str, candidate_id: str
) -> dict[str, str]:
    return {
        "conflict_id": conflict_id,
        "record_type": "candidate",
        "record_key": candidate_id,
        "field": "screening_status",
        "value_a": "candidate",
        "value_b": "excluded",
        "resolution": "",
        "resolver": "",
        "resolution_evidence": "",
    }


def _bibliography(candidate: dict[str, str]) -> dict[str, str]:
    return {
        "candidate_id": candidate["candidate_id"],
        "cite_key": candidate["cite_key"],
        "entry_type": "article",
        "key_author": "Author",
        "authors": candidate["authors"],
        "author_kinds": "personal",
        "title": candidate["title"],
        "year": candidate["year"],
        "venue_field": "journal",
        "venue": candidate["venue"],
        "doi": candidate["doi"],
        "url": candidate["url"],
    }


def _rating(
    assignment: dict[str, str], changes: dict[str, str] | None = None
) -> dict[str, str]:
    candidate_id = assignment["candidate_id"]
    archive = f"https://archive.example.org/{candidate_id}/versions/1.0/"
    source = f"https://example.org/{candidate_id}"
    row = {
        "assignment_id": assignment["assignment_id"],
        "phase": assignment["phase"],
        "candidate_id": candidate_id,
        "input_sha256": assignment["input_sha256"],
        "snapshot_sha256": assignment["snapshot_sha256"],
        "batch_id": assignment["batch_id"],
        "coder_id": assignment["batch_id"],
        "screened_on": "2026-06-30",
        "screening_status": "included",
        "criterion": "include-1",
        "access_status": "full_text",
        "source_urls": f"{archive};{source}",
        "evidence_version": "version-of-record-1",
        "evidence_retrieved_on": "2026-06-29",
        "evidence_archive_url": archive,
        "evidence_sha256": hashlib.sha256(
            f"artifact:{candidate_id}".encode("utf-8")
        ).hexdigest(),
        "screening_locator": "Algorithm 1; Section 3, page 7",
        "exclusion_reason": "NR",
        "notes": "NR",
    }
    if changes:
        row.update(changes)
    return row


@dataclass
class ScreeningCase:
    root: Path
    coordinator: Path
    calibration_release: Path
    calibration: Path
    calibration_decision: Path
    main_release: Path
    main: Path
    citation_keys: Path
    candidates: list[dict[str, str]]
    stages: dict[tuple[str, str], Path]
    conflicts: list[dict[str, str]]
    ratings: list[dict[str, str]]
    calibration_hash: str
    main_hash: str
    execution_register_path: Path | None = None

    @property
    def primary_hash(self) -> str:
        return _combined_primary_hash(self.calibration_hash, self.main_hash)

    def ratings_for(self, candidate_id: str) -> list[dict[str, str]]:
        return sorted(
            (
                row
                for row in self.ratings
                if row["candidate_id"] == candidate_id
            ),
            key=lambda row: row["assignment_id"].encode("utf-8"),
        )


def _write_phase_result_inputs(
    root: Path,
    ratings: list[dict[str, str]],
    phase: str,
) -> list[Path]:
    result_paths: list[Path] = []
    for batch_id in screening_results.BATCH_IDS:
        path = root / phase / f"{batch_id}.csv"
        phase_rows = [
            row
            for row in ratings
            if row["phase"] == phase and row["batch_id"] == batch_id
        ]
        _write_csv(path, screening_results.RESULT_HEADER, phase_rows)
        result_paths.append(path)
    return result_paths

def _stage_release_fixture(
    reviewer_release: Path,
    phase: str,
    staging_root: Path,
) -> dict[tuple[str, str], Path]:
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(staging_root, 0o700)
    relative_paths = {
        "execution_profile.json",
        "protocol.md",
        "release_manifest.csv",
        "reviewer_prompt_template.md",
        "SHA256SUMS",
        *(
            f"packets/{filename}"
            for filename in screening_batches.PACKET_FILENAMES
        ),
    }
    payloads = {
        relative: (reviewer_release / relative).read_bytes()
        for relative in relative_paths
    }
    task = (
        "calibration-screening"
        if phase == "calibration"
        else "main-screening"
    )
    stages: dict[tuple[str, str], Path] = {}
    for filename in screening_batches.PACKET_FILENAMES:
        role_id = filename.removesuffix(".csv")
        token = hashlib.sha256(
            f"{phase}:{role_id}".encode("utf-8")
        ).hexdigest()[:32]
        private_parent = staging_root / f"{role_id}-{token}"
        private_parent.mkdir(mode=0o700)
        os.chmod(private_parent, 0o700)
        stage = private_parent / "v1"
        artifacts = screening_batches.build_reviewer_stage_artifacts(
            payloads,
            role_id,
            stage,
        )
        screening_batches.publish_snapshot(stage, artifacts)
        screening_batches.validate_reviewer_stage_snapshot(stage)
        stages[(task, role_id)] = stage
    return stages



def _build_case(
    tmp_path: Path,
    *,
    scenarios: dict[str, tuple[dict[str, str], dict[str, str]]] | None = None,
    conflicts: list[dict[str, str]] | None = None,
    keyless: frozenset[str] = frozenset(),
    calibration_status_disagreements: int = 0,
) -> ScreeningCase:
    scenarios = dict(scenarios or {})
    conflicts = conflicts or []
    candidates = [
        _candidate(f"C{number:04d}", keyless=f"C{number:04d}" in keyless)
        for number in range(1, 203)
    ]
    bibliography = [
        _bibliography(row)
        for row in candidates
        if row["screening_status"] != "excluded"
    ]
    bibliography.sort(
        key=lambda row: (
            row["cite_key"].casefold(),
            row["cite_key"],
            row["candidate_id"],
        )
    )
    citation_keys = [
        {"candidate_id": row["candidate_id"], "cite_key": row["cite_key"]}
        for row in candidates
        if row["cite_key"]
    ]

    inputs = tmp_path / "inputs"
    _write_csv(
        inputs / "candidates.csv",
        screening_batches.CANDIDATE_HEADER,
        candidates,
    )
    _write_csv(
        inputs / "conflicts.csv",
        screening_batches.CONFLICT_HEADER,
        conflicts,
    )
    _write_csv(
        inputs / "bibliography.csv",
        screening_batches.BIBLIOGRAPHY_HEADER,
        bibliography,
    )
    _write_csv(
        inputs / "citation_keys.csv",
        screening_batches.CITATION_KEY_HEADER,
        citation_keys,
    )

    coordinator = tmp_path / "coordinator" / "v1"
    coordinator.parent.mkdir(parents=True)
    screening_batches.freeze_snapshot(
        candidates=inputs / "candidates.csv",
        conflicts=inputs / "conflicts.csv",
        bibliography=inputs / "bibliography.csv",
        citation_keys=inputs / "citation_keys.csv",
        taxonomy=TAXONOMY,
        protocol=PROTOCOL,
        execution_profile=EXECUTION_PROFILE,
        reviewer_prompt_template=REVIEWER_PROMPT,
        output_dir=coordinator,
    )
    manifest_header, manifest = _read_csv(coordinator / "manifest.csv")
    assert manifest_header == screening_batches.MANIFEST_HEADER
    assert len(manifest) == 404

    assignments_by_candidate: dict[str, list[dict[str, str]]] = {}
    for assignment in manifest:
        assignments_by_candidate.setdefault(
            assignment["candidate_id"], []
        ).append(assignment)
    for assignments in assignments_by_candidate.values():
        assignments.sort(
            key=lambda row: row["assignment_id"].encode("utf-8")
        )

    calibration_ids = sorted(
        (
            candidate_id
            for candidate_id, assignments in assignments_by_candidate.items()
            if assignments[0]["phase"] == "calibration"
        ),
        key=lambda value: value.encode("utf-8"),
    )
    disagreement_ids = [
        candidate_id
        for candidate_id in calibration_ids
        if candidate_id not in scenarios and candidate_id not in keyless
    ][:calibration_status_disagreements]
    if len(disagreement_ids) != calibration_status_disagreements:
        raise AssertionError("not enough calibration candidates for disagreement fixture")
    for candidate_id in disagreement_ids:
        scenarios[candidate_id] = (
            {},
            {
                "screening_status": "excluded",
                "criterion": "exclude-out-of-scope",
                "exclusion_reason": (
                    "The inspected source contains no generated course geometry "
                    "or transferable course-generation procedure."
                ),
            },
        )

    diversity_id = next(
        candidate_id
        for candidate_id in calibration_ids
        if candidate_id not in scenarios and candidate_id not in keyless
    )
    diversity_reason = (
        "The report varies only sensor noise while every course geometry "
        "remains fixed throughout the experiments."
    )
    scenarios[diversity_id] = (
        {
            "screening_status": "excluded",
            "criterion": "exclude-appearance-dynamics",
            "exclusion_reason": diversity_reason,
        },
        {
            "screening_status": "excluded",
            "criterion": "exclude-appearance-dynamics",
            "exclusion_reason": diversity_reason,
        },
    )

    ratings: list[dict[str, str]] = []
    for candidate_id in sorted(
        assignments_by_candidate, key=lambda value: value.encode("utf-8")
    ):
        candidate_specs = scenarios.get(candidate_id, ({}, {}))
        for assignment, changes in zip(
            assignments_by_candidate[candidate_id],
            candidate_specs,
            strict=True,
        ):
            ratings.append(_rating(assignment, changes))

    calibration_release = tmp_path / "calibration-reviewer-release" / "v1"
    calibration_release.parent.mkdir(parents=True)
    screening_batches.release_snapshot(
        coordinator,
        "calibration",
        calibration_release,
    )
    stages = _stage_release_fixture(
        calibration_release,
        "calibration",
        tmp_path / "execution-stages",
    )
    calibration = tmp_path / "calibration-results" / "v1"
    calibration.parent.mkdir(parents=True)
    screening_results.seal_phase_results(
        coordinator,
        "calibration",
        _write_phase_result_inputs(
            tmp_path / "result-inputs", ratings, "calibration"
        ),
        calibration,
        reviewer_release_snapshot_dir=calibration_release,
    )
    captured_calibration = screening_results.validate_phase_result_snapshot(
        calibration,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=calibration_release,
    )

    calibration_selection = [
        row["candidate_id"]
        for row in _read_csv(coordinator / "calibration_selection.csv")[1]
    ]
    calibration_rows = captured_calibration.rows
    by_calibration_candidate: dict[str, list[dict[str, str]]] = {}
    for row in calibration_rows:
        by_calibration_candidate.setdefault(
            row["candidate_id"], []
        ).append(row)
    agreement_numerator = sum(
        len({row["screening_status"] for row in pair}) == 1
        for pair in by_calibration_candidate.values()
    )
    assignment_ids = [row["assignment_id"] for row in calibration_rows]
    decision_input = tmp_path / "calibration-decision-input.csv"
    decision_row = {
        "decision_id": "calibration-gate-v1",
        "protocol_sha256": captured_calibration.protocol_sha256,
        "coordinator_snapshot_sha256": (
            captured_calibration.coordinator_snapshot_sha256
        ),
        "calibration_result_snapshot_sha256": (
            captured_calibration.snapshot_sha256
        ),
        "candidate_ids_sha256": screening_results.sequence_ids_sha256(
            calibration_selection
        ),
        "assignment_ids_sha256": screening_results.ordered_ids_sha256(
            assignment_ids
        ),
        "status_agreement_numerator": str(agreement_numerator),
        "status_agreement_denominator": "30",
        "status_agreement": screening_results.canonical_ratio(
            agreement_numerator, 30
        ),
        "systematic_ambiguity": "false",
        "decision": (
            "release" if agreement_numerator >= 24 else "revise"
        ),
        "decided_on": "2026-06-30",
        "decision_makers": "accountable-author",
        "resolution_evidence": (
            "The accountable author reviewed all calibration disagreements "
            "and found no systematic protocol ambiguity before main release."
        ),
    }
    _write_csv(
        decision_input,
        screening_results.CALIBRATION_DECISION_HEADER,
        [decision_row],
    )
    calibration_decision = tmp_path / "calibration-decision" / "v1"
    calibration_decision.parent.mkdir(parents=True)
    screening_results.seal_calibration_decision(
        coordinator,
        calibration,
        decision_input,
        calibration_decision,
        calibration_reviewer_release_snapshot_dir=calibration_release,
    )

    main_release = tmp_path / "main-reviewer-release" / "v1"
    main_release.parent.mkdir(parents=True)
    screening_batches.release_snapshot(
        coordinator,
        "main",
        main_release,
        calibration_reviewer_release_snapshot=calibration_release,
        calibration_result_snapshot=calibration,
        calibration_decision_snapshot=calibration_decision,
    )
    stages.update(
        _stage_release_fixture(
            main_release,
            "main",
            tmp_path / "execution-stages",
        )
    )
    main = tmp_path / "main-results" / "v1"
    main.parent.mkdir(parents=True)
    screening_results.seal_phase_results(
        coordinator,
        "main",
        _write_phase_result_inputs(tmp_path / "result-inputs", ratings, "main"),
        main,
        reviewer_release_snapshot_dir=main_release,
        calibration_reviewer_release_snapshot_dir=calibration_release,
        calibration_result_snapshot_dir=calibration,
        calibration_decision_snapshot_dir=calibration_decision,
    )
    captured_main = screening_results.validate_phase_result_snapshot(
        main,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=main_release,
        calibration_reviewer_release_snapshot_dir=calibration_release,
        calibration_result_snapshot_dir=calibration,
        calibration_decision_snapshot_dir=calibration_decision,
    )

    return ScreeningCase(
        root=tmp_path,
        coordinator=coordinator,
        calibration_release=calibration_release,
        calibration=calibration,
        calibration_decision=calibration_decision,
        main_release=main_release,
        main=main,
        stages=stages,
        citation_keys=inputs / "citation_keys.csv",
        candidates=candidates,
        conflicts=conflicts,
        ratings=ratings,
        calibration_hash=captured_calibration.snapshot_sha256,
        main_hash=captured_main.snapshot_sha256,
    )


def _trigger_scenarios() -> dict[
    str, tuple[dict[str, str], dict[str, str]]
]:
    reason_one = (
        "The report generates traffic participants while the course geometry "
        "remains fixed throughout the evaluation."
    )
    reason_two = (
        "The source synthesizes traffic behavior on roads whose geometry is "
        "fixed for every reported experiment."
    )
    direct_reason = (
        "The report varies only vehicle mass while every course geometry "
        "remains fixed throughout the experiments."
    )
    return {
        "C0001": (
            {},
            {
                "screening_status": "excluded",
                "criterion": "exclude-out-of-scope",
                "exclusion_reason": (
                    "The report does not generate or characterize course geometry "
                    "in the inspected full text."
                ),
            },
        ),
        "C0002": (
            {"criterion": "include-1"},
            {"criterion": "include-2"},
        ),
        "C0003": (
            {
                "screening_status": "excluded",
                "criterion": "exclude-traffic-only",
                "exclusion_reason": reason_one,
            },
            {
                "screening_status": "excluded",
                "criterion": "exclude-traffic-only",
                "exclusion_reason": reason_two,
            },
        ),
        "C0004": (
            {
                "screening_status": "excluded",
                "criterion": "exclude-appearance-dynamics",
                "exclusion_reason": direct_reason,
            },
            {
                "screening_status": "excluded",
                "criterion": "exclude-appearance-dynamics",
                "exclusion_reason": direct_reason,
            },
        ),
    }



def _resolution_evidence(row: dict[str, str]) -> dict[str, object]:
    parsed = json.loads(row["resolution_evidence"])
    assert isinstance(parsed, dict)
    return parsed


def _store_resolution_evidence(
    row: dict[str, str],
    evidence: dict[str, object],
) -> None:
    row["resolution_evidence"] = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _mutate_resolution_evidence(
    row: dict[str, str],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    evidence = _resolution_evidence(row)
    mutation(evidence)
    _store_resolution_evidence(row, evidence)

def _adjudication_row(
    case: ScreeningCase,
    candidate_id: str,
    *,
    screening_status: str = "included",
    criterion: str = "include-1",
    exclusion_reason: str = "NR",
    resolved_conflict_ids: tuple[str, ...] | None = None,
    screening_locator: str = "Algorithm 1; Section 3, page 7",
    transfer_source_fact: str | None = None,
) -> dict[str, str]:
    ratings = case.ratings_for(candidate_id)
    assert len(ratings) == 2
    archive = f"https://archive.example.org/{candidate_id}/versions/1.0/"
    source = f"https://example.org/{candidate_id}"
    available_conflicts = tuple(
        sorted(
            (
                row
                for row in case.conflicts
                if row["record_type"] == "candidate"
                and row["record_key"] == candidate_id
                and row["field"] == "screening_status"
                and not row["resolution"]
            ),
            key=lambda row: row["conflict_id"].encode("utf-8"),
        )
    )
    available_by_id = {
        row["conflict_id"]: row for row in available_conflicts
    }
    selected_conflict_ids = (
        tuple(available_by_id)
        if resolved_conflict_ids is None
        else resolved_conflict_ids
    )
    locator = screening_locator
    assignment_ids = tuple(row["assignment_id"] for row in ratings)
    triggers: list[str] = []
    if ratings[0]["screening_status"] != ratings[1]["screening_status"]:
        triggers.append("A1")
    if ratings[0]["criterion"] != ratings[1]["criterion"]:
        triggers.append("A2")
    if (
        ratings[0]["screening_status"] == "excluded"
        and ratings[1]["screening_status"] == "excluded"
        and ratings[0]["criterion"] == ratings[1]["criterion"]
        and integration._normalize_exclusion_reason(
            ratings[0]["exclusion_reason"]
        )
        != integration._normalize_exclusion_reason(
            ratings[1]["exclusion_reason"]
        )
    ):
        triggers.append("A3")
    if available_conflicts:
        triggers.append("A4")

    if screening_status == "excluded":
        deciding_fact = exclusion_reason
        deciding_fact_kind = "exclusion_reason"
    else:
        deciding_fact = transfer_source_fact or (
            f"The inspected {candidate_id} source defines generated course "
            "geometry and exposes a reusable course-generation procedure."
        )
        deciding_fact_kind = "transfer_source_fact"
    comparison_parts = [
        (
            f"For candidate {candidate_id}, the adjudicator examines how the "
            "inspected algorithm constructs geometry, exposes reusable "
            "implementation behavior, and links technical method details to "
            "course generation, whereas the locked reviews emphasize different "
            "source passages and contribution boundaries."
        )
    ]
    if "A1" in triggers:
        comparison_parts.append(
            f"The {ratings[0]['screening_status']} and "
            f"{ratings[1]['screening_status']} readings reach different "
            "eligibility outcomes after inspecting the same source method."
        )
    if "A2" in triggers:
        comparison_parts.append(
            f"The {ratings[0]['criterion']} and {ratings[1]['criterion']} "
            "classifications emphasize different technical contributions in "
            "the implementation and representation."
        )
    if "A3" in triggers:
        normalized_reasons = [
            integration._normalize_exclusion_reason(
                rating["exclusion_reason"]
            ).split()
            for rating in ratings
        ]
        unique_words = [
            next(
                word
                for word in words
                if len(word) >= 5
                and word.isalpha()
                and word not in set(normalized_reasons[1 - index])
            )
            for index, words in enumerate(normalized_reasons)
        ]
        comparison_parts.append(
            f"The first exclusion uniquely emphasizes {unique_words[0]}, "
            f"whereas the second uniquely emphasizes {unique_words[1]}; "
            "these independently worded source observations require comparison "
            "against the inspected fixed-geometry evidence."
        )
    if "A4" in triggers:
        comparison_parts.append(
            "The adjudicator traces unresolved conflict records "
            + ", ".join(selected_conflict_ids)
            + " against the frozen coordinator provenance and deciding source."
        )
    comparison = " ".join(comparison_parts)
    evidence = {
        "schema_version": "1",
        "raw_ratings": [
            {
                "assignment_id": rating["assignment_id"],
                "criterion": rating["criterion"],
                "screening_status": rating["screening_status"],
            }
            for rating in ratings
        ],
        "controlling_rules": triggers,
        "raw_exclusion_reasons": (
            [
                {
                    "assignment_id": rating["assignment_id"],
                    "reason": rating["exclusion_reason"],
                }
                for rating in ratings
            ]
            if "A3" in triggers
            else []
        ),
        "resolved_conflicts": [
            {
                "conflict_id": conflict_id,
                "field": available_by_id[conflict_id]["field"],
                "value_a": available_by_id[conflict_id]["value_a"],
                "value_b": available_by_id[conflict_id]["value_b"],
            }
            for conflict_id in selected_conflict_ids
            if conflict_id in available_by_id
        ],
        "final_decision": {
            "criterion": criterion,
            "screening_status": screening_status,
        },
        "deciding_fact": {
            "kind": deciding_fact_kind,
            "text": deciding_fact,
        },
        "source_url": source,
        "deciding_locator": locator,
        "comparison_analysis": comparison,
    }
    return {
        "candidate_id": candidate_id,
        "input_sha256": ratings[0]["input_sha256"],
        "snapshot_sha256": ratings[0]["snapshot_sha256"],
        "primary_snapshot_sha256": case.primary_hash,
        "assignment_ids": ";".join(assignment_ids),
        "adjudicator_id": "adjudicator-01",
        "reviewer_ids": ";".join(row["coder_id"] for row in ratings),
        "decided_on": "2026-06-30",
        "screening_status": screening_status,
        "criterion": criterion,
        "access_status": "full_text",
        "source_urls": f"{archive};{source}",
        "evidence_version": "version-of-record-1",
        "evidence_retrieved_on": "2026-06-29",
        "evidence_archive_url": archive,
        "evidence_sha256": hashlib.sha256(
            f"adjudication:{candidate_id}".encode("utf-8")
        ).hexdigest(),
        "screening_locator": locator,
        "exclusion_reason": exclusion_reason,
        "resolution_evidence": json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "resolved_conflict_ids": (
            ";".join(selected_conflict_ids)
            if selected_conflict_ids
            else "NR"
        ),
        "notes": "NR",
    }


def _execution_registry_rows(
    case: ScreeningCase,
    adjudications: list[dict[str, str]],
) -> list[dict[str, str]]:
    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def row_for(
        *,
        role_id: str,
        role_type: str,
        task: str,
        work_item_id: str,
        result_file_sha256: str,
    ) -> dict[str, str]:
        common = {
            "execution_id": f"exec-{role_id}-{task}",
            "role_id": role_id,
            "role_type": role_type,
            "context_id": f"context-{role_id}-{task}",
            "task": task,
            "work_item_id": work_item_id,
            "started_on": "2026-06-28",
            "completed_on": "2026-06-30",
            "result_file_sha256": result_file_sha256,
        }
        if role_type == "automated":
            return {
                **common,
                "model_identifier": "gpt-5",
                "model_version": "2026-06-01",
                "configuration_sha256": digest(
                    f"configuration:{role_id}:{task}"
                ),
                "prompt_sha256": digest(f"prompt:{role_id}:{task}"),
                "provider": "openai",
                "runtime": "codex-1.0",
                "tool_configuration": (
                    '{"browser":"enabled","filesystem":"read-only"}'
                ),
                "retrieval_configuration": (
                    '{"cache":"isolated","sources":"public"}'
                ),
                "decoding_parameters": '{"temperature":0,"top_p":1}',
                "system_instruction_sha256": digest(
                    f"system:{role_id}:{task}"
                ),
                "developer_instruction_sha256": digest(
                    f"developer:{role_id}:{task}"
                ),
                "user_instruction_sha256": digest(
                    f"user:{role_id}:{task}"
                ),
                "cache_isolation_statement": (
                    "Fresh context; no shared conversation history, memory, "
                    "ratings, results, or retrieval cache."
                ),
                "human_role": "NR",
                "training_calibration_exposure": "NR",
                "automated_actions": "NR",
            }
        return {
            **common,
            "model_identifier": "NR",
            "model_version": "NR",
            "configuration_sha256": "NR",
            "prompt_sha256": "NR",
            "provider": "NR",
            "runtime": "NR",
            "tool_configuration": "NR",
            "retrieval_configuration": "NR",
            "decoding_parameters": "NR",
            "system_instruction_sha256": "NR",
            "developer_instruction_sha256": "NR",
            "user_instruction_sha256": "NR",
            "cache_isolation_statement": "NR",
            "human_role": "eligibility-adjudicator",
            "training_calibration_exposure": (
                "completed protocol training and calibration review"
            ),
            "automated_actions": "none",
        }

    rows: list[dict[str, str]] = []
    for rating in case.ratings:
        task = (
            "calibration-screening"
            if rating["phase"] == "calibration"
            else "main-screening"
        )
        snapshot = (
            case.calibration
            if rating["phase"] == "calibration"
            else case.main
        )
        result_digest = hashlib.sha256(
            (snapshot / f"{rating['batch_id']}.csv").read_bytes()
        ).hexdigest()
        rows.append(
            row_for(
                role_id=rating["coder_id"],
                role_type="automated",
                task=task,
                work_item_id=rating["assignment_id"],
                result_file_sha256=result_digest,
            )
        )
    adjudication_digest = hashlib.sha256(
        _csv_bytes(integration.ADJUDICATION_HEADER, adjudications)
    ).hexdigest()
    for adjudication in adjudications:
        rows.append(
            row_for(
                role_id=adjudication["adjudicator_id"],
                role_type="human",
                task="adjudication",
                work_item_id=adjudication["candidate_id"],
                result_file_sha256=adjudication_digest,
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            row["task"].encode("utf-8"),
            row["work_item_id"].encode("utf-8"),
            row["role_id"].encode("utf-8"),
        ),
    )

def _use_limited_provider_provenance(
    rows: list[dict[str, str]],
) -> None:
    limitations = {
        "backend_model_version": "provider-not-exposed",
        "decoding_parameters": "provider-not-exposed",
        "developer_instruction_bytes": "provider-not-exposed",
        "retrieval_cache_isolation": "provider-not-exposed",
        "system_instruction_bytes": "provider-not-exposed",
    }
    for row in rows:
        if row["role_type"] != "automated":
            continue
        tool_configuration = json.loads(row["tool_configuration"])
        tool_configuration["provider_metadata_limitations"] = limitations
        row["tool_configuration"] = json.dumps(
            tool_configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        row["model_version"] = "requested:gpt-5-2026-06-30"
        row["decoding_parameters"] = "NR"
        row["system_instruction_sha256"] = "NR"
        row["developer_instruction_sha256"] = "NR"
        row["cache_isolation_statement"] = (
            LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT
        )

def _execution_provenance_row() -> dict[str, str]:
    digest = "a" * 64
    return {
        "role_type": "automated",
        "model_identifier": "gpt-5",
        "model_version": "2026-06-01",
        "configuration_sha256": digest,
        "prompt_sha256": digest,
        "provider": "openai",
        "runtime": "codex-1.0",
        "tool_configuration": '{"browser":"enabled"}',
        "retrieval_configuration": '{"sources":"public"}',
        "decoding_parameters": '{"temperature":0}',
        "system_instruction_sha256": digest,
        "developer_instruction_sha256": digest,
        "user_instruction_sha256": digest,
        "cache_isolation_statement": (
            "Fresh context; no shared conversation history, memory, ratings, "
            "results, or retrieval cache."
        ),
        "human_role": "NR",
        "training_calibration_exposure": "NR",
        "automated_actions": "NR",
    }


def _set_provider_metadata_limitations(
    row: dict[str, str],
    limitations: object | None,
) -> None:
    tool_configuration = json.loads(row["tool_configuration"])
    if limitations is None:
        tool_configuration.pop("provider_metadata_limitations", None)
    else:
        tool_configuration["provider_metadata_limitations"] = limitations
    row["tool_configuration"] = json.dumps(
        tool_configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _limited_execution_provenance_row() -> dict[str, str]:
    row = _execution_provenance_row()
    _set_provider_metadata_limitations(
        row,
        {
            "backend_model_version": "provider-not-exposed",
            "decoding_parameters": "provider-not-exposed",
            "developer_instruction_bytes": "provider-not-exposed",
            "retrieval_cache_isolation": "provider-not-exposed",
            "system_instruction_bytes": "provider-not-exposed",
        },
    )
    row["model_version"] = "requested:gpt-5-2026-06-30"
    row["decoding_parameters"] = "NR"
    row["system_instruction_sha256"] = "NR"
    row["developer_instruction_sha256"] = "NR"
    row["cache_isolation_statement"] = (
        LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT
    )
    return row



def _write_execution_register(
    case: ScreeningCase,
    adjudications: list[dict[str, str]],
    *,
    name: str = "execution-register.csv",
    rows: list[dict[str, str]] | None = None,
) -> Path:
    path = case.root / name
    _write_csv(
        path,
        EXECUTION_REGISTER_HEADER,
        rows if rows is not None else _execution_registry_rows(
            case, adjudications
        ),
    )
    return path


def _trigger_case(tmp_path: Path) -> tuple[ScreeningCase, list[dict[str, str]]]:
    case = _build_case(
        tmp_path,
        scenarios=_trigger_scenarios(),
        conflicts=[_conflict("X57B57E64E501", "C0143")],
    )
    rows = [
        _adjudication_row(case, "C0001"),
        _adjudication_row(case, "C0002", criterion="include-2"),
        _adjudication_row(
            case,
            "C0003",
            screening_status="excluded",
            criterion="exclude-traffic-only",
            exclusion_reason=(
                "The report only generates traffic participants on fixed road "
                "geometry and contributes no course-generation method."
            ),
        ),
        _adjudication_row(case, "C0143"),
    ]
    return case, rows


def test_unresolved_conflict_retains_discovery_provenance(
    tmp_path: Path,
) -> None:
    conflict = {
        **_conflict("X57B57E64E501", "C0143"),
        "resolution_evidence": (
            "value_a=paper/data/agent_runs/blind-ground.csv#BG-022; "
            "value_b=paper/data/candidates.csv#C0143"
        ),
    }
    case = _build_case(tmp_path, conflicts=[conflict])
    context = integration._load_context(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
    )

    unresolved = integration._unresolved_screening_conflicts(context)

    assert tuple(unresolved) == ("C0143",)
    assert unresolved["C0143"][0]["conflict_id"] == "X57B57E64E501"


def _seal_adjudications(
    case: ScreeningCase,
    rows: list[dict[str, str]],
    *,
    version: str = "v1",
    execution_rows: list[dict[str, str]] | None = None,
) -> Path:
    input_path = case.root / f"adjudication-input-{version}.csv"
    _write_csv(input_path, integration.ADJUDICATION_HEADER, rows)
    execution_register = _write_execution_register(
        case,
        rows,
        name=f"execution-register-{version}.csv",
        rows=execution_rows,
    )
    case.execution_register_path = execution_register
    output = case.root / "adjudications" / version
    output.parent.mkdir(parents=True, exist_ok=True)
    integration.seal_adjudication_results(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        input_path,
        execution_register,
        output,
    )
    return output


def _case_execution_register(case: ScreeningCase) -> Path:
    assert case.execution_register_path is not None
    return case.execution_register_path


def _write_author_verification(
    case: ScreeningCase,
    adjudications: Path,
    *,
    execution_register: Path | None = None,
    citation_keys: Path | None = None,
    name: str = "author-verification.csv",
) -> Path:
    register = execution_register or _case_execution_register(case)
    ledger = citation_keys or case.citation_keys
    result = integration.integrate_screening(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        adjudications,
        register,
        ledger,
    )
    bindings = integration._author_decision_bindings(result)
    rows = [
        {
            "candidate_id": candidate_id,
            "primary_snapshot_sha256": result.primary_snapshot_sha256,
            "adjudication_snapshot_sha256": (
                result.adjudication_snapshot_sha256
            ),
            **bindings[candidate_id],
            "verified_by": "author-01",
            "verified_role": "accountable-author",
            "verified_on": "2026-06-30",
            "verification_status": "verified",
            "verification_evidence": (
                f"Accountable author verified {candidate_id} status, criterion, "
                "provenance, rationale, duplicate grouping, and every deciding "
                "locator directly against the cited source."
            ),
        }
        for candidate_id in sorted(bindings, key=lambda value: value.encode("utf-8"))
    ]
    path = case.root / name
    _write_csv(path, integration.AUTHOR_VERIFICATION_HEADER, rows)
    return path


def _seal_projection(
    case: ScreeningCase,
    adjudications: Path,
    output: Path,
    *,
    execution_register: Path | None = None,
    citation_keys: Path | None = None,
    author_verification: Path | None = None,
) -> Path:
    register = execution_register or _case_execution_register(case)
    ledger = citation_keys or case.citation_keys
    signoff = author_verification or _write_author_verification(
        case,
        adjudications,
        execution_register=register,
        citation_keys=ledger,
        name=f"author-verification-{output.parent.name}-{output.name}.csv",
    )
    integration.seal_screening_projection(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        adjudications,
        register,
        ledger,
        signoff,
        output,
    )
    return signoff


def _replace_with_identical_copy(
    path: Path,
    scratch: Path,
) -> None:
    clone = scratch / "clone" / path.name
    displaced = scratch / "displaced" / path.name
    clone.parent.mkdir(parents=True, exist_ok=True)
    displaced.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, clone)
        path.rename(displaced)
        clone.rename(path)
    else:
        shutil.copy2(path, clone)
        os.replace(clone, path)


def test_mutable_input_rejects_relative_protected_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    mutable_input = protected / "execution-register.csv"
    mutable_input.write_bytes(b"captured input\n")
    captured = screening_results.capture_input(
        mutable_input,
        "execution register",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="mutable input overlaps an immutable snapshot",
    ):
        integration._reject_captured_input_aliases(
            (captured,),
            (Path("protected"),),
        )


def test_mutable_input_rejects_protected_snapshot_file_identity(
    tmp_path: Path,
) -> None:
    mutable_input = tmp_path / "execution-register.csv"
    mutable_input.write_bytes(b"captured input\n")
    captured = screening_results.capture_input(
        mutable_input,
        "execution register",
    )
    protected = screening_results.FileFingerprint(
        path=tmp_path / "immutable-snapshot" / "artifact.csv",
        identity=captured.fingerprint.identity,
        sha256=captured.fingerprint.sha256,
        size=captured.fingerprint.size,
        mode=captured.fingerprint.mode,
        link_count=captured.fingerprint.link_count,
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="aliases an immutable snapshot file",
    ):
        integration._reject_captured_input_aliases(
            (captured,),
            (),
            protected_fingerprints=(protected,),
        )


def _captured_input_with_path(
    captured: screening_results.CapturedInput,
    path: Path,
) -> screening_results.CapturedInput:
    fingerprint = captured.fingerprint
    return screening_results.CapturedInput(
        fingerprint=screening_results.FileFingerprint(
            path=path,
            identity=fingerprint.identity,
            sha256=fingerprint.sha256,
            size=fingerprint.size,
            mode=fingerprint.mode,
            link_count=fingerprint.link_count,
            tree=fingerprint.tree,
        ),
        payload=captured.payload,
    )


@pytest.mark.parametrize(
    "label",
    [
        "execution register",
        "citation key ledger",
        "adjudication result",
        "author verification",
    ],
)
@pytest.mark.parametrize(
    "relation",
    [
        "exact",
        "mutable-ancestor",
        "mutable-descendant",
        "relative",
        "symlink",
    ],
)
def test_mutable_input_overlap_matrix_rejects_path_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    relation: str,
) -> None:
    source = tmp_path / f"{label.replace(' ', '-')}.csv"
    source.write_bytes(b"captured input\n")
    captured = screening_results.capture_input(source, label)
    anchor = tmp_path / "immutable-anchor"

    if relation == "exact":
        mutable_path = anchor
        protected = anchor
    elif relation == "mutable-ancestor":
        mutable_path = anchor
        protected = anchor / "snapshot"
    elif relation == "mutable-descendant":
        mutable_path = anchor / source.name
        protected = anchor
    elif relation == "relative":
        mutable_path = anchor / source.name
        protected = Path("immutable-anchor")
        monkeypatch.chdir(tmp_path)
    else:
        anchor.mkdir()
        alias = tmp_path / "immutable-alias"
        alias.symlink_to(anchor, target_is_directory=True)
        mutable_path = anchor / source.name
        protected = alias

    aliased = _captured_input_with_path(captured, mutable_path)
    output = tmp_path / "unpublished-output"

    with pytest.raises(integration.ScreeningIntegrationError):
        integration._reject_captured_input_aliases(
            (aliased,),
            (protected,),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "label",
    [
        "execution register",
        "citation key ledger",
        "adjudication result",
        "author verification",
    ],
)
def test_mutable_input_overlap_matrix_rejects_hard_links(
    tmp_path: Path,
    label: str,
) -> None:
    source = tmp_path / "immutable.csv"
    source.write_bytes(b"immutable input\n")
    alias = tmp_path / f"{label.replace(' ', '-')}.csv"
    os.link(source, alias)
    output = tmp_path / "unpublished-output"

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="hard link",
    ):
        integration._call(
            screening_results.capture_input,
            alias,
            label,
        )

    assert not output.exists()


def test_adjudication_output_guard_covers_every_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = object()
    protected = tuple(
        tmp_path / "immutable" / name
        for name in (
            "coordinator",
            "calibration-release",
            "calibration-result",
            "calibration-decision",
            "main-release",
            "main-result",
        )
    )
    adjudication_input = tmp_path / "mutable" / "adjudications.csv"
    execution_register = tmp_path / "mutable" / "execution-register.csv"
    output = tmp_path / "unpublished-adjudication"

    monkeypatch.setattr(integration, "_load_context", lambda *args: context)
    monkeypatch.setattr(
        integration,
        "_context_protected_paths",
        lambda captured_context: protected,
    )
    monkeypatch.setattr(
        integration,
        "_context_protected_fingerprints",
        lambda captured_context: (),
    )

    def reject_output_overlap(
        output_dir: Path,
        protected_inputs: tuple[Path, ...],
    ) -> None:
        assert Path(output_dir) == output
        assert tuple(protected_inputs) == (
            *protected,
            adjudication_input,
            execution_register,
        )
        raise screening_results.ScreeningResultError("sentinel overlap")

    monkeypatch.setattr(
        screening_results,
        "reject_output_overlap",
        reject_output_overlap,
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="sentinel overlap",
    ):
        integration.seal_adjudication_results(
            *protected,
            adjudication_input,
            execution_register,
            output,
        )

    assert not output.exists()


def test_projection_output_guard_covers_every_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutable = tmp_path / "mutable"
    mutable.mkdir()
    execution_path = mutable / "execution-register.csv"
    ledger_path = mutable / "citation-keys.csv"
    author_path = mutable / "author-verification.csv"
    for path in (execution_path, ledger_path, author_path):
        path.write_bytes(f"{path.name}\n".encode("utf-8"))

    execution_register = screening_results.capture_input(
        execution_path,
        "execution register",
    )
    citation_key_ledger = screening_results.capture_input(
        ledger_path,
        "citation key ledger",
    )
    context = object()

    class AdjudicationStub:
        directory = tmp_path / "immutable" / "adjudication"
        fingerprints: tuple[screening_results.FileFingerprint, ...] = ()

    captured = integration._CapturedIntegrationInputs(
        context=context,
        adjudication=AdjudicationStub(),
        execution_register=execution_register,
        citation_key_ledger=citation_key_ledger,
    )
    protected = tuple(
        tmp_path / "immutable" / name
        for name in (
            "coordinator",
            "calibration-release",
            "calibration-result",
            "calibration-decision",
            "main-release",
            "main-result",
        )
    )
    output = tmp_path / "unpublished-projection"

    monkeypatch.setattr(
        integration,
        "_capture_integration_inputs",
        lambda *args: captured,
    )
    monkeypatch.setattr(
        integration,
        "_context_protected_paths",
        lambda captured_context: protected,
    )
    monkeypatch.setattr(
        integration,
        "_context_protected_fingerprints",
        lambda captured_context: (),
    )

    def reject_output_overlap(
        output_dir: Path,
        protected_inputs: tuple[Path, ...],
    ) -> None:
        assert Path(output_dir) == output
        assert tuple(protected_inputs) == (
            *protected,
            AdjudicationStub.directory,
            execution_register.fingerprint.path,
            citation_key_ledger.fingerprint.path,
            author_path.resolve(),
        )
        raise screening_results.ScreeningResultError("sentinel overlap")

    monkeypatch.setattr(
        screening_results,
        "reject_output_overlap",
        reject_output_overlap,
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="sentinel overlap",
    ):
        integration.seal_screening_projection(
            *protected,
            AdjudicationStub.directory,
            execution_path,
            ledger_path,
            author_path,
            output,
        )

    assert not output.exists()



def test_screening_context_canonicalizes_relative_snapshot_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path)
    monkeypatch.chdir(case.root)

    context = integration._load_context(
        case.coordinator.relative_to(case.root),
        case.calibration_release.relative_to(case.root),
        case.calibration.relative_to(case.root),
        case.calibration_decision.relative_to(case.root),
        case.main_release.relative_to(case.root),
        case.main.relative_to(case.root),
    )

    assert (
        context.coordinator_dir,
        context.calibration_release_dir,
        context.calibration_result_dir,
        context.calibration_decision_dir,
        context.main_release_dir,
        context.main_result_dir,
    ) == (
        case.coordinator.resolve(),
        case.calibration_release.resolve(),
        case.calibration.resolve(),
        case.calibration_decision.resolve(),
        case.main_release.resolve(),
        case.main.resolve(),
    )


def test_sealing_rejects_release_aliases_for_every_mutable_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, rows = _trigger_case(tmp_path)
    adjudication_snapshot = _seal_adjudications(case, rows)
    execution_register = _case_execution_register(case)
    adjudication_input = case.root / "adjudication-input-v1.csv"
    author_verification = _write_author_verification(
        case,
        adjudication_snapshot,
    )
    main_release_file = next(
        path
        for path in sorted(case.main_release.rglob("*"))
        if path.is_file()
    )

    def assert_unpublished(
        output_name: str,
        operation: Callable[[Path], None],
    ) -> None:
        output = case.root / "rejected-aliases" / output_name / "v1"
        output.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(integration.ScreeningIntegrationError):
            operation(output)
        assert not output.exists()

    monkeypatch.chdir(case.root)
    relative_release_file = main_release_file.relative_to(case.root)
    assert_unpublished(
        "relative-execution-register",
        lambda output: integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            relative_release_file,
            output,
        ),
    )
    assert_unpublished(
        "exact-adjudication",
        lambda output: integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            main_release_file,
            execution_register,
            output,
        ),
    )

    ledger_alias = case.root / "ledger-symlink.csv"
    ledger_alias.symlink_to(main_release_file)
    assert_unpublished(
        "symlink-ledger",
        lambda output: integration.seal_screening_projection(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_snapshot,
            execution_register,
            ledger_alias,
            author_verification,
            output,
        ),
    )

    adjudication_descendant = (
        case.calibration_release / "nested-adjudication-output"
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            execution_register,
            adjudication_descendant,
        )
    assert not adjudication_descendant.exists()

    projection_descendant = case.main_release / "nested-projection-output"
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_screening_projection(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_snapshot,
            execution_register,
            case.citation_keys,
            author_verification,
            projection_descendant,
        )
    assert not projection_descendant.exists()

    main_release_parent = case.main_release.parent
    main_parent_entries = tuple(sorted(main_release_parent.iterdir()))
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            execution_register,
            main_release_parent,
        )
    assert tuple(sorted(main_release_parent.iterdir())) == main_parent_entries

    calibration_release_parent = case.calibration_release.parent
    calibration_parent_entries = tuple(
        sorted(calibration_release_parent.iterdir())
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_screening_projection(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_snapshot,
            execution_register,
            case.citation_keys,
            author_verification,
            calibration_release_parent,
        )
    assert (
        tuple(sorted(calibration_release_parent.iterdir()))
        == calibration_parent_entries
    )

    adjudication_input_bytes = adjudication_input.read_bytes()
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            execution_register,
            adjudication_input,
        )
    assert adjudication_input.read_bytes() == adjudication_input_bytes

    author_verification_bytes = author_verification.read_bytes()
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_screening_projection(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_snapshot,
            execution_register,
            case.citation_keys,
            author_verification,
            author_verification,
        )
    assert author_verification.read_bytes() == author_verification_bytes

    calibration_release_file = next(
        path
        for path in sorted(case.calibration_release.rglob("*"))
        if path.is_file()
    )
    author_alias = case.root / "author-hard-link.csv"
    os.link(calibration_release_file, author_alias)
    assert_unpublished(
        "hard-link-author",
        lambda output: integration.seal_screening_projection(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_snapshot,
            execution_register,
            case.citation_keys,
            author_alias,
            output,
        ),
    )


def test_public_contract_uses_protocol_schema_and_has_no_legacy_loose_api() -> None:
    assert integration.EXECUTION_REGISTER_HEADER == EXECUTION_REGISTER_HEADER
    assert integration.ADJUDICATION_HEADER == (
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
    assert integration.SCREENING_RESULT_HEADER == screening_results.RESULT_HEADER
    assert (
        integration.SCREENING_AGREEMENT_HEADER
        == screening_agreement.AGREEMENT_REPORT_HEADER
    )
    for legacy_name in (
        "seal_primary_results",
        "seal_adjudication_result",
        "write_integration_outputs",
        "write_screening_outputs",
    ):
        assert not hasattr(integration, legacy_name)


def test_execution_register_is_embedded_and_manifest_bound(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    adjudication_input = tmp_path / "adjudications.csv"
    _write_csv(adjudication_input, integration.ADJUDICATION_HEADER, rows)
    execution_register = _write_execution_register(case, rows)
    output = tmp_path / "adjudication-registry" / "v1"
    output.parent.mkdir(parents=True)

    integration.seal_adjudication_results(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        adjudication_input,
        execution_register,
        output,
    )

    assert {path.name for path in output.iterdir()} == {
        "adjudications.csv",
        "execution_registry.csv",
        "manifest.csv",
        "SHA256SUMS",
    }
    assert (
        (output / "execution_registry.csv").read_bytes()
        == execution_register.read_bytes()
    )
    _, manifest_rows = _read_csv(output / "manifest.csv")
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["execution_registry_sha256"] == hashlib.sha256(
        execution_register.read_bytes()
    ).hexdigest()
    assert manifest_rows[0]["execution_row_count"] == str(404 + len(rows))
    captured = integration.validate_adjudication_snapshot(
        output,
        coordinator_snapshot_dir=case.coordinator,
        calibration_reviewer_release_snapshot_dir=case.calibration_release,
        calibration_result_snapshot_dir=case.calibration,
        calibration_decision_snapshot_dir=case.calibration_decision,
        main_reviewer_release_snapshot_dir=case.main_release,
        main_result_snapshot_dir=case.main,
        execution_register_path=execution_register,
    )
    assert len(captured.execution_registry) == 404 + len(rows)


def test_execution_provenance_preserves_full_provider_rows() -> None:
    integration._validate_execution_provenance(
        _execution_provenance_row(),
        context_label="full-provider-row",
    )


@pytest.mark.parametrize(
    ("model_version", "limitations"),
    [
        ("requested:gpt-5-2026-06-30", None),
        (
            "2026-06-01",
            {"backend_model_version": "provider-not-exposed"},
        ),
        (
            "requested:",
            {"backend_model_version": "provider-not-exposed"},
        ),
    ],
    ids=("missing", "unjustified", "mismatch"),
)
def test_backend_model_limitation_requires_requested_version_sentinel(
    model_version: str,
    limitations: dict[str, str] | None,
) -> None:
    row = _execution_provenance_row()
    row["model_version"] = model_version
    _set_provider_metadata_limitations(row, limitations)

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="backend_model_version",
    ):
        integration._validate_execution_provenance(
            row,
            context_label="backend-model-version",
        )


@pytest.mark.parametrize(
    "field",
    [
        "configuration_sha256",
        "prompt_sha256",
        "user_instruction_sha256",
    ],
)
def test_limited_provider_keeps_unconditional_hashes_mandatory(
    field: str,
) -> None:
    row = _limited_execution_provenance_row()
    row[field] = "NR"

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match=field,
    ):
        integration._validate_execution_provenance(
            row,
            context_label="limited-provider-row",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "non-object",
        "empty",
        "unknown-key",
        "wrong-value",
        "system-missing",
        "system-unjustified",
        "developer-missing",
        "developer-unjustified",
        "decoding-missing",
        "decoding-unjustified",
        "retrieval-missing",
        "retrieval-unjustified",
        "mismatched-field",
    ],
)
def test_provider_metadata_limitations_reject_bad_combinations(
    mutation: str,
) -> None:
    row = _execution_provenance_row()
    limitations: object = {}
    if mutation == "non-object":
        limitations = "provider-not-exposed"
    elif mutation == "empty":
        limitations = {}
    elif mutation == "unknown-key":
        limitations = {"unknown_bytes": "provider-not-exposed"}
    elif mutation == "wrong-value":
        limitations = {"system_instruction_bytes": "not-recorded"}
    elif mutation == "system-missing":
        row["system_instruction_sha256"] = "NR"
        limitations = {}
    elif mutation == "system-unjustified":
        limitations = {
            "system_instruction_bytes": "provider-not-exposed"
        }
    elif mutation == "developer-missing":
        row["developer_instruction_sha256"] = "NR"
        limitations = {}
    elif mutation == "developer-unjustified":
        limitations = {
            "developer_instruction_bytes": "provider-not-exposed"
        }
    elif mutation == "decoding-missing":
        row["decoding_parameters"] = "NR"
        limitations = {}
    elif mutation == "decoding-unjustified":
        limitations = {"decoding_parameters": "provider-not-exposed"}
    elif mutation == "retrieval-missing":
        row[
            "cache_isolation_statement"
        ] = LIMITED_PROVIDER_CACHE_ISOLATION_STATEMENT
        limitations = {}
    elif mutation == "retrieval-unjustified":
        limitations = {
            "retrieval_cache_isolation": "provider-not-exposed"
        }
    elif mutation == "mismatched-field":
        row["system_instruction_sha256"] = "NR"
        limitations = {
            "developer_instruction_bytes": "provider-not-exposed"
        }
    _set_provider_metadata_limitations(row, limitations)

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="provider_metadata_limitations",
    ):
        integration._validate_execution_provenance(
            row,
            context_label=f"bad-limitation-{mutation}",
        )

def test_execution_register_accepts_limited_provider_provenance(
    tmp_path: Path,
) -> None:
    case, adjudications = _trigger_case(tmp_path)
    rows = _execution_registry_rows(case, adjudications)
    _use_limited_provider_provenance(rows)

    automated = next(
        row for row in rows if row["role_type"] == "automated"
    )
    for field in (
        "configuration_sha256",
        "prompt_sha256",
        "user_instruction_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", automated[field])

    snapshot = _seal_adjudications(
        case,
        adjudications,
        version="v10",
        execution_rows=rows,
    )
    assert snapshot.is_dir()

@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "role",
        "unstable-execution",
        "unstable-context",
        "shared-reviewer-context",
        "adjudicator-context",
        "invalid-role-type",
        "missing-model",
        "missing-instruction",
        "invalid-tool-configuration",
        "weak-cache-isolation",
        "contradictory-cache-isolation",
        "result-digest",
        "human-exposure",
        "human-actions",
        "invalid-date",
        "started-after-completed",
        "invalid-id",
    ],
)
def test_execution_register_identity_and_context_contract_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    case, adjudications = _trigger_case(tmp_path)
    rows = _execution_registry_rows(case, adjudications)
    by_work_item = {row["work_item_id"]: row for row in rows}
    automated = next(row for row in rows if row["role_type"] == "automated")
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(
            {
                **rows[0],
                "work_item_id": "A-C9999-01",
            }
        )
    elif mutation == "role":
        by_work_item[case.ratings[0]["assignment_id"]][
            "role_id"
        ] = "screening-99"
    elif mutation in {"unstable-execution", "unstable-context"}:
        role = rows[0]["role_id"]
        task = rows[0]["task"]
        peer = next(
            row
            for row in rows[1:]
            if row["role_id"] == role and row["task"] == task
        )
        field = (
            "execution_id"
            if mutation == "unstable-execution"
            else "context_id"
        )
        peer[field] = f"{field}-drifted"
    elif mutation == "shared-reviewer-context":
        pair = case.ratings_for("C0001")
        first = by_work_item[pair[0]["assignment_id"]]
        second = by_work_item[pair[1]["assignment_id"]]
        second["context_id"] = first["context_id"]
    elif mutation == "adjudicator-context":
        pair = case.ratings_for("C0001")
        reviewer = by_work_item[pair[0]["assignment_id"]]
        by_work_item["C0001"]["context_id"] = reviewer["context_id"]
    elif mutation == "invalid-role-type":
        rows[0]["role_type"] = "agent"
    elif mutation == "missing-model":
        automated["model_identifier"] = "NR"
    elif mutation == "missing-instruction":
        automated["developer_instruction_sha256"] = "NR"
    elif mutation == "invalid-tool-configuration":
        automated["tool_configuration"] = '{"browser": true}'
    elif mutation == "weak-cache-isolation":
        automated["cache_isolation_statement"] = "fresh context"
    elif mutation == "contradictory-cache-isolation":
        for row in rows:
            if row["role_type"] == "automated":
                row["cache_isolation_statement"] = (
                    "Fresh context; no shared conversation history, memory, "
                    "ratings, results, or retrieval cache. Shared retrieval "
                    "cache is nevertheless permitted."
                )
    elif mutation == "result-digest":
        rows[0]["result_file_sha256"] = "0" * 64
    elif mutation == "human-exposure":
        by_work_item["C0001"]["training_calibration_exposure"] = "NR"
    elif mutation == "human-actions":
        by_work_item["C0001"]["automated_actions"] = "NR"
    elif mutation == "invalid-date":
        rows[0]["completed_on"] = "2026-02-30"
    elif mutation == "started-after-completed":
        rows[0]["started_on"] = "2026-07-01"
    elif mutation == "invalid-id":
        rows[0]["context_id"] = "bad context"

    rows.sort(
        key=lambda row: (
            row["task"].encode("utf-8"),
            row["work_item_id"].encode("utf-8"),
            row["role_id"].encode("utf-8"),
        )
    )
    adjudication_input = tmp_path / "adjudications.csv"
    _write_csv(
        adjudication_input,
        integration.ADJUDICATION_HEADER,
        adjudications,
    )
    execution_register = _write_execution_register(
        case,
        adjudications,
        rows=rows,
    )
    output = tmp_path / "invalid-registry" / "v1"
    output.parent.mkdir(parents=True)

    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            execution_register,
            output,
        )
    assert not output.exists()


def test_cache_isolation_accepts_only_nfkc_whitespace_case_equivalence(
    tmp_path: Path,
) -> None:
    case, adjudications = _trigger_case(tmp_path)
    rows = _execution_registry_rows(case, adjudications)
    for row in rows:
        if row["role_type"] == "automated":
            row["cache_isolation_statement"] = (
                "Ｆｒｅｓｈ   context； no shared conversation history, "
                "memory, ratings, results, or retrieval cache．"
            )
    snapshot = _seal_adjudications(
        case,
        adjudications,
        version="v3",
        execution_rows=rows,
    )
    assert snapshot.is_dir()


@pytest.mark.parametrize(
    "field",
    [
        "tool_configuration",
        "retrieval_configuration",
        "decoding_parameters",
    ],
)
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_execution_register_rejects_nonstandard_json_constants(
    tmp_path: Path,
    field: str,
    constant: str,
) -> None:
    case, adjudications = _trigger_case(tmp_path)
    rows = _execution_registry_rows(case, adjudications)
    automated = next(
        row for row in rows if row["role_type"] == "automated"
    )
    for row in rows:
        if (
            row["role_id"] == automated["role_id"]
            and row["task"] == automated["task"]
        ):
            row[field] = f'{{"value":{constant}}}'

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="canonical JSON",
    ):
        _seal_adjudications(
            case,
            adjudications,
            version="v9",
            execution_rows=rows,
        )


def test_distinct_coder_ids_cannot_mask_one_human_identity(
    tmp_path: Path,
) -> None:
    case, adjudications = _trigger_case(tmp_path)
    rows = _execution_registry_rows(case, adjudications)
    reviewer_roles = {
        rating["coder_id"] for rating in case.ratings_for("C0001")
    }
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
    for row in rows:
        if row["role_id"] not in reviewer_roles:
            continue
        row["role_type"] = "human"
        for field in automated_fields:
            row[field] = "NR"
        row["human_role"] = "same-human-reviewer"
        row["training_calibration_exposure"] = (
            "completed protocol training before independent review"
        )
        row["automated_actions"] = "none"

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="human reviewer identities must be distinct",
    ):
        _seal_adjudications(
            case,
            adjudications,
            version="same-human",
            execution_rows=rows,
        )


def test_hybrid_execution_roles_bind_both_provenance_classes(
    tmp_path: Path,
) -> None:
    case, adjudications = _trigger_case(tmp_path)
    rows = _execution_registry_rows(case, adjudications)
    digest = hashlib.sha256
    for row in rows:
        if row["task"] != "adjudication":
            continue
        row.update(
            role_type="hybrid",
            model_identifier="gpt-5",
            model_version="2026-06-01",
            configuration_sha256=digest(b"hybrid-config").hexdigest(),
            prompt_sha256=digest(b"hybrid-prompt").hexdigest(),
            provider="openai",
            runtime="codex-1.0",
            tool_configuration=(
                '{"browser":"enabled","filesystem":"read-only"}'
            ),
            retrieval_configuration=(
                '{"cache":"isolated","sources":"public"}'
            ),
            decoding_parameters='{"temperature":0}',
            system_instruction_sha256=digest(b"hybrid-system").hexdigest(),
            developer_instruction_sha256=digest(
                b"hybrid-developer"
            ).hexdigest(),
            user_instruction_sha256=digest(b"hybrid-user").hexdigest(),
            cache_isolation_statement=(
                "Fresh context; no shared conversation history, memory, "
                "ratings, results, or retrieval cache."
            ),
            automated_actions="source retrieval and evidence comparison",
        )
    snapshot = _seal_adjudications(
        case,
        adjudications,
        version="v2",
        execution_rows=rows,
    )
    captured = integration.validate_adjudication_snapshot(
        snapshot,
        coordinator_snapshot_dir=case.coordinator,
        calibration_reviewer_release_snapshot_dir=case.calibration_release,
        calibration_result_snapshot_dir=case.calibration,
        calibration_decision_snapshot_dir=case.calibration_decision,
        main_reviewer_release_snapshot_dir=case.main_release,
        main_result_snapshot_dir=case.main,
        execution_register_path=_case_execution_register(case),
    )
    assert {
        row["role_type"]
        for row in captured.execution_registry
        if row["task"] == "adjudication"
    } == {"hybrid"}


def test_full_202_phase_integration_preserves_raw_ratings_and_provenance(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    adjudications = _seal_adjudications(case, rows)

    result = integration.integrate_screening(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        adjudications,
        _case_execution_register(case),
        case.citation_keys,
    )

    assert len(result.candidates) == 202
    assert len(result.screening_decisions) == 404
    assert len(result.screening_agreement) == 2
    assert result.primary_snapshot_sha256 == case.primary_hash
    assert {
        row["scope"] for row in result.screening_agreement
    } == {"calibration", "full_corpus"}
    for row in result.screening_agreement:
        assert (
            row["calibration_result_snapshot_sha256"]
            == case.calibration_hash
        )
        assert row["main_result_snapshot_sha256"] == case.main_hash
        assert row["primary_result_snapshot_sha256"] == case.primary_hash

    candidates = {
        row["candidate_id"]: row for row in result.candidates
    }
    assert candidates["C0001"]["screening_status"] == "included"
    assert candidates["C0002"]["screening_status"] == "included"
    assert candidates["C0003"]["screening_status"] == "excluded"
    assert candidates["C0004"]["screening_status"] == "excluded"
    assert candidates["C0004"]["cite_key"] == ""
    assert any(
        row["candidate_id"] == "C0004"
        and row["cite_key"] == "Author2026C0004"
        for row in result.citation_keys
    )
    assert candidates["C0143"]["screening_status"] == "included"

    conflict = next(
        row
        for row in result.conflicts
        if row["conflict_id"] == "X57B57E64E501"
    )
    assert conflict["resolution"] == "included"
    assert conflict["resolver"] == "adjudicator-01"

    c0001 = [
        row
        for row in result.screening_decisions
        if row["candidate_id"] == "C0001"
    ]
    assert {
        row["screening_status"] for row in c0001
    } == {"included", "excluded"}
    assert all(row["adjudicated"] == "yes" for row in c0001)
    for decision in c0001:
        for field in integration.ADJUDICATION_HEADER:
            assert decision[f"adjudication_{field}"] == rows[0][field]


def test_phase_snapshots_are_exactly_60_plus_344_and_not_interchangeable(
    tmp_path: Path,
) -> None:
    case = _build_case(tmp_path)
    calibration = screening_results.validate_phase_result_snapshot(
        case.calibration,
        coordinator_snapshot_dir=case.coordinator,
        reviewer_release_snapshot_dir=case.calibration_release,
    )
    main = screening_results.validate_phase_result_snapshot(
        case.main,
        coordinator_snapshot_dir=case.coordinator,
        reviewer_release_snapshot_dir=case.main_release,
        calibration_reviewer_release_snapshot_dir=case.calibration_release,
        calibration_result_snapshot_dir=case.calibration,
        calibration_decision_snapshot_dir=case.calibration_decision,
    )
    assert (len(calibration.rows), len(main.rows)) == (60, 344)
    assert len({row["candidate_id"] for row in calibration.rows}) == 30
    assert len({row["candidate_id"] for row in main.rows}) == 172
    empty = _seal_adjudications(case, [])

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="calibration|main|phase",
    ):
        integration.integrate_screening(
            case.coordinator,
            case.calibration_release,
            case.main,
            case.calibration_decision,
            case.main_release,
            case.calibration,
            empty,
            _case_execution_register(case),
            case.citation_keys,
        )


def test_below_threshold_calibration_decision_blocks_main_release(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        screening_batches.SnapshotError,
        match="calibration gate decision is not release",
    ):
        _build_case(
            tmp_path,
            calibration_status_disagreements=7,
        )

    decision_snapshot = tmp_path / "calibration-decision" / "v1"
    _, decisions = _read_csv(decision_snapshot / "decision.csv")
    assert decisions[0]["status_agreement"] == "0.766667"
    assert decisions[0]["decision"] == "revise"
    assert not (tmp_path / "main-reviewer-release" / "v1").exists()
    assert not (tmp_path / "main-results" / "v1").exists()


def test_main_results_cannot_predate_or_bypass_the_passing_gate(
    tmp_path: Path,
) -> None:
    case = _build_case(tmp_path)
    early_release = tmp_path / "early-main-reviewer-release" / "v1"
    early_release.parent.mkdir(parents=True)

    with pytest.raises(
        screening_batches.SnapshotError,
        match="main release requires.*calibration-decision-snapshot",
    ):
        screening_batches.release_snapshot(
            case.coordinator,
            "main",
            early_release,
            calibration_reviewer_release_snapshot=case.calibration_release,
            calibration_result_snapshot=case.calibration,
        )
    assert not early_release.exists()

    bypass_result = tmp_path / "bypass-main-results" / "v1"
    bypass_result.parent.mkdir(parents=True)
    with pytest.raises(
        screening_results.ScreeningResultError,
        match="main phase sealing requires calibration reviewer release",
    ):
        screening_results.seal_phase_results(
            case.coordinator,
            "main",
            _write_phase_result_inputs(
                tmp_path / "bypass-result-inputs",
                case.ratings,
                "main",
            ),
            bypass_result,
            reviewer_release_snapshot_dir=case.main_release,
        )
    assert not bypass_result.exists()


@pytest.mark.parametrize(
    "omitted_argument",
    [
        "calibration_reviewer_release_snapshot_dir",
        "main_reviewer_release_snapshot_dir",
    ],
)
def test_integration_api_requires_each_reviewer_release(
    tmp_path: Path,
    omitted_argument: str,
) -> None:
    arguments = {
        "coordinator_snapshot_dir": tmp_path / "coordinator" / "v1",
        "calibration_reviewer_release_snapshot_dir": (
            tmp_path / "calibration-reviewer-release" / "v1"
        ),
        "calibration_result_snapshot_dir": (
            tmp_path / "calibration-results" / "v1"
        ),
        "calibration_decision_snapshot_dir": (
            tmp_path / "calibration-decision" / "v1"
        ),
        "main_reviewer_release_snapshot_dir": (
            tmp_path / "main-reviewer-release" / "v1"
        ),
        "main_result_snapshot_dir": tmp_path / "main-results" / "v1",
        "adjudication_result_snapshot_dir": (
            tmp_path / "adjudications" / "v1"
        ),
        "execution_register_path": tmp_path / "execution-register.csv",
        "citation_key_ledger_path": tmp_path / "citation-keys.csv",
    }
    del arguments[omitted_argument]

    with pytest.raises(TypeError, match=omitted_argument):
        integration.integrate_screening(**arguments)


@pytest.mark.parametrize(
    ("release_argument", "replacement"),
    [
        (
            "calibration_reviewer_release_snapshot_dir",
            "main_release",
        ),
        ("main_reviewer_release_snapshot_dir", "calibration_release"),
    ],
)
def test_integration_rejects_substituted_reviewer_release(
    tmp_path: Path,
    release_argument: str,
    replacement: str,
) -> None:
    case = _build_case(tmp_path)
    adjudications = _seal_adjudications(case, [])
    arguments = {
        "coordinator_snapshot_dir": case.coordinator,
        "calibration_reviewer_release_snapshot_dir": (
            case.calibration_release
        ),
        "calibration_result_snapshot_dir": case.calibration,
        "calibration_decision_snapshot_dir": case.calibration_decision,
        "main_reviewer_release_snapshot_dir": case.main_release,
        "main_result_snapshot_dir": case.main,
        "adjudication_result_snapshot_dir": adjudications,
        "execution_register_path": _case_execution_register(case),
        "citation_key_ledger_path": case.citation_keys,
    }
    arguments[release_argument] = getattr(case, replacement)

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="reviewer release|release manifest|authorization|phase",
    ):
        integration.integrate_screening(**arguments)


@pytest.mark.parametrize("missing_id", ["C0001", "C0002", "C0003", "C0143"])
def test_every_adjudication_trigger_is_mandatory(
    tmp_path: Path, missing_id: str
) -> None:
    case, rows = _trigger_case(tmp_path)
    rows = [row for row in rows if row["candidate_id"] != missing_id]

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="adjudication coverage",
    ):
        _seal_adjudications(case, rows)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "THE source varies traffic only; course geometry remains FIXED "
            "throughout every experiment.",
            "the source varies traffic only---course geometry remains fixed "
            "throughout every experiment",
        ),
        (
            "The source varies traffic only; course geometry remains fixed "
            "throughout every experiment.",
            "Ｔｈｅ ｓｏｕｒｃｅ ｖａｒｉｅｓ ｔｒａｆｆｉｃ ｏｎｌｙ； "
            "ｃｏｕｒｓｅ ｇｅｏｍｅｔｒｙ ｒｅｍａｉｎｓ ｆｉｘｅｄ "
            "ｔｈｒｏｕｇｈｏｕｔ ｅｖｅｒｙ ｅｘｐｅｒｉｍｅｎｔ．",
        ),
        (
            "The source varies traffic_only while course geometry remains "
            "fixed throughout every experiment.",
            "the source varies traffic-only while course geometry remains "
            "fixed throughout every experiment",
        ),
    ],
)
def test_a3_normalization_uses_nfkc_casefold_and_non_alphanumeric_collapse(
    tmp_path: Path,
    first: str,
    second: str,
) -> None:
    scenarios = {
        "C0009": (
            {
                "screening_status": "excluded",
                "criterion": "exclude-traffic-only",
                "exclusion_reason": first,
            },
            {
                "screening_status": "excluded",
                "criterion": "exclude-traffic-only",
                "exclusion_reason": second,
            },
        )
    }
    case = _build_case(tmp_path, scenarios=scenarios)
    empty = _seal_adjudications(case, [])

    result = integration.integrate_screening(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        empty,
        _case_execution_register(case),
        case.citation_keys,
    )
    candidate = next(
        row for row in result.candidates if row["candidate_id"] == "C0009"
    )
    assert candidate["screening_status"] == "excluded"
    assert candidate["exclusion_reason"] in {first, second}


@pytest.mark.parametrize("mutation", ["prefix", "suffix"])
def test_deciding_locator_section_rejects_prefix_and_suffix_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == "C0001")
    canonical = target["screening_locator"]
    replacement = (
        f"Redirected locator {canonical}"
        if mutation == "prefix"
        else canonical + "0"
    )
    _mutate_resolution_evidence(
        target,
        lambda evidence: evidence.update(deciding_locator=replacement),
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="locator|resolution_evidence",
    ):
        _seal_adjudications(case, rows)


@pytest.mark.parametrize(
    ("candidate_id", "mutate"),
    [
        (
            "C0001",
            lambda row: row.update(
                assignment_ids=";".join(
                    reversed(row["assignment_ids"].split(";"))
                )
            ),
        ),
        (
            "C0001",
            lambda row: row.update(
                reviewer_ids="screening-98;screening-99"
            ),
        ),
        (
            "C0001",
            lambda row: row.update(
                adjudicator_id=row["reviewer_ids"].split(";")[0]
            ),
        ),
        (
            "C0001",
            lambda row: row.update(primary_snapshot_sha256="0" * 64),
        ),
        (
            "C0143",
            lambda row: row.update(
                resolved_conflict_ids="X-NOT-C0143"
            ),
        ),
        (
            "C0001",
            lambda row: row.update(
                source_urls=(
                    "https://example.org/C0001;"
                    "https://archive.example.org/C0001/versions/1.0/"
                )
            ),
        ),
        (
            "C0001",
            lambda row: row.update(
                evidence_archive_url=(
                    "https://archive.example.org/C0001/latest"
                )
            ),
        ),
        (
            "C0001",
            lambda row: row.update(screening_locator="Discussion"),
        ),
        (
            "C0001",
            lambda row: row.update(
                resolution_evidence=row["resolution_evidence"].replace(
                    row["assignment_ids"].split(";")[0],
                    "redacted-assignment",
                )
            ),
        ),
        (
            "C0001",
            lambda row: row.update(
                resolution_evidence=row["resolution_evidence"].replace(
                    "excluded", "redacted-status"
                )
            ),
        ),
        (
            "C0002",
            lambda row: row.update(
                resolution_evidence=row["resolution_evidence"].replace(
                    "include-2", "redacted-criterion"
                )
            ),
        ),
        (
            "C0003",
            lambda row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["raw_exclusion_reasons"][0].update(
                    reason="reason text"
                ),
            ),
        ),
        (
            "C0143",
            lambda row: row.update(
                resolution_evidence=row["resolution_evidence"].replace(
                    "X57B57E64E501", "redacted-conflict"
                )
            ),
        ),
        (
            "C0001",
            lambda row: row.update(
                resolution_evidence=(
                    "Both locked ratings were compared under the controlling "
                    "criterion and status rules. Algorithm 1 supplies deciding "
                    "evidence and the final result preserves both primary ratings."
                )
            ),
        ),
        (
            "C0001",
            lambda row: row.update(resolution_evidence="Too short"),
        ),
    ],
)
def test_adjudication_identity_conflict_and_evidence_bindings_fail_closed(
    tmp_path: Path,
    candidate_id: str,
    mutate: Callable[[dict[str, str]], None],
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == candidate_id)
    mutate(target)

    with pytest.raises(integration.ScreeningIntegrationError):
        _seal_adjudications(case, rows)


def test_genuinely_structured_adjudication_rationales_pass(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    snapshot = _seal_adjudications(case, rows)
    assert snapshot.is_dir()


def test_canonical_evidence_accepts_control_phrases_as_bound_data(
    tmp_path: Path,
) -> None:
    scenarios = _trigger_scenarios()
    first_reason = (
        "The source keeps geometry fixed; Rationale: no reusable layout "
        "procedure appears in the inspected method description."
    )
    second_reason = (
        "Only traffic varies; Source URL: the static course remains unchanged "
        "through every experiment and evaluation."
    )
    scenarios["C0003"] = (
        {
            "screening_status": "excluded",
            "criterion": "exclude-traffic-only",
            "exclusion_reason": first_reason,
        },
        {
            "screening_status": "excluded",
            "criterion": "exclude-traffic-only",
            "exclusion_reason": second_reason,
        },
    )
    conflict = {
        **_conflict("X57B57E64E501", "C0143"),
        "value_a": "candidate Rationale: retained",
        "value_b": "excluded Source URL: legacy record",
    }
    case = _build_case(
        tmp_path,
        scenarios=scenarios,
        conflicts=[conflict],
    )
    rows = [
        _adjudication_row(
            case,
            "C0001",
            screening_locator=(
                "Section 3 Rationale: discussion; Source URL: appendix index"
            ),
            transfer_source_fact=(
                "The inspected source states Source URL: embedded text and "
                "Deciding locator: discussion while generated geometry remains "
                "procedurally reusable."
            ),
        ),
        _adjudication_row(case, "C0002", criterion="include-2"),
        _adjudication_row(
            case,
            "C0003",
            screening_status="excluded",
            criterion="exclude-traffic-only",
            exclusion_reason=first_reason,
        ),
        _adjudication_row(case, "C0143"),
    ]

    snapshot = _seal_adjudications(case, rows)
    assert snapshot.is_dir()


def test_fully_populated_token_dump_resolution_evidence_fails(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == "C0003")
    ratings = case.ratings_for("C0003")
    target["resolution_evidence"] = " ".join(
        (
            "Both ratings were compared in a complete token inventory.",
            target["assignment_ids"],
            ratings[0]["screening_status"],
            ratings[1]["screening_status"],
            ratings[0]["criterion"],
            ratings[1]["criterion"],
            ratings[0]["exclusion_reason"],
            ratings[1]["exclusion_reason"],
            "A3",
            "The exclusion reason disagreement was considered.",
            target["screening_status"],
            target["criterion"],
            target["exclusion_reason"],
            target["source_urls"],
            target["screening_locator"],
            "All listed values support the final result under the rule.",
        )
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="structured|rationale|resolution_evidence",
    ):
        _seal_adjudications(case, rows)


def test_grammar_complete_token_inventory_rationale_fails(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == "C0001")
    ratings = case.ratings_for("C0001")
    comparison = (
        "Candidate C0001 complete token inventory repeats A1 A2 included "
        "excluded include-1 exclude-out-of-scope whereas every known label, "
        "identifier, status, criterion, locator, URL, reason, rule, and final "
        "value is listed without independent candidate-specific comparison "
        "analysis or additional source-grounded observations."
    )
    _mutate_resolution_evidence(
        target,
        lambda evidence: evidence.update(comparison_analysis=comparison),
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="comparison|generic|token|rationale",
    ):
        _seal_adjudications(case, rows)


@pytest.mark.parametrize("mutation", ["prefix", "suffix"])
def test_source_url_section_rejects_prefix_and_suffix_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == "C0001")
    canonical = target["source_urls"].split(";")[1]
    if mutation == "prefix":
        replacement = f"https://redirect.invalid/?target={canonical}"
    else:
        replacement = canonical + "?unexpected-suffix=1"
    _mutate_resolution_evidence(
        target,
        lambda evidence: evidence.update(source_url=replacement),
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="source URL|Source URL|resolution_evidence",
    ):
        _seal_adjudications(case, rows)


@pytest.mark.parametrize(
    ("candidate_id", "mutate"),
    [
        (
            "C0001",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["controlling_rules"].remove("A1"),
            ),
        ),
        (
            "C0002",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["controlling_rules"].remove("A2"),
            ),
        ),
        (
            "C0003",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["controlling_rules"].remove("A3"),
            ),
        ),
        (
            "C0143",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["controlling_rules"].remove("A4"),
            ),
        ),
        (
            "C0003",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["raw_exclusion_reasons"][0].update(
                    reason="redacted complete raw exclusion reason"
                ),
            ),
        ),
        (
            "C0143",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["resolved_conflicts"][0].update(
                    field="redacted"
                ),
            ),
        ),
        (
            "C0143",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["resolved_conflicts"][0].update(
                    value_a="redacted"
                ),
            ),
        ),
        (
            "C0001",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["final_decision"].pop(
                    "screening_status"
                ),
            ),
        ),
        (
            "C0003",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["deciding_fact"].update(
                    text="redacted"
                ),
            ),
        ),
        (
            "C0001",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence["deciding_fact"].update(
                    kind="generic"
                ),
            ),
        ),
        (
            "C0001",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence.pop("source_url"),
            ),
        ),
        (
            "C0001",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence.pop("deciding_locator"),
            ),
        ),
        (
            "C0001",
            lambda _case, row: _mutate_resolution_evidence(
                row,
                lambda evidence: evidence.pop("comparison_analysis"),
            ),
        ),
    ],
)
def test_structured_resolution_evidence_sections_fail_closed(
    tmp_path: Path,
    candidate_id: str,
    mutate: Callable[[ScreeningCase, dict[str, str]], None],
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == candidate_id)
    mutate(case, target)

    with pytest.raises(integration.ScreeningIntegrationError):
        _seal_adjudications(case, rows)


def test_a4_adjudication_must_resolve_complete_conflict_set(
    tmp_path: Path,
) -> None:
    conflicts = [
        _conflict("X-C0143-KEEP", "C0143"),
        _conflict("X-C0143-RESOLVE", "C0143"),
        {
            **_conflict("X-C0143-TITLE", "C0143"),
            "field": "title",
            "value_a": "Old title",
            "value_b": "New title",
        },
    ]
    case = _build_case(tmp_path, conflicts=conflicts)
    decision = _adjudication_row(
        case,
        "C0143",
        resolved_conflict_ids=("X-C0143-RESOLVE",),
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="exactly resolve all|resolved_conflict_ids",
    ):
        _seal_adjudications(case, [decision])



def test_authoritative_short_conflict_id_is_accepted(
    tmp_path: Path,
) -> None:
    conflict = _conflict("X1", "C0143")
    case = _build_case(tmp_path, conflicts=[conflict])
    decision = _adjudication_row(
        case,
        "C0143",
        resolved_conflict_ids=("X1",),
    )
    adjudications = _seal_adjudications(case, [decision])

    result = integration.integrate_screening(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        adjudications,
        _case_execution_register(case),
        case.citation_keys,
    )
    resolved = next(
        row for row in result.conflicts if row["conflict_id"] == "X1"
    )
    assert resolved["resolution"] == "included"
    assert resolved["resolver"] == "adjudicator-01"


def test_adjudication_provenance_allows_protocol_nr_alternatives(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    full_text = next(row for row in rows if row["candidate_id"] == "C0001")
    full_text.update(
        evidence_archive_url="NR",
        evidence_sha256="NR",
    )
    official = next(row for row in rows if row["candidate_id"] == "C0002")
    official.update(
        access_status="official_documentation",
        evidence_sha256="NR",
    )

    snapshot = _seal_adjudications(case, rows)
    captured = integration.validate_adjudication_snapshot(
        snapshot,
        coordinator_snapshot_dir=case.coordinator,
        calibration_reviewer_release_snapshot_dir=case.calibration_release,
        calibration_result_snapshot_dir=case.calibration,
        calibration_decision_snapshot_dir=case.calibration_decision,
        main_reviewer_release_snapshot_dir=case.main_release,
        main_result_snapshot_dir=case.main,
        execution_register_path=_case_execution_register(case),
    )
    assert len(captured.rows) == 4


def test_official_documentation_adjudication_requires_pin_or_digest(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    target = next(row for row in rows if row["candidate_id"] == "C0001")
    target.update(
        access_status="official_documentation",
        evidence_archive_url="NR",
        evidence_sha256="NR",
    )

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="official_documentation|archive|digest",
    ):
        _seal_adjudications(case, rows)


def test_c0143_activation_requires_append_only_audited_key_issuance(
    tmp_path: Path,
) -> None:
    case = _build_case(
        tmp_path,
        conflicts=[_conflict("X57B57E64E501", "C0143")],
        keyless=frozenset({"C0143"}),
    )
    row = _adjudication_row(case, "C0143")
    adjudications = _seal_adjudications(case, [row])

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="C0143.*audited citation key|citation key.*C0143",
    ):
        integration.integrate_screening(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudications,
            _case_execution_register(case),
            case.citation_keys,
        )

    header, ledger_rows = _read_csv(case.citation_keys)
    assert header == screening_batches.CITATION_KEY_HEADER
    ledger_rows.append(
        {
            "candidate_id": "C0143",
            "cite_key": "Peltomaki2022WassersteinGenerative",
        }
    )
    extended_ledger = tmp_path / "citation-keys-extended.csv"
    _write_csv(
        extended_ledger,
        screening_batches.CITATION_KEY_HEADER,
        ledger_rows,
    )
    result = integration.integrate_screening(
        case.coordinator,
        case.calibration_release,
        case.calibration,
        case.calibration_decision,
        case.main_release,
        case.main,
        adjudications,
        _case_execution_register(case),
        extended_ledger,
    )
    c0143 = next(
        candidate
        for candidate in result.candidates
        if candidate["candidate_id"] == "C0143"
    )
    assert c0143["cite_key"] == "Peltomaki2022WassersteinGenerative"
    assert result.citation_keys[-1] == ledger_rows[-1]



def test_tampered_coordinator_is_rejected_by_authoritative_producer_validator(
    tmp_path: Path,
) -> None:
    case = _build_case(tmp_path)
    adjudications = _seal_adjudications(case, [])
    candidates_path = case.coordinator / "candidates.csv"
    candidates_path.write_bytes(candidates_path.read_bytes() + b"tamper\n")

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="canonical|checksum|content|CSV|snapshot",
    ):
        integration.integrate_screening(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudications,
            _case_execution_register(case),
            case.citation_keys,
        )


def test_projection_is_flat_no_clobber_and_validates_by_full_replay(
    tmp_path: Path,
) -> None:
    case, rows = _trigger_case(tmp_path)
    adjudications = _seal_adjudications(case, rows)
    projection = tmp_path / "projection" / "v1"
    projection.parent.mkdir(parents=True)

    author_verification = _seal_projection(
        case,
        adjudications,
        projection,
    )
    assert {path.name for path in projection.iterdir()} == {
        "candidates.csv",
        "citation_keys.csv",
        "conflicts.csv",
        "screening_decisions.csv",
        "screening_agreement.csv",
        "author_verification.csv",
        "manifest.csv",
        "SHA256SUMS",
    }
    captured = integration.validate_screening_projection(
        projection,
        coordinator_snapshot_dir=case.coordinator,
        calibration_reviewer_release_snapshot_dir=case.calibration_release,
        calibration_result_snapshot_dir=case.calibration,
        calibration_decision_snapshot_dir=case.calibration_decision,
        main_reviewer_release_snapshot_dir=case.main_release,
        main_result_snapshot_dir=case.main,
        adjudication_result_snapshot_dir=adjudications,
        execution_register_path=_case_execution_register(case),
        citation_key_ledger_path=case.citation_keys,
        author_verification_path=author_verification,
    )
    assert captured.primary_snapshot_sha256 == case.primary_hash
    _, projection_manifest = _read_csv(projection / "manifest.csv")
    assert projection_manifest[0]["execution_registry_sha256"] == hashlib.sha256(
        _case_execution_register(case).read_bytes()
    ).hexdigest()
    assert captured.execution_registry_sha256 == projection_manifest[0][
        "execution_registry_sha256"
    ]
    assert projection_manifest[0][
        "calibration_decision_snapshot_sha256"
    ] == _read_csv(case.calibration_decision / "manifest.csv")[1][0][
        "calibration_decision_snapshot_sha256"
    ]
    assert projection_manifest[0]["citation_key_ledger_sha256"] == hashlib.sha256(
        case.citation_keys.read_bytes()
    ).hexdigest()
    assert projection_manifest[0]["author_verification_sha256"] == hashlib.sha256(
        author_verification.read_bytes()
    ).hexdigest()
    assert (projection / "author_verification.csv").read_bytes() == (
        author_verification.read_bytes()
    )
    before = {
        path.name: path.read_bytes() for path in projection.iterdir()
    }

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="already exists",
    ):
        _seal_projection(
            case,
            adjudications,
            projection,
            author_verification=author_verification,
        )
    assert {
        path.name: path.read_bytes() for path in projection.iterdir()
    } == before

    decisions = projection / "screening_decisions.csv"
    decisions.write_bytes(decisions.read_bytes() + b"tamper\n")
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.validate_screening_projection(
            projection,
            coordinator_snapshot_dir=case.coordinator,
            calibration_reviewer_release_snapshot_dir=case.calibration_release,
            calibration_result_snapshot_dir=case.calibration,
            calibration_decision_snapshot_dir=case.calibration_decision,
            main_reviewer_release_snapshot_dir=case.main_release,
            main_result_snapshot_dir=case.main,
            adjudication_result_snapshot_dir=adjudications,
            execution_register_path=_case_execution_register(case),
            citation_key_ledger_path=case.citation_keys,
            author_verification_path=author_verification,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing-candidate", "decision-digest", "pending-status"],
)
def test_projection_rejects_incomplete_or_unbound_author_signoff(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = _build_case(tmp_path)
    adjudications = _seal_adjudications(case, [])
    signoff = _write_author_verification(
        case,
        adjudications,
        name=f"author-signoff-{mutation}.csv",
    )
    _, rows = _read_csv(signoff)
    if mutation == "missing-candidate":
        rows.pop()
    elif mutation == "decision-digest":
        rows[0]["decision_sha256"] = "0" * 64
    else:
        rows[0]["verification_status"] = "pending"
    _write_csv(signoff, integration.AUTHOR_VERIFICATION_HEADER, rows)
    output = tmp_path / "invalid-author-projection" / mutation / "v1"
    output.parent.mkdir(parents=True)

    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="author verification|author_verification|final decision binding",
    ):
        _seal_projection(
            case,
            adjudications,
            output,
            author_verification=signoff,
        )
    assert not output.exists()


def test_projection_publication_detects_post_rename_staged_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _build_case(tmp_path)
    adjudications = _seal_adjudications(case, [])
    output = tmp_path / "projection-race" / "v1"
    output.parent.mkdir(parents=True)
    real_rename = screening_batches._rename_noreplace_at

    def rename_then_mutate(
        parent_fd: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        real_rename(parent_fd, source_name, destination_name)
        root_fd = os.open(
            destination_name,
            screening_batches._DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
        try:
            descriptor = os.open(
                "manifest.csv",
                os.O_WRONLY | os.O_APPEND,
                dir_fd=root_fd,
            )
            try:
                os.write(descriptor, b"tamper")
            finally:
                os.close(descriptor)
        finally:
            os.close(root_fd)

    monkeypatch.setattr(
        screening_batches, "_rename_noreplace_at", rename_then_mutate
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        _seal_projection(
            case,
            adjudications,
            output,
        )
    assert not output.exists()


def test_projection_rolls_back_after_post_publish_input_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _build_case(tmp_path)
    adjudications = _seal_adjudications(case, [])
    output = tmp_path / "projection-input-race" / "v1"
    output.parent.mkdir(parents=True)
    real_publish = screening_batches._publish_artifacts

    def publish_then_mutate(
        output_dir: Path,
        artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        def mutate_then_check() -> None:
            candidate_file = case.coordinator / "candidates.csv"
            candidate_file.write_bytes(
                candidate_file.read_bytes() + b"drift\n"
            )
            if post_publish_check is not None:
                post_publish_check()

        real_publish(
            output_dir,
            artifacts,
            post_publish_check=mutate_then_check,
        )

    monkeypatch.setattr(
        screening_batches, "_publish_artifacts", publish_then_mutate
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        _seal_projection(
            case,
            adjudications,
            output,
        )
    assert not output.exists()


def test_adjudication_rolls_back_after_post_publish_phase_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _build_case(tmp_path)
    input_path = tmp_path / "adjudication-race-input.csv"
    _write_csv(input_path, integration.ADJUDICATION_HEADER, [])
    execution_register = _write_execution_register(case, [])
    output = tmp_path / "adjudication-input-race" / "v1"
    output.parent.mkdir(parents=True)
    real_publish = screening_batches._publish_artifacts

    def publish_then_mutate(
        output_dir: Path,
        artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        def mutate_then_check() -> None:
            phase_manifest = case.main / "manifest.csv"
            phase_manifest.write_bytes(
                phase_manifest.read_bytes() + b"drift\n"
            )
            if post_publish_check is not None:
                post_publish_check()

        real_publish(
            output_dir,
            artifacts,
            post_publish_check=mutate_then_check,
        )

    monkeypatch.setattr(
        screening_batches, "_publish_artifacts", publish_then_mutate
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            input_path,
            execution_register,
            output,
        )
    assert not output.exists()


def test_adjudication_publication_rejects_identical_output_swap_after_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path)
    adjudication_input = tmp_path / "output-swap-adjudications.csv"
    _write_csv(adjudication_input, integration.ADJUDICATION_HEADER, [])
    execution_register = _write_execution_register(case, [])
    output = tmp_path / "adjudication-output-swap" / "v1"
    output.parent.mkdir(parents=True)
    real_publish = screening_batches._publish_artifacts

    def publish_then_swap(
        output_dir: Path,
        artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        def check_then_swap() -> None:
            assert post_publish_check is not None
            post_publish_check()
            _replace_with_identical_copy(
                Path(output_dir),
                tmp_path / "post-callback-output-swap",
            )

        real_publish(
            output_dir,
            artifacts,
            post_publish_check=check_then_swap,
        )

    monkeypatch.setattr(
        screening_batches, "_publish_artifacts", publish_then_swap
    )
    with pytest.raises(
        integration.ScreeningIntegrationError,
        match="changed during post-publication validation",
    ):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            execution_register,
            output,
        )
    assert (output / "manifest.csv").is_file()


@pytest.mark.parametrize(
    "target_name",
    [
        "coordinator",
        "calibration-release",
        "calibration-result",
        "calibration-decision",
        "main-release",
        "main-result",
        "adjudication",
        "register",
        "ledger",
        "author-verification",
    ],
)
def test_projection_rolls_back_after_identical_input_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    case = _build_case(tmp_path)
    adjudications = _seal_adjudications(case, [])
    execution_register = _case_execution_register(case)
    author_verification = _write_author_verification(
        case,
        adjudications,
        execution_register=execution_register,
        name="path-swap-author-verification.csv",
    )
    targets = {
        "coordinator": case.coordinator,
        "calibration-release": case.calibration_release,
        "calibration-result": case.calibration,
        "calibration-decision": case.calibration_decision,
        "main-release": case.main_release,
        "main-result": case.main,
        "adjudication": adjudications,
        "register": execution_register,
        "ledger": case.citation_keys,
        "author-verification": author_verification,
    }
    output = tmp_path / "projection-path-swap" / target_name / "v1"
    output.parent.mkdir(parents=True)
    real_publish = screening_batches._publish_artifacts

    def publish_then_swap(
        output_dir: Path,
        artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        def swap_then_check() -> None:
            _replace_with_identical_copy(
                targets[target_name],
                tmp_path / "projection-swaps" / target_name,
            )
            assert post_publish_check is not None
            post_publish_check()

        real_publish(
            output_dir,
            artifacts,
            post_publish_check=swap_then_check,
        )

    monkeypatch.setattr(
        screening_batches, "_publish_artifacts", publish_then_swap
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        _seal_projection(
            case,
            adjudications,
            output,
            execution_register=execution_register,
            author_verification=author_verification,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "target_name",
    [
        "coordinator",
        "calibration-release",
        "calibration-result",
        "calibration-decision",
        "main-release",
        "main-result",
        "adjudication-input",
        "register",
    ],
)
def test_adjudication_rolls_back_after_identical_input_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    case = _build_case(tmp_path)
    adjudication_input = tmp_path / "path-swap-adjudications.csv"
    _write_csv(adjudication_input, integration.ADJUDICATION_HEADER, [])
    execution_register = _write_execution_register(case, [])
    targets = {
        "coordinator": case.coordinator,
        "calibration-release": case.calibration_release,
        "calibration-result": case.calibration,
        "calibration-decision": case.calibration_decision,
        "main-release": case.main_release,
        "main-result": case.main,
        "adjudication-input": adjudication_input,
        "register": execution_register,
    }
    output = tmp_path / "adjudication-path-swap" / target_name / "v1"
    output.parent.mkdir(parents=True)
    real_publish = screening_batches._publish_artifacts

    def publish_then_swap(
        output_dir: Path,
        artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        def swap_then_check() -> None:
            _replace_with_identical_copy(
                targets[target_name],
                tmp_path / "adjudication-swaps" / target_name,
            )
            assert post_publish_check is not None
            post_publish_check()

        real_publish(
            output_dir,
            artifacts,
            post_publish_check=swap_then_check,
        )

    monkeypatch.setattr(
        screening_batches, "_publish_artifacts", publish_then_swap
    )
    with pytest.raises(integration.ScreeningIntegrationError):
        integration.seal_adjudication_results(
            case.coordinator,
            case.calibration_release,
            case.calibration,
            case.calibration_decision,
            case.main_release,
            case.main,
            adjudication_input,
            execution_register,
            output,
        )
    assert not output.exists()


def test_cli_has_only_seal_adjudication_and_seal_projection_roundtrip(
    tmp_path: Path,
) -> None:
    case = _build_case(tmp_path)
    adjudication_input = tmp_path / "cli-adjudications.csv"
    _write_csv(adjudication_input, integration.ADJUDICATION_HEADER, [])
    execution_register = _write_execution_register(case, [])
    adjudications = tmp_path / "cli-adjudication" / "v1"
    adjudications.parent.mkdir(parents=True)

    assert integration.main(
        [
            "--seal-adjudication",
            "--coordinator-snapshot",
            str(case.coordinator),
            "--calibration-reviewer-release",
            str(case.calibration_release),
            "--calibration-result-snapshot",
            str(case.calibration),
            "--calibration-decision-snapshot",
            str(case.calibration_decision),
            "--main-reviewer-release",
            str(case.main_release),
            "--main-result-snapshot",
            str(case.main),
            "--adjudication-result",
            str(adjudication_input),
            "--execution-register",
            str(execution_register),
            "--output-dir",
            str(adjudications),
        ]
    ) == 0

    case.execution_register_path = execution_register
    author_verification = _write_author_verification(
        case,
        adjudications,
        execution_register=execution_register,
        name="cli-author-verification.csv",
    )

    projection = tmp_path / "cli-projection" / "v1"
    projection.parent.mkdir(parents=True)
    assert integration.main(
        [
            "--seal-projection",
            "--coordinator-snapshot",
            str(case.coordinator),
            "--calibration-reviewer-release",
            str(case.calibration_release),
            "--calibration-result-snapshot",
            str(case.calibration),
            "--calibration-decision-snapshot",
            str(case.calibration_decision),
            "--main-reviewer-release",
            str(case.main_release),
            "--main-result-snapshot",
            str(case.main),
            "--adjudication-result-snapshot",
            str(adjudications),
            "--execution-register",
            str(execution_register),
            "--citation-key-ledger",
            str(case.citation_keys),
            "--author-verification",
            str(author_verification),
            "--output-dir",
            str(projection),
        ]
    ) == 0
    assert (
        integration.validate_screening_projection(
            projection,
            coordinator_snapshot_dir=case.coordinator,
            calibration_reviewer_release_snapshot_dir=case.calibration_release,
            calibration_result_snapshot_dir=case.calibration,
            calibration_decision_snapshot_dir=case.calibration_decision,
            main_reviewer_release_snapshot_dir=case.main_release,
            main_result_snapshot_dir=case.main,
            adjudication_result_snapshot_dir=adjudications,
            execution_register_path=execution_register,
            citation_key_ledger_path=case.citation_keys,
            author_verification_path=author_verification,
        ).candidate_count
        == 202
    )

    with pytest.raises(SystemExit):
        integration.main(["--integrate"])


@pytest.mark.parametrize(
    ("mode", "required_flag"),
    [
        (mode, flag)
        for mode in ("--seal-adjudication", "--seal-projection")
        for flag in (
            "--coordinator-snapshot",
            "--calibration-reviewer-release",
            "--calibration-result-snapshot",
            "--calibration-decision-snapshot",
            "--main-reviewer-release",
            "--main-result-snapshot",
            "--execution-register",
            "--output-dir",
        )
    ]
    + [
        ("--seal-adjudication", "--adjudication-result"),
        ("--seal-projection", "--adjudication-result-snapshot"),
        ("--seal-projection", "--citation-key-ledger"),
        ("--seal-projection", "--author-verification"),
    ],
)
def test_cli_modes_require_every_bound_input(
    mode: str,
    required_flag: str,
) -> None:
    arguments = [
        mode,
        "--coordinator-snapshot",
        "coordinator/v1",
        "--calibration-reviewer-release",
        "calibration-reviewer-release/v1",
        "--calibration-result-snapshot",
        "calibration/v1",
        "--calibration-decision-snapshot",
        "calibration-decision/v1",
        "--main-reviewer-release",
        "main-reviewer-release/v1",
        "--main-result-snapshot",
        "main/v1",
        "--execution-register",
        "execution-register.csv",
        "--output-dir",
        "output/v1",
    ]
    if mode == "--seal-adjudication":
        arguments.extend(("--adjudication-result", "adjudications.csv"))
    else:
        arguments.extend(
            (
                "--adjudication-result-snapshot",
                "adjudications/v1",
                "--citation-key-ledger",
                "citation-keys.csv",
                "--author-verification",
                "author-verification.csv",
            )
        )
    index = arguments.index(required_flag)
    del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        integration.main(arguments)


@pytest.mark.parametrize(
    ("mode", "forbidden_flag"),
    [
        ("--seal-adjudication", "--adjudication-result-snapshot"),
        ("--seal-projection", "--adjudication-result"),
    ],
)
def test_cli_modes_reject_opposite_adjudication_input(
    mode: str,
    forbidden_flag: str,
) -> None:
    arguments = [
        mode,
        "--coordinator-snapshot",
        "coordinator/v1",
        "--calibration-reviewer-release",
        "calibration-reviewer-release/v1",
        "--calibration-result-snapshot",
        "calibration/v1",
        "--calibration-decision-snapshot",
        "calibration-decision/v1",
        "--main-reviewer-release",
        "main-reviewer-release/v1",
        "--main-result-snapshot",
        "main/v1",
        "--execution-register",
        "execution-register.csv",
        "--output-dir",
        "output/v1",
    ]
    if mode == "--seal-adjudication":
        arguments.extend(("--adjudication-result", "adjudications.csv"))
    else:
        arguments.extend(
            (
                "--adjudication-result-snapshot",
                "adjudications/v1",
                "--citation-key-ledger",
                "citation-keys.csv",
                "--author-verification",
                "author-verification.csv",
            )
        )
    arguments.extend((forbidden_flag, "forbidden"))

    with pytest.raises(SystemExit):
        integration.main(arguments)


def test_cli_requires_exactly_one_seal_mode() -> None:
    with pytest.raises(SystemExit):
        integration.main([])
    with pytest.raises(SystemExit):
        integration.main(["--seal-adjudication", "--seal-projection"])
