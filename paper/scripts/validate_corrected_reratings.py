"""Validate the append-only v1 corrected-rerating sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Sequence


class CorrectedReratingValidationError(ValueError):
    """The corrected-rerating sidecar is incomplete or inconsistent."""


SNAPSHOT_SUFFIX = Path("corrected_reratings/v1")
RATING_HEADER = (
    "candidate_id", "reviewer_slot", "reviewer_agent_id", "reviewer_model",
    "reviewer_reasoning_effort", "retention_role", "screening_status", "criterion",
    "access_status", "source_urls", "evidence_version", "evidence_retrieved_on",
    "evidence_archive_url", "evidence_sha256", "screening_locator",
    "exclusion_reason", "protocol_reasoning", "deciding_source_fact", "notes",
)
CORRECTIONS_HEADER = (
    "candidate_id", "correction_version", "old_assignment_ids", "old_evidence_sha256",
    "corrected_evidence_path", "corrected_evidence_sha256", "correction_reason",
)
AGREEMENT_HEADER = (
    "candidate_id", "reviewer_slots", "status_agreement", "criterion_agreement",
    "evidence_sha256_agreement", "agreed_screening_status", "agreed_criterion",
    "agreed_evidence_sha256",
)
REGISTRY_HEADER = (
    "candidate_id", "reviewer_slot", "agent_id", "model", "reasoning_effort",
    "human_role", "execution_context", "metadata_limit",
)
BINDINGS_HEADER = ("binding", "bound_path", "bound_sha256", "purpose")
CHECKSUM_HEADER = ("record_type", "path", "sha256", "row_count")
CALIBRATION_MANIFEST_HEADER = (
    "manifest_version", "phase_result_snapshot_sha256",
    "coordinator_snapshot_sha256", "protocol_sha256", "reviewer_release_sha256",
    "phase", "batch_id", "coder_id", "result_filename", "result_file_sha256",
    "row_count",
)
FROZEN_RATING_HEADER = (
    "assignment_id", "phase", "candidate_id", "input_sha256", "snapshot_sha256",
    "batch_id", "coder_id", "screened_on", "screening_status", "criterion",
    "access_status", "source_urls", "evidence_version", "evidence_retrieved_on",
    "evidence_archive_url", "evidence_sha256", "screening_locator",
    "exclusion_reason", "notes",
)
CALIBRATION_MANIFEST_PATH = (
    "paper/data/screening_results/calibration/v8/manifest.csv"
)
REVIEWERS = {
    ("C0046", "A"): "019f39f4-2366-7080-9b3b-28da615980eb",
    ("C0046", "B"): "019f39f4-23fb-74a1-8692-33338053f0f4",
    ("C0173", "A"): "019f39f4-238c-7210-93bd-e1c08d06e89b",
    ("C0173", "B"): "019f39f4-23ce-73c2-b723-6f22d61c5222",
}
CORRECTIONS = {
    "C0046": {
        "assignments": "A-C0046-03;A-C0046-04",
        "old_digest": "72e78ea7f50b48779f8f6e6344cdba945a5d0d1502139ff5dd16018215f5e2ef",
        "path": "paper/data/source_archive/v8/C0046/metadrive_composing_diverse_driving_scenarios_arxiv_2109.12674v3.pdf",
        "digest": "ae04f7e4a3976f977ccff136a218d0daca5a84224f68ded684a6a734f8a87e2e",
    },
    "C0173": {
        "assignments": "A-C0173-04;A-C0173-05",
        "old_digest": "77902e38743747989a4e4e79784d96ba11f29e8c92c5881e211a6790030fa227",
        "path": "paper/data/source_archive/v8/C0173/bayesrace-pmlr155-jain21b-corrected.pdf",
        "digest": "83f24aad0cb3242fdcb9f81dbe8d53de0b832bf7ffb7517c7a114c129c98d0ba",
    },
}
REQUIRED_FILES = frozenset(
    {
        "README.md", "PROCEDURAL-LIMITATIONS.md", "corrections.csv", "ratings.csv",
        "agreement.csv", "execution_registry.csv", "bindings.csv", "SHA256SUMS",
        "manifest/checksums.csv",
        *(f"inputs/trackgen-corrected-rerating-{candidate}-{slot}.json" for candidate in ("C0046", "C0173") for slot in ("A", "B")),
    }
)
REQUIRED_LIMITS = (
    "not a sealed primary, adjudication, or projection snapshot",
    "coordinator-recorded agent/model metadata is not provider-side attestation",
    "/tmp is same-user procedural isolation",
    "no human accountable-author verification",
    "not integrated into a final-version primary snapshot",
    "no final projection or quantitative claim",
    "frozen v8 releases, results, and adjudication drafts are not modified",
)


def _fail(message: str) -> None:
    raise CorrectedReratingValidationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            _fail(f"{label} must be a regular non-symlink file")
        return path.read_bytes()
    except OSError as exc:
        raise CorrectedReratingValidationError(f"unable to read {label}") from exc


def _read_csv(path: Path, header: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = _regular_bytes(path, label=str(path)).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != header:
            _fail(f"{path.name} has an unexpected header")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise CorrectedReratingValidationError(f"{path.name} is not valid UTF-8 CSV") from exc
    if any(set(row) != set(header) or any(value is None for value in row.values()) for row in rows):
        _fail(f"{path.name} has malformed rows")
    return rows


def _snapshot_path(snapshot: Path) -> Path:
    if snapshot.parts[-2:] != SNAPSHOT_SUFFIX.parts:
        _fail(f"snapshot path must end with {SNAPSHOT_SUFFIX}")
    try:
        resolved = snapshot.resolve(strict=True)
    except OSError as exc:
        raise CorrectedReratingValidationError("snapshot is unavailable") from exc
    if snapshot.is_symlink() or not resolved.is_dir():
        _fail("snapshot must be a real directory")
    return resolved


def _validate_file_set(snapshot: Path) -> None:
    files: set[str] = set()
    for path in snapshot.rglob("*"):
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
            _fail("snapshot contains a symlink or non-regular path")
        if path.is_file():
            files.add(path.relative_to(snapshot).as_posix())
    if files != REQUIRED_FILES:
        _fail("snapshot has an unexpected file set")


def _validate_corrections(
    snapshot: Path,
    repository_root: Path,
) -> list[dict[str, str]]:
    rows = _read_csv(snapshot / "corrections.csv", CORRECTIONS_HEADER)
    if [row["candidate_id"] for row in rows] != ["C0046", "C0173"]:
        _fail("corrections roster is not exact")
    for row in rows:
        expected = CORRECTIONS[row["candidate_id"]]
        if (
            row["correction_version"] != "1"
            or row["old_assignment_ids"] != expected["assignments"]
            or row["old_evidence_sha256"] != expected["old_digest"]
            or row["corrected_evidence_path"] != expected["path"]
            or row["corrected_evidence_sha256"] != expected["digest"]
            or not row["correction_reason"]
            or row["old_evidence_sha256"] == row["corrected_evidence_sha256"]
        ):
            _fail("correction record does not match the fixed corrected evidence")
        evidence = repository_root / row["corrected_evidence_path"]
        if _sha256(_regular_bytes(evidence, label="corrected evidence")) != row["corrected_evidence_sha256"]:
            _fail("corrected evidence digest mismatch")

    return rows

def _validate_ratings(snapshot: Path) -> list[dict[str, str]]:
    rows = _read_csv(snapshot / "ratings.csv", RATING_HEADER)
    expected_keys = [("C0046", "A"), ("C0046", "B"), ("C0173", "A"), ("C0173", "B")]
    keys = [(row["candidate_id"], row["reviewer_slot"]) for row in rows]
    if keys != expected_keys:
        _fail("ratings roster must be sorted C0046/C0173 with A/B slots")
    if len({row["reviewer_agent_id"] for row in rows}) != 4:
        _fail("reviewer agents must be unique")
    for row in rows:
        key = (row["candidate_id"], row["reviewer_slot"])
        expected = CORRECTIONS[row["candidate_id"]]
        if (
            row["reviewer_agent_id"] != REVIEWERS[key]
            or row["reviewer_model"] != "gpt-5.6-terra"
            or row["reviewer_reasoning_effort"] != "high"
            or row["retention_role"] != "core"
            or row["screening_status"] != "included"
            or row["criterion"] != "include-relevant"
            or row["access_status"] != "full_text"
            or row["evidence_sha256"] != expected["digest"]
        ):
            _fail("ratings row does not match the fixed reviewer or evidence contract")
        input_path = snapshot / "inputs" / f"trackgen-corrected-rerating-{key[0]}-{key[1]}.json"
        try:
            raw = json.loads(_regular_bytes(input_path, label="raw input").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorrectedReratingValidationError("raw input is not valid JSON") from exc
        rating_fields = {field: row[field] for field in RATING_HEADER if field not in {"reviewer_agent_id", "reviewer_model", "reviewer_reasoning_effort"}}
        if set(raw) != set(rating_fields):
            _fail("raw input fields do not match normalized rating fields")
        normalized_raw = dict(raw)
        normalized_raw["retention_role"] = "core"
        if normalized_raw != rating_fields:
            _fail("raw input does not match normalized rating")
    return rows


def _validate_agreement(snapshot: Path, ratings: list[dict[str, str]]) -> None:
    rows = _read_csv(snapshot / "agreement.csv", AGREEMENT_HEADER)
    if [row["candidate_id"] for row in rows] != ["C0046", "C0173"]:
        _fail("agreement roster is not exact")
    for row in rows:
        candidate_ratings = [rating for rating in ratings if rating["candidate_id"] == row["candidate_id"]]
        values = {field: {rating[field] for rating in candidate_ratings} for field in ("screening_status", "criterion", "evidence_sha256")}
        if (
            row["reviewer_slots"] != "A;B"
            or row["status_agreement"] != "exact"
            or row["criterion_agreement"] != "exact"
            or row["evidence_sha256_agreement"] != "exact"
            or any(len(value) != 1 for value in values.values())
            or row["agreed_screening_status"] != values["screening_status"].pop()
            or row["agreed_criterion"] != values["criterion"].pop()
            or row["agreed_evidence_sha256"] != values["evidence_sha256"].pop()
        ):
            _fail("agreement record is not an exact duplicate-rating agreement")


def _validate_registry(snapshot: Path) -> None:
    rows = _read_csv(snapshot / "execution_registry.csv", REGISTRY_HEADER)
    if len(rows) != 4:
        _fail("execution registry must contain four rows")
    keys = {(row["candidate_id"], row["reviewer_slot"]) for row in rows}
    if keys != set(REVIEWERS):
        _fail("execution registry roster is not exact")
    if len({row["agent_id"] for row in rows}) != 4:
        _fail("execution registry must contain four unique agent IDs")
    for row in rows:
        key = (row["candidate_id"], row["reviewer_slot"])
        if (
            key not in REVIEWERS
            or row["agent_id"] != REVIEWERS[key]
            or row["model"] != "gpt-5.6-terra"
            or row["reasoning_effort"] != "high"
            or row["human_role"] != "NR"
            or row["execution_context"] != "isolated /tmp packet handoff"
            or row["metadata_limit"] != "coordinator-recorded; not provider-side attestation"
        ):
            _fail("execution registry does not match the fixed rerating record")


def _validate_bindings(
    snapshot: Path,
    repository_root: Path,
) -> list[dict[str, str]]:
    rows = _read_csv(snapshot / "bindings.csv", BINDINGS_HEADER)
    expected_paths = {
        "paper/data/screening_work/v8/protocol.md",
        "paper/data/screening_results/calibration/v8/manifest.csv",
        *(value["path"] for value in CORRECTIONS.values()),
    }
    if {row["bound_path"] for row in rows} != expected_paths or len(rows) != 4:
        _fail("bindings roster is not exact")
    for row in rows:
        if not row["binding"] or not row["purpose"]:
            _fail("binding must include a purpose")
        bound = repository_root / row["bound_path"]
        if _sha256(_regular_bytes(bound, label="bound artifact")) != row["bound_sha256"]:
            _fail("bound artifact digest mismatch")

    return rows


def _validate_old_assignment_provenance(
    repository_root: Path,
    corrections: list[dict[str, str]],
    bindings: list[dict[str, str]],
) -> None:
    manifest_binding = next(
        row for row in bindings if row["bound_path"] == CALIBRATION_MANIFEST_PATH
    )
    manifest_path = repository_root / manifest_binding["bound_path"]
    manifest_rows = _read_csv(manifest_path, CALIBRATION_MANIFEST_HEADER)
    expected_assignments = [
        (assignment_id, correction["candidate_id"], correction["old_evidence_sha256"])
        for correction in corrections
        for assignment_id in correction["old_assignment_ids"].split(";")
    ]
    batch_ids = {
        f"screening-{assignment_id.rsplit('-', 1)[1]}"
        for assignment_id, _, _ in expected_assignments
    }
    relevant_manifest_rows = [
        row for row in manifest_rows if row["batch_id"] in batch_ids
    ]
    if (
        len(relevant_manifest_rows) != len(batch_ids)
        or {row["batch_id"] for row in relevant_manifest_rows} != batch_ids
    ):
        _fail("old assignment provenance manifest entries are not exact")

    frozen_rows: list[dict[str, str]] = []
    for row in relevant_manifest_rows:
        batch_id = row["batch_id"]
        if (
            row["manifest_version"] != "1"
            or row["phase"] != "calibration"
            or row["coder_id"] != batch_id
            or row["result_filename"] != f"{batch_id}.csv"
            or not row["row_count"].isdigit()
        ):
            _fail("old assignment provenance manifest entry is invalid")
        result_path = manifest_path.parent / row["result_filename"]
        result_payload = _regular_bytes(result_path, label="frozen calibration result")
        if _sha256(result_payload) != row["result_file_sha256"]:
            _fail("old assignment provenance result digest mismatch")
        result_rows = _read_csv(result_path, FROZEN_RATING_HEADER)
        if len(result_rows) != int(row["row_count"]):
            _fail("old assignment provenance result row count mismatch")
        frozen_rows.extend(result_rows)

    for assignment_id, candidate_id, old_digest in expected_assignments:
        matches = [
            row for row in frozen_rows if row["assignment_id"] == assignment_id
        ]
        if (
            len(matches) != 1
            or matches[0]["candidate_id"] != candidate_id
            or matches[0]["evidence_sha256"] != old_digest
        ):
            _fail("old assignment provenance does not match corrections.csv")

def _validate_limits(snapshot: Path) -> None:
    text = "\n".join(
        _regular_bytes(snapshot / name, label=name).decode("utf-8").lower()
        for name in ("README.md", "PROCEDURAL-LIMITATIONS.md")
    )
    if any(limit not in text for limit in REQUIRED_LIMITS):
        _fail("required non-final procedural limitations are missing")


def _validate_checksums(snapshot: Path) -> None:
    rows = _read_csv(snapshot / "manifest/checksums.csv", CHECKSUM_HEADER)
    expected_paths = REQUIRED_FILES - {"SHA256SUMS", "manifest/checksums.csv"}
    if {row["path"] for row in rows} != expected_paths or len(rows) != len(expected_paths):
        _fail("checksum manifest has an unexpected file set")
    actual: dict[str, str] = {}
    for row in rows:
        if row["record_type"] not in {"artifact", "input"} or not row["row_count"].isdigit():
            _fail("checksum manifest has an invalid row")
        path = snapshot / row["path"]
        digest = _sha256(_regular_bytes(path, label="checksummed artifact"))
        if digest != row["sha256"]:
            _fail("checksum manifest digest mismatch")
        actual[row["path"]] = digest
    expected_sums = [
        f"{_sha256(_regular_bytes(snapshot / 'manifest/checksums.csv', label='checksum manifest'))}  manifest/checksums.csv\n",
        *(f"{digest}  {path}\n" for path, digest in sorted(actual.items())),
    ]
    if _regular_bytes(snapshot / "SHA256SUMS", label="SHA256SUMS") != "".join(expected_sums).encode("ascii"):
        _fail("SHA256SUMS is not deterministic")


def validate_snapshot(*, repository_root: Path, snapshot: Path) -> None:
    """Validate the fixed, append-only corrected-rerating snapshot."""
    snapshot = _snapshot_path(snapshot)
    _validate_file_set(snapshot)
    corrections = _validate_corrections(snapshot, repository_root)
    ratings = _validate_ratings(snapshot)
    _validate_agreement(snapshot, ratings)
    _validate_registry(snapshot)
    bindings = _validate_bindings(snapshot, repository_root)
    _validate_old_assignment_provenance(repository_root, corrections, bindings)
    _validate_limits(snapshot)
    _validate_checksums(snapshot)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        validate_snapshot(repository_root=args.repository_root.resolve(), snapshot=args.snapshot)
    except CorrectedReratingValidationError as exc:
        print(f"corrected-rerating validation failed: {exc}", file=sys.stderr)
        return 1
    print("corrected-rerating snapshot validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
