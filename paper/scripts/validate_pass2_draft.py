"""Validate the isolated, non-final Pass-2 v1 coding release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.scripts.prepare_pass2_draft import (
    ALLOWED_ACCESS,
    C0110_STAGED_RELATIVE,
    CANDIDATES_HEADER,
    CLAIMS_HEADER,
    DRAFT_KEY_PREFIX,
    EVIDENCE_HEADER,
    METRICS_HEADER,
    PACKET_FIELDS,
    PRIMARY_BATCH_COUNT,
    RELEASE_MANIFEST_HEADER,
    RELEASE_NAME,
    ROSTER_SIZE,
    SIMULATORS_HEADER,
    SOURCE_ARCHIVE_RELATIVE,
    SOURCE_INDEX_HEADER,
    _build_rows,
    _csv_bytes,
    _nonfinal_markdown,
    _primary_assignment,
    _read_csv,
    _regular_bytes,
    _release_payloads,
    _required_input_paths,
    _relative,
    _sha256,
)


DRAFT_KEY_PATTERN = re.compile(r"DRAFT_C[0-9]{4}\Z")
FIELD_LOCATOR_PATTERN = re.compile(
    r"(?:^|;)\s*(domain|vehicle|course_object|representation_family|"
    r"generator_family|generation_role|validity_strategy|geometry_metrics|"
    r"difficulty_metrics|diversity_metrics|training_distribution|"
    r"evaluation_suite|simulator|export_format|code_status|asset_status|"
    r"reproducibility_fields)=[^;]+"
)
BANNED_RELEASE_MARKERS = (
    "final screening projection",
    "final corpus",
    "production corpus",
    "paper/data/evidence.csv",
    "paper/data/claims.csv",
    "paper/data/metrics.csv",
    "paper/data/simulators.csv",
)


class DraftValidationError(ValueError):
    """The draft release is malformed, non-deterministic, or unsafe."""


def _fail(message: str) -> None:
    raise DraftValidationError(message)


def _read_release_csv(path: Path, header: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        return _read_csv(path, header)
    except ValueError as exc:
        _fail(str(exc))


def _release_path(root: Path, release: Path) -> Path:
    expected_suffix = Path("pass2_drafts") / RELEASE_NAME
    if release.parts[-2:] != expected_suffix.parts:
        _fail(f"release path must end with {expected_suffix}")
    try:
        resolved = release.resolve(strict=True)
    except OSError as exc:
        _fail(f"release path is unavailable: {exc}")
    if release.is_symlink() or not resolved.is_dir():
        _fail("release must be a real directory")
    try:
        resolved.relative_to(root)
    except ValueError:
        # Focused tests intentionally use a temporary, isolated release root.
        pass
    return resolved


def _safe_relative_posix(value: str, *, direct: bool) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or (direct and len(path.parts) != 1)
    ):
        _fail("manifest paths must be safe relative POSIX paths")
    return path


def _manifest_file(root: Path, release: Path, row: dict[str, str]) -> Path:
    relative = _safe_relative_posix(
        row["path"], direct=row["record_type"] == "generated"
    )
    base = release if row["record_type"] == "generated" else root
    candidate = base.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        if candidate.is_symlink() or resolved != candidate or not resolved.is_file():
            _fail(f"manifest path is missing or aliased: {row['path']}")
        if row["record_type"] == "input":
            resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(f"manifest path is missing or aliased: {row['path']}")
    return resolved


def _expected_manifest_rows(
    root: Path,
    source_index: list[dict[str, str]],
    candidates: list[dict[str, str]],
    batches: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    payloads = _release_payloads(source_index, candidates, batches)
    rows = [
        {
            "record_type": "generated",
            "path": path,
            "sha256": _sha256(payload),
            "row_count": "NR",
        }
        for path, payload in sorted(payloads.items())
    ]
    rows.extend(
        {
            "record_type": "input",
            "path": _relative(root, path),
            "sha256": _sha256(_regular_bytes(path, label="authoritative input")),
            "row_count": "NR",
        }
        for path in _required_input_paths(root)
    )
    return sorted(rows, key=lambda row: (row["record_type"], row["path"]))


def _manifest_rows(root: Path, release: Path) -> list[dict[str, str]]:
    rows = _read_release_csv(release / "release_manifest.csv", RELEASE_MANIFEST_HEADER)
    if not rows:
        _fail("release manifest must not be empty")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["record_type"], row["path"])
        if key in seen or row["record_type"] not in {"generated", "input"}:
            _fail("release manifest has duplicate or invalid records")
        seen.add(key)
        _safe_relative_posix(
            row["path"], direct=row["record_type"] == "generated"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            _fail("release manifest has invalid SHA-256")
        if row["row_count"] != "NR":
            _fail("release manifest row_count must be NR")
    return rows


def _verify_manifest_and_sums(
    root: Path,
    release: Path,
    source_index: list[dict[str, str]],
    candidates: list[dict[str, str]],
    batches: dict[str, list[dict[str, str]]],
) -> None:
    rows = _manifest_rows(root, release)
    expected_rows = _expected_manifest_rows(root, source_index, candidates, batches)
    if _csv_bytes(RELEASE_MANIFEST_HEADER, rows) != _csv_bytes(
        RELEASE_MANIFEST_HEADER, expected_rows
    ):
        _fail("release manifest must contain exact expected records")
    actual: dict[tuple[str, str], str] = {}
    for row in rows:
        path = _manifest_file(root, release, row)
        actual[(row["record_type"], row["path"])] = _sha256(
            _regular_bytes(path, label="manifest artifact")
        )
    expected = {(row["record_type"], row["path"]): row["sha256"] for row in rows}
    if actual != expected:
        _fail("release manifest checksum mismatch")
    sums_path = release / "SHA256SUMS"
    sums = _regular_bytes(sums_path, label="SHA256SUMS")
    expected_lines = [
        f"{_sha256(_regular_bytes(release / 'release_manifest.csv', label='release manifest'))}  generated/release_manifest.csv\n"
    ]
    expected_lines.extend(
        f"{digest}  {record_type}/{path}\n"
        for (record_type, path), digest in sorted(actual.items())
    )
    if sums != "".join(sorted(expected_lines)).encode("ascii"):
        _fail("SHA256SUMS is not deterministic")


def _validate_nonfinal_text(release: Path) -> None:
    for path in release.iterdir():
        if path.is_symlink() or not path.is_file():
            _fail("release contains an unsafe path")
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            _fail(f"release artifact is not UTF-8: {path.name}")
        if any(marker in text for marker in BANNED_RELEASE_MARKERS):
            _fail(f"release contains a prohibited production or final-corpus marker: {path.name}")
    if _regular_bytes(release / "DRAFT-NONFINAL.md", label="limitations") != _nonfinal_markdown():
        _fail("DRAFT-NONFINAL.md is not the deterministic non-final notice")


def _validate_roster(
    root: Path,
    release: Path,
    evidence_archive: Path,
    c0110_packet_bytes: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    expected_index, expected_candidates, expected_batches = _build_rows(
        root, evidence_archive, c0110_packet_bytes
    )
    actual_index = _read_release_csv(release / "source_index.csv", SOURCE_INDEX_HEADER)
    actual_candidates = _read_release_csv(release / "candidates.csv", CANDIDATES_HEADER)
    if len(actual_index) != ROSTER_SIZE or len(actual_candidates) != ROSTER_SIZE:
        _fail(f"repository release must contain exactly {ROSTER_SIZE} sources")
    source_ids = [row["source_candidate_id"] for row in actual_index]
    draft_keys = [row["draft_key"] for row in actual_index]
    if len(set(source_ids)) != ROSTER_SIZE or len(set(draft_keys)) != ROSTER_SIZE:
        _fail("source index keys must be unique")
    if any(not DRAFT_KEY_PATTERN.fullmatch(key) for key in draft_keys):
        _fail("source index draft keys must use DRAFT_C####")
    if any(row["candidate_id"] != row["cite_key"] or not DRAFT_KEY_PATTERN.fullmatch(row["cite_key"]) for row in actual_candidates):
        _fail("candidates must use only DRAFT_C#### coding keys")
    if _csv_bytes(SOURCE_INDEX_HEADER, actual_index) != _csv_bytes(SOURCE_INDEX_HEADER, expected_index):
        _fail("source packet provenance, access status, hashes, or roster binding changed")
    if _csv_bytes(CANDIDATES_HEADER, actual_candidates) != _csv_bytes(CANDIDATES_HEADER, expected_candidates):
        _fail("draft candidates are not deterministic")
    c0143 = next((row for row in actual_index if row["source_candidate_id"] == "C0143"), None)
    if c0143 is None or c0143["canonical_cite_key"] or c0143["citation_activation_status"] != "blocked":
        _fail("C0143 must remain blocked from citation activation")
    return expected_index, expected_candidates, expected_batches


def _validate_evidence_rows(rows: list[dict[str, str]], taxonomy: dict[str, list[str]]) -> None:
    analytical = (
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
    )
    controlled = {
        "domain": "domain",
        "course_object": "course_object",
        "representation_family": "representation_family",
        "generator_family": "generator_family",
        "generation_role": "generation_role",
        "validity_strategy": "validity_strategy",
        "code_status": "code_status",
        "asset_status": "code_status",
    }
    for row in rows:
        if not DRAFT_KEY_PATTERN.fullmatch(row["cite_key"]):
            _fail("evidence rows must use DRAFT_C#### cite keys")
        values = [row[field] for field in EVIDENCE_HEADER if field != "cite_key"]
        if not any(values):
            continue
        tier = row["survey_evidence_tier"]
        tier_labels = [label.strip() for label in tier.split(";") if label.strip()]
        if "NR" in tier_labels and tier_labels != ["NR"]:
            _fail("controlled fields may use NR only as a sole NR sentinel")
        if tier != "NR" and tier not in taxonomy["survey_evidence_tier"]:
            _fail("completed evidence row requires a scalar controlled tier")
        for field, taxonomy_key in controlled.items():
            value = row[field]
            if not value:
                continue
            labels = [label.strip() for label in value.split(";") if label.strip()]
            if "NR" in labels:
                if labels != ["NR"]:
                    _fail("controlled fields may use NR only as a sole NR sentinel")
                continue
            if not labels or any(label not in taxonomy[taxonomy_key] for label in labels):
                _fail(f"invalid taxonomy-controlled value for {field}")
            if field in {"code_status", "asset_status"} and len(labels) != 1:
                _fail(f"{field} must be scalar")
            if field not in {"code_status", "asset_status"}:
                order = [taxonomy[taxonomy_key].index(label) for label in labels]
                if order != sorted(order) or len(set(labels)) != len(labels):
                    _fail(f"{field} labels must follow taxonomy order without duplicates")
        non_nr = [field for field in analytical if row[field] and row[field] != "NR"]
        if non_nr:
            locator = row["evidence_locator"]
            if not locator or any(
                not re.search(rf"(?:^|;)\s*{re.escape(field)}=[^;]+", locator)
                for field in non_nr
            ):
                _fail("completed evidence rows require field-addressable precise locators")


def _validate_templates(release: Path, source_index: list[dict[str, str]], batches: dict[str, list[dict[str, str]]], root: Path) -> None:
    expected_keys = {row["draft_key"] for row in source_index}
    taxonomy = json.loads((root / "paper/data/taxonomy.json").read_text(encoding="utf-8"))
    evidence = _read_release_csv(release / "evidence_template.csv", EVIDENCE_HEADER)
    if {row["cite_key"] for row in evidence} != expected_keys or len(evidence) != ROSTER_SIZE:
        _fail("evidence template must contain one row per draft source")
    _validate_evidence_rows(evidence, taxonomy)
    for batch_id, expected_rows in batches.items():
        rows = _read_release_csv(release / f"{batch_id}.csv", EVIDENCE_HEADER)
        if _csv_bytes(EVIDENCE_HEADER, rows) != _csv_bytes(EVIDENCE_HEADER, expected_rows):
            _fail(f"{batch_id} is not deterministic")
        _validate_evidence_rows(rows, taxonomy)
    for filename, header in (
        ("claims_template.csv", CLAIMS_HEADER),
        ("metrics_template.csv", METRICS_HEADER),
        ("simulators_template.csv", SIMULATORS_HEADER),
    ):
        rows = _read_release_csv(release / filename, header)
        for row in rows:
            references = row.get("cite_keys", "") or row.get("cite_key", "")
            if references:
                for key in references.split(";"):
                    if key not in expected_keys:
                        _fail(f"{filename} has a source reference outside the draft roster")


def _validate_references(
    filename: str,
    rows: list[dict[str, str]],
    roster: set[str],
    *, references_field: str, plural: bool,
) -> None:
    for row in rows:
        value = row[references_field]
        if not value:
            continue
        if not plural and ";" in value:
            _fail(f"{filename} cite_key must contain exactly one draft key")
        keys = [key.strip() for key in value.split(";")]
        if not all(keys) or any(key not in roster for key in keys):
            _fail(f"{filename} has a source reference outside the draft roster")


def validate_coding_output(
    *, repository_root: Path, release: Path, coding_output: Path
) -> None:
    """Validate mutable coder output without changing the immutable release."""
    try:
        root = repository_root.resolve(strict=True)
        release_root = _release_path(root, release)
        output = coding_output.resolve(strict=True)
        if coding_output.is_symlink() or not output.is_dir():
            _fail("coding output must be a real directory")
        allowed = {"evidence.csv", "claims.csv", "metrics.csv", "simulators.csv"}
        names = {path.name for path in output.iterdir()}
        if "evidence.csv" not in names or not names <= allowed:
            _fail("coding output must contain evidence.csv and no other artifacts")
        if any(path.is_symlink() or not path.is_file() for path in output.iterdir()):
            _fail("coding output contains an unsafe path")
        for path in output.iterdir():
            try:
                text = path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                _fail(f"coding output artifact is not UTF-8: {path.name}")
            if any(marker in text for marker in BANNED_RELEASE_MARKERS):
                _fail(f"coding output contains a prohibited production or final-corpus marker: {path.name}")
        source_index = _read_release_csv(release_root / "source_index.csv", SOURCE_INDEX_HEADER)
        roster = {row["draft_key"] for row in source_index}
        if len(source_index) != ROSTER_SIZE or len(roster) != ROSTER_SIZE:
            _fail("release roster is not a 75-key draft roster")
        taxonomy = json.loads(
            (root / "paper/data/taxonomy.json").read_text(encoding="utf-8")
        )
        evidence = _read_release_csv(output / "evidence.csv", EVIDENCE_HEADER)
        if len(evidence) != ROSTER_SIZE or {row["cite_key"] for row in evidence} != roster:
            _fail("coding evidence must contain exactly the 75 draft keys")
        _validate_evidence_rows(evidence, taxonomy)
        optional = (
            ("claims.csv", CLAIMS_HEADER, "cite_keys", True),
            ("metrics.csv", METRICS_HEADER, "cite_keys", True),
            ("simulators.csv", SIMULATORS_HEADER, "cite_key", False),
        )
        for filename, header, references_field, plural in optional:
            path = output / filename
            if not path.exists():
                continue
            rows = _read_release_csv(path, header)
            if filename == "claims.csv":
                for row in rows:
                    status = row["evidence_status"]
                    if status and status not in taxonomy["evidence_status"]:
                        _fail("claims.csv has an invalid evidence_status")
            _validate_references(
                filename, rows, roster, references_field=references_field, plural=plural
            )
    except DraftValidationError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DraftValidationError(str(exc)) from exc


def _validate_deterministic_payloads(
    root: Path,
    release: Path,
    source_index: list[dict[str, str]],
    candidates: list[dict[str, str]],
    batches: dict[str, list[dict[str, str]]],
) -> None:
    expected = _release_payloads(source_index, candidates, batches)
    expected_names = set(expected) | {"release_manifest.csv", "SHA256SUMS"}
    actual_names = {path.name for path in release.iterdir()}
    if actual_names != expected_names:
        _fail("release has missing or unexpected artifacts")
    for name, payload in expected.items():
        if name in {"evidence_template.csv"}:
            continue
        if _regular_bytes(release / name, label="generated artifact") != payload:
            _fail(f"generated artifact is not deterministic: {name}")


def validate_release(
    *,
    repository_root: Path,
    release: Path,
    evidence_archive: Path,
    c0110_packet_bytes: Path,
) -> None:
    try:
        root = repository_root.resolve(strict=True)
        release_root = _release_path(root, release)
        expected_archive = (root / Path(*SOURCE_ARCHIVE_RELATIVE.parts)).resolve(strict=True)
        if evidence_archive.resolve(strict=True) != expected_archive:
            _fail("evidence archive must be supplied explicitly at the approved v8 location")
        expected_c0110 = (root / Path(*C0110_STAGED_RELATIVE.parts)).resolve(strict=True)
        if c0110_packet_bytes.resolve(strict=True) != expected_c0110:
            _fail("C0110 requires its exact frozen calibration packet location")
        _validate_nonfinal_text(release_root)
        source_index, candidates, batches = _validate_roster(
            root, release_root, evidence_archive, c0110_packet_bytes
        )
        _verify_manifest_and_sums(
            root, release_root, source_index, candidates, batches
        )
        _validate_templates(release_root, source_index, batches, root)
        _validate_deterministic_payloads(root, release_root, source_index, candidates, batches)
    except DraftValidationError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DraftValidationError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("paper/data/screening_work/v8/pass2_drafts/v1"),
    )
    parser.add_argument("--evidence-archive", type=Path, required=True)
    parser.add_argument("--c0110-packet-bytes", type=Path, required=True)
    parser.add_argument("--coding-output", type=Path)
    arguments = parser.parse_args(argv)
    validate_release(
        repository_root=arguments.repository_root,
        release=arguments.release,
        evidence_archive=arguments.evidence_archive,
        c0110_packet_bytes=arguments.c0110_packet_bytes,
    )
    if arguments.coding_output is not None:
        validate_coding_output(
            repository_root=arguments.repository_root,
            release=arguments.release,
            coding_output=arguments.coding_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
