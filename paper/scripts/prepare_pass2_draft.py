"""Prepare the isolated, non-final Pass-2 v1 coding release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


RELEASE_NAME = "v1"
ROSTER_SIZE = 75
DRAFT_KEY_PREFIX = "DRAFT_"
PRIMARY_BATCH_COUNT = 6
PRIMARY_RANK_PREFIX = b"trackgen-pass2-primary-v1\0"
ALLOWED_ACCESS = frozenset(
    {"full_text", "full_text_and_supplement", "official_documentation"}
)
SOURCE_ARCHIVE_RELATIVE = PurePosixPath("paper/data/source_archive/v8")
C0110_STAGED_RELATIVE = PurePosixPath(
    "paper/data/screening_staging/v8/calibration/"
    "screening-02-260efd3e5c074756703b061e28ca3f23/v1/evidence/"
    "C0110/primary-report/C0110.pdf"
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
EVIDENCE_HEADER = (
    "cite_key",
    "survey_evidence_tier",
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
CLAIMS_HEADER = (
    "claim_id",
    "section",
    "claim_text",
    "cite_keys",
    "evidence_status",
    "reviewer_notes",
)
METRICS_HEADER = (
    "metric_id",
    "layer",
    "name",
    "definition",
    "formula_or_procedure",
    "units",
    "direction",
    "domain",
    "requires_dynamics",
    "minimum_reporting",
    "cite_keys",
    "limitations",
)
SIMULATORS_HEADER = (
    "system",
    "cite_key",
    "domain",
    "input_representation",
    "export_format",
    "load_validation",
    "coordinate_frame",
    "units",
    "collision_geometry",
    "spawn_reset",
    "rl_interface",
    "oss_status",
    "evidence_locator",
)
CANDIDATES_HEADER = (*PACKET_FIELDS, "source_candidate_id", "canonical_cite_key", "citation_activation_status")
SOURCE_INDEX_HEADER = (
    "draft_key",
    "source_candidate_id",
    "canonical_cite_key",
    "citation_activation_status",
    "selection_basis",
    "sealed_assignment_ids",
    "draft_adjudication_status",
    "needs_accountable_author_review",
    "packet_phase",
    "packet_manifest_path",
    "packet_manifest_sha256",
    "packet_artifact_id",
    "packet_artifact_role",
    "packet_source_url",
    "evidence_version",
    "access_status",
    "evidence_sha256",
    "evidence_bytes_mode",
    "evidence_bytes_locator",
    "evidence_bytes_sha256",
    "primary_batch_id",
    "primary_rank_sha256",
)
RELEASE_MANIFEST_HEADER = ("record_type", "path", "sha256", "row_count")


class DraftPreparationError(ValueError):
    """The isolated draft cannot be prepared from the supplied inputs."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise DraftPreparationError(f"path must remain under repository root: {path}") from exc


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise DraftPreparationError(f"{label}: must be a regular non-symlink file")
        return path.read_bytes()
    except OSError as exc:
        raise DraftPreparationError(f"{label}: unable to read {path}") from exc


def _read_csv(path: Path, expected_header: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    payload = _regular_bytes(path, label="CSV input")
    if b"\r" in payload:
        raise DraftPreparationError(f"{path}: CSV must use LF line endings")
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DraftPreparationError(f"{path}: malformed UTF-8 CSV") from exc
    if not header or any(field is None or not field for field in header):
        raise DraftPreparationError(f"{path}: missing CSV header")
    if len(set(header)) != len(header):
        raise DraftPreparationError(f"{path}: duplicate CSV header")
    if expected_header is not None and header != expected_header:
        raise DraftPreparationError(f"{path}: header must be exactly {expected_header}")
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise DraftPreparationError(f"{path}:{row_number}: malformed CSV row")
    return rows


def _csv_bytes(header: tuple[str, ...], rows: Iterable[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _safe_local_filename(value: str, *, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        value == "NR"
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DraftPreparationError(f"{context}: unsafe local_filename")
    return path


def _draft_key(candidate_id: str) -> str:
    if len(candidate_id) != 5 or not candidate_id.startswith("C") or not candidate_id[1:].isdigit():
        raise DraftPreparationError(f"invalid source candidate id {candidate_id!r}")
    return f"{DRAFT_KEY_PREFIX}{candidate_id}"


def _required_input_paths(root: Path) -> list[Path]:
    fixed = [
        root / "paper/data/screening_work/v8/adjudication_drafts/adjudications.csv",
        root / "paper/data/screening_work/v8/adjudication_drafts/adjudication_workbook.csv",
        root / "paper/data/candidates.csv",
        root / "paper/data/bibliography.csv",
        root / "paper/data/taxonomy.json",
        root / "paper/data/screening_protocol.md",
    ]
    for phase in ("calibration", "main"):
        fixed.extend(sorted((root / f"paper/data/screening_results/{phase}/v8").glob("*")))
        fixed.extend(sorted((root / f"paper/data/screening_releases/{phase}/v8").rglob("*")))
    required = [path for path in fixed if not path.is_dir()]
    paths = [path for path in required if path.is_file()]
    if len(paths) != len(required):
        missing = [str(path) for path in required if not path.is_file()]
        raise DraftPreparationError(f"required authoritative inputs are missing: {missing}")
    return sorted(paths, key=lambda path: _relative(root, path))


def _load_sealed_results(root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    ratings: dict[str, list[dict[str, str]]] = defaultdict(list)
    phase_by_candidate: dict[str, str] = {}
    for phase in ("calibration", "main"):
        directory = root / f"paper/data/screening_results/{phase}/v8"
        for path in sorted(directory.glob("screening-*.csv")):
            for row in _read_csv(path):
                candidate_id = row.get("candidate_id", "")
                if not candidate_id or row.get("phase") != phase:
                    raise DraftPreparationError(f"{path}: invalid sealed result row")
                ratings[candidate_id].append(row)
                if candidate_id in phase_by_candidate and phase_by_candidate[candidate_id] != phase:
                    raise DraftPreparationError(f"{candidate_id}: appears in both sealed phases")
                phase_by_candidate[candidate_id] = phase
    for candidate_id, rows in ratings.items():
        if len(rows) != 2:
            raise DraftPreparationError(f"{candidate_id}: requires exactly two sealed duplicate ratings")
        if len({row["assignment_id"] for row in rows}) != 2:
            raise DraftPreparationError(f"{candidate_id}: duplicate sealed assignment id")
    return ratings, phase_by_candidate


def _load_adjudication(root: Path) -> dict[str, dict[str, str]]:
    workbook = _read_csv(root / "paper/data/screening_work/v8/adjudication_drafts/adjudication_workbook.csv")
    details = _read_csv(root / "paper/data/screening_work/v8/adjudication_drafts/adjudications.csv")
    detail_ids = {row.get("candidate_id", "") for row in details}
    indexed: dict[str, dict[str, str]] = {}
    for row in workbook:
        candidate_id = row.get("candidate_id", "")
        if not candidate_id or candidate_id in indexed or candidate_id not in detail_ids:
            raise DraftPreparationError("draft adjudication inputs are not one-to-one")
        if row.get("needs_accountable_author_review") not in {"true", "false"}:
            raise DraftPreparationError(f"{candidate_id}: invalid accountable-review flag")
        indexed[candidate_id] = row
    return indexed


def _load_packet_manifests(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    manifests: dict[str, dict[str, dict[str, str]]] = {}
    for phase in ("calibration", "main"):
        path = root / f"paper/data/screening_releases/{phase}/v8/evidence_packet_manifest.csv"
        rows = _read_csv(path)
        indexed: dict[str, dict[str, str]] = {}
        for row in rows:
            candidate_id = row.get("candidate_id", "")
            if not candidate_id or candidate_id in indexed:
                raise DraftPreparationError(f"{path}: duplicate or blank candidate id")
            indexed[candidate_id] = row
        manifests[phase] = indexed
    return manifests


def _load_candidates(root: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(root / "paper/data/candidates.csv")
    indexed = {row.get("candidate_id", ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise DraftPreparationError("candidate input has blank or duplicate ids")
    return indexed


def _primary_assignment(draft_key: str) -> tuple[str, str]:
    digest = _sha256(PRIMARY_RANK_PREFIX + draft_key.encode("utf-8"))
    return "", digest


def _build_rows(
    root: Path,
    evidence_archive: Path,
    c0110_packet_bytes: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    ratings, phase_by_candidate = _load_sealed_results(root)
    adjudication = _load_adjudication(root)
    packets = _load_packet_manifests(root)
    candidates = _load_candidates(root)
    sealed_included = {
        candidate_id
        for candidate_id, rows in ratings.items()
        if all(row.get("screening_status") == "included" for row in rows)
    }
    draft_included = {
        candidate_id
        for candidate_id, row in adjudication.items()
        if row.get("draft_status") == "included"
    }
    blocked = {
        candidate_id
        for candidate_id, row in adjudication.items()
        if row.get("needs_accountable_author_review") == "true"
    }
    roster = sorted((sealed_included | draft_included) - blocked)
    if len(roster) != ROSTER_SIZE:
        raise DraftPreparationError(f"expected {ROSTER_SIZE} draft sources, found {len(roster)}")
    archive_root = evidence_archive.resolve(strict=True)
    expected_c0110 = (root / Path(*C0110_STAGED_RELATIVE.parts)).resolve(strict=True)
    if c0110_packet_bytes.resolve(strict=True) != expected_c0110:
        raise DraftPreparationError("C0110 requires its exact frozen calibration packet location")
    ranked_keys = sorted(
        (_draft_key(candidate_id) for candidate_id in roster),
        key=lambda draft_key: (_primary_assignment(draft_key)[1], draft_key),
    )
    batch_by_key = {
        draft_key: f"primary-batch-{position % PRIMARY_BATCH_COUNT + 1:02d}"
        for position, draft_key in enumerate(ranked_keys)
    }

    source_index: list[dict[str, str]] = []
    draft_candidates: list[dict[str, str]] = []
    batches: dict[str, list[dict[str, str]]] = {
        f"primary-batch-{number:02d}": [] for number in range(1, PRIMARY_BATCH_COUNT + 1)
    }
    for candidate_id in roster:
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise DraftPreparationError(f"{candidate_id}: missing candidate metadata")
        phase = phase_by_candidate[candidate_id]
        packet = packets[phase].get(candidate_id)
        if packet is None:
            raise DraftPreparationError(f"{candidate_id}: missing {phase} packet manifest row")
        access_status = packet.get("access_status", "")
        digest = packet.get("evidence_sha256", "")
        if access_status not in ALLOWED_ACCESS or len(digest) != 64:
            raise DraftPreparationError(f"{candidate_id}: non-inspectable packet evidence")
        if candidate_id == "C0110":
            bytes_path = expected_c0110
            bytes_mode = "role_private_frozen_packet"
        else:
            local = _safe_local_filename(packet.get("local_filename", ""), context=candidate_id)
            bytes_path = (archive_root / Path(*local.parts)).resolve(strict=True)
            try:
                bytes_path.relative_to(archive_root)
            except ValueError as exc:
                raise DraftPreparationError(f"{candidate_id}: archive artifact escapes explicit root") from exc
            bytes_mode = "explicit_manifest_attested_archive"
        payload = _regular_bytes(bytes_path, label=f"{candidate_id} evidence bytes")
        if _sha256(payload) != digest:
            raise DraftPreparationError(f"{candidate_id}: packet evidence SHA-256 mismatch")
        draft_key = _draft_key(candidate_id)
        _, rank = _primary_assignment(draft_key)
        batch_id = batch_by_key[draft_key]
        canonical_key = candidate.get("cite_key", "")
        if candidate_id == "C0143":
            if canonical_key:
                raise DraftPreparationError("C0143 must remain without an approved canonical cite key")
            activation = "blocked"
        elif not canonical_key:
            raise DraftPreparationError(f"{candidate_id}: missing canonical cite key")
        else:
            activation = "not_activated"
        if canonical_key == "Peltomaki2022WassersteinGAN":
            raise DraftPreparationError("unsealed Peltomaki proposal must not be activated")
        packet_manifest = root / f"paper/data/screening_releases/{phase}/v8/evidence_packet_manifest.csv"
        source_index.append(
            {
                "draft_key": draft_key,
                "source_candidate_id": candidate_id,
                "canonical_cite_key": canonical_key,
                "citation_activation_status": activation,
                "selection_basis": "sealed_duplicate_included" if candidate_id in sealed_included else "draft_adjudication_included",
                "sealed_assignment_ids": ";".join(sorted(row["assignment_id"] for row in ratings[candidate_id])),
                "draft_adjudication_status": adjudication.get(candidate_id, {}).get("draft_status", ""),
                "needs_accountable_author_review": adjudication.get(candidate_id, {}).get("needs_accountable_author_review", "false"),
                "packet_phase": phase,
                "packet_manifest_path": _relative(root, packet_manifest),
                "packet_manifest_sha256": _sha256(_regular_bytes(packet_manifest, label="packet manifest")),
                "packet_artifact_id": packet.get("artifact_id", ""),
                "packet_artifact_role": packet.get("artifact_role", ""),
                "packet_source_url": packet.get("source_url", ""),
                "evidence_version": packet.get("evidence_version", ""),
                "access_status": access_status,
                "evidence_sha256": digest,
                "evidence_bytes_mode": bytes_mode,
                "evidence_bytes_locator": _relative(root, bytes_path),
                "evidence_bytes_sha256": _sha256(payload),
                "primary_batch_id": batch_id,
                "primary_rank_sha256": rank,
            }
        )
        draft_candidates.append(
            {
                "candidate_id": draft_key,
                "cite_key": draft_key,
                "title": candidate.get("title", ""),
                "authors": candidate.get("authors", ""),
                "year": candidate.get("year", ""),
                "venue": candidate.get("venue", ""),
                "doi": candidate.get("doi", ""),
                "url": candidate.get("url", ""),
                "source_type": candidate.get("source_type", ""),
                "source_candidate_id": candidate_id,
                "canonical_cite_key": canonical_key,
                "citation_activation_status": activation,
            }
        )
        batches[batch_id].append(dict.fromkeys(EVIDENCE_HEADER, "") | {"cite_key": draft_key})
    source_index.sort(key=lambda row: row["draft_key"])
    draft_candidates.sort(key=lambda row: row["cite_key"])
    for rows in batches.values():
        rows.sort(key=lambda row: row["cite_key"])
    counts = sorted(len(rows) for rows in batches.values())
    if counts != [12, 12, 12, 13, 13, 13]:
        raise DraftPreparationError(f"primary batch allocation is imbalanced: {counts}")
    return source_index, draft_candidates, batches


def _nonfinal_markdown() -> bytes:
    return b"""# Pass-2 Draft Release v1\n\nThis is an isolated, non-final Pass-2 coding release for 75 provisional sources. It is not a screening projection, citation activation set, or production data artifact.\n\n## Procedural limitations\n\n- Draft adjudications remain unsealed and can change after accountable-author review.\n- Draft keys are the only coding keys in this release. Canonical citation keys are retained only as inactive source metadata.\n- C0143 has no approved canonical cite key and is blocked from citation activation. The historical Peltomaki2022WassersteinGAN proposal is not used.\n- Evidence byte locators remain bound to their released packet manifests and SHA-256 values. C0110 is explicitly bound to its role-private frozen calibration packet because the attested archive path is absent.\n- This release supports qualitative coding preparation only. It must not be used to calculate or report prevalence.\n\n## Promotion conditions\n\nPromotion requires sealed adjudications, accountable-author disposition of every flagged candidate, authoritative byte verification, approved canonical citation keys, and a separately reviewed projection process.\n"""


def _coding_instructions() -> bytes:
    return b"""# Pass-2 Coding Instructions\n\nCode only with `DRAFT_C####` keys. The evidence template uses the exact `evidence.csv` header and intentionally has no `coder_id` column.\n\n- `survey_evidence_tier` is scalar. When evidence could support multiple tiers, use `core`, then `supporting`, then `contextual` precedence.\n- Controlled multi-label fields use semicolon-separated labels in the order listed by `paper/data/taxonomy.json`; the first `domain` label is the primary domain because reliability sampling stratifies on it.\n- Every non-`NR` analytical field requires a source-native, field-addressable locator in `evidence_locator`, written as `field_name=locator` entries separated by semicolons. Use page, section, table, figure, algorithm, appendix, repository path and lines, or stable documentation anchors.\n- `supporting` rows may only state fixed-course properties that are directly established by the source and mapped by the protocol. `contextual` rows may only support field, terminology, or literature-gap context. Neither tier establishes a source-native course-generation method.\n- `asset_status` is prospectively controlled with the same scalar vocabulary as `code_status`: `official_open`, `unofficial_open`, `closed`, `not_found`, or `not_applicable`.\n\nMutable coding output uses `evidence.csv` with exactly these 75 draft keys. `claims.csv`, `metrics.csv`, and `simulators.csv` are optional until populated, but each must use its exact release template header and may reference only draft keys. Validate an output directory with `validate_pass2_draft.py --coding-output`; this checks output only and never rewrites the immutable release or its checksums.\n\nLeave all analytical fields blank while a row remains a template row.\n"""


def _release_payloads(
    source_index: list[dict[str, str]],
    candidates: list[dict[str, str]],
    batches: dict[str, list[dict[str, str]]],
) -> dict[str, bytes]:
    blank_evidence = [dict.fromkeys(EVIDENCE_HEADER, "") | {"cite_key": row["draft_key"]} for row in source_index]
    payloads = {
        "DRAFT-NONFINAL.md": _nonfinal_markdown(),
        "CODING-INSTRUCTIONS.md": _coding_instructions(),
        "source_index.csv": _csv_bytes(SOURCE_INDEX_HEADER, source_index),
        "candidates.csv": _csv_bytes(CANDIDATES_HEADER, candidates),
        "evidence_template.csv": _csv_bytes(EVIDENCE_HEADER, blank_evidence),
        "claims_template.csv": _csv_bytes(CLAIMS_HEADER, ()),
        "metrics_template.csv": _csv_bytes(METRICS_HEADER, ()),
        "simulators_template.csv": _csv_bytes(SIMULATORS_HEADER, ()),
    }
    for batch_id, rows in batches.items():
        payloads[f"{batch_id}.csv"] = _csv_bytes(EVIDENCE_HEADER, rows)
    return payloads


def prepare_release(
    *,
    repository_root: Path,
    output: Path,
    evidence_archive: Path,
    c0110_packet_bytes: Path,
) -> None:
    root = repository_root.resolve(strict=True)
    expected_suffix = Path("pass2_drafts") / RELEASE_NAME
    if output.parts[-2:] != expected_suffix.parts:
        raise DraftPreparationError(f"output must end with {expected_suffix}")
    if output.exists() or output.is_symlink():
        raise DraftPreparationError("refusing to overwrite or alias an existing draft release")
    source_index, candidates, batches = _build_rows(root, evidence_archive, c0110_packet_bytes)
    payloads = _release_payloads(source_index, candidates, batches)
    inputs = _required_input_paths(root)
    manifest_rows = [
        {"record_type": "generated", "path": path, "sha256": _sha256(payload), "row_count": "NR"}
        for path, payload in sorted(payloads.items())
    ]
    manifest_rows.extend(
        {
            "record_type": "input",
            "path": _relative(root, path),
            "sha256": _sha256(_regular_bytes(path, label="authoritative input")),
            "row_count": "NR",
        }
        for path in inputs
    )
    manifest_rows.sort(key=lambda row: (row["record_type"], row["path"]))
    manifest_payload = _csv_bytes(RELEASE_MANIFEST_HEADER, manifest_rows)
    payloads["release_manifest.csv"] = manifest_payload
    sums = [
        f"{_sha256(payload)}  generated/{path}\n"
        for path, payload in sorted(payloads.items())
    ]
    sums.extend(
        f"{row['sha256']}  input/{row['path']}\n"
        for row in manifest_rows
        if row["record_type"] == "input"
    )
    payloads["SHA256SUMS"] = "".join(sorted(sums)).encode("ascii")
    output.mkdir(parents=True, exist_ok=False)
    for path, payload in sorted(payloads.items()):
        destination = output / path
        destination.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/data/screening_work/v8/pass2_drafts/v1"),
    )
    parser.add_argument("--evidence-archive", type=Path, required=True)
    parser.add_argument("--c0110-packet-bytes", type=Path, required=True)
    arguments = parser.parse_args(argv)
    prepare_release(
        repository_root=arguments.repository_root,
        output=arguments.output,
        evidence_archive=arguments.evidence_archive,
        c0110_packet_bytes=arguments.c0110_packet_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
