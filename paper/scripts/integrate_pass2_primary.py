"""Integrate the six completed primary Pass-2 coding batches into a snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.scripts.prepare_pass2_draft import (
    EVIDENCE_HEADER,
    ROSTER_SIZE,
    SOURCE_INDEX_HEADER,
    _csv_bytes,
    _read_csv,
)
from paper.scripts.validate_pass2_draft import DraftValidationError, _validate_evidence_rows


MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "high"
REGISTRY_HEADER = (
    "role",
    "agent_id",
    "model",
    "reasoning_effort",
    "human_role",
    "row_count",
    "source_input_filename",
    "source_input_sha256",
    "integrated_batch_filename",
    "integrated_batch_sha256",
)
CHECKSUM_HEADER = ("artifact_type", "path", "sha256", "row_count")
ANALYTICAL_FIELDS = (
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


class IntegrationError(ValueError):
    """The completed primary batches cannot form the requested snapshot."""


@dataclass(frozen=True)
class BatchSpec:
    number: int
    role: str
    agent_id: str
    source_digest: str


BATCH_SPECS = (
    BatchSpec(1, "pass2-primary-01", "019f3952-5cc3-7ac0-b45b-a85d09eb459c", "53436028264e61d23f35c083ecc8cbed58836a3f8f1ea223a8cb7eb55872d507"),
    BatchSpec(2, "pass2-primary-02", "019f3952-5cec-70e0-bf51-cf59f159aa53", "498e2f3092a6591dfab7493a81367220c4acbdedffc3453fdf2e9c2fd9d5466c"),
    BatchSpec(3, "pass2-primary-03", "019f3952-5dbf-7250-ad10-38411837bbda", "752bed14ac9ad5be0b6513f09a77007ad53ec3e2e78ccef1fc0b9febf851ac16"),
    BatchSpec(4, "pass2-primary-04", "019f3952-5d5b-7c23-9f4f-491aae449174", "51dbccc2a60e7081ae9860f3676735ce572f18661c376574bd70bc6d971bd410"),
    BatchSpec(5, "pass2-primary-05", "019f3952-5d2d-74f1-a2ed-f5dfa8a58dea", "8fa47b1cbabbe446860e7a0f26e7cf337541ce144c30f10f18d4d7b4f03e1645"),
    BatchSpec(6, "pass2-primary-06", "019f3952-5d94-7023-9bce-89acde705820", "9c71b1d35b69d7fd35fdd18cbde434dc056fd066b9281626f8fb4930d91dc793"),
)


def _fail(message: str) -> None:
    raise IntegrationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            _fail(f"{label} must be a regular non-symlink file")
        return path.read_bytes()
    except OSError as exc:
        raise IntegrationError(f"unable to read {label}: {path}") from exc


def _read_evidence(path: Path, *, label: str) -> tuple[bytes, list[dict[str, str]]]:
    payload = _regular_bytes(path, label=label)
    try:
        return payload, _read_csv(path, EVIDENCE_HEADER)
    except ValueError as exc:
        raise IntegrationError(str(exc)) from exc


def _read_release_assignment(release: Path, number: int | str) -> list[str]:
    path = release / f"primary-batch-{int(number):02d}.csv"
    _, rows = _read_evidence(path, label=f"release batch {number}")
    return [row["cite_key"] for row in rows]


def _release_roster(release: Path) -> set[str]:
    try:
        rows = _read_csv(release / "source_index.csv", SOURCE_INDEX_HEADER)
    except ValueError as exc:
        raise IntegrationError(str(exc)) from exc
    roster = {row["draft_key"] for row in rows}
    if len(rows) != ROSTER_SIZE or len(roster) != ROSTER_SIZE:
        _fail("release must expose exactly the 75 draft keys")
    return roster


def _validate_batch(
    *,
    source: Path,
    spec: BatchSpec,
    release: Path,
    taxonomy: dict[str, list[str]],
) -> tuple[bytes, list[dict[str, str]]]:
    payload, rows = _read_evidence(source, label=f"batch input {spec.number}")
    if _sha256(payload) != spec.source_digest:
        _fail(f"batch {spec.number}: digest mismatch")
    expected_keys = _read_release_assignment(release, spec.number)
    keys = [row["cite_key"] for row in rows]
    if len(rows) != len(expected_keys) or len(set(keys)) != len(keys):
        _fail(f"batch {spec.number}: duplicate or missing batch keys")
    if keys != expected_keys:
        _fail(f"batch {spec.number}: assignment or row order differs from its release batch")
    for row in rows:
        blank = [field for field in ANALYTICAL_FIELDS if not row[field].strip()]
        if blank:
            _fail(f"batch {spec.number}: blank analytical fields: {', '.join(blank)}")
    try:
        _validate_evidence_rows(rows, taxonomy)
    except DraftValidationError as exc:
        raise IntegrationError(str(exc)) from exc
    return payload, rows


def _documentation() -> dict[str, bytes]:
    prompt = """# Frozen Coder Prompt\n\nThis records the common instructions supplied to all six primary coding batches. No prelaunch timestamp is asserted.\n\nCode only with `DRAFT_C####` keys. The evidence template uses the exact `evidence.csv` header and intentionally has no `coder_id` column.\n\n- `survey_evidence_tier` is scalar. When evidence could support multiple tiers, use `core`, then `supporting`, then `contextual` precedence.\n- Controlled multi-label fields use semicolon-separated labels in the order listed by `paper/data/taxonomy.json`; the first `domain` label is the primary domain because reliability sampling stratifies on it.\n- Every non-`NR` analytical field requires a source-native, field-addressable locator in `evidence_locator`, written as `field_name=locator` entries separated by semicolons. Use page, section, table, figure, algorithm, appendix, repository path and lines, or stable documentation anchors.\n- `supporting` rows may only state fixed-course properties that are directly established by the source and mapped by the protocol. `contextual` rows may only support field, terminology, or literature-gap context. Neither tier establishes a source-native course-generation method.\n- `asset_status` is prospectively controlled with the same scalar vocabulary as `code_status`: `official_open`, `unofficial_open`, `closed`, `not_found`, or `not_applicable`.\n\nMutable coding output uses `evidence.csv` with exactly these 75 draft keys. `claims.csv`, `metrics.csv`, and `simulators.csv` are optional until populated, but each must use its exact release template header and may reference only draft keys. Validate an output directory with `validate_pass2_draft.py --coding-output`; this checks output only and never rewrites the immutable release or its checksums.\n\nLeave all analytical fields blank while a row remains a template row.\n"""
    limitations = """# Procedural Limitations\n\nThe coordinator-recorded role, agent identifier, model, and reasoning-effort metadata in `execution_registry.csv` is not provider-side immutable execution attestation. No prelaunch timestamp is asserted.\n\nThe `/tmp/trackgen-pass2-primary-*.csv` locations are procedural same-user isolation paths, not durable provenance locations. Their supplied SHA-256 digests, the copied batch digests, and the immutable draft-release checksums are recorded in this snapshot.\n\nThis is non-final draft coding. It makes no prevalence or final projection claim and does not alter `paper/data/screening_work/v8/pass2_drafts/v1`.\n"""
    readme = """# Pass-2 Primary Coding Snapshot\n\nThis no-clobber snapshot integrates the six completed primary Pass-2 coding batches against the immutable `pass2_drafts/v1` release. `batches/` retains the supplied CSV files byte-for-byte; `coding/evidence.csv` is their deterministic 75-row merge sorted by `cite_key`.\n\n`execution_registry.csv` records the coordinator-supplied roles, agent identifiers, model, reasoning effort, `human_role=NR`, row counts, and source/output digests. `manifest/checksums.csv` binds those outputs, documentation, and the immutable release manifest and `SHA256SUMS`.\n\nSee `PROCEDURAL-LIMITATIONS.md` for the provenance and non-final-use limits.\n"""
    return {
        "README.md": readme.encode("utf-8"),
        "FROZEN-CODER-PROMPT.md": prompt.encode("utf-8"),
        "PROCEDURAL-LIMITATIONS.md": limitations.encode("utf-8"),
    }


def _checksums(
    *,
    release: Path,
    source_payloads: Mapping[int | str, bytes],
    batch_payloads: Mapping[int | str, bytes],
    evidence_payload: bytes,
    registry_payload: bytes,
    documents: Mapping[str, bytes],
    batch_row_counts: Mapping[int | str, int],
) -> bytes:
    rows = [
        {"artifact_type": "immutable_release", "path": "immutable-release/release_manifest.csv", "sha256": _sha256(_regular_bytes(release / "release_manifest.csv", label="release manifest")), "row_count": "NR"},
        {"artifact_type": "immutable_release", "path": "immutable-release/SHA256SUMS", "sha256": _sha256(_regular_bytes(release / "SHA256SUMS", label="release SHA256SUMS")), "row_count": "NR"},
        {"artifact_type": "coding_output", "path": "coding/evidence.csv", "sha256": _sha256(evidence_payload), "row_count": str(ROSTER_SIZE)},
        {"artifact_type": "registry", "path": "execution_registry.csv", "sha256": _sha256(registry_payload), "row_count": str(len(BATCH_SPECS))},
    ]
    for number, payload in source_payloads.items():
        rows.append({"artifact_type": "input_batch", "path": f"input/trackgen-pass2-primary-{int(number):02d}.csv", "sha256": _sha256(payload), "row_count": str(batch_row_counts[number])})
    for number, payload in batch_payloads.items():
        rows.append({"artifact_type": "integrated_batch", "path": f"batches/pass2-primary-{int(number):02d}.csv", "sha256": _sha256(payload), "row_count": str(batch_row_counts[number])})
    for path, payload in documents.items():
        rows.append({"artifact_type": "documentation", "path": path, "sha256": _sha256(payload), "row_count": "NR"})
    return _csv_bytes(CHECKSUM_HEADER, sorted(rows, key=lambda row: (row["artifact_type"], row["path"])))


def integrate_primary_batches(
    *,
    repository_root: Path,
    release: Path,
    output: Path,
    batch_sources: Mapping[int | str, Path],
    batch_specs: Sequence[BatchSpec] = BATCH_SPECS,
) -> None:
    """Create one separate snapshot after validating all six completed inputs."""
    if output.exists() or output.is_symlink():
        _fail("output snapshot must not already exist")
    if release.is_symlink() or not release.is_dir():
        _fail("immutable release must be a real directory")
    try:
        root = repository_root.resolve(strict=True)
        release_root = release.resolve(strict=True)
        release_root.relative_to(root)
        taxonomy = json.loads((root / "paper/data/taxonomy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IntegrationError(str(exc)) from exc
    if release_root.parts[-2:] != ("pass2_drafts", "v1"):
        _fail("release must be the immutable pass2_drafts/v1 directory")
    if len(batch_specs) != 6 or {spec.number for spec in batch_specs} != set(range(1, 7)):
        _fail("integration requires exactly the six primary batch specifications")

    source_payloads: dict[int | str, bytes] = {}
    batch_rows: dict[int | str, list[dict[str, str]]] = {}
    for spec in sorted(batch_specs, key=lambda item: int(item.number)):
        source = batch_sources.get(spec.number)
        if source is None:
            _fail(f"batch {spec.number}: source input is required")
        payload, rows = _validate_batch(
            source=source, spec=spec, release=release_root, taxonomy=taxonomy
        )
        source_payloads[spec.number] = payload
        batch_rows[spec.number] = rows

    evidence_rows = sorted(
        (row for rows in batch_rows.values() for row in rows), key=lambda row: row["cite_key"]
    )
    roster = _release_roster(release_root)
    evidence_keys = [row["cite_key"] for row in evidence_rows]
    if len(evidence_rows) != ROSTER_SIZE or len(set(evidence_keys)) != ROSTER_SIZE or set(evidence_keys) != roster:
        _fail("integrated evidence has duplicate or missing draft keys")
    evidence_payload = _csv_bytes(EVIDENCE_HEADER, evidence_rows)
    batch_payloads = dict(source_payloads)
    registry_rows = [
        {
            "role": spec.role,
            "agent_id": spec.agent_id,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "human_role": "NR",
            "row_count": str(len(batch_rows[spec.number])),
            "source_input_filename": f"trackgen-pass2-primary-{int(spec.number):02d}.csv",
            "source_input_sha256": _sha256(source_payloads[spec.number]),
            "integrated_batch_filename": f"batches/pass2-primary-{int(spec.number):02d}.csv",
            "integrated_batch_sha256": _sha256(batch_payloads[spec.number]),
        }
        for spec in sorted(batch_specs, key=lambda item: int(item.number))
    ]
    registry_payload = _csv_bytes(REGISTRY_HEADER, registry_rows)
    documents = _documentation()
    checksums_payload = _checksums(
        release=release_root,
        source_payloads=source_payloads,
        batch_payloads=batch_payloads,
        evidence_payload=evidence_payload,
        registry_payload=registry_payload,
        documents=documents,
        batch_row_counts={number: len(rows) for number, rows in batch_rows.items()},
    )

    output.mkdir(parents=True)
    (output / "batches").mkdir()
    (output / "coding").mkdir()
    (output / "manifest").mkdir()
    for spec in sorted(batch_specs, key=lambda item: int(item.number)):
        (output / f"batches/pass2-primary-{int(spec.number):02d}.csv").write_bytes(batch_payloads[spec.number])
    (output / "coding/evidence.csv").write_bytes(evidence_payload)
    (output / "execution_registry.csv").write_bytes(registry_payload)
    for path, payload in documents.items():
        (output / path).write_bytes(payload)
    (output / "manifest/checksums.csv").write_bytes(checksums_payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release", type=Path, default=Path("paper/data/screening_work/v8/pass2_drafts/v1"))
    parser.add_argument("--output", type=Path, default=Path("paper/data/screening_work/v8/pass2_coding/primary/v1"))
    for number in range(1, 7):
        parser.add_argument(
            f"--batch-{number:02d}",
            type=Path,
            default=Path(f"/tmp/trackgen-pass2-primary-{number:02d}.csv"),
        )
    arguments = parser.parse_args(argv)
    try:
        integrate_primary_batches(
            repository_root=arguments.repository_root,
            release=arguments.release,
            output=arguments.output,
            batch_sources={number: getattr(arguments, f"batch_{number:02d}") for number in range(1, 7)},
        )
    except IntegrationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
