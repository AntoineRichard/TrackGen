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
class _BatchSpec:
    number: int
    role: str
    agent_id: str
    model: str
    reasoning_effort: str
    filename: str
    source_digest: str
    row_count: int


EXPECTED_BATCH_SPECS = (
    _BatchSpec(1, "pass2-primary-01", "019f3952-5cc3-7ac0-b45b-a85d09eb459c", "gpt-5.6-terra", "high", "trackgen-pass2-primary-01.csv", "53436028264e61d23f35c083ecc8cbed58836a3f8f1ea223a8cb7eb55872d507", 13),
    _BatchSpec(2, "pass2-primary-02", "019f3952-5cec-70e0-bf51-cf59f159aa53", "gpt-5.6-terra", "high", "trackgen-pass2-primary-02.csv", "498e2f3092a6591dfab7493a81367220c4acbdedffc3453fdf2e9c2fd9d5466c", 13),
    _BatchSpec(3, "pass2-primary-03", "019f3952-5dbf-7250-ad10-38411837bbda", "gpt-5.6-terra", "high", "trackgen-pass2-primary-03.csv", "752bed14ac9ad5be0b6513f09a77007ad53ec3e2e78ccef1fc0b9febf851ac16", 13),
    _BatchSpec(4, "pass2-primary-04", "019f3952-5d5b-7c23-9f4f-491aae449174", "gpt-5.6-terra", "high", "trackgen-pass2-primary-04.csv", "51dbccc2a60e7081ae9860f3676735ce572f18661c376574bd70bc6d971bd410", 12),
    _BatchSpec(5, "pass2-primary-05", "019f3952-5d2d-74f1-a2ed-f5dfa8a58dea", "gpt-5.6-terra", "high", "trackgen-pass2-primary-05.csv", "8fa47b1cbabbe446860e7a0f26e7cf337541ce144c30f10f18d4d7b4f03e1645", 12),
    _BatchSpec(6, "pass2-primary-06", "019f3952-5d94-7023-9bce-89acde705820", "gpt-5.6-terra", "high", "trackgen-pass2-primary-06.csv", "9c71b1d35b69d7fd35fdd18cbde434dc056fd066b9281626f8fb4930d91dc793", 12),
)

V2_EXPECTED_BATCH_SPECS = (
    _BatchSpec(1, "pass2-primary-01", "019f3952-5cc3-7ac0-b45b-a85d09eb459c", MODEL, REASONING_EFFORT, "trackgen-pass2-v2-primary-01.csv", "de5977bf359b55859bf77c2a0e9a807ab1213428bc29a69e260f2a12c7085724", 13),
    _BatchSpec(2, "pass2-primary-02", "019f3952-5cec-70e0-bf51-cf59f159aa53", MODEL, REASONING_EFFORT, "trackgen-pass2-v2-primary-02.csv", "48eb8c4f405ab2e0fc37100c974de6fcfffb2a66fab34bdddcb549a0b108c650", 13),
    _BatchSpec(3, "pass2-primary-03", "019f3952-5dbf-7250-ad10-38411837bbda", MODEL, REASONING_EFFORT, "trackgen-pass2-v2-primary-03.csv", "ebd978a182d64ec08dc88a8503feecd2eabd02ccf75c918e59c58dc4dbbdeb2f", 13),
    _BatchSpec(4, "pass2-primary-04", "019f3952-5d5b-7c23-9f4f-491aae449174", MODEL, REASONING_EFFORT, "trackgen-pass2-v2-primary-04.csv", "2e6faeb1acda510fab0aa05cf0e750f7d191c6ead376b141bf63a5df0a4e6a29", 12),
    _BatchSpec(5, "pass2-primary-05", "019f3952-5d2d-74f1-a2ed-f5dfa8a58dea", MODEL, REASONING_EFFORT, "trackgen-pass2-v2-primary-05.csv", "d5ee12d5cb17f76709b0db837d14c9c59de1294dafde3f68397c21ec18c2dab7", 12),
    _BatchSpec(6, "pass2-primary-06", "019f3952-5d94-7023-9bce-89acde705820", MODEL, REASONING_EFFORT, "trackgen-pass2-v2-primary-06.csv", "919ff91a683cce156782fdc670bb5aa6ae6ae1f3caa761efe8feca1bfdf72920", 12),
)
V2_BINDING_HEADER = ("binding", "bound_path", "bound_sha256", "purpose")
NORMALIZATION_HEADER = ("metric", "count")
V2_BINDINGS = (
    ("draft_release_manifest", "paper/data/screening_work/v8/pass2_drafts/v1/release_manifest.csv", "84b38a2c5069779b6b79a12f053205d9a8495174d532f380420ba5f97f6ca678", "immutable draft release artifact set"),
    ("draft_release_checksums", "paper/data/screening_work/v8/pass2_drafts/v1/SHA256SUMS", "c6a0550993ecb473321eabd5b6fcbc3428e41a4a598699ebd15ec11947de0203", "immutable draft release checksum set"),
    ("primary_v1_snapshot", "paper/data/screening_work/v8/pass2_coding/primary/v1/manifest/checksums.csv", "ee7bafa35203956779d6be6ac2bb17f5c9c35969db31de68ad12a3919d2da876", "primary v1 snapshot artifact set"),
    ("pilot_v1_codebook_v2", "paper/data/screening_work/v8/pass2_reliability/pilot-v1/CODEBOOK-v2.md", "2f39b955e455f618ef3104962f10e983a98b9ac25dfe8b340d62fd3673f1a4c4", "prospective pilot-v1 codebook v2"),
)
PRIMARY_V1_EVIDENCE = Path(
    "paper/data/screening_work/v8/pass2_coding/primary/v1/coding/evidence.csv"
)
OUTPUT_BY_VERSION = {
    "v1": Path("paper/data/screening_work/v8/pass2_coding/primary/v1"),
    "v2": Path("paper/data/screening_work/v8/pass2_coding/primary/v2"),
}

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


def _validate_batch_rows(
    *,
    rows: list[dict[str, str]],
    spec: _BatchSpec,
    release: Path,
    taxonomy: dict[str, list[str]],
) -> None:
    expected_keys = _read_release_assignment(release, spec.number)
    keys = [row["cite_key"] for row in rows]
    if len(rows) != spec.row_count or len(set(keys)) != len(keys):
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
    source_payloads: Mapping[int, bytes],
    batch_payloads: Mapping[int, bytes],
    evidence_payload: bytes,
    registry_payload: bytes,
    documents: Mapping[str, bytes],
    specs: Sequence[_BatchSpec],
    extra_artifacts: Mapping[str, tuple[str, bytes, int]] | None = None,
) -> bytes:
    rows = [
        {"artifact_type": "immutable_release", "path": "immutable-release/release_manifest.csv", "sha256": _sha256(_regular_bytes(release / "release_manifest.csv", label="release manifest")), "row_count": "NR"},
        {"artifact_type": "immutable_release", "path": "immutable-release/SHA256SUMS", "sha256": _sha256(_regular_bytes(release / "SHA256SUMS", label="release SHA256SUMS")), "row_count": "NR"},
        {"artifact_type": "coding_output", "path": "coding/evidence.csv", "sha256": _sha256(evidence_payload), "row_count": str(ROSTER_SIZE)},
        {"artifact_type": "registry", "path": "execution_registry.csv", "sha256": _sha256(registry_payload), "row_count": str(len(specs))},
    ]
    for spec in specs:
        rows.append({"artifact_type": "input_batch", "path": f"input/{spec.filename}", "sha256": _sha256(source_payloads[spec.number]), "row_count": str(spec.row_count)})
        rows.append({"artifact_type": "integrated_batch", "path": f"batches/pass2-primary-{spec.number:02d}.csv", "sha256": _sha256(batch_payloads[spec.number]), "row_count": str(spec.row_count)})
    for path, payload in documents.items():
        rows.append({"artifact_type": "documentation", "path": path, "sha256": _sha256(payload), "row_count": "NR"})
    for artifact_type, (path, payload, row_count) in (extra_artifacts or {}).items():
        rows.append({"artifact_type": artifact_type, "path": path, "sha256": _sha256(payload), "row_count": str(row_count)})
    return _csv_bytes(CHECKSUM_HEADER, sorted(rows, key=lambda row: (row["artifact_type"], row["path"])))


def _v2_documents() -> dict[str, bytes]:
    documents = _documentation()
    prompt = documents["FROZEN-CODER-PROMPT.md"].decode("utf-8")
    prompt += "\nThis v2 normalization applies the prospective `pilot-v1/CODEBOOK-v2.md` without changing the immutable draft release.\n"
    limitations = """# Procedural Limitations

This v2 snapshot is a same-six-primary-coder normalization of `primary/v1`, not independent or blind reliability. The coordinator-recorded role, agent identifier, model, and reasoning-effort metadata in `execution_registry.csv` is not provider-side immutable execution attestation.

The fixed `/tmp/trackgen-pass2-v2-primary-XX.csv` locations are procedural same-user isolation paths, not durable provenance locations. Their supplied SHA-256 digests, copied batch digests, bindings to the immutable draft release and primary v1 snapshot, and the prospective pilot-v1 `CODEBOOK-v2.md` are recorded here.

This remains non-final draft coding. It makes no prevalence, taxonomy, final count, or final projection claim. No final counts may be reported until fresh independent blind reliability is completed against the frozen v2 codebook. This snapshot does not alter the bound v1 artifacts.
"""
    readme = """# Pass-2 Primary Coding Snapshot v2

This no-clobber snapshot integrates the six fixed v2 primary batches against the immutable `pass2_drafts/v1` release. `batches/` retains each supplied CSV byte-for-byte; `coding/evidence.csv` is their deterministic 75-row merge sorted by `cite_key`.

The same six primary coders normalized the v1 coding under the prospective `pass2_reliability/pilot-v1/CODEBOOK-v2.md`. This is not independent or blind reliability. `bindings.csv` freezes the exact draft-release, primary-v1, and codebook artifacts used; `normalization_summary.csv` deterministically compares the v2 and primary-v1 evidence rows and cells.

`execution_registry.csv` records coordinator-supplied coder metadata and source/output digests. `manifest/checksums.csv` records every integrated batch, generated artifact, documentation record, and immutable release binding. No final counts, prevalence, or taxonomy claims may be made until fresh blind reliability is completed.
"""
    return {
        "README.md": readme.encode("utf-8"),
        "FROZEN-CODER-PROMPT.md": prompt.encode("utf-8"),
        "PROCEDURAL-LIMITATIONS.md": limitations.encode("utf-8"),
    }


def _v2_binding_payload(repository_root: Path) -> bytes:
    rows = []
    for binding, relative_path, digest, purpose in V2_BINDINGS:
        payload = _regular_bytes(repository_root / relative_path, label=f"v2 binding {binding}")
        if _sha256(payload) != digest:
            _fail(f"v2 binding digest mismatch: {binding}")
        rows.append(
            {
                "binding": binding,
                "bound_path": relative_path,
                "bound_sha256": digest,
                "purpose": purpose,
            }
        )
    return _csv_bytes(V2_BINDING_HEADER, rows)


def _normalization_payload(repository_root: Path, evidence_rows: Sequence[dict[str, str]]) -> bytes:
    _, v1_rows = _read_evidence(
        repository_root / PRIMARY_V1_EVIDENCE, label="primary v1 evidence"
    )
    baseline = {row["cite_key"]: row for row in v1_rows}
    current_keys = {row["cite_key"] for row in evidence_rows}
    if len(v1_rows) != ROSTER_SIZE or len(baseline) != ROSTER_SIZE or current_keys != set(baseline):
        _fail("primary v1 evidence roster mismatch")
    changed_rows = 0
    changed_cells = 0
    for row in evidence_rows:
        prior = baseline[row["cite_key"]]
        differences = sum(row[field] != prior[field] for field in EVIDENCE_HEADER)
        changed_rows += bool(differences)
        changed_cells += differences
    return _csv_bytes(
        NORMALIZATION_HEADER,
        (
            {"metric": "changed_rows", "count": str(changed_rows)},
            {"metric": "changed_fields", "count": str(changed_cells)},
        ),
    )


def _specs_for_version(version: str) -> Sequence[_BatchSpec]:
    if version == "v1":
        return EXPECTED_BATCH_SPECS
    if version == "v2":
        return V2_EXPECTED_BATCH_SPECS
    _fail(f"unsupported version: {version}")


def integrate_primary_batches(
    *,
    repository_root: Path,
    release: Path,
    output: Path,
    input_root: Path = Path("/tmp"),
    version: str = "v1",
) -> None:
    """Create one separate snapshot from the six fixed primary inputs."""
    specs = _specs_for_version(version)
    if output.exists() or output.is_symlink():
        _fail("output snapshot must not already exist")
    if release.is_symlink() or not release.is_dir():
        _fail("immutable release must be a real directory")
    if input_root.is_symlink() or not input_root.is_dir():
        _fail("input root must be a real directory")
    try:
        root = repository_root.resolve(strict=True)
        release_root = release.resolve(strict=True)
        release_root.relative_to(root)
        taxonomy = json.loads((root / "paper/data/taxonomy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IntegrationError(str(exc)) from exc
    if release_root.parts[-2:] != ("pass2_drafts", "v1"):
        _fail("release must be the immutable pass2_drafts/v1 directory")

    source_payloads: dict[int, bytes] = {}
    batch_rows: dict[int, list[dict[str, str]]] = {}
    identities: set[tuple[int, int]] = set()
    for spec in specs:
        source = input_root / spec.filename
        try:
            if source.is_symlink() or not source.is_file():
                _fail(f"batch input {spec.number} must be a regular non-symlink file")
            stat = source.stat()
        except OSError as exc:
            raise IntegrationError(f"unable to inspect batch input {spec.number}") from exc
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            _fail("batch inputs must not share a hard-link alias")
        identities.add(identity)
        payload, rows = _read_evidence(source, label=f"batch input {spec.number}")
        if _sha256(payload) != spec.source_digest:
            _fail(f"batch {spec.number}: digest mismatch")
        _validate_batch_rows(rows=rows, spec=spec, release=release_root, taxonomy=taxonomy)
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
    registry_rows = [
        {
            "role": spec.role,
            "agent_id": spec.agent_id,
            "model": spec.model,
            "reasoning_effort": spec.reasoning_effort,
            "human_role": "NR",
            "row_count": str(spec.row_count),
            "source_input_filename": spec.filename,
            "source_input_sha256": spec.source_digest,
            "integrated_batch_filename": f"batches/pass2-primary-{spec.number:02d}.csv",
            "integrated_batch_sha256": _sha256(source_payloads[spec.number]),
        }
        for spec in specs
    ]
    registry_payload = _csv_bytes(REGISTRY_HEADER, registry_rows)
    documents = _documentation() if version == "v1" else _v2_documents()
    extra_artifacts: dict[str, tuple[str, bytes, int]] = {}
    if version == "v2":
        bindings_payload = _v2_binding_payload(root)
        normalization_payload = _normalization_payload(root, evidence_rows)
        extra_artifacts = {
            "bindings": ("bindings.csv", bindings_payload, len(V2_BINDINGS)),
            "normalization": ("normalization_summary.csv", normalization_payload, 2),
        }
    checksums_payload = _checksums(
        release=release_root,
        source_payloads=source_payloads,
        batch_payloads=source_payloads,
        evidence_payload=evidence_payload,
        registry_payload=registry_payload,
        documents=documents,
        specs=specs,
        extra_artifacts=extra_artifacts,
    )

    output.mkdir(parents=True)
    (output / "batches").mkdir()
    (output / "coding").mkdir()
    (output / "manifest").mkdir()
    for spec in specs:
        (output / f"batches/pass2-primary-{spec.number:02d}.csv").write_bytes(source_payloads[spec.number])
    (output / "coding/evidence.csv").write_bytes(evidence_payload)
    (output / "execution_registry.csv").write_bytes(registry_payload)
    for path, payload in documents.items():
        (output / path).write_bytes(payload)
    for _, (artifact_path, payload, _) in extra_artifacts.items():
        (output / artifact_path).write_bytes(payload)
    (output / "manifest/checksums.csv").write_bytes(checksums_payload)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    arguments = parser.parse_args(argv)
    try:
        integrate_primary_batches(
            repository_root=Path.cwd(),
            release=Path("paper/data/screening_work/v8/pass2_drafts/v1"),
            output=OUTPUT_BY_VERSION[arguments.version],
            version=arguments.version,
        )
    except IntegrationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
