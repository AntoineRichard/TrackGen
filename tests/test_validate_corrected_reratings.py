from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from paper.scripts.validate_corrected_reratings import (
    CorrectedReratingValidationError,
    validate_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "paper/data/screening_work/v8/corrected_reratings/v1"


def test_repository_corrected_rerating_snapshot_validates() -> None:
    validate_snapshot(repository_root=ROOT, snapshot=SNAPSHOT)


def test_validator_rejects_raw_input_that_differs_from_normalized_rating(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "corrected_reratings/v1"
    shutil.copytree(SNAPSHOT, snapshot)
    input_path = snapshot / "inputs/trackgen-corrected-rerating-C0046-A.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["criterion"] = "exclude-insufficient-detail"
    input_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CorrectedReratingValidationError, match="raw input"):
        validate_snapshot(repository_root=ROOT, snapshot=snapshot)


def test_validator_rejects_duplicate_execution_registry_key(tmp_path: Path) -> None:
    snapshot = tmp_path / "corrected_reratings/v1"
    shutil.copytree(SNAPSHOT, snapshot)
    registry_path = snapshot / "execution_registry.csv"
    with registry_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    rows[-1] = dict(rows[-2])
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    registry_digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    manifest_path = snapshot / "manifest/checksums.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, strict=True))
    next(
        row for row in manifest_rows if row["path"] == "execution_registry.csv"
    )["sha256"] = registry_digest
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_rows[0],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sums_path = snapshot / "SHA256SUMS"
    sums = {
        path: digest
        for digest, path in (
            line.split("  ", 1)
            for line in sums_path.read_text(encoding="ascii").splitlines()
        )
    }
    sums["execution_registry.csv"] = registry_digest
    sums["manifest/checksums.csv"] = manifest_digest
    sums_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sums.items()),
        encoding="ascii",
    )

    with pytest.raises(CorrectedReratingValidationError, match="registry roster"):
        validate_snapshot(repository_root=ROOT, snapshot=snapshot)


def test_validator_rejects_tampered_old_assignment_provenance(
    tmp_path: Path,
) -> None:
    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=rows[0],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    repository_root = tmp_path / "repository"
    snapshot = (
        repository_root
        / "paper/data/screening_work/v8/corrected_reratings/v1"
    )
    shutil.copytree(SNAPSHOT, snapshot)
    required_repository_files = (
        "paper/data/screening_work/v8/protocol.md",
        "paper/data/screening_results/calibration/v8/manifest.csv",
        "paper/data/screening_results/calibration/v8/screening-03.csv",
        "paper/data/screening_results/calibration/v8/screening-04.csv",
        "paper/data/screening_results/calibration/v8/screening-05.csv",
        "paper/data/source_archive/v8/C0046/metadrive_composing_diverse_driving_scenarios_arxiv_2109.12674v3.pdf",
        "paper/data/source_archive/v8/C0173/bayesrace-pmlr155-jain21b-corrected.pdf",
    )
    for relative_path in required_repository_files:
        destination = repository_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)

    result_path = (
        repository_root
        / "paper/data/screening_results/calibration/v8/screening-03.csv"
    )
    with result_path.open(encoding="utf-8", newline="") as handle:
        result_rows = list(csv.DictReader(handle, strict=True))
    next(
        row for row in result_rows if row["assignment_id"] == "A-C0046-03"
    )["evidence_sha256"] = "0" * 64
    write_rows(result_path, result_rows)

    calibration_manifest = (
        repository_root
        / "paper/data/screening_results/calibration/v8/manifest.csv"
    )
    with calibration_manifest.open(encoding="utf-8", newline="") as handle:
        calibration_rows = list(csv.DictReader(handle, strict=True))
    next(
        row for row in calibration_rows if row["batch_id"] == "screening-03"
    )["result_file_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    write_rows(calibration_manifest, calibration_rows)

    bindings_path = snapshot / "bindings.csv"
    with bindings_path.open(encoding="utf-8", newline="") as handle:
        binding_rows = list(csv.DictReader(handle, strict=True))
    next(
        row for row in binding_rows if row["binding"] == "calibration_manifest"
    )["bound_sha256"] = hashlib.sha256(
        calibration_manifest.read_bytes()
    ).hexdigest()
    write_rows(bindings_path, binding_rows)

    bindings_digest = hashlib.sha256(bindings_path.read_bytes()).hexdigest()
    checksum_manifest = snapshot / "manifest/checksums.csv"
    with checksum_manifest.open(encoding="utf-8", newline="") as handle:
        checksum_rows = list(csv.DictReader(handle, strict=True))
    next(row for row in checksum_rows if row["path"] == "bindings.csv")[
        "sha256"
    ] = bindings_digest
    write_rows(checksum_manifest, checksum_rows)

    checksum_digest = hashlib.sha256(checksum_manifest.read_bytes()).hexdigest()
    sums_path = snapshot / "SHA256SUMS"
    sums = {
        path: digest
        for digest, path in (
            line.split("  ", 1)
            for line in sums_path.read_text(encoding="ascii").splitlines()
        )
    }
    sums["bindings.csv"] = bindings_digest
    sums["manifest/checksums.csv"] = checksum_digest
    sums_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sums.items()),
        encoding="ascii",
    )

    with pytest.raises(
        CorrectedReratingValidationError,
        match="old assignment provenance",
    ):
        validate_snapshot(repository_root=repository_root, snapshot=snapshot)
