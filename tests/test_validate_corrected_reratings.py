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
