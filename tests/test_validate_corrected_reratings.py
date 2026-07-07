from __future__ import annotations

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
