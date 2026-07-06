from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from paper.scripts.prepare_pass2_draft import prepare_release
from paper.scripts.validate_pass2_draft import DraftValidationError, validate_release


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "paper/data/source_archive/v8"
C0110_STAGED = (
    ROOT
    / "paper/data/screening_staging/v8/calibration"
    / "screening-02-260efd3e5c074756703b061e28ca3f23/v1/evidence"
    / "C0110/primary-report/C0110.pdf"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prepare_and_validate_repository_v1_release(tmp_path: Path) -> None:
    release = tmp_path / "pass2_drafts/v1"

    prepare_release(
        repository_root=ROOT,
        output=release,
        evidence_archive=ARCHIVE,
        c0110_packet_bytes=C0110_STAGED,
    )

    source_index = read_csv(release / "source_index.csv")
    candidates = read_csv(release / "candidates.csv")
    evidence = read_csv(release / "evidence_template.csv")
    assert len(source_index) == len(candidates) == len(evidence) == 75
    assert len({row["draft_key"] for row in source_index}) == 75
    assert all(re.fullmatch(r"DRAFT_C\d{4}", row["draft_key"]) for row in source_index)
    assert all(row["cite_key"].startswith("DRAFT_C") for row in candidates)
    assert all(row["cite_key"].startswith("DRAFT_C") for row in evidence)
    batch_sizes = sorted(
        len(read_csv(release / f"primary-batch-{number:02d}.csv"))
        for number in range(1, 7)
    )
    assert batch_sizes == [12, 12, 12, 13, 13, 13]
    c0110 = next(row for row in source_index if row["source_candidate_id"] == "C0110")
    assert c0110["evidence_bytes_locator"].endswith("C0110/primary-report/C0110.pdf")
    assert c0110["evidence_bytes_sha256"] == hashlib.sha256(
        C0110_STAGED.read_bytes()
    ).hexdigest()
    c0143 = next(row for row in source_index if row["source_candidate_id"] == "C0143")
    assert c0143["canonical_cite_key"] == ""
    assert c0143["citation_activation_status"] == "blocked"
    assert "final corpus" not in (release / "DRAFT-NONFINAL.md").read_text(
        encoding="utf-8"
    ).lower()

    validate_release(
        repository_root=ROOT,
        release=release,
        evidence_archive=ARCHIVE,
        c0110_packet_bytes=C0110_STAGED,
    )

    result = subprocess.run(
        (
            "python",
            "paper/scripts/validate_pass2_draft.py",
            "--repository-root",
            ".",
            "--release",
            str(release),
            "--evidence-archive",
            str(ARCHIVE),
            "--c0110-packet-bytes",
            str(C0110_STAGED),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_validator_rejects_manifest_tampering(tmp_path: Path) -> None:
    release = tmp_path / "pass2_drafts/v1"
    prepare_release(
        repository_root=ROOT,
        output=release,
        evidence_archive=ARCHIVE,
        c0110_packet_bytes=C0110_STAGED,
    )
    candidates = release / "candidates.csv"
    candidates.write_text(candidates.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DraftValidationError, match="checksum"):
        validate_release(
            repository_root=ROOT,
            release=release,
            evidence_archive=ARCHIVE,
            c0110_packet_bytes=C0110_STAGED,
        )
