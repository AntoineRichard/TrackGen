from __future__ import annotations

import csv
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from paper.scripts.prepare_pass2_draft import (
    DraftPreparationError,
    _load_adjudication,
    prepare_release,
)
from paper.scripts.validate_pass2_draft import (
    DraftValidationError,
    validate_coding_output,
    validate_release,
)


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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_test_release(tmp_path: Path) -> Path:
    release = tmp_path / "pass2_drafts/v1"
    prepare_release(
        repository_root=ROOT,
        output=release,
        evidence_archive=ARCHIVE,
        c0110_packet_bytes=C0110_STAGED,
    )
    return release


def adjudication_test_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repository"
    directory = root / "paper/data/screening_work/v8/adjudication_drafts"
    directory.mkdir(parents=True)
    details = directory / "adjudications.csv"
    workbook = directory / "adjudication_workbook.csv"
    shutil.copyfile(
        ROOT / "paper/data/screening_work/v8/adjudication_drafts/adjudications.csv",
        details,
    )
    shutil.copyfile(
        ROOT / "paper/data/screening_work/v8/adjudication_drafts/adjudication_workbook.csv",
        workbook,
    )
    return root, details, workbook


@pytest.mark.parametrize("invalid_id", ["", "duplicate"])
def test_load_adjudication_rejects_blank_or_duplicate_detail_candidate_ids(
    tmp_path: Path, invalid_id: str
) -> None:
    root, details, _ = adjudication_test_root(tmp_path)
    rows = read_csv(details)
    if invalid_id == "":
        rows[0]["candidate_id"] = ""
    else:
        rows.append(dict(rows[0]))
    write_csv(details, rows)

    with pytest.raises(DraftPreparationError, match="detail candidate IDs"):
        _load_adjudication(root)


def test_load_adjudication_requires_exact_candidate_id_set_equality(
    tmp_path: Path,
) -> None:
    root, details, _ = adjudication_test_root(tmp_path)
    rows = read_csv(details)
    write_csv(details, rows[1:])

    with pytest.raises(DraftPreparationError, match="candidate-ID sets"):
        _load_adjudication(root)


@pytest.mark.parametrize("aliased_input", ["archive", "c0110"])
def test_prepare_rejects_symlinked_evidence_inputs(
    tmp_path: Path, aliased_input: str
) -> None:
    archive = ARCHIVE
    c0110 = C0110_STAGED
    alias = tmp_path / aliased_input
    alias.symlink_to(archive if aliased_input == "archive" else c0110)
    if aliased_input == "archive":
        archive = alias
    else:
        c0110 = alias

    with pytest.raises(DraftPreparationError, match="symlink"):
        prepare_release(
            repository_root=ROOT,
            output=tmp_path / "pass2_drafts/v1",
            evidence_archive=archive,
            c0110_packet_bytes=c0110,
        )


@pytest.mark.parametrize("aliased_input", ["archive", "c0110"])
def test_validator_rejects_symlinked_evidence_inputs(
    tmp_path: Path, aliased_input: str
) -> None:
    release = prepare_test_release(tmp_path)
    archive = ARCHIVE
    c0110 = C0110_STAGED
    alias = tmp_path / f"{aliased_input}-alias"
    alias.symlink_to(archive if aliased_input == "archive" else c0110)
    if aliased_input == "archive":
        archive = alias
    else:
        c0110 = alias

    with pytest.raises(DraftValidationError, match="symlink"):
        validate_release(
            repository_root=ROOT,
            release=release,
            evidence_archive=archive,
            c0110_packet_bytes=c0110,
        )


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
    release = prepare_test_release(tmp_path)
    candidates = release / "candidates.csv"
    candidates.write_text(candidates.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DraftValidationError, match="checksum"):
        validate_release(
            repository_root=ROOT,
            release=release,
            evidence_archive=ARCHIVE,
            c0110_packet_bytes=C0110_STAGED,
        )


def test_validator_rejects_incomplete_manifest_before_checksum_check(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    manifest = release / "release_manifest.csv"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join((lines[0], *lines[2:])) + "\n", encoding="utf-8")

    with pytest.raises(DraftValidationError, match="exact expected records"):
        validate_release(
            repository_root=ROOT,
            release=release,
            evidence_archive=ARCHIVE,
            c0110_packet_bytes=C0110_STAGED,
        )


def test_validator_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    manifest = release / "release_manifest.csv"
    rows = read_csv(manifest)
    rows[0]["path"] = "../candidates.csv"
    write_csv(manifest, rows)

    with pytest.raises(DraftValidationError, match="safe relative POSIX"):
        validate_release(
            repository_root=ROOT,
            release=release,
            evidence_archive=ARCHIVE,
            c0110_packet_bytes=C0110_STAGED,
        )


def test_validator_rejects_final_projection_marker(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    notice = release / "DRAFT-NONFINAL.md"
    notice.write_text(notice.read_text(encoding="utf-8") + "\nfinal screening projection\n", encoding="utf-8")

    with pytest.raises(DraftValidationError, match="prohibited"):
        validate_release(
            repository_root=ROOT,
            release=release,
            evidence_archive=ARCHIVE,
            c0110_packet_bytes=C0110_STAGED,
        )


def test_coding_output_rejects_completed_evidence_without_field_locator(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    evidence = output / "evidence.csv"
    shutil.copyfile(release / "evidence_template.csv", evidence)
    rows = read_csv(evidence)
    rows[0]["survey_evidence_tier"] = "core"
    rows[0]["domain"] = "ground"
    rows[0]["evidence_locator"] = "PDF p. 1"
    write_csv(evidence, rows)

    with pytest.raises(DraftValidationError, match="field-addressable"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_accepts_initial_evidence_only_directory(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    shutil.copyfile(release / "evidence_template.csv", output / "evidence.csv")

    validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_rejects_prohibited_final_marker(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    evidence = output / "evidence.csv"
    shutil.copyfile(release / "evidence_template.csv", evidence)
    rows = read_csv(evidence)
    rows[0]["coding_notes"] = "final corpus"
    write_csv(evidence, rows)

    with pytest.raises(DraftValidationError, match="prohibited"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_rejects_multiple_simulator_cite_keys(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    shutil.copyfile(release / "evidence_template.csv", output / "evidence.csv")
    simulators = read_csv(release / "simulators_template.csv")
    simulators.append(
        {
            "system": "draft-system",
            "cite_key": "DRAFT_C0006;DRAFT_C0008",
            "domain": "ground",
            "input_representation": "NR",
            "export_format": "NR",
            "load_validation": "NR",
            "coordinate_frame": "NR",
            "units": "NR",
            "collision_geometry": "NR",
            "spawn_reset": "NR",
            "rl_interface": "NR",
            "oss_status": "NR",
            "evidence_locator": "NR",
        }
    )
    write_csv(output / "simulators.csv", simulators)

    with pytest.raises(DraftValidationError, match="exactly one"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_rejects_invalid_simulator_domain(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    shutil.copyfile(release / "evidence_template.csv", output / "evidence.csv")
    simulators = read_csv(release / "simulators_template.csv")
    simulators.append(
        {
            "system": "draft-system",
            "cite_key": "DRAFT_C0006",
            "domain": "gruond",
            "input_representation": "NR",
            "export_format": "NR",
            "load_validation": "NR",
            "coordinate_frame": "NR",
            "units": "NR",
            "collision_geometry": "NR",
            "spawn_reset": "NR",
            "rl_interface": "NR",
            "oss_status": "NR",
            "evidence_locator": "NR",
        }
    )
    write_csv(output / "simulators.csv", simulators)

    with pytest.raises(DraftValidationError, match="simulators.csv.*domain"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_accepts_sole_nr_controlled_evidence_values(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    evidence = output / "evidence.csv"
    shutil.copyfile(release / "evidence_template.csv", evidence)
    rows = read_csv(evidence)
    for field in (
        "survey_evidence_tier",
        "domain",
        "course_object",
        "representation_family",
        "generator_family",
        "generation_role",
        "validity_strategy",
        "code_status",
        "asset_status",
    ):
        rows[0][field] = "NR"
    write_csv(evidence, rows)

    validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_rejects_mixed_nr_controlled_evidence_value(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    evidence = output / "evidence.csv"
    shutil.copyfile(release / "evidence_template.csv", evidence)
    rows = read_csv(evidence)
    rows[0]["survey_evidence_tier"] = "NR;core"
    write_csv(evidence, rows)

    with pytest.raises(DraftValidationError, match="sole NR"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


@pytest.mark.parametrize("value", ["NR;", ";NR", "NR;;", "ground;;aerial"])
def test_coding_output_rejects_empty_controlled_field_separator_elements(
    tmp_path: Path, value: str
) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    evidence = output / "evidence.csv"
    shutil.copyfile(release / "evidence_template.csv", evidence)
    rows = read_csv(evidence)
    rows[0]["survey_evidence_tier"] = "NR"
    rows[0]["domain"] = value
    write_csv(evidence, rows)

    with pytest.raises(DraftValidationError, match="empty separator"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)


def test_coding_output_rejects_reference_outside_draft_roster(tmp_path: Path) -> None:
    release = prepare_test_release(tmp_path)
    output = tmp_path / "coding"
    output.mkdir()
    shutil.copyfile(release / "evidence_template.csv", output / "evidence.csv")
    claims = read_csv(release / "claims_template.csv")
    claims.append(
        {
            "claim_id": "CL-1",
            "section": "results",
            "claim_text": "Draft-only claim.",
            "cite_keys": "C0001",
            "evidence_status": "direct",
            "reviewer_notes": "NR",
        }
    )
    write_csv(output / "claims.csv", claims)

    with pytest.raises(DraftValidationError, match="outside the draft roster"):
        validate_coding_output(repository_root=ROOT, release=release, coding_output=output)
