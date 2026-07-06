from __future__ import annotations

import csv
from pathlib import Path

import pytest

from paper.scripts.integrate_pass2_reliability import (
    INPUT_SPECS,
    ReliabilityIntegrationError,
    ReliabilityValidationError,
    integrate_reliability_pilot,
    validate_reliability_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path("/tmp")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_snapshot(tmp_path: Path) -> Path:
    output = tmp_path / "pass2_reliability/pilot-v1"
    integrate_reliability_pilot(
        repository_root=ROOT, output=output, input_root=INPUT_ROOT
    )
    return output


def test_builds_byte_exact_reliability_snapshot_and_validates(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)

    for spec in INPUT_SPECS:
        assert (output / "inputs" / spec.filename).read_bytes() == (
            INPUT_ROOT / spec.filename
        ).read_bytes()

    assert len(read_csv(output / "inputs/trackgen-pass2-reliability-selection.csv")) == 18
    assert len(read_csv(output / "inputs/trackgen-pass2-reliability-disagreements.csv")) == 42
    validate_reliability_pilot(
        repository_root=ROOT, snapshot=output, input_root=INPUT_ROOT
    )


def test_records_failed_gates_and_prospective_rules(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    readme = (output / "README.md").read_text(encoding="utf-8")
    limitations = (output / "PROCEDURAL-LIMITATIONS.md").read_text(encoding="utf-8")
    codebook = (output / "CODEBOOK-v2.md").read_text(encoding="utf-8")

    for value in (
        "representation_family: 8/18 (0.444) - FAIL",
        "generator_family: 12/18 (0.667) - FAIL",
        "generation_role: 8/18 (0.444) - FAIL",
        "validity_strategy: 9/18 (0.500) - FAIL",
    ):
        assert value in readme
    assert "cannot support final prevalence/taxonomy claims" in readme
    assert "source-native" in codebook
    assert "course-defining" in codebook
    assert "multi-label" in codebook
    assert "constructive" in codebook and "stochastic_procedural" in codebook
    assert "benchmark_only" in codebook and "task_selection" in codebook
    assert "not_reported" in codebook and "NR" in codebook
    assert "core" in codebook and "supporting" in codebook
    assert "availability" in codebook
    assert "code_status/asset_status may be sole NR" in codebook
    assert "completed rows cannot use NR" not in codebook
    assert "all 75" in limitations
    assert "fresh blind reliability" in limitations
    assert "exact-set >=0.80 for each of the eight fields" in limitations
    assert "does not require two consecutive 30-source rounds" in limitations
    assert "pre-submission replication is recommended" in limitations


def test_records_exact_coordinator_metadata_and_bindings(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    registry = read_csv(output / "execution_registry.csv")
    assert registry == [
        {
            "role": "pass2-reliability-01",
            "agent_id": "019f3969-3bbb-7d01-9abd-f302a8643dc4",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "high",
            "fork_context": "false",
            "scope": "independent blind reliability coding",
        },
        {
            "role": "source-adjudicator",
            "agent_id": "019f396e-256a-7091-bd44-c5e3d2fe6f63",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "high",
            "fork_context": "NR",
            "scope": "source-level adjudication after reliability comparison",
        },
        {
            "role": "methods-reviewer",
            "agent_id": "019f396e-2543-78b1-ba9c-17d0d6a1ba80",
            "model": "gpt-5.6-terra",
            "reasoning_effort": "high",
            "fork_context": "NR",
            "scope": "prospective codebook review",
        },
    ]
    bindings = read_csv(output / "bindings.csv")
    assert [row["binding"] for row in bindings] == [
        "primary_snapshot",
        "draft_release",
    ]
    assert all(row["bound_sha256"] for row in bindings)


def test_rejects_existing_snapshot_directory(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    with pytest.raises(ReliabilityIntegrationError, match="must not already exist"):
        integrate_reliability_pilot(
            repository_root=ROOT, output=output, input_root=INPUT_ROOT
        )


def test_validator_rejects_tampered_input_copy(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    copied = output / "inputs/trackgen-pass2-reliability-summary.csv"
    copied.write_bytes(copied.read_bytes() + b"\n")

    with pytest.raises(ReliabilityValidationError, match="checksum"):
        validate_reliability_pilot(
            repository_root=ROOT, snapshot=output, input_root=INPUT_ROOT
        )


def test_validator_rejects_primary_binding_change(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    bindings = output / "bindings.csv"
    rows = read_csv(bindings)
    rows[0]["bound_sha256"] = "0" * 64
    with bindings.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ReliabilityValidationError, match="primary snapshot binding"):
        validate_reliability_pilot(
            repository_root=ROOT, snapshot=output, input_root=INPUT_ROOT
        )
