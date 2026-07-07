from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import paper.scripts.integrate_pass2_reliability as reliability_snapshot


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path("/tmp")
INPUTS = (
    "trackgen-pass2-v2-holdout-manifest.csv",
    "trackgen-pass2-v2-reliability-selection.csv",
    "trackgen-pass2-v2-reliability-packet.csv",
    "trackgen-pass2-v2-reliability-template.csv",
    "trackgen-pass2-v2-reliability-coded.csv",
    "trackgen-pass2-v2-reliability-primary-sample.csv",
    "trackgen-pass2-v2-reliability-summary.csv",
)
CORE_FIELDS = (
    "survey_evidence_tier",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "code_status",
    "asset_status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_snapshot(tmp_path: Path) -> Path:
    output = tmp_path / "pass2_reliability/v2"
    reliability_snapshot.integrate_reliability_v2(
        repository_root=ROOT, output=output, input_root=INPUT_ROOT
    )
    return output


def test_creates_byte_exact_v2_snapshot_with_locked_holdout_and_summary(
    tmp_path: Path,
) -> None:
    output = build_snapshot(tmp_path)

    for filename in INPUTS:
        assert (output / "inputs" / filename).read_bytes() == (
            INPUT_ROOT / filename
        ).read_bytes()

    holdout = read_csv(output / "inputs/trackgen-pass2-v2-holdout-manifest.csv")
    assert len(holdout) == 18
    assert {
        domain: sum(row["first_domain"] == domain for row in holdout)
        for domain in ("ground", "adjacent", "aerial", "maritime", "NR")
    } == {"ground": 10, "adjacent": 3, "aerial": 2, "maritime": 2, "NR": 1}
    assert [row["cite_key"] for row in holdout if row["pilot_overlap"] == "true"] == [
        "DRAFT_C0063"
    ]

    summary = read_csv(output / "inputs/trackgen-pass2-v2-reliability-summary.csv")
    assert [(row["field"], row["agreement"], row["passes"]) for row in summary] == [
        ("survey_evidence_tier", "0.888889", "true"),
        ("course_object", "0.666667", "false"),
        ("representation_family", "0.500000", "false"),
        ("generator_family", "0.611111", "false"),
        ("generation_role", "0.666667", "false"),
        ("validity_strategy", "0.500000", "false"),
        ("code_status", "0.500000", "false"),
        ("asset_status", "0.888889", "true"),
    ]
    reliability_snapshot.validate_reliability_v2(
        repository_root=ROOT, snapshot=output, input_root=INPUT_ROOT
    )


def test_v2_derives_disagreements_and_records_required_scope(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    primary = {
        row["cite_key"]: row
        for row in read_csv(output / "inputs/trackgen-pass2-v2-reliability-primary-sample.csv")
    }
    reliability = {
        row["cite_key"]: row
        for row in read_csv(output / "inputs/trackgen-pass2-v2-reliability-coded.csv")
    }
    expected = [
        {
            "cite_key": cite_key,
            "field": field,
            "primary_value": primary[cite_key][field],
            "reliability_value": reliability[cite_key][field],
        }
        for cite_key in sorted(primary)
        for field in CORE_FIELDS
        if primary[cite_key][field] != reliability[cite_key][field]
    ]
    assert read_csv(output / "disagreements.csv") == expected
    assert len(expected) == 50

    registry = read_csv(output / "execution_registry.csv")
    assert registry[0]["agent_id"] == "019f39a2-f37d-77b0-8397-1ef173ab8a56"
    assert registry[0]["model"] == "gpt-5.6-terra"
    assert registry[0]["reasoning_effort"] == "high"
    assert registry[0]["fork_context"] == "false"
    assert [row["binding"] for row in read_csv(output / "bindings.csv")] == [
        "primary_snapshot",
        "draft_release",
        "pilot_v1_codebook_v2",
    ]

    documentation = (
        (output / "README.md").read_text(encoding="utf-8")
        + (output / "PROCEDURAL-LIMITATIONS.md").read_text(encoding="utf-8")
    )
    for phrase in (
        "second failed reliability round",
        "triggers stopping label-by-label codebook exception iteration",
        "must not adjudicate to manufacture agreement",
        "prevalence, frequency, or comparative taxonomy claims",
        "source-backed qualitative synthesis with direct locators",
        "scalar primary representation versus realization tags",
        "mechanism, stochasticity, and adaptation axes",
        "separate evidenced operations",
        "validity applicability from mechanism",
        "dedicated retrieval audit",
    ):
        assert phrase in documentation
    assert "CODEBOOK-v3" not in documentation


def test_v2_rejects_existing_output_and_tampered_disagreements(tmp_path: Path) -> None:
    output = build_snapshot(tmp_path)
    with pytest.raises(reliability_snapshot.ReliabilityIntegrationError, match="must not already exist"):
        reliability_snapshot.integrate_reliability_v2(
            repository_root=ROOT, output=output, input_root=INPUT_ROOT
        )

    disagreements = output / "disagreements.csv"
    disagreements.write_bytes(disagreements.read_bytes() + b"\n")
    with pytest.raises(
        reliability_snapshot.ReliabilityValidationError, match="derived disagreements mismatch"
    ):
        reliability_snapshot.validate_reliability_v2(
            repository_root=ROOT, snapshot=output, input_root=INPUT_ROOT
        )


def rewrite_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mutate_v2_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot: Path,
    mutations: dict[str, list[dict[str, str]]],
) -> Path:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    for filename in INPUTS:
        shutil.copyfile(INPUT_ROOT / filename, input_root / filename)
    for filename, rows in mutations.items():
        rewrite_csv(input_root / filename, rows)
        shutil.copyfile(input_root / filename, snapshot / "inputs" / filename)

    monkeypatch.setattr(
        reliability_snapshot,
        "V2_INPUT_SPECS",
        tuple(
            replace(
                spec,
                sha256=hashlib.sha256(
                    (input_root / spec.filename).read_bytes()
                ).hexdigest(),
            )
            if spec.filename in mutations
            else spec
            for spec in reliability_snapshot.V2_INPUT_SPECS
        ),
    )
    return input_root


def test_v2_rejects_tampered_salted_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = build_snapshot(tmp_path)
    filename = "trackgen-pass2-v2-holdout-manifest.csv"
    holdout = read_csv(INPUT_ROOT / filename)
    holdout[0]["salted_rank_sha256"] = "0" * 64
    input_root = mutate_v2_inputs(
        monkeypatch, tmp_path, output, {filename: holdout}
    )

    with pytest.raises(
        reliability_snapshot.ReliabilityValidationError,
        match="salted holdout deterministic selection mismatch",
    ):
        reliability_snapshot.validate_reliability_v2(
            repository_root=ROOT, snapshot=output, input_root=input_root
        )


def test_v2_rejects_quota_preserving_cherry_picked_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = build_snapshot(tmp_path)
    holdout_filename = "trackgen-pass2-v2-holdout-manifest.csv"
    selection_filename = "trackgen-pass2-v2-reliability-selection.csv"
    holdout = read_csv(INPUT_ROOT / holdout_filename)
    selection = read_csv(INPUT_ROOT / selection_filename)
    replacement = "DRAFT_C0023"
    evidence_sha256 = holdout[0]["evidence_sha256"]
    salt = reliability_snapshot.V2_HOLDOUT_SALT

    holdout[0] = {
        "cite_key": replacement,
        "first_domain": "ground",
        "selection_salt": salt,
        "salted_rank_sha256": hashlib.sha256(
            f"{salt}\0{replacement}".encode("utf-8")
        ).hexdigest(),
        "pilot_overlap": "false",
        "evidence_sha256": evidence_sha256,
    }
    selection[0] = {
        "cite_key": replacement,
        "first_domain": "ground",
        "rank_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
        "evidence_sha256": evidence_sha256,
    }
    input_root = mutate_v2_inputs(
        monkeypatch,
        tmp_path,
        output,
        {holdout_filename: holdout, selection_filename: selection},
    )

    with pytest.raises(
        reliability_snapshot.ReliabilityValidationError,
        match="salted holdout deterministic selection mismatch",
    ):
        reliability_snapshot.validate_reliability_v2(
            repository_root=ROOT, snapshot=output, input_root=input_root
        )
