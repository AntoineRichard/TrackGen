from __future__ import annotations

import csv
import inspect
import json
import shutil
from pathlib import Path

import pytest

from paper.scripts.integrate_pass2_primary import (
    EXPECTED_BATCH_SPECS,
    IntegrationError,
    _read_evidence,
    _validate_batch_rows,
    integrate_primary_batches,
    main,
)
from paper.scripts.prepare_pass2_draft import EVIDENCE_HEADER
from paper.scripts.validate_pass2_draft import validate_coding_output

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "paper/data/screening_work/v8/pass2_drafts/v1"
SNAPSHOT = ROOT / "paper/data/screening_work/v8/pass2_coding/primary/v1"
EXPECTED = (
    (1, "pass2-primary-01", "019f3952-5cc3-7ac0-b45b-a85d09eb459c", "trackgen-pass2-primary-01.csv", "53436028264e61d23f35c083ecc8cbed58836a3f8f1ea223a8cb7eb55872d507", 13),
    (2, "pass2-primary-02", "019f3952-5cec-70e0-bf51-cf59f159aa53", "trackgen-pass2-primary-02.csv", "498e2f3092a6591dfab7493a81367220c4acbdedffc3453fdf2e9c2fd9d5466c", 13),
    (3, "pass2-primary-03", "019f3952-5dbf-7250-ad10-38411837bbda", "trackgen-pass2-primary-03.csv", "752bed14ac9ad5be0b6513f09a77007ad53ec3e2e78ccef1fc0b9febf851ac16", 13),
    (4, "pass2-primary-04", "019f3952-5d5b-7c23-9f4f-491aae449174", "trackgen-pass2-primary-04.csv", "51dbccc2a60e7081ae9860f3676735ce572f18661c376574bd70bc6d971bd410", 12),
    (5, "pass2-primary-05", "019f3952-5d2d-74f1-a2ed-f5dfa8a58dea", "trackgen-pass2-primary-05.csv", "8fa47b1cbabbe446860e7a0f26e7cf337541ce144c30f10f18d4d7b4f03e1645", 12),
    (6, "pass2-primary-06", "019f3952-5d94-7023-9bce-89acde705820", "trackgen-pass2-primary-06.csv", "9c71b1d35b69d7fd35fdd18cbde434dc056fd066b9281626f8fb4930d91dc793", 12),
)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def input_root(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    root.mkdir()
    for spec in EXPECTED_BATCH_SPECS:
        shutil.copyfile(
            SNAPSHOT / f"batches/pass2-primary-{spec.number:02d}.csv",
            root / spec.filename,
        )
    return root


def integrate(tmp_path: Path, root: Path) -> Path:
    output = tmp_path / "pass2_coding/primary/v1"
    integrate_primary_batches(
        repository_root=ROOT,
        release=RELEASE,
        output=output,
        input_root=root,
    )
    return output


def test_specs_are_exact_and_cannot_be_overridden() -> None:
    assert tuple(
        (s.number, s.role, s.agent_id, s.filename, s.source_digest, s.row_count)
        for s in EXPECTED_BATCH_SPECS
    ) == EXPECTED
    assert all(s.model == "gpt-5.6-terra" for s in EXPECTED_BATCH_SPECS)
    assert all(s.reasoning_effort == "high" for s in EXPECTED_BATCH_SPECS)
    parameters = inspect.signature(integrate_primary_batches).parameters
    assert "input_root" in parameters
    assert "batch_sources" not in parameters
    assert "batch_specs" not in parameters


def test_integrates_fixed_inputs_deterministically(tmp_path: Path) -> None:
    root = input_root(tmp_path)
    output = integrate(tmp_path, root)

    for spec in EXPECTED_BATCH_SPECS:
        copied = output / f"batches/pass2-primary-{spec.number:02d}.csv"
        assert copied.read_bytes() == (root / spec.filename).read_bytes()
    evidence = csv_rows(output / "coding/evidence.csv")
    assert len(evidence) == 75
    assert [row["cite_key"] for row in evidence] == sorted(
        row["cite_key"] for row in evidence
    )
    registry = csv_rows(output / "execution_registry.csv")
    for row, spec in zip(registry, EXPECTED_BATCH_SPECS, strict=True):
        assert row["role"] == spec.role
        assert row["agent_id"] == spec.agent_id
        assert row["model"] == spec.model
        assert row["reasoning_effort"] == spec.reasoning_effort
        assert row["row_count"] == str(spec.row_count)
        assert row["source_input_filename"] == spec.filename
        assert row["source_input_sha256"] == spec.source_digest
    validate_coding_output(
        repository_root=ROOT, release=RELEASE, coding_output=output / "coding"
    )


def test_rejects_digest_mismatch(tmp_path: Path) -> None:
    root = input_root(tmp_path)
    source = root / EXPECTED_BATCH_SPECS[0].filename
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(IntegrationError, match="digest mismatch"):
        integrate(tmp_path, root)


def test_rejects_symlinked_input(tmp_path: Path) -> None:
    root = input_root(tmp_path)
    source = root / EXPECTED_BATCH_SPECS[0].filename
    target = root / "actual.csv"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(IntegrationError, match="regular non-symlink"):
        integrate(tmp_path, root)


def test_rejects_hard_linked_inputs(tmp_path: Path) -> None:
    root = input_root(tmp_path)
    first = root / EXPECTED_BATCH_SPECS[0].filename
    second = root / EXPECTED_BATCH_SPECS[1].filename
    second.unlink()
    second.hardlink_to(first)
    with pytest.raises(IntegrationError, match="hard-link alias"):
        integrate(tmp_path, root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("assignment", "assignment"),
        ("blank", "blank analytical"),
        ("duplicate", "duplicate or missing"),
        ("taxonomy", "taxonomy"),
        ("locator", "locator"),
    ],
)
def test_rejects_invalid_batch_rows(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = input_root(tmp_path)
    spec = EXPECTED_BATCH_SPECS[0]
    rows = csv_rows(root / spec.filename)
    if mutation == "assignment":
        rows[0]["cite_key"] = csv_rows(root / EXPECTED_BATCH_SPECS[1].filename)[0][
            "cite_key"
        ]
    elif mutation == "blank":
        rows[0]["domain"] = ""
    elif mutation == "duplicate":
        rows[1]["cite_key"] = rows[0]["cite_key"]
    elif mutation == "taxonomy":
        rows[0]["domain"] = "not-a-domain"
        rows[0]["evidence_locator"] = "domain=PDF p. 1"
    else:
        rows[0]["domain"] = "ground"
        rows[0]["evidence_locator"] = "NR"
    taxonomy = json.loads(
        (ROOT / "paper/data/taxonomy.json").read_text(encoding="utf-8")
    )
    with pytest.raises(IntegrationError, match=message):
        _validate_batch_rows(
            rows=rows, spec=spec, release=RELEASE, taxonomy=taxonomy
        )


def test_rejects_wrong_header(tmp_path: Path) -> None:
    root = input_root(tmp_path)
    source = root / EXPECTED_BATCH_SPECS[0].filename
    rows = csv_rows(source)
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(reversed(EVIDENCE_HEADER)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(IntegrationError, match="header"):
        _read_evidence(source, label="batch input 1")


def test_cli_rejects_batch_path_override(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--batch-01", "/tmp/alternate.csv"])
    assert "unrecognized arguments: --batch-01" in capsys.readouterr().err


def test_rejects_existing_snapshot_directory(tmp_path: Path) -> None:
    root = input_root(tmp_path)
    output = integrate(tmp_path, root)
    with pytest.raises(IntegrationError, match="must not already exist"):
        integrate_primary_batches(
            repository_root=ROOT,
            release=RELEASE,
            output=output,
            input_root=root,
        )
