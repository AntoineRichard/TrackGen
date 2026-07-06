from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from paper.scripts.integrate_pass2_primary import (
    BatchSpec,
    IntegrationError,
    integrate_primary_batches,
)
from paper.scripts.prepare_pass2_draft import EVIDENCE_HEADER
from paper.scripts.validate_pass2_draft import validate_coding_output


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "paper/data/screening_work/v8/pass2_drafts/v1"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def completed_rows(path: Path) -> list[dict[str, str]]:
    return [
        {field: row["cite_key"] if field == "cite_key" else "NR" for field in EVIDENCE_HEADER}
        for row in csv_rows(path)
    ]


def batch_specs(tmp_path: Path) -> tuple[tuple[BatchSpec, ...], dict[int, Path]]:
    specs: list[BatchSpec] = []
    sources: dict[int, Path] = {}
    for number in range(1, 7):
        source = tmp_path / f"pass2-primary-{number:02d}.csv"
        write_rows(source, completed_rows(RELEASE / f"primary-batch-{number:02d}.csv"))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        specs.append(
            BatchSpec(
                number=number,
                role=f"pass2-primary-{number:02d}",
                agent_id=f"agent-{number:02d}",
                source_digest=digest,
            )
        )
        sources[number] = source
    return tuple(specs), sources


def integrate(tmp_path: Path, specs: tuple[BatchSpec, ...], sources: dict[int, Path]) -> Path:
    output = tmp_path / "pass2_coding/primary/v1"
    integrate_primary_batches(
        repository_root=ROOT,
        release=RELEASE,
        output=output,
        batch_sources=sources,
        batch_specs=specs,
    )
    return output


def test_integrates_six_batches_as_a_deterministic_snapshot(tmp_path: Path) -> None:
    specs, sources = batch_specs(tmp_path)
    output = integrate(tmp_path, specs, sources)

    for number, source in sources.items():
        assert (output / f"batches/pass2-primary-{number:02d}.csv").read_bytes() == source.read_bytes()
    evidence = csv_rows(output / "coding/evidence.csv")
    assert len(evidence) == 75
    assert [row["cite_key"] for row in evidence] == sorted(row["cite_key"] for row in evidence)
    assert len({row["cite_key"] for row in evidence}) == 75
    registry = csv_rows(output / "execution_registry.csv")
    assert len(registry) == 6
    assert all(row["human_role"] == "NR" for row in registry)
    assert all(row["model"] == "gpt-5.6-terra" for row in registry)
    checksums = csv_rows(output / "manifest/checksums.csv")
    assert {row["path"] for row in checksums} >= {
        "coding/evidence.csv",
        "execution_registry.csv",
        "README.md",
        "FROZEN-CODER-PROMPT.md",
        "PROCEDURAL-LIMITATIONS.md",
    }
    assert "Code only with `DRAFT_C####` keys" in (output / "FROZEN-CODER-PROMPT.md").read_text(encoding="utf-8")
    limitations = (output / "PROCEDURAL-LIMITATIONS.md").read_text(encoding="utf-8")
    assert "not provider-side immutable execution attestation" in limitations
    assert "no prevalence or final projection claim" in limitations
    validate_coding_output(repository_root=ROOT, release=RELEASE, coding_output=output / "coding")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("digest", "digest mismatch"),
        ("alias", "regular non-symlink"),
        ("header", "header"),
        ("assignment", "assignment"),
        ("blank", "blank analytical"),
        ("duplicate", "duplicate or missing"),
        ("taxonomy", "taxonomy"),
        ("locator", "locator"),
    ],
)
def test_rejects_invalid_batch_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    specs, sources = batch_specs(tmp_path)
    source = sources[1]
    if mutation == "digest":
        source.write_bytes(source.read_bytes() + b"\n")
    elif mutation == "alias":
        target = source.with_name("actual.csv")
        source.rename(target)
        source.symlink_to(target)
    elif mutation == "header":
        rows = completed_rows(RELEASE / "primary-batch-01.csv")
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(reversed(EVIDENCE_HEADER)), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    elif mutation == "assignment":
        rows = completed_rows(RELEASE / "primary-batch-01.csv")
        rows[0]["cite_key"] = completed_rows(RELEASE / "primary-batch-02.csv")[0]["cite_key"]
        write_rows(source, rows)
    elif mutation == "blank":
        rows = completed_rows(RELEASE / "primary-batch-01.csv")
        rows[0]["domain"] = ""
        write_rows(source, rows)
    elif mutation == "duplicate":
        rows = completed_rows(RELEASE / "primary-batch-01.csv")
        rows[1]["cite_key"] = rows[0]["cite_key"]
        write_rows(source, rows)
    elif mutation == "taxonomy":
        rows = completed_rows(RELEASE / "primary-batch-01.csv")
        rows[0]["domain"] = "not-a-domain"
        rows[0]["evidence_locator"] = "domain=PDF p. 1"
        write_rows(source, rows)
    else:
        rows = completed_rows(RELEASE / "primary-batch-01.csv")
        rows[0]["domain"] = "ground"
        write_rows(source, rows)

    if mutation not in {"digest", "alias"}:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        specs = tuple(
            BatchSpec(
                number=spec.number,
                role=spec.role,
                agent_id=spec.agent_id,
                source_digest=digest if spec.number == 1 else spec.source_digest,
            )
            for spec in specs
        )
    with pytest.raises(IntegrationError, match=message):
        integrate(tmp_path, specs, sources)


def test_rejects_existing_snapshot_directory(tmp_path: Path) -> None:
    specs, sources = batch_specs(tmp_path)
    output = integrate(tmp_path, specs, sources)

    with pytest.raises(IntegrationError, match="must not already exist"):
        integrate_primary_batches(
            repository_root=ROOT,
            release=RELEASE,
            output=output,
            batch_sources=sources,
            batch_specs=specs,
        )
