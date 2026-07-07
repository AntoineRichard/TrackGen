from __future__ import annotations

import csv
import hashlib
import os
import stat
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import paper.scripts.integrate_metadata as metadata_integration
import paper.scripts.prepare_metadata_batches as metadata_batches

from paper.scripts.prepare_metadata_batches import (
    MANIFEST_HEADER,
    ManifestError,
    build_manifest,
    main,
    validate_manifest_inputs,
)
from paper.scripts.validate_corpus import HEADERS


def candidate_row(candidate_id: str, **values: str) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["candidates.csv"], "")
    row.update(
        candidate_id=candidate_id,
        title=f"Title {candidate_id}",
        screening_status="candidate",
        metadata_status="unverified",
    )
    row.update(values)
    return row


def conflict_row(
    conflict_id: str,
    candidate_id: str,
    field: str,
    **values: str,
) -> dict[str, str]:
    row = dict.fromkeys(HEADERS["conflicts.csv"], "")
    row.update(
        conflict_id=conflict_id,
        record_type="candidate",
        record_key=candidate_id,
        field=field,
        value_a=f"old-{conflict_id}",
        value_b=f"new-{conflict_id}",
    )
    row.update(values)
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HEADERS[path.name],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_inputs(
    directory: Path,
    candidates: list[dict[str, str]],
    conflicts: list[dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    directory.mkdir()
    candidates_path = directory / "candidates.csv"
    conflicts_path = directory / "conflicts.csv"
    write_rows(candidates_path, candidates)
    write_rows(conflicts_path, conflicts or [])
    return candidates_path, conflicts_path


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_HEADER,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def rows_by_candidate(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


def file_state(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), stat.S_IMODE(path.stat().st_mode)


def persistent_file_state(path: Path) -> tuple[bytes, int, int, int]:
    file_stat = path.stat()
    return (
        path.read_bytes(),
        stat.S_IMODE(file_stat.st_mode),
        file_stat.st_ino,
        file_stat.st_mtime_ns,
    )

def recovery_directories(manifest_path: Path) -> list[Path]:
    return sorted(
        manifest_path.parent.glob(f".{manifest_path.name}.recovery.*")
    )


def build_refreeze_case(tmp_path):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert (
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )
        == 0
    )
    original_manifest = manifest_path.read_bytes()
    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    return (
        candidates_path,
        conflicts_path,
        manifest_path,
        snapshot_root,
        snapshot_root / "v2",
        base_arguments,
        original_manifest,
    )


def test_every_candidate_is_assigned_once_and_batches_are_row_balanced(tmp_path):
    candidates = [candidate_row(f"C{number:04d}") for number in range(1, 18)]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates
    )

    manifest = build_manifest(candidates_path, conflicts_path)

    candidate_ids = [row["candidate_id"] for row in manifest]
    assert Counter(candidate_ids) == Counter(row["candidate_id"] for row in candidates)
    assert all(count == 1 for count in Counter(candidate_ids).values())
    batch_counts = Counter(row["batch_id"] for row in manifest)
    counts = [batch_counts[f"metadata-{number:02d}"] for number in range(1, 7)]
    assert max(counts) - min(counts) <= 1
    assert [tuple(row) for row in manifest] == [MANIFEST_HEADER] * len(manifest)
    assert manifest == sorted(
        manifest, key=lambda row: (row["batch_id"], row["candidate_id"])
    )
    assert len({row["snapshot_sha256"] for row in manifest}) == 1


def test_manifest_bytes_are_identical_for_shuffled_inputs(tmp_path):
    candidates = [
        candidate_row("C0001", title="Track, One"),
        candidate_row("C0002", title="Track Two"),
        candidate_row("C0003", title="Track Three"),
    ]
    conflicts = [
        conflict_row("CF0001", "C0001", "title"),
        conflict_row("CF0002", "C0001", "authors"),
        conflict_row("CF0003", "C0003", "screening_status"),
    ]
    forward = build_inputs(tmp_path / "forward", candidates, conflicts)
    shuffled = build_inputs(
        tmp_path / "shuffled", list(reversed(candidates)), list(reversed(conflicts))
    )

    forward_output = tmp_path / "forward-manifest.csv"
    shuffled_output = tmp_path / "shuffled-manifest.csv"
    for inputs, output in (
        (forward, forward_output),
        (shuffled, shuffled_output),
    ):
        assert (
            main(
                [
                    "--candidates",
                    str(inputs[0]),
                    "--conflicts",
                    str(inputs[1]),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )

    assert shuffled_output.read_bytes() == forward_output.read_bytes()


def test_unresolved_bibliographic_conflicts_balance_total_weight(tmp_path):
    candidates = [candidate_row(f"C{number:04d}") for number in range(1, 13)]
    conflicts = [
        conflict_row(
            f"CF{candidate_number:02d}{conflict_number:02d}",
            f"C{candidate_number:04d}",
            field,
        )
        for candidate_number in range(1, 7)
        for conflict_number, field in enumerate(
            ("title", "authors", "doi"), start=1
        )
    ]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates, conflicts
    )

    manifest = build_manifest(candidates_path, conflicts_path)

    by_candidate = rows_by_candidate(manifest)
    assert all(
        by_candidate[f"C{number:04d}"]["weight"] == "4"
        for number in range(1, 7)
    )
    total_weights: defaultdict[str, int] = defaultdict(int)
    for row in manifest:
        total_weights[row["batch_id"]] += int(row["weight"])
    assert set(total_weights.values()) == {5}


def test_resolved_and_nonbibliographic_conflicts_do_not_add_weight(tmp_path):
    candidates = [candidate_row("C0001", cite_key="Example2026")]
    conflicts = [
        conflict_row("CF0001", "C0001", "title"),
        conflict_row(
            "CF0002",
            "C0001",
            "year",
            resolution="2026",
            resolver="reviewer",
            resolution_evidence="https://example.invalid/year",
        ),
        conflict_row("CF0003", "C0001", "screening_status"),
        conflict_row("CF0004", "C0001", "exclusion_reason"),
        conflict_row("CF0005", "C0001", "metadata_evidence"),
        conflict_row(
            "CF0006",
            "Example2026",
            "domain",
            record_type="evidence",
        ),
    ]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates, conflicts
    )

    manifest = build_manifest(candidates_path, conflicts_path)

    assert manifest[0]["weight"] == "2"


def test_candidate_count_below_batch_count_uses_earliest_batches(tmp_path):
    candidates = [candidate_row(f"C{number:04d}") for number in range(1, 4)]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates
    )

    manifest = build_manifest(candidates_path, conflicts_path)

    assert {row["batch_id"] for row in manifest} == {
        "metadata-01",
        "metadata-02",
        "metadata-03",
    }


def test_empty_candidate_corpus_is_rejected(tmp_path):
    candidates_path, conflicts_path = build_inputs(tmp_path / "inputs", [])

    with pytest.raises(ManifestError, match="at least one candidate"):
        build_manifest(candidates_path, conflicts_path)


def test_only_six_batches_are_supported(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )

    with pytest.raises(ManifestError, match="exactly 6"):
        build_manifest(candidates_path, conflicts_path, batch_count=5)


def test_validate_manifest_inputs_accepts_an_unchanged_snapshot(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001"), candidate_row("C0002")],
        [conflict_row("CF0001", "C0001", "title")],
    )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path, build_manifest(candidates_path, conflicts_path)
    )

    assert (
        validate_manifest_inputs(
            manifest_path, candidates_path, conflicts_path
        )
        is None
    )


def test_candidate_fingerprint_drift_is_rejected(tmp_path):
    candidates = [candidate_row("C0001", title="Original title")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates
    )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path, build_manifest(candidates_path, conflicts_path)
    )
    candidates[0]["title"] = "Changed title"
    write_rows(candidates_path, candidates)

    with pytest.raises(ManifestError, match="input_sha256"):
        validate_manifest_inputs(manifest_path, candidates_path, conflicts_path)


@pytest.mark.parametrize("drift", ["changed", "added", "removed"])
def test_conflict_fingerprint_drift_is_rejected(tmp_path, drift):
    candidates = [candidate_row("C0001")]
    conflicts = [conflict_row("CF0001", "C0001", "title")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates, conflicts
    )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path, build_manifest(candidates_path, conflicts_path)
    )
    if drift == "changed":
        conflicts[0]["value_b"] = "changed"
    elif drift == "added":
        conflicts.append(conflict_row("CF0002", "C0001", "venue"))
    else:
        conflicts.clear()
    write_rows(conflicts_path, conflicts)

    with pytest.raises(ManifestError, match="input_sha256"):
        validate_manifest_inputs(manifest_path, candidates_path, conflicts_path)


def _duplicate_manifest_id(rows: list[dict[str, str]]) -> None:
    rows[1]["candidate_id"] = rows[0]["candidate_id"]


def _remove_manifest_id(rows: list[dict[str, str]]) -> None:
    rows.pop()


def _add_manifest_id(rows: list[dict[str, str]]) -> None:
    extra = dict(rows[-1])
    extra["candidate_id"] = "C9999"
    rows.append(extra)


def _change_snapshot(rows: list[dict[str, str]]) -> None:
    rows[0]["snapshot_sha256"] = "0" * 64


def _change_input_hash(rows: list[dict[str, str]]) -> None:
    rows[0]["input_sha256"] = "0" * 64


def _change_weight(rows: list[dict[str, str]]) -> None:
    rows[0]["weight"] = str(int(rows[0]["weight"]) + 1)


def _change_batch(rows: list[dict[str, str]]) -> None:
    rows[0]["batch_id"] = "metadata-06"


def _change_version(rows: list[dict[str, str]]) -> None:
    rows[0]["manifest_version"] = "2"


def _reorder_manifest(rows: list[dict[str, str]]) -> None:
    rows[0], rows[1] = rows[1], rows[0]


@pytest.mark.parametrize(
    "mutate",
    [
        _duplicate_manifest_id,
        _remove_manifest_id,
        _add_manifest_id,
        _change_snapshot,
        _change_input_hash,
        _change_weight,
        _change_batch,
        _change_version,
        _reorder_manifest,
    ],
    ids=lambda mutate: mutate.__name__.removeprefix("_"),
)
def test_manifest_tampering_is_rejected(tmp_path, mutate):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row(f"C{number:04d}") for number in range(1, 8)],
    )
    manifest = build_manifest(candidates_path, conflicts_path)
    mutate(manifest)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, manifest)

    with pytest.raises(ManifestError):
        validate_manifest_inputs(manifest_path, candidates_path, conflicts_path)


def test_duplicate_candidate_ids_are_rejected(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001"), candidate_row("C0001")],
    )

    with pytest.raises(ManifestError, match="duplicate candidate_id"):
        build_manifest(candidates_path, conflicts_path)


def test_orphan_candidate_conflicts_are_rejected(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
        [conflict_row("CF0001", "C9999", "title")],
    )

    with pytest.raises(ManifestError, match="orphaned"):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize("filename", ["candidates.csv", "conflicts.csv"])
def test_noncanonical_input_headers_are_rejected(tmp_path, filename):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
    path = candidates_path if filename == "candidates.csv" else conflicts_path
    header = list(HEADERS[filename])
    header[0], header[1] = header[1], header[0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(header)

    with pytest.raises(ManifestError, match="headers"):
        build_manifest(candidates_path, conflicts_path)


def test_structurally_malformed_input_row_is_rejected(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
    with candidates_path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            ["C0002", *([""] * len(HEADERS["candidates.csv"]))]
        )

    with pytest.raises(ManifestError, match="malformed CSV row"):
        build_manifest(candidates_path, conflicts_path)


def test_manifest_header_must_be_exact(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "candidate_id,batch_id\nC0001,metadata-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="headers"):
        validate_manifest_inputs(manifest_path, candidates_path, conflicts_path)


def test_existing_valid_manifest_is_validated_without_replacement(
    tmp_path, monkeypatch
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001"), candidate_row("C0002")],
        [conflict_row("CF0001", "C0001", "title")],
    )
    output_path = tmp_path / "metadata_manifest.csv"
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
    ]
    original_candidates = candidates_path.read_bytes()
    original_conflicts = conflicts_path.read_bytes()

    assert main(arguments) == 0
    output_path.chmod(0o640)
    frozen_state = file_state(output_path)
    monkeypatch.setattr(
        metadata_batches,
        "_write_manifest",
        lambda *_: pytest.fail("existing manifest was rewritten"),
    )

    assert main(arguments) == 0

    assert file_state(output_path) == frozen_state
    assert candidates_path.read_bytes() == original_candidates
    assert conflicts_path.read_bytes() == original_conflicts
    with output_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == MANIFEST_HEADER
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_existing_stale_manifest_is_refused_without_replacement(tmp_path):
    candidates = [candidate_row("C0001", title="Original")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates
    )
    output_path = tmp_path / "metadata_manifest.csv"
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
    ]
    assert main(arguments) == 0
    output_path.chmod(0o640)
    frozen_state = file_state(output_path)
    candidates[0]["title"] = "Changed"
    write_rows(candidates_path, candidates)

    with pytest.raises(ManifestError, match="input_sha256"):
        main(arguments)

    assert file_state(output_path) == frozen_state


def test_existing_tampered_manifest_is_refused_without_replacement(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
    output_path = tmp_path / "metadata_manifest.csv"
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
    ]
    assert main(arguments) == 0
    with output_path.open(encoding="utf-8", newline="") as handle:
        tampered = list(csv.DictReader(handle))
    tampered[0]["snapshot_sha256"] = "0" * 64
    write_manifest(output_path, tampered)
    output_path.chmod(0o640)
    tampered_state = file_state(output_path)

    with pytest.raises(ManifestError, match="snapshot_sha256"):
        main(arguments)

    assert file_state(output_path) == tampered_state


def test_refreeze_deliberately_replaces_a_stale_manifest(tmp_path):
    candidates = [candidate_row("C0001", title="Original")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates
    )
    output_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
    ]
    assert main([*arguments, "--snapshot-dir", str(snapshot_root / "v1")]) == 0
    output_path.chmod(0o640)
    original_state = file_state(output_path)
    candidates[0]["title"] = "Changed"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"

    assert (
        main(
            [
                *arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )
        == 0
    )

    assert output_path.read_bytes() != original_state[0]
    assert stat.S_IMODE(output_path.stat().st_mode) == original_state[1]
    validate_manifest_inputs(
        output_path,
        version_two / "candidates.csv",
        version_two / "conflicts.csv",
    )


def test_refreeze_validates_inputs_before_replacing_existing_output(tmp_path):
    candidates = [candidate_row("C0001")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates
    )
    output_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
    ]
    assert main([*arguments, "--snapshot-dir", str(snapshot_root / "v1")]) == 0
    frozen_state = file_state(output_path)
    write_rows(candidates_path, [candidates[0], dict(candidates[0])])
    version_two = snapshot_root / "v2"

    with pytest.raises(ManifestError, match="duplicate candidate_id"):
        main(
            [
                *arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert file_state(output_path) == frozen_state
    assert not version_two.exists()


@pytest.mark.parametrize(
    "field",
    ("candidate_id", "title", "screening_status", "metadata_status"),
)
def test_candidate_required_fields_are_rejected_when_blank(tmp_path, field):
    candidate = candidate_row("C0001")
    candidate[field] = " "
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate]
    )

    with pytest.raises(ManifestError, match=rf"{field} is required"):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize("candidate_id", ["1", "C123", "c0001", "C0001x"])
def test_candidate_ids_must_use_the_canonical_pattern(tmp_path, candidate_id):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row(candidate_id)]
    )

    with pytest.raises(
        ManifestError, match="C followed by at least four digits"
    ):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("screening_status", "reviewed"),
        ("metadata_status", "complete"),
    ],
)
def test_candidate_controlled_statuses_must_be_exact(tmp_path, field, value):
    candidate = candidate_row("C0001")
    candidate[field] = value
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate]
    )

    with pytest.raises(ManifestError, match=field):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize(
    "field",
    ("conflict_id", "record_type", "record_key", "field", "value_a", "value_b"),
)
def test_conflict_required_fields_are_rejected_when_blank(tmp_path, field):
    conflict = conflict_row("CF0001", "C0001", "title")
    conflict[field] = " "
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")], [conflict]
    )

    with pytest.raises(ManifestError, match=rf"{field} is required"):
        build_manifest(candidates_path, conflicts_path)


def test_duplicate_conflict_ids_are_rejected(tmp_path):
    conflicts = [
        conflict_row("CF0001", "C0001", "title"),
        conflict_row("CF0001", "C0001", "authors"),
    ]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")], conflicts
    )

    with pytest.raises(ManifestError, match="duplicate conflict_id"):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize(
    ("record_type", "record_key"),
    [
        ("candidate", "C9999"),
        ("evidence", "UnknownKey"),
    ],
)
def test_conflict_record_keys_must_resolve(
    tmp_path, record_type, record_key
):
    conflict = conflict_row(
        "CF0001",
        record_key,
        "title" if record_type == "candidate" else "domain",
        record_type=record_type,
    )
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001", cite_key="Example2026")],
        [conflict],
    )

    with pytest.raises(ManifestError, match="record_key"):
        build_manifest(candidates_path, conflicts_path)


def test_conflict_record_type_must_be_supported(tmp_path):
    conflict = conflict_row(
        "CF0001",
        "C0001",
        "title",
        record_type="claim",
    )
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")], [conflict]
    )

    with pytest.raises(ManifestError, match="unsupported record_type"):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize(
    ("record_type", "record_key"),
    [
        ("candidate", "C0001"),
        ("evidence", "Example2026"),
    ],
)
def test_conflict_field_must_exist_in_target_schema(
    tmp_path, record_type, record_key
):
    conflict = conflict_row(
        "CF0001",
        record_key,
        "not_a_column",
        record_type=record_type,
    )
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001", cite_key="Example2026")],
        [conflict],
    )

    with pytest.raises(ManifestError, match="not a column"):
        build_manifest(candidates_path, conflicts_path)


@pytest.mark.parametrize("missing", ["resolver", "resolution_evidence"])
def test_resolved_conflict_requires_resolver_and_evidence(tmp_path, missing):
    conflict = conflict_row(
        "CF0001",
        "C0001",
        "title",
        resolution="Resolved title",
        resolver="reviewer",
        resolution_evidence="https://example.invalid/evidence",
    )
    conflict[missing] = ""
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")], [conflict]
    )

    with pytest.raises(ManifestError, match=rf"{missing} is required"):
        build_manifest(candidates_path, conflicts_path)


def test_manifest_version_one_hash_contract_is_frozen(tmp_path):
    candidates = [
        candidate_row(
            "C0001",
            title="Génération de pistes côtières",
            authors="Élodie Müller",
            year="2024",
        ),
        candidate_row(
            "C0002",
            title="Beta Track",
            authors="B. Two",
            year="2025",
        ),
    ]
    conflicts = [
        conflict_row(
            "CF0002",
            "C0001",
            "screening_status",
            value_a="candidate",
            value_b="boundary",
        ),
        conflict_row(
            "CF0001",
            "C0001",
            "title",
            value_a="Génération de pistes côtières",
            value_b="Génération de pistes côtières révisée",
        ),
    ]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates, conflicts
    )

    manifest = build_manifest(candidates_path, conflicts_path)

    # These literals come from the documented version-1 canonical JSON.
    # Changing them requires a manifest-version migration, not a normal test
    # update.
    assert manifest == [
        {
            "manifest_version": "1",
            "snapshot_sha256": (
                "d668279291724004e3b34a436e54d69b3b04d1eeec0b189d4656e98e42023fad"
            ),
            "batch_id": "metadata-01",
            "candidate_id": "C0001",
            "input_sha256": (
                "f3cd3549dd1e52331c718c713c0692ba0bcc7b12b5b7bd28d317e6b54a9bf615"
            ),
            "weight": "2",
        },
        {
            "manifest_version": "1",
            "snapshot_sha256": (
                "d668279291724004e3b34a436e54d69b3b04d1eeec0b189d4656e98e42023fad"
            ),
            "batch_id": "metadata-02",
            "candidate_id": "C0002",
            "input_sha256": (
                "69e2ea57279ce98cd06699660487a13a9404a2c63e252ee932724b3ff97a2fc7"
            ),
            "weight": "1",
        },
    ]


def test_skewed_weights_choose_least_weight_batch_without_exceeding_capacity(
    tmp_path,
):
    candidates = [
        candidate_row(f"C{number:04d}") for number in range(1, 14)
    ]
    conflicts = [
        conflict_row(
            f"CF{number:04d}",
            "C0001",
            "title",
            value_a=f"old-{number}",
            value_b=f"new-{number}",
        )
        for number in range(1, 21)
    ]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", candidates, conflicts
    )

    manifest = build_manifest(candidates_path, conflicts_path)

    assert {row["candidate_id"]: row["batch_id"] for row in manifest} == {
        "C0001": "metadata-01",
        "C0013": "metadata-02",
        "C0011": "metadata-02",
        "C0004": "metadata-02",
        "C0012": "metadata-03",
        "C0009": "metadata-03",
        "C0010": "metadata-03",
        "C0002": "metadata-04",
        "C0005": "metadata-04",
        "C0006": "metadata-05",
        "C0007": "metadata-05",
        "C0008": "metadata-06",
        "C0003": "metadata-06",
    }
    batch_counts = Counter(row["batch_id"] for row in manifest)
    assert max(batch_counts.values()) == 3
    assert sorted(batch_counts.values()) == [1, 2, 2, 2, 3, 3]
    batch_weights: defaultdict[str, int] = defaultdict(int)
    for row in manifest:
        batch_weights[row["batch_id"]] += int(row["weight"])
    assert batch_weights == {
        "metadata-01": 21,
        "metadata-02": 3,
        "metadata-03": 3,
        "metadata-04": 2,
        "metadata-05": 2,
        "metadata-06": 2,
    }


@pytest.mark.parametrize("failure_point", ["write", "fsync", "chmod", "exchange"])
def test_refreeze_failure_preserves_output_and_cleans_temporary_file(
    tmp_path, monkeypatch, failure_point
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
    output_path = tmp_path / "metadata_manifest.csv"
    output_path.write_bytes(b"frozen manifest sentinel\n")
    output_path.chmod(0o640)
    frozen_state = file_state(output_path)
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_two = snapshot_root / "v2"
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
        "--snapshot-dir",
        str(version_two),
        "--refreeze",
    ]

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "write":
        real_dict_writer = metadata_batches.csv.DictWriter

        def failing_dict_writer(*args, **kwargs):
            writer = real_dict_writer(*args, **kwargs)
            writer.writerows = fail
            return writer

        monkeypatch.setattr(
            metadata_batches.csv,
            "DictWriter",
            failing_dict_writer,
        )
    elif failure_point == "fsync":
        monkeypatch.setattr(metadata_batches.os, "fsync", fail)
    elif failure_point == "chmod":
        monkeypatch.setattr(metadata_batches.os, "chmod", fail)
    else:
        monkeypatch.setattr(metadata_batches, "_rename_exchange", fail)

    with pytest.raises(OSError, match=rf"injected {failure_point} failure"):
        main(arguments)

    assert file_state(output_path) == frozen_state
    assert not version_two.exists()
    assert not list(snapshot_root.glob(".v2.*.tmp"))
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


@pytest.mark.parametrize("output_name", ["candidates", "conflicts"])
def test_output_alias_is_rejected_for_each_input(tmp_path, output_name):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    paths = {
        "candidates": candidates_path,
        "conflicts": conflicts_path,
    }
    original_states = {
        name: file_state(path) for name, path in paths.items()
    }
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(paths[output_name]),
        "--snapshot-dir",
        str(snapshot_root / "v2"),
        "--refreeze",
    ]

    with pytest.raises(ManifestError, match=rf"differ from {output_name} input"):
        main(arguments)

    assert {
        name: file_state(path) for name, path in paths.items()
    } == original_states
    assert not list((tmp_path / "inputs").glob(".*.tmp"))


def test_initial_snapshot_freeze_preserves_exact_input_bytes(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001", title="Frozen, exactly")],
        [conflict_row("CF0001", "C0001", "title")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_dir = tmp_path / "metadata_inputs" / "v1"
    snapshot_dir.parent.mkdir()
    expected_candidates = candidates_path.read_bytes()
    expected_conflicts = conflicts_path.read_bytes()

    assert (
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )

    frozen_candidates = snapshot_dir / "candidates.csv"
    frozen_conflicts = snapshot_dir / "conflicts.csv"
    assert frozen_candidates.read_bytes() == expected_candidates
    assert frozen_conflicts.read_bytes() == expected_conflicts
    validate_manifest_inputs(
        manifest_path,
        frozen_candidates,
        frozen_conflicts,
    )


def test_snapshot_validation_is_non_mutating_and_needs_no_direct_inputs(
    tmp_path,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_dir = tmp_path / "metadata_inputs" / "v1"
    snapshot_dir.parent.mkdir()
    assert (
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )
    frozen_paths = (
        manifest_path,
        snapshot_dir / "candidates.csv",
        snapshot_dir / "conflicts.csv",
    )
    states = {path: persistent_file_state(path) for path in frozen_paths}

    assert (
        main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )

    assert {
        path: persistent_file_state(path) for path in frozen_paths
    } == states


def test_refreeze_publishes_new_version_and_preserves_old_version(tmp_path):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    initial_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--snapshot-dir",
        str(version_one),
        "--output",
        str(manifest_path),
    ]
    assert main(initial_arguments) == 0
    version_one_states = {
        path: persistent_file_state(path)
        for path in (
            version_one / "candidates.csv",
            version_one / "conflicts.csv",
        )
    }
    original_manifest = manifest_path.read_bytes()

    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"
    expected_candidates = candidates_path.read_bytes()
    expected_conflicts = conflicts_path.read_bytes()

    assert (
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(version_two),
                "--output",
                str(manifest_path),
                "--refreeze",
            ]
        )
        == 0
    )

    assert {
        path: persistent_file_state(path) for path in version_one_states
    } == version_one_states
    assert (version_two / "candidates.csv").read_bytes() == expected_candidates
    assert (version_two / "conflicts.csv").read_bytes() == expected_conflicts
    assert manifest_path.read_bytes() != original_manifest
    validate_manifest_inputs(
        manifest_path,
        version_two / "candidates.csv",
        version_two / "conflicts.csv",
    )
    with pytest.raises(ManifestError, match="input_sha256"):
        validate_manifest_inputs(
            manifest_path,
            version_one / "candidates.csv",
            version_one / "conflicts.csv",
        )


def test_refreeze_refuses_to_overwrite_an_existing_snapshot_version(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_dir = tmp_path / "metadata_inputs" / "v1"
    snapshot_dir.parent.mkdir()
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--snapshot-dir",
        str(snapshot_dir),
        "--output",
        str(manifest_path),
    ]
    assert main(arguments) == 0
    protected_paths = (
        manifest_path,
        snapshot_dir / "candidates.csv",
        snapshot_dir / "conflicts.csv",
    )
    states = {path: persistent_file_state(path) for path in protected_paths}

    with pytest.raises(ManifestError, match="snapshot version already exists"):
        main([*arguments, "--refreeze"])

    assert {
        path: persistent_file_state(path) for path in protected_paths
    } == states


def test_refreeze_requires_an_explicit_snapshot_version(tmp_path):
    candidates = [candidate_row("C0001", title="Original")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert main(arguments) == 0
    frozen_manifest = persistent_file_state(manifest_path)
    candidates[0]["title"] = "Changed"
    write_rows(candidates_path, candidates)

    with pytest.raises(ManifestError, match="requires --snapshot-dir"):
        main([*arguments, "--refreeze"])

    assert persistent_file_state(manifest_path) == frozen_manifest


@pytest.mark.parametrize("filename", ["candidates.csv", "conflicts.csv"])
def test_snapshot_validation_rejects_tampered_frozen_inputs(
    tmp_path,
    filename,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
        [conflict_row("CF0001", "C0001", "title")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_dir = tmp_path / "metadata_inputs" / "v1"
    snapshot_dir.parent.mkdir()
    assert (
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )
    tampered_path = snapshot_dir / filename
    original, replacement = {
        "candidates.csv": (b"Title C0001", b"Changed title"),
        "conflicts.csv": (b"new-CF0001", b"changed-conflict"),
    }[filename]
    tampered_path.write_bytes(
        tampered_path.read_bytes().replace(original, replacement)
    )
    protected_paths = (
        manifest_path,
        snapshot_dir / "candidates.csv",
        snapshot_dir / "conflicts.csv",
    )
    tampered_states = {
        path: persistent_file_state(path) for path in protected_paths
    }

    with pytest.raises(ManifestError, match="input_sha256"):
        main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )

    assert {
        path: persistent_file_state(path) for path in protected_paths
    } == tampered_states


@pytest.mark.parametrize("failure_point", ["snapshot", "manifest"])
def test_refreeze_publication_failure_preserves_the_previous_snapshot_set(
    tmp_path,
    monkeypatch,
    failure_point,
):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--snapshot-dir",
        str(version_one),
        "--output",
        str(manifest_path),
    ]
    assert main(base_arguments) == 0
    protected_paths = (
        manifest_path,
        version_one / "candidates.csv",
        version_one / "conflicts.csv",
    )
    protected_states = {
        path: persistent_file_state(path) for path in protected_paths
    }
    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} publication failure")

    if failure_point == "snapshot":
        real_rename = metadata_batches._rename_noreplace

        def fail_snapshot_rename(source, target):
            if Path(target) == version_two:
                fail()
            return real_rename(source, target)

        monkeypatch.setattr(
            metadata_batches,
            "_rename_noreplace",
            fail_snapshot_rename,
        )
    else:
        real_exchange = metadata_batches._rename_exchange

        def fail_manifest_exchange(source, target):
            if Path(target) == manifest_path:
                fail()
            return real_exchange(source, target)

        monkeypatch.setattr(
            metadata_batches,
            "_rename_exchange",
            fail_manifest_exchange,
        )

    with pytest.raises(
        OSError,
        match=rf"injected {failure_point} publication failure",
    ):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(version_two),
                "--output",
                str(manifest_path),
                "--refreeze",
            ]
        )

    assert {
        path: persistent_file_state(path) for path in protected_paths
    } == protected_states
    assert not version_two.exists()
    assert not list(snapshot_root.glob(".v2.*.tmp"))
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


def test_refreeze_rejects_manifest_inode_swap_during_backup(
    tmp_path,
    monkeypatch,
):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert (
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )
        == 0
    )

    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"
    racing_manifest = tmp_path / "racing-manifest.csv"
    racing_manifest.write_bytes(b"racing manifest\n")
    racing_manifest.chmod(0o640)
    racing_state = persistent_file_state(racing_manifest)

    real_link = metadata_batches.os.link

    def swap_manifest_after_backup(source, destination, *args, **kwargs):
        result = real_link(source, destination, *args, **kwargs)
        if Path(source) == manifest_path:
            racing_manifest.replace(manifest_path)
        return result

    monkeypatch.setattr(
        metadata_batches.os,
        "link",
        swap_manifest_after_backup,
    )

    with pytest.raises(
        ManifestError,
        match="changed while creating rollback backup",
    ):
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert persistent_file_state(manifest_path) == racing_state
    assert not version_two.exists()
    assert {path.name for path in snapshot_root.iterdir()} == {"v1"}
    assert not list(snapshot_root.glob(".v2.*.tmp"))
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


@pytest.mark.parametrize(
    "phase",
    ["before", "before_content", "at", "after", "after_content"],
)
def test_refreeze_final_exchange_preserves_a_racing_manifest(
    tmp_path,
    monkeypatch,
    phase,
):
    (
        candidates_path,
        _conflicts_path,
        manifest_path,
        _snapshot_root,
        version_two,
        base_arguments,
        original_manifest,
    ) = build_refreeze_case(tmp_path)
    racing_manifest = tmp_path / "racing-manifest.csv"
    racing_manifest.write_bytes(b"racing manifest\n")
    racing_manifest.chmod(0o640)
    expected_state: dict[str, tuple[bytes, int, int, int]] = {}

    real_boundary = metadata_batches._rename_exchange
    boundary_calls = 0

    def race_at_final_boundary(source, destination):
        nonlocal boundary_calls
        if Path(destination) != manifest_path or boundary_calls:
            return real_boundary(source, destination)
        boundary_calls += 1
        if phase in {"before", "at"}:
            racing_manifest.replace(manifest_path)
            expected_state["manifest"] = persistent_file_state(manifest_path)
        elif phase == "before_content":
            manifest_path.write_bytes(b"racing in-place content\n")
            manifest_path.chmod(0o640)
            expected_state["manifest"] = persistent_file_state(manifest_path)
        result = real_boundary(source, destination)
        if phase == "at":
            raise KeyboardInterrupt("injected final exchange interrupt")
        if phase == "after":
            racing_manifest.replace(manifest_path)
            expected_state["manifest"] = persistent_file_state(manifest_path)
        elif phase == "after_content":
            manifest_path.write_bytes(b"racing in-place content\n")
            manifest_path.chmod(0o640)
            expected_state["manifest"] = persistent_file_state(manifest_path)
        return result

    monkeypatch.setattr(
        metadata_batches,
        "_rename_exchange",
        race_at_final_boundary,
    )

    error_type = KeyboardInterrupt if phase == "at" else ManifestError
    error_match = (
        "injected final exchange interrupt"
        if phase == "at"
        else "manifest changed during refreeze"
    )
    with pytest.raises(error_type, match=error_match):
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert persistent_file_state(manifest_path) == expected_state["manifest"]
    assert not version_two.exists()
    recoveries = recovery_directories(manifest_path)
    if phase in {"after", "after_content"}:
        assert len(recoveries) == 1
        recovery_dir = recoveries[0]
        assert (recovery_dir / "old-manifest.csv").read_bytes() == original_manifest
        assert (
            recovery_dir / "snapshot" / "candidates.csv"
        ).read_bytes() == candidates_path.read_bytes()
    else:
        assert recoveries == []


def test_incomplete_manifest_restoration_retains_recovery_state(
    tmp_path,
    monkeypatch,
):
    (
        candidates_path,
        _conflicts_path,
        manifest_path,
        _snapshot_root,
        version_two,
        base_arguments,
        original_manifest,
    ) = build_refreeze_case(tmp_path)
    real_boundary = metadata_batches._rename_exchange
    primary_error = OSError("primary final publication failure")
    boundary_calls = 0

    def fail_publication_then_restoration(source, destination):
        nonlocal boundary_calls
        if Path(destination) != manifest_path:
            return real_boundary(source, destination)
        boundary_calls += 1
        if boundary_calls == 1:
            real_boundary(source, destination)
            raise primary_error
        raise OSError("injected restore exchange failure")

    monkeypatch.setattr(
        metadata_batches,
        "_rename_exchange",
        fail_publication_then_restoration,
    )

    with pytest.raises(
        OSError,
        match="primary final publication failure",
    ) as caught:
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert caught.value is primary_error
    assert not version_two.exists()
    recoveries = recovery_directories(manifest_path)
    assert len(recoveries) == 1
    recovery_dir = recoveries[0]
    assert (recovery_dir / "old-manifest.csv").read_bytes() == original_manifest
    assert (
        recovery_dir / "snapshot" / "candidates.csv"
    ).read_bytes() == candidates_path.read_bytes()
    report = " ".join(
        [
            *getattr(caught.value, "__notes__", ()),
            *(str(argument) for argument in caught.value.args),
        ]
    )
    assert "injected restore exchange failure" in report
    assert f"recovery retained at {recovery_dir}" in report


def test_committed_v1_snapshot_raw_bytes_are_frozen():
    repository_root = Path(__file__).resolve().parents[1]
    data_dir = repository_root / "paper" / "data"
    snapshot_dir = data_dir / "metadata_inputs" / "v1"
    expected_files = {
        "candidates.csv": (
            119864,
            "62b7fc3a2716f923422b77d538e9cfb4c95cefb1687bf979af4cb953656e90a3",
        ),
        "conflicts.csv": (
            75130,
            "4495d57179822dd099299825015bc27a6ddf91e397ecbba8a4ac63ec1363ca52",
        ),
    }

    for filename, (expected_size, expected_sha256) in expected_files.items():
        payload = (snapshot_dir / filename).read_bytes()
        assert len(payload) == expected_size
        assert hashlib.sha256(payload).hexdigest() == expected_sha256



def test_committed_v1_snapshots_validate_manifest_semantically():
    repository_root = Path(__file__).resolve().parents[1]
    data_dir = repository_root / "paper" / "data"
    snapshot_dir = data_dir / "metadata_inputs" / "v1"

    manifest_path = data_dir / "metadata_manifest.csv"
    validate_manifest_inputs(
        manifest_path,
        snapshot_dir / "candidates.csv",
        snapshot_dir / "conflicts.csv",
    )
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        snapshot_hashes = {
            row["snapshot_sha256"] for row in csv.DictReader(handle)
        }
    assert snapshot_hashes == {
        "303e6b0efb7a94be636ce0b22de9454062029ef9193ef5e0442aee263227ecec"
    }


def replay_committed_metadata(
    repository_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    data_dir = repository_root / "paper" / "data"
    runs_dir = data_dir / "metadata_runs"
    metadata_results = sorted(runs_dir.glob("metadata-[0-9][0-9].csv"))
    conflict_results = sorted(
        runs_dir.glob("metadata-[0-9][0-9]-conflicts.csv")
    )
    assert len(metadata_results) == 6
    assert len(conflict_results) == 6

    output_dir.mkdir()
    outputs = {
        "candidates.csv": output_dir / "candidates.csv",
        "conflicts.csv": output_dir / "conflicts.csv",
        "bibliography.csv": output_dir / "bibliography.csv",
        "references.bib": output_dir / "references.bib",
    }
    snapshot_dir = data_dir / "metadata_inputs" / "v1"
    parser_options = {
        option
        for action in metadata_integration._argument_parser()._actions
        for option in action.option_strings
    }
    arguments = [
        "--candidates",
        str(snapshot_dir / "candidates.csv"),
        "--conflicts",
        str(snapshot_dir / "conflicts.csv"),
        "--manifest",
        str(data_dir / "metadata_manifest.csv"),
    ]
    if "--citation-keys" in parser_options:
        citation_keys_path = data_dir / "citation_keys.csv"
        if not citation_keys_path.is_file():
            pytest.skip(
                "citation-key CLI is present but its ledger is not installed"
            )
        arguments.extend(
            ["--citation-keys", str(citation_keys_path)]
        )
    for result_path in metadata_results:
        arguments.extend(["--metadata-result", str(result_path)])
    for result_path in conflict_results:
        arguments.extend(["--conflict-result", str(result_path)])
    arguments.extend(
        [
            "--output-candidates",
            str(outputs["candidates.csv"]),
            "--output-conflicts",
            str(outputs["conflicts.csv"]),
            "--output-bibliography",
            str(outputs["bibliography.csv"]),
            "--output-bibtex",
            str(outputs["references.bib"]),
        ]
    )

    assert metadata_integration.main(arguments) == 0
    return outputs


def test_committed_metadata_replay_is_byte_deterministic(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]

    first = replay_committed_metadata(repository_root, tmp_path / "first")
    second = replay_committed_metadata(repository_root, tmp_path / "second")

    assert {
        filename: path.read_bytes() for filename, path in first.items()
    } == {
        filename: path.read_bytes() for filename, path in second.items()
    }


def test_committed_metadata_replay_matches_current_canonical_artifacts(
    tmp_path,
):
    repository_root = Path(__file__).resolve().parents[1]
    actual = replay_committed_metadata(repository_root, tmp_path / "replay")
    expected = {
        "candidates.csv": repository_root / "paper" / "data" / "candidates.csv",
        "conflicts.csv": repository_root / "paper" / "data" / "conflicts.csv",
        "bibliography.csv": (
            repository_root / "paper" / "data" / "bibliography.csv"
        ),
        "references.bib": repository_root / "paper" / "references.bib",
    }

    mismatches = [
        filename
        for filename, expected_path in expected.items()
        if actual[filename].read_bytes() != expected_path.read_bytes()
    ]
    assert not mismatches, (
        f"metadata replay differs from canonical artifacts: {mismatches}"
    )

def test_refreeze_refuses_to_replace_a_dangling_snapshot_version_symlink(
    tmp_path,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    direct_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert main(direct_arguments) == 0
    manifest_state = persistent_file_state(manifest_path)

    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    snapshot_dir = snapshot_root / "v2"
    snapshot_dir.symlink_to("missing-version", target_is_directory=True)

    with pytest.raises(ManifestError, match="snapshot version already exists"):
        main(
            [
                *direct_arguments,
                "--snapshot-dir",
                str(snapshot_dir),
                "--refreeze",
            ]
        )

    assert persistent_file_state(manifest_path) == manifest_state
    assert snapshot_dir.is_symlink()
    assert snapshot_dir.readlink() == Path("missing-version")

@pytest.mark.parametrize("boundary", ["snapshot", "manifest"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_post_publication_base_exception_restores_previous_snapshot_set(
    tmp_path,
    monkeypatch,
    boundary,
    error_type,
):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert (
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )
        == 0
    )
    protected_paths = (
        manifest_path,
        version_one / "candidates.csv",
        version_one / "conflicts.csv",
    )
    protected_states = {
        path: persistent_file_state(path) for path in protected_paths
    }

    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"
    message = f"post-{boundary} {error_type.__name__}"

    if boundary == "snapshot":
        real_boundary = getattr(
            metadata_batches,
            "_rename_noreplace",
            lambda source, destination: source.rename(destination),
        )

        def fail_after_snapshot_publish(source, destination):
            real_boundary(source, destination)
            raise error_type(message)

        monkeypatch.setattr(
            metadata_batches,
            "_rename_noreplace",
            fail_after_snapshot_publish,
            raising=False,
        )
    else:
        real_boundary = metadata_batches._rename_exchange

        def fail_after_manifest_exchange(source, destination):
            real_boundary(source, destination)
            raise error_type(message)

        monkeypatch.setattr(
            metadata_batches,
            "_rename_exchange",
            fail_after_manifest_exchange,
        )

    with pytest.raises(error_type, match=message):
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert {
        path: persistent_file_state(path) for path in protected_paths
    } == protected_states
    assert not version_two.exists()
    assert not version_two.is_symlink()
    assert {path.name for path in snapshot_root.iterdir()} == {"v1"}
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


@pytest.mark.parametrize(
    "rollback_failure",
    [
        "manifest restoration",
        "manifest ownership probe",
        "post-restore ownership probe",
        "no-exchange content probe",
    ],
)
def test_rollback_failure_preserves_primary_exception_and_reports_note(
    tmp_path,
    monkeypatch,
    rollback_failure,
):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert (
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )
        == 0
    )
    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"

    primary_error = OSError("primary manifest replacement failure")
    real_exchange = metadata_batches._rename_exchange
    manifest_exchange_calls = 0

    def fail_primary_then_restore(source, destination):
        nonlocal manifest_exchange_calls
        if Path(destination) == manifest_path:
            manifest_exchange_calls += 1
            if manifest_exchange_calls == 1:
                if rollback_failure == "no-exchange content probe":
                    raise primary_error
                real_exchange(source, destination)
                raise primary_error
            if rollback_failure == "post-restore ownership probe":
                return real_exchange(source, destination)
            raise OSError("injected manifest restoration failure")
        return real_exchange(source, destination)

    monkeypatch.setattr(
        metadata_batches,
        "_rename_exchange",
        fail_primary_then_restore,
    )

    if rollback_failure == "manifest ownership probe":
        real_path_identity = metadata_batches._path_identity
        identity_probe_failed = False

        def fail_rollback_identity_probe(path):
            nonlocal identity_probe_failed
            if (
                manifest_exchange_calls
                and Path(path) == manifest_path
                and not identity_probe_failed
            ):
                identity_probe_failed = True
                raise KeyboardInterrupt(
                    "injected manifest ownership probe failure"
                )
            return real_path_identity(path)

        monkeypatch.setattr(
            metadata_batches,
            "_path_identity",
            fail_rollback_identity_probe,
        )

    elif rollback_failure == "post-restore ownership probe":
        real_has_identity = metadata_batches._path_has_identity
        identity_probe_failed = False

        def fail_post_restore_identity_probe(path, identity):
            nonlocal identity_probe_failed
            if (
                manifest_exchange_calls >= 2
                and Path(path) == manifest_path
                and not identity_probe_failed
            ):
                identity_probe_failed = True
                raise KeyboardInterrupt(
                    "injected post-restore ownership probe failure"
                )
            return real_has_identity(path, identity)

        monkeypatch.setattr(
            metadata_batches,
            "_path_has_identity",
            fail_post_restore_identity_probe,
        )

    elif rollback_failure == "no-exchange content probe":
        real_has_identity = metadata_batches._path_has_identity
        identity_probe_failed = False

        def fail_no_exchange_content_probe(path, identity):
            nonlocal identity_probe_failed
            if (
                manifest_exchange_calls
                and Path(path) == manifest_path
                and not identity_probe_failed
            ):
                identity_probe_failed = True
                raise KeyboardInterrupt(
                    "injected no-exchange content probe failure"
                )
            return real_has_identity(path, identity)

        monkeypatch.setattr(
            metadata_batches,
            "_path_has_identity",
            fail_no_exchange_content_probe,
        )

    with pytest.raises(
        OSError,
        match="primary manifest replacement failure",
    ) as caught:
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert caught.value is primary_error
    assert any(
        "rollback manifest restoration failed" in note
        and f"injected {rollback_failure} failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not version_two.exists()
    assert {path.name for path in snapshot_root.iterdir()} == {"v1"}


def test_rollback_reporting_is_compatible_without_baseexception_add_note():
    class Python310StyleError(OSError):
        add_note = None

    primary_error = Python310StyleError("primary failure")
    rollback_error = KeyboardInterrupt("rollback interrupt")

    metadata_batches._record_rollback_error(
        primary_error,
        "manifest restoration",
        rollback_error,
    )

    report = " ".join(str(argument) for argument in primary_error.args)
    report += " " + " ".join(
        getattr(primary_error, "__rollback_notes__", ())
    )
    assert "rollback manifest restoration failed" in report
    assert "KeyboardInterrupt: rollback interrupt" in report


@pytest.mark.parametrize(
    "entry_kind",
    ["empty_directory", "dangling_symlink", "live_symlink"],
)
def test_snapshot_publication_never_replaces_a_racing_entry(
    tmp_path,
    monkeypatch,
    entry_kind,
):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert (
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )
        == 0
    )
    manifest_state = persistent_file_state(manifest_path)
    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"
    live_target = snapshot_root / "racing-target"

    real_rename = metadata_batches._rename_noreplace

    def create_racing_entry_then_publish(source, destination):
        if entry_kind == "empty_directory":
            destination.mkdir()
        elif entry_kind == "dangling_symlink":
            destination.symlink_to(
                "missing-racing-target",
                target_is_directory=True,
            )
        else:
            live_target.mkdir()
            destination.symlink_to(
                live_target,
                target_is_directory=True,
            )
        return real_rename(source, destination)

    monkeypatch.setattr(
        metadata_batches,
        "_rename_noreplace",
        create_racing_entry_then_publish,
    )

    with pytest.raises(ManifestError, match="snapshot version already exists"):
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert persistent_file_state(manifest_path) == manifest_state
    if entry_kind == "empty_directory":
        assert version_two.is_dir()
        assert not any(version_two.iterdir())
    else:
        assert version_two.is_symlink()
        expected_target = (
            Path("missing-racing-target")
            if entry_kind == "dangling_symlink"
            else live_target
        )
        assert version_two.readlink() == expected_target
    assert not list(snapshot_root.glob(".v2.*.tmp"))
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


@pytest.mark.parametrize(
    "symlink_target",
    ["snapshot_dir", "candidates.csv", "conflicts.csv", "manifest"],
)
@pytest.mark.parametrize("dangling", [False, True])
def test_snapshot_validation_rejects_symlinked_contract_paths(
    tmp_path,
    symlink_target,
    dangling,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    snapshot_dir = snapshot_root / "v1"
    assert (
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )

    if symlink_target == "snapshot_dir":
        real_path = snapshot_root / "real-v1"
        snapshot_dir.rename(real_path)
        link_target = (
            snapshot_root / ("missing-v1" if dangling else "real-v1")
        )
        snapshot_dir.symlink_to(link_target, target_is_directory=True)
    elif symlink_target == "manifest":
        real_path = tmp_path / "real-manifest.csv"
        manifest_path.rename(real_path)
        link_target = (
            tmp_path / ("missing-manifest.csv" if dangling else real_path.name)
        )
        manifest_path.symlink_to(link_target)
    else:
        path = snapshot_dir / symlink_target
        real_path = snapshot_dir / f"real-{symlink_target}"
        path.rename(real_path)
        link_target = (
            f"missing-{symlink_target}" if dangling else real_path.name
        )
        path.symlink_to(link_target)

    with pytest.raises(ManifestError, match="symlink"):
        main(
            [
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )


@pytest.mark.parametrize("input_name", ["candidates", "conflicts"])
def test_direct_validation_rejects_hard_linked_input_output_alias(
    tmp_path,
    input_name,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    inputs = {"candidates": candidates_path, "conflicts": conflicts_path}
    input_path = inputs[input_name]
    output_path = tmp_path / "metadata_manifest.csv"
    os.link(input_path, output_path)
    input_state = persistent_file_state(input_path)

    with pytest.raises(ManifestError, match=rf"aliases {input_name}"):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--output",
                str(output_path),
            ]
        )

    assert persistent_file_state(input_path) == input_state
    assert os.path.samefile(input_path, output_path)


@pytest.mark.parametrize("manifest_state", ["missing", "directory", "symlink"])
def test_refreeze_requires_existing_regular_non_symlink_manifest(
    tmp_path,
    manifest_state,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    output_path = tmp_path / "metadata_manifest.csv"
    if manifest_state == "directory":
        output_path.mkdir()
    elif manifest_state == "symlink":
        real_manifest = tmp_path / "real-manifest.csv"
        real_manifest.write_text("manifest sentinel\n", encoding="utf-8")
        output_path.symlink_to(real_manifest)

    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_two = snapshot_root / "v2"

    with pytest.raises(
        ManifestError,
        match="refreeze requires an existing regular non-symlink manifest",
    ):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(version_two),
                "--output",
                str(output_path),
                "--refreeze",
            ]
        )

    assert not version_two.exists()
    assert not version_two.is_symlink()


def test_initial_snapshot_freeze_refuses_an_existing_manifest(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    output_path = tmp_path / "metadata_manifest.csv"
    direct_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
    ]
    assert main(direct_arguments) == 0
    manifest_state = persistent_file_state(output_path)

    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    with pytest.raises(
        ManifestError,
        match="initial snapshot freeze requires an absent manifest",
    ):
        main(
            [
                *direct_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )

    assert persistent_file_state(output_path) == manifest_state
    assert not version_one.exists()


def test_initial_freeze_never_replaces_a_racing_manifest(
    tmp_path,
    monkeypatch,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    snapshot_dir = snapshot_root / "v1"
    racing_manifest = tmp_path / "racing-manifest.csv"
    racing_manifest.write_bytes(b"racing manifest\n")
    racing_manifest.chmod(0o640)
    racing_state = persistent_file_state(racing_manifest)

    real_rename = metadata_batches._rename_noreplace

    def create_manifest_after_snapshot_publish(source, destination):
        result = real_rename(source, destination)
        if Path(destination) == snapshot_dir:
            racing_manifest.replace(manifest_path)
        return result

    monkeypatch.setattr(
        metadata_batches,
        "_rename_noreplace",
        create_manifest_after_snapshot_publish,
    )

    with pytest.raises(ManifestError, match="manifest already exists"):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--output",
                str(manifest_path),
            ]
        )

    assert persistent_file_state(manifest_path) == racing_state
    assert not snapshot_dir.exists()
    assert {path.name for path in snapshot_root.iterdir()} == set()
    assert not list(snapshot_root.glob(".v1.*.tmp"))
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


@pytest.mark.parametrize(
    "entry_kind",
    ["regular_file", "dangling_symlink", "empty_directory"],
)
def test_legacy_manifest_creation_never_replaces_a_late_entry(
    tmp_path,
    monkeypatch,
    entry_kind,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    symlink_target = tmp_path / "missing-manifest-target.csv"
    real_stage_manifest = metadata_batches._stage_manifest

    def stage_then_create_racing_entry(path, rows):
        staged_path = real_stage_manifest(path, rows)
        if entry_kind == "regular_file":
            manifest_path.write_bytes(b"racing manifest\n")
            manifest_path.chmod(0o640)
        elif entry_kind == "dangling_symlink":
            manifest_path.symlink_to(symlink_target)
        else:
            manifest_path.mkdir()
        return staged_path

    monkeypatch.setattr(
        metadata_batches,
        "_stage_manifest",
        stage_then_create_racing_entry,
    )

    with pytest.raises(ManifestError, match="manifest already exists"):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--output",
                str(manifest_path),
            ]
        )

    if entry_kind == "regular_file":
        assert manifest_path.read_bytes() == b"racing manifest\n"
        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o640
    elif entry_kind == "dangling_symlink":
        assert manifest_path.is_symlink()
        assert manifest_path.readlink() == symlink_target
    else:
        assert manifest_path.is_dir()
        assert not any(manifest_path.iterdir())
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))


def test_snapshot_publication_sets_deterministic_permissions_despite_umask(
    tmp_path,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_dir = tmp_path / "metadata_inputs" / "v1"
    snapshot_dir.parent.mkdir()

    previous_umask = os.umask(0o077)
    try:
        assert (
            main(
                [
                    "--candidates",
                    str(candidates_path),
                    "--conflicts",
                    str(conflicts_path),
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--output",
                    str(manifest_path),
                ]
            )
            == 0
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(snapshot_dir.stat().st_mode) == 0o755
    assert (
        stat.S_IMODE((snapshot_dir / "candidates.csv").stat().st_mode)
        == 0o644
    )
    assert (
        stat.S_IMODE((snapshot_dir / "conflicts.csv").stat().st_mode)
        == 0o644
    )


def test_git_treats_metadata_input_snapshots_as_binary():
    repository_root = Path(__file__).resolve().parents[1]
    snapshot_path = "paper/data/metadata_inputs/v1/candidates.csv"

    completed = subprocess.run(
        [
            "git",
            "check-attr",
            "diff",
            "merge",
            "text",
            "--",
            snapshot_path,
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        f"{snapshot_path}: diff: unset",
        f"{snapshot_path}: merge: unset",
        f"{snapshot_path}: text: unset",
    ]


@pytest.mark.parametrize("input_name", ["candidates", "conflicts"])
def test_initial_freeze_rejects_symlinked_direct_inputs(
    tmp_path,
    input_name,
):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    paths = {
        "candidates": candidates_path,
        "conflicts": conflicts_path,
    }
    path = paths[input_name]
    real_path = path.with_name(f"real-{path.name}")
    path.rename(real_path)
    path.symlink_to(real_path.name)

    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    with pytest.raises(ManifestError, match=rf"{input_name}.*symlink"):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_root / "v1"),
                "--output",
                str(tmp_path / "metadata_manifest.csv"),
            ]
        )


def test_initial_freeze_rejects_symlinked_snapshot_parent(tmp_path):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        [candidate_row("C0001")],
    )
    real_snapshot_root = tmp_path / "real-metadata-inputs"
    real_snapshot_root.mkdir()
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.symlink_to(real_snapshot_root, target_is_directory=True)

    with pytest.raises(
        ManifestError,
        match="snapshot parent directory.*symlink",
    ):
        main(
            [
                "--candidates",
                str(candidates_path),
                "--conflicts",
                str(conflicts_path),
                "--snapshot-dir",
                str(snapshot_root / "v1"),
                "--output",
                str(tmp_path / "metadata_manifest.csv"),
            ]
        )

    assert not (real_snapshot_root / "v1").exists()


def test_metadata_readme_shell_blocks_are_strict_and_self_contained():
    repository_root = Path(__file__).resolve().parents[1]
    readme = (repository_root / "paper" / "data" / "README.md").read_text(
        encoding="utf-8"
    )
    section = readme.split(
        "## `metadata_manifest.csv` and `metadata_runs/`",
        maxsplit=1,
    )[1].split("## `candidate_aliases.csv`", maxsplit=1)[0]
    bash_blocks = [
        block.split("\n~~~", maxsplit=1)[0]
        for block in section.split("~~~bash\n")[1:]
    ]

    assert len(bash_blocks) == 5
    assert all(
        block.startswith("set -euo pipefail\n")
        for block in bash_blocks
    )
    (
        raw_hash_block,
        validation_block,
        replay_block,
        publication_block,
        refreeze_block,
    ) = bash_blocks
    assert "sha256sum --check --strict" in raw_hash_block
    assert "prepare_metadata_batches.py" not in raw_hash_block
    assert "--snapshot-dir paper/data/metadata_inputs/v1" in validation_block
    assert "sha256sum" not in validation_block
    assert "metadata_replay_args=(" in replay_block
    assert "--citation-keys paper/data/citation_keys.csv" in replay_block
    assert "trap cleanup EXIT" in replay_block
    assert replay_block.count("\ncmp -- ") == 4
    assert "|| true" not in replay_block
    assert "metadata_replay_args=(" in publication_block
    assert "--citation-keys paper/data/citation_keys.csv" in publication_block
    assert 'snapshot_dir="paper/data/metadata_inputs/v2"' in refreeze_block
    assert ".metadata_manifest.csv.recovery.*" in section
    assert "old-manifest.csv" in section
    assert "swapped-manifest.csv" in section
    assert "bounded guarantee" in section


def test_refreeze_rejects_manifest_swapped_to_symlink_before_backup(
    tmp_path,
    monkeypatch,
):
    candidates = [candidate_row("C0001", title="Version one")]
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs",
        candidates,
    )
    manifest_path = tmp_path / "metadata_manifest.csv"
    snapshot_root = tmp_path / "metadata_inputs"
    snapshot_root.mkdir()
    version_one = snapshot_root / "v1"
    base_arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(manifest_path),
    ]
    assert (
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_one),
            ]
        )
        == 0
    )

    candidates[0]["title"] = "Version two"
    write_rows(candidates_path, candidates)
    version_two = snapshot_root / "v2"
    displaced_manifest = tmp_path / "displaced-manifest.csv"
    symlink_target = tmp_path / "racing-manifest.csv"
    symlink_target.write_text("racing manifest\n", encoding="utf-8")

    real_stage_manifest = metadata_batches._stage_manifest

    def stage_then_swap_manifest(path, rows):
        staged = real_stage_manifest(path, rows)
        manifest_path.replace(displaced_manifest)
        manifest_path.symlink_to(symlink_target)
        return staged

    monkeypatch.setattr(
        metadata_batches,
        "_stage_manifest",
        stage_then_swap_manifest,
    )

    with pytest.raises(ManifestError, match="manifest must not be a symlink"):
        main(
            [
                *base_arguments,
                "--snapshot-dir",
                str(version_two),
                "--refreeze",
            ]
        )

    assert manifest_path.is_symlink()
    assert manifest_path.readlink() == symlink_target
    assert displaced_manifest.is_file()
    assert not version_two.exists()
    assert {path.name for path in snapshot_root.iterdir()} == {"v1"}
    assert not list(snapshot_root.glob(".v2.*.tmp"))
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))
