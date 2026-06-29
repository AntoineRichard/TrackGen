from __future__ import annotations

import csv
import stat
from collections import Counter, defaultdict
from pathlib import Path

import pytest
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
    original_state = file_state(output_path)
    candidates[0]["title"] = "Changed"
    write_rows(candidates_path, candidates)

    assert main([*arguments, "--refreeze"]) == 0

    assert output_path.read_bytes() != original_state[0]
    assert stat.S_IMODE(output_path.stat().st_mode) == original_state[1]
    validate_manifest_inputs(output_path, candidates_path, conflicts_path)


def test_refreeze_validates_inputs_before_replacing_existing_output(tmp_path):
    candidates = [candidate_row("C0001")]
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
    frozen_state = file_state(output_path)
    write_rows(candidates_path, [candidates[0], dict(candidates[0])])

    with pytest.raises(ManifestError, match="duplicate candidate_id"):
        main([*arguments, "--refreeze"])

    assert file_state(output_path) == frozen_state


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


@pytest.mark.parametrize("failure_point", ["write", "fsync", "chmod", "replace"])
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
    arguments = [
        "--candidates",
        str(candidates_path),
        "--conflicts",
        str(conflicts_path),
        "--output",
        str(output_path),
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
        monkeypatch.setattr(type(output_path), "replace", fail)

    with pytest.raises(OSError, match=rf"injected {failure_point} failure"):
        main(arguments)

    assert file_state(output_path) == frozen_state
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


@pytest.mark.parametrize("output_name", ["candidates", "conflicts"])
def test_output_alias_is_rejected_for_each_input(tmp_path, output_name):
    candidates_path, conflicts_path = build_inputs(
        tmp_path / "inputs", [candidate_row("C0001")]
    )
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
        "--refreeze",
    ]

    with pytest.raises(ManifestError, match=rf"differ from {output_name} input"):
        main(arguments)

    assert {
        name: file_state(path) for name, path in paths.items()
    } == original_states
    assert not list((tmp_path / "inputs").glob(".*.tmp"))
