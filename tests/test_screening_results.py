from __future__ import annotations

import csv
import errno
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import is_dataclass, replace
from pathlib import Path

import pytest

import paper.scripts.screening_results as screening_results
from paper.scripts.prepare_screening_batches import MANIFEST_HEADER
from tests.test_screening_batches import build_inputs, freeze


RESULT_HEADER = (
    "assignment_id",
    "phase",
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "batch_id",
    "coder_id",
    "screened_on",
    "screening_status",
    "criterion",
    "access_status",
    "source_urls",
    "evidence_version",
    "evidence_retrieved_on",
    "evidence_archive_url",
    "evidence_sha256",
    "screening_locator",
    "exclusion_reason",
    "notes",
)

PHASE_MANIFEST_HEADER = (
    "manifest_version",
    "phase_result_snapshot_sha256",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "reviewer_release_sha256",
    "phase",
    "batch_id",
    "coder_id",
    "result_filename",
    "result_file_sha256",
    "row_count",
)


DECISION_HEADER = (
    "decision_id",
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "candidate_ids_sha256",
    "assignment_ids_sha256",
    "status_agreement_numerator",
    "status_agreement_denominator",
    "status_agreement",
    "systematic_ambiguity",
    "decision",
    "decided_on",
    "decision_makers",
    "resolution_evidence",
)

DECISION_MANIFEST_HEADER = (
    "manifest_version",
    "calibration_decision_snapshot_sha256",
    "protocol_sha256",
    "coordinator_snapshot_sha256",
    "calibration_result_snapshot_sha256",
    "decision_id",
    "decision_file_sha256",
    "candidate_ids_file_sha256",
    "assignment_ids_file_sha256",
    "row_count",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "paper" / "scripts" / "screening_results.py"


def test_public_facade_dataclasses_and_phase_coordinator_keyword() -> None:
    assert is_dataclass(screening_results.CoordinatorSnapshot)
    assert is_dataclass(screening_results.ReviewerReleaseSnapshot)
    assert is_dataclass(screening_results.CapturedInput)
    assert screening_results._Coordinator is screening_results.CoordinatorSnapshot
    assert screening_results._CapturedInput is screening_results.CapturedInput

    parameters = inspect.signature(
        screening_results.validate_phase_result_snapshot
    ).parameters
    assert "coordinator" in parameters
    assert "_coordinator" not in parameters
    assert (
        parameters["reviewer_release_snapshot_dir"].default
        is inspect.Parameter.empty
    )
    seal_parameters = inspect.signature(
        screening_results.seal_phase_results
    ).parameters
    assert (
        seal_parameters["reviewer_release_snapshot_dir"].default
        is inspect.Parameter.empty
    )
    assert {
        "calibration_reviewer_release_snapshot_dir",
        "calibration_result_snapshot_dir",
        "calibration_decision_snapshot_dir",
    } <= set(parameters) & set(seal_parameters)


def test_public_facade_dynamically_delegates_to_private_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = object()
    fingerprint_groups = object()
    protected = (Path("protected"),)
    cases = (
        (
            "parse_canonical_csv",
            "_parse_csv",
            (b"payload", "label", ("field",)),
            {"no_blank_cells": False},
            (b"payload", "label", ("field",)),
        ),
        (
            "render_canonical_csv",
            "_csv_bytes",
            (("field",), [{"field": "value"}]),
            {},
            (("field",), [{"field": "value"}]),
        ),
        (
            "render_sha256sums",
            "_checksums",
            ({"artifact": b"payload"},),
            {},
            ({"artifact": b"payload"},),
        ),
        (
            "capture_coordinator_snapshot",
            "_capture_coordinator",
            (Path("coordinator"),),
            {},
            (Path("coordinator"),),
        ),
        (
            "reattest_coordinator_snapshot",
            "_reattest_coordinator",
            (coordinator,),
            {},
            (coordinator,),
        ),
        (
            "reattest_snapshot_set",
            "_coherent_final_attestation",
            (coordinator, fingerprint_groups),
            {},
            (fingerprint_groups, coordinator),
        ),
        (
            "capture_input",
            "_capture_input",
            (Path("input.csv"), "input"),
            {},
            (Path("input.csv"), "input"),
        ),
        (
            "capture_flat_snapshot",
            "_capture_flat_snapshot",
            (Path("v1"), ("one.csv",)),
            {},
            (Path("v1"), ("one.csv",)),
        ),
        (
            "reject_output_overlap",
            "_reject_output_overlap",
            (Path("output"), protected),
            {},
            (Path("output"), protected),
        ),
        (
            "validate_iso_date",
            "_validate_iso_date",
            ("2026-06-30",),
            {"field": "date", "context": "row"},
            ("2026-06-30",),
        ),
        (
            "validate_result_decision",
            "_validate_result_decision",
            ({"candidate_id": "C0001"},),
            {"context": "row"},
            ({"candidate_id": "C0001"},),
        ),
    )

    sentinel = object()
    for public_name, private_name, args, kwargs, expected_args in cases:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def private(*actual_args, **actual_kwargs):
            calls.append((actual_args, actual_kwargs))
            return sentinel

        monkeypatch.setattr(screening_results, private_name, private)
        assert getattr(screening_results, public_name)(*args, **kwargs) is sentinel
        assert calls == [(expected_args, kwargs)]




def test_public_snapshot_publisher_dynamically_delegates_to_private_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = screening_results.screening_batches
    output = tmp_path / "v1"
    artifacts = {"manifest.csv": b"payload"}
    callback = lambda: None
    calls: list[tuple[Path, dict[str, bytes], object]] = []

    def private_publisher(
        output_dir: Path,
        supplied_artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        calls.append((output_dir, supplied_artifacts, post_publish_check))

    monkeypatch.setattr(publisher, "_publish_artifacts", private_publisher)
    publisher.publish_snapshot(
        output,
        artifacts,
        post_publish_check=callback,
    )

    assert calls == [(output, artifacts, callback)]


def test_public_identifier_validator_is_total_for_external_inputs() -> None:
    assert screening_results.is_valid_identifier("reviewer-01")
    assert not screening_results.is_valid_identifier("x")
    assert not screening_results.is_valid_identifier("contains whitespace")
    assert not screening_results.is_valid_identifier(None)


def _write_csv(
    path: Path, header: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=header, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        return list(reader)


@pytest.fixture(scope="module")
def coordinator_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("screening-results-coordinator")
    inputs = build_inputs(root / "inputs", count=202)
    snapshot = root / "v1"
    freeze(inputs, snapshot)
    return snapshot


@pytest.fixture
def coordinator(
    tmp_path: Path, coordinator_template: Path
) -> Path:
    destination = tmp_path / "coordinator" / "v1"
    destination.parent.mkdir()
    # copytree intentionally gives every test independent mutable inodes.
    import shutil

    shutil.copytree(coordinator_template, destination)
    return destination


@pytest.fixture(scope="module")
def v6_coordinator_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("screening-results-v6-coordinator")
    inputs = build_inputs(root / "inputs", count=202)
    taxonomy = json.loads(inputs.taxonomy.read_text(encoding="utf-8"))
    taxonomy["screening_inclusion_criterion"] = ["include-relevant"]
    inputs.taxonomy.write_bytes(
        screening_results.screening_batches._canonical_json_bytes(taxonomy)
    )
    snapshot = root / "v6"
    freeze(inputs, snapshot)
    return snapshot


@pytest.fixture
def v6_coordinator(
    tmp_path: Path, v6_coordinator_template: Path
) -> Path:
    destination = tmp_path / "coordinator" / "v6"
    destination.parent.mkdir()
    import shutil

    shutil.copytree(v6_coordinator_template, destination)
    return destination


def _manifest(coordinator: Path) -> list[dict[str, str]]:
    with (coordinator / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, strict=True)
        assert tuple(reader.fieldnames or ()) == MANIFEST_HEADER
        return list(reader)


def _decision_for_assignment(
    manifest_row: dict[str, str],
    *,
    status: str = "included",
) -> dict[str, str]:
    if status == "included":
        criterion = "include-1"
        exclusion_reason = "NR"
    elif status == "boundary":
        criterion = "boundary"
        exclusion_reason = "NR"
    else:
        criterion = "exclude-out-of-scope"
        exclusion_reason = (
            "The inspected source does not contribute course-generation "
            "geometry, representations, metrics, or interchange."
        )
    candidate_id = manifest_row["candidate_id"]
    return {
        "assignment_id": manifest_row["assignment_id"],
        "phase": manifest_row["phase"],
        "candidate_id": candidate_id,
        "input_sha256": manifest_row["input_sha256"],
        "snapshot_sha256": manifest_row["snapshot_sha256"],
        "batch_id": manifest_row["batch_id"],
        "coder_id": manifest_row["batch_id"],
        "screened_on": "2026-06-29",
        "screening_status": status,
        "criterion": criterion,
        "access_status": "full_text",
        "source_urls": f"https://example.test/source/{candidate_id}",
        "evidence_version": "publisher-version-of-record",
        "evidence_retrieved_on": "2026-06-28",
        "evidence_archive_url": (
            f"https://archive.example.test/20260628/{candidate_id}"
        ),
        "evidence_sha256": hashlib.sha256(
            f"evidence:{candidate_id}".encode("ascii")
        ).hexdigest(),
        "screening_locator": "Section 2; Algorithm 1",
        "exclusion_reason": exclusion_reason,
        "notes": "NR",
    }


@pytest.mark.parametrize("criterion", screening_results.INCLUSION_CRITERIA)
def test_unbound_decision_validation_retains_legacy_inclusion_values(
    coordinator: Path,
    criterion: str,
) -> None:
    row = _decision_for_assignment(_manifest(coordinator)[0])
    row["criterion"] = criterion

    screening_results.validate_result_decision(row, context="unbound")


def test_unbound_decision_validation_rejects_include_relevant(
    coordinator: Path,
) -> None:
    row = _decision_for_assignment(_manifest(coordinator)[0])
    row["criterion"] = "include-relevant"

    with pytest.raises(screening_results.ScreeningResultError, match="criterion"):
        screening_results.validate_result_decision(row, context="unbound")


def test_v6_phase_accepts_include_relevant(
    v6_coordinator: Path,
    tmp_path: Path,
) -> None:
    release = _release_phase(v6_coordinator, tmp_path, "calibration")
    paths = _v6_phase_result_paths(
        tmp_path / "raw-v6", v6_coordinator, criterion="include-relevant"
    )
    output = tmp_path / "sealed-v6" / "v1"
    output.parent.mkdir()

    screening_results.seal_phase_results(
        coordinator_snapshot_dir=v6_coordinator,
        phase="calibration",
        result_paths=paths,
        output_dir=output,
        reviewer_release_snapshot_dir=release,
    )


@pytest.mark.parametrize("criterion", screening_results.INCLUSION_CRITERIA)
def test_v6_phase_rejects_legacy_inclusion_values(
    v6_coordinator: Path,
    tmp_path: Path,
    criterion: str,
) -> None:
    release = _release_phase(v6_coordinator, tmp_path, "calibration")
    paths = _v6_phase_result_paths(
        tmp_path / "raw-v6", v6_coordinator, criterion=criterion
    )
    output = tmp_path / "sealed-v6" / "v1"
    output.parent.mkdir()

    with pytest.raises(screening_results.ScreeningResultError, match="criterion"):
        screening_results.seal_phase_results(
            coordinator_snapshot_dir=v6_coordinator,
            phase="calibration",
            result_paths=paths,
            output_dir=output,
            reviewer_release_snapshot_dir=release,
        )


def test_committed_v5_calibration_snapshot_still_validates() -> None:
    captured = screening_results.validate_phase_result_snapshot(
        Path("paper/data/screening_results/calibration/v5"),
        coordinator_snapshot_dir=Path("paper/data/screening_inputs/v5"),
        reviewer_release_snapshot_dir=Path(
            "paper/data/screening_releases/calibration/v5"
        ),
    )

    assert captured.phase == "calibration"


def _coordinator_inclusion_criterion(coordinator: Path) -> str:
    taxonomy = json.loads(
        (coordinator / "taxonomy.json").read_text(encoding="utf-8")
    )
    criteria = taxonomy.get("screening_inclusion_criterion")
    return "include-1" if criteria is None else criteria[0]


def _phase_result_paths(
    root: Path,
    coordinator: Path,
    phase: str,
    *,
    disagreements: int = 0,
) -> list[Path]:
    root.mkdir()
    inclusion_criterion = _coordinator_inclusion_criterion(coordinator)
    phase_manifest = [
        row for row in _manifest(coordinator) if row["phase"] == phase
    ]
    by_candidate: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in phase_manifest:
        by_candidate[row["candidate_id"]].append(row)
    disagreement_ids = set(sorted(by_candidate)[:disagreements])

    rows_by_batch: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate_id in sorted(by_candidate):
        assignments = sorted(
            by_candidate[candidate_id], key=lambda row: row["assignment_id"]
        )
        for index, manifest_row in enumerate(assignments):
            status = (
                "excluded"
                if candidate_id in disagreement_ids and index == 1
                else "included"
            )
            decision = _decision_for_assignment(
                manifest_row, status=status
            )
            if status == "included":
                decision["criterion"] = inclusion_criterion
            rows_by_batch[manifest_row["batch_id"]].append(decision)

    paths = []
    for number in range(1, 7):
        batch_id = f"screening-{number:02d}"
        path = root / f"reviewer-output-{7 - number}.csv"
        rows = sorted(
            rows_by_batch[batch_id],
            key=lambda row: (row["candidate_id"], row["assignment_id"]),
        )
        _write_csv(path, RESULT_HEADER, rows)
        paths.append(path)
    return list(reversed(paths))


def _v6_phase_result_paths(
    root: Path,
    coordinator: Path,
    *,
    criterion: str,
) -> list[Path]:
    paths = _phase_result_paths(root, coordinator, "calibration")
    for path in paths:
        rows = _read_csv(path)
        for row in rows:
            if row["screening_status"] == "included":
                row["criterion"] = criterion
        _write_csv(path, RESULT_HEADER, rows)
    return paths


def _file_payloads(snapshot: Path) -> dict[str, bytes]:
    return {
        path.relative_to(snapshot).as_posix(): path.read_bytes()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }


def _release_phase(
    coordinator: Path,
    tmp_path: Path,
    phase: str,
    *,
    calibration_result_snapshot: Path | None = None,
    calibration_decision_snapshot: Path | None = None,
    version: str = "v1",
) -> Path:
    output_root = tmp_path / f"release-{phase}"
    output_root.mkdir(exist_ok=True)
    output = output_root / version
    screening_results.screening_batches.release_snapshot(
        coordinator,
        phase,
        output,
        calibration_result_snapshot=calibration_result_snapshot,
        calibration_decision_snapshot=calibration_decision_snapshot,
    )
    return output





def _ensure_calibration_release(
    coordinator: Path,
    output: Path,
) -> Path:
    root = coordinator.parents[1]
    version = output.name
    release = root / "release-calibration" / version
    if release.exists():
        return release
    return _release_phase(
        coordinator,
        root,
        "calibration",
        version=version,
    )


def _seal_calibration_phase(
    coordinator: Path,
    paths: list[Path],
    output: Path,
) -> Path:
    release = _ensure_calibration_release(coordinator, output)
    screening_results.seal_phase_results(
        coordinator,
        "calibration",
        paths,
        output,
        reviewer_release_snapshot_dir=release,
    )
    _register_phase_inputs(output, reviewer_release=release)
    return release


def _decision_row(
    coordinator: Path,
    reviewer_release: Path,
    phase_snapshot: Path,
    *,
    ambiguity: str = "false",
    decision: str = "release",
) -> dict[str, str]:
    sealed = screening_results.validate_phase_result_snapshot(
        phase_snapshot,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=reviewer_release,
    )
    by_candidate: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sealed.rows:
        by_candidate[row["candidate_id"]].append(row)
    candidate_ids = [
        row["candidate_id"]
        for row in _read_csv(
            coordinator / "calibration_selection.csv"
        )
    ]
    assert set(candidate_ids) == set(by_candidate)
    numerator = sum(
        len({row["screening_status"] for row in rows}) == 1
        for rows in by_candidate.values()
    )
    denominator = len(by_candidate)
    return {
        "decision_id": "calibration-gate-v1",
        "protocol_sha256": sealed.protocol_sha256,
        "coordinator_snapshot_sha256": (
            sealed.coordinator_snapshot_sha256
        ),
        "calibration_result_snapshot_sha256": sealed.snapshot_sha256,
        "candidate_ids_sha256": screening_results.sequence_ids_sha256(
            candidate_ids
        ),
        "assignment_ids_sha256": screening_results.ordered_ids_sha256(
            row["assignment_id"] for row in sealed.rows
        ),
        "status_agreement_numerator": str(numerator),
        "status_agreement_denominator": str(denominator),
        "status_agreement": screening_results.canonical_ratio(
            numerator, denominator
        ),
        "systematic_ambiguity": ambiguity,
        "decision": decision,
        "decided_on": "2026-06-30",
        "decision_makers": "accountable-author;survey-coordinator",
        "resolution_evidence": (
            "The locked status pairs and disagreement log were reviewed "
            "against every calibration rule boundary."
        ),
    }


_PHASE_VALIDATION_INPUTS: dict[Path, dict[str, Path]] = {}


def _phase_key(snapshot: Path) -> Path:
    return Path(os.path.abspath(os.fspath(snapshot)))


def _register_phase_inputs(
    snapshot: Path,
    *,
    reviewer_release: Path,
    calibration_reviewer_release: Path | None = None,
    calibration_result: Path | None = None,
    calibration_decision: Path | None = None,
) -> None:
    values = {
        "reviewer_release_snapshot_dir": reviewer_release,
    }
    optional = {
        "calibration_reviewer_release_snapshot_dir": (
            calibration_reviewer_release
        ),
        "calibration_result_snapshot_dir": calibration_result,
        "calibration_decision_snapshot_dir": calibration_decision,
    }
    values.update(
        {name: value for name, value in optional.items() if value is not None}
    )
    _PHASE_VALIDATION_INPUTS[_phase_key(snapshot)] = values


def _phase_validation_inputs(snapshot: Path) -> dict[str, Path]:
    return dict(_PHASE_VALIDATION_INPUTS[_phase_key(snapshot)])


def _publish_main_release(
    coordinator: Path,
    calibration_release: Path,
    calibration_result: Path,
    calibration_decision: Path,
    output: Path,
) -> None:
    captured_coordinator = screening_results.capture_coordinator_snapshot(
        coordinator
    )
    captured_calibration = screening_results.validate_phase_result_snapshot(
        calibration_result,
        coordinator=captured_coordinator,
        reviewer_release_snapshot_dir=calibration_release,
    )
    captured_decision = _validate_calibration_decision_snapshot(
        calibration_decision,
        coordinator_snapshot_dir=coordinator,
        calibration_reviewer_release_snapshot_dir=calibration_release,
        calibration_result_snapshot_dir=calibration_result,
    )
    artifacts = (
        screening_results.screening_batches.build_reviewer_release_artifacts(
            captured_coordinator.payloads,
            "main",
            calibration_result_snapshot_sha256=(
                captured_calibration.snapshot_sha256
            ),
            calibration_decision_snapshot_sha256=(
                captured_decision.snapshot_sha256
            ),
        )
    )
    screening_results.screening_batches.publish_snapshot(output, artifacts)


def _passing_main_authorization(
    coordinator: Path,
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    _, calibration_result = _seal_phase(
        coordinator,
        tmp_path,
        "calibration",
        version="v90",
    )
    calibration_release = _phase_validation_inputs(calibration_result)[
        "reviewer_release_snapshot_dir"
    ]
    decision_input = tmp_path / "main-gate-decision-input.csv"
    _write_csv(
        decision_input,
        DECISION_HEADER,
        [
            _decision_row(
                coordinator,
                calibration_release,
                calibration_result,
            )
        ],
    )
    decision_root = tmp_path / "main-gate-decisions"
    decision_root.mkdir(exist_ok=True)
    decision_snapshot = decision_root / "v1"
    _seal_calibration_decision(
        coordinator,
        calibration_result,
        decision_input,
        decision_snapshot,
        calibration_reviewer_release_snapshot_dir=calibration_release,
    )
    release_root = tmp_path / "release-main"
    release_root.mkdir(exist_ok=True)
    main_release = release_root / "v1"
    _publish_main_release(
        coordinator,
        calibration_release,
        calibration_result,
        decision_snapshot,
        main_release,
    )
    return (
        main_release,
        calibration_release,
        calibration_result,
        decision_snapshot,
    )


def _seal_phase(
    coordinator: Path,
    tmp_path: Path,
    phase: str = "calibration",
    *,
    disagreements: int = 0,
    version: str = "v1",
) -> tuple[list[Path], Path]:
    authorization: tuple[Path, Path, Path, Path] | None = None
    if phase == "calibration":
        release = _release_phase(
            coordinator,
            tmp_path,
            "calibration",
            version=version,
        )
    else:
        authorization = _passing_main_authorization(coordinator, tmp_path)
        release = authorization[0]
    paths = _phase_result_paths(
        tmp_path / f"raw-{phase}-{version}",
        coordinator,
        phase,
        disagreements=disagreements,
    )
    output_root = tmp_path / f"sealed-{phase}"
    output_root.mkdir(exist_ok=True)
    output = output_root / version
    kwargs: dict[str, Path] = {
        "reviewer_release_snapshot_dir": release,
    }
    if authorization is not None:
        (
            _,
            calibration_release,
            calibration_result,
            calibration_decision,
        ) = authorization
        kwargs.update(
            calibration_reviewer_release_snapshot_dir=calibration_release,
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=calibration_decision,
        )
    screening_results.seal_phase_results(
        coordinator_snapshot_dir=coordinator,
        phase=phase,
        result_paths=paths,
        output_dir=output,
        **kwargs,
    )
    _register_phase_inputs(
        output,
        reviewer_release=release,
        calibration_reviewer_release=(
            authorization[1] if authorization is not None else None
        ),
        calibration_result=(
            authorization[2] if authorization is not None else None
        ),
        calibration_decision=(
            authorization[3] if authorization is not None else None
        ),
    )
    return paths, output





def _seal_calibration_decision(
    coordinator_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    decision_input_path: Path,
    output_dir: Path,
    *,
    calibration_reviewer_release_snapshot_dir: Path | None = None,
) -> None:
    release = calibration_reviewer_release_snapshot_dir
    if release is None:
        release = _phase_validation_inputs(calibration_result_snapshot_dir)[
            "reviewer_release_snapshot_dir"
        ]
    screening_results.seal_calibration_decision(
        coordinator_snapshot_dir,
        calibration_result_snapshot_dir,
        decision_input_path,
        output_dir,
        calibration_reviewer_release_snapshot_dir=release,
    )


def _validate_calibration_decision_snapshot(
    snapshot_dir: Path,
    *,
    coordinator_snapshot_dir: Path,
    calibration_result_snapshot_dir: Path,
    calibration_reviewer_release_snapshot_dir: Path | None = None,
) -> screening_results.CalibrationDecisionSnapshot:
    release = calibration_reviewer_release_snapshot_dir
    if release is None:
        release = _phase_validation_inputs(calibration_result_snapshot_dir)[
            "reviewer_release_snapshot_dir"
        ]
    return screening_results.validate_calibration_decision_snapshot(
        snapshot_dir,
        coordinator_snapshot_dir=coordinator_snapshot_dir,
        calibration_reviewer_release_snapshot_dir=release,
        calibration_result_snapshot_dir=calibration_result_snapshot_dir,
    )


def test_exact_public_headers() -> None:
    assert screening_results.RESULT_HEADER == RESULT_HEADER
    assert screening_results.PHASE_RESULT_MANIFEST_HEADER == (
        PHASE_MANIFEST_HEADER
    )
    assert screening_results.CALIBRATION_DECISION_HEADER == DECISION_HEADER
    assert (
        screening_results.CALIBRATION_DECISION_MANIFEST_HEADER
        == DECISION_MANIFEST_HEADER
    )


def test_phase_result_api_requires_release_and_binds_its_digest(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    release = _release_phase(coordinator, tmp_path, "calibration")
    paths = _phase_result_paths(
        tmp_path / "raw-calibration", coordinator, "calibration"
    )
    output = tmp_path / "sealed-calibration" / "v1"
    output.parent.mkdir()

    screening_results.seal_phase_results(
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=release,
        phase="calibration",
        result_paths=paths,
        output_dir=output,
    )
    captured = screening_results.validate_phase_result_snapshot(
        output,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=release,
    )

    release_payloads = _file_payloads(release)
    expected_release_sha256 = screening_results._canonical_sha256(
        {
            "files": [
                {
                    "path": path,
                    "sha256": hashlib.sha256(
                        release_payloads[path]
                    ).hexdigest(),
                }
                for path in sorted(
                    release_payloads,
                    key=lambda value: value.encode("utf-8"),
                )
            ],
            "manifest_version": screening_results.MANIFEST_VERSION,
            "phase": "calibration",
        }
    )
    assert captured.reviewer_release_sha256 == expected_release_sha256
    assert len(captured.reviewer_release_sha256) == 64
    assert set(captured.reviewer_release_sha256) <= set("0123456789abcdef")
    assert {
        row["reviewer_release_sha256"] for row in captured.manifest
    } == {captured.reviewer_release_sha256}
    result_payloads = {
        batch_id: (output / f"{batch_id}.csv").read_bytes()
        for batch_id in screening_results.BATCH_IDS
    }
    _, alternate_manifest, alternate_snapshot_sha256, _, _ = (
        screening_results._validate_phase_payloads(
            screening_results.capture_coordinator_snapshot(coordinator),
            "calibration",
            result_payloads,
            reviewer_release_sha256="f" * 64,
        )
    )
    assert alternate_snapshot_sha256 != captured.snapshot_sha256

    import shutil

    forged_result = tmp_path / "forged-result" / "v1"
    forged_result.parent.mkdir()
    shutil.copytree(output, forged_result)
    (forged_result / "manifest.csv").write_bytes(
        screening_results._csv_bytes(
            screening_results.PHASE_RESULT_MANIFEST_HEADER,
            alternate_manifest,
        )
    )
    forged_artifacts = {
        path.name: path.read_bytes()
        for path in forged_result.iterdir()
        if path.name != "SHA256SUMS"
    }
    (forged_result / "SHA256SUMS").write_bytes(
        screening_results._checksums(forged_artifacts)
    )
    with pytest.raises(
        screening_results.ScreeningResultError,
        match="reviewer release|bind",
    ):
        screening_results.validate_phase_result_snapshot(
            forged_result,
            coordinator_snapshot_dir=coordinator,
            reviewer_release_snapshot_dir=release,
        )

    assert (
        "reviewer_release_sha256"
        in screening_results.PHASE_RESULT_MANIFEST_HEADER
    )

    copied_release = tmp_path / "copied-release" / "v1"
    copied_release.parent.mkdir()
    shutil.copytree(release, copied_release)
    copied = screening_results.validate_phase_result_snapshot(
        output,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=copied_release,
    )
    assert copied.reviewer_release_sha256 == captured.reviewer_release_sha256
    assert copied.snapshot_sha256 == captured.snapshot_sha256


def test_main_results_bind_the_passing_gated_main_release(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    calibration_release = _release_phase(
        coordinator, tmp_path, "calibration"
    )
    calibration_paths = _phase_result_paths(
        tmp_path / "raw-calibration-gate",
        coordinator,
        "calibration",
    )
    calibration_result = tmp_path / "calibration-results" / "v1"
    calibration_result.parent.mkdir()
    screening_results.seal_phase_results(
        coordinator,
        "calibration",
        calibration_paths,
        calibration_result,
        reviewer_release_snapshot_dir=calibration_release,
    )
    decision_input = tmp_path / "decision-input.csv"
    _write_csv(
        decision_input,
        DECISION_HEADER,
        [
            _decision_row(
                coordinator,
                calibration_release,
                calibration_result,
            )
        ],
    )
    decision_snapshot = tmp_path / "calibration-decisions" / "v1"
    decision_snapshot.parent.mkdir()
    _seal_calibration_decision(
        coordinator,
        calibration_result,
        decision_input,
        decision_snapshot,
        calibration_reviewer_release_snapshot_dir=calibration_release,
    )
    captured_coordinator = screening_results.capture_coordinator_snapshot(
        coordinator
    )
    captured_calibration = screening_results.validate_phase_result_snapshot(
        calibration_result,
        coordinator=captured_coordinator,
        reviewer_release_snapshot_dir=calibration_release,
    )
    captured_decision = _validate_calibration_decision_snapshot(
        decision_snapshot,
        coordinator_snapshot_dir=coordinator,
        calibration_result_snapshot_dir=calibration_result,
        calibration_reviewer_release_snapshot_dir=calibration_release,
    )
    main_release = tmp_path / "release-main" / "v1"
    main_release.parent.mkdir()
    artifacts = screening_results.screening_batches.build_reviewer_release_artifacts(
        captured_coordinator.payloads,
        "main",
        calibration_result_snapshot_sha256=(
            captured_calibration.snapshot_sha256
        ),
        calibration_decision_snapshot_sha256=(
            captured_decision.snapshot_sha256
        ),
    )
    screening_results.screening_batches.publish_snapshot(
        main_release,
        artifacts,
    )
    ungated_release = tmp_path / "release-main-self-declared" / "v1"
    ungated_release.parent.mkdir()
    ungated_artifacts = (
        screening_results.screening_batches.build_reviewer_release_artifacts(
            captured_coordinator.payloads,
            "main",
            calibration_result_snapshot_sha256="a" * 64,
            calibration_decision_snapshot_sha256="b" * 64,
        )
    )
    screening_results.screening_batches.publish_snapshot(
        ungated_release,
        ungated_artifacts,
    )
    main_paths = _phase_result_paths(
        tmp_path / "raw-main-gated",
        coordinator,
        "main",
    )
    rejected_root = tmp_path / "rejected-main-results"
    rejected_root.mkdir()
    for number, unauthorized_release in enumerate(
        (calibration_release, ungated_release),
        start=1,
    ):
        with pytest.raises(
            screening_results.ScreeningResultError,
            match="release|phase|authorization|binding",
        ):
            screening_results.seal_phase_results(
                coordinator,
                "main",
                main_paths,
                rejected_root / f"v{number}",
                reviewer_release_snapshot_dir=unauthorized_release,
                calibration_reviewer_release_snapshot_dir=(
                    calibration_release
                ),
                calibration_result_snapshot_dir=calibration_result,
                calibration_decision_snapshot_dir=decision_snapshot,
            )

    main_result = tmp_path / "main-results" / "v1"
    main_result.parent.mkdir()
    screening_results.seal_phase_results(
        coordinator,
        "main",
        main_paths,
        main_result,
        reviewer_release_snapshot_dir=main_release,
        calibration_reviewer_release_snapshot_dir=calibration_release,
        calibration_result_snapshot_dir=calibration_result,
        calibration_decision_snapshot_dir=decision_snapshot,
    )

    captured_main = screening_results.validate_phase_result_snapshot(
        main_result,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=main_release,
        calibration_reviewer_release_snapshot_dir=calibration_release,
        calibration_result_snapshot_dir=calibration_result,
        calibration_decision_snapshot_dir=decision_snapshot,
    )
    assert {
        row["reviewer_release_sha256"] for row in captured_main.manifest
    } == {captured_main.reviewer_release_sha256}

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="requires calibration reviewer release",
    ):
        screening_results.validate_phase_result_snapshot(
            main_result,
            coordinator_snapshot_dir=coordinator,
            reviewer_release_snapshot_dir=main_release,
        )

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="release|phase|authorization",
    ):
        screening_results.validate_phase_result_snapshot(
            main_result,
            coordinator_snapshot_dir=coordinator,
            reviewer_release_snapshot_dir=calibration_release,
            calibration_reviewer_release_snapshot_dir=calibration_release,
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=decision_snapshot,
        )





def test_main_phase_output_cannot_overlap_calibration_reviewer_release(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    (
        main_release,
        calibration_release,
        calibration_result,
        calibration_decision,
    ) = _passing_main_authorization(coordinator, tmp_path)
    paths = _phase_result_paths(
        tmp_path / "raw-main-output-overlap",
        coordinator,
        "main",
    )

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="overlap|disjoint",
    ):
        screening_results.seal_phase_results(
            coordinator,
            "main",
            paths,
            calibration_release / "v2",
            reviewer_release_snapshot_dir=main_release,
            calibration_reviewer_release_snapshot_dir=(
                calibration_release
            ),
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=calibration_decision,
        )


def test_main_phase_result_cannot_overlap_calibration_reviewer_release(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    (
        main_release,
        calibration_release,
        calibration_result,
        calibration_decision,
    ) = _passing_main_authorization(coordinator, tmp_path)
    paths = _phase_result_paths(
        tmp_path / "raw-main-result-overlap",
        coordinator,
        "main",
    )
    paths[0] = calibration_release / "packets" / "screening-01.csv"
    output_root = tmp_path / "main-overlap-results"
    output_root.mkdir()

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="overlap|disjoint",
    ):
        screening_results.seal_phase_results(
            coordinator,
            "main",
            paths,
            output_root / "v1",
            reviewer_release_snapshot_dir=main_release,
            calibration_reviewer_release_snapshot_dir=(
                calibration_release
            ),
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=calibration_decision,
        )





def test_main_phase_result_cannot_hardlink_calibration_gate_input(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    (
        main_release,
        calibration_release,
        calibration_result,
        calibration_decision,
    ) = _passing_main_authorization(coordinator, tmp_path)
    paths = _phase_result_paths(
        tmp_path / "raw-main-result-hardlink",
        coordinator,
        "main",
    )
    paths[0].unlink()
    os.link(calibration_result / "screening-01.csv", paths[0])
    output_root = tmp_path / "main-hardlink-results"
    output_root.mkdir()

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="alias|hard.?link|immutable authorization",
    ):
        screening_results.seal_phase_results(
            coordinator,
            "main",
            paths,
            output_root / "v1",
            reviewer_release_snapshot_dir=main_release,
            calibration_reviewer_release_snapshot_dir=(
                calibration_release
            ),
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=calibration_decision,
        )





@pytest.mark.parametrize(
    "mutation_target",
    (
        "coordinator",
        "main-release",
        "calibration-release",
        "calibration-result",
        "calibration-decision",
        "result",
    ),
)
def test_main_phase_post_publish_mutation_removes_owned_snapshot(
    coordinator: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
) -> None:
    (
        main_release,
        calibration_release,
        calibration_result,
        calibration_decision,
    ) = _passing_main_authorization(coordinator, tmp_path)
    paths = _phase_result_paths(
        tmp_path / "raw-main-post-publish",
        coordinator,
        "main",
    )
    output_root = tmp_path / "main-post-publish"
    output_root.mkdir()
    output = output_root / "v1"
    targets = {
        "coordinator": coordinator / "manifest.csv",
        "main-release": main_release / "release_manifest.csv",
        "calibration-release": (
            calibration_release / "release_manifest.csv"
        ),
        "calibration-result": calibration_result / "manifest.csv",
        "calibration-decision": calibration_decision / "manifest.csv",
        "result": paths[0],
    }
    target = targets[mutation_target]
    rename = screening_results.screening_batches._rename_noreplace_at

    def publish_then_mutate(
        parent_fd: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        rename(parent_fd, source_name, destination_name)
        target.write_bytes(target.read_bytes() + b"late mutation")

    monkeypatch.setattr(
        screening_results.screening_batches,
        "_rename_noreplace_at",
        publish_then_mutate,
    )

    with pytest.raises(screening_results.ScreeningResultError):
        screening_results.seal_phase_results(
            coordinator,
            "main",
            paths,
            output,
            reviewer_release_snapshot_dir=main_release,
            calibration_reviewer_release_snapshot_dir=(
                calibration_release
            ),
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=calibration_decision,
        )
    assert not os.path.lexists(output)
    assert not list(output_root.glob(".v1.*.tmp"))


@pytest.mark.parametrize(
    ("phase", "expected_count"),
    [("calibration", 60), ("main", 344)],
)
def test_seal_phase_results_covers_only_requested_phase(
    coordinator: Path,
    tmp_path: Path,
    phase: str,
    expected_count: int,
) -> None:
    _, output = _seal_phase(coordinator, tmp_path, phase)

    captured = screening_results.validate_phase_result_snapshot(
        output,
        coordinator_snapshot_dir=coordinator,
        **_phase_validation_inputs(output),
    )

    assert captured.phase == phase
    assert len(captured.rows) == expected_count
    assert {row["phase"] for row in captured.rows} == {phase}
    assert Counter(row["candidate_id"] for row in captured.rows) == Counter(
        {row["candidate_id"]: 2 for row in captured.rows}
    )
    assert len(captured.fingerprints) == 8
    assert all(fingerprint.reattest() for fingerprint in captured.fingerprints)
    manifest = _read_csv(output / "manifest.csv")
    assert len(manifest) == 6
    assert {row["batch_id"] for row in manifest} == {
        f"screening-{number:02d}" for number in range(1, 7)
    }
    assert all(row["coder_id"] == row["batch_id"] for row in manifest)
    assert sum(int(row["row_count"]) for row in manifest) == expected_count
    assert {row["phase"] for row in manifest} == {phase}
    assert {row["phase_result_snapshot_sha256"] for row in manifest} == {
        captured.snapshot_sha256
    }


def test_phase_sealing_is_order_independent_deterministic_and_no_clobber(
    coordinator: Path, tmp_path: Path
) -> None:
    paths = _phase_result_paths(
        tmp_path / "raw", coordinator, "calibration"
    )
    root = tmp_path / "sealed"
    root.mkdir()
    first = root / "v1"
    second = root / "v2"

    _seal_calibration_phase(coordinator, paths, first)
    _seal_calibration_phase(
        coordinator, list(reversed(paths)), second
    )

    assert _file_payloads(first) == _file_payloads(second)
    before = _file_payloads(first)
    with pytest.raises(screening_results.ScreeningResultError, match="exists"):
        _seal_calibration_phase(coordinator, paths, first)
    assert _file_payloads(first) == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("coder", "coder_id"),
        ("phase", "phase"),
        ("input", "input_sha256"),
        ("missing", "coverage"),
        ("extra", "assignment"),
    ],
)
def test_phase_sealing_rejects_assignment_and_coverage_drift(
    coordinator: Path,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    target = paths[0]
    rows = _read_csv(target)
    if mutation == "coder":
        rows[0]["coder_id"] = "screening-99"
    elif mutation == "phase":
        rows[0]["phase"] = "main"
    elif mutation == "input":
        rows[0]["input_sha256"] = "0" * 64
    elif mutation == "missing":
        rows.pop()
    else:
        rows.append(dict(rows[0]))
        rows[-1]["assignment_id"] = "A-C9999-01"
        rows[-1]["candidate_id"] = "C9999"
    _write_csv(target, RESULT_HEADER, rows)
    output_root = tmp_path / "sealed"
    output_root.mkdir()

    with pytest.raises(screening_results.ScreeningResultError, match=message):
        _seal_calibration_phase(
            coordinator, paths, output_root / "v1"
        )
    assert not (output_root / "v1").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("screened_on", "2026-02-30", "screened_on"),
        ("screening_status", "maybe", "screening_status"),
        ("criterion", "exclude-out-of-scope", "criterion"),
        ("access_status", "abstract_only", "abstract_only"),
        ("source_urls", "HTTPS://EXAMPLE.TEST/source/C0001", "canonical"),
        ("source_urls", "https://example.test/a;https://example.test/a", "duplicate"),
        ("evidence_version", "NR", "evidence_version"),
        ("evidence_retrieved_on", "yesterday", "evidence_retrieved_on"),
        ("evidence_archive_url", "ftp://archive.test/item", "HTTP"),
        ("evidence_sha256", "abc", "evidence_sha256"),
        ("screening_locator", "abstract", "locator"),
        ("screening_locator", "section", "locator"),
        ("exclusion_reason", "not relevant", "exclusion_reason"),
        ("notes", "", "blank"),
    ],
)
def test_phase_sealing_rejects_invalid_decision_and_provenance_fields(
    coordinator: Path,
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    target = paths[0]
    rows = _read_csv(target)
    rows[0][field] = value
    _write_csv(target, RESULT_HEADER, rows)
    output_root = tmp_path / "sealed"
    output_root.mkdir()

    with pytest.raises(screening_results.ScreeningResultError, match=message):
        _seal_calibration_phase(
            coordinator, paths, output_root / "v1"
        )


@pytest.mark.parametrize("alias_kind", ["same-path", "hard-link", "symlink"])
def test_phase_sealing_rejects_result_aliases(
    coordinator: Path,
    tmp_path: Path,
    alias_kind: str,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    if alias_kind == "same-path":
        paths[1] = paths[0]
    elif alias_kind == "hard-link":
        paths[1].unlink()
        os.link(paths[0], paths[1])
    else:
        paths[1].unlink()
        paths[1].symlink_to(paths[0])
    output_root = tmp_path / "sealed"
    output_root.mkdir()

    with pytest.raises(screening_results.ScreeningResultError, match="alias|link"):
        _seal_calibration_phase(
            coordinator, paths, output_root / "v1"
        )
    assert not (output_root / "v1").exists()


def test_phase_snapshot_validation_detects_byte_mutation(
    coordinator: Path, tmp_path: Path
) -> None:
    _, output = _seal_phase(coordinator, tmp_path)
    target = output / "screening-01.csv"
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(screening_results.ScreeningResultError, match="checksum|changed|CSV"):
        screening_results.validate_phase_result_snapshot(
            output,
            coordinator_snapshot_dir=coordinator,
            **_phase_validation_inputs(output),
        )


def test_phase_snapshot_rejects_self_declared_provenance_without_coordinator(
    coordinator: Path, tmp_path: Path
) -> None:
    _, output = _seal_phase(coordinator, tmp_path)

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="exactly one authoritative coordinator",
    ):
        screening_results.validate_phase_result_snapshot(
            output,
            **_phase_validation_inputs(output),
        )


def test_phase_snapshot_fingerprint_reattests_after_anchored_validation(
    coordinator: Path, tmp_path: Path
) -> None:
    _, output = _seal_phase(coordinator, tmp_path)

    captured = screening_results.validate_phase_result_snapshot(
        output,
        coordinator_snapshot_dir=coordinator,
        **_phase_validation_inputs(output),
    )

    assert captured.phase == "calibration"
    assert len(captured.rows) == 60
    assert captured.coordinator_snapshot_sha256 == _manifest(coordinator)[0][
        "snapshot_sha256"
    ]
    assert captured.protocol_sha256 == _manifest(coordinator)[0][
        "protocol_sha256"
    ]
    fingerprint = next(
        item for item in captured.fingerprints if item.path.name == "manifest.csv"
    )
    fingerprint.path.write_bytes(fingerprint.path.read_bytes() + b"tamper")
    with pytest.raises(screening_results.ScreeningResultError, match="changed"):
        fingerprint.reattest()


@pytest.mark.parametrize("trigger", [1, 2])
def test_phase_snapshot_validation_rechecks_root_directory(
    coordinator: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trigger: int,
) -> None:
    _, output = _seal_phase(coordinator, tmp_path)
    reader = screening_results.screening_batches._read_regular_file_at
    injected = False
    matching_reads = 0

    def inject_after_result_read(directory_fd: int, name: str, label: str):
        nonlocal injected, matching_reads
        captured = reader(directory_fd, name, label)
        if str(output) in label and name == "screening-06.csv":
            matching_reads += 1
        if not injected and matching_reads == trigger:
            injected = True
            (output / "injected.txt").write_text(
                "tamper\n", encoding="utf-8"
            )
        return captured

    monkeypatch.setattr(
        screening_results.screening_batches,
        "_read_regular_file_at",
        inject_after_result_read,
    )

    with pytest.raises(screening_results.ScreeningResultError, match="changed|entries"):
        screening_results.validate_phase_result_snapshot(
            output,
            coordinator_snapshot_dir=coordinator,
            **_phase_validation_inputs(output),
        )


def test_phase_sealing_rejects_output_nested_in_coordinator(
    coordinator: Path, tmp_path: Path
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")

    with pytest.raises(screening_results.ScreeningResultError, match="overlap"):
        _seal_calibration_phase(
            coordinator, paths, coordinator / "v2"
        )


def test_ordered_identifier_hash_uses_sorted_lf_terminated_utf8() -> None:
    expected = hashlib.sha256(b"C0001\nC0002\n").hexdigest()

    assert screening_results.ordered_ids_sha256(
        ["C0002", "C0001"]
    ) == expected



def test_sequence_identifier_hash_preserves_lf_terminated_utf8_order() -> None:
    expected = hashlib.sha256(b"C0002\nC0001\n").hexdigest()

    assert screening_results.sequence_ids_sha256(["C0002", "C0001"]) == expected
    assert expected != screening_results.ordered_ids_sha256(["C0002", "C0001"])


def test_calibration_release_decision_binds_derived_gate_values(
    coordinator: Path, tmp_path: Path
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    decision_path = tmp_path / "decision-input.csv"
    decision = _decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )
    assert decision["status_agreement_numerator"] == "25"
    assert decision["status_agreement_denominator"] == "30"
    assert decision["status_agreement"] == "0.833333"
    selection_ids = [
        row["candidate_id"]
        for row in _read_csv(
            coordinator / "calibration_selection.csv"
        )
    ]
    assert selection_ids != sorted(
        selection_ids,
        key=lambda value: value.encode("utf-8"),
    )
    candidate_preimage = "".join(
        f"{candidate_id}\n" for candidate_id in selection_ids
    ).encode("utf-8")
    assert decision["candidate_ids_sha256"] == hashlib.sha256(
        candidate_preimage
    ).hexdigest()
    _write_csv(decision_path, DECISION_HEADER, [decision])
    output_root = tmp_path / "decisions"
    output_root.mkdir()
    output = output_root / "v1"

    _seal_calibration_decision(
        coordinator_snapshot_dir=coordinator,
        calibration_result_snapshot_dir=phase_snapshot,
        decision_input_path=decision_path,
        output_dir=output,
    )
    captured = _validate_calibration_decision_snapshot(
        output,
        coordinator_snapshot_dir=coordinator,
        calibration_result_snapshot_dir=phase_snapshot,
    )

    phase = screening_results.validate_phase_result_snapshot(
        phase_snapshot,
        coordinator_snapshot_dir=coordinator,
        **_phase_validation_inputs(phase_snapshot),
    )
    assignment_ids = sorted(
        (row["assignment_id"] for row in phase.rows),
        key=lambda value: value.encode("utf-8"),
    )
    assignment_preimage = "".join(
        f"{assignment_id}\n" for assignment_id in assignment_ids
    ).encode("utf-8")

    assert {path.name for path in output.iterdir()} == {
        "decision.csv",
        "candidate_ids.txt",
        "assignment_ids.txt",
        "manifest.csv",
        "SHA256SUMS",
    }
    assert captured.decision == decision
    assert len(captured.fingerprints) == 5
    assert all(fingerprint.reattest() for fingerprint in captured.fingerprints)
    assert _read_csv(output / "decision.csv") == [decision]
    assert (output / "candidate_ids.txt").read_bytes() == candidate_preimage
    assert (output / "assignment_ids.txt").read_bytes() == assignment_preimage
    manifest = _read_csv(output / "manifest.csv")
    assert len(manifest) == 1
    assert tuple(manifest[0]) == DECISION_MANIFEST_HEADER
    assert manifest[0]["calibration_result_snapshot_sha256"] == (
        decision["calibration_result_snapshot_sha256"]
    )
    assert manifest[0]["decision_file_sha256"] == hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    assert manifest[0]["candidate_ids_file_sha256"] == hashlib.sha256(
        candidate_preimage
    ).hexdigest()
    assert manifest[0]["assignment_ids_file_sha256"] == hashlib.sha256(
        assignment_preimage
    ).hexdigest()


def _reseal_tampered_decision_snapshot(snapshot: Path) -> None:
    manifest = _read_csv(snapshot / "manifest.csv")[0]
    decision_sha256 = hashlib.sha256(
        (snapshot / "decision.csv").read_bytes()
    ).hexdigest()
    candidate_sha256 = hashlib.sha256(
        (snapshot / "candidate_ids.txt").read_bytes()
    ).hexdigest()
    assignment_sha256 = hashlib.sha256(
        (snapshot / "assignment_ids.txt").read_bytes()
    ).hexdigest()
    manifest["decision_file_sha256"] = decision_sha256
    manifest["candidate_ids_file_sha256"] = candidate_sha256
    manifest["assignment_ids_file_sha256"] = assignment_sha256
    manifest["calibration_decision_snapshot_sha256"] = (
        screening_results._canonical_sha256(
            {
                "assignment_ids_file_sha256": assignment_sha256,
                "calibration_result_snapshot_sha256": manifest[
                    "calibration_result_snapshot_sha256"
                ],
                "candidate_ids_file_sha256": candidate_sha256,
                "coordinator_snapshot_sha256": manifest[
                    "coordinator_snapshot_sha256"
                ],
                "decision_file_sha256": decision_sha256,
                "decision_id": manifest["decision_id"],
                "manifest_version": manifest["manifest_version"],
                "protocol_sha256": manifest["protocol_sha256"],
                "row_count": 1,
            }
        )
    )
    _write_csv(
        snapshot / "manifest.csv",
        DECISION_MANIFEST_HEADER,
        [manifest],
    )
    artifacts = {
        name: (snapshot / name).read_bytes()
        for name in (
            "assignment_ids.txt",
            "candidate_ids.txt",
            "decision.csv",
            "manifest.csv",
        )
    }
    (snapshot / "SHA256SUMS").write_bytes(
        screening_results._checksums(artifacts)
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "reordered", "extra", "tampered"],
)
def test_calibration_decision_snapshot_rejects_preimage_file_drift(
    coordinator: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    decision_input = tmp_path / "decision-input.csv"
    _write_csv(
        decision_input,
        DECISION_HEADER,
        [_decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )],
    )
    output_root = tmp_path / "decisions"
    output_root.mkdir()
    output = output_root / "v1"
    _seal_calibration_decision(
        coordinator,
        phase_snapshot,
        decision_input,
        output,
    )

    if mutation == "missing":
        (output / "candidate_ids.txt").unlink()
    elif mutation == "reordered":
        path = output / "candidate_ids.txt"
        lines = path.read_bytes().splitlines(keepends=True)
        lines[0], lines[1] = lines[1], lines[0]
        path.write_bytes(b"".join(lines))
        _reseal_tampered_decision_snapshot(output)
    elif mutation == "extra":
        (output / "unexpected.txt").write_text(
            "unexpected\n", encoding="utf-8"
        )
    else:
        path = output / "assignment_ids.txt"
        path.write_bytes(path.read_bytes() + b"A-C9999-01\n")
        _reseal_tampered_decision_snapshot(output)

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="assignment|candidate|entries|missing",
    ):
        _validate_calibration_decision_snapshot(
            output,
            coordinator_snapshot_dir=coordinator,
            calibration_result_snapshot_dir=phase_snapshot,
        )


@pytest.mark.parametrize(
    ("disagreements", "ambiguity", "decision", "expected"),
    [
        (7, "false", "release", "decision"),
        (5, "true", "release", "decision"),
        (5, "false", "revise", "decision"),
        (7, "false", "revise", None),
        (5, "true", "revise", None),
        (6, "false", "release", None),
    ],
)
def test_calibration_gate_enforces_threshold_and_ambiguity(
    coordinator: Path,
    tmp_path: Path,
    disagreements: int,
    ambiguity: str,
    decision: str,
    expected: str | None,
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=disagreements
    )
    decision_path = tmp_path / "decision-input.csv"
    row = _decision_row(
        coordinator,
        _phase_validation_inputs(phase_snapshot)[
            "reviewer_release_snapshot_dir"
        ],
        phase_snapshot,
        ambiguity=ambiguity,
        decision=decision,
    )
    _write_csv(decision_path, DECISION_HEADER, [row])
    output_root = tmp_path / "decisions"
    output_root.mkdir()

    if expected:
        with pytest.raises(screening_results.ScreeningResultError, match=expected):
            _seal_calibration_decision(
                coordinator,
                phase_snapshot,
                decision_path,
                output_root / "v1",
            )
    else:
        _seal_calibration_decision(
            coordinator,
            phase_snapshot,
            decision_path,
            output_root / "v1",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_ids_sha256", "0" * 64, "candidate_ids_sha256"),
        ("assignment_ids_sha256", "0" * 64, "assignment_ids_sha256"),
        ("status_agreement_numerator", "29", "numerator"),
        ("status_agreement_denominator", "29", "denominator"),
        ("status_agreement", "0.83333", "agreement"),
        ("systematic_ambiguity", "none", "systematic_ambiguity"),
        ("decided_on", "2026-06-31", "decided_on"),
        ("decision_makers", "NR", "decision_makers"),
        ("resolution_evidence", "agreed", "resolution_evidence"),
    ],
)
def test_calibration_decision_rejects_tampered_or_weak_fields(
    coordinator: Path,
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    row = _decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )
    row[field] = value
    decision_path = tmp_path / "decision-input.csv"
    _write_csv(decision_path, DECISION_HEADER, [row])
    output_root = tmp_path / "decisions"
    output_root.mkdir()

    with pytest.raises(screening_results.ScreeningResultError, match=message):
        _seal_calibration_decision(
            coordinator,
            phase_snapshot,
            decision_path,
            output_root / "v1",
        )


def test_calibration_decision_requires_accountable_author_role(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    row = _decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )
    row["decision_makers"] = "survey-coordinator"
    decision_path = tmp_path / "decision-input.csv"
    _write_csv(decision_path, DECISION_HEADER, [row])
    output_root = tmp_path / "decisions"
    output_root.mkdir()

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="accountable-author",
    ):
        _seal_calibration_decision(
            coordinator,
            phase_snapshot,
            decision_path,
            output_root / "v1",
        )


def test_calibration_decision_rejects_main_result_snapshot(
    coordinator: Path, tmp_path: Path
) -> None:
    _, main_snapshot = _seal_phase(coordinator, tmp_path, phase="main")
    decision_path = tmp_path / "decision-input.csv"
    decision_path.write_text(",".join(DECISION_HEADER) + "\n", encoding="utf-8")
    output_root = tmp_path / "decisions"
    output_root.mkdir()

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="calibration|main|release",
    ):
        _seal_calibration_decision(
            coordinator,
            main_snapshot,
            decision_path,
            output_root / "v1",
        )


def test_calibration_decision_replay_is_byte_identical_and_no_clobber(
    coordinator: Path, tmp_path: Path
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    decision_path = tmp_path / "decision-input.csv"
    _write_csv(
        decision_path,
        DECISION_HEADER,
        [_decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )],
    )
    root = tmp_path / "decisions"
    root.mkdir()
    first = root / "v1"
    second = root / "v2"

    _seal_calibration_decision(
        coordinator, phase_snapshot, decision_path, first
    )
    _seal_calibration_decision(
        coordinator, phase_snapshot, decision_path, second
    )

    assert _file_payloads(first) == _file_payloads(second)
    before = _file_payloads(first)
    with pytest.raises(screening_results.ScreeningResultError, match="exists"):
        _seal_calibration_decision(
            coordinator, phase_snapshot, decision_path, first
        )
    assert _file_payloads(first) == before


def test_cli_seals_phase_and_calibration_decision(
    coordinator: Path, tmp_path: Path
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    release = _release_phase(coordinator, tmp_path, "calibration")
    phase_root = tmp_path / "phase"
    phase_root.mkdir()
    phase_snapshot = phase_root / "v1"
    phase_command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--seal-phase",
        "--coordinator-snapshot",
        str(coordinator),
        "--reviewer-release-snapshot",
        str(release),
        "--phase",
        "calibration",
        "--output-dir",
        str(phase_snapshot),
    ]
    for path in paths:
        phase_command.extend(("--result", str(path)))
    completed = subprocess.run(
        phase_command,
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    _register_phase_inputs(
        phase_snapshot,
        reviewer_release=release,
    )

    decision_path = tmp_path / "decision.csv"
    _write_csv(
        decision_path,
        DECISION_HEADER,
        [_decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )],
    )
    decision_root = tmp_path / "decision-snapshots"
    decision_root.mkdir()
    decision_snapshot = decision_root / "v1"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seal-calibration-decision",
            "--coordinator-snapshot",
            str(coordinator),
            "--reviewer-release-snapshot",
            str(release),
            "--calibration-result-snapshot",
            str(phase_snapshot),
            "--decision-input",
            str(decision_path),
            "--output-dir",
            str(decision_snapshot),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    _validate_calibration_decision_snapshot(
        decision_snapshot,
        coordinator_snapshot_dir=coordinator,
        calibration_result_snapshot_dir=phase_snapshot,
    )


@pytest.mark.parametrize(
    "archive_url",
    [
        "https://doi.org/10.1016/j.scico.2024.103171",
        "https://arxiv.org/pdf/2109.12674v3",
        "https://proceedings.mlr.press/v123/madaan20a/madaan20a.pdf",
        "https://openaccess.thecvf.com/content/CVPR2021/papers/Mi_HDMapGen.pdf",
        "https://docs.un.org/en/E/ECE/TRANS/505/Rev.3/Add.156",
        (
            "https://documents.un.org/api/symbol/access?"
            "s=E/ECE/TRANS/505/Rev.3/Add.156&l=en&t=pdf"
        ),
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42021X0389",
        "https://mediatum.ub.tum.de/1379638",
        "https://digitalcollection.zhaw.ch/bitstreams/c06a4f2a-a833-4e82-8880-c4159addb4d4/download",
        "https://www.indyautonomouschallenge.com/s/2022-ACTMS-Rules-v100.pdf",
        "https://cgl.ethz.ch/Downloads/Publications/Papers/2001/p_Par01.pdf",
        "http://cogprints.org/5573/1/Togelius2007Towards.pdf",
        (
            "https://raw.githubusercontent.com/mlresearch/v270/main/"
            "assets/liang25a/liang25a.pdf"
        ),
        "https://robonation.org/app/uploads/sites/3/2025/10/handbook.pdf",
    ],
)
def test_persistent_scholarly_and_standards_identifiers_are_valid(
    coordinator: Path,
    archive_url: str,
) -> None:
    row = _decision_for_assignment(_manifest(coordinator)[0])
    row["evidence_archive_url"] = archive_url
    screening_results.validate_result_decision(row, context="persistent-source")


@pytest.mark.parametrize(
    "locator",
    [
        (
            "PDF pp. 4-5, Section V.A-C (Track Representation; "
            "Track Encoding; Genotype to Phenotype Mapping), Algorithm 1"
        ),
        "Class RandomRouteAction > Description",
        "Statement tab; Research tab, Research Topics",
        "Statement; Adversarial Multi-Agent Systems",
        "OpenAlex work W4385326949, abstract_inverted_index positions 0-75",
        "paragraphs 5.2.1, 6.2.3(g), and 7.1",
        "Crossref record message.abstract field",
        "OpenAlex abstract sentences 2-5 describing geometric constraints",
        "About Us, competition-history and team sections",
        "About OpenStreetMap; The Map; Mapping; Using OpenStreetMap data",
        "OpenDRIVE standalone mode; Run a standalone map; client.generate_opendrive_world()",
        "SDF worlds; Defining a world; Adding models",
        "What's New in SDFormat 1.7; Pose and Frame Semantics; Frame semantics in nested models",
    ],
)
def test_compound_formal_and_stable_heading_locators_are_valid(
    coordinator: Path,
    locator: str,
) -> None:
    row = _decision_for_assignment(_manifest(coordinator)[0])
    row["screening_locator"] = locator
    screening_results.validate_result_decision(row, context="compound-locator")


@pytest.mark.parametrize(
    ("provenance_case", "should_fail"),
    [
        ("sha-only", False),
        ("archive-only-official", False),
        ("sha-pinned-retrieval-url", False),
        ("missing-official", True),
        ("mutable-version", True),
        ("mutable-version-with-sha", True),
        ("abstract-insufficient", False),
        ("abstract-direct-exclusion", False),
        ("abstract-attempt-log", False),
        ("abstract-no-notes", True),
    ],
)
def test_result_provenance_and_abstract_exclusions_follow_protocol(
    coordinator: Path,
    tmp_path: Path,
    provenance_case: str,
    should_fail: bool,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    target = paths[0]
    rows = _read_csv(target)
    row = rows[0]
    if provenance_case == "sha-only":
        row["evidence_archive_url"] = "NR"
    elif provenance_case == "archive-only-official":
        row["access_status"] = "official_documentation"
        row["evidence_sha256"] = "NR"
    elif provenance_case == "sha-pinned-retrieval-url":
        row["evidence_archive_url"] = (
            "https://archive.example.test/unversioned/document.pdf"
        )
    elif provenance_case == "missing-official":
        row["access_status"] = "official_documentation"
        row["evidence_archive_url"] = "NR"
        row["evidence_sha256"] = "NR"
    elif provenance_case in {"mutable-version", "mutable-version-with-sha"}:
        row["access_status"] = "official_documentation"
        row["evidence_archive_url"] = (
            "https://archive.example.test/item?version=latest"
        )
        if provenance_case == "mutable-version":
            row["evidence_sha256"] = "NR"
    else:
        row["access_status"] = "abstract_only"
        row["screening_status"] = "excluded"
        row["evidence_archive_url"] = "NR"
        row["evidence_sha256"] = "NR"
        row["screening_locator"] = "Section Abstract, sentence 1"
        row["notes"] = (
            "DOI, title, repository, and official-project retrieval attempts "
            "were exhausted."
        )
        if provenance_case == "abstract-attempt-log":
            row["notes"] = (
                "DOI, publisher, title search, indexes, repositories, and "
                "author pages were attempted."
            )
        if provenance_case == "abstract-no-notes":
            row["notes"] = "NR"
        if provenance_case in {"abstract-insufficient", "abstract-no-notes"}:
            row["criterion"] = "exclude-insufficient-detail"
            row["exclusion_reason"] = (
                "Only the abstract was accessible and it does not provide "
                "enough technical detail to support a survey claim."
            )
        else:
            row["criterion"] = "exclude-out-of-scope"
            row["exclusion_reason"] = (
                "The abstract directly and unambiguously states that the work "
                "addresses image classification rather than courses."
            )
    _write_csv(target, RESULT_HEADER, rows)
    output_root = tmp_path / "sealed"
    output_root.mkdir()
    output = output_root / "v1"

    if should_fail:
        with pytest.raises(
            screening_results.ScreeningResultError,
            match="provenance|archive|SHA-256|abstract_only|notes",
        ):
            _seal_calibration_phase(coordinator, paths, output)
        assert not output.exists()
    else:
        _seal_calibration_phase(coordinator, paths, output)
        screening_results.validate_phase_result_snapshot(
            output,
            coordinator_snapshot_dir=coordinator,
            **_phase_validation_inputs(output),
        )


@pytest.mark.parametrize(
    "mutation_target",
    ["coordinator", "reviewer-release", "result"],
)
def test_phase_post_publish_mutation_removes_owned_snapshot(
    coordinator: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    output_root = tmp_path / "sealed"
    output_root.mkdir()
    output = output_root / "v1"
    release = _ensure_calibration_release(coordinator, output)
    targets = {
        "coordinator": coordinator / "manifest.csv",
        "reviewer-release": release / "release_manifest.csv",
        "result": paths[0],
    }
    target = targets[mutation_target]
    rename = screening_results.screening_batches._rename_noreplace_at

    def publish_then_mutate(
        parent_fd: int, source_name: str, destination_name: str
    ) -> None:
        rename(parent_fd, source_name, destination_name)
        target.write_bytes(target.read_bytes() + b"late mutation")

    monkeypatch.setattr(
        screening_results.screening_batches,
        "_rename_noreplace_at",
        publish_then_mutate,
    )

    with pytest.raises(screening_results.ScreeningResultError):
        _seal_calibration_phase(coordinator, paths, output)
    assert not os.path.lexists(output)
    assert not list(output_root.glob(".v1.*.tmp"))


@pytest.mark.parametrize(
    "mutation_target",
    ["coordinator", "reviewer-release", "phase-snapshot", "decision-input"],
)
def test_decision_post_publish_mutation_removes_owned_snapshot(
    coordinator: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    decision_path = tmp_path / "decision-input.csv"
    _write_csv(
        decision_path,
        DECISION_HEADER,
        [_decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )],
    )
    output_root = tmp_path / "decisions"
    output_root.mkdir()
    output = output_root / "v1"
    targets = {
        "coordinator": coordinator / "manifest.csv",
        "reviewer-release": (
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ]
            / "release_manifest.csv"
        ),
        "phase-snapshot": phase_snapshot / "manifest.csv",
        "decision-input": decision_path,
    }
    target = targets[mutation_target]
    rename = screening_results.screening_batches._rename_noreplace_at

    def publish_then_mutate(
        parent_fd: int, source_name: str, destination_name: str
    ) -> None:
        rename(parent_fd, source_name, destination_name)
        target.write_bytes(target.read_bytes() + b"late mutation")

    monkeypatch.setattr(
        screening_results.screening_batches,
        "_rename_noreplace_at",
        publish_then_mutate,
    )

    with pytest.raises(screening_results.ScreeningResultError):
        _seal_calibration_decision(
            coordinator, phase_snapshot, decision_path, output
        )
    assert not os.path.lexists(output)
    assert not list(output_root.glob(".v1.*.tmp"))


def test_authoritative_validation_rejects_self_consistent_forged_gate(
    coordinator: Path, tmp_path: Path
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    decision_path = tmp_path / "decision-input.csv"
    _write_csv(
        decision_path,
        DECISION_HEADER,
        [_decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )],
    )
    output_root = tmp_path / "decisions"
    output_root.mkdir()
    output = output_root / "v1"
    _seal_calibration_decision(
        coordinator, phase_snapshot, decision_path, output
    )

    forged = _read_csv(output / "decision.csv")[0]
    forged["status_agreement_numerator"] = "30"
    forged["status_agreement"] = "1.000000"
    _write_csv(output / "decision.csv", DECISION_HEADER, [forged])
    decision_payload = (output / "decision.csv").read_bytes()
    coordinator_state = screening_results._capture_coordinator(coordinator)
    calibration_state = screening_results.validate_phase_result_snapshot(
        phase_snapshot,
        coordinator_snapshot_dir=coordinator,
        **_phase_validation_inputs(phase_snapshot),
    )
    manifest, _ = screening_results._decision_manifest(
        coordinator_state,
        calibration_state,
        decision_payload,
        forged,
    )
    _write_csv(
        output / "manifest.csv",
        screening_results.CALIBRATION_DECISION_MANIFEST_HEADER,
        [manifest],
    )
    checksum_inputs = {
        name: (output / name).read_bytes()
        for name in (
            "assignment_ids.txt",
            "candidate_ids.txt",
            "decision.csv",
            "manifest.csv",
        )
    }
    (output / "SHA256SUMS").write_bytes(
        screening_results._checksums(checksum_inputs)
    )

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="numerator",
    ):
        _validate_calibration_decision_snapshot(
            output,
            coordinator_snapshot_dir=coordinator,
            calibration_result_snapshot_dir=phase_snapshot,
        )


@pytest.mark.parametrize(
    ("fragment_case", "should_fail"),
    [
        ("represented-stable", False),
        ("github-lines", False),
        ("github-subranges", False),
        ("unrepresented", True),
        ("mutable-anchor", True),
    ],
)
def test_url_fragments_require_precise_locator_semantics(
    coordinator: Path,
    tmp_path: Path,
    fragment_case: str,
    should_fail: bool,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    target = paths[0]
    rows = _read_csv(target)
    row = rows[0]
    if fragment_case == "mutable-anchor":
        fragment = "top"
    elif fragment_case == "github-lines":
        fragment = "L1-L49"
    elif fragment_case == "github-subranges":
        fragment = "L3-L101"
    else:
        fragment = "methods"
    host = "github.com" if fragment_case == "github-subranges" else "example.test"
    row["source_urls"] = (
        f"https://{host}/source/{row['candidate_id']}#{fragment}"
    )
    if fragment_case == "github-lines":
        row["screening_locator"] = "repository/path.xml lines 1-49"
    elif fragment_case == "github-subranges":
        row["screening_locator"] = "README.md lines 3-28, 74-101"
    elif fragment_case != "unrepresented":
        row["screening_locator"] = f"Section 2; Anchor #{fragment}"
    _write_csv(target, RESULT_HEADER, rows)
    output_root = tmp_path / "sealed"
    output_root.mkdir()
    output = output_root / "v1"

    if should_fail:
        with pytest.raises(
            screening_results.ScreeningResultError,
            match="fragment|anchor|locator",
        ):
            _seal_calibration_phase(coordinator, paths, output)
    else:
        _seal_calibration_phase(coordinator, paths, output)


def test_flat_snapshot_rejects_extra_directory_and_fingerprint_binds_tree(
    coordinator: Path, tmp_path: Path
) -> None:
    _, output = _seal_phase(coordinator, tmp_path)
    extra = output / "packets"
    extra.mkdir()
    extra.chmod(0o755)

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="entries|extra",
    ):
        screening_results.validate_phase_result_snapshot(
            output,
            coordinator_snapshot_dir=coordinator,
            **_phase_validation_inputs(output),
        )

    extra.rmdir()
    captured = screening_results.validate_phase_result_snapshot(
        output,
        coordinator_snapshot_dir=coordinator,
        **_phase_validation_inputs(output),
    )
    extra.mkdir()
    extra.chmod(0o755)
    (extra / "later.txt").write_text("late content\n", encoding="utf-8")

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="tree|entries|changed",
    ):
        captured.fingerprints[0].reattest()


@pytest.mark.parametrize(
    "mutation_target", ["published-output", "raw-input", "coordinator", "all"]
)
def test_phase_nested_validation_mutation_fails_coherent_attestation(
    coordinator: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
) -> None:
    paths = _phase_result_paths(tmp_path / "raw", coordinator, "calibration")
    output_root = tmp_path / "sealed"
    output_root.mkdir()
    output = output_root / "v1"
    validator = screening_results.validate_phase_result_snapshot
    injected = False

    def validate_then_mutate(*args, **kwargs):
        nonlocal injected
        captured = validator(*args, **kwargs)
        if not injected and Path(args[0]) == output:
            injected = True
            targets = {
                "published-output": output / "manifest.csv",
                "raw-input": paths[0],
                "coordinator": coordinator / "manifest.csv",
            }
            selected = (
                tuple(targets.values())
                if mutation_target == "all"
                else (targets[mutation_target],)
            )
            for target in selected:
                target.write_bytes(target.read_bytes() + b"nested mutation")
        return captured

    monkeypatch.setattr(
        screening_results,
        "validate_phase_result_snapshot",
        validate_then_mutate,
    )

    with pytest.raises(screening_results.ScreeningResultError):
        _seal_calibration_phase(coordinator, paths, output)
    assert injected
    assert not os.path.lexists(output)


@pytest.mark.parametrize(
    "mutation_target",
    [
        "published-output",
        "decision-input",
        "phase-snapshot",
        "coordinator",
        "all",
    ],
)
def test_decision_nested_validation_mutation_fails_coherent_attestation(
    coordinator: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
) -> None:
    _, phase_snapshot = _seal_phase(
        coordinator, tmp_path, disagreements=5
    )
    decision_path = tmp_path / "decision-input.csv"
    _write_csv(
        decision_path,
        DECISION_HEADER,
        [_decision_row(
            coordinator,
            _phase_validation_inputs(phase_snapshot)[
                "reviewer_release_snapshot_dir"
            ],
            phase_snapshot,
        )],
    )
    output_root = tmp_path / "decisions"
    output_root.mkdir()
    output = output_root / "v1"
    validator = screening_results.validate_calibration_decision_snapshot
    injected = False

    def validate_then_mutate(*args, **kwargs):
        nonlocal injected
        captured = validator(*args, **kwargs)
        if not injected and Path(args[0]) == output:
            injected = True
            targets = {
                "published-output": output / "decision.csv",
                "decision-input": decision_path,
                "phase-snapshot": phase_snapshot / "manifest.csv",
                "coordinator": coordinator / "manifest.csv",
            }
            selected = (
                tuple(targets.values())
                if mutation_target == "all"
                else (targets[mutation_target],)
            )
            for target in selected:
                target.write_bytes(target.read_bytes() + b"nested mutation")
        return captured

    monkeypatch.setattr(
        screening_results,
        "validate_calibration_decision_snapshot",
        validate_then_mutate,
    )

    with pytest.raises(screening_results.ScreeningResultError):
        _seal_calibration_decision(
            coordinator, phase_snapshot, decision_path, output
        )
    assert injected
    assert not os.path.lexists(output)


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest-input-sha256",
        "calibration-candidate-ids",
        "snapshot-sha256",
        "protocol-sha256",
        "payload-mapping",
        "allowed-inclusion-criteria",
    ),
)
def test_public_coordinator_reattest_recaptures_complete_state(
    coordinator: Path,
    mutation: str,
) -> None:
    captured = screening_results.capture_coordinator_snapshot(coordinator)
    if mutation == "manifest-input-sha256":
        captured.manifest[0]["input_sha256"] = "f" * 64
        forged = captured
    elif mutation == "calibration-candidate-ids":
        forged = replace(
            captured,
            calibration_candidate_ids=(
                "C-forged-calibration",
                *captured.calibration_candidate_ids[1:],
            ),
        )
    elif mutation == "snapshot-sha256":
        forged = replace(captured, snapshot_sha256="e" * 64)
    elif mutation == "protocol-sha256":
        forged = replace(captured, protocol_sha256="d" * 64)
    elif mutation == "allowed-inclusion-criteria":
        forged = replace(
            captured, allowed_inclusion_criteria=("include-forged",)
        )
    else:
        captured.payloads["manifest.csv"] += b"forged"
        forged = captured

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="coordinator|authoritative|changed",
    ):
        screening_results.reattest_coordinator_snapshot(forged)


@pytest.mark.parametrize(
    ("public_name", "private_name", "args"),
    (
        (
            "capture_coordinator_snapshot",
            "_capture_coordinator",
            (Path("coordinator"),),
        ),
        (
            "reattest_coordinator_snapshot",
            "_reattest_coordinator",
            (object(),),
        ),
        (
            "reattest_snapshot_set",
            "_coherent_final_attestation",
            (object(), ()),
        ),
        ("capture_input", "_capture_input", (Path("input.csv"), "input")),
        (
            "capture_flat_snapshot",
            "_capture_flat_snapshot",
            (Path("v1"), ("manifest.csv",)),
        ),
    ),
)
def test_public_capture_and_reattest_boundaries_chain_oserror(
    monkeypatch: pytest.MonkeyPatch,
    public_name: str,
    private_name: str,
    args: tuple[object, ...],
) -> None:
    error = PermissionError(errno.EACCES, "portable denied access")

    def denied(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(screening_results, private_name, denied)
    with pytest.raises(screening_results.ScreeningResultError) as caught:
        getattr(screening_results, public_name)(*args)
    assert caught.value.__cause__ is error


def test_public_coordinator_reattest_normalizes_renamed_directory(
    coordinator: Path,
) -> None:
    captured = screening_results.capture_coordinator_snapshot(coordinator)
    renamed = coordinator.with_name("v1-renamed")
    coordinator.rename(renamed)

    with pytest.raises(screening_results.ScreeningResultError) as caught:
        screening_results.reattest_coordinator_snapshot(captured)
    causes: list[BaseException] = []
    cause: BaseException | None = caught.value
    while cause is not None and all(cause is not item for item in causes):
        causes.append(cause)
        cause = cause.__cause__
    assert any(isinstance(item, OSError) for item in causes)


def test_public_coordinator_reattest_rejects_symlink_swap(
    coordinator: Path,
) -> None:
    captured = screening_results.capture_coordinator_snapshot(coordinator)
    moved = coordinator.with_name("v1-moved")
    coordinator.rename(moved)
    try:
        coordinator.symlink_to(moved, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(screening_results.ScreeningResultError):
        screening_results.reattest_coordinator_snapshot(captured)


def test_cli_modes_are_mutually_exclusive() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seal-phase",
            "--seal-calibration-decision",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0


def _stage_role_result(
    coordinator: Path,
    tmp_path: Path,
) -> tuple[Path, Path]:
    release = _release_phase(coordinator, tmp_path, "calibration")
    (tmp_path / "staging").mkdir()
    stage = screening_results.screening_batches.stage_reviewer_execution(
        coordinator, release, "screening-01", tmp_path / "staging"
    )
    result = stage.parent / "screening-01-result.csv"
    rows = [
        _decision_for_assignment(row)
        for row in _read_csv(stage / "packet.csv")
    ]
    configuration = json.loads(
        (stage / "execution_configuration.json").read_text(encoding="utf-8")
    )
    if configuration["configuration_version"] == "2":
        for row in rows:
            if row["screening_status"] == "included":
                row["criterion"] = configuration[
                    "allowed_inclusion_criteria"
                ][0]
    _write_csv(result, RESULT_HEADER, rows)
    return stage, result


def _validate_role_result(stage: Path, result: str | Path) -> int:
    return screening_results.main(
        (
            "--validate-role-result",
            "--reviewer-stage",
            str(stage),
            "--result",
            str(result),
        )
    )


def test_v2_role_result_rejects_legacy_inclusion_value(
    v6_coordinator: Path,
    tmp_path: Path,
) -> None:
    stage, result = _stage_role_result(v6_coordinator, tmp_path)
    configuration = json.loads(
        (stage / "execution_configuration.json").read_text(encoding="utf-8")
    )
    assert configuration["configuration_version"] == "2"
    assert configuration["allowed_inclusion_criteria"] == ["include-relevant"]

    rows = _read_csv(result)
    for row in rows:
        row["screening_status"] = "excluded"
        row["criterion"] = "exclude-out-of-scope"
        row["exclusion_reason"] = (
            "The inspected source does not contribute course-generation "
            "geometry, representations, metrics, or interchange."
        )
    rows[0]["screening_status"] = "included"
    rows[0]["criterion"] = "include-relevant"
    rows[0]["exclusion_reason"] = "NR"
    rows[0]["criterion"] = "include-1"
    _write_csv(result, RESULT_HEADER, rows)

    with pytest.raises(screening_results.ScreeningResultError, match="criterion"):
        _validate_role_result(stage, result)


def test_cli_validates_canonical_role_result(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    stage, result = _stage_role_result(coordinator, tmp_path)
    assert _validate_role_result(stage, result) == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("locator", "locator"),
        ("path", "result path"),
        ("assignment", "candidate_id"),
        ("order", "order"),
    ),
)
def test_cli_role_result_rejects_invalid_data(
    coordinator: Path,
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    stage, result = _stage_role_result(coordinator, tmp_path)
    target = tmp_path / "wrong-result.csv" if mutation == "path" else result
    if mutation == "path":
        target.write_bytes(result.read_bytes())
    else:
        rows = _read_csv(result)
        if mutation == "locator":
            rows[0]["screening_locator"] = "source discussion"
        elif mutation == "assignment":
            rows[0]["candidate_id"] = "C9999"
        else:
            rows.reverse()
        _write_csv(result, RESULT_HEADER, rows)
    with pytest.raises(screening_results.ScreeningResultError, match=match):
        _validate_role_result(stage, target)


def test_cli_role_result_rejects_nonexact_result_path_spelling(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    stage, result = _stage_role_result(coordinator, tmp_path)
    nonexact = f"{result.parent}/./{result.name}"

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="result path",
    ):
        _validate_role_result(stage, nonexact)


def test_cli_role_result_rejects_noncanonical_bytes(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    stage, result = _stage_role_result(coordinator, tmp_path)
    result.write_bytes(result.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="canonical CSV bytes",
    ):
        _validate_role_result(stage, result)


def test_cli_role_result_mode_is_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        screening_results.main(
            ("--seal-phase", "--validate-role-result")
        )


@pytest.mark.parametrize(
    "arguments",
    (
        ("--validate-role-result", "--result", "result.csv"),
        ("--validate-role-result", "--reviewer-stage", "stage"),
    ),
)
def test_cli_role_result_requires_stage_and_result(
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(screening_results.ScreeningResultError, match="missing"):
        screening_results.main(arguments)


def test_cli_role_result_requires_exactly_one_result() -> None:
    with pytest.raises(screening_results.ScreeningResultError, match="exactly one"):
        screening_results.main(
            (
                "--validate-role-result",
                "--reviewer-stage",
                "stage",
                "--result",
                "one.csv",
                "--result",
                "two.csv",
            )
        )


@pytest.mark.parametrize(
    "legacy_arguments",
    (
        ("--coordinator-snapshot", "coordinator"),
        ("--reviewer-release-snapshot", "release"),
        ("--phase", "calibration"),
        ("--calibration-reviewer-release-snapshot", "release"),
        ("--calibration-result-snapshot", "results"),
        ("--calibration-decision-snapshot", "decision"),
        ("--decision-input", "decision.csv"),
        ("--output-dir", "output"),
    ),
)
def test_cli_role_result_rejects_legacy_arguments(
    legacy_arguments: tuple[str, str],
) -> None:
    with pytest.raises(
        screening_results.ScreeningResultError,
        match="only accepts",
    ):
        screening_results.main(
            (
                "--validate-role-result",
                "--reviewer-stage",
                "stage",
                "--result",
                "result.csv",
                *legacy_arguments,
            )
        )


@pytest.mark.parametrize(
    "legacy_arguments",
    (
        (
            "--seal-phase",
            "--coordinator-snapshot",
            "coordinator",
            "--reviewer-release-snapshot",
            "release",
            "--phase",
            "calibration",
            "--result",
            "result.csv",
            "--output-dir",
            "output",
        ),
        (
            "--seal-calibration-decision",
            "--coordinator-snapshot",
            "coordinator",
            "--reviewer-release-snapshot",
            "release",
            "--calibration-result-snapshot",
            "results",
            "--decision-input",
            "decision.csv",
            "--output-dir",
            "output",
        ),
    ),
)
def test_cli_legacy_modes_reject_reviewer_stage(
    legacy_arguments: tuple[str, ...],
) -> None:
    with pytest.raises(
        screening_results.ScreeningResultError,
        match="reviewer-stage",
    ):
        screening_results.main(
            (*legacy_arguments, "--reviewer-stage", "stage")
        )


def _forge_phase_input_binding(
    source: screening_results.PhaseResultSnapshot,
    coordinator: screening_results.CoordinatorSnapshot,
    output_dir: Path,
) -> tuple[screening_results.CoordinatorSnapshot, Path]:
    rows_by_batch = {
        batch_id: _read_csv(source.directory / f"{batch_id}.csv")
        for batch_id in screening_results.BATCH_IDS
    }
    target = next(row for rows in rows_by_batch.values() for row in rows)
    forged_input_sha256 = "f" * 64
    target["input_sha256"] = forged_input_sha256

    forged_manifest = [dict(row) for row in coordinator.manifest]
    for row in forged_manifest:
        if row["assignment_id"] == target["assignment_id"]:
            row["input_sha256"] = forged_input_sha256
    forged_coordinator = replace(
        coordinator,
        manifest=tuple(forged_manifest),
    )
    payloads_by_batch = {
        batch_id: screening_results._csv_bytes(
            screening_results.RESULT_HEADER,
            rows_by_batch[batch_id],
        )
        for batch_id in screening_results.BATCH_IDS
    }
    _, manifest, _, _, _ = screening_results._validate_phase_payloads(
        forged_coordinator,
        source.phase,
        payloads_by_batch,
        reviewer_release_sha256=source.reviewer_release_sha256,
    )
    artifacts = {
        f"{batch_id}.csv": payloads_by_batch[batch_id]
        for batch_id in screening_results.BATCH_IDS
    }
    artifacts["manifest.csv"] = screening_results._csv_bytes(
        screening_results.PHASE_RESULT_MANIFEST_HEADER,
        manifest,
    )
    artifacts["SHA256SUMS"] = screening_results._checksums(artifacts)
    output_dir.mkdir(parents=True)
    output_dir.chmod(0o755)
    for name, payload in artifacts.items():
        path = output_dir / name
        path.write_bytes(payload)
        path.chmod(0o644)
    return forged_coordinator, output_dir


def test_phase_validation_rejects_forged_coordinator_canonical_reseal(
    coordinator: Path,
    tmp_path: Path,
) -> None:
    _, snapshot_dir = _seal_phase(coordinator, tmp_path)
    authoritative = screening_results.validate_phase_result_snapshot(
        snapshot_dir,
        coordinator_snapshot_dir=coordinator,
        **_phase_validation_inputs(snapshot_dir),
    )
    forged_coordinator, forged_snapshot = _forge_phase_input_binding(
        authoritative,
        screening_results.capture_coordinator_snapshot(coordinator),
        tmp_path / "forged-phase" / "v1",
    )

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="input_sha256|coordinator|authoritative|reviewer release",
    ):
        screening_results.validate_phase_result_snapshot(
            forged_snapshot,
            coordinator=forged_coordinator,
            **_phase_validation_inputs(snapshot_dir),
        )


@pytest.mark.parametrize(
    "field",
    (
        "assignment_id",
        "phase",
        "candidate_id",
        "input_sha256",
        "snapshot_sha256",
        "batch_id",
    ),
)
def test_release_assignment_comparison_checks_every_shared_immutable_field(
    field: str,
) -> None:
    packet_row = {
        "assignment_id": "A-C0001-01",
        "phase": "calibration",
        "candidate_id": "C0001",
        "input_sha256": "1" * 64,
        "snapshot_sha256": "2" * 64,
        "batch_id": "screening-01",
    }
    result_row = dict(packet_row)
    result_row[field] = {
        "assignment_id": "A-C0001-02",
        "phase": "main",
        "candidate_id": "C9999",
        "input_sha256": "3" * 64,
        "snapshot_sha256": "4" * 64,
        "batch_id": "screening-02",
    }[field]

    with pytest.raises(screening_results.ScreeningResultError, match=field):
        screening_results._validate_reviewer_release_assignments(
            (result_row,),
            (packet_row,),
        )


def test_paths_overlap_compares_absolute_lexical_and_resolved_forms(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    assert screening_results.paths_overlap(alias / "nested", real)
    assert screening_results.paths_overlap(real, nested)
    assert "resolved" in (screening_results.paths_overlap.__doc__ or "")


def test_result_publication_routes_through_public_snapshot_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "v1"
    artifacts = {"manifest.csv": b"payload"}
    callback = lambda: None
    calls: list[tuple[Path, dict[str, bytes], object]] = []

    def public_publisher(
        output_dir: Path,
        supplied_artifacts: dict[str, bytes],
        *,
        post_publish_check=None,
    ) -> None:
        calls.append((output_dir, supplied_artifacts, post_publish_check))

    def private_publisher(*_args, **_kwargs) -> None:
        pytest.fail("screening_results bypassed public publish_snapshot")

    monkeypatch.setattr(
        screening_results.screening_batches,
        "publish_snapshot",
        public_publisher,
    )
    monkeypatch.setattr(
        screening_results.screening_batches,
        "_publish_artifacts",
        private_publisher,
    )

    screening_results._publish(
        output,
        artifacts,
        post_publish_check=callback,
    )

    assert calls == [(output, artifacts, callback)]


@pytest.fixture(scope="module")
def binary_coordinator_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    root = tmp_path_factory.mktemp("screening-results-binary-coordinator")
    inputs = build_inputs(root / "inputs", count=202)
    taxonomy = json.loads(inputs.taxonomy.read_text(encoding="utf-8"))
    taxonomy["screening_result_status"] = ["included", "excluded"]
    inputs.taxonomy.write_bytes(
        screening_results.screening_batches._canonical_json_bytes(taxonomy)
    )
    snapshot = root / "v7"
    freeze(inputs, snapshot)
    return snapshot


@pytest.fixture
def binary_coordinator(
    tmp_path: Path, binary_coordinator_template: Path
) -> Path:
    destination = tmp_path / "coordinator" / "v7"
    destination.parent.mkdir()
    import shutil

    shutil.copytree(binary_coordinator_template, destination)
    return destination


def test_unbound_decision_validation_retains_legacy_boundary(
    coordinator: Path,
) -> None:
    row = _decision_for_assignment(
        _manifest(coordinator)[0], status="boundary"
    )

    screening_results.validate_result_decision(row, context="unbound")


@pytest.mark.parametrize("status", ("included", "excluded"))
def test_binary_phase_accepts_allowed_screening_statuses(
    binary_coordinator: Path,
    tmp_path: Path,
    status: str,
) -> None:
    release = _release_phase(binary_coordinator, tmp_path, "calibration")
    paths = _phase_result_paths(
        tmp_path / f"raw-binary-{status}",
        binary_coordinator,
        "calibration",
    )
    for path in paths:
        rows = _read_csv(path)
        for row in rows:
            if status == "excluded":
                row["screening_status"] = "excluded"
                row["criterion"] = "exclude-out-of-scope"
                row["exclusion_reason"] = (
                    "The inspected source does not contribute course-generation "
                    "geometry, representations, metrics, or interchange."
                )
        _write_csv(path, RESULT_HEADER, rows)
    output = tmp_path / f"sealed-binary-{status}" / "v1"
    output.parent.mkdir()

    screening_results.seal_phase_results(
        coordinator_snapshot_dir=binary_coordinator,
        phase="calibration",
        result_paths=paths,
        output_dir=output,
        reviewer_release_snapshot_dir=release,
    )


def test_binary_phase_rejects_boundary_screening_status(
    binary_coordinator: Path,
    tmp_path: Path,
) -> None:
    release = _release_phase(binary_coordinator, tmp_path, "calibration")
    paths = _phase_result_paths(
        tmp_path / "raw-binary-boundary",
        binary_coordinator,
        "calibration",
    )
    for path in paths:
        rows = _read_csv(path)
        for row in rows:
            row["screening_status"] = "boundary"
            row["criterion"] = "boundary"
        _write_csv(path, RESULT_HEADER, rows)
    output = tmp_path / "sealed-binary-boundary" / "v1"
    output.parent.mkdir()

    with pytest.raises(
        screening_results.ScreeningResultError,
        match="invalid screening_status",
    ):
        screening_results.seal_phase_results(
            coordinator_snapshot_dir=binary_coordinator,
            phase="calibration",
            result_paths=paths,
            output_dir=output,
            reviewer_release_snapshot_dir=release,
        )


@pytest.mark.parametrize(
    ("phase", "version"),
    (
        ("calibration", "v1"),
        ("calibration", "v2"),
        ("main", "v2"),
        ("calibration", "v4"),
        ("calibration", "v5"),
        ("calibration", "v6"),
    ),
)
def test_committed_result_snapshot_still_validates(
    phase: str,
    version: str,
) -> None:
    gate_inputs: dict[str, Path] = {}
    if phase == "main":
        gate_inputs = {
            "calibration_reviewer_release_snapshot_dir": Path(
                "paper/data/screening_releases/calibration/v2"
            ),
            "calibration_result_snapshot_dir": Path(
                "paper/data/screening_results/calibration/v2"
            ),
            "calibration_decision_snapshot_dir": Path(
                "paper/data/screening_decisions/v2"
            ),
        }
    captured = screening_results.validate_phase_result_snapshot(
        Path("paper/data/screening_results") / phase / version,
        coordinator_snapshot_dir=Path("paper/data/screening_inputs") / version,
        reviewer_release_snapshot_dir=(
            Path("paper/data/screening_releases") / phase / version
        ),
        **gate_inputs,
    )

    assert captured.phase == phase


def test_binary_coordinator_captures_allowed_screening_statuses(
    binary_coordinator: Path,
) -> None:
    captured = screening_results.capture_coordinator_snapshot(binary_coordinator)

    assert captured.allowed_screening_statuses == ("included", "excluded")
