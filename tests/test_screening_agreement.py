from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import paper.scripts.prepare_screening_batches as screening_batches
import paper.scripts.screening_agreement as agreement
import paper.scripts.screening_results as screening_results
from tests.test_screening_batches import build_inputs, freeze


PROTOCOL_HASH = "1" * 64
COORDINATOR_HASH = "2" * 64
CALIBRATION_HASH = "3" * 64
MAIN_HASH = "4" * 64
CALIBRATION_RELEASE_HASH = "5" * 64
MAIN_RELEASE_HASH = "6" * 64
PRIMARY_HASH = "11f65fdced90dcc0b4c931730113def8827f29480b2261bba4b0e358586dcfd1"
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _SealedAgreementSnapshots:
    coordinator: Path
    calibration_release: Path
    calibration: screening_results.PhaseResultSnapshot
    calibration_decision: screening_results.CalibrationDecisionSnapshot
    main_release: Path
    main: screening_results.PhaseResultSnapshot


def _sealed_result_row(
    manifest_row: dict[str, str], status: str
) -> dict[str, str]:
    candidate_id = manifest_row["candidate_id"]
    criterion = {
        "included": "include-relevant",
        "boundary": "boundary",
        "excluded": "exclude-out-of-scope",
    }[status]
    return {
        "assignment_id": manifest_row["assignment_id"],
        "phase": manifest_row["phase"],
        "candidate_id": candidate_id,
        "input_sha256": manifest_row["input_sha256"],
        "snapshot_sha256": manifest_row["snapshot_sha256"],
        "batch_id": manifest_row["batch_id"],
        "coder_id": manifest_row["batch_id"],
        "screened_on": "2026-06-30",
        "screening_status": status,
        "criterion": criterion,
        "access_status": "full_text",
        "source_urls": f"https://example.test/source/{candidate_id}",
        "evidence_version": "publisher-version-of-record",
        "evidence_retrieved_on": "2026-06-29",
        "evidence_archive_url": (
            f"https://archive.example.test/20260629/{candidate_id}"
        ),
        "evidence_sha256": hashlib.sha256(
            f"evidence:{candidate_id}".encode("ascii")
        ).hexdigest(),
        "screening_locator": "Section 2; Algorithm 1",
        "exclusion_reason": (
            "NR"
            if status != "excluded"
            else "The source does not contribute in-scope course generation."
        ),
        "notes": "NR",
    }


_SEALED_PHASE_INPUTS: dict[Path, dict[str, Path]] = {}


def _seal_phase(
    root: Path,
    coordinator: Path,
    manifest: list[dict[str, str]],
    phase: str,
    *,
    reviewer_release: Path,
    calibration_reviewer_release: Path | None = None,
    calibration_result: Path | None = None,
    calibration_decision: Path | None = None,
) -> screening_results.PhaseResultSnapshot:
    phase_rows: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    by_candidate: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        if row["phase"] == phase:
            by_candidate[row["candidate_id"]].append(row)
    statuses = ("included", "excluded")
    for candidate_number, candidate_id in enumerate(
        sorted(by_candidate, key=lambda value: value.encode("utf-8"))
    ):
        assignments = sorted(
            by_candidate[candidate_id],
            key=lambda row: row["assignment_id"].encode("utf-8"),
        )
        base_index = candidate_number % len(statuses)
        for rating_number, manifest_row in enumerate(assignments):
            status_index = base_index
            if candidate_number % 5 == 0 and rating_number == 1:
                status_index = (base_index + 1) % len(statuses)
            result = _sealed_result_row(
                manifest_row, statuses[status_index]
            )
            phase_rows[result["batch_id"]].append(result)

    raw_root = root / f"raw-{phase}"
    raw_root.mkdir()
    paths: list[Path] = []
    for batch_id in screening_results.BATCH_IDS:
        result_path = raw_root / f"{batch_id}.csv"
        with result_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=screening_results.RESULT_HEADER,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                sorted(
                    phase_rows[batch_id],
                    key=lambda row: (
                        row["candidate_id"].encode("utf-8"),
                        row["assignment_id"].encode("utf-8"),
                    ),
                )
            )
        paths.append(result_path)

    snapshot_root = root / f"{phase}-snapshots"
    snapshot_root.mkdir()
    snapshot_dir = snapshot_root / "v1"
    kwargs: dict[str, Path] = {
        "reviewer_release_snapshot_dir": reviewer_release,
    }
    if phase == "main":
        assert calibration_reviewer_release is not None
        assert calibration_result is not None
        assert calibration_decision is not None
        kwargs.update(
            calibration_reviewer_release_snapshot_dir=(
                calibration_reviewer_release
            ),
            calibration_result_snapshot_dir=calibration_result,
            calibration_decision_snapshot_dir=calibration_decision,
        )
    screening_results.seal_phase_results(
        coordinator,
        phase,
        paths,
        snapshot_dir,
        **kwargs,
    )
    _SEALED_PHASE_INPUTS[snapshot_dir] = dict(kwargs)
    return screening_results.validate_phase_result_snapshot(
        snapshot_dir,
        coordinator_snapshot_dir=coordinator,
        **kwargs,
    )


@pytest.fixture(scope="module")
def sealed_snapshots(
    tmp_path_factory: pytest.TempPathFactory,
) -> _SealedAgreementSnapshots:
    root = tmp_path_factory.mktemp("screening-agreement-sealed")
    inputs = build_inputs(root / "inputs", count=202)
    coordinator_root = root / "coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    with (coordinator / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle, strict=True))

    calibration_release_root = root / "calibration-release"
    calibration_release_root.mkdir()
    calibration_release = calibration_release_root / "v1"
    screening_batches.release_snapshot(
        coordinator,
        "calibration",
        calibration_release,
    )
    calibration = _seal_phase(
        root,
        coordinator,
        manifest,
        "calibration",
        reviewer_release=calibration_release,
    )

    by_candidate: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in calibration.rows:
        by_candidate[row["candidate_id"]].append(row)
    numerator = sum(
        len({row["screening_status"] for row in rows}) == 1
        for rows in by_candidate.values()
    )
    with (coordinator / "calibration_selection.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        candidate_ids = [
            row["candidate_id"]
            for row in csv.DictReader(handle, strict=True)
        ]
    decision_row = {
        "decision_id": "agreement-calibration-gate-v1",
        "protocol_sha256": calibration.protocol_sha256,
        "coordinator_snapshot_sha256": (
            calibration.coordinator_snapshot_sha256
        ),
        "calibration_result_snapshot_sha256": calibration.snapshot_sha256,
        "candidate_ids_sha256": screening_results.sequence_ids_sha256(
            candidate_ids
        ),
        "assignment_ids_sha256": screening_results.ordered_ids_sha256(
            row["assignment_id"] for row in calibration.rows
        ),
        "status_agreement_numerator": str(numerator),
        "status_agreement_denominator": str(len(by_candidate)),
        "status_agreement": screening_results.canonical_ratio(
            numerator, len(by_candidate)
        ),
        "systematic_ambiguity": "false",
        "decision": "release",
        "decided_on": "2026-06-30",
        "decision_makers": "accountable-author;survey-coordinator",
        "resolution_evidence": (
            "The locked calibration pairs and every disagreement were reviewed "
            "against the frozen operational rules before main release."
        ),
    }
    decision_input = root / "calibration-decision-input.csv"
    with decision_input.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=screening_results.CALIBRATION_DECISION_HEADER,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(decision_row)
    decision_root = root / "calibration-decisions"
    decision_root.mkdir()
    decision = decision_root / "v1"
    screening_results.seal_calibration_decision(
        coordinator,
        calibration.directory,
        decision_input,
        decision,
        calibration_reviewer_release_snapshot_dir=calibration_release,
    )
    captured_decision = (
        screening_results.validate_calibration_decision_snapshot(
            decision,
            coordinator_snapshot_dir=coordinator,
            calibration_reviewer_release_snapshot_dir=calibration_release,
            calibration_result_snapshot_dir=calibration.directory,
        )
    )

    captured_coordinator = screening_results.capture_coordinator_snapshot(
        coordinator
    )
    main_release_root = root / "main-release"
    main_release_root.mkdir()
    main_release = main_release_root / "v1"
    main_artifacts = screening_batches.build_reviewer_release_artifacts(
        captured_coordinator.payloads,
        "main",
        calibration_result_snapshot_sha256=calibration.snapshot_sha256,
        calibration_decision_snapshot_sha256=(
            captured_decision.snapshot_sha256
        ),
    )
    screening_batches.publish_snapshot(main_release, main_artifacts)
    main = _seal_phase(
        root,
        coordinator,
        manifest,
        "main",
        reviewer_release=main_release,
        calibration_reviewer_release=calibration_release,
        calibration_result=calibration.directory,
        calibration_decision=decision,
    )
    return _SealedAgreementSnapshots(
        coordinator=coordinator,
        calibration_release=calibration_release,
        calibration=calibration,
        calibration_decision=captured_decision,
        main_release=main_release,
        main=main,
    )


def _public_arguments(
    snapshots: _SealedAgreementSnapshots,
) -> dict[str, object]:
    return {
        "coordinator_snapshot": snapshots.coordinator,
        "calibration_reviewer_release_snapshot_dir": (
            snapshots.calibration_release
        ),
        "calibration": snapshots.calibration,
        "calibration_decision": snapshots.calibration_decision,
        "main_reviewer_release_snapshot_dir": snapshots.main_release,
        "main": snapshots.main,
    }


def _forge_self_declared_phase_snapshot(
    source: screening_results.PhaseResultSnapshot,
    output_dir: Path,
    *,
    coordinator_sha256: str,
    protocol_sha256: str,
) -> screening_results.PhaseResultSnapshot:
    payloads: dict[str, bytes] = {}
    for batch_id in screening_results.BATCH_IDS:
        with (source.directory / f"{batch_id}.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle, strict=True))
        for row in rows:
            row["snapshot_sha256"] = coordinator_sha256
            if row["screening_status"] == "included":
                row["criterion"] = "include-1"
        payloads[batch_id] = screening_results._csv_bytes(
            screening_results.RESULT_HEADER, rows
        )

    rows, manifest, snapshot_sha256, _, _ = (
        screening_results._validate_phase_payloads(
            None,
            source.phase,
            payloads,
            reviewer_release_sha256=source.reviewer_release_sha256,
            sealed_coordinator_hash=coordinator_sha256,
            sealed_protocol_hash=protocol_sha256,
        )
    )
    artifacts = {
        f"{batch_id}.csv": payloads[batch_id]
        for batch_id in screening_results.BATCH_IDS
    }
    artifacts["manifest.csv"] = screening_results._csv_bytes(
        screening_results.PHASE_RESULT_MANIFEST_HEADER, manifest
    )
    artifacts["SHA256SUMS"] = screening_results._checksums(artifacts)
    output_dir.mkdir(parents=True)
    output_dir.chmod(0o755)
    for name, payload in artifacts.items():
        target = output_dir / name
        target.write_bytes(payload)
        target.chmod(0o644)
    return screening_results.PhaseResultSnapshot(
        directory=output_dir,
        phase=source.phase,
        rows=tuple(rows),
        snapshot_sha256=snapshot_sha256,
        coordinator_snapshot_sha256=coordinator_sha256,
        protocol_sha256=protocol_sha256,
        reviewer_release_sha256=source.reviewer_release_sha256,
        manifest=tuple(manifest),
        fingerprints=(),
    )


def _status(criterion: str) -> str:
    if criterion in screening_results.INCLUSION_CRITERIA:
        return "included"
    if criterion == "boundary":
        return "boundary"
    assert criterion in screening_results.EXCLUSION_CRITERIA
    return "excluded"


def _metadata() -> list[dict[str, str]]:
    source_types = (
        "ISO standard",
        "robot competition",
        "benchmark dataset",
        "software repository",
        "journal article",
        "official documentation",
    )
    return [
        {
            "candidate_id": f"C{number:04d}",
            "source_type": source_types[(number - 1) % 6],
            "discovery_stream": f"stream-{number % 13};shared-{number % 5}",
            "discovery_query": f"query-{number % 17}",
        }
        for number in range(1, 203)
    ]


def _row(
    candidate_id: str, rating: int, phase: str, criterion: str
) -> dict[str, str]:
    status = _status(criterion)
    row = dict.fromkeys(screening_results.RESULT_HEADER, "NR")
    row.update(
        assignment_id=f"A-{candidate_id}-{rating:02d}",
        phase=phase,
        candidate_id=candidate_id,
        input_sha256=hashlib.sha256(candidate_id.encode()).hexdigest(),
        snapshot_sha256=COORDINATOR_HASH,
        batch_id=f"screening-{rating:02d}",
        coder_id=f"reviewer-{rating}",
        screened_on="2026-06-30",
        screening_status=status,
        criterion=criterion,
        access_status="full_text",
        source_urls=f"https://example.org/{candidate_id}",
        evidence_version="publisher-version-of-record",
        evidence_retrieved_on="2026-06-29",
        evidence_archive_url=f"https://archive.example.org/{candidate_id}",
        evidence_sha256=hashlib.sha256(
            f"evidence:{candidate_id}".encode()
        ).hexdigest(),
        screening_locator="Section 3; Algorithm 1",
        exclusion_reason=(
            "NR" if status != "excluded" else "The generated object is out of scope."
        ),
        notes="NR",
    )
    return row


def _snapshots() -> tuple[
    screening_results.PhaseResultSnapshot,
    screening_results.PhaseResultSnapshot,
]:
    metadata = _metadata()
    calibration_ids = screening_batches._select_calibration_candidate_ids(metadata)
    assert len(calibration_ids) == 30
    assert calibration_ids != frozenset(row["candidate_id"] for row in metadata[:30])
    rows: dict[str, list[dict[str, str]]] = {"calibration": [], "main": []}
    criteria = tuple(
        criterion
        for criterion in agreement.CRITERION_CATEGORIES
        if criterion != "include-relevant"
    )
    for number, candidate in enumerate(metadata, start=1):
        candidate_id = candidate["candidate_id"]
        phase = "calibration" if candidate_id in calibration_ids else "main"
        left = criteria[(number - 1) % len(criteria)]
        right = left
        status = _status(left)
        if number % 7 == 0 and status == "included":
            options = tuple(screening_results.INCLUSION_CRITERIA)
            right = options[(options.index(left) + 1) % len(options)]
        elif number % 7 == 0 and status == "excluded":
            options = agreement.EXCLUSION_CRITERIA
            right = options[(options.index(left) + 1) % len(options)]
        if number % 11 == 0:
            right = "boundary" if status != "boundary" else "include-1"
        rows[phase].extend(
            (_row(candidate_id, 1, phase, left), _row(candidate_id, 2, phase, right))
        )
    return (
        screening_results.PhaseResultSnapshot(
            Path("/immutable/calibration"),
            "calibration",
            tuple(reversed(rows["calibration"])),
            CALIBRATION_HASH,
            COORDINATOR_HASH,
            PROTOCOL_HASH,
            CALIBRATION_RELEASE_HASH,
            (),
            (),
        ),
        screening_results.PhaseResultSnapshot(
            Path("/immutable/main"),
            "main",
            tuple(reversed(rows["main"])),
            MAIN_HASH,
            COORDINATOR_HASH,
            PROTOCOL_HASH,
            MAIN_RELEASE_HASH,
            (),
            (),
        ),
    )


def _build(
    calibration: screening_results.PhaseResultSnapshot,
    main: screening_results.PhaseResultSnapshot,
    replicates: int = 40,
) -> list[dict[str, str]]:
    return agreement._build_agreement_report(
        calibration,
        main,
        coordinator_snapshot_sha256=COORDINATOR_HASH,
        protocol_sha256=PROTOCOL_HASH,
        bootstrap_replicates=replicates,
    )


def _matrix_pairs(
    matrix: tuple[tuple[int, ...], ...],
) -> list[agreement.RatingPair]:
    criteria = {
        "included": "include-1",
        "boundary": "boundary",
        "excluded": "exclude-out-of-scope",
    }
    pairs: list[agreement.RatingPair] = []
    number = 0
    for row_index, left in enumerate(agreement.STATUS_CATEGORIES):
        for column_index, right in enumerate(agreement.STATUS_CATEGORIES):
            for _ in range(matrix[row_index][column_index]):
                number += 1
                pairs.append(
                    agreement.RatingPair(
                        f"R{number:04d}",
                        left,
                        right,
                        criteria[left],
                        criteria[right],
                    )
                )
    return pairs


def test_imports_result_contract_and_closed_vocabularies() -> None:
    assert agreement.RESULT_HEADER is screening_results.RESULT_HEADER
    assert agreement.CRITERION_CATEGORIES == (
        "include-1",
        "include-2",
        "include-3",
        "include-4",
        "include-relevant",
        "boundary",
        "exclude-fixed-racing-line",
        "exclude-appearance-dynamics",
        "exclude-traffic-only",
        "exclude-insufficient-detail",
        "exclude-out-of-scope",
    )


def test_reference_matrix_matches_independent_values() -> None:
    estimates = agreement._point_estimates(
        _matrix_pairs(((3, 1, 0), (1, 2, 1), (0, 1, 3)))
    )
    assert estimates.exact_status_count == 8
    assert estimates.exact_status_rate == Fraction(2, 3)
    assert estimates.positive_agreement == {
        "included": Fraction(3, 4),
        "boundary": Fraction(1, 2),
        "excluded": Fraction(3, 4),
    }
    assert estimates.negative_agreement == {
        "included": Fraction(7, 8),
        "boundary": Fraction(3, 4),
        "excluded": Fraction(7, 8),
    }
    assert estimates.krippendorff_alpha == Fraction(25, 48)
    assert estimates.gwet_ac1 == Fraction(1, 2)


def test_public_api_revalidates_against_coordinator_and_derives_provenance(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    calibration = sealed_snapshots.calibration
    main = sealed_snapshots.main
    report = agreement.build_agreement_report(
        **_public_arguments(sealed_snapshots)
    )
    primary_hash = agreement._combined_primary_snapshot_sha256(
        calibration.snapshot_sha256, main.snapshot_sha256
    )

    assert [row["scope"] for row in report] == [
        "calibration",
        "full_corpus",
    ]
    assert [row["candidate_count"] for row in report] == ["30", "202"]
    assert all(
        row["protocol_sha256"] == calibration.protocol_sha256
        and row["coordinator_snapshot_sha256"]
        == calibration.coordinator_snapshot_sha256
        and row["calibration_result_snapshot_sha256"]
        == calibration.snapshot_sha256
        and row["main_result_snapshot_sha256"] == main.snapshot_sha256
        and row["primary_result_snapshot_sha256"] == primary_hash
        and row["bootstrap_algorithm"] == "screening-bootstrap-v1"
        for row in report
    )
    assert all(
        row[f"{metric}_bootstrap_replicates"] == "10000"
        for row in report
        for metric in agreement.BOOTSTRAP_METRICS
    )


def test_v7_binary_agreement_report_uses_bound_criteria_in_bootstrap(
    sealed_snapshots: _SealedAgreementSnapshots,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agreement, "PRODUCTION_BOOTSTRAP_REPLICATES", 2)

    report = agreement.build_agreement_report(
        **_public_arguments(sealed_snapshots)
    )

    field = (
        "criterion_a_include_relevant_criterion_b_exclude_out_of_scope"
    )
    assert field in agreement.AGREEMENT_REPORT_HEADER
    assert all(
        row["overall_exact_status_agreement_bootstrap_replicates"] == "2"
        for row in report
    )
    assert any(row[field] != "0" for row in report)


def test_public_api_removes_hash_and_bootstrap_overrides(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    arguments = _public_arguments(sealed_snapshots)
    with pytest.raises(TypeError, match="coordinator_snapshot_sha256"):
        agreement.build_agreement_report(
            **arguments,
            coordinator_snapshot_sha256=(
                sealed_snapshots.calibration.coordinator_snapshot_sha256
            ),
            protocol_sha256=sealed_snapshots.calibration.protocol_sha256,
        )
    with pytest.raises(TypeError, match="bootstrap_replicates"):
        agreement.build_agreement_report(
            **arguments,
            bootstrap_replicates=10,
        )


@pytest.mark.parametrize(
    "omitted_argument",
    (
        "calibration_reviewer_release_snapshot_dir",
        "calibration_decision",
        "main_reviewer_release_snapshot_dir",
    ),
)
def test_public_api_requires_gate_replay_provenance(
    sealed_snapshots: _SealedAgreementSnapshots,
    omitted_argument: str,
) -> None:
    arguments = _public_arguments(sealed_snapshots)
    del arguments[omitted_argument]

    with pytest.raises(TypeError, match=omitted_argument):
        agreement.build_agreement_report(**arguments)


@pytest.mark.parametrize(
    ("release_argument", "replacement"),
    (
        (
            "calibration_reviewer_release_snapshot_dir",
            "main_release",
        ),
        (
            "main_reviewer_release_snapshot_dir",
            "calibration_release",
        ),
    ),
)
def test_public_api_rejects_substituted_reviewer_release(
    sealed_snapshots: _SealedAgreementSnapshots,
    release_argument: str,
    replacement: str,
) -> None:
    arguments = _public_arguments(sealed_snapshots)
    arguments[release_argument] = getattr(sealed_snapshots, replacement)

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="authoritative revalidation",
    ):
        agreement.build_agreement_report(**arguments)


def test_public_api_rejects_directly_tampered_snapshot_object(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    calibration = sealed_snapshots.calibration
    rows = [dict(row) for row in calibration.rows]
    rows[0]["notes"] = "tampered in-memory decision"
    tampered = replace(calibration, rows=tuple(rows))
    arguments = _public_arguments(sealed_snapshots)
    arguments["calibration"] = tampered

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="rows.*authoritative|does not match authoritative",
    ):
        agreement.build_agreement_report(**arguments)


def test_public_api_rejects_tampered_embedded_release_hash(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    arguments = _public_arguments(sealed_snapshots)
    arguments["calibration"] = replace(
        sealed_snapshots.calibration,
        reviewer_release_sha256="f" * 64,
    )

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="reviewer_release_sha256.*authoritative",
    ):
        agreement.build_agreement_report(**arguments)


def test_public_api_rejects_tampered_calibration_decision_object(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    decision = dict(sealed_snapshots.calibration_decision.decision)
    decision["decision"] = "hold"
    arguments = _public_arguments(sealed_snapshots)
    arguments["calibration_decision"] = replace(
        sealed_snapshots.calibration_decision,
        decision=decision,
    )

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="calibration decision snapshot decision.*authoritative",
    ):
        agreement.build_agreement_report(**arguments)


def test_public_revalidation_accepts_coordinator_object_and_reconstruction(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    calibration = sealed_snapshots.calibration
    reconstructed = screening_results.PhaseResultSnapshot(
        directory=calibration.directory,
        phase=calibration.phase,
        rows=tuple(dict(row) for row in calibration.rows),
        snapshot_sha256=calibration.snapshot_sha256,
        coordinator_snapshot_sha256=(
            calibration.coordinator_snapshot_sha256
        ),
        protocol_sha256=calibration.protocol_sha256,
        reviewer_release_sha256=calibration.reviewer_release_sha256,
        manifest=tuple(dict(row) for row in calibration.manifest),
        fingerprints=tuple(calibration.fingerprints),
    )

    captured_coordinator = screening_results.capture_coordinator_snapshot(
        sealed_snapshots.coordinator
    )
    authoritative = agreement._revalidate_phase_snapshot(
        reconstructed,
        expected_phase="calibration",
        coordinator_snapshot=captured_coordinator,
        reviewer_release_snapshot_dir=(
            sealed_snapshots.calibration_release
        ),
    )

    assert authoritative.rows == calibration.rows
    assert authoritative.manifest == calibration.manifest
    assert authoritative.fingerprints == calibration.fingerprints


def test_public_revalidation_rejects_untyped_coordinator_duck_object(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    arguments = _public_arguments(sealed_snapshots)
    arguments["coordinator_snapshot"] = SimpleNamespace(
        directory=sealed_snapshots.coordinator
    )

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="Path or CoordinatorSnapshot",
    ):
        agreement.build_agreement_report(**arguments)


def test_forged_self_declared_phase_hashes_fail_real_coordinator_anchor(
    sealed_snapshots: _SealedAgreementSnapshots,
    tmp_path: Path,
) -> None:
    calibration = sealed_snapshots.calibration
    main = sealed_snapshots.main
    fake_coordinator = "e" * 64
    fake_protocol = "d" * 64
    forged_calibration = _forge_self_declared_phase_snapshot(
        calibration,
        tmp_path / "forged-calibration" / "v1",
        coordinator_sha256=fake_coordinator,
        protocol_sha256=fake_protocol,
    )
    forged_main = _forge_self_declared_phase_snapshot(
        main,
        tmp_path / "forged-main" / "v1",
        coordinator_sha256=fake_coordinator,
        protocol_sha256=fake_protocol,
    )
    assert forged_calibration.coordinator_snapshot_sha256 == fake_coordinator
    assert forged_main.protocol_sha256 == fake_protocol
    arguments = _public_arguments(sealed_snapshots)
    arguments["calibration"] = forged_calibration
    arguments["main"] = forged_main

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="authoritative revalidation|frozen assignment|coordinator",
    ):
        agreement.build_agreement_report(**arguments)


def test_normative_combined_hash_and_first_draws() -> None:
    assert agreement.combined_primary_snapshot_sha256(
        CALIBRATION_HASH, MAIN_HASH
    ) == PRIMARY_HASH
    assert (
        agreement._combined_primary_snapshot_sha256
        is agreement.combined_primary_snapshot_sha256
    )
    assert agreement._bootstrap_index(
        PRIMARY_HASH, "calibration", replicate=0, draw=0, population_size=30
    ) == 28
    assert agreement._bootstrap_index(
        PRIMARY_HASH, "full_corpus", replicate=0, draw=0, population_size=202
    ) == 68


def test_full_corpus_uses_global_utf8_candidate_order() -> None:
    calibration, main = _snapshots()
    calibration_pairs, main_pairs, full_pairs = agreement._validated_snapshot_pairs(
        calibration,
        main,
        coordinator_snapshot_sha256=COORDINATOR_HASH,
        protocol_sha256=PROTOCOL_HASH,
    )
    expected = sorted(
        (pair.candidate_id for pair in (*calibration_pairs, *main_pairs)),
        key=lambda value: value.encode("utf-8"),
    )
    assert [pair.candidate_id for pair in full_pairs] == expected
    assert [pair.candidate_id for pair in full_pairs] != [
        pair.candidate_id for pair in (*calibration_pairs, *main_pairs)
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (
            lambda calibration, main: (
                replace(calibration, coordinator_snapshot_sha256="9" * 64),
                main,
                COORDINATOR_HASH,
                PROTOCOL_HASH,
            ),
            "coordinator",
        ),
        (
            lambda calibration, main: (
                calibration,
                replace(main, protocol_sha256="9" * 64),
                COORDINATOR_HASH,
                PROTOCOL_HASH,
            ),
            "protocol",
        ),
        (
            lambda calibration, main: (
                main,
                calibration,
                COORDINATOR_HASH,
                PROTOCOL_HASH,
            ),
            "phase",
        ),
        (
            lambda calibration, main: (
                calibration,
                main,
                "9" * 64,
                PROTOCOL_HASH,
            ),
            "coordinator",
        ),
    ),
)
def test_rejects_arbitrary_hashes_and_phase_swaps(change, message: str) -> None:
    calibration, main = _snapshots()
    first, second, coordinator_hash, protocol_hash = change(calibration, main)
    with pytest.raises(agreement.ScreeningAgreementError, match=message):
        agreement._build_agreement_report(
            first,
            second,
            coordinator_snapshot_sha256=coordinator_hash,
            protocol_sha256=protocol_hash,
            bootstrap_replicates=5,
        )


def test_rejects_rows_instead_of_phase_snapshot() -> None:
    calibration, main = _snapshots()
    with pytest.raises(agreement.ScreeningAgreementError, match="PhaseResultSnapshot"):
        agreement._build_agreement_report(
            list(calibration.rows),
            main,
            coordinator_snapshot_sha256=COORDINATOR_HASH,
            protocol_sha256=PROTOCOL_HASH,
            bootstrap_replicates=5,
        )


def test_rejects_invalid_status_criterion_pair() -> None:
    calibration, main = _snapshots()
    rows = [dict(row) for row in calibration.rows]
    rows[0].update(
        screening_status="included", criterion="exclude-out-of-scope"
    )
    with pytest.raises(
        agreement.ScreeningAgreementError, match="criterion.*invalid for included"
    ):
        _build(replace(calibration, rows=tuple(rows)), main, 5)


def test_v7_binary_agreement_rejects_injected_boundary_status(
    sealed_snapshots: _SealedAgreementSnapshots,
) -> None:
    coordinator = screening_results.capture_coordinator_snapshot(
        sealed_snapshots.coordinator
    )
    rows = [dict(row) for row in sealed_snapshots.calibration.rows]
    rows[0].update(screening_status="boundary", criterion="boundary")

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="invalid screening_status",
    ):
        agreement._build_agreement_report(
            replace(sealed_snapshots.calibration, rows=tuple(rows)),
            sealed_snapshots.main,
            coordinator_snapshot_sha256=coordinator.snapshot_sha256,
            protocol_sha256=coordinator.protocol_sha256,
            allowed_inclusion_criteria=coordinator.allowed_inclusion_criteria,
            allowed_screening_statuses=coordinator.allowed_screening_statuses,
            bootstrap_replicates=5,
        )


@pytest.mark.parametrize("version", ("v5", "v6"))
def test_historical_agreement_coordinator_keeps_legacy_boundary_status(
    version: str,
) -> None:
    coordinator = screening_results.capture_coordinator_snapshot(
        ROOT / "paper" / "data" / "screening_inputs" / version
    )

    assert coordinator.allowed_screening_statuses == (
        "included",
        "boundary",
        "excluded",
    )
    agreement._validate_status_criterion(
        "boundary",
        "boundary",
        candidate_id="C0001",
        allowed_inclusion_criteria=coordinator.allowed_inclusion_criteria,
        allowed_screening_statuses=coordinator.allowed_screening_statuses,
    )


def test_criterion_disagreement_table_matches_raw_pairs() -> None:
    calibration, main = _snapshots()
    full_row = _build(calibration, main)[1]
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in (*calibration.rows, *main.rows):
        grouped[row["candidate_id"]].append(row)
    expected: Counter[tuple[str, str]] = Counter()
    for ratings in grouped.values():
        first, second = sorted(
            ratings, key=lambda row: row["assignment_id"].encode("utf-8")
        )
        if first["criterion"] != second["criterion"]:
            expected[(first["criterion"], second["criterion"])] += 1
    for left in agreement.CRITERION_CATEGORIES:
        for right in agreement.CRITERION_CATEGORIES:
            if left != right:
                field = agreement._criterion_disagreement_field(left, right)
                assert full_row[field] == str(expected[(left, right)])
    assert sum(
        int(full_row[field]) for field in agreement.CRITERION_DISAGREEMENT_FIELDS
    ) == 202 - int(full_row["exact_criterion_agreement_count"])


def test_sparse_valid_alpha_uses_all_finite_replicates() -> None:
    pairs = [
        agreement.RatingPair(
            f"C{number:04d}",
            "excluded" if number == 30 else "included",
            "excluded" if number == 30 else "included",
            "exclude-out-of-scope" if number == 30 else "include-1",
            "exclude-out-of-scope" if number == 30 else "include-1",
        )
        for number in range(1, 31)
    ]
    intervals = agreement._bootstrap_intervals(
        pairs,
        scope="calibration",
        primary_snapshot_sha256=PRIMARY_HASH,
        replicates=200,
    )
    alpha = intervals["krippendorff_alpha_nominal"]
    assert 0 < alpha.valid_replicates < 190
    assert alpha.lower is not None and alpha.upper is not None


def test_zero_valid_bootstrap_metric_is_hard_error() -> None:
    pairs = [
        agreement.RatingPair(
            f"C{number:04d}",
            "included",
            "included",
            "include-1",
            "include-1",
        )
        for number in range(1, 31)
    ]
    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="zero valid.*krippendorff_alpha_nominal",
    ):
        agreement._bootstrap_intervals(
            pairs,
            scope="calibration",
            primary_snapshot_sha256=PRIMARY_HASH,
            replicates=20,
        )


def test_order_and_adjudication_extras_do_not_change_report() -> None:
    calibration, main = _snapshots()
    calibration_rows = [dict(row) for row in calibration.rows]
    main_rows = [dict(row) for row in main.rows]
    random.Random(91).shuffle(calibration_rows)
    random.Random(72).shuffle(main_rows)
    for row in (*calibration_rows, *main_rows):
        row["final_screening_status"] = "excluded"
        row["adjudicator_id"] = "ignored"
    assert _build(
        replace(calibration, rows=tuple(calibration_rows)),
        replace(main, rows=tuple(main_rows)),
    ) == _build(calibration, main)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda rows: rows.pop(), "60 ratings|exactly two"),
        (
            lambda rows: rows[1].__setitem__("assignment_id", rows[0]["assignment_id"]),
            "distinct assignment",
        ),
        (
            lambda rows: rows[1].__setitem__("coder_id", rows[0]["coder_id"]),
            "distinct coder",
        ),
        (
            lambda rows: rows[1].__setitem__(
                "input_sha256", hashlib.sha256(b"different").hexdigest()
            ),
            "matching input_sha256",
        ),
    ),
)
def test_missing_duplicate_and_mismatch_fail(mutation, message: str) -> None:
    calibration, main = _snapshots()
    rows = [dict(row) for row in calibration.rows]
    mutation(rows)
    with pytest.raises(agreement.ScreeningAgreementError, match=message):
        _build(replace(calibration, rows=tuple(rows)), main, 5)


def test_rendered_csv_is_canonical() -> None:
    calibration, main = _snapshots()
    payload = agreement.render_agreement_csv(_build(calibration, main, 10))
    assert payload.endswith(b"\n") and b"\r" not in payload
    reader = csv.DictReader(io.StringIO(payload.decode(), newline=""))
    assert tuple(reader.fieldnames or ()) == agreement.AGREEMENT_REPORT_HEADER
    assert [row["scope"] for row in reader] == ["calibration", "full_corpus"]


def _forge_main_input_binding(
    source: screening_results.PhaseResultSnapshot,
    coordinator: screening_results.CoordinatorSnapshot,
    output_dir: Path,
) -> tuple[
    screening_results.CoordinatorSnapshot,
    screening_results.PhaseResultSnapshot,
]:
    rows_by_batch: dict[str, list[dict[str, str]]] = {}
    for batch_id in screening_results.BATCH_IDS:
        with (source.directory / f"{batch_id}.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            rows_by_batch[batch_id] = list(csv.DictReader(handle, strict=True))
    target = next(row for rows in rows_by_batch.values() for row in rows)
    target_assignment_ids = {
        row["assignment_id"]
        for rows in rows_by_batch.values()
        for row in rows
        if row["candidate_id"] == target["candidate_id"]
    }
    forged_input_sha256 = "f" * 64
    for rows in rows_by_batch.values():
        for row in rows:
            if row["assignment_id"] in target_assignment_ids:
                row["input_sha256"] = forged_input_sha256

    manifest = [dict(row) for row in coordinator.manifest]
    for row in manifest:
        if row["assignment_id"] in target_assignment_ids:
            row["input_sha256"] = forged_input_sha256
    forged_coordinator = replace(coordinator, manifest=tuple(manifest))
    payloads_by_batch = {
        batch_id: screening_results._csv_bytes(
            screening_results.RESULT_HEADER,
            rows_by_batch[batch_id],
        )
        for batch_id in screening_results.BATCH_IDS
    }
    rows, phase_manifest, snapshot_sha256, coordinator_hash, protocol_hash = (
        screening_results._validate_phase_payloads(
            forged_coordinator,
            source.phase,
            payloads_by_batch,
            reviewer_release_sha256=source.reviewer_release_sha256,
        )
    )
    artifacts = {
        f"{batch_id}.csv": payloads_by_batch[batch_id]
        for batch_id in screening_results.BATCH_IDS
    }
    artifacts["manifest.csv"] = screening_results._csv_bytes(
        screening_results.PHASE_RESULT_MANIFEST_HEADER,
        phase_manifest,
    )
    artifacts["SHA256SUMS"] = screening_results._checksums(artifacts)
    output_dir.mkdir(parents=True)
    output_dir.chmod(0o755)
    for name, payload in artifacts.items():
        path = output_dir / name
        path.write_bytes(payload)
        path.chmod(0o644)
    _, fingerprints = screening_results.capture_flat_snapshot(
        output_dir,
        (*screening_results.RESULT_FILENAMES, "manifest.csv", "SHA256SUMS"),
    )
    return forged_coordinator, screening_results.PhaseResultSnapshot(
        directory=output_dir,
        phase=source.phase,
        rows=tuple(rows),
        snapshot_sha256=snapshot_sha256,
        coordinator_snapshot_sha256=coordinator_hash,
        protocol_sha256=protocol_hash,
        reviewer_release_sha256=source.reviewer_release_sha256,
        manifest=tuple(phase_manifest),
        fingerprints=fingerprints,
    )


def test_public_agreement_rejects_nested_coordinator_forgery(
    sealed_snapshots: _SealedAgreementSnapshots,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged_coordinator, forged_main = _forge_main_input_binding(
        sealed_snapshots.main,
        screening_results.capture_coordinator_snapshot(
            sealed_snapshots.coordinator
        ),
        tmp_path / "forged-main" / "v1",
    )
    arguments = _public_arguments(sealed_snapshots)
    arguments["coordinator_snapshot"] = forged_coordinator
    arguments["main"] = forged_main
    monkeypatch.setattr(agreement, "PRODUCTION_BOOTSTRAP_REPLICATES", 1)

    with pytest.raises(
        agreement.ScreeningAgreementError,
        match="coordinator|authoritative revalidation",
    ):
        agreement.build_agreement_report(**arguments)

