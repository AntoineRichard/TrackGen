from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import shutil
import re
import stat
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

import paper.scripts.prepare_screening_batches as screening_batches
import paper.scripts.screening_results as screening_results


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "paper" / "data"
PROTOCOL_PATH = DATA_ROOT / "screening_protocol.md"
EXECUTION_PROFILE_PATH = DATA_ROOT / "screening_execution_profile.json"
REVIEWER_PROMPT_TEMPLATE_PATH = (
    DATA_ROOT / "screening_reviewer_prompt.md"
)
SCRIPT_PATH = (
    REPOSITORY_ROOT / "paper" / "scripts" / "prepare_screening_batches.py"
)


def _exception_text(error: BaseException) -> str:
    return "\n".join((str(error), *getattr(error, "__notes__", ())))


def test_cleanup_source_has_no_destructive_or_overwriting_namespace_calls():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = []
    forbidden_qualified = {
        ("os", "remove"),
        ("os", "rename"),
        ("os", "replace"),
        ("os", "rmdir"),
        ("os", "unlink"),
        ("shutil", "rmtree"),
    }
    forbidden_native = {"remove", "rename", "rmdir", "unlink", "unlinkat"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (function.value.id, function.attr) in forbidden_qualified
        ):
            forbidden.append((node.lineno, f"{function.value.id}.{function.attr}"))
        if (
            isinstance(function, ast.Attribute)
            and function.attr in {"rename", "rmdir", "rmtree", "unlink"}
        ):
            forbidden.append((node.lineno, function.attr))
        if (
            isinstance(function, ast.Name)
            and function.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in forbidden_native
        ):
            forbidden.append((node.lineno, str(node.args[1].value)))
    assert forbidden == []

CANDIDATE_HEADER = (
    "candidate_id",
    "cite_key",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
    "discovery_stream",
    "discovery_query",
    "discovery_agent",
    "screening_status",
    "exclusion_reason",
    "metadata_status",
    "metadata_evidence",
)
CONFLICT_HEADER = (
    "conflict_id",
    "record_type",
    "record_key",
    "field",
    "value_a",
    "value_b",
    "resolution",
    "resolver",
    "resolution_evidence",
)
BIBLIOGRAPHY_HEADER = (
    "candidate_id",
    "cite_key",
    "entry_type",
    "key_author",
    "authors",
    "author_kinds",
    "title",
    "year",
    "venue_field",
    "venue",
    "doi",
    "url",
)
CITATION_KEY_HEADER = ("candidate_id", "cite_key")
CALIBRATION_SELECTION_HEADER = ("candidate_id",)
MANIFEST_HEADER = (
    "manifest_version",
    "snapshot_sha256",
    "protocol_sha256",
    "execution_profile_sha256",
    "prompt_template_sha256",
    "assignment_id",
    "batch_id",
    "phase",
    "candidate_id",
    "cite_key",
    "input_sha256",
    "weight",
)
PACKET_HEADER = (
    "assignment_id",
    "candidate_id",
    "input_sha256",
    "snapshot_sha256",
    "batch_id",
    "phase",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
RELEASE_MANIFEST_HEADER = (
    "manifest_version",
    "phase",
    "coordinator_snapshot_sha256",
    "protocol_sha256",
    "execution_profile_sha256",
    "prompt_template_sha256",
    "assignment_count",
    "calibration_result_snapshot_sha256",
    "calibration_decision_snapshot_sha256",
)
RAW_FILENAMES = (
    "candidates.csv",
    "conflicts.csv",
    "bibliography.csv",
    "citation_keys.csv",
    "taxonomy.json",
    "protocol.md",
    "execution_profile.json",
    "reviewer_prompt_template.md",
)
RANKING_SALT = "trackgen-screening-calibration-v1"
REVIEWER_PAIRS = (
    ("screening-01", "screening-02"),
    ("screening-01", "screening-03"),
    ("screening-01", "screening-04"),
    ("screening-02", "screening-05"),
    ("screening-02", "screening-06"),
    ("screening-03", "screening-05"),
    ("screening-04", "screening-06"),
    ("screening-01", "screening-05"),
    ("screening-01", "screening-06"),
    ("screening-02", "screening-03"),
    ("screening-02", "screening-04"),
    ("screening-03", "screening-04"),
    ("screening-03", "screening-06"),
    ("screening-04", "screening-05"),
    ("screening-05", "screening-06"),
)

EVIDENCE_PACKET_HEADER = (
    "candidate_id",
    "artifact_id",
    "artifact_role",
    "source_url",
    "evidence_version",
    "evidence_retrieved_on",
    "access_status",
    "evidence_archive_url",
    "evidence_sha256",
    "local_filename",
    "redistribution_status",
    "retrieval_notes",
)


def _execution_profile() -> dict[str, object]:
    return {
        "decoding_parameters": None,
        "developer_instruction": None,
        "model_identifier": "gpt-5.6-sol",
        "model_version": "requested:gpt-5.6-sol",
        "profile_version": "1",
        "provider": "openai",
        "provider_metadata_limitations": {
            "backend_model_version": "provider-not-exposed",
            "decoding_parameters": "provider-not-exposed",
            "developer_instruction_bytes": "provider-not-exposed",
            "retrieval_cache_isolation": "provider-not-exposed",
            "system_instruction_bytes": "provider-not-exposed",
        },
        "retrieval_configuration": {
            "fresh_context": True,
            "provider_retrieval_cache_isolation": "provider-not-exposed",
            "public_retrieval_only": True,
            "ratings_supplied": False,
            "results_supplied": False,
            "shared_conversation_history": False,
            "shared_memory": False,
        },
        "runtime": "codex-subagent-spawn",
        "system_instruction": None,
        "tool_configuration": {
            "filesystem_policy": "immutable-stage-read-role-result-write",
            "fork_context": False,
            "host_security_boundary": (
                "shared-same-user-host-no-acl-container-mount-guarantee"
            ),
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "staging_isolation": "procedural-role-private-path",
            "web_retrieval_policy": "public-only",
        },
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fresh_taxonomy(path: Path) -> Path:
    taxonomy = json.loads((DATA_ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    taxonomy["screening_inclusion_criterion"] = ["include-relevant"]
    taxonomy["screening_result_status"] = ["included", "excluded"]
    path.write_bytes(_canonical_json_bytes(taxonomy))
    return path


@dataclass(frozen=True)
class Inputs:
    root: Path
    candidates: Path
    conflicts: Path
    bibliography: Path
    citation_keys: Path
    taxonomy: Path
    protocol: Path
    execution_profile: Path
    reviewer_prompt_template: Path


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[dict[str, str]],
    *,
    lineterminator: str = "\n",
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=header,
            lineterminator=lineterminator,
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, strict=True))


def _candidate(index: int, *, issued: bool = True) -> dict[str, str]:
    candidate_id = f"C{index:04d}"
    software = index == 2
    cite_key = f"Researcher2024Track{index}" if issued else ""
    return {
        "candidate_id": candidate_id,
        "cite_key": cite_key,
        "title": (
            "Track caf\u00e9 generator" if index == 1 else f"Track source {index}"
        ),
        "authors": "Example Project" if software else "Ada Researcher",
        "year": "" if software else "2024",
        "venue": "Official documentation" if software else "Robotics Journal",
        "doi": "" if software else f"10.1234/track.{index}",
        "url": f"https://example.test/sources/{candidate_id}",
        "source_type": "software documentation" if software else "paper",
        "discovery_stream": "blind-ground",
        "discovery_query": f"private discovery query {index}",
        "discovery_agent": "private-agent",
        "screening_status": "candidate" if issued else "excluded",
        "exclusion_reason": "" if issued else "legacy out-of-scope decision",
        "metadata_status": "verified",
        "metadata_evidence": f"official::https://example.test/{candidate_id}",
    }


def _bibliography_row(candidate: dict[str, str]) -> dict[str, str]:
    software = candidate["source_type"] == "software documentation"
    return {
        "candidate_id": candidate["candidate_id"],
        "cite_key": candidate["cite_key"],
        "entry_type": "misc" if software else "article",
        "key_author": "ExampleProject" if software else "Researcher",
        "authors": candidate["authors"],
        "author_kinds": "corporate" if software else "personal",
        "title": candidate["title"],
        "year": candidate["year"],
        "venue_field": "howpublished" if software else "journal",
        "venue": candidate["venue"],
        "doi": candidate["doi"],
        "url": candidate["url"],
    }


def build_inputs(
    root: Path,
    *,
    count: int = 202,
    include_conflict: bool = True,
) -> Inputs:
    root.mkdir()
    candidates = [
        _candidate(index, issued=index != count)
        for index in range(1, count + 1)
    ]
    bibliography = [
        _bibliography_row(row) for row in candidates if row["cite_key"]
    ]
    bibliography.sort(
        key=lambda row: (
            row["cite_key"].casefold(),
            row["cite_key"],
            row["candidate_id"],
        )
    )
    citation_keys = [
        {"candidate_id": row["candidate_id"], "cite_key": row["cite_key"]}
        for row in candidates
        if row["cite_key"]
    ]
    conflicts = []
    if include_conflict:
        conflicts.append(
            {
                "conflict_id": "CF0001",
                "record_type": "candidate",
                "record_key": "C0001",
                "field": "title",
                "value_a": "Track cafe generator",
                "value_b": "Track caf\u00e9 generator",
                "resolution": "Track caf\u00e9 generator",
                "resolver": "metadata-integrator",
                "resolution_evidence": "publisher::https://example.test/C0001",
            }
        )

    paths = Inputs(
        root=root,
        candidates=root / "candidates-source.csv",
        conflicts=root / "conflicts-source.csv",
        bibliography=root / "bibliography-source.csv",
        citation_keys=root / "citation-keys-source.csv",
        taxonomy=root / "taxonomy-source.json",
        protocol=root / "protocol-source.md",
        execution_profile=root / "execution-profile-source.json",
        reviewer_prompt_template=(
            root / "reviewer-prompt-template-source.md"
        ),
    )
    _write_csv(paths.candidates, CANDIDATE_HEADER, candidates)
    _write_csv(paths.conflicts, CONFLICT_HEADER, conflicts)
    _write_csv(paths.bibliography, BIBLIOGRAPHY_HEADER, bibliography)
    _write_csv(paths.citation_keys, CITATION_KEY_HEADER, citation_keys)
    taxonomy = json.loads((DATA_ROOT / "taxonomy.json").read_text(encoding="utf-8"))
    taxonomy["screening_inclusion_criterion"] = ["include-relevant"]
    taxonomy["screening_result_status"] = ["included", "excluded"]
    paths.taxonomy.write_bytes(_canonical_json_bytes(taxonomy))
    paths.protocol.write_bytes(PROTOCOL_PATH.read_bytes())
    paths.execution_profile.write_bytes(
        _canonical_json_bytes(_execution_profile())
    )
    paths.reviewer_prompt_template.write_bytes(
        REVIEWER_PROMPT_TEMPLATE_PATH.read_bytes()
    )
    return paths


def freeze_arguments(inputs: Inputs, output: Path) -> list[str]:
    return [
        "--freeze",
        "--candidates",
        str(inputs.candidates),
        "--conflicts",
        str(inputs.conflicts),
        "--bibliography",
        str(inputs.bibliography),
        "--citation-keys",
        str(inputs.citation_keys),
        "--taxonomy",
        str(inputs.taxonomy),
        "--protocol",
        str(inputs.protocol),
        "--output-dir",
        str(output),
        "--execution-profile",
        str(inputs.execution_profile),
        "--reviewer-prompt-template",
        str(inputs.reviewer_prompt_template),
    ]


def freeze(inputs: Inputs, output: Path) -> None:
    assert screening_batches.main(freeze_arguments(inputs, output)) == 0


def release_arguments(
    snapshot: Path,
    phase: str,
    output: Path,
    calibration_gate: tuple[Path, Path, Path] | None = None,
) -> list[str]:
    arguments = [
        "--release",
        "--snapshot-dir",
        str(snapshot),
        "--phase",
        phase,
        "--output-dir",
        str(output),
    ]
    if calibration_gate is not None:
        release_snapshot, result_snapshot, decision_snapshot = calibration_gate
        arguments.extend(
            [
                "--calibration-reviewer-release-snapshot",
                str(release_snapshot),
                "--calibration-result-snapshot",
                str(result_snapshot),
                "--calibration-decision-snapshot",
                str(decision_snapshot),
            ]
        )
    return arguments


def release(
    snapshot: Path,
    phase: str,
    output: Path,
    calibration_gate: tuple[Path, Path, Path] | None = None,
) -> None:
    assert screening_batches.main(
        release_arguments(snapshot, phase, output, calibration_gate)
    ) == 0


def _snapshot_files(snapshot: Path) -> list[Path]:
    return sorted(path for path in snapshot.rglob("*") if path.is_file())


def _persistent_state(path: Path) -> tuple[bytes, int, int, int]:
    file_stat = path.stat()
    return (
        path.read_bytes(),
        stat.S_IMODE(file_stat.st_mode),
        file_stat.st_ino,
        file_stat.st_mtime_ns,
    )


def _snapshot_state(snapshot: Path) -> dict[str, tuple[bytes, int, int, int]]:
    return {
        path.relative_to(snapshot).as_posix(): _persistent_state(path)
        for path in _snapshot_files(snapshot)
    }


def _manifest_by_id(snapshot: Path) -> dict[str, dict[str, str]]:
    return {
        row["candidate_id"]: row
        for row in _read_csv(snapshot / "manifest.csv")
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _screening_rank(candidate_id: str, protocol: bytes) -> tuple[str, bytes]:
    del protocol
    payload = (
        RANKING_SALT.encode("utf-8")
        + b"\0"
        + candidate_id.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest(), candidate_id.encode("utf-8")


def test_screening_protocol_is_available_as_frozen_input() -> None:
    assert PROTOCOL_PATH.is_file()
    assert PROTOCOL_PATH.read_bytes()


def test_execution_profile_validator_is_public_and_current_profile_is_canonical(
) -> None:
    assert hasattr(screening_batches, "validate_execution_profile")
    payload = EXECUTION_PROFILE_PATH.read_bytes()
    assert payload == _canonical_json_bytes(_execution_profile())
    assert screening_batches.validate_execution_profile(payload) == (
        _execution_profile()
    )


@pytest.mark.parametrize(
    "mutation",
    ["noncanonical", "missing-limitation", "unjustified-limitation"],
)
def test_execution_profile_validator_rejects_invalid_contracts(
    mutation: str,
) -> None:
    profile = _execution_profile()
    if mutation == "noncanonical":
        payload = json.dumps(profile, indent=2).encode("utf-8")
    elif mutation == "missing-limitation":
        del profile["provider_metadata_limitations"][
            "system_instruction_bytes"
        ]
        payload = _canonical_json_bytes(profile)
    else:
        profile["system_instruction"] = "visible system instruction"
        payload = _canonical_json_bytes(profile)

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.validate_execution_profile(payload)


def test_screening_batch_cli_exists() -> None:
    assert SCRIPT_PATH.is_file()


def test_freeze_requires_and_binds_execution_profile_and_prompt_template(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    freeze(inputs, output)

    profile_payload = inputs.execution_profile.read_bytes()
    prompt_payload = inputs.reviewer_prompt_template.read_bytes()
    assert (output / "execution_profile.json").read_bytes() == profile_payload
    assert (
        output / "reviewer_prompt_template.md"
    ).read_bytes() == prompt_payload
    manifest = _read_csv(output / "manifest.csv")
    assert {
        row["execution_profile_sha256"] for row in manifest
    } == {hashlib.sha256(profile_payload).hexdigest()}
    assert {
        row["prompt_template_sha256"] for row in manifest
    } == {hashlib.sha256(prompt_payload).hexdigest()}

    missing_profile = freeze_arguments(inputs, tmp_path / "v2")
    profile_index = missing_profile.index("--execution-profile")
    del missing_profile[profile_index : profile_index + 2]
    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.main(missing_profile)


def test_allowed_inclusion_public_role_staging_is_single_packet_and_derivation_bound(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator = tmp_path / "coordinator" / "v1"
    coordinator.parent.mkdir()
    freeze(inputs, coordinator)
    reviewer_release = tmp_path / "releases" / "v1"
    reviewer_release.parent.mkdir()
    release(coordinator, "calibration", reviewer_release)

    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)
    role_id = "screening-01"
    stage = screening_batches.stage_reviewer_execution(
        coordinator,
        reviewer_release,
        role_id,
        staging_root,
    )

    assert stage.name == "v1"
    assert stage.parent.parent == staging_root
    assert re.fullmatch(
        rf"{role_id}-[0-9a-f]{{32}}",
        stage.parent.name,
    )
    assert stat.S_IMODE(stage.parent.stat().st_mode) == 0o700
    assert {path.name for path in stage.iterdir()} == (
        screening_batches.REVIEWER_STAGE_ROOT_FILENAMES
    )
    assert {path.name for path in stage.iterdir()} == {
        "execution_configuration.json",
        "execution_profile.json",
        "packet.csv",
        "protocol.md",
        "reviewer_prompt.md",
        "reviewer_prompt_template.md",
        "stage_manifest.csv",
        "SHA256SUMS",
    }
    assert all(path.is_file() for path in stage.iterdir())
    assert not any("screening-02" in path.name for path in stage.rglob("*"))
    assert (stage / "packet.csv").read_bytes() == (
        reviewer_release / "packets" / f"{role_id}.csv"
    ).read_bytes()

    validated = screening_batches.validate_reviewer_stage_snapshot(stage)
    assert set(validated) == {path.name for path in stage.iterdir()}

    stage_manifest = _read_csv(stage / "stage_manifest.csv")
    assert len(stage_manifest) == 1
    manifest = stage_manifest[0]
    assert tuple(manifest) == screening_batches.STAGE_MANIFEST_HEADER
    packet_sha256 = hashlib.sha256(
        (stage / "packet.csv").read_bytes()
    ).hexdigest()
    protocol_sha256 = hashlib.sha256(
        (stage / "protocol.md").read_bytes()
    ).hexdigest()
    template_sha256 = hashlib.sha256(
        (stage / "reviewer_prompt_template.md").read_bytes()
    ).hexdigest()
    profile_sha256 = hashlib.sha256(
        (stage / "execution_profile.json").read_bytes()
    ).hexdigest()
    prompt_sha256 = hashlib.sha256(
        (stage / "reviewer_prompt.md").read_bytes()
    ).hexdigest()
    assert manifest["packet_sha256"] == packet_sha256
    assert manifest["protocol_sha256"] == protocol_sha256
    assert manifest["prompt_template_sha256"] == template_sha256
    assert manifest["execution_profile_sha256"] == profile_sha256
    assert manifest["prompt_sha256"] == prompt_sha256
    assert manifest["user_instruction_sha256"] == prompt_sha256
    assert manifest["assignment_count"] == str(
        len(_read_csv(stage / "packet.csv"))
    )
    assert manifest["stage_path"] == str(stage)
    assert manifest["result_path"] == str(
        stage.parent / f"{role_id}-result.csv"
    )

    profile = json.loads(
        (stage / "execution_profile.json").read_text(encoding="utf-8")
    )
    configuration = json.loads(
        (stage / "execution_configuration.json").read_text(
            encoding="utf-8"
        )
    )
    coordinator_manifest = _read_csv(coordinator / "manifest.csv")[0]
    assert (stage / "execution_configuration.json").read_bytes() == (
        _canonical_json_bytes(configuration)
    )
    assert configuration == {
        "allowed_inclusion_criteria": ["include-relevant"],
        "configuration_version": "2",
        "coordinator_snapshot_sha256": coordinator_manifest[
            "snapshot_sha256"
        ],
        "execution_profile": profile,
        "execution_profile_sha256": profile_sha256,
        "packet": {
            "path": str(stage / "packet.csv"),
            "sha256": packet_sha256,
        },
        "phase": "calibration",
        "prompt": {
            "path": str(stage / "reviewer_prompt.md"),
            "template_path": str(
                stage / "reviewer_prompt_template.md"
            ),
            "template_sha256": template_sha256,
        },
        "protocol": {
            "path": str(stage / "protocol.md"),
            "sha256": protocol_sha256,
        },
        "result": {"path": str(stage.parent / f"{role_id}-result.csv")},
        "reviewer_release_sha256": manifest[
            "reviewer_release_sha256"
        ],
        "role_id": role_id,
        "stage_path": str(stage),
        "task": "calibration-screening",
        "user_instruction_delivery": (
            "exact-rendered-visible-prompt-bytes"
        ),
        "work_item_scope": "one-role-packet",
    }
    assert configuration["execution_profile"]["tool_configuration"] == {
        "filesystem_policy": "immutable-stage-read-role-result-write",
        "fork_context": False,
        "host_security_boundary": (
            "shared-same-user-host-no-acl-container-mount-guarantee"
        ),
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "staging_isolation": "procedural-role-private-path",
        "web_retrieval_policy": "public-only",
    }

    expected_prompt = (
        stage / "reviewer_prompt_template.md"
    ).read_text(encoding="utf-8")
    for placeholder, value in (
        ("ROLE_ID", role_id),
        ("STAGE_PATH", str(stage)),
        ("PROTOCOL_PATH", str(stage / "protocol.md")),
        ("PROTOCOL_SHA256", protocol_sha256),
        ("PACKET_PATH", str(stage / "packet.csv")),
        ("PACKET_SHA256", packet_sha256),
        ("OUTPUT_PATH", str(stage.parent / f"{role_id}-result.csv")),
    ):
        expected_prompt = expected_prompt.replace(
            "{{" + placeholder + "}}",
            value,
        )
    rendered_prompt = (stage / "reviewer_prompt.md").read_text(
        encoding="utf-8"
    )
    assert rendered_prompt == expected_prompt
    assert "{{STAGE_PATH}}" not in rendered_prompt
    assert str(stage) in rendered_prompt
    assert str(stage.parent / f"{role_id}-result.csv") in rendered_prompt
    assert "{{ROWS_WRITTEN}}" in rendered_prompt
    assert "{{OUTPUT_SHA256}}" in rendered_prompt


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda configuration: configuration.update(
                {"configuration_version": "3"}
            ),
            "execution configuration version is invalid",
        ),
        (
            lambda configuration: configuration.update(
                {
                    "configuration_version": "2",
                    "allowed_inclusion_criteria": ["include-1"],
                }
            ),
            "allowed inclusion criteria are invalid",
        ),
    ],
)
def test_allowed_inclusion_stage_validation_rejects_unknown_or_invalid_v2_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator = tmp_path / "coordinator" / "v1"
    coordinator.parent.mkdir()
    freeze(inputs, coordinator)
    reviewer_release = tmp_path / "releases" / "v1"
    reviewer_release.parent.mkdir()
    release(coordinator, "calibration", reviewer_release)
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)
    private_parent = staging_root / ("screening-01-" + "a" * 32)
    private_parent.mkdir(mode=0o700)
    os.chmod(private_parent, 0o700)
    stage = private_parent / "v1"
    release_payloads = {
        relative: (reviewer_release / relative).read_bytes()
        for relative in {
            "execution_profile.json",
            "protocol.md",
            "release_manifest.csv",
            "reviewer_prompt_template.md",
            "SHA256SUMS",
            *(
                f"packets/{filename}"
                for filename in screening_batches.PACKET_FILENAMES
            ),
        }
    }
    build_configuration = screening_batches._execution_configuration

    def build_invalid_configuration(**kwargs):
        configuration = build_configuration(**kwargs)
        mutation(configuration)
        return configuration

    monkeypatch.setattr(
        screening_batches,
        "_execution_configuration",
        build_invalid_configuration,
    )
    artifacts = screening_batches.build_reviewer_stage_artifacts(
        release_payloads,
        "screening-01",
        stage,
    )
    monkeypatch.undo()
    screening_batches.publish_snapshot(stage, artifacts)

    with pytest.raises(screening_batches.SnapshotError, match=message):
        screening_batches.validate_reviewer_stage_snapshot(stage)


def test_stage_cli_publishes_and_reports_random_role_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    coordinator, reviewer_release, _, source_archive = _v2_calibration_release(
        tmp_path
    )
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)

    assert screening_batches.main(
        [
            "--stage-role",
            "--snapshot-dir",
            str(coordinator),
            "--reviewer-release-snapshot",
            str(reviewer_release),
            "--role-id",
            "screening-02",
            "--staging-root",
            str(staging_root),
            "--source-archive",
            str(source_archive),
        ]
    ) == 0
    output = Path(capsys.readouterr().out.strip())
    assert output.parent.parent == staging_root
    assert output.name == "v1"
    screening_batches.validate_reviewer_stage_snapshot(output)


def _v2_calibration_release(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator = tmp_path / "coordinator" / "v7"
    coordinator.parent.mkdir()
    freeze(inputs, coordinator)
    evidence_manifest, source_archive = _phase_evidence_inputs(
        tmp_path, coordinator, "calibration"
    )
    reviewer_release = tmp_path / "releases" / "v2"
    reviewer_release.parent.mkdir()
    screening_batches.release_snapshot(
        coordinator,
        "calibration",
        reviewer_release,
        evidence_manifest=evidence_manifest,
        source_archive=source_archive,
    )
    return coordinator, reviewer_release, evidence_manifest, source_archive


def test_stage_role_cli_forwards_source_archive_for_v2_release(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    coordinator, reviewer_release, _, source_archive = _v2_calibration_release(
        tmp_path
    )
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)

    assert screening_batches.main(
        [
            "--stage-role",
            "--snapshot-dir",
            str(coordinator),
            "--reviewer-release-snapshot",
            str(reviewer_release),
            "--role-id",
            "screening-02",
            "--staging-root",
            str(staging_root),
            "--source-archive",
            str(source_archive),
        ]
    ) == 0

    stage = Path(capsys.readouterr().out.strip())
    screening_batches.validate_reviewer_stage_snapshot(stage)



def test_configuration_v3_stages_role_filtered_evidence_and_exact_bytes(
    tmp_path: Path,
) -> None:
    coordinator, reviewer_release, _, source_archive = _v2_calibration_release(
        tmp_path
    )
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)

    stage = screening_batches.stage_reviewer_execution(
        coordinator,
        reviewer_release,
        "screening-01",
        staging_root,
        source_archive=source_archive,
    )

    configuration = json.loads(
        (stage / "execution_configuration.json").read_text(encoding="utf-8")
    )
    assert configuration["configuration_version"] == "3"
    assert configuration["allowed_screening_statuses"] == ["included", "excluded"]
    assert configuration["allowed_inclusion_criteria"] == ["include-relevant"]

    packet_candidate_ids = {
        row["candidate_id"] for row in _read_csv(stage / "packet.csv")
    }
    staged_rows = _read_csv(stage / "evidence_packet_manifest.csv")
    release_rows = _read_csv(reviewer_release / "evidence_packet_manifest.csv")
    expected_rows = [
        row for row in release_rows if row["candidate_id"] in packet_candidate_ids
    ]
    assert staged_rows == expected_rows
    assert {
        row["candidate_id"] for row in staged_rows
    } == packet_candidate_ids
    assert configuration["evidence_packet_manifest_sha256"] == hashlib.sha256(
        (stage / "evidence_packet_manifest.csv").read_bytes()
    ).hexdigest()

    for row in staged_rows:
        destination = (
            stage
            / "evidence"
            / row["candidate_id"]
            / row["artifact_id"]
            / Path(row["local_filename"]).name
        )
        assert destination.read_bytes() == (
            source_archive / row["local_filename"]
        ).read_bytes()
    assert {
        path.relative_to(stage / "evidence").parts[0]
        for path in (stage / "evidence").rglob("*")
        if path.is_file()
    } == packet_candidate_ids


def test_configuration_v3_all_metadata_only_stage_has_no_evidence_directory(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator = tmp_path / "coordinator" / "v7"
    coordinator.parent.mkdir()
    freeze(inputs, coordinator)
    evidence_manifest, source_archive = _phase_evidence_inputs(
        tmp_path, coordinator, "calibration"
    )
    role_candidate_ids = {
        row["candidate_id"]
        for row in _read_csv(coordinator / "packets" / "screening-01.csv")
        if row["phase"] == "calibration"
    }
    evidence_rows = _read_csv(evidence_manifest)
    for row in evidence_rows:
        if row["candidate_id"] not in role_candidate_ids:
            continue
        row["evidence_sha256"] = "NR"
        row["local_filename"] = "NR"
        row["redistribution_status"] = "metadata-only"
        row["retrieval_notes"] = (
            "attempted: doi_or_publisher=publisher supplied metadata only | "
            "title_author=title and author search found no copy | "
            "scholarly_index_or_repository=index search found no accessible copy | "
            "official_page=official page supplied metadata only; "
            "outcome: no local evidence bytes were available"
        )
    evidence_manifest.write_bytes(_evidence_packet_bytes(evidence_rows))

    reviewer_release = tmp_path / "releases" / "v2"
    reviewer_release.parent.mkdir()
    screening_batches.release_snapshot(
        coordinator,
        "calibration",
        reviewer_release,
        evidence_manifest=evidence_manifest,
        source_archive=source_archive,
    )
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)
    stage = screening_batches.stage_reviewer_execution(
        coordinator,
        reviewer_release,
        "screening-01",
        staging_root,
        source_archive=source_archive,
    )

    configuration = json.loads(
        (stage / "execution_configuration.json").read_text(encoding="utf-8")
    )
    staged_manifest = stage / "evidence_packet_manifest.csv"
    staged_rows = _read_csv(staged_manifest)
    assert configuration["configuration_version"] == "3"
    assert configuration["allowed_screening_statuses"] == ["included", "excluded"]
    assert configuration["allowed_inclusion_criteria"] == ["include-relevant"]
    assert configuration["evidence_packet_manifest_sha256"] == hashlib.sha256(
        staged_manifest.read_bytes()
    ).hexdigest()
    assert {row["candidate_id"] for row in staged_rows} == role_candidate_ids
    assert {row["redistribution_status"] for row in staged_rows} == {
        "metadata-only"
    }
    assert not (stage / "evidence").exists()

    screening_batches.validate_reviewer_stage_snapshot(stage)


@pytest.mark.parametrize("mutation", ("missing", "extra", "swapped", "mutated"))
def test_staged_evidence_validation_rejects_tree_mutation(
    tmp_path: Path, mutation: str
) -> None:
    coordinator, reviewer_release, _, source_archive = _v2_calibration_release(
        tmp_path
    )
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)
    stage = screening_batches.stage_reviewer_execution(
        coordinator,
        reviewer_release,
        "screening-01",
        staging_root,
        source_archive=source_archive,
    )
    evidence_files = [
        path for path in sorted((stage / "evidence").rglob("*")) if path.is_file()
    ]
    assert len(evidence_files) >= 2
    if mutation == "missing":
        evidence_files[0].unlink()
    elif mutation == "extra":
        (stage / "evidence" / "unexpected.bin").write_bytes(b"unexpected\\n")
    elif mutation == "swapped":
        evidence_files[0].write_bytes(evidence_files[1].read_bytes())
    else:
        evidence_files[0].write_bytes(b"mutated\\n")

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.validate_reviewer_stage_snapshot(stage)

def test_stage_role_cli_v2_release_requires_source_archive(
    tmp_path: Path,
) -> None:
    coordinator, reviewer_release, _, _ = _v2_calibration_release(tmp_path)
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)

    with pytest.raises(
        screening_batches.SnapshotError,
        match="v2 reviewer staging requires a source archive",
    ):
        screening_batches.main(
            [
                "--stage-role",
                "--snapshot-dir",
                str(coordinator),
                "--reviewer-release-snapshot",
                str(reviewer_release),
                "--role-id",
                "screening-02",
                "--staging-root",
                str(staging_root),
            ]
        )


def test_stage_role_cli_historical_v1_works_without_source_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)

    assert screening_batches.main(
        [
            "--stage-role",
            "--snapshot-dir",
            str(DATA_ROOT / "screening_inputs" / "v5"),
            "--reviewer-release-snapshot",
            str(DATA_ROOT / "screening_releases" / "calibration" / "v5"),
            "--role-id",
            "screening-02",
            "--staging-root",
            str(staging_root),
        ]
    ) == 0
    screening_batches.validate_reviewer_stage_snapshot(
        Path(capsys.readouterr().out.strip())
    )


def test_stage_role_cli_rejects_evidence_manifest(
    tmp_path: Path,
) -> None:
    coordinator, reviewer_release, evidence_manifest, source_archive = (
        _v2_calibration_release(tmp_path)
    )

    with pytest.raises(
        screening_batches.SnapshotError,
        match="--stage-role does not accept --evidence-manifest",
    ):
        screening_batches.main(
            [
                "--stage-role",
                "--snapshot-dir",
                str(coordinator),
                "--reviewer-release-snapshot",
                str(reviewer_release),
                "--role-id",
                "screening-02",
                "--staging-root",
                str(tmp_path / "staging"),
                "--source-archive",
                str(source_archive),
                "--evidence-manifest",
                str(evidence_manifest),
            ]
        )


@pytest.mark.parametrize(
    "snapshot_path",
    [
        "paper/data/screening_inputs/v1/candidates.csv",
        "paper/data/screening_releases/calibration/v1/manifest.csv",
        "paper/data/screening_runs/calibration/v1/screening-01.csv",
        "paper/data/screening_results/calibration/v1/manifest.csv",
        "paper/data/screening_decisions/calibration/v1/decision.csv",
        "paper/data/screening_adjudication_inputs/v1/adjudication.csv",
        "paper/data/screening_adjudication/v1/adjudication.csv",
        "paper/data/screening_projection/v1/candidates.csv",
    ],
)
def test_git_treats_screening_snapshots_as_binary(
    snapshot_path: str,
) -> None:
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
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        f"{snapshot_path}: diff: unset",
        f"{snapshot_path}: merge: unset",
        f"{snapshot_path}: text: unset",
    ]


def test_snapshot_hash_preimage_binds_calibration_selection_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    captured: list[dict[str, object]] = []
    canonical_sha256 = screening_batches._canonical_sha256

    def record_preimage(value: object) -> str:
        if (
            isinstance(value, dict)
            and "assignments" in value
            and "raw_files" in value
        ):
            captured.append(value)
        return canonical_sha256(value)

    monkeypatch.setattr(
        screening_batches,
        "_canonical_sha256",
        record_preimage,
    )
    freeze(inputs, tmp_path / "v1")

    assert len(captured) == 1
    preimage = captured[0]
    assert set(preimage) == {
        "assignments",
        "calibration_selection",
        "manifest_version",
        "raw_files",
    }
    selection = preimage["calibration_selection"]
    assignments = preimage["assignments"]
    assert isinstance(selection, list)
    assert len(selection) == 30
    assert isinstance(assignments, list)
    assert all(
        "calibration_selection" not in assignment
        for assignment in assignments
    )


def test_freeze_creates_complete_canonical_snapshot(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"

    freeze(inputs, snapshot)

    assert {path.name for path in snapshot.iterdir()} == {
        *RAW_FILENAMES,
        "calibration_selection.csv",
        "manifest.csv",
        "SHA256SUMS",
        "packets",
    }
    assert {path.name for path in (snapshot / "packets").iterdir()} == {
        f"screening-{number:02d}.csv" for number in range(1, 7)
    }
    for source, frozen_name in (
        (inputs.candidates, "candidates.csv"),
        (inputs.conflicts, "conflicts.csv"),
        (inputs.bibliography, "bibliography.csv"),
        (inputs.citation_keys, "citation_keys.csv"),
        (inputs.taxonomy, "taxonomy.json"),
        (inputs.protocol, "protocol.md"),
    ):
        assert (snapshot / frozen_name).read_bytes() == source.read_bytes()

    with (snapshot / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        assert tuple(reader.fieldnames or ()) == MANIFEST_HEADER
        manifest = list(reader)
    assert manifest == sorted(
        manifest,
        key=lambda row: (
            row["batch_id"],
            row["candidate_id"],
            row["assignment_id"],
        ),
    )
    assert len(manifest) == 404
    assert Counter(row["candidate_id"] for row in manifest) == Counter(
        {f"C{number:04d}": 2 for number in range(1, 203)}
    )
    assert len({row["assignment_id"] for row in manifest}) == len(manifest)
    assert all(
        row["assignment_id"]
        == f"A-{row['candidate_id']}-{row['batch_id'].removeprefix('screening-')}"
        for row in manifest
    )
    assert all(row["weight"] == "1" for row in manifest)
    assert Counter(row["phase"] for row in manifest) == {
        "calibration": 60,
        "main": 344,
    }
    assert len({row["snapshot_sha256"] for row in manifest}) == 1
    assert {
        row["protocol_sha256"] for row in manifest
    } == {hashlib.sha256(inputs.protocol.read_bytes()).hexdigest()}
    for candidate_id in {row["candidate_id"] for row in manifest}:
        rows = [row for row in manifest if row["candidate_id"] == candidate_id]
        assert len({row["batch_id"] for row in rows}) == 2
        assert len({row["input_sha256"] for row in rows}) == 1

    selection_path = snapshot / "calibration_selection.csv"
    with selection_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        assert tuple(reader.fieldnames or ()) == CALIBRATION_SELECTION_HEADER
        selection = list(reader)
    selected_ids = [row["candidate_id"] for row in selection]
    assert len(selected_ids) == 30
    assert len(set(selected_ids)) == 30
    assert selected_ids == sorted(
        selected_ids,
        key=lambda candidate_id: _screening_rank(
            candidate_id, inputs.protocol.read_bytes()
        ),
    )
    assert set(selected_ids) == {
        row["candidate_id"]
        for row in manifest
        if row["phase"] == "calibration"
    }

    ranked_ids = sorted(
        {row["candidate_id"] for row in manifest},
        key=lambda candidate_id: _screening_rank(
            candidate_id, inputs.protocol.read_bytes()
        ),
    )
    expected_pairs = {
        candidate_id: REVIEWER_PAIRS[index % len(REVIEWER_PAIRS)]
        for index, candidate_id in enumerate(ranked_ids)
    }
    actual_pairs = {
        candidate_id: tuple(
            sorted(
                row["batch_id"]
                for row in manifest
                if row["candidate_id"] == candidate_id
            )
        )
        for candidate_id in ranked_ids
    }
    assert actual_pairs == expected_pairs

    packet_assignments: set[str] = set()
    for number in range(1, 7):
        batch_id = f"screening-{number:02d}"
        packet_path = snapshot / "packets" / f"{batch_id}.csv"
        with packet_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            assert tuple(reader.fieldnames or ()) == PACKET_HEADER
            packet = list(reader)
        assert packet == sorted(
            packet,
            key=lambda row: (row["candidate_id"], row["assignment_id"]),
        )
        assert all(row["batch_id"] == batch_id for row in packet)
        assert all(value for row in packet for value in row.values())
        packet_assignments.update(row["assignment_id"] for row in packet)
    assert packet_assignments == {row["assignment_id"] for row in manifest}


def test_freeze_rejects_incomplete_candidate_corpus(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs", count=201)
    output = tmp_path / "v1"

    with pytest.raises(screening_batches.SnapshotError, match="exactly 202"):
        freeze(inputs, output)
    assert not os.path.lexists(output)


def test_packets_are_blinded_and_use_nr_for_missing_metadata(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)

    packets = [
        row
        for number in range(1, 7)
        for row in _read_csv(
            snapshot / "packets" / f"screening-{number:02d}.csv"
        )
    ]
    by_id = {row["candidate_id"]: row for row in packets}
    assert len(packets) == 404
    assert Counter(row["candidate_id"] for row in packets) == Counter(
        {f"C{number:04d}": 2 for number in range(1, 203)}
    )

    assert set(by_id) == {f"C{number:04d}" for number in range(1, 203)}
    assert tuple(by_id["C0001"]) == PACKET_HEADER
    assert by_id["C0002"]["year"] == "NR"
    assert by_id["C0002"]["doi"] == "NR"
    packet_bytes = b"".join(
        (snapshot / "packets" / f"screening-{number:02d}.csv").read_bytes()
        for number in range(1, 7)
    )
    for blinded_value in (
        b"private discovery query",
        b"private-agent",
        b"legacy out-of-scope decision",
        b"screening_status",
        b"exclusion_reason",
        b"discovery_stream",
        b"cite_key",
    ):
        assert blinded_value not in packet_bytes


def test_input_hash_uses_canonical_structured_complete_inputs(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)

    candidate = _read_csv(inputs.candidates)[0]
    conflict = _read_csv(inputs.conflicts)[0]
    bibliography = next(
        row
        for row in _read_csv(inputs.bibliography)
        if row["candidate_id"] == candidate["candidate_id"]
    )
    citation_key = next(
        row
        for row in _read_csv(inputs.citation_keys)
        if row["candidate_id"] == candidate["candidate_id"]
    )
    expected = _canonical_sha256(
        {
            "bibliography": bibliography,
            "candidate": candidate,
            "citation_key": citation_key,
            "conflicts": [conflict],
            "protocol_sha256": hashlib.sha256(
                inputs.protocol.read_bytes()
            ).hexdigest(),
            "taxonomy_sha256": hashlib.sha256(
                inputs.taxonomy.read_bytes()
            ).hexdigest(),
        }
    )

    assert _manifest_by_id(snapshot)["C0001"]["input_sha256"] == expected


def test_sha256sums_covers_every_other_file_in_lexical_order(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)

    covered = sorted(
        path.relative_to(snapshot).as_posix()
        for path in _snapshot_files(snapshot)
        if path.name != "SHA256SUMS"
    )
    expected = "".join(
        f"{hashlib.sha256((snapshot / relative).read_bytes()).hexdigest()}  "
        f"{relative}\n"
        for relative in covered
    ).encode("utf-8")
    assert (snapshot / "SHA256SUMS").read_bytes() == expected


@pytest.mark.parametrize(
    ("phase", "expected_count"),
    [("calibration", 60), ("main", 344)],
)
def test_reviewer_release_contains_only_phase_filtered_blinded_packets(
    tmp_path: Path,
    phase: str,
    expected_count: int,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator_root = tmp_path / "coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    release_root = tmp_path / "reviewer-releases"
    release_root.mkdir()
    output = release_root / "v1"

    gate = None
    if phase == "main":
        decision_root = tmp_path / "release-decisions"
        decision_root.mkdir()
        gate = _write_decision_snapshot(snapshot, decision_root / "v1")
    release(snapshot, phase, output, gate)

    assert {path.name for path in output.iterdir()} == {
        "protocol.md",
        "execution_profile.json",
        "reviewer_prompt_template.md",
        "release_manifest.csv",
        "SHA256SUMS",
        "packets",
    }
    assert {path.name for path in (output / "packets").iterdir()} == {
        f"screening-{number:02d}.csv" for number in range(1, 7)
    }
    assert (output / "protocol.md").read_bytes() == (
        snapshot / "protocol.md"
    ).read_bytes()
    assert (output / "execution_profile.json").read_bytes() == (
        snapshot / "execution_profile.json"
    ).read_bytes()
    assert (output / "reviewer_prompt_template.md").read_bytes() == (
        snapshot / "reviewer_prompt_template.md"
    ).read_bytes()

    coordinator_manifest = _read_csv(snapshot / "manifest.csv")[0]
    if gate is None:
        result_snapshot_sha256 = "NR"
        decision_snapshot_sha256 = "NR"
    else:
        _, result_snapshot, decision_snapshot = gate
        result_snapshot_sha256 = _read_csv(
            result_snapshot / "manifest.csv"
        )[0]["phase_result_snapshot_sha256"]
        decision_snapshot_sha256 = _read_csv(
            decision_snapshot / "manifest.csv"
        )[0]["calibration_decision_snapshot_sha256"]
    release_manifest = _read_csv(output / "release_manifest.csv")
    assert release_manifest == [
        {
            "manifest_version": screening_batches.MANIFEST_VERSION,
            "phase": phase,
            "coordinator_snapshot_sha256": coordinator_manifest[
                "snapshot_sha256"
            ],
            "protocol_sha256": coordinator_manifest["protocol_sha256"],
            "execution_profile_sha256": coordinator_manifest[
                "execution_profile_sha256"
            ],
            "prompt_template_sha256": coordinator_manifest[
                "prompt_template_sha256"
            ],
            "assignment_count": str(expected_count),
            "calibration_result_snapshot_sha256": result_snapshot_sha256,
            "calibration_decision_snapshot_sha256": (
                decision_snapshot_sha256
            ),
        }
    ]
    assert tuple(release_manifest[0]) == RELEASE_MANIFEST_HEADER

    released_rows: list[dict[str, str]] = []
    expected_rows: list[dict[str, str]] = []
    for number in range(1, 7):
        filename = f"screening-{number:02d}.csv"
        packet = _read_csv(output / "packets" / filename)
        assert all(tuple(row) == PACKET_HEADER for row in packet)
        assert all(row["phase"] == phase for row in packet)
        assert all("cite_key" not in row for row in packet)
        released_rows.extend(packet)
        expected_rows.extend(
            row
            for row in _read_csv(snapshot / "packets" / filename)
            if row["phase"] == phase
        )
    assert released_rows == expected_rows
    assert len(released_rows) == expected_count

    covered = sorted(
        path.relative_to(output).as_posix()
        for path in _snapshot_files(output)
        if path.name != "SHA256SUMS"
    )
    expected_sums = "".join(
        f"{hashlib.sha256((output / relative).read_bytes()).hexdigest()}  "
        f"{relative}\n"
        for relative in covered
    ).encode("utf-8")
    assert (output / "SHA256SUMS").read_bytes() == expected_sums


def test_main_release_post_publish_validation_rejects_wrong_gate_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "wrong-binding-inputs")
    coordinator_root = tmp_path / "wrong-binding-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    decision_root = tmp_path / "wrong-binding-decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(coordinator, decision_root / "v1")
    build = screening_batches.build_reviewer_release_artifacts

    def build_with_wrong_binding(*args, **kwargs):
        artifacts = build(*args, **kwargs)
        rows = screening_batches._read_csv_bytes(
            artifacts["release_manifest.csv"],
            "release_manifest.csv",
            RELEASE_MANIFEST_HEADER,
        )
        rows[0]["calibration_decision_snapshot_sha256"] = "0" * 64
        artifacts["release_manifest.csv"] = screening_batches._csv_bytes(
            RELEASE_MANIFEST_HEADER,
            rows,
        )
        checksum_inputs = {
            name: payload
            for name, payload in artifacts.items()
            if name != "SHA256SUMS"
        }
        artifacts["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(checksum_inputs[name]).hexdigest()}  {name}\n"
            for name in sorted(checksum_inputs)
        ).encode("utf-8")
        return artifacts

    monkeypatch.setattr(
        screening_batches,
        "build_reviewer_release_artifacts",
        build_with_wrong_binding,
    )
    release_root = tmp_path / "wrong-binding-releases"
    release_root.mkdir()
    output = release_root / "v1"

    with pytest.raises(
        screening_batches.SnapshotError,
        match="release manifest|binding",
    ):
        release(coordinator, "main", output, gate)
    assert not os.path.lexists(output)


@pytest.mark.parametrize(
    "field",
    [
        "calibration_result_snapshot_sha256",
        "calibration_decision_snapshot_sha256",
    ],
)
def test_release_validation_rejects_self_consistent_tampered_gate_binding(
    tmp_path: Path,
    field: str,
) -> None:
    inputs = build_inputs(tmp_path / f"tampered-{field}-inputs")
    coordinator_root = tmp_path / f"tampered-{field}-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    decision_root = tmp_path / f"tampered-{field}-decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(coordinator, decision_root / "v1")
    release_root = tmp_path / f"tampered-{field}-releases"
    release_root.mkdir()
    output = release_root / "v1"
    release(coordinator, "main", output, gate)

    expected_manifest = _read_csv(output / "release_manifest.csv")[0]
    tampered_manifest = dict(expected_manifest)
    tampered_manifest[field] = "0" * 64
    _write_csv(
        output / "release_manifest.csv",
        RELEASE_MANIFEST_HEADER,
        [tampered_manifest],
    )
    checksum_inputs = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in _snapshot_files(output)
        if path.name != "SHA256SUMS"
    }
    (output / "SHA256SUMS").write_bytes(
        "".join(
            f"{hashlib.sha256(checksum_inputs[name]).hexdigest()}  {name}\n"
            for name in sorted(checksum_inputs)
        ).encode("utf-8")
    )

    with pytest.raises(
        screening_batches.SnapshotError,
        match="release manifest|binding",
    ):
        screening_batches.validate_reviewer_release_snapshot(
            output,
            expected_manifest=expected_manifest,
            coordinator_snapshot=screening_batches.validate_snapshot(
                coordinator
            ),
        )


def test_release_validation_rejects_swapped_packet_membership(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "swapped-packets-inputs")
    coordinator_root = tmp_path / "swapped-packets-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    release_root = tmp_path / "swapped-packets-releases"
    release_root.mkdir()
    output = release_root / "v1"
    release(coordinator, "calibration", output)
    expected_manifest = _read_csv(output / "release_manifest.csv")[0]

    first = output / "packets" / "screening-01.csv"
    second = output / "packets" / "screening-02.csv"
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    first.write_bytes(second_payload)
    second.write_bytes(first_payload)
    checksum_inputs = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in _snapshot_files(output)
        if path.name != "SHA256SUMS"
    }
    (output / "SHA256SUMS").write_bytes(
        "".join(
            f"{hashlib.sha256(checksum_inputs[name]).hexdigest()}  {name}\n"
            for name in sorted(checksum_inputs)
        ).encode("utf-8")
    )

    with pytest.raises(
        screening_batches.SnapshotError,
        match=(
            "batch_id|assignment.*membership|coordinator snapshot derivation"
        ),
    ):
        screening_batches.validate_reviewer_release_snapshot(
            output,
            expected_manifest=expected_manifest,
            coordinator_snapshot=screening_batches.validate_snapshot(
                coordinator
            ),
        )


def test_release_validation_rejects_self_consistently_rehashed_packet_metadata(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "tampered-packet-metadata-inputs")
    coordinator_root = tmp_path / "tampered-packet-metadata-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    release_root = tmp_path / "tampered-packet-metadata-releases"
    release_root.mkdir()
    output = release_root / "v1"
    release(coordinator, "calibration", output)
    expected_manifest = _read_csv(output / "release_manifest.csv")[0]

    packet = output / "packets" / "screening-01.csv"
    packet_rows = _read_csv(packet)
    packet_rows[0]["title"] = "Tampered reviewer-facing title"
    _write_csv(packet, PACKET_HEADER, packet_rows)
    checksum_inputs = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in _snapshot_files(output)
        if path.name != "SHA256SUMS"
    }
    (output / "SHA256SUMS").write_bytes(
        "".join(
            f"{hashlib.sha256(checksum_inputs[name]).hexdigest()}  {name}\n"
            for name in sorted(checksum_inputs)
        ).encode("utf-8")
    )

    with pytest.raises(
        screening_batches.SnapshotError,
        match="coordinator snapshot derivation",
    ):
        screening_batches.validate_reviewer_release_snapshot(
            output,
            expected_manifest=expected_manifest,
            coordinator_snapshot=screening_batches.validate_snapshot(
                coordinator
            ),
        )


def test_reviewer_release_is_deterministic_nonmutating_and_no_clobber(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator_root = tmp_path / "coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    before = _snapshot_state(snapshot)
    release_root = tmp_path / "reviewer-releases"
    release_root.mkdir()
    first = release_root / "v1"
    second = release_root / "v2"

    release(snapshot, "calibration", first)
    release(snapshot, "calibration", second)

    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in _snapshot_files(first)
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in _snapshot_files(second)
    }
    assert _snapshot_state(snapshot) == before
    with pytest.raises(screening_batches.SnapshotError, match="already exists"):
        release(snapshot, "calibration", first)


def test_reviewer_release_rejects_tampered_coordinator_snapshot(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator_root = tmp_path / "coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    decision_root = tmp_path / "tampered-coordinator-decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(snapshot, decision_root / "v1")
    manifest = snapshot / "manifest.csv"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    release_root = tmp_path / "reviewer-releases"
    release_root.mkdir()
    output = release_root / "v1"

    with pytest.raises(screening_batches.SnapshotError):
        release(snapshot, "main", output, gate)
    assert not os.path.lexists(output)


def test_generated_text_is_utf8_lf_and_has_exact_headers(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)

    derived = [
        snapshot / "calibration_selection.csv",
        snapshot / "manifest.csv",
        snapshot / "SHA256SUMS",
    ] + [
        snapshot / "packets" / f"screening-{number:02d}.csv"
        for number in range(1, 7)
    ]
    for path in derived:
        payload = path.read_bytes()
        payload.decode("utf-8")
        assert b"\r" not in payload
        assert payload.endswith(b"\n")
        assert b"\n\n" not in payload


def test_two_freezes_are_byte_identical_and_do_not_mutate_inputs(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    source_paths = (
        inputs.candidates,
        inputs.conflicts,
        inputs.bibliography,
        inputs.citation_keys,
        inputs.taxonomy,
        inputs.protocol,
    )
    before = {path: _persistent_state(path) for path in source_paths}

    first = tmp_path / "v1"
    second = tmp_path / "v2"
    freeze(inputs, first)
    freeze(inputs, second)

    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in _snapshot_files(first)
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in _snapshot_files(second)
    }
    assert {path: _persistent_state(path) for path in source_paths} == before


def test_freeze_preserves_crlf_source_bytes_exactly(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    original = inputs.candidates.read_bytes().replace(b"\n", b"\r\n")
    inputs.candidates.write_bytes(original)

    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)

    assert (snapshot / "candidates.csv").read_bytes() == original


def test_freeze_sets_deterministic_permissions_despite_umask(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    previous_umask = os.umask(0o077)
    try:
        freeze(inputs, snapshot)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o755
    assert stat.S_IMODE((snapshot / "packets").stat().st_mode) == 0o755
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o644
        for path in _snapshot_files(snapshot)
    )


def test_post_publish_cleanup_preserves_original_error_and_captures_expected_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "published"
    parent.mkdir()
    output = parent / "v1"
    rename = screening_batches._rename_noreplace_at

    def publish_then_restrict(
        parent_fd: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        rename(parent_fd, source_name, destination_name)
        os.chmod(destination_name, 0, dir_fd=parent_fd)

    monkeypatch.setattr(
        screening_batches,
        "_rename_noreplace_at",
        publish_then_restrict,
    )

    try:
        with pytest.raises(
            screening_batches.SnapshotError,
            match="staged snapshot: directory mode",
        ):
            screening_batches._publish_artifacts(
                output,
                {"manifest.csv": b"immutable\n"},
            )
        assert not os.path.lexists(output)
    finally:
        if os.path.lexists(output):
            output.chmod(0o700)
        for retired in parent.glob(".trackgen-retired-*"):
            retired.chmod(0o700)


def test_post_publish_byte_identical_output_replacement_is_rejected(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "published"
    parent.mkdir()
    output = parent / "v1"
    displaced = parent / "displaced-owned-v1"

    def replace_output() -> None:
        output.rename(displaced)
        shutil.copytree(displaced, output)

    with pytest.raises(
        screening_batches.SnapshotError,
        match="changed during post-publication validation",
    ):
        screening_batches._publish_artifacts(
            output,
            {"manifest.csv": b"immutable\n"},
            post_publish_check=replace_output,
        )

    assert output.is_dir()
    assert displaced.is_dir()
    assert output.stat().st_ino != displaced.stat().st_ino
    assert (output / "manifest.csv").read_bytes() == b"immutable\n"
    assert (displaced / "manifest.csv").read_bytes() == b"immutable\n"


def test_validation_is_read_only(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    before = _snapshot_state(snapshot)

    assert screening_batches.main(["--snapshot-dir", str(snapshot)]) == 0

    assert _snapshot_state(snapshot) == before

@pytest.mark.parametrize(
    ("relative_path", "mutation"),
    [
        ("protocol.md", lambda payload: payload + b"\ntampered\n"),
        ("taxonomy.json", lambda payload: payload.replace(b"{", b"{\n ", 1)),
        (
            "calibration_selection.csv",
            lambda payload: payload.replace(b"C", b"X", 1),
        ),
        ("manifest.csv", lambda payload: payload.replace(b",1\n", b",0\n", 1)),
        (
            "packets/screening-01.csv",
            lambda payload: payload.replace(b"Track", b"Tampered", 1),
        ),
        (
            "SHA256SUMS",
            lambda payload: (b"1" if payload[:1] == b"0" else b"0") + payload[1:],
        ),
    ],
)
def test_validation_rejects_raw_or_derived_tampering(
    tmp_path: Path,
    relative_path: str,
    mutation,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    target = snapshot / relative_path
    target.write_bytes(mutation(target.read_bytes()))

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.main(["--snapshot-dir", str(snapshot)])


@pytest.mark.parametrize(
    ("operation", "relative_path"),
    [
        ("missing", "calibration_selection.csv"),
        ("missing", "manifest.csv"),
        ("missing", "packets/screening-06.csv"),
        ("extra", "unexpected.txt"),
        ("extra", "packets/unexpected.csv"),
    ],
)
def test_validation_rejects_missing_or_extra_files(
    tmp_path: Path,
    operation: str,
    relative_path: str,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    target = snapshot / relative_path
    if operation == "missing":
        target.unlink()
    else:
        target.write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.main(["--snapshot-dir", str(snapshot)])


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra"])
def test_validation_rejects_duplicate_missing_or_extra_manifest_rows(
    tmp_path: Path,
    kind: str,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    manifest_path = snapshot / "manifest.csv"
    rows = _read_csv(manifest_path)
    if kind == "duplicate":
        rows.append(dict(rows[0]))
    elif kind == "missing":
        rows.pop(0)
    else:
        extra = dict(rows[0])
        extra["candidate_id"] = "C9999"
        rows.append(extra)
    _write_csv(manifest_path, MANIFEST_HEADER, rows)

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.main(["--snapshot-dir", str(snapshot)])


def test_validation_rejects_nonpositive_or_noncanonical_weight(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    manifest_path = snapshot / "manifest.csv"
    rows = _read_csv(manifest_path)
    rows[0]["weight"] = "0"
    _write_csv(manifest_path, MANIFEST_HEADER, rows)

    with pytest.raises(screening_batches.SnapshotError, match="weight"):
        screening_batches.main(["--snapshot-dir", str(snapshot)])


@pytest.mark.parametrize(
    "field",
    [
        "discovery_query",
        "conflict",
        "key_author",
        "taxonomy",
        "protocol",
    ],
)
def test_candidate_input_and_snapshot_hashes_bind_every_input_class(
    tmp_path: Path,
    field: str,
) -> None:
    baseline_inputs = build_inputs(tmp_path / "baseline-inputs")
    baseline_snapshot = tmp_path / "v1"
    freeze(baseline_inputs, baseline_snapshot)
    baseline = _manifest_by_id(baseline_snapshot)["C0001"]

    changed_inputs = build_inputs(tmp_path / "changed-inputs")
    if field == "discovery_query":
        rows = _read_csv(changed_inputs.candidates)
        rows[0]["discovery_query"] = "different hidden query"
        _write_csv(changed_inputs.candidates, CANDIDATE_HEADER, rows)
    elif field == "conflict":
        rows = _read_csv(changed_inputs.conflicts)
        rows[0]["resolution_evidence"] += "/changed"
        _write_csv(changed_inputs.conflicts, CONFLICT_HEADER, rows)
    elif field == "key_author":
        rows = _read_csv(changed_inputs.bibliography)
        target = next(row for row in rows if row["candidate_id"] == "C0001")
        target["key_author"] = "DifferentAuthor"
        _write_csv(changed_inputs.bibliography, BIBLIOGRAPHY_HEADER, rows)
    elif field == "taxonomy":
        taxonomy = json.loads(changed_inputs.taxonomy.read_text(encoding="utf-8"))
        taxonomy["screening_fixture_marker"] = ["changed"]
        changed_inputs.taxonomy.write_text(
            json.dumps(taxonomy, indent=2) + "\n", encoding="utf-8"
        )
    else:
        changed_inputs.protocol.write_bytes(
            changed_inputs.protocol.read_bytes() + b"\nChanged protocol note.\n"
        )

    changed_snapshot = tmp_path / "v2"
    freeze(changed_inputs, changed_snapshot)
    changed = _manifest_by_id(changed_snapshot)["C0001"]

    assert changed["input_sha256"] != baseline["input_sha256"]
    assert changed["snapshot_sha256"] != baseline["snapshot_sha256"]


def test_source_validation_requires_verified_metadata(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    rows = _read_csv(inputs.candidates)
    rows[0]["metadata_status"] = "unverified"
    _write_csv(inputs.candidates, CANDIDATE_HEADER, rows)

    with pytest.raises(screening_batches.SnapshotError, match="metadata_status"):
        freeze(inputs, tmp_path / "v1")


@pytest.mark.parametrize(
    ("table", "mutation", "message"),
    [
        (
            "candidates",
            lambda rows: rows.__setitem__(-1, dict(rows[0])),
            "duplicate candidate_id",
        ),
        (
            "conflicts",
            lambda rows: rows[0].update(field="unknown_field"),
            "field",
        ),
        (
            "conflicts",
            lambda rows: rows[0].update(record_key="C9999"),
            "record_key",
        ),
        (
            "bibliography",
            lambda rows: rows[0].update(title="Mismatched title"),
            "does not match",
        ),
        (
            "citation_keys",
            lambda rows: rows[0].update(candidate_id="C9999"),
            "candidate_id",
        ),
    ],
)
def test_source_validation_rejects_schema_and_referential_errors(
    tmp_path: Path,
    table: str,
    mutation,
    message: str,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    path = getattr(inputs, table)
    header = {
        "candidates": CANDIDATE_HEADER,
        "conflicts": CONFLICT_HEADER,
        "bibliography": BIBLIOGRAPHY_HEADER,
        "citation_keys": CITATION_KEY_HEADER,
    }[table]
    rows = _read_csv(path)
    mutation(rows)
    _write_csv(path, header, rows)

    with pytest.raises(screening_batches.SnapshotError, match=message):
        freeze(inputs, tmp_path / "v1")


def test_source_validation_rejects_wrong_header_and_invalid_taxonomy(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    inputs.candidates.write_bytes(
        inputs.candidates.read_bytes().replace(b"candidate_id", b"wrong_id", 1)
    )
    with pytest.raises(screening_batches.SnapshotError, match="headers"):
        freeze(inputs, tmp_path / "v1")

    inputs = build_inputs(tmp_path / "inputs-two")
    inputs.taxonomy.write_text("[]\n", encoding="utf-8")
    with pytest.raises(screening_batches.SnapshotError, match="taxonomy"):
        freeze(inputs, tmp_path / "v2")


@pytest.mark.parametrize(
    "inclusion_criteria",
    [
        None,
        ["include-1"],
        ["include-relevant", "include-1"],
        [],
    ],
)
def test_freeze_requires_exact_current_inclusion_criterion(
    tmp_path: Path,
    inclusion_criteria: list[str] | None,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    taxonomy = json.loads(inputs.taxonomy.read_text(encoding="utf-8"))
    if inclusion_criteria is None:
        del taxonomy["screening_inclusion_criterion"]
    else:
        taxonomy["screening_inclusion_criterion"] = inclusion_criteria
    inputs.taxonomy.write_bytes(_canonical_json_bytes(taxonomy))

    with pytest.raises(screening_batches.SnapshotError):
        freeze(inputs, tmp_path / "v1")


@pytest.mark.parametrize(
    ("result_statuses", "valid"),
    [
        (["included", "excluded"], True),
        (None, False),
        ([], False),
        (["included"], False),
        (["included", "boundary", "excluded"], False),
    ],
)
def test_fresh_freeze_requires_exact_current_screening_result_statuses(
    tmp_path: Path,
    result_statuses: list[str] | None,
    valid: bool,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    taxonomy = json.loads(inputs.taxonomy.read_text(encoding="utf-8"))
    if result_statuses is None:
        del taxonomy["screening_result_status"]
    else:
        taxonomy["screening_result_status"] = result_statuses
    inputs.taxonomy.write_bytes(_canonical_json_bytes(taxonomy))

    if valid:
        freeze(inputs, tmp_path / "v1")
    else:
        with pytest.raises(
            screening_batches.SnapshotError,
            match="screening_result_status",
        ):
            freeze(inputs, tmp_path / "v1")


def test_screening_result_status_resolver_returns_allowed_binary_statuses() -> None:
    assert screening_batches._resolve_screening_result_statuses(
        {"screening_result_status": ["included", "excluded"]},
        strict_new=True,
    ) == ("included", "excluded")


def test_stage_derives_allowed_screening_statuses_without_serializing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator = tmp_path / "coordinator" / "v1"
    coordinator.parent.mkdir()
    freeze(inputs, coordinator)
    reviewer_release = tmp_path / "releases" / "v1"
    reviewer_release.parent.mkdir()
    release(coordinator, "calibration", reviewer_release)

    received_statuses: list[tuple[str, ...] | None] = []
    build_configuration = screening_batches._execution_configuration

    def capture_configuration(**kwargs):
        received_statuses.append(kwargs.get("allowed_screening_statuses"))
        return build_configuration(**kwargs)

    monkeypatch.setattr(
        screening_batches,
        "_execution_configuration",
        capture_configuration,
    )
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    os.chmod(staging_root, 0o700)
    stage = screening_batches.stage_reviewer_execution(
        coordinator,
        reviewer_release,
        "screening-01",
        staging_root,
    )

    assert ("included", "excluded") in received_statuses
    configuration = json.loads(
        (stage / "execution_configuration.json").read_text(encoding="utf-8")
    )
    assert configuration["configuration_version"] == "2"
    assert "allowed_screening_statuses" not in configuration


@pytest.mark.parametrize("version", range(1, 7))
def test_committed_historical_coordinator_validation_leaves_v1_through_v6_unchanged(
    version: int,
) -> None:
    snapshot = DATA_ROOT / "screening_inputs" / f"v{version}"
    before = {
        path.relative_to(snapshot): path.read_bytes()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    taxonomy = json.loads(before[Path("taxonomy.json")].decode("utf-8"))
    assert "screening_result_status" not in taxonomy

    screening_batches.validate_snapshot(snapshot)

    after = {
        path.relative_to(snapshot): path.read_bytes()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_source_validation_requires_resolver_for_resolved_conflict(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    rows = _read_csv(inputs.conflicts)
    rows[0]["resolver"] = ""
    _write_csv(inputs.conflicts, CONFLICT_HEADER, rows)

    with pytest.raises(screening_batches.SnapshotError, match="resolver"):
        freeze(inputs, tmp_path / "v1")


def test_freeze_refuses_existing_directory_file_and_dangling_symlink(
    tmp_path: Path,
) -> None:
    for kind in ("directory", "file", "symlink"):
        case = tmp_path / kind
        case.mkdir()
        inputs = build_inputs(case / "inputs")
        output = case / "v1"
        if kind == "directory":
            output.mkdir()
        elif kind == "file":
            output.write_text("occupied\n", encoding="utf-8")
        else:
            output.symlink_to(case / "missing")

        with pytest.raises(screening_batches.SnapshotError, match="already exists"):
            freeze(inputs, output)
        assert os.path.lexists(output)


def test_freeze_rejects_current_pointer_name(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")

    with pytest.raises(screening_batches.SnapshotError, match="version"):
        freeze(inputs, tmp_path / "current")


def test_freeze_rejects_symlinked_input_and_output_parent(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    real_protocol = inputs.protocol.with_name("real-protocol.md")
    inputs.protocol.rename(real_protocol)
    inputs.protocol.symlink_to(real_protocol)
    with pytest.raises(screening_batches.SnapshotError, match="symlink"):
        freeze(inputs, tmp_path / "v1")

    second = build_inputs(tmp_path / "inputs-two")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(screening_batches.SnapshotError, match="symlink"):
        freeze(second, linked_parent / "v2")


def test_freeze_rejects_hard_linked_inputs(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    inputs.conflicts.unlink()
    os.link(inputs.candidates, inputs.conflicts)

    with pytest.raises(screening_batches.SnapshotError, match="hard link|aliases"):
        freeze(inputs, tmp_path / "v1")


def test_validation_rejects_symlink_and_hardlink_aliases(tmp_path: Path) -> None:
    symlink_inputs = build_inputs(tmp_path / "symlink-inputs")
    symlink_snapshot = tmp_path / "v1"
    freeze(symlink_inputs, symlink_snapshot)
    packet = symlink_snapshot / "packets" / "screening-01.csv"
    target = symlink_snapshot / "packets" / "screening-02.csv"
    packet.unlink()
    packet.symlink_to(target.name)
    with pytest.raises(screening_batches.SnapshotError, match="symlink"):
        screening_batches.main(["--snapshot-dir", str(symlink_snapshot)])

    hardlink_inputs = build_inputs(tmp_path / "hardlink-inputs")
    hardlink_snapshot = tmp_path / "v2"
    freeze(hardlink_inputs, hardlink_snapshot)
    first = hardlink_snapshot / "packets" / "screening-01.csv"
    second = hardlink_snapshot / "packets" / "screening-02.csv"
    second.unlink()
    os.link(first, second)
    with pytest.raises(screening_batches.SnapshotError, match="hard link|aliases"):
        screening_batches.main(["--snapshot-dir", str(hardlink_snapshot)])


def test_validation_rechecks_file_hash_after_post_read_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    reader = getattr(screening_batches, "_read_regular_file_at", None)
    assert reader is not None
    mutated = False

    def mutate_after_read(directory_fd: int, name: str, label: str):
        nonlocal mutated
        result = reader(directory_fd, name, label)
        if label == "manifest.csv" and not mutated:
            mutated = True
            manifest = snapshot / "manifest.csv"
            manifest.write_bytes(manifest.read_bytes() + b"tamper")
        return result

    monkeypatch.setattr(
        screening_batches,
        "_read_regular_file_at",
        mutate_after_read,
    )

    with pytest.raises(screening_batches.SnapshotError, match="changed|hash"):
        screening_batches.main(["--snapshot-dir", str(snapshot)])


def test_validation_rechecks_link_count_after_post_read_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze(inputs, snapshot)
    reader = getattr(screening_batches, "_read_regular_file_at", None)
    assert reader is not None
    mutated = False

    def hardlink_after_read(directory_fd: int, name: str, label: str):
        nonlocal mutated
        result = reader(directory_fd, name, label)
        if label == "packets/screening-06.csv" and not mutated:
            mutated = True
            manifest = snapshot / "manifest.csv"
            manifest.unlink()
            os.link(snapshot / "protocol.md", manifest)
        return result

    monkeypatch.setattr(
        screening_batches,
        "_read_regular_file_at",
        hardlink_after_read,
    )

    with pytest.raises(screening_batches.SnapshotError, match="hard link|changed"):
        screening_batches.main(["--snapshot-dir", str(snapshot)])


@pytest.mark.parametrize("mutation", ["bytes", "hardlink"])
def test_staged_artifacts_are_reverified_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    original = screening_batches._stage_artifacts

    def mutate_after_staging(staging, artifacts, identities) -> None:
        original(staging, artifacts, identities)
        root = (
            Path(f"/proc/self/fd/{staging}")
            if isinstance(staging, int)
            else Path(staging)
        )
        first = root / "packets" / "screening-01.csv"
        if mutation == "bytes":
            first.write_bytes(first.read_bytes() + b"tamper")
        else:
            second = root / "packets" / "screening-02.csv"
            second.unlink()
            os.link(first, second)

    monkeypatch.setattr(
        screening_batches,
        "_stage_artifacts",
        mutate_after_staging,
    )

    with pytest.raises(screening_batches.SnapshotError, match="staged|hard link"):
        freeze(inputs, output)
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".v1.*.tmp"))


def test_stage_is_verified_again_after_parent_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    original = screening_batches._verify_staged_artifacts
    calls = 0

    def mutate_after_first_verification(staging, artifacts, identities) -> None:
        nonlocal calls
        calls += 1
        original(staging, artifacts, identities)
        if calls == 1:
            packet = (
                Path(f"/proc/self/fd/{staging}")
                / "packets"
                / "screening-01.csv"
            )
            packet.write_bytes(packet.read_bytes() + b"late-tamper")

    monkeypatch.setattr(
        screening_batches,
        "_verify_staged_artifacts",
        mutate_after_first_verification,
    )

    with pytest.raises(screening_batches.SnapshotError, match="staged.*hash"):
        freeze(inputs, output)
    assert calls == 2
    assert not os.path.lexists(output)


def test_cleanup_quarantine_replacement_is_restored_without_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "v1"
    packets = root / "packets"
    packets.mkdir(parents=True)
    packet = packets / "screening-01.csv"
    packet.write_bytes(b"owned packet\n")
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    identities = {
        ".": root_identity,
        "packets": (packets.stat().st_dev, packets.stat().st_ino),
        "packets/screening-01.csv": (
            packet.stat().st_dev,
            packet.stat().st_ino,
        ),
    }
    parked_owned = tmp_path / "parked-owned-v1"
    replacement = tmp_path / "concurrent-v1"
    replacement.mkdir()
    (replacement / "marker.txt").write_bytes(b"concurrent tree\n")
    replacement_identity = (
        replacement.stat().st_dev,
        replacement.stat().st_ino,
    )
    rename_noreplace = screening_batches._cleanup_rename_noreplace_at
    swapped = False

    def rename_then_swap(
        directory_fd: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        rename_noreplace(
            directory_fd,
            source_name,
            destination_name,
        )
        if (
            not swapped
            and source_name == root.name
            and destination_name.startswith(".trackgen-retired-")
        ):
            swapped = True
            directory = Path(f"/proc/self/fd/{directory_fd}")
            (directory / destination_name).rename(parked_owned)
            replacement.rename(directory / destination_name)

    monkeypatch.setattr(
        screening_batches,
        "_cleanup_rename_noreplace_at",
        rename_then_swap,
    )
    parent_fd = os.open(tmp_path, screening_batches._DIRECTORY_OPEN_FLAGS)
    root_fd = os.open(root, screening_batches._DIRECTORY_OPEN_FLAGS)
    try:
        with pytest.raises(
            screening_batches.SnapshotError,
            match="captured foreign entry",
        ):
            screening_batches._capture_snapshot_root_at(
                parent_fd,
                root.name,
                identities,
                root_fd=root_fd,
            )
    finally:
        os.close(root_fd)
        os.close(parent_fd)

    assert swapped
    assert (root / "marker.txt").read_bytes() == b"concurrent tree\n"
    assert (root.stat().st_dev, root.stat().st_ino) == replacement_identity
    assert (parked_owned.stat().st_dev, parked_owned.stat().st_ino) == (
        root_identity
    )
    assert (
        parked_owned / "packets" / "screening-01.csv"
    ).read_bytes() == b"owned packet\n"


def test_snapshot_cleanup_quarantines_foreign_root_when_source_refills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "v1"
    captured_foreign = tmp_path / "captured-foreign-root"
    captured_foreign.mkdir()
    (captured_foreign / "marker.txt").write_bytes(b"captured foreign root\n")
    captured_identity = (
        captured_foreign.stat().st_dev,
        captured_foreign.stat().st_ino,
    )
    refill_foreign = tmp_path / "refill-foreign-root"
    refill_foreign.mkdir()
    (refill_foreign / "marker.txt").write_bytes(b"refilled foreign root\n")
    refill_identity = (
        refill_foreign.stat().st_dev,
        refill_foreign.stat().st_ino,
    )
    parked_expected = tmp_path / "parked-published-root"
    cleanup_started = False
    source_replaced = False
    source_refilled = False
    quarantine_path = None
    real_identity_at = screening_batches._identity_at
    real_rename = screening_batches._cleanup_rename_noreplace_at

    def fail_after_publish() -> None:
        nonlocal cleanup_started
        cleanup_started = True
        raise OSError("primary publication failure")

    def identity_then_replace(directory_fd: int, name: str):
        nonlocal source_replaced
        identity = real_identity_at(directory_fd, name)
        if (
            cleanup_started
            and not source_replaced
            and name == output.name
            and identity is not None
        ):
            output.rename(parked_expected)
            captured_foreign.rename(output)
            source_replaced = True
        return identity

    def capture_then_refill(
        directory_fd: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal quarantine_path, source_refilled
        real_rename(directory_fd, source_name, destination_name)
        if (
            source_replaced
            and not source_refilled
            and source_name == output.name
            and destination_name.startswith(".trackgen-retired-")
        ):
            quarantine_path = output.parent / destination_name
            refill_foreign.rename(output)
            source_refilled = True

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("cleanup must not delete a mutable pathname")

    monkeypatch.setattr(screening_batches, "_identity_at", identity_then_replace)
    monkeypatch.setattr(
        screening_batches,
        "_cleanup_rename_noreplace_at",
        capture_then_refill,
    )
    monkeypatch.setattr(screening_batches.os, "unlink", forbidden_delete)
    monkeypatch.setattr(screening_batches.os, "remove", forbidden_delete)
    monkeypatch.setattr(screening_batches.os, "rmdir", forbidden_delete)

    with pytest.raises(OSError, match="primary publication failure") as raised:
        screening_batches._publish_artifacts(
            output,
            {"manifest.csv": b"immutable\n"},
            post_publish_check=fail_after_publish,
        )

    assert source_replaced
    assert source_refilled
    assert quarantine_path is not None
    assert quarantine_path.is_absolute()
    assert (quarantine_path / "marker.txt").read_bytes() == (
        b"captured foreign root\n"
    )
    assert (quarantine_path.stat().st_dev, quarantine_path.stat().st_ino) == (
        captured_identity
    )
    assert (output / "marker.txt").read_bytes() == b"refilled foreign root\n"
    assert (output.stat().st_dev, output.stat().st_ino) == refill_identity
    recovery = _exception_text(raised.value)
    assert str(quarantine_path) in recovery
    assert f"(dev, ino)=({captured_identity[0]}, {captured_identity[1]})" in recovery


def test_cleanup_captures_complete_expected_tree_without_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "v1"
    packets = root / "packets"
    packets.mkdir(parents=True)
    packet = packets / "screening-01.csv"
    packet.write_bytes(b"owned packet\n")
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    identities = {
        ".": root_identity,
        "packets": (packets.stat().st_dev, packets.stat().st_ino),
        "packets/screening-01.csv": (
            packet.stat().st_dev,
            packet.stat().st_ino,
        ),
    }

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("cleanup must not delete a mutable pathname")

    monkeypatch.setattr(screening_batches.os, "unlink", forbidden_delete)
    monkeypatch.setattr(screening_batches.os, "remove", forbidden_delete)
    monkeypatch.setattr(screening_batches.os, "rmdir", forbidden_delete)
    parent_fd = os.open(tmp_path, screening_batches._DIRECTORY_OPEN_FLAGS)
    root_fd = os.open(root, screening_batches._DIRECTORY_OPEN_FLAGS)
    try:
        screening_batches._capture_snapshot_root_at(
            parent_fd,
            root.name,
            identities,
            root_fd=root_fd,
        )
    finally:
        os.close(root_fd)
        os.close(parent_fd)

    assert not root.exists()
    retired_roots = list(tmp_path.glob(".trackgen-retired-*"))
    assert len(retired_roots) == 1
    retired_root = retired_roots[0]
    assert (retired_root.stat().st_dev, retired_root.stat().st_ino) == (
        root_identity
    )
    assert (
        retired_root / "packets" / "screening-01.csv"
    ).read_bytes() == b"owned packet\n"


def test_partial_write_failure_captures_complete_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    original_write = screening_batches.os.write
    injected = False

    def partial_write(descriptor: int, payload) -> int:
        nonlocal injected
        if not injected:
            injected = True
            original_write(descriptor, payload[: max(1, len(payload) // 2)])
            raise OSError("injected partial write failure")
        return original_write(descriptor, payload)

    def forbidden_delete(*_args, **_kwargs):
        raise AssertionError("cleanup must not delete a mutable pathname")

    monkeypatch.setattr(screening_batches.os, "write", partial_write)
    monkeypatch.setattr(screening_batches.os, "unlink", forbidden_delete)
    monkeypatch.setattr(screening_batches.os, "remove", forbidden_delete)
    monkeypatch.setattr(screening_batches.os, "rmdir", forbidden_delete)

    with pytest.raises(OSError, match="partial write"):
        freeze(inputs, output)
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".v1.*.tmp"))
    retired_roots = list(tmp_path.glob(".trackgen-retired-*"))
    assert len(retired_roots) == 1
    assert any(path.is_file() for path in retired_roots[0].rglob("*"))

def test_output_parent_retarget_is_detected_and_owned_stage_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    publish_parent = tmp_path / "publish"
    publish_parent.mkdir()
    displaced_parent = tmp_path / "displaced-publish"
    attacker_parent = tmp_path / "attacker"
    attacker_parent.mkdir()
    output = publish_parent / "v1"
    original = screening_batches._stage_artifacts

    def retarget_after_staging(staging, artifacts, identities) -> None:
        original(staging, artifacts, identities)
        publish_parent.rename(displaced_parent)
        publish_parent.symlink_to(attacker_parent, target_is_directory=True)

    monkeypatch.setattr(
        screening_batches,
        "_stage_artifacts",
        retarget_after_staging,
    )

    with pytest.raises(screening_batches.SnapshotError, match="parent.*changed"):
        freeze(inputs, output)
    assert not (attacker_parent / "v1").exists()
    assert not (displaced_parent / "v1").exists()
    assert not list(displaced_parent.glob(".v1.*.tmp"))


def test_staging_failure_captures_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    original = screening_batches._write_snapshot_file_at
    calls = 0

    def fail_during_staging(
        directory_fd: int,
        name: str,
        payload: bytes,
        *,
        on_created=None,
    ):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected staging failure")
        return original(
            directory_fd, name, payload, on_created=on_created
        )

    monkeypatch.setattr(
        screening_batches,
        "_write_snapshot_file_at",
        fail_during_staging,
    )

    with pytest.raises(OSError, match="injected staging failure"):
        freeze(inputs, output)
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".v1.*.tmp"))


def test_publication_race_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    original = screening_batches._rename_noreplace_at

    def race(parent_fd: int, source_name: str, destination_name: str) -> None:
        parent = Path(f"/proc/self/fd/{parent_fd}")
        destination = parent / destination_name
        destination.mkdir()
        (destination / "racer.txt").write_text("racer\n", encoding="utf-8")
        original(parent_fd, source_name, destination_name)

    monkeypatch.setattr(screening_batches, "_rename_noreplace_at", race)

    with pytest.raises(screening_batches.SnapshotError, match="already exists"):
        freeze(inputs, output)
    assert (output / "racer.txt").read_text(encoding="utf-8") == "racer\n"
    assert not list(tmp_path.glob(".v1.*.tmp"))


def test_post_publication_failure_captures_expected_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    output = tmp_path / "v1"
    original = screening_batches._rename_noreplace_at

    def publish_then_fail(
        parent_fd: int, source_name: str, destination_name: str
    ) -> None:
        original(parent_fd, source_name, destination_name)
        raise OSError("injected post-publication failure")

    monkeypatch.setattr(
        screening_batches,
        "_rename_noreplace_at",
        publish_then_fail,
    )

    with pytest.raises(OSError, match="injected"):
        freeze(inputs, output)
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".v1.*.tmp"))


def test_actual_cli_freeze_and_validation(tmp_path: Path) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    snapshot = tmp_path / "v1"
    freeze_result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *freeze_arguments(inputs, snapshot)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert freeze_result.returncode == 0, freeze_result.stderr

    validate_result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--snapshot-dir", str(snapshot)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate_result.returncode == 0, validate_result.stderr

    release_root = tmp_path / "reviewer-releases"
    release_root.mkdir()
    release_output = release_root / "v1"
    release_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            *release_arguments(snapshot, "calibration", release_output),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert release_result.returncode == 0, release_result.stderr
    assert (release_output / "packets" / "screening-01.csv").is_file()


def test_current_corpus_freezes_to_required_duplicate_batch_counts(
    tmp_path: Path,
) -> None:
    inputs = Inputs(
        root=DATA_ROOT,
        candidates=DATA_ROOT / "candidates.csv",
        conflicts=DATA_ROOT / "conflicts.csv",
        bibliography=DATA_ROOT / "bibliography.csv",
        citation_keys=DATA_ROOT / "citation_keys.csv",
        taxonomy=_fresh_taxonomy(tmp_path / "taxonomy.json"),
        protocol=PROTOCOL_PATH,
        execution_profile=EXECUTION_PROFILE_PATH,
        reviewer_prompt_template=REVIEWER_PROMPT_TEMPLATE_PATH,
    )
    snapshot = tmp_path / "v1"

    freeze(inputs, snapshot)

    manifest = _read_csv(snapshot / "manifest.csv")
    assert len(manifest) == 404
    assert Counter(row["candidate_id"] for row in manifest) == Counter(
        {row["candidate_id"]: 2 for row in _read_csv(inputs.candidates)}
    )
    assert Counter(row["batch_id"] for row in manifest) == {
        "screening-01": 68,
        "screening-02": 68,
        "screening-03": 67,
        "screening-04": 67,
        "screening-05": 67,
        "screening-06": 67,
    }
    assert len({row["assignment_id"] for row in manifest}) == 404
    assert all(row["weight"] == "1" for row in manifest)

    protocol = inputs.protocol.read_bytes()
    ranked_ids = sorted(
        {row["candidate_id"] for row in manifest},
        key=lambda candidate_id: _screening_rank(candidate_id, protocol),
    )
    expected_pairs = {
        candidate_id: REVIEWER_PAIRS[index % len(REVIEWER_PAIRS)]
        for index, candidate_id in enumerate(ranked_ids)
    }
    for candidate_id in ranked_ids:
        rows = [row for row in manifest if row["candidate_id"] == candidate_id]
        assert tuple(sorted(row["batch_id"] for row in rows)) == expected_pairs[
            candidate_id
        ]
        assert len({row["input_sha256"] for row in rows}) == 1

    candidate_rows = _read_csv(inputs.candidates)
    calibration_ids = _expected_calibration_ids(candidate_rows)
    assert len(calibration_ids) == 30
    assert Counter(row["phase"] for row in manifest) == {
        "calibration": 60,
        "main": 344,
    }
    assert {
        row["candidate_id"]: row["phase"] for row in manifest
    } == {
        candidate_id: (
            "calibration" if candidate_id in calibration_ids else "main"
        )
        for candidate_id in ranked_ids
    }
    assert Counter(
        row["candidate_id"]
        for row in manifest
        if row["candidate_id"] in calibration_ids
    ) == Counter({candidate_id: 2 for candidate_id in calibration_ids})
    assert screening_batches.main(["--snapshot-dir", str(snapshot)]) == 0


@pytest.mark.parametrize("mutation", ["bytes", "hardlink"])
def test_post_real_rename_drift_is_detected_and_published_tree_is_captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inputs = build_inputs(tmp_path / "post-rename-inputs")
    output = tmp_path / "v97"
    original_stage = screening_batches._stage_artifacts
    original_rename = screening_batches._rename_noreplace_at
    staged_fd: int | None = None

    def capture_stage(staging, artifacts, identities) -> None:
        nonlocal staged_fd
        staged_fd = staging
        original_stage(staging, artifacts, identities)

    def rename_then_mutate(
        parent_fd: int, source_name: str, destination_name: str
    ) -> None:
        original_rename(parent_fd, source_name, destination_name)
        assert staged_fd is not None
        packets = Path(f"/proc/self/fd/{staged_fd}") / "packets"
        first = packets / "screening-01.csv"
        if mutation == "bytes":
            first.write_bytes(first.read_bytes() + b"post-rename-tamper")
        else:
            second = packets / "screening-02.csv"
            second.unlink()
            os.link(first, second)

    monkeypatch.setattr(screening_batches, "_stage_artifacts", capture_stage)
    monkeypatch.setattr(
        screening_batches, "_rename_noreplace_at", rename_then_mutate
    )

    with pytest.raises(screening_batches.SnapshotError, match="staged|hard link"):
        freeze(inputs, output)
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".v97.*.tmp"))


def test_post_create_pre_return_failure_captures_complete_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "post-create-inputs")
    output = tmp_path / "v98"
    original = screening_batches._write_snapshot_file_at
    injected = False

    def create_then_fail(
        directory_fd: int,
        name: str,
        payload: bytes,
        *,
        on_created=None,
    ):
        nonlocal injected
        identity = original(
            directory_fd, name, payload, on_created=on_created
        )
        if not injected:
            injected = True
            raise OSError("injected post-create return failure")
        return identity

    monkeypatch.setattr(
        screening_batches, "_write_snapshot_file_at", create_then_fail
    )

    with pytest.raises(OSError, match="post-create return failure"):
        freeze(inputs, output)
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".v98.*.tmp"))


SOURCE_CLASSES = (
    "standard-specification",
    "competition",
    "benchmark-dataset",
    "software",
    "scholarly",
    "official-other",
)


def _normalize_metadata_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    pieces: list[str] = []
    separating = False
    for character in normalized:
        if character.isalnum():
            pieces.append(character)
            separating = False
        elif pieces and not separating:
            pieces.append(" ")
            separating = True
    return "".join(pieces).strip()


def test_metadata_normalization_treats_underscore_like_hyphen() -> None:
    assert _normalize_metadata_token("alpha_beta") == "alpha beta"
    assert _normalize_metadata_token("alpha-beta") == "alpha beta"
    assert screening_batches._normalize_metadata_token("alpha_beta") == "alpha beta"
    assert screening_batches._normalize_metadata_token("alpha-beta") == "alpha beta"


def _coarse_source_class(source_type: str) -> str:
    value = _normalize_metadata_token(source_type)
    if any(marker in value for marker in ("standard", "specification", "file format")):
        return "standard-specification"
    if "competition" in value:
        return "competition"
    if any(marker in value for marker in ("benchmark", "dataset")):
        return "benchmark-dataset"
    if any(marker in value for marker in (
        "software", "repository", "simulator", "platform", "package",
        "game", "engine", "tool",
    )):
        return "software"
    if any(marker in value for marker in (
        "article", "paper", "preprint", "chapter", "thesis", "report", "survey",
    )):
        return "scholarly"
    return "official-other"


def _discovery_labels(row: dict[str, str]) -> set[str]:
    labels: set[str] = set()
    for field, prefix in (("discovery_stream", "stream:"), ("discovery_query", "query:")):
        for raw in row[field].split(";"):
            value = _normalize_metadata_token(raw)
            if value:
                labels.add(prefix + value)
    return labels


def _candidate_rank(candidate_id: str) -> tuple[str, bytes]:
    payload = RANKING_SALT.encode() + b"\0" + candidate_id.encode()
    return hashlib.sha256(payload).hexdigest(), candidate_id.encode()


def _expected_calibration_ids(rows: list[dict[str, str]]) -> set[str]:
    assert len(rows) == 202
    grouped = {
        source_class: [
            row for row in rows
            if _coarse_source_class(row["source_type"]) == source_class
        ]
        for source_class in SOURCE_CLASSES
    }
    populated = [source_class for source_class in SOURCE_CLASSES if grouped[source_class]]
    quotas = {
        source_class: min(2, len(grouped[source_class]))
        for source_class in populated
    }
    remaining = 30 - sum(quotas.values())
    capacities = {
        source_class: len(grouped[source_class]) - quotas[source_class]
        for source_class in populated
    }
    capacity_total = sum(capacities.values())
    remainders: list[tuple[int, int, str]] = []
    for class_index, source_class in enumerate(SOURCE_CLASSES):
        if source_class not in quotas:
            continue
        increment, remainder = divmod(
            remaining * capacities[source_class], capacity_total
        )
        quotas[source_class] += increment
        remainders.append((remainder, -class_index, source_class))
    for _, _, source_class in sorted(remainders, reverse=True)[: 30 - sum(quotas.values())]:
        quotas[source_class] += 1

    selected: set[str] = set()
    for source_class in SOURCE_CLASSES:
        if source_class not in quotas:
            continue
        pool = list(grouped[source_class])
        seen_labels: set[str] = set()
        for _ in range(quotas[source_class]):
            chosen = min(
                pool,
                key=lambda row: (
                    -len(_discovery_labels(row) - seen_labels),
                    *_candidate_rank(row["candidate_id"]),
                ),
            )
            pool.remove(chosen)
            selected.add(chosen["candidate_id"])
            seen_labels.update(_discovery_labels(chosen))
    return selected


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant", "alias"])
def test_reviewer_release_path_must_be_disjoint_from_coordinator_snapshot(
    tmp_path: Path,
    relation: str,
) -> None:
    outer = tmp_path / "v90"
    outer.mkdir()
    coordinator_root = outer / "coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    inputs = build_inputs(tmp_path / f"disjoint-inputs-{relation}")
    freeze(inputs, snapshot)
    before = _snapshot_state(snapshot)

    if relation == "equal":
        output = snapshot
    elif relation == "ancestor":
        output = outer
    elif relation == "descendant":
        output = snapshot / "v2"
    else:
        alias = tmp_path / "snapshot-alias"
        alias.symlink_to(snapshot, target_is_directory=True)
        output = alias / "v3"

    with pytest.raises(screening_batches.SnapshotError, match="disjoint|overlap"):
        release(snapshot, "calibration", output)
    assert _snapshot_state(snapshot) == before
    if relation == "descendant":
        assert not os.path.lexists(output)


def test_calibration_selection_is_metadata_stratified_and_protocol_stable(
    tmp_path: Path,
) -> None:
    first_inputs = build_inputs(tmp_path / "stratified-inputs-one")
    rows = _read_csv(first_inputs.candidates)
    source_types = (
        "conference paper",
        "software repository",
        "benchmark dataset",
        "official competition specification",
        "official standard",
        "official documentation",
    )
    streams = ("alpha", "beta", "gamma", "alpha; delta")
    for index, row in enumerate(rows):
        row["source_type"] = source_types[index % len(source_types)]
        row["discovery_stream"] = streams[index % len(streams)]
    _write_csv(first_inputs.candidates, CANDIDATE_HEADER, rows)
    bibliography = [_bibliography_row(row) for row in rows if row["cite_key"]]
    bibliography.sort(
        key=lambda row: (
            row["cite_key"].casefold(), row["cite_key"], row["candidate_id"]
        )
    )
    _write_csv(first_inputs.bibliography, BIBLIOGRAPHY_HEADER, bibliography)

    first = tmp_path / "v91"
    freeze(first_inputs, first)
    first_calibration = {
        row["candidate_id"]
        for row in _read_csv(first / "manifest.csv")
        if row["phase"] == "calibration"
    }

    first_order = [
        row["candidate_id"]
        for row in _read_csv(first / "calibration_selection.csv")
    ]

    first_inputs.protocol.write_bytes(
        first_inputs.protocol.read_bytes() + b"\nProtocol revision.\n"
    )
    revised = tmp_path / "v92"
    freeze(first_inputs, revised)
    revised_calibration = {
        row["candidate_id"]
        for row in _read_csv(revised / "manifest.csv")
        if row["phase"] == "calibration"
    }

    revised_order = [
        row["candidate_id"]
        for row in _read_csv(revised / "calibration_selection.csv")
    ]

    assert first_calibration == _expected_calibration_ids(rows)
    assert revised_calibration == first_calibration
    assert first_order == sorted(first_calibration, key=_candidate_rank)
    assert revised_order == first_order
    assert len(first_calibration) == 30


def test_actual_corpus_calibration_strata_and_discovery_labels_are_covered(
    tmp_path: Path,
) -> None:
    inputs = Inputs(
        root=DATA_ROOT,
        candidates=DATA_ROOT / "candidates.csv",
        conflicts=DATA_ROOT / "conflicts.csv",
        bibliography=DATA_ROOT / "bibliography.csv",
        citation_keys=DATA_ROOT / "citation_keys.csv",
        taxonomy=_fresh_taxonomy(tmp_path / "taxonomy.json"),
        protocol=PROTOCOL_PATH,
        execution_profile=EXECUTION_PROFILE_PATH,
        reviewer_prompt_template=REVIEWER_PROMPT_TEMPLATE_PATH,
    )
    rows = _read_csv(inputs.candidates)
    snapshot = tmp_path / "v93"
    freeze(inputs, snapshot)
    manifest = _read_csv(snapshot / "manifest.csv")
    calibration_ids = {
        row["candidate_id"] for row in manifest if row["phase"] == "calibration"
    }

    assert calibration_ids == _expected_calibration_ids(rows)
    assert Counter(row["phase"] for row in manifest) == {
        "calibration": 60,
        "main": 344,
    }
    assert Counter(row["batch_id"] for row in manifest) == {
        "screening-01": 68,
        "screening-02": 68,
        "screening-03": 67,
        "screening-04": 67,
        "screening-05": 67,
        "screening-06": 67,
    }
    class_counts = Counter(_coarse_source_class(row["source_type"]) for row in rows)
    selected_counts = Counter(
        _coarse_source_class(row["source_type"])
        for row in rows
        if row["candidate_id"] in calibration_ids
    )
    assert class_counts == {
        "scholarly": 143,
        "software": 29,
        "benchmark-dataset": 7,
        "competition": 7,
        "standard-specification": 13,
        "official-other": 3,
    }
    assert selected_counts == {
        "scholarly": 15,
        "software": 5,
        "benchmark-dataset": 2,
        "competition": 3,
        "standard-specification": 3,
        "official-other": 2,
    }
    available_label_counts: dict[str, int] = {}
    selected_label_counts: dict[str, int] = {}
    for source_class in SOURCE_CLASSES:
        available_labels = set().union(
            *(
                _discovery_labels(row)
                for row in rows
                if _coarse_source_class(row["source_type"]) == source_class
            )
        )
        selected_labels = set().union(
            *(
                _discovery_labels(row)
                for row in rows
                if row["candidate_id"] in calibration_ids
                and _coarse_source_class(row["source_type"]) == source_class
            )
        )
        assert selected_labels <= available_labels
        available_label_counts[source_class] = len(available_labels)
        selected_label_counts[source_class] = len(selected_labels)
    assert available_label_counts == {
        "standard-specification": 17,
        "competition": 10,
        "benchmark-dataset": 12,
        "software": 23,
        "scholarly": 138,
        "official-other": 5,
    }
    assert selected_label_counts == {
        "standard-specification": 7,
        "competition": 6,
        "benchmark-dataset": 8,
        "software": 17,
        "scholarly": 45,
        "official-other": 4,
    }


DECISION_HEADER = screening_results.CALIBRATION_DECISION_HEADER
DECISION_MANIFEST_HEADER = (
    screening_results.CALIBRATION_DECISION_MANIFEST_HEADER
)


@pytest.mark.parametrize(
    ("taxonomy", "expected"),
    [
        (
            {"screening_inclusion_criterion": ["include-relevant"]},
            "include-relevant",
        ),
        ({}, "include-1"),
    ],
)
def test_calibration_result_fixture_uses_bound_or_historical_criterion(
    taxonomy: dict[str, list[str]],
    expected: str,
) -> None:
    assert _fixture_inclusion_criterion(taxonomy) == expected


def _fixture_inclusion_criterion(
    taxonomy: dict[str, list[str]],
) -> str:
    criteria = taxonomy.get(
        screening_batches.SCREENING_INCLUSION_CRITERION_KEY
    )
    if criteria is None:
        return "include-1"
    return criteria[0]


def _calibration_result_row(
    manifest_row: dict[str, str],
    inclusion_criterion: str,
) -> dict[str, str]:
    candidate_id = manifest_row["candidate_id"]
    return {
        "assignment_id": manifest_row["assignment_id"],
        "phase": manifest_row["phase"],
        "candidate_id": candidate_id,
        "input_sha256": manifest_row["input_sha256"],
        "snapshot_sha256": manifest_row["snapshot_sha256"],
        "batch_id": manifest_row["batch_id"],
        "coder_id": manifest_row["batch_id"],
        "screened_on": "2026-06-30",
        "screening_status": "included",
        "criterion": inclusion_criterion,
        "access_status": "full_text",
        "source_urls": f"https://example.test/source/{candidate_id}",
        "evidence_version": "publisher-version-of-record",
        "evidence_retrieved_on": "2026-06-29",
        "evidence_archive_url": (
            f"https://archive.example.test/web/20260629/{candidate_id}"
        ),
        "evidence_sha256": hashlib.sha256(
            f"evidence:{candidate_id}".encode("ascii")
        ).hexdigest(),
        "screening_locator": "Section 2; Algorithm 1",
        "exclusion_reason": "NR",
        "notes": "NR",
    }


def _write_forged_decision_snapshot(
    output: Path,
    row: dict[str, str],
    candidate_ids: list[str],
    assignment_ids: list[str],
) -> None:
    output.mkdir(mode=0o755)
    output.chmod(0o755)
    _write_csv(output / "decision.csv", DECISION_HEADER, [row])
    (output / "decision.csv").chmod(0o644)
    (output / "candidate_ids.txt").write_bytes(
        "".join(f"{value}\n" for value in candidate_ids).encode("utf-8")
    )
    (output / "candidate_ids.txt").chmod(0o644)
    (output / "assignment_ids.txt").write_bytes(
        "".join(f"{value}\n" for value in assignment_ids).encode("utf-8")
    )
    (output / "assignment_ids.txt").chmod(0o644)
    decision_sha256 = hashlib.sha256(
        (output / "decision.csv").read_bytes()
    ).hexdigest()
    candidate_ids_sha256 = hashlib.sha256(
        (output / "candidate_ids.txt").read_bytes()
    ).hexdigest()
    assignment_ids_sha256 = hashlib.sha256(
        (output / "assignment_ids.txt").read_bytes()
    ).hexdigest()
    snapshot_sha256 = _canonical_sha256(
        {
            "assignment_ids_file_sha256": assignment_ids_sha256,
            "calibration_result_snapshot_sha256": row[
                "calibration_result_snapshot_sha256"
            ],
            "candidate_ids_file_sha256": candidate_ids_sha256,
            "coordinator_snapshot_sha256": row[
                "coordinator_snapshot_sha256"
            ],
            "decision_file_sha256": decision_sha256,
            "decision_id": row["decision_id"],
            "manifest_version": screening_results.MANIFEST_VERSION,
            "protocol_sha256": row["protocol_sha256"],
            "row_count": 1,
        }
    )
    _write_csv(
        output / "manifest.csv",
        DECISION_MANIFEST_HEADER,
        [
            {
                "manifest_version": screening_results.MANIFEST_VERSION,
                "calibration_decision_snapshot_sha256": snapshot_sha256,
                "protocol_sha256": row["protocol_sha256"],
                "coordinator_snapshot_sha256": row[
                    "coordinator_snapshot_sha256"
                ],
                "calibration_result_snapshot_sha256": row[
                    "calibration_result_snapshot_sha256"
                ],
                "decision_id": row["decision_id"],
                "decision_file_sha256": decision_sha256,
                "candidate_ids_file_sha256": candidate_ids_sha256,
                "assignment_ids_file_sha256": assignment_ids_sha256,
                "row_count": "1",
            }
        ],
    )
    (output / "manifest.csv").chmod(0o644)
    checksum_names = (
        "assignment_ids.txt",
        "candidate_ids.txt",
        "decision.csv",
        "manifest.csv",
    )
    (output / "SHA256SUMS").write_bytes(
        "".join(
            f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
            for name in checksum_names
        ).encode("utf-8")
    )
    (output / "SHA256SUMS").chmod(0o644)


def _write_decision_snapshot(
    coordinator: Path,
    output: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    manifest = _read_csv(coordinator / "manifest.csv")
    calibration = [row for row in manifest if row["phase"] == "calibration"]
    taxonomy = json.loads(
        (coordinator / "taxonomy.json").read_text(encoding="utf-8")
    )
    inclusion_criterion = _fixture_inclusion_criterion(taxonomy)

    release_output = (
        output.parent / "calibration-reviewer-releases" / output.name
    )
    release_output.parent.mkdir()
    release(coordinator, "calibration", release_output)

    raw_root = output.parent / "raw-calibration-results" / output.name
    raw_root.mkdir(parents=True)
    result_paths: list[Path] = []
    for batch_id in screening_results.BATCH_IDS:
        result_path = raw_root / f"{batch_id}.csv"
        rows = [
            _calibration_result_row(row, inclusion_criterion)
            for row in calibration
            if row["batch_id"] == batch_id
        ]
        rows.sort(key=lambda row: row["assignment_id"].encode("utf-8"))
        _write_csv(result_path, screening_results.RESULT_HEADER, rows)
        result_paths.append(result_path)

    result_output = output.parent / "calibration-results" / output.name
    result_output.parent.mkdir()
    screening_results.seal_phase_results(
        coordinator_snapshot_dir=coordinator,
        phase="calibration",
        result_paths=result_paths,
        output_dir=result_output,
        reviewer_release_snapshot_dir=release_output,
    )
    sealed = screening_results.validate_phase_result_snapshot(
        result_output,
        coordinator_snapshot_dir=coordinator,
        reviewer_release_snapshot_dir=release_output,
    )
    candidate_ids = [
        row["candidate_id"]
        for row in _read_csv(coordinator / "calibration_selection.csv")
    ]
    assert set(candidate_ids) == {row["candidate_id"] for row in sealed.rows}
    assignment_ids = sorted(
        (row["assignment_id"] for row in sealed.rows),
        key=lambda value: value.encode("utf-8"),
    )
    by_candidate = {
        candidate_id: [
            row
            for row in sealed.rows
            if row["candidate_id"] == candidate_id
        ]
        for candidate_id in candidate_ids
    }
    numerator = sum(
        len({row["screening_status"] for row in rows}) == 1
        for rows in by_candidate.values()
    )
    decision = {
        "decision_id": "calibration-release-v1",
        "protocol_sha256": sealed.protocol_sha256,
        "coordinator_snapshot_sha256": sealed.coordinator_snapshot_sha256,
        "calibration_result_snapshot_sha256": sealed.snapshot_sha256,
        "candidate_ids_sha256": screening_results.sequence_ids_sha256(
            candidate_ids
        ),
        "assignment_ids_sha256": screening_results.ordered_ids_sha256(
            assignment_ids
        ),
        "status_agreement_numerator": str(numerator),
        "status_agreement_denominator": str(len(candidate_ids)),
        "status_agreement": screening_results.canonical_ratio(
            numerator, len(candidate_ids)
        ),
        "systematic_ambiguity": "false",
        "decision": "release",
        "decided_on": "2026-06-30",
        "decision_makers": "accountable-author;methodologist-a;methodologist-b",
        "resolution_evidence": (
            "Both locked calibration rating pairs were reviewed against "
            "every protocol boundary before release."
        ),
    }
    decision.update(overrides or {})

    if overrides:
        _write_forged_decision_snapshot(
            output,
            decision,
            candidate_ids,
            assignment_ids,
        )
    else:
        decision_input = output.parent / f"{output.name}-decision-input.csv"
        _write_csv(decision_input, DECISION_HEADER, [decision])
        screening_results.seal_calibration_decision(
            coordinator_snapshot_dir=coordinator,
            calibration_result_snapshot_dir=result_output,
            decision_input_path=decision_input,
            output_dir=output,
            calibration_reviewer_release_snapshot_dir=release_output,
        )
    return release_output, result_output, output


def _identical_path_swap(snapshot: Path):
    replacement = snapshot.with_name(f".{snapshot.name}.replacement")
    parked = snapshot.with_name(f".{snapshot.name}.parked")
    shutil.copytree(snapshot, replacement)

    def swap() -> None:
        snapshot.rename(parked)
        replacement.rename(snapshot)

    return swap


def test_main_release_requires_valid_immutable_calibration_decision(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "gated-inputs")
    coordinator_root = tmp_path / "gated-coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    release_root = tmp_path / "gated-releases"
    release_root.mkdir()

    with pytest.raises(screening_batches.SnapshotError, match="requires.*decision"):
        release(snapshot, "main", release_root / "v1")

    decision_root = tmp_path / "decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(snapshot, decision_root / "v1")
    release(snapshot, "main", release_root / "v2", gate)
    released = [
        row
        for number in range(1, 7)
        for row in _read_csv(
            release_root / "v2" / "packets" / f"screening-{number:02d}.csv"
        )
    ]
    assert len(released) == 344
    assert {row["phase"] for row in released} == {"main"}


def test_main_release_output_is_disjoint_from_gate_snapshots(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "overlap-inputs")
    coordinator_root = tmp_path / "overlap-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    decision_root = tmp_path / "overlap-decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(coordinator, decision_root / "v1")

    for label, protected in zip(
        ("calibration-release", "result", "decision"),
        gate,
        strict=True,
    ):
        alias = tmp_path / f"{label}-snapshot-alias"
        alias.symlink_to(protected, target_is_directory=True)
        cases = {
            "equal": protected,
            "ancestor": protected.parent,
            "descendant": protected / "v7",
            "alias": alias / "v8",
        }
        before = _snapshot_state(protected)
        for relation, output in cases.items():
            with pytest.raises(
                screening_batches.SnapshotError,
                match="alias|disjoint|overlap",
            ):
                release(coordinator, "main", output, gate)
            assert _snapshot_state(protected) == before, (label, relation)
            if relation in {"descendant", "alias"}:
                assert not os.path.lexists(output)


@pytest.mark.parametrize("boundary", ["pre_publish", "post_publish"])
@pytest.mark.parametrize(
    "target_name",
    ["coordinator", "calibration_release", "result", "decision"],
)
def test_main_release_rejects_same_byte_path_swap_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    target_name: str,
) -> None:
    inputs = build_inputs(tmp_path / "race-inputs")
    coordinator_root = tmp_path / "race-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    decision_root = tmp_path / "race-decisions"
    decision_root.mkdir()
    calibration_release, result_snapshot, decision_snapshot = (
        _write_decision_snapshot(
            coordinator, decision_root / "v1"
        )
    )
    target = {
        "coordinator": coordinator,
        "calibration_release": calibration_release,
        "result": result_snapshot,
        "decision": decision_snapshot,
    }[target_name]
    swap = _identical_path_swap(target)
    triggered = False

    if boundary == "pre_publish":
        build = screening_batches.build_reviewer_release_artifacts

        def build_then_swap(*args, **kwargs):
            nonlocal triggered
            artifacts = build(*args, **kwargs)
            triggered = True
            swap()
            return artifacts

        monkeypatch.setattr(
            screening_batches,
            "build_reviewer_release_artifacts",
            build_then_swap,
        )
    else:
        rename = screening_batches._rename_noreplace_at

        def publish_then_swap(*args, **kwargs) -> None:
            nonlocal triggered
            rename(*args, **kwargs)
            if not triggered:
                triggered = True
                swap()

        monkeypatch.setattr(
            screening_batches,
            "_rename_noreplace_at",
            publish_then_swap,
        )

    release_root = tmp_path / "race-releases"
    release_root.mkdir()
    output = release_root / "v1"
    with pytest.raises(screening_batches.SnapshotError):
        release(
            coordinator,
            "main",
            output,
            (
                calibration_release,
                result_snapshot,
                decision_snapshot,
            ),
        )
    assert triggered
    assert not os.path.lexists(output)


def test_calibration_release_rejects_coordinator_replacement_after_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = build_inputs(tmp_path / "calibration-race-inputs")
    coordinator_root = tmp_path / "calibration-race-coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v1"
    freeze(inputs, coordinator)
    swap = _identical_path_swap(coordinator)
    rename = screening_batches._rename_noreplace_at
    triggered = False

    def publish_then_swap(*args, **kwargs) -> None:
        nonlocal triggered
        rename(*args, **kwargs)
        if not triggered:
            triggered = True
            swap()

    monkeypatch.setattr(
        screening_batches,
        "_rename_noreplace_at",
        publish_then_swap,
    )
    release_root = tmp_path / "calibration-race-releases"
    release_root.mkdir()
    output = release_root / "v1"

    with pytest.raises(
        screening_batches.SnapshotError,
        match=(
            "coordinator.*changed|snapshot tree changed|"
            "coordinator.*does not match authoritative disk capture"
        ),
    ):
        release(coordinator, "calibration", output)

    assert triggered
    assert not os.path.lexists(output)


def test_calibration_release_rejects_decision_gate_argument(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "calibration-gate-inputs")
    coordinator_root = tmp_path / "calibration-gate-coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    decision_root = tmp_path / "calibration-gate-decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(snapshot, decision_root / "v1")
    release_root = tmp_path / "calibration-gate-releases"
    release_root.mkdir()

    with pytest.raises(screening_batches.SnapshotError, match="calibration.*rejects|does not accept"):
        release(snapshot, "calibration", release_root / "v1", gate)
    assert not os.path.lexists(release_root / "v1")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"protocol_sha256": "0" * 64}, "protocol"),
        ({"coordinator_snapshot_sha256": "0" * 64}, "coordinator"),
        ({"candidate_ids_sha256": "0" * 64}, "candidate"),
        ({"assignment_ids_sha256": "0" * 64}, "assignment"),
        ({"status_agreement_denominator": "29"}, "denominator"),
        ({"status_agreement_numerator": "23"}, "agreement"),
        ({"status_agreement": "0.79"}, "agreement"),
        ({"systematic_ambiguity": "yes"}, "ambiguity"),
        ({"decision": "revise"}, "decision"),
    ],
)
def test_main_release_rejects_invalid_calibration_decision_semantics(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    inputs = build_inputs(tmp_path / "invalid-gate-inputs")
    coordinator_root = tmp_path / "invalid-gate-coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    decision_root = tmp_path / "invalid-gate-decisions"
    decision_root.mkdir()
    gate = _write_decision_snapshot(
        snapshot, decision_root / "v1", overrides=overrides
    )
    release_root = tmp_path / "invalid-gate-releases"
    release_root.mkdir()

    with pytest.raises(screening_batches.SnapshotError, match=message):
        release(snapshot, "main", release_root / "v1", gate)
    assert not os.path.lexists(release_root / "v1")


@pytest.mark.parametrize("tamper", ["decision", "manifest", "checksums", "extra"])
def test_main_release_rejects_nonexact_decision_snapshot(
    tmp_path: Path,
    tamper: str,
) -> None:
    inputs = build_inputs(tmp_path / "decision-integrity-inputs")
    coordinator_root = tmp_path / "decision-integrity-coordinator"
    coordinator_root.mkdir()
    snapshot = coordinator_root / "v1"
    freeze(inputs, snapshot)
    decision_root = tmp_path / "decision-integrity-decisions"
    decision_root.mkdir()
    calibration_gate = _write_decision_snapshot(
        snapshot, decision_root / "v1"
    )
    gate = calibration_gate[2]
    if tamper == "decision":
        path = gate / "decision.csv"
        path.write_bytes(path.read_bytes() + b"tamper")
    elif tamper == "manifest":
        path = gate / "manifest.csv"
        path.write_bytes(path.read_bytes().replace(b"1,", b"2,", 1))
    elif tamper == "checksums":
        path = gate / "SHA256SUMS"
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        (gate / "extra.txt").write_text("extra\n", encoding="utf-8")
    release_root = tmp_path / "decision-integrity-releases"
    release_root.mkdir()

    with pytest.raises(screening_batches.SnapshotError):
        release(
            snapshot, "main", release_root / "v1", calibration_gate
        )
    assert not os.path.lexists(release_root / "v1")


def test_snapshot_cleanup_reports_expected_root_moved_before_precheck(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v1"
    moved_root = tmp_path / "moved-v1"
    root.mkdir()
    (root / "marker.txt").write_bytes(b"expected root\n")
    root_status = root.stat()
    root_identity = (root_status.st_dev, root_status.st_ino)
    parent_fd = os.open(tmp_path, screening_batches._DIRECTORY_OPEN_FLAGS)
    root_fd = os.open(root, screening_batches._DIRECTORY_OPEN_FLAGS)
    root.rename(moved_root)
    try:
        with pytest.raises(screening_batches.SnapshotError) as raised:
            screening_batches._capture_snapshot_root_at(
                parent_fd,
                root.name,
                {".": root_identity},
                root_fd=root_fd,
            )
    finally:
        os.close(root_fd)
        os.close(parent_fd)

    diagnostic = _exception_text(raised.value)
    assert f"(dev, ino)=({root_identity[0]}, {root_identity[1]})" in diagnostic
    assert str(moved_root) in diagnostic
    assert (moved_root / "marker.txt").read_bytes() == b"expected root\n"
    assert not root.exists()


def test_python310_publisher_cleanup_detail_does_not_mask_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyPrimaryError(RuntimeError):
        def __getattribute__(self, name):
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    output = tmp_path / "v1"
    primary = LegacyPrimaryError("primary publication failure")
    quarantine = tmp_path / ".trackgen-retired-legacy"
    recovery_identity = (41, 42)

    def fail_after_publish() -> None:
        raise primary

    def fail_cleanup(*_args, **_kwargs) -> None:
        raise screening_batches.SnapshotError(
            f"recovery at {quarantine}; "
            f"(dev, ino)=({recovery_identity[0]}, {recovery_identity[1]})"
        )

    monkeypatch.setattr(
        screening_batches,
        "_capture_snapshot_root_at",
        fail_cleanup,
    )

    with pytest.raises(LegacyPrimaryError) as raised:
        screening_batches._publish_artifacts(
            output,
            {"manifest.csv": b"immutable\n"},
            post_publish_check=fail_after_publish,
        )

    assert raised.value is primary
    assert "primary publication failure" in str(raised.value)
    assert str(quarantine) in str(raised.value)
    assert "(dev, ino)=(41, 42)" in str(raised.value)


def _evidence_packet_bytes(rows: list[dict[str, str]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=EVIDENCE_PACKET_HEADER,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _evidence_packet_row(
    *,
    candidate_id: str = "C0001",
    artifact_id: str = "paper-v1",
    local_filename: str = "papers/paper.pdf",
    evidence_sha256: str | None = None,
    redistribution_status: str = "local-restricted",
) -> dict[str, str]:
    payload = b"attested evidence bytes\n"
    return {
        "candidate_id": candidate_id,
        "artifact_id": artifact_id,
        "artifact_role": "primary-report",
        "source_url": "https://example.test/source",
        "evidence_version": "v1.0",
        "evidence_retrieved_on": "2026-07-02",
        "access_status": "full_text",
        "evidence_archive_url": "https://archive.example.test/paper-v1.pdf",
        "evidence_sha256": evidence_sha256 or hashlib.sha256(payload).hexdigest(),
        "local_filename": local_filename,
        "redistribution_status": redistribution_status,
        "retrieval_notes": "Retrieved from the publisher archive.",
    }


def _write_evidence_archive(root: Path) -> Path:
    archive = root / "source-archive"
    (archive / "papers").mkdir(parents=True)
    (archive / "papers" / "paper.pdf").write_bytes(b"attested evidence bytes\n")
    return archive


def _phase_evidence_inputs(
    root: Path,
    coordinator: Path,
    phase: str,
) -> tuple[Path, Path]:
    """Create one locally attested artifact for every candidate in a phase."""

    archive = root / f"{phase}-source-archive"
    manifest = root / f"{phase}-evidence-packet.csv"
    rows: list[dict[str, str]] = []
    candidate_ids = sorted(
        {
            row["candidate_id"]
            for filename in screening_batches.PACKET_FILENAMES
            for row in _read_csv(coordinator / "packets" / filename)
            if row["phase"] == phase
        }
    )
    for candidate_id in candidate_ids:
        payload = f"attested evidence for {candidate_id}\n".encode("ascii")
        relative = f"evidence/{candidate_id}.txt"
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        rows.append(
            _evidence_packet_row(
                candidate_id=candidate_id,
                artifact_id="primary-v1",
                local_filename=relative,
                evidence_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    manifest.write_bytes(_evidence_packet_bytes(rows))
    return manifest, archive


def test_binary_calibration_release_requires_and_binds_phase_evidence(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    coordinator_root = tmp_path / "coordinator"
    coordinator_root.mkdir()
    coordinator = coordinator_root / "v7"
    freeze(inputs, coordinator)
    output_root = tmp_path / "releases"
    output_root.mkdir()

    with pytest.raises(screening_batches.SnapshotError, match="evidence"):
        screening_batches.release_snapshot(
            coordinator, "calibration", output_root / "v1"
        )

    evidence_manifest, source_archive = _phase_evidence_inputs(
        tmp_path, coordinator, "calibration"
    )
    screening_batches.release_snapshot(
        coordinator,
        "calibration",
        output_root / "v2",
        evidence_manifest=evidence_manifest,
        source_archive=source_archive,
    )

    release_manifest = _read_csv(output_root / "v2" / "release_manifest.csv")
    assert release_manifest[0]["manifest_version"] == "3"
    assert (output_root / "v2" / "evidence_packet_manifest.csv").read_bytes() == (
        evidence_manifest.read_bytes()
    )
    assert release_manifest[0]["evidence_artifact_count"] == "30"


def test_release_v2_preserves_historical_v1_release_creation(
    tmp_path: Path,
) -> None:
    coordinator = DATA_ROOT / "screening_inputs" / "v5"
    artifacts = screening_batches.build_reviewer_release_artifacts(
        screening_batches.validate_snapshot(coordinator), "calibration"
    )

    assert set(artifacts) == {
        "protocol.md",
        "execution_profile.json",
        "reviewer_prompt_template.md",
        "release_manifest.csv",
        "SHA256SUMS",
        *(f"packets/{filename}" for filename in screening_batches.PACKET_FILENAMES),
    }
    manifest = screening_batches._read_csv_bytes(
        artifacts["release_manifest.csv"],
        "release_manifest.csv",
        RELEASE_MANIFEST_HEADER,
    )
    assert tuple(manifest[0]) == RELEASE_MANIFEST_HEADER
    assert manifest[0]["manifest_version"] == screening_batches.MANIFEST_VERSION


@pytest.mark.parametrize("operation", ("missing", "extra"))
def test_release_v2_rejects_nonexact_phase_evidence_coverage(
    tmp_path: Path, operation: str
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    root = tmp_path / "coordinator"
    root.mkdir()
    coordinator = root / "v7"
    freeze(inputs, coordinator)
    manifest, archive = _phase_evidence_inputs(tmp_path, coordinator, "calibration")
    rows = _read_csv(manifest)
    if operation == "missing":
        rows.pop()
    else:
        main_id = next(
            row["candidate_id"]
            for row in _read_csv(coordinator / "packets" / "screening-01.csv")
            if row["phase"] == "main"
        )
        extra = _evidence_packet_row(candidate_id=main_id, artifact_id="extra-v1")
        rows.append(extra)
        rows.sort(key=lambda row: (row["candidate_id"], row["artifact_id"]))
    manifest.write_bytes(_evidence_packet_bytes(rows))
    releases = tmp_path / "releases"
    releases.mkdir()

    with pytest.raises(screening_batches.SnapshotError, match="candidate coverage|unknown candidate"):
        screening_batches.release_snapshot(
            coordinator,
            "calibration",
            releases / "v2",
            evidence_manifest=manifest,
            source_archive=archive,
        )


@pytest.mark.parametrize("operation", ("mutate", "remove", "add", "forged-count"))
def test_release_v2_validation_rejects_evidence_manifest_tampering(
    tmp_path: Path, operation: str
) -> None:
    inputs = build_inputs(tmp_path / "inputs")
    root = tmp_path / "coordinator"
    root.mkdir()
    coordinator = root / "v7"
    freeze(inputs, coordinator)
    manifest, archive = _phase_evidence_inputs(tmp_path, coordinator, "calibration")
    releases = tmp_path / "releases"
    releases.mkdir()
    release = releases / "v2"
    screening_batches.release_snapshot(
        coordinator,
        "calibration",
        release,
        evidence_manifest=manifest,
        source_archive=archive,
    )
    evidence = release / "evidence_packet_manifest.csv"
    if operation == "mutate":
        evidence.write_bytes(evidence.read_bytes().replace(b"primary-v1", b"primary-v2", 1))
    elif operation == "remove":
        evidence.unlink()
    elif operation == "add":
        (release / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    else:
        rows = _read_csv(release / "release_manifest.csv")
        rows[0]["evidence_artifact_count"] = "31"
        release_manifest_header = tuple(rows[0])
        _write_csv(release / "release_manifest.csv", release_manifest_header, rows)

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.validate_reviewer_release_snapshot(
            release,
            expected_manifest=screening_batches._release_manifest_row(
                phase="calibration",
                coordinator_snapshot_sha256=_read_csv(coordinator / "manifest.csv")[0]["snapshot_sha256"],
                protocol_sha256=_read_csv(coordinator / "manifest.csv")[0]["protocol_sha256"],
                execution_profile_sha256=_read_csv(coordinator / "manifest.csv")[0]["execution_profile_sha256"],
                prompt_template_sha256=_read_csv(coordinator / "manifest.csv")[0]["prompt_template_sha256"],
            ),
            coordinator_snapshot=screening_batches.validate_snapshot(coordinator),
        )


def _complete_limited_access_notes(final_outcome: str) -> str:
    return (
        "attempted: "
        "doi_or_publisher=DOI and publisher returned metadata only | "
        "title_author=exact title-author search found no manuscript | "
        "scholarly_index_or_repository=scholarly repository search found no copy | "
        "official_page=not applicable after source-type review; "
        f"outcome: {final_outcome}"
    )


def test_evidence_packet_manifest_requires_exact_canonical_bytes_header_and_order(
    tmp_path: Path,
) -> None:
    assert screening_batches.EVIDENCE_PACKET_HEADER == EVIDENCE_PACKET_HEADER
    archive = _write_evidence_archive(tmp_path)
    first = _evidence_packet_row()
    second = _evidence_packet_row(
        candidate_id="C0002", artifact_id="supplement-v1"
    )
    payload = _evidence_packet_bytes([first, second])

    rows = screening_batches.parse_evidence_packet_manifest(
        payload,
        allowed_candidate_ids={"C0001", "C0002"},
        source_archive=archive,
    )

    assert rows == [first, second]
    for invalid in (
        _evidence_packet_bytes([]),
        payload.replace(b"\n", b"\r\n"),
        payload.rstrip(b"\n"),
        payload.replace(b"artifact_id", b"artifact", 1),
        _evidence_packet_bytes([second, first]),
    ):
        with pytest.raises(screening_batches.SnapshotError):
            screening_batches.parse_evidence_packet_manifest(
                invalid,
                allowed_candidate_ids={"C0001", "C0002"},
                source_archive=archive,
            )


def test_evidence_packet_manifest_rejects_duplicate_and_unknown_candidate(
    tmp_path: Path,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    duplicate = _evidence_packet_row()
    duplicate["artifact_role"] = "supplement"
    unknown = _evidence_packet_row(candidate_id="C9999")

    for rows in ([duplicate, duplicate], [unknown]):
        with pytest.raises(screening_batches.SnapshotError):
            screening_batches.parse_evidence_packet_manifest(
                _evidence_packet_bytes(rows),
                allowed_candidate_ids={"C0001"},
                source_archive=archive,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", "../paper"),
        ("artifact_role", "role/name"),
        ("source_url", "https://Example.test/source"),
        ("evidence_version", "NR"),
        ("evidence_retrieved_on", "2026-2-07"),
        ("access_status", "downloaded"),
        ("evidence_archive_url", "ftp://archive.example.test/paper"),
        ("redistribution_status", "restricted"),
        ("retrieval_notes", "x"),
    ],
)
def test_evidence_packet_manifest_rejects_invalid_field_values(
    tmp_path: Path, field: str, value: str
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row()
    row[field] = value

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    ("archive_url", "evidence_sha256", "local_filename", "redistribution_status"),
    [
        (
            "https://archive.example.test/paper.pdf",
            "NR",
            "NR",
            "metadata-only",
        ),
        (
            "https://archive.example.test/latest/paper.pdf",
            hashlib.sha256(b"attested evidence bytes\n").hexdigest(),
            "papers/paper.pdf",
            "local-restricted",
        ),
    ],
)
def test_evidence_packet_manifest_requires_version_pinned_archive_urls(
    tmp_path: Path,
    archive_url: str,
    evidence_sha256: str,
    local_filename: str,
    redistribution_status: str,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row(
        evidence_sha256=evidence_sha256,
        local_filename=local_filename,
        redistribution_status=redistribution_status,
    )
    row["evidence_archive_url"] = archive_url
    row["retrieval_notes"] = _complete_limited_access_notes(
        "publisher access was unavailable"
    )

    with pytest.raises(screening_batches.SnapshotError, match="version|mutable"):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"access_status": "abstract_only", "retrieval_notes": "N/A"},
        {
            "redistribution_status": "metadata-only",
            "evidence_sha256": "NR",
            "local_filename": "NR",
            "retrieval_notes": "N/A",
        },
        {"retrieval_notes": "access blocked"},
    ],
    ids=("abstract-only", "metadata-only", "blocked-access"),
)
def test_evidence_packet_manifest_requires_substantive_limited_access_notes(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row()
    row.update(overrides)

    with pytest.raises(screening_batches.SnapshotError, match="substantive"):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    "notes",
    [
        "Abstract available; full text unavailable.",
        "Retrieval attempted at the DOI and publisher; full text unavailable.",
        "attempted: ; outcome: publisher returned no full text",
        "attempted: DOI; outcome: blocked",
        "outcome: publisher access was blocked; attempted: DOI and repository search",
        "Attempted: DOI and publisher full-text retrieval; outcome: access was blocked",
        "attempted:  DOI and publisher full-text retrieval; outcome: access was blocked",
        (
            "attempted: DOI and publisher full-text retrieval; "
            "outcome: access was blocked; outcome: no local bytes"
        ),
    ],
    ids=(
        "generic",
        "missing-markers",
        "empty-attempted",
        "short-segments",
        "reversed-markers",
        "case-changed-marker",
        "untrimmed-segment",
        "extra-separator",
    ),
)
def test_evidence_packet_manifest_rejects_unstructured_limited_access_notes(
    tmp_path: Path,
    notes: str,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row()
    row["access_status"] = "abstract_only"
    row["retrieval_notes"] = notes

    with pytest.raises(screening_batches.SnapshotError, match="attempted.*outcome"):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    ("operation", "component"),
    [
        ("omit", "doi_or_publisher"),
        ("omit", "title_author"),
        ("omit", "scholarly_index_or_repository"),
        ("omit", "official_page"),
        ("short", "doi_or_publisher"),
        ("short", "title_author"),
        ("short", "scholarly_index_or_repository"),
        ("short", "official_page"),
        ("duplicate", "title_author"),
        ("reorder", "title_author"),
    ],
)
def test_evidence_packet_manifest_requires_each_retrieval_audit_component(
    tmp_path: Path,
    operation: str,
    component: str,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    components = [
        ["doi_or_publisher", "DOI and publisher returned metadata only"],
        ["title_author", "exact title-author search found no manuscript"],
        [
            "scholarly_index_or_repository",
            "scholarly repository search found no copy",
        ],
        ["official_page", "not applicable after source-type review"],
    ]
    index = next(
        index for index, (label, _) in enumerate(components) if label == component
    )
    if operation == "omit":
        components.pop(index)
    elif operation == "short":
        components[index][1] = "short"
    elif operation == "duplicate":
        components.insert(index, components[index].copy())
    elif operation == "reorder":
        components[index - 1], components[index] = (
            components[index],
            components[index - 1],
        )
    attempted = " | ".join(f"{label}={value}" for label, value in components)
    row = _evidence_packet_row()
    row["access_status"] = "abstract_only"
    row["retrieval_notes"] = (
        f"attempted: {attempted}; outcome: only an abstract record was available"
    )

    with pytest.raises(screening_batches.SnapshotError, match="doi_or_publisher"):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "access_status": "abstract_only",
            "retrieval_notes": _complete_limited_access_notes(
                "only the abstract record was available"
            ),
        },
        {
            "redistribution_status": "metadata-only",
            "evidence_sha256": "NR",
            "local_filename": "NR",
            "retrieval_notes": _complete_limited_access_notes(
                "no redistributable local bytes were available"
            ),
        },
        {
            "retrieval_notes": _complete_limited_access_notes(
                "publisher access was blocked by HTTP 403"
            ),
        },
    ],
    ids=("abstract-only", "metadata-only", "blocked-access"),
)
def test_evidence_packet_manifest_accepts_structured_limited_access_notes(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row()
    row.update(overrides)

    assert screening_batches.parse_evidence_packet_manifest(
        _evidence_packet_bytes([row]),
        allowed_candidate_ids={"C0001"},
        source_archive=archive,
    ) == [row]


def test_evidence_packet_manifest_allows_nr_notes_for_full_text(
    tmp_path: Path,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row()
    row["retrieval_notes"] = "NR"

    assert screening_batches.parse_evidence_packet_manifest(
        _evidence_packet_bytes([row]),
        allowed_candidate_ids={"C0001"},
        source_archive=archive,
    ) == [row]


@pytest.mark.parametrize(
    ("evidence_sha256", "local_filename"),
    [
        ("NR", "papers/paper.pdf"),
        (hashlib.sha256(b"attested evidence bytes\n").hexdigest(), "NR"),
    ],
)
def test_evidence_packet_manifest_rejects_invalid_nr_pairing(
    tmp_path: Path, evidence_sha256: str, local_filename: str
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row(
        evidence_sha256=evidence_sha256,
        local_filename=local_filename,
    )

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    ("local_filename", "evidence_sha256"),
    [
        ("papers/missing.pdf", hashlib.sha256(b"attested evidence bytes\n").hexdigest()),
        ("papers/paper.pdf", "0" * 64),
        ("/papers/paper.pdf", hashlib.sha256(b"attested evidence bytes\n").hexdigest()),
        ("../papers/paper.pdf", hashlib.sha256(b"attested evidence bytes\n").hexdigest()),
        ("papers\\paper.pdf", hashlib.sha256(b"attested evidence bytes\n").hexdigest()),
    ],
)
def test_evidence_packet_manifest_rejects_unattested_or_unsafe_local_bytes(
    tmp_path: Path, local_filename: str, evidence_sha256: str
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row(
        local_filename=local_filename,
        evidence_sha256=evidence_sha256,
    )

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


def test_evidence_packet_manifest_rejects_symlinked_local_bytes(
    tmp_path: Path,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    target = archive / "papers" / "paper.pdf"
    (archive / "linked").symlink_to(target)
    row = _evidence_packet_row(local_filename="linked/paper.pdf")

    with pytest.raises(screening_batches.SnapshotError, match="symlink"):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


@pytest.mark.parametrize(
    ("redistribution_status", "evidence_sha256", "local_filename"),
    [
        (
            "metadata-only",
            hashlib.sha256(b"attested evidence bytes\n").hexdigest(),
            "papers/paper.pdf",
        ),
        ("metadata-only", "NR", "papers/paper.pdf"),
    ],
)
def test_evidence_packet_manifest_enforces_metadata_only_constraints(
    tmp_path: Path,
    redistribution_status: str,
    evidence_sha256: str,
    local_filename: str,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row(
        redistribution_status=redistribution_status,
        evidence_sha256=evidence_sha256,
        local_filename=local_filename,
    )

    with pytest.raises(screening_batches.SnapshotError):
        screening_batches.parse_evidence_packet_manifest(
            _evidence_packet_bytes([row]),
            allowed_candidate_ids={"C0001"},
            source_archive=archive,
        )


def test_evidence_packet_manifest_accepts_metadata_only_without_local_bytes(
    tmp_path: Path,
) -> None:
    archive = _write_evidence_archive(tmp_path)
    row = _evidence_packet_row(
        evidence_sha256="NR",
        local_filename="NR",
        redistribution_status="metadata-only",
    )
    row["retrieval_notes"] = _complete_limited_access_notes(
        "full text was unavailable from all attempted routes"
    )

    assert screening_batches.parse_evidence_packet_manifest(
        _evidence_packet_bytes([row]),
        allowed_candidate_ids={"C0001"},
        source_archive=archive,
    ) == [row]


def test_validate_evidence_manifest_cli_reports_deterministic_counts(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, CANDIDATE_HEADER, [_candidate(1), _candidate(2)])
    archive = _write_evidence_archive(tmp_path)
    manifest = tmp_path / "evidence.csv"
    manifest.write_bytes(
        _evidence_packet_bytes(
            [
                _evidence_packet_row(),
                _evidence_packet_row(
                    candidate_id="C0002", artifact_id="supplement-v1"
                ),
            ]
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--validate-evidence-manifest",
            "--candidates",
            str(candidates),
            "--evidence-manifest",
            str(manifest),
            "--source-archive",
            str(archive),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "evidence manifest valid: candidates=2 artifacts=2\n"


@pytest.mark.parametrize(
    "mode_arguments",
    [
        ["--freeze"],
        ["--snapshot-dir", "snapshot"],
        [
            "--snapshot-dir",
            "snapshot",
            "--reviewer-release-snapshot",
            "release",
        ],
        [
            "--stage-role",
            "--snapshot-dir",
            "snapshot",
            "--reviewer-release-snapshot",
            "release",
            "--role-id",
            "screening-01",
            "--staging-root",
            "staging",
        ],
    ],
    ids=("freeze", "snapshot-validation", "release-validation", "stage-role"),
)
def test_validate_evidence_manifest_cli_arguments_are_isolated_to_their_mode(
    mode_arguments: list[str],
) -> None:
    with pytest.raises(
        screening_batches.SnapshotError,
        match="evidence-manifest|source-archive",
    ):
        screening_batches.main(
            [
                *mode_arguments,
                "--evidence-manifest",
                "evidence.csv",
                "--source-archive",
                "source-archive",
            ]
        )
