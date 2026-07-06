"""Create and validate the immutable Pass-2 reliability pilot snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class ReliabilityIntegrationError(ValueError):
    """The supplied reliability-pilot inputs cannot be integrated."""


class ReliabilityValidationError(ValueError):
    """The reliability-pilot snapshot does not match its fixed record."""


@dataclass(frozen=True)
class InputSpec:
    filename: str
    sha256: str
    row_count: int
    header: tuple[str, ...] | None


INPUT_SPECS = (
    InputSpec(
        "trackgen-pass2-reliability-selection.csv",
        "9ab6691b40a4e2a4e501a86ff71846963b77180eb9bbb45c936075587ed257b5",
        18,
        ("cite_key", "first_domain", "rank_sha256", "evidence_sha256"),
    ),
    InputSpec(
        "trackgen-pass2-reliability-packet.csv",
        "df555cbf65af08766c521819133ac050606c5e4df66faf8b6639424683b0de98",
        18,
        (
            "candidate_id",
            "cite_key",
            "title",
            "authors",
            "year",
            "venue",
            "doi",
            "url",
            "source_type",
        ),
    ),
    InputSpec(
        "trackgen-pass2-reliability-template.csv",
        "f0841bf803913b55dca3ba45d9e92970c1e84f74f028197a5a556ce87db260fd",
        18,
        (
            "cite_key",
            "survey_evidence_tier",
            "course_object",
            "representation_family",
            "generator_family",
            "generation_role",
            "validity_strategy",
            "code_status",
            "asset_status",
        ),
    ),
    InputSpec(
        "trackgen-pass2-reliability-coded.csv",
        "1b67aded975952d6c2f6e8245640a990b6ca1730038ac2c8f2619a6e2d9add81",
        18,
        (
            "cite_key",
            "survey_evidence_tier",
            "course_object",
            "representation_family",
            "generator_family",
            "generation_role",
            "validity_strategy",
            "code_status",
            "asset_status",
        ),
    ),
    InputSpec(
        "trackgen-pass2-reliability-primary-sample.csv",
        "b0fb3e2130254dc60d951af3ab73ce30e5e3e23dded11f705b8dbff1fe82e2ba",
        18,
        None,
    ),
    InputSpec(
        "trackgen-pass2-reliability-summary.csv",
        "fe7e19dbeac34e8e50b55ca3611fb74094b961a8e3dcbabbc32bd30b4f70726e",
        8,
        ("field", "n", "agreement", "kappa", "passes"),
    ),
    InputSpec(
        "trackgen-pass2-reliability-disagreements.csv",
        "96525d3d545665b3ce1b7c6d0239640567fa050bda10c62faa50910f018d5b73",
        42,
        None,
    ),
    InputSpec(
        "trackgen-pass2-reliability-adjudication.csv",
        "79aed3cc2bf792149edb2f8e94cd8fcc1912808245ee7f688859ea57be925a0a",
        42,
        None,
    ),
    InputSpec(
        "trackgen-pass2-codebook-review.md",
        "386e73d38a5a0d08446fb52f3383f88bf223e6df3c0dd8cbca7eef58b1ea3ea8",
        198,
        None,
    ),
)

SUMMARY_ROWS = (
    ("survey_evidence_tier", "18", "0.888889", "0.723077", "true"),
    ("course_object", "18", "0.944444", "0.935018", "true"),
    ("representation_family", "18", "0.444444", "0.372822", "false"),
    ("generator_family", "18", "0.666667", "0.597015", "false"),
    ("generation_role", "18", "0.444444", "0.138756", "false"),
    ("validity_strategy", "18", "0.500000", "0.325000", "false"),
    ("code_status", "18", "0.833333", "0.689655", "true"),
    ("asset_status", "18", "0.944444", "0.857143", "true"),
)

PRIMARY_RELATIVE = Path("paper/data/screening_work/v8/pass2_coding/primary/v1")
DRAFT_RELATIVE = Path("paper/data/screening_work/v8/pass2_drafts/v1")
MANIFEST_HEADER = ("record_type", "path", "sha256", "row_count")
REGISTRY_HEADER = (
    "role",
    "agent_id",
    "model",
    "reasoning_effort",
    "fork_context",
    "scope",
)
BINDINGS_HEADER = ("binding", "bound_path", "bound_sha256", "purpose")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_bytes(path: Path, *, label: str, error_type: type[ValueError]) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise error_type(f"{label}: must be a regular non-symlink file")
        return path.read_bytes()
    except OSError as exc:
        raise error_type(f"{label}: unable to read {path}") from exc


def _csv_rows(path: Path, *, error_type: type[ValueError]) -> list[dict[str, str]]:
    payload = _regular_bytes(path, label="CSV file", error_type=error_type)
    try:
        return list(csv.DictReader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise error_type(f"{path}: invalid CSV") from exc


def _csv_header(path: Path, *, error_type: type[ValueError]) -> tuple[str, ...]:
    payload = _regular_bytes(path, label="CSV file", error_type=error_type)
    try:
        header = next(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (StopIteration, UnicodeDecodeError, csv.Error) as exc:
        raise error_type(f"{path}: missing CSV header") from exc
    return tuple(header)


def _write_csv(path: Path, header: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _input_path(input_root: Path, spec: InputSpec) -> Path:
    return input_root / spec.filename


def _validate_source_input(input_root: Path, spec: InputSpec) -> bytes:
    path = _input_path(input_root, spec)
    payload = _regular_bytes(path, label="source input", error_type=ReliabilityIntegrationError)
    if _sha256(payload) != spec.sha256:
        raise ReliabilityIntegrationError(f"source input digest mismatch: {spec.filename}")
    if spec.header is not None:
        if _csv_header(path, error_type=ReliabilityIntegrationError) != spec.header:
            raise ReliabilityIntegrationError(f"source input header mismatch: {spec.filename}")
        if len(_csv_rows(path, error_type=ReliabilityIntegrationError)) != spec.row_count:
            raise ReliabilityIntegrationError(f"source input row count mismatch: {spec.filename}")
    return payload


def _registry_rows() -> tuple[dict[str, str], ...]:
    return (
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
    )


def _binding_rows(repository_root: Path) -> tuple[dict[str, str], ...]:
    primary = repository_root / PRIMARY_RELATIVE / "manifest/checksums.csv"
    draft = repository_root / DRAFT_RELATIVE / "release_manifest.csv"
    return (
        {
            "binding": "primary_snapshot",
            "bound_path": (PRIMARY_RELATIVE / "manifest/checksums.csv").as_posix(),
            "bound_sha256": _sha256(
                _regular_bytes(primary, label="primary snapshot binding", error_type=ReliabilityIntegrationError)
            ),
            "purpose": "binds the sampled primary coding snapshot without altering it",
        },
        {
            "binding": "draft_release",
            "bound_path": (DRAFT_RELATIVE / "release_manifest.csv").as_posix(),
            "bound_sha256": _sha256(
                _regular_bytes(draft, label="draft release binding", error_type=ReliabilityIntegrationError)
            ),
            "purpose": "binds the non-final pass2 draft release without altering it",
        },
    )


def _readme() -> str:
    return """# Pass-2 Reliability Pilot v1

This no-clobber snapshot preserves the completed 18-source Pass-2 reliability pilot and its follow-up adjudication as supplied. All files in `inputs/` are byte-for-byte copies of the coordinator inputs. `execution_registry.csv` records the supplied reliability-coder, source-adjudicator, and methods-reviewer metadata. `bindings.csv` binds this record to `pass2_coding/primary/v1` and the non-final `pass2_drafts/v1` release; neither bound release is modified.

## Pilot outcome

The pilot failed all four required exact-set gates:

- `representation_family: 8/18 (0.444) - FAIL`
- `generator_family: 12/18 (0.667) - FAIL`
- `generation_role: 8/18 (0.444) - FAIL`
- `validity_strategy: 9/18 (0.500) - FAIL`

The pilot therefore cannot support final prevalence/taxonomy claims. The passed diagnostic fields do not change this conclusion. `PROCEDURAL-LIMITATIONS.md` records the restrictions; `CODEBOOK-v2.md` is prospective and contains no source-specific answer keys.

## Integrity

`manifest/checksums.csv` records SHA-256 hashes and row counts for each copied input and generated artifact. `SHA256SUMS` additionally checks the manifest itself. Run `python paper/scripts/integrate_pass2_reliability.py --repository-root . --validate --snapshot paper/data/screening_work/v8/pass2_reliability/pilot-v1 --input-root /tmp` from the repository root to validate this record.
"""


def _limitations() -> str:
    return """# Procedural Limitations

This is a failed, 18-source reliability pilot, not a final coding release or an empirical basis for prevalence, taxonomy, or comparative claims. Its purpose is to preserve the observed disagreement pattern and motivate a prospective codebook revision.

The copied selection, packet, template, reliability coding, primary sample, summary, disagreement, and adjudication files are historical inputs. Adjudication records source-level resolutions only; they are not a replacement for blind agreement and must not be used as answer keys for a future reliability round.

The next round must first recode all 75 sources under frozen `CODEBOOK-v2.md`, then run fresh blind reliability using an independently drawn holdout. The draft release gate is exact-set >=0.80 for each of the eight fields (`survey_evidence_tier`, `course_object`, `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, `code_status`, and `asset_status`) and no repeated ambiguity class. This current draft does not require two consecutive 30-source rounds. Stronger pre-submission replication is recommended with a further fresh holdout.

The bound primary snapshot and draft release remain unchanged. This snapshot neither revises primary v1 values nor releases a final codebook or final taxonomy.
"""


def _codebook_v2() -> str:
    return """# Pass-2 Codebook v2 (Prospective Draft)

## Status and scope

This is a prospective, frozen-for-next-round decision codebook distilled from pilot disagreement patterns and independent review. It does not contain source-specific answer keys and does not revise the existing v1 release. Every non-missing label needs a source-addressable locator for the particular claim.

## Representation family

1. Code a representation only when it is source-native and course-defining: the source explicitly defines, emits, consumes, serializes, directly inspects, or releases it as course state, parameterization, or a reusable course artifact.
2. Do not infer a representation from a renderer mesh, simulator internals, physics shape, occupancy map, coordinate array, visualization, cache, import, export, or downstream conversion unless the source establishes that object as a course-defining representation.
3. Use multiple labels only when separate course-defining representations are each directly evidenced; each multi-label assignment requires a distinct locator. Do not add labels merely because a pipeline has stages. Use `hybrid` only for an explicitly composite representation whose definition requires the components together.
4. A fixed benchmark or simulator may receive a directly documented fixed-course representation. Otherwise use `NR`; do not infer undocumented substrate details.

## Generator family

1. `constructive` requires explicit rules, assembly, grammar, geometry construction, or parameter-to-course computation. Random initialization or random parameter values alone do not add `stochastic_procedural`.
2. `stochastic_procedural` requires a named random sampling or stochastic assembly step that determines alternative topology or geometry. It may combine with `constructive` only when both the constructor and the geometry-determining stochastic operation are directly evidenced.
3. `learned_generative` requires a trained model to output course state, parameters, or geometry. A learned controller is not a learned generator solely because it is evaluated on courses.
4. `environment_design` requires selection, adaptation, or optimization of an environment/course distribution using learner, agent, or task-performance feedback. Combine it with `learned_generative` only when both mechanisms are separately evidenced.
5. `human_designed` requires human course-defining layout decisions, not merely choosing a seed or inspecting output. It can combine with `constructive` for an evidenced authoring-plus-construction workflow.
6. `selection_replay` is retrieval, replay, permutation, or selection of already complete courses. Assembling new geometry from primitives is `constructive`.

## Generation role

1. `geometry_synthesis` creates new course geometry or course-defining spatial structure.
2. `mutation` requires an explicit operation that transforms an existing complete course into another candidate. It may combine with `geometry_synthesis` only if both the initial construction and whole-course mutation are directly evidenced.
3. `task_selection` chooses, weights, schedules, or adapts among already defined courses/tasks without changing their geometry.
4. `benchmark_only` means the source contributes a fixed course/benchmark for use or evaluation and establishes no source-native course-changing operation. It is mutually exclusive with `geometry_synthesis`, `mutation`, `repair`, and `task_selection` for that contribution.
5. `NR` means no source-native course-operation role is established. For analytical fields, `NR` is sole-valued and is not shorthand for reviewer uncertainty.

## Validity and missingness

1. `by_construction` requires an explicit rule, parameterization, or invariant that ensures the stated validity property before candidate testing.
2. `rejection` requires generation followed by an explicit test that discards failing candidates. Bounded sampling is `by_construction` when bounds guarantee validity without candidate-level discard.
3. `penalty`, `repair_projection`, and `constraint_solver` each require the corresponding mechanism. Use multiple validity labels only for separate evidenced stages or conditions.
4. `simulation_validation` requires a simulated run whose outcome assesses, accepts, rejects, or reports course feasibility or validity. Simulation used only to train or evaluate an agent is not enough.
5. `not_reported` applies when a source-native generation or selection contribution makes a field applicable but the frozen source does not state the mechanism. Its locator identifies the inspected generation/selection material.
6. `NR` is structural non-applicability: no source-native contribution exists to which the field applies. It is sole-valued for analytical fields.

## Core and supporting evidence

1. A source is `core` for these fields when it defines, implements, or releases a parameterized or stochastic mechanism that changes course geometry or course-defining spatial constraints.
2. A source is `supporting` when it establishes a fixed-course interface, benchmark, simulator constraint, or reporting practice only. Supporting evidence may code directly documented fixed-course representation or `benchmark_only`, but uses `NR` for an unestablished generator, course-changing role, and associated validity strategy.
3. The label `simulator` alone does not decide the evidence tier.

## Availability evidence

1. `official_open` requires a frozen, inspectable official release/repository/documentation locator connecting the artifact to the authors, project, publisher, or official organization, including revision or path and public access terms.
2. `unofficial_open` requires an inspectable third-party release explicitly linked to the work; it must not be attributed as an author release.
3. `closed` requires affirmative evidence of restricted, proprietary, request-only, or otherwise unavailable material. `not_found` requires the documented source-first availability check to find no qualifying release. `not_applicable` means no code or course asset could reasonably be released.
4. Preserve the repository rule: code_status/asset_status may be sole NR when availability was not assessed or not reported. This draft does not adopt the review suggestion that completed rows cannot use `NR`.

## Next-round protocol

1. Freeze this codebook, a concise decision table, and synthetic boundary examples. Discuss only the synthetic examples during calibration; do not use pilot sources or adjudications as answer keys.
2. Recode all 75 sources under frozen v2. Validate completed rows with the existing release validator before reliability comparison.
3. Draw a fresh blind holdout and require exact-set >=0.80 for each of the eight fields. The draft gate also requires no repeated ambiguity class after locked ratings; revise the codebook rather than adjudicating individual sources to manufacture agreement.
4. This draft does not require two consecutive 30-source rounds. Stronger pre-submission replication with an additional fresh blind holdout is recommended.
"""


def _line_count(payload: bytes) -> int:
    return len(payload.splitlines())


def _artifact_records(output: Path) -> list[dict[str, str]]:
    paths = [
        *(Path("inputs") / spec.filename for spec in INPUT_SPECS),
        Path("README.md"),
        Path("PROCEDURAL-LIMITATIONS.md"),
        Path("CODEBOOK-v2.md"),
        Path("execution_registry.csv"),
        Path("bindings.csv"),
    ]
    records: list[dict[str, str]] = []
    for relative in paths:
        payload = (output / relative).read_bytes()
        row_count = (
            len(_csv_rows(output / relative, error_type=ReliabilityIntegrationError))
            if relative.suffix == ".csv"
            else _line_count(payload)
        )
        records.append(
            {
                "record_type": "input" if relative.parts[0] == "inputs" else "artifact",
                "path": relative.as_posix(),
                "sha256": _sha256(payload),
                "row_count": str(row_count),
            }
        )
    return sorted(records, key=lambda row: row["path"])


def _write_sha256s(output: Path) -> None:
    records = _artifact_records(output)
    manifest = output / "manifest/checksums.csv"
    manifest_sha = _sha256(manifest.read_bytes())
    lines = [f"{row['sha256']}  {row['path']}" for row in records]
    lines.append(f"{manifest_sha}  manifest/checksums.csv")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def integrate_reliability_pilot(
    *, repository_root: Path, output: Path, input_root: Path
) -> None:
    """Create a new no-clobber pilot snapshot from the fixed coordinator inputs."""
    if output.exists():
        raise ReliabilityIntegrationError(f"output must not already exist: {output}")

    source_payloads = {
        spec.filename: _validate_source_input(input_root, spec) for spec in INPUT_SPECS
    }
    bindings = _binding_rows(repository_root)

    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    for spec in INPUT_SPECS:
        (inputs / spec.filename).write_bytes(source_payloads[spec.filename])

    (output / "README.md").write_text(_readme(), encoding="utf-8")
    (output / "PROCEDURAL-LIMITATIONS.md").write_text(_limitations(), encoding="utf-8")
    (output / "CODEBOOK-v2.md").write_text(_codebook_v2(), encoding="utf-8")
    _write_csv(output / "execution_registry.csv", REGISTRY_HEADER, _registry_rows())
    _write_csv(output / "bindings.csv", BINDINGS_HEADER, bindings)
    manifest_directory = output / "manifest"
    manifest_directory.mkdir()
    _write_csv(manifest_directory / "checksums.csv", MANIFEST_HEADER, _artifact_records(output))
    _write_sha256s(output)


def _validate_input_copy(snapshot: Path, spec: InputSpec) -> None:
    path = snapshot / "inputs" / spec.filename
    payload = _regular_bytes(path, label="snapshot input", error_type=ReliabilityValidationError)
    if _sha256(payload) != spec.sha256:
        raise ReliabilityValidationError(f"snapshot input checksum mismatch: {spec.filename}")
    if spec.header is not None:
        if _csv_header(path, error_type=ReliabilityValidationError) != spec.header:
            raise ReliabilityValidationError(f"snapshot input header mismatch: {spec.filename}")
        if len(_csv_rows(path, error_type=ReliabilityValidationError)) != spec.row_count:
            raise ReliabilityValidationError(f"snapshot input row count mismatch: {spec.filename}")


def _validate_summary(snapshot: Path) -> None:
    path = snapshot / "inputs/trackgen-pass2-reliability-summary.csv"
    rows = _csv_rows(path, error_type=ReliabilityValidationError)
    actual = tuple(
        (row["field"], row["n"], row["agreement"], row["kappa"], row["passes"])
        for row in rows
    )
    if actual != SUMMARY_ROWS:
        raise ReliabilityValidationError("reliability summary values mismatch")
    if sum(row["passes"] == "false" for row in rows) != 4:
        raise ReliabilityValidationError("reliability summary must contain four failed fields")


def _validate_bindings(repository_root: Path, snapshot: Path) -> None:
    path = snapshot / "bindings.csv"
    rows = _csv_rows(path, error_type=ReliabilityValidationError)
    if _csv_header(path, error_type=ReliabilityValidationError) != BINDINGS_HEADER:
        raise ReliabilityValidationError("bindings header mismatch")
    expected = _binding_rows_for_validation(repository_root)
    if rows != list(expected):
        raise ReliabilityValidationError("primary snapshot binding or draft release binding mismatch")


def _binding_rows_for_validation(repository_root: Path) -> tuple[dict[str, str], ...]:
    try:
        return _binding_rows(repository_root)
    except ReliabilityIntegrationError as exc:
        raise ReliabilityValidationError(str(exc)) from exc


def _validate_manifest(snapshot: Path) -> None:
    manifest = snapshot / "manifest/checksums.csv"
    if _csv_header(manifest, error_type=ReliabilityValidationError) != MANIFEST_HEADER:
        raise ReliabilityValidationError("manifest header mismatch")
    actual = _csv_rows(manifest, error_type=ReliabilityValidationError)
    expected = _artifact_records_for_validation(snapshot)
    if actual != expected:
        raise ReliabilityValidationError("manifest checksum or row count mismatch")
    sums = snapshot / "SHA256SUMS"
    expected_lines = [f"{row['sha256']}  {row['path']}" for row in expected]
    expected_lines.append(f"{_sha256(manifest.read_bytes())}  manifest/checksums.csv")
    if _regular_bytes(sums, label="SHA256SUMS", error_type=ReliabilityValidationError) != (
        "\n".join(expected_lines) + "\n"
    ).encode("utf-8"):
        raise ReliabilityValidationError("SHA256SUMS checksum mismatch")


def _artifact_records_for_validation(snapshot: Path) -> list[dict[str, str]]:
    try:
        return _artifact_records(snapshot)
    except ReliabilityIntegrationError as exc:
        raise ReliabilityValidationError(str(exc)) from exc


def _validate_docs(snapshot: Path) -> None:
    required = {
        "README.md": (
            "representation_family: 8/18 (0.444) - FAIL",
            "generator_family: 12/18 (0.667) - FAIL",
            "generation_role: 8/18 (0.444) - FAIL",
            "validity_strategy: 9/18 (0.500) - FAIL",
            "cannot support final prevalence/taxonomy claims",
        ),
        "PROCEDURAL-LIMITATIONS.md": (
            "recode all 75",
            "fresh blind reliability",
            "exact-set >=0.80 for each of the eight fields",
            "pre-submission replication is recommended with a further fresh holdout",
        ),
        "CODEBOOK-v2.md": (
            "source-native and course-defining",
            "Use multiple labels only",
            "stochastic_procedural",
            "Generation role",
            "not_reported",
            "core",
            "supporting",
            "availability",
            "code_status/asset_status may be sole NR",
        ),
    }
    for filename, needles in required.items():
        text = _regular_bytes(
            snapshot / filename, label=filename, error_type=ReliabilityValidationError
        ).decode("utf-8")
        if any(needle not in text for needle in needles):
            raise ReliabilityValidationError(f"{filename}: required content missing")
    codebook = (snapshot / "CODEBOOK-v2.md").read_text(encoding="utf-8")
    if "completed rows cannot use NR" in codebook:
        raise ReliabilityValidationError("CODEBOOK-v2.md adopts a rejected NR rule")


def _validate_expected_files(snapshot: Path) -> None:
    expected = {
        *(Path("inputs") / spec.filename for spec in INPUT_SPECS),
        Path("README.md"),
        Path("PROCEDURAL-LIMITATIONS.md"),
        Path("CODEBOOK-v2.md"),
        Path("execution_registry.csv"),
        Path("bindings.csv"),
        Path("manifest/checksums.csv"),
        Path("SHA256SUMS"),
    }
    actual = {path.relative_to(snapshot) for path in snapshot.rglob("*") if path.is_file()}
    if actual != expected:
        raise ReliabilityValidationError("snapshot must contain exactly the expected records")


def validate_reliability_pilot(
    *, repository_root: Path, snapshot: Path, input_root: Path
) -> None:
    """Validate the copied pilot inputs, fixed facts, bindings, and checksums."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ReliabilityValidationError("snapshot must be a directory")
    _validate_expected_files(snapshot)
    for spec in INPUT_SPECS:
        source = _input_path(input_root, spec)
        payload = _regular_bytes(source, label="source input", error_type=ReliabilityValidationError)
        if _sha256(payload) != spec.sha256:
            raise ReliabilityValidationError(f"source input checksum mismatch: {spec.filename}")
        _validate_input_copy(snapshot, spec)
        if payload != (snapshot / "inputs" / spec.filename).read_bytes():
            raise ReliabilityValidationError(f"snapshot input is not byte-exact: {spec.filename}")
    _validate_summary(snapshot)
    _validate_bindings(repository_root, snapshot)
    _validate_docs(snapshot)
    registry = _csv_rows(snapshot / "execution_registry.csv", error_type=ReliabilityValidationError)
    if _csv_header(snapshot / "execution_registry.csv", error_type=ReliabilityValidationError) != REGISTRY_HEADER or registry != list(_registry_rows()):
        raise ReliabilityValidationError("execution registry mismatch")
    _validate_manifest(snapshot)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)

    if args.validate:
        if args.snapshot is None or args.output is not None:
            parser.error("--validate requires --snapshot and does not accept --output")
        validate_reliability_pilot(
            repository_root=args.repository_root,
            snapshot=args.snapshot,
            input_root=args.input_root,
        )
        print(f"validated {args.snapshot}")
        return 0
    if args.output is None or args.snapshot is not None:
        parser.error("creation requires --output and does not accept --snapshot")
    integrate_reliability_pilot(
        repository_root=args.repository_root,
        output=args.output,
        input_root=args.input_root,
    )
    print(f"created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
