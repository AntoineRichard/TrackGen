from __future__ import annotations

import csv
import json
from pathlib import Path


class CorpusError(ValueError):
    pass


HEADERS = {
    "search_log.csv": (
        "search_id", "search_date", "stream", "agent", "query", "search_surface",
        "results_screened", "candidates_added", "notes",
    ),
    "candidates.csv": (
        "candidate_id", "cite_key", "title", "authors", "year", "venue", "doi",
        "url", "source_type", "discovery_stream", "discovery_query",
        "discovery_agent", "screening_status", "exclusion_reason",
        "metadata_status", "metadata_evidence",
    ),
    "seed_coverage.csv": (
        "source_path", "source_heading", "source_label", "candidate_id",
        "coverage_status", "notes",
    ),
    "evidence.csv": (
        "cite_key", "domain", "vehicle", "course_object", "representation_family",
        "generator_family", "generation_role", "validity_strategy",
        "geometry_metrics", "difficulty_metrics", "diversity_metrics",
        "training_distribution", "evaluation_suite", "simulator", "export_format",
        "code_status", "asset_status", "reproducibility_fields",
        "evidence_locator", "coding_notes",
    ),
    "claims.csv": (
        "claim_id", "section", "claim_text", "cite_keys", "evidence_status",
        "reviewer_notes",
    ),
    "metrics.csv": (
        "metric_id", "layer", "name", "definition", "formula_or_procedure", "units",
        "direction", "domain", "requires_dynamics", "minimum_reporting", "cite_keys",
        "limitations",
    ),
    "simulators.csv": (
        "system", "cite_key", "domain", "input_representation", "export_format",
        "load_validation", "coordinate_frame", "units", "collision_geometry",
        "spawn_reset", "rl_interface", "oss_status", "evidence_locator",
    ),
    "conflicts.csv": (
        "conflict_id", "record_type", "record_key", "field", "value_a", "value_b",
        "resolution", "resolver", "resolution_evidence",
    ),
}

DEFAULT_TAXONOMY = {
    "domain": ["ground", "aerial", "maritime", "mixed", "adjacent"],
    "course_object": [
        "closed_track", "open_corridor", "gate_chain", "waypoint_sequence",
        "road_network", "buoy_course", "world_asset", "fixed_benchmark",
    ],
    "representation_family": [
        "segment_grammar", "tile_grid", "parametric_curve", "sampled_centerline",
        "centerline_plus_width", "boundary_pair", "gate_poses", "waypoint_graph",
        "occupancy_heightfield_mesh", "simulator_native", "hybrid",
    ],
    "generator_family": [
        "constructive", "stochastic_procedural", "search_evolutionary",
        "learned_generative", "environment_design", "human_designed",
        "repair_projection", "selection_replay",
    ],
    "generation_role": [
        "geometry_synthesis", "task_selection", "mutation", "repair",
        "serialization", "benchmark_only", "boundary_case",
    ],
    "validity_strategy": [
        "by_construction", "rejection", "penalty", "repair_projection",
        "constraint_solver", "simulation_validation", "not_reported",
    ],
    "screening_status": ["candidate", "included", "excluded", "boundary"],
    "metadata_status": ["unverified", "verified", "conflict"],
    "code_status": [
        "official_open", "unofficial_open", "closed", "not_found",
        "not_applicable",
    ],
    "evidence_status": ["direct", "triangulated", "inferred", "unsupported"],
}

CONTROLLED_FIELDS = {
    "candidates.csv": {
        "screening_status": "screening_status",
        "metadata_status": "metadata_status",
    },
    "evidence.csv": {
        "domain": "domain",
        "course_object": "course_object",
        "representation_family": "representation_family",
        "generator_family": "generator_family",
        "generation_role": "generation_role",
        "validity_strategy": "validity_strategy",
        "code_status": "code_status",
    },
    "claims.csv": {"evidence_status": "evidence_status"},
}

SCALAR_CONTROLLED_FIELDS = {
    ("candidates.csv", "screening_status"),
    ("candidates.csv", "metadata_status"),
    ("evidence.csv", "code_status"),
    ("claims.csv", "evidence_status"),
}

REQUIRED_FIELDS = {
    "search_log.csv": (
        "search_id", "search_date", "stream", "agent", "query",
        "search_surface", "results_screened", "candidates_added",
    ),
    "candidates.csv": (
        "candidate_id", "title", "screening_status", "metadata_status",
    ),
    "seed_coverage.csv": (
        "source_path", "source_heading", "source_label", "coverage_status",
    ),
    "evidence.csv": (
        "cite_key", "domain", "course_object", "representation_family",
        "generator_family", "generation_role", "validity_strategy",
        "code_status", "evidence_locator",
    ),
    "claims.csv": (
        "claim_id", "section", "claim_text", "evidence_status",
    ),
    "metrics.csv": (
        "metric_id", "layer", "name", "definition", "domain",
        "requires_dynamics", "minimum_reporting",
    ),
    "simulators.csv": ("system", "domain"),
    "conflicts.csv": (
        "conflict_id", "record_type", "record_key", "field", "value_a",
        "value_b",
    ),
}

FORBIDDEN_MARKERS = ("TO" + "DO", "T" + "BD", "FIX" + "ME", "CITATION " + "NEEDED")


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value.rstrip("/")


def split_citation_keys(
    filename: str, row_number: int, value: str
) -> list[str]:
    if not value.strip():
        return []
    values = [item.strip() for item in value.split(";")]
    if any(not item for item in values):
        raise CorpusError(
            f"{filename}:{row_number}: cite_keys contains an empty list element"
        )
    return values


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise CorpusError(f"{path}: file is missing")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            actual = tuple(reader.fieldnames or ())
            if actual != required:
                raise CorpusError(f"{path}: headers {actual!r} != {required!r}")
            rows = list(reader)
        except UnicodeError as exc:
            raise CorpusError(f"{path}: invalid UTF-8: {exc}") from exc
        except csv.Error as exc:
            raise CorpusError(
                f"{path}:{reader.line_num}: CSV parse error: {exc}"
            ) from exc
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise CorpusError(f"{path}:{row_number}: malformed CSV row")
        if not any(value.strip() for value in row.values()):
            raise CorpusError(f"{path}:{row_number}: row is entirely blank")
    return rows


def _require(
    filename: str,
    row_number: int,
    row: dict[str, str],
    field: str,
) -> str:
    value = row[field].strip()
    if not value:
        raise CorpusError(f"{filename}:{row_number}: {field} is required")
    return value


def _validate_required(filename: str, rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for field in REQUIRED_FIELDS.get(filename, ()):
            _require(filename, row_number, row, field)


def _check_unique(
    filename: str,
    rows: list[dict[str, str]],
    field: str,
    label: str,
    normalizer=lambda value: value.strip(),
) -> None:
    seen: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        raw = row[field]
        if not raw.strip():
            continue
        value = normalizer(raw)
        if value in seen:
            raise CorpusError(
                f"{filename}:{row_number}: duplicate {label} {value!r}; "
                f"first seen on row {seen[value]}"
            )
        seen[value] = row_number


def _validate_controlled(
    filename: str,
    rows: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
) -> None:
    for field, vocabulary_name in CONTROLLED_FIELDS.get(filename, {}).items():
        allowed = set(taxonomy[vocabulary_name])
        scalar = (filename, field) in SCALAR_CONTROLLED_FIELDS
        for row_number, row in enumerate(rows, start=2):
            values = [item.strip() for item in row[field].split(";")]
            if any(not value for value in values):
                raise CorpusError(
                    f"{filename}:{row_number}: {field} contains an empty list element"
                )
            if "NR" in values:
                if filename != "evidence.csv":
                    raise CorpusError(
                        f"{filename}:{row_number}: {field}='NR' is outside "
                        f"{vocabulary_name}"
                    )
                if len(values) != 1:
                    raise CorpusError(
                        f"{filename}:{row_number}: {field}: NR must be used alone"
                    )
                row[field] = "NR"
                continue
            if scalar and len(values) != 1:
                raise CorpusError(
                    f"{filename}:{row_number}: {field} must contain exactly one value"
                )
            for value in values:
                if value not in allowed:
                    raise CorpusError(
                        f"{filename}:{row_number}: {field}={value!r} "
                        f"is outside {vocabulary_name}"
                    )
            row[field] = values[0] if scalar else "; ".join(values)


def _validate_markers(filename: str, rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        for field, value in row.items():
            for marker in FORBIDDEN_MARKERS:
                if marker in value.upper():
                    raise CorpusError(
                        f"{filename}:{row_number}: {field} contains {marker!r}"
                    )


def _read_taxonomy(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise CorpusError(f"{path}: file is missing")
    try:
        taxonomy = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CorpusError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(taxonomy, dict):
        raise CorpusError(f"{path}: top-level JSON value must be an object")
    for name in DEFAULT_TAXONOMY:
        if name not in taxonomy:
            raise CorpusError(f"{path}: missing vocabulary {name!r}")
        values = taxonomy[name]
        if not isinstance(values, list):
            raise CorpusError(f"{path}: {name!r} must be a list")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise CorpusError(f"{path}: {name!r} values must be nonempty strings")
        if len(values) != len(set(values)):
            raise CorpusError(f"{path}: duplicate value in {name!r}")
    return taxonomy


def validate_directory(data_dir: Path) -> None:
    taxonomy_path = data_dir / "taxonomy.json"
    taxonomy = _read_taxonomy(taxonomy_path)

    tables = {
        filename: read_csv(data_dir / filename, header)
        for filename, header in HEADERS.items()
    }
    for filename, rows in tables.items():
        _validate_markers(filename, rows)
        _validate_required(filename, rows)
        _validate_controlled(filename, rows, taxonomy)

    candidates = tables["candidates.csv"]
    for row_number, row in enumerate(candidates, start=2):
        status = row["screening_status"]
        if status in {"included", "boundary"}:
            _require("candidates.csv", row_number, row, "cite_key")
            if row["metadata_status"] != "verified":
                raise CorpusError(
                    f"candidates.csv:{row_number}: {status} source requires "
                    "metadata_status=verified"
                )
        if status == "excluded" and not row["exclusion_reason"].strip():
            raise CorpusError(
                f"candidates.csv:{row_number}: exclusion_reason is required"
            )

    _check_unique("candidates.csv", candidates, "candidate_id", "candidate_id")
    _check_unique("candidates.csv", candidates, "cite_key", "cite_key")
    _check_unique(
        "candidates.csv", candidates, "doi", "DOI", normalize_doi
    )
    _check_unique(
        "search_log.csv",
        tables["search_log.csv"],
        "search_id",
        "search_id",
    )
    _check_unique("claims.csv", tables["claims.csv"], "claim_id", "claim_id")
    _check_unique("metrics.csv", tables["metrics.csv"], "metric_id", "metric_id")
    _check_unique(
        "conflicts.csv",
        tables["conflicts.csv"],
        "conflict_id",
        "conflict_id",
    )

    by_id = {row["candidate_id"]: row for row in candidates}
    screened_keys = {
        row["cite_key"]
        for row in candidates
        if row["screening_status"] in {"included", "boundary"}
    }
    evidence_rows = tables["evidence.csv"]
    _check_unique("evidence.csv", evidence_rows, "cite_key", "cite_key")
    evidence_keys = {row["cite_key"] for row in evidence_rows}
    if evidence_keys != screened_keys:
        missing = sorted(screened_keys - evidence_keys)
        extra = sorted(evidence_keys - screened_keys)
        raise CorpusError(
            f"evidence.csv: cite_key mismatch; missing={missing}, extra={extra}"
        )

    for filename in ("claims.csv", "metrics.csv"):
        for row_number, row in enumerate(tables[filename], start=2):
            cite_keys = split_citation_keys(
                filename, row_number, row["cite_keys"]
            )
            for cite_key in cite_keys:
                if cite_key not in screened_keys:
                    raise CorpusError(
                        f"{filename}:{row_number}: unknown cite_key {cite_key!r}"
                    )
    for row_number, row in enumerate(tables["simulators.csv"], start=2):
        cite_key = row["cite_key"].strip()
        if cite_key and cite_key not in screened_keys:
            raise CorpusError(
                f"simulators.csv:{row_number}: unknown cite_key {cite_key!r}"
            )

    for row_number, row in enumerate(tables["seed_coverage.csv"], start=2):
        status = _require(
            "seed_coverage.csv", row_number, row, "coverage_status"
        )
        if status not in {"unreviewed", "linked", "excluded"}:
            raise CorpusError(
                f"seed_coverage.csv:{row_number}: invalid coverage_status "
                f"{status!r}"
            )
        candidate_id = row["candidate_id"].strip()
        if status in {"linked", "excluded"} and candidate_id not in by_id:
            raise CorpusError(
                f"seed_coverage.csv:{row_number}: unknown candidate_id "
                f"{candidate_id!r}"
            )

    conflict_targets = {
        "candidate": (
            "candidates.csv",
            {candidate_id.strip() for candidate_id in by_id},
        ),
        "evidence": (
            "evidence.csv",
            {cite_key.strip() for cite_key in evidence_keys},
        ),
    }
    for row_number, row in enumerate(tables["conflicts.csv"], start=2):
        record_type = row["record_type"].strip()
        if record_type not in conflict_targets:
            raise CorpusError(
                f"conflicts.csv:{row_number}: record_type={record_type!r} "
                "is unsupported"
            )
        target_filename, target_keys = conflict_targets[record_type]
        record_key = row["record_key"].strip()
        if record_key not in target_keys:
            raise CorpusError(
                f"conflicts.csv:{row_number}: {record_type} "
                f"record_key={record_key!r} does not resolve"
            )
        field = row["field"].strip()
        if field not in HEADERS[target_filename]:
            raise CorpusError(
                f"conflicts.csv:{row_number}: {record_type} field={field!r} "
                f"is not a column in {target_filename}"
            )
        if row["resolution"].strip():
            _require("conflicts.csv", row_number, row, "resolver")
            _require("conflicts.csv", row_number, row, "resolution_evidence")


if __name__ == "__main__":
    validate_directory(Path(__file__).resolve().parents[1] / "data")
    print("survey corpus validation passed")
