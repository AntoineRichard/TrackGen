from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit


class CorpusError(ValueError):
    pass


HEADERS = {
    "search_queries.csv": (
        "query_id", "stream", "domain", "query", "rationale",
    ),
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
        "cite_key", "survey_evidence_tier", "domain", "vehicle",
        "course_object", "representation_family", "generator_family",
        "generation_role", "validity_strategy",
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
    "bibliography.csv": (
        "candidate_id", "cite_key", "entry_type", "key_author", "authors",
        "author_kinds", "title", "year", "venue_field", "venue", "doi",
        "url",
    ),
    "citation_keys.csv": (
        "candidate_id", "cite_key",
    ),
}


SEARCH_QUERY_STREAMS = frozenset(
    {
        "blind-ground",
        "blind-aerial-maritime",
        "aware-geometry-rl",
        "aware-simulation",
        "survey-exemplars",
    }
)
CANDIDATE_ID_PATTERN = re.compile(r"C[0-9]{4,}")
ISO_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
CITE_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._/+\-]*")
YEAR_PATTERN = re.compile(r"[0-9]{4}")
DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}/[a-z0-9._;()/:+\-]+")
SUPPORTED_ENTRY_TYPES = frozenset(
    {"article", "inproceedings", "misc", "techreport", "book"}
)
VENUE_FIELD_BY_ENTRY_TYPE = {
    "article": "journal",
    "inproceedings": "booktitle",
    "misc": "howpublished",
    "techreport": "institution",
    "book": "publisher",
}
AUTHOR_KINDS = frozenset({"personal", "corporate"})
DOI_RESOLVER_HOSTS = frozenset({"doi.org", "dx.doi.org", "www.doi.org"})
PAPER_SOURCE_PATTERN = re.compile(
    r"paper|article|preprint|proceedings|journal|conference|publication|"
    r"survey|thesis|report|book"
)
NONPAPER_SOURCE_PATTERN = re.compile(
    r"software|documentation|standard|repository|benchmark|competition|"
    r"simulator|platform|release|dataset|artifact|tool|package|system"
)
INCOMPLETE_AUTHOR_PATTERN = re.compile(r"\bet[\W_]*al\b", re.IGNORECASE)

BIBLIOGRAPHY_REQUIRED_FIELDS = (
    "candidate_id",
    "cite_key",
    "entry_type",
    "key_author",
    "authors",
    "author_kinds",
    "title",
)


DEFAULT_TAXONOMY = {
    "domain": ["ground", "aerial", "maritime", "mixed", "adjacent"],
    "survey_evidence_tier": ["core", "supporting", "contextual"],
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
        "survey_evidence_tier": "survey_evidence_tier",
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
    ("evidence.csv", "survey_evidence_tier"),
    ("claims.csv", "evidence_status"),
}

REQUIRED_FIELDS = {
    "search_queries.csv": (
        "query_id", "stream", "domain", "query", "rationale",
    ),
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
        "cite_key", "survey_evidence_tier", "domain", "course_object",
        "representation_family", "generator_family", "generation_role",
        "validity_strategy",
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
    "citation_keys.csv": ("candidate_id", "cite_key"),
}

FORBIDDEN_MARKERS = ("TO" + "DO", "T" + "BD", "FIX" + "ME", "CITATION " + "NEEDED")


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value.rstrip("/")


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(';') if item.strip()]


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


def _validate_search_queries(
    rows: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
) -> None:
    _check_unique(
        "search_queries.csv",
        rows,
        "query_id",
        "query_id",
    )
    allowed_domains = set(taxonomy["domain"])
    for row_number, row in enumerate(rows, start=2):
        stream = row["stream"].strip()
        if stream not in SEARCH_QUERY_STREAMS:
            raise CorpusError(
                f"search_queries.csv:{row_number}: stream={stream!r} "
                "is outside frozen query matrix streams"
            )
        domain = row["domain"].strip()
        if domain not in allowed_domains:
            raise CorpusError(
                f"search_queries.csv:{row_number}: domain={domain!r} "
                "is outside domain"
            )


def _validate_search_log(rows: list[dict[str, str]]) -> None:
    for action_number, row in enumerate(rows, start=1):
        row_number = action_number + 1
        expected_id = f"S{action_number:04d}"
        if row["search_id"] != expected_id:
            raise CorpusError(
                f"search_log.csv:{row_number}: search_id={row['search_id']!r}; "
                f"search_id values must be exactly sequential in row order "
                f"(expected {expected_id!r})"
            )
        search_date = row["search_date"].strip()
        try:
            parsed_date = date.fromisoformat(search_date)
        except ValueError as exc:
            raise CorpusError(
                f"search_log.csv:{row_number}: search_date={search_date!r} "
                "must be an ISO date (YYYY-MM-DD)"
            ) from exc
        if (
            not ISO_DATE_PATTERN.fullmatch(search_date)
            or parsed_date.isoformat() != search_date
        ):
            raise CorpusError(
                f"search_log.csv:{row_number}: search_date={search_date!r} "
                "must be an ISO date (YYYY-MM-DD)"
            )
        for field in ("results_screened", "candidates_added"):
            value = row[field].strip()
            if value == "NR":
                if row["stream"].strip() == "bootstrap":
                    raise CorpusError(
                        f"search_log.csv:{row_number}: {field}={value!r} "
                        "must be a nonnegative integer"
                    )
                if not re.search(
                    r"\bcounts?\s+(?:was|were)\s+not captured\b",
                    row["notes"],
                    re.IGNORECASE,
                ):
                    raise CorpusError(
                        f"search_log.csv:{row_number}: {field}={value!r} requires "
                        "notes to state that the count was not captured"
                    )
                continue
            if not re.fullmatch(r"[0-9]+", value):
                raise CorpusError(
                    f"search_log.csv:{row_number}: {field}={value!r} "
                    "must be a nonnegative integer"
                )


def _validate_local_corpus_counts(
    search_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
) -> None:
    for row_number, row in enumerate(search_rows, start=2):
        if (
            row["stream"].strip() != "bootstrap"
            or row["search_surface"].strip() != "local-corpus"
        ):
            continue
        query_path = row["query"].strip()
        expected = sum(
            seed["source_path"].strip() == query_path for seed in seed_rows
        )
        actual = int(row["results_screened"])
        if actual != expected:
            raise CorpusError(
                f"search_log.csv:{row_number}: results_screened={actual} "
                f"but seed_coverage rows={expected} for {query_path!r}"
            )


def _is_nonpaper_misc(entry_type: str, source_type: str) -> bool:
    normalized_source_type = source_type.casefold()
    return (
        entry_type == "misc"
        and bool(NONPAPER_SOURCE_PATTERN.search(normalized_source_type))
        and not PAPER_SOURCE_PATTERN.search(normalized_source_type)
    )


def _split_bibliography_list(
    value: str,
    *,
    field: str,
    row_number: int,
) -> list[str]:
    values = [item.strip() for item in value.split(";")]
    if any(not item for item in values):
        raise CorpusError(
            f"bibliography.csv:{row_number}: {field} contains an empty "
            "semicolon element"
        )
    if "NR" in values:
        raise CorpusError(
            f"bibliography.csv:{row_number}: {field}: NR must be the sole list sentinel"
        )
    return values


def _validate_citation_key_rows(
    rows: list[dict[str, str]],
    candidates: list[dict[str, str]],
) -> None:
    _check_unique(
        "citation_keys.csv",
        rows,
        "candidate_id",
        "candidate_id",
    )
    _check_unique(
        "citation_keys.csv",
        rows,
        "cite_key",
        "cite_key",
        lambda value: value.strip().casefold(),
    )

    candidates_by_id = {
        row["candidate_id"]: row for row in candidates
    }
    ledger_by_id: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        for field_name, value in row.items():
            if value != value.strip():
                raise CorpusError(
                    f"citation_keys.csv:{row_number}: {field_name} "
                    "contains surrounding whitespace"
                )
        candidate_id = row["candidate_id"]
        cite_key = row["cite_key"]
        if CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
            raise CorpusError(
                f"citation_keys.csv:{row_number}: candidate_id="
                f"{candidate_id!r} must be C followed by at least four digits"
            )
        if CITE_KEY_PATTERN.fullmatch(cite_key) is None:
            raise CorpusError(
                f"citation_keys.csv:{row_number}: cite_key={cite_key!r} "
                "is not BibTeX-safe"
            )
        if candidate_id not in candidates_by_id:
            raise CorpusError(
                f"citation_keys.csv:{row_number}: candidate_id="
                f"{candidate_id!r} does not exist in candidates.csv"
            )
        ledger_by_id[candidate_id] = cite_key

    for row_number, candidate in enumerate(candidates, start=2):
        eligible = (
            candidate["metadata_status"] == "verified"
            and candidate["screening_status"] != "excluded"
        )
        if not eligible:
            continue
        candidate_id = candidate["candidate_id"]
        ledger_key = ledger_by_id.get(candidate_id)
        if ledger_key is None:
            raise CorpusError(
                f"candidates.csv:{row_number}: candidate_id={candidate_id!r} "
                "is missing from citation key ledger"
            )
        if candidate["cite_key"] != ledger_key:
            raise CorpusError(
                f"candidates.csv:{row_number}: candidate_id={candidate_id!r}: "
                f"cite_key={candidate['cite_key']!r} does not match citation "
                f"key ledger value {ledger_key!r}"
            )


def _validate_bibliography_row(
    row: dict[str, str],
    row_number: int,
) -> None:
    for field, value in row.items():
        if "\r" in value:
            raise CorpusError(
                f"bibliography.csv:{row_number}: {field} contains carriage return"
            )
        if value != value.strip():
            raise CorpusError(
                f"bibliography.csv:{row_number}: {field} contains "
                "surrounding whitespace"
            )
    for field in BIBLIOGRAPHY_REQUIRED_FIELDS:
        _require("bibliography.csv", row_number, row, field)

    cite_key = row["cite_key"]
    if CITE_KEY_PATTERN.fullmatch(cite_key) is None:
        raise CorpusError(
            f"bibliography.csv:{row_number}: cite_key={cite_key!r} "
            "is not BibTeX-safe"
        )

    entry_type = row["entry_type"]
    if entry_type not in SUPPORTED_ENTRY_TYPES:
        raise CorpusError(
            f"bibliography.csv:{row_number}: unsupported "
            f"entry_type {entry_type!r}"
        )

    for field in ("authors", "key_author"):
        value = row[field]
        if value == "NR":
            raise CorpusError(
                f"bibliography.csv:{row_number}: {field} cannot be NR"
            )
        if INCOMPLETE_AUTHOR_PATTERN.search(value):
            raise CorpusError(
                f"bibliography.csv:{row_number}: {field} contains "
                "incomplete author marker"
            )

    authors = _split_bibliography_list(
        row["authors"], field="authors", row_number=row_number
    )
    author_kinds = _split_bibliography_list(
        row["author_kinds"],
        field="author_kinds",
        row_number=row_number,
    )
    if len(authors) != len(author_kinds):
        raise CorpusError(
            f"bibliography.csv:{row_number}: author_kinds must align "
            "one-to-one with authors"
        )
    invalid_kinds = sorted(set(author_kinds) - AUTHOR_KINDS)
    if invalid_kinds:
        raise CorpusError(
            f"bibliography.csv:{row_number}: invalid author kind "
            f"{invalid_kinds[0]!r}"
        )

    venue = row["venue"]
    venue_field = row["venue_field"]
    expected_venue_field = VENUE_FIELD_BY_ENTRY_TYPE[entry_type]
    if venue and venue_field != expected_venue_field:
        raise CorpusError(
            f"bibliography.csv:{row_number}: {entry_type} requires "
            f"venue_field={expected_venue_field!r}"
        )
    if not venue and venue_field:
        raise CorpusError(
            f"bibliography.csv:{row_number}: venue_field must be empty "
            "when venue is empty"
        )

    year = row["year"]
    if year and YEAR_PATTERN.fullmatch(year) is None:
        raise CorpusError(
            f"bibliography.csv:{row_number}: year={year!r} must be "
            "four digits when present"
        )

    doi = row["doi"]
    if doi and (
        normalize_doi(doi) != doi
        or DOI_PATTERN.fullmatch(doi) is None
    ):
        raise CorpusError(
            f"bibliography.csv:{row_number}: doi={doi!r} must use "
            "canonical DOI form"
        )

    url = row["url"]
    if not url:
        return
    if any(character.isspace() for character in url):
        raise CorpusError(
            f"bibliography.csv:{row_number}: url contains whitespace"
        )
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise CorpusError(
            f"bibliography.csv:{row_number}: url={url!r} must be an "
            "absolute HTTP/HTTPS URL with a valid authority"
        ) from exc
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise CorpusError(
            f"bibliography.csv:{row_number}: url={url!r} must be an "
            "absolute HTTP/HTTPS URL with a valid authority and no credentials"
        )

    normalized_host = host.casefold().rstrip(".")
    if normalized_host in DOI_RESOLVER_HOSTS:
        resolver_doi = normalize_doi(unquote(parsed.path).lstrip("/"))
        if (
            not doi
            or resolver_doi != doi
            or DOI_PATTERN.fullmatch(resolver_doi) is None
        ):
            raise CorpusError(
                f"bibliography.csv:{row_number}: DOI resolver URL "
                f"{url!r} does not match doi={doi!r}"
            )
        raise CorpusError(
            f"bibliography.csv:{row_number}: redundant DOI resolver URL "
            f"{url!r} must be omitted"
        )


def _validate_bibliography_rows(
    rows: list[dict[str, str]],
    candidates: list[dict[str, str]],
) -> None:
    for row_number, row in enumerate(rows, start=2):
        _validate_bibliography_row(row, row_number)

    _check_unique(
        "bibliography.csv", rows, "candidate_id", "candidate_id"
    )
    _check_unique(
        "bibliography.csv",
        rows,
        "cite_key",
        "cite_key",
        lambda value: value.strip().casefold(),
    )

    eligible_by_id = {
        row["candidate_id"]: row
        for row in candidates
        if row["metadata_status"] == "verified"
        and row["screening_status"] != "excluded"
    }
    seen_ids: set[str] = set()
    candidate_fields = (
        "cite_key",
        "title",
        "authors",
        "year",
        "venue",
        "doi",
    )
    for row_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"]
        if candidate_id not in eligible_by_id:
            raise CorpusError(
                f"bibliography.csv:{row_number}: "
                f"candidate_id={candidate_id!r} is not eligible; expected "
                "metadata_status='verified' and screening_status!='excluded'"
            )
        seen_ids.add(candidate_id)
        candidate = eligible_by_id[candidate_id]
        if not _is_nonpaper_misc(
            row["entry_type"], candidate["source_type"]
        ):
            source_label = (
                "paper-like misc"
                if row["entry_type"] == "misc"
                else row["entry_type"]
            )
            for field in ("year", "venue"):
                if not row[field]:
                    raise CorpusError(
                        f"bibliography.csv:{row_number}: {field} is "
                        f"required for {source_label}"
                    )

        for field in candidate_fields:
            if row[field] != candidate[field]:
                raise CorpusError(
                    f"bibliography.csv:{row_number}: "
                    f"candidate_id={candidate_id!r}: {field}={row[field]!r} "
                    f"does not match candidates.csv value "
                    f"{candidate[field]!r}"
                )

    missing = sorted(set(eligible_by_id) - seen_ids)
    if missing:
        raise CorpusError(
            "bibliography.csv: candidate_id mismatch; "
            f"missing={missing}, extra=[]"
        )

    expected = sorted(
        rows,
        key=lambda row: (
            row["cite_key"].casefold(),
            row["cite_key"],
            row["candidate_id"],
        ),
    )
    if rows != expected:
        actual_keys = [row["cite_key"] for row in rows]
        expected_keys = [row["cite_key"] for row in expected]
        raise CorpusError(
            "bibliography.csv: rows are not in canonical cite_key order; "
            f"actual={actual_keys}, expected={expected_keys}"
        )


@dataclass(frozen=True)
class _BibtexEntry:
    entry_type: str
    cite_key: str
    fields: tuple[tuple[str, str], ...]
    line_number: int


class _BibtexParser:
    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        self.text = text
        self.index = 0

    def _line_number(self, index: int | None = None) -> int:
        position = self.index if index is None else index
        return self.text.count("\n", 0, position) + 1

    def _error(
        self,
        message: str,
        *,
        index: int | None = None,
    ) -> None:
        raise CorpusError(
            f"{self.path}:{self._line_number(index)}: {message}"
        )

    def _skip_whitespace(self) -> None:
        while (
            self.index < len(self.text)
            and self.text[self.index].isspace()
        ):
            self.index += 1

    def _expect(self, character: str, message: str) -> None:
        if (
            self.index >= len(self.text)
            or self.text[self.index] != character
        ):
            self._error(message)
        self.index += 1

    def _parse_braced_value(
        self,
        *,
        cite_key: str,
        field: str,
    ) -> str:
        self._expect(
            "{",
            f"key={cite_key!r}: field {field!r} requires a braced value",
        )
        start = self.index
        depth = 1
        while self.index < len(self.text):
            character = self.text[self.index]
            if character == "\\":
                self.index += 1
                if self.index < len(self.text):
                    self.index += 1
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    value = self.text[start:self.index]
                    self.index += 1
                    return value
            self.index += 1
        self._error(
            f"key={cite_key!r}: field {field!r} has an unclosed braced value",
            index=start - 1,
        )
        raise AssertionError("unreachable")

    def _parse_entry(self) -> _BibtexEntry:
        entry_start = self.index
        line_number = self._line_number()
        self._expect("@", "expected '@' to start a BibTeX entry")

        type_start = self.index
        while (
            self.index < len(self.text)
            and self.text[self.index].isalpha()
        ):
            self.index += 1
        if self.index == type_start:
            self._error("entry type is required", index=entry_start)
        entry_type = self.text[type_start:self.index]
        self._skip_whitespace()
        self._expect(
            "{",
            f"entry_type={entry_type!r}: expected '{{' before cite key",
        )

        key_start = self.index
        while (
            self.index < len(self.text)
            and self.text[self.index] not in ",}\r\n"
        ):
            self.index += 1
        cite_key = self.text[key_start:self.index].strip()
        if not cite_key:
            self._error("BibTeX entry cite key is required", index=key_start)
        self._expect(
            ",",
            f"key={cite_key!r}: expected ',' after cite key",
        )

        fields: list[tuple[str, str]] = []
        field_lines: dict[str, tuple[str, int]] = {}
        while True:
            self._skip_whitespace()
            if self.index >= len(self.text):
                self._error(
                    f"key={cite_key!r}: expected a closing brace for entry",
                    index=entry_start,
                )
            if self.text[self.index] == "}":
                self.index += 1
                break

            field_start = self.index
            while (
                self.index < len(self.text)
                and (
                    self.text[self.index].isalnum()
                    or self.text[self.index] == "_"
                )
            ):
                self.index += 1
            field = self.text[field_start:self.index]
            if not field:
                self._error(
                    f"key={cite_key!r}: expected a field name",
                    index=field_start,
                )
            self._skip_whitespace()
            self._expect(
                "=",
                f"key={cite_key!r}: field {field!r} requires '='",
            )
            self._skip_whitespace()
            value = self._parse_braced_value(
                cite_key=cite_key,
                field=field,
            )
            field_line = self._line_number(field_start)
            folded_field = field.casefold()
            if folded_field in field_lines:
                first_field, first_line = field_lines[folded_field]
                self._error(
                    f"key={cite_key!r}: case-insensitive duplicate field "
                    f"{field!r}; first spelling {first_field!r} on line "
                    f"{first_line}",
                    index=field_start,
                )
            field_lines[folded_field] = (field, field_line)
            fields.append((field, value))

            self._skip_whitespace()
            if self.index >= len(self.text):
                self._error(
                    f"key={cite_key!r}: expected a closing brace for entry",
                    index=entry_start,
                )
            if self.text[self.index] == ",":
                self.index += 1
            elif self.text[self.index] != "}":
                self._error(
                    f"key={cite_key!r}: expected ',' or a closing brace "
                    f"after field {field!r}"
                )

        return _BibtexEntry(
            entry_type=entry_type,
            cite_key=cite_key,
            fields=tuple(fields),
            line_number=line_number,
        )

    def parse(self) -> list[_BibtexEntry]:
        entries: list[_BibtexEntry] = []
        key_lines: dict[str, tuple[str, int]] = {}
        self._skip_whitespace()
        while self.index < len(self.text):
            entry = self._parse_entry()
            folded_key = entry.cite_key.casefold()
            if folded_key in key_lines:
                first_key, first_line = key_lines[folded_key]
                raise CorpusError(
                    f"{self.path}:{entry.line_number}: case-insensitive "
                    f"duplicate entry key {entry.cite_key!r}; first spelling "
                    f"{first_key!r} on line {first_line}"
                )
            key_lines[folded_key] = (entry.cite_key, entry.line_number)
            entries.append(entry)
            self._skip_whitespace()
        return entries


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value: str) -> str:
    return "".join(
        LATEX_ESCAPES.get(character, character)
        for character in value
    )


def _bibtex_fields(row: dict[str, str]) -> list[tuple[str, str]]:
    authors = [item.strip() for item in row["authors"].split(";")]
    author_kinds = [
        item.strip() for item in row["author_kinds"].split(";")
    ]
    rendered_authors = []
    for author, kind in zip(authors, author_kinds, strict=True):
        escaped = _latex_escape(author)
        rendered_authors.append(
            f"{{{escaped}}}" if kind == "corporate" else escaped
        )

    fields = [
        ("author", " and ".join(rendered_authors)),
        ("title", _latex_escape(row["title"])),
    ]
    if row["venue_field"] and row["venue"]:
        fields.append(
            (row["venue_field"], _latex_escape(row["venue"]))
        )
    fields.extend(
        (field, _latex_escape(row[field]))
        for field in ("year", "doi", "url")
        if row[field]
    )
    return fields


def _render_bibtex(rows: list[dict[str, str]]) -> str:
    rendered_entries = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["cite_key"].casefold(),
            item["cite_key"],
        ),
    ):
        fields = _bibtex_fields(row)
        lines = [f"@{row['entry_type']}{{{row['cite_key']},"]
        lines.extend(
            f"  {field} = {{{value}}}"
            f"{',' if index < len(fields) - 1 else ''}"
            for index, (field, value) in enumerate(fields)
        )
        lines.append("}")
        rendered_entries.append("\n".join(lines))
    return (
        "\n\n".join(rendered_entries)
        + ("\n" if rendered_entries else "")
    )


def _read_references_bib(path: Path) -> tuple[bytes, str]:
    if not path.is_file():
        raise CorpusError(f"{path}: file is missing")
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise CorpusError(f"{path}: invalid UTF-8: {exc}") from exc
    if b"\r" in raw_bytes:
        raise CorpusError(
            f"{path}: references.bib must use deterministic LF bytes"
        )
    return raw_bytes, text


def _validate_references_bib(
    path: Path,
    bibliography_rows: list[dict[str, str]],
) -> None:
    raw_bytes, text = _read_references_bib(path)
    entries = _BibtexParser(path, text).parse()
    expected_rows = sorted(
        bibliography_rows,
        key=lambda row: (
            row["cite_key"].casefold(),
            row["cite_key"],
        ),
    )
    expected_keys = [row["cite_key"] for row in expected_rows]
    actual_keys = [entry.cite_key for entry in entries]
    missing = sorted(set(expected_keys) - set(actual_keys))
    extra = sorted(set(actual_keys) - set(expected_keys))
    if missing or extra:
        raise CorpusError(
            f"{path}: entry mismatch; missing={missing}, extra={extra}"
        )
    if actual_keys != expected_keys:
        raise CorpusError(
            f"{path}: entries are not in canonical order; "
            f"actual={actual_keys}, expected={expected_keys}"
        )

    entries_by_key = {entry.cite_key: entry for entry in entries}
    for row in expected_rows:
        cite_key = row["cite_key"]
        entry = entries_by_key[cite_key]
        context = (
            f"{path}:{entry.line_number}: key={cite_key!r}"
        )
        if entry.entry_type != row["entry_type"]:
            raise CorpusError(
                f"{context}: entry_type={entry.entry_type!r} does not "
                f"match bibliography.csv value {row['entry_type']!r}"
            )

        expected_fields = _bibtex_fields(row)
        expected_names = [field for field, _ in expected_fields]
        actual_names = [field for field, _ in entry.fields]
        missing_fields = sorted(
            set(expected_names) - set(actual_names)
        )
        extra_fields = sorted(set(actual_names) - set(expected_names))
        if missing_fields or extra_fields:
            raise CorpusError(
                f"{context}: field mismatch; missing={missing_fields}, "
                f"extra={extra_fields}"
            )
        if actual_names != expected_names:
            raise CorpusError(
                f"{context}: fields are not in deterministic order; "
                f"actual={actual_names}, expected={expected_names}"
            )

        actual_fields = dict(entry.fields)
        for field, expected_value in expected_fields:
            actual_value = actual_fields[field]
            if actual_value != expected_value:
                raise CorpusError(
                    f"{context}: field {field!r}={actual_value!r} does "
                    "not match bibliography.csv rendered value "
                    f"{expected_value!r}"
                )

    expected_text = _render_bibtex(expected_rows)
    expected_bytes = expected_text.encode("utf-8")
    if raw_bytes != expected_bytes:
        actual_lines = text.splitlines(keepends=True)
        expected_lines = expected_text.splitlines(keepends=True)
        differing_line = 1
        for index in range(max(len(actual_lines), len(expected_lines))):
            actual_line = (
                actual_lines[index] if index < len(actual_lines) else None
            )
            expected_line = (
                expected_lines[index]
                if index < len(expected_lines)
                else None
            )
            if actual_line != expected_line:
                differing_line = index + 1
                break
        raise CorpusError(
            f"{path}:{differing_line}: does not match deterministic "
            "rendering"
        )


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
    _validate_search_queries(tables["search_queries.csv"], taxonomy)
    _validate_search_log(tables["search_log.csv"])
    _validate_local_corpus_counts(
        tables["search_log.csv"], tables["seed_coverage.csv"]
    )

    candidates = tables["candidates.csv"]
    for row_number, row in enumerate(candidates, start=2):
        candidate_id = row["candidate_id"].strip()
        if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
            raise CorpusError(
                f"candidates.csv:{row_number}: candidate_id={candidate_id!r} "
                "must be C followed by at least four digits"
            )
        candidate_url = row["url"]
        if candidate_url and any(
            character.isspace() for character in candidate_url
        ):
            raise CorpusError(
                f"candidates.csv:{row_number}: url contains whitespace"
            )
        status = row["screening_status"]
        metadata_status = row["metadata_status"]
        if (
            status in {"included", "boundary"}
            and metadata_status != "verified"
        ):
            raise CorpusError(
                f"candidates.csv:{row_number}: {status} source requires "
                "metadata_status=verified"
            )
        if status == "excluded" and not row["exclusion_reason"].strip():
            raise CorpusError(
                f"candidates.csv:{row_number}: exclusion_reason is required"
            )

        cite_key = row["cite_key"].strip()
        eligible = (
            metadata_status == "verified" and status != "excluded"
        )
        if eligible and not cite_key:
            raise CorpusError(
                f"candidates.csv:{row_number}: cite_key is required for "
                "verified, non-excluded candidates"
            )
        if not eligible and cite_key:
            raise CorpusError(
                f"candidates.csv:{row_number}: cite_key must be blank for "
                "ineligible candidates"
            )

    _check_unique("candidates.csv", candidates, "candidate_id", "candidate_id")
    _check_unique(
        "candidates.csv",
        candidates,
        "cite_key",
        "cite_key",
        lambda value: value.strip().casefold(),
    )
    _check_unique(
        "candidates.csv", candidates, "doi", "DOI", normalize_doi
    )
    _validate_citation_key_rows(
        tables["citation_keys.csv"], candidates
    )
    _validate_bibliography_rows(tables["bibliography.csv"], candidates)
    _validate_references_bib(
        data_dir.parent / "references.bib",
        tables["bibliography.csv"],
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
