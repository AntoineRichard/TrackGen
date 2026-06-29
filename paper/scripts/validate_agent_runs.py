from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class AgentRunError(ValueError):
    pass


COMMON_HEADER = (
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
    "domain",
    "vehicle",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "geometry_metrics",
    "difficulty_metrics",
    "diversity_metrics",
    "training_distribution",
    "evaluation_suite",
    "simulator",
    "export_format",
    "code_status",
    "asset_status",
    "reproducibility_fields",
    "evidence_locator",
    "coding_notes",
)

LIST_FIELDS = (
    "authors",
    "source_type",
    "metadata_evidence",
    "domain",
    "vehicle",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "geometry_metrics",
    "difficulty_metrics",
    "diversity_metrics",
    "training_distribution",
    "evaluation_suite",
    "simulator",
    "export_format",
    "reproducibility_fields",
    "evidence_locator",
)
SCREENING_STATUSES = frozenset({"candidate", "included", "excluded", "boundary"})
METADATA_STATUSES = frozenset({"unverified", "verified", "conflict"})
AVAILABILITY_STATUSES = frozenset(
    {
        "official_open",
        "unofficial_open",
        "closed",
        "not_found",
        "not_applicable",
        "NR",
    }
)
URL_PATTERN = re.compile(r"https?://[^\s;,)\]\"']+")
INLINE_QUERY_PATTERN = re.compile(
    r"^-\s+" + chr(96) + r"([^\x60]+)" + chr(96) + r"\s*$"
)
SEARCH_LOG_HEADER = (
    "search_id",
    "search_date",
    "stream",
    "agent",
    "query",
    "search_surface",
    "results_screened",
    "candidates_added",
    "notes",
)
AWARE_CONTROLLED_FIELDS = (
    "domain",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
)
AWARE_RUNS = frozenset(
    {"aware-geometry-rl", "aware-simulation-benchmarks"}
)
PROVENANCE_PATTERN = re.compile(
    r"\b(?:bootstrap|seed|newly discovered)\b",
    re.IGNORECASE,
)
PROVENANCE_LEDGER_PATTERN = re.compile(
    r"^\|\s*(?P<candidate_id>(?:AGRL|ASIM)[0-9]{4})\s*\|\s*"
    + chr(96)
    + r"(?P<provenance>(?:seed|citation)::[^\x60]+)"
    + chr(96)
    + r"\s*\|"
)
STABLE_SOURCE_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/-]*"
)
SATURATION_STATEMENT_PATTERN = (
    r"[0-9]+/[0-9]+ = [0-9]+(?:\.[0-9]+)?%"
)
SUMMARY_NOTES_PATTERN = re.compile(
    r"^Source: (?P<source>[^;]+); "
    r"(?P<retained>[0-9]+) retained, (?P<excluded>[0-9]+) excluded; "
    r"final saturation arithmetic: (?P<first>"
    + SATURATION_STATEMENT_PATTERN
    + r") and (?P<second>"
    + SATURATION_STATEMENT_PATTERN
    + r")\. Total screened-hit count was not captured\.$"
)
LOCATOR_MARKER_PATTERN = re.compile(
    r"(?i)(?:\bp{1,2}\.?\s+\d+|\bpages?\s+\d+"
    r"|\b(?:sections?|secs?\.?|tables?|figures?|appendi(?:x|ces)|"
    r"chapters?|algorithms?|lines?)\s+[A-Z0-9]"
    r"|\bsource[- ]path\s*:)",
)


@dataclass(frozen=True)
class RunSpec:
    id_pattern: re.Pattern[str]
    expected_ids: tuple[str, ...]
    discovery_streams: frozenset[str]
    discovery_agent: str
    search_stream: str


@dataclass(frozen=True)
class ReportQuery:
    section: str
    query: str


@dataclass(frozen=True)
class ReportData:
    text: str
    queries: tuple[ReportQuery, ...]
    provenance: dict[str, str]


def _numbered_ids(prefix: str, count: int, width: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{number:0{width}d}" for number in range(1, count + 1))


BLIND_GROUND_STREAMS = frozenset(
    {
        "Formula Student Driverless",
        "autonomous racing",
        "autonomous-vehicle testing",
        "learned map generation",
        "legged-robot obstacle courses",
        "legged-robot terrain curricula",
        "open-ended terrain curricula",
        "racing-game PCG transfer",
        "robot-learning benchmark",
        "targeted refinement round 1",
        "targeted refinement round 2",
    }
)

RUN_SPECS = {
    "blind-ground": RunSpec(
        id_pattern=re.compile(r"BG-[0-9]{3}"),
        expected_ids=_numbered_ids("BG-", 45, 3),
        discovery_streams=BLIND_GROUND_STREAMS,
        discovery_agent="blind-ground",
        search_stream="blind-ground",
    ),
    "blind-aerial-maritime": RunSpec(
        id_pattern=re.compile(r"BAAM-[AM][0-9]{3}"),
        expected_ids=(
            _numbered_ids("BAAM-A", 17, 3)
            + _numbered_ids("BAAM-M", 15, 3)
        ),
        discovery_streams=frozenset({"aerial", "maritime"}),
        discovery_agent="blind-aerial-maritime",
        search_stream="blind-aerial-maritime",
    ),
    "aware-geometry-rl": RunSpec(
        id_pattern=re.compile(r"AGRL[0-9]{4}"),
        expected_ids=_numbered_ids("AGRL", 55, 4),
        discovery_streams=frozenset({"aware-geometry-rl"}),
        discovery_agent="aware-geometry-rl",
        search_stream="aware-geometry-rl",
    ),
    "aware-simulation-benchmarks": RunSpec(
        id_pattern=re.compile(r"ASIM[0-9]{4}"),
        expected_ids=_numbered_ids("ASIM", 30, 4),
        discovery_streams=frozenset({"aware-simulation"}),
        discovery_agent="aware-simulation-benchmarks",
        search_stream="aware-simulation",
    ),
}

REPORT_SECTION_MATCHERS = {
    "search surfaces": lambda heading: (
        "search" in heading
        and any(word in heading for word in ("surface", "channel", "source"))
    ),
    "queries": lambda heading: "quer" in heading,
    "boundary": lambda heading: any(
        word in heading
        for word in ("boundary", "inclusion", "screening decision", "scope and screening")
    ),
    "terminology": lambda heading: any(
        word in heading for word in ("terminolog", "vocabular")
    ),
    "sparse": lambda heading: (
        "sparse" in heading
        or "contradict" in heading
        or ("gap" in heading and "conflict" in heading)
    ),
    "saturation": lambda heading: (
        "stopping rule" in heading
        or (
            "saturat" in heading
            and any(
                word in heading
                for word in ("arithmetic", "accounting", "refinement", "yield")
            )
        )
    ),
    "retrieval or limitations": lambda heading: any(
        word in heading for word in ("retriev", "limitation")
    ),
    "validation": lambda heading: (
        "validat" in heading
        or (
            "verification" in heading
            and any(word in heading for word in ("status", "limitation", "record"))
        )
    ),
}


def normalize_doi(value: str) -> str:
    normalized = value.strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized.rstrip("/")


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _expected_filenames() -> set[str]:
    return {
        f"{slug}{suffix}"
        for slug in RUN_SPECS
        for suffix in (".csv", ".md")
    }


def _validate_files(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise AgentRunError(f"{data_dir}: agent-run directory is missing")
    actual = {
        path.name
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix in {".csv", ".md"}
    }
    expected = _expected_filenames()
    unexpected = sorted(actual - expected)
    if unexpected:
        raise AgentRunError(
            f"{data_dir}: unexpected agent-run files: {unexpected}"
        )
    missing = sorted(expected - actual)
    if missing:
        raise AgentRunError(
            f"{data_dir}: expected exactly the four CSV/report pairs; "
            f"missing={missing}"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual_header = tuple(reader.fieldnames or ())
            if actual_header != COMMON_HEADER:
                raise AgentRunError(
                    f"{path}: header {actual_header!r} != {COMMON_HEADER!r}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise AgentRunError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise AgentRunError(f"{path}: CSV parse error: {exc}") from exc

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise AgentRunError(
                f"{path}:{row_number}: malformed row; expected 35 columns"
            )
    return rows


def _read_search_log(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AgentRunError(f"{path}: search log is missing")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual_header = tuple(reader.fieldnames or ())
            if actual_header != SEARCH_LOG_HEADER:
                raise AgentRunError(
                    f"{path}: header {actual_header!r} != {SEARCH_LOG_HEADER!r}"
                )
            rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise AgentRunError(f"{path}: invalid search log: {exc}") from exc
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise AgentRunError(f"{path}:{row_number}: malformed CSV row")
    return rows


def _read_taxonomy(path: Path) -> dict[str, frozenset[str]]:
    if not path.is_file():
        raise AgentRunError(f"{path}: taxonomy is missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AgentRunError(f"{path}: invalid taxonomy: {exc}") from exc
    if not isinstance(raw, dict):
        raise AgentRunError(
            f"{path}: taxonomy top-level value must be a mapping"
        )
    taxonomy: dict[str, frozenset[str]] = {}
    for field in AWARE_CONTROLLED_FIELDS:
        values = raw.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise AgentRunError(f"{path}: invalid taxonomy field {field!r}")
        taxonomy[field] = frozenset(values)
    return taxonomy


def _validate_required(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    for field in COMMON_HEADER:
        if field == "exclusion_reason":
            continue
        if not row[field].strip():
            raise AgentRunError(
                f"{path}:{row_number}: {field} must use NR rather than blank"
            )


def _validate_list_fields(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    for field in LIST_FIELDS:
        value = row[field]
        if value != value.strip():
            raise AgentRunError(
                f"{path}:{row_number}: {field} must use canonical whitespace"
            )
        if ";" not in value:
            continue
        if re.search(r";(?! )|; {2,}", value):
            raise AgentRunError(
                f"{path}:{row_number}: {field} has malformed semicolon spacing"
            )
        values = value.split("; ")
        if any(not item or item != item.strip() for item in values):
            raise AgentRunError(
                f"{path}:{row_number}: {field} has a malformed list"
            )
        if "NR" in values and len(values) != 1:
            raise AgentRunError(
                f"{path}:{row_number}: {field}: NR must be used alone"
            )


def _validate_statuses(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    screening = row["screening_status"].strip()
    if screening not in SCREENING_STATUSES:
        raise AgentRunError(
            f"{path}:{row_number}: screening_status={screening!r} is invalid"
        )
    metadata = row["metadata_status"].strip()
    if metadata not in METADATA_STATUSES:
        raise AgentRunError(
            f"{path}:{row_number}: metadata_status={metadata!r} is invalid"
        )
    exclusion_reason = row["exclusion_reason"].strip()
    if screening == "excluded" and exclusion_reason in {"", "NR"}:
        raise AgentRunError(
            f"{path}:{row_number}: excluded row requires exclusion_reason"
        )
    for field in ("code_status", "asset_status"):
        value = row[field].strip()
        if value not in AVAILABILITY_STATUSES:
            raise AgentRunError(
                f"{path}:{row_number}: {field}={value!r} must be one scalar "
                "availability status"
            )


def _validate_aware_fields(
    path: Path,
    row_number: int,
    row: dict[str, str],
    taxonomy: dict[str, frozenset[str]],
) -> None:
    for field in AWARE_CONTROLLED_FIELDS:
        values = row[field].split("; ")
        if values == ["NR"]:
            continue
        invalid = [value for value in values if value not in taxonomy[field]]
        if invalid:
            raise AgentRunError(
                f"{path}:{row_number}: {field} values {invalid!r} are outside taxonomy"
            )


def _validate_aware_provenance(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    if not PROVENANCE_PATTERN.search(row["coding_notes"]):
        raise AgentRunError(
            f"{path}:{row_number}: coding_notes must state aware-run "
            "bootstrap/seed lineage or newly discovered provenance"
        )


def _has_precise_locator(value: str) -> bool:
    for url in URL_PATTERN.findall(value):
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return True
    return bool(LOCATOR_MARKER_PATTERN.search(value))


def _is_forbidden_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if any(
        host == domain or host.endswith(f".{domain}")
        for domain in ("wikipedia.org", "medium.com", "semanticscholar.org")
    ):
        return True
    if (
        host.startswith("search.yahoo.")
        or host == "bing.com"
        or host.endswith(".bing.com")
        or host == "duckduckgo.com"
        or host.endswith(".duckduckgo.com")
    ):
        return True
    if "google." in host and parsed.path in {"/search", "/url"}:
        return True
    return False


def _validate_evidence_surfaces(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    for field in ("metadata_evidence", "evidence_locator"):
        value = row[field]
        if "semantic scholar" in value.casefold():
            raise AgentRunError(
                f"{path}:{row_number}: {field} uses a forbidden secondary evidence surface"
            )
        for url in URL_PATTERN.findall(value):
            if _is_forbidden_url(url):
                raise AgentRunError(
                    f"{path}:{row_number}: {field} uses a forbidden secondary "
                    f"evidence surface: {url}"
                )
    if not _has_precise_locator(row["evidence_locator"]):
        raise AgentRunError(
            f"{path}:{row_number}: evidence_locator must contain a complete "
            "http(s) URL or precise locator marker"
        )


def _check_duplicates(path: Path, rows: list[dict[str, str]]) -> None:
    seen_dois: dict[str, int] = {}
    seen_titles: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        doi = normalize_doi(row["doi"])
        if doi and doi != "nr":
            if doi in seen_dois:
                raise AgentRunError(
                    f"{path}:{row_number}: duplicate DOI {doi!r}; "
                    f"first seen on row {seen_dois[doi]}"
                )
            seen_dois[doi] = row_number

        title = normalize_title(row["title"])
        if title in seen_titles:
            raise AgentRunError(
                f"{path}:{row_number}: duplicate normalized title {title!r}; "
                f"first seen on row {seen_titles[title]}"
            )
        seen_titles[title] = row_number


def _validate_ids(
    path: Path,
    rows: list[dict[str, str]],
    spec: RunSpec,
) -> None:
    candidate_ids = [row["candidate_id"].strip() for row in rows]
    for row_number, candidate_id in enumerate(candidate_ids, start=2):
        if not spec.id_pattern.fullmatch(candidate_id):
            raise AgentRunError(
                f"{path}:{row_number}: candidate_id={candidate_id!r} "
                "does not match the run prefix and format"
            )
    seen: set[str] = set()
    for row_number, candidate_id in enumerate(candidate_ids, start=2):
        if candidate_id in seen:
            raise AgentRunError(
                f"{path}:{row_number}: duplicate candidate_id {candidate_id!r}"
            )
        seen.add(candidate_id)
    if tuple(candidate_ids) != spec.expected_ids:
        raise AgentRunError(
            f"{path}: candidate_id values must be sequential; "
            f"expected {spec.expected_ids[0]} through {spec.expected_ids[-1]}"
        )


def _validate_csv(
    path: Path,
    spec: RunSpec,
    slug: str,
    taxonomy: dict[str, frozenset[str]],
) -> list[dict[str, str]]:
    rows = _read_csv(path)
    _validate_ids(path, rows, spec)
    for row_number, row in enumerate(rows, start=2):
        _validate_required(path, row_number, row)
        stream = row["discovery_stream"].strip()
        if stream not in spec.discovery_streams:
            raise AgentRunError(
                f"{path}:{row_number}: discovery_stream={stream!r} is invalid "
                "for this run"
            )
        agent = row["discovery_agent"].strip()
        if agent != spec.discovery_agent:
            raise AgentRunError(
                f"{path}:{row_number}: discovery_agent={agent!r}; "
                f"expected {spec.discovery_agent!r}"
            )
        _validate_statuses(path, row_number, row)
        _validate_list_fields(path, row_number, row)
        if slug in AWARE_RUNS:
            _validate_aware_fields(path, row_number, row, taxonomy)
            _validate_aware_provenance(path, row_number, row)
        _validate_evidence_surfaces(path, row_number, row)
    _check_duplicates(path, rows)
    return rows


def _validate_report(path: Path) -> ReportData:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise AgentRunError(f"{path}: invalid UTF-8: {exc}") from exc

    headings: list[tuple[int, str]] = []
    report_headings: list[str] = []
    queries: list[ReportQuery] = []
    provenance: dict[str, str] = {}
    fenced_lines: list[str] = []
    fenced_section = ""
    capture_fence = False
    in_fence = False

    def query_context() -> bool:
        context = " ".join(value.casefold() for _, value in headings)
        return any(
            word in context for word in ("quer", "refinement", "saturation")
        )

    def section_name() -> str:
        return " / ".join(
            value for level, value in headings if level >= 2
        )

    for line_number, line in enumerate(text.splitlines(), start=1):
        heading = re.fullmatch(r"(#{1,6})\s+(.+?)\s*", line)
        if heading and not in_fence:
            level = len(heading.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            value = heading.group(2).strip()
            headings.append((level, value))
            report_headings.append(value.casefold())
            continue

        if line.lstrip().startswith(chr(96) * 3):
            if not in_fence:
                in_fence = True
                capture_fence = query_context()
                fenced_section = section_name()
                fenced_lines = []
            else:
                if capture_fence:
                    queries.extend(
                        ReportQuery(fenced_section, query)
                        for query in fenced_lines
                        if query.strip()
                    )
                in_fence = False
                capture_fence = False
                fenced_lines = []
            continue

        if in_fence:
            if capture_fence:
                fenced_lines.append(line)
            continue

        ledger_match = PROVENANCE_LEDGER_PATTERN.match(line)
        if ledger_match:
            candidate_id = ledger_match.group("candidate_id")
            if candidate_id in provenance:
                raise AgentRunError(
                    f"{path}:{line_number}: duplicate provenance ledger row "
                    f"for {candidate_id}"
                )
            provenance[candidate_id] = ledger_match.group("provenance")

        inline = INLINE_QUERY_PATTERN.fullmatch(line)
        if inline and query_context():
            queries.append(ReportQuery(section_name(), inline.group(1)))

    for category, matcher in REPORT_SECTION_MATCHERS.items():
        if not any(matcher(heading) for heading in report_headings):
            raise AgentRunError(
                f"{path}: missing required report section: {category}"
            )
    if not queries:
        raise AgentRunError(f"{path}: exact query ledger is empty")
    return ReportData(text, tuple(queries), provenance)


def _query_batches_follow_report_order(
    log_rows: list[dict[str, str]],
    logged_actions: list[ReportQuery],
    report_actions: tuple[ReportQuery, ...],
) -> bool:
    batches: list[list[ReportQuery]] = []
    previous_id: int | None = None
    for row, action in zip(log_rows, logged_actions):
        match = re.fullmatch(r"S([0-9]+)", row["search_id"])
        if not match:
            raise AgentRunError(
                f"search_log.csv: invalid search_id {row['search_id']!r}"
            )
        search_id = int(match.group(1))
        if previous_id is None or search_id != previous_id + 1:
            batches.append([])
        batches[-1].append(action)
        previous_id = search_id

    for batch in batches:
        position = 0
        for action in batch:
            while (
                position < len(report_actions)
                and report_actions[position] != action
            ):
                position += 1
            if position == len(report_actions):
                return False
            position += 1
    return True


def _read_bootstrap_candidate_ids(path: Path) -> frozenset[str]:
    if not path.is_file():
        raise AgentRunError(f"{path}: bootstrap candidate table is missing")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = set(reader.fieldnames or ())
            required = {"candidate_id", "discovery_stream"}
            if not required.issubset(fields):
                raise AgentRunError(
                    f"{path}: candidate table lacks {sorted(required)!r}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise AgentRunError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise AgentRunError(f"{path}: CSV parse error: {exc}") from exc

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise AgentRunError(f"{path}:{row_number}: malformed CSV row")
    return frozenset(
        row["candidate_id"].strip()
        for row in rows
        if row["discovery_stream"].strip() == "bootstrap"
    )


def _exact_query_action(
    slug: str,
    report_path: str,
    report: ReportData,
    row: dict[str, str],
) -> ReportQuery:
    prefix = f"Source: {report_path}; section: "
    suffix = (
        ". Per-query screened-hit and candidate-add counts were not captured."
    )
    notes = row["notes"]
    if (
        not notes.startswith(prefix)
        or not notes.endswith(suffix)
        or len(notes) <= len(prefix) + len(suffix)
    ):
        raise AgentRunError(
            f"{slug}: exact-query section/source notes do not match the "
            "paired report"
        )
    action = ReportQuery(
        notes[len(prefix):-len(suffix)],
        row["query"],
    )
    report_queries = {item.query for item in report.queries}
    if action.query in report_queries and action not in set(report.queries):
        raise AgentRunError(
            f"{slug}: exact-query section does not match the paired report"
        )
    return action


def _validate_candidate_queries(
    slug: str,
    csv_path: Path,
    candidate_rows: list[dict[str, str]],
    report: ReportData,
    logged_actions: list[ReportQuery],
    bootstrap_candidate_ids: frozenset[str],
) -> None:
    report_queries = {item.query for item in report.queries}
    logged_queries = {item.query for item in logged_actions}

    for row_number, row in enumerate(candidate_rows, start=2):
        value = row["discovery_query"]
        if slug not in AWARE_RUNS:
            if value == "NR":
                continue
            if value in report_queries and value in logged_queries:
                continue
            raise AgentRunError(
                f"{csv_path}:{row_number}: discovery_query={value!r} "
                "is absent from the report and exact-query log query ledgers"
            )

        if value.startswith("query::"):
            literal = value.removeprefix("query::")
            if (
                literal
                and literal in report_queries
                and literal in logged_queries
            ):
                continue
            raise AgentRunError(
                f"{csv_path}:{row_number}: discovery_query {value!r}; "
                "query:: literal must match an exact-query report and log action"
            )

        if value.startswith("seed::"):
            candidate_id = value.removeprefix("seed::")
            if not re.fullmatch(r"C[0-9]{4,}", candidate_id):
                raise AgentRunError(
                    f"{csv_path}:{row_number}: discovery_query={value!r} "
                    "does not follow the aware provenance grammar"
                )
            if candidate_id not in bootstrap_candidate_ids:
                raise AgentRunError(
                    f"{csv_path}:{row_number}: {value} does not reference an "
                    "existing bootstrap candidate"
                )
            if report.provenance.get(row["candidate_id"]) != value:
                raise AgentRunError(
                    f"{csv_path}:{row_number}: {value} has no matching "
                    "provenance ledger relationship in the paired report"
                )
            continue

        if value.startswith("citation::"):
            source_identifier = value.removeprefix("citation::")
            if not STABLE_SOURCE_IDENTIFIER_PATTERN.fullmatch(
                source_identifier
            ):
                raise AgentRunError(
                    f"{csv_path}:{row_number}: citation:: provenance requires "
                    "a nonempty stable source identifier"
                )
            if report.provenance.get(row["candidate_id"]) != value:
                raise AgentRunError(
                    f"{csv_path}:{row_number}: {value} has no matching "
                    "provenance ledger relationship in the paired report"
                )
            continue

        raise AgentRunError(
            f"{csv_path}:{row_number}: discovery_query={value!r} does not "
            "follow the aware provenance grammar "
            "(query::<literal>, seed::<C####>, or citation::<source identifier>)"
        )


def _normalize_ratio_statement(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _validate_summary(
    slug: str,
    spec: RunSpec,
    candidate_rows: list[dict[str, str]],
    report_path: str,
    report: ReportData,
    summary: dict[str, str],
) -> None:
    if (
        summary["stream"].strip() != spec.search_stream
        or summary["agent"].strip() != spec.discovery_agent
        or summary["search_surface"].strip() != "documented-agent-run"
    ):
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY has incorrect stream, agent, or surface"
        )
    if summary["results_screened"] != "NR":
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY results_screened must be NR"
        )

    try:
        candidates_added = int(summary["candidates_added"])
    except ValueError as exc:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY candidates_added must be the CSV row count"
        ) from exc
    if candidates_added != len(candidate_rows):
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY candidates_added={candidates_added} "
            f"does not match CSV row count {len(candidate_rows)}"
        )

    notes = summary["notes"]
    required_missing_text = "Total screened-hit count was not captured."
    if required_missing_text not in notes:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY must explicitly state that the total "
            "screened-hit count was not captured"
        )
    match = SUMMARY_NOTES_PATTERN.fullmatch(notes)
    if not match:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY notes do not follow the structured "
            "source/count/saturation contract"
        )
    if match.group("source") != report_path:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY source path does not match the paired report"
        )

    excluded = sum(
        row["screening_status"].strip() == "excluded"
        for row in candidate_rows
    )
    retained = len(candidate_rows) - excluded
    if int(match.group("retained")) != retained:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY retained count does not match CSV statuses"
        )
    if int(match.group("excluded")) != excluded:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY excluded count does not match CSV statuses"
        )

    statements = (match.group("first"), match.group("second"))
    if statements[0] == statements[1]:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY saturation arithmetic must identify two "
            "distinct final statements"
        )
    compact_report = _normalize_ratio_statement(report.text)
    for statement in statements:
        if _normalize_ratio_statement(statement) not in compact_report:
            raise AgentRunError(
                f"{slug}: RUN-SUMMARY saturation statement {statement!r} "
                "is absent from the paired report"
            )


def _validate_search_integration(
    slug: str,
    spec: RunSpec,
    csv_path: Path,
    candidate_rows: list[dict[str, str]],
    report: ReportData,
    search_rows: list[dict[str, str]],
    bootstrap_candidate_ids: frozenset[str],
) -> None:
    exact_rows = [
        row
        for row in search_rows
        if row["stream"].strip() == spec.search_stream
        and row["agent"].strip() == spec.discovery_agent
        and row["search_surface"].strip() == "mixed-primary-web"
    ]
    report_path = f"paper/data/agent_runs/{slug}.md"
    logged_actions = [
        _exact_query_action(slug, report_path, report, row)
        for row in exact_rows
    ]
    expected = Counter(report.queries)
    actual = Counter(logged_actions)
    missing = list((expected - actual).elements())
    if missing:
        raise AgentRunError(
            f"{slug}: exact-query ledger is missing "
            f"{[item.query for item in missing]!r}"
        )
    extra = list((actual - expected).elements())
    if extra:
        raise AgentRunError(
            f"{slug}: exact-query ledger has extra queries "
            f"{[item.query for item in extra]!r}"
        )
    if not _query_batches_follow_report_order(
        exact_rows,
        logged_actions,
        report.queries,
    ):
        raise AgentRunError(
            f"{slug}: exact-query rows do not preserve report order "
            "within append batches"
        )

    summary_query = f"RUN-SUMMARY:{report_path}"
    summaries = [
        row for row in search_rows if row["query"] == summary_query
    ]
    if len(summaries) != 1:
        raise AgentRunError(
            f"{slug}: RUN-SUMMARY must appear exactly one time"
        )
    _validate_summary(
        slug,
        spec,
        candidate_rows,
        report_path,
        report,
        summaries[0],
    )
    _validate_candidate_queries(
        slug,
        csv_path,
        candidate_rows,
        report,
        logged_actions,
        bootstrap_candidate_ids,
    )


def validate_agent_runs(data_dir: Path) -> None:
    data_dir = Path(data_dir)
    _validate_files(data_dir)
    taxonomy = _read_taxonomy(data_dir.parent / "taxonomy.json")
    search_rows = _read_search_log(data_dir.parent / "search_log.csv")
    bootstrap_candidate_ids = _read_bootstrap_candidate_ids(
        data_dir.parent / "candidates.csv"
    )
    for slug, spec in RUN_SPECS.items():
        csv_path = data_dir / f"{slug}.csv"
        candidate_rows = _validate_csv(
            csv_path,
            spec,
            slug,
            taxonomy,
        )
        report = _validate_report(data_dir / f"{slug}.md")
        _validate_search_integration(
            slug,
            spec,
            csv_path,
            candidate_rows,
            report,
            search_rows,
            bootstrap_candidate_ids,
        )


def main(argv: list[str] | None = None) -> None:
    default_dir = Path(__file__).resolve().parents[1] / "data" / "agent_runs"
    parser = argparse.ArgumentParser(description="Validate survey agent-run outputs.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=default_dir,
        help="directory containing the four agent-run CSV/report pairs",
    )
    args = parser.parse_args(argv)
    validate_agent_runs(args.data_dir)
    print("agent discovery validation passed")


if __name__ == "__main__":
    main()
