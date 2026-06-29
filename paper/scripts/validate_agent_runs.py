from __future__ import annotations

import argparse
import csv
import re
import unicodedata
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

REQUIRED_FIELDS = ("title", "metadata_evidence", "evidence_locator")
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
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RunSpec:
    id_pattern: re.Pattern[str]
    expected_ids: tuple[str, ...]
    discovery_streams: frozenset[str]
    discovery_agent: str


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
    ),
    "blind-aerial-maritime": RunSpec(
        id_pattern=re.compile(r"BAAM-[AM][0-9]{3}"),
        expected_ids=(
            _numbered_ids("BAAM-A", 17, 3)
            + _numbered_ids("BAAM-M", 15, 3)
        ),
        discovery_streams=frozenset({"aerial", "maritime"}),
        discovery_agent="blind-aerial-maritime",
    ),
    "aware-geometry-rl": RunSpec(
        id_pattern=re.compile(r"AGRL[0-9]{4}"),
        expected_ids=_numbered_ids("AGRL", 55, 4),
        discovery_streams=frozenset({"aware-geometry-rl"}),
        discovery_agent="aware-geometry-rl",
    ),
    "aware-simulation-benchmarks": RunSpec(
        id_pattern=re.compile(r"ASIM[0-9]{4}"),
        expected_ids=_numbered_ids("ASIM", 30, 4),
        discovery_streams=frozenset({"aware-simulation"}),
        discovery_agent="aware-simulation-benchmarks",
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


def _validate_required(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    for field in REQUIRED_FIELDS:
        if not row[field].strip():
            raise AgentRunError(
                f"{path}:{row_number}: {field} is required"
            )


def _validate_list_fields(
    path: Path,
    row_number: int,
    row: dict[str, str],
) -> None:
    for field in LIST_FIELDS:
        value = row[field]
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
    for field in ("code_status", "asset_status"):
        value = row[field].strip()
        if value not in AVAILABILITY_STATUSES:
            raise AgentRunError(
                f"{path}:{row_number}: {field}={value!r} must be one scalar "
                "availability status"
            )


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


def _validate_csv(path: Path, spec: RunSpec) -> None:
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
        _validate_evidence_surfaces(path, row_number, row)
    _check_duplicates(path, rows)


def _validate_report(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise AgentRunError(f"{path}: invalid UTF-8: {exc}") from exc
    headings = [
        match.group(1).strip().casefold()
        for match in HEADING_PATTERN.finditer(text)
    ]
    for category, matcher in REPORT_SECTION_MATCHERS.items():
        if not any(matcher(heading) for heading in headings):
            raise AgentRunError(
                f"{path}: missing required report section: {category}"
            )


def validate_agent_runs(data_dir: Path) -> None:
    data_dir = Path(data_dir)
    _validate_files(data_dir)
    for slug, spec in RUN_SPECS.items():
        _validate_csv(data_dir / f"{slug}.csv", spec)
        _validate_report(data_dir / f"{slug}.md")


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
