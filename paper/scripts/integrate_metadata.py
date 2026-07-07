from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import SplitResult, unquote, urlsplit

if __package__:
    from .prepare_metadata_batches import (
        MANIFEST_HEADER,
        SUPPORTED_BATCH_COUNT,
        validate_manifest_inputs,
    )
    from .validate_corpus import HEADERS, normalize_doi
else:
    from prepare_metadata_batches import (
        MANIFEST_HEADER,
        SUPPORTED_BATCH_COUNT,
        validate_manifest_inputs,
    )
    from validate_corpus import HEADERS, normalize_doi


class MetadataIntegrationError(ValueError):
    pass


METADATA_RESULT_HEADER = (
    "candidate_id",
    "input_sha256",
    "agent_id",
    "verified_on",
    "metadata_status",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
    "title_evidence",
    "authors_evidence",
    "year_evidence",
    "venue_evidence",
    "doi_evidence",
    "url_evidence",
    "source_type_evidence",
    "bib_entry_type",
    "bib_venue_field",
    "bib_url",
    "key_author",
    "author_kinds",
    "notes",
)
CONFLICT_RESULT_HEADER = (
    "candidate_id",
    "input_sha256",
    "agent_id",
    "input_conflict_id",
    "field",
    "value_a",
    "value_b",
    "resolution",
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

CITATION_KEYS_HEADER = ("candidate_id", "cite_key")

CANDIDATE_HEADER = HEADERS["candidates.csv"]
CONFLICT_HEADER = HEADERS["conflicts.csv"]
BIBLIOGRAPHIC_FIELDS = (
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
IMMUTABLE_CANDIDATE_FIELDS = (
    "candidate_id",
    "discovery_stream",
    "discovery_query",
    "discovery_agent",
    "screening_status",
    "exclusion_reason",
)
METADATA_STATUS_VALUES = frozenset({"unverified", "verified", "conflict"})
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
AUTHORITATIVE_SOURCE_KINDS = frozenset(
    {
        "publisher",
        "proceedings",
        "doi",
        "preprint_repository",
        "official_repository",
        "official_documentation",
        "official_standard",
        "official_organizer",
        "official_release",
        "citation_cff",
        "issuing_body",
    }
)
OFFICIAL_ARTIFACT_SOURCE_KINDS = frozenset(
    {
        "official_repository",
        "official_documentation",
        "official_standard",
        "official_organizer",
        "official_release",
        "citation_cff",
        "issuing_body",
    }
)
CORROBORATING_SOURCE_KINDS = frozenset({"crossref"})
PUBLISHED_CORE_SOURCE_KINDS = frozenset({"publisher", "proceedings"})
PUBLISHED_DOI_SOURCE_KINDS = frozenset({"publisher", "proceedings", "doi"})
PUBLISHED_URL_SOURCE_KINDS = frozenset(
    {"publisher", "proceedings", "doi", "preprint_repository"}
)
PAPER_CORE_EVIDENCE_FIELDS = frozenset(
    {"title", "authors", "year", "venue", "source_type"}
)
PREPRINT_CORE_SOURCE_KINDS = frozenset({"preprint_repository"})
PREPRINT_DOI_URL_SOURCE_KINDS = frozenset(
    {"preprint_repository", "doi"}
)
DOI_RESOLVER_HOSTS = frozenset({"doi.org", "dx.doi.org", "www.doi.org"})
DISCOVERY_SOURCE_KINDS = frozenset(
    {
        "aggregator",
        "dblp",
        "discovery",
        "generic_search",
        "google",
        "google_scholar",
        "openalex",
        "search",
        "search_engine",
        "search_snippet",
        "semantic_scholar",
        "wikipedia",
    }
)
EVIDENCE_FIELDS = tuple(
    (field_name, f"{field_name}_evidence")
    for field_name in BIBLIOGRAPHIC_FIELDS
)
EVIDENCE_TOKEN_PATTERN = re.compile(
    r"(?P<kind>[a-z][a-z0-9_]*)::(?P<url>https://\S+)"
)
NEW_CONFLICT_PATTERN = re.compile(r"NEW-[A-Za-z0-9][A-Za-z0-9._-]*")
CITE_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._/+\-]*")
CANDIDATE_ID_PATTERN = re.compile(r"C[0-9]{4,}")
YEAR_PATTERN = re.compile(r"[0-9]{4}")
DOI_PATTERN = re.compile(
    r"10\.[0-9]{4,9}/[a-z0-9._;()/:+\-]+"
)
PAPER_SOURCE_PATTERN = re.compile(
    r"paper|article|preprint|proceedings|journal|conference|publication|"
    r"survey|thesis|report|book"
)
NONPAPER_SOURCE_PATTERN = re.compile(
    r"software|documentation|standard|repository|benchmark|competition|"
    r"simulator|platform|release|dataset|artifact|tool|package|system"
)
INCOMPLETE_AUTHOR_PATTERN = re.compile(r"\bet[\W_]*al\b", re.IGNORECASE)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "via",
        "with",
    }
)

CandidateRow = dict[str, str]


@dataclass(frozen=True)
class ResultFilePair:
    metadata_path: Path
    conflict_path: Path


@dataclass(frozen=True)
class InputFingerprint:
    path: Path
    resolved_path: Path
    sha256: str


@dataclass(frozen=True)
class MetadataIntegrationResult:
    candidates: list[CandidateRow]
    conflicts: list[CandidateRow]
    bibliography: list[CandidateRow]
    bibtex: str
    citation_keys: list[CandidateRow]
    new_citation_keys: list[CandidateRow]
    citation_keys_bytes: bytes
    input_fingerprints: tuple[InputFingerprint, ...] = field(
        compare=False, repr=False
    )
    input_paths: tuple[Path, ...]

    @property
    def bibliography_rows(self) -> list[CandidateRow]:
        return self.bibliography

    @property
    def bibtex_text(self) -> str:
        return self.bibtex


IntegrationResult = MetadataIntegrationResult


@dataclass(frozen=True)
class _EvidenceToken:
    kind: str
    url: str

    @property
    def serialized(self) -> str:
        return f"{self.kind}::{self.url}"


def _read_rows(
    path: Path,
    header: tuple[str, ...],
    *,
    result_file: bool = False,
) -> list[CandidateRow]:
    path = Path(path)
    if not path.is_file():
        raise MetadataIntegrationError(f"{path}: file is missing")
    reader: csv.DictReader | None = None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual = tuple(reader.fieldnames or ())
            if actual != header:
                raise MetadataIntegrationError(
                    f"{path}: headers {actual!r} != {header!r}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise MetadataIntegrationError(
            f"{path}: invalid UTF-8: {exc}"
        ) from exc
    except csv.Error as exc:
        line_number = reader.line_num if reader is not None else 1
        raise MetadataIntegrationError(
            f"{path}:{line_number}: CSV parse error: {exc}"
        ) from exc

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise MetadataIntegrationError(
                f"{path}:{row_number}: malformed CSV row"
            )
        for field_name, value in row.items():
            stripped = value.strip()
            if value != stripped:
                raise MetadataIntegrationError(
                    f"{path}:{row_number}: {field_name} contains "
                    "surrounding whitespace"
                )
            row[field_name] = stripped
            if (
                result_file
                and stripped != "NR"
                and any(part.strip() == "NR" for part in stripped.split(";"))
            ):
                raise MetadataIntegrationError(
                    f"{path}:{row_number}: {field_name}: NR must be the sole sentinel"
                )
        if not any(value.strip() for value in row.values()):
            raise MetadataIntegrationError(
                f"{path}:{row_number}: row is entirely blank"
            )
        if result_file:
            blanks = [name for name, value in row.items() if not value.strip()]
            if blanks:
                raise MetadataIntegrationError(
                    f"{path}:{row_number}: blank result fields {blanks}; "
                    "use NR for values that were not reported"
                )
    return rows


def _capture_input_fingerprint(path: Path) -> InputFingerprint:
    absolute_path = Path(path).absolute()
    try:
        resolved_path = absolute_path.resolve(strict=True)
        payload = absolute_path.read_bytes()
    except OSError as exc:
        raise MetadataIntegrationError(
            f"{absolute_path}: cannot fingerprint immutable input: {exc}"
        ) from exc
    return InputFingerprint(
        path=absolute_path,
        resolved_path=resolved_path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _capture_input_fingerprints(
    paths: Sequence[Path],
) -> tuple[InputFingerprint, ...]:
    return tuple(_capture_input_fingerprint(path) for path in paths)


def _assert_input_fingerprints_current(
    fingerprints: Sequence[InputFingerprint],
) -> None:
    for fingerprint in fingerprints:
        try:
            resolved_path = fingerprint.path.resolve(strict=True)
            payload = fingerprint.path.read_bytes()
        except OSError as exc:
            raise MetadataIntegrationError(
                f"input {fingerprint.path} changed since integration: {exc}"
            ) from exc
        current_sha256 = hashlib.sha256(payload).hexdigest()
        if (
            resolved_path != fingerprint.resolved_path
            or current_sha256 != fingerprint.sha256
        ):
            raise MetadataIntegrationError(
                f"input {fingerprint.path} changed since integration"
            )


def _validate_citation_key_rows(
    rows: list[CandidateRow],
    candidates: list[CandidateRow],
    *,
    path: Path,
) -> None:
    candidate_ids = {row["candidate_id"] for row in candidates}
    seen_ids: dict[str, int] = {}
    seen_keys: dict[str, tuple[str, int]] = {}
    for row_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"]
        cite_key = row["cite_key"]
        if not candidate_id:
            raise MetadataIntegrationError(
                f"{path}:{row_number}: candidate_id is required"
            )
        if not cite_key:
            raise MetadataIntegrationError(
                f"{path}:{row_number}: cite_key is required"
            )
        if CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
            raise MetadataIntegrationError(
                f"{path}:{row_number}: candidate_id={candidate_id!r} must "
                "be C followed by at least four digits"
            )
        if candidate_id not in candidate_ids:
            raise MetadataIntegrationError(
                f"{path}:{row_number}: candidate_id={candidate_id!r} does "
                "not exist in candidates input"
            )
        if candidate_id in seen_ids:
            raise MetadataIntegrationError(
                f"{path}:{row_number}: duplicate candidate_id "
                f"{candidate_id!r}; first seen on row "
                f"{seen_ids[candidate_id]}"
            )
        seen_ids[candidate_id] = row_number

        if CITE_KEY_PATTERN.fullmatch(cite_key) is None:
            raise MetadataIntegrationError(
                f"{path}:{row_number}: cite_key={cite_key!r} is not "
                "BibTeX-safe"
            )
        folded_key = cite_key.casefold()
        if folded_key in seen_keys:
            first_key, first_row = seen_keys[folded_key]
            raise MetadataIntegrationError(
                f"{path}:{row_number}: case-insensitive duplicate cite_key "
                f"{cite_key!r}; first spelling {first_key!r} on row "
                f"{first_row}"
            )
        seen_keys[folded_key] = cite_key, row_number


def _normalize_result_pairs(
    result_pairs: Sequence[
        ResultFilePair | tuple[Path, Path] | list[Path] | Path
    ]
    | None,
    conflict_result_paths: Sequence[Path] | None,
    metadata_result_paths: Sequence[Path] | None,
) -> list[ResultFilePair]:
    if metadata_result_paths is not None:
        if result_pairs is not None:
            raise MetadataIntegrationError(
                "pass result_pairs or metadata_result_paths, not both"
            )
        result_pairs = list(metadata_result_paths)

    if conflict_result_paths is not None:
        metadata_paths = list(result_pairs or ())
        conflict_paths = list(conflict_result_paths)
        if len(metadata_paths) != len(conflict_paths):
            raise MetadataIntegrationError(
                "metadata and conflict result path counts differ"
            )
        normalized = [
            ResultFilePair(Path(metadata_path), Path(conflict_path))
            for metadata_path, conflict_path in zip(
                metadata_paths, conflict_paths, strict=True
            )
        ]
    else:
        normalized = []
        for item in result_pairs or ():
            if isinstance(item, ResultFilePair):
                normalized.append(
                    ResultFilePair(
                        Path(item.metadata_path), Path(item.conflict_path)
                    )
                )
                continue
            if isinstance(item, (str, bytes, Path)):
                raise MetadataIntegrationError(
                    "result_pairs entries must contain metadata and "
                    "conflict paths"
                )
            try:
                metadata_path, conflict_path = item
            except (TypeError, ValueError) as exc:
                raise MetadataIntegrationError(
                    "result_pairs entries must contain exactly two paths"
                ) from exc
            normalized.append(
                ResultFilePair(Path(metadata_path), Path(conflict_path))
            )

    if len(normalized) != SUPPORTED_BATCH_COUNT:
        raise MetadataIntegrationError(
            "metadata integration requires exactly 6 result-file pairs, "
            f"got {len(normalized)}"
        )
    all_paths = [
        path
        for pair in normalized
        for path in (pair.metadata_path, pair.conflict_path)
    ]
    for index, path in enumerate(all_paths):
        if any(_paths_alias(path, other) for other in all_paths[:index]):
            raise MetadataIntegrationError(
                f"result file paths must be distinct; duplicate {path}"
            )
    return normalized


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def _parse_absolute_url(
    value: str,
    *,
    context: str,
    schemes: frozenset[str],
) -> tuple[SplitResult, str]:
    if any(character.isspace() for character in value):
        raise MetadataIntegrationError(
            f"{context}: URL must not contain whitespace"
        )

    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        _ = parsed.port
        username = parsed.username
        password = parsed.password
    except ValueError as exc:
        raise MetadataIntegrationError(
            f"{context}: invalid URL authority or port: {exc}"
        ) from exc
    if (
        parsed.scheme.casefold() not in schemes
        or not parsed.netloc
        or not host
    ):
        expected = "/".join(sorted(scheme.upper() for scheme in schemes))
        raise MetadataIntegrationError(
            f"{context}: expected an absolute {expected} URL "
            "with a valid authority"
        )
    if username is not None or password is not None:
        raise MetadataIntegrationError(
            f"{context}: URL authority must not contain credentials"
        )
    normalized_host = host.casefold().rstrip(".")
    if not normalized_host:
        raise MetadataIntegrationError(
            f"{context}: URL authority must contain a hostname"
        )
    return parsed, normalized_host


def _classified_host_kind(parsed: SplitResult, host: str) -> str | None:
    if _host_matches(host, "crossref.org"):
        return "crossref"
    if _host_matches(host, "doi.org"):
        return "doi"
    for domain, kind in (
        ("openalex.org", "openalex"),
        ("semanticscholar.org", "semantic_scholar"),
        ("wikipedia.org", "wikipedia"),
        ("dblp.org", "dblp"),
    ):
        if _host_matches(host, domain):
            return kind
    if host.startswith("scholar.google."):
        return "google_scholar"
    if (
        (host == "google.com" or host.startswith("www.google."))
        and parsed.path.rstrip("/") == "/search"
    ):
        return "google"
    return None


def _discovery_host_kind(url: str) -> str | None:
    parsed, host = _parse_absolute_url(
        url,
        context="URL",
        schemes=frozenset({"http", "https"}),
    )
    return _classified_host_kind(parsed, host)


def _parse_evidence(
    value: str,
    *,
    context: str,
) -> tuple[_EvidenceToken, ...]:
    if value == "NR":
        return ()
    parts = value.split(";")
    if any(not part.strip() for part in parts):
        raise MetadataIntegrationError(
            f"{context}: malformed semicolon-separated evidence"
        )
    if any(part.strip() == "NR" for part in parts):
        raise MetadataIntegrationError(
            f"{context}: NR must be the sole evidence sentinel"
        )
    tokens = []
    for part in parts:
        serialized = part.strip()
        match = EVIDENCE_TOKEN_PATTERN.fullmatch(serialized)
        if match is None:
            raise MetadataIntegrationError(
                f"{context}: evidence must use source_kind::https://..."
            )
        kind = match.group("kind")
        url = match.group("url")
        parsed, host = _parse_absolute_url(
            url,
            context=f"{context} evidence URL",
            schemes=frozenset({"https"}),
        )
        host_kind = _classified_host_kind(parsed, host)
        if (
            host_kind in {"crossref", "doi"}
            and kind != host_kind
        ):
            source_name = "Crossref" if host_kind == "crossref" else "DOI"
            raise MetadataIntegrationError(
                f"{context}: {source_name} host cannot be relabeled as {kind!r}"
            )
        if host_kind not in {None, "crossref", "doi"}:
            raise MetadataIntegrationError(
                f"{context}: known discovery/aggregator host "
                f"{host!r} cannot be verification evidence"
            )
        if kind in DISCOVERY_SOURCE_KINDS:
            raise MetadataIntegrationError(
                f"{context}: discovery/aggregator source kind {kind!r} "
                "cannot be verification evidence"
            )
        if kind not in AUTHORITATIVE_SOURCE_KINDS | CORROBORATING_SOURCE_KINDS:
            raise MetadataIntegrationError(
                f"{context}: unsupported evidence source kind {kind!r}; "
                "discovery and aggregator sources are not accepted"
            )
        tokens.append(_EvidenceToken(kind, url))
    return tuple(tokens)


def _require_authoritative(
    tokens: tuple[_EvidenceToken, ...],
    *,
    context: str,
    allowed: frozenset[str] = AUTHORITATIVE_SOURCE_KINDS,
) -> None:
    if any(token.kind in allowed for token in tokens):
        return
    if tokens and all(
        token.kind in CORROBORATING_SOURCE_KINDS for token in tokens
    ):
        raise MetadataIntegrationError(
            f"{context}: Crossref may corroborate but cannot be the sole "
            "verification evidence"
        )
    raise MetadataIntegrationError(
        f"{context}: authoritative verification evidence is required from "
        f"one of {sorted(allowed)}"
    )


def _split_semicolon(value: str, *, context: str) -> list[str]:
    parts = value.split(";")
    stripped = [part.strip() for part in parts]
    if any(not part for part in stripped):
        raise MetadataIntegrationError(
            f"{context}: semicolon-separated values cannot be empty"
        )
    if "NR" in stripped:
        raise MetadataIntegrationError(
            f"{context}: NR must be the sole list sentinel"
        )
    return stripped


def _validate_authors(row: CandidateRow, *, context: str) -> None:
    authors_value = row["authors"]
    kinds_value = row["author_kinds"]
    if authors_value == "NR" or kinds_value == "NR":
        if authors_value != kinds_value:
            raise MetadataIntegrationError(
                f"{context}: authors and author_kinds must both be NR"
            )
        return
    authors = _split_semicolon(authors_value, context=f"{context} authors")
    kinds = _split_semicolon(
        kinds_value, context=f"{context} author_kinds"
    )
    if len(authors) != len(kinds):
        raise MetadataIntegrationError(
            f"{context}: author_kinds must align one-to-one with authors"
        )
    invalid = sorted(set(kinds) - AUTHOR_KINDS)
    if invalid:
        raise MetadataIntegrationError(
            f"{context}: invalid author kinds {invalid}"
        )
    if any(INCOMPLETE_AUTHOR_PATTERN.search(author) for author in authors):
        raise MetadataIntegrationError(
            f"{context}: verified authors must be complete, not et al."
        )


def _is_nonpaper_misc(row: CandidateRow) -> bool:
    source_type = row["source_type"].casefold()
    return (
        row["bib_entry_type"] == "misc"
        and bool(NONPAPER_SOURCE_PATTERN.search(source_type))
        and not PAPER_SOURCE_PATTERN.search(source_type)
    )


def _allowed_sources_for_bibliographic_field(
    row: CandidateRow,
    field_name: str,
) -> frozenset[str]:
    if _is_nonpaper_misc(row):
        return OFFICIAL_ARTIFACT_SOURCE_KINDS
    preprint_only = "preprint" in row["source_type"].casefold()
    if field_name in PAPER_CORE_EVIDENCE_FIELDS:
        return (
            PREPRINT_CORE_SOURCE_KINDS
            if preprint_only
            else PUBLISHED_CORE_SOURCE_KINDS
        )
    if field_name == "doi":
        return (
            PREPRINT_DOI_URL_SOURCE_KINDS
            if preprint_only
            else PUBLISHED_DOI_SOURCE_KINDS
        )
    if field_name == "url":
        return (
            PREPRINT_DOI_URL_SOURCE_KINDS
            if preprint_only
            else PUBLISHED_URL_SOURCE_KINDS
        )
    raise MetadataIntegrationError(
        f"unsupported bibliographic evidence field {field_name!r}"
    )


def _validate_url(value: str, *, context: str) -> None:
    _parse_absolute_url(
        value,
        context=context,
        schemes=frozenset({"http", "https"}),
    )


def _doi_resolver_path(value: str, *, context: str) -> str:
    parsed, host = _parse_absolute_url(
        value,
        context=context,
        schemes=frozenset({"http", "https"}),
    )
    if host not in DOI_RESOLVER_HOSTS:
        raise MetadataIntegrationError(
            f"{context}: DOI URL must use a canonical doi.org resolver"
        )
    path = unquote(parsed.path).lstrip("/")
    if not path:
        raise MetadataIntegrationError(
            f"{context}: DOI resolver URL requires a nonempty DOI path"
        )
    return path


def _canonical_doi(value: str, *, context: str) -> str:
    if value == "NR":
        return ""
    raw = value
    if raw.casefold().startswith("doi:"):
        raw = raw[4:]
    elif "://" in raw:
        raw = _doi_resolver_path(value, context=context)
    normalized = normalize_doi(raw.casefold())
    if DOI_PATTERN.fullmatch(normalized) is None:
        raise MetadataIntegrationError(
            f"{context}: malformed DOI {value!r}; expected "
            "10.<registrant>/<suffix>"
        )
    return normalized


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    return parsed._replace(
        scheme=parsed.scheme.casefold(),
        netloc=parsed.netloc.casefold(),
    ).geturl()


def _is_stable_url(value: str) -> bool:
    try:
        parsed, host = _parse_absolute_url(
            value,
            context="stable URL",
            schemes=frozenset({"http", "https"}),
        )
    except MetadataIntegrationError:
        return False
    host_kind = _classified_host_kind(parsed, host)
    if host_kind == "doi":
        return bool(_doi_from_resolver_url(value))
    if host_kind is not None:
        return False
    return True


def _substantive_notes(value: str) -> bool:
    if value == "NR":
        return False
    words = re.findall(r"[A-Za-z0-9]+", value.casefold())
    placeholders = {
        ("n", "a"),
        ("not", "available"),
        ("not", "reported"),
        ("none",),
        ("unknown",),
    }
    return len(words) >= 3 and tuple(words) not in placeholders


def _validate_bibliography_url(
    row: CandidateRow,
    evidence: dict[str, tuple[_EvidenceToken, ...]],
    *,
    context: str,
) -> None:
    if row["bib_url"] == "NR":
        return

    _, bibliography_host = _parse_absolute_url(
        row["bib_url"],
        context=f"{context} bib_url",
        schemes=frozenset({"http", "https"}),
    )
    if bibliography_host in DOI_RESOLVER_HOSTS:
        canonical_doi = _canonical_doi(
            row["doi"], context=f"{context} doi"
        )
        resolver_doi = _doi_from_resolver_url(row["bib_url"])
        if not canonical_doi or resolver_doi != canonical_doi:
            raise MetadataIntegrationError(
                f"{context}: DOI resolver bib_url requires a nonempty "
                "matching doi"
            )
        return

    bibliography_url = _canonical_url(row["bib_url"])
    if (
        row["url"] != "NR"
        and bibliography_url == _canonical_url(row["url"])
    ):
        return
    evidence_urls = {
        _canonical_url(token.url)
        for tokens in evidence.values()
        for token in tokens
        if token.kind in AUTHORITATIVE_SOURCE_KINDS
    }
    if bibliography_url not in evidence_urls:
        raise MetadataIntegrationError(
            f"{context}: bib_url must match the candidate URL or "
            "authoritative field evidence"
        )


def _validate_verified_row(
    row: CandidateRow,
    evidence: dict[str, tuple[_EvidenceToken, ...]],
    *,
    context: str,
) -> None:
    for field_name in ("title", "authors", "source_type"):
        if row[field_name] == "NR":
            raise MetadataIntegrationError(
                f"{context}: verified {field_name} cannot be NR"
            )
    if row["year"] != "NR" and YEAR_PATTERN.fullmatch(row["year"]) is None:
        raise MetadataIntegrationError(
            f"{context}: year must be four digits or NR"
        )
    for field_name in ("url", "bib_url"):
        if row[field_name] != "NR":
            _validate_url(row[field_name], context=f"{context} {field_name}")
    canonical_doi = _canonical_doi(
        row["doi"], context=f"{context} doi"
    )

    _validate_authors(row, context=context)
    if row["bib_entry_type"] not in SUPPORTED_ENTRY_TYPES:
        raise MetadataIntegrationError(
            f"{context}: unsupported bib_entry_type "
            f"{row['bib_entry_type']!r}"
        )
    key_author = row["key_author"]
    if not key_author or key_author == "NR":
        raise MetadataIntegrationError(
            f"{context}: verified key_author cannot be empty or NR"
        )
    if INCOMPLETE_AUTHOR_PATTERN.search(key_author):
        raise MetadataIntegrationError(
            f"{context}: verified key_author must be complete, not et al."
        )

    expected_venue_field = VENUE_FIELD_BY_ENTRY_TYPE[row["bib_entry_type"]]
    if row["venue"] == "NR":
        if row["bib_venue_field"] != "NR":
            raise MetadataIntegrationError(
                f"{context}: bib_venue_field must be NR when venue is NR"
            )
    elif row["bib_venue_field"] != expected_venue_field:
        raise MetadataIntegrationError(
            f"{context}: {row['bib_entry_type']} requires "
            f"bib_venue_field={expected_venue_field!r}"
        )

    _validate_bibliography_url(row, evidence, context=context)

    nonpaper_misc = _is_nonpaper_misc(row)
    if nonpaper_misc:
        if row["url"] == "NR":
            raise MetadataIntegrationError(
                f"{context}: verified non-paper artifact requires an "
                "official URL"
            )
        for field_name in BIBLIOGRAPHIC_FIELDS:
            if row[field_name] != "NR":
                _require_authoritative(
                    evidence[field_name],
                    context=f"{context} {field_name}_evidence",
                    allowed=_allowed_sources_for_bibliographic_field(
                        row, field_name
                    ),
                )
        if any(row[field] == "NR" for field in ("year", "venue", "doi")):
            if not _substantive_notes(row["notes"]):
                raise MetadataIntegrationError(
                    f"{context}: substantive notes must explain NR year, "
                    "venue, or DOI"
                )
        return

    if not canonical_doi and (
        row["url"] == "NR" or not _is_stable_url(row["url"])
    ):
        raise MetadataIntegrationError(
            f"{context}: verified paper requires a canonical DOI or stable URL"
        )
    for field_name in ("year", "venue"):
        if row[field_name] == "NR":
            raise MetadataIntegrationError(
                f"{context}: verified paper-like {field_name} cannot be NR"
            )
    for field_name in BIBLIOGRAPHIC_FIELDS:
        if row[field_name] != "NR":
            _require_authoritative(
                evidence[field_name],
                context=f"{context} {field_name}_evidence",
                allowed=_allowed_sources_for_bibliographic_field(
                    row, field_name
                ),
            )


def _validate_metadata_row(row: CandidateRow) -> tuple[str, ...]:
    candidate_id = row["candidate_id"]
    context = f"candidate_id={candidate_id!r}"
    status = row["metadata_status"]
    if status not in METADATA_STATUS_VALUES:
        raise MetadataIntegrationError(
            f"{context}: invalid metadata_status {status!r}"
        )
    if row["verified_on"] != "NR":
        try:
            date.fromisoformat(row["verified_on"])
        except ValueError as exc:
            raise MetadataIntegrationError(
                f"{context}: verified_on must be an ISO date or NR"
            ) from exc
    if status == "verified" and row["verified_on"] == "NR":
        raise MetadataIntegrationError(
            f"{context}: verified metadata requires verified_on"
        )
    if row["doi"] != "NR":
        _canonical_doi(
            row["doi"], context=f"{context} doi"
        )

    evidence: dict[str, tuple[_EvidenceToken, ...]] = {}
    all_tokens = set()
    for field_name, evidence_field in EVIDENCE_FIELDS:
        tokens = _parse_evidence(
            row[evidence_field], context=f"{context} {evidence_field}"
        )
        evidence[field_name] = tokens
        all_tokens.update(token.serialized for token in tokens)
    _validate_authors(row, context=context)
    if status == "verified":
        _validate_verified_row(row, evidence, context=context)
    return tuple(sorted(all_tokens, key=lambda value: (value.casefold(), value)))


def _conflict_candidate_ids(
    candidates: list[CandidateRow], conflicts: list[CandidateRow]
) -> dict[str, str]:
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    candidates_by_key: defaultdict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        if candidate["cite_key"]:
            candidates_by_key[candidate["cite_key"]].append(
                candidate["candidate_id"]
            )

    mapped = {}
    for conflict in conflicts:
        if conflict["record_type"] == "candidate":
            candidate_id = conflict["record_key"]
            if candidate_id not in candidates_by_id:
                raise MetadataIntegrationError(
                    f"conflict {conflict['conflict_id']!r} is orphaned"
                )
        elif conflict["record_type"] == "evidence":
            matches = candidates_by_key.get(conflict["record_key"], [])
            if len(matches) != 1:
                raise MetadataIntegrationError(
                    f"conflict {conflict['conflict_id']!r} evidence key "
                    "does not identify one candidate"
                )
            candidate_id = matches[0]
        else:
            raise MetadataIntegrationError(
                f"conflict {conflict['conflict_id']!r} has unsupported "
                f"record_type {conflict['record_type']!r}"
            )
        mapped[conflict["conflict_id"]] = candidate_id
    return mapped


def _check_result_assignment(
    metadata_rows: list[CandidateRow],
    conflict_rows: list[CandidateRow],
    manifest_by_id: dict[str, CandidateRow],
    *,
    pair: ResultFilePair,
) -> str | None:
    agent_ids = {
        row["agent_id"] for row in (*metadata_rows, *conflict_rows)
    }
    if len(agent_ids) > 1:
        raise MetadataIntegrationError(
            f"result-file pair {pair.metadata_path}, {pair.conflict_path} "
            f"impersonates multiple agent_id values {sorted(agent_ids)}"
        )
    agent_id = next(iter(agent_ids), None)
    for row in (*metadata_rows, *conflict_rows):
        candidate_id = row["candidate_id"]
        manifest_row = manifest_by_id.get(candidate_id)
        if manifest_row is None:
            continue
        if row["input_sha256"] != manifest_row["input_sha256"]:
            raise MetadataIntegrationError(
                f"candidate_id={candidate_id!r}: stale input_sha256"
            )
        if row["agent_id"] != manifest_row["batch_id"]:
            raise MetadataIntegrationError(
                f"candidate_id={candidate_id!r}: agent_id "
                f"{row['agent_id']!r} does not match assigned batch "
                f"{manifest_row['batch_id']!r}"
            )
    return agent_id


def _candidate_result_map(
    rows: list[CandidateRow], manifest_by_id: dict[str, CandidateRow]
) -> dict[str, CandidateRow]:
    counts = Counter(row["candidate_id"] for row in rows)
    duplicate = sorted(candidate_id for candidate_id, count in counts.items() if count > 1)
    if duplicate:
        raise MetadataIntegrationError(
            f"duplicate candidate metadata results {duplicate}"
        )
    actual = set(counts)
    expected = set(manifest_by_id)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise MetadataIntegrationError(
            "candidate metadata result mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return {row["candidate_id"]: row for row in rows}


def _validate_conflict_result_identity(
    result_row: CandidateRow,
    input_row: CandidateRow,
    expected_candidate_id: str,
) -> None:
    expected = {
        "candidate_id": expected_candidate_id,
        "field": input_row["field"],
        "value_a": input_row["value_a"],
        "value_b": input_row["value_b"],
    }
    for field_name, value in expected.items():
        if result_row[field_name] != value:
            raise MetadataIntegrationError(
                f"conflict {input_row['conflict_id']!r}: {field_name} "
                f"does not match frozen input; expected {value!r}, "
                f"found {result_row[field_name]!r}"
            )


def _apply_resolution(
    conflict: CandidateRow,
    result_row: CandidateRow,
    candidate: CandidateRow,
    metadata_row: CandidateRow,
) -> None:
    resolution = result_row["resolution"]
    resolution_evidence = result_row["resolution_evidence"]
    if resolution == "NR":
        if resolution_evidence != "NR":
            raise MetadataIntegrationError(
                f"conflict {conflict['conflict_id']!r}: unresolved result "
                "must use NR resolution_evidence"
            )
        return
    if resolution_evidence == "NR":
        raise MetadataIntegrationError(
            f"conflict {conflict['conflict_id']!r}: resolution requires "
            "authoritative evidence"
        )
    canonical_resolution = (
        _canonical_doi(
            resolution,
            context=f"conflict {conflict['conflict_id']!r} resolution",
        )
        if conflict["field"] == "doi"
        else resolution
    )
    canonical = candidate[conflict["field"]]
    if not canonical or canonical_resolution != canonical:
        raise MetadataIntegrationError(
            f"conflict {conflict['conflict_id']!r}: resolution must equal "
            f"canonical output value {canonical!r}"
        )
    tokens = _parse_evidence(
        resolution_evidence,
        context=f"conflict {conflict['conflict_id']!r} resolution_evidence",
    )
    _require_authoritative(
        tokens,
        context=f"conflict {conflict['conflict_id']!r} resolution_evidence",
        allowed=_allowed_sources_for_bibliographic_field(
            metadata_row, conflict["field"]
        ),
    )
    if conflict["resolution"]:
        existing_resolution = (
            _canonical_doi(
                conflict["resolution"],
                context=f"conflict {conflict['conflict_id']!r} existing resolution",
            )
            if conflict["field"] == "doi"
            else conflict["resolution"]
        )
        if existing_resolution != canonical_resolution:
            raise MetadataIntegrationError(
                f"conflict {conflict['conflict_id']!r}: existing resolution "
                "cannot be replaced"
            )
        conflict["resolution"] = canonical_resolution
        return
    conflict["resolution"] = canonical_resolution
    conflict["resolver"] = result_row["agent_id"]
    conflict["resolution_evidence"] = resolution_evidence


def _normalize_title(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    characters = []
    for character in decomposed:
        category = unicodedata.category(character)
        if category.startswith("M"):
            continue
        characters.append(character if character.isalnum() else " ")
    return " ".join("".join(characters).split())


def _conflict_value(field_name: str, value: str) -> str:
    if field_name == "doi":
        return _canonical_doi(value, context="conflict DOI value")
    if field_name == "title":
        return _normalize_title(value)
    return value


def _semantic_conflict_signature(
    candidate_id: str,
    field_name: str,
    value_a: str,
    value_b: str,
) -> tuple[str, str, str, str]:
    normalized_a = _conflict_value(field_name, value_a)
    normalized_b = _conflict_value(field_name, value_b)
    if normalized_a == normalized_b:
        raise MetadataIntegrationError(
            f"candidate_id={candidate_id!r} field={field_name!r}: "
            "conflict values are semantically equivalent and do not disagree"
        )
    ordered = sorted(
        (normalized_a, normalized_b),
        key=lambda value: (value.casefold(), value),
    )
    return candidate_id, field_name, ordered[0], ordered[1]


def _stable_new_conflict(
    result_row: CandidateRow,
    candidate: CandidateRow,
    metadata_row: CandidateRow,
) -> CandidateRow:
    field_name = result_row["field"]
    if field_name not in BIBLIOGRAPHIC_FIELDS:
        raise MetadataIntegrationError(
            f"{result_row['input_conflict_id']!r}: new metadata conflict "
            f"field {field_name!r} is not bibliographic"
        )
    if result_row["value_a"] == "NR" or result_row["value_b"] == "NR":
        raise MetadataIntegrationError(
            f"{result_row['input_conflict_id']!r}: new conflict values "
            "cannot be NR"
        )
    _semantic_conflict_signature(
        candidate["candidate_id"],
        field_name,
        result_row["value_a"],
        result_row["value_b"],
    )

    value_a = result_row["value_a"]
    value_b = result_row["value_b"]
    if (
        _conflict_value(field_name, value_b).casefold(),
        value_b.casefold(),
        value_b,
    ) < (
        _conflict_value(field_name, value_a).casefold(),
        value_a.casefold(),
        value_a,
    ):
        value_a, value_b = value_b, value_a
    signature = "\0".join(
        (
            "candidate",
            candidate["candidate_id"],
            field_name,
            _conflict_value(field_name, value_a),
            _conflict_value(field_name, value_b),
        )
    )
    conflict_id = "X" + hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:12].upper()
    conflict = dict.fromkeys(CONFLICT_HEADER, "")
    conflict.update(
        conflict_id=conflict_id,
        record_type="candidate",
        record_key=candidate["candidate_id"],
        field=field_name,
        value_a=value_a,
        value_b=value_b,
    )
    _apply_resolution(conflict, result_row, candidate, metadata_row)
    return conflict


def _integrate_conflicts(
    input_conflicts: list[CandidateRow],
    conflict_candidate_ids: dict[str, str],
    result_rows: list[CandidateRow],
    candidates_by_id: dict[str, CandidateRow],
    metadata_by_id: dict[str, CandidateRow],
) -> list[CandidateRow]:
    input_by_id = {row["conflict_id"]: row for row in input_conflicts}
    seen_existing: Counter[str] = Counter()
    seen_new_local: set[tuple[str, str]] = set()
    output_by_id = {
        row["conflict_id"]: dict(row) for row in input_conflicts
    }
    semantic_signatures: set[tuple[str, str, str, str]] = set()
    for conflict in input_conflicts:
        if (
            conflict["record_type"] != "candidate"
            or conflict["field"] not in BIBLIOGRAPHIC_FIELDS
        ):
            continue
        signature = _semantic_conflict_signature(
            conflict["record_key"],
            conflict["field"],
            conflict["value_a"],
            conflict["value_b"],
        )
        if signature in semantic_signatures:
            raise MetadataIntegrationError(
                f"duplicate frozen conflict semantics for {signature[:2]}"
            )
        semantic_signatures.add(signature)

    ordered_results = sorted(
        result_rows,
        key=lambda row: (
            row["agent_id"],
            row["input_conflict_id"],
            row["candidate_id"],
            row["field"],
            row["value_a"],
            row["value_b"],
        ),
    )
    for result_row in ordered_results:
        input_conflict_id = result_row["input_conflict_id"]
        if input_conflict_id in input_by_id:
            seen_existing[input_conflict_id] += 1
            input_row = input_by_id[input_conflict_id]
            _validate_conflict_result_identity(
                result_row,
                input_row,
                conflict_candidate_ids[input_conflict_id],
            )
            conflict = output_by_id[input_conflict_id]
            if (
                conflict["record_type"] != "candidate"
                or conflict["field"] not in BIBLIOGRAPHIC_FIELDS
            ):
                if (
                    result_row["resolution"] != "NR"
                    or result_row["resolution_evidence"] != "NR"
                ):
                    raise MetadataIntegrationError(
                        f"conflict {input_conflict_id!r}: metadata agents "
                        "cannot resolve screening/evidence conflicts"
                    )
                continue
            _apply_resolution(
                conflict,
                result_row,
                candidates_by_id[result_row["candidate_id"]],
                metadata_by_id[result_row["candidate_id"]],
            )
            continue

        if NEW_CONFLICT_PATTERN.fullmatch(input_conflict_id) is None:
            raise MetadataIntegrationError(
                f"unknown input_conflict_id {input_conflict_id!r}"
            )
        local_key = (result_row["agent_id"], input_conflict_id)
        if local_key in seen_new_local:
            raise MetadataIntegrationError(
                f"duplicate new conflict ID {input_conflict_id!r} in "
                f"{result_row['agent_id']}"
            )
        seen_new_local.add(local_key)
        semantic_signature = _semantic_conflict_signature(
            result_row["candidate_id"],
            result_row["field"],
            result_row["value_a"],
            result_row["value_b"],
        )
        if semantic_signature in semantic_signatures:
            raise MetadataIntegrationError(
                "new disagreement is a semantic duplicate of a frozen or "
                f"new conflict: {semantic_signature[:2]}"
            )
        semantic_signatures.add(semantic_signature)
        conflict = _stable_new_conflict(
            result_row,
            candidates_by_id[result_row["candidate_id"]],
            metadata_by_id[result_row["candidate_id"]],
        )
        if conflict["conflict_id"] in output_by_id:
            raise MetadataIntegrationError(
                f"new disagreement duplicates stable conflict "
                f"{conflict['conflict_id']}"
            )
        output_by_id[conflict["conflict_id"]] = conflict

    duplicate = sorted(
        conflict_id
        for conflict_id, count in seen_existing.items()
        if count > 1
    )
    missing = sorted(set(input_by_id) - set(seen_existing))
    if duplicate:
        raise MetadataIntegrationError(
            f"duplicate existing conflict results {duplicate}"
        )
    if missing:
        raise MetadataIntegrationError(
            f"missing existing conflict results {missing}"
        )
    return sorted(
        output_by_id.values(),
        key=lambda row: (
            row["record_type"],
            row["record_key"].casefold(),
            row["record_key"],
            row["field"],
            row["conflict_id"],
        ),
    )


def _ascii_alphanumeric(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = decomposed.encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in ascii_value if character.isalnum())


def _title_key_tokens(title: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_value = decomposed.encode("ascii", "ignore").decode("ascii")
    raw_tokens = re.findall(r"[A-Za-z0-9]+", ascii_value)
    tokens = [token for token in raw_tokens if token.casefold() not in STOPWORDS]
    return [token[0].upper() + token[1:] for token in tokens[:2]]


def _citation_base(row: CandidateRow, metadata_row: CandidateRow) -> str:
    candidate_fallback = f"Candidate{row['candidate_id'][1:]}"
    author = (
        _ascii_alphanumeric(metadata_row["key_author"])
        or candidate_fallback
    )
    year = row["year"] if YEAR_PATTERN.fullmatch(row["year"]) else "Nodate"
    title = "".join(_title_key_tokens(row["title"]))
    if not title and author != candidate_fallback:
        title = candidate_fallback
    return f"{author}{year}{title}"


def _letter_suffix(number: int) -> str:
    value = number + 1
    characters = []
    while value:
        value, remainder = divmod(value - 1, 26)
        characters.append(chr(ord("a") + remainder))
    return "".join(reversed(characters))


def _candidate_id_sort_key(candidate_id: str) -> tuple[int, str]:
    return int(candidate_id[1:]), candidate_id


def _assign_citation_keys_from_ledger(
    candidates: list[CandidateRow],
    original_by_id: dict[str, CandidateRow],
    metadata_by_id: dict[str, CandidateRow],
    citation_keys: list[CandidateRow],
    *,
    extend: bool,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    ledger_by_id = {
        row["candidate_id"]: row["cite_key"] for row in citation_keys
    }
    for candidate_id, original in original_by_id.items():
        snapshot_key = original["cite_key"]
        if snapshot_key and ledger_by_id.get(candidate_id) != snapshot_key:
            raise MetadataIntegrationError(
                f"candidate_id={candidate_id!r}: snapshot cite_key "
                f"{snapshot_key!r} does not match citation key ledger"
            )

    candidates_by_id = {
        row["candidate_id"]: row for row in candidates
    }
    active_ids = sorted(
        (
            candidate_id
            for candidate_id, candidate in candidates_by_id.items()
            if candidate["metadata_status"] == "verified"
            and candidate["screening_status"] != "excluded"
        ),
        key=_candidate_id_sort_key,
    )
    missing_ids = [
        candidate_id
        for candidate_id in active_ids
        if candidate_id not in ledger_by_id
    ]
    if missing_ids and not extend:
        raise MetadataIntegrationError(
            f"candidate_id={missing_ids[0]!r}: active candidate requires "
            "a citation key ledger assignment"
        )

    used = {key.casefold() for key in ledger_by_id.values()}
    bases = {
        candidate_id: _citation_base(
            candidates_by_id[candidate_id], metadata_by_id[candidate_id]
        )
        for candidate_id in missing_ids
    }
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for candidate_id, base in bases.items():
        groups[base.casefold()].append(candidate_id)

    for folded_base in sorted(groups):
        member_ids = sorted(
            groups[folded_base], key=_candidate_id_sort_key
        )
        canonical_base = bases[member_ids[0]]
        collision = len(member_ids) > 1 or folded_base in used
        if not collision:
            ledger_by_id[member_ids[0]] = canonical_base
            used.add(folded_base)
            continue

        suffix_number = 0
        for candidate_id in member_ids:
            while True:
                proposed = canonical_base + _letter_suffix(suffix_number)
                suffix_number += 1
                if proposed.casefold() not in used:
                    break
            ledger_by_id[candidate_id] = proposed
            used.add(proposed.casefold())

    new_citation_keys = [
        {
            "candidate_id": candidate_id,
            "cite_key": ledger_by_id[candidate_id],
        }
        for candidate_id in missing_ids
    ]
    for candidate_id, candidate in candidates_by_id.items():
        candidate["cite_key"] = (
            ledger_by_id[candidate_id]
            if candidate_id in active_ids
            else ""
        )
    return [*citation_keys, *new_citation_keys], new_citation_keys


def _doi_from_resolver_url(value: str) -> str:
    try:
        path = _doi_resolver_path(value, context="DOI resolver URL")
        return _canonical_doi(path, context="DOI resolver URL")
    except MetadataIntegrationError:
        return ""


def _bibliography_rows(
    candidates: list[CandidateRow], metadata_by_id: dict[str, CandidateRow]
) -> list[CandidateRow]:
    rows = []
    for candidate in candidates:
        if not candidate["cite_key"]:
            continue
        metadata = metadata_by_id[candidate["candidate_id"]]
        url = "" if metadata["bib_url"] == "NR" else metadata["bib_url"]
        if (
            url
            and candidate["doi"]
            and _doi_from_resolver_url(url) == normalize_doi(candidate["doi"])
        ):
            url = ""
        row = dict.fromkeys(BIBLIOGRAPHY_HEADER, "")
        row.update(
            candidate_id=candidate["candidate_id"],
            cite_key=candidate["cite_key"],
            entry_type=metadata["bib_entry_type"],
            key_author=metadata["key_author"],
            authors=candidate["authors"],
            author_kinds=metadata["author_kinds"],
            title=candidate["title"],
            year=candidate["year"],
            venue_field=(
                "" if metadata["bib_venue_field"] == "NR"
                else metadata["bib_venue_field"]
            ),
            venue=candidate["venue"],
            doi=candidate["doi"],
            url=url,
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["cite_key"].casefold(),
            row["cite_key"],
            row["candidate_id"],
        ),
    )


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
    return "".join(LATEX_ESCAPES.get(character, character) for character in value)


def _bibtex_authors(authors_value: str, kinds_value: str) -> str:
    authors = _split_semicolon(authors_value, context="bibliography authors")
    kinds = _split_semicolon(kinds_value, context="bibliography author_kinds")
    rendered = []
    for author, kind in zip(authors, kinds, strict=True):
        escaped = _latex_escape(author)
        rendered.append(f"{{{escaped}}}" if kind == "corporate" else escaped)
    return " and ".join(rendered)


def render_bibtex(rows: Sequence[CandidateRow]) -> str:
    for row in rows:
        for field_name in BIBLIOGRAPHY_HEADER:
            if "\r" in row[field_name]:
                raise MetadataIntegrationError(
                    f"bibliography field {field_name!r} contains "
                    "carriage return"
                )

    entries = []
    for row in sorted(
        rows,
        key=lambda item: (item["cite_key"].casefold(), item["cite_key"]),
    ):
        fields = [
            ("author", _bibtex_authors(row["authors"], row["author_kinds"])),
            ("title", _latex_escape(row["title"])),
        ]
        if row["venue_field"] and row["venue"]:
            fields.append((row["venue_field"], _latex_escape(row["venue"])))
        fields.extend(
            (field_name, _latex_escape(row[field_name]))
            for field_name in ("year", "doi", "url")
            if row[field_name]
        )
        lines = [f"@{row['entry_type']}{{{row['cite_key']},"]
        lines.extend(
            f"  {field_name} = {{{value}}}{',' if index < len(fields) - 1 else ''}"
            for index, (field_name, value) in enumerate(fields)
        )
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + ("\n" if entries else "")


def integrate_metadata(
    candidates_path: Path,
    conflicts_path: Path,
    manifest_path: Path,
    result_pairs: Sequence[
        ResultFilePair | tuple[Path, Path] | list[Path] | Path
    ]
    | None = None,
    conflict_result_paths: Sequence[Path] | None = None,
    *,
    metadata_result_paths: Sequence[Path] | None = None,
    citation_keys_path: Path,
    extend_citation_keys: bool = False,
) -> MetadataIntegrationResult:
    """Validate immutable agent results and derive canonical citation outputs.

    This function performs no writes. ``result_pairs`` may be supplied in any
    order. For CLI-style callers, metadata paths and conflict paths may instead
    be passed as parallel sequences.
    """
    candidates_path = Path(candidates_path)
    conflicts_path = Path(conflicts_path)
    manifest_path = Path(manifest_path)
    citation_keys_path = Path(citation_keys_path)

    normalized_pairs = _normalize_result_pairs(
        result_pairs, conflict_result_paths, metadata_result_paths
    )
    input_source_paths = (
        candidates_path,
        conflicts_path,
        manifest_path,
        citation_keys_path,
        *sorted(
            (
                Path(path)
                for pair in normalized_pairs
                for path in (pair.metadata_path, pair.conflict_path)
            ),
            key=lambda path: path.as_posix(),
        ),
    )
    # Capture every immutable input before validation so a mutation during the
    # manifest gate cannot become the accepted fingerprint baseline.
    input_fingerprints = _capture_input_fingerprints(input_source_paths)
    validate_manifest_inputs(manifest_path, candidates_path, conflicts_path)
    _assert_input_fingerprints_current(input_fingerprints)
    citation_keys_input_bytes = citation_keys_path.read_bytes()

    input_candidates = _read_rows(candidates_path, CANDIDATE_HEADER)
    citation_keys = _read_rows(
        citation_keys_path, CITATION_KEYS_HEADER
    )
    _validate_citation_key_rows(
        citation_keys,
        input_candidates,
        path=citation_keys_path,
    )
    input_conflicts = _read_rows(conflicts_path, CONFLICT_HEADER)
    manifest_rows = _read_rows(manifest_path, MANIFEST_HEADER)
    manifest_by_id = {
        row["candidate_id"]: row for row in manifest_rows
    }

    metadata_rows = []
    conflict_rows = []
    claimed_agents: dict[str, ResultFilePair] = {}
    for pair in normalized_pairs:
        pair_metadata = _read_rows(
            pair.metadata_path,
            METADATA_RESULT_HEADER,
            result_file=True,
        )
        pair_conflicts = _read_rows(
            pair.conflict_path,
            CONFLICT_RESULT_HEADER,
            result_file=True,
        )
        agent_id = _check_result_assignment(
            pair_metadata,
            pair_conflicts,
            manifest_by_id,
            pair=pair,
        )
        if agent_id is not None:
            if agent_id in claimed_agents:
                raise MetadataIntegrationError(
                    f"duplicate result-file pairs claim agent_id {agent_id!r}"
                )
            claimed_agents[agent_id] = pair
        metadata_rows.extend(pair_metadata)
        conflict_rows.extend(pair_conflicts)

    metadata_by_id = _candidate_result_map(metadata_rows, manifest_by_id)
    expected_agents = {row["batch_id"] for row in manifest_rows}
    missing_agents = sorted(expected_agents - set(claimed_agents))
    if missing_agents:
        raise MetadataIntegrationError(
            f"missing result-file pairs for assigned batches {missing_agents}"
        )

    conflict_candidate_ids = _conflict_candidate_ids(
        input_candidates, input_conflicts
    )
    input_conflicts_by_id = {
        row["conflict_id"]: row for row in input_conflicts
    }
    for result_row in conflict_rows:
        candidate_id = result_row["candidate_id"]
        manifest_row = manifest_by_id.get(candidate_id)
        if manifest_row is None:
            raise MetadataIntegrationError(
                f"extra conflict result for unknown candidate_id "
                f"{candidate_id!r}"
            )
        input_conflict_id = result_row["input_conflict_id"]
        if input_conflict_id in input_conflicts_by_id:
            expected_candidate = conflict_candidate_ids[input_conflict_id]
            if expected_candidate != candidate_id:
                raise MetadataIntegrationError(
                    f"conflict {input_conflict_id!r}: candidate_id does not "
                    "match frozen input"
                )

    original_by_id = {
        row["candidate_id"]: dict(row) for row in input_candidates
    }
    integrated_candidates = []
    for original in input_candidates:
        candidate_id = original["candidate_id"]
        metadata = metadata_by_id[candidate_id]
        evidence_tokens = _validate_metadata_row(metadata)
        integrated = dict(original)
        for field_name in BIBLIOGRAPHIC_FIELDS:
            value = metadata[field_name]
            if value == "NR":
                integrated[field_name] = ""
            elif field_name == "doi":
                integrated[field_name] = _canonical_doi(
                    value,
                    context=f"candidate_id={candidate_id!r} doi",
                )
            else:
                integrated[field_name] = value
        integrated["metadata_status"] = metadata["metadata_status"]
        integrated["metadata_evidence"] = ";".join(evidence_tokens)
        for field_name in IMMUTABLE_CANDIDATE_FIELDS:
            if integrated[field_name] != original[field_name]:
                raise MetadataIntegrationError(
                    f"candidate_id={candidate_id!r}: immutable field "
                    f"{field_name!r} changed"
                )
        integrated_candidates.append(integrated)
    integrated_candidates.sort(key=lambda row: row["candidate_id"])
    candidates_by_id = {
        row["candidate_id"]: row for row in integrated_candidates
    }

    integrated_conflicts = _integrate_conflicts(
        input_conflicts,
        conflict_candidate_ids,
        conflict_rows,
        candidates_by_id,
        metadata_by_id,
    )
    unresolved_by_candidate: defaultdict[str, list[str]] = defaultdict(list)
    for conflict in integrated_conflicts:
        if (
            conflict["record_type"] == "candidate"
            and conflict["field"] in BIBLIOGRAPHIC_FIELDS
            and not conflict["resolution"]
        ):
            unresolved_by_candidate[conflict["record_key"]].append(
                conflict["conflict_id"]
            )
    for candidate_id, candidate in candidates_by_id.items():
        conflict_ids = unresolved_by_candidate.get(candidate_id, [])
        has_unresolved = bool(conflict_ids)
        has_conflict_status = candidate["metadata_status"] == "conflict"
        if has_unresolved != has_conflict_status:
            raise MetadataIntegrationError(
                f"candidate_id={candidate_id!r}: metadata_status=conflict "
                "if and only if at least one bibliographic candidate "
                f"conflict is unresolved; unresolved={sorted(conflict_ids)}"
            )
        if (
            candidate["screening_status"] in {"included", "boundary"}
            and candidate["metadata_status"] != "verified"
        ):
            raise MetadataIntegrationError(
                f"candidate_id={candidate_id!r}: included/boundary "
                "candidates must finish metadata_status=verified"
            )

    citation_keys, new_citation_keys = _assign_citation_keys_from_ledger(
        integrated_candidates,
        original_by_id,
        metadata_by_id,
        citation_keys,
        extend=extend_citation_keys,
    )
    for candidate in integrated_candidates:
        if (
            candidate["screening_status"] in {"included", "boundary"}
            and not candidate["cite_key"]
        ):
            raise MetadataIntegrationError(
                f"candidate_id={candidate['candidate_id']!r}: "
                "included/boundary verified candidate requires cite_key"
            )
    bibliography = _bibliography_rows(
        integrated_candidates, metadata_by_id
    )
    bibtex = render_bibtex(bibliography)
    citation_keys_bytes = _append_citation_key_rows(
        citation_keys_input_bytes,
        new_citation_keys,
    )
    _assert_input_fingerprints_current(input_fingerprints)
    immutable_input_paths = tuple(
        fingerprint.resolved_path for fingerprint in input_fingerprints
    )
    return MetadataIntegrationResult(
        candidates=integrated_candidates,
        conflicts=integrated_conflicts,
        bibliography=bibliography,
        bibtex=bibtex,
        citation_keys=citation_keys,
        new_citation_keys=new_citation_keys,
        citation_keys_bytes=citation_keys_bytes,
        input_fingerprints=input_fingerprints,
        input_paths=immutable_input_paths,
    )


def _csv_bytes(
    header: tuple[str, ...], rows: Iterable[CandidateRow]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=header,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _append_citation_key_rows(
    source_bytes: bytes,
    new_rows: list[CandidateRow],
) -> bytes:
    if not new_rows:
        return source_bytes
    rendered = _csv_bytes(CITATION_KEYS_HEADER, new_rows)
    header = _csv_bytes(CITATION_KEYS_HEADER, [])
    appended_rows = rendered[len(header) :]
    separator = b"" if source_bytes.endswith((b"\n", b"\r")) else b"\n"
    return source_bytes + separator + appended_rows


def _paths_alias(left: Path, right: Path) -> bool:
    left = Path(left)
    right = Path(right)
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return left.resolve() == right.resolve()


def _validate_output_paths(
    output_paths: Sequence[Path], input_paths: Sequence[Path]
) -> None:
    for index, output in enumerate(output_paths):
        if any(_paths_alias(output, other) for other in output_paths[:index]):
            raise MetadataIntegrationError(
                f"output paths must be distinct; duplicate {output}"
            )
        for input_path in input_paths:
            if _paths_alias(output, input_path):
                raise MetadataIntegrationError(
                    f"output {output} must differ from and not alias input "
                    f"{input_path}"
                )
        if not output.parent.is_dir():
            raise MetadataIntegrationError(
                f"{output.parent}: output directory is missing"
            )


def _stage_bytes(
    destination: Path,
    payload: bytes,
    *,
    backup: bool = False,
    restore: bool = False,
) -> Path:
    suffix = ".bak.tmp" if backup else ".restore.tmp" if restore else ".tmp"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=suffix,
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            os.chmod(
                temporary_path,
                stat.S_IMODE(destination.stat().st_mode),
            )
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _cleanup_temporaries(paths: dict[Path, Path]) -> list[str]:
    errors = []
    for destination, temporary_path in list(paths.items()):
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(
                f"{temporary_path} for {destination}: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            paths.pop(destination)
    return errors


def _restore_output(destination: Path, backup_path: Path) -> None:
    restore_path = _stage_bytes(
        destination,
        backup_path.read_bytes(),
        restore=True,
    )
    try:
        restore_path.replace(destination)
    finally:
        restore_path.unlink(missing_ok=True)


def _recovery_paths(backups: dict[Path, Path]) -> str:
    return ", ".join(
        f"{destination}={backup_path}"
        for destination, backup_path in sorted(
            backups.items(), key=lambda item: item[0].as_posix()
        )
    )


def write_integration_outputs(
    result: MetadataIntegrationResult,
    *,
    candidates_path: Path,
    conflicts_path: Path,
    bibliography_path: Path,
    bibtex_path: Path,
    citation_keys_path: Path | None = None,
    input_paths: Sequence[Path] | None = None,
) -> None:
    """Stage and replace all integration outputs with best-effort rollback."""
    destinations = [
        Path(candidates_path),
        Path(conflicts_path),
        Path(bibliography_path),
        Path(bibtex_path),
    ]
    if citation_keys_path is not None:
        destinations.append(Path(citation_keys_path))
    if not result.input_paths:
        raise MetadataIntegrationError(
            "writer input paths are required for mandatory alias protection"
        )
    if not result.input_fingerprints:
        raise MetadataIntegrationError(
            "writer input fingerprints are required"
        )
    _assert_input_fingerprints_current(result.input_fingerprints)
    protected_inputs = list(result.input_paths)
    if input_paths is not None:
        protected_inputs.extend(Path(path).resolve() for path in input_paths)
    _validate_output_paths(destinations, protected_inputs)
    payloads = [
        _csv_bytes(CANDIDATE_HEADER, result.candidates),
        _csv_bytes(CONFLICT_HEADER, result.conflicts),
        _csv_bytes(BIBLIOGRAPHY_HEADER, result.bibliography),
        result.bibtex.encode("utf-8"),
    ]

    if citation_keys_path is not None:
        payloads.append(result.citation_keys_bytes)
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    existed = {destination: destination.exists() for destination in destinations}
    attempted: list[Path] = []
    try:
        for destination, payload in zip(destinations, payloads, strict=True):
            staged[destination] = _stage_bytes(destination, payload)
        for destination in destinations:
            if existed[destination]:
                backups[destination] = _stage_bytes(
                    destination, destination.read_bytes(), backup=True
                )
        _assert_input_fingerprints_current(result.input_fingerprints)
        for destination in destinations:
            attempted.append(destination)
            staged[destination].replace(destination)
            staged.pop(destination)
    except BaseException as write_error:
        rollback_errors = []
        for destination in reversed(attempted):
            try:
                backup_path = backups.get(destination)
                if backup_path is not None:
                    _restore_output(destination, backup_path)
                elif not existed[destination]:
                    destination.unlink(missing_ok=True)
            except BaseException as exc:
                rollback_errors.append(
                    f"{destination}: {type(exc).__name__}: {exc}"
                )

        staging_cleanup_errors = _cleanup_temporaries(staged)
        if rollback_errors:
            details = "; ".join(rollback_errors + staging_cleanup_errors)
            raise MetadataIntegrationError(
                "write failed and rollback incomplete; "
                f"recovery backup paths: {_recovery_paths(backups)}; "
                f"rollback errors: {details}"
            ) from write_error

        backup_cleanup_errors = _cleanup_temporaries(backups)
        cleanup_errors = staging_cleanup_errors + backup_cleanup_errors
        if cleanup_errors:
            raise MetadataIntegrationError(
                "write failed and temporary cleanup was incomplete; "
                f"retained recovery backup paths: {_recovery_paths(backups)}; "
                f"cleanup errors: {'; '.join(cleanup_errors)}"
            ) from write_error
        raise

    cleanup_errors = _cleanup_temporaries(staged)
    cleanup_errors.extend(_cleanup_temporaries(backups))
    if cleanup_errors:
        raise MetadataIntegrationError(
            "outputs were replaced but temporary cleanup was incomplete; "
            f"retained recovery backup paths: {_recovery_paths(backups)}; "
            f"cleanup errors: {'; '.join(cleanup_errors)}"
        )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrate validated metadata verification results."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--conflicts", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--citation-keys", required=True, type=Path)
    parser.add_argument(
        "--extend-citation-keys", action="store_true"
    )
    parser.add_argument(
        "--metadata-result", required=True, action="append", type=Path
    )
    parser.add_argument(
        "--conflict-result", required=True, action="append", type=Path
    )
    parser.add_argument("--output-candidates", required=True, type=Path)
    parser.add_argument("--output-conflicts", required=True, type=Path)
    parser.add_argument("--output-bibliography", required=True, type=Path)
    parser.add_argument("--output-bibtex", required=True, type=Path)
    parser.add_argument("--output-citation-keys", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    if (
        arguments.extend_citation_keys
        and arguments.output_citation_keys is None
    ):
        parser.error(
            "--output-citation-keys is required with --extend-citation-keys"
        )
    if (
        not arguments.extend_citation_keys
        and arguments.output_citation_keys is not None
    ):
        parser.error(
            "--output-citation-keys is forbidden without --extend-citation-keys"
        )
    input_paths = [
        arguments.candidates,
        arguments.conflicts,
        arguments.manifest,
        arguments.citation_keys,
        *arguments.metadata_result,
        *arguments.conflict_result,
    ]
    output_paths = [
        arguments.output_candidates,
        arguments.output_conflicts,
        arguments.output_bibliography,
        arguments.output_bibtex,
    ]
    if arguments.output_citation_keys is not None:
        output_paths.append(arguments.output_citation_keys)
    _validate_output_paths(output_paths, input_paths)
    result = integrate_metadata(
        arguments.candidates,
        arguments.conflicts,
        arguments.manifest,
        arguments.metadata_result,
        arguments.conflict_result,
        citation_keys_path=arguments.citation_keys,
        extend_citation_keys=arguments.extend_citation_keys,
    )
    write_integration_outputs(
        result,
        candidates_path=arguments.output_candidates,
        conflicts_path=arguments.output_conflicts,
        bibliography_path=arguments.output_bibliography,
        bibtex_path=arguments.output_bibtex,
        citation_keys_path=arguments.output_citation_keys,
        input_paths=input_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
