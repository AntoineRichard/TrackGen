from __future__ import annotations

import argparse
import hashlib
import csv
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlsplit

if __package__:
    from .validate_corpus import HEADERS, normalize_doi, split_values
else:
    from validate_corpus import HEADERS, normalize_doi, split_values


class MergeError(ValueError):
    pass


BIBLIOGRAPHIC_FIELDS = (
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_type",
)
PROVENANCE_FIELDS = (
    "discovery_stream",
    "discovery_query",
    "discovery_agent",
)
IDENTITY_ORDER = {"doi": 0, "title": 1, "arxiv": 2}
CONFLICT_FIELD_ORDER = {
    name: index
    for index, name in enumerate(
        (*BIBLIOGRAPHIC_FIELDS, "screening_status", "exclusion_reason")
    )
}
SCREENING_STATUS_ORDER = {
    "candidate": 0,
    "included": 1,
    "boundary": 2,
    "excluded": 3,
}
STABLE_CANDIDATE_PATTERN = re.compile(r"C([0-9]{4,})")
ARXIV_ID_PATTERN = re.compile(
    r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z0-9.-]+/[0-9]{7})",
    re.IGNORECASE,
)
ARXIV_VERSION_PATTERN = re.compile(r"v[0-9]+$", re.IGNORECASE)
CANDIDATE_HEADER = HEADERS["candidates.csv"]
CONFLICT_HEADER = HEADERS["conflicts.csv"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALIAS_HEADER = (
    "retired_candidate_id",
    "surviving_candidate_id",
    "reason",
    "evidence",
)
INCOMING_SORT_FIELDS = (
    *BIBLIOGRAPHIC_FIELDS,
    *PROVENANCE_FIELDS,
    "screening_status",
    "exclusion_reason",
    "metadata_evidence",
)

CandidateRow = dict[str, str]
IdentityKey = tuple[str, str]
ConflictSignature = tuple[str, str, str, str]


@dataclass(frozen=True)
class IncomingRecord:
    row: CandidateRow
    source_file: str
    local_id: str

    @property
    def source(self) -> str:
        return f"{self.source_file}#{self.local_id or '<missing>'}"


@dataclass(frozen=True)
class CandidateAlias:
    retired_candidate_id: str
    surviving_candidate_id: str
    reason: str
    evidence: str


@dataclass
class PendingConflict:
    candidate_id: str
    field_name: str
    current: str
    proposed: str
    value_a_sources: set[str] = field(default_factory=set)
    value_b_sources: set[str] = field(default_factory=set)


@dataclass
class MergeStats:
    existing_count: int
    incoming_total: int = 0
    new_count: int = 0
    duplicate_count: int = 0
    source_files: set[str] = field(default_factory=set)
    source_streams: set[str] = field(default_factory=set)
    incoming_by_file: Counter[str] = field(default_factory=Counter)
    new_by_file: Counter[str] = field(default_factory=Counter)
    duplicate_by_file: Counter[str] = field(default_factory=Counter)
    incoming_by_stream: Counter[str] = field(default_factory=Counter)
    new_by_stream: Counter[str] = field(default_factory=Counter)
    duplicate_by_stream: Counter[str] = field(default_factory=Counter)
    identity_matches: Counter[str] = field(default_factory=Counter)
    duplicate_matches: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    retirement_aliases: dict[str, str] = field(default_factory=dict)


def _is_absent(value: str) -> bool:
    return not value.strip() or value.strip().casefold() == "nr"


def normalize_title(value: str) -> str:
    if _is_absent(value):
        return ""
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    characters = []
    for character in decomposed:
        category = unicodedata.category(character)
        if category.startswith("M"):
            continue
        characters.append(character if character.isalnum() else " ")
    return " ".join("".join(characters).split())


def _normalized_doi(value: str) -> str:
    if _is_absent(value):
        return ""
    normalized = normalize_doi(value)
    return "" if _is_absent(normalized) else normalized


def _stable_arxiv_id(value: str) -> str:
    candidate = value.strip().strip("/")
    if candidate.casefold().startswith("arxiv:"):
        candidate = candidate[6:]
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    candidate = ARXIV_VERSION_PATTERN.sub("", candidate)
    candidate = candidate.casefold()
    return candidate if ARXIV_ID_PATTERN.fullmatch(candidate) else ""


def _arxiv_id_from_doi(value: str) -> str:
    doi = _normalized_doi(value)
    prefix = "10.48550/arxiv."
    if not doi.startswith(prefix):
        return ""
    return _stable_arxiv_id(doi[len(prefix) :])


def _arxiv_id_from_url(value: str) -> str:
    if _is_absent(value):
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    hostname = (parsed.hostname or "").casefold()
    if hostname != "arxiv.org" and not hostname.endswith(".arxiv.org"):
        return ""
    path_parts = unquote(parsed.path).strip("/").split("/")
    if len(path_parts) < 2 or path_parts[0].casefold() not in {"abs", "pdf"}:
        return ""
    return _stable_arxiv_id("/".join(path_parts[1:]))


def _arxiv_id(row: CandidateRow) -> str:
    identities = {
        value for value in (_arxiv_id_from_doi(row["doi"]),) if value
    }
    identities.update(
        identity
        for url in split_values(row["url"])
        if (identity := _arxiv_id_from_url(url))
    )
    if len(identities) > 1:
        raise MergeError(
            f"{row['title']!r} contains conflicting arXiv identities "
            f"{sorted(identities)}"
        )
    return next(iter(identities), "")


def _identity_keys(row: CandidateRow) -> tuple[IdentityKey, ...]:
    keys = []
    doi = _normalized_doi(row["doi"])
    if doi:
        keys.append(("doi", doi))
    title = normalize_title(row["title"])
    if title:
        keys.append(("title", title))
    arxiv_id = _arxiv_id(row)
    if arxiv_id:
        keys.append(("arxiv", arxiv_id))
    return tuple(keys)


def identity_key(row: CandidateRow) -> IdentityKey:
    keys = _identity_keys(row)
    return keys[0] if keys else ("title", "")


def _sanitize_row(row: dict[str | None, str | None]) -> CandidateRow:
    sanitized = {
        name: (row.get(name) or "").strip() for name in CANDIDATE_HEADER
    }
    for name in BIBLIOGRAPHIC_FIELDS:
        if _is_absent(sanitized[name]):
            sanitized[name] = ""
    for name in ("exclusion_reason", "metadata_evidence"):
        if _is_absent(sanitized[name]):
            sanitized[name] = ""
    return sanitized


def _read_candidate_rows(path: Path, *, existing: bool) -> list[CandidateRow]:
    reader: csv.DictReader | None = None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual = tuple(reader.fieldnames or ())
            missing = sorted(set(CANDIDATE_HEADER) - set(actual))
            if missing:
                raise MergeError(
                    f"{path}: missing candidate columns {missing}"
                )
            duplicates = sorted(
                name for name in CANDIDATE_HEADER if actual.count(name) > 1
            )
            if duplicates:
                raise MergeError(
                    f"{path}: duplicate candidate columns {duplicates}"
                )

            rows = []
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(
                    value is None for value in raw_row.values()
                ):
                    raise MergeError(f"{path}:{row_number}: malformed CSV row")
                row = _sanitize_row(raw_row)
                if not any(row.values()):
                    raise MergeError(
                        f"{path}:{row_number}: row is entirely blank"
                    )
                if not row["title"]:
                    raise MergeError(
                        f"{path}:{row_number}: title is required"
                    )
                if existing:
                    if not STABLE_CANDIDATE_PATTERN.fullmatch(
                        row["candidate_id"]
                    ):
                        raise MergeError(
                            f"{path}:{row_number}: existing candidate_id "
                            f"{row['candidate_id']!r} is not stable"
                        )
                elif row["screening_status"] not in {"candidate", "excluded"}:
                    raise MergeError(
                        f"{path}:{row_number}: incoming screening_status must "
                        "be candidate or excluded"
                    )
                if (
                    row["screening_status"] == "excluded"
                    and not row["exclusion_reason"]
                ):
                    raise MergeError(
                        f"{path}:{row_number}: excluded candidate requires a "
                        "specific exclusion_reason"
                    )
                rows.append(row)
            return rows
    except UnicodeDecodeError as exc:
        raise MergeError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        line_number = reader.line_num if reader is not None else 1
        raise MergeError(
            f"{path}:{line_number}: CSV parse error: {exc}"
        ) from exc


def _value_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value

def _candidate_alias_path(
    existing_path: Path, aliases_path: Path | None
) -> Path | None:
    if aliases_path is not None:
        path = Path(aliases_path)
        if not path.exists():
            raise MergeError(f"{path}: candidate alias file does not exist")
        return path
    sibling = existing_path.parent / "candidate_aliases.csv"
    return sibling if sibling.exists() else None


def _read_candidate_aliases(
    path: Path | None, existing_ids: set[str]
) -> dict[str, CandidateAlias]:
    if path is None:
        return {}

    reader: csv.DictReader | None = None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual = tuple(reader.fieldnames or ())
            if actual != ALIAS_HEADER:
                raise MergeError(
                    f"{path}: alias columns must be exactly "
                    f"{list(ALIAS_HEADER)}; found {list(actual)}"
                )

            aliases: dict[str, CandidateAlias] = {}
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(
                    value is None for value in raw_row.values()
                ):
                    raise MergeError(f"{path}:{row_number}: malformed CSV row")
                values = {
                    name: (raw_row.get(name) or "").strip()
                    for name in ALIAS_HEADER
                }
                retired = values["retired_candidate_id"]
                survivor = values["surviving_candidate_id"]
                for field_name, candidate_id in (
                    ("retired_candidate_id", retired),
                    ("surviving_candidate_id", survivor),
                ):
                    if not STABLE_CANDIDATE_PATTERN.fullmatch(candidate_id):
                        raise MergeError(
                            f"{path}:{row_number}: {field_name} "
                            f"{candidate_id!r} is not a stable candidate ID"
                        )
                if retired == survivor:
                    raise MergeError(
                        f"{path}:{row_number}: alias cannot retire {retired} "
                        "to itself"
                    )
                if retired in aliases:
                    raise MergeError(
                        f"{path}:{row_number}: duplicate retired candidate "
                        f"{retired}"
                    )
                if not values["reason"] or not values["evidence"]:
                    raise MergeError(
                        f"{path}:{row_number}: alias reason and evidence "
                        "must be nonempty"
                    )
                aliases[retired] = CandidateAlias(
                    retired_candidate_id=retired,
                    surviving_candidate_id=survivor,
                    reason=values["reason"],
                    evidence=values["evidence"],
                )
    except UnicodeDecodeError as exc:
        raise MergeError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        line_number = reader.line_num if reader is not None else 1
        raise MergeError(
            f"{path}:{line_number}: CSV parse error: {exc}"
        ) from exc

    retired_ids = set(aliases)
    chained = sorted(
        alias.surviving_candidate_id
        for alias in aliases.values()
        if alias.surviving_candidate_id in retired_ids
    )
    if chained:
        raise MergeError(
            f"{path}: aliases must be direct and acyclic; chained IDs "
            f"{chained}"
        )
    missing_survivors = sorted(
        {
            alias.surviving_candidate_id
            for alias in aliases.values()
            if alias.surviving_candidate_id not in existing_ids
        },
        key=_candidate_id_sort_key,
    )
    if missing_survivors:
        raise MergeError(
            f"{path}: alias survivors are absent from candidates: "
            f"{missing_survivors}"
        )
    return aliases


def _canonical_union(*values: str) -> str:
    items = {
        item
        for value in values
        for item in split_values(value)
        if not _is_absent(item)
    }
    return "; ".join(sorted(items, key=_value_sort_key))


def _append_unique_values(existing: str, incoming: str) -> str:
    values = [
        item for item in split_values(existing) if not _is_absent(item)
    ]
    seen = set(values)
    additions = {
        item
        for item in split_values(incoming)
        if not _is_absent(item) and item not in seen
    }
    values.extend(sorted(additions, key=_value_sort_key))
    return "; ".join(values)


def _incoming_sort_key(incoming: IncomingRecord) -> tuple[object, ...]:
    row = incoming.row
    identities = tuple(
        (IDENTITY_ORDER[kind], value) for kind, value in _identity_keys(row)
    )
    values = tuple(
        _value_sort_key(row[name]) for name in INCOMING_SORT_FIELDS
    )
    return (
        identities,
        values,
        _value_sort_key(incoming.source_file),
        _value_sort_key(incoming.local_id),
    )


def _candidate_id_sort_key(candidate_id: str) -> tuple[int, int, str]:
    match = STABLE_CANDIDATE_PATTERN.fullmatch(candidate_id)
    return (0, int(match.group(1)), candidate_id) if match else (1, 0, candidate_id)


def _candidate_sort_key(row: CandidateRow) -> tuple[int, int, str]:
    return _candidate_id_sort_key(row["candidate_id"])


def _next_candidate_number(
    rows: list[CandidateRow],
    reserved_ids: Sequence[str] = (),
) -> int:
    numbers = [
        int(match.group(1))
        for row in rows
        if (match := STABLE_CANDIDATE_PATTERN.fullmatch(row["candidate_id"]))
    ]
    numbers.extend(
        int(match.group(1))
        for candidate_id in reserved_ids
        if (match := STABLE_CANDIDATE_PATTERN.fullmatch(candidate_id))
    )
    return max(numbers, default=0) + 1


def _add_to_lookup(
    lookup: dict[IdentityKey, set[int]], row: CandidateRow, index: int
) -> None:
    for key in _identity_keys(row):
        lookup.setdefault(key, set()).add(index)


def _find_match(
    incoming: CandidateRow,
    merged: list[CandidateRow],
    lookup: dict[IdentityKey, set[int]],
) -> tuple[int | None, str | None]:
    keys = _identity_keys(incoming)
    matches = {
        index for key in keys for index in lookup.get(key, set())
    }
    if len(matches) > 1:
        candidate_ids = sorted(
            (merged[index]["candidate_id"] for index in matches),
            key=_candidate_id_sort_key,
        )
        raise MergeError(
            f"{incoming['title']!r} bridges multiple existing identities: "
            + ", ".join(candidate_ids)
        )
    if not matches:
        return None, None
    index = next(iter(matches))
    match_type = next(
        kind
        for kind, value in keys
        if index in lookup.get((kind, value), set())
    )
    return index, match_type


def _equivalent(field_name: str, left: str, right: str) -> bool:
    if field_name == "doi":
        return _normalized_doi(left) == _normalized_doi(right)
    if field_name == "title":
        return normalize_title(left) == normalize_title(right)
    return left == right


def _conflict_value(field_name: str, value: str) -> str:
    if field_name == "doi":
        return _normalized_doi(value)
    if field_name == "title":
        return normalize_title(value)
    return value


def _conflict_pair_sort_key(
    field_name: str, value: str
) -> tuple[object, ...]:
    normalized = _conflict_value(field_name, value)
    if field_name == "screening_status":
        return (
            SCREENING_STATUS_ORDER.get(value, len(SCREENING_STATUS_ORDER)),
            *_value_sort_key(normalized),
        )
    return (*_value_sort_key(normalized), *_value_sort_key(value))


def _canonical_conflict_values(
    field_name: str, left: str, right: str
) -> tuple[str, str]:
    if _conflict_pair_sort_key(field_name, right) < _conflict_pair_sort_key(
        field_name, left
    ):
        return right, left
    return left, right


def _record_conflict(
    conflicts: dict[ConflictSignature, PendingConflict],
    candidate_id: str,
    field_name: str,
    current: str,
    proposed: str,
    current_sources: set[str],
    proposed_source: str,
) -> None:
    value_a_sources = set(current_sources)
    value_b_sources = {proposed_source}
    value_a, value_b = _canonical_conflict_values(
        field_name, current, proposed
    )
    if (value_a, value_b) != (current, proposed):
        current, proposed = proposed, current
        value_a_sources, value_b_sources = value_b_sources, value_a_sources
    signature = (
        candidate_id,
        field_name,
        _conflict_value(field_name, current),
        _conflict_value(field_name, proposed),
    )
    pending = conflicts.setdefault(
        signature,
        PendingConflict(
            candidate_id=candidate_id,
            field_name=field_name,
            current=current,
            proposed=proposed,
        ),
    )
    pending.value_a_sources.update(value_a_sources)
    pending.value_b_sources.update(value_b_sources)


def _build_conflict_rows(
    conflicts: dict[ConflictSignature, PendingConflict],
) -> list[CandidateRow]:
    values = sorted(
        conflicts.values(),
        key=lambda item: (
            _candidate_id_sort_key(item.candidate_id),
            CONFLICT_FIELD_ORDER[item.field_name],
            _value_sort_key(
                _conflict_value(item.field_name, item.proposed)
            ),
            _value_sort_key(item.proposed),
        ),
    )
    rows = []
    for pending in values:
        row = dict.fromkeys(CONFLICT_HEADER, "")
        source_context = [
            f"value_a={source}"
            for source in sorted(
                pending.value_a_sources, key=_value_sort_key
            )
        ]
        source_context.extend(
            f"value_b={source}"
            for source in sorted(
                pending.value_b_sources, key=_value_sort_key
            )
        )
        signature_text = "\0".join(
            (
                "candidate",
                pending.candidate_id,
                pending.field_name,
                _conflict_value(pending.field_name, pending.current),
                _conflict_value(pending.field_name, pending.proposed),
            )
        )
        conflict_id = "X" + hashlib.sha256(
            signature_text.encode("utf-8")
        ).hexdigest()[:12].upper()

        row.update(
            conflict_id=conflict_id,
            record_type="candidate",
            record_key=pending.candidate_id,
            field=pending.field_name,
            value_a=pending.current,
            value_b=pending.proposed,
            resolution_evidence="; ".join(source_context),
        )
        rows.append(row)
    return rows


def _new_record(incoming: CandidateRow, candidate_id: str) -> CandidateRow:
    record = dict(incoming)
    record["candidate_id"] = candidate_id
    record["cite_key"] = ""
    record["metadata_status"] = "unverified"
    for name in PROVENANCE_FIELDS:
        record[name] = _canonical_union(record[name])
    record["metadata_evidence"] = _append_unique_values(
        "", record["metadata_evidence"]
    )
    return record


@dataclass(frozen=True)
class ComponentNode:
    row: CandidateRow
    source_file: str
    source_label: str
    local_id: str
    row_number: int
    existing: bool

    @property
    def source(self) -> str:
        return f"{self.source_label}#{self.local_id or '<missing>'}"


def _portable_path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        parts = resolved.parts
        for index in range(len(parts) - 1):
            if parts[index : index + 2] == ("paper", "data"):
                return Path(*parts[index:]).as_posix()
    return resolved.as_posix()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


PAPER_SOURCE_PATTERN = re.compile(r"paper|article|preprint|survey")
ARTIFACT_SOURCE_PATTERN = re.compile(
    r"software|repository|package|documentation|simulator|benchmark|"
    r"platform|standard|system|competition|artifact"
)
SEED_PATTERN = re.compile(r"(?<![A-Za-z0-9])seed::(C[0-9]{4,})(?![0-9])")
URL_PATTERN = re.compile(r"https?://[^\s;]+", re.IGNORECASE)
EXTENDED_IDENTITY_ORDER = {
    **IDENTITY_ORDER,
    "artifact": len(IDENTITY_ORDER),
    "retirement": len(IDENTITY_ORDER) + 1,
}


def _strict_nonpaper_artifact(row: CandidateRow) -> bool:
    source_type = row["source_type"].casefold()
    return (
        not PAPER_SOURCE_PATTERN.search(source_type)
        and bool(ARTIFACT_SOURCE_PATTERN.search(source_type))
    )


def _urls_in_cell(value: str) -> tuple[str, ...]:
    urls = []
    for item in split_values(value):
        match = URL_PATTERN.search(item)
        if match:
            urls.append(match.group(0).rstrip(".,)]"))
    return tuple(urls)


def _normalize_artifact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    host = (parsed.hostname or "").casefold()
    parts = [unquote(part) for part in parsed.path.split("/") if part]

    if host == "github.com":
        if len(parts) < 2:
            return ""
        owner = parts[0].casefold()
        repository = parts[1].removesuffix(".git").casefold()
        return f"github.com/{owner}/{repository}" if repository else ""


    if host == "gitlab.com":
        if len(parts) < 2:
            return ""
        for marker in ("-", "blob", "tree", "raw"):
            if marker in parts[1:]:
                parts = parts[: parts.index(marker)]
                break
        if len(parts) < 2:
            return ""
        parts[-1] = parts[-1].removesuffix(".git")
        return "gitlab.com/" + "/".join(part.casefold() for part in parts)

    if host == "pypi.org":
        if len(parts) < 2 or parts[0].casefold() != "project":
            return ""
        project = re.sub(r"[-_.]+", "-", parts[1]).casefold()
        return f"pypi.org/project/{project}" if project else ""

    return ""


def _artifact_identity_keys(row: CandidateRow) -> tuple[IdentityKey, ...]:
    if not _strict_nonpaper_artifact(row):
        return ()
    primary_urls = _urls_in_cell(row["url"])
    primary_aliases = {
        normalized
        for url in primary_urls
        if (normalized := _normalize_artifact_url(url))
    }
    urls = list(primary_urls)
    if not primary_urls or primary_aliases:
        urls.extend(_urls_in_cell(row["metadata_evidence"]))
    aliases = {
        normalized
        for url in urls
        if (normalized := _normalize_artifact_url(url))
    }
    return tuple(("artifact", alias) for alias in sorted(aliases))


def _retirement_identity_tokens(value: str) -> tuple[str, ...]:
    tokens: set[str] = set()
    for item in split_values(value):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = urlsplit(item)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.scheme.casefold() in {"http", "https"}:
            host = (parsed.hostname or "").casefold()
            path = unquote(parsed.path).rstrip("/")
            if host and path:
                tokens.add(f"url:{host}{path}")
            if host in {"doi.org", "dx.doi.org"}:
                doi = _normalized_doi(item)
                if doi:
                    tokens.add(f"doi:{doi}")
            arxiv_id = _arxiv_id_from_url(item)
            if arxiv_id:
                tokens.add(f"arxiv:{arxiv_id}")
            artifact = _normalize_artifact_url(item)
            if artifact:
                tokens.add(f"artifact:{artifact}")
            continue

        normalized_doi = _normalized_doi(item)
        if re.fullmatch(r"10\.[0-9]{4,9}/\S+", normalized_doi):
            tokens.add(f"doi:{normalized_doi}")
        arxiv_id = _arxiv_id_from_doi(item) or _stable_arxiv_id(item)
        if arxiv_id:
            tokens.add(f"arxiv:{arxiv_id}")
    return tuple(sorted(tokens))


def _row_retirement_identity_tokens(row: CandidateRow) -> tuple[str, ...]:
    tokens = _retirement_identity_tokens(
        _canonical_union(row["doi"], row["url"])
    )
    if not PAPER_SOURCE_PATTERN.search(row["source_type"].casefold()):
        return tokens
    repository_url_prefixes = (
        "url:github.com/",
        "url:gitlab.com/",
        "url:pypi.org/",
    )
    return tuple(
        token
        for token in tokens
        if not token.startswith("artifact:")
        and not token.startswith(repository_url_prefixes)
    )


def _component_identity_keys(row: CandidateRow) -> tuple[IdentityKey, ...]:
    return (*_identity_keys(row), *_artifact_identity_keys(row))


def _seed_targets(row: CandidateRow) -> tuple[str, ...]:
    return tuple(sorted(set(SEED_PATTERN.findall(row["discovery_query"]))))


def _component_node_sort_key(node: ComponentNode) -> tuple[object, ...]:
    return _incoming_sort_key(
        IncomingRecord(node.row, node.source_file, node.local_id)
    )


def _component_sort_key(nodes: list[ComponentNode]) -> tuple[object, ...]:
    aliases = sorted(
        {
            (EXTENDED_IDENTITY_ORDER[kind], value)
            for node in nodes
            for kind, value in _component_identity_keys(node.row)
        }
    )
    row_values = sorted(
        tuple(_value_sort_key(node.row[name]) for name in INCOMING_SORT_FIELDS)
        for node in nodes
    )
    return tuple(aliases), tuple(row_values)


def _connection_type(
    node: ComponentNode, component: list[ComponentNode]
) -> str:
    node_keys = set(_component_identity_keys(node.row))
    other_keys = {
        key
        for other in component
        if other is not node
        for key in _component_identity_keys(other.row)
    }
    shared = sorted(
        node_keys & other_keys,
        key=lambda item: (EXTENDED_IDENTITY_ORDER[item[0]], item[1]),
    )
    if shared:
        return shared[0][0]
    return "unknown"


def _merge_component_record(
    base: CandidateRow,
    base_source: str,
    preserve_reviewed_status: bool,
    incoming_nodes: list[ComponentNode],
    conflicts: dict[ConflictSignature, PendingConflict],
) -> CandidateRow:
    record = dict(base)
    origins = {
        name: ({base_source} if record[name] else set())
        for name in BIBLIOGRAPHIC_FIELDS
    }
    if not incoming_nodes:
        return record

    for name in PROVENANCE_FIELDS:
        record[name] = _canonical_union(
            record[name], *(node.row[name] for node in incoming_nodes)
        )
    for node in incoming_nodes:
        record["metadata_evidence"] = _append_unique_values(
            record["metadata_evidence"], node.row["metadata_evidence"]
        )

    if preserve_reviewed_status:
        for node in incoming_nodes:
            proposed_status = node.row["screening_status"]
            if proposed_status != record["screening_status"]:
                _record_conflict(
                    conflicts,
                    record["candidate_id"],
                    "screening_status",
                    record["screening_status"],
                    proposed_status,
                    {base_source},
                    node.source,
                )
            proposed_reason = node.row["exclusion_reason"]
            if (
                proposed_status == "excluded"
                and proposed_reason != record["exclusion_reason"]
            ):
                _record_conflict(
                    conflicts,
                    record["candidate_id"],
                    "exclusion_reason",
                    record["exclusion_reason"] or "<empty>",
                    proposed_reason or "<empty>",
                    {base_source},
                    node.source,
                )
    else:
        observations = [(record, base_source)] + [
            (node.row, node.source) for node in incoming_nodes
        ]
        candidate_sources = {
            source
            for row, source in observations
            if row["screening_status"] == "candidate"
        }
        excluded_observations = [
            (row, source)
            for row, source in observations
            if row["screening_status"] == "excluded"
        ]
        reason_sources: dict[str, set[str]] = {}
        for row, source in excluded_observations:
            reason_sources.setdefault(row["exclusion_reason"], set()).add(
                source
            )
        exclusion_reasons = sorted(reason_sources, key=_value_sort_key)

        if candidate_sources and excluded_observations:
            for _, source in excluded_observations:
                _record_conflict(
                    conflicts,
                    record["candidate_id"],
                    "screening_status",
                    "candidate",
                    "excluded",
                    candidate_sources,
                    source,
                )

        if excluded_observations and not candidate_sources:
            selected_reason = exclusion_reasons[0]
            record["screening_status"] = "excluded"
            record["exclusion_reason"] = selected_reason
            for alternative in exclusion_reasons[1:]:
                for source in reason_sources[alternative]:
                    _record_conflict(
                        conflicts,
                        record["candidate_id"],
                        "exclusion_reason",
                        selected_reason,
                        alternative,
                        reason_sources[selected_reason],
                        source,
                    )
        elif excluded_observations:
            if len(exclusion_reasons) > 1:
                record["screening_status"] = "candidate"
                record["exclusion_reason"] = ""
                for reason in exclusion_reasons:
                    for source in reason_sources[reason]:
                        _record_conflict(
                            conflicts,
                            record["candidate_id"],
                            "exclusion_reason",
                            "<empty>",
                            reason,
                            candidate_sources,
                            source,
                        )
            else:
                record["screening_status"] = "excluded"
                record["exclusion_reason"] = exclusion_reasons[0]

    for node in incoming_nodes:
        for name in BIBLIOGRAPHIC_FIELDS:
            current = record[name]
            proposed = node.row[name]
            if not proposed:
                continue
            if not current:
                record[name] = proposed
                origins[name] = {node.source}
                continue
            if _equivalent(name, current, proposed):
                origins[name].add(node.source)
                continue
            _record_conflict(
                conflicts,
                record["candidate_id"],
                name,
                current,
                proposed,
                origins[name],
                node.source,
            )
            record["metadata_status"] = "conflict"
    return record


def _stable_component_survivor(
    stable_ids: list[str],
    aliases: dict[str, CandidateAlias],
) -> str | None:
    if not stable_ids:
        return None
    if len(stable_ids) == 1:
        return stable_ids[0]

    survivors = [
        candidate_id
        for candidate_id in stable_ids
        if candidate_id not in aliases
    ]
    if len(survivors) == 1:
        survivor = survivors[0]
        if all(
            candidate_id == survivor
            or aliases.get(candidate_id) is not None
            and aliases[candidate_id].surviving_candidate_id == survivor
            for candidate_id in stable_ids
        ):
            return survivor

    raise MergeError(
        "identity component bridges multiple existing identities without "
        "complete retirement aliases (baseline collision): "
        + ", ".join(stable_ids)
    )


def _merge_candidate_files(
    existing_path: Path,
    agent_paths: Sequence[Path],
    aliases_path: Path | None = None,
) -> tuple[list[CandidateRow], list[CandidateRow], MergeStats]:
    existing = _read_candidate_rows(existing_path, existing=True)
    candidate_ids = [row["candidate_id"] for row in existing]
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise MergeError(
            f"{existing_path}: duplicate candidate IDs {duplicate_ids}"
        )
    aliases = _read_candidate_aliases(
        _candidate_alias_path(existing_path, aliases_path), set(candidate_ids)
    )

    stats = MergeStats(
        existing_count=len(existing),
        retirement_aliases={
            retired_id: alias.surviving_candidate_id
            for retired_id, alias in aliases.items()
        },
    )
    nodes = [
        ComponentNode(
            row=row,
            source_file=existing_path.name,
            source_label=_portable_path_label(existing_path),
            local_id=row["candidate_id"],
            row_number=row_number,
            existing=True,
        )
        for row_number, row in enumerate(existing, start=2)
    ]
    existing_indexes = {
        node.local_id: index for index, node in enumerate(nodes)
    }

    incoming_nodes: list[ComponentNode] = []
    for path in sorted((Path(path) for path in agent_paths), key=str):
        source_file = path.name
        rows = _read_candidate_rows(path, existing=False)
        stats.source_files.add(source_file)
        stats.incoming_total += len(rows)
        stats.incoming_by_file[source_file] += len(rows)
        for row_number, row in enumerate(rows, start=2):
            streams = split_values(row["discovery_stream"]) or ["<missing>"]
            for stream in streams:
                stats.source_streams.add(stream)
                stats.incoming_by_stream[stream] += 1
            incoming_nodes.append(
                ComponentNode(
                    row=row,
                    source_file=source_file,
                    source_label=_portable_path_label(path),
                    local_id=row["candidate_id"],
                    row_number=row_number,
                    existing=False,
                )
            )
    incoming_nodes.sort(key=_component_node_sort_key)
    nodes.extend(incoming_nodes)

    union_find = UnionFind(len(nodes))
    alias_indexes: dict[IdentityKey, int] = {}
    for index, node in enumerate(nodes):
        for alias in _component_identity_keys(node.row):
            previous = alias_indexes.setdefault(alias, index)
            union_find.union(previous, index)

    retirement_indexes: dict[str, set[int]] = {}
    for index, node in enumerate(nodes):
        for token in _row_retirement_identity_tokens(node.row):
            retirement_indexes.setdefault(token, set()).add(index)

    retirement_matches: set[int] = set()
    for retired_id, alias in aliases.items():
        survivor_index = existing_indexes[alias.surviving_candidate_id]
        for token in _retirement_identity_tokens(alias.evidence):
            for matching_index in retirement_indexes.get(token, set()):
                union_find.union(survivor_index, matching_index)
                retirement_matches.add(id(nodes[matching_index]))
        retired_index = existing_indexes.get(retired_id)
        if retired_index is not None:
            union_find.union(retired_index, survivor_index)

    for node in nodes:
        if node.existing:
            continue
        for target in _seed_targets(node.row):
            if target not in existing_indexes:
                raise MergeError(
                    f"{node.source}: seed target {target} does not exist in "
                    f"{existing_path}"
                )

    grouped: dict[int, list[ComponentNode]] = {}
    for index, node in enumerate(nodes):
        grouped.setdefault(union_find.find(index), []).append(node)

    components = list(grouped.values())
    component_survivors: dict[int, str | None] = {}
    for component in components:
        stable_ids = sorted(
            (node.local_id for node in component if node.existing),
            key=_candidate_id_sort_key,
        )
        component_survivors[id(component)] = _stable_component_survivor(
            stable_ids, aliases
        )

    new_components = sorted(
        (
            component
            for component in components
            if not any(node.existing for node in component)
        ),
        key=_component_sort_key,
    )
    next_number = _next_candidate_number(
        existing,
        (
            *aliases,
            *(alias.surviving_candidate_id for alias in aliases.values()),
        ),
    )
    assigned_ids = {
        id(component): f"C{next_number + offset:04d}"
        for offset, component in enumerate(new_components)
    }

    conflict_values: dict[ConflictSignature, PendingConflict] = {}
    merged = []
    for component in components:
        survivor_id = component_survivors[id(component)]
        existing_node = next(
            (
                node
                for node in component
                if node.existing and node.local_id == survivor_id
            ),
            None,
        )
        incoming = sorted(
            (node for node in component if not node.existing),
            key=_component_node_sort_key,
        )
        if existing_node is not None:
            observations = sorted(
                (
                    node
                    for node in component
                    if node is not existing_node
                ),
                key=_component_node_sort_key,
            )
            record = _merge_component_record(
                existing_node.row,
                existing_node.source,
                existing_node.row["screening_status"]
                in {"included", "boundary", "excluded"},
                observations,
                conflict_values,
            )
            duplicate_nodes = incoming
            new_node = None
        else:
            new_node = incoming[0]
            record = _new_record(new_node.row, assigned_ids[id(component)])
            record = _merge_component_record(
                record,
                new_node.source,
                False,
                incoming[1:], conflict_values,
            )
            duplicate_nodes = incoming[1:]
            stats.new_count += 1
            stats.new_by_file[new_node.source_file] += 1
            for stream in split_values(new_node.row["discovery_stream"]) or [
                "<missing>"
            ]:
                stats.new_by_stream[stream] += 1

        for node in duplicate_nodes:
            identity_type = _connection_type(node, component)
            if identity_type == "unknown" and id(node) in retirement_matches:
                identity_type = "retirement"
            stats.duplicate_count += 1
            stats.duplicate_by_file[node.source_file] += 1
            stats.identity_matches[identity_type] += 1
            for stream in split_values(node.row["discovery_stream"]) or [
                "<missing>"
            ]:
                stats.duplicate_by_stream[stream] += 1
                stats.duplicate_matches[(identity_type, stream)] += 1
        merged.append(record)

    conflicts = _build_conflict_rows(conflict_values)
    return sorted(merged, key=_candidate_sort_key), conflicts, stats
def merge_candidate_files(
    existing_path: Path,
    agent_paths: list[Path],
    aliases_path: Path | None = None,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    merged, conflicts, _ = _merge_candidate_files(
        Path(existing_path),
        [Path(path) for path in agent_paths],
        Path(aliases_path) if aliases_path is not None else None,
    )
    return merged, conflicts


def _read_conflict_rows(path: Path) -> list[CandidateRow]:
    reader: csv.DictReader | None = None
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            actual = tuple(reader.fieldnames or ())
            if actual != CONFLICT_HEADER:
                raise MergeError(
                    f"{path}: conflict columns must be exactly "
                    f"{list(CONFLICT_HEADER)}; found {list(actual)}"
                )
            rows = []
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(
                    value is None for value in raw_row.values()
                ):
                    raise MergeError(f"{path}:{row_number}: malformed CSV row")
                row = {
                    name: (raw_row.get(name) or "").strip()
                    for name in CONFLICT_HEADER
                }
                if not any(row.values()):
                    raise MergeError(
                        f"{path}:{row_number}: row is entirely blank"
                    )
                rows.append(row)
            return rows
    except UnicodeDecodeError as exc:
        raise MergeError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        line_number = reader.line_num if reader is not None else 1
        raise MergeError(
            f"{path}:{line_number}: CSV parse error: {exc}"
        ) from exc


def _ledger_conflict_signature(row: CandidateRow) -> tuple[str, ...]:
    field_name = row["field"]
    value_a, value_b = _canonical_conflict_values(
        field_name, row["value_a"], row["value_b"]
    )
    return (
        row["record_type"],
        row["record_key"],
        field_name,
        _conflict_value(field_name, value_a),
        _conflict_value(field_name, value_b),
    )


def _split_resolution_evidence(
    value: str,
) -> tuple[dict[str, set[str]], list[str]]:
    origins = {"value_a": set(), "value_b": set()}
    notes = []
    for item in split_values(value):
        label, separator, source = item.partition("=")
        if label in origins and separator and source:
            origins[label].add(
                re.sub(r"@row:[0-9]+$", "", source)
            )
        else:
            notes.append(item)
    return origins, notes


def _is_candidate_ledger_origin(source: str) -> bool:
    path = source.split("#", 1)[0]
    return Path(path).name == "candidates.csv"


def _merge_resolution_evidence(
    existing: CandidateRow, generated: CandidateRow
) -> str:
    origins, notes = _split_resolution_evidence(
        existing["resolution_evidence"]
    )
    generated_origins, generated_notes = _split_resolution_evidence(
        generated["resolution_evidence"]
    )
    field_name = existing["field"]
    same_orientation = _conflict_value(
        field_name, existing["value_a"]
    ) == _conflict_value(field_name, generated["value_a"])
    if not same_orientation:
        generated_origins = {
            "value_a": generated_origins["value_b"],
            "value_b": generated_origins["value_a"],
        }
    for label in origins:
        additions = generated_origins[label]
        if origins[label]:
            additions = {
                source
                for source in additions
                if not _is_candidate_ledger_origin(source)
            }
        origins[label].update(additions)
    for note in generated_notes:
        if note not in notes:
            notes.append(note)

    parts = list(notes)
    for label in ("value_a", "value_b"):
        parts.extend(
            f"{label}={source}"
            for source in sorted(origins[label], key=_value_sort_key)
        )
    return "; ".join(parts)


def _merge_reconciled_conflict(
    existing: CandidateRow, generated: CandidateRow
) -> CandidateRow:
    row = dict(existing)
    row["resolution_evidence"] = _merge_resolution_evidence(
        existing, generated
    )
    return row


def _migrate_conflict_record_key(
    row: CandidateRow, aliases: dict[str, str]
) -> CandidateRow:
    migrated = dict(row)
    if migrated["record_type"] == "candidate":
        migrated["record_key"] = aliases.get(
            migrated["record_key"], migrated["record_key"]
        )
    return migrated


def _reconcile_conflict_rows(
    generated: list[CandidateRow],
    existing: list[CandidateRow],
    *,
    replace: bool = False,
    aliases: dict[str, str] | None = None,
) -> list[CandidateRow]:
    alias_map = aliases or {}
    if replace:
        return [
            _migrate_conflict_record_key(row, alias_map)
            for row in generated
        ]

    reconciled: list[CandidateRow] = []
    indexes: dict[tuple[str, ...], int] = {}
    for source_row in existing:
        row = _migrate_conflict_record_key(source_row, alias_map)
        signature = _ledger_conflict_signature(row)
        if signature in indexes:
            index = indexes[signature]
            current = reconciled[index]
            current_reviewed = bool(current["resolution"] or current["resolver"])
            row_reviewed = bool(row["resolution"] or row["resolver"])
            if row_reviewed and not current_reviewed:
                reconciled[index] = _merge_reconciled_conflict(row, current)
            else:
                reconciled[index] = _merge_reconciled_conflict(current, row)
            continue
        indexes[signature] = len(reconciled)
        reconciled.append(row)

    for source_row in generated:
        row = _migrate_conflict_record_key(source_row, alias_map)
        signature = _ledger_conflict_signature(row)
        if signature in indexes:
            index = indexes[signature]
            reconciled[index] = _merge_reconciled_conflict(
                reconciled[index], row
            )
            continue
        indexes[signature] = len(reconciled)
        reconciled.append(row)
    return reconciled


def _write_temporary_rows(
    path: Path, rows: list[CandidateRow], header: tuple[str, ...]
) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=header,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _backup_existing_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.backup.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    backup = Path(name)
    try:
        shutil.copy2(path, backup)
        with backup.open("rb") as handle:
            os.fsync(handle.fileno())
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _restore_file(path: Path, backup: Path | None) -> None:
    if backup is None:
        path.unlink(missing_ok=True)
        return
    shutil.copy2(backup, path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _atomic_write_outputs(
    existing_path: Path,
    merged: list[CandidateRow],
    conflicts_path: Path,
    conflicts: list[CandidateRow],
) -> None:
    """Replace paired ledgers with best-effort rollback for caught failures.

    This is not crash-atomic across two files. Both outputs are staged and
    backed up before replacement. A caught replacement failure triggers
    rollback; incomplete rollback retains backups and reports recovery paths.
    """
    if existing_path.resolve() == conflicts_path.resolve():
        raise MergeError(
            "candidate and conflict output paths must be distinct: "
            f"{existing_path}"
        )

    staged_paths: list[Path] = []
    backups: dict[Path, Path | None] = {}
    targets = (existing_path, conflicts_path)
    retain_backups = False
    try:
        candidate_temporary = _write_temporary_rows(
            existing_path, merged, CANDIDATE_HEADER
        )
        staged_paths.append(candidate_temporary)
        conflict_temporary = _write_temporary_rows(
            conflicts_path, conflicts, CONFLICT_HEADER
        )
        staged_paths.append(conflict_temporary)

        for path in targets:
            backup = _backup_existing_file(path)
            backups[path] = backup

        try:
            candidate_temporary.replace(existing_path)
            conflict_temporary.replace(conflicts_path)
        except OSError as replacement_error:
            rollback_errors: list[tuple[Path, OSError]] = []
            for path in targets:
                try:
                    _restore_file(path, backups[path])
                except OSError as rollback_error:
                    rollback_errors.append((path, rollback_error))
            if rollback_errors:
                retain_backups = True
                recovery_paths = [
                    backup
                    for path in targets
                    if (backup := backups.get(path)) is not None
                ]
                recovery_text = ", ".join(
                    str(path) for path in recovery_paths
                ) or "<none available>"
                rollback_text = "; ".join(
                    f"{path}: {error}"
                    for path, error in rollback_errors
                )
                raise MergeError(
                    "pair replacement failed and rollback was incomplete; "
                    f"recovery backups: {recovery_text}; "
                    f"rollback errors: {rollback_text}"
                ) from replacement_error
            raise
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        if not retain_backups:
            for backup in backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)


def _print_report(
    mode: str,
    merged: list[CandidateRow],
    conflicts: list[CandidateRow],
    stats: MergeStats,
) -> None:
    print(f"mode={mode}")
    print(f"merged_total={len(merged)}")
    print(f"incoming_total={stats.incoming_total}")
    print(f"new_count={stats.new_count}")
    print(
        "excluded_count="
        f"{sum(row['screening_status'] == 'excluded' for row in merged)}"
    )
    print(f"duplicate_total={stats.duplicate_count}")

    for source_file in sorted(stats.source_files, key=_value_sort_key):
        print(
            f"source_file[{source_file}].incoming="
            f"{stats.incoming_by_file[source_file]}"
        )
        print(
            f"source_file[{source_file}].new="
            f"{stats.new_by_file[source_file]}"
        )
        print(
            f"source_file[{source_file}].duplicate="
            f"{stats.duplicate_by_file[source_file]}"
        )

    streams = sorted(stats.source_streams, key=_value_sort_key)
    for stream in streams:
        print(
            f"source_stream[{stream}].incoming="
            f"{stats.incoming_by_stream[stream]}"
        )
        print(f"source_stream[{stream}].new={stats.new_by_stream[stream]}")
        print(
            f"source_stream[{stream}].duplicate="
            f"{stats.duplicate_by_stream[stream]}"
        )

    for identity_type in EXTENDED_IDENTITY_ORDER:
        print(
            f"identity_matches[{identity_type}]="
            f"{stats.identity_matches[identity_type]}"
        )
        for stream in streams:
            print(
                f"duplicate_matches[{identity_type}][{stream}]="
                f"{stats.duplicate_matches[(identity_type, stream)]}"
            )
    if not streams:
        print("duplicate_matches=0")


    conflict_counts = Counter(row["field"] for row in conflicts)
    print(f"conflict_total={len(conflicts)}")
    for name in CONFLICT_FIELD_ORDER:
        print(f"conflicts[{name}]={conflict_counts[name]}")

    typed_counts = Counter(
        (row["record_type"] or "<missing>", row["field"] or "<missing>")
        for row in conflicts
    )
    for (record_type, field_name), count in sorted(
        typed_counts.items(),
        key=lambda item: (
            _value_sort_key(item[0][0]), _value_sort_key(item[0][1])
        ),
    ):
        print(
            f"conflicts_by_type[{record_type}][{field_name}]={count}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically merge survey candidate streams."
    )
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument(
        "--agent",
        "--agent-file",
        dest="agent",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--replace-conflicts", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.write and not arguments.agent:
        parser.error("--write requires at least one --agent-file")

    merged, conflicts, stats = _merge_candidate_files(
        arguments.existing, arguments.agent, arguments.aliases
    )
    if arguments.write:
        conflicts_path = arguments.existing.parent / "conflicts.csv"
        if arguments.existing.resolve() == conflicts_path.resolve():
            raise MergeError(
                "candidate and conflict output paths must be distinct: "
                f"{arguments.existing}"
            )
        existing_conflicts = (
            _read_conflict_rows(conflicts_path)
            if not arguments.replace_conflicts and conflicts_path.exists()
            else []
        )
        conflicts = _reconcile_conflict_rows(
            conflicts,
            existing_conflicts,
            replace=arguments.replace_conflicts,
            aliases=stats.retirement_aliases,
        )
        _atomic_write_outputs(
            arguments.existing,
            merged,
            conflicts_path,
            conflicts,
        )
    _print_report(
        "write" if arguments.write else "dry-run",
        merged,
        conflicts,
        stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
