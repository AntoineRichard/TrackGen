from __future__ import annotations

import argparse
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
    name: index for index, name in enumerate(BIBLIOGRAPHIC_FIELDS)
}
STABLE_CANDIDATE_PATTERN = re.compile(r"C([0-9]{4,})")
ARXIV_ID_PATTERN = re.compile(
    r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z0-9.-]+/[0-9]{7})",
    re.IGNORECASE,
)
ARXIV_VERSION_PATTERN = re.compile(r"v[0-9]+$", re.IGNORECASE)
CANDIDATE_HEADER = HEADERS["candidates.csv"]
CONFLICT_HEADER = HEADERS["conflicts.csv"]
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


@dataclass
class PendingConflict:
    candidate_id: str
    field_name: str
    current: str
    proposed: str
    existing_source: str
    incoming_sources: set[str] = field(default_factory=set)


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


def _next_candidate_number(rows: list[CandidateRow]) -> int:
    numbers = [
        int(match.group(1))
        for row in rows
        if (match := STABLE_CANDIDATE_PATTERN.fullmatch(row["candidate_id"]))
    ]
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


def _record_conflict(
    conflicts: dict[ConflictSignature, PendingConflict],
    candidate_id: str,
    field_name: str,
    current: str,
    proposed: str,
    incoming_source: str,
) -> None:
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
            existing_source=f"candidates.csv#{candidate_id}",
        ),
    )
    pending.incoming_sources.add(incoming_source)


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
    for number, pending in enumerate(values, start=1):
        row = dict.fromkeys(CONFLICT_HEADER, "")
        source_context = [f"existing={pending.existing_source}"]
        source_context.extend(
            f"incoming={source}"
            for source in sorted(
                pending.incoming_sources, key=_value_sort_key
            )
        )
        row.update(
            conflict_id=f"X{number:04d}",
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


def _merge_candidate_files(
    existing_path: Path,
    agent_paths: Sequence[Path],
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

    merged = [dict(row) for row in existing]
    lookup: dict[IdentityKey, set[int]] = {}
    for index, row in enumerate(merged):
        _add_to_lookup(lookup, row, index)

    stats = MergeStats(existing_count=len(existing))
    incoming_rows: list[IncomingRecord] = []
    for path in sorted((Path(path) for path in agent_paths), key=str):
        source_file = path.name
        rows = _read_candidate_rows(path, existing=False)
        stats.source_files.add(source_file)
        stats.incoming_total += len(rows)
        stats.incoming_by_file[source_file] += len(rows)
        for row in rows:
            streams = split_values(row["discovery_stream"]) or ["<missing>"]
            for stream in streams:
                stats.source_streams.add(stream)
                stats.incoming_by_stream[stream] += 1
            incoming_rows.append(
                IncomingRecord(row, source_file, row["candidate_id"])
            )
    incoming_rows.sort(key=_incoming_sort_key)

    next_number = _next_candidate_number(merged)
    conflict_values: dict[ConflictSignature, PendingConflict] = {}

    for incoming_record in incoming_rows:
        incoming = incoming_record.row
        streams = split_values(incoming["discovery_stream"]) or ["<missing>"]
        index, match_type = _find_match(incoming, merged, lookup)
        if index is None:
            record = _new_record(incoming, f"C{next_number:04d}")
            next_number += 1
            merged.append(record)
            index = len(merged) - 1
            _add_to_lookup(lookup, record, index)
            stats.new_count += 1
            stats.new_by_file[incoming_record.source_file] += 1
            for stream in streams:
                stats.new_by_stream[stream] += 1
            continue

        identity_type = match_type or "unknown"
        stats.duplicate_count += 1
        stats.duplicate_by_file[incoming_record.source_file] += 1
        stats.identity_matches[identity_type] += 1
        for stream in streams:
            stats.duplicate_by_stream[stream] += 1
            stats.duplicate_matches[(identity_type, stream)] += 1

        record = merged[index]
        if incoming["screening_status"] == "excluded":
            record["screening_status"] = "excluded"
            record["exclusion_reason"] = incoming["exclusion_reason"]
        for name in PROVENANCE_FIELDS:
            record[name] = _canonical_union(record[name], incoming[name])
        record["metadata_evidence"] = _append_unique_values(
            record["metadata_evidence"], incoming["metadata_evidence"]
        )

        for name in BIBLIOGRAPHIC_FIELDS:
            current = record[name]
            proposed = incoming[name]
            if not current and proposed:
                record[name] = proposed
                continue
            if not proposed or _equivalent(name, current, proposed):
                continue
            _record_conflict(
                conflict_values,
                record["candidate_id"],
                name,
                current,
                proposed,
                incoming_record.source,
            )
            record["metadata_status"] = "conflict"
        _add_to_lookup(lookup, record, index)

    conflicts = _build_conflict_rows(conflict_values)
    return sorted(merged, key=_candidate_sort_key), conflicts, stats


def merge_candidate_files(
    existing_path: Path,
    agent_paths: list[Path],
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    merged, conflicts, _ = _merge_candidate_files(
        Path(existing_path), [Path(path) for path in agent_paths]
    )
    return merged, conflicts


def _write_temporary_rows(
    path: Path, rows: list[CandidateRow], header: tuple[str, ...]
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=header, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    if path.exists():
        os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
    return temporary_path


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
    else:
        os.replace(backup, path)


def _atomic_write_outputs(
    existing_path: Path,
    merged: list[CandidateRow],
    conflicts_path: Path,
    conflicts: list[CandidateRow],
) -> None:
    """Replace both ledgers with rollback-guaranteed cross-file atomicity.

    Both files are staged and backed up before replacement. If either data
    replacement fails, both original byte streams are restored before the
    replacement error is propagated.
    """
    temporary_paths = []
    backups: dict[Path, Path | None] = {}
    targets = (existing_path, conflicts_path)
    try:
        candidate_temporary = _write_temporary_rows(
            existing_path, merged, CANDIDATE_HEADER
        )
        temporary_paths.append(candidate_temporary)
        conflict_temporary = _write_temporary_rows(
            conflicts_path, conflicts, CONFLICT_HEADER
        )
        temporary_paths.append(conflict_temporary)
        backups = {path: _backup_existing_file(path) for path in targets}
        temporary_paths.extend(
            backup for backup in backups.values() if backup is not None
        )
        try:
            candidate_temporary.replace(existing_path)
            conflict_temporary.replace(conflicts_path)
        except OSError as replacement_error:
            rollback_errors = []
            for path in targets:
                try:
                    _restore_file(path, backups[path])
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise MergeError(
                    "pair replacement failed and rollback was incomplete"
                ) from replacement_error
            raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


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

    for identity_type in IDENTITY_ORDER:
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
    for name in BIBLIOGRAPHIC_FIELDS:
        print(f"conflicts[{name}]={conflict_counts[name]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically merge survey candidate streams."
    )
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--agent", type=Path, action="append", default=[])
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args(argv)

    merged, conflicts, stats = _merge_candidate_files(
        arguments.existing, arguments.agent
    )
    if arguments.write:
        _atomic_write_outputs(
            arguments.existing,
            merged,
            arguments.existing.parent / "conflicts.csv",
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
