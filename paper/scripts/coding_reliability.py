from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


CORE_FIELDS = (
    "domain",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "code_status",
)


def _required_text(
    row: dict[str, str],
    field: str,
    *,
    row_number: int,
    source: str,
) -> str:
    if field not in row:
        raise ValueError(f"{source} row {row_number}: missing required field {field!r}")
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} row {row_number}: {field} must be nonblank")
    return value


def _required_cite_key(
    row: dict[str, str],
    *,
    row_number: int,
    source: str,
) -> str:
    cite_key = _required_text(
        row,
        "cite_key",
        row_number=row_number,
        source=source,
    )
    if cite_key != cite_key.strip():
        raise ValueError(
            f"{source} row {row_number}: cite_key must not contain "
            "surrounding whitespace"
        )
    return cite_key


def select_reliability_sample(
    evidence: list[dict[str, str]],
    fraction: float = 0.20,
) -> list[dict[str, str]]:
    try:
        valid_fraction = math.isfinite(fraction) and 0 < fraction <= 1
    except (TypeError, ValueError):
        valid_fraction = False
    if not valid_fraction:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")

    by_domain: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    seen_keys: set[str] = set()
    for row_number, row in enumerate(evidence, start=1):
        cite_key = _required_cite_key(
            row,
            row_number=row_number,
            source="evidence",
        )
        if cite_key in seen_keys:
            raise ValueError(
                f"evidence row {row_number}: duplicate cite_key {cite_key!r}"
            )
        seen_keys.add(cite_key)

        domain = _required_text(
            row,
            "domain",
            row_number=row_number,
            source="evidence",
        )
        labels = [label.strip() for label in domain.split(";") if label.strip()]
        if not labels:
            raise ValueError(f"evidence row {row_number}: domain must be nonblank")
        by_domain[labels[0]].append(row)

    selected: dict[str, dict[str, str]] = {}
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        count = min(len(rows), max(2, math.ceil(fraction * len(rows))))
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(row["cite_key"].encode("utf-8")).hexdigest(),
                row["cite_key"],
            ),
        )
        for row in ranked[:count]:
            selected.setdefault(row["cite_key"], row)
    return [selected[cite_key] for cite_key in sorted(selected)]


def _canonical(value: str) -> str:
    return ";".join(
        sorted(label.strip() for label in value.split(";") if label.strip())
    )


def cohens_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must have equal nonzero length")
    sample_size = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / sample_size
    left_counts = Counter(left)
    right_counts = Counter(right)
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[value] / sample_size) * (right_counts[value] / sample_size)
        for value in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _index_codings(
    rows: list[dict[str, str]],
    *,
    source: str,
) -> dict[str, dict[str, str]]:
    if not rows:
        raise ValueError(f"{source} coding sample must be nonempty")

    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        cite_key = _required_cite_key(
            row,
            row_number=row_number,
            source=source,
        )
        if cite_key in indexed:
            raise ValueError(
                f"{source} row {row_number}: duplicate cite_key {cite_key!r}"
            )
        for field in CORE_FIELDS:
            if field not in row:
                raise ValueError(
                    f"{source} row {row_number}: missing required field {field!r}"
                )
            value = row[field]
            if not isinstance(value, str):
                raise ValueError(
                    f"{source} row {row_number}: field {field!r} must be text"
                )
            if not _canonical(value):
                raise ValueError(
                    f"{source} row {row_number}: field {field!r} must have "
                    "a nonempty canonical value"
                )
        indexed[cite_key] = row
    return indexed


def compare_codings(
    primary: list[dict[str, str]],
    reliability: list[dict[str, str]],
) -> list[dict[str, str]]:
    left = _index_codings(primary, source="primary")
    right = _index_codings(reliability, source="reliability")
    if set(left) != set(right):
        raise ValueError(
            "coding samples differ: "
            f"primary={sorted(left)}, reliability={sorted(right)}"
        )

    keys = sorted(left)
    summary: list[dict[str, str]] = []
    for field in CORE_FIELDS:
        values_left = [_canonical(left[key][field]) for key in keys]
        values_right = [_canonical(right[key][field]) for key in keys]
        agreement = (
            sum(a == b for a, b in zip(values_left, values_right)) / len(keys)
        )
        left_categories = set(values_left)
        right_categories = set(values_right)
        kappa = (
            f"{cohens_kappa(values_left, values_right):.6f}"
            if len(left_categories) >= 2 and len(right_categories) >= 2
            else "NR"
        )
        summary.append(
            {
                "field": field,
                "n": str(len(keys)),
                "agreement": f"{agreement:.6f}",
                "kappa": kappa,
                "passes": str(agreement >= 0.80).lower(),
            }
        )
    return summary


SUMMARY_FIELDS = ("field", "n", "agreement", "kappa", "passes")


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"{path}: CSV file is missing")

    reader: csv.DictReader
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = tuple(reader.fieldnames or ())
            if not header:
                raise ValueError(f"{path}: CSV header is missing")
            if any(field is None or not field.strip() for field in header):
                raise ValueError(f"{path}: CSV header contains a blank column name")
            duplicate_fields = sorted(
                field
                for field, count in Counter(header).items()
                if count > 1
            )
            if duplicate_fields:
                raise ValueError(
                    f"{path}: CSV header contains duplicate columns "
                    f"{duplicate_fields}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8: {exc}") from exc
    except csv.Error as exc:
        line_number = getattr(reader, "line_num", "?")
        raise ValueError(
            f"{path}:{line_number}: CSV parse error: {exc}"
        ) from exc

    validated_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{row_number}: malformed CSV row")
        validated_rows.append(row)
    if not validated_rows:
        raise ValueError(f"{path}: CSV must contain at least one data row")
    return header, validated_rows


def _require_columns(
    path: Path,
    header: tuple[str, ...],
    required: tuple[str, ...],
) -> None:
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(f"{path}: required columns are missing: {missing}")


def _atomic_write_csv(
    path: Path,
    rows: list[dict[str, str]],
    header: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        destination_mode = (
            stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        )
        os.chmod(temporary_path, destination_mode)
        temporary_path.replace(path)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and compare deterministic survey coding samples."
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_mode(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    if arguments.prepare:
        if arguments.primary is not None or arguments.reliability is not None:
            parser.error(
                "--prepare cannot be combined with --primary or --reliability"
            )
        if arguments.evidence is None:
            parser.error("--prepare requires --evidence")
        return

    if arguments.evidence is not None:
        parser.error("--evidence is only valid with --prepare")
    if arguments.primary is None or arguments.reliability is None:
        parser.error(
            "comparison mode requires both --primary and --reliability"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    _validate_mode(parser, arguments)

    if arguments.prepare:
        evidence_path = arguments.evidence
        assert evidence_path is not None
        header, evidence = _read_csv(evidence_path)
        _require_columns(evidence_path, header, ("cite_key", "domain"))
        rows = select_reliability_sample(evidence)
        output_header = header
    else:
        primary_path = arguments.primary
        reliability_path = arguments.reliability
        assert primary_path is not None
        assert reliability_path is not None
        primary_header, primary = _read_csv(primary_path)
        reliability_header, reliability = _read_csv(reliability_path)
        required = ("cite_key", *CORE_FIELDS)
        _require_columns(primary_path, primary_header, required)
        _require_columns(reliability_path, reliability_header, required)
        rows = compare_codings(primary, reliability)
        output_header = SUMMARY_FIELDS

    _atomic_write_csv(arguments.output, rows, output_header)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
