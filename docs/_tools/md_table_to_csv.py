#!/usr/bin/env python3
"""Convert the Markdown table a benchmark prints into the CSV a docs page shows.

Reads stdin, writes CSV to stdout. Keeps only the lines that start with ``|`` (Warp's
init banner and its ``Module ... load`` lines go to stdout too, and are dropped), drops the
``| --- |`` separator, and optionally selects / prepends columns::

    python -m benchmarks.compare_generators --E 512 --seed 0 \\
        --generators bezier checkpoint hull polar voronoi \\
        | python docs/_tools/md_table_to_csv.py > docs/_data/metrics-cpu.csv

Options:
  --columns a,b,c       keep these columns, in this order (an unknown name is an error)
  --add-column K=V      prepend a column K whose value is V on every row (repeatable),
                        e.g. ``--add-column E=8192`` to concatenate runs at several E
  --no-header           omit the header row (append a second run with ``>>``)
  --placeholder TEXT    ignore stdin and write the header (``--columns`` and
                        ``--add-column`` names) plus one row whose first cell is TEXT: the
                        marked placeholder a data file holds until its command is run

Exit status 1 when stdin holds no table (and no ``--placeholder``).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys

_SEPARATOR = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$")


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    rows = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line.startswith("|") or _SEPARATOR.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def convert(text: str, *, columns=None, add=(), header=True, placeholder=None) -> str:
    extra_names = [k for k, _ in add]
    extra_vals = [v for _, v in add]
    if placeholder is not None:
        if not columns and not extra_names:
            raise SystemExit("md_table_to_csv: --placeholder needs --columns")
        out_rows = [[placeholder]]
        if extra_names:
            out_rows = [extra_vals + [placeholder]]
        names = extra_names + list(columns or [])
    else:
        names, rows = parse_table(text)
        if not names:
            raise SystemExit("md_table_to_csv: no Markdown table on stdin")
        if columns:
            missing = [c for c in columns if c not in names]
            if missing:
                raise SystemExit(f"md_table_to_csv: unknown columns {missing}; "
                                 f"the table has {names}")
            idx = [names.index(c) for c in columns]
            rows = [[r[i] if i < len(r) else "" for i in idx] for r in rows]
            names = list(columns)
        out_rows = [extra_vals + r for r in rows]
        names = extra_names + names
    buf = []

    class _W:
        def write(self, s):
            buf.append(s)

    w = csv.writer(_W(), lineterminator="\n")
    if header:
        w.writerow(names)
    w.writerows(out_rows)
    return "".join(buf)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--columns", type=lambda s: [c.strip() for c in s.split(",") if c.strip()])
    ap.add_argument("--add-column", action="append", default=[], metavar="K=V")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--placeholder")
    a = ap.parse_args(argv)
    add = []
    for kv in a.add_column:
        k, eq, v = kv.partition("=")
        if not eq or not k:
            ap.error(f"--add-column needs K=V, got {kv!r}")
        add.append((k, v))
    text = "" if a.placeholder is not None else sys.stdin.read()
    sys.stdout.write(convert(text, columns=a.columns, add=add, header=not a.no_header,
                             placeholder=a.placeholder))
    return 0


if __name__ == "__main__":
    sys.exit(main())
