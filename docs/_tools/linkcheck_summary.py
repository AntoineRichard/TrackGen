#!/usr/bin/env python3
"""Summarise a ``sphinx -b linkcheck`` run as Markdown (for ``$GITHUB_STEP_SUMMARY``).

Usage: ``python docs/_tools/linkcheck_summary.py <linkcheck outdir>``

Reads ``<outdir>/output.json`` (one JSON object per checked link: ``filename``,
``lineno``, ``status``, ``code``, ``uri``, ``info``) and prints the count per status and
a table of every broken or timed-out link with its page and line. A link listed in
``EXPECTED_BROKEN`` is marked as expected, with its reason (it is still checked, and
still fails a strict run: remove the entry once the reason is gone). Missing output (the
build failed before checking) prints a one-line note. Always exits 0: the linkcheck step
itself carries the verdict.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

FAILED = ("broken", "timeout")

# Links broken for a known, temporary reason (the baseline): uri -> reason.
EXPECTED_BROKEN = {
    "https://github.com/AntoineRichard/TrackGen/blob/main/CHANGELOG.md":
        "expected until v2 merges into main (main has no CHANGELOG.md before that)",
}


def summarize(outdir: Path) -> str:
    path = outdir / "output.json"
    if not path.is_file():
        return f"### Link check\n\nNo `{path.name}` in `{outdir}`: the build stopped first.\n"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    counts = Counter(r.get("status", "?") for r in rows)
    out = ["### Link check", "",
           ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no links", ""]
    bad = [r for r in rows if r.get("status") in FAILED]
    if bad:
        out += ["| Page | Line | Status | URL | Detail |", "|---|---|---|---|---|"]
        for r in sorted(bad, key=lambda r: (r.get("filename", ""), r.get("lineno", 0))):
            info = str(r.get("info", "")).replace("|", "\\|").replace("\n", " ")[:160]
            status = r.get("status")
            if r.get("uri") in EXPECTED_BROKEN:
                status = f"{status} (expected: {EXPECTED_BROKEN[r['uri']]})"
            out.append(f"| {r.get('filename', '')} | {r.get('lineno', '')} | "
                       f"{status} | {r.get('uri', '')} | {info} |")
    else:
        out.append("No broken or timed-out links.")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    sys.stdout.write(summarize(Path(argv[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
