#!/usr/bin/env python3
"""Check README.md's links into the published docs against a built HTML tree.

Usage: ``python docs/_tools/check_readme_anchors.py <html outdir> [README.md]``

Every ``https://antoinerichard.github.io/TrackGen/<path>.html[#<anchor>]`` in the README
must name a page of the build (``<outdir>/<path>.html``) and, with an anchor, an element
of that page with exactly ``id="<anchor>"`` (the ids the build wrote, not a
re-implementation of docutils' slugs). The site root (``.../TrackGen/``) maps to
``index.html``. Exit status 1 with one line per broken link.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SITE = "https://antoinerichard.github.io/TrackGen/"
_LINK = re.compile(re.escape(SITE) + r"(?P<path>[^)\s#\"'>]*)(?:#(?P<anchor>[^)\s\"'>]+))?")


def check(outdir: Path, readme: Path) -> list[str]:
    problems = []
    ids_cache: dict[Path, set[str]] = {}
    text = readme.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in _LINK.finditer(line):
            path = m.group("path") or "index.html"
            if path.endswith("/"):
                path += "index.html"
            page = outdir / path
            if not page.is_file():
                problems.append(f"{readme.name}:{lineno}: {m.group(0)}: no page {path} in "
                                f"{outdir}")
                continue
            anchor = m.group("anchor")
            if anchor:
                if page not in ids_cache:
                    ids_cache[page] = set(re.findall(r'\bid="([^"]+)"',
                                                     page.read_text(encoding="utf-8")))
                if anchor not in ids_cache[page]:
                    problems.append(f"{readme.name}:{lineno}: {m.group(0)}: no id="
                                    f"\"{anchor}\" in {path}")
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or len(argv) > 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    outdir = Path(argv[0])
    readme = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2] / "README.md"
    if not (outdir / "index.html").is_file():
        print(f"{outdir}: not an HTML build (no index.html)", file=sys.stderr)
        return 2
    problems = check(outdir, readme)
    for p in problems:
        print(p)
    if not problems:
        print(f"{readme.name}: every published-docs link resolves in {outdir}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
