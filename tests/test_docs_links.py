"""README and docs links stay valid without building the docs (CPU lane, no docutils).

* Every ``https://antoinerichard.github.io/TrackGen/<path>.html`` in ``README.md`` names
  a page the docs build publishes: ``docs/<path>.rst`` exists and is not in an excluded
  directory. Anchors are checked after the HTML build by
  ``docs/_tools/check_readme_anchors.py`` (against the ids the build wrote).
* Every relative image / file path in ``README.md`` exists.
* No page links into the directories the site excludes (``superpowers``,
  ``library-readiness``, ``code-review-*``): no ``:doc:`` target and no published URL.
"""
from __future__ import annotations

import posixpath
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"
SITE = "https://antoinerichard.github.io/TrackGen/"
EXCLUDED = ("superpowers/", "library-readiness/", "code-review-")
NOT_SOURCES = {"_build", "_ext", "_tools", "_data", "_static", "superpowers",
               "library-readiness"}

_SITE_LINK = re.compile(re.escape(SITE) + r"(?P<path>[^)\s#\"'>]*)")
_MD_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)\s]+)\)")
_IMG_SRC = re.compile(r"<img\b[^>]*\bsrc=\"(?P<target>[^\"]+)\"")
_DOC_ROLE = re.compile(r":doc:`(?:[^`<]*<)?(?P<target>[^`>]+)>?`")


def _pages() -> list[Path]:
    return [p for p in sorted(DOCS.rglob("*.rst"))
            if p.relative_to(DOCS).parts[0] not in NOT_SOURCES]


def _readme_site_paths():
    text = README.read_text(encoding="utf-8")
    return sorted({m.group("path") for m in _SITE_LINK.finditer(text)})


@pytest.mark.parametrize("path", _readme_site_paths())
def test_readme_published_link_has_a_page(path):
    page = path or "index.html"
    if page.endswith("/"):
        page += "index.html"
    assert page.endswith(".html"), f"README links {SITE}{path}: not a page URL"
    assert not page.startswith(EXCLUDED), f"README links an excluded page: {SITE}{path}"
    src = DOCS / (page[:-len(".html")] + ".rst")
    assert src.is_file(), (f"README links {SITE}{path} but docs/{src.relative_to(DOCS)} "
                           f"does not exist (the page moved: update the README)")


def test_readme_relative_paths_exist():
    text = README.read_text(encoding="utf-8")
    targets = [m.group("target") for m in _MD_LINK.finditer(text)]
    targets += [m.group("target") for m in _IMG_SRC.finditer(text)]
    missing = []
    for t in targets:
        if re.match(r"^[a-z]+://|^#|^mailto:", t):
            continue
        if not (ROOT / t.split("#", 1)[0]).exists():
            missing.append(t)
    assert not missing, f"README.md links missing files: {missing}"


def _doc_target(rel: Path, t: str) -> str:
    """The docs-relative document a ``:doc:`` target in page ``rel`` names (``..``
    resolved)."""
    if t.startswith("/"):
        return posixpath.normpath(t.lstrip("/"))
    return posixpath.normpath((rel.parent / t).as_posix())


def test_doc_target_normalises_relative_paths():
    assert _doc_target(Path("how-it-works/cuda-graph.rst"), "../library-readiness/x") == \
        "library-readiness/x"
    assert _doc_target(Path("a/b.rst"), "./c") == "a/c"
    assert _doc_target(Path("a/b.rst"), "/superpowers/y") == "superpowers/y"
    assert _doc_target(Path("index.rst"), "user-guide/results") == "user-guide/results"


def test_no_link_into_excluded_dirs():
    bad = []
    for p in _pages():
        rel = p.relative_to(DOCS)
        text = p.read_text(encoding="utf-8")
        for m in _DOC_ROLE.finditer(text):
            t = m.group("target").strip()
            target = _doc_target(rel, t)
            if target.startswith(EXCLUDED):
                bad.append(f"{rel}: :doc:`{t}`")
        for m in _SITE_LINK.finditer(text):
            if m.group("path").startswith(EXCLUDED):
                bad.append(f"{rel}: {m.group(0)}")
    for m in _SITE_LINK.finditer(README.read_text(encoding="utf-8")):
        if m.group("path").startswith(EXCLUDED):
            bad.append(f"README.md: {m.group(0)}")
    assert not bad, "links into pages the site excludes:\n" + "\n".join(bad)
