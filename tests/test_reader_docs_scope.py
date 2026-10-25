"""The reader site describes current behavior; project records stay excluded."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXCLUDED = {"_build", "_ext", "_tools", "_data", "_static", "library-readiness", "superpowers"}


def test_history_is_archived_outside_reader_sources():
    for name in ("upgrading", "related-work"):
        assert not (DOCS / name).exists()
        assert (DOCS / "library-readiness" / "archive" / name).is_dir()
    assert not (DOCS / "appendix" / "future-generators.rst").exists()
    assert (DOCS / "library-readiness" / "handoff-0.3" / "pr16-spec.md").is_file()


def test_reader_pages_do_not_link_to_archived_upgrade_labels():
    bad=[]
    for page in DOCS.rglob("*.rst"):
        if page.relative_to(DOCS).parts[0] in EXCLUDED:
            continue
        for i,line in enumerate(page.read_text().splitlines(),1):
            if re.search(r":ref:`[^`]*upgrading-|historical measurements|pre-0\.3|Coming from 0\.1|(?:scheduled|arrives) after 1\.0|Plugins written before Params existed",line,re.I):
                bad.append(f"{page.relative_to(DOCS)}:{i}: {line}")
    assert not bad, "\n".join(bad)
