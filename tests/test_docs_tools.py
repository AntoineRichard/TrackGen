"""The docs build tooling in ``docs/_tools`` (CPU lane, no Sphinx needed).

* ``md_table_to_csv``: keeps only the Markdown table of a benchmark's stdout (Warp's
  banner and ``Module ... load`` lines dropped, fixture: a real ``compare_generators``
  run), drops the separator, selects / prepends columns, writes the marked placeholder.
* ``strip_warp_log``: drops Warp's log lines so a collision output starts at its own line.
* ``check_readme_anchors``: a README link resolves only to a built page with that exact id.
* ``build_ci_env``: blocks the packages the docs CI environment does not install.
* ``check_labels``: the writer that owns each page.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "docs" / "_tools"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "docs_tools"


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"_docs_tool_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def md():
    return _tool("md_table_to_csv")


@pytest.fixture(scope="module")
def compare_output():
    return (FIXTURES / "compare_generators_cpu.txt").read_text()


def _rows(text):
    return list(csv.reader(io.StringIO(text)))


def test_md_table_keeps_only_the_table(md, compare_output):
    assert "Warp 1.18.0 initialized" in compare_output
    assert "load on device 'cpu'" in compare_output
    rows = _rows(md.convert(compare_output))
    assert rows[0][:2] == ["generator", "yield"] and rows[0][-1] == "gen_ms_per_call"
    assert len(rows[0]) == 14
    assert [r[0] for r in rows[1:]] == ["bezier", "checkpoint", "hull", "polar", "voronoi"]
    assert all(len(r) == 14 for r in rows)
    assert not any(c.startswith("---") for r in rows for c in r)


def test_md_table_columns_and_added_columns(md, compare_output):
    rows = _rows(md.convert(compare_output, columns=["generator", "yield", "gen_ms_per_call"],
                            add=[("E", "512")]))
    assert rows[0] == ["E", "generator", "yield", "gen_ms_per_call"]
    assert rows[1][:3] == ["512", "bezier", "1"]
    rows = _rows(md.convert(compare_output, columns=["generator"], header=False))
    assert rows[0] == ["bezier"]
    with pytest.raises(SystemExit):
        md.convert(compare_output, columns=["generator", "no_such_column"])
    with pytest.raises(SystemExit):
        md.convert("Warp 1.18.0 initialized:\n   no table here\n")


def test_md_table_placeholder(md):
    rows = _rows(md.convert("", columns=["generator", "gen_ms_per_call"], add=[("E", "64")],
                            placeholder="pending GPU re-measurement"))
    assert rows == [["E", "generator", "gen_ms_per_call"],
                    ["64", "pending GPU re-measurement"]]


def test_md_table_cli(md, compare_output, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(compare_output))
    assert md.main(["--columns", "generator,yield"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "generator,yield"
    assert out.splitlines()[1] == "bezier,1"


def test_strip_warp_log():
    strip = _tool("strip_warp_log").strip
    raw = (FIXTURES / "benchmark_collision_racing_cpu.txt").read_text()
    out = strip(raw)
    assert out.splitlines()[0].startswith("racing near-edge: device cpu")
    assert "initialized" not in out and "load on device" not in out
    assert "sdf flag agreement 99.949%" in out


def test_check_readme_anchors(tmp_path):
    tool = _tool("check_readme_anchors")
    html = tmp_path / "html"
    (html / "user-guide").mkdir(parents=True)
    (html / "index.html").write_text("<html></html>")
    (html / "user-guide" / "results.html").write_text(
        '<section id="invalid-envs"><h2>Invalid envs</h2></section>')
    readme = tmp_path / "README.md"
    site = tool.SITE
    readme.write_text(f"[a]({site}user-guide/results.html#invalid-envs) [b]({site})\n")
    assert tool.check(html, readme) == []
    readme.write_text(f"[a]({site}user-guide/results.html#invalid-env)\n"
                      f"[b]({site}user-guide/gone.html)\n")
    problems = tool.check(html, readme)
    assert len(problems) == 2
    assert 'no id="invalid-env"' in problems[0] and "no page" in problems[1]


def test_build_ci_env_blocks_the_test_only_packages():
    tool = _tool("build_ci_env")
    mods = {"torch": object(), "torch.cuda": object(), "_pytest.fixtures": object(),
            "numpy": object(), "sphinx": object()}
    tool.block_packages(mods)
    assert all(mods[n] is None for n in tool.BLOCKED)
    assert "torch.cuda" not in mods and "_pytest.fixtures" not in mods
    assert mods["numpy"] is not None and mods["sphinx"] is not None
    # in a fresh interpreter: as in the docs environment, import fails and find_spec
    # returns None (it never raises), and track_gen resolves to this checkout
    import subprocess
    script = (
        "import importlib, importlib.util, sys\n"
        f"sys.path.insert(0, {str(TOOLS)!r})\n"
        "import build_ci_env as b\n"
        "b.block_packages()\n"
        "for n in ('pytest', 'torch', 'scipy', 'matplotlib'):\n"
        "    assert importlib.util.find_spec(n) is None, n\n"
        "    try:\n"
        "        importlib.import_module(n)\n"
        "    except ModuleNotFoundError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(n)\n"
        "print(b.use_checkout())\n")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120, cwd=str(TOOLS))
    assert out.returncode == 0, out.stderr[-2000:]
    assert Path(out.stdout.strip()) == ROOT / "track_gen" / "__init__.py"


def test_check_labels_owners():
    owner_of = _tool("check_labels").owner_of
    assert owner_of("upgrading/from-0.1") == "W1"
    assert owner_of("user-guide/generating") == "W2"
    assert owner_of("user-guide/rl-auto-reset") == "W3"
    assert owner_of("utilities/collision") == "W3"
    assert owner_of("contributing/plugin-abis") == "W4"
    assert owner_of("contributing/dev-setup") == "W6"
    assert owner_of("reference/plugin") == "W5"
    assert owner_of("configuration/reference") == "W5"
    assert owner_of("configuration/tuning") == "W2"
    assert owner_of("getting-started/concepts") == "W7"


def test_linkcheck_summary(tmp_path):
    tool = _tool("linkcheck_summary")
    assert "the build stopped first" in tool.summarize(tmp_path)
    rows = [
        {"filename": "a.rst", "lineno": 3, "status": "working", "code": 0,
         "uri": "https://ok.example", "info": ""},
        {"filename": "b.rst", "lineno": 9, "status": "broken", "code": 404,
         "uri": "https://gone.example", "info": "404 Client Error | Not Found"},
        {"filename": "a.rst", "lineno": 5, "status": "timeout", "code": 0,
         "uri": "https://slow.example", "info": ""},
        {"filename": "a.rst", "lineno": 7, "status": "ignored", "code": 0,
         "uri": "https://paywall.example", "info": ""},
    ]
    (tmp_path / "output.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tool.summarize(tmp_path)
    assert "broken: 1, ignored: 1, timeout: 1, working: 1" in out
    table = [l for l in out.splitlines() if l.startswith("| ") and "example" in l]
    assert [l.split(" | ")[0] for l in table] == ["| a.rst", "| b.rst"]
    assert "https://gone.example" in table[1] and "404 Client Error \\| Not Found" in table[1]
    assert "ok.example" not in out and "paywall.example" not in out
    (tmp_path / "output.json").write_text(json.dumps(rows[0]) + "\n")
    assert "No broken or timed-out links." in tool.summarize(tmp_path)
    # a baseline entry is still listed, marked as expected with its reason
    uri = next(iter(tool.EXPECTED_BROKEN))
    (tmp_path / "output.json").write_text(json.dumps(
        {"filename": "upgrading/from-0.1.rst", "lineno": 16, "status": "broken", "code": 404,
         "uri": uri, "info": "404"}) + "\n")
    out = tool.summarize(tmp_path)
    assert f"broken (expected: {tool.EXPECTED_BROKEN[uri]})" in out and uri in out
