"""The documentation examples run (the extraction, grammar and runner: ``tests/_doc_snippets.py``).

* ``test_snippets_cpu[<page>::<group>]``: every group of every page under ``docs/``,
  ``README.md`` and the public docstrings, on the Warp ``cpu`` device (``"cuda"`` /
  ``"cuda:N"`` string literals rewritten to ``"cpu"``). Groups marked ``cuda`` have no CPU
  item; groups marked ``slow`` carry the ``slow`` marker (not in the fast lane).
* ``test_snippets_cuda[<page>::<group>]`` (``cuda`` marker, GPU lane): every group,
  unmodified, on the default CUDA device.
* ``test_snippet_metadata``: the ``.. snippet:`` grammar and layout rules hold on every
  page (a leftover ``page-pending`` marker fails it); prints the per-page report (blocks
  run / skipped by reason).

These run in the existing lanes: the CPU items in ``-m "not cuda and not slow and not
benchmark"``, the ``cuda`` items in ``-m cuda`` (``TRACKGEN_REQUIRE_CUDA=1`` on the GPU
workstation: none may skip).

Select one page: ``pytest tests/test_doc_snippets.py -k batch-of-tracks``.
"""
from __future__ import annotations

import pytest

from tests import _doc_snippets as ds

_PAGES = ds.collect()


def _params(cuda: bool):
    out = []
    for pid, page in sorted(_PAGES.items()):
        for gname, group in page.groups().items():
            if not group.runnable:
                continue
            if group.cuda and not cuda:
                continue
            marks = []
            if group.slow:
                marks.append(pytest.mark.slow)
            if cuda:
                marks.append(pytest.mark.cuda)
            out.append(pytest.param(pid, gname, marks=marks, id=f"{pid}::{gname}"))
    return out


@pytest.mark.parametrize("page,group", _params(cuda=False))
def test_snippets_cpu(page, group, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds.run_group(_PAGES, page, group, cpu=True)


@pytest.mark.parametrize("page,group", _params(cuda=True))
def test_snippets_cuda(page, group, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds.run_group(_PAGES, page, group, cpu=False)


def test_snippet_metadata():
    print(ds.report(_PAGES))
    errors = ds.metadata_errors(_PAGES)
    assert not errors, "documentation snippet metadata errors:\n" + "\n".join(errors)


# --------------------------------------------------------------------------------------
# the grammar and the runner, on synthetic pages
# --------------------------------------------------------------------------------------

def _rst(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    page = ds.Page(name, p, "rst")
    ds.parse_rst(page)
    return page


_GOOD = """\
Title
=====

.. snippet: setup

   given = 2

.. code-block:: python

   import warnings
   x = given + 1

.. snippet: after

   assert x == 3

.. snippet: group=other register=reg

.. code-block:: python

   seen = []
   def reg(registry):
       seen.append(registry)

.. snippet: group=other expect-warning=DeprecationWarning

.. code-block:: python

   import track_gen
   track_gen.set_capturing(False)
   assert len(seen) == 1

.. snippet: skip=signature

.. code-block:: python

   generate(seeds, *, advance=False) -> Track

.. snippet: group=boom expect-error=ZeroDivisionError

.. code-block:: python

   device = "cuda:0"
   assert device == "cpu"
   1 / 0
"""


def test_grammar_and_runner(tmp_path):
    page = _rst(tmp_path, "good.rst", _GOOD)
    assert page.errors == []
    groups = page.groups()
    assert sorted(groups) == ["boom", "main", "other"]
    main = groups["main"]
    assert [b.lineno for b in main.blocks] == [10, 37]
    assert main.blocks[0].setup[0].code == "given = 2\n"
    assert main.blocks[0].after[0].lineno == 15
    assert main.blocks[1].skipped
    pages = {page.id: page}
    for g in groups:
        ds.run_group(pages, page.id, g, cpu=True)


def test_grammar_requires_and_cycles(tmp_path):
    a = _rst(tmp_path, "a.rst", "A\n=\n\n.. snippet: group=base\n\n.. code-block:: python\n\n"
                                "   value = 41\n")
    b = _rst(tmp_path, "b.rst", "B\n=\n\n.. snippet: requires=a#base\n\n"
                                ".. code-block:: python\n\n   assert value + 1 == 42\n")
    pages = {a.id: a, b.id: b}
    assert ds.resolve_requires(pages, "b.rst", "main") == [("a.rst", "base")]
    ds.run_group(pages, "b.rst", "main", cpu=True)
    c = _rst(tmp_path, "c.rst", "C\n=\n\n.. snippet: requires=c#x\n\n.. code-block:: python\n\n"
                                "   pass\n\n.. snippet: group=x requires=c\n\n"
                                ".. code-block:: python\n\n   pass\n")
    with pytest.raises(ValueError, match="cycle"):
        ds.resolve_requires({c.id: c}, "c.rst", "main")


@pytest.mark.parametrize("text,needle", [
    ("T\n=\n\nprose\n.. snippet: skip=fragment\n\n.. code-block:: python\n\n   x\n",
     "must be blank"),
    ("T\n=\n\n.. snippet: group=g\n\nprose in between\n\n.. code-block:: python\n\n   x\n",
     "not followed by a python block"),
    ("T\n=\n\n.. snippet: skip=lazy\n\n.. code-block:: python\n\n   x\n", "skip reason"),
    ("T\n=\n\n.. snippet: bogus\n\n.. code-block:: python\n\n   x\n", "unknown snippet token"),
    ("T\n=\n\n.. code-block:: python\n\n   x = 1\n\n.. snippet: cuda\n\n"
     ".. code-block:: python\n\n   y = 2\n", "group token"),
    ("T\n=\n\n.. code-block:: python\n\n   d = f\"cuda:{0}\"\n", "f-string"),
    (".. snippet: page-pending\n\nT\n=\n\n.. code-block:: python\n\n   x = 1\n",
     "page-pending is retired"),
    ("T\n=\n\n.. snippet: register=nothere\n\n.. code-block:: python\n\n   x = 1\n",
     "names no callable"),
    # Python blocks in a form the extractor does not run are refused, not silently skipped
    ("T\n=\n\n.. code-block:: python3\n\n   raise RuntimeError\n", "renders as Python"),
    ("T\n=\n\n.. code:: python\n\n   raise RuntimeError\n", "renders as Python"),
    ("T\n=\n\n.. code-block:: py\n\n   raise RuntimeError\n", "renders as Python"),
    ("T\n=\n\n.. sourcecode:: python\n\n   raise RuntimeError\n", "renders as Python"),
    ("T\n=\n\n.. code-block::\n\n   raise RuntimeError\n", "renders as Python"),
    ("T\n=\n\n.. code-block:: pycon\n\n   >>> raise RuntimeError\n", "renders as Python"),
    # expect-* must name a specific class
    ("T\n=\n\n.. snippet: expect-error=Exception\n\n.. code-block:: python\n\n   1 / 0\n",
     "would accept anything"),
    ("T\n=\n\n.. snippet: expect-error=BaseException\n\n.. code-block:: python\n\n   1 / 0\n",
     "would accept anything"),
    ("T\n=\n\n.. snippet: expect-warning=Warning\n\n.. code-block:: python\n\n   x = 1\n",
     "would accept anything"),
    # a CUDA-only example belongs in a cuda group, not behind a skip
    ("T\n=\n\n.. snippet: skip=fragment\n\n.. code-block:: python\n\n"
     "   s = torch.cuda.Stream()\n", "CUDA-only API"),
])
def test_grammar_errors(tmp_path, text, needle):
    page = _rst(tmp_path, "bad.rst", text)
    assert any(needle in e for e in page.errors), page.errors


@pytest.mark.parametrize("fence", ["   ```python", "```py", "```python3", "```python title"])
def test_grammar_markdown_refuses_unrun_python_fences(tmp_path, fence):
    md = tmp_path / "README.md"
    md.write_text(f"# T\n\n- item\n\n{fence}\nraise RuntimeError\n```\n")
    page = ds.Page("README.md", md, "md")
    ds.parse_md(page)
    assert any("Python fence" in e for e in page.errors), page.errors
    assert page.blocks == []


def test_grammar_markdown(tmp_path):
    md = tmp_path / "README.md"
    md.write_text("# T\n\n<!-- snippet: setup\n"
                  "robot = 3\n-->\n\n```python\nassert robot == 3\n```\n\n"
                  "<!-- snippet: after\nassert robot == 3\n-->\n")
    mpage = ds.Page("README.md", md, "md")
    ds.parse_md(mpage)
    assert mpage.errors == []
    (blk,) = mpage.blocks
    assert blk.lineno == 8 and blk.setup[0].code == "robot = 3\n" and blk.after
    ds.run_group({"README.md": mpage}, "README.md", "main", cpu=True)


def test_grammar_page_pending_is_retired(tmp_path):
    """The transitional page marker of the documentation restructure is refused on every
    page kind, so a leftover one fails ``test_snippet_metadata``."""
    page = _rst(tmp_path, "p.rst", ".. snippet: page-pending\n\nT\n=\n\n"
                                   ".. code-block:: python\n\n   x = 1\n")
    assert any("page-pending is retired" in e for e in page.errors), page.errors
    assert [b.lineno for b in page.blocks] == [8]     # the block itself still parses
    md = tmp_path / "README.md"
    md.write_text("<!-- snippet: page-pending -->\n\n# T\n\n```python\nx = 1\n```\n")
    mpage = ds.Page("README.md", md, "md")
    ds.parse_md(mpage)
    assert any("page-pending is retired" in e for e in mpage.errors), mpage.errors
    src = tmp_path / "mod.py"
    src.write_text('"""Module.\n\nRuns::\n\n    z = 1 + 1\n"""\n# .. snippet: page-pending\n')
    ppage = ds.Page("mod.py", src, "py")
    ds.parse_py(ppage, {""})
    assert any(e.startswith("mod.py:7:") and "page-pending is retired" in e
               for e in ppage.errors), ppage.errors
    assert not any(hasattr(p, "pending") for p in (page, mpage, ppage))


def test_runner_restores_an_initialised_warp_config(tmp_path):
    """The ``wp.config`` snapshot is taken after ``wp.init()``: a run of ``run_group``
    outside pytest (no conftest ``wp.init()``) whose first group initialises Warp used to
    restore ``kernel_cache_dir`` to ``None``, and the next group then failed to load its
    kernels. Runs in a fresh interpreter, where Warp is not initialised yet."""
    import subprocess
    import sys
    page = tmp_path / "w.rst"
    kernel = ("   import warp as wp\n\n   @wp.kernel\n   def fill_{n}(a: wp.array(dtype=float)):\n"
              "       a[wp.tid()] = {n}.0\n\n   x = wp.zeros(4, dtype=float)\n"
              "   wp.launch(fill_{n}, dim=4, inputs=[x])\n"
              "   assert x.numpy().sum() == 4 * {n}.0\n")
    page.write_text("T\n=\n\n.. code-block:: python\n\n" + kernel.format(n=1) +
                    "\n.. snippet: group=second\n\n.. code-block:: python\n\n" +
                    kernel.format(n=2))
    script = (
        "import sys, warp as wp\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ds.ROOT)!r})\n"
        "from tests import _doc_snippets as ds\n"
        "assert wp.config.kernel_cache_dir is None, 'Warp must not be initialised yet'\n"
        f"page = ds.Page('w.rst', Path({str(page)!r}), 'rst')\n"
        "ds.parse_rst(page)\n"
        "assert page.errors == [], page.errors\n"
        "pages = {page.id: page}\n"
        "ds.run_group(pages, page.id, 'main')\n"
        "assert wp.config.kernel_cache_dir is not None\n"
        "ds.run_group(pages, page.id, 'second')\n"
        "ds.run_group(pages, page.id, 'main')\n"
        "print('OK')\n")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=300, cwd=tmp_path)
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


def test_runner_restores_linecache_of_docstring_pages(tmp_path):
    """A docstring group registers its synthetic source under the real module file: the
    previous linecache entry is restored afterwards, so later tracebacks and
    ``inspect.getsource`` of that module see the real source."""
    import linecache
    src = tmp_path / "docmod.py"
    src.write_text('"""Module.\n\nRuns::\n\n    z = 1 + 1\n"""\n\n\ndef f():\n    return 1\n')
    page = ds.Page("docmod.py", src, "py")
    ds.parse_py(page, {""})
    assert page.errors == [] and len(page.blocks) == 1
    real = linecache.getlines(str(src))
    assert "def f():\n" in real
    ds.run_group({page.id: page}, page.id, "module", cpu=True)
    assert linecache.getlines(str(src)) == real
    linecache.cache.pop(str(src), None)
    ds.run_group({page.id: page}, page.id, "module", cpu=True)
    assert str(src) not in linecache.cache


def test_runner_expect_warning_is_specific(tmp_path):
    """``expect-warning`` needs that category attributed to the example itself, and a
    deprecation of another category still fails the block."""
    helper = tmp_path / "docsnip_helper_mod.py"
    helper.write_text("import warnings\n\ndef noisy():\n"
                      "    warnings.warn('from the helper', UserWarning, stacklevel=1)\n")
    good = _rst(tmp_path, "w1.rst", "T\n=\n\n.. snippet: expect-warning=UserWarning\n\n"
                                    ".. code-block:: python\n\n   import warnings\n"
                                    "   warnings.warn('mine', UserWarning)\n")
    ds.run_group({good.id: good}, good.id, "main", cpu=True)
    other = _rst(tmp_path, "w2.rst", "T\n=\n\n.. snippet: expect-warning=UserWarning\n\n"
                                     ".. code-block:: python\n\n   import sys\n"
                                     f"   sys.path.insert(0, {str(tmp_path)!r})\n"
                                     "   import docsnip_helper_mod\n"
                                     "   docsnip_helper_mod.noisy()\n")
    with pytest.raises(AssertionError, match="attributed to the example"):
        ds.run_group({other.id: other}, other.id, "main", cpu=True)
    dep = _rst(tmp_path, "w3.rst", "T\n=\n\n.. snippet: expect-warning=UserWarning\n\n"
                                   ".. code-block:: python\n\n   import warnings\n"
                                   "   warnings.warn('mine', UserWarning)\n"
                                   "   warnings.warn('old', DeprecationWarning)\n")
    with pytest.raises(DeprecationWarning):
        ds.run_group({dep.id: dep}, dep.id, "main", cpu=True)


def test_runner_timeout_is_not_swallowed(tmp_path, monkeypatch):
    """The group timeout is a ``BaseException``: an ``except Exception`` in the example
    cannot swallow it."""
    monkeypatch.setattr(ds, "GROUP_TIMEOUT_S", 1.0)
    page = _rst(tmp_path, "t.rst", "T\n=\n\n.. code-block:: python\n\n   import time\n"
                                   "   t0 = time.time()\n   while time.time() - t0 < 20:\n"
                                   "       try:\n           time.sleep(0.05)\n"
                                   "       except Exception:\n           pass\n")
    import time
    t0 = time.time()
    with pytest.raises(ds.SnippetTimeout):
        ds.run_group({page.id: page}, page.id, "main", cpu=True)
    assert time.time() - t0 < 10
