"""Executed documentation examples: extraction, metadata grammar and runner.

Every ``.. code-block:: python`` of every page under ``docs/`` (except the excluded
directories), every ```` ```python ```` fence of ``README.md``, and every ``::`` literal
block / ``.. code-block:: python`` in the docstrings of the public modules (and of the
files that define their ``__all__`` objects) runs in the CPU test lane, so the examples
cannot go stale. ``tests/test_doc_snippets.py`` turns them into tests; this module has no
pytest dependency (``python tests/_doc_snippets.py`` prints the per-page report).

A plain line parser: the CPU lane has no docutils.

Metadata grammar
----------------

An RST comment ``.. snippet: <tokens>`` (README: ``<!-- snippet: <tokens> -->``; in a
docstring, the same RST comment inside the docstring). Consecutive snippet comments
(blank lines between them allowed) form a cluster attached to the next python block at
the same indentation (for a ``::`` block in a docstring: the paragraph that introduces
it), with only blank lines in between; ``after`` instead follows its block.

============================  =====================  ===================================
Token                         Scope                  Meaning
============================  =====================  ===================================
``group=<name>``              this block only        run the block in namespace
                                                     ``<name>``; unannotated blocks run in
                                                     the page's default group ``main``; a
                                                     group's blocks run in page order
``requires=<page>#<group>,…`` group (first block)    before the group runs, execute those
                                                     groups (their ``register=`` and
                                                     ``after`` included) into this group's
                                                     namespace, in order; recursive
``register=<callable>``       block                  after the block, call
                                                     ``<callable>(track_gen.plugin.registry)``
                                                     (inside the group's registry scope)
``setup``                     the next block         the comment's indented body is hidden
                                                     code run right before the block
                                                     (only inputs the prose states)
``after``                     the previous block     placed after a block; its indented
                                                     body runs right after it (asserts)
``cuda`` / ``slow``           group (first block)    ``cuda``: GPU lane only; ``slow``:
                                                     not in the fast lane
``skip=<reason>``             block                  not run; reason in fragment,
                                                     signature, pseudo, legacy,
                                                     plugin-package, output
``expect-warning=<Category>`` block                  the block must emit that warning
                                                     (or a subclass), attributed to the
                                                     block; other deprecations still fail
``expect-error=<Exception>``  block                  the block must raise it (the root
                                                     classes are refused)
============================  =====================  ===================================

In a docstring every docstring is its own group (``group=`` and ``requires=`` are
refused). There is no page-level marker: every runnable block of every page runs, and a
block that cannot run carries ``skip=<reason>`` (the per-page report lists them). The
transitional ``page-pending`` marker of the documentation restructure is refused
(``RETIRED_TOKENS``), so a leftover one fails ``test_snippet_metadata``.

Runner rules: each (page, group) runs in a real ``types.ModuleType`` named
``docsnip_<page-slug>__<group>`` registered in ``sys.modules`` for the run; each piece is
compiled with ``dont_inherit=True`` under the page's path at its real line numbers, the
page's synthetic source registered in ``linecache`` (Warp reads kernel source from it);
inside ``track_gen.plugin.registry.scoped()``; ``os.environ`` and the public ``wp.config``
attributes restored afterwards. CPU lane: inside ``wp.ScopedDevice("cpu")``, every string
constant matching ``^cuda(:\\d+)?$`` rewritten to ``"cpu"``. A ``DeprecationWarning`` /
``FutureWarning`` attributed to snippet code is an error (``expect-warning`` overrides it
for its block). A block over 10 s on CPU without ``slow`` emits a
:class:`SnippetSlowWarning`; a group is stopped after 120 s (600 s with ``slow``).
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import faulthandler
import importlib
import inspect
import linecache
import os
import re
import signal
import sys
import textwrap
import time
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"
EXCLUDED_DIRS = {"_build", "_ext", "_tools", "_data", "_static", "superpowers",
                 "library-readiness"}
SKIP_REASONS = {"fragment", "signature", "pseudo", "legacy", "plugin-package", "output"}
GROUP_ONLY = {"cuda", "slow", "requires"}
VALUE_TOKENS = {"group", "requires", "register", "skip", "expect-warning", "expect-error"}
FLAG_TOKENS = {"setup", "after", "cuda", "slow"}
# Tokens that existed once and are refused with a pointed message.
RETIRED_TOKENS = {
    "page-pending": "page-pending is retired (it marked pages whose examples were being "
                    "fixed during the documentation restructure): make the page's examples "
                    "pass, or mark a block that cannot run with skip=<reason>",
}
GROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CUDA_LITERAL = re.compile(r"^cuda(:\d+)?$")
SLOW_BLOCK_S = 10.0
GROUP_TIMEOUT_S = 120.0
SLOW_GROUP_TIMEOUT_S = 600.0

_RST_META = re.compile(r"^(?P<ind>[ \t]*)\.\. snippet:(?P<rest>.*)$")
_RST_CODE = re.compile(r"^(?P<ind>[ \t]*)\.\. code-block::[ \t]*python[ \t]*$")
_ADORN = re.compile(r"^([=\-~^\"'`:.*+#])\1{2,}\s*$")
_DIRECTIVE = re.compile(r"^\s*(\.\. [\w:-]+::|:[\w -]+:)")
_MD_META = re.compile(r"^<!--\s*snippet:(?P<rest>.*?)(?P<close>-->)?\s*$")
_MD_FENCE = re.compile(r"^```python\s*$")
# Code blocks that Sphinx / GitHub render as Python but that are not the one form the
# extractor runs: refused (a metadata error), so no Python example goes unchecked.
PY_LANGS = {"", "python", "python3", "py", "py3", "pycon", "ipython", "ipython3", "default"}
_RST_ANY_CODE = re.compile(r"^(?P<ind>[ \t]*)\.\. (?P<dir>code-block|code|sourcecode)::"
                           r"[ \t]*(?P<lang>\S*)[ \t]*$")
_MD_ANY_FENCE = re.compile(r"^(?P<ind>[ \t]*)```[ \t]*(?P<lang>[\w+-]*)")
_MD_PY_LANGS = PY_LANGS - {"", "default"}
# A skipped block that uses a CUDA-only API belongs in a ``cuda`` group (GPU lane).
_CUDA_ONLY_API = re.compile(r"torch\.cuda\b|\bwp\.Stream\(|\bstream_from_torch\b")
_PY_RETIRED = re.compile(r"^\s*#\s*\.\.\s*snippet:\s*(?P<tok>page-pending)\b")


class SnippetSlowWarning(UserWarning):
    """A documentation example ran longer than the fast-lane budget on CPU."""


class SnippetTimeout(BaseException):
    """A documentation example group exceeded its hard time limit.

    A ``BaseException`` (like ``KeyboardInterrupt``), so an ``except Exception:`` in the
    example, or in library code running when the alarm fires, cannot swallow it."""


@dataclasses.dataclass
class Piece:
    lineno: int          # 1-based line of the piece's first code line in its file
    code: str            # dedented source


@dataclasses.dataclass
class Block:
    page: str
    lineno: int
    code: str
    tokens: dict
    meta_lineno: int = 0
    setup: list = dataclasses.field(default_factory=list)
    after: list = dataclasses.field(default_factory=list)

    @property
    def group(self) -> str:
        return self.tokens.get("group", "main")

    @property
    def skipped(self) -> bool:
        return "skip" in self.tokens


@dataclasses.dataclass
class Group:
    page: str
    name: str
    blocks: list

    @property
    def first(self) -> Block:
        return self.blocks[0]

    @property
    def cuda(self) -> bool:
        return "cuda" in self.first.tokens

    @property
    def slow(self) -> bool:
        return "slow" in self.first.tokens

    @property
    def requires(self) -> list[tuple[str, str]]:
        raw = self.first.tokens.get("requires")
        if not raw:
            return []
        out = []
        for item in raw.split(","):
            page, _, group = item.partition("#")
            out.append((normalize_page(page), group or "main"))
        return out

    @property
    def runnable(self) -> list[Block]:
        return [b for b in self.blocks if not b.skipped]


@dataclasses.dataclass
class Page:
    id: str               # "tutorials/batch-of-tracks.rst", "README.md", "track_gen/x.py"
    path: Path
    kind: str             # "rst" | "md" | "py"
    blocks: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)
    # py pages: one group per docstring, named after its owner
    py_groups: dict = dataclasses.field(default_factory=dict)

    def groups(self) -> dict[str, Group]:
        out: dict[str, Group] = {}
        for b in self.blocks:
            out.setdefault(b.group, Group(self.id, b.group, [])).blocks.append(b)
        return out

    def error(self, lineno: int, msg: str):
        self.errors.append(f"{self.id}:{lineno}: {msg}")


def normalize_page(page: str) -> str:
    """A ``requires=`` page reference: a docs-relative path (``.rst`` optional),
    ``README.md``, or a repository-relative ``.py`` path."""
    page = page.strip()
    if page in ("README", "README.md"):
        return "README.md"
    if page.endswith((".py", ".rst")):
        return page
    return page + ".rst"


def parse_tokens(rest: str) -> dict:
    toks = {}
    for t in rest.split():
        k, eq, v = t.partition("=")
        toks[k] = v if eq else True
    return toks


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _body(lines: list[str], start: int, ind: int):
    """The block of lines after ``start`` indented deeper than ``ind`` (blank lines inside
    kept). Returns (first body line index or None, dedented text, index after the body)."""
    j = start
    last = None
    first = None
    while j < len(lines):
        ln = lines[j]
        if ln.strip():
            if _indent(ln) <= ind:
                break
            if first is None:
                first = j
            last = j
        j += 1
    if first is None:
        return None, "", start
    text = textwrap.dedent("".join(l if l.endswith("\n") else l + "\n"
                                   for l in lines[first:last + 1]))
    return first, text, last + 1


# --------------------------------------------------------------------------------------
# metadata checks shared by every page kind
# --------------------------------------------------------------------------------------

def _check_tokens(page: Page, lineno: int, toks: dict, *, docstring=False):
    for k, v in toks.items():
        if k in RETIRED_TOKENS:
            page.error(lineno, RETIRED_TOKENS[k])
        elif k not in VALUE_TOKENS and k not in FLAG_TOKENS:
            page.error(lineno, f"unknown snippet token {k!r}")
        elif k in VALUE_TOKENS and v is True:
            page.error(lineno, f"snippet token {k!r} needs a value ({k}=...)")
        elif k in FLAG_TOKENS and v is not True:
            page.error(lineno, f"snippet token {k!r} takes no value")
    if "skip" in toks and toks["skip"] is not True and toks["skip"] not in SKIP_REASONS:
        page.error(lineno, f"skip reason {toks['skip']!r} not in {sorted(SKIP_REASONS)}")
    if "group" in toks and toks["group"] is not True:
        if docstring:
            page.error(lineno, "group= is not allowed in a docstring (one group per docstring)")
        elif not GROUP_NAME.match(toks["group"]):
            page.error(lineno, f"bad group name {toks['group']!r}")
    if docstring and "requires" in toks:
        page.error(lineno, "requires= is not allowed in a docstring")
    for k in ("register", "expect-warning", "expect-error"):
        if k in toks and toks[k] is not True and not re.match(r"^[A-Za-z_][\w.]*$", toks[k]):
            page.error(lineno, f"{k}= needs a dotted name, got {toks[k]!r}")
    for k, roots in (("expect-error", ("Exception", "BaseException")),
                     ("expect-warning", ("Warning",))):
        v = toks.get(k)
        if isinstance(v, str) and v.split(".")[-1] in roots:
            page.error(lineno, f"{k}={v} would accept anything: name the specific class")


def _finish(page: Page):
    """Group-level rules once every block of the page is known."""
    for name, g in page.groups().items():
        for b in g.blocks[1:]:
            for k in GROUP_ONLY:
                if k in b.tokens:
                    page.error(b.meta_lineno or b.lineno,
                               f"{k!r} is a group token: only on the first block of group "
                               f"{name!r}")
        reg_names = [b.tokens["register"] for b in g.blocks if "register" in b.tokens]
        if reg_names:
            defined = set()
            for b in g.blocks:
                for piece in [Piece(b.lineno, b.code), *b.setup, *b.after]:
                    try:
                        tree = ast.parse(piece.code)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                             ast.ClassDef)):
                            defined.add(node.name)
                        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                            defined.add(node.id)
            for r in reg_names:
                if r.split(".")[0] not in defined:
                    page.error(g.first.lineno, f"register={r} names no callable defined by "
                                               f"group {name!r}")
        for b in g.blocks:
            if b.skipped and _CUDA_ONLY_API.search(b.code):
                page.error(b.meta_lineno or b.lineno,
                           "a skipped block that uses a CUDA-only API (torch.cuda, wp.Stream, "
                           "stream_from_torch): make it a 'cuda' group so the GPU lane runs it")
        if not g.cuda:
            for b in g.blocks:
                for piece in [Piece(b.lineno, b.code), *b.setup, *b.after]:
                    try:
                        tree = ast.parse(piece.code)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, ast.JoinedStr) and any(
                                isinstance(v, ast.Constant) and isinstance(v.value, str)
                                and "cuda" in v.value for v in node.values):
                            page.error(piece.lineno + node.lineno - 1,
                                       "an f-string in a CPU group contains 'cuda' (the CPU "
                                       "lane cannot rewrite it)")


# --------------------------------------------------------------------------------------
# RST pages (and docstrings, which share the grammar)
# --------------------------------------------------------------------------------------

def _parse_rst_lines(page: Page, lines: list[str], line0: int, *, docstring=False,
                     group: str | None = None):
    """Parse RST ``lines`` whose first line is file line ``line0 + 1``."""
    cluster: list = []          # (index, indent, tokens, setup piece, end index)
    last_block: Block | None = None
    last_block_end = -1
    last_block_ind = -1
    i = 0
    n = len(lines)

    def L(idx):        # 1-based file line of lines[idx]
        return line0 + idx + 1

    def dangling(idx):
        if cluster:
            page.error(L(cluster[0][0]), "snippet comment not followed by a python block "
                                         "at its indentation (only blank lines between)")
            cluster.clear()

    while i < n:
        line = lines[i].rstrip("\n")
        m_meta = _RST_META.match(line)
        m_code = _RST_CODE.match(line)
        if not m_code:
            m_any = _RST_ANY_CODE.match(line)
            if m_any and m_any.group("lang").lower() in PY_LANGS:
                page.error(L(i), f"'.. {m_any.group('dir')}:: {m_any.group('lang')}' renders "
                                 f"as Python but the snippet check does not run it: write "
                                 f"'.. code-block:: python' (or skip=<reason>), or name the "
                                 f"real language")
        code_start = None
        lit_ind = None
        if docstring and not m_meta and not m_code and line.rstrip().endswith("::") \
                and not line.lstrip().startswith(".."):
            lit_ind = _indent(line)
        if m_meta:
            ind = len(m_meta.group("ind"))
            toks = parse_tokens(m_meta.group("rest"))
            _check_tokens(page, L(i), toks, docstring=docstring)
            # layout: the previous line
            prev = lines[i - 1].rstrip("\n") if i > 0 else ""
            if i > 0 and prev.strip() and not _ADORN.match(prev) and not (
                    _DIRECTIVE.match(prev) and _indent(prev) < ind) and not _RST_META.match(prev):
                page.error(L(i), "the line before a snippet comment must be blank, a heading "
                                 "underline or a directive line at lower indentation (docutils "
                                 "would render the comment as text)")
            first, body, end = _body(lines, i + 1, ind)
            if end < n and lines[end].strip() and not _RST_CODE.match(lines[end].rstrip("\n")) \
                    and not _RST_META.match(lines[end].rstrip("\n")):
                page.error(L(end), "the line after a snippet comment must be blank or the "
                                   "code-block itself")
            if set(toks) & set(RETIRED_TOKENS):
                i = end         # reported by _check_tokens; attaches to nothing
                continue
            if "after" in toks:
                rest = {k: v for k, v in toks.items() if k != "after"}
                if rest:
                    page.error(L(i), "an 'after' comment takes no other token")
                between = [l for l in lines[last_block_end:i] if l.strip()] \
                    if last_block is not None else ["x"]
                if last_block is None or between or last_block_ind != ind:
                    page.error(L(i), "an 'after' comment must follow its python block (same "
                                     "indentation, only blank lines between)")
                elif first is None:
                    page.error(L(i), "an 'after' comment needs an indented body")
                else:
                    last_block.after.append(Piece(L(first), body))
                    last_block_end = end
                i = end
                continue
            if cluster and (cluster[-1][1] != ind or any(
                    l.strip() for l in lines[cluster[-1][4]:i])):
                dangling(i)
            setup = None
            if "setup" in toks:
                if first is None:
                    page.error(L(i), "a 'setup' comment needs an indented body")
                else:
                    setup = Piece(L(first), body)
            elif first is not None:
                page.error(L(i), "only 'setup' and 'after' comments have a body")
            cluster.append((i, ind, toks, setup, end))
            i = end
            continue
        if m_code or lit_ind is not None:
            if m_code:
                ind = len(m_code.group("ind"))
                j = i + 1
                while j < n and lines[j].strip() and _indent(lines[j]) > ind \
                        and lines[j].strip().startswith(":"):
                    j += 1
                anchor = i
            else:
                ind = lit_ind
                j = i + 1
                # the cluster attaches at the introducing paragraph's first line
                anchor = i
                while anchor > 0 and lines[anchor - 1].strip() and \
                        _indent(lines[anchor - 1]) == ind:
                    anchor -= 1
            first, code, end = _body(lines, j, ind)
            if first is None:
                i += 1
                continue
            toks: dict = {}
            setups: list[Piece] = []
            meta_line = 0
            if cluster:
                ok = cluster[-1][1] == ind and not any(
                    l.strip() for l in lines[cluster[-1][4]:anchor])
                if ok:
                    for (ci, _, ctoks, csetup, _) in cluster:
                        meta_line = meta_line or L(ci)
                        t = dict(ctoks)
                        t.pop("setup", None)
                        for k, v in t.items():
                            if k in toks and toks[k] != v:
                                page.error(L(ci), f"token {k!r} given twice in one cluster")
                            toks[k] = v
                        if csetup is not None:
                            setups.append(csetup)
                    cluster.clear()
                else:
                    dangling(i)
            if docstring:
                toks.setdefault("group", group)
            blk = Block(page.id, L(first), code, toks, meta_line, setups)
            page.blocks.append(blk)
            last_block, last_block_end, last_block_ind = blk, end, ind
            i = end
            continue
        i += 1
    dangling(n)


def parse_rst(page: Page):
    lines = page.path.read_text(encoding="utf-8").splitlines(keepends=True)
    _parse_rst_lines(page, lines, 0)
    _finish(page)


# --------------------------------------------------------------------------------------
# README (Markdown)
# --------------------------------------------------------------------------------------

def parse_md(page: Page):
    lines = page.path.read_text(encoding="utf-8").splitlines(keepends=True)
    cluster: list[tuple[int, dict, Piece | None, int]] = []
    last_block, last_end = None, -1
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        m = _MD_META.match(line)
        if m:
            toks = parse_tokens(m.group("rest"))
            _check_tokens(page, i + 1, toks)
            body_lines, end = [], i + 1
            if not m.group("close"):
                j = i + 1
                while j < n and lines[j].strip() != "-->":
                    body_lines.append(lines[j])
                    j += 1
                if j >= n:
                    page.error(i + 1, "unterminated <!-- snippet: comment")
                end = j + 1
            body = textwrap.dedent("".join(body_lines))
            if set(toks) & set(RETIRED_TOKENS):
                pass            # reported by _check_tokens; attaches to nothing
            elif "after" in toks:
                if last_block is None or any(l.strip() for l in lines[last_end:i]):
                    page.error(i + 1, "an 'after' comment must follow its python block")
                elif not body.strip():
                    page.error(i + 1, "an 'after' comment needs a body")
                else:
                    last_block.after.append(Piece(i + 2, body))
                    last_end = end
            else:
                if cluster and any(l.strip() for l in lines[cluster[-1][3]:i]):
                    page.error(cluster[0][0] + 1, "snippet comment not followed by a python "
                                                  "block")
                    cluster.clear()
                setup = Piece(i + 2, body) if "setup" in toks and body.strip() else None
                if "setup" in toks and setup is None:
                    page.error(i + 1, "a 'setup' comment needs a body")
                cluster.append((i, toks, setup, end))
            i = end
            continue
        m_any = _MD_ANY_FENCE.match(line)
        if m_any and not _MD_FENCE.match(line) and m_any.group("lang").lower() in _MD_PY_LANGS:
            page.error(i + 1, "a Python fence the snippet check does not run: write a "
                              "column-0 ```python fence (or skip=<reason>)")
        if _MD_FENCE.match(line):
            j = i + 1
            while j < n and not lines[j].startswith("```"):
                j += 1
            code = "".join(lines[i + 1:j])
            toks, setups, meta_line = {}, [], 0
            if cluster:
                if not any(l.strip() for l in lines[cluster[-1][3]:i]):
                    for (ci, ctoks, csetup, _) in cluster:
                        meta_line = meta_line or ci + 1
                        t = dict(ctoks)
                        t.pop("setup", None)
                        toks.update(t)
                        if csetup is not None:
                            setups.append(csetup)
                else:
                    page.error(cluster[0][0] + 1, "snippet comment not followed by a python "
                                                  "block")
                cluster.clear()
            blk = Block(page.id, i + 2, code, toks, meta_line, setups)
            page.blocks.append(blk)
            last_block, last_end = blk, j + 1
            i = j + 1
            continue
        i += 1
    if cluster:
        page.error(cluster[0][0] + 1, "snippet comment not followed by a python block")
    _finish(page)


# --------------------------------------------------------------------------------------
# docstrings
# --------------------------------------------------------------------------------------

def _public_modules():
    sys.path.insert(0, str(DOCS / "_ext"))
    try:
        import trackgen_public_modules as tpm
    finally:
        sys.path.pop(0)
    return tpm


def docstring_targets() -> dict[Path, set[str]]:
    """{source file: qualnames whose docstrings are checked} ('' = the module docstring).
    The public modules' own docstrings, and the docstrings of every class / function in
    a public ``__all__`` and of the public methods of those classes."""
    tpm = _public_modules()
    out: dict[Path, set[str]] = {}
    for m in tpm.PUBLIC_MODULES:
        if m in tpm.STATIC_ONLY:
            spec = importlib.util.find_spec(m)
            out.setdefault(Path(spec.origin), set()).add("")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = importlib.import_module(m)
        f = getattr(mod, "__file__", None)
        if f:
            out.setdefault(Path(f), set()).add("")
        for n in getattr(mod, "__all__", ()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(mod, n)
            obj = getattr(obj, "func", obj) if type(obj).__module__.startswith("warp") else obj
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            try:
                src = inspect.getsourcefile(obj)
            except TypeError:
                continue
            if not src or not str(Path(src).resolve()).startswith(str(ROOT / "track_gen")):
                continue
            qn = obj.__qualname__
            s = out.setdefault(Path(src), set())
            s.add(qn)
            if inspect.isclass(obj):
                for an, av in vars(obj).items():
                    if not an.startswith("_"):
                        s.add(f"{qn}.{an}")
    return {Path(k).resolve(): v for k, v in out.items()}


def parse_py(page: Page, qualnames: set[str]):
    text = page.path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        m = _PY_RETIRED.match(line)
        if m:
            page.error(i, RETIRED_TOKENS[m.group("tok")])
    tree = ast.parse(text)

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = f"{prefix}{child.name}"
                if qn in qualnames:
                    yield qn, child
                if isinstance(child, ast.ClassDef):
                    yield from visit(child, qn + ".")

    owners = []
    if "" in qualnames:
        owners.append(("module", tree))
    owners.extend(visit(tree, ""))
    for qn, node in owners:
        body = getattr(node, "body", [])
        if not body or not isinstance(body[0], ast.Expr) or not isinstance(
                getattr(body[0], "value", None), ast.Constant) or not isinstance(
                body[0].value.value, str):
            continue
        const = body[0].value
        raw = const.value.splitlines(keepends=True)
        # the docstring's first raw line sits on the line of the opening quotes
        _parse_rst_lines(page, raw, const.lineno - 1, docstring=True, group=qn)
    _finish(page)


# --------------------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------------------

def rst_pages() -> list[Path]:
    out = []
    for p in sorted(DOCS.rglob("*.rst")):
        rel = p.relative_to(DOCS)
        if rel.parts[0] in EXCLUDED_DIRS:
            continue
        out.append(p)
    return out


_CACHE: dict[str, Page] | None = None


def collect(*, docstrings: bool = True) -> dict[str, Page]:
    """Every page with its blocks and metadata errors, keyed by page id."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    pages: dict[str, Page] = {}
    for p in rst_pages():
        page = Page(p.relative_to(DOCS).as_posix(), p, "rst")
        parse_rst(page)
        pages[page.id] = page
    if README.exists():
        page = Page("README.md", README, "md")
        parse_md(page)
        pages[page.id] = page
    if docstrings:
        for path, qns in sorted(docstring_targets().items()):
            page = Page(path.relative_to(ROOT).as_posix(), path, "py")
            parse_py(page, qns)
            if page.blocks or page.errors:
                pages[page.id] = page
    # cross-page rules: requires= targets exist, no cycles
    for page in pages.values():
        for g in page.groups().values():
            for (rp, rg) in g.requires:
                if rp not in pages or rg not in pages[rp].groups():
                    page.error(g.first.lineno, f"requires={rp}#{rg}: no such page / group")
    for page in pages.values():
        for g in page.groups().values():
            try:
                resolve_requires(pages, page.id, g.name)
            except ValueError as e:
                page.error(g.first.lineno, str(e))
            except KeyError:
                pass
    _CACHE = pages
    return pages


def resolve_requires(pages, page_id: str, group: str, _stack=()) -> list[tuple[str, str]]:
    """The (page, group) pairs to execute before ``group``, in order (deduplicated)."""
    key = (page_id, group)
    if key in _stack:
        raise ValueError(f"requires= cycle: {' -> '.join(f'{p}#{g}' for p, g in _stack + (key,))}")
    g = pages[page_id].groups()[group]
    out: list[tuple[str, str]] = []
    for dep in g.requires:
        for d in resolve_requires(pages, *dep, _stack=_stack + (key,)) + [dep]:
            if d not in out:
                out.append(d)
    return out


def metadata_errors(pages: dict[str, Page] | None = None) -> list[str]:
    pages = pages or collect()
    return [e for p in pages.values() for e in p.errors]


def report(pages: dict[str, Page] | None = None) -> str:
    """Per page: blocks, run / skipped (by reason), groups (and how many are cuda)."""
    pages = pages or collect()
    rows = []
    for pid, p in sorted(pages.items()):
        if not p.blocks:
            continue
        reasons: dict[str, int] = {}
        for b in p.blocks:
            if b.skipped:
                reasons[b.tokens["skip"]] = reasons.get(b.tokens["skip"], 0) + 1
        nskip = sum(reasons.values())
        groups = p.groups()
        ncuda = sum(g.cuda for g in groups.values())
        flags = f"cuda-groups={ncuda} " if ncuda else ""
        why = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
        rows.append(f"{pid}: {len(p.blocks)} blocks, {len(p.blocks) - nskip} run, "
                    f"{nskip} skipped ({nskip / len(p.blocks):.0%}{': ' + why if why else ''}), "
                    f"{len(groups)} groups {flags}".rstrip())
    nblocks = sum(len(p.blocks) for p in pages.values())
    nskip = sum(b.skipped for p in pages.values() for b in p.blocks)
    rows.append(f"total: {nblocks} blocks, {nblocks - nskip} run, {nskip} skipped")
    return "\n".join(rows)


# --------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------

class _CudaToCpu(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and CUDA_LITERAL.match(node.value):
            return ast.copy_location(ast.Constant("cpu"), node)
        return node

    def visit_JoinedStr(self, node):     # f-string parts are not rewritten
        return node


def _slug(s: str) -> str:
    return re.sub(r"\W", "_", s)


_ABSENT = object()


def _register_source(page: Page):
    """The page's synthetic source (every block and hidden body at its real line, blank
    elsewhere) in linecache under the page's path: Warp reads kernel source from it and
    tracebacks show the right lines."""
    n = len(page.path.read_text(encoding="utf-8").splitlines()) + 1
    src = ["\n"] * n
    for b in page.blocks:
        for piece in [Piece(b.lineno, b.code), *b.setup, *b.after]:
            for k, line in enumerate(piece.code.splitlines(keepends=True)):
                idx = piece.lineno - 1 + k
                if idx < len(src):
                    src[idx] = line if line.endswith("\n") else line + "\n"
    fn = str(page.path)
    linecache.cache[fn] = (sum(map(len, src)), None, src, fn)


def _compile(piece: Piece, filename: str, cpu: bool):
    # padded, so line numbers (SyntaxErrors included) are the file's
    tree = ast.parse("\n" * (piece.lineno - 1) + piece.code, filename=filename)
    if cpu:
        tree = _CudaToCpu().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", flags=0, dont_inherit=True)


def _resolve_name(name: str, ns: dict):
    import builtins
    if name in ns:
        return ns[name]
    if hasattr(builtins, name):
        return getattr(builtins, name)
    head, _, rest = name.partition(".")
    obj = ns.get(head) or (importlib.import_module(head) if head else None)
    for part in rest.split(".") if rest else []:
        obj = getattr(obj, part)
    return obj


def _warning_category(name: str, ns: dict):
    """The warning class an ``expect-warning=`` token names: a builtin, a (dotted) name
    the group's namespace resolves, or a warning class exported by ``track_gen``,
    ``track_gen.plugin`` or ``track_gen.exec``; ``None`` if none is found."""
    import builtins
    last = name.split(".")[-1]
    cands = [getattr(builtins, last, None)]
    try:
        cands.append(_resolve_name(name, ns))
    except Exception:  # noqa: BLE001 - not resolvable from the namespace
        pass
    for modname in ("track_gen", "track_gen.plugin", "track_gen.exec"):
        cands.append(getattr(importlib.import_module(modname), last, None))
    for c in cands:
        if isinstance(c, type) and issubclass(c, Warning) and c.__name__ == last:
            return c
    return None


def _mro_names(cls) -> set[str]:
    return {c.__name__ for c in getattr(cls, "__mro__", ())} | {
        f"{c.__module__}.{c.__qualname__}" for c in getattr(cls, "__mro__", ())}


@contextlib.contextmanager
def _time_limit(seconds: float):
    if threading_main() and hasattr(signal, "setitimer"):
        def _raise(signum, frame):
            raise SnippetTimeout(f"documentation example group exceeded {seconds:.0f} s")
        old = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        faulthandler.dump_traceback_later(seconds + 30, exit=True)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
            faulthandler.cancel_dump_traceback_later()
    else:
        yield


def threading_main() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


@contextlib.contextmanager
def _restored_state():
    import warp as wp
    # Initialise first: wp.init() fills wp.config (kernel_cache_dir, ...), so a snapshot
    # taken before it would restore those to None and break every later group of a run
    # that does not initialise Warp up front (pytest's conftest does; a standalone run of
    # run_group may not).
    wp.init()
    env = dict(os.environ)
    cfg = {k: v for k, v in vars(wp.config).items()
           if not k.startswith("_") and not callable(v) and not isinstance(v, types.ModuleType)}
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env)
        for k, v in cfg.items():
            setattr(wp.config, k, v)


def _run_piece(piece: Piece, filename: str, ns: dict, cpu: bool, *, block: Block | None,
               timings: list | None):
    expect_w = block.tokens.get("expect-warning") if block else None
    expect_e = block.tokens.get("expect-error") if block else None
    expect_cat = None
    if expect_w:
        expect_cat = _warning_category(expect_w, ns)
        if expect_cat is None:
            raise AssertionError(f"{filename}:{piece.lineno}: expect-warning={expect_w}: no "
                                 f"such warning class (use a dotted name)")
    t0 = time.perf_counter()
    code = _compile(piece, filename, cpu)
    raised = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=DeprecationWarning, module=r"^docsnip_")
        warnings.filterwarnings("error", category=FutureWarning, module=r"^docsnip_")
        if expect_cat is not None:
            # the expected category (and only it) is recorded instead of raised; every
            # other deprecation attributed to the example still fails it
            warnings.filterwarnings("always", category=expect_cat)
        try:
            exec(code, ns)
        except SnippetTimeout:
            raise
        except BaseException as exc:  # noqa: BLE001 - matched against expect-error
            if expect_e and expect_e.split(".")[-1] in _mro_names(type(exc)) | {
                    n.split(".")[-1] for n in _mro_names(type(exc))}:
                raised = exc
            else:
                raise
    dt = time.perf_counter() - t0
    if expect_e and raised is None:
        raise AssertionError(f"{filename}:{piece.lineno}: expected {expect_e} to be raised")
    if expect_cat is not None and not any(
            issubclass(w.category, expect_cat) and w.filename == filename for w in rec):
        raise AssertionError(
            f"{filename}:{piece.lineno}: expected a {expect_w} warning attributed to the "
            f"example; got {[(w.category.__name__, w.filename, w.lineno) for w in rec]}")
    if timings is not None:
        timings.append((filename, piece.lineno, dt))
    return dt


def run_group(pages: dict[str, Page], page_id: str, group: str, *, cpu: bool = True,
              timings: list | None = None):
    """Execute one (page, group), its ``requires=`` first, as described in the module
    docstring. Raises the first failure."""
    import warp as wp
    from track_gen.plugin import registry

    page = pages[page_id]
    g = page.groups()[group]
    order = resolve_requires(pages, page_id, group) + [(page_id, group)]
    name = f"docsnip_{_slug(page_id)}__{_slug(group)}"
    module = types.ModuleType(name)
    module.__file__ = str(page.path)
    ns = module.__dict__
    sys.modules[name] = module
    limit = SLOW_GROUP_TIMEOUT_S if g.slow else GROUP_TIMEOUT_S
    # The synthetic sources replace linecache's entry for each page's file (for a
    # docstring page: the real library module) while the group runs: restore the
    # previous entries afterwards, so later tracebacks and inspect.getsource() of that
    # module see its real source. Warp has read each kernel's source at decoration.
    saved_lines = {}
    for pid, _ in order:
        fn = str(pages[pid].path)
        saved_lines.setdefault(fn, linecache.cache.get(fn, _ABSENT))
    try:
        with _restored_state(), registry.scoped(), _time_limit(limit), \
                (wp.ScopedDevice("cpu") if cpu else contextlib.nullcontext()):
            os.environ.setdefault("MPLBACKEND", "Agg")
            for pid, gname in order:
                p = pages[pid]
                _register_source(p)
                fn = str(p.path)
                for b in p.groups()[gname].runnable:
                    for s in b.setup:
                        _run_piece(s, fn, ns, cpu, block=None, timings=None)
                    dt = _run_piece(Piece(b.lineno, b.code), fn, ns, cpu, block=b,
                                    timings=timings)
                    if "register" in b.tokens:
                        _resolve_name(b.tokens["register"], ns)(registry)
                    for a in b.after:
                        _run_piece(a, fn, ns, cpu, block=None, timings=None)
                    if cpu and dt > SLOW_BLOCK_S and not g.slow:
                        warnings.warn(SnippetSlowWarning(
                            f"{pid}:{b.lineno} took {dt:.1f} s on CPU (budget "
                            f"{SLOW_BLOCK_S:.0f} s): shrink it or mark its group 'slow'"))
    finally:
        sys.modules.pop(name, None)
        for fn, entry in saved_lines.items():
            if entry is _ABSENT:
                linecache.cache.pop(fn, None)
            else:
                linecache.cache[fn] = entry


if __name__ == "__main__":
    pages = collect()
    print(report(pages))
    errs = metadata_errors(pages)
    if errs:
        print(f"\n{len(errs)} metadata errors:")
        print("\n".join(errs))
        sys.exit(1)
