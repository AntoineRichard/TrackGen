"""Local Sphinx checks for the TrackGen documentation.

Registered in ``docs/conf.py``. Three checks run inside every build, so ``sphinx -W``
fails on them locally and in CI:

``api``
    Every public symbol documented (the module lists live in
    ``trackgen_public_modules.py``): each public module and every name in its
    ``__all__`` is documented, by path or (classes and functions) by identity; every
    public class / function / Warp function, every public method / property /
    classmethod / staticmethod in a public class, and every public member a public class
    inherits from a private base has a docstring (the inherited ones must also be
    documented as ``<PublicClass>.<name>``); every non-underscore module under
    ``track_gen`` is classified public or internal, and every name in an internal
    module's ``__all__`` is re-exported by a public module. ``track_gen.testing.pytest_plugin``
    is read with ``ast``, never imported (the docs environment has no pytest).
    Exemptions: ``trackgen_api_exempt = {"qualified.name": "reason"}``.
``jargon``
    No roadmap / version jargon (``PR13b``, ``roadmap``, ``option C``, ``0.2.0.devN``,
    ``(preview)``, ``Preview API``) in any rendered page except ``upgrading/*``
    (``trackgen_jargon_exclude``), docstrings included; and no internal vocabulary
    (``self-graph``, ``plan bound``) outside the pages that define it
    (``trackgen_internal_terms_exclude``: the internals pages and ``reference/exec``).
    The highlighted source listings that ``sphinx.ext.viewcode`` writes
    (``_modules/**``) are not documents, so this check does not cover them: they show
    the library's source comments as they are. Their roadmap / version mentions are
    counted at the end of the build (INFO only, ``jargon-source`` lines in
    ``docscheck.txt``) and never fail it.
``nitpick``
    ``trackgen_nitpick_allowlist`` (``(type, target)`` pairs, read by ``docs/conf.py``
    from ``_tools/nitpick_allowlist.txt``) is applied here instead of ``nitpick_ignore``:
    an allowlisted unresolved reference is silenced and counted, and after a full build
    every entry that silenced nothing is reported as stale (remove it), so a target that
    starts resolving cannot keep masking a later break.
``data``
    Every ``literalinclude`` / ``include`` target and ``csv-table`` ``:file:`` exists,
    checked over every page on every build (an incremental build does not re-read an
    unchanged page, so docutils' own error would not fire again).

Severity: ``trackgen_docscheck = "report"`` logs the ``api`` and ``jargon`` findings at
INFO level (``docscheck: ...`` lines, also written to ``<doctreedir>/docscheck.txt``)
without failing the build; ``"error"`` turns them into warnings. ``data`` findings are
always warnings.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import types
import warnings
from pathlib import Path

from docutils import nodes
from docutils.utils import get_source_line
from sphinx.util import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import trackgen_public_modules as tpm  # noqa: E402

JARGON = re.compile(
    r"\bPR\d+[a-z]?\b|\broadmap\b|\boption C\b|0\.2\.0\.dev\d|\(preview\)|Preview API")
# Internal vocabulary user pages replace (the docs' vocabulary map): "self-graph" is the
# generator's own CUDA graph, "plan bound" the longest track the generator can produce.
INTERNAL_TERMS = re.compile(r"\bself-graph(s|ing)?\b|\bplan bound\b")
DATA_DIRECTIVE = re.compile(r"^\s*\.\. (literalinclude|include)::\s*(\S+)\s*$")
DATA_FILE_OPT = re.compile(r"^\s*:file:\s*(\S+)\s*$")

_FINDINGS: list[tuple[str, str, str]] = []


def _emit(app, kind: str, location: str | None, message: str, *, always_warn=False):
    _FINDINGS.append((kind, location or "", message))
    text = f"docscheck: {kind}: {message}"
    if always_warn or app.config.trackgen_docscheck == "error":
        logger.warning(text, location=location, type="docscheck", subtype=kind)
    else:
        logger.info(f"{location}: {text}" if location else text)


# ------------------------------------------------------------------------------------
# jargon
# ------------------------------------------------------------------------------------

def _jargon(app, doctree, docname):
    _RESOLVED.add(docname)
    patterns = []
    if not any(docname.startswith(p) for p in app.config.trackgen_jargon_exclude):
        patterns.append(JARGON)
    if not any(docname.startswith(p) for p in app.config.trackgen_internal_terms_exclude):
        patterns.append(INTERNAL_TERMS)
    if not patterns:
        return
    for node in doctree.findall(nodes.Text):
        text = node.astext()
        for pat in patterns:
            for m in pat.finditer(text):
                src, line = (get_source_line(node.parent) if node.parent is not None
                             else (None, None))
                loc = f"{src}:{line}" if src else None
                _emit(app, "jargon", loc or docname,
                      f"{m.group(0)!r} in {text.strip()[:90]!r}")


def _source_listings(app):
    """Count roadmap / version jargon in viewcode's ``_modules/**`` listings (report
    only: they render the library's source comments, which the check does not cover)."""
    root = Path(app.outdir) / "_modules"
    if not root.is_dir():
        return
    tag = re.compile(r"<[^>]+>")
    for path in sorted(root.rglob("*.html")):
        try:
            text = tag.sub("", path.read_text(encoding="utf-8"))
        except OSError:
            continue
        n = len(JARGON.findall(text))
        if n:
            _FINDINGS.append(("jargon-source", str(path.relative_to(app.outdir)),
                              f"{n} roadmap / version mentions in source comments"))


# ------------------------------------------------------------------------------------
# nitpick allowlist
# ------------------------------------------------------------------------------------

_RESOLVED: set = set()
_ALLOW_HITS: set = set()


def _allowlist(app) -> set:
    return {(str(a), str(b)) for a, b in app.config.trackgen_nitpick_allowlist}


def _warn_missing_reference(app, domain, node):
    typ, target = node.get("reftype"), node.get("reftarget")
    keys = [(f"{domain.name}:{typ}" if domain else typ, target)]
    if not domain or domain.name == "std":
        keys.append((typ, target))
    allow = _allowlist(app)
    for key in keys:
        if key in allow:
            _ALLOW_HITS.add(key)
            return True
    return None


def _stale_allowlist(app):
    """After a build that resolved every document (a full, serial build), report each
    allowlist entry that silenced nothing."""
    env = app.env
    if getattr(app, "parallel", 0) > 1 or not env.found_docs <= _RESOLVED:
        return
    for typ, target in sorted(_allowlist(app) - _ALLOW_HITS):
        _emit(app, "nitpick", None,
              f"stale allowlist entry {typ} {target}: the reference resolves (or is gone); "
              f"remove it from docs/_tools/nitpick_allowlist.txt")


# ------------------------------------------------------------------------------------
# data files
# ------------------------------------------------------------------------------------

def _data_files(app, env):
    srcdir = Path(app.srcdir)
    for docname in sorted(env.found_docs):
        path = Path(env.doc2path(docname))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            m = DATA_DIRECTIVE.match(line)
            target = m.group(2) if m else None
            if target is None:
                m = DATA_FILE_OPT.match(line)
                target = m.group(1) if m else None
            if target is None:
                continue
            full = srcdir / target.lstrip("/") if target.startswith("/") else path.parent / target
            if not full.exists():
                _emit(app, "data", f"{path}:{i}", f"missing data file {target!r}",
                      always_warn=True)


# ------------------------------------------------------------------------------------
# api: every public symbol documented
# ------------------------------------------------------------------------------------

def _import(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module(name)


def _resolve(path: str):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = _import(".".join(parts[:i]))
        except Exception:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for p in parts[i:]:
                    obj = getattr(obj, p)
            return obj
        except AttributeError:
            return None
    return None


def _is_private_module(modname: str) -> bool:
    return any(p.startswith("_") for p in modname.split("."))


def _own_doc(obj) -> str:
    """The docstring that documents ``obj`` (Warp functions: their Python function's;
    a dataclass's generated signature does not count)."""
    if type(obj).__module__.startswith("warp") and hasattr(obj, "func"):
        return (getattr(obj.func, "__doc__", None) or "").strip()
    if inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__")
        if doc and hasattr(obj, "__dataclass_fields__") and doc.startswith(obj.__name__ + "("):
            doc = None          # the signature dataclasses writes when there is none
        return (doc or "").strip()
    return (inspect.getdoc(obj) or "").strip()


def _member_fn(val):
    if isinstance(val, property):
        return val.fget
    if isinstance(val, (classmethod, staticmethod)):
        return val.__func__
    if inspect.isfunction(val):
        return val
    return None


def _walk_modules(pkgname: str):
    pkg = _import(pkgname)
    out = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        full = f"{pkgname}.{m.name}"
        out.append(full)
        if m.ispkg:
            out.extend(_walk_modules(full))
    return out


def _static_all(modname: str):
    """``(__all__, {name: docstring})`` of a module, read with ``ast`` (never imported)."""
    spec = importlib.util.find_spec(modname)
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    names, docs = [], {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                names = [e.value for e in node.value.elts]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docs[node.name] = ast.get_docstring(node) or ""
    return names, docs, ast.get_docstring(tree) or ""


def _api(app, env):
    exempt = app.config.trackgen_api_exempt
    py = env.get_domain("py")
    documented = {name: entry.objtype for name, entry in py.objects.items()}
    documented_modules = set(py.modules)

    by_id: dict[int, list[str]] = {}
    for name, typ in documented.items():
        if typ in ("class", "exception", "function", "method", "classmethod",
                   "staticmethod", "property", "attribute", "data"):
            obj = _resolve(name)
            if obj is not None and not isinstance(obj, (int, float, str, bool, type(None))):
                by_id.setdefault(id(obj), []).append(name)

    def report(q, msg):
        if q in exempt:
            return
        _emit(app, "api", None, f"{q}: {msg}")

    public = list(tpm.PUBLIC_MODULES)
    internal = list(tpm.INTERNAL_MODULES)
    static = set(tpm.STATIC_ONLY)

    # rule 4: classification
    for mod in _walk_modules("track_gen"):
        if mod not in public and mod not in internal:
            report(mod, "module is neither in PUBLIC_MODULES nor in INTERNAL_MODULES "
                        "(docs/_ext/trackgen_public_modules.py)")

    # every object a public module exports (by name), for the private-base and the
    # internal-module rules
    exports: dict[str, list[tuple[str, object]]] = {}
    exported_ids: set[int] = set()
    for m in public:
        if m in static:
            continue
        mod = _import(m)
        for n in getattr(mod, "__all__", ()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(mod, n)
            if not isinstance(obj, types.ModuleType):
                exports.setdefault(n, []).append((m, obj))
                exported_ids.add(id(obj))

    for m in public:
        if m in static:
            names, docs, mdoc = _static_all(m)
            if m not in documented_modules:
                report(m, "module not documented")
            if not mdoc:
                report(m, "module has no docstring")
            for n in names:
                q = f"{m}.{n}"
                if q not in documented:
                    report(q, "not documented (by path: this module is never imported)")
                if n in docs and not docs[n].strip():
                    report(q, "no docstring")
            continue
        mod = _import(m)
        if m not in documented_modules:
            report(m, "module not documented")
        if not (mod.__doc__ or "").strip():
            report(m, "module has no docstring")
        for n in getattr(mod, "__all__", ()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(mod, n)
            if isinstance(obj, types.ModuleType):
                continue
            q = f"{m}.{n}"
            code = inspect.isclass(obj) or inspect.isfunction(obj) or (
                callable(obj) and type(obj).__module__.startswith("warp"))
            if q not in documented and not (code and id(obj) in by_id):
                report(q, f"not documented ({type(obj).__name__}; by path"
                          f"{' or identity' if code else ''})")
            if code and not _own_doc(obj):
                report(q, "no docstring")
            if not inspect.isclass(obj):
                continue
            cls_paths = [q] + by_id.get(id(obj), [])
            for an, av in vars(obj).items():
                if an.startswith("_"):
                    continue
                fn = _member_fn(av)
                if fn is not None and not (inspect.getdoc(fn) or "").strip():
                    report(f"{q}.{an}", "no docstring")
            seen = set(vars(obj))
            for base in obj.__mro__[1:]:
                bm = getattr(base, "__module__", "") or ""
                if base is object or not bm.startswith("track_gen"):
                    continue
                # a base is private when its name is, or when it lives in a private
                # module and no public module exports it (a public base such as
                # tools.Component documents its own members)
                private = base.__name__.startswith("_") or (
                    _is_private_module(bm) and id(base) not in exported_ids
                    and id(base) not in by_id)
                for an, av in vars(base).items():
                    if an.startswith("_") or an in seen:
                        continue
                    seen.add(an)
                    fn = _member_fn(av)
                    if fn is None or not private:
                        continue
                    if not (inspect.getdoc(fn) or "").strip():
                        report(f"{q}.{an}", f"no docstring (inherited from {bm}.{base.__name__})")
                    if not any(f"{p}.{an}" in documented for p in cls_paths):
                        report(f"{q}.{an}", f"inherited from private {bm}.{base.__name__}: "
                                            f"document it as {q}.{an} (:inherited-members: "
                                            "or an explicit entry) or exempt it")

    for m in internal:
        mod = _import(m)
        for n in getattr(mod, "__all__", ()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(mod, n)
            if isinstance(obj, types.ModuleType):
                continue
            if any(o is obj for _, o in exports.get(n, ())):
                continue
            report(f"{m}.{n}", "in an internal module's __all__ but re-exported by no public "
                               "module: re-export it, or exempt it with a reason")


def _check(app, env):
    _FINDINGS[:] = [f for f in _FINDINGS if f[0] == "jargon"]
    _data_files(app, env)
    try:
        _api(app, env)
    except Exception as exc:  # an import failure must surface, not crash the build
        _emit(app, "api", None, f"the API check could not run: {type(exc).__name__}: {exc}",
              always_warn=True)


def _builder_inited(app):
    _FINDINGS.clear()
    _RESOLVED.clear()
    _ALLOW_HITS.clear()


def _build_finished(app, exc):
    if exc is not None:
        return
    _stale_allowlist(app)
    _source_listings(app)
    out = Path(app.doctreedir) / "docscheck.txt"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{k}\t{loc}\t{msg}\n" for k, loc, msg in _FINDINGS),
                       encoding="utf-8")
    except OSError:
        return
    counts = {}
    for k, _, _ in _FINDINGS:
        counts[k] = counts.get(k, 0) + 1
    if counts.get("jargon-source"):
        counts["jargon-source"] = f"{counts['jargon-source']} files (report only)"
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"
    logger.info(f"docscheck ({app.config.trackgen_docscheck}): {summary}; "
                f"details in {out}")


def setup(app):
    app.add_config_value("trackgen_docscheck", "error", "env", [str])
    app.add_config_value("trackgen_api_exempt", {}, "env", [dict])
    app.add_config_value("trackgen_jargon_exclude", ("upgrading/",), "env", [tuple, list])
    app.add_config_value("trackgen_internal_terms_exclude",
                         ("upgrading/", "how-it-works/", "relaxation/", "reference/exec"),
                         "env", [tuple, list])
    app.add_config_value("trackgen_nitpick_allowlist", [], "env", [list, tuple])
    app.connect("builder-inited", _builder_inited)
    app.connect("warn-missing-reference", _warn_missing_reference)
    app.connect("doctree-resolved", _jargon)
    app.connect("env-check-consistency", _check)
    app.connect("build-finished", _build_finished)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
