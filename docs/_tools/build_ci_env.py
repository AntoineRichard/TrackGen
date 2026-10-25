#!/usr/bin/env python3
"""Run sphinx-build as the docs CI environment would: without pytest, torch, scipy or
matplotlib, on this checkout.

The docs workflow installs ``.[docs]`` only (sphinx and the theme extensions), so an
autodoc import that needs one of those packages, or a docs check that imports
``track_gen.testing.pytest_plugin``, fails there and nowhere else. This wrapper blocks
those packages (``sys.modules[name] = None``: ``import`` raises
``ModuleNotFoundError`` and ``importlib.util.find_spec`` returns ``None``, as when they
are not installed), puts this checkout's root first on ``sys.path`` (as
``python -m sphinx`` run from the root does in docs.yml, so autodoc and the docs checks
read this tree, not whichever ``track_gen`` the environment installed), prints where
``track_gen`` resolves, then calls ``sphinx.cmd.build.main`` with the arguments it was
given::

    python docs/_tools/build_ci_env.py -W --keep-going -E -b html \\
        -d <private doctrees> docs <private outdir>

Always pass ``-E`` with a private outdir and doctree dir: an incremental build does not
re-read unchanged pages and can pass falsely.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BLOCKED = ("pytest", "_pytest", "torch", "scipy", "matplotlib")
ROOT = Path(__file__).resolve().parents[2]


def block_packages(modules=None, names=BLOCKED) -> None:
    """Make ``names`` (top-level packages) unimportable in ``modules`` (default
    ``sys.modules``): drop any loaded submodule and set the package to ``None``."""
    modules = sys.modules if modules is None else modules
    for mod in list(modules):
        if mod.split(".")[0] in names and mod not in names:
            del modules[mod]
    for name in names:
        modules[name] = None


def use_checkout(path=None) -> str:
    """Put this checkout's root first on ``sys.path``; return where ``track_gen``
    resolves (its ``__init__.py``) without importing it."""
    root = str(ROOT)
    path = sys.path if path is None else path
    if root in path:
        path.remove(root)
    path.insert(0, root)
    spec = importlib.util.find_spec("track_gen")
    return spec.origin if spec is not None else "<not found>"


def main(argv=None) -> int:
    block_packages()
    origin = use_checkout()
    print(f"build_ci_env: track_gen from {origin}", file=sys.stderr)
    if not origin.endswith(".py") or Path(origin).resolve().parents[1] != ROOT:
        print(f"build_ci_env: WARNING: track_gen does not come from this checkout "
              f"({ROOT})", file=sys.stderr)
    from sphinx.cmd.build import main as sphinx_main
    return sphinx_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
