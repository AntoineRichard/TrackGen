"""The public / internal module classification of ``track_gen`` (one source of truth).

Read by ``docs/conf.py`` (the ``trackgen_docscheck`` extension: "every public symbol
documented") and by ``tests/_doc_snippets.py`` (docstring examples are extracted from the
public modules and from the files that define their ``__all__`` objects).

- ``PUBLIC_MODULES``: the module itself and every name in its ``__all__`` must be
  documented (by path, or by identity for classes and functions), with a non-empty
  docstring.
- ``INTERNAL_MODULES``: implementation submodules re-exported by their package. Every name
  in their ``__all__`` must be re-exported by a public module, or exempted in
  ``trackgen_api_exempt`` (``docs/conf.py``).
- Every non-underscore module under ``track_gen`` must be in one of the two lists.
- ``STATIC_ONLY``: public modules the docs build never imports (their ``__all__`` and
  docstrings are read with ``ast``): ``pytest_plugin`` imports pytest, which the docs
  environment does not install.

Plain data, no imports: safe to load from conf.py, the extension and the test runner.
"""

PUBLIC_MODULES = (
    "track_gen",
    "track_gen.results",
    "track_gen.seeds",
    "track_gen.exec",
    "track_gen.ops",
    "track_gen.ops.funcs",
    "track_gen.ops.constants",
    "track_gen.ops.rng",
    "track_gen.plugin",
    "track_gen.pipeline",
    "track_gen.pipeline.presets",
    "track_gen.pipeline.legacy",
    "track_gen.interop.torch",
    "track_gen.testing",
    "track_gen.testing.capture",
    "track_gen.testing.contract",
    "track_gen.testing.trace",
    "track_gen.testing.conformance",
    "track_gen.testing.components",
    "track_gen.testing.pytest_plugin",
    "track_gen.tools",
    "track_gen.collision",
    "track_gen.props",
    "track_gen.checkpoints",
    "track_gen.progress",
    "track_gen.localize",
    "track_gen.course",
    "track_gen.heightfield",
    "track_gen.stages.params",
    "track_gen.stages.randomize",
    "track_gen.stages.external",
)

INTERNAL_MODULES = (
    "track_gen.exec.context",
    "track_gen.exec.errors",
    "track_gen.exec.host",
    "track_gen.exec.runner",
    "track_gen.interop",
    "track_gen.ops.check",
    "track_gen.ops.curves",
    "track_gen.ops.fill",
    "track_gen.ops.geometry",
    "track_gen.ops.hash",
    "track_gen.ops.reduce",
    "track_gen.ops.resample",
    "track_gen.ops.sampling",
    "track_gen.pipeline.ingest",
    "track_gen.pipeline.planner",
    "track_gen.pipeline.producer",
    "track_gen.pipeline.program",
    "track_gen.pipeline.spec",
    "track_gen.plugin.params",
    "track_gen.plugin.stage",
    "track_gen.stages",
    "track_gen.stages.channels",
    "track_gen.stages.commit",
    "track_gen.stages.gates",
    "track_gen.stages.track",
)

STATIC_ONLY = ("track_gen.testing.pytest_plugin",)
