"""Public extension surface: register track and gate generators, and relax / width /
elevation / validity stages of the track pipeline, without editing track_gen.

A pip-installable package adds generators through an entry point in the
``"track_gen.plugins"`` group (:data:`ENTRY_POINT_GROUP`) whose target is a
``register(registry)`` callable. In the plugin package's ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."track_gen.plugins"]
    acme = "acme_tracks:register"

and in ``acme_tracks/__init__.py`` (``Superformula`` is the package's generator, a
:class:`Stage` subclass; a :class:`GeneratorSpec` registers the same way):

.. snippet: skip=plugin-package

.. code-block:: python

    from acme_tracks.superformula import Superformula

    track_gen_api = 1     # optional: the plugin API version this package targets

    def register(reg):
        reg.add(Superformula)     # the class, not an instance

after which ``TrackGenConfig(generator="acme.superformula")`` selects it. Names:
third-party names should be dotted (``"package.name"``); a non-dotted one registers with
a one-time ``UserWarning``. Code that registers at run time (scripts, tests) calls
:data:`registry`'s :meth:`~Registry.add` directly; ``registry.scoped()`` snapshots and
restores the registry around a test (``scoped(fresh=True)`` also re-runs discovery).

Loading. ``import track_gen`` loads nothing. The registry fills itself lazily on its
first call (``get``, ``available``, ``describe``, ``add``, ``unregister``,
``quarantined``, ``scoped()``, and the deprecated ``track_gen._src`` registry shims),
once per process and under a lock:

1. the built-ins, registered explicitly by ``track_gen._src.builtin_plugins.register``
   (no entry point needed, so a source checkout without metadata works);
2. third-party plugins: every entry point of the group, sorted by (distribution name,
   entry-point name), each loaded and called with the registry.

Policy. The ``TRACK_GEN_PLUGINS`` environment variable (:data:`POLICY_ENV`, read when
the registry loads, case-insensitive) selects what happens to a third-party plugin that
fails to load: ``warn`` (default: it is quarantined and one ``RuntimeWarning`` names
it), ``strict`` (raise :class:`PluginLoadError` at load, e.g. for CI) or ``off`` (skip
third-party discovery entirely). Any other value emits a ``RuntimeWarning`` and falls
back to ``warn``. A plugin fails to load when importing or calling it raises, when it
declares a newer plugin API than :data:`API_VERSION` (``track_gen_api = <int>`` in the
module its entry point names, or on the register callable), when it registers a name
that already exists without ``replace=True`` (built-ins always win unless a plugin
explicitly overrides them), or when it unregisters a name it did not register.

Parameters and stages. A generator declares its own parameters as a frozen
:class:`Params` dataclass (fields marked :func:`static` or :func:`dynamic`) and users
select them with ``TrackGenConfig(generator=..., generator_params=...)`` (an instance or
a mapping of its fields). A simple :class:`GeneratorSpec` takes a static Params class
through ``params=``; a :class:`Stage` subclass (``kind="generator"``,
``output="centerline"``, registered as a class) allocates through a
:class:`StageBuilder` and may declare dynamic, per-env params that callers write
through ``gen.params["generator"]`` without re-capturing a graph.
:meth:`Registry.describe` reports each spec's Params JSON schema
(:attr:`SpecInfo.params_schema`).

Pipeline stages. A :class:`Stage` of kind ``"relax"``, ``"width"``,
``"elevation"`` or ``"validity"`` (registered as a class, like a generator Stage)
replaces or extends a stage of the track pipeline: select it with
:func:`track_gen.pipeline.presets.track` (``relax="acme.taubin"``,
``checks=("default", "acme.max_curvature")``, ...), or with the flat shortcuts
``TrackGenConfig(relax_solver=...)`` / ``TrackGenConfig(z_profile=...)``. TrackGen's own
stages of these kinds (``relax.xpbd``, ``elevation.noise``, ``validity.default``, ...)
are registered with the built-in generators. :func:`track_gen.testing.check_stage` runs
the conformance checks on one.

Quarantine. A quarantined plugin's registrations are rolled back and it is listed in
:attr:`Registry.quarantined`; an unknown-name error lists the quarantined plugins with
their errors. Asking for a name the plugin tried to register, or a name namespaced under
its entry-point name, entry-point module or distribution name (``acme.x`` for an entry
point named ``acme``), raises :class:`PluginLoadError` chained from its load error.
"""
from ._specs import GeneratorSpec
from ._registry import (
    API_VERSION,
    ENTRY_POINT_GROUP,
    POLICY_ENV,
    DuplicateSpecError,
    PluginLoadError,
    PluginOverrideWarning,
    Registry,
    SpecInfo,
    registry,
)
from .params import NoParams, ParamTable, Params, StageParams, dynamic, static
from .stage import CaptureClass, Layout, Stage, StageBuilder

__all__ = [
    "API_VERSION",
    "ENTRY_POINT_GROUP",
    "POLICY_ENV",
    "CaptureClass",
    "DuplicateSpecError",
    "GeneratorSpec",
    "Layout",
    "NoParams",
    "ParamTable",
    "Params",
    "PluginLoadError",
    "PluginOverrideWarning",
    "Registry",
    "SpecInfo",
    "Stage",
    "StageBuilder",
    "StageParams",
    "dynamic",
    "registry",
    "static",
]
