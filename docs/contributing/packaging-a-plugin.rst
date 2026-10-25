.. _packaging-a-plugin:

Packaging a Plugin
==================

.. admonition:: Before you start

   - **What you write:** a pip-installable package with one entry point in the
     ``track_gen.plugins`` group, and the ``register(reg)`` function it names.
   - **What you need to know:** the plugins your package ships: track generators
     (:ref:`writing-a-generator`), gate generators (:ref:`writing-a-gate-generator`) and
     pipeline :term:`stages <stage>` (:ref:`writing-a-stage`). Runtime
     :term:`components <component>` are not registered: a :term:`course` takes them as
     objects (:ref:`writing-a-component`).
   - **How you test it:** run your CI with ``TRACK_GEN_PLUGINS=strict``, wrap registry
     edits in ``registry.scoped()``, and run the conformance kit on every name you
     register (:ref:`testing-a-plugin`).

A plugin is an ordinary Python package. It imports only TrackGen's public modules, never
edits TrackGen, and is found through a Python entry point when it is installed. The
shipped generators register through the same kind of ``register(reg)`` function as a
third party's, so a plugin is held to the rules the built-ins follow. The registry's
classes and functions are listed in :ref:`reference-plugin`.

The package
-----------

A plugin package holds its plugin classes and one ``register(reg)`` function:

.. code-block:: text

   acme-tracks/
     pyproject.toml
     acme_tracks/
       __init__.py      # register(reg) and track_gen_api
       wobble.py        # a track generator
       gates.py         # a gate generator
       taubin.py        # a relax stage

**The entry point** names the ``register`` function, in the plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."track_gen.plugins"]
   acme = "acme_tracks:register"

The entry-point name (``acme``) identifies the plugin in warnings and in
``registry.quarantined``. Name it after the prefix of your names (``acme`` for
``acme.*``): a request for one of its names then reports the plugin's load error if it
ever fails to load (:ref:`plugin-quarantine`).

**The** ``register`` **function** receives the registry and adds every plugin of the
package, whatever its kind:

.. snippet: skip=plugin-package

.. code-block:: python

   # acme_tracks/__init__.py
   from acme_tracks.gates import Scatter
   from acme_tracks.taubin import Taubin
   from acme_tracks.wobble import Wobble

   track_gen_api = 1  # the plugin API version this package targets


   def register(reg):
       reg.add(Wobble)    # kind "generator", read from the class
       reg.add(Scatter)   # kind "gate_generator"
       reg.add(Taubin)    # kind "relax"

It adds classes (a ``Stage`` subclass) or spec objects (a ``GeneratorSpec``,
:ref:`plugin-abis`), never instances of a Stage. Inside ``register(reg)`` a plugin may
only add names and unregister names it added itself. Install the package (an editable
install is enough: ``pip install -e ./acme-tracks``) and every name it registers is
selectable, ``presets.track(generator="acme.wobble", relax="acme.taubin", ...)``, with no
import in the user's code.

**Importing a module does not register it.** The registry calls your ``register(reg)``
when it loads; TrackGen's own generator modules do not register themselves when
imported either. Without an install (a notebook, a quick experiment), call it yourself:
``acme_tracks.register(track_gen.plugin.registry)``.

**The plugin API version.** ``track_gen.plugin.API_VERSION`` (currently ``1``) is the
version of this plugin interface. A plugin declares the version it targets by setting
``track_gen_api = 1`` in the module its entry point names (``acme_tracks`` for
``"acme_tracks:register"``), or as an attribute of the register function. A plugin that
declares a newer version than the installed TrackGen's is quarantined with a message
saying so, instead of failing later in a less obvious way. An integer ``API_VERSION`` in
the same places is read too, as a fallback; a non-integer one (an unrelated constant) is
ignored. A ``Stage`` also carries an ``api_version`` class attribute (default ``1``),
checked when it is added.

The registry
------------

``track_gen.plugin.registry`` is the process-wide ``Registry``. It keys specs by *kind*
and name: ``"generator"`` (a ``Stage`` class or a ``GeneratorSpec``),
``"gate_generator"`` (a ``Stage`` class) and the
pipeline stage kinds of :ref:`writing-a-stage` (``"relax"``, ``"width"``,
``"elevation"``, ``"validity"``, ``"order"``, ``"stations"``). The set of kinds is
open: later releases add more.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Call
     - What it does
   * - ``registry.add(spec, *, kind=None, replace=False)``
     - Register ``spec`` (``reg.add`` inside ``register(reg)``). The kind is read from a
       ``Stage`` class (its ``kind``) or a ``GeneratorSpec``;
       any other spec type needs ``kind=``. A Stage's declaration and its Params class
       are checked first (:ref:`generator-stage-contract`, :ref:`generator-params`,
       :ref:`stage-ports-channels`).
   * - ``registry.get(kind, name)``
     - The spec (for a Stage, the class). An unknown name raises ``ValueError`` listing
       the available names and the quarantined plugins with their errors.
   * - ``registry.available(kind=None)``
     - The sorted names of one kind, or every ``"kind:name"`` when ``kind`` is ``None``.
   * - ``registry.describe(kind, name)``
     - A ``SpecInfo`` (`Describing what is installed`_).
   * - ``registry.unregister(kind, name)``
     - Remove a spec. Inside its ``register(reg)``, a plugin may only unregister names it
       registered itself.
   * - ``registry.scoped(*, fresh=False)``
     - Context manager: snapshot every kind on entry and restore it on exit (for
       tests). ``fresh=True`` starts the scope from an empty, unloaded registry, so the
       next call re-runs built-in and entry-point discovery.
   * - ``registry.quarantined``
     - Read-only mapping of plugin entry-point name to the error that quarantined it.
       When two quarantined plugins share an entry-point name, their keys are qualified
       as ``"<distribution>:<entry point>"``.

With the Stage quick start's ``Wobble`` class (:ref:`writing-a-generator`):

.. snippet: group=registry requires=contributing/writing-a-generator#wobble

.. code-block:: python

   from track_gen.plugin import registry

   with registry.scoped():                    # restored on exit: nothing leaks
       registry.add(Wobble)
       names = registry.available("generator")    # [..., 'acme.wobble', 'bezier', ...]
       info = registry.describe("generator", "acme.wobble")
       info.capture, dict(info.writes)            # STATIC, {'xy': 'raw.xy', ...}

.. snippet: after

   assert "acme.wobble" in names and "bezier" in names
   assert info.distribution is None and info.entry_point is None and not info.builtin

.. _plugin-names:

Names
-----

Third-party names are dotted, ``<package>.<name>`` (``acme.wobble``), which keeps plugins
from colliding with each other and with built-ins, present or future. A third-party name
without a dot still registers, with a one-time ``UserWarning``. Built-in names
(``bezier``, ``polar``, ``xpbd``, ...) stay undotted.

.. snippet: group=undotted requires=contributing/writing-a-generator#spec expect-warning=UserWarning

.. code-block:: python

   import dataclasses

   from track_gen.plugin import registry

   with registry.scoped():
       spec = registry.get("generator", "acme.wobbly")
       registry.add(dataclasses.replace(spec, name="wobbly"))   # UserWarning: not dotted

A name is part of your users' code and data: it is the ``generator=`` (or ``relax=``,
``order=``, ...) string in their configs and in the specs they save
(:ref:`generating-saving`). Renaming a plugin breaks
both; register the old name too, as a second ``reg.add`` of a renamed copy, for as long
as you support it.

Duplicates and overrides
------------------------

Adding a name that already exists in that kind raises ``DuplicateSpecError`` (a
``ValueError``) naming both origins: the module of each spec and, for a plugin, the
distribution that registered it. Nothing is overwritten silently. To override on
purpose, pass ``replace=True``: the new spec replaces the old one and a
``PluginOverrideWarning`` (a ``UserWarning``) names both origins. Adding the identical
spec object again is a no-op, so re-running ``registry.add(SPEC)`` (a notebook cell) is
safe.

.. snippet: group=override requires=contributing/writing-a-generator#wobble expect-warning=PluginOverrideWarning

.. code-block:: python

   from track_gen.plugin import registry


   class LoudWobble(Wobble):        # the same name, "acme.wobble"
       pass


   with registry.scoped():
       registry.add(LoudWobble, replace=True)   # PluginOverrideWarning names both origins

Built-ins register first, so a plugin that reuses a built-in name without
``replace=True`` never replaces it: that plugin is quarantined (or raises, under
``strict``) and the built-in stays. A plugin cannot get around this by unregistering the
built-in first: inside ``register(reg)``, unregistering a name the plugin did not add
quarantines it too.

Loading
-------

The registry fills itself lazily, on the first registry call (``get``, ``available``,
``describe``, ``add``, ``unregister``, ``quarantined``, ``scoped()``, or one of the
public registry methods): in practice when the first
``TrackGenerator``, ``GateGenerator`` or ``Course`` is constructed. Calling
``acme_tracks.register(registry)`` by hand therefore also discovers the installed entry
points first. ``import track_gen`` imports no generator module and loads no plugin.
Loading runs once per process, under a lock, so concurrent first lookups load everything
exactly once:

1. **Built-ins.** TrackGen's own ``register(reg)`` imports the shipped generator and
   stage modules and adds them. Built-ins need no entry point, so a source checkout
   without installed metadata works.
2. **Third parties.** Every entry point of the ``track_gen.plugins`` group, sorted by
   distribution name and then entry-point name, is loaded and called with the registry.
   The order does not depend on install order.

A generator resolves its spec once, at construction. Registering, replacing or
unregistering a name later never changes a live ``TrackGenerator``, ``GateGenerator`` or
``Course``.

.. _plugin-quarantine:

Broken plugins: quarantine and ``TRACK_GEN_PLUGINS``
----------------------------------------------------

One broken plugin must not take the library down. The environment variable
``TRACK_GEN_PLUGINS``, read when the registry loads, sets what happens to a plugin that
fails to import, raises inside ``register``, registers a name that is already taken
without ``replace=True``, unregisters a name it did not register, fails a declaration
check, or declares a newer plugin API than this TrackGen supports. The value is
case-insensitive and stripped; any other value emits a ``RuntimeWarning`` and falls back
to ``warn``, so the built-ins keep working:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Behaviour
   * - ``warn`` (default)
     - The plugin is **quarantined**: one ``RuntimeWarning``, an entry in
       ``registry.quarantined``, and every other plugin still loads.
   * - ``strict``
     - Loading raises ``PluginLoadError``. Use it in CI.
   * - ``off``
     - Third-party discovery is skipped: only the built-ins are registered.

A lookup for an unknown name lists the quarantined plugins and their errors next to the
available names. Asking for what a quarantined plugin provides (e.g.
``presets.track(generator="acme.wobble", ...)`` when ``acme`` failed to load) raises
``PluginLoadError`` chained from the plugin's original error, not a bare "unknown
generator". A plugin that fails before it registers anything cannot say which names it
provides, so the registry matches by namespace: a name raises ``PluginLoadError`` when it
is the plugin's entry-point name or entry-point module, is namespaced under one of them
(``acme.wobble`` for the entry point ``acme`` or the module ``acme``), or its first
dotted component is the plugin's distribution name (normalized, ``acme_tracks.x`` for
``acme-tracks``). Any other name raises the plain ``ValueError``. Name the entry point
after your names' prefix so this matching works. The same holds for a stage: a
``relax_solver`` or ``z_profile`` that names a stage of a quarantined plugin raises its
``PluginLoadError``.

.. code-block:: bash

   TRACK_GEN_PLUGINS=strict python -m pytest -q    # a load failure fails the run

Describing what is installed
----------------------------

``registry.describe(kind, name)`` returns a ``SpecInfo``: what the registry knows about
a spec without building anything. Tools read it instead of a spec's attributes (a Stage
class has no ``capturable`` or ``alloc_scratch``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``kind``, ``name``
     - The registry key.
   * - ``origin``
     - The module that defines the spec.
   * - ``distribution``
     - ``(name, version)`` of the package whose entry point registered it; ``None`` for
       a built-in and for a spec added at run time.
   * - ``entry_point``
     - The name of the entry point that registered it, or ``None``.
   * - ``builtin``
     - ``True`` for the specs shipped with TrackGen.
   * - ``capturable``, ``capture``
     - Whether a caller can record it into a CUDA graph, and its ``CaptureClass``, both
       for the Params defaults (:ref:`stage-capture-classes`). A ``GeneratorSpec``
       reports its ``capturable`` and no ``capture``.
   * - ``accepts_ctx``
     - Always ``True`` for a Stage or a ``GeneratorSpec``
       (:ref:`generator-streams`).
   * - ``output``, ``reads``, ``writes``, ``provides``
     - A Stage's ports and the attributes it provides (``declare()`` at the Params
       defaults for a centerline generator); ``None`` and empty for a ``GeneratorSpec``.
   * - ``orderings``, ``excluded_orderings``, ``preferred``
     - A gate generator's accepted orders and preferred stages; a centerline generator's
       ``preferred`` relax (:ref:`gate-generator-stage`).
   * - ``supported_orderings``
     - A gate generator Stage's accepted orders, resolved against the registry.
   * - ``seeded``
     - Whether the output depends on the :term:`seed` (``True`` for a ``GeneratorSpec``):
       code that enumerates generators to check seed dependence filters on it.
   * - ``params_schema``
     - The JSON schema of its Params class, ``{}`` without one
       (:ref:`generator-describe`).

``registry.available("generator")`` lists every registered generator, ``external``
included (it ingests your centerlines and draws nothing). To pick generators at random
per run, keep the seeded ones:

.. snippet: group=seeded

.. code-block:: python

   from track_gen.plugin import registry

   seeded = [n for n in registry.available("generator")
             if registry.describe("generator", n).seeded]

.. snippet: after

   assert "external" not in seeded and "bezier" in seeded

Testing a package
-----------------

- Run the plugin's CI with ``TRACK_GEN_PLUGINS=strict``, so a plugin that fails to load
  fails the run instead of being quarantined with a warning.
- Wrap any test that adds, replaces or removes specs in ``registry.scoped()``: it
  snapshots every kind on entry and restores it on exit, so nothing leaks into the next
  test.
- Check that the entry point, not a test helper, registered your names: an installed
  spec has a ``distribution``.
- Run the conformance kit on every name you register (:ref:`testing-a-plugin`); it looks
  names up in the registry, so it runs on the installed package as your users get it.

.. snippet: group=package-tests

.. code-block:: python

   from track_gen.plugin import registry
   from track_gen.testing import check_generator


   def test_the_entry_point_registers_acme():
       info = registry.describe("generator", "acme.wobble")
       assert info.distribution is not None and info.entry_point == "acme"
       assert not registry.quarantined


   def test_acme_wobble_conforms():
       check_generator("acme.wobble").raise_for_failures()
