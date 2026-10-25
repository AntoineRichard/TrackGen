.. _api-reference:

API reference
=============

Every name documented in this reference is public API. Breaking changes and their
replacements are listed in the upgrade guides. Private names (a leading underscore,
anything under ``track_gen._src``) and the modules this reference does not list may
change in any release.

The reference is split by module. This page holds the module map, the top-level
``track_gen`` names, every exception and warning the library raises, and the environment
variables it reads. The configuration classes and the built-in Params are described in
:ref:`configuration-reference`.

Module map
----------

Importing any ``track_gen`` module first imports the ``track_gen`` package itself, so the
last column lists only what a module loads beyond ``import track_gen``.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Module
     - Reference page
     - What importing it loads
   * - ``track_gen``
     - this page (:ref:`api-top-level`)
     - Warp, the generators' facades, the result types, ``track_gen.exec``,
       ``track_gen.ops``, ``track_gen.tools`` and the runtime utilities. It loads no
       generator, no plugin, no pipeline module and no torch: the plugin registry loads
       on its first lookup, and a generator's modules load when it is built.
   * - ``track_gen.results``
     - :ref:`reference-results`
     - Nothing more.
   * - ``track_gen.pipeline``, ``track_gen.pipeline.presets``,
       ``track_gen.pipeline.legacy``
     - :ref:`reference-pipeline`
     - The planner and ``track_gen.plugin``.
   * - ``track_gen.stages.params``, ``track_gen.stages.randomize``,
       ``track_gen.stages.external``
     - :ref:`reference-pipeline`
     - ``track_gen.plugin``.
   * - ``track_gen.plugin``
     - :ref:`reference-plugin`
     - The registry module. The registry itself fills on its first lookup: the
       built-ins, then every installed entry point.
   * - ``track_gen.ops``, ``track_gen.ops.funcs``, ``track_gen.ops.constants``
     - :ref:`reference-ops`
     - Nothing more.
   * - ``track_gen.exec``, ``track_gen.seeds``
     - :ref:`reference-exec`
     - Nothing more.
   * - ``track_gen.interop.torch``
     - :ref:`reference-exec`
     - Nothing more: torch is imported when one of its functions is called.
   * - ``track_gen.course``, ``track_gen.tools``, ``track_gen.collision``,
       ``track_gen.props``, ``track_gen.checkpoints``, ``track_gen.progress``,
       ``track_gen.localize``, ``track_gen.heightfield``
     - :ref:`reference-runtime`
     - Nothing more.
   * - ``track_gen.testing`` and its submodules ``capture``, ``contract``, ``trace``,
       ``conformance``, ``components``
     - :ref:`reference-testing`
     - The conformance kit, without torch or pytest (the torch capture modes import
       torch when they run).
   * - ``track_gen.testing.pytest_plugin``
     - :ref:`reference-testing`
     - pytest.

``track_gen.pipeline``, ``track_gen.plugin``, ``track_gen.stages``,
``track_gen.interop`` and ``track_gen.testing`` are not in ``track_gen.__all__``: import
them explicitly.

.. _api-top-level:

Top-level names
---------------

.. automodule:: track_gen
   :no-members:

``import track_gen`` gives:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Name
     - Documented
   * - ``TrackGenerator``, ``GateGenerator``, ``PerEnvSeededRNG``, ``__version__``
     - Below.
   * - ``TrackGenConfig``, ``GateGenConfig``
     - :ref:`config-classes`.
   * - ``Track``, ``GateSequence``, ``Layout`` and the protocols ``HasBand``,
       ``HasCenterline``, ``HasCheckpoints``, ``HasGateFrames``, ``HasGates``,
       ``HasGeneration``, ``HasValidity``
     - The same objects as in ``track_gen.results``: :ref:`reference-results`.
   * - The submodules ``results``, ``tools``, ``collision``, ``props``,
       ``checkpoints``, ``progress``, ``localize``, ``course``, ``heightfield``
     - :ref:`reference-results` and :ref:`reference-runtime`.

.. py:data:: __version__
   :module: track_gen
   :type: str

   The TrackGen version string, set in ``track_gen/_version.py`` (the version the
   package was built as).

Generators
~~~~~~~~~~

A generator builds a batch of :term:`E` tracks or gate sequences, one per
:term:`env`, from a :term:`preset`, a :term:`spec`, or a configuration; see
:ref:`generating` for how to choose.

.. autoclass:: track_gen.TrackGenerator
   :members:

.. autoclass:: track_gen.GateGenerator
   :members:

Seeds
~~~~~

Each env's :term:`seed`, with its :term:`epoch`, determines its output; the generators
keep both in a :class:`~track_gen.seeds.SeedState` (:ref:`reference-exec`).

.. autoclass:: track_gen.PerEnvSeededRNG
   :members:

.. _api-errors:

Errors and warnings
-------------------

The exceptions and warnings TrackGen defines, with the usual cause and fix. Besides
these, the library raises ``TypeError`` / ``ValueError`` for a bad argument (naming it),
and emits ``DeprecationWarning`` for a deprecated name (attributed to your code).

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Class (base)
     - Raised when
     - What to do
   * - :class:`~track_gen.exec.CaptureError` (``RuntimeError``)
     - A call is made in a :term:`capture` state it does not support: building a
       generator, a course or a utility while its stream (or torch, or a CPU capture) is
       capturing, since construction runs a real generation; calling a course built with
       ``stream=`` while another stream captures; recording a producer built with
       ``prepare=False`` before its warm-up; ``interop.torch.registered()`` outside
       ``torch.cuda.graph``; a result's ``clone()`` / ``alloc()``, or
       ``load_centerlines``, inside a capture.
     - Build everything before the capture and record only calls; pass ``stream=``
       (the capturing stream) to a call you record. See :ref:`capture-troubleshooting`.
       All the capture errors below derive from it.
   * - :class:`~track_gen.exec.NotCapturableError` (``CaptureError``)
     - A producer that cannot be recorded (a pipeline with ``HOST`` stages, a generator
       registered with ``capturable=False``) is called inside a caller's capture. It is
       raised before any device work, so your capture is left untouched.
     - Call it outside the capture (it then runs eagerly), or use stages that can be
       recorded; ``PipelineSpec(require_caller_capturable=True)`` rejects ``HOST`` stages
       when the spec is built. See :ref:`capture-troubleshooting`.
   * - :class:`~track_gen.pipeline.PlanError` (``ValueError``)
     - :func:`~track_gen.pipeline.build` (hence ``TrackGenerator(spec)`` /
       ``GateGenerator(spec)``) refuses a spec: a :term:`channel` read before any
       :term:`stage` writes it, a randomizer on a static or unknown field, gate counts
       that do not fit the generator's capacity, and the other planner rules.
     - The message names the stage and the rule. See :ref:`pipeline-stages` and
       :ref:`config-randomize`.
   * - :class:`~track_gen.plugin.DuplicateSpecError` (``ValueError``)
     - ``registry.add`` is given a ``(kind, name)`` that is already registered, without
       ``replace=True``. The message names both origins.
     - Choose another (dotted) name, or pass ``replace=True`` on purpose; in tests,
       register inside ``registry.scoped()``. See :ref:`packaging-a-plugin`.
   * - :class:`~track_gen.plugin.PluginLoadError` (``RuntimeError``)
     - A third-party plugin failed to load: when the registry loads under
       ``TRACK_GEN_PLUGINS=strict``, and on a lookup of a name the quarantined plugin
       provides.
     - Read the chained cause (the plugin's import or ``register`` error); see
       :ref:`plugin-quarantine` and :ref:`api-environment`.
   * - :class:`~track_gen.testing.HotPathViolation` (``AssertionError``)
     - A guard of :mod:`track_gen.testing.contract` caught a host sync, an allocation, a
       launch without ``stream=`` or a nested capture in the guarded block.
     - Fix the call the message locates; see :ref:`testing-a-plugin`.
   * - :class:`~track_gen.exec.RebindAfterCaptureWarning` (``UserWarning``)
     - A bound input array is replaced after a method that reads it was recorded into a
       capture: the recorded graph keeps reading the old arrays.
     - Write new values into the bound arrays instead; rebind only before recording,
       then record again (filter the warning when you re-record on purpose). See
       :ref:`capture-troubleshooting`.
   * - :class:`~track_gen.plugin.PluginOverrideWarning` (``UserWarning``)
     - ``registry.add(..., replace=True)`` replaced an existing spec. The message names
       both origins.
     - Nothing, if the override is intended. See :ref:`packaging-a-plugin`.

.. _api-environment:

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Variable
     - Effect
   * - ``TRACKGEN_DEBUG_SYNC``
     - ``1``: every library call blocks the host on its stream after its launches, so an
       asynchronous CUDA error surfaces at the call that caused it. A debugging aid,
       read at every call. It never blocks on the CPU device or while the call's stream
       is capturing, but it cannot see a capture on another stream (a
       ``torch.cuda.graph`` on a side stream), which the sync may invalidate: leave it
       unset when you record that way. The hot-path guards of
       :mod:`track_gen.testing.contract` switch it off while they run.
   * - ``TRACK_GEN_PLUGINS``
     - What happens to a third-party plugin that fails to load
       (:data:`track_gen.plugin.POLICY_ENV`): ``warn`` (the default: the plugin is
       quarantined and one ``RuntimeWarning`` names it), ``strict`` (raise
       :class:`~track_gen.plugin.PluginLoadError`, for CI) or ``off`` (skip third-party
       discovery). Case-insensitive, read once, when the registry loads; any other
       value emits a ``RuntimeWarning`` and falls back to ``warn``.

The test suite's ``TRACKGEN_REQUIRE_CUDA`` is described in :ref:`dev-setup`.

.. toctree::
   :maxdepth: 1
   :hidden:

   results
   pipeline
   plugin
   ops
   runtime
   exec
   testing
