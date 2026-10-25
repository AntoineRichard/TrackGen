.. _reference-plugin:

Plugins
=======

The plugin registry and the plugin-facing types: track and gate generators, the
``relax``, ``width``, ``elevation`` and ``validity`` :term:`stages <stage>` of the track
pipeline and the ``order`` and ``stations`` stages of the gate pipeline, selected by name,
built-ins and installed packages alike (entry point group ``track_gen.plugins``), their
:term:`Params` and the ``Stage`` protocol. :ref:`writing-a-generator` and
:ref:`writing-a-stage` show how they fit together. ``track_gen.plugin`` is a subpackage
outside ``track_gen.__all__``: import it explicitly. Importing ``track_gen`` loads no
generator module and no plugin; the registry loads on its first lookup.

.. automodule:: track_gen.plugin
   :no-members:

Registry
--------

.. py:data:: API_VERSION
   :type: int
   :value: 1

   Version of the plugin API this TrackGen provides. A plugin declares the version it
   targets as ``track_gen_api = <int>`` in the module its entry point names (or as an
   attribute of its register callable); a plugin that declares a newer version is
   quarantined instead of being called.

.. py:data:: ENTRY_POINT_GROUP
   :type: str
   :value: "track_gen.plugins"

   The ``importlib.metadata`` entry-point group scanned for third-party plugins.

.. py:data:: POLICY_ENV
   :type: str
   :value: "TRACK_GEN_PLUGINS"

   The environment variable selecting the third-party plugin policy: ``warn``
   (default), ``strict`` or ``off`` (:ref:`api-environment`). Any other value emits a
   ``RuntimeWarning`` and falls back to ``warn``.

.. py:data:: registry
   :type: track_gen.plugin.Registry

   The process-global :class:`~track_gen.plugin.Registry`. ``TrackGenerator``,
   ``GateGenerator`` and ``Course`` resolve ``config.generator`` through it, and
   ``TrackGenerator``, ``GateGenerator`` and :func:`~track_gen.pipeline.build` the
   pipeline stages of the open kinds (``config.relax_solver``, ``config.z_profile`` and
   ``GateGenConfig.gate_ordering`` included).

.. autoclass:: track_gen.plugin.Registry
   :members:

.. autoclass:: track_gen.plugin.SpecInfo
   :no-members:

.. autoclass:: track_gen.plugin.GeneratorSpec
   :no-members:

   A supported simple track-generator form. Its generate callable always accepts
   ``ctx=``; use a Stage for dynamic parameters and gate generators (:ref:`plugin-abis`).

.. autoexception:: track_gen.plugin.DuplicateSpecError

.. autoexception:: track_gen.plugin.PluginOverrideWarning

.. autoexception:: track_gen.plugin.PluginLoadError

Parameters
----------

A generator's parameters are selected with ``generator_params`` and reported by
``SpecInfo.params_schema``. Runtime ``gen.params`` is the live per-stage
:class:`~track_gen.plugin.ParamTable` (:ref:`generator-runtime-params`).

.. autoclass:: track_gen.plugin.Params
   :members: validate, from_legacy

.. autoclass:: track_gen.plugin.NoParams
   :no-members:

.. autofunction:: track_gen.plugin.static

.. autofunction:: track_gen.plugin.dynamic

.. autoclass:: track_gen.plugin.ParamTable
   :members: set

.. autoclass:: track_gen.plugin.StageParams
   :members: stage_id, params, drawn, set, in_effect, bounds

The built-in generators' and elevation stages' Params have dynamic fields too (the
generators' shape knobs and ``size``, ``bezier`` / ``hull``'s ``int32``
``min_num_points``, ``z_min`` / ``z_max`` / ``z_max_step`` / ``z_noise_amplitude``, the
gate pipelines' ``normalize.size`` and ``gate_frames.gate_width``), with declared bounds:
see :ref:`builtin-generator-params` and :ref:`generators-per-env`. ``in_effect`` and
``bounds`` are the curriculum bookkeeping of :ref:`the tuning guide <tuning-in-effect>`.
Reach a built-in generator's Params class with
``registry.get("generator", name).Params``.

Stages
------

The ``Stage`` protocol: a ``Stage`` subclass, registered as a class, that allocates
through the ``StageBuilder`` it is handed and returns an ``enqueue(ctx)`` callable.
Plugins register ``kind="generator"`` stages with ``output="centerline"`` (see
:ref:`generator-stage-contract`; an ingestion generator reads ``per="raw"`` inputs and
may declare its ports per params with ``declare``, see :ref:`ingestion-generator`),
``kind="gate_generator"`` stages with ``output="anchors"`` (see
:ref:`gate-generator-stage`; they also declare ``provides``, ``orderings`` or
``excluded_orderings``, and ``preferred``), and stages of the open pipeline kinds
``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"`` and
``"stations"``, which read :term:`channels <channel>` through ``StageBuilder.read`` and
rewrite them into ``StageBuilder.write`` (see :ref:`writing-a-stage`). The other stages
are TrackGen's built-ins (see :ref:`pipeline-stages`). A stage's ``CaptureClass`` is
``STATIC``, ``BOUNDED`` (a loop through ``LaunchContext.bounded_loop``, see
:ref:`stage-bounded`) or ``HOST`` (host work through ``LaunchContext.host()`` and
``StageBuilder.host_buffer``, see :ref:`stage-host`).

.. autoclass:: track_gen.plugin.Stage
   :members: capture_class, capacity, build, declare

   Besides ``kind``, ``name``, ``Params``, ``reads``, ``writes``, ``provides``,
   ``capture``, ``output`` and ``topology``, a stage may set ``seeded`` (``True`` by
   default; ``False`` for a generator whose output does not depend on the seed, see
   :ref:`ingestion-generator`) and ``preferred`` (a gate generator's ``"order"`` /
   ``"normalize"``, a centerline generator's ``"relax"``).

   Every built-in generator is a ``Stage`` class: ``registry.get("generator",
   "bezier")`` (``hull``, ``polar``, ``voronoi``, ``checkpoint``: ``STATIC``;
   ``repulsive``: ``BOUNDED``; ``external``) and ``registry.get("gate_generator",
   name)``. Internal module functions are private; plugins use the public ops
   described in :ref:`generator-ops`.

.. autoclass:: track_gen.plugin.StageBuilder
   :members: read, write, temp_like, stride_of, per_of, dtype_of, scratch, param, input,
      salt, fail_bit, host_buffer

.. autoclass:: track_gen.plugin.CaptureClass
   :no-members:

.. py:class:: track_gen.plugin.Layout
   :no-index:

   The run's shape handed to ``Params.validate``, ``Stage.capacity`` and the
   ``StageBuilder``: the same class as :class:`track_gen.results.Layout` (see
   :ref:`results-layout`).
