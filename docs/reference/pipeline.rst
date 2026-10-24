.. _reference-pipeline:

Pipelines and presets
=====================

The track and gate pipelines as ordered lists of :term:`stages <stage>`: the
:term:`spec` types, the :term:`presets <preset>`, the planner and the ``Program`` it
builds, and the Params of the built-in stages. :ref:`generating` shows how to use them;
:ref:`pipeline-stages` describes the default stages and the planner.

``TrackGenerator(spec)`` builds the spec it is given, usually from :func:`presets.track
<track_gen.pipeline.presets.track>`; ``TrackGenerator(config)`` builds the spec of
:func:`legacy.to_pipeline_spec <track_gen.pipeline.legacy.to_pipeline_spec>` with
:func:`~track_gen.pipeline.build`. ``GateGenerator`` does the same with
:func:`presets.gates <track_gen.pipeline.presets.gates>` (a spec whose
``BatchSpec.max_gates`` is set) and :func:`legacy.to_gate_pipeline_spec
<track_gen.pipeline.legacy.to_gate_pipeline_spec>`. ``track_gen.pipeline`` is a
subpackage outside ``track_gen.__all__``: import it explicitly (``import track_gen``
does not import it).

.. automodule:: track_gen.pipeline
   :no-members:

Specs
-----

.. autoclass:: track_gen.pipeline.StageRef
   :no-members:

.. autoclass:: track_gen.pipeline.BatchSpec
   :no-members:

.. autoclass:: track_gen.pipeline.PipelineSpec
   :members: ids, replace, insert_after, insert_before, remove, with_randomize, to_dict,
      from_dict
   :exclude-members: batch, stages, require_caller_capturable

Presets
-------

.. automodule:: track_gen.pipeline.presets
   :no-members:

.. autofunction:: track_gen.pipeline.presets.track

.. autofunction:: track_gen.pipeline.presets.gates

.. py:data:: StageSel
   :module: track_gen.pipeline.presets

   The type of a preset's stage keyword: a stage name, a ``(name, params)`` pair, a
   ``StageRef``, or ``None``.

From a configuration
--------------------

.. automodule:: track_gen.pipeline.legacy
   :no-members:

.. autofunction:: track_gen.pipeline.legacy.to_pipeline_spec

.. autofunction:: track_gen.pipeline.legacy.to_gate_pipeline_spec

.. autofunction:: track_gen.pipeline.legacy.generator_config

.. autofunction:: track_gen.pipeline.legacy.gate_generator_config

Planner and Program
-------------------

.. autofunction:: track_gen.pipeline.build

.. autoclass:: track_gen.pipeline.Program
   :members: run, enqueue, enqueue_segment, prepare, prepared, default_stream, result,
      layout, spec, tap, channels, fail_bits, on_invalid, attributes, seeds, seed, inputs,
      input_info, params, graph, segment_graphs, capturable, capture_class, segments,
      host_stage_ids

.. autoclass:: track_gen.pipeline.Producer
   :members: result, layout, enqueue

.. autoclass:: track_gen.pipeline.Segment
   :members: host
   :exclude-members: index, capture_class, stage_ids

.. autoclass:: track_gen.pipeline.AttributeInfo
   :no-members:

.. autoclass:: track_gen.pipeline.InputInfo
   :no-members:

.. autoexception:: track_gen.pipeline.PlanError

Fail bits
---------

The bits of ``fail_mask``, the :term:`fail bits` that say why an env is not
:term:`valid`.

.. py:currentmodule:: track_gen.pipeline

.. py:data:: BUILTIN_FAIL_BITS
   :type: Mapping[str, int]

   The ``valid.fail`` bits of TrackGen's built-in checks, a read-only mapping:
   ``{"generator": 0, "turning": 1, "finite": 2, "thickness": 3, "w_floor": 4,
   "border": 5, "grade": 6, "truncation": 7}``. Bits 8 to 15 are reserved for future
   built-in checks, 16 to 30 are assigned to plugin checks by
   :meth:`StageBuilder.fail_bit() <track_gen.plugin.StageBuilder.fail_bit>`, and bit
   31 is never used. ``Program.fail_bits`` and ``Track.fail_bits`` extend it with the
   plugin checks of a pipeline; see :ref:`pipeline-fail-bits`.

.. py:data:: GATE_FAIL_BITS
   :type: Mapping[str, int]

   The ``valid.fail`` bits of ``validity.gate_default``, the gate pipeline's built-in
   check, a read-only mapping: ``{"count": 0, "finite": 1, "tangent": 2, "spacing": 3,
   "crossing": 4, "grade": 5}``. Bits 6 to 15 are reserved, 16 to 30 go to plugin checks
   as for tracks. ``Program.fail_bits`` and ``GateSequence.fail_bits`` extend it with the
   plugin checks of a gate pipeline; see :ref:`pipeline-gate-fail-bits`.

.. py:currentmodule:: None

Ingestion
---------

Loading your own centerlines into a generator that declares the ``centerline.*`` inputs,
such as the built-in ``external``; see :ref:`ingesting-centerlines`.

.. autofunction:: track_gen.pipeline.load_centerlines

Built-in stage parameters
-------------------------

Each built-in stage's :term:`Params`. Most keep the ``TrackGenConfig`` /
``GateGenConfig`` field names (their ``from_legacy`` copies the flat values, see
:ref:`pipeline-stages`); these take more: ``width.from_channel`` (see
:ref:`pipeline-width`), the ``stations`` stages (see :ref:`pipeline-stations-xyz`),
``randomize.uniform`` (see :ref:`config-randomize`), the ``external`` generator (see
:ref:`ingesting-centerlines`), and ``normalize.extent``, whose one field is the per-env
:term:`size` of the ``polar``, ``voronoi`` and ``checkpoint`` gate generators (key
``normalize.size``, set by ``GateGenConfig.size``; see :ref:`config-size`). A
:term:`dynamic field` of these classes is a live per-env array (:ref:`curricula`).

.. automodule:: track_gen.stages.params
   :no-members:

.. autoclass:: track_gen.stages.params.WidthFromChannelParams
   :no-members:

.. autoclass:: track_gen.stages.params.StationsParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.NormalizeExtentParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

Randomizers
~~~~~~~~~~~

.. automodule:: track_gen.stages.randomize
   :no-members:

.. autoclass:: track_gen.stages.randomize.RandomizeUniform
   :no-members:

.. autoclass:: track_gen.stages.randomize.RandomizeUniformParams
   :no-members:

The ``external`` generator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: track_gen.stages.external
   :no-members:

.. autoclass:: track_gen.stages.external.External
   :members: declare

.. autoclass:: track_gen.stages.external.ExternalParams
   :no-members:

The other built-in stages
~~~~~~~~~~~~~~~~~~~~~~~~~

Their fields keep the ``TrackGenConfig`` / ``GateGenConfig`` names, and ``from_legacy``
reads them from a config.

.. autoclass:: track_gen.stages.params.ConstantSpacingParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.ElevationFlatParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.ElevationNoiseParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.ElevationRandomWalkParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.ElevationUniformParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.GateFramesParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.GateFramesPosedParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.GateRelaxParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.GateValidityParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.LiftFlatParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.OffsetParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.ValidityParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.WidthConstantParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy

.. autoclass:: track_gen.stages.params.XpbdParams
   :no-members:
   :no-show-inheritance:

   .. automethod:: from_legacy
