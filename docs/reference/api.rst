API reference
=============

Runtime facades
---------------

.. autoclass:: track_gen.TrackGenerator
   :members:

.. autoclass:: track_gen.GateGenerator
   :members:

Seeded RNG
----------

.. autoclass:: track_gen.PerEnvSeededRNG
   :members:

Seed state
----------

Per-env seeds and device-side epochs, the pair that keys every generation; see
:doc:`/how-it-works/cuda-graph`.

.. automodule:: track_gen.seeds
   :no-members:

.. autoclass:: track_gen.seeds.SeedState
   :members:

.. autofunction:: track_gen.seeds.effective_seed

Execution layer
---------------

Stream resolution, the record / eager / self-graph policy shared by the producers, the
device-side conditional skip, and the capture errors; see :doc:`/how-it-works/cuda-graph`.

.. automodule:: track_gen.exec
   :no-members:

.. autoclass:: track_gen.exec.LaunchContext
   :members:

.. autoclass:: track_gen.exec.GraphRunner
   :members:

.. autofunction:: track_gen.exec.resolve_stream

.. autoexception:: track_gen.exec.CaptureError

.. autoexception:: track_gen.exec.NotCapturableError

.. autoexception:: track_gen.exec.LegacyPluginStreamError

Device ops
----------

The building blocks the built-in generators are made of, public for plugin generators;
:doc:`/contributing/writing-a-generator` shows how they fit together. The ops modules
import only Warp and each other, so importing ``track_gen.ops`` loads no generator module
and no torch.

.. automodule:: track_gen.ops
   :no-members:

Resampling
~~~~~~~~~~

.. autoclass:: track_gen.ops.ArcScratch
   :members: alloc

.. autofunction:: track_gen.ops.arc_resample

.. autofunction:: track_gen.ops.resample_uniform

.. autofunction:: track_gen.ops.resample_constant_spacing

Polygon geometry
~~~~~~~~~~~~~~~~

.. autofunction:: track_gen.ops.self_intersections

.. autofunction:: track_gen.ops.turning_number

.. autofunction:: track_gen.ops.thickness

.. autofunction:: track_gen.ops.normalize_extent

.. autofunction:: track_gen.ops.angle_sort

Random sampling
~~~~~~~~~~~~~~~

Each env's stream is ``wp.rand_init(seeds[e] * salt + offset)`` in int32 wrap-around
arithmetic, the keying the built-in generators use. The ``(salt, offset)`` pairs the
built-ins already use are listed in :ref:`the generator guide <ops-rng-keying>`.

.. autofunction:: track_gen.ops.grid_cells

.. autofunction:: track_gen.ops.sample_counts

.. autofunction:: track_gen.ops.sample_grid_points

Curves
~~~~~~

.. autofunction:: track_gen.ops.closed_catmull_rom

.. autofunction:: track_gen.ops.closed_polyline

Fills, flags and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: track_gen.ops.fill

.. autofunction:: track_gen.ops.any_set

.. autofunction:: track_gen.ops.check_array

Hashing
~~~~~~~

.. py:currentmodule:: track_gen.ops

.. py:function:: hash32(x: uint32) -> uint32

   ``@wp.func``: the MurmurHash3 ``fmix32`` finaliser, a ``uint32`` bijection with full
   avalanche (wrap-around arithmetic). Call it inside a kernel; it is also exported as
   ``track_gen.ops.funcs.hash32``.

.. autofunction:: track_gen.ops.hash32_host

Device functions: ``track_gen.ops.funcs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: track_gen.ops.funcs
   :no-members:

.. py:currentmodule:: track_gen.ops.funcs

.. py:function:: ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float

   Orientation of ``(o, p, q)``: ``(qy - oy)*(px - ox) - (py - oy)*(qx - ox)``, twice the
   signed triangle area; positive when ``o -> p -> q`` turns counter-clockwise.

.. py:function:: cross2(a: vec2f, b: vec2f) -> float

   2-D cross product ``a.x*b.y - a.y*b.x``.

.. py:function:: nan0(x: float) -> float

   ``x`` if finite, else ``0.0`` (a no-op on finite inputs).

.. py:function:: safe_normalize2(v: vec2f) -> vec2f

   ``v / max(|v|, 1e-8)``: a unit vector with a length floor. An all-NaN vector (a
   pruned corner) stays all-NaN. A NaN length is floored to ``1e-8`` like a short one,
   so a vector with one NaN component keeps the other component, scaled by ``1e8``.

.. py:function:: catmull_rom(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, u: float) -> vec2f

   Uniform Catmull-Rom (tension 0.5) segment from ``p1`` to ``p2`` at ``u`` in
   ``[0, 1]``: ``u = 0`` gives ``p1``, ``u = 1`` gives ``p2``. The function behind
   :func:`track_gen.ops.closed_catmull_rom`.

.. py:function:: segments_cross(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> int

   ``1`` if segments ``ab`` and ``cd`` cross properly, else ``0``. A strict sign test:
   touching endpoints and collinear overlap are not crossings. It is the gate validity
   predicate, not the one :func:`track_gen.ops.self_intersections` uses.

These are the operations the built-in kernels perform, in the same order. A plugin
kernel built from them matches the built-ins bit for bit on CPU. On CUDA, Warp compiles
with FMA contraction on by default (``wp.config.fuse_fp``), and which multiply-add pairs
become FMAs depends on the code around the inlined call, so a differently structured
kernel can differ in the last bit.

Constants
~~~~~~~~~

.. automodule:: track_gen.ops.constants
   :no-members:

.. autodata:: track_gen.ops.constants.TARGET_EXTENT

Plugin registry
---------------

Track and gate generators, the ``relax``, ``width``, ``elevation`` and ``validity``
stages of the track pipeline and the ``order`` and ``stations`` stages of the gate
pipeline, selected by name, built-ins and installed packages alike (entry point group
``track_gen.plugins``), their parameters and the ``Stage`` protocol; see
:doc:`/contributing/writing-a-generator` and :doc:`/contributing/writing-a-stage`.
``track_gen.plugin`` is a subpackage outside ``track_gen.__all__``: import it explicitly.
Importing ``track_gen`` loads no generator module and no plugin; the registry loads on its
first lookup.

.. automodule:: track_gen.plugin
   :no-members:

.. py:data:: track_gen.plugin.API_VERSION
   :type: int
   :value: 1

   Version of the plugin API this TrackGen provides. A plugin declares the version it
   targets as ``track_gen_api = <int>`` in the module its entry point names (or as an
   attribute of its register callable); a plugin that declares a newer version is
   quarantined instead of being called.

.. py:data:: track_gen.plugin.ENTRY_POINT_GROUP
   :type: str
   :value: "track_gen.plugins"

   The ``importlib.metadata`` entry-point group scanned for third-party plugins.

.. py:data:: track_gen.plugin.POLICY_ENV
   :type: str
   :value: "TRACK_GEN_PLUGINS"

   The environment variable selecting the third-party plugin policy: ``warn``
   (default), ``strict`` or ``off``. Any other value emits a ``RuntimeWarning`` and
   falls back to ``warn``.

.. py:data:: track_gen.plugin.registry
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

.. autoclass:: track_gen.plugin.GateGeneratorSpec
   :no-members:

   **Deprecated**: binding one (``GateGenerator``, ``Course``, ``build``) emits a
   ``DeprecationWarning``; it runs through an adapter until 1.0. Write a gate generator
   ``Stage`` instead (:ref:`gate-generator-stage`).

.. autoexception:: track_gen.plugin.DuplicateSpecError

.. autoexception:: track_gen.plugin.PluginOverrideWarning

.. autoexception:: track_gen.plugin.PluginLoadError

Generator parameters
~~~~~~~~~~~~~~~~~~~~

A generator's own parameters, set with ``TrackGenConfig.generator_params`` /
``GateGenConfig.generator_params`` and reported by ``SpecInfo.params_schema``; see
:ref:`generator-params` and :ref:`generator-params-config`. At run time,
``TrackGenerator.params`` and ``GateGenerator.params`` return a :class:`ParamTable
<track_gen.plugin.ParamTable>` (see :ref:`generator-runtime-params`).

.. autoclass:: track_gen.plugin.Params
   :members: validate, from_legacy

.. autoclass:: track_gen.plugin.NoParams
   :no-members:

.. autofunction:: track_gen.plugin.static

.. autofunction:: track_gen.plugin.dynamic

.. autoclass:: track_gen.plugin.ParamTable
   :members: set

.. autoclass:: track_gen.plugin.StageParams
   :members: stage_id, params, set

Stages
~~~~~~

The ``Stage`` protocol: a ``Stage`` subclass, registered as a class, that allocates
through the ``StageBuilder`` it is handed and returns an ``enqueue(ctx)`` callable.
Plugins register ``kind="generator"`` stages with ``output="centerline"`` (see
:ref:`generator-stage-contract`), ``kind="gate_generator"`` stages with
``output="anchors"`` (see :ref:`gate-generator-stage`; they also declare
``provides``, ``orderings`` or ``excluded_orderings``, and ``preferred``), and stages of
the open pipeline kinds ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``,
``"order"`` and ``"stations"``, which read channels through ``StageBuilder.read`` and
rewrite them into ``StageBuilder.write`` (see :doc:`/contributing/writing-a-stage`). The
other stages are TrackGen's built-ins (see :doc:`/how-it-works/pipeline-preview`).

.. autoclass:: track_gen.plugin.Stage
   :members: capture_class, capacity, build

.. autoclass:: track_gen.plugin.StageBuilder
   :members: read, write, temp_like, stride_of, per_of, scratch, param, input, salt,
      fail_bit

.. autoclass:: track_gen.plugin.CaptureClass
   :no-members:

.. autoclass:: track_gen.plugin.Layout
   :no-members:

Pipeline (preview)
------------------

.. warning::

   **Preview.** ``track_gen.pipeline`` may change before 1.0. ``TrackGenerator`` and
   ``GateGenerator`` are the stable entry points; each accepts a ``PipelineSpec`` as well
   as its config.

The track and gate pipelines as ordered lists of stages: the spec types, the presets,
the planner and the ``Program`` it builds; see :doc:`/how-it-works/pipeline-preview`.
``TrackGenerator(config)`` builds its pipeline with :func:`legacy.to_pipeline_spec
<track_gen.pipeline.legacy.to_pipeline_spec>` and :func:`~track_gen.pipeline.build`;
``TrackGenerator(spec)`` builds the spec it is given, usually from
:func:`presets.track <track_gen.pipeline.presets.track>`. ``GateGenerator`` does the same
with :func:`legacy.to_gate_pipeline_spec
<track_gen.pipeline.legacy.to_gate_pipeline_spec>` and :func:`presets.gates
<track_gen.pipeline.presets.gates>` (a spec whose ``BatchSpec.max_gates`` is set). ``track_gen.pipeline`` is a
subpackage outside ``track_gen.__all__``: import it explicitly. ``import track_gen``
does not import it.

.. automodule:: track_gen.pipeline
   :no-members:

Specs
~~~~~

.. autoclass:: track_gen.pipeline.StageRef
   :no-members:

.. autoclass:: track_gen.pipeline.BatchSpec
   :no-members:

.. autoclass:: track_gen.pipeline.PipelineSpec
   :members: ids, replace, insert_after, insert_before, remove, to_dict, from_dict

.. autofunction:: track_gen.pipeline.legacy.to_pipeline_spec

.. autofunction:: track_gen.pipeline.legacy.to_gate_pipeline_spec

Presets
~~~~~~~

.. autofunction:: track_gen.pipeline.presets.track

.. autofunction:: track_gen.pipeline.presets.gates

Planner and Program
~~~~~~~~~~~~~~~~~~~

.. autofunction:: track_gen.pipeline.build

.. autoclass:: track_gen.pipeline.Program
   :members: run, enqueue, prepare, result, spec, tap, channels, fail_bits, on_invalid,
      seeds, seed, inputs, params, graph, capturable, capture_class

.. autoexception:: track_gen.pipeline.PlanError

Fail bits
~~~~~~~~~

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

Torch interop
-------------

Registering a ``torch.cuda.graph`` capture with Warp, and tying producers' lifetime to a
torch graph; see :doc:`/tutorials/cuda-graph-in-a-sim`. Importing ``track_gen`` does not
import torch.

.. automodule:: track_gen.interop.torch
   :no-members:

.. autofunction:: track_gen.interop.torch.registered

.. autofunction:: track_gen.interop.torch.retain

Conformance kit
---------------

The checks a generator or stage plugin runs in its own CI, and TrackGen runs over every
built-in; see :doc:`/contributing/testing-a-plugin`. ``track_gen.testing`` is a subpackage outside
``track_gen.__all__``: import it explicitly. ``import track_gen`` does not import it,
and importing it imports neither torch nor pytest.

.. automodule:: track_gen.testing
   :no-members:

Conformance checks
~~~~~~~~~~~~~~~~~~

.. automodule:: track_gen.testing.conformance
   :no-members:

.. autofunction:: track_gen.testing.check_generator

.. autofunction:: track_gen.testing.check_gate_generator

.. autofunction:: track_gen.testing.check_stage

.. py:currentmodule:: track_gen.testing

.. py:data:: CHECKS
   :type: Mapping[str, tuple[str, ...]]

   The check names of each registry kind (``"generator"``, ``"gate_generator"``,
   ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"``,
   ``"stations"``), in run order. A read-only mapping.

.. autoclass:: track_gen.testing.Report
   :members: ok, failures, failed_names, result, raise_for_failures

   ``kind`` and ``name`` identify the generator or stage; ``results`` is the list of
   :class:`CheckResult <track_gen.testing.CheckResult>`, one per check and device.

.. autoclass:: track_gen.testing.CheckResult
   :no-members:

Hot-path guards
~~~~~~~~~~~~~~~

.. automodule:: track_gen.testing.contract
   :no-members:

.. autofunction:: track_gen.testing.hot_path_guard

.. autofunction:: track_gen.testing.no_host_sync

.. autofunction:: track_gen.testing.no_allocation

.. autofunction:: track_gen.testing.stream_explicit

The four functions above are the entry points; the class below is the type they return.

.. autoclass:: track_gen.testing.HotPathGuard
   :members: ok, of, allocated, report, raise_for_violations

.. autoclass:: track_gen.testing.Violation
   :no-members:

.. autoclass:: track_gen.testing.Site
   :no-members:

.. py:currentmodule:: track_gen.testing

.. py:data:: CATEGORIES
   :type: tuple[str, ...]
   :value: ("host_sync", "allocation", "stream", "capture")

   The :class:`~track_gen.testing.Violation` categories, in report order (the argument
   of :meth:`HotPathGuard.of() <track_gen.testing.HotPathGuard.of>`).

.. autoexception:: track_gen.testing.HotPathViolation

Launch traces
~~~~~~~~~~~~~

.. automodule:: track_gen.testing.trace
   :no-members:

.. py:currentmodule:: track_gen.testing

.. py:function:: trace_launches(*, include_warp=False) -> LaunchTrace

   A :class:`~track_gen.testing.LaunchTrace` to use as a context manager (it yields
   itself). Calls Warp makes internally, such as the copy inside ``wp.clone``, are left
   out unless ``include_warp=True``.

   .. code-block:: python

      with trace_launches() as trace:
          gen.generate()

The type :func:`~track_gen.testing.trace_launches` returns:

.. autoclass:: track_gen.testing.LaunchTrace
   :members: kernels, topology, diff, same_topology, foreign_pointers

.. autoclass:: track_gen.testing.LaunchRecord
   :no-members:

.. autofunction:: track_gen.testing.buffers_of

.. autoclass:: track_gen.testing.BufferSet
   :members: add, locate

.. autofunction:: track_gen.testing.assert_same_topology

.. autofunction:: track_gen.testing.assert_pointers_within

Caller capture (CUDA)
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: track_gen.testing.capture
   :no-members:

.. autofunction:: track_gen.testing.assert_caller_capturable

.. py:currentmodule:: track_gen.testing

.. py:data:: CAPTURE_MODES
   :type: tuple[str, ...]
   :value: ("warp", "external", "torch")

   Every capture mode :func:`~track_gen.testing.assert_caller_capturable` knows, in run
   order.

The building blocks of :func:`~track_gen.testing.assert_caller_capturable`, for custom
capture tests (import them from ``track_gen.testing.capture``):

.. autofunction:: track_gen.testing.capture.side_stream

.. autofunction:: track_gen.testing.capture.mempool_disabled

.. autofunction:: track_gen.testing.capture.capture_launch_forbidden

.. autofunction:: track_gen.testing.capture.recording

.. autofunction:: track_gen.testing.capture.poison

.. autofunction:: track_gen.testing.capture.poison_all

.. autofunction:: track_gen.testing.capture.result_arrays

pytest plugin
~~~~~~~~~~~~~

``track_gen.testing.pytest_plugin`` is opt-in (there is no ``pytest11`` entry point):
load it with ``pytest_plugins = ["track_gen.testing.pytest_plugin"]`` in a top-level
``conftest.py``, or ``-p track_gen.testing.pytest_plugin``. It imports pytest.

.. py:module:: track_gen.testing.pytest_plugin
   :no-index:

.. py:currentmodule:: track_gen.testing.pytest_plugin

.. py:function:: track_gen_conformance(name, params=None, *, kind="generator", **kwargs)

   A pytest fixture whose value is this factory. It runs
   :func:`~track_gen.testing.check_generator` (with ``kind="gate_generator"``,
   :func:`~track_gen.testing.check_gate_generator`; with ``kind`` one of ``"relax"``,
   ``"width"``, ``"elevation"``, ``"validity"``, ``"order"``, ``"stations"``,
   :func:`~track_gen.testing.check_stage`)
   on ``name`` with ``params`` and ``kwargs``, fails the test listing every failed check
   unless all pass, and returns the :class:`~track_gen.testing.Report`.

.. py:function:: generator_names(kind="generator", *, builtin=None, exclude=()) -> list[str]

   The names registered under ``kind`` (``"generator"``, ``"gate_generator"`` or one of
   the six stage kinds), sorted, without those in ``exclude``. ``builtin=True`` keeps only the built-ins,
   ``builtin=False`` only the third-party generators.

.. py:function:: parametrize_generators(kind="generator", *, argname="generator", builtin=None, exclude=())

   ``pytest.mark.parametrize(argname, generator_names(kind, builtin=builtin,
   exclude=exclude))``. It reads the registry at collection time, so the plugins its
   entry points name load then.

.. py:currentmodule:: None

Result types
------------

.. autoclass:: track_gen.Track
   :no-members:

   .. automethod:: clone

   ``fail_mask`` and ``fail_bits`` say why an env is invalid: the bit table is
   :data:`~track_gen.pipeline.BUILTIN_FAIL_BITS` plus the pipeline's plugin checks (see
   :ref:`pipeline-fail-bits`).

.. autoclass:: track_gen.GateSequence
   :no-members:

   .. automethod:: clone

   ``fail_mask`` and ``fail_bits`` say why an env is invalid: the bit table is
   :data:`~track_gen.pipeline.GATE_FAIL_BITS` plus the pipeline's plugin checks (see
   :ref:`pipeline-gate-fail-bits`). ``fail_mask`` is ``None`` on a ``GateSequence`` built
   elsewhere; when it is set, ``valid[e] == (fail_mask[e] == 0)``.

Collision queries
-----------------

Box-vs-track out-of-bounds checks with full contact info. See
``track_gen.collision`` for backend trade-offs (exact ``segments`` scan vs
baked ``sdf`` grids).

.. figure:: ../assets/utilities-overview.png
   :alt: Cones and walls placed by track_gen.props, spacing comparison, and
         the collision SDF field with boxes classified by the exact backend.

   The query/instancing utilities on one generated track: ``track_gen.props``
   cones (points mode) and wall pieces (segments mode) on both boundaries,
   the effect of the spacing knob, and ``track_gen.collision``'s baked SDF
   field with boxes classified by the exact segments backend (green inside,
   red out of bounds, dotted lines to the nearest boundary point).
   Regenerate with ``python -m viz.render_utility_assets``.

.. automodule:: track_gen.collision
   :no-members:

.. autoclass:: track_gen.collision.CollisionChecker
   :members:

.. autoclass:: track_gen.collision.BoxContact
   :no-members:

   .. automethod:: clone

.. autoclass:: track_gen.collision.DiscChecker
   :members:

.. autoclass:: track_gen.collision.DiscContact
   :no-members:

   .. automethod:: clone

Performance
~~~~~~~~~~~

Measured numbers, backend trade-offs, and the reproduce command live in
:doc:`the collision utility page </utilities/collision>`.

Boundary props
--------------

Rendering-only instancing poses along track boundaries — cone lines
(``mode="points"``) and wall pieces (``mode="segments"``). Complementary to
the collision utility: props never collide. Results for invalid envs
(``Track.valid == 0``) are undefined — always gate on ``valid``.

.. automodule:: track_gen.props
   :no-members:

.. autoclass:: track_gen.props.PropSampler
   :members:

.. autoclass:: track_gen.props.PropSet
   :no-members:

   .. automethod:: clone

Checkpoints
-----------

Ordered course goals from gates (zero-copy) or subsampled track centerlines.

.. automodule:: track_gen.checkpoints
   :no-members:

.. autoclass:: track_gen.checkpoints.CheckpointSampler
   :members:

.. autoclass:: track_gen.checkpoints.CheckpointSet
   :no-members:

   .. automethod:: from_gates

   .. automethod:: clone

Progress tracking
-----------------

Stateful per-env course progress over any ``CheckpointSet``.

.. automodule:: track_gen.progress
   :no-members:

.. autoclass:: track_gen.progress.ProgressTracker
   :members:

.. autoclass:: track_gen.progress.ProgressEvents
   :no-members:

   .. automethod:: clone

Track-frame localization
------------------------

Per-env (s, n) centerline projection with an optional warm start, plus the
per-generation curvature and speed-profile helpers.

.. automodule:: track_gen.localize
   :no-members:

.. autoclass:: track_gen.localize.TrackLocalizer
   :members:

.. autoclass:: track_gen.localize.TrackFrame
   :no-members:

   .. automethod:: clone

.. autofunction:: track_gen.localize.curvature

.. autofunction:: track_gen.localize.speed_profile

Course facade
-------------

One object bundling generation, collision, and progress per mode; see the
:doc:`Course facade page </utilities/course>` for the lifecycle.

.. automodule:: track_gen.course
   :no-members:

.. autoclass:: track_gen.course.CourseConfig
   :no-members:

.. autoclass:: track_gen.course.Course
   :members:

.. autoclass:: track_gen.course.StepResult
   :no-members:

   .. automethod:: clone

.. autofunction:: track_gen.course.set_capturing
