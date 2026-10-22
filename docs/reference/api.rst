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

Track and gate generators selected by name, built-ins and installed packages alike (entry
point group ``track_gen.plugins``), their parameters and the ``Stage`` protocol; see
:doc:`/contributing/writing-a-generator`.
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
   ``GateGenerator`` and ``Course`` resolve ``config.generator`` through it.

.. autoclass:: track_gen.plugin.Registry
   :members:

.. autoclass:: track_gen.plugin.SpecInfo
   :no-members:

.. autoclass:: track_gen.plugin.GeneratorSpec
   :no-members:

.. autoclass:: track_gen.plugin.GateGeneratorSpec
   :no-members:

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

Stage generators
~~~~~~~~~~~~~~~~

The ``Stage`` protocol, generator subset: a ``Stage`` subclass with ``kind="generator"``
and ``output="centerline"``, registered as a class, that allocates through the
``StageBuilder`` it is handed and returns an ``enqueue(ctx)`` callable; see
:ref:`generator-stage-contract`. Gate generator stages (``output="anchors"``) arrive with
roadmap PR11.

.. autoclass:: track_gen.plugin.Stage
   :members: capture_class, capacity, build

.. autoclass:: track_gen.plugin.StageBuilder
   :members: write, scratch, param, input, salt

.. autoclass:: track_gen.plugin.CaptureClass
   :no-members:

.. autoclass:: track_gen.plugin.Layout
   :no-members:

Torch interop
-------------

Registering a ``torch.cuda.graph`` capture with Warp, and tying producers' lifetime to a
torch graph; see :doc:`/tutorials/cuda-graph-in-a-sim`. Importing ``track_gen`` does not
import torch.

.. automodule:: track_gen.interop.torch
   :no-members:

.. autofunction:: track_gen.interop.torch.registered

.. autofunction:: track_gen.interop.torch.retain

Result types
------------

.. autoclass:: track_gen.Track
   :no-members:

   .. automethod:: clone

.. autoclass:: track_gen.GateSequence
   :no-members:

   .. automethod:: clone

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
