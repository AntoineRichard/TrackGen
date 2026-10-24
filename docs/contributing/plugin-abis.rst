.. _plugin-abis:

GeneratorSpec and the 0.1 plugin ABIs
=====================================

.. admonition:: Before you start

   - **Who this page is for:** you maintain a generator written as a ``GeneratorSpec``,
     a gate generator written as a ``GateGeneratorSpec``, or a plugin that imports from
     ``track_gen._src`` (the 0.1 plugin interface).
   - **What to do:** a ``GeneratorSpec`` keeps working as it is; a new generator is
     better written as a ``Stage`` (:ref:`writing-a-generator`,
     :ref:`writing-a-gate-generator`). Everything else on this page is deprecated: the
     status table says what warns and until when, and each section maps the old names
     to their replacement. The upgrade guide lists the steps for a 0.1 plugin:
     :ref:`upgrading-registry`, :ref:`upgrading-plugins` and
     :ref:`upgrading-plugin-private-names`.
   - **How you test it:** ``check_generator`` and ``check_gate_generator`` check both
     forms against the same contract (:ref:`testing-a-plugin`).

Status
------

.. list-table::
   :header-rows: 1
   :widths: 34 22 26 18

   * - Interface
     - Status
     - Warning
     - Runs in
   * - ``track_gen.plugin.GeneratorSpec`` (with ``accepts_ctx=True``)
     - Supported. A ``Stage`` is recommended for new generators (dynamic parameters,
       attributes, ``provides``).
     - none
     - every 0.x release
   * - ``GeneratorSpec`` with ``accepts_ctx=False`` (a *legacy plugin*)
     - Supported; set ``accepts_ctx=True`` (`Streams: accepts_ctx`_).
     - none; recording it on another stream raises ``LegacyPluginStreamError``
     - every 0.x release
   * - ``track_gen.plugin.GateGeneratorSpec`` (the 0.1 gate ABI)
     - Deprecated: write an anchors ``Stage``.
     - ``DeprecationWarning`` when bound (once per name); registering is silent
     - 0.x; removed in 1.0
   * - The ``track_gen._src`` registry shims: ``generator_registry`` /
       ``gate_generator_registry`` ``register``, ``get``, ``available``,
       ``GENERATORS``, ``GATE_GENERATORS``
     - Deprecated: use ``track_gen.plugin.registry`` and an entry point.
     - silent (``register`` warns when it overwrites a name)
     - 0.x; removed in 1.0
   * - The ``warp_gate`` helpers: ``alloc_order_scratch``, ``order_points``,
       ``finish_ordered_gates``, ``_GateStaging``
     - Deprecated: the pipeline's order and normalize stages replace them.
     - ``DeprecationWarning`` on each use
     - 0.x; removed in 1.0
   * - The ``_src`` helpers of 0.1 track plugins (``warp_pipeline._fill_i32_k``,
       ``self_intersections_inplace``, ...; `Legacy _src helpers`_)
     - Deprecated: use ``track_gen.ops``.
     - silent
     - 0.x; removed in 1.0
   * - ``track_gen._src.warp_pipeline.GenScratch``
     - Deprecated: moved to ``track_gen._src.warp_generate.GenScratch``.
     - ``DeprecationWarning``
     - 0.x; removed in 1.0
   * - ``registry.get("generator", <built-in>).params``
     - Deprecated: the built-ins are Stage classes; read ``.Params``.
     - ``DeprecationWarning``
     - 0.x; removed in 1.0
   * - A ``TrackGenConfig`` subclass carrying a plugin's knobs
     - Superseded by a ``Params`` class and ``generator_params``
       (`Per-generator parameters`_).
     - none
     - every 0.x release

The hard rules, the registry and the conformance kit apply to both forms; what a
``GeneratorSpec`` cannot do is listed where it matters (no :term:`dynamic field`, no
per-point attributes, no ``provides``, no per-env :term:`size`).

The ``GeneratorSpec`` contract
------------------------------

The quick start of :ref:`writing-a-generator` shows a complete ``GeneratorSpec`` next to
the same generator written as a ``Stage``.

What you implement
~~~~~~~~~~~~~~~~~~

A ``GeneratorSpec`` has six fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``name``
     - The ``generator=`` string that selects it. Dotted for third parties (see
       :ref:`plugin-names`).
   * - ``alloc_scratch``
     - ``(config) -> scratch``, below.
   * - ``generate``
     - ``(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None) -> None``,
       below.
   * - ``capturable``
     - ``True`` (default): ``generate`` obeys the :ref:`hard rules
       <generator-hard-rules>` and is recorded into CUDA graphs. ``False``: see
       :ref:`generator-capture-opt-out`.
   * - ``accepts_ctx``
     - ``True``: ``generate`` takes a ``ctx=`` keyword and emits all its device work
       through it (see :ref:`generator-streams`). The default ``False`` exists for
       plugins written before it; set it to ``True``.
   * - ``params``
     - Optional: the generator's ``Params`` class, static fields only (see
       :ref:`generator-spec-params`). The default ``None`` takes no
       ``generator_params``.

``alloc_scratch(config) -> scratch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allocate your generator's private working buffers **once**. Use fixed shapes derived
from ``config`` (``num_envs``, ``num_points``, ...), all on ``str(config.device)``.
Return any object exposing the buffers your ``generate`` function uses. Do **not**
allocate the output centerline or the validity array here: the pipeline owns those.

``generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None) -> None``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a closed centerline of ``config.num_points`` points per :term:`env` into the output
arrays in place. Parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``seeds_wp``
     - ``[E]`` int32 ``wp.array``: one base seed per env (the effective seed derived
       from the env's :term:`seed` and :term:`epoch`, see :ref:`generating-seeds`).
       Read-only.
   * - ``config``
     - The generator's own copy of the configuration it was constructed with. With
       ``params=``, ``config.generator_params`` is the resolved, validated Params
       instance.
   * - ``out_centerline``
     - ``[E * num_points]`` ``wp.vec2f``: write the closed centerline in place.
   * - ``out_valid_wp``
     - ``[E]`` int32: fill it with ``1`` for every env, or ``0`` to reject one
       outright. Final validity is decided later, by the pipeline's validity stage.
   * - ``scratch``
     - The object your ``alloc_scratch`` returned.
   * - ``ctx``
     - The call's ``track_gen.exec.LaunchContext`` (one device, one stream), passed when
       the spec sets ``accepts_ctx=True``. Emit every launch, copy and fill through it.

The rest of the generator contract is the Stage's: the raw loop, its size and what you
do not have to guarantee (:ref:`generator-stage-contract`), and the
:ref:`hard rules <generator-hard-rules>`, with ``alloc_scratch`` in the place of the
``StageBuilder`` and a constant salt of your own in the place of ``b.salt``.

.. _generator-streams:

Streams: ``accepts_ctx``
~~~~~~~~~~~~~~~~~~~~~~~~

Set ``accepts_ctx=True`` and emit all device work through ``ctx``: ``ctx.launch(kernel,
dim, inputs)``, ``ctx.copy(dst, src)``, ``ctx.fill(arr, value)`` (or
``track_gen.ops.fill(ctx, arr, value)``). Every ``track_gen.ops`` op takes ``ctx`` as
its first argument; pass ``ctx=ctx`` to any other helper that takes it. Never call
``arr.zero_()``, ``arr.fill_()``, or a ``wp.launch`` / ``wp.copy`` without ``stream=``:
they target Warp's current stream, which need not be the stream the call resolved (a
side stream, or the stream torch is capturing). With ``accepts_ctx=True`` a generator can
be recorded into a caller's CUDA graph on any stream.

A spec that leaves ``accepts_ctx=False`` is a *legacy plugin*. It still runs, inside
``ctx.scope()``, so the library helpers it calls inherit the stream, but its own direct
``wp.launch`` calls go to Warp's current stream. The generator's own graph is then
recorded behind a process-wide lock, and recording it into a caller's graph requires the
capture stream to be Warp's current stream (``LegacyPluginStreamError`` otherwise); see
:ref:`capture-troubleshooting`.

.. _generator-capture-opt-out:

Opting out of capture: ``capturable=False``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A generator may set ``capturable=False`` to opt out of hard rules 2 and 3: it may
allocate per call and use host-side control flow (Python loops driving stage
transitions, readbacks that branch on generated data). It is then a ``HOST`` generator:
``TrackGenerator`` runs it **eagerly** on every call, on CUDA too, as the first segment
of the pipeline, and replays the rest of the pipeline from a graph
(:ref:`cuda-graph-host`). This is a supported path, not a fallback for a broken
generator. A caller cannot record it into a graph (``NotCapturableError``). Rules 1
(pure Warp, no torch) and 4 (fixed-shape buffers) still apply. ``GateGeneratorSpec``
carries the same flag. A ``Stage`` declares ``capture = CaptureClass.HOST`` instead and
reaches the host through ``ctx.host()`` (:ref:`stage-host`), which the conformance kit
can check; prefer it.

Before opting out, see how ``repulsive`` made a host-driven loop recordable: its
coarse-to-fine stage transitions, resample cadence and stall-check cadence form a fixed
schedule precomputed on the host from the configuration alone; the readback-driven early
exit became a per-env freeze on device (each env stops changing once it converges, and
every kernel skips a frozen env); and its growth loop is a ``ctx.bounded_loop`` over that
freeze (a ``BOUNDED`` Stage, :ref:`stage-bounded`). Rule 5 still applies, with a caveat
for chaotic per-iteration methods: a generator that accumulates its CUDA gradients with
``atomic_add`` can be only *statistically* reproducible on CUDA, since the float
summation order varies. ``repulsive`` avoids it with hand-written analytic adjoints (a
per-vertex gather, no atomics), so it stays byte-identical run to run on CPU and CUDA;
``track_gen/_src/warp_generate_repulsive.py``'s module docstring has the full account.

When your code runs
~~~~~~~~~~~~~~~~~~~

Your ``generate`` body runs once, eagerly, when the ``TrackGenerator`` is constructed
(the warm-up pass that loads every kernel). After that it runs once per eager call
(``cpu``) and once per recording: into the generator's own graph on the first ``cuda``
call, or into a caller's graph when ``generate()`` is called inside the caller's
:term:`capture`. Replays do not run Python at all, so host scalars and Python state
freeze at record time. A ``capturable=False`` generator runs once per call instead (the
rest of the pipeline still replays from a graph), and raises ``NotCapturableError`` under
a caller's capture. :ref:`cuda-graph-masked-regeneration` and
:ref:`cuda-graph-warm-up` describe the execution model.

.. _generator-spec-params:

Per-generator parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Declare your knobs as a :ref:`Params class <generator-params>` and pass it as
``GeneratorSpec(..., params=AcmeParams)``. Users set them with
``presets.track(generator="acme.wobbly", generator_params={...})`` (or the same
argument of ``TrackGenConfig``); ``alloc_scratch`` and ``generate`` read the resolved,
validated instance as ``config.generator_params`` (``config.generator_params.lobes``),
and ``describe()`` reports its schema. The class may have static fields only:
``registry.add`` rejects a dynamic field, which needs a Stage's ``b.param``. The shared
fields (``num_envs``, ``num_points``, ``scale``, ``device``, ...) stay on the config.

Plugins written before Params existed read their knobs from a ``TrackGenConfig``
subclass (``getattr(config, "acme_lobes", 3)``). That still works, but it is no longer
needed: move the knobs to a Params class and override ``from_legacy`` to read the old
fields, so the subclass configs your users already have resolve to the same values.
Registry-wide tools (``benchmarks/compare_generators.py``, ``viz/param_explorer.py``)
build a plain ``TrackGenConfig`` for every registered generator, so a generator that
needs a config subclass must still fall back to defaults there; a Params class needs
nothing more.

Porting a ``GeneratorSpec`` to a ``Stage``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two quick starts of :ref:`writing-a-generator` are the same generator in both
forms. The port is mechanical:

- ``alloc_scratch`` becomes ``build(self, b, params)``: each buffer becomes a
  ``b.scratch(name, dtype, shape)``, and the two outputs ``b.write("xy", ...)`` and
  ``b.write("valid", ...)`` (declared in ``writes``);
- ``generate`` becomes the ``enqueue(ctx)`` closure that ``build`` returns: ``seeds_wp``
  is ``b.seed``, ``config.num_points`` is ``b.raw``, ``config.num_envs`` is ``b.E``,
  ``config.scale`` is ``b.scale``, and host scalars computed from ``config`` are
  computed once in ``build``;
- your constant salt becomes ``b.salt("<stream>")`` (new streams, so new per-seed
  outputs; keep the constant to keep the bytes);
- ``params=`` becomes the ``Params`` attribute, and a knob a curriculum varies can then
  become a dynamic field (``b.param``);
- ``capturable=False`` becomes ``capture = CaptureClass.HOST`` with ``ctx.host()`` for
  the readbacks (:ref:`stage-host`).

The name, the entry point and the users' ``generator=`` strings stay the same.

The legacy gate generator ABI: ``GateGeneratorSpec``
----------------------------------------------------

Gate generators written against the 0.1 interface are ``GateGeneratorSpec`` objects
(exported by ``track_gen.plugin``). They keep working, unchanged, through an adapter
stage (:ref:`pipeline-legacy-gates`), and are **deprecated**: registering one is silent,
but the first ``GateGenerator``, ``Course``, ``build`` or conformance run that binds one
emits a ``DeprecationWarning`` (once per name), pointing at the code that constructed
the generator. They run until 1.0. A ``GateGeneratorSpec`` has the ``name``,
``alloc_scratch``, ``capturable`` and ``accepts_ctx`` fields of a ``GeneratorSpec``,
with the same meaning, the same hard rules and the same stream rules, plus:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``generate``
     - ``(seeds_wp, config, out, scratch, ctx=None) -> None``. ``out.position`` is the
       ``[E * max_gates]`` ``wp.vec2f`` plan-view staging buffer (``max_gates`` is
       ``config.max_gates``): write env ``e``'s gates in order at ``e * max_gates + g``
       for ``g < count``, and NaN in every slot after them. ``out.count`` is ``[E]``
       int32: write each env's real gate count.
   * - ``max_gates``
     - ``(config) -> int``: the most gates ``generate`` can emit for ``config``.
       ``GateGenerator`` raises ``ValueError`` unless ``config.min_gates <=
       max_gates(config) <= config.max_gates``.
   * - ``supported_orderings``
     - ``frozenset`` of the ``GateGenConfig.gate_ordering`` values that ``generate``
       honours. No order stage follows a legacy generator: ``generate`` emits the gates
       in the configured order, and ``GateGenerator`` raises ``ValueError`` for an
       ordering not in this set. ``GateGenConfig.gate_ordering=None`` (the default) is
       resolved first (``"ccw"`` when supported, else the first of this set, sorted):
       ``config.gate_ordering`` is never ``None`` in ``generate``, ``alloc_scratch``,
       ``max_gates`` or the ``params`` class's ``from_legacy``.
   * - ``params``
     - Optional: a static ``Params`` class, as for a ``GeneratorSpec``. Users set it
       with ``GateGenConfig.generator_params``, and ``generate``, ``alloc_scratch`` and
       ``max_gates`` read the resolved instance as ``config.generator_params``.

``presets.gates`` with a legacy generator refuses ``order=``: choose its ordering with
``GateGenConfig(gate_ordering=...)``.

To port one to an anchors ``Stage`` (:ref:`writing-a-gate-generator`):

- move the body of ``generate`` up to (excluding) its ordering into ``build`` /
  ``enqueue``, writing ``anchors.xy`` at the generator's own stride (``per="anchors"``)
  instead of ``out.position``, and ``anchors.count`` instead of ``out.count``;
- drop the ordering (``ops.angle_sort`` is ``order.ccw``) and the extent normalization
  (declare ``preferred={"normalize": "extent"}``), and declare the orders it accepts
  with ``excluded_orderings`` (or ``orderings``) and ``preferred={"order": ...}``;
- turn ``max_gates(config)`` into ``capacity(params, layout) -> {"gates": A,
  "min_gates": k}``, and ``alloc_scratch`` into ``b.write`` / ``b.scratch`` calls.

The private ``warp_gate`` helpers that 0.1 gate plugins called from ``generate`` and
``alloc_scratch`` (``alloc_order_scratch``, ``order_points``, ``finish_ordered_gates``
and the ``_GateStaging`` name) are deprecated and removed in 1.0: until then they keep
working, with a ``DeprecationWarning``. When porting such a spec, drop the call: the
pipeline's ``order.raw``, ``order.ccw`` and ``order.random_pairs`` stages are the three
orderings ``order_points`` dispatched to (``ops.angle_sort`` is ``order.ccw``), and
``preferred={"normalize": "extent"}`` replaces ``normalize_extent=``.

.. _legacy-src-helpers:

Legacy ``_src`` helpers
-----------------------

Plugins written before ``track_gen.ops`` existed import private helpers from
``track_gen._src.warp_pipeline`` (usually as ``_pipe``), ``track_gen._src.warp_generate``
and ``track_gen._src.warp_generate_hull``. They are still importable: each keeps its
name and parameter list, which ``tests/test_legacy_plugin_abi.py`` pins for the 0.x
series, and delegates to the same kernels as the op that replaces it, so its results are
unchanged. They are deprecated in favour of ``track_gen.ops`` (they do not warn); port a
plugin with this table (:ref:`upgrading-plugin-private-names` lists the private kernel
names that are gone):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy helper
     - Replacement
   * - ``_pipe._init()``, ``_pipe._sync(device)``
     - Not needed: the facades initialise Warp before calling ``generate``, and the ops
       never sync.
   * - ``_pipe._fill_i32_k`` (a kernel)
     - ``ops.fill(ctx, arr, value)``
   * - ``_pipe._ccw`` (a ``wp.func``)
     - ``ops.funcs.ccw``
   * - ``_pipe._arc_resample_inplace``, ``_pipe._arc_resample_selected_inplace``
       (and their ``arc_length_resample_*`` twins)
     - ``ops.arc_resample(..., active=...)``, with an ``ops.ArcScratch`` wrapping the
       same four buffers
   * - ``_pipe.self_intersections_inplace``
     - ``ops.self_intersections``
   * - ``_pipe.turning_number_inplace``
     - ``ops.turning_number``
   * - ``_pipe.resample_uniform``
     - ``ops.resample_uniform`` (pass the scratch: the helper allocates when ``seg_wp``
       or ``s_wp`` is ``None``, the op never does)
   * - ``_pipe.resample_constant_spacing``
     - ``ops.resample_constant_spacing`` (the helper's allocating mode also reads the
       truncation flag back to warn; the op leaves it on the device)
   * - ``warp_generate.corner_count_sample_inplace``,
       ``warp_generate_hull.point_count_sample_inplace``
     - ``ops.sample_counts`` with the salts of :ref:`ops-rng-keying`
   * - ``warp_generate.corner_sample_inplace``,
       ``warp_generate_hull.point_sample_inplace``
     - ``ops.grid_cells`` and ``ops.sample_grid_points``
   * - ``warp_generate.ccw_sort_inplace``
     - ``ops.angle_sort`` (the helper sorts one point when ``count[e] < 1``; the op
       treats ``count[e] <= 0`` as empty)

The helpers keep their own conventions: ``ctx=None`` as the last argument (the ambient
context when omitted), a debug-only ``ctx.sync()`` after their launches, and the config
objects the samplers read. ``track_gen._src.warp_pipeline.GenScratch`` moved to
``track_gen._src.warp_generate.GenScratch``; the old name still works and emits a
``DeprecationWarning``.

The built-in generators' module functions keep their former signatures and scalar
kernels (their bytes are pinned by the fingerprint tests): ``generate_centerline_warp``
and ``bezier_alloc_scratch``, ``generate_hull_warp`` / ``hull_alloc_scratch``,
``generate_polar_warp`` / ``polar_alloc_scratch``, ``generate_voronoi_warp`` /
``voronoi_alloc_scratch``, ``generate_checkpoint_warp`` / ``checkpoint_alloc_scratch``,
``warp_zprofile.apply_z_profile`` / ``apply_z_profile_knots``, and the float sampling
helpers ``sample_polar_controls`` (which still expects pre-clamped jitters),
``sample_checkpoint_rings`` and ``select_voronoi_anchors``. The registered generators are
Stages that no longer call them (they launch the array-reading twins). They go at 1.0,
except the sampling helpers' ``*_per_env`` forms.

The old private registries
--------------------------

Plugins written for 0.1 imported ``track_gen._src.generator_registry`` (or
``gate_generator_registry``) and called its ``register(spec)`` at import time. That keeps
working: ``register``, ``get`` and ``available`` delegate to ``track_gen.plugin.registry``,
and ``GENERATORS`` / ``GATE_GENERATORS`` are read-only views of it. Each old ``register``
keeps its old duplicate rule. The track registry's still overwrites an existing name (it
is ``registry.add(spec, replace=True)``). The gate registry's still raises when a
different module registered the name (now ``DuplicateSpecError``, still a
``ValueError``) and overwrites a spec from the same module. Either emits a
``DeprecationWarning`` when it overwrites. Such a plugin must still be imported before
use (TrackGen's own generator modules no longer register when imported: the registry
calls their ``register(reg)``). Move it to an entry point and ``registry.add`` to drop
that requirement (:ref:`packaging-a-plugin`). The shims and the built-in generators'
``.params`` alias go at 1.0 (:ref:`upgrading-removals`).
