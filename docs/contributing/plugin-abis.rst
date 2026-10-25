.. _plugin-abis:

GeneratorSpec plugins
=====================

``track_gen.plugin.GeneratorSpec`` is the simple track-generator interface: two
callables, a capture flag and an optional static Params class. It is supported for
fixed-shape generators. Use a :class:`~track_gen.plugin.Stage` when you need dynamic
parameters, exported attributes, ``provides`` or a bounded device loop; gate generators
use Stages (:ref:`writing-a-gate-generator`). Both track forms use the same registry,
entry points and conformance kit (:ref:`testing-a-plugin`). The two quick starts in
:ref:`writing-a-generator` show the same loop in each form.

The ``GeneratorSpec`` contract
------------------------------

The quick start of :ref:`writing-a-generator` shows a complete ``GeneratorSpec`` next to
the same generator written as a ``Stage``.

What you implement
~~~~~~~~~~~~~~~~~~

A ``GeneratorSpec`` has five fields:

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
     - ``(seeds_wp, config, out_centerline, out_valid_wp, scratch, *, ctx) -> None``,
       below.
   * - ``capturable``
     - ``True`` (default): ``generate`` obeys the :ref:`hard rules
       <generator-hard-rules>` and is recorded into CUDA graphs. ``False``: see
       :ref:`generator-capture-opt-out`.
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

``generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, *, ctx) -> None``
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
     - The generator's own frozen copy of the configuration it was constructed with.
       Config subclasses must be frozen dataclasses too. With
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
     - The call's ``track_gen.exec.LaunchContext`` (one device, one stream), always
       passed by the pipeline. Emit every launch, copy and fill through it.

The rest of the generator contract is the Stage's: the raw loop, its size and what you
do not have to guarantee (:ref:`generator-stage-contract`), and the
:ref:`hard rules <generator-hard-rules>`, with ``alloc_scratch`` in the place of the
``StageBuilder`` and a constant salt of your own in the place of ``b.salt``.

.. _generator-streams:

Streams and launch contexts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``generate`` must accept the ``ctx=`` keyword and emit every launch, copy and fill
through that :class:`~track_gen.exec.LaunchContext`: ``ctx.launch``, ``ctx.copy``,
``ctx.fill`` or the public ``track_gen.ops`` functions. The context selects the call's
stream, including a caller's capturing stream. Direct launches on Warp's current
stream violate this contract. There is no context-free adapter or ``accepts_ctx`` flag.

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
(pure Warp, no torch) and 4 (fixed-shape buffers) still apply. A ``Stage`` declares ``capture = CaptureClass.HOST`` instead and
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

Public geometric operations
---------------------------

Use :ref:`generator-ops` to fill arrays, sample points, sort anchors, resample loops,
compute turning and detect proper self-crossings. Operations take ``ctx`` first and
write preallocated buffers. :ref:`reference-ops` lists signatures and scratch storage;
:ref:`reference-plugin` specifies registration, loading and schema APIs.
