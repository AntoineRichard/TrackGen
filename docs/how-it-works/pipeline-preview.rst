Pipeline (preview)
==================

.. admonition:: Preview API
   :class: warning

   ``track_gen.pipeline`` is new in 0.2.0.dev3 and is a **preview**: names and signatures
   may change before roadmap PR10b. Only the generator stage can come from a plugin in
   this release; the other stages are TrackGen's own built-ins. ``TrackGenerator`` stays
   the stable entry point, and it builds its pipeline the way this page describes.

The track pipeline of :doc:`pipeline` is described by data. A ``PipelineSpec`` lists the
stages in order, ``build()`` plans and allocates it once, and the ``Program`` it returns
runs the stages, one after another, on every generation. ``TrackGenerator`` turns its
config into the default ``PipelineSpec`` (``legacy.to_pipeline_spec``), builds it, and
runs the ``Program``. The outputs are byte-identical to 0.2.0.dev2, with the same kernel
launches in the same order: the stages call the existing kernels and wrappers, and only
the bookkeeping around them moved.

``track_gen.pipeline`` is a subpackage outside ``track_gen.__all__``: import it
explicitly. ``import track_gen`` does not import it.

.. code-block:: python

   from track_gen import TrackGenConfig
   from track_gen.pipeline import StageRef, build, legacy

   config = TrackGenConfig(num_envs=64, device="cpu")
   spec = legacy.to_pipeline_spec(config)
   spec.ids
   # ('generator', 'resample', 'relax', 'uniform', 'width', 'frames', 'offset',
   #  'elevation', 'validity', 'lift')

   program = build(spec, seeds=1234)     # plan, allocate, warm up: once
   track = program.run(sync=True)        # the same Track instance on every call

   unrelaxed = build(spec.replace("relax", StageRef("relax", "none")))

The data model
--------------

``StageRef(kind, name, params=None, id=None, ports={}, capacity={})``
   One stage slot, frozen. ``kind`` and ``name`` select the stage (``"relax"``,
   ``"xpbd"``). ``params`` is the stage's Params instance, a mapping of its fields, or
   ``None`` for its defaults; a mapping is coerced and validated at build time, like
   ``generator_params``. ``id`` names the slot (``slot_id`` is ``id or kind``): two
   stages of one kind need explicit ids (the default list has two ``resample`` stages,
   ``"resample"`` and ``"uniform"``), and the generator's slot is always
   ``"generator"``. ``ports`` rewires a local port to another channel of the same dtype
   and size. ``capacity`` is the generator's ``{"raw": num_points}``, the DEFAULT raw
   point count (``BatchSpec`` has no ``num_points``); a Stage generator's own
   ``Stage.capacity()`` overrides it. ``build`` rejects any other key, and any
   ``capacity`` on a non-generator stage.

``BatchSpec(num_envs, device, n_max, closed=True, scale=1.0)``
   The batch every stage shares: ``E`` envs on one device, the per-env point capacity
   ``N_max``, and ``TrackGenConfig.scale``. Open tracks (``closed=False``) are refused.

``PipelineSpec(batch, stages)``
   The batch and the ordered tuple of ``StageRef``. ``ids`` lists the slot ids.
   ``replace(id, ref)``, ``insert_after(id, ref)``, ``insert_before(id, ref)`` and
   ``remove(id)`` return a new spec; ``replace`` keeps the slot's id when ``ref.id`` is
   ``None``. They are pure data operations: an unknown anchor id raises ``ValueError``
   at once, everything else is checked by ``build``. ``to_dict()`` returns a
   JSON-serialisable dict (``"format": "track_gen.pipeline/1"``), ``from_dict(d)`` reads
   it back without a registry (a missing key raises a ``ValueError`` naming it, such as
   ``stages[3] is missing 'kind'``), and building the round-tripped spec gives
   byte-identical outputs.

``legacy.to_pipeline_spec(config, *, registry=None)``
   The default spec for a ``TrackGenConfig``, the one ``TrackGenerator`` builds (see
   `The default track stages`_). It resolves the generator's params through
   ``registry``, and raises the ``relax_solver`` and ``smooth_finish`` errors
   ``TrackGenerator`` always raised.

``build()`` and the planner
---------------------------

``build(spec, *, registry=None, seeds=None, out=None, stream=None, prepare=True)``
checks the spec before it allocates or launches anything, then allocates once:

1. **Batch and ids.** ``num_envs >= 1``, ``n_max >= 3``, ``closed=True``; slot ids are
   unique; exactly one ``generator`` stage, first, with the id ``"generator"``; only
   the generator has a ``capacity``, and only ``"raw"``; ``seeds`` is one of the forms
   below.
2. **Resolve.** The generator is resolved through ``registry`` (default
   ``track_gen.plugin.registry``) exactly as ``TrackGenerator`` resolves
   ``config.generator``. Every other stage comes from TrackGen's built-in catalogue.
3. **Params.** Each stage's params are coerced to its Params class and validated with the
   messages ``TrackGenConfig`` uses for the same fields. While the kernels still take
   ``half_width`` as a host scalar, every stage that has a ``half_width`` field must
   agree on it.
4. **Channels.** Walking the stages in order, every channel a stage reads must have been
   written by an earlier stage, and a channel written twice must be read by its second
   writer (an in-place rewrite, see `Versions and the result`_). Only the channels of
   the catalogue below exist, and no stage writes ``seed``. Every ``Track`` field has a
   writer (the ``Track`` reads them all: removing ``validity`` or ``lift`` is refused),
   and ``lift.flat``, which lifts at a constant ``z_base``, may not follow an elevation
   other than ``elevation.flat`` (use ``lift.elevated``).
5. **Allocate.** The ``SeedState`` (its own checks run first), the ``Track`` (or use the
   one passed as ``out=``, checked field by field beforehand), the ``[E]``
   effective-seed buffer, one buffer per channel version and each stage's scratch,
   namespaced ``scratch:<id>/<name>``. Then each stage's ``build`` runs, once, against
   a ``StageBuilder``; right after it, every write port the stage declares must have
   been written.
6. **Program.** With ``prepare=True`` the Program runs once eagerly (the warm-up of
   :doc:`cuda-graph`).

A rule broken in steps 1 to 4 raises ``PlanError``, a ``ValueError`` naming the stage,
its ``kind.name``, the port and the channel, before anything is allocated. Some errors
keep their own type: params raise their own ``ValueError`` / ``TypeError``, an unknown
generator raises the registry's ``ValueError`` (an unknown non-generator stage is a
``PlanError``), and a bad ``seeds`` raises ``TypeError``. A stage whose ``build`` skips
one of its write ports raises ``PlanError`` after allocation, since only the build
shows it. ``build`` inside a capture raises ``CaptureError`` before allocating.

.. code-block:: python

   build(spec.remove("width"))
   # PlanError: stage 'validity' (validity.default) reads 'cl.half_width'
   #   (port 'half_width'), but no earlier stage writes it

``seeds`` is anything ``SeedState`` accepts (``None``, an ``int``, an ``[E]`` int32
``wp.array``, a ``PerEnvSeededRNG``) or a ``SeedState`` for the batch. Anything else,
a NumPy array or a list included, raises ``TypeError`` before allocating.

``build`` derives the generator's config from the ``BatchSpec`` and the generator's
``StageRef`` (its params, and ``capacity["raw"]`` as ``num_points``, 256 when absent).
That covers the built-ins and Stage generators. A ``GeneratorSpec`` plugin that reads
config fields outside its Params class needs ``TrackGenerator``.

The ``Program``
---------------

``Program`` is a ``GraphRunner`` launchable, so it follows the record / eager /
self-graph rules of :doc:`cuda-graph` exactly as ``TrackGenerator`` does.

- ``run(*, stream=None, sync=False, advance=False, mask=None, policy="auto")`` runs one
  generation: recorded into the caller's capture, replayed from the Program's own graph,
  or eager. ``advance`` and ``mask`` mean what they mean for ``generate()``.
- ``enqueue(ctx, *, advance=False, mask=None)`` launches one generation on ``ctx``: the
  seed prologue (advance, effective seed, parameter latch), every stage in order, then
  the ``generation`` write. It allocates nothing and never blocks.
- ``prepare()`` runs the warm-up, when ``build(..., prepare=False)``. Otherwise the
  first ``run()`` outside a capture runs it, and recording a Program that was never
  prepared (``run()`` or ``enqueue()`` inside a capture) raises ``CaptureError``.
- ``result`` is the ``Track``, the same instance with the same array addresses forever.
- ``spec`` is the resolved spec: every ``StageRef.params`` is the stage's Params
  instance.
- ``seeds`` (the ``SeedState``), ``seed`` (the ``[E]`` effective seed), ``inputs``
  (``"regen_mask"`` plus the inputs the stages declare), ``params`` (a ``ParamTable``
  keyed by stage id; the built-in stages' fields are all static), ``graph``,
  ``capturable`` and ``capture_class`` (``STATIC``, or ``HOST`` for a non-capturable
  generator).

Channels
--------

Stages talk through named per-env channels; a stage declares the ones it reads and
writes as ``{local port: channel}`` maps (``Stage.reads`` / ``Stage.writes``), and
``StageBuilder.read(port)`` / ``write(port, dtype, per=...)`` hand it the buffers.
``per="env"`` is ``[E]``, ``per="cl"`` is ``[E * N_max]``, ``per="raw"`` is
``[E * raw]``. The catalogue in this release:

.. list-table::
   :header-rows: 1
   :widths: 26 10 10 54

   * - channel
     - per
     - dtype
     - meaning (``Track`` field it becomes)
   * - ``seed``
     - env
     - int32
     - The effective seed, written by the Program's prologue; no stage writes it. Stages
       read it through a ``seed`` read port (``StageBuilder.seed`` is the same
       buffer).
   * - ``raw.xy``
     - raw
     - vec2f
     - The generator's raw closed centerline.
   * - ``gen.valid``
     - env
     - int32
     - The generator's per-env flag, ANDed into validity.
   * - ``cl.xy``
     - cl
     - vec2f
     - The centerline: constant-spacing, then relaxed, then re-uniformised (three
       versions).
   * - ``cl.count``
     - env
     - int32
     - Real points per env; NaN padding after.
   * - ``cl.truncated``
     - env
     - int32
     - ``1`` where ``N_max`` cut the loop (``Track.truncated``).
   * - ``cl.half_width``
     - cl
     - float32
     - Per-point half-width.
   * - ``cl.tangent2``, ``cl.normal2``
     - cl
     - vec2f
     - Plan-view frames.
   * - ``cl.kappa``
     - cl
     - float32
     - Curvature.
   * - ``cl.arclen2``, ``cl.length2``
     - cl, env
     - float32
     - Plan-view cumulative arc length and loop length (``Track.arclen``,
       ``Track.length`` when no stage writes the 3D tables).
   * - ``cl.winding``
     - env
     - float32
     - Signed winding (``Track.winding``).
   * - ``band.outer2``, ``band.inner2``
     - cl
     - vec2f
     - Plan-view borders.
   * - ``cl.z``
     - cl
     - float32
     - Per-point elevation (zero on ``flat``).
   * - ``track.count``, ``track.valid``
     - env
     - int32
     - ``Track.count``, ``Track.valid``.
   * - ``track.arclen``, ``track.length``
     - cl, env
     - float32
     - The 3D arc length and loop length of the lifted centerline (``Track.arclen``,
       ``Track.length``), written by ``lift.elevated``.
   * - ``track.outer``, ``track.center``, ``track.inner``, ``track.tangent``,
       ``track.normal``
     - cl
     - vec3f
     - The lifted 3D geometry (the ``Track`` fields of the same names).

The default track stages
------------------------

``legacy.to_pipeline_spec(config)`` produces this list. Every stage launches what the
0.2.0.dev2 pipeline launched at the same point, with the same arguments.

.. list-table::
   :header-rows: 1
   :widths: 10 22 30 38

   * - id
     - stage
     - reads
     - writes
   * - ``generator``
     - ``generator.<config.generator>``
     - ``seed``
     - ``raw.xy``, ``gen.valid``
   * - ``resample``
     - ``resample.constant_spacing``
     - ``raw.xy``
     - ``cl.xy``, ``cl.count``, ``cl.truncated``
   * - ``relax``
     - ``relax.xpbd``, or ``relax.none`` when ``relax_enable=False``
     - ``cl.xy``, ``cl.count`` (``relax.none``: nothing)
     - ``cl.xy`` (``relax.none``: nothing)
   * - ``uniform``
     - ``resample.uniform``
     - ``cl.xy``, ``cl.count``
     - ``cl.xy``
   * - ``width``
     - ``width.constant``
     - nothing
     - ``cl.half_width``
   * - ``frames``
     - ``frames.default``
     - ``cl.xy``, ``cl.count``
     - ``track.count``, ``cl.tangent2``, ``cl.normal2``, ``cl.kappa``, ``cl.arclen2``,
       ``cl.length2``, ``cl.winding``
   * - ``offset``
     - ``offset.default``
     - ``cl.xy``, ``cl.normal2``, ``cl.count``
     - ``band.outer2``, ``band.inner2``
   * - ``elevation``
     - ``elevation.<config.z_profile>``: ``flat``, ``uniform``, ``random_walk`` or
       ``noise``
     - ``seed``, ``cl.count``, ``cl.arclen2``, ``cl.length2`` (``flat``: nothing)
     - ``cl.z`` (``flat`` launches nothing: it stays zero)
   * - ``validity``
     - ``validity.default``
     - ``cl.xy``, ``cl.half_width``, ``cl.count``, ``gen.valid``, ``band.outer2``,
       ``band.inner2``, ``cl.z``, ``cl.truncated``
     - ``track.valid``
   * - ``lift``
     - ``lift.flat`` on ``z_profile="flat"``, else ``lift.elevated``
     - ``band.outer2``, ``cl.xy``, ``band.inner2``, ``cl.normal2``; ``lift.flat`` also
       ``cl.tangent2``; ``lift.elevated`` also ``cl.z``, ``cl.count``
     - ``track.outer``, ``track.center``, ``track.inner``, ``track.tangent``,
       ``track.normal``; ``lift.elevated`` also ``track.arclen``, ``track.length``

Each stage's params keep the flat ``TrackGenConfig`` field names, copied from the config:
``resample.constant_spacing`` takes ``spacing``; ``relax.xpbd`` takes ``half_width`` and
the ``relax_*`` fields; ``width.constant`` and ``offset.default`` take ``half_width``;
the elevation stages take the ``z_*`` fields their profile uses; ``validity.default``
takes ``half_width``, ``turning_tol``, ``w_floor``, ``relax_tol``, ``z_valid_grade`` and
``validity_border_check``; ``lift.flat`` takes ``z_base``. The generator's params are its
``generator_params`` (see :ref:`generator-params`).

Versions and the result
-----------------------

A stage that both reads and writes a channel (``relax.xpbd`` and ``resample.uniform``)
rewrites it in place in the data flow, not in memory: its ``b.write(port)`` returns a
new buffer, its ``b.read(port)`` the previous version, and later stages see the new
one. The pre-relax centerline therefore survives the relax stage, and ``relax.none``
(or removing the slot) leaves ``resample.uniform`` reading the constant-spacing
version. A second writer that does not read the channel is a ``PlanError``.

Such a stage must write its whole output into the buffer ``b.write(port)`` returns:
the version it reads is the previous stage's and is never seen downstream, so an
update made there (for example OR-ing a bit into a read mask, or copying a result back
into the read buffer) is lost. A form that aliases the read and write buffers is a
roadmap PR10b decision.

The ``Track`` fields are bound to the final version of their channel: the stage that
writes ``track.valid``, ``cl.truncated`` or ``track.arclen`` writes straight into the
``Track`` array, with no copy. ``Track.arclen`` and ``Track.length`` come from
``track.arclen`` / ``track.length`` when a stage writes them (``lift.elevated``), and
otherwise from the plan-view ``cl.arclen2`` / ``cl.length2`` (``lift.flat``: a constant
``z_base`` changes neither). Intermediate versions get buffers of their own.
``Track.generation`` is not a channel: the Program's epilogue writes it.

Not open yet
------------

In this preview, ``build`` places TrackGen's own non-generator stages only, and
``registry.add`` still rejects a ``Stage`` whose ``kind`` is not ``"generator"``.
Roadmap PR10b opens the ``relax``, ``width``, ``elevation``, ``validity`` and commit
kinds to plugins, adds presets with replace / insert, a masked commit that writes only
the masked envs, ``on_invalid``, ``Program.tap()`` and ``StageBuilder.fail_bit()``. The
gate pipeline stays as it is until roadmap PR11. Consumers of ``cl.half_width`` (a width
that varies along the track) arrive with roadmap PR13.

The 0.2.0.dev2 composition (``warp_pipeline._run_pipeline``, ``inflate_warp``,
``_inflate_warp_alloc``) is unchanged and still importable. ``TrackGenerator`` no longer
calls it; the tests use it as the byte-identity reference.
