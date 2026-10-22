Pipeline (preview)
==================

.. admonition:: Preview API
   :class: warning

   ``track_gen.pipeline`` is a **preview**: names and signatures may change before 1.0.
   ``TrackGenerator`` stays the stable entry point. It accepts a ``PipelineSpec`` as well
   as a ``TrackGenConfig``, and builds its pipeline the way this page describes.

The track pipeline of :doc:`pipeline` is described by data. A ``PipelineSpec`` lists the
stages in order, ``build()`` plans and allocates it once, and the ``Program`` it returns
runs the stages, one after another, on every generation; a short epilogue, the *commit*,
then settles what each env's ``Track`` rows hold. ``TrackGenerator`` turns a config into
the default ``PipelineSpec`` (``legacy.to_pipeline_spec``), or takes a spec you built,
usually with ``presets.track``.

Four stage kinds are open to plugins: ``relax``, ``width``, ``elevation`` and
``validity``. They are registered, discovered and quarantined like generators (see
:doc:`/contributing/writing-a-stage`); the other kinds (``resample``, ``frames``,
``offset``, ``lift``) are TrackGen's own built-ins. The default pipeline's outputs are
byte-identical to 0.2.0.dev2, with the same launches in the same order; a masked
regeneration adds two (`The commit`_).

``track_gen.pipeline`` is a subpackage outside ``track_gen.__all__``: import it
explicitly. ``import track_gen`` does not import it.

.. code-block:: python

   from track_gen import PerEnvSeededRNG, TrackGenerator
   from track_gen.pipeline import StageRef, presets

   spec = presets.track(num_envs=64, device="cpu")
   spec.ids
   # ('generator', 'resample', 'relax', 'uniform', 'width', 'frames', 'offset',
   #  'elevation', 'validity', 'lift')

   rng = PerEnvSeededRNG(seeds=1234, num_envs=64, device="cpu")
   gen = TrackGenerator(spec, rng)
   track = gen.generate()              # the same Track instance on every call
   track.fail_mask                     # [E] int32: one bit per failed check
   track.fail_bits                     # {'generator': 0, 'turning': 1, ..., 'truncation': 7}

   # Other stages: by name, (name, params), a StageRef, or None.
   hilly = presets.track(num_envs=64, device="cpu", relax=None,
                         elevation=("noise", {"z_noise_amplitude": 0.3}))
   # Edit operations on any spec:
   strict = spec.insert_after("validity",
                              StageRef("validity", "acme.max_curvature", id="kmax"))

   gen.program.tap("cl.xy", after="resample")   # an intermediate channel, read-only

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

``BatchSpec(num_envs, device, n_max, closed=True, scale=1.0, on_invalid="keep")``
   The batch every stage shares: ``E`` envs on one device, the per-env point capacity
   ``N_max``, ``TrackGenConfig.scale``, and what the commit does with an invalid env
   (``"keep"`` or ``"poison"``, see `The commit`_). Open tracks (``closed=False``) are
   refused.

``PipelineSpec(batch, stages)``
   The batch and the ordered tuple of ``StageRef``. ``ids`` lists the slot ids.
   ``replace(id, ref)``, ``insert_after(id, ref)``, ``insert_before(id, ref)`` and
   ``remove(id)`` return a new spec; ``replace`` keeps the slot's id when ``ref.id`` is
   ``None``. They are pure data operations: an unknown anchor id raises ``ValueError``
   at once, everything else is checked by ``build``. ``to_dict()`` returns a
   JSON-serialisable dict (``"format": "track_gen.pipeline/1"``; ``on_invalid`` is
   optional when reading, ``"keep"`` when absent), ``from_dict(d)`` reads it back without
   a registry (a missing key raises a ``ValueError`` naming it, such as ``stages[3] is
   missing 'kind'``), and building the round-tripped spec gives byte-identical outputs,
   plugin stages and their params included.

``legacy.to_pipeline_spec(config, *, registry=None)``
   The default spec for a ``TrackGenConfig``, the one ``TrackGenerator(config)`` builds
   (see `The default track stages`_). It resolves the generator's params, and the
   ``relax_solver`` and ``z_profile`` names (see :ref:`flat-stage-shortcuts`), through
   ``registry``, and raises the ``relax_solver``, ``smooth_finish`` and ``z_profile``
   errors of ``TrackGenerator(config)``.

.. _pipeline-presets:

Presets
-------

``presets.track(...)`` returns the ``PipelineSpec`` of a track pipeline, with each
replaceable stage chosen by a keyword. It is pure data: it never touches the registry,
and an unknown stage name fails at ``build`` (or ``TrackGenerator``) time.

.. code-block:: python

   presets.track(*, num_envs, device, n_max=384, scale=1.0,
                 generator="bezier", generator_params=None, num_points=256,
                 half_width=0.1, spacing=None,
                 relax="xpbd", width="constant", elevation="flat",
                 checks=("default",), on_invalid="keep",
                 z_base=0.0) -> PipelineSpec

The stages, by id: ``generator``, ``resample`` (``resample.constant_spacing``),
``relax``, ``uniform`` (``resample.uniform``), ``width``, ``frames``, ``offset``,
``elevation``, the ``checks`` (the first one's id is ``"validity"``), and ``lift``.
``presets.track(num_envs=E, device=d)`` builds what
``legacy.to_pipeline_spec(TrackGenConfig(num_envs=E, device=d))`` builds: the same stage
list, equal params once resolved, byte-identical outputs.

``relax``, ``width``, ``elevation`` and each entry of ``checks`` take one of:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Form
     - Meaning
   * - ``"name"``
     - The stage registered as ``kind:name`` (``"xpbd"``, ``"acme.taubin"``), with its
       default params. ``"default"`` in ``checks`` is ``validity.default``.
   * - ``("name", params)``
     - The same with ``params``: a Params instance, a mapping of its fields, or ``None``.
   * - ``StageRef(kind, name, ...)``
     - Used as given (id and ports included). Its ``kind`` must match the keyword
       (``ValueError`` otherwise); without an id it takes the slot's.
   * - ``None``
     - ``relax=None`` is ``relax.none``, ``elevation=None`` is ``elevation.flat``.
       ``width=None`` raises ``ValueError``: something must write ``cl.half_width``.

Anything else raises ``TypeError``, and an empty name ``ValueError``. ``checks`` is a
sequence of selections whose first is ``"default"`` (``validity.default`` starts
``valid.fail``; plugin checks follow it): a single selection raises ``TypeError``, an
empty sequence or another first check ``ValueError``. The other arguments:

- ``generator`` is a generator name, with ``generator_params`` (as
  ``TrackGenConfig.generator_params``), or a ``StageRef(kind="generator", ...)``, which
  carries its own params (passing ``generator_params`` too raises ``ValueError``).
  ``num_points`` becomes the generator slot's ``capacity={"raw": num_points}`` (unless
  the ``StageRef`` sets a capacity).
- ``half_width`` is filled into the built-in stages that bake it as a host scalar
  (``relax.xpbd``, ``width.constant``, ``offset.default``, ``validity.default``) when
  their params, in any selection form, are ``None`` or a mapping without
  ``half_width``. A Params instance is taken as is. A plugin stage with a
  ``half_width`` field must be given the same value: the planner requires every stage
  with that field to agree, until roadmap PR13 makes the kernels read
  ``cl.half_width``. Params (of any stage) that set another ``half_width`` than the
  preset's raise ``ValueError`` naming ``half_width=``.
- ``spacing`` defaults to ``0.6 * half_width``, as in ``TrackGenConfig``.
- ``lift`` is ``lift.flat`` (at ``z_base``, default ``0``, as
  ``TrackGenConfig.z_base``) when the elevation is ``elevation.flat``, else
  ``lift.elevated`` (a nonzero ``z_base`` then raises ``ValueError``: the elevation's
  own params set its altitude).
- Each further check gets the id ``ref.id or name``. Two checks with the same id raise
  ``ValueError``: give one an explicit ``StageRef(..., id=...)``.
- ``on_invalid`` is ``"keep"`` or ``"poison"`` (``ValueError`` otherwise).

A preset is a starting point: use the edit operations for the rest, for example
``presets.track(...).insert_after("validity", StageRef("validity",
"acme.max_curvature", id="kmax"))`` or ``.replace("resample", StageRef("resample",
"constant_spacing", {"spacing": 0.05}))``.

Running a spec: ``TrackGenerator(spec)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TrackGenerator(config_or_spec, rng=None, *, seeds=None, stream=None)`` takes a
``TrackGenConfig`` (as before) or a ``PipelineSpec``. Seeds come from ``rng`` (a
``PerEnvSeededRNG``) or ``seeds`` (any form ``SeedState`` accepts, see
`build() and the planner`_), not both (``ValueError``); with neither, env ``e`` gets seed
``e``. Anything other than a config or a spec raises ``TypeError``.

- ``gen.config`` returns what was passed (the config snapshot, or the spec), so
  ``TrackGenerator(gen.config, rng)`` rebuilds an identical generator.
- ``gen.params`` lists ``"generator"`` plus every stage with at least one *dynamic*
  field (``["generator", "relax"]`` with a relax stage that has one); see
  :ref:`stage-runtime-params`.
- ``gen.program`` is the ``Program`` (preview): its taps, fail bits and ``on_invalid``.

Everything else (``generate``, ``enqueue``, ``seeds``, ``inputs``, graphs, masked
regeneration) works as for a config. ``Course`` still takes a ``TrackGenConfig`` only.

.. _pipeline-stage-kinds:

Stage kinds
-----------

.. list-table::
   :header-rows: 1
   :widths: 14 22 26 38

   * - kind
     - built-ins
     - may write
     - where it comes from
   * - ``generator``
     - ``bezier``, ``hull``, ... (:doc:`/generators/overview`)
     - ``raw.xy``, ``gen.valid``
     - ``track_gen.plugin.registry`` (:doc:`/contributing/writing-a-generator`)
   * - ``relax``
     - ``xpbd``, ``none``
     - ``cl.xy``
     - ``track_gen.plugin.registry``
   * - ``width``
     - ``constant``
     - ``cl.half_width``
     - ``track_gen.plugin.registry``
   * - ``elevation``
     - ``flat``, ``uniform``, ``random_walk``, ``noise``
     - ``cl.z``
     - ``track_gen.plugin.registry``
   * - ``validity``
     - ``default``
     - ``valid.fail``
     - ``track_gen.plugin.registry``
   * - ``resample``, ``frames``, ``offset``, ``lift``
     - see `The default track stages`_
     - their catalogue channels
     - TrackGen's private catalogue (not open to plugins)

A plugin stage of an open kind may also write namespaced plugin channels,
``cl.<package>.<name>`` (see `Channels`_). The built-ins are registered under their
plain names by TrackGen's own ``register(reg)``, exactly as a plugin registers, so
``registry.available("relax")`` lists them with the installed plugins and
``registry.describe("elevation", "noise")`` reports their reads, writes and params
schema.

The planner resolves a stage of an open kind with ``registry.get(kind, name)`` when it
builds, and keeps that class: a later registry change never alters a built ``Program``.
An unknown name raises ``PlanError`` with the registry's list, for example
``stage 'relax' (relax.nope): unknown relax 'nope'; available: ['none', 'xpbd']``; a
name served by a quarantined plugin raises its ``PluginLoadError``. An unknown stage of
a closed kind lists the catalogue's names and says that the kind is not open to plugins.

``build()`` and the planner
---------------------------

``build(spec, *, registry=None, seeds=None, out=None, stream=None, prepare=True)``
checks the spec before it allocates or launches anything, then allocates once:

1. **Batch and ids.** ``num_envs >= 1``, ``n_max >= 3``, ``closed=True``,
   ``on_invalid`` in ``("keep", "poison")``; slot ids are unique; exactly one
   ``generator`` stage, first, with the id ``"generator"``; only the generator has a
   ``capacity``, and only ``"raw"``; ``seeds`` is one of the forms below.
2. **Resolve.** The generator and every stage of an open kind are resolved through
   ``registry`` (default ``track_gen.plugin.registry``), the generator exactly as
   ``TrackGenerator`` resolves ``config.generator``. The other stages come from
   TrackGen's built-in catalogue.
3. **Params.** Each stage's params are coerced to its Params class and validated (the
   built-ins with the messages ``TrackGenConfig`` uses for the same fields). While the
   kernels still take ``half_width`` as a host scalar, every stage that has a
   ``half_width`` field must agree on it.
4. **Channels.** After ``ports`` rewiring, the registration rules hold again (a plugin
   stage writes only what its kind may write, and rewrites a channel through one port;
   see :ref:`stage-ports-channels`). Walking the stages in order, every channel a stage
   reads must have been written by an earlier stage, and a channel written twice must
   be read by its second writer (an in-place rewrite, see `In-place rewrites`_). No
   stage writes ``seed``. Every ``Track`` field has a writer (the ``Track`` reads them
   all: removing ``validity`` or ``lift`` is refused), ``valid.fail`` starts at
   ``validity.default`` (a check that writes it without reading it would drop the
   built-in checks, truncation included, and is refused until the split checks
   arrive), and ``lift.flat``, which lifts at a constant ``z_base``, may not follow an
   elevation other than ``elevation.flat`` (use ``lift.elevated``).
5. **Capture class.** The Program's capture class is the highest of its stages'
   (``STATIC`` < ``HOST``): one ``HOST`` stage makes the whole Program run eagerly and
   refuse caller capture. A stage that returns ``BOUNDED`` raises ``PlanError`` (roadmap
   PR12).
6. **Allocate.** The ``SeedState`` (its own checks run first), the ``Track`` (or use the
   one passed as ``out=``, checked field by field beforehand; its ``generation`` and
   ``fail_mask`` may be ``None``), the ``[E]`` effective-seed buffer, one buffer per
   channel version and each stage's scratch, namespaced ``scratch:<id>/<name>``. Then
   each stage's ``build`` runs, once, in pipeline order, against a ``StageBuilder``;
   right after it, every write port the stage declares must have been written, and a
   validity stage must own a fail bit (fail bits are handed out in this order, see
   `Fail bits`_). Last, the commit and its held copy of the ``Track``.
7. **Program.** With ``prepare=True`` the Program runs once eagerly (the warm-up of
   :doc:`cuda-graph`).

A rule broken in steps 1 to 5 raises ``PlanError``, a ``ValueError`` naming the stage,
its ``kind.name``, the port and the channel, before anything is allocated. Some errors
keep their own type: params raise their own ``ValueError`` / ``TypeError``, an unknown
generator raises the registry's ``ValueError``, a quarantined plugin its
``PluginLoadError``, and a bad ``seeds`` raises ``TypeError``. The rules only a stage's
``build`` reveals (a skipped write port, a validity stage without a fail bit, a 16th
plugin fail bit, a fail-bit name taken twice) raise ``PlanError`` after allocation.
``build`` inside a capture raises ``CaptureError`` before allocating.

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
config fields outside its Params class needs ``TrackGenerator`` with a config.

The ``Program``
---------------

``Program`` is a ``GraphRunner`` launchable, so it follows the record / eager /
self-graph rules of :doc:`cuda-graph` exactly as ``TrackGenerator`` does.

- ``run(*, stream=None, sync=False, advance=False, mask=None, policy="auto")`` runs one
  generation: recorded into the caller's capture, replayed from the Program's own graph,
  or eager. ``advance`` and ``mask`` mean what they mean for ``generate()``.
- ``enqueue(ctx, *, advance=False, mask=None)`` launches one generation on ``ctx``: the
  seed prologue (advance, effective seed, parameter latch), with a mask the hold of the
  unmasked envs' ``Track`` rows, every stage in order for every env, the commit, and the
  ``generation`` write. It allocates nothing and never blocks.
- ``prepare()`` runs the warm-up, when ``build(..., prepare=False)``. Otherwise the
  first ``run()`` outside a capture runs it, and recording a Program that was never
  prepared (``run()`` or ``enqueue()`` inside a capture) raises ``CaptureError``.
- ``result`` is the ``Track``, the same instance with the same array addresses forever.
- ``spec`` is the resolved spec: every ``StageRef.params`` is the stage's Params
  instance.
- ``tap(channel, *, after=None)``, ``channels``, ``fail_bits`` and ``on_invalid``: see
  `Taps`_, `Fail bits`_ and `The commit`_.
- ``seeds`` (the ``SeedState``), ``seed`` (the ``[E]`` effective seed), ``inputs``
  (``"regen_mask"`` plus the inputs the stages declare), ``params`` (a ``ParamTable``
  keyed by stage id; the built-in stages' fields are all static), ``graph``,
  ``capturable`` and ``capture_class`` (``STATIC``, or ``HOST`` when a stage or the
  generator is not capturable).

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
     - The generator's per-env flag (fail bit 0 when ``0``).
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
     - Per-point half-width. In this release only ``validity.default``'s width-floor
       check reads it; the band offset and XPBD read it from roadmap PR13.
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
   * - ``valid.fail``
     - env
     - int32
     - The per-env fail bitmask: bit ``i`` set means check ``i`` failed
       (``Track.fail_mask``; ``Track.valid`` is ``fail_mask == 0``). See `Fail bits`_.
   * - ``track.count``
     - env
     - int32
     - ``Track.count``.
   * - ``track.valid``
     - env
     - int32
     - ``validity.default``'s own 0/1 flag: ``Track.valid`` when no check runs after
       ``validity.default``. Otherwise the commit derives ``Track.valid`` from
       ``valid.fail``; the two are equal when no other check runs.
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

**Plugin channels.** A plugin stage may also read and write channels of its own, named
``cl.<package>.<name>`` (at least two dot-separated identifiers after ``cl.``, such as
``cl.acme.curvature_rank``), for example to hand a buffer from one of its stages to
another. Their dtype and size are fixed by their first writer's ``b.write(port, dtype,
per=...)`` (``per`` is ``"env"`` or ``"cl"``); a later writer must match, and reading one
needs an earlier writer like any channel. They are not copied to the ``Track``
(``Track.attributes`` arrives with roadmap PR13); read them with ``Program.tap``.

The default track stages
------------------------

``legacy.to_pipeline_spec(config)`` and ``presets.track(...)`` produce this list. Every
stage launches what the 0.2.0.dev2 pipeline launched at the same point, with the same
arguments.

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
     - ``relax.<config.relax_solver>`` (``relax.xpbd``), or ``relax.none`` when
       ``relax_enable=False``
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
     - ``elevation.<config.z_profile>``: ``flat``, ``uniform``, ``random_walk``,
       ``noise`` or a plugin's
     - ``seed``, ``cl.count``, ``cl.arclen2``, ``cl.length2`` (``flat``: nothing)
     - ``cl.z`` (``flat`` launches nothing: it stays zero)
   * - ``validity``
     - ``validity.default``
     - ``cl.xy``, ``cl.half_width``, ``cl.count``, ``gen.valid``, ``band.outer2``,
       ``band.inner2``, ``cl.z``, ``cl.truncated``
     - ``track.valid``, ``valid.fail``
   * - ``lift``
     - ``lift.flat`` on ``z_profile="flat"``, else ``lift.elevated``
     - ``band.outer2``, ``cl.xy``, ``band.inner2``, ``cl.normal2``; ``lift.flat`` also
       ``cl.tangent2``; ``lift.elevated`` also ``cl.z``, ``cl.count``
     - ``track.outer``, ``track.center``, ``track.inner``, ``track.tangent``,
       ``track.normal``; ``lift.elevated`` also ``track.arclen``, ``track.length``

Each built-in stage's params keep the flat ``TrackGenConfig`` field names, copied from
the config: ``resample.constant_spacing`` takes ``spacing``; ``relax.xpbd`` takes
``half_width`` and the ``relax_*`` fields; ``width.constant`` and ``offset.default`` take
``half_width``; the elevation stages take the ``z_*`` fields their profile uses;
``validity.default`` takes ``half_width``, ``turning_tol``, ``w_floor``, ``relax_tol``,
``z_valid_grade`` and ``validity_border_check``; ``lift.flat`` takes ``z_base``. The
generator's params are its ``generator_params`` (see :ref:`generator-params`). A plugin
stage named by ``relax_solver`` or ``z_profile`` gets its ``Params.from_legacy(config)``
(its defaults unless it maps flat fields).

.. _pipeline-inplace:

In-place rewrites
-----------------

A stage that both reads and writes a channel (``relax.xpbd``, ``resample.uniform``, a
chained validity check) *rewrites* it: in place in the data flow, not in memory. There
is one model, versioned rewrites, and no aliasing form:

1. **One port.** A rewrite declares the channel in ``reads`` and in ``writes`` through
   the same port name: ``reads = {"xy": "cl.xy", ...}``, ``writes = {"xy": "cl.xy"}``.
   Reading a channel through one port and writing it through another is refused, at
   registration and again at plan time after ``ports`` rewiring; a port present in both
   maps must name one channel.
2. **Two buffers.** ``b.read(port)`` returns the version the last earlier writer
   produced; ``b.write(port, ...)`` returns a NEW buffer, allocated once at build and
   zero-initialised, which every later stage reads. The version read is never handed to
   a later stage again.
3. **Whole output.** The stage writes its whole output into the write buffer (every
   ``[E * N_max]`` slot of a per-point channel, NaN padding included; every env of an
   ``[E]`` channel) and never writes a read buffer. An update made only to the read
   buffer is lost, and it corrupts that version as ``tap(after=...)`` shows it.
   ``check_stage``'s ``stage.isolation`` check catches it.
4. **One first writer.** A second writer of a channel that does not read it is a
   ``PlanError``; a stage has at most one write port per channel and never writes
   ``seed``.
5. **Scratch through the builder.** Ping-pong partners and temporaries come from
   ``b.temp_like(port)`` / ``b.scratch(name, ...)``, allocated once and namespaced.

So the pre-relax centerline survives the relax stage (``tap("cl.xy",
after="resample")``), and ``relax.none`` (or removing the slot) leaves
``resample.uniform`` reading the constant-spacing version. Byte identity with 0.2.0.dev2
follows by construction: the built-ins read and write exactly the buffers they did
before. The price is one buffer per rewrite. An iterative stage needs no extra copy: a
ping-pong between ``b.temp_like(port)`` and the write buffer with an even number of
steps ends in the write buffer (see :doc:`/contributing/writing-a-stage`), and a chained
check is one kernel, ``fail_out[e] = fail_in[e] | bit``.

.. _pipeline-fail-bits:

Fail bits
---------

Validity is a bitmask, the ``[E]`` int32 channel ``valid.fail``: bit ``i`` is set when
check ``i`` failed for that env. ``validity.default`` writes the first version (it
writes ``valid.fail`` without reading it), and each further validity stage rewrites it,
OR-ing in the bits it owns. The final version is ``Track.fail_mask``, and
``Track.valid == (fail_mask == 0)``: ``validity.default``'s flag when it is the last
check, else derived by the commit.

.. list-table::
   :header-rows: 1
   :widths: 12 18 70

   * - bit
     - name
     - set when
   * - 0
     - ``generator``
     - the generator wrote ``gen.valid = 0``
   * - 1
     - ``turning``
     - the turning number is not ``±2π`` within ``turning_tol`` (always set for an env
       with ``count == 0``)
   * - 2
     - ``finite``
     - a real centerline point is NaN or infinite
   * - 3
     - ``thickness``
     - the loop's thickness is below ``(1 - relax_tol) * half_width``
   * - 4
     - ``w_floor``
     - a real point's ``cl.half_width`` is not above ``w_floor``
   * - 5
     - ``border``
     - a border self-intersects (only with ``validity_border_check=True``)
   * - 6
     - ``grade``
     - a segment's ``|dz| / ds`` exceeds ``z_valid_grade`` (only when it is ``> 0``)
   * - 7
     - ``truncation``
     - ``N_max`` cut the loop (``Track.truncated``)
   * - 8-15
     - reserved
     - TrackGen's future built-in checks
   * - 16-30
     - plugin checks
     - assigned by ``StageBuilder.fail_bit()``, in pipeline order
   * - 31
     - unused
     - never set (the sign bit)

``validity.default`` owns bits 0 to 7 and keeps the single fused kernel of 0.2.0.dev2:
the kernel gained one output and still computes ``track.valid`` with the same
expression, so ``Track.valid`` is byte-identical. An env with ``count == 0`` gets bit 1
plus bits 0 and 7 when they apply. ``track_gen.pipeline.BUILTIN_FAIL_BITS`` is the
name-to-bit table above (a read-only mapping); a name is listed even when its check is
off (bits 5 and 6), and is then never set.

A validity plugin calls ``b.fail_bit(name=None)`` in its ``build`` for each independent
check it runs and gets bits 16, 17, ... in pipeline order; the name (default: the stage
id) keys the bit in ``Program.fail_bits`` and ``Track.fail_bits``. ``fail_bit()`` from a
stage of another kind raises ``ValueError``; a 16th plugin bit in one pipeline, or a
name that is already taken, raises ``PlanError``; a validity stage that never calls it
raises ``PlanError`` (a check owns at least one bit). Bits 0 to 15 are reserved for
TrackGen's own checks.

.. code-block:: python

   spec = presets.track(num_envs=64, device="cpu",
                        checks=("default", ("acme.max_curvature", {"kappa_max": 8.0})))
   gen = TrackGenerator(spec, rng)
   gen.program.fail_bits["acme.max_curvature"]    # 16
   track = gen.generate()
   too_curvy = (track.fail_mask.numpy() >> 16) & 1

``Track.fail_mask`` is ``None`` and ``Track.fail_bits`` empty on a ``Track`` built
elsewhere (the standalone ``inflate_warp`` path, for example). When ``fail_mask`` is set,
``valid[e] == (fail_mask[e] == 0)`` for every env.

.. _pipeline-commit:

The commit
----------

The ``Track`` arrays are the final versions of the channels it exports: the stage that
last writes ``track.outer`` (``lift``), ``cl.truncated`` (``resample``), ``cl.winding``
(``frames``) or ``valid.fail`` (the last check) writes straight into the ``Track`` array,
with no copy. ``Track.arclen`` / ``Track.length`` are ``track.arclen`` /
``track.length`` when a stage writes them (``lift.elevated``), else the plan-view
``cl.arclen2`` / ``cl.length2`` (``lift.flat``: a constant ``z_base`` changes neither);
``Track.fail_mask`` is the final ``valid.fail``; ``Track.valid`` is ``validity.default``'s
``track.valid`` unless a check runs after it. Intermediate versions get buffers of their
own. ``Track.generation`` is not a channel.

The *commit* is the Program's epilogue: it runs after the stages and settles what each
env's ``Track`` rows hold at the end of the generation. A generation is: advance the
masked epochs, compute the effective seeds, latch the parameters, with a mask hold the
unmasked envs' rows, run every stage for every env, commit, write ``generation``.

**Masked regeneration is exact.** The stages have fixed shapes, so under
``generate(advance=True, mask=m)`` they still recompute, and write, every env's rows.
Before them, a kernel copies the ``Track`` rows of every env with ``m[e] == 0`` into a
held copy; after them, the commit copies those rows back, bit for bit. An env outside
the mask ends the generation with exactly the bytes it had, ``generation`` included
(written for the masked envs only), whatever its recomputation gave: masked regeneration
no longer relies on the stages being bitwise deterministic, and an all-zero mask is an
exact no-op.

This is a hold / restore, not a commit that alone writes the ``Track``: within one
generation the stages overwrite the unmasked envs' rows with their recomputation, and
the commit then restores them, both on the generation's stream. A consumer ordered
after the generation on that stream (later work on it, or work waiting on an event
recorded on it, as in a caller's graph) never observes the recomputed rows. A consumer
on another stream with no such dependency can: for a while an unmasked env may show its
recomputed track (after a reseed, a different one; under ``on_invalid="poison"``, a
poisoned env with ``count > 0`` and finite geometry). Order such readers after the
generation, as for any write to the ``Track``.

**Invalid envs.** ``BatchSpec.on_invalid`` (``presets.track(on_invalid=...)``) chooses
what the commit does with an env whose ``fail_mask`` is not zero:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - value
     - an invalid env leaves the commit with
   * - ``"keep"`` (default)
     - its computed geometry, as in 0.2.0.dev2, and ``valid = 0``.
   * - ``"poison"``
     - ``count = 0``; ``outer``, ``center``, ``inner``, ``tangent``, ``normal`` all
       ``(NaN, NaN, NaN)`` and ``arclen`` NaN on every slot; ``length = 0`` and
       ``winding = 0`` (what an empty loop gives); ``valid = 0``. ``fail_mask`` and
       ``truncated`` keep the computed values, so the cause stays readable, and
       ``generation`` is written as usual.

With ``"poison"``, a utility fed the ``Track`` meets an empty track for an invalid env
instead of finite geometry that failed a check, so its ``count < 3`` handling runs: for
that env ``TrackLocalizer`` returns a NaN ``s`` and ``segment = -1``,
``CollisionChecker`` reports ``oob = 1`` with ``boundary = -1`` and a NaN distance, and
``CheckpointSampler`` and ``PropSampler`` place no point. Valid envs are byte-identical
under both policies, and a masked generation poisons only
the masked envs (the others are restored). Results for invalid envs remain undefined for
the utilities: keep gating on ``Track.valid``. ``"poison"`` is planned to become the
default at 1.0.

**Cost.** The commit launches only when it has work to do:

.. list-table::
   :header-rows: 1
   :widths: 54 46

   * - generation
     - launches added
   * - unmasked, ``on_invalid="keep"``, ``validity.default`` the last check (the
       default pipelines)
     - none: the 0.2.0.dev2 launch sequence (152 launches for a default ``bezier``
       generation at ``E = 2``)
   * - unmasked, with a check after ``validity.default``
     - one (``valid`` derived, ``E`` threads)
   * - unmasked, ``on_invalid="poison"``
     - one (the commit, ``E * N_max`` threads)
   * - masked (``advance=True, mask=m``)
     - two (the hold before the stages, the commit after them, ``E * N_max`` threads
       each, however few envs are masked)

The held copy costs the size of the ``Track``: 64 B per point (five ``vec3f`` and one
``float32``) and 24 B per env, about 100 MB at ``E = 4096``, ``N_max = 384``, allocated
once at build for every Program, whether or not it ever runs a masked generation. The
``Track`` and every working buffer keep their addresses forever.

``Program.on_invalid`` reports the policy. ``TrackGenConfig`` has no ``on_invalid``
field: build the spec with ``presets.track(..., on_invalid="poison")``, or edit a spec's
batch (both are frozen dataclasses):

.. code-block:: python

   import dataclasses

   spec = legacy.to_pipeline_spec(config)
   spec = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch,
                                                              on_invalid="poison"))

.. _pipeline-taps:

Taps
----

``Program.tap(channel, *, after=None)`` returns the working buffer of a channel: its
final version, or with ``after=<stage id>`` the version visible right after that stage.
It replaces reading private scratch (``gen._scratch``) in visualisation and benchmark
code:

.. code-block:: python

   p = gen.program
   raw = p.tap("raw.xy")                          # the generator's raw loop
   resampled = p.tap("cl.xy", after="resample")   # before relaxation
   relaxed = p.tap("cl.xy", after="relax")
   kappa = p.tap("cl.kappa")
   p.channels["cl.xy"]                            # ('resample', 'relax', 'uniform')

- A tap is **read-only** by contract: never write it, the next stage reads it.
- It is stable: the same array for the Program's lifetime, recordable by pointer.
- It holds the last generation's values once that generation has completed on its
  stream. The stages recompute every env, so an intermediate version holds every env's
  recomputation, also for the envs a masked regeneration left out. The final version of
  a channel the ``Track`` exports *is* the ``Track``'s array, so it holds the committed
  values: an unmasked env's previous track after a masked regeneration, the poisoned
  rows under ``on_invalid="poison"``. One exception: when a plugin check runs after
  ``validity.default``, the commit derives ``Track.valid`` from ``valid.fail``, and
  ``tap("track.valid")`` stays ``validity.default``'s own flag, blind to the later
  checks and to poison. Read ``Track.valid`` or ``tap("valid.fail") == 0`` instead.
- ``tap("seed")`` is the effective seed. ``Program.channels`` maps each written channel
  to the ids of its writers, in order (``"seed"`` to ``("<prologue>",)``).
- An unknown channel, an unknown stage, or a stage before the channel's first writer
  raises ``KeyError``.

Not open yet
------------

- **Gate pipeline** (roadmap PR11): gate stages, ``presets.gates``, the ordering
  registry and the ``stations`` adapter; ``GateGenConfig.z_profile`` keeps its closed
  set of names until then.
- **Iterative and host-segmented stages** (roadmap PR12): ``BOUNDED`` stages; a
  ``HOST`` stage currently makes the whole Program eager.
- **Channels and results** (roadmap PR13): per-point channels carried from ``raw.*`` to
  ``cl.*``, ``cl.half_width`` consumers (a width that varies along the band, a width
  stage before relax), ``Track.half_width`` and ``Track.attributes``, and the removal of
  the per-stage ``half_width`` fields and of the parameter latch.
- Opening ``resample``, ``frames``, ``offset`` and ``lift`` to plugins; split built-in
  checks (``validity.turning``, ...); an aliasing in-place write form; a
  ``TrackGenConfig.on_invalid`` field; ``Course`` built from a ``PipelineSpec``.

The 0.2.0.dev2 composition (``warp_pipeline._run_pipeline``, ``inflate_warp``,
``_inflate_warp_alloc``) is still importable. ``TrackGenerator`` no longer calls it; the
tests use it as the byte-identity reference. ``gen._scratch`` is a view of the Program's
buffers kept for the same reason; new code reads ``gen.program.tap``.
