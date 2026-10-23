Pipeline (preview)
==================

.. admonition:: Preview API
   :class: warning

   ``track_gen.pipeline`` is a **preview**: names and signatures may change before 1.0.
   ``TrackGenerator`` and ``GateGenerator`` stay the stable entry points. Each accepts a
   ``PipelineSpec`` as well as its config, and builds its pipeline the way this page
   describes.

The track pipeline of :doc:`pipeline` is described by data. A ``PipelineSpec`` lists the
stages in order, ``build()`` plans and allocates it once, and the ``Program`` it returns
runs the stages, one after another, on every generation; a short epilogue, the *commit*,
then settles what each env's ``Track`` rows hold. ``TrackGenerator`` turns a config into
the default ``PipelineSpec`` (``legacy.to_pipeline_spec``), or takes a spec you built,
usually with ``presets.track``. The gate pipeline of ``GateGenerator`` is described the
same way, with its own preset (``presets.gates``) and its own result, a
``GateSequence`` (see `The gate pipeline`_).

Six stage kinds are open to plugins: ``relax``, ``width``, ``elevation`` and
``validity`` in the track pipeline, ``order`` and ``stations`` in the gate pipeline
(``elevation`` and ``validity`` serve both). They are registered, discovered and
quarantined like generators (see :doc:`/contributing/writing-a-stage`); the other kinds
(``resample``, ``frames``, ``offset``, ``lift`` and the gate stages listed under
`The gate stages`_) are TrackGen's own built-ins. The default track pipeline's outputs
are byte-identical to 0.2.0.dev2, with the same launches in the same order; a masked
regeneration adds two (`The commit`_). The default gate pipeline's outputs are
byte-identical to 0.2.0.dev3 (`Launches`_).

``track_gen.pipeline`` is a subpackage outside ``track_gen.__all__``: import it
explicitly. ``import track_gen`` does not import it.

.. code-block:: python

   from track_gen import PerEnvSeededRNG, TrackGenerator
   from track_gen.pipeline import StageRef, presets

   spec = presets.track(num_envs=64, device="cpu")
   spec.ids
   # ('generator', 'resample', 'width', 'relax', 'uniform', 'frames', 'offset',
   #  'elevation', 'validity', 'lift')

   rng = PerEnvSeededRNG(seeds=1234, num_envs=64, device="cpu")
   gen = TrackGenerator(spec, rng)
   track = gen.generate()              # the same Track instance on every call
   track.fail_mask                     # [E] int32: one bit per failed check
   track.fail_bits                     # {'generator': 0, 'turning': 1, ..., 'truncation': 7}
   track.half_width                    # [E * N_max] float32: the band's half-width

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
   and size, and wires a stage's required read ports (``width.from_channel``'s ``src``,
   see `Required read ports`_). ``capacity`` is the generator's
   ``{"raw": num_points}``, the DEFAULT raw point count (``BatchSpec`` has no
   ``num_points``); a Stage generator's own ``Stage.capacity()`` overrides it.
   ``build`` rejects any other key, and any ``capacity`` on a non-generator stage.

``BatchSpec(num_envs, device, n_max, closed=True, scale=1.0, on_invalid="keep", max_gates=None)``
   The batch every stage shares: ``E`` envs on one device, the per-env point capacity
   ``N_max``, ``TrackGenConfig.scale``, and what the commit does with an invalid env
   (``"keep"`` or ``"poison"``, see `The commit`_). Open tracks (``closed=False``) are
   refused. ``max_gates`` selects the result: ``None`` (default) builds a track
   pipeline whose result is a ``Track``; an ``int >= 2`` builds a gate pipeline whose
   result is a ``GateSequence`` with ``max_gates`` slots per env (``G``, see
   `The gate pipeline`_). In a gate pipeline ``n_max`` is read only by a centerline
   source (``source="stations"``: the generator's layout; ``"relaxed_stations"``: also
   the resample and relax stages); the presets pass 384.

``PipelineSpec(batch, stages, require_caller_capturable=False)``
   The batch and the ordered tuple of ``StageRef``. ``ids`` lists the slot ids.
   ``require_caller_capturable=True`` makes ``build`` reject every ``HOST`` stage (see
   `Capture classes and segments`_), so the Program can always be recorded into a
   caller's graph; ``to_dict()`` writes it only when it is set, and the edit operations
   keep it.
   ``replace(id, ref)``, ``insert_after(id, ref)``, ``insert_before(id, ref)`` and
   ``remove(id)`` return a new spec; ``replace`` keeps the slot's id when ``ref.id`` is
   ``None``. They are pure data operations: an unknown anchor id raises ``ValueError``
   at once, everything else is checked by ``build``. ``to_dict()`` returns a
   JSON-serialisable dict (``"format": "track_gen.pipeline/2"``; ``on_invalid`` and
   ``max_gates`` are optional when reading, ``"keep"`` and ``None`` when absent),
   ``from_dict(d)`` reads it back without
   a registry (a missing key raises a ``ValueError`` naming it, such as ``stages[3] is
   missing 'kind'``), and building the round-tripped spec gives byte-identical outputs,
   plugin stages and their params included. See `Saved specs: format /1 and /2`_ for
   the ``/1`` dicts written by 0.2.0.dev3 and by the 0.2.0.dev4 branch before the width
   moved (roadmap PR11 / PR12, gate specs included).

``legacy.to_pipeline_spec(config, *, registry=None)``
   The default spec for a ``TrackGenConfig``, the one ``TrackGenerator(config)`` builds
   (see `The default track stages`_). It resolves the generator's params, and the
   ``relax_solver`` and ``z_profile`` names (see :ref:`flat-stage-shortcuts`), through
   ``registry``, and raises the ``relax_solver``, ``smooth_finish`` and ``z_profile``
   errors of ``TrackGenerator(config)``.

``legacy.to_gate_pipeline_spec(config, *, registry=None)``
   The default gate spec for a ``GateGenConfig``, the one ``GateGenerator(config)``
   builds (see `The gate stages`_). It resolves the gate generator and its params, the
   ``gate_ordering`` (``None``: the generator's default order) and the ``z_profile``
   through ``registry``, and raises the errors of ``GateGenerator(config)`` (unknown
   names, an unsupported ordering, the gate-count checks of
   :ref:`gate-capacity-config`). A legacy ``GateGeneratorSpec`` emits its
   ``DeprecationWarning`` here (see `Legacy gate generators`_).

.. _pipeline-spec-format:

Saved specs: format /1 and /2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0.2.0.dev4 moved the width stage before ``relax`` and removed the ``half_width`` field
of ``relax.xpbd``, ``offset.default`` and ``validity.default``: they read the per-point
``cl.half_width`` instead (see `Width`_). ``to_dict()`` writes
``"format": "track_gen.pipeline/2"``. ``from_dict`` still reads the
``"track_gen.pipeline/1"`` dicts written by 0.2.0.dev3 and by the 0.2.0.dev4 branch
before the width moved (roadmap PR11 / PR12, gate specs included), and migrates them on
the data, with no registry:

1. the ``half_width`` key is popped from the params of every ``relax.xpbd``,
   ``offset.default`` and ``validity.default`` stage; a popped value that differs from
   the width stage's (``width.constant``'s own ``half_width``, or 0.1, the dev3
   default) raises ``ValueError``;
2. a ``width`` stage that sits after the first ``relax`` stage and before the first
   ``frames`` stage (the dev3 slot) moves to just before that ``relax`` stage;
3. a ``relax.xpbd`` stage with no ``width`` stage before it (a
   ``source="relaxed_stations"`` gate spec written before the width moved: roadmap
   PR11 / PR12, format ``/1``) gets a ``width.constant`` inserted before it, with the
   popped value (or 0.1);
4. one ``DeprecationWarning`` says that the dict was migrated.

With ``width.constant`` (the dev3 default), the migrated spec builds the Program the
``/1`` spec built, byte for byte. Another width stage (a plugin, e.g. ``acme.ripple``)
now runs before ``relax`` and XPBD, the offset and validity read its per-point width,
so the outputs change (:ref:`pipeline-width`); the warning then names the stage. A
dev3 width stage placed after ``frames`` (one that read ``cl.kappa``) in a spec whose
``relax.xpbd`` has no width before it raises ``ValueError``: make it a rewrite of
``cl.half_width`` after a ``width.constant`` before ``relax``, or move it before
``relax``. Any other format string raises ``ValueError``. Specs built in Python against 0.2.0.dev3 are not
migrated (they are not data): they fail at ``build`` with a message that names the
change, see `Specs written for 0.2.0.dev3`_.

.. _pipeline-presets:

Presets
-------

``presets.track(...)`` returns the ``PipelineSpec`` of a track pipeline, with each
replaceable stage chosen by a keyword. It reads the registry (``registry``, default
``track_gen.plugin.registry``) for two things only: whether the generator provides its
own altitudes or half-widths (`Generator-provided altitude and width`_), and whether the
width stage has a ``half_width`` field. A name the registry does not know is treated as
a plain stage, and fails at ``build`` (or ``TrackGenerator``) time.
``presets.gates(...)``, the gate pipeline's preset, is described under
`Gate presets: presets.gates`_.

.. code-block:: python

   presets.track(*, num_envs, device, n_max=384, scale=1.0,
                 generator="bezier", generator_params=None, num_points=256,
                 half_width=0.1, spacing=None,
                 relax="xpbd", width="constant", elevation="flat",
                 checks=("default",), on_invalid="keep",
                 z_base=0.0, registry=None) -> PipelineSpec

The stages, by id: ``generator``, ``resample`` (``resample.constant_spacing``),
``width``, ``relax``, ``uniform`` (``resample.uniform``), ``frames``, ``offset``,
``elevation``, the ``checks`` (the first one's id is ``"validity"``), and ``lift``. A
generator that provides its half-widths has no ``width`` stage, one that provides its
altitudes no ``elevation`` stage. ``presets.track(num_envs=E, device=d)`` builds what
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
       ``width=None`` raises ``ValueError``: something must write ``cl.half_width``
       (a generator that provides it drops the slot, see `Generator-provided altitude
       and width`_).

Anything else raises ``TypeError``, and an empty name ``ValueError``. ``checks`` is a
sequence of selections whose first is ``"default"`` (``validity.default`` starts
``valid.fail``; plugin checks follow it): a single selection raises ``TypeError``, an
empty sequence or another first check ``ValueError``. The other arguments:

- ``generator`` is a generator name, with ``generator_params`` (as
  ``TrackGenConfig.generator_params``), or a ``StageRef(kind="generator", ...)``, which
  carries its own params (passing ``generator_params`` too raises ``ValueError``).
  ``num_points`` becomes the generator slot's ``capacity={"raw": num_points}`` (unless
  the ``StageRef`` sets a capacity).
- ``half_width`` is the value of the width stage: it is filled into the ``width``
  selection when that stage's Params class has a ``half_width`` field
  (``width.constant``, or a plugin's such as ``acme.ripple``) and its params, in any
  selection form, are ``None`` or a mapping without ``half_width``. A Params instance
  is taken as is. Width params that set another ``half_width`` than the preset's raise
  ``ValueError`` naming ``half_width=`` (``width=("constant", {"half_width": 0.2})``
  with the default ``half_width=0.1``). ``relax.xpbd``, ``offset.default`` and
  ``validity.default`` have no ``half_width`` field any more: they read
  ``cl.half_width`` (see `Width`_), and params that set one raise the error of
  `Specs written for 0.2.0.dev3`_.
- ``spacing`` defaults to ``0.6 * half_width``, as in ``TrackGenConfig``. For a width
  stage without a ``half_width`` field (``width.from_channel``) or a generator that
  provides its half-widths, ``half_width`` is only this nominal width for the default
  spacing; a ``half_width`` other than the default 0.1 that nothing reads (such a width
  and an explicit ``spacing``) raises ``ValueError``.
- ``lift`` is ``lift.flat`` (at ``z_base``, default ``0``, as
  ``TrackGenConfig.z_base``) when the elevation is ``elevation.flat``, else
  ``lift.elevated`` (a nonzero ``z_base`` then raises ``ValueError``: the elevation's
  own params set its altitude). A generator that provides its altitudes gets
  ``lift.elevated`` and no elevation stage.
- ``registry`` is the registry the preset reads (default ``track_gen.plugin.registry``).
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
``TrackGenerator`` refuses a gate pipeline's spec (``BatchSpec.max_gates`` set) with a
``ValueError`` that points at ``GateGenerator``, before it binds any generator.
``GateGenerator(spec)`` is described under `Running a gate spec: GateGenerator(spec)`_.

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
     - ``raw.xy``, ``gen.valid``, plus ``raw.z`` / ``raw.half_width`` (what it
       ``provides``) and ``raw.<package>.<name>`` attributes
     - ``track_gen.plugin.registry`` (:doc:`/contributing/writing-a-generator`)
   * - ``gate_generator``
     - ``bezier``, ``hull``, ``polar``, ``checkpoint``, ``voronoi``
     - ``anchors.xy``, ``anchors.count``, plus the ``anchors.*`` attributes it
       ``provides`` and ``anchors.<package>.<name>`` attributes
     - ``track_gen.plugin.registry`` (:ref:`gate-generator-stage`)
   * - ``relax``
     - ``xpbd``, ``none``
     - ``cl.xy``
     - ``track_gen.plugin.registry``
   * - ``width``
     - ``constant``, ``from_channel``
     - ``cl.half_width``
     - ``track_gen.plugin.registry``
   * - ``elevation``
     - ``flat``, ``uniform``, ``random_walk``, ``noise``
     - ``cl.z`` (``gates.z`` in a gate pipeline, `Elevation on gates`_)
     - ``track_gen.plugin.registry``
   * - ``validity``
     - ``default``, ``gate_default``
     - ``valid.fail``
     - ``track_gen.plugin.registry``
   * - ``order``
     - ``raw``, ``ccw``, ``random_pairs``
     - ``gates.xy``, ``gates.count``
     - ``track_gen.plugin.registry`` (:ref:`writing-an-ordering`)
   * - ``stations``
     - ``raw``, ``relaxed``
     - ``anchors.xy``, ``anchors.count``, plus ``anchors.z`` and
       ``anchors.<package>.<name>``
     - ``track_gen.plugin.registry``
   * - ``resample``, ``frames``, ``offset``, ``lift``
     - see `The default track stages`_
     - their catalogue channels
     - TrackGen's private catalogue (not open to plugins)
   * - ``normalize``, ``carry``, ``gate_relax``, ``gate_arclen``, ``tangents``,
       ``gate_frames`` (and ``lift.gates``)
     - see `The gate stages`_
     - their catalogue channels
     - TrackGen's private catalogue (not open to plugins)

A plugin stage may also write namespaced plugin channels of its domain:
``cl.<package>.<name>`` for a track stage, ``gates.<package>.<name>`` for a gate stage
(see `Plugin channels`_). The built-ins are registered under their plain names by
TrackGen's own ``register(reg)``, exactly as a plugin registers, so
``registry.available("relax")``
lists them with the installed plugins and ``registry.describe("elevation", "noise")``
reports their reads, writes and params schema.

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
   ``on_invalid`` in ``("keep", "poison")``, ``max_gates`` ``None`` or an int ``>= 2``;
   slot ids are unique; exactly one generator stage (kind ``generator`` or
   ``gate_generator``), first, with the id ``"generator"``; only the generator has a
   ``capacity``, and only ``"raw"``; ``seeds`` is one of the forms below. A gate
   pipeline starts with a ``gate_generator``, or with a ``generator`` followed by a
   ``stations`` stage; a ``gate_generator`` in a track pipeline is refused.
2. **Resolve.** The generator and every stage of an open kind are resolved through
   ``registry`` (default ``track_gen.plugin.registry``), the generator exactly as
   ``TrackGenerator`` (or, for a ``gate_generator``, ``GateGenerator``) resolves
   ``config.generator``. The other stages come from TrackGen's built-in catalogue.
3. **Params.** Each stage's params are coerced to its Params class and validated (the
   built-ins with the messages ``TrackGenConfig`` / ``GateGenConfig`` use for the same
   fields). Every stage that has a ``half_width``, ``gate_width`` or ``gate_radius``
   field must agree on its value: for ``half_width`` that binds ``width.constant`` and
   the plugin stages that keep such a field (the nominal width; the per-point width is
   ``cl.half_width``). In a gate pipeline each stage's ``Layout`` is derived from its
   ports before its params are validated (see `Two point domains`_).
4. **Channels.** After ``ports`` rewiring, the registration rules hold again (a plugin
   stage writes only what its kind may write, and rewrites a channel through one port;
   see :ref:`stage-ports-channels`), and every required read port is wired (`Required
   read ports`_). The resamplers and the stations get one implicit read and write port
   per carried channel (`Carried channels`_), so the rules below see the carries too.
   Walking the stages in order, every channel a stage
   reads must have been written by an earlier stage, and a channel written twice must
   be read by its second writer (an in-place rewrite, see `In-place rewrites`_). No
   stage writes ``seed``, and no stage writes ``cl.half_width`` after ``offset``
   (`Width`_). Every ``Track`` field has a writer (the ``Track`` reads them
   all: removing ``validity`` or ``lift`` is refused), ``valid.fail`` starts at
   ``validity.default`` (a check that writes it without reading it would drop the
   built-in checks, truncation included, and is refused until the split checks
   arrive), and ``lift.flat``, which lifts at a constant ``z_base``, may not follow an
   elevation other than ``elevation.flat`` (use ``lift.elevated``). A gate pipeline has
   the same rules for the ``GateSequence`` and ``validity.gate_default``, plus its own
   (`Planner rules for gate pipelines`_).
5. **Capture class and segments.** Each stage's capture class comes from its
   ``capture_class(params)`` (a legacy ``GeneratorSpec`` / ``GateGeneratorSpec``:
   ``STATIC``, or ``HOST`` with ``capturable=False``). The Program's is the highest
   (``STATIC`` < ``BOUNDED`` < ``HOST``), and the stages are grouped into segments of
   equal host-ness (see `Capture classes and segments`_). With
   ``PipelineSpec.require_caller_capturable=True`` a ``HOST`` stage raises
   ``PlanError``, naming every one.
6. **Allocate.** The ``SeedState`` (its own checks run first), the ``Track`` or the
   ``GateSequence`` (or use the one passed as ``out=``, checked field by field
   beforehand; its ``generation``, ``fail_mask`` and ``half_width`` may be ``None``,
   and its ``attributes`` may leave keys out, see `Per-point attributes in the
   result`_), the ``[E]`` effective-seed buffer, one buffer per
   channel version and each stage's scratch, namespaced ``scratch:<id>/<name>``. Then
   each stage's ``build`` runs, once, in pipeline order, against a ``StageBuilder``;
   right after it, every write port the stage declares must have been written, and a
   validity stage must own a fail bit (fail bits are handed out in this order, see
   `Fail bits`_). Last, the commit and its held copy of the result.
7. **Program.** With ``prepare=True`` the Program runs once eagerly (the warm-up of
   :doc:`cuda-graph`, with every bounded loop unrolled).

A rule broken in steps 1 to 5 raises ``PlanError``, a ``ValueError`` naming the stage,
its ``kind.name``, the port and the channel, before anything is allocated. Some errors
keep their own type: params raise their own ``ValueError`` / ``TypeError``, an unknown
generator raises the registry's ``ValueError``, a quarantined plugin its
``PluginLoadError``, and a bad ``seeds`` raises ``TypeError``. The rules only a stage's
``build`` reveals (a skipped write port, a validity stage without a fail bit, a 16th
plugin fail bit, a fail-bit name taken twice, a stage-written ``cl.*`` plugin channel
read across the uniform resample: the stale-read rule of `Carried channels`_) raise
``PlanError`` after allocation, as do the ``interp=`` / ``export=`` rules of
``b.write``, with their own ``TypeError`` / ``ValueError``.
``build`` inside a capture raises ``CaptureError`` before allocating.

.. code-block:: python

   build(spec.remove("width"))
   # PlanError: stage 'relax' (relax.xpbd) reads 'cl.half_width' (port 'half_width'),
   #   but no earlier stage writes it; the width stage runs before relax since
   #   0.2.0.dev4 (generator, resample, width, relax, uniform, ...): move it, or insert
   #   StageRef('width', 'constant') before 'relax'

``seeds`` is anything ``SeedState`` accepts (``None``, an ``int``, an ``[E]`` int32
``wp.array``, a ``PerEnvSeededRNG``) or a ``SeedState`` for the batch. Anything else,
a NumPy array or a list included, raises ``TypeError`` before allocating.

``build`` derives the generator's config from the ``BatchSpec`` and the generator's
``StageRef`` (its params, and ``capacity["raw"]`` as ``num_points``, 256 when absent).
That covers the built-ins and Stage generators. A ``GeneratorSpec`` plugin that reads
config fields outside its Params class needs ``TrackGenerator`` with a config. A gate
generator is bound the same way from a ``GateGenConfig`` derived from the spec
(``num_envs``, ``device``, ``max_gates``, ``scale``, its params, and ``min_gates``,
``gate_ordering`` and the gate scalars from the stages that hold them); a legacy
``GateGeneratorSpec`` that reads other config fields needs ``GateGenerator`` with a
config.

``build`` accepts both kinds of spec: the ``Program`` it returns has a ``Track`` or a
``GateSequence`` as its ``result``.

.. _pipeline-program:

The ``Program``
---------------

``Program`` is a ``GraphRunner`` launchable, so it follows the record / eager /
self-graph rules of :doc:`cuda-graph` exactly as ``TrackGenerator`` does.

- ``run(*, stream=None, sync=False, advance=False, mask=None, policy="auto")`` runs one
  generation: recorded into the caller's capture, replayed from the Program's own graph
  (from one graph per capturable segment when it has ``HOST`` stages), or eager.
  ``advance`` and ``mask`` mean what they mean for ``generate()``.
- ``enqueue(ctx, *, advance=False, mask=None)`` launches one generation on ``ctx``: the
  seed prologue (advance, effective seed, parameter latch), with a mask the hold of the
  unmasked envs' ``Track`` rows, every stage in order for every env, the commit, and the
  ``generation`` write. It allocates nothing and never host-syncs (a ``HOST`` stage's
  own host operations aside); called eagerly on ``cuda``, a ``BOUNDED`` stage runs its
  full iteration bound (see :ref:`cuda-graph-bounded`). It is every segment's
  ``enqueue_segment`` in order.
- ``enqueue_segment(i, ctx, *, advance=False, mask=None)`` launches segment ``i`` only
  (see `Capture classes and segments`_).
- ``prepare()`` runs the warm-up, when ``build(..., prepare=False)``. Otherwise the
  first ``run()`` outside a capture runs it, and recording a Program that was never
  prepared (``run()`` or ``enqueue()`` inside a capture) raises ``CaptureError``.
- ``result`` is the ``Track`` (the ``GateSequence`` of a gate pipeline), the same
  instance with the same array addresses forever.
- ``spec`` is the resolved spec: every ``StageRef.params`` is the stage's Params
  instance.
- ``tap(channel, *, after=None)``, ``channels``, ``fail_bits`` and ``on_invalid``: see
  `Taps`_, `Fail bits`_ and `The commit`_.
- ``attributes``: the per-point attributes the result exports, a read-only mapping from
  key to ``AttributeInfo`` (see `Per-point attributes in the result`_).
- ``seeds`` (the ``SeedState``), ``seed`` (the ``[E]`` effective seed), ``inputs``
  (``"regen_mask"`` plus the inputs the stages declare), ``params`` (a ``ParamTable``
  keyed by stage id; the built-in stages' fields are all static) and ``graph`` (``None``
  for a Program with ``HOST`` stages, whose per-segment graphs ``segment_graphs(key=None)``
  returns, ``None`` at the ``HOST`` segments).
- ``capture_class`` (the highest of its stages', ``STATIC`` < ``BOUNDED`` < ``HOST``),
  ``capturable`` (``True`` when a caller can record it: no ``HOST`` stage),
  ``host_stage_ids`` and ``segments`` (see `Capture classes and segments`_).

Capture classes and segments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every stage has a capture class (:ref:`stage-capture-classes`): ``STATIC`` (a fixed
launch sequence), ``BOUNDED`` (an iteration with a static bound and a data-dependent
early stop, through ``ctx.bounded_loop``; capturable by anything, see
:ref:`cuda-graph-bounded`) or ``HOST`` (host readbacks or host control flow: eager only,
see :ref:`cuda-graph-host`). The planner groups consecutive stages into **segments**:
maximal runs whose stages are all ``HOST`` or all capturable, ``STATIC`` and ``BOUNDED``
together. A Program without ``HOST`` stages has one segment, and runs exactly as a single
body. ``Program.segments`` is a tuple of ``Segment`` (``track_gen.pipeline.Segment``):

- ``index``, the segment's position;
- ``capture_class``, ``HOST``, or the highest of its stages' (``STATIC`` or ``BOUNDED``);
- ``stage_ids``, its slot ids in pipeline order;
- ``host``, ``True`` for a ``HOST`` segment.

A legacy ``GeneratorSpec(capturable=False)`` gives ``(HOST: generator) + (STATIC: the
rest)``; a ``HOST`` relax stage between capturable stages gives three segments. Outside a
capture on ``cuda`` the Program records every capturable segment into a graph of its own
and runs the ``HOST`` ones eagerly between the graph launches (the segmented self-graph
of :ref:`cuda-graph-host`). Under a caller's capture, ``run`` and ``enqueue`` of a
Program with ``HOST`` stages raise ``NotCapturableError`` naming them, before any launch.

``enqueue_segment(i, ctx, *, advance=False, mask=None)`` lets an advanced caller do what
the segmented self-graph does: record the capturable segments into its own graph and run
the ``HOST`` segments eagerly.

- Segment ``0`` opens with the seed prologue (the epoch advance, the effective seed, the
  parameter latch and, with a ``mask``, the hold); the last segment closes with the
  commit and the ``generation`` write. A one-segment Program does both, and
  ``enqueue(ctx)`` is every segment in order. Pass the same ``advance`` and ``mask`` to
  every segment of a generation, on one stream: a recorded segment keeps the arguments
  it was recorded with, so run the eager segments with those.
- A ``HOST`` segment raises ``NotCapturableError`` when ``ctx`` is capturing; a capturable
  segment can be recorded into a caller's capture once the Program is prepared, even when
  other segments are ``HOST``.
- Mixing ``advance`` / ``mask`` between the segments of a generation, running them out
  of order, or skipping one is not checked (a replayed segment runs no Python, so the
  Program cannot see its arguments) and leaves the output undefined.
- A legacy-ABI generator (``Program.legacy_stream``) lives in segment ``0`` and launches
  on Warp's current stream: when ``ctx``'s stream is another one, ``enqueue`` and
  ``enqueue_segment(0, ...)`` raise ``LegacyPluginStreamError`` if ``ctx`` is capturing,
  and otherwise run on the current stream, ordered after and before ``ctx``'s stream with
  events.

The checks run on the host before any launch, in this order: ``run`` validates ``mask``
and ``advance`` (``ValueError``; ``enqueue`` and ``enqueue_segment`` leave that to their
caller) and ``enqueue_segment`` its index (``IndexError``); then ``run``, ``enqueue`` and
``enqueue_segment`` of a ``HOST`` segment refuse a capturing ``ctx``
(``NotCapturableError``, also for a Program built with ``prepare=False``); then
recording a Program that was never prepared raises ``CaptureError``.

``PipelineSpec(..., require_caller_capturable=True)`` refuses ``HOST`` stages at build,
before anything is allocated:

.. code-block:: python

   build(dataclasses.replace(spec, require_caller_capturable=True))
   # PlanError: PipelineSpec.require_caller_capturable=True, but stage(s) 'generator'
   #   (generator.acme.blob) is HOST (they read device data back or run host control
   #   flow, so they run eagerly and refuse a caller's capture): use STATIC / BOUNDED
   #   stages or drop require_caller_capturable

A legacy ``GeneratorSpec(capturable=False)`` is named ``'generator' (generator 'acme.old',
GeneratorSpec capturable=False)``.

Channels
--------

Stages talk through named per-env channels; a stage declares the ones it reads and
writes as ``{local port: channel}`` maps (``Stage.reads`` / ``Stage.writes``), and
``StageBuilder.read(port)`` / ``write(port, dtype, per=...)`` hand it the buffers.
``per="env"`` is ``[E]``, ``per="cl"`` is ``[E * N_max]``, ``per="raw"`` is
``[E * raw]``; the gate pipeline adds ``per="anchors"`` (``[E * A]``) and ``per="gates"``
(``[E * G]``), and its channels (``anchors.*``, ``gates.*``) are listed under
`Gate channels`_. The track catalogue in this release:

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
   * - ``raw.z``, ``raw.half_width``
     - raw
     - float32
     - Altitude and half-width per raw point, written by a generator that ``provides``
       them (`Generator-provided altitude and width`_); carried to ``cl.z`` /
       ``cl.half_width`` (``interp="linear"``).
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
     - Per-point half-width, written by the width stage (or carried from
       ``raw.half_width``) before ``relax``, re-interpolated by the uniform resample.
       ``relax.xpbd``, ``offset.default`` and ``validity.default`` read it (`Width`_).
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
     - Per-point elevation (zero on ``flat``), or the carry of a provided ``raw.z``.
   * - ``track.half_width``
     - cl
     - float32
     - The band's half-width on the real points, NaN padding (``Track.half_width``),
       written by ``offset.default``.
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

.. _pipeline-plugin-channels:

Plugin channels
~~~~~~~~~~~~~~~

A plugin may also read and write channels of its own, named
``<domain>.<package>.<name>`` (at least two dot-separated identifiers after the prefix,
such as ``cl.acme.curvature_rank``), for example to hand a buffer from one of its stages
to another, or to give the result a per-point attribute. The prefix is the point domain:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - prefix
     - ``per``
     - written by
   * - ``raw.<package>.<name>``
     - ``"raw"`` (fixed by the prefix)
     - A centerline generator: a per-point attribute of its raw loop, carried to
       ``cl.<package>.<name>`` (`Carried channels`_).
   * - ``cl.<package>.<name>``
     - ``"cl"`` or ``"env"`` (default ``"env"``)
     - Any track-domain stage, and the resamplers (the image of a raw attribute).
   * - ``anchors.<package>.<name>``
     - ``"anchors"`` (fixed by the prefix)
     - A gate generator, or a ``stations`` stage (`Gate plugin channels`_).
   * - ``gates.<package>.<name>``
     - ``"gates"`` or ``"env"`` (default ``"env"``)
     - Any gate-domain stage, and ``carry.attributes`` (the image of an anchors
       attribute).

A channel's dtype and ``per`` are fixed by its first writer's ``b.write(port, dtype,
per=...)``; a later writer must match, and reading one needs an earlier writer like any
channel. A stage uses the channels of its own domain: a gate-domain stage with a
``cl.<package>.<name>`` port raises ``PlanError`` (``plugin channel 'cl.acme.x' is a
centerline channel; a gate-domain stage reads and writes gates.<pkg>.<name>``), and so
does a track stage with a ``gates.*`` one. Read any of them with ``Program.tap``; a
per-point one written with ``export=True`` is also a field of the result (`Per-point
attributes in the result`_).

.. _pipeline-carried-channels:

Carried channels
~~~~~~~~~~~~~~~~

The centerline changes point sets twice: the constant-spacing resample maps the
generator's raw points to the *beads* (``cl.*``), and the uniform resample maps the
relaxed beads to the final points. A per-point value the generator computes (an
altitude, a half-width, a surface label) follows the points through both: it is a
*carried channel*. A carried channel and its images on the later point sets form a
**family** (``raw.acme.grip -> cl.acme.grip``; ``raw.z -> cl.z``; ``raw.half_width ->
cl.half_width``; in a gate pipeline also ``-> anchors.acme.grip -> gates.acme.grip``).
Its dtype, interpolation policy and export flag are fixed by its first writer, the
generator's ``b.write``, and shared by every image:

.. code-block:: python

   # in a centerline generator whose writes include "grip": "raw.acme.grip"
   grip = b.write("grip", wp.float32, per="raw", interp="linear", export=True)

A carried channel is ``float32``, ``int32``, ``vec2f`` or ``vec3f`` (``TypeError``
otherwise). ``interp=`` is required on every ``raw.<package>.<name>`` write (a raw
channel exists only to be carried), and only there: a stage-written channel is not
resampled, and ``interp=`` on it raises ``ValueError``. With ``a`` and ``b`` the source
values at the ends of the source segment and ``f`` the output point's fraction along it:

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - ``interp``
     - dtypes
     - value
   * - ``"linear"``
     - float32, vec2f, vec3f
     - ``a + f * (b - a)``, the expression the resamplers use for the points, so a
       carried copy of the source xy reproduces the resampled points byte for byte.
   * - ``"nearest"``
     - any
     - ``a if f <= 0.5 else b`` (a tie goes to the segment start). Labels never blend;
       a label boundary moves by up to half a source step per resampling.
   * - ``"previous"``
     - any
     - ``a``: a zero-order hold, the value of the last source point at or before the
       output point. A label boundary lags (never leads) by at most one source step
       per resampling (a full step only when an output point lands exactly on a source
       point: the lookup assigns it to the segment that ends there, ``f = 1``). For
       per-segment labels.
   * - ``"angle"``
     - float32 (radians)
     - ``wrap(a + f * wrap(b - a))``, ``wrap(x) = x - 2π floor((x + π) / 2π)``: the
       short way round, in ``[-π, π)``. The source values must lie in ``[-π, π)``;
       others are normalised by the carry. ``f = 1`` need not give ``b`` to the bit.

An ``int32`` channel takes ``"nearest"`` or ``"previous"`` (``TypeError`` otherwise).
The catalogue families have fixed policies (``raw.z`` and ``raw.half_width`` are
``"linear"``), and ``interp=`` on a later writer of a family raises ``ValueError``
(policy and export belong to the first writer).

**Who carries what.** The sets are static: the planner derives them from the declared
writes and reads before it allocates anything.

- ``resample.constant_spacing`` carries every raw family (every ``raw.*`` channel but
  ``raw.xy``) to its ``cl.*`` image.
- ``resample.uniform`` re-interpolates every carried ``cl.*`` family onto the final
  points, as a new version (a rewrite): ``cl.half_width`` and ``cl.z`` whoever wrote
  them, and the image of every raw attribute.
- A relax stage carries nothing (bead attachment, below).
- ``stations.raw`` and ``stations.relaxed`` carry a family of their source to
  ``anchors.<attr>`` only when a later stage reads that image (`3D station spacing`_,
  `Gate plugin channels`_).
- ``carry.attributes`` copies every ``anchors.<package>.<name>`` to
  ``gates.<package>.<name>`` by anchor index.

The carriers are writers like any stage: ``Program.channels["cl.half_width"]`` is
``('width', 'uniform')`` in the default pipeline, and ``tap(channel, after=<carrier>)``
reads each image.

**How.** A carrying resampler runs a variant of its lookup kernel that also writes the
*resample tables* (``ops.ResampleTables``): for every output point, the global indices
``i0`` and ``i1`` of its source segment's ends and its fraction ``frac`` along it. Then
``ops.resample_channels`` interpolates each family through the tables, one launch per
family. A carried channel thus uses exactly the segment and fraction of the points
themselves. A resampler with nothing to carry runs its 0.2.0.dev3 kernels, and a
constant family costs no interpolation (`Width`_), so the default pipelines launch what
0.2.0.dev3 launched.

**Padding.** NaN (every component of a float vector) or ``-1`` (``int32``) at every
slot past ``count[e]``, written by every carrier (an output point outside the real ones
has the table sentinel ``i0 = i1 = -1``, ``frac = NaN``) and by the commit when it
poisons an env. One exception: the uniform image of a *constant* ``cl.half_width``
(``width.constant``) fills its value into every slot, padding included, as 0.2.0.dev3
did; read ``Track.half_width`` for a NaN-padded half-width.

.. _pipeline-bead-attachment:

**Through relaxation: bead-attached.** A relax stage moves points and never re-indexes
them: point ``i`` of its output is point ``i`` of its input, moved. A carried channel
keeps its bead index through any relax stage (no launch, no new version:
``tap("cl.acme.grip", after="relax")`` is the ``after="resample"`` version), and the
uniform resample re-interpolates it with the relaxed loop's own tables. So the
half-width the uniform resample gives a final point is the one XPBD separated that
point with, and a label (a curb, a surface) stays with the point that carries it.
Re-assigning attributes by arc fraction after relaxation would decouple them from the
geometry: XPBD moves beads a long way (a median displacement of 0.085 to 0.55 times the
two-bead distance per batch, up to 0.89 on single envs, in the measurements behind this
choice). A relax plugin must keep this contract (a relax stage that resamples internally
must not be combined with carried channels); the conformance kit does not check it
(:ref:`the relax contract <stage-relax-contract>`).

**Blending.** ``"linear"`` and ``"angle"`` interpolate across the closing segment (the
last bead back to bead ``0``), so a sawtooth attribute such as an arc fraction going
from 0 to 1 takes intermediate values at the seam: carry a ``(cos, sin)`` pair, or use
``"angle"``. On a truncated env (always invalid) the closing segment is the chord over
the dropped arc.

**Stage-written channels are not carried.** A ``cl.<package>.<name>`` channel a stage
writes (``per="cl"``) lives on the points that stage saw. Reading it on the other side of
the uniform resample (written before ``uniform``, read after it) raises ``PlanError``
when the reading stage is built: ``stage 'validity_acme' reads cl.acme.curv, written by
'width' before the uniform resample ('uniform'), which does not carry stage-written
channels (the values sit on the pre-uniform points): write it after 'uniform', or carry
it from the generator (raw.acme.curv)``. The commit's read of an exported channel is
checked the same way. Relax stages do not trigger it (bead attachment), and the gate
domain has no resampler after the ``gates.*`` channels exist.

**Cost.**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - carrier
     - launches added
   * - ``resample.constant_spacing``
     - one per raw family; none (the 0.2.0.dev3 kernels) without one.
   * - ``resample.uniform``
     - one per non-constant family; a constant ``cl.half_width`` costs the one fill
       0.2.0.dev3 already launched.
   * - ``stations.raw``, ``stations.relaxed``
     - one per family a later stage reads; none (roadmap PR11's ``_stations_k``)
       otherwise.
   * - ``carry.attributes``
     - one per ``anchors.<package>.<name>`` family, on ``[E * G]``.

Memory: the tables, 12 B per output point for each carrying resampler, and one buffer
per family per carrier, plus a held copy per exported attribute (`The commit`_). A
generator with four raw attributes adds eight launches to a track generation (two per
family).

.. _pipeline-required-ports:

Required read ports
~~~~~~~~~~~~~~~~~~~

A ``reads`` value may name a domain instead of a channel, ``"cl.*"`` or ``"gates.*"``:
the stage reads a channel the user chooses. Such a port has no default: the
``StageRef`` must wire it with ``ports`` to a concrete channel of that domain, and the
planner refuses one left unwired:

.. code-block:: python

   StageRef("width", "from_channel", ports={"src": "cl.acme.hw"})   # wired

   build(presets.track(num_envs=8, device="cpu", generator="acme.ribbon",
                       width="from_channel"))
   # PlanError: stage 'width' (width.from_channel): read port 'src' (cl.*) must be
   #   wired: StageRef('width', 'from_channel', ports={'src': 'cl.<pkg>.<name>'});
   #   plugin channels written before it: ['cl.acme.grip', 'cl.acme.hw', ...]

A required port is never a write port. ``registry.describe(kind, name).reads`` shows it
as is (``{"src": "cl.*", "count": "cl.count"}`` for ``width.from_channel``).

The default track stages
------------------------

``legacy.to_pipeline_spec(config)`` and ``presets.track(...)`` produce this list. Every
stage launches what the 0.2.0.dev2 pipeline launched at the same point, with the same
arguments: ``width.constant`` launches nothing (its value is filled at build), and the
uniform resample fills the final ``cl.half_width`` with it, the launch 0.2.0.dev3's width
stage made at that point (see `Width`_).

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
     - ``raw.xy``, ``gen.valid`` (plus any ``raw.*`` it carries)
   * - ``resample``
     - ``resample.constant_spacing``
     - ``raw.xy`` (plus every raw family)
     - ``cl.xy``, ``cl.count``, ``cl.truncated`` (plus the ``cl.*`` image of every raw
       family)
   * - ``width``
     - ``width.constant``
     - nothing
     - ``cl.half_width``
   * - ``relax``
     - ``relax.<config.relax_solver>`` (``relax.xpbd``), or ``relax.none`` when
       ``relax_enable=False``
     - ``cl.xy``, ``cl.count``, ``cl.half_width`` (``relax.none``: nothing)
     - ``cl.xy`` (``relax.none``: nothing)
   * - ``uniform``
     - ``resample.uniform``
     - ``cl.xy``, ``cl.count``, ``cl.half_width`` (and every carried ``cl.*`` family)
     - ``cl.xy``, ``cl.half_width`` (and every carried ``cl.*`` family)
   * - ``frames``
     - ``frames.default``
     - ``cl.xy``, ``cl.count``
     - ``track.count``, ``cl.tangent2``, ``cl.normal2``, ``cl.kappa``, ``cl.arclen2``,
       ``cl.length2``, ``cl.winding``
   * - ``offset``
     - ``offset.default``
     - ``cl.xy``, ``cl.normal2``, ``cl.count``, ``cl.half_width``
     - ``band.outer2``, ``band.inner2``, ``track.half_width``
   * - ``elevation``
     - ``elevation.<config.z_profile>``: ``flat``, ``uniform``, ``random_walk``,
       ``noise`` or a plugin's
     - ``seed``, ``cl.count``, ``cl.arclen2``, ``cl.length2`` (``flat``: ``cl.count``)
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
the config: ``resample.constant_spacing`` takes ``spacing``; ``width.constant`` takes
``half_width``; ``relax.xpbd`` takes the ``relax_*`` fields; the elevation stages take
the ``z_*`` fields their profile uses; ``validity.default`` takes ``turning_tol``,
``w_floor``, ``relax_tol``, ``z_valid_grade`` and ``validity_border_check``;
``lift.flat`` takes ``z_base``. ``relax.xpbd``, ``offset.default`` and
``validity.default`` took ``half_width`` too in 0.2.0.dev3: they read ``cl.half_width``
now. ``elevation.flat`` takes ``z_base`` too (the gate altitude; it must be 0 on a track,
where ``lift.flat`` carries the track's flat altitude). The
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

.. _pipeline-width:

Width
-----

``cl.half_width`` is the band's per-point half-width, and its only source. The width
stage writes it on the constant-spacing beads, before ``relax``; XPBD separates the
beads with it; the uniform resample re-interpolates it onto the final points (it is
bead-attached, see `Carried channels`_); ``offset.default`` builds the band from it and
writes ``Track.half_width``; ``validity.default`` checks the band's thickness and its
floor against it.

.. list-table::
   :header-rows: 1
   :widths: 26 52 22

   * - writer
     - ``cl.half_width``
     - launches
   * - ``width.constant``
     - Its ``half_width`` (static, default 0.1) at every slot.
     - none: the buffer is filled when it is allocated
   * - ``width.from_channel``
     - ``min(max(scale * src, lo), max(hi, lo))`` per real point, from the channel
       wired to its ``src`` port; a NaN source point stays NaN; NaN padding.
     - one
   * - a generator that provides ``half_width``
     - The carry of its ``raw.half_width`` (no width stage, see `Generator-provided
       altitude and width`_).
     - the carry
   * - a plugin width stage
     - Whatever it computes (:ref:`stage-width`).
     - its own

``width.from_channel`` takes ``scale`` (dynamic, default 1.0, ``>= 0``), ``lo``
(dynamic, default 0.0, ``>= 0``) and ``hi`` (dynamic, default ``1e9``, ``>= 0``), with
``lo <= hi`` checked at build (a run-time ``hi`` below ``lo`` clamps to ``lo``). Its ``src`` is a required read port (`Required read ports`_): wire it to the
image of a generator's raw attribute (``raw.acme.hw`` is carried to ``cl.acme.hw``) or to
a ``per="cl"`` plugin channel written before it. ``build`` refuses a source that is not
a per-point ``float32`` channel (``ValueError``), and ``cl.half_width`` itself, the
stage's own output (``PlanError``: rescaling the current width is a rewrite stage of
its own). ``lo`` defaults to 0 on purpose: a
source that pinches below ``w_floor`` must fail the floor check, not be clamped away.

.. code-block:: python

   spec = presets.track(
       num_envs=64, device="cpu", generator="acme.ribbon", half_width=0.1,
       width=StageRef("width", "from_channel", {"scale": 1.0, "hi": 0.2},
                      ports={"src": "cl.acme.hw"}))

**Constant width keeps its kernels.** ``width.constant`` declares its value constant (a
private declaration of the built-in). A consumer that reads a constant ``cl.half_width``
runs the 0.2.0.dev3 kernels with that value as a host scalar; any other width selects
per-point kernels when the Program is built:

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - consumer
     - constant width (``hw``)
     - per-point width (``w_i``)
   * - ``relax.xpbd``
     - pair target ``2 hw (1 + margin)``, bend floor ``R_min = hw (1 + margin)``,
       neighbour band from ``2 hw``
     - ``r_i = w_i (1 + margin)``, one launch per solve; pair target ``r_i + r_j``,
       ``R_min = r_i``, neighbour band from ``2 max_i w_i``; the same schedule
   * - ``offset.default``
     - ``c ± hw n``
     - ``c_i ± w_i n_i``
   * - ``validity.default``, thickness (bit 3)
     - ``min(curvature radius, half the closest non-neighbour distance) >=
       (1 - relax_tol) hw``
     - every non-neighbouring pair ``|p_i - p_j| >= (1 - relax_tol)(w_i + w_j)``, and
       every point's curvature radius ``>= (1 - relax_tol) w_i``
   * - ``validity.default``, floor (bit 4)
     - ``w_i > w_floor`` at every real point
     - the same

Two kernels, because a target computed in ``float32`` inside the kernel is not
bit-exact against the host's double-precision scalars of 0.2.0.dev3 (for 24 of 104
``(half_width, margin)`` pairs tried; at ``half_width=0.07, margin=0.2`` a full solve
moves 3909 of 12288 points by up to ``6e-3``). The constant path therefore keeps every
default pipeline byte-identical, and the per-point kernels reproduce the scalar ones at
constant width when given the same ``float32`` scalars. A plugin width that writes a
constant value runs the per-point path: its outputs are close to, not equal to,
``width.constant``'s.

**The floor and the thickness checks read the final points.** Fail bit 4 is set when a
real final point ``i < count`` has ``not (cl.half_width[i] > w_floor)`` (a NaN fails): the
band pinches to the floor somewhere. In 0.2.0.dev3 the check compared a constant fill
with ``w_floor``, the same answer at every point of every env; at constant width it
still is. Both checks read the half-width the uniform resample carried to the final
points (``Track.half_width``), the band the ``Track`` has: a pinch on a single bead that
falls between two final samples is not in the band and does not fail. Two band edges
overlap exactly when two non-neighbouring centerline points are closer than the sum of
their half-widths, or a point's curvature radius is below its own half-width: that is
the per-point thickness test.

**Non-finite widths.** Warp's ``min`` and ``max`` drop a NaN operand, so in the
per-point kernels a NaN half-width does not enter the neighbour band, gives no bend
correction and never triggers a separation: XPBD leaves the bead unconstrained. Such an
env fails bit 4 (``not (NaN > w_floor)``); XPBD does not clamp, the verdict stays
validity's. A negative half-width fails bit 4 for any ``w_floor >= 0``.

**Placement.** The width stage runs before ``relax``. A stage that writes
``cl.half_width`` after ``offset`` raises ``PlanError`` (``stage 'X' writes
cl.half_width after 'offset': the band and Track.half_width come from offset's read, so
validity would check another width than the band; move it before 'offset'``). A width
stage after ``uniform`` is legitimate: a rewrite of the final points' half-width, which
XPBD did not see. That is the recipe for a geometry-dependent width, such as one scaled
by the curvature ``frames`` computes: keep a nominal width in the width slot, for XPBD,
and insert the geometry-dependent stage after ``frames`` as a rewrite of
``cl.half_width``:

.. code-block:: python

   spec = presets.track(num_envs=64, device="cpu", half_width=0.1)
   spec = spec.insert_after("frames", StageRef("width", "acme.curvature_scaled",
                                               id="width_kappa"))

:ref:`stage-width` shows how to write width stages, both kinds.

.. _pipeline-provided:

Generator-provided altitude and width
-------------------------------------

A centerline generator may compute an altitude or a half-width per raw point itself. It
writes ``raw.z`` or ``raw.half_width`` (``float32``, carried with ``interp="linear"``)
and declares it in ``provides``, a subset of ``{"z", "half_width"}`` naming exactly what
it writes. The constant-spacing resample carries them to ``cl.z`` / ``cl.half_width``,
the uniform resample re-interpolates them, and the stages that read those channels use
them:

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - ``provides``
     - ``presets.track`` / ``legacy.to_pipeline_spec``
     - then
   * - ``"z"``
     - no ``elevation`` stage; ``lift.elevated``
     - ``Track.center.z`` is the carried altitude; ``validity.default``'s grade check
       reads it.
   * - ``"half_width"``
     - no ``width`` stage
     - XPBD, the band and the checks use the carried half-width (the per-point kernels).

The presets refuse the selection a provider replaces: an explicit non-flat elevation or
a non-zero ``z_base`` (``ValueError: presets.track(elevation='noise'): generator
'acme.helix_loop' provides its own altitudes (z); leave elevation at its default
('flat')``), and a ``width=`` other than the default ``"constant"`` (``... provides
its own half-widths (raw.half_width -> cl.half_width); leave width at its default, or
add a rewrite of cl.half_width to a hand-built spec``). A hand-built spec that keeps the stage gets the
planner's second-writer ``PlanError``, with a hint (``generator 'acme.helix_loop'
provides z (raw.z -> cl.z): drop the elevation stage``). In a gate pipeline, a ``z``
provider puts the altitude on the loop before the stations (`3D station spacing`_).

.. _pipeline-dev3-specs:

Specs written for 0.2.0.dev3
----------------------------

The width stage moved and three stages lost a field, so a spec built in Python against
0.2.0.dev3 may need an edit; a saved dict migrates on its own (`Saved specs: format /1
and /2`_). ``TrackGenConfig`` users, ``presets.track(half_width=...)`` users and every
``Track`` consumer are unaffected.

- A ``half_width`` key in the params of ``relax.xpbd``, ``offset.default`` or
  ``validity.default`` raises, before the params are coerced: ``stage 'relax'
  (relax.xpbd): half_width is no longer a relax.xpbd param (0.2.0.dev4): the half-width
  is cl.half_width, written by the width stage (presets.track(half_width=...) /
  width.constant)``.
- A spec in the old order (``relax``, ``uniform``, ``width``) has ``relax.xpbd`` read
  ``cl.half_width`` before any writer: the read error names the new order (the example
  under `build() and the planner`_). A hand-built ``relaxed_stations`` gate spec written
  before the width moved (roadmap PR11 / PR12), which had no width stage, gets the same
  message.
- An old-order spec without relaxation (``relax.none``, ``uniform``, ``width``, ...)
  still plans and gives the same bytes, with one launch fewer than in 0.2.0.dev3: its
  width stage now writes ``cl.half_width`` after the uniform resample, which has no
  version of it to fill.

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
     - the band overlaps itself: at constant width, the loop's thickness is below
       ``(1 - relax_tol) * half_width``; at a per-point width, two non-neighbouring
       points are closer than ``(1 - relax_tol)`` times the sum of their half-widths,
       or a curvature radius is below ``(1 - relax_tol)`` times its point's (`Width`_)
   * - 4
     - ``w_floor``
     - a real final point's ``cl.half_width`` (``Track.half_width``) is not above
       ``w_floor`` (NaN included)
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
``track.valid`` unless a check runs after it; ``Track.half_width`` is ``track.half_width``
(``offset``), and each ``Track.attributes[key]`` the final version of an exported
channel (`Per-point attributes in the result`_). Intermediate versions get buffers of
their own. ``Track.generation`` is not a channel.

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
       ``(NaN, NaN, NaN)`` and ``arclen`` and ``half_width`` NaN on every slot; every
       attribute its padding (NaN, or ``-1`` for ``int32``) on every slot;
       ``length = 0`` and ``winding = 0`` (what an empty loop gives); ``valid = 0``.
       ``fail_mask`` and ``truncated`` keep the computed values, so the cause stays
       readable, and ``generation`` is written as usual.

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
   * - with exported attributes
     - per attribute: two more for a masked generation (its hold and its commit), one
       more for an unmasked ``on_invalid="poison"`` one, none otherwise

The held copy costs the size of the ``Track``: 68 B per point (five ``vec3f`` and two
``float32``, ``arclen`` and ``half_width``), plus each exported attribute's size, and 24 B
per env: about 107 MB at ``E = 4096``, ``N_max = 384``, allocated once at build for
every Program, whether or not it ever runs a masked generation. The ``Track`` and every
working buffer keep their addresses forever.

``Program.on_invalid`` reports the policy. ``TrackGenConfig`` has no ``on_invalid``
field: build the spec with ``presets.track(..., on_invalid="poison")``, or edit a spec's
batch (both are frozen dataclasses):

.. code-block:: python

   import dataclasses

   spec = legacy.to_pipeline_spec(config)
   spec = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch,
                                                              on_invalid="poison"))

.. _pipeline-attributes:

Per-point attributes in the result
----------------------------------

``Track.half_width`` is the band's half-width per point, ``[E * N_max]`` ``float32``:
the value on the real points, NaN on the padding (the constant on every real point at
constant width). It is ``offset.default``'s ``track.half_width``, so the band and the
field agree: ``|outer - center| == half_width`` on every real point. It is ``None`` on a
``Track`` built by hand.

``Track.attributes`` maps a key to the final version of each per-point plugin channel
exported with ``export=``; the key is the channel's name without its domain prefix:

.. code-block:: python

   # in a generator: writes = {..., "surface": "raw.acme.surface_id"}
   b.write("surface", wp.int32, per="raw", interp="nearest", export=True)

   track = gen.generate()
   track.attributes["acme.surface_id"]        # [E * N_max] int32, -1 on the padding
   info = gen.program.attributes["acme.surface_id"]   # an AttributeInfo
   info.channel, info.interp, info.padding, info.domain
   # ('cl.acme.surface_id', 'nearest', -1, 'cl'); info.dtype is wp.int32

- Each array is ``[E * N_max]`` (``[E * G]`` on gates), of the family's dtype, padded
  with NaN or ``-1``; it is allocated once at build and keeps its address forever, like
  every ``Track`` array. The mapping is read-only.
- ``export=True``, or the key itself (``export="acme.surface_id"``): another string
  raises ``ValueError`` (``export='surface': the key of raw.acme.surface_id is
  'acme.surface_id' (renaming is not supported)``), another type ``TypeError``.
- ``export=`` is for per-point plugin channels: a ``raw`` or ``anchors`` family, or a
  ``cl`` / ``gates`` channel a stage writes with ``per="cl"`` / ``per="gates"``. On a
  per-env channel it raises ``ValueError`` (``export= needs a per-point channel:
  per='cl' (got the default per='env')``), on a catalogue family too (``raw.z reaches
  Track.center.z; export= is for plugin channels``). A later writer of the family may not
  pass it.
- An exported raw family reaches ``Track.attributes`` through the resamplers, on every
  track pipeline. A channel a stage writes is exported as written, so the stale-read rule
  of `Carried channels`_ applies to the commit's read of it.
- The commit holds and restores attributes under a masked regeneration and fills them
  with their padding when it poisons an env (`The commit`_). ``Track.clone()`` copies
  ``half_width`` and every attribute.
- ``GateSequence.attributes`` is the same for the gate pipeline's
  ``gates.<package>.<name>`` channels (`Gate plugin channels`_), with one difference:
  a family with no index path to the gates is left out (best effort).
- ``Program.attributes`` maps each key of the result to an ``AttributeInfo`` (``key``,
  ``channel``, ``dtype``, ``interp``, ``padding``, ``domain``;
  ``track_gen.pipeline.AttributeInfo``), the policy a consumer that samples an
  attribute along the track needs (``interp`` is ``None`` for a channel a stage wrote on
  the result's points, or one carried by index only).
  ``registry.describe("generator", name)`` lists the raw channels a generator declares
  (``interp`` and ``export`` are decided in ``build``).

**Caller-owned results** (``build(..., out=...)``) follow one rule for the optional
fields (``fail_mask``, ``generation``, ``half_width``, attribute keys): a field the caller
leaves ``None``, or a key it leaves out of ``out.attributes``, is not added to the
caller's object, and the Program keeps that work buffer for itself (``tap`` reaches it).
A key of ``out.attributes`` the Program exports is bound after a check (``[E * N_max]``,
the dtype, the device, contiguous); a key it does not export raises ``ValueError``
(``out.attributes['x']: this pipeline does not export 'x'; exported: [...] (out= with
keys from another pipeline is reserved for PR13b's mixed batches)``).

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
   beads_hw = p.tap("cl.half_width", after="width")   # the width XPBD separates with
   p.channels["cl.half_width"]                    # ('width', 'uniform')

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

.. _pipeline-gates:

The gate pipeline
-----------------

``GateGenerator`` runs a ``Program`` too. Its spec is a *gate pipeline*:
``BatchSpec.max_gates`` is set, the result is a ``GateSequence``, and the stages take
gate *anchors*, the plan-view points a generator emits, to ordered, separated,
elevated, framed and validated gates. Every rule of this page applies to it unchanged
(the planner, versioned rewrites, fail bits, the commit, taps, capture classes, runtime
parameters); this section lists what differs.

.. code-block:: python

   from track_gen import GateGenConfig, GateGenerator
   from track_gen.pipeline import presets

   spec = presets.gates(num_envs=64, device="cpu")     # bezier anchors, ccw order
   spec.ids
   # ('generator', 'order', 'gate_relax', 'elevation', 'lift', 'tangents',
   #  'gate_frames', 'validity')

   gen = GateGenerator(spec)                            # env e gets seed e
   gates = gen.generate()
   gates.fail_mask                                      # [E] int32: one bit per failed check
   gates.fail_bits                                      # {'count': 0, 'finite': 1, ..., 'grade': 5}
   gen.program.tap("anchors.xy")                        # the anchors, before ordering

   # The same Program from the flat config (byte-identical):
   GateGenerator(GateGenConfig(num_envs=64, device="cpu"))

   # Another order and elevation, by registered name:
   presets.gates(num_envs=64, device="cpu", generator="polar", order="raw",
                 elevation=("noise", {"z_noise_amplitude": 0.3}))

   # Gates for a track generator: arc-length stations on its centerline.
   presets.gates(num_envs=64, device="cpu", generator="bezier", source="stations",
                 gates=12)

.. _pipeline-gate-stages:

The gate stages
~~~~~~~~~~~~~~~

``legacy.to_gate_pipeline_spec(config)`` and ``presets.gates(...)`` produce this list
from a native gate generator. Stages in *italics* below are present only when their
condition holds. Every stage launches what the 0.2.0.dev3 gate pipeline launched at the
same point, with the same arguments (`Launches`_ lists the one reordering).

.. list-table::
   :header-rows: 1
   :widths: 11 25 30 34

   * - id
     - stage
     - reads
     - writes
   * - ``generator``
     - ``gate_generator.<config.generator>``
     - nothing (its seed is ``b.seed``)
     - ``anchors.xy``, ``anchors.count``, plus the attributes it provides and its
       ``anchors.<package>.<name>`` channels
   * - ``order``
     - ``order.<config.gate_ordering>`` (``None``: the generator's default order,
       :ref:`pipeline-ordering-registry`)
     - ``anchors.xy``, ``anchors.count`` (``random_pairs``: also ``seed``)
     - ``gates.xy``, ``gates.count``
   * - *normalize*
     - ``normalize.extent``, when the generator prefers it (``polar``, ``checkpoint``,
       ``voronoi``)
     - ``gates.xy``, ``gates.count``
     - ``gates.xy`` (a rewrite)
   * - *carry_<attr>*
     - ``carry.<attr>``, one per attribute the generator provides
     - ``anchors.<attr>``, ``gates.count``
     - ``gates.<attr>``
   * - *carry_attributes*
     - ``carry.attributes``, when the generator writes ``anchors.<package>.<name>``
       channels
     - every ``anchors.<package>.<name>``, ``gates.count``
     - the matching ``gates.<package>.<name>``
   * - ``gate_relax``
     - ``gate_relax.spheres``; ``gate_relax.none`` when ``gate_solve_iters == 0`` or
       the float32 target ``2 * gate_radius`` is ``0``
     - ``gates.xy``, ``gates.count`` (``.none``: nothing)
     - ``gates.xy`` (a rewrite; ``.none``: nothing)
   * - *gate_arclen*
     - ``gate_arclen.default``, unless the elevation is ``flat`` or the generator
       provides ``z``
     - ``gates.xy``, ``gates.count``
     - ``gates.cum``, ``gates.perim``
   * - *elevation*
     - ``elevation.<config.z_profile>``, unless the generator provides ``z``
     - ``seed``, ``gates.count``, ``gates.cum``, ``gates.perim`` (``flat``:
       ``gates.count``)
     - ``gates.z``
   * - ``lift``
     - ``lift.gates``
     - ``gates.xy``, ``gates.z``
     - ``gates.position``
   * - *tangents*
     - ``tangents.central``, unless the generator provides ``tangent``
     - ``gates.position``, ``gates.count``
     - ``gates.tangent``
   * - ``gate_frames``
     - ``gate_frames.default``; ``gate_frames.posed`` when the generator provides
       ``half_size`` and ``roll``
     - ``gates.position``, ``gates.tangent``, ``gates.count`` (``.posed``: also
       ``gates.half_size``, ``gates.roll``)
     - ``gates.position``, ``gates.tangent`` (rewrites), ``gates.forward``,
       ``gates.orientation``, ``gates.half_size``, ``gates.left``, ``gates.right``
   * - ``validity``
     - ``validity.gate_default``
     - the seven frame outputs, ``gates.count``
     - ``gates.valid``, ``valid.fail``

The built-in stages' params keep the flat ``GateGenConfig`` field names, copied from the
config: ``gate_relax.spheres`` takes ``gate_radius`` and ``gate_solve_iters`` (and refuses
a pair that would not move anything: use ``gate_relax.none``); ``gate_frames.default``
takes ``gate_width`` and ``gate_align``, ``gate_frames.posed`` ``gate_align`` only; the
elevation stages take the ``z_*`` fields their profile uses (``elevation.flat`` takes
``z_base`` on gates); ``validity.gate_default`` takes ``min_gates``, ``gate_radius``,
``gate_width``, ``z_valid_grade`` and ``check_crossing`` (see `Gate fail bits`_). The
generator's params are its ``generator_params``; the order, ``normalize`` and ``carry``
stages take none (``normalize.extent`` scales to ``ops.constants.TARGET_EXTENT *
BatchSpec.scale``).

Then the commit (`The gate commit`_) and the ``generation`` write. A legacy
``GateGeneratorSpec`` replaces ``generator``, ``order`` and ``normalize`` with one
adapter slot (`Legacy gate generators`_), and a centerline generator replaces
``generator`` with ``generator`` + ``stations`` (`Stations: gates from any track
generator`_).

Two point domains
~~~~~~~~~~~~~~~~~

A gate pipeline has two point domains. The anchors domain is ``[E * A]``, where ``A`` is
the anchors producer's ``capacity()["gates"]``: a generator writes its anchors at its
own stride, as it always did. The gate slots are ``[E * G]``, ``G =
BatchSpec.max_gates``, the ``GateSequence`` layout. The ``order`` stage maps the first
onto the second; every stage after it works on gate slots.

The planner gives each stage its own ``Layout``, derived from its ports before its params
are validated: ``b.layout.stride`` is ``G`` for a gate-domain stage (an anchors
generator, ``stations.raw``, an order, the gate stages, a validity plugin) and ``N_max``
for the centerline generator of a stations pipeline and for any stage with a ``cl.*``
port (the ``resample``, ``width`` and ``relax`` stages of ``source="relaxed_stations"``,
with ``elevation_on="centerline"`` also its ``frames`` and elevation, and
``stations.relaxed``). A stage with both ``cl.*`` and ``gates.*`` ports raises
``PlanError``: a stage runs on one point domain. Size a buffer from the port, not from the
layout: ``b.stride_of(port)`` is the row stride of the channel a port is wired to
(``raw``, ``N_max``, ``A`` or ``G``; ``1`` for an ``[E]`` channel) and ``b.per_of(port)``
its ``per``.

.. _pipeline-gate-channels:

Gate channels
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - channel
     - per
     - dtype
     - meaning (``GateSequence`` field it becomes)
   * - ``anchors.xy``
     - anchors
     - vec2f
     - The generator's (or ``stations``') anchors, in its own order. Slots past
       ``anchors.count`` are never read.
   * - ``anchors.count``
     - env
     - int32
     - Real anchors per env.
   * - ``anchors.z``, ``anchors.tangent``, ``anchors.half_size``, ``anchors.roll``
     - anchors
     - float32, vec3f, float32, float32
     - Per-anchor attributes an anchors generator ``provides`` (`Provided attributes`_).
       ``anchors.z`` is also the stations' carry of a centerline altitude (`3D station
       spacing`_).
   * - ``anchors.<package>.<name>``
     - anchors
     - float32, int32, vec2f, vec3f
     - A plugin attribute per anchor: written by a gate generator, carried by the
       stations from a centerline family (`Gate plugin channels`_).
   * - ``gates.xy``
     - gates
     - vec2f
     - Plan-view gate centers: ordered (``order``), then normalized, then separated
       (``gate_relax``), one version each.
   * - ``gates.count``
     - env
     - int32
     - Real gates per env (``GateSequence.count``); normally a copy of
       ``anchors.count``.
   * - ``gates.cum``, ``gates.perim``
     - gates, env
     - float32
     - Plan-view cumulative chord length and loop length: the elevation stages' arc
       tables.
   * - ``gates.z``
     - gates
     - float32
     - Per-gate altitude.
   * - ``gates.position``, ``gates.tangent``
     - gates
     - vec3f
     - Lifted centers and 3D tangents, rewritten by ``gate_frames``
       (``GateSequence.position``, ``GateSequence.tangent``).
   * - ``gates.forward``, ``gates.left``, ``gates.right``
     - gates
     - vec3f
     - The frame's forward axis and bar endpoints (the ``GateSequence`` fields).
   * - ``gates.orientation``
     - gates
     - quatf
     - The gate pose (``GateSequence.orientation``).
   * - ``gates.half_size``
     - gates
     - float32
     - Opening half-extent (``GateSequence.half_size``).
   * - ``gates.roll``
     - gates
     - float32
     - A provided roll, carried from ``anchors.roll`` for ``gate_frames.posed``.
   * - ``gates.valid``
     - env
     - int32
     - ``validity.gate_default``'s own 0/1 flag (``GateSequence.valid`` unless a check
       runs after it, as ``track.valid``).
   * - ``valid.fail``
     - env
     - int32
     - The fail bitmask, shared with tracks (``GateSequence.fail_mask``).
   * - ``gates.<package>.<name>``
     - gates or env
     - any
     - A gate-domain plugin channel; exported ones become ``GateSequence.attributes``
       (`Gate plugin channels`_).

``GateSequence.generation`` is not a channel. As for tracks, the final version of each
exported channel *is* the ``GateSequence`` array. A gate-domain stage uses
``gates.<package>.<name>`` plugin channels; a port on a ``cl.<package>.<name>`` channel
raises ``PlanError`` (`Plugin channels`_).

Anchors generators
~~~~~~~~~~~~~~~~~~

A native gate generator is a ``Stage`` with ``kind="gate_generator"`` and
``output="anchors"``: it reads no channel, writes ``anchors.xy`` and ``anchors.count``,
and does not order, normalize or lift anything (:ref:`gate-generator-stage` is the
guide). Its ``capacity(params, layout)`` returns ``{"gates": A, "min_gates": k}``: ``A``
is the most anchors it writes per env (its row stride), ``k`` the fewest (default ``0``).
This replaces the legacy ``GateGeneratorSpec.max_gates`` callable, which meant both "how
many slots I need" and "how many gates I can make". The planner checks it against the
batch and the validity stage, with the messages ``GateGenConfig`` users already know:

- ``A > BatchSpec.max_gates``: ``PlanError``: ``... BatchSpec.max_gates=8 is too small for
  gate generator 'polar'; required max_gates 12.``
- ``validity.gate_default``'s ``min_gates > A``: ``PlanError``: ``... min_gates=16 is too
  large for gate generator 'polar'; generator can produce at most 12 gates.``

The five built-ins are anchors generators: ``bezier`` and ``hull`` write an unordered
cloud of grid points (``A = max_num_points``), ``polar``, ``checkpoint`` and
``voronoi`` a loop in generation order (``A`` = their knot, checkpoint or control-point
count) that the pipeline then normalizes to ``TARGET_EXTENT * scale``.

.. _pipeline-provides:

Provided attributes
~~~~~~~~~~~~~~~~~~~

An anchors generator may also write ``anchors.z``, ``anchors.tangent``,
``anchors.half_size`` and ``anchors.roll``, declared in ``provides`` (a subset of
``{"z", "tangent", "half_size", "roll"}`` that names exactly the attributes it writes;
``half_size`` and ``roll`` come together). The preset then drops the stages that would
compute them and carries the provided values to the gate slots instead:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``provides``
     - dropped
     - added or changed
   * - ``"z"``
     - ``gate_arclen``, ``elevation``
     - ``carry_z`` (``carry.z``)
   * - ``"tangent"``
     - ``tangents``
     - ``carry_tangent`` (``carry.tangent``)
   * - ``"half_size"``, ``"roll"``
     - nothing
     - ``carry_half_size``, ``carry_roll``; ``gate_frames.posed`` instead of
       ``.default``, and ``validity.gate_default(check_crossing="always")``

The ``carry`` stages copy by anchor index (padding ``0.0`` for ``z`` and ``roll``, NaN for
the others), so they need the index-preserving ``order.raw``: a generator that provides
attributes declares ``orderings=frozenset({"raw"})``. What reaches the ``GateSequence``:

- ``z`` becomes ``position.z`` bit for bit;
- ``tangent`` is normalized by the frames stage (``GateSequence.tangent`` is the
  normalized ``anchors.tangent``; a unit float32 tangent may change in the last ulp);
- ``half_size`` becomes ``GateSequence.half_size`` bit for bit, and ``gate_width`` is
  ignored (a non-zero ``gate_width`` raises ``ValueError``);
- ``roll`` turns the frame about its forward axis: ``orientation``, ``left`` and
  ``right`` include it.

The plan-view sphere separation still runs after the generator and may move a gate's
xy, so a provided ``z`` or ``tangent`` describes the anchor as emitted, not the moved
gate: pass ``gate_solve_iters=0`` to keep the emitted xy.

The planner holds a hand-built or edited spec to the same rules: a provided attribute
that no later stage reads raises ``PlanError`` (``stage 'generator'
(gate_generator.acme.helix) provides 'z' (anchors.z), but no stage reads it; stage
'elevation' (elevation.noise) writes gates.z instead: carry it with carry.z (presets.gates
drops the stages a generator provides)``), and so does a ``carry`` stage after an order
other than ``order.raw``. The presets refuse an
explicit elevation (``ValueError``: ``... provides its own altitudes (z); leave elevation
at its default ('flat')``), and ``GateGenConfig`` a ``z_profile`` other than ``"flat"`` or a
non-zero ``z_base``, for a generator that provides ``z``.

.. _pipeline-gate-plugin-channels:

Gate plugin channels
~~~~~~~~~~~~~~~~~~~~

A gate-domain stage (an elevation, a check, an order plugin) reads and writes its own
``gates.<package>.<name>`` channels (``per="gates"``, ``[E * G]``, or the default
``per="env"``, ``[E]``), and exports the per-gate ones to ``GateSequence.attributes``:

.. code-block:: python

   # a gate check: writes = {"fail": "valid.fail", "clearance": "gates.acme.clearance"}
   clearance = b.write("clearance", wp.float32, per="gates", export=True)

   gates = gen.generate()
   gates.attributes["acme.clearance"]      # [E * G] float32, NaN on the padding

Per-anchor attributes reach the gate slots by anchor index, as provided attributes do:

- a gate generator writes ``anchors.<package>.<name>`` channels (any dtype of
  `Carried channels`_, with ``export=`` but no ``interp=``: they are copied, not
  interpolated); it must declare ``orderings=frozenset({"raw"})``;
- the built-in stations carry a centerline generator's ``raw.<package>.<name>``
  attributes (their ``cl.*`` images, for ``stations.relaxed``) to
  ``anchors.<package>.<name>``, when a later stage reads them;
- ``carry.attributes``, which ``presets.gates`` inserts after an ``order.raw`` when such
  channels exist, copies every ``anchors.<package>.<name>`` to
  ``gates.<package>.<name>`` (padding NaN or ``-1``), one launch per channel, exported
  or not. After an order other than ``order.raw`` it raises ``PlanError``, as
  ``carry.<attr>`` does.

An exported per-point attribute of a centerline generator thus reaches
``GateSequence.attributes`` along ``generator -> stations -> order.raw ->
carry.attributes``. This is **best effort**: with another order (``order="ccw"``), a
plugin stations stage, or a per-point channel a stage writes on the centerline (an
elevation on the relaxed centerline, ``elevation_on="centerline"``, exporting
``cl.<pkg>.<name>``: the stations carry generator families only) there is no index
path, and the key is left out of ``GateSequence.attributes`` (``Program.attributes``
lists the keys that are there); asking
for it through ``out.attributes`` raises ``ValueError`` (``out.attributes['acme.grip']:
this pipeline does not export 'acme.grip'; exported: [...]``).
An ``anchors.*`` attribute a gate generator or a plugin stations stage writes must be
read by a later stage (``carry.z``, ``carry.attributes``), else ``PlanError`` names the
writer and the stage to add. The plugin channels of an elevation stage are rewired to
the gate slots by prefix (`Elevation on gates`_).

.. _pipeline-ordering-registry:

The ordering registry
~~~~~~~~~~~~~~~~~~~~~

Ordering is a stage of the open kind ``order``, applied by the pipeline: generators no
longer order their own anchors. An order stage reads ``anchors.xy`` and
``anchors.count``, writes the first ``m = min(count, A, G)`` gate slots of ``gates.xy``
(NaN after them) and ``gates.count``. The built-ins are 0.2.0.dev3's kernels:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - stage
     - order
   * - ``order.raw``
     - The anchors as the generator emitted them (index-preserving).
   * - ``order.ccw``
     - Sorted by angle around each env's centroid. The name is kept for compatibility:
       the convention is clockwise in standard xy coordinates.
   * - ``order.random_pairs``
     - Random gate pairs for training objectives, keyed on the effective seed.

``GateGenConfig.gate_ordering`` and ``presets.gates(order=...)`` name any registered
order, a third party's included (:ref:`writing-an-ordering`). An unknown name raises
``ValueError`` with the registry's list (``GateGenConfig.gate_ordering: unknown order
'spiral'; available: ['ccw', 'random_pairs', 'raw']``), a quarantined plugin its
``PluginLoadError``.

A generator declares which orders it accepts, in one of two ways:

- ``excluded_orderings`` (with ``orderings=None``): every registered order but these.
  The built-ins use it: ``bezier`` and ``hull`` exclude ``raw`` (a grid-point cloud has no
  order of its own), ``polar``, ``checkpoint`` and ``voronoi`` exclude ``random_pairs``.
  A third-party order therefore runs with every built-in generator.
- ``orderings``, a whitelist. Generators that provide attributes declare
  ``frozenset({"raw"})``; a legacy ``GateGeneratorSpec``'s ``supported_orderings`` is
  one.

``preferred = {"order": name, "normalize": name}`` names the generator's default order
and its normalization (``preferred["normalize"] = "extent"`` inserts
``normalize.extent``). An order the generator does not accept raises (``PlanError`` from
the planner, ``ValueError`` from the config): ``gate generator 'polar' does not support
gate_ordering 'random_pairs'; supported: ['ccw', 'raw'] (its preferred order is 'ccw')``,
the hint only when the generator declares a preferred order. With only the built-in orders
installed the supported lists are those of 0.2.0.dev3.

The default order is one rule, for an anchors generator: its ``preferred["order"]`` when
it accepts it, else ``"ccw"`` when it accepts it, else the first order it accepts
(``"raw"`` for stations). ``presets.gates(order=None)``,
``GateGenConfig.gate_ordering=None`` (the config default, resolved by
``legacy.to_gate_pipeline_spec`` when ``GateGenerator`` or ``Course`` is constructed)
and the conformance kit all apply it, so ``GateGenerator(GateGenConfig(
generator="acme.helix"))`` runs ``order.raw``. The run config (the config a legacy
``GateGeneratorSpec`` receives) holds the resolved name, never ``None``. For the five
built-ins the rule gives ``"ccw"``, the 0.2.0.dev3 default. An explicit name is checked
as before.

.. _pipeline-stations:

Stations: gates from any track generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A track generator (``output="centerline"``) makes gates too: ``presets.gates`` inserts a
``stations`` stage that places ``gates`` anchors at equal arc-length spacing along its
closed centerline, then the shared gate stages run. It works for every built-in track
generator, ``repulsive`` included, and for third-party ones, with no gate-specific code.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``source=``
     - stages before ``order``
   * - ``"stations"``
     - ``generator`` (the centerline generator, ``capacity={"raw": num_points}``), then
       ``stations.raw`` on its raw loop (``raw.xy``).
   * - ``"relaxed_stations"``
     - The stages of ``presets.track`` up to ``relax`` (``generator``, ``resample``,
       ``width``, ``relax``, with the same keywords: ``n_max``, ``num_points``,
       ``half_width``, ``spacing``, ``width``, ``relax``; no ``width`` stage for a
       generator that provides its half-widths), then ``stations.relaxed`` on the
       relaxed centerline (``cl.xy``). No uniform resample, offset or track validity run
       (``frames`` and an elevation do with ``elevation_on="centerline"``, see `3D
       station spacing`_).
   * - ``"auto"`` (default)
     - ``"anchors"`` when the name is registered as a gate generator (the built-in names
       are both kinds: the gate generator wins, as before), else ``"stations"`` when it
       is a track generator, else ``ValueError`` listing both kinds.

Each station ``k`` sits at arc length ``L * k / A`` from point 0 along the closed loop
(``A = gates``, default 12; ``L`` the loop's length, plan-view by default), so the
anchors are in loop order and the default order is ``raw``. An env whose generator
flagged it invalid, whose loop was truncated, or whose length is not positive gets
``anchors.count = 0`` (NaN anchors), and fails the ``count`` check. ``capacity()`` is
``{"gates": gates, "min_gates": 0}``; ``max_gates`` defaults to ``gates``.

The stations carry the centerline's families that a later stage reads to
``anchors.<attr>``, through the same resample tables as the anchors: the altitude
(``anchors.z``, read by ``carry.z``) and the plugin attributes (read by
``carry.attributes``, `Gate plugin channels`_). ``stations.relaxed`` carries the
``cl.*`` families except ``cl.half_width``.

.. _pipeline-stations-xyz:

3D station spacing
~~~~~~~~~~~~~~~~~~

``StationsParams.spacing_metric`` chooses the arc length the stations measure:
``"xy"`` (default, the plan-view length of roadmap PR11's stations, byte for byte) or
``"xyz"``, the 3D length along the loop, so that the gates are equally spaced in flight
distance on a course that climbs. ``presets.gates(spacing_metric=...)`` sets it. The ``"xyz"`` branch
accumulates the segment lengths in ``float64`` (the 3D spacing is equal to well within
``1e-6``, relative), and carries the altitude to the anchors with ``"linear"``.

3D spacing needs an altitude on the loop *before* the stations are placed. It comes from
one of two places:

- **the generator**, one that provides ``z`` (``raw.z``, `Generator-provided altitude
  and width`_), with ``source="stations"`` or ``"relaxed_stations"``;
- **an elevation stage on the relaxed centerline**, with ``source="relaxed_stations"``,
  a non-flat elevation and ``elevation_on="centerline"``: the preset runs ``frames`` and
  the elevation on the relaxed loop, before ``stations.relaxed``.

The stations then carry the altitude to ``anchors.z``, ``order.raw`` keeps the anchor
index, ``carry.z`` copies it to ``gates.z``, and the gate-level ``gate_arclen`` and
elevation stages are dropped. ``elevation_on`` (``"gates"`` or ``"centerline"``) chooses
where the elevation runs, independently of the metric: ``None`` (default) is
``"centerline"`` for ``"xyz"`` and ``"gates"`` for ``"xy"``. The two are separate
controls because an elevation on the centerline is a different course: a profile drawn
along the loop's arc length, sampled at the stations, instead of one drawn over the
gates (a knot profile such as ``uniform`` then puts its knots along the loop, not at the
gates). Toggling ``spacing_metric`` with the default ``elevation_on`` therefore also
moves the elevation; pass ``elevation_on`` explicitly to change one thing at a time. It
also changes which attributes reach the gates: a plugin elevation's exported channel
(``cl.acme.slope``, rewired to ``gates.acme.slope`` on gates) is in
``GateSequence.attributes`` when the elevation runs on the gates, and left out when it
runs on the centerline (the stations carry generator families only, not channels a
stage writes); ``Program.attributes`` lists what is there.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - source, metric, altitude
     - gate stages
   * - any source, ``"xy"``, no altitude on the loop (the defaults)
     - those of roadmap PR11 (gate-level ``gate_arclen`` and ``elevation``);
       byte-identical to them (golden ``tests/goldens/pr13a_pr11_relaxed_stations.json``;
       ``relaxed_stations`` gains its ``width`` slot)
   * - an altitude on the loop (a ``z`` provider, or ``relaxed_stations`` with a
       non-flat elevation and ``elevation_on="centerline"``), either metric
     - ``frames`` and the elevation on the centerline (not for a provider), the
       stations, ``order.raw``, ``carry.z``, ...; no ``gate_arclen`` and no gate-level
       elevation
   * - ``"xyz"`` with ``source="stations"``, a non-flat elevation and no ``z`` provider
     - ``ValueError``: use ``source="relaxed_stations"`` (a raw loop has no
       sub-pipeline to elevate)
   * - ``"xyz"`` with no altitude on the loop (a flat elevation and no ``z``
       provider)
     - ``ValueError``: use a ``z`` provider, or ``source="relaxed_stations"`` with a
       non-flat elevation
   * - ``"xyz"`` with ``elevation_on="gates"`` and no ``z`` provider, or
       ``elevation_on="centerline"`` with ``source="stations"``
     - ``ValueError``
   * - an explicit ``elevation_on`` with a ``z`` provider (no elevation runs)
     - ``ValueError``: leave ``elevation_on`` at ``None``
   * - ``spacing_metric="xyz"`` with an anchors generator
     - ``ValueError`` (``spacing_metric= places stations; gate generator 'bezier'
       places its own gates``)
   * - any explicit ``elevation_on`` with an anchors generator
     - ``ValueError`` (``elevation_on= places the elevation of a relaxed_stations
       source; gate generator 'bezier' has no centerline``)
   * - an altitude on the loop and an explicit non-``raw`` order
     - ``ValueError`` (``generator 'acme.helix_loop' provides z, carried to the gates by
       index: order='raw' (the stations default)``)

A hand-built spec (not made by the preset) with ``spacing_metric="xyz"`` and no
altitude on the loop raises ``PlanError`` (``stage 'stations' (stations.raw):
spacing_metric='xyz' measures 3D arc length along the loop, which carries no altitude:
use a generator that provides z (raw.z), or source='relaxed_stations' with a non-flat
elevation``). In a gate pipeline,
an ``elevation`` slot before the anchors producer keeps its ``cl.*`` ports (it runs on
the centerline); the gate rewiring of `Elevation on gates`_ applies to the slots after
it.

.. code-block:: python

   spec = presets.gates(num_envs=64, device="cpu", generator="bezier",
                        source="relaxed_stations", gates=12,
                        elevation=("noise", {"z_noise_amplitude": 0.3,
                                             "z_min": -0.3, "z_max": 0.3}),
                        spacing_metric="xyz")
   spec.ids
   # ('generator', 'resample', 'width', 'relax', 'frames', 'elevation', 'stations',
   #  'order', 'carry_z', 'gate_relax', 'lift', 'tangents', 'gate_frames', 'validity')
   # the noise profile runs on the relaxed loop, the stations are 3D-equally spaced

A plugin stations stage gets ``spacing_metric`` filled into its params when its Params
class has the field; otherwise ``"xyz"`` raises ``ValueError``. Plugin stations stages
carry no centerline channels in this release: one may write ``anchors.z`` (it is then
treated as providing the altitude: ``carry.z``, ``order.raw``) and its own
``anchors.<package>.<name>`` channels, and a generator that provides ``z`` raises
``ValueError`` with one (``stations 'acme.st' carries no centerline channels (plugin
carriers are deferred); generator 'acme.helix_loop' provides z``).

``gate_arclen`` measures plan-view arc length, and stays so: its arc table feeds the
gate elevation that produces ``gates.z``, so a 3D metric there would be circular (and
whenever an altitude exists upstream, ``gate_arclen`` and the gate elevation are not
in the pipeline). The spacing is decided where the gates are placed, by the stations.

.. _pipeline-gate-elevation:

Elevation on gates
~~~~~~~~~~~~~~~~~~

The gate pipeline runs the same registered ``elevation`` stages as tracks, built-ins and
plugins alike: the planner rewires every ``elevation`` slot of a gate pipeline from the
centerline channels to their gate counterparts, ``cl.xy -> gates.xy``, ``cl.count ->
gates.count``, ``cl.arclen2 -> gates.cum``, ``cl.length2 -> gates.perim``, ``cl.z ->
gates.z``, and every plugin channel by prefix, ``cl.<package>.<name> ->
gates.<package>.<name>`` (``seed`` is unchanged). An explicit ``StageRef.ports`` entry
overrides one port. An ``elevation`` slot placed before the anchors producer (the
centerline elevation of `3D station spacing`_) is not rewired.
The presets emit plain ``StageRef("elevation", name, params)``, so
``spec.replace("elevation", StageRef("elevation", "acme.sinusoid_hills"))`` works on a
gate spec whose elevation is already non-flat (e.g. ``presets.gates(...,
elevation="noise")``). A flat gate spec has no ``gate_arclen`` slot, so also insert the
arc tables, as the ``PlanError`` says: ``spec.replace("elevation", StageRef("elevation",
"acme.sinusoid_hills")).insert_before("elevation", StageRef("gate_arclen",
"default"))``, which builds the spec of ``presets.gates(elevation="acme.sinusoid_hills")``
(or build with ``presets.gates(elevation=...)`` directly). A stage that sizes its buffers with ``b.layout.stride`` and
``b.write("z", wp.float32, per="cl")``, as the hills example of
:doc:`/contributing/writing-a-stage` does, runs on gates unchanged: ``per`` is checked
against the port's declared channel and the buffer sized by the wired one.

``GateGenConfig.z_profile`` names any registered elevation stage, as
``TrackGenConfig.z_profile`` does; a plugin gets its ``Params.from_legacy(config)``,
where ``config`` is the ``GateGenConfig``: only the ``z_*`` fields are shared with
``TrackGenConfig``, so a ``from_legacy`` written for tracks should read any other field
with ``getattr(config, name, default)`` (one that reads a ``TrackGenConfig``-only field
raises ``ValueError`` naming it). An unknown name raises at ``GateGenerator``
construction: ``GateGenConfig.z_profile: unknown elevation 'bogus'; available: ['flat',
'noise', 'random_walk', 'uniform']``. An elevation port left on a ``cl.*`` channel with no
gate counterpart raises ``PlanError``.

On gates, ``elevation.uniform`` and ``elevation.random_walk`` draw one altitude per gate
(0.2.0.dev3's gate path): the gates are the knots, so their ``z_control_points`` is not
read (``GateGenConfig`` has no such field; a value set through ``presets.gates(
elevation=("uniform", {...}))`` changes nothing). ``elevation.flat`` writes ``z_base``
into every real gate (``presets.gates(z_base=...)``, ``GateGenConfig.z_base``); on a
track it launches nothing and ``lift.flat`` adds the altitude, so a non-zero ``z_base``
on a track's ``elevation.flat`` raises: ``ValueError`` from ``presets.track``,
``PlanError`` from the planner, before anything is allocated.

.. _pipeline-gate-presets:

Gate presets: ``presets.gates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   presets.gates(*, num_envs, device, generator="bezier", generator_params=None,
                 source="auto", gates=None, max_gates=None, min_gates=4,
                 order=None, stations=None,
                 gate_radius=0.025, gate_solve_iters=8, gate_width=0.0,
                 gate_align="yaw_only", elevation="flat", z_base=0.0,
                 z_valid_grade=0.0, checks=("gate_default",), scale=1.0,
                 on_invalid="keep", n_max=384, num_points=256, half_width=0.1,
                 spacing=None, relax="xpbd", width="constant",
                 spacing_metric="xy", elevation_on=None,
                 registry=None) -> PipelineSpec

``presets.gates(num_envs=E, device=d)`` builds what
``legacy.to_gate_pipeline_spec(GateGenConfig(num_envs=E, device=d))`` builds: the same
stage list, equal params once resolved, byte-identical ``GateSequence``. It reads the
registry (``registry``, default ``track_gen.plugin.registry``) to see the generator's
``output``, ``provides`` and orderings, which decide the stage list.

- ``generator`` is a name (with ``generator_params``) or a ``StageRef`` of kind
  ``gate_generator`` or ``generator``; ``source`` is ``"auto"``, ``"anchors"``,
  ``"stations"`` or ``"relaxed_stations"`` (`Stations: gates from any track
  generator`_).
- ``gates`` is the station count of a centerline source (default 12); with an anchors
  generator it raises ``ValueError`` (the generator decides its own count). It fills the
  ``gates`` field of a ``stations`` selection that does not set it; a selection that sets
  another count raises ``ValueError``. ``max_gates`` defaults to 32 for anchors, to the
  station count for stations.
- ``order``, ``stations``, ``elevation``, ``relax`` and each entry of ``checks`` take the
  forms of `Presets`_ (a name, ``(name, params)``, a ``StageRef``, or ``None``:
  ``elevation=None`` is ``elevation.flat``, ``relax=None`` is ``relax.none``).
  ``order=None`` is the generator's default order: its preferred order when it accepts
  it, else ``"ccw"`` when it accepts it, else the first order it accepts (the rule of
  ``GateGenConfig.gate_ordering=None`` and of the conformance kit). ``checks`` starts
  with ``"gate_default"``. ``stations=`` with an anchors generator raises
  ``ValueError`` (it writes its own anchors).
- ``gate_radius``, ``gate_solve_iters``, ``gate_width``, ``gate_align``, ``z_base``,
  ``z_valid_grade``, ``min_gates`` and ``scale`` are the ``GateGenConfig`` fields, passed
  to every stage that takes them (one value per pipeline, see `Planner rules for gate
  pipelines`_). ``validity.gate_default`` gets them (and ``check_crossing="always"`` for
  posed gates) unless its params in ``checks`` set them; params setting another
  ``gate_radius`` or ``gate_width`` raise ``ValueError``. ``z_base`` is
  ``elevation.flat``'s altitude: a nonzero value with another elevation or with a
  generator that provides ``z`` raises ``ValueError``. ``gate_align`` is ``"yaw_only"``
  or ``"full_tangent"``.
- ``n_max``, ``num_points``, ``half_width``, ``spacing``, ``relax`` and ``width``
  configure the centerline sub-pipeline: a ``"stations"`` source reads ``n_max`` and
  ``num_points``, a ``"relaxed_stations"`` source all six (``width`` as in
  ``presets.track``), and setting one that the source does not read raises
  ``ValueError``.
- ``spacing_metric`` (``"xy"`` or ``"xyz"``) is the stations' arc-length metric, and
  ``elevation_on`` (``None``, ``"gates"`` or ``"centerline"``) where the elevation runs;
  both apply to a centerline source, and ``"xyz"`` with an anchors generator raises
  ``ValueError`` (`3D station spacing`_).
- ``on_invalid`` is ``"keep"`` or ``"poison"``.
- A legacy ``GateGeneratorSpec`` orders its own gates: ``order=`` with one raises
  ``ValueError``, choose its ordering with ``GateGenConfig(gate_ordering=...)``.

.. _pipeline-gate-spec:

Running a gate spec: ``GateGenerator(spec)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GateGenerator(config_or_spec, rng=None, *, seeds=None, stream=None)`` takes a
``GateGenConfig`` (as before) or a gate pipeline's ``PipelineSpec``, with the seed rules of
``TrackGenerator`` (``rng`` or ``seeds``, not both; with neither, env ``e`` gets seed
``e``). A track pipeline's spec raises ``ValueError`` pointing at ``TrackGenerator``,
anything else ``TypeError``.

- ``gen.config`` returns what was passed, so ``GateGenerator(gen.config, rng)`` rebuilds
  an identical generator.
- ``gen.program`` is the ``Program``; ``gen.num_envs``, ``gen.device`` and
  ``gen.max_gates`` read the batch whichever form was passed.
- ``gen.params`` lists ``"generator"`` plus every stage with a dynamic field.

``generate``, ``enqueue``, ``seeds``, ``inputs``, graphs, masked regeneration and
``Course`` (which builds its ``GateGenerator`` from ``CourseConfig.gen``) work as before.

.. _pipeline-gate-fail-bits:

Gate fail bits
~~~~~~~~~~~~~~

``validity.gate_default`` keeps 0.2.0.dev3's single fused validity kernel, so
``GateSequence.valid`` is byte-identical, and writes one bit per check into
``valid.fail``. ``GateSequence.fail_mask`` is its final version, and ``valid[e] ==
(fail_mask[e] == 0)``.

.. list-table::
   :header-rows: 1
   :widths: 12 16 72

   * - bit
     - name
     - set when
   * - 0
     - ``count``
     - ``count < min_gates`` or ``count > max_gates``
   * - 1
     - ``finite``
     - a real gate's position, tangent, forward, orientation, half_size, left or right
       is NaN or infinite
   * - 2
     - ``tangent``
     - a real gate's 3D tangent has ``dot(t, t) <= 1e-12``
   * - 3
     - ``spacing``
     - two real gates are closer in xy than ``2 * gate_radius``
   * - 4
     - ``crossing``
     - two gate bars cross (only when ``gate_width > 0``, or with
       ``check_crossing="always"``)
   * - 5
     - ``grade``
     - a chord's ``|dz| / ds`` exceeds ``z_valid_grade`` (only when it is ``> 0``)
   * - 6-15
     - reserved
     - TrackGen's future built-in gate checks
   * - 16-30
     - plugin checks
     - assigned by ``StageBuilder.fail_bit()``, in pipeline order (as for tracks)

``track_gen.pipeline.GATE_FAIL_BITS`` is this table (a read-only mapping, next to
``BUILTIN_FAIL_BITS``). A gate validity plugin is a ``validity`` stage that reads the gate
channels it checks and rewrites ``valid.fail`` exactly as a track check does (see
:ref:`stage-validity-check`); add it with ``presets.gates(...,
checks=("gate_default", "acme.max_height"))``. In a gate pipeline ``valid.fail`` must
start at ``validity.gate_default``.

``check_crossing`` is ``"width"`` (default: the crossing check runs when ``gate_width >
0``, as before) or ``"always"``, which posed gates need: their widths come from the
generator, not from ``gate_width``. The preset sets it for a generator that provides
``half_size``, and the planner refuses ``gate_frames.posed`` with ``"width"``.

.. _pipeline-gate-commit:

The gate commit
~~~~~~~~~~~~~~~

The gate Program ends with the commit of `The commit`_, over the ``GateSequence``:

- **Masked regeneration is exact.** ``generate(advance=True, mask=m)`` holds the
  ``GateSequence`` rows of every env with ``m[e] == 0`` before the stages and restores
  them after, bit for bit (``position``, ``tangent``, ``forward``, ``left``, ``right``,
  ``orientation``, ``half_size``, ``count``, ``valid``, ``fail_mask``, every attribute;
  ``generation`` is written for the masked envs only).
- **Invalid envs.** ``BatchSpec.on_invalid`` applies: ``"keep"`` (default) leaves an invalid env's computed gates and
  ``valid = 0``. ``"poison"`` leaves it with ``count = 0``, every per-gate field NaN
  (``orientation`` and ``half_size`` included), every attribute its padding and
  ``valid = 0``, keeping ``fail_mask``;
  ``CheckpointSet.from_gates`` and ``ProgressTracker`` then take their empty-course path
  for that env. There is no ``GateGenConfig.on_invalid``: use ``presets.gates(...,
  on_invalid="poison")`` or edit the spec's batch.
- **Cost.** As for tracks: no launch for an unmasked default generation, one for a derived
  ``valid`` (a check after ``validity.gate_default``) or for poison, two for a masked
  generation, plus per exported attribute two for a masked generation and one for
  poison. The held copy costs 80 B per gate slot (plus each attribute's size) and 12 B
  per env, allocated once.

.. _pipeline-gate-launches:

Launches
~~~~~~~~

The default gate Programs launch the kernels of 0.2.0.dev3, with the same dimensions and
scalars, in the same order (three kernels gained trailing array arguments, and
``_finalize_validity_k``'s grade-check wrap is also bounded by ``max_gates``: the same
arithmetic for every count <= max_gates). The one other difference: the normalizing generators (``polar``, ``checkpoint``, ``voronoi``) normalized
their ordered gates in place before copying the count; the ``normalize`` stage now runs
after the order stage's count copy, and runs the same arithmetic out of place
(``_normalize_extent_to_k`` reads the order's ``gates.xy`` and writes its own version:
same dimensions and scalars, one more array argument). The launch count and the bytes
are identical.

.. list-table::
   :header-rows: 1
   :widths: 64 18 18

   * - generation (CPU, launches per ``generate()``)
     - 0.2.0.dev3
     - now
   * - ``bezier`` / ``hull``, any ordering, flat, ``gate_solve_iters=8``
     - 14
     - 14 (identical)
   * - ``polar`` / ``checkpoint`` / ``voronoi``, flat, ``gate_solve_iters=8``
     - 14
     - 14 (the count copy before the normalization)
   * - masked (``advance=True, mask=m``)
     - as ``advance=True`` alone (one more than above: the epoch advance)
     - two more than 0.2.0.dev3 (hold, commit)
   * - ``on_invalid="poison"``
     - (not available)
     - one more (the commit)
   * - a plugin check after ``validity.gate_default``
     - (not available)
     - one more for the commit (the derived ``valid``), plus the check's own launches
       (and its parameter latch, for a check with dynamic params)

A non-flat elevation adds the ``gate_arclen`` launch, ``gate_solve_iters=0`` drops the
relax launch, as before.

.. _pipeline-gate-plan-rules:

Planner rules for gate pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides the rules of `build() and the planner`_, a gate pipeline raises ``PlanError``
before allocating anything when:

- the generator's ``capacity()`` does not fit the batch or the validity stage's
  ``min_gates`` (`Anchors generators`_), or ``capacity()`` returns keys other than
  ``gates`` / ``min_gates``, ``gates < 1``, or ``min_gates`` outside ``[0, gates]``;
- the order stage is one the generator does not accept (`The ordering registry`_);
- a provided attribute, or an ``anchors.*`` attribute a gate generator or a plugin
  stations stage writes, is not read, or a ``carry`` stage (``carry.attributes``
  included) follows an order other than ``order.raw`` (`Provided attributes`_, `Gate
  plugin channels`_);
- the stations measure ``spacing_metric="xyz"`` on a loop that carries no altitude
  (`3D station spacing`_);
- a stage has both ``cl.*`` and ``gates.*`` ports, a gate-domain stage uses a
  ``cl.<package>.<name>`` plugin channel, or an elevation port after the anchors
  producer stays on a ``cl.*`` channel without gate counterpart (`Two point domains`_,
  `Elevation on gates`_);
- stages disagree on ``gate_width`` or ``gate_radius`` (``gate_frames`` bakes the bar
  width, ``gate_relax.spheres`` the separation target, ``validity.gate_default`` both
  checks): one value per pipeline;
- ``gate_frames.posed`` runs with ``check_crossing="width"`` (`Gate fail bits`_);
- a ``GateSequence`` field has no writer, or ``valid.fail`` does not start at
  ``validity.gate_default``.

.. _pipeline-legacy-gates:

Legacy gate generators
~~~~~~~~~~~~~~~~~~~~~~

A ``GateGeneratorSpec`` (the 0.2.0.dev3 gate plugin ABI: ``generate(seeds, config, out,
scratch[, ctx])`` writing ordered plan-view gates into a staging view, ``alloc_scratch``,
``max_gates``, ``supported_orderings``) keeps working, through an adapter slot that
writes ``gates.xy`` and ``gates.count`` directly. It orders its own gates as before, so no
order or normalize stage follows it, and it reads the user's ``GateGenConfig``, exactly
the fields it read before. A ``capturable=False`` spec is a ``HOST`` generator: it runs
eagerly as the Program's first segment, ahead of a captured tail (see `Capture classes
and segments`_); an ``accepts_ctx=False`` spec runs inside ``ctx.scope()`` (see
:doc:`cuda-graph`).

It is **deprecated**: registering one stays silent, but binding one (``GateGenerator``,
``Course``, ``build``, ``legacy.to_gate_pipeline_spec``, the conformance kit) emits a
``DeprecationWarning`` once per registry and name, attributed to the calling code:
``GateGeneratorSpec 'acme.ring' (...) is deprecated: write a Stage with
kind='gate_generator' and output='anchors' (docs/contributing/writing-a-generator.rst);
legacy gate generators run through an adapter until 1.0``. Port it with
:ref:`gate-generator-stage`.

Not open yet
------------

- **Results** (roadmap PR13b): ``Track.layout``, ``validate()``, ``view()`` and
  ``copy_from()``; a ``clone()`` that iterates the fields; ``out=`` on the facades and
  ``out.attributes`` with keys from another pipeline (mixed batches); the capability
  protocols (``HasCenterline``, ``HasBand``, ``HasGates``); utilities that validate
  their input; ``randomize.uniform`` and the removal of the parameter latch; an
  ingestion stage for external centerlines; the ``closed`` flag through the kernels.
- **Channels** (later): carries by plugin stations stages (a public
  ``StageBuilder.carried()``); carried channels written by a stage rather than the
  generator (an ``annotate`` kind, ``interp=`` on stage writes); an order that publishes
  its permutation, so attributes reach the gates through any order;
  ``ops.arc_resample(tables=)``; a public constant declaration for width stages; an
  arc-fraction attribute policy; renaming on ``export=``; per-env attributes;
  ``GateGenConfig.spacing_metric``; an elevation on the raw loop (``"xyz"`` with
  ``source="stations"`` needs a ``z`` provider); a dynamic ``width.constant``.
- **Gates** (later): composite generators (``Recipe``), provided attributes other than
  ``z`` from ``stations`` (``tangent``, ``half_size``, ``roll``), ``roll`` on
  ``gate_frames.default``, ``preferred`` hints for track
  generators, split gate checks, and a ``GateGenConfig.on_invalid`` field.
- Opening ``resample``, ``frames``, ``offset`` and ``lift`` to plugins; split built-in
  checks (``validity.turning``, ...); an aliasing in-place write form; a
  ``TrackGenConfig.on_invalid`` field; ``Course`` built from a ``PipelineSpec``, and a
  segmented ``Course`` (over a generator with a ``HOST`` stage, ``Course.regenerate()``
  runs eagerly).

The 0.2.0.dev2 composition (``warp_pipeline._run_pipeline``, ``inflate_warp``,
``_inflate_warp_alloc``) is still importable. ``TrackGenerator`` no longer calls it; the
tests use it as the byte-identity reference. ``gen._scratch`` is a view of the Program's
buffers kept for the same reason; new code reads ``gen.program.tap``. The 0.2.0.dev3 gate
composition (``warp_gate._run_gate_pipeline``, ``_gate_warp_alloc``) is removed: the
tests keep a frozen copy as the byte-identity reference, and ``GateGenerator._scratch`` is
gone (read ``gen.program.tap``). The ``warp_gate`` helpers 0.1 gate plugins call
(``alloc_order_scratch``, ``order_points``, ``finish_ordered_gates``, ``_GateStaging``)
stay, deprecated, until 1.0; the library no longer calls them.
