.. _pipeline-stages:

Pipeline stages
===============

A generator runs its :term:`spec` as a ``Program``: an ordered list of :term:`stages
<stage>` that hand each other per-:term:`env` :term:`channels <channel>`, planned once
when the generator is built and launched once per :term:`generation`. This page
describes that machinery for tracks: the stage kinds, how ``build()`` plans a spec, the
``Program`` it returns, the channel catalogue, the default stages, width, fail bits, the
commit, per-point attributes, taps and the :ref:`parameter latch <pipeline-latch>`.
:ref:`generating` shows the user-facing side (presets, specs, Params); :ref:`gate-pipeline`
lists what differs for gates; :doc:`cuda-graph` covers how a Program is recorded and
replayed.

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
     - ``cl.z`` (``gates.z`` in a gate pipeline, :ref:`Elevation on gates <pipeline-gate-elevation>`)
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
   * - ``randomize``
     - ``uniform``
     - another stage's dynamic ``float32`` field (no channel)
     - TrackGen's private catalogue (not open to plugins; :ref:`Per-env draws <pipeline-randomize>`)
   * - ``resample``, ``frames``, ``offset``, ``lift``
     - see `The default track stages`_
     - their catalogue channels
     - TrackGen's private catalogue (not open to plugins)
   * - ``normalize``, ``carry``, ``gate_relax``, ``gate_arclen``, ``tangents``,
       ``gate_frames`` (and ``lift.gates``)
     - see :ref:`The gate stages <pipeline-gate-stages>`
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

.. _pipeline-stages-build-and-the-planner:

``build()`` and the planner
---------------------------

``build(spec, *, registry=None, seeds=None, out=None, stream=None, prepare=True)``
checks the spec before it allocates or launches anything, then allocates once:

1. **Batch and ids.** ``num_envs >= 1``, ``n_max >= 3``, ``closed`` a ``bool``,
   ``on_invalid`` in ``("keep", "poison")``, ``max_gates`` ``None`` or an int ``>= 2``;
   slot ids are unique; exactly one generator stage (kind ``generator`` or
   ``gate_generator``), the first stage that is not a randomizer, with the id
   ``"generator"`` (only randomizers of the generator's own fields may precede it); only
   the generator has a ``capacity``, and only ``"raw"``; ``seeds`` is one of the forms
   below. Then **topology**: with ``closed=False`` the planner looks up every stage's
   class (nothing is bound or validated yet) and raises, naming those that declare
   ``topology="closed"`` (:ref:`results-topology`). A gate
   pipeline starts with a ``gate_generator``, or with a ``generator`` followed by a
   ``stations`` stage; a ``gate_generator`` in a track pipeline is refused.
2. **Resolve.** The generator and every stage of an open kind are resolved through
   ``registry`` (default ``track_gen.plugin.registry``), the generator exactly as
   ``TrackGenerator`` (or, for a ``gate_generator``, ``GateGenerator``) resolves
   ``config.generator``. The other stages come from TrackGen's built-in catalogue.
3. **Params.** Each stage's params are coerced to its :term:`Params` class and validated (the
   built-ins with the messages ``TrackGenConfig`` / ``GateGenConfig`` use for the same
   fields). Stages with a shared ``half_width`` or ``gate_radius`` must agree on its
   value. Bar width is owned by the dynamic ``gate_frames.default.gate_width`` field;
   gate validity reads the resulting bars rather than a duplicate setting. For ``half_width``
   that binds ``width.constant`` and
   the plugin stages that keep such a field (the nominal width; the per-point width is
   ``cl.half_width``). In a gate pipeline each stage's ``Layout`` is derived from its
   ports before its params are validated (see :ref:`Two point domains <gate-pipeline-two-point-domains>`).
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
   built-in checks, truncation included, and is refused), and ``lift.flat``, which lifts at a constant ``z_base``, may not follow an
   elevation other than ``elevation.flat`` (use ``lift.elevated``). A gate pipeline has
   the same rules for the ``GateSequence`` and ``validity.gate_default``, plus its own
   (:ref:`Planner rules for gate pipelines <pipeline-gate-plan-rules>`). Each randomizer is checked against its target
   (:ref:`Per-env draws <pipeline-randomize>`).
5. **Capture class and segments.** Each stage's capture class comes from its
   ``capture_class(params)`` (a ``GeneratorSpec``:
   ``STATIC``, or ``HOST`` with ``capturable=False``). The Program's is the highest
   (``STATIC`` < ``BOUNDED`` < ``HOST``), and the stages are grouped into segments of
   equal host-ness (see `Capture classes and segments`_). With
   ``PipelineSpec.require_caller_capturable=True`` a ``HOST`` stage raises
   ``PlanError``, naming every one.
6. **Allocate.** The ``SeedState`` (its own checks run first), the ``Track`` or the
   ``GateSequence`` (or use the one passed as ``out=``, validated beforehand, which may
   be a view of a larger result written by several generators, see
   :ref:`results-out`), the ``[E]`` effective-seed buffer, one buffer per
   channel version and each stage's scratch, namespaced ``scratch:<id>/<name>``. Then
   each stage's ``build`` runs, once, in pipeline order, against a ``StageBuilder``;
   right after it, every write port the stage declares must have been written, and a
   validity stage must own a fail bit (fail bits are handed out in this order, see
   `Fail bits`_). Last, the commit and its held copy of the result.
7. **Program.** With ``prepare=True`` the Program runs once eagerly (the
   :ref:`warm-up <cuda-graph-warm-up>`, with every bounded loop unrolled).

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

.. snippet: expect-error=PlanError

.. code-block:: python

   from track_gen.pipeline import PlanError, StageRef, build, presets

   spec = presets.track(num_envs=64, device="cpu")
   build(spec.remove("width"))
   # PlanError: stage 'relax' (relax.xpbd) reads 'cl.half_width' (port 'half_width'),
   #   but no earlier stage writes it; the width stage runs before relax ... (generator,
   #   resample, width, relax, uniform, ...): move it, or insert
   #   StageRef('width', 'constant') before 'relax'

``seeds`` is anything ``SeedState`` accepts (``None``, an ``int``, an ``[E]`` int32
``wp.array`` of :term:`seeds <seed>`, a ``PerEnvSeededRNG``) or a ``SeedState`` for the
batch. Anything else,
a NumPy array or a list included, raises ``TypeError`` before allocating. With ``out=``
a view starting at env ``k`` of its root result, ``None`` and an ``int`` are offset by
``k`` (env ``e`` gets ``base + k + e``), so the generators of a mixed batch get the
seeds one batch would (:ref:`results-mixed-batches`).

``build`` derives the generator's config from the ``BatchSpec`` and the generator's
``StageRef`` (its params, and ``capacity["raw"]`` as ``num_points``, 256 when absent).
That covers the built-ins and Stage generators. A ``GeneratorSpec`` plugin that reads
config fields outside its Params class needs ``TrackGenerator`` with a config. A gate
generator is bound the same way from a ``GateGenConfig`` derived from the spec
(``num_envs``, ``device``, ``max_gates``, ``scale``, its params, and ``min_gates``,
``gate_ordering`` and the gate scalars from the stages that hold them).

``build`` accepts both kinds of spec: the ``Program`` it returns has a ``Track`` or a
``GateSequence`` as its ``result``.

.. _pipeline-program:

The ``Program``
---------------

``Program`` is a ``GraphRunner`` launchable, so it follows the rules of
:doc:`cuda-graph` exactly as ``TrackGenerator`` does: recorded into a caller's
:term:`capture`, replayed from its own CUDA graph, or run eagerly.

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
  instance with the same array addresses forever (``out`` when one was passed), and
  ``layout`` its ``Layout`` (:ref:`results-layout`).
- ``spec`` is the resolved spec: every ``StageRef.params`` is the stage's Params
  instance.
- ``tap(channel, *, after=None)``, ``channels``, ``fail_bits`` and ``on_invalid``: see
  `Taps`_, `Fail bits`_ and `The commit`_.
- ``attributes``: the per-point attributes the result exports, a read-only mapping from
  key to ``AttributeInfo`` (see `Per-point attributes in the result`_).
- ``seeds`` (the ``SeedState``), ``seed`` (the ``[E]`` effective seed), ``inputs``
  (``"regen_mask"`` plus the inputs the stages declare), ``input_info`` (a read-only
  mapping from each name of ``inputs`` to a ``track_gen.pipeline.InputInfo``: ``name``,
  ``dtype``, ``per``, ``length``, ``latched``, ``stage_id``), ``params`` (a
  ``ParamTable`` keyed by stage id; the :term:`dynamic fields <dynamic field>` of the
  built-ins are the
  generators' shape knobs, the elevation stages' ``z_min`` / ``z_max`` /
  ``z_max_step`` / ``z_noise_amplitude``, ``width.from_channel``'s ``scale`` / ``lo`` /
  ``hi`` and a randomizer's ``lo`` / ``hi``, see :ref:`Dynamic built-in knobs <pipeline-dynamic-builtins>`) and
  ``graph`` (``None``
  for a Program with ``HOST`` stages, whose per-segment graphs ``segment_graphs(key=None)``
  returns, ``None`` at the ``HOST`` segments).
- ``capture_class`` (the highest of its stages', ``STATIC`` < ``BOUNDED`` < ``HOST``),
  ``capturable`` (``True`` when a caller can record it: no ``HOST`` stage),
  ``host_stage_ids`` and ``segments`` (see `Capture classes and segments`_).
- ``prepared`` (whether the warm-up has run), ``device``, ``default_stream`` (the stream
  a call runs on when it passes no ``stream=`` and no ambient scope applies), ``runner``
  (its ``GraphRunner``).

.. _pipeline-segments:

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

A ``GeneratorSpec(capturable=False)`` gives ``(HOST: generator) + (STATIC: the
rest)``; a ``HOST`` relax stage between capturable stages gives three segments. Outside a
capture on ``cuda`` the Program records every capturable segment into a graph of its own
and runs the ``HOST`` ones eagerly between the graph launches (:ref:`cuda-graph-host`).
Under a caller's capture, ``run`` and ``enqueue`` of a
Program with ``HOST`` stages raise ``NotCapturableError`` naming them, before any launch.

``enqueue_segment(i, ctx, *, advance=False, mask=None)`` lets an advanced caller do what
the Program does outside a capture: record the capturable segments into its own graph and
run the ``HOST`` segments eagerly.

- Segment ``0`` opens with the seed prologue (the epoch advance, the effective seed, the
  parameter latch and, with a ``mask``, the hold); the last segment closes with the
  commit and the ``generation`` write. So a write to any dynamic parameter must be
  enqueued before segment ``0`` to apply to that generation: one recorded in a later
  segment's graph (after the latch) applies at the next generation. A one-segment Program does both, and
  ``enqueue(ctx)`` is every segment in order. Pass the same ``advance`` and ``mask`` to
  every segment of a generation, on one stream: a recorded segment keeps the arguments
  it was recorded with, so run the eager segments with those.
- A ``HOST`` segment raises ``NotCapturableError`` when ``ctx`` is capturing; a capturable
  segment can be recorded into a caller's capture once the Program is prepared, even when
  other segments are ``HOST``.
- Mixing ``advance`` / ``mask`` between the segments of a generation, running them out
  of order, or skipping one is not checked (a replayed segment runs no Python, so the
  Program cannot see its arguments) and leaves the output undefined.


The checks run on the host before any launch, in this order: ``run`` validates ``mask``
and ``advance`` (``ValueError``; ``enqueue`` and ``enqueue_segment`` leave that to their
caller) and ``enqueue_segment`` its index (``IndexError``); then ``run``, ``enqueue`` and
``enqueue_segment`` of a ``HOST`` segment refuse a capturing ``ctx``
(``NotCapturableError``, also for a Program built with ``prepare=False``); then
recording a Program that was never prepared raises ``CaptureError``.

``PipelineSpec(..., require_caller_capturable=True)`` refuses ``HOST`` stages at build,
before anything is allocated:

.. snippet: skip=plugin-package

.. code-block:: python

   build(dataclasses.replace(spec, require_caller_capturable=True))
   # PlanError: PipelineSpec.require_caller_capturable=True, but stage(s) 'generator'
   #   (generator.acme.blob) is HOST (they read device data back or run host control
   #   flow, so they run eagerly and refuse a caller's capture): use STATIC / BOUNDED
   #   stages or drop require_caller_capturable

A ``GeneratorSpec(capturable=False)`` is named ``'generator' (generator 'acme.old',
GeneratorSpec capturable=False)``.

Channels
--------

Stages talk through named per-env channels; a stage declares the ones it reads and
writes as ``{local port: channel}`` maps (``Stage.reads`` / ``Stage.writes``), and
``StageBuilder.read(port)`` / ``write(port, dtype, per=...)`` hand it the buffers.
``per="env"`` is ``[E]``, ``per="cl"`` is ``[E * N_max]`` (:term:`N_max` points per
env), ``per="raw"`` is
``[E * raw]``; the gate pipeline adds ``per="anchors"`` (``[E * A]``) and ``per="gates"``
(``[E * G]``), and its channels (``anchors.*``, ``gates.*``) are listed under
:ref:`Gate channels <pipeline-gate-channels>`. The track catalogue in this release:

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
     - A gate generator, or a ``stations`` stage (:ref:`Gate plugin channels <pipeline-gate-plugin-channels>`).
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

.. snippet: skip=fragment

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
  ``anchors.<attr>`` only when a later stage reads that image (:ref:`3D station spacing <pipeline-stations-xyz>`,
  :ref:`Gate plugin channels <pipeline-gate-plugin-channels>`).
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
themselves. A resampler with nothing to carry runs its plain lookup kernels, and a
constant family costs no interpolation (`Width`_), so the default pipelines pay nothing
for the mechanism.

**Padding.** NaN (every component of a float vector) or ``-1`` (``int32``) at every
slot past ``count[e]``, written by every carrier (an output point outside the real ones
has the table sentinel ``i0 = i1 = -1``, ``frac = NaN``) and by the commit when it
poisons an env. One exception: the uniform image of a *constant* ``cl.half_width``
(``width.constant``) fills its value into every slot, padding included; read
``Track.half_width`` for a NaN-padded half-width.

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
     - one per raw family; none (the plain lookup kernels) without one.
   * - ``resample.uniform``
     - one per non-constant family; a constant ``cl.half_width`` costs one fill (the
       final ``cl.half_width``, which every pipeline writes).
   * - ``stations.raw``, ``stations.relaxed``
     - one per family a later stage reads; none (the plain stations kernel)
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

.. snippet: expect-error=PlanError

.. code-block:: python

   StageRef("width", "from_channel", ports={"src": "cl.acme.hw"})   # wired

   build(presets.track(num_envs=64, device="cpu", width="from_channel"))
   # PlanError: stage 'width' (width.from_channel): read port 'src' (cl.*) must be
   #   wired: StageRef('width', 'from_channel', ports={'src': 'cl.<pkg>.<name>'});
   #   the cl image of a generator's raw.<pkg>.<name> is cl.<pkg>.<name>

With plugin channels written before the stage, the message lists them instead
(``plugin channels written before it: ['cl.acme.grip', 'cl.acme.hw', ...]``).

A required port is never a write port. ``registry.describe(kind, name).reads`` shows it
as is (``{"src": "cl.*", "count": "cl.count"}`` for ``width.from_channel``).

.. _pipeline-stages-the-default-track-stages:

The default track stages
------------------------

``legacy.to_pipeline_spec(config)`` and ``presets.track(...)`` produce this list.
``width.constant`` launches nothing (its value is filled at build), and the uniform
resample fills the final ``cl.half_width`` with it (see `Width`_). The kernels that read a
dynamic built-in knob read it from an ``[E]`` array, filled by the `Parameter latch`_
at the start of each generation (:ref:`Dynamic built-in knobs
<pipeline-dynamic-builtins>` lists them).

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
``validity.default`` take no ``half_width``: they read ``cl.half_width``. The elevation
stages' ``z_min`` and ``z_max`` (and ``z_max_step`` for
``random_walk``, ``z_noise_amplitude`` for ``noise``) are dynamic (:ref:`Dynamic built-in
knobs <pipeline-dynamic-builtins>`). ``elevation.flat`` takes ``z_base`` too (the gate altitude; it must be 0 on a track,
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
``resample.uniform`` reading the constant-spacing version. The price is one buffer per
rewrite. An iterative stage needs no extra copy: a
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

.. snippet: skip=plugin-package

.. code-block:: python

   spec = presets.track(
       num_envs=64, device="cpu", generator="acme.ribbon", half_width=0.1,
       width=StageRef("width", "from_channel", {"scale": 1.0, "hi": 0.2},
                      ports={"src": "cl.acme.hw"}))

**Constant width keeps its kernels.** ``width.constant`` declares its value constant (a
private declaration of the built-in). A consumer that reads a constant ``cl.half_width``
runs its scalar-width kernels with that value as a host scalar; any other width selects
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
bit-exact against the scalar kernels' double-precision host scalars (for 24 of 104
``(half_width, margin)`` pairs tried; at ``half_width=0.07, margin=0.2`` a full solve
moves 3909 of 12288 points by up to ``6e-3``). The constant path therefore keeps every
default pipeline on the scalar kernels, and the per-point kernels reproduce the scalar ones at
constant width when given the same ``float32`` scalars. A plugin width that writes a
constant value runs the per-point path: its outputs are close to, not equal to,
``width.constant``'s.

**The floor and the thickness checks read the final points.** Fail bit 4 is set when a
real final point ``i < count`` has ``not (cl.half_width[i] > w_floor)`` (a NaN fails): the
band pinches to the floor somewhere. At constant width it gives the same answer at
every point of every env. Both checks read the half-width the uniform resample carried to the final
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
   spec.ids
   # ('generator', 'resample', 'width', 'relax', 'uniform', 'frames', 'width_kappa',
   #  'offset', 'elevation', 'validity', 'lift')

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
provider puts the altitude on the loop before the stations (:ref:`3D station spacing <pipeline-stations-xyz>`).

.. _pipeline-fail-bits:

Fail bits
---------

Validity is a bitmask of :term:`fail bits`, the ``[E]`` int32 channel ``valid.fail``: bit ``i`` is set when
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
     - Reserved built-in check range
   * - 16-30
     - plugin checks
     - assigned by ``StageBuilder.fail_bit()``, in pipeline order
   * - 31
     - unused
     - never set (the sign bit)

``validity.default`` owns bits 0 to 7 and computes all of them in one fused kernel,
which also writes its own 0/1 flag ``track.valid``. An env with ``count == 0`` gets bit 1
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

.. snippet: skip=plugin-package

.. code-block:: python

   spec = presets.track(num_envs=64, device="cpu",
                        checks=("default", ("acme.max_curvature", {"kappa_max": 8.0})))
   gen = TrackGenerator(spec, rng)
   gen.program.fail_bits["acme.max_curvature"]    # 16
   track = gen.generate()
   too_curvy = (track.fail_mask.numpy() >> 16) & 1

``Track.fail_mask`` is a required array. A generated Track has
``valid[e] == (fail_mask[e] == 0)`` for every env. Manually allocated storage begins
with zero count, validity and fail mask, and an empty ``fail_bits`` mapping; the
mapping is populated when a pipeline writes it.

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
what the commit does with an :term:`invalid env`, one whose ``fail_mask`` is not zero
(:ref:`invalid-envs` is the user-facing account):

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - value
     - an invalid env leaves the commit with
   * - ``"keep"`` (explicit opt-in)
     - its computed geometry, and ``valid = 0``.
   * - ``"poison"`` (default)
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
the masked envs (the others are restored). The utilities answer an
invalid env the same way under both policies (:ref:`utilities-invalid-envs`); keep gating
rewards and observations on ``valid``. Poison is the default for configs, presets,
``BatchSpec`` and missing policies in saved ``/2`` batches.

**Cost.** The commit launches only when it has work to do:

.. list-table::
   :header-rows: 1
   :widths: 54 46

   * - generation
     - launches added
   * - unmasked, explicit ``on_invalid="keep"``, ``validity.default`` the last check
     - none (156 launches for a default ``bezier`` generation with explicit keep, the four latches of
       ``min_num_points``, ``rad``, ``handle_clamp_frac`` and ``size`` included:
       `Parameter latch`_)
   * - unmasked, explicit keep, with a check after ``validity.default``
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

``Program.on_invalid`` reports the policy. Both config classes expose
``on_invalid="poison"``. Set it to ``"keep"`` explicitly for failed-geometry diagnostics,
or replace a spec's frozen batch:

.. code-block:: python

   import dataclasses

   from track_gen import TrackGenConfig
   from track_gen.pipeline import legacy

   config = TrackGenConfig(num_envs=64, device="cpu", on_invalid="keep")
   spec = legacy.to_pipeline_spec(config)
   spec = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch,
                                                              on_invalid="poison"))
   assert config.on_invalid == "keep" and spec.batch.on_invalid == "poison"

.. _pipeline-attributes:

Per-point attributes in the result
----------------------------------

``Track.half_width`` is the band's half-width per point, ``[E * N_max]`` ``float32``:
the value on the real points, NaN on the padding (the constant on every real point at
constant width). It is ``offset.default``'s ``track.half_width``, so the band and the
field agree: ``|outer - center| == half_width`` on every real point. It is required
on manually built Tracks too; a pipeline with no road offset keeps an allocated
NaN half-width array where no width is available.

``Track.attributes`` maps a key to the final version of each per-point plugin channel
exported with ``export=``; the key is the channel's name without its domain prefix:

.. snippet: skip=plugin-package

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
  ``gates.<package>.<name>`` channels (:ref:`Gate plugin channels <pipeline-gate-plugin-channels>`), with one difference:
  a family with no index path to the gates is left out (best effort).
- ``Program.attributes`` maps each key of the result to an ``AttributeInfo`` (``key``,
  ``channel``, ``dtype``, ``interp``, ``padding``, ``domain``;
  ``track_gen.pipeline.AttributeInfo``), the policy a consumer that samples an
  attribute along the track needs (``interp`` is ``None`` for a channel a stage wrote on
  the result's points, or one carried by index only).
  ``registry.describe("generator", name)`` lists the raw channels a generator declares
  (``interp`` and ``export`` are decided in ``build``).

**Caller-owned results** (``build(..., out=...)``, ``TrackGenerator(..., out=...)``)
follow one rule for the optional fields (``fail_mask``, ``generation``, ``half_width``,
attribute keys): a field the caller leaves ``None``, or a key it leaves
out of ``out.attributes``, is not added to the caller's object, and the Program keeps that
work buffer for itself (``tap`` reaches it). A key of ``out.attributes`` the Program
exports is bound after a check (``[E * N_max]``, the dtype, the device, contiguous). On a
view of a larger result (a mixed batch), a key it does not export (a family it does not
write, or one it carries without exporting it) is checked, filled once with its padding
at build and never written again, so the batch can share one attribute array between
generators that export it and generators that do not; on a whole result such a key
raises ``ValueError`` (:ref:`results-out-attributes`).

.. _pipeline-taps:

Taps
----

``Program.tap(channel, *, after=None)`` returns the working buffer of a channel: its
final version, or with ``after=<stage id>`` the version visible right after that stage.
It replaces reading private scratch (``gen._scratch``) in visualisation and benchmark
code:

.. code-block:: python

   from track_gen import TrackGenerator

   gen = TrackGenerator(presets.track(num_envs=64, device="cpu"))
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

.. _pipeline-latch:

Parameter latch
---------------

A :term:`dynamic field` has two ``[E]`` arrays per stage: the *live* array that
``gen.params[stage][field]`` returns, which you write (``set``, your own kernels, a
curriculum), and a *working copy*, which the stage's kernels read. At the start of every
generation, right after the epoch advance and the effective seed, the Program's prologue
*latches* each live array into its working copy, for the envs it regenerates: every env,
or only the masked ones under ``generate(advance=True, mask=m)``. An input a stage
declares ``per="env"`` is latched the same way. Three consequences:

- a write to a live array applies at the next generation, to the envs that generation
  regenerates;
- an env a masked regeneration leaves out keeps the values its rows were built with, so
  masked regeneration stays exact for dynamic parameters too (`The commit`_);
- in a Program with ``HOST`` stages the latch belongs to segment ``0``: a write recorded
  into a later segment's graph applies at the following generation (`Capture classes
  and segments`_).

A *drawn* field does not go through the latch: the stage that draws it (a
``randomize.uniform`` stage) writes the one array the target reads, during the generation. So the runtime
arrays come in two kinds:

.. list-table::
   :header-rows: 1
   :widths: 16 32 52

   * - kind
     - what
     - when a write applies
   * - latched
     - dynamic parameters, ``per="env"`` inputs
     - Snapshotted at the start of a generation: a write applies to the envs the next
       generation regenerates.
   * - unlatched
     - a drawn field; a ``per="raw"`` input such as ``inputs["centerline.xy"]``
     - Read during the generation. A drawn field belongs to the stage that draws it
       (``set`` raises, a direct write is overwritten). A ``per="raw"`` input holds your
       latest write: write it on the generation's stream, or order the two with events.

``Program.input_info`` (``gen.input_info``) says which inputs are latched, and
``StageParams.drawn`` (``gen.params[stage].drawn``) maps each drawn field to the id of the
stage that draws it.

**Randomizers.** ``randomize.uniform`` is a ``STATIC`` stage of a closed kind. Its Params,
``RandomizeUniformParams``, hold ``stage`` and ``field`` (static: the target) and ``lo``
and ``hi`` (dynamic, so latched). It reads ``seed``, writes no channel, and draws with
the salt ``b.salt("draw")`` under its own id, so two randomizers never share a stream
and a draw does not depend on the randomizer's position. It is listed right before its
target and runs there, so the target reads this generation's draw; only the generator's
randomizers may precede the generator. Before a ``HOST`` generator, those randomizers
form segment ``0`` (a graph segment that opens with the prologue) and the generator
segment ``1``. :ref:`pipeline-randomize` has the user-level rules and the checks
``build`` runs on them.

**What a stage read.** ``gen.params[stage].in_effect(field)`` returns the ``[E]`` array
the stage read for the current rows: the working copy of a latched field, the drawn array
of a drawn field. Without ``out=`` it is that internal array (zero-copy, a stable pointer,
recordable, read-only by contract); ``in_effect(field, out=arr)`` enqueues one copy into
your array. A latched field is current once segment ``0`` has run; a drawn field once the
segment that holds its drawing stage has run. Both are written on the generation's stream.
:ref:`pipeline-in-effect` is the user-level account.

**Kernels and bounds.** Each built-in kernel that reads a dynamic field (the stages and
fields are listed in :ref:`pipeline-dynamic-builtins`) loads its env's value once and
clamps it to the field's declared bounds with comparisons, so a NaN and every in-range
value, signed zeros included, pass bit for bit, and an unbounded side is not clamped.
Then it runs the body of the scalar-argument kernel it replaces, verbatim: a
configuration gives the same bytes through either kernel. A scalar is
checked on the host instead, in ``float32`` for a ``float32`` field (at construction, by
``set``, for a randomizer's initial range). ``StageParams.bounds(field)`` reports the
bounds, narrowed per instance (``min_num_points``: ``(2, max_num_points)``, which ``set``
also enforces), so the value a built-in kernel used is
``clamp(in_effect(field), *bounds(field))``. Used as written, without a clamp:
``width.from_channel``'s ``scale`` / ``lo`` / ``hi`` (it clamps its output), a
randomizer's ``lo`` / ``hi`` (it clamps its draw).

**Size.** ``size`` multiplies where ``scale`` already acts, after it: the corner grid
``((x * cell_size + jitter) * scale) * size[e]``, or the normalization target
``target * size[e]`` before its division. Neither product feeds an addition, so a size of
``1`` is bit-exact on CPU and CUDA (no fused multiply-add is created or removed; a CPU
test locks the generated CUDA source of each kernel pair). ``repulsive``'s final copy
multiplies only finite values (``where(size == 1 or not finite, v, w * size)``, with ``w``
the finite-gated value): a size-1 copy keeps every finite value bit for bit, and a
non-finite point is copied unscaled at every size (a NaN bit for bit, payload included).

**Corner count.** ``_sample_counts_dyn_k`` loads env ``e``'s ``min_num_points``, clamps
it to ``[2, max_num_points]`` and draws the count as the scalar kernel does.

**Cross-field rules.** At build, every stage whose Params class defines the private
``_check_draws`` hook is checked, drawn or not: crossing ``z_min`` / ``z_max``
randomizers raise ``PlanError``; an inert elevation window, a ``rad`` drawn above its
clamp, a ``generator.size`` range (or construction value) that would truncate the nominal
longest loop at the pipeline's ``n_max`` emit a ``UserWarning``. :ref:`tuning-per-env`
and :ref:`tuning-size` give the rules from the user's side.

**Launches.** One latch launch per latched dynamic field per generation, right after the
effective seed (and the advance), in slot order then field order: an ``int32`` field
through ``_latch_all_i32_k``, a ``float32`` one through ``_latch_all_f32_k``, and under a
mask the ``_latch_masked_*`` twins. A drawn field has none. A randomizer adds two launches
per generation: its draw, and the latches of its ``lo`` and ``hi``, minus the latch its
target field no longer needs. In a launch trace, each kernel that reads a knob from an
array keeps the position, dimensions, arrays and other scalars of its scalar-argument
twin; its knob scalars become arrays (``_assemble_dyn_k`` also takes the four bound
scalars; ``_sample_counts_dyn_k`` and ``_finalize_frame_dyn_k`` take the array in place
of ``lo`` and ``gate_width``), the size kernels append the ``[E]`` size array, and
``repulsive``'s final copy is the ``_copy_sized_k`` launch.

.. list-table::
   :header-rows: 1
   :widths: 64 18 18

   * - generation (CPU, launches per ``generate()``)
     - launches
     - of which latches
   * - default track (``bezier``, flat)
     - 157
     - 4
   * - ``hull`` / ``polar`` / ``voronoi`` / ``checkpoint``
     - 155 / 147 / 155 / 151
     - 3 / 3 / 3 / 4
   * - ``bezier`` + elevation ``uniform`` / ``random_walk`` / ``noise``
     - 163 / 164 / 162
     - 6 / 7 / 7
   * - ``bezier`` + ``randomize={"generator.rad": (lo, hi)}``
     - 159
     - 5 (and the draw)
   * - gates: ``bezier`` / ``hull`` (the default gate pipeline)
     - 17
     - 2 (``size``, ``gate_width``)
   * - gates: ``polar`` / ``checkpoint`` / ``voronoi``
     - 19 / 19 / 20
     - 4 (two ring knobs, ``normalize.size``, ``gate_width``)
   * - gates: ``bezier`` + elevation ``uniform`` / ``random_walk`` / ``noise``
     - 20 / 21 / 21
     - 4 / 5 / 5
   * - gates: ``source="stations"`` (``bezier``, 12 gates)
     - 33
     - 5 (the track generator's four, ``gate_width``)

These counts include the default poison commit; explicit ``on_invalid="keep"``
removes that launch when the default validity check is last. They were remeasured
on the CPU pipeline.

A masked generation runs the same latches as ``_latch_masked_*`` launches after the
advance and the masked effective seed, and the hold comes right after them (index 6 of a
default track generation, 4 of a default gate generation).

Current limits
--------------

The pipeline supports closed loops. Capacity, buffer-shaping counts and execution
structure are static. Only declared dynamic fields change per environment without
rebuilding; uniform randomizers target dynamic float32 fields.

Plugin extension kinds are generator, gate generator, relaxation, width, elevation,
validity, ordering and stations. The remaining structural stages are built in.
Provided gate attributes require index-preserving raw order. A ``Course`` with a host
stage regenerates eagerly rather than recording that host work into a CUDA graph.
See :ref:`stage-capture-classes` and :ref:`generator-provides` for these contracts.

Reference compositions
----------------------

The tests keep independent numeric reference compositions for parity checks. New
code reads intermediate pipeline values through ``gen.program.tap``; private scratch
views and kernels are not extension interfaces.
