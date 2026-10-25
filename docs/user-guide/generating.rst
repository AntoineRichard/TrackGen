.. _generating:

Generating tracks and gates
===========================

A generator builds one track (``TrackGenerator``) or one gate sequence
(``GateGenerator``) per :term:`env`, for a whole batch of :term:`E` envs at once, from
one :term:`seed` per env. It allocates everything once, at construction, and every call
rewrites the same result object in place: a ``Track`` or a ``GateSequence``
(:doc:`/user-guide/results`). This page covers how to build a generator, its parameters
and stages, how seeds and calls work, and how to save and inspect a pipeline. For a
guided first run, start with :doc:`/tutorials/batch-of-tracks` or
:doc:`/tutorials/gate-sequences`.

.. code-block:: python

   from track_gen import GateGenerator, TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   gen = TrackGenerator(presets.track(num_envs=E, device=device))  # runs one generation
   track = gen.generate()            # the result: a Track (the same object on every call)
   valid = track.valid.numpy()       # [E] int32: 1 where env e holds a usable track

   gates = GateGenerator(presets.gates(num_envs=E, device=device, gate_width=0.1))
   seq = gates.generate()            # a GateSequence

.. _generating-ways:

The ways to build a generator
-----------------------------

Every generator runs a :term:`spec`: a ``PipelineSpec``, the ordered list of the
:term:`stages <stage>` it runs and the batch they share. There are three ways to get
one, and they give the same generator:

- **A** :term:`preset` **(recommended).** ``presets.track(...)`` and ``presets.gates(...)``
  return a spec, with each replaceable stage chosen by a keyword (`Presets:
  presets.track`_, `Gate presets: presets.gates`_).
- **A config.** ``TrackGenConfig`` and ``GateGenConfig`` hold one flat field per knob
  (:ref:`config-classes`). ``TrackGenerator(cfg)`` builds the default spec of the config,
  ``legacy.to_pipeline_spec(cfg)`` (``legacy.to_gate_pipeline_spec`` for gates).
  ``presets.track(num_envs=E, device=d)`` builds what
  ``legacy.to_pipeline_spec(TrackGenConfig(num_envs=E, device=d))`` builds: the same
  stage list, equal parameters once resolved, byte-identical outputs.
- **An edited spec.** A spec is plain data: ``replace``, ``insert_after``,
  ``insert_before``, ``remove`` and ``with_randomize`` return a new spec (`What a spec
  holds`_).

.. code-block:: python

   import numpy as np
   from track_gen import TrackGenConfig, TrackGenerator
   from track_gen.pipeline import StageRef, presets

   E, device = 64, "cuda:0"  # or "cpu"

   spec = presets.track(num_envs=E, device=device, half_width=0.05)        # a preset
   cfg = TrackGenConfig(num_envs=E, device=device, half_width=0.05)         # a config
   no_relax = spec.replace("relax", StageRef("relax", "none"))              # an edited spec

   a = TrackGenerator(spec)
   b = TrackGenerator(cfg)
   c = TrackGenerator(no_relax)
   same = np.array_equal(a.generate().center.numpy(), b.generate().center.numpy(),
                         equal_nan=True)                                     # True

.. snippet: after

   assert same
   assert c.config.ids == spec.ids and c.config.stages[3].name == "none"

Constructing a generator
~~~~~~~~~~~~~~~~~~~~~~~~

``TrackGenerator(config_or_spec, rng=None, *, seeds=None, stream=None, out=None)`` takes
a ``TrackGenConfig`` or a track pipeline's ``PipelineSpec``:

- Seeds come from ``rng`` (a ``PerEnvSeededRNG``) or ``seeds`` (an ``int`` ``k``: env
  ``e`` gets seed ``k + e``; an ``[E]`` int32 ``wp.array``, bound by pointer; a
  ``PerEnvSeededRNG`` or a ``SeedState``; see `Seeds, epochs and the generation
  counter`_), not both (``ValueError``). With neither, env ``e`` gets seed ``e``.
- ``stream`` is the default stream of the calls that name none (`Calling a generator`_).
- ``out`` is a ``Track`` you own for the generator to write (:ref:`results-out`).

Construction checks the spec and every stage's parameters (`Parameters`_), allocates
every buffer the generator will ever use, and runs one generation, so the result already
holds a batch when the constructor returns. Construct a generator before you begin a
:term:`capture`: construction inside one raises ``CaptureError``
(:ref:`capture-rules`). A gate pipeline's spec raises ``ValueError`` pointing at
``GateGenerator``, before anything is bound; anything other than a config or a spec
raises ``TypeError``.

- ``gen.config`` returns what was passed (a copy of a config; a spec is immutable and
  returned as is), so ``TrackGenerator(gen.config, rng)`` rebuilds an identical generator.
- ``gen.params`` maps ``"generator"`` and every stage with a :term:`dynamic field` to its
  live parameters (`Parameters`_).
- ``gen.program`` is the pipeline the spec built: its result (``gen.program.result``),
  its :term:`fail bits`, its taps (`Inspecting a generator`_).
- ``gen.inputs`` and ``gen.input_info`` are the generator's device inputs (``regen_mask``,
  and those a stage declares, such as ``external``'s centerlines).
- ``gen.num_envs``, ``gen.device`` and ``gen.n_max`` read the batch, whichever form was
  passed.

A :term:`course` (``Course``) takes a spec, a config or a generator (:doc:`/utilities/course`).

.. _pipeline-presets:

Presets: ``presets.track``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``presets.track(...)`` returns the ``PipelineSpec`` of a track pipeline, with each
replaceable stage chosen by a keyword. It reads the registry (``registry``, default
``track_gen.plugin.registry``) for two things only: whether the generator provides its
own altitudes or half-widths (:ref:`Generator-provided altitude and width
<pipeline-provided>`), and whether the width stage has a ``half_width`` field. A name the
registry does not know is treated as a plain stage, and fails when the generator is
built.

.. snippet: skip=signature

.. code-block:: python

   presets.track(*, num_envs, device, n_max=384, scale=1.0, size=1.0,
                 generator="bezier", generator_params=None, num_points=256,
                 half_width=0.1, spacing=None,
                 relax="auto", width="constant", elevation="flat",
                 checks=("default",), on_invalid="keep",
                 z_base=0.0, registry=None, randomize=None) -> PipelineSpec

The stages, by id: ``generator``, ``resample`` (``resample.constant_spacing``),
``width``, ``relax``, ``uniform`` (``resample.uniform``), ``frames``, ``offset``,
``elevation``, the ``checks`` (the first one's id is ``"validity"``), and ``lift``
(`Choosing stages`_). A generator that provides its half-widths has no ``width`` stage,
one that provides its altitudes no ``elevation`` stage. Each ``randomize=`` entry adds a
``randomize.<stage>.<field>`` stage right before its target (:ref:`pipeline-randomize`).

``relax``, ``width``, ``elevation`` and each entry of ``checks`` take one of:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Form
     - Meaning
   * - ``"name"``
     - The stage registered as ``kind:name`` (``"xpbd"``, ``"acme.taubin"``), with its
       default parameters. ``"default"`` in ``checks`` is ``validity.default``.
   * - ``("name", params)``
     - The same with ``params``: a :term:`Params` instance, a mapping of its fields, or
       ``None``.
   * - ``StageRef(kind, name, ...)``
     - Used as given (id and ports included). Its ``kind`` must match the keyword
       (``ValueError`` otherwise); without an id it takes the slot's.
   * - ``"auto"`` (``relax`` only, the default)
     - The generator's preferred relax stage when it declares one, else ``"xpbd"``:
       ``relax.xpbd`` for every built-in shape generator, ``relax.none`` for
       ``external`` (:ref:`config-relax-auto`).
   * - ``None``
     - ``relax=None`` is ``relax.none``, ``elevation=None`` is ``elevation.flat``.
       ``width=None`` raises ``ValueError``: something must write the per-point
       half-width (a generator that provides it drops the slot, see
       :ref:`Generator-provided altitude and width <pipeline-provided>`).

Anything else raises ``TypeError``, and an empty name ``ValueError``. ``checks`` is a
sequence of selections whose first is ``"default"`` (``validity.default`` runs the
built-in checks; plugin checks follow it): a single selection raises ``TypeError``, an
empty sequence or another first check ``ValueError``. The other arguments:

- ``generator`` is a generator name, with ``generator_params`` (`Parameters`_), or a
  ``StageRef(kind="generator", ...)``, which carries its own parameters (passing
  ``generator_params`` too raises ``ValueError``). ``num_points`` is the generator's raw
  point count (unless the ``StageRef`` sets a ``capacity``).
- ``n_max`` is :term:`N_max`, the per-env point capacity of the result. Size it for the
  longest track: ``n_max >= longest perimeter / spacing + 1``; a track that needs more
  points is truncated and invalid (:ref:`invalid-envs`, :ref:`tuning-size`).
- ``size`` is the initial per-env course :term:`size` (the generator's dynamic ``size``,
  ``generator.size``; :ref:`tuning-size`): ``1.0`` adds nothing to the spec, another value
  goes into the generator's parameters. A generator that declares no ``size``
  (``external``) raises ``ValueError`` with a hint, and so does a ``size`` given both here
  and in ``generator_params`` (give it once). ``scale`` is the static coordinate scale of
  the whole batch.
- ``half_width`` is the road half-width: it is filled into the ``width`` selection when
  that stage's Params class has a ``half_width`` field (``width.constant``, or a plugin's
  such as ``acme.ripple``) and its parameters, in any selection form, are ``None`` or a
  mapping without ``half_width``. A Params instance is taken as is. A selection whose
  parameters set another ``half_width`` than the preset's raises ``ValueError`` naming
  ``half_width=`` (``width=("constant", {"half_width": 0.2})`` with the default
  ``half_width=0.1``). The relax, offset and validity stages read the per-point
  half-width the width stage writes (:ref:`pipeline-width`).
- ``spacing`` (:term:`spacing`) defaults to ``0.6 * half_width``, as in
  ``TrackGenConfig``. For a width stage without a ``half_width`` field
  (``width.from_channel``) or a generator that provides its half-widths, ``half_width``
  is only this nominal width for the default spacing; a ``half_width`` other than the
  default 0.1 that nothing reads (such a width and an explicit ``spacing``) raises
  ``ValueError``.
- ``lift`` is ``lift.flat`` (at ``z_base``, default ``0``, as ``TrackGenConfig.z_base``)
  when the elevation is ``elevation.flat``, else ``lift.elevated`` (a nonzero ``z_base``
  then raises ``ValueError``: the elevation's own parameters set its altitude). A
  generator that provides its altitudes gets ``lift.elevated`` and no elevation stage.
- ``registry`` is the registry the preset reads (default ``track_gen.plugin.registry``).
- Each further check gets the id ``ref.id or name``. Two checks with the same id raise
  ``ValueError``: give one an explicit ``StageRef(..., id=...)``.
- ``on_invalid`` is ``"keep"`` or ``"poison"`` (``ValueError`` otherwise): what an
  :term:`invalid env`'s rows hold (:ref:`invalid-envs`).
- ``randomize`` is ``None``, a mapping ``{"<stage id>.<field>": (lo, hi)}`` or a
  sequence of ``StageRef("randomize", "uniform", ...)``: per-env draws of dynamic
  ``float32`` fields (:ref:`pipeline-randomize`, :ref:`config-randomize`). A key naming a
  stage the preset does not have raises ``ValueError``.
- ``generator="external"`` ingests centerlines you load (`Your own centerlines`_): with
  ``generator_params={"with_z": True}`` it provides the altitude, with
  ``{"with_half_width": True}`` the half-width (the preset drops the elevation or width
  stage), and ``relax="auto"`` runs no relax.

A preset is a starting point: use the edit operations for the rest, for example
``presets.track(...).insert_after("validity", StageRef("validity",
"acme.max_curvature", id="kmax"))`` or ``.replace("resample", StageRef("resample",
"constant_spacing", {"spacing": 0.05}))``.

.. _pipeline-gate-presets:

Gate presets: ``presets.gates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. snippet: skip=signature

.. code-block:: python

   presets.gates(*, num_envs, device, generator="bezier", generator_params=None,
                 source="auto", gates=None, max_gates=None, min_gates=4,
                 order=None, stations=None,
                 gate_radius=0.025, gate_solve_iters=8, gate_width=0.0,
                 gate_align="yaw_only", elevation="flat", z_base=0.0,
                 z_valid_grade=0.0, checks=("gate_default",), scale=1.0, size=1.0,
                 on_invalid="keep", n_max=384, num_points=256, half_width=0.1,
                 spacing=None, relax="auto", width="constant",
                 spacing_metric="xy", elevation_on=None,
                 registry=None, randomize=None) -> PipelineSpec

``presets.gates(num_envs=E, device=d)`` builds what
``legacy.to_gate_pipeline_spec(GateGenConfig(num_envs=E, device=d))`` builds: the same
stage list, equal parameters once resolved, a byte-identical ``GateSequence``. It reads
the registry (``registry``, default ``track_gen.plugin.registry``) to see the
generator's output, what it provides and the orderings it accepts, which decide the
stage list.

- ``generator`` is a name (with ``generator_params``) or a ``StageRef`` of kind
  ``gate_generator`` or ``generator``; ``source`` is ``"auto"``, ``"anchors"``,
  ``"stations"`` or ``"relaxed_stations"``: gates from a gate generator's anchors, or at
  equal arc length along any track generator's centerline (:ref:`Stations: gates from
  any track generator <pipeline-stations>`).
- ``gates`` is the station count of a centerline source (default 12); with an anchors
  generator it raises ``ValueError`` (the generator decides its own count). It fills the
  ``gates`` field of a ``stations`` selection that does not set it; a selection that sets
  another count raises ``ValueError``. ``max_gates`` defaults to 32 for anchors, to the
  station count for stations.
- ``order``, ``stations``, ``elevation``, ``relax`` and each entry of ``checks`` take the
  forms of `Presets: presets.track`_ (a name, ``(name, params)``, a ``StageRef``, or
  ``None``: ``elevation=None`` is ``elevation.flat``, ``relax=None`` is ``relax.none``;
  ``relax`` defaults to ``"auto"``, the centerline generator's preferred relax, else
  ``"xpbd"``). ``order=None`` is the generator's default order: its preferred order when
  it accepts it, else ``"ccw"`` when it accepts it, else the first order it accepts
  (the rule of ``GateGenConfig.gate_ordering=None``). ``checks`` starts with
  ``"gate_default"``. ``stations=`` with an anchors generator raises ``ValueError`` (it
  writes its own anchors).
- ``gate_radius``, ``gate_solve_iters``, ``gate_width``, ``gate_align``, ``z_base``,
  ``z_valid_grade``, ``min_gates`` and ``scale`` are the ``GateGenConfig`` fields, passed
  to every stage that takes them (one value per pipeline, see :ref:`Planner rules for gate
  pipelines <pipeline-gate-plan-rules>`). ``gate_width`` is the initial value of
  ``gate_frames.default``'s per-env bar width (:ref:`gate-width-per-env`): keep it
  small, wide bars make many envs invalid (:ref:`gate-width-invalid-fraction`).
  ``validity.gate_default`` gets these values unless its parameters in ``checks`` set
  them; parameters setting another ``gate_radius`` or ``gate_width`` raise
  ``ValueError``. ``z_base`` is ``elevation.flat``'s altitude: a nonzero value with
  another elevation or with a generator that provides ``z`` raises ``ValueError``.
  ``gate_align`` is ``"yaw_only"`` or ``"full_tangent"`` (:doc:`/gates-3d`).
- ``size`` is the initial per-env course size (:ref:`gate-width-per-env`), routed to the
  stage that holds it: the generator's parameters (``generator.size``) for the
  ``bezier`` / ``hull`` gate generators and for the track generator of a
  ``"stations"`` / ``"relaxed_stations"`` source, ``normalize.extent``'s
  (``normalize.size``) for the ``polar``, ``voronoi`` and ``checkpoint`` gate
  generators. ``1.0`` adds nothing to the spec; a generator with neither raises
  ``ValueError``, as does a ``size`` in its ``generator_params`` too.
- ``n_max``, ``num_points``, ``half_width``, ``spacing``, ``relax`` and ``width``
  configure the centerline part of a station source: a ``"stations"`` source reads
  ``n_max`` and ``num_points``, a ``"relaxed_stations"`` source all six (``width`` as in
  ``presets.track``), and setting one that the source does not read raises
  ``ValueError``.
- ``spacing_metric`` (``"xy"`` or ``"xyz"``) is the stations' arc-length metric, and
  ``elevation_on`` (``None``, ``"gates"`` or ``"centerline"``) where the elevation runs;
  both apply to a centerline source, and ``"xyz"`` with an anchors generator raises
  ``ValueError`` (:ref:`gates-3d-stations`).
- ``on_invalid`` is ``"keep"`` or ``"poison"``.
- ``randomize`` is the per-env draws of `Presets: presets.track`_
  (:ref:`pipeline-randomize`); a randomizer of a gate generator's field
  (``{"generator.m": (3.0, 9.0)}``) is placed before the generator.
- ``generator="external"`` with ``source="stations"`` or ``"relaxed_stations"`` places
  gates on ingested centerlines (`Your own centerlines`_); its ``with_z`` altitude and
  its attributes reach the gates through the stations.


.. _pipeline-gate-spec:

Running a gate spec: ``GateGenerator(spec)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GateGenerator(config_or_spec, rng=None, *, seeds=None, stream=None, out=None)`` takes
a ``GateGenConfig`` or a gate pipeline's ``PipelineSpec`` (and ``out=`` a
``GateSequence`` you own, :ref:`results-out`), with the seed rules of
``TrackGenerator`` (``rng`` or ``seeds``, not both; with neither, env ``e`` gets seed
``e``). A track pipeline's spec raises ``ValueError`` pointing at ``TrackGenerator``,
anything else ``TypeError``.

- ``gen.config`` returns what was passed, so ``GateGenerator(gen.config, rng)`` rebuilds
  an identical generator.
- ``gen.program`` is the pipeline; ``gen.num_envs``, ``gen.device`` and
  ``gen.max_gates`` read the batch whichever form was passed.
- ``gen.params`` maps ``"generator"`` and every stage with a dynamic field to its live
  parameters (``gate_frames`` for the bar width, ``normalize`` for the size of the
  ``polar``, ``voronoi`` and ``checkpoint`` gate generators).

Everything else (``generate``, ``enqueue``, ``seeds``, ``inputs``, the generator's own
CUDA graph, masked regeneration) works as for tracks.

What a spec holds
~~~~~~~~~~~~~~~~~

``StageRef(kind, name, params=None, id=None, ports={}, capacity={})``
   One stage slot, frozen. ``kind`` and ``name`` select the stage (``"relax"``,
   ``"xpbd"``). ``params`` is the stage's Params instance, a mapping of its fields, or
   ``None`` for its defaults; a mapping is coerced and validated when the generator is
   built. ``id`` names the slot (``slot_id`` is ``id or kind``): two stages of one kind
   need explicit ids (the default list has two ``resample`` stages, ``"resample"`` and
   ``"uniform"``), and the generator's slot is always ``"generator"``. ``ports`` rewires
   a stage's port to another :term:`channel` of the same dtype and size, and wires a
   stage's required read ports (``width.from_channel``'s ``src``, see
   :ref:`Required read ports <pipeline-required-ports>`). ``capacity`` is the generator's
   ``{"raw": num_points}``, its default raw point count; a Stage generator's own
   ``Stage.capacity()`` overrides it. A ``capacity`` on any other stage is refused.

``BatchSpec(num_envs, device, n_max, closed=True, scale=1.0, on_invalid="keep", max_gates=None)``
   The batch every stage shares: ``E`` envs on one device, the per-env point capacity
   ``N_max``, the coordinate ``scale``, and what happens to an invalid env's rows
   (``"keep"`` or ``"poison"``, :ref:`invalid-envs`). ``closed`` must be a ``bool``, and
   open tracks (``closed=False``) are refused, naming the stages that support closed
   loops only (:ref:`results-topology`). ``max_gates`` selects the result: ``None``
   (default) builds a track pipeline whose result is a ``Track``; an ``int >= 2`` builds
   a gate pipeline whose result is a ``GateSequence`` with ``max_gates`` slots per env.
   In a gate pipeline ``n_max`` is read only by a centerline source; the presets pass
   384.

``PipelineSpec(batch, stages, require_caller_capturable=False)``
   The batch and the ordered tuple of ``StageRef``. ``ids`` lists the slot ids.
   ``require_caller_capturable=True`` makes the build reject every stage that runs on the
   host, so the pipeline can always be recorded into your graph (:ref:`cuda-graph-host`).
   ``replace(id, ref)``, ``insert_after(id, ref)``, ``insert_before(id, ref)`` and
   ``remove(id)`` return a new spec; ``replace`` keeps the slot's id when ``ref.id`` is
   ``None``. They are pure data operations: an unknown anchor id raises ``ValueError`` at
   once, everything else is checked when the generator is built.
   ``with_randomize(randomize)`` inserts randomizers, as the presets do
   (:ref:`config-randomize`). ``to_dict()`` and ``from_dict(d)`` save and load it
   (`Saving and loading specs`_).

``legacy.to_pipeline_spec(config, *, registry=None)``
   The default spec for a ``TrackGenConfig``, the one ``TrackGenerator(config)`` builds
   (see :ref:`The default track stages <pipeline-stages-the-default-track-stages>`). It
   resolves the generator's parameters, and the ``relax_solver`` and ``z_profile`` names
   (see :ref:`flat-stage-shortcuts`), through ``registry``, and raises the
   ``relax_solver``, ``smooth_finish`` and ``z_profile`` errors of
   ``TrackGenerator(config)``.

``legacy.to_gate_pipeline_spec(config, *, registry=None)``
   The default gate spec for a ``GateGenConfig``, the one ``GateGenerator(config)``
   builds (see :ref:`The gate stages <pipeline-gate-stages>`). It resolves the gate
   generator and its parameters, the ``gate_ordering`` (``None``: the generator's default
   order) and the ``z_profile`` through ``registry``, and raises the errors of
   ``GateGenerator(config)`` (unknown names, an unsupported ordering, the gate-count
   checks of :ref:`gate-capacity-config`).

.. _generating-params:

Parameters
----------

Each stage has a typed :term:`Params` class, a frozen dataclass of its knobs: the
generator's (``BezierParams``, ``PolarParams``, ...; :ref:`builtin-generator-params`)
and every other stage's (``XpbdParams``, ``ElevationNoiseParams``, ...; `Choosing
stages`_). Give them as a mapping of field names, or as an instance:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets
   from track_gen.plugin import registry

   E, device = 64, "cuda:0"  # or "cpu"

   PolarParams = registry.get("generator", "polar").Params     # no public import path
   spec = presets.track(num_envs=E, device=device, generator="polar",
                        generator_params={"polar_num_knots": 9, "polar_radial_jitter": 0.5},
                        relax=("xpbd", {"relax_iters": 80}))
   gen = TrackGenerator(spec)

   gen.params["generator"].params        # the resolved PolarParams(polar_num_knots=9, ...)
   gen.params["generator"].set("polar_radial_jitter", 0.3)    # a dynamic field
   track = gen.generate()                # built with the new value

.. snippet: after

   assert isinstance(gen.params["generator"].params, PolarParams)
   assert gen.params["generator"].params.polar_num_knots == 9
   assert abs(float(gen.params["generator"].in_effect("polar_radial_jitter").numpy()[0])
              - 0.3) < 1e-6

- **Checked at construction.** The spec only holds the values; the generator checks
  them when it is constructed (``ValueError`` naming the field and its rule, such as
  ``polar_radial_jitter must be in [0, 0.85], got 2.0``; an unknown key lists the valid
  fields). The rules of each built-in are in :ref:`builtin-generator-params`.
- **Static or dynamic.** A static field (a count, a buffer size, a branch:
  ``polar_num_knots``, ``max_num_points``, ``half_width``, ``spacing``, the relax knobs)
  is fixed when the generator is built. A :term:`dynamic field` (the generators' shape
  knobs, their per-env ``size``, the elevation ranges, the gate bar width) is a per-env
  ``[E]`` array, initialised from the configured value, that you can rewrite between
  generations or draw per env with ``randomize=``: :doc:`/user-guide/curricula` lists
  them and shows how.
- **The configuration is copied at construction.** ``gen.config`` and
  ``gen.params[stage].params`` keep the construction values; mutating the config or the
  mapping you passed afterwards has no effect. To change a dynamic field, write
  ``gen.params[stage]`` (``set(field, value)``, a scalar or an ``[E]`` array, or your own
  kernel into ``gen.params[stage][field]``); to change a static field, build a new
  generator. ``set`` on a static field raises ``KeyError`` with that advice.

``registry.describe(kind, name).params_schema`` is the JSON schema of a stage's
parameters (fields, defaults, bounds, which ones are dynamic).

.. _generating-stages:

Choosing stages
---------------

A preset picks one stage for each slot from a keyword; ``replace`` or ``insert_after``
pick any registered stage. The built-in track stages, in pipeline order:

.. list-table::
   :header-rows: 1
   :widths: 13 22 31 18 16

   * - Slot id
     - Stage
     - What it does
     - Params
     - Preset argument
   * - ``generator``
     - ``generator.bezier``, ``hull``, ``polar``, ``voronoi``, ``checkpoint``,
       ``repulsive``, ``external``
     - draws each env's raw centerline from its seed (:doc:`/tutorials/choosing-a-generator`)
     - the generator's (:ref:`builtin-generator-params`)
     - ``generator=``, ``generator_params=``, ``num_points=``, ``size=``
   * - ``resample``
     - ``resample.constant_spacing``
     - resamples the loop at a constant arc spacing into ``N_max`` slots
     - :class:`~track_gen.stages.params.ConstantSpacingParams`
     - ``spacing=``, ``n_max=``
   * - ``width``
     - ``width.constant``, ``width.from_channel``
     - writes the per-point half-width
     - :class:`~track_gen.stages.params.WidthConstantParams`,
       :class:`~track_gen.stages.params.WidthFromChannelParams`
     - ``width=``, ``half_width=``
   * - ``relax``
     - ``relax.xpbd``, ``relax.none``
     - pushes the loop apart where the band would overlap itself
       (:doc:`/relaxation/overview`)
     - :class:`~track_gen.stages.params.XpbdParams`
     - ``relax=``
   * - ``uniform``
     - ``resample.uniform``
     - restores the uniform spacing after the relax
     - none
     -
   * - ``frames``
     - ``frames.default``
     - tangents, normals, curvature, arc length, winding
     - none
     -
   * - ``offset``
     - ``offset.default``
     - inflates the centerline into the ``outer`` and ``inner`` borders
     - :class:`~track_gen.stages.params.OffsetParams`
     -
   * - ``elevation``
     - ``elevation.flat``, ``uniform``, ``random_walk``, ``noise``
     - the altitude of each point (:doc:`/tracks-25d`)
     - :class:`~track_gen.stages.params.ElevationFlatParams`, ``...UniformParams``,
       ``...RandomWalkParams``, ``...NoiseParams``
     - ``elevation=``, ``z_base=``
   * - ``validity`` (and plugin checks)
     - ``validity.default``
     - the checks that set ``valid`` and ``fail_mask`` (:ref:`invalid-envs`)
     - :class:`~track_gen.stages.params.ValidityParams`
     - ``checks=``
   * - ``lift``
     - ``lift.flat``, ``lift.elevated``
     - writes the ``vec3f`` result; on an elevated track, the 3D tangent and arc length
     - :class:`~track_gen.stages.params.LiftFlatParams`, none
     - follows ``elevation=``
   * - ``randomize.<stage>.<field>``
     - ``randomize.uniform``
     - draws one dynamic field per env, before its target (:doc:`/user-guide/curricula`)
     - :class:`~track_gen.stages.randomize.RandomizeUniformParams`
     - ``randomize=``

The gate pipeline (``presets.gates``) runs a gate generator (``gate_generator.bezier``,
``hull``, ``polar``, ``voronoi``, ``checkpoint``) or a track generator with a
``stations`` stage (``stations.raw``, ``stations.relaxed``,
:class:`~track_gen.stages.params.StationsParams`; ``source=``, ``gates=``), then
``order`` (``ccw``, ``raw``, ``random_pairs``; ``order=``), ``normalize``
(``normalize.extent``, :class:`~track_gen.stages.params.NormalizeExtentParams`; the
``polar``, ``voronoi`` and ``checkpoint`` gate generators), ``gate_relax``
(``gate_relax.spheres``, :class:`~track_gen.stages.params.GateRelaxParams`;
``gate_radius=``, ``gate_solve_iters=``), ``elevation``, ``lift``, ``tangents``,
``gate_frames`` (``gate_frames.default``,
:class:`~track_gen.stages.params.GateFramesParams`; ``gate_width=``, ``gate_align=``)
and ``validity`` (``validity.gate_default``,
:class:`~track_gen.stages.params.GateValidityParams`; ``min_gates=``,
``z_valid_grade=``); :ref:`pipeline-gate-stages` has the full list.

Plugins can register stages of six kinds: ``relax``, ``width``, ``elevation`` and
``validity`` in the track pipeline, ``order`` and ``stations`` in the gate pipeline
(``elevation`` and ``validity`` serve both); the other kinds are TrackGen's own
(:ref:`pipeline-stage-kinds`). List what is installed with
``registry.available(kind)``; :doc:`/contributing/writing-a-stage` shows how to write
one.

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets
   from track_gen.plugin import registry

   E, device = 64, "cuda:0"  # or "cpu"

   registry.available("elevation")      # ['flat', 'noise', 'random_walk', 'uniform', ...]
   hilly = presets.track(num_envs=E, device=device, relax=None,
                         elevation=("noise", {"z_min": -0.3, "z_max": 0.3,
                                              "z_noise_amplitude": 0.3}))
   hilly.ids
   # ('generator', 'resample', 'width', 'relax', 'uniform', 'frames', 'offset',
   #  'elevation', 'validity', 'lift')
   gen = TrackGenerator(hilly)

.. snippet: after

   assert hilly.stages[hilly.ids.index("lift")].name == "elevated"
   assert hilly.stages[hilly.ids.index("relax")].name == "none"

.. _generating-seeds:

Seeds, epochs and the generation counter
----------------------------------------

Each env is keyed by a pair ``(seed, epoch)``, held by the generator's
:class:`~track_gen.seeds.SeedState` (``gen.seeds``):

- ``seeds``: ``[E]`` int32. A generator binds ``rng.seeds_warp`` by pointer, so
  ``rng.set_seeds_warp(...)`` reseeds it, and the rng must live on the generator's
  device (construction raises ``ValueError`` otherwise). ``SeedState`` and the
  generators never write a seed array they were given (a ``wp.array`` or a
  ``PerEnvSeededRNG``'s): such a state is ``bound``, and ``set()`` on it raises
  ``RuntimeError``. Write that array yourself instead. The rng is another matter:
  ``PerEnvSeededRNG.set_seeds_warp`` and ``Course.generate(seeds=...)`` overwrite
  ``rng.seeds_warp`` in place, and that IS your array when the rng was built from a
  ``wp.array``. A ``SeedState`` built from ``None`` or an ``int`` ``k`` owns its seeds
  (``k + arange(E)``, as ``PerEnvSeededRNG(seeds=k)``), as does one built with
  ``copy=True``.
- ``epoch``: ``[E]`` int32, owned, zero at construction and changed only on device:
  ``advance(mask=None)`` adds 1 where ``mask[e] != 0`` (to every env without a mask),
  ``reset_epoch(mask=None)`` zeroes it the same way, and ``set(seeds, ids=None)`` scatters
  new seeds into owned seeds and zeroes those envs' epochs. Each is one launch (``set``
  with ``ids=None``: a copy and a launch) on the call's stream, allocation-free, so it can
  be recorded into a graph. Without ``stream=``, the generator's ``seeds`` methods run on
  its constructor ``stream=``, like its ``generate()``.

Every generation opens with one kernel that folds the pair into the effective seed the
stages read (right after the epoch advance when ``advance=True``):

- epoch ``0``: the seed itself;
- epoch ``k != 0``: ``hash32(uint32(seed) ^ hash32(uint32(k) * 0x9E3779B9)) & 0x7FFFFFFF``,
  a non-negative int32, where ``hash32`` is the MurmurHash3 finaliser
  (``track_gen.ops.hash32``).

``(seed, epoch)`` fully determines an env's output: a generator whose env ``e`` has the
same seed and the same epoch rebuilds the same track or gate sequence. So does a
generator at epoch ``0`` seeded with ``track_gen.seeds.effective_seed(seed, epoch)``, the
bit-exact host twin of that kernel. The kernel is recorded with the rest of the
generation, so device-side writes to the seeds or to the epochs reach every replay, of
your graph or of the generator's own.

.. code-block:: python

   import numpy as np
   import warp as wp
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets
   from track_gen.seeds import effective_seed

   E, device = 64, "cuda:0"  # or "cpu"

   gen = TrackGenerator(presets.track(num_envs=E, device=device), seeds=0)  # env e: seed e
   gen.generate(advance=True)                  # a new batch: every env at epoch 1
   gen.generate(advance=True)                  # epoch 2
   gen.generate().generation.numpy()[:4]       # [2 2 2 2]

   # the same batch from a fresh generator, seeded with the effective seeds
   twin_seeds = wp.array([effective_seed(e, 2) for e in range(E)], dtype=wp.int32,
                         device=device)
   twin = TrackGenerator(presets.track(num_envs=E, device=device), seeds=twin_seeds)

.. snippet: after

   assert np.array_equal(gen.generate().center.numpy(), twin.generate().center.numpy(),
                         equal_nan=True)

Each generation ends by writing ``generation[e] = epoch[e]`` into the result (for the
masked envs only, on a masked generation). ``Track.generation`` and
``GateSequence.generation`` (``[E]`` int32) therefore tell a consumer, on device, which
:term:`epoch` each env's current output was built with: its :term:`generation`. It is
the epoch, not a change counter: a reseed at the same epoch does not change it, and after
``reset_epoch()`` earlier values come back. It changes exactly when an env gets new
output only while nothing reseeds or resets the epochs. The field defaults to ``None``
so results built by hand still construct, and ``clone()`` copies it.

Reseeding through the rng leaves the epochs alone: after an advance,
``rng.set_seeds_warp(new_seeds, None)`` gives the batch of ``(new_seeds, epoch)``, not
that of a fresh generator. Call ``gen.seeds.reset_epoch()`` as well to reproduce a fresh
generator; ``Course.generate(seeds=...)`` does both.

.. _generating-calls:

Calling a generator
-------------------

Construction already ran one generation. The calls:

- ``generate(num_or_ids=None, *, stream=None, sync=True, advance=False, mask=None)``:
  with ``advance=True`` first advances the epoch of every env (of the envs with
  ``mask[e] != 0`` when ``mask`` is given; ``mask`` requires ``advance=True``), then
  runs the generation (recorded into your graph, a replay of the generator's own CUDA
  graph, or eager; see `Execution modes`_), writes the result's ``generation`` and
  returns the result. Without ``advance`` it rebuilds the current batch from the current
  seeds, epochs and parameters. ``num_or_ids`` must be ``None`` (the batch is fixed).
  See :ref:`Masked regeneration <cuda-graph-masked-regeneration>` and
  :ref:`rl-masked-regeneration`.
- ``enqueue(*, stream=None, advance=False, mask=None)``: the same generation, recorded
  into the active capture or run eagerly on the resolved stream. It never builds the
  generator's own graph and never waits on the host (a stage that runs on the host
  blocks for its own host work only, :ref:`cuda-graph-host`). An iterative stage
  (``repulsive``) runs its full iteration bound when called eagerly on ``cuda``; the
  generator's own graph skips the converged windows on device
  (:ref:`cuda-graph-bounded`).
- ``graph``: the generator's own ``wp.Graph`` for a plain ``generate()`` (no
  ``advance``, no ``mask``). It is ``None`` until the first such ``cuda`` call outside a
  capture, and stays ``None`` on ``cpu`` and for a generator with a stage that runs on
  the host, whose recordable parts get graphs of their own (:ref:`cuda-graph-host`).
  Every other ``(advance, mask)`` combination gets a graph of its own.
- ``seeds``: the :class:`~track_gen.seeds.SeedState` (`Seeds, epochs and the generation
  counter`_).
- ``inputs``: ``{"regen_mask": ...}``, a persistent, zero-initialised ``[E]`` int32
  array with a stable address, for your own kernels to write a mask into, plus every
  per-env input a stage declares.
- ``params``: the live parameters (`Parameters`_), written in place between or inside
  replays with no new capture.
- ``Track.clone()`` / ``GateSequence.clone()``: a snapshot of the current output
  (:ref:`results-views`).

.. code-block:: python

   import warp as wp
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   gen = TrackGenerator(presets.track(num_envs=E, device=device))
   track = gen.generate()                   # waits for its own stream only
   track = gen.generate(sync=False)         # returns at once; results are stream-ordered
   gen.enqueue(advance=True)                # a new batch, never waits on the host

   done = wp.array([1, 0] * (E // 2), dtype=wp.int32, device=device)
   gen.generate(advance=True, mask=done)    # new tracks for the envs with done[e] != 0

.. snippet: after

   g = gen.generate().generation.numpy()
   assert g[0] == 2 and g[1] == 1

Execution modes
~~~~~~~~~~~~~~~

Each call resolves ONE stream: an explicit ``stream=`` (a ``wp.Stream`` or ``"torch"``),
then an enclosing ``LaunchContext.scope()``, then the constructor's ``stream=``, then
torch's capture stream while ``torch.cuda.graph`` is capturing on the device, else Warp's
current stream. The state of that stream picks the mode:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Mode
     - When
     - What ``generate()`` does
   * - Record
     - The resolved stream is capturing.
     - Enqueues the whole generation on that stream, so it becomes nodes of your graph.
       There is no graph of its own, no ``wp.capture_launch`` (a graph launch cannot be
       recorded into another capture) and no host wait, whatever ``sync`` says. A
       generator with a stage that runs on the host raises ``NotCapturableError``
       instead, before any launch.
   * - Own graph
     - ``cuda``, not capturing, every stage recordable.
     - The first call records the generation once into the generator's own graph on a
       private stream (one graph per ``(advance, mask)`` combination). Every call then
       launches that graph on the resolved stream with ``wp.capture_launch``.
   * - Own graph, in parts
     - ``cuda``, not capturing, a stage that runs on the host.
     - The first call records each recordable part of the generation into a graph of its
       own. Every call then runs the parts in order on the resolved stream: the host
       stage eagerly, the others with ``wp.capture_launch`` (:ref:`cuda-graph-host`).
   * - Eager
     - ``cpu``, or ``enqueue()`` outside a capture.
     - Launches the stages directly on the resolved stream.

The seed kernel is part of what is recorded, in the generator's own graph and in yours
alike. Every replay therefore reads the rng's CURRENT seeds and the CURRENT epochs:
``rng.set_seeds_warp(new_seeds, None)`` writes into the rng's existing buffer, and the
next replay generates the new batch with no new capture. A recorded
``generate(advance=True)`` advances the epochs on device, so each replay generates a new
batch without any reseed.

Outside a capture, ``generate()`` waits for its stream to finish before returning
(``sync=True``, the default). The wait covers that one stream. It is skipped while torch
is capturing in the calling thread (a host wait there could invalidate torch's capture),
and it runs with the thread's CUDA capture mode relaxed, so it never invalidates a
capture another thread has open. ``generate(sync=False)`` returns at once, and the
results are stream-ordered like every other library call (see
:doc:`/utilities/overview`).

A call is recorded only when the stream it RESOLVES is capturing. When you capture a
stream that is not Warp's current one, pass it (``generate(stream=s)``) or wrap the calls
in ``LaunchContext(device, s).scope()``. Otherwise the call runs, or replays its own
graph, on another stream and is silently missing from your graph. A generator
constructed with ``stream=`` resolves to that stream before torch's capture stream, so
inside ``torch.cuda.graph`` pass ``stream="torch"`` to it. The rules for recording a
generator into your graph, and what to do when a capture fails, are in
:ref:`capture-rules` and :ref:`capture-troubleshooting`.

``Course.regenerate()`` follows the same modes with one body: generation plus the
downstream refresh (see :doc:`/utilities/course`). It is not split in parts: over a
generator with a stage that runs on the host it runs eagerly outside a capture, and
raises ``NotCapturableError`` inside one.

.. _generating-saving:

Saving and loading specs
------------------------

A spec is plain data. ``spec.to_dict()`` returns a JSON-serialisable dict, and
``PipelineSpec.from_dict(d)`` reads it back without a registry; building the round-tripped
spec gives byte-identical outputs, plugin stages and their parameters included (a plugin
stage needs its plugin installed to build).

.. code-block:: python

   import json
   from track_gen import TrackGenerator
   from track_gen.pipeline import PipelineSpec, presets

   E, device = 64, "cuda:0"  # or "cpu"

   spec = presets.track(num_envs=E, device=device, generator="polar",
                        randomize={"generator.size": (0.8, 1.5)})
   with open("spec.json", "w") as f:
       json.dump(spec.to_dict(), f, indent=2)

   with open("spec.json") as f:
       loaded = PipelineSpec.from_dict(json.load(f))
   gen = TrackGenerator(loaded)

.. snippet: after

   assert loaded == spec

The device is saved with the batch: to load a spec on another device, edit
``d["batch"]["device"]`` before ``from_dict``. A ``TrackGenConfig`` is saved through its
spec: ``legacy.to_pipeline_spec(cfg).to_dict()``.

.. _pipeline-spec-format:

The saved format
~~~~~~~~~~~~~~~~

``to_dict()`` writes ``{"format": "track_gen.pipeline/2", "batch": {...}, "stages":
[...]}``: the batch's fields (``on_invalid`` and ``max_gates`` are optional when reading,
``"keep"`` and ``None`` when absent), then one entry per stage with its ``kind``,
``name``, ``params`` and the optional ``id``, ``ports`` and ``capacity``. A missing key
raises a ``ValueError`` naming it (``stages[3] is missing 'kind'``), and so does any
format string ``from_dict`` does not know. ``require_caller_capturable`` is written only
when it is set.

``from_dict`` also reads the older ``"track_gen.pipeline/1"`` dicts and migrates them,
with one ``DeprecationWarning``; the migration guide says :ref:`how, and which ones
cannot be migrated <pipeline-dev3-specs>`.

Inspecting a generator
----------------------

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets
   from track_gen.plugin import registry

   E, device = 64, "cuda:0"  # or "cpu"

   gen = TrackGenerator(presets.track(num_envs=E, device=device))
   gen.config.ids                         # the slot ids, in pipeline order
   gen.program.fail_bits                  # {'generator': 0, 'turning': 1, ..., 'truncation': 7}
   xy = gen.program.tap("cl.xy", after="resample")   # an intermediate channel, read-only
   info = registry.describe("generator", "bezier")   # origin, capture, Params schema, ...
   info.seeded                            # True: its output depends on the seed

.. snippet: after

   assert gen.program.fail_bits["truncation"] == 7
   assert xy.shape[0] == E * gen.n_max

- ``gen.program.tap(channel, after=stage_id)`` reads a :term:`channel` (an intermediate
  array, such as the raw loop ``raw.xy`` or the resampled ``cl.xy``) as it stood after a
  stage, on the last generation (:ref:`pipeline-taps`); ``gen.program.channels`` lists
  them. Measure the longest raw loop of your configuration with
  ``gen.program.tap("raw.xy")`` (:ref:`tuning-size`).
- ``gen.program.fail_bits`` names the bits of ``fail_mask`` (:ref:`invalid-envs`).
- ``registry.describe(kind, name)`` describes a registered stage: where it comes from,
  whether it can be recorded into a graph, its parameter schema, what it provides, and
  whether its output depends on the seed (``seeded``).

Your own centerlines
--------------------

The ``external`` generator runs centerlines you supply (a dataset, a learned model, a
hand-drawn circuit) through the same stages: resample, width, checks, band, elevation and
lift, or the gate stations. Load them with ``load_centerlines``, in your data's units
(set ``spacing``, ``half_width`` and ``n_max`` for them), before a generation:

.. code-block:: python

   import numpy as np
   from track_gen import TrackGenerator
   from track_gen.pipeline import load_centerlines, presets

   E, device = 64, "cuda:0"  # or "cpu"

   t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
   loops = [np.stack([(1.0 + 0.01 * e) * np.cos(t), 0.6 * np.sin(t)], axis=1)
            for e in range(E)]                       # [m_e, 2] each, not closed by hand

   gen = TrackGenerator(presets.track(num_envs=E, device=device, generator="external",
                                      half_width=0.1, n_max=384))
   load_centerlines(gen, loops)
   track = gen.generate()

.. snippet: after

   assert (track.valid.numpy() == 1).all()

:ref:`ingesting-centerlines` has the inputs, the sizing check, altitudes, per-point
widths, attributes and how to load new rows for a masked regeneration.
