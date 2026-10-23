Configuration reference
=======================

A ``TrackGenerator`` takes a ``TrackGenConfig`` and a ``GateGenerator`` a
``GateGenConfig``. The core fields (batch, device, relaxation, inflation, spacing, Z
profile, gate counts) apply whatever the generator. A generator's own knobs are its
*Params*: set them through ``generator_params``, or, for the built-in generators, through
the flat fields that carry the same names. This page lists:

- the two config classes and every field (:ref:`config-classes`);
- how ``relax_solver`` and ``z_profile`` select registered pipeline stages
  (:ref:`flat-stage-shortcuts`), and ``gate_ordering`` and ``z_profile`` the gate
  pipeline's (:ref:`gate-stage-shortcuts`);
- how a gate generator's capacity meets ``max_gates`` and ``min_gates``
  (:ref:`gate-capacity-config`);
- the road's half-width, its floor and the gate stations' spacing metric
  (:ref:`width-config`);
- the presets' relax default (:ref:`config-relax-auto`) and per-env draws of dynamic
  parameters (:ref:`config-randomize`);
- how ``generator_params`` resolves (:ref:`generator-params-config`);
- each built-in generator's Params, with defaults and rules
  (:ref:`builtin-generator-params`);
- when the generator-specific rules are checked (:ref:`generator-params-validation`);
- the deprecated fields (:ref:`deprecated-config-fields`).

.. _config-classes:

Config classes
--------------

.. autoclass:: track_gen.TrackGenConfig
   :no-members:

.. autoclass:: track_gen.GateGenConfig
   :no-members:

.. _gate-width-config:

Gate width and radius
~~~~~~~~~~~~~~~~~~~~~

Two ``GateGenConfig`` fields size a gate, and they are easy to confuse:

- ``gate_width`` is the **full** width of the gate bar, from ``left`` to ``right``:
  ``GateSequence.half_size`` is ``0.5 * gate_width``, and ``left`` / ``right`` sit at
  ``position ± half_size * left_axis``. It is not a half-width (earlier documentation
  said so). The default ``0.0`` makes point gates (``left == right == position``), and
  the bar-crossing check (fail bit ``crossing``) runs only when it is positive. It does
  not move the gates.
- ``gate_radius`` is the radius of the disk each gate center occupies in the plan-view
  separation: the solve pushes centers to at least ``2 * gate_radius`` apart, and the
  ``spacing`` check rejects an env whose gates stay closer. It does not size the bar.

In a pipeline spec each value lives in the stages that use it (``gate_frames`` and
``validity.gate_default`` for ``gate_width``; ``gate_relax.spheres`` and
``validity.gate_default`` for ``gate_radius``), and the planner requires them to agree.
A generator that provides its own ``half_size`` refuses a non-zero ``gate_width``
(``ValueError``); its bar widths come from the generator (:ref:`pipeline-provides`).

.. _flat-stage-shortcuts:

Pipeline stages: ``relax_solver`` and ``z_profile``
---------------------------------------------------

``TrackGenerator(config)`` builds the default pipeline of
:doc:`/how-it-works/pipeline-preview`, and two flat fields pick its replaceable stages by
name through the plugin registry (``track_gen.plugin.registry``), the way
``generator`` picks the generator:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Field
     - Selects
     - Params of the stage
   * - ``relax_solver``
     - The ``relax`` stage registered under that name: ``"xpbd"`` (default) or a
       plugin's, such as ``"acme.taubin"``. ``relax_enable=False`` selects
       ``relax.none`` instead (``relax_solver`` is still resolved first).
     - ``relax.xpbd``: the ``relax_*`` fields, as before (it reads the half-width from
       the width stage, see :ref:`width-config`). A plugin: its
       ``Params.from_legacy(config)``.
   * - ``z_profile``
     - The ``elevation`` stage registered under that name: ``"flat"`` (default),
       ``"uniform"``, ``"random_walk"``, ``"noise"`` or a plugin's, such as
       ``"acme.sinusoid_hills"``. Any name other than ``"flat"`` lifts with
       ``lift.elevated`` (3D tangents and arc length).
     - The built-ins: the ``z_*`` fields their profile uses, as before. A plugin: its
       ``Params.from_legacy(config)``, which may map the flat ``z_*`` fields.

``Params.from_legacy`` returns the class defaults unless the plugin overrides it, so a
plugin stage selected by a flat field runs with its defaults; to pass other values, build
the pipeline with :ref:`presets.track <pipeline-presets>` (``relax=("acme.taubin",
{"passes": 20})``) and hand the spec to ``TrackGenerator``. The width and validity
stages have no flat field: use the preset (``width=...``, ``checks=(...)``).

The names are resolved when the generator is constructed, in the order
``relax_solver``, ``smooth_finish``, ``z_profile``, after the generator and its
``generator_params``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Cause
     - Error at ``TrackGenerator`` construction
   * - ``relax_solver="energy"`` or ``"tp_sobolev"``
     - ``ValueError``: an oracle-only backend (``tests/_oracle``), not a relax stage; the
       message lists the registered relax stages.
   * - Another unknown ``relax_solver``
     - ``ValueError``: ``TrackGenConfig.relax_solver: unknown relax 'bogus'; available:
       ['none', 'xpbd']`` (plus the installed plugins' names).
   * - An unknown ``z_profile``
     - ``ValueError``: ``TrackGenConfig.z_profile: unknown elevation 'bogus';
       available: ['flat', 'noise', 'random_walk', 'uniform']`` (plus the installed
       plugins' names).
   * - A name served by a plugin that failed to load
     - That plugin's ``PluginLoadError``.

``TrackGenConfig`` construction no longer checks ``z_profile`` against a fixed set: it
only requires a non-empty string, since the registry, not the config, knows the
installed stages. ``TrackGenConfig(z_profile="bogus")`` therefore constructs, and
``TrackGenerator`` raises the error above; the ``z_*`` range rules
(``z_min <= z_max``, ...) still run at config construction. ``Course`` takes a
``TrackGenConfig``, so both fields work there too.

.. _gate-stage-shortcuts:

Gate pipeline stages: ``gate_ordering`` and ``z_profile``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GateGenerator(config)`` builds the default gate pipeline
(:ref:`pipeline-gate-stages`), and two ``GateGenConfig`` fields pick its replaceable
stages through the registry, the way ``generator`` picks the gate generator:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Field
     - Selects
     - Params of the stage
   * - ``gate_ordering``
     - The ``order`` stage registered under that name: ``"ccw"``, ``"raw"``,
       ``"random_pairs"`` or a plugin's, such as ``"acme.reverse"``
       (:ref:`writing-an-ordering`). It must be one the generator accepts (see below).
       ``None`` (default): the generator's default order (see below).
     - None for the built-ins. A plugin order runs with its default Params; pass others
       with ``presets.gates(order=(name, params))``.
   * - ``z_profile``
     - The ``elevation`` stage registered under that name, as for tracks: ``"flat"``
       (default), ``"uniform"``, ``"random_walk"``, ``"noise"`` or a plugin's, such as
       ``"acme.sinusoid_hills"``. It runs on the gates
       (:ref:`pipeline-gate-elevation`).
     - The built-ins: the ``z_*`` fields their profile uses, as before
       (``uniform`` / ``random_walk`` draw one altitude per gate: the gates are the
       knots, so ``z_control_points`` is not read on gates). A plugin: its
       ``Params.from_legacy(config)``, which receives the ``GateGenConfig`` (only the
       ``z_*`` fields are shared with ``TrackGenConfig``).

Which orders a generator accepts is its own declaration: every registered order except
``"raw"`` for ``bezier`` and ``hull`` (an unordered cloud of points), every registered
order except ``"random_pairs"`` for ``polar``, ``checkpoint`` and ``voronoi`` (a loop in
generation order), so a plugin's order works with all five. A plugin generator may
whitelist its orders instead; one that provides its own altitudes or tangents accepts
``"raw"`` only. ``gate_ordering=None`` (the default) selects the generator's default
order: its preferred order when it accepts it, else ``"ccw"`` when it accepts it, else
the first order it accepts, the rule of ``presets.gates(order=None)`` and of the
conformance kit. That is ``"ccw"`` for the five built-ins (the default of earlier
releases) and ``"raw"`` for such a provider, so ``GateGenConfig(generator="acme.helix")``
needs no ``gate_ordering``. The name is resolved when ``GateGenerator`` (hence
``Course``) is constructed, by ``legacy.to_gate_pipeline_spec``; the generator's run
config holds it (a legacy ``GateGeneratorSpec`` never reads ``None``), while
``GateGenerator.config`` reports the config as passed.

As for tracks, the config only requires non-empty strings, and ``GateGenerator`` (hence
``Course``) resolves the names, after the generator and its ``generator_params``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Cause
     - Error at ``GateGenerator`` construction
   * - An unknown ``gate_ordering``
     - ``ValueError``: ``GateGenConfig.gate_ordering: unknown order 'spiral'; available:
       ['ccw', 'random_pairs', 'raw']`` (plus the installed plugins' names).
   * - An order the generator does not accept
     - ``ValueError``: ``gate generator 'polar' does not support gate_ordering
       'random_pairs'; supported: ['ccw', 'raw'] (its preferred order is 'ccw')`` (the
       hint only when the generator declares a preferred order).
   * - An unknown ``z_profile``
     - ``ValueError``: ``GateGenConfig.z_profile: unknown elevation 'bogus'; available:
       ['flat', 'noise', 'random_walk', 'uniform']`` (plus the installed plugins'
       names).
   * - A ``z_profile`` other than ``"flat"``, or a non-zero ``z_base``, with a generator
       that provides its own altitudes
     - ``ValueError`` naming the generator.
   * - A name served by a plugin that failed to load
     - That plugin's ``PluginLoadError``.

``GateGenConfig(gate_ordering="spiral")`` and ``GateGenConfig(z_profile="bogus")``
therefore construct; before this release both raised at config construction. A legacy
``GateGeneratorSpec`` orders its own gates: ``gate_ordering`` must be in its
``supported_orderings``, as before (``None``: ``"ccw"`` when it supports it, else the
first of them, sorted).

.. _gate-capacity-config:

Gate counts: ``max_gates``, ``min_gates`` and the generator's capacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three numbers bound the gates of an env:

- ``GateGenConfig.max_gates`` (``BatchSpec.max_gates``) is the ``GateSequence`` row
  stride: the slots per env, NaN past ``count[e]``.
- The generator's **capacity** is the most anchors it emits per env (a gate generator
  Stage's ``capacity()["gates"]``, a legacy spec's ``max_gates(config)``): 13 for
  ``bezier`` and ``hull`` at the default ``max_num_points``, ``polar_num_knots`` for
  ``polar``, ``checkpoint_count`` for ``checkpoint``, ``voronoi_control_points`` for
  ``voronoi``. A Stage also declares the fewest it emits (``capacity()["min_gates"]``).
- ``GateGenConfig.min_gates`` is the validity floor: an env with fewer real gates is
  invalid (fail bit ``count``).

``GateGenerator`` requires ``min_gates <= capacity <= max_gates``, computed from the
resolved ``generator_params``:
``GateGenConfig.min_gates=16 is too large for gate generator 'polar'; generator can
produce at most 12 gates.`` or ``GateGenConfig.max_gates=8 is too small for gate
generator 'polar'; required max_gates 12.`` (``PlanError`` with ``BatchSpec.max_gates`` /
``min_gates`` in the text when a spec is built). So
``GateGenConfig(generator="polar", generator_params={"polar_num_knots": 9})`` makes 9
gates per env.

.. _width-config:

Road width: ``half_width``, ``w_floor`` and the width stage
-----------------------------------------------------------

``TrackGenConfig.half_width`` is the road's half-width. ``TrackGenerator(config)`` turns
it into the pipeline's width stage, ``width.constant(half_width=config.half_width)``,
which writes the per-point half-width ``cl.half_width`` before relaxation; XPBD, the band
offset and the validity checks all read that channel (see :ref:`pipeline-width`). It also
sets the default ``spacing`` (``0.6 * half_width``). ``Track.half_width`` reports it per
point: the constant on the real points, NaN on the padding. The outputs of a
``TrackGenConfig`` are unchanged from 0.2.0.dev3.

A half-width that varies along the road needs a pipeline spec (the flat config has no
width shortcut):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``presets.track(...)``
     - Effect
   * - ``half_width=0.1`` (default)
     - The value of ``width.constant``, or of the ``half_width`` field of another width
       stage that has one (a plugin's), filled when its params do not set it; params
       that set another value raise ``ValueError``. It also sets the default
       ``spacing``. With a width stage that has no such field, or a generator that
       provides its half-widths, it only sets the default spacing, and a ``half_width``
       other than the default 0.1 with an explicit ``spacing`` raises ``ValueError``
       (nothing would read it).
   * - ``width="constant"`` (default)
     - The width stage: a registered name, ``(name, params)`` or a ``StageRef``, such as
       ``StageRef("width", "from_channel", {"hi": 0.3}, ports={"src": "cl.acme.hw"})``
       (a generator's per-point attribute as the width). ``None`` raises
       ``ValueError``; a ``width`` other than the default ``"constant"`` with a
       generator that provides ``half_width`` raises too.
   * - ``registry=None``
     - The registry the preset reads to see what the generator provides and which
       width stages have a ``half_width`` field (default ``track_gen.plugin.registry``).

``presets.gates(width=...)`` is the same for a ``source="relaxed_stations"`` gate
pipeline, whose centerline is relaxed with the width before the stations are placed.

``TrackGenConfig.w_floor`` (``validity.default``'s ``w_floor``) is the smallest
half-width a valid track may have: fail bit 4 (``w_floor``) is set when a real final
point's half-width is not above it, or is NaN. At a constant half-width every point
passes or fails together, as in earlier releases (where the check could only compare the
configured ``half_width`` with ``w_floor``); with a per-point width it finds the pinch.
``relax_tol`` is the tolerance of the thickness check (fail bit 3), which compares each
point's curvature radius and its distance to the non-neighbouring points with the
half-widths there.

``StationsParams.spacing_metric`` (``presets.gates(spacing_metric=...)``) is the arc
length the ``stations`` stages measure: ``"xy"`` (default, plan-view) or ``"xyz"`` (3D,
which needs an altitude on the centerline: a generator that provides ``z``, or
``source="relaxed_stations"`` with ``elevation_on="centerline"``).
``presets.gates(elevation_on=...)`` chooses where the elevation runs: ``"gates"`` (on the
gate slots, the default for ``"xy"``) or ``"centerline"`` (on the relaxed centerline,
before the stations, the default for ``"xyz"``). ``GateGenConfig`` has no such fields:
its gate generators place their own gates. See :ref:`pipeline-stations-xyz`.

A pipeline spec saved with ``to_dict()`` carries ``"format": "track_gen.pipeline/2"``.
``PipelineSpec.from_dict`` still reads the ``/1`` dicts of 0.2.0.dev3 and of the
0.2.0.dev4 branch before the width moved (gate specs included), migrating them (the
``half_width`` keys of ``relax.xpbd``, ``offset.default`` and ``validity.default``
removed, the width stage moved before ``relax``) with one ``DeprecationWarning``; see
:ref:`pipeline-spec-format`.

.. _config-relax-auto:

The relax stage of a preset: ``relax="auto"``
---------------------------------------------

``presets.track(relax=...)`` and ``presets.gates(relax=...)`` default to ``"auto"``:
the generator's preferred relax stage (``Stage.preferred["relax"]``) when it declares
one, else ``"xpbd"``. No built-in shape generator declares one, so their pipelines are
unchanged (XPBD, byte for byte); the ingestion generator ``external`` prefers ``"none"``,
so loaded centerlines are resampled but not moved. An explicit ``relax=`` (a name,
``(name, params)``, a ``StageRef``, or ``None`` for ``relax.none``) overrides it.
``TrackGenConfig`` has no ``"auto"``: ``relax_enable`` and ``relax_solver`` select the
stage as before.

.. _config-randomize:

Per-env randomization: ``randomize=``
-------------------------------------

``presets.track(randomize=...)`` and ``presets.gates(randomize=...)`` draw dynamic
parameters per env, on the device, at every generation: each env gets its own value of a
field, drawn uniformly in a range, inside the Program (and so inside a recorded graph).

.. code-block:: python

   spec = presets.track(
       num_envs=4096, device="cuda:0",
       generator=StageRef("generator", "acme.superformula"),
       randomize={"generator.m": (3.0, 9.0),          # "<stage id>.<field>": (lo, hi)
                  "generator.n1": (0.2, 0.8)},
   )

**What can be drawn: dynamic ``float32`` fields.** A field declared with
``plugin.dynamic`` (a plugin generator's or stage's) and the three dynamic fields of the
built-in ``width.from_channel`` (``width.scale``, ``width.lo``, ``width.hi``). **The
built-in generators' and stages' knobs are static in 0.x** (``rad``,
``polar_radial_jitter``, the elevation ranges, ...): naming one raises ``PlanError``,
which lists the fields of the pipeline that can be drawn (``generator.rad is static:
built-in knobs are static in 0.x; dynamic float32 fields in this pipeline: [...]``). So
``randomize=`` is per-env sampling of *dynamic* fields, not yet a domain randomization of
the built-in generators; for those, mix generators or configurations in one batch
(:ref:`results-mixed-batches`) or use ``bezier``'s ``style_sampling``. ``int32`` fields
cannot be drawn in 0.x either.

**The form.** A mapping from ``"<stage id>.<field>"`` (split at the last dot: stage ids
may contain dots, field names may not) to ``(lo, hi)``, or a sequence of ``StageRef``\ s
of kind ``randomize``. Each entry becomes a stage
``StageRef("randomize", "uniform", {"stage": s, "field": f, "lo": lo, "hi": hi},
id=f"randomize.{s}.{f}")`` inserted right **before** stage ``s``: listing order is
execution order, so the draw is written before its target reads it, in the same
generation. ``to_dict`` / ``from_dict`` round-trip it (format ``/2``, unchanged). In a
spec you build by hand, use ``spec.insert_before("generator", StageRef("randomize",
"uniform", {...}, id=...))``; ``RandomizeUniformParams`` has ``stage`` and ``field``
(static) and ``lo`` and ``hi`` (dynamic, default ``0.0`` and ``1.0``).

**The draw.** Env ``e`` gets ``lo[e] + u * (hi[e] - lo[e])``, ``u`` uniform in ``[0,
1)``, from a stream keyed on its effective seed and the randomizer's salt (hashed from its
stage id), then clamped to the target field's declared ``lo`` / ``hi``. So the draws are
deterministic per ``(seed, epoch)``, differ across envs and across randomizers, and
change with ``advance=True`` (each replay of a graph that records
``generate(advance=True)`` redraws) or a reseed. A NaN bound gives NaN (the target's
own checks then fail the env).

**Bounds and curricula.** The randomizer's ``lo`` and ``hi`` are dynamic: write
``gen.params["randomize.generator.m"]["hi"]`` (or ``.set("hi", ...)``), in your own graph
too, and the next generation draws in the new range. The initial range must lie within
the target field's declared bounds, and ``lo <= hi`` (``PlanError`` / ``ValueError`` at
build). At run time ``set("lo" | "hi", value)`` checks a scalar against the same
declared bounds (``ValueError``). An array write (``set`` with an array, or your own
kernel) is not checked: a range wider than the field's bounds is clamped to them, so the
excess piles onto the bound (the draws are no longer uniform).

**Masks.** Under ``generate(advance=True, mask=m)`` the masked envs get new draws; the
others keep their effective seed and bounds, so they redraw the same values and their
rows are restored by the commit: nothing changes for them.

**The drawn field** is owned by its randomizer: ``gen.params["generator"]["m"]`` is the
``[E]`` array of the current draws (read it to log what each env was built with), but
``set("m", ...)`` raises and a direct write is overwritten at the next generation (see
:ref:`generator-runtime-params`).

**Rules** (each a ``PlanError`` at build, before anything is allocated): a randomizer
listed after its target; a stage other than the generator's randomizers before the
generator (the generator stays the first stage that is not a randomizer, with id
``"generator"``); an unknown target stage or field; a target that is itself a
randomizer; a static, ``int32`` or unknown field; two randomizers on one field. A
``randomize=`` key naming a stage the preset does not have raises ``ValueError``
(``randomize: key 'gnerator.amp' names stage 'gnerator', which this pipeline does not
have; stages: [...]``).

**Cost.** Two more launches per randomizer per generation, net: the draw and the latch
of its ``lo`` and ``hi``, minus the latch the drawn field no longer needs. A pipeline
without a randomizer launches what it launched before. ``TrackGenConfig`` has no
``randomize``.

.. _generator-params-config:

Per-generator parameters: ``generator_params``
----------------------------------------------

Every generator can declare its parameters as a Params class (see
:ref:`the generator guide <generator-params>`). The last field of both configs,
``generator_params``, selects the values:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``generator_params``
     - Meaning
   * - ``None`` (default)
     - The generator's ``Params.from_legacy(config)``. For a built-in, that copies the flat
       fields listed below (``polar_num_knots``, ``voronoi_control_points``, ...), so every
       existing config keeps its meaning and its outputs, byte for byte. A plugin gets its
       class defaults unless it overrides ``from_legacy``.
   * - A mapping
     - The generator's complete parameter set, by field name. Keys left out take the
       class defaults, not the flat fields. An unknown key raises ``ValueError`` listing
       the valid fields.
   * - A Params instance
     - The same, as an instance of the generator's own class. An instance of another
       class raises ``TypeError`` naming the expected one.

These three configs build identical generators:

.. code-block:: python

   from track_gen import TrackGenConfig
   from track_gen.plugin import registry

   PolarParams = registry.get("generator", "polar").params

   TrackGenConfig(generator="polar", polar_num_knots=9)
   TrackGenConfig(generator="polar", generator_params={"polar_num_knots": 9})
   TrackGenConfig(generator="polar", generator_params=PolarParams(polar_num_knots=9))

Rules:

- **Set a knob in one place.** With ``generator_params`` given, a flat field that the
  selected built-in reads must stay at its dataclass default or equal the
  ``generator_params`` value; otherwise construction raises ``ValueError`` naming the
  fields. The flat fields of the other generators are ignored, as before.
- **Switching generators means resetting it.** ``dataclasses.replace(cfg,
  generator="polar")`` on a config whose ``generator_params`` is a ``BezierParams`` raises
  ``TypeError``; pass ``generator_params=None`` (or polar's) in the same ``replace``.
- **It is resolved when the generator is constructed** (``TrackGenerator``,
  ``GateGenerator``, hence ``Course``), together with the rules of the selected generator
  (see :ref:`generator-params-validation`). Config construction checks only its type: a
  value that is not ``None``, a mapping or a dataclass instance raises ``TypeError``.
- **A generator without a Params class takes none.** A non-empty mapping raises
  ``ValueError`` ("takes no generator_params").
- ``gen.config`` returns the config as passed, with a mapping normalised to the Params
  instance, so ``TrackGenerator(gen.config, rng)`` builds an identical generator.
  ``gen.params["generator"].params`` is the resolved instance, also when
  ``generator_params`` is ``None``. Both hold the construction-time (initial) values of a
  Stage's *dynamic* parameters: after ``gen.params["generator"].set("amp", ...)`` they
  still report the initial value, and a generator rebuilt from ``gen.config`` starts
  from it. The current values are the live arrays, ``gen.params["generator"]["amp"]``.
- A ``Course`` passes ``generator_params`` through:
  ``CourseConfig(gen=TrackGenConfig(..., generator_params=...))``.
- ``GateGenConfig.generator_params`` works the same way, and the gate-count check
  (``min_gates <= capacity <= config.max_gates``, see :ref:`gate-capacity-config`) sees
  the resolved values: ``GateGenConfig(generator="polar", generator_params={"polar_num_knots":
  9})`` makes 9 gates.

The built-in Params classes have no public import path: pass a mapping, or reach a class
through ``track_gen.plugin.registry.get(kind, name).params`` and its schema through
``registry.describe(kind, name).params_schema``. Their field names are the flat ones,
prefixes included (``polar_num_knots``, ``voronoi_control_points``, ``repulsive_*``), and
they are the built-ins' permanent parameter names through 1.0: ``generator_params``
mappings and YAML files written with them stay valid. ``generator``, ``num_envs``, ``device``,
``num_points``, ``scale`` and every relaxation, inflation, spacing and Z field are core
fields: they stay flat and belong to no Params class. A Stage generator receives ``scale``
as ``StageBuilder.scale``, and ``num_points`` is its default raw point count.

.. _builtin-generator-params:

Built-in generator parameters
-----------------------------

Every field is static, its default is the config field's default, and the rule in the last
column is checked when the generator is constructed, with the same message as before
(``lo``/``hi`` in the schema mirror these rules). A field without a rule is used as given;
where the generator clamps a value, the description says so. The descriptions of each knob
are on the generator pages (:doc:`/generators/overview`).

Track generators
~~~~~~~~~~~~~~~~

``BezierParams`` (``generator="bezier"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``min_num_points``
     - ``9``
     - ``>= 2``
   * - ``max_num_points``
     - ``13``
     - ``>= min_num_points``
   * - ``num_points_per_segment``
     - ``30``
     - ``>= 2``
   * - ``min_point_distance``
     - ``0.05``
     - in ``(0, 0.5]``
   * - ``rad``
     - ``0.4``
     -
   * - ``edgy``
     - ``0.0``
     -
   * - ``handle_clamp_frac``
     - ``0.4``
     -
   * - ``style_sampling``
     - ``False``
     -
   * - ``rad_range``
     - ``None``
     - ``(lo, hi)`` or ``None``
   * - ``scale_range``
     - ``None``
     - ``(lo, hi)`` or ``None``
   * - ``handle_clamp_frac_range``
     - ``None``
     - ``(lo, hi)`` or ``None``

``HullParams`` (``generator="hull"``): ``min_num_points``, ``max_num_points``,
``num_points_per_segment`` and ``min_point_distance`` with the defaults and rules of
``bezier`` (the two share the corner sampler), plus:

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``hull_displacement``
     - ``0.15``
     -

``PolarParams`` (``generator="polar"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``num_points_per_segment``
     - ``30``
     - ``>= 2``
   * - ``polar_num_knots``
     - ``12``
     - none (the generator clamps it to ``>= 4``)
   * - ``polar_radial_jitter``
     - ``0.60``
     - none (clamped to ``[0, 0.85]``)
   * - ``polar_angular_jitter``
     - ``0.30``
     - none (clamped to ``[0, 0.45]``)

``VoronoiParams`` (``generator="voronoi"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``num_points_per_segment``
     - ``30``
     - ``>= 2``
   * - ``voronoi_num_sites``
     - ``256``
     - ``>= voronoi_control_points``
   * - ``voronoi_site_layout``
     - ``"void_ring"``
     - one of ``"ring"``, ``"void_ring"``, ``"clustered"``, ``"mixed"``
   * - ``voronoi_control_points``
     - ``18``
     - ``>= 6``
   * - ``voronoi_radial_variation``
     - ``0.62``
     - ``>= 0``
   * - ``voronoi_angular_jitter``
     - ``0.08``
     - ``>= 0``

``CheckpointParams`` (``generator="checkpoint"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``checkpoint_count``
     - ``12``
     - ``>= 3``
   * - ``checkpoint_radius_min_frac``
     - ``0.33``
     - in ``[0, 1)``
   * - ``checkpoint_angle_jitter``
     - ``0.55``
     -
   * - ``checkpoint_turn_rate``
     - ``0.42``
     -
   * - ``checkpoint_steer_gain``
     - ``0.65``
     -
   * - ``checkpoint_lookahead_frac``
     - ``0.16``
     -
   * - ``checkpoint_best_of_k``
     - ``4``
     - ``>= 1``
   * - ``checkpoint_clip_fallback``
     - ``False``
     -

``RepulsiveParams`` (``generator="repulsive"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``repulsive_grow_mult_min``
     - ``4.5``
     - ``>= 1``
   * - ``repulsive_grow_mult_max``
     - ``5.5``
     - ``> 1`` and ``>= repulsive_grow_mult_min``
   * - ``repulsive_domain_frac``
     - ``0.35``
     - ``> 0``
   * - ``repulsive_domain_init_ratio``
     - ``4.0``
     - ``> 1``
   * - ``repulsive_obstacle_count_min``
     - ``8``
     - ``>= 1``
   * - ``repulsive_obstacle_count_max``
     - ``12``
     - ``>= repulsive_obstacle_count_min``
   * - ``repulsive_obstacle_radius_min_frac``
     - ``0.02``
     - ``> 0``
   * - ``repulsive_obstacle_radius_max_frac``
     - ``0.045``
     - ``>= repulsive_obstacle_radius_min_frac``
   * - ``repulsive_ratchet_rate``
     - ``0.012``
     - ``> 0``
   * - ``repulsive_alpha``
     - ``3.0``
     - ``> 0``
   * - ``repulsive_beta``
     - ``6.0``
     - ``> 0``
   * - ``repulsive_tau``
     - ``0.4``
     -
   * - ``repulsive_w_len``
     - ``30.0``
     -
   * - ``repulsive_stages``
     - ``(64, 128, 256)``
     - non-empty, positive multiples of 4, strictly increasing; the last equals
       ``num_points``
   * - ``repulsive_settle_iters``
     - ``40``
     - ``>= 1``
   * - ``repulsive_resample_every``
     - ``25``
     - ``>= 1``
   * - ``repulsive_stall_window``
     - ``16``
     - ``>= 1``
   * - ``repulsive_stall_area_tol``
     - ``0.05``
     -
   * - ``repulsive_deactivate_obstacles``
     - ``True``
     -

``ExternalParams`` (``generator="external"``, the ingestion generator of
:ref:`ingesting-centerlines`; ``track_gen.stages.external.ExternalParams``). It has no
flat field: set it with ``generator_params``. Every field is static.

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Field
     - Default
     - Rule
   * - ``with_z``
     - ``False``
     - ``True`` adds the input ``centerline.z`` and makes the generator provide ``z``
       (no elevation stage)
   * - ``with_half_width``
     - ``False``
     - ``True`` adds the input ``centerline.half_width`` and makes it provide
       ``half_width`` (no width stage)
   * - ``attributes``
     - ``()``
     - ``(key, dtype, interp)`` triples: ``key`` a unique plugin key
       ``<package>.<name>``, ``dtype`` one of ``"float32"``, ``"int32"``,
       ``"vec2f"``, ``"vec3f"``, ``interp`` a policy valid for that dtype

Its raw capacity is ``num_points`` (``TrackGenConfig.num_points``,
``presets.track(num_points=)``): the most points per env you can load.
``TrackGenConfig(generator="external", generator_params={...})`` works through the
legacy path; there the config's ``relax_enable`` (``True`` by default) decides whether
XPBD runs, whereas the presets run no relax for it by default (:ref:`config-relax-auto`).

Gate generators
~~~~~~~~~~~~~~~

The built-in gate generators are anchors Stages (:ref:`gate-generator-stage`). Each one's
capacity (:ref:`gate-capacity-config`) and accepted orders are given with its Params.

``PointGateParams`` (``generator="bezier"`` and ``generator="hull"``; capacity
``max_num_points``, at least ``min_num_points``; every order except ``"raw"``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``min_num_points``
     - ``9``
     - ``>= 2``
   * - ``max_num_points``
     - ``13``
     - ``>= min_num_points``
   * - ``min_point_distance``
     - ``0.05``
     - ``> 0``

``PolarGateParams`` (``generator="polar"``; one gate per knot, capacity
``polar_num_knots``; every order except ``"random_pairs"``; normalized to
``TARGET_EXTENT * scale``):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``polar_num_knots``
     - ``12``
     -
   * - ``polar_radial_jitter``
     - ``0.60``
     -
   * - ``polar_angular_jitter``
     - ``0.30``
     -

``VoronoiGateParams`` (``generator="voronoi"``; anchors are emitted directly as gates, so
three control points suffice; capacity ``voronoi_control_points``; every order except
``"random_pairs"``; normalized):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``voronoi_num_sites``
     - ``256``
     - ``>= voronoi_control_points``
   * - ``voronoi_site_layout``
     - ``"void_ring"``
     - one of ``"ring"``, ``"void_ring"``, ``"clustered"``, ``"mixed"``
   * - ``voronoi_control_points``
     - ``18``
     - ``>= 3``
   * - ``voronoi_radial_variation``
     - ``0.62``
     - ``>= 0``
   * - ``voronoi_angular_jitter``
     - ``0.08``
     - ``>= 0``

``CheckpointGateParams`` (``generator="checkpoint"``; one gate per checkpoint, capacity
``checkpoint_count``; every order except ``"random_pairs"``; normalized):

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Field
     - Default
     - Rule
   * - ``checkpoint_count``
     - ``12``
     - ``>= 3``
   * - ``checkpoint_radius_min_frac``
     - ``0.33``
     - in ``[0, 1)``
   * - ``checkpoint_angle_jitter``
     - ``0.55``
     -

The ``GateGenConfig`` fields ``num_points_per_segment``, ``rad``, ``edgy``,
``handle_clamp_frac`` and ``hull_displacement`` belong to no gate generator: they are
inert, and ``generator_params`` cannot set them.

.. _generator-params-validation:

When the rules are checked
--------------------------

Config construction checks the core fields only (for tracks ``num_envs``,
``half_width``, the relaxation and smoothing knobs, ``output_mode``, spacing and the Z
fields, with ``z_profile`` only required to be a non-empty string; for gates
``num_envs``, the gate counts, radius, width, solver iterations, alignment and the Z
fields, with ``z_profile`` only required to be a non-empty string and ``gate_ordering``
``None`` or a non-empty string)
and the type of ``generator_params``. The ``relax_solver``, ``gate_ordering`` and
``z_profile`` names are resolved by ``TrackGenerator`` and ``GateGenerator`` (see
:ref:`flat-stage-shortcuts` and :ref:`gate-stage-shortcuts`). The
generator-specific rules in the tables above run when the generator is constructed
(``TrackGenerator``, ``GateGenerator``, and so ``Course``), and only for the selected
generator: first its ``validate()``, with the same conditions and messages as in
0.2.0.dev1, then the generic ``lo``/``hi``/``choices`` checks. In 0.2.0.dev1 every config
ran every rule in its ``__post_init__``. What follows from the move:

- **An invalid knob of a generator that is not selected is accepted and ignored.**
  ``TrackGenConfig(generator="bezier", voronoi_control_points=3)`` used to raise; it now
  builds, and only ``generator="voronoi"`` rejects it.
- **The errors move from the config to the generator.**
  ``TrackGenConfig(generator="voronoi", voronoi_control_points=3)`` constructs, and
  ``TrackGenerator(config, rng)`` raises the ``ValueError`` the config used to raise.
- **An unknown generator name is reported first**, before any parameter error.
- **The facades' own checks come before the rules.** ``TrackGenerator`` and
  ``GateGenerator`` check ``rng`` (``None``, or a seed count other than ``num_envs``)
  first, and ``Course`` refuses a capture (``CaptureError``) and allocates its rng
  first. With several errors, these are reported before the parameter error: for example
  a ``Course`` built with a bad generator knob inside a capture raises ``CaptureError``,
  where the config used to raise ``ValueError``.
- **The first error can differ when several rules fail.** The rules run in the order of the
  selected generator's ``validate()``. For ``repulsive``, the ``repulsive_stages[-1] ==
  num_points`` rule (which only ever applied to ``generator="repulsive"``) now runs after
  the ``settle_iters``, ``resample_every`` and ``stall_window`` rules.
- **Private helpers do not validate.** Calling ``spec.alloc_scratch(cfg)``,
  ``warp_pipeline._inflate_warp_alloc`` or ``_run_pipeline`` directly skips the
  generator-specific rules.
- **The parameter explorer reports these errors when it generates** (``generate_batch``)
  rather than when it builds the config.

The moved rules are exactly those in the tables: the point-family rules of ``bezier`` and
``hull`` (track and gate), ``num_points_per_segment >= 2`` for ``bezier``, ``hull``,
``polar`` and ``voronoi``, and every ``voronoi_*``, ``repulsive_*`` and ``checkpoint_*``
rule. The new errors, all at generator construction unless noted:

.. list-table::
   :header-rows: 1
   :widths: 56 44

   * - Cause
     - Error
   * - An unknown key in a ``generator_params`` mapping
     - ``ValueError`` listing the valid fields
   * - A non-empty ``generator_params`` for a generator without a Params class
     - ``ValueError`` ("takes no generator_params")
   * - A Params instance of another class
     - ``TypeError`` naming the expected class
   * - A flat field and ``generator_params`` setting the selected built-in's knob to
       different values
     - ``ValueError`` naming the fields
   * - A value outside a field's ``lo``/``hi`` or ``choices``
     - ``ValueError``
   * - A ``generator_params`` that is not ``None``, a mapping or a dataclass instance
     - ``TypeError``, at config construction

.. _deprecated-config-fields:

Deprecated fields
-----------------

``TrackGenerator`` does not use the ``TrackGenConfig`` fields below (it only rejects
``smooth_finish=True``); only the torch oracle in the tests, a benchmark or
``track_gen._experimental.fourier`` read them. Setting any of them to a value other than
its default (the default of the config's own class, for a subclass) emits one
``DeprecationWarning`` at config construction, attributed to the code that constructs the
config and naming every such field: ``TrackGenConfig fields [...] are deprecated and
unused by TrackGenerator (oracle/experimental only; smooth_finish=True is rejected); they
will be removed in 1.0``. Nothing is removed in 0.x, and the fields keep their defaults.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Fields
     - Read only by
   * - ``energy_steps``, ``energy_lr``, ``energy_w_sep``, ``energy_w_len``,
       ``energy_w_bend``, ``energy_w_anchor``
     - The torch oracle and ``benchmarks/benchmark_relaxation.py``
   * - ``tp_iters``, ``tp_tau``, ``tp_alpha``, ``tp_beta``
     - The torch oracle
   * - ``smooth_finish``, ``smooth_finish_iters``, ``smooth_finish_tau``
     - The torch oracle. ``TrackGenerator`` still raises ``ValueError`` for
       ``smooth_finish=True``.
   * - ``relax_chunk_size``, ``relax_use_warp``
     - The torch oracle
   * - ``max_regen_iters``
     - The torch oracle's accept-first-valid loop
   * - ``min_angle``
     - The corner-angle kernel, launched only by the tests
   * - ``num_harmonics``, ``decay_p``, ``amplitude``, ``num_centerline_samples``
     - ``track_gen._experimental.fourier``
   * - ``output_mode``
     - Nothing. It accepts only ``"constant_spacing"`` (any other value raises
       ``ValueError``), so it never warns: it is deprecated in this documentation only.

The inert ``GateGenConfig`` fields listed above are not deprecated in this release.
