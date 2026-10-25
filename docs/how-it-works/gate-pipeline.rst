.. _gate-pipeline:

.. _pipeline-gates:

The gate pipeline
=================

``GateGenerator`` runs a ``Program`` too. Its :term:`spec` is a *gate pipeline*:
``BatchSpec.max_gates`` is set, the result is a ``GateSequence``, and the :term:`stages
<stage>` take gate *anchors*, the plan-view points a generator emits, to ordered,
separated, elevated, framed and validated gates, one sequence per :term:`env` and
:term:`generation`. Every rule of :doc:`pipeline-stages` applies to it unchanged (the
planner, versioned rewrites of :term:`channels <channel>`, :term:`fail bits`, the commit,
taps, capture classes, the :ref:`parameter latch <pipeline-latch>`); this page lists
what differs.
:ref:`pipeline-gate-presets` shows the user-facing side.

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
                 elevation=("noise", {"z_min": -0.3, "z_max": 0.3,
                                      "z_noise_amplitude": 0.3}))

   # Gates for a track generator: arc-length stations on its centerline.
   presets.gates(num_envs=64, device="cpu", generator="bezier", source="stations",
                 gates=12)

.. _pipeline-gate-stages:

The gate stages
---------------

``legacy.to_gate_pipeline_spec(config)`` and ``presets.gates(...)`` produce this list
from a native gate generator. Stages in *italics* below are present only when their
condition holds (`Launches`_ counts what they launch).

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
``z_valid_grade`` (see `Gate fail bits`_). ``gate_frames.default``'s
``gate_width`` is a :term:`dynamic field` (per env) (:ref:`gate-width-per-env`). The generator's
params are its ``generator_params``; the order and ``carry`` stages take none, and
``normalize.extent`` takes ``size`` (``NormalizeExtentParams``, a dynamic per-env
multiplier, default ``1.0``, ``GateGenConfig.size`` for the ``polar``, ``voronoi`` and
``checkpoint`` gate generators): it scales to ``ops.constants.TARGET_EXTENT *
BatchSpec.scale * size[e]``.

Then the commit (`The gate commit`_) and the ``generation`` write. A centerline generator replaces
``generator`` with ``generator`` + ``stations`` (`Stations: gates from any track
generator`_).

.. _gate-pipeline-two-point-domains:

Two point domains
-----------------

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
-------------

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
raises ``PlanError`` (:ref:`Plugin channels <pipeline-plugin-channels>`).

Anchors generators
------------------

A native gate generator is a ``Stage`` with ``kind="gate_generator"`` and
``output="anchors"``: it reads no channel, writes ``anchors.xy`` and ``anchors.count``,
and does not order, normalize or lift anything (:ref:`writing-a-gate-generator` is the
guide, :ref:`gate-generator-stage` the stage's contract). Its ``capacity(params, layout)`` returns ``{"gates": A, "min_gates": k}``: ``A``
is the most anchors it writes per env (its row stride), ``k`` the fewest (default ``0``).


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
-------------------

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
       ``.default``; validity checks the resulting nonzero bars

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
--------------------

A gate-domain stage (an elevation, a check, an order plugin) reads and writes its own
``gates.<package>.<name>`` channels (``per="gates"``, ``[E * G]``, or the default
``per="env"``, ``[E]``), and exports the per-gate ones to ``GateSequence.attributes``:

.. snippet: skip=plugin-package

.. code-block:: python

   # a gate check: writes = {"fail": "valid.fail", "clearance": "gates.acme.clearance"}
   clearance = b.write("clearance", wp.float32, per="gates", export=True)

   gates = gen.generate()
   gates.attributes["acme.clearance"]      # [E * G] float32, NaN on the padding

Per-anchor attributes reach the gate slots by anchor index, as provided attributes do:

- a gate generator writes ``anchors.<package>.<name>`` channels (any dtype of
  :ref:`Carried channels <pipeline-carried-channels>`, with ``export=`` but no ``interp=``: they are copied, not
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
---------------------

Ordering is a stage of the open kind ``order``, applied by the pipeline: generators no
longer order their own anchors. An order stage reads ``anchors.xy`` and
``anchors.count``, writes the first ``m = min(count, A, G)`` gate slots of ``gates.xy``
(NaN after them) and ``gates.count``. The built-ins:

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
  ``frozenset({"raw"})``.

``preferred = {"order": name, "normalize": name}`` names the generator's default order
and its normalization (``preferred["normalize"] = "extent"`` inserts
``normalize.extent``). An order the generator does not accept raises (``PlanError`` from
the planner, ``ValueError`` from the config): ``gate generator 'polar' does not support
gate_ordering 'random_pairs'; supported: ['ccw', 'raw'] (its preferred order is 'ccw')``,
the hint only when the generator declares a preferred order. ``presets.gates`` does not
check the order: ``build`` (or the generator's construction) does.

The default order is one rule, for an anchors generator: its ``preferred["order"]`` when
it accepts it, else ``"ccw"`` when it accepts it, else the first order it accepts
(``"raw"`` for stations). ``presets.gates(order=None)``,
``GateGenConfig.gate_ordering=None`` (the config default, resolved by
``bind_generator`` before the run config and pipeline spec are constructed)
and the conformance kit all apply it, so ``GateGenerator(GateGenConfig(
generator="acme.helix"))`` runs ``order.raw``. The run config holds the resolved name, never ``None``. For the five
built-ins the rule gives ``"ccw"``. An explicit name is checked against the orders the
generator accepts.

.. _pipeline-stations:

Stations: gates from any track generator
----------------------------------------

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
------------------

``StationsParams.spacing_metric`` chooses the arc length the stations measure:
``"xy"`` (default, the plan-view length) or
``"xyz"``, the 3D length along the loop, so that the gates are equally spaced in flight
distance on a course that climbs. ``presets.gates(spacing_metric=...)`` sets it. The ``"xyz"`` branch
accumulates the segment lengths in ``float64`` (the 3D spacing is equal to well within
``1e-6``, relative), and carries the altitude to the anchors with ``"linear"``.

3D spacing needs an altitude on the loop *before* the stations are placed. It comes from
one of two places:

- **the generator**, one that provides ``z`` (``raw.z``, :ref:`Generator-provided altitude
  and width <pipeline-provided>`), with ``source="stations"`` or ``"relaxed_stations"``;
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
     - the stations, the order, then gate-level ``gate_arclen`` and ``elevation`` (a
       non-flat profile) and the other gate stages
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

   from track_gen.pipeline import presets

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
------------------

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

On gates, ``elevation.uniform`` and ``elevation.random_walk`` draw one altitude per
gate: the gates are the knots, so their ``z_control_points`` is not
read (``GateGenConfig`` has no such field; a value set through ``presets.gates(
elevation=("uniform", {...}))`` changes nothing). ``elevation.flat`` writes ``z_base``
into every real gate (``presets.gates(z_base=...)``, ``GateGenConfig.z_base``); on a
track it launches nothing and ``lift.flat`` adds the altitude, so a non-zero ``z_base``
on a track's ``elevation.flat`` raises: ``ValueError`` from ``presets.track``,
``PlanError`` from the planner, before anything is allocated. The dynamic fields of the
elevation stages (``z_min``, ``z_max``, ``z_max_step``, ``z_noise_amplitude``) are per
env on gates too, under the same stage id: ``presets.gates(randomize={"elevation.z_max":
(lo, hi)})`` (:ref:`Dynamic built-in knobs <pipeline-dynamic-builtins>`).

.. _pipeline-gate-fail-bits:

Gate fail bits
--------------

``validity.gate_default`` runs one fused kernel, ``_finalize_validity_bars_k``, which
writes its own 0/1 flag ``gates.valid`` and one bit per check into ``valid.fail``, and
decides the crossing check per env (below). ``GateSequence.fail_mask`` is the final
``valid.fail``, and ``valid[e] == (fail_mask[e] == 0)``.

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
     - two gate bars cross (checked in each env that has a real gate with a non-zero
       ``half_size``, a NaN included)
   * - 5
     - ``grade``
     - a chord's ``|dz| / ds`` exceeds ``z_valid_grade`` (only when it is ``> 0``)
   * - 6-15
     - reserved
     - Reserved built-in gate-check range
   * - 16-30
     - plugin checks
     - assigned by ``StageBuilder.fail_bit()``, in pipeline order (as for tracks)

``track_gen.pipeline.GATE_FAIL_BITS`` is this table (a read-only mapping, next to
``BUILTIN_FAIL_BITS``). A gate validity plugin is a ``validity`` stage that reads the gate
channels it checks and rewrites ``valid.fail`` exactly as a track check does (see
:ref:`stage-validity-check`); add it with ``presets.gates(...,
checks=("gate_default", "acme.max_height"))``. In a gate pipeline ``valid.fail`` must
start at ``validity.gate_default``.

**The crossing check runs per env.** While it scans an env's real gates for the other
checks, the kernel notes whether any has a non-zero ``half_size`` (a NaN counts as
non-zero); only then does it run the pairwise bar-crossing loop. A zero-width bar has
``left == right == position``, and a degenerate segment never crosses, so skipping the
loop for a point-gate env changes no decision: the valid and fail bits are those of a
kernel that checks every env. So a run-time bar width (``gate_frames.gate_width`` drawn or set
from point gates, :ref:`gate-width-per-env`) is checked like a width built in, and the
default point-gate pipelines pay nothing (one compare per real gate in an existing loop).
``GateValidityParams`` accepts ``min_gates``, ``gate_radius`` and ``z_valid_grade``.
Crossing is derived from the gate geometry, including provided poses; it has no separate
width or crossing-mode setting.

.. _pipeline-gate-commit:

The gate commit
---------------

The gate Program ends with the commit of :ref:`The commit <pipeline-commit>`, over the ``GateSequence``:

- **Masked regeneration is exact.** ``generate(advance=True, mask=m)`` holds the
  ``GateSequence`` rows of every env with ``m[e] == 0`` before the stages and restores
  them after, bit for bit (``position``, ``tangent``, ``forward``, ``left``, ``right``,
  ``orientation``, ``half_size``, ``count``, ``valid``, ``fail_mask``, every attribute;
  ``generation`` is written for the masked envs only).
- **Invalid envs.** ``BatchSpec.on_invalid`` applies: explicit ``"keep"`` leaves an
  :term:`invalid env`'s computed gates and
  ``valid = 0``. The default, ``"poison"``, leaves it with ``count = 0``, every per-gate field NaN
  (``orientation`` and ``half_size`` included), every attribute its padding and
  ``valid = 0``, keeping ``fail_mask``;
  ``CheckpointSet.from_gates`` and ``ProgressTracker`` then take their empty-course path
  for that env. ``GateGenConfig`` and ``presets.gates`` expose the same policy;
  a saved ``/2`` batch missing the key also defaults to poison.
- **Cost.** As for tracks: no launch for an unmasked explicit-keep generation, one for a derived
  ``valid`` (a check after ``validity.gate_default``) or for poison, two for a masked
  generation, plus per exported attribute two for a masked generation and one for
  poison. The held copy costs 80 B per gate slot (plus each attribute's size) and 12 B
  per env, allocated once.

.. _pipeline-gate-launches:

Launches
--------

A default gate generation is a short, fixed launch sequence: the effective seed, the
parameter latches (:ref:`pipeline-latch`), the generator's kernels, the order (its
``gates.xy`` and a ``gates.count`` copy), the normalization for the generators that
prefer it, the sphere separation, the elevation, the lift, the tangents, the frames and
the validity kernel, the default poison commit, then the ``generation`` write. The normalizing
generators (``polar``, ``checkpoint``, ``voronoi``) normalize after the order stage, out
of place: ``_normalize_extent_to_sized_k`` reads the order's ``gates.xy`` and writes its
own version.

.. list-table::
   :header-rows: 1
   :widths: 64 36

   * - generation (CPU, launches per ``generate()``)
     - launches
   * - ``bezier`` / ``hull``, any ordering, flat, ``gate_solve_iters=8``
     - 17 (2 latches: ``size`` and ``gate_frames.gate_width``)
   * - ``polar`` / ``checkpoint``, flat, ``gate_solve_iters=8``
     - 19 (4 latches: the two ring knobs, ``normalize.size`` and
       ``gate_frames.gate_width``)
   * - ``voronoi``, flat, ``gate_solve_iters=8``
     - 20 (4 latches)
   * - masked (``advance=True, mask=m``)
     - two more: the epoch advance and the hold (the poison commit becomes masked)
   * - explicit ``on_invalid="keep"``
     - one fewer when the default validity check is last (no commit)
   * - a plugin check after ``validity.gate_default``
     - the check's own launches (and its parameter latch for dynamic params); the
       default poison commit also derives the final ``valid``

These counts include the default poison policy and were remeasured with the current
CPU pipeline. A non-flat elevation adds the ``gate_arclen`` launch and its latches (2 for ``uniform``,
3 for ``random_walk`` / ``noise``); ``gate_solve_iters=0`` drops the separation launch.

.. _pipeline-gate-plan-rules:

Planner rules for gate pipelines
--------------------------------

Besides the rules of :ref:`build() and the planner <pipeline-stages-build-and-the-planner>`, a gate pipeline raises ``PlanError``
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
- stages disagree on ``gate_radius``: ``gate_relax.spheres`` and
  ``validity.gate_default`` share one separation radius. Bar width belongs to
  ``gate_frames.default`` or the provider's ``half_size`` channel;
- a ``GateSequence`` field has no writer, or ``valid.fail`` does not start at
  ``validity.gate_default``.

.. _pipeline-legacy-gates:

Gate plugin construction
------------------------

Gate generators use Stages. :ref:`writing-a-gate-generator` gives a complete example,
including native anchors, supplied attributes, ordering and the conformance checks.
