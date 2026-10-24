Writing a Component
===================

A **component** is a runtime consumer of a generator's result: what the runtime utilities
(:doc:`/utilities/overview`) are, packaged so that a composition root can build it, wire
it and drive it without knowing what it is. It declares what it reads, allocates
everything once when it is attached (cold), and runs three hot methods: ``on_generate``
after each generation, ``step`` once per sim step and ``reset`` on respawns. Each hot
method takes a :class:`~track_gen.exec.LaunchContext`, launches through it only, and
allocates and synchronizes nothing, so it records into your CUDA graph as-is.

``track_gen.tools`` holds the base class (``Component``), what ``attach`` is given
(``ComponentContext``), the declaration of a bound input (``InputSpec``) and ten adapters,
one per built-in utility (`The built-in adapters`_). In 0.2.0.dev6 ``Course`` builds its
sub-tools as these adapters internally; a ``Course`` that takes your components
(``Course(generator, components)``) is roadmap PR15. Until then you attach and drive a
component yourself (`Driving it by hand`_), and ``track_gen.testing.check_component``
checks it against the contract on this page (`Checking it`_).

A component imports only public modules: ``track_gen.tools``, ``track_gen.results``,
``track_gen.exec``, the utility modules (``track_gen.localize``, ...) and Warp.

Quick start: a curvature lookahead
----------------------------------

A controller wants to know how sharp the road is a few points ahead of the car, and how
far that point is. ``CurvatureLookahead`` computes the centerline curvature once per
generation (``localize.curvature`` into a buffer it owns) and, every step, reads it
``ahead`` points past the segment a ``Localizer`` found for the agent:

.. code-block:: python

   # acme_tools/lookahead.py
   import dataclasses
   import types

   import warp as wp

   from track_gen.localize import curvature
   from track_gen.results import HasCenterline, HasValidity
   from track_gen.tools import Component, InputSpec


   @wp.kernel
   def _lookahead_k(kappa: wp.array(dtype=wp.float32), center: wp.array(dtype=wp.vec3f),
                    count: wp.array(dtype=wp.int32), S: int, ahead: int,
                    segment: wp.array(dtype=wp.int32), position: wp.array(dtype=wp.vec3f),
                    out_kappa: wp.array(dtype=wp.float32),
                    out_distance: wp.array(dtype=wp.float32)):
       e = wp.tid()
       m = count[e]
       if m > S:
           m = S
       seg = segment[e]
       if seg < 0 or m < 3:                 # no frame (invalid env, NaN position): NaN
           out_kappa[e] = wp.nan
           out_distance[e] = wp.nan
           return
       i = (seg + ahead) % m
       out_kappa[e] = kappa[e * S + i]
       out_distance[e] = wp.length(center[e * S + i] - position[e])


   @dataclasses.dataclass
   class Lookahead:
       kappa: wp.array                      # [E] float32, 1 / track units
       distance: wp.array                   # [E] float32, agent to that point


   class CurvatureLookahead(Component):
       """Centerline curvature ``ahead`` points past the agent, per env."""

       requires = (HasCenterline,)          # center, count read (arclen, length come with it)
       optional = (HasValidity,)            # valid: read when the source has it
       inputs = types.MappingProxyType({"position": InputSpec(wp.vec3f, doc="the agent")})

       def __init__(self, ahead=6, *, window=2, localizer="localizer", name="lookahead"):
           self.ahead, self.window, self.name = int(ahead), int(window), name
           self.localizer = localizer
           self.after = (localizer,)        # reads that component's TrackFrame

       def attach(self, source, cc):
           lay = self.layout                # the source's Layout, set by cc.add
           E, S, dev = lay.num_envs, lay.stride, lay.device
           self._source = source
           self._center, self._count = source.center, source.count   # read once, here
           self._frame = cc.component(self.localizer).result    # public: a TrackFrame
           self._kappa = wp.zeros(E * S, dtype=wp.float32, device=dev)
           self._scratch = wp.zeros(E * S, dtype=wp.float32, device=dev)
           self.result = Lookahead(wp.zeros(E, dtype=wp.float32, device=dev),
                                   wp.zeros(E, dtype=wp.float32, device=dev))
           self._position = None
           self.on_generate(cc.ctx, None)   # the current batch's curvature (cold)
           wp.load_module(__name__, device=dev)   # step() can be recorded first

       def on_bind(self, position):
           self._position = position        # checked by bind(); pinned from here on

       def on_generate(self, ctx, mask):
           curvature(self._source, self.window, ctx,
                     out=self._kappa, scratch=self._scratch)

       def step(self, ctx):
           lay, r = self.layout, self.result
           ctx.launch(_lookahead_k, lay.num_envs,
                      [self._kappa, self._center, self._count, lay.stride,
                       self.ahead, self._frame.segment, self._position,
                       r.kappa, r.distance])
           return r

It overrides ``attach``, ``on_bind``, ``on_generate`` and ``step``. It keeps no per-env
state between steps (``stateful`` stays ``False``, so ``reset`` is the inherited no-op).
Its one input, ``position``, is the same array the ``Localizer`` reads: a composition
root binds one array to every component that declares the name. ``on_generate``
recomputes the whole batch, which leaves the rows of an unmasked env unchanged after a
masked regeneration, as the contract requires. TrackGen's own suite runs this class
(``tests/_component_toys.py``) through every check of ``check_component``.

Driving it by hand
------------------

``ComponentContext.for_producer`` reads the generator's layout and its static capacities;
``add`` checks each component's declaration against what it reads, then attaches it:

.. code-block:: python

   import warp as wp

   from track_gen import TrackGenConfig, TrackGenerator
   from track_gen.exec import LaunchContext
   from track_gen.tools import ComponentContext, Localizer

   from acme_tools.lookahead import CurvatureLookahead

   E, device = 64, "cuda:0"
   gen = TrackGenerator(TrackGenConfig(num_envs=E, device=device), seeds=0)
   track = gen.generate()

   cc = ComponentContext.for_producer(gen)     # layout + capacities, from the plan
   loc = cc.add(Localizer(), track)            # name "localizer"
   look = cc.add(CurvatureLookahead(ahead=6), track)

   pos = wp.zeros(E, dtype=wp.vec3f, device=device)   # your sim's stable buffer
   for c in (loc, look):
       c.bind(position=pos)
   done = wp.zeros(E, dtype=wp.int32, device=device)  # your per-env done mask

   ctx = LaunchContext.for_call(device)
   with wp.ScopedCapture(device=device) as cap:
       sim_step(pos)                           # your physics
       loc.step(ctx)                           # upstream first
       res = look.step(ctx)                    # res.kappa, res.distance: [E] float32
       update_done(res, done)                  # your kernel
       gen.generate(advance=True, mask=done)   # new courses where done[e] != 0
       for c in (loc, look):
           c.on_generate(ctx, done)            # refresh, in attach order
       loc.reset(ctx, done)                    # stateful components: reset the masked envs
   step = cap.graph

The order is the composition root's job: ``on_generate`` and ``step`` in attach order
(upstream first), ``reset`` for the stateful components after a regeneration and on
respawns. A component never calls another one.

Checking it
-----------

``check_component`` builds a producer, attaches the chain through
``ComponentContext.for_producer`` (the upstream factories first, in order), and runs every
check of :ref:`kit-check-component` on ``cpu``, and on ``cuda:0`` when Warp sees one:

.. code-block:: python

   from track_gen.testing import check_component
   from track_gen.tools import Localizer

   report = check_component(lambda cc: CurvatureLookahead(ahead=6),
                            upstream=[lambda cc: Localizer()])
   report.raise_for_failures()

The factories take the ``ComponentContext``, so a component that owns device arrays from
construction can build them for the kit's device and ``num_envs``. In pytest, the opt-in
plugin's ``track_gen_check_component`` fixture runs the same call and fails the test
listing every failed check.

The base class
--------------

``Component`` is a plain base class with immutable class-level defaults (not a
``typing.Protocol``: the library duck-types a component and never ``isinstance``-checks
one). Set the declaration as class attributes, or as instance attributes in
``__init__`` when it depends on the arguments (as the adapters' ``inputs`` do):

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Attribute
     - Default
     - Meaning
   * - ``name``
     - ``"component"``
     - Unique within one ``ComponentContext`` (a ``Course``). The built-in adapters
       default to the ``Course`` attribute they back (``"collision"``,
       ``"checkpoints"``, ``"progress"``, ``"localizer"``, ``"heightfield"``,
       ``"centerline"``).
   * - ``source``
     - ``None``
     - The read target: ``None`` reads the course result (the generator's ``Track`` or
       ``GateSequence``), a name reads that component's ``result`` (``Progress`` reads
       ``"checkpoints"``). The named component is a dependency, like ``after``.
   * - ``requires``
     - ``()``
     - Protocols of ``track_gen.results`` whose members must all be present: strict, a
       missing member raises ``TypeError`` before ``attach``.
   * - ``optional``
     - ``()``
     - Protocols whose members are read when present: checked like required ones when
       they are there (shape, dtype, device). When the read target has no layout and
       nothing is required, the layout is the one its present members give
       (``num_envs`` from ``count`` or a per-env member, the stride from a per-point
       member: ``M`` on a ``CheckpointSet``), the context's layout filling what they do
       not. ``None`` / missing gives the documented fallback (`Read targets`_).
   * - ``topology``
     - ``"closed"``
     - ``"closed"`` refuses a read target whose ``layout.closed`` is ``False``;
       ``"any"`` accepts both.
   * - ``stateful``
     - ``False``
     - ``True`` when the component keeps per-env runtime state between steps that
       ``reset(ctx, mask)`` clears.
   * - ``inputs``
     - empty mapping
     - ``{name: InputSpec}``: the caller arrays ``bind`` takes (`Inputs`_).
   * - ``after``
     - ``()``
     - Names of other components it reads (they are attached first).
   * - ``layout``
     - ``None``
     - Set by ``ComponentContext.add`` before ``attach``: the read target's
       ``Layout``; the context's when nothing is declared; with only ``optional`` and a
       read target without a layout, the layout its present members give, with the
       context's layout filling the rest (``Layout(E, M)`` on a ``CheckpointSet``). A
       read target of the producer
       (``source=None``) must hold the context's batch (same ``num_envs`` and device).
   * - ``result``
     - ``None``
     - Set by ``attach``: the stable object ``step`` returns (its arrays reachable
       through fields, mapping values, tuple items or attributes).

The built-in adapters also expose the legacy utility they wrap as ``utility``: an
adapter attribute, not part of the contract.

The lifecycle methods:

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Method
     - When
     - Contract
   * - ``attach(source, cc)``
     - once, cold, outside any capture, through ``cc.add``
     - ``source`` is the resolved read target, already checked against ``requires`` and
       ``optional``; ``self.layout`` is set. Allocates every buffer and sets
       ``self.result`` from ``self.layout`` and ``cc.capacity``; every cold launch goes
       through ``cc.ctx``. Ends with one ``on_generate(cc.ctx, None)`` (the first
       sample, bake or refresh), then initialises per-env state from it, and loads the
       Warp module of every kernel its hot methods launch (``wp.load_module``), so
       ``step`` can be recorded into a graph with no eager call before it. The default
       raises ``NotImplementedError``.
   * - ``bind(**arrays)``
     - cold, before recording ``step``
     - Inherited, never overridden: routes defaults (``InputSpec.default``), raises
       ``TypeError`` naming a missing required or an unknown name, checks each array
       (``[E * slots]`` of ``dtype`` on the layout's device, contiguous), then calls
       ``on_bind(**arrays)``.
   * - ``on_bind(**arrays)``
     - from ``bind``
     - Your hook: keep the (checked) arrays; pointers are pinned from here on. Default:
       nothing.
   * - ``on_generate(ctx, mask)``
     - after every generation of the source; ``mask`` its ``[E]`` int32 mask or
       ``None``
     - Hot. The rows and state of envs with ``mask[e] == 0`` stay bitwise unchanged (a
       whole-batch recompute qualifies). Default: nothing.
   * - ``step(ctx)``
     - every sim step
     - Hot. Reads the bound inputs and returns ``self.result`` (the same instance, the
       same arrays, every call). Default: returns ``self.result``.
   * - ``reset(ctx, mask)``
     - after a regeneration and on respawns
     - Hot. Clears per-env runtime state where ``mask[e] != 0`` and nothing else. A
       stateless component emits nothing (the default).

**Hot-path rules** (the same as a stage's ``enqueue``, :ref:`generator-hard-rules`): launch,
copy and fill through ``ctx`` only (``ctx.launch``, ``ctx.copy``, ``ctx.fill``, or a
library call given ``ctx=``); never allocate, synchronize, read device data back, or
branch in Python on device data or on ``ctx.capturing``; never change Python state that
changes a later emission. **Stable results:** ``result`` and every array reachable from
it are allocated in ``attach`` (or aliased from the read target) and never replaced.

Read targets
------------

``ComponentContext.add(component, source)`` resolves the read target (``source`` when
``component.source is None``, else ``cc.component(component.source).result``), and checks
it with :func:`~track_gen.results.check_source`: every member of ``requires``, plus the
members of ``optional`` that are present, with ``optional=`` naming the latter
(:ref:`results-check-source`). Error messages name the component:

.. code-block:: text

   TypeError: lookahead needs a HasCenterline source (center, arclen, length, count;
     optional: valid): SimpleNamespace has no 'arclen'

``attach`` then reads those members (``source.center``, ...), once: what it reads is what
was checked. Declare every member you read. ``check_component``'s
``requires.complete`` attaches your component to a proxy that exposes only the declared
members and expects the same outputs; a read outside the declaration fails there.

**Optional members and their fallbacks.** The library's components read two optional
capabilities:

- ``HasValidity`` (``valid``, ``[E]`` int32): absent, every env is valid.
- ``HasGeneration`` (``generation``, ``[E]`` int32): absent, the component never
  self-resets (`Per-generation state and self-reset`_).

A kernel that reads an optional member takes it with an ``int`` flag (``has_valid``) and,
when it is absent, any ``[E]`` int32 array of the source as a placeholder (the library
passes ``count``): no allocation for the missing member, the same launch either way.

**Topology.** A component that reads a result with closed loops declares
``topology = "closed"`` (the default); one with no read target (``requires = ()``, e.g.
``DiscCollision`` on raw disc arrays) declares ``"any"``.

``ComponentContext``
--------------------

.. code-block:: python

   ComponentContext(layout, capacity=None, *, ctx=None)
   ComponentContext.for_producer(producer, *, ctx=None)

- ``layout``: the producer's :class:`~track_gen.results.Layout`.
- ``capacity``: a read-only mapping of static capacities (below).
- ``ctx``: the cold :class:`~track_gen.exec.LaunchContext` of ``attach``. ``None`` means
  ``LaunchContext.for_call(layout.device)``: an enclosing ``ctx.scope()``, else Warp's
  current stream. Attach after the producer's work has completed, or pass the producer's
  stream.
- ``component(name)``: an attached component (``KeyError`` listing the attached ones).
- ``add(component, source)``: (1) every name in ``after`` and ``source`` is attached and
  ``name`` is free (``ValueError``); (2) an input name an attached component also
  declares has the same ``(dtype, slots)`` (``ValueError`` naming both); (3) resolves and
  checks the read target (a producer result must hold the context's batch: same
  ``num_envs`` and device, else ``ValueError``), sets ``component.layout``; (4) calls
  ``attach``; (5) registers the component and returns it.
- ``for_producer(producer)``: a ``TrackGenerator``, a ``GateGenerator``, a ``Program`` or
  a ``Course`` (its generator). It reads the producer's ``layout``, ``spec`` and
  ``params``, never its device arrays. A result (``gen.generate()``'s ``Track``) has no
  plan and raises ``TypeError``: for a hand-built source, construct
  ``ComponentContext(layout, {...})`` yourself.

**Capacities.** One key in 0.x, ``max_loop_length``: an upper bound, with measured
headroom, on the 3D length of the centerline and of either boundary loop of every valid
env. ``for_producer`` derives it from the plan, never from a batch:

.. code-block:: text

   max_loop_length = 1.5 * N_max * spacing + dZ

- ``N_max * spacing`` bounds the plan-view centerline of every env that is not truncated
  (a longer one is truncated and fails validity). The ``1.5`` covers the relax growth
  and a constant-width boundary's extra ``2 pi * half_width``: **measured, not proven**
  (at most 0.90 of ``N_max * spacing`` just below truncation; 0.21 to 0.35 at size 1).
- ``dZ`` bounds the elevation's total variation from the elevation knobs' declared
  ``bounds()`` (never their current values, which are dynamic): ``0`` for
  ``elevation.flat``; ``K * (z_max_hi - z_min_lo)`` for ``uniform`` / ``random_walk``
  (``K = z_control_points``); for ``noise`` the smaller of ``2 * H * (z_max_hi -
  z_min_lo)`` and ``4 * A_hi * H / (1 + 1/2 + ... + 1/H)`` (``H = z_noise_harmonics``,
  ``A_hi`` the amplitude's upper bound); with the built-in ``validity.default``'s
  ``z_valid_grade = g > 0``, ``g * 1.5 * N_max * spacing`` (a plugin check's field of
  that name does not count). The smallest available term is used.
- The key is **omitted** when no ``dZ`` term is finite (every elevated
  ``TrackGenConfig`` today: ``z_min`` / ``z_max`` declare no bounds, and a randomizer's
  range does not give them one: its ``lo`` / ``hi`` are dynamic), when the generator
  provides its altitudes (``provides`` ``z``: no elevation stage, ``lift.elevated``) and
  no grade is declared, when no ``width.constant`` follows the resample (another width
  stage, or a generator that provides ``half_width``: unbounded per point), when a
  plugin ``relax`` / ``width`` / ``offset`` / ``elevation`` / ``lift`` / ``resample``
  stage follows the constant-spacing resample, and without that resample (gate
  pipelines, ``external``).

At the ``TrackGenConfig`` defaults (``N_max = 384``, ``spacing = 0.06``, flat) it is
``34.56``. Treat it as a sizing bound, not a guarantee: clamp each env to your capacity and
report what you clipped (the built-in samplers' ``truncated``). A component that needs the
key and does not find it raises ``ValueError`` naming its explicit capacity argument
(``tools.Checkpoints(spacing, max_checkpoints=...)``).

Inputs
------

.. code-block:: python

   InputSpec(dtype, slots=1, required=True, default=None, doc="")

An input is a caller array the component reads in place every ``step``: ``[E * slots]``
elements of ``dtype`` (``wp.vec3f``, ``wp.quatf``, ``wp.vec2f``, ``wp.float32``, ...) on
the layout's device. ``default`` names another input used when this one is not given:
the collision adapters declare ``box_position`` with ``default="position"`` when they
have one box per env, so ``bind(position=p, ...)`` feeds both progress and collision (a
routed array's error names both: ``input box_position (routed from position) must have
shape ...``). ``required=False`` makes an input optional: ``bind`` without it calls
``on_bind`` with ``None`` for it, so the hook can take it as a plain parameter.
``doc`` is not part of the equality. The built-in names are ``position`` (``[E]``
``vec3f``, the agent) and ``box_position`` / ``orientation`` / ``half_extents`` (``[E *
max_boxes]`` ``vec3f`` / ``quatf`` / ``vec2f``); reuse them with the same meaning, since a
composition root routes one array to every component that declares a name and refuses two
declarations of one name with different ``(dtype, slots)``.

A component keeps its input arrays in ``on_bind`` (``CurvatureLookahead.on_bind`` above).
A graph recorded with ``step`` keeps reading the arrays bound at the time: rebinding
other arrays afterwards needs a new recording (`Rebinding after a capture`_).

Per-generation state and self-reset
-----------------------------------

Two kinds of state:

- **Per-generation data** (a resampled ring, a baked grid, a curvature profile) is a
  function of the current course. ``on_generate`` recomputes it; nothing else needs to.
- **Per-env runtime state** (the previous position, the next checkpoint, a warm-start
  segment) depends on the agent's history. It is wrong as soon as the env's course
  changes. Declare ``stateful = True`` and clear it in ``reset(ctx, mask)``.

A stateful component can also clear it on its own, on device: read the source's
``generation`` (``[E]`` int32, an optional ``HasGeneration`` member), keep a
``seen_generation`` ``[E]`` int32 of your own (a copy of ``generation`` made at the end of
``attach``), and at the top of the step kernel:

.. code-block:: python

   if has_gen != 0:
       gen = generation[e]
       if seen[e] != gen:                   # the env got a new course since its state was built
           seen[e] = gen
           # ... write exactly what reset() writes for env e ...

Writing exactly what ``reset`` writes, before the state is read, makes it idempotent: a
self-reset and an explicit ``reset`` give the same bytes, alone or together. ``reset`` does
not touch ``seen``. ``Progress`` and ``Localizer`` do this (:ref:`utilities-self-reset`).

``generation`` is the env's **epoch**, not a counter of new courses: an advancing
regeneration changes it for the envs it advances (masked or all); a regeneration without
``advance`` does not; a reseed changes it only when it zeroes a non-zero epoch
(``Course.generate(seeds=)``, ``SeedState.set``), not on a bare ``rng.set_seeds_warp``; and
``reset_epoch()`` followed by the same number of advances between two steps leaves it
equal. So a caller that reseeds or rolls epochs back must still call ``reset(mask)``. A
read target without ``generation`` (a hand-built source) never self-resets.

If your per-generation data comes from another component (a ring sampled by
``Checkpoints``), read the ``generation`` that component writes with its data, not the
course's: a ``step`` between the regeneration and the resample must not reset against the
old data. ``Checkpoints`` owns the ``CheckpointSet``'s ``valid`` / ``generation`` for that
reason.

Invalid envs
------------

**Rule:** for an env with ``valid[e] == 0``, every array the component owns holds exactly
what it holds for the same env poisoned (``count = 0``, NaN geometry, ``length = 0``,
``winding = 0``): an invalid env reads like a poisoned one, whatever the generator's
``on_invalid``. The library's table is in :ref:`utilities-invalid-envs`: NaN geometry,
``count = 0``, no hit, ``oob = 1``, frozen progress. Read ``valid`` as an optional member
and take your degenerate branch when it is ``0``, the same branch a poisoned env reaches
(``count < 3``, a NaN input). ``CurvatureLookahead`` gets it for free: ``curvature`` writes
NaN rows and the ``Localizer`` writes ``segment = -1`` for an invalid env, so its kernel
takes the NaN branch.

A component whose ``result`` only aliases its read target (``GateCheckpoints``, a
zero-copy view of the gates) owns nothing: its consumers apply the rule.
``check_component``'s ``invalid.sentinel`` fails half the envs of a batch through the
commit's own kernel twice, once keeping their geometry (``valid = 0``) and once poisoned,
and compares every owned array of your result over a trajectory.

.. _component-rebind:

Rebinding after a capture
-------------------------

A recorded graph keeps the pointers it was recorded with: rebinding new arrays afterwards
leaves it reading the old ones. The library's utilities with bound inputs, their
adapters and ``Course.bind`` therefore note, in their hot methods, when they ran under a
capture (a capturing stream, or a CPU capture on the device), and emit
``track_gen.exec.RebindAfterCaptureWarning`` (a ``UserWarning``) at the caller's line when
a bound array is replaced afterwards. Binding the same arrays again is silent. The
documented recipe "rebind, then record again" warns once too; filter the warning
(``warnings.simplefilter("ignore", RebindAfterCaptureWarning)``) when you re-record on
purpose. The probe is a host flag set once and changes no launch: the one exception to
"never change Python state in a hot method".

The built-in adapters
---------------------

Each adapter wraps one legacy utility (``adapter.utility``), which keeps every legacy name
and method; ``adapter.result`` is that utility's ``.result``.

.. list-table::
   :header-rows: 1
   :widths: 22 18 30 30

   * - Adapter (``track_gen.tools``)
     - Wraps
     - Reads: ``requires`` / ``optional``
     - Inputs, state
   * - ``BandCollision(method="segments", *, max_boxes=1, sdf_resolution=128, sdf_padding=None, size_hi=None, reference_resolution=128)``
     - ``CollisionChecker``
     - course: ``HasBand`` / none
     - ``box_position`` (default ``position`` at one box), ``orientation``,
       ``half_extents``; stateless; ``on_generate`` bakes (``sdf``)
   * - ``DiscCollision(radius, *, max_boxes=1, discs=None, count=None, valid=None)``
     - ``DiscChecker``
     - posts mode (``discs=None``): course, ``HasGateFrames`` / none; raw mode: no read
       target (``topology="any"``)
     - as ``BandCollision``; stateless; posts mode rebuilds the gate posts in
       ``on_generate``
   * - ``FrameCollision(radius, frame_thickness, frame_depth, *, window=4, progress="progress")``
     - ``FrameChecker``
     - course: ``HasGateFrames`` / none; ``after`` the progress component (its
       ``next_checkpoint``)
     - ``position``; stateless
   * - ``Props(spacing, *, boundary="outer", mode="points", max_props=None)``
     - ``PropSampler``
     - course: ``HasBand`` / none
     - none; stateless; ``on_generate`` samples
   * - ``Checkpoints(spacing, *, max_checkpoints=None)``
     - ``CheckpointSampler``
     - course: ``HasBand`` / ``HasGeneration``
     - none; stateless; ``on_generate`` samples
   * - ``GateCheckpoints()``
     - ``CheckpointSet.from_gates`` (zero copy; ``utility`` is ``None``)
     - course: ``HasGateFrames`` / ``HasGeneration``
     - none; stateless
   * - ``Progress(*, source="checkpoints")`` (``checkpoints=``: a 0.x alias)
     - ``ProgressTracker``
     - the ``source`` component: ``HasCheckpoints`` / ``HasValidity``,
       ``HasGeneration``
     - ``position``; stateful, self-resets
   * - ``Localizer(*, warm_window=None, source=None)``
     - ``TrackLocalizer``
     - course or ``source``'s result: ``HasCenterline`` / ``HasValidity``,
       ``HasGeneration``
     - ``position``; stateful (both modes), self-resets
   * - ``HeightFieldBake(resolution, *, padding=None, size_hi=None, reference_resolution=128)``
     - ``HeightFieldBaker``
     - course: ``HasBand`` / none
     - none; stateless; ``on_generate`` bakes
   * - ``CenterlineFromGates(samples_per_gate=8)``
     - ``CourseLine``
     - course: ``HasGates`` / ``HasGeneration``
     - none; stateless; ``on_generate`` refreshes; ``result`` is the ``Centerline``

Every adapter also takes ``name=``. The keywords follow one rule: ``source=`` names the
read target (``Progress``, ``Localizer``), and a dependency read through ``after`` is
named by its role (``FrameCollision(progress=...)``). ``Props`` and ``Checkpoints`` size
their ring from ``cc.capacity["max_loop_length"]`` when their capacity is ``None``
(``max(3, ceil(max_loop_length / spacing))``) and raise ``ValueError`` without the key.
``BandCollision(method="sdf")`` and ``HeightFieldBake`` take the largest course size they
will see (``size_hi``) and warn at ``attach`` when the grid is too coarse for it
(:ref:`collision-resolution-check`).

TrackGen's own suite runs ``check_component`` over every adapter (on CPU, and on
``cuda:0`` in its GPU lane).
