.. _writing-a-gate-generator:

Writing a Gate Generator
========================

.. admonition:: Before you start

   - **What you write:** a ``Stage`` subclass with ``kind = "gate_generator"`` and
     ``output = "anchors"`` (optionally a ``Params`` class), or an ``order`` stage that
     sequences the gates, plus the ``register(reg)`` function of your package.
   - **What you need to know:** the track generator guide (:ref:`writing-a-generator`):
     ``Params``, the ``StageBuilder``, salts, capture classes and the hard rules of
     ``enqueue`` are the same for a gate generator, and only what differs is repeated
     here. What the gate pipeline does with your anchors: :ref:`gate-pipeline`.
   - **How you test it:** ``track_gen.testing.check_gate_generator`` for a generator,
     ``check_stage("order", ...)`` for an ordering (:ref:`testing-a-plugin`).

A **gate generator** writes plan-view gate *anchors*, a few points per :term:`env`. The
gate pipeline then orders them (an ``order`` :term:`stage`), sizes them when the
generator asks for it, spreads them apart (``gate_solve_iters``), applies the altitude
profile, lifts them to 3D, derives tangents and frames, and decides validity
(:ref:`pipeline-gates`). ``presets.gates(generator=...)`` (or
``GateGenConfig.generator``) selects one by name.

A gate generator is not the only way to get gates: ``presets.gates`` also places gates
on any track generator's centerline (arc-length stations, :ref:`pipeline-stations`), so
a plugin track generator makes gates with no code of its own.

.. _gate-generator-contract:

Quick start: a gate generator
-----------------------------

A gate generator is a ``Stage`` with ``kind="gate_generator"`` and ``output="anchors"``.
This one is built from ``track_gen.ops`` the way the built-in ``bezier`` and ``hull``
gate generators are: it draws a per-env gate count and that many jittered grid points.
It does not order or size them: it declares the order it prefers and asks the pipeline
to normalize the ordered gates to ``ops.constants.TARGET_EXTENT * scale``, as ``polar``,
``voronoi`` and ``checkpoint`` do. Its ``register`` is the one your package's entry
point names (:ref:`packaging-a-plugin`), next to any track generators:

.. snippet: group=scatter register=register

.. code-block:: python

   import warp as wp

   from track_gen import ops, plugin

   MIN_GATES, MAX_GATES = 6, 10


   class Scatter(plugin.Stage):
       kind = "gate_generator"
       name = "acme.scatter"
       output = "anchors"
       writes = {"xy": "anchors.xy", "count": "anchors.count"}
       excluded_orderings = frozenset({"raw"})        # an unordered cloud
       preferred = {"order": "ccw", "normalize": "extent"}

       def capacity(self, params, layout):
           return {"gates": MAX_GATES, "min_gates": MIN_GATES}

       def build(self, b, params):
           xy = b.write("xy", wp.vec2f, per="anchors")    # [E * MAX_GATES]
           count = b.write("count", wp.int32)             # [E]
           used = b.scratch("used", wp.int32, b.E * MAX_GATES)
           num_cells, cell = ops.grid_cells(0.1)          # a 5 x 5 grid
           salt = b.salt("points") & 0x7FFFFFFF           # int32 for the ops samplers

           def enqueue(ctx):
               ops.sample_counts(ctx, b.seed, salt, 101, MIN_GATES, MAX_GATES, count)
               ops.sample_grid_points(ctx, b.seed, salt, 102, MAX_GATES, num_cells, cell,
                                      1.0, used, xy)
           return enqueue


   def register(reg):
       reg.add(Scatter)

Select it by name. ``order=None`` (the default of ``presets.gates``, and
``gate_ordering=None`` of ``GateGenConfig``) picks its preferred ``"ccw"``:

.. snippet: group=scatter

.. code-block:: python

   from track_gen import GateGenerator
   from track_gen.pipeline import presets

   device = "cuda:0"  # or "cpu"
   gen = GateGenerator(presets.gates(num_envs=64, device=device, generator="acme.scatter"))
   gates = gen.generate()
   gates.count.numpy()       # 6 to 10 gates per env

.. snippet: after

   _count = gates.count.numpy()
   _valid = gates.valid.numpy() != 0
   assert ((_count[_valid] >= MIN_GATES) & (_count[_valid] <= MAX_GATES)).all()
   assert (_count[~_valid] == 0).all()
   assert _valid.any()

Because it excludes ``"raw"`` instead of whitelisting ``"ccw"``, it also runs with
``"random_pairs"`` and with any order a plugin registers (`Writing an ordering`_).
``capacity()`` gives the most and the fewest anchors per env: it is checked against the
pipeline's ``max_gates`` and ``min_gates`` (:ref:`gate-capacity-config`).

.. _gate-generator-stage:

The anchors stage contract
--------------------------

A gate generator writes two :term:`channels <channel>`, ``anchors.xy`` (``[E * A]``
``vec2f``, env ``e``'s anchors at ``e * A + i``) and ``anchors.count`` (``[E]``
``int32``), and may write per-anchor attributes it ``provides``. It does not order its
anchors (an ``order`` stage does), size them to the course extent
(``normalize.extent`` does, when it declares ``preferred={"normalize": "extent"}``; that
stage's dynamic ``size``, key ``normalize.size``, is then the pipeline's per-env course
:term:`size`), or compute altitudes, tangents and frames. A gate generator that sizes its
own anchors instead declares a ``size`` field by the convention of
:ref:`generator-size-convention` (key ``generator.size``, as the ``bezier`` / ``hull``
gate generators do).

This one places ``gates`` anchors on a circle with a random phase per env, and provides
their altitude (a wave that climbs ``turns`` times per lap) and their analytic tangent,
so the pipeline keeps both instead of running an elevation stage and a finite-difference
tangent:

.. snippet: group=helix register=register

.. code-block:: python

   import dataclasses

   import warp as wp

   from track_gen import plugin
   from track_gen.ops import rng


   @wp.kernel
   def _helix_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, A: int, r: float,
                height: float, turns: int, xy: wp.array(dtype=wp.vec2f),
                z: wp.array(dtype=wp.float32), tangent: wp.array(dtype=wp.vec3f)):
       t = wp.tid()                         # one thread per anchor slot: t = e * A + i
       e = t // A
       i = t - e * A
       state = rng.init(seed[e], salt)
       phase = 6.2831853 * wp.randf(state)  # the same draw for every anchor of e
       s = 6.2831853 * float(i) / float(A)  # lap parameter in [0, 2*pi)
       u = s + phase
       w = float(turns) * s
       xy[t] = wp.vec2f(r * wp.cos(u), r * wp.sin(u))
       z[t] = height * wp.sin(w)
       # d(position)/ds; the frames stage normalizes it
       tangent[t] = wp.vec3f(-r * wp.sin(u), r * wp.cos(u),
                             height * float(turns) * wp.cos(w))


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class HelixParams(plugin.Params):
       gates: int = plugin.static(16, lo=4, doc="gates per lap")
       radius: float = plugin.static(0.8, lo=0.0, doc="loop radius, times scale")
       height: float = plugin.static(0.3, lo=0.0, doc="altitude amplitude")
       turns: int = plugin.static(3, lo=1, doc="climbs per lap")


   class Helix(plugin.Stage):
       kind = "gate_generator"
       name = "acme.helix"
       output = "anchors"
       Params = HelixParams
       writes = {"xy": "anchors.xy", "count": "anchors.count",
                 "z": "anchors.z", "tangent": "anchors.tangent"}
       provides = frozenset({"z", "tangent"})
       orderings = frozenset({"raw"})   # provided attributes travel by anchor index
       preferred = {"order": "raw"}

       def capacity(self, params, layout):
           return {"gates": params.gates, "min_gates": params.gates}

       def build(self, b, params):
           A = int(params.gates)
           xy = b.write("xy", wp.vec2f, per="anchors")         # [E * A]
           count = b.write("count", wp.int32)                  # [E]
           z = b.write("z", wp.float32, per="anchors")
           tangent = b.write("tangent", wp.vec3f, per="anchors")
           salt = b.salt("phase")
           r = float(params.radius) * float(b.scale)
           height, turns = float(params.height), int(params.turns)

           def enqueue(ctx):
               ctx.fill(count, A)                              # every env: A anchors
               ctx.launch(_helix_k, b.E * A,
                          [b.seed, salt, A, r, height, turns, xy, z, tangent])
           return enqueue


   def register(reg):
       reg.add(Helix)

Select it through the :term:`preset`, which reads the generator's declaration to build
the stage list:

.. snippet: group=helix

.. code-block:: python

   from track_gen import GateGenConfig, GateGenerator
   from track_gen.pipeline import presets

   spec = presets.gates(num_envs=64, device="cuda:0", generator="acme.helix")
   spec.ids
   # ('generator', 'order', 'carry_z', 'carry_tangent', 'gate_relax', 'lift',
   #  'gate_frames', 'validity')
   gates = GateGenerator(spec).generate()

   # The flat config too: gate_ordering=None (its default) is the generator's
   # default order, "raw" here.
   GateGenerator(GateGenConfig(num_envs=64, device="cuda:0", generator="acme.helix"))

.. snippet: after

   assert spec.ids == ("generator", "order", "carry_z", "carry_tangent", "gate_relax",
                       "lift", "gate_frames", "validity")
   assert gates.valid.numpy().all()

``gate_arclen``, ``elevation`` and ``tangents`` are gone: ``carry.z`` and
``carry.tangent`` copy the provided values to the gate slots, so
``GateSequence.position`` has the helix's altitude bit for bit and
``GateSequence.tangent`` its normalized tangent. The sphere separation (``gate_relax``)
still runs and may move a gate's xy; these gates are far apart, so it does not.

The contract:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute or method
     - Meaning
   * - ``kind``, ``output``
     - ``"gate_generator"`` and ``"anchors"``. A ``kind="generator"`` stage with
       ``output="anchors"`` is refused (declare ``kind="gate_generator"``).
   * - ``name``
     - The ``presets.gates(generator=...)`` (or ``GateGenConfig.generator``) string.
       Dotted for third parties (:ref:`plugin-names`).
   * - ``reads``
     - Empty: a generator reads no channel; its seed is ``b.seed``.
   * - ``writes``
     - ``anchors.xy`` and ``anchors.count`` (both required), plus ``anchors.z``,
       ``anchors.tangent``, ``anchors.half_size`` and ``anchors.roll`` for the
       attributes it provides, and its own per-anchor attributes
       ``anchors.<package>.<name>`` (``float32``, ``int32``, ``vec2f`` or ``vec3f``;
       ``export=True`` makes one a ``GateSequence.attributes`` entry, see
       :ref:`pipeline-gate-plugin-channels`); no other channel.
   * - ``provides``
     - Exactly the attributes it writes, a subset of ``{"z", "tangent", "half_size",
       "roll"}``; ``half_size`` and ``roll`` come together (write ``roll = 0`` for
       upright gates). See :ref:`pipeline-provides` for what the pipeline drops and
       what reaches the ``GateSequence``.
   * - ``orderings``
     - A whitelist (``frozenset``) of the order names it accepts, or ``None``. A
       generator that provides attributes, or writes ``anchors.<package>.<name>``
       channels, must declare ``frozenset({"raw"})``: the attributes are carried by
       anchor index (``carry.<attr>``, ``carry.attributes``), and only ``order.raw``
       keeps it.
   * - ``excluded_orderings``
     - With ``orderings=None``: the orders it refuses, every other registered order
       being accepted (third-party ones included). The built-ins use it: ``bezier`` and
       ``hull`` exclude ``raw``, ``polar``, ``checkpoint`` and ``voronoi`` exclude
       ``random_pairs``. Declare one of the two, not both.
   * - ``preferred``
     - ``{"order": name, "normalize": name}``, both optional: the default order of
       ``presets.gates`` (it must be accepted), and ``"extent"`` to have the pipeline
       scale the ordered gates to ``TARGET_EXTENT * scale`` as ``polar`` does.
   * - ``capacity(params, layout)``
     - ``{"gates": A, "min_gates": k}``: ``A >= 1`` anchors per env at most (the row
       stride of ``anchors.*``), ``k`` in ``[0, A]`` at least (default ``0``). The
       planner requires ``A <= max_gates`` and the validity stage's ``min_gates <= A``.
   * - ``build(b, params)``
     - As for any stage. ``b.layout.stride`` is ``max_gates``; size the anchors with
       ``per="anchors"`` (``A``), never with the layout. Write ``count[e]`` and the
       first ``count[e]`` anchors of every env on every call; the slots after them are
       never read (write them deterministically, NaN or not, as the built-in samplers
       do).

Everything else is the generator contract of :ref:`generator-stage-contract`:
:term:`Params` (static and dynamic), ``capture`` and ``capture_class``, salts, the
:ref:`hard rules <generator-hard-rules>` of ``enqueue``, per-env determinism.
``registry.add`` raises ``TypeError`` or ``ValueError`` (quarantine under ``warn``) for a
declaration that breaks the table, for example ``Stage Helix provides per-anchor
attributes ['tangent', 'z']: they are carried to the gate slots by index, so orderings
must be frozenset({'raw'})``.

The anchors keep the generator's own order, and the pipeline orders them: an unordered
cloud (as ``bezier`` emits) excludes ``raw``; a loop in course order (as ``polar``
emits) excludes ``random_pairs`` and is fine with ``raw`` or ``ccw``. Size the anchors
as you like: with ``preferred["normalize"] = "extent"`` the pipeline centres them and
scales the longest bounding-box side to ``ops.constants.TARGET_EXTENT * scale``; without
it, they are used as written (``bezier`` and ``hull`` keep their grid's size).

.. _writing-an-ordering:

Writing an ordering
-------------------

An ``order`` stage decides the sequence in which the gates are flown. It reads the
anchors (``A``-strided) and writes the first ``m`` gate slots (``G``-strided) of
``gates.xy``, NaN after them, where ``m = min(count[e], A, G)``, and ``gates.count``,
normally a copy of ``anchors.count``. This one flies a generator's own order backwards:

.. snippet: group=reverse requires=contributing/writing-a-gate-generator#helix register=register

.. code-block:: python

   @wp.kernel
   def _reverse_k(src: wp.array(dtype=wp.vec2f), A: int, count: wp.array(dtype=wp.int32),
                  G: int, dst: wp.array(dtype=wp.vec2f)):
       t = wp.tid()                         # one thread per gate slot: t = e * G + i
       e = t // G
       i = t - e * G
       m = wp.clamp(count[e], 0, wp.min(A, G))
       if i < m:
           dst[t] = src[e * A + (m - 1 - i)]
       else:
           dst[t] = wp.vec2f(wp.nan, wp.nan)   # padding: every slot is written


   class Reverse(plugin.Stage):
       kind = "order"
       name = "acme.reverse"
       reads = {"anchors": "anchors.xy", "n": "anchors.count"}
       writes = {"xy": "gates.xy", "count": "gates.count"}

       def build(self, b, params):
           src, n = b.read("anchors"), b.read("n")
           dst = b.write("xy", wp.vec2f, per="gates")      # [E * G]
           count = b.write("count", wp.int32)              # [E]
           A, G = b.stride_of("anchors"), b.stride_of("xy")

           def enqueue(ctx):
               ctx.launch(_reverse_k, b.E * G, [src, A, n, G, dst])
               ctx.copy(count, n)
           return enqueue


   def register(reg):
       reg.add(Reverse)

The read and write ports have different names: ``anchors.xy`` and ``gates.xy`` are two
channels, and a port in both maps must name one channel. ``A`` and ``G`` come from
``b.stride_of``: the anchors are written at the generator's own stride, the gate slots at
``max_gates``.

Select it by name, with the preset or the flat config:

.. snippet: group=reverse

.. code-block:: python

   presets.gates(num_envs=64, device="cuda:0", generator="polar", order="acme.reverse")
   GateGenerator(GateGenConfig(num_envs=64, device="cuda:0", generator="polar",
                               gate_ordering="acme.reverse"))

Every built-in gate generator accepts a third-party order: they declare
``excluded_orderings``, not a whitelist. A generator with a whitelist (``orderings``, as
one that provides attributes) accepts only the orders it lists, and the planner refuses
the others with ``gate generator 'acme.helix' does not support gate_ordering
'acme.reverse'; supported: ['raw'] (its preferred order is 'raw')``. An order that needs
randomness reads the ``seed`` channel (``reads = {..., "seed": "seed"}``) and keys its
stream on ``b.salt(<stream>)`` (:ref:`stage-salts`); the built-in ``order.random_pairs``
uses the fixed namespace ``order.random_pairs/pairs`` and
``ops.rng.init_at(seed, salt, pair_index)`` (:ref:`ops-rng-keying`). The registry and
the built-in orders are described in :ref:`pipeline-ordering-registry`.

Gate fail bits
--------------

A gate generator sets no :term:`fail bits`: it writes ``anchors`` and their count, and
the gate pipeline's ``validity.gate_default`` judges the finished gates, owning bits 0 to
5: ``count`` (fewer than ``min_gates`` or more than ``max_gates`` gates), ``finite``,
``tangent``, ``spacing`` (two gates closer than ``2 * gate_radius``), ``crossing`` (two
gate bars cross) and ``grade`` (:ref:`pipeline-gate-fail-bits`). So an env your
generator cannot fill well is rejected by writing fewer anchors than the pipeline's
``min_gates``; declare the fewest anchors you may write as ``capacity()["min_gates"]``
(the kit checks every env's count against ``capacity()``). A check of your own on the
finished gates is a validity stage that reads gate channels and owns bits from 16 on
(:ref:`stage-validity-check`, "On gates").

Gate stages in a pipeline
--------------------------

Register a ``gate_generator`` Stage with ``output="anchors"``. Write anchor positions
and real counts, then declare the orderings it accepts. A generator that provides its
own gate poses or attributes uses index-preserving raw order and declares those provided
channels; the downstream pipeline carries them into the result. See
:ref:`gate-generator-stage` for the complete contract.

Testing a gate generator
------------------------

``check_gate_generator`` runs the conformance checks on the ``GateSequence`` and checks
that every env's ``anchors.count`` lies within ``capacity()``; an ordering is checked as
a pipeline stage, inside the default gate pipeline (:ref:`kit-check-stage`):

.. snippet: requires=contributing/writing-a-gate-generator#scatter,contributing/writing-a-gate-generator#reverse

.. code-block:: python

   from track_gen.testing import check_gate_generator, check_stage

   check_gate_generator("acme.scatter").raise_for_failures()
   check_gate_generator("acme.helix").raise_for_failures()
   check_stage("order", "acme.reverse").raise_for_failures()

:ref:`testing-a-plugin` lists the checks and what each failure usually means.
