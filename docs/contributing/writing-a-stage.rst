Writing a Stage
===============

A **stage** is one step of the track pipeline or of the gate pipeline (see
:doc:`/how-it-works/pipeline-preview`). Besides the generators, six kinds of stage are
open to plugins:

- ``relax`` reshapes the resampled centerline (``cl.xy``), as the built-in XPBD solver
  does;
- ``width`` writes the per-point half-width (``cl.half_width``);
- ``elevation`` writes the per-point altitude (``cl.z``), and the per-gate altitude
  (``gates.z``) when it runs in a gate pipeline;
- ``validity`` runs a per-env check and sets a fail bit (``valid.fail``), on a track or
  on gates;
- ``order`` puts a gate generator's anchors in course order (``gates.xy``), see
  `Writing an ordering`_;
- ``stations`` places gate anchors on a track generator's centerline
  (``anchors.xy``).

A gate generator is a stage too, of kind ``gate_generator``: see `Writing a gate
generator (anchors)`_.

A stage is a ``Stage`` subclass, the protocol a generator Stage uses (see
:ref:`generator-stage-contract`), with read ports. It ships the way a generator ships:
in its own pip-installable package, found through the ``track_gen.plugins`` entry point,
registered by the same ``register(reg)`` function, which may add generators and stages
alike. It imports only public modules (``track_gen.plugin``, ``track_gen.exec``,
``track_gen.ops``), and ``track_gen.testing.check_stage`` checks it against the contract
on this page (see :ref:`kit-check-stage`).

This page assumes the :doc:`generator guide <writing-a-generator>`: Params, the
``StageBuilder``, salts, capture classes and the hard rules of ``enqueue`` are the same
for every stage, and only what differs is repeated here.

Quick start: a relax stage
--------------------------

Taubin smoothing alternates a shrinking Laplacian step (``lam > 0``) and an inflating one
(``mu < 0``), which smooths a closed polyline without shrinking it. It makes a relax
stage with one static parameter (the number of passes, which fixes the launch count) and
two dynamic ones (the step factors, per env, changeable at run time).

**1. Declare the entry point**, as for a generator (skip it if the package already has
one; one ``register`` adds everything):

.. code-block:: toml

   [project.entry-points."track_gen.plugins"]
   acme = "acme_tracks:register"

**2. Write the stage and register it** (``acme_tracks/taubin.py``):

.. code-block:: python

   import dataclasses

   import warp as wp

   from track_gen import plugin


   @wp.kernel
   def _laplace_k(src: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
                  f: wp.array(dtype=wp.float32), S: int, dst: wp.array(dtype=wp.vec2f)):
       t = wp.tid()                         # one thread per point slot: t = e * S + i
       e = t // S
       i = t - e * S
       n = count[e]
       if i >= n:
           dst[t] = src[t]                  # carry the NaN padding: write every slot
           return
       base = e * S
       x = src[t]
       prev = src[base + (i + n - 1) % n]   # closed loop: wrap around the real points
       nxt = src[base + (i + 1) % n]
       dst[t] = x + f[e] * (0.5 * (prev + nxt) - x)


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class TaubinParams(plugin.Params):
       passes: int = plugin.static(10, lo=1, doc="shrink/inflate pass pairs")
       lam: float = plugin.dynamic(0.5, lo=0.0, hi=1.0, doc="shrink factor")
       mu: float = plugin.dynamic(-0.53, lo=-1.0, hi=0.0, doc="inflate factor")


   class Taubin(plugin.Stage):
       kind = "relax"
       name = "acme.taubin"
       Params = TaubinParams
       reads = {"xy": "cl.xy", "count": "cl.count"}
       writes = {"xy": "cl.xy"}           # same channel, same port: a rewrite

       def build(self, b, params):
           x, count = b.read("xy"), b.read("count")   # the version before this stage
           out = b.write("xy", wp.vec2f, per="cl")    # a NEW version, [E * N_max]
           tmp = b.temp_like("xy")                    # ping-pong partner, same shape
           lam, mu = b.param("lam"), b.param("mu")    # [E] float32, read by pointer
           S = b.layout.stride
           steps = [f for _ in range(params.passes) for f in (lam, mu)]

           def enqueue(ctx):
               src = x
               for k, f in enumerate(steps):        # x -> tmp -> out -> tmp -> ... -> out
                   dst = tmp if k % 2 == 0 else out
                   ctx.launch(_laplace_k, b.E * S, [src, count, f, S, dst])
                   src = dst
           return enqueue


   def register(reg):
       reg.add(Taubin)          # the class, not an instance

The stage *rewrites* ``cl.xy``: it reads the channel and writes it through the same port.
``b.read("xy")`` is the version the previous stage wrote, and ``b.write("xy", ...)`` a new
buffer that every later stage reads (see :ref:`pipeline-inplace`). The stage never writes
the buffer it reads. The steps alternate between ``tmp`` and ``out`` and their number is
even, so the last one lands in ``out`` with no copy. ``passes`` is static: it fixes the
number of launches, which a recorded graph replays forever.

**3. Select it.** Through a preset, by name, with params as a mapping or a
``TaubinParams``:

.. code-block:: python

   from track_gen import PerEnvSeededRNG, TrackGenerator
   from track_gen.pipeline import presets

   rng = PerEnvSeededRNG(seeds=0, num_envs=1024, device="cuda:0")
   spec = presets.track(num_envs=1024, device="cuda:0",
                        relax=("acme.taubin", {"passes": 20}))
   gen = TrackGenerator(spec, rng)
   track = gen.generate()

   gen.params["relax"].set("lam", 0.6)     # every env, from the next generation on
   track = gen.generate(advance=True)

Or through the flat config, whose ``relax_solver`` names any registered relax stage (it
then takes ``TaubinParams.from_legacy(config)``, its defaults here):

.. code-block:: python

   from track_gen import TrackGenConfig

   gen = TrackGenerator(TrackGenConfig(num_envs=1024, device="cuda:0",
                                       relax_solver="acme.taubin"), rng)

The rest of the pipeline is unchanged: the uniform resample, frames, offsets, elevation,
validity and lift run on the smoothed loop, and ``gen.program.tap("cl.xy",
after="resample")`` still holds the loop before it.

This example shows the mechanics, not a replacement for XPBD. Taubin smoothing enforces
no separation between distant parts of the loop, so it fails the thickness check (fail
bit 3) where ``relax=None`` does: at the default ``half_width=0.1`` every ``bezier``
track does, and at ``half_width=0.03`` (with ``n_max=1024``) about a third pass. A relax
stage you ship must leave a loop the band can be inflated around; ``Track.fail_mask``
tells you which check it fails.

The open kinds
--------------

.. list-table::
   :header-rows: 1
   :widths: 13 22 20 25 20

   * - kind
     - slot (preset keyword)
     - writes
     - usual reads
     - flat-config shortcut
   * - ``relax``
     - ``relax``, between ``resample`` and ``uniform``
     - ``cl.xy`` (a rewrite)
     - ``cl.xy``, ``cl.count``
     - ``relax_solver``
   * - ``width``
     - ``width``, after ``uniform``
     - ``cl.half_width``
     - ``cl.count``, ``cl.xy``
     - none
   * - ``elevation``
     - ``elevation``, after ``offset`` (gates: after ``gate_relax``, and after
       ``gate_arclen`` for a non-flat profile)
     - ``cl.z`` (gates: ``gates.z``, rewired by the planner)
     - ``seed``, ``cl.count``, ``cl.arclen2``, ``cl.length2``
     - ``z_profile`` (both configs)
   * - ``validity``
     - ``checks``, after ``elevation`` (gates: after ``gate_frames``)
     - ``valid.fail``
     - ``valid.fail`` and anything written before it
     - none
   * - ``order``
     - ``order`` (gates), after the gate generator
     - ``gates.xy``, ``gates.count``
     - ``anchors.xy``, ``anchors.count``, ``seed``
     - ``GateGenConfig.gate_ordering``
   * - ``stations``
     - ``stations`` (gates), after a centerline generator
     - ``anchors.xy``, ``anchors.count``
     - ``raw.xy``, ``gen.valid`` (or ``cl.xy``, ``cl.count``, ``cl.truncated``)
     - none

A stage of the four track kinds may also write plugin channels, ``cl.<package>.<name>``
(gate plugin channels arrive with roadmap PR13). Each kind's built-ins are registered
under plain names (``relax:xpbd``, ``relax:none``, ``width:constant``,
``elevation:flat`` / ``uniform`` / ``random_walk`` / ``noise``, ``validity:default``,
``validity:gate_default``, ``order:raw`` / ``ccw`` / ``random_pairs``,
``stations:raw`` / ``relaxed``); ``registry.available("elevation")`` lists them with the
installed plugins.

.. _stage-ports-channels:

Ports and channels
------------------

``reads`` and ``writes`` map a **local port** (the name your ``build`` uses) to a
**channel** (the pipeline's name for a buffer). The channels are those of the catalogue
(see :doc:`/how-it-works/pipeline-preview`) and your own namespaced ones,
``cl.<package>.<name>``, such as ``cl.acme.rank`` (at least two dot-separated
identifiers after ``cl.``). A user can rewire a port to another channel of the same
dtype and size with ``StageRef(..., ports={"xy": "cl.acme.smoothed"})``.

``registry.add`` checks the declaration, and raises ``TypeError`` or ``ValueError`` (the
plugin is then quarantined under ``warn``) when:

- the class is an instance, abstract, or of a kind that is not open (``resample``,
  ``frames``, ``offset``, ``lift``: "not open to plugins");
- ``output`` is set (it is for generators), ``provides`` is not empty, ``topology`` is
  not ``"closed"`` or ``"any"``, ``api_version`` is above ``API_VERSION``, ``capture`` is
  not a ``CaptureClass``, or the Params class breaks the rules of
  :ref:`generator-params`;
- a channel is neither in the catalogue nor a namespaced plugin channel;
- two write ports name one channel, or one channel is read through one port and written
  through another (a rewrite reads and writes through **one** port), or a port in both
  maps names two channels;
- the stage writes a catalogue channel its kind may not write: a relax stage writes
  ``cl.xy`` only, a width stage ``cl.half_width``, an elevation stage ``cl.z`` (or
  ``gates.z``, the channel the planner rewires it to in a gate pipeline), a
  validity stage ``valid.fail`` (plus plugin channels for all four; ``seed`` never), an
  order stage ``gates.xy`` and ``gates.count``, a stations stage ``anchors.xy`` and
  ``anchors.count``. For example ``Stage Taubin (kind 'relax') writes 'cl.kappa' (port
  'k'): a relax stage may write ['cl.xy'] and plugin channels 'cl.<package>.<name>'``;
- a validity stage does not write ``valid.fail``, an order stage does not write both
  ``gates.xy`` and ``gates.count``, or a stations stage both ``anchors.xy`` and
  ``anchors.count``;
- a stage that is not a gate generator declares a non-empty ``preferred`` (a gate
  generator attribute: ``Stage X.preferred is read for gate generators only``).

A non-Stage spec (a ``GeneratorSpec``-like object) registered under one of these kinds
raises ``TypeError``: pipeline stages are ``Stage`` subclasses only.

The planner checks the rest when it builds a pipeline (each as a ``PlanError`` naming
the stage, the port and the channel): the rules above again after ``ports`` rewiring,
every read channel has an earlier writer (a width stage that reads ``cl.kappa`` must come
after ``frames``), and a second writer of a channel reads it (a rewrite). ``b.write``
checks that a later writer of a plugin channel keeps the dtype and ``per`` its first
writer gave it (``TypeError``). The rewrite model, with
its five rules, is :ref:`pipeline-inplace`; the one rule you write code for is:
**write your whole output into the buffer** ``b.write`` **returns, and never write a
buffer** ``b.read`` **returned**.

The ``StageBuilder`` in a pipeline stage
----------------------------------------

The builder is the one :ref:`stage-builder` describes, with these differences:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute or method
     - In a pipeline stage
   * - ``stage_id``
     - The slot id (``"relax"``, ``"elevation"``, or the id a ``StageRef`` gives). It
       keys ``gen.params``, the scratch names and the salts.
   * - ``layout.stride``
     - ``N_max`` in a track pipeline: a per-point channel is ``[E * N_max]``, env
       ``e``'s slots are ``e * N_max`` to ``e * N_max + N_max - 1``, and the first
       ``count[e]`` are real. In a gate pipeline, ``G = max_gates`` for a stage on the
       gate slots and ``N_max`` for one with a ``cl.*`` port (see
       :ref:`pipeline-gates`).
   * - ``stride_of(port)``, ``per_of(port)``
     - The row stride (``1`` for an ``[E]`` channel) and the ``per`` of the channel the
       port is wired to, after ``ports`` rewiring: size reads from them when a stage
       reads two domains (an order stage reads ``A``-strided anchors and writes
       ``G``-strided gates).
   * - ``seed``
     - The ``[E]`` effective seed, as for a generator. Declaring a ``seed`` read port
       (``reads = {"seed": "seed"}``) gives the same buffer.
   * - ``read(port)``
     - The current version of ``reads[port]``: written by an earlier stage, read-only
       for this one.
   * - ``write(port, dtype=None, *, per=None)``
     - A new buffer for ``writes[port]``, zero-initialised once, read by every later
       stage. ``per`` is ``"env"`` (``[E]``) or ``"cl"`` (``[E * N_max]``), and for the
       gate catalogue ``"anchors"`` (``[E * A]``) or ``"gates"`` (``[E * G]``). For a
       catalogue channel, or a plugin channel an earlier stage wrote, omit both (they
       are the channel's) or pass matching ones (``TypeError`` otherwise); a plugin
       channel's first writer fixes them (``dtype`` required, ``per`` defaults to
       ``"env"``). A rewired port is checked against the channel the stage declared and
       sized by the one it is wired to. Call it once per write port, in ``build``.
   * - ``temp_like(port)``
     - A private zero-initialised buffer shaped like a write port's channel (call
       ``write`` first for a plugin channel): a ping-pong partner. It works on
       ``anchors.*`` and ``gates.*`` ports too.
   * - ``scratch``, ``param``, ``input``, ``salt``
     - As for a generator, namespaced by the stage id: ``scratch:<id>/<name>``, and
       ``salt(stream)`` hashes ``"<id>/<stream>"``, so a stage moved to another slot id
       draws another random stream.
   * - ``fail_bit(name=None)``
     - Validity stages only: the index (16 to 30) of a new fail bit this stage owns (see
       `A validity check`_).
   * - ``host_buffer(dtype, shape)``
     - ``HOST`` stages only (``ValueError`` otherwise): a host staging buffer for
       ``ctx.host()``, pinned on ``cuda`` (see `Writing a host-driven (HOST) stage`_).

``build`` runs once, in pipeline order, after every earlier stage's ``build``. When it
returns, every declared write port must have been written (``PlanError`` otherwise).
``enqueue(ctx)`` follows the generator's :ref:`hard rules <generator-hard-rules>`: device
work through ``ctx`` and ``track_gen.ops`` only, no allocation, no host sync, no branch
on device data, a launch sequence fixed at ``build`` (an iteration that stops early is a
``ctx.bounded_loop``, see `Writing an iterative (BOUNDED) stage`_). Per env, its output must depend
only on that env's inputs (its rows of the channels it reads, its seed, its dynamic
parameters and inputs) and the static parameters.

A width stage
-------------

A width stage fills ``cl.half_width``, every slot of it. This one narrows the road
``waves`` times per lap, by a per-env fraction ``depth``:

.. code-block:: python

   @wp.kernel
   def _ripple_k(count: wp.array(dtype=wp.int32), hw: float, waves: int,
                 depth: wp.array(dtype=wp.float32), S: int,
                 w: wp.array(dtype=wp.float32)):
       t = wp.tid()
       e = t // S
       i = t - e * S
       n = count[e]
       if i >= n:
           w[t] = hw                    # padding: the whole output is written
           return
       u = float(i) / float(n)
       w[t] = hw * (1.0 - depth[e] * 0.5 * (1.0 - wp.cos(6.2831853 * float(waves) * u)))


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class RippleParams(plugin.Params):
       half_width: float = plugin.static(0.1, doc="nominal half-width")
       waves: int = plugin.static(4, lo=1, doc="narrowings per lap")
       depth: float = plugin.dynamic(0.3, lo=0.0, hi=0.9, doc="narrowing fraction")


   class Ripple(plugin.Stage):
       kind = "width"
       name = "acme.ripple"
       Params = RippleParams
       reads = {"count": "cl.count"}
       writes = {"half_width": "cl.half_width"}

       def build(self, b, params):
           count = b.read("count")
           w = b.write("half_width", wp.float32, per="cl")
           depth, S = b.param("depth"), b.layout.stride
           hw, waves = float(params.half_width), int(params.waves)

           def enqueue(ctx):
               ctx.launch(_ripple_k, b.E * S, [count, hw, waves, depth, S, w])
           return enqueue

Select it with ``presets.track(..., width="acme.ripple")`` (there is no flat-config
shortcut for widths).

Two limits of this release. First, ``cl.half_width`` reaches only ``validity.default``'s
width-floor check (fail bit 4): the band offsets, XPBD's separation target and the
thickness check still take the single ``half_width`` value, until roadmap PR13 makes
them read the channel. Until then a width stage is mostly the hook that work builds on,
and a per-point width your own code reads with ``gen.program.tap("cl.half_width")``.
Second, a field named ``half_width`` joins the planner's agreement rule: every stage with
that field must hold the same value. The preset fills ``half_width`` in for the
built-ins only, so pass it to yours when it differs from the default:
``width=("acme.ripple", {"half_width": 0.15})`` with ``half_width=0.15`` (params that set
another value than the preset's ``half_width`` raise ``ValueError`` naming it). The flat
config shortcut (``relax_solver`` / ``z_profile``) sets such a field to
``TrackGenConfig.half_width`` after your ``from_legacy``, so it needs no mapping there.

An elevation stage
------------------

An elevation stage writes ``cl.z``, one altitude per point, from the plan-view arc
length (``cl.arclen2``, cumulative, and ``cl.length2``, the lap length) that ``frames``
writes before it. The lift then builds level cross-sections at that altitude, and the 3D
tangent and arc length. This one draws a random phase per env and makes ``waves``
sinusoidal hills per lap with a per-env amplitude:

.. code-block:: python

   from track_gen.ops.funcs import hash32


   @wp.kernel
   def _hills_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
                cum: wp.array(dtype=wp.float32), perim: wp.array(dtype=wp.float32),
                count: wp.array(dtype=wp.int32), amp: wp.array(dtype=wp.float32),
                waves: int, base: float, S: int, z: wp.array(dtype=wp.float32)):
       t = wp.tid()
       e = t // S
       i = t - e * S
       if i >= count[e]:
           z[t] = 0.0                   # padding, as the built-in profiles write it
           return
       key = hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)
       state = wp.rand_init(wp.int32(key))
       phase = 6.2831853 * wp.randf(state)          # the same draw for every point of e
       u = cum[t] / wp.max(perim[e], 1.0e-9)        # lap fraction in [0, 1)
       z[t] = base + amp[e] * wp.sin(6.2831853 * float(waves) * u + phase)


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class HillsParams(plugin.Params):
       amplitude: float = plugin.dynamic(0.2, lo=0.0, hi=5.0, doc="hill height")
       waves: int = plugin.static(3, lo=1, doc="hills per lap")
       base: float = plugin.static(0.0, doc="mean altitude")

       @classmethod
       def from_legacy(cls, cfg):
           return cls(base=float(cfg.z_base))      # TrackGenConfig(z_profile=...) users


   class SinusoidHills(plugin.Stage):
       kind = "elevation"
       name = "acme.sinusoid_hills"
       Params = HillsParams
       reads = {"cum": "cl.arclen2", "perim": "cl.length2", "count": "cl.count"}
       writes = {"z": "cl.z"}

       def build(self, b, params):
           cum, perim, count = b.read("cum"), b.read("perim"), b.read("count")
           z = b.write("z", wp.float32, per="cl")
           amp, salt, S = b.param("amplitude"), b.salt("phase"), b.layout.stride
           waves, base = int(params.waves), float(params.base)

           def enqueue(ctx):
               ctx.launch(_hills_k, b.E * S,
                          [b.seed, salt, cum, perim, count, amp, waves, base, S, z])
           return enqueue

Select it with ``presets.track(..., elevation="acme.sinusoid_hills")``, or with
``TrackGenConfig(z_profile="acme.sinusoid_hills")``: ``z_profile`` names any registered
elevation stage, and a plugin's Params come from its ``from_legacy(config)``, which may
map the flat ``z_*`` fields (here ``z_base``). Either way the pipeline lifts with
``lift.elevated``, since the elevation is not ``elevation.flat``, and
``validity.default``'s grade check (``z_valid_grade``, fail bit 6) reads the new ``cl.z``.

The random stream is keyed on the effective seed and the stage's salt
(:ref:`stage-salts`), so env ``e``'s hills change when its seed or epoch changes and
depend on nothing else.

**On gates.** The same class runs in a gate pipeline, with no change:
``presets.gates(..., elevation="acme.sinusoid_hills")`` or
``GateGenConfig(z_profile="acme.sinusoid_hills")``. The planner rewires its ports to the
gate channels (``cum -> gates.cum``, ``perim -> gates.perim``, ``count -> gates.count``,
``z -> gates.z``; see :ref:`pipeline-gate-elevation`), ``b.layout.stride`` is then
``max_gates``, and ``b.write("z", wp.float32, per="cl")`` returns a ``[E * max_gates]``
buffer: ``per`` is checked against the channel the stage declares. The salt is keyed by
the slot id, ``"elevation"`` in both pipelines. A stage that sizes buffers itself should
use ``b.stride_of(port)`` rather than assume ``N_max``. Through
``GateGenConfig(z_profile=...)``, ``from_legacy(config)`` receives the ``GateGenConfig``:
only the ``z_*`` fields are shared with ``TrackGenConfig``, so read any other field with
``getattr(config, name, default)`` (a ``from_legacy`` that reads a ``TrackGenConfig``-only
field raises ``ValueError`` naming it). Test it on gates with
``check_stage("elevation", "acme.sinusoid_hills", pipeline="gates")``
(:ref:`kit-check-stage`).

.. _stage-validity-check:

A validity check
----------------

A validity stage reads ``valid.fail``, ORs in the bit it owns where its check fails,
and writes the result to a new version. ``b.fail_bit()`` assigns the bit: 16 for the
first plugin check of a pipeline, 17 for the next, in pipeline order. Pass it to the
kernel as a host scalar:

.. code-block:: python

   @wp.kernel
   def _kmax_k(kappa: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
               k_max: wp.array(dtype=wp.float32), S: int, bit: int,
               fail_in: wp.array(dtype=wp.int32), fail_out: wp.array(dtype=wp.int32)):
       e = wp.tid()                     # one thread per env
       m = float(0.0)
       for i in range(count[e]):
           m = wp.max(m, wp.abs(kappa[e * S + i]))
       f = fail_in[e]                   # the bits of the earlier checks
       if m > k_max[e]:
           f = f | (1 << bit)
       fail_out[e] = f                  # every env, failed or not


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class KappaParams(plugin.Params):
       kappa_max: float = plugin.dynamic(10.0, lo=0.0, doc="largest |curvature| allowed")


   class MaxCurvature(plugin.Stage):
       kind = "validity"
       name = "acme.max_curvature"
       Params = KappaParams
       reads = {"kappa": "cl.kappa", "count": "cl.count", "fail": "valid.fail"}
       writes = {"fail": "valid.fail"}

       def build(self, b, params):
           kappa, count = b.read("kappa"), b.read("count")
           fail_in = b.read("fail")
           fail_out = b.write("fail", wp.int32)
           bit = b.fail_bit()                       # 16 for the first plugin check
           k_max, S = b.param("kappa_max"), b.layout.stride

           def enqueue(ctx):
               ctx.launch(_kmax_k, b.E, [kappa, count, k_max, S, bit, fail_in, fail_out])
           return enqueue

Add it after the built-in checks:

.. code-block:: python

   from track_gen.pipeline import StageRef, presets

   spec = presets.track(num_envs=1024, device="cuda:0",
                        checks=("default", ("acme.max_curvature", {"kappa_max": 8.0})))
   gen = TrackGenerator(spec, rng)
   gen.program.fail_bits      # {'generator': 0, ..., 'truncation': 7, 'acme.max_curvature': 16}

   # or into any spec, under an id of your choice:
   spec = spec.insert_after("validity", StageRef("validity", "acme.max_curvature", id="kmax"))

``Track.valid`` is ``fail_mask == 0``, so the check rejects envs like the built-in ones,
and ``Track.fail_mask`` tells which check rejected each env. The rules of fail bits:

- Call ``b.fail_bit(name=None)`` in ``build``, once per independent check the stage
  runs. ``name`` keys the bit in ``Program.fail_bits`` / ``Track.fail_bits`` and
  defaults to the stage id (``"acme.max_curvature"`` for a check added by name in
  ``checks``, ``"kmax"`` above). A stage of another kind gets ``ValueError``.
- A validity stage owns at least one bit (``PlanError`` otherwise), sets only its own
  bits and keeps the others: ``fail_out = fail_in | own``. ``check_stage``'s
  ``validity.bits`` check enforces it.
- A pipeline holds at most 15 plugin bits (16 to 30); one more, or a name already taken,
  raises ``PlanError``. Bits 0 to 15 belong to TrackGen's checks (see
  :ref:`pipeline-fail-bits`).
- Write every env. An env that passes gets ``fail_in[e]`` unchanged.

A check must read ``valid.fail``. One that writes it without reading it would start the
bitmask in place of ``validity.default``, dropping the generator, turning, thickness and
truncation checks (a truncated loop would pass): the planner refuses such a pipeline
(``PlanError``) until the split built-in checks arrive, and ``validity.default`` comes
first in ``checks``.

With ``on_invalid="poison"`` the commit empties every env with a bit set, yours included
(see :ref:`pipeline-commit`).

**On gates.** A gate check is a validity stage that reads gate channels
(``gates.position``, ``gates.count``, ...; :ref:`pipeline-gate-channels`) with
``b.layout.stride == max_gates``, under the same rules: it reads and rewrites
``valid.fail``, owns bits from 16 on, and follows ``validity.gate_default``, which owns
bits 0 to 5 (:ref:`pipeline-gate-fail-bits`). Add it with
``presets.gates(..., checks=("gate_default", "acme.max_height"))``; its bit then shows in
``GateSequence.fail_mask`` and ``GateSequence.fail_bits``. ``check_stage`` runs a check
that reads a ``gates.*`` or ``anchors.*`` channel in the default gate pipeline.

.. _gate-generator-stage:

Writing a gate generator (anchors)
----------------------------------

A gate generator emits *anchors*: the plan-view points the gate pipeline then orders,
normalizes, separates, elevates, lifts and frames (:ref:`pipeline-gates`). It is a
``Stage`` with ``kind="gate_generator"`` and ``output="anchors"``. It writes two
channels, ``anchors.xy`` (``[E * A]`` ``vec2f``, env ``e``'s anchors at ``e * A + i``) and
``anchors.count`` (``[E]`` ``int32``), and may write per-anchor attributes it
``provides``. It does not order its anchors (an ``order`` stage does), size them to the
course extent (``normalize.extent`` does, when it declares
``preferred={"normalize": "extent"}``), or compute altitudes, tangents and frames.

This one places ``gates`` anchors on a circle with a random phase per env, and provides
their altitude (a wave that climbs ``turns`` times per lap) and their analytic tangent, so
the pipeline keeps both instead of running an elevation and a finite-difference tangent:

.. code-block:: python

   import dataclasses

   import warp as wp

   from track_gen import plugin
   from track_gen.ops.funcs import hash32


   @wp.kernel
   def _helix_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, A: int, r: float,
                height: float, turns: int, xy: wp.array(dtype=wp.vec2f),
                z: wp.array(dtype=wp.float32), tangent: wp.array(dtype=wp.vec3f)):
       t = wp.tid()                         # one thread per anchor slot: t = e * A + i
       e = t // A
       i = t - e * A
       key = hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)
       state = wp.rand_init(wp.int32(key))
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

Select it through the preset, which reads the generator's declaration to build the
stage list:

.. code-block:: python

   from track_gen import GateGenConfig, GateGenerator
   from track_gen.pipeline import presets

   spec = presets.gates(num_envs=1024, device="cuda:0", generator="acme.helix")
   spec.ids
   # ('generator', 'order', 'carry_z', 'carry_tangent', 'gate_relax', 'lift',
   #  'gate_frames', 'validity')
   gates = GateGenerator(spec).generate()

   # The flat config too: gate_ordering=None (its default) is the generator's
   # default order, "raw" here.
   GateGenerator(GateGenConfig(num_envs=1024, device="cuda:0", generator="acme.helix"))

``gate_arclen``, ``elevation`` and ``tangents`` are gone: ``carry.z`` and
``carry.tangent`` copy the provided values to the gate slots, so ``GateSequence.position``
has the helix's altitude bit for bit and ``GateSequence.tangent`` its normalized tangent.
The sphere separation (``gate_relax``) still runs and may move a gate's xy; these gates
are far apart, so it does not.

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
     - The ``GateGenConfig.generator`` (or ``presets.gates(generator=...)``) string.
       Dotted for third parties.
   * - ``reads``
     - Empty: a generator reads no channel; its seed is ``b.seed``.
   * - ``writes``
     - ``anchors.xy`` and ``anchors.count`` (both required), plus ``anchors.z``,
       ``anchors.tangent``, ``anchors.half_size`` and ``anchors.roll`` for the
       attributes it provides; no other channel.
   * - ``provides``
     - Exactly the attributes it writes, a subset of ``{"z", "tangent", "half_size",
       "roll"}``; ``half_size`` and ``roll`` come together (write ``roll = 0`` for
       upright gates). See :ref:`pipeline-provides` for what the pipeline drops and
       what reaches the ``GateSequence``.
   * - ``orderings``
     - A whitelist (``frozenset``) of the order names it accepts, or ``None``. A
       generator that provides attributes must declare ``frozenset({"raw"})``: the
       attributes are carried by anchor index, and only ``order.raw`` keeps it.
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
       stride of ``anchors.*``), ``k`` in ``[0, A]`` at least (default ``0``). The planner
       requires ``A <= max_gates`` and the validity stage's ``min_gates <= A``.
   * - ``build(b, params)``
     - As for any stage. ``b.layout.stride`` is ``max_gates``; size the anchors with
       ``per="anchors"`` (``A``), never with the layout. Write ``count[e]`` and the first
       ``count[e]`` anchors of every env on every call; the slots after them are never
       read (write them deterministically, NaN or not, as the built-in samplers do).

Everything else is the generator contract of :ref:`generator-stage-contract`: Params
(static and dynamic), ``capture`` and ``capture_class``, salts, the hard rules of
``enqueue``, per-env determinism. ``registry.add`` raises ``TypeError`` or ``ValueError``
(quarantine under ``warn``) for a declaration that breaks the table, for example
``Stage Helix provides per-anchor attributes ['tangent', 'z']: they are carried to the gate
slots by index, so orderings must be frozenset({'raw'})``.

The anchors keep the generator's own order, and the pipeline orders them: an unordered
cloud (as ``bezier`` emits) excludes ``raw``; a loop in course order (as ``polar``
emits) excludes ``random_pairs`` and is fine with ``raw`` or ``ccw``. Size the anchors as
you like: with ``preferred["normalize"] = "extent"`` the pipeline centers them and scales
the longest bounding-box side to ``ops.constants.TARGET_EXTENT * scale``; without it,
they are used as written (``bezier`` and ``hull`` keep their grid's size).

Test it with ``check_gate_generator("acme.helix")`` (:doc:`testing-a-plugin`); it also
checks that every env's ``anchors.count`` lies within ``capacity()``. A legacy
``GateGeneratorSpec`` keeps working but is deprecated
(:ref:`gate-generator-contract`).

.. _writing-an-ordering:

Writing an ordering
-------------------

An ``order`` stage decides the sequence in which the gates are flown. It reads the
anchors (``A``-strided) and writes the first ``m`` gate slots (``G``-strided) of
``gates.xy``, NaN after them, where ``m = min(count[e], A, G)``, and ``gates.count``,
normally a copy of ``anchors.count``. This one flies a generator's own order backwards:

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

The read and write ports have different names: ``anchors.xy`` and ``gates.xy`` are two
channels, and a port in both maps must name one channel. ``A`` and ``G`` come from
``b.stride_of``: the anchors are written at the generator's own stride, the gate slots at
``max_gates``.

Select it by name, with the flat config or the preset:

.. code-block:: python

   GateGenerator(GateGenConfig(num_envs=1024, device="cuda:0", generator="polar",
                               gate_ordering="acme.reverse"))
   presets.gates(num_envs=1024, device="cuda:0", generator="polar",
                 order="acme.reverse")

Every built-in gate generator accepts a third-party order: they declare
``excluded_orderings``, not a whitelist. A generator with a whitelist (``orderings``, as
one that provides attributes) accepts only the orders it lists, and the planner refuses
the others with ``gate generator 'acme.helix' does not support gate_ordering
'acme.reverse'; supported: ['raw'] (its preferred order is 'raw')``. An order that
needs randomness reads the ``seed`` channel (``reads = {..., "seed": "seed"}``) and keys
its stream on ``b.salt(<stream>)`` (:ref:`stage-salts`); the built-in
``order.random_pairs`` keeps 0.2.0.dev3's fixed salt 7919 for byte identity. Test it
with ``check_stage("order", "acme.reverse")``.

.. _stage-runtime-params:

Runtime parameters
------------------

A stage's dynamic parameters work as a generator's (see
:ref:`generator-runtime-params`), keyed by the stage id: ``gen.params`` lists
``"generator"`` and every stage with a dynamic field, ``gen.params["relax"]["lam"]`` is
the live ``[E]`` array, and ``gen.params["relax"].set("lam", 0.6)`` writes it, on the
resolved stream, recordable inside a caller's graph. The new value is used from the next
generation. ``b.param`` returns the stage's working copy, latched at the start of every
generation (for the masked envs only, under a masked regeneration); read it in
``enqueue``, never write it.

Only static fields reach ``build`` and ``capture_class`` (reading a dynamic one raises
``AttributeError``): a host value derived from a dynamic field would be baked into the
launches and any recorded graph.

Capture classes
---------------

As for a generator (:ref:`stage-capture-classes`), and the same in a gate pipeline:

- ``STATIC`` (default): a fixed launch sequence, recorded and replayed like the
  built-ins.
- ``BOUNDED``: an iteration with a static bound that may stop early, through
  ``ctx.bounded_loop``. Capturable by anything: a self-graph, or a Warp-registered
  capture (``wp.ScopedCapture``, ``wp.capture_begin``, ``registered(conditional=True)``)
  on CUDA 12.4+, skips the converged windows on device; other captures (a plain
  ``torch.cuda.graph``, inside ``track_gen.exec.unconditional()``) and eager ``cuda``
  calls run the full bound (see `Writing an iterative (BOUNDED) stage`_).
- ``HOST``: host readbacks or host control flow. The stage runs eagerly on every call,
  as its own segment of the ``Program``: the stages before and after it still replay
  from graphs. A caller cannot record the Program (``NotCapturableError`` before any
  launch), and ``PipelineSpec(require_caller_capturable=True)`` refuses it at build (see
  `Writing a host-driven (HOST) stage`_).

The Program's capture class is the highest of its stages' (``STATIC`` < ``BOUNDED`` <
``HOST``); ``Program.segments`` shows how the stages are grouped (see
:doc:`/how-it-works/pipeline-preview`).

.. _stage-bounded:

Writing an iterative (BOUNDED) stage
------------------------------------

A stage that iterates until each env has converged keeps a static launch sequence by
writing its loop as ``ctx.bounded_loop(n_iters, body, active=..., first_check=...,
check_every=...)``: ``n_iters`` bounds the iteration count, ``body(ctx, it)`` emits one
iteration (``it`` is a plain ``int``), and ``active``, a ``[1]`` int32 flag the body
writes on device, says whether any env still needs the next window of ``check_every``
iterations. Where the call runs decides whether the loop really stops early (see
:ref:`cuda-graph-bounded`): an eager ``cpu`` run polls ``active`` and stops, a self-graph
or another Warp-registered capture with conditional nodes skips the remaining windows on
device, every other run executes the full bound. The outputs must be the same bytes in
every case.

This relax stage moves each env toward a fixed point (a Laplacian smoothing step plus a
pull back to its input loop, an iteration that contracts) and freezes an env once its
largest step falls under ``tol``:

.. code-block:: python

   import dataclasses

   import warp as wp

   from track_gen import ops, plugin


   @wp.kernel
   def _step_k(x: wp.array(dtype=wp.vec2f), x0: wp.array(dtype=wp.vec2f),
               count: wp.array(dtype=wp.int32), S: int, lam: float, pull: float,
               running: wp.array(dtype=wp.int32), tmp: wp.array(dtype=wp.vec2f),
               moved: wp.array(dtype=wp.float32)):
       t = wp.tid()
       e = t // S
       i = t - e * S
       n = count[e]
       if running[e] == 0 or i >= n:        # a frozen env is never touched again
           return
       base = e * S
       p = x[t]
       mid = 0.5 * (x[base + (i + n - 1) % n] + x[base + (i + 1) % n])
       d = lam * (mid - p) + pull * (x0[t] - p)
       tmp[t] = p + d
       wp.atomic_max(moved, e, wp.length(d))   # max: independent of the thread order


   @wp.kernel
   def _apply_k(tmp: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32), S: int,
                running: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec2f)):
       t = wp.tid()
       e = t // S
       if running[e] != 0 and t - e * S < count[e]:
           x[t] = tmp[t]


   @wp.kernel
   def _freeze_k(moved: wp.array(dtype=wp.float32), tol: float,
                 running: wp.array(dtype=wp.int32)):
       e = wp.tid()
       if moved[e] < tol:
           running[e] = 0                   # absorbing: nothing sets it back to 1


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class SettleParams(plugin.Params):
       max_iters: int = plugin.static(64, lo=1, doc="iteration bound")
       check_every: int = plugin.static(8, lo=1, doc="iterations per window")
       lam: float = plugin.static(0.3, lo=0.0, hi=0.5, doc="smoothing factor")
       pull: float = plugin.static(0.2, lo=0.0, hi=0.5, doc="pull back to the input")
       tol: float = plugin.static(1e-5, lo=0.0, doc="largest step of a settled env")


   class Settle(plugin.Stage):
       kind = "relax"
       name = "acme.settle"
       capture = plugin.CaptureClass.BOUNDED   # its loop may stop early
       Params = SettleParams
       reads = {"xy": "cl.xy", "count": "cl.count"}
       writes = {"xy": "cl.xy"}

       def build(self, b, params):
           x, count = b.read("xy"), b.read("count")
           out = b.write("xy", wp.vec2f, per="cl")
           tmp = b.temp_like("xy")
           running = b.scratch("running", wp.int32, b.E)    # 1 while the env moves
           moved = b.scratch("moved", wp.float32, b.E)      # its largest step
           active = b.scratch("active", wp.int32, 1)        # any env still running
           S, E, K = b.layout.stride, b.E, params.check_every
           lam, pull, tol = float(params.lam), float(params.pull), float(params.tol)

           def body(ctx, it):
               ctx.fill(moved, 0.0)
               ctx.launch(_step_k, E * S,
                          [out, x, count, S, lam, pull, running, tmp, moved])
               ctx.launch(_apply_k, E * S, [tmp, count, S, running, out])
               if (it + 1) % K == 0:                   # the iteration before a window
                   ctx.launch(_freeze_k, E, [moved, tol, running])
                   ops.any_set(ctx, running, active)

           def enqueue(ctx):
               ctx.copy(out, x)                        # the padding too
               ctx.fill(running, 1)
               ctx.bounded_loop(params.max_iters, body, active=active,
                                first_check=K, check_every=K)
           return enqueue

For eight ``bezier`` tracks (seeds 0 to 7) on ``cpu``, the loop stops after three of its
seven windows, with the same bytes as all 64 iterations give. Like the Taubin quick
start, the stage shows the mechanics: it enforces no separation, so at the default
``half_width`` its tracks fail the thickness check, as they do with ``relax=None``.

The checklist of a ``BOUNDED`` stage:

- **Declare it.** ``capture = plugin.CaptureClass.BOUNDED`` (or a ``capture_class`` that
  returns it). A stage declared ``STATIC`` that calls ``bounded_loop(active=...)`` raises
  ``ValueError`` at construction (``... declare capture = CaptureClass.BOUNDED``).
  ``active=None`` (a plain static unroll) is fine in any stage.
- **Make the freeze absorbing.** Once ``active`` reads ``0``, every remaining iteration
  must leave the stage's outputs (and anything it reads after the loop) bitwise
  unchanged: a per-env flag that only ever goes from running to frozen, and every kernel
  that writes an output returning early for a frozen env. Scratch the loop alone uses
  (``moved``, ``active`` here) may differ.
- **Refresh** ``active`` **on the iteration before each window.** The windows start at
  ``first_check``, ``first_check + check_every``, ...; the check before the window at
  ``lo`` reads what iteration ``lo - 1`` wrote. Here ``first_check = check_every = K``
  and the refresh runs when ``(it + 1) % K == 0``. Write it before the first check, in
  every generation: a check must never read last generation's value.
- **Keep the body pure in** ``it``. Branch only on ``it`` and on static params; mutate no
  Python state that code after the loop reads (the loop may stop at any window), and
  emit only through the ``ctx`` the body is given.
- **Keep each window device-only.** Under lowering (b) a window is the body of a CUDA
  conditional node, which may hold kernel launches, memsets and device-to-device copies
  only. Launch kernels and use ``ctx.copy`` / ``ctx.fill`` between device arrays on
  ``ctx``'s stream; no copy to or from host or pinned memory, no event, no other stream
  (no ``wait_stream`` fork or join). Anything else fails inside ``wp.capture_if`` while
  the caller's capture is being recorded, so the iterations before the window are
  already recorded. ``check_stage(..., devices=["cuda:0"])`` (``bounded.lowering``)
  records every window that way.
- **Mind the cost.** A self-graph replays a conditional node per window; a caller's plain
  ``torch.cuda.graph`` (and any eager ``cuda`` call) runs the full ``n_iters``, so pick a
  bound you can afford, with the per-iteration work of a frozen env reduced to early
  returns.

``check_stage("relax", "acme.settle")`` checks it like any stage, plus
``bounded.absorbing`` (the host-poll lowering against the unrolled one, with ``active``
poisoned)
and, on CUDA, ``bounded.lowering`` (see :doc:`testing-a-plugin`). A loop that keeps
moving a converged env also fails ``batch_independence``: under an early stop it moves
until the WHOLE batch has converged, so its output depends on the other envs.

``ops.any_set(ctx, mask, flag)`` (any env set) and ``ops.all_set(ctx, mask, flag)``
(every env set) reduce an ``[E]`` int32 mask into the ``[1]`` flag in two launches,
deterministically. ``repulsive`` is the built-in example: its stall check freezes an env
on device, and its growth loop is a ``bounded_loop`` over that freeze (see
:doc:`/generators/repulsive`).

.. _stage-host:

Writing a host-driven (HOST) stage
----------------------------------

Some work needs the host: a NumPy or SciPy routine, a readback that decides what runs
next. Declare ``capture = CaptureClass.HOST``. The stage then runs eagerly on every call,
outside every capture, as its own segment: the stages around it still replay from graphs
on ``cuda`` (see :ref:`cuda-graph-host`). It reaches the host through ``ctx.host()``, a
:class:`~track_gen.exec.HostOps`, with staging buffers it allocates at build with
``b.host_buffer(dtype, shape)`` (pinned on ``cuda``, so nothing allocates per call).

This generator draws each env's loop on the host with NumPy, from the env's effective
seed:

.. code-block:: python

   import dataclasses

   import numpy as np
   import warp as wp

   from track_gen import ops, plugin


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class BlobParams(plugin.Params):
       points: int = plugin.static(64, lo=8, doc="points per loop")


   class Blob(plugin.Stage):
       kind = "generator"
       name = "acme.blob"
       output = "centerline"
       capture = plugin.CaptureClass.HOST      # host RNG: runs eagerly
       Params = BlobParams
       writes = {"xy": "raw.xy", "valid": "gen.valid"}

       def capacity(self, params, layout):
           return {"raw": params.points}

       def build(self, b, params):
           n, E = params.points, b.E
           xy = b.write("xy", wp.vec2f, per="raw")         # [E * n] on the device
           valid = b.write("valid", wp.int32)              # [E]
           seed_h = b.host_buffer(wp.int32, E)             # host staging buffers
           xy_h = b.host_buffer(wp.vec2f, E * n)
           theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
           radius = 0.5 * ops.constants.TARGET_EXTENT * b.scale

           def enqueue(ctx):
               h = ctx.host()
               seeds = h.readback(b.seed, via=seed_h)      # a NumPy view of seed_h
               pts = np.empty((E, n, 2), np.float32)
               for e, s in enumerate(seeds):
                   rng = np.random.default_rng(int(s) & 0x7FFFFFFF)
                   lobes, phase = rng.integers(2, 6), rng.uniform(0.0, 2.0 * np.pi)
                   r = radius * (1.0 + 0.2 * np.sin(lobes * theta + phase))
                   pts[e, :, 0], pts[e, :, 1] = r * np.cos(theta), r * np.sin(theta)
               h.upload(xy, pts.reshape(E * n, 2), via=xy_h)   # returns once copied
               ctx.fill(valid, 1)                          # device work goes through ctx
           return enqueue

``HostOps`` has four operations, each on the call's stream:

- ``read_i32(arr, index=0)``: one element of an int32 device array, through a
  one-element host buffer the Program allocates for the stage.
- ``readback(src, via=...)``: copies ``src`` into ``via`` (a ``host_buffer`` of the same
  dtype, at least as large), waits, and returns a NumPy view of ``via`` shaped like
  ``src``, valid until ``via`` is used again.
- ``upload(dst, data, via=...)``: copies ``data`` into ``via``, then ``via`` into
  ``dst``, and waits, so ``via`` is free again on return.
- ``sync()``: waits until the call's stream has drained (a no-op on ``cpu``).

The rules of a ``HOST`` stage:

- ``ctx.host()`` works only inside a ``HOST`` stage's ``enqueue``: a ``STATIC`` or
  ``BOUNDED`` stage that calls it raises ``ValueError`` on its first run, naming the
  stage and its declared class (as ``bounded_loop(active=...)`` in a ``STATIC`` stage
  does), and so does ``b.host_buffer`` in another stage.
- Never under a capture: a caller's capture of the Program raises ``NotCapturableError``
  naming the ``HOST`` stages before any launch, and ``ctx.host()`` raises it too.
- Reach the host only through ``ctx.host()``, with ``via=`` buffers from
  ``b.host_buffer`` (on ``cuda`` a ``via`` that is not pinned raises ``ValueError``). A
  direct ``.numpy()``, ``wp.synchronize*()`` or a per-call allocation reads on the wrong
  stream, may allocate, and can invalidate another thread's CUDA capture: the
  conformance kit fails ``hot_path.host_sync`` / ``hot_path.allocation`` for them, while
  the ``HostOps`` operations pass (they are safe while other threads capture on the
  device).
- The rest of the contract holds: per-env determinism in the seed, no state carried from
  one generation to the next, every output slot written on every call.

Users who need to record the whole pipeline into their own graph build it with
``PipelineSpec(..., require_caller_capturable=True)``, which refuses ``HOST`` stages at
build with a ``PlanError`` naming them. ``check_generator("acme.blob")`` (or
``check_stage`` for a pipeline stage) runs the ``HOST`` variants of the checks: eager
determinism and statelessness, the segmented self-graph against an eager run, and the
refusals (see :doc:`testing-a-plugin`).

Registering and selecting
-------------------------

``register(reg)`` adds stage classes with ``reg.add(MyStage)``, next to any generators.
Names follow the generator rules (dotted, prefixed by your entry-point name, see
:ref:`plugin-names`), and so do duplicates, loading and quarantine (see
:ref:`how a quarantined plugin is matched <plugin-quarantine>`): a ``z_profile`` or
``relax_solver`` that names a stage of a quarantined plugin raises its
``PluginLoadError``. ``registry.describe(kind, name)``
reports a stage's ``reads``, ``writes``, ``capture``, ``capturable`` and
``params_schema``; ``output`` is ``None`` (``"anchors"`` for a gate generator, which also
reports ``orderings``, ``excluded_orderings`` and ``preferred``).

A stage is selected by:

- ``presets.track(..., relax=..., width=..., elevation=..., checks=(...))``, with a name,
  ``(name, params)`` or a ``StageRef`` (see :ref:`pipeline-presets`);
- ``presets.gates(..., generator=..., order=..., stations=..., elevation=...,
  checks=(...))``, with the same forms (see :ref:`pipeline-gate-presets`);
- the edit operations of any ``PipelineSpec`` (``replace``, ``insert_after``, ...);
- for relax and elevation, the flat ``TrackGenConfig`` fields ``relax_solver`` and
  ``z_profile`` (see :ref:`flat-stage-shortcuts`); for gates, the ``GateGenConfig``
  fields ``generator``, ``gate_ordering`` and ``z_profile``.

Testing a stage
---------------

``track_gen.testing.check_stage`` runs a stage inside the default track pipeline (an
``order`` or ``stations`` stage, or a check that reads gate channels, inside the default
gate pipeline) through the conformance checks: shapes and NaN padding of the ``Track``
(or ``GateSequence``), determinism, batch
independence, statelessness, the hot-path guards, caller capture on CUDA, the
``bounded.*`` checks for a stage that runs a ``bounded_loop`` and ``host.*`` for a
``HOST`` stage, and two stage-specific checks, ``stage.isolation`` (the stage writes none
of its inputs and recomputes its output from them) and, for a validity stage,
``validity.bits``:

.. code-block:: python

   from track_gen.testing import check_stage

   check_stage("relax", "acme.taubin", {"passes": 4}).raise_for_failures()
   check_stage("elevation", "acme.sinusoid_hills").raise_for_failures()
   check_stage("validity", "acme.max_curvature",
               upstream={"elevation": "acme.sinusoid_hills"}).raise_for_failures()
   check_stage("order", "acme.reverse").raise_for_failures()

See :ref:`kit-check-stage` for the checks and the arguments. Test the stage's own
properties too: that a ``tap`` of its output matches a NumPy reference on a small batch,
that a dynamic parameter set with ``gen.params[...].set`` takes effect, that a check sets
its bit exactly where it should (:ref:`plugin-own-tests` shows the patterns).
