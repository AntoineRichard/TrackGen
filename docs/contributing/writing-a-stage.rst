.. _writing-a-stage:

Writing a Stage
===============

.. admonition:: Before you start

   - **What you write:** a ``Stage`` subclass of one of the open kinds below (a relax,
     width, elevation, validity, order or stations stage), optionally its ``Params``
     class, and a ``reg.add(...)`` line in your package's ``register(reg)``.
   - **What you need to know:** the generator guide (:ref:`writing-a-generator`):
     ``Params``, the ``StageBuilder``, salts, capture classes and the hard rules of
     ``enqueue`` are the same for every stage, and only what differs is repeated here.
     Where your stage runs and what it reads: :ref:`pipeline-stages` and
     :ref:`pipeline-stage-kinds`.
   - **How you test it:** ``track_gen.testing.check_stage`` runs it in the default
     pipeline through the conformance checks (:ref:`kit-check-stage`).

A :term:`stage` is one step of the track pipeline or of the gate pipeline (see
:ref:`pipeline-stages` and :ref:`gate-pipeline`). Besides the generators, six kinds of
stage are open to plugins:

- ``relax`` reshapes the resampled centerline (``cl.xy``), as the built-in XPBD solver
  does;
- ``width`` writes the per-point half-width (``cl.half_width``) that XPBD, the band and
  the checks read;
- ``elevation`` writes the per-point altitude (``cl.z``), and the per-gate altitude
  (``gates.z``) when it runs in a gate pipeline;
- ``validity`` runs a check per :term:`env` and sets one of the :term:`fail bits`
  (``valid.fail``), on a track or on gates;
- ``order`` puts a gate generator's anchors in course order (``gates.xy``), see
  :ref:`writing-an-ordering`;
- ``stations`` places gate anchors on a track generator's centerline
  (``anchors.xy``).

A gate generator is a stage too, of kind ``gate_generator``, and the ``order`` kind is
described with it: see :ref:`writing-a-gate-generator`.

A stage is a ``Stage`` subclass, the protocol a generator Stage uses (see
:ref:`generator-stage-contract`), with read ports: it reads and writes
:term:`channels <channel>`. It ships the way a generator ships: in its own
pip-installable package, found through the ``track_gen.plugins`` entry point, registered
by the same ``register(reg)`` function, which may add generators and stages alike
(:ref:`packaging-a-plugin`). It imports only public modules (``track_gen.plugin``,
``track_gen.exec``, ``track_gen.ops``), and ``track_gen.testing.check_stage`` checks it
against the contract on this page (see :ref:`kit-check-stage`).

Quick start: a relax stage
--------------------------

Taubin smoothing alternates a shrinking Laplacian step (``lam > 0``) and an inflating one
(``mu < 0``), which smooths a closed polyline without shrinking it. It makes a relax
stage with one static parameter (the number of passes, which fixes the launch count) and
two :term:`dynamic fields <dynamic field>` (the step factors, per env, changeable at run
time).

**1. Declare the entry point**, as for a generator (skip it if the package already has
one; one ``register`` adds everything):

.. code-block:: toml

   [project.entry-points."track_gen.plugins"]
   acme = "acme_tracks:register"

**2. Write the stage and register it** (``acme_tracks/taubin.py``):

.. snippet: group=taubin register=register

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

**3. Select it.** Through a :term:`preset`, by name, with params as a mapping or a
``TaubinParams``:

.. snippet: group=taubin

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   device = "cuda:0"  # or "cpu"
   spec = presets.track(num_envs=64, device=device,
                        relax=("acme.taubin", {"passes": 20}))
   gen = TrackGenerator(spec)
   track = gen.generate()

   gen.params["relax"].set("lam", 0.6)     # every env, from the next generation on
   track = gen.generate(advance=True)

Or through the flat config, whose ``relax_solver`` names any registered relax stage (it
then takes ``TaubinParams.from_legacy(config)``, its defaults here):

.. snippet: group=taubin

.. code-block:: python

   from track_gen import TrackGenConfig

   gen = TrackGenerator(TrackGenConfig(num_envs=64, device=device,
                                       relax_solver="acme.taubin"))

.. snippet: after

   assert gen.params["relax"].params.passes == 10

The rest of the pipeline is unchanged: the uniform resample, frames, offsets, elevation,
validity and lift run on the smoothed loop, and ``gen.program.tap("cl.xy",
after="resample")`` still holds the loop before it.

This example shows the mechanics, not a replacement for XPBD. Taubin smoothing enforces
no separation between distant parts of the loop, so it fails the thickness check (fail
bit 3) when the road is too wide for the remaining separation or local curvature.
Measure the failure bits for your own shape and width settings. A relax
stage you ship must leave a loop the band can be inflated around; ``Track.fail_mask``
tells you which check it fails. To separate the loop by the band's own width, read
``cl.half_width`` (see :ref:`stage-read-half-width`).

More stage recipes and contracts
--------------------------------

The example above supplies a relax stage. Follow the focused sections for
:ref:`width stages <stage-width>`, :ref:`elevation stages <stage-elevation>`,
:ref:`validity checks <stage-validity-check>`, :ref:`live settings <stage-runtime-params>`,
:ref:`bounded device loops <stage-bounded>` and :ref:`host stages <stage-host>`.
The contracts specify read/write ports, per-point widths and capture behavior.

.. toctree::
   :maxdepth: 1

   /reference/stage-contracts
