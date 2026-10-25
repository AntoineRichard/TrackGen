.. _writing-a-generator:

Writing a Generator
===================

.. admonition:: Before you start

   - **What you write:** a ``Stage`` subclass with ``kind = "generator"``, its ``Params``
     class, and a ``register(reg)`` function, in a package of your own.
   - **What you need to know:** Warp kernels (``@wp.kernel``, one thread per point or per
     :term:`env`), the flat per-env layout of the buffers (env ``e`` owns rows
     ``[e * n, (e + 1) * n)``), and what the pipeline does with your loop afterwards
     (:ref:`generating`, :ref:`pipeline-stages`).
   - **How you test it:** ``track_gen.testing.check_generator`` runs the contract on this
     page in one call (:ref:`testing-a-plugin`); :ref:`packaging-a-plugin` covers the
     entry point that ships it.

A **track generator** produces the raw closed centerline of every env; the pipeline then
resamples it, gives it a width, relaxes it (XPBD), checks it and inflates it into a road
(:ref:`pipeline-stages-the-default-track-stages`). A generator is a :term:`stage` of the
track pipeline, the first one: a ``Stage`` subclass that declares its own parameters as a
:term:`Params` dataclass, allocates its buffers through a ``StageBuilder``, and may expose
per-env :term:`dynamic fields <dynamic field>` that a caller changes between or inside
CUDA graph replays without recording anything again.

Two related forms have pages of their own:

- a **gate generator** writes plan-view gate anchors that the gate pipeline orders,
  spreads, lifts and frames: :ref:`writing-a-gate-generator`. A track generator also
  makes gates with no code of its own: the gates :term:`preset` (``presets.gates``)
  places arc-length stations on its centerline (:ref:`pipeline-stations`);
- a ``GeneratorSpec`` wraps two callables for a simple fixed-shape track generator.
  The quick start below shows it; :ref:`plugin-abis` describes its contract.

Centerlines that come from elsewhere (a dataset, a learned model, a hand-drawn track)
need no generator code: the built-in ``external`` generator reads them from device arrays
you fill (:ref:`ingesting-centerlines`).

A plugin ships as its own pip-installable package. It imports only public modules
(``track_gen.plugin``, ``track_gen.exec``, ``track_gen.ops``), it does not edit TrackGen,
and TrackGen finds it through a Python entry point. The shipped generators register
through a ``register(reg)`` function with the same signature as the one a third party
writes. The classes and functions used on this page are listed in :ref:`reference-plugin`
and :ref:`reference-ops`.

Quick start: a Stage generator
------------------------------

A plugin needs three pieces: an entry point in the ``track_gen.plugins`` group, the
``register(reg)`` function it names, and the generators that function adds.

**1. Declare the entry point** in the plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."track_gen.plugins"]
   acme = "acme_tracks:register"

The value names the callable TrackGen calls with the registry. The entry-point name
(``acme``) identifies the plugin in warnings and in ``registry.quarantined``; name it
after the prefix of your generator names (:ref:`plugin-names`).

**2. Write the generator and** ``register`` (``acme_tracks/__init__.py``). This one draws
a per-env phase from the :term:`seed` and writes a wobbly circle with two parameters:
``lobes``, a *static* parameter (one host value, fixed for the generator's lifetime), and
``amp``, a *dynamic* one (one value per env, read on the device and changeable at run
time). It then uses ``track_gen.ops`` (:ref:`generator-ops`) to centre the loop and scale
it to the size the built-ins use:

.. snippet: group=wobble register=register

.. code-block:: python

   import dataclasses

   import warp as wp

   from track_gen import ops, plugin
   from track_gen.ops.funcs import hash32

   track_gen_api = 1  # optional: the plugin API version this package targets


   @wp.kernel
   def _wobble_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
                 amp: wp.array(dtype=wp.float32), lobes: int, n: int,
                 xy: wp.array(dtype=wp.vec2f)):
       t = wp.tid()                                  # one thread per point: t = e * n + i
       e = t // n
       key = hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)
       state = wp.rand_init(wp.int32(key))           # this env's stream for this salt
       phase = 6.2831853 * wp.randf(state)
       theta = 6.2831853 * float(t - e * n) / float(n)
       r = 1.0 + amp[e] * wp.sin(float(lobes) * theta + phase)
       xy[t] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class WobbleParams(plugin.Params):
       lobes: int = plugin.static(3, lo=2, hi=12, doc="lobes per loop")
       amp: float = plugin.dynamic(0.2, lo=0.0, hi=0.4, doc="radial wobble amplitude",
                                   ui="slider")


   class Wobble(plugin.Stage):
       kind = "generator"
       name = "acme.wobble"
       output = "centerline"
       Params = WobbleParams
       writes = {"xy": "raw.xy", "valid": "gen.valid"}   # local port -> channel

       def build(self, b, params):
           # Runs once, at construction: allocate and bind everything here.
           n = b.raw                                     # points per env (num_points)
           xy = b.write("xy", wp.vec2f, per="raw")       # [E * n], owned by the pipeline
           valid = b.write("valid", wp.int32)            # [E]
           amp = b.param("amp")                          # [E] float32, read by pointer
           salt = b.salt("shape")
           lobes = int(params.lobes)                     # static: a host scalar
           extent = ops.constants.TARGET_EXTENT * b.scale

           def enqueue(ctx):
               # Runs once per eager call or recording: ctx launches only.
               ctx.launch(_wobble_k, b.E * n, [b.seed, salt, amp, lobes, n, xy])
               ops.normalize_extent(ctx, xy, n, extent)
               ops.fill(ctx, valid, 1)

           return enqueue


   def register(reg):
       reg.add(Wobble)  # the class, not an instance

``build`` runs once, when a ``TrackGenerator`` is constructed. It takes everything the
generator needs from the ``StageBuilder`` ``b`` (its output buffers, the device array of
``amp``, a salt for its random stream, the per-env seed) and returns ``enqueue``, the code
that runs for every :term:`generation`. ``enqueue`` emits device work through ``ctx``
only; see :ref:`generator-stage-contract`.

**3. Install the package and select the generator by name.** Nothing imports
``acme_tracks``: the registry discovers the entry point on its first lookup. Parameters
go in ``generator_params``, as a mapping or a ``WobbleParams``; the fields you leave out
keep their defaults:

.. code-block:: bash

   pip install -e ./acme-tracks

.. snippet: group=wobble

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   device = "cuda:0"  # or "cpu"
   spec = presets.track(num_envs=64, device=device, generator="acme.wobble",
                        generator_params={"lobes": 5})
   gen = TrackGenerator(spec)                  # construction runs the first generation
   track = gen.generate()                      # the same batch: env e, seed e

   gen.params["generator"].set("amp", 0.35)    # every env, from the next generation on
   track = gen.generate(advance=True)          # new tracks, with the new amp

.. snippet: after

   assert track.valid.numpy().all()
   assert gen.params["generator"].params.lobes == 5

The flat configuration selects it the same way:
``TrackGenConfig(generator="acme.wobble", generator_params={"lobes": 5}, ...)``
(:ref:`generating-ways`).

Entry points are read from installed package metadata, so an editable install is enough
but a bare source tree on ``PYTHONPATH`` is not. Without an install (a notebook, a quick
experiment), register the plugin yourself before constructing the generator:

.. snippet: skip=plugin-package

.. code-block:: python

   import acme_tracks
   from track_gen.plugin import registry

   acme_tracks.register(registry)

Skip this when the package is installed: its entry point has already registered the
generators, and adding a name again with a different object raises ``DuplicateSpecError``
(adding the identical object again is a no-op).

On the ``cpu`` device, Warp's first compile of a module that uses
``track_gen.ops.funcs.hash32`` (as ``Wobble`` does) prints ``-Wconstant-conversion``
compiler warnings for its ``uint32`` constants. They are harmless: the values round-trip,
and the CPU kernel matches ``track_gen.ops.hash32_host`` bit for bit.

.. _generator-spec-recipe:

Quick start: a ``GeneratorSpec`` generator
------------------------------------------

A ``GeneratorSpec`` wraps two callables: ``alloc_scratch`` allocates the generator's
buffers once, and ``generate`` writes into the buffers the pipeline passes on every call.
It is the supported simple track-plugin form (:ref:`plugin-abis` has its contract
and how to port one to a Stage). The same wobbly circle, with a fixed shape and no
parameters:

.. snippet: group=spec register=register

.. code-block:: python

   from types import SimpleNamespace

   import warp as wp

   from track_gen import ops
   from track_gen.exec import resolve
   from track_gen.plugin import GeneratorSpec

   SALT = 0x61636D65  # this plugin's own RNG stream (hard rule 5)
   track_gen_api = 1  # optional: the plugin API version this package targets


   @wp.kernel
   def _phase_k(seeds: wp.array(dtype=wp.int32), salt: int,
                phase: wp.array(dtype=wp.float32)):
       e = wp.tid()                              # one thread per env
       state = wp.rand_init(seeds[e], salt)
       phase[e] = wp.randf(state) * 6.2831853


   @wp.kernel
   def _wobbly_circle_k(phase: wp.array(dtype=wp.float32), n: int,
                        out: wp.array(dtype=wp.vec2f)):
       t = wp.tid()                              # one thread per point: t = e * n + i
       e = t // n
       theta = 6.2831853 * float(t - e * n) / float(n)
       r = 1.0 + 0.2 * wp.sin(3.0 * theta + phase[e])
       out[t] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


   def alloc_scratch(config):
       E = int(config.num_envs)
       return SimpleNamespace(
           phase=wp.empty(E, dtype=wp.float32, device=str(config.device)))


   def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, *, ctx):
       ctx = resolve(ctx, out_centerline.device)  # the facades always pass ctx
       E, N = int(config.num_envs), int(config.num_points)
       ctx.launch(_phase_k, E, [seeds_wp, SALT, scratch.phase])
       ctx.launch(_wobbly_circle_k, E * N, [scratch.phase, N, out_centerline])
       # Centre each loop and scale its longest side to the size the built-ins use.
       ops.normalize_extent(ctx, out_centerline, N,
                            ops.constants.TARGET_EXTENT * float(config.scale))
       ops.fill(ctx, out_valid_wp, 1)


   def register(reg):
       reg.add(GeneratorSpec(
           name="acme.wobbly",
           alloc_scratch=alloc_scratch,
           generate=generate,
       ))

.. snippet: after

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   _gen = TrackGenerator(presets.track(num_envs=8, device="cpu", generator="acme.wobbly"))
   assert _gen.generate().valid.numpy().all()

It ships, registers and is selected exactly like the Stage
(``presets.track(generator="acme.wobbly", ...)``). Its knobs can be a ``Params`` class
too, static fields only, passed as ``GeneratorSpec(..., params=...)``
(:ref:`generator-spec-params`).

Choose the next level of detail
-------------------------------

The recipes above are complete fixed-shape and dynamic examples. Use
:ref:`generator-params` for typed fields, :ref:`generator-stage-contract` for Stage
ports/build/enqueue, :ref:`generator-attributes` for data carried along the route,
:ref:`generator-runtime-params` for live arrays and streams, and
:ref:`generator-ops` for reusable geometric operations.

For existing geometry, the shorter :ref:`importing-a-track` recipe is enough.
The full :ref:`generator-contracts` retains input, capture, ownership, bounded-loop and
conformance details.

.. toctree::
   :maxdepth: 1

   /reference/generator-contracts
