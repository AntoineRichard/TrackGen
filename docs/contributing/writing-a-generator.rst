Writing a Generator
===================

A **track generator** produces the initial closed centerline that the pipeline then
resamples, relaxes (XPBD), and inflates. A **gate generator** produces the ordered
plan-view gate anchors that the gate pipeline then spreads, elevates, lifts to 3D, and
frames. Both are plugins added to the registry of ``track_gen.plugin``. A track generator
takes one of two forms:

- a ``Stage`` subclass (the recommended form). It declares its own parameters as a
  ``Params`` dataclass, allocates its buffers through a ``StageBuilder``, and may expose
  per-env *dynamic* parameters and inputs that a caller changes between or inside CUDA
  graph replays without re-capturing;
- a ``GeneratorSpec``: a few callables (``alloc_scratch``, ``generate``) and flags. It is
  still fully supported, and it can declare a ``Params`` class too (static fields only).

Gate generators are ``GateGeneratorSpec`` objects in this release: gate Stages arrive with
roadmap PR11.

A plugin ships as its own pip-installable package. It imports only public modules
(``track_gen.plugin``, ``track_gen.exec``, ``track_gen.ops``), it does not edit TrackGen,
and TrackGen finds it through a Python entry point. The shipped generators register through
a ``register(reg)`` function with the same signature as the one a third party writes.
Once it runs, ``track_gen.testing`` checks it against the contract on this page in one
call: see :doc:`testing-a-plugin`.

Quick start: a Stage generator
------------------------------

A plugin needs three pieces: an entry point in the ``track_gen.plugins`` group, the
``register(reg)`` function it names, and the generators that function adds.

**1. Declare the entry point** in the plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."track_gen.plugins"]
   acme = "acme_tracks:register"

The value names the callable TrackGen calls with the registry. The entry-point name
(``acme``) identifies the plugin in warnings and in ``registry.quarantined``. Name it after
the prefix of your generator names (``acme`` for ``acme.*``): if the plugin ever fails to
load, a request for one of its names then reports the plugin's error (see
:ref:`how a quarantined plugin is matched <plugin-quarantine>`).

**2. Write the generator and** ``register`` (``acme_tracks/__init__.py``). This one draws a
per-env phase from the seed and writes a wobbly circle with two parameters: ``lobes``, a
*static* parameter (one host value, fixed for the generator's lifetime), and ``amp``, a
*dynamic* one (one value per env, read on the device and changeable at run time). It then
uses ``track_gen.ops`` (see :ref:`generator-ops`) to centre the loop and scale it to the
size the built-ins use:

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
that runs for every generation. ``enqueue`` emits device work through ``ctx`` only; see
:ref:`generator-stage-contract`.

**3. Install the package and select the generator by name.** Nothing imports
``acme_tracks``: the registry discovers the entry point on its first lookup. Parameters go
in ``generator_params``, as a mapping or a ``WobbleParams``; the fields you leave out keep
their defaults:

.. code-block:: bash

   pip install -e ./acme-tracks

.. code-block:: python

   from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator

   config = TrackGenConfig(generator="acme.wobble", num_envs=1024, device="cuda:0",
                           generator_params={"lobes": 5})
   rng = PerEnvSeededRNG(seeds=0, num_envs=1024, device="cuda:0")
   gen = TrackGenerator(config, rng)
   track = gen.generate()

   gen.params["generator"].set("amp", 0.35)   # every env, from the next generation on
   track = gen.generate(advance=True)        # new tracks, with the new amp

Entry points are read from installed package metadata, so an editable install is enough
but a bare source tree on ``PYTHONPATH`` is not. Without an install (a notebook, a quick
experiment), register the plugin yourself before constructing the generator:

.. code-block:: python

   import acme_tracks
   from track_gen.plugin import registry

   acme_tracks.register(registry)

Skip this when the package is installed: its entry point has already registered the
generators, and adding a name again with a different object raises ``DuplicateSpecError``
(adding the identical object again is a no-op).

Quick start: a ``GeneratorSpec`` generator
------------------------------------------

A ``GeneratorSpec`` wraps two callables: ``alloc_scratch`` allocates the generator's
buffers once, and ``generate`` writes into the buffers the pipeline passes on every call.
This is the form every plugin used before ``Stage`` existed, and the form gate generators
use (see :ref:`gate-generator-contract`). The same wobbly circle, with a fixed shape and
no parameters:

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


   def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None):
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
           accepts_ctx=True,  # generate takes ctx= and emits all its work through it
       ))

It ships, registers and is selected exactly like the Stage:
``TrackGenConfig(generator="acme.wobbly", ...)``. Its knobs can be a ``Params`` class
too, passed as ``GeneratorSpec(..., params=...)`` (see :ref:`generator-spec-params`).

.. _generator-params:

Parameters: ``Params``
----------------------

A generator declares its own parameters as a ``Params`` class: a frozen, keyword-only
dataclass deriving from ``track_gen.plugin.Params``. Users set the values with
``TrackGenConfig(generator=..., generator_params=...)``, so a plugin does not subclass
``TrackGenConfig`` to add knobs. A ``Stage`` names the class in its ``Params`` attribute; a
``GeneratorSpec`` or ``GateGeneratorSpec`` passes it as ``params=``.

Each field is made with ``static`` or ``dynamic`` (a plain dataclass field counts as
static):

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Made with
     - ``static(default, *, lo=None, hi=None, choices=None, doc="", ui=None)``
     - ``dynamic(default, *, lo=None, hi=None, doc="", ui=None)``
   * - The stage gets
     - A host value, on the ``params`` object ``build`` receives
       (``config.generator_params`` for a ``GeneratorSpec``).
     - An ``[E]`` device array, ``b.param(field)``, that its kernels read by pointer.
   * - Types
     - Any value: ``int``, ``float``, ``bool``, ``str``, tuples, ``None``, enums.
     - ``float`` (a ``wp.float32`` array), ``int`` or ``bool`` (a ``wp.int32`` array).
   * - It changes
     - Never: build a new generator to change it.
     - Per env, at run time, between or inside graph replays, with no re-capture
       (see :ref:`generator-runtime-params`).
   * - Use it for
     - Shapes, buffer sizes, iteration counts, enums, anything a Python branch reads.
     - Continuous knobs a curriculum or a domain randomisation varies per env.
   * - Declared by
     - A ``Stage``, a ``GeneratorSpec`` or a ``GateGeneratorSpec``.
     - A ``Stage`` only.

The keyword arguments describe the field. ``lo`` and ``hi`` are inclusive bounds: a value
with ``v < lo`` or ``v > hi`` raises ``ValueError`` at construction (a NaN passes, as with
any comparison), and so does a scalar ``set`` outside them. A field annotated ``int``,
``float`` or ``bool`` must hold a value of that type, static or dynamic: an integral (not
a bool) for ``int``, any real (not a bool) for ``float``, a bool (or ``0`` / ``1``) for
``bool``; anything else, such as the string ``"70"``, raises ``ValueError``. ``choices`` lists the values a
static field accepts. ``doc`` is a one-line description and ``ui`` a free-form hint for
tools: a widget name such as ``"slider"``, or a mapping with ``widget``, ``label``,
``step`` or ``group`` keys. All of them appear in the generator's
:ref:`schema <generator-describe>`.

``registry.add`` checks the class before it registers the generator (a plugin that fails a
check is quarantined under ``warn``):

- it derives from ``Params`` and is decorated with
  ``@dataclasses.dataclass(frozen=True, kw_only=True)`` itself (an undecorated subclass is
  rejected);
- every field has a default, inside its own ``lo``/``hi``/``choices``, so the class can
  be built with no arguments and ``from_legacy`` always works;
- no field is named ``validate``, ``from_legacy``, ``to_dict``, ``from_dict``,
  ``replace``, ``schema``, ``fields`` or ``describe``, or starts with ``_`` (reserved for
  Params helpers);
- a dynamic field is annotated ``float``, ``int`` or ``bool``;
- the ``params=`` class of a ``GeneratorSpec`` or ``GateGeneratorSpec`` has static fields
  only: a dynamic field needs a Stage's ``b.param``.

Two methods can be overridden:

- ``validate(self, layout)`` checks the object's own fields and raises ``ValueError``. It
  runs when the generator is constructed, and only for the selected generator, before the
  generic ``lo``/``hi``/``choices`` checks. ``layout`` is the run's ``Layout``
  (``num_envs``, ``stride``, ``device``, ``closed``). The default checks nothing.
- ``from_legacy(cls, cfg)``, a classmethod, gives the instance used when
  ``generator_params`` is ``None``. The default returns ``cls()``. A plugin that used to
  read its knobs from a ``TrackGenConfig`` subclass overrides it, so the configs its users
  already have keep their meaning:

.. code-block:: python

   @classmethod
   def from_legacy(cls, cfg):
       return cls(lobes=getattr(cfg, "acme_lobes", 3))

When a ``TrackGenerator`` or ``GateGenerator`` (or a ``Course``) is constructed, it
resolves the parameters before it allocates anything:

1. The registry looks the generator up, so an unknown name is reported first.
2. ``config.generator_params`` becomes an instance of the class. ``None`` gives
   ``from_legacy(config)``. An instance of the class is used as it is. A mapping gives
   ``WobbleParams(**mapping)`` (for the quick start): keys left out take the class
   defaults, and an unknown key raises ``ValueError`` listing the valid fields. Any other
   type raises ``TypeError`` naming the expected class. A mapping's values are converted
   by each field's annotation: a list becomes a tuple for a ``tuple[...]`` field (also
   inside ``T | None``), and a value becomes the member for an ``Enum`` field. A mapping
   shaped like the generator's :ref:`schema <generator-describe>` (from JSON, YAML or
   Hydra) therefore gives the same, hashable instance.
3. ``validate(layout)`` runs, then the ``lo``/``hi``/``choices`` checks and the type checks
   of the dynamic fields.

``gen.params["generator"].params`` is then the resolved instance, and
``gen.config.generator_params`` the value as passed, a mapping normalised to the instance.
A generator without a Params class takes no ``generator_params``: a non-empty mapping
raises ``ValueError``. The configuration reference covers the built-ins' classes and how
``generator_params`` combines with their flat fields: :ref:`generator-params-config`.

.. _generator-stage-contract:

The ``Stage`` contract
----------------------

Register the class (``reg.add(Wobble)``), not an instance. The orchestrator creates one
instance per generator, with no arguments, and calls ``capacity`` and ``build`` on it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Class attribute
     - Meaning
   * - ``kind``
     - ``"generator"``. ``registry.add`` reads the kind from it.
   * - ``name``
     - The ``TrackGenConfig.generator`` string that selects it. Dotted for third parties
       (see :ref:`plugin-names`).
   * - ``output``
     - ``"centerline"``, required for a generator. ``"anchors"`` (a gate generator)
       arrives with roadmap PR11: register a ``GateGeneratorSpec`` until then.
   * - ``Params``
     - The stage's Params class. The default ``NoParams`` takes no ``generator_params``.
   * - ``writes``
     - ``{local port: channel}``. A generator maps one port to ``"raw.xy"`` (required) and
       may map one to ``"gen.valid"``.
   * - ``capture``
     - A ``CaptureClass``, ``STATIC`` by default (see :ref:`stage-capture-classes`).
       ``BOUNDED`` is rejected until roadmap PR12.
   * - ``topology``
     - ``"closed"`` (default) or ``"any"``. Open tracks are not supported in 0.x.
   * - ``reads``, ``provides``
     - Must stay empty: read ports arrive with roadmap PR10, ``provides`` with PR11 and
       PR13.
   * - ``api_version``
     - The plugin API version the stage targets (default ``1``). A version above
       ``track_gen.plugin.API_VERSION`` is rejected.

And three methods:

- ``build(self, b, params) -> enqueue`` (abstract) allocates everything through ``b``,
  once, and returns ``enqueue(ctx)``.
- ``capacity(self, params, layout)`` returns the buffer sizes the stage needs. A generator
  may return ``{"raw": n}``, its ``"raw.xy"`` points per env (an integer ``n >= 3``, a
  numpy integer included); the default ``{}`` means ``TrackGenConfig.num_points``. Any
  other key raises ``ValueError``. The pipeline resamples the raw loop by arc length, so
  ``n`` need not be ``num_points``.
- ``capture_class(cls, params)``, a classmethod, returns the capture class for ``params``.
  The default returns ``capture``; override it when the capture class depends on a
  parameter. Returning ``BOUNDED`` raises ``ValueError`` until roadmap PR12.

All three receive the params with their **static fields only**: reading a dynamic field
there raises ``AttributeError`` ("read it on the device through ``b.param('amp')``"). A
host scalar computed from a dynamic field would be baked into the launches (and into any
recorded graph), so a later ``set`` would silently do nothing. ``isinstance(params,
WobbleParams)`` still holds; ``gen.params["generator"].params`` is the full instance.

``registry.add`` checks the declaration before it registers the class, and raises
``TypeError`` or ``ValueError`` (a plugin is then quarantined under ``warn``) for an
instance, an abstract class, a missing or mismatched ``kind``, ``kind="gate_generator"``,
an ``api_version`` above ``API_VERSION``, a ``capture`` that is not a ``CaptureClass`` or
is ``BOUNDED``, a Params class that breaks the rules above, an ``output`` other than
``"centerline"``, a ``writes`` without ``"raw.xy"`` or with another channel (or two ports
on one channel), a non-empty ``reads`` or ``provides``, or another ``topology``. A Stage of
any other kind (``"relax"``, ``"elevation"``, ...) is rejected: other kinds arrive with
roadmap PR10, which will check their ports, channels and topology.

``build`` and ``enqueue``
~~~~~~~~~~~~~~~~~~~~~~~~~

``build`` runs once, at construction, before the warm-up pass. Allocate everything there,
through ``b`` only, and compute the host scalars ``enqueue`` needs from ``params``; once
``build`` has returned, the builder refuses to allocate (``RuntimeError``).
``enqueue(ctx)`` runs when a ``GeneratorSpec``'s ``generate`` would (see
`When your code runs`_): once at the warm-up, then once per eager call and once per
recording. Replays do not run Python.

``enqueue`` follows the :ref:`hard rules <generator-hard-rules>`: device work only through
``ctx.launch``, ``ctx.copy``, ``ctx.fill`` and ``track_gen.ops``; no allocation; no host
sync; no branch on device data or on ``ctx.capturing`` (a ``HOST`` stage may sync). Per
env, its output must depend only on ``b.seed[e]``, env ``e``'s dynamic parameters and
inputs, and the static parameters: carry no state from one generation to the next. A
masked regeneration recomputes every env and relies on the envs it leaves out reproducing
their current output bit for bit (see :doc:`/how-it-works/cuda-graph`). A scratch buffer
is zeroed once, at allocation: write it before you read it.

.. _stage-builder:

The ``StageBuilder``
~~~~~~~~~~~~~~~~~~~~

The orchestrator creates the builder and hands it to ``build``; plugins never construct
one. Its attributes:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Attribute
     - Meaning
   * - ``E``
     - The number of envs.
   * - ``layout``
     - The run's ``Layout``: ``num_envs``, ``stride`` (``N_max`` for tracks), ``device``
       (canonical, e.g. ``"cuda:0"``) and ``closed``.
   * - ``device``
     - The ``wp.Device``.
   * - ``stage_id``
     - ``"generator"``.
   * - ``scale``
     - ``TrackGenConfig.scale``, a core field rather than a parameter.
   * - ``seed``
     - ``[E]`` int32: each env's effective seed (its seed and epoch folded), written at
       the start of every generation. Read it in ``enqueue``.
   * - ``raw``
     - The points per env of ``"raw.xy"`` (``capacity()["raw"]``, else ``num_points``). A
       generator-stage attribute: it stays in later releases.

Call its methods in ``build`` (``write``, ``scratch`` and ``input`` allocate, and raise
``RuntimeError`` once ``build`` has returned):

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Method
     - What it returns
   * - ``write(port, dtype, *, per="env")``
     - The output buffer of ``writes[port]``: ``[E * raw]`` ``wp.vec2f`` with
       ``per="raw"`` for ``"raw.xy"``, ``[E]`` int32 with ``per="env"`` for
       ``"gen.valid"``. The pipeline owns it and reads it directly (no copy). Call it once
       per port. An undeclared port, another channel or a second call raises
       ``ValueError``, a wrong ``dtype`` or ``per`` ``TypeError``, and ``interp=`` /
       ``export=`` ``NotImplementedError`` (roadmap PR13).
   * - ``scratch(name, dtype, shape)``
     - A private zero-initialised buffer, named ``"scratch:generator/<name>"``. A second
       buffer with the same name raises ``ValueError``.
   * - ``param(field)``
     - The ``[E]`` working array (``float32`` or ``int32``) of a dynamic field, latched
       from the live array at every generation (see :ref:`generator-runtime-params`).
       **Read-only** for the stage: never write it, not even as scratch. A static field
       raises ``KeyError`` (read it from ``params``), and so does an unknown one.
   * - ``input(name, dtype, default=0)``
     - A per-env device input that is not a parameter (a difficulty level your training
       loop computes, say): a live ``[E]`` array published as ``gen.inputs[name]`` and
       filled with ``default``, and the working copy this returns, latched like a dynamic
       parameter and **read-only** for the stage likewise. ``dtype`` is ``wp.float32`` or
       ``wp.int32``; a name that exists already (``"regen_mask"`` included) raises
       ``ValueError``.
   * - ``salt(stream)``
     - A ``uint32`` salt for one random stream (see :ref:`stage-salts`).

When ``build`` returns, ``"raw.xy"`` must have been written (``ValueError`` otherwise) and
the return value must be callable (``TypeError`` otherwise). A generator that never writes
``"gen.valid"`` gets it filled with ``1`` once, at construction. One that writes it must
write it every generation: ``0`` rejects the env outright, since the post-relax validity
gate ANDs the flag in.

.. _stage-salts:

Random streams: ``b.salt``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``b.salt(stream)`` returns ``hash32_host(fnv1a_32(f"{stage_id}/{stream}".encode()))``, an
unsigned 32-bit integer computed from the stage id and the stream name only (the stage id
is ``"generator"`` for every generator in this release). The formula is frozen for plugin
API version 1, so a plugin's outputs stay reproducible across TrackGen releases. Key each
env's stream on the effective seed and the salt, as the quick start does:

.. code-block:: python

   key = hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)
   state = wp.rand_init(wp.int32(key))

``hash32`` is ``track_gen.ops.funcs.hash32``. Unlike the built-ins' linear
``seed * salt + offset`` keying (see :ref:`ops-rng-keying`), this hashed keying does not
collapse at seed 0: streams with different salts differ for every seed. Use one stream
name per independent draw. Asking for the same name again returns the same salt, and two
names whose salts collide raise ``ValueError``.

.. _stage-capture-classes:

Capture classes
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``CaptureClass``
     - Behaviour
   * - ``STATIC`` (default)
     - A fixed launch sequence, captured into the generator's own CUDA graph and recorded
       into callers' graphs: a ``GeneratorSpec`` with ``capturable=True``.
   * - ``BOUNDED``
     - An iteration with a static bound, through ``ctx.bounded_loop``. Reserved: both
       arrive with roadmap PR12, and until then a stage that declares or returns
       ``BOUNDED`` is rejected. Declare ``STATIC``.
   * - ``HOST``
     - Host readbacks or host control flow: runs eagerly on every call, on CUDA too, and
       may sync (a ``GeneratorSpec`` with ``capturable=False``, see
       :ref:`generator-capture-opt-out`). Recording it into a caller's graph raises
       ``NotCapturableError`` before any launch.

``registry.describe(kind, name).capturable`` is ``False`` for a stage whose
``capture_class``, called with the Params defaults, returns ``HOST``; it reflects the
defaults only, so a stage whose capture class depends on its params can differ for other
values (``describe().capture`` reports the class for the defaults).

.. _generator-runtime-params:

Runtime parameters and inputs
-----------------------------

``TrackGenerator.params`` (and ``GateGenerator.params``) is a ``ParamTable``: a read-only
mapping from stage id to that stage's ``StageParams``. It holds one stage in this release,
``"generator"``; later releases key every pipeline stage the same way.

- ``gen.params["generator"].params`` is the resolved Params instance: the static values and
  the *initial* dynamic values. It is ``None`` for a ``GeneratorSpec`` without ``params=``.
- ``gen.params["generator"]`` maps each **dynamic** field to its *live* ``[E]`` device
  array. The pointer is stable for the generator's lifetime, so your own kernels can write
  it, inside your own CUDA graph too. A static field raises ``KeyError`` ("build a new
  generator to change it"). The built-in generators' fields are all static.
- ``gen.params["generator"].set(field, value, *, stream=None)``, or
  ``gen.params.set("generator", field, value, stream=...)``, writes the live array in
  place. A host scalar is checked against ``lo``/``hi`` on the host, then broadcast with
  one fill. An ``[E]`` contiguous ``wp.array`` of the field's dtype on the generator's
  device is copied with one copy, with no range check and no host sync. Either way it is
  an enqueue on the resolved stream (``stream=``, then the ambient scope, the generator's
  constructor stream, torch's capture stream and Warp's current stream) and never
  allocates. Inside a capture it is recorded: a scalar is baked into the caller's graph,
  an array is read at each replay. A wrong shape, dtype or device, a value that is
  neither a scalar nor a ``wp.array``, or a scalar out of bounds raises ``ValueError``
  before any launch.

A write takes effect at the next generation. Each generation first *latches* the live
arrays into the stage's working copies (the arrays ``b.param`` and ``b.input`` returned)
for the envs it regenerates: every env, or only those with ``mask[e] != 0`` under
``generate(advance=True, mask=m)``. An env that a masked regeneration leaves out therefore
keeps the value its current track was built with, and is recomputed bit for bit. The latch
costs one small launch per dynamic field or input per generation, for Stage generators
only. It is an implementation detail: the stage only ever reads the arrays ``b.param``
and ``b.input`` returned, and a later release may read the live arrays directly.

.. code-block:: python

   import warp as wp

   amp = gen.params["generator"]["amp"]           # [E] float32, stable pointer
   gen.params["generator"].set("amp", 0.3)        # every env
   gen.params["generator"].set("amp", per_env)    # an [E] wp.float32 array

   # In your own graph: a curriculum kernel writes amp, then the generator runs.
   with wp.ScopedCapture(stream=s) as capture:
       wp.launch(curriculum_k, dim=E, inputs=[success_rate, amp], stream=s)
       gen.generate(advance=True, stream=s)
   wp.capture_launch(capture.graph, stream=s)     # every replay reads amp anew

**Inputs.** ``gen.inputs`` lists ``"regen_mask"`` and every ``b.input`` the generator
declared. Each is a live ``[E]`` array with a stable pointer, latched like a dynamic
parameter: write it in place, as you write ``regen_mask``, and the next generation reads
it.

A ``Course`` has no parameter table of its own: use ``course.generator.params`` and
``course.generator.inputs``.

.. _generator-describe:

Describing a generator: ``describe()``
--------------------------------------

``registry.describe(kind, name).params_schema`` is the JSON schema (draft 2020-12) of the
generator's Params class, a plain dict that ``json.dumps`` accepts; it is ``{}`` for a
generator without parameters (no Params class, or ``NoParams``). Tools build their controls from it instead of hard-coding each
generator's knobs (``viz/param_explorer.py`` shows a plugin generator's parameters this
way). The quick start's ``WobbleParams`` gives:

.. code-block:: json

   {"$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WobbleParams", "type": "object", "additionalProperties": false,
    "properties": {
      "lobes": {"type": "integer", "default": 3, "minimum": 2, "maximum": 12,
                "description": "lobes per loop", "x-static": true},
      "amp": {"type": "number", "default": 0.2, "minimum": 0.0, "maximum": 0.4,
              "description": "radial wobble amplitude", "x-dynamic": true,
              "x-dtype": "float32", "x-ui": "slider"}}}

Each field's ``type`` comes from its annotation:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Annotation
     - JSON schema
   * - ``bool``, ``int``, ``float``, ``str``
     - ``"boolean"``, ``"integer"``, ``"number"``, ``"string"``
   * - ``tuple[T, ...]``, ``list[T]``
     - ``"array"`` with ``"items"``
   * - ``tuple[T, T]``
     - ``"array"`` with ``"minItems"`` and ``"maxItems"`` of 2
   * - ``T | None``
     - ``"type": [T, "null"]``
   * - an ``Enum``
     - ``"enum"`` of its values
   * - anything else
     - no ``"type"``

``default`` is the field's default (tuples as lists), ``minimum`` and ``maximum`` come from
``lo`` and ``hi``, ``enum`` from ``choices`` and ``description`` from ``doc`` (left out
when empty). Exactly one of ``x-static`` and ``x-dynamic`` is ``true``; ``x-dtype`` is
given for dynamic fields, and ``x-ui`` when ``ui`` is set.

``describe()`` also reports ``capturable`` and ``accepts_ctx`` for both forms: a Stage
always accepts a ``ctx``, and is capturable unless its capture class is ``HOST``. For a
Stage it also reports ``capture`` (the ``CaptureClass`` for the Params defaults),
``output``, ``writes`` and ``provides``; they are ``None``, ``None``, ``{}`` and empty for
a ``GeneratorSpec``. Code that walks the registry should read these from ``describe()``:
a Stage class has no ``capturable`` or ``alloc_scratch`` attribute (and
``registry.get`` returns the class).

The ``GeneratorSpec`` contract
------------------------------

What you implement
~~~~~~~~~~~~~~~~~~

A ``GeneratorSpec`` has six fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``name``
     - The ``TrackGenConfig.generator`` string that selects it. Dotted for third parties
       (see :ref:`plugin-names`).
   * - ``alloc_scratch``
     - ``(config) -> scratch``, below.
   * - ``generate``
     - ``(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None) -> None``,
       below.
   * - ``capturable``
     - ``True`` (default): ``generate`` obeys the :ref:`hard rules
       <generator-hard-rules>` and is captured into CUDA graphs. ``False``: see
       :ref:`generator-capture-opt-out`.
   * - ``accepts_ctx``
     - ``True``: ``generate`` takes a ``ctx=`` keyword and emits all its device work
       through it (see :ref:`generator-streams`). The default ``False`` exists for plugins
       written before it; set it to ``True``.
   * - ``params``
     - Optional: the generator's ``Params`` class, static fields only (see
       :ref:`generator-spec-params`). The default ``None`` takes no
       ``generator_params``.

``alloc_scratch(config) -> scratch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allocate your generator's private working buffers **once**.  Use fixed shapes
derived from ``config`` (e.g. ``num_envs``, ``max_num_points``,
``num_points_per_segment``, ``num_points``), all on ``str(config.device)``.
Return any object exposing the buffers your ``generate`` function uses.

Do **not** allocate the output centerline or the validity array here — the
orchestrator owns those.

``generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None) -> None``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a closed centerline of ``config.num_points`` points per environment into
the output arrays in place.  Parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``seeds_wp``
     - ``[E]`` int32 ``wp.array`` — one base seed per environment (the effective seed
       derived from the env's seed and epoch, see :doc:`/how-it-works/cuda-graph`).
       Read-only.
   * - ``config``
     - The generator's own snapshot of the ``TrackGenConfig`` it was constructed with.
       With ``params=``, ``config.generator_params`` is the resolved, validated Params
       instance.
   * - ``out_centerline``
     - ``[E*num_points]`` ``wp.vec2f`` — write the closed centerline in place.
   * - ``out_valid_wp``
     - ``[E]`` int32 — current runtime generators fill this stage flag with
       ``1`` for every env.  Final geometric validity is decided later by the
       shared post-relax inflation validity gate.
   * - ``scratch``
     - The object your ``alloc_scratch`` returned.
   * - ``ctx``
     - The call's ``track_gen.exec.LaunchContext`` (one device, one stream), passed when
       the spec sets ``accepts_ctx=True``. Emit every launch, copy, and fill through it.

The loop is closed implicitly: do not repeat point 0 at the end. Size it to
``ops.constants.TARGET_EXTENT * config.scale`` (``1.44`` at scale 1) on its longest
bounding-box side: ``polar``, ``voronoi`` and ``checkpoint`` center each loop and scale it
to exactly that, ``bezier`` and ``hull`` land near it by construction of their sampling
grid (a longest side of about 1.0 to 1.8, not centered), and ``repulsive`` is larger
(about 2.2). The default ``half_width``, spacing and ``N_max`` are tuned for that size,
and a much larger loop needs more than ``N_max`` resampled points and fails as truncated.
``ops.normalize_extent(ctx, out_centerline, N, ops.constants.TARGET_EXTENT *
config.scale)`` does the sizing in one launch, as in the quick start.

.. _generator-hard-rules:

Hard rules
~~~~~~~~~~

Five rules must hold for every generator whose ``GeneratorSpec`` uses the default
``capturable=True``, and for the ``enqueue`` of every Stage whose capture class is not
``HOST``:

1. **Pure Warp kernels only.**  Launch through ``ctx.launch``; one env per row.  No
   torch code: TrackGen does not depend on torch.
2. **Zero dynamic allocation inside** ``generate``.  All buffers come from
   ``alloc_scratch`` (for a Stage, from the ``StageBuilder`` in ``build``).
   ``track_gen.ops`` never allocates: you pass it the working buffers, allocated there.
3. **CUDA-graph capturable.**  No host sync, no host-side retry loop
   conditioned on generated data, no per-env Python branching inside
   ``generate``.
4. **Fixed bounds for every loop and buffer.**  Graph capture requires static
   shapes.
5. **Deterministic in** ``(per-env seed, config)``.  Draw every random number with
   the Warp RNG from ``seeds_wp[e]``, salted with a constant of your own so your
   stream is not correlated with the built-ins' streams for the same seed:
   ``wp.rand_init(seeds[e], SALT)`` in your own kernels, and a ``(salt, offset)`` pair
   no built-in uses for the ``track_gen.ops`` samplers (see :ref:`ops-rng-keying`). A
   Stage takes its salts from ``b.salt`` (see :ref:`stage-salts`).
   CPU vs CUDA RNG results may differ, as elsewhere in the codebase.

The conformance kit checks these rules mechanically: ``check_generator`` fails a
generator that host-syncs, allocates, launches off the call's stream, changes its launch
sequence from one call to the next, or is not reproducible from its seeds (see
:doc:`testing-a-plugin`).

.. _generator-streams:

Streams: ``accepts_ctx``
~~~~~~~~~~~~~~~~~~~~~~~~

Set ``accepts_ctx=True`` and emit all device work through ``ctx``: ``ctx.launch(kernel,
dim, inputs)``, ``ctx.copy(dst, src)``, ``ctx.fill(arr, value)`` (or
``track_gen.ops.fill(ctx, arr, value)``). Every ``track_gen.ops`` op takes ``ctx`` as its
first argument; pass ``ctx=ctx`` to any other helper that takes it.
Never call ``arr.zero_()``, ``arr.fill_()``, or a ``wp.launch`` / ``wp.copy`` without
``stream=``: they target Warp's current stream, which need not be the stream the call
resolved (a side stream, or the stream torch is capturing). With ``accepts_ctx=True`` a
generator can be recorded into a caller's CUDA graph on any stream.

A spec that leaves ``accepts_ctx=False`` is a *legacy plugin*. It still runs, inside
``ctx.scope()``, so the library helpers it calls inherit the stream, but its own direct
``wp.launch`` calls go to Warp's current stream. Its self-graph is then captured behind a
process-wide lock, and recording it into a caller's graph requires the capture stream to be
Warp's current stream (``LegacyPluginStreamError`` otherwise); see
:doc:`/how-it-works/cuda-graph`.

.. _generator-capture-opt-out:

Opting out of capture: ``capturable=False``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GeneratorSpec`` also has a ``capturable: bool = True`` field. A generator may instead set
``capturable=False`` to opt out of rules 2 and 3: it may allocate per-call and use host-side
control flow — Python loops driving stage transitions, readbacks that branch on generated
data. ``TrackGenerator`` then runs it **eagerly** on CUDA every call — the same code path as
``cpu`` — instead of capturing it into a replayable graph; this is a supported, non-error
path, not a fallback for a broken generator. Rules 1 (pure Warp, no torch) and 4 (fixed-shape
buffers) still apply: only the *capture* of a static launch topology is waived, not the
fixed-shape scratch discipline. No shipped generator uses ``capturable=False`` today
(``GateGeneratorSpec`` carries the same flag). Before opting out, see how ``repulsive`` made a
host-driven loop capturable: its coarse-to-fine stage transitions, resample cadence and
stall-check cadence form a fixed schedule precomputed on the host from the config alone, and
the readback-driven early exit was replaced by an absorbing per-env freeze on device (frozen
envs are skipped by every kernel), so nothing is read back. Rule 5 still applies, with a
caveat for chaotic per-iteration methods: if such a generator accumulates its CUDA gradients
via ``atomic_add``, the varying float summation order can make a chaotically sensitive flow
only *statistically* reproducible on CUDA rather than bit-identical. Avoid it the way
``repulsive`` does — its gradient is hand-written analytic adjoints (a per-vertex gather, no
atomics), so it stays byte-identical run-to-run on both CPU and CUDA; see
``track_gen/_src/warp_generate_repulsive.py``'s module docstring for the full account.

When your code runs
~~~~~~~~~~~~~~~~~~~

Your ``generate`` body runs once, eagerly, when the ``TrackGenerator`` is constructed (the
warm-up pass that loads every kernel). After that it runs once per eager call (``cpu``) and
once per recording: into the generator's own graph on the first ``cuda`` call, or into a
caller's graph when ``generate()`` is called inside the caller's capture. Replays do not run
Python at all, so host scalars and Python state freeze at record time. A
``capturable=False`` generator runs once per call instead, and raises
``NotCapturableError`` under a caller's capture. See :doc:`/how-it-works/cuda-graph` for
the full execution model, including the stream rule for plugins that do not set
``accepts_ctx=True``.

What you do NOT have to guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple (non-self-intersecting) loop is preferred but not required.  Local
fallback is generator-specific: ``bezier``, ``hull``, and ``voronoi`` use
selected polygon-style rescues (``ops.self_intersections`` flags the envs whose smooth
curve crosses itself, and only those are rebuilt from straight segments and resampled
with ``ops.arc_resample(..., active=...)``; see :ref:`generator-ops`); ``checkpoint``
uses best-of-K plus an optional clip fallback; ``polar`` has no local fallback.
Residual bad geometry is handled by the common post-relax validity gate, which also ANDs
in your ``out_valid_wp`` flag: write ``0`` there to reject an env outright.

Output must be finite (no NaN) for generated environments.

.. _generator-spec-params:

Per-generator parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Declare your knobs as a :ref:`Params class <generator-params>` and pass it as
``GeneratorSpec(..., params=AcmeParams)``. Users set them with
``TrackGenConfig(generator="acme.wobbly", generator_params={...})``; ``alloc_scratch`` and
``generate`` read the resolved, validated instance as ``config.generator_params``
(``config.generator_params.lobes``), and ``describe()`` reports its schema. The class may
have static fields only: ``registry.add`` rejects a dynamic field, which needs a Stage's
``b.param``. The shared fields (``num_envs``, ``num_points``, ``scale``, ``device``, ...)
stay on the config.

Plugins written before Params existed read their knobs from a ``TrackGenConfig`` subclass
(``getattr(config, "acme_lobes", 3)``). That still works, but it is no longer needed:
move the knobs to a Params class and override ``from_legacy`` to read the old fields, so
the subclass configs your users already have resolve to the same values. Registry-wide
tools (``benchmarks/compare_generators.py``, ``viz/param_explorer.py``) build a plain
``TrackGenConfig`` for every registered generator, so a generator that needs a config
subclass must still fall back to defaults there; a Params class needs nothing more.

.. _gate-generator-contract:

Gate generators
---------------

A gate generator writes ordered plan-view gate anchors. The gate pipeline then spreads
them apart (``gate_solve_iters``), applies the Z profile, lifts them to 3D, derives
tangents and frames, and decides validity. ``GateGenConfig.generator`` selects one. A
``GateGeneratorSpec`` (also exported by ``track_gen.plugin``) has the ``name``,
``alloc_scratch``, ``capturable``, and ``accepts_ctx`` fields of a ``GeneratorSpec``, with
the same meaning, the same hard rules and the same stream rules, plus:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``generate``
     - ``(seeds_wp, config, out, scratch, ctx=None) -> None``. ``out.position`` is the
       ``[E*max_gates]`` ``wp.vec2f`` plan-view staging buffer (``max_gates`` is
       ``config.max_gates``): write env ``e``'s gates in order at ``e*max_gates + g``
       for ``g < count``, and NaN in every slot after them. ``out.count`` is ``[E]``
       int32: write each env's real gate count.
   * - ``max_gates``
     - ``(config) -> int``: the most gates ``generate`` can emit for ``config``.
       ``GateGenerator`` raises ``ValueError`` unless ``config.min_gates <=
       max_gates(config) <= config.max_gates``.
   * - ``supported_orderings``
     - ``frozenset`` of the ``GateGenConfig.gate_ordering`` values (``"ccw"``, ``"raw"``,
       ``"random_pairs"``) that ``generate`` honours. The pipeline does not reorder the
       gates: ``generate`` emits them in the configured order, and ``GateGenerator``
       raises ``ValueError`` for an ordering not in this set.
   * - ``params``
     - Optional: a static ``Params`` class, as for a ``GeneratorSpec``. Users set it with
       ``GateGenConfig.generator_params``, and ``generate``, ``alloc_scratch`` and
       ``max_gates`` read the resolved instance as ``config.generator_params``.

``GateGenerator`` checks ``supported_orderings`` and ``max_gates`` against the resolved
config, so ``max_gates`` sees the values ``generator_params`` sets. A gate generator cannot
be a ``Stage`` yet: ``registry.add`` rejects a Stage with ``kind="gate_generator"`` or
``output="anchors"`` until roadmap PR11.

Size the anchors like the ``polar``, ``voronoi`` and ``checkpoint`` gate generators:
centered, with the longest bounding-box side at ``ops.constants.TARGET_EXTENT *
config.scale`` (the ``bezier`` and ``hull`` gate generators keep their grid's size). The
example below is built from
``track_gen.ops`` the way the built-in ``bezier`` and ``hull`` gate generators are: draw a
per-env gate count and that many jittered grid points, sort them by angle straight into
``out.position`` (``angle_sort`` writes NaN into every slot past ``count[e]``, which is the
padding the contract asks for), then size them. It is added by the same ``register``
function:

.. code-block:: python

   from track_gen.plugin import GateGeneratorSpec

   MIN_GATES, MAX_GATES = 6, 10


   def alloc_gate_scratch(config):
       E, G = int(config.num_envs), int(config.max_gates)
       dev = str(config.device)
       return SimpleNamespace(
           points=wp.empty(E * MAX_GATES, dtype=wp.vec2f, device=dev),
           used=wp.empty(E * MAX_GATES, dtype=wp.int32, device=dev),
           keys=wp.empty(E * G, dtype=wp.float32, device=dev))


   def generate_scatter_gates(seeds_wp, config, out, scratch, ctx=None):
       ctx = resolve(ctx, out.position.device)
       G = int(config.max_gates)
       num_cells, cell = ops.grid_cells(0.1)             # a 5 x 5 grid
       ops.sample_counts(ctx, seeds_wp, SALT, 101, MIN_GATES, MAX_GATES, out.count)
       ops.sample_grid_points(ctx, seeds_wp, SALT, 102, MAX_GATES, num_cells, cell, 1.0,
                              scratch.used, scratch.points)
       # The built-ins' "ccw" order; slots count[e] .. G-1 are written NaN.
       ops.angle_sort(ctx, scratch.points, MAX_GATES, scratch.keys, out.position,
                      count=out.count, out_stride=G)
       ops.normalize_extent(ctx, out.position, G,
                            ops.constants.TARGET_EXTENT * float(config.scale),
                            count=out.count)


   def register(reg):
       ...  # reg.add(GeneratorSpec(name="acme.wobbly", ...)) as above
       reg.add(GateGeneratorSpec(
           name="acme.scatter",
           alloc_scratch=alloc_gate_scratch,
           generate=generate_scatter_gates,
           max_gates=lambda config: MAX_GATES,
           supported_orderings=frozenset({"ccw"}),   # angle_sort is the "ccw" order
           accepts_ctx=True,
       ))

``GateGenerator(GateGenConfig(generator="acme.scatter", ...), rng)`` then selects it
(``gate_ordering`` defaults to ``"ccw"``).

.. _generator-ops:

Building blocks: ``track_gen.ops``
----------------------------------

The built-in generators are assembled from the device ops in ``track_gen.ops``, and a
plugin can use the same ones. They take no config object: every size, salt and bound is
an explicit argument. Every op:

- takes the call's ``LaunchContext`` as its first argument and launches only through it,
  so its work lands on the call's stream and is recorded into whatever graph is capturing;
- never allocates, reads back or host-synchronizes, so it is safe inside a capturable
  ``generate``. Working buffers are arguments: allocate them in ``alloc_scratch``;
- works on the flat per-env layout of the outputs: env ``e`` owns
  ``[e*stride, (e+1)*stride)`` of a flat array, and per-env scalars are ``[E]`` arrays.
  ``E`` is ``src.shape[0] // stride`` for every op that takes an input ``src`` (the
  samplers take it from ``seeds``), and every other per-env array must be exactly
  ``[E]``;
- names its arguments the same way: ``src`` is the input point array and ``stride`` its
  points per env, ``num`` an exact number of output points per env and ``out_stride`` an
  output row capacity (NaN-padded past the env's count);
- takes ``count`` and the env mask ``active`` as keyword-only arguments. Where an op takes
  a ``count`` array, only the first ``count[e]`` points of a row are real (a count above
  the stride is clamped to it), and ops that produce points write ``(nan, nan)`` in the
  slots after them;
- validates its arguments on the host before launching anything. Data arrays must have
  the exact shape, the dtype and ``ctx.device``; scratch arrays need the dtype, the
  device and at least the stated length. A mismatch raises ``ValueError`` naming the
  argument, except that ``fill`` raises ``TypeError`` for an unsupported dtype and
  ``any_set`` for an argument that is not an ``int32`` ``wp.array``.

:doc:`/reference/api` documents each op's shapes and semantics. In outline:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Op
     - What it does
   * - ``sample_counts(ctx, seeds, salt, offset, lo, hi, out)``
     - One integer in ``[lo, hi]`` per env (how many corners, points or gates).
   * - ``sample_grid_points(ctx, seeds, salt, offset, num, num_cells, cell_size, scale,
       used, out)``, ``grid_cells(min_point_distance)``
     - ``num`` jittered points per env on distinct cells of a grid where possible (the
       ``bezier`` and ``hull`` sampler). The jitter is an absolute ``[-0.5, 0.5)``, so
       the points lie in ``[-0.5, (num_cells-1)*cell_size + 0.5) * scale`` and have no
       minimum spacing; distinct cells are best effort (8 redraws). ``grid_cells``
       derives the built-in grid from the ``min_point_distance`` config knob (in
       ``(0, 0.5]``) on the host. ``scale`` is a float or an ``[E]`` array.
   * - ``angle_sort(ctx, src, stride, keys, out, *, count, out_stride=None)``
     - The first ``count[e]`` points of each env, sorted by angle around their centroid
       (ascending ``atan2(x - cx, y - cy)``), NaN after them. This is the order the
       built-ins emit for ``gate_ordering="ccw"``; in a y-up frame it runs clockwise
       (``turning_number`` of the result is about ``-2*pi``).
   * - ``closed_catmull_rom(ctx, src, stride, per_seg, out, *, count=None,
       endpoint=False)``
     - ``per_seg`` samples per segment of the closed Catmull-Rom spline through each env's
       controls; with ``count``, segments past ``count[e]`` are NaN.
   * - ``closed_polyline(ctx, src, stride, per_seg, out, *, count=None, endpoint=False,
       active=None)``
     - The straight closed polyline through the controls, in the same layout. With
       ``active``, only envs with ``active[e] > 0`` are written: the fallback for envs
       whose smooth curve self-crossed.
   * - ``arc_resample(ctx, src, stride, num, scratch, out, count_out, *, active=None)``
     - ``stride`` points in, ``num`` out per env, evenly spaced by arc length along the
       closed loop. NaN points are dropped first, so a NaN-padded dense curve resamples
       cleanly; an env with fewer than two real points gets a NaN row and
       ``count_out[e] = 0``. With ``active``, every other env's row is left as it was.
   * - ``resample_uniform(ctx, src, stride, scratch, out, *, count)``
     - Count-aware evenly-spaced resample: ``count[e]`` points in and out, NaN after them.
   * - ``resample_constant_spacing(ctx, src, stride, spacing, out_stride, scratch, out,
       count_out, truncated_out)``
     - Points every ``spacing`` along each closed loop, up to ``out_stride``, with a
       per-env count and truncation flag.
   * - ``self_intersections(ctx, src, stride, out, *, count)``
     - Proper self-crossings of each closed polygon (``0`` for a simple loop).
   * - ``turning_number(ctx, src, stride, out, *, count)``,
       ``thickness(ctx, src, stride, band, out, *, count)``
     - Total signed turning (``+2*pi`` for a simple counter-clockwise loop) and the
       thickness (curvature and self-distance bound) of each closed polygon.
       ``thickness`` excludes vertex pairs within ``band[e]`` points of each other; the
       built-in validity gate uses ``band = max(1, round(2 * half_width / l0))`` for a
       mean segment length ``l0``.
   * - ``normalize_extent(ctx, src, stride, target_extent, *, count=None)``
     - Centres each env's bounding box and scales it so the longest side is
       ``target_extent``, in place; with ``count``, only the real points count and the
       slots after them are written NaN.
   * - ``fill(ctx, arr, value)``, ``any_set(ctx, mask, flag_out)``
     - A stream-explicit fill, and a device-side "is any ``mask[e]`` set" flag.
   * - ``check_array(name, arr, shape, dtype, device)``
     - The host-side validator the ops use, for your own wrappers.

Device functions for your own kernels are in ``track_gen.ops.funcs``: ``ccw``, ``cross2``,
``nan0``, ``safe_normalize2``, ``catmull_rom``, ``segments_cross`` and ``hash32``. They are
the functions the built-in kernels call: the same operations in the same order. On CPU a
kernel built from them reproduces the built-in arithmetic bit for bit. On CUDA, Warp
contracts multiply-adds into FMAs by default (``wp.config.fuse_fp``), and which pairs it
contracts depends on the code around the inlined call, so a differently structured kernel
can differ in the last bit. ``track_gen.ops.constants.TARGET_EXTENT`` is the size the
``polar``, ``voronoi`` and ``checkpoint`` generators normalise to.

Scratch
~~~~~~~

The resample ops take an ``ops.ArcScratch``, which holds the four buffers they work in.
Allocate it once in ``alloc_scratch`` with ``ops.ArcScratch.alloc(E, stride, device)``
for ``stride`` input points per env, or wrap buffers you already own (slices included) with
``ops.ArcScratch(real, seg, s, count_real)``. ``resample_uniform`` and
``resample_constant_spacing`` use only ``seg`` and ``s``: ``alloc(..., compact=False)``
leaves the other two ``None``. The remaining scratch arguments (``keys`` of
``angle_sort``, ``used`` of ``sample_grid_points``) are plain arrays of at least the
stated length.

A generator from ops
~~~~~~~~~~~~~~~~~~~~

This generator follows the ``hull`` recipe without its midpoint displacement: sampled
points sorted into a star-shaped polygon, a smooth spline through them, a straight
fallback for the envs whose spline crosses itself, and the built-ins' size. Each step is
one op, and it reuses ``SALT`` and the imports of the quick start:

.. code-block:: python

   K, PER_SEG = 12, 16            # controls per env, dense samples per segment
   M = K * PER_SEG                # dense points per env


   def alloc_star_scratch(config):
       E, dev = int(config.num_envs), str(config.device)
       return SimpleNamespace(
           count=wp.empty(E, dtype=wp.int32, device=dev),
           used=wp.empty(E * K, dtype=wp.int32, device=dev),
           points=wp.empty(E * K, dtype=wp.vec2f, device=dev),
           keys=wp.empty(E * K, dtype=wp.float32, device=dev),
           ctrl=wp.empty(E * K, dtype=wp.vec2f, device=dev),
           dense=wp.empty(E * M, dtype=wp.vec2f, device=dev),
           arc=ops.ArcScratch.alloc(E, M, dev),
           arc_count=wp.empty(E, dtype=wp.int32, device=dev),
           crossings=wp.empty(E, dtype=wp.int32, device=dev))


   def generate_star(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None):
       ctx = resolve(ctx, out_centerline.device)
       N = int(config.num_points)
       sc = scratch
       num_cells, cell = ops.grid_cells(0.1)
       # 1. 6..K jittered grid points per env, sorted by angle into a star-shaped polygon.
       ops.sample_counts(ctx, seeds_wp, SALT, 103, 6, K, sc.count)
       ops.sample_grid_points(ctx, seeds_wp, SALT, 104, K, num_cells, cell, 1.0,
                              sc.used, sc.points)
       ops.angle_sort(ctx, sc.points, K, sc.keys, sc.ctrl, count=sc.count)  # NaN past count
       # 2. Closed Catmull-Rom spline through the first count[e] controls (NaN past them).
       ops.closed_catmull_rom(ctx, sc.ctrl, K, PER_SEG, sc.dense, count=sc.count)
       # 3. Drop the NaN samples; N points evenly spaced by arc length.
       ops.arc_resample(ctx, sc.dense, M, N, sc.arc, out_centerline, sc.arc_count)
       # 4. Envs whose spline crosses itself fall back to the straight polygon.
       ops.self_intersections(ctx, out_centerline, N, sc.crossings, count=sc.arc_count)
       ops.closed_polyline(ctx, sc.ctrl, K, PER_SEG, sc.dense, count=sc.count,
                           active=sc.crossings)
       ops.arc_resample(ctx, sc.dense, M, N, sc.arc, out_centerline, sc.arc_count,
                        active=sc.crossings)
       # 5. The built-ins' size, then the stage flag.
       ops.normalize_extent(ctx, out_centerline, N,
                            ops.constants.TARGET_EXTENT * float(config.scale))
       ops.fill(ctx, out_valid_wp, 1)

It registers like the quick start's generator: ``reg.add(GeneratorSpec(name="acme.star",
alloc_scratch=alloc_star_scratch, generate=generate_star, accepts_ctx=True))``.

Step 4 rewrites only the rows of the envs that crossed: ``arc_resample(..., active=...)``
leaves every other row bit for bit and writes ``count_out[e] = 0`` for it. Use the first
call's ``count_out`` (as ``self_intersections`` does here) before the second call
overwrites it.

.. _ops-rng-keying:

RNG keying and the built-in salts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The samplers (``sample_counts``, ``sample_grid_points``) key env ``e``'s stream as
``wp.rand_init(seeds[e] * salt + offset)`` in int32 wrap-around arithmetic, the keying the
built-in generators use, and take ``salt`` and ``offset`` as arguments. Give each stream of
your generator its own pair, and pick salts no built-in uses. At seed 0 every key reduces
to ``rand_init(offset)`` whatever the salt, so two streams with the same offset coincide
for that seed: give your streams offsets that differ from each other and from the small
ones the built-ins use (the examples on this page use 101 to 104). The pairs already
taken:

.. list-table::
   :header-rows: 1
   :widths: 44 18 38

   * - Stream
     - Salt
     - Offset
   * - ``bezier`` corner count
     - 6151
     - ``attempt`` (0 in the built-in)
   * - ``bezier`` corner positions
     - 9781
     - ``attempt`` (0 in the built-in)
   * - ``bezier`` style sampling
     - 2741
     - 0
   * - ``hull`` point count
     - 5119
     - 1
   * - ``hull`` point positions
     - 7919
     - 1
   * - ``hull`` midpoint displacement
     - 3083
     - 1
   * - ``polar`` controls
     - 7919
     - 17
   * - ``voronoi`` sites
     - 8171
     - 19
   * - ``voronoi`` anchors
     - 3911
     - 31
   * - ``checkpoint`` candidate ``k``
     - 4099
     - ``k * 2741 + 1``
   * - ``repulsive`` obstacles
     - 6247
     - 23
   * - ``repulsive`` growth target
     - 5209
     - 41
   * - gate ``"random_pairs"`` ordering, pair ``p``
     - 3187
     - ``7919 + p * 104729``
   * - gate Z profile: uniform (gate ``i``), walk, noise (harmonic ``k``)
     - 15679, 15683, 15731
     - ``i``, 0, ``k``
   * - ``PerEnvSeededRNG`` states
     - 1
     - 0

The built-in gate generators reuse the matching track generator's streams. Hashed keying
that removes the seed-0 collapse (``ops.rng``) is planned for roadmap PR16; it changes the
built-ins' outputs, so it is not done yet.

Legacy ``_src`` helpers
~~~~~~~~~~~~~~~~~~~~~~~

Plugins written before ``track_gen.ops`` existed import private helpers from
``track_gen._src.warp_pipeline`` (usually as ``_pipe``), ``track_gen._src.warp_generate``
and ``track_gen._src.warp_generate_hull``. They are still importable: each keeps its name
and parameter list, which
``tests/test_legacy_plugin_abi.py`` pins for the 0.x series, and now delegates to the same
kernels as the op that replaces it, so its results are unchanged. They are deprecated in
favour of ``track_gen.ops`` (they do not warn yet); port a plugin with this table:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy helper
     - Replacement
   * - ``_pipe._init()``, ``_pipe._sync(device)``
     - Not needed: the facades initialise Warp before calling ``generate``, and the ops
       never sync.
   * - ``_pipe._fill_i32_k`` (a kernel)
     - ``ops.fill(ctx, arr, value)``
   * - ``_pipe._ccw`` (a ``wp.func``)
     - ``ops.funcs.ccw``
   * - ``_pipe._arc_resample_inplace``, ``_pipe._arc_resample_selected_inplace``
       (and their ``arc_length_resample_*`` twins)
     - ``ops.arc_resample(..., active=...)``, with an ``ops.ArcScratch`` wrapping the same
       four buffers
   * - ``_pipe.self_intersections_inplace``
     - ``ops.self_intersections``
   * - ``_pipe.turning_number_inplace``
     - ``ops.turning_number``
   * - ``_pipe.resample_uniform``
     - ``ops.resample_uniform`` (pass the scratch: the helper allocates when ``seg_wp`` or
       ``s_wp`` is ``None``, the op never does)
   * - ``_pipe.resample_constant_spacing``
     - ``ops.resample_constant_spacing`` (the helper's allocating mode also reads the
       truncation flag back to warn; the op leaves it on the device)
   * - ``warp_generate.corner_count_sample_inplace``,
       ``warp_generate_hull.point_count_sample_inplace``
     - ``ops.sample_counts`` with the salts above
   * - ``warp_generate.corner_sample_inplace``,
       ``warp_generate_hull.point_sample_inplace``
     - ``ops.grid_cells`` and ``ops.sample_grid_points``
   * - ``warp_generate.ccw_sort_inplace``
     - ``ops.angle_sort`` (the helper sorts one point when ``count[e] < 1``; the op
       treats ``count[e] <= 0`` as empty)

The helpers keep their own conventions: ``ctx=None`` as the last argument (the ambient
context when omitted), a debug-only ``ctx.sync()`` after their launches, and the config
objects the samplers read. ``track_gen._src.warp_pipeline.GenScratch`` moved to
``track_gen._src.warp_generate.GenScratch``; the old name still works and emits a
``DeprecationWarning``.

Registering and discovery
-------------------------

``track_gen.plugin.registry`` is the process-wide ``Registry``. It keys specs by *kind* and
name. Two kinds exist today: ``"generator"`` (a ``GeneratorSpec`` or a ``Stage`` class)
and ``"gate_generator"`` (``GateGeneratorSpec``). The set of kinds is open: later releases
add kinds for other pipeline stages.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Call
     - What it does
   * - ``registry.add(spec, *, kind=None, replace=False)``
     - Register ``spec`` (``reg.add`` inside ``register(reg)``). The kind is inferred
       from a ``GeneratorSpec``, a ``GateGeneratorSpec`` or a ``Stage`` class (its
       ``kind``); any other spec type needs ``kind=``. A Stage's declaration and a
       ``params=`` class are checked first (see :ref:`generator-stage-contract` and
       :ref:`generator-params`).
   * - ``registry.get(kind, name)``
     - The spec. An unknown name raises ``ValueError`` listing the available names and
       the quarantined plugins with their errors.
   * - ``registry.available(kind=None)``
     - The sorted names of one kind, or every ``"kind:name"`` when ``kind`` is ``None``.
   * - ``registry.describe(kind, name)``
     - A ``SpecInfo``: ``kind``, ``name``, ``origin`` (the module that defines the spec),
       ``distribution`` (``(name, version)`` of the package whose entry point registered
       it; ``None`` for a built-in and for a spec added at run time), ``capturable``,
       ``accepts_ctx``, ``builtin`` (``True`` for the specs shipped with TrackGen),
       ``entry_point`` (the name of the entry point that registered it, or ``None``) and
       ``params_schema`` (the JSON schema of its Params class, ``{}`` without one; see
       :ref:`generator-describe`).
   * - ``registry.unregister(kind, name)``
     - Remove a spec. Inside its ``register(reg)``, a plugin may only unregister names
       it registered itself.
   * - ``registry.scoped(*, fresh=False)``
     - Context manager: snapshot every kind on entry and restore it on exit (for tests).
       ``fresh=True`` starts the scope from an empty, unloaded registry, so the next
       call re-runs built-in and entry-point discovery.
   * - ``registry.quarantined``
     - Read-only mapping of plugin entry-point name to the error that quarantined it.
       When two quarantined plugins share an entry-point name, their keys are qualified
       as ``"<distribution>:<entry point>"``.

.. _plugin-names:

Names
~~~~~

Third-party names are dotted, ``<package>.<name>`` (``acme.wobbly``), which keeps plugins
from colliding with each other and with built-ins, present or future. A third-party name
without a dot still registers, with a one-time ``UserWarning``. Built-in names
(``bezier``, ``polar``, ...) stay undotted.

Duplicates
~~~~~~~~~~

Adding a name that already exists in that kind raises ``DuplicateSpecError`` (a
``ValueError``) naming both origins: the module of each spec's callables and, for a plugin,
the distribution that registered it. Nothing is overwritten silently. To override on
purpose, pass ``replace=True``: the new spec replaces the old one and a
``PluginOverrideWarning`` (a ``UserWarning``) names both origins. Adding the identical spec
object again is a no-op, so re-running ``registry.add(SPEC)`` (a notebook cell) is safe.

Built-ins register first, so a plugin that reuses a built-in name without ``replace=True``
never replaces it: that plugin is quarantined (or raises, under ``strict``) and the
built-in stays. A plugin cannot get around this by unregistering the built-in first: inside
``register(reg)``, unregistering a name the plugin did not add quarantines it too.

Loading
~~~~~~~

The registry fills itself lazily, on the first registry call (``get``, ``available``,
``describe``, ``add``, ``unregister``, ``quarantined``, ``scoped()``, or one of the old
private registry shims) — in practice when the first ``TrackGenerator``, ``GateGenerator``
or ``Course`` is constructed. Calling ``acme_tracks.register(registry)`` by hand therefore
also discovers the installed entry points first. ``import track_gen`` imports no generator module and loads no plugin. Loading
runs once per process, under a lock, so concurrent first lookups load everything exactly
once:

1. **Built-ins.** ``track_gen._src.builtin_plugins.register(reg)`` imports the shipped
   generator modules and calls each one's ``register(reg)``. Built-ins need no entry
   point, so a source checkout without installed metadata works.
2. **Third parties.** Every entry point of the ``track_gen.plugins`` group, sorted by
   distribution name and then entry-point name, is loaded and called with the registry.
   The order does not depend on install order.

A facade resolves its spec once, at construction. Registering, replacing or unregistering a
name later never changes a live ``TrackGenerator``, ``GateGenerator`` or ``Course``.

Broken plugins: ``TRACK_GEN_PLUGINS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One broken plugin must not take the library down. The environment variable
``TRACK_GEN_PLUGINS``, read when the registry loads, sets what happens to a plugin that
fails to import, raises inside ``register``, registers a name that is already taken
without ``replace=True``, unregisters a name it did not register, or declares a newer
plugin API than this TrackGen supports. The value is case-insensitive and stripped; any
other value emits a ``RuntimeWarning`` and falls back to ``warn``, so the built-ins keep
working:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Behaviour
   * - ``warn`` (default)
     - The plugin is **quarantined**: one ``RuntimeWarning``, an entry in
       ``registry.quarantined``, and every other plugin still loads.
   * - ``strict``
     - Loading raises ``PluginLoadError``. Use it in CI.
   * - ``off``
     - Third-party discovery is skipped: only the built-ins are registered.

.. _plugin-quarantine:

A lookup for an unknown name lists the quarantined plugins and their errors next to the
available names. Asking for what a quarantined plugin provides (e.g.
``TrackGenConfig(generator="acme.wobbly")`` when ``acme`` failed to load) raises
``PluginLoadError`` chained from the plugin's original error, not a bare "unknown
generator". A plugin that fails before it registers anything cannot say which names it
provides, so the registry matches by namespace: a name raises ``PluginLoadError`` when it
is the plugin's entry-point name or entry-point module, is namespaced under one of them
(``acme.wobbly`` for the entry point ``acme`` or the module ``acme``), or its first dotted
component is the plugin's distribution name (normalized, ``acme_tracks.x`` for
``acme-tracks``). Any other name raises the plain ``ValueError``. Name the entry point after
your names' prefix so this matching works.

``track_gen.plugin.API_VERSION`` (currently ``1``) is the version of this plugin interface.
A plugin declares the version it targets by setting ``track_gen_api = 1`` in the module its
entry point names (``acme_tracks`` for ``"acme_tracks:register"``), or as an attribute of
the register function. A plugin that declares a newer version than the installed
TrackGen's is quarantined with a message saying so, instead of failing later in a less
obvious way. An integer ``API_VERSION`` in the same places is read too, as a fallback; a
non-integer one (an unrelated constant) is ignored.

.. _plugin-own-tests:

Testing a plugin
~~~~~~~~~~~~~~~~

Start with the conformance kit: ``track_gen.testing.check_generator("acme.wobbly")``
(``check_gate_generator`` for a gate generator) runs every contract check on CPU, and on
CUDA when a device is present; :doc:`testing-a-plugin` covers it and its pytest plugin.
The tests below are for what the kit cannot know: your generator's own properties.

Run the plugin's CI with ``TRACK_GEN_PLUGINS=strict`` so a load failure fails the run. On
CPU the facades run directly (``device="cpu"``), and every rule above holds there too.
Wrap any test that adds, replaces or removes specs in ``registry.scoped()``: it snapshots
every kind on entry and restores it on exit, so nothing leaks into the next test.

.. code-block:: python

   import dataclasses

   from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
   from track_gen.plugin import registry


   def test_wobbly_runs_on_cpu():
       config = TrackGenConfig(generator="acme.wobbly", num_envs=8, device="cpu")
       rng = PerEnvSeededRNG(seeds=0, num_envs=8, device="cpu")
       track = TrackGenerator(config, rng).generate()
       assert track.valid.numpy().all()
       assert registry.describe("generator", "acme.wobbly").distribution is not None


   def test_a_variant_without_leaking_it():
       with registry.scoped():
           base = registry.get("generator", "acme.wobbly")
           registry.add(dataclasses.replace(base, name="acme.wobbly_variant"))
           assert "acme.wobbly_variant" in registry.available("generator")
       assert "acme.wobbly_variant" not in registry.available("generator")

(A Stage variant is a subclass with another ``name``.) Test a dynamic parameter against a
generator built with the same value: a ``set`` followed by a generation must give its
output bit for bit.

.. code-block:: python

   import numpy as np


   def _wobble(**params):
       config = TrackGenConfig(generator="acme.wobble", num_envs=8, device="cpu",
                               generator_params=params)
       return TrackGenerator(config, PerEnvSeededRNG(seeds=0, num_envs=8, device="cpu"))


   def test_amp_takes_effect_at_the_next_generation():
       gen = _wobble()
       gen.params["generator"].set("amp", 0.3)
       np.testing.assert_array_equal(gen.generate().center.numpy(),
                                     _wobble(amp=0.3).generate().center.numpy())

The old private registries
~~~~~~~~~~~~~~~~~~~~~~~~~~

Plugins written for 0.2.0.dev1 or earlier imported ``track_gen._src.generator_registry``
(or ``gate_generator_registry``) and called its ``register(spec)`` at import time. That
keeps working: ``register``, ``get`` and ``available`` delegate to
``track_gen.plugin.registry``, and ``GENERATORS`` / ``GATE_GENERATORS`` are read-only views
of it. Each old ``register`` keeps its old duplicate rule. The track registry's still
overwrites an existing name (it is ``registry.add(spec, replace=True)``). The gate
registry's still raises when a different module registered the name (now
``DuplicateSpecError``, still a ``ValueError``) and overwrites a spec from the same module.
Either emits a ``DeprecationWarning`` when it overwrites. Such a plugin must still be
imported before use; move it to an entry point and ``registry.add`` to drop that
requirement.

Current standard generators
---------------------------

The standard runtime generators are ``bezier``, ``checkpoint``, ``hull``,
``polar``, ``repulsive``, and ``voronoi``.  Call
``track_gen.plugin.registry.available("generator")`` for the authoritative list, installed
plugins included (``available("gate_generator")`` for gate generators). Each built-in is a
``GeneratorSpec`` (gate: ``GateGeneratorSpec``) with a static Params class whose fields are
the flat ``TrackGenConfig`` / ``GateGenConfig`` knobs it reads, under the same names; the
:ref:`configuration reference <builtin-generator-params>` lists them.

The Voronoi method is implemented as a fixed-budget site-field / graph-cycle
generator rather than exact Voronoi ridge walking; exact Delaunay/Voronoi
construction remains an offline diagnostic until it can satisfy the same
fixed-shape Warp contract.  The checkpoint method is the bounded,
graph-capturable version of the Gymnasium CarRacing checkpoint-steering family:
fixed-N steering, additive heading-ramp closure, best-of-K selection, and
optional clip fallback.  The repulsive method is a self-repulsive curve-growth
generator (see :doc:`/generators/repulsive`); it is ~1000× slower than the other
five, and stays capturable by running a fixed iteration schedule with an absorbing
per-env freeze instead of a host-side early exit.

Adding a built-in generator
---------------------------

A generator shipped inside TrackGen follows the same contract. It lives in
``track_gen/_src/warp_generate_<name>.py`` (gate variants in
``warp_generate_<name>_gates.py``), defines the same ``register(reg)`` function, and is
listed once in ``track_gen/_src/builtin_plugins.py``, the only shared registration
touch-point. Built-in names are undotted. Built-ins use ``track_gen.ops`` for every shared
step (see :ref:`generator-ops`), like a plugin, and never import ``warp_pipeline``. A
built-in gate generator that shares a track generator's sampling imports that module's
public names (``CORNER_SALT``, ``sample_polar_controls``, ...), never its underscore
kernels or constants. ``tests/test_ops.py`` checks both rules on the source, together with
the short allow-list of private ``track_gen.ops`` launchers the legacy wrappers use. A
new built-in RNG stream takes a ``(salt, offset)`` pair that is not in the
:ref:`table of built-in salts <ops-rng-keying>`, and adds its row there. The built-ins'
Params classes live in ``track_gen/_src/builtin_params.py``: each keeps the flat config
field names, holds the generator's rules in ``validate()`` (with ``lo``/``hi`` only where a
rule rejects the same bound), and is listed in the
:ref:`configuration reference <builtin-generator-params>`.

.. _how-a-generator-is-judged:

How a generator is judged
-------------------------

Run ``benchmarks/compare_generators.py`` to characterise a new generator
against the existing ones (see the :doc:`Benchmarks page </generators/benchmarks>` for baseline
numbers and the comparison methodology).

Generators are characterised, never gated: a method that scores worse on yield
but better on speed or style stays selectable. The contract is gated: every built-in
passes ``track_gen.testing``'s conformance checks in TrackGen's CI, and a plugin should
pass them in its own (see :doc:`testing-a-plugin`).
