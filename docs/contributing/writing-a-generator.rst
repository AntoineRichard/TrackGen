Writing a Generator
===================

A **track generator** produces the initial closed centerline that the pipeline then
resamples, relaxes (XPBD), and inflates. A **gate generator** produces the ordered
plan-view gate anchors that the gate pipeline then spreads, elevates, lifts to 3D, and
frames. Both are plugins: a few callables wrapped in a ``GeneratorSpec`` or
``GateGeneratorSpec`` and added to the registry of ``track_gen.plugin``.

A plugin ships as its own pip-installable package. It imports only public modules
(``track_gen.plugin``, ``track_gen.exec``, ``track_gen.ops``), it does not edit TrackGen,
and TrackGen finds it through a Python entry point. The shipped generators register through
a ``register(reg)`` function with the same signature as the one a third party writes.

Quick start: an out-of-tree generator
-------------------------------------

A plugin needs three pieces: an entry point in the ``track_gen.plugins`` group, the
``register(reg)`` function it names, and the specs that function adds.

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
per-env phase from the seed, then writes a three-lobed wobbly circle:

.. code-block:: python

   from types import SimpleNamespace

   import warp as wp

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
   def _wobbly_circle_k(phase: wp.array(dtype=wp.float32), n: int, radius: float,
                        out: wp.array(dtype=wp.vec2f)):
       t = wp.tid()                              # one thread per point: t = e * n + i
       e = t // n
       theta = 6.2831853 * float(t - e * n) / float(n)
       r = radius * (1.0 + 0.2 * wp.sin(3.0 * theta + phase[e]))
       out[t] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


   def alloc_scratch(config):
       E = int(config.num_envs)
       return SimpleNamespace(
           phase=wp.empty(E, dtype=wp.float32, device=str(config.device)))


   def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None):
       ctx = resolve(ctx, out_centerline.device)  # the facades always pass ctx
       E, N = int(config.num_envs), int(config.num_points)
       ctx.launch(_phase_k, E, [seeds_wp, SALT, scratch.phase])
       ctx.launch(_wobbly_circle_k, E * N,
                  [scratch.phase, N, 0.6 * float(config.scale), out_centerline])
       ctx.fill(out_valid_wp, 1)


   def register(reg):
       reg.add(GeneratorSpec(
           name="acme.wobbly",
           alloc_scratch=alloc_scratch,
           generate=generate,
           accepts_ctx=True,  # generate takes ctx= and emits all its work through it
       ))

**3. Install the package and select the generator by name.** Nothing imports
``acme_tracks``: the registry discovers the entry point on its first lookup.

.. code-block:: bash

   pip install -e ./acme-tracks

.. code-block:: python

   from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator

   config = TrackGenConfig(generator="acme.wobbly", num_envs=1024, device="cuda:0")
   rng = PerEnvSeededRNG(seeds=0, num_envs=1024, device="cuda:0")
   track = TrackGenerator(config, rng).generate()

Entry points are read from installed package metadata, so an editable install is enough
but a bare source tree on ``PYTHONPATH`` is not. Without an install (a notebook, a quick
experiment), register the plugin yourself before constructing the generator:

.. code-block:: python

   import acme_tracks
   from track_gen.plugin import registry

   acme_tracks.register(registry)

Skip this when the package is installed: its entry point has already registered the
specs, and adding a name again with a different spec object raises
``DuplicateSpecError`` (adding the identical object again is a no-op).

The track generator contract
----------------------------

What you implement
~~~~~~~~~~~~~~~~~~

A ``GeneratorSpec`` has five fields:

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

The loop is closed implicitly: do not repeat point 0 at the end. Size it like the
built-ins, which center each loop and scale its longest bounding-box side to
``1.44 * config.scale``; the default ``half_width``, spacing and ``N_max`` are tuned for
that, and a much larger loop needs more than ``N_max`` resampled points and fails as
truncated.

.. _generator-hard-rules:

Hard rules
~~~~~~~~~~

Five rules must hold for every generator whose ``GeneratorSpec`` uses the default
``capturable=True``:

1. **Pure Warp kernels only.**  Launch through ``ctx.launch``; one env per row.  No
   torch code: TrackGen does not depend on torch.
2. **Zero dynamic allocation inside** ``generate``.  All buffers come from
   ``alloc_scratch``.
3. **CUDA-graph capturable.**  No host sync, no host-side retry loop
   conditioned on generated data, no per-env Python branching inside
   ``generate``.
4. **Fixed bounds for every loop and buffer.**  Graph capture requires static
   shapes.
5. **Deterministic in** ``(per-env seed, config)``.  Draw every random number with
   the Warp RNG from ``seeds_wp[e]``, salted with a constant of your own
   (``wp.rand_init(seeds[e], SALT)``) so your stream is not correlated with the
   library's stages for the same seed; the built-ins use ``track_gen._src.rng_*``.
   CPU vs CUDA RNG results may differ, as elsewhere in the codebase.

.. _generator-streams:

Streams: ``accepts_ctx``
~~~~~~~~~~~~~~~~~~~~~~~~

Set ``accepts_ctx=True`` and emit all device work through ``ctx``: ``ctx.launch(kernel,
dim, inputs)``, ``ctx.copy(dst, src)``, ``ctx.fill(arr, value)`` (or
``track_gen.ops.fill(ctx, arr, value)``), and pass ``ctx=ctx`` to every helper you call.
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
selected polygon-style rescues; ``checkpoint`` uses best-of-K plus an optional
clip fallback; ``polar`` has no local fallback.  Residual bad geometry is
handled by the common post-relax validity gate, which also ANDs in your
``out_valid_wp`` flag: write ``0`` there to reject an env outright.

Output must be finite (no NaN) for generated environments.

Per-generator parameters
~~~~~~~~~~~~~~~~~~~~~~~~

``TrackGenConfig`` has no per-generator namespace yet (roadmap PR8 adds one). Read the
shared fields (``num_envs``, ``num_points``, ``scale``, ``device``, ...) and, for knobs of
your own, ``getattr(config, "acme_lobes", 3)`` on a config subclass your users construct.
Always fall back to a default: tools such as ``benchmarks/compare_generators.py`` build a
plain ``TrackGenConfig`` for every registered generator.

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

Size the anchors like the built-ins: centered, with the longest bounding-box side at
``1.44 * config.scale``. A ring of eight gates around the same per-env phase, added by the
same ``register`` function:

.. code-block:: python

   from track_gen.plugin import GateGeneratorSpec

   RING_GATES = 8


   @wp.kernel
   def _ring_gates_k(phase: wp.array(dtype=wp.float32), k: int, g_max: int,
                     radius: float, position: wp.array(dtype=wp.vec2f),
                     count: wp.array(dtype=wp.int32)):
       t = wp.tid()                              # one thread per slot: t = e * g_max + g
       e = t // g_max
       g = t - e * g_max
       if g < k:
           theta = 6.2831853 * float(g) / float(k) + phase[e]
           position[t] = wp.vec2f(radius * wp.cos(theta), radius * wp.sin(theta))
       else:
           position[t] = wp.vec2f(wp.nan, wp.nan)  # NaN-pad the unused slots
       if g == 0:
           count[e] = k


   def generate_ring_gates(seeds_wp, config, out, scratch, ctx=None):
       ctx = resolve(ctx, out.position.device)
       E, G = int(config.num_envs), int(config.max_gates)
       ctx.launch(_phase_k, E, [seeds_wp, SALT, scratch.phase])
       ctx.launch(_ring_gates_k, E * G,
                  [scratch.phase, RING_GATES, G, 0.72 * float(config.scale),
                   out.position, out.count])


   def register(reg):
       ...  # reg.add(GeneratorSpec(name="acme.wobbly", ...)) as above
       reg.add(GateGeneratorSpec(
           name="acme.ring",
           alloc_scratch=alloc_scratch,          # the same [E] phase buffer
           generate=generate_ring_gates,
           max_gates=lambda config: RING_GATES,
           supported_orderings=frozenset({"ccw", "raw"}),  # emitted counter-clockwise
           accepts_ctx=True,
       ))

``GateGenerator(GateGenConfig(generator="acme.ring", ...), rng)`` then selects it.

Registering and discovery
-------------------------

``track_gen.plugin.registry`` is the process-wide ``Registry``. It keys specs by *kind* and
name. Two kinds exist today: ``"generator"`` (``GeneratorSpec``) and ``"gate_generator"``
(``GateGeneratorSpec``). The set of kinds is open: later releases add kinds for other
pipeline stages.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Call
     - What it does
   * - ``registry.add(spec, *, kind=None, replace=False)``
     - Register ``spec`` (``reg.add`` inside ``register(reg)``). The kind is inferred
       from a ``GeneratorSpec`` or a ``GateGeneratorSpec``; any other spec type needs
       ``kind=``.
   * - ``registry.get(kind, name)``
     - The spec. An unknown name raises ``ValueError`` listing the available names and
       the quarantined plugins with their errors.
   * - ``registry.available(kind=None)``
     - The sorted names of one kind, or every ``"kind:name"`` when ``kind`` is ``None``.
   * - ``registry.describe(kind, name)``
     - A ``SpecInfo``: ``kind``, ``name``, ``origin`` (the module that defines the spec),
       ``distribution`` (``(name, version)`` of the package whose entry point registered
       it; ``None`` for a built-in and for a spec added at run time), ``capturable``,
       ``accepts_ctx``, ``builtin`` (``True`` for the specs shipped with TrackGen) and
       ``entry_point`` (the name of the entry point that registered it, or ``None``).
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

Testing a plugin
~~~~~~~~~~~~~~~~

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

Using library internals
~~~~~~~~~~~~~~~~~~~~~~~

The example above uses public API only. The built-ins also use helpers that are not
public yet: NaN-aware arc-length resampling (``warp_pipeline._arc_resample_inplace``),
gate ordering and NaN padding (``warp_gate.finish_ordered_gates``), the ``_fill_i32_k``
kernel and the ``track_gen._src.rng_*`` samplers. They stay private until roadmap PR7
publishes them in ``track_gen.ops`` (which today has ``fill``, ``any_set`` and
``hash32``). A plugin may import them at its own risk: they can change in any 0.x release,
so pin the TrackGen version if you do, and pass ``ctx=ctx`` to every one that takes it.

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
plugins included (``available("gate_generator")`` for gate generators).

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
touch-point. Built-in names are undotted, and built-ins may use the private helpers above.

How a generator is judged
-------------------------

Run ``benchmarks/compare_generators.py`` to characterise a new generator
against the existing ones (see the :doc:`Benchmarks page </generators/benchmarks>` for baseline
numbers and the comparison methodology).

Generators are characterised, never gated: a method that scores worse on yield
but better on speed or style stays selectable.
