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
per-env phase from the seed, writes a three-lobed wobbly circle, and then uses
``track_gen.ops`` (see :ref:`generator-ops`) to scale it to the size the built-ins use:

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
``capturable=True``:

1. **Pure Warp kernels only.**  Launch through ``ctx.launch``; one env per row.  No
   torch code: TrackGen does not depend on torch.
2. **Zero dynamic allocation inside** ``generate``.  All buffers come from
   ``alloc_scratch``. ``track_gen.ops`` never allocates: you pass it the working
   buffers, allocated there.
3. **CUDA-graph capturable.**  No host sync, no host-side retry loop
   conditioned on generated data, no per-env Python branching inside
   ``generate``.
4. **Fixed bounds for every loop and buffer.**  Graph capture requires static
   shapes.
5. **Deterministic in** ``(per-env seed, config)``.  Draw every random number with
   the Warp RNG from ``seeds_wp[e]``, salted with a constant of your own so your
   stream is not correlated with the built-ins' streams for the same seed:
   ``wp.rand_init(seeds[e], SALT)`` in your own kernels, and a ``(salt, offset)`` pair
   no built-in uses for the ``track_gen.ops`` samplers (see :ref:`ops-rng-keying`).
   CPU vs CUDA RNG results may differ, as elsewhere in the codebase.

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
touch-point. Built-in names are undotted. Built-ins use ``track_gen.ops`` for every shared
step (see :ref:`generator-ops`), like a plugin, and never import ``warp_pipeline``. A
built-in gate generator that shares a track generator's sampling imports that module's
public names (``CORNER_SALT``, ``sample_polar_controls``, ...), never its underscore
kernels or constants. ``tests/test_ops.py`` checks both rules on the source, together with
the short allow-list of private ``track_gen.ops`` launchers the legacy wrappers use. A
new built-in RNG stream takes a ``(salt, offset)`` pair that is not in the
:ref:`table of built-in salts <ops-rng-keying>`, and adds its row there.

How a generator is judged
-------------------------

Run ``benchmarks/compare_generators.py`` to characterise a new generator
against the existing ones (see the :doc:`Benchmarks page </generators/benchmarks>` for baseline
numbers and the comparison methodology).

Generators are characterised, never gated: a method that scores worse on yield
but better on speed or style stays selectable.
