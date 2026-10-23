Testing a Plugin
================

``track_gen.testing`` is the conformance kit for generator and stage plugins. One call
runs a registered generator through every check the :doc:`generator contract
<writing-a-generator>` implies: output shapes and NaN padding, determinism, batch
independence, no hidden state, no host sync or allocation on the hot path, a static
launch sequence, the absorbing freeze of an iterative (``BOUNDED``) loop and, on CUDA,
recording into a caller's graph. ``check_stage`` does the
same for a relax, width, elevation, validity, order or stations stage (see
:ref:`kit-check-stage`).
TrackGen's own test suite runs the same checks over every built-in generator and stage,
so a plugin is held to the standard the built-ins meet.

The kit checks the contract, not the quality of the tracks: yield, shape variety and
speed are measured by ``benchmarks/compare_generators.py`` (see
:ref:`how-a-generator-is-judged`), and your generator's own properties need tests of
their own (see :ref:`plugin-own-tests`).

``track_gen.testing`` is a subpackage outside ``track_gen.__all__``: import it
explicitly. ``import track_gen`` does not load it, and importing it imports neither torch
(only the torch capture modes do, and they are skipped when torch is missing) nor pytest
(only the opt-in pytest plugin does).

Quick start
-----------

.. code-block:: python

   from track_gen.testing import check_generator

   report = check_generator("acme.wobble", params={"lobes": 5})
   report.raise_for_failures()

``check_generator`` looks the generator up in the registry, so the plugin must be
installed (its entry point registers it) or registered by hand first, as in the
:doc:`generator guide <writing-a-generator>`. It runs every check on ``cpu``, and on
``cuda:0`` too when Warp sees a CUDA device, and returns a ``Report``.
``raise_for_failures()`` raises one ``AssertionError`` listing every failed check, and
``print(report)`` prints the whole table (here on a machine without CUDA):

.. code-block:: text

   generator 'acme.wobble': 16/16 checks passed
     cpu  construct            PASS
     cpu  output.shape         PASS
     cpu  output.finite        PASS
     ...
     cpu  hot_path.topology    PASS
     cpu  hot_path.buffers     PASS

In pytest, load the kit's plugin from your top-level ``conftest.py`` and call the
``track_gen_conformance`` fixture, which fails the test with the same list:

.. code-block:: python

   # conftest.py
   pytest_plugins = ["track_gen.testing.pytest_plugin"]

.. code-block:: python

   # test_conformance.py
   def test_wobble_conforms(track_gen_conformance):
       track_gen_conformance("acme.wobble", params={"lobes": 5})

Run that job with ``TRACK_GEN_PLUGINS=strict``, so a plugin that fails to load fails the
run instead of being quarantined with a warning (see :ref:`quarantined plugins
<plugin-quarantine>`).

.. _conformance-checks:

The checks
----------

Each check has a stable name, so a test can assert which check a generator fails.
``track_gen.testing.CHECKS["generator"]`` lists them in run order. Every check runs once
per device, except that the ``hot_path.*`` checks and ``host.require_caller_capturable``
run on CPU devices only and the ``capture.*`` checks and ``bounded.lowering`` on CUDA
devices only. On CPU every ``generate()`` runs its Python
eagerly, so the guards see every call it makes; on CUDA the generator replays a graph,
and recording it into a real capture is the stronger test. A failed check's message
names the env, the field or the call site (``module.function:line``), and the rule it
broke.

.. list-table::
   :header-rows: 1
   :widths: 21 42 37

   * - Check
     - Passes when
     - Usual causes of a failure
   * - ``construct``
     - The generator resolves and validates its params, allocates its buffers, runs
       its warm-up pass and one more ``generate()``, the batch the other checks read.
     - An exception in ``validate``, ``alloc_scratch``, ``build`` or ``generate``; the
       message names it and where it was raised.
   * - ``output.shape``
     - The raw centerline is ``[E*R]`` ``wp.vec2f`` with ``R >= 3``, the raw validity is
       ``[E]`` ``int32``, every ``Track`` field has its documented shape and dtype, the
       result's ``layout`` equals the Program's and ``result.validate()`` passes
       (:ref:`results-validate`), and both validity arrays hold only ``0`` and ``1``.
     - Writing a flag other than ``0`` or ``1`` into ``out_valid`` (a Stage: its
       ``"gen.valid"`` port), or not writing it on every call (the kit's poison, ``7``,
       survives).
   * - ``output.view``
     - The pipeline built with ``out=`` the middle view ``big.view(1, E + 1)`` of a
       byte-poisoned ``Track.alloc`` (``GateSequence.alloc`` for gates) of ``E + 2``
       envs, attributes included, with the kit's seed array, gives the bytes of the
       kit's standalone run in the view's rows, after construction and after a masked
       regeneration (``mask = [1, 0, 1, ...]``), and every byte outside the view keeps
       its poison (:ref:`results-view-safety`). It runs for every kind:
       ``check_generator``, ``check_gate_generator`` and ``check_stage``.
     - A kernel that writes outside ``[0, E * stride)`` of a result array, or derives an
       index from something other than ``wp.tid()`` and the per-env layout.
   * - ``output.finite``
     - The raw centerline is finite for every env the generator marked valid, and every
       per-point ``Track`` field is finite before ``count[e]`` for every valid env. The
       kit poisons the raw output before the generation it checks, so a point never
       written is reported as such.
     - A NaN from a zero-length normalisation or a division by zero; a scratch value read
       before it is written; a point the kernel never writes (a launch one point short).
       Write ``0`` to ``out_valid`` to reject an env instead.
   * - ``output.padding``
     - Every per-point ``Track`` field is NaN past ``count[e]``, for every env.
     - The pipeline writes these fields from the raw output, so a failure points at a
       raw output the pipeline mishandles, or at a pipeline bug: report it with the
       seeds.
   * - ``output.count``
     - ``0 <= count[e] <= N_max`` for every env, and ``count[e] >= 3`` for valid ones.
     - As for ``output.padding``. A gate generator: a count outside its ``capacity()``
       written into ``anchors.count`` (a legacy spec: outside ``[0, max_gates]``, into
       ``out.count``).
   * - ``output.closed``
     - Each valid env's raw loop closes (the gap from its last point back to its first
       is at most 3 times its median step, or 1.5 times its longest step, so a straight
       that closes the loop passes when the loop has one as long elsewhere), and the
       pipeline accepts at least one env.
     - Writing an open curve, or one that spans less than a full turn. A loop far larger
       than ``TARGET_EXTENT * scale`` needs more than ``N_max`` resampled points, fails
       as truncated, and leaves no valid env (see :ref:`generator-hard-rules`).
   * - ``attributes.written``
     - Every slot of every per-point attribute the generator writes (``raw.z``,
       ``raw.half_width``, ``raw.<package>.<name>``; a gate generator's
       ``anchors.<package>.<name>``, before its ``anchors.count``) is written, for every
       env the generator marks valid: the kit poisons those buffers before the
       generation it checks, as it does the centerline. Skipped ("no attributes") for a
       generator that writes none.
     - A launch that covers ``raw.xy`` but not the attribute (a different ``dim``); an
       attribute written only on some calls (the raw buffers are not cleared between
       generations).
   * - ``attributes.carried``
     - Every exported attribute of the result (``Track.attributes`` /
       ``GateSequence.attributes``) holds its padding (NaN, or ``-1``) past ``count[e]``
       and, on the real points of every env the generator marks valid, finite values
       (floats) that its policy can produce from the env's raw values: ``"linear"``
       within the env's raw ``[min, max]`` per component (with a slack of 4 float32 ulps
       of the largest magnitude), ``"nearest"`` / ``"previous"`` (and an attribute
       carried by index) among the env's raw values, ``"angle"`` in ``[-π, π]``.
       Skipped when nothing is exported.
     - A NaN written into a raw attribute (a non-finite real point). It does not
       catch ``"angle"`` values in degrees (the carry wraps every output into
       ``[-π, π)``) or a label carried ``"linear"`` (its blends stay within the raw
       range): those are the rules of :ref:`generator-attributes`.
   * - ``determinism``
     - Two generators built with the same seeds give bitwise identical batches: the raw
       output and every ``Track`` field, NaN padding included.
     - A random number not drawn from the env's seed (Python's ``random``, numpy's
       global generator, the time); float ``atomic_add`` accumulations, whose order
       varies on CUDA.
   * - ``diversity``
     - Each env changes when its seed changes, and no two envs of one batch are
       identical. Only envs valid in both batches are compared: two rejected envs may be
       alike. ``details`` holds the smallest raw-output difference found, between the
       two batches (``min_raw_delta``) and between two envs (``min_pair_delta``).
       Skipped ("generator declares seeded=False") for a generator that declares
       ``seeded = False``, such as an ingestion generator.
     - A kernel that ignores ``seeds[e]``, or keys every env's stream on the salt only.
   * - ``batch_independence``
     - Env ``e``'s track depends on its own seed only: it is identical when generated
       alone (``E = 1``), inside a larger batch (shifted by one env), and when the other
       envs' seeds are permuted.
     - Reading a neighbour's row (``e + 1``); a statistic over the batch (scaling every
       env by the largest extent); ``E`` or the env index folded into an env's random
       stream instead of its seed alone.
   * - ``statelessness``
     - ``generate()`` twice with unchanged seeds reproduces the first batch; after a
       reseed, it equals a fresh generator built with the new seeds; reseeding back
       reproduces the first batch.
     - State carried from one call to the next: a scratch buffer read before it is
       written, an output accumulated into, an output slot left unwritten, a Python
       counter, seeds latched at construction instead of read from ``seeds_wp``
       (``b.seed``) at every call.
   * - ``bounded.absorbing``
     - For a generator (or the stage under test) that runs a ``ctx.bounded_loop`` with
       ``active``, whatever capture class it declares: with ``active`` poisoned to ``0`` and the outputs poisoned, an eager
       generation with the host-poll lowering (a), which stops at the first window
       whose check reads ``0``, gives the same bytes as one with every window unrolled
       (c): the raw output, the result and the stage's channels, for batch A then B.
       Skipped as not applicable when no such loop runs, and as not exercised when no
       window was skipped for the kit's seeds.
     - An iteration after the freeze that still writes an output (a kernel without the
       frozen-env early return, a converged env that keeps moving); ``active`` written
       only after the first check; Python state the loop body changes and later code
       reads. The message names the window where (a) stopped and the envs and channels
       that differ.
   * - ``bounded.lowering`` (CUDA)
     - The same loop, in every lowering of :ref:`cuda-graph-bounded`, gives the bytes
       of an eager generation: the host poll (a), a Warp capture with conditional
       windows (b, skipped with the reason when ``wp.is_conditional_graph_supported()``
       is false) and a capture with every window unrolled (c), replayed into poisoned
       outputs, before and after a reseed.
     - As for ``bounded.absorbing``, now on the device; a host value baked into one
       window.
   * - ``host.require_caller_capturable`` (CPU)
     - For a pipeline with ``HOST`` stages: building it with
       ``require_caller_capturable=True`` raises ``PlanError`` naming every one of
       ``Program.host_stage_ids``. Skipped as not applicable otherwise.
     - Not a plugin bug: the check covers the planner's refusal. Report a failure.
   * - ``hot_path.host_sync`` (CPU)
     - Two ``generate()`` calls make no ``wp.synchronize*`` call, no host readback
       (``wp.array.numpy()``, ``wp.array.list()``, ``np.asarray`` of an array) and no
       ``wp.copy`` between two devices. The host poll of a ``bounded_loop`` and a
       ``HOST`` stage's ``HostOps`` operations are the sanctioned readbacks: counted in
       ``details["sanctioned"]`` / ``details["host_ops"]``, not reported.
     - A readback to branch on generated data or to size a launch. Branch on the device
       instead (a per-env flag), with fixed bounds; stop a loop with
       ``ctx.bounded_loop``.
   * - ``hot_path.allocation`` (CPU)
     - Two ``generate()`` calls allocate no Warp array (``wp.empty``, ``zeros``,
       ``full``, ``ones``, ``clone``, ``wp.array(data)``, ``*_like``, ...) and create no
       ``wp.HashGrid`` (nor call ``reserve``), ``wp.Mesh``, ``wp.Volume`` or
       ``wp.Bvh``.
     - A temporary allocated per call, or an allocate-if-``None`` helper: allocate it
       once in ``alloc_scratch`` (a Stage: in ``build``, through the builder).
   * - ``hot_path.stream`` (CPU)
     - Every ``wp.launch``, ``wp.launch_tiled``, recorded ``wp.Launch.launch()`` and
       ``wp.copy`` issued from ``track_gen`` or from the plugin's top-level package (or
       by a Warp utility they call) passes ``stream=``, and none of them calls
       ``wp.array.assign()``, ``zero_()`` or ``fill_()`` or enters
       ``wp.ScopedStream``. On CPU this sees that a ``stream=`` is passed, not which
       one: ``stream=None`` (what a CPU ``LaunchContext`` passes) or a launch through a
       context other than the call's ``ctx`` is caught by ``capture.caller`` on CUDA.
     - Launching with ``wp.launch`` instead of ``ctx.launch``; ``arr.zero_()``;
       ``out.assign(tmp)`` instead of ``ctx.copy(out, tmp)``; a legacy spec with
       ``accepts_ctx=False`` (see :ref:`generator-streams`).
   * - ``hot_path.capture`` (CPU)
     - Two ``generate()`` calls open, end or launch no CUDA graph
       (``wp.ScopedCapture``, ``capture_begin`` / ``capture_end`` /
       ``capture_launch`` / ``capture_if`` / ``capture_while``).
     - A generator that captures and replays its own graph: the facade does that
       already, and a graph launch cannot be recorded into a caller's capture.
   * - ``hot_path.topology`` (CPU)
     - The two calls emit the same launch sequence: the same kernels in the same order,
       with the same dimensions, buffers and scalar arguments.
     - Launch dimensions or a scalar argument taken from a call counter or from host
       data; buffers swapped from one call to the next (ping-pong across calls). A
       recorded graph replays the first sequence forever.
   * - ``hot_path.buffers`` (CPU)
     - Every array the two calls launch on belongs to the generator (its result,
       scratch, seed and parameter buffers, ``wp.struct`` fields included), and a second
       generator built alike launches on none of them.
     - A buffer shared by every generator instance: a module global, a variable the
       spec's functions close over, a class attribute (two generators then overwrite
       each other, and a graph keeps the pointer).
   * - ``capture.caller`` (CUDA)
     - ``assert_caller_capturable`` passes: the generation is recorded into a caller's
       capture in each mode, and its replays equal an eager twin, before and after a
       reseed (see :ref:`kit-capture`).
     - Anything the ``hot_path`` checks catch, now on a real capture; a host value
       baked into the recording, such as a seed read once at record time.
   * - ``capture.self_graph`` (CUDA)
     - ``generate()`` captures the generator's own graph, and its replays into poisoned
       outputs equal the eager batch, before and after a reseed.
     - An output the graph never writes (the poison survives the replay); a host value
       baked at capture.

The ``hot_path.*`` checks run their two generations with every ``bounded_loop`` window
unrolled, the superset of what any lowering launches, so a sync or an allocation in a
late window is seen even when the kit's seeds would stop the loop early; one more
generation with the default lowering reports the sanctioned polls.

.. _kit-host-checks:

**HOST generators and skipped checks.** A ``HOST`` generator (a ``Stage`` whose capture
class is ``HOST``) runs eagerly on every call, so its launch sequence may depend on host
data: ``hot_path.topology`` and ``hot_path.buffers`` are reported as skipped ("waived:
HOST stage(s) [...] run eagerly every call"). ``hot_path.host_sync`` and
``hot_path.allocation`` still run: the ``ctx.host()`` operations pass (counted in
``details["host_ops"]``), while a direct ``.numpy()``, a ``wp.synchronize*()`` or a
per-call allocation in the stage fails. A legacy ``GeneratorSpec`` registered with
``capturable=False`` cannot use ``ctx.host()``, so its ``hot_path.host_sync`` and
``hot_path.allocation`` are skipped too ("waived: capturable=False"). ``hot_path.stream``
and ``hot_path.capture`` always apply (see :ref:`generator-capture-opt-out`). On CUDA,
``capture.caller`` checks the refusal instead: recorded inside a caller's capture after a
marker copy, ``generate()`` must raise ``NotCapturableError`` naming the ``HOST`` stages,
and the replayed caller graph must hold the marker and nothing of the generation (the
outputs keep their poison, the seeds and epochs are unchanged). ``capture.self_graph``
checks the segmented self-graph: ``graph`` stays ``None``, a graph exists exactly for
each capturable segment, and replays into poisoned outputs equal the eager batch, before
and after a reseed. When ``construct`` fails, every other check is reported as not run,
and the report fails.

Arguments
---------

.. code-block:: python

   check_generator(name, params=None, *, devices=None, num_envs=8, seeds=(11, 1009),
                   config=None, capture_modes=("warp", "external", "torch"), skip=(),
                   modules=())

- ``params``: the generator's ``generator_params``, a Params instance or a mapping of
  its fields; ``None`` gives the defaults. A generator whose parameters select different
  code paths (a fallback, a mode) needs one call per path.
- ``devices``: a device or a list of devices. The default is ``"cpu"``, plus
  ``"cuda:0"`` when Warp sees a CUDA device.
- ``num_envs``: the batch size ``E``, at least 2 (diversity and batch independence
  compare envs).
- ``seeds``: two base seeds. Batch A seeds env ``e`` with ``seeds[0] + e``, batch B (the
  "different seeds" batch, also used to reseed) with ``seeds[1] + e``. The two ranges
  must not overlap.
- ``config``: extra ``TrackGenConfig`` fields, for example ``{"num_points": 64}`` to keep
  a slow generator fast. The kit sets ``generator``, ``device`` and ``num_envs`` itself;
  pass the Params either as ``params=`` or as ``config["generator_params"]``, not both.
- ``capture_modes``: the modes ``capture.caller`` runs (see :ref:`kit-capture`).
- ``skip``: check names, or a prefix such as ``"hot_path"``, to report as skipped
  instead of running. Use it sparingly, and say why next to the call.
- ``modules``: more dotted module prefixes for ``hot_path.stream`` to probe, besides
  ``track_gen`` and the top-level package of the module that defines the generator
  (``registry.describe(kind, name).origin``; a ``functools.partial`` counts as the
  function it wraps). Name a helper package of yours with another top-level name here.

An unknown check name, overlapping seeds or an unknown capture mode raises
``ValueError``; an unknown generator name raises the registry's error
(``PluginLoadError`` for a quarantined plugin, carrying its load error).

Reading a report
----------------

A ``Report`` holds one ``CheckResult`` per check and device, in run order, and iterates
over them. ``report.ok`` is ``True`` when every check passed or was skipped,
``report.failures`` lists the failed results and ``report.failed_names`` their sorted
names. ``report.result(name, device=None)`` returns one result (on the first device
when ``device`` is ``None``). A ``CheckResult`` has ``name``, ``device``, ``passed``,
``skipped``, ``message`` (why it failed or was skipped), ``details`` (measurements, such
as ``valid_fraction`` for ``output.closed``) and ``status`` (``"PASS"``, ``"FAIL"`` or
``"SKIP"``).

Here is the report of a generator that reads its phase array back with ``.numpy()``,
allocates a temporary with ``wp.empty`` and copies it into the output with a bare
``wp.copy``, all inside ``generate`` (messages shortened):

.. code-block:: text

   generator 'acme.broken': 13/16 checks passed
     cpu  construct            PASS
     ...
     cpu  statelessness        PASS
     cpu  hot_path.host_sync   FAIL  2 host sync/readback(s) in two generate() calls: wp.array.numpy() from acme_broken.generate:31. A capturable hot path must not synchronize ...
     cpu  hot_path.allocation  FAIL  2 allocation(s) in two generate() calls: wp.empty() from acme_broken.generate:33. A capturable hot path must not allocate ...
     cpu  hot_path.stream      FAIL  2 call(s) off the call's stream in two generate() calls: wp.copy without stream= from acme_broken.generate:35. Emit device work through the LaunchContext ...
     cpu  hot_path.capture     PASS
     cpu  hot_path.topology    PASS
     cpu  hot_path.buffers     PASS

The hot path runs twice, so each breach is counted twice. Each message names the line to
fix and ends with the rule it breaks and how to comply.

Gate generators
---------------

``check_gate_generator`` takes the same arguments (``config`` holds ``GateGenConfig``
fields) and runs the same checks on the ``GateSequence``, listed in
``CHECKS["gate_generator"]``. It checks anchors Stages (:ref:`gate-generator-stage`) and
legacy ``GateGeneratorSpec`` objects alike (a legacy spec emits its
``DeprecationWarning``). The raw output is the generator's anchors, checked before
``count[e]``: ``tap("anchors.xy")``, ``[E*A]`` ``wp.vec2f`` with ``A`` the generator's
``capacity()["gates"]`` (a legacy spec: the ``[E*G]`` staging buffer, ``G`` =
``config.max_gates``). ``output.count`` asks for ``0 <= count[e] <= max_gates`` for every
env and ``count[e] >= min_gates`` for valid ones, and for a Stage also
``capacity()["min_gates"] <= anchors.count[e] <= capacity()["gates"]`` for every env;
``output.accepted`` (the gate pipeline accepts at least one env) replaces
``output.closed``. Without a ``gate_ordering`` in ``config``, the kit uses the
generator's default order, as ``GateGenConfig(gate_ordering=None)`` does: its
``preferred["order"]`` when it accepts it, else ``"ccw"`` when it accepts it, else the
first order it accepts (a legacy spec: ``"ccw"`` if supported, else the first of its
``supported_orderings``).

.. code-block:: python

   from track_gen.testing import check_gate_generator

   check_gate_generator("acme.scatter").raise_for_failures()

.. _kit-check-stage:

Pipeline stages: ``check_stage``
--------------------------------

``check_stage`` checks a stage of an open pipeline kind (see
:doc:`writing-a-stage`) by running it where it belongs: in the default track pipeline
of ``presets.track``, in the slot of its kind, built through ``TrackGenerator``; or, for
the gate kinds, in the default gate pipeline of ``presets.gates``, built through
``GateGenerator``:

- an ``order`` stage runs as ``order=`` of a stations pipeline (``generator="bezier"``,
  ``source="stations"``), so no generator's ordering whitelist gets in the way;
- a ``stations`` stage runs as ``stations=``, on ``source="relaxed_stations"`` when it
  reads ``cl.*`` channels, else on ``source="stations"``;
- a ``validity`` stage that reads a ``gates.*`` or ``anchors.*`` channel is appended
  after ``"gate_default"`` in a gate pipeline; any other runs in the track pipeline;
- an ``elevation`` stage runs in the track pipeline, or with ``pipeline="gates"`` as
  ``elevation=`` of the default gate pipeline (the ``bezier`` gate generator unless
  ``upstream`` names another), its ports rewired to ``gates.*``: the ``output.*``
  checks read the ``GateSequence`` and ``stage.isolation`` re-runs the stage on
  ``gates.cum`` / ``gates.perim`` / ``gates.count`` into ``gates.z``. Run it once per
  pipeline to check both.

.. code-block:: python

   from track_gen.testing import check_stage

   check_stage("relax", "acme.taubin", {"passes": 4}).raise_for_failures()
   check_stage("validity", "acme.max_curvature",
               upstream={"elevation": "acme.sinusoid_hills"}).raise_for_failures()
   check_stage("order", "acme.reverse").raise_for_failures()
   check_stage("elevation", "acme.sinusoid_hills").raise_for_failures()
   check_stage("elevation", "acme.sinusoid_hills",
               pipeline="gates").raise_for_failures()
   check_stage("width", "acme.ripple").raise_for_failures()
   check_stage("width", "from_channel", ports={"src": "cl.acme.hw"},
               upstream={"generator": "acme.ribbon"}).raise_for_failures()

.. code-block:: python

   check_stage(kind, name, params=None, *, upstream=None, ports=None, pipeline=None,
               devices=None, num_envs=8, seeds=(11, 1009),
               capture_modes=("warp", "external", "torch"), skip=(), modules=())

- ``kind`` is ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"`` or
  ``"stations"`` (``ValueError`` otherwise); ``name`` is looked up in the registry like a
  generator's (an unknown name raises the registry's ``ValueError``, a quarantined plugin
  its ``PluginLoadError``).
- ``params`` is the stage's params: a Params instance, a mapping of its fields, or
  ``None`` for the defaults.
- ``pipeline`` is ``None`` (the pipeline the kind belongs to, as above), ``"track"`` or
  ``"gates"``. ``"gates"`` takes ``elevation``, ``validity``, ``order`` and ``stations``
  stages, ``"track"`` ``relax``, ``width``, ``elevation`` and ``validity`` stages; any
  other value or combination raises ``ValueError``.
- ``upstream`` holds more ``presets.track`` keyword arguments, the rest of the pipeline
  the stage runs in: ``generator``, ``generator_params``, ``num_points``, ``n_max``,
  ``half_width``, the other kinds' stages, ``on_invalid``. ``num_envs``, ``device`` and
  the slot under test (``relax``, ``width`` or ``elevation``) are the kit's and raise
  ``ValueError``. For the gate kinds, ``upstream`` holds ``presets.gates`` keyword
  arguments instead, with the same reserved keys (``order`` or ``stations``). For a
  validity stage, ``upstream["checks"]`` lists the checks that run
  before it (default ``("default",)``, or ``("gate_default",)`` for a check that runs in
  the gate pipeline), and the stage is appended to them under its own
  name as id; a validity stage that writes ``valid.fail`` without reading it replaces
  them instead.
- ``ports`` is the stage's ``StageRef.ports``: it wires a required read port
  (``{"src": "cl.acme.hw"}`` for ``width.from_channel``) or rewires another; the channel
  must exist in the pipeline ``upstream`` describes (here a generator that carries
  ``raw.acme.hw``).
- ``devices``, ``num_envs`` (at least 2), ``seeds``, ``capture_modes``, ``skip`` and
  ``modules`` are those of ``check_generator`` (see `Arguments`_).

The report reads like a generator's, under ``CHECKS[kind]``. The checks are those of the
generator kit, on the whole pipeline's ``Track`` or ``GateSequence`` (``output.*`` checks it as
``check_generator`` does after the pipeline), without ``diversity`` and
``output.closed`` (the raw loop is the generator's business; a stage need not add
diversity), plus:

.. list-table::
   :header-rows: 1
   :widths: 21 42 37

   * - Check
     - Passes when
     - Usual causes of a failure
   * - ``output.accepted`` (not for validity; for order and stations, the gate
       pipeline accepts at least one env)
     - The pipeline with the stage accepts at least one env. When it accepts none, the
       same pipeline with the slot's neutral built-in (``relax.none``,
       ``width.constant``, ``elevation.flat``, ``order.raw``, ``stations.raw``) runs:
       if that one accepts none either,
       the check is skipped (the stage did not reject them).
     - A relax stage that leaves NaNs or self-crossings, a width under ``w_floor``, an
       elevation steeper than ``z_valid_grade``.
   * - ``stage.isolation``
     - After a generation, the stage's write buffers are filled with a poison byte and
       only its ``enqueue`` re-runs on the same inputs: it changes none of the channel
       versions it reads and reproduces every version it writes, byte for byte (a slot
       still holding the poison is reported unwritten). A stage that launches nothing
       (``elevation.flat``: its output is the buffer's build-time zero) writes nothing
       to compare.
     - Writing the buffer ``b.read`` returned (copying a result back into it); reading
       its own previous output or a scratch value before writing it; an output slot left
       unwritten (padding included).
   * - ``validity.bits`` (validity only)
     - The stage owns at least one fail bit, the bits it does not own are exactly those
       of the same pipeline without it, and ``valid == (fail_mask == 0)`` (on the
       ``Track`` or the ``GateSequence``). Then
       the stage alone is re-run on a synthetic ``valid.fail`` input holding bits it
       does not own (alternating patterns over bits 0-30) into a write buffer filled
       with ``-1``: every env must come out with exactly those bits besides its own.
       ``details`` holds its bits and the fraction of envs with one of them set.
       ``validity.default`` and ``validity.gate_default``, which start the bitmask (the
       only stages the planner lets do so), must leave ``fail_mask`` with their own bits
       only.
     - OR-ing a built-in bit (``fail_in | 1``), clearing bits of earlier checks (writing
       ``own`` instead of ``fail_in | own``), skipping the envs that pass.
   * - ``width.values`` (width only)
     - The stage's own ``cl.half_width`` version (``tap("cl.half_width",
       after=<its id>)``) is finite and ``>= 0`` on every real point of every env the
       generator marks valid; ``details`` holds its minimum and maximum. A half-width
       under ``w_floor`` is validity's verdict (fail bit 4), not this check's. A width
       stage that reads ``cl.half_width`` (a rewrite) runs right after
       ``width.constant``; one with a required read port needs ``ports=``.
     - A negative or NaN width (a NaN source channel of ``width.from_channel`` included:
       the generator's ``attributes.carried`` check fails too when the source attribute
       is exported). A real point left unwritten keeps the buffer's zero, which passes
       here: ``stage.isolation`` reports it.

``determinism``, ``batch_independence`` and ``statelessness`` compare whole ``Track``
batches, ``fail_mask``, ``half_width`` and every attribute included. A relax stage's
bead-identity contract (point ``i`` out is point ``i`` in, moved; see
:ref:`the relax contract <stage-relax-contract>`) is documented, not checked: XPBD's own
drift along the loop reaches the size of an off-by-one shift on some envs, so no
threshold separates the two. The ``hot_path.*`` checks guard two generations of the
whole Program and probe ``track_gen`` and the stage's own top-level package; the
``capture.*`` checks record the whole Program into a caller's capture
(``assert_caller_capturable``) and replay its own graph. With a ``HOST`` stage the
Program runs that stage eagerly in its own segment, and the checks change as for a
``HOST`` generator (see :ref:`HOST generators <kit-host-checks>` above).
``bounded.absorbing`` and ``bounded.lowering`` probe the loops the stage under test
runs, so a stage that calls ``bounded_loop(active=...)`` while declaring ``STATIC``
fails ``construct`` instead (see :ref:`stage-bounded`).

The pytest plugin
-----------------

``track_gen.testing.pytest_plugin`` is opt-in: it has no ``pytest11`` entry point, which
would load it into every pytest run of every environment that installs TrackGen. Load it
with ``pytest_plugins = ["track_gen.testing.pytest_plugin"]`` in your top-level
``conftest.py`` (or ``-p track_gen.testing.pytest_plugin`` on the command line). It
provides:

- the ``track_gen_conformance`` fixture, a factory:
  ``track_gen_conformance(name, params=None, *, kind="generator", **kwargs)`` runs
  ``check_generator`` (``check_gate_generator`` with ``kind="gate_generator"``,
  ``check_stage`` with ``kind`` one of ``"relax"``, ``"width"``, ``"elevation"``,
  ``"validity"``, ``"order"``, ``"stations"``) with ``kwargs``, fails the test listing
  every failed check, and returns the ``Report``;
- ``generator_names(kind="generator", *, builtin=None, exclude=())``, the sorted names
  registered under ``kind``, a generator kind or one of the six stage kinds
  (``builtin=False`` keeps the third-party ones, ``builtin=True`` the built-ins);
- ``parametrize_generators(kind="generator", *, argname="generator", builtin=None,
  exclude=())``, the matching ``pytest.mark.parametrize``. It reads the registry at
  collection time, which loads the installed plugins then.

.. code-block:: python

   import pytest

   from track_gen.testing.pytest_plugin import parametrize_generators

   ACME = ["acme.wobble", "acme.wobbly", "acme.star"]


   @pytest.mark.parametrize("name", ACME)
   def test_acme_generator_conforms(name, track_gen_conformance):
       track_gen_conformance(name)


   def test_scatter_gates_conform(track_gen_conformance):
       track_gen_conformance("acme.scatter", kind="gate_generator")


   # Every third-party generator installed in this environment, yours included:
   @parametrize_generators("generator", builtin=False)
   def test_installed_plugins_conform(generator, track_gen_conformance):
       track_gen_conformance(generator)

``builtin=False`` selects every installed third-party plugin, not only yours: list your
names explicitly when the environment may hold others.

Building blocks
---------------

``check_generator`` is assembled from three public pieces. Use them directly to guard a
code path the checks do not drive: a helper your package exports, a custom
``enqueue``, a generator wrapped in your own simulation step.

.. _kit-contract-guards:

Contract guards: ``track_gen.testing.contract``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hot_path_guard()`` watches a block for the four kinds of breach the ``hot_path``
checks report: host syncs and readbacks, Warp allocations, work off the call's stream,
and nested captures or graph launches. ``no_host_sync()``, ``no_allocation()`` and
``stream_explicit(modules)`` watch one kind each.

.. code-block:: python

   from track_gen.testing import hot_path_guard

   gen = TrackGenerator(config, rng)         # construction may allocate and sync
   with hot_path_guard(modules=("track_gen", "acme_tracks")):
       gen.generate()                        # raises HotPathViolation at the first breach

- ``modules`` lists the dotted module prefixes whose launches, copies, memsets and
  stream scopes the stream check covers (``"acme_tracks"`` covers
  ``acme_tracks.kernels`` too). Host syncs, allocations and captures are checked from
  every caller.
- With ``strict=True`` (the default) a breach raises ``HotPathViolation`` (an
  ``AssertionError``) before the call runs, and the block raises again on exit if a
  breach was recorded, so a broad ``except`` in the code under test cannot hide one.
  With ``strict=False`` the guard only records: ``guard.violations`` lists every
  ``Violation`` (``category``, ``api``, ``site``), ``guard.of(category)`` filters them,
  ``guard.report()`` formats them and ``guard.raise_for_violations()`` raises.
- ``allow`` lets known host syncs through, as ``(api, module, function)`` keys
  (``("wp.array.numpy", "acme_tracks.gen", "generate")``), and records them in
  ``guard.allowed``.
- The host poll of a ``ctx.bounded_loop`` (its lowering on an eager ``cpu`` call, see
  :ref:`cuda-graph-bounded`) and the operations of a ``HOST`` stage's ``ctx.host()``
  (``HostOps``) are sanctioned: they go through and are recorded in ``guard.sanctioned``,
  in neither ``allowed`` nor ``violations``. Any other sync is reported, one made inside
  a loop body or directly in a ``HOST`` stage included.

The guards patch Warp's entry points while a guard is active, wherever a loaded module
binds them (``from warp import launch`` at import time included), and dispatch on a
context variable: calls made from other threads, or from a context without a guard, go
straight to Warp and are not seen. Classes are not replaced (their ``__init__`` is), so
``isinstance`` and subclassing keep working everywhere. While the host-sync check is
active, ``TRACKGEN_DEBUG_SYNC`` is switched off: its sync is deliberate and opt-in. The
switch is the environment variable, so it is off for the whole process until the guard
exits.
Guard a call that runs eagerly. On CPU every generation does; on CUDA, guard
``gen.enqueue()``, since ``generate()`` replays the generator's own graph, which the
guard reports as a ``wp.capture_launch``.

.. _kit-launch-trace:

Launch traces: ``track_gen.testing.trace``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``trace_launches()`` records every ``wp.launch``, ``wp.launch_tiled``,
``wp.Launch.launch()``, ``wp.copy`` and ``wp.array.assign`` in a block as a
``LaunchRecord``: the kernel, the dimensions, the stream handle passed, the pointer of
each array argument and the value of each other argument (a ``wp.struct`` argument is
flattened into its fields, keyed ``"<index>.<field>"``). ``buffers_of(gen)`` collects
every ``wp.array`` reachable from a producer (its result, scratch, seed, parameter and
input buffers) into a ``BufferSet``, against which pointers are checked. It does not
enter a registry spec (``GeneratorSpec``): what a spec's functions close over is shared
by every generator built from it.

.. code-block:: python

   from track_gen.testing import (assert_pointers_within, assert_same_topology,
                                  buffers_of, trace_launches)

   with trace_launches() as first:
       gen.generate()
   with trace_launches() as second:
       gen.generate()
   owned = buffers_of(gen, root="gen")
   assert_same_topology(first, second, buffers=owned)   # hot_path.topology
   assert_pointers_within(first, owned)                 # hot_path.buffers

``LaunchTrace.diff(other, buffers=...)`` returns the differences as text (``[]`` when
there are none), and ``foreign_pointers(buffers)`` the arguments that point outside
``buffers``. With ``buffers``, each pointer is compared as the buffer it falls in (its
label and byte offset), and a pointer outside them as ``"<foreign>"``.

On ``cpu`` a ``bounded_loop`` stops at the first window whose check finds every env
converged, so the length of an eager trace depends on the data: compare two calls on the
same seeds, as above. The kit's ``hot_path.topology`` compares calls with every window
unrolled.

.. _kit-capture:

Recording into a caller's graph: ``track_gen.testing.capture``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``assert_caller_capturable(producer)`` is the CUDA half of the hot-path contract. It
records ``producer.generate(stream=s)`` into a capture the caller owns, replays it and
compares every output with an eager twin, built fresh from the producer's config. The
recording follows the discipline of TrackGen's own capture tests:

- the capture runs on a side stream that is not Warp's current stream, so work that
  ignores the call's stream escapes the capture and shows as a wrong replay;
- the Warp mempool is disabled on the device, so an allocation inside the capture fails
  instead of being served from the pool;
- ``wp.capture_launch`` raises, since a graph launch cannot be recorded into another
  capture (CUDA error 900), and the recording must not create or replace the producer's
  own graph;
- the outputs are poisoned with finite, dtype-aware values before each replay (``-7``
  for signed integers, ``-12345.0`` for every float component), so a slot the replay
  never writes is caught; a NaN poison would hide it, since outputs are NaN-padded;
- the producer is reseeded between replays, and the replay after the reseed must equal a
  twin built with the new seeds: the graph reads the live seeds.

.. code-block:: python

   from track_gen.testing import assert_caller_capturable

   config = TrackGenConfig(generator="acme.wobble", num_envs=8, device="cuda:0")
   gen = TrackGenerator(config, PerEnvSeededRNG(seeds=0, num_envs=8, device="cuda:0"))
   status = assert_caller_capturable(gen)   # {"warp": "ok", "external": "ok", "torch": "ok"}

``modes`` selects the captures: ``"warp"`` (a ``wp.ScopedCapture`` on the side stream),
``"external"`` (a ``torch.cuda.graph`` capture registered with Warp through
``track_gen.interop.torch.registered``) and ``"torch"`` (a plain ``torch.cuda.graph``
capture: the producer is called without ``stream=`` and must adopt torch's capture
stream). The torch modes import torch inside the mode and report
``"skipped: torch with CUDA is not available"`` without it. The producer must be on a
CUDA device and never advanced (every epoch 0); otherwise, or for an unknown mode, it
raises ``ValueError``. ``record=`` replaces the recorded call (default
``producer.generate(stream=stream)``), ``twin=`` the eager reference (default: a fresh
producer of the same type and config, run once with ``enqueue()``), and ``reseed=`` the
seeds of the second comparison (default: the current seeds plus 7919). The helpers it is
built from (``side_stream``, ``mempool_disabled``,
``capture_launch_forbidden``, ``recording``, ``poison``, ``poison_all`` and
``result_arrays``) are public in ``track_gen.testing.capture`` for custom capture
tests; :doc:`/how-it-works/cuda-graph` describes the execution model they test.

A CI recipe
-----------

Two lanes. The CPU lane runs on any runner and catches every breach the kit can see
without a GPU; the CUDA lane runs the same tests on a GPU runner, where the defaults add
``cuda:0`` and the ``capture.*`` checks.

.. code-block:: bash

   # CPU lane, every pull request
   TRACK_GEN_PLUGINS=strict python -m pytest -q

   # CUDA lane, on a GPU runner (install torch with CUDA for the torch capture modes)
   TRACK_GEN_PLUGINS=strict python -m pytest -q

Keep the tests that check your generator's own properties next to the conformance test:
:ref:`plugin-own-tests` shows the registry and parameter patterns.

What the kit does not see
-------------------------

- **Calls outside Warp's Python API.** The guards patch Warp's Python entry points. A
  sync inside a C extension, a torch operation, a call made from another thread, or a
  call through a Warp function bound to a local variable or a closure before the guard
  was entered is invisible to them; so is polling a stream or an event
  (``is_complete``). Only the CUDA lane's real capture catches their effect.
- **Which stream, on CPU.** ``hot_path.stream`` checks that ``stream=`` is passed.
  ``stream=None``, or a launch through ``LaunchContext.current(device)`` instead of the
  call's ``ctx``, passes on CPU and fails ``capture.caller`` on CUDA.
- **Stream discipline outside the probed packages.** ``hot_path.stream`` covers launches
  from ``track_gen`` and from the top-level package of the module that defines the
  generator (``registry.describe(kind, name).origin``). Add a helper package of yours
  with another top-level name with ``modules=``, or guard it with
  ``stream_explicit(modules=...)``.
- **Native handles shared between instances.** ``hot_path.buffers`` compares array
  pointers. A ``wp.HashGrid`` / ``wp.Mesh`` id passed as a ``uint64`` is a scalar to it.
- **A launch sequence that changes after the first call.** ``hot_path.topology``
  compares two calls made long after construction's warm-up pass: a one-time
  initialisation launch on the first call only is not seen.
- **A loop that never stops early for the kit's seeds.** ``bounded.absorbing`` can only
  compare an early stop with the full bound when some window is skipped: with seeds that
  never converge before ``n_iters`` it reports "not exercised". Pass ``seeds=`` (or
  params) under which the loop does stop.
- **CPU against CUDA.** Determinism is checked per device. CPU and CUDA results may
  differ (see :ref:`generator-hard-rules`), and the kit never compares them.
- **One configuration per call.** The checks exercise the parameters you pass. Run the
  kit once per setting that selects a different code path.
- **Masked regeneration and dynamic parameters.** The kit does not drive
  ``generate(advance=True, mask=...)`` or ``params.set``; test a Stage's dynamic
  parameters as :ref:`plugin-own-tests` shows. (A masked regeneration is exact by
  construction: the commit restores the envs outside the mask; see
  :ref:`pipeline-commit`.)
- **A stage in another position or pipeline.** ``check_stage`` runs the stage in its
  default slot of ``presets.track`` with the ``upstream`` you give. A stage you insert
  elsewhere, or rewire with ``ports``, needs a test of that pipeline.
- **Track quality.** One valid env is enough for ``output.closed``; yield, shape and
  speed are for ``benchmarks/compare_generators.py``.
