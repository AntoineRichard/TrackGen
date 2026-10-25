CUDA Graph — capture, replay, and buffer reuse
================================================

This page covers the mechanics behind recording a generator into a CUDA graph. The rules
you need as a user are elsewhere: :term:`seeds <seed>` and :term:`epochs <epoch>` in
:ref:`generating-seeds`, what ``generate()``, ``enqueue()`` and ``stream=`` do in
:ref:`generating-calls`, and the rules for recording into your own graph in
:ref:`capture-rules`.

``TrackGenerator`` and ``GateGenerator`` own the production generation path. Construction
resolves the selected generator, pre-allocates the persistent ``Track`` (or
``GateSequence``), all per-stage scratch groups, the ``[E]`` seed buffer and the per-env
seed state (``seeds``, with its ``[E]`` epoch array), then runs the pipeline once eagerly.
``generate()`` always returns that same result instance with stable ``wp.array`` pointers;
callers that need a snapshot use ``Track.clone()``.

A :term:`generation` can be recorded into YOUR CUDA graph (a :term:`capture`:
``wp.ScopedCapture``, ``wp.capture_begin(external=True)`` or ``torch.cuda.graph``) next to
your physics and the per-step utilities. Called outside any capture on ``cuda``, it
replays the generator's own CUDA graph instead, which this page calls its *self-graph*
(on ``cpu`` a generator has none: such calls run eagerly).

.. _cuda-graph-masked-regeneration:

Masked regeneration
-------------------

``generate(advance=True, mask=m)`` advances the epoch of the :term:`envs <env>` with
``m[e] != 0`` only, then runs the whole body with the mask: the seed prologue re-derives
the effective seed of the masked envs only, and only their ``generation`` is written. The
pipeline has fixed shapes, so it still recomputes EVERY env; the ``Track`` rows of the unmasked envs are
held before the stages and restored, bit for bit, by the commit after them (see
:ref:`pipeline-commit`). An unmasked env therefore ends the generation with exactly the
bytes it had, whatever its recomputation gave: masked regeneration does not rely on the
generators being deterministic. Only the masked envs get new output, whatever happened to
the other envs' seeds or epochs since their last generation: a reseed or a
``reset_epoch()`` made in between reaches an env at its next masked generation, or every
env at the next generation without a mask. An all-zero mask is therefore an exact no-op.
A masked generation costs three launches more than a plain ``generate()``: the epoch
advance, then the hold and the commit (two more than an unmasked
``generate(advance=True)``).

The unmasked rows are still overwritten for a moment: the stages write their
recomputation, then the commit restores the held bytes, both on the generation's stream.
Anything ordered after the generation on that stream (including the rest of a caller's
graph recorded on it, or a stream waiting on an event recorded after it) sees only the
restored rows. A reader on another stream with no dependency on the generation may see
an unmasked env's recomputed track mid-generation; order it after the generation.

``m`` must be a contiguous ``[E]`` int32 array on the generator's device, passed with
``advance=True``; the facade checks both on the host (``ValueError``) before any launch.
Its contents are read on device when the body runs, but the array itself is recorded by
pointer:

- Under your capture, keep the same array and rewrite it in place between replays, or
  from your own kernels inside the graph.
- Outside a capture on ``cuda``, the self-graph is cached per ``(advance, mask array)``:
  the first call with a new combination records a graph of its own, so pass the same
  array on every call. ``graph`` stays the graph of a plain ``generate()``. A producer
  caches at most 16 such graphs (``track_gen.exec.runner.MAX_KEYED_GRAPHS``); calls with
  yet another mask array then run eagerly, with a one-time ``RuntimeWarning``.

``inputs["regen_mask"]`` is a persistent, zero-initialised ``[E]`` int32 array owned by
the producer: a stable pointer for your kernels to write the mask into. The library reads
it and never clears it.

``Course.regenerate(mask=m, advance=...)`` adds the course-level part: the downstream
refresh, a per-env reset of the masked envs' runtime state only, and the conditional skip
below (see :ref:`rl-masked-regeneration` and :ref:`cuda-graph-sim-loop`).

Conditional skip
----------------

``LaunchContext.skip_unless(flag, body)`` (``track_gen.exec``) runs ``body(ctx)`` subject
to ``flag[0]``, a ``[1]`` int32 device array. ``track_gen.ops.any_set(ctx, mask, flag)``
writes that flag on device: it zeroes it, then sets it to ``1`` if any ``mask[e] != 0``
(an integer ``atomic_max``, so the result is deterministic).
``Course.regenerate(mask=...)`` puts its whole body, the epoch advance included, behind
``any_set`` + ``skip_unless``, and ``LaunchContext.bounded_loop`` puts each window of an
iterative stage behind it (see `Iterative stages (BOUNDED)`_).
``track_gen.ops.all_set(ctx, mask, flag)`` is the "every env" twin of ``any_set``.

``skip_unless`` records the body under a CUDA :term:`conditional node` (``wp.capture_if``),
which skips it on device when ``flag[0] == 0``, only when BOTH of these hold (it then
returns ``True``; ``False`` when it ran the body unconditionally):

- the context's stream is capturing, in a capture that Warp registered on that stream:
  ``wp.ScopedCapture``, ``wp.capture_begin`` (``external=True`` included, except a
  hand-made registration of the ``torch.cuda.graph`` capture running in the calling
  thread), a producer's own self-graph, or a ``torch.cuda.graph`` inside
  ``track_gen.interop.torch.registered(conditional=True)``;
- ``wp.is_conditional_graph_supported()`` is true: the CUDA driver and the CUDA toolkit
  Warp was built with are both 12.4 or newer.

Everywhere else (eager calls, the ``cpu`` device, a plain ``torch.cuda.graph``, one
registered without ``conditional=True`` or by hand with
``wp.capture_begin(stream=..., external=True)`` on torch's capture stream, a side stream
that only joined a capture registered on another stream, an older driver or toolkit,
inside ``track_gen.exec.unconditional()``) ``body`` runs, or is recorded,
unconditionally. It never takes Warp's eager ``capture_if`` path, which reads the flag
back on the host and synchronizes the stream (``skip_unless`` raises ``ValueError`` for a
flag that is not on the context's device, where ``capture_if`` would take that path).

The contract that makes both lowerings equivalent: ``body`` must be an exact no-op when
``flag[0] == 0``, leaving every buffer bitwise unchanged. ``Course.regenerate(mask=...)``
meets it unconditionally: with no env masked nothing advances, no effective seed is
re-derived (so the recompute reproduces the current values even if seeds or epochs were
changed since the last generation) and no env is reset. Running the body unconditionally
is therefore always correct; it only costs a full generation per replay (:ref:`rl-cost`).

A graph that holds a conditional node cannot be used as a CUDA child graph: you cannot
pass it to ``wp.capture_if(cond, on_true=graph)`` or insert it into another graph. If you
nest your step graph that way, nest through a Python callback instead (record the step's
calls inside the ``on_true`` callback), or keep the conditional out of it: record inside
``track_gen.exec.unconditional()``, record without a Warp registration, or use a plain
``regenerate()``. The same holds for a graph that records a ``BOUNDED`` stage
(``repulsive``) in a registered capture, and for such a producer's own ``graph``: its
windows are conditional nodes (unless its first CUDA call, which captures it, ran inside
``unconditional()``).

``wp.capture_if`` ends the stream's capture and resumes it around each conditional node.
CUDA refuses to end a capture while a side stream that joined it (``s2.wait_event(...)``
on an event recorded on the capture stream) has not been joined back
(``cudaErrorStreamCaptureUnjoined``), and the caller's capture is then invalidated. Join
every forked stream back before a call that records a conditional node (a masked
``Course.regenerate``, any generation with a ``BOUNDED`` stage), or record that call
inside ``unconditional()``:

.. snippet: group=unconditional

.. code-block:: python

   import warp as wp

   from track_gen import TrackGenerator
   from track_gen.exec import unconditional
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(num_envs=8, device="cuda:0", generator="repulsive"))
   with wp.ScopedCapture(device="cuda:0") as cap:
       with unconditional():           # no conditional node in this recording
           gen.generate()              # repulsive: every window recorded
   wp.capture_launch(cap.graph)
   # cap.graph can be nested as a child graph; the replays give the same bytes

A Warp capture of torch's current stream (``torch.cuda.stream(t)`` +
``wp.ScopedStream(wp.stream_from_torch(t))`` + ``wp.ScopedCapture()``) records no
conditional node by default: Warp does not record who began a capture, and a hand-made
``capture_begin(external=True)`` of a running ``torch.cuda.graph`` must not get one. The
results are the same; to keep the device-side skips, record inside
``track_gen.exec.conditional()`` (only when torch is not the capturer). A producer's own
``graph`` is unaffected.

Under ``torch.cuda.graph``, enter ``track_gen.interop.torch.registered()`` inside the
torch capture. It resolves torch's current stream to the library's long-lived Warp
wrapper and, unless Warp already has a capture registered on that handle (or the stream
only joined one), opens an external Warp capture on it
(``wp.ScopedCapture(stream=s, external=True)``: no new CUDA capture, only a registration of
torch's), closes it with the same object on exit, and yields the ``wp.Stream``. Outside a
torch capture it raises ``CaptureError``. By default the registration serves lifetime
anchoring only; ``registered(conditional=True)`` also allows the conditional skip. It is
opt-in because ``wp.capture_if`` ends and restarts the stream's capture around the node,
which gives the capture a new CUDA capture id, and torch's caching allocator recognises
its graph's capture by the id it saw at ``capture_begin``. After the first conditional
node, torch ops later in the same ``torch.cuda.graph`` (inside or after the
``registered()`` block) no longer allocate from the graph's private pool: they fail to
capture or take blocks from the general pool that eager code can reuse while the graph
still writes them. With ``conditional=True``, record the masked regeneration (and any
generation with a ``BOUNDED`` stage, whose windows are conditional nodes too) after every
torch op that allocates, or preallocate those outputs. A capture you register by hand
with ``wp.capture_begin(stream=..., external=True)`` on torch's running capture stream
follows the same default: no conditional node unless it goes through
``registered(conditional=True)``.

.. snippet: group=torch-conditional cuda

.. code-block:: python

   import torch
   import warp as wp

   from track_gen.course import Course
   from track_gen.interop.torch import registered, retain
   from track_gen.pipeline import presets

   E = 64
   course = Course.track(presets.track(num_envs=E, device="cuda:0"), checkpoint_spacing=0.6)
   done = wp.zeros(E, dtype=wp.int32, device="cuda:0")   # your kernels write it each step
   pose = torch.zeros(E, 3, device="cuda:0")

   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       obs = pose * 2.0             # torch ops that allocate: BEFORE the regeneration
       with registered(conditional=True) as s:   # torch's stream, registered with Warp
           course.regenerate(mask=done, advance=True)   # conditional node, recorded last
   retain(g, course)                # keep course alive as long as g
   g.replay()

Pass ``capture_mode=`` when torch captures with a non-default ``capture_error_mode``, and
``graph=g`` to retain on ``g`` the producers recorded inside the block (see :ref:`Lifetime <capture-lifetime>`).

.. _cuda-graph-bounded:

Iterative stages (BOUNDED)
--------------------------

A stage whose work is an iteration that may stop early (a bounded optimizer whose
envs can freeze before the budget) declares ``capture = CaptureClass.BOUNDED`` and emits its loop
through ``LaunchContext.bounded_loop``:

.. snippet: skip=signature

.. code-block:: python

   ctx.bounded_loop(n_iters, body, *, active=None, first_check=0, check_every=1)

``body(ctx, it)`` runs for ``it`` in ``range(n_iters)``, with ``it`` a plain Python
``int``: a cadence keyed on it (``if (it + 1) % 25 == 0``) is resolved on the host, and
the launch sequence of every iteration is static. Iterations ``[0, first_check)`` always
run. The remaining ones form windows of ``check_every`` iterations (the last one may be
shorter), each preceded by a check of ``active[0]``, a ``[1]`` int32 array on the
context's device: when it reads ``0``, that window and every later one may be skipped.
The body writes ``active`` on device, on the iteration just before each window starts;
``ops.any_set(ctx, running, active)`` over a per-env "still running" mask is the usual
way. With ``active=None`` the loop is a plain static unroll (no check, any stage may use
it).

``n_iters`` is a bound, not a count: how many iterations execute depends on where the call
runs. The lowering is chosen once per call, from the capture state, never from device data:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Where the loop runs
     - Lowering
   * - ``cpu``, eager, with no CPU graph capture open on the device
     - **(a) host poll.** ``active[0]`` is read on the host before each window, and the
       loop stops at the first ``0``. The read is four bytes, and CPU launches are
       synchronous.
   * - ``cuda``, eager: ``enqueue()``, ``policy="eager"``, a keyed-graph overflow, a
       ``Course`` over a ``HOST`` generator, the ``HOST`` segments of a segmented call
     - **(c) unrolled.** Every window runs, with no host sync, so ``enqueue()`` keeps its
       "never host-syncs" contract. A frozen env costs launch overhead only when the
       stage's kernels return early for it, as ``repulsive``'s do.
   * - Construction and ``prepare()`` (the warm-up)
     - **(c) unrolled**, so the warm-up loads every kernel a window can launch.
   * - Inside a capture that Warp registered on the context's stream (the producer's own
       self-graph, ``wp.ScopedCapture``, ``wp.capture_begin``,
       ``registered(conditional=True)``) when ``wp.is_conditional_graph_supported()``
     - **(b) windowed.** Each window is recorded under a conditional node
       (``skip_unless``, see `Conditional skip`_): replays skip the converged windows on
       device.
   * - Any other capture: a plain ``torch.cuda.graph``, ``registered()`` without
       ``conditional=True`` (or a hand-made ``external=True`` registration of torch's
       capture), a side stream that only joined another capture, a driver or toolkit
       older than 12.4, a CPU ``wp.ScopedCapture("cpu")``, any recording inside
       ``track_gen.exec.unconditional()``
     - **(c) unrolled** into the graph.

The three lowerings give bitwise identical results when the loop is **absorbing** (each
env stops changing once it converges, independently of the rest of the batch): once
``active[0]`` reads ``0`` at a check, every remaining iteration leaves everything read
after the loop (the stage's outputs, and any state or scratch it reads later) bitwise
unchanged. In practice, freeze an env on device when it converges, never unfreeze it
within the generation, and make every kernel that writes an output return early for a
frozen env. Two more rules complete the contract: the body writes ``active`` before the
first check (so a check never reads a value left over from an earlier generation), and it
never changes Python state that code after the loop reads, since the loop may stop early.
An absorbing per-env freeze also keeps the stage batch-independent: under (a), a loop that
keeps moving a converged env does so until the WHOLE batch has converged, so that env's
output depends on the other envs. The conformance kit checks the contract
(``bounded.absorbing`` on every device, ``bounded.lowering`` on CUDA; see
:doc:`/contributing/testing-a-plugin`).

A stage that calls ``bounded_loop(active=...)`` must not declare ``STATIC`` (declare
``BOUNDED``): one declared ``STATIC`` raises ``ValueError`` (``... declare capture =
CaptureClass.BOUNDED``) on its first run, at construction. ``BOUNDED`` stages are
capturable by anything: they share a segment with the ``STATIC`` stages around them, and
a caller can record them.

The host poll of lowering (a) is the one host readback the library itself performs on a
capturable path (a ``HOST`` stage's ``ctx.host()`` operations aside), and only on ``cpu``. :func:`~track_gen.testing.no_host_sync` lets it through and
counts it in ``HotPathGuard.sanctioned`` (not in ``allowed``), as it does a ``HOST``
stage's ``ctx.host()`` operations; any other sync, including one made inside the loop's
body, is still a violation. The conformance kit runs its hot-path checks with every
window unrolled, the superset of what any lowering launches.

``repulsive`` is the built-in ``BOUNDED`` stage. At final-resolution stall
checks it writes ``active = any env not frozen``. The host derives the first
window boundary as ``W * ceil((s_final + 1) / W)``, where ``W`` is
``repulsive_stall_window`` and ``s_final`` is the scheduled start iteration of the
final resolution; subsequent boundaries are ``W`` iterations apart. The total
bound follows the configured growth and settle budgets. Frozen envs remain
unchanged, so skipping later windows does not change the result. A feasible
target-reaching curve may also finish valid at the full bound without freezing;
optional early exit is separate from validity (:doc:`/generators/repulsive`).

.. _cuda-graph-warm-up:

Warm-up at construction
-----------------------

Construction runs the pipeline ONCE eagerly, outside any capture, with every window of a
``bounded_loop`` run. Launching for real loads every kernel module, generic-kernel
overload and Warp-internal copy kernel the hot path uses, so a later recording (the generator's own or yours) never compiles or loads a module
mid-capture. The pass also leaves the batch for the rng's current seeds in the result,
which is valid right after construction: construction blocks until the pass has finished
on its stream, so a later call on another stream, or a readback, never races it.

Construct generators (and ``Course`` objects) before you begin a capture. Construction
allocates and zero-fills on Warp's current stream, so it raises ``CaptureError`` (from
``track_gen.exec``) before allocating anything when the stream it resolves or Warp's
current stream is capturing, when torch is capturing in the calling thread, or, on
``cpu``, when a CPU ``wp.ScopedCapture("cpu")`` is open (it would record the warm-up
instead of running it).

For plugin authors, a generator's Python body runs as follows:

.. list-table::
   :header-rows: 1
   :widths: 46 24 30

   * - Moment
     - Python body runs
     - Kernels execute
   * - Construction (warm-up pass)
     - once
     - once
   * - First self-graph ``generate()`` (recording), per ``(advance, mask)``
     - once
     - once, by the launch that follows
   * - Every later self-graph ``generate()``
     - not at all (a ``HOST`` stage: once per call)
     - once per call
   * - Eager call (``cpu``, ``enqueue()``)
     - once per call
     - once per call
   * - Recording into a caller's graph
     - once, at record time
     - once per caller replay

Buffer contents persist across replays. Host scalars and Python state freeze at record
time: anything that must change between replays has to be device data (the seeds, bound
buffers).

.. _cuda-graph-host:

HOST stages and segments
------------------------

A stage that must read device data back or run host control flow declares ``capture =
CaptureClass.HOST``; a ``GeneratorSpec``
registered with ``capturable=False`` is a ``HOST`` generator too. A ``HOST`` stage runs
eagerly, outside every capture, on every call. It no longer makes the whole pipeline
eager: the planner groups the stages into **segments**, maximal runs of consecutive
stages that are all ``HOST`` or all capturable (``STATIC`` and ``BOUNDED`` stages share a
segment), and the ``Program`` reports them:

.. snippet: skip=plugin-package

.. code-block:: python

   program = gen.program               # a generator whose "generator" stage is HOST
   [(s.capture_class.name, s.stage_ids[:2]) for s in program.segments]
   # [('HOST', ('generator',)), ('STATIC', ('resample', 'relax'))]
   program.host_stage_ids              # ('generator',)
   program.capture_class               # CaptureClass.HOST, the highest of its stages'
   program.capturable                  # False: a caller cannot record it

How such a generator runs:

- **Outside a capture on** ``cuda`` (segmented self-graph): the first call records each
  capturable segment into a graph of its own, on the private stream. Every call then
  runs the segments in order on the resolved stream: a ``HOST`` segment eagerly (its
  Python runs every call), the others with ``wp.capture_launch``. The seed prologue
  belongs to the first segment, the commit and the ``generation`` write to the last.
  ``gen.graph`` stays ``None``; masked regeneration keys a set of segment graphs per
  ``(advance, mask)`` like the plain self-graph (`Masked regeneration`_). A
  ``GeneratorSpec(capturable=False)`` runs as a ``HOST`` segment followed by a captured
  tail.
- **On** ``cpu`` **and with** ``enqueue()``: every segment eagerly, in order.
- **Under a caller's capture**: ``generate()``, ``enqueue()``, ``Program.run()`` and
  ``Program.enqueue()`` raise ``NotCapturableError`` before launching anything, so your
  graph is never left half-recorded. The error names the producer and the ``HOST``
  stages (its ``stage_ids`` attribute):
  ``'acme.blob' cannot be recorded into a graph capture: HOST stage(s) ['generator'] read
  device data back or run host control flow, so they run eagerly ...``. A CPU
  ``wp.ScopedCapture("cpu")`` counts as a capture here: it records launches without
  running them, so host reads would see stale data. ``Course.generate(seeds=...)``
  refuses before its reseed launches.

An advanced caller may record the capturable segments itself and run the ``HOST`` ones
eagerly, as the segmented self-graph does, with ``Program.enqueue_segment(i, ctx,
advance=False, mask=None)`` (see :ref:`pipeline-program`). To make sure a pipeline can be
recorded whole, build it with ``PipelineSpec(..., require_caller_capturable=True)``: the
planner then rejects every ``HOST`` stage at build, naming them, before allocating
anything.

A ``HOST`` stage reaches the host through ``ctx.host()``, a
:class:`~track_gen.exec.HostOps`: ``read_i32(arr, index=0)``, ``readback(src, via=...)``,
``upload(dst, data, via=...)`` and ``sync()``. Every transfer runs on the call's stream
through a pinned host buffer the stage allocated at build with
``StageBuilder.host_buffer(dtype, shape)``, so nothing allocates per call (on ``cuda`` a
``via=`` buffer that is not pinned raises ``ValueError``). ``ctx.host()`` raises
``ValueError`` inside a stage that does not declare ``HOST`` (naming it and its class),
``CaptureError`` outside every pipeline stage, and ``NotCapturableError`` when the call's
stream is capturing or a CPU graph capture is open on the device. :ref:`stage-host` shows
a ``HOST`` stage.

A ``HOST`` stage may run while other threads capture on the device, provided it reaches
the host only through ``ctx.host()`` with ``b.host_buffer`` buffers: its copies are
stream-ordered asynchronous copies between device memory and pinned host buffers, and its
syncs run with the thread's capture mode relaxed. A direct ``.numpy()`` or
``wp.synchronize*()`` in a ``HOST`` stage allocates or syncs in the default mode, and can
invalidate another thread's capture in torch's default global mode.

Plugin launch contexts
~~~~~~~~~~~~~~~~~~~~~~

Every plugin generator emits device work through the call's launch context. A
``GeneratorSpec`` generate callable must accept ``ctx=``; a Stage returns
``enqueue(ctx)``. This lets a caller record either form on its own stream.

Buffer reuse
------------

All persistent buffers — the output ``Track`` arrays, per-stage scratch (generator scratch,
resample buffers, relax scratch including broadphase cache arrays), the seed buffer, the
epoch array and ``inputs["regen_mask"]`` — are allocated once at construction and reused
across every ``generate()`` call. There is no per-call allocation on the hot path.

``RelaxScratch`` holds the double buffer pair (positions A and B), the ``band`` and ``L0``
arrays, and — in cached broadphase mode — the fixed-slot candidate list and overflow
counter. All addresses are stable across replays.

Performance characteristics
----------------------------

At large batches the pipeline is compute-bound (relaxation dominates), so graph replay is
approximately the same wall-clock as the eager call. The graph's value is a single,
GPU-resident, deployable replayable unit — not a raw speedup — that eliminates kernel
launch overhead and driver overhead for the full pipeline in a single ``capture_launch``
call. Recorded into your graph, generation needs no launch of its own: it replays with the
rest of your step.

.. note::

   On ``cpu`` a generator has no self-graph: every call outside a capture runs eagerly.
   A CPU ``wp.ScopedCapture(device="cpu")`` records launches without running them and
   ``wp.capture_launch`` replays them, so recording into your graph works on ``cpu`` too,
   with every bounded window unrolled (no conditional nodes). On ``cuda``, a ``HOST``
   stage always runs eagerly while the rest of its pipeline replays from graphs, and a
   caller can record only a pipeline without ``HOST`` stages.
