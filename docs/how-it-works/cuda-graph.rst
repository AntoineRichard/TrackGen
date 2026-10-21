CUDA Graph — capture, replay, and buffer reuse
================================================

``TrackGenerator`` and ``GateGenerator`` own the production generation path. Construction
resolves the selected generator, pre-allocates the persistent ``Track`` (or
``GateSequence``), all per-stage scratch groups and the ``[E]`` seed buffer, then runs the
pipeline once eagerly. ``generate()`` always returns that same result instance with stable
``wp.array`` pointers; callers that need a snapshot use ``Track.clone()``.

Generation can be recorded into YOUR CUDA graph (``wp.ScopedCapture``,
``wp.capture_begin(external=True)`` or ``torch.cuda.graph``) next to your physics and the
per-step utilities. Called outside any capture, it replays the generator's own graph
instead.

Runtime facade
--------------

The public interface of both facades:

- ``__init__(config, rng, *, stream=None)`` — validates the config, keeps a private copy
  of it, allocates every persistent buffer and runs one eager warm-up pass (see
  `Warm-up at construction`_). ``stream=`` is the default stream for calls that name none.
- ``generate(num_or_ids=None, *, stream=None, sync=True)`` — copies the rng's current seeds
  into the seed buffer, runs the pipeline (recorded, self-graph replay or eager; see
  `Execution modes`_) and returns the stable result.
- ``enqueue(*, stream=None)`` — the same body, recorded into the active capture or run
  eagerly on the resolved stream. It never builds a self-graph and never blocks.
- ``graph`` — the generator's own ``wp.Graph``. It is ``None`` until the first ``cuda``
  ``generate()`` outside a capture, and stays ``None`` on ``cpu`` and for non-capturable
  generators.
- ``config`` — a copy of the construction config (a ``wp.Device`` value is shared, not
  copied). Mutating it, or the config object you passed in, has no effect on the
  generator: construct a new one to change settings.
- ``Track.clone()`` / ``GateSequence.clone()`` — snapshot the current output arrays.

Execution modes
---------------

Each call resolves ONE stream: an explicit ``stream=`` (a ``wp.Stream`` or ``"torch"``),
then an enclosing ``LaunchContext.scope()``, then the constructor's ``stream=``, then
torch's capture stream while ``torch.cuda.graph`` is capturing on the device, else Warp's
current stream. The state of that stream picks the mode:

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Mode
     - When
     - What ``generate()`` does
   * - Record
     - The resolved stream is capturing.
     - Enqueues the seed copy and the whole pipeline on that stream, so they become nodes
       of the caller's graph. There is no self-capture, no ``wp.capture_launch`` (a graph
       launch cannot be recorded into another capture) and no host sync, whatever
       ``sync`` says.
   * - Self-graph
     - ``cuda``, a capturable generator, not capturing.
     - The first call records the body once into the generator's own graph on a private
       stream. Every call then launches that graph on the resolved stream with
       ``wp.capture_launch``.
   * - Eager
     - ``cpu``, a ``capturable=False`` generator, or ``enqueue()`` outside a capture.
     - Launches the pipeline directly on the resolved stream.

The seed copy (``rng.seeds_warp`` into the seed buffer) is part of the recorded body, in
the self-graph and in a caller's graph alike. Every replay therefore reads the rng's
CURRENT seeds: ``rng.set_seeds_warp(new_seeds, None)`` writes into the rng's existing
buffer, and the next replay generates the new batch with no re-capture.

Outside a capture, ``generate()`` waits for its stream to finish before returning
(``sync=True``, the default). The wait covers that one stream; earlier releases called the
device-wide ``wp.synchronize()``. It is skipped while torch is capturing in the calling
thread (a host sync there could invalidate torch's capture), and on Warp 1.18 or later it
runs with the thread's CUDA capture mode relaxed, so it never invalidates a capture another
thread has open. ``generate(sync=False)`` returns at once, and the results are
stream-ordered like every other library call (see :doc:`/utilities/overview`).

A call is recorded only when the stream it RESOLVES is capturing. When you capture a
stream that is not Warp's current one, pass it (``generate(stream=s)``) or wrap the calls
in ``LaunchContext(device, s).scope()``. Otherwise the call runs, or self-graphs, on
another stream and is silently missing from your graph. A generator constructed with
``stream=`` resolves to that stream before torch's capture stream, so inside
``torch.cuda.graph`` pass ``stream="torch"`` to it.

``Course.regenerate()`` follows the same three modes with one body: generation plus the
downstream refresh (see :doc:`/utilities/course`).

Warm-up at construction
-----------------------

Construction runs the pipeline ONCE eagerly, outside any capture. Launching for real loads
every kernel module, generic-kernel overload and Warp-internal copy kernel the hot path
uses, so a later recording (the generator's own or yours) never compiles or loads a module
mid-capture. The pass also leaves the batch for the rng's current seeds in the result,
which is valid right after construction: construction blocks until the pass has finished
on its stream, so a later call on another stream, or a readback, never races it. It
replaces the former first-call warm-up: three eager runs, a blocking ``wp.synchronize()``
and a capture under a process-wide lock.

Construct generators (and ``Course`` objects) before you begin a capture. Construction
allocates and zero-fills on Warp's current stream, so it raises ``CaptureError`` (from
``track_gen.exec``) before allocating anything when the stream it resolves or Warp's
current stream is capturing, or when torch is capturing in the calling thread.

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
   * - First self-graph ``generate()`` (recording)
     - once
     - once, by the launch that follows
   * - Every later self-graph ``generate()``
     - not at all
     - once per call
   * - Eager call (``cpu``, ``capturable=False``, ``enqueue()``)
     - once per call
     - once per call
   * - Recording into a caller's graph
     - once, at record time
     - once per caller replay

Buffer contents persist across replays. Host scalars and Python state freeze at record
time: anything that must change between replays has to be device data (the seeds, bound
buffers).

Capture requirements
--------------------

The capture works because every stage is pure Warp and fixed-shape:

- No stage host-synchronizes or reads device data back. Host-blocking syncs are illegal
  inside capture, and the stream (or the graph) already orders the launches. The
  wrappers' ``_sync`` hook is a no-op unless ``TRACKGEN_DEBUG_SYNC=1``, and even then it
  is skipped while the stream is capturing.
- Nothing allocates on a ``generate()`` / ``enqueue()`` path: every buffer is allocated at
  construction. Recording into your graph works with Warp's mempool disabled.
- The seed buffer address is stable; replay reuses the same buffer and reads the new seed
  contents on device.
- ``output_mode="constant_spacing"`` captures too: per-track ``count[e]`` is device-side
  data, and count-aware kernels keep static launch dimensions via ``N_max``.
- Generator selection and ``relax_enable`` are Python branches resolved before capture. The
  captured graph is fixed for that generator's config, and the config is snapshotted at
  construction, so it cannot drift afterwards.

Non-capturable and legacy plugin generators
-------------------------------------------

A generator registered with ``capturable=False`` (``GeneratorSpec`` or
``GateGeneratorSpec``) runs eagerly on ``cuda`` every call and never gets a self-graph.
Under a caller's capture, its ``generate()`` and ``enqueue()`` raise ``NotCapturableError``
before launching anything, so your graph is never left half-recorded.

A plugin generator that does not set ``accepts_ctx=True`` (the legacy ABI) issues its own
``wp.launch`` calls on Warp's current stream rather than on the stream the call resolved.
Its self-graph is therefore captured on Warp's current stream, behind the old process-wide
lock. Outside a capture, when a call resolves another stream (``stream=``, a constructor
``stream=``, an ambient scope), the whole body runs on Warp's current stream instead,
ordered after and before the resolved stream with events, so the plugin's launches and the
library's stages stay ordered. Recording it into your graph requires the capture stream to
BE Warp's current stream; otherwise the call raises ``LegacyPluginStreamError`` before any
launch. Capture Warp's current stream (a plain ``wp.ScopedCapture(device)``), or make your
capture stream current before the capture begins. With ``torch.cuda.graph``, pass it that
explicit stream (a plain ``torch.cuda.graph(g)`` captures on torch's internal stream, which
you cannot make current beforehand):

.. code-block:: python

   s = torch.cuda.Stream()
   g = torch.cuda.CUDAGraph()
   with wp.ScopedStream(wp.stream_from_torch(s)):   # Warp's current stream is now s
       with torch.cuda.graph(g, stream=s):           # torch captures that same stream
           gen.generate()

Every shipped generator sets ``accepts_ctx=True`` and has neither restriction.

Inside ``torch.cuda.graph``, adoption reuses the ``wp.Stream`` that already wraps torch's
capture stream (Warp's current stream, as in the recipe above, or the stream an open Warp
capture was started on), and otherwise one wrapper per torch stream kept for the process.
It never creates a short-lived wrapper per call: dropping one would unregister the handle
from Warp for the whole process, breaking an open Warp capture on it and Warp's current
stream.

Lifetime
--------

A ``wp.Graph`` keeps its kernel modules alive, not the buffers its nodes read and write.
When ``generate()`` / ``enqueue()`` is recorded into a capture that Warp registered
(``wp.ScopedCapture``, or ``wp.capture_begin``, including ``external=True``), the generator
attaches itself to the resulting ``wp.Graph``, so its buffers live as long as that graph.
Warp does not see a ``torch.cuda.graph`` capture. Keep the generator (or ``Course``) alive
for as long as you replay a torch graph.

Threads
-------

There is no process-wide lock any more: a built-in generator captures its self-graph on
its own private stream, so independent generators on different threads run concurrently.
Each generator (and ``Course``) is single-writer, though. Do not run the same instance
from two threads, or on two streams, at once; use one instance per concurrent producer.

The exception is a producer built on a legacy plugin (no ``accepts_ctx=True``). It keeps
the process-wide lock, and its one-time self-capture records Warp's current stream, which
every thread shares. Until every such producer has captured its self-graph, calls of other
producers that resolve to Warp's current stream (the default) take the lock too, and
decide whether they are being recorded only once they hold it; otherwise they could
mistake the legacy capture for their caller's and record themselves into it. Calls on any
other stream (``stream=``, a constructor ``stream=``, a scope) stay lock-free.

Two caveats apply to multi-threaded programs:

- While a self-graph is being recorded (once per generator, on its first ``cuda``
  ``generate()``), Warp refuses ``wp.synchronize()``, ``wp.synchronize_device()`` and
  ``wp.synchronize_stream()`` on that device from every thread. ``generate()``'s own wait
  is not affected. If other threads call those functions, run each generator's first
  ``generate()`` before starting them.
- The private capture stream is a blocking stream. Work that another thread issues on the
  legacy default stream during that one-time capture can invalidate it, so the library
  retries the capture once before raising.

Buffer reuse
------------

All persistent buffers — the output ``Track`` arrays, per-stage scratch (generator scratch,
resample buffers, relax scratch including broadphase cache arrays), and the seed buffer —
are allocated once at construction and reused across every ``generate()`` call. There is no
per-call allocation on the hot path.

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

   ``cpu``-device runs are always eager. Graph capture, self-graph or caller recording, is
   ``cuda``-only, and even on ``cuda`` it only applies to ``capturable=True`` generators.
