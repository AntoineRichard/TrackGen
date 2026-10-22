CUDA Graph — capture, replay, and buffer reuse
================================================

``TrackGenerator`` and ``GateGenerator`` own the production generation path. Construction
resolves the selected generator, pre-allocates the persistent ``Track`` (or
``GateSequence``), all per-stage scratch groups, the ``[E]`` seed buffer and the per-env
seed state (``seeds``, with its ``[E]`` epoch array), then runs the pipeline once eagerly.
``generate()`` always returns that same result instance with stable ``wp.array`` pointers;
callers that need a snapshot use ``Track.clone()``.

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
- ``generate(num_or_ids=None, *, stream=None, sync=True, advance=False, mask=None)`` —
  with ``advance=True`` first advances the epoch of every env (of the envs with
  ``mask[e] != 0`` when ``mask`` is given; ``mask`` requires ``advance=True``), then runs
  the seed prologue and the pipeline (recorded, self-graph replay or eager; see
  `Execution modes`_), writes the result's ``generation`` and returns the stable result.
  See `Seeds, epochs and the generation counter`_ and `Masked regeneration`_.
- ``enqueue(*, stream=None, advance=False, mask=None)`` — the same body, recorded into the
  active capture or run eagerly on the resolved stream. It never builds a self-graph and
  never blocks.
- ``graph`` — the generator's own ``wp.Graph`` for a plain ``generate()`` (no ``advance``,
  no ``mask``). It is ``None`` until the first such ``cuda`` call outside a capture, and
  stays ``None`` on ``cpu`` and for non-capturable generators. Every other
  ``(advance, mask)`` combination gets a self-graph of its own (see `Masked
  regeneration`_).
- ``seeds`` — the :class:`~track_gen.seeds.SeedState`: per-env seeds (bound to
  ``rng.seeds_warp``) and epochs.
- ``inputs`` — ``{"regen_mask": ...}``, a persistent, zero-initialised ``[E]`` int32
  array with a stable address, for your own kernels to write a mask into, plus every
  per-env input a Stage generator declares.
- ``params`` — ``{"generator": StageParams}``: the resolved generator parameters and, for
  a Stage generator, the live ``[E]`` array of each dynamic parameter, written in place
  between or inside replays with no re-capture (see
  :ref:`generator-runtime-params`).
- ``config`` — a copy of the construction config (a ``wp.Device`` value is shared, not
  copied), with a ``generator_params`` mapping normalised to its Params instance.
  Mutating it, or the config object you passed in, has no effect on the generator:
  construct a new one to change settings, or write a dynamic parameter.
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
     - Enqueues the seed prologue and the whole pipeline on that stream, so they become
       nodes of the caller's graph. There is no self-capture, no ``wp.capture_launch`` (a
       graph launch cannot be recorded into another capture) and no host sync, whatever
       ``sync`` says.
   * - Self-graph
     - ``cuda``, a capturable generator, not capturing.
     - The first call records the body once into the generator's own graph on a private
       stream (one graph per ``(advance, mask)`` combination). Every call then launches
       that graph on the resolved stream with ``wp.capture_launch``.
   * - Eager
     - ``cpu``, a ``capturable=False`` generator, or ``enqueue()`` outside a capture.
     - Launches the pipeline directly on the resolved stream.

The seed prologue (one kernel that derives the seed buffer from ``rng.seeds_warp`` and the
epochs, see `Seeds, epochs and the generation counter`_) is part of the recorded body, in
the self-graph and in a caller's graph alike. Every replay therefore reads the rng's
CURRENT seeds and the CURRENT epochs: ``rng.set_seeds_warp(new_seeds, None)`` writes into
the rng's existing buffer, and the next replay generates the new batch with no
re-capture. A recorded ``generate(advance=True)`` advances the epochs on device, so each
replay generates a new batch without any reseed.

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

Seeds, epochs and the generation counter
----------------------------------------

Each env is keyed by a pair ``(seed, epoch)``, held by the facade's
:class:`~track_gen.seeds.SeedState` (``gen.seeds``):

- ``seeds`` — ``[E]`` int32. A facade binds ``rng.seeds_warp`` by pointer, so
  ``rng.set_seeds_warp(...)`` keeps working as before, and the rng must live on the
  generator's device (construction raises ``ValueError`` otherwise). ``SeedState`` and
  the generators never write a seed array they were given (a ``wp.array`` or a
  ``PerEnvSeededRNG``'s): such a state is ``bound``, and ``set()`` on it raises
  ``RuntimeError``. Write that array yourself instead. The rng is another matter:
  ``PerEnvSeededRNG.set_seeds_warp`` and ``Course.generate(seeds=...)`` overwrite
  ``rng.seeds_warp`` in place, and that IS your array when the rng (or
  ``CourseConfig.seeds``) was built from a ``wp.array``. A ``SeedState`` built from
  ``None`` or an ``int`` ``k`` owns its seeds (``k + arange(E)``, as
  ``PerEnvSeededRNG(seeds=k)``), as does one built with ``copy=True``.
- ``epoch`` — ``[E]`` int32, owned, zero at construction and changed only on device:
  ``advance(mask=None)`` adds 1 where ``mask[e] != 0`` (to every env without a mask),
  ``reset_epoch(mask=None)`` zeroes it the same way, and ``set(seeds, ids=None)`` scatters
  new seeds into owned seeds and zeroes those envs' epochs. Each is one launch (``set``
  with ``ids=None``: a copy and a launch) on the call's stream, allocation-free, so it can
  be recorded into a graph. Without ``stream=``, a facade's ``seeds`` methods run on the
  facade's constructor ``stream=``, like its ``generate()``.

Every generation body opens with the seed prologue (right after the epoch advance when
``advance=True``): one kernel that folds the pair into the effective seed the pipeline
reads:

- epoch ``0``: the seed itself. A generator whose epochs are never advanced produces
  exactly the batches of earlier releases (the goldens are unchanged).
- epoch ``k != 0``: ``hash32(uint32(seed) ^ hash32(uint32(k) * 0x9E3779B9)) & 0x7FFFFFFF``,
  a non-negative int32, where ``hash32`` is the MurmurHash3 finaliser
  (``track_gen.ops.hash32``).

``(seed, epoch)`` fully determines an env's output: a generator whose env ``e`` has the
same seed and the same epoch rebuilds the same track or gate sequence. So does a generator
at epoch ``0`` seeded with ``track_gen.seeds.effective_seed(seed, epoch)``, the bit-exact
host twin of the prologue. Because the prologue is recorded with the rest of the body,
device-side writes to the seeds or to the epochs reach every replay, of your graph or of
the generator's own.

The body ends by writing ``generation[e] = epoch[e]`` into the result (for the masked
envs only, on a masked generation). ``Track.generation`` and ``GateSequence.generation``
(``[E]`` int32) therefore tell a consumer, on device, which epoch each env's current
output was built with. It is the epoch, not a change counter: a reseed at the same epoch
does not change it, and after ``reset_epoch()`` earlier values come back. It changes
exactly when an env gets new output only while nothing reseeds or resets the epochs. The
field defaults to ``None`` so results built by hand still construct, and ``clone()``
copies it.

Reseeding through the rng leaves the epochs alone: after an advance,
``rng.set_seeds_warp(new_seeds, None)`` gives the batch of ``(new_seeds, epoch)``, not
that of a fresh generator. Call ``gen.seeds.reset_epoch()`` as well to reproduce a fresh
generator; ``Course.generate(seeds=...)`` does both.

Masked regeneration
-------------------

``generate(advance=True, mask=m)`` advances the epoch of the envs with ``m[e] != 0`` only,
then runs the whole body with the mask: the seed prologue re-derives the effective seed
of the masked envs only, and only their ``generation`` is written. The pipeline has fixed
shapes, so it still recomputes EVERY env, but an unmasked env keeps the effective seed
its current output was built from and is recomputed to identical values (the generators
are deterministic per device). Only the masked envs get new output, whatever happened to
the other envs' seeds or epochs since their last generation: a reseed or a
``reset_epoch()`` made in between reaches an env at its next masked generation, or every
env at the next generation without a mask. An all-zero mask is therefore an exact no-op.
A masked commit that writes only the masked envs, and so avoids relying on the
recompute, is planned (roadmap PR10).

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
below (see :doc:`/utilities/course` and :doc:`/tutorials/cuda-graph-in-a-sim`).

Conditional skip
----------------

``LaunchContext.skip_unless(flag, body)`` (``track_gen.exec``) runs ``body(ctx)`` subject
to ``flag[0]``, a ``[1]`` int32 device array. ``track_gen.ops.any_set(ctx, mask, flag)``
writes that flag on device: it zeroes it, then sets it to ``1`` if any ``mask[e] != 0``
(an integer ``atomic_max``, so the result is deterministic).
``Course.regenerate(mask=...)`` puts its whole body, the epoch advance included, behind
``any_set`` + ``skip_unless``.

``skip_unless`` records the body under a CUDA conditional graph node (``wp.capture_if``),
which skips it on device when ``flag[0] == 0``, only when BOTH of these hold:

- the context's stream is capturing, in a capture that Warp registered on that stream:
  ``wp.ScopedCapture``, ``wp.capture_begin`` (``external=True`` included), a producer's
  own self-graph, or a ``torch.cuda.graph`` inside
  ``track_gen.interop.torch.registered(conditional=True)``;
- ``wp.is_conditional_graph_supported()`` is true: the CUDA driver and the CUDA toolkit
  Warp was built with are both 12.4 or newer.

Everywhere else (eager calls, the ``cpu`` device, a plain ``torch.cuda.graph`` or one
registered without ``conditional=True``, a side stream that only joined a capture
registered on another stream, an older driver or toolkit) ``body`` runs, or is recorded,
unconditionally. It never takes Warp's eager ``capture_if`` path, which reads the flag
back on the host and synchronizes the stream (``skip_unless`` raises ``ValueError`` for a
flag that is not on the context's device, where ``capture_if`` would take that path).

The contract that makes both lowerings equivalent: ``body`` must be an exact no-op when
``flag[0] == 0``, leaving every buffer bitwise unchanged. ``Course.regenerate(mask=...)``
meets it unconditionally: with no env masked nothing advances, no effective seed is
re-derived (so the recompute reproduces the current values even if seeds or epochs were
changed since the last generation) and no env is reset. Running the body unconditionally
is therefore always correct; it only costs a full generation per replay.

A graph that holds a conditional node cannot be used as a CUDA child graph: you cannot
pass it to ``wp.capture_if(cond, on_true=graph)`` or insert it into another graph. If you
nest your step graph that way, nest through a Python callback instead (record the step's
calls inside the ``on_true`` callback), or keep the conditional out of it: record the
masked regeneration without a Warp registration, or use a plain ``regenerate()``.

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
still writes them. With ``conditional=True``, record the masked regeneration after every
torch op that allocates, or preallocate those outputs.

.. code-block:: python

   from track_gen.interop.torch import registered, retain

   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       obs = compute_obs(...)       # torch ops that allocate: BEFORE the regeneration
       with registered(conditional=True) as s:   # torch's stream, registered with Warp
           course.regenerate(mask=done, advance=True)   # conditional node, recorded last
   retain(g, course)                # keep course alive as long as g

Pass ``capture_mode=`` when torch captures with a non-default ``capture_error_mode``, and
``graph=g`` to retain on ``g`` the producers recorded inside the block (see `Lifetime`_).

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
   * - First self-graph ``generate()`` (recording), per ``(advance, mask)``
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
- Nothing allocates on a ``generate()`` / ``enqueue()`` path, with or without ``advance``
  and ``mask``, nor in ``SeedState.advance()`` / ``set()`` / ``reset_epoch()``: every
  buffer is allocated at construction. Recording into your graph works with Warp's mempool
  disabled.
- The seed buffer, the bound seed array, the epoch array, the result's ``generation``,
  ``inputs["regen_mask"]`` and a Stage generator's dynamic-parameter and input arrays have
  stable addresses; replay reads their current contents on device. A ``mask=`` array is
  recorded by pointer: rewrite it in place.
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
for as long as you replay a torch graph, or tie it to the torch graph:
``track_gen.interop.torch.retain(g, gen)`` stores strong references on the torch graph
object ``g``, so its arguments live as long as ``g``. Inside
``track_gen.interop.torch.registered(graph=g)`` the capture is registered with Warp, so
a recorded generation anchors its producer as above, and the registration's Warp graph
object is retained on ``g`` when the block exits.

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

   ``cpu``-device runs are always eager. Graph capture, self-graph or caller recording, is
   ``cuda``-only, and even on ``cuda`` it only applies to ``capturable=True`` generators.
