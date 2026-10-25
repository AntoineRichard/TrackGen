.. _cuda-graph-sim-loop:

CUDA graphs in a sim loop
=========================

Every built-in generator, ``repulsive`` included, can be recorded into a CUDA graph.
Generation runs from one of two graphs:

- **The generator's own CUDA graph.** On ``cuda``, ``generate()`` called outside any
  :term:`capture` records the generation once into the generator's own ``wp.Graph``
  (``gen.graph``) and replays it on every later call.
- **Your graph.** ``generate()`` (or ``Course.regenerate()``) called while you capture is
  recorded into YOUR graph, next to your physics and the per-step utilities. One
  ``wp.capture_launch`` (or ``g.replay()``) then runs all of it.

A plugin generator with a step that runs on the host (one registered with
``capturable=False``) runs that step eagerly on every call and cannot be recorded into
your graph. This page covers the fixed batch, construction and the generator's own graph,
recording generation and a :term:`course` into your graph (with ``torch.cuda.graph`` and
on a side stream), a mixed batch, the per-step utilities, the rules for recording, and
what to do when a recording fails. Per-env auto-reset inside the graph has its own page,
:doc:`/user-guide/rl-auto-reset`; :doc:`/how-it-works/cuda-graph` has the execution model.

.. contents:: On this page
   :local:
   :depth: 1

A fixed batch
-------------

A generator is **fixed-batch**: its number of :term:`envs <env>`, :term:`E`, is set at
construction and cannot change between calls. A CUDA graph records one fixed execution
over fixed-shape arrays, so a different batch size needs a new generator.

Every buffer (scratch, seeds, epochs, the result ``Track``) is allocated once, at
construction; shapes and device addresses stay stable for the generator's lifetime. The
generator keeps a private copy of its config: changing the config object afterwards has
no effect. Change a :term:`dynamic field` through ``gen.params[...]`` instead
(:ref:`curricula`), or build a new generator for a static one.

Construction and the generator's own graph
------------------------------------------

Construction runs the generation **once, eagerly**, outside any capture. That pass loads
every Warp kernel the hot path uses, so no later recording compiles anything, and it
leaves the batch for the current :term:`seeds <seed>` in the result: the ``Track`` is
ready right after construction (construction waits until that pass has finished).
Construct generators before you begin a capture: construction raises ``CaptureError``,
before allocating anything, when its stream or Warp's current stream is capturing.

After construction:

- **On** ``cpu`` every ``generate()`` runs the generation eagerly; there is no CUDA graph
  (a CPU capture still records it into yours, below).
- **On** ``cuda``, outside any capture, the first call records the generation into the
  generator's own graph (``gen.graph``) on a private stream, then launches it; every later
  call launches that graph on the call's stream.

The first step of every generation derives each env's seed from its base seed and its
:term:`epoch`, on device, so every replay generates from the CURRENT seeds and epochs
(``track.generation`` records the epoch, the :term:`generation`, of each env's track).
By default ``generate()`` then waits for its stream to finish (a wait on that one stream,
not a device-wide ``wp.synchronize()``); ``generate(sync=False)`` returns at once, the
results being ordered on the stream like every other library call.

.. code-block:: python

   import numpy as np
   import warp as wp

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"                     # or "cpu"
   # Thin tracks need n_max >= 512: the default 384 truncates some of them, and a
   # truncated track is invalid.
   spec = presets.track(num_envs=E, device=device, half_width=0.03, n_max=512)
   seeds = wp.array(np.arange(E, dtype=np.int32), device=device)   # bound by pointer
   gen = TrackGenerator(spec, seeds=seeds)      # runs one generation: the batch is ready

   track = gen.generate()                       # cuda: records gen.graph, then launches it
   seeds.assign(np.arange(E, dtype=np.int32) + 1000)   # new seeds, written in place
   track = gen.generate()                       # replays the graph: reads the new seeds
   track = gen.generate(sync=False)             # returns at once (stream-ordered)
   track = gen.generate(advance=True)           # next epoch for every env: a new batch,
                                                # no reseed (a graph of its own)

A ``PerEnvSeededRNG`` works the same way: pass it as ``seeds=`` (or as the second
argument) and reseed with ``rng.set_seeds_warp(new_seeds, None)``.

The generation can be recorded because it is **entirely static**:

- all buffer shapes and loop counts are fixed at construction, and nothing allocates per
  call;
- there is no host-side branching on device data (choices that depend on the config are
  made at construction);
- each track's ``count[e]`` is device data, and kernels that depend on it keep static
  launch dimensions through :term:`N_max`;
- nothing host-synchronizes or reads device data back; ordering comes from the stream.

``repulsive`` stands out by its cost, not by its recording: its growth loop runs a fixed
iteration bound, two orders of magnitude more work than the other generators (its own
cost table is in :doc:`/generators/repulsive`). Each env stops changing once it
converges, independently of the rest of the batch, so each env still depends only on its
own seed. Its own graph on ``cuda`` skips, on device, the iterations left once every env
has converged (CUDA 12.4+ :term:`conditional nodes <conditional node>`), while a plain
``torch.cuda.graph`` records the full bound and an eager call on ``cuda`` runs it (an
eager ``cpu`` call stops at the first check that finds every env converged). Its
construction pays one full run; see :doc:`/generators/repulsive`.

Buffer reuse and ``clone()``
----------------------------

``generate()`` always returns the **same** ``Track`` instance, with the **same** Warp
arrays, overwritten in place on every call. Arrays read after a second call hold the
second batch. To keep a batch while generating the next one, ``clone()`` it first:

.. code-block:: python

   track = gen.generate()
   saved = track.clone()                        # an independent copy
   track = gen.generate(advance=True)           # overwrites gen's buffers
   # saved still holds the first batch

.. snippet: after

   assert not np.array_equal(saved.center.numpy(), track.center.numpy(), equal_nan=True)

The same holds for ``GateGenerator`` and ``GateSequence.clone()``. ``clone()`` allocates,
so call it outside your capture (inside one, it raises ``CaptureError``; copy into a
preallocated result with ``copy_from`` instead, :ref:`results`). On ``cpu``,
``.numpy()`` is a view of the live array, not a copy: copy it to keep its values.

An episode loop on the generator's own graph
--------------------------------------------

The simplest loop leaves generation outside your graph: at each episode boundary the
generator replays its own graph, and the whole batch is regenerated together.

.. code-block:: python

   for episode in range(3):
       track = gen.generate(advance=True)       # a new batch per episode
       valid = track.valid.numpy() != 0         # gate everything on valid
       count = track.count.numpy()
       center = track.center.numpy().reshape(E, -1, 3)   # NaN past count[e]
       # ... run the episode on the valid envs ...

Gate everything on :term:`valid`. Read results with ``.numpy()`` (a copy on ``cuda``, a
view on ``cpu``), or keep zero-copy torch views
(``wp.to_torch(track.center).view(E, gen.n_max, 3)``): the buffers keep their addresses,
so views made once stay bound across calls. An env with ``valid[e] == 0`` is an
:term:`invalid env`: see :ref:`invalid-envs`. For envs that finish at different steps,
see :doc:`/user-guide/rl-auto-reset`.

Recording generation into your graph
------------------------------------

Call ``generate()`` while you capture, and it is recorded into your graph: the seed step
and every launch of the generation become nodes of YOUR graph. Nothing is launched from
the generator's own graph (a graph launch cannot be recorded into another capture), and
nothing blocks the host, whatever ``sync`` says. Recording allocates nothing, so it also
works with Warp's mempool disabled.

.. snippet: setup

   @wp.kernel
   def _physics_k(pos: wp.array(dtype=wp.vec3f)):
       e = wp.tid()
       pos[e] = pos[e] + wp.vec3f(0.001, 0.0, 0.0)

   _body = wp.zeros(E, dtype=wp.vec3f, device=device)

   def physics_step():
       wp.launch(_physics_k, E, [_body], device=device)

.. code-block:: python

   with wp.ScopedCapture(device=device) as cap:     # gen was built OUTSIDE the capture
       physics_step()                              # your kernels
       track = gen.generate()                      # recorded, not executed
   graph = cap.graph

   for episode in range(3):
       seeds.assign(np.arange(E, dtype=np.int32) + 2000 + episode * E)  # the graph reads it
       wp.capture_launch(graph)

``wp.ScopedCapture`` records on ``cpu`` too (Warp records the launches and
``wp.capture_launch`` replays them), so the same code runs on a CPU machine.
``gen.enqueue(stream=None)`` records exactly like ``generate()`` under a capture; outside
one it runs the generation eagerly on the resolved stream, never builds the generator's
own graph and never blocks. Every replay generates from the seeds and epochs at replay
time; :ref:`capture-rules` lists everything that decides what a recording does.

The recommended pattern with a course
-------------------------------------

With a :doc:`course </utilities/course>`, ``course.regenerate()`` is ONE body: the
generation plus the refresh of every utility (checkpoint resample, SDF and height-field
bakes, gate posts, progress reset). It records into your graph exactly like
``generate()``, so one graph can hold new courses AND the sim steps that drive them:

1. Build everything outside the capture: the course (construction runs the generation and
   a warm-up), your stable sim buffers, and the ``bind()``.
2. Record one graph with ``course.regenerate(advance=True)`` followed by the per-step
   ``course.step()`` / ``course.reset(mask)`` calls around your physics.
3. Replay it: each replay advances every env's epoch on device and builds new courses.

.. snippet: setup

   @wp.kernel
   def _sim_k(center: wp.array(dtype=wp.vec3f), stride: int, t: wp.array(dtype=wp.int32),
              pos: wp.array(dtype=wp.vec3f)):
       e = wp.tid()
       pos[e] = center[e * stride + t[0]]

   @wp.kernel
   def _tick_k(t: wp.array(dtype=wp.int32)):
       t[0] = (t[0] + 1) % 200

   @wp.kernel
   def _crash_k(oob: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
                crashed: wp.array(dtype=wp.int32)):
       e = wp.tid()
       crashed[e] = wp.where(oob[e] != 0 and valid[e] != 0, 1, 0)

   _t = wp.zeros(1, dtype=wp.int32, device=device)

   def sim_step(pos, quat, stream=None):
       r = course.result
       wp.launch(_sim_k, E, [r.center, r.layout.stride, _t, pos], device=device,
                 stream=stream)
       wp.launch(_tick_k, 1, [_t], device=device, stream=stream)

   def update_crashed(res, crashed, stream=None):
       wp.launch(_crash_k, E, [res.contacts.oob, res.valid, crashed], device=device,
                 stream=stream)

.. code-block:: python

   from track_gen.course import Course

   K = 64                                       # sim steps per recorded episode

   # 1. Build everything OUTSIDE the capture.
   course = Course.track(spec, checkpoint_spacing=0.6, collision="segments", seeds=0)
   pos = wp.zeros(E, dtype=wp.vec3f, device=device)          # your sim's stable buffers
   quat = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   he = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)
   course.bind(position=pos, orientation=quat, half_extents=he)
   crashed = course.inputs["reset_mask"]        # stable [E] int32: 1 respawns env e

   # 2. Record one episode: new courses for the whole batch, then K sim steps.
   with wp.ScopedCapture(device=device) as cap:
       course.regenerate(advance=True)          # generation + refresh, recorded
       for _ in range(K):
           sim_step(pos, quat)                  # your physics: writes the bound buffers
           res = course.step()                  # progress events + contacts + valid
           update_crashed(res, crashed)         # your kernel: a valid env that left the road
           course.reset(crashed)                # respawn on the SAME course
   episode = cap.graph

   # 3. Replay: every replay builds new courses and runs K steps.
   for _ in range(3):
       wp.capture_launch(episode)
       # res.events / res.contacts now hold the last recorded step's results

.. snippet: after

   assert int(course.result.generation.numpy()[0]) == 3

Notes on the pattern:

- ``res`` is the course's stable ``StepResult``: every replay rewrites its arrays in place.
  Keep writing into the SAME bound buffers; rebinding after the capture leaves the graph
  reading the old arrays (``course.bind`` then emits ``RebindAfterCaptureWarning``:
  record the graph again).
- An invalid env (``res.valid[e] == 0``) stays invalid until the next regeneration of this
  pattern: ``course.reset(crashed)`` would respawn it on the same course, where it reads
  ``oob = 1`` and frozen progress at every step. Gate its reward and observations on
  ``valid`` with a select, or give it a new course in the step graph with a masked
  regeneration (:ref:`rl-in-graph`).
- Gates courses (``Course.gates``) record the same way.
- If episodes are long, or you regenerate less often than you step, record
  ``course.regenerate(advance=True)`` into a graph of its own and launch it only at
  episode boundaries. Both graphs work on the same course buffers.
- To choose the seeds, record ``course.generate(seeds=seed_buf)`` in place of
  ``course.regenerate(advance=True)``, with ``seed_buf`` a stable ``[E]`` int32 array
  that you rewrite in place between replays. Do not pass an ``int`` under a capture: it is
  baked into the graph, so every replay reseeds with the same value and rebuilds the same
  courses.
- ``course.regenerate()`` called outside a capture replays the course's own CUDA graph
  (``course.graph``) on ``cuda``; ``course.enqueue_generate()`` records under a capture,
  runs eagerly outside one, and never builds that graph.

``torch.cuda.graph`` variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside ``torch.cuda.graph(g)``, every library call that names no stream adopts torch's
capture stream, so the same body records into ``g`` without any ``stream=`` argument.
Your own Warp launches must target that stream too: get it from
:func:`track_gen.interop.torch.registered`, which yields the library's long-lived Warp
wrapper of torch's capture stream:

.. snippet: group=torch cuda requires=tutorials/cuda-graph-in-a-sim#main

.. code-block:: python

   import torch
   from track_gen.interop.torch import registered, retain

   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       with registered() as s:                  # torch's capture stream, known to Warp
           course.regenerate(advance=True)      # adopted: recorded into g
           for _ in range(K):
               sim_step(pos, quat, stream=s)    # your physics, on torch's capture stream
               res = course.step()
               update_crashed(res, crashed, stream=s)
               course.reset(crashed)
   retain(g, course)                            # the course now lives as long as g

   for _ in range(3):
       g.replay()

.. _torch-stream-wrapper:

Do not wrap the capture stream yourself with a fresh ``wp.stream_from_torch()`` for each
capture: when such a wrapper is garbage-collected, Warp unregisters the stream handle for
the whole process, which breaks a later capture on the same handle (torch reuses its
capture stream across graphs). If you need your own wrapper, keep it alive for the life
of the process. Warp does not see torch graphs, so nothing else ties the course's buffers
to ``g``: keep ``course`` alive for as long as you replay ``g``, or call
``track_gen.interop.torch.retain(g, course)`` once after the capture, as here (or pass
``registered(graph=g)``). With ``conditional=True``, ``registered`` also lets a masked
regeneration skip itself on device (:ref:`rl-cost`).

Side-stream variant
~~~~~~~~~~~~~~~~~~~

To capture a stream that is not Warp's current one, every library call must land on it.
Enter ``LaunchContext(device, s).scope()`` once, or pass ``stream=s`` to each call:

.. snippet: group=side-stream cuda requires=tutorials/cuda-graph-in-a-sim#main

.. code-block:: python

   from track_gen.exec import LaunchContext

   s = wp.Stream(device)
   with wp.ScopedCapture(device=device, stream=s) as cap:
       with LaunchContext(device, s).scope():   # every library call below resolves to s
           course.regenerate(advance=True)
           for _ in range(K):
               sim_step(pos, quat, stream=s)    # your launches still need stream=s
               res = course.step()
               update_crashed(res, crashed, stream=s)
               course.reset(crashed)
   episode = cap.graph
   # Explicit form, without the scope:
   #   course.regenerate(advance=True, stream=s); course.step(stream=s); ...

   for _ in range(3):
       wp.capture_launch(episode, stream=s)

Without the scope or ``stream=s``, a call resolves to Warp's current stream, which is not
capturing: it runs there, outside your graph. Arrays you upload for a replay (new seeds,
for example) are written on Warp's current stream: make ``s`` current around the upload
(``with wp.ScopedStream(s):``) or write a preallocated buffer on ``s``, so the replay on
``s`` is ordered after it.

A mixed batch in one graph
--------------------------

Envs from different generators can share one ``Track`` and one step graph: each
``TrackGenerator`` writes an env-slice view of a preallocated result (``out=``), and the
utilities read the whole result:

.. code-block:: python

   from track_gen.collision import CollisionChecker
   from track_gen.results import Layout, Track

   big = Track.alloc(Layout(E, 384, device))        # cold: before any capture
   half = E // 2
   mixed_seeds = wp.array(np.arange(E, dtype=np.int32), device=device)
   gens = [TrackGenerator(presets.track(num_envs=half, device=device, generator=name),
                          seeds=mixed_seeds[k * half:(k + 1) * half],
                          out=big.view(k * half, (k + 1) * half))
           for k, name in enumerate(("bezier", "polar"))]
   collision = CollisionChecker(big, max_boxes=1)   # reads all E envs
   done = wp.zeros(E, dtype=wp.int32, device=device)  # your per-env mask

   with wp.ScopedCapture(device=device) as cap:
       contacts = collision.query(pos, quat, he)
       for k, g in enumerate(gens):              # new courses for the masked envs
           g.generate(advance=True, mask=done[k * half:(k + 1) * half])
   step = cap.graph
   wp.capture_launch(step)

Each generator regenerates the masked envs of its own range and leaves every other row
untouched; views given to two live generators must be disjoint (the build raises
otherwise). Every generator of the batch uses the same ``n_max``. Pass each one its slice
of one seed array, as above (with ``seeds=None`` or an ``int``, the seeds are offset by the
view's first env, so the batch is still seeded like one batch). A ``Course`` drives one
generator, so a mixed batch wires its utilities by hand. See :ref:`results-mixed-batches`
for the rules.

Per-env auto-reset in the graph
-------------------------------

``course.regenerate(mask=done, advance=True)`` gives new courses to exactly the envs your
kernel marked done, and records into your step graph next to ``course.step()``: one replay
is then one full RL step, auto-reset included, with no host work between steps.
:doc:`/user-guide/rl-auto-reset` has the recipe (:ref:`rl-in-graph`), gated on ``valid``,
and what it costs (:ref:`rl-cost`).

Recording the per-step utilities
--------------------------------

The runtime utilities (collision, progress, localizer, props, checkpoints, height field),
their :term:`components <component>` and ``Course.step()`` / ``reset()`` never
host-synchronize, so their per-step calls record into your own sim graph as they are,
with no flag to set:

.. code-block:: python

   from track_gen.checkpoints import CheckpointSampler
   from track_gen.progress import ProgressTracker

   checkpoints = CheckpointSampler(track, spacing=0.6).sample()
   tracker = ProgressTracker(checkpoints, position=pos)   # bind once, outside the capture
   tracker.update()                             # a first call outside the capture
   with wp.ScopedCapture(device=device) as cap:
       physics_step()
       events = tracker.update()
   for _ in range(3):                           # every sim step
       wp.capture_launch(cap.graph)

The same calls outside a capture return without blocking the host. Results are ordered by
the Warp stream they were launched on: ``.numpy()``, later Warp launches and torch on its
default stream all see finished data. A consumer on a different stream (for example a
torch side stream) must order itself explicitly, e.g. with ``wp.synchronize_device()`` or
a recorded ``wp.Event``. Benchmarks must synchronize before reading a timer. To debug
asynchronous CUDA errors, set ``TRACKGEN_DEBUG_SYNC=1`` to make every library call block
on its stream; it is skipped while that stream (or a Warp capture on the device) is
capturing. Do not combine it with a foreign capture on another stream (e.g. a default
``torch.cuda.graph``), which it cannot detect and whose capture the wait would invalidate.

.. _capture-rules:

Rules for recording into your graph
-----------------------------------

These rules hold for every generator, course and utility (:ref:`upgrading-capture` lists
what changed from 0.1).

- **Build outside, record inside.** Generators, courses and utilities are constructed
  outside any capture (construction allocates and runs a real pass; inside a capture it
  raises ``CaptureError``). Bind your buffers outside too.
- **The stream decides.** A call is recorded when the stream it resolves is capturing. A
  plain ``wp.ScopedCapture(device)`` captures Warp's current stream, which is what a call
  without ``stream=`` resolves to, and inside ``torch.cuda.graph(g)`` torch's capture
  stream is adopted automatically. When you capture a side stream, pass it
  (``gen.generate(stream=s)``) or wrap the calls in ``LaunchContext(device, s).scope()``.
  A call that resolves to a stream that is NOT capturing runs there, and is silently
  missing from your graph. A generator or course built with ``stream=s`` runs on ``s`` by
  default: pass the capturing stream when you record it.
- **Seeds and epochs are read live.** Every replay generates from the current seeds and
  epochs (``gen.seeds``). Reseed between replays by writing the seed array in place (the
  array you passed as ``seeds=``, or ``rng.set_seeds_warp(new_seeds, None)``), or record
  ``generate(advance=True)``: each replay then advances every env's epoch on device and
  builds a new batch, with no host work between replays.
- **Host values are frozen.** Python scalars and Python-side state are baked in at record
  time (``course.generate(seeds=42)`` included); anything that must change between replays has
  to be device data, written in place.
- **Keep everything alive.** The graph reads the buffers of the generator, the course and
  the arrays you bound; :ref:`capture-lifetime` says who keeps them alive.
- **Only recordable generators.** A generator with a step that runs on the host (a plugin
  registered with ``capturable=False``) raises ``NotCapturableError`` before launching
  anything; build its pipeline with a :term:`PipelineSpec` with
  ``require_caller_capturable=True``
  to have that refused at construction instead.

.. _capture-requirements:

What makes a call recordable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library holds these for every built-in generator, course and utility:

- **No host synchronization.** No call host-synchronizes or reads device data back:
  syncs are illegal inside a capture, and the stream (or the graph) already orders the
  launches. ``TRACKGEN_DEBUG_SYNC=1`` adds a wait for debugging, skipped while the stream
  is capturing. A step that runs on the host is the exception: it runs outside every
  capture (:ref:`cuda-graph-host`).
- **Iterative steps have a fixed bound.** A loop with a data-dependent stop runs a fixed
  number of iterations on device, each env stopping once it converges, never a host-side
  ``while`` (:ref:`cuda-graph-bounded`).
- **No allocation.** Nothing allocates on a ``generate()`` / ``enqueue()`` path, with or
  without ``advance`` and ``mask``, nor in ``SeedState.advance()`` / ``set()`` /
  ``reset_epoch()``: every buffer is allocated at construction. Recording works with
  Warp's mempool disabled.
- **Stable addresses.** The seed array, the epochs, the result's ``generation``,
  ``inputs["regen_mask"]`` and a generator's dynamic-parameter and input arrays keep
  their addresses; a replay reads their current contents on device. A ``mask=`` array is
  recorded by pointer: rewrite it in place.
- **Device-side counts.** Each track's ``count[e]`` is device data, and the kernels that
  depend on it keep static launch dimensions through ``N_max``.
- **Choices made at construction.** Which generator and which stages run are decided at
  construction, and the generator copies its config then, so a recorded graph cannot
  drift.

.. _capture-lifetime:

Lifetime
~~~~~~~~

A ``wp.Graph`` keeps its kernel modules alive, not the buffers its nodes read and write.
Who keeps those buffers alive depends on the capture:

- **A capture Warp registered** (``wp.ScopedCapture``, ``wp.capture_begin`` including
  ``external=True``, or ``track_gen.interop.torch.registered()`` inside a
  ``torch.cuda.graph``). A recorded ``generate()`` / ``enqueue()`` attaches the
  generator to the resulting ``wp.Graph``, so its buffers live as long as the graph.
  Recording any course call (``step()``, ``reset()``, ``regenerate()``, ``generate()``,
  ``enqueue_generate()``) attaches the course, and with it the generator, every
  component and every bound array, plus the arrays the recorded call reads that the
  course does not own: the binding ``step()`` read (after a rebind the course drops its
  references to the old arrays; the graph keeps them) and a ``mask`` / ``seeds`` array
  you passed. Drop every reference, or rebind, and the graph still finds every buffer it
  reads.
- **A CPU graph** (``wp.ScopedCapture("cpu")``) keeps the arrays its launches touch by
  itself: the course object is released, the arrays live as long as the graph.
- **A plain** ``torch.cuda.graph`` is not seen by Warp. Keep the generator or course
  (and the masks and seed arrays you passed) alive for as long as you replay the graph,
  or tie them to it: ``track_gen.interop.torch.retain(g, course, mask, ...)`` stores
  strong references on the torch graph object ``g``. Inside
  ``track_gen.interop.torch.registered(graph=g)`` the capture is registered with Warp, so
  recorded calls attach as above, and the registration's Warp graph object is retained on
  ``g`` when the block exits.

.. _capture-threads:

Threads
~~~~~~~

Built-in generators take no process-wide lock: each records its own graph on a private
stream, so independent generators on different threads run concurrently. Each generator
(and each course) is single-writer, though: do not run the same instance from two
threads, or on two streams, at once; use one instance per concurrent generator.

Two caveats apply to multi-threaded programs:

- While a generator records its own graph (once, on its first ``generate()`` on
  ``cuda``), Warp refuses ``wp.synchronize()``, ``wp.synchronize_device()`` and
  ``wp.synchronize_stream()`` on that device from every thread. ``generate()``'s own wait
  is not affected. If other threads call those functions, run each generator's first
  ``generate()`` before starting them.
- The private stream that recording uses is a blocking stream. Work that another thread
  issues on the legacy default stream during that one-time recording can invalidate it,
  so the library retries the recording once before raising.

.. _capture-troubleshooting:

Troubleshooting capture
-----------------------

:ref:`api-errors` lists the exception and warning classes. Every capture error derives
from ``track_gen.exec.CaptureError``, a ``RuntimeError``.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Symptom
     - Cause
     - Fix
   * - ``CaptureError``: "... must be constructed (prepared) outside any ... capture"
     - a generator, ``Program`` or course was constructed inside a capture (a CPU capture
       included)
     - construct it before beginning the capture
   * - ``CaptureError``: "... was built with stream= ... and is called while another
       stream is capturing"
     - a course built with ``stream=s`` was called without ``stream=`` while another
       stream records
     - pass ``stream=`` (the capturing stream, or ``"torch"``) to every recorded call
   * - ``CaptureError``: "... was built with prepare=False and is being recorded"
     - the course (or ``Program``) never ran its warm-up
     - call ``course.prepare()`` outside the capture first
   * - ``CaptureError`` naming the components, at ``regenerate(mask=...)``
     - a component's ``on_generate`` or ``reset`` uses an operation a conditional body
       cannot hold (a copy to or from pageable host memory, an event, another stream)
     - find it with ``check_component`` (:ref:`kit-check-component`); record the call
       inside ``track_gen.exec.unconditional()``
   * - ``CaptureError`` from ``Track.clone()`` / ``GateSequence.clone()``
     - ``clone()`` allocates
     - clone outside the capture, or ``copy_from`` into a result preallocated with
       ``alloc()``
   * - ``CaptureError`` from ``track_gen.interop.torch.registered()``
     - it was entered outside a ``torch.cuda.graph`` capture
     - enter it inside ``with torch.cuda.graph(g):``
   * - ``NotCapturableError``
     - the generator has a step that runs on the host (a plugin registered with
       ``capturable=False``); nothing was launched
     - call it outside the capture (it runs eagerly), or use a recordable generator;
       ``PipelineSpec(require_caller_capturable=True)`` refuses such a pipeline at
       construction
   * - ``RebindAfterCaptureWarning``
     - ``bind`` replaced an array that a recorded ``step()`` / ``query()`` /
       ``update()`` reads: the graph still reads the old one
     - record the graph again after rebinding; filter the warning when you re-record on
       purpose
   * - A replay does not update some result
     - that call resolved a stream that was not capturing, so it ran once, eagerly,
       outside your graph
     - pass ``stream=s``, or enter ``LaunchContext(device, s).scope()``, when you capture
       a side stream
   * - Every replay builds the same courses
     - ``generate(seeds=<int>)`` was recorded: the ``int`` is baked into the graph; or
       the seeds were never rewritten
     - record ``regenerate(advance=True)``, or ``generate(seeds=seed_buf)`` with a stable
       array you rewrite in place
   * - A crash or garbage after deleting the course, replaying a ``torch.cuda.graph``
     - Warp does not see torch graphs, so nothing kept the buffers alive
     - ``track_gen.interop.torch.retain(g, course)``, or record inside
       ``registered(graph=g)`` (:ref:`capture-lifetime`)
   * - A masked regeneration costs a full generation on every replay of a torch graph
     - a torch capture records the masked body unconditionally
     - record it inside ``track_gen.interop.torch.registered(conditional=True)``; for
       your own ``wp.ScopedCapture`` of torch's current stream, inside
       ``track_gen.exec.conditional()`` (:ref:`rl-cost`)
   * - ``cudaErrorStreamCaptureUnjoined`` at ``regenerate(mask=...)``; the capture is
       invalidated
     - a side stream forked in your capture is not joined yet, and the conditional node
       restarts the capture
     - join it first, or record the call inside ``track_gen.exec.unconditional()``
   * - Adding the step graph as a CUDA child graph fails
     - it holds a conditional node (a masked regeneration, ``repulsive``'s loop)
     - record it inside ``track_gen.exec.unconditional()``
   * - A torch op fails to capture after a conditional regeneration
     - torch allocated after the node under ``registered(conditional=True)``
     - compute torch values before the regeneration, or into preallocated tensors

Performance notes
-----------------

At large batch sizes the generation is compute-bound (the relaxation dominates). A CUDA
graph's benefit is a single, GPU-resident, deployable unit, not a speedup over eager
execution. Replay time per generation (``gen_ms_per_call``, milliseconds) and yield at
``E = 8192``:

.. csv-table:: Generation cost on CUDA, ``E = 8192``
   :file: ../_data/cost-cuda-e8192.csv
   :header-rows: 1

The machine and the date of these numbers:

.. include:: ../_data/machine-cuda.txt

:doc:`/contributing/rendering-assets` gives the command that regenerates the table.

When the relaxation is the bottleneck, the separation cache (``relax_sep_every``,
``relax_sep_cache_slots``, ``relax_sep_cache_skin``) can cut the replay time
substantially; the separation-cache recipe in :doc:`/configuration/tuning` gives a starting
configuration and its measured effect. Validate the yield in your own regime before
relying on it.

CPU fallback
------------

On the ``cpu`` device every ``generate()`` runs the generation eagerly; there is no CUDA
graph, and ``gen.graph`` stays ``None``. The same API works on both devices: only the
``device`` changes. A ``wp.ScopedCapture(device="cpu")`` records library calls and
``wp.capture_launch`` replays them, so every recording on this page also runs on a CPU
machine (a masked regeneration then always runs in full, :ref:`rl-cost`).
