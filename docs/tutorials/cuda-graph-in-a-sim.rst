Integrating the CUDA Graph into a Batched RL Sim Loop
=======================================================

Every shipped generator, ``repulsive`` included, has ``capturable=True``: on ``cuda`` its
whole generation pipeline runs as a CUDA graph. Generation can use either of two graphs:

- **Its own graph (self-graph).** ``generate()`` called outside any capture records the
  pipeline once into the generator's own ``wp.Graph`` and replays it on every call.
- **Your graph.** ``generate()`` (or ``Course.regenerate()``) called while you capture is
  recorded into YOUR graph, next to your physics and the per-step utilities. One
  ``wp.capture_launch`` (or ``g.replay()``) then runs all of it.

A plugin generator registered with ``capturable=False`` (a ``HOST`` stage) runs eagerly on
CUDA every call, ahead of the captured rest of its pipeline, and cannot be recorded. This page explains the fixed-batch contract, the warm-up at
construction, both graph modes, reused output buffers, and how to slot generation into a
batched reinforcement-learning loop, up to a per-env auto-reset that gives new courses to
the envs that finished without leaving your step graph (`Per-Env Auto-Reset Inside the
Graph`_). :doc:`/how-it-works/cuda-graph` has the full execution model.

The Fixed-Batch Contract
--------------------------

``TrackGenerator`` is **fixed-batch**: the batch size ``E = config.num_envs`` is
set at construction time and cannot change between calls. For a capturable generator,
the CUDA graph records one fixed execution with fixed-shape tensors. Passing a different
batch size at replay time is not possible — construct a new ``TrackGenerator`` instead.

All pre-allocated buffers (generator scratch, pipeline scratch, seed buffer,
the persistent ``Track``) are created once in ``__init__``. Their shapes and
device addresses are stable for the lifetime of the ``TrackGenerator`` object.
The generator keeps a private copy of its config: changing the config object afterwards
has no effect.

Construction, Self-Graph, and Replay
--------------------------------------

Construction runs the pipeline **once eagerly**, outside any capture. That pass loads every
Warp kernel the hot path uses, so no later recording compiles anything, and it leaves the
batch for the rng's current seeds in the result: the ``Track`` is valid right after
construction (construction blocks until that pass has finished). Construct generators
before you begin a capture: construction raises ``CaptureError``, before allocating
anything, when its stream or Warp's current stream is capturing.

On the ``cpu`` device every ``generate()`` call then runs the pipeline eagerly — there is
no graph.

On ``cuda``, outside any capture:

1. **First call** — records the pipeline once into the generator's own graph
   (``gen.graph``) on a private stream, then launches it.
2. **Every later call** — launches that graph with ``wp.capture_launch`` on the call's
   stream.

The seed prologue, the kernel that opens the generation body, is part of the graph. It
derives each env's seed from ``rng.seeds_warp`` and the env's epoch (zero until you
advance it, see `Per-Env Auto-Reset Inside the Graph`_), so every replay generates from
the rng's CURRENT seeds and the CURRENT epochs. By default ``generate()`` then waits for
its stream to finish (a wait on that one stream, not a device-wide
``wp.synchronize()``). ``generate(sync=False)`` returns at once, and the results are
ordered on the stream like every other library call.

The graph belongs to the ``TrackGenerator`` instance: a new instance records its own on its
own first call.

Capture works because a capturable generator's pipeline is **entirely static**:

- All buffer shapes and loop counts are fixed at construction time, and nothing allocates
  per call.
- There is no host-side branching on device tensor data (branching on config
  fields is resolved before capture).
- Per-track ``count[e]`` values are device-side data; count-aware kernels keep
  static launch dimensions via ``N_max``.
- The pipeline never host-synchronizes or reads device data back (both are
  illegal inside a CUDA graph scope); ordering comes from the stream.

``repulsive`` is the outlier on cost, not on capture: its growth loop runs a
fixed, host-precomputed iteration schedule (~1000× slower than the other generators).
Converged envs are frozen on device, so each env still depends only on its own seed,
and the loop is a ``ctx.bounded_loop``: its self-graph skips, on device, the stall
windows left once every env has frozen (CUDA 12.4+ conditional nodes), while a plain
``torch.cuda.graph`` records, and an eager ``cuda`` call runs, the full iteration bound.
Its construction pays one full eager run, and its first ``generate()`` on ``cuda``
records its graph. See :doc:`/generators/repulsive`.

.. code-block:: python

   import numpy as np
   import warp as wp
   wp.init()

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E = 512
   # Thin tracks need N_max >= 512: the default 384 covers only ~98% of them, and
   # truncated tracks fail validity (see Track.truncated and the tuning guide).
   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device="cuda")
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device="cuda")

   gen = TrackGenerator(config, rng)       # runs the pipeline once: gen's Track is valid
   assert gen.graph is None

   # First call: records the self-graph, then launches it.
   track = gen.generate()
   assert gen.graph is not None

   # Later calls on the SAME generator replay the graph, which reads the rng's current
   # seeds: reseed in place to get a new batch.
   rng.set_seeds_warp(wp.array(42 + np.arange(E), dtype=wp.int32, device="cuda"), None)
   track = gen.generate()                  # replay; returns once the stream is done
   track = gen.generate(sync=False)        # replay; returns at once (stream-ordered)

   # Or keep the seeds and advance every env's epoch on device: a new batch, no reseed.
   # (advance=True has a self-graph of its own, recorded on its first call.)
   track = gen.generate(advance=True)

Buffer Reuse and ``clone()``
-----------------------------

``generate()`` always returns the **same** ``Track`` instance with the **same**
underlying Warp array pointers. The arrays are overwritten in-place on every
call. If you read the arrays after a second ``generate()`` call without cloning,
you will see the second batch's data, not the first.

For a sim loop that needs to keep the previous episode's tracks while generating
a new batch, use ``track.clone()`` before the next call:

.. code-block:: python

   track = gen.generate()
   saved = track.clone()   # independent copy; safe to store across generate()

   track = gen.generate()  # overwrites gen's internal buffers
   # saved still holds the first batch's data

The same pattern applies to ``GateGenerator`` and ``GateSequence.clone()``.

A Self-Graph Episode Loop
---------------------------

The simplest loop leaves generation outside your graph: the generator replays its own
graph at each episode boundary, and the full batch is regenerated together. For envs that
finish at different steps, see `Per-Env Auto-Reset Inside the Graph`_.

.. code-block:: python

   import warp as wp
   import numpy as np
   wp.init()
   import torch

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E, device = 256, "cuda"
   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device=device)
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gen    = TrackGenerator(config, rng)

   # First reset: the first call records the self-graph.
   track = gen.generate()

   center = wp.to_torch(track.center).view(E, config.N_max, 3)
   outer  = wp.to_torch(track.outer).view(E, config.N_max, 3)
   inner  = wp.to_torch(track.inner).view(E, config.N_max, 3)
   valid  = wp.to_torch(track.valid).bool()
   count  = wp.to_torch(track.count)

   # --- Sim step loop ---
   for episode in range(1000):

       # Work with the current tracks.  All views above stay valid until the
       # next generate() call.  Clone if you need them to survive past that.
       obs = center[:, :int(count.max()), :]   # example: max-length slice

       # ... run RL steps ...

       # On episode boundary: generate a fresh batch (fast CUDA graph replay).
       new_seeds = wp.array(episode + 1 + np.arange(E), dtype=wp.int32, device=device)
       rng.set_seeds_warp(new_seeds, None)
       track = gen.generate()

.. note::

   Keep one ``TrackGenerator`` and its existing ``PerEnvSeededRNG`` alive across
   episodes. Build a per-environment ``wp.int32`` seed array, call
   ``rng.set_seeds_warp(new_seeds, None)``, then call ``gen.generate()``. The
   replayed graph derives the seeds from the RNG's current seeds on device, into the
   pre-allocated seed buffer. ``gen.generate(advance=True)`` is the reseed-free
   alternative: it advances every env's epoch on device instead. Because the persistent
   ``Track`` buffers retain stable addresses, the Torch views above remain bound and do
   not need to be recreated.

Recording Generation into Your Own Graph
------------------------------------------

Call ``generate()`` while you capture, and it is recorded into your graph: the seed
prologue and every pipeline launch become nodes of YOUR graph. Nothing is self-captured,
nothing is launched with ``wp.capture_launch`` (a graph launch cannot be recorded into
another capture), and nothing blocks the host, whatever ``sync`` says. Recording allocates
nothing, so it also works with Warp's mempool disabled.

.. code-block:: python

   gen = TrackGenerator(config, rng)           # build OUTSIDE the capture

   with wp.ScopedCapture(device="cuda") as cap:
       physics_step()                          # your kernels
       track = gen.generate()                  # recorded, not executed
   graph = cap.graph

   for episode in range(num_episodes):
       # seeds_for(ep): an [E] int32 wp.array of new seeds (see the Course example)
       rng.set_seeds_warp(seeds_for(episode), None)   # in place; the graph reads it
       wp.capture_launch(graph)

The rules:

- **The stream decides.** A call is recorded when the stream it resolves is capturing. A
  plain ``wp.ScopedCapture(device)`` captures Warp's current stream, which is what a call
  without ``stream=`` resolves to, and inside ``torch.cuda.graph(g)`` torch's capture
  stream is adopted automatically. When you capture a side stream, pass it
  (``gen.generate(stream=s)``) or wrap the calls in ``LaunchContext(device, s).scope()``.
  A call that resolves to a stream that is NOT capturing runs there and is silently
  missing from your graph.
- **Seeds and epochs are read live.** Every replay generates from the rng's current seeds
  and the generator's current epochs (``gen.seeds.epoch``). Reseed between replays with
  ``rng.set_seeds_warp(new_seeds, None)``, or record that call into the graph against a
  stable seed array that you rewrite in place. Or record ``gen.generate(advance=True)``:
  each replay then advances every env's epoch on device and builds a new batch, with no
  host work between replays.
- **Host values are frozen.** Python scalars and Python-side state are baked in at record
  time; anything that must change between replays has to be device data.
- **Keep the generator alive.** A Warp capture (``wp.ScopedCapture``, ``wp.capture_begin``)
  keeps the generator's buffers alive with the graph; a ``torch.cuda.graph`` does not, so
  hold on to the generator while you replay it, or tie it to the torch graph with
  ``track_gen.interop.torch.retain(g, gen)``.
- **Non-capturable generators raise.** A ``capturable=False`` plugin (any pipeline
  with a ``HOST`` stage) raises ``NotCapturableError`` (from ``track_gen.exec``), naming
  the ``HOST`` stages, before launching anything. Build the pipeline with
  ``PipelineSpec(..., require_caller_capturable=True)`` to have that refused at
  construction instead. A legacy
  plugin without ``accepts_ctx=True`` can only be recorded on Warp's current stream and
  raises ``LegacyPluginStreamError`` otherwise; see :doc:`/how-it-works/cuda-graph`.

``gen.enqueue(stream=None)`` records exactly like ``generate()`` under a capture. Outside
one it runs the pipeline eagerly on the resolved stream: it never builds a self-graph and
never blocks.

The Recommended Sim-Loop Pattern
----------------------------------

With a :doc:`Course </utilities/course>`, ``course.regenerate()`` is ONE body: whole-batch
generation plus the coherent refresh (checkpoint resample, SDF and height-field bakes, gate
posts, full progress reset). It records into your graph exactly like ``generate()``, so a
single caller graph can hold new courses AND the sim steps that drive them:

1. Build everything outside the capture: the ``Course`` (construction runs the pipeline
   eagerly, sizes the sub-tools from that real batch and warms the regenerate body), your
   stable sim buffers, and the ``bind()``.
2. Record one graph with ``course.regenerate()`` followed by the per-step
   ``course.step()`` / ``course.reset(mask)`` calls around your physics.
3. Replay it, reseeding the rng in place between replays to get new courses.

.. code-block:: python

   import numpy as np
   import warp as wp
   wp.init()

   from track_gen import TrackGenConfig
   from track_gen.course import Course, CourseConfig

   E, K, device = 256, 64, "cuda:0"            # K control steps per recorded episode

   def seeds_for(ep):                          # [E] int32 seeds for episode ep
       return wp.array(ep * E + np.arange(E), dtype=wp.int32, device=device)

   # 1. Build everything OUTSIDE the capture.
   course = Course(CourseConfig(
       mode="track",
       gen=TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device=device),
       seeds=0,
       collision="segments",
       checkpoint_spacing=0.6,
   ))
   # Your sim's stable buffers, written in place every step.
   pos  = wp.zeros(E, dtype=wp.vec3f, device=device)
   quat = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   he   = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)
   done = wp.zeros(E, dtype=wp.int32, device=device)   # your per-env respawn mask
   course.bind(position=pos, orientation=quat, half_extents=he)

   # 2. Record one episode: new courses for the whole batch, then K sim steps.
   with wp.ScopedCapture(device=device) as cap:
       course.regenerate()                     # generation + refresh, recorded
       for _ in range(K):
           sim_step(pos, quat)                 # your physics: writes the bound buffers
           res = course.step()                 # progress events + contacts
           update_done(res, done)              # your kernel: done[e] = 1 respawns env e
           course.reset(done)                  # respawn on the SAME course
   episode = cap.graph

   # 3. Replay. The graph reads the rng's live seeds: reseed in place first.
   for ep in range(1, num_episodes + 1):
       course.rng.set_seeds_warp(seeds_for(ep), None)
       wp.capture_launch(episode)
       # res.events / res.contacts now hold the last recorded step's results

Notes on the pattern:

- ``res`` is the course's stable ``StepResult``: every replay rewrites its arrays in place.
  Keep writing into the SAME bound buffers; rebinding after the capture leaves the graph
  reading the old pointers.
- Gates mode (``mode="gates"``) records the same way.
- If episodes are long, or you regenerate less often than you step, record
  ``course.regenerate()`` into a graph of its own and launch it only at episode
  boundaries. Both graphs work on the same course buffers.
- To reseed inside the graph, record ``course.generate(seeds=seed_buf)`` in place of
  ``course.regenerate()``, with ``seed_buf`` a stable ``[E]`` int32 array that you rewrite
  in place between replays (a kernel of yours, or a ``wp.to_torch`` view). Do not pass an
  ``int`` under a capture: it is baked into the graph, so every replay reseeds with the
  same value and rebuilds the same courses.
- ``course.regenerate()`` called outside a capture replays the course's own graph
  (``course.graph``) instead; ``course.enqueue_generate()`` records or runs eagerly and
  never builds that graph.
- To vary the courses without touching the seeds, record
  ``course.regenerate(advance=True)``: every replay advances each env's epoch on device.
  To give new courses only to the envs that finished, inside the step graph, see
  `Per-Env Auto-Reset Inside the Graph`_.

``torch.cuda.graph`` variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside ``torch.cuda.graph(g)``, every library call that names no stream adopts torch's
capture stream, so the same body records into ``g`` without any ``stream=`` argument:

.. code-block:: python

   import torch

   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       course.regenerate()                     # adopted: recorded into g
       for _ in range(K):
           sim_step()                          # your physics, on torch's capture stream
           res = course.step()
           update_done(res, done)
           course.reset(done)

   for ep in range(1, num_episodes + 1):
       course.rng.set_seeds_warp(seeds_for(ep), None, stream="torch")  # replay's stream
       g.replay()

Your own Warp launches inside ``torch.cuda.graph`` must target the capture stream as well
(``stream=wp.stream_from_torch()``). Warp does not see torch graphs, so nothing ties the
course's buffers to ``g``: keep ``course`` alive for as long as you replay ``g``, or call
``track_gen.interop.torch.retain(g, course)`` once after the capture.

Side-stream variant
~~~~~~~~~~~~~~~~~~~~

To capture a stream that is not Warp's current one, every library call must land on it.
Enter ``LaunchContext(device, s).scope()`` once, or pass ``stream=s`` to each call:

.. code-block:: python

   from track_gen.exec import LaunchContext

   s = wp.Stream(device)
   with wp.ScopedCapture(device=device, stream=s) as cap:
       with LaunchContext(device, s).scope():  # every library call below resolves to s
           course.regenerate()
           for _ in range(K):
               sim_step(pos, quat, stream=s)   # your launches still need stream=s
               res = course.step()
               update_done(res, done, stream=s)
               course.reset(done)
   episode = cap.graph
   # Explicit form, without the scope:
   #   course.regenerate(stream=s); course.step(stream=s); course.reset(done, stream=s)

   for ep in range(1, num_episodes + 1):
       with wp.ScopedStream(s):   # the seed upload, the reseed and the temporary's free ride s
           course.rng.set_seeds_warp(seeds_for(ep), None)
       wp.capture_launch(episode, stream=s)

``seeds_for(ep)`` builds its array with ``wp.array(...)``, which uploads on Warp's CURRENT
stream and frees the temporary there afterwards. Passing only ``stream=s`` to
``set_seeds_warp`` would read it on ``s`` with nothing ordering the two streams, so make
``s`` current for the whole statement (or fill a preallocated seed buffer on ``s``).

Without the scope or ``stream=s``, a call resolves to Warp's current stream, which is not
capturing: it runs there, outside your graph.

Per-Env Auto-Reset Inside the Graph
-------------------------------------

In an RL loop, envs finish at different steps.
``course.regenerate(mask=done, advance=True)`` gives a new course to exactly the envs
with ``done[e] != 0``, and it records into your step graph next to ``course.step()``.
One replay of that graph is then one full RL step, auto-reset included, with no host
involvement between steps:

- **Masked envs** (``mask[e] != 0``) advance their epoch, so they get a new course, and
  their per-env runtime state is reset as ``course.reset(mask)`` would reset it (progress,
  and the localizer in gates mode).
- **Unmasked envs** (``mask[e] == 0``) are left unchanged: their course, the sub-tool data
  derived from it and their progress state come out bitwise the same.
- **No env masked:** the regeneration is skipped on device where CUDA supports it (see
  `Skipping the regeneration on device`_), and runs as an exact no-op everywhere else.

Build the course and bind your stable buffers as in `The Recommended Sim-Loop Pattern`_,
then record one step:

.. code-block:: python

   done = course.inputs["regen_mask"]          # stable [E] int32, zero-initialised

   with wp.ScopedCapture(device=device) as cap:
       sim_step(pos, quat)                     # your physics: writes the bound buffers
       res = course.step()                     # progress events + contacts
       update_done(res, done)                  # your kernel: writes done[e] for EVERY env
       course.regenerate(mask=done, advance=True)  # new courses where done[e] != 0
       respawn(done, pos, quat)                # your kernel: restart masked envs on them
   step = cap.graph

   for _ in range(num_steps):
       wp.capture_launch(step)                 # no host work between steps

``regenerate(mask=...)`` resets the masked envs itself, so this graph needs no
``course.reset(done)``. The rules for the mask:

- It is a contiguous ``[E]`` int32 array on the course device (checked on the host
  before anything launches); nonzero entries select envs. ``course.inputs["regen_mask"]`` is a ready-made,
  zero-initialised one with a stable address. Any stable array of yours works too.
- The library reads it and never clears it. Your kernel must write every entry on every
  step: an entry left at ``1`` regenerates that env on every replay.
- It is recorded by pointer, so keep writing into the same array. Outside a capture on
  ``cuda``, the course keeps one self-graph per ``(mask array, advance)`` pair, recorded on
  its first use: pass the same array on every call. ``course.graph`` is the self-graph of
  a plain ``regenerate()``.

What happens to each env, and what it costs:

- The generator pipeline has fixed shapes, so a masked regeneration still recomputes
  EVERY env. An unmasked env keeps the effective seed its current course was built from
  and is recomputed to identical values, and so is everything derived from it (checkpoint
  resample, SDF and height-field bakes, course line, gate posts); the generator's
  commit also restores an unmasked env's ``Track`` rows bit for bit after the recompute.
  Only the masked envs' runtime state is reset. A step with at least one masked env
  therefore costs one full-batch generation.
- Seeds you write (``course.rng.seeds_warp``, ``course.rng.set_seeds_warp(...)``) and
  epochs you reset (``course.seeds.reset_epoch(...)``) between regenerations take effect
  for an env when a masked regeneration selects it (its new course and its state reset
  come together), or for every env at the next full ``course.regenerate()``. A masked
  regeneration never changes an unmasked env, whatever you wrote.
- With ``advance=False``, the masked envs are rebuilt from their current
  ``(seed, epoch)``: the same course, unless your own kernel wrote new seeds for them
  first. That is how you choose the seeds yourself (a curriculum, for example);
  ``advance=True`` moves to the next epoch instead.
- ``course.result.generation[e]`` is the epoch env ``e``'s current course was built with.
  As long as you only advance epochs (no reseed, no epoch reset), it changes exactly when
  env ``e`` gets a new course, so your observation and logging kernels can detect new
  courses on device. It is the epoch, not a counter: a reseed at the same epoch leaves it
  unchanged, and after an epoch reset earlier values come back.
- ``(seed, epoch)`` fully determines an env's course. Epoch ``0`` uses the seed itself (the
  batches of earlier releases); any other epoch uses a hash of the pair. Log both to
  rebuild an episode's course later: ``track_gen.seeds.effective_seed(seed, epoch)``
  returns the seed that rebuilds it in a fresh course (at epoch ``0``).
- ``course.generate(seeds=...)`` zeroes every epoch, so it reproduces a fresh course built
  with those seeds. ``course.rng.set_seeds_warp(...)`` alone keeps the epochs; follow it
  with ``course.seeds.reset_epoch()`` and a FULL ``course.regenerate()`` (no mask) for the
  same effect. A masked regeneration in between applies them only to the envs it selects.
  ``generate(seeds=...)`` writes the seeds into ``course.rng.seeds_warp``, which is your
  own array if ``CourseConfig.seeds`` was a ``wp.array``.

On a bare generator, ``gen.generate(advance=True, mask=m)`` and
``gen.enqueue(advance=True, mask=m)`` advance only the masked envs' epochs, then
regenerate; ``gen.inputs["regen_mask"]`` is the generator's own stable mask. A generator
has no runtime state to reset.

Skipping the regeneration on device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``regenerate(mask=...)`` first reduces the mask to a one-element flag on device
(``any_set``), then puts the rest of its body behind
``LaunchContext.skip_unless(flag, body)``. That becomes a CUDA conditional graph node
(``wp.capture_if``), so a step where no env is done costs two small kernels instead of a
generation, when both of these hold:

- the call is recorded into a capture that Warp registered on that stream:
  ``wp.ScopedCapture``, ``wp.capture_begin`` (``external=True`` included), the course's own
  self-graph, or a ``torch.cuda.graph`` inside
  ``track_gen.interop.torch.registered(conditional=True)`` (below);
- ``wp.is_conditional_graph_supported()`` is true: the CUDA driver and the CUDA toolkit
  Warp was built with are both 12.4 or newer.

Everywhere else (eager calls, the ``cpu`` device, a plain ``torch.cuda.graph`` or one
registered without ``conditional=True``, an older driver or toolkit) the body runs
unconditionally. That is still correct: with no env masked it recomputes identical values
and resets nothing. It only costs a full generation on every replay. The flag is never
read back on the host.

A step graph that holds the conditional node cannot be used as a CUDA child graph (for
example as ``wp.capture_if(cond, on_true=step)``). To nest it, record the step's calls
inside a Python ``on_true`` callback instead, or keep the conditional out of the step
graph by recording it without a Warp registration.

Warp does not see a plain ``torch.cuda.graph`` capture, so the skip falls back to the
unconditional body there. ``track_gen.interop.torch.registered(conditional=True)``,
entered INSIDE ``torch.cuda.graph(g)``, registers torch's capture stream with Warp for the
duration of the block, allows the conditional node in it, and yields the stream as a
``wp.Stream``:

.. code-block:: python

   import torch
   from track_gen.interop.torch import registered, retain

   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       with registered(conditional=True) as s:  # torch's capture stream, known to Warp
           sim_step(pos, quat, stream=s)       # your own Warp launches go to s
           res = course.step()                 # library calls adopt it on their own
           update_done(res, done, stream=s)
           course.regenerate(mask=done, advance=True)
           respawn(done, pos, quat, stream=s)  # Warp only: no torch allocation after it
   retain(g, course)                           # course now lives as long as g

   for _ in range(num_steps):
       g.replay()

``conditional=True`` comes with one rule: after the conditional regeneration, the rest of
the ``torch.cuda.graph`` (inside the block and after it) must not run torch ops that
allocate. ``wp.capture_if`` ends and restarts the stream's capture around the node, which
gives it a new CUDA capture id, and torch's caching allocator only routes allocations to
the graph's private pool for the id it saw when the capture began. A torch op recorded
later would fail to capture or take a block from the general pool that eager code can
reuse while the graph still writes it. So compute torch observations and rewards BEFORE
the regeneration, or into preallocated tensors (``out=``, in-place ops). Without
``conditional=True``, ``registered()`` only registers the capture (for the lifetime
anchoring below) and the regeneration runs unconditionally, with no such rule.

Without ``registered()`` the same block still records (library calls adopt torch's capture
stream), with the regeneration running unconditionally. Notes:

- ``retain(g, *owners)`` stores strong references on the torch graph object, so the owners
  live as long as ``g``; Warp cannot do that for a torch graph. ``registered(graph=g)``
  does the same for every producer whose generation is recorded inside the block.
- ``registered()`` raises ``CaptureError`` outside a torch capture. If you capture with a
  non-default ``torch.cuda.graph(g, capture_error_mode=...)``, pass the same mode as
  ``registered(capture_mode=...)``.
- Importing ``track_gen`` does not import torch; ``track_gen.interop.torch`` imports it
  only when one of its functions is called.

A Mixed Batch in One Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~

Envs from different generators can share one ``Track`` and one step graph: each
``TrackGenerator`` writes an env-slice view of a preallocated result (``out=``), and the
utilities read the whole result. A ``Course`` builds its own generator, so wire the
utilities by hand:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.collision import CollisionChecker
   from track_gen.pipeline import presets
   from track_gen.results import Layout, Track

   big = Track.alloc(Layout(E, 384, device))           # cold: before any capture
   half = E // 2
   seeds = seeds_for(0)                                # [E] int32: one array, sliced per generator
   gens = [TrackGenerator(presets.track(num_envs=half, device=device, generator=g),
                          seeds=seeds[k * half:(k + 1) * half],
                          out=big.view(k * half, (k + 1) * half))
           for k, g in enumerate(("bezier", "polar"))]
   collision = CollisionChecker(big, max_boxes=1)      # reads all E envs

   s = wp.Stream(device)
   with wp.ScopedCapture(stream=s) as cap:
       sim_step(pos, quat, stream=s)
       contacts = collision.query(pos, quat, he, stream=s)  # he: [E] vec2f, above
       update_done(contacts, done, stream=s)           # done: your [E] int32 mask
       for k, gen in enumerate(gens):
           gen.generate(stream=s, advance=True, mask=done[k * half:(k + 1) * half])
   step = cap.graph

Each generator regenerates the masked envs of its own range and leaves every other row
untouched; views given to two live generators must be disjoint (the build raises
otherwise). Every generator of the batch uses the same ``n_max``. Pass each one its slice
of one seed array, as above (with ``seeds=None`` or an ``int``, the seeds are offset by the
view's first env, so the batch is still seeded like one batch). See
:ref:`results-mixed-batches` for the rules.

Recording the Per-Step Utilities
----------------------------------

The runtime utilities (collision, progress, localizer, props, checkpoints,
height field) and ``Course.step()``/``reset()`` never host-synchronize, so
their per-step calls record into your own sim graph directly. No flag is
needed: ``track_gen.set_capturing`` is a deprecated no-op.

.. code-block:: python

   tracker = ProgressTracker(checkpoints, position=sim_pos)  # bind once
   tracker.update()                                         # warm up outside capture
   with wp.ScopedCapture(device="cuda") as cap:
       physics_step()
       tracker.update()
   # every sim step:
   wp.capture_launch(cap.graph)

The same calls outside a capture return without blocking the host. Results are
ordered by the Warp stream they were launched on: ``.numpy()``, later Warp
launches and torch on its default stream all see finished data. A consumer on a
different stream (for example a torch side stream) must order itself explicitly,
e.g. with ``wp.synchronize_device()`` or a recorded ``wp.Event``. Benchmarks must
synchronize before reading a timer. For debugging asynchronous CUDA errors, set
``TRACKGEN_DEBUG_SYNC=1`` to make every library call block on its stream. It is
skipped while that stream (or a Warp capture on the device) is capturing; do not
combine it with a foreign capture on another stream (e.g. a default
``torch.cuda.graph``), which it cannot detect and whose capture the sync would
invalidate.

Performance Notes
------------------

At large batch sizes the pipeline is compute-bound (XPBD relaxation dominates).
The CUDA graph's benefit is a single, GPU-resident, deployable unit — not a
speedup over eager execution. Representative replay times at ``E=8192``:

.. list-table::
   :header-rows: 1

   * - Generator
     - Replay time (CUDA)
   * - ``"bezier"``
     - 93.2 ms
   * - ``"checkpoint"``
     - 92.2 ms
   * - ``"hull"``
     - 98.9 ms
   * - ``"polar"``
     - 80.8 ms
   * - ``"voronoi"``
     - 81.3 ms

For high-throughput scenarios where relaxation is the bottleneck, the advanced
XPBD separation cache (``relax_sep_every``, ``relax_sep_cache_slots``,
``relax_sep_cache_skin``) can reduce per-replay time significantly. The
recommended starting point from the benchmark is:

.. code-block:: python

   config = TrackGenConfig(
       num_envs=E,
       half_width=0.03,
       N_max=512,
       relax_iters=150,      # the long budget is what makes relax_sep_every=40 safe
       relax_sep_every=40,
       relax_sep_cache_slots=16,
       relax_sep_cache_skin=0.0,
       device="cuda",
   )

On the benchmark machine this setting achieved roughly ``0.066 s`` per replay
versus ``0.366 s`` for the dense baseline at ``E=8192`` (``relax_iters=150``). Keep
``relax_sep_every`` well below ``relax_iters``; see :doc:`/configuration/tuning`.
Validate yield in your target regime before relying on the cache.

CPU Fallback
-------------

On the ``cpu`` device the pipeline runs eagerly on every ``generate()`` call —
there is no CUDA graph capture. The same ``TrackGenerator`` API works on both
devices; only the device string in ``TrackGenConfig`` needs to change.

.. code-block:: python

   config_cpu = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device="cpu")
   rng_cpu    = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
   gen_cpu    = TrackGenerator(config_cpu, rng_cpu)
   track_cpu  = gen_cpu.generate()   # eager, no graph
