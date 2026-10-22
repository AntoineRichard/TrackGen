The Course facade
=================

The :doc:`runtime utilities </utilities/overview>` can be wired by hand — or bundled by
``track_gen.course.Course``, which owns the orchestration invariants
(rebake/resample/posts rebuild and a progress reset on every regeneration, full or
limited to the envs a mask selects; per-env respawns via a mask):

.. code-block:: python

   from track_gen import TrackGenConfig
   from track_gen.course import Course, CourseConfig

   course = Course(CourseConfig(
       mode="track",
       gen=TrackGenConfig(num_envs=E, device="cuda"),
       seeds=42,
       collision="segments",          # or "sdf" / None
       checkpoint_spacing=0.6,
   ))                                 # first batch + sub-tools are ready here
   course.bind(position=robot_pos, orientation=robot_quat, half_extents=robot_he)

   for _ in range(steps):
       sim.step()                     # writes the bound buffers in place
       res = course.step()            # events + contacts, no args
       course.reset(done_mask)        # respawn finished envs on the same course
       # or: course.regenerate(mask=done_mask, advance=True)  -> on a NEW course
   course.generate(seeds=next_seed)   # new courses for everyone
   course.regenerate()                # or: rebuild from the rng's current seeds
   course.regenerate(advance=True)    # or: next epoch for everyone, same seeds

Construction does the cold work, outside any capture. It builds the rng and the
generator, whose own construction runs the pipeline once, so ``course.result`` holds the
first batch. It then sizes and builds every sub-tool from that real batch (the read-backs
that size them happen only here) and runs the regenerate body once more as its warm-up.
Expect a construction cost of roughly two generations, which matters for ``repulsive``.
Every sub-tool exists right after construction, so ``step()`` and ``reset()`` only need a
prior ``bind()``. Construct the course before you begin a capture: construction raises
``CaptureError``, before allocating anything, when its stream or Warp's current stream is
capturing.

Sub-tools are sized from that construction batch (``config.seeds``), not from a later
``generate(seeds=...)``. That includes an auto-derived ``max_checkpoints`` (and with it
the shape of every checkpoints array): pass ``max_checkpoints`` explicitly for sizing
that does not depend on the seeds.

The generator pipelines are fixed-batch, so every regeneration recomputes the whole
batch. Per-env control is ``reset(mask)`` (respawn on the same course) and
``regenerate(mask=...)`` (new courses for the masked envs only, see `Masked
regeneration`_).

- ``regenerate(*, mask=None, advance=False, stream=None, sync=True)`` reruns the
  generator from the rng's CURRENT seeds and the envs' epochs, refreshes everything
  downstream, then returns ``course.result``. ``advance=True`` first advances the epoch of
  every env, so the call builds new courses without a reseed. Without ``mask`` the
  progress (and localizer) reset is full; with ``mask`` only the masked envs advance and
  are reset (see `Masked regeneration`_).
- ``generate(seeds=None, *, stream=None, sync=True)`` is the reseeding wrapper. Without
  ``seeds`` it is ``regenerate()``. ``seeds=`` takes an ``int`` (env ``e`` gets
  ``seeds + e``, as when constructing ``PerEnvSeededRNG``) or an ``[E]`` int32
  ``wp.array``. It reseeds the rng and zeroes every env's epoch on the same stream as the
  regeneration, without allocating, so ``generate(seeds=s)`` reproduces the courses of a
  fresh ``Course`` built with ``seeds=s``. The new seeds are written into
  ``course.rng.seeds_warp`` in place: if ``CourseConfig.seeds`` was a ``wp.array``, the
  rng holds that very array, so ``generate(seeds=...)`` rewrites it.
- ``enqueue_generate(*, stream=None, mask=None, advance=False)`` runs the same body as
  ``regenerate()``. Under a capture it is recorded; outside one it runs eagerly on the
  resolved stream, never building a self-graph and never blocking.
- ``inputs`` is ``{"regen_mask": ...}``: a persistent, zero-initialised ``[E]`` int32
  array with a stable address, for your kernels to write the mask of
  ``regenerate(mask=...)`` into.
- ``seeds`` is the generator's :class:`~track_gen.seeds.SeedState` (the same object as
  ``course.generator.seeds``), holding each env's seed (bound to
  ``course.rng.seeds_warp``) and epoch; ``result.generation`` (``[E]`` int32) is the
  epoch each env's current course was built with.

The generators are deterministic under an unchanged RNG: ``regenerate()`` (or
``generate()`` WITHOUT ``seeds=``) reproduces the identical batch (plus a full progress
reset), while ``regenerate(advance=True)`` (next epoch, same seeds) and
``generate(seeds=...)`` (new seeds, epoch zero) give new courses. Each env's course
depends only on its ``(seed, epoch)`` pair. This reproduction is
byte-identical per device for all generators, ``repulsive`` included (its analytic-adjoint
gradient makes it bit-reproducible run-to-run on CUDA as well as CPU); only cross-*device*
output differs, and then only statistically (fp32 rounding). In gates mode the same
object wraps ``GateGenerator`` + gate progress + optional ``DiscChecker``
posts (``post_radius > 0``). The underlying tools stay reachable
(``course.collision``, ``course.progress``, ``course.checkpoints``).
``step()`` and ``reset()`` never host-synchronize, so they record into your
own sim graph as-is (no flag; ``track_gen.set_capturing`` is a deprecated
no-op). Once ``step()`` is captured, keep writing into the SAME bound
buffers — rebinding after capture leaves the captured graph replaying
against the old pointers (silently divergent results), so rebind only
before (re)capturing.

In gates mode with ``post_radius > 0``, ``course.step().contacts`` is a
:class:`~track_gen.collision.DiscContact` instead of a
:class:`~track_gen.collision.BoxContact` — same ``StepResult`` field, a
different contact type depending on the course's collision checker:

.. code-block:: python

   res = course.step()
   hit = wp.to_torch(res.contacts.hit)      # DiscContact here, not BoxContact

In gates mode ``StepResult`` also carries a third field, ``res.frame`` — the
:class:`~track_gen.localize.TrackFrame` localizing the agent on the gate
``CourseLine`` spline (``s`` arc length, ``n`` signed
right offset, ``n_up`` signed vertical offset, ``segment`` index). It is
``None`` in track mode.

One body, one graph
-------------------

``regenerate()`` runs ONE body: the generator pipeline, then the post-generation
refresh (checkpoint resample, SDF and height-field rebakes, course-line refresh, gate-post
rebuild, and the progress and localizer reset: full, or limited to the masked envs with
``mask=``). Where it runs depends on the resolved stream:

- **Inside your capture** (``wp.ScopedCapture``, ``wp.capture_begin(external=True)`` or
  ``torch.cuda.graph``) the body is recorded into YOUR graph, so new courses and the
  per-step ``step()`` / ``reset()`` calls can share one graph. Nothing is synchronized.
- **Outside a capture on** ``cuda`` the body is recorded once into the course's own graph
  (``course.graph``, ``None`` until that first call) and replayed on every later call.
  Each ``(mask array, advance)`` combination has a self-graph of its own, recorded on its
  first use; ``course.graph`` is the one of a plain ``regenerate()``. With the default
  ``sync=True`` the call waits for its stream before returning.
- **On** ``cpu``, or with a ``capturable=False`` generator, it runs eagerly. Under a
  capture, a ``capturable=False`` generator raises ``NotCapturableError`` before launching
  anything.

That single graph replaces the two graphs of earlier releases (the generator's pipeline
graph plus a refresh graph replayed after it).

.. code-block:: python

   with wp.ScopedCapture(device="cuda") as cap:
       course.regenerate()            # new courses for the whole batch
       for _ in range(K):
           sim_step()                 # your physics: writes the bound buffers
           res = course.step()
           course.reset(done_mask)
   episode = cap.graph

   course.rng.set_seeds_warp(new_seeds, None)   # in place: the replay reads it
   wp.capture_launch(episode)

Every replay of a recorded ``regenerate()`` reads the rng's live seeds and the live
epochs. To change the courses, record ``regenerate(advance=True)`` (each replay advances
every env's epoch on device), call ``course.rng.set_seeds_warp(new_seeds, None)`` between
replays, or record ``generate(seeds=seed_buf)`` with a stable ``[E]`` int32 ``seed_buf``
that you rewrite in place. An ``int`` passed to ``generate(seeds=...)`` inside a capture is
baked into the graph: every replay reseeds with that same value and rebuilds the same
courses. See :doc:`/tutorials/cuda-graph-in-a-sim` for the full pattern, including the
``torch.cuda.graph`` and side-stream variants.

Masked regeneration
-------------------

``regenerate(mask=m, advance=True)`` gives new courses to the envs with ``m[e] != 0`` and
leaves the others alone, so an RL step graph can auto-reset finished envs on device:

- **Masked envs** advance their epoch (with ``advance=True``), get the course of their
  new ``(seed, epoch)``, and have their runtime state reset as ``reset(m)`` would reset
  it: progress, and the localizer in gates mode.
- **Unmasked envs** keep their course, the sub-tool data derived from it, and their
  progress and localizer state, bitwise.
- **An all-zero mask** makes the call an exact no-op.

The generator pipeline has fixed shapes, so the regeneration still recomputes EVERY env,
and the refresh (checkpoint resample, SDF and height-field bakes, course line, gate posts)
covers the whole batch. Unmasked envs keep the effective seed their current course was
built from (the seed prologue re-derives it for the masked envs only), so they are
recomputed to identical values, even if their seed or epoch was changed since. Such a
change (``course.rng.set_seeds_warp``, a write to ``course.rng.seeds_warp``,
``course.seeds.reset_epoch()``) takes effect for an env when a masked regeneration
selects it, or for every env at the next ``regenerate()`` without a mask. Any call with a
masked env costs a full-batch generation. A masked commit that avoids the recompute is
planned (roadmap PR10).

``advance=False`` rebuilds the masked envs from their current ``(seed, epoch)``: the same
course, unless your own kernel wrote new seeds for them into ``course.rng.seeds_warp``
first. Use it to choose seeds yourself; ``advance=True`` moves to the next epoch instead.

The mask is a contiguous ``[E]`` int32 array on the course device, checked on the host
before anything launches. ``course.inputs["regen_mask"]`` is a stable, zero-initialised one for
your kernels to write; the library reads it and never clears it, so write every entry on
every step. A mask array is recorded by pointer: keep using the same one (outside a
capture, each new array also records a new self-graph).

.. code-block:: python

   done = course.inputs["regen_mask"]
   with wp.ScopedCapture(device="cuda") as cap:
       sim_step()                     # your physics: writes the bound buffers
       res = course.step()
       update_done(res, done)         # your kernel: done[e] = 1 for finished envs, else 0
       course.regenerate(mask=done, advance=True)   # new courses + resets where done[e]
   step = cap.graph                   # one replay = one RL step, auto-reset included

The body starts with ``track_gen.ops.any_set``, which reduces the mask to a flag on
device, and runs the rest under ``LaunchContext.skip_unless``. When no env is masked the
regeneration is then skipped on device, provided that:

- the call is recorded into a capture Warp registered (``wp.ScopedCapture``,
  ``wp.capture_begin`` with or without ``external=True``, the course's own self-graph, or
  a ``torch.cuda.graph`` inside ``track_gen.interop.torch.registered(conditional=True)``,
  which comes with a rule on later torch allocations), and
- ``wp.is_conditional_graph_supported()`` is true: CUDA driver and toolkit 12.4 or newer.

Otherwise the body runs unconditionally, which is correct (an all-zero mask is an exact
no-op) and costs a full generation per call. A graph that holds the conditional node
cannot be used as a CUDA child graph. See :doc:`/how-it-works/cuda-graph` for the
execution model and :doc:`/tutorials/cuda-graph-in-a-sim` for the full RL pattern,
including the ``torch.cuda.graph`` variant.

Lifetime
--------

Recording ``regenerate()`` / ``generate()`` / ``enqueue_generate()`` into a Warp capture
(``wp.ScopedCapture`` or ``wp.capture_begin``) anchors the course to that graph, so its
buffers live as long as the graph. A graph that records only ``step()`` / ``reset()`` does
not anchor anything, and neither does a ``torch.cuda.graph``: keep ``course`` alive for as
long as you replay such a graph, or call ``track_gen.interop.torch.retain(g, course)`` to
tie it to the torch graph ``g``. When ``max_checkpoints`` is auto-derived, check
``course.checkpoint_sampler.truncated`` after regenerating onto much longer tracks.

API: :class:`track_gen.course.Course`,
:class:`track_gen.course.CourseConfig` — see the
:doc:`API reference </reference/api>`.
