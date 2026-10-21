The Course facade
=================

The :doc:`runtime utilities </utilities/overview>` can be wired by hand — or bundled by
``track_gen.course.Course``, which owns the orchestration invariants
(rebake/resample/posts rebuild and a full progress reset on every
regeneration; per-env respawns via a mask):

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
   course.generate(seeds=next_seed)   # new courses for everyone
   course.regenerate()                # or: rebuild from the rng's current seeds

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

``regenerate()`` and ``generate()`` are whole-batch (the generator pipelines are
fixed-batch CUDA graphs); per-env control is ``reset(mask)``.

- ``regenerate(*, stream=None, sync=True)`` reruns the generator from the rng's CURRENT
  seeds and refreshes everything downstream, then returns ``course.result``.
- ``generate(seeds=None, *, stream=None, sync=True)`` is the reseeding wrapper. Without
  ``seeds`` it is ``regenerate()``. ``seeds=`` takes an ``int`` (env ``e`` gets
  ``seeds + e``, as when constructing ``PerEnvSeededRNG``) or an ``[E]`` int32
  ``wp.array``. It reseeds the rng on the same stream as the regeneration, without
  allocating.
- ``enqueue_generate(*, stream=None)`` runs the same body as ``regenerate()``. Under a
  capture it is recorded; outside one it runs eagerly on the resolved stream, never
  building the course's own graph and never blocking.

The generators are deterministic under an unchanged RNG: ``regenerate()`` (or
``generate()`` WITHOUT ``seeds=``) reproduces the identical batch (plus a full progress
reset), while passing ``seeds=`` advances the RNG for new courses. This reproduction is
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
rebuild, and the full progress and localizer reset). Where it runs depends on the
resolved stream:

- **Inside your capture** (``wp.ScopedCapture``, ``wp.capture_begin(external=True)`` or
  ``torch.cuda.graph``) the body is recorded into YOUR graph, so new courses and the
  per-step ``step()`` / ``reset()`` calls can share one graph. Nothing is synchronized.
- **Outside a capture on** ``cuda`` the body is recorded once into the course's own graph
  (``course.graph``, ``None`` until that first call) and replayed on every later call.
  With the default ``sync=True`` the call waits for its stream before returning.
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

Every replay of a recorded ``regenerate()`` reads the rng's live seeds. To change them,
call ``course.rng.set_seeds_warp(new_seeds, None)`` between replays, or record
``generate(seeds=seed_buf)`` with a stable ``[E]`` int32 ``seed_buf`` that you rewrite in
place. An ``int`` passed to ``generate(seeds=...)`` inside a capture is baked into the
graph: every replay reseeds with that same value and rebuilds the same courses. See
:doc:`/tutorials/cuda-graph-in-a-sim` for the full pattern, including the
``torch.cuda.graph`` and side-stream variants.

Lifetime: recording ``regenerate()`` / ``generate()`` / ``enqueue_generate()`` into a Warp
capture (``wp.ScopedCapture`` or ``wp.capture_begin``) anchors the course to that graph, so
its buffers live as long as the graph. A graph that records only ``step()`` / ``reset()``
does not anchor anything, and neither does a ``torch.cuda.graph``: keep ``course`` alive
for as long as you replay such a graph. When ``max_checkpoints`` is auto-derived, check
``course.checkpoint_sampler.truncated`` after regenerating onto much longer tracks.

API: :class:`track_gen.course.Course`,
:class:`track_gen.course.CourseConfig` — see the
:doc:`API reference </reference/api>`.
