The Course facade
=================

Everything above can be wired by hand — or bundled by
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
   ))
   course.bind(position=robot_pos, yaw=robot_yaw, half_extents=robot_he)

   track = course.generate()          # whole batch + coherent refresh
   for _ in range(steps):
       sim.step()                     # writes the bound buffers in place
       res = course.step()            # events + contacts, no args
       course.reset(done_mask)        # respawn finished envs on the same course
   course.generate(seeds=next_seed)   # new courses for everyone

``generate()`` is whole-batch (the generator pipelines are fixed-batch
captured graphs); per-env control is ``reset(mask)``. The generators are
deterministic under an unchanged RNG: calling ``generate()`` again WITHOUT
``seeds=`` reproduces the identical batch (plus a full progress reset), while
passing ``seeds=`` advances the RNG for new courses. In gates mode the same
object wraps ``GateGenerator`` + gate progress + optional ``DiscChecker``
posts (``post_radius > 0``). The underlying tools stay reachable
(``course.collision``, ``course.progress``, ``course.checkpoints``) and
``track_gen.set_capturing(True)`` flips the ONE shared capture flag used by
collision, props, checkpoints, progress, and the facade itself, all at
once, when you capture ``step()`` into your own sim graph. Once ``step()``
is captured, keep writing into the SAME bound buffers — rebinding after
capture leaves the captured graph replaying against the old pointers
(silently divergent results), so rebind only before (re)capturing.

In gates mode with ``post_radius > 0``, ``course.step().contacts`` is a
:class:`~track_gen.collision.DiscContact` instead of a
:class:`~track_gen.collision.BoxContact` — same ``StepResult`` field, a
different contact type depending on the course's collision checker:

.. code-block:: python

   res = course.step()
   hit = wp.to_torch(res.contacts.hit)      # DiscContact here, not BoxContact

Under the hood, two CUDA graphs do the heavy lifting on ``cuda`` devices:
the generator's own pipeline graph (captured on the first ``generate()``)
and a facade-owned refresh graph covering the post-generation work — SDF
rebake, checkpoint resample, gate-post rebuild, and the full progress
reset. Without a ``seeds=`` argument, ``generate()`` reproduces the
identical batch (the generators are deterministic under an unchanged RNG);
pass seeds to vary the courses. When ``max_checkpoints`` is auto-derived,
check ``course.checkpoint_sampler.truncated`` after regenerating onto
much longer tracks.

API: :class:`track_gen.course.Course`,
:class:`track_gen.course.CourseConfig` — see the
:doc:`API reference </reference/api>`.
