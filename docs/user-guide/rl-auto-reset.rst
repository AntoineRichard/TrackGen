.. _rl-auto-reset:

RL auto-reset
=============

In a reinforcement-learning loop the :term:`envs <env>` of a batch finish their episodes
at different steps. A :term:`course` gives a new course to exactly the envs that finished,
leaves every other env untouched, and does it inside your step graph if you record one.
Two rules make such a loop correct:

- **Two masks.** ``course.regenerate(mask=..., advance=True)`` gives the masked envs a NEW
  course; ``course.reset(mask)`` respawns them on the SAME course.
- **Gate everything on** ``valid``. An :term:`invalid env` (:term:`valid` ``== 0``) has
  no usable course: end its episode and give it a new one; never reward or observe it.

This page shows both, first as a host loop, then inside a CUDA graph, and what a masked
regeneration costs. It builds on :doc:`/utilities/course` (the course API) and
:doc:`/tutorials/cuda-graph-in-a-sim` (recording into your graph).

.. contents:: On this page
   :local:
   :depth: 1

.. _rl-masked-regeneration:

Two masks: regenerate and reset
-------------------------------

.. code-block:: python

   import numpy as np
   import warp as wp

   from track_gen.course import Course
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"                     # or "cpu"
   course = Course.track(presets.track(num_envs=E, device=device),
                         checkpoint_spacing=0.6, collision="segments", seeds=42)

   regen = course.inputs["regen_mask"]          # stable [E] int32 masks, zero-initialised:
   respawn_mask = course.inputs["reset_mask"]   # the library reads them, never clears them

   before = course.result.center.numpy().reshape(E, -1, 3).copy()   # .copy(): on "cpu",
   epoch = course.result.generation.numpy().copy()                  # .numpy() is a view
   before_valid = course.result.valid.numpy().copy() != 0
   regen.assign(np.isin(np.arange(E), [0, 5]).astype(np.int32))   # envs 0 and 5
   course.regenerate(mask=regen, advance=True)  # new courses for envs 0 and 5 only
   after = course.result.center.numpy().reshape(E, -1, 3)

.. snippet: after

   assert course.result.generation.numpy()[0] == epoch[0] + 1
   if before_valid[0] and course.result.valid.numpy()[0] != 0:
       assert not np.array_equal(before[0], after[0], equal_nan=True)
   assert np.array_equal(before[1], after[1], equal_nan=True)

``course.regenerate(mask=m, advance=True)``:

- **Masked envs** (``m[e] != 0``) advance their :term:`epoch`, get the course of their
  new ``(seed, epoch)`` pair, and have their runtime state reset as ``course.reset(m)``
  resets it (their progress, and the localizer's memory).
- **Unmasked envs** keep their course, everything derived from it (checkpoints, SDF and
  height-field bakes, gate posts, the course line) and their progress and localizer
  state, bitwise.
- **An all-zero mask** makes the call an exact no-op.

``course.reset(m)`` respawns the masked envs on the course they already have: it resets
their progress (and the localizer's memory) and nothing else. ``course.reset()`` resets
every env. Use it for a valid env whose agent crashed, never for an invalid env (below).

Both masks are contiguous ``[E]`` int32 arrays on the course device (checked on the host
before anything launches); nonzero entries select envs. ``course.inputs`` holds a stable, zero-initialised one for each call
(``"regen_mask"`` and ``"reset_mask"``), and any stable array of yours works too. The
library reads a mask and never clears it: write **every** entry on every step, or an
entry left at ``1`` keeps regenerating (or respawning) that env. A mask is recorded by
pointer, so keep writing into the same array.

What decides an env's course:

- ``(seed, epoch)`` (the :term:`seed` and the epoch) fully determines it. Epoch ``0`` uses the seed itself; any other epoch
  uses a hash of the pair. Log both to rebuild an episode's course later:
  :func:`track_gen.seeds.effective_seed` gives the seed that rebuilds it in a fresh course,
  at epoch ``0``.
- ``course.result.generation[e]`` (the :term:`generation`) is the epoch env ``e``'s
  current course was built with.
  As long as you only advance epochs, it changes exactly when the env gets a new course,
  so your kernels can detect new courses on device. It is the epoch, not a counter: a
  reseed at the same epoch leaves it unchanged.
- ``advance=False`` rebuilds the masked envs from their current ``(seed, epoch)``: the
  same course, unless you wrote new seeds for them first (into the seed array you passed
  as ``seeds=``, or ``course.rng.seeds_warp`` with ``seeds=PerEnvSeededRNG(...)``). Use it
  to choose seeds yourself; ``advance=True`` moves to the next epoch instead.
- Seeds you write and epochs you reset (``course.seeds.reset_epoch()``) take effect for an
  env when a masked regeneration selects it, or for every env at the next
  ``course.regenerate()`` without a mask. A masked regeneration never changes an unmasked
  env, whatever you wrote. ``course.generate(seeds=...)`` reseeds every env and zeroes
  every epoch, so it reproduces a fresh course built with those seeds.

A bare generator has the same masked call, without the runtime state:
``gen.generate(advance=True, mask=m)`` (or ``gen.enqueue(...)``) advances the masked envs'
epochs and regenerates them, and ``gen.inputs["regen_mask"]`` is its own stable mask.

.. _course-invalid-envs:

Invalid envs in a loop
----------------------

A generator marks an env invalid (``valid[e] == 0``) when its course failed a check: for a
track, turning, thickness or truncation at :term:`N_max`; for gates, crossing bars
(``fail_mask`` says which). :ref:`invalid-envs` defines it, lists the :term:`fail bits` and the rates at the default
settings, and says which knob lowers each. In a loop, an invalid env is a dead env:

- every utility answers for it as for an env with no course (:ref:`utilities-invalid-envs`
  has every field): ``oob = 1``, NaN ``distance`` / ``nearest`` / ``normal``, no post
  hit, frozen progress with ``dist_to_next = 0``, NaN ``s`` / ``n``;
- ``course.reset(mask)`` respawns it on the same invalid course, so it stays dead;
- ``res.valid`` (``StepResult.valid``, the generator result's ``valid``) is the only
  signal. It is live: a regeneration rewrites it for the regenerated envs, so read it
  before regenerating.

So every reward, observation and done flag is gated on ``valid``, with a **select**
(``np.where``, ``torch.where``, ``wp.where``), never a multiply: ``NaN * 0`` is NaN, and
one NaN env makes a batch mean NaN. An invalid env is marked done (``done |= 1 - valid``)
and gets a NEW course from a masked regeneration; that course can be invalid again, and
the next step then marks it done again. Measure the fraction after construction:

.. code-block:: python

   invalid = (course.result.valid.numpy() == 0).mean()   # 0 to a few % for tracks

In gates mode the bar width decides most of it: see :ref:`gate-width-invalid-fraction`.

.. _rl-valid-recipe:

The recipe on the host
----------------------

The loop below reads the step results back on the host, gates them on ``valid``, and
uses both masks: a valid env that crashed respawns on its course, a finished or invalid
env gets a new one. ``pos``, ``quat`` and ``he`` are your sim's stable buffers (one
oriented box per env), ``sim_step`` your physics and ``respawn`` the code that moves
the masked agents back to their start line.

.. snippet: setup

   @wp.kernel
   def _agent_k(center: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32),
                valid: wp.array(dtype=wp.int32), stride: int, t: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.int32),
                use_mask: int, pos: wp.array(dtype=wp.vec3f)):
       e = wp.tid()
       if use_mask != 0:
           if mask[e] == 0:
               return
           t[e] = 0
       n = count[e]
       if valid[e] != 0 and n > 0:
           pos[e] = center[e * stride + (t[e] * 3) % n]
       if use_mask == 0:
           t[e] = t[e] + 1

   _clock = wp.zeros(E, dtype=wp.int32, device=device)

   def sim_step(pos, quat, stream=None):
       r = course.result
       wp.launch(_agent_k, E, [r.center, r.count, r.valid, r.layout.stride, _clock, regen, 0, pos],
                 device=device, stream=stream)

   def respawn(mask, pos, quat=None, stream=None):
       if not isinstance(mask, wp.array):
           mask = wp.array(np.asarray(mask, np.int32), dtype=wp.int32, device=device)
       r = course.result
       wp.launch(_agent_k, E, [r.center, r.count, r.valid, r.layout.stride, _clock, mask, 1, pos],
                 device=device, stream=stream)

.. code-block:: python

   pos = wp.zeros(E, dtype=wp.vec3f, device=device)          # your sim's stable buffers
   quat = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   he = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)
   course.bind(position=pos, orientation=quat, half_extents=he)
   respawn(np.ones(E, bool), pos)                            # every agent at its start

   for _ in range(100):
       sim_step(pos, quat)                     # your physics: writes the bound buffers
       res = course.step()
       valid = res.valid.numpy() != 0          # read it first: a regeneration rewrites it
       oob = res.contacts.oob.numpy() != 0
       passed = res.events.passed.numpy()
       dist = res.events.dist_to_next.numpy()
       reward = np.where(valid, 10.0 * passed - 5.0 * oob, 0.0)   # a select, never "* valid"
       obs_dist = np.where(valid, dist, 0.0)                       # no NaN reaches the policy
       crashed = oob & valid                                       # same course
       finished = (res.events.laps.numpy() >= 3) | ~valid          # done |= 1 - valid
       if crashed.any():
           respawn_mask.assign(crashed.astype(np.int32))
           course.reset(respawn_mask)          # respawn on the SAME course
       if finished.any():                      # a host check: see the cost below
           regen.assign(finished.astype(np.int32))
           course.regenerate(mask=regen, advance=True)   # NEW courses, reset included
       respawn(crashed | finished, pos)        # your code: back to the start line

.. snippet: after

   assert np.isfinite(reward).all() and np.isfinite(obs_dist).all()

``regenerate(mask=...)`` resets the envs it regenerates, so a finished env needs no
``reset`` of its own. The ``if`` guards read the masks on the host, and they matter: an
eager ``regenerate(mask=...)`` runs the whole generator even when no env is masked
(:ref:`rl-cost`). This loop keeps a crashed episode's course; put ``crashed`` into
``finished`` instead to give it a new one.

With torch, read the results through zero-copy views and write the mask through one:

.. code-block:: python

   import torch

   res = course.step()
   valid = wp.to_torch(res.valid).bool()                  # read it before regenerating
   oob = wp.to_torch(res.contacts.oob).bool()
   reward = torch.where(valid, 10.0 * wp.to_torch(res.events.passed) - 5.0 * oob.float(),
                        0.0)
   done = (wp.to_torch(res.events.laps) >= 3) | ~valid    # done |= 1 - valid
   if done.any():
       wp.to_torch(regen).copy_(done.int())
       course.regenerate(mask=regen, advance=True)

.. _rl-in-graph:

.. _course-rl-loop:

The recipe inside your graph
----------------------------

Recorded into a CUDA graph (a :term:`capture`), the whole RL step, auto-reset included,
is one replay with no host work between steps. Write the reward, the observations and the done mask with one
kernel of your own, gated on ``res.valid``, then record the masked regeneration right
after it:

.. code-block:: python

   @wp.kernel
   def reward_and_done(valid: wp.array(dtype=wp.int32), oob: wp.array(dtype=wp.int32),
                       passed: wp.array(dtype=wp.int32), dist: wp.array(dtype=wp.float32),
                       laps: wp.array(dtype=wp.int32), max_laps: int,
                       reward: wp.array(dtype=wp.float32), obs_dist: wp.array(dtype=wp.float32),
                       done: wp.array(dtype=wp.int32)):
       e = wp.tid()
       ok = valid[e] != 0
       r = 10.0 * float(passed[e]) - 5.0 * float(oob[e])
       reward[e] = wp.where(ok, r, 0.0)        # a select: never r * float(valid[e])
       obs_dist[e] = wp.where(ok, dist[e], 0.0)
       d = 0
       if oob[e] != 0 or laps[e] >= max_laps:
           d = 1
       if not ok:
           d = 1                               # done |= 1 - valid
       done[e] = d                             # EVERY entry, every step

   done = course.inputs["regen_mask"]
   reward = wp.zeros(E, dtype=wp.float32, device=device)
   obs_dist = wp.zeros(E, dtype=wp.float32, device=device)

   with wp.ScopedCapture(device=device) as cap:
       sim_step(pos, quat)                     # your physics: writes the bound buffers
       res = course.step()                     # progress events + contacts + valid
       wp.launch(reward_and_done, E,
                 [res.valid, res.contacts.oob, res.events.passed, res.events.dist_to_next,
                  res.events.laps, 3, reward, obs_dist, done], device=device)
       course.regenerate(mask=done, advance=True)  # new courses where done[e] != 0
       respawn(done, pos, quat)                # your kernel: restart the masked agents
   step = cap.graph

   for _ in range(20):
       wp.capture_launch(step)                 # one RL step, auto-reset included

.. snippet: after

   assert np.isfinite(reward.numpy()).all() and np.isfinite(obs_dist.numpy()).all()

- Build the course and bind your buffers outside the capture (construction raises
  ``CaptureError`` inside one), and keep writing into the same bound buffers.
- Read ``reward`` and every observation BEFORE the regeneration in the graph, as here:
  after it, a masked env's arrays describe its new course.
- The graph needs no ``course.reset(done)``: ``regenerate(mask=...)`` resets the envs it
  regenerates. For respawns on the same course, record ``course.reset(reset_mask)`` with
  a ``reset_mask`` your kernel writes.
- ``wp.ScopedCapture`` records on the ``cpu`` device too (Warp records the launches and
  ``wp.capture_launch`` replays them), so this recipe runs unchanged on a CPU machine.
- **Side streams.** Join every side stream forked inside your capture before
  ``course.regenerate(mask=...)``, or record that call inside
  :func:`track_gen.exec.unconditional`: on CUDA the call puts its body behind a
  :term:`conditional node`, whose recording ends and restarts the stream's capture, and
  CUDA refuses that while a fork is open (``cudaErrorStreamCaptureUnjoined``, which
  invalidates your capture).
- A course built with ``stream=s`` runs on ``s`` by default: pass ``stream=`` (the
  capturing stream) to every call you record, or it raises ``CaptureError`` instead of
  running outside your graph (:ref:`capture-troubleshooting`).

With ``torch.cuda.graph``
~~~~~~~~~~~~~~~~~~~~~~~~~

Inside ``torch.cuda.graph(g)`` library calls adopt torch's capture stream on their own.
Enter :func:`track_gen.interop.torch.registered` with ``conditional=True`` inside the
graph to let the masked regeneration skip itself on device when no env is done (see
:ref:`rl-cost`); it yields the capture stream for your own Warp launches:

.. snippet: group=torch cuda requires=user-guide/rl-auto-reset#main

.. code-block:: python

   import torch
   from track_gen.interop.torch import registered, retain

   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       with registered(conditional=True) as s:  # torch's capture stream, known to Warp
           sim_step(pos, quat, stream=s)       # your own Warp launches go to s
           res = course.step()                 # library calls adopt it on their own
           wp.launch(reward_and_done, E,
                     [res.valid, res.contacts.oob, res.events.passed,
                      res.events.dist_to_next, res.events.laps, 3, reward, obs_dist, done],
                     stream=s)
           course.regenerate(mask=done, advance=True)
           respawn(done, pos, quat, stream=s)  # Warp only: no torch allocation after it
   retain(g, course)                           # the course now lives as long as g

   for _ in range(20):
       g.replay()

- **No torch allocation after the conditional regeneration.** The conditional node ends
  and restarts the stream's capture, which gives it a new capture id, and torch's caching
  allocator routes a graph's allocations by the id it saw when the capture began. A torch
  op that allocates after the regeneration (in the block or after it) fails to capture,
  or takes a block from the general pool that eager code can reuse while the graph still
  writes it. Compute torch observations and rewards before the regeneration, or into
  preallocated tensors (``out=``, in-place ops).
- Without ``conditional=True``, ``registered()`` only registers the capture (so the
  course is kept alive with the graph, :ref:`capture-lifetime`) and the regeneration is
  recorded unconditionally, with no such rule. Without ``registered()`` at all the block
  still records; keep ``course`` alive yourself (``retain(g, course)``).
- ``registered()`` raises ``CaptureError`` outside a torch capture. If you capture with a
  non-default ``torch.cuda.graph(g, capture_error_mode=...)``, pass the same mode as
  ``registered(capture_mode=...)``.
- Importing ``track_gen`` does not import torch; ``track_gen.interop.torch`` imports it
  only when one of its functions is called.

.. _rl-cost:

What a masked regeneration costs
--------------------------------

The generator has fixed shapes, so a masked regeneration still recomputes **every** env:
an unmasked env is recomputed to identical values and restored bit for bit, and the
refresh (checkpoints, bakes, gate posts, course line) covers the whole batch. A step with
at least one masked env therefore costs one full-batch generation, whatever the number
of masked envs.

With NO env masked the call is an exact no-op, and what it costs depends on where it
runs:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - The call runs in (or is recorded into)
     - With an all-zero mask
   * - ``wp.ScopedCapture`` / ``wp.capture_begin`` on CUDA, the course's own CUDA graph
       (what a ``regenerate`` call outside a capture on ``cuda`` replays), or a
       ``torch.cuda.graph`` inside ``track_gen.interop.torch.registered(conditional=True)``,
       when ``wp.is_conditional_graph_supported()`` is true (CUDA driver and toolkit
       >= 12.4)
     - skipped on device: two small kernels reduce the mask to a flag, and a
       :term:`conditional node` skips the rest
   * - a plain ``torch.cuda.graph`` or ``registered()`` without ``conditional=True``, a
       CPU capture, an older CUDA, or anything recorded inside
       :func:`track_gen.exec.unconditional`
     - recorded unconditionally: a full generation per replay
   * - an eager call: any call on ``cpu``, ``enqueue_generate(mask=...)`` outside a
       capture, a generator with a step that runs on the host
     - a full generation per call: guard it with a host check, as the host recipe does

So call ``regenerate(mask=...)`` unconditionally only where it can be skipped; anywhere
else, guard it with a host check or accept a full generation per step. Two limits come
with the conditional node:

- A graph that holds it cannot be used as a CUDA child graph (for example as
  ``wp.capture_if(cond, on_true=step)``). Record the step's calls inside a Python
  ``on_true`` callback instead, or record them inside
  :func:`track_gen.exec.unconditional`.
- Your own ``wp.ScopedCapture`` of torch's current stream records no conditional node
  unless you record inside :func:`track_gen.exec.conditional` (under
  ``torch.cuda.graph`` use ``registered(conditional=True)`` instead).

:doc:`/how-it-works/cuda-graph` has the mechanics (:ref:`cuda-graph-masked-regeneration`).

Respawn or regenerate
---------------------

- **Respawn** (``course.reset(mask)``): the same course, progress reset. For a valid env
  whose agent crashed or timed out, when you want it to retry the same course.
- **Regenerate** (``course.regenerate(mask=..., advance=True)``): a new course, progress
  reset. For an env that finished its course, and always for an invalid env.
- **Regenerate everyone** (``course.regenerate(advance=True)``, or
  ``course.generate(seeds=...)``): a new batch at an episode boundary shared by every
  env. It costs the same full generation as a masked call.

Between episodes, a curriculum changes the generator's
:term:`dynamic fields <dynamic field>` (``course.generator.params[...]``) before the
regeneration that should see them: the new values reach the envs the next regeneration
rebuilds. :ref:`curricula` covers the knobs, and :ref:`curricula-course` a course size
curriculum.

Gates mode
----------

A gates course follows the same recipe: ``res.contacts`` is a ``DiscContact`` (posts,
``hit``) or a ``FrameContact`` (frames), and ``res.frame`` locates the agent on the
gates' centerline. Two things differ:

.. code-block:: python

   gates = Course.gates(presets.gates(num_envs=E, device=device, gate_width=0.1),
                        collision=0.02, seeds=7)          # posts of radius 0.02
   invalid = (gates.result.valid.numpy() == 0).mean()

- **The bar width decides how many envs are valid.** Each env's bars are checked for
  crossings, and wide bars cross often: at ``gate_width = 0.4`` most of a ``bezier`` batch
  is invalid. Keep the width low or the course ``scale`` high, and measure the fraction
  after construction (:ref:`gate-width-invalid-fraction`).
- **A width-0 gate can never be passed**, so ``Course.gates`` refuses a construction width
  of ``0``, and a per-env width you set later must stay ``> 0`` in every env
  (:doc:`/utilities/course`).
