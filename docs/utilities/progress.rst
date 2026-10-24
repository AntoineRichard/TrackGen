Checkpoints & progress
======================

Progress logic is identical for drone-racing gates and car-racing tracks:
discrete pass events plus a distance-to-next-goal signal. The utilities
split it into a shared goal contract and a stateful tracker.

Checkpoints: one contract, two sources
--------------------------------------

``track_gen.checkpoints.CheckpointSet`` is an ordered list of course goals
per env — center ``position``, a physical crossing segment ``left <->
right``, and a forward ``tangent``:

.. code-block:: python

   from track_gen.checkpoints import CheckpointSampler, CheckpointSet

   cps = CheckpointSet.from_gates(gate_seq)          # zero-copy gate view
   cps = CheckpointSampler(track, spacing=0.6).sample()   # virtual gates

``from_gates`` aliases the ``GateSequence`` buffers (regenerated gates are
seen automatically). ``CheckpointSampler`` subsamples the CENTERLINE at a
coarse spacing; because track polylines are index-aligned, each checkpoint's
crossing segment is the road cross-section between ``inner`` and ``outer``.

Both accept any source with the members they read, checked at construction
(:ref:`utilities-sources`): ``from_gates`` a ``HasGateFrames`` source (``position``,
``left``, ``right``, ``forward``, ``half_size``, ``count``; ``valid`` and ``generation``
when present), ``CheckpointSampler`` a ``HasBand`` one (``center``, ``inner``,
``outer``, ``count``, ``valid``; ``generation`` when present); env-slice views
(``seq.view(e0, e1)``, ``track.view(e0, e1)``) included. ``ProgressTracker`` takes any
``HasCheckpoints`` source (``position``, ``left``, ``right``, ``tangent``, ``up_half``,
``count``: a ``CheckpointSet``), and reads its ``valid`` / ``generation`` when present.

A ``CheckpointSet`` has two optional trailing fields since 0.2.0.dev6, ``valid`` and
``generation`` (``[E]`` int32; positional construction is unchanged, ``clone()`` copies
them):

- the sampler's set **owns** them: ``sample()`` writes each env's ``valid`` and
  ``generation`` together with its ring, so ``(count, rows, valid, generation)`` always
  describe the same ring (an env of an invalid track gets ``count = 0``, NaN slots and
  ``valid = 0``);
- ``from_gates`` **aliases** the gates' arrays (the rows are the gates), or leaves them
  ``None`` for a source without them;
- a hand-built set may leave them ``None``: every env is valid and the tracker never
  resets on its own.

**Capacity.** ``CheckpointSampler(track, spacing)`` with ``max_checkpoints=None`` sizes
its ring once, at construction, from the bound batch (``ceil(1.5 * longest valid
centerline / spacing)``, a readback; a supported legacy convenience).
``sampler.max_checkpoints`` reads it. The component route needs no readback:
``tools.Checkpoints(spacing)`` attached through ``ComponentContext.for_producer(gen)``
takes the plan's bound, which a ``Course`` uses by default
(:ref:`course-checkpoint-capacity`).

.. figure:: ../assets/checkpoints-overview.png
   :alt: Track-sourced virtual gates beside gate-sourced checkpoints.

   The same ``CheckpointSet`` contract from a subsampled track (left) and a
   gate sequence (right).

Progress tracking & rewards
---------------------------

``track_gen.progress.ProgressTracker`` consumes any ``CheckpointSet`` and
maintains per-env device state: previous position, next target, laps, total
progress. Each ``update()`` detects forward pass-through of the target's
crossing segment (swept-segment test), wrong-way and wrong-checkpoint
crossings, and reports ``dist_to_next`` — the distance to the next goal:

.. code-block:: python

   from track_gen.progress import ProgressTracker

   tracker = ProgressTracker(cps, position=robot_pos)  # latch onto sim buffer
   prev_d = None
   for _ in range(steps):
       sim.step()                       # writes robot_pos in place
       ev = tracker.update()            # no args: bound mode
       valid = wp.to_torch(cps.valid) != 0     # the set's valid: what the tracker freezes on
       d = wp.to_torch(ev.dist_to_next)
       reward = (prev_d - d) if prev_d is not None else torch.zeros_like(d)  # -delta
       reward = reward + 10.0 * wp.to_torch(ev.passed)        # pass bonus
       reward = torch.where(valid, reward, 0.0)               # a select: NaN-safe
       prev_d = d.clone()
   tracker.reset(done_mask)             # episodic resets, per env
   # `prev_d` above still holds the FINISHED episode's distance for envs
   # reset this step; mask it (e.g. zero the reward) before differencing.

``reset(mask)`` arms a NaN previous-position sentinel, so the first step
after a reset (or a teleport respawn) can never emit a spurious crossing.
That sentinel protects the TRACKER's own crossing/wrong-way detection only —
the caller-side ``-delta distance`` reward term needs its own masking, since
``prev_d`` for a just-reset env is stale (from the episode that just ended)
and differencing it against the freshly reset ``dist_to_next`` is not a
meaningful reward.

**After regenerating** the course (gates or track), reset the regenerated envs:
``reset(mask)``, or let the tracker do it. Since 0.2.0.dev6 it keeps a
``seen_generation`` per env and, at the top of every ``update()``, resets on device every
env whose set's ``generation`` changed, writing exactly what ``reset(mask)`` writes (the
same bytes with or without an explicit reset). On a sampled set that happens at the first
``update()`` after ``sample()``, never between the regeneration and the resample (the set's
``generation`` is written by ``sample()``); on a ``from_gates`` set, at the first
``update()`` after the regeneration. ``generation`` is the epoch, so a regeneration
without ``advance``, a bare reseed, or ``reset_epoch()`` followed by as many advances
does not change it: call ``reset(mask)`` after those (:ref:`utilities-self-reset`).

**Invalid envs** (``valid[e] == 0`` on the set): the tracker freezes the env. It writes
``next_checkpoint`` / ``laps`` / ``progress`` from its state, ``passed = 0``,
``checkpoint_passed = -1``, ``wrong_way = 0``, ``wrong_checkpoint = -1`` and
``dist_to_next = 0``, and re-arms the NaN previous position, so an env that becomes valid
again (an un-advanced regeneration, a reseed at epoch 0, a ``copy_from`` of a valid
snapshot) cannot emit a crossing against a position from before. ``dist_to_next`` is
therefore ``0`` for an invalid env and NaN for a valid env with no checkpoints (a
hand-built set without ``valid``); ``valid`` is the only invalid-env signal
(:ref:`utilities-invalid-envs`).

.. figure:: ../assets/progress-tracking.png
   :alt: Agent path colored by progress with a dist_to_next lower panel.

   A scripted agent threading track checkpoints; the lower panel shows the
   ``dist_to_next`` sawtooth your negative-delta reward differentiates.

API: :class:`track_gen.checkpoints.CheckpointSet`,
:class:`track_gen.checkpoints.CheckpointSampler`,
:class:`track_gen.progress.ProgressTracker` — see the
:doc:`API reference </reference/api>`.
