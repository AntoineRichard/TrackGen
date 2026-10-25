Generating Gate Sequences for Drone Courses
============================================

For drone-style courses where road width is irrelevant, TrackGen provides
``GateGenerator``. It emits gate centers and orientations directly from a gate
generator's anchors and skips the constant-spacing resample, the relaxation and the
inflation of the track pipeline, which makes it much faster.

.. seealso::

   Gate courses are **2.5D**: this tutorial covers the plan-view layout and the runtime
   loop, while :doc:`/gates-3d` covers altitude — the four ``z_profile`` families, the
   ``gate_align`` modes, the 3D localizer, and frame collision.

Prerequisites
-------------

Install TrackGen (:doc:`/getting-started/installation`). The examples run on ``cpu`` as
well as on ``cuda``.

Building the Gate Generator
---------------------------

``presets.gates`` returns the :term:`spec` of a gate pipeline; ``GateGenerator`` builds
it. The key arguments are ``generator`` (which gate generator draws the anchors),
``order`` (how the gates are ordered), ``num_envs`` (the batch size :term:`E`, one
gate sequence per :term:`env`),
``max_gates``, ``gate_width``, ``gate_radius`` and ``gate_solve_iters``
(:ref:`pipeline-gate-presets` lists them all).

.. code-block:: python

   from track_gen import GateGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"
   spec = presets.gates(
       num_envs=E,
       device=device,
       generator="bezier",
       max_gates=32,
       gate_width=0.1,          # the full bar width: keep it small (see below)
       gate_radius=0.025,
       gate_solve_iters=8,
   )
   generator = GateGenerator(spec, seeds=0)     # env e gets seed 0 + e

Each env's :term:`seed` decides its course. A ``GateGenConfig`` builds the same generator (``GateGenerator(GateGenConfig(...))``,
:ref:`generating-ways`).

Generating the Gate Sequence
----------------------------

.. code-block:: python

   gates = generator.generate()

   G = generator.max_gates
   position = gates.position.numpy().reshape(E, G, 3)   # vec3f gate centers
   tangent = gates.tangent.numpy().reshape(E, G, 3)
   valid = gates.valid.numpy() != 0                     # [E] bool
   count = gates.count.numpy()                          # [E] real gates per env

The default ``on_invalid="poison"`` gives invalid envs zero count and NaN fields.
Check ``valid`` before using geometry, and guard zero count before modulo indexing.
Set ``on_invalid="keep"`` explicitly when inspecting failed gate geometry.

``GateGenerator`` allocates all its buffers and runs the pipeline once at construction,
exactly like ``TrackGenerator``; ``generate(advance=True)`` builds a new batch. On
``cuda``, the first ``generate()`` records the pipeline into the generator's own CUDA
graph and every call replays it. Called while you :term:`capture` a graph of your own,
``generate()`` is recorded into your graph instead (see
:doc:`/tutorials/cuda-graph-in-a-sim`).

The pipeline it runs is a list of :term:`stages <stage>` (generator, order, separation,
altitude, lift, tangents, frames, validity) that :ref:`pipeline-gate-stages` describes.
``presets.gates`` can also place gates at equal arc-length stations along any *track*
generator's centerline; ``repulsive``, which has no gate generator, does that by default:

.. code-block:: python

   spec = presets.gates(num_envs=8, device=device, generator="repulsive", gates=12)
   serpentine = GateGenerator(spec, seeds=0).generate()   # 12 gates per valid env

.. snippet: after

   _valid = serpentine.valid.numpy() != 0
   _count = serpentine.count.numpy()
   assert (_count[_valid] == 12).all() and (_count[~_valid] == 0).all()

(``repulsive`` is an iterative generator, much slower than the others: the example uses
8 envs.)

Reading the Output Fields
-------------------------

Shapes after the reshape (``G = max_gates``; on the device every per-gate field is a flat
``[E * G]`` array, :doc:`/user-guide/results`):

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``position``
     - ``[E, G, 3]`` (``vec3f``)
     - Gate centers. Slots at index ``i >= count[e]`` hold NaN padding.
   * - ``tangent``
     - ``[E, G, 3]`` (``vec3f``)
     - Unit tangent at each gate (direction of traversal).
   * - ``orientation``
     - ``[E, G, 4]`` (``quatf``)
     - Roll-free gate pose ``(x, y, z, w)`` with local x = gate forward, y = left,
       z = up; the left axis is ``wp.quat_rotate(orientation[e, i], wp.vec3f(0, 1,
       0))``, and ``forward`` (``[E, G, 3]``) is its x axis.
   * - ``half_size``
     - ``[E, G]`` (``float32``)
     - Square-opening half-extent of each gate: ``0.5 * gate_width``, where
       ``gate_width`` is the full bar width (see :ref:`gate-width-config`), per env
       (:ref:`gate-width-per-env`).
   * - ``left``, ``right``
     - ``[E, G, 3]`` (``vec3f``)
     - Gate endpoints: ``position ± half_size * left_axis``. With ``gate_width=0.0``
       (the ``presets.gates`` default, hence ``half_size=0``) these collapse onto the
       gate center (point gates).
   * - ``valid``
     - ``[E]`` (``int32``)
     - ``1`` when the sequence passed every check (:term:`valid`), ``0`` for an
       :term:`invalid env`. A
       sequence with bars (a gate with a non-zero ``half_size``) is also invalid when
       two of its gate bars cross: `Bar width and invalid envs`_.
   * - ``fail_mask``
     - ``[E]`` (``int32``)
     - Why a sequence is invalid: one bit per failed check (:term:`fail bits`:
       ``count``, ``finite``,
       ``tangent``, ``spacing``, ``crossing``, ``grade``; ``gates.fail_bits`` maps the
       names to bits). ``valid`` is ``fail_mask == 0``. See :ref:`invalid-envs`.
   * - ``count``
     - ``[E]`` (``int32``)
     - Real gates per env. Slots at index ``i >= count[e]`` are NaN-padded.
   * - ``generation``
     - ``[E]`` (``int32``)
     - The :term:`epoch` each sequence was generated at: ``0`` until you advance it,
       then ``k`` after the ``k``-th advance (:ref:`generating-seeds`).

Slicing Real Gates
------------------

Like a ``Track``, the ``GateSequence`` arrays are NaN-padded past each env's real gate
count. Use ``count`` to slice the real gates, and skip the invalid envs:

.. code-block:: python

   if valid[0]:
       n0 = int(count[0])
       pos_env0 = position[0, :n0]   # real gate centers of env 0
       tan_env0 = tangent[0, :n0]    # real gate tangents of env 0
   courses = [position[e, :count[e]] for e in range(E) if valid[e]]

Gate Figure
-----------

.. figure:: ../assets/readme-gate-strip.png

   Gate self-collision relaxation. Top: raw anchors (``gate_solve_iters=0``);
   bottom: after the solve, with gate tangents and ``gate_width`` bars.

Gate Ordering per Generator
---------------------------

The ordering of the gates in the output sequence is ``presets.gates(order=...)``
(``GateGenConfig(gate_ordering=...)`` in a config). An ordering is a registered ``order``
stage that the pipeline applies to the generator's anchors
(:ref:`pipeline-ordering-registry`): the built-ins are ``"ccw"``, ``"raw"`` and
``"random_pairs"``, and a plugin can register its own (:ref:`writing-an-ordering`). The
default, ``order=None``, is the generator's default order: ``"ccw"`` for the five
built-ins, ``"raw"`` for a plugin generator that accepts only ``"raw"``. Each generator
rules out the orderings that make no sense for its anchors, and accepts every other one,
a plugin's included:

.. list-table::
   :header-rows: 1

   * - Generator
     - Anchors
     - Excluded ordering
     - Built-in orderings it accepts
   * - ``"bezier"``, ``"hull"``
     - an unordered cloud of grid points
     - ``"raw"``
     - ``"ccw"``, ``"random_pairs"``
   * - ``"polar"``, ``"voronoi"``, ``"checkpoint"``
     - a loop, in generation order
     - ``"random_pairs"``
     - ``"ccw"``, ``"raw"``

.. code-block:: python

   spec = presets.gates(num_envs=E, device=device, generator="bezier",
                        order="random_pairs", gate_width=0.05)
   shuffled = GateGenerator(spec, seeds=0).generate()

An unknown ordering raises ``ValueError`` listing the registered orders; one the
generator does not accept raises ``ValueError`` listing the orders it accepts. Both are
raised when the generator is built. The ordering changes the course, and so how many
sequences are valid at a given bar width (``random_pairs`` makes more bars cross than
``ccw``).

.. note::

   The ``"ccw"`` ordering name is kept for API compatibility. Its current
   centroid-angle convention is **clockwise** in standard xy coordinates.

The ``gate_solve_iters`` Collision Behavior
-------------------------------------------

Gate centers are treated as disks with radius ``gate_radius``. The center
spacing target between adjacent gates is ``2 * gate_radius``. Setting
``gate_solve_iters > 0`` runs up to that many iterations of a deterministic
pairwise push that separates overlapping gate disks before recomputing tangents.

Set ``gate_solve_iters=0`` to inspect the raw anchors before the collision
solve. The anchors are still ordered and bounding-box normalized — they are just
not spread apart yet, so close gates may overlap.

When ``gate_solve_iters > 0``, the collision solve may expand the final bounding
box when necessary to satisfy the requested ``gate_radius``. A sequence with a
positive bar width whose gate bars still cross after the solve is invalid (the check runs
per env: :ref:`gate-width-per-env`).

.. seealso::

   :doc:`/relaxation/gates` documents the gate collision solve in depth — the sphere
   model and ``2 * gate_radius`` target, the per-env Gauss-Seidel semantics, the
   determinism of the coincident-gate tie-break, and its behaviour under graph capture.

.. note::

   ``max_gates`` must be at least the chosen generator's reachable gate count (its
   capacity). ``min_gates`` rejects a configuration that the generator cannot satisfy.
   Both bounds are checked when the generator is built (see
   :ref:`gate-capacity-config`).

.. _gate-width-per-env:

Per-env bar width and course size
---------------------------------

**Bar width.** ``gate_width`` is a :term:`dynamic field` of the ``gate_frames.default``
stage: a per-env ``[E]`` ``float32`` array, the **full** bar width (``>= 0``;
``half_size = 0.5 * gate_width``), initialised from ``presets.gates(gate_width=)`` or
``GateGenConfig.gate_width``. Draw it per env, or change it with a curriculum (start
wide, then narrow it), with no rebuild (:doc:`/user-guide/curricula`):

.. code-block:: python

   # every env draws its own bar width at every generation
   drawn = GateGenerator(presets.gates(num_envs=E, device=device,
                                       randomize={"gate_frames.gate_width": (0.02, 0.2)}))

   # or a curriculum: start with wide gates (easy to pass through), narrow them later;
   # a larger size keeps the yield of wide bars (below)
   gen = GateGenerator(presets.gates(num_envs=E, device=device, gate_width=0.4,
                                     size=3.0))
   gen.params["gate_frames"].set("gate_width", 0.2)       # a scalar, or an [E] array
   gen.generate(advance=True)

.. snippet: after

   current = gen.generate()
   hs = current.half_size.numpy().reshape(E, -1)
   current_valid = current.valid.numpy() != 0
   assert (abs(hs[current_valid, 0] - 0.1) < 1e-6).all()
   assert (current.count.numpy()[~current_valid] == 0).all()

``gate_width`` is the opening the agent passes through (the posts and frame members sit
at its ends): a wider gate is easier to fly through, a narrower one harder, and a
width-0 (point) gate can never be passed. Only the generation yield moves the other way:
wide bars cross more often (`Bar width and invalid envs`_).

A negative array value is clamped to ``0`` where the kernel reads it (a negative scalar
raises ``ValueError``; in a ``Course`` in gates mode, a scalar ``<= 0`` too). The bars are
absolute: they do not grow with the course size.

**The crossing check runs per env.** ``validity.gate_default`` checks bar crossings (fail
bit ``crossing``) for every env that has a real gate with a non-zero ``half_size`` (a
NaN counts as non-zero), and skips the check for the others. So a curriculum that starts
from point gates is checked as soon as its bars widen, with the same decisions as a
pipeline built with that width, and point-gate envs cost nothing (bars of width zero never
cross). Posed gates (a generator that provides ``half_size``) are checked the same way.

**Course size.** The gate generators have a per-env :term:`size` too, under one of two keys:
``generator.size`` for the ``bezier`` and ``hull`` gate generators (it scales their
corner grid) and for the track generator of ``source="stations"`` / ``"relaxed_stations"``
gates; ``normalize.size`` for ``polar``, ``voronoi`` and ``checkpoint`` (their anchors are
re-normalized by the ``normalize.extent`` stage, which holds the size):

.. code-block:: python

   presets.gates(num_envs=E, device=device, randomize={"generator.size": (0.8, 1.5)})
   presets.gates(num_envs=E, device=device, generator="polar",
                 randomize={"normalize.size": (0.8, 1.5)})

The wrong key raises with a hint naming the right one; ``presets.gates(size=)`` and
``GateGenConfig.size`` set the initial value under the right key. The size scales the
gate positions, not ``gate_radius`` or the bars. The gate **count** is not a per-env
axis: a gate generator's ``min_num_points`` (``bezier``, ``hull``) or knot, anchor or
checkpoint count is its gate capacity. See :ref:`tuning-per-env`.

.. _gate-width-invalid-fraction:

Bar width and invalid envs
--------------------------

Every gate pipeline checks each env's bars for crossings (fail bit ``crossing``), and an
env whose bars cross is invalid: a slot your agents cannot train on until it is
regenerated (:ref:`invalid-envs`). Bars cross when they are wide relative to the gate
spacing, so **the bar width decides how many envs are valid**. The table is a historical
measurement before the 0.3 hashed-RNG switch (``E = 64``, seeds ``0..63``, CPU).
Run the example below to measure the current seed mapping:

.. list-table::
   :header-rows: 1
   :widths: 28 14 14 14 14 16

   * - gate generator
     - ``gate_width`` 0.05
     - 0.1
     - 0.2
     - 0.3
     - 0.4
   * - ``bezier`` (the default)
     - 0 %
     - 3 %
     - 56 %
     - 86 %
     - 92 %
   * - ``hull``
     - 0 %
     - 8 %
     - 47 %
     - 81 %
     - 94 %
   * - ``voronoi``
     - 0 %
     - 2 %
     - 16 %
     - 39 %
     - 73 %
   * - ``checkpoint``
     - 0 %
     - 3 %
     - 14 %
     - 31 %
     - 52 %
   * - ``polar``
     - 0 %
     - 0 %
     - 0 %
     - 5 %
     - 19 %
   * - ``bezier``, ``size = 2``
     - 0 %
     - 0 %
     - 9 %
     - 39 %
     - 56 %
   * - ``bezier``, ``size = 3``
     - 0 %
     - 0 %
     - 2 %
     - 9 %
     - 33 %

.. code-block:: python

   widths = (0.05, 0.1, 0.2, 0.3, 0.4)
   rows = [("bezier", 1.0), ("hull", 1.0), ("voronoi", 1.0), ("checkpoint", 1.0),
           ("polar", 1.0), ("bezier", 2.0), ("bezier", 3.0)]
   table = {}
   for name, size in rows:
       fractions = []
       for w in widths:
           gen = GateGenerator(presets.gates(num_envs=E, device=device, generator=name,
                                             gate_width=w, size=size), seeds=0)
           fractions.append((gen.generate().valid.numpy() == 0).mean())
       table[name, size] = fractions
       print(f"{name:<11} size {size}: " + "  ".join(f"{f:4.0%}" for f in fractions))

.. snippet: after

   assert len(table) == len(rows)
   assert all(0.0 <= fraction <= 1.0 for row in table.values() for fraction in row)
   assert all(len(row) == len(widths) for row in table.values())

- **Keep ``gate_width`` small** at ``size = 1`` (``0.1`` or below for ``bezier`` and
  ``hull``), or **raise the size with the width**: the gates spread out while the bars
  stay absolute, so wide gates are usable on every generator at a larger size. The
  ordering matters too: historical ``order="random_pairs"`` measurements produced
  more invalid envs than the default ordering. Measure it for your current seeds.
- **Check the fraction** after construction, and at the ``hi`` of a randomized or
  curriculum range: ``(gen.generate().valid.numpy() == 0).mean()``
  (``course.result.valid`` for a ``Course``).
- **In an RL loop**, an invalid env needs a new course, not a respawn: at
  ``gate_width = 0.2`` the auto-reset of :ref:`rl-auto-reset` regenerates about half of a
  ``bezier`` batch at every step.

Buffer Reuse and Snapshots
--------------------------

The same ``GateSequence`` instance and its Warp arrays are reused on every generation.
For an independent snapshot use ``gates.clone()`` (it allocates and blocks: off the hot
path, :ref:`results-views`):

.. code-block:: python

   snapshot = gates.clone()   # independent copy; safe to keep across generate()

Flying Through the Gates: Progress and Post Collision for RL
------------------------------------------------------------

Generation gives you the gate geometry; an RL or controller loop also needs to
know, each step, *did the drone just clear the next gate?* and *did it clip a
gate post?* A :term:`course` (:doc:`/utilities/course`) bundles gate generation
with the checkpoint and progress tracker and an optional gate-post collision checker
in one object, so you never wire those together by hand. Its lifecycle is
**construct → bind → step / reset / regenerate**, identical to track mode: only the
preset and the contact type differ.

In gates mode the gates themselves become the checkpoints, so there is no
``checkpoint_spacing``. Gate-post collision is ``collision=<post radius>``: a
``DiscChecker`` over the gate-bar endpoints (``left`` / ``right``), with the posts array
rebuilt on device on every regeneration (``collision=tools.FrameCollision(...)`` tests the
gate frames instead, :doc:`/gates-3d`; ``None`` is progress only). ``Course.gates``
requires ``gate_width > 0``: a width-0 gate has no bar and can never be passed.

Build a course from a preset, ``bind`` the sim's own state buffers (they are read in
place every step), then in the step loop read the progress ``events`` for gate rewards,
``contacts`` (a ``DiscContact`` here) for a post-collision penalty, and ``valid`` to
leave out the envs whose gate sequence is invalid:

.. snippet: group=course

.. code-block:: python

   import numpy as np
   import warp as wp

   from track_gen.course import Course
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   course = Course.gates(presets.gates(num_envs=E, device=device, gate_width=0.1,
                                       max_gates=32),   # 0.4-wide bars cross at size 1
                         collision=0.03,                # post radius: DiscChecker posts
                         seeds=7)

   # Bind the sim's own buffers: position (vec3f) drives gate progress;
   # orientation (quatf) + half_extents (vec2f) are the oriented box tested
   # against the posts. The sim writes these in place each step. (Without
   # post collision, bind position alone.)
   position = wp.zeros(E, dtype=wp.vec3f, device=device)
   orientation = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   half_extents = wp.full(E, wp.vec2f(0.01, 0.01), dtype=wp.vec2f, device=device)
   course.bind(position=position, orientation=orientation, half_extents=half_extents)

   regen = course.inputs["regen_mask"]      # stable [E] int32 mask for regenerate(mask=)

   for step in range(40):
       # sim.step() writes `position` (and `orientation`) in place here.

       res = course.step()                       # events + contacts + frame + valid
       valid = res.valid.numpy() != 0            # read it first: a regeneration rewrites it
       passed = res.events.passed.numpy()        # [E] int32: cleared a gate this step
       dist = res.events.dist_to_next.numpy()    # [E] float32: distance to next gate
       laps = res.events.laps.numpy()            # [E] int32: laps completed
       hit = res.contacts.hit.numpy()            # [E] int32: 1 == box touched a post
       disc = res.contacts.disc.numpy()          # [E] int32: deepest post, -1 == none
       s = res.frame.s.numpy()                   # [E] float32: arc length on the course line

       # Gate on `valid` with a select, never a multiply: NaN * 0 is NaN.
       reward = np.where(valid, 10.0 * passed - 5.0 * hit, 0.0)   # gate bonus, post penalty
       obs_s = np.where(valid, s, 0.0)                            # NaN-free observation
       done = (hit != 0) | (laps >= 2) | ~valid                   # done |= 1 - valid

       if done.any():
           regen.assign(done.astype(np.int32))
           course.regenerate(mask=regen, advance=True)   # NEW gate courses for the done envs
           # ...and respawn those envs' agents on their new course in your sim.

   # Per-env respawn on the SAME course (a valid env that crashed): clear its progress.
   respawn = np.zeros(E, np.int32)
   respawn[0] = 1
   course.reset(wp.array(respawn, dtype=wp.int32, device=device))

   # New courses for everyone, from new seeds: whole-batch regeneration + progress reset.
   # The posts are rebuilt on device onto the new gates automatically.
   course.generate(seeds=123)

.. snippet: after

   assert np.isfinite(reward).all() and np.isfinite(obs_s).all()

``res.events`` gives the progress signals: ``passed`` for a gate-clear bonus,
``dist_to_next`` for a negative-delta-distance shaping term, and ``progress`` for the
total gates cleared since the last reset (``next_checkpoint`` is the current target
gate). ``res.contacts`` is a :class:`~track_gen.collision.DiscContact` (not the
``BoxContact`` of track mode): ``hit`` is the reward and termination signal, ``disc`` the
deepest-penetrating post (``gate = disc // 2`` since posts interleave ``left`` /
``right``), and ``depth`` / ``nearest`` drive a contact response. With
``collision=None`` the course is progress-only and ``res.contacts is None``.

``res.valid`` is the generator's ``[E]`` validity (``fail_mask`` says which check
failed). The bar width sets how many courses are valid: with the ``bezier`` gate
generator at size 1, 62 of 64 envs (seeds 7) at ``gate_width=0.1``, 5 of 64 at ``0.4``
(crossing bars; `Bar width and invalid envs`_). An invalid env reads like an env with no
gates: no post hit, frozen progress (``passed = 0``, ``dist_to_next = 0``), a NaN
``frame`` (``segment = -1``). Ungated, its frozen progress and NaN frame would reach the
reward and the observation at every step, and ``reset`` would respawn it on the same
invalid course forever: end it and regenerate it with ``advance=True``, as above
(:ref:`rl-valid-recipe`). ``regenerate(mask=...)`` recomputes the whole batch on every
eager call, even with no env masked, so only call it when one is; recorded into a CUDA
graph with conditional nodes, an all-zero mask is skipped on device (:ref:`rl-cost`).

Post collision backend
~~~~~~~~~~~~~~~~~~~~~~

Gates mode has no out-of-bounds band (there is nothing to leave), so the only collision
is the optional gate posts or frames. ``collision=<radius>`` builds a ``DiscChecker``
over the interleaved ``left`` / ``right`` bar endpoints, rebuilt on device at every
regeneration with each env's post count (``2 * count``) and the gates' ``valid``: only
the real posts are scanned, and an invalid env reports no hit. See
:ref:`collision-backends` for the disc-obstacle checker (the same one makes cones
physical) and :doc:`/utilities/progress` for the checkpoint and reward contract that
gates reuse.

CUDA-graph composability
~~~~~~~~~~~~~~~~~~~~~~~~

``step()`` and ``reset()`` never wait on the host, so the whole step (gate progress
update plus post-collision query) records into your own sim graph alongside the physics,
with no flag to set. ``course.regenerate()`` records into the same graph too, so new gate
courses and the steps that drive them can replay as one unit. Keep writing into the SAME
bound buffers after capture (rebinding leaves the captured graph reading the old
pointers, and emits a ``RebindAfterCaptureWarning``). See
:doc:`/tutorials/cuda-graph-in-a-sim` for the capture-and-replay pattern and
:doc:`/user-guide/rl-auto-reset` for per-env auto-reset.

Full Example
------------

.. snippet: group=full

.. code-block:: python

   import numpy as np
   from track_gen import GateGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"
   spec = presets.gates(num_envs=E, device=device, generator="bezier", max_gates=32,
                        gate_width=0.1, gate_radius=0.025, gate_solve_iters=8)
   generator = GateGenerator(spec, seeds=0)
   gates = generator.generate()

   G = generator.max_gates
   position = gates.position.numpy().reshape(E, G, 3)
   left = gates.left.numpy().reshape(E, G, 3)
   right = gates.right.numpy().reshape(E, G, 3)
   valid = gates.valid.numpy() != 0
   count = gates.count.numpy()
   print(f"{(~valid).mean():.1%} invalid")     # measure the current batch

   if valid[0]:
       n0 = int(count[0])
       print(f"env 0: {n0} gates")
       print("gate centers:", position[0, :n0])
