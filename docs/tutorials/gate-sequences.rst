Generating Gate Sequences for Drone Courses
============================================

For drone-style courses where road width is irrelevant, ``track_gen`` provides
``GateGenerator``. It emits gate centers and orientations directly from the
first-stage generator anchors and skips constant-spacing resampling, XPBD
relaxation, and inflation — making it faster than the full track pipeline.

Prerequisites
-------------

Install ``track_gen`` with the ``dev`` extra:

.. code-block:: bash

   uv pip install -e ".[dev]"

Building the Gate Config
--------------------------

``GateGenConfig`` controls every gate-generation parameter. The key fields are
``generator`` (which first-stage method to use), ``gate_ordering``,
``num_envs``, ``max_gates``, ``gate_width``, ``gate_radius``, and
``gate_solve_iters``.

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

   E, device = 64, "cuda"
   config = GateGenConfig(
       generator="bezier",
       gate_ordering="random_pairs",
       num_envs=E,
       max_gates=32,
       gate_width=0.4,
       gate_radius=0.025,
       gate_solve_iters=8,
       device=device,
   )
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

Generating the Gate Sequence
------------------------------

.. code-block:: python

   gates = GateGenerator(config, rng).generate()

   position = wp.to_torch(gates.position).view(E, config.max_gates, 2)
   tangent  = wp.to_torch(gates.tangent).view(E, config.max_gates, 2)
   valid    = wp.to_torch(gates.valid).bool()

``GateGenerator`` pre-allocates all buffers in ``__init__``, exactly like
``TrackGenerator``. On ``cuda``, the first ``generate()`` captures the pipeline
into a ``wp.Graph``; subsequent calls replay it with updated seeds.

Reading the Output Fields
--------------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``position``
     - ``[E, G, 2]``
     - Gate centers (``G = max_gates``). Slots at index ``i >= count[e]`` hold
       NaN padding.
   * - ``tangent``
     - ``[E, G, 2]``
     - Unit tangent at each gate (direction of traversal).
   * - ``normal``
     - ``[E, G, 2]``
     - Unit left-normal at each gate (perpendicular to ``tangent``).
   * - ``left``, ``right``
     - ``[E, G, 2]``
     - Gate endpoints: ``center ± 0.5 * gate_width * normal``. With the default
       ``gate_width=0.0`` these collapse onto the gate center (point gates).
   * - ``valid``
     - ``[E]`` bool
     - Per-sequence validity. A positive ``gate_width`` additionally invalidates
       any sequence whose gate bars cross.
   * - ``count``
     - ``[E]`` int
     - Real gates per environment. Slots at index ``i >= count[e]`` are NaN-padded.

Slicing Real Gates
-------------------

Like ``Track``, the ``GateSequence`` arrays are NaN-padded past each
environment's real gate count. Use ``count`` to slice valid gates:

.. code-block:: python

   count = wp.to_torch(gates.count)

   n0 = int(count[0])
   pos_env0  = position[0, :n0]   # real gate centers for env 0
   tan_env0  = tangent[0,  :n0]   # real gate tangents for env 0

Gate Figure
-----------

.. figure:: ../assets/readme-gate-strip.png

   Phase-2 gate collision solve. Top: raw anchors (``gate_solve_iters=0``);
   bottom: after the solve, with gate tangents and ``gate_width`` bars.

Gate Ordering per Generator
-----------------------------

The ordering of gates in the output sequence is controlled by
``GateGenConfig(gate_ordering=...)``. Not all orderings are supported by every
generator:

.. list-table::
   :header-rows: 1

   * - Generator
     - Supported orderings
   * - ``"bezier"``
     - ``"ccw"``, ``"random_pairs"``
   * - ``"hull"``
     - ``"ccw"``, ``"random_pairs"``
   * - ``"polar"``
     - ``"ccw"``, ``"raw"``
   * - ``"voronoi"``
     - ``"ccw"``, ``"raw"``
   * - ``"checkpoint"``
     - ``"ccw"``, ``"raw"``

.. note::

   The ``"ccw"`` ordering name is kept for API compatibility. Its current
   centroid-angle convention is **clockwise** in standard xy coordinates.

The ``gate_solve_iters`` Collision Behavior
--------------------------------------------

Gate centers are treated as disks with radius ``gate_radius``. The center
spacing target between adjacent gates is ``2 * gate_radius``. Setting
``gate_solve_iters > 0`` runs up to that many iterations of a deterministic
pairwise push that separates overlapping gate disks before recomputing tangents.

Set ``gate_solve_iters=0`` to inspect the raw anchors before the collision
solve. The anchors are still ordered and bounding-box normalized — they are just
not spread apart yet, so close gates may overlap.

When ``gate_solve_iters > 0``, the collision solve may expand the final bounding
box when necessary to satisfy the requested ``gate_radius``. A positive
``gate_width`` causes any sequence whose gate bars still cross after the solve to
be marked invalid.

.. seealso::

   :doc:`/relaxation/gates` documents the gate collision solve in depth — the sphere
   model and ``2 * gate_radius`` target, the per-env Gauss-Seidel semantics, the
   determinism of the coincident-gate tie-break, and its behaviour under graph capture.

.. note::

   ``max_gates`` must be at least the chosen generator's reachable gate count.
   ``min_gates`` rejects a config that the generator cannot satisfy. Both bounds
   are checked at ``GateGenerator`` construction time.

Buffer Reuse and Snapshots
---------------------------

The same ``GateSequence`` instance and its underlying Warp buffers are reused on
every ``generate()`` call. For an independent snapshot use ``gates.clone()``:

.. code-block:: python

   snapshot = gates.clone()   # independent copy; safe to keep across generate()

Flying Through the Gates: Progress and Post Collision for RL
------------------------------------------------------------

Generation gives you the gate geometry; an RL or controller loop also needs to
know, each step, *did the drone just clear the next gate?* and *did it clip a
gate post?* The :doc:`Course facade </utilities/course>` bundles gate generation
with the checkpoint/progress tracker and an optional gate-post collision checker
in one object, so you never wire those together by hand. Its lifecycle is
**construct → bind → generate → step / reset** — identical to track mode, only
the config and the contact type differ.

In ``mode="gates"`` the gates themselves become the checkpoints
(``CheckpointSet.from_gates`` → ``ProgressTracker``), so there is no
``checkpoint_spacing`` — passing it is rejected, as is the track-mode
``collision=`` option. Gate-post collision is opt-in via ``post_radius > 0``: a
``DiscChecker`` over the gate-bar endpoints (``left``/``right``), with the posts
array rebuilt device-side on every regeneration. It requires ``gate_width > 0``
(a width-0 gate has no bar and no posts to hit).

Build a ``Course`` in gates mode, ``bind`` the sim's own state buffers (they are
read in place every step), ``generate`` the batch, then in the step loop read
the progress ``events`` for gate rewards and ``contacts`` (a ``DiscContact``
here) for a post-collision penalty:

.. code-block:: python

   import numpy as np
   import warp as wp
   wp.init()

   from track_gen import GateGenConfig
   from track_gen.course import Course, CourseConfig

   E, device = 4, "cpu"

   course = Course(CourseConfig(
       mode="gates",
       gen=GateGenConfig(num_envs=E, gate_width=0.4, gate_radius=0.025,
                         max_gates=32, device=device),
       seeds=7,
       post_radius=0.03,          # > 0 enables DiscChecker gate-post collision;
                                  # 0 (default) = progress-only. No collision= /
                                  # checkpoint_spacing here — those are track-mode.
   ))

   # Bind the sim's own buffers: position drives gate progress; yaw + half_extents
   # are the oriented box tested against the posts. The sim writes these in place
   # each step. (With post_radius == 0, bind position alone.)
   position     = wp.zeros(E, dtype=wp.vec2f, device=device)
   yaw          = wp.zeros(E, dtype=wp.float32, device=device)
   half_extents = wp.array(np.full((E, 2), 0.01, np.float32),
                           dtype=wp.vec2f, device=device)
   course.bind(position=position, yaw=yaw, half_extents=half_extents)

   seq = course.generate()        # whole batch + posts rebuild + progress reset

   for step in range(40):
       # sim.step() writes `position` (and `yaw`) in place here.

       res    = course.step()                     # events + contacts, no args
       passed = res.events.passed.numpy()         # [E] int32: cleared a gate this step
       dist   = res.events.dist_to_next.numpy()   # [E] float32: distance to next gate
       hit    = res.contacts.hit.numpy()          # [E] int32: 1 == box touched a post
       disc   = res.contacts.disc.numpy()         # [E] int32: deepest post, -1 == none

       reward = 10.0 * passed - 5.0 * hit         # gate bonus, post penalty (shaping)

   # Per-env respawn on the SAME course: clear progress only where mask[e] == 1.
   done = np.zeros(E, np.int32)
   done[0] = 1
   course.reset(wp.array(done, dtype=wp.int32, device=device))

   # New courses for everyone: whole-batch regenerate + full progress reset.
   # The posts are rebuilt device-side onto the new gates automatically.
   course.generate(seeds=123)

``res.events`` gives the progress signals — ``passed`` for a gate-clear bonus,
``dist_to_next`` for a negative-delta-distance shaping term, and ``progress``
for the total gates cleared since the last reset (``next_checkpoint`` is the
current target gate). ``res.contacts`` is a
:class:`~track_gen.collision.DiscContact` (not the ``BoxContact`` of track mode):
``hit`` is the reward/termination signal, ``disc`` the deepest-penetrating post
(``gate = disc // 2`` since posts interleave ``left``/``right``), and ``depth`` /
``nearest`` drive a contact response. With ``post_radius == 0`` the course is
progress-only and ``res.contacts is None``.

Post collision backend
~~~~~~~~~~~~~~~~~~~~~~~~

Gates mode has no out-of-bounds band — there is nothing to leave — so the only
collision is the optional gate posts. ``post_radius > 0`` builds a
``DiscChecker`` over the interleaved ``left``/``right`` bar endpoints; NaN
padding past each env's real gate count carries over and NaN posts are skipped,
so no per-env count bookkeeping is needed. See
:doc:`/utilities/collision` for the disc-obstacle checker (the same one makes
cones physical) and :doc:`/utilities/progress` for the checkpoint/reward
contract that gates reuse.

CUDA-graph composability
~~~~~~~~~~~~~~~~~~~~~~~~~~

``step()`` and ``reset()`` are warp-native and capture-ready: flip the single
shared capture flag with ``track_gen.set_capturing(True)`` and the whole step —
gate progress update plus post-collision query — records into your own sim graph
alongside the physics. Keep writing into the SAME bound buffers after capture
(rebinding leaves the captured graph reading the old pointers). See
:doc:`/tutorials/cuda-graph-in-a-sim` for the capture-and-replay pattern.

Full Example
------------

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

   E, device = 64, "cuda"
   config = GateGenConfig(
       generator="bezier",
       gate_ordering="random_pairs",
       num_envs=E,
       max_gates=32,
       gate_width=0.4,
       gate_radius=0.025,
       gate_solve_iters=8,
       device=device,
   )
   rng   = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gates = GateGenerator(config, rng).generate()

   position = wp.to_torch(gates.position).view(E, config.max_gates, 2)
   tangent  = wp.to_torch(gates.tangent).view(E, config.max_gates, 2)
   left     = wp.to_torch(gates.left).view(E, config.max_gates, 2)
   right    = wp.to_torch(gates.right).view(E, config.max_gates, 2)
   valid    = wp.to_torch(gates.valid).bool()
   count    = wp.to_torch(gates.count)

   n0 = int(count[0])
   print(f"env 0: {n0} gates, valid={bool(valid[0])}")
   print("gate centers:", position[0, :n0])
