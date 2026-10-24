Generating Gate Sequences for Drone Courses
============================================

For drone-style courses where road width is irrelevant, ``track_gen`` provides
``GateGenerator``. It emits gate centers and orientations directly from the
centerline-generator anchors and skips constant-spacing resampling, XPBD
relaxation, and inflation — making it faster than the full track pipeline.

.. seealso::

   Gate courses are **2.5D**: this tutorial covers the plan-view layout and the runtime
   loop, while :doc:`/gates-3d` covers altitude — the four ``z_profile`` families, the
   ``gate_align`` modes, the 3D localizer, and frame collision.

Prerequisites
-------------

Install ``track_gen`` with the ``dev`` extra:

.. code-block:: bash

   uv pip install -e ".[dev]"

Building the Gate Config
--------------------------

``GateGenConfig`` controls every gate-generation parameter. The key fields are
``generator`` (which centerline generator to use), ``gate_ordering``,
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
       gate_width=0.1,
       gate_radius=0.025,
       gate_solve_iters=8,
       device=device,
   )
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

Generating the Gate Sequence
------------------------------

.. code-block:: python

   gates = GateGenerator(config, rng).generate()

   position = wp.to_torch(gates.position).view(E, config.max_gates, 3)
   tangent  = wp.to_torch(gates.tangent).view(E, config.max_gates, 3)
   valid    = wp.to_torch(gates.valid).bool()

``GateGenerator`` pre-allocates all buffers in ``__init__`` and runs the pipeline once
there, exactly like ``TrackGenerator``. On ``cuda``, the first ``generate()`` captures the
pipeline into a ``wp.Graph``; subsequent calls replay it with updated seeds. Called while
you capture, ``generate()`` is recorded into your own graph instead (see
:doc:`/tutorials/cuda-graph-in-a-sim`).

The pipeline it runs is a list of stages (generator, order, separation, altitude, lift,
tangents, frames, validity) that :doc:`/how-it-works/pipeline-preview` describes.
``GateGenerator`` also accepts a spec built with ``presets.gates``, which can, for
example, place gates at equal arc-length stations along any *track* generator's
centerline:

.. code-block:: python

   from track_gen.pipeline import presets

   spec = presets.gates(num_envs=E, device=device, generator="repulsive", gates=12)
   gates = GateGenerator(spec, rng).generate()

Reading the Output Fields
--------------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``position``
     - ``[E, G, 3]``
     - ``vec3f`` gate centers (``G = max_gates``). Slots at index
       ``i >= count[e]`` hold NaN padding.
   * - ``tangent``
     - ``[E, G, 3]``
     - ``vec3f`` unit tangent at each gate (direction of traversal).
   * - ``orientation``
     - ``[E, G]`` ``quatf``
     - Roll-free gate pose ``wp.quatf(x, y, z, w)`` with local x = gate forward,
       y = left, z = up. Replaces the old planar ``normal`` field; recover the
       left axis as ``wp.quat_rotate(orientation[e, i], wp.vec3f(0, 1, 0))``.
   * - ``half_size``
     - ``[E, G]`` ``float32``
     - Square-opening half-extent for each gate: ``0.5 * gate_width``, where
       ``gate_width`` is the full bar width (see :ref:`gate-width-config`), per env
       (`Per-env bar width and course size`_).
   * - ``left``, ``right``
     - ``[E, G, 3]``
     - ``vec3f`` gate endpoints: ``position ± half_size * left_axis``. With the
       default ``gate_width=0.0`` (hence ``half_size=0``) these collapse onto the
       gate center (point gates).
   * - ``valid``
     - ``[E]`` bool
     - Per-sequence validity. A sequence with bars (a gate with a non-zero
       ``half_size``) is also invalid when two of its gate bars cross.
   * - ``fail_mask``
     - ``[E]`` int32
     - Why a sequence is invalid: one bit per failed check (``count``, ``finite``,
       ``tangent``, ``spacing``, ``crossing``, ``grade``; ``gates.fail_bits`` maps the
       names to bits). ``valid`` is ``fail_mask == 0``. See
       :ref:`pipeline-gate-fail-bits`.
   * - ``count``
     - ``[E]`` int
     - Real gates per environment. Slots at index ``i >= count[e]`` are NaN-padded.
   * - ``generation``
     - ``[E]`` int
     - The epoch each sequence was generated at: ``0`` until you advance it (e.g.
       ``generate(advance=True)``), then ``k`` after the ``k``-th advance. A reseed
       through the rng does not change it (see
       :doc:`/tutorials/cuda-graph-in-a-sim`).

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

   Gate self-collision relaxation. Top: raw anchors (``gate_solve_iters=0``);
   bottom: after the solve, with gate tangents and ``gate_width`` bars.

Gate Ordering per Generator
-----------------------------

The ordering of gates in the output sequence is controlled by
``GateGenConfig(gate_ordering=...)``. An ordering is a registered ``order`` stage that the
pipeline applies to the generator's anchors (:ref:`pipeline-ordering-registry`): the
built-ins are ``"ccw"``, ``"raw"`` and ``"random_pairs"``, and a plugin can register its
own (:ref:`writing-an-ordering`). The default, ``gate_ordering=None``, is the generator's
default order: ``"ccw"`` for the five built-ins, ``"raw"`` for a plugin generator that
accepts only ``"raw"``. Each generator rules out the orderings that make no sense for its
anchors, and accepts every other one, a plugin's included:

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

An unknown ordering raises ``ValueError`` listing the registered orders; one the
generator does not accept raises ``ValueError`` listing the orders it accepts. Both are
raised when the ``GateGenerator`` is constructed.

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
box when necessary to satisfy the requested ``gate_radius``. A sequence with a
positive bar width whose gate bars still cross after the solve is marked invalid (the
check runs per env: `Per-env bar width and course size`_).

.. seealso::

   :doc:`/relaxation/gates` documents the gate collision solve in depth — the sphere
   model and ``2 * gate_radius`` target, the per-env Gauss-Seidel semantics, the
   determinism of the coincident-gate tie-break, and its behaviour under graph capture.

.. note::

   ``max_gates`` must be at least the chosen generator's reachable gate count (its
   capacity). ``min_gates`` rejects a config that the generator cannot satisfy. Both
   bounds are checked at ``GateGenerator`` construction time (see
   :ref:`gate-capacity-config`).

.. _gate-width-per-env:

Per-env bar width and course size
---------------------------------

**Bar width.** ``gate_width`` is a dynamic field of the ``gate_frames.default`` stage (since
0.2.0.dev5): a per-env ``[E]`` ``float32`` array, the **full** bar width (``>= 0``;
``half_size = 0.5 * gate_width``), initialised from ``GateGenConfig.gate_width`` or
``presets.gates(gate_width=)``. Draw it per env, or change it with a curriculum (start
wide, then narrow it), with no rebuild:

.. code-block:: python

   from track_gen.pipeline import presets

   # every env draws its own bar width at every generation
   gen = GateGenerator(presets.gates(num_envs=E, device=device,
                                     randomize={"gate_frames.gate_width": (0.0, 0.3)}))

   # or a curriculum: start with wide gates (easy to pass through), narrow them later;
   # a larger size keeps the yield of wide bars (below)
   gen = GateGenerator(presets.gates(num_envs=E, device=device, gate_width=0.4,
                                     size=2.0))
   gen.params["gate_frames"].set("gate_width", 0.2)       # a scalar, or an [E] array
   gen.generate(advance=True)

``gate_width`` is the opening the agent passes through (the posts and frame members sit
at its ends): a wider gate is easier to fly through, a narrower one harder, and a
width-0 (point) gate can never be passed. Only the generation yield moves the other way:
wide bars cross more often (below).

A negative array value is clamped to ``0`` where the kernel reads it (a negative scalar
raises ``ValueError``; in a ``Course`` in gates mode, a scalar ``<= 0`` too). The bars are
absolute: they do not grow with the course size.

**The crossing check runs per env.** ``validity.gate_default`` checks bar crossings (fail
bit ``crossing``) for every env that has a real gate with a non-zero ``half_size`` (a
NaN counts as non-zero), and skips the check for the others. So a curriculum that starts
from point gates is checked as soon as its bars widen, with the same decisions as a
pipeline built with that width, and point-gate envs cost nothing (bars of width zero never
cross). ``validity.gate_default``'s own ``gate_width`` and ``check_crossing`` are inert
since 0.2.0.dev5 (kept so existing specs load, removed in roadmap PR16); left at its
default ``0`` it takes no part in the planner's agreement check, and set to another
value it must equal ``gate_frames.default``'s initial ``gate_width`` (the presets and
``GateGenConfig`` set both). Posed gates (a generator that provides ``half_size``)
are checked the same way, with ``check_crossing="width"`` as well as ``"always"``.

**Yield.** Wide bars cross often, so check the yield at the ``hi`` of your range. Valid
fraction of the default gate pipelines by bar width (CPU, ``E = 512``, the widths mixed
per env in one batch):

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - generator
     - ``gate_width = 0``
     - ``0.2``
     - ``0.5``
   * - ``bezier``
     - 1.000
     - 0.508
     - 0.039
   * - ``hull``
     - 1.000
     - 0.477
     - 0.023
   * - ``polar``
     - 1.000
     - 1.000
     - 0.555
   * - ``voronoi``
     - 1.000
     - 0.812
     - 0.039
   * - ``checkpoint``
     - 1.000
     - 0.828
     - 0.281

(``bezier`` alone, every env at one width: 0.88 at ``0.1``, 0.25 at ``0.3``.)

These yields are at ``size = 1``. The bars cross when they are wide relative to the gate
spacing, and the spacing grows with the course size while the bars stay absolute, so
**raising the size together with** ``gate_width`` **keeps the yield**: wide gates are
usable on every generator at a larger size. Valid fraction of pipelines built at one
width and one size (CPU, ``E = 256``):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - generator, ``gate_width``
     - ``size = 1``
     - ``2``
     - ``3``
   * - ``bezier``, ``0.3``
     - 0.27
     - 0.71
     - 0.87
   * - ``bezier``, ``0.5``
     - 0.04
     - 0.39
     - 0.66
   * - ``hull``, ``0.3``
     - 0.29
     - 0.71
     - 0.87
   * - ``polar``, ``0.5``
     - 0.53
     - 0.98
     - 1.00
   * - ``voronoi``, ``0.3``
     - 0.46
     - 0.90
     - 0.95
   * - ``checkpoint``, ``0.3``
     - 0.69
     - 0.92
     - 0.95

**Course size.** The gate generators have a per-env ``size`` too, under one of two keys:
``generator.size`` for the ``bezier`` and ``hull`` gate generators (it scales their
corner grid) and for the track generator of ``source="stations"`` / ``"relaxed_stations"``
gates; ``normalize.size`` for ``polar``, ``voronoi`` and ``checkpoint`` (their anchors are
re-normalized by the ``normalize.extent`` stage, which holds the size):

.. code-block:: python

   presets.gates(num_envs=E, device=device, randomize={"generator.size": (0.8, 1.5)})
   presets.gates(num_envs=E, device=device, generator="polar",
                 randomize={"normalize.size": (0.8, 1.5)})

The wrong key raises with a hint naming the right one; ``GateGenConfig.size`` and
``presets.gates(size=)`` set the initial value under the right key. The size scales the
gate positions, not ``gate_radius`` or the bars. The gate **count** is not a per-env axis
in 0.x: a gate generator's ``min_num_points`` (``bezier``, ``hull``) or knot, anchor or
checkpoint count is its gate capacity. See :ref:`tuning-per-env`.

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
(a width-0 gate has no bar and no posts to hit): the configured width, checked when the
``Course`` is constructed.

Build a ``Course`` in gates mode, ``bind`` the sim's own state buffers (they are
read in place every step), ``generate`` the batch, then in the step loop read
the progress ``events`` for gate rewards, ``contacts`` (a ``DiscContact``
here) for a post-collision penalty, and ``valid`` to leave out the envs whose gate
sequence is invalid:

.. code-block:: python

   import numpy as np
   import warp as wp
   wp.init()

   from track_gen import GateGenConfig
   from track_gen.course import Course, CourseConfig

   E, device = 4, "cpu"

   course = Course(CourseConfig(
       mode="gates",
       gen=GateGenConfig(num_envs=E, gate_width=0.1, gate_radius=0.025,
                         max_gates=32, device=device),   # 0.4-wide bars cross at size 1
       seeds=7,
       post_radius=0.03,          # > 0 enables DiscChecker gate-post collision;
                                  # 0 (default) = progress-only. No collision= /
                                  # checkpoint_spacing here — those are track-mode.
   ))

   # Bind the sim's own buffers: position (vec3f) drives gate progress;
   # orientation (quatf) + half_extents (vec2f) are the oriented box tested
   # against the posts. The sim writes these in place each step. (With
   # post_radius == 0, bind position alone.)
   position     = wp.zeros(E, dtype=wp.vec3f, device=device)
   orientation  = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (E, 1)).astype(np.float32),
                           dtype=wp.quatf, device=device)   # identity quats
   half_extents = wp.array(np.full((E, 2), 0.01, np.float32),
                           dtype=wp.vec2f, device=device)
   course.bind(position=position, orientation=orientation, half_extents=half_extents)

   seq = course.generate()        # whole batch + posts rebuild + progress reset
   regen = course.inputs["regen_mask"]            # stable [E] int32 mask for regenerate(mask=)

   for step in range(40):
       # sim.step() writes `position` (and `orientation`) in place here.

       res    = course.step()                     # events + contacts + frame + valid
       valid  = res.valid.numpy() != 0            # [E] bool: False == invalid gate sequence
       passed = res.events.passed.numpy()         # [E] int32: cleared a gate this step
       dist   = res.events.dist_to_next.numpy()   # [E] float32: distance to next gate
       laps   = res.events.laps.numpy()           # [E] int32: laps completed
       hit    = res.contacts.hit.numpy()          # [E] int32: 1 == box touched a post
       disc   = res.contacts.disc.numpy()         # [E] int32: deepest post, -1 == none
       s      = res.frame.s.numpy()               # [E] float32: arc length on the course line

       # Gate on `valid` with a select, never a multiply: NaN * 0 is NaN.
       reward = np.where(valid, 10.0 * passed - 5.0 * hit, 0.0)   # gate bonus, post penalty
       obs_s  = np.where(valid, s, 0.0)                            # NaN-free observation
       done   = (hit != 0) | (laps >= 2) | ~valid                  # done |= 1 - valid

       if done.any():
           regen.assign(done.astype(np.int32))
           course.regenerate(mask=regen, advance=True)   # NEW gate courses for the done envs
           # ...and respawn those envs' agents on their new course in your sim.

   # Per-env respawn on the SAME course (a valid env that crashed): clear its progress.
   respawn = np.zeros(E, np.int32)
   respawn[0] = 1
   course.reset(wp.array(respawn, dtype=wp.int32, device=device))

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

``res.valid`` is the generator's ``[E]`` validity (``seq.valid``; ``fail_mask`` says
which check failed). The bar width sets how many courses are valid: with the ``bezier``
gate generator at size 1, 62 of 64 envs (seeds 7) at ``gate_width=0.1``, 5 of 64 at
``0.4`` (crossing bars, fail bit 4; see the yield tables above). An invalid env reads
like an env with no gates: no post hit, frozen
progress (``passed = 0``, ``dist_to_next = 0``), a NaN ``frame`` (``segment = -1``).
Ungated, its frozen progress and NaN frame would reach the reward and the observation at
every step, and ``reset`` would respawn it on the same invalid course forever: end it
and regenerate it with ``advance=True``, as above. ``regenerate(mask=...)`` recomputes the
whole batch on every eager call, even with no env masked, so only call it when one is.
An all-zero mask is skipped on device only in a CUDA graph recorded through a capture
Warp registered, with conditional nodes (CUDA driver and toolkit >= 12.4); a CPU capture
or a plain ``torch.cuda.graph`` reruns it on every replay
(:doc:`/tutorials/cuda-graph-in-a-sim`).

Post collision backend
~~~~~~~~~~~~~~~~~~~~~~~~

Gates mode has no out-of-bounds band — there is nothing to leave — so the only
collision is the optional gate posts. ``post_radius > 0`` builds a
``DiscChecker`` over the interleaved ``left``/``right`` bar endpoints, rebuilt on device
at every regeneration with each env's post count (``2 * count``) and the gates'
``valid``: only the real posts are scanned, and an invalid env reports no hit. See
:doc:`/utilities/collision` for the disc-obstacle checker (the same one makes
cones physical) and :doc:`/utilities/progress` for the checkpoint/reward
contract that gates reuse.

CUDA-graph composability
~~~~~~~~~~~~~~~~~~~~~~~~~~

``step()`` and ``reset()`` are warp-native and never host-synchronize, so the
whole step — gate progress update plus post-collision query — records into your
own sim graph alongside the physics, with no flag to set. ``course.regenerate()``
records into the same graph too, so new gate courses and the steps that drive them can
replay as one unit. Keep writing into the SAME bound buffers after capture (rebinding
leaves the captured graph reading the old pointers, and emits a
``RebindAfterCaptureWarning``). See
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
       gate_width=0.1,
       gate_radius=0.025,
       gate_solve_iters=8,
       device=device,
   )
   rng   = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gates = GateGenerator(config, rng).generate()

   position = wp.to_torch(gates.position).view(E, config.max_gates, 3)
   tangent  = wp.to_torch(gates.tangent).view(E, config.max_gates, 3)
   left     = wp.to_torch(gates.left).view(E, config.max_gates, 3)
   right    = wp.to_torch(gates.right).view(E, config.max_gates, 3)
   valid    = wp.to_torch(gates.valid).bool()
   count    = wp.to_torch(gates.count)

   n0 = int(count[0])
   print(f"env 0: {n0} gates, valid={bool(valid[0])}")
   print("gate centers:", position[0, :n0])
