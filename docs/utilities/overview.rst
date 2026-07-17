Runtime utilities
=================

Beyond generating tracks and gates, ``track_gen`` ships a family of
GPU-batched runtime utilities for the sim loop:

- :doc:`Out-of-bounds & obstacle collision </utilities/collision>` —
  oriented boxes vs the drivable band (exact or SDF backends) and vs disc
  obstacles such as gate posts.
- :doc:`Boundary props </utilities/props>` — cone lines and wall pieces
  along the boundaries, for rendering-only instancing.
- :doc:`Checkpoints & progress </utilities/progress>` — ordered course
  goals from gate sequences or subsampled track centerlines, with pass/lap
  events and reward-ready distances.
- :doc:`Track-frame localization </utilities/localize>` — arc-length ``s``
  and signed lateral offset ``n`` per env, plus centerline curvature and
  curvature-limited speed profiles.
- :doc:`The Course facade </utilities/course>` — one object bundling
  generation, collision, and progress per mode.

.. figure:: ../assets/utilities-overview.png
   :alt: Overview of the collision and props utilities on one track.

   Cones and walls placed by ``track_gen.props``, the effect of spacing,
   and ``track_gen.collision``'s SDF field with boxes classified by the
   exact backend.

Family conventions
------------------

Every utility follows the same contracts, so learning one teaches all:

- **Flat batched layouts** — arrays are flat ``[E * stride]`` Warp arrays,
  NaN-padded past each env's real count.
- **Preallocated in-place results** — per-step methods return the SAME
  result object every call, overwritten in place; call ``clone()`` for a
  snapshot.
- **Eager, NaN-proof validation** — shapes, dtypes, and devices are checked
  when you construct or bind, never in the hot path.
- **Input binding** — latch a tool onto your sim's stable pose buffers once
  (constructor kwargs or ``bind``/``bind_inputs``) and call
  ``update()``/``query()`` with no arguments thereafter.
- **Undefined for invalid envs** — gate on ``valid`` from the generator
  result, as everywhere in the library.

CUDA graphs
-----------

All utilities preallocate their state and results once (stable pointers) and
never allocate in the hot path. Per-step inputs can be BOUND once instead of
passed per call — ``ProgressTracker(cps, position=buf)``,
``DiscChecker(..., position=..., orientation=..., half_extents=...)``, and
``CollisionChecker.bind_inputs(...)`` — after which ``update()``/``query()``
take no arguments and read the buffers in place. Under graph capture this is
the intended pattern: the sim writes its stable pose buffers, then replays
the captured update. (Per-call mode also works under capture, but the SAME
arrays must be passed at capture and every replay.)

Outside capture, every utility follows a per-call ``wp.synchronize()`` after
its kernel launch (blocking, but simple — the codebase idiom). To record
``query()``/``update()``/``sample()`` into YOUR OWN CUDA graph — instead of
relying on the :doc:`Course facade </utilities/course>`'s built-in capture — call
``track_gen.set_capturing(True)`` before opening the capture region so those
per-call syncs are skipped, then restore it (``set_capturing(False)``, or
the value you saved beforehand) once the capture region is closed.

After regenerating
------------------

What needs refreshing after regenerating when wiring tools by hand is listed below.
``Course`` refreshes only its integrated checkpoint, progress, and collision helpers.
External ``TrackLocalizer`` and ``PropSampler`` instances remain caller-owned and
require manual reset or resampling.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - After a regeneration
   * - ``CollisionChecker`` (``method="segments"``)
     - Nothing — reads the current ``Track`` buffers directly.
   * - ``CollisionChecker`` (``method="sdf"``)
     - Call ``bake()``.
   * - ``PropSampler``
     - Call ``sample()``; external samplers are caller-owned.
   * - ``CheckpointSampler``
     - Call ``sample()``.
   * - ``ProgressTracker``
     - Call ``reset(mask)`` for ALL envs (a full progress reset).
   * - ``TrackLocalizer``
     - Call ``reset(mask)`` for ALL envs when warm-started
       (``warm_window=``); nothing for cold-scan localizers. Recompute any
       ``curvature()`` / ``speed_profile()`` arrays.
   * - ``CheckpointSet.from_gates``
     - Nothing — aliases the ``GateSequence`` buffers automatically.
   * - ``Course``
     - ``generate()`` refreshes integrated checkpoint, progress, and collision helpers.
       Reset or resample any caller-owned ``TrackLocalizer`` and ``PropSampler``
       instances manually.
