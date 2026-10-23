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
  when you construct or bind, never in the hot path: the members of the source a
  utility reads, and the arrays you bind (see `What a utility reads`_).
- **Any source with the right members** — a utility takes a ``Track``, a
  ``GateSequence``, a ``Centerline``, an env-slice ``view()`` of one, or your own
  object with the members it reads (the capability protocols of
  :ref:`results-protocols`).
- **Input binding** — latch a tool onto your sim's stable pose buffers once
  (constructor kwargs or ``bind``/``bind_inputs``) and call
  ``update()``/``query()`` with no arguments thereafter.
- **Undefined for invalid envs** — gate on ``valid`` from the generator
  result, as everywhere in the library.

.. _utilities-sources:

What a utility reads
--------------------

Each utility names the capability it needs as a protocol of ``track_gen.results``
(:ref:`results-protocols`) and, at construction, checks exactly the members it reads,
through :func:`~track_gen.results.check_source`: each must exist, be a contiguous 1-D
array of its schema dtype, ``[E * stride]`` per point or ``[E]`` per env, on one device.
The layout comes from the source's ``layout`` when it is a ``Layout``, else is inferred
from ``count`` and the first per-point member (an unrelated ``layout`` attribute of your
own object is ignored). Members the utility does not read are never looked at, so a
``Track`` with a placeholder ``winding`` works with a ``CollisionChecker``.

**The arrays are bound at construction.** A utility reads each member it checked once,
when it is built, and passes those arrays (stable pointers) to every call, eager or
recorded in your graph: reassigning a member of the source afterwards (or a proxy that
would compute a new array on each access) changes nothing, and nothing unchecked ever
reaches a kernel. Write new data INTO the source's arrays (a generator does), or build a
new utility for new arrays.

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Utility
     - Protocol
     - Members checked (= read)
   * - ``CollisionChecker`` (both methods)
     - ``HasBand``
     - ``center``, ``outer``, ``inner``, ``count``
   * - ``PropSampler``
     - ``HasBand``
     - ``outer``, ``inner``, ``count``, ``valid``
   * - ``CheckpointSampler``
     - ``HasBand``
     - ``center``, ``inner``, ``outer``, ``count``, ``valid``
   * - ``HeightFieldBaker``
     - ``HasBand``
     - ``outer``, ``center``, ``count``, ``valid``
   * - ``TrackLocalizer``
     - ``HasCenterline``
     - ``center``, ``arclen``, ``length``, ``count``
   * - ``localize.curvature``
     - ``HasCenterline``
     - ``center``, ``count``
   * - ``localize.speed_profile``
     - ``HasCenterline``
     - ``center``, ``count``, ``arclen``, ``length``
   * - ``CheckpointSet.from_gates``
     - ``HasGateFrames``
     - ``position``, ``left``, ``right``, ``forward``, ``half_size``, ``count``
   * - ``FrameChecker`` (internal to a gates-mode ``Course`` in 0.x)
     - ``HasGateFrames``
     - ``position``, ``orientation``, ``half_size``, ``count``, ``valid`` (its
       ``num_envs`` argument must equal the source's)
   * - ``DiscChecker``, ``ProgressTracker``
     - none
     - raw arrays (``DiscChecker``) or a ``CheckpointSet`` (``ProgressTracker``),
       checked as before

A source that lacks a member raises ``TypeError`` naming it, and a member of the wrong
shape, dtype or device ``ValueError``, prefixed by the utility:

.. code-block:: text

   TypeError: CollisionChecker needs a HasBand source (center, outer, inner, count, valid):
     Centerline has no 'outer', 'inner'
   ValueError: CollisionChecker: Track.inner: expected a contiguous [3072] vec3f
     ([E*stride] = [8*384]) array on cpu, got shape (3064,) vec3f on cpu

These replace the earlier ``track batch layout invalid`` errors, which inferred the
stride from one field and could miss a mismatched one. A ``Centerline`` (for example
``course.course_line.centerline`` in gates mode) works with ``TrackLocalizer``,
``curvature`` and ``speed_profile``; the band utilities refuse it with the ``TypeError``
above.

**Views.** A utility built on ``track.view(e0, e1)`` reads that env range exactly as it
would read an owned copy of the same rows, and a utility built on a mixed batch's shared
result reads every env (:ref:`results-mixed-batches`).

**Topology.** Every utility that reads a result supports closed loops only (its
``topology`` class attribute is ``"closed"``) and refuses a source whose
``layout.closed`` is ``False`` (:ref:`results-topology`). ``DiscChecker`` reads raw
disc arrays and has no topology.

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

No utility host-synchronizes or reads device data back in
``query()``/``update()``/``reset()``/``sample()``/``bake()``/``refresh()``:
each call only enqueues kernels on the device's current Warp stream. They
therefore record into YOUR OWN CUDA graph as-is, with no flag to set (the
former ``track_gen.set_capturing`` is a deprecated no-op).

Because the calls no longer block the host, results are ordered by the stream,
not by the call returning. Reading them with ``.numpy()``, from later Warp
launches, or from torch on its default stream is safe as before. A consumer on
a DIFFERENT stream (for example a torch side stream) must order itself
explicitly, e.g. with ``wp.synchronize_device()`` or a recorded ``wp.Event``.
Set ``TRACKGEN_DEBUG_SYNC=1`` to make every library call block on its stream
while debugging asynchronous CUDA errors. It is skipped while the stream (or a Warp
capture on the device) is capturing; do not combine it with a foreign capture on
another stream (e.g. a default ``torch.cuda.graph``), which it cannot detect.

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
