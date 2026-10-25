.. _collision-backends:

Collision backends
==================

``track_gen.collision`` answers the two questions every batched sim asks: *did the agent
leave the drivable band?* and *did it hit an obstacle?* (a gate post, a cone, a gate
frame). Every check is batched over the :term:`envs <env>` and records into a CUDA graph.

The accuracy and timing tables below are historical measurements from before the
0.3 hashed-RNG switch. The query algorithms are retained; remeasure current geometry
and seeds when comparing performance or accuracy (:ref:`upgrading-0.3-rng`).

Choosing a backend
------------------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Question
     - Utility
     - In a :term:`course`
   * - Out of bounds, exact
     - ``CollisionChecker(track, max_boxes, method="segments")``
     - the :term:`preset` ``Course.track(..., collision="segments")``, which builds
       ``tools.BandCollision("segments")``
   * - Out of bounds, from a baked distance grid
     - ``CollisionChecker(track, max_boxes, method="sdf")``
     - ``collision="sdf"``, or ``collision=tools.BandCollision("sdf",
       sdf_resolution=256, max_boxes=2)`` for other settings
   * - Gate posts, cones (disc obstacles)
     - ``DiscChecker(discs, radius, max_boxes)``
     - ``Course.gates(..., collision=r)``: posts of radius ``r``
       (``tools.DiscCollision(r)``); your own discs:
       ``tools.DiscCollision(r, discs=...)``
   * - Gate frames (a sphere against the frames around the target gate)
     - built by the component
     - ``Course.gates(..., collision=tools.FrameCollision(radius, frame_thickness,
       frame_depth))``
   * - Anything else
     - your own utility
     - your own :term:`component`, named ``"collision"`` to fill
       ``StepResult.contacts`` (:doc:`/contributing/writing-a-component`)

``segments`` is the default method: it is exact, needs no memory and no
rebake, and reads a regenerated track at once. ``sdf`` trades exactness and memory for
cheaper queries: it pays off when a track batch serves many queries between
regenerations and only the out-of-bounds flag and an approximate clearance are consumed
(`Cost`_ has the break-even).

Out-of-bounds checking
----------------------

``track_gen.collision.CollisionChecker`` answers, per oriented box, whether it left the
drivable band, with its signed clearance, the nearest boundary point and the inward
normal.

.. figure:: ../assets/oob-collision.png
   :alt: Oriented boxes on a track, green inside the band and red out of
         bounds, each with its signed clearance, nearest boundary point,
         and inward normal drawn.

   The full per-box contact report: OOB flag (color), signed clearance
   (label), nearest boundary point (dotted) and inward normal (arrow).

``CollisionChecker(track, ...)`` takes any ``HasBand`` source (a ``Track``, a
``track.view(e0, e1)``, or an object with ``center``, ``outer``, ``inner``, ``count``
and ``valid``) and checks the members it reads, ``center``, ``outer``, ``inner`` and
``count``, plus ``valid`` when present, at construction (:ref:`utilities-sources`). A
``Centerline`` has no band and is refused with a ``TypeError``. ``checker.method`` reads
the backend (``"segments"`` or ``"sdf"``), and ``checker.result`` the ``BoxContact``
every ``query()`` returns. With ``method="sdf"``, call ``bake()`` after every
regeneration (a course does it for you).

**Invalid envs.** An env with :term:`valid` ``== 0`` (an :term:`invalid env`) reads like
an env with no track, with both backends: every active box gets ``oob = 1``, NaN
``distance`` / ``nearest`` / ``normal`` and ``boundary = -1``. The SDF bake writes NaN
texels and a ``(-1, -1)``-``(1, 1)`` box for it, whose queries reach the same answer.
Inactive box slots (a NaN position) keep ``oob = 0``. An RL loop that penalises ``oob``
or ends episodes on it must gate on ``valid`` (:ref:`utilities-invalid-envs`,
:ref:`rl-valid-recipe`).

What the ``sdf`` query computes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bake()`` stores, per env, the signed distance to the band boundary at the texel
centres of an ``R × R`` grid (``R = sdf_resolution``; positive on the road). A query
reads the grid through bilinear interpolation, clamped to the edge texels, and reports
the exact minimum of that interpolated field over the box **outline** and its centre:

- Inside one grid cell the field along a box side is a quadratic, so the query splits
  each side at the grid lines it crosses (at most ``2R + 1`` pieces) and minimises every
  piece in closed form. The result is the exact minimum of the interpolated field on the
  outline, not a sampled one.
- ``distance`` is that minimum and ``oob = 1`` exactly when it is negative. An edge that
  crosses a box side between four on-road corners, typically the inner edge of a tight
  bend, therefore reads OOB, with ``distance`` close to minus the intrusion depth.
- ``nearest`` and ``normal`` are taken at the point where the minimum is reached:
  ``normal`` is the normalised central-difference gradient of the field there (one cell
  each way), ``nearest`` is that point moved by ``-distance`` along it, and ``boundary``
  is the baked boundary id at ``nearest``.
- Past the grid edge the field clamps to the edge texels, which are negative, so a box
  reaching beyond the grid reads OOB; its ``distance`` is the edge texel's, not the true
  depth.
- A box whose corners are not finite (NaN or inf half-extents, yaw or position, or
  corners that overflow float32) reads ``oob = 1`` with NaN ``distance``, ``nearest`` and
  ``normal`` and ``boundary = -1``, like an env with a degenerate track; the grid is not
  read.

The query is only as good as the grid: a feature narrower than about one cell (a strip of
off-road between two parts of the track, a sharp corner) can vanish from the interpolated
field, and a boundary loop lying entirely inside a box, away from its outline and centre,
is not seen. SDF normals are noisy near the middle of the band (its medial axis) and at
sharp features: prefer ``segments`` when normals or nearest points drive a contact
response. :ref:`upgrading-sdf` says how this query differs from the one of 0.1.

.. _collision-resolution-check:

Grid resolution and course size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The grid covers each env's own bounding box, so a larger course (a per-env :term:`size`,
:ref:`tuning-size`) gets larger cells while the road keeps its width: with ``bezier`` at
``R = 128``, 12.3 / 6.7 / 4.5 cells across the road at size 1 / 2 / 3, and OOB flags
agreeing with ``segments`` on 99.93 / 99.80 / 99.68 % of the boxes. Under a size
curriculum, accuracy therefore varies per env with the difficulty. Keep
``sdf_resolution >= R_ref * size_hi`` (``R_ref`` the resolution accurate enough at size
1; the memory grows as ``size_hi^2``: the ~670 MB of 128² grids at ``E = 8192`` become
~6 GB at ``size_hi = 3``), or use ``segments``, which is exact at every size.

The components check it for you: ``tools.BandCollision(method="sdf", size_hi=...)`` and
``tools.HeightFieldBake(resolution, size_hi=...)`` emit a ``UserWarning`` at ``attach``
when ``resolution < ceil(reference_resolution * size_hi)`` (``reference_resolution``
defaults to 128, the resolution the accuracy figures here were measured at), naming the
resolution that keeps the accuracy and its memory. ``size_hi="auto"`` (the default) reads
the largest size the generator can draw at construction: a size randomizer's ``hi``, else
the construction size. So a course built by a preset checks it at construction, while a
``CourseConfig`` course declares no ``size_hi`` and never warns. To accept a coarser grid
on purpose, pass a lower ``reference_resolution``; ``size_hi <= 1`` and
``method="segments"`` never warn. A curriculum that widens the size range after
construction is not checked again: pass its final ``hi`` as ``size_hi=``
(:ref:`tuning-size-utilities`).

.. _collision-accuracy:

Accuracy
--------

The accuracy of ``sdf`` against the exact backend, measured on CPU (the results do not
depend on the device; the machine is in ``docs/_data/machine-cpu.txt``):

- **Default workload** (``docs/_data/collision-accuracy-cpu.txt``: the default
  ``TrackGenConfig`` at ``E = 8192``, 8 random boxes per env near the centerline, 128²
  grids): the OOB flags agree on 99.91 % of the boxes, and the signed-distance error is
  0.22 mm on average (p99 4.4 mm, max 40 mm).

  .. literalinclude:: ../_data/collision-accuracy-cpu.txt
     :language: text

- **Racing near-edge workload** (``docs/_data/collision-racing-accuracy-cpu.txt``:
  ``bezier`` tracks at ``scale = 17`` with a 1 m road, a 0.46 × 0.34 m car box placed
  near the edges, 256² grids with cells of about 118 mm, 100 envs × 1024 boxes): the OOB
  flags agree on 99.949 % of the boxes, with 32 false negatives and 20 false positives.
  The query returns the exact outline minimum of the interpolated field on every one of
  them: they come from the grid's resolution.

  .. literalinclude:: ../_data/collision-racing-accuracy-cpu.txt
     :language: text

Reproduce (from the repository root; ``strip_warp_log.py`` drops Warp's load messages):

.. code-block:: bash

   python -m benchmarks.benchmark_collision --E 8192 --cpu 2>&1 \
       | python docs/_tools/strip_warp_log.py
   python -m benchmarks.benchmark_collision --racing --E 100 --B 1024 --no-timing --cpu 2>&1 \
       | python docs/_tools/strip_warp_log.py

.. _collision-cost:

Cost
----

A ``segments`` query scans every boundary segment of the env for every box; an ``sdf``
query reads a few texels per box side, but each regeneration pays a ``bake()`` and the
grids hold ``E × R² × 5`` bytes (a float32 distance and an int8 boundary id per texel).
With ``t_seg`` and ``t_sdf`` the time of one query over the batch and ``t_bake`` the
time of one bake, ``sdf`` is cheaper when a track batch serves more than

.. code-block:: text

   N_queries > t_bake / (t_seg - t_sdf)

queries between two regenerations. The measured times on CUDA (``E = 8192`` tracks × 8
boxes; graph replay is GPU time from events):

.. literalinclude:: ../_data/collision-cuda-default.txt
   :language: text

The racing near-edge workload (4096 envs × 1 box, then × 16 boxes, 256² grids):

.. literalinclude:: ../_data/collision-cuda-racing.txt
   :language: text

.. literalinclude:: ../_data/collision-cuda-racing-b16.txt
   :language: text

The machine and the date of these numbers:

.. include:: ../_data/machine-cuda.txt

:doc:`/contributing/rendering-assets` gives the commands that regenerate them. On CPU the
same benchmark prints eager times only (the default accuracy output above).

Disc obstacles (gate posts, cones)
----------------------------------

``track_gen.collision.DiscChecker`` checks oriented boxes against disc obstacles. Gate
posts take a few lines:

.. code-block:: python

   import numpy as np
   import warp as wp

   from track_gen import GateGenerator
   from track_gen.collision import DiscChecker
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"                     # or "cpu"
   gen = GateGenerator(presets.gates(num_envs=E, device=device, gate_width=0.1))
   seq = gen.generate()
   G = gen.max_gates

   posts = np.empty((E, 2 * G, 3), np.float32)  # left and right posts, interleaved
   posts[:, 0::2] = seq.left.numpy().reshape(E, G, 3)
   posts[:, 1::2] = seq.right.numpy().reshape(E, G, 3)
   checker = DiscChecker(wp.array(posts.reshape(-1, 3), dtype=wp.vec3f, device=device),
                         radius=0.03, max_boxes=1, num_envs=E, valid=seq.valid)

Query it every sim step with your boxes (``[E]`` positions, orientations and planar half
sizes, one box per env here):

.. snippet: setup

   pos = wp.zeros(E, dtype=wp.vec3f, device=device)
   quat = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   he = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)

.. code-block:: python

   contact = checker.query(pos, quat, he)
   hit = contact.hit.numpy() != 0
   gate = contact.disc.numpy() // 2             # the gate of the deepest hit post

NaN padding in the gate arrays carries over and NaN discs are skipped, so no per-env
count bookkeeping is needed. ``DiscChecker`` takes raw disc arrays, not a result object,
so no source check or topology applies to it. A hit reports the deepest disc; for
interleaved posts, ``gate = disc // 2``.

Two optional per-env arrays narrow the scan: ``count=`` (``[E]`` int32, the real discs of
each env: only those are scanned) and ``valid=`` (``[E]`` int32, checked at construction
and read in place): an env with ``valid[e] == 0`` reports no hit (``hit = 0``,
``disc = -1``, ``depth = 0``, NaN ``nearest``). For gate posts pass the sequence's
``valid``, as above.

``posts`` is a host-side snapshot taken through ``.numpy()``: it does NOT alias
``seq.left`` / ``seq.right``, so a later regeneration is invisible to it; rebuild it
after every ``generate()``. The ``tools.DiscCollision(radius)`` component does that on
device, in its posts mode: it reads the gates, builds the posts and their count (``2 *
count``) on device at every regeneration, and passes the gates' ``valid``. It is what
``Course.gates(..., collision=r)`` builds.

.. figure:: ../assets/disc-collision.png
   :alt: Gate posts as disc obstacles with boxes colored by hit.

   Gate posts as discs; the same checker makes cones physical.

Gate frames
-----------

``tools.FrameCollision(radius, frame_thickness, frame_depth)`` checks a sphere of
``radius`` around the agent against the square frames of the gates around its current
target: each frame is four thin boxes (two posts, two bars) of the given thickness and
depth around the gate's opening, and the query covers ``window`` gates (default 4) from
the one before the progress component's ``next_checkpoint``. It reads the gates, takes
the ``position`` input, and returns a frame contact at every step (``hit``, and the
deepest penetration ``depth``); an invalid env hits nothing. Use it
as the collision of a gates course:
``Course.gates(spec, collision=tools.FrameCollision(0.05, 0.02, 0.02))``.

API: :class:`track_gen.collision.CollisionChecker`,
:class:`track_gen.collision.DiscChecker`, :class:`track_gen.tools.BandCollision`,
:class:`track_gen.tools.DiscCollision`, :class:`track_gen.tools.FrameCollision`
(:ref:`reference-runtime`).
