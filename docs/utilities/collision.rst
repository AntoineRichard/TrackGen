Out-of-bounds & obstacle collision
==================================

``track_gen.collision`` answers two questions every batched sim asks: *did
the agent leave the drivable band?* and *did it hit a point obstacle?*

Out-of-bounds checking
----------------------

``track_gen.collision.CollisionChecker`` answers, per oriented box, whether
it left the drivable band — with signed clearance, nearest boundary point,
and inward normal. Two backends: the exact ``segments`` scan (default; no
precompute, reads regenerated tracks automatically) and baked ``sdf`` grids
(cheap queries after a per-regeneration ``bake()``; distances accurate to
about one grid cell). See Performance below for measured numbers; the rule
of thumb: ``segments`` unless a track batch serves hundreds of queries
between regenerations.

.. figure:: ../assets/oob-collision.png
   :alt: Oriented boxes on a track, green inside the band and red out of
         bounds, each with its signed clearance, nearest boundary point,
         and inward normal drawn.

   The full per-box contact report: OOB flag (color), signed clearance
   (label), nearest boundary point (dotted) and inward normal (arrow).

What the ``sdf`` query computes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bake()`` stores, per env, the signed distance to the band boundary at the
texel centres of an ``R × R`` grid (``R = sdf_resolution``; positive on the
road). A query reads the grid through bilinear interpolation, clamped to the
edge texels, and reports the minimum of that interpolated field over the box
**outline** and its centre:

- Inside one grid cell the field along a box side is a quadratic, so the
  query splits each side at the grid lines it crosses (at most ``2R + 1``
  pieces) and minimises every piece in closed form. The result is the exact
  minimum of the interpolated field on the outline, not a sampled one.
- ``distance`` is that minimum and ``oob = 1`` exactly when it is negative.
  An edge that crosses a box side between four on-road corners, typically
  the inner edge of a tight bend, therefore reads OOB, with ``distance``
  close to minus the intrusion depth.
- ``nearest`` and ``normal`` are taken at the point where the minimum is
  reached: ``normal`` is the normalised central-difference gradient of the
  field there (one cell each way), ``nearest`` is that point moved by
  ``-distance`` along it, and ``boundary`` is the baked boundary id at
  ``nearest``.
- Past the grid edge the field clamps to the edge texels, which are
  negative, so a box reaching beyond the grid reads OOB; its ``distance`` is
  the edge texel's, not the true depth.
- A box with non-finite half-extents or yaw reads ``oob = 1`` with NaN
  ``distance``, ``nearest`` and ``normal`` and ``boundary = -1``, like an
  env with a degenerate track; the grid is not read.

The query is only as good as the grid: a feature narrower than about one
cell (a strip of off-road between two parts of the track, a sharp corner)
can vanish from the interpolated field, and a boundary loop lying entirely
inside a box, away from its outline and centre, is not seen.

Performance
-----------

Measured with ``benchmarks/benchmark_collision.py`` on an RTX 5090
(warp-lang 1.18, E = 8192 generated tracks × 8 boxes = 65,536 box queries
per call, default ``TrackGenConfig``; graph replay is GPU time from events):

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 38

   * - backend
     - eager / query
     - graph replay / query
     - precompute per regeneration
   * - ``segments`` (exact)
     - ~0.34 ms
     - ~0.28 ms
     - none
   * - ``sdf`` (128² grids)
     - ~0.045 ms
     - ~0.041 ms
     - ~45 ms ``bake()`` + ~670 MB

Accuracy of ``sdf`` against the exact backend on the same workload (median
cell 16 mm): the signed-distance error is 0.22 mm on average (p99 4.4 mm,
about 0.3 cells); OOB flags agree on 99.91% of boxes. SDF normals are noisy
near the middle of the band (medial axis) and at sharp features — prefer
``segments`` when normals or nearest points drive a contact response.

The racing near-edge mode (``--racing``) places a 0.46 × 0.34 m car box
near the edges of bezier tracks at ``scale=17`` with a 1 m road, on 256²
grids (cells about 120 mm). There, one ``sdf`` query takes ~23 µs per graph
replay for 4096 worlds × 1 box and ~45 µs for 4096 × 16 (``segments``:
~0.85 ms). On 100 worlds × 1024 boxes its OOB flags agree with the exact
backend on 99.949% of boxes: 32 misses (at most 22 mm deep) and 20 false
positives (at most 5 mm from the edge). The query returns the exact
outline minimum of the interpolated field on every one of them; they come
from the grid's resolution.

Rule of thumb: ``segments`` is the default — exact, no memory, no rebake,
and already ~5 ns per box. Switch to ``sdf`` when a track batch serves more
than roughly 200 queries between regenerations and only the OOB flag and
approximate clearance are consumed (e.g. RL reward shaping).

Reproduce::

   python -m benchmarks.benchmark_collision --E 8192
   python -m benchmarks.benchmark_collision --racing
   python -m benchmarks.benchmark_collision --racing --E 100 --B 1024 --no-timing

Disc obstacles (gate posts, cones)
----------------------------------

``track_gen.collision.DiscChecker`` checks oriented boxes against disc
obstacles. Gate posts are two lines of code:

.. code-block:: python

   import numpy as np, warp as wp
   from track_gen.collision import DiscChecker

   dev = seq.left.device                       # bind posts on the same device
   posts = np.empty((E, 2 * G, 3), np.float32)
   posts[:, 0::2] = seq.left.numpy().reshape(E, G, 3)
   posts[:, 1::2] = seq.right.numpy().reshape(E, G, 3)
   checker = DiscChecker(wp.array(posts.reshape(-1, 3), dtype=wp.vec3f,
                                  device=dev),
                         radius=0.03, max_boxes=1, num_envs=E)

NaN padding in the gate arrays carries over and NaN discs are skipped, so no
per-env count bookkeeping is needed. A hit reports the deepest disc; for
interleaved posts, ``gate = disc // 2``.

``posts`` is a host-side snapshot taken via ``.numpy()`` — it does NOT alias
``seq.left``/``seq.right``, so a later regeneration is invisible to it;
rebuild ``posts`` (repeat this recipe) after every ``generate()``. The
``Course`` facade's built-in ``post_radius`` option does this rebuild
device-side, automatically, on every ``generate()``.

.. figure:: ../assets/disc-collision.png
   :alt: Gate posts as disc obstacles with boxes colored by hit.

   Gate posts as discs; the same checker makes cones physical.

API: :class:`track_gen.collision.CollisionChecker`,
:class:`track_gen.collision.DiscChecker` — see the
:doc:`API reference </reference/api>`.
