Hull Generator
==============

The ``hull`` generator produces closed track centerlines in the angle-sorted-loop-with-lobes
family.  It samples a set of random grid-jittered points, orders them by centroid-relative
angle as a cheap static stand-in for a convex hull, interleaves each edge with one
radially displaced midpoint to introduce lobes and pinches, and then smooths the augmented
loop with a closed uniform Catmull-Rom spline.  The key identity of the method is
the midpoint-displacement layer: without it the output would collapse toward a plain
angle-sorted polygon; with it the generator produces the pronounced lobes, tight pinches,
and extended straights that give racing tracks their character. It writes one closed
centerline per :term:`env` at every :term:`generation`, and its shape knobs are per-env
:term:`dynamic fields <dynamic field>` (`Per-env (dynamic) parameters`_).

.. figure:: ../assets/generator-hull.png
   :alt: Sample tracks produced by the hull generator
   :align: center

   Representative centerlines produced by the hull generator at the default configuration.
   Note the mix of outward lobes, inward pinches, and near-straight sections created by
   the midpoint displacement layer.

How It Works
------------

The hull generator executes a fixed sequence of pure-Warp GPU kernels with no
per-call allocation and no host-side branching on generated data, making it
CUDA-graph-capturable.

1. **Point count and position sampling.**
   ``ops.sample_counts`` (salt ``5119``, offset ``1``) draws a per-environment point
   count ``m`` from the uniform integer range ``[min_num_points, max_num_points]`` (the
   Stage reads env ``e``'s ``min_num_points`` from its per-env array, clamped to ``[2,
   max_num_points]``).
   ``ops.sample_grid_points`` (salt ``7919``, offset ``1``) then draws
   ``P = max_num_points`` grid-jittered points per environment from the
   ``ops.grid_cells(min_point_distance)`` grid with bounded duplicate-cell rejection (up
   to eight retries per point).  The cell size is ``min_point_distance * 2``; each
   accepted cell receives a jitter of ``U(-0.5, 0.5)`` in each axis before being
   multiplied by ``scale``, then by the env's ``size[e]``.  The salts are distinct from
   the Bezier generator's
   (``6151`` / ``9781``) so the two generators draw different point sets even at the
   same seed.

2. **Angle-sort (hull-like ordering).**
   The ``ops.angle_sort`` kernel (with the legacy count floor of 1, like the Bezier
   generator) computes the centroid of the first ``m`` points and sorts them
   ascending by ``atan2(dx, dy)`` (x-first convention, matching the Bezier ccw-sort)
   around that centroid.  Entries beyond index ``m`` are written as NaN.  This is an
   insertion sort over a fixed ``P``-length window, so the kernel is single-threaded
   per environment and runs in bounded time.  The result is a simple closed base loop
   — not a true convex hull (which requires dynamic allocation), but a static,
   fixed-shape stand-in that gives a topologically simple ordered polygon.

3. **Midpoint-displacement layer.**
   ``_midpoint_displace_dyn_k`` interleaves each adjacent pair of angle-sorted vertices
   with one displaced edge midpoint, producing an augmented loop of ``2m`` vertices
   (it reads env ``e``'s ``hull_displacement`` from the per-env parameter array):

   .. code-block:: text

       [v_0, mid_01', v_1, mid_12', v_2, ..., mid_(m-1,0)']

   For each edge ``(v_i, v_{i+1 mod m})``, the geometric midpoint ``M_i`` is computed
   and then displaced along the radial direction from the centroid through ``M_i`` by a
   signed per-edge random amount.  Positive values push the midpoint outward (lobe);
   negative values pull it inward (pinch or straight).  Entries beyond index ``2m`` are
   written NaN and ignored by the arc-resampler downstream.  The augmented-loop stride
   is ``2P`` to match the fixed allocation.

4. **Closed Catmull-Rom smoothing.**
   ``ops.closed_catmull_rom(..., count=aug_count, endpoint=True)`` evaluates a closed
   uniform Catmull-Rom spline over the ``2m``
   augmented vertices into a dense buffer of ``2P * num_points_per_segment`` slots per
   environment.  Each segment from vertex ``i`` to vertex ``i+1`` is sampled at
   ``num_points_per_segment`` equally-spaced parameter values ``u = s / (npseg - 1)``.
   Segment indices ``i >= 2m`` (beyond the real augmented count) write NaN and are
   skipped by the arc-resampler.  Index wrap-around is handled by taking indices
   modulo ``cnt = aug_count[e]``, ensuring the spline closes back to the first vertex.

5. **Arc-resample.**
   ``ops.arc_resample`` (shared with the Bezier generator) converts the dense
   Catmull-Rom polyline into exactly ``num_points`` arc-length-uniform output points
   per environment by accumulating segment lengths, building a cumulative arc-length
   table, and interpolating.

6. **Fallback for self-crossers.**
   ``ops.self_intersections`` counts proper crossings in the smooth resampled loop.
   For environments where the smooth result self-crosses,
   ``ops.closed_polyline(..., count=aug_count, endpoint=True, active=crossers)`` emits a
   straight piecewise-linear version of the augmented loop into the dense buffer (only
   for those rows), and ``ops.arc_resample(..., active=crossers)`` overwrites the
   corresponding output centerline rows in place.  Non-crossing rows are never touched
   by the selected resampler.  XPBD can re-round the straight polygonal fallback
   downstream.

Math
----

**Closed uniform Catmull-Rom spline.**

For segment ``i`` (from control vertex ``P_i`` to ``P_{i+1}``), the spline is
evaluated at parameter ``u \in [0, 1)`` using the four surrounding control vertices
``P_{i-1}``, ``P_i``, ``P_{i+1}``, ``P_{i+2}`` (indices taken modulo the augmented
vertex count to close the loop):

.. math::

   P(u) = \tfrac{1}{2}
   \begin{bmatrix} 1 & u & u^2 & u^3 \end{bmatrix}
   \begin{bmatrix}
    0 &  2 &  0 &  0 \\
   -1 &  0 &  1 &  0 \\
    2 & -5 &  4 & -1 \\
   -1 &  3 & -3 &  1
   \end{bmatrix}
   \begin{bmatrix} P_{i-1} \\ P_i \\ P_{i+1} \\ P_{i+2} \end{bmatrix}

Expanding the matrix product, the kernel evaluates:

.. math::

   P(u) = \tfrac{1}{2} \bigl(
     2 P_i
     + (-P_{i-1} + P_{i+1})\,u
     + (2 P_{i-1} - 5 P_i + 4 P_{i+1} - P_{i+2})\,u^2
     + (-P_{i-1} + 3 P_i - 3 P_{i+1} + P_{i+2})\,u^3
   \bigr)

This is the standard tension-0.5 uniform Catmull-Rom formulation: the spline passes
exactly through ``P_i`` at ``u = 0`` and through ``P_{i+1}`` at ``u = 1``, with
C1 continuity at every knot.  The source (``ops.funcs.catmull_rom``, called by
``ops.closed_catmull_rom``) implements this expression directly; the matrix form and the
expanded scalar form are equivalent.

**Midpoint radial displacement.**

Let ``C`` be the centroid of the ``m`` angle-sorted vertices.  For edge ``i``,
let ``M_i = \tfrac{1}{2}(v_i + v_{i+1 \bmod m})`` be the geometric midpoint and
let :math:`\hat{e}_i = (M_i - C) / \|M_i - C\|` be the outward unit radial direction.
The displaced midpoint is:

.. math::

   M_i' = M_i + r_i\,\hat{e}_i,
   \qquad
   r_i \sim U\!\bigl(-d\cdot\|M_i - C\|,\; +d\cdot\|M_i - C\|\bigr)

where ``d = hull_displacement``.  The displacement magnitude therefore scales with
the centroid-to-midpoint distance, making the fractional perturbation relative to
local track radius uniform across edges regardless of the overall shape size.
Positive ``r_i`` bulges the lobe outward; negative ``r_i`` pinches it inward.

*Note:* The task brief describes the displacement range as simply
:math:`[-d, +d]` with ``d = hull_displacement``.  The source shows the range is
:math:`[-d \cdot \|M_i - C\|,\; +d \cdot \|M_i - C\|]` — i.e., scaled by the
centroid-to-midpoint distance.  The source is authoritative.

Parameters
----------

The following ``TrackGenConfig`` fields are consumed by the hull generator.
For the full parameter reference and default values, see the configuration
reference in ``docs/reference/``.

The generator-specific fields below are also the fields of ``HullParams``, the
generator's Params class (core fields such as ``scale`` are not): set them flat or
through ``generator_params`` (see :ref:`builtin-generator-params`). Their rules are
checked when the ``TrackGenerator`` is constructed, and only for ``generator="hull"``.

``min_num_points`` : int
    Minimum number of base corner-anchor points per environment.  Must be >= 2.
    Shared with the Bezier generator.  Larger values guarantee more vertices in
    the base loop, producing more complex shapes even at small ``hull_displacement``.
    Dynamic, an ``int32`` per env in ``[2, max_num_points]`` (below).

``max_num_points`` : int
    Maximum number of corner-anchor points.  Also sets ``P``, the fixed allocation
    width.  Must be >= ``min_num_points``.  A wider range adds per-environment
    diversity; the augmented loop has up to ``2 * max_num_points`` vertices.

``num_points_per_segment`` : int
    Number of Catmull-Rom samples emitted per segment of the augmented loop before
    arc-resampling.  Must be >= 2.  Higher values give the arc-resampler a finer
    cumulative arc-length grid, reducing interpolation error in the final
    ``num_points``-point centerline.

``min_point_distance`` : float
    Minimum grid-cell spacing for corner-position sampling.  Must be in ``(0, 0.5]``.
    Controls the cell grid as ``num_cells = int(1 / (min_point_distance * 2))``.
    Smaller values allow more densely packed anchor points; larger values spread them
    further apart.

``scale`` : float
    Isotropic scale multiplier applied to all sampled corner positions.  Shared with
    the Bezier generator; scales the coordinate range seen by the downstream
    constant-spacing, XPBD-relax, and inflate stages. A core field, static.

``size`` : float
    Per-env size multiplier of the plan view, applied to the corners right after
    ``scale`` (as for ``bezier``: at ``scale = 1``, ``size = s`` is bitwise ``scale =
    s``). Default ``1.0``. Dynamic, in ``[0.1, 10]`` (below).

``hull_displacement`` : float
    *Hull generator only.*  Maximum per-edge radial midpoint displacement expressed
    as a fraction of the centroid-to-midpoint distance (see Math above).  Default
    ``0.15``.  At ``0.0`` the midpoints remain on the edges and the method collapses
    toward a plain angle-sorted loop.  Typical range ``[0.1, 0.4]``; larger values
    produce stronger lobes, pinches, and straight sections at the cost of a higher
    downstream relaxation and thickness burden. Dynamic, in ``[0, 1]`` (below).

.. _hull-dynamic:

Per-env (dynamic) parameters
----------------------------

``hull_displacement`` and ``size`` are **dynamic** ``float32`` fields of ``HullParams``,
and ``min_num_points`` a dynamic ``int32`` one: per-env ``[E]`` arrays, initialised from
the configured values. Draw the ``float32`` ones per env with
``presets.track(generator="hull", randomize={"generator.hull_displacement": (lo,
hi)})``, write any of them with ``gen.params["generator"].set("hull_displacement", ...)``
or from your own graph (a curriculum): see :ref:`tuning-per-env`. A scalar outside the
bounds raises ``ValueError``; an array value outside them is clamped to them when the
kernel reads it.

.. list-table::
   :header-rows: 1
   :widths: 24 9 10 22 22 13

   * - Field (``randomize=`` key)
     - Default
     - Bounds
     - Harder / more varied when
     - Effective range
     - Yield at lo / default / hi
   * - ``hull_displacement`` (``generator.hull_displacement``)
     - ``0.15``
     - ``[0, 1]``
     - higher: lobier, more irregular, tighter pinches
     - the whole range; above 1 a midpoint could be pushed past the centroid, hence
       the bound
     - 0.98 / 0.98 / 0.88
   * - ``size`` (``generator.size``)
     - ``1.0``
     - ``[0.1, 10]``
     - higher: a longer course relative to the (absolute) road width
     - about ``[0.75, 2.8]`` at the default ``n_max`` (384): from about ``2.8`` the
       longest tracks of a large batch truncate; small sizes lose yield to the road
       width
     - 0.83 / 0.99 / 1.00 at 0.5 / 1 / 2 (1.00 at 3)
   * - ``min_num_points`` (``set`` only: ``int32``)
     - ``9``
     - ``[2, max_num_points]``
     - higher: more base corners
     - the whole range; it only narrows the count toward ``max_num_points``
     - 0.96 / 0.99 / 0.98 at 2 / 9 / 13

*Yield* is the fraction of valid tracks with the default pipeline, ``E = 128``, seeds
``0..127`` (CPU), with the knob set to the value for every env (``size``: ``E = 256``,
``min_num_points``: ``E = 512``, seeds ``7919 * e + 13``).

**Size** multiplies the corner grid right after ``scale``, before the midpoint
displacement and the spline, exactly as for ``bezier`` (:ref:`bezier-dynamic`): the plan
view scales, the road width, spacing and z stay absolute, and a larger course needs more
points. The longest default ``hull`` loop of 8192 envs is 8.30 (p99 6.90) at ``scale =
1``; with the default ``n_max = 384`` and spacing ``0.06`` truncation (a failed env)
starts at about ``size = 2.8``. Size ``n_max`` and ``num_points`` for the largest size
with the rule of :ref:`tuning-size`.

**Corner count.** ``min_num_points`` behaves as ``bezier``'s (:ref:`bezier-corner-count`):
the count is uniform in ``[clamp(min_e, 2, max_num_points), max_num_points]`` per env,
from the same draw, over one fixed set of corner positions, so the axis only narrows the
count toward the static ``max_num_points``; ``set`` refuses a scalar outside ``[2,
max_num_points]``; it cannot be drawn by ``randomize=`` (``int32``). The augmented loop
has ``2 * count[e]`` vertices.

**Static knobs, and why.** ``max_num_points`` (sizes the point buffers),
``num_points_per_segment`` (sizes the dense buffer), ``min_point_distance`` (sets the
point grid on the host) and the core ``scale`` (per-env: ``size``).

What Makes It Distinct
----------------------

The hull generator occupies a different region of the shape space than the other four
generators:

- **Versus bezier:** Both generators sample grid-jittered corners and angle-sort them
  around a centroid.  Bezier assembles a closed cubic Bézier spline through those
  corners with handle-length and edginess knobs; hull instead inserts displaced
  midpoints between every pair of sorted vertices and applies Catmull-Rom smoothing.
  The midpoint-displacement layer gives hull its distinctive lobed and pinched
  character at the cost of the fine curviness control that ``rad``, ``edgy``, and
  ``handle_clamp_frac`` provide in bezier.

- **Versus polar:** Polar builds its shape in polar coordinates from the start,
  guaranteeing a smooth, centered, approximately star-shaped output with no local
  fallback needed.  Hull is Cartesian and explicitly inserts non-convex features via
  displacement; it is not centered by construction and carries a higher relaxation
  burden than polar.

- **Versus voronoi:** Voronoi selects anchor sites from a pre-sampled field under one
  of four layout modes and rounds the resulting cycle with Chaikin and Catmull-Rom.
  Hull is simpler (no site field, no layout modes) and its shape family is controlled
  by a single ``hull_displacement`` scalar rather than by site-field topology.

- **Versus checkpoint:** Checkpoint steers a continuous path through radial checkpoints,
  producing an organic flowing loop with continuously varying curvature.  Hull produces
  a star-shaped loop (around the corner centroid) with locally inserted features; the
  two generators produce qualitatively different shape classes.

- **Within the hull family:** At ``hull_displacement = 0``, the output is nearly
  identical to a plain angle-sorted polygon smoothed by Catmull-Rom — essentially a
  smoother bezier with ``rad = 0``.  Increasing ``hull_displacement`` progressively
  introduces lobes and pinches.  Because the displacement direction is always radial,
  the overall topology stays star-shaped around the centroid; very large displacements
  can produce self-crossings that trigger the polygon fallback.

Fallback and Validity
---------------------

The hull generator has a local per-environment polygon fallback for the self-crossing
case.  After the smooth Catmull-Rom centerline is arc-resampled, its self-intersections
are counted.  Any environment whose smooth result self-crosses has its output
overwritten in place by the arc-resampled straight augmented polygon — the
``2m``-vertex piecewise-linear loop of original corners interleaved with displaced
midpoints, without Catmull-Rom smoothing.  This fallback is provably simple if the
displacement is small enough that the midpoints do not cross each other; XPBD
re-rounds the polygonal fallback in the shared relaxation stage.

Environments that survive the smooth pass and those rescued by the polygon fallback
all receive ``out_valid_wp = 1`` at this stage.  Final geometric validity — turning
number, minimum half-width, NaN checks — is decided by the shared post-relax inflation
validity gate, exactly as for the other generators.  The hull generator never rejects
an environment at generation time.
