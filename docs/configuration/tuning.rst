Tuning guide
============

This page collects practical tuning advice: the yield of a batch (the fraction of
:term:`valid` envs) against its diversity, the road width and the point
:term:`spacing`, the relaxation knobs, a throughput recipe and per-generator tips. The
knobs are named by their ``TrackGenConfig`` fields; in a :term:`preset` they are the
parameters of a stage (:ref:`generating-params`): the ``relax_*`` fields are
``relax.xpbd``'s (``presets.track(relax=("xpbd", {"relax_iters": 80}))``), the shape
knobs the generator's (``generator_params={...}``), and ``half_width``, ``spacing``,
``n_max`` and ``scale`` are keywords of ``presets.track``. For the complete parameter
descriptions, see the :doc:`Configuration reference <reference>`.

.. code-block:: python

   from track_gen import TrackGenConfig, TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   cfg = TrackGenConfig(num_envs=E, device=device, relax_iters=80, half_width=0.05)
   spec = presets.track(num_envs=E, device=device, half_width=0.05,
                        relax=("xpbd", {"relax_iters": 80}))
   a, b = TrackGenerator(cfg), TrackGenerator(spec)     # the same generator

.. snippet: after

   import numpy as np
   assert np.array_equal(a.generate().center.numpy(), b.generate().center.numpy(),
                         equal_nan=True)

Yield vs. diversity
-------------------

*Yield* is the fraction of envs that pass the checks after relaxation (the others are
:term:`invalid envs <invalid env>`, :ref:`invalid-envs`). *Diversity* is how varied the resulting track shapes are across a batch.
These two goals pull in opposite directions: the settings that reliably produce
valid tracks tend to constrain the shape space.

A few principles:

- **Raise ``relax_iters`` first.** The default of 50 Chebyshev-accelerated
  sweeps is calibrated for approximately 99.9 % yield at the library defaults
  (what 150 plain-Jacobi sweeps reach, ``relax_accel="none"``). If yield is low, the
  most conservative fix is to increase ``relax_iters``; cost is linear in
  sweeps and it does not restrict shapes. See
  :doc:`/relaxation/convergence` for the measured yield-vs-sweeps curves.

- **Widen the generator's shape range for diversity.** Draw the generator's dynamic
  knobs per env with ``randomize=`` (:ref:`tuning-per-env`): ``rad`` /
  ``handle_clamp_frac`` for the Bezier generator, ``polar_radial_jitter`` for polar,
  ``checkpoint_radius_min_frac`` for checkpoint, and so on, and ``generator.size`` for a
  per-env course size (:ref:`tuning-size`). For
  the Bezier and hull generators, also widen ``[min_num_points, max_num_points]``. For
  the checkpoint generator, increase ``checkpoint_count``; for polar, ``polar_num_knots``.
  Wider shape ranges can introduce harder-to-relax geometries, so watch yield when you
  push these.

- **Mix generators across workers** rather than pushing a single generator to
  its limits. Each centerline generator (``"bezier"``, ``"hull"``, ``"polar"``,
  ``"voronoi"``, ``"checkpoint"``) produces a visually distinct family; using
  different configs across parallel workers is the lowest-friction path to a
  broad distribution.

- **Or mix them inside one batch.** Several generators can write disjoint env ranges
  of one shared ``Track`` (``out=big.view(e0, e1)``), recorded in one graph and read by
  one set of utilities: see :ref:`results-mixed-batches`.

- **Use the parameter explorer to iterate quickly.** The interactive Gradio UI
  (``python -m viz.param_explorer``, :doc:`/getting-started/parameter-explorer`) shows
  valid-yield and quality stats over a full batch in real time, so shape/yield trade-offs
  are immediately visible.

Per-env randomization and curricula
-----------------------------------

The generators' shape knobs, their per-env course size, the elevation ranges and the
gate bar width are dynamic: drawn per env with ``randomize=`` or moved by a curriculum
with no rebuild. :doc:`/user-guide/curricula` covers them: which fields are dynamic, the
bounds, ``in_effect``, the difficulty axes, the sizing rule for a size range and what a
size range needs from the utilities.

``half_width`` and ``spacing`` coupling
----------------------------------------

``spacing`` controls the arc-length step between adjacent output points.
``half_width`` controls the track width and feeds directly into the XPBD
relaxation geometry.

When ``spacing`` is left at its default (``None``), it is auto-coupled to
``0.6 * half_width``. This is the *relax-friendly* rule of thumb: at that
spacing each bead is roughly 0.6 road-widths apart, which gives the Jacobi
XPBD solver enough room to converge within the default iteration budget.

.. warning::

   A fixed ``spacing`` value does **not** scale with ``half_width``. If you
   change ``half_width`` without adjusting ``spacing``, the per-bead geometry
   shifts and the solver may under-converge. Unless you have a specific reason
   to override, leave ``spacing=None`` and let the auto-coupling handle it.

The :term:`N_max` buffer must be large enough to hold the longest track in the
batch (at its largest ``size``: :ref:`tuning-size`). The formula for the required size
is::

    N_max >= floor(max_perimeter / spacing) + 1

A track whose real point count exceeds ``N_max`` is truncated: its closing segment
would span the dropped arc, so the pipeline flags it (``Track.truncated[e] = 1``) and
fails it (``Track.valid[e] = 0``). At the library
defaults (``half_width=0.1``, auto ``spacing=0.06``) the default ``N_max=384``
gives ample headroom (max ~141 points per track; full coverage down to
``half_width≈0.05``, ~281 max). For a thin-track
regime such as ``half_width=0.03`` (auto ``spacing=0.018``, ~468 max points per
track), raise ``N_max`` to at least 512 to avoid truncation-driven invalids.

Summary of the coupling rules:

- ``spacing = None`` → auto ``0.6 * half_width``.
- ``N_max`` must satisfy ``N_max >= max_perimeter / spacing + 1``; raise it when
  you lower ``half_width`` or ``spacing``.
- ``scale`` controls the track coordinate range; doubling ``scale`` roughly
  doubles perimeters, so ``N_max`` scales proportionally. The per-env ``size`` does the
  same per env: size ``N_max`` (and ``num_points``) for its largest value.
- Ingested centerlines (``generator="external"``) are in your data's units, not
  ``scale``'s: set ``half_width``, ``spacing`` and ``N_max`` for them explicitly.
  ``load_centerlines`` checks ``N_max >= perimeter / spacing + 1`` per env before
  uploading (:ref:`ingesting-centerlines`).

Relaxation knobs
----------------

The XPBD solve has three classes of knobs.

**Iteration count and solver constraints**

``relax_iters`` (default 50) is the number of Jacobi sweeps. Each sweep
applies three corrections to every bead:

1. Non-neighbour separation — pushes beads that are too close apart.
2. Edge spacing — restores the constant bead-to-bead rest length.
3. Bending/radius guard — removes sharp kinks introduced by separation pushes.

**Chebyshev acceleration**

``relax_accel`` (default ``"chebyshev"``) accelerates the Jacobi sweeps with
the Chebyshev semi-iterative method: after ``relax_cheby_start`` (default 8)
plain warmup sweeps, each sweep blends the Jacobi update with the current and
previous iterates using a precomputed omega schedule driven by
``relax_cheby_rho`` (default 0.98) and damped by ``relax_cheby_gamma``
(default 0.9). This is why the default sweep count is 50 rather than 150.
``relax_accel="none"`` restores plain Jacobi. The defaults are a measured
operating point — ``relax_cheby_rho >= 0.99`` or ``relax_cheby_start <= 5``
were measured to *reduce* yield. For the full account (mechanics, measured
tables, and dead ends) see :doc:`/relaxation/solver` and
:doc:`/relaxation/convergence`.

``relax_margin`` (default 0.15) expands the separation target beyond the exact
road diameter:

.. code-block:: text

    target = 2 * half_width * (1 + relax_margin)

A 15 % margin leaves room for the downstream thickness validity gate (the gate
checks that minimum thickness is at least ``(1 - relax_tol) * half_width``).
Raising it buys yield but was measured to cost 20–40 % spacing-RMS quality —
prefer raising ``relax_iters`` instead. Decreasing it tightens the packing but
produces more near-threshold invalids; see the gate-coupling discussion in
:doc:`/relaxation/constraints`.

``relax_tol`` (default 0.02) is the validity thickness tolerance: the minimum
accepted per-point half-width is ``(1 - relax_tol) * half_width``.

**Smoothing tail**

After the main sweeps, a short post-solve smoothing tail runs:
``relax_smooth_passes`` (default 5) shrink-free Taubin smoothing passes followed
by ``relax_smooth_spacing_iters`` (default 10) spacing-only polish sweeps. This
strips the fine sub-``R_min`` curvature noise the bending guard's deadband cannot
see, cutting curvature-difference RMS by roughly a third with no validity loss.
Set both to 0 to disable the tail. For the mechanics (Taubin ``lambda``/``mu``,
why a plain Laplacian is *not* substituted, and the graph-capture note) see
:doc:`/relaxation/solver`.

**Per-constraint relaxation factors**

Three scale factors control how aggressively each correction is applied per
sweep:

- ``relax_sep_relax`` (default 1.0) — separation correction scale. Reduce
  toward 0 to soften separation enforcement (useful if the solver oscillates).
- ``relax_spc_relax`` (default 1.0) — edge-spacing correction scale.
- ``relax_bend_relax`` (default 1.5) — bending/radius-guard correction scale.
  The default slightly over-corrects to quickly remove sharp corners; reduce if
  the solve oscillates.

**Neighbour exclusion band**

``relax_band`` (default ``None``) sets the bead-index exclusion band for
separation: beads within band positions of each other along the chain are not
separated. ``None`` auto-computes ``round(2 * half_width / L0)`` per track,
where ``L0`` is the rest edge length. Override it only if you observe the
road trying to push adjacent beads apart — and never *widen* it to save
separation work: auto-band + 1 was measured to collapse defaults-regime yield
from 0.99 to 0.74 (see the gate coupling in :doc:`/relaxation/constraints`).

Separation-cache throughput recipe
-----------------------------------

The dominant cost in the XPBD solve is the non-neighbour separation scan, which
is ``O(count[e]^2)`` per track per sweep in the dense baseline. Three advanced
knobs expose a graph-capturable broadphase cache:

.. list-table:: Separation-cache knobs
   :header-rows: 1

   * - Setting
     - Meaning
     - Default
   * - ``relax_sep_every``
     - Broadphase refresh interval *K*. Without a cache this is a naive skip
       cadence; with cache enabled it rebuilds candidates every *K* sweeps.
       Keep it well below ``relax_iters`` (see the warning below).
     - ``20``
   * - ``relax_sep_cache_slots``
     - Fixed candidate capacity per bead. ``0`` disables caching. Larger values
       use more memory and narrowphase work but reduce candidate-overflow risk.
     - ``16``
   * - ``relax_sep_cache_skin``
     - Extra broadphase radius as a fraction of ``target``: cache radius is
       ``target * (1 + skin)``. The exact separation push is still applied only
       for ``dist < target``.
     - ``0.5``

Cached mode is active when ``relax_sep_cache_slots > 0`` **and**
``relax_sep_every > 1``. In cached mode, ``_build_sep_cache_kernel`` rebuilds a
fixed-size per-bead candidate list every *K* sweeps using the broadphase radius.
``_step_cached_kernel`` then runs every sweep, re-testing each cached candidate
with the exact narrowphase ``dist < target`` before applying the push. This is
fundamentally different from the naive cadence path (``relax_sep_cache_slots=0,
relax_sep_every > 1``), which simply skips separation on intermediate sweeps.

``skin=0.0`` stores only pairs already inside the exact separation target at
refresh time. Positive skin retains near-pairs that may enter contact during a
long refresh interval, which is safer for large *K* but slower.

.. warning::

   The candidate cache is built at sweep 0 and refreshed only at multiples of
   ``relax_sep_every``. A run shorter than ``relax_sep_every`` sweeps never
   refreshes it mid-solve and silently misses contacts that emerge during the
   run — with ``relax_sep_every=40`` this produced a measured yield cliff
   between 40 and 50 sweeps. That is why the default is 20 for the default 50
   sweeps. If you shorten ``relax_iters``, keep ``relax_sep_every`` comfortably
   below it. See :doc:`/relaxation/convergence`.

A high-throughput configuration from an ``E=8192``, ``half_width=0.03``,
``relax_iters=150`` CUDA graph benchmark (the long 150-sweep budget is what makes
``relax_sep_every=40`` safe there):

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   spec = presets.track(num_envs=E, device=device, half_width=0.03, n_max=512,
                        relax=("xpbd", {"relax_iters": 150,
                                        "relax_sep_every": 40,
                                        "relax_sep_cache_slots": 16,
                                        "relax_sep_cache_skin": 0.0}))
   gen = TrackGenerator(spec)     # TrackGenConfig(relax_iters=150, relax_sep_every=40, ...)

On the GPU it was measured on when the cache was introduced, this ran in approximately
**0.066 s** per graph replay versus **0.366 s** for the dense baseline
(``relax_sep_cache_slots=0``), with effectively unchanged validity in the checked runs.
No benchmark script regenerates these two numbers: measure your own regime.

.. note::

   Keep the dense baseline (``relax_sep_cache_slots=0``) for maximum
   conservatism. Use the cache knobs when throughput matters and validate yield
   in your target regime before committing to the cached config.

Per-generator tuning tips
--------------------------

Bezier (``generator="bezier"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``rad`` (default 0.4) is the primary curviness dial — the Bezier handle
  length as a fraction of each segment chord. Higher values produce rounder
  corners; very high values combined with a small ``handle_clamp_frac`` can
  introduce self-crossings that fall back to the polygon rescue path.
- ``handle_clamp_frac`` (default 0.4) prevents handles from overshooting nearby
  corners. Setting it equal to ``rad`` (the default) means the clamp only trims
  genuine overshoot; setting it below ``rad`` makes every handle respect the
  shorter adjacent edge.
- ``min_num_points`` / ``max_num_points`` (default 9–13) control the corner
  count per env. A wider range increases diversity; many corners with high
  ``rad`` can push more tracks into the polygon fallback. ``min_num_points`` is
  per env (``int32``): a curriculum raises it toward ``max_num_points``
  (:ref:`bezier-corner-count`).
- To let each env draw its own style, randomize ``rad``, ``handle_clamp_frac`` and
  ``size`` (``randomize={"generator.rad": (0.1, 0.4), "generator.size": (0.8, 1.2)}``;
  keep ``rad``'s range at or below the clamp, the lowest clamp draw if you draw it too).
  :ref:`bezier-style-vs-randomize` describes per-env style randomization.

Hull (``generator="hull"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``hull_displacement`` (default 0.15) is the main shape dial — the maximum
  radial midpoint displacement as a fraction of the centroid-to-midpoint
  distance. At 0.0 the method collapses toward a plain angle-sorted loop; larger
  values (typical range 0.1–0.4) create stronger lobes and straights, but can
  increase the downstream relaxation burden.

Polar (``generator="polar"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``polar_num_knots`` (default 12) controls how many frequency components the
  spline has. More knots add higher-frequency features.
- ``polar_radial_jitter`` (default 0.60) is the fractional radial jitter per
  knot. Larger values produce more eccentric shapes.
- ``polar_angular_jitter`` (default 0.30) adds angular irregularity. It is
  bounded to ``[0, 0.45]`` so knots stay in sorted angular order
  (``polar_radial_jitter`` to ``[0, 0.85]``): a value outside raises ``ValueError``.
- The polar generator has no local fallback: any rare bad geometry is handled
  by the shared post-relaxation validity gate, so unusually high jitter at low
  ``relax_iters`` may reduce yield.

Voronoi (``generator="voronoi"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``voronoi_site_layout`` (default ``"void_ring"``) selects the site distribution.
  ``"void_ring"`` biases sites into an annular band with a center void, which
  tends to produce usable closed loops. ``"ring"`` fills the bounding box
  uniformly (not an annular band despite the name). ``"clustered"`` and
  ``"mixed"`` create more complex layouts.
- ``voronoi_control_points`` (default 18) sets how many angular anchors are
  selected from the site field. More control points add local features and can
  raise the relaxation burden.
- ``voronoi_radial_variation`` (default 0.62) controls how eccentric the anchor
  ring is. At 0 all anchors are placed at a uniform annulus radius.
- ``voronoi_angular_jitter`` (default 0.08, radians) breaks the rotational regularity.
  Keep it below about ``0.3`` for full yield: it drops to 0.71-0.85 at ``0.5`` and near
  zero from ``1.2``.
- Higher ``voronoi_num_sites`` enriches the anchor snap targets but increases
  the all-sites scan cost per bead.

Checkpoint (``generator="checkpoint"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``checkpoint_count`` (default 12) controls the number of radial checkpoints.
  More checkpoints produce wavier, chicane-rich tracks; fewer produce calmer,
  rounder shapes.
- ``checkpoint_best_of_k`` (default 4) generates *K* decorrelated candidates
  per env and keeps the one with the fewest self-intersections. Increasing *K*
  reduces pre-relaxation self-crossings at the cost of extra generation work.
- ``checkpoint_turn_rate`` (default 0.42 rad/step) bounds the path curvature.
  Too large allows kinks; too small prevents the path from tracking tight inlets.
- ``checkpoint_clip_fallback`` (default ``False``) opts into a single-crossing
  clip rescue. Leave it off unless you are seeing a meaningful fraction of
  crossing survivors that the validity gate is discarding.
