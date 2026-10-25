Voronoi Generator
=================

.. note::

   Sample figures, seed-specific ranges and measured yields on this page are historical
   measurements from before the 0.3 RNG switch. The algorithms and parameter contracts
   describe current code; remeasure current seeds (:ref:`upgrading-0.3-rng`).

The ``voronoi`` generator produces compact, feature-rich closed centerlines by snapping a
cycle of angular anchor targets onto a randomly placed field of cell sites, rounding the
resulting polyline, and arc-resampling it into the standard pipeline buffer.  It is the
production distillation of the Voronoi / random-geometric prototype: the useful "right number
of cells" control is retained through the fixed site field, while exact Delaunay / Voronoi
ridge traversal and dynamic cycle extraction — which are incompatible with the fixed-shape
CUDA-graph contract — are replaced by a static angular-snap cycle that is allocation-free and
deterministic in ``(seed, config)`` (:term:`seed`).  Of the five sampling generators,
``voronoi`` yields the highest typical compactness (:doc:`benchmarks`) and effectively
eliminates pre-relax self-intersections without a generation retry loop. It writes one
closed centerline per :term:`env` at every :term:`generation`, and its shape knobs are
per-env :term:`dynamic fields <dynamic field>` (`Per-env (dynamic) parameters`_).

.. figure:: ../assets/generator-voronoi.png
   :alt: Sample tracks produced by the voronoi generator
   :align: center

   Sample tracks produced by the ``voronoi`` generator across several seeds.


How It Works
------------

1. **Draw the site field.**
   ``_sample_sites_k`` draws ``S = voronoi_num_sites`` points per environment with
   ``ops.rng.init`` and the ``generator.voronoi/sites`` salt, using one of
   four ``voronoi_site_layout`` modes:

   - ``ring`` — uniform box fill.  Each site is drawn independently and uniformly within the
     full bounding box ``[-L/2, L/2]²`` (``L = box_size = 2.0``).  Despite its name, this
     mode does *not* produce an annular band; it is the flat, unstructured baseline.
     Source: ``_sample_sites_k``, lines 103–106 — the ``_LAYOUT_RING`` branch leaves
     ``use_annulus = 0`` and falls through to the uniform ``x, y`` draw.

   - ``void_ring`` (default) — area-uniform annular band with a center void.  Every site is
     placed in the annulus ``[0.14·L, 0.52·L]`` using the area-preserving radius formula
     ``r = sqrt(r_min² + rand·(r_max² - r_min²))``, leaving the center of the box empty.
     Source: ``_sample_sites_k``, lines 75–76 (``use_annulus = 1``) and 97–102 (annulus
     draw).

   - ``clustered`` — Gaussian-style clusters around six evenly spaced angular positions.
     Each point is assigned to a random cluster whose center lies at approximately
     ``0.24–0.34·L`` from the origin; points are scattered with noise radius
     ``~0.025–0.105·L``.  Roughly 22 % of clustered points are redirected to the annulus
     draw instead (source: ``_sample_sites_k``, line 90, ``wp.randf(state) < 0.22``).

   - ``mixed`` — blends the two placement modes per site: each site independently uses the
     annulus draw with probability 0.65 and the uniform box fill with probability 0.35
     (source: ``_sample_sites_k``, line 78, ``wp.randf(state) < 0.65``).

2. **Select anchor sites.**
   ``_select_anchor_sites_dyn_k`` uses the ``generator.voronoi/anchors`` salt
   (:ref:`ops-rng-keying`) and draws ``K = voronoi_control_points`` angular sector
   targets, with env ``e``'s ``voronoi_radial_variation`` and ``voronoi_angular_jitter``
   read from the per-env parameter arrays.
   A global random rotation ``ρ`` is applied; sector ``i`` is centered at angle
   ``ρ + (2π/K)·i`` and jittered by ``εᵢ ∈ [-j, +j]`` where ``j = voronoi_angular_jitter``.
   Each target also has a radial component: a multi-harmonic profile ``φ(θ)`` (see Math
   section) modulates the target radius, scaled by ``voronoi_radial_variation``.  The anchor
   is then chosen as the site that minimises a combined angle + radius cost, with a large
   penalty (1 000) for already-used sites — an O(S) scan that keeps the selected cycle free
   of duplicate anchors.

3. **Chaikin subdivision.**
   ``_chaikin_once_k`` applies one round of Chaikin corner-cutting to the ``K`` selected
   anchors, converting each edge ``(pᵢ, pᵢ₊₁)`` into two new vertices at the 1/4 and 3/4
   points.  This doubles the polygon to ``AP = 2K`` augmented points and softens the raw
   anchor-snap polygon before spline evaluation.

4. **Closed Catmull-Rom spline.**
   ``ops.closed_catmull_rom`` evaluates a closed uniform Catmull-Rom spline over the ``AP``
   augmented vertices, producing a dense ``M = AP × num_points_per_segment``-point loop per
   environment.

5. **Arc-resample and normalize.**
   ``ops.arc_resample`` arc-resamples the dense loop to ``num_points`` uniformly spaced
   points.  ``ops.normalize_extent`` then centers the environment's bounding box and
   isotropically rescales the longest dimension to ``scale × 1.44``, matching the shared
   target extent of the other generators, times the env's ``size[e]``
   (``ops.normalize_extent(..., size=...)``).

6. **Fallback for self-crossing smooth loops.**
   ``ops.self_intersections`` counts proper crossings on the ``num_points``-point loop.
   For environments where the smooth loop self-intersects,
   ``ops.closed_polyline(..., active=crossers)`` emits the straight Chaikin-augmented
   (``AP``-vertex) polygon by linear interpolation, and
   ``ops.arc_resample(..., active=crossers)`` overwrites those rows with an arc-resampled
   version of that polygon.  A second ``ops.normalize_extent`` pass re-applies the bbox
   rescale (to the same per-env target).
   Non-crossing rows are left untouched.


Math
----

**Angular anchor targets.**
Anchor ``i`` has target angle

.. math::

   \theta_i = \rho + \frac{2\pi}{K}\,i + \varepsilon_i,
   \quad \varepsilon_i \sim \mathcal{U}(-j,\, j),
   \quad j = \texttt{voronoi\_angular\_jitter},

where ``ρ`` is a per-environment random offset drawn once and ``K = voronoi_control_points``.

**Radial variation profile.**
The target radius for anchor ``i`` is

.. math::

   r_i = \operatorname{clip}\!\Bigl(
       0.34\,L\,\bigl(1 + A\,\varphi(\theta_i)\bigr),\;
       0.16\,L,\; 0.52\,L
   \Bigr),

where ``L = box_size = 2.0``,

.. math::

   \varphi(\theta) = \frac{1}{1.5}\Bigl[
       0.62\sin(2\theta + \phi_0)
       + 0.44\cos(3\theta + \phi_1)
       + 0.26\sin(5\theta + \phi_2)
       + 0.18\cos(7\theta + \phi_3)
   \Bigr],

and the phase offsets ``φ₀, φ₁, φ₂, φ₃`` are independent per-environment random angles.
The amplitude is clamped:

.. math::

   A = \operatorname{clip}(\texttt{voronoi\_radial\_variation} \times b,\; 0,\; 0.85),

with layout boost ``b = 1.10`` for ``void_ring``, ``1.16`` for ``clustered``, ``1.06`` for
``mixed``, and ``1.00`` for ``ring``.

**Anchor snap cost.**
The nearest-unused-site snap minimises

.. math::

   c(s) = \frac{|\Delta\theta_s|}{\sigma}
          + \frac{|r_s - r_i|}{0.13\,L}
          + 1000\cdot\mathbb{1}[\text{used}(s)],
   \quad \sigma = \frac{2\pi}{K},

over all ``S`` sites, where ``Δθₛ`` is the signed angular difference between site ``s`` and
the target angle (wrapped to ``(-π, π]``).

**Chaikin corner-cutting.**
Each edge ``(pᵢ, pᵢ₊₁)`` of the ``K``-anchor polygon is replaced by two vertices:

.. math::

   q_{2i}   = \tfrac{3}{4}\,p_i + \tfrac{1}{4}\,p_{i+1}, \qquad
   q_{2i+1} = \tfrac{1}{4}\,p_i + \tfrac{3}{4}\,p_{i+1},

producing the ``AP = 2K`` augmented vertices.

**Closed uniform Catmull-Rom.**
The dense loop is evaluated as

.. math::

   P(t) = \tfrac{1}{2}
   \begin{bmatrix} 1 & t & t^2 & t^3 \end{bmatrix}
   \begin{bmatrix}
    0 &  2 &  0 &  0 \\
   -1 &  0 &  1 &  0 \\
    2 & -5 &  4 & -1 \\
   -1 &  3 & -3 &  1
   \end{bmatrix}
   \begin{bmatrix} P_{i-1} \\ P_i \\ P_{i+1} \\ P_{i+2} \end{bmatrix},

where indices wrap modulo ``AP``.


Parameters
----------

The following ``TrackGenConfig`` fields are owned by the ``voronoi`` generator.  All other
fields (``num_points``, ``num_points_per_segment``, ``scale``, ``half_width``, etc.) are
shared pipeline parameters described in the configuration reference.

The generator-specific fields below are also the fields of ``VoronoiParams``, the
generator's Params class (core fields such as ``scale`` are not): set them flat or
through ``generator_params`` (see :ref:`builtin-generator-params`). Their rules are
checked when the ``TrackGenerator`` is constructed, and only for
``generator="voronoi"``. ``VoronoiParams`` also holds ``num_points_per_segment``.

``voronoi_num_sites`` (``S``, default 256)
   Number of cell sites drawn per environment.  Higher values make the anchor-snap richer
   and improve coverage of the site field, but scale the per-anchor O(S) cost scan linearly.

``voronoi_site_layout`` (default ``"void_ring"``)
   Spatial distribution of sites.  One of ``"ring"`` (uniform box fill),
   ``"void_ring"`` (annular band with a center void), ``"clustered"`` (six Gaussian-style
   clusters), or ``"mixed"`` (65 % annulus / 35 % uniform, per site).  The layout also
   adjusts the amplitude boost applied to ``voronoi_radial_variation``.

``voronoi_control_points`` (``K``, default 18)
   Number of angular anchors selected from the site field.  More anchors add local features
   and tighter curvature, but also increase the smoothing and relaxation burden.

``voronoi_radial_variation`` (default 0.62)
   Amplitude of the per-anchor radius modulation.  At 0 all anchor targets sit at the same
   base radius (0.34·L); at larger values the multi-harmonic profile drives anchors to vary
   between 0.16·L and 0.52·L, producing more irregular, non-circular layouts. The kernel
   uses ``min(voronoi_radial_variation * boost, 0.85)`` with a per-layout ``boost >= 1``,
   so the effect saturates at ``0.85 / boost`` and nothing changes above ``0.85``:
   dynamic, in ``[0, 0.85]`` (a larger value raises ``ValueError`` at construction).

``voronoi_angular_jitter`` (default 0.08)
   Maximum angular perturbation (in radians) applied to each sector target.  Small values
   keep anchors close to their ideal evenly-spaced sectors; larger values break the rotational
   regularity. Dynamic, in ``[0, pi]``; the useful range is much smaller (below).

``size`` (default 1.0)
   Per-env size multiplier: both normalizations scale env ``e``'s longest bounding-box
   side to ``f32(scale * 1.44) * size[e]``. Dynamic, in ``[0.1, 10]`` (below).

.. _voronoi-dynamic:

Per-env (dynamic) parameters
----------------------------

``voronoi_radial_variation``, ``voronoi_angular_jitter`` and ``size`` are **dynamic**
fields of ``VoronoiParams`` (the first two also of ``VoronoiGateParams``: the voronoi gate
generator selects its anchors with the same kernel): per-env ``[E]`` ``float32`` arrays,
initialised from the configured values. Draw them per env with ``presets.track(generator="voronoi",
randomize={"generator.voronoi_radial_variation": (lo, hi)})`` (``presets.gates`` takes
the same keys), write them with ``gen.params["generator"].set(...)`` or from your own
graph (a curriculum): see :ref:`tuning-per-env`. A scalar outside the bounds raises
``ValueError``; an array value outside them is clamped to them when the kernel reads it.

.. list-table::
   :header-rows: 1
   :widths: 26 9 11 20 21 13

   * - Field (``randomize=`` key)
     - Default
     - Bounds
     - Harder / more varied when
     - Effective range
     - Yield at lo / default / hi
   * - ``voronoi_radial_variation`` (``generator.voronoi_radial_variation``)
     - ``0.62``
     - ``[0, 0.85]``
     - higher: stronger radial modulation, less circular layouts
     - saturates at ``0.85 / boost`` (``0.85`` for ``"ring"``, about ``0.77`` for the
       default ``"void_ring"``)
     - 1.00 / 1.00 / 1.00
   * - ``voronoi_angular_jitter`` (``generator.voronoi_angular_jitter``)
     - ``0.08``
     - ``[0, pi]``
     - higher: more irregular anchor spacing
     - **about** ``[0, 0.5]``: ``pi`` only bounds what is legal (``±pi`` spans the
       circle), the yield collapses well before it
     - 1.00 / 1.00 / 0.00
   * - ``size`` (``generator.size``)
     - ``1.0``
     - ``[0.1, 10]``
     - higher: a longer course relative to the (absolute) road width
     - about ``[0.5, 4.6]`` at the default ``n_max`` (384): from about ``4.6`` the
       longest tracks of a large batch truncate
     - 0.996 / 1.00 / 1.00 at 0.5 / 1 / 2 (1.00 at 3)

*Yield* is the fraction of valid tracks with the default pipeline, ``E = 128``, seeds
``0..127`` (CPU), with the knob set to the value for every env (``size``: ``E = 256``,
seeds ``7919 * e + 13``). For ``voronoi_angular_jitter`` it falls fast past ``0.3``
(every layout):

.. list-table::
   :header-rows: 1
   :widths: 28 9 9 9 9 9 9 9 9

   * - ``voronoi_angular_jitter``
     - 0.08
     - 0.2
     - 0.3
     - 0.5
     - 0.8
     - 1.2
     - 1.6
     - pi
   * - yield (the four layouts)
     - 1.00
     - 1.00
     - 0.97-0.99
     - 0.71-0.85
     - 0.29-0.38
     - 0.03-0.05
     - 0.00-0.01
     - 0.00

Keep a randomized or curriculum range within about ``[0, 0.5]``, and check the yield at
both ends of it.

**Size** scales the normalization target of both normalizations (step 5 and the
fallback's): env ``e``'s longest bounding-box side is ``f32(scale * 1.44) * size[e]``,
within one ``float32`` rounding of ``scale = s`` at ``size = s``, and bitwise unchanged
at ``size = 1``. It scales the plan view only (road width, spacing and z stay absolute).
The longest default ``voronoi`` loop of 8192 envs is 5.01 (p99 4.78) at ``scale = 1``:
with the default ``n_max = 384`` and spacing ``0.06`` truncation (a failed env) starts
at about ``size = 4.6``. Size ``n_max`` and
``num_points`` for the largest size with the rule of :ref:`tuning-size`. On **gates**
the size is the ``normalize.extent`` stage's (key ``normalize.size``), as for ``polar``
(:ref:`polar-dynamic`).

**Static knobs, and why.** ``voronoi_num_sites`` and ``voronoi_control_points`` (size
the site, anchor and dense buffers and the launch dimensions), ``voronoi_site_layout``
(an enum the kernels branch on; categorical draws are not planned yet),
``num_points_per_segment`` (sizes the dense buffer) and the core ``scale`` (a host
product with the target extent; per-env: ``size``).


What Makes It Distinct
----------------------

``voronoi`` is the only generator built around a pre-placed site field.  The angular-snap
construction produces tracks anchored to a concrete spatial texture rather than drawn from
pure random Cartesian or polar coordinates, which gives the family a characteristic
"cell-like" compactness absent from the other four generators.

Key contrasts:

- **Highest compactness** of the sampling generators (:doc:`benchmarks`).  The
  ``void_ring`` default biases sites into an annular
  band and selects anchors around the full perimeter, consistently producing compact,
  looping layouts with high area-to-perimeter ratios.

- **Zero pre-relax self-intersections in practice.**  The site field and the annular bias of
  the default layout keep anchors at meaningful separations.  The Chaikin + Catmull-Rom
  smoothing pass almost never produces a self-crossing loop, so the polygon fallback
  path is rarely triggered.

- **Multi-scale control.**  ``voronoi_num_sites`` controls global cell density; ``voronoi_control_points``
  controls route complexity; ``voronoi_radial_variation`` and ``voronoi_angular_jitter``
  control irregularity within the route.  No other generator exposes a site-density knob
  independent of its curve-complexity knob.

- **No style-sampling.** Unlike ``bezier``, the Voronoi generator does not support
  per-environment parameter sampling from a range; its randomness comes entirely from the
  seed-driven site draw and anchor selection.


Fallback and Validity
---------------------

If the Chaikin-smoothed Catmull-Rom loop for an environment self-intersects after
arc-resampling, the generator falls back to the corresponding Chaikin-augmented
``AP``-vertex straight polygon, arc-resampled via ``ops.arc_resample(..., active=...)``.  Only
the crossing rows are overwritten; non-crossing rows are untouched.  Both the smooth-loop
and polygon-fallback paths write ``out_valid_wp = 1`` for every environment.

Final geometric validity — turning number close to 2π, minimum track thickness, absence of
NaNs, and width floor — is decided after the shared resample → XPBD relax → inflate
pipeline, not by the generator itself.  The XPBD relaxation pass will re-round the
flat-polygon fallback into a smoother loop, just as it does for the ``bezier`` and ``hull``
fallback paths.
