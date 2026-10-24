Tuning guide
============

This page collects practical tuning advice for ``TrackGenConfig``. For the
complete parameter descriptions, see the :doc:`Configuration reference <reference>`.

Yield vs. diversity
-------------------

*Yield* is the fraction of environments that pass the post-relaxation validity
gate. *Diversity* is how varied the resulting track shapes are across a batch.
These two goals pull in opposite directions: the settings that reliably produce
valid tracks tend to constrain the shape space.

A few principles:

- **Raise ``relax_iters`` first.** The default of 50 Chebyshev-accelerated
  sweeps is calibrated for approximately 99.9 % yield at the library defaults
  (matching what 150 plain-Jacobi sweeps used to need). If yield is low, the
  most conservative fix is to increase ``relax_iters``; cost is linear in
  sweeps and it does not restrict shapes. See
  :doc:`/relaxation/convergence` for the measured yield-vs-sweeps curves.

- **Widen the generator's shape range for diversity.** Draw the generator's dynamic
  knobs per env with ``randomize=`` (`Per-env parameters: randomization and
  curricula`_): ``rad`` / ``handle_clamp_frac`` for the Bezier generator,
  ``polar_radial_jitter`` for polar, ``checkpoint_radius_min_frac`` for checkpoint, and
  so on. For the Bezier generator, also widen ``[min_num_points, max_num_points]``
  (and use ``style_sampling=True`` with ``scale_range`` for per-env size). For the
  checkpoint generator, increase ``checkpoint_count``; for polar, ``polar_num_knots``.
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
  (``.venv/bin/python -m viz.param_explorer``) shows valid-yield and quality stats over a
  full batch in real time, so shape/yield trade-offs are immediately visible.

.. _tuning-per-env:

Per-env parameters: randomization and curricula
-----------------------------------------------

A **dynamic** parameter is a per-env ``[E]`` ``float32`` array the kernels read at every
generation. You can draw it per env with ``randomize=``, or write it yourself (a
curriculum), with no rebuild and no re-capture. The dynamic fields are:

- the built-in generators' shape knobs: ``rad`` and ``handle_clamp_frac`` (``bezier``),
  ``hull_displacement`` (``hull``), ``polar_radial_jitter`` and
  ``polar_angular_jitter`` (``polar``), ``voronoi_radial_variation`` and
  ``voronoi_angular_jitter`` (``voronoi``), ``checkpoint_radius_min_frac``,
  ``checkpoint_angle_jitter`` and ``checkpoint_turn_rate`` (``checkpoint``). The polar,
  voronoi and checkpoint **gate** generators have the same ring knobs (not
  ``checkpoint_turn_rate``: they do not steer). Each generator page gives their bounds,
  effective range and measured yield (:ref:`generators-per-env`);
- the elevation profiles' ``z_min`` and ``z_max`` (``uniform``, ``random_walk``,
  ``noise``), ``z_max_step`` (``random_walk``) and ``z_noise_amplitude`` (``noise``), on
  tracks, on gates and on the centerline before the stations alike;
- ``width.from_channel``'s ``scale``, ``lo`` and ``hi``, and every field a plugin
  declares with ``plugin.dynamic``.

Everything else is static (point and knot counts, ``half_width``, ``spacing``,
``edgy``, ``scale``, ``z_base``, the relax and validity tolerances, the ``repulsive``
knobs): build a new generator to change it. Naming a static field in ``randomize=``
raises ``PlanError`` listing the fields of the pipeline that can be drawn.

**Randomize.** ``presets.track(randomize={"<stage id>.<field>": (lo, hi)})`` draws each
env's value uniformly in ``[lo, hi)`` at every generation (:ref:`config-randomize`). The
generator's stage id is ``"generator"``, the elevation's ``"elevation"``, in
``presets.track`` and ``presets.gates`` alike:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   spec = presets.track(
       num_envs=4096, device="cuda:0", generator="polar",
       elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}),
       randomize={"generator.polar_radial_jitter": (0.3, 0.85),
                  "elevation.z_noise_amplitude": (0.0, 0.3)},
   )
   gen = TrackGenerator(spec)
   gen.params["generator"]["polar_radial_jitter"]   # [E]: what each env was drawn

For a pipeline built from a ``TrackGenConfig``, add the randomizers to its spec:
``TrackGenerator(legacy.to_pipeline_spec(cfg).with_randomize({...}), rng)``; from a
``GateGenConfig``:
``GateGenerator(legacy.to_gate_pipeline_spec(gcfg).with_randomize({...}), rng)``
(``PipelineSpec.with_randomize`` is what the presets call; the configs have no
``randomize`` field). A field the spec already randomizes is refused: to change its
range for a curriculum phase, ``spec.replace(...)`` that randomizer or write its live
``lo`` / ``hi`` (below).

**Curricula.** ``gen.params[stage][field]`` is the live ``[E]`` array of a dynamic field,
at a stable address. Write it with ``gen.params[stage].set(field, value)`` (a scalar for
every env, or an ``[E]`` ``wp.float32`` array), or from a kernel recorded in your own
graph before ``generate()``: the next generation reads it. A field you write must not be
drawn by a randomizer (below), so this curriculum's generator, unlike the one above,
draws only the hill amplitude and leaves ``polar_radial_jitter`` and ``z_max`` to the
curriculum kernel:

.. code-block:: python

   import warp as wp

   spec = presets.track(
       num_envs=E, device="cuda:0", generator="polar",
       elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}),
       randomize={"elevation.z_noise_amplitude": (0.0, 0.3)},   # not the jitter
   )
   gen = TrackGenerator(spec)
   rjit = gen.params["generator"]["polar_radial_jitter"]     # [E] float32, stable
   z_max = gen.params["elevation"]["z_max"]

   with wp.ScopedCapture(stream=s) as capture:
       wp.launch(difficulty_k, dim=E, inputs=[success_rate, rjit, z_max], stream=s)
       gen.generate(advance=True, stream=s)
   wp.capture_launch(capture.graph, stream=s)   # each replay reads the new values

A field drawn by a randomizer belongs to it: ``set`` raises and a direct write is
overwritten at the next generation (with the generator of the randomize example above,
the kernel's ``polar_radial_jitter`` write would be lost). Move the draw instead, by
writing the randomizer's bounds,
``gen.params["randomize.generator.polar_radial_jitter"]["hi"]`` (a curriculum on a
distribution; keep it in ``[lo, 0.85]``).

**One rule for bounds.** A dynamic built-in knob declares its bounds (each generator
page lists them). A **scalar** is checked on the host: a value outside the bounds raises
``ValueError``, at construction, from ``set``, and for a randomizer's initial range
(``PlanError``). The comparison is made in ``float32``, the precision the array holds, so
a value read back from an array is always accepted. An **array** value (an ``[E]``
``set``, your own kernel, a randomizer's draw from bounds you wrote) is not checked: the
kernel clamps it to the bounds when it reads it (a NaN passes through, and an unbounded
side, such as ``z_min``'s or ``handle_clamp_frac``'s upper one, is not clamped). So a
curriculum may overshoot a bound without failing; the excess piles onto the bound.

**Check the yield at both ends of every range.** Many knobs lose yield, or stop
mattering, well inside their bounds: ``rad`` has no effect above ``handle_clamp_frac``
(drawing both independently leaves ``rad`` inert in every env where it lands at or
above the clamp: keep ``hi(rad) <= lo(handle_clamp_frac)``; a ``generator.rad``
randomizer whose ``hi`` exceeds every value the clamp can take, its drawn ``hi`` or its
value, emits a ``UserWarning`` when the pipeline is built),
``voronoi_angular_jitter`` is legal up to ``pi`` but its yield is at most 0.85 from
``0.5`` (0.71-0.85 by layout) and near zero from ``1.2``, ``checkpoint_radius_min_frac`` at ``0.99`` keeps two
thirds of the tracks. The yield-versus-diversity advice above applies to the whole
range, not its middle.

**Elevation: the knob that sets the difficulty depends on the profile.** The window
``[z_min, z_max]`` clips every profile, so:

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - Profile
     - Difficulty knob
     - Window
   * - ``uniform``
     - ``z_max`` (higher: bigger hills)
     - keep ``z_min`` fixed (at the ground) and draw or raise ``z_max``
   * - ``random_walk``
     - ``z_max_step`` (higher: steeper grades)
     - a wide window: at ``z_min == z_max`` (the default ``[0, 0]``) the walk cannot
       move, and ``z_max_step`` has no effect
   * - ``noise``
     - ``z_noise_amplitude`` (higher: bigger hills)
     - a wide window **symmetric around** ``z_base`` (``z_min = -z_max`` at
       ``z_base = 0``): the default ``[0, 0]`` makes the amplitude inert, and ``[0,
       z_max]`` clips the negative half of every hill

When the pipeline is built, an inert window emits a ``UserWarning``: ``z_min == z_max``
with neither drawn, while ``z_max_step`` (``random_walk``) or ``z_noise_amplitude``
(``noise``) can be positive (its configured value, or the ``hi`` of its randomizer). A
window drawn by randomizers is open and does not warn; a window you open only at run
time (``set``) cannot be seen then, so a curriculum starting flat gets the warning.
Randomizers on ``z_min`` and
``z_max`` that could cross (``hi`` of ``z_min`` above ``lo`` of ``z_max``, or one drawn
range crossing the other's fixed value) raise ``PlanError`` when the pipeline is built.
A crossing you write yourself at run time is not refused: ``uniform`` then draws in
``[z_max, z_min]``, ``random_walk`` and ``noise`` return ``z_max`` everywhere (exactly
what the scalar kernels did).

**Masked regeneration.** At the start of a generation the Program copies (*latches*)
the live arrays into the working copies the kernels read, for the envs it regenerates
only. Under ``generate(advance=True, mask=m)`` an env left out keeps the value its
current track was built with, so the live array may already hold a newer value than
that env's track: for bookkeeping (which difficulty is each env running?) keep your own
per-env copy of what you wrote when you regenerated it. A drawn field is not latched:
``gen.params["generator"][field]`` holds the draw each env's track was built with.

**Segmented Programs.** In a Program with a ``HOST`` stage (a ``HOST`` plugin generator
or relax stage), the latch runs at the start of segment ``0``. A write must be enqueued
before segment ``0`` to apply to that generation; a write recorded in a later segment's
graph applies one generation later (:ref:`pipeline-segments`).

**What stays static, and how to vary it anyway.** Mix generators or configurations in
one batch (:ref:`results-mixed-batches`), each writing its own env range of one
``Track``; use ``bezier``'s ``style_sampling=True`` with ``scale_range`` for per-env
size; or write a plugin generator whose knobs are ``plugin.dynamic`` (with declared
``lo`` / ``hi``, :ref:`generator-runtime-params`).

.. _tuning-difficulty-axes:

Difficulty axes and the knobs that reach them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Axis
     - ``randomize=`` key, by generator (or why it cannot be drawn yet)
   * - Curviness, corner sharpness
     - ``generator.rad`` (lower: sharper) and ``generator.handle_clamp_frac`` (bezier);
       ``generator.checkpoint_turn_rate`` (higher: sharper kinks; checkpoint)
   * - Irregularity
     - ``generator.hull_displacement`` (hull); ``generator.polar_angular_jitter``
       (polar); ``generator.voronoi_angular_jitter`` (voronoi, up to about ``0.5``);
       ``generator.checkpoint_angle_jitter`` (checkpoint)
   * - Eccentricity, inlets
     - ``generator.polar_radial_jitter`` (polar); ``generator.voronoi_radial_variation``
       (voronoi); ``generator.checkpoint_radius_min_frac`` (lower: deeper; checkpoint);
       ``generator.hull_displacement`` (pinches; hull)
   * - Hills
     - ``elevation.z_max`` with ``z_min`` fixed (``uniform``);
       ``elevation.z_noise_amplitude`` in a wide symmetric window (``noise``)
   * - Grade
     - ``elevation.z_max_step`` in a wide window (``random_walk``)
   * - Size
     - not reachable in 0.2.0.dev5: a per-env ``generator.size`` is planned (roadmap
       PR13d); today, ``bezier``'s ``style_sampling`` with ``scale_range``
   * - Corner count
     - not reachable in 0.2.0.dev5: a dynamic ``min_num_points`` for ``bezier`` /
       ``hull`` is planned (PR13d); the knot, anchor and checkpoint counts size buffers
   * - Corner asymmetry
     - not reachable in 0.2.0.dev5: ``bezier``'s ``edgy`` (roadmap PR16)
   * - Gate size
     - not reachable in 0.2.0.dev5: ``gate_frames.default``'s ``gate_width`` (PR13d)

The gate pipelines take the same keys for the polar, voronoi and checkpoint ring knobs
and the elevation (``presets.gates(randomize=...)``).

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

The ``N_max`` buffer must be large enough to hold the longest track in the
batch. The formula for the required size is::

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
  doubles perimeters, so ``N_max`` scales proportionally.
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

.. code-block:: python

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
   run — under the old default of 40 this produced a measured yield cliff
   between 40 and 50 sweeps. That is why the default dropped to 20 alongside
   the ``relax_iters`` 150 → 50 change. If you shorten ``relax_iters``, keep
   ``relax_sep_every`` comfortably below it. See
   :doc:`/relaxation/convergence`.

A high-throughput config from the ``E=8192``, ``half_width=0.03``,
``relax_iters=150`` CUDA graph benchmark (note the long 150-sweep budget is
what makes ``relax_sep_every=40`` safe there):

.. code-block:: python

    TrackGenConfig(
        ...,
        relax_sep_every=40,
        relax_sep_cache_slots=16,
        relax_sep_cache_skin=0.0,
    )

On the benchmark machine this ran in approximately **0.066 s** per graph replay
versus **0.366 s** for the dense baseline (``relax_sep_cache_slots=0``), with
effectively unchanged validity in the checked runs.

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
  ``rad`` can push more tracks into the polygon fallback.
- To let each env draw its own style, randomize ``rad`` and ``handle_clamp_frac``
  (``randomize={"generator.rad": (0.1, 0.4)}``; keep ``rad``'s range at or below the
  clamp, the lowest clamp draw if you draw it too) and, for per-env size, enable
  ``style_sampling=True`` with ``scale_range``. ``style_sampling``'s ``rad_range`` and
  ``handle_clamp_frac_range`` still work, but ``randomize=`` supersedes them
  (:ref:`bezier-style-vs-randomize`).

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
