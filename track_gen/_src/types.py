"""Shared dataclasses for the track generation pipeline.

This module imports nothing from the rest of the package (no generators, no
inflation, no track_generator, no rng_utils). It is the shared home for
``TrackGenConfig`` and ``Track`` so that ``warp_pipeline.py`` and the facade can
both import them without a circular import. Warp is a core dependency (``Track``
fields are ``wp.array``).
"""

import math
from dataclasses import dataclass

import warp as wp


@dataclass
class TrackGenConfig:
    """Single configuration object passed to every stage of the pipeline.

    Fields configure the fixed-shape Warp pipeline. Scalar style knobs are used by
    default; opt-in per-env style sampling is available through ``style_sampling`` and
    the ``*_range`` fields for the Bezier generator. Only ``relax_solver="xpbd"`` is
    used by the runtime path; the ``energy_*``, ``tp_*``, and ``smooth_finish*`` fields
    serve the torch oracle backends in ``tests/_oracle/`` and are **not used by
    TrackGenerator**. The Fourier fields (``num_harmonics``, ``decay_p``,
    ``amplitude``, ``num_centerline_samples``) are retained only for compatibility with
    ``track_gen._experimental.fourier`` and older experiment sweeps.

    Attributes
    ----------
    generator : str
        Registered generator name.  One of ``"bezier"`` (default), ``"hull"``,
        ``"polar"``, ``"voronoi"``, or ``"checkpoint"``.  Resolved through
        ``track_gen._src.generator_registry`` at ``TrackGenerator`` construction time.
    device : str
        Warp device string: ``"cpu"`` (GPU-free, for tests/CI) or ``"cuda"``
        (captures a replay graph on first ``generate()`` call).
    num_envs : int
        Number of environments in the batch.  Must be >= 1.  Determines the
        leading dimension of all output arrays.

    min_num_points : int
        Minimum Bezier/hull corner-anchor count per env.  Must be >= 2.  Shared
        by the ``"bezier"`` and ``"hull"`` generators; feeds ``wp.randi`` with the
        range ``[min_num_points, max_num_points]``.
    max_num_points : int
        Maximum corner-anchor count per env.  Must be >= ``min_num_points``.  A
        wider range adds more per-env diversity in corner count.
    num_points_per_segment : int
        Dense Bezier (or Catmull-Rom) samples emitted per corner-to-corner segment
        before arc-resampling to ``num_points``.  Must be >= 2; higher values give
        the arc-resampler a finer cumulative-arc-length grid, reducing interpolation
        error.
    min_point_distance : float
        Minimum grid-cell spacing for corner-position sampling.  Must be in
        ``(0, 0.5]``.  Governs the cell grid as
        ``num_cells = int(1 / (min_point_distance * 2))``.
    min_angle : float
        Minimum interior angle (radians) at constrained Bezier corners; default
        ~0.218 rad (12.5 deg).  Used by the gate-angle validity kernel in
        ``warp_generate.py``; not applied directly to track-centerline generation.
    rad : float
        Cubic-Bezier handle length as a fraction of the segment chord — the primary
        curviness dial.  Typical range ``[0.2, 0.6]``.  Only live when
        ``rad <= handle_clamp_frac``; if ``rad`` exceeds the clamp,
        ``handle_clamp_frac`` binds every segment and ``rad`` has no effect (see
        ``handle_clamp_frac``).
    edgy : float
        Corner-tangent blend weight.  Converted internally as
        ``p = atan(edgy) / pi + 0.5``.  At 0.0 (default), ``p = 0.5`` gives a
        symmetric bisector tangent; larger positive values push ``p`` toward 1,
        making turns asymmetrically sharper (outgoing-edge-dominant).
    scale : float
        Isotropic scale multiplier applied to sampled corner positions.  Scales all
        track coordinates; interacts with ``spacing`` and ``half_width`` through
        the coordinate range.
    handle_clamp_frac : float
        Adaptive Bezier-handle clamp.  Each corner's handle is capped at
        ``handle_clamp_frac * (its shorter incident edge)`` to prevent overshoot
        past a nearby corner.  Set equal to ``rad`` (the default) so the clamp only
        trims genuine overshoot; set below ``rad`` to bind every handle regardless
        of ``rad`` (producing near-polygonal tracks); set very large to disable.
        Tracks that still self-cross after clamping fall back to their corner
        polygon (~5% at 0.4), which XPBD re-rounds.

    hull_displacement : float
        *Hull generator only.*  Maximum per-edge radial midpoint displacement as a
        fraction of the centroid-to-midpoint distance.  At 0.0 the method collapses
        toward a plain angle-sorted loop; larger values (typical range ``[0.1, 0.4]``)
        bulge or pinch lobes for more racing-shape variety, at the cost of a higher
        downstream relaxation burden.

    checkpoint_count : int
        *Checkpoint generator only.*  Number of radial checkpoints (``C``).  Must
        be >= 3.  More checkpoints produce wavier tracks (chicane count scales ~C);
        fewer produce calmer, rounder tracks.  CarRacing canonical default: 12.
    checkpoint_radius_min_frac : float
        Inner-radius fraction for checkpoint placement.  Checkpoint radii are drawn
        from ``U(checkpoint_radius_min_frac * R, R)`` where ``R = 1``.  Must be in
        ``[0, 1)``.  The CarRacing canonical value 0.33 gives the inlets.
    checkpoint_angle_jitter : float
        Angular jitter as a fraction of the per-checkpoint angular slot
        (``slot = 2*pi/C``).  Kept below 1.0 so the checkpoint sequence stays
        angle-monotone, ensuring the steered path winds once around.  Default 0.55.
    checkpoint_turn_rate : float
        Maximum heading change (radians) per steering step.  Bounds path curvature;
        too large allows kinks, too small prevents tracking the inlets.  Default 0.42.
    checkpoint_steer_gain : float
        Proportional steering gain toward the current target bearing, in ``(0, 1]``.
        Controls how aggressively the heading chases each checkpoint per step, coupled
        to one lap by the fixed step length ``dl``.  Default 0.65.
    checkpoint_lookahead_frac : float
        Advance-to-next-checkpoint threshold as a fraction of outer radius ``R``.
        The path moves to the next checkpoint once within
        ``checkpoint_lookahead_frac * R`` of the current target, keeping flow from
        stalling at each checkpoint.  Default 0.16.
    checkpoint_best_of_k : int
        Best-of-K candidate count.  Must be >= 1.  Generates ``K`` decorrelated
        candidates per env and keeps the one with the fewest self-intersections
        (deterministic argmin; ties broken by lowest ``k``).  Replaces CarRacing's
        unbounded reject-retry with a bounded, graph-capturable pool; K=4 leaves
        ≲0.4% of tracks with a pre-relax self-intersection at the default config.
    checkpoint_clip_fallback : bool
        Opt-in single-crossing clip fallback (default ``False``).  When ``True``,
        the first self-crossing of the selected centerline is clipped: split at the
        intersection point, keep the longer sub-loop arc, resample to ``num_points``.
        A capture-time Python branch; the captured graph stays fixed either way.

    style_sampling : bool
        Opt-in per-env Bezier style randomisation (default ``False``).  When
        ``False``, the scalar ``rad`` / ``scale`` / ``handle_clamp_frac`` are used
        for all envs unchanged.  When ``True``, each env draws its own values from
        the ``*_range`` fields via the per-env Warp RNG, so a single batch spans a
        *family* of styles.  Resolved at CUDA-graph capture time (a Python branch);
        per-env values are device data so the captured graph remains fixed.
    rad_range : tuple of float or None
        ``(lo, hi)`` range for per-env ``rad`` when ``style_sampling=True``.
        ``None`` collapses to the scalar ``rad`` for all envs.
    scale_range : tuple of float or None
        ``(lo, hi)`` range for per-env ``scale`` when ``style_sampling=True``.
        ``None`` collapses to the scalar ``scale`` for all envs.
    handle_clamp_frac_range : tuple of float or None
        ``(lo, hi)`` range for per-env ``handle_clamp_frac`` when
        ``style_sampling=True``.  ``None`` collapses to the scalar
        ``handle_clamp_frac`` for all envs.

    polar_num_knots : int
        *Polar generator only.*  Number of periodic cubic-spline control knots.
        More knots add higher-frequency shape variety.  Default 12.
    polar_radial_jitter : float
        *Polar generator only.*  Radial jitter magnitude for polar control knots, as
        a fraction of the base radius.  Clamped internally to keep radii positive;
        larger values produce more eccentric track shapes.  Default 0.60.
    polar_angular_jitter : float
        *Polar generator only.*  Angular jitter per polar knot, as a fraction of a
        half angular cell, so the sorted order is preserved without a runtime sort.
        Larger values add more angular irregularity.  Default 0.30.

    voronoi_num_sites : int
        *Voronoi generator only.*  Total cell sites in the fixed site field.  Must
        be >= ``voronoi_control_points``.  More sites enrich anchor-snap targets at
        the cost of a larger all-sites scan.  Default 256.
    voronoi_site_layout : str
        *Voronoi generator only.*  Site distribution layout.  One of ``"ring"``
        (uniform box fill — NOT an annular band; name kept for config compatibility),
        ``"void_ring"`` (default — annular band with center void,
        r ∈ [0.14·box, 0.52·box], biased for usable closed loops),
        ``"clustered"`` (6 radial Gaussian clusters with ~22% annular fallback per
        site), or ``"mixed"`` (65% annular sites blended with 35% uniform box fill).
    voronoi_control_points : int
        *Voronoi generator only.*  Number of angular anchor sites selected from the
        site field (``K``).  Must be >= 6 for Chaikin/Catmull-Rom densification to
        produce a smooth loop.  More control points add local features and can raise
        the downstream relaxation burden.
    voronoi_radial_variation : float
        *Voronoi generator only.*  Radial modulation of anchor target positions.
        Must be >= 0.  At 0, all targets are placed at a uniform annulus radius;
        larger values allow more eccentric loops.  Default 0.62.
    voronoi_angular_jitter : float
        *Voronoi generator only.*  Angular jitter for anchor sector targets, as a
        fraction of a full sector.  Must be >= 0.  Larger values randomise the
        angular spacing of targets.  Default 0.08.

    repulsive_grow_mult_min : float
        *Repulsive generator only.*  Lower bound of the per-env target-perimeter
        multiplier draw ``U(min, max)``: the grown loop's target perimeter is
        ``grow_mult * 2π * r_init``.  Higher → denser mazes at some yield cost.

        .. warning::

            The ``repulsive`` generator is a **host-driven, non-graph-capturable
            optimizer** — roughly **1000× slower** than ``bezier`` (~0.18 s @ E=64,
            ~5.1 s @ E=8192 on an RTX 4090, vs ~2 ms for bezier).  It runs eagerly on
            CUDA every call (no captured replay).  Regenerate on a **slow cadence** or
            in **staggered per-env slices**, not every frame.
    repulsive_grow_mult_max : float
        *Repulsive generator only.*  Upper bound of the target-perimeter multiplier
        draw.  Must be >= ``repulsive_grow_mult_min``.  Overfill ratio ≈
        ``grow_mult / repulsive_domain_init_ratio``.
    repulsive_domain_frac : float
        *Repulsive generator only.*  Confinement-disc radius as a fraction of the
        world-scale reference length (``r_dom = repulsive_domain_frac * scale_ref *
        config.scale``).  Sets the grown curve's absolute scale vs ``half_width``.
        Must be > 0.  Tighter → richer folds.
    repulsive_domain_init_ratio : float
        *Repulsive generator only.*  Ratio ``r_dom / r_init`` of the domain radius to
        the initial seed-circle radius.  Must be > 1.  Reference value 4.0.
    repulsive_obstacle_count_min : int
        *Repulsive generator only.*  Lower bound of the per-env inner-disc obstacle
        count ``k ~ randi[min, max]``.  Must be >= 1.
    repulsive_obstacle_count_max : int
        *Repulsive generator only.*  Upper bound of the inner-disc count.  Must be >=
        ``repulsive_obstacle_count_min``.
    repulsive_obstacle_radius_min_frac : float
        *Repulsive generator only.*  Lower bound of the inner-disc radius as a fraction
        of ``r_dom``.  Must be > 0.
    repulsive_obstacle_radius_max_frac : float
        *Repulsive generator only.*  Upper bound of the inner-disc radius fraction.
        Must be >= ``repulsive_obstacle_radius_min_frac``.
    repulsive_ratchet_rate : float
        *Repulsive generator only.*  Per-iteration length-growth factor of the hard
        ratchet.  Must be > 0.  ``<= 0.013`` holds full yield with folds; ``>= 0.016``
        collapses the folds (physical limit).
    repulsive_alpha : float
        *Repulsive generator only.*  Tangent-point energy exponent alpha.  Must be > 0.
    repulsive_beta : float
        *Repulsive generator only.*  Tangent-point energy exponent beta; obstacle
        inverse power is ``p = beta - alpha``.  Must be > 0.
    repulsive_tau : float
        *Repulsive generator only.*  Normalized flow step size.
    repulsive_w_len : float
        *Repulsive generator only.*  Weight of the small inert length regularizer.
    repulsive_stages : tuple of int
        *Repulsive generator only.*  Coarse-to-fine resolution schedule ``N = 64 →
        128 → 256``.  Must be non-empty, strictly increasing, and each a positive
        multiple of 4.  The last entry must equal ``num_points`` (the tail input
        resolution) when ``generator == "repulsive"``.
    repulsive_settle_iters : int
        *Repulsive generator only.*  Settle-phase iteration budget above the ratchet.
    repulsive_resample_every : int
        *Repulsive generator only.*  Periodic arc-length reparameterization interval.
    repulsive_stall_window : int
        *Repulsive generator only.*  Iterations between stall checks / early-exit
        readbacks.
    repulsive_stall_area_tol : float
        *Repulsive generator only.*  Freeze an env once it is past target length AND
        its enclosed-area relative change over a stall window is below this
        (reparameterization-invariant shape-convergence tolerance).
    repulsive_deactivate_obstacles : bool
        *Repulsive generator only.*  When ``True`` (default), zero the inner-disc
        obstacle weights once an env reaches its target length (the wall stays live),
        closing the disc halos.

    num_harmonics : int
        *Fourier / experimental — not used by the runtime path.*  Number of Fourier
        harmonics (``K``).  Retained for compatibility with
        ``track_gen._experimental.fourier`` and older sweeps.
    decay_p : int
        *Fourier / experimental — not used by the runtime path.*  Fourier amplitude
        decay exponent: amplitude ~ ``amp / k ** decay_p``.
    amplitude : float
        *Fourier / experimental — not used by the runtime path.*  Fourier base
        amplitude.
    num_centerline_samples : int
        *Fourier / experimental — not used by the runtime path.*  Dense sample count
        for the Fourier generator (``M_max``).

    half_width : float
        Track half-width in the same coordinate units as the centerline.  Must be
        > 0.  All spacing, relaxation, and inflation distances couple to this value:
        ``spacing`` defaults to ``0.6 * half_width``; the XPBD separation target is
        ``2 * half_width * (1 + relax_margin)``; ``N_max`` headroom scales with
        ``1 / half_width`` at a given track scale.

    relax_enable : bool
        When ``True`` (default), the XPBD relaxation solve runs before inflation.
        When ``False``, the constant-spacing centerline is inflated directly; useful
        for ablation studies but typically produces lower-quality tracks.
    relax_solver : str
        Backend selector.  Only ``"xpbd"`` is used by the runtime path
        (``TrackGenerator``).  ``"energy"`` (Adam) and ``"tp_sobolev"`` are
        oracle-only backends in ``tests/_oracle/relaxation.py``; ``TrackGenerator``
        rejects any solver other than ``"xpbd"`` at construction time.
    relax_chunk_size : int or None
        *Energy oracle backend only — not used by the runtime path.*  Env-chunk
        size for the dense ``[E, N, N]`` interaction term in the energy backend.
        ``None`` disables chunking.
    relax_use_warp : bool or None
        *Oracle-only — not used by the runtime path.*  When ``False``, oracle tests
        redirect relaxation to the pure-torch path instead of Warp.  Ignored by
        ``TrackGenerator`` and ``warp_relax.py``.
    relax_tol : float
        Validity thickness tolerance.  The minimum valid track half-width is
        ``(1 - relax_tol) * half_width``.  Default 0.02 (2% tolerance).
    relax_band : int or None
        Bead-index exclusion band for the separation constraint.  ``None`` (default)
        auto-computes ``round(2 * half_width / L0)`` per track, where ``L0`` is the
        rest edge length.  Set explicitly to override the per-track auto-band.
    relax_iters : int
        Number of XPBD Jacobi sweeps.  Default 150.  Each sweep applies separation,
        spacing, and bending corrections; calibrated for ~0.999 end-to-end valid
        yield at the library defaults.

    relax_sep_every : int
        Broadphase refresh interval in sweeps.  Must be >= 1.  At 1, the dense
        ``O(count²)`` separation scan (or cache rebuild) runs every sweep.  At
        ``K > 1`` with ``relax_sep_cache_slots == 0``, the dense scan is skipped
        between refreshes (fast but may miss transient contacts).  With
        ``relax_sep_cache_slots > 0``, this is the cache-rebuild interval; the exact
        narrowphase still runs every sweep on cached candidates.  Default 40.
    relax_sep_cache_slots : int
        Broadphase candidate cache capacity per bead.  0 disables the cache (uses
        the dense cadenced scan).  When > 0, each bead stores up to this many
        non-neighbour candidate indices; the exact separation push runs against
        cached candidates every sweep.  Larger values reduce missed contacts at the
        cost of memory and narrowphase work.  Default 16.
    relax_sep_cache_skin : float
        Broadphase skin as a fraction of the exact separation target.  Must be >= 0.
        The cache stores candidates within ``target * (1 + skin)``; the exact
        narrowphase test still requires ``dist < target``.  Skin 0 is fastest;
        positive values retain candidates that may enter contact during a long cache
        interval.  Default 0.5.
    relax_sep_relax : float
        Scale applied to the separation-constraint correction.  Default 1.0 (full
        correction).  Reduce toward 0 to soften separation enforcement.
    relax_spc_relax : float
        Scale applied to the edge-spacing (rest-length) correction.  Default 1.0.
        Reduce to allow more spacing deviation.
    relax_bend_relax : float
        Scale applied to the bending/radius-guard correction.  Default 1.5 (slightly
        over-corrects to quickly remove sharp corners introduced by separation
        pushes).  Reduce if the solver oscillates.
    relax_margin : float
        Fractional expansion of the separation target beyond the road diameter:
        ``target = 2 * half_width * (1 + relax_margin)``.  Default 0.15 (15%
        margin).  Leaves room for the downstream thickness validity gate.

    energy_steps : int
        *Energy oracle backend — not used by the runtime path.*  Adam optimiser
        steps for the ``"energy"`` oracle backend in ``tests/_oracle/relaxation.py``.
    energy_lr : float
        *Energy oracle backend — not used by the runtime path.*  Adam learning rate.
    energy_w_sep : float
        *Energy oracle backend — not used by the runtime path.*  Separation loss
        weight.
    energy_w_len : float
        *Energy oracle backend — not used by the runtime path.*  Edge-length loss
        weight.
    energy_w_bend : float
        *Energy oracle backend — not used by the runtime path.*  Bending loss weight.
    energy_w_anchor : float
        *Energy oracle backend — not used by the runtime path.*  Anchor loss weight.

    tp_iters : int
        *TP-Sobolev oracle backend — not used by the runtime path.*  Flow steps for
        the ``"tp_sobolev"`` oracle backend in ``tests/_oracle/relaxation.py``.
    tp_tau : float
        *TP-Sobolev oracle backend — not used by the runtime path.*  Step size for
        the TP-Sobolev flow.
    tp_alpha : float
        *TP-Sobolev oracle backend — not used by the runtime path.*  Sobolev kernel
        exponent alpha; shared with the ``smooth_finish`` finisher.
    tp_beta : float
        *TP-Sobolev oracle backend — not used by the runtime path.*  Sobolev kernel
        exponent beta; shared with the ``smooth_finish`` finisher.
    smooth_finish : bool
        *Oracle-only — not used by the runtime path.*  When ``True``, a
        tangent-point/Sobolev smoothing finisher runs after the primary relaxation
        solve.  ``TrackGenerator`` rejects ``smooth_finish=True`` at construction
        time.
    smooth_finish_iters : int
        *Oracle-only — not used by the runtime path.*  Flow steps for the smoothing
        finisher.
    smooth_finish_tau : float
        *Oracle-only — not used by the runtime path.*  Step size for the smoothing
        finisher.

    num_points : int
        Intermediate dense-resample resolution (``N``) used before the
        constant-spacing step.  Default 256.  The constant-spacing resampler then
        derives ``count[e] = floor(perimeter / spacing) + 1`` real points per env.
    output_mode : str
        Output resampling mode.  The only supported value is
        ``"constant_spacing"`` (enforced by ``__post_init__``).  Constant arc
        spacing produces smoother XPBD convergence and higher valid-track yield than
        a fixed point count.
    spacing : float or None
        Constant arc-length step (m) between adjacent output points.  ``None``
        (default) auto-couples to ``0.6 * half_width`` — the relaxation-friendly
        value; a fixed default would be wrong as ``half_width`` varies.  Must
        resolve to > 0 after auto-coupling.
    N_max : int
        Output buffer width: maximum constant-spacing points per track.  Tracks
        whose true ``count[e]`` exceeds ``N_max`` are truncated and fail validity.
        Default 384: ample headroom at ``half_width = 0.1`` (spacing ≈ 0.06,
        ~141 max points), full coverage to ``half_width ≈ 0.05`` (~281 max), and
        ~98% of the ``half_width = 0.03`` regime (~468 max).  Raise ``N_max`` or
        coarsen ``spacing`` for finer/larger-scale regimes.

    max_regen_iters : int
        *Vestigial on the Warp runtime path.*  Maximum regeneration iterations for
        the torch oracle's accept-first-valid loop in
        ``tests/_oracle/generators.py``.  Ignored by ``_run_pipeline``, which is
        single-pass with no host-side retry loop.
    turning_tol : float
        Absolute angular tolerance in radians for the turning-number validity gate.
        Tracks whose signed curvature integral deviates from the expected full turn by
        more than this tolerance are marked invalid.  Default 0.1.
    w_floor : float
        Width floor for validity.  Every real track point must have computed
        half-width > ``w_floor``.  Default ``1e-3``.  Guards against near-degenerate
        geometry that passes the global thickness check.
    validity_border_check : bool
        When ``True``, an explicit border self-intersection check is added to the
        validity gate.  Default ``False``.  Redundant with the thickness/separation
        gate (a self-crossing fat band drives
        ``separation_min → 0 → thickness < half_width → invalid``), so the default
        saves two ``O(N²)`` passes with no change to the valid mask.
    """

    # --- Generator selection + batching ---
    generator: str = "bezier"
    device: str = "cpu"
    num_envs: int = 1

    # --- Bezier params ---
    min_num_points: int = 9
    max_num_points: int = 13
    num_points_per_segment: int = 30
    min_point_distance: float = 0.05
    min_angle: float = (12.5 / 180) * math.pi
    rad: float = 0.4
    edgy: float = 0.0
    scale: float = 1.0
    handle_clamp_frac: float = 0.4

    # --- Hull generator params (generator="hull") ---
    hull_displacement: float = 0.15

    # --- Checkpoint-steering generator params (generator="checkpoint", method #5) ---
    checkpoint_count: int = 12
    checkpoint_radius_min_frac: float = 0.33
    checkpoint_angle_jitter: float = 0.55
    checkpoint_turn_rate: float = 0.42
    checkpoint_steer_gain: float = 0.65
    checkpoint_lookahead_frac: float = 0.16
    checkpoint_best_of_k: int = 4
    checkpoint_clip_fallback: bool = False

    # --- Per-env style sampling (method #1: "Per-env style randomization") ---
    style_sampling: bool = False
    rad_range: tuple[float, float] | None = None
    scale_range: tuple[float, float] | None = None
    handle_clamp_frac_range: tuple[float, float] | None = None

    # --- Polar / Fourier params ---
    polar_num_knots: int = 12
    polar_radial_jitter: float = 0.60
    polar_angular_jitter: float = 0.30

    # --- Voronoi / graph-cycle generator params (generator="voronoi") ---
    voronoi_num_sites: int = 256
    voronoi_site_layout: str = "void_ring"  # {"ring", "void_ring", "clustered", "mixed"}
    voronoi_control_points: int = 18
    voronoi_radial_variation: float = 0.62
    voronoi_angular_jitter: float = 0.08
    # --- Repulsive-growth generator params (generator="repulsive") ---
    repulsive_grow_mult_min: float = 4.5
    repulsive_grow_mult_max: float = 5.5
    repulsive_domain_frac: float = 0.35
    repulsive_domain_init_ratio: float = 4.0
    repulsive_obstacle_count_min: int = 8
    repulsive_obstacle_count_max: int = 12
    repulsive_obstacle_radius_min_frac: float = 0.02
    repulsive_obstacle_radius_max_frac: float = 0.045
    repulsive_ratchet_rate: float = 0.012
    repulsive_alpha: float = 3.0
    repulsive_beta: float = 6.0
    repulsive_tau: float = 0.4
    repulsive_w_len: float = 30.0
    repulsive_stages: tuple[int, ...] = (64, 128, 256)
    repulsive_settle_iters: int = 40
    repulsive_resample_every: int = 25
    repulsive_stall_window: int = 16
    repulsive_stall_area_tol: float = 0.05
    repulsive_deactivate_obstacles: bool = True

    # Experimental torch-only Fourier generator fields retained for compatibility with
    # track_gen._experimental.fourier and older sweeps.
    num_harmonics: int = 5  # K
    decay_p: int = 2  # decay exponent: amplitude ~ amp / k**decay_p
    amplitude: float = 1.0
    num_centerline_samples: int = 256  # Fourier dense sample count (M_max)

    # --- Width params ---
    half_width: float = 0.1

    # --- Relaxation: backend selection + scale ---
    relax_enable: bool = True
    relax_solver: str = "xpbd"            # {"xpbd","energy","tp_sobolev"}
    relax_chunk_size: int | None = None
    relax_use_warp: bool | None = None
    relax_tol: float = 0.02
    relax_band: int | None = None
    relax_iters: int = 150
    relax_sep_every: int = 40
    relax_sep_cache_slots: int = 16
    relax_sep_cache_skin: float = 0.5
    relax_sep_relax: float = 1.0
    relax_spc_relax: float = 1.0
    relax_bend_relax: float = 1.5
    relax_margin: float = 0.15

    # energy (Adam) — oracle-only backend
    energy_steps: int = 800
    energy_lr: float = 3e-3
    energy_w_sep: float = 80.0
    energy_w_len: float = 8.0
    energy_w_bend: float = 1.0
    energy_w_anchor: float = 0.01
    # tp_sobolev — oracle-only backend (standalone + finisher share tp_alpha/tp_beta)
    tp_iters: int = 100
    tp_tau: float = 0.7
    tp_alpha: float = 2.0
    tp_beta: float = 4.5
    # optional tangent-point/Sobolev smoothing finisher — oracle-only
    smooth_finish: bool = False
    smooth_finish_iters: int = 8
    smooth_finish_tau: float = 0.2

    # --- Output params ---
    num_points: int = 256  # N: intermediate dense->resample resolution before constant-spacing
    output_mode: str = "constant_spacing"  # the only supported mode (see __post_init__)
    spacing: float | None = None
    N_max: int = 384

    # --- Robustness params ---
    max_regen_iters: int = 10
    turning_tol: float = 0.1
    w_floor: float = 1e-3
    validity_border_check: bool = False

    def __post_init__(self):
        if int(self.num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs!r}")
        if float(self.half_width) <= 0.0:
            raise ValueError(f"half_width must be > 0, got {self.half_width!r}")
        # Point-family sampler inputs (shared bezier/hull corner sampler). min_point_distance
        # is a divisor: num_cells = int(1/(min_point_distance*2)) in warp_generate.py, so
        # values <= 0 or > 0.5 drive a divide-by-zero in the grid kernels; min/max_num_points
        # feed wp.randi and must be a non-inverted range. GateGenConfig already guards these.
        if int(self.min_num_points) < 2:
            raise ValueError(f"min_num_points must be >= 2, got {self.min_num_points!r}")
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}")
        if not (0.0 < float(self.min_point_distance) <= 0.5):
            raise ValueError(
                f"min_point_distance must be in (0, 0.5], got {self.min_point_distance!r}")
        if int(self.num_points_per_segment) < 2:
            raise ValueError(
                f"num_points_per_segment must be >= 2, got {self.num_points_per_segment!r}")
        if int(self.voronoi_control_points) < 6:
            raise ValueError(
                f"voronoi_control_points must be >= 6, got {self.voronoi_control_points!r}")
        if int(self.voronoi_num_sites) < int(self.voronoi_control_points):
            raise ValueError(
                "voronoi_num_sites must be >= voronoi_control_points, got "
                f"{self.voronoi_num_sites!r} < {self.voronoi_control_points!r}")
        if self.voronoi_site_layout not in {"ring", "void_ring", "clustered", "mixed"}:
            raise ValueError(
                "voronoi_site_layout must be one of "
                "{'ring', 'void_ring', 'clustered', 'mixed'}, got "
                f"{self.voronoi_site_layout!r}")
        if float(self.voronoi_radial_variation) < 0.0:
            raise ValueError(
                f"voronoi_radial_variation must be >= 0, got {self.voronoi_radial_variation!r}")
        if float(self.voronoi_angular_jitter) < 0.0:
            raise ValueError(
                f"voronoi_angular_jitter must be >= 0, got {self.voronoi_angular_jitter!r}")

        # Repulsive-growth generator validation (Spec §3.1). Ordered/positive knobs;
        # the stages schedule is coarse-to-fine (strictly increasing, multiple of 4) and
        # its last entry is the tail input resolution, so it must equal num_points for
        # the repulsive generator.
        # grow_mult_min >= 1 and grow_mult_max > 1: a multiplier <= 1 shrinks (or freezes) the
        # init circle so the ratchet never grows, which drives the iteration budget
        # n_ratchet = ceil(log(grow_mult_max)/log1p(rate)) to <= 0 -> a zero-length growth loop
        # that leaves the centerline mis-strided at the coarsest stage (silent garbage tracks).
        if float(self.repulsive_grow_mult_min) < 1.0:
            raise ValueError(
                f"repulsive_grow_mult_min must be >= 1, got {self.repulsive_grow_mult_min!r}")
        if float(self.repulsive_grow_mult_max) <= 1.0:
            raise ValueError(
                f"repulsive_grow_mult_max must be > 1, got {self.repulsive_grow_mult_max!r}")
        if float(self.repulsive_grow_mult_max) < float(self.repulsive_grow_mult_min):
            raise ValueError(
                "repulsive_grow_mult_max must be >= repulsive_grow_mult_min, got "
                f"{self.repulsive_grow_mult_max!r} < {self.repulsive_grow_mult_min!r}")
        if float(self.repulsive_domain_frac) <= 0.0:
            raise ValueError(
                f"repulsive_domain_frac must be > 0, got {self.repulsive_domain_frac!r}")
        if float(self.repulsive_domain_init_ratio) <= 1.0:
            raise ValueError(
                "repulsive_domain_init_ratio must be > 1, got "
                f"{self.repulsive_domain_init_ratio!r}")
        if int(self.repulsive_obstacle_count_min) < 1:
            raise ValueError(
                "repulsive_obstacle_count_min must be >= 1, got "
                f"{self.repulsive_obstacle_count_min!r}")
        if int(self.repulsive_obstacle_count_max) < int(self.repulsive_obstacle_count_min):
            raise ValueError(
                "repulsive_obstacle_count_max must be >= repulsive_obstacle_count_min, got "
                f"{self.repulsive_obstacle_count_max!r} < {self.repulsive_obstacle_count_min!r}")
        if float(self.repulsive_obstacle_radius_min_frac) <= 0.0:
            raise ValueError(
                "repulsive_obstacle_radius_min_frac must be > 0, got "
                f"{self.repulsive_obstacle_radius_min_frac!r}")
        if (float(self.repulsive_obstacle_radius_max_frac)
                < float(self.repulsive_obstacle_radius_min_frac)):
            raise ValueError(
                "repulsive_obstacle_radius_max_frac must be >= "
                "repulsive_obstacle_radius_min_frac, got "
                f"{self.repulsive_obstacle_radius_max_frac!r} < "
                f"{self.repulsive_obstacle_radius_min_frac!r}")
        if float(self.repulsive_ratchet_rate) <= 0.0:
            raise ValueError(
                f"repulsive_ratchet_rate must be > 0, got {self.repulsive_ratchet_rate!r}")
        if float(self.repulsive_alpha) <= 0.0:
            raise ValueError(f"repulsive_alpha must be > 0, got {self.repulsive_alpha!r}")
        if float(self.repulsive_beta) <= 0.0:
            raise ValueError(f"repulsive_beta must be > 0, got {self.repulsive_beta!r}")
        stages = tuple(self.repulsive_stages)
        if len(stages) == 0:
            raise ValueError("repulsive_stages must be non-empty")
        if any(int(s) <= 0 or int(s) % 4 != 0 for s in stages):
            raise ValueError(
                f"repulsive_stages entries must be positive multiples of 4, got {stages!r}")
        if any(int(stages[i]) <= int(stages[i - 1]) for i in range(1, len(stages))):
            raise ValueError(
                f"repulsive_stages must be strictly increasing, got {stages!r}")
        if self.generator == "repulsive" and int(stages[-1]) != int(self.num_points):
            raise ValueError(
                "repulsive_stages[-1] must equal num_points for generator='repulsive', got "
                f"{stages[-1]!r} != {self.num_points!r}")
        # settle_iters >= 1 keeps the growth loop non-empty even at grow_mult_max just above 1
        # (n_iters = ceil(n_ratchet*1.6) + settle_iters); resample_every / stall_window feed the
        # `(it+1) % k` guards in the growth loop, so k=0 would be a ZeroDivisionError mid-generate.
        if int(self.repulsive_settle_iters) < 1:
            raise ValueError(
                f"repulsive_settle_iters must be >= 1, got {self.repulsive_settle_iters!r}")
        if int(self.repulsive_resample_every) < 1:
            raise ValueError(
                f"repulsive_resample_every must be >= 1, got {self.repulsive_resample_every!r}")
        if int(self.repulsive_stall_window) < 1:
            raise ValueError(
                f"repulsive_stall_window must be >= 1, got {self.repulsive_stall_window!r}")

        if int(self.relax_sep_every) < 1:
            raise ValueError(f"relax_sep_every must be >= 1, got {self.relax_sep_every!r}")
        if int(self.relax_sep_cache_slots) < 0:
            raise ValueError(
                f"relax_sep_cache_slots must be >= 0, got {self.relax_sep_cache_slots!r}")
        if float(self.relax_sep_cache_skin) < 0.0:
            raise ValueError(
                f"relax_sep_cache_skin must be >= 0, got {self.relax_sep_cache_skin!r}")

        # Checkpoint-steering generator validation: a loop needs >= 3 checkpoints, the
        # radius fraction must be a proper sub-unit inner radius, and best-of-K needs >= 1.
        if int(self.checkpoint_count) < 3:
            raise ValueError(
                f"checkpoint_count must be >= 3, got {self.checkpoint_count!r}")
        if int(self.checkpoint_best_of_k) < 1:
            raise ValueError(
                f"checkpoint_best_of_k must be >= 1, got {self.checkpoint_best_of_k!r}")
        if not (0.0 <= float(self.checkpoint_radius_min_frac) < 1.0):
            raise ValueError(
                f"checkpoint_radius_min_frac must be in [0, 1), "
                f"got {self.checkpoint_radius_min_frac!r}")

        # Only constant_spacing is supported: a constant link SIZE (~0.6*half_width) relaxes
        # to smoother, higher-yield tracks than a constant point COUNT, which over-resolves
        # the centerline (jagged XPBD -> folded roads).
        if self.output_mode != "constant_spacing":
            raise ValueError(
                f"output_mode must be 'constant_spacing' (the only supported mode), "
                f"got {self.output_mode!r}")
        # Auto-couple spacing to half_width (~0.6*half_width relaxes to smoother tracks);
        # a fixed spacing default would be wrong as half_width varies (too coarse -> degenerate).
        if self.spacing is None:
            self.spacing = 0.6 * self.half_width
        if float(self.spacing) <= 0.0:
            raise ValueError(f"spacing must be > 0, got {self.spacing!r}")


@dataclass
class GateGenConfig:
    """Configuration for fixed-batch native gate sequence generation.

    Used by ``GateGenerator`` to produce batched ``GateSequence`` results.  The
    gate path only samples corner anchors and emits them directly as gates; it never
    runs Bezier or hull curve assembly.  As a result, ``num_points_per_segment``,
    ``rad``, ``edgy``, ``handle_clamp_frac``, and ``hull_displacement`` are **inert
    for gate output** (kept for config parity with ``TrackGenConfig``).  The
    effective knobs for anchor placement are ``min_num_points``,
    ``max_num_points``, ``min_point_distance``, and ``scale``.

    Attributes
    ----------
    generator : str
        Registered generator name.  One of ``"bezier"`` (default), ``"hull"``,
        ``"polar"``, ``"voronoi"``, or ``"checkpoint"``.  Selects the anchor
        sampling family; only the anchor positions are used as gates.
    device : str
        Warp device string: ``"cpu"`` or ``"cuda"``.
    num_envs : int
        Number of environments in the batch.  Must be >= 1.

    min_gates : int
        Minimum real gate count per env.  Must be >= 2.
    max_gates : int
        Maximum real gate count per env.  Must be >= ``min_gates``.
    gate_radius : float
        Gate detection radius used for gate-constraint solve proximity checks.
        Must be >= 0.
    gate_solve_iters : int
        Number of gate-constraint solve iterations.  Must be >= 0.
    gate_width : float
        Physical gate half-width (visual/collision extent).  Must be >= 0.  Does
        not affect anchor sampling geometry.
    gate_ordering : str
        Gate output ordering.  One of ``"ccw"`` (counter-clockwise sort by angle
        around the batch centroid; default), ``"raw"`` (generator order as sampled),
        or ``"random_pairs"`` (random gate pairs for training objectives).

    min_num_points : int
        Minimum corner-anchor count per env for the bezier/hull point sampler.
        Must be >= 2.  Affects the number of sampled anchor positions.
    max_num_points : int
        Maximum corner-anchor count per env.  Must be >= ``min_num_points``.
    num_points_per_segment : int
        *Inert for gate output.*  Dense samples per segment for Bezier/hull curve
        assembly; the gate path does not assemble curves.  Kept for config parity
        with ``TrackGenConfig``.
    min_point_distance : float
        Minimum grid-cell spacing for anchor-position sampling.  Must be > 0.
        Governs the cell grid as ``num_cells = int(1 / (min_point_distance * 2))``.
    rad : float
        *Inert for gate output.*  Bezier handle-length fraction.  Kept for config
        parity with ``TrackGenConfig``.
    edgy : float
        *Inert for gate output.*  Corner-tangent blend weight.  Kept for config
        parity with ``TrackGenConfig``.
    scale : float
        Isotropic scale multiplier for anchor positions.  Does affect the sampled
        anchor coordinates and hence gate positions.
    handle_clamp_frac : float
        *Inert for gate output.*  Bezier handle clamp.  Kept for config parity with
        ``TrackGenConfig``.
    hull_displacement : float
        *Inert for gate output.*  Hull midpoint displacement fraction.  Kept for
        config parity with ``TrackGenConfig``.

    polar_num_knots : int
        *Polar generator only.*  Number of periodic cubic-spline control knots.
        Default 12.
    polar_radial_jitter : float
        *Polar generator only.*  Radial jitter magnitude for polar control knots,
        as a fraction of the base radius.  Default 0.60.
    polar_angular_jitter : float
        *Polar generator only.*  Angular jitter per polar knot, as a fraction of a
        half angular cell.  Default 0.30.

    voronoi_num_sites : int
        *Voronoi generator only.*  Total cell sites in the fixed site field.  Must
        be >= ``voronoi_control_points``.  Default 256.
    voronoi_site_layout : str
        *Voronoi generator only.*  Site distribution layout.  One of ``"ring"``,
        ``"void_ring"`` (default), ``"clustered"``, or ``"mixed"``.
    voronoi_control_points : int
        *Voronoi generator only.*  Anchor sites selected from the site field.  Must
        be >= 3 (the gate-native Voronoi path emits anchors directly as gates, so
        it needs only 3 for a non-degenerate ring — unlike ``TrackGenConfig``'s >= 6
        floor for Chaikin/Catmull-Rom densification).
    voronoi_radial_variation : float
        *Voronoi generator only.*  Radial modulation of anchor target positions.
        Must be >= 0.  Default 0.62.
    voronoi_angular_jitter : float
        *Voronoi generator only.*  Angular jitter for anchor sector targets.  Must
        be >= 0.  Default 0.08.

    checkpoint_count : int
        *Checkpoint generator only.*  Number of radial checkpoints.  Must be >= 3.
        Default 12.
    checkpoint_radius_min_frac : float
        *Checkpoint generator only.*  Inner-radius fraction for checkpoint
        placement.  Must be in ``[0, 1)``.  Default 0.33.
    checkpoint_angle_jitter : float
        *Checkpoint generator only.*  Angular jitter as a fraction of the
        per-checkpoint angular slot.  Default 0.55.
    """

    generator: str = "bezier"
    device: str = "cpu"
    num_envs: int = 1

    min_gates: int = 4
    max_gates: int = 32
    gate_radius: float = 0.025
    gate_solve_iters: int = 8
    gate_width: float = 0.0
    gate_ordering: str = "ccw"

    # Point-family (bezier/hull) sampler inputs. The gate path only samples corner
    # anchors and emits them as gates; it never runs the Bezier/hull curve assembly,
    # so num_points_per_segment, rad, edgy, handle_clamp_frac, and hull_displacement
    # are inert for gate output (they remain for parity with TrackGenConfig and are
    # hidden in the explorer UI). min_num_points, max_num_points, min_point_distance,
    # and scale do affect the sampled anchors.
    min_num_points: int = 9
    max_num_points: int = 13
    num_points_per_segment: int = 30
    min_point_distance: float = 0.05
    rad: float = 0.4
    edgy: float = 0.0
    scale: float = 1.0
    handle_clamp_frac: float = 0.4
    hull_displacement: float = 0.15

    polar_num_knots: int = 12
    polar_radial_jitter: float = 0.60
    polar_angular_jitter: float = 0.30

    voronoi_num_sites: int = 256
    voronoi_site_layout: str = "void_ring"
    voronoi_control_points: int = 18
    voronoi_radial_variation: float = 0.62
    voronoi_angular_jitter: float = 0.08

    checkpoint_count: int = 12
    checkpoint_radius_min_frac: float = 0.33
    checkpoint_angle_jitter: float = 0.55

    def __post_init__(self):
        if int(self.num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs!r}")
        if int(self.min_gates) < 2:
            raise ValueError(f"min_gates must be >= 2, got {self.min_gates!r}")
        if int(self.max_gates) < int(self.min_gates):
            raise ValueError(
                f"max_gates must be >= min_gates, got "
                f"{self.max_gates!r} < {self.min_gates!r}"
            )
        if float(self.gate_radius) < 0.0:
            raise ValueError(f"gate_radius must be >= 0, got {self.gate_radius!r}")
        if int(self.gate_solve_iters) < 0:
            raise ValueError(
                f"gate_solve_iters must be >= 0, got {self.gate_solve_iters!r}"
            )
        if float(self.gate_width) < 0.0:
            raise ValueError(f"gate_width must be >= 0, got {self.gate_width!r}")
        if self.gate_ordering not in {"ccw", "raw", "random_pairs"}:
            raise ValueError(
                "gate_ordering must be one of {'ccw', 'raw', 'random_pairs'}, "
                f"got {self.gate_ordering!r}"
            )
        # Point-family (bezier/hull) sampler inputs. These feed the shared corner
        # sampler, where min_point_distance divides a cell count and the
        # [min_num_points, max_num_points] range feeds wp.randi; validate them here
        # so direct API callers get a clean config-time error instead of an opaque
        # ZeroDivisionError or inverted-range sample deep inside generation.
        if float(self.min_point_distance) <= 0.0:
            raise ValueError(
                f"min_point_distance must be > 0, got {self.min_point_distance!r}"
            )
        if int(self.min_num_points) < 2:
            raise ValueError(
                f"min_num_points must be >= 2, got {self.min_num_points!r}"
            )
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}"
            )
        # Floor is 3 here, not 6 as in TrackGenConfig: the gate-native Voronoi
        # generator selects anchor sites and emits them directly as gates, so it
        # never runs the Chaikin/Catmull-Rom densification that TrackGenConfig's
        # >= 6 floor protects. Three control points is the smallest non-degenerate
        # gate ring, which lets callers request short gate sequences.
        if int(self.voronoi_control_points) < 3:
            raise ValueError(
                f"voronoi_control_points must be >= 3, got {self.voronoi_control_points!r}"
            )
        if int(self.voronoi_num_sites) < int(self.voronoi_control_points):
            raise ValueError(
                "voronoi_num_sites must be >= voronoi_control_points, got "
                f"{self.voronoi_num_sites!r} < {self.voronoi_control_points!r}"
            )
        if float(self.voronoi_radial_variation) < 0.0:
            raise ValueError(
                "voronoi_radial_variation must be >= 0, got "
                f"{self.voronoi_radial_variation!r}"
            )
        if float(self.voronoi_angular_jitter) < 0.0:
            raise ValueError(
                "voronoi_angular_jitter must be >= 0, got "
                f"{self.voronoi_angular_jitter!r}"
            )
        if self.voronoi_site_layout not in {"ring", "void_ring", "clustered", "mixed"}:
            raise ValueError(
                "voronoi_site_layout must be one of "
                "{'ring', 'void_ring', 'clustered', 'mixed'}, got "
                f"{self.voronoi_site_layout!r}"
            )
        if int(self.checkpoint_count) < 3:
            raise ValueError(
                f"checkpoint_count must be >= 3, got {self.checkpoint_count!r}"
            )
        if not (0.0 <= float(self.checkpoint_radius_min_frac) < 1.0):
            raise ValueError(
                "checkpoint_radius_min_frac must be in [0, 1), got "
                f"{self.checkpoint_radius_min_frac!r}"
            )


@dataclass
class GateSequence:
    """Batched fixed-stride gate result returned by ``GateGenerator``.

    Gate pose arrays are flat ``[E * max_gates]`` ``vec2f`` buffers; reshape via
    ``wp.to_torch(...).view(E, max_gates, 2)``. ``count[e]`` gives the real gate
    count for environment ``e`` and slots ``i >= count[e]`` are NaN-padded.

    .. warning::

        ``GateGenerator.generate()`` returns the SAME ``GateSequence`` instance on
        every call and overwrites its buffers in place.  A reference held across two
        ``generate()`` calls will see mutated data.  Call ``GateSequence.clone()`` to
        obtain a fully-owned deep copy before the next call.

    Attributes
    ----------
    position : wp.array
        Flat ``[E * max_gates]`` ``vec2f`` gate centres.  Reshape via
        ``wp.to_torch(...).view(E, max_gates, 2)``.  Slots ``i >= count[e]`` are
        NaN-padded.
    tangent : wp.array
        Flat ``[E * max_gates]`` ``vec2f`` unit tangent vectors at each gate centre.
        NaN-padded past ``count[e]``.
    normal : wp.array
        Flat ``[E * max_gates]`` ``vec2f`` unit normals perpendicular to ``tangent``.
        NaN-padded past ``count[e]``.
    left : wp.array
        Flat ``[E * max_gates]`` ``vec2f`` left gate endpoints
        (``position + gate_width * normal``).  NaN-padded past ``count[e]``.
    right : wp.array
        Flat ``[E * max_gates]`` ``vec2f`` right gate endpoints
        (``position - gate_width * normal``).  NaN-padded past ``count[e]``.
    valid : wp.array
        ``[E]`` ``int32`` environment validity flags (0 or 1).  Convert to a boolean
        tensor via ``wp.to_torch(...).bool()``.
    count : wp.array
        ``[E]`` ``int32`` real gate counts per env.  All other arrays are NaN-padded
        for slots ``i >= count[e]``.
    """

    position: wp.array
    tangent: wp.array
    normal: wp.array
    left: wp.array
    right: wp.array
    valid: wp.array
    count: wp.array

    def clone(self) -> "GateSequence":
        """Return a deep copy whose Warp buffers do not alias this sequence."""
        return GateSequence(
            position=wp.clone(self.position),
            tangent=wp.clone(self.tangent),
            normal=wp.clone(self.normal),
            left=wp.clone(self.left),
            right=wp.clone(self.right),
            valid=wp.clone(self.valid),
            count=wp.clone(self.count),
        )


@dataclass
class Track:
    """Final batched result of the track generation pipeline.

    All boundary arrays are index-aligned: ``outer[i]``, ``center[i]`` and
    ``inner[i]`` share a single cross-section normal. Half-width is not stored;
    recover it as the outer-center norm along dim=-1.
    Fields are ``wp.array``; convert at the boundary via the wp bridge.

    .. warning::

        ``TrackGenerator.generate()`` returns the SAME ``Track`` instance on every
        call and overwrites its buffers in place.  A reference held across two
        ``generate()`` calls will see mutated data.  Call ``Track.clone()`` to obtain
        a fully-owned deep copy before the next call.

    Attributes
    ----------
    outer : wp.array
        Flat ``[E * N_max]`` ``vec2f`` outer boundary points.  Reshape via
        ``wp.to_torch(...).view(E, N_max, 2)``.  Points at ``i >= count[e]`` are
        NaN-padded.
    center : wp.array
        Flat ``[E * N_max]`` ``vec2f`` centerline points.  Reshape via
        ``wp.to_torch(...).view(E, N_max, 2)``.  Index-aligned with ``outer`` and
        ``inner``; ``‖outer[i] - center[i]‖`` gives the per-point half-width.
        Points at ``i >= count[e]`` are NaN-padded.
    inner : wp.array
        Flat ``[E * N_max]`` ``vec2f`` inner boundary points.  Reshape via
        ``wp.to_torch(...).view(E, N_max, 2)``.  Points at ``i >= count[e]`` are
        NaN-padded.
    tangent : wp.array
        Flat ``[E * N_max]`` ``vec2f`` unit tangent vectors at each centerline
        point, derived from central differences.  NaN-padded past ``count[e]``.
    normal : wp.array
        Flat ``[E * N_max]`` ``vec2f`` unit left-normals at each centerline point:
        ``normal = (-tangent.y, tangent.x)`` (``tangent`` rotated +90°).  Which
        boundary it faces is winding-dependent (see ``winding``): it points toward
        ``outer`` for clockwise loops and toward ``inner`` for counter-clockwise
        loops.  The ``outer``/``inner`` split itself is winding-agnostic (assigned by
        signed-area magnitude), so use ``outer``/``inner`` — not the normal's sign —
        to identify the boundaries.  NaN-padded past ``count[e]``.
    arclen : wp.array
        Flat ``[E * N_max]`` ``float32`` cumulative arc length along the centerline
        at each point.  Reshape via ``wp.to_torch(...).view(E, N_max)``.  NaN-padded
        past ``count[e]``.
    length : wp.array
        ``[E]`` ``float32`` total arc length (perimeter) of each track's centerline
        loop.
    valid : wp.array
        ``[E]`` ``int32`` environment validity flags (0 or 1).  A track is valid
        when: turning number ≈ 1, no NaN points, width floor exceeded everywhere,
        minimum thickness ≥ ``(1 − relax_tol) * half_width``, and (when
        ``validity_border_check=True``) zero border self-intersections.  Convert via
        ``wp.to_torch(...).bool()``.
    count : wp.array
        ``[E]`` ``int32`` real point counts, derived from constant-spacing resampling
        as ``count[e] = floor(perimeter / spacing) + 1``, capped at ``N_max``.  All
        other arrays are NaN-padded for indices ``i >= count[e]``.
    winding : wp.array
        ``[E]`` ``float32`` signed loop winding: ``+1.0`` for a counter-clockwise
        centerline, ``-1.0`` for clockwise (the sign of the loop's signed area), and
        ``0.0`` for degenerate/empty loops.  Winding is generator-dependent (bezier
        and hull wind clockwise; polar, voronoi and checkpoint wind counter-clockwise),
        so this field lets a consumer orient itself: for a CCW loop ``outer`` lies to
        the right of increasing arc length, for a CW loop to the left.
    """

    outer: wp.array
    center: wp.array
    inner: wp.array
    tangent: wp.array
    normal: wp.array
    arclen: wp.array
    length: wp.array
    valid: wp.array
    count: wp.array
    winding: wp.array

    def clone(self) -> "Track":
        """Return a fully-owned deep copy of this Track.

        Each field is cloned via ``wp.clone`` (torch-free), so the returned Track owns
        independent buffers unaffected by future ``generate()`` calls that overwrite
        this instance in place.
        """
        return Track(
            outer=wp.clone(self.outer),
            center=wp.clone(self.center),
            inner=wp.clone(self.inner),
            tangent=wp.clone(self.tangent),
            normal=wp.clone(self.normal),
            arclen=wp.clone(self.arclen),
            length=wp.clone(self.length),
            valid=wp.clone(self.valid),
            count=wp.clone(self.count),
            winding=wp.clone(self.winding),
        )
