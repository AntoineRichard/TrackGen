"""Shared dataclasses for the track generation pipeline.

This module imports nothing from the rest of the package (no generators, no
inflation, no track_generator, no rng_utils). It is the shared home for
``TrackGenConfig`` and ``Track`` so that ``warp_pipeline.py`` and the facade can
both import them without a circular import. Warp is a core dependency (``Track``
fields are ``wp.array``).
"""

import dataclasses
import math
import operator
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import warp as wp


def _validate_z_fields(cfg) -> None:
    """Shared Z-profile validation for ``TrackGenConfig`` and ``GateGenConfig``.

    Operates purely on the ``z_*`` attribute names both configs carry on the same
    terms (min/max ordering, non-negative knobs), so a single call site works for
    either dataclass's ``__post_init__``. ``z_profile`` membership is not checked
    here: in both configs it names any registered elevation stage and is resolved
    through the registry at ``TrackGenerator`` / ``GateGenerator`` construction.
    """
    if float(cfg.z_min) > float(cfg.z_max):
        raise ValueError(
            f"z_min must be <= z_max, got {cfg.z_min!r} > {cfg.z_max!r}"
        )
    if float(cfg.z_max_step) < 0.0:
        raise ValueError(
            f"z_max_step must be >= 0, got {cfg.z_max_step!r}"
        )
    if float(cfg.z_noise_amplitude) < 0.0:
        raise ValueError(
            "z_noise_amplitude must be >= 0, got "
            f"{cfg.z_noise_amplitude!r}"
        )
    if int(cfg.z_noise_harmonics) < 1:
        raise ValueError(
            f"z_noise_harmonics must be >= 1, got {cfg.z_noise_harmonics!r}"
        )
    if float(cfg.z_valid_grade) < 0.0:
        raise ValueError(
            f"z_valid_grade must be >= 0, got {cfg.z_valid_grade!r}"
        )


def _validate_invalid_policy(cfg) -> None:
    """Check the invalid-output policy without loading generators or plugins."""
    if cfg.on_invalid not in ("keep", "poison"):
        raise ValueError(f"on_invalid must be 'keep' or 'poison', got {cfg.on_invalid!r}")


def _check_generator_params(cfg) -> None:
    """Registry-free type check of ``generator_params`` (the generator resolves and
    validates the value at construction)."""
    gp = cfg.generator_params
    if gp is not None and not isinstance(gp, Mapping) and not (
            dataclasses.is_dataclass(gp) and not isinstance(gp, type)):
        raise TypeError("generator_params must be a Params instance, a mapping of its "
                        f"fields, or None; got {type(gp).__name__}")


@dataclass(frozen=True)
class TrackGenConfig:
    """Single configuration object passed to every stage of the pipeline.

    Fields configure the fixed-shape Warp pipeline.

    Per-generator parameters: each generator declares its own Params class
    (``track_gen.plugin.registry.describe("generator", name).params_schema``). Set
    them either through the flat fields below (the built-in generators' knobs, e.g.
    ``polar_num_knots``) or through ``generator_params`` (a Params instance or a mapping
    of its fields, e.g. ``generator_params={"polar_num_knots": 9}``), not both with
    different values. The generator-specific rules (``voronoi_control_points >= 6``,
    the ``repulsive_*`` rules, ...) are checked when the generator is constructed, and
    only for the selected generator.

    Attributes
    ----------
    generator : str
        Registered generator name.  One of ``"bezier"`` (default), ``"hull"``,
        ``"polar"``, ``"voronoi"``, ``"checkpoint"`` or ``"repulsive"``, or a name a
        plugin registered.  Resolved through ``track_gen.plugin.registry`` (kind
        ``"generator"``) at ``TrackGenerator`` construction time.
    device : str
        Warp device string: ``"cpu"`` (GPU-free, for tests/CI) or ``"cuda"``
        (captures a replay graph on first ``generate()`` call).
    num_envs : int
        Number of environments in the batch.  Must be >= 1.  Determines the
        leading dimension of all output arrays.

    min_num_points : int
        Minimum Bezier/hull corner-anchor count per env.  Must be >= 2.  Shared
        by the ``"bezier"`` and ``"hull"`` generators: each env's count is uniform on
        ``[min_num_points, max_num_points]``.  Dynamic: the initial
        value of the generator's per-env ``int32`` ``min_num_points``
        (``gen.params["generator"].set("min_num_points", k)`` checks ``[2,
        max_num_points]``; an array is clamped to it at use).  Raising it only
        narrows the count toward ``max_num_points``.
    max_num_points : int
        Maximum corner-anchor count per env.  Must be >= ``min_num_points``.  A
        wider range adds more per-env diversity in corner count.  Static (it sizes
        the corner buffers).
    num_points_per_segment : int
        Dense Bezier (or Catmull-Rom) samples emitted per corner-to-corner segment
        before arc-resampling to ``num_points``.  Must be >= 2; higher values give
        the arc-resampler a finer cumulative-arc-length grid, reducing interpolation
        error.
    min_point_distance : float
        Minimum grid-cell spacing for corner-position sampling.  Must be in
        ``(0, 0.5]``.  Governs the cell grid as
        ``num_cells = int(1 / (min_point_distance * 2))``.
    rad : float
        Cubic-Bezier handle length as a fraction of the segment chord — the primary
        curviness dial.  Typical range ``[0.2, 0.6]``.  Only live when
        ``rad <= handle_clamp_frac``; if ``rad`` exceeds the clamp,
        ``handle_clamp_frac`` binds every segment and ``rad`` has no effect (see
        ``handle_clamp_frac``).  Must be in ``[0, 1]``.  Dynamic:
        the initial value of the generator's per-env ``rad``
        (``gen.params["generator"]["rad"]``, ``randomize={"generator.rad": ...}``).
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
        polygon (~5% at 0.4), which XPBD re-rounds.  Must be >= 0.  Dynamic
        (per env, like ``rad``).

    hull_displacement : float
        *Hull generator only.*  Maximum per-edge radial midpoint displacement as a
        fraction of the centroid-to-midpoint distance.  At 0.0 the method collapses
        toward a plain angle-sorted loop; larger values (typical range ``[0.1, 0.4]``)
        bulge or pinch lobes for more racing-shape variety, at the cost of a higher
        downstream relaxation burden.  Must be in ``[0, 1]``; dynamic (per env).

    checkpoint_count : int
        *Checkpoint generator only.*  Number of radial checkpoints (``C``).  Must
        be >= 3.  More checkpoints produce wavier tracks (chicane count scales ~C);
        fewer produce calmer, rounder tracks.  CarRacing canonical default: 12.
    checkpoint_radius_min_frac : float
        Inner-radius fraction for checkpoint placement.  Checkpoint radii are drawn
        from ``U(checkpoint_radius_min_frac * R, R)`` where ``R = 1``.  Must be in
        ``[0, 1)``.  The CarRacing canonical value 0.33 gives the inlets.  Dynamic
        (per env): ``set`` and ``randomize=`` accept the declared bound ``1.0``
        (every checkpoint on the outer radius), and a per-env array value is clamped
        to ``[0, 1]``; construction requires a configured value in ``[0, 1)``.
    checkpoint_angle_jitter : float
        Angular jitter as a fraction of the per-checkpoint angular slot
        (``slot = 2*pi/C``).  Must be in ``[0, 1]`` (dynamic, per env); below 1 each
        checkpoint stays near its slot; above 0.5 adjacent checkpoints may swap
        angular order.  Default 0.55.
    checkpoint_turn_rate : float
        Maximum heading change (radians) per steering step.  Bounds path curvature;
        too large allows kinks, too small prevents tracking the inlets.  Must be in
        ``[0, pi]``; dynamic (per env).  Default 0.42.
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

    polar_num_knots : int
        *Polar generator only.*  Number of periodic cubic-spline control knots.
        More knots add higher-frequency shape variety.  Default 12.
    polar_radial_jitter : float
        *Polar generator only.*  Radial jitter magnitude for polar control knots, as
        a fraction of the base radius.  Must be in ``[0, 0.85]`` (keeps radii
        positive; a value outside is rejected, not clamped);
        larger values produce more eccentric track shapes.  Dynamic (per env).
        Default 0.60.
    polar_angular_jitter : float
        *Polar generator only.*  Angular jitter per polar knot, as a fraction of one
        knot spacing (``2*pi/K``), so the sorted order is preserved without a runtime
        sort.  Must be in ``[0, 0.45]`` (rejected outside, not clamped).  Larger values
        add more angular irregularity.  Dynamic (per env).  Default 0.30.

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
        Must be in ``[0, 0.85]``.  At 0, all targets are placed at a uniform annulus
        radius; larger values allow more eccentric loops.  Dynamic (per env).
        Default 0.62.
    voronoi_angular_jitter : float
        *Voronoi generator only.*  Angular jitter of the anchor sector targets, in
        radians (added to each target angle).  Must be in ``[0, pi]`` (useful up to
        ~0.5).  Larger values randomise the angular spacing of targets.  Dynamic
        (per env).  Default 0.08.

    repulsive_grow_mult_min : float
        *Repulsive generator only.*  Lower bound of the per-env target-perimeter
        multiplier draw ``U(min, max)``: the grown loop's target perimeter is
        ``grow_mult * 2π * r_init``.  Higher → denser mazes at some yield cost.

        .. warning::

            The ``repulsive`` generator is an iterative optimizer — about **two orders
            of magnitude slower** than ``bezier`` (its cost table is in
            :doc:`/generators/repulsive`).  It is
            graph-captured on CUDA like the other generators; its loop has a fixed
            iteration bound (its own graph on CUDA 12.4+ and eager ``cpu`` calls stop
            at the first stall window after every env has frozen; eager ``cuda`` calls
            and plain ``torch.cuda.graph`` captures run the full bound).  Regenerate on
            a **slow cadence**
            or in **staggered per-env slices**, not every frame.
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
        *Repulsive generator only.*  Weight of the secondary target-length gradient.
        Hard growth/length enforcement is separate from this soft correction.
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
        *Repulsive generator only.*  Iterations between per-env stall checks (an env
        whose area has stopped changing is frozen on device).
    repulsive_stall_area_tol : float
        *Repulsive generator only.*  Freeze an env once it is past target length AND
        its enclosed-area relative change over a stall window is below this
        (reparameterization-invariant shape-convergence tolerance).
    repulsive_deactivate_obstacles : bool
        *Repulsive generator only.*  When ``True`` (default), zero the inner-disc
        obstacle weights once an env reaches its target length (the wall stays live),
        closing the disc halos.

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
        The relax stage ``TrackGenerator`` runs: any registered ``relax`` stage
        (``track_gen.plugin.registry.available("relax")``; default ``"xpbd"``, the
        XPBD solver configured by the ``relax_*`` fields below; a plugin stage gets
        its ``Params.from_legacy(config)``). Resolved at ``TrackGenerator``
        construction, which raises ``ValueError`` listing the available names for an
        unknown one.  ``"energy"`` (Adam) and ``"tp_sobolev"`` are oracle-only
        backends in ``tests/_oracle/relaxation.py``, not relax stages:
        ``TrackGenerator`` rejects them.
    relax_tol : float
        Validity thickness tolerance.  The minimum valid track half-width is
        ``(1 - relax_tol) * half_width``.  Default 0.02 (2% tolerance).
    relax_band : int or None
        Bead-index exclusion band for the separation constraint.  ``None`` (default)
        auto-computes ``round(2 * half_width / L0)`` per track, where ``L0`` is the
        rest edge length.  Set explicitly to override the per-track auto-band.
    relax_iters : int
        Number of XPBD Jacobi sweeps.  Default 50.  Each sweep applies separation,
        spacing, and bending corrections; with the default Chebyshev-accelerated
        solve (``relax_accel="chebyshev"``), 50 sweeps are calibrated for ~0.999
        end-to-end valid yield at the library defaults (matching what 150 plain
        Jacobi sweeps used to need).
    relax_accel : str
        Iteration-acceleration scheme for the XPBD Jacobi sweeps.  ``"chebyshev"``
        (default) applies Chebyshev semi-iterative acceleration (Macklin et al.
        2014): after ``relax_cheby_start`` plain warmup sweeps, each sweep blends
        the Jacobi update with the current and previous iterates using a
        host-precomputed omega recurrence. Its benefit depends on the constraint system;
        ``"none"`` runs plain damped Jacobi.
    relax_cheby_rho : float
        Chebyshev spectral-radius estimate of the underlying Jacobi iteration.
        Must be in ``[0, 1)``.  Default 0.98.  Larger values accelerate the
        long-wavelength modes harder but risk overshoot on the nonsmooth
        separation contacts; values >= 0.99 were observed to reduce yield.
    relax_cheby_gamma : float
        Chebyshev under-relaxation factor applied to the Jacobi step inside the
        blend.  Must be in ``(0, 1]``.  Default 0.9.  Values below 1 damp the
        contact-switching noise that acceleration would otherwise amplify.
    relax_cheby_start : int
        Sweep index at which Chebyshev acceleration engages; earlier sweeps run
        plain Jacobi so the separation contact set settles first.  Must be >= 1.
        Default 8.
    relax_smooth_passes : int
        Number of shrink-free Taubin smoothing passes applied after the main sweep
        loop.  Must be >= 0.  Default 5.  Each pass is two Laplacian half-steps
        (:math:`lambda > 0` then :math:`mu < -lambda`), which removes the
        sub-``R_min`` curvature noise the bending guard's deadband cannot see (bending
        only fires below the minimum radius; gentler wiggles are invisible to all three
        constraints) without shrinking corners through the validity gate the way a plain
        Laplacian would.  0 disables the smoothing tail entirely.
    relax_smooth_spacing_iters : int
        Number of spacing-only polish sweeps applied after the Taubin passes.  Must be
        >= 0.  Default 10.  Each is the ordinary edge-spacing correction with separation
        and bending off, restoring bead uniformity that the Taubin passes perturb.  0
        disables the polish.

    relax_sep_every : int
        Broadphase refresh interval in sweeps.  Must be >= 1.  At 1, the dense
        ``O(count²)`` separation scan (or cache rebuild) runs every sweep.  At
        ``K > 1`` with ``relax_sep_cache_slots == 0``, the dense scan is skipped
        between refreshes (fast but may miss transient contacts).  With
        ``relax_sep_cache_slots > 0``, this is the cache-rebuild interval; the exact
        narrowphase still runs every sweep on cached candidates.  Default 20: the
        broadphase is built at sweep 0 and refreshed only at multiples of this
        interval, so runs shorter than ``relax_sep_every`` sweeps never refresh it
        mid-solve. Choose the cadence and skin together to retain candidate contacts.
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

    num_points : int
        Intermediate dense-resample resolution (``N``) used before the
        constant-spacing step.  Default 256.  The constant-spacing resampler then
        derives ``count[e] = floor(perimeter / spacing) + 1`` real points per env.
    spacing : float or None
        Constant arc-length step in coordinate units between adjacent output points.  ``None``
        (default) auto-couples to ``0.6 * half_width`` — the relaxation-friendly
        value; a fixed default would be wrong as ``half_width`` varies.  Must
        resolve to > 0 after auto-coupling.
    N_max : int
        Output buffer width: maximum constant-spacing points per track.  Tracks
        whose true ``count[e]`` exceeds ``N_max`` are truncated, flagged in
        ``Track.truncated`` and fail validity.
        Default 384. Budget at least ``floor(max_perimeter / spacing) + 1``
        slots, with headroom for the largest size and post-relaxation perimeter.
        Raise ``N_max`` or coarsen ``spacing`` for finer or larger courses.

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
        validity gate. Default ``False``. It tests proper nonadjacent segment
        crossings within each inner/outer border, adding two ``O(N²)`` passes.
        Thickness uses sampled bead distances and local curvature; it is not an
        exact segment-separation proof, so the optional check can add rejections.
        It does not test crossings between the inner and outer loops.

    z_profile : str
        Per-point altitude (Z) profile: the name of a registered ``elevation``
        stage (``track_gen.plugin.registry.available("elevation")``), resolved at
        ``TrackGenerator`` construction (an unknown name raises ``ValueError``
        listing the available ones; the config itself only checks it is a
        non-empty string).  The built-ins: ``"flat"`` (default, every real point
        at ``z_base``), ``"uniform"`` (i.i.d. uniform in ``[z_min, z_max]``),
        ``"random_walk"`` (Brownian-bridge walk clamped to ``[z_min, z_max]``
        with a per-step grade cap of ``z_max_step``), or ``"noise"`` (periodic
        harmonic noise around ``z_base``), applied along the resampled
        centerline's plan-view arc length.  A plugin profile gets its
        ``Params.from_legacy(config)`` (it may read the flat ``z_*`` fields).
    z_base : float
        Baseline altitude for the ``"flat"``, ``"random_walk"``, and
        ``"noise"`` profiles.  Default 0.0.
    z_min : float
        Lower altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be <= ``z_max``.  Default 0.0.  Dynamic (the
        elevation stage's per-env ``gen.params["elevation"]["z_min"]``).
    z_max : float
        Upper altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be >= ``z_min``.  Default 0.0.  Dynamic (per env).
    z_max_step : float
        *Random-walk profile only.*  Maximum ``|dz|`` per unit plan-view arc
        length (a grade cap), applied per random-walk step before the
        closure bridge.  Must be >= 0.  Default 0.0.  Dynamic (per env); inert
        in a closed window ``z_min == z_max``.
    z_noise_amplitude : float
        *Noise profile only.*  Amplitude of the harmonic altitude
        oscillation around ``z_base``.  Must be >= 0.  Default 0.0.  Dynamic
        (per env); clipped by the window: use a wide, symmetric one.
    z_noise_harmonics : int
        *Noise profile only.*  Number of summed harmonics in the periodic
        noise field.  Must be >= 1.  Default 3.
    z_control_points : int
        Number of arc-length-spaced altitude control knots.  Altitude is decided
        at these knots and interpolated between them with a periodic monotone
        cubic, so smoothness is independent of the resampled point count.  Must
        be >= 3.  Applies to ``z_profile="uniform"`` and ``"random_walk"``;
        inert for ``"flat"`` (constant) and ``"noise"`` (analytically smooth —
        use ``z_noise_harmonics`` for its frequency).  Default 10.
    z_valid_grade : float
        Maximum allowed ``|dz|``/ds grade for post-hoc course validity checking,
        where ``ds`` is measured along the PLAN-VIEW (XY) arc length, not the
        3D lifted arc length.  0 disables the check.  Must be >= 0.  Default 0.0.

    size : float
        Per-env course-size multiplier: the INITIAL value of the
        selected generator's dynamic ``size`` (``generator.size``; per-env values:
        ``gen.params["generator"]``, ``randomize={"generator.size": (lo, hi)}``),
        applied where the generator applies ``scale`` (bezier / hull: the corner
        grid, so it equals ``scale`` there at ``scale=1``; polar / voronoi /
        checkpoint: the normalization target; repulsive: its output).  It scales the
        plan view only: ``half_width``, ``spacing`` and every ``z_*`` stay absolute.
        Not the batch size (``num_envs``).  In ``[0.1, 10]``, default 1.0; a generator
        that declares no ``size`` (``external``, a plugin without the field) refuses
        a value other than 1.  The only construction-time route for ``Course``.  Size
        up with ``N_max`` (truncated envs fail validity).

    generator_params : Params, Mapping or None
        The selected generator's parameters.  ``None`` (default): the generator's
        ``Params.from_legacy(config)`` (for the built-ins, the flat fields above).  A
        Params instance of the generator's class, or a mapping of its fields (left-out
        keys take the class defaults; unknown keys raise listing the valid fields):
        the generator's complete parameter set.  A flat field the selected built-in
        owns must then stay at its default or equal the given value.  Switching
        generators means resetting it to ``None``.  Resolved and validated at
        ``TrackGenerator`` construction; ``TrackGenerator.config`` reports it
        normalised to the Params instance.
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


    # --- Width params ---
    half_width: float = 0.1

    # --- Relaxation: backend selection + scale ---
    relax_enable: bool = True
    relax_solver: str = "xpbd"            # {"xpbd","energy","tp_sobolev"}
    relax_tol: float = 0.02
    relax_band: int | None = None
    relax_iters: int = 50
    relax_accel: str = "chebyshev"        # {"none","chebyshev"}
    relax_cheby_rho: float = 0.98
    relax_cheby_gamma: float = 0.9
    relax_cheby_start: int = 8
    relax_smooth_passes: int = 5
    relax_smooth_spacing_iters: int = 10
    relax_sep_every: int = 20
    relax_sep_cache_slots: int = 16
    relax_sep_cache_skin: float = 0.5
    relax_sep_relax: float = 1.0
    relax_spc_relax: float = 1.0
    relax_bend_relax: float = 1.5
    relax_margin: float = 0.15


    # --- Output params ---
    num_points: int = 256  # N: intermediate dense->resample resolution before constant-spacing
    spacing: float | None = None
    N_max: int = 384

    # --- Robustness params ---
    turning_tol: float = 0.1
    w_floor: float = 1e-3
    validity_border_check: bool = False

    # --- Z-profile (altitude) params, shared with GateGenConfig via _validate_z_fields ---
    z_profile: str = "flat"
    z_base: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    z_max_step: float = 0.0
    z_noise_amplitude: float = 0.0
    z_noise_harmonics: int = 3
    z_control_points: int = 10
    z_valid_grade: float = 0.0

    # --- Per-env size (roadmap PR13d): the initial value of the dynamic size ---
    size: float = 1.0

    # --- Per-generator parameters (None | Params instance | mapping of its fields) ---
    generator_params: Any = None

    on_invalid: str = "poison"

    def __post_init__(self):
        _validate_invalid_policy(self)
        if int(self.num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs!r}")
        if float(self.half_width) <= 0.0:
            raise ValueError(f"half_width must be > 0, got {self.half_width!r}")
        _check_generator_params(self)
        # Generator-specific rules (point family, num_points_per_segment, voronoi_*,
        # repulsive_*, checkpoint_*) live in the generators' Params.validate() and run
        # at generator construction, for the selected generator only.
        if int(self.relax_sep_every) < 1:
            raise ValueError(f"relax_sep_every must be >= 1, got {self.relax_sep_every!r}")
        if int(self.relax_sep_cache_slots) < 0:
            raise ValueError(
                f"relax_sep_cache_slots must be >= 0, got {self.relax_sep_cache_slots!r}")
        if float(self.relax_sep_cache_skin) < 0.0:
            raise ValueError(
                f"relax_sep_cache_skin must be >= 0, got {self.relax_sep_cache_skin!r}")

        # Chebyshev-acceleration validation: rho is a spectral-radius estimate (< 1 or
        # the omega recurrence diverges), gamma in (0, 1] keeps the blended step a
        # damped Jacobi step, and start >= 1 guarantees at least one plain warmup
        # sweep before the previous-iterate blend engages.
        if str(self.relax_accel) not in ("none", "chebyshev"):
            raise ValueError(
                f"relax_accel must be 'none' or 'chebyshev', got {self.relax_accel!r}")
        if not (0.0 <= float(self.relax_cheby_rho) < 1.0):
            raise ValueError(
                f"relax_cheby_rho must be in [0, 1), got {self.relax_cheby_rho!r}")
        if not (0.0 < float(self.relax_cheby_gamma) <= 1.0):
            raise ValueError(
                f"relax_cheby_gamma must be in (0, 1], got {self.relax_cheby_gamma!r}")
        if int(self.relax_cheby_start) < 1:
            raise ValueError(
                f"relax_cheby_start must be >= 1, got {self.relax_cheby_start!r}")

        # Post-solve smoothing tail: both counts are simple non-negative sweep budgets
        # (0 disables). No cross-constraint between them — either may run alone.
        if int(self.relax_smooth_passes) < 0:
            raise ValueError(
                f"relax_smooth_passes must be >= 0, got {self.relax_smooth_passes!r}")
        if int(self.relax_smooth_spacing_iters) < 0:
            raise ValueError(
                "relax_smooth_spacing_iters must be >= 0, got "
                f"{self.relax_smooth_spacing_iters!r}")

        # Only constant_spacing is supported: a constant link SIZE (~0.6*half_width) relaxes
        # to smoother, higher-yield tracks than a constant point COUNT, which over-resolves
        # the centerline (jagged XPBD -> folded roads).
        # Auto-couple spacing to half_width (~0.6*half_width relaxes to smoother tracks);
        # a fixed spacing default would be wrong as half_width varies (too coarse -> degenerate).
        if self.spacing is None:
            object.__setattr__(self, "spacing", 0.6 * self.half_width)
        if float(self.spacing) <= 0.0:
            raise ValueError(f"spacing must be > 0, got {self.spacing!r}")

        if not isinstance(self.z_profile, str) or not self.z_profile:
            raise ValueError(f"z_profile must be a non-empty string (an elevation stage "
                             f"name), got {self.z_profile!r}")
        _validate_z_fields(self)
        if int(self.z_control_points) < 3:
            raise ValueError(
                "z_control_points must be >= 3, got "
                f"{self.z_control_points!r}")


@dataclass(frozen=True)
class GateGenConfig:
    """Configuration for fixed-batch native gate sequence generation.

    Gate generators sample anchors directly.

    Attributes
    ----------
    generator : str
        Registered gate generator name.  One of ``"bezier"`` (default), ``"hull"``,
        ``"polar"``, ``"voronoi"``, or ``"checkpoint"``, or a name a plugin registered
        (``track_gen.plugin.registry``, kind ``"gate_generator"``).  Selects the anchor
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
        Gate sphere radius: the sphere solve (``gate_relax.spheres``) separates gate
        centres to ``2 * gate_radius`` and ``validity.gate_default`` rejects real
        gates closer than that.  Must be >= 0.
    gate_solve_iters : int
        Maximum sphere-solve sweeps (0 skips the solve).  Must be >= 0.
    gate_width : float
        FULL physical gate bar width (visual/collision extent):
        ``GateSequence.half_size = 0.5 * gate_width``, the bar endpoints are
        ``position +/- half_size * left_axis``, and every env with a bar of non-zero
        half-size is bar-crossing checked.  Must be >= 0.  Does not affect anchor
        sampling geometry.  Dynamic: the initial value of
        ``gate_frames.default``'s per-env ``gate_width``
        (``gen.params["gate_frames"]``, ``randomize={"gate_frames.gate_width": ...}``).
    gate_ordering : str or None
        Gate output ordering: any registered ``order`` stage the generator supports
        (``track_gen.plugin.registry.available("order")``; checked at
        ``GateGenerator`` construction).  Built-ins: ``"ccw"`` (an angle sort around
        the env's anchor centroid, clockwise in a y-up frame),
        ``"raw"`` (the generator's own order) and ``"random_pairs"`` (random gate pairs
        for training objectives).  ``None`` (default): the generator's default order,
        resolved at ``GateGenerator`` (hence ``Course``) construction by
        ``legacy.to_gate_pipeline_spec`` with the rule of ``presets.gates(order=None)``
        and the conformance kit: its ``preferred["order"]`` when it accepts it, else
        ``"ccw"`` when it accepts it, else the first order it accepts.  That is
        ``"ccw"`` for the five built-ins, and ``"raw"`` for a generator that provides
        per-anchor attributes. The pipeline selects the resolved ordering Stage
        before building the gate generator's downstream stages.

    min_num_points : int
        Minimum corner-anchor count per env for the bezier/hull point sampler.
        Must be >= 2.  Affects the number of sampled anchor positions.
    max_num_points : int
        Maximum corner-anchor count per env.  Must be >= ``min_num_points``.
    min_point_distance : float
        Minimum grid-cell spacing for anchor-position sampling.  Must be > 0.
        Governs the cell grid as ``num_cells = int(1 / (min_point_distance * 2))``.
    scale : float
        Isotropic scale multiplier for anchor positions.  Does affect the sampled
        anchor coordinates and hence gate positions.
    polar_num_knots : int
        *Polar generator only.*  Number of periodic cubic-spline control knots.
        Default 12.
    polar_radial_jitter : float
        *Polar generator only.*  Radial jitter magnitude for polar control knots,
        as a fraction of the base radius.  Must be in ``[0, 0.85]``; dynamic (per
        env).  Default 0.60.
    polar_angular_jitter : float
        *Polar generator only.*  Angular jitter per polar knot, as a fraction of one
        knot spacing (``2*pi/K``).  Must be in ``[0, 0.45]``; dynamic (per env).
        Default 0.30.

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
        Must be in ``[0, 0.85]``; dynamic (per env).  Default 0.62.
    voronoi_angular_jitter : float
        *Voronoi generator only.*  Angular jitter of the anchor sector targets, in
        radians.  Must be in ``[0, pi]`` (useful up to ~0.5); dynamic (per env).
        Default 0.08.

    checkpoint_count : int
        *Checkpoint generator only.*  Number of radial checkpoints.  Must be >= 3.
        Default 12.
    checkpoint_radius_min_frac : float
        *Checkpoint generator only.*  Inner-radius fraction for checkpoint
        placement.  Must be in ``[0, 1)``; dynamic (per env; ``set`` /
        ``randomize=`` accept the declared bound ``1.0``).  Default 0.33.
    checkpoint_angle_jitter : float
        *Checkpoint generator only.*  Angular jitter as a fraction of the
        per-checkpoint angular slot.  Must be in ``[0, 1]``; dynamic (per env).
        Default 0.55.

    z_profile : str
        Per-gate altitude (Z) profile: any registered ``elevation`` stage (the same
        stages as tracks, their ports rewired to the gates; checked at
        ``GateGenerator`` construction).  Built-ins: ``"flat"`` (default, every real
        gate at ``z_base``), ``"uniform"`` (i.i.d. uniform in ``[z_min, z_max]``),
        ``"random_walk"`` (Brownian-bridge walk clamped to ``[z_min, z_max]`` with a
        per-step grade cap of ``z_max_step``), or ``"noise"`` (periodic harmonic noise
        around ``z_base``).  The gate pipeline's ``elevation`` stage, after the XY
        ordering/relaxation and before the 3D lift.
    z_base : float
        Baseline altitude for the ``"flat"``, ``"random_walk"``, and
        ``"noise"`` profiles.  Default 0.0.
    z_min : float
        Lower altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be <= ``z_max``.  Default 0.0.  Dynamic (per env, the
        elevation stage's ``gen.params["elevation"]["z_min"]``).
    z_max : float
        Upper altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be >= ``z_min``.  Default 0.0.  Dynamic (per env).
    z_max_step : float
        *Random-walk profile only.*  Maximum ``|dz|`` per unit plan-view arc
        length (a grade cap), applied per random-walk step before the
        closure bridge.  Must be >= 0.  Default 0.0.  Dynamic (per env); inert
        in a closed window ``z_min == z_max``.
    z_noise_amplitude : float
        *Noise profile only.*  Amplitude of the harmonic altitude
        oscillation around ``z_base``.  Must be >= 0.  Default 0.0.  Dynamic
        (per env); inert in a closed window ``z_min == z_max``: use a wide,
        symmetric one.
    z_noise_harmonics : int
        *Noise profile only.*  Number of summed harmonics in the periodic
        noise field.  Must be >= 1.  Default 3.
    z_valid_grade : float
        Maximum allowed ``|dz|``/ds chord grade for the course validity check
        (``validity.gate_default``).  0 disables the check.  Must be >= 0.
        Default 0.0.
    gate_align : str
        Gate-frame alignment mode.  ``"yaw_only"`` (default) keeps gate posts
        horizontal (the frame forward axis is the plan-view tangent projected
        onto the XY plane, so left/right posts share the center's altitude);
        ``"full_tangent"`` aligns the frame forward axis with the full 3D
        tangent, tilting posts to follow the course slope.

    size : float
        Per-env course-size multiplier: the INITIAL value of the
        dynamic ``size`` of the stage that holds it: ``generator.size`` for the
        ``"bezier"`` / ``"hull"`` gate generators (the anchor grid, after ``scale``),
        ``normalize.size`` for ``"polar"`` / ``"voronoi"`` / ``"checkpoint"`` (the
        normalization target).  It scales the gate layout's plan view only:
        ``gate_radius``, ``gate_width`` and the ``z_*`` fields stay absolute.  Not the
        batch size (``num_envs``).  In ``[0.1, 10]``, default 1.0; a pipeline with no
        stage declaring ``size`` refuses a value other
        than 1.

    generator_params : Params, Mapping or None
        The selected gate generator's parameters, as
        ``TrackGenConfig.generator_params`` (``None``: the flat fields above).  The
        generator-specific rules (point family, ``voronoi_*``, ``checkpoint_*``)
        are checked at ``GateGenerator`` construction, for the selected generator
        only; ``max_gates`` sees the resolved values.
    """

    generator: str = "bezier"
    device: str = "cpu"
    num_envs: int = 1

    min_gates: int = 4
    max_gates: int = 32
    gate_radius: float = 0.025
    gate_solve_iters: int = 8
    gate_width: float = 0.0
    gate_ordering: "str | None" = None

    # Point-family (bezier/hull) anchor sampler inputs.
    min_num_points: int = 9
    max_num_points: int = 13
    min_point_distance: float = 0.05
    scale: float = 1.0

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

    z_profile: str = "flat"
    z_base: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    z_max_step: float = 0.0
    z_noise_amplitude: float = 0.0
    z_noise_harmonics: int = 3
    z_valid_grade: float = 0.0
    gate_align: str = "yaw_only"

    # --- Per-env size (roadmap PR13d): the initial value of the dynamic size ---
    size: float = 1.0

    # --- Per-generator parameters (None | Params instance | mapping of its fields) ---
    generator_params: Any = None

    on_invalid: str = "poison"

    def __post_init__(self):
        _validate_invalid_policy(self)
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
        if self.gate_ordering is not None and (not isinstance(self.gate_ordering, str)
                                               or not self.gate_ordering):
            raise ValueError(f"gate_ordering must be a non-empty string (an order stage "
                             f"name) or None (the generator's default order), got "
                             f"{self.gate_ordering!r}")
        _check_generator_params(self)
        # Generator-specific rules (point family, voronoi_*, checkpoint_*) live in the
        # gate generators' Params.validate() and run at GateGenerator construction.
        if not isinstance(self.z_profile, str) or not self.z_profile:
            raise ValueError(f"z_profile must be a non-empty string (an elevation stage "
                             f"name), got {self.z_profile!r}")
        _validate_z_fields(self)
        if self.gate_align not in {"yaw_only", "full_tangent"}:
            raise ValueError(
                "gate_align must be one of {'yaw_only', 'full_tangent'}, got "
                f"{self.gate_align!r}"
            )


# ---------------------------------------------------------------------------------------
# Layout and the result schemas (roadmap PR13b)
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """The shape of a batch: the envs, the per-env row capacity, the device and the
    topology. Handed to :meth:`Params.validate <track_gen.plugin.Params.validate>`,
    :meth:`Stage.capacity <track_gen.plugin.Stage.capacity>` and the
    :class:`~track_gen.plugin.StageBuilder`, and carried by every result
    (``Track.layout``, ``GateSequence.layout``, ``Centerline.layout``).

    ``track_gen.results.Layout``, ``track_gen.Layout`` and
    ``track_gen.plugin.Layout`` are this class.

    Attributes:
        num_envs: ``E`` (an int ``>= 1``; numpy integers are accepted).
        stride: The per-env row capacity (an int ``>= 1``): ``N_max`` for a ``Track``
            (and a track stage), ``max_gates`` for a ``GateSequence`` (and a gate
            stage), and for a ``Centerline`` its own capacity (``max_gates *
            samples_per_gate`` for a gates-mode ``Course``'s
            ``CourseLine.centerline``). A channel's own row stride is
            :meth:`StageBuilder.stride_of <track_gen.plugin.StageBuilder.stride_of>`.
        device: The device string, stored AS GIVEN (``"cuda"`` stays ``"cuda"``). The
            layouts TrackGen makes (the planner, ``alloc``, ``view``, an inferred one)
            store the canonical ``str(wp.get_device(d))`` (``"cuda:0"``), so ``==``
            holds between them; the library's own checks compare the canonical device,
            so a hand-set ``Layout(E, S, "cuda")`` matches ``"cuda:0"``.
        closed: Whether the loops are closed. Only ``True`` is supported in 0.x (a
            ``BatchSpec(closed=False)`` is refused by the planner, a source with
            ``closed=False`` by every utility).

    Raises:
        TypeError: a non-integer (or bool) ``num_envs`` / ``stride``, a non-str
            ``device``, a non-bool ``closed``.
        ValueError: ``num_envs < 1`` or ``stride < 1``.
    """

    num_envs: int
    stride: int
    device: str
    closed: bool = True

    def __post_init__(self) -> None:
        for name in ("num_envs", "stride"):
            v = getattr(self, name)
            if isinstance(v, bool):
                raise TypeError(f"Layout.{name} must be an int, got {v!r}")
            try:
                iv = operator.index(v)
            except TypeError:
                raise TypeError(f"Layout.{name} must be an int, got "
                                f"{type(v).__name__}") from None
            if iv < 1:
                raise ValueError(f"Layout.{name} must be >= 1, got {iv}")
            object.__setattr__(self, name, int(iv))
        if not isinstance(self.device, str):
            raise TypeError(f"Layout.device must be a device string, got "
                            f"{type(self.device).__name__}")
        if not isinstance(self.closed, bool):
            raise TypeError(f"Layout.closed must be a bool, got {self.closed!r}")


def _canonical_device(device: str) -> str:
    return str(wp.get_device(device))


def _layout_key(layout: Layout) -> tuple:
    """The library's comparison key of a layout: the canonical device (evaluated now)."""
    return (int(layout.num_envs), int(layout.stride), _canonical_device(layout.device),
            bool(layout.closed))


def _fmt_layout(layout: Layout) -> str:
    return (f"Layout({layout.num_envs}, {layout.stride}, {layout.device!r}, "
            f"{layout.closed})")


def _dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype))


# (field, per, dtype), in dataclass field order; per: "point" ([E*stride]) or "env" ([E]).
_TRACK_SCHEMA = (
    ("outer", "point", wp.vec3f), ("center", "point", wp.vec3f),
    ("inner", "point", wp.vec3f), ("tangent", "point", wp.vec3f),
    ("normal", "point", wp.vec3f), ("arclen", "point", wp.float32),
    ("length", "env", wp.float32), ("valid", "env", wp.int32),
    ("count", "env", wp.int32), ("winding", "env", wp.float32),
    ("truncated", "env", wp.int32), ("generation", "env", wp.int32),
    ("fail_mask", "env", wp.int32), ("half_width", "point", wp.float32),
)
_GATE_SCHEMA = (
    ("position", "point", wp.vec3f), ("tangent", "point", wp.vec3f),
    ("forward", "point", wp.vec3f), ("orientation", "point", wp.quatf),
    ("half_size", "point", wp.float32), ("left", "point", wp.vec3f),
    ("right", "point", wp.vec3f), ("valid", "env", wp.int32),
    ("count", "env", wp.int32), ("generation", "env", wp.int32),
    ("fail_mask", "env", wp.int32),
)
_CENTERLINE_SCHEMA = (
    ("center", "point", wp.vec3f), ("tangent", "point", wp.vec3f),
    ("arclen", "point", wp.float32), ("length", "env", wp.float32),
    ("count", "env", wp.int32), ("valid", "env", wp.int32),
    ("generation", "env", wp.int32),
)
_CENTERLINE_OPTIONAL = frozenset({"generation"})
# Every result member by name (one per-domain / dtype per name across the three schemas):
# what track_gen.results.check_source checks a protocol member against.
_MEMBER_SCHEMA = {name: (per, dt) for schema in (_TRACK_SCHEMA, _GATE_SCHEMA,
                                                 _CENTERLINE_SCHEMA)
                  for name, per, dt in schema}
# Roadmap PR14: a CheckpointSet's per-checkpoint vertical half-opening
# (track_gen.results.HasCheckpoints; the set's other members share the names above).
_MEMBER_SCHEMA["up_half"] = ("point", wp.float32)
# The non-array metadata fields of the result dataclasses.
_RESULT_METADATA = ("fail_bits", "attributes", "layout")
# The attribute dtypes (Track.attributes / GateSequence.attributes).
_ATTR_DTYPES = (wp.float32, wp.int32, wp.vec2f, wp.vec3f)
_ATTR_DTYPES_TEXT = "float32 / int32 / vec2f / vec3f"


def _form(per: str, E: int, S: int) -> str:
    return f"[E*stride] = [{E}*{S}]" if per == "point" else f"[E] = [{E}]"


def _got(arr) -> str:
    if not isinstance(arr, wp.array):
        return type(arr).__name__
    extra = "" if arr.is_contiguous else " (not contiguous)"
    return f"shape {tuple(int(s) for s in arr.shape)} {_dtype_name(arr.dtype)} on " \
           f"{arr.device}{extra}"


def _array_ok(arr, n: int, dtype, dev: str) -> bool:
    return isinstance(arr, wp.array) and arr.ndim == 1 and int(arr.shape[0]) == n \
        and (dtype is None or arr.dtype is dtype) and str(arr.device) == dev \
        and arr.is_contiguous


def _check_flat(arr, owner: str, name: str, per: str) -> None:
    """A result array is flat 1-D storage: name a multi-dimensional one plainly (the
    natural ``[E, S]`` user layout) instead of reporting a shape mismatch."""
    if isinstance(arr, wp.array) and arr.ndim != 1:
        form = "[E*stride]" if per == "point" else "[E]"
        raise ValueError(f"{owner}.{name}: expected a flat 1-D {form} array, got a "
                         f"{arr.ndim}-D array of shape {tuple(int(s) for s in arr.shape)} "
                         f"(flatten it, e.g. arr.reshape((-1,)))")


def _infer_layout(obj, owner: str, schema, prefer: str) -> Layout:
    """``obj``'s layout from the members of ``schema`` only (``(name, per, dtype)``):
    ``E`` from ``count`` when it is one of them (else from the first per-env member),
    ``stride`` from the first per-point member (``prefer`` first; ``1`` when there is
    none: the stride is then undefined), ``closed=True``, the device of the member that
    gave ``E``."""
    names = [n for n, _, _ in schema]
    env_names = (["count"] if "count" in names else []) + \
        [n for n, per, _ in schema if per == "env" and n != "count"]
    for n in env_names:
        _check_flat(getattr(obj, n, None), owner, n, "env")
    key = next((n for n in env_names if isinstance(getattr(obj, n, None), wp.array)), None)
    if key is None:
        want = env_names[0] if env_names else "count"
        raise ValueError(f"{owner}.{want}: expected an [E] wp.array (it gives num_envs), "
                         f"got {type(getattr(obj, want, None)).__name__}")
    ref = getattr(obj, key)
    if int(ref.shape[0]) < 1:
        raise ValueError(f"{owner}.{key}: expected a non-empty [E] array, got "
                         f"{_got(ref)}")
    E = int(ref.shape[0])
    points = [n for n, per, _ in schema if per == "point"]
    order = ([prefer] if prefer in points else []) + [n for n in points if n != prefer]
    for n in order:
        _check_flat(getattr(obj, n, None), owner, n, "point")
    anchor = next((n for n in order if isinstance(getattr(obj, n, None), wp.array)), None)
    S = 1
    if anchor is not None:
        arr = getattr(obj, anchor)
        n = int(arr.shape[0])
        if n < E or n % E:
            raise ValueError(f"{owner}.{anchor}: {n} slots is not a multiple of "
                             f"num_envs={E} ({key} has {E} entries)")
        S = n // E
    return Layout(E, S, str(ref.device), True)


def _check_members(obj, owner: str, layout: Layout, members, *, need=(),
                   consumer: "str | None" = None) -> None:
    """Check each of ``members`` (``(name, per, dtype)``) of ``obj`` against ``layout``:
    a 1-D contiguous array of its dtype, ``[E*stride]`` or ``[E]``, on the layout's
    (canonical) device. ``None`` passes unless the member is in ``need``."""
    E, S = int(layout.num_envs), int(layout.stride)
    dev = _canonical_device(layout.device)
    for name, per, dtype in members:
        arr = getattr(obj, name, None)
        if arr is None:
            if name in need:
                who = consumer or "this consumer"
                raise ValueError(f"{owner}.{name} is None; {who} needs it")
            continue
        _check_flat(arr, owner, name, per)
        n = E * S if per == "point" else E
        if not _array_ok(arr, n, dtype, dev):
            raise ValueError(f"{owner}.{name}: expected a contiguous [{n}] "
                             f"{_dtype_name(dtype)} ({_form(per, E, S)}) array on {dev}, "
                             f"got {_got(arr)}")


def _check_attributes(obj, owner: str, layout: Layout) -> None:
    attrs = getattr(obj, "attributes", None)
    if attrs is None:
        return
    if not isinstance(attrs, Mapping):
        raise ValueError(f"{owner}.attributes: expected a mapping of wp.array, got "
                         f"{type(attrs).__name__}")
    n = int(layout.num_envs) * int(layout.stride)
    dev = _canonical_device(layout.device)
    for key, arr in attrs.items():
        if not _array_ok(arr, n, None, dev) or arr.dtype not in _ATTR_DTYPES:
            raise ValueError(f"{owner}.attributes[{key!r}]: expected a contiguous [{n}] "
                             f"array of {_ATTR_DTYPES_TEXT} on {dev}, got {_got(arr)}")


def _stored_layout(obj, owner: str) -> "Layout | None":
    lay = getattr(obj, "layout", None)
    if lay is not None and not isinstance(lay, Layout):
        raise TypeError(f"{owner}.layout must be a track_gen.Layout or None, got "
                        f"{type(lay).__name__}")
    return lay


def _require_result_storage(self, owner=None) -> None:
    """Reject missing concrete result storage without inspecting or touching arrays."""
    owner = owner or type(self)._RESULT_NAME
    for name in (*[n for n, _, _ in type(self)._SCHEMA], "layout"):
        if getattr(self, name, None) is None:
            raise ValueError(f"{owner}.{name} is None; required result storage")


def _validate(self, need=()) -> Layout:
    """Check every field of this result against its layout; return the layout.

    Track and GateSequence require every array field and a stored :attr:`layout`.
    A Centerline without a layout infers one (``num_envs`` from ``count``, ``stride``
    from the first per-point field, the device of ``count``, ``closed=True``;
    nothing is mutated). Every array field present must be a contiguous 1-D array
    of its dtype, ``[E * stride]`` (per point) or
    ``[E]`` (per env), on the layout's device (compared canonically: ``"cuda"`` ==
    ``"cuda:0"``), and every ``attributes`` entry a contiguous ``[E * stride]``
    float32 / int32 / vec2f / vec3f array on that device. Host only (a few
    microseconds): no device work, safe inside a capture.

    Args:
        need: Field names that must not be ``None`` (for an optional Centerline
            field a consumer reads). Track and GateSequence require all arrays.

    Returns:
        The stored :class:`~track_gen.results.Layout`, or an inferred Centerline layout.

    Raises:
        ValueError: the first field that does not match, naming it and the expected
            shape (e.g. ``Track.inner: expected a contiguous [3072] vec3f
            ([E*stride] = [8*384]) array on cpu, got shape (3064,) vec3f on cpu``); a
            required Track/GateSequence array or layout that is ``None``; a field
            of ``need`` that is ``None``; a ``count`` that is not an array.
        TypeError: :attr:`layout` is neither a ``Layout`` nor ``None``.
    """
    return _validate_against(self, need, None, None)


def _validate_against(self, need, _owner: "str | None",
                      _layout: "Layout | None", _consumer: "str | None" = None) -> Layout:
    """:meth:`validate`, naming the result ``_owner`` in messages (``"out"`` for the
    planner's ``out=`` check) and against ``_layout`` (the Program's) when given: a
    stored layout must then equal it (canonical device). ``_consumer`` names who needs
    a ``need`` field that is ``None``."""
    cls = type(self)
    owner = _owner or cls._RESULT_NAME
    if getattr(cls, "_REQUIRED_STORAGE", False):
        _require_result_storage(self, owner)
    if isinstance(need, str):
        need = (need,)
    stored = _stored_layout(self, owner)
    if _layout is not None and stored is not None and \
            _layout_key(stored) != _layout_key(_layout):
        raise ValueError(f"{owner}: a {cls._RESULT_NAME} for {_fmt_layout(stored)}; this "
                         f"pipeline writes {_fmt_layout(_layout)}")
    lay = _layout if _layout is not None else stored
    if lay is None:
        lay = _infer_layout(self, owner, cls._SCHEMA, cls._ANCHOR)
    unknown = [n for n in need if n not in {f for f, _, _ in cls._SCHEMA}]
    if unknown:
        raise ValueError(f"{cls._RESULT_NAME}.validate(need=...): unknown fields "
                         f"{unknown}")
    if not isinstance(getattr(self, "count", None), wp.array):
        raise ValueError(f"{owner}.count: expected an [E] int32 wp.array, got "
                         f"{type(getattr(self, 'count', None)).__name__}")
    _check_members(self, owner, lay, cls._SCHEMA, need=tuple(need), consumer=_consumer)
    _check_attributes(self, owner, lay)
    return lay


def _check_env_range(name: str, e0, e1, E: int) -> "tuple[int, int]":
    for v in (e0, e1):
        if isinstance(v, bool):
            raise TypeError(f"{name}.view(): env indices must be ints, got {v!r}")
    try:
        a, b = operator.index(e0), operator.index(e1)
    except TypeError:
        raise TypeError(f"{name}.view(): env indices must be ints, got {e0!r}, "
                        f"{e1!r}") from None
    if not (0 <= a <= E and 0 <= b <= E):
        raise IndexError(f"{name}.view({a}, {b}): env range outside [0, {E})")
    if a >= b:
        raise IndexError(f"{name}.view({a}, {b}): empty env range: need e0 < e1")
    return int(a), int(b)


def _view(self, e0: int, e1: int):
    """A zero-copy view of envs ``[e0, e1)`` of this result.

    Every array is a slice of this result's array (per point ``[e0 * stride : e1 *
    stride]``, per env ``[e0 : e1]``): the same memory, read and written in place, and
    kept alive by the view. ``None`` fields stay ``None``; ``attributes`` are sliced
    alike; ``fail_bits`` is shared; ``layout`` is ``Layout(e1 - e0, stride, device,
    closed)``. A view is a result like any other: every TrackGen kernel indexes it from
    its own element 0, so it can be generated into (``TrackGenerator(..., out=
    big.view(0, 4))``: mixed batches), consumed by the utilities, copied, cloned and
    viewed again. No device work and no allocation: callable inside a capture.

    Raises:
        IndexError: unless ``0 <= e0 < e1 <= num_envs``.
        TypeError: non-integer indices.
        ValueError: this result does not validate (:meth:`validate`).
    """
    lay = self.validate()
    E, S = int(lay.num_envs), int(lay.stride)
    a, b = _check_env_range(type(self)._RESULT_NAME, e0, e1, E)
    kw = {}
    for name, per, _ in type(self)._SCHEMA:
        arr = getattr(self, name, None)
        if arr is None:
            kw[name] = None
        elif per == "point":
            kw[name] = arr[a * S:b * S]
        else:
            kw[name] = arr[a:b]
    if hasattr(self, "attributes"):
        kw["attributes"] = types.MappingProxyType(
            {k: v[a * S:b * S] for k, v in dict(self.attributes or {}).items()})
    if hasattr(self, "fail_bits"):
        kw["fail_bits"] = self.fail_bits
    kw["layout"] = Layout(b - a, S, _canonical_device(lay.device), lay.closed)
    out = dataclasses.replace(self, **kw)
    out._env_offset = int(getattr(self, "_env_offset", 0)) + a
    out._producer_stream = getattr(self, "_producer_stream", None)
    return out


def _span(arr) -> "tuple[int, int]":
    p = int(arr.ptr or 0)
    return p, p + int(arr.shape[0]) * int(wp.types.type_size_in_bytes(arr.dtype))


def _copy_pairs(self, src) -> list:
    """The ``(name, dst, src)`` copies of ``self.copy_from(src)`` (checked)."""
    cls = type(self)
    name = cls._RESULT_NAME
    pairs = []
    for f, _, _ in cls._SCHEMA:
        d = getattr(self, f, None)
        if d is None:
            continue
        s = getattr(src, f, None)
        if s is None:
            raise ValueError(f"{name}.copy_from: src.{f} is None, but this {name} holds "
                             f"{f}")
        pairs.append((f, d, s))
    mine = dict(getattr(self, "attributes", None) or {})
    theirs = dict(getattr(src, "attributes", None) or {})
    for k, d in mine.items():
        s = theirs.get(k)
        if s is None:
            raise ValueError(f"{name}.copy_from: src has no attribute {k!r} "
                             f"(attributes: {sorted(theirs)})")
        if s.dtype is not d.dtype:
            raise ValueError(f"{name}.copy_from: attributes[{k!r}] is "
                             f"{_dtype_name(d.dtype)} here, {_dtype_name(s.dtype)} in "
                             f"src")
        pairs.append((f"attributes[{k!r}]", d, s))
    out = []
    for f, d, s in pairs:
        if d is s:
            continue
        (d0, d1), (s0, s1) = _span(d), _span(s)
        if (d0, d1) == (s0, s1):
            continue                       # the same memory: nothing to copy
        if d0 < s1 and s0 < d1:
            raise ValueError(f"{name}.copy_from: {f} partially overlaps src.{f} (a view "
                             f"of the same memory at another env offset)")
        out.append((f, d, s))
    return out


def _copy_from(self, src, *, stream=None, ctx=None) -> None:
    """Copy every field this result holds from ``src`` (same class, same layout).

    One stream-ordered device copy per held field (a field ``None`` here is skipped)
    and per ``attributes`` key of this result (``src`` may hold more keys). No
    allocation and no host sync, so it is the hot-path form of a snapshot (copy into a
    preallocated :meth:`alloc` result) and it records into a caller's capture (memcpy
    nodes). Host metadata (``fail_bits``, ``layout``) is not copied. ``t.copy_from(t)``
    is a no-op.

    The copies run on ``ctx``, else ``stream=`` > an ambient ``LaunchContext.scope`` >
    the stream that produced ``src`` (the generator's constructor ``stream=``) > an
    enclosing torch capture > Warp's current stream. ``copy_from`` only knows the
    producer's DEFAULT stream: pass the stream you generated on (or order the streams
    with events) when a generation ran elsewhere.

    Args:
        src: A result of the same class (``Track`` / ``GateSequence`` /
            ``Centerline``).
        stream: Optional ``wp.Stream`` (or ``"torch"``).
        ctx: Optional :class:`~track_gen.exec.LaunchContext` (not with ``stream=``).

    Raises:
        TypeError: ``src`` is not of this result class; ``stream=`` and ``ctx=``
            together.
        ValueError: either side does not validate; the layouts differ (canonical
            device); a field held here is ``None`` in ``src``; an attribute key here is
            missing from ``src`` or has another dtype; a partial overlap of a field
            with its source.
    """
    from ..exec.context import LaunchContext  # noqa: PLC0415

    cls = type(self)
    base = cls._RESULT_BASE
    name = cls._RESULT_NAME
    if not isinstance(src, base):
        raise TypeError(f"{name}.copy_from: src must be a {name}, got "
                        f"{type(src).__name__}")
    if stream is not None and ctx is not None:
        raise TypeError("pass either stream= or ctx=, not both")
    a = self.validate()
    b = src.validate()
    if _layout_key(a) != _layout_key(b):
        raise ValueError(f"{name}.copy_from: layouts differ: this {name} is "
                         f"{_fmt_layout(a)}, src is {_fmt_layout(b)}")
    pairs = _copy_pairs(self, src)
    if ctx is None:
        ctx = LaunchContext.for_call(_canonical_device(a.device), stream,
                                     default=getattr(src, "_producer_stream", None))
    for _, d, s in pairs:
        ctx.copy(d, s)


def _refuse_capture(dev, ctxs, what: str, message: "str | None" = None) -> None:
    """``CaptureError`` when an allocation on Warp's current stream (or the work of
    ``ctxs``) would land in a capture (``message``: the error text)."""
    from ..exec import _compat  # noqa: PLC0415
    from ..exec.context import _torch_capturing  # noqa: PLC0415
    from ..exec.errors import CaptureError  # noqa: PLC0415

    capturing = any(c.capturing for c in ctxs)
    if dev.is_cuda:
        capturing = capturing or wp.get_stream(dev).is_capturing or \
            _torch_capturing(dev) is not None
    else:
        capturing = capturing or _compat.cpu_capture_active(dev)
    if capturing:
        raise CaptureError(message or (
            f"{what} allocates on Warp's current stream: call it outside any capture "
            f"(inside one, copy_from into a preallocated result, e.g. from "
            f"{what.split('.')[0]}.alloc)"))


def _empty_like(self, lay: Layout, dev):
    kw = {}
    E, S = int(lay.num_envs), int(lay.stride)
    for name, per, dtype in type(self)._SCHEMA:
        arr = getattr(self, name, None)
        kw[name] = None if arr is None else wp.empty(E * S if per == "point" else E,
                                                     dtype=dtype, device=dev)
    if hasattr(self, "attributes"):
        kw["attributes"] = {k: wp.empty(E * S, dtype=v.dtype, device=dev)
                            for k, v in dict(self.attributes or {}).items()}
    if hasattr(self, "fail_bits"):
        kw["fail_bits"] = self.fail_bits
    kw["layout"] = Layout(E, S, str(dev), lay.closed)
    return dataclasses.replace(self, **kw)


def _clone(self, *, stream=None):
    """A fully-owned deep copy (new memory for every held field and attribute).

    COLD: allocates and blocks every call (use :meth:`copy_from` into a preallocated
    :meth:`alloc` result on a hot path). Allocated and copied on Warp's CURRENT stream
    (where ``wp.empty`` allocates), then a host wait on that stream: the clone is
    complete when it returns, so it is safe to read or write on ANY stream right away
    (``.numpy()``, a utility, ``copy_from`` on a generator's stream). When the result
    was produced on another stream (``stream=``, else an ambient scope, else the
    generator's constructor ``stream=``), Warp's current stream first waits on it.
    ``None`` fields stay ``None``, ``attributes`` are copied, ``fail_bits`` is shared
    (an immutable mapping), ``layout`` is set; the class (a subclass) is kept.

    Args:
        stream: The stream that PRODUCED this result (``wp.Stream`` or ``"torch"``),
            e.g. ``track = gen.generate(stream=s, sync=False);
            snap = track.clone(stream=s)``.

    Raises:
        track_gen.exec.CaptureError: inside any capture (Warp CPU / CUDA or torch),
            before allocating; inside a graph, ``copy_from`` into a preallocated
            :meth:`alloc` result instead.
        ValueError: this result does not validate (a malformed hand-built result).
    """
    from ..exec import _compat  # noqa: PLC0415
    from ..exec.context import LaunchContext  # noqa: PLC0415

    lay = self.validate()
    dev = wp.get_device(lay.device)
    producer = LaunchContext.for_call(dev, stream,
                                      default=getattr(self, "_producer_stream", None))
    current = LaunchContext.current(dev)
    _refuse_capture(dev, (producer, current), f"{type(self)._RESULT_NAME}.clone()")
    out = _empty_like(self, lay, dev)
    if dev.is_cuda and producer.stream is not None and current.stream is not None and \
            int(producer.stream.cuda_stream or 0) != int(current.stream.cuda_stream or 0):
        current.stream.wait_stream(producer.stream)
    out.copy_from(self, ctx=current)
    if dev.is_cuda:
        # The memory was handed out in the current stream's order: complete the copies
        # (and that stream's earlier work on recycled blocks) before any other stream
        # touches the clone.
        _compat.stream_synchronize(current.stream)
    return out


# The padding of an allocated result: per point NaN (int32: -1), per env 0.
def _padding(per: str, dtype):
    nan = float("nan")
    if per == "env":
        return 0 if dtype is wp.int32 else 0.0
    if dtype is wp.int32:
        return -1
    if dtype is wp.float32:
        return nan
    if dtype is wp.quatf:
        return wp.quatf(nan, nan, nan, nan)
    if dtype is wp.vec2f:
        return wp.vec2f(nan, nan)
    return wp.vec3f(nan, nan, nan)


def _alloc(cls, layout: Layout, *, attributes=None):
    """A result with every field allocated, in the "no track yet" state.

    Per-point fields (and attributes) hold their padding (NaN; ``-1`` for int32),
    ``count`` / ``valid`` / ``truncated`` / ``generation`` / ``fail_mask`` are 0,
    ``length`` / ``winding`` 0; ``layout`` is ``layout`` with the canonical device;
    ``fail_bits`` is empty. Allocation and fills run on Warp's CURRENT stream (like
    ``wp.full``), then a host wait on it: the result is complete when it returns, so it
    can be read or written on ANY stream right away (``snap.copy_from(track)`` on a
    generator's stream, ``TrackGenerator(..., out=big.view(0, 4), stream=s)``).
    COLD: allocates and blocks.

    Args:
        layout: The :class:`~track_gen.results.Layout` (``num_envs``, ``stride``,
            ``device``, ``closed``).
        attributes: Optional ``{key: dtype}`` of ``attributes`` to allocate (``Track``
            and ``GateSequence``): ``key`` a plugin key ``"<package>.<name>"``, ``dtype``
            ``wp.float32``, ``wp.int32``, ``wp.vec2f`` or ``wp.vec3f`` (or its name,
            ``"float32"``, as in ``ExternalParams.attributes``).

    Raises:
        TypeError: ``layout`` is not a ``Layout``; an attribute dtype outside the
            four; ``attributes`` on a ``Centerline``.
        ValueError: an attribute key that is not ``"<package>.<name>"``.
        track_gen.exec.CaptureError: inside a capture (before allocating).
    """
    from ..exec import _compat  # noqa: PLC0415
    from ..exec.context import LaunchContext  # noqa: PLC0415
    from ..stages.channels import is_plugin_channel  # noqa: PLC0415

    if not isinstance(layout, Layout):
        raise TypeError(f"{cls._RESULT_NAME}.alloc: layout must be a track_gen.Layout, "
                        f"got {type(layout).__name__}")
    attrs = dict(attributes or {})
    has_attrs = "attributes" in {f.name for f in dataclasses.fields(cls)}
    if attrs and not has_attrs:
        raise TypeError(f"{cls._RESULT_NAME}.alloc: a {cls._RESULT_NAME} has no "
                        f"attributes")
    by_name = {_dtype_name(dt): dt for dt in _ATTR_DTYPES}
    for k, dt in list(attrs.items()):
        if not isinstance(k, str) or not k:
            raise TypeError(f"{cls._RESULT_NAME}.alloc: attribute keys are non-empty "
                            f"strings, got {k!r}")
        if not is_plugin_channel(f"raw.{k}"):
            raise ValueError(f"{cls._RESULT_NAME}.alloc: attribute key {k!r} must be a "
                             f"plugin key '<package>.<name>' (e.g. 'acme.grip')")
        if isinstance(dt, str) and dt in by_name:
            attrs[k] = dt = by_name[dt]
        if dt not in _ATTR_DTYPES:
            got = f"the str {dt!r}" if isinstance(dt, str) else _dtype_name(dt)
            raise TypeError(f"{cls._RESULT_NAME}.alloc: attributes[{k!r}] must be "
                            f"{_ATTR_DTYPES_TEXT} (a wp dtype or its name), got {got}")
    dev = wp.get_device(layout.device)
    ctx = LaunchContext.current(dev)
    _refuse_capture(dev, (ctx,), f"{cls._RESULT_NAME}.alloc()")
    E, S = int(layout.num_envs), int(layout.stride)
    kw = {}
    for name, per, dtype in cls._SCHEMA:
        arr = wp.empty(E * S if per == "point" else E, dtype=dtype, device=dev)
        ctx.fill(arr, _padding(per, dtype))
        kw[name] = arr
    if has_attrs:
        out_attrs = {}
        for k, dt in attrs.items():
            arr = wp.empty(E * S, dtype=dt, device=dev)
            ctx.fill(arr, _padding("point", dt))
            out_attrs[k] = arr
        kw["attributes"] = out_attrs
    if "fail_bits" in {f.name for f in dataclasses.fields(cls)}:
        kw["fail_bits"] = {}
    kw["layout"] = Layout(E, S, str(dev), layout.closed)
    if dev.is_cuda:
        # Complete the fills (and the current stream's earlier work on recycled blocks)
        # before another stream writes the result: it is usable on any stream.
        _compat.stream_synchronize(ctx.stream)
    return cls(**kw)


@dataclass
class GateSequence:
    """Batched fixed-stride gate result returned by ``GateGenerator``.

    Gate pose arrays are flat ``[E * max_gates]`` ``vec3f`` buffers; reshape via
    ``wp.to_torch(...).view(E, max_gates, 3)``. ``count[e]`` gives the real gate
    count for environment ``e`` and slots ``i >= count[e]`` are NaN-padded (any
    NaN component marks a padded slot). Each real slot's ``z`` follows the
    configured ``GateGenConfig.z_profile``; it is 0 only for the default
    ``"flat"`` profile at ``z_base=0``.

    .. warning::

        ``GateGenerator.generate()`` returns the SAME ``GateSequence`` instance on
        every call and overwrites its buffers in place.  A reference held across two
        ``generate()`` calls will see mutated data.  Call ``GateSequence.clone()`` to
        obtain a fully-owned deep copy before the next call.

    Methods (see :mod:`track_gen.results`): :meth:`validate` (check every
    field against the layout), :meth:`view` (a zero-copy env slice, e.g. ``out=`` of a
    mixed batch), :meth:`copy_from` (stream-ordered copies, no allocation),
    :meth:`clone` (an owned copy; allocates) and :meth:`alloc` (a fully allocated
    result).

    Construction requires all concrete arrays and layout and rejects explicit None;
    :meth:`validate` checks their shape, dtype and device. Metadata mappings can be
    omitted and start empty.

    Attributes
    ----------
    position : wp.array
        Flat ``[E * max_gates]`` ``vec3f`` gate centres.  Reshape via
        ``wp.to_torch(...).view(E, max_gates, 3)``.  Slots ``i >= count[e]`` are
        NaN-padded.
    tangent : wp.array
        Flat ``[E * max_gates]`` ``vec3f`` unit tangent vectors at each gate centre.
        NaN-padded past ``count[e]``.
    forward : wp.array
        Flat ``[E * max_gates]`` ``vec3f`` unit pose forward (x-axis of
        ``orientation``): the physical gate-plane normal used for pass detection;
        equals ``tangent`` for ``full_tangent`` and for planar ``yaw_only`` gates,
        the horizontal projection of ``tangent`` for sloped ``yaw_only`` gates;
        NaN-padded past ``count[e]``.
    orientation : wp.array
        Flat ``[E * max_gates]`` ``quatf`` gate poses (``wp.quatf(x, y, z, w)``):
        local x = gate forward, y = left, z = up. Default frames are roll-free;
        ``gate_frames.posed`` additionally applies the generator-provided roll.
        NaN-padded past ``count[e]``.
    half_size : wp.array
        Flat ``[E * max_gates]`` ``float32`` square-opening half-extents
        (``0.5 * gate_width``, or the generator's own when it provides
        ``half_size``).  NaN-padded past ``count[e]``.
    left : wp.array
        Flat ``[E * max_gates]`` ``vec3f`` left gate endpoints
        (``position + half_size * left_axis``).  NaN-padded past ``count[e]``.
    right : wp.array
        Flat ``[E * max_gates]`` ``vec3f`` right gate endpoints
        (``position - half_size * left_axis``).  NaN-padded past ``count[e]``.
    valid : wp.array
        ``[E]`` ``int32`` environment validity flags (0 or 1).  Convert to a boolean
        tensor via ``wp.to_torch(...).bool()``.
    count : wp.array
        ``[E]`` ``int32`` real gate counts per env.  All other arrays are NaN-padded
        for slots ``i >= count[e]``.
    generation : wp.array
        ``[E]`` ``int32``: the seed epoch (``GateGenerator.seeds.epoch``) each
        env's current gates were built at.  ``0`` until the first advance.  It is
        the epoch, not a change counter: a reseed at the same epoch
        (``rng.set_seeds_warp``, ``Course.generate(seeds=...)``) does not change
        it, and after ``seeds.reset_epoch()`` the same values come back.  Only
        while epochs are advanced and nothing reseeds or resets them does it
        change exactly when an env gets a NEW sequence
        (``generate(advance=True, ...)`` / ``Course.regenerate(advance=True,
        ...)``), so that comparing it on device to a stored copy detects every
        regeneration. Required also on a GateSequence built by hand.
    fail_mask : wp.array
        ``[E]`` ``int32`` per-env fail bitmask: bit ``i`` set means check ``i``
        failed (``fail_bits`` names the bits: ``validity.gate_default``'s 0-5 are
        ``count``, ``finite``, ``tangent``, ``spacing``, ``crossing``, ``grade``;
        plugin checks own 16-30). ``valid[e] == (fail_mask[e] == 0)``.
        Required also on a GateSequence built by hand.
    fail_bits : Mapping[str, int]
        ``name -> bit index`` of every bit of ``fail_mask`` the pipeline assigns
        (``GateGenerator.program.fail_bits``); the crossing / grade bits are named
        even when their check is off (then never set).  Empty on a ``GateSequence``
        built elsewhere.
    attributes : Mapping[str, wp.array]
        The per-gate attributes the pipeline exports, keyed by the channel name
        without its domain prefix (``anchors.acme.tag`` / ``gates.acme.tag`` ->
        ``"acme.tag"``): flat ``[E * max_gates]`` arrays, NaN (floats) / ``-1``
        (int32) past ``count[e]``.  Best effort: an attribute with no path to the gate
        slots (a non-``raw`` order, a plugin stations stage, a channel a stage writes
        on the centerline, e.g. an elevation with ``elevation_on="centerline"``) is
        left out (``GateGenerator.program.attributes`` lists what is there).  Empty
        unless a stage exports one.
    layout : Layout
        The required batch :class:`~track_gen.results.Layout` (``stride`` =
        ``max_gates``), including on a GateSequence built by hand. Use :meth:`alloc`
        to allocate every concrete array for a layout.
    """

    position: wp.array
    tangent: wp.array
    forward: wp.array
    orientation: wp.array
    half_size: wp.array
    left: wp.array
    right: wp.array
    valid: wp.array
    count: wp.array
    generation: wp.array
    fail_mask: wp.array
    layout: Layout
    fail_bits: Mapping[str, int] = field(default_factory=dict)
    attributes: Mapping[str, wp.array] = field(default_factory=dict)

    _RESULT_NAME = "GateSequence"
    _SCHEMA = _GATE_SCHEMA
    _OPTIONAL = frozenset()
    _REQUIRED_STORAGE = True
    __post_init__ = _require_result_storage
    _ANCHOR = "position"
    validate = _validate
    view = _view
    copy_from = _copy_from
    clone = _clone
    alloc = classmethod(_alloc)


@dataclass
class Track:
    """Final batched result of the track generation pipeline.

    All boundary arrays are index-aligned: ``outer[i]``, ``center[i]`` and
    ``inner[i]`` share a single cross-section normal. ``half_width[i]`` is the band's
    half-width there (``‖outer[i] - center[i]‖`` in plan view).
    Fields are ``wp.array``; convert at the boundary via the wp bridge.

    .. warning::

        ``TrackGenerator.generate()`` returns the SAME ``Track`` instance on every
        call and overwrites its buffers in place.  A reference held across two
        ``generate()`` calls will see mutated data.  Call ``Track.clone()`` to obtain
        a fully-owned deep copy before the next call (or ``copy_from`` into a
        preallocated ``Track.alloc`` result: no allocation, capture-safe).

    Methods (see :mod:`track_gen.results`): :meth:`validate` (check every
    field against the layout), :meth:`view` (a zero-copy env slice, e.g. ``out=`` of a
    mixed batch), :meth:`copy_from` (stream-ordered copies, no allocation: the hot-path
    snapshot), :meth:`clone` (an owned copy; allocates) and :meth:`alloc` (a fully
    allocated result).

    Construction requires all concrete arrays and layout and rejects explicit None;
    :meth:`validate` checks their shape, dtype and device. Metadata mappings can be
    omitted and start empty.

    Attributes
    ----------
    outer : wp.array
        Flat ``[E * N_max]`` ``vec3f`` outer boundary points.  Each real point's
        ``z`` follows the configured ``TrackGenConfig.z_profile`` — 0 only for
        the default ``"flat"`` profile at ``z_base=0`` — and shares the
        ``center`` point's altitude at the same index (level cross-sections;
        see :doc:`/tracks-25d`).  Reshape via
        ``wp.to_torch(...).view(E, N_max, 3)``.  Points at ``i >= count[e]`` are
        NaN-padded (NaN xy; z carries the flat path's ``z_base``, 0 by
        default).
    center : wp.array
        Flat ``[E * N_max]`` ``vec3f`` centerline points.  ``z`` follows the
        configured ``TrackGenConfig.z_profile`` (see ``outer``).  Reshape via
        ``wp.to_torch(...).view(E, N_max, 3)``.  Index-aligned with ``outer`` and
        ``inner``; ``‖outer[i] - center[i]‖`` gives the per-point half-width
        (an XY-plane distance: level cross-sections share a Z, so the vertical
        component cancels).  Points at ``i >= count[e]`` are NaN-padded.
    inner : wp.array
        Flat ``[E * N_max]`` ``vec3f`` inner boundary points.  ``z`` follows the
        configured ``TrackGenConfig.z_profile`` (see ``outer``).  Reshape
        via ``wp.to_torch(...).view(E, N_max, 3)``.  Points at ``i >= count[e]``
        are NaN-padded.
    tangent : wp.array
        Flat ``[E * N_max]`` ``vec3f`` unit tangent vectors at each centerline
        point.  For the default ``"flat"`` profile this is the planar tangent
        (``z = 0``) from central differences over the 2D centerline; for a
        non-flat ``z_profile`` it is instead a true 3D central difference over
        the LIFTED centerline (see :doc:`/tracks-25d`).  NaN-padded past
        ``count[e]``.
    normal : wp.array
        Flat ``[E * N_max]`` ``vec3f`` unit left-normals at each centerline point —
        still the planar left-normal, always ``z = 0``, computed as
        ``normal = (-tangent2d.y, tangent2d.x, 0)`` (the PLAN-VIEW tangent rotated
        +90° about +z) BEFORE the elevation lift.  For a non-flat ``z_profile`` this
        predates the 3D recompute described under ``tangent``: the two fields are
        no longer literally related by that rotation formula (``normal``'s XY
        direction still matches ``tangent``'s XY projection, but ``tangent`` is
        unit-length in 3D, so its XY projection alone is sub-unit and rotating it
        will not reproduce a unit ``normal``).  Which boundary it faces is
        winding-dependent (see ``winding``): it points toward
        ``outer`` for clockwise loops and toward ``inner`` for counter-clockwise
        loops.  The ``outer``/``inner`` split itself is winding-agnostic (assigned by
        signed-area magnitude), so use ``outer``/``inner`` — not the normal's sign —
        to identify the boundaries.  NaN-padded past ``count[e]``.
    arclen : wp.array
        Flat ``[E * N_max]`` ``float32`` cumulative arc length along the centerline
        at each point.  For the default ``"flat"`` profile this is the plan-view
        (2D) arc length; for a non-flat ``z_profile`` it is the true 3D arc length
        of the LIFTED centerline (strictly >= the plan-view length whenever the
        profile is non-constant; see :doc:`/tracks-25d`).  Reshape via
        ``wp.to_torch(...).view(E, N_max)``.  NaN-padded past ``count[e]``.
    length : wp.array
        ``[E]`` ``float32`` total arc length (perimeter) of each track's centerline
        loop — plan-view for the default ``"flat"`` profile, true 3D length
        otherwise (see ``arclen``).
    valid : wp.array
        ``[E]`` ``int32`` environment validity flags (0 or 1).  A track is valid
        when: turning number ≈ 1, no NaN points, width floor exceeded everywhere,
        minimum thickness ≥ ``(1 − relax_tol) * half_width``, the track was not
        truncated at ``N_max`` (see ``truncated``), and (when
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
    truncated : wp.array
        ``[E]`` ``int32`` N_max truncation flags (0 or 1), written on device by the
        constant-spacing resample.  ``1`` means the loop needed more than ``N_max``
        points (``floor(perimeter / spacing) + 1 > N_max``), so ``count[e]`` was
        clamped to ``N_max`` and the tail of the loop was dropped; such envs always
        have ``valid == 0``.  Increase ``N_max`` (or ``spacing``) if any are set.
        Required also on a Track built by hand.
    generation : wp.array
        ``[E]`` ``int32``: the seed epoch (``TrackGenerator.seeds.epoch``) each
        env's current track was built at.  ``0`` until the first advance.  It is
        the epoch, not a change counter: a reseed at the same epoch
        (``rng.set_seeds_warp``, ``Course.generate(seeds=...)``) does not change
        it, and after ``seeds.reset_epoch()`` the same values come back.  Only
        while epochs are advanced and nothing reseeds or resets them does it
        change exactly when an env gets a NEW track
        (``generate(advance=True, ...)`` / ``Course.regenerate(advance=True,
        ...)``), so that comparing it on device to a stored copy detects every
        regeneration. Required also on a Track built by hand.
    fail_mask : wp.array
        ``[E]`` ``int32`` per-env fail bitmask: bit ``i`` set means check ``i``
        failed (``fail_bits`` names the bits: the built-in checks' 0-7 are
        ``generator``, ``turning``, ``finite``, ``thickness``, ``w_floor``,
        ``border``, ``grade``, ``truncation``; plugin checks own 16-30).
        ``valid[e] == (fail_mask[e] == 0)``. Required also on a Track built by hand.
    fail_bits : Mapping[str, int]
        ``name -> bit index`` of every bit of ``fail_mask`` the pipeline assigns
        (``TrackGenerator.program.fail_bits``); the border / grade bits are named
        even when their check is off (then never set).  Empty on a ``Track`` built
        elsewhere.
    half_width : wp.array
        Flat ``[E * N_max]`` ``float32`` half-width of the band at each centerline
        point (the offset's: ``‖outer[i] - center[i]‖`` in plan view), NaN past
        ``count[e]``.  The constant ``half_width`` of the config on every real point
        at constant width (``width.constant``); the per-point width of a width stage
        or a generator that provides one otherwise. Required on every Track;
        a pipeline without an offset stage retains an all-NaN array.
    attributes : Mapping[str, wp.array]
        The per-point attributes the pipeline exports (``StageBuilder.write(...,
        export=True)``), keyed by the channel name without its domain prefix
        (``raw.acme.surface_id`` -> ``"acme.surface_id"``): flat ``[E * N_max]``
        arrays of the attribute's dtype, carried to the final points with its
        ``interp`` policy, NaN (floats) / ``-1`` (int32) past ``count[e]``.  Stable
        pointers for the generator's lifetime.  Empty unless a stage exports one
        (``TrackGenerator.program.attributes`` describes them).
    layout : Layout
        The required batch :class:`~track_gen.results.Layout` (``num_envs``,
        ``stride`` = ``N_max``, ``device``, ``closed``), including on a Track built
        by hand. Use :meth:`alloc` to allocate every concrete array for a layout.
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
    truncated: wp.array
    generation: wp.array
    fail_mask: wp.array
    half_width: wp.array
    layout: Layout
    fail_bits: Mapping[str, int] = field(default_factory=dict)
    attributes: Mapping[str, wp.array] = field(default_factory=dict)

    _RESULT_NAME = "Track"
    _SCHEMA = _TRACK_SCHEMA
    _OPTIONAL = frozenset()
    _REQUIRED_STORAGE = True
    __post_init__ = _require_result_storage
    _ANCHOR = "center"
    validate = _validate
    view = _view
    copy_from = _copy_from
    clone = _clone
    alloc = classmethod(_alloc)


@dataclass
class Centerline:
    """A batched closed 3D centerline without a road band.

    What a gates-mode ``Course`` threads through its gates
    (``course.course_line.centerline``; the course line is internal to ``Course`` in
    0.x): the members a centerline consumer reads
    (:class:`~track_gen.results.HasCenterline`: ``TrackLocalizer``,
    ``localize.curvature`` / ``speed_profile``), with the same flat storage as a
    :class:`~track_gen.results.Track`: per-point arrays ``[E * stride]`` (NaN past
    ``count[e]``), per-env arrays ``[E]``.

    Attributes:
        center: ``[E * stride]`` ``vec3f`` centerline points.
        tangent: ``[E * stride]`` ``vec3f`` unit tangents.
        arclen: ``[E * stride]`` ``float32`` cumulative arc length.
        length: ``[E]`` ``float32`` loop length.
        count: ``[E]`` ``int32`` real point counts.
        valid: ``[E]`` ``int32`` validity flags (0 / 1).
        generation: ``[E]`` ``int32`` generation epoch, or ``None``.
        layout: Its :class:`~track_gen.results.Layout` (``None`` on one built by hand:
            :meth:`validate` infers it).
    """

    center: wp.array
    tangent: wp.array
    arclen: wp.array
    length: wp.array
    count: wp.array
    valid: wp.array
    generation: "wp.array | None" = None
    layout: "Layout | None" = None

    _RESULT_NAME = "Centerline"
    _SCHEMA = _CENTERLINE_SCHEMA
    _OPTIONAL = _CENTERLINE_OPTIONAL
    _ANCHOR = "center"
    validate = _validate
    view = _view
    copy_from = _copy_from
    clone = _clone
    alloc = classmethod(_alloc)


GateSequence._RESULT_BASE = GateSequence
Track._RESULT_BASE = Track
Centerline._RESULT_BASE = Centerline
