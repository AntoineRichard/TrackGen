"""Shared dataclasses for the track generation pipeline.

This module imports nothing from the rest of the package (no generators, no
inflation, no track_generator, no rng_utils). It is the shared home for
``TrackGenConfig`` and ``Track`` so that ``warp_pipeline.py`` and the facade can
both import them without a circular import. Warp is a core dependency (``Track``
fields are ``wp.array``).
"""

import dataclasses
import math
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import warp as wp


def _validate_z_fields(cfg, check_profile: bool = True) -> None:
    """Shared Z-profile validation for ``TrackGenConfig`` and ``GateGenConfig``.

    Operates purely on the eight ``z_*`` attribute names both configs carry
    on the same terms (profile membership, min/max ordering, non-negative
    knobs), so a single call site works for either dataclass's
    ``__post_init__``. ``check_profile=False`` (``TrackGenConfig``) skips the
    membership check: a track's ``z_profile`` names any registered elevation stage
    and is resolved through the registry at ``TrackGenerator`` construction.
    """
    if check_profile and cfg.z_profile not in {"flat", "uniform", "random_walk", "noise"}:
        raise ValueError(
            "z_profile must be one of "
            "{'flat', 'uniform', 'random_walk', 'noise'}, got "
            f"{cfg.z_profile!r}"
        )
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


def _check_generator_params(cfg) -> None:
    """Registry-free type check of ``generator_params`` (the generator resolves and
    validates the value at construction)."""
    gp = cfg.generator_params
    if gp is not None and not isinstance(gp, Mapping) and not (
            dataclasses.is_dataclass(gp) and not isinstance(gp, type)):
        raise TypeError("generator_params must be a Params instance, a mapping of its "
                        f"fields, or None; got {type(gp).__name__}")


# TrackGenConfig fields TrackGenerator does not use (torch-oracle / experimental only;
# it only rejects smooth_finish=True): setting one to a value other than its class's
# default emits a DeprecationWarning. ``output_mode`` is
# deprecated too, but its own ValueError already rejects every non-default value.
_DEAD_TRACK_FIELDS = (
    "energy_steps", "energy_lr", "energy_w_sep", "energy_w_len", "energy_w_bend",
    "energy_w_anchor", "tp_iters", "tp_tau", "tp_alpha", "tp_beta",
    "smooth_finish", "smooth_finish_iters", "smooth_finish_tau",
    "relax_chunk_size", "relax_use_warp", "max_regen_iters", "min_angle",
    "num_harmonics", "decay_p", "amplitude", "num_centerline_samples",
)


def _warn_dead_fields(cfg) -> None:
    # Defaults of the config's own class: a subclass that changes one is not "set".
    defaults = {f.name: f.default for f in dataclasses.fields(type(cfg))}
    names = [n for n in _DEAD_TRACK_FIELDS if getattr(cfg, n) != defaults[n]]
    if names:
        # Point at the code constructing the config: skip every frame of this object's
        # construction (the generated __init__, and each __post_init__ of a subclass
        # chain that calls super().__post_init__()).
        level, frame = 2, sys._getframe(1)
        while frame is not None and frame.f_locals.get("self") is cfg:
            frame, level = frame.f_back, level + 1
        warnings.warn(
            f"TrackGenConfig fields {names} are deprecated and unused by TrackGenerator "
            f"(oracle/experimental only; smooth_finish=True is rejected); they will be "
            f"removed in 1.0",
            DeprecationWarning, stacklevel=level)


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

    Per-generator parameters: each generator declares its own Params class
    (``track_gen.plugin.registry.describe("generator", name).params_schema``). Set
    them either through the flat fields below (the built-in generators' knobs, e.g.
    ``polar_num_knots``) or through ``generator_params`` (a Params instance or a mapping
    of its fields, e.g. ``generator_params={"polar_num_knots": 9}``), not both with
    different values. The generator-specific rules (``voronoi_control_points >= 6``,
    the ``repulsive_*`` rules, ...) are checked when the generator is constructed, and
    only for the selected generator.

    Deprecated (a ``DeprecationWarning`` when set to a non-default value; removed in
    1.0): ``energy_*``, ``tp_*``, ``smooth_finish*``, ``relax_chunk_size``,
    ``relax_use_warp``, ``max_regen_iters``, ``min_angle``, ``num_harmonics``,
    ``decay_p``, ``amplitude``, ``num_centerline_samples`` (and ``output_mode``, which
    only accepts its default).

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

            The ``repulsive`` generator is an iterative optimizer — roughly **1000×
            slower** than ``bezier`` (~0.18 s @ E=64, ~5.1 s @ E=8192 on an RTX 4090,
            vs ~2 ms for bezier; measured when it still ran eagerly).  It is
            graph-captured on CUDA like the other generators, but the replay still
            runs its full fixed iteration budget.  Regenerate on a **slow cadence**
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
        The relax stage ``TrackGenerator`` runs: any registered ``relax`` stage
        (``track_gen.plugin.registry.available("relax")``; default ``"xpbd"``, the
        XPBD solver configured by the ``relax_*`` fields below; a plugin stage gets
        its ``Params.from_legacy(config)``). Resolved at ``TrackGenerator``
        construction, which raises ``ValueError`` listing the available names for an
        unknown one.  ``"energy"`` (Adam) and ``"tp_sobolev"`` are oracle-only
        backends in ``tests/_oracle/relaxation.py``, not relax stages:
        ``TrackGenerator`` rejects them.
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
        host-precomputed omega recurrence, converging ~3x faster per sweep.
        ``"none"`` runs plain damped Jacobi (the legacy behaviour).
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
        (:math:`\\lambda > 0` then :math:`\\mu < -\\lambda`), which removes the
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
        mid-solve — the cadence dropped from 40 alongside the ``relax_iters``
        150 -> 50 default so the accelerated solve still refreshes mid-solve.
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
        whose true ``count[e]`` exceeds ``N_max`` are truncated, flagged in
        ``Track.truncated`` and fail validity.
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
        ``"noise"``.  Must be <= ``z_max``.  Default 0.0.
    z_max : float
        Upper altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be >= ``z_min``.  Default 0.0.
    z_max_step : float
        *Random-walk profile only.*  Maximum ``|dz|`` per unit plan-view arc
        length (a grade cap), applied per random-walk step before the
        closure bridge.  Must be >= 0.  Default 0.0.
    z_noise_amplitude : float
        *Noise profile only.*  Amplitude of the harmonic altitude
        oscillation around ``z_base``.  Must be >= 0.  Default 0.0.
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

    generator_params : Params, mapping or None
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

    # --- Per-generator parameters (None | Params instance | mapping of its fields) ---
    generator_params: Any = None

    def __post_init__(self):
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

        if not isinstance(self.z_profile, str) or not self.z_profile:
            raise ValueError(f"z_profile must be a non-empty string (an elevation stage "
                             f"name), got {self.z_profile!r}")
        _validate_z_fields(self, check_profile=False)
        if int(self.z_control_points) < 3:
            raise ValueError(
                "z_control_points must be >= 3, got "
                f"{self.z_control_points!r}")
        _warn_dead_fields(self)


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

    z_profile : str
        Per-gate altitude (Z) profile family.  One of ``"flat"`` (default,
        every real gate at ``z_base``), ``"uniform"`` (i.i.d. uniform in
        ``[z_min, z_max]``), ``"random_walk"`` (Brownian-bridge walk clamped
        to ``[z_min, z_max]`` with a per-step grade cap of ``z_max_step``), or
        ``"noise"`` (periodic harmonic noise around ``z_base``).  Applied by
        ``warp_zprofile.apply_z_profile`` after XY ordering/relaxation and
        before the 3D lift, wired into the generation pipeline by
        ``warp_gate._run_gate_pipeline``.
    z_base : float
        Baseline altitude for the ``"flat"``, ``"random_walk"``, and
        ``"noise"`` profiles.  Default 0.0.
    z_min : float
        Lower altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be <= ``z_max``.  Default 0.0.
    z_max : float
        Upper altitude clamp for ``"uniform"``, ``"random_walk"``, and
        ``"noise"``.  Must be >= ``z_min``.  Default 0.0.
    z_max_step : float
        *Random-walk profile only.*  Maximum ``|dz|`` per unit plan-view arc
        length (a grade cap), applied per random-walk step before the
        closure bridge.  Must be >= 0.  Default 0.0.
    z_noise_amplitude : float
        *Noise profile only.*  Amplitude of the harmonic altitude
        oscillation around ``z_base``.  Must be >= 0.  Default 0.0.
    z_noise_harmonics : int
        *Noise profile only.*  Number of summed harmonics in the periodic
        noise field.  Must be >= 1.  Default 3.
    z_valid_grade : float
        Maximum allowed ``|dz|``/ds grade for post-hoc course validity checking.
        0 disables the check.  Must be >= 0.  Default 0.0.  (Consumed
        starting in Task 4; unused by ``warp_zprofile`` itself.)
    gate_align : str
        Gate-frame alignment mode.  ``"yaw_only"`` (default) keeps gate posts
        horizontal (the frame forward axis is the plan-view tangent projected
        onto the XY plane, so left/right posts share the center's altitude);
        ``"full_tangent"`` aligns the frame forward axis with the full 3D
        tangent, tilting posts to follow the course slope.

    generator_params : Params, mapping or None
        The selected gate generator's parameters, as
        ``TrackGenConfig.generator_params`` (``None``: the flat fields above).  The
        inert fields (``num_points_per_segment``, ``rad``, ``edgy``,
        ``handle_clamp_frac``, ``hull_displacement``) belong to no gate generator.
        The generator-specific rules (point family, ``voronoi_*``, ``checkpoint_*``)
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

    z_profile: str = "flat"
    z_base: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    z_max_step: float = 0.0
    z_noise_amplitude: float = 0.0
    z_noise_harmonics: int = 3
    z_valid_grade: float = 0.0
    gate_align: str = "yaw_only"

    # --- Per-generator parameters (None | Params instance | mapping of its fields) ---
    generator_params: Any = None

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
        _check_generator_params(self)
        # Generator-specific rules (point family, voronoi_*, checkpoint_*) live in the
        # gate generators' Params.validate() and run at GateGenerator construction.
        _validate_z_fields(self)
        if self.gate_align not in {"yaw_only", "full_tangent"}:
            raise ValueError(
                "gate_align must be one of {'yaw_only', 'full_tangent'}, got "
                f"{self.gate_align!r}"
            )


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
        the roll-free frame with local x = gate forward, y = left, z = up.
        NaN-padded past ``count[e]``.
    half_size : wp.array
        Flat ``[E * max_gates]`` ``float32`` square-opening half-extents
        (``0.5 * gate_width``).  NaN-padded past ``count[e]``.
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
    generation : wp.array or None
        ``[E]`` ``int32``: the seed epoch (``GateGenerator.seeds.epoch``) each
        env's current gates were built at.  ``0`` until the first advance.  It is
        the epoch, not a change counter: a reseed at the same epoch
        (``rng.set_seeds_warp``, ``Course.generate(seeds=...)``) does not change
        it, and after ``seeds.reset_epoch()`` the same values come back.  Only
        while epochs are advanced and nothing reseeds or resets them does it
        change exactly when an env gets a NEW sequence
        (``generate(advance=True, ...)`` / ``Course.regenerate(advance=True,
        ...)``), so that comparing it on device to a stored copy detects every
        regeneration.  ``GateGenerator`` always allocates it; ``None`` on a
        ``GateSequence`` built elsewhere.
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
    generation: "wp.array | None" = None

    def clone(self) -> "GateSequence":
        """Return a deep copy whose Warp buffers do not alias this sequence."""
        return GateSequence(
            position=wp.clone(self.position),
            tangent=wp.clone(self.tangent),
            forward=wp.clone(self.forward),
            orientation=wp.clone(self.orientation),
            half_size=wp.clone(self.half_size),
            left=wp.clone(self.left),
            right=wp.clone(self.right),
            valid=wp.clone(self.valid),
            count=wp.clone(self.count),
            generation=None if self.generation is None else wp.clone(self.generation),
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
    truncated : wp.array or None
        ``[E]`` ``int32`` N_max truncation flags (0 or 1), written on device by the
        constant-spacing resample.  ``1`` means the loop needed more than ``N_max``
        points (``floor(perimeter / spacing) + 1 > N_max``), so ``count[e]`` was
        clamped to ``N_max`` and the tail of the loop was dropped; such envs always
        have ``valid == 0``.  Increase ``N_max`` (or ``spacing``) if any are set.
        ``TrackGenerator`` always allocates it; ``None`` on a ``Track`` built
        elsewhere (e.g. the standalone ``inflate_warp`` path).
    generation : wp.array or None
        ``[E]`` ``int32``: the seed epoch (``TrackGenerator.seeds.epoch``) each
        env's current track was built at.  ``0`` until the first advance.  It is
        the epoch, not a change counter: a reseed at the same epoch
        (``rng.set_seeds_warp``, ``Course.generate(seeds=...)``) does not change
        it, and after ``seeds.reset_epoch()`` the same values come back.  Only
        while epochs are advanced and nothing reseeds or resets them does it
        change exactly when an env gets a NEW track
        (``generate(advance=True, ...)`` / ``Course.regenerate(advance=True,
        ...)``), so that comparing it on device to a stored copy detects every
        regeneration.  ``TrackGenerator`` always allocates it; ``None`` on a
        ``Track`` built elsewhere.
    fail_mask : wp.array or None
        ``[E]`` ``int32`` per-env fail bitmask: bit ``i`` set means check ``i``
        failed (``fail_bits`` names the bits: the built-in checks' 0-7 are
        ``generator``, ``turning``, ``finite``, ``thickness``, ``w_floor``,
        ``border``, ``grade``, ``truncation``; plugin checks own 16-30).  When it
        is not ``None``, ``valid[e] == (fail_mask[e] == 0)``.  ``TrackGenerator``
        always allocates it; ``None`` on a ``Track`` built elsewhere (e.g. the
        standalone ``inflate_warp`` path).
    fail_bits : Mapping[str, int]
        ``name -> bit index`` of every bit of ``fail_mask`` the pipeline assigns
        (``TrackGenerator.program.fail_bits``); the border / grade bits are named
        even when their check is off (then never set).  Empty on a ``Track`` built
        elsewhere.
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
    truncated: "wp.array | None" = None
    generation: "wp.array | None" = None
    fail_mask: "wp.array | None" = None
    fail_bits: Mapping[str, int] = field(default_factory=dict)

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
            truncated=None if self.truncated is None else wp.clone(self.truncated),
            generation=None if self.generation is None else wp.clone(self.generation),
            fail_mask=None if self.fail_mask is None else wp.clone(self.fail_mask),
            fail_bits=self.fail_bits,
        )
