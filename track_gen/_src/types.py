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
    the ``*_range`` fields for the Bezier generator.
    """

    # --- Generator selection + batching ---
    generator: str = "bezier"  # registered generator name; see generator_registry.available()
    device: str = "cpu"
    num_envs: int = 1

    # --- Bezier params ---
    min_num_points: int = 9
    max_num_points: int = 13
    num_points_per_segment: int = 30
    min_point_distance: float = 0.05
    min_angle: float = (12.5 / 180) * math.pi
    # Cubic-Bezier handle length as a fraction of the segment chord -- the main curviness
    # dial. Only LIVE when rad <= handle_clamp_frac; if rad exceeds the clamp, the clamp binds
    # every segment and rad does nothing (see handle_clamp_frac).
    rad: float = 0.4
    edgy: float = 0.0
    scale: float = 1.0
    # Adaptive Bezier-handle clamp (F2): each corner's handle is capped at
    # handle_clamp_frac * (its shorter incident edge), so a long handle can't overshoot past a
    # nearby corner and self-cross. The clamp does its narrow job (and leaves `rad` as the live
    # curviness dial) only when handle_clamp_frac >= rad; set BELOW rad it binds EVERY segment
    # and pins the handle to handle_clamp_frac*edge regardless of rad -- which is what produced
    # near-polygonal (straight) tracks at the old 0.10 default. Kept == rad here so the clamp
    # only trims genuine overshoot corners. Set very large to disable. Generation is single-pass
    # (no gate, no regen): any track that still self-crosses falls back to its corner polygon
    # (handle_clamp_frac=0, applied to the WHOLE track), which XPBD re-rounds -- so this knob
    # trades corner roundness against how often that polygon fallback fires (~5% at 0.4).
    handle_clamp_frac: float = 0.4

    # --- Hull generator params (generator="hull") ---
    # Max per-edge radial midpoint displacement as a fraction of the centroid->midpoint
    # distance. 0.0 collapses the augmented loop toward the plain angle-sort (≈ bezier-like
    # diversity); larger values bulge/pinch the lobes for more racing-shape variety, but
    # can create pinched loops that fail the thickness gate.
    hull_displacement: float = 0.15

    # --- Checkpoint-steering generator params (generator="checkpoint", method #5) ---
    # A fixed-shape NVIDIA-Warp port of Gymnasium CarRacing's track family: sample C radial
    # checkpoints, steer a bounded-turn-rate path that chases them once around, close the
    # heading to turning-number-1 additively (preserving local curvature), and keep the
    # best-of-K candidate by self-intersection count. See _experimental/checkpoint_proto.py
    # for the validated reference these defaults were tuned against.
    #
    # checkpoint_count C: number of radial checkpoints (CarRacing's canonical 12). More -> wavier
    #   (chicane count scales ~ C); fewer -> calmer/rounder. Must be >= 3 for a loop.
    checkpoint_count: int = 12
    # Inner radius fraction: checkpoint radius ~ U(checkpoint_radius_min_frac*R, R), R=1. This is
    #   CarRacing's R/3 exactly — the radial drama that gives the inlets. Must be in [0, 1).
    checkpoint_radius_min_frac: float = 0.33
    # Angle noise as +/- this fraction of the per-checkpoint angular slot (slot = 2*pi/C). Kept
    #   < 1 so the checkpoint sequence stays angle-monotone -> the steered path winds once.
    checkpoint_angle_jitter: float = 0.55
    # Max heading change per steering step (rad). Bounds the path curvature -> smooth flow; too
    #   large lets the path kink, too small can't track the inlets.
    checkpoint_turn_rate: float = 0.42
    # Proportional steering gain toward the current target bearing (0..1]. How aggressively the
    #   heading chases the target each step (the bounded-turn dl couples it to one lap).
    checkpoint_steer_gain: float = 0.65
    # Advance to the NEXT checkpoint once within this * R of the current target. Lookahead that
    #   keeps the path flowing through (not stalling at) each checkpoint.
    checkpoint_lookahead_frac: float = 0.16
    # best-of-K: generate K decorrelated candidates per env and KEEP the one with the fewest
    #   self-intersections (deterministic argmin, ties -> lowest k). Replaces CarRacing's
    #   unbounded reject-retry with a bounded, capturable pool. 4 is the shipped value (proven
    #   ~0.4% pre-relax SI); the prototype's 8 was only its render driver's K. Must be >= 1.
    checkpoint_best_of_k: int = 4
    # OPT-IN single-crossing clip fallback (default off, like the bezier/hull polygon fallback):
    #   when True, after best-of-K selection the selected centerline's FIRST self-crossing is
    #   clipped — split at the intersection point, keep the longer sub-loop arc, resample to N.
    #   A capture-time Python branch (not per-env), so the captured graph stays fixed either way.
    checkpoint_clip_fallback: bool = False

    # --- Per-env style sampling (method #1: "Per-env style randomization") ---
    # OPT-IN. When False (the default) the bezier generator is BYTE-FOR-BYTE unchanged: the
    # kernels consume the scalar `rad`/`scale`/`handle_clamp_frac` above exactly as before.
    # When True, each env draws its OWN `rad`/`scale`/`handle_clamp_frac` from the *_range
    # fields below via the per-env Warp RNG (seeded from seeds_wp[e]), so a single batch
    # spans a *family* of styles instead of one config scalar -> much richer per-env
    # diversity. Corner-count spread already varies per env via min/max_num_points, so it
    # needs no range here. The flag selects the code path at CUDA-graph CAPTURE time (a
    # Python branch over which kernel to launch); the per-env values themselves live in
    # device arrays, so the captured graph stays fixed and capturable. A *_range left None
    # while style_sampling=True falls back to the corresponding scalar for every env (that
    # knob is simply not varied).
    style_sampling: bool = False
    rad_range: tuple[float, float] | None = None
    scale_range: tuple[float, float] | None = None
    handle_clamp_frac_range: tuple[float, float] | None = None

    # --- Polar / Fourier params ---
    # Supported polar generator (generator="polar"): random sorted polar control knots ->
    # periodic cubic spline. Knots default to a denser, non-round design space than the old
    # low-pass Fourier variant; jitter values are clamped by the generator to preserve
    # positive radii and monotone angular order.
    polar_num_knots: int = 12
    polar_radial_jitter: float = 0.60
    polar_angular_jitter: float = 0.30

    # --- Voronoi / graph-cycle generator params (generator="voronoi") ---
    # Samples a fixed field of cell sites, snaps angular anchor targets to nearby unused
    # sites, smooths the resulting graph-cycle loop, then resamples to num_points. This is
    # Warp-native and graph-capturable: exact Voronoi ridge construction remains out of the
    # runtime path because dynamic Delaunay/cycle-basis traversal does not fit the contract.
    voronoi_num_sites: int = 256
    voronoi_site_layout: str = "void_ring"  # {"ring", "void_ring", "clustered", "mixed"}
    voronoi_control_points: int = 18
    voronoi_radial_variation: float = 0.62
    voronoi_angular_jitter: float = 0.08
    # Experimental torch-only Fourier generator fields retained for compatibility with
    # track_gen._experimental.fourier and older sweeps.
    num_harmonics: int = 5  # K
    decay_p: int = 2  # decay exponent: amplitude ~ amp / k**decay_p
    amplitude: float = 1.0
    num_centerline_samples: int = 256  # Fourier dense sample count (M_max)

    # --- Width params ---
    half_width: float = 0.1  # w_max

    # --- Relaxation: backend selection + scale ---
    relax_enable: bool = True
    relax_solver: str = "xpbd"            # {"xpbd","energy","tp_sobolev"}
    relax_chunk_size: int | None = None   # env-chunk the dense [E,N,N] term
    relax_use_warp: bool | None = None    # ignored by the warp runtime; read only by the torch oracle (tests)
    relax_tol: float = 0.02               # target = (1 - tol) * half_width
    relax_band: int | None = None         # None => round(D / L0) per track
    relax_iters: int = 150
    # Separation broadphase refresh interval.
    #
    # The XPBD separation target is target = 2*half_width*(1 + relax_margin).
    # With relax_sep_cache_slots == 0, this is a naive cadence: the dense O(N^2)
    # separation scan runs only every K sweeps and is skipped in between. That is fast
    # but can miss transient contacts. With relax_sep_cache_slots > 0, this becomes the
    # broadphase refresh interval: every K sweeps we rebuild a fixed-size candidate
    # cache, while the exact narrowphase distance test and separation push still run on
    # cached candidates every sweep. K=1 preserves the dense baseline.
    relax_sep_every: int = 40
    # Cached separation candidate capacity per bead. 0 disables the cache. When enabled,
    # each bead stores up to this many non-neighbour bead indices from the latest
    # broadphase refresh. Larger values add memory and narrowphase work, but reduce the
    # chance of dropping candidates in crowded tracks.
    relax_sep_cache_slots: int = 16
    # Broadphase skin as a fraction of the exact separation target. The cache stores
    # candidates within target*(1 + skin), but every cached sweep still applies
    # separation only when the current exact distance is < target. skin=0 is fastest;
    # positive values are more conservative when contacts may enter during a long K.
    relax_sep_cache_skin: float = 0.5
    relax_sep_relax: float = 1.0
    relax_spc_relax: float = 1.0
    relax_bend_relax: float = 1.5
    relax_margin: float = 0.15

    # energy (Adam)
    energy_steps: int = 800
    energy_lr: float = 3e-3
    energy_w_sep: float = 80.0
    energy_w_len: float = 8.0
    energy_w_bend: float = 1.0
    energy_w_anchor: float = 0.01
    # tp_sobolev (standalone backend + finisher share tp_alpha/tp_beta)
    tp_iters: int = 100
    tp_tau: float = 0.7
    tp_alpha: float = 2.0
    tp_beta: float = 4.5
    # optional tangent-point/Sobolev smoothing finisher
    smooth_finish: bool = False
    smooth_finish_iters: int = 8
    smooth_finish_tau: float = 0.2

    # --- Output params ---
    num_points: int = 256  # N: intermediate dense->resample resolution before constant-spacing
    output_mode: str = "constant_spacing"  # the only supported mode (see __post_init__)
    # constant_spacing arc-length step (m). None -> auto 0.6*half_width (the relax-friendly
    # value); set explicitly to override. A fixed default would be wrong across half_widths.
    spacing: float | None = None
    # Output buffer width: max constant-spacing points per track. count ~= perimeter/spacing,
    # and spacing auto-couples to half_width, so the real (per-env-diverse) count distribution
    # scales with 1/half_width at a given scale: the default half_width=0.1 (spacing=0.06) needs
    # ~141 max, half_width=0.05 ~281, the finer half_width=0.03 (spacing=0.018) regime ~468 max.
    # 384 is a grounded middle: huge headroom at the default, full coverage to half_width~0.05,
    # and ~98% of the half_width=0.03 regime. Tracks whose true count still exceeds N_max are
    # truncated with an explicit RuntimeWarning (see resample_constant_spacing) — raise N_max
    # (or coarsen spacing) for finer/larger-scale regimes.
    N_max: int = 384

    # --- Robustness params ---
    max_regen_iters: int = 10
    turning_tol: float = 0.1
    w_floor: float = 1e-3  # validity: every real point must have w > w_floor
    # Optional extra border self-intersection check in validity. Redundant with the
    # thickness/separation gate (a self-crossing / fat-band overlap drives separation_min->0
    # -> thickness < half_width -> invalid), so default OFF saves two O(N^2) passes with no
    # change to the valid mask. Set True to re-enable the explicit border crossing check.
    validity_border_check: bool = False

    def __post_init__(self):
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


@dataclass
class GateGenConfig:
    """Configuration for fixed-batch native gate sequence generation."""

    generator: str = "bezier"
    device: str = "cpu"
    num_envs: int = 1

    min_gates: int = 4
    max_gates: int = 32
    gate_radius: float = 0.025
    gate_solve_iters: int = 8
    gate_width: float = 0.0
    gate_ordering: str = "ccw"

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

    **Aliasing warning**: ``GateGenerator.generate()`` returns the SAME
    ``GateSequence`` instance on every call and overwrites its buffers in place. A
    reference held across two ``generate()`` calls will see mutated data. Use
    ``GateSequence.clone()`` to obtain a fully-owned deep copy before the next call.
    """

    position: wp.array  # flat [E*max_gates] gate centers; NaN-padded past count[e]
    tangent: wp.array   # flat [E*max_gates] unit tangent vectors
    normal: wp.array    # flat [E*max_gates] unit normals, perpendicular to tangent
    left: wp.array      # flat [E*max_gates] left gate endpoints
    right: wp.array     # flat [E*max_gates] right gate endpoints
    valid: wp.array     # [E] int32 (0/1; wp.to_torch(...).bool() to recover)
    count: wp.array     # [E] int32 real gate counts

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

    **Aliasing warning**: ``TrackGenerator.generate()`` returns the SAME ``Track``
    instance on every call and overwrites its buffers in place. A reference held
    across two ``generate()`` calls will see mutated data. Use ``Track.clone()`` to
    obtain a fully-owned deep copy before the next call.
    """

    # flat [E*N_max] vec2f storage; reshape via wp.to_torch(...).view(E, N_max, 2)
    outer: wp.array    # flat [E*N_max] vec2f; reshape via wp.to_torch(...).view(E, N_max, 2)
    center: wp.array   # flat [E*N_max] vec2f; reshape via wp.to_torch(...).view(E, N_max, 2)
    inner: wp.array    # flat [E*N_max] vec2f; reshape via wp.to_torch(...).view(E, N_max, 2)
    tangent: wp.array  # flat [E*N_max] vec2f; reshape via wp.to_torch(...).view(E, N_max, 2)
    normal: wp.array   # flat [E*N_max] vec2f; reshape via wp.to_torch(...).view(E, N_max, 2)
    arclen: wp.array   # flat [E*N_max] float32; reshape via wp.to_torch(...).view(E, N_max)
    length: wp.array   # [E] float32
    valid: wp.array    # [E] int32 (0/1; wp.to_torch(...).bool() to recover)
    count: wp.array    # [E] int32

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
        )
