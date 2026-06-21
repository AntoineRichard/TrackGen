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

    Fields mirror design spec section 3.2. ``rad``, ``edgy`` and ``half_width``
    are scalars for now (per-env sampling of their ranges is intentionally
    deferred — see the "Deferred (YAGNI)" note at the end of the plan).
    """

    # --- Generator selection + batching ---
    generator: str = "bezier"  # registered generator name; see generator_registry.available() (currently "bezier")
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
    # diversity); larger values bulge/pinch the lobes for more racing-shape variety. ~0.5 is
    # a grounded default (lobes vary by up to ±half the local radius).
    hull_displacement: float = 0.5

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

    # --- Fourier params (EXPERIMENTAL: consumed only by track_gen._experimental.fourier;
    #     the supported Warp pipeline ignores them) ---
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
        if int(self.relax_sep_every) < 1:
            raise ValueError(f"relax_sep_every must be >= 1, got {self.relax_sep_every!r}")
        if int(self.relax_sep_cache_slots) < 0:
            raise ValueError(
                f"relax_sep_cache_slots must be >= 0, got {self.relax_sep_cache_slots!r}")
        if float(self.relax_sep_cache_skin) < 0.0:
            raise ValueError(
                f"relax_sep_cache_skin must be >= 0, got {self.relax_sep_cache_skin!r}")

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
