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
    generator: str = "bezier"  # one of {"bezier", "fourier"}
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
    relax_use_warp: bool | None = None    # xpbd separation: None=auto (Warp on CUDA), True=force Warp
    relax_tol: float = 0.02               # target = (1 - tol) * half_width
    relax_band: int | None = None         # None => round(D / L0) per track
    relax_iters: int = 150
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
    N_max: int = 256

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
        # Only constant_spacing is supported: the legacy "fixed" (constant point COUNT) mode
        # over-resolved the centerline (jagged XPBD -> folded roads) and was dropped in favour
        # of constant link SIZE (~0.6*half_width), which relaxes to smoother, higher-yield tracks.
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
    """

    outer: wp.array    # [E, N] vec2f
    center: wp.array   # [E, N] vec2f
    inner: wp.array    # [E, N] vec2f
    tangent: wp.array  # [E, N] vec2f
    normal: wp.array   # [E, N] vec2f
    arclen: wp.array   # [E, N] float32
    length: wp.array   # [E] float32
    valid: wp.array    # [E] bool/int32
    count: wp.array    # [E] int32
