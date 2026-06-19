# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free leaf dataclasses shared across the pipeline.

This module imports NOTHING from the rest of the package (no generators, no
inflation, no track_generator, no rng_utils, no warp). It is the shared home for
``TrackGenConfig`` and ``Track`` so that ``inflation.py`` and the facade can both
import them without a circular import, and so CPU-only tests never drag in NVIDIA
Warp.
"""

import math
from dataclasses import dataclass

from torch import Tensor


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
    rad: float = 0.2
    edgy: float = 0.0
    scale: float = 1.0
    # Adaptive Bezier-handle clamp (F2): each corner's handle is capped at
    # handle_clamp_frac * (its shorter incident edge), so a long handle can't overshoot past a
    # nearby corner and self-cross. Lower = fewer overshoot crossings but rounder->tighter
    # corners (less shape diversity); ~0.10 takes single-attempt crossing-free ~96.4%->~99.2%
    # at the fat-band regime with ~8% roundness narrowing. Set very large to disable. The
    # generation gate + regen still guarantee 100% crossing-free on ACCEPTED tracks; this
    # knob only trades corner roundness against regen pressure.
    handle_clamp_frac: float = 0.10

    # --- Fourier params ---
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
    relax_use_warp: bool | None = None    # xpbd separation: None=auto (Warp on CUDA), False=torch, True=force Warp
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
    num_points: int = 256  # N
    output_mode: str = "fixed"  # one of {"fixed", "constant_spacing"}
    spacing: float = 0.1            # constant_spacing arc-length step (m). Warp relax: set ~0.6*half_width.
    N_max: int = 256

    # --- Robustness params ---
    max_regen_iters: int = 10
    turning_tol: float = 0.1
    w_floor: float = 1e-3  # validity: every real point must have w > w_floor

    def __post_init__(self):
        if self.output_mode not in ("fixed", "constant_spacing"):
            raise ValueError(
                f"output_mode must be 'fixed' or 'constant_spacing', got {self.output_mode!r}")


@dataclass
class Track:
    """Final batched result of the track generation pipeline.

    All boundary arrays are index-aligned: ``outer[i]``, ``center[i]`` and
    ``inner[i]`` share a single cross-section normal. Half-width is not stored;
    recover it as ``torch.linalg.norm(outer - center, dim=-1)``.
    """

    outer: Tensor  # [E, N, 2]
    center: Tensor  # [E, N, 2]
    inner: Tensor  # [E, N, 2]
    tangent: Tensor  # [E, N, 2] unit tangent along centerline
    normal: Tensor  # [E, N, 2] unit left-normal along centerline
    arclen: Tensor  # [E, N] cumulative arc length
    length: Tensor  # [E] total length per track
    valid: Tensor  # [E] bool validity mask
    count: Tensor  # [E] int real point count (== N in fixed mode)
