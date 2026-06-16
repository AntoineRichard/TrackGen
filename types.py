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

import torch
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

    # --- Fourier params ---
    num_harmonics: int = 5  # K
    decay_p: int = 2  # decay exponent: amplitude ~ amp / k**decay_p
    amplitude: float = 1.0
    num_centerline_samples: int = 256  # Fourier dense sample count (M_max)

    # --- Width params ---
    half_width: float = 0.1  # w_max
    alpha: float = 0.9  # curvature safety fraction; w * kappa <= alpha < 1
    clamp_self_distance: bool = False
    self_distance_margin: float = 0.0
    self_distance_band: int = 8
    self_distance_decimation: int = 64

    # --- Output params ---
    num_points: int = 256  # N
    output_mode: str = "fixed"  # one of {"fixed", "constant_spacing"}
    spacing: float = 0.1
    N_max: int = 256

    # --- Robustness params ---
    max_regen_iters: int = 10
    turning_tol: float = 0.1
    w_floor: float = 1e-3  # validity: every real point must have w > w_floor


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
