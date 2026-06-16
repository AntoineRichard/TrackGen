# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inflation stage: turn a dense Centerline into a Track (outer/center/inner + metadata).

Pure batched torch, device-agnostic, CPU-testable. Depends only on geometry.py and
the warp-free leaf dataclasses in types.py (Track, TrackGenConfig). It must NOT import
from track_generator (that would create a circular import); Track comes from .types.
"""

import math
from collections import namedtuple

import torch

from . import geometry
from .types import Track, TrackGenConfig  # noqa: F401  (TrackGenConfig used for typing)

# Intermediate result of the resample stage; replaced by a full Track once inflate() is complete.
_ResampleResult = namedtuple("_ResampleResult", ["center", "count"])


def _valid_mask_from_points(points: torch.Tensor) -> torch.Tensor:
    """A point is valid iff neither of its two coordinates is NaN. Returns [E, M] bool."""
    return ~torch.isnan(points).any(dim=-1)


def _resample_stage(centerline, config) -> _ResampleResult:
    """Masked arc-length resample of the centerline per config.output_mode.

    fixed mode            -> num = config.num_points, count == num.
    constant_spacing mode -> spacing = config.spacing, padded to config.N_max with NaN.
    """
    points = centerline.points  # [E, M_max, 2]
    valid_mask = _valid_mask_from_points(points)  # [E, M_max]

    if config.output_mode == "fixed":
        resampled, count = geometry.arc_length_resample(
            points, num=config.num_points, valid_mask=valid_mask
        )
    elif config.output_mode == "constant_spacing":
        resampled, count = geometry.arc_length_resample(
            points, spacing=config.spacing, valid_mask=valid_mask, n_max=config.N_max
        )
    else:
        raise ValueError(f"Unknown output_mode: {config.output_mode!r}")

    return _ResampleResult(center=resampled, count=count)


def _frame_curvature_stage(center: torch.Tensor):
    """Compute the per-point frame and curvature on the resampled centerline.

    Returns:
        T:     [E, N, 2] unit tangent (central difference).
        Nrm:   [E, N, 2] unit left-normal, Nrm = (-T_y, T_x).
        kappa: [E, N]    non-negative Menger curvature.
    """
    T, Nrm = geometry.tangents_normals(center)
    kappa = geometry.menger_curvature(center)
    return T, Nrm, kappa
