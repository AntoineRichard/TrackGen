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


def _width_stage(center: torch.Tensor, kappa: torch.Tensor, config, eps: float = 1e-8):
    """Per-point half-width via curvature clamp + optional self-distance clamp.

    Args:
        center: [E, N, 2] resampled centerline.
        kappa:  [E, N]    non-negative curvature.
        config: TrackGenConfig (half_width, alpha, clamp_self_distance,
                self_distance_margin, self_distance_band, self_distance_decimation).
    Returns:
        w: [E, N] non-negative half-width.
    """
    w_max = float(config.half_width)
    alpha = float(config.alpha)

    w_curv = torch.where(
        kappa > eps,
        alpha / kappa.clamp_min(eps),
        torch.full_like(kappa, w_max),
    )
    w = w_curv.clamp_max(w_max)

    if config.clamp_self_distance:
        d = geometry.nearest_nonadjacent_distance(
            center, config.self_distance_band, config.self_distance_decimation
        )  # [E, N]
        w_self = 0.5 * (d - float(config.self_distance_margin))
        w = torch.minimum(w, w_self)

    return w.clamp_min(0.0)


def _offset_stage(center: torch.Tensor, Nrm: torch.Tensor, w: torch.Tensor):
    """Offset the centerline by +/- w along the left-normal and assign outer/inner.

    outer = the candidate with the LARGER |polygon_area|; inner = the smaller.
    Robust to loop orientation. Areas are computed on NaN-zeroed copies so
    constant_spacing padding (NaN slots) cannot poison the per-env area.

    Args:
        center: [E, N, 2]
        Nrm:    [E, N, 2] unit left-normal
        w:      [E, N]    half-width
    Returns:
        outer: [E, N, 2], inner: [E, N, 2]
    """
    wn = w.unsqueeze(-1) * Nrm  # [E, N, 2]
    a = center + wn
    b = center - wn

    area_a = geometry.polygon_area(torch.nan_to_num(a, nan=0.0)).abs()  # [E]
    area_b = geometry.polygon_area(torch.nan_to_num(b, nan=0.0)).abs()  # [E]

    a_is_outer = (area_a >= area_b).view(-1, 1, 1)  # [E, 1, 1] for broadcasting
    outer = torch.where(a_is_outer, a, b)
    inner = torch.where(a_is_outer, b, a)
    return outer, inner


def _real_point_mask(count: torch.Tensor, n: int, device) -> torch.Tensor:
    """[E, N] bool mask: slot j is real iff j < count[env]. Fixed mode -> all True."""
    idx = torch.arange(n, device=device).unsqueeze(0)  # [1, N]
    return idx < count.unsqueeze(1)  # [E, N]


def _validity_stage(center, w, count, gen_valid, config) -> torch.Tensor:
    """Per-track validity: generation flag AND closed-loop turning AND width floor AND no-NaN.

    Args:
        center:    [E, N, 2]
        w:         [E, N]
        count:     [E]   number of real points per env.
        gen_valid: [E]   bool generation-time validity.
        config:    TrackGenConfig (turning_tol, w_floor).
    Returns:
        valid: [E] bool.
    """
    e, n = w.shape
    real = _real_point_mask(count, n, w.device)  # [E, N]

    turning = geometry.turning_number(center)  # [E]
    turn_ok = (turning.abs() - 2.0 * math.pi).abs() <= float(config.turning_tol)

    w_ok = torch.where(real, w > float(config.w_floor), torch.ones_like(real)).all(dim=1)

    nan_per_point = torch.isnan(center).any(dim=-1)  # [E, N]
    nan_real = (nan_per_point & real).any(dim=1)  # [E]
    no_nan = ~nan_real

    return gen_valid.to(torch.bool) & turn_ok & w_ok & no_nan
