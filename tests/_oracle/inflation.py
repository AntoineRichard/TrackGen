"""Inflation stage: turn a dense Centerline into a Track (outer/center/inner + metadata).

Pure batched torch, device-agnostic, CPU-testable. Depends only on geometry.py and
the warp-free leaf dataclasses in types.py (Track, TrackGenConfig). It must NOT import
from track_generator (that would create a circular import); Track comes from .types.
"""

import math
from collections import namedtuple

import torch

from . import geometry
from track_gen._src.types import Track, TrackGenConfig  # noqa: F401  (TrackGenConfig used for typing)

# Intermediate result of the resample stage; replaced by a full Track once inflate() is complete.
_ResampleResult = namedtuple("_ResampleResult", ["center", "count"])


def _valid_mask_from_points(points: torch.Tensor) -> torch.Tensor:
    """A point is valid iff neither of its two coordinates is NaN. Returns [E, M] bool."""
    return ~torch.isnan(points).any(dim=-1)


def _resample_stage(centerline, config) -> _ResampleResult:
    """Masked constant-spacing arc-length resample of the centerline.

    Spacing = config.spacing (~0.6*half_width), padded to config.N_max with NaN; the per-env
    real-point count is returned. (The legacy fixed-count mode was dropped.)
    """
    points = centerline.points  # [E, M_max, 2]
    valid_mask = _valid_mask_from_points(points)  # [E, M_max]
    resampled, count = geometry.arc_length_resample(
        points, spacing=config.spacing, valid_mask=valid_mask, n_max=config.N_max
    )
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


def _width_stage(center: torch.Tensor, kappa: torch.Tensor, config):
    """Constant half-width. Relaxation guarantees thickness >= half_width upstream, so
    no curvature/self-distance clamp is needed. kappa is accepted for signature
    compatibility but unused."""
    w = torch.full(center.shape[:2], float(config.half_width), device=center.device, dtype=center.dtype)
    return w


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


def _validity_stage(center, w, count, gen_valid, config, outer=None, inner=None) -> torch.Tensor:
    """Real per-track validity: generation flag AND closed-loop turning AND width floor
    AND no-NaN AND thickness >= (1-tol)*half_width AND zero border self-intersections.

    Note: this torch path is the verification *oracle*, and its gate here does NOT
    count-mask the NaN padding -- the turning and thickness checks run over the full
    ``center`` tensor, so in ``output_mode="constant_spacing"`` the NaN padding beyond
    ``count`` poisons those two metrics (-> NaN -> fails) and such padded tracks are
    flagged invalid. That is a limitation of this oracle path only. The runtime
    pure-Warp pipeline's ``warp_pipeline._validity_k`` IS count-masked, so
    ``output_mode="constant_spacing"`` is fully supported on the Warp path."""
    e, n = w.shape
    real = _real_point_mask(count, n, w.device)  # [E, N]

    turning = geometry.turning_number(center)
    turn_ok = (turning.abs() - 2.0 * math.pi).abs() <= float(config.turning_tol)
    w_ok = torch.where(real, w > float(config.w_floor), torch.ones_like(real)).all(dim=1)
    nan_per_point = torch.isnan(center).any(dim=-1)
    no_nan = ~(nan_per_point & real).any(dim=1)

    D = 2.0 * float(config.half_width)
    L0 = geometry.mean_seg_len(center).clamp_min(1e-9)
    band = (D / L0).round().long().clamp_min(1)
    th = geometry.thickness(center, band)
    th_ok = th >= (1.0 - float(config.relax_tol)) * float(config.half_width)

    if outer is None or inner is None:
        border_ok = torch.ones(e, dtype=torch.bool, device=center.device)
    else:
        crossings = geometry.self_intersections(torch.nan_to_num(outer, nan=0.0)) + \
                    geometry.self_intersections(torch.nan_to_num(inner, nan=0.0))
        border_ok = crossings == 0

    return gen_valid.to(torch.bool) & turn_ok & w_ok & no_nan & th_ok & border_ok


def _arclength(center: torch.Tensor, count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative arc length [E, N] (0 at index 0) and closed-loop total length [E].

    Uses only real points: padded (NaN) slots contribute zero-length segments. The
    closing wrap segment (last real point -> point 0) is added explicitly so the
    total length matches the closed-loop definition in both output modes.
    """
    e, n, _ = center.shape
    real = _real_point_mask(count, n, center.device)  # [E, N]

    nxt = torch.roll(center, shifts=-1, dims=1)
    seg = nxt - center  # [E, N, 2]
    seg_len = torch.linalg.norm(seg, dim=-1)  # [E, N]
    real_next = torch.roll(real, shifts=-1, dims=1)
    seg_real = real & real_next  # [E, N]
    seg_len = torch.where(seg_real, seg_len, torch.zeros_like(seg_len))

    cum = torch.cumsum(seg_len, dim=1)  # length at i is sum of seg[0..i]
    arclen = torch.zeros_like(cum)
    arclen[:, 1:] = cum[:, :-1]

    # Closing wrap segment: last real point (index count-1) -> first point (index 0).
    # In fixed mode this is already captured by seg at index N-1; in constant_spacing
    # mode it is NOT (the next slot after count-1 is padding), so add it explicitly.
    last_idx = (count - 1).clamp_min(0)  # [E]
    first_pt = center[:, 0, :]  # [E, 2]
    last_pt = center[torch.arange(e, device=center.device), last_idx]  # [E, 2]
    wrap_already_counted = real_next.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # [E] bool
    wrap_len = torch.linalg.norm(first_pt - last_pt, dim=-1)  # [E]
    # Add the wrap only when it was NOT already counted as a real segment and count>=2.
    add_wrap = (~wrap_already_counted) & (count >= 2)
    wrap_contrib = torch.where(add_wrap, wrap_len, torch.zeros_like(wrap_len))

    length = seg_len.sum(dim=1) + wrap_contrib
    return arclen, length


def inflate(centerline, config) -> Track:
    """Inflate a dense Centerline into a Track (outer/center/inner + frame + metadata).

    Stages: resample -> frame+curvature -> width -> offset -> validity -> assemble.
    """
    res = _resample_stage(centerline, config)
    center, count = res.center, res.count
    T, Nrm, kappa = _frame_curvature_stage(center)
    w = _width_stage(center, kappa, config)
    outer, inner = _offset_stage(center, Nrm, w)
    # Border self_intersections is optional (config.validity_border_check, default off):
    # redundant with the thickness/separation gate, so skip it by passing None borders.
    _bc = getattr(config, "validity_border_check", False)
    valid = _validity_stage(center, w, count, centerline.valid, config,
                            outer=(outer if _bc else None), inner=(inner if _bc else None))
    arclen, length = _arclength(center, count)
    return Track(outer=outer, center=center, inner=inner, tangent=T, normal=Nrm,
                 arclen=arclen, length=length, valid=valid, count=count)
