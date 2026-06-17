# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centerline relaxation: reshape a closed, arc-length-uniform centerline so a
constant-width inflation becomes valid (thickness >= half_width).

Pure batched torch, device-agnostic (CPU+GPU), CPU-testable, RNG-free (deterministic).
Three selectable backends behind relax(): xpbd (default), energy, tp_sobolev, plus an
optional tangent-point/Sobolev smoothing finisher. Reference (validated) spikes live
under docs/superpowers/spikes/2026-06-17-relaxation-bakeoff/.
"""
from __future__ import annotations

import torch

from . import geometry


def _roll(x, k):
    return torch.roll(x, shifts=k, dims=1)


def _safe_norm(v, eps=1e-9):
    return torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _band(center: torch.Tensor, config) -> torch.Tensor:
    """Excluded-neighbour half-window per track: round(D / L0), >= 1. [E] long."""
    if config.relax_band is not None:
        E = center.shape[0]
        return torch.full((E,), int(config.relax_band), dtype=torch.long, device=center.device)
    D = 2.0 * float(config.half_width)
    L0 = geometry.mean_seg_len(center)
    return (D / L0).round().long().clamp_min(1)


def _resample_uniform(center: torch.Tensor, n: int) -> torch.Tensor:
    """Arc-length-uniform resample of each closed loop to n points (keeps n)."""
    E = center.shape[0]
    closed = torch.cat([center, center[:, :1]], dim=1)               # [E,n+1,2]
    seg = torch.linalg.norm(closed[:, 1:] - closed[:, :-1], dim=-1)  # [E,n]
    s = torch.cat([torch.zeros(E, 1, device=center.device, dtype=center.dtype),
                   torch.cumsum(seg, dim=1)], dim=1)                 # [E,n+1]
    total = s[:, -1:]
    targets = torch.arange(n, dtype=center.dtype, device=center.device)[None] * (total / n)
    out = torch.empty_like(center)
    for e in range(E):
        idx = torch.searchsorted(s[e, 1:], targets[e], right=False).clamp(max=seg.shape[1] - 1)
        frac = ((targets[e] - s[e, idx]) / seg[e, idx].clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
        out[e] = closed[e, idx] + frac * (closed[e, idx + 1] - closed[e, idx])
    return out


# ---------------------------------------------------------------------------
# XPBD backend (default)
# ---------------------------------------------------------------------------

def _separation_disp(center, mask_keep, D, margin):
    """Jacobi-averaged symmetric push for non-adjacent pairs closer than D*(1+margin)."""
    diff = center[:, :, None, :] - center[:, None, :, :]    # [E,N,N,2] i - j
    dist = _safe_norm(diff)                                 # [E,N,N,1]
    target = D * (1.0 + margin)
    pen = (target - dist.squeeze(-1)).clamp_min(0.0)        # [E,N,N]
    violated = (pen > 0) & mask_keep
    unit = diff / dist
    corr = 0.5 * pen.unsqueeze(-1) * unit * violated.unsqueeze(-1)
    disp = corr.sum(dim=2)                                  # [E,N,2]
    cnt = violated.sum(dim=2).clamp_min(1).unsqueeze(-1)
    return disp / cnt


def _spacing_disp(center, L0):
    """Project each edge toward rest length L0; each bead is in 2 edges -> /2."""
    d = _roll(center, -1) - center
    dist = _safe_norm(d)
    unit = d / dist
    err = (dist.squeeze(-1) - L0.unsqueeze(1))
    fwd = 0.5 * err.unsqueeze(-1) * unit
    return (fwd - _roll(fwd, 1)) / 2.0


def _bending_disp(center, R_min):
    """Pull the apex toward its neighbours' midpoint when local radius < R_min.
    Returns (raw_disp, apex->midpoint vector) so the caller can clamp the step
    to never flip the corner."""
    pp, pc, pn = _roll(center, 1), center, _roll(center, -1)
    a, b, c = pc - pp, pn - pc, pn - pp
    la, lb, lc = (_safe_norm(x).squeeze(-1) for x in (a, b, c))
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    area = 0.5 * cross.abs()
    kappa = 4.0 * area / (la * lb * lc).clamp_min(1e-12)
    radius = 1.0 / kappa.clamp_min(1e-12)
    mid = 0.5 * (pp + pn)
    toward = mid - pc
    deficit = (R_min - radius).clamp_min(0.0) / R_min
    return deficit.unsqueeze(-1) * toward, toward


def _relax_xpbd(center0, band, config):
    E, N, _ = center0.shape
    hw = float(config.half_width)
    D = 2.0 * hw
    R_min = hw
    target = (1.0 - float(config.relax_tol)) * hw
    sep_relax = float(config.relax_sep_relax)
    spc_relax = float(config.relax_spc_relax)
    bend_relax = float(config.relax_bend_relax)
    margin = float(config.relax_margin)

    center = center0.clone()
    L0 = geometry.perimeter(center0) / N
    circ = geometry.circ_index_dist(N, center0.device)
    mask_keep = circ[None] > band.view(E, 1, 1)
    active = torch.ones(E, dtype=torch.bool, device=center0.device)

    for _ in range(int(config.relax_iters)):
        if not bool(active.any()):
            break
        disp = sep_relax * _separation_disp(center, mask_keep, D, margin)
        disp = disp + spc_relax * _spacing_disp(center, L0)
        if bend_relax > 0.0:
            bend, toward = _bending_disp(center, R_min)
            step = bend_relax * bend
            max_len = torch.linalg.norm(toward, dim=-1, keepdim=True)
            step_len = torch.linalg.norm(step, dim=-1, keepdim=True)
            disp = disp + step * (max_len / step_len.clamp_min(1e-12)).clamp(max=1.0)
        center = torch.where(active[:, None, None], center + disp, center)
        th = geometry.thickness(center, band)
        active = active & (th < target)

    return _resample_uniform(center, N)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {"xpbd": _relax_xpbd}  # energy/tp_sobolev added in later tasks


def relax(center: torch.Tensor, config) -> torch.Tensor:
    """Reshape a closed, arc-length-uniform centerline [E,N,2] so thickness >= half_width.
    Dispatches on config.relax_solver; returns the relaxed centerline (same N)."""
    if not config.relax_enable:
        return center
    backend = _BACKENDS.get(config.relax_solver)
    if backend is None:
        raise ValueError(f"Unknown relax_solver {config.relax_solver!r}; "
                         f"expected one of {sorted(_BACKENDS)}.")
    band = _band(center, config)
    return backend(center, band, config)
