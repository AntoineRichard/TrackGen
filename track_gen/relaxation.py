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
    margin = float(config.relax_margin)
    D = 2.0 * hw
    # Aim BOTH constraints slightly past the validity target (the separation already
    # over-shoots to D*(1+margin); give bending the same headroom via R_min). The final
    # arc-length resample can shift per-point Menger curvature, so a track relaxed only
    # to the bare target can drop back under it; the margin absorbs that. We run a FIXED
    # number of sweeps (no per-track early stop) so every track fully converges to a
    # smooth, resample-stable shape — early-stopping froze under-converged tracks whose
    # thickness then collapsed on the final resample.
    R_min = hw * (1.0 + margin)
    sep_relax = float(config.relax_sep_relax)
    spc_relax = float(config.relax_spc_relax)
    bend_relax = float(config.relax_bend_relax)

    center = center0.clone()
    L0 = geometry.perimeter(center0) / N
    circ = geometry.circ_index_dist(N, center0.device)
    mask_keep = circ[None] > band.view(E, 1, 1)

    for _ in range(int(config.relax_iters)):
        disp = sep_relax * _separation_disp(center, mask_keep, D, margin)
        disp = disp + spc_relax * _spacing_disp(center, L0)
        if bend_relax > 0.0:
            bend, toward = _bending_disp(center, R_min)
            step = bend_relax * bend
            max_len = torch.linalg.norm(toward, dim=-1, keepdim=True)
            step_len = torch.linalg.norm(step, dim=-1, keepdim=True)
            disp = disp + step * (max_len / step_len.clamp_min(1e-12)).clamp(max=1.0)
        center = center + disp

    return _resample_uniform(center, N)


# ---------------------------------------------------------------------------
# Energy (Adam soft-penalty) backend
# ---------------------------------------------------------------------------

def _energy(center, x0, circ, band, D, w_sep, w_len, w_bend, w_anchor, L0):
    E, N, _ = center.shape
    dmat = torch.cdist(center, center)                      # [E,N,N]
    mask = circ[None] > band.view(E, 1, 1)
    viol = torch.relu(D - dmat) * mask
    e_sep = 0.5 * w_sep * (viol ** 2).sum()
    seg = _roll(center, -1) - center
    seglen = torch.linalg.norm(seg, dim=-1)
    e_len = w_len * ((seglen - L0.view(E, 1)) ** 2).sum()
    lap = _roll(center, -1) - 2.0 * center + _roll(center, 1)
    e_bend = w_bend * (lap ** 2).sum()
    e_anchor = w_anchor * ((center - x0) ** 2).sum()
    return e_sep + e_len + e_bend + e_anchor


def _relax_energy(center0, band, config):
    E, N, _ = center0.shape
    D = 2.0 * float(config.half_width)
    circ = geometry.circ_index_dist(N, center0.device).to(center0.dtype)
    L0 = geometry.mean_seg_len(center0).detach()
    x0 = center0.detach().clone()
    x = center0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=float(config.energy_lr))
    for _ in range(int(config.energy_steps)):
        opt.zero_grad(set_to_none=True)
        e = _energy(x, x0, circ, band, D, float(config.energy_w_sep), float(config.energy_w_len),
                    float(config.energy_w_bend), float(config.energy_w_anchor), L0)
        e.backward()
        opt.step()
    return _resample_uniform(x.detach(), N)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {"xpbd": _relax_xpbd}  # energy/tp_sobolev added in later tasks
_BACKENDS["energy"] = _relax_energy


def _chunks(e: int, size):
    if not size or size >= e:
        yield slice(0, e)
        return
    for start in range(0, e, size):
        yield slice(start, min(start + size, e))


def relax(center: torch.Tensor, config) -> torch.Tensor:
    if not config.relax_enable:
        return center
    backend = _BACKENDS.get(config.relax_solver)
    if backend is None:
        raise ValueError(f"Unknown relax_solver {config.relax_solver!r}; "
                         f"expected one of {sorted(_BACKENDS)}.")
    band = _band(center, config)
    outs = []
    for sl in _chunks(center.shape[0], config.relax_chunk_size):
        outs.append(backend(center[sl], band[sl], config))
    return torch.cat(outs, dim=0)
