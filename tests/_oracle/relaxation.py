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
from track_gen._src import warp_relax


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
    """Arc-length-uniform resample of each closed loop to n points (keeps n).

    Fully batched (no per-env loop): batched ``searchsorted`` + ``gather`` so it runs as
    a handful of GPU kernels regardless of E, instead of E serial searchsorted calls.
    """
    E = center.shape[0]
    closed = torch.cat([center, center[:, :1]], dim=1)               # [E,n+1,2]
    seg = torch.linalg.norm(closed[:, 1:] - closed[:, :-1], dim=-1)  # [E,n]
    s = torch.cat([torch.zeros(E, 1, device=center.device, dtype=center.dtype),
                   torch.cumsum(seg, dim=1)], dim=1)                 # [E,n+1]
    total = s[:, -1:]
    targets = torch.arange(n, dtype=center.dtype, device=center.device)[None] * (total / n)  # [E,n]
    idx = torch.searchsorted(s[:, 1:].contiguous(), targets, right=False).clamp(max=seg.shape[1] - 1)  # [E,n] (clamp to #input segments)
    s0 = torch.gather(s, 1, idx)                                     # [E,n] arc-len at segment start
    seg_l = torch.gather(seg, 1, idx).clamp_min(1e-12)              # [E,n]
    frac = ((targets - s0) / seg_l).clamp(0.0, 1.0).unsqueeze(-1)   # [E,n,1]
    idx2 = idx.unsqueeze(-1).expand(-1, -1, 2)                       # [E,n,2]
    p0 = torch.gather(closed, 1, idx2)
    p1 = torch.gather(closed, 1, idx2 + 1)
    return p0 + frac * (p1 - p0)


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
    target = D * (1.0 + margin)
    sep_relax = float(config.relax_sep_relax)
    spc_relax = float(config.relax_spc_relax)
    bend_relax = float(config.relax_bend_relax)
    L0 = geometry.perimeter(center0) / N

    # On CUDA with Warp: run the whole fixed-iteration solve in fused kernels
    # (separation + spacing + bending per sweep, double-buffered) — no [E,N,N]
    # materialization, no per-iter sync, ~900x over the torch loop and O(E*N) memory
    # (so no chunking needed). CPU / no-Warp falls through to the pure-torch path
    # below, which stays the validated, CPU-testable reference.
    if warp_relax.should_use(center0.device, config):
        relaxed = warp_relax.xpbd_solve(center0, band, L0, config)
        return _resample_uniform(relaxed, N)

    center = center0.clone()
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
# Tangent-point + fractional-Sobolev (Repulsive Curves) backend + finisher core
# ---------------------------------------------------------------------------

def _dual_weights(center):
    e = _roll(center, -1) - center
    el = torch.linalg.norm(e, dim=-1)
    return 0.5 * (el + _roll(el, 1))


def _tp_tangents(center):
    return geometry.safe_normalize(_roll(center, -1) - _roll(center, 1))


def _tp_energy(center, pair_mask, alpha, beta, eps):
    T = _tp_tangents(center)
    w = _dual_weights(center)
    diff = center[:, None, :, :] - center[:, :, None, :]      # [E,N,N,2] (x_j - x_i)
    d2 = (diff * diff).sum(-1)
    wedge = diff[..., 0] * T[:, :, None, 1] - diff[..., 1] * T[:, :, None, 0]
    num = (wedge.abs() + eps) ** alpha
    den = (d2 + eps * eps) ** (beta * 0.5)
    k = (num / den) * (w[:, :, None] * w[:, None, :]) * pair_mask
    return k.sum()


def _length_grad(center):
    u_fwd = geometry.safe_normalize(_roll(center, -1) - center)
    return -u_fwd + _roll(u_fwd, 1)


def _ring_spectral_filter(n, s, eps_reg, device, dtype):
    k = torch.arange(n // 2 + 1, device=device, dtype=dtype)
    lam = 2.0 - 2.0 * torch.cos(2.0 * torch.pi * k / n)
    return 1.0 / (lam.clamp_min(0.0) ** s + eps_reg)


def _precondition_fft(grad, inv_filter):
    G = torch.fft.rfft(grad, dim=1) * inv_filter[None, :, None]
    return torch.fft.irfft(G, n=grad.shape[1], dim=1)


def _tp_flow(center0, band, config, n_steps, tau, early_stop):
    """Shared tangent-point/Sobolev gradient flow. Used by the standalone backend
    (early_stop=True, n_steps=tp_iters) and the smoothing finisher (early_stop=False)."""
    device = center0.device
    E, N, _ = center0.shape
    alpha = float(config.tp_alpha); beta = float(config.tp_beta)
    eps = 1e-4
    s = (beta - 1.0) / (2.0 * alpha)
    eps_reg = 1e-3
    hw = float(config.half_width)
    target = (1.0 - float(config.relax_tol)) * hw

    circ = geometry.circ_index_dist(N, device)
    pair_mask = (circ[None] > band.view(E, 1, 1)).to(center0.dtype)
    center = center0.detach().clone()
    L0_total = geometry.perimeter(center0).detach()
    inv_filter = _ring_spectral_filter(N, s, eps_reg, device, center0.dtype)
    active = torch.ones(E, dtype=torch.bool, device=device)

    for _ in range(int(n_steps)):
        if early_stop:
            th = geometry.thickness(center, band)
            active = active & (th < target)
            if not bool(active.any()):
                break
        x = center.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(_tp_energy(x, pair_mask, alpha, beta, eps), x)
        with torch.no_grad():
            g = _precondition_fft(grad, inv_filter)
            lg = _length_grad(center)
            Ainv_lg = _precondition_fft(lg, inv_filter)
            num = (g * lg).sum(dim=(1, 2))
            den = (lg * Ainv_lg).sum(dim=(1, 2)).clamp_min(1e-12)
            g = g - (num / den)[:, None, None] * Ainv_lg
            g = g - g.mean(dim=1, keepdim=True)
            gmax = torch.linalg.norm(g, dim=-1).amax(dim=1).clamp_min(1e-12)
            step = (tau * geometry.mean_seg_len(center) / gmax)[:, None, None] * g
            move = active[:, None, None].to(center.dtype) if early_stop else 1.0
            center = center - step * move
            cur_len = geometry.perimeter(center).clamp_min(1e-9)
            bc = center.mean(dim=1, keepdim=True)
            scale = (L0_total / cur_len)[:, None, None]
            if early_stop:
                scale = torch.where(active[:, None, None], scale, torch.ones_like(scale))
            center = bc + (center - bc) * scale
    return _resample_uniform(center, N)


def _relax_tp(center0, band, config):
    return _tp_flow(center0, band, config, n_steps=config.tp_iters, tau=config.tp_tau, early_stop=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {"xpbd": _relax_xpbd}  # energy/tp_sobolev added in later tasks
_BACKENDS["energy"] = _relax_energy
_BACKENDS["tp_sobolev"] = _relax_tp


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
    outs = [backend(center[sl], band[sl], config)
            for sl in _chunks(center.shape[0], config.relax_chunk_size)]
    out = torch.cat(outs, dim=0)
    if config.smooth_finish:
        fb = _band(out, config)
        outs = [_tp_flow(out[sl], fb[sl], config,
                         n_steps=config.smooth_finish_iters, tau=config.smooth_finish_tau,
                         early_stop=False)
                for sl in _chunks(out.shape[0], config.relax_chunk_size)]
        out = torch.cat(outs, dim=0)
    return out
