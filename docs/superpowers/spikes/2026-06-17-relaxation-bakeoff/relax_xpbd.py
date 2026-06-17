"""XPBD / position-based-dynamics constraint-projection solver for relaxing
race-track centerlines so a constant-width inflation (half_width) becomes valid.

Closed bead-chain model. Fully batched across E envs and vectorized over the N
beads / N*N pairs. No per-track python loops; a fixed number of Jacobi-style
projection sweeps with under-relaxation.

Constraints:
  - SEPARATION (key): every non-adjacent pair (circ-index-dist > band) closer
    than D = 2*half_width is pushed symmetrically apart to D. Jacobi-averaged
    (divide each bead's accumulated correction by its violated-pair count) and
    under-relaxed, so a bead in many violated pairs does not overshoot.
  - SPACING / inextensibility: each edge projected toward rest length
    L0 = perimeter/N (per track). Jacobi-averaged over the 2 incident edges.
  - BENDING (optional): cap the turn angle at each bead so local radius >= R_min
    by pulling the apex toward its neighbours' midpoint when too sharp.

Geometry conventions mirror common.py (closed loop, indices wrap).
"""
from __future__ import annotations
import time
import torch

import common


def _roll(x, k):
    return torch.roll(x, shifts=k, dims=1)


def _safe_norm(v, eps=1e-9):
    return torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


# ---------------------------------------------------------------------------
# Constraint projections (each returns a per-bead displacement [E,N,2])
# ---------------------------------------------------------------------------

def _separation_correction(center, mask_keep, D, margin):
    """Symmetric push for every non-adjacent pair closer than D*(1+margin).

    mask_keep: [E,N,N] bool, True where the pair is a real (non-adjacent,
    non-self) candidate. Returns Jacobi-averaged per-bead displacement.
    """
    E, N, _ = center.shape
    diff = center[:, :, None, :] - center[:, None, :, :]      # [E,N,N,2] i - j
    dist = _safe_norm(diff)                                   # [E,N,N,1]
    target = D * (1.0 + margin)
    pen = (target - dist.squeeze(-1)).clamp_min(0.0)          # [E,N,N] penetration
    violated = (pen > 0) & mask_keep                          # [E,N,N]
    # each bead i moves +0.5 * pen * unit(i-j) away from j
    unit = diff / dist                                        # [E,N,N,2]
    corr = 0.5 * pen.unsqueeze(-1) * unit                     # [E,N,N,2] contribution to i
    corr = corr * violated.unsqueeze(-1)
    disp = corr.sum(dim=2)                                    # [E,N,2] sum over j
    cnt = violated.sum(dim=2).clamp_min(1).unsqueeze(-1)      # [E,N,1]
    return disp / cnt, violated


def _spacing_correction(center, L0):
    """Project each of the N edges toward rest length L0 (per track). Each bead
    is in 2 edges -> average the two contributions (divide by 2)."""
    nxt = _roll(center, -1)
    d = nxt - center                                          # edge i -> i+1
    dist = _safe_norm(d)
    unit = d / dist
    err = (dist.squeeze(-1) - L0.unsqueeze(1))                # [E,N]
    # half the correction to each endpoint: bead i gets +0.5*err*unit,
    # bead i+1 gets -0.5*err*unit.
    fwd = 0.5 * err.unsqueeze(-1) * unit                      # applied to bead i (pull toward i+1 if too long)
    disp = fwd - _roll(fwd, 1)                                # bead i: +fwd_i (edge i) - fwd_{i-1} (edge i-1)
    return disp / 2.0


def _bending_correction(center, R_min):
    """If local Menger radius < R_min, pull apex bead toward midpoint of its
    neighbours (reduces turn sharpness).

    The raw move is `deficit * (mid - apex)` with deficit in [0,1]; multiplying
    by a relaxation > 1 outside can overshoot the midpoint and flip the corner,
    so the per-bead move is returned as `deficit*(mid-apex)` and the caller's
    relaxation is applied, but the *applied* step is clamped to never exceed the
    full apex->midpoint vector (see relax loop) to keep it stable."""
    pp, pc, pn = _roll(center, 1), center, _roll(center, -1)
    a, b, c = pc - pp, pn - pc, pn - pp
    la = _safe_norm(a).squeeze(-1)
    lb = _safe_norm(b).squeeze(-1)
    lc = _safe_norm(c).squeeze(-1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    area = 0.5 * cross.abs()
    kappa = 4.0 * area / (la * lb * lc).clamp_min(1e-12)      # [E,N]
    radius = 1.0 / kappa.clamp_min(1e-12)
    mid = 0.5 * (pp + pn)
    toward = mid - pc                                         # [E,N,2] apex -> midpoint
    deficit = (R_min - radius).clamp_min(0.0) / R_min         # [0,1] fraction
    return deficit.unsqueeze(-1) * toward, toward, (radius < R_min)


# ---------------------------------------------------------------------------
# Arc-length resampler (keeps N points), batched
# ---------------------------------------------------------------------------

def _resample_uniform(center, N):
    E = center.shape[0]
    closed = torch.cat([center, center[:, :1]], dim=1)        # [E,N+1,2]
    seg = torch.linalg.norm(closed[:, 1:] - closed[:, :-1], dim=-1)  # [E,N]
    s = torch.cat([torch.zeros(E, 1), torch.cumsum(seg, dim=1)], dim=1)  # [E,N+1]
    total = s[:, -1:]
    targets = torch.arange(N, dtype=torch.float32)[None] * (total / N)   # [E,N]
    out = torch.empty(E, N, 2)
    for e in range(E):
        idx = torch.searchsorted(s[e, 1:], targets[e], right=False).clamp(max=seg.shape[1] - 1)
        frac = ((targets[e] - s[e, idx]) / seg[e, idx].clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
        out[e] = closed[e, idx] + frac * (closed[e, idx + 1] - closed[e, idx])
    return out


# ---------------------------------------------------------------------------
# Main relax
# ---------------------------------------------------------------------------

def relax(center0, half_width, band, **hp):
    iters       = int(hp.get("iters", 100))
    sep_relax   = float(hp.get("sep_relax", 1.0))
    spc_relax   = float(hp.get("spc_relax", 1.0))
    bend_relax  = float(hp.get("bend_relax", 0.0))   # 0 => bending off
    margin      = float(hp.get("margin", 0.04))
    resample    = bool(hp.get("resample", True))
    resample_every = int(hp.get("resample_every", 0))  # 0 => only at end

    E, N, _ = center0.shape
    D = 2.0 * half_width
    R_min = half_width
    device = center0.device

    center = center0.clone()
    L0 = common.perimeter(center0) / N                       # [E] rest spacing (fixed from init)

    circ = common.circ_index_dist(N, device)                 # [N,N]
    mask_keep = circ[None] > band.view(E, 1, 1)              # [E,N,N] non-adjacent real pairs

    n_viol_hist = []
    for it in range(iters):
        disp = torch.zeros_like(center)

        sep_disp, violated = _separation_correction(center, mask_keep, D, margin)
        disp = disp + sep_relax * sep_disp

        spc_disp = _spacing_correction(center, L0)
        disp = disp + spc_relax * spc_disp

        if bend_relax > 0.0:
            bend_disp, toward, _ = _bending_correction(center, R_min)
            step = bend_relax * bend_disp
            # clamp: never move the apex past its neighbours' midpoint (|step| of the
            # bend component <= |apex->midpoint|), which would flip the corner.
            max_len = torch.linalg.norm(toward, dim=-1, keepdim=True)
            step_len = torch.linalg.norm(step, dim=-1, keepdim=True)
            scale = (max_len / step_len.clamp_min(1e-12)).clamp(max=1.0)
            disp = disp + step * scale

        center = center + disp

        if resample_every and (it + 1) % resample_every == 0:
            center = _resample_uniform(center, N)

        if (it + 1) % max(1, iters // 10) == 0:
            n_viol_hist.append(int(violated.sum().item()) // 2)

    if resample:
        center = _resample_uniform(center, N)

    info = {
        "iters": iters,
        "sep_relax": sep_relax, "spc_relax": spc_relax, "bend_relax": bend_relax,
        "margin": margin, "resample": resample,
        "n_viol_hist": n_viol_hist,
    }
    return center, info


def main():
    center0 = common.load_tracks()
    hw = 0.03
    band = common.band_per_track(center0, hw)

    # Chosen via ablation (see report): bending is the dominant lever because the
    # binding constraint is *local curvature*, not pairwise separation (59/64 tracks
    # are curvature-limited at init). The apex-toward-midpoint move is clamped to the
    # apex->midpoint vector so it cannot flip a corner, which makes strong bending
    # (relax=1.5) stable. margin=0.15 over-inflates separation slightly so the 2% tol
    # is comfortably met. 150 sweeps is past the convergence knee (~100).
    hp = dict(iters=150, sep_relax=1.0, spc_relax=1.0, bend_relax=1.5,
              margin=0.15, resample=True)

    t0 = time.time()
    center_relaxed, info = relax(center0, hw, band, **hp)
    seconds = time.time() - t0
    iters = info["iters"]

    sc = common.evaluate("XPBD projection", center0, center_relaxed, hw, seconds, iters)
    common.print_scorecard(sc)
    print("info:", {k: v for k, v in info.items() if k != "valid_mask"})
    common.plot_before_after(center0, center_relaxed, hw, "/tmp/tg_run/bakeoff/after_xpbd.png")


if __name__ == "__main__":
    main()
