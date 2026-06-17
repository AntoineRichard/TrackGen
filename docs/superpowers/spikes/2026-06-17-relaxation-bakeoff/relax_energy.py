"""Differentiable energy-minimization solver for relaxing race-track centerlines.

SPIKE (throwaway): treat the N bead positions per track as optimization variables
of a closed bead-chain and minimize a differentiable energy with torch autograd
(Adam), batched over the E envs. Goal: make a constant-width inflation valid
(thickness >= 0.98*half_width, zero border/centerline crossings).

Energy terms (all batched, no per-track python loops in the opt loop):
  - separation:  for non-adjacent pairs (circ-index-dist > band) with d < D=2*hw,
                 w_sep * relu(D - d)^2   (smooth hinge -> quadratic, good gradients)
  - length:      w_len * (||x_{i+1}-x_i|| - L0)^2   (keeps beads ~inextensible/uniform)
  - bending:     w_bend * ||x_{i+1} - 2 x_i + x_{i-1}||^2  (curvature radius up)
  - anchor:      w_anchor * ||x - x0||^2  (stay near the original Bezier shape)

Optional final HARD-PROJECTION cleanup (reported separately, never folded into the
pure-energy number): a few Gauss-Seidel-style position corrections that push apart
violating non-adjacent pairs and re-uniformize spacing. Reported honestly.
"""
from __future__ import annotations
import torch
import common


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def _roll(x, k):
    return torch.roll(x, shifts=k, dims=1)


def _energy(center, x0, circ, band, D, w_sep, w_len, w_bend, w_anchor, L0):
    """Scalar energy (summed over all envs and beads). center: [E,N,2]."""
    E, N, _ = center.shape

    # --- separation: non-adjacent pairs closer than D ---
    dmat = torch.cdist(center, center)                       # [E,N,N]
    mask = circ[None] > band.view(E, 1, 1)                   # consider only non-adjacent
    viol = torch.relu(D - dmat) * mask                       # [E,N,N], 0 where ok/adjacent
    # each unordered pair counted twice -> 0.5 factor
    e_sep = 0.5 * w_sep * (viol ** 2).sum()

    # --- length / spacing ---
    seg = _roll(center, -1) - center                         # [E,N,2]
    seglen = torch.linalg.norm(seg, dim=-1)                  # [E,N]
    e_len = w_len * ((seglen - L0.view(E, 1)) ** 2).sum()

    # --- bending (discrete Laplacian) ---
    lap = _roll(center, -1) - 2.0 * center + _roll(center, 1)
    e_bend = w_bend * (lap ** 2).sum()

    # --- anchor to original ---
    e_anchor = w_anchor * ((center - x0) ** 2).sum()

    return e_sep + e_len + e_bend + e_anchor


# ---------------------------------------------------------------------------
# Optional hard-projection cleanup (reported separately)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _project(center, circ, band, D, n_iters=40, slack=1.001):
    """Gauss-Seidel-style hard projection: push apart violating non-adjacent pairs
    and gently re-uniformize spacing. Returns a corrected copy."""
    E, N, _ = center.shape
    c = center.clone()
    L0 = common.mean_seg_len(center).view(E, 1)
    for _ in range(n_iters):
        # separation: for each bead, find its nearest violating non-adjacent neighbour
        dmat = torch.cdist(c, c)                              # [E,N,N]
        mask = circ[None] > band.view(E, 1, 1)
        dmat_masked = dmat.masked_fill(~mask, float("inf"))
        dmat_masked = dmat_masked.masked_fill(dmat_masked < 1e-9, float("inf"))
        nn_d, nn_j = dmat_masked.min(dim=-1)                 # [E,N]
        target = D * slack
        need = (nn_d < target)                               # [E,N]
        j = nn_j                                             # partner index per bead
        xj = torch.gather(c, 1, j.unsqueeze(-1).expand(-1, -1, 2))
        dir_ = c - xj
        dlen = torch.linalg.norm(dir_, dim=-1, keepdim=True).clamp_min(1e-9)
        dir_ = dir_ / dlen
        push = (target - nn_d).clamp_min(0.0).unsqueeze(-1)  # how far to move
        # move each endpoint half the deficit
        delta = 0.5 * push * dir_ * need.unsqueeze(-1)
        c = c + delta

        # spacing relaxation: pull beads toward midpoint to keep ~uniform length
        seg = _roll(c, -1) - c
        seglen = torch.linalg.norm(seg, dim=-1, keepdim=True).clamp_min(1e-9)
        excess = (seglen - L0.unsqueeze(-1))                 # [E,N,1]
        corr = 0.25 * excess * (seg / seglen)
        c = c + corr - _roll(corr, 1)
    return c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def relax(center0, half_width, band, **hp):
    """Relax centerlines via differentiable energy minimization (Adam).

    Returns (center_relaxed [E,N,2], info dict). N is preserved (=256).
    """
    device = center0.device
    E, N, _ = center0.shape
    D = 2.0 * half_width

    # hyperparameters (tuned defaults; see report).
    # 800 Adam steps @ lr=3e-3 is the knee: more steps with this stiff w_sep
    # starts oscillating and re-introduces border crossings.
    steps    = int(hp.get("steps", 800))
    lr       = float(hp.get("lr", 3e-3))
    w_sep    = float(hp.get("w_sep", 80.0))
    w_len    = float(hp.get("w_len", 8.0))
    w_bend   = float(hp.get("w_bend", 1.0))
    w_anchor = float(hp.get("w_anchor", 0.01))
    do_project = bool(hp.get("project", False))
    proj_iters = int(hp.get("proj_iters", 60))

    circ = common.circ_index_dist(N, device).float()
    L0 = common.mean_seg_len(center0).detach()               # [E] rest length (fixed)
    x0 = center0.detach().clone()

    x = center0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        e = _energy(x, x0, circ, band, D, w_sep, w_len, w_bend, w_anchor, L0)
        e.backward()
        opt.step()

    center_relaxed = x.detach()
    info = {"iters": steps, "optimizer": "adam", "lr": lr,
            "w_sep": w_sep, "w_len": w_len, "w_bend": w_bend, "w_anchor": w_anchor}

    if do_project:
        center_relaxed = _project(center_relaxed, circ, band, D, n_iters=proj_iters)
        info["projected"] = True
        info["proj_iters"] = proj_iters

    return center_relaxed, info


def main():
    import time
    torch.manual_seed(0)

    center0 = common.load_tracks()                           # [64,256,2]
    hw = 0.03
    band = common.band_per_track(center0, hw)

    # --- pure energy ---
    t0 = time.time()
    center_relaxed, info = relax(center0, hw, band)
    seconds = time.time() - t0
    sc = common.evaluate("Energy / gradient", center0, center_relaxed, hw, seconds, info["iters"])
    common.print_scorecard(sc)
    common.plot_before_after(center0, center_relaxed, hw, "/tmp/tg_run/bakeoff/after_energy.png")


if __name__ == "__main__":
    main()
