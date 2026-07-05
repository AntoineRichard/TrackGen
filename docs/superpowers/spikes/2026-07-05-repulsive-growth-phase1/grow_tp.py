"""Repulsive-growth phase-1 generator spike.

Tests the generation strategy of Henrich et al., "Generating Race Tracks With
Repulsive Curves" (IEEE 10645670; based on Yu/Schumacher/Crane "Repulsive
Curves", SIGGRAPH 2021) as a phase-1 centerline generator for track_gen:

  * start each env from a small circle (embedded by construction),
  * ratchet a per-env target length upward (~1%/iter) while the tangent-point
    energy keeps the curve self-avoiding,
  * confine growth in a disc domain seeded with per-env random disc obstacles
    (paper formulation: domain wall and obstacles are point rings with plain
    inverse-power repulsion, p = beta - alpha; wall weight 1, inner 0.25),
  * resample to per-env CONSTANT SPACING 0.6*hw (the runtime pipeline's
    pre-XPBD calibration; feeding the raw N=256 curve at ~3x finer spacing put
    XPBD in a sawtooth regime -- see README "spacing-mismatch fix"), then run
    the STANDARD tail (oracle XPBD relax, bucketed by per-env count -> constant-
    width inflate) and score yield / diversity / wall-clock against the runtime
    ``bezier`` generator.

This inverts the two documented TP-Sobolev failures (2026-06-17 bake-off,
2026-06-18 finisher): TP is never asked to fix a jagged curvature-limited
curve or untangle anything -- the curve is embedded at every step and TP works
in its native separation-limited regime.

SPIKE (throwaway). Batched torch over E envs, dense O(N^2) pairs (E=64, N=256
is trivial); a production port would use wp.Bvh for the far-field sums.
Deterministic: fixed seeds, no wall-clock in the math.

  .venv/bin/python docs/superpowers/spikes/2026-07-05-repulsive-growth-phase1/grow_tp.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tests._oracle import geometry as G  # noqa: E402
from tests._oracle import relaxation as R  # noqa: E402

SPIKE_DIR = Path(__file__).resolve().parent
SEED = 11


# ---------------------------------------------------------------------------
# Per-env random obstacle layouts (domain wall ring + K inner disc rings)
# ---------------------------------------------------------------------------

def sample_obstacles(E, r_dom, r_init, gen, device, disc_clearance=0.0,
                     wall_clearance=0.0, k_range=(8, 12), r_frac=(0.02, 0.045),
                     c_frac=0.90, n_wall=96, n_disc=12):
    """Returns (pts [E,M,2], mass [E,M], weight [E,M], layouts, n_wall) NaN-padded.

    Mirrors the reference Unity layout (EnergyCurve.cs): a domain-wall ring
    (weight 1.0) plus K inner disc rings (weight 0.25) placed at EVENLY-SPACED
    ANGLES with a uniform-random RADIAL distance, not by rejection-sampled 2D
    position. Their recipe: numObstacles=10 inner discs of radius
    innerObstacleRadius=1 in an outerRadius=4*innerRadius domain (obstacle/domain
    ratio 1/40=0.025), radial distance ~ U[innerRadius+innerObstacleRadius+1,
    outerRadius]. Here ``r_frac`` (fraction of r_dom) defaults to (0.02, 0.045)
    to bracket their 0.025 ratio, ``k_range`` to (8, 12) to bracket their 10.

    Angular stratification (one disc per 2*pi/k wedge, plus a per-env random
    phase for cross-env variety) gives a busier, more uniform obstacle field than
    the old 2D rejection sampler while still leaving the annulus between discs
    open for the curve to thread -- the folds-per-track lever.

    Clearances are split: ``disc_clearance`` (default 0 -- rings sit at the exact
    physical radius, matching the paper; the old 0.6*hw inflated the drawn discs
    by 25-55% of their radius, the "obstacles look bigger than they are" effect)
    and ``wall_clearance`` (kept small so the grown curve stays a touch inside the
    drawn domain during growth; the wall is deactivated at target length anyway).
    The lower radial bound still clears the init circle by the RING edge so the
    p=3 energy can't blow up at t=0 (first-attempt failure mode).

    mass = ring arc spacing at the point-placement radius (Obstacle.cs lumped
    length). Returns n_wall so the caller can build the deactivation column mask.
    """
    K_max = k_range[1]
    M = n_wall + K_max * n_disc
    pts = torch.full((E, M, 2), float("nan"), device=device)
    mass = torch.zeros(E, M, device=device)
    weight = torch.zeros(E, M, device=device)

    r_wall = r_dom + wall_clearance
    ang_w = torch.arange(n_wall, device=device) * (2 * torch.pi / n_wall)
    wall = r_wall * torch.stack([torch.cos(ang_w), torch.sin(ang_w)], dim=-1)
    pts[:, :n_wall] = wall
    mass[:, :n_wall] = 2 * torch.pi * r_wall / n_wall
    weight[:, :n_wall] = 1.0

    layouts = []
    n_placed = 0
    ang_d = torch.arange(n_disc, device=device) * (2 * torch.pi / n_disc)
    ring = torch.stack([torch.cos(ang_d), torch.sin(ang_d)], dim=-1)
    for e in range(E):
        k = int(torch.randint(k_range[0], k_range[1] + 1, (1,), generator=gen,
                              device=device))
        phase = float(2 * torch.pi * torch.rand(1, generator=gen, device=device))
        discs = []
        for j in range(k):
            r = float((r_frac[0] + (r_frac[1] - r_frac[0])
                       * torch.rand(1, generator=gen, device=device)) * r_dom)
            # Radial band: ring's inner edge clears the init circle (else the
            # p=3 energy explodes at t=0); outer bound c_frac*r_dom stays inside
            # the wall. Reference uses U[innerR+obsR+1, outerR].
            lo = r_init + r + disc_clearance + 0.05 * r_dom
            hi = c_frac * r_dom
            if lo >= hi:
                continue
            rad = lo + (hi - lo) * float(torch.rand(1, generator=gen, device=device))
            ang = phase + j * 2 * torch.pi / k
            c = rad * torch.tensor([np.cos(ang), np.sin(ang)], device=device)
            r_ring = r + disc_clearance
            discs.append((c, r))
            sl = slice(n_wall + j * n_disc, n_wall + (j + 1) * n_disc)
            pts[e, sl] = c[None] + r_ring * ring
            mass[e, sl] = 2 * torch.pi * r_ring / n_disc
            weight[e, sl] = 0.25
        n_placed += len(discs)
        layouts.append([(c.cpu().numpy(), r) for c, r in discs])
    print(f"[obstacles] {n_placed} discs placed over {E} envs "
          f"({n_placed / E:.2f}/env); disc clearance {disc_clearance:.4f}, "
          f"wall clearance {wall_clearance:.4f}, obstacle/domain ratio "
          f"{r_frac[0]:.3f}-{r_frac[1]:.3f}")
    return pts, mass, weight, layouts, n_wall


def obstacle_energy(x, obs_pts, obs_mass_w, p):
    """sum_i sum_m weight_m * mass_m / |x_i - p_m|^p  (Obstacle.cs BodyEnergy).

    x [E,N,2]; obs_pts [E,M,2] NaN-padded; obs_mass_w [E,M] = mass*weight
    (0 on padding). NaN-padded points are neutralized via nan_to_num after
    masking, so autograd sees finite values everywhere.
    """
    d2 = ((x[:, :, None, :] - torch.nan_to_num(obs_pts)[:, None, :, :]) ** 2).sum(-1)
    inv = (d2 + 1e-8) ** (-p / 2)
    return (inv * obs_mass_w[:, None, :]).sum()


# ---------------------------------------------------------------------------
# Growth flow: TP self-avoidance + obstacle repulsion + length ratchet
# ---------------------------------------------------------------------------

def grow(E, N, r_init, r_dom, L_final, obs_pts, obs_mass_w, device,
         alpha=3.0, beta=6.0, tau=0.4, growth=0.012, settle_iters=40,
         resample_every=25, w_len=30.0, snap_every=None, snap_envs=(),
         n_wall=96, deac=True, deac_wall=False):
    """Grow E closed curves from circles of radius r_init to perimeters
    L_final [E]. Returns (center [E,N,2], snapshots, n_iters).

    ``deac`` mirrors the reference's ``deacObsAfterScaling``: once an env's
    ratcheted target reaches its final length L_final[e], that env's obstacle
    weights are zeroed for the rest of the run, so its settle phase is pure
    TP + length constraint. The reference deactivates the ENTIRE obstacle list
    (the wall is obstacles[0]), but ``deac_wall`` defaults FALSE here: the
    reference stops its flow on stall, whereas this spike runs a fixed iteration
    budget, so with the wall gone pure TP + fixed length unfolds every curve back
    into a circle. Keeping the wall preserves the confinement that holds the
    folds; only the inner discs are dropped at target length.

    Growth follows the paper's moving length CONSTRAINT (not a penalty). Each
    iter: (1) descend the TP + obstacle gradient under the fractional-Sobolev
    preconditioner; (2) project the step orthogonal to the length-increase
    direction *in the Sobolev inner product* (the Repulsive-Curves constrained
    step, oracle ``_tp_flow``); (3) rescale the curve to the ratcheted target
    L_target about its barycenter. The Sobolev-orthogonal projection is what
    keeps the enforced length growth in LOW modes -- slack folds into smooth
    buckles instead of the high-frequency sawtooth that a *naive* uniform
    rescale produced on the first attempt (see README failure modes).

    A soft penalty w_len*((L - L_target)/L_init)^2 was tried first and shown
    (spike probe) to be ~10^3-10^5x too weak against the plain-inverse-power
    wall/obstacle repulsion -- the curve just collapsed. w_len is kept as a
    small live regularizer nudging L toward the fresh target before the hard
    rescale snaps it exactly; the constraint does the real work.

    Pair exclusion is a constant +-2 neighbours (paper): the TP wedge factor
    discounts along-curve neighbours itself; a wide thickness-style band would
    hide sub-band wavelengths and let noise absorb the ratcheted length.
    """
    p_obs = beta - alpha
    s = (beta - 1.0) / (2.0 * alpha)
    eps = 1e-4
    inv_filter = R._ring_spectral_filter(N, s, 1e-3, device, torch.float32)

    ang = torch.arange(N, device=device) * (2 * torch.pi / N)
    center = r_init * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    center = center[None].repeat(E, 1, 1).contiguous()

    L_init = G.perimeter(center)                                    # [E]
    L_target = L_init.clone()
    n_grow = int(np.ceil(np.log(float((L_final / L_init).max()))
                         / np.log1p(growth) * 1.6))  # 60% stall allowance
    n_iters = n_grow + settle_iters

    circ = G.circ_index_dist(N, device)
    pair_mask = (circ[None] > 2).float().expand(E, N, N)
    snapshots = {e: [] for e in snap_envs}

    # Deactivation column mask: which obstacle points survive once an env has
    # reached its target length. deac_wall -> drop the wall too (reference).
    M = obs_mass_w.shape[1]
    col_keep = torch.ones(M, device=device)
    if deac:
        col_keep[n_wall:] = 0.0                      # inner discs always dropped
        if deac_wall:
            col_keep[:n_wall] = 0.0                  # wall dropped too (paper)
    deac_reported = False

    for it in range(n_iters):
        # Hard rescale keeps L == L_target, so the ratchet advances every iter
        # up to L_final (no per-env stall needed once the constraint holds).
        L_target = torch.minimum(L_target * (1.0 + growth), L_final)
        if snap_every and it % snap_every == 0:
            for e in snap_envs:
                snapshots[e].append(center[e].detach().cpu().clone())

        # Per-env obstacle deactivation at target length (reference: on
        # TargetLengthReached, obstacles.ForEach(Disable)). Reached envs keep
        # only col_keep columns; others keep all obstacles.
        if deac:
            reached = L_target >= (L_final - 1e-9)          # [E]
            scale = torch.where(reached[:, None], col_keep[None, :],
                                torch.ones_like(col_keep)[None, :])
            obs_mw_it = obs_mass_w * scale
            if not deac_reported and bool(reached.any()):
                print(f"[deac] first env reached target at iter {it} "
                      f"({int(reached.sum())}/{E} envs); "
                      f"{'wall+discs' if deac_wall else 'discs only'} zeroed")
                deac_reported = True
        else:
            obs_mw_it = obs_mass_w

        x = center.detach().clone().requires_grad_(True)
        L = G.perimeter(x)
        energy = (R._tp_energy(x, pair_mask, alpha, beta, eps)
                  + obstacle_energy(x, obs_pts, obs_mw_it, p_obs)
                  + (w_len * ((L - L_target) / L_init).pow(2)).sum())
        (grad,) = torch.autograd.grad(energy, x)

        with torch.no_grad():
            g = R._precondition_fft(grad, inv_filter)
            # Project out the length-increase direction in the Sobolev metric
            # (Repulsive-Curves constrained gradient), then hard-rescale to
            # the target length about the barycenter.
            lg = R._length_grad(center)
            Ainv_lg = R._precondition_fft(lg, inv_filter)
            num = (g * lg).sum(dim=(1, 2))
            den = (lg * Ainv_lg).sum(dim=(1, 2)).clamp_min(1e-12)
            g = g - (num / den)[:, None, None] * Ainv_lg
            g = g - g.mean(dim=1, keepdim=True)      # barycenter pin
            gmax = torch.linalg.norm(g, dim=-1).amax(dim=1).clamp_min(1e-12)
            center = center - (tau * G.mean_seg_len(center) / gmax)[:, None, None] * g
            cur_len = G.perimeter(center).clamp_min(1e-9)
            bc = center.mean(dim=1, keepdim=True)
            center = bc + (center - bc) * (L_target / cur_len)[:, None, None]

        if (it + 1) % resample_every == 0:
            center = R._resample_uniform(center, N)

    center = R._resample_uniform(center, N)
    for e in snap_envs:
        snapshots[e].append(center[e].detach().cpu().clone())
    return center, snapshots, n_iters


# ---------------------------------------------------------------------------
# Scoring (validity rule of the 2026-06-17 bake-off) + diversity
# ---------------------------------------------------------------------------

def validity(center, half_width, tol=0.02):
    band = (2.0 * half_width / G.mean_seg_len(center)).round().long().clamp_min(1)
    th = G.thickness(center, band)
    T, Nrm = G.tangents_normals(center)
    outer, inner = center + half_width * Nrm, center - half_width * Nrm
    xings = (G.self_intersections(center) + G.self_intersections(outer)
             + G.self_intersections(inner))
    valid = (th >= (1.0 - tol) * half_width) & (xings == 0)
    return valid, th, xings


def validity_batch(center, count, half_width, tol=0.02):
    """Per-env validity on the REAL (non-NaN) points of a NaN-padded [E,n_max,2]
    buffer with per-env ``count``. Reuses the dense single-env ``validity`` on each
    env's first ``count[e]`` points. Returns (valid [E] bool, th [E], xings [E])."""
    E = center.shape[0]
    valid = torch.zeros(E, dtype=torch.bool, device=center.device)
    th = torch.full((E,), float("nan"), device=center.device)
    xings = torch.zeros(E, dtype=torch.long, device=center.device)
    for e in range(E):
        n_e = int(count[e])
        if n_e < 3:
            continue
        v, t, x = validity(center[e, :n_e][None], half_width, tol)
        valid[e], th[e], xings[e] = v[0], t[0], x[0]
    return valid, th, xings


def disp_per_env(a, b, count):
    """Mean per-bead |a-b| over each env's real points; NaN for degenerate envs."""
    E = a.shape[0]
    d = torch.full((E,), float("nan"), device=a.device)
    for e in range(E):
        n_e = int(count[e])
        if n_e >= 3:
            d[e] = torch.linalg.norm(a[e, :n_e] - b[e, :n_e], dim=-1).mean()
    return d


def diversity_batch(center, count, valid):
    """Diversity over the valid envs of a NaN-padded [E,n_max,2] buffer, measured
    per env on real points (bead counts differ across envs)."""
    Ps, As, ks = [], [], []
    for e in range(center.shape[0]):
        n_e = int(count[e])
        if not bool(valid[e]) or n_e < 3:
            continue
        c = center[e, :n_e][None]
        Ps.append(G.perimeter(c))
        As.append(G.polygon_area(c).abs())
        ks.append(G.menger_curvature(c).amax(dim=1))
    P, A, kmax = torch.cat(Ps), torch.cat(As), torch.cat(ks)
    compact = 4 * torch.pi * A / P.pow(2)
    return {"compact_mean": compact.mean().item(), "compact_std": compact.std().item(),
            "kmax_med": kmax.median().item(), "kmax_std": kmax.std().item(),
            "perim_mean": P.mean().item(), "perim_std": P.std().item()}


def tail_bucketed(center, count, cfg):
    """Torch-oracle XPBD tail on a NaN-padded [E,n_max,2] buffer, bucketed by
    per-env bead count so equal-count envs relax together as a dense batch.
    Results are written back NaN-padded (same width). Constant spacing 0.6*hw
    makes each bucket's L0/band match the runtime XPBD calibration exactly."""
    relaxed = torch.full_like(center, float("nan"))
    for n_e in torch.unique(count):
        n_e = int(n_e)
        if n_e < 3:
            continue
        idx = (count == n_e).nonzero(as_tuple=True)[0]
        out = R.relax(center[idx, :n_e], cfg)          # [b, n_e, 2]
        relaxed[idx, :out.shape[1]] = out
    return relaxed


def fmt_div(name, d):
    return (f"{name}: compactness {d['compact_mean']:.3f}+-{d['compact_std']:.3f}  "
            f"kmax med {d['kmax_med']:.1f} std {d['kmax_std']:.1f}  "
            f"perimeter {d['perim_mean']:.2f}+-{d['perim_std']:.2f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _real(center_e, count_e):
    """First ``count_e`` (real, non-NaN) points of one env's NaN-padded row."""
    n_e = int(count_e)
    return center_e[:n_e] if n_e >= 3 else center_e[:0]


def plot_grid(center, count, valid, half_width, path, title, layouts=None,
              r_dom=None, n_show=16):
    n = min(n_show, center.shape[0])
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    for e, ax in enumerate(axes.flat):
        if e >= n:
            ax.axis("off")
            continue
        if r_dom is not None:
            ax.add_patch(plt.Circle((0, 0), r_dom, fill=False, color="0.7", ls="--"))
        if layouts is not None:
            for c, r in layouts[e]:
                ax.add_patch(plt.Circle(c, r, color="0.8"))
        col = "#1a9641" if valid[e] else "#d7191c"
        c = _real(center[e], count[e])
        if c.shape[0] >= 3:
            _, Nrm = G.tangents_normals(c[None])
            outer, inner = c + half_width * Nrm[0], c - half_width * Nrm[0]
            for curve, lw in ((c, 0.8), (outer, 1.4), (inner, 1.4)):
                cc = torch.cat([curve, curve[:1]]).cpu().numpy()
                ax.plot(cc[:, 0], cc[:, 1], "-", color=col if lw > 1 else "0.5", lw=lw)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"env {e} {'valid' if valid[e] else 'INVALID'}", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_pre_post(grown, relaxed, count, v_pre, v_post, half_width, path, title,
                   layouts=None, r_dom=None, n_show=16):
    """Overlay pre-XPBD (grown) and post-XPBD (relaxed) centerlines per env.

    Both buffers are NaN-padded [E,n_max,2] sharing one per-env ``count`` (relax
    keeps N). Pre-XPBD is drawn as a thin orange line; post-XPBD as a thicker
    green/red line (green if v_post[e] else red), plus its constant-width
    inflated borders (center +- half_width * normal) in the same color, thin.
    """
    n = min(n_show, grown.shape[0])
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = np.atleast_1d(axes)
    for e, ax in enumerate(axes.flat):
        if e >= n:
            ax.axis("off")
            continue
        if r_dom is not None:
            ax.add_patch(plt.Circle((0, 0), r_dom, fill=False, color="0.7", ls="--"))
        if layouts is not None:
            for c, r in layouts[e]:
                ax.add_patch(plt.Circle(c, r, color="0.8"))
        col = "#1a9641" if v_post[e] else "#d7191c"

        gc = _real(grown[e], count[e])
        rc = _real(relaxed[e], count[e])
        pre_line = post_line = None
        if gc.shape[0] >= 3:
            pre = torch.cat([gc, gc[:1]]).cpu().numpy()
            pre_line, = ax.plot(pre[:, 0], pre[:, 1], "-", color="#d95f02", lw=0.9,
                                label="grown (pre-XPBD)")
        if rc.shape[0] >= 3:
            post = torch.cat([rc, rc[:1]]).cpu().numpy()
            post_line, = ax.plot(post[:, 0], post[:, 1], "-", color=col, lw=1.8,
                                 label="after XPBD + inflation")
            _, Nrm = G.tangents_normals(rc[None])
            outer, inner = rc + half_width * Nrm[0], rc - half_width * Nrm[0]
            for curve in (outer, inner):
                cc = torch.cat([curve, curve[:1]]).cpu().numpy()
                ax.plot(cc[:, 0], cc[:, 1], "-", color=col, lw=0.7)

        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        pre_ok = "ok" if v_pre[e] else "FAIL"
        post_ok = "ok" if v_post[e] else "FAIL"
        ax.set_title(f"env {e}: pre {pre_ok} -> post {post_ok}", fontsize=9)
        if e == 0 and pre_line is not None and post_line is not None:
            ax.legend(handles=[pre_line, post_line], loc="upper right", fontsize=6)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_snapshots(snapshots, layouts, r_dom, path):
    envs = sorted(snapshots)
    cols = len(snapshots[envs[0]])
    fig, axes = plt.subplots(len(envs), cols, figsize=(2.6 * cols, 2.6 * len(envs)))
    axes = np.atleast_2d(axes)
    for i, e in enumerate(envs):
        for j, snap in enumerate(snapshots[e]):
            ax = axes[i, j]
            ax.add_patch(plt.Circle((0, 0), r_dom, fill=False, color="0.7", ls="--"))
            for c, r in layouts[e]:
                ax.add_patch(plt.Circle(c, r, color="0.85"))
            cc = torch.cat([snap, snap[:1]]).numpy()
            ax.plot(cc[:, 0], cc[:, 1], "-", color="#d95f02", lw=1.2)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            lim = 1.15 * r_dom
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            if i == 0:
                ax.set_title(f"snap {j}", fontsize=9)
    fig.suptitle("TP-repulsive growth under length ratchet (rows = envs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--growth", type=float, default=0.008)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--r-dom-frac", type=float, default=0.35,
                    help="domain radius as a fraction of the baseline median perimeter. "
                         "Sets the absolute scale; the init circle and target length are then "
                         "derived from it by the reference's ratios (--dom-init-ratio, "
                         "--grow-mult).")
    ap.add_argument("--dom-init-ratio", type=float, default=4.0,
                    help="domain radius / init-circle radius. Reference uses outerRadius = "
                         "4*innerRadius (EnergyCurve.cs). r_init = r_dom / this. The old spike "
                         "used ~6 (a smaller init circle).")
    ap.add_argument("--w-len", type=float, default=30.0,
                    help="length-ratchet penalty weight")
    ap.add_argument("--grow-mult", type=float, nargs=2, default=(4.5, 5.5),
                    help="per-env final perimeter as a multiple of the INIT-CIRCLE perimeter "
                         "(reference: lengthScale=6, i.e. target = 6*initial length). Per-env "
                         "U[lo, hi]. THE primary folds-per-track lever: at dom-init-ratio 4 the "
                         "overfill (target / domain circumference) is grow_mult/(2*dom_init_ratio) "
                         "so 4.5-5.5 -> overfill ~1.28 (near the reference's 1.5). Yield/fold "
                         "frontier (seed 11, keep-wall deac): 4.5-5.5 -> 64/64 compactness 0.146; "
                         "5-6 -> 63/64, 0.133; 6-7 -> 63/64, 0.104. Default 4.5-5.5 keeps 64/64 "
                         "with rich folds; raise for denser mazes at 63/64. Capped by the "
                         "fill-fraction check printed at startup.")
    ap.add_argument("--disc-clearance", type=float, default=0.0,
                    help="extra radius added to inner disc repulsion rings beyond their physical "
                         "radius. Default 0 (paper: rings at the exact obstacle radius). The old "
                         "spike used 0.6*hw, inflating the drawn discs 25-55%.")
    ap.add_argument("--wall-clearance-hw", type=float, default=0.0,
                    help="wall repulsion ring clearance, in units of hw. Kept at 0 by default; "
                         "raise if the fixed-length curve escapes the drawn domain during growth.")
    ap.add_argument("--k-range", type=int, nargs=2, default=(8, 12),
                    help="inner disc count per env, U{lo..hi}. Reference: 10.")
    ap.add_argument("--r-frac", type=float, nargs=2, default=(0.02, 0.045),
                    help="inner disc radius as a fraction of r_dom. Reference ratio 0.025.")
    ap.add_argument("--no-deac", action="store_true",
                    help="disable the deactivate-obstacles-after-target-length step entirely "
                         "(obstacles stay live through the settle phase). With deac ON (default) "
                         "the disc halos close up and the domain fills uniformly; OFF leaves the "
                         "curve clustered with clear halos around discs (the 'exaggerated distance' "
                         "look). deac ON costs ~1 env vs OFF at equal geometry.")
    ap.add_argument("--deac-wall", action="store_true",
                    help="also deactivate the domain WALL at target length (the reference's literal "
                         "behavior: it drops the whole obstacle list). NOT the default: the "
                         "reference stops its flow on stall, but we run fixed iters, so with the "
                         "wall gone pure TP+fixed-length unfolds every curve into a CIRCLE "
                         "(compactness ~1.0). Default keeps the wall so confinement preserves the "
                         "folds; only the inner discs are dropped.")
    ap.add_argument("--settle-iters", type=int, default=40,
                    help="pure TP+length iters after target length (post-deactivation relax).")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(SEED)

    # ---------------- Phase 0: runtime baseline (bezier generator) ----------
    import warp as wp  # noqa: F401  (initializes)
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator

    E, N = args.E, args.N
    cfg = TrackGenConfig(num_envs=E, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=SEED, num_envs=E, device=dev))
    gen.generate()                                     # warmup (module load etc.)
    t0 = time.time()
    track = gen.generate()
    sec_base = time.time() - t0
    hw = float(cfg.half_width)

    base_valid = torch.as_tensor(track.valid.numpy(), dtype=torch.bool)
    n_max = track.center.shape[0] // E
    raw = torch.as_tensor(track.center.numpy()).reshape(E, n_max, 2)
    count = torch.as_tensor(track.count.numpy(), dtype=torch.long)
    pad = torch.arange(n_max)[None] >= count[:, None]
    raw = raw.masked_fill(pad[:, :, None], float("nan"))
    base_center = torch.stack([
        R._resample_uniform(raw[e][~pad[e]][None], N)[0] if base_valid[e]
        else torch.full((N, 2), float("nan")) for e in range(E)]).to(dev)
    P_ref = float(G.perimeter(base_center[base_valid.to(dev)]).median())
    print(f"[baseline] bezier E={E}: {int(base_valid.sum())}/{E} valid, "
          f"{sec_base:.3f}s/generate, P_ref={P_ref:.2f}, hw={hw}")

    # ---------------- Phase 1: repulsive growth -----------------------------
    # Geometry follows the reference's ratios: r_dom sets the scale, r_init =
    # r_dom / dom_init_ratio (reference outerRadius = 4*innerRadius), and the
    # per-env target is grow_mult * the INIT-CIRCLE perimeter (reference
    # lengthScale=6). Overfill (target / domain circumference) and the planar
    # constant-width fill fraction (strip area P*2*hw / domain area) are the
    # capacity check -- fill must stay well under ~60% or the folds pinch.
    r_dom = args.r_dom_frac * P_ref
    r_init = r_dom / args.dom_init_ratio
    L_init_circle = 2 * torch.pi * r_init
    g_lo, g_hi = args.grow_mult
    gen_t = torch.Generator(device=dev).manual_seed(SEED)
    L_final = L_init_circle * (g_lo + (g_hi - g_lo)
                               * torch.rand(E, generator=gen_t, device=dev))
    L_med = float(L_final.median())
    overfill = L_med / (2 * torch.pi * r_dom)
    fill = L_med * 2 * hw / (torch.pi * r_dom ** 2)
    print(f"[geometry] r_dom={r_dom:.3f} (={args.r_dom_frac}*P_ref) "
          f"r_init={r_init:.3f} (dom/init={args.dom_init_ratio}) "
          f"L_init_circle={L_init_circle:.3f} grow_mult={g_lo}-{g_hi} -> "
          f"L_final med {L_med:.2f} (vs P_ref {P_ref:.2f}); "
          f"overfill (target/wall-circumference)={overfill:.2f}, "
          f"fill-fraction={fill:.0%}"
          + ("  [WARN fill>60%: folds may pinch]" if fill > 0.60 else ""))

    obs_pts, obs_mass, obs_w, layouts, n_wall = sample_obstacles(
        E, r_dom, r_init, gen_t, dev, disc_clearance=args.disc_clearance,
        wall_clearance=args.wall_clearance_hw * hw,
        k_range=tuple(args.k_range), r_frac=tuple(args.r_frac))
    t0 = time.time()
    grown_fine, snapshots, n_iters = grow(
        E, N, r_init, r_dom, L_final, obs_pts, obs_mass * obs_w, dev,
        alpha=args.alpha, beta=args.beta, tau=args.tau, growth=args.growth,
        w_len=args.w_len, settle_iters=args.settle_iters, n_wall=n_wall,
        deac=not args.no_deac, deac_wall=args.deac_wall,
        snap_every=30, snap_envs=(0, 1, 2, 3))
    sec_grow = time.time() - t0

    # Runtime-matched tail resampling. The runtime pipeline resamples the
    # centerline to CONSTANT SPACING 0.6*hw before XPBD (warp_pipeline.py calls
    # resample_constant_spacing(..., config.spacing), config.spacing default =
    # 0.6*half_width), so the Jacobi solver's per-track rest length L0 and
    # exclusion band are CALIBRATED for that spacing. Feeding the N=256 grown
    # curve straight in (spacing ~P/256 ~0.021, ~3x finer) is the XPBD sawtooth
    # regime: an over-fine band + tiny rest length make Jacobi separation
    # over-correct into high-frequency zigzag on pinched folds.
    #
    # We resample per-env at spacing 0.6*hw into a NaN-padded [E, n_max, 2] buffer
    # with per-env counts (exactly like the codebase / warp_pipeline), using the
    # oracle's own G.arc_length_resample (the torch mirror of the runtime Warp
    # resampler). Growth itself stays at N=256 (the TP flow needs that
    # resolution); only the tail input is coarsened. Pre-tail validity is on the
    # resampled curve so pre/post are directly comparable.
    spacing = 0.6 * hw
    grown, count = G.arc_length_resample(grown_fine, spacing=spacing)   # [E,n_max,2],[E]
    cnt = count.float()
    print(f"[tail-resample] per-env constant spacing {spacing:.4f} (=0.6*hw): "
          f"count min/med/max {int(cnt.min())}/{int(cnt.median())}/{int(cnt.max())}, "
          f"n_max={grown.shape[1]} (growth kept N={N})")

    v_pre, th_pre, x_pre = validity_batch(grown, count, hw)
    print(f"[grow] {n_iters} iters in {sec_grow:.2f}s "
          f"({1000 * sec_grow / n_iters:.1f} ms/iter): pre-tail valid "
          f"{int(v_pre.sum())}/{E} (thickness med {th_pre.nanmedian():.4f} "
          f"vs hw {hw}, {int((x_pre > 0).sum())} envs with crossings)")

    # ---------------- Phase 2: standard tail (oracle XPBD -> inflate) -------
    # Shipped the torch ORACLE relax bucketed by per-env count (not the Warp tail):
    # the oracle is the validated reference the Warp XPBD is allclose to, and it
    # avoids hand-building the Warp _Scratch buffers for a throwaway spike. Envs
    # sharing a bead count are stacked and relaxed together; results are written
    # back NaN-padded. Constant spacing 0.6*hw makes each bucket's L0 and band
    # match the runtime calibration exactly (band = round(2*hw/0.06) = 3).
    t0 = time.time()
    relaxed = tail_bucketed(grown, count, cfg)
    sec_tail = time.time() - t0
    v_post, th_post, x_post = validity_batch(relaxed, count, hw)
    disp = disp_per_env(relaxed, grown, count)
    print(f"[tail] oracle XPBD ({cfg.relax_iters} it, bucketed) in {sec_tail:.2f}s: valid "
          f"{int(v_post.sum())}/{E}, XPBD displacement med {disp.nanmedian():.4f} "
          f"(how much phase 2 had to fix)")

    # ---------------- Phase 3: diversity + figures --------------------------
    base_count = torch.where(base_valid.to(dev), N, 0)
    print("[diversity] " + fmt_div("grown+tail", diversity_batch(relaxed, count, v_post)))
    print("[diversity] " + fmt_div("baseline  ",
                                    diversity_batch(base_center, base_count, base_valid.to(dev))))
    print(f"[wall-clock] growth+tail {sec_grow + sec_tail:.2f}s vs baseline "
          f"generate {sec_base:.3f}s (E={E})")

    plot_snapshots(snapshots, layouts, r_dom, SPIKE_DIR / "growth-snapshots.png")
    plot_grid(relaxed.cpu(), count.cpu(), v_post.cpu(), hw, SPIKE_DIR / "grown-grid.png",
              f"repulsive growth -> XPBD -> inflate ({int(v_post.sum())}/{E} valid)",
              layouts=layouts, r_dom=r_dom)
    plot_grid(base_center.cpu(), base_count.cpu(), base_valid, hw,
              SPIKE_DIR / "baseline-grid.png",
              f"runtime bezier baseline ({int(base_valid.sum())}/{E} valid)")
    plot_pre_post(
        grown.cpu(), relaxed.cpu(), count.cpu(), v_pre.cpu(), v_post.cpu(), hw,
        SPIKE_DIR / "pre-post-xpbd.png",
        f"XPBD tail: pre {int(v_pre.sum())}/{E} valid -> post {int(v_post.sum())}/{E} valid",
        layouts=layouts, r_dom=r_dom)
    print(f"figures: {SPIKE_DIR}/growth-snapshots.png, grown-grid.png, baseline-grid.png, "
          f"pre-post-xpbd.png")


if __name__ == "__main__":
    main()
