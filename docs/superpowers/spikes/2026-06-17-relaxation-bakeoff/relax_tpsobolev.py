"""Tangent-point energy + (fractional-Sobolev preconditioned) gradient flow for
relaxing race-track centerlines (Repulsive Curves, Yu/Schumacher/Crane SIGGRAPH'21),
evaluated on the SHARED bake-off harness (common.py).

SPIKE (throwaway). Batched pure-torch on CPU over E=64 envs, vectorized over N points.

WHAT THIS IMPLEMENTS
--------------------
Energy (tangent-point, vertex form, double sum over the closed polyline):

    k(x_i, x_j) = | (x_j - x_i) wedge T_i |^alpha / |x_j - x_i|^beta      (2D scalar wedge)
    E = sum_{i,j, j not adjacent to i} k(x_i,x_j) * w_i * w_j

with w_i = dual edge length (0.5*(|e_{i-1}|+|e_i|)) so the double sum approximates
the arc-length double integral. alpha=2, beta=4.5 (Repulsive-Curves regime beta-alpha>1).
The kernel automatically discounts along-curve neighbours (chord ~ parallel to T => wedge ~ 0)
and blows up when a DISTANT strand approaches => smooth surrogate for min self-distance.
We exclude pairs with circular-index-distance <= band (the harness's k_skip exclusion).

PRECONDITIONER (this is the key trick from the paper, and what we approximate):
The faithful object is the fractional Slobodeckij inner product A of order 2s=(beta-1)/alpha
~ 1.75 (an N x N dense matrix per curve built from a double sum over edge pairs). Building
and assembling the *exact* B / B0 operator from the paper (with the |T_i wedge .|^? weights
and the 1/|x-y|^{2s+1} low-order term) correctly in a batched-torch SPIKE is heavy and
error-prone, so we use a DOCUMENTED PRAGMATIC SURROGATE that captures the essential
de-stiffening behaviour:

    A = L_ring^s + eps * I ,   s = (beta - 1) / (2*alpha) ~ 0.875

where L_ring is the graph Laplacian of the closed UNIFORM ring. Because the curves are
arc-length-uniform by construction, the unweighted ring Laplacian is the natural
discretization; it is CIRCULANT, so A^{-1} is applied as a real-FFT per-mode scaling
(eigenvalues lam_k = 2-2cos(2*pi*k/N), filter 1/(lam_k^s+eps)), O(E*N log N) per step
instead of a per-iteration dense [N,N] eigendecomposition. The high-frequency modes of E's
Hessian scale like |xi|^{2s}; multiplying the gradient by A^{-1}=L^{-s} damps exactly those
stiff high modes, giving a resolution-independent, well-conditioned descent direction. This
is precisely the "fractional-Sobolev preconditioning" idea, with the (circulant) ring
Laplacian standing in for the full dense Slobodeckij operator. (See report for honesty notes:
on a uniform closed curve these coincide up to the variable |T wedge|-weighting we drop.)

We additionally PROJECT OUT the length-change direction (Repulsive-Curves-style constraint
projection): the descent direction g is made orthogonal (in the A inner product) to the
gradient of total length, so the flow does not collapse the curve. A barycenter constraint
keeps the curve from translating.

EARLY-STOP: each track is masked out of further updates as soon as its discrete thickness
reaches 0.98*half_width (the harness target), giving minimal deviation.

API:  relax(center0, half_width, band, **hp) -> (center_relaxed [E,N,2], info dict)
"""
from __future__ import annotations
import time
import torch

import common


def _roll(x, k):
    return torch.roll(x, shifts=k, dims=1)


# ---------------------------------------------------------------------------
# Tangent-point energy + autograd gradient
# ---------------------------------------------------------------------------

def _dual_weights(center):
    """w_i = 0.5*(|e_{i-1}| + |e_i|): dual (lumped) arc-length per vertex. [E,N]."""
    e = _roll(center, -1) - center                       # edge i: x_i -> x_{i+1}
    el = torch.linalg.norm(e, dim=-1)                    # [E,N]
    return 0.5 * (el + _roll(el, 1))                     # |e_i| + |e_{i-1}|


def _tangents(center):
    """Centered unit tangent at each vertex. [E,N,2]."""
    t = _roll(center, -1) - _roll(center, 1)
    return common.safe_normalize(t)


def _tp_energy(center, pair_mask, alpha, beta, eps):
    """Tangent-point energy, scalar (summed over envs & vertex pairs).

    k(i,j) = |(x_j-x_i) wedge T_i|^alpha / |x_j-x_i|^beta, double sum over kept pairs,
    weighted by dual lengths w_i w_j. center: [E,N,2]. pair_mask: [E,N,N] bool (kept pairs)."""
    E, N, _ = center.shape
    T = _tangents(center)                                # [E,N,2]
    w = _dual_weights(center)                            # [E,N]
    diff = center[:, None, :, :] - center[:, :, None, :]  # [E,N,N,2]  (x_j - x_i) at [.,i,j]
    d2 = (diff * diff).sum(-1)                            # [E,N,N] |x_j-x_i|^2
    # 2D wedge of (x_j-x_i) with T_i : diff_x*Ty_i - diff_y*Tx_i
    wedge = diff[..., 0] * T[:, :, None, 1] - diff[..., 1] * T[:, :, None, 0]   # [E,N,N]
    num = (wedge.abs() + eps) ** alpha
    den = (d2 + eps * eps) ** (beta * 0.5)
    k = num / den                                        # [E,N,N]
    ww = w[:, :, None] * w[:, None, :]                   # [E,N,N]
    k = k * ww * pair_mask
    return k.sum()


def _length_grad(center):
    """Gradient of total polyline length wrt each vertex. [E,N,2]."""
    e_fwd = _roll(center, -1) - center                   # e_i
    u_fwd = common.safe_normalize(e_fwd)                 # unit e_i
    # vertex i appears in edge i (as start, +/-) and edge i-1 (as end)
    # d/dx_i sum |e| = -u_fwd_i (from edge i) + u_fwd_{i-1} (from edge i-1)
    return -u_fwd + _roll(u_fwd, 1)


# ---------------------------------------------------------------------------
# Fractional-Laplacian Sobolev preconditioner (graph-Laplacian surrogate)
# ---------------------------------------------------------------------------

def _ring_spectral_filter(N, s, eps_reg, device, dtype):
    """Precompute the per-mode inverse filter for A = L_ring^s + eps_reg*I on the
    closed UNIFORM ring. The unweighted ring Laplacian is circulant, diagonalized by
    the DFT, with eigenvalues lam_k = 2 - 2cos(2*pi*k/N). Since the curves are
    arc-length-UNIFORM (by construction), the unweighted ring Laplacian is the natural
    discretization and lets us apply A^{-1} as a real FFT mode-scaling (O(N log N))
    instead of a per-iteration dense [N,N] eigendecomposition. Returns inv_filter [N]
    (real, indexed by rfft frequency 0..N/2)."""
    k = torch.arange(N // 2 + 1, device=device, dtype=dtype)   # rfft bins
    lam = 2.0 - 2.0 * torch.cos(2.0 * torch.pi * k / N)        # ring Laplacian eigenvalues
    a = lam.clamp_min(0.0) ** s + eps_reg
    return 1.0 / a                                              # [N//2+1]


def _precondition_fft(grad, inv_filter):
    """g = A^{-1} grad applied per coordinate via real FFT mode-scaling. grad [E,N,2].
    A^{-1} = F^{-1} diag(inv_filter) F (circulant). O(E*N log N)."""
    G = torch.fft.rfft(grad, dim=1)                            # [E,N//2+1,2] complex
    G = G * inv_filter[None, :, None]
    return torch.fft.irfft(G, n=grad.shape[1], dim=1)          # [E,N,2] real


# ---------------------------------------------------------------------------
# Main relax
# ---------------------------------------------------------------------------

def relax(center0, half_width, band, **hp):
    device = center0.device
    E, N, _ = center0.shape

    max_iters = int(hp.get("max_iters", 80))
    alpha     = float(hp.get("alpha", 2.0))
    beta      = float(hp.get("beta", 4.5))
    eps       = float(hp.get("eps", 1e-4))      # kernel softening (length scale ~ track scale)
    tau       = float(hp.get("tau", 0.7))       # base step size (after preconditioning + normalization)
    s         = float(hp.get("s", (beta - 1.0) / (2.0 * alpha)))  # ~0.875
    eps_reg   = float(hp.get("eps_reg", 1e-3))  # A = L^s + eps_reg*I  regularizer
    project_length = bool(hp.get("project_length", True))
    target_frac = float(hp.get("target_frac", 0.98))
    stop_margin = float(hp.get("stop_margin", 1.0))   # require thickness >= stop_margin*target before freezing
    w_anchor = float(hp.get("w_anchor", 0.0))         # L2 anchor to init (bounds over-rounding)
    extra_after_conv = int(hp.get("extra_after_conv", 0))  # for hybrid smoothing: run fixed n iters, ignore early stop

    D = 2.0 * half_width
    target = target_frac * half_width
    circ = common.circ_index_dist(N, device)             # [N,N]
    pair_mask = (circ[None] > band.view(E, 1, 1)).to(center0.dtype)  # [E,N,N] kept pairs

    center = center0.detach().clone()
    x0 = center0.detach().clone()
    L0_total = common.perimeter(center0).detach()        # fixed target length
    inv_filter = _ring_spectral_filter(N, s, eps_reg, device, center0.dtype)  # [N//2+1] precomputed once

    active = torch.ones(E, dtype=torch.bool, device=device)
    iters_to_conv = torch.full((E,), -1, dtype=torch.long, device=device)

    info = {"step_times": [], "solve_times": [], "energy_hist": []}

    for it in range(max_iters):
        if extra_after_conv == 0:
            # early stop: freeze converged tracks
            th = common.thickness(center, band)           # [E]
            newly = active & (th >= stop_margin * target)
            iters_to_conv[newly] = it
            active = active & (th < stop_margin * target)
            if not active.any():
                break

        t_step = time.time()
        x = center.detach().clone().requires_grad_(True)
        e_tp = _tp_energy(x, pair_mask, alpha, beta, eps)
        # anchor: keep points near the (per-track scale-normalized) original to bound
        # over-rounding/displacement. Scaled by w_anchor * E_tp-scale so it is comparable.
        if w_anchor > 0.0:
            e_anchor = w_anchor * ((x - x0) ** 2).sum()
            e_val = e_tp + e_anchor
        else:
            e_val = e_tp
        (grad,) = torch.autograd.grad(e_val, x)          # [E,N,2] L2 gradient
        info["energy_hist"].append(float(e_tp.item()))

        with torch.no_grad():
            t_solve = time.time()
            g = _precondition_fft(grad, inv_filter)       # preconditioned descent dir (FFT solve)
            info["solve_times"].append(time.time() - t_solve)

            if project_length:
                # project g to be A-orthogonal to the length gradient (so step preserves length to 1st order)
                lg = _length_grad(center)                 # [E,N,2]
                Ainv_lg = _precondition_fft(lg, inv_filter)  # A^{-1} lg
                # remove component of g along A^{-1}lg in the A inner product:
                # <g, lg> (euclid) / <lg, A^{-1} lg> * A^{-1} lg
                num = (g * lg).sum(dim=(1, 2))            # <g, lg>
                den = (lg * Ainv_lg).sum(dim=(1, 2)).clamp_min(1e-12)
                g = g - (num / den)[:, None, None] * Ainv_lg

            # remove net translation (barycenter constraint)
            g = g - g.mean(dim=1, keepdim=True)

            # normalize per-track step so tau acts as a fraction of mean segment length
            gmax = torch.linalg.norm(g, dim=-1).amax(dim=1).clamp_min(1e-12)  # [E]
            L_seg = common.mean_seg_len(center)           # [E]
            step_scale = (tau * L_seg / gmax)             # [E]
            step = step_scale[:, None, None] * g

            # only move active tracks
            move = active[:, None, None].to(center.dtype)
            center = center - step * move

            # hard length rescale about barycenter to kill drift (keeps total length ~fixed)
            cur_len = common.perimeter(center).clamp_min(1e-9)
            bc = center.mean(dim=1, keepdim=True)
            scale = (L0_total / cur_len)[:, None, None]
            center = bc + (center - bc) * torch.where(active[:, None, None], scale, torch.ones_like(scale))

        info["step_times"].append(time.time() - t_step)

        if extra_after_conv and it + 1 >= extra_after_conv:
            break

    # tracks never converged
    if extra_after_conv == 0:
        th = common.thickness(center, band)
        still = (iters_to_conv < 0)
        iters_to_conv[still & (th >= target)] = max_iters

    info["iters"] = max_iters
    info["iters_to_conv"] = iters_to_conv.clone()
    info["n_converged"] = int((iters_to_conv >= 0).sum())
    info["alpha"], info["beta"], info["s"], info["tau"], info["eps"], info["eps_reg"] = \
        alpha, beta, s, tau, eps, eps_reg
    return center, info


# ---------------------------------------------------------------------------
# Extra reporting metrics
# ---------------------------------------------------------------------------

def _half_clearance(center, band):
    """Per-point achievable half-width = min(local curvature radius proxy, half nearest
    non-adjacent distance). Returns [E,N]. Used for clearance-evenness."""
    E, N, _ = center.shape
    kappa = common.menger_curvature(center)
    crad = 1.0 / kappa.clamp_min(1e-12)                   # [E,N] local curvature radius
    dmat = torch.cdist(center, center)
    circ = common.circ_index_dist(N, center.device)
    mask = circ[None] <= band.view(E, 1, 1)
    dmat = dmat.masked_fill(mask, float("inf"))
    nn = dmat.amin(dim=-1)                                # [E,N] nearest non-adjacent dist
    return torch.minimum(crad, 0.5 * nn)


def clearance_evenness(center, band):
    """Per-track std/mean and max/min ratio of half-clearance. Returns dict of [E] tensors."""
    hc = _half_clearance(center, band)
    mean = hc.mean(dim=1)
    std = hc.std(dim=1)
    cv = std / mean.clamp_min(1e-12)                      # coefficient of variation
    ratio = hc.amax(dim=1) / hc.amin(dim=1).clamp_min(1e-12)
    return {"cv": cv, "ratio": ratio, "hc": hc}


def curvature_stats(center):
    """Max Menger curvature per track and its along-curve variation (std). [E]."""
    k = common.menger_curvature(center)
    return {"kmax": k.amax(dim=1), "kstd": k.std(dim=1), "kmean": k.mean(dim=1)}


def _fmt_dist(name, t):
    t = t.float()
    return (f"{name}: med={t.median().item():.4g} mean={t.mean().item():.4g} "
            f"min={t.min().item():.4g} max={t.max().item():.4g}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    c0 = common.load_tracks()
    hw = 0.03
    band = common.band_per_track(c0, hw)

    # ============== Run A: pure tangent-point Sobolev ==============
    # Locked config (a few tuning rounds): no anchor (anchoring pins the very corners
    # that must be rounded -> destroys validity), stop_margin=1.0 (overshooting the
    # target only balloons area further), tau=0.7 = best valid_frac/over-rounding balance.
    print("\n##### RUN A: pure Tangent-point + Sobolev (from Bezier init) #####")
    t0 = time.time()
    relaxedA, infoA = relax(c0, hw, band, max_iters=100, tau=0.7, w_anchor=0.0, stop_margin=1.0)
    secA = time.time() - t0
    itc = infoA["iters_to_conv"].float()
    iters_med = int(itc[itc >= 0].median().item()) if (itc >= 0).any() else -1
    iters_max = int(itc.max().item())
    scA = common.evaluate("Tangent-point + Sobolev", c0, relaxedA, hw, secA, iters_max)
    common.print_scorecard(scA)
    print(f"iters-to-converge: median={iters_med} max={iters_max} n_converged={infoA['n_converged']}/64")
    print(f"per-Sobolev-step time: mean={1000*sum(infoA['step_times'])/max(1,len(infoA['step_times'])):.1f}ms"
          f"  dense-solve(eigh) mean={1000*sum(infoA['solve_times'])/max(1,len(infoA['solve_times'])):.1f}ms"
          f"  solve-fraction={sum(infoA['solve_times'])/max(1e-9,sum(infoA['step_times'])):.0%}")
    common.plot_before_after(c0, relaxedA, hw, "/tmp/tg_run/bakeoff/after_tpsobolev.png")

    # ============== Run B: hybrid PBD -> TP-Sobolev smoothing ==============
    print("\n##### RUN B: Hybrid PBD -> TP-Sobolev #####")
    import relax_xpbd
    hp_pbd = dict(iters=150, sep_relax=1.0, spc_relax=1.0, bend_relax=1.5, margin=0.15, resample=True)
    t0 = time.time()
    feasible, _ = relax_xpbd.relax(c0, hw, band, **hp_pbd)
    sec_pbd = time.time() - t0
    # Gentle smoothing: tau=0.2, 8 steps = clearest clearance/curvature-evenness gain
    # while keeping area growth < 1.1x and preserving PBD's 64/64 feasibility.
    n_smooth = int(8)
    t0 = time.time()
    relaxedB, infoB = relax(feasible, hw, band, extra_after_conv=n_smooth, tau=0.2, w_anchor=0.0)
    sec_smooth = time.time() - t0
    scB = common.evaluate("Hybrid PBD->TP-Sobolev", c0, relaxedB, hw, sec_pbd + sec_smooth, n_smooth)
    common.print_scorecard(scB)
    print(f"time split: PBD={sec_pbd:.3f}s  TP-Sobolev smoothing({n_smooth} iters)={sec_smooth:.3f}s")
    common.plot_before_after(c0, relaxedB, hw, "/tmp/tg_run/bakeoff/after_hybrid.png")

    # ============== Extra metrics + XPBD reference ==============
    print("\n##### EXTRA METRICS (vs pure XPBD reference) #####")
    t0 = time.time()
    xpbd_out, _ = relax_xpbd.relax(c0, hw, band, **hp_pbd)
    sec_xpbd = time.time() - t0
    scX = common.evaluate("XPBD (reference)", c0, xpbd_out, hw, sec_xpbd, 150)
    print(f"[XPBD ref] valid={scX['n_valid']}/64 time={sec_xpbd:.2f}s mean_disp_med={scX['mean_displacement_med']:.4f}")

    for label, c in [("XPBD", xpbd_out), ("TP-Sobolev(A)", relaxedA), ("Hybrid(B)", relaxedB)]:
        ce = clearance_evenness(c, band)
        cs = curvature_stats(c)
        print(f"\n-- {label} --")
        print("  " + _fmt_dist("clearance CV (std/mean, lower=more even)", ce["cv"]))
        print("  " + _fmt_dist("clearance max/min ratio", ce["ratio"]))
        print("  " + _fmt_dist("max Menger curvature", cs["kmax"]))
        print("  " + _fmt_dist("curvature std along curve", cs["kstd"]))

    print("\nFigures: /tmp/tg_run/bakeoff/after_tpsobolev.png , after_hybrid.png")


if __name__ == "__main__":
    main()
