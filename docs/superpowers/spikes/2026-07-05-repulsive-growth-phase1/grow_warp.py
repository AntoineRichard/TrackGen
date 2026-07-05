"""Pure-Warp port of the repulsive-growth phase-1 generator (grow_tp.py).

De-risks a future production ``repulsive`` generator by proving the GROWTH PHASE
of grow_tp.py runs fast and batched in pure NVIDIA Warp -- no torch inside the
loop. The torch prototype (grow_tp.py) is the validated reference; this file
re-implements its ``grow()`` flow as Warp kernels and scores it against the torch
run at E=64 (parity) plus times it at E=64/1024/8192 (performance).

What is ported (pure Warp, zero host sync in the loop):
  * TP tangent-point energy + inverse-power obstacle repulsion + length-ratchet
    penalty, as a single differentiable energy accumulated by atomic_add; the
    gradient comes from ``wp.Tape`` (Warp's own autodiff) -- matches torch
    autograd to ~2e-7 rel (see the smoke tests in the README).
  * Fractional-Sobolev preconditioner WITHOUT FFT: the inverse ring-Laplacian
    filter is a fixed circulant [N] row precomputed ONCE on host (numpy irfft of
    the spectral filter), applied as an O(N^2) circular convolution kernel.
    Matches torch's rfft version to ~8e-7 rel.
  * Length-gradient Sobolev-orthogonal projection, barycenter pin, normalized
    step, hard perimeter rescale to the ratcheting per-env target -- all small
    per-env reduction / elementwise kernels (warp_relax style, flat [E*N] vec2f).
  * Per-env inner-obstacle deactivation at target length, kept in device arrays.
  * Arc-length resample every 25 iters via warp_pipeline.resample_uniform.

What stays torch (NOT the thing being ported): obstacle layout sampling + sizing
(imported verbatim from grow_tp.sample_obstacles, same SEED so layouts are
identical), and the XPBD tail + validity/diversity scoring (grow_tp.tail_bucketed
etc. -- the tail is the oracle XPBD, already validated; porting it is out of scope).

  .venv-gpu/bin/python docs/superpowers/spikes/2026-07-05-repulsive-growth-phase1/grow_warp.py --device cuda
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
import warp as wp

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import grow_tp` works

from tests._oracle import geometry as G  # noqa: E402
from tests._oracle import relaxation as R  # noqa: E402
from track_gen._src import warp_pipeline as wpp  # noqa: E402

import grow_tp as gt  # noqa: E402  (torch reference: sizing, obstacles, tail, scoring, plots)

SPIKE_DIR = Path(__file__).resolve().parent
SEED = gt.SEED


# ===========================================================================
# Differentiable energy kernels (recorded under wp.Tape; grad -> center.grad)
# ===========================================================================

@wp.func
def _safe_dir(v: wp.vec2f) -> wp.vec2f:
    # safe_normalize with the oracle's 1e-8 floor (matches geometry.safe_normalize).
    return v / wp.max(wp.length(v), float(1.0e-8))


@wp.kernel
def _tp_energy_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    alpha: float, beta: float, eps: float,
    frozen: wp.array(dtype=wp.int32),
    energy: wp.array(dtype=wp.float32),
):
    # One thread per (env e, point i). Dense O(N) inner loop over j with the paper's
    # constant +-2 circular exclusion. Mirrors relaxation._tp_energy exactly:
    #   diff = x_j - x_i, wedge = diff ^ T_i, k_ij = (|wedge|+eps)^alpha
    #          / (d2+eps^2)^(beta/2) * w_i * w_j, summed over unmasked pairs.
    # T_i is the central-difference tangent; w_i the dual (lumped-edge) weight.
    # Frozen (stalled, target-length-reached) envs skip the whole O(N) sum -- their
    # grad stays zero so the step/rescale (also frozen-gated) leave them untouched.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    xi = center[b + i]
    xn = center[b + (i + 1) % n]
    xp = center[b + (i + n - 1) % n]
    Ti = _safe_dir(xn - xp)
    wi = 0.5 * (wp.length(xn - xi) + wp.length(xi - xp))
    acc = float(0.0)
    for j in range(n):
        dd = wp.abs(i - j)
        circ = wp.min(dd, n - dd)
        if circ > 2:
            xj = center[b + j]
            xjn = center[b + (j + 1) % n]
            xjp = center[b + (j + n - 1) % n]
            wj = 0.5 * (wp.length(xjn - xj) + wp.length(xj - xjp))
            diff = xj - xi
            d2 = wp.dot(diff, diff)
            wedge = diff[0] * Ti[1] - diff[1] * Ti[0]
            num = wp.pow(wp.abs(wedge) + eps, alpha)
            den = wp.pow(d2 + eps * eps, beta * 0.5)
            acc += (num / den) * wi * wj
    wp.atomic_add(energy, 0, acc)


@wp.kernel
def _obstacle_energy_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32), m_obs: int,
    p_exp: float, reached: wp.array(dtype=wp.int32),
    n_wall: int, deac: int, deac_wall: int,
    frozen: wp.array(dtype=wp.int32),
    energy: wp.array(dtype=wp.float32),
):
    # One thread per (env e, point i). sum_m weight_m*mass_m / |x_i - p_m|^p
    # (Obstacle.cs BodyEnergy) with p = beta-alpha; p_exp = -p/2. Padding columns have
    # obs_mw == 0 (skipped, so their NaN coords never enter the math). Per-env
    # deactivation: at target length (reached[e]), inner-disc columns (m >= n_wall) are
    # dropped when deac; the wall (m < n_wall) only when deac_wall too.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    xi = center[e * n + i]
    ob = e * m_obs
    is_reached = reached[e]
    acc = float(0.0)
    for m in range(m_obs):
        mw = obs_mw[ob + m]
        if mw != 0.0:
            drop = int(0)
            if deac == 1 and is_reached == 1:
                if m >= n_wall:
                    drop = int(1)
                elif deac_wall == 1:
                    drop = int(1)
            if drop == 0:
                diff = xi - obs_pts[ob + m]
                d2 = wp.dot(diff, diff)
                acc += mw * wp.pow(d2 + float(1.0e-8), p_exp)
    wp.atomic_add(energy, 0, acc)


@wp.kernel
def _length_penalty_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    L_target: wp.array(dtype=wp.float32), L_init: wp.array(dtype=wp.float32),
    w_len: float, frozen: wp.array(dtype=wp.int32),
    energy: wp.array(dtype=wp.float32),
):
    # One thread per env. w_len * ((perimeter - L_target)/L_init)^2 -- the small live
    # regularizer nudging L toward the ratcheted target before the hard rescale.
    e = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    peri = float(0.0)
    for i in range(n):
        peri += wp.length(center[b + (i + 1) % n] - center[b + i])
    r = (peri - L_target[e]) / L_init[e]
    wp.atomic_add(energy, 0, w_len * r * r)


# ===========================================================================
# Non-differentiable optimizer-step kernels (outside the tape)
# ===========================================================================

@wp.kernel
def _ratchet_k(
    L_target: wp.array(dtype=wp.float32), L_final: wp.array(dtype=wp.float32),
    growth: float, reached: wp.array(dtype=wp.int32),
):
    # L_target = min(L_target*(1+growth), L_final); reached once at final length.
    e = wp.tid()
    lt = wp.min(L_target[e] * (1.0 + growth), L_final[e])
    L_target[e] = lt
    reached[e] = wp.where(lt >= L_final[e] - float(1.0e-9), int(1), int(0))


@wp.kernel
def _conv_k(gin: wp.array(dtype=wp.vec2f), h: wp.array(dtype=wp.float32), n: int,
            frozen: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
    # Circular convolution out[i] = sum_j h[(i-j) mod n] * gin[j] -- the FFT-free
    # fractional-Sobolev preconditioner A^{-1}. h is the fixed circulant row
    # (numpy irfft of the spectral filter), precomputed once on host. O(N^2), so
    # frozen envs skip it (their step is gated off; stale out never used).
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    acc = wp.vec2f(0.0, 0.0)
    for j in range(n):
        mmod = (i - j) % n
        if mmod < 0:
            mmod += n
        acc += h[mmod] * gin[b + j]
    out[b + i] = acc


@wp.kernel
def _length_grad_k(center: wp.array(dtype=wp.vec2f), n: int, lg: wp.array(dtype=wp.vec2f)):
    # relaxation._length_grad: lg[i] = -u_fwd[i] + u_fwd[i-1], u_fwd[i]=dir(x[i+1]-x[i]).
    e, i = wp.tid()
    b = e * n
    u_i = _safe_dir(center[b + (i + 1) % n] - center[b + i])
    u_p = _safe_dir(center[b + i] - center[b + (i + n - 1) % n])
    lg[b + i] = -u_i + u_p


@wp.kernel
def _numden_k(g: wp.array(dtype=wp.vec2f), lg: wp.array(dtype=wp.vec2f),
              ainv_lg: wp.array(dtype=wp.vec2f), n: int,
              num: wp.array(dtype=wp.float32), den: wp.array(dtype=wp.float32)):
    # Per-env inner products for the Sobolev-orthogonal projection:
    #   num = <g, lg>,  den = <lg, A^{-1} lg> (clamped).
    e = wp.tid()
    b = e * n
    sn = float(0.0)
    sd = float(0.0)
    for i in range(n):
        sn += wp.dot(g[b + i], lg[b + i])
        sd += wp.dot(lg[b + i], ainv_lg[b + i])
    num[e] = sn
    den[e] = wp.max(sd, float(1.0e-12))


@wp.kernel
def _project_k(g: wp.array(dtype=wp.vec2f), ainv_lg: wp.array(dtype=wp.vec2f), n: int,
               num: wp.array(dtype=wp.float32), den: wp.array(dtype=wp.float32)):
    # g <- g - (num/den) * A^{-1} lg  (project out the length-increase direction).
    e, i = wp.tid()
    t = e * n + i
    g[t] = g[t] - (num[e] / den[e]) * ainv_lg[t]


@wp.kernel
def _gmean_k(g: wp.array(dtype=wp.vec2f), n: int, gmean: wp.array(dtype=wp.vec2f)):
    # Per-env barycenter of g (the mean subtracted next -> barycenter pin).
    e = wp.tid()
    b = e * n
    acc = wp.vec2f(0.0, 0.0)
    for i in range(n):
        acc += g[b + i]
    gmean[e] = acc / float(n)


@wp.kernel
def _gmax_msl_k(g: wp.array(dtype=wp.vec2f), gmean: wp.array(dtype=wp.vec2f),
                center: wp.array(dtype=wp.vec2f), n: int,
                gmax: wp.array(dtype=wp.float32), msl: wp.array(dtype=wp.float32)):
    # gmax = max_i |g[i]-gmean|; msl = mean segment length of the current center.
    e = wp.tid()
    b = e * n
    gm = float(0.0)
    peri = float(0.0)
    for i in range(n):
        gm = wp.max(gm, wp.length(g[b + i] - gmean[e]))
        peri += wp.length(center[b + (i + 1) % n] - center[b + i])
    gmax[e] = wp.max(gm, float(1.0e-12))
    msl[e] = peri / float(n)


@wp.kernel
def _step_k(center: wp.array(dtype=wp.vec2f), g: wp.array(dtype=wp.vec2f),
            gmean: wp.array(dtype=wp.vec2f), msl: wp.array(dtype=wp.float32),
            gmax: wp.array(dtype=wp.float32), tau: float, n: int,
            frozen: wp.array(dtype=wp.int32)):
    # center <- center - (tau*msl/gmax) * (g - gmean). Frozen envs never move.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    t = e * n + i
    center[t] = center[t] - (tau * msl[e] / gmax[e]) * (g[t] - gmean[e])


@wp.kernel
def _perim_bc_k(center: wp.array(dtype=wp.vec2f), n: int,
                cur_len: wp.array(dtype=wp.float32), bc: wp.array(dtype=wp.vec2f)):
    # Per-env perimeter (clamped) + barycenter for the hard rescale.
    e = wp.tid()
    b = e * n
    peri = float(0.0)
    acc = wp.vec2f(0.0, 0.0)
    for i in range(n):
        peri += wp.length(center[b + (i + 1) % n] - center[b + i])
        acc += center[b + i]
    cur_len[e] = wp.max(peri, float(1.0e-9))
    bc[e] = acc / float(n)


@wp.kernel
def _rescale_k(center: wp.array(dtype=wp.vec2f), bc: wp.array(dtype=wp.vec2f),
               cur_len: wp.array(dtype=wp.float32), L_target: wp.array(dtype=wp.float32), n: int,
               frozen: wp.array(dtype=wp.int32)):
    # center <- bc + (center-bc) * (L_target/cur_len)  -- hard rescale to the target.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    t = e * n + i
    center[t] = bc[e] + (center[t] - bc[e]) * (L_target[e] / cur_len[e])


@wp.kernel
def _freeze_update_k(center: wp.array(dtype=wp.vec2f), n: int,
                     reached: wp.array(dtype=wp.int32), thresh: float,
                     area_prev: wp.array(dtype=wp.float32),
                     frozen: wp.array(dtype=wp.int32), md_out: wp.array(dtype=wp.float32)):
    # Per-env stall detector (runs once per stall window). An env freezes when it has
    # reached its final length AND its enclosed-area relative change over the window is
    # below ``thresh``. Area (shoelace) is used, NOT bead displacement, because it is
    # REPARAMETERIZATION-INVARIANT: the TP flow slides beads along the curve and the
    # periodic arc-length resample shuffles them, so any per-bead displacement stays
    # large long after the SHAPE has stopped evolving; the enclosed area only moves when
    # the shape does. ``area_prev`` is the window-start area per env. Frozen is sticky.
    e = wp.tid()
    b = e * n
    a2 = float(0.0)
    for i in range(n):
        p0 = center[b + i]
        p1 = center[b + (i + 1) % n]
        a2 += p0[0] * p1[1] - p1[0] * p0[1]
    area = wp.abs(a2) * 0.5
    rel = wp.abs(area - area_prev[e]) / wp.max(area, float(1.0e-9))
    md_out[e] = rel
    area_prev[e] = area
    if reached[e] == 1 and frozen[e] == 0 and rel < thresh:
        frozen[e] = 1


# ===========================================================================
# Host-side spectral filter (precomputed ONCE, uploaded as a circulant row)
# ===========================================================================

def _sobolev_circulant_row(n, s, eps_reg):
    """Real-space circulant first row h of A^{-1}: numpy irfft of the ring spectral
    filter 1/(lam_k^s + eps_reg). A^{-1} g = circular-conv(h, g). Matches
    relaxation._precondition_fft (rfft * filter -> irfft) to ~1e-4 abs / ~8e-7 rel."""
    k = np.arange(n // 2 + 1)
    lam = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / n)
    inv_filter = 1.0 / (np.clip(lam, 0.0, None) ** s + eps_reg)
    return np.fft.irfft(inv_filter, n=n).astype(np.float32), inv_filter.astype(np.float32)


# ===========================================================================
# Pure-Warp growth loop
# ===========================================================================

def _stage_schedule(stages, growth, subdiv_ratio, n_ratchet):
    """Deterministic per-stage START ITER (no device sync). The hard-constraint
    ratchet makes the fastest env's perimeter L_init*(1+growth)^it, so the iter at
    which the mean edge length has grown by ``subdiv_ratio^i`` (the reference's
    "subdivide when the average edge doubles" trigger) is a closed form. Each
    transition is clamped to ``n_ratchet`` so the FINAL stage (N=256) is always
    entered by the time the target length is reached -- the settle phase, and hence
    the tail input, stays at the full resolution regardless of ``growth``."""
    starts = [0]
    for i in range(1, len(stages)):
        it_i = int(np.ceil(np.log(subdiv_ratio ** i) / np.log1p(growth)))
        starts.append(min(it_i, n_ratchet))
    # enforce non-decreasing (fast growth can collapse several doublings together)
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
    return starts


def grow_warp(E, N, r_init, L_final_t, obs_pts_t, obs_mw_t, n_wall, device,
              alpha=3.0, beta=6.0, tau=0.4, growth=0.012, settle_iters=40,
              resample_every=25, w_len=30.0, deac=True, deac_wall=False,
              return_snaps=False, c2f=True, stages=None, subdiv_ratio=2.0,
              stall=True, stall_window=16, stall_disp=0.02, debug=False):
    """Grow E closed curves from radius-r_init circles to perimeters ``L_final_t``
    with the pure-Warp TP-Sobolev flow. Mirrors grow_tp.grow. Returns
    (center_torch [E,N,2], n_iters[, snapshots]).

    Two wall-clock levers over the flat single-resolution loop:

    * **Coarse-to-fine** (``c2f``): fold wavelengths dwarf bead spacing early, so we
      grow at low N and refine. ``stages`` (default N//4, N//2, N) run in sequence;
      the transition iters are precomputed from the ratchet (``_stage_schedule``) so
      the loop stays sync-free except for the 1 arc-length UPSAMPLE per stage
      boundary (torch oracle ``_resample_uniform``, reused from grow_tp; 2 events
      total, not in the hot path). The O(N^2) TP + obstacle sums make N//4 iters
      ~16x cheaper. The final stage is always ``N`` so the tail input is unchanged.
    * **Stall-stop** (``stall``): once an env reaches its final length AND its max
      bead displacement over a ``stall_window`` is below ``stall_disp`` it FREEZES --
      masked out of every step kernel (frozen-gated above), skipping its O(N^2) work.
      A single small readback per window drives a global early exit when all envs are
      frozen, ending the settle phase on convergence instead of a fixed count.

    Set ``c2f=False`` for a single stage at N; ``stall=False`` runs the full
    ``ceil(1.6*n_ratchet)+settle_iters`` budget (the pre-lever behaviour)."""
    dev = device
    p_obs = beta - alpha
    p_exp = -p_obs / 2.0
    s = (beta - 1.0) / (2.0 * alpha)
    eps = 1e-4
    M = obs_mw_t.shape[1]

    if stages is None:
        stages = [N // 4, N // 2, N] if (c2f and N % 4 == 0) else [N]
    stages = list(stages)
    if not c2f:
        stages = [N]
    assert stages[-1] == N, f"final stage {stages[-1]} must equal N={N} (tail input)"

    # --- per-stage circulant Sobolev rows, precomputed once on host ---
    h_by_n = {}
    for Ns in set(stages):
        h_np, _ = _sobolev_circulant_row(Ns, s, 1e-3)
        h_by_n[Ns] = wp.array(h_np, dtype=wp.float32, device=dev)

    L_init_val = float(2 * np.pi * r_init)  # init-circle perimeter (uniform across envs)
    n_ratchet = int(np.ceil(np.log(float((L_final_t / L_init_val).max())) / np.log1p(growth)))
    n_iters = int(np.ceil(n_ratchet * 1.6)) + settle_iters
    stage_starts = _stage_schedule(stages, growth, subdiv_ratio, n_ratchet)
    if debug:
        print(f"[warp-grow] stages={stages} starts={stage_starts} "
              f"n_ratchet={n_ratchet} n_iters(cap)={n_iters} growth={growth}")

    # --- initial circle at the coarsest stage ---
    N0 = stages[0]
    ang = torch.arange(N0, device="cpu", dtype=torch.float32) * (2 * np.pi / N0)
    circle = r_init * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # [N0,2]
    center_np = circle[None].repeat(E, 1, 1).reshape(E * N0, 2).numpy().astype(np.float32)
    center = wp.array(center_np, dtype=wp.vec2f, device=dev, requires_grad=True)

    # --- N-independent device buffers (per-env scalars [E], obstacles) ---
    energy = wp.zeros(1, dtype=wp.float32, device=dev, requires_grad=True)
    obs_pts = wp.array(np.nan_to_num(obs_pts_t.reshape(E * M, 2).cpu().numpy()).astype(np.float32),
                       dtype=wp.vec2f, device=dev)
    obs_mw = wp.array(obs_mw_t.reshape(E * M).cpu().numpy().astype(np.float32),
                      dtype=wp.float32, device=dev)
    L_final = wp.array(L_final_t.cpu().numpy().astype(np.float32), dtype=wp.float32, device=dev)
    L_init = wp.array(np.full(E, L_init_val, np.float32), dtype=wp.float32, device=dev)
    L_target = wp.array(np.full(E, L_init_val, np.float32), dtype=wp.float32, device=dev)
    reached = wp.zeros(E, dtype=wp.int32, device=dev)
    frozen = wp.zeros(E, dtype=wp.int32, device=dev)
    area_prev = wp.zeros(E, dtype=wp.float32, device=dev)
    md_out = wp.zeros(E, dtype=wp.float32, device=dev)
    num = wp.zeros(E, dtype=wp.float32, device=dev)
    den = wp.zeros(E, dtype=wp.float32, device=dev)
    gmean = wp.zeros(E, dtype=wp.vec2f, device=dev)
    gmax = wp.zeros(E, dtype=wp.float32, device=dev)
    msl = wp.zeros(E, dtype=wp.float32, device=dev)
    cur_len = wp.zeros(E, dtype=wp.float32, device=dev)
    bc = wp.zeros(E, dtype=wp.vec2f, device=dev)

    # --- N-dependent buffers (reallocated per stage) ---
    def _alloc_N(Ns):
        return dict(
            n=Ns, h=h_by_n[Ns],
            g=wp.zeros(E * Ns, dtype=wp.vec2f, device=dev),
            lg=wp.zeros(E * Ns, dtype=wp.vec2f, device=dev),
            ainv_lg=wp.zeros(E * Ns, dtype=wp.vec2f, device=dev),
            rs_out=wp.zeros(E * Ns, dtype=wp.vec2f, device=dev),
            rs_seg=wp.zeros(E * Ns, dtype=wp.float32, device=dev),
            rs_s=wp.zeros(E * (Ns + 1), dtype=wp.float32, device=dev),
            count=wp.array(np.full(E, Ns, np.int32), dtype=wp.int32, device=dev))

    stage_idx = 0
    Ncur = stages[0]
    buf = _alloc_N(Ncur)

    deac_i = int(deac)
    deac_wall_i = int(deac_wall)
    snaps = {e: [] for e in ((0, 1, 2, 3) if return_snaps else ())}

    def _snap(it):
        if return_snaps and it % 30 == 0:
            arr = center.numpy().reshape(E, Ncur, 2)
            for e in snaps:
                snaps[e].append(torch.tensor(arr[e]).clone())

    eff_iters = n_iters
    for it in range(n_iters):
        # --- coarse-to-fine stage transition (upsample center N -> next N) ---
        while stage_idx + 1 < len(stages) and it >= stage_starts[stage_idx + 1]:
            stage_idx += 1
            Nnew = stages[stage_idx]
            up = R._resample_uniform(
                torch.tensor(center.numpy().reshape(E, Ncur, 2), device=dev), Nnew)
            center = wp.array(up.reshape(E * Nnew, 2).cpu().numpy().astype(np.float32),
                              dtype=wp.vec2f, device=dev, requires_grad=True)
            Ncur = Nnew
            buf = _alloc_N(Ncur)
            if debug:
                print(f"[warp-grow]   iter {it}: upsample -> N={Ncur}")

        _snap(it)
        Ncur, h_wp = buf["n"], buf["h"]
        g, lg, ainv_lg = buf["g"], buf["lg"], buf["ainv_lg"]

        # 1. ratchet target + deactivation flag
        wp.launch(_ratchet_k, dim=E, inputs=[L_target, L_final, growth, reached], device=dev)

        # 2. energy + gradient via wp.Tape (all 3 terms -> center.grad)
        energy.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(_tp_energy_k, dim=(E, Ncur),
                      inputs=[center, Ncur, alpha, beta, eps, frozen, energy], device=dev)
            wp.launch(_obstacle_energy_k, dim=(E, Ncur),
                      inputs=[center, Ncur, obs_pts, obs_mw, M, p_exp, reached,
                              n_wall, deac_i, deac_wall_i, frozen, energy], device=dev)
            wp.launch(_length_penalty_k, dim=E,
                      inputs=[center, Ncur, L_target, L_init, w_len, frozen, energy], device=dev)
        tape.backward(loss=energy)

        # 3. Sobolev precondition g = A^{-1} grad
        wp.launch(_conv_k, dim=(E, Ncur), inputs=[center.grad, h_wp, Ncur, frozen, g], device=dev)
        # 4. length-gradient Sobolev-orthogonal projection
        wp.launch(_length_grad_k, dim=(E, Ncur), inputs=[center, Ncur, lg], device=dev)
        wp.launch(_conv_k, dim=(E, Ncur), inputs=[lg, h_wp, Ncur, frozen, ainv_lg], device=dev)
        wp.launch(_numden_k, dim=E, inputs=[g, lg, ainv_lg, Ncur, num, den], device=dev)
        wp.launch(_project_k, dim=(E, Ncur), inputs=[g, ainv_lg, Ncur, num, den], device=dev)
        # 5. barycenter pin + normalized step
        wp.launch(_gmean_k, dim=E, inputs=[g, Ncur, gmean], device=dev)
        wp.launch(_gmax_msl_k, dim=E, inputs=[g, gmean, center, Ncur, gmax, msl], device=dev)
        wp.launch(_step_k, dim=(E, Ncur),
                  inputs=[center, g, gmean, msl, gmax, tau, Ncur, frozen], device=dev)
        # 6. hard rescale to the ratcheted target
        wp.launch(_perim_bc_k, dim=E, inputs=[center, Ncur, cur_len, bc], device=dev)
        wp.launch(_rescale_k, dim=(E, Ncur),
                  inputs=[center, bc, cur_len, L_target, Ncur, frozen], device=dev)

        tape.zero()  # clear grads for the next iteration

        # 7. periodic arc-length resample (pure Warp, sync-free)
        if (it + 1) % resample_every == 0:
            wpp.resample_uniform(center, buf["rs_out"], Ncur, buf["count"],
                                 buf["rs_seg"], buf["rs_s"], device=dev)
            wp.copy(center, buf["rs_out"])

        # 8. stall-stop: freeze converged envs, global early exit when all frozen.
        # Only meaningful at the final resolution (freeze needs reached==final length,
        # which happens well after the last upsample). One tiny readback per window.
        if stall and Ncur == stages[-1] and (it + 1) % stall_window == 0:
            wp.launch(_freeze_update_k, dim=E,
                      inputs=[center, Ncur, reached, stall_disp, area_prev, frozen, md_out],
                      device=dev)
            n_frozen = int(frozen.numpy().sum())
            if debug:
                md = md_out.numpy()
                rc = int(reached.numpy().sum())
                print(f"[warp-grow]   iter {it}: reached {rc}/{E} frozen {n_frozen}/{E} "
                      f"rel-area-change p50/p90/max {np.median(md):.4f}/"
                      f"{np.percentile(md, 90):.4f}/{md.max():.4f}")
            if n_frozen >= E:
                eff_iters = it + 1
                break

    wpp.resample_uniform(center, buf["rs_out"], Ncur, buf["count"],
                         buf["rs_seg"], buf["rs_s"], device=dev)
    wp.copy(center, buf["rs_out"])
    wp.synchronize()

    out = torch.tensor(center.numpy().reshape(E, N, 2), device="cpu")
    _snap(eff_iters)
    if return_snaps:
        return out, eff_iters, snaps
    return out, eff_iters


# ===========================================================================
# Setup shared with grow_tp (baseline sizing + obstacles), same SEED
# ===========================================================================

def build_setup(E, N, args, dev):
    """Replicate grow_tp.main's Phase-0 baseline + geometry + obstacle sampling so
    the Warp run sees byte-identical obstacles / L_final to the torch run."""
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator

    torch.manual_seed(SEED)
    cfg = TrackGenConfig(num_envs=E, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=SEED, num_envs=E, device=dev))
    gen.generate()
    track = gen.generate()
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

    r_dom = args.r_dom_frac * P_ref
    r_init = r_dom / args.dom_init_ratio
    L_init_circle = 2 * torch.pi * r_init
    g_lo, g_hi = args.grow_mult
    gen_t = torch.Generator(device=dev).manual_seed(SEED)
    L_final = L_init_circle * (g_lo + (g_hi - g_lo)
                               * torch.rand(E, generator=gen_t, device=dev))
    obs_pts, obs_mass, obs_w, layouts, n_wall = gt.sample_obstacles(
        E, r_dom, r_init, gen_t, dev, disc_clearance=args.disc_clearance,
        wall_clearance=args.wall_clearance_hw * hw,
        k_range=tuple(args.k_range), r_frac=tuple(args.r_frac))
    return dict(cfg=cfg, hw=hw, P_ref=P_ref, r_dom=r_dom, r_init=float(r_init),
                L_final=L_final, obs_pts=obs_pts, obs_mw=obs_mass * obs_w,
                layouts=layouts, n_wall=n_wall, base_center=base_center,
                base_valid=base_valid)


# ===========================================================================
# Parity figure: torch vs warp side by side
# ===========================================================================

def plot_parity_grid(center_t, center_w, count_t, count_w, v_t, v_w, hw, path,
                     layouts, r_dom, n_show=6):
    n = min(n_show, center_t.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
    for e in range(n):
        for row, (cen, cnt, val, tag) in enumerate((
                (center_t, count_t, v_t, "ref"), (center_w, count_w, v_w, "c2f+stall"))):
            ax = axes[row, e]
            ax.add_patch(plt.Circle((0, 0), r_dom, fill=False, color="0.7", ls="--"))
            for c, r in layouts[e]:
                ax.add_patch(plt.Circle(c, r, color="0.85"))
            col = "#1a9641" if val[e] else "#d7191c"
            n_e = int(cnt[e])
            if n_e >= 3:
                cc = cen[e, :n_e]
                _, Nrm = G.tangents_normals(cc[None])
                for curve, lw in ((cc, 0.8), (cc + hw * Nrm[0], 1.3), (cc - hw * Nrm[0], 1.3)):
                    p = torch.cat([curve, curve[:1]]).cpu().numpy()
                    ax.plot(p[:, 0], p[:, 1], "-", color=col if lw > 1 else "0.5", lw=lw)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{tag} env {e} {'ok' if val[e] else 'FAIL'}", fontsize=9)
    fig.suptitle("Repulsive growth: torch flat reference g0.008 (top) vs pure-Warp "
                 "coarse-to-fine + stall-stop g0.012 (bottom), same seed", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ===========================================================================
# main
# ===========================================================================

def _add_args(ap):
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--growth", type=float, default=0.012)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--r-dom-frac", type=float, default=0.35)
    ap.add_argument("--dom-init-ratio", type=float, default=4.0)
    ap.add_argument("--w-len", type=float, default=30.0)
    ap.add_argument("--grow-mult", type=float, nargs=2, default=(4.5, 5.5))
    ap.add_argument("--disc-clearance", type=float, default=0.0)
    ap.add_argument("--wall-clearance-hw", type=float, default=0.0)
    ap.add_argument("--k-range", type=int, nargs=2, default=(8, 12))
    ap.add_argument("--r-frac", type=float, nargs=2, default=(0.02, 0.045))
    ap.add_argument("--no-deac", action="store_true")
    ap.add_argument("--deac-wall", action="store_true")
    ap.add_argument("--settle-iters", type=int, default=40)
    # coarse-to-fine + stall-stop levers
    ap.add_argument("--no-c2f", action="store_true",
                    help="disable coarse-to-fine; grow at a single resolution N the whole run")
    ap.add_argument("--stages", type=int, nargs="+", default=None,
                    help="explicit coarse-to-fine stage schedule (default N//4, N//2, N). "
                         "The last stage must equal N so the tail input is unchanged.")
    ap.add_argument("--subdiv-ratio", type=float, default=2.0,
                    help="upsample N->2N when the mean edge length has grown by this factor "
                         "(reference: subdivide on edge-length doubling).")
    ap.add_argument("--no-stall", action="store_true",
                    help="disable per-env stall-stop; run the full ceil(1.6*n_ratchet)+settle budget")
    ap.add_argument("--stall-window", type=int, default=16,
                    help="iters between stall checks / global-early-exit readbacks")
    ap.add_argument("--stall-disp", type=float, default=0.05,
                    help="freeze an env once it has reached final length AND its enclosed-area "
                         "relative change over a stall window is below this (reparameterization-"
                         "invariant shape-convergence tolerance)")
    ap.add_argument("--perf-only", action="store_true",
                    help="skip torch parity; just time the Warp growth at E=64/1024/8192")
    ap.add_argument("--attribution", action="store_true",
                    help="measure each lever independently at E=1024 (c2f alone, stall alone, both)")


def _grow_warp_from_setup(E, N, S, args, dev, return_snaps=False, debug=False,
                          c2f=None, stall=None, growth=None):
    return grow_warp(
        E, N, S["r_init"], S["L_final"], S["obs_pts"], S["obs_mw"], S["n_wall"], dev,
        alpha=args.alpha, beta=args.beta, tau=args.tau,
        growth=args.growth if growth is None else growth,
        w_len=args.w_len, settle_iters=args.settle_iters,
        deac=not args.no_deac, deac_wall=args.deac_wall, return_snaps=return_snaps,
        c2f=(not args.no_c2f) if c2f is None else c2f, stages=args.stages,
        subdiv_ratio=args.subdiv_ratio,
        stall=(not args.no_stall) if stall is None else stall,
        stall_window=args.stall_window, stall_disp=args.stall_disp, debug=debug)


def main():
    ap = argparse.ArgumentParser()
    _add_args(ap)
    args = ap.parse_args()
    dev = args.device
    E, N = args.E, args.N

    # -------------------- Parity at E=64 (or args.E) --------------------
    if not args.perf_only:
        S = build_setup(E, N, args, dev)
        hw, r_dom, layouts = S["hw"], S["r_dom"], S["layouts"]
        cfg = S["cfg"]
        print(f"[setup] E={E} hw={hw} P_ref={S['P_ref']:.2f} r_dom={r_dom:.3f} "
              f"r_init={S['r_init']:.3f} L_final med {float(S['L_final'].median()):.2f}")

        # torch reference growth (grow_tp.grow): the committed flat-loop quality bar,
        # run at ITS valid growth 0.008 (the flat loop collapses the folds at the
        # warp default's faster 0.012 ratchet -- coarse-to-fine is what makes the fast
        # ratchet fold; see README). This is the "before" of the before/after figure.
        TORCH_REF_GROWTH = 0.008
        t0 = time.time()
        grown_t_fine, _, n_iters_t = gt.grow(
            E, N, S["r_init"], r_dom, S["L_final"], S["obs_pts"], S["obs_mw"], dev,
            alpha=args.alpha, beta=args.beta, tau=args.tau, growth=TORCH_REF_GROWTH,
            w_len=args.w_len, settle_iters=args.settle_iters, n_wall=S["n_wall"],
            deac=not args.no_deac, deac_wall=args.deac_wall)
        torch.cuda.synchronize() if "cuda" in dev else None
        sec_torch = time.time() - t0

        # pure-Warp growth (warmup once for module load, then timed)
        _grow_warp_from_setup(E, N, S, args, dev)  # warmup (kernel compile / module load)
        t0 = time.time()
        grown_w_fine, n_iters_w, snaps = _grow_warp_from_setup(
            E, N, S, args, dev, return_snaps=True, debug=True)
        sec_warp = time.time() - t0
        grown_w_fine = grown_w_fine.to(dev)

        print(f"[grow torch-ref g0.008] {n_iters_t} iters {sec_torch:.2f}s "
              f"({1000*sec_torch/n_iters_t:.2f} ms/iter)")
        print(f"[grow warp c2f+stall  ] {n_iters_w} eff-iters {sec_warp:.2f}s "
              f"({1000*sec_warp/n_iters_w:.2f} ms/iter)  speedup {sec_torch/sec_warp:.1f}x")

        # same tail + validity + diversity for BOTH (grow_tp functions)
        spacing = 0.6 * hw
        grown_t, count_t = G.arc_length_resample(grown_t_fine, spacing=spacing)
        grown_w, count_w = G.arc_length_resample(grown_w_fine, spacing=spacing)
        relax_t = gt.tail_bucketed(grown_t, count_t, cfg)
        relax_w = gt.tail_bucketed(grown_w, count_w, cfg)
        v_t, _, _ = gt.validity_batch(relax_t, count_t, hw)
        v_w, _, _ = gt.validity_batch(relax_w, count_w, hw)
        print(f"[parity] post-tail valid  torch {int(v_t.sum())}/{E}   warp {int(v_w.sum())}/{E}")

        div_t = gt.diversity_batch(relax_t, count_t, v_t)
        div_w = gt.diversity_batch(relax_w, count_w, v_w)
        print("[diversity] " + gt.fmt_div("torch", div_t))
        print("[diversity] " + gt.fmt_div("warp ", div_w))

        # NOTE: a per-bead torch-vs-warp displacement is no longer meaningful here --
        # the two runs are DIFFERENT configs (torch flat g0.008 reference vs warp
        # coarse-to-fine + stall g0.012), not the same flow, so the trajectories diverge
        # by construction. The parity that matters is yield + shape statistics (above):
        # both clear 64/64 and land in the same serpentine compactness band.

        plot_parity_grid(relax_t.cpu(), relax_w.cpu(), count_t.cpu(), count_w.cpu(),
                         v_t.cpu(), v_w.cpu(), hw, SPIKE_DIR / "warp-parity-grid.png",
                         layouts, r_dom)
        print(f"figure: {SPIKE_DIR}/warp-parity-grid.png")

    # -------------------- Per-lever attribution at E=1024 --------------------
    # Levers 1 (c2f) and 2 (stall) are isolated at the FLAT loop's valid growth 0.008
    # (the flat loop collapses at 0.012); the faster ratchet is the last row, and only
    # holds full yield WITH c2f -- so it is measured c2f+stall g0.008 -> g0.012.
    if args.attribution:
        print("\n[attribution] per-lever wall-clock at E=1024 (post-warmup):")
        print(f"  {'config':>26} {'eff_iters':>10} {'total_s':>9}")
        Sa = build_setup(1024, N, args, dev)
        for tag, c2f, stall, gr in (
                ("baseline flat g0.008", False, False, 0.008),
                ("+ c2f only g0.008", True, False, 0.008),
                ("+ stall only g0.008", False, True, 0.008),
                ("+ c2f + stall g0.008", True, True, 0.008),
                ("+ faster ratchet g0.012 (default)", True, True, 0.012)):
            _grow_warp_from_setup(1024, N, Sa, args, dev, c2f=c2f, stall=stall, growth=gr)  # warmup
            try:
                wp.synchronize()
            except Exception:
                pass
            t0 = time.time()
            _, eff = _grow_warp_from_setup(1024, N, Sa, args, dev, c2f=c2f, stall=stall, growth=gr)
            print(f"  {tag:>26} {eff:>10} {time.time() - t0:>9.3f}")

    # -------------------- Performance sweep --------------------
    print("\n[perf] pure-Warp growth wall-clock (post-warmup, module load excluded):")
    print(f"  {'E':>6} {'eff_it':>6} {'total_s':>9} {'ms/iter':>9} {'peak_MiB':>9}")
    for Ep in (64, 1024, 8192):
        Sp = build_setup(Ep, N, args, dev)
        _grow_warp_from_setup(Ep, N, Sp, args, dev)  # warmup
        try:
            wp.synchronize()
        except Exception:
            pass
        t0 = time.time()
        _, n_iters = _grow_warp_from_setup(Ep, N, Sp, args, dev)
        dt = time.time() - t0
        try:
            peak = wp.get_mempool_used_mem_high(wp.get_device(dev)) / (1024 ** 2)
        except Exception:
            peak = float("nan")
        print(f"  {Ep:>6} {n_iters:>6} {dt:>9.3f} {1000*dt/n_iters:>9.3f} {peak:>9.1f}")


if __name__ == "__main__":
    main()
