"""Near/far-field split (RESPA / multiple-timestepping) for the repulsive TP gradient.

SPIKE -- does not modify ``track_gen/_src/``. Reuses the production kernels
(``warp_generate_repulsive``) verbatim where possible and adds the near/far machinery:

  * NEAR field -- TP pairs whose partner is inside a Euclidean cutoff radius (circ>2),
    from a fixed-slot per-vertex candidate list rebuilt every ``K`` iters with a
    staleness margin. Recomputed EXACTLY every iteration against the (possibly stale)
    cached partner set -- same safety argument as ``warp_relax``'s separation cache.
  * FAR field -- everything else. Recomputed exactly on refresh iters (``g_far =
    g_full_tp - g_near_tp``) and held FROZEN in between (the far field is smooth,
    ~1%/iter). No truncation: the global packing pressure is preserved, temporally
    coarsened.

The split is a strict partition BY CONSTRUCTION: g_far is the exact full TP gradient
minus the exact near TP gradient at refresh time, so no pair is double-counted or
dropped regardless of the cutoff/margin. The cutoff/margin only govern ACCURACY between
refreshes (which pairs get the fresh short-range treatment).

K=1 / coarse stages fall back to the EXACT production combined gather
(``rep._grad_gather_k``) -- so K=1 reduces byte-for-byte to current behavior and is the
ground truth for gating.

The obstacle + length terms are cheap-ish (O(N*M), O(N)) and correctness-critical for
domain confinement, so they are kept EXACT every iteration (not split).
"""
from __future__ import annotations

import numpy as np
import warp as wp

import time as _time

from track_gen._src import warp_generate_repulsive as rep
from track_gen._src import warp_pipeline as _pipe

# Diagnostics: when True, generate() records synchronized wall-clock at each stage boundary
# into _STAGE_LOG (list of (label, seconds)). Perturbs timing (adds syncs) -- profiling only.
PROFILE = False
_STAGE_LOG: list = []


# ===========================================================================
# Candidate list (broadphase). Fixed-slot per-vertex, ascending-j order ->
# deterministic. Rebuilt every K iters at the final stage only.
# ===========================================================================

@wp.kernel
def _cand_build_k(
    center: wp.array(dtype=wp.vec2f), n: int, maxnbr: int,
    cutoff_beads_eff: float, peri_e: wp.array(dtype=wp.float32),
    frozen: wp.array(dtype=wp.int32),
    nbr: wp.array(dtype=wp.int32), ncount: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    # One thread per vertex v. Scan partners j in ascending order; a partner qualifies as
    # NEAR when circ>2 (same ring exclusion as the TP energy) AND |x_j-x_v|^2 < cutoff2.
    # The cutoff radius is cutoff_beads_eff * (per-env mean segment length = peri_e[e]/n),
    # so the horizon tracks the current spacing WITHOUT a host readback (byte-deterministic,
    # sync-free). Fixed-slot: first maxnbr qualifiers are stored (ascending j -> deterministic
    # ordering, no atomics), the rest raise the per-env overflow counter. Unused slots are -1.
    e, v = wp.tid()
    base = e * n
    slot_base = (base + v) * maxnbr
    msl = peri_e[e] / float(n)
    cutoff = cutoff_beads_eff * msl
    cutoff2 = cutoff * cutoff
    if frozen[e] == 1:
        for s in range(maxnbr):
            nbr[slot_base + s] = int(-1)
        ncount[base + v] = int(0)
        return
    xv = center[base + v]
    cnt = int(0)
    ovf = int(0)
    for j in range(n):
        dd = wp.abs(v - j)
        circ = wp.min(dd, n - dd)
        if circ > 2:
            diff = center[base + j] - xv
            if wp.dot(diff, diff) < cutoff2:
                if cnt < maxnbr:
                    nbr[slot_base + cnt] = j
                    cnt += 1
                else:
                    ovf += 1
    for s in range(cnt, maxnbr):
        nbr[slot_base + s] = int(-1)
    ncount[base + v] = cnt
    if ovf > 0:
        wp.atomic_add(overflow, e, ovf)


# ===========================================================================
# NEAR prepass + gather -- exact TP mechanisms restricted to the cached partner
# set. Byte-identical math to rep._tp_prepass_k / the TP part of rep._grad_gather_k,
# only the partner loop is replaced by the candidate list.
# ===========================================================================

@wp.kernel
def _tp_prepass_near_k(
    center: wp.array(dtype=wp.vec2f), n: int, maxnbr: int,
    alpha: float, beta: float, eps: float,
    frozen: wp.array(dtype=wp.int32),
    tang: wp.array(dtype=wp.vec2f), wdual: wp.array(dtype=wp.float32),
    nbr: wp.array(dtype=wp.int32),
    wcoef: wp.array(dtype=wp.float32), btan: wp.array(dtype=wp.vec2f),
):
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    slot_base = (b + v) * maxnbr
    xv = center[b + v]
    xvn = center[b + (v + 1) % n]
    xvp = center[b + (v + n - 1) % n]
    lv = wp.max(wp.length(xvn - xvp), float(1.0e-8))
    Tv = tang[b + v]
    wv = wdual[b + v]
    hb = beta * 0.5
    C = float(0.0)
    D = float(0.0)
    A = wp.vec2f(0.0, 0.0)
    for s in range(maxnbr):
        j = nbr[slot_base + s]
        if j < 0:
            break
        xj = center[b + j]
        Tj = tang[b + j]
        wj = wdual[b + j]
        diff = xj - xv
        d2 = wp.dot(diff, diff)
        den = wp.pow(d2 + eps * eps, hb)
        wedge = diff[0] * Tv[1] - diff[1] * Tv[0]
        aw = wp.abs(wedge) + eps
        P_vj = wp.pow(aw, alpha) / den
        g_num = alpha * wp.pow(aw, alpha - 1.0) * wp.where(wedge < 0.0, float(-1.0), float(1.0))
        A += (g_num / den) * wp.vec2f(-diff[1], diff[0]) * wj
        C += P_vj * wj
        wedge2 = -diff[0] * Tj[1] + diff[1] * Tj[0]
        P_jv = wp.pow(wp.abs(wedge2) + eps, alpha) / den
        D += P_jv * wj
    wcoef[b + v] = C + D
    Af = wv * A
    btan[b + v] = (Af - Tv * wp.dot(Tv, Af)) / lv


@wp.kernel
def _tp_gather_full_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    alpha: float, beta: float, eps: float,
    tang: wp.array(dtype=wp.vec2f), wdual: wp.array(dtype=wp.float32),
    wcoef: wp.array(dtype=wp.float32), btan: wp.array(dtype=wp.vec2f),
    frozen: wp.array(dtype=wp.int32),
    g_tp: wp.array(dtype=wp.vec2f),
):
    # TP-only part of rep._grad_gather_k (diff + weight stencil + tangent stencil), all pairs.
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    xv = center[b + v]
    xvn = center[b + (v + 1) % n]
    xvp = center[b + (v + n - 1) % n]
    Tv = tang[b + v]
    wv = wdual[b + v]
    hb = beta * 0.5
    g = wp.vec2f(0.0, 0.0)
    for t in range(n):
        dd = wp.abs(v - t)
        circ = wp.min(dd, n - dd)
        if circ > 2:
            xt = center[b + t]
            Tt = tang[b + t]
            wt = wdual[b + t]
            ww = wv * wt
            diff = xt - xv
            d2 = wp.dot(diff, diff)
            de2 = d2 + eps * eps
            den = wp.pow(de2, hb)
            wedge = diff[0] * Tv[1] - diff[1] * Tv[0]
            aw = wp.abs(wedge) + eps
            P = wp.pow(aw, alpha) / den
            gn = alpha * wp.pow(aw, alpha - 1.0) * wp.where(wedge < 0.0, float(-1.0), float(1.0))
            dP_vt = (gn / den) * wp.vec2f(Tv[1], -Tv[0]) - (P * beta / de2) * diff
            diff2 = xv - xt
            wedge2 = diff2[0] * Tt[1] - diff2[1] * Tt[0]
            aw2 = wp.abs(wedge2) + eps
            P2 = wp.pow(aw2, alpha) / den
            gn2 = alpha * wp.pow(aw2, alpha - 1.0) * wp.where(wedge2 < 0.0, float(-1.0), float(1.0))
            dP_tv = (gn2 / den) * wp.vec2f(Tt[1], -Tt[0]) - (P2 * beta / de2) * diff2
            g += ww * (dP_tv - dP_vt)
    ef = xvn - xv
    eb = xv - xvp
    def_dir = rep._safe_dir(ef)
    deb_dir = rep._safe_dir(eb)
    Wv = wcoef[b + v]
    Wprev = wcoef[b + (v + n - 1) % n]
    Wnext = wcoef[b + (v + 1) % n]
    g += 0.5 * deb_dir * (Wv + Wprev) - 0.5 * def_dir * (Wv + Wnext)
    g += btan[b + (v + n - 1) % n] - btan[b + (v + 1) % n]
    g_tp[b + v] = g


@wp.kernel
def _tp_gather_near_k(
    center: wp.array(dtype=wp.vec2f), n: int, maxnbr: int,
    alpha: float, beta: float, eps: float,
    tang: wp.array(dtype=wp.vec2f), wdual: wp.array(dtype=wp.float32),
    wcoef: wp.array(dtype=wp.float32), btan: wp.array(dtype=wp.vec2f),
    nbr: wp.array(dtype=wp.int32),
    frozen: wp.array(dtype=wp.int32),
    g_tp: wp.array(dtype=wp.vec2f),
):
    # TP-only gradient restricted to the cached near partner set. Same math as
    # _tp_gather_full_k with the partner loop over nbr and the stencils reading the
    # NEAR reductions (wcoef/btan came from _tp_prepass_near_k).
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    slot_base = (b + v) * maxnbr
    xv = center[b + v]
    xvn = center[b + (v + 1) % n]
    xvp = center[b + (v + n - 1) % n]
    Tv = tang[b + v]
    wv = wdual[b + v]
    hb = beta * 0.5
    g = wp.vec2f(0.0, 0.0)
    for s in range(maxnbr):
        t = nbr[slot_base + s]
        if t < 0:
            break
        xt = center[b + t]
        Tt = tang[b + t]
        wt = wdual[b + t]
        ww = wv * wt
        diff = xt - xv
        d2 = wp.dot(diff, diff)
        de2 = d2 + eps * eps
        den = wp.pow(de2, hb)
        wedge = diff[0] * Tv[1] - diff[1] * Tv[0]
        aw = wp.abs(wedge) + eps
        P = wp.pow(aw, alpha) / den
        gn = alpha * wp.pow(aw, alpha - 1.0) * wp.where(wedge < 0.0, float(-1.0), float(1.0))
        dP_vt = (gn / den) * wp.vec2f(Tv[1], -Tv[0]) - (P * beta / de2) * diff
        diff2 = xv - xt
        wedge2 = diff2[0] * Tt[1] - diff2[1] * Tt[0]
        aw2 = wp.abs(wedge2) + eps
        P2 = wp.pow(aw2, alpha) / den
        gn2 = alpha * wp.pow(aw2, alpha - 1.0) * wp.where(wedge2 < 0.0, float(-1.0), float(1.0))
        dP_tv = (gn2 / den) * wp.vec2f(Tt[1], -Tt[0]) - (P2 * beta / de2) * diff2
        g += ww * (dP_tv - dP_vt)
    ef = xvn - xv
    eb = xv - xvp
    def_dir = rep._safe_dir(ef)
    deb_dir = rep._safe_dir(eb)
    Wv = wcoef[b + v]
    Wprev = wcoef[b + (v + n - 1) % n]
    Wnext = wcoef[b + (v + 1) % n]
    g += 0.5 * deb_dir * (Wv + Wprev) - 0.5 * def_dir * (Wv + Wnext)
    g += btan[b + (v + n - 1) % n] - btan[b + (v + 1) % n]
    g_tp[b + v] = g


@wp.kernel
def _obs_len_gather_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32), m_obs: int,
    p_exp: float, reached: wp.array(dtype=wp.int32),
    n_wall: int, deac: int,
    len_coef: wp.array(dtype=wp.float32),
    frozen: wp.array(dtype=wp.int32),
    g_ol: wp.array(dtype=wp.vec2f),
):
    # Obstacle self-term + length regularizer part of rep._grad_gather_k (kept exact every iter).
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    xv = center[b + v]
    xvn = center[b + (v + 1) % n]
    xvp = center[b + (v + n - 1) % n]
    g = wp.vec2f(0.0, 0.0)
    ob = e * m_obs
    is_reached = reached[e]
    for m in range(m_obs):
        mw = obs_mw[ob + m]
        if mw != 0.0:
            drop = int(0)
            if deac == 1 and is_reached == 1 and m >= n_wall:
                drop = int(1)
            if drop == 0:
                d = xv - obs_pts[ob + m]
                d2 = wp.dot(d, d)
                g += (mw * p_exp * wp.pow(d2 + float(1.0e-8), p_exp - 1.0) * 2.0) * d
    def_dir = rep._safe_dir(xvn - xv)
    deb_dir = rep._safe_dir(xv - xvp)
    g += len_coef[e] * (deb_dir - def_dir)
    g_ol[b + v] = g


@wp.kernel
def _combine_refresh_k(
    g_full_tp: wp.array(dtype=wp.vec2f), g_near_tp: wp.array(dtype=wp.vec2f),
    g_ol: wp.array(dtype=wp.vec2f), n: int,
    frozen: wp.array(dtype=wp.int32),
    g_far: wp.array(dtype=wp.vec2f), grad: wp.array(dtype=wp.vec2f),
):
    # Refresh iter: re-anchor the frozen far field and emit the EXACT full gradient.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    t = e * n + i
    g_far[t] = g_full_tp[t] - g_near_tp[t]
    grad[t] = g_full_tp[t] + g_ol[t]


@wp.kernel
def _combine_frozen_k(
    g_near_tp: wp.array(dtype=wp.vec2f), g_far: wp.array(dtype=wp.vec2f),
    g_ol: wp.array(dtype=wp.vec2f), n: int,
    frozen: wp.array(dtype=wp.int32),
    grad: wp.array(dtype=wp.vec2f),
):
    # Between refreshes: fresh near + frozen far + fresh obstacle/length.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    t = e * n + i
    grad[t] = g_near_tp[t] + g_far[t] + g_ol[t]


# ===========================================================================
# Near/far scratch (side buffers, keyed off the production RepulsiveScratch)
# ===========================================================================

class _NFBuffers:
    __slots__ = ("maxnbr", "E", "Nmax", "nbr", "ncount", "overflow", "wcoef_n", "btan_n",
                 "g_full_tp", "g_near_tp", "g_ol", "g_far")


_NF_CACHE: dict[int, _NFBuffers] = {}


def _nf_buffers(scratch, maxnbr, dev):
    # NB: id(scratch) can be REUSED after a scratch is GC'd, so we must revalidate E/Nmax
    # (not just maxnbr) or a new scratch could pick up an undersized buffer -> OOB.
    key = id(scratch)
    E, Nmax = scratch.E, scratch.Nmax
    nf = _NF_CACHE.get(key)
    if nf is not None and nf.maxnbr == maxnbr and nf.E == E and nf.Nmax == Nmax:
        return nf
    nf = _NFBuffers()
    nf.maxnbr = maxnbr
    nf.E = E
    nf.Nmax = Nmax
    nf.nbr = wp.empty(E * Nmax * maxnbr, dtype=wp.int32, device=dev)
    nf.ncount = wp.zeros(E * Nmax, dtype=wp.int32, device=dev)
    nf.overflow = wp.zeros(E, dtype=wp.int32, device=dev)
    nf.wcoef_n = wp.zeros(E * Nmax, dtype=wp.float32, device=dev)
    nf.btan_n = wp.zeros(E * Nmax, dtype=wp.vec2f, device=dev)
    nf.g_full_tp = wp.zeros(E * Nmax, dtype=wp.vec2f, device=dev)
    nf.g_near_tp = wp.zeros(E * Nmax, dtype=wp.vec2f, device=dev)
    nf.g_ol = wp.zeros(E * Nmax, dtype=wp.vec2f, device=dev)
    nf.g_far = wp.zeros(E * Nmax, dtype=wp.vec2f, device=dev)
    _NF_CACHE[key] = nf
    return nf


# ===========================================================================
# Growth loop with the near/far split. A near-clone of
# rep.generate_repulsive_warp; only the gradient assembly (step 2) changes.
# ===========================================================================

def make_generate(K: int, cutoff_beads: float, maxnbr: int,
                  split_final_only: bool = True, margin_taus: float = 2.0):
    """Return a ``generate(seeds, config, out_centerline, out_valid, scratch)`` closure
    that runs the growth loop with an order-K near/far RESPA split of the TP gradient.

    K=1 (or coarse stages when split_final_only) => the exact production combined gather.
    cutoff_beads: near-horizon radius in units of the current mean segment length.
    maxnbr: fixed candidate slots per vertex.
    margin_taus: staleness margin = margin_taus * tau * K bead-spacings added to the
        build cutoff so a pair cannot cross the eval horizon within a K-window.
    """

    def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch):
        _pipe._init()
        s = scratch
        E = s.E
        dev = str(out_centerline.device)

        alpha = float(config.repulsive_alpha)
        beta = float(config.repulsive_beta)
        tau = float(config.repulsive_tau)
        growth = float(config.repulsive_ratchet_rate)
        w_len = float(config.repulsive_w_len)
        resample_every = int(config.repulsive_resample_every)
        stall_window = int(config.repulsive_stall_window)
        stall_tol = float(config.repulsive_stall_area_tol)
        deac_i = int(bool(config.repulsive_deactivate_obstacles))
        eps = 1e-4
        p_exp = -(beta - alpha) / 2.0
        n_wall = s.n_wall
        M = s.M
        stages = s.stages
        stage_starts = s.stage_starts
        n_iters = s.n_iters
        Nfinal = stages[-1]

        center = s.center
        nf = _nf_buffers(s, maxnbr, dev)
        nf.overflow.zero_()

        rep._sample_obstacles_inplace(seeds_wp, config, s.obs_pts, s.obs_mw, dev)
        grow_lo = float(config.repulsive_grow_mult_min)
        grow_hi = float(config.repulsive_grow_mult_max)
        wp.launch(rep._seed_lfinal_k, dim=E,
                  inputs=[seeds_wp, s.r_init, grow_lo, grow_hi,
                          s.L_final, s.L_init, s.L_target, s.reached, s.frozen, s.area_prev],
                  device=dev)

        N0 = stages[0]
        wp.launch(rep._init_circle_k, dim=(E, N0), inputs=[center, N0, s.r_init], device=dev)

        stage_idx = 0
        Ncur = N0

        def _upsample_to_next():
            nonlocal stage_idx, Ncur
            stage_idx += 1
            Nnew = stages[stage_idx]
            _pipe.arc_length_resample_inplace(
                center[0:E * Ncur], Ncur, Nnew,
                s.arc_real[0:E * Ncur], s.rs_seg[0:E * Ncur], s.rs_s[0:E * (Ncur + 1)],
                s.arc_cr, s.arc_co, s.rs_out[0:E * Nnew], dev)
            wp.copy(center, s.rs_out, count=E * Nnew)
            Ncur = Nnew

        # counts the iteration index since entering the final stage (drives the K cadence).
        final_iter = -1
        if PROFILE:
            _pipe._sync(dev); _t_loop0 = _time.perf_counter(); _t_final = None

        for it in range(n_iters):
            while stage_idx + 1 < len(stages) and it >= stage_starts[stage_idx + 1]:
                _upsample_to_next()
                if PROFILE and stage_idx == len(stages) - 1:
                    _pipe._sync(dev); _t_final = _time.perf_counter()

            h_wp = s.h_by_n[Ncur]

            wp.launch(rep._ratchet_k, dim=E, inputs=[s.L_target, s.L_final, growth, s.reached], device=dev)

            # Per-vertex tangents/weights + length coefficient (shared by all paths).
            wp.launch(rep._tangent_weight_k, dim=(E, Ncur),
                      inputs=[center, Ncur, s.frozen, s.tang, s.wdual], device=dev)
            wp.launch(rep._len_coef_k, dim=E,
                      inputs=[center, Ncur, s.L_target, s.L_init, w_len, s.frozen,
                              s.len_coef, s.peri_e], device=dev)

            split_active = (K > 1) and (Ncur == Nfinal) and (
                (not split_final_only) or stage_idx == len(stages) - 1)

            if split_active:
                final_iter += 1
                is_refresh = (final_iter % K == 0)
                if is_refresh:
                    # cutoff = (cutoff_beads + K-window staleness margin) bead-spacings, applied
                    # per-env on-device from peri_e (just computed by _len_coef_k). The margin
                    # 2*tau*K spacings covers the max a pair can approach in a K-window (steps are
                    # capped at ~tau*msl/iter), so no pair crosses the eval horizon between refreshes.
                    cutoff_beads_eff = cutoff_beads + margin_taus * tau * float(K)
                    wp.launch(_cand_build_k, dim=(E, Ncur),
                              inputs=[center, Ncur, maxnbr, cutoff_beads_eff, s.peri_e, s.frozen,
                                      nf.nbr, nf.ncount, nf.overflow], device=dev)
                    # full TP
                    wp.launch(rep._tp_prepass_k, dim=(E, Ncur),
                              inputs=[center, Ncur, alpha, beta, eps, s.frozen,
                                      s.tang, s.wdual, s.wcoef, s.btan], device=dev)
                    wp.launch(_tp_gather_full_k, dim=(E, Ncur),
                              inputs=[center, Ncur, alpha, beta, eps, s.tang, s.wdual,
                                      s.wcoef, s.btan, s.frozen, nf.g_full_tp], device=dev)
                    # near TP
                    wp.launch(_tp_prepass_near_k, dim=(E, Ncur),
                              inputs=[center, Ncur, maxnbr, alpha, beta, eps, s.frozen,
                                      s.tang, s.wdual, nf.nbr, nf.wcoef_n, nf.btan_n], device=dev)
                    wp.launch(_tp_gather_near_k, dim=(E, Ncur),
                              inputs=[center, Ncur, maxnbr, alpha, beta, eps, s.tang, s.wdual,
                                      nf.wcoef_n, nf.btan_n, nf.nbr, s.frozen, nf.g_near_tp], device=dev)
                    wp.launch(_obs_len_gather_k, dim=(E, Ncur),
                              inputs=[center, Ncur, s.obs_pts, s.obs_mw, M, p_exp, s.reached,
                                      n_wall, deac_i, s.len_coef, s.frozen, nf.g_ol], device=dev)
                    wp.launch(_combine_refresh_k, dim=(E, Ncur),
                              inputs=[nf.g_full_tp, nf.g_near_tp, nf.g_ol, Ncur, s.frozen,
                                      nf.g_far, s.grad], device=dev)
                else:
                    wp.launch(_tp_prepass_near_k, dim=(E, Ncur),
                              inputs=[center, Ncur, maxnbr, alpha, beta, eps, s.frozen,
                                      s.tang, s.wdual, nf.nbr, nf.wcoef_n, nf.btan_n], device=dev)
                    wp.launch(_tp_gather_near_k, dim=(E, Ncur),
                              inputs=[center, Ncur, maxnbr, alpha, beta, eps, s.tang, s.wdual,
                                      nf.wcoef_n, nf.btan_n, nf.nbr, s.frozen, nf.g_near_tp], device=dev)
                    wp.launch(_obs_len_gather_k, dim=(E, Ncur),
                              inputs=[center, Ncur, s.obs_pts, s.obs_mw, M, p_exp, s.reached,
                                      n_wall, deac_i, s.len_coef, s.frozen, nf.g_ol], device=dev)
                    wp.launch(_combine_frozen_k, dim=(E, Ncur),
                              inputs=[nf.g_near_tp, nf.g_far, nf.g_ol, Ncur, s.frozen,
                                      s.grad], device=dev)
            else:
                # EXACT production path (K=1 or coarse stage).
                wp.launch(rep._tp_prepass_k, dim=(E, Ncur),
                          inputs=[center, Ncur, alpha, beta, eps, s.frozen,
                                  s.tang, s.wdual, s.wcoef, s.btan], device=dev)
                wp.launch(rep._grad_gather_k, dim=(E, Ncur),
                          inputs=[center, Ncur, alpha, beta, eps, s.tang, s.wdual, s.wcoef, s.btan,
                                  s.obs_pts, s.obs_mw, M, p_exp, s.reached,
                                  n_wall, deac_i, s.len_coef, s.frozen, s.grad], device=dev)

            # --- optimizer tail (identical to production) ---
            wp.launch(rep._conv_k, dim=(E, Ncur), inputs=[s.grad, h_wp, Ncur, s.frozen, s.g], device=dev)
            wp.launch(rep._length_grad_k, dim=(E, Ncur), inputs=[center, Ncur, s.lg], device=dev)
            wp.launch(rep._conv_k, dim=(E, Ncur), inputs=[s.lg, h_wp, Ncur, s.frozen, s.ainv_lg], device=dev)
            wp.launch(rep._numden_k, dim=E, inputs=[s.g, s.lg, s.ainv_lg, Ncur, s.num, s.den], device=dev)
            wp.launch(rep._project_k, dim=(E, Ncur), inputs=[s.g, s.ainv_lg, Ncur, s.num, s.den], device=dev)
            wp.launch(rep._gmean_k, dim=E, inputs=[s.g, Ncur, s.gmean], device=dev)
            wp.launch(rep._gmax_msl_k, dim=E, inputs=[s.g, s.gmean, s.peri_e, Ncur, s.gmax, s.msl], device=dev)
            wp.launch(rep._step_k, dim=(E, Ncur),
                      inputs=[center, s.g, s.gmean, s.msl, s.gmax, tau, Ncur, s.frozen], device=dev)
            wp.launch(rep._perim_bc_k, dim=E, inputs=[center, Ncur, s.cur_len, s.bc], device=dev)
            wp.launch(rep._rescale_k, dim=(E, Ncur),
                      inputs=[center, s.bc, s.cur_len, s.L_target, Ncur, s.frozen], device=dev)

            if (it + 1) % resample_every == 0:
                wp.launch(_pipe._fill_i32_k, dim=E, inputs=[s.count, Ncur], device=dev)
                _pipe.resample_uniform(center, s.rs_out, Ncur, s.count, s.rs_seg, s.rs_s, device=dev)
                wp.copy(center, s.rs_out, count=E * Ncur)
                # A re-parameterization shuffles which bead is where -> the cached partner
                # list is invalidated. Force a refresh on the next split iter.
                final_iter = -1 if split_active else final_iter

            if Ncur == stages[-1] and (it + 1) % stall_window == 0:
                wp.launch(rep._freeze_update_k, dim=E,
                          inputs=[center, Ncur, s.reached, stall_tol, s.area_prev, s.frozen, s.md_out],
                          device=dev)
                if int(s.frozen.numpy().sum()) >= E:
                    break

        if PROFILE:
            _pipe._sync(dev); _t_end = _time.perf_counter()
            coarse = (_t_final - _t_loop0) if _t_final else 0.0
            final = (_t_end - _t_final) if _t_final else 0.0
            _STAGE_LOG.append(("coarse", coarse)); _STAGE_LOG.append(("final+settle", final))

        while stage_idx + 1 < len(stages):
            _upsample_to_next()

        wp.launch(_pipe._fill_i32_k, dim=E, inputs=[s.count, Ncur], device=dev)
        _pipe.resample_uniform(center, s.rs_out, Ncur, s.count, s.rs_seg, s.rs_s, device=dev)
        wp.copy(out_centerline, s.rs_out, count=E * Ncur)
        wp.launch(_pipe._fill_i32_k, dim=E, inputs=[out_valid_wp, 1], device=dev)
        _pipe._sync(dev)

    return generate


def last_overflow(scratch):
    """Total candidate-list overflow (pairs dropped past maxnbr) from the last run."""
    nf = _NF_CACHE.get(id(scratch))
    if nf is None:
        return None
    return int(nf.overflow.numpy().sum())
