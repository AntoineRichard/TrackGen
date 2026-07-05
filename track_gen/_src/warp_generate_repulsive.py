"""Repulsive-growth centerline generator (``generator="repulsive"``).

Grows each env's closed centerline from a small circle under a hard ratcheting
length constraint while a tangent-point (TP) energy keeps the curve self-avoiding,
confined to a disc domain seeded with per-env random disc obstacles. Coarse-to-fine
``N = 64 -> 128 -> 256`` with an area-based stall-stop. The result is paper-quality
serpentine tracks (Henrich et al., *Generating Race Tracks With Repulsive Curves*;
Yu/Schumacher/Crane, *Repulsive Curves* SIGGRAPH 2021).

This is the FIRST non-CUDA-graph-capturable generator: the growth loop reads back an
area-convergence scalar every stall window to drive the early exit and transitions the
coarse-to-fine stages on the host -- both illegal inside a capture region. It registers
with ``capturable=False`` and ``TrackGenerator`` runs it EAGERLY on CUDA (see the cost
warning on ``TrackGenConfig.repulsive_grow_mult_min``). The per-iteration ``wp.Tape``
that used to be the interior capture blocker is GONE (replaced by hand-written analytic
adjoints below); actual graph capture is now just a wiring exercise (future work).

Ported from the validated spike ``docs/superpowers/spikes/2026-07-05-repulsive-growth-
phase1/grow_warp.py`` (proven 64/64 through the standard tail). Production changes:
(a) obstacle layout is a seed-driven Warp kernel (not host torch rejection sampling);
(b) scratch is allocated once, coarse stages slice the ``[0:E*N_stage]`` prefix;
(c) the stage upsample uses ``warp_pipeline.arc_length_resample_inplace``.

**torch-free:** imports only ``warp`` + ``numpy`` (host precompute uses ``numpy.fft``
for the Sobolev circulant rows and ``numpy.log1p`` for the stage schedule).

**Determinism (byte-identical per device, CPU AND CUDA):**
The growth gradient comes from HAND-WRITTEN ANALYTIC ADJOINTS (see the gradient kernels
below), not ``wp.Tape``. Every gradient kernel is a per-vertex GATHER -- one thread sums
dE/dx_v into a local register in a fixed loop order, with NO atomics anywhere. That makes
the whole flow bit-reproducible run-to-run on a given device: same config + seeds -> the
same centerline, byte for byte, on both CPU and CUDA. (The old ``wp.Tape`` path accumulated
the adjoint through nondeterministic ``atomic_add``, whose ~2e-6 float-order noise amplified
chaotically -- a fold is a near-buckling instability -- into macroscopically different, if
statistically equivalent, CUDA tracks; that is fixed.) Cross-DEVICE equality is NOT claimed:
fp32 rounding differs between CPU and CUDA, so a CPU track and a CUDA track for the same seed
are statistically equivalent but not bit-identical to each other. The per-generator
determinism test asserts byte-identity independently on each available device.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import warp_pipeline as _pipe

# --- Obstacle layout constants (mirror the spike's sample_obstacles reference layout) ---
# Independent RNG salt, distinct from the site/anchor/checkpoint/control salts.
_OBSTACLE_SALT = 6247
_N_WALL = 96          # points on the domain-wall ring (weight 1.0)
_N_DISC = 12          # points per inner-disc ring (weight 0.25)
_C_FRAC = 0.90        # inner-disc radial-band outer bound as a fraction of r_dom
_WALL_WEIGHT = 1.0
_DISC_WEIGHT = 0.25

# --- World-scale anchor (design §5, amended) ---
# All absolute sizes anchor to ``config.scale`` like every other generator (polar/voronoi
# anchor to the bezier extent 1.44). ``_DOMAIN_SCALE_REF`` is the world-scale reference
# LENGTH in config.scale units: at scale=1.0 it equals the spike's build_setup median
# bezier perimeter, so the domain radius / init radius / obstacle radii / target lengths
# are numerically identical to the spike's validated 64/64 regime.
#
# NOTE (amended from the spec's §5): the design cited a bezier-coupled, per-batch-measured
# ``_BEZIER_PERIMETER_REF ≈ 5.05``. That is REPLACED here by a fixed scale-anchored
# constant (no bezier coupling, no per-batch measurement, no final rescale). The exact
# value is the spike's *actual* build_setup median at scale=1 (E=64, the validated config),
# which measures 4.9029 today -- the design's 5.05 was an earlier/approximate figure that
# has drifted with the bezier defaults (exactly the risk the design flagged). Using the
# spike-derived value keeps the generator IDENTICAL to the validated ground truth.
_DOMAIN_SCALE_REF = 4.9029


class _RepulsiveScalars:
    """Host-side derived scalars for one config (geometry + obstacle layout bounds)."""

    __slots__ = (
        "E", "P_ref", "r_dom", "r_init", "M", "n_wall", "n_disc",
        "k_min", "k_max", "r_frac_lo", "r_frac_hi", "c_frac",
    )

    def __init__(self, config) -> None:
        self.E = int(config.num_envs)
        self.P_ref = _DOMAIN_SCALE_REF * float(config.scale)
        self.r_dom = float(config.repulsive_domain_frac) * self.P_ref
        self.r_init = self.r_dom / float(config.repulsive_domain_init_ratio)
        self.n_wall = _N_WALL
        self.n_disc = _N_DISC
        self.k_min = int(config.repulsive_obstacle_count_min)
        self.k_max = int(config.repulsive_obstacle_count_max)
        # M sizes the obstacle buffer at the max possible disc count (K_max = k_max).
        self.M = self.n_wall + self.k_max * self.n_disc
        self.r_frac_lo = float(config.repulsive_obstacle_radius_min_frac)
        self.r_frac_hi = float(config.repulsive_obstacle_radius_max_frac)
        self.c_frac = _C_FRAC


# ===========================================================================
# Seed-driven, device-side obstacle layout (replaces the spike's host torch
# rejection sampling). Deterministic per (seed, env); no rejection loop.
# ===========================================================================

@wp.kernel
def _sample_obstacles_k(
    seeds: wp.array(dtype=wp.int32),
    r_dom: float, r_init: float,
    r_frac_lo: float, r_frac_hi: float, c_frac: float,
    k_min: int, k_max: int, n_wall: int, n_disc: int, m_obs: int,
    nan_val: float,
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32),
):
    # One thread per env. Writes a domain-wall ring (weight 1.0) plus k inner-disc rings
    # (weight 0.25) at angular-stratified positions with an analytic radial band -- no
    # rejection loop. Unused / skipped disc columns are NaN-padded with weight 0 so the
    # obstacle-energy kernel (which guards on ``mw != 0``) never reads them.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _OBSTACLE_SALT + 23)
    two_pi = 2.0 * wp.pi
    base = e * m_obs

    # --- domain-wall ring at r_dom (weight 1.0; mass = ring arc spacing) ---
    r_wall = r_dom
    wall_mw = (two_pi * r_wall / float(n_wall)) * _WALL_WEIGHT
    for iw in range(n_wall):
        ang = float(iw) * two_pi / float(n_wall)
        obs_pts[base + iw] = wp.vec2f(r_wall * wp.cos(ang), r_wall * wp.sin(ang))
        obs_mw[base + iw] = wall_mw

    # --- k inner discs, one per 2*pi/k wedge, per-env phase ---
    k = wp.randi(state, k_min, k_max + 1)
    phase = wp.randf(state) * two_pi
    for j in range(k_max):
        col = base + n_wall + j * n_disc
        if j < k:
            r_disc = (r_frac_lo + (r_frac_hi - r_frac_lo) * wp.randf(state)) * r_dom
            # Radial band: the ring's inner edge clears the init circle by 0.05*r_dom (else
            # the p=beta-alpha energy blows up at t=0); the outer bound c_frac*r_dom stays
            # inside the wall. If the band is empty (tight config) the column is left unused.
            lo = r_init + r_disc + 0.05 * r_dom
            hi = c_frac * r_dom
            rad = lo + (hi - lo) * wp.randf(state)
            active = float(1.0)
            if lo >= hi:
                active = float(0.0)
            ang = phase + float(j) * two_pi / float(k)
            cx = rad * wp.cos(ang)
            cy = rad * wp.sin(ang)
            disc_mw = (two_pi * r_disc / float(n_disc)) * _DISC_WEIGHT
            for idx in range(n_disc):
                a = float(idx) * two_pi / float(n_disc)
                if active == 1.0:
                    obs_pts[col + idx] = wp.vec2f(cx + r_disc * wp.cos(a),
                                                  cy + r_disc * wp.sin(a))
                    obs_mw[col + idx] = disc_mw
                else:
                    obs_pts[col + idx] = wp.vec2f(nan_val, nan_val)
                    obs_mw[col + idx] = 0.0
        else:
            for idx in range(n_disc):
                obs_pts[col + idx] = wp.vec2f(nan_val, nan_val)
                obs_mw[col + idx] = 0.0


def _sample_obstacles_inplace(seeds_wp: wp.array, config,
                              obs_pts: wp.array, obs_mw: wp.array,
                              device: str) -> None:
    """Launch the seed-driven obstacle kernel in place. obs_pts/obs_mw are [E*M]."""
    _pipe._init()
    s = _RepulsiveScalars(config)
    wp.launch(_sample_obstacles_k, dim=s.E,
              inputs=[seeds_wp, s.r_dom, s.r_init, s.r_frac_lo, s.r_frac_hi, s.c_frac,
                      s.k_min, s.k_max, s.n_wall, s.n_disc, s.M, float("nan"),
                      obs_pts, obs_mw],
              device=device)
    # No host sync here: the growth-loop launches that consume obs_pts/obs_mw are already
    # stream-ordered after this kernel, and generate_repulsive_warp syncs once before the
    # first host readback. (Callers that read obs back on the host -- e.g. tests -- go through
    # wp.array.numpy(), which synchronizes on its own.)


# RNG salt for the per-env target-length draw (distinct from the obstacle salt).
_GROW_SALT = 5209
# Coarse-to-fine upsample trigger: subdivide N -> 2N when the mean edge has grown by this
# factor (the reference's edge-length-doubling subdivision rule). Spike default 2.0.
_SUBDIV_RATIO = 2.0


@wp.kernel
def _seed_lfinal_k(
    seeds: wp.array(dtype=wp.int32),
    r_init: float, grow_lo: float, grow_hi: float,
    L_final: wp.array(dtype=wp.float32), L_init: wp.array(dtype=wp.float32),
    L_target: wp.array(dtype=wp.float32), reached: wp.array(dtype=wp.int32),
    frozen: wp.array(dtype=wp.int32), area_prev: wp.array(dtype=wp.float32),
):
    # One thread per env. Draw the per-env target-perimeter multiple and seed all per-env
    # growth state. L_init = L_target = init-circle perimeter; L_final = grow_mult * L_init.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _GROW_SALT + 41)
    grow_mult = grow_lo + (grow_hi - grow_lo) * wp.randf(state)
    l0 = 2.0 * wp.pi * r_init
    L_init[e] = l0
    L_target[e] = l0
    L_final[e] = grow_mult * l0
    reached[e] = 0
    frozen[e] = 0
    area_prev[e] = 0.0


@wp.kernel
def _init_circle_k(center: wp.array(dtype=wp.vec2f), n: int, r_init: float):
    # One thread per (env, point). Writes the radius-r_init circle into the coarsest-stage
    # prefix (stride n). Uniform across envs; growth breaks the symmetry.
    e, i = wp.tid()
    ang = float(i) * (2.0 * wp.pi / float(n))
    center[e * n + i] = wp.vec2f(r_init * wp.cos(ang), r_init * wp.sin(ang))


# ===========================================================================
# Hand-written analytic gradient kernels (replace wp.Tape; grad -> scratch.grad).
#
# The total per-iteration energy is E = E_TP + E_obs + E_len, matching the spike's three
# forward kernels exactly:
#   E_TP  = sum_{i,j: circ>2} (|wedge_ij|+eps)^alpha / (d2_ij+eps^2)^(beta/2) * w_i * w_j
#           with T_i = safe_dir(x_{i+1}-x_{i-1}), w_i = 0.5(|x_{i+1}-x_i|+|x_i-x_{i-1}|),
#           diff_ij = x_j - x_i, wedge_ij = diff_ij x T_i (cross), d2_ij = |diff_ij|^2.
#   E_obs = sum_i sum_m mw_m * (|x_i - p_m|^2 + 1e-8)^p_exp        (obstacles fixed)
#   E_len = w_len * ((perimeter - L_target)/L_init)^2
#
# dE/dx_v is assembled as a per-vertex GATHER (one thread per v, register accumulate, no
# atomics). x_v enters E_TP four ways: as x_i in pairs (v,*), as x_j in pairs (*,v), through
# the tangents T_{v-1},T_{v+1} of its neighbors, and through the dual weights w_{v-1},w_v,
# w_{v+1}. The tangent/weight couplings reduce to per-vertex accumulators (btan_i = J_i A_i,
# wcoef_k = C_k + D_k) computed in a first pass and read at the v+-1 stencil in the gather;
# the diff coupling is a genuine O(N) pair sum done in the gather. Derivation validated
# against a float64 numpy reference + central finite differences to ~1e-10 rel (see
# tests/test_generate_repulsive.py::test_analytic_gradient_matches_reference).
# ===========================================================================

@wp.func
def _safe_dir(v: wp.vec2f) -> wp.vec2f:
    # safe_normalize with the oracle's 1e-8 floor (matches geometry.safe_normalize).
    return v / wp.max(wp.length(v), float(1.0e-8))


@wp.kernel
def _tangent_weight_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    frozen: wp.array(dtype=wp.int32),
    tang: wp.array(dtype=wp.vec2f), wdual: wp.array(dtype=wp.float32),
):
    # O(E*N) precompute (once per iteration): the per-vertex tangent T_v = safe_dir(x_{v+1}-
    # x_{v-1}) and dual weight w_v = 0.5(|x_{v+1}-x_v|+|x_v-x_{v-1}|). Both TP pair kernels
    # (_tp_prepass_k, _grad_gather_k) read T[j]/w[j] and T[t]/w[t] from these buffers instead
    # of recomputing them N times inside their O(N^2) partner loops. Bit-identical to the old
    # inline formulas (same ops, same inputs), so the gradient and trajectory are unchanged.
    # Frozen envs skip it (both consumers also skip frozen envs -> stale values never read).
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    xv = center[b + v]
    xvn = center[b + (v + 1) % n]
    xvp = center[b + (v + n - 1) % n]
    tang[b + v] = _safe_dir(xvn - xvp)
    wdual[b + v] = 0.5 * (wp.length(xvn - xv) + wp.length(xv - xvp))


@wp.kernel
def _tp_prepass_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    alpha: float, beta: float, eps: float,
    frozen: wp.array(dtype=wp.int32),
    tang: wp.array(dtype=wp.vec2f), wdual: wp.array(dtype=wp.float32),
    wcoef: wp.array(dtype=wp.float32), btan: wp.array(dtype=wp.vec2f),
):
    # First pass of the TP gradient: one thread per vertex v accumulates the tangent- and
    # weight-mechanism reductions that the gather reads at the v+-1 stencil.
    #   wcoef[v] = C_v + D_v,  C_v = sum_j P(v,j) w_j,  D_v = sum_i P(i,v) w_i
    #   btan[v]  = J_v A_v,    A_v = sum_j dP/dT_v(v,j) * w_v * w_j,  J_v = (I - T_v T_v^T)/|u_v|
    # where P(i,j) = (|wedge_ij|+eps)^alpha / (d2_ij+eps^2)^(beta/2). Per-vertex tangents/weights
    # come from _tangent_weight_k's precomputed buffers. Frozen envs skip it (the gather also
    # skips them, so the stale values are never read).
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
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
    for j in range(n):
        dd = wp.abs(v - j)
        circ = wp.min(dd, n - dd)
        if circ > 2:
            xj = center[b + j]
            Tj = tang[b + j]
            wj = wdual[b + j]
            diff = xj - xv                     # pair (v,j): owner v
            d2 = wp.dot(diff, diff)
            den = wp.pow(d2 + eps * eps, hb)
            wedge = diff[0] * Tv[1] - diff[1] * Tv[0]
            aw = wp.abs(wedge) + eps
            P_vj = wp.pow(aw, alpha) / den
            g_num = alpha * wp.pow(aw, alpha - 1.0) * wp.where(wedge < 0.0, float(-1.0), float(1.0))
            # dP/dT_v = (g_num/den) * (-diff.y, diff.x)
            A += (g_num / den) * wp.vec2f(-diff[1], diff[0]) * wj
            C += P_vj * wj
            # pair (j,v): owner j, diff2 = xv - xj (= -diff), same d2/den, tangent T_j
            wedge2 = -diff[0] * Tj[1] + diff[1] * Tj[0]
            P_jv = wp.pow(wp.abs(wedge2) + eps, alpha) / den
            D += P_jv * wj
    wcoef[b + v] = C + D
    Af = wv * A
    btan[b + v] = (Af - Tv * wp.dot(Tv, Af)) / lv


@wp.kernel
def _len_coef_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    L_target: wp.array(dtype=wp.float32), L_init: wp.array(dtype=wp.float32),
    w_len: float, frozen: wp.array(dtype=wp.int32),
    len_coef: wp.array(dtype=wp.float32), peri_out: wp.array(dtype=wp.float32),
):
    # Per-env scalar dE_len/dperimeter = 2*w_len*(perimeter - L_target)/L_init^2. The gather
    # multiplies it by dperimeter/dx_v = dir(x_v-x_{v-1}) - dir(x_{v+1}-x_v). The perimeter is
    # also stashed in peri_out so _gmax_msl_k (same iteration, center still unmodified) reuses
    # it for msl instead of summing the whole loop a second time. Computed for every env
    # (including frozen ones) so peri_out is always fresh.
    e = wp.tid()
    b = e * n
    peri = float(0.0)
    for i in range(n):
        peri += wp.length(center[b + (i + 1) % n] - center[b + i])
    peri_out[e] = peri
    if frozen[e] == 1:
        len_coef[e] = 0.0
        return
    li = L_init[e]
    len_coef[e] = 2.0 * w_len * (peri - L_target[e]) / (li * li)


@wp.kernel
def _grad_gather_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    alpha: float, beta: float, eps: float,
    tang: wp.array(dtype=wp.vec2f), wdual: wp.array(dtype=wp.float32),
    wcoef: wp.array(dtype=wp.float32), btan: wp.array(dtype=wp.vec2f),
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32), m_obs: int,
    p_exp: float, reached: wp.array(dtype=wp.int32),
    n_wall: int, deac: int,
    len_coef: wp.array(dtype=wp.float32),
    frozen: wp.array(dtype=wp.int32),
    grad: wp.array(dtype=wp.vec2f),
):
    # One thread per vertex v: gather the full dE/dx_v (TP diff + weight + tangent stencils,
    # obstacle self-term, length regularizer) into a register and overwrite grad[v]. No
    # atomics. Per-vertex tangents/weights read from _tangent_weight_k's precomputed buffers.
    # Frozen envs return early (grad[v] stale; the preconditioner also skips them).
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

    # --- TP diff mechanism: sum over partners t of ww*(dP_ddiff(t,v) - dP_ddiff(v,t)) ---
    for t in range(n):
        dd = wp.abs(v - t)
        circ = wp.min(dd, n - dd)
        if circ > 2:
            xt = center[b + t]
            Tt = tang[b + t]
            wt = wdual[b + t]
            ww = wv * wt
            diff = xt - xv                     # pair (v,t): owner v
            d2 = wp.dot(diff, diff)
            de2 = d2 + eps * eps
            den = wp.pow(de2, hb)
            wedge = diff[0] * Tv[1] - diff[1] * Tv[0]
            aw = wp.abs(wedge) + eps
            P = wp.pow(aw, alpha) / den
            gn = alpha * wp.pow(aw, alpha - 1.0) * wp.where(wedge < 0.0, float(-1.0), float(1.0))
            dP_vt = (gn / den) * wp.vec2f(Tv[1], -Tv[0]) - (P * beta / de2) * diff
            # pair (t,v): owner t, diff2 = xv - xt (= -diff), same d2/den, tangent T_t
            diff2 = xv - xt
            wedge2 = diff2[0] * Tt[1] - diff2[1] * Tt[0]
            aw2 = wp.abs(wedge2) + eps
            P2 = wp.pow(aw2, alpha) / den
            gn2 = alpha * wp.pow(aw2, alpha - 1.0) * wp.where(wedge2 < 0.0, float(-1.0), float(1.0))
            dP_tv = (gn2 / den) * wp.vec2f(Tt[1], -Tt[0]) - (P2 * beta / de2) * diff2
            g += ww * (dP_tv - dP_vt)

    # --- TP weight mechanism (v+-1 stencil over wcoef) ---
    ef = xvn - xv
    eb = xv - xvp
    def_dir = _safe_dir(ef)
    deb_dir = _safe_dir(eb)
    Wv = wcoef[b + v]
    Wprev = wcoef[b + (v + n - 1) % n]
    Wnext = wcoef[b + (v + 1) % n]
    g += 0.5 * deb_dir * (Wv + Wprev) - 0.5 * def_dir * (Wv + Wnext)

    # --- TP tangent mechanism (v+-1 stencil over btan) ---
    g += btan[b + (v + n - 1) % n] - btan[b + (v + 1) % n]

    # --- obstacle self-term ---
    ob = e * m_obs
    is_reached = reached[e]
    for m in range(m_obs):
        mw = obs_mw[ob + m]
        if mw != 0.0:
            # Deactivating only closes the inner-disc halos (m >= n_wall) once an env reaches
            # its target length; the domain wall (m < n_wall) always stays live. The spike
            # documented wall-deactivation as UNSAFE under a fixed iteration budget (the curve
            # can escape the domain before it settles), so there is no wall-deactivation knob.
            # See docs/superpowers/spikes/2026-07-05-repulsive-growth-phase1/README.md.
            drop = int(0)
            if deac == 1 and is_reached == 1 and m >= n_wall:
                drop = int(1)
            if drop == 0:
                d = xv - obs_pts[ob + m]
                d2 = wp.dot(d, d)
                g += (mw * p_exp * wp.pow(d2 + float(1.0e-8), p_exp - 1.0) * 2.0) * d

    # --- length regularizer ---
    g += len_coef[e] * (deb_dir - def_dir)

    grad[b + v] = g


# ===========================================================================
# Optimizer-step kernels (consume the gradient; no adjoint needed)
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
    # fractional-Sobolev preconditioner A^{-1}. h is the fixed circulant row (numpy irfft
    # of the spectral filter). O(N^2); frozen envs skip it (stale out never used).
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
                peri_in: wp.array(dtype=wp.float32), n: int,
                gmax: wp.array(dtype=wp.float32), msl: wp.array(dtype=wp.float32)):
    # gmax = max_i |g[i]-gmean|; msl = mean segment length. The perimeter comes from
    # _len_coef_k's peri_out (same iteration, center unchanged between the two launches), so
    # this kernel no longer re-sums the whole loop -- msl = perimeter / n, bit-identical.
    e = wp.tid()
    b = e * n
    gm = float(0.0)
    for i in range(n):
        gm = wp.max(gm, wp.length(g[b + i] - gmean[e]))
    gmax[e] = wp.max(gm, float(1.0e-12))
    msl[e] = peri_in[e] / float(n)


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
    # Per-env stall detector (once per stall window). An env freezes when it has reached its
    # final length AND its enclosed-area relative change over the window is below thresh.
    # Area (shoelace) is REPARAMETERIZATION-INVARIANT, unlike per-bead displacement.
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
# Host-side spectral filter + stage schedule (numpy only, precomputed once)
# ===========================================================================

def _sobolev_circulant_row(n, s, eps_reg):
    """Real-space circulant first row h of A^{-1}: numpy irfft of the ring spectral filter
    1/(lam_k^s + eps_reg). A^{-1} g = circular-conv(h, g). Matches the rfft preconditioner
    to ~1e-4 abs / ~8e-7 rel."""
    k = np.arange(n // 2 + 1)
    lam = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / n)
    inv_filter = 1.0 / (np.clip(lam, 0.0, None) ** s + eps_reg)
    return np.fft.irfft(inv_filter, n=n).astype(np.float32)


def _stage_schedule(stages, growth, subdiv_ratio, n_ratchet):
    """Deterministic per-stage START ITER (no device sync). The hard-constraint ratchet
    makes the fastest env's perimeter L_init*(1+growth)^it, so the iter at which the mean
    edge has grown by subdiv_ratio^i is a closed form. Each transition is clamped to
    n_ratchet so the FINAL stage is always entered by the time the target length is reached
    (the settle phase, and hence the tail input, stays at full resolution)."""
    starts = [0]
    for i in range(1, len(stages)):
        it_i = int(np.ceil(np.log(subdiv_ratio ** i) / np.log1p(growth)))
        starts.append(min(it_i, n_ratchet))
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
    return starts


# ===========================================================================
# Scratch (allocated ONCE at the max stage N=256; coarse stages slice the prefix)
# ===========================================================================

class RepulsiveScratch:
    """Private working buffers for the repulsive generator (one alloc per generator).

    All N-dependent buffers are sized at the max stage ``Nmax = stages[-1] = num_points``;
    coarse stages operate on the ``[0:E*N_stage]`` prefix. The per-stage Sobolev circulant
    rows and the closed-form stage schedule are precomputed once on host (numpy).
    """

    __slots__ = (
        "E", "Nmax", "M", "n_wall", "n_disc", "r_dom", "r_init",
        "stages", "stage_starts", "n_iters", "h_by_n",
        "center", "grad", "tang", "wdual", "wcoef", "btan", "len_coef", "peri_e",
        "obs_pts", "obs_mw",
        "g", "lg", "ainv_lg", "rs_out", "rs_seg", "rs_s",
        "arc_real", "arc_cr", "arc_co", "count",
        "L_final", "L_init", "L_target", "reached", "frozen", "area_prev", "md_out",
        "num", "den", "gmax", "msl", "cur_len", "gmean", "bc",
    )

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def repulsive_alloc_scratch(config):
    """Allocate the repulsive generator's PRIVATE scratch (one alloc per generator).

    Sizes every N-dependent buffer at ``Nmax`` and precomputes the host-side Sobolev rows +
    stage schedule + iteration budget from the config's ratchet/stage bounds (no device
    readback needed to size the loop, since ``n_ratchet`` follows from ``grow_mult_max``)."""
    _pipe._init()
    sc = _RepulsiveScalars(config)
    E = sc.E
    dev = str(config.device)
    stages = [int(s) for s in config.repulsive_stages]
    Nmax = stages[-1]
    M = sc.M

    growth = float(config.repulsive_ratchet_rate)
    settle_iters = int(config.repulsive_settle_iters)
    alpha = float(config.repulsive_alpha)
    beta = float(config.repulsive_beta)
    # n_ratchet from the UPPER grow-mult bound: L_final/L_init <= grow_mult_max, so this
    # bounds every env's ratchet length without a device readback.
    n_ratchet = int(np.ceil(np.log(float(config.repulsive_grow_mult_max)) / np.log1p(growth)))
    n_iters = int(np.ceil(n_ratchet * 1.6)) + settle_iters
    stage_starts = _stage_schedule(stages, growth, _SUBDIV_RATIO, n_ratchet)

    s_exp = (beta - 1.0) / (2.0 * alpha)
    h_by_n = {}
    for Ns in set(stages):
        h_np = _sobolev_circulant_row(Ns, s_exp, 1e-3)
        h_by_n[Ns] = wp.array(h_np, dtype=wp.float32, device=dev)

    def vec(n):
        return wp.zeros(n, dtype=wp.vec2f, device=dev)

    def f32(n):
        return wp.zeros(n, dtype=wp.float32, device=dev)

    def i32(n):
        return wp.zeros(n, dtype=wp.int32, device=dev)

    return RepulsiveScratch(
        E=E, Nmax=Nmax, M=M, n_wall=sc.n_wall, n_disc=sc.n_disc,
        r_dom=sc.r_dom, r_init=sc.r_init,
        stages=stages, stage_starts=stage_starts, n_iters=n_iters, h_by_n=h_by_n,
        center=wp.zeros(E * Nmax, dtype=wp.vec2f, device=dev),
        grad=vec(E * Nmax), tang=vec(E * Nmax), wdual=f32(E * Nmax),
        wcoef=f32(E * Nmax), btan=vec(E * Nmax), len_coef=f32(E), peri_e=f32(E),
        obs_pts=vec(E * M), obs_mw=f32(E * M),
        g=vec(E * Nmax), lg=vec(E * Nmax), ainv_lg=vec(E * Nmax), rs_out=vec(E * Nmax),
        rs_seg=f32(E * Nmax), rs_s=f32(E * (Nmax + 1)),
        # arc_real is the only stage-transition-specific scratch; the seg/s scan buffers alias
        # the periodic-resample rs_seg/rs_s (stage transitions and periodic resamples are
        # strictly sequential on the same stream, never live at once). Saves ~2*E*Nmax f32.
        arc_real=vec(E * Nmax),
        arc_cr=i32(E), arc_co=i32(E), count=i32(E),
        L_final=f32(E), L_init=f32(E), L_target=f32(E), reached=i32(E), frozen=i32(E),
        area_prev=f32(E), md_out=f32(E), num=f32(E), den=f32(E), gmax=f32(E), msl=f32(E),
        cur_len=f32(E), gmean=vec(E), bc=vec(E),
    )


# ===========================================================================
# Host-driven coarse-to-fine growth loop (eager; non-capturable)
# ===========================================================================

def generate_repulsive_warp(seeds_wp: wp.array, config,
                            out_centerline: wp.array, out_valid_wp: wp.array,
                            scratch) -> None:
    """Grow E closed centerlines from small circles with the pure-Warp TP-Sobolev flow and
    write the final ``[E*num_points]`` closed loops into ``out_centerline`` (valid=1 for all
    envs; the shared downstream gate decides real validity).

    Host-driven (stage transitions, per-window stall readback, early exit) -- illegal inside
    a CUDA graph capture, hence ``capturable=False``. The gradient is hand-written analytic
    adjoints (per-vertex gather, no atomics), so the flow is byte-deterministic per device.
    Ported from the validated spike ``grow_warp.grow_warp`` (with the tape replaced).
    """
    _pipe._init()
    assert scratch is not None, "generate_repulsive_warp requires scratch"
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

    center = s.center

    # 1. Seed-driven obstacle layout + per-env target length.
    _sample_obstacles_inplace(seeds_wp, config, s.obs_pts, s.obs_mw, dev)
    grow_lo = float(config.repulsive_grow_mult_min)
    grow_hi = float(config.repulsive_grow_mult_max)
    wp.launch(_seed_lfinal_k, dim=E,
              inputs=[seeds_wp, s.r_init, grow_lo, grow_hi,
                      s.L_final, s.L_init, s.L_target, s.reached, s.frozen, s.area_prev],
              device=dev)

    # 2. Initial circle at the coarsest stage prefix.
    N0 = stages[0]
    wp.launch(_init_circle_k, dim=(E, N0), inputs=[center, N0, s.r_init], device=dev)

    stage_idx = 0
    Ncur = N0

    def _upsample_to_next():
        # Advance one coarse-to-fine stage: arc-length resample center N -> next N. The
        # seg/s scan scratch aliases the periodic-resample rs_seg/rs_s (never live at once).
        nonlocal stage_idx, Ncur
        stage_idx += 1
        Nnew = stages[stage_idx]
        _pipe.arc_length_resample_inplace(
            center[0:E * Ncur], Ncur, Nnew,
            s.arc_real[0:E * Ncur], s.rs_seg[0:E * Ncur], s.rs_s[0:E * (Ncur + 1)],
            s.arc_cr, s.arc_co, s.rs_out[0:E * Nnew], dev)
        wp.copy(center, s.rs_out, count=E * Nnew)
        Ncur = Nnew

    for it in range(n_iters):
        # --- coarse-to-fine stage transition (upsample center N -> next N) ---
        while stage_idx + 1 < len(stages) and it >= stage_starts[stage_idx + 1]:
            _upsample_to_next()

        h_wp = s.h_by_n[Ncur]

        # 1. ratchet target + deactivation flag
        wp.launch(_ratchet_k, dim=E, inputs=[s.L_target, s.L_final, growth, s.reached], device=dev)

        # 2. energy gradient via hand-written analytic adjoints (all 3 terms -> s.grad).
        #    Per-vertex gather, no atomics -> byte-deterministic per device. Per-vertex
        #    tangents/weights are precomputed once (O(E*N)) and shared by both pair kernels.
        wp.launch(_tangent_weight_k, dim=(E, Ncur),
                  inputs=[center, Ncur, s.frozen, s.tang, s.wdual], device=dev)
        wp.launch(_tp_prepass_k, dim=(E, Ncur),
                  inputs=[center, Ncur, alpha, beta, eps, s.frozen,
                          s.tang, s.wdual, s.wcoef, s.btan], device=dev)
        wp.launch(_len_coef_k, dim=E,
                  inputs=[center, Ncur, s.L_target, s.L_init, w_len, s.frozen,
                          s.len_coef, s.peri_e], device=dev)
        wp.launch(_grad_gather_k, dim=(E, Ncur),
                  inputs=[center, Ncur, alpha, beta, eps, s.tang, s.wdual, s.wcoef, s.btan,
                          s.obs_pts, s.obs_mw, M, p_exp, s.reached,
                          n_wall, deac_i, s.len_coef, s.frozen, s.grad], device=dev)

        # 3. Sobolev precondition g = A^{-1} grad
        wp.launch(_conv_k, dim=(E, Ncur), inputs=[s.grad, h_wp, Ncur, s.frozen, s.g], device=dev)
        # 4. length-gradient Sobolev-orthogonal projection
        wp.launch(_length_grad_k, dim=(E, Ncur), inputs=[center, Ncur, s.lg], device=dev)
        wp.launch(_conv_k, dim=(E, Ncur), inputs=[s.lg, h_wp, Ncur, s.frozen, s.ainv_lg], device=dev)
        wp.launch(_numden_k, dim=E, inputs=[s.g, s.lg, s.ainv_lg, Ncur, s.num, s.den], device=dev)
        wp.launch(_project_k, dim=(E, Ncur), inputs=[s.g, s.ainv_lg, Ncur, s.num, s.den], device=dev)
        # 5. barycenter pin + normalized step
        wp.launch(_gmean_k, dim=E, inputs=[s.g, Ncur, s.gmean], device=dev)
        wp.launch(_gmax_msl_k, dim=E, inputs=[s.g, s.gmean, s.peri_e, Ncur, s.gmax, s.msl], device=dev)
        wp.launch(_step_k, dim=(E, Ncur),
                  inputs=[center, s.g, s.gmean, s.msl, s.gmax, tau, Ncur, s.frozen], device=dev)
        # 6. hard rescale to the ratcheted target
        wp.launch(_perim_bc_k, dim=E, inputs=[center, Ncur, s.cur_len, s.bc], device=dev)
        wp.launch(_rescale_k, dim=(E, Ncur),
                  inputs=[center, s.bc, s.cur_len, s.L_target, Ncur, s.frozen], device=dev)

        # 7. periodic arc-length resample (pure Warp)
        if (it + 1) % resample_every == 0:
            wp.launch(_pipe._fill_i32_k, dim=E, inputs=[s.count, Ncur], device=dev)
            _pipe.resample_uniform(center, s.rs_out, Ncur, s.count, s.rs_seg, s.rs_s, device=dev)
            wp.copy(center, s.rs_out, count=E * Ncur)

        # 8. stall-stop: freeze converged envs, global early exit when all frozen.
        if Ncur == stages[-1] and (it + 1) % stall_window == 0:
            wp.launch(_freeze_update_k, dim=E,
                      inputs=[center, Ncur, s.reached, stall_tol, s.area_prev, s.frozen, s.md_out],
                      device=dev)
            if int(s.frozen.numpy().sum()) >= E:
                break

    # Defense in depth: guarantee the centerline reaches the final stage resolution
    # (stages[-1] == num_points) even if the iteration budget was too small to trigger every
    # coarse-to-fine transition on schedule. Otherwise the tail resample/copy below would pack
    # stride-Ncur data into the stride-Nmax out_centerline (silent mis-strided garbage). Config
    # validation already forbids n_iters<=0; this keeps the stride invariant unconditional.
    while stage_idx + 1 < len(stages):
        _upsample_to_next()

    # final periodic resample at Nmax, then hand the closed loop to the standard tail.
    wp.launch(_pipe._fill_i32_k, dim=E, inputs=[s.count, Ncur], device=dev)
    _pipe.resample_uniform(center, s.rs_out, Ncur, s.count, s.rs_seg, s.rs_s, device=dev)
    wp.copy(out_centerline, s.rs_out, count=E * Ncur)
    wp.launch(_pipe._fill_i32_k, dim=E, inputs=[out_valid_wp, 1], device=dev)
    _pipe._sync(dev)


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="repulsive",
    alloc_scratch=repulsive_alloc_scratch,
    generate=generate_repulsive_warp,
    capturable=False,
))
