"""Repulsive growth with analytic circular barriers and bounded accepted steps.

The tangent-point energy uses disjoint-edge tangents and trapezoidal endpoint
quadrature. Its fixed circulant Sobolev surrogate uses the intended differential
order and normalized nonzero ring modes. Shape steps use bounded Armijo
backtracking with perimeter projection recomputed at each trial; a separate bounded expansion approaches the
ratcheted target. Every accepted shape/growth update and same-resolution
resampling clears all analytic discs and the domain, including swept edges.
Refinement inserts vertices along existing edges and preserves the exact loop.

The sampled rings contribute a curve-length-weighted shaping potential. An
always-active analytic log potential steers the curve before hard constraints
become active. Known constant road width is reserved during growth; the actual
finished offset band is checked by the pipeline after its shared tail.

The static iteration/refinement schedule is CUDA-graph capturable through
``ctx.bounded_loop``. An environment freezes only after reaching its actual
length target with small area change, accepted movement and stationarity;
rejected searches do not count as convergence. Freeze is absorbing, making
poll/unrolled/conditional lowerings byte-equivalent on each device. Unmet or
infeasible target lengths report invalid through the standard commit policy.
A feasible target-length budget result can be valid without reaching optimizer
equilibrium; frozen is only an early-stop diagnostic. Scratch is allocated once;
all launches and copies use the caller's context.
"""
from __future__ import annotations

import math
import types

import numpy as np
import warp as wp

from ..ops import rng as _rng

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs
from ..ops.constants import SIZE_BOUNDS
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import RepulsiveParams
from .runtime import _init
from .repulsive_tangent_energy import directed_edge_pair_energy, directed_edge_pair_gradient
from .repulsive_barrier_energy import (barrier_potential_gradient,
                                      segment_barrier_potential_gradient)
from .repulsive_clearance import (RepulsiveBarriers, publish_barriers_k,
                                  segment_disc_clear, swept_segment_disc_clear,
                                  _tolerance as _clearance_tolerance)

# --- Obstacle layout constants (mirror the spike's sample_obstacles reference layout) ---
# Independent RNG salt, distinct from the site/anchor/checkpoint/control salts.

_OBSTACLE_SALT = 0x15710913
_N_WALL = 96          # points on the domain-wall ring (weight 1.0)
_N_DISC = 12          # points per inner-disc ring (weight 0.25)
_C_FRAC = 0.90        # inner-disc radial-band outer bound as a fraction of r_dom
_WALL_WEIGHT = 1.0
_DISC_WEIGHT = 0.25

# --- World-scale anchor ---
# Geometry anchors to config.scale. This fixed length was measured from the early
# prototype's Bezier setup; keeping the numeric anchor avoids coupling layouts to
# a different generator or to batch-dependent host measurements. Absolute road
# width is reserved separately, so Repulsive's size also affects raw feasibility.
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
    write_discs: int, disc_centers: wp.array(dtype=wp.vec2f),
    disc_radii: wp.array(dtype=wp.float32), use_margin: int,
    margin: wp.array(dtype=wp.float32),
):
    # One thread per env. Writes a domain-wall ring (weight 1.0) plus k inner-disc rings
    # (weight 0.25) at angular-stratified positions with an analytic radial band -- no
    # rejection loop. Unused / skipped disc columns are NaN-padded with weight 0 so the
    # obstacle-energy kernel (which guards on ``mw != 0``) never reads them.
    e = wp.tid()
    state = _rng.init(seeds[e], wp.uint32(_OBSTACLE_SALT))
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
            if use_margin == 1:
                lo += margin[e]
            hi = c_frac * r_dom
            rad = lo + (hi - lo) * wp.randf(state)
            active = float(1.0)
            if lo >= hi:
                active = float(0.0)
            ang = phase + float(j) * two_pi / float(k)
            cx = rad * wp.cos(ang)
            cy = rad * wp.sin(ang)
            disc_mw = (two_pi * r_disc / float(n_disc)) * _DISC_WEIGHT
            if write_discs == 1:
                disc_centers[e * k_max + j] = wp.vec2f(cx, cy)
                disc_radii[e * k_max + j] = wp.where(active == 1.0, r_disc, 0.0)
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
            if write_discs == 1:
                disc_centers[e * k_max + j] = wp.vec2f(0.0, 0.0)
                disc_radii[e * k_max + j] = 0.0
            for idx in range(n_disc):
                obs_pts[col + idx] = wp.vec2f(nan_val, nan_val)
                obs_mw[col + idx] = 0.0


def _sample_obstacles_inplace(seeds_wp: wp.array, config,
                              obs_pts: wp.array, obs_mw: wp.array,
                              device: str, ctx: "LaunchContext | None" = None, *,
                              disc_centers=None, disc_radii=None, margin=None) -> None:
    """Launch the seed-driven obstacle kernel in place. obs_pts/obs_mw are [E*M].

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, device)
    s = _RepulsiveScalars(config)
    ctx.launch(_sample_obstacles_k, s.E,
               [seeds_wp, s.r_dom, s.r_init, s.r_frac_lo, s.r_frac_hi, s.c_frac,
                s.k_min, s.k_max, s.n_wall, s.n_disc, s.M, float("nan"),
                obs_pts, obs_mw, int(disc_centers is not None),
                obs_pts if disc_centers is None else disc_centers,
                obs_mw if disc_radii is None else disc_radii, int(margin is not None),
                obs_mw if margin is None else margin])
    # No host sync here: the growth-loop launches that consume obs_pts/obs_mw are already
    # stream-ordered after this kernel. (Callers that read obs back on the host, e.g.
    # tests, go through wp.array.numpy(), which synchronizes on its own.)


# RNG salt for the per-env target-length draw (distinct from the obstacle salt).
_GROW_SALT = 0xCD2AA706
# Coarse-to-fine upsample trigger: subdivide N -> 2N when the mean edge has grown by this
# factor (the reference's edge-length-doubling subdivision rule). Spike default 2.0.
_SUBDIV_RATIO = 2.0


@wp.kernel
def _seed_lfinal_k(
    seeds: wp.array(dtype=wp.int32),
    r_init: float, grow_lo: float, grow_hi: float,
    L_final: wp.array(dtype=wp.float64), L_init: wp.array(dtype=wp.float64),
    L_target: wp.array(dtype=wp.float64), reached: wp.array(dtype=wp.int32),
    frozen: wp.array(dtype=wp.int32), area_prev: wp.array(dtype=wp.float64),
):
    # One thread per env. Draw the per-env target-perimeter multiple and seed all per-env
    # growth state. L_init = L_target = init-circle perimeter; L_final = grow_mult * L_init.
    e = wp.tid()
    state = _rng.init(seeds[e], wp.uint32(_GROW_SALT))
    grow_mult = grow_lo + (grow_hi - grow_lo) * wp.randf(state)
    l0 = wp.float64(2.0) * wp.float64(wp.pi) * wp.float64(r_init)
    L_init[e] = l0
    L_target[e] = l0
    L_final[e] = wp.float64(grow_mult) * l0
    reached[e] = 0
    frozen[e] = 0
    area_prev[e] = wp.float64(0.0)


@wp.kernel
def _init_circle_k(center: wp.array(dtype=wp.vec2f), n: int, r_init: float):
    # One thread per (env, point). Writes the radius-r_init circle into the coarsest-stage
    # prefix (stride n). Uniform across envs; growth breaks the symmetry.
    e, i = wp.tid()
    ang = float(i) * (2.0 * wp.pi / float(n))
    center[e * n + i] = wp.vec2f(r_init * wp.cos(ang), r_init * wp.sin(ang))


# ===========================================================================
# Complete analytic gradients: edge TP endpoint gathers plus weighted obstacle
# integrals. The isolated helpers have independent float64/finite-difference tests.
# Every vertex sums its incident-edge contributions locally, with no atomics.
# ===========================================================================

@wp.kernel
def _dual_weight_k(center: wp.array(dtype=wp.vec2f), n: int,
                    frozen: wp.array(dtype=wp.int32), wdual: wp.array(dtype=wp.float32)):
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    wdual[b + v] = 0.5 * (wp.length(center[b + (v + 1) % n] - center[b + v])
                          + wp.length(center[b + v] - center[b + (v + n - 1) % n]))


@wp.kernel
def _perimeter_k(center: wp.array(dtype=wp.vec2f), n: int,
                  peri_out: wp.array(dtype=wp.float64)):
    e = wp.tid()
    peri = wp.float64(0.0)
    for i in range(n):
        peri += wp.length(wp.vec2d(center[e * n + (i + 1) % n]) - wp.vec2d(center[e * n + i]))
    peri_out[e] = peri


@wp.kernel
def _obstacle_energy_k(center: wp.array(dtype=wp.vec2f), n: int,
                       obs_pts: wp.array(dtype=wp.vec2f),
                       obs_mw: wp.array(dtype=wp.float32), m_obs: int, p_exp: float,
                       disc_centers: wp.array(dtype=wp.vec2f),
                       disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                       margin: wp.array(dtype=wp.float32),
                       frozen: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32),
                       barrier_grad: wp.array(dtype=wp.vec2f),
                       edge_barrier_grad: wp.array(dtype=wp.vec2f)):
    # The obstacle integral uses curve dual lengths. Cache each vertex's
    # unweighted potential for the weight derivative in the gather.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    potential = float(0.0)
    for m in range(m_obs):
        mw = obs_mw[e * m_obs + m]
        if mw != 0.0:
            d = center[e * n + i] - obs_pts[e * m_obs + m]
            potential += mw * wp.pow(wp.dot(d, d) + 1.0e-8, p_exp)
    barrier, bg = barrier_potential_gradient(
        center[e * n + i], disc_centers, disc_radii, e * k, 0, r_dom, margin[e], -2.0 * p_exp)
    out[e * n + i] = potential + barrier
    barrier_grad[e * n + i] = bg
    xi = center[e * n + i]
    xn = center[e * n + (i + 1) % n]
    xp = center[e * n + (i + n - 1) % n]
    forward, gf, unused_f = segment_barrier_potential_gradient(
        xi, xn, disc_centers, disc_radii, e * k, k, r_dom, margin[e], -2.0 * p_exp)
    backward, unused_b, gb = segment_barrier_potential_gradient(
        xp, xi, disc_centers, disc_radii, e * k, k, r_dom, margin[e], -2.0 * p_exp)
    # Differentiate both endpoint gaps and edge-length quadrature weights.
    edge_barrier_grad[e * n + i] = (
        wp.length(xn - xi) * gf - forward * funcs.safe_normalize2(xn - xi)
        + wp.length(xi - xp) * gb + backward * funcs.safe_normalize2(xi - xp))


@wp.kernel
def _grad_gather_k(
    center: wp.array(dtype=wp.vec2f), n: int,
    alpha: float, beta: float, eps: float,
    wdual: wp.array(dtype=wp.float32),
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32), m_obs: int,
    p_exp: float, obstacle_energy: wp.array(dtype=wp.float32),
    barrier_grad: wp.array(dtype=wp.vec2f), edge_barrier_grad: wp.array(dtype=wp.vec2f),
    frozen: wp.array(dtype=wp.int32),
    grad: wp.array(dtype=wp.vec2f),
):
    # Gather the two incident edges as owner and partner of every ordered
    # disjoint edge pair. The local helper includes endpoint, tangent and both
    # edge-length derivatives; no atomics or cancellation-prone point tangent.
    e, v = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    xv = center[b + v]
    xvn = center[b + (v + 1) % n]
    xvp = center[b + (v + n - 1) % n]
    wv = wdual[b + v]
    g = wp.vec2f(0.0, 0.0)
    for incident in range(2):
        i = v
        if incident == 0:
            i = (v + n - 1) % n
        a = center[b + i]
        z = center[b + (i + 1) % n]
        for j in range(n):
            distance = wp.abs(i - j)
            if wp.min(distance, n - distance) > 1:
                c = center[b + j]
                d = center[b + (j + 1) % n]
                value, ga, gb, gc, gd = directed_edge_pair_gradient(a, z, c, d, alpha, beta, eps)
                reverse, ha, hb, hc, hd = directed_edge_pair_gradient(c, d, a, z, alpha, beta, eps)
                if incident == 0:
                    g += gb + hd
                else:
                    g += ga + hc
    def_dir = funcs.safe_normalize2(xvn - xv)
    deb_dir = funcs.safe_normalize2(xv - xvp)

    # --- obstacle direct derivative (curve dual length times potential gradient) ---
    ob = e * m_obs
    for m in range(m_obs):
        mw = obs_mw[ob + m]
        if mw != 0.0:
            d = xv - obs_pts[ob + m]
            d2 = wp.dot(d, d)
            g += (wv * mw * p_exp * wp.pow(d2 + 1.0e-8, p_exp - 1.0) * 2.0) * d
    g += wv * barrier_grad[b + v] + edge_barrier_grad[b + v]

    # --- obstacle dual-weight derivative (the curve-length quadrature) ---
    ov = obstacle_energy[b + v]
    op = obstacle_energy[b + (v + n - 1) % n]
    on = obstacle_energy[b + (v + 1) % n]
    g += 0.5 * (op + ov) * deb_dir - 0.5 * (ov + on) * def_dir

    grad[b + v] = g


# ===========================================================================
# Optimizer-step kernels (consume the gradient; no adjoint needed)
# ===========================================================================

@wp.kernel
def _ratchet_k(
    L_target: wp.array(dtype=wp.float64), L_final: wp.array(dtype=wp.float64),
    growth: float, reached: wp.array(dtype=wp.int32),
):
    # L_target = min(L_target*(1+growth), L_final); reached once at final length.
    e = wp.tid()
    lt = wp.min(L_target[e] * (wp.float64(1.0) + wp.float64(growth)), L_final[e])
    L_target[e] = lt
    reached[e] = wp.where(lt >= L_final[e] - wp.float64(1.0e-9), int(1), int(0))


@wp.kernel
def _conv_k(gin: wp.array(dtype=wp.vec2f), h: wp.array(dtype=wp.float64), n: int,
            frozen: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2d)):
    # Circular convolution out[i] = sum_j h[(i-j) mod n] * gin[j] -- the FFT-free
    # fractional-Sobolev preconditioner A^{-1}. h is the fixed circulant row (numpy irfft
    # of the spectral filter). O(N^2); frozen envs skip it (stale out never used).
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    acc = wp.vec2d(0.0, 0.0)
    reference = wp.vec2d(gin[b + i])
    for j in range(n):
        mmod = (i - j) % n
        if mmod < 0:
            mmod += n
        acc += h[mmod] * (wp.vec2d(gin[b + j]) - reference)
    out[b + i] = acc


@wp.kernel
def _length_grad_k(center: wp.array(dtype=wp.vec2f), n: int, lg: wp.array(dtype=wp.vec2f)):
    # relaxation._length_grad: lg[i] = -u_fwd[i] + u_fwd[i-1], u_fwd[i]=dir(x[i+1]-x[i]).
    e, i = wp.tid()
    b = e * n
    ef = wp.vec2d(center[b + (i + 1) % n]) - wp.vec2d(center[b + i])
    u_i = ef / wp.max(wp.length(ef), wp.float64(1.0e-8))
    eb = wp.vec2d(center[b + i]) - wp.vec2d(center[b + (i + n - 1) % n])
    u_p = eb / wp.max(wp.length(eb), wp.float64(1.0e-8))
    lg[b + i] = wp.vec2f(-u_i + u_p)


@wp.kernel
def _numden_k(g: wp.array(dtype=wp.vec2d), lg: wp.array(dtype=wp.vec2f),
              ainv_lg: wp.array(dtype=wp.vec2d), n: int,
              num: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64)):
    # Per-env inner products for the Sobolev-orthogonal projection:
    #   num = <g, lg>,  den = <lg, A^{-1} lg> (clamped).
    e = wp.tid()
    b = e * n
    sn = wp.float64(0.0)
    sd = wp.float64(0.0)
    for i in range(n):
        sn += wp.dot(wp.vec2d(g[b + i]), wp.vec2d(lg[b + i]))
        sd += wp.dot(wp.vec2d(lg[b + i]), wp.vec2d(ainv_lg[b + i]))
    num[e] = sn
    den[e] = wp.max(sd, wp.float64(1.0e-12))


@wp.kernel
def _project_k(g: wp.array(dtype=wp.vec2d), ainv_lg: wp.array(dtype=wp.vec2d), n: int,
               num: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64)):
    # g <- g - (num/den) * A^{-1} lg  (project out the length-increase direction).
    e, i = wp.tid()
    t = e * n + i
    g[t] = g[t] - (num[e] / den[e]) * ainv_lg[t]


@wp.kernel
def _gmean_k(g: wp.array(dtype=wp.vec2d), n: int, gmean: wp.array(dtype=wp.vec2d)):
    # Per-env barycenter of g (the mean subtracted next -> barycenter pin).
    e = wp.tid()
    b = e * n
    acc = wp.vec2d(0.0, 0.0)
    for i in range(n):
        acc += wp.vec2d(g[b + i])
    gmean[e] = acc / wp.float64(n)


@wp.func
def _flow_factor(n: int, r_dom: float, tau: float, energy_scale: wp.float64,
                 gmax: wp.float64) -> wp.float64:
    return wp.float64(tau) * wp.min(
        wp.float64(n) * wp.float64(r_dom) * wp.float64(r_dom) / wp.max(energy_scale, wp.float64(1.0e-12)),
        wp.float64(r_dom) / wp.max(gmax, wp.float64(1.0e-12)))


@wp.kernel
def _direction_metrics_k(g: wp.array(dtype=wp.vec2d), gmean: wp.array(dtype=wp.vec2d),
                         grad: wp.array(dtype=wp.vec2f), peri: wp.array(dtype=wp.float64),
                         energy_scale: wp.array(dtype=wp.float64), n: int,
                         tau: float, r_dom: float, frozen: wp.array(dtype=wp.int32),
                         gmax: wp.array(dtype=wp.float32), msl: wp.array(dtype=wp.float32),
                         stationarity: wp.array(dtype=wp.float32),
                         step_slope: wp.array(dtype=wp.float64),
                         predicted_move: wp.array(dtype=wp.float64),
                         numerical_stationary: wp.array(dtype=wp.int32),
                         metric_failed: wp.array(dtype=wp.int32),
                         lg: wp.array(dtype=wp.vec2f),
                         num: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64),
                         h: wp.array(dtype=wp.float64)):
    # Measure the actual projected metric field BEFORE step scaling or vec2f
    # coordinate quantization. sqrt(N R^2 <grad,H_c grad>)/E is dimensionless,
    # independent of tau, search fraction and point density. The field norm
    # guards cancellation in the quadratic form (the inverse metric norm <=1).
    e = wp.tid()
    if frozen[e] == 1:
        return
    coefficient = num[e] / den[e]
    residual_mean = wp.vec2d(0.0, 0.0)
    length_squared = wp.float64(0.0)
    row_l1 = wp.float64(0.0)
    for i in range(n):
        length_gradient = wp.vec2d(lg[e * n + i])
        residual_mean += wp.vec2d(grad[e * n + i]) - coefficient * length_gradient
        length_squared += wp.dot(length_gradient, length_gradient)
        row_l1 += wp.abs(h[i])
    residual_mean /= wp.float64(n)
    residual_squared = wp.float64(0.0)
    q = wp.float64(0.0)
    absolute = wp.float64(0.0)
    field_squared = wp.float64(0.0)
    gradient_squared = wp.float64(0.0)
    maximum = wp.float64(0.0)
    mean = wp.vec2d(gmean[e])
    for i in range(n):
        field = wp.vec2d(g[e * n + i]) - mean
        gradient = wp.vec2d(grad[e * n + i])
        residual = gradient - coefficient * wp.vec2d(lg[e * n + i]) - residual_mean
        q += wp.dot(residual, field)
        absolute += wp.abs(residual[0] * field[0]) + wp.abs(residual[1] * field[1])
        residual_squared += wp.dot(residual, residual)
        field_squared += wp.dot(field, field)
        gradient_squared += wp.dot(gradient, gradient)
        maximum = wp.max(maximum, wp.length(field))
    # r=P0(grad-lambda*lg), p=P0(Hgrad-lambda*Hlg), so q=r.H.r>=0.
    # Bound roundoff using PRE-projection norms, including the convolution and
    # mean subtraction. Difference-form convolution has at most 3N operations
    # per summand; Young's inequality supplies its row-L1 operator bound.
    unit_roundoff = wp.float64(1.1102230246251565e-16)
    gamma_n = wp.float64(n) * unit_roundoff / (wp.float64(1.0) - wp.float64(n) * unit_roundoff)
    gamma_2n = wp.float64(2 * n) * unit_roundoff / (wp.float64(1.0) - wp.float64(2 * n) * unit_roundoff)
    gamma_3n = wp.float64(3 * n) * unit_roundoff / (wp.float64(1.0) - wp.float64(3 * n) * unit_roundoff)
    input_norm = wp.sqrt(gradient_squared) + wp.abs(coefficient) * wp.sqrt(length_squared)
    field_error = (wp.float64(4.0) * gamma_3n + wp.float64(4.0) * gamma_n
                   + wp.float64(8.0) * unit_roundoff) * row_l1 * input_norm
    residual_error = (wp.float64(4.0) * unit_roundoff + wp.float64(2.0) * gamma_n) * input_norm
    q_roundoff = (gamma_2n * absolute + wp.sqrt(residual_squared) * field_error
                  + residual_error * (wp.sqrt(field_squared) + field_error))
    relative_roundoff = wp.float64(64.0 * 1.1920928955078125e-7)
    scale = energy_scale[e]
    failed = (not wp.isfinite(q) or not wp.isfinite(field_squared)
              or not wp.isfinite(gradient_squared) or not wp.isfinite(scale)
              or scale <= wp.float64(0.0) or q < -q_roundoff)
    gmax[e] = wp.float32(wp.max(maximum, wp.float64(1.0e-12)))
    msl[e] = wp.float32(peri[e] / wp.float64(n))
    numerical_stationary[e] = 0
    if failed or metric_failed[e] != 0:
        metric_failed[e] = 1
        stationarity[e] = wp.inf
        step_slope[e] = wp.float64(0.0)
        predicted_move[e] = wp.float64(wp.inf)
        return
    radius_squared = wp.float64(r_dom) * wp.float64(r_dom)
    measured = wp.sqrt(wp.float64(n) * radius_squared * wp.max(q, field_squared)) / scale
    if not wp.isfinite(measured) or measured > wp.float64(3.4028234663852886e38):
        metric_failed[e] = 1
        stationarity[e] = wp.inf
        step_slope[e] = wp.float64(0.0)
        predicted_move[e] = wp.float64(wp.inf)
        return
    stationarity[e] = wp.float32(measured)
    factor = _flow_factor(n, r_dom, tau, scale, maximum)
    step_slope[e] = -factor * wp.max(q, wp.float64(0.0))
    predicted_move[e] = factor * maximum
    # A numerical stationary result is distinct from accepted progress. It is
    # permitted only when the genuine projected field is at its arithmetic
    # resolution AND its unrounded update is below coordinate storage precision.
    if (wp.abs(q) <= q_roundoff
            and field_squared <= relative_roundoff * relative_roundoff * gradient_squared
            and predicted_move[e] <= wp.float64(8.0 * 1.1920928955078125e-7) * wp.float64(r_dom)):
        numerical_stationary[e] = 1


@wp.func
def _target_length_ok(perimeter: wp.float64, target: wp.float64, n: int, r_dom: float) -> bool:
    # A float32 coordinate perturbation changes each edge length by O(eps*R).
    # This bound covers storage/retraction rounding, not a percent target shortfall.
    tolerance = wp.float64(16.0 * 1.1920928955078125e-7) * wp.float64(n) * wp.float64(r_dom)
    return (wp.isfinite(perimeter) and wp.isfinite(target) and target > wp.float64(0.0)
            and wp.abs(perimeter - target) <= tolerance)


@wp.kernel
def _step_k(center: wp.array(dtype=wp.vec2f), g: wp.array(dtype=wp.vec2d),
            gmean: wp.array(dtype=wp.vec2d), energy: wp.array(dtype=wp.float64),
            gmax: wp.array(dtype=wp.float32), tau: float, r_dom: float, n: int,
            frozen: wp.array(dtype=wp.int32), proposal: wp.array(dtype=wp.vec2f)):
    # The length-weighted energy has per-vertex gradient O(1/n). Normalize
    # its continuum density by E/R^2, retaining gradient magnitude so a
    # stationary curve stops moving. The trust cap is only an upper bound.
    e, i = wp.tid()
    if frozen[e] == 1:
        return
    t = e * n + i
    factor = _flow_factor(n, r_dom, tau, energy[e], wp.float64(gmax[e]))
    proposal[t] = wp.vec2f(wp.vec2d(center[t]) - factor * (wp.vec2d(g[t]) - wp.vec2d(gmean[e])))


@wp.kernel
def _freeze_update_k(center: wp.array(dtype=wp.vec2f), n: int,
                     reached: wp.array(dtype=wp.int32), thresh: float,
                     area_prev: wp.array(dtype=wp.float64),
                     frozen: wp.array(dtype=wp.int32), md_out: wp.array(dtype=wp.float64),
                     final_target: wp.array(dtype=wp.float64),
                     window_move: wp.array(dtype=wp.float32),
                     stationarity: wp.array(dtype=wp.float32),
                     blocked: wp.array(dtype=wp.int32), r_dom: float):
    # Per-env stall detector (once per stall window). An env freezes when it has reached its
    # actual final length with small area change, movement and projected stationarity.
    # A blocked search is not convergence.
    e = wp.tid()
    b = e * n
    a2 = wp.float64(0.0)
    peri = wp.float64(0.0)
    for i in range(n):
        p0 = wp.vec2d(center[b + i])
        p1 = wp.vec2d(center[b + (i + 1) % n])
        a2 += p0[0] * p1[1] - p1[0] * p0[1]
        peri += wp.length(p1 - p0)
    area = wp.abs(a2) * wp.float64(0.5)
    rel = wp.abs(area - area_prev[e]) / wp.max(area, wp.float64(1.0e-9))
    md_out[e] = rel
    area_prev[e] = area
    if (reached[e] == 1 and frozen[e] == 0 and rel < wp.float64(thresh)
            and _target_length_ok(peri, final_target[e], n, r_dom) and window_move[e] < thresh
            and stationarity[e] < thresh and blocked[e] == 0):
        frozen[e] = 1
    window_move[e] = 0.0


@wp.kernel
def _active_k(frozen: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32)):
    # active[0] = any env still growing (active was zeroed just before): the bounded
    # loop's flag, refreshed by every stall check (writes only `active`).
    e = wp.tid()
    if frozen[e] == 0:
        wp.atomic_max(active, 0, 1)


@wp.func
def _proposal_point(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                    t: int, fraction: float):
    if fraction == 0.0:
        return center[t]
    if fraction == 1.0:
        return proposal[t]
    a = wp.vec2d(center[t])
    return wp.vec2f(a + wp.float64(fraction) * (wp.vec2d(proposal[t]) - a))


@wp.func
def _proposal_clear(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                    e: int, n: int, fraction: float,
                    disc_centers: wp.array(dtype=wp.vec2f),
                    disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                    margin: float, swept: int):
    limit = wp.float64(r_dom) - wp.float64(margin) - _clearance_tolerance(r_dom)
    if not wp.isfinite(limit) or limit <= wp.float64(0.0):
        return False
    limit_squared = limit * limit
    clear = bool(True)
    b = e * n
    for i in range(n):
        a = _proposal_point(center, proposal, b + i, fraction)
        z = _proposal_point(center, proposal, b + (i + 1) % n, fraction)
        # Cast before the dot product: float32 norms can round an outside
        # point below the wall at large coordinates. The domain is convex.
        ad = wp.vec2d(a)
        distance_squared = wp.dot(ad, ad)
        if not wp.isfinite(distance_squared) or distance_squared >= limit_squared:
            clear = False
        for j in range(k):
            radius = disc_radii[e * k + j]
            if radius > 0.0:
                c = disc_centers[e * k + j]
                if not segment_disc_clear(a, z, c, radius + margin):
                    clear = False
                if swept == 1 and not swept_segment_disc_clear(
                        center[b + i], center[b + (i + 1) % n], a, z, c, radius + margin):
                    clear = False
    return clear


@wp.func
def _proposal_energy_parts(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                     e: int, n: int, fraction: float, alpha: float, beta: float,
                     eps: float, obs_pts: wp.array(dtype=wp.vec2f),
                     obs_mw: wp.array(dtype=wp.float32), m_obs: int,
                     p_exp: float, disc_centers: wp.array(dtype=wp.vec2f),
                     disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                     margin: float):
    # Exactly the discrete objective differentiated by the gather; deterministic
    # reductions and no atomics. The projected-out length penalty is omitted.
    tangent_energy = wp.float64(0.0)
    for i in range(n):
        a = _proposal_point(center, proposal, e * n + i, fraction)
        z = _proposal_point(center, proposal, e * n + (i + 1) % n, fraction)
        for j in range(n):
            distance = wp.abs(i - j)
            if wp.min(distance, n - distance) > 1:
                c = _proposal_point(center, proposal, e * n + j, fraction)
                d = _proposal_point(center, proposal, e * n + (j + 1) % n, fraction)
                tangent_energy += wp.float64(directed_edge_pair_energy(a, z, c, d, alpha, beta, eps))
    total = tangent_energy
    magnitude = tangent_energy
    b = e * n
    for i in range(n):
        xi = _proposal_point(center, proposal, b + i, fraction)
        xn = _proposal_point(center, proposal, b + (i + 1) % n, fraction)
        xp = _proposal_point(center, proposal, b + (i + n - 1) % n, fraction)
        wi = 0.5 * (wp.length(xn - xi) + wp.length(xi - xp))
        for j in range(m_obs):
            mw = obs_mw[e * m_obs + j]
            if mw != 0.0:
                d = xi - obs_pts[e * m_obs + j]
                value = wp.float64(wi) * wp.float64(mw) * wp.float64(wp.pow(wp.dot(d, d) + 1.0e-8, p_exp))
                total += value
                magnitude += value
        barrier, unused_grad = barrier_potential_gradient(
            xi, disc_centers, disc_radii, e * k, 0, r_dom, margin, -2.0 * p_exp)
        edge_barrier, unused_a, unused_b = segment_barrier_potential_gradient(
            xi, xn, disc_centers, disc_radii, e * k, k, r_dom, margin, -2.0 * p_exp)
        edge_length = wp.length(xn - xi)
        total += wp.float64(wi) * wp.float64(barrier) + wp.float64(edge_length) * wp.float64(edge_barrier)
        magnitude += wp.float64(wi) * wp.float64(wp.abs(barrier)) + wp.float64(edge_length) * wp.float64(wp.abs(edge_barrier))
    return total, magnitude


@wp.func
def _proposal_energy(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                     e: int, n: int, fraction: float, alpha: float, beta: float,
                     eps: float, obs_pts: wp.array(dtype=wp.vec2f),
                     obs_mw: wp.array(dtype=wp.float32), m_obs: int,
                     p_exp: float, disc_centers: wp.array(dtype=wp.vec2f),
                     disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                     margin: float):
    total, unused_tangent = _proposal_energy_parts(
        center, proposal, e, n, fraction, alpha, beta, eps, obs_pts, obs_mw, m_obs,
        p_exp, disc_centers, disc_radii, k, r_dom, margin)
    return total


@wp.kernel
def _shape_energy_k(center: wp.array(dtype=wp.vec2f), n: int, alpha: float, beta: float,
                      eps: float, obs_pts: wp.array(dtype=wp.vec2f),
                      obs_mw: wp.array(dtype=wp.float32), m_obs: int, p_exp: float,
                      disc_centers: wp.array(dtype=wp.vec2f),
                      disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                      margin: wp.array(dtype=wp.float32), frozen: wp.array(dtype=wp.int32),
                      out: wp.array(dtype=wp.float64), scale_out: wp.array(dtype=wp.float64)):
    e = wp.tid()
    if frozen[e] == 0:
        total, magnitude = _proposal_energy_parts(center, center, e, n, 0.0, alpha, beta, eps, obs_pts,
                                                obs_mw, m_obs, p_exp, disc_centers, disc_radii, k, r_dom, margin[e])
        out[e] = total
        scale_out[e] = wp.max(magnitude, wp.float64(wp.pow(r_dom, 2.0 + 2.0 * p_exp)))


@wp.kernel
def _accept_shape_k(
    center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
    trial_points: wp.array(dtype=wp.vec2f),
    grad: wp.array(dtype=wp.vec2f), energy: wp.array(dtype=wp.float64),
    energy_scale: wp.array(dtype=wp.float64), n: int, alpha: float, beta: float, eps: float,
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32), m_obs: int,
    p_exp: float,
    disc_centers: wp.array(dtype=wp.vec2f), disc_radii: wp.array(dtype=wp.float32),
    k: int, r_dom: float, margin: wp.array(dtype=wp.float32), msl: wp.array(dtype=wp.float32),
    frozen: wp.array(dtype=wp.int32), window_move: wp.array(dtype=wp.float32),
    stationarity: wp.array(dtype=wp.float32), blocked: wp.array(dtype=wp.int32),
    step_slope: wp.array(dtype=wp.float64), numerical_stationary: wp.array(dtype=wp.int32),
    metric_failed: wp.array(dtype=wp.int32), accepted_out: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    accepted_out[e] = 0
    if metric_failed[e] != 0:
        blocked[e] = 1
        return
    if numerical_stationary[e] == 1:
        blocked[e] = wp.int32(not _proposal_clear(
            center, center, e, n, 0.0, disc_centers, disc_radii, k, r_dom, margin[e], 0))
        return
    before = energy[e]
    if step_slope[e] >= wp.float64(0.0):
        blocked[e] = 1
        return
    peri_before = wp.float64(0.0)
    for i in range(n):
        peri_before += wp.length(wp.vec2d(center[b + (i + 1) % n]) - wp.vec2d(center[b + i]))
    fraction = float(1.0)
    accepted = int(0)
    for trial in range(24):
        # Recompute the nonlinear perimeter projection at EVERY trial.
        # Interpolating an already-projected endpoint need not be descent.
        bc = wp.vec2d(0.0, 0.0)
        peri = wp.float64(0.0)
        for i in range(n):
            xi = _proposal_point(center, proposal, b + i, fraction)
            xn = _proposal_point(center, proposal, b + (i + 1) % n, fraction)
            bc += wp.vec2d(xi)
            peri += wp.length(wp.vec2d(xn) - wp.vec2d(xi))
        bc /= wp.float64(n)
        scale = peri_before / wp.max(peri, wp.float64(1.0e-9))
        actual_move = float(0.0)
        for i in range(n):
            xi = _proposal_point(center, proposal, b + i, fraction)
            trial_points[b + i] = wp.vec2f(wp.vec2d(xi) + (scale - wp.float64(1.0)) * (wp.vec2d(xi) - bc))
            actual_move = wp.max(actual_move, wp.length(trial_points[b + i] - center[b + i]))
        if actual_move == 0.0:
            break
        if _proposal_clear(center, trial_points, e, n, 1.0, disc_centers,
                           disc_radii, k, r_dom, margin[e], 1):
            after = _proposal_energy(center, trial_points, e, n, 1.0, alpha, beta, eps,
                                     obs_pts, obs_mw, m_obs, p_exp, disc_centers, disc_radii, k, r_dom, margin[e])
            if (wp.isfinite(after) and after < before
                    and after <= before + wp.float64(1.0e-4) * wp.float64(fraction) * step_slope[e]):
                accepted = 1
                break
        fraction *= 0.5
    blocked[e] = wp.int32(accepted == 0)
    accepted_out[e] = accepted
    movement = float(0.0)
    if accepted == 1:
        for i in range(n):
            movement = wp.max(movement, wp.length(trial_points[b + i] - center[b + i]))
            center[b + i] = trial_points[b + i]
    window_move[e] = wp.max(window_move[e], movement / wp.max(msl[e], 1.0e-9))


@wp.kernel
def _grow_feasible_k(
    center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f), n: int,
    target: wp.array(dtype=wp.float64), growth: float, disc_centers: wp.array(dtype=wp.vec2f),
    disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
    margin: wp.array(dtype=wp.float32), frozen: wp.array(dtype=wp.int32),
):
    # Growth is a separate forced constraint operation: energy may rise with
    # length, but every accepted partial expansion remains analytically feasible.
    e = wp.tid()
    if frozen[e] == 1:
        return
    b = e * n
    bc = wp.vec2d(0.0, 0.0)
    peri = wp.float64(0.0)
    for i in range(n):
        bc += wp.vec2d(center[b + i])
        peri += wp.length(wp.vec2d(center[b + (i + 1) % n]) - wp.vec2d(center[b + i]))
    bc /= wp.float64(n)
    scale = wp.min(target[e] / wp.max(peri, wp.float64(1.0e-9)), wp.float64(1.0) + wp.float64(growth))
    for i in range(n):
        proposal[b + i] = wp.vec2f(wp.vec2d(center[b + i]) + (scale - wp.float64(1.0)) * (wp.vec2d(center[b + i]) - bc))
    fraction = float(1.0)
    accepted = float(0.0)
    for trial in range(16):
        if _proposal_clear(center, proposal, e, n, fraction, disc_centers,
                           disc_radii, k, r_dom, margin[e], 1):
            accepted = fraction
            break
        fraction *= 0.5
    if accepted > 0.0:
        for i in range(n):
            center[b + i] = _proposal_point(center, proposal, b + i, accepted)


@wp.kernel
def _accept_resample_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                        n: int, disc_centers: wp.array(dtype=wp.vec2f),
                        disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                        margin: wp.array(dtype=wp.float32), frozen: wp.array(dtype=wp.int32),
                        window_move: wp.array(dtype=wp.float32), skip_frozen: int,
                        alpha: float, beta: float, eps: float,
                        obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32),
                        m_obs: int, p_exp: float):
    e = wp.tid()
    if skip_frozen == 1 and frozen[e] == 1:
        return
    b = e * n
    old_peri = wp.float64(0.0)
    new_peri = wp.float64(0.0)
    bc = wp.vec2d(0.0, 0.0)
    for i in range(n):
        old_peri += wp.length(wp.vec2d(center[b + (i + 1) % n]) - wp.vec2d(center[b + i]))
        new_peri += wp.length(wp.vec2d(proposal[b + (i + 1) % n]) - wp.vec2d(proposal[b + i]))
        bc += wp.vec2d(proposal[b + i])
    bc /= wp.float64(n)
    scale = old_peri / wp.max(new_peri, wp.float64(1.0e-9))
    for i in range(n):
        proposal[b + i] = wp.vec2f(wp.vec2d(proposal[b + i]) + (scale - wp.float64(1.0)) * (wp.vec2d(proposal[b + i]) - bc))
    # Arc interpolation changes chords, not just point labels. Treat it as
    # another shape proposal: same perimeter, full swept barriers, no energy kick.
    if _proposal_clear(center, proposal, e, n, 1.0, disc_centers, disc_radii, k, r_dom, margin[e], 1):
        before = _proposal_energy(center, center, e, n, 0.0, alpha, beta, eps, obs_pts,
                                  obs_mw, m_obs, p_exp, disc_centers, disc_radii, k, r_dom, margin[e])
        after = _proposal_energy(center, proposal, e, n, 1.0, alpha, beta, eps, obs_pts,
                                 obs_mw, m_obs, p_exp, disc_centers, disc_radii, k, r_dom, margin[e])
        if wp.isfinite(after) and after <= before:
            movement = float(0.0)
            for i in range(n):
                movement = wp.max(movement, wp.length(proposal[b + i] - center[b + i]))
                center[b + i] = proposal[b + i]
            window_move[e] = wp.max(window_move[e], wp.float32(wp.float64(movement) / wp.max(old_peri / wp.float64(n), wp.float64(1.0e-9))))


@wp.kernel
def _upsample_safe_k(old: wp.array(dtype=wp.vec2f), n_old: int,
                     out: wp.array(dtype=wp.vec2f), n_new: int):
    # Preserve every old vertex, subdividing each edge. The geometric loop is
    # identical, so refinement cannot cut a corner or change obstacle homotopy.
    e = wp.tid()
    base_count = n_new // n_old
    extra = n_new % n_old
    cursor = int(0)
    for i in range(n_old):
        count = base_count + wp.int32(i < extra)
        a = old[e * n_old + i]
        b = old[e * n_old + (i + 1) % n_old]
        for j in range(count):
            out[e * n_new + cursor] = a + float(j) / float(count) * (b - a)
            cursor += 1


@wp.kernel
def _finish_valid_k(center: wp.array(dtype=wp.vec2f), n: int,
                    disc_centers: wp.array(dtype=wp.vec2f),
                    disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                    margin: wp.array(dtype=wp.float32), final_target: wp.array(dtype=wp.float64),
                    metric_failed: wp.array(dtype=wp.int32), out_valid: wp.array(dtype=wp.int32)):
    e = wp.tid()
    peri = wp.float64(0.0)
    for i in range(n):
        peri += wp.length(wp.vec2d(center[e * n + (i + 1) % n]) - wp.vec2d(center[e * n + i]))
    clear = _proposal_clear(center, center, e, n, 0.0, disc_centers, disc_radii, k, r_dom, margin[e], 0)
    out_valid[e] = wp.int32(clear and metric_failed[e] == 0
                           and _target_length_ok(peri, final_target[e], n, r_dom))


# ===========================================================================
# Host-side spectral filter + stage schedule (numpy only, precomputed once)
# ===========================================================================

def _sobolev_circulant_row(n, s, eps_reg):
    """Positive ring inverse metric, represented and accumulated in float64.

    The Laplacian has differential order two; its power is s=(beta-1)/alpha.
    Normalize by the first nonzero mode and omit the constrained translation
    mode. Construction rejects filters whose double row cannot retain the
    intended positive spectrum; clipping would silently change the operator.
    """
    k = np.arange(n // 2 + 1, dtype=np.float64)
    lam = 4.0 * np.sin(np.pi * k / n) ** 2
    lam /= lam[1]
    inv_filter = 1.0 / (lam ** s + eps_reg)
    inv_filter[0] = 0.0
    row = np.fft.irfft(inv_filter, n=n)
    actual = np.fft.rfft(row).real - math.fsum(row)
    relative = np.max(np.abs(actual[1:] - inv_filter[1:]) / inv_filter[1:])
    if (not np.isfinite(row).all() or not np.all(actual[1:] > 0.0)
            or relative > 1.0e-5):
        raise ValueError(
            f"Repulsive Sobolev metric at resolution {n}, order {s:g} is not "
            "representably positive and accurate in float64; reduce resolution or order")
    return row


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
        "center", "grad", "wdual", "obstacle_energy", "peri_e",
        "obs_pts", "obs_mw", "disc_centers", "disc_radii", "K",
        "g", "lg", "ainv_lg", "rs_out", "barrier_grad", "edge_barrier_grad", "rs_seg", "rs_s",
        "count",
        "L_final", "L_init", "L_target", "reached", "frozen", "area_prev", "md_out",
        "num", "den", "gmax", "msl", "gmean",
        "active", "energy", "energy_scale", "step_slope", "predicted_move",
        "numerical_stationary", "metric_failed", "step_accepted", "margin", "half_width", "proposal", "trial_points", "window_move", "stationarity", "blocked",
    )

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _alloc_repulsive_scratch(config, *, alloc=None):
    """Allocate the repulsive generator's PRIVATE scratch (one alloc per generator).

    Sizes every N-dependent buffer at ``Nmax`` and precomputes the host-side Sobolev rows +
    stage schedule + iteration budget from the config's ratchet/stage bounds (no device
    readback needed to size the loop, since ``n_ratchet`` follows from ``grow_mult_max``).

    Args:
        alloc: ``alloc(name, dtype, shape) -> wp.array`` (zero-initialised) for every
            buffer, named after its :class:`RepulsiveScratch` slot (a Sobolev row:
            ``"h<N>"``); default ``wp.zeros`` on ``config.device``. The Stage passes its
            ``StageBuilder.scratch``.
    """
    _init()
    sc = _RepulsiveScalars(config)
    E = sc.E
    dev = str(config.device)
    if alloc is None:
        def alloc(name, dtype, shape):
            return wp.zeros(shape, dtype=dtype, device=dev)
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

    s_exp = (beta - 1.0) / alpha
    h_by_n = {}
    for Ns in sorted(set(stages)):
        h_np = _sobolev_circulant_row(Ns, s_exp, 1e-3)
        h = alloc(f"h{Ns}", wp.float64, Ns)
        # Construction-time copy on Warp's current stream (where the scratch was zeroed).
        LaunchContext.current(h.device).copy(
            h, wp.array(h_np, dtype=wp.float64, device="cpu"))
        h_by_n[Ns] = h

    bufs = {}

    def buf(name, dtype, n):
        bufs[name] = alloc(name, dtype, n)

    for name in ("center", "grad", "lg", "rs_out",
                 "proposal", "trial_points", "barrier_grad", "edge_barrier_grad"):
        buf(name, wp.vec2f, E * Nmax)
    for name in ("wdual", "rs_seg", "obstacle_energy"):
        buf(name, wp.float32, E * Nmax)
    buf("rs_s", wp.float32, E * (Nmax + 1))
    buf("obs_pts", wp.vec2f, E * M)
    buf("obs_mw", wp.float32, E * M)
    buf("disc_centers", wp.vec2f, E * sc.k_max)
    buf("disc_radii", wp.float32, E * sc.k_max)
    for name in ("gmax", "msl", "window_move", "stationarity", "margin"):
        buf(name, wp.float32, E)
    for name in ("energy", "energy_scale", "peri_e", "L_final", "L_init", "L_target",
                 "area_prev", "md_out", "num", "den", "step_slope", "predicted_move"):
        buf(name, wp.float64, E)
    for name in ("count", "reached", "frozen", "blocked", "numerical_stationary",
                 "metric_failed", "step_accepted"):
        buf(name, wp.int32, E)
    for name in ("g", "ainv_lg"):
        buf(name, wp.vec2d, E * Nmax)
    buf("gmean", wp.vec2d, E)
    buf("active", wp.int32, 1)               # the bounded loop's flag (any env growing)

    return RepulsiveScratch(
        E=E, Nmax=Nmax, M=M, n_wall=sc.n_wall, n_disc=sc.n_disc,
        r_dom=sc.r_dom, r_init=sc.r_init, K=sc.k_max,
        half_width=float(getattr(config, "_repulsive_half_width", 0.0)),
        stages=stages, stage_starts=stage_starts, n_iters=n_iters, h_by_n=h_by_n,
        **bufs,
    )


def _first_check(stage_starts, stall_window: int) -> int:
    """The first window start of the bounded growth loop: ``W * ceil((s_f + 1) / W)``,
    ``s_f`` the start of the final stage, ``W`` the stall window. The iteration before
    every window start is a stall check (it refreshes ``active``), and no window holds a
    stage transition."""
    W = int(stall_window)
    return W * -(-(int(stage_starts[-1]) + 1) // W)


# ===========================================================================
# The sized output copy (roadmap PR13d)
# ===========================================================================

_C_SIZE_LO = wp.constant(wp.float32(SIZE_BOUNDS[0]))
_C_SIZE_HI = wp.constant(wp.float32(SIZE_BOUNDS[1]))


@wp.kernel
def _clearance_margin_k(size: wp.array(dtype=wp.float32), half_width: float,
                         out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    scale = wp.clamp(size[e], _C_SIZE_LO, _C_SIZE_HI)
    out[e] = half_width / scale


@wp.kernel
def _copy_sized_k(src: wp.array(dtype=wp.vec2f), n: int,
                  size_e: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.vec2f)):
    # out = src * size[e] (size clamped to SIZE_BOUNDS by comparisons), and EXACTLY src
    # at size 1 or for a non-finite point (any bit pattern: NaN payloads included). The
    # multiplied operand w is the finite-gated value, never the selected v: no select
    # fold can turn the size-1 copy into a multiply of a NaN (design PR13d section 1.3).
    t = wp.tid()
    e = t // n
    size = size_e[e]
    size = wp.where(size < _C_SIZE_LO, _C_SIZE_LO, size)
    size = wp.where(size > _C_SIZE_HI, _C_SIZE_HI, size)
    v = src[t]
    fin = wp.isfinite(v[0]) and wp.isfinite(v[1])
    w = wp.where(fin, v, wp.vec2f(0.0, 0.0))
    out[t] = wp.where(size == 1.0 or not fin, v, w * size)


# ===========================================================================
# Fixed-schedule coarse-to-fine growth loop (a bounded loop: capture class BOUNDED)
# ===========================================================================



def _generate_repulsive(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                        size, barriers=None) -> None:
    """The repulsive generator's launches on ``ctx`` (the private core of
    the :class:`Repulsive` Stage). The per-env size sets the unscaled road-width
    reservation; the final copy transforms the grown loop into world coordinates."""
    s = scratch
    E = s.E
    dev = str(out_centerline.device)

    alpha = float(config.repulsive_alpha)
    beta = float(config.repulsive_beta)
    tau = float(config.repulsive_tau)
    growth = float(config.repulsive_ratchet_rate)
    resample_every = int(config.repulsive_resample_every)
    stall_window = int(config.repulsive_stall_window)
    stall_tol = float(config.repulsive_stall_area_tol)
    eps = 1e-4 * s.r_dom / (_DOMAIN_SCALE_REF * 0.35)
    p_exp = -(beta - alpha) / 2.0
    M = s.M
    stages = s.stages
    stage_starts = s.stage_starts
    n_iters = s.n_iters

    center = s.center
    # Periodic-resample scratch (seg/s only; the [0:E*Ncur] prefix is used per stage).
    rs_scratch = ops.ArcScratch(None, s.rs_seg, s.rs_s, None)

    # The static schedule: the stages upsampled at the top of each iteration, the stage
    # resolution of each iteration, and the transitions the budget never reached.
    ups: "list[list[int]]" = [[] for _ in range(n_iters)]
    ncur = [0] * n_iters
    idx = 0
    for it in range(n_iters):
        while idx + 1 < len(stages) and it >= stage_starts[idx + 1]:
            idx += 1
            ups[it].append(idx)
        ncur[it] = stages[idx]
    tail_ups = list(range(idx + 1, len(stages)))

    # 1. Seed-driven obstacle layout + per-env target length.
    ctx.launch(_clearance_margin_k, E, [size, s.half_width, s.margin])
    _sample_obstacles_inplace(seeds_wp, config, s.obs_pts, s.obs_mw, dev, ctx=ctx,
                              disc_centers=s.disc_centers, disc_radii=s.disc_radii, margin=s.margin)
    grow_lo = float(config.repulsive_grow_mult_min)
    grow_hi = float(config.repulsive_grow_mult_max)
    ctx.launch(_seed_lfinal_k, E,
               [seeds_wp, s.r_init, grow_lo, grow_hi,
                s.L_final, s.L_init, s.L_target, s.reached, s.frozen, s.area_prev])
    # The loop flag: a check can only ever read a value written in THIS generation.
    ops.fill(ctx, s.active, 1)
    ops.fill(ctx, s.window_move, 0.0)
    ops.fill(ctx, s.blocked, 0)
    ops.fill(ctx, s.metric_failed, 0)
    ops.fill(ctx, s.step_accepted, 0)
    ops.fill(ctx, s.numerical_stationary, 0)

    # 2. Initial circle at the coarsest stage prefix.
    N0 = stages[0]
    ctx.launch(_init_circle_k, (E, N0), [center, N0, s.r_init])

    def upsample(c, i):
        # Advance the resolution while retaining every old vertex and edge.
        Nold, Nnew = stages[i - 1], stages[i]
        c.launch(_upsample_safe_k, E, [center, Nold, s.rs_out, Nnew])
        c.copy(center, s.rs_out, count=E * Nnew)
        ops.fill(c, s.window_move, 1.0)

    def body(c, it):
        # --- coarse-to-fine stage transition (upsample center N -> next N) ---
        for i in ups[it]:
            upsample(c, i)
        Ncur = ncur[it]
        h_wp = s.h_by_n[Ncur]

        # 1. ratchet target
        c.launch(_ratchet_k, E, [s.L_target, s.L_final, growth, s.reached])

        # 2. Analytic endpoint/tangent/length derivatives and weighted obstacle terms.
        #    Per-vertex gather avoids atomics; shared dual weights are O(E*N).
        c.launch(_dual_weight_k, (E, Ncur), [center, Ncur, s.frozen, s.wdual])
        c.launch(_perimeter_k, E, [center, Ncur, s.peri_e])
        c.launch(_obstacle_energy_k, (E, Ncur),
                 [center, Ncur, s.obs_pts, s.obs_mw, M, p_exp, s.disc_centers,
                  s.disc_radii, s.K, s.r_dom, s.margin, s.frozen, s.obstacle_energy, s.barrier_grad, s.edge_barrier_grad])
        c.launch(_grad_gather_k, (E, Ncur),
                 [center, Ncur, alpha, beta, eps, s.wdual,
                  s.obs_pts, s.obs_mw, M, p_exp, s.obstacle_energy, s.barrier_grad, s.edge_barrier_grad, s.frozen, s.grad])

        # 3. Sobolev precondition g = A^{-1} grad
        c.launch(_conv_k, (E, Ncur), [s.grad, h_wp, Ncur, s.frozen, s.g])
        # 4. length-gradient Sobolev-orthogonal projection
        c.launch(_length_grad_k, (E, Ncur), [center, Ncur, s.lg])
        c.launch(_conv_k, (E, Ncur), [s.lg, h_wp, Ncur, s.frozen, s.ainv_lg])
        c.launch(_numden_k, E, [s.g, s.lg, s.ainv_lg, Ncur, s.num, s.den])
        c.launch(_project_k, (E, Ncur), [s.g, s.ainv_lg, Ncur, s.num, s.den])
        # 5. Shape descent at the CURRENT perimeter, with Armijo acceptance and
        # analytic barriers. Growth is a separate feasible forced expansion.
        c.launch(_gmean_k, E, [s.g, Ncur, s.gmean])
        c.launch(_shape_energy_k, E,
                 [center, Ncur, alpha, beta, eps, s.obs_pts, s.obs_mw, M, p_exp,
                  s.disc_centers, s.disc_radii, s.K, s.r_dom, s.margin, s.frozen, s.energy, s.energy_scale])
        c.launch(_direction_metrics_k, E,
                 [s.g, s.gmean, s.grad, s.peri_e, s.energy_scale, Ncur, tau, s.r_dom, s.frozen,
                  s.gmax, s.msl, s.stationarity, s.step_slope, s.predicted_move,
                  s.numerical_stationary, s.metric_failed, s.lg, s.num, s.den, h_wp])
        c.launch(_step_k, (E, Ncur),
                 [center, s.g, s.gmean, s.energy_scale, s.gmax, tau, s.r_dom, Ncur, s.frozen, s.proposal])
        c.launch(_accept_shape_k, E,
                 [center, s.proposal, s.trial_points, s.grad, s.energy, s.energy_scale, Ncur, alpha, beta, eps, s.obs_pts, s.obs_mw,
                  M, p_exp, s.disc_centers, s.disc_radii,
                  s.K, s.r_dom, s.margin, s.msl, s.frozen, s.window_move, s.stationarity, s.blocked,
                  s.step_slope, s.numerical_stationary, s.metric_failed, s.step_accepted])
        c.launch(_grow_feasible_k, E,
                 [center, s.proposal, Ncur, s.L_target, growth, s.disc_centers, s.disc_radii,
                  s.K, s.r_dom, s.margin, s.frozen])

        # 7. periodic arc-length resample (pure Warp). Frozen envs are left untouched
        #    (absorbing freeze); every other kernel above already skips them.
        if (it + 1) % resample_every == 0:
            ops.fill(c, s.count, Ncur)
            ops.resample_uniform(c, center[0:E * Ncur], Ncur, rs_scratch,
                                 s.rs_out[0:E * Ncur], count=s.count)
            c.launch(_accept_resample_k, E, [center, s.rs_out, Ncur, s.disc_centers,
                     s.disc_radii, s.K, s.r_dom, s.margin, s.frozen, s.window_move, 1,
                     alpha, beta, eps, s.obs_pts, s.obs_mw, M, p_exp])

        # 8. stall-stop: freeze converged envs (device-side), then refresh the loop flag.
        if Ncur == stages[-1] and (it + 1) % stall_window == 0:
            c.launch(_freeze_update_k, E,
                     [center, Ncur, s.reached, stall_tol, s.area_prev, s.frozen, s.md_out,
                      s.L_final, s.window_move, s.stationarity, s.blocked, s.r_dom])
            ops.fill(c, s.active, 0)
            c.launch(_active_k, E, [s.frozen, s.active])

    # Windows of stall_window iterations from the first window start at the final stage:
    # the iteration before each window is a stall check (it refreshes `active`), no window
    # holds a stage transition, and once every env is frozen an iteration changes nothing
    # read below (frozen envs return early), so every lowering writes the same bytes.
    ctx.bounded_loop(n_iters, body, active=s.active,
                     first_check=_first_check(stage_starts, stall_window),
                     check_every=stall_window)

    # Defense in depth: guarantee the centerline reaches the final stage resolution
    # (stages[-1] == num_points) even if the iteration budget was too small to trigger every
    # coarse-to-fine transition on schedule. Otherwise the tail resample/copy below would pack
    # stride-Ncur data into the stride-Nmax out_centerline (silent mis-strided garbage). Config
    # validation already forbids n_iters<=0; this keeps the stride invariant unconditional.
    for i in tail_ups:
        upsample(ctx, i)
    Ncur = stages[-1]

    # final periodic resample at Nmax, then hand the closed loop to the standard tail.
    ops.fill(ctx, s.count, Ncur)
    ops.resample_uniform(ctx, center[0:E * Ncur], Ncur, rs_scratch, s.rs_out[0:E * Ncur],
                         count=s.count)
    ctx.launch(_accept_resample_k, E, [center, s.rs_out, Ncur, s.disc_centers,
               s.disc_radii, s.K, s.r_dom, s.margin, s.frozen, s.window_move, 0,
               alpha, beta, eps, s.obs_pts, s.obs_mw, M, p_exp])
    ctx.launch(_finish_valid_k, E, [center, Ncur, s.disc_centers, s.disc_radii, s.K,
               s.r_dom, s.margin, s.L_final, s.metric_failed, out_valid_wp])
    ctx.launch(_copy_sized_k, E * Ncur, [center, Ncur, size, out_centerline])
    if barriers is not None:
        ctx.launch(publish_barriers_k, E, [s.disc_centers, s.disc_radii, s.K, s.r_dom,
                   size, barriers])


# ===========================================================================
# The registered generator: a BOUNDED Stage
# ===========================================================================

class Repulsive(Stage):
    """``"repulsive"``: the repulsive-growth track generator
    as a ``kind="generator"`` Stage of capture class ``BOUNDED`` (its growth loop is a
    ``ctx.bounded_loop`` with an absorbing per-env freeze). Roadmap PR13d: its dynamic
    ``size`` scales the domain and sampled circles, while the road width remains
    absolute. Growth reserves that width in the unscaled solver coordinates."""

    kind = "generator"
    name = "repulsive"
    output = "centerline"
    capture = CaptureClass.BOUNDED
    Params = RepulsiveParams
    writes = {"xy": "raw.xy", "valid": "gen.valid", "barriers": "gen.repulsive.barriers"}

    def capacity(self, params, layout):
        # The last coarse-to-fine stage is the raw resolution (== num_points, checked by
        # RepulsiveParams against the config).
        return {"raw": int(tuple(params.repulsive_stages)[-1])}

    @staticmethod
    def _run_view(params, *, num_envs: int, scale: float, device: str):
        """The static view read by the scratch allocator and private enqueue core:
        every static RepulsiveParams field, plus ``num_envs``, ``scale`` and ``device``
        (the dynamic ``size`` is read on the device)."""
        view = types.SimpleNamespace(num_envs=int(num_envs), scale=float(scale),
                                     device=str(device))
        for name in RepulsiveParams.__dataclass_fields__:
            if name == "size":
                continue
            setattr(view, name, getattr(params, name))
        return view

    def build(self, b, p):
        cfg = self._run_view(p, num_envs=b.E, scale=b.scale, device=str(b.device))
        cfg._repulsive_half_width = getattr(b, "_repulsive_half_width", 0.0)
        scratch = _alloc_repulsive_scratch(cfg, alloc=b.scratch)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        barriers = b.write("barriers", RepulsiveBarriers)
        seed = b.seed
        size = b.param("size")
        dev = str(xy.device)

        def enqueue(ctx):
            _init()
            _generate_repulsive(seed, cfg, xy, valid, scratch, _resolve_ctx(ctx, dev),
                                size, barriers)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"repulsive"`` track generator (the :class:`Repulsive`
    Stage) into ``reg`` (called by ``track_gen._src.builtin_plugins``; importing this
    module registers nothing)."""
    reg.add(Repulsive)
