"""Parallel rows of Repulsive's unchanged tangent/obstacle objective.

Each (environment, owner edge) evaluates its ordered tangent pairs and sampled
obstacles in fixed order, using the existing FP32 local primitives and FP64 sums.
A fixed row-order FP64 reduction joins the rows. Association changes from the
scalar oracle, with the usual gamma_K absolute-term summation bound; it does not
restore the inaccurate FP32 objective reduction. Scratch is O(E*N), allocated
once, and every hot operation goes through the supplied LaunchContext.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .repulsive_barrier_energy import barrier_potential_gradient, segment_barrier_potential_gradient
from .repulsive_tangent_energy import directed_edge_pair_energy


@dataclass(frozen=True, slots=True)
class EnergyModel:
    alpha: float
    beta: float
    eps: float
    obs_pts: wp.array
    obs_mw: wp.array
    m_obs: int
    p_exp: float
    disc_centers: wp.array
    disc_radii: wp.array
    k: int
    r_dom: float
    margin: wp.array


@dataclass(frozen=True, slots=True)
class EnergyWorkspace:
    E: int
    Nmax: int
    tp_rows: wp.array
    obs_rows: wp.array
    barrier_rows: wp.array
    barrier_abs_rows: wp.array


def alloc_energy_workspace(E, Nmax, device, alloc=None):
    if alloc is None:
        def alloc(name, dtype, shape):
            return wp.zeros(shape, dtype=dtype, device=device)
    rows = [alloc(name, wp.float64, E * Nmax)
            for name in ('tp_rows', 'obs_rows', 'barrier_rows', 'barrier_abs_rows')]
    return EnergyWorkspace(E, Nmax, *rows)


@wp.func
def proposal_point(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                   t: int, fraction: float) -> wp.vec2f:
    # Preserve the exact coordinate arithmetic and endpoint identity branches of
    # the scalar solver. Helpers never construct an interpolated endpoint in f32.
    if fraction == 0.0:
        return center[t]
    if fraction == 1.0:
        return proposal[t]
    a = wp.vec2d(center[t])
    return wp.vec2f(a + wp.float64(fraction) * (wp.vec2d(proposal[t]) - a))


@wp.kernel
def _energy_rows_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                   n: int, fraction: float, alpha: float, beta: float, eps: float,
                   obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32),
                   m_obs: int, p_exp: float, disc_centers: wp.array(dtype=wp.vec2f),
                   disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                   margin: wp.array(dtype=wp.float32), active: wp.array(dtype=wp.int32),
                   tp_rows: wp.array(dtype=wp.float64), obs_rows: wp.array(dtype=wp.float64),
                   barrier_rows: wp.array(dtype=wp.float64),
                   barrier_abs_rows: wp.array(dtype=wp.float64)):
    e, i = wp.tid()
    b = e * n
    t = b + i
    # Every participating partial is initialized on every call, including masks.
    tp_rows[t] = wp.float64(0.0)
    obs_rows[t] = wp.float64(0.0)
    barrier_rows[t] = wp.float64(0.0)
    barrier_abs_rows[t] = wp.float64(0.0)
    if active[e] != 1:
        return
    a = proposal_point(center, proposal, t, fraction)
    z = proposal_point(center, proposal, b + (i + 1) % n, fraction)
    previous = proposal_point(center, proposal, b + (i + n - 1) % n, fraction)
    tangent_energy = wp.float64(0.0)
    for j in range(n):
        distance = wp.abs(i - j)
        if wp.min(distance, n - distance) > 1:
            c = proposal_point(center, proposal, b + j, fraction)
            d = proposal_point(center, proposal, b + (j + 1) % n, fraction)
            tangent_energy += wp.float64(directed_edge_pair_energy(a, z, c, d, alpha, beta, eps))
    weight = 0.5 * (wp.length(z - a) + wp.length(a - previous))
    obstacle_energy = wp.float64(0.0)
    for j in range(m_obs):
        mass = obs_mw[e * m_obs + j]
        if mass != 0.0:
            difference = a - obs_pts[e * m_obs + j]
            value = (wp.float64(weight) * wp.float64(mass)
                     * wp.float64(wp.pow(wp.dot(difference, difference) + 1.0e-8, p_exp)))
            obstacle_energy += value
    barrier, unused_grad = barrier_potential_gradient(
        a, disc_centers, disc_radii, e * k, 0, r_dom, margin[e], -2.0 * p_exp)
    edge_barrier, unused_a, unused_b = segment_barrier_potential_gradient(
        a, z, disc_centers, disc_radii, e * k, k, r_dom, margin[e], -2.0 * p_exp)
    edge_length = wp.length(z - a)
    tp_rows[t] = tangent_energy
    obs_rows[t] = obstacle_energy
    barrier_rows[t] = wp.float64(weight) * wp.float64(barrier) + wp.float64(edge_length) * wp.float64(edge_barrier)
    barrier_abs_rows[t] = (wp.float64(weight) * wp.float64(wp.abs(barrier))
                           + wp.float64(edge_length) * wp.float64(wp.abs(edge_barrier)))


@wp.kernel
def _reduce_energy_rows_k(n: int, r_dom: float, p_exp: float,
                          active: wp.array(dtype=wp.int32),
                          tp_rows: wp.array(dtype=wp.float64),
                          obs_rows: wp.array(dtype=wp.float64),
                          barrier_rows: wp.array(dtype=wp.float64),
                          barrier_abs_rows: wp.array(dtype=wp.float64),
                          total: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64)):
    e = wp.tid()
    if active[e] != 1:
        return
    b = e * n
    tangent_energy = wp.float64(0.0)
    for i in range(n):
        tangent_energy += tp_rows[b + i]
    value = tangent_energy
    magnitude = tangent_energy
    for i in range(n):
        value += obs_rows[b + i]
        magnitude += obs_rows[b + i]
        value += barrier_rows[b + i]
        magnitude += barrier_abs_rows[b + i]
    total[e] = value
    scale[e] = wp.max(magnitude, wp.float64(wp.pow(r_dom, 2.0 + 2.0 * p_exp)))


def evaluate_energy(ctx, center, proposal, n, fraction, model, active, workspace, total, magnitude):
    """Write active objective values and positive scales; masked outputs stay intact.

    ``magnitude`` includes the natural R**(2-p) floor used by the scalar flow.
    All arrays are preallocated and the current stage uses a packed E*n prefix.
    """
    ctx.launch(_energy_rows_k, (workspace.E, n),
               [center, proposal, n, fraction, model.alpha, model.beta, model.eps,
                model.obs_pts, model.obs_mw, model.m_obs, model.p_exp,
                model.disc_centers, model.disc_radii, model.k, model.r_dom, model.margin,
                active, workspace.tp_rows, workspace.obs_rows, workspace.barrier_rows,
                workspace.barrier_abs_rows])
    ctx.launch(_reduce_energy_rows_k, workspace.E,
               [n, model.r_dom, model.p_exp, active, workspace.tp_rows, workspace.obs_rows,
                workspace.barrier_rows, workspace.barrier_abs_rows, total, magnitude])
