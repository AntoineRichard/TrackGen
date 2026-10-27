"""Test-only sequential oracle for the current Repulsive objective/proposals.

This preserves scalar operation order to test the parallel decomposition. It is
not a generator, public compatibility path, or seeded-output reproduction mode.
The equations and geometric primitives are shared with the live implementation.
"""
from __future__ import annotations

import warp as wp

from track_gen._src.warp_generate_repulsive import _proposal_clear
from track_gen._src.repulsive_parallel_energy import proposal_point as _proposal_point
from track_gen._src.repulsive_tangent_energy import directed_edge_pair_energy
from track_gen._src.repulsive_barrier_energy import (barrier_potential_gradient,
                                                    segment_barrier_potential_gradient)

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


