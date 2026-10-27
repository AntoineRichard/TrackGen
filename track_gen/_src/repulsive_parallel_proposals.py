"""Preallocated parallel certificates and bounded Repulsive proposal searches.

Perimeter and centroid preparation retain the scalar solver's FP64 operation
order. Point updates and exact wall/disc certificates run per edge. Energy rows
use the shared parallel evaluator, then scalar decisions preserve first-accepted
trial semantics. Every operation uses the caller's context; no host decisions,
allocation or synchronization occur in these wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .repulsive_clearance import _tolerance, segment_disc_clear, swept_segment_disc_clear
from .repulsive_parallel_energy import (EnergyModel, EnergyWorkspace,
                                        alloc_energy_workspace, evaluate_energy,
                                        proposal_point)


@dataclass(frozen=True, slots=True)
class ProposalWorkspace:
    E: int
    Nmax: int
    energy_workspace: EnergyWorkspace
    trial_points: wp.array
    edge_clear: wp.array
    move_rows: wp.array
    pending: wp.array
    stationary: wp.array
    regular: wp.array
    energy_active: wp.array
    accepted: wp.array
    fraction: wp.array
    accepted_fraction: wp.array
    centroid: wp.array
    projection_scale: wp.array
    old_perimeter: wp.array
    before: wp.array
    after: wp.array
    magnitude: wp.array


def alloc_proposal_workspace(E, Nmax, device, alloc=None):
    if alloc is None:
        def alloc(name, dtype, shape):
            return wp.zeros(shape, dtype=dtype, device=device)

    def energy_alloc(name, dtype, shape):
        return alloc('energy_' + name, dtype, shape)

    energy = alloc_energy_workspace(E, Nmax, device, alloc=energy_alloc)
    arrays = {}
    for name, dtype, shape in [
        ('trial_points', wp.vec2f, E*Nmax), ('edge_clear', wp.int32, E*Nmax),
        ('move_rows', wp.float32, E*Nmax), ('pending', wp.int32, E),
        ('stationary', wp.int32, E), ('regular', wp.int32, E),
        ('energy_active', wp.int32, E), ('accepted', wp.int32, E),
        ('fraction', wp.float32, E), ('accepted_fraction', wp.float32, E),
        ('centroid', wp.vec2d, E), ('projection_scale', wp.float64, E),
        ('old_perimeter', wp.float64, E), ('before', wp.float64, E),
        ('after', wp.float64, E), ('magnitude', wp.float64, E),
    ]:
        arrays[name] = alloc(name, dtype, shape)
    return ProposalWorkspace(E, Nmax, energy, **arrays)


@wp.kernel
def _reset_rows_k(n: int, edge_clear: wp.array(dtype=wp.int32),
                   move_rows: wp.array(dtype=wp.float32)):
    e, i = wp.tid()
    edge_clear[e*n+i] = 0
    move_rows[e*n+i] = 0.0


@wp.kernel
def _init_shape_k(center: wp.array(dtype=wp.vec2f), n: int,
                   frozen: wp.array(dtype=wp.int32), step_slope: wp.array(dtype=wp.float64),
                   numerical_stationary: wp.array(dtype=wp.int32),
                   metric_failed: wp.array(dtype=wp.int32),
                   blocked: wp.array(dtype=wp.int32), accepted_out: wp.array(dtype=wp.int32),
                   pending: wp.array(dtype=wp.int32), stationary: wp.array(dtype=wp.int32),
                   regular: wp.array(dtype=wp.int32), energy_active: wp.array(dtype=wp.int32),
                   accepted: wp.array(dtype=wp.int32), fraction: wp.array(dtype=wp.float32),
                   accepted_fraction: wp.array(dtype=wp.float32),
                   old_perimeter: wp.array(dtype=wp.float64), before: wp.array(dtype=wp.float64),
                   after: wp.array(dtype=wp.float64), magnitude: wp.array(dtype=wp.float64)):
    e = wp.tid()
    pending[e] = 0
    stationary[e] = 0
    regular[e] = 0
    energy_active[e] = 0
    accepted[e] = 0
    fraction[e] = 1.0
    accepted_fraction[e] = 0.0
    old_perimeter[e] = wp.float64(0.0)
    before[e] = wp.float64(0.0)
    after[e] = wp.float64(0.0)
    magnitude[e] = wp.float64(0.0)
    # Preserve the scalar kernel's untouched external state for frozen rows.
    if frozen[e] == 1:
        return
    accepted_out[e] = 0
    if metric_failed[e] != 0:
        blocked[e] = 1
        return
    if numerical_stationary[e] == 1:
        stationary[e] = 1
        return
    if step_slope[e] >= wp.float64(0.0):
        blocked[e] = 1
        return
    peri = wp.float64(0.0)
    for i in range(n):
        peri += wp.length(wp.vec2d(center[e*n+(i+1) % n]) - wp.vec2d(center[e*n+i]))
    old_perimeter[e] = peri
    pending[e] = 1
    regular[e] = 1


@wp.kernel
def _prepare_shape_trial_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                            n: int, pending: wp.array(dtype=wp.int32),
                            fraction: wp.array(dtype=wp.float32),
                            old_perimeter: wp.array(dtype=wp.float64),
                            centroid: wp.array(dtype=wp.vec2d),
                            projection_scale: wp.array(dtype=wp.float64)):
    e = wp.tid()
    if pending[e] != 1:
        return
    b = e*n
    bc = wp.vec2d(0.0, 0.0)
    peri = wp.float64(0.0)
    for i in range(n):
        xi = proposal_point(center, proposal, b+i, fraction[e])
        xn = proposal_point(center, proposal, b+(i+1) % n, fraction[e])
        bc += wp.vec2d(xi)
        peri += wp.length(wp.vec2d(xn) - wp.vec2d(xi))
    centroid[e] = bc / wp.float64(n)
    projection_scale[e] = old_perimeter[e] / wp.max(peri, wp.float64(1.0e-9))


@wp.kernel
def _make_trial_points_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                          out: wp.array(dtype=wp.vec2f), n: int,
                          pending: wp.array(dtype=wp.int32), fraction: wp.array(dtype=wp.float32),
                          centroid: wp.array(dtype=wp.vec2d),
                          projection_scale: wp.array(dtype=wp.float64),
                          move_rows: wp.array(dtype=wp.float32)):
    e, i = wp.tid()
    if pending[e] != 1:
        return
    t = e*n+i
    xi = proposal_point(center, proposal, t, fraction[e])
    out[t] = wp.vec2f(wp.vec2d(xi) + (projection_scale[e] - wp.float64(1.0))
                     * (wp.vec2d(xi) - centroid[e]))
    move_rows[t] = wp.length(out[t] - center[t])


@wp.kernel
def _edge_certificate_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                         n: int, active: wp.array(dtype=wp.int32),
                         stationary: wp.array(dtype=wp.int32),
                         fraction: wp.array(dtype=wp.float32), fractional: int,
                         disc_centers: wp.array(dtype=wp.vec2f),
                         disc_radii: wp.array(dtype=wp.float32), k: int, r_dom: float,
                         margin: wp.array(dtype=wp.float32),
                         edge_clear: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    if active[e] != 1:
        return
    b = e*n
    f = float(1.0)
    if fractional == 1:
        f = fraction[e]
    if stationary[e] == 1:
        f = 0.0
    a = proposal_point(center, proposal, b+i, f)
    z = proposal_point(center, proposal, b+(i+1) % n, f)
    limit = wp.float64(r_dom) - wp.float64(margin[e]) - _tolerance(r_dom)
    ad = wp.vec2d(a)
    squared = wp.dot(ad, ad)
    clear = (wp.isfinite(limit) and limit > wp.float64(0.0)
             and wp.isfinite(squared) and squared < limit*limit)
    for j in range(k):
        radius = disc_radii[e*k+j]
        if radius > 0.0:
            c = disc_centers[e*k+j]
            if not segment_disc_clear(a, z, c, radius + margin[e]):
                clear = False
            if stationary[e] != 1 and not swept_segment_disc_clear(
                    center[b+i], center[b+(i+1) % n], a, z, c, radius + margin[e]):
                clear = False
    edge_clear[b+i] = wp.int32(clear)


@wp.kernel
def _finish_stationary_k(n: int, stationary: wp.array(dtype=wp.int32),
                          edge_clear: wp.array(dtype=wp.int32), blocked: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if stationary[e] != 1:
        return
    clear = int(1)
    for i in range(n):
        if edge_clear[e*n+i] == 0:
            clear = 0
    blocked[e] = 1-clear
    stationary[e] = 0


@wp.kernel
def _reduce_trial_certificate_k(n: int, pending: wp.array(dtype=wp.int32),
                                 edge_clear: wp.array(dtype=wp.int32),
                                 move_rows: wp.array(dtype=wp.float32), require_motion: int,
                                 energy_active: wp.array(dtype=wp.int32)):
    e = wp.tid()
    energy_active[e] = 0
    if pending[e] != 1:
        return
    clear = int(1)
    movement = float(0.0)
    for i in range(n):
        if edge_clear[e*n+i] == 0:
            clear = 0
        movement = wp.max(movement, move_rows[e*n+i])
    if require_motion == 1 and movement == 0.0:
        pending[e] = 0
        return
    energy_active[e] = clear


@wp.kernel
def _decide_shape_k(energy: wp.array(dtype=wp.float64), after: wp.array(dtype=wp.float64),
                     step_slope: wp.array(dtype=wp.float64),
                     pending: wp.array(dtype=wp.int32), energy_active: wp.array(dtype=wp.int32),
                     fraction: wp.array(dtype=wp.float32), accepted: wp.array(dtype=wp.int32),
                     accepted_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if pending[e] != 1:
        return
    if (energy_active[e] == 1 and wp.isfinite(after[e]) and after[e] < energy[e]
            and after[e] <= energy[e] + wp.float64(1.0e-4)*wp.float64(fraction[e])*step_slope[e]):
        accepted[e] = 1
        accepted_out[e] = 1
        pending[e] = 0
    else:
        fraction[e] *= 0.5


@wp.kernel
def _finish_shape_k(center: wp.array(dtype=wp.vec2f), trial_points: wp.array(dtype=wp.vec2f),
                     n: int, regular: wp.array(dtype=wp.int32), accepted: wp.array(dtype=wp.int32),
                     msl: wp.array(dtype=wp.float32), blocked: wp.array(dtype=wp.int32),
                     window_move: wp.array(dtype=wp.float32)):
    e = wp.tid()
    if regular[e] != 1:
        return
    blocked[e] = wp.int32(accepted[e] == 0)
    movement = float(0.0)
    if accepted[e] == 1:
        for i in range(n):
            t = e*n+i
            movement = wp.max(movement, wp.length(trial_points[t] - center[t]))
            center[t] = trial_points[t]
    window_move[e] = wp.max(window_move[e], movement / wp.max(msl[e], 1.0e-9))


@wp.kernel
def _init_geometric_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                       n: int, frozen: wp.array(dtype=wp.int32), skip_frozen: int,
                       target: wp.array(dtype=wp.float64), growth: float, is_growth: int,
                       pending: wp.array(dtype=wp.int32), stationary: wp.array(dtype=wp.int32),
                       regular: wp.array(dtype=wp.int32), energy_active: wp.array(dtype=wp.int32),
                       accepted: wp.array(dtype=wp.int32), fraction: wp.array(dtype=wp.float32),
                       accepted_fraction: wp.array(dtype=wp.float32),
                       centroid: wp.array(dtype=wp.vec2d), projection_scale: wp.array(dtype=wp.float64),
                       old_perimeter: wp.array(dtype=wp.float64), before: wp.array(dtype=wp.float64),
                       after: wp.array(dtype=wp.float64), magnitude: wp.array(dtype=wp.float64)):
    e = wp.tid()
    pending[e] = 0
    stationary[e] = 0
    regular[e] = 0
    energy_active[e] = 0
    accepted[e] = 0
    fraction[e] = 1.0
    accepted_fraction[e] = 0.0
    centroid[e] = wp.vec2d(0.0, 0.0)
    projection_scale[e] = wp.float64(1.0)
    old_perimeter[e] = wp.float64(0.0)
    before[e] = wp.float64(0.0)
    after[e] = wp.float64(0.0)
    magnitude[e] = wp.float64(0.0)
    if skip_frozen == 1 and frozen[e] == 1:
        return
    old_peri = wp.float64(0.0)
    new_peri = wp.float64(0.0)
    bc = wp.vec2d(0.0, 0.0)
    for i in range(n):
        t = e*n+i
        old_peri += wp.length(wp.vec2d(center[e*n+(i+1) % n]) - wp.vec2d(center[t]))
        if is_growth == 1:
            bc += wp.vec2d(center[t])
        else:
            new_peri += wp.length(wp.vec2d(proposal[e*n+(i+1) % n]) - wp.vec2d(proposal[t]))
            bc += wp.vec2d(proposal[t])
    centroid[e] = bc / wp.float64(n)
    old_perimeter[e] = old_peri
    if is_growth == 1:
        projection_scale[e] = wp.min(target[e] / wp.max(old_peri, wp.float64(1.0e-9)),
                                     wp.float64(1.0) + wp.float64(growth))
    else:
        projection_scale[e] = old_peri / wp.max(new_peri, wp.float64(1.0e-9))
    pending[e] = 1


@wp.kernel
def _make_growth_proposal_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                            n: int, pending: wp.array(dtype=wp.int32),
                            centroid: wp.array(dtype=wp.vec2d),
                            projection_scale: wp.array(dtype=wp.float64)):
    e, i = wp.tid()
    if pending[e] == 1:
        t = e*n+i
        proposal[t] = wp.vec2f(wp.vec2d(center[t]) + (projection_scale[e] - wp.float64(1.0))
                              * (wp.vec2d(center[t]) - centroid[e]))


@wp.kernel
def _decide_growth_k(pending: wp.array(dtype=wp.int32), energy_active: wp.array(dtype=wp.int32),
                      fraction: wp.array(dtype=wp.float32), accepted: wp.array(dtype=wp.int32),
                      accepted_fraction: wp.array(dtype=wp.float32)):
    e = wp.tid()
    if pending[e] != 1:
        return
    if energy_active[e] == 1:
        accepted[e] = 1
        accepted_fraction[e] = fraction[e]
        pending[e] = 0
    else:
        fraction[e] *= 0.5


@wp.kernel
def _finish_growth_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f), n: int,
                      accepted: wp.array(dtype=wp.int32),
                      accepted_fraction: wp.array(dtype=wp.float32)):
    e, i = wp.tid()
    if accepted[e] == 1:
        t = e*n+i
        center[t] = proposal_point(center, proposal, t, accepted_fraction[e])


@wp.kernel
def _finish_resample_k(center: wp.array(dtype=wp.vec2f), proposal: wp.array(dtype=wp.vec2f),
                        n: int, energy_active: wp.array(dtype=wp.int32),
                        before: wp.array(dtype=wp.float64), after: wp.array(dtype=wp.float64),
                        old_perimeter: wp.array(dtype=wp.float64),
                        window_move: wp.array(dtype=wp.float32)):
    e = wp.tid()
    if energy_active[e] != 1:
        return
    if wp.isfinite(after[e]) and after[e] <= before[e]:
        movement = float(0.0)
        for i in range(n):
            t = e*n+i
            movement = wp.max(movement, wp.length(proposal[t] - center[t]))
            center[t] = proposal[t]
        window_move[e] = wp.max(window_move[e], wp.float32(wp.float64(movement)
                                 / wp.max(old_perimeter[e] / wp.float64(n), wp.float64(1.0e-9))))


def _certificate(ctx, center, proposal, n, model, ws, *, active, fractional=0):
    ctx.launch(_edge_certificate_k, (ws.E, n),
               [center, proposal, n, active, ws.stationary, ws.fraction, fractional,
                model.disc_centers, model.disc_radii, model.k, model.r_dom, model.margin,
                ws.edge_clear])


def accept_shape_inplace(ctx, center, proposal, n, model: EnergyModel, energy, msl,
                          frozen, window_move, blocked, step_slope,
                          numerical_stationary, metric_failed, accepted_out,
                          workspace: ProposalWorkspace):
    ws = workspace
    ctx.launch(_reset_rows_k, (ws.E, n), [n, ws.edge_clear, ws.move_rows])
    ctx.launch(_init_shape_k, ws.E,
               [center, n, frozen, step_slope, numerical_stationary, metric_failed,
                blocked, accepted_out, ws.pending, ws.stationary, ws.regular,
                ws.energy_active, ws.accepted, ws.fraction, ws.accepted_fraction,
                ws.old_perimeter, ws.before, ws.after, ws.magnitude])
    _certificate(ctx, center, center, n, model, ws, active=ws.stationary)
    ctx.launch(_finish_stationary_k, ws.E, [n, ws.stationary, ws.edge_clear, blocked])
    for _ in range(24):
        ctx.launch(_prepare_shape_trial_k, ws.E,
                   [center, proposal, n, ws.pending, ws.fraction, ws.old_perimeter,
                    ws.centroid, ws.projection_scale])
        ctx.launch(_make_trial_points_k, (ws.E, n),
                   [center, proposal, ws.trial_points, n, ws.pending, ws.fraction,
                    ws.centroid, ws.projection_scale, ws.move_rows])
        _certificate(ctx, center, ws.trial_points, n, model, ws, active=ws.pending)
        ctx.launch(_reduce_trial_certificate_k, ws.E,
                   [n, ws.pending, ws.edge_clear, ws.move_rows, 1, ws.energy_active])
        evaluate_energy(ctx, center, ws.trial_points, n, 1., model, ws.energy_active,
                        ws.energy_workspace, ws.after, ws.magnitude)
        ctx.launch(_decide_shape_k, ws.E,
                   [energy, ws.after, step_slope, ws.pending, ws.energy_active,
                    ws.fraction, ws.accepted, accepted_out])
    ctx.launch(_finish_shape_k, ws.E,
               [center, ws.trial_points, n, ws.regular, ws.accepted, msl, blocked, window_move])


def _init_geometric(ctx, center, proposal, n, frozen, skip_frozen, target, growth,
                    is_growth, ws):
    ctx.launch(_reset_rows_k, (ws.E, n), [n, ws.edge_clear, ws.move_rows])
    ctx.launch(_init_geometric_k, ws.E,
               [center, proposal, n, frozen, skip_frozen, target, growth, is_growth,
                ws.pending, ws.stationary, ws.regular, ws.energy_active, ws.accepted,
                ws.fraction, ws.accepted_fraction, ws.centroid, ws.projection_scale,
                ws.old_perimeter, ws.before, ws.after, ws.magnitude])


def grow_feasible_inplace(ctx, center, proposal, n, target, growth, model: EnergyModel,
                           frozen, workspace: ProposalWorkspace):
    ws = workspace
    _init_geometric(ctx, center, proposal, n, frozen, 1, target, growth, 1, ws)
    ctx.launch(_make_growth_proposal_k, (ws.E, n),
               [center, proposal, n, ws.pending, ws.centroid, ws.projection_scale])
    for _ in range(16):
        _certificate(ctx, center, proposal, n, model, ws, active=ws.pending, fractional=1)
        ctx.launch(_reduce_trial_certificate_k, ws.E,
                   [n, ws.pending, ws.edge_clear, ws.move_rows, 0, ws.energy_active])
        ctx.launch(_decide_growth_k, ws.E,
                   [ws.pending, ws.energy_active, ws.fraction, ws.accepted, ws.accepted_fraction])
    ctx.launch(_finish_growth_k, (ws.E, n),
               [center, proposal, n, ws.accepted, ws.accepted_fraction])


def accept_resample_inplace(ctx, center, proposal, n, model: EnergyModel, frozen,
                             window_move, skip_frozen, workspace: ProposalWorkspace):
    ws = workspace
    # target is not read in resample mode; this reuses a preallocated compatible buffer.
    _init_geometric(ctx, center, proposal, n, frozen, skip_frozen, ws.before, 0., 0, ws)
    ctx.launch(_make_trial_points_k, (ws.E, n),
               [center, proposal, proposal, n, ws.pending, ws.fraction,
                ws.centroid, ws.projection_scale, ws.move_rows])
    _certificate(ctx, center, proposal, n, model, ws, active=ws.pending)
    ctx.launch(_reduce_trial_certificate_k, ws.E,
               [n, ws.pending, ws.edge_clear, ws.move_rows, 0, ws.energy_active])
    evaluate_energy(ctx, center, center, n, 0., model, ws.energy_active,
                    ws.energy_workspace, ws.before, ws.magnitude)
    evaluate_energy(ctx, center, proposal, n, 1., model, ws.energy_active,
                    ws.energy_workspace, ws.after, ws.magnitude)
    ctx.launch(_finish_resample_k, ws.E,
               [center, proposal, n, ws.energy_active, ws.before, ws.after,
                ws.old_perimeter, window_move])
