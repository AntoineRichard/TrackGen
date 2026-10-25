"""Per-edge tangent projection and bounded Newton retraction for Repulsive.

For edge unit vectors u_i, ``J v = u_i dot (v[i+1]-v[i])``. The Euclidean
projector ``P = I-J.T*(J*J.T)^-1*J`` can surround an SPD Sobolev inverse H:
``d=P*H*P*grad`` is tangent and ``grad dot d=(P*grad) dot H*(P*grad)>=0``.
This is a constrained SPD search operator, not the minimum-H-inverse-norm
projector. Its Gram matrix is cyclic tridiagonal, so it needs no dense solve.

All geometry, cyclic factors and reductions use float64. Coordinates and fields
remain vec2f; retraction validates the actual stored float32 candidate. Target scales remain float64; targets
are each reference edge length times target_scale. Unequal reference edges stay
unequal: this preserves the reference parameterization, not global equispacing.

Allocate the workspace once. Device functions never allocate or synchronize.
A failed retraction may leave its candidate scratch dirty; the reference is
read-only and must not alias candidate/output. Callers must honor false/status0
as a blocked proposal, never as stationarity. Accepted geometry still needs the
caller's obstacle/domain and swept-clearance checks.
"""
from __future__ import annotations

import warp as wp


@wp.struct
class EdgeConstraintWorkspace:
    unit: wp.array(dtype=wp.vec2d)
    upper: wp.array(dtype=wp.float64)
    rhs: wp.array(dtype=wp.float64)
    rank: wp.array(dtype=wp.float64)
    delta: wp.array(dtype=wp.vec2d)


def alloc_edge_constraint_workspace(num_envs: int, n_max: int, device, *, alloc=None):
    """Construction-only storage; Stage callers pass their ``b.scratch`` allocator."""
    if int(num_envs) < 0 or int(n_max) < 0:
        raise ValueError('edge constraint workspace dimensions must be nonnegative')
    if alloc is None:
        def alloc(name, dtype, shape):
            return wp.zeros(shape, dtype=dtype, device=device)
    count = int(num_envs) * int(n_max)
    workspace = EdgeConstraintWorkspace()
    workspace.unit = alloc('constraint_unit', wp.vec2d, count)
    workspace.upper = alloc('constraint_upper', wp.float64, count)
    workspace.rhs = alloc('constraint_rhs', wp.float64, count)
    workspace.rank = alloc('constraint_rank', wp.float64, count)
    workspace.delta = alloc('constraint_delta', wp.vec2d, count)
    return workspace


@wp.func
def _finite2(value: wp.vec2d) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1])


@wp.func
def _prepare_units(points: wp.array(dtype=wp.vec2f), e: int, n: int,
                   workspace: EdgeConstraintWorkspace) -> bool:
    if n < 3:
        return False
    base = e * n
    for i in range(n):
        point = wp.vec2d(points[base + i])
        following = wp.vec2d(points[base + (i + 1) % n])
        if not _finite2(point) or not _finite2(following):
            return False
        edge = following - point
        length_sq = wp.dot(edge, edge)
        if not wp.isfinite(length_sq) or length_sq <= wp.float64(0.0):
            return False
        workspace.unit[base + i] = edge / wp.sqrt(length_sq)
    return True


@wp.func
def _solve_edge_gram(e: int, n: int, workspace: EdgeConstraintWorkspace) -> bool:
    # Rank-one cyclic correction: B=T+u*v.T, gamma=-B00, u=(gamma,0,...,wrap),
    # v=(1,0,...,wrap/gamma). Thomas solves both T*y=rhs and T*z=u.
    base = e * n
    first = workspace.unit[base]
    last = workspace.unit[base + n - 1]
    wrap = -wp.dot(last, first)
    gamma = -wp.float64(2.0) * wp.dot(first, first)
    pivot_floor = wp.float64(1.4210854715202004e-14)
    if not wp.isfinite(gamma) or gamma >= -pivot_floor:
        return False
    for i in range(n):
        unit = workspace.unit[base + i]
        diagonal = wp.float64(2.0) * wp.dot(unit, unit)
        if i == 0:
            diagonal -= gamma
        if i == n - 1:
            diagonal -= wrap * wrap / gamma
        lower = wp.float64(0.0)
        if i > 0:
            lower = -wp.dot(workspace.unit[base + i - 1], unit)
            diagonal -= lower * workspace.upper[base + i - 1]
        if not wp.isfinite(diagonal) or diagonal <= pivot_floor:
            return False
        upper = wp.float64(0.0)
        if i < n - 1:
            upper = -wp.dot(unit, workspace.unit[base + i + 1]) / diagonal
        workspace.upper[base + i] = upper
        value = workspace.rhs[base + i]
        rank_value = wp.float64(0.0)
        if i == 0:
            rank_value = gamma
        if i == n - 1:
            rank_value = wrap
        if i > 0:
            value -= lower * workspace.rhs[base + i - 1]
            rank_value -= lower * workspace.rank[base + i - 1]
        workspace.rhs[base + i] = value / diagonal
        workspace.rank[base + i] = rank_value / diagonal
    for index in range(n - 1):
        i = n - 2 - index
        workspace.rhs[base + i] -= workspace.upper[base + i] * workspace.rhs[base + i + 1]
        workspace.rank[base + i] -= workspace.upper[base + i] * workspace.rank[base + i + 1]
    denominator = (wp.float64(1.0) + workspace.rank[base]
                   + wrap / gamma * workspace.rank[base + n - 1])
    # Straight or nearly rank-deficient closed polygons must fail explicitly.
    if not wp.isfinite(denominator) or denominator <= pivot_floor:
        return False
    factor = (workspace.rhs[base] + wrap / gamma * workspace.rhs[base + n - 1]) / denominator
    for i in range(n):
        value = workspace.rhs[base + i] - factor * workspace.rank[base + i]
        if not wp.isfinite(value):
            return False
        workspace.rhs[base + i] = value
    return True


@wp.func
def project_edge_tangent_field(reference: wp.array(dtype=wp.vec2f),
                               field: wp.array(dtype=wp.vec2f), e: int, n: int,
                               workspace: EdgeConstraintWorkspace,
                               out: wp.array(dtype=wp.vec2f)) -> bool:
    """Write P*field, caching the entire RHS before any possibly in-place output."""
    if not _prepare_units(reference, e, n, workspace):
        return False
    base = e * n
    norm_field = wp.float64(0.0)
    for i in range(n):
        value = wp.vec2d(field[base + i])
        following = wp.vec2d(field[base + (i + 1) % n])
        if not _finite2(value) or not _finite2(following):
            return False
        workspace.rhs[base + i] = wp.dot(workspace.unit[base + i], following - value)
        norm_field += wp.dot(value, value)
    if not _solve_edge_gram(e, n, workspace):
        return False
    for i in range(n):
        previous = (i + n - 1) % n
        correction = (workspace.rhs[base + previous] * workspace.unit[base + previous]
                      - workspace.rhs[base + i] * workspace.unit[base + i])
        value = wp.vec2d(field[base + i]) - correction
        stored = wp.vec2f(value)
        if not _finite2(value) or not wp.isfinite(stored[0]) or not wp.isfinite(stored[1]):
            return False
        workspace.delta[base + i] = value
    residual = wp.float64(0.0)
    for i in range(n):
        difference = workspace.delta[base + (i + 1) % n] - workspace.delta[base + i]
        error = wp.dot(workspace.unit[base + i], difference)
        residual += error * error
    if not wp.isfinite(residual) or residual > wp.float64(1.0e-20) * wp.max(norm_field, wp.float64(1.0e-300)):
        return False
    for i in range(n):
        out[base + i] = wp.vec2f(workspace.delta[base + i])
    return True


@wp.func
def _target_error(reference: wp.array(dtype=wp.vec2f), candidate: wp.array(dtype=wp.vec2f),
                  e: int, n: int, target_scale: wp.float64,
                  workspace: EdgeConstraintWorkspace, fraction: wp.float64):
    base = e * n
    maximum = wp.float64(0.0)
    squared = wp.float64(0.0)
    valid = bool(True)
    for i in range(n):
        j = (i + 1) % n
        ref = wp.vec2d(reference[base + i])
        ref_next = wp.vec2d(reference[base + j])
        point = wp.vec2d(candidate[base + i])
        following = wp.vec2d(candidate[base + j])
        if fraction > wp.float64(0.0):
            # Evaluate exactly the storage format the accepted update will write.
            point = wp.vec2d(wp.vec2f(point + fraction * workspace.delta[base + i]))
            following = wp.vec2d(wp.vec2f(following + fraction * workspace.delta[base + j]))
        target = target_scale * wp.length(ref_next - ref)
        actual = wp.length(following - point)
        if (not _finite2(ref) or not _finite2(ref_next) or not _finite2(point)
                or not _finite2(following) or not wp.isfinite(target) or target <= wp.float64(0.0)
                or not wp.isfinite(actual) or actual <= wp.float64(0.0)):
            valid = False
        else:
            error = actual - target
            maximum = wp.max(maximum, wp.abs(error) / target)
            squared += error * error
    return maximum, squared, valid


@wp.func
def _retraction_system(reference: wp.array(dtype=wp.vec2f), candidate: wp.array(dtype=wp.vec2f),
                       e: int, n: int, target_scale: wp.float64,
                       workspace: EdgeConstraintWorkspace) -> bool:
    if not _prepare_units(candidate, e, n, workspace):
        return False
    base = e * n
    for i in range(n):
        j = (i + 1) % n
        actual = wp.length(wp.vec2d(candidate[base + j]) - wp.vec2d(candidate[base + i]))
        target = target_scale * wp.length(wp.vec2d(reference[base + j]) - wp.vec2d(reference[base + i]))
        workspace.rhs[base + i] = actual - target
    return _solve_edge_gram(e, n, workspace)


@wp.func
def retract_edge_lengths(reference: wp.array(dtype=wp.vec2f),
                          candidate: wp.array(dtype=wp.vec2f), e: int, n: int,
                          target_scale: wp.float64, workspace: EdgeConstraintWorkspace,
                          relative_tol: float, max_iters: int, max_backtracks: int) -> bool:
    """Retract candidate scratch to scaled reference edge lengths or return false.

    Each Newton update must decrease squared constraint error after vec2f storage.
    Success also requires the final maximum relative error and a nonsingular Gram.
    Reference/candidate must be disjoint; failure can leave candidate scratch dirty.
    """
    if (n < 3 or not wp.isfinite(target_scale) or target_scale <= 0.0
            or not wp.isfinite(relative_tol) or relative_tol <= 0.0
            or max_iters <= 0 or max_backtracks <= 0):
        return False
    if not _prepare_units(reference, e, n, workspace):
        return False
    base = e * n
    for i in range(n):
        workspace.rhs[base + i] = wp.float64(0.0)
    if not _solve_edge_gram(e, n, workspace):
        return False
    scale = wp.float64(target_scale)
    tolerance = wp.float64(relative_tol)
    for iteration in range(max_iters):
        error, old_squared, valid = _target_error(reference, candidate, e, n, scale, workspace, wp.float64(0.0))
        if not valid or not _retraction_system(reference, candidate, e, n, scale, workspace):
            return False
        if error <= tolerance:
            return True
        for i in range(n):
            previous = (i + n - 1) % n
            workspace.delta[base + i] = (
                workspace.rhs[base + i] * workspace.unit[base + i]
                - workspace.rhs[base + previous] * workspace.unit[base + previous])
        fraction = wp.float64(1.0)
        accepted = bool(False)
        for trial in range(max_backtracks):
            unused_error, squared, finite = _target_error(reference, candidate, e, n, scale, workspace, fraction)
            if finite and squared < old_squared:
                accepted = True
                break
            fraction *= wp.float64(0.5)
        if not accepted:
            return False
        for i in range(n):
            candidate[base + i] = wp.vec2f(wp.vec2d(candidate[base + i]) + fraction * workspace.delta[base + i])
    error, unused_squared, valid = _target_error(reference, candidate, e, n, scale, workspace, wp.float64(0.0))
    return valid and error <= tolerance and _retraction_system(reference, candidate, e, n, scale, workspace)


@wp.kernel
def _project_edges_k(reference: wp.array(dtype=wp.vec2f), field: wp.array(dtype=wp.vec2f),
                       n: int, frozen: wp.array(dtype=wp.int32), workspace: EdgeConstraintWorkspace,
                       status: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
    e = wp.tid()
    if frozen[e] == 1:
        return
    success = project_edge_tangent_field(reference, field, e, n, workspace, out)
    status[e] = wp.int32(success)
    if not success:
        for i in range(n):
            out[e * n + i] = wp.vec2f(0.0, 0.0)


@wp.kernel
def _retract_edges_k(reference: wp.array(dtype=wp.vec2f), candidate: wp.array(dtype=wp.vec2f),
                       n: int, scales: wp.array(dtype=wp.float64), frozen: wp.array(dtype=wp.int32),
                       workspace: EdgeConstraintWorkspace, relative_tol: float,
                       max_iters: int, max_backtracks: int, status: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if frozen[e] == 1:
        return
    status[e] = wp.int32(retract_edge_lengths(reference, candidate, e, n, scales[e], workspace,
                                             relative_tol, max_iters, max_backtracks))


def project_edges(ctx, reference, field, n: int, frozen, workspace, status, out) -> None:
    """Enqueue the projector on the caller's context using only fixed storage."""
    if reference.ptr == out.ptr:
        raise ValueError('edge projection reference must not alias output')
    ctx.launch(_project_edges_k, int(frozen.shape[0]),
               [reference, field, int(n), frozen, workspace, status, out])


def retract_edges(ctx, reference, candidate, n: int, scales, frozen, workspace,
                    relative_tol: float, max_iters: int, max_backtracks: int, status) -> None:
    """Enqueue bounded Newton retraction; status0 leaves unpublished candidate scratch."""
    if reference.ptr == candidate.ptr:
        raise ValueError('edge retraction reference must not alias candidate')
    ctx.launch(_retract_edges_k, int(frozen.shape[0]),
               [reference, candidate, int(n), scales, frozen, workspace,
                float(relative_tol), int(max_iters), int(max_backtracks), status])
