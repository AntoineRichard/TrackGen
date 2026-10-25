"""Continuous log potentials for Repulsive's filled circular obstacles and wall.

``barrier_potential_gradient`` returns a point potential O_i and its coordinate
gradient in a frame whose origin is the domain center. The curve energy is
``sum_i w_i * O_i``, with dual edge lengths w_i. Its full vertex derivative is
``w_i * grad(O_i) + .5*u_back*(O_prev + O_i)
                   - .5*u_forward*(O_i + O_next)``.
Both the direct and dual-weight terms are required. With p = beta - alpha, this
energy scales as length**(2-p), matching the tangent-point double integral.

The barrier is a search potential, not a collision certificate. Its gaps reserve
the caller's road-width margin; accepted proposals still require segment and
swept geometry checks. Every existing circle and the wall stay active.
"""
from __future__ import annotations

import warp as wp


@wp.func
def barrier_potential_gradient(
    x: wp.vec2f,
    centers: wp.array(dtype=wp.vec2f),
    radii: wp.array(dtype=wp.float32),
    offset: int,
    count: int,
    domain_radius: float,
    margin: float,
    p: float,
):
    """Return ``R**(1-p) * sum(-log(gap/R))`` and its point gradient.

    Positive-radius circles occupy ``[offset:offset+count]``; zero-radius slots
    are unused and their centers are never read. A nonpositive or nonfinite gap
    returns positive infinity and zero gradient, so a line search rejects it
    without propagating a fabricated force. At the domain origin the radial wall
    potential uses the symmetric zero subgradient.
    """
    zero = wp.vec2f(0.0, 0.0)
    infinity = wp.inf
    if (not wp.isfinite(x[0]) or not wp.isfinite(x[1])
            or not wp.isfinite(domain_radius) or domain_radius <= 0.0
            or not wp.isfinite(margin) or margin < 0.0
            or not wp.isfinite(p) or offset < 0 or count < 0):
        return infinity, zero

    distance = wp.length(x)
    wall_gap = domain_radius - margin - distance
    if not wp.isfinite(wall_gap) or wall_gap <= 0.0:
        return infinity, zero
    coefficient = wp.pow(domain_radius, 1.0 - p)
    potential = -wp.log(wall_gap / domain_radius)
    gradient = zero
    if distance > 0.0:
        gradient = x / (distance * wall_gap)

    for j in range(count):
        radius = radii[offset + j]
        if not wp.isfinite(radius) or radius < 0.0:
            return infinity, zero
        if radius > 0.0:
            delta = x - centers[offset + j]
            distance = wp.length(delta)
            gap = distance - radius - margin
            if not wp.isfinite(gap) or gap <= 0.0:
                return infinity, zero
            potential -= wp.log(gap / domain_radius)
            gradient -= delta / (distance * gap)

    potential *= coefficient
    gradient *= coefficient
    if (not wp.isfinite(potential) or not wp.isfinite(gradient[0])
            or not wp.isfinite(gradient[1])):
        return infinity, zero
    return potential, gradient


@wp.func
def segment_barrier_potential_gradient(
    a: wp.vec2f,
    b: wp.vec2f,
    centers: wp.array(dtype=wp.vec2f),
    radii: wp.array(dtype=wp.float32),
    offset: int,
    count: int,
    domain_radius: float,
    margin: float,
    p: float,
):
    """Return the filled-disc segment potential and its endpoint derivatives.

    The potential is ``R**(1-p) * sum(-log(segment_gap/R))``; there is no
    wall term. Integrating along edges means ``sum_edges length * potential``.
    Its endpoint derivatives are ``length*grad_a - potential*unit_edge`` and
    ``length*grad_b + potential*unit_edge``. At an interior closest point with
    clamped segment fraction t, the envelope theorem gives gap derivatives
    ``(1-t)*normal`` and ``t*normal``. These also cover endpoint contacts.

    Invalid gaps return infinity and zero endpoint gradients. A zero-length
    segment uses t=1/2, a symmetric subgradient of its point-distance value.
    """
    zero = wp.vec2f(0.0, 0.0)
    infinity = wp.inf
    if (not wp.isfinite(a[0]) or not wp.isfinite(a[1])
            or not wp.isfinite(b[0]) or not wp.isfinite(b[1])
            or not wp.isfinite(domain_radius) or domain_radius <= 0.0
            or not wp.isfinite(margin) or margin < 0.0
            or not wp.isfinite(p) or offset < 0 or count < 0):
        return infinity, zero, zero
    edge = b - a
    length_sq = wp.dot(edge, edge)
    if not wp.isfinite(length_sq):
        return infinity, zero, zero
    potential = float(0.0)
    gradient_a = zero
    gradient_b = zero
    for j in range(count):
        radius = radii[offset + j]
        if not wp.isfinite(radius) or radius < 0.0:
            return infinity, zero, zero
        if radius > 0.0:
            center = centers[offset + j]
            fraction = float(0.5)
            if length_sq > 0.0:
                fraction = wp.clamp(wp.dot(center - a, edge) / length_sq, 0.0, 1.0)
            delta = a + fraction * edge - center
            distance = wp.length(delta)
            gap = distance - radius - margin
            if not wp.isfinite(gap) or gap <= 0.0:
                return infinity, zero, zero
            potential -= wp.log(gap / domain_radius)
            force = -delta / (distance * gap)
            gradient_a += (1.0 - fraction) * force
            gradient_b += fraction * force
    coefficient = wp.pow(domain_radius, 1.0 - p)
    potential *= coefficient
    gradient_a *= coefficient
    gradient_b *= coefficient
    if (not wp.isfinite(potential) or not wp.isfinite(gradient_a[0])
            or not wp.isfinite(gradient_a[1]) or not wp.isfinite(gradient_b[0])
            or not wp.isfinite(gradient_b[1])):
        return infinity, zero, zero
    return potential, gradient_a, gradient_b
