"""Edge-tangent trapezoidal tangent-point energy and complete endpoint derivatives.

This is the quadrature in Yu, Schumacher and Crane, "Repulsive Curves",
Section 5.1, Equations (17)-(18). For directed disjoint edges I=(a,b), J=(c,d):
``|I|*|J|/4 * sum_{u in I,v in J} |(v-u) cross T_I|**alpha
                                      / (|v-u|**2+eps**2)**(beta/2)``.
The complete curve energy sums ordered edge pairs. Callers omit edges sharing
an endpoint, including the closing edge; on a simple closed ring this is circular
edge-index distance <=1. The numerator has no added epsilon or absolute-value
cusp. The distance regularization should scale with coordinates.

Each call returns local values only, making deterministic per-vertex gathers
possible without atomics, device allocation or host synchronization. Unsupported
exponents, zero edges, coincident quadrature points or nonfinite arithmetic yield
infinite energy and zero derivatives, so proposal acceptance can reject them.
"""
from __future__ import annotations

import warp as wp


@wp.func
def _separated_finite(a: wp.vec2f, b: wp.vec2f) -> bool:
    d = b - a
    squared = wp.dot(d, d)
    return wp.isfinite(squared) and squared > 0.0


@wp.func
def _pair_supported(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f,
                    alpha: float, beta: float, eps: float) -> bool:
    parameters = (wp.isfinite(alpha) and wp.isfinite(beta) and wp.isfinite(eps)
                  and alpha > 1.0 and beta >= alpha + 2.0
                  and beta < 2.0 * alpha + 1.0 and eps >= 0.0)
    return (parameters and _separated_finite(a, b) and _separated_finite(c, d)
            and _separated_finite(a, c) and _separated_finite(a, d)
            and _separated_finite(b, c) and _separated_finite(b, d))


@wp.func
def _kernel_value(diff: wp.vec2f, tangent: wp.vec2f,
                  alpha: float, beta: float, eps: float) -> float:
    wedge = diff[0] * tangent[1] - diff[1] * tangent[0]
    denominator = wp.pow(wp.dot(diff, diff) + eps * eps, 0.5 * beta)
    return wp.pow(wp.abs(wedge), alpha) / denominator


@wp.func
def _kernel_terms(diff: wp.vec2f, tangent: wp.vec2f,
                  alpha: float, beta: float, eps: float):
    squared = wp.dot(diff, diff) + eps * eps
    denominator = wp.pow(squared, 0.5 * beta)
    wedge = diff[0] * tangent[1] - diff[1] * tangent[0]
    absolute = wp.abs(wedge)
    value = wp.pow(absolute, alpha) / denominator
    slope = float(0.0)
    # For every alpha>1, d|w|**alpha/dw is zero at w=0, including alpha<2.
    # Avoid 0**(alpha-2)*w, whose intermediate is undefined for that regime.
    if absolute > 0.0:
        sign = float(1.0)
        if wedge < 0.0:
            sign = -1.0
        slope = alpha * wp.pow(absolute, alpha - 1.0) * sign / denominator
    gradient_diff = slope * wp.vec2f(tangent[1], -tangent[0]) \
        - (beta * value / squared) * diff
    gradient_tangent = slope * wp.vec2f(-diff[1], diff[0])
    return value, gradient_diff, gradient_tangent


@wp.func
def directed_edge_pair_energy(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f,
                              alpha: float, beta: float, eps: float) -> float:
    """Directed disjoint-edge energy; the owner tangent belongs to (a,b)."""
    if not _pair_supported(a, b, c, d, alpha, beta, eps):
        return wp.inf
    owner_length = wp.length(b - a)
    partner_length = wp.length(d - c)
    tangent = (b - a) / owner_length
    value = _kernel_value(c - a, tangent, alpha, beta, eps)
    value += _kernel_value(d - a, tangent, alpha, beta, eps)
    value += _kernel_value(c - b, tangent, alpha, beta, eps)
    value += _kernel_value(d - b, tangent, alpha, beta, eps)
    result = 0.25 * owner_length * partner_length * value
    if not wp.isfinite(result):
        return wp.inf
    return result


@wp.func
def directed_edge_pair_gradient(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f,
                                alpha: float, beta: float, eps: float):
    """Return (energy, grad_a, grad_b, grad_c, grad_d), including both lengths.

    The owner tangent Jacobian is ``(I-T*T.T)/owner_length``. Both edge-length
    derivatives are included in addition to the four endpoint-difference terms.
    Reverse directed pairs have the partner's tangent and must also be gathered.
    """
    zero = wp.vec2f(0.0, 0.0)
    if not _pair_supported(a, b, c, d, alpha, beta, eps):
        return wp.inf, zero, zero, zero, zero
    owner_length = wp.length(b - a)
    partner_length = wp.length(d - c)
    tangent = (b - a) / owner_length
    partner_tangent = (d - c) / partner_length
    kac, gac, tac = _kernel_terms(c - a, tangent, alpha, beta, eps)
    kad, gad, tad = _kernel_terms(d - a, tangent, alpha, beta, eps)
    kbc, gbc, tbc = _kernel_terms(c - b, tangent, alpha, beta, eps)
    kbd, gbd, tbd = _kernel_terms(d - b, tangent, alpha, beta, eps)
    total_kernel = kac + kad + kbc + kbd
    coefficient = 0.25 * owner_length * partner_length
    energy = coefficient * total_kernel
    tangent_force = tac + tad + tbc + tbd
    tangent_force = coefficient / owner_length * (
        tangent_force - tangent * wp.dot(tangent, tangent_force))
    owner_length_force = 0.25 * partner_length * total_kernel * tangent
    partner_length_force = 0.25 * owner_length * total_kernel * partner_tangent
    ga = -coefficient * (gac + gad) - owner_length_force - tangent_force
    gb = -coefficient * (gbc + gbd) + owner_length_force + tangent_force
    gc = coefficient * (gac + gbc) - partner_length_force
    gd = coefficient * (gad + gbd) + partner_length_force
    if (not wp.isfinite(energy) or not wp.isfinite(ga[0]) or not wp.isfinite(ga[1])
            or not wp.isfinite(gb[0]) or not wp.isfinite(gb[1])
            or not wp.isfinite(gc[0]) or not wp.isfinite(gc[1])
            or not wp.isfinite(gd[0]) or not wp.isfinite(gd[1])):
        return wp.inf, zero, zero, zero, zero
    return energy, ga, gb, gc, gd
