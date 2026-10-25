"""Edge-tangent trapezoidal TP quadrature and complete deterministic derivatives."""
from __future__ import annotations

from functools import lru_cache
import importlib
import importlib.util

import numpy as np
import pytest
import warp as wp

from tests.conftest import DEVS


@lru_cache(maxsize=1)
def _model():
    name = 'track_gen._src.repulsive_tangent_energy'
    assert importlib.util.find_spec(name) is not None, 'Faithful TP quadrature needs edge tangents'
    return importlib.import_module(name)


@lru_cache(maxsize=1)
def _pair_kernel():
    model = _model()

    @wp.kernel
    def evaluate(points: wp.array(dtype=wp.vec2f), alpha: float, beta: float, eps: float,
                 values: wp.array(dtype=wp.float32), gradients: wp.array(dtype=wp.vec2f)):
        a = points[0]
        b = points[1]
        c = points[2]
        d = points[3]
        energy, ga, gb, gc, gd = model.directed_edge_pair_gradient(a, b, c, d, alpha, beta, eps)
        values[0] = model.directed_edge_pair_energy(a, b, c, d, alpha, beta, eps)
        values[1] = energy
        gradients[0] = ga
        gradients[1] = gb
        gradients[2] = gc
        gradients[3] = gd
    return evaluate


@lru_cache(maxsize=1)
def _curve_kernels():
    model = _model()

    @wp.kernel
    def energy(points: wp.array(dtype=wp.vec2f), n: int, alpha: float, beta: float,
               eps: float, values: wp.array(dtype=wp.float32)):
        i = wp.tid()
        value = float(0.0)
        for j in range(n):
            distance = wp.abs(i - j)
            if wp.min(distance, n - distance) > 1:
                value += model.directed_edge_pair_energy(
                    points[i], points[(i + 1) % n], points[j], points[(j + 1) % n],
                    alpha, beta, eps)
        values[i] = value

    @wp.kernel
    def gradient(points: wp.array(dtype=wp.vec2f), n: int, alpha: float, beta: float,
                 eps: float, out: wp.array(dtype=wp.vec2f)):
        v = wp.tid()
        result = wp.vec2f(0.0, 0.0)
        for incident in range(2):
            i = v
            if incident == 0:
                i = (v + n - 1) % n
            for j in range(n):
                distance = wp.abs(i - j)
                if wp.min(distance, n - distance) > 1:
                    value, ga, gb, gc, gd = model.directed_edge_pair_gradient(
                        points[i], points[(i + 1) % n], points[j], points[(j + 1) % n],
                        alpha, beta, eps)
                    reverse, ha, hb, hc, hd = model.directed_edge_pair_gradient(
                        points[j], points[(j + 1) % n], points[i], points[(i + 1) % n],
                        alpha, beta, eps)
                    if incident == 0:
                        result += gb + hd
                    else:
                        result += ga + hc
        out[v] = result
    return energy, gradient


def _pair(points, alpha=3., beta=6., eps=1e-4, *, dev='cpu'):
    x = wp.array(np.asarray(points, np.float32), dtype=wp.vec2f, device=dev)
    values = wp.empty(2, dtype=wp.float32, device=dev)
    gradient = wp.empty(4, dtype=wp.vec2f, device=dev)
    wp.launch(_pair_kernel(), dim=1, inputs=[x, alpha, beta, eps, values, gradient], device=dev)
    return values.numpy().astype(float), gradient.numpy().astype(float)


def _curve(points, alpha=3., beta=6., eps=1e-4, *, gradient=True, dev='cpu'):
    x = wp.array(np.asarray(points, np.float32), dtype=wp.vec2f, device=dev)
    values = wp.empty(len(points), dtype=wp.float32, device=dev)
    out = wp.empty(len(points), dtype=wp.vec2f, device=dev)
    energy_kernel, gradient_kernel = _curve_kernels()
    wp.launch(energy_kernel, dim=len(points), inputs=[x, len(points), alpha, beta, eps, values], device=dev)
    if gradient:
        wp.launch(gradient_kernel, dim=len(points), inputs=[x, len(points), alpha, beta, eps, out], device=dev)
    return values.numpy().astype(float).sum(), out.numpy().astype(float) if gradient else None


def _pair_reference(points, alpha=3., beta=6., eps=1e-4):
    a, b, c, d = np.asarray(points, float)
    ell = np.linalg.norm(b - a)
    partner_ell = np.linalg.norm(d - c)
    tangent = (b - a) / ell
    result = 0.
    for owner in [a, b]:
        for partner in [c, d]:
            diff = partner - owner
            wedge = diff[0] * tangent[1] - diff[1] * tangent[0]
            result += abs(wedge) ** alpha / (np.dot(diff, diff) + eps ** 2) ** (beta / 2.)
    return ell * partner_ell * result / 4.


def _curve_reference(points, alpha=3., beta=6., eps=1e-4):
    n = len(points)
    total = 0.
    for i in range(n):
        for j in range(n):
            if min(abs(i - j), n - abs(i - j)) > 1:
                total += _pair_reference([points[i], points[(i + 1) % n],
                                          points[j], points[(j + 1) % n]], alpha, beta, eps)
    return total


def _finite_difference(function, x, h=1e-6):
    result = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        for d in range(2):
            plus = np.array(x, float); plus[i, d] += h
            minus = np.array(x, float); minus[i, d] -= h
            result[i, d] = (function(plus) - function(minus)) / (2 * h)
    return result


@pytest.mark.parametrize('alpha,beta', [(3., 6.), (1.5, 3.75), (2., 4.5)])
@pytest.mark.parametrize('dev', DEVS)
def test_directed_pair_full_endpoint_derivative_matches_independent_finite_difference(alpha, beta, dev):
    points = np.array([[-.8, .2], [.7, .5], [-.3, 1.4], [1.3, .9]])
    values, gradient = _pair(points, alpha, beta, dev=dev)
    reference = _pair_reference(points, alpha, beta)
    np.testing.assert_allclose(values, reference, rtol=3e-6)
    fd = _finite_difference(lambda x: _pair_reference(x, alpha, beta), points)
    np.testing.assert_allclose(gradient, fd, rtol=3e-5, atol=3e-7)


@pytest.mark.parametrize('n', [4, 7, 12])
@pytest.mark.parametrize('dev', DEVS)
def test_ordered_curve_gather_matches_full_energy_derivative_and_wrap_exclusion(n, dev):
    angle = np.arange(n) * (2 * np.pi / n)
    points = np.stack([1.2 * np.cos(angle), .8 * np.sin(angle)], axis=1)
    points += np.random.default_rng(121 + n).normal(scale=.03, size=points.shape)
    value, gradient = _curve(points, dev=dev)
    reference = _curve_reference(points)
    assert value == pytest.approx(reference, rel=3e-6)
    fd = _finite_difference(_curve_reference, points)
    np.testing.assert_allclose(gradient, fd, rtol=5e-5, atol=2e-6)
    np.testing.assert_allclose(gradient.sum(axis=0), [0., 0.], atol=2e-6)


def test_directed_pair_is_rotation_translation_and_endpoint_reversal_equivariant():
    points = np.array([[-.8, .2], [.7, .5], [-.3, 1.4], [1.3, .9]])
    values, gradient = _pair(points)
    rotation = np.array([[0., -1.], [1., 0.]])
    moved_values, moved_gradient = _pair(points @ rotation.T + [2., -3.])
    np.testing.assert_allclose(moved_values, values, rtol=3e-6)
    np.testing.assert_allclose(moved_gradient, gradient @ rotation.T, rtol=5e-6, atol=3e-7)
    permutation = [1, 0, 3, 2]
    reversed_values, reversed_gradient = _pair(points[permutation])
    np.testing.assert_allclose(reversed_values, values, rtol=3e-6)
    np.testing.assert_allclose(reversed_gradient, gradient[permutation], rtol=5e-6, atol=3e-7)


@pytest.mark.parametrize('scale', [.25, 4.])
def test_pair_energy_and_gradient_have_tangent_point_scale_exponents(scale):
    points = np.array([[-.8, .2], [.7, .5], [-.3, 1.4], [1.3, .9]])
    values, gradient = _pair(points)
    scaled_values, scaled_gradient = _pair(points * scale, eps=1e-4 * scale)
    np.testing.assert_allclose(scaled_values, values / scale, rtol=3e-6)
    np.testing.assert_allclose(scaled_gradient, gradient / scale ** 2, rtol=5e-6, atol=3e-7)


@pytest.mark.parametrize('dev', DEVS)
def test_noninteger_alpha_has_zero_collinear_derivative_without_nan(dev):
    values, gradient = _pair([[-1., 0.], [1., 0.], [2., 0.], [3., 0.]], 1.5, 3.75, dev=dev)
    np.testing.assert_array_equal(values, [0., 0.])
    np.testing.assert_array_equal(gradient, np.zeros((4, 2)))


@pytest.mark.parametrize('alpha,beta', [(1., 3.), (3., 4.), (3., 7.)])
def test_pair_rejects_unsupported_sobolev_exponents(alpha, beta):
    values, gradient = _pair([[-1., 0.], [1., 0.], [2., 1.], [3., 1.]], alpha, beta)
    assert np.isposinf(values).all()
    np.testing.assert_array_equal(gradient, np.zeros((4, 2)))


@pytest.mark.parametrize('points', [
    [[0., 0.], [0., 0.], [2., 1.], [3., 1.]],
    [[-1., 0.], [1., 0.], [2., 1.], [2., 1.]],
    [[-1., 0.], [1., 0.], [-1., 0.], [2., 1.]],
    [[np.nan, 0.], [1., 0.], [2., 1.], [3., 1.]],
])
def test_degenerate_or_nonfinite_pair_returns_infinite_energy_and_zero_derivative(points):
    values, gradient = _pair(points)
    assert np.isposinf(values).all()
    np.testing.assert_array_equal(gradient, np.zeros((4, 2)))


def test_circle_quadrature_converges_to_smooth_circle_integral():
    exact = np.pi ** 2 / 2.
    errors = []
    for n in [32, 64, 128]:
        a = np.arange(n) * (2 * np.pi / n)
        value, _ = _curve(np.stack([np.cos(a), np.sin(a)], axis=1), eps=0., gradient=False)
        errors.append(abs(value - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / exact < .05


def test_checkerboard_mode_remains_stiffer_than_quarter_frequency_mode():
    # Centered vertex tangents incorrectly cancel the alternating displacement.
    n = 128
    angle = np.arange(n) * (2 * np.pi / n)
    points = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    length = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    base, _ = _curve(points, eps=0., gradient=False)
    stiffness = []
    amplitude = 1e-4
    for k in [n // 4, n // 2]:
        perturbation = points * np.cos(k * angle)[:, None]
        energies = []
        for sign in [-1., 1.]:
            trial = points + sign * amplitude * perturbation
            trial *= length / np.linalg.norm(np.roll(trial, -1, axis=0) - trial, axis=1).sum()
            energy, _ = _curve(trial, eps=0., gradient=False)
            energies.append(energy)
        stiffness.append((sum(energies) - 2 * base) / (amplitude ** 2 * (perturbation ** 2).sum()))
    assert stiffness[0] > 0
    assert stiffness[1] > 2. * stiffness[0]
