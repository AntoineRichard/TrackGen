"""Continuous Repulsive barrier derivatives, including curve quadrature weights."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import warp as wp


from track_gen._src.repulsive_barrier_energy import barrier_potential_gradient
from track_gen._src import repulsive_barrier_energy as _barrier_module
from tests.conftest import DEVS


@wp.kernel
def _point_values(points: wp.array(dtype=wp.vec2f),
                  centers: wp.array(dtype=wp.vec2f), radii: wp.array(dtype=wp.float32),
                  offset: int, count: int, radius: float, margin: float, power: float,
                  values: wp.array(dtype=wp.float32), gradients: wp.array(dtype=wp.vec2f)):
    i = wp.tid()
    value, gradient = barrier_potential_gradient(
        points[i], centers, radii, offset, count, radius, margin, power)
    values[i] = value
    gradients[i] = gradient


@wp.kernel
def _weighted_curve(points: wp.array(dtype=wp.vec2f),
                    centers: wp.array(dtype=wp.vec2f), radii: wp.array(dtype=wp.float32),
                    count: int, radius: float, margin: float, power: float, n: int,
                    values: wp.array(dtype=wp.float32), gradients: wp.array(dtype=wp.vec2f)):
    i = wp.tid()
    previous = points[(i + n - 1) % n]
    point = points[i]
    following = points[(i + 1) % n]
    before, unused_before = barrier_potential_gradient(
        previous, centers, radii, 0, count, radius, margin, power)
    value, gradient = barrier_potential_gradient(
        point, centers, radii, 0, count, radius, margin, power)
    after, unused_after = barrier_potential_gradient(
        following, centers, radii, 0, count, radius, margin, power)
    back = point - previous
    forward = following - point
    lb = wp.length(back)
    lf = wp.length(forward)
    weight = 0.5 * (lb + lf)
    values[i] = weight * value
    gradients[i] = weight * gradient + 0.5 * (before + value) * back / lb \
        - 0.5 * (value + after) * forward / lf


def _evaluate(points, centers=(), radii=(), *, radius=3., margin=.1, power=3.,
              offset=0, count=None, weighted=False, dev='cpu'):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    centers = np.asarray(centers, np.float32).reshape(-1, 2)
    radii = np.asarray(radii, np.float32)
    count = len(radii) - offset if count is None else count
    x = wp.array(points, dtype=wp.vec2f, device=dev)
    c = wp.array(centers, dtype=wp.vec2f, device=dev)
    r = wp.array(radii, dtype=wp.float32, device=dev)
    values = wp.empty(len(points), dtype=wp.float32, device=dev)
    gradients = wp.empty(len(points), dtype=wp.vec2f, device=dev)
    if weighted:
        wp.launch(_weighted_curve, dim=len(points),
                  inputs=[x, c, r, count, radius, margin, power, len(points), values, gradients],
                  device=dev)
    else:
        wp.launch(_point_values, dim=len(points),
                  inputs=[x, c, r, offset, count, radius, margin, power, values, gradients],
                  device=dev)
    return values.numpy().astype(float), gradients.numpy().astype(float)


def _reference(points, centers, radii, radius, margin, power, weighted):
    points = np.asarray(points, float)
    gaps = [radius - margin - np.linalg.norm(points, axis=1)]
    for center, r in zip(centers, radii):
        if r > 0:
            gaps.append(np.linalg.norm(points - center, axis=1) - r - margin)
    potential = -radius ** (1. - power) * np.log(np.stack(gaps) / radius).sum(axis=0)
    if weighted:
        edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        potential *= .5 * (edges + np.roll(edges, 1))
    return potential.sum()


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('dev', DEVS)
def test_barrier_gradient_matches_finite_differences(weighted, dev):
    # Missing dual-weight derivatives or a wrong force sign fails this test.
    a = np.arange(12) * (2 * np.pi / 12)
    x = np.stack([1.1 * np.cos(a), .8 * np.sin(a)], axis=1)
    x += np.random.default_rng(917).normal(scale=.025, size=x.shape)
    centers = np.array([[2.2, .4], [-1.8, 1.7]])
    radii = np.array([.15, .25])
    values, gradient = _evaluate(x, centers, radii, weighted=weighted, dev=dev)
    reference = _reference(x, centers, radii, 3., .1, 3., weighted)
    assert values.sum() == pytest.approx(reference, rel=2e-6)
    fd = np.empty_like(x)
    step = 1e-6
    for i in range(len(x)):
        for d in range(2):
            plus = x.copy(); plus[i, d] += step
            minus = x.copy(); minus[i, d] -= step
            fd[i, d] = (_reference(plus, centers, radii, 3., .1, 3., weighted)
                        - _reference(minus, centers, radii, 3., .1, 3., weighted)) / (2 * step)
    np.testing.assert_allclose(gradient, fd, rtol=2e-5, atol=2e-7)


def test_circle_and_wall_forces_point_toward_clearance():
    values, gradient = _evaluate([[1., 0.]], radius=2., margin=.1)
    assert values[0] == pytest.approx(.199626924, rel=2e-6)
    np.testing.assert_allclose(gradient[0], [1. / 3.6, 0.], rtol=2e-6, atol=1e-7)
    _, disc_gradient = _evaluate([[0., 0.]], [[.6, 0.]], [.2])
    assert disc_gradient[0, 0] == pytest.approx(1. / 2.7, rel=2e-6)
    assert disc_gradient[0, 1] == 0.


@pytest.mark.parametrize('points,centers,radii,radius,margin', [
    ([[1., 0.]], [], [], 1., 0.),
    ([[1.1, 0.]], [], [], 1., 0.),
    ([[.4, 0.]], [[.4, 0.]], [.2], 3., .1),
    ([[float('nan'), 0.]], [], [], 3., .1),
    ([[0., 0.]], [[float('nan'), 0.]], [.2], 3., .1),
    ([[0., 0.]], [], [], 0., 0.),
])
def test_invalid_gap_returns_positive_infinity_and_zero_gradient(
        points, centers, radii, radius, margin):
    value, gradient = _evaluate(points, centers, radii, radius=radius, margin=margin)
    assert np.isposinf(value[0])
    np.testing.assert_array_equal(gradient[0], [0., 0.])


def test_unused_disc_slots_and_environment_offset_are_respected():
    expected = _evaluate([[.8, .2]], [[2.2, .4]], [.15])
    actual = _evaluate([[.8, .2]], [[.8, .2], [2.2, .4], [np.nan, np.nan]],
                       [1., .15, 0.], offset=1, count=2)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-7)


@pytest.mark.parametrize('scale', [.25, 4.])
def test_weighted_energy_and_gradient_scale_like_tangent_point_energy(scale):
    # alpha=3,beta=6 => p=3, E scales c^-1 and coordinate gradient c^-2.
    a = np.arange(16) * (2 * np.pi / 16)
    x = np.stack([np.cos(a), .8 * np.sin(a)], axis=1)
    c = np.array([[2.2, .4], [-1.8, 1.7]])
    r = np.array([.15, .25])
    energy, gradient = _evaluate(x, c, r, weighted=True)
    scaled_energy, scaled_gradient = _evaluate(x * scale, c * scale, r * scale,
                                              radius=3. * scale, margin=.1 * scale,
                                              weighted=True)
    np.testing.assert_allclose(scaled_energy, energy / scale, rtol=3e-6)
    np.testing.assert_allclose(scaled_gradient, gradient / scale ** 2, rtol=2e-5, atol=1e-6)


def test_domain_local_frame_is_translation_and_rotation_equivariant():
    x = np.array([[.8, .2], [-.3, .9]])
    c = np.array([[2.2, .4], [-1.8, 1.7]])
    r = [.15, .25]
    value, gradient = _evaluate(x, c, r)
    rotation = np.array([[0., -1.], [1., 0.]])
    origin = np.array([3., -7.])
    # Domain coordinates are translated back into the helper's origin-centered frame.
    transformed_x = x @ rotation.T + origin
    transformed_c = c @ rotation.T + origin
    moved_value, moved_gradient = _evaluate(transformed_x - origin, transformed_c - origin, r)
    np.testing.assert_allclose(moved_value, value, rtol=2e-6)
    np.testing.assert_allclose(moved_gradient, gradient @ rotation.T, rtol=2e-6, atol=1e-7)


def test_dual_weighted_circle_quadrature_converges_under_refinement():
    energies = []
    for n in [64, 128]:
        angle = np.arange(n) * (2 * np.pi / n)
        x = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        values, _ = _evaluate(x, radius=3., margin=0., weighted=True)
        energies.append(values.sum())
    exact = 2 * np.pi * (-np.log(2. / 3.)) / 9.
    assert abs(energies[1] - exact) < abs(energies[0] - exact)
    assert energies[1] == pytest.approx(exact, rel=2e-4)
    assert energies[1] / energies[0] == pytest.approx(1., rel=5e-4)


@lru_cache(maxsize=1)
def _segment_kernel():
    helper = getattr(_barrier_module, 'segment_barrier_potential_gradient', None)
    assert helper is not None, 'Filled-disc segment contact needs a continuous edge barrier'

    @wp.kernel
    def evaluate(a: wp.vec2f, b: wp.vec2f, centers: wp.array(dtype=wp.vec2f),
                 radii: wp.array(dtype=wp.float32), offset: int, count: int,
                 radius: float, margin: float, power: float, weighted: int,
                 value: wp.array(dtype=wp.float32), gradient: wp.array(dtype=wp.vec2f)):
        potential, ga, gb = helper(a, b, centers, radii, offset, count, radius, margin, power)
        if weighted == 1:
            length = wp.length(b - a)
            direction = (b - a) / wp.max(length, 1.0e-12)
            ga = length * ga - potential * direction
            gb = length * gb + potential * direction
            potential *= length
        value[0] = potential
        gradient[0] = ga
        gradient[1] = gb
    return evaluate


def _segment_evaluate(a, b, centers, radii, *, radius=3., margin=.1, power=3.,
                      weighted=False, offset=0, count=None, dev='cpu'):
    c = wp.array(np.asarray(centers, np.float32).reshape(-1, 2), dtype=wp.vec2f, device=dev)
    r = wp.array(np.asarray(radii, np.float32), dtype=wp.float32, device=dev)
    count = len(radii) - offset if count is None else count
    value = wp.empty(1, dtype=wp.float32, device=dev)
    gradient = wp.empty(2, dtype=wp.vec2f, device=dev)
    wp.launch(_segment_kernel(), dim=1,
              inputs=[wp.vec2f(*a), wp.vec2f(*b), c, r, offset, count,
                      radius, margin, power, int(weighted), value, gradient], device=dev)
    return value.numpy().astype(float)[0], gradient.numpy().astype(float)


def _segment_reference(endpoints, centers, radii, radius=3., margin=.1, power=3.,
                       weighted=False):
    a, b = np.asarray(endpoints, float)
    edge = b - a
    length = np.linalg.norm(edge)
    total = 0.
    for center, r in zip(centers, radii):
        if r > 0:
            fraction = np.clip(np.dot(center - a, edge) / np.dot(edge, edge), 0., 1.)
            gap = np.linalg.norm(a + fraction * edge - center) - r - margin
            total -= np.log(gap / radius)
    return radius ** (1. - power) * total * (length if weighted else 1.)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('dev', DEVS)
@pytest.mark.parametrize('endpoints,centers,radii', [
    ([[-1., 1.], [1., 1.]], [[0., 0.]], [.2]),
    ([[-1., .8], [.5, .8]], [[2., 0.]], [.2]),
    ([[-.9, .7], [.6, 1.1]], [[-.7, -.3], [1.8, 1.6]], [.15, .2]),
])
def test_segment_barrier_endpoint_gradient_matches_finite_differences(
        weighted, endpoints, centers, radii, dev):
    x = np.array(endpoints, float)
    value, gradient = _segment_evaluate(*x, centers, radii, weighted=weighted, dev=dev)
    assert value == pytest.approx(_segment_reference(x, centers, radii, weighted=weighted), rel=2e-6)
    fd = np.zeros_like(x)
    step = 1e-6
    for i in range(2):
        for d in range(2):
            plus = x.copy(); plus[i, d] += step
            minus = x.copy(); minus[i, d] -= step
            fd[i, d] = (_segment_reference(plus, centers, radii, weighted=weighted)
                        - _segment_reference(minus, centers, radii, weighted=weighted)) / (2 * step)
    np.testing.assert_allclose(gradient, fd, rtol=2e-5, atol=2e-7)


def test_segment_crossing_disc_is_rejected_despite_clear_endpoints():
    value, gradient = _segment_evaluate([-1., 0.], [1., 0.], [[0., 0.]], [.2])
    assert np.isposinf(value)
    np.testing.assert_array_equal(gradient, np.zeros((2, 2)))


@pytest.mark.parametrize('a,b,center,radius', [
    ([-1., .375], [1., .375], [0., 0.], .25),
    ([np.nan, 1.], [1., 1.], [0., 0.], .2),
    ([-1., 1.], [1., 1.], [np.nan, 0.], .2),
    ([-1., 1.], [1., 1.], [0., 0.], np.inf),
])
def test_segment_invalid_contact_returns_infinity_without_nonfinite_force(a, b, center, radius):
    value, gradient = _segment_evaluate(a, b, [center], [radius], margin=.125)
    assert np.isposinf(value)
    np.testing.assert_array_equal(gradient, np.zeros((2, 2)))


def test_zero_length_segment_has_symmetric_point_subgradient():
    value, gradient = _segment_evaluate([0., 1.], [0., 1.], [[0., 0.]], [.2])
    assert value == pytest.approx(-np.log(.7 / 3.) / 9., rel=2e-6)
    expected = [[0., -1. / 12.6], [0., -1. / 12.6]]
    np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=1e-7)


def test_segment_has_no_wall_term_and_skips_unused_disc_slots():
    value, gradient = _segment_evaluate([4., 1.], [5., 1.], [[np.nan, np.nan]], [0.])
    assert value == 0.
    np.testing.assert_array_equal(gradient, np.zeros((2, 2)))


def test_segment_barrier_reversal_and_rigid_transform():
    a, b = np.array([[-1., .8], [.5, .8]])
    c = np.array([[2., 0.]])
    value, gradient = _segment_evaluate(a, b, c, [.2], weighted=True)
    reversed_value, reversed_gradient = _segment_evaluate(b, a, c, [.2], weighted=True)
    assert reversed_value == pytest.approx(value, rel=2e-6)
    np.testing.assert_allclose(reversed_gradient, gradient[::-1], rtol=2e-6, atol=1e-7)
    rotation = np.array([[0., -1.], [1., 0.]])
    shift = np.array([2., -3.])
    moved_value, moved_gradient = _segment_evaluate(
        a @ rotation.T + shift, b @ rotation.T + shift, c @ rotation.T + shift,
        [.2], weighted=True)
    assert moved_value == pytest.approx(value, rel=2e-6)
    np.testing.assert_allclose(moved_gradient, gradient @ rotation.T, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize('scale', [.25, 4.])
def test_length_weighted_segment_barrier_has_tangent_point_scaling(scale):
    a, b = np.array([[-1., 1.], [1., 1.]])
    c = np.array([[0., 0.]])
    r = np.array([.2])
    value, gradient = _segment_evaluate(a, b, c, r, weighted=True)
    scaled_value, scaled_gradient = _segment_evaluate(
        a * scale, b * scale, c * scale, r * scale, radius=3. * scale,
        margin=.1 * scale, weighted=True)
    assert scaled_value == pytest.approx(value / scale, rel=2e-6)
    np.testing.assert_allclose(scaled_gradient, gradient / scale ** 2, rtol=3e-6, atol=1e-6)
