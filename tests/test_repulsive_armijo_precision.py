"""Acceptance energy must resolve changes near the constrained stopping scale."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests.conftest import DEVS
from track_gen._src import warp_generate_repulsive as rep
from track_gen.exec import LaunchContext


@wp.kernel
def _objective(points: wp.array(dtype=wp.vec2f), n: int,
               samples: wp.array(dtype=wp.vec2f), mass: wp.array(dtype=wp.float32), m: int,
               centers: wp.array(dtype=wp.vec2f), radii: wp.array(dtype=wp.float32),
               radius: float, margin: float, eps: float,
               out: wp.array(dtype=wp.float64)):
    e = wp.tid()
    value, magnitude = rep._proposal_energy_parts(
        points, points, e, n, 0., 3., 6., eps, samples, mass, m, -1.5,
        centers, radii, 1, radius, margin)
    out[3 * e] = wp.float64(rep._proposal_energy(
        points, points, e, n, 0., 3., 6., eps, samples, mass, m, -1.5,
        centers, radii, 1, radius, margin))
    out[3 * e + 1] = wp.float64(value)
    out[3 * e + 2] = wp.float64(magnitude)


def _fixture(phase):
    n, m = 64, 96
    radius, margin, eps = (float(np.float32(x)) for x in [1.716015, .1, 1e-4])
    angle = np.arange(n) * (2 * np.pi / n) + phase
    first = (1.3165 * np.stack([np.cos(angle), np.sin(angle)], axis=1)).astype(np.float32)
    second = (first * np.float32(1. + 2. ** -22)).astype(np.float32)
    wall_angle = np.arange(m) * (2 * np.pi / m)
    samples = (radius * np.stack([np.cos(wall_angle), np.sin(wall_angle)], axis=1)).astype(np.float32)
    mass = np.full(m, 2 * np.pi * radius / m, np.float32)
    return np.stack([first, second]), samples, mass, radius, margin, eps


def _reference(points, samples, mass, radius, margin, eps):
    x = points.astype(np.float64)
    following = np.roll(x, -1, axis=0)
    ell = np.linalg.norm(following - x, axis=1)
    tangent = (following - x) / ell[:, None]
    weight = .5 * (ell + np.roll(ell, 1))
    n = len(x)
    difference = abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    disjoint = np.minimum(difference, n - difference) > 1
    kernels = np.zeros((n, n), np.float64)
    for i in [0, 1]:
        for j in [0, 1]:
            diff = np.roll(x, -j, axis=0)[None, :, :] - np.roll(x, -i, axis=0)[:, None, :]
            wedge = diff[:, :, 0] * tangent[:, None, 1] - diff[:, :, 1] * tangent[:, None, 0]
            kernels += abs(wedge) ** 3 / ((diff * diff).sum(axis=2) + eps * eps) ** 3
    total = (kernels * ell[:, None] * ell[None, :] * disjoint).sum() / 4.
    diff = x[:, None, :] - samples.astype(np.float64)[None, :, :]
    total += (weight[:, None] * mass.astype(np.float64)[None, :]
              * ((diff * diff).sum(axis=2) + 1e-8) ** -1.5).sum()
    total += (weight * radius ** -2
              * -np.log((radius - margin - np.linalg.norm(x, axis=1)) / radius)).sum()
    return float(total)


def _evaluate(phase, dev):
    points, samples, mass, radius, margin, eps = _fixture(phase)
    e, n, _ = points.shape
    m = len(mass)
    x = wp.array(points.reshape(-1, 2), dtype=wp.vec2f, device=dev)
    obs = wp.array(np.tile(samples, (e, 1)), dtype=wp.vec2f, device=dev)
    weights = wp.array(np.tile(mass, e), dtype=wp.float32, device=dev)
    centers = wp.full(e, wp.vec2f(float('nan'), float('nan')), dtype=wp.vec2f, device=dev)
    radii = wp.zeros(e, dtype=wp.float32, device=dev)
    out = wp.empty(e * 3, dtype=wp.float64, device=dev)
    LaunchContext.current(dev).launch(
        _objective, e, [x, n, obs, weights, m, centers, radii, radius, margin, eps, out])
    expected = np.array([_reference(row, samples, mass, radius, margin, eps) for row in points])
    return out.numpy().reshape(e, 3), expected


@pytest.mark.parametrize('phase', [0., .173])
@pytest.mark.parametrize('dev', DEVS)
def test_objective_resolves_near_stationary_circle_energy(phase, dev):
    actual, expected = _evaluate(phase, dev)
    # Every term is positive in this fixture, so the magnitude is the same energy.
    # A float32 O(N**2) scalar reduction misses this absolute scale by ~1e-4;
    # float64 accumulation leaves only the much smaller local FP32 kernel error.
    np.testing.assert_allclose(actual, np.repeat(expected[:, None], 3, axis=1), rtol=0., atol=1e-5)


@pytest.mark.parametrize('phase', [0., .173])
@pytest.mark.parametrize('dev', DEVS)
def test_nearby_candidate_energy_change_has_correct_sign_and_precision(phase, dev):
    actual, expected = _evaluate(phase, dev)
    delta = expected[1] - expected[0]
    assert delta > 1e-4
    assert actual[1, 0] > actual[0, 0]
    assert actual[1, 0] - actual[0, 0] == pytest.approx(delta, rel=0., abs=1e-5)
