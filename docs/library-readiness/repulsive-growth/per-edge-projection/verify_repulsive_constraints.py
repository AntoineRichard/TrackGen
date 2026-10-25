"""Per-edge tangent projection and bounded float32 curve retraction."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import warp as wp

from tests.conftest import DEVS
from track_gen.exec import LaunchContext
from track_gen.testing.contract import no_allocation, no_host_sync


@lru_cache(maxsize=1)
def _model():
    name = '_trackgen_archived_repulsive_constraints'
    path = Path(__file__).with_name('repulsive_constraints.py')
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, 'Archived constraint helper is missing'
    model = importlib.util.module_from_spec(spec)
    sys.modules[name] = model
    spec.loader.exec_module(model)
    return model


def _curve(n):
    angle = np.arange(n) * (2. * np.pi / n)
    points = np.stack([1.2 * np.cos(angle), .8 * np.sin(angle)], axis=-1)
    points += np.random.default_rng(122 + n).normal(scale=.02 / n, size=points.shape)
    return points.astype(np.float32)


def _jacobian(points):
    points = np.asarray(points, np.float64)
    edge = np.roll(points, -1, axis=0) - points
    unit = edge / np.linalg.norm(edge, axis=1)[:, None]
    n = len(points)
    jacobian = np.zeros((n, 2 * n), dtype=np.float64)
    for i in range(n):
        jacobian[i, 2 * i:2 * i + 2] = -unit[i]
        j = (i + 1) % n
        jacobian[i, 2 * j:2 * j + 2] = unit[i]
    return jacobian


def _dense_project(points, field):
    jacobian = _jacobian(points)
    vector = np.asarray(field, np.float64).reshape(-1)
    return (vector - jacobian.T @ np.linalg.solve(jacobian @ jacobian.T,
                                                  jacobian @ vector)).reshape(-1, 2)


def _project(points, field, dev='cpu', *, inplace=False, frozen=None):
    model = _model()
    points = np.asarray(points, np.float32)
    field = np.asarray(field, np.float32)
    if points.ndim == 2:
        points = points[None]
        field = field[None]
    e, n, _ = points.shape
    reference = wp.array(points.reshape(-1, 2), dtype=wp.vec2f, device=dev)
    values = wp.array(field.reshape(-1, 2), dtype=wp.vec2f, device=dev)
    out = values if inplace else wp.full(e * n, wp.vec2f(7., 11.), dtype=wp.vec2f, device=dev)
    frozen = wp.array(np.zeros(e, np.int32) if frozen is None else np.asarray(frozen, np.int32),
                      dtype=wp.int32, device=dev)
    status = wp.full(e, 9, dtype=wp.int32, device=dev)
    workspace = model.alloc_edge_constraint_workspace(e, n, dev)
    model.project_edges(LaunchContext.current(dev), reference, values, n, frozen,
                        workspace, status, out)
    return out.numpy().reshape(e, n, 2), status.numpy(), reference.numpy()


@pytest.mark.parametrize('n', [4, 16, 64, 256])
@pytest.mark.parametrize('dev', DEVS)
def test_projector_matches_independent_dense_constraint_projection(n, dev):
    points = _curve(n)
    field = np.random.default_rng(75 + n).normal(size=points.shape).astype(np.float32)
    expected = _dense_project(points, field)
    actual, status, reference = _project(points, field, dev)
    assert status.tolist() == [1]
    np.testing.assert_allclose(actual[0], expected, rtol=2e-5, atol=2e-6)
    assert np.linalg.norm(_jacobian(points) @ actual[0].astype(float).ravel()) < 2e-6 * np.linalg.norm(field)
    assert reference.tobytes() == points.tobytes()


@pytest.mark.parametrize('dev', DEVS)
def test_projector_caches_rhs_before_in_place_output(dev):
    points = _curve(64)
    field = np.random.default_rng(19).normal(size=points.shape).astype(np.float32)
    actual, status, _ = _project(points, field, dev, inplace=True)
    expected = _dense_project(points, field)
    assert status.tolist() == [1]
    np.testing.assert_allclose(actual[0], expected, rtol=2e-5, atol=2e-6)
    again, status, _ = _project(points, actual[0], dev)
    assert status.tolist() == [1]
    np.testing.assert_allclose(again[0], actual[0], rtol=2e-5, atol=3e-7)


@pytest.mark.parametrize('dev', DEVS)
def test_projected_spd_metric_is_a_tangent_descent_direction(dev):
    points = _curve(64)
    gradient = np.random.default_rng(27).normal(size=points.shape).astype(np.float32)
    first, status, _ = _project(points, gradient, dev)
    assert status.tolist() == [1]
    n = len(points)
    k = np.arange(n // 2 + 1)
    eigen = (np.sin(np.pi * k / n) / np.sin(np.pi / n)) ** 2
    spectrum = 1. / (eigen ** (5. / 3.) + 1e-3)
    spectrum[0] = 0.
    middle = np.fft.irfft(np.fft.rfft(first[0].astype(float), axis=0)
                          * spectrum[:, None], n=n, axis=0)
    direction, status, _ = _project(points, middle.astype(np.float32), dev)
    assert status.tolist() == [1]
    value = float(np.sum(gradient.astype(float) * direction[0]))
    expected = float(np.sum(first[0].astype(float) * middle))
    assert value > 0.
    assert value == pytest.approx(expected, rel=3e-6, abs=2e-7)
    assert np.linalg.norm(_jacobian(points) @ direction[0].astype(float).ravel()) < 2e-6 * np.linalg.norm(direction)


def _retract(reference, candidate, *, scale=1., tol=2e-5, iterations=8, backtracks=8, dev='cpu'):
    model = _model()
    reference = np.asarray(reference, np.float32)
    candidate = np.asarray(candidate, np.float32)
    n = len(reference)
    x = wp.array(reference, dtype=wp.vec2f, device=dev)
    out = wp.array(candidate, dtype=wp.vec2f, device=dev)
    scales = wp.full(1, scale, dtype=wp.float64, device=dev)
    frozen = wp.zeros(1, dtype=wp.int32, device=dev)
    status = wp.full(1, 9, dtype=wp.int32, device=dev)
    workspace = model.alloc_edge_constraint_workspace(1, n, dev)
    model.retract_edges(LaunchContext.current(dev), x, out, n, scales, frozen,
                        workspace, tol, iterations, backtracks, status)
    return out.numpy(), status.numpy(), x.numpy()


def _edge_error(reference, actual, scale):
    target = np.linalg.norm(np.roll(reference.astype(float), -1, axis=0)
                            - reference.astype(float), axis=1) * scale
    actual_length = np.linalg.norm(np.roll(actual.astype(float), -1, axis=0)
                                   - actual.astype(float), axis=1)
    return np.max(np.abs(actual_length - target) / target)


@pytest.mark.parametrize('dev', DEVS)
def test_newton_retraction_restores_each_reference_edge_not_only_perimeter(dev):
    points = _curve(64)
    field = np.random.default_rng(115).normal(size=points.shape)
    tangent = _dense_project(points, field)
    tangent /= np.linalg.norm(tangent, axis=1).max()
    mean = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).mean()
    candidate = points + .25 * mean * tangent
    actual, status, reference = _retract(points, candidate, dev=dev)
    assert status.tolist() == [1]
    assert _edge_error(points, actual, 1.) <= 2e-5
    assert reference.tobytes() == points.tobytes()


@pytest.mark.parametrize('n', [64, 256, 1024])
@pytest.mark.parametrize('dev', DEVS)
def test_growth_retraction_damps_newton_and_checks_actual_float32_edge_residual(n, dev):
    points = _curve(n)
    actual, status, reference = _retract(points, points, scale=1.012, dev=dev)
    assert status.tolist() == [1]
    assert _edge_error(points, actual, 1.012) <= 2e-5
    assert reference.tobytes() == points.tobytes()


def test_exhausted_retraction_reports_failure_and_preserves_reference():
    points = _curve(64)
    actual, status, reference = _retract(points, points, scale=2., iterations=1, backtracks=1)
    assert status.tolist() == [0]
    assert _edge_error(points, actual, 2.) > 2e-5
    assert reference.tobytes() == points.tobytes()


@pytest.mark.parametrize('kind', ['duplicate', 'collinear', 'near_collinear', 'nonfinite'])
def test_singular_or_invalid_reference_fails_closed(kind):
    if kind == 'duplicate':
        points = np.array([[0., 0.], [1., 0.], [1., 0.], [0., 1.]], np.float32)
    elif kind in ('collinear', 'near_collinear'):
        y = 0. if kind == 'collinear' else 1e-10
        points = np.array([[-1., 0.], [0., y], [1., 0.], [0., -y]], np.float32)
    else:
        points = _curve(4)
        points[0, 0] = np.nan
    field = np.ones_like(points)
    actual, status, reference = _project(points, field)
    assert status.tolist() == [0]
    assert np.isfinite(actual).all()
    assert reference.tobytes() == points.tobytes()
    _, status, reference = _retract(points, points, scale=1.01)
    assert status.tolist() == [0]
    assert reference.tobytes() == points.tobytes()


@pytest.mark.parametrize('kwargs', [{'scale': 0.}, {'tol': -1.}, {'iterations': 0}, {'backtracks': 0}])
def test_invalid_or_empty_retraction_budget_fails_closed(kwargs):
    points = _curve(16)
    _, status, reference = _retract(points, points, **kwargs)
    assert status.tolist() == [0]
    assert reference.tobytes() == points.tobytes()


@pytest.mark.parametrize('dev', DEVS)
def test_frozen_rows_and_batch_independence(dev):
    points = np.stack([_curve(32), _curve(32) + [2., -1.]])
    field = np.random.default_rng(43).normal(size=points.shape).astype(np.float32)
    actual, status, _ = _project(points, field, dev, frozen=[0, 1])
    assert status.tolist() == [1, 9]
    np.testing.assert_array_equal(actual[1], np.tile([7., 11.], (32, 1)))
    alone, single_status, _ = _project(points[0], field[0], dev)
    assert single_status.tolist() == [1]
    assert alone[0].tobytes() == actual[0].tobytes()


@pytest.mark.parametrize('dev', DEVS)
def test_projector_capture_replay_uses_preallocated_workspace_without_sync_or_allocation(dev):
    model = _model()
    points = _curve(32)
    field = np.random.default_rng(1).normal(size=points.shape).astype(np.float32)
    reference = wp.array(points, dtype=wp.vec2f, device=dev)
    values = wp.array(field, dtype=wp.vec2f, device=dev)
    out = wp.empty(len(points), dtype=wp.vec2f, device=dev)
    frozen = wp.zeros(1, dtype=wp.int32, device=dev)
    status = wp.zeros(1, dtype=wp.int32, device=dev)
    workspace = model.alloc_edge_constraint_workspace(1, len(points), dev)
    ctx = LaunchContext.current(dev)
    model.project_edges(ctx, reference, values, len(points), frozen, workspace, status, out)
    expected = out.numpy().copy()
    with wp.ScopedCapture(device=dev) as capture:
        with no_host_sync(), no_allocation():
            model.project_edges(ctx, reference, values, len(points), frozen, workspace, status, out)
    out.fill_(wp.vec2f(-7., -11.))
    wp.capture_launch(capture.graph)
    assert out.numpy().tobytes() == expected.tobytes()
    assert status.numpy().tolist() == [1]


def test_sub_float32_growth_scale_is_not_silently_treated_as_identity():
    # The local Newton correction rounds away at these exact binary coordinates.
    # Honest failure is required; rounding the requested scale to1 would report success.
    square = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], np.float32)
    requested_scale = 1. + 2. ** -24
    assert requested_scale != float(np.float32(requested_scale))
    actual, status, reference = _retract(square, square, scale=requested_scale, tol=1e-10)
    assert status.tolist() == [0]
    assert _edge_error(square, actual, requested_scale) > 1e-10
    assert reference.tobytes() == square.tobytes()
