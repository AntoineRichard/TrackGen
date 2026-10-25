"""Parallel objective rows retain the current equations and precision budget."""
from __future__ import annotations


import numpy as np
import pytest
import warp as wp

from tests.conftest import DEVS
from tests.test_repulsive_armijo_precision import _fixture, _reference
from tests import _repulsive_serial_reference as serial
from track_gen._src import repulsive_parallel_energy as parallel
from track_gen.exec import LaunchContext
from track_gen.testing.contract import no_allocation, no_host_sync



@wp.kernel
def _serial_objective(x: wp.array(dtype=wp.vec2f), y: wp.array(dtype=wp.vec2f), n: int,
                      fraction: float, alpha: float, beta: float, eps: float, p_exp: float,
                      samples: wp.array(dtype=wp.vec2f),
                      mass: wp.array(dtype=wp.float32), m: int,
                      centers: wp.array(dtype=wp.vec2f), radii: wp.array(dtype=wp.float32), k: int,
                      radius: float, margin: wp.array(dtype=wp.float32),
                      total: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64)):
    e = wp.tid()
    value, magnitude = serial._proposal_energy_parts(
        x, y, e, n, fraction, alpha, beta, eps, samples, mass, m, p_exp,
        centers, radii, k, radius, margin[e])
    total[e] = value
    scale[e] = wp.max(magnitude, wp.float64(wp.pow(radius, 2. + 2. * p_exp)))


def _data(points, dev, *, circle=False, alpha=3., beta=6.):
    e, n, _ = points.shape
    if circle:
        _, samples, mass, radius, margin, _ = _fixture(.173)
        centers = np.full((1, 2), np.nan, np.float32)
        radii = np.zeros(1, np.float32)
    else:
        radius, margin = 3., .05
        samples = np.array([[2., .1], [float('nan'), float('nan')], [-1.9, 1.3]], np.float32)
        mass = np.array([.2, 0., .3], np.float32)
        centers = np.array([[2., .1], [-1.9, 1.3]], np.float32)
        radii = np.array([.08, .12], np.float32)
    samples_wp = wp.array(np.tile(samples, (e, 1)), dtype=wp.vec2f, device=dev)
    mass_wp = wp.array(np.tile(mass, e), dtype=wp.float32, device=dev)
    centers_wp = wp.array(np.tile(centers, (e, 1)), dtype=wp.vec2f, device=dev)
    radii_wp = wp.array(np.tile(radii, e), dtype=wp.float32, device=dev)
    margins = wp.full(e, margin, dtype=wp.float32, device=dev)
    model = parallel.EnergyModel(alpha, beta, 1e-4, samples_wp, mass_wp, len(mass), -.5 * (beta-alpha),
                                 centers_wp, radii_wp, len(radii), radius, margins)
    return parallel, model


def _call(points, proposal, fraction, dev, *, circle=False, active=None, nmax=None, alpha=3., beta=6.):
    e, n, _ = points.shape
    parallel, model = _data(points, dev, circle=circle, alpha=alpha, beta=beta)
    workspace = parallel.alloc_energy_workspace(e, n if nmax is None else nmax, dev)
    x = wp.array(points.reshape(-1, 2), dtype=wp.vec2f, device=dev)
    y = wp.array(proposal.reshape(-1, 2), dtype=wp.vec2f, device=dev)
    mask = wp.array(np.ones(e, np.int32) if active is None else active, dtype=wp.int32, device=dev)
    total = wp.full(e, 17., dtype=wp.float64, device=dev)
    scale = wp.full(e, 19., dtype=wp.float64, device=dev)
    ctx = LaunchContext.for_call(dev)
    parallel.evaluate_energy(ctx, x, y, n, fraction, model, mask, workspace, total, scale)
    reference = wp.zeros(e, dtype=wp.float64, device=dev)
    reference_scale = wp.zeros(e, dtype=wp.float64, device=dev)
    ctx.launch(_serial_objective, e,
               [x, y, n, fraction, model.alpha, model.beta, model.eps, model.p_exp, model.obs_pts, model.obs_mw, model.m_obs,
                model.disc_centers, model.disc_radii, model.k, model.r_dom, model.margin,
                reference, reference_scale])
    return total.numpy(), scale.numpy(), reference.numpy(), reference_scale.numpy()


@pytest.mark.parametrize('dev', DEVS)
@pytest.mark.parametrize('n,fraction,alpha,beta', [(16, 0., 3., 6.), (64, .125, 3., 6.),
                                                    (256, 1., 3., 6.), (32, .125, 1.5, 3.75),
                                                    (32, .125, 2., 4.5)])
def test_row_reduction_matches_serial_with_derived_double_summation_budget(dev, n, fraction, alpha, beta):
    angle = np.arange(n) * (2 * np.pi / n)
    first = np.stack([.6 * np.cos(angle), .45 * np.sin(angle)], axis=-1)
    second = np.stack([.5 * np.cos(angle), .55 * np.sin(angle)], axis=-1)
    points = np.stack([first, first * .9]).astype(np.float32)
    proposal = np.stack([second, second * .9]).astype(np.float32)
    total, scale, reference, reference_scale = _call(points, proposal, fraction, dev, nmax=256, alpha=alpha, beta=beta)
    # The individual FP32 primitives are unchanged. Both sequential and grouped
    # FP64 sums obey gamma_K times their absolute-term magnitude. This bound
    # allows only their association error, not a new FP32 reduction floor.
    k = n * (n - 3) + n * 3 + 6 * n
    gamma = k * (np.finfo(np.float64).eps / 2) / (1 - k * (np.finfo(np.float64).eps / 2))
    budget = 2 * gamma * np.maximum(scale, reference_scale)
    assert np.all(np.abs(total - reference) <= budget)
    assert np.all(np.abs(scale - reference_scale) <= budget)


@pytest.mark.parametrize('dev', DEVS)
@pytest.mark.parametrize('phase', [0., .173])
def test_row_objective_preserves_near_stationary_precision_and_sign(dev, phase):
    points, samples, mass, radius, margin, eps = _fixture(phase)
    actual, _, _, _ = _call(points, points, 0., dev, circle=True)
    expected = np.array([_reference(row, samples, mass, radius, margin, eps) for row in points])
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-5)
    assert actual[1] > actual[0]
    assert actual[1] - actual[0] == pytest.approx(expected[1] - expected[0], rel=0., abs=1e-5)


@pytest.mark.parametrize('dev', DEVS)
def test_masks_and_poisoned_partials_are_safe_across_capture_replay(dev):
    n, e = 16, 3
    angle = np.arange(n) * (2 * np.pi / n)
    points = np.tile(np.stack([.6 * np.cos(angle), .45 * np.sin(angle)], axis=-1), (e, 1, 1)).astype(np.float32)
    parallel, model = _data(points, dev)
    workspace = parallel.alloc_energy_workspace(e, 64, dev)
    x = wp.array(points.reshape(-1, 2), dtype=wp.vec2f, device=dev)
    mask = wp.array(np.array([1, 0, 1], np.int32), dtype=wp.int32, device=dev)
    total = wp.full(e, 17., dtype=wp.float64, device=dev)
    scale = wp.full(e, 19., dtype=wp.float64, device=dev)
    ctx = LaunchContext.for_call(dev)
    def evaluate():
        parallel.evaluate_energy(ctx, x, x, n, 0., model, mask, workspace, total, scale)
    evaluate()
    with no_allocation(), no_host_sync():
        evaluate()
    initial = total.numpy().copy()
    assert initial[1] == 17.
    for name in ('tp_rows', 'obs_rows', 'barrier_rows', 'barrier_abs_rows'):
        assert np.all(getattr(workspace, name).numpy()[n:2*n] == 0.)
    with wp.ScopedCapture(device=dev) as capture:
        evaluate()
    total.fill_(17.); scale.fill_(19.)
    for name in ('tp_rows', 'obs_rows', 'barrier_rows', 'barrier_abs_rows'):
        getattr(workspace, name).fill_(float('nan'))
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(total.numpy(), initial)
    mask.assign(np.array([0, 1, 0], np.int32))
    total.fill_(17.); scale.fill_(19.)
    wp.capture_launch(capture.graph)
    assert total.numpy().tolist() == [17., initial[0], 17.]

    mask.assign(np.zeros(e, np.int32))
    total.fill_(17.); scale.fill_(19.)
    wp.capture_launch(capture.graph)
    assert total.numpy().tolist() == [17.] * e
    assert scale.numpy().tolist() == [19.] * e
    for name in ('tp_rows', 'obs_rows', 'barrier_rows', 'barrier_abs_rows'):
        assert np.all(getattr(workspace, name).numpy()[:e*n] == 0.)


@pytest.mark.parametrize('dev', DEVS)
def test_invalid_barrier_energy_stays_nonfinite(dev):
    n = 16
    angle = np.arange(n) * (2 * np.pi / n)
    points = np.stack([.6 * np.cos(angle), .45 * np.sin(angle)], axis=-1).astype(np.float32)
    points[0] = [2., .1]  # exactly inside one of the real filled discs
    total, scale, reference, reference_scale = _call(points[None], points[None], 0., dev)
    assert not np.isfinite(total[0])
    assert not np.isfinite(reference[0])
    assert not np.isfinite(scale[0])
    assert not np.isfinite(reference_scale[0])
