"""Analytic barriers and accepted-step regressions for Repulsive growth."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import warp_generate_repulsive as rep
from tests.conftest import DEVS
from track_gen.exec import LaunchContext
from track_gen._src.repulsive_parallel_energy import EnergyModel, evaluate_energy
from track_gen._src import repulsive_parallel_proposals as proposals


def _segment_distances(points, center):
    delta = np.roll(points, -1, axis=0) - points
    t = np.clip(np.sum((center - points) * delta, axis=1) /
                np.maximum(np.sum(delta * delta, axis=1), 1e-30), 0., 1.)
    return np.linalg.norm(points + t[:, None] * delta - center, axis=1)


def _raw_gaps(gen):
    raw = gen._scratch.gen_centerline.numpy().reshape(gen.num_envs, -1, 2)
    scratch = gen._scratch.gen.scratch
    pts = scratch['scratch:generator/obs_pts'].numpy().reshape(gen.num_envs, -1, 2)
    mass = scratch['scratch:generator/obs_mw'].numpy().reshape(gen.num_envs, -1)
    domain = rep._RepulsiveScalars(gen.config).r_dom
    gaps = []
    for e, p in enumerate(raw):
        row = [domain - np.linalg.norm(p, axis=1).max()]
        for j in range(96, pts.shape[1], 12):
            if mass[e, j] != 0:
                c = (pts[e, j] + pts[e, j + 6]) * .5
                r = np.linalg.norm(pts[e, j] - c)
                row.append(_segment_distances(p, c).min() - r)
        gaps.append(row)
    return gaps


@pytest.mark.parametrize('tau', [.4, .1])
def test_seed7_raw_never_silently_crosses_analytic_barriers(tau):
    cfg = TrackGenConfig(generator='repulsive', device='cpu', num_envs=1,
                         repulsive_tau=tau, on_invalid='keep')
    seeds = wp.array(np.array([7], dtype=np.int32), dtype=wp.int32, device='cpu')
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=1, device='cpu'))
    gen.generate()
    assert min(_raw_gaps(gen)[0]) >= -2e-6


@pytest.mark.parametrize('n', [16, 64, 256])
def test_sobolev_metric_has_positive_resolution_normalized_modes_and_no_translation(n):
    s = (6. - 1.) / 3.
    row = rep._sobolev_circulant_row(n, s, 1e-3)
    modes = np.fft.rfft(row).real
    assert abs(modes[0]) < 2e-6
    assert (modes[1:] > 0).all()
    assert modes[1] == pytest.approx(1. / 1.001, rel=2e-5)
    lam = (2. - 2. * np.cos(2. * np.pi * np.arange(n // 2 + 1) / n))
    expected = 1. / ((lam[1:] / lam[1]) ** s + 1e-3)
    np.testing.assert_allclose(modes[1:], expected, rtol=.025, atol=2e-8)


@pytest.mark.parametrize('beta', [3., 2.])
def test_obstacle_force_exponent_must_be_repulsive(beta):
    with pytest.raises(ValueError, match='repulsive_beta must be > repulsive_alpha'):
        TrackGenerator(TrackGenConfig(generator='repulsive', device='cpu', num_envs=1,
                                      repulsive_beta=beta))


@pytest.mark.parametrize('alpha,beta,match', [(1., 3., 'repulsive_alpha must be > 1'),
                                             (3., 4., 'repulsive_beta must be >= repulsive_alpha'),
                                             (3., 7., 'repulsive_beta must be < 2')])
def test_energy_exponents_require_the_supported_sobolev_order(alpha, beta, match):
    with pytest.raises(ValueError, match=match):
        TrackGenerator(TrackGenConfig(generator='repulsive', device='cpu', num_envs=1,
                                      repulsive_alpha=alpha, repulsive_beta=beta))


def test_default_seed7_is_a_usable_finished_road():
    cfg = TrackGenConfig(generator='repulsive', device='cpu', num_envs=1)
    seeds = wp.array(np.array([7], dtype=np.int32), dtype=wp.int32, device='cpu')
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=1, device='cpu'))
    track = gen.generate()
    assert track.valid.numpy()[0] == 1, track.fail_mask.numpy()


@pytest.mark.parametrize('movement,stationarity,blocked,expected',
                         [(.4, .001, 0, 0), (.001, .4, 0, 0),
                          (.0, .001, 1, 0), (.001, .001, 0, 1)])
def test_area_stability_requires_small_unblocked_shape_motion(movement, stationarity,
                                                            blocked, expected):
    n = 16
    a = np.arange(n) * (2 * np.pi / n)
    points = np.stack([np.cos(a), np.sin(a)], axis=-1).astype(np.float32)
    center = wp.array(points, dtype=wp.vec2f, device='cpu')
    perimeter = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
    nxt = np.roll(points, -1, axis=0)
    area = .5 * abs((points[:, 0] * nxt[:, 1] - points[:, 1] * nxt[:, 0]).sum())
    def f(value):
        return wp.array(np.array([value], np.float32), dtype=wp.float32, device='cpu')
    def i(value):
        return wp.array(np.array([value], np.int32), dtype=wp.int32, device='cpu')
    def fd(value):
        return wp.array(np.array([value], np.float64), dtype=wp.float64, device='cpu')
    frozen = i(0)
    wp.launch(rep._freeze_update_k, dim=1,
              inputs=[center, n, i(1), .05, fd(area), frozen, fd(0), fd(perimeter),
                      f(movement), f(stationarity), i(blocked), 1.], device='cpu')
    assert frozen.numpy()[0] == expected


def test_step_vanishes_with_a_tiny_projected_gradient():
    n = 16
    x = wp.zeros(n, dtype=wp.vec2f, device='cpu')
    mean = wp.zeros(1, dtype=wp.vec2d, device='cpu')
    length = wp.full(1, .1, dtype=wp.float64, device='cpu')
    frozen = wp.zeros(1, dtype=wp.int32, device='cpu')
    displacement = []
    for magnitude in (1e-4, 1e-6):
        values = np.zeros((n, 2), dtype=np.float32)
        values[::2, 0] = magnitude
        values[1::2, 0] = -magnitude
        gradient = wp.array(values, dtype=wp.vec2d, device='cpu')
        maximum = wp.full(1, magnitude, dtype=wp.float32, device='cpu')
        proposal = wp.zeros(n, dtype=wp.vec2f, device='cpu')
        wp.launch(rep._step_k, dim=(1, n),
                  inputs=[x, gradient, mean, length, maximum, .4, 1., n, frozen, proposal],
                  device='cpu')
        displacement.append(np.linalg.norm(proposal.numpy(), axis=-1).max())
    assert displacement[1] < .02 * displacement[0]


def test_periodic_resampling_cannot_inject_a_rougher_shape():
    n = 32
    angle = np.arange(n) * (2 * np.pi / n)
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    old = (unit * .5).astype(np.float32)
    radii = np.where(np.arange(n) % 2 == 0, .7, .3)
    rough = (unit * radii[:, None]).astype(np.float32)
    center = wp.array(old, dtype=wp.vec2f, device='cpu')
    proposal = wp.array(rough, dtype=wp.vec2f, device='cpu')
    dc = wp.zeros(1, dtype=wp.vec2f, device='cpu')
    dr = wp.zeros(1, dtype=wp.float32, device='cpu')
    margin = wp.zeros(1, dtype=wp.float32, device='cpu')
    frozen = wp.zeros(1, dtype=wp.int32, device='cpu')
    movement = wp.zeros(1, dtype=wp.float32, device='cpu')
    model = EnergyModel(3., 6., 1e-4, dc, dr, 0, -1.5, dc, dr, 0, 3., margin)
    workspace = proposals.alloc_proposal_workspace(1, n, 'cpu')
    proposals.accept_resample_inplace(LaunchContext.current('cpu'), center, proposal, n,
                                      model, frozen, movement, 0, workspace)
    assert center.numpy().tobytes() == old.tobytes()


def test_satisfied_length_constraint_is_an_exact_noop():
    n = 32
    angle = np.arange(n) * (2 * np.pi / n)
    points = (np.stack([.5 * np.cos(angle), .7 * np.sin(angle)], axis=-1) + [.137, -.083]).astype(np.float32)
    center = wp.array(points, dtype=wp.vec2f, device='cpu')
    target = wp.zeros(1, dtype=wp.float64, device='cpu')
    wp.launch(rep._perimeter_k, dim=1, inputs=[center, n, target], device='cpu')
    proposal = wp.zeros(n, dtype=wp.vec2f, device='cpu')
    dc = wp.zeros(1, dtype=wp.vec2f, device='cpu')
    dr = wp.zeros(1, dtype=wp.float32, device='cpu')
    margin = wp.zeros(1, dtype=wp.float32, device='cpu')
    frozen = wp.zeros(1, dtype=wp.int32, device='cpu')
    model = EnergyModel(3., 6., 1e-4, dc, dr, 0, -1.5, dc, dr, 0, 3., margin)
    workspace = proposals.alloc_proposal_workspace(1, n, 'cpu')
    proposals.grow_feasible_inplace(LaunchContext.current('cpu'), center, proposal, n,
                                   target, .012, model, frozen, workspace)
    assert center.numpy().tobytes() == points.tobytes()


def test_perimeter_reduction_resolves_float32_coordinate_geometry():
    n = 256
    angle = np.arange(n) * (2 * np.pi / n)
    points = (np.stack([.5 * np.cos(angle), .7 * np.sin(angle)], axis=-1) + [.137, -.083]).astype(np.float32)
    center = wp.array(points, dtype=wp.vec2f, device='cpu')
    perimeter = wp.zeros(1, dtype=wp.float64, device='cpu')
    wp.launch(rep._perimeter_k, dim=1, inputs=[center, n, perimeter], device='cpu')
    exact = points.astype(np.float64)
    reference = np.linalg.norm(np.roll(exact, -1, axis=0) - exact, axis=-1).sum()
    assert abs(float(perimeter.numpy()[0]) - reference) < 2e-12


def test_feasible_target_reaching_unfrozen_budget_result_is_valid():
    cfg = TrackGenConfig(generator='repulsive', device='cpu', num_envs=3,
                         repulsive_grow_mult_min=2., repulsive_grow_mult_max=3.)
    seeds = wp.array(np.array([7, 23, 41], dtype=np.int32), dtype=wp.int32, device='cpu')
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=3, device='cpu'))
    track = gen.generate()
    assert gen._scratch.gen.scratch['scratch:generator/frozen'].numpy().tolist() == [0, 0, 0]
    assert gen.program.tap('gen.valid').numpy().tolist() == [1, 1, 1]
    assert track.valid.numpy().tolist() == [1, 1, 1]
    assert track.fail_mask.numpy().tolist() == [0, 0, 0]


def test_target_shortfall_cannot_be_reported_valid():
    n = 32
    angle = np.arange(n) * (2 * np.pi / n)
    points = np.stack([.5 * np.cos(angle), .5 * np.sin(angle)], axis=-1).astype(np.float32)
    center = wp.array(points, dtype=wp.vec2f, device='cpu')
    length = np.linalg.norm(np.roll(points.astype(float), -1, axis=0) - points, axis=-1).sum()
    target = wp.array(np.array([length / .98], dtype=np.float64), dtype=wp.float64, device='cpu')
    dc = wp.zeros(1, dtype=wp.vec2f, device='cpu')
    dr = wp.zeros(1, dtype=wp.float32, device='cpu')
    margin = wp.zeros(1, dtype=wp.float32, device='cpu')
    failed = wp.zeros(1, dtype=wp.int32, device='cpu')
    valid = wp.zeros(1, dtype=wp.int32, device='cpu')
    wp.launch(rep._finish_valid_k, dim=1,
              inputs=[center, n, dc, dr, 0, 3., margin, target, failed, valid], device='cpu')
    assert valid.numpy().tolist() == [0]


def test_quantized_unchanged_candidate_is_not_accepted_as_progress():
    n = 16
    angle = np.arange(n) * (2 * np.pi / n)
    points = np.stack([.5 * np.cos(angle), .5 * np.sin(angle)], axis=-1).astype(np.float32)
    center = wp.array(points, dtype=wp.vec2f, device='cpu')
    proposal = wp.array(points.copy(), dtype=wp.vec2f, device='cpu')
    dc = wp.zeros(1, dtype=wp.vec2f, device='cpu')
    dr = wp.zeros(1, dtype=wp.float32, device='cpu')
    margin = wp.zeros(1, dtype=wp.float32, device='cpu')
    frozen = wp.zeros(1, dtype=wp.int32, device='cpu')
    energy = wp.zeros(1, dtype=wp.float64, device='cpu')
    scale = wp.zeros(1, dtype=wp.float64, device='cpu')
    model = EnergyModel(3., 6., 1e-4, dc, dr, 0, -1.5, dc, dr, 0, 3., margin)
    workspace = proposals.alloc_proposal_workspace(1, n, 'cpu')
    evaluate_energy(LaunchContext.current('cpu'), center, center, n, 0., model,
                    wp.ones(1, dtype=wp.int32, device='cpu'), workspace.energy_workspace,
                    energy, scale)
    msl = wp.full(1, .1, dtype=wp.float32, device='cpu')
    movement = wp.zeros(1, dtype=wp.float32, device='cpu')
    blocked = wp.zeros(1, dtype=wp.int32, device='cpu')
    slope = wp.full(1, -1., dtype=wp.float64, device='cpu')
    numerical = wp.zeros(1, dtype=wp.int32, device='cpu')
    failed = wp.zeros(1, dtype=wp.int32, device='cpu')
    accepted = wp.zeros(1, dtype=wp.int32, device='cpu')
    proposals.accept_shape_inplace(LaunchContext.current('cpu'), center, proposal, n,
                                  model, energy, msl, frozen, movement, blocked, slope,
                                  numerical, failed, accepted, workspace)
    assert accepted.numpy().tolist() == [0]
    assert blocked.numpy().tolist() == [1]
    assert center.numpy().tobytes() == points.tobytes()


def test_stationarity_is_independent_of_tau_and_quantized_proposal_motion():
    assert hasattr(rep, '_direction_metrics_k'), 'Stopping must measure the pre-step projected field'
    n = 16
    values = np.tile([[1., 0.], [-1., 0.]], (n // 2, 1)).astype(np.float32)
    field = wp.array(values, dtype=wp.vec2d, device='cpu')
    gradient = wp.array(values, dtype=wp.vec2f, device='cpu')
    mean = wp.zeros(1, dtype=wp.vec2d, device='cpu')
    perimeter = wp.full(1, 16., dtype=wp.float64, device='cpu')
    energy_scale = wp.ones(1, dtype=wp.float64, device='cpu')
    frozen = wp.zeros(1, dtype=wp.int32, device='cpu')
    metrics = []
    for tau in (.4, 1e-20):
        gmax = wp.zeros(1, dtype=wp.float32, device='cpu')
        msl = wp.zeros(1, dtype=wp.float32, device='cpu')
        stationarity = wp.zeros(1, dtype=wp.float32, device='cpu')
        slope = wp.zeros(1, dtype=wp.float64, device='cpu')
        predicted = wp.zeros(1, dtype=wp.float64, device='cpu')
        numerical = wp.zeros(1, dtype=wp.int32, device='cpu')
        failed = wp.zeros(1, dtype=wp.int32, device='cpu')
        wp.launch(rep._direction_metrics_k, dim=1,
                  inputs=[field, mean, gradient, perimeter, energy_scale, n, tau, 1., frozen,
                          gmax, msl, stationarity, slope, predicted, numerical, failed,
                          wp.zeros(n, dtype=wp.vec2f, device='cpu'),
                          wp.zeros(1, dtype=wp.float64, device='cpu'),
                          wp.ones(1, dtype=wp.float64, device='cpu'),
                          wp.array(rep._sobolev_circulant_row(n, 5./3., 1e-3), dtype=wp.float64, device='cpu')],
                  device='cpu')
        metrics.append(float(stationarity.numpy()[0]))
        assert metrics[-1] > 1.
        assert numerical.numpy().tolist() == [0]
        assert failed.numpy().tolist() == [0]
    assert metrics[0] == metrics[1]


@pytest.mark.parametrize('dev', DEVS)
def test_near_circle_length_projection_retains_positive_metric(dev):
    cfg = TrackGenConfig(generator='repulsive', device=dev, num_envs=2,
                         num_points=64, repulsive_stages=(64,), half_width=.01,
                         repulsive_domain_init_ratio=1.15,
                         repulsive_grow_mult_min=1.0001, repulsive_grow_mult_max=1.001)
    seeds = wp.array(np.array([7, 23], np.int32), dtype=wp.int32, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=2, device=dev))
    gen.generate()
    scratch = gen._scratch.gen.scratch
    assert scratch['scratch:generator/metric_failed'].numpy().tolist() == [0, 0]
    assert np.isfinite(scratch['scratch:generator/stationarity'].numpy()).all()


@pytest.mark.parametrize('on_invalid', ['keep', 'poison'])
def test_high_growth_raw_success_does_not_bypass_finished_band_rejection(on_invalid):
    # CPU integration repro: this target reaches a feasible raw loop, then the
    # shared tail pushes its width band into a disc. Optimizer trajectories may
    # differ by device; deterministic finished-band fixtures cover DEVS directly.
    dev = 'cpu'
    cfg = TrackGenConfig(generator='repulsive', device=dev, num_envs=1,
                         repulsive_grow_mult_min=3., repulsive_grow_mult_max=3.5,
                         on_invalid=on_invalid)
    seeds = wp.array(np.array([7], np.int32), dtype=wp.int32, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=1, device=dev))
    track = gen.generate()
    assert gen.program.tap('gen.valid').numpy().tolist() == [1]
    assert track.valid.numpy().tolist() == [0]
    assert int(track.fail_mask.numpy()[0]) & 256
    if on_invalid == 'poison':
        assert np.isnan(track.center.numpy()).all()
        assert track.count.numpy().tolist() == [0]
    else:
        assert np.isfinite(track.center.numpy()).any()
        assert track.count.numpy()[0] > 0


@pytest.mark.parametrize('dev', DEVS)
@pytest.mark.parametrize('direction_sign', [1., -1.])
def test_stationarity_uses_projected_residual_without_large_normal_cancellation(dev, direction_sign):
    n = 16
    signs = np.where(np.arange(n) % 2 == 0, 1., -1.)[:, None]
    field = wp.array(signs * np.array([[-1e-10, direction_sign * 1e-4]]), dtype=wp.vec2d, device=dev)
    gradient = wp.array(signs * np.array([[1e8, 1.]]), dtype=wp.vec2f, device=dev)
    mean = wp.zeros(1, dtype=wp.vec2d, device=dev)
    perimeter = wp.full(1, 16., dtype=wp.float64, device=dev)
    energy = wp.ones(1, dtype=wp.float64, device=dev)
    frozen = wp.zeros(1, dtype=wp.int32, device=dev)
    gmax = wp.zeros(1, dtype=wp.float32, device=dev)
    msl = wp.zeros(1, dtype=wp.float32, device=dev)
    stationarity = wp.zeros(1, dtype=wp.float32, device=dev)
    slope = wp.zeros(1, dtype=wp.float64, device=dev)
    predicted = wp.zeros(1, dtype=wp.float64, device=dev)
    numerical = wp.zeros(1, dtype=wp.int32, device=dev)
    failed = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(rep._direction_metrics_k, dim=1,
              inputs=[field, mean, gradient, perimeter, energy, n, .4, 1., frozen,
                      gmax, msl, stationarity, slope, predicted, numerical, failed,
                      wp.array(signs * np.array([[1., 0.]]), dtype=wp.vec2f, device=dev),
                      wp.full(1, 1e8, dtype=wp.float64, device=dev),
                      wp.ones(1, dtype=wp.float64, device=dev),
                      wp.array(rep._sobolev_circulant_row(n, 5./3., 1e-3), dtype=wp.float64, device=dev)], device=dev)
    assert failed.numpy().tolist() == [int(direction_sign < 0)]
    if direction_sign > 0:
        assert stationarity.numpy()[0] == pytest.approx(.16, rel=1e-6)
    else:
        assert failed.numpy().tolist() == [1]
        assert np.isinf(stationarity.numpy()[0])


@wp.kernel
def _core_wall_certificate_k(points: wp.array(dtype=wp.vec2f), radius: float, margin: float,
                             centers: wp.array(dtype=wp.vec2f), radii: wp.array(dtype=wp.float32),
                             out: wp.array(dtype=wp.int32)):
    out[0] = int(rep._proposal_clear(points, points, 0, 4, 1., centers, radii, 0,
                                      radius, margin, 0))


@pytest.mark.parametrize('dev', DEVS)
@pytest.mark.parametrize('radius,margin,point,expected', [
    (1e6, 0., (471861.03125, -881672.9375), 0),
    (1e6, 0., (999995., 0.), 0),
    (1e6, 123.5, (999850., 0.), 1),
    (1., 1.1, (0., 0.), 0),
    (float('inf'), 0., (0., 0.), 0),
    (float('nan'), 0., (0., 0.), 0),
])
def test_core_wall_certificate_uses_double_geometry_and_scaled_tolerance(dev, radius, margin,
                                                                       point, expected):
    # The first exactly stored f32 point lies outside R by .000766, although
    # its f32 norm rounds down to999999.9375. The same conservative relative
    # tolerance is required by raw-update and finished-road wall certificates.
    points = wp.array(np.array([point, (0., 0.), (0., 0.), (0., 0.)], np.float32),
                      dtype=wp.vec2f, device=dev)
    centers = wp.zeros(0, dtype=wp.vec2f, device=dev)
    radii = wp.zeros(0, dtype=wp.float32, device=dev)
    out = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(_core_wall_certificate_k, dim=1,
              inputs=[points, radius, margin, centers, radii, out], device=dev)
    assert out.numpy().tolist() == [expected]
