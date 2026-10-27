"""Parallel proposal stages preserve the serial acceptance and barrier rules."""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util

import numpy as np
import pytest
import warp as wp

from tests.conftest import DEVS
from tests import _repulsive_serial_reference as serial
from track_gen._src.repulsive_parallel_energy import EnergyModel
from track_gen.exec import LaunchContext


def _parallel():
    name = 'track_gen._src.repulsive_parallel_proposals'
    assert importlib.util.find_spec(name) is not None, 'The parallel proposal helper is missing'
    return importlib.import_module(name)


def _array(values, dtype, dev):
    return wp.array(np.asarray(values), dtype=dtype, device=dev)


def _curve(n, *, rough=False):
    a = np.arange(n) * (2. * np.pi / n)
    radius = .5 * (1. + .12 * np.cos(3. * a)) if rough else np.full(n, .5)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1).astype(np.float32)


def _model(e, dev, *, domain=3., margin=.02):
    return EnergyModel(3., 6., 1e-4, wp.zeros(0, dtype=wp.vec2f, device=dev),
                       wp.zeros(0, dtype=wp.float32, device=dev), 0, -1.5,
                       wp.zeros(0, dtype=wp.vec2f, device=dev),
                       wp.zeros(0, dtype=wp.float32, device=dev), 0, domain,
                       wp.full(e, margin, dtype=wp.float32, device=dev))


def _poison(workspace):
    for field in dataclasses.fields(workspace):
        value = getattr(workspace, field.name)
        if dataclasses.is_dataclass(value):
            _poison(value)
        elif isinstance(value, wp.array):
            value.fill_(57 if value.dtype == wp.int32 else float('nan'))


def _shape_inputs(dev):
    n, e = 12, 6
    old = np.tile(_curve(n, rough=True), (e, 1))
    candidate = np.tile(_curve(n), (e, 1))
    candidate[5*n:6*n] = old[5*n:6*n]  # a quantized/no-op proposal
    frozen = _array([0, 1, 0, 0, 0, 0], wp.int32, dev)
    numerical = _array([0, 0, 0, 1, 0, 0], wp.int32, dev)
    failed = _array([0, 0, 1, 0, 0, 0], wp.int32, dev)
    slope = _array([-1e-5, -1e-5, -1e-5, -1e-5, 0., -1e-5], wp.float64, dev)
    center = _array(old, wp.vec2f, dev)
    proposal = _array(candidate, wp.vec2f, dev)
    energy = wp.zeros(e, dtype=wp.float64, device=dev)
    scale = wp.zeros(e, dtype=wp.float64, device=dev)
    model = _model(e, dev)
    wp.launch(serial._shape_energy_k, e,
              inputs=[center, n, model.alpha, model.beta, model.eps, model.obs_pts,
                      model.obs_mw, model.m_obs, model.p_exp, model.disc_centers,
                      model.disc_radii, model.k, model.r_dom, model.margin,
                      wp.zeros(e, dtype=wp.int32, device=dev), energy, scale], device=dev)
    msl = wp.full(e, .25, dtype=wp.float32, device=dev)
    window = _array([.2, .3, .4, .5, .6, .7], wp.float32, dev)
    blocked = wp.full(e, 9, dtype=wp.int32, device=dev)
    accepted = wp.full(e, 7, dtype=wp.int32, device=dev)
    return dict(n=n, e=e, old=old, candidate=candidate, center=center, proposal=proposal,
                model=model, frozen=frozen, numerical=numerical, failed=failed,
                slope=slope, energy=energy, scale=scale, msl=msl, window=window,
                blocked=blocked, accepted=accepted)


def _serial_shape(data):
    d, m = data, data['model']
    wp.launch(serial._accept_shape_k, d['e'],
              inputs=[d['center'], d['proposal'], wp.empty(d['e']*d['n'], dtype=wp.vec2f, device=d['center'].device),
                      wp.empty(d['e']*d['n'], dtype=wp.vec2f, device=d['center'].device),
                      d['energy'], d['scale'], d['n'], m.alpha, m.beta, m.eps,
                      m.obs_pts, m.obs_mw, m.m_obs, m.p_exp, m.disc_centers, m.disc_radii,
                      m.k, m.r_dom, m.margin, d['msl'], d['frozen'], d['window'],
                      wp.zeros(d['e'], dtype=wp.float32, device=d['center'].device),
                      d['blocked'], d['slope'], d['numerical'], d['failed'], d['accepted']],
              device=d['center'].device)


def _new_shape(data, workspace):
    d = data
    _parallel().accept_shape_inplace(LaunchContext.current(d['center'].device),
        d['center'], d['proposal'], d['n'], d['model'], d['energy'], d['msl'],
        d['frozen'], d['window'], d['blocked'], d['slope'], d['numerical'],
        d['failed'], d['accepted'], workspace)


def _shape_bytes(data):
    return tuple(data[name].numpy().tobytes() for name in ['center', 'window', 'blocked', 'accepted'])


@pytest.mark.parametrize('dev', DEVS)
def test_shape_acceptance_matches_serial_special_paths_and_poisoned_reuse(dev):
    mod = _parallel()
    ref, actual = _shape_inputs(dev), _shape_inputs(dev)
    _serial_shape(ref)
    workspace = mod.alloc_proposal_workspace(actual['e'], 16, dev)
    _poison(workspace)
    _new_shape(actual, workspace)
    assert _shape_bytes(actual) == _shape_bytes(ref)
    assert actual['accepted'].numpy().tolist() == [1, 7, 0, 0, 0, 0]
    assert actual['blocked'].numpy().tolist() == [0, 9, 1, 0, 1, 1]
    _poison(workspace)
    again = _shape_inputs(dev)
    _new_shape(again, workspace)
    assert _shape_bytes(again) == _shape_bytes(ref)


@pytest.mark.parametrize('dev', DEVS)
def test_numerical_stationary_path_still_checks_precise_wall(dev):
    mod = _parallel()
    d = _shape_inputs(dev)
    n, e = d['n'], d['e']
    p = np.zeros((e*n, 2), np.float32)
    p[::n] = [471861.03125, -881672.9375]
    d['center'] = _array(p, wp.vec2f, dev)
    d['proposal'] = _array(p, wp.vec2f, dev)
    d['model'] = _model(e, dev, domain=1e6, margin=0.)
    d['frozen'].fill_(0)
    d['failed'].fill_(0)
    d['numerical'].fill_(1)
    _new_shape(d, mod.alloc_proposal_workspace(e, n, dev))
    assert d['accepted'].numpy().tolist() == [0]*e
    assert d['blocked'].numpy().tolist() == [1]*e


@pytest.mark.parametrize('dev', DEVS)
def test_growth_backtracking_and_frozen_rows_match_serial(dev):
    mod = _parallel()
    n, e = 12, 3
    old = np.tile(_curve(n), (e, 1))
    model = _model(e, dev, domain=.505, margin=0.)
    target = wp.full(e, 4., dtype=wp.float64, device=dev)
    frozen = _array([0, 1, 0], wp.int32, dev)
    ref = _array(old, wp.vec2f, dev)
    actual = _array(old, wp.vec2f, dev)
    ref_proposal = wp.full(e*n, wp.vec2f(7., 9.), dtype=wp.vec2f, device=dev)
    new_proposal = wp.clone(ref_proposal)
    wp.launch(serial._grow_feasible_k, e,
              inputs=[ref, ref_proposal, n, target, .012, model.disc_centers,
                      model.disc_radii, model.k, model.r_dom, model.margin, frozen], device=dev)
    workspace = mod.alloc_proposal_workspace(e, 16, dev)
    _poison(workspace)
    mod.grow_feasible_inplace(LaunchContext.current(dev), actual, new_proposal, n,
                             target, .012, model, frozen, workspace)
    assert actual.numpy().tobytes() == ref.numpy().tobytes()
    assert new_proposal.numpy().tobytes() == ref_proposal.numpy().tobytes()
    assert not np.array_equal(actual.numpy()[:n], old[:n])
    np.testing.assert_array_equal(actual.numpy()[n:2*n], old[n:2*n])
    assert np.linalg.norm(actual.numpy()[:n], axis=1).max() < .505


@pytest.mark.parametrize('dev', DEVS)
@pytest.mark.parametrize('skip_frozen', [0, 1])
def test_guarded_resampling_matches_serial_acceptance_and_frozen_policy(dev, skip_frozen):
    mod = _parallel()
    n, e = 12, 3
    old = np.tile(_curve(n, rough=True), (e, 1))
    candidate = np.tile(_curve(n), (e, 1))
    model = _model(e, dev)
    frozen = _array([0, 1, 0], wp.int32, dev)
    ref, actual = _array(old, wp.vec2f, dev), _array(old, wp.vec2f, dev)
    rp, ap = _array(candidate, wp.vec2f, dev), _array(candidate, wp.vec2f, dev)
    rw, aw = wp.zeros(e, dtype=wp.float32, device=dev), wp.zeros(e, dtype=wp.float32, device=dev)
    wp.launch(serial._accept_resample_k, e,
              inputs=[ref, rp, n, model.disc_centers, model.disc_radii, model.k,
                      model.r_dom, model.margin, frozen, rw, skip_frozen,
                      model.alpha, model.beta, model.eps, model.obs_pts, model.obs_mw,
                      model.m_obs, model.p_exp], device=dev)
    ws = mod.alloc_proposal_workspace(e, 16, dev)
    _poison(ws)
    mod.accept_resample_inplace(LaunchContext.current(dev), actual, ap, n, model,
                               frozen, aw, skip_frozen, ws)
    assert actual.numpy().tobytes() == ref.numpy().tobytes()
    assert ap.numpy().tobytes() == rp.numpy().tobytes()
    assert aw.numpy().tobytes() == rw.numpy().tobytes()
    assert not np.array_equal(actual.numpy()[:n], old[:n])


@pytest.mark.parametrize('dev', DEVS)
def test_all_frozen_rows_leave_external_state_unchanged(dev):
    mod = _parallel()
    data = _shape_inputs(dev)
    data['frozen'].fill_(1)
    before = _shape_bytes(data)
    ws = mod.alloc_proposal_workspace(data['e'], 16, dev)
    _poison(ws)
    _new_shape(data, ws)
    assert _shape_bytes(data) == before


@pytest.mark.parametrize('dev', DEVS)
def test_capture_replay_matches_eager_with_poisoned_workspace(dev):
    mod = _parallel()
    ref, captured = _shape_inputs(dev), _shape_inputs(dev)
    ws = mod.alloc_proposal_workspace(ref['e'], 16, dev)
    _new_shape(ref, ws)
    expected = _shape_bytes(ref)
    _poison(ws)
    with wp.ScopedCapture(device=dev) as cap:
        _new_shape(captured, ws)
    fresh = _shape_inputs(dev)
    for name in ['center', 'window', 'blocked', 'accepted']:
        wp.copy(captured[name], fresh[name])
    _poison(ws)
    wp.capture_launch(cap.graph)
    assert _shape_bytes(captured) == expected


def _disc_model(e, dev, *, growth=False):
    centers = np.full((e, 2, 2), np.nan, np.float32)
    radii = np.zeros((e, 2), np.float32)
    for row in range(e):
        centers[row, 0] = ([.55 if growth else .66, 0.] if row % 2 == 0 else [-.57 if growth else -.535, 0.])
        radii[row, 0] = .03
    return dataclasses.replace(_model(e, dev, domain=.8 if growth else 1., margin=0.),
                               disc_centers=_array(centers.reshape(-1, 2), wp.vec2f, dev),
                               disc_radii=_array(radii.reshape(-1), wp.float32, dev), k=2)


def _replace_shape_model(data, model):
    data['model'] = model
    data['frozen'].fill_(0)
    data['failed'].fill_(0)
    data['numerical'].fill_(0)
    data['slope'].fill_(-1e-5)
    m = model
    wp.launch(serial._shape_energy_k, data['e'],
              inputs=[data['center'], data['n'], m.alpha, m.beta, m.eps, m.obs_pts,
                      m.obs_mw, m.m_obs, m.p_exp, m.disc_centers, m.disc_radii,
                      m.k, m.r_dom, m.margin, data['frozen'], data['energy'], data['scale']],
              device=data['center'].device)


@pytest.mark.parametrize('dev', DEVS)
def test_shape_real_disc_offsets_match_serial_with_nan_unused_slots(dev):
    mod = _parallel()
    ref, actual = _shape_inputs(dev), _shape_inputs(dev)
    model = _disc_model(actual['e'], dev)
    _replace_shape_model(ref, model)
    _replace_shape_model(actual, model)
    _serial_shape(ref)
    ws = mod.alloc_proposal_workspace(actual['e'], 16, dev)
    _poison(ws)
    _new_shape(actual, ws)
    assert _shape_bytes(actual) == _shape_bytes(ref)
    assert actual['accepted'].numpy()[:2].tolist() == [1, 1]
    # The negative-side disc blocks the full candidate only in odd rows.
    points = actual['center'].numpy().reshape(actual['e'], actual['n'], 2)
    assert not np.array_equal(points[0], points[1])


@pytest.mark.parametrize('dev', DEVS)
def test_growth_real_disc_contacts_use_each_environment_offset(dev):
    mod = _parallel()
    n, e = 12, 2
    old = np.tile(_curve(n), (e, 1))
    model = _disc_model(e, dev, growth=True)
    frozen = wp.zeros(e, dtype=wp.int32, device=dev)
    target = wp.full(e, 4., dtype=wp.float64, device=dev)
    ref, actual = _array(old, wp.vec2f, dev), _array(old, wp.vec2f, dev)
    rp, ap = wp.empty(e*n, dtype=wp.vec2f, device=dev), wp.empty(e*n, dtype=wp.vec2f, device=dev)
    wp.launch(serial._grow_feasible_k, e,
              inputs=[ref, rp, n, target, .08, model.disc_centers, model.disc_radii,
                      model.k, model.r_dom, model.margin, frozen], device=dev)
    ws = mod.alloc_proposal_workspace(e, 16, dev)
    _poison(ws)
    mod.grow_feasible_inplace(LaunchContext.current(dev), actual, ap, n, target,
                             .08, model, frozen, ws)
    assert actual.numpy().tobytes() == ref.numpy().tobytes()
    assert ap.numpy().tobytes() == rp.numpy().tobytes()
    final = actual.numpy().reshape(e, n, 2)
    assert np.linalg.norm(final[0], axis=1).max() < np.linalg.norm(final[1], axis=1).max()
    assert ws.accepted_fraction.numpy().tolist() == [.25, .5]


@pytest.mark.parametrize('dev', DEVS)
def test_resampling_disc_contact_rejection_uses_each_environment_offset(dev):
    mod = _parallel()
    n, e = 12, 2
    old = np.tile(_curve(n, rough=True), (e, 1))
    candidate = np.tile(_curve(n), (e, 1))
    model = _disc_model(e, dev)
    frozen = wp.zeros(e, dtype=wp.int32, device=dev)
    ref, actual = _array(old, wp.vec2f, dev), _array(old, wp.vec2f, dev)
    rp, ap = _array(candidate, wp.vec2f, dev), _array(candidate, wp.vec2f, dev)
    rw, aw = wp.zeros(e, dtype=wp.float32, device=dev), wp.zeros(e, dtype=wp.float32, device=dev)
    wp.launch(serial._accept_resample_k, e,
              inputs=[ref, rp, n, model.disc_centers, model.disc_radii, model.k,
                      model.r_dom, model.margin, frozen, rw, 0, model.alpha, model.beta,
                      model.eps, model.obs_pts, model.obs_mw, model.m_obs, model.p_exp], device=dev)
    ws = mod.alloc_proposal_workspace(e, 16, dev)
    _poison(ws)
    mod.accept_resample_inplace(LaunchContext.current(dev), actual, ap, n, model,
                               frozen, aw, 0, ws)
    assert actual.numpy().tobytes() == ref.numpy().tobytes()
    assert ap.numpy().tobytes() == rp.numpy().tobytes()
    assert aw.numpy().tobytes() == rw.numpy().tobytes()
    assert not np.array_equal(actual.numpy()[:n], old[:n])
    np.testing.assert_array_equal(actual.numpy()[n:], old[n:])


@pytest.mark.parametrize('dev', DEVS)
def test_parallel_edge_certificate_rejects_sweep_with_clear_endpoints(dev):
    mod = _parallel()
    n, e = 3, 2
    old = np.tile(np.array([[-1., -.1], [-1., .1], [-1.2, 0.]], np.float32), (e, 1))
    candidate = old + np.array([2.2, 0.], np.float32)
    centers = _array([[0., 0.], [np.nan, np.nan], [0., 2.], [np.nan, np.nan]], wp.vec2f, dev)
    radii = _array([.1, 0., .1, 0.], wp.float32, dev)
    model = dataclasses.replace(_model(e, dev, margin=0.), disc_centers=centers,
                               disc_radii=radii, k=2)
    ws = mod.alloc_proposal_workspace(e, n, dev)
    ws.stationary.fill_(0)
    ws.fraction.fill_(1.)
    ws.pending.fill_(1)
    ws.edge_clear.fill_(57)
    mod._certificate(LaunchContext.current(dev), _array(old, wp.vec2f, dev),
                     _array(candidate, wp.vec2f, dev), n, model, ws, active=ws.pending)
    result = ws.edge_clear.numpy().reshape(e, n)
    assert (result[0] == 0).any()
    assert (result[1] == 1).all()
    assert (np.linalg.norm(old, axis=1) > .1).all()
    assert (np.linalg.norm(candidate, axis=1) > .1).all()


@pytest.mark.parametrize('dev', DEVS)
def test_proposal_wrappers_have_no_hot_allocation_or_host_readback(dev):
    from track_gen.testing.contract import no_allocation, no_host_sync

    mod = _parallel()
    data = _shape_inputs(dev)
    ws = mod.alloc_proposal_workspace(data['e'], 16, dev)
    target = wp.full(data['e'], 4., dtype=wp.float64, device=dev)
    ctx = LaunchContext.current(dev)
    _new_shape(data, ws)  # warm modules before the mechanical hot-path guard
    with no_allocation(), no_host_sync():
        _new_shape(data, ws)
        mod.grow_feasible_inplace(ctx, data['center'], data['proposal'], data['n'],
                                 target, .012, data['model'], data['frozen'], ws)
        mod.accept_resample_inplace(ctx, data['center'], data['proposal'], data['n'],
                                   data['model'], data['frozen'], data['window'], 1, ws)
