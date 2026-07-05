"""curvature() and speed_profile(): analytics, numpy oracle, feasibility."""
from __future__ import annotations

import numpy as np
import pytest

from tests import _localize_oracle as oracle
from tests._collision_fixtures import make_annulus_track
from track_gen.localize import curvature, speed_profile


def _seg_lengths(track, e, n_max):
    m = int(track.count.numpy()[e])
    al = track.arclen.numpy()[e * n_max:e * n_max + m].astype(np.float64)
    L = float(track.length.numpy()[e])
    return np.diff(np.append(al, L))


def test_curvature_of_circle_is_one_over_radius():
    R = 2.0
    track = make_annulus_track(E=2, n=256, r_center=R, counts=[256, 200])
    n_max = track.center.shape[0] // 2
    for window in (0, 2):
        kap = curvature(track, window=window).numpy().reshape(2, n_max)
        for e, m in enumerate((256, 200)):
            # CCW loop: positive sign; chordal discretization error is
            # O((pi/m)^2), far below the tolerance.
            np.testing.assert_allclose(kap[e, :m], 1.0 / R, rtol=1e-3,
                                       err_msg=f"window={window} env={e}")
            assert np.isnan(kap[e, m:]).all(), "NaN padding lost"


def test_curvature_validation():
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="window"):
        curvature(track, window=-1)


def test_speed_profile_on_circle_is_uniform_lateral_limit():
    R = 2.0
    E, m = 2, 256
    track = make_annulus_track(E=E, n=m, r_center=R)
    n_max = track.center.shape[0] // E
    v = speed_profile(track, a_lat_max=1.0, a_accel=2.0, a_brake=4.0,
                      v_cap=5.0).numpy().reshape(E, n_max)
    # Uniform curvature: the accel/brake passes never bind and the profile
    # is min(sqrt(a_lat_max * R), v_cap) everywhere.
    np.testing.assert_allclose(v[:, :m], np.sqrt(1.0 * R), rtol=1e-3)
    assert np.isnan(v[:, m:]).all()
    # A tiny cap binds instead.
    v_capped = speed_profile(track, a_lat_max=1.0, a_accel=2.0, a_brake=4.0,
                             v_cap=0.5).numpy().reshape(E, n_max)
    np.testing.assert_allclose(v_capped[:, :m], 0.5, rtol=1e-6)


def test_speed_profile_validation():
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="a_lat_max"):
        speed_profile(track, a_lat_max=0.0, a_accel=1.0, a_brake=1.0, v_cap=1.0)
    with pytest.raises(ValueError, match="a_accel"):
        speed_profile(track, a_lat_max=1.0, a_accel=-1.0, a_brake=1.0, v_cap=1.0)
    with pytest.raises(ValueError, match="a_brake"):
        speed_profile(track, a_lat_max=1.0, a_accel=1.0, a_brake=-1.0, v_cap=1.0)
    with pytest.raises(ValueError, match="v_cap"):
        speed_profile(track, a_lat_max=1.0, a_accel=1.0, a_brake=1.0, v_cap=0.0)
    with pytest.raises(ValueError, match="kappa"):
        speed_profile(track, a_lat_max=1.0, a_accel=1.0, a_brake=1.0,
                      v_cap=1.0, kappa=track.arclen[:8])  # wrong shape


def test_speed_profile_oracle_and_feasibility_on_generated_tracks():
    """Warp profile vs numpy oracle + accel/brake feasibility (cpu)."""
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    a_lat, a_acc, a_brk, cap = 3.0, 1.5, 2.5, 1.2
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.center.shape[0] // E
    kap = curvature(track)
    v = speed_profile(track, a_lat_max=a_lat, a_accel=a_acc, a_brake=a_brk,
                      v_cap=cap, kappa=kap).numpy().reshape(E, n_max)
    kap_np = kap.numpy().reshape(E, n_max)
    checked = 0
    for e in range(E):
        if not valid[e]:
            continue
        m = int(counts[e])
        seg = _seg_lengths(track, e, n_max)
        ref = oracle.speed_profile(kap_np[e, :m], seg, a_lat, a_acc, a_brk, cap)
        np.testing.assert_allclose(v[e, :m], ref, rtol=1e-4,
                                   err_msg=f"env {e}")
        # Steady-state + pass-limit properties (including the wrap):
        ve = v[e, :m].astype(np.float64)
        vn = np.roll(ve, -1)
        assert np.all(ve <= cap + 1e-5)
        assert np.all(ve >= 0.0)
        assert np.all(vn ** 2 <= ve ** 2 + 2.0 * a_acc * seg + 1e-4), \
            f"env {e}: acceleration limit violated"
        assert np.all(ve ** 2 <= vn ** 2 + 2.0 * a_brk * seg + 1e-4), \
            f"env {e}: braking distance violated"
        assert np.all(ve <= np.minimum(
            np.sqrt(a_lat / np.maximum(np.abs(kap_np[e, :m]), 1e-9)), cap) + 1e-4)
        assert np.isnan(v[e, m:]).all()
        checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"


def test_speed_profile_internal_kappa_matches_precomputed():
    track = make_annulus_track(E=1, n=128)
    kap = curvature(track, window=2)
    v_int = speed_profile(track, a_lat_max=1.0, a_accel=1.0, a_brake=1.0,
                          v_cap=2.0, window=2)
    v_pre = speed_profile(track, a_lat_max=1.0, a_accel=1.0, a_brake=1.0,
                          v_cap=2.0, kappa=kap)
    np.testing.assert_array_equal(v_int.numpy(), v_pre.numpy())


def test_degenerate_envs_yield_nan_rows():
    # count < 3: no turn angles, no profile — the whole row is NaN while the
    # healthy sibling env stays finite.
    track = make_annulus_track(E=2, n=128, counts=[2, 128])
    n_max = track.center.shape[0] // 2
    kap = curvature(track).numpy().reshape(2, n_max)
    assert np.isnan(kap[0]).all()
    assert np.isfinite(kap[1, :128]).all()
    v = speed_profile(track, a_lat_max=1.0, a_accel=1.0, a_brake=1.0,
                      v_cap=2.0).numpy().reshape(2, n_max)
    assert np.isnan(v[0]).all()
    assert np.isfinite(v[1, :128]).all()


def test_speed_profile_zero_accel_and_zero_brake_edges():
    """a_accel=0 / a_brake=0 vs the oracle, plus the collapse-to-min law."""
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    a_lat, cap = 3.0, 1.2
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.center.shape[0] // E
    kap = curvature(track)
    kap_np = kap.numpy().reshape(E, n_max)
    cases = ((0.0, 2.5), (1.5, 0.0), (0.0, 0.0))
    profiles = [speed_profile(track, a_lat_max=a_lat, a_accel=a, a_brake=b,
                              v_cap=cap, kappa=kap).numpy().reshape(E, n_max)
                for a, b in cases]
    checked = 0
    for e in range(E):
        if not valid[e]:
            continue
        m = int(counts[e])
        seg = _seg_lengths(track, e, n_max)
        for (a, b), v in zip(cases, profiles):
            ref = oracle.speed_profile(kap_np[e, :m], seg, a_lat, a, b, cap)
            np.testing.assert_allclose(v[e, :m], ref, rtol=1e-4,
                                       err_msg=f"env {e} a={a} b={b}")
        # With a_accel=0 nothing may speed up around the closed loop, so
        # after the wrap laps the profile collapses to the global corner
        # minimum everywhere (a_brake=0 then holds it there too).
        vs_min = np.min(np.minimum(
            np.sqrt(a_lat / np.maximum(np.abs(kap_np[e, :m]), 1e-9)), cap))
        np.testing.assert_allclose(profiles[0][e, :m], vs_min, rtol=1e-4)
        np.testing.assert_allclose(profiles[2][e, :m], vs_min, rtol=1e-4)
        checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"
