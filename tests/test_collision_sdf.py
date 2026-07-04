"""SDF backend: bake correctness on the analytic annulus, then queries."""
from __future__ import annotations

import numpy as np
import pytest

from tests._collision_fixtures import make_annulus_track, make_boxes

N = 512
N_MAX = N + 8
R = 64


def _sdf_checker(track, B, res=R, padding=None):
    from track_gen.collision import CollisionChecker
    return CollisionChecker(track, max_boxes=B, method="sdf",
                            sdf_resolution=res, sdf_padding=padding)


def test_bake_grid_bounds_auto_padding():
    track = make_annulus_track(E=1, n=N)
    checker = _sdf_checker(track, 1)
    lo = checker._sdf_lo.numpy().reshape(-1, 2)[0]
    hi = checker._sdf_hi.numpy().reshape(-1, 2)[0]
    # AABB of the outer 1.3-circle is [-1.3, 1.3]^2; auto pad = 0.1 * 2.6 = 0.26.
    np.testing.assert_allclose(lo, [-1.56, -1.56], atol=2e-2)
    np.testing.assert_allclose(hi, [1.56, 1.56], atol=2e-2)


def test_bake_phi_matches_analytic_annulus():
    track = make_annulus_track(E=1, n=N)
    checker = _sdf_checker(track, 1)
    lo = checker._sdf_lo.numpy().reshape(-1, 2)[0]
    hi = checker._sdf_hi.numpy().reshape(-1, 2)[0]
    phi = checker._sdf_phi.numpy().reshape(R, R)
    bid = checker._sdf_bid.numpy().reshape(R, R)
    xs = lo[0] + (np.arange(R) + 0.5) / R * (hi[0] - lo[0])
    ys = lo[1] + (np.arange(R) + 0.5) / R * (hi[1] - lo[1])
    X, Y = np.meshgrid(xs, ys)          # row gy, col gx — matches bake layout
    r = np.hypot(X, Y)
    phi_true = np.minimum(r - 0.7, 1.3 - r)   # signed: + in band, - outside
    np.testing.assert_allclose(phi, phi_true, atol=5e-3)
    # Boundary id: 0 where inner circle is closer, 1 where outer is (skip ties).
    d_in, d_out = np.abs(r - 0.7), np.abs(r - 1.3)
    clear = np.abs(d_in - d_out) > 0.02
    np.testing.assert_array_equal(bid[clear] == 1, (d_out < d_in)[clear])


def test_bake_refreshes_after_track_buffer_update():
    import warp as wp
    track = make_annulus_track(E=1, n=N)
    checker = _sdf_checker(track, 1)
    phi_before = checker._sdf_phi.numpy().copy()
    bigger = make_annulus_track(E=1, n=N, r_center=2.0)
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    wp.copy(track.center, bigger.center)
    checker.bake()
    assert not np.allclose(checker._sdf_phi.numpy(), phi_before)


def test_bake_rejected_for_segments_method():
    track = make_annulus_track(E=1, n=64)
    from track_gen.collision import CollisionChecker
    checker = CollisionChecker(track, max_boxes=1, method="segments")
    with pytest.raises(ValueError, match="bake"):
        checker.bake()
