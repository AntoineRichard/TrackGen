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


def test_sdf_query_annulus_cases():
    track = make_annulus_track(E=1, n=N)
    B = 4
    checker = _sdf_checker(track, B, res=128)
    cell = (2 * 1.56) / 128
    pos, yaw, he = make_boxes(1, B, {
        (0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),   # inside
        (0, 1): (0.0, 0.0, 0.0, 0.10, 0.05),   # in the hole
        (0, 2): (2.0, 0.1, 0.0, 0.05, 0.05),   # far outside
        # slot 3 inactive
    })
    contact = checker.query(pos, yaw, he)
    oob = contact.oob.numpy()
    d = contact.distance.numpy()
    assert list(oob[:3]) == [0, 1, 1]
    assert oob[3] == 0 and np.isnan(d[3])
    np.testing.assert_allclose(d[0], 1.3 - np.hypot(1.15, 0.05), atol=2 * cell)
    # nearest lies near the corresponding circle; normal is unit and inward.
    near = contact.nearest.numpy().reshape(-1, 2)
    nrm = contact.normal.numpy().reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(near[0]), 1.3, atol=2 * cell)
    np.testing.assert_allclose(np.linalg.norm(nrm[0]), 1.0, atol=1e-4)
    assert np.dot(nrm[0], near[0]) < 0  # inward = toward the origin, near outer
    assert contact.boundary.numpy()[0] == 1
    assert contact.boundary.numpy()[1] == 0


def test_sdf_agrees_with_segments_backend():
    from track_gen.collision import CollisionChecker
    rng = np.random.default_rng(3)
    track = make_annulus_track(E=2, n=N, counts=[N, 300])
    B = 16
    res = 128
    cell = (2 * (1.3 + 0.6)) / res
    slots = {}
    for e in range(2):
        for b in range(B):
            r = rng.uniform(0.3, 1.7)
            th = rng.uniform(0.0, 2 * np.pi)
            slots[(e, b)] = (r * np.cos(th), r * np.sin(th),
                             rng.uniform(0, 2 * np.pi),
                             rng.uniform(0.02, 0.12), rng.uniform(0.02, 0.12))
    pos, yaw, he = make_boxes(2, B, slots)
    exact = CollisionChecker(track, max_boxes=B, method="segments").query(
        pos, yaw, he).clone()
    # Explicit padding: random box corners reach radius ~1.87, beyond the 10%
    # auto padding (grid edge 1.56). Per the sdf contract, distance is only
    # grid-accurate within the padded extent — outside it only the OOB flag is
    # guaranteed — so pad to 0.6 (grid edge 1.9) to cover every sampled corner.
    approx = _sdf_checker(track, B, res=res, padding=0.6).query(pos, yaw, he)
    d_ex = exact.distance.numpy()
    d_ap = approx.distance.numpy()
    oob_ex = exact.oob.numpy()
    oob_ap = approx.oob.numpy()
    np.testing.assert_allclose(d_ap, d_ex, atol=2 * cell)
    # OOB flags may only disagree in the +-2 cell band around zero clearance.
    disagree = oob_ex != oob_ap
    assert np.all(np.abs(d_ex[disagree]) < 2 * cell)
