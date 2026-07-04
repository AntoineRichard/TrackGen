"""Segments-backend collision tests against the analytic annulus."""
from __future__ import annotations

import numpy as np

from tests._collision_fixtures import annulus_polylines, make_annulus_track, make_boxes

N = 512
N_MAX = N + 8


def _checker(track, B):
    from track_gen.collision import CollisionChecker
    return CollisionChecker(track, max_boxes=B, method="segments")


def test_oob_flags_annulus():
    track = make_annulus_track(E=1, n=N)
    B = 8
    boxes = {
        (0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),    # fully inside the band
        (0, 1): (0.0, 0.0, 0.3, 0.10, 0.05),    # inside the hole
        (0, 2): (1.3, 0.0, 0.0, 0.10, 0.10),    # straddling the outer boundary
        (0, 3): (2.0, 0.1, 0.0, 0.05, 0.05),    # fully outside the outer loop
        (0, 4): (0.7, 0.0, np.pi / 4, 0.10, 0.02),  # straddling the inner boundary
        (0, 5): (1.0, 0.0, 0.7, 0.04, 0.02),    # inside, rotated
        # slots 6, 7 left inactive (NaN position)
    }
    pos, yaw, he = make_boxes(1, B, boxes)
    contact = _checker(track, B).query(pos, yaw, he)
    oob = contact.oob.numpy()
    assert list(oob[:6]) == [0, 1, 1, 1, 1, 0]
    assert list(oob[6:]) == [0, 0]              # inactive slots
    dist = contact.distance.numpy()
    assert np.all(np.isnan(dist[6:]))           # NaN outputs for inactive slots
    assert list(contact.boundary.numpy()[6:]) == [-1, -1]


def test_import_surface():
    import track_gen
    from track_gen.collision import BoxContact, CollisionChecker  # noqa: F401
    assert "collision" in track_gen.__all__
    assert track_gen.collision.CollisionChecker is CollisionChecker


def test_clearance_inside_matches_analytic():
    track = make_annulus_track(E=1, n=N)
    B = 2
    pos, yaw, he = make_boxes(1, B, {(0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),
                                     (0, 1): (0.75, 0.0, 0.0, 0.02, 0.02)})
    contact = _checker(track, B).query(pos, yaw, he)
    d = contact.distance.numpy()
    bnd = contact.boundary.numpy()
    # Box 0: outer is nearest. Clearance = ro - |farthest corner|.
    np.testing.assert_allclose(d[0], 1.3 - np.hypot(1.15, 0.05), atol=2e-3)
    assert bnd[0] == 1
    # Box 1: inner is nearest. Clearance = |closest box point| - ri = 0.73 - 0.7.
    np.testing.assert_allclose(d[1], 0.03, atol=2e-3)
    assert bnd[1] == 0
    # nearest lies ON the corresponding boundary circle.
    near = contact.nearest.numpy().reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(near[0]), 1.3, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(near[1]), 0.7, atol=2e-3)


def test_penetration_depth_when_oob():
    track = make_annulus_track(E=1, n=N)
    B = 2
    pos, yaw, he = make_boxes(1, B, {(0, 0): (1.3, 0.0, 0.0, 0.1, 0.1),
                                     (0, 1): (0.0, 0.0, 0.0, 0.1, 0.05)})
    contact = _checker(track, B).query(pos, yaw, he)
    d = contact.distance.numpy()
    # Box 0 straddles the outer: deepest corners (1.4, +-0.1).
    np.testing.assert_allclose(d[0], -(np.hypot(1.4, 0.1) - 1.3), atol=2e-3)
    # Box 1 fully in the hole: deepest corner is the one closest to the origin
    # (max penetration = ri - min corner radius).
    np.testing.assert_allclose(d[1], -(0.7 - np.hypot(0.1, 0.05)), atol=2e-3)


def test_normal_points_into_band():
    from tests import _collision_oracle as oracle
    track = make_annulus_track(E=1, n=N)
    inner, outer = annulus_polylines(track, 0, N_MAX)
    B = 4
    pos, yaw, he = make_boxes(1, B, {
        (0, 0): (1.1, 0.2, 0.0, 0.05, 0.05),   # near outer
        (0, 1): (0.8, -0.3, 0.4, 0.03, 0.03),  # near inner
        (0, 2): (1.35, 0.0, 0.0, 0.02, 0.02),  # outside the outer loop
        (0, 3): (0.3, 0.3, 0.0, 0.02, 0.02),   # in the hole
    })
    contact = _checker(track, B).query(pos, yaw, he)
    near = contact.nearest.numpy().reshape(-1, 2)
    nrm = contact.normal.numpy().reshape(-1, 2)
    eps = 1e-3
    for i in range(B):
        np.testing.assert_allclose(np.linalg.norm(nrm[i]), 1.0, atol=1e-5)
        probe_in = near[i] + eps * nrm[i]
        probe_out = near[i] - eps * nrm[i]
        in_band = (oracle.point_in_poly(probe_in, outer)
                   and not oracle.point_in_poly(probe_in, inner))
        out_band = (oracle.point_in_poly(probe_out, outer)
                    and not oracle.point_in_poly(probe_out, inner))
        assert in_band and not out_band, f"box {i}: normal not oriented into band"


def test_matches_oracle_on_random_boxes():
    from tests import _collision_oracle as oracle
    rng = np.random.default_rng(7)
    track = make_annulus_track(E=1, n=256, N_max=N_MAX)
    inner, outer = annulus_polylines(track, 0, N_MAX)
    B = 32
    slots = {}
    for b in range(B):
        r = rng.uniform(0.4, 1.6)
        th = rng.uniform(0.0, 2 * np.pi)
        slots[(0, b)] = (r * np.cos(th), r * np.sin(th),
                         rng.uniform(0, 2 * np.pi),
                         rng.uniform(0.01, 0.15), rng.uniform(0.01, 0.15))
    pos, yaw, he = make_boxes(1, B, slots)
    contact = _checker(track, B).query(pos, yaw, he)
    oob = contact.oob.numpy()
    dist = contact.distance.numpy()
    for b in range(B):
        px, py, yw, hx, hy = slots[(0, b)]
        ref = oracle.box_contact(inner, outer, (px, py), yw, (hx, hy))
        assert oob[b] == ref["oob"], f"box {b} oob mismatch"
        np.testing.assert_allclose(dist[b], ref["distance"], atol=1e-4,
                                   err_msg=f"box {b} distance mismatch")


def test_per_env_count_variation():
    E, B = 3, 1
    track = make_annulus_track(E=E, n=N, counts=[N, 300, 64])
    pos, yaw, he = make_boxes(E, B, {(e, 0): (1.1, 0.0, 0.0, 0.05, 0.05)
                                     for e in range(E)})
    contact = _checker(track, B).query(pos, yaw, he)
    d = contact.distance.numpy()
    expected = 1.3 - np.hypot(1.15, 0.05)
    np.testing.assert_allclose(d[0], expected, atol=2e-3)
    np.testing.assert_allclose(d[1], expected, atol=2e-3)
    np.testing.assert_allclose(d[2], expected, atol=6e-3)  # coarse 64-gon
    assert list(contact.oob.numpy()) == [0, 0, 0]
