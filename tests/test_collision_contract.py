"""CollisionChecker construction/query contracts: validation, reuse, clone."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._collision_fixtures import annulus_polylines, make_annulus_track, make_boxes
from track_gen.collision import BoxContact, CollisionChecker


def test_constructor_validation():
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="max_boxes"):
        CollisionChecker(track, max_boxes=0)
    with pytest.raises(ValueError, match="method"):
        CollisionChecker(track, max_boxes=1, method="bvh")
    with pytest.raises(ValueError, match="sdf_resolution"):
        CollisionChecker(track, max_boxes=1, sdf_resolution=4)
    with pytest.raises(ValueError, match="sdf_padding"):
        CollisionChecker(track, max_boxes=1, sdf_padding=-0.5)


def test_query_validation():
    track = make_annulus_track(E=2, n=64)
    checker = CollisionChecker(track, max_boxes=4)
    pos, yaw, he = make_boxes(2, 4, {})
    bad_pos, bad_yaw, _ = make_boxes(2, 3, {})
    with pytest.raises(ValueError, match="position"):
        checker.query(bad_pos, yaw, he)
    with pytest.raises(ValueError, match="yaw"):
        checker.query(pos, bad_yaw, he)
    with pytest.raises(ValueError, match="yaw"):
        checker.query(pos, pos, he)  # wrong dtype (vec2f where float32 expected)
    with pytest.raises(ValueError, match="position"):
        checker.query(np.zeros((8, 2), np.float32), yaw, he)  # not a wp.array


def test_query_returns_same_instance_and_clone_detaches():
    track = make_annulus_track(E=1, n=64)
    checker = CollisionChecker(track, max_boxes=1)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (1.0, 0.0, 0.0, 0.05, 0.05)})
    c1 = checker.query(pos, yaw, he)
    snap = c1.clone()
    assert isinstance(snap, BoxContact)
    d_before = float(c1.distance.numpy()[0])
    # Move the box out of bounds and re-query: c1 mutates, snap must not.
    pos2, yaw2, he2 = make_boxes(1, 1, {(0, 0): (3.0, 0.0, 0.0, 0.05, 0.05)})
    c2 = checker.query(pos2, yaw2, he2)
    assert c2 is c1
    assert float(c1.distance.numpy()[0]) < 0.0
    np.testing.assert_allclose(float(snap.distance.numpy()[0]), d_before)


def test_segments_sees_track_buffer_updates_without_rebind():
    # The checker aliases the Track buffers; writing new geometry into the SAME
    # buffers (as TrackGenerator.generate() does) must be reflected in queries.
    track = make_annulus_track(E=1, n=64)
    checker = CollisionChecker(track, max_boxes=1)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (1.0, 0.0, 0.0, 0.02, 0.02)})
    assert int(checker.query(pos, yaw, he).oob.numpy()[0]) == 0
    bigger = make_annulus_track(E=1, n=64, r_center=3.0)  # same shapes
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    wp.copy(track.center, bigger.center)
    # Box at r=1 is now inside the hole of the r in [2.7, 3.3] annulus.
    assert int(checker.query(pos, yaw, he).oob.numpy()[0]) == 1


def test_generated_tracks_property_oracle():
    """Random boxes vs the numpy oracle on REAL generated tracks (cpu)."""
    from tests import _collision_oracle as oracle
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.outer.shape[0] // E
    checker = CollisionChecker(track, max_boxes=4)
    rng = np.random.default_rng(0)
    center = track.center.numpy().reshape(E, n_max, 2)
    slots = {}
    for e in range(E):
        if not valid[e]:
            continue
        for b in range(4):
            i = int(rng.integers(0, counts[e]))
            jitter = rng.normal(0.0, 0.15, 2)
            px, py = center[e, i] + jitter
            slots[(e, b)] = (float(px), float(py), float(rng.uniform(0, 6.28)),
                             float(rng.uniform(0.005, 0.08)),
                             float(rng.uniform(0.005, 0.08)))
    pos, yaw, he = make_boxes(E, 4, slots)
    contact = checker.query(pos, yaw, he)
    oob = contact.oob.numpy()
    dist = contact.distance.numpy()
    inner_np = track.inner.numpy().reshape(E, n_max, 2)
    outer_np = track.outer.numpy().reshape(E, n_max, 2)
    checked = 0
    for (e, b), (px, py, yw, hx, hy) in slots.items():
        m = int(counts[e])
        ref = oracle.box_contact(inner_np[e, :m].astype(np.float64),
                                 outer_np[e, :m].astype(np.float64),
                                 (px, py), yw, (hx, hy))
        i = e * 4 + b
        assert oob[i] == ref["oob"], f"env {e} box {b}: oob mismatch"
        np.testing.assert_allclose(dist[i], ref["distance"], atol=1e-4,
                                   err_msg=f"env {e} box {b}")
        checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"
