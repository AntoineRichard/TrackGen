"""Stable input binding for CollisionChecker (retrofit)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._collision_fixtures import make_annulus_track, make_boxes
from track_gen.collision import CollisionChecker


def test_bound_mode_equivalence_and_live_buffer():
    track = make_annulus_track(E=1, n=256)
    B = 2
    checker_free = CollisionChecker(track, max_boxes=B)
    checker_bound = CollisionChecker(track, max_boxes=B)
    pos, yaw, he = make_boxes(1, B, {(0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),
                                     (0, 1): (0.0, 0.0, 0.0, 0.05, 0.05)})
    checker_bound.bind_inputs(pos, yaw, he)
    r_free = checker_free.query(pos, yaw, he).clone()
    r_bound = checker_bound.query()
    np.testing.assert_array_equal(r_bound.oob.numpy(), r_free.oob.numpy())
    np.testing.assert_allclose(r_bound.distance.numpy(), r_free.distance.numpy(),
                               equal_nan=True)
    # Writing new poses into the bound buffer is seen without re-binding.
    pos2, _, _ = make_boxes(1, B, {(0, 0): (3.0, 0.0, 0.0, 0.05, 0.05),
                                   (0, 1): (0.0, 0.0, 0.0, 0.05, 0.05)})
    wp.copy(pos, pos2)
    r2 = checker_bound.query()
    assert int(r2.oob.numpy()[0]) == 1  # box 0 teleported out of bounds


def test_mode_misuse_errors():
    track = make_annulus_track(E=1, n=256)
    checker = CollisionChecker(track, max_boxes=1)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (1.0, 0.0, 0.0, 0.05, 0.05)})
    with pytest.raises(ValueError, match="not bound"):
        checker.query()
    checker.bind_inputs(pos, yaw, he)
    with pytest.raises(ValueError, match="bound"):
        checker.query(pos, yaw, he)


def test_bind_validation():
    track = make_annulus_track(E=2, n=256)
    checker = CollisionChecker(track, max_boxes=2)
    bad_pos, yaw, he = make_boxes(2, 1, {})  # wrong stride
    _, good_yaw, good_he = make_boxes(2, 2, {})
    with pytest.raises(ValueError, match="position"):
        checker.bind_inputs(bad_pos, good_yaw, good_he)
