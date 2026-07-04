"""Segments-backend collision tests against the analytic annulus."""
from __future__ import annotations

import numpy as np
import pytest

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
