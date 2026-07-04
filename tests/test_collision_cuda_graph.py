"""CUDA-only: CollisionChecker.query()/bake() inside wp.ScopedCapture.

Follows the test_warp_graph.py pattern: whole module skipped without CUDA.
The _CAPTURING flag suppresses the checker's post-launch wp.synchronize so the
capture region stays sync-free.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda"),
]

import warp as wp  # noqa: E402
from tests._collision_fixtures import make_annulus_track, make_boxes  # noqa: E402
from track_gen._src import collision as collision_mod  # noqa: E402
from track_gen.collision import CollisionChecker  # noqa: E402

DEV = "cuda:0"


@pytest.mark.parametrize("method", ["segments", "sdf"])
def test_query_graph_replay_matches_eager(method):
    track = make_annulus_track(E=4, n=256, device=DEV)
    B = 8
    checker = CollisionChecker(track, max_boxes=B, method=method,
                               sdf_resolution=64)
    slots = {(e, b): (1.1, 0.0, 0.3 * b, 0.05, 0.03)
             for e in range(4) for b in range(4)}
    pos, yaw, he = make_boxes(4, B, slots, device=DEV)

    eager = checker.query(pos, yaw, he).clone()

    prev = collision_mod._CAPTURING
    collision_mod._CAPTURING = True
    try:
        checker.query(pos, yaw, he)  # warmup: modules loaded before capture
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            checker.query(pos, yaw, he)
    finally:
        collision_mod._CAPTURING = prev

    wp.capture_launch(cap.graph)
    wp.synchronize()
    replay = checker._contact

    np.testing.assert_array_equal(replay.oob.numpy(), eager.oob.numpy())
    np.testing.assert_allclose(replay.distance.numpy(), eager.distance.numpy(),
                               rtol=1e-5, atol=1e-6, equal_nan=True)


def test_bake_graph_capturable():
    track = make_annulus_track(E=2, n=256, device=DEV)
    checker = CollisionChecker(track, max_boxes=1, method="sdf",
                               sdf_resolution=64)
    phi_eager = checker._sdf_phi.numpy().copy()
    prev = collision_mod._CAPTURING
    collision_mod._CAPTURING = True
    try:
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            checker.bake()
    finally:
        collision_mod._CAPTURING = prev
    wp.capture_launch(cap.graph)
    wp.synchronize()
    np.testing.assert_allclose(checker._sdf_phi.numpy(), phi_eager,
                               rtol=1e-6, atol=1e-7, equal_nan=True)
