"""CUDA-only: checkpoints/progress/discs under wp.ScopedCapture (bound mode).

Poisoned-buffer replay proves each captured graph recomputes results. The
progress test compares a captured-replay trace against an eager twin tracker
stepped over the same positions.
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
from track_gen._src import checkpoints as cps_mod  # noqa: E402
from track_gen._src import collision_discs as discs_mod  # noqa: E402
from track_gen._src import progress as prog_mod  # noqa: E402
from track_gen.checkpoints import CheckpointSampler  # noqa: E402
from track_gen.collision import DiscChecker  # noqa: E402
from track_gen.progress import ProgressTracker  # noqa: E402

DEV = "cuda:0"


def test_checkpoint_sample_graph_replay():
    track = make_annulus_track(E=4, n=256, device=DEV)
    sampler = CheckpointSampler(track, spacing=0.8)
    eager = sampler.sample().clone()
    prev = cps_mod._CAPTURING
    cps_mod._CAPTURING = True
    try:
        sampler.sample()
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            sampler.sample()
    finally:
        cps_mod._CAPTURING = prev
    sampler._set.position.fill_(12345.0)
    sampler._set.count.fill_(-7)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    np.testing.assert_array_equal(sampler._set.count.numpy(), eager.count.numpy())
    np.testing.assert_allclose(sampler._set.position.numpy(),
                               eager.position.numpy(), rtol=1e-5, equal_nan=True)


def test_progress_update_graph_replay_matches_eager_twin():
    E = 4
    track = make_annulus_track(E=E, n=256, device=DEV)
    cps = CheckpointSampler(track, spacing=0.8).sample()
    pos_buf = wp.zeros(E, dtype=wp.vec2f, device=DEV)
    bound = ProgressTracker(cps, position=pos_buf)
    eager = ProgressTracker(cps)

    # Capture one bound update (warmup on a twin state, then reset).
    prev = prog_mod._CAPTURING
    prog_mod._CAPTURING = True
    try:
        bound.update()
        wp.synchronize()
        mask = wp.full(E, 1, dtype=wp.int32, device=DEV)
        bound.reset(mask)
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            bound.update()
    finally:
        prog_mod._CAPTURING = prev

    # Walk the annulus centerline CCW; both trackers see identical positions.
    steps = [np.stack([np.cos(a) * 1.0 * np.ones(E), np.sin(a) * np.ones(E)],
                      axis=1).astype(np.float32)
             for a in np.deg2rad(np.arange(-20.0, 340.0, 40.0))]
    for s in steps:
        arr = wp.array(s, dtype=wp.vec2f, device=DEV)
        wp.copy(pos_buf, arr)
        bound._events.passed.fill_(-7)         # poison: replay must recompute
        bound._events.dist_to_next.fill_(12345.0)
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ev_e = eager.update(arr)
        np.testing.assert_array_equal(bound._events.passed.numpy(),
                                      ev_e.passed.numpy())
        np.testing.assert_array_equal(bound._events.next_checkpoint.numpy(),
                                      ev_e.next_checkpoint.numpy())
        np.testing.assert_allclose(bound._events.dist_to_next.numpy(),
                                   ev_e.dist_to_next.numpy(), rtol=1e-5,
                                   equal_nan=True)


def test_progress_reset_graph_replay():
    E = 4
    track = make_annulus_track(E=E, n=256, device=DEV)
    cps = CheckpointSampler(track, spacing=0.8).sample()
    tracker = ProgressTracker(cps)

    # Advance state eagerly so progress > 0 before capturing the reset.
    steps = [np.stack([np.cos(a) * 1.0 * np.ones(E), np.sin(a) * np.ones(E)],
                      axis=1).astype(np.float32)
             for a in np.deg2rad(np.arange(-20.0, 100.0, 40.0))]
    for s in steps:
        tracker.update(wp.array(s, dtype=wp.vec2f, device=DEV))
    assert int(tracker._progress.numpy().sum()) > 0

    mask = wp.full(E, 1, dtype=wp.int32, device=DEV)
    prev = prog_mod._CAPTURING
    prog_mod._CAPTURING = True
    try:
        tracker.reset(mask)  # warmup
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            tracker.reset(mask)
    finally:
        prog_mod._CAPTURING = prev

    # Re-advance state so the buffers are non-zero/non-NaN before replay.
    for s in steps:
        tracker.update(wp.array(s, dtype=wp.vec2f, device=DEV))
    assert int(tracker._progress.numpy().sum()) > 0

    wp.capture_launch(cap.graph)
    wp.synchronize()
    assert np.all(tracker._progress.numpy() == 0)
    assert np.all(tracker._next.numpy() == 0)
    assert np.isnan(tracker._prev_pos.numpy()).all()


def test_disc_query_graph_replay_bound():
    E, B = 2, 4
    discs = wp.array(np.array([[0.12, 0.0], [0.5, 0.5]] * E, np.float32),
                     dtype=wp.vec2f, device=DEV)
    pos, yaw, he = make_boxes(E, B, {(e, 0): (0.0, 0.0, 0.0, 0.1, 0.05)
                                     for e in range(E)}, device=DEV)
    checker = DiscChecker(discs, radius=0.03, max_boxes=B, num_envs=E,
                          position=pos, yaw=yaw, half_extents=he)
    eager = checker.query().clone()
    prev = discs_mod._CAPTURING
    discs_mod._CAPTURING = True
    try:
        checker.query()
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            checker.query()
    finally:
        discs_mod._CAPTURING = prev
    checker._contact.hit.fill_(-7)
    checker._contact.depth.fill_(12345.0)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    np.testing.assert_array_equal(checker._contact.hit.numpy(), eager.hit.numpy())
    np.testing.assert_allclose(checker._contact.depth.numpy(),
                               eager.depth.numpy(), rtol=1e-6)
