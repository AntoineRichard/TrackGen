"""CUDA-only: TrackLocalizer.query/reset under wp.ScopedCapture (bound mode).

Poisoned-buffer replay proves the captured graph recomputes results; the
warm-started replay trace is compared against an eager cold-scan twin.
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
from track_gen._src import runtime  # noqa: E402
from tests._collision_fixtures import make_annulus_track  # noqa: E402
from track_gen.localize import TrackLocalizer  # noqa: E402

DEV = "cuda:0"


def test_localize_query_graph_replay_matches_eager_cold_twin():
    E = 4
    track = make_annulus_track(E=E, n=256, device=DEV)
    pos_buf = wp.zeros(E, dtype=wp.vec2f, device=DEV)
    bound = TrackLocalizer(track, warm_window=16, position=pos_buf)
    eager = TrackLocalizer(track)

    # Capture one bound query (warmup, then reset the warm memory so the
    # first replay starts from the same cold state as construction).
    prev = runtime._CAPTURING
    runtime._CAPTURING = True
    try:
        bound.query()
        wp.synchronize()
        bound.reset(wp.full(E, 1, dtype=wp.int32, device=DEV))
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            bound.query()
    finally:
        runtime._CAPTURING = prev

    # Walk the annulus; steps stay well inside the warm window.
    for a in np.deg2rad(np.arange(0.0, 360.0, 10.0)):
        r = 1.0 + 0.2 * np.sin(3.0 * a)
        s = np.stack([r * np.cos(a) * np.ones(E), r * np.sin(a) * np.ones(E)],
                     axis=1).astype(np.float32)
        arr = wp.array(s, dtype=wp.vec2f, device=DEV)
        wp.copy(pos_buf, arr)
        bound._frame.s.fill_(12345.0)          # poison: replay must recompute
        bound._frame.segment.fill_(-7)
        wp.capture_launch(cap.graph)
        wp.synchronize()
        f_e = eager.query(arr)
        np.testing.assert_array_equal(bound._frame.segment.numpy(),
                                      f_e.segment.numpy())
        np.testing.assert_array_equal(bound._frame.s.numpy(), f_e.s.numpy())
        np.testing.assert_array_equal(bound._frame.n.numpy(), f_e.n.numpy())


def test_localize_reset_graph_replay():
    E = 4
    track = make_annulus_track(E=E, n=256, device=DEV)
    loc = TrackLocalizer(track, warm_window=8)
    mask = wp.full(E, 1, dtype=wp.int32, device=DEV)

    prev = runtime._CAPTURING
    runtime._CAPTURING = True
    try:
        loc.reset(mask)  # warmup
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            loc.reset(mask)
    finally:
        runtime._CAPTURING = prev

    # Seed warm memory, then replay the captured reset: memory must drop.
    pts = np.tile(np.array([[1.1, 0.0]], np.float32), (E, 1))
    loc.query(wp.array(pts, dtype=wp.vec2f, device=DEV))
    assert np.all(loc._last.numpy() >= 0)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    assert np.all(loc._last.numpy() == -1)
