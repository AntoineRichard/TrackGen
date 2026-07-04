"""CUDA-only: PropSampler.sample() inside wp.ScopedCapture.

Buffers are poisoned between capture and replay so the comparison proves the
captured graph recomputes results (not stale pre-capture buffer contents).
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
from tests._collision_fixtures import make_annulus_track  # noqa: E402
from track_gen._src import props as props_mod  # noqa: E402
from track_gen.props import PropSampler  # noqa: E402

DEV = "cuda:0"


@pytest.mark.parametrize("mode", ["points", "segments"])
def test_sample_graph_replay_matches_eager(mode):
    track = make_annulus_track(E=4, n=256, device=DEV)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode=mode)
    eager = sampler.sample().clone()

    prev = props_mod._CAPTURING
    props_mod._CAPTURING = True
    try:
        sampler.sample()  # warmup: modules loaded before capture
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            sampler.sample()
    finally:
        props_mod._CAPTURING = prev

    # Poison outputs so the comparison proves the REPLAY recomputed them.
    replay = sampler._props
    replay.position.fill_(12345.0)
    replay.count.fill_(-7)
    replay.step.fill_(12345.0)

    wp.capture_launch(cap.graph)
    wp.synchronize()

    np.testing.assert_array_equal(replay.count.numpy(), eager.count.numpy())
    np.testing.assert_allclose(replay.step.numpy(), eager.step.numpy(),
                               rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(replay.position.numpy(), eager.position.numpy(),
                               rtol=1e-5, atol=1e-6, equal_nan=True)
