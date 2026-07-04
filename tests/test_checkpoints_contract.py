"""CheckpointSampler contracts: reuse, aliasing, truncation, oracle."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests._checkpoints_oracle import sample_checkpoints
from tests._collision_fixtures import make_annulus_track
from track_gen.checkpoints import CheckpointSampler

N = 512


def test_sample_returns_same_set_and_clone_detaches():
    track = make_annulus_track(E=1, n=N)
    sampler = CheckpointSampler(track, spacing=0.8)
    s1 = sampler.sample()
    snap = s1.clone()
    pos_before = snap.position.numpy().copy()
    bigger = make_annulus_track(E=1, n=N, r_center=2.0)
    wp.copy(track.center, bigger.center)
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    s2 = sampler.sample()
    assert s2 is s1
    assert int(s1.count.numpy()[0]) > int(snap.count.numpy()[0])
    np.testing.assert_allclose(snap.position.numpy(), pos_before)


def test_truncation_flag():
    track = make_annulus_track(E=1, n=N)
    sampler = CheckpointSampler(track, spacing=0.2, max_checkpoints=8)
    cps = sampler.sample()
    assert int(cps.count.numpy()[0]) == 8
    assert int(sampler.truncated.numpy()[0]) == 1


def test_matches_oracle_on_generated_tracks():
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.outer.shape[0] // E
    spacing = 0.5
    sampler = CheckpointSampler(track, spacing=spacing, max_checkpoints=64)
    cps = sampler.sample()
    M = sampler._M
    center = track.center.numpy().reshape(E, n_max, 2)
    inner = track.inner.numpy().reshape(E, n_max, 2)
    outer = track.outer.numpy().reshape(E, n_max, 2)
    checked = 0
    for e in range(E):
        if not valid[e]:
            continue
        m = int(counts[e])
        ref = sample_checkpoints(center[e, :m], inner[e, :m], outer[e, :m],
                                 spacing, 64)
        assert int(cps.count.numpy()[e]) == ref["n"]
        sl = slice(e * M, e * M + ref["n"])
        np.testing.assert_allclose(cps.position.numpy().reshape(-1, 2)[sl],
                                   ref["position"], atol=1e-4)
        np.testing.assert_allclose(cps.left.numpy().reshape(-1, 2)[sl],
                                   ref["left"], atol=1e-4)
        np.testing.assert_allclose(cps.right.numpy().reshape(-1, 2)[sl],
                                   ref["right"], atol=1e-4)
        np.testing.assert_allclose(cps.tangent.numpy().reshape(-1, 2)[sl],
                                   ref["tangent"], atol=1e-3)
        checked += 1
    assert checked > 0
