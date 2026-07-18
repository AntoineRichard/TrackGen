"""PropSampler contracts: in-place reuse, clone, aliasing, oracle property test."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests._collision_fixtures import make_annulus_track
from tests._props_oracle import sample_boundary
from track_gen.props import PropSampler, PropSet

N = 512


def test_sample_returns_same_instance_and_clone_detaches():
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1)
    p1 = sampler.sample()
    snap = p1.clone()
    assert isinstance(snap, PropSet)
    n_before = int(p1.count.numpy()[0])
    pos_before = snap.position.numpy().copy()
    # Mutate the bound track (same buffers, as generate() would) and resample.
    bigger = make_annulus_track(E=1, n=N, r_center=2.0)
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    p2 = sampler.sample()
    assert p2 is p1
    assert int(p1.count.numpy()[0]) > n_before  # longer boundary -> more props
    np.testing.assert_allclose(snap.position.numpy(), pos_before)  # snapshot intact


def test_matches_oracle_on_generated_tracks():
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.outer.shape[0] // E
    spacing = 0.07
    sampler = PropSampler(track, spacing=spacing, boundary="outer", mode="points",
                          max_props=512)
    props = sampler.sample()
    outer = track.outer.numpy().reshape(E, n_max, 3)[..., :2]
    M = sampler._M
    checked = 0
    for e in range(E):
        if not valid[e]:
            continue
        poly = outer[e, :int(counts[e])]
        ref_pos, ref_tang, ref_n, ref_step, ref_trunc = sample_boundary(
            poly, spacing, 512)
        assert int(props.count.numpy()[e]) == ref_n
        np.testing.assert_allclose(props.step.numpy()[e], ref_step, rtol=1e-4)
        got = props.position.numpy().reshape(-1, 3)[e * M:e * M + ref_n]
        ref_pos3 = np.column_stack([ref_pos, np.zeros(ref_n)])  # flat: z = 0
        np.testing.assert_allclose(got, ref_pos3, atol=1e-4,
                                   err_msg=f"env {e} positions")
        got_t = props.tangent.numpy().reshape(-1, 2)[e * M:e * M + ref_n]
        np.testing.assert_allclose(got_t, ref_tang, atol=1e-3,
                                   err_msg=f"env {e} tangents")
        checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"


def test_segments_are_chords_of_points():
    # segments mode must equal chords between consecutive points-mode samples.
    track = make_annulus_track(E=1, n=N)
    pts = PropSampler(track, spacing=0.15, mode="points").sample().clone()
    segs = PropSampler(track, spacing=0.15, mode="segments").sample()
    n = int(pts.count.numpy()[0])
    assert int(segs.count.numpy()[0]) == n
    p = pts.position.numpy().reshape(-1, 3)[:n]  # flat fixture: z = 0
    p_next = np.roll(p, -1, axis=0)
    np.testing.assert_allclose(segs.position.numpy().reshape(-1, 3)[:n],
                               (p + p_next) / 2, atol=1e-4)
    np.testing.assert_allclose(segs.length.numpy()[:n],
                               np.linalg.norm(p_next - p, axis=1), atol=1e-4)
