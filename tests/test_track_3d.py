import numpy as np
import pytest
import warp as wp

from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

E = 8


def _gen(**kw):
    cfg = TrackGenConfig(device="cpu", num_envs=E, **kw)
    rng = PerEnvSeededRNG(seeds=1234, num_envs=E, device="cpu")
    return TrackGenerator(cfg, rng).generate(), cfg


def _env(track, cfg, e):
    n_max = track.center.shape[0] // E
    m = int(track.count.numpy()[e])
    sl = slice(e * n_max, e * n_max + m)
    return (track.center.numpy()[sl], track.outer.numpy()[sl],
            track.inner.numpy()[sl], track.arclen.numpy()[sl],
            float(track.length.numpy()[e]))


def test_walk_profile_lifts_level_cross_sections():
    track, cfg = _gen(z_profile="random_walk", z_base=1.0, z_min=0.2,
                      z_max=2.0, z_max_step=0.3)
    valid = track.valid.numpy()
    assert valid.any()
    for e in np.flatnonzero(valid):
        ctr, out, inn, _, _ = _env(track, cfg, e)
        assert (ctr[:, 2] >= 0.2 - 1e-5).all() and (ctr[:, 2] <= 2.0 + 1e-5).all()
        assert ctr[:, 2].std() > 0.0
        np.testing.assert_allclose(out[:, 2], ctr[:, 2], atol=0.0)  # level
        np.testing.assert_allclose(inn[:, 2], ctr[:, 2], atol=0.0)


def test_3d_arclen_consistency():
    track, cfg = _gen(z_profile="noise", z_base=1.0, z_noise_amplitude=0.5,
                      z_min=0.0, z_max=2.0)
    for e in np.flatnonzero(track.valid.numpy()):
        ctr, _, _, arc, length = _env(track, cfg, e)
        chords = np.linalg.norm(np.diff(ctr, axis=0), axis=1)
        np.testing.assert_allclose(np.diff(arc), chords, atol=1e-4)
        closing = np.linalg.norm(ctr[0] - ctr[-1])
        np.testing.assert_allclose(length, arc[-1] + closing, atol=1e-4)
        # a hilly loop is strictly longer in 3D than in plan view
        plan = np.linalg.norm(np.diff(ctr[:, :2], axis=0), axis=1).sum()
        assert arc[-1] > plan


def test_walk_closes_loop():
    track, cfg = _gen(z_profile="random_walk", z_base=1.0, z_min=0.2,
                      z_max=2.0, z_max_step=0.3)
    for e in np.flatnonzero(track.valid.numpy()):
        ctr, _, _, _, _ = _env(track, cfg, e)
        assert abs(ctr[0, 2] - ctr[-1, 2]) < 0.5


def test_grade_validity_and_disable():
    _, _ = _gen()  # warm
    t_off, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0)
    t_on, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0,
                   z_valid_grade=0.5)
    assert t_on.valid.numpy().sum() < max(1, t_off.valid.numpy().sum())


def test_flat_matches_goldens_locally():
    track, cfg = _gen()
    for e in np.flatnonzero(track.valid.numpy()):
        ctr, _, _, _, _ = _env(track, cfg, e)
        assert (ctr[:, 2] == 0.0).all()


def test_standalone_inflate_rejects_nonflat_without_seeds():
    from track_gen._src import warp_pipeline
    cfg = TrackGenConfig(device="cpu", num_envs=1, z_profile="uniform",
                         z_min=0.5, z_max=1.0)
    center = wp.zeros(cfg.num_points, dtype=wp.vec2f, device="cpu")
    with pytest.raises(ValueError, match="seeds"):
        warp_pipeline.inflate_warp(center, cfg)
