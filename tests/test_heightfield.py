import numpy as np

from track_gen._src.heightfield import HeightFieldBaker
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

E, RES = 4, 64


def _track(**kw):
    cfg = TrackGenConfig(device="cpu", num_envs=E, **kw)
    rng = PerEnvSeededRNG(seeds=5, num_envs=E, device="cpu")
    return TrackGenerator(cfg, rng).generate()


def _sample(hf, e, xy):
    lo = hf.lo.numpy()[e]
    hi = hf.hi.numpy()[e]
    g = hf.height.numpy().reshape(E, RES, RES)
    fx = (xy[0] - lo[0]) / (hi[0] - lo[0]) * RES - 0.5
    fy = (xy[1] - lo[1]) / (hi[1] - lo[1]) * RES - 0.5
    xi = int(np.clip(round(fx), 0, RES - 1))
    yi = int(np.clip(round(fy), 0, RES - 1))
    return g[e, yi, xi]


def test_flat_track_bakes_constant_sheet():
    track = _track(z_base=0.5)
    hf = HeightFieldBaker(track, RES).bake()
    g = hf.height.numpy().reshape(E, RES, RES)
    for e in np.flatnonzero(track.valid.numpy()):
        np.testing.assert_allclose(g[e], 0.5, atol=1e-5)


def test_pixel_under_centerline_matches_road_z():
    track = _track(z_profile="noise", z_base=1.0, z_noise_amplitude=0.4,
                   z_min=0.2, z_max=2.0)
    hf = HeightFieldBaker(track, RES).bake()
    n_max = track.center.shape[0] // E
    ctr = track.center.numpy().reshape(E, n_max, 3)
    for e in np.flatnonzero(track.valid.numpy()):
        m = int(track.count.numpy()[e])
        for i in range(0, m, max(1, m // 8)):
            z = _sample(hf, e, ctr[e, i, :2])
            # one-cell tolerance: nearest cross-section within a cell can
            # differ by up to the local grade * cell diagonal
            assert abs(z - ctr[e, i, 2]) < 0.25


def test_offroad_pixel_continues_nearest_edge():
    track = _track(z_profile="random_walk", z_base=1.0, z_min=0.2,
                   z_max=2.0, z_max_step=0.3)
    hf = HeightFieldBaker(track, RES).bake()
    g = hf.height.numpy().reshape(E, RES, RES)
    for e in np.flatnonzero(track.valid.numpy()):
        z = g[e]
        assert np.isfinite(z).all()          # continuation covers the grid
        lo = track.center.numpy().reshape(E, -1, 3)[e, :, 2]
        lo = lo[np.isfinite(lo)]
        assert z.min() >= lo.min() - 1e-4 and z.max() <= lo.max() + 1e-4


def test_invalid_env_bakes_nan():
    track = _track()
    valid = track.valid.numpy()
    if (valid == 0).any():
        hf = HeightFieldBaker(track, RES).bake()
        g = hf.height.numpy().reshape(E, RES, RES)
        e = int(np.flatnonzero(valid == 0)[0])
        assert np.isnan(g[e]).all()
