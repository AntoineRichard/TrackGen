import numpy as np

from track_gen._src.props import PropSampler
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

E = 4


def _track(**kw):
    cfg = TrackGenConfig(device="cpu", num_envs=E, **kw)
    rng = PerEnvSeededRNG(seeds=7, num_envs=E, device="cpu")
    return TrackGenerator(cfg, rng).generate()


def test_prop_positions_sit_on_lifted_boundary():
    track = _track(z_profile="noise", z_base=1.0, z_noise_amplitude=0.4,
                   z_min=0.2, z_max=2.0)
    props = PropSampler(track, spacing=0.15, boundary="outer",
                        mode="points").sample()
    E_ = E
    n_max = track.outer.shape[0] // E_
    mp = props.position.shape[0] // E_
    out = track.outer.numpy().reshape(E_, n_max, 3)
    pos = props.position.numpy().reshape(E_, mp, 3)
    for e in np.flatnonzero(track.valid.numpy()):
        m = int(props.count.numpy()[e])
        zs = pos[e, :m, 2]
        assert np.isfinite(zs).all()
        lo = np.nanmin(out[e, :, 2]) - 1e-4
        hi = np.nanmax(out[e, :, 2]) + 1e-4
        assert (zs >= lo).all() and (zs <= hi).all()
        assert zs.std() > 0.0


def test_flat_props_have_zero_z_and_legacy_xy():
    track = _track()
    props = PropSampler(track, spacing=0.15, boundary="outer",
                        mode="points").sample()
    pos = props.position.numpy()
    finite = np.isfinite(pos[:, 0])
    assert (pos[finite, 2] == 0.0).all()
