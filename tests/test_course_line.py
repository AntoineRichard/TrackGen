import numpy as np
import warp as wp

from track_gen._src.course_line import CourseLine
from track_gen._src.gate_generator import GateGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.types import GateGenConfig

E = 4


def _seq(**kw):
    cfg = GateGenConfig(device="cpu", num_envs=E, gate_width=0.05, **kw)
    rng = PerEnvSeededRNG(seeds=3, num_envs=E, device="cpu")
    return GateGenerator(cfg, rng).generate(), cfg


def test_interpolates_gate_anchors():
    seq, cfg = _seq(z_profile="random_walk", z_base=1.0, z_min=0.2,
                    z_max=2.0, z_max_step=0.4)
    line = CourseLine(seq, samples_per_gate=8)
    line.refresh()
    G, spg = int(cfg.max_gates), 8
    n_max = G * spg
    ctr = line.track.center.numpy().reshape(E, n_max, 3)
    gp = seq.position.numpy().reshape(E, G, 3)
    for e in np.flatnonzero(seq.valid.numpy()):
        n = int(seq.count.numpy()[e])
        for i in range(n):
            # sample j = i*spg sits exactly on gate i (CR interpolates knots)
            np.testing.assert_allclose(ctr[e, i * spg], gp[e, i], atol=1e-5)


def test_arclen_monotone_and_closed():
    seq, cfg = _seq(z_profile="uniform", z_min=0.5, z_max=1.5)
    line = CourseLine(seq, samples_per_gate=8)
    line.refresh()
    n_max = int(cfg.max_gates) * 8
    arc = line.track.arclen.numpy().reshape(E, n_max)
    length = line.track.length.numpy()
    cnt = line.track.count.numpy()
    for e in np.flatnonzero(seq.valid.numpy()):
        m = int(cnt[e])
        a = arc[e, :m]
        assert (np.diff(a) > 0).all()
        assert length[e] > a[-1] > 0.0


def test_refresh_tracks_regeneration():
    seq, cfg = _seq(z_profile="uniform", z_min=0.5, z_max=1.5)
    line = CourseLine(seq, samples_per_gate=4)
    line.refresh()
    before = line.track.center.numpy().copy()
    # GateGenerator overwrites in place on regenerate; rerun + refresh
    # (fresh rng advance via new seeds)
    rng2 = PerEnvSeededRNG(seeds=99, num_envs=E, device="cpu")
    seq2 = GateGenerator(cfg, rng2).generate()
    line2 = CourseLine(seq2, samples_per_gate=4)
    line2.refresh()
    assert not np.allclose(before, line2.track.center.numpy(), equal_nan=True)
