"""ProgressTracker vs numpy oracle on generated gates AND track checkpoints."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests._progress_oracle import ProgressOracle
from track_gen.progress import ProgressTracker

STEPS = 60
FIELDS = ("passed", "checkpoint_passed", "next_checkpoint", "laps",
          "progress", "wrong_way", "wrong_checkpoint")


def _run_and_compare(cps, E, M, centers, rng):
    """Random-walk positions around each env's course; compare every field."""
    tracker = ProgressTracker(cps)
    counts = cps.count.numpy()
    pos_np = cps.position.numpy().reshape(E, M, 2)
    left_np = cps.left.numpy().reshape(E, M, 2)
    right_np = cps.right.numpy().reshape(E, M, 2)
    tang_np = cps.tangent.numpy().reshape(E, M, 2)
    oracles = {}
    for e in range(E):
        n = int(counts[e])
        if n >= 1:
            oracles[e] = ProgressOracle(pos_np[e, :n], left_np[e, :n],
                                        right_np[e, :n], tang_np[e, :n])
    walk = centers + rng.normal(0.0, 0.35, (STEPS, E, 2))
    for s in range(STEPS):
        p = wp.array(walk[s].astype(np.float32), dtype=wp.vec2f, device="cpu")
        ev = tracker.update(p)
        got = {f: getattr(ev, f).numpy() for f in FIELDS}
        dist = ev.dist_to_next.numpy()
        for e, oracle in oracles.items():
            ref = oracle.update(walk[s, e])
            for f in FIELDS:
                assert got[f][e] == ref[f], f"step {s} env {e} field {f}"
            np.testing.assert_allclose(dist[e], ref["dist_to_next"], atol=1e-4)
    assert oracles, "no env with checkpoints"


def test_oracle_on_generated_gates():
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.checkpoints import CheckpointSet
    E = 4
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.15)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=21, num_envs=E, device="cpu"))
    seq = gen.generate()
    cps = CheckpointSet.from_gates(seq)
    M = cps.position.shape[0] // E
    valid = seq.valid.numpy().astype(bool)
    counts = cps.count.numpy()
    pos = np.nan_to_num(cps.position.numpy().reshape(E, M, 2), nan=0.0)
    centers = np.zeros((E, 2))
    for e in range(E):
        if valid[e] and counts[e] > 0:
            centers[e] = pos[e, :counts[e]].mean(axis=0)
    # Only valid envs are compared (undefined otherwise): zero out others.
    cps2 = cps.clone()
    cnp = cps2.count.numpy()
    cnp[~valid] = 0
    wp.copy(cps2.count, wp.array(cnp, dtype=wp.int32, device="cpu"))
    _run_and_compare(cps2, E, M, centers, np.random.default_rng(1))


def test_oracle_on_track_checkpoints():
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.checkpoints import CheckpointSampler
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=31, num_envs=E, device="cpu"))
    track = gen.generate()
    sampler = CheckpointSampler(track, spacing=0.6, max_checkpoints=48)
    cps = sampler.sample()
    M = sampler._M
    valid = track.valid.numpy().astype(bool)
    counts = cps.count.numpy()
    pos = np.nan_to_num(cps.position.numpy().reshape(E, M, 2), nan=0.0)
    centers = np.zeros((E, 2))
    for e in range(E):
        if valid[e] and counts[e] > 0:
            centers[e] = pos[e, :counts[e]].mean(axis=0)
    cps2 = cps.clone()
    cnp = cps2.count.numpy()
    cnp[~valid] = 0
    wp.copy(cps2.count, wp.array(cnp, dtype=wp.int32, device="cpu"))
    _run_and_compare(cps2, E, M, centers, np.random.default_rng(2))
