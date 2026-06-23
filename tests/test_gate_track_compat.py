import pytest
import torch

pytest.importorskip("warp")

from track_gen import (
    GateGenConfig,
    GateGenerator,
    PerEnvSeededRNG,
    TrackGenConfig,
    TrackGenerator,
)
from tests._warp_compare import to_t


def _track_snapshot(seed=123):
    E = 4
    cfg = TrackGenConfig(
        generator="bezier",
        num_envs=E,
        num_points=64,
        N_max=128,
        device="cpu",
    )
    rng = PerEnvSeededRNG(seeds=seed, num_envs=E, device="cpu")
    track = TrackGenerator(cfg, rng).generate(E).clone()
    return (
        to_t(track.center).clone(),
        to_t(track.outer).clone(),
        to_t(track.inner).clone(),
        to_t(track.tangent).clone(),
        to_t(track.normal).clone(),
        to_t(track.arclen).clone(),
        to_t(track.length).clone(),
        to_t(track.valid).clone(),
        to_t(track.count).clone(),
    )


def test_gate_generation_does_not_change_track_generation_outputs():
    before = _track_snapshot()

    gate_cfg = GateGenConfig(num_envs=4, max_gates=32, device="cpu", min_gate_distance=0.0)
    gate_rng = PerEnvSeededRNG(seeds=77, num_envs=4, device="cpu")
    GateGenerator(gate_cfg, gate_rng).generate(4)

    after = _track_snapshot()

    for lhs, rhs in zip(before, after):
        if lhs.is_floating_point():
            assert torch.equal(torch.isnan(lhs), torch.isnan(rhs))
            assert torch.equal(lhs[~torch.isnan(lhs)], rhs[~torch.isnan(rhs)])
        else:
            assert torch.equal(lhs, rhs)
