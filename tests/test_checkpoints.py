"""Analytic annulus + aliasing tests for track_gen.checkpoints."""
from __future__ import annotations

import numpy as np
import pytest

from tests._collision_fixtures import annulus_polylines, make_annulus_track

N = 512
N_MAX = N + 8
RC, RI, RO = 1.0, 0.7, 1.3  # annulus fixture center/inner/outer radii


def _center_perimeter(track, e=0):
    m = int(track.count.numpy()[e])
    center = track.center.numpy().reshape(-1, N_MAX, 3)[e, :m, :2]
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    return float(seg.sum())


def test_import_surface():
    import track_gen
    from track_gen.checkpoints import CheckpointSampler, CheckpointSet  # noqa: F401
    assert "checkpoints" in track_gen.__all__


def test_sampler_snap_count_and_positions_on_centerline():
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=1, n=N)
    spacing = 0.8  # coarse: gate-like checkpoint counts
    sampler = CheckpointSampler(track, spacing=spacing)
    cps = sampler.sample()
    perim = _center_perimeter(track)
    n = int(cps.count.numpy()[0])
    assert n == int(round(perim / spacing))
    np.testing.assert_allclose(sampler.step.numpy()[0], perim / n, rtol=1e-5)
    pos = cps.position.numpy().reshape(-1, 3)[:n]
    assert np.all(pos[:, 2] == 0.0)
    np.testing.assert_allclose(np.linalg.norm(pos[:, :2], axis=1), RC, atol=2e-3)


def test_crossing_segments_are_road_cross_sections():
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=1, n=N)
    cps = CheckpointSampler(track, spacing=0.8).sample()
    n = int(cps.count.numpy()[0])
    left = cps.left.numpy().reshape(-1, 3)[:n, :2]
    right = cps.right.numpy().reshape(-1, 3)[:n, :2]
    pos = cps.position.numpy().reshape(-1, 3)[:n, :2]
    tang = cps.tangent.numpy().reshape(-1, 3)[:n, :2]
    up_half = cps.up_half.numpy()[:n]
    assert np.all(up_half >= 1.0e29)  # track cross-sections: unbounded (_BIG)
    # left on the inner circle, right on the outer circle, radially aligned
    # with the checkpoint position (annulus: cross-sections are radial).
    np.testing.assert_allclose(np.linalg.norm(left, axis=1), RI, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(right, axis=1), RO, atol=2e-3)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    np.testing.assert_allclose(left, RI * radial, atol=5e-3)
    np.testing.assert_allclose(right, RO * radial, atol=5e-3)
    # tangent unit, perpendicular to radial (CCW travel direction).
    np.testing.assert_allclose(np.linalg.norm(tang, axis=1), 1.0, atol=1e-5)
    assert np.abs((tang * radial).sum(axis=1)).max() < 0.02


def test_nan_padding_and_degenerate_env():
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=2, n=N, counts=[N, 2])
    sampler = CheckpointSampler(track, spacing=0.8, max_checkpoints=32)
    cps = sampler.sample()
    counts = cps.count.numpy()
    assert counts[0] > 0 and counts[1] == 0
    M = sampler._M
    pos = cps.position.numpy().reshape(-1, 3)
    assert np.all(np.isnan(pos[counts[0]:M]))       # env 0 tail
    assert np.all(np.isnan(pos[M:2 * M]))           # env 1: all NaN
    up_half = cps.up_half.numpy()
    assert np.all(np.isnan(up_half[counts[0]:M]))
    assert np.all(np.isnan(up_half[M:2 * M]))


def test_from_gates_is_zero_copy_alias():
    import warp as wp
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.checkpoints import CheckpointSet
    E = 2
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.05)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=5, num_envs=E, device="cpu"))
    seq = gen.generate()
    cps = CheckpointSet.from_gates(seq)
    # Zero-copy: same underlying buffers, not copies.
    assert cps.position.ptr == seq.position.ptr
    assert cps.left.ptr == seq.left.ptr
    assert cps.right.ptr == seq.right.ptr
    assert cps.tangent.ptr == seq.tangent.ptr
    assert cps.up_half.ptr == seq.half_size.ptr
    assert cps.count.ptr == seq.count.ptr
    # Mutating the gate buffer is visible through the set (aliasing contract).
    wp.copy(seq.position, wp.zeros_like(seq.position))
    assert float(np.nanmax(np.abs(cps.position.numpy()))) == 0.0


def test_sampler_validation_and_derivation():
    import warp as wp
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=1, n=N)
    with pytest.raises(ValueError, match="spacing"):
        CheckpointSampler(track, spacing=0.0)
    with pytest.raises(ValueError, match="spacing"):
        CheckpointSampler(track, spacing=float("nan"))
    with pytest.raises(ValueError, match="max_checkpoints"):
        CheckpointSampler(track, spacing=0.8, max_checkpoints=2)
    perim = _center_perimeter(track)
    sampler = CheckpointSampler(track, spacing=0.8)
    assert sampler._M == max(3, int(np.ceil(1.5 * perim / 0.8)))
    wp.copy(track.valid, wp.zeros(1, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="max_checkpoints"):
        CheckpointSampler(track, spacing=0.8)
