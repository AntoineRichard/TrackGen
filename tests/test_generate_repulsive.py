"""End-to-end tests for the repulsive-growth first-stage generator.

Task 3 lands the seed-driven obstacle-layout determinism test first; Task 4 extends
this module with the full per-generator contract (registration, output shape, closed
loop, NaN-tail, determinism/diversity, yield, compactness, no-graph on CUDA).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402

from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src import warp_generate_repulsive as rep  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _sample(seeds: np.ndarray, config, dev: str):
    """Run the obstacle kernel; return (obs_pts [E,M,2], obs_mw [E,M]) numpy arrays."""
    s = rep._RepulsiveScalars(config)
    E, M = s.E, s.M
    seeds_wp = wp.array(seeds.astype(np.int32), dtype=wp.int32, device=dev)
    obs_pts = wp.empty(E * M, dtype=wp.vec2f, device=dev)
    obs_mw = wp.empty(E * M, dtype=wp.float32, device=dev)
    rep._sample_obstacles_inplace(seeds_wp, config, obs_pts, obs_mw, dev)
    pts = obs_pts.numpy().reshape(E, M, 2)
    mw = obs_mw.numpy().reshape(E, M)
    return pts, mw, s


@pytest.mark.parametrize("dev", DEVS)
def test_obstacle_layout_deterministic(dev):
    E = 32
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
    seeds_a = np.arange(E) + 7
    seeds_b = np.arange(E) + 1000  # distinct seeds

    pts1, mw1, s = _sample(seeds_a, cfg, dev)
    pts2, mw2, _ = _sample(seeds_a, cfg, dev)
    pts3, mw3, _ = _sample(seeds_b, cfg, dev)

    # Determinism: identical seeds -> byte-identical layout (NaN pattern must match too).
    assert np.array_equal(np.nan_to_num(pts1), np.nan_to_num(pts2))
    assert np.array_equal(mw1, mw2)
    assert np.array_equal(np.isnan(pts1), np.isnan(pts2))
    # Diversity: distinct seeds -> different layout.
    assert not np.array_equal(mw1, mw3)

    n_wall, n_disc, M = s.n_wall, s.n_disc, s.M
    wall_mw = 2.0 * np.pi * s.r_dom / n_wall  # weight 1.0 baked into the wall arc mass

    # Wall columns: finite points on the r_dom ring, obs_mw == wall arc mass (weight 1.0).
    wall_pts = pts1[:, :n_wall]
    wall_mw_arr = mw1[:, :n_wall]
    assert np.isfinite(wall_pts).all()
    assert np.allclose(wall_mw_arr, wall_mw, rtol=1e-4)
    assert np.allclose(np.linalg.norm(wall_pts, axis=2), s.r_dom, rtol=1e-4)

    disc_mw = mw1[:, n_wall:].reshape(E, -1, n_disc)  # [E, K_max, n_disc]
    disc_pts = pts1[:, n_wall:].reshape(E, -1, n_disc, 2)
    # Each disc column block is uniform (all n_disc points share one obs_mw).
    for e in range(E):
        for j in range(disc_mw.shape[1]):
            block = disc_mw[e, j]
            assert np.allclose(block, block[0])
            if block[0] == 0.0:
                # Unused / skipped column: NaN-padded points.
                assert np.isnan(disc_pts[e, j]).all()
            else:
                # Active disc column: recover the ring radius from the chord and confirm
                # the weight factor is 0.25 (obs_mw == 0.25 * 2*pi*r_ring / n_disc).
                ring = disc_pts[e, j]
                chord = np.linalg.norm(ring[1] - ring[0])
                r_ring = chord / (2.0 * np.sin(np.pi / n_disc))
                weight = block[0] * n_disc / (2.0 * np.pi * r_ring)
                assert np.isclose(weight, 0.25, rtol=1e-3)

    # Every finite obstacle point lies within the domain radius.
    finite = np.isfinite(pts1).all(axis=2)
    norms = np.linalg.norm(np.nan_to_num(pts1), axis=2)
    assert (norms[finite] <= s.r_dom + 1e-4).all()
    # At default config every env places at least one inner disc (k_min=8 discs fit).
    assert (mw1[:, n_wall:] > 0.0).any(axis=1).all()
