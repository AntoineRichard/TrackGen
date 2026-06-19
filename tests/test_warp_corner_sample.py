# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Property-based tests for warp_pipeline.corner_sample (Warp built-in RNG).

This is an ACCEPTED RNG REDESIGN: the pure-Warp corner sampler does NOT match the
torch oracle (_sample_corner_points) bit-for-bit. It is validated by structural
properties only (shape/finiteness/box bounds/reproducibility/diversity/spread).
"""
import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen._src import warp_pipeline as wpl
from track_gen._src.types import TrackGenConfig

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bounds(config):
    """Per-axis [lo, hi] box that every sampled coord must fall inside.

    x/y cell indices lie in [0, num_cells); noise in [-0.5, 0.5); coord =
    (cell_idx * cell_size + noise) * scale. So lo = (-0.5)*scale and
    hi = ((num_cells-1)*cell_size + 0.5)*scale.
    """
    num_cells = int(1.0 / (config.min_point_distance * 2))
    cell_size = config.min_point_distance * 2.0
    lo = (-0.5) * config.scale
    hi = ((num_cells - 1) * cell_size + 0.5) * config.scale
    return lo, hi, num_cells, cell_size


@pytest.mark.parametrize("dev", DEVS)
def test_shape_and_finite(dev):
    config = TrackGenConfig()
    E = 6
    seeds = torch.arange(E, device=dev)
    out = wpl.corner_sample(seeds, 0, config)
    assert out.shape == (E, config.max_num_points, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dev", DEVS)
def test_in_box(dev):
    config = TrackGenConfig()
    E = 6
    seeds = torch.arange(E, device=dev)
    out = wpl.corner_sample(seeds, 0, config)
    lo, hi, _, _ = _bounds(config)
    assert (out >= lo - 1e-6).all()
    assert (out <= hi + 1e-6).all()


@pytest.mark.parametrize("dev", DEVS)
def test_reproducible(dev):
    config = TrackGenConfig()
    E = 6
    seeds = torch.arange(E, device=dev)
    a = wpl.corner_sample(seeds, 0, config)
    b = wpl.corner_sample(seeds, 0, config)
    assert torch.equal(a, b)


@pytest.mark.parametrize("dev", DEVS)
def test_env_diversity(dev):
    config = TrackGenConfig()
    E = 4
    seeds = torch.arange(E, device=dev)
    out = wpl.corner_sample(seeds, 0, config)
    # env 0 vs env 1 must not be identical.
    assert not torch.equal(out[0], out[1])


@pytest.mark.parametrize("dev", DEVS)
def test_attempt_diversity(dev):
    config = TrackGenConfig()
    E = 4
    seeds = torch.arange(E, device=dev)
    a = wpl.corner_sample(seeds, 0, config)
    b = wpl.corner_sample(seeds, 1, config)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("dev", DEVS)
def test_dedup_distinct_cells(dev):
    # The additive per-corner noise (±0.5 vs cell_size≈0.1) makes the chosen cell
    # unrecoverable from the scaled position, so we inspect the chosen CELLS directly
    # via _corner_sample_raw. With the default grid (num_cells**2 = 100 cells) and only
    # P=13 corners, bounded duplicate rejection (8 retries) should resolve every
    # collision -> all 13 cells distinct per env.
    config = TrackGenConfig()
    E = 8
    P = config.max_num_points
    seeds = torch.arange(E, device=dev)
    _, _, num_cells, _ = _bounds(config)
    assert num_cells * num_cells >= P  # precondition for full dedup

    _, cells = wpl._corner_sample_raw(seeds, 0, config)
    assert cells.shape == (E, P)
    assert (cells >= 0).all() and (cells < num_cells * num_cells).all()
    for e in range(E):
        ids = cells[e].tolist()
        assert len(set(ids)) == P, f"env {e}: {P - len(set(ids))} duplicate cell(s): {ids}"


@pytest.mark.parametrize("dev", DEVS)
def test_grid_mapping_matches_oracle(dev):
    # Lock the exact coordinate construction against the chosen cells: with cell known,
    # coord/scale - cell_coord*cell_size must equal the per-corner noise, which lies in
    # [-0.5, 0.5). This pins x=cell%num_cells, y=cell//num_cells, *cell_size, +noise,
    # *scale — the same geometry as generators._sample_corner_points.
    config = TrackGenConfig()
    E = 6
    seeds = torch.arange(E, device=dev)
    _, _, num_cells, cell_size = _bounds(config)

    out, cells = wpl._corner_sample_raw(seeds, 0, config)
    cx = (cells % num_cells).to(torch.float32)
    cy = (cells // num_cells).to(torch.float32)
    cell_coord = torch.stack([cx, cy], dim=-1)  # [E, P, 2]
    noise = out / config.scale - cell_coord * cell_size
    assert (noise >= -0.5 - 1e-6).all() and (noise < 0.5 + 1e-6).all()
