# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import pytest
from track_gen import geometry


@pytest.fixture
def warp_rng():
    pytest.importorskip("warp")
    import warp as wp; wp.init()
    from track_gen.rng_utils import PerEnvSeededRNG

    def make(E, seed=20):
        seeds = torch.arange(E, dtype=torch.int32) + seed
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
        rng.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
        return rng
    return make


def test_xpbd_pipeline_makes_constant_width_tracks_valid(warp_rng):
    from track_gen.types import TrackGenConfig
    from track_gen.track_generator import TrackGenerator
    E = 32
    cfg = TrackGenConfig(generator="bezier", device="cpu", num_envs=E, scale=1.0,
                         half_width=0.03, num_points=256, output_mode="fixed",
                         relax_solver="xpbd", relax_iters=200, relax_bend_relax=1.5,
                         relax_margin=0.15, max_regen_iters=20)
    track = TrackGenerator(cfg, warp_rng(E)).generate(E)
    # Relaxed + constant-width inflation: a large majority must be valid (was ~3% before).
    assert track.valid.float().mean().item() >= 0.9
    # Width is constant where valid.
    w = torch.linalg.norm(track.outer - track.center, dim=-1)
    assert torch.allclose(w, torch.full_like(w, 0.03), atol=1e-3)
