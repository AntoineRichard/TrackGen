# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

pytest.importorskip("warp")

from track_gen import PerEnvSeededRNG
from track_gen.track_generator import TrackGenConfig, generate_tracks


def _make_rng(num_envs, device="cpu"):
    import warp as wp
    wp.init()
    return PerEnvSeededRNG(seeds=0, num_envs=num_envs, device=device)


def test_compat_shim_returns_centerline_shaped_tensor():
    E, N = 5, 48
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=N, device="cpu")
    rng = _make_rng(E)

    centerline = generate_tracks(E, config=cfg, rng=rng)

    assert isinstance(centerline, torch.Tensor)
    assert centerline.shape == (E, N, 2)


def test_compat_shim_emits_deprecation_warning():
    E, N = 3, 32
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=N, device="cpu")
    rng = _make_rng(E)

    with pytest.warns(DeprecationWarning):
        generate_tracks(E, config=cfg, rng=rng)
