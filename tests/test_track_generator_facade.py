# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

pytest.importorskip("warp")

from track_gen import PerEnvSeededRNG
from track_gen.track_generator import Track, TrackGenConfig, TrackGenerator


def _make_rng(num_envs, device="cpu"):
    import warp as wp
    wp.init()
    return PerEnvSeededRNG(seeds=0, num_envs=num_envs, device=device)


def test_bezier_path_returns_track_with_aligned_boundaries():
    E, N = 4, 64
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=N, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)

    assert isinstance(track, Track)
    assert track.outer.shape == (E, N, 2)
    assert track.center.shape == (E, N, 2)
    assert track.inner.shape == (E, N, 2)
    assert track.valid.shape == (E,)
    assert track.valid.dtype == torch.bool


def test_fourier_generator_is_routed():
    E, N = 4, 64
    cfg = TrackGenConfig(generator="fourier", num_envs=E, num_points=N, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    from track_gen.generators import FourierCenterlineGenerator

    assert isinstance(gen._generator, FourierCenterlineGenerator)

    track = gen.generate(E)
    assert isinstance(track, Track)
    assert track.center.shape == (E, N, 2)


def test_unknown_generator_raises():
    cfg = TrackGenConfig(generator="spline", num_envs=2, device="cpu")
    rng = _make_rng(2)
    with pytest.raises(ValueError):
        TrackGenerator(cfg, rng)
