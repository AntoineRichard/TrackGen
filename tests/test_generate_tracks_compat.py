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
    # constant_spacing: the shim returns track.center, now [E, N_max, 2] NaN-padded
    # with a per-env real-point count in track.count. The shaped-tensor contract is
    # therefore against N_max (set explicitly for determinism), and the real points
    # (masked to [:count[e]]) must be finite while the padding is NaN.
    E, N_max = 5, 192
    cfg = TrackGenConfig(
        generator="bezier", num_envs=E, num_points=64, N_max=N_max, device="cpu"
    )
    rng = _make_rng(E)

    centerline = generate_tracks(E, config=cfg, rng=rng)

    assert isinstance(centerline, torch.Tensor)
    assert centerline.shape == (E, N_max, 2)

    # Per-env real-point counts: derive the count from the finite mask (the shim drops
    # the count tensor, so recover it from the NaN padding) and verify it is a sane,
    # varying-length prefix with finite head and NaN tail.
    finite = torch.isfinite(centerline).all(dim=-1)  # [E, N_max]
    counts = finite.sum(dim=1)
    assert (counts >= 3).all(), "each track needs at least a few real points"
    assert (counts <= N_max).all()
    for e in range(E):
        c = int(counts[e])
        # The finite points form a contiguous prefix [:c]; the rest is NaN padding.
        assert finite[e, :c].all(), "real points must be a finite leading prefix"
        assert not finite[e, c:].any(), "padding beyond count must be NaN"


def test_compat_shim_emits_deprecation_warning():
    E = 3
    cfg = TrackGenConfig(
        generator="bezier", num_envs=E, num_points=32, N_max=128, device="cpu"
    )
    rng = _make_rng(E)

    with pytest.warns(DeprecationWarning):
        generate_tracks(E, config=cfg, rng=rng)
