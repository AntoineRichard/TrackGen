"""Tests for the static-regen pure-Warp centerline generator (Task 12).

Covers ``corner_count_sample`` (per-(env, attempt) Warp-RNG corner count) and
``generate_centerline_warp`` (fixed-iteration masked accept-first-valid generation
fused with the final arc-length resample to ``num_points``). Reproducibility is
asserted WITHIN a device only (Warp RNG may legitimately differ cpu vs cuda).
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("warp")

from track_gen._src import warp_pipeline as wpl  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
def test_corner_count_sample(dev):
    config = TrackGenConfig(num_envs=256)
    E = 256
    seeds = torch.arange(E, device=dev)

    count = wpl.corner_count_sample(seeds, 0, config)

    # shape + dtype
    assert count.shape == (E,)
    assert count.dtype == torch.int32
    # range: every count in [min_num_points, max_num_points] inclusive
    assert int(count.min()) >= int(config.min_num_points)
    assert int(count.max()) <= int(config.max_num_points)

    # reproducible: same seeds/attempt -> equal
    count2 = wpl.corner_count_sample(seeds, 0, config)
    assert torch.equal(count, count2)

    # different attempts differ (somewhere across the 256 envs)
    count_a1 = wpl.corner_count_sample(seeds, 1, config)
    assert not torch.equal(count, count_a1)


@pytest.mark.parametrize("dev", DEVS)
def test_generate_centerline_warp(dev):
    config = TrackGenConfig(num_envs=256)
    E = 256
    N = int(config.num_points)
    seeds = torch.arange(E, device=dev)

    centerline, valid = wpl.generate_centerline_warp(seeds, config)

    # shapes + dtypes
    assert centerline.shape == (E, N, 2)
    assert valid.shape == (E,)
    assert valid.dtype == torch.bool

    # Single-pass + Fix B (no regen, no generation gating): every env is gen-valid and gets a
    # real centerline (final validity is decided post-relaxation, not here).
    assert valid.all()
    assert torch.isfinite(centerline).all()

    # Fix B (self-crossing track -> corner-polygon fallback) + the collinear-robust detector:
    # the generated centerline is ~always simple.
    cf = (wpl.self_intersections(centerline) == 0).float().mean().item()
    assert cf >= 0.99, f"crossing-free {cf} < 0.99 on {dev}"

    # reproducible within a device
    centerline2, valid2 = wpl.generate_centerline_warp(seeds, config)
    assert torch.equal(valid, valid2)
    assert torch.equal(centerline, centerline2)
