"""Tests for the single-pass pure-Warp centerline generator.

Covers ``corner_count_sample`` (per-(env, attempt) Warp-RNG corner count) and
``generate_centerline_warp`` (single-pass generation with Fix B polygon fallback,
routed through the ``tests/_warp_compare`` helpers). Reproducibility is asserted
WITHIN a device only (Warp RNG may legitimately differ cpu vs cuda).
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402
wp.init()
from track_gen._src import warp_pipeline as wpl  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402
from tests._warp_compare import corner_count_sample, self_intersections  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
def test_corner_count_sample(dev):
    config = TrackGenConfig(num_envs=256)
    E = 256
    seeds = torch.arange(E, device=dev)

    count = corner_count_sample(seeds, 0, config)

    # shape + dtype
    assert count.shape == (E,)
    assert count.dtype == torch.int32
    # range: every count in [min_num_points, max_num_points] inclusive
    assert int(count.min()) >= int(config.min_num_points)
    assert int(count.max()) <= int(config.max_num_points)

    # reproducible: same seeds/attempt -> equal
    count2 = corner_count_sample(seeds, 0, config)
    assert torch.equal(count, count2)

    # different attempts differ (somewhere across the 256 envs)
    count_a1 = corner_count_sample(seeds, 1, config)
    assert not torch.equal(count, count_a1)


@pytest.mark.parametrize("dev", DEVS)
def test_generate_centerline_warp(dev):
    config = TrackGenConfig(num_envs=256, device=dev)
    E = 256
    N = int(config.num_points)

    # Allocate buffers via the owned pre-alloc helper (uses config.device).
    _, scratch = wpl._inflate_warp_alloc(config)

    # seeds_wp: [E] int32 wp.array on same device as config
    seeds_t = torch.arange(E, dtype=torch.int32, device=dev)
    seeds_wp = wp.from_torch(seeds_t, dtype=wp.int32)

    wpl.generate_centerline_warp(
        seeds_wp, config,
        out_centerline=scratch.gen_centerline,
        out_valid_wp=scratch.gen_valid,
        scratch=scratch,
    )
    if "cuda" in dev:
        wp.synchronize()

    centerline = wp.to_torch(scratch.gen_centerline).view(E, N, 2)
    valid = wp.to_torch(scratch.gen_valid).bool()

    # shapes + dtypes
    assert centerline.shape == (E, N, 2)
    assert valid.shape == (E,)

    # Single-pass + Fix B (no regen, no generation gating): every env is gen-valid and gets a
    # real centerline (final validity is decided post-relaxation, not here).
    assert valid.all()
    assert torch.isfinite(centerline).all()

    # Fix B (self-crossing track -> corner-polygon fallback) + the collinear-robust detector:
    # the generated centerline is ~always simple.
    cf = (self_intersections(centerline) == 0).float().mean().item()
    assert cf >= 0.99, f"crossing-free {cf} < 0.99 on {dev}"

    # reproducible within a device: re-run with same seeds, same scratch
    _, scratch2 = wpl._inflate_warp_alloc(config)
    seeds_wp2 = wp.from_torch(seeds_t.clone(), dtype=wp.int32)
    wpl.generate_centerline_warp(
        seeds_wp2, config,
        out_centerline=scratch2.gen_centerline,
        out_valid_wp=scratch2.gen_valid,
        scratch=scratch2,
    )
    if "cuda" in dev:
        wp.synchronize()

    centerline2 = wp.to_torch(scratch2.gen_centerline).view(E, N, 2)
    valid2 = wp.to_torch(scratch2.gen_valid).bool()
    assert torch.equal(valid, valid2)
    assert torch.equal(centerline, centerline2)
