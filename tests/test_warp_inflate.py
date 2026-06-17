# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""inflate_warp == torch inflation.inflate on a clean fixed-N centerline.

Builds two closed loops (a fat circle -> valid, a thin ellipse -> invalid) and
checks the pure-Warp inflate composition field-by-field against the torch oracle
on both the Warp cpu and cuda devices.
"""

import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen import inflation  # noqa: E402
from track_gen import warp_pipeline as wpl  # noqa: E402
from track_gen.generators import Centerline  # noqa: E402
from track_gen.types import Track, TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _build_center(dev: str) -> torch.Tensor:
    """[2, 256, 2]: a fat circle (r=2) and a thin ellipse (x=2cos, y=0.04sin)."""
    n = 256
    t = torch.linspace(0.0, 2.0 * math.pi, n + 1, device=dev)[:-1]  # closed, no dup
    circle = torch.stack([2.0 * torch.cos(t), 2.0 * torch.sin(t)], dim=-1)
    ellipse = torch.stack([2.0 * torch.cos(t), 0.04 * torch.sin(t)], dim=-1)
    return torch.stack([circle, ellipse], dim=0).to(torch.float32)


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_matches_oracle(dev):
    center = _build_center(dev)
    config = TrackGenConfig(half_width=0.1, num_points=256)
    valid = torch.ones(2, dtype=torch.bool, device=dev)

    ref = inflation.inflate(Centerline(points=center, valid=valid), config)
    got = wpl.inflate_warp(center, config, valid)

    assert isinstance(got, Track)

    # Positions / frame: Warp float32 sqrt vs torch differs by ~ULP per point.
    for field in ("center", "outer", "inner", "tangent", "normal"):
        g = getattr(got, field)
        r = getattr(ref, field)
        assert torch.allclose(g, r, atol=1e-4), f"{field} mismatch on {dev}"

    # arclen/length: a float32-accumulated cumulative sum (vs the oracle's
    # torch.cumsum) drifts more than the per-point fields, hence the looser atol.
    assert torch.allclose(got.arclen, ref.arclen, atol=1e-3), f"arclen on {dev}"
    assert torch.allclose(got.length, ref.length, atol=1e-3), f"length on {dev}"

    assert torch.equal(got.valid.cpu(), ref.valid.cpu()), f"valid on {dev}"
    assert torch.equal(got.count.cpu(), ref.count.cpu()), f"count on {dev}"

    # Circle is fat enough to be valid; thin ellipse is not.
    assert got.valid.cpu().tolist() == ref.valid.cpu().tolist()
    assert got.valid.cpu().tolist() == [True, False]


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_valid_defaults_to_all_true(dev):
    # Omitting `valid` must behave like the oracle with an all-True generation flag.
    center = _build_center(dev)
    config = TrackGenConfig(half_width=0.1, num_points=256)

    ref = inflation.inflate(
        Centerline(points=center, valid=torch.ones(2, dtype=torch.bool, device=dev)), config
    )
    got = wpl.inflate_warp(center, config)  # valid omitted -> all-True default
    assert torch.equal(got.valid.cpu(), ref.valid.cpu())


def test_inflate_warp_rejects_non_fixed_output_mode():
    # The clean fixed-N contract is enforced, not silently mismatched.
    center = _build_center("cpu")
    config = TrackGenConfig(half_width=0.1, num_points=256, output_mode="constant_spacing")
    with pytest.raises(AssertionError):
        wpl.inflate_warp(center, config)
