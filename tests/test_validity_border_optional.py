# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The validity border self_intersections check is OPTIONAL and default-OFF.

It is redundant with the thickness/separation gate (a self-crossing or fat-band overlap
drives separation_min -> 0 -> thickness < half_width -> invalid), so disabling it does not
change the valid mask. The flag exists only to re-enable the extra O(N^2) check if wanted.
"""
import dataclasses

import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen._src import warp_pipeline as wpl  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_validity_border_check_defaults_off():
    assert TrackGenConfig().validity_border_check is False


@pytest.mark.parametrize("dev", DEVS)
def test_border_check_is_redundant_so_default_off_loses_nothing(dev):
    cfg = TrackGenConfig(num_envs=512, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device=dev)
    seeds = torch.arange(512, dtype=torch.int32, device=dev)
    v_off = wpl.generate_tracks_warp(cfg, seeds).valid                          # default: border check off
    v_on = wpl.generate_tracks_warp(dataclasses.replace(cfg, validity_border_check=True), seeds).valid
    # identical -> the border self_intersections adds no rejections beyond thickness/separation
    assert torch.equal(v_off, v_on)
