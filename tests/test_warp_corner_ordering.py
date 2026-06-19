# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Behaviour: prune-then-sort eliminates winding-0 figure-eights at the fat-band regime.

The old sort-then-prune ordering produced ~3.5% figure-eights (winding != +-1) on a single
generation attempt; prune-then-sort drops that to ~0.1%. We assert the figure-8 rate is well
under 1% and record the per-attempt accept rate.
"""
import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen._src import warp_pipeline as wpl  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
def test_prune_then_sort_eliminates_figure_eights(dev):
    E = 1024
    cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device=dev)
    seeds = torch.arange(E, dtype=torch.int32, device=dev)

    # Reproduce generate_centerline_warp's attempt-0 corner pipeline (prune-then-sort).
    count = wpl.corner_count_sample(seeds, 0, cfg)
    corners = wpl.ccw_sort(wpl.corner_sample(seeds, 0, cfg), count)
    dense = wpl.assemble(corners, count, cfg)
    rs30, _ = wpl.arc_length_resample_warp(dense, int(cfg.num_points_per_segment))
    turn = wpl.turning_number(rs30)
    turn_ok = (turn.abs() - 2.0 * math.pi).abs() <= float(cfg.turning_tol)

    fig8_rate = 1.0 - turn_ok.float().mean().item()
    assert fig8_rate < 0.01, f"figure-8 rate {fig8_rate:.4f} not < 1% (old sort-then-prune ~3.5%)"

    # Per-attempt accept should clear the old single-attempt baseline (~0.51). Deterministic
    # (fixed seeds), so not flaky; this is a secondary sanity floor -- the figure-8 assertion
    # above is the primary check. Observed ~0.62 (cuda) / ~0.77 (cpu).
    accept = wpl.gates(corners, dense, count, cfg)
    acc = accept.float().mean().item()
    assert acc > 0.55, f"per-attempt accept {acc:.4f} not > 0.55 (old baseline ~0.51)"
