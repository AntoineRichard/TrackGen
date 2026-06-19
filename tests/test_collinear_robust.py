# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""self_intersections must be robust to near-collinear segments.

The f32 proper-crossing test false-positives when two non-adjacent segments are nearly
collinear (the orientation determinant is a tiny true-~0 that f32 rounding flips). This is
most visible on polygonal (straight-piece) centerlines, which are PROVABLY simple (the
corner polygon never self-crosses) yet the raw f32 detector reports crossings on them.
A scale-relative collinearity tolerance fixes it without missing genuine crossings.
"""
import pytest
import torch

from track_gen.geometry import self_intersections


def test_genuine_crossing_still_detected():
    """A bow-tie quad properly self-crosses -> must still be detected."""
    bowtie = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])  # [1,4,2]
    assert int(self_intersections(bowtie)[0]) > 0


def test_collinear_simple_loop_not_flagged():
    """A simple polygon densely resampled (many near-collinear segments) is simple -> 0."""
    # A non-convex hexagon at metric scale, each edge subdivided into ~20 collinear points.
    hexagon = torch.tensor([[10.0, 0.0], [6.0, 7.0], [-5.0, 8.0],
                            [-8.0, 1.0], [-3.0, -6.0], [7.0, -7.0]])
    pts = []
    n = hexagon.shape[0]
    for i in range(n):
        a = hexagon[i]; b = hexagon[(i + 1) % n]
        for t in torch.linspace(0, 1, 21)[:-1]:
            pts.append(a * (1 - t) + b * t)
    loop = torch.stack(pts).unsqueeze(0)            # [1, 120, 2] simple polygon
    assert int(self_intersections(loop.double())[0]) == 0          # truth (f64)
    assert int(self_intersections(loop)[0]) == 0                   # f32 must agree (collinear-robust)


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_polygonal_fallback_not_flagged(dev):
    """The corner-polygon fallback (Fix B) is provably simple; neither detector may flag it."""
    pytest.importorskip("warp")
    import dataclasses
    import warp as wp
    wp.init()
    from track_gen import warp_pipeline as wpl
    from track_gen.types import TrackGenConfig

    E = 2048
    cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device=dev)
    seeds = torch.arange(E, dtype=torch.int32, device=dev)
    cc = wpl.corner_count_sample(seeds, 0, cfg)
    corners = wpl.ccw_sort(wpl.corner_sample(seeds, 0, cfg), cc)
    cfg0 = dataclasses.replace(cfg, handle_clamp_frac=0.0)          # polygonal
    dense0 = wpl.assemble(corners, cc, cfg0)
    cl0, _ = wpl.arc_length_resample_warp(dense0, 256)

    truth = self_intersections(cl0.double())                       # f64 = 0 everywhere
    assert int((truth > 0).sum()) == 0
    # torch f32 detector must match the truth (no collinear false positives)
    assert int((self_intersections(cl0) > 0).sum()) == 0
    # warp f32 detector likewise
    assert int((wpl.self_intersections(cl0) > 0).sum()) == 0
