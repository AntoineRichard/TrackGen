# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""gates(): pure-Warp accept mask == the torch oracle inside
BezierCenterlineGenerator.generate (angle & turn & finite & simple conjunction).

Four fixed envs hit each gate cleanly (far from thresholds so the ~5e-4 Warp-vs-torch
resample drift cannot flip a boolean):
  A: clean octagon                       -> all gates pass            -> accept True
  B: octagon with one 2.86-deg spike     -> angle_ok False            -> reject
  C: bowtie corner order (figure-eight)  -> simple_ok & turn_ok False -> reject
  D: count = 1 (degenerate)              -> finite_ok False           -> reject
"""
from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen import geometry  # noqa: E402
from track_gen import warp_pipeline as wpl  # noqa: E402
from track_gen.generators import BezierCenterlineGenerator  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

P = 8  # corners per env (config.max_num_points for this test)


def _octagon(r: float = 2.0):
    return [[r * math.cos(2 * math.pi * i / P), r * math.sin(2 * math.pi * i / P)] for i in range(P)]


def _corner_sets():
    """Fixed raw corner sets + per-env counts giving accept pattern [T, F, F, F]."""
    A = _octagon(2.0)  # clean convex-ish loop: every gate passes.

    B = _octagon(2.0)  # one needle spike: interior angle ~2.86 deg << 12.5 deg min_angle.
    B[0] = [5.0, 0.0]
    B[1] = [1.0, 0.1]
    B[7] = [1.0, -0.1]

    # Bowtie / figure-eight corner order: the dense loop self-crosses and its
    # turning number collapses toward 0 (lobes wind oppositely).
    C = [[-2, -2], [2, 2], [2, -2], [-2, 2], [-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [-1.5, 1.5]]

    D = _octagon(2.0)  # only 1 real corner -> < 2 real dense points -> finite_ok False.

    corners = torch.tensor([A, B, C, D], dtype=torch.float32)
    count = torch.tensor([P, P, P, 1], dtype=torch.long)
    return corners, count


def _oracle_accept(gen, pruned, dense, config):
    """Replicate generate()'s ok = angle_ok & turn_ok & finite_ok & simple_ok."""
    angles = gen._corner_angles(pruned)
    real = torch.isfinite(pruned).all(dim=-1)
    constrained = real & torch.roll(real, 1, dims=1) & torch.roll(real, -1, dims=1)
    angle_ok = ((angles > config.min_angle) | ~constrained).all(dim=1)

    turn, finite_ok = gen._real_turning_and_finite(dense)
    turn_ok = (turn.abs() - 2.0 * math.pi).abs() <= config.turning_tol

    simple_res, _ = geometry.arc_length_resample(dense, num=config.num_points)
    simple_ok = geometry.self_intersections(simple_res) == 0

    return angle_ok & turn_ok & finite_ok & simple_ok


@pytest.mark.parametrize("dev", DEVS)
def test_gates_matches_oracle(dev):
    config = TrackGenConfig()
    config.device = dev          # oracle Bernstein basis -> right device for _assemble_centerline
    config.max_num_points = P

    corners, count = _corner_sets()
    corners = corners.to(dev)
    count = count.to(dev)

    gen = BezierCenterlineGenerator(config, rng=None)

    # Prune (matches _prune_corners' NaN step), then build the SAME dense for both sides.
    row = torch.arange(P, device=dev)
    keep = (row < count[:, None]).unsqueeze(-1)
    pruned = torch.where(keep, corners, torch.full_like(corners, float("nan")))
    dense = gen._assemble_centerline(pruned)

    ref = _oracle_accept(gen, pruned, dense, config)
    got = wpl.gates(corners, dense, count, config)

    assert got.dtype == torch.bool
    assert torch.equal(got.cpu(), ref.cpu())
    # And it matches the intended clear-cut pattern.
    assert got.cpu().tolist() == [True, False, False, False]
