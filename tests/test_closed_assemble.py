# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""F1 (proper closed assemble) + F2 (adaptive handle clamp).

F1: the assemble must close the loop with a REAL cubic Bezier over all `count` corners
(wrap mod count) instead of dropping the first & last corner and closing with a straight
chord. Concretely, every one of the `count` segments -- including the closing
(count-1 -> 0) -- is a real Bezier, so the dense buffer has exactly count*npseg finite
samples (today's drop-2/straight-close keeps only (count-3)*npseg).

Oracle<->Warp parity is guarded by tests/test_warp_assemble.py (unchanged): both paths get
the same closed behaviour, so they must still match bit-for-bit.
"""
import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen._src import warp_pipeline as wpl
from track_gen._src.types import TrackGenConfig

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fixed_corners(P: int, device) -> torch.Tensor:
    """Three envs of distinct jittered-polygon vertices, shape [3, P, 2]."""
    envs = []
    for e in range(3):
        ang = torch.arange(P, dtype=torch.float32) * (2.0 * math.pi / P)
        r = 1.0 + 0.1 * e
        phase = 0.37 * (e + 1)
        x = r * torch.cos(ang + phase) + 0.05 * torch.cos(3.0 * ang)
        y = r * torch.sin(ang + phase) + 0.05 * torch.sin(2.0 * ang)
        envs.append(torch.stack([x, y], dim=1))
    return torch.stack(envs, dim=0).to(device)


@pytest.mark.parametrize("dev", DEVS)
def test_closed_assemble_keeps_all_count_segments(dev):
    """Every one of the `count` segments is a real Bezier -> count*npseg finite samples."""
    config = TrackGenConfig()
    config.device = dev
    P = config.max_num_points
    npseg = config.num_points_per_segment

    corners = _fixed_corners(P, dev)
    # full (==P) and short counts, all distinct corners so no degenerate segments.
    count = torch.tensor([P, P - 3, P - 4], dtype=torch.long, device=corners.device)

    dense = wpl.assemble(corners, count, config)        # [E, P*npseg, 2]
    finite = torch.isfinite(dense).all(dim=-1)          # [E, P*npseg]

    for e in range(count.shape[0]):
        got = int(finite[e].sum())
        want = int(count[e]) * npseg
        assert got == want, (
            f"env {e}: {got} finite samples, expected count*npseg={want} "
            f"(closing segment must be a real Bezier, not dropped)")


def _fatband_cfg(E: int, dev: str) -> TrackGenConfig:
    """The fat-band regime from tests/test_warp_corner_ordering.py."""
    return TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                          output_mode="constant_spacing", spacing=0.30, N_max=384, device=dev)


def _attempt0_centerline(cfg, dev):
    """Production attempt-0 corners -> ccw_sort -> assemble -> resample to num_points."""
    E = cfg.num_envs
    seeds = torch.arange(E, dtype=torch.int32, device=dev)
    count = wpl.corner_count_sample(seeds, 0, cfg)
    corners = wpl.ccw_sort(wpl.corner_sample(seeds, 0, cfg), count)
    dense = wpl.assemble(corners, count, cfg)
    rs, _ = wpl.arc_length_resample_warp(dense, int(cfg.num_points))
    return rs


@pytest.mark.parametrize("dev", DEVS)
def test_adaptive_handle_clamp_drives_out_crossings(dev):
    """F2: a TIGHT adaptive per-corner handle clamp removes most residual Bezier-overshoot
    self-crossings -- single-attempt crossing-free ~96.4% (F1-alone) -> ~99.2% at frac=0.10.
    The production default is now frac=0.4 (== rad, for curvier tracks), so assemble ALONE is
    not ~always simple; the Fix B whole-track polygon fallback in generate_centerline_warp is
    what guarantees the FINAL centerline is simple. Pin both: the clamp MECHANISM at a tight
    frac (independent of the production default) and the Fix B end-to-end guarantee."""
    cfg = _fatband_cfg(2048, dev)
    cfg.handle_clamp_frac = 0.10  # tight clamp: exercise the overshoot-removal mechanism
    rs = _attempt0_centerline(cfg, dev)
    crossing_free = (wpl.self_intersections(rs) == 0).float().mean().item()
    assert crossing_free >= 0.985, (
        f"crossing-free rate {crossing_free:.4f} < 0.985 -- a tight handle clamp "
        f"should remove most Bezier overshoot (F1-alone is ~0.964)")

    # Fix B is the production guarantee: at the default clamp the assembled centerline still
    # self-crosses sometimes, but the whole-track polygon fallback drives the FINAL centerline
    # to ~always simple.
    seeds = torch.arange(cfg.num_envs, dtype=torch.int32, device=dev)
    centerline, _ = wpl.generate_centerline_warp(seeds, _fatband_cfg(2048, dev))
    final_simple = (wpl.self_intersections(centerline) == 0).float().mean().item()
    assert final_simple >= 0.999, (
        f"Fix B polygon fallback should make the final centerline ~always simple, "
        f"got {final_simple:.4f}")


@pytest.mark.parametrize("dev", DEVS)
def test_handle_clamp_preserves_shape_diversity(dev):
    """The clamp must not collapse shape diversity: at the production default frac (== rad)
    the per-track area spread stays within ~15% of the unclamped (frac=inf) spread. (At the
    old 0.10 default the clamp bound EVERY segment and did throttle diversity; the default is
    now 0.4 so the clamp only trims genuine overshoot corners.)"""
    cfg = _fatband_cfg(2048, dev)
    default_frac = cfg.handle_clamp_frac  # production default (== rad); read before mutation
    seeds = torch.arange(cfg.num_envs, dtype=torch.int32, device=dev)
    count = wpl.corner_count_sample(seeds, 0, cfg)
    corners = wpl.ccw_sort(wpl.corner_sample(seeds, 0, cfg), count)

    def area_std(frac):
        cfg.handle_clamp_frac = frac
        dense = wpl.assemble(corners, count, cfg)
        rs, _ = wpl.arc_length_resample_warp(dense, int(cfg.num_points))
        x, y = rs[..., 0], rs[..., 1]
        xn, yn = torch.roll(x, -1, 1), torch.roll(y, -1, 1)
        area = 0.5 * (x * yn - xn * y).nansum(1).abs()
        return area.std().item()

    clamped = area_std(default_frac)
    unclamped = area_std(1.0e9)
    assert clamped >= 0.85 * unclamped, (
        f"area spread collapsed under the clamp: {clamped:.2f} < 0.85*{unclamped:.2f}")
