#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render high-resolution track grids for visual inspection.

Generates a batch of tracks through the REAL public facade (the pure-Warp pipeline)
and lays them out as ``--images`` PNGs, each an ``--rows`` x ``--cols`` grid of tracks
(default 10 images of 9x9 = 81 tracks each => 810 tracks). Invalid tracks (failed the
validity gate) get a red-tinted title so failures are obvious at a glance.

Headless (Agg backend); PNGs land in ``viz/out/``. Run directly:

    .venv/bin/python -m viz.plot_tracks                      # 10 x 9x9, auto device
    .venv/bin/python -m viz.plot_tracks --half_width 0.04 --dpi 200
    .venv/bin/python -m viz.plot_tracks --images 4 --rows 6 --cols 6 --cpu
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the flat "track_gen" package importable regardless of cwd (this file lives at
# <pkg_parent>/track_gen/viz/plot_tracks.py, so the package parent is three levels up).
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch

import warp as wp

from track_gen import PerEnvSeededRNG
from track_gen.types import Track, TrackGenConfig
from track_gen.track_generator import TrackGenerator

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def make_rng(num_envs: int, seed: int, device: str) -> PerEnvSeededRNG:
    """Per-env seeded RNG, mirroring the tests / plot_ablations.

    wp.init() is idempotent and required before any Warp array/kernel use. Seeds are
    ``arange(num_envs) + seed`` so each env is reproducibly distinct; the seed/id tensors
    are placed on ``device`` (Warp's set_states kernel runs on the array's device, so a
    CUDA run needs on-device int32 tensors).
    """
    wp.init()
    seeds = (torch.arange(num_envs, dtype=torch.int32) + seed).to(device)
    ids = torch.arange(num_envs, dtype=torch.int32).to(device)
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=device)
    rng.set_seeds(seeds, ids=ids)
    return rng


def _np_loop(arr2d: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Tensor [N, 2] -> closed (x, y) numpy arrays with NaN rows dropped."""
    pts = arr2d.detach().cpu().numpy()
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.array([]), np.array([])
    pts = np.vstack([pts, pts[0]])  # close the loop
    return pts[:, 0], pts[:, 1]


def draw_track(ax, track: Track, e: int) -> None:
    """Plot env ``e``'s track into ``ax``: filled band + outer/inner borders + dashed
    centerline. Invalid tracks get a red title."""
    cx, cy = _np_loop(track.center[e])
    ox, oy = _np_loop(track.outer[e])
    ix, iy = _np_loop(track.inner[e])

    if ox.size and ix.size and ox.size == ix.size:
        ax.fill(np.concatenate([ox, ix[::-1]]), np.concatenate([oy, iy[::-1]]),
                color="0.80", zorder=1, linewidth=0)
    if ox.size:
        ax.plot(ox, oy, color="#1f77b4", lw=0.9, zorder=3)
    if ix.size:
        ax.plot(ix, iy, color="#d62728", lw=0.9, zorder=3)
    if cx.size:
        ax.plot(cx, cy, color="0.25", lw=0.6, ls="--", zorder=4)

    invalid = not bool(track.valid[e].item())
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"env {e}{'  INVALID' if invalid else ''}",
                 fontsize=6, color=("red" if invalid else "black"), pad=1.5)


def render(images=10, rows=9, cols=9, half_width=0.05, num_points=256, scale=1.0,
           device="cuda", seed=0, dpi=150, cell_in=1.8):
    """Generate images*rows*cols tracks and save ``images`` grid PNGs to viz/out/."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    per_image = rows * cols
    E = images * per_image
    config = TrackGenConfig(num_envs=E, num_points=num_points, half_width=half_width,
                            scale=scale, device=device)
    rng = make_rng(E, seed=seed, device=device)
    track = TrackGenerator(config, rng).generate(E)

    valid = track.valid
    print(f"generated {E} tracks on {device} (hw={half_width}); "
          f"overall valid yield: {valid.float().mean().item():.3f}")

    paths = []
    for img in range(images):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_in, rows * cell_in))
        base = img * per_image
        for k, ax in enumerate(axes.flat):
            draw_track(ax, track, base + k)
        n_valid = int(valid[base:base + per_image].sum().item())
        fig.suptitle(f"tracks {base}-{base + per_image - 1}   "
                     f"({n_valid}/{per_image} valid)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = os.path.join(OUT_DIR, f"tracks_grid_{img:02d}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
        print(f"  wrote {path}  ({n_valid}/{per_image} valid)")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=10)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--half_width", type=float, default=0.05)
    ap.add_argument("--num_points", type=int, default=256)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    render(images=a.images, rows=a.rows, cols=a.cols, half_width=a.half_width,
           num_points=a.num_points, scale=a.scale, seed=a.seed, dpi=a.dpi,
           device="cpu" if a.cpu else "cuda")
