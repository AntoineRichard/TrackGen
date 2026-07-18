#!/usr/bin/env python3
"""Render 2.5D tracks (plan view + elevation + heightfield) for visual inspection.

Generates a batch of tracks through the pure-Warp pipeline (``TrackGenerator``)
with a non-flat ``z_profile``, bakes a per-env road heightfield
(``HeightFieldBaker``), and for each valid env draws three panels:

- plan view: outer/center/inner XY borders (band fill + centerline).
- elevation profile: arclength vs centerline z.
- heightfield: ``imshow`` of the baked per-env height grid (extent from
  ``lo``/``hi``, ``origin="lower"``), with the plan-view centerline overlaid so
  the ridge can be checked against the track by eye.

Headless (Agg backend); PNG lands in ``viz/out/`` by default. Run directly:

    .venv/bin/python -m viz.plot_tracks_3d                          # default grid
    .venv/bin/python -m viz.plot_tracks_3d --envs 3 --seed 5
    .venv/bin/python -m viz.plot_tracks_3d --out docs/_static/tracks-25d.png --envs 3
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the flat "track_gen" package importable regardless of cwd (this file lives at
# <pkg_parent>/viz/plot_tracks_3d.py, so the package parent is two levels up).
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch

import warp as wp

from track_gen import PerEnvSeededRNG
from track_gen.heightfield import HeightFieldBaker
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def make_rng(num_envs: int, seed: int, device: str) -> PerEnvSeededRNG:
    """Per-env seeded RNG, mirroring plot_tracks.py / plot_gate_courses.py."""
    wp.init()
    seeds = (torch.arange(num_envs, dtype=torch.int32) + seed).to(device)
    ids = torch.arange(num_envs, dtype=torch.int32).to(device)
    wp_seeds = wp.from_torch(seeds, dtype=wp.int32)
    wp_ids = wp.from_torch(ids, dtype=wp.int32)
    rng = PerEnvSeededRNG(seeds=wp_seeds, num_envs=num_envs, device=device)
    rng.set_seeds_warp(wp_seeds, ids=wp_ids)
    return rng


def draw_env(ax_plan, ax_elev, ax_hf, e: int, *, outer, center, inner, arclen,
             m: int, hf_grid, hf_lo, hf_hi) -> None:
    """Draw env ``e``'s plan view, elevation profile, and heightfield.

    ``outer``/``center``/``inner``/``arclen`` are already-sliced ``[m, 3]`` /
    ``[m]`` numpy arrays for this env (real points only); ``hf_grid`` is the
    env's ``[res, res]`` height grid; ``hf_lo``/``hf_hi`` are its ``[2]`` world
    AABB corners.
    """
    ox, oy = np.append(outer[:, 0], outer[0, 0]), np.append(outer[:, 1], outer[0, 1])
    cx, cy, cz = center[:, 0], center[:, 1], center[:, 2]
    cxl, cyl = np.append(cx, cx[0]), np.append(cy, cy[0])
    ix, iy = np.append(inner[:, 0], inner[0, 0]), np.append(inner[:, 1], inner[0, 1])

    # Plan view: filled band + outer/inner borders + dashed centerline.
    ax_plan.fill(np.concatenate([ox, ix[::-1]]), np.concatenate([oy, iy[::-1]]),
                 color="0.80", zorder=1, linewidth=0)
    ax_plan.plot(ox, oy, color="#1f77b4", lw=0.9, zorder=3)
    ax_plan.plot(ix, iy, color="#d62728", lw=0.9, zorder=3)
    ax_plan.plot(cxl, cyl, color="0.25", lw=0.7, ls="--", zorder=4)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlabel("x [m]", fontsize=6)
    ax_plan.set_ylabel("y [m]", fontsize=6)
    ax_plan.tick_params(labelsize=5)
    ax_plan.set_title(f"env {e}  plan", fontsize=6.5)

    # Elevation profile: arclength vs centerline z.
    s = np.append(arclen, arclen[-1] + np.linalg.norm(center[0] - center[-1]))
    ax_elev.plot(s, np.append(cz, cz[0]), color="0.25", lw=1.0, zorder=2)
    ax_elev.set_xlabel("arclength [m]", fontsize=6)
    ax_elev.set_ylabel("z [m]", fontsize=6)
    ax_elev.tick_params(labelsize=5)
    ax_elev.set_title(f"env {e}  elevation", fontsize=6.5)

    # Heightfield: baked grid with the world-mapped extent, centerline overlaid
    # so the ridge can be checked against the plan view by eye.
    extent = (hf_lo[0], hf_hi[0], hf_lo[1], hf_hi[1])
    im = ax_hf.imshow(hf_grid, origin="lower", extent=extent, cmap="terrain",
                      aspect="equal")
    ax_hf.plot(cxl, cyl, color="black", lw=0.8, ls="--", zorder=3, alpha=0.6)
    ax_hf.set_xlabel("x [m]", fontsize=6)
    ax_hf.set_ylabel("y [m]", fontsize=6)
    ax_hf.tick_params(labelsize=5)
    ax_hf.set_title(f"env {e}  heightfield", fontsize=6.5)
    cb = plt.colorbar(im, ax=ax_hf, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)


def render(envs=12, seed=0, out=None, dpi=150, cell_in=2.4, resolution=96,
           z_profile="random_walk", z_base=1.0, z_min=0.2, z_max=2.0,
           z_max_step=0.3, device="cpu"):
    """Generate ``envs`` 2.5D tracks on ``device`` and save a three-panel-per-env grid PNG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if out is None:
        out = os.path.join(OUT_DIR, "tracks_3d.png")

    config = TrackGenConfig(device=device, num_envs=envs, z_profile=z_profile,
                            z_base=z_base, z_min=z_min, z_max=z_max,
                            z_max_step=z_max_step)
    rng = make_rng(envs, seed=seed, device=device)
    track = TrackGenerator(config, rng).generate()

    hf = HeightFieldBaker(track, resolution).bake()

    n_max = track.center.shape[0] // envs
    outer_np = wp.to_torch(track.outer).view(envs, n_max, 3).cpu().numpy()
    center_np = wp.to_torch(track.center).view(envs, n_max, 3).cpu().numpy()
    inner_np = wp.to_torch(track.inner).view(envs, n_max, 3).cpu().numpy()
    arclen_np = wp.to_torch(track.arclen).view(envs, n_max).cpu().numpy()
    count_np = wp.to_torch(track.count).cpu().numpy()
    valid_np = wp.to_torch(track.valid).bool().cpu().numpy()

    grid_np = wp.to_torch(hf.height).view(envs, resolution, resolution).cpu().numpy()
    lo_np = wp.to_torch(hf.lo).view(envs, 2).cpu().numpy()
    hi_np = wp.to_torch(hf.hi).view(envs, 2).cpu().numpy()

    valid_e = np.flatnonzero(valid_np)
    print(f"generated {envs} tracks on {device} "
          f"(z_profile={z_profile}, z_base={z_base}, resolution={resolution}): "
          f"{valid_e.size}/{envs} valid")

    if valid_e.size == 0:
        raise RuntimeError(
            "no valid tracks generated; try a different --seed or --envs")

    fig, axes = plt.subplots(valid_e.size, 3,
                              figsize=(3 * cell_in, valid_e.size * cell_in),
                              squeeze=False)
    for row, e in enumerate(valid_e):
        e = int(e)
        m = int(count_np[e])
        draw_env(axes[row, 0], axes[row, 1], axes[row, 2], e,
                 outer=outer_np[e, :m], center=center_np[e, :m],
                 inner=inner_np[e, :m], arclen=arclen_np[e, :m], m=m,
                 hf_grid=grid_np[e], hf_lo=lo_np[e], hf_hi=hi_np[e])

    fig.suptitle(f"2.5D tracks  z_profile={z_profile}  seed={seed}  "
                 f"{valid_e.size}/{envs} valid", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output PNG path (default viz/out/tracks_3d.png)")
    ap.add_argument("--resolution", type=int, default=96, help="heightfield grid resolution")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    render(envs=a.envs, seed=a.seed, out=a.out, dpi=a.dpi, resolution=a.resolution,
           device="cuda" if a.cuda else "cpu")
