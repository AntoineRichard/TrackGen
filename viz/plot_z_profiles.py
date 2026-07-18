#!/usr/bin/env python3
"""Compare the four ``z_profile`` altitude models on the SAME track layout.

Generates one track batch per profile (``flat``, ``uniform``, ``random_walk``,
``noise``) with an identical seed, identical Z knobs, and identical plan-view
layout (the XY relaxation runs before elevation exists, so it is bit-identical
across profiles — see :doc:`/tracks-25d`'s "Why 2.5D" section). Draws a
2-column grid, one row per profile:

- plan view: outer/inner borders for context, centerline colored by z.
- elevation profile: arclength vs. centerline z.

Headless (Agg backend); PNG lands in ``viz/out/`` by default. Run directly:

    .venv/bin/python -m viz.plot_z_profiles
    .venv/bin/python -m viz.plot_z_profiles --envs 3 --seed 5
    .venv/bin/python -m viz.plot_z_profiles --out docs/_static/z-profiles.png
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the flat "track_gen" package importable regardless of cwd (this file lives at
# <pkg_parent>/viz/plot_z_profiles.py, so the package parent is two levels up).
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
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Common Z knobs shared by every profile row; each profile only reads the
# subset of these it actually uses (see docs/tracks-25d.rst's knobs table).
Z_BASE = 1.0
Z_MIN = 0.5
Z_MAX = 1.5
Z_MAX_STEP = 0.3
Z_NOISE_AMPLITUDE = 0.4
Z_CONTROL_POINTS = 10

PROFILES = [
    ("flat", "z_base=1.0"),
    ("uniform", "z_min=0.5, z_max=1.5"),
    ("random_walk", "z_max_step=0.3"),
    ("noise", "z_noise_amplitude=0.4"),
]


def make_rng(num_envs: int, seed: int, device: str) -> PerEnvSeededRNG:
    """Per-env seeded RNG, mirroring plot_tracks_3d.py."""
    wp.init()
    seeds = (torch.arange(num_envs, dtype=torch.int32) + seed).to(device)
    ids = torch.arange(num_envs, dtype=torch.int32).to(device)
    wp_seeds = wp.from_torch(seeds, dtype=wp.int32)
    wp_ids = wp.from_torch(ids, dtype=wp.int32)
    rng = PerEnvSeededRNG(seeds=wp_seeds, num_envs=num_envs, device=device)
    rng.set_seeds_warp(wp_seeds, ids=wp_ids)
    return rng


def draw_row(ax_plan, ax_elev, *, profile: str, knobs: str, outer, center,
             inner, arclen, vmin: float, vmax: float) -> None:
    """Draw one profile's plan view (colored by z) and elevation profile.

    ``outer``/``center``/``inner``/``arclen`` are already-sliced ``[m, 3]`` /
    ``[m]`` numpy arrays for this env (real points only).
    """
    ox, oy = np.append(outer[:, 0], outer[0, 0]), np.append(outer[:, 1], outer[0, 1])
    cx, cy, cz = center[:, 0], center[:, 1], center[:, 2]
    cxl, cyl, czl = np.append(cx, cx[0]), np.append(cy, cy[0]), np.append(cz, cz[0])
    ix, iy = np.append(inner[:, 0], inner[0, 0]), np.append(inner[:, 1], inner[0, 1])

    # Plan view: outer/inner borders for context, centerline colored by z.
    ax_plan.plot(ox, oy, color="0.75", lw=0.7, zorder=1)
    ax_plan.plot(ix, iy, color="0.75", lw=0.7, zorder=1)
    sc = ax_plan.scatter(cxl, cyl, c=czl, cmap="terrain", vmin=vmin, vmax=vmax,
                         s=5, zorder=3)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlabel("x [m]", fontsize=6)
    ax_plan.set_ylabel("y [m]", fontsize=6)
    ax_plan.tick_params(labelsize=5)
    ax_plan.set_title(f"{profile}  ({knobs})  plan", fontsize=6.5)
    cb = plt.colorbar(sc, ax=ax_plan, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("z [m]", fontsize=6)

    # Elevation profile: arclength vs centerline z.
    s = np.append(arclen, arclen[-1] + np.linalg.norm(center[0] - center[-1]))
    ax_elev.plot(s, czl, color="0.25", lw=1.0, zorder=2)
    ax_elev.set_ylim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
    ax_elev.set_xlabel("arclength [m]", fontsize=6)
    ax_elev.set_ylabel("z [m]", fontsize=6)
    ax_elev.tick_params(labelsize=5)
    ax_elev.set_title(f"{profile}  ({knobs})  elevation", fontsize=6.5)


def render(envs=6, seed=0, out=None, dpi=150, cell_in=2.6, device="cpu"):
    """Generate one track batch per profile on ``device`` and save the grid PNG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if out is None:
        out = os.path.join(OUT_DIR, "z_profiles.png")

    fig, axes = plt.subplots(len(PROFILES), 2,
                             figsize=(2 * cell_in, len(PROFILES) * cell_in),
                             squeeze=False)

    first_env = None
    for row, (profile, knobs) in enumerate(PROFILES):
        config = TrackGenConfig(device=device, num_envs=envs, z_profile=profile,
                                z_base=Z_BASE, z_min=Z_MIN, z_max=Z_MAX,
                                z_max_step=Z_MAX_STEP,
                                z_noise_amplitude=Z_NOISE_AMPLITUDE,
                                z_control_points=Z_CONTROL_POINTS)
        rng = make_rng(envs, seed=seed, device=device)
        track = TrackGenerator(config, rng).generate()

        valid_np = wp.to_torch(track.valid).bool().cpu().numpy()
        valid_e = np.flatnonzero(valid_np)
        if valid_e.size == 0:
            raise RuntimeError(
                f"no valid tracks generated for profile={profile!r}; "
                "try a different --seed or --envs")
        # Same seed + envs => XPBD layout (and validity) is identical across
        # profiles (elevation is applied strictly after relaxation), so every
        # row picks the SAME env index for a genuine like-for-like comparison.
        e = int(valid_e[0]) if first_env is None else first_env
        first_env = e

        n_max = track.center.shape[0] // envs
        m = int(wp.to_torch(track.count).cpu().numpy()[e])
        outer_np = wp.to_torch(track.outer).view(envs, n_max, 3).cpu().numpy()[e, :m]
        center_np = wp.to_torch(track.center).view(envs, n_max, 3).cpu().numpy()[e, :m]
        inner_np = wp.to_torch(track.inner).view(envs, n_max, 3).cpu().numpy()[e, :m]
        arclen_np = wp.to_torch(track.arclen).view(envs, n_max).cpu().numpy()[e, :m]

        draw_row(axes[row, 0], axes[row, 1], profile=profile, knobs=knobs,
                 outer=outer_np, center=center_np, inner=inner_np,
                 arclen=arclen_np, vmin=Z_MIN, vmax=Z_MAX)

    fig.suptitle(f"Altitude profiles  env={first_env}  seed={seed}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=6,
                    help="candidate envs to generate per profile (first valid one is plotted)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output PNG path (default viz/out/z_profiles.png)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    render(envs=a.envs, seed=a.seed, out=a.out, dpi=a.dpi,
           device="cuda" if a.cuda else "cpu")
