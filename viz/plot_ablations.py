#!/usr/bin/env python3
"""Visual ablation harness for the GPU-batched race-track generator.

Renders three 4x4 figures that let a human eyeball how generation settings drive
track shape and self-intersection:

    fig1_bezier_rad_edgy.png   -- Bezier handle-length (rad) x edginess (edgy)
    fig3_bezier_vs_fourier.png -- the two generators side by side per seed
    fig4_min_angle_min_dist.png-- Bezier min_angle x min_point_distance

Everything is headless (Agg backend); PNGs land in ``viz/out/`` next to this file.
The pipeline is driven through the REAL public API exactly as the test-suite does:

    import warp as wp; wp.init()
    rng = PerEnvSeededRNG(seeds=<int32 tensor on device>, num_envs=E, device=device)
    rng.set_seeds(seeds, ids=<int32 tensor on device>)
    track = TrackGenerator(TrackGenConfig(...), rng).generate(E)

NOTE on devices: ``PerEnvSeededRNG`` keeps a tensor seed on whatever device the
tensor lives on, so for CUDA the seed/id tensors must be created on-device or the
underlying warp ``set_states`` kernel raises a device-mismatch error.
"""

from __future__ import annotations

import math
import os
import sys

# Make the flat "track_gen" package importable no matter the cwd: this file lives
# at <pkg_parent>/track_gen/viz/plot_ablations.py, so the parent of the package
# dir is three levels up.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")  # headless; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch

import warp as wp

from track_gen import PerEnvSeededRNG
from track_gen._src.types import Track, TrackGenConfig
from track_gen._src.track_generator import TrackGenerator

# --------------------------------------------------------------------------- #
# Paths / device
# --------------------------------------------------------------------------- #

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DPI = 130
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# RNG construction -- mirrors track_gen/tests/test_generators.py::_make_rng
# --------------------------------------------------------------------------- #


def make_rng(num_envs: int, seed: int = 1234, device: str = DEVICE) -> PerEnvSeededRNG:
    """Build a per-env seeded RNG exactly as the tests do.

    wp.init() is idempotent and required before any warp array/kernel use.
    Seeds are ``arange(num_envs) + seed`` so env e is reproducibly distinct, and
    the seed/id tensors are placed on ``device`` (warp's set_states kernel runs on
    the array's device, so CUDA needs on-device int32 tensors).
    """
    wp.init()
    seeds = (torch.arange(num_envs, dtype=torch.int32) + seed).to(device)
    ids = torch.arange(num_envs, dtype=torch.int32).to(device)
    wp_seeds = wp.from_torch(seeds, dtype=wp.int32)
    wp_ids = wp.from_torch(ids, dtype=wp.int32)
    rng = PerEnvSeededRNG(seeds=wp_seeds, num_envs=num_envs, device=device)
    rng.set_seeds_warp(wp_seeds, ids=wp_ids)
    return rng


def generate(config: TrackGenConfig, num_envs: int, seed: int = 1234) -> Track:
    """Run the full facade pipeline for one config and a fresh, reproducible RNG."""
    rng = make_rng(num_envs, seed=seed, device=config.device)
    return TrackGenerator(config, rng).generate(num_envs)


# --------------------------------------------------------------------------- #
# Small numpy helpers
# --------------------------------------------------------------------------- #


def _np_loop(arr2d: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Tensor [N, 2] -> closed (x, y) numpy arrays with NaN rows dropped.

    Padding slots (constant_spacing mode) and any degenerate NaN points are
    removed before we close the loop back to the first real point.
    """
    pts = arr2d.detach().cpu().numpy()
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return np.array([]), np.array([])
    pts = np.vstack([pts, pts[0]])  # close the loop
    return pts[:, 0], pts[:, 1]


# --------------------------------------------------------------------------- #
# The shared cell drawer
# --------------------------------------------------------------------------- #


def draw_track(ax, track: Track, env_index: int, title: str, invalid_flag: bool) -> None:
    """Plot one env's track into ``ax``.

    Renders the inner/outer borders as a filled band plus two solid lines and the
    centerline as a dashed line. When ``invalid_flag`` is True the title is tinted
    red so failed envs are obvious at a glance.
    """
    cx, cy = _np_loop(track.center[env_index])
    ox, oy = _np_loop(track.outer[env_index])
    ix, iy = _np_loop(track.inner[env_index])

    # Filled band between outer and inner (only when both have matching length).
    if ox.size and ix.size and ox.size == ix.size:
        band_x = np.concatenate([ox, ix[::-1]])
        band_y = np.concatenate([oy, iy[::-1]])
        ax.fill(band_x, band_y, color="0.80", zorder=1, linewidth=0)

    if ox.size:
        ax.plot(ox, oy, color="#1f77b4", lw=1.1, zorder=3)
    if ix.size:
        ax.plot(ix, iy, color="#d62728", lw=1.1, zorder=3)
    if cx.size:
        ax.plot(cx, cy, color="0.25", lw=0.8, ls="--", zorder=4)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    color = "red" if invalid_flag else "black"
    ax.set_title(title, fontsize=7.5, color=color, pad=2)


def _draw_polyline(ax, pts2d: torch.Tensor, **kw) -> None:
    """Plot a raw [N, 2] tensor as a closed polyline (NaN-dropped). For the naive row."""
    x, y = _np_loop(pts2d)
    if x.size:
        ax.plot(x, y, **kw)


def _style_cell(ax, title: str, invalid_flag: bool = False) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.5, color=("red" if invalid_flag else "black"), pad=2)


def _is_invalid(track: Track, env_index: int) -> bool:
    return not bool(track.valid[env_index].item())


# --------------------------------------------------------------------------- #
# Figure 1 -- Bezier rad x edgy (pure ablation: same seed -> same control points)
# --------------------------------------------------------------------------- #


def figure1_rad_edgy(seed: int = 7) -> str:
    """rad rows x edgy cols on a FIXED seed.

    rad and edgy do not touch point sampling (the grid draws / count draws are
    seed-only), so a fixed seed yields identical control corners across the whole
    grid; the only thing that moves is the Bezier handle length (rad) and the
    tangent-blend edginess (edgy) -> a clean view of overshoot -> self-intersection.
    """
    rads = [0.1, 0.2, 0.3, 0.45]
    edgys = [-0.3, 0.0, 0.3, 0.6]
    n = len(rads)

    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for r, rad in enumerate(rads):
        for c, edgy in enumerate(edgys):
            cfg = TrackGenConfig(
                generator="bezier",
                device=DEVICE,
                num_envs=1,
                num_points=256,
                rad=rad,
                edgy=edgy,
            )
            track = generate(cfg, 1, seed=seed)
            ax = axes[r, c]
            draw_track(
                ax,
                track,
                0,
                f"rad={rad}, edgy={edgy}",
                _is_invalid(track, 0),
            )

    fig.suptitle(
        "Figure 1 -- Bezier rad (rows) x edgy (cols), FIXED seed\n"
        "Same seed => same control corners; only handle-length (rad) and "
        "edginess (edgy) change, driving overshoot into self-intersection",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT_DIR, "fig1_bezier_rad_edgy.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 3 -- Bezier vs Fourier, one seed per row
# --------------------------------------------------------------------------- #


def figure3_bezier_vs_fourier() -> str:
    """4x4: left 2 cols Bezier, right 2 cols Fourier; each row a distinct seed.

    Same seed across the row so the two generators are compared on matched RNG
    streams; titles flag which envs come out valid.
    """
    seeds = [1, 2, 3, 4]
    n = 4

    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for r, s in enumerate(seeds):
        bez = generate(
            TrackGenConfig(generator="bezier", device=DEVICE, num_envs=2, num_points=256),
            2,
            seed=s,
        )
        fou = generate(
            TrackGenConfig(
                generator="fourier",
                device=DEVICE,
                num_envs=2,
                num_points=256,
                num_harmonics=5,
                decay_p=2,
                num_centerline_samples=256,
            ),
            2,
            seed=s + 1000,  # decouple stream from the bezier column on the same row
        )

        for c in range(2):  # cols 0,1 -> Bezier envs 0,1
            draw_track(
                axes[r, c],
                bez,
                c,
                f"Bezier  seed {s}.{c}\nvalid={bool(bez.valid[c])}",
                _is_invalid(bez, c),
            )
        for c in range(2):  # cols 2,3 -> Fourier envs 0,1
            draw_track(
                axes[r, c + 2],
                fou,
                c,
                f"Fourier  seed {s}.{c}\nvalid={bool(fou.valid[c])}",
                _is_invalid(fou, c),
            )

    fig.suptitle(
        "Figure 3 -- Bezier (left 2 cols) vs Fourier (right 2 cols), one seed per row\n"
        "Bezier: piecewise-cubic corners; Fourier: smooth-by-construction; red title = invalid",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT_DIR, "fig3_bezier_vs_fourier.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 4 -- Bezier min_angle x min_point_distance (sampling-changing settings)
# --------------------------------------------------------------------------- #


def figure4_min_angle_min_dist(seed: int = 1234) -> str:
    """min_angle rows (degrees) x min_point_distance cols.

    Both settings change the point sampling (min_point_distance resizes the corner
    grid; min_angle changes which draws survive regeneration), so a fixed seed is
    only REPRESENTATIVE here, not a pure ablation -- noted in the suptitle. Invalid
    cells are tinted red.
    """
    angles_deg = [5.0, 12.5, 20.0, 30.0]
    min_dists = [0.03, 0.05, 0.08, 0.12]
    n = 4

    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for r, ad in enumerate(angles_deg):
        for c, md in enumerate(min_dists):
            cfg = TrackGenConfig(
                generator="bezier",
                device=DEVICE,
                num_envs=1,
                num_points=256,
                min_angle=(ad / 180.0) * math.pi,
                min_point_distance=md,
            )
            track = generate(cfg, 1, seed=seed)
            draw_track(
                axes[r, c],
                track,
                0,
                f"min_angle={ad}deg\nmin_dist={md}  valid={bool(track.valid[0])}",
                _is_invalid(track, 0),
            )

    fig.suptitle(
        "Figure 4 -- Bezier min_angle (rows) x min_point_distance (cols)\n"
        "These settings CHANGE the sampling, so the fixed seed is representative, "
        "not a pure ablation; red title = invalid env",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT_DIR, "fig4_min_angle_min_dist.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    wp.init()  # warp must be initialised before any RNG/kernel use

    paths = [
        figure1_rad_edgy(),
        figure3_bezier_vs_fourier(),
        figure4_min_angle_min_dist(),
    ]
    print(f"device={DEVICE}")
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
