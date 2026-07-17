#!/usr/bin/env python3
"""Render 3D gate courses for visual inspection.

Generates a batch of gate sequences through the pure-Warp gate pipeline
(``GateGenerator``), resamples each into a closed 3D centerline
(``CourseLine``), and for each valid env draws two panels:

- plan view: XY centerline plus per-gate left/right segments.
- elevation profile: arclength vs z of the centerline, with gate altitude
  markers.

Headless (Agg backend); PNG lands in ``viz/out/`` by default. Run directly:

    .venv/bin/python -m viz.plot_gate_courses                    # default grid
    .venv/bin/python -m viz.plot_gate_courses --envs 20 --seed 3
    .venv/bin/python -m viz.plot_gate_courses --out /tmp/gate_courses.png
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the flat "track_gen" package importable regardless of cwd (this file lives at
# <pkg_parent>/viz/plot_gate_courses.py, so the package parent is two levels up).
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
from track_gen._src.course_line import CourseLine
from track_gen._src.gate_generator import GateGenerator
from track_gen._src.types import GateGenConfig

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# scale=3.0: at the default scale=1.0, 0.2-wide gates overlap/cross and the
# finalizer rejects every env on cpu (see tests/test_course_gates_3d.py), making
# a plot of "the" default config vacuous. A larger loop keeps envs valid.
DEFAULT_SCALE = 3.0
SAMPLES_PER_GATE = 8


def make_rng(num_envs: int, seed: int, device: str) -> PerEnvSeededRNG:
    """Per-env seeded RNG, mirroring plot_tracks.py / the test fixtures."""
    wp.init()
    seeds = (torch.arange(num_envs, dtype=torch.int32) + seed).to(device)
    ids = torch.arange(num_envs, dtype=torch.int32).to(device)
    wp_seeds = wp.from_torch(seeds, dtype=wp.int32)
    wp_ids = wp.from_torch(ids, dtype=wp.int32)
    rng = PerEnvSeededRNG(seeds=wp_seeds, num_envs=num_envs, device=device)
    rng.set_seeds_warp(wp_seeds, ids=wp_ids)
    return rng


def draw_env(ax_plan, ax_elev, e: int, *, center, arclen, m: int,
             gate_left, gate_right, gate_pos, n_gates: int) -> None:
    """Draw env ``e``'s plan view (left) and elevation profile (right).

    ``center``/``arclen`` are already-sliced [n_max, 3] / [n_max] numpy arrays
    for this env; ``m`` is the valid sample count. ``gate_left``/``gate_right``/
    ``gate_pos`` are [max_gates, 3] numpy arrays; ``n_gates`` is the real gate
    count for this env.
    """
    cx, cy, cz = center[:m, 0], center[:m, 1], center[:m, 2]
    s = arclen[:m]

    # Plan view: closed centerline + per-gate left-right segments.
    ax_plan.plot(np.append(cx, cx[0]), np.append(cy, cy[0]),
                 color="0.25", lw=1.0, zorder=2)
    for i in range(n_gates):
        lx, ly = gate_left[i, 0], gate_left[i, 1]
        rx, ry = gate_right[i, 0], gate_right[i, 1]
        ax_plan.plot([lx, rx], [ly, ry], color="#1f77b4", lw=1.6, zorder=3)
    ax_plan.plot(gate_pos[:n_gates, 0], gate_pos[:n_gates, 1], "o",
                 color="#d62728", ms=2.5, zorder=4)
    ax_plan.set_aspect("equal")
    ax_plan.set_xlabel("x [m]", fontsize=6)
    ax_plan.set_ylabel("y [m]", fontsize=6)
    ax_plan.tick_params(labelsize=5)
    ax_plan.set_title(f"env {e}  plan  ({n_gates} gates)", fontsize=6.5)

    # Elevation profile: arclength vs z, with gate altitude markers.
    ax_elev.plot(s, cz, color="0.25", lw=1.0, zorder=2)
    gate_s = arclen[np.arange(n_gates) * SAMPLES_PER_GATE]
    gate_z = center[np.arange(n_gates) * SAMPLES_PER_GATE, 2]
    ax_elev.plot(gate_s, gate_z, "o", color="#d62728", ms=2.5, zorder=3)
    ax_elev.set_xlabel("arclength [m]", fontsize=6)
    ax_elev.set_ylabel("z [m]", fontsize=6)
    ax_elev.tick_params(labelsize=5)
    ax_elev.set_title(f"env {e}  elevation", fontsize=6.5)


def render(envs=12, seed=0, out=None, dpi=150, cell_in=2.4,
           z_profile="random_walk", z_base=1.5, z_min=0.5, z_max=2.5,
           z_max_step=0.4, gate_align="full_tangent", gate_width=0.2,
           scale=DEFAULT_SCALE, device="cpu"):
    """Generate ``envs`` gate courses on ``device`` and save a two-panel-per-env grid PNG."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if out is None:
        out = os.path.join(OUT_DIR, "gate_courses.png")

    config = GateGenConfig(device=device, num_envs=envs, z_profile=z_profile,
                           z_base=z_base, z_min=z_min, z_max=z_max,
                           z_max_step=z_max_step, gate_align=gate_align,
                           gate_width=gate_width, scale=scale)
    rng = make_rng(envs, seed=seed, device=device)
    gen = GateGenerator(config, rng)
    seq = gen.generate()

    line = CourseLine(seq, samples_per_gate=SAMPLES_PER_GATE)
    line.refresh()

    max_gates = seq.position.shape[0] // envs
    n_max = line.track.center.shape[0] // envs

    position_np = wp.to_torch(seq.position).view(envs, max_gates, 3).cpu().numpy()
    left_np = wp.to_torch(seq.left).view(envs, max_gates, 3).cpu().numpy()
    right_np = wp.to_torch(seq.right).view(envs, max_gates, 3).cpu().numpy()
    gate_count_np = wp.to_torch(seq.count).cpu().numpy()
    gate_valid_np = wp.to_torch(seq.valid).bool().cpu().numpy()

    center_np = wp.to_torch(line.track.center).view(envs, n_max, 3).cpu().numpy()
    arclen_np = wp.to_torch(line.track.arclen).view(envs, n_max).cpu().numpy()
    line_count_np = wp.to_torch(line.track.count).cpu().numpy()
    line_valid_np = wp.to_torch(line.track.valid).bool().cpu().numpy()

    valid_e = np.flatnonzero(gate_valid_np & line_valid_np)
    print(f"generated {envs} gate courses on {device} "
          f"(z_profile={z_profile}, gate_align={gate_align}, scale={scale}): "
          f"{valid_e.size}/{envs} valid")

    if valid_e.size == 0:
        raise RuntimeError(
            "no valid gate courses generated; try a different --seed or --envs")

    fig, axes = plt.subplots(valid_e.size, 2,
                              figsize=(2 * cell_in, valid_e.size * cell_in),
                              squeeze=False)
    for row, e in enumerate(valid_e):
        e = int(e)
        m = int(line_count_np[e])
        n_gates = int(gate_count_np[e])
        draw_env(axes[row, 0], axes[row, 1], e,
                 center=center_np[e], arclen=arclen_np[e], m=m,
                 gate_left=left_np[e], gate_right=right_np[e],
                 gate_pos=position_np[e], n_gates=n_gates)

    fig.suptitle(f"gate courses  seed={seed}  {valid_e.size}/{envs} valid", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output PNG path (default viz/out/gate_courses.png)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    render(envs=a.envs, seed=a.seed, out=a.out, dpi=a.dpi,
           device="cuda" if a.cuda else "cpu")
