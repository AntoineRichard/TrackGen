"""Render deterministic PNG assets used by the README.

The images are generated from the real TrackGen runtime pipeline on the Warp CPU device,
then written under ``docs/assets`` so they can be committed and displayed by GitHub.

Run from the repository root:

    .venv/bin/python -m viz.render_readme_assets
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator  # noqa: E402

OUT_DIR = Path("docs/assets")
GENERATORS = [
    ("bezier", "Bezier"),
    ("checkpoint", "Checkpoint"),
    ("hull", "Hull"),
    ("polar", "Polar"),
    ("voronoi", "Voronoi"),
]


def _generate(name: str, seed: int, needed: int = 5):
    batch = 24
    cfg = TrackGenConfig(
        generator=name,
        num_envs=batch,
        device="cpu",
        half_width=0.08,
        scale=5.0,
        spacing=0.12,
        N_max=384,
        relax_iters=80,
    )
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    center = wp.to_torch(track.center).cpu().numpy().reshape(batch, cfg.N_max, 2)
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, cfg.N_max, 2)
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, cfg.N_max, 2)
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(track.count).cpu().numpy().astype(int)

    chosen: list[int] = []
    for e in range(batch):
        n = int(count[e])
        if valid[e] and n >= 4 and np.isfinite(center[e, :n]).all():
            chosen.append(e)
        if len(chosen) >= needed:
            break
    if len(chosen) < needed:
        for e in range(batch):
            if e not in chosen and int(count[e]) >= 4:
                chosen.append(e)
            if len(chosen) >= needed:
                break
    return center, outer, inner, valid, count, chosen[:needed]


def _finite_rows(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=1)]


def _draw_track(ax, center, outer, inner, valid, count, e: int, *, lw: float, center_lw: float) -> None:
    n = int(count[e])
    c = _finite_rows(center[e, :n])
    o = _finite_rows(outer[e, :n])
    inn = _finite_rows(inner[e, :n])

    if len(o) >= 3 and len(inn) >= 3:
        band = np.vstack([o, inn[::-1]])
        ax.fill(band[:, 0], band[:, 1], color="#2f343b", alpha=0.96, linewidth=0, zorder=1)
        closed_o = np.vstack([o, o[0]])
        closed_i = np.vstack([inn, inn[0]])
        ax.plot(closed_o[:, 0], closed_o[:, 1], color="#111827", lw=lw, zorder=2)
        ax.plot(closed_i[:, 0], closed_i[:, 1], color="#111827", lw=lw, zorder=2)
    if len(c) >= 3:
        closed_c = np.vstack([c, c[0]])
        ax.plot(
            closed_c[:, 0],
            closed_c[:, 1],
            color="#f8fafc",
            lw=center_lw,
            ls=(0, (5, 4)),
            zorder=3,
        )
        xmin, ymin = c.min(axis=0)
        xmax, ymax = c.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        pad = max(dx, dy) * 0.22 + 0.1
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
    if not bool(valid[e]):
        ax.text(
            0.5,
            0.5,
            "invalid",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#dc2626",
            fontsize=8,
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _load_samples() -> dict[str, tuple]:
    wp.init()
    samples = {}
    for idx, (name, _label) in enumerate(GENERATORS):
        samples[name] = _generate(name, seed=100 + 17 * idx, needed=5)
    return samples


def render_readme_assets(output_dir: Path = OUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples()

    fig, axes = plt.subplots(
        len(GENERATORS), 5, figsize=(10.8, 8.6), dpi=170, facecolor="white"
    )
    for row, (name, label) in enumerate(GENERATORS):
        center, outer, inner, valid, count, chosen = samples[name]
        for col, env_id in enumerate(chosen):
            ax = axes[row, col]
            _draw_track(ax, center, outer, inner, valid, count, env_id, lw=0.75, center_lw=0.5)
            if col == 0:
                ax.set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=18,
                    fontsize=11,
                    fontweight="bold",
                    color="#111827",
                )
    fig.suptitle(
        "TrackGen standard first-stage generators",
        fontsize=16,
        fontweight="bold",
        y=0.985,
        color="#111827",
    )
    fig.tight_layout(rect=(0.055, 0.0, 1.0, 0.965), h_pad=0.45, w_pad=0.20)
    grid_path = output_dir / "readme-generator-grid.png"
    fig.savefig(grid_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(GENERATORS), figsize=(12.0, 2.75), dpi=180, facecolor="white")
    for ax, (name, label) in zip(axes, GENERATORS):
        center, outer, inner, valid, count, chosen = samples[name]
        _draw_track(ax, center, outer, inner, valid, count, chosen[0], lw=1.15, center_lw=0.8)
        ax.set_title(label, fontsize=12, fontweight="bold", color="#111827", pad=8)
    fig.tight_layout(w_pad=0.45)
    strip_path = output_dir / "readme-generator-strip.png"
    fig.savefig(strip_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return [grid_path, strip_path]


def main() -> None:
    for path in render_readme_assets():
        print(path)


if __name__ == "__main__":
    main()
