"""Render deterministic PNG assets used by the README.

The images are generated from the real TrackGen runtime pipeline with fixed seeds and
default geometry parameters, then written under ``docs/assets`` so they can be committed
and displayed by GitHub.

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
README_PIPELINE_HALF_WIDTH = 0.05

GENERATORS = [
    ("bezier", "Bezier"),
    ("checkpoint", "Checkpoint"),
    ("hull", "Hull"),
    ("polar", "Polar"),
    ("voronoi", "Voronoi"),
]


def _choose_phase1_examples(
    phase1: np.ndarray,
    phase1_valid: np.ndarray,
    final_valid: np.ndarray,
    needed: int,
) -> list[int]:
    chosen: list[int] = []
    for e in range(phase1.shape[0]):
        if phase1_valid[e] and final_valid[e] and np.isfinite(phase1[e]).all():
            chosen.append(e)
        if len(chosen) >= needed:
            return chosen
    for e in range(phase1.shape[0]):
        if e not in chosen and phase1_valid[e] and np.isfinite(phase1[e]).all():
            chosen.append(e)
        if len(chosen) >= needed:
            return chosen
    return chosen


def _generate(name: str, seed: int, needed: int = 5):
    batch = 24
    cfg = TrackGenConfig(generator=name, num_envs=batch, device="cpu")
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    generator = TrackGenerator(cfg, rng)
    track = generator.generate()
    scratch = generator._scratch
    phase1 = wp.to_torch(scratch.gen_centerline).cpu().numpy().reshape(batch, cfg.num_points, 2)
    phase1_valid = wp.to_torch(scratch.gen_valid).cpu().numpy().astype(bool)
    final_valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    chosen = _choose_phase1_examples(phase1, phase1_valid, final_valid, needed)
    return phase1, phase1_valid, chosen


def _generate_pipeline_sample(name: str = "bezier", seed: int = 100):
    batch = 24
    cfg = TrackGenConfig(
        generator=name,
        num_envs=batch,
        device="cpu",
        half_width=README_PIPELINE_HALF_WIDTH,
    )
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    generator = TrackGenerator(cfg, rng)
    track = generator.generate()
    scratch = generator._scratch

    valid_arr = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    count_arr = wp.to_torch(track.count).cpu().numpy().astype(int)
    final_all = wp.to_torch(track.center).cpu().numpy().reshape(batch, cfg.N_max, 2)
    env_id = 0
    for e in range(batch):
        n = int(count_arr[e])
        if valid_arr[e] and n >= 4 and np.isfinite(final_all[e, :n]).all():
            env_id = e
            break

    count = int(count_arr[env_id])
    raw = wp.to_torch(scratch.gen_centerline).cpu().numpy().reshape(batch, cfg.num_points, 2)[env_id]
    cs = wp.to_torch(scratch.cs_center).cpu().numpy().reshape(batch, cfg.N_max, 2)[env_id, :count]
    relaxed = wp.to_torch(scratch.relax.relaxed).cpu().numpy().reshape(batch, cfg.N_max, 2)[env_id, :count]
    final_center = final_all[env_id, :count]
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, cfg.N_max, 2)[env_id, :count]
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, cfg.N_max, 2)[env_id, :count]
    return raw, cs, relaxed, final_center, outer, inner, bool(valid_arr[env_id]), float(cfg.half_width)


def _finite_rows(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=1)]


def _set_track_limits(ax, points: np.ndarray, pad_frac: float = 0.22) -> None:
    pts = _finite_rows(points)
    if len(pts) < 3:
        return
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    pad = max(dx, dy) * pad_frac + 0.1
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)


def _style_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_phase1_output(ax, phase1: np.ndarray, valid: np.ndarray, e: int, *, lw: float) -> None:
    pts = _finite_rows(phase1[e])
    if len(pts) >= 3:
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#2563eb", lw=lw, zorder=2)
        _set_track_limits(ax, pts)
    if not bool(valid[e]):
        ax.text(0.5, 0.5, "invalid", transform=ax.transAxes, ha="center", va="center", color="#dc2626")
    _style_axis(ax)


def _draw_centerline_stage(ax, points: np.ndarray, *, color: str, title: str, dots: bool = False) -> None:
    pts = _finite_rows(points)
    if len(pts) >= 3:
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.45, zorder=2)
        if dots:
            every = max(1, len(pts) // 42)
            ax.scatter(pts[::every, 0], pts[::every, 1], s=7, color="#111827", alpha=0.85, zorder=3)
        _set_track_limits(ax, pts)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#111827", pad=8)
    _style_axis(ax)


def _offset_polyline(points: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    tangent = next_pts - prev_pts
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 1e-8)
    return points + half_width * normal, points - half_width * normal


def _draw_relaxed_stage(ax, points: np.ndarray, *, relax_half_width: float) -> None:
    pts = _finite_rows(points)
    if len(pts) >= 3:
        outer, inner = _offset_polyline(pts, relax_half_width)
        band = np.vstack([outer, inner[::-1]])
        ax.fill(band[:, 0], band[:, 1], color="#ede9fe", alpha=0.92, linewidth=0, zorder=1)
        for edge in (outer, inner):
            closed_edge = np.vstack([edge, edge[0]])
            ax.plot(closed_edge[:, 0], closed_edge[:, 1], color="#7c3aed", lw=0.85, zorder=2)
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#4c1d95", lw=1.0, zorder=3)
        every = max(1, len(pts) // 42)
        ax.scatter(pts[::every, 0], pts[::every, 1], s=6, color="#111827", alpha=0.78, zorder=4)
        _set_track_limits(ax, np.vstack([outer, inner, pts]))
    ax.set_title("XPBD Relaxed", fontsize=11, fontweight="bold", color="#111827", pad=8)
    _style_axis(ax)


def _draw_final_stage(ax, center: np.ndarray, outer: np.ndarray, inner: np.ndarray, *, valid: bool) -> None:
    c = _finite_rows(center)
    o = _finite_rows(outer)
    inn = _finite_rows(inner)
    if len(o) >= 3 and len(inn) >= 3:
        band = np.vstack([o, inn[::-1]])
        ax.fill(band[:, 0], band[:, 1], color="#2f343b", alpha=0.96, linewidth=0, zorder=1)
        for pts in (o, inn):
            closed = np.vstack([pts, pts[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="#111827", lw=1.15, zorder=2)
    if len(c) >= 3:
        closed = np.vstack([c, c[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#f8fafc", lw=0.75, ls=(0, (5, 4)), zorder=3)
        _set_track_limits(ax, np.vstack([o, inn, c]))
    if not valid:
        ax.text(0.5, 0.5, "invalid", transform=ax.transAxes, ha="center", va="center", color="#dc2626")
    ax.set_title("Inflated Track", fontsize=11, fontweight="bold", color="#111827", pad=8)
    _style_axis(ax)


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
        phase1, valid, chosen = samples[name]
        for col, env_id in enumerate(chosen):
            ax = axes[row, col]
            _draw_phase1_output(ax, phase1, valid, env_id, lw=1.15)
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
        "TrackGen standard phase-1 generator outputs",
        fontsize=16,
        fontweight="bold",
        y=0.985,
        color="#111827",
    )
    fig.tight_layout(rect=(0.055, 0.0, 1.0, 0.965), h_pad=0.45, w_pad=0.20)
    grid_path = output_dir / "readme-generator-grid.png"
    fig.savefig(grid_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    raw, cs, relaxed, final_center, outer, inner, valid, relax_half_width = _generate_pipeline_sample()
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.0), dpi=180, facecolor="white")
    _draw_centerline_stage(axes[0], raw, color="#2563eb", title="Phase 1", dots=False)
    _draw_centerline_stage(axes[1], cs, color="#0f766e", title="Constant Spacing", dots=True)
    _draw_relaxed_stage(axes[2], relaxed, relax_half_width=relax_half_width)
    _draw_final_stage(axes[3], final_center, outer, inner, valid=valid)
    fig.suptitle(
        "Phase-1 centerline to final road band",
        fontsize=15,
        fontweight="bold",
        y=1.02,
        color="#111827",
    )
    fig.tight_layout(w_pad=0.55)
    pipeline_path = output_dir / "readme-pipeline-stages.png"
    fig.savefig(pipeline_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(GENERATORS), figsize=(12.0, 2.75), dpi=180, facecolor="white")
    for ax, (name, label) in zip(axes, GENERATORS):
        phase1, valid, chosen = samples[name]
        _draw_phase1_output(ax, phase1, valid, chosen[0], lw=1.65)
        ax.set_title(label, fontsize=12, fontweight="bold", color="#111827", pad=8)
    fig.tight_layout(w_pad=0.45)
    strip_path = output_dir / "readme-generator-strip.png"
    fig.savefig(strip_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return [grid_path, pipeline_path, strip_path]


def main() -> None:
    for path in render_readme_assets():
        print(path)


if __name__ == "__main__":
    main()
