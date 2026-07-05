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
    ("repulsive", "Repulsive"),
]

# GateGenerator does not register "repulsive" (it is not a gate-sequence generator), so the
# gate-asset strip stays at the original five.
GATE_GENERATORS = [
    ("bezier", "Bezier"),
    ("checkpoint", "Checkpoint"),
    ("hull", "Hull"),
    ("polar", "Polar"),
    ("voronoi", "Voronoi"),
]

# Illustrative gate geometry for the asset: a non-zero opening so the gate bars are
# visible, and a radius large enough that raw anchors overlap before the collision solve
# (so the phase-2 separation is unmistakable in the before/after).
GATE_ASSET_GATE_WIDTH = 0.16
GATE_ASSET_GATE_RADIUS = 0.13
GATE_ASSET_SOLVE_ITERS = 16


def _gate_batch(name: str, *, seed: int, solve_iters: int):
    """Run one CPU gate batch; return (position, tangent, left, right, valid, count) numpy."""
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

    batch = 24
    cfg = GateGenConfig(
        generator=name,
        num_envs=batch,
        device="cpu",
        gate_radius=GATE_ASSET_GATE_RADIUS,
        gate_width=GATE_ASSET_GATE_WIDTH,
        gate_solve_iters=solve_iters,
    )
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    gates = GateGenerator(cfg, rng).generate()
    g = gates.position.shape[0] // batch
    position = wp.to_torch(gates.position).cpu().numpy().reshape(batch, g, 2)
    tangent = wp.to_torch(gates.tangent).cpu().numpy().reshape(batch, g, 2)
    left = wp.to_torch(gates.left).cpu().numpy().reshape(batch, g, 2)
    right = wp.to_torch(gates.right).cpu().numpy().reshape(batch, g, 2)
    valid = wp.to_torch(gates.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(gates.count).cpu().numpy().astype(int)
    return position, tangent, left, right, valid, count


def _choose_gate_env(valid: np.ndarray, count: np.ndarray, raw_pos: np.ndarray, solved_pos: np.ndarray) -> int:
    """Pick the valid, finite, >=4-gate env where the collision solve moved gates most."""
    best_e, best_disp = -1, -1.0
    for e in range(solved_pos.shape[0]):
        c = int(count[e])
        if c < 4 or not valid[e]:
            continue
        r, s = raw_pos[e, :c], solved_pos[e, :c]
        if not (np.isfinite(r).all() and np.isfinite(s).all()):
            continue
        disp = float(np.linalg.norm(s - r, axis=1).max())
        if disp > best_disp:
            best_disp, best_e = disp, e
    if best_e >= 0:
        return best_e
    for e in range(solved_pos.shape[0]):
        c = int(count[e])
        if c >= 2 and np.isfinite(solved_pos[e, :c]).all():
            return e
    return 0


def _set_gate_limits(ax, pts: np.ndarray) -> None:
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < 2:
        return
    xmin, ymin = finite.min(axis=0)
    xmax, ymax = finite.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0e-3)
    pad = 0.18 * span + 1.6 * GATE_ASSET_GATE_RADIUS
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ax.set_xlim(cx - 0.5 * span - pad, cx + 0.5 * span + pad)
    ax.set_ylim(cy - 0.5 * span - pad, cy + 0.5 * span + pad)


def _draw_gate_asset(ax, position, tangent, left, right, count, e: int, *, draw_frames: bool) -> None:
    c = int(count[e])
    pos = position[e, :c]
    finite = np.isfinite(pos).all(axis=1)
    pos = pos[finite]
    if len(pos) >= 2:
        closed = np.vstack([pos, pos[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.35", lw=0.7, ls="--", alpha=0.6, zorder=1)
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=16, color="#111827", zorder=4)
    for pnt in pos:
        ax.add_patch(plt.Circle((pnt[0], pnt[1]), GATE_ASSET_GATE_RADIUS, fill=False,
                                color="#64748b", lw=0.8, alpha=0.85, zorder=2))
    if draw_frames:
        tan = tangent[e, :c][finite]
        lft = left[e, :c][finite]
        rgt = right[e, :c][finite]
        if len(tan) == len(pos) and len(pos) > 0:
            ax.quiver(pos[:, 0], pos[:, 1], tan[:, 0], tan[:, 1], angles="xy",
                      scale_units="xy", scale=12, width=0.005, color="#f97316",
                      alpha=0.85, zorder=3)
        if len(lft) == len(rgt) == len(pos):
            for li, ri in zip(lft, rgt):
                ax.plot([li[0], ri[0]], [li[1], ri[1]], color="#2563eb", lw=1.3, zorder=3)
    _set_gate_limits(ax, pos)
    _style_axis(ax)


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

    gate_strip_path = render_gate_assets(output_dir)

    panel_paths = render_generator_panels(output_dir)

    return [grid_path, pipeline_path, strip_path, gate_strip_path, *panel_paths]


def render_generator_panels(output_dir: Path = OUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    written: list[Path] = []
    for idx, (name, label) in enumerate(GENERATORS):
        phase1, valid, chosen = _generate(name, seed=100 + 17 * idx, needed=5)
        ncol = max(1, len(chosen))
        fig, axes = plt.subplots(1, ncol, figsize=(2.3 * ncol, 2.6), dpi=170, facecolor="white")
        axes = axes if ncol > 1 else [axes]
        for ax, env_id in zip(axes, chosen):
            _draw_phase1_output(ax, phase1, valid, env_id, lw=1.5)
        fig.suptitle(f"{label} — representative phase-1 outputs", fontsize=13,
                     fontweight="bold", color="#111827")
        fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.4)
        path = output_dir / f"generator-{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)
    return written


def render_gate_assets(output_dir: Path = OUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    n = len(GATE_GENERATORS)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 4.8), dpi=170, facecolor="white")
    for col, (name, label) in enumerate(GATE_GENERATORS):
        seed = 100 + 23 * col
        solved = _gate_batch(name, seed=seed, solve_iters=GATE_ASSET_SOLVE_ITERS)
        s_pos, s_tan, s_left, s_right, s_valid, s_count = solved
        raw = _gate_batch(name, seed=seed, solve_iters=0)
        r_pos, r_tan, r_left, r_right, r_valid, r_count = raw
        env = _choose_gate_env(s_valid, s_count, r_pos, s_pos)
        _draw_gate_asset(axes[0, col], r_pos, r_tan, r_left, r_right, r_count, env, draw_frames=False)
        _draw_gate_asset(axes[1, col], s_pos, s_tan, s_left, s_right, s_count, env, draw_frames=True)
        axes[0, col].set_title(label, fontsize=12, fontweight="bold", color="#111827", pad=8)
    axes[0, 0].set_ylabel("raw anchors\n(gate_solve_iters=0)", rotation=0, ha="right",
                          va="center", labelpad=18, fontsize=9.5, fontweight="bold", color="#111827")
    axes[1, 0].set_ylabel("collision-solved", rotation=0, ha="right", va="center",
                          labelpad=18, fontsize=9.5, fontweight="bold", color="#111827")
    fig.suptitle("Phase-2 gate collision solve: raw anchors vs separated gates",
                 fontsize=15, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.96), h_pad=0.6, w_pad=0.3)
    path = output_dir / "readme-gate-strip.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    for path in render_readme_assets():
        print(path)


if __name__ == "__main__":
    main()
