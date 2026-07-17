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
# (so the gate self-collision separation is unmistakable in the before/after).
GATE_ASSET_GATE_WIDTH = 0.16
GATE_ASSET_GATE_RADIUS = 0.13
GATE_ASSET_SOLVE_ITERS = 16

# Metric benchmark regime used by the relaxation assets (the "hard" regime of the
# relaxation convergence benchmarks): 1 m road on a ~20 m track, constant 0.30 m spacing.
RELAX_ASSET_OVERRIDES = dict(half_width=0.5, scale=10.0, spacing=0.30, N_max=384)
# Old-solver configuration (pre-Chebyshev defaults) for the comparison assets.
RELAX_OLD_SOLVER = dict(relax_accel="none", relax_sep_every=40)
# Categorical two-series palette for the convergence chart (validated: CVD-safe on white).
RELAX_COLOR_NEW = "#2563eb"
RELAX_COLOR_OLD = "#ea580c"


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
    position = wp.to_torch(gates.position).cpu().numpy().reshape(batch, g, 3)[..., :2]
    tangent = wp.to_torch(gates.tangent).cpu().numpy().reshape(batch, g, 3)[..., :2]
    left = wp.to_torch(gates.left).cpu().numpy().reshape(batch, g, 3)[..., :2]
    right = wp.to_torch(gates.right).cpu().numpy().reshape(batch, g, 3)[..., :2]
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


def _choose_centerline_examples(
    centerline: np.ndarray,
    centerline_valid: np.ndarray,
    final_valid: np.ndarray,
    needed: int,
) -> list[int]:
    chosen: list[int] = []
    for e in range(centerline.shape[0]):
        if centerline_valid[e] and final_valid[e] and np.isfinite(centerline[e]).all():
            chosen.append(e)
        if len(chosen) >= needed:
            return chosen
    for e in range(centerline.shape[0]):
        if e not in chosen and centerline_valid[e] and np.isfinite(centerline[e]).all():
            chosen.append(e)
        if len(chosen) >= needed:
            return chosen
    return chosen


def _generate(name: str, seed: int, needed: int = 5, overrides: dict | None = None):
    batch = 24
    cfg = TrackGenConfig(generator=name, num_envs=batch, device="cpu", **(overrides or {}))
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    generator = TrackGenerator(cfg, rng)
    track = generator.generate()
    scratch = generator._scratch
    centerline = wp.to_torch(scratch.gen_centerline).cpu().numpy().reshape(batch, cfg.num_points, 2)
    centerline_valid = wp.to_torch(scratch.gen_valid).cpu().numpy().astype(bool)
    final_valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    chosen = _choose_centerline_examples(centerline, centerline_valid, final_valid, needed)
    return centerline, centerline_valid, chosen


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
    final_all = wp.to_torch(track.center).cpu().numpy().reshape(batch, cfg.N_max, 3)[..., :2]
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
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, cfg.N_max, 3)[env_id, :count, :2]
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, cfg.N_max, 3)[env_id, :count, :2]
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


def _draw_centerline_output(ax, centerline: np.ndarray, valid: np.ndarray, e: int, *, lw: float) -> None:
    pts = _finite_rows(centerline[e])
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
        centerline, valid, chosen = samples[name]
        for col, env_id in enumerate(chosen):
            ax = axes[row, col]
            _draw_centerline_output(ax, centerline, valid, env_id, lw=1.15)
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
        "TrackGen centerline generator outputs",
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
    _draw_centerline_stage(axes[0], raw, color="#2563eb", title="Raw Centerline", dots=False)
    _draw_centerline_stage(axes[1], cs, color="#0f766e", title="Constant Spacing", dots=True)
    _draw_relaxed_stage(axes[2], relaxed, relax_half_width=relax_half_width)
    _draw_final_stage(axes[3], final_center, outer, inner, valid=valid)
    fig.suptitle(
        "Generated centerline to final road band",
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
        centerline, valid, chosen = samples[name]
        _draw_centerline_output(ax, centerline, valid, chosen[0], lw=1.65)
        ax.set_title(label, fontsize=12, fontweight="bold", color="#111827", pad=8)
    fig.tight_layout(w_pad=0.45)
    strip_path = output_dir / "readme-generator-strip.png"
    fig.savefig(strip_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    gate_strip_path = render_gate_assets(output_dir)

    batch_path = render_batch_of_tracks(output_dir)

    panel_paths = render_generator_panels(output_dir)

    relaxation_paths = render_relaxation_assets(output_dir)

    return [grid_path, pipeline_path, strip_path, gate_strip_path, batch_path,
            *panel_paths, *relaxation_paths]


def render_generator_panels(output_dir: Path = OUT_DIR,
                            config_overrides: dict | None = None) -> list[Path]:
    # config_overrides maps a generator name to TrackGenConfig kwargs used only for that
    # panel. Left None (the committed-asset path), every generator renders at its full default
    # config. Tests pass a reduced repulsive config to keep the CPU smoke test fast without
    # touching the other panels or regenerating the committed PNGs.
    overrides = config_overrides or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    written: list[Path] = []
    for idx, (name, label) in enumerate(GENERATORS):
        centerline, valid, chosen = _generate(name, seed=100 + 17 * idx, needed=5,
                                          overrides=overrides.get(name))
        ncol = max(1, len(chosen))
        fig, axes = plt.subplots(1, ncol, figsize=(2.3 * ncol, 2.6), dpi=170, facecolor="white")
        axes = axes if ncol > 1 else [axes]
        for ax, env_id in zip(axes, chosen):
            _draw_centerline_output(ax, centerline, valid, env_id, lw=1.5)
        fig.suptitle(f"{label} — representative centerline outputs", fontsize=13,
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
    fig.suptitle("Gate self-collision relaxation: raw anchors vs separated gates",
                 fontsize=15, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.96), h_pad=0.6, w_pad=0.3)
    path = output_dir / "readme-gate-strip.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_batch_of_tracks(output_dir: Path = OUT_DIR) -> Path:
    """Grid of finished, inflated track bands from ONE deterministic CPU batch — the end
    product of ``TrackGenerator.generate()`` at the track tutorial's default geometry
    (``half_width=0.03``). The first eight valid, finite tracks in seed order are shown as
    filled constant-width road bands (dark fill, outer/inner borders, dashed centerline),
    the same band styling as the ``Inflated Track`` panel of the pipeline figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    batch, seed, n_show = 64, 100, 8
    cfg = TrackGenConfig(num_envs=batch, device="cpu", half_width=0.03)
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(track.count).cpu().numpy().astype(int)
    center = wp.to_torch(track.center).cpu().numpy().reshape(batch, cfg.N_max, 3)[..., :2]
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, cfg.N_max, 3)[..., :2]
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, cfg.N_max, 3)[..., :2]

    # Rank valid tracks by roundness (isoperimetric ratio 4*pi*A / P^2, in [0, 1];
    # higher == rounder, more open) and show the eight cleanest — deterministic.
    scored: list[tuple[float, int]] = []
    for e in range(batch):
        c = int(count[e])
        if not valid[e] or c < 4 or not np.isfinite(center[e, :c]).all():
            continue
        pts = center[e, :c]
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        perim = float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())
        if perim <= 0.0:
            continue
        scored.append((4.0 * np.pi * area / (perim * perim), e))
    scored.sort(reverse=True)
    chosen = [e for _, e in scored[:n_show]]

    ncol, nrow = 4, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), dpi=170,
                             facecolor="white")
    flat = axes.ravel()
    for ax, e in zip(flat, chosen):
        c = int(count[e])
        _draw_final_stage(ax, center[e, :c], outer[e, :c], inner[e, :c], valid=True)
        ax.set_title("")  # drop the per-panel "Inflated Track" label; the suptitle covers it
    for ax in flat[len(chosen):]:
        ax.axis("off")
    fig.suptitle("A generated batch: finished constant-width track bands",
                 fontsize=14, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=0.25, h_pad=0.4)
    path = output_dir / "batch-of-tracks.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _relax_track_batch(seed: int, batch: int, overrides: dict | None = None):
    """One deterministic CPU batch in the relaxation metric regime; returns the Track
    plus (center [E,N,2], valid [E], count [E]) numpy views."""
    cfg = TrackGenConfig(num_envs=batch, device="cpu",
                         **RELAX_ASSET_OVERRIDES, **(overrides or {}))
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    E = valid.shape[0]
    n_max = track.center.shape[0] // E
    center = wp.to_torch(track.center).cpu().numpy().reshape(E, n_max, 3)[..., :2]
    count = wp.to_torch(track.count).cpu().numpy().astype(int)
    return track, center, valid, count


def _relax_row_label(ax, text: str) -> None:
    ax.set_ylabel(text, rotation=0, ha="right", va="center", labelpad=18,
                  fontsize=9.5, fontweight="bold", color="#111827")


def _menger_kappa(points: np.ndarray) -> np.ndarray:
    """Per-vertex Menger curvature over a closed polyline (circular indexing).

    Uses the same 4*area / (|a||b||c|) triangle formula as the XPBD bending kernel
    (``_step_kernel``), evaluated at each vertex from its two neighbours. Returns an
    array of length ``len(points)`` (units: 1 / length)."""
    pm = np.roll(points, 1, axis=0)
    pp = np.roll(points, -1, axis=0)
    a = points - pm
    b = pp - points
    c = pp - pm
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    area = 0.5 * np.abs(cross)
    denom = np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1),
        1.0e-12,
    )
    return 4.0 * area / denom


def _dkappa_rms(center: np.ndarray, count: int, half_width: float) -> float:
    """Curvature-noise metric: RMS of adjacent per-vertex Menger-curvature differences,
    made dimensionless by scaling with ``half_width``. Higher == rippled centerline."""
    kappa = _menger_kappa(center[:count])
    dk = kappa - np.roll(kappa, -1)
    return float(np.sqrt(np.mean(dk * dk)) * half_width)


def _relax_borders_np(track, batch: int):
    """Return (center, outer, inner) as [E, N_max, 2] numpy views of a relaxation Track."""
    n_max = track.center.shape[0] // batch
    center = wp.to_torch(track.center).cpu().numpy().reshape(batch, n_max, 3)[..., :2]
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, n_max, 3)[..., :2]
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, n_max, 3)[..., :2]
    return center, outer, inner


def _draw_band_zoom(ax, center, outer, inner, count, e, idx, half: int,
                    xlim, ylim, *, lw: float = 1.4) -> None:
    """Draw the real pipeline band (outer/inner borders + dashed centerline, IDENTICAL
    construction to ``draw_track``) for env ``e`` but clipped to fixed ``xlim``/``ylim``
    and with beads marked, so a short rippled border segment reads clearly."""
    n = int(count[e])
    lo, hi = idx - half, idx + half + 1
    win = np.arange(lo, hi) % n
    c, o, inn = center[e, win], outer[e, win], inner[e, win]
    if np.isfinite(o).all() and np.isfinite(inn).all():
        band = np.vstack([o, inn[::-1]])
        ax.fill(band[:, 0], band[:, 1], color="0.85", zorder=1, linewidth=0)
    ax.plot(o[:, 0], o[:, 1], color="#1f77b4", lw=lw, zorder=3)
    ax.plot(o[:, 0], o[:, 1], ".", color="#1f77b4", ms=3.2, zorder=4)
    ax.plot(inn[:, 0], inn[:, 1], color="#d62728", lw=lw, zorder=3)
    ax.plot(c[:, 0], c[:, 1], color="0.25", lw=0.7, ls="--", zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _style_axis(ax)


def render_relaxation_smoothing(output_dir: Path = OUT_DIR) -> Path:
    """Smoothing-tail benefit figure. One batch generated TWICE with identical seeds in
    the metric benchmark regime: once with the tail OFF (``relax_smooth_passes=0``,
    ``relax_smooth_spacing_iters=0``) and once with the defaults ON. The four valid tracks
    with the highest curvature noise (dkappa RMS) in the WITHOUT-tail batch are shown in
    both states — top row without the tail, bottom row with it — as full-track bands built
    from the real pipeline ``outer``/``inner`` borders, plus a 5th column zooming the most
    rippled border segment of the worst track so the sub-R_min wiggle the tail removes is
    visible."""
    from viz.plot_tracks import draw_track

    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    batch, seed, n_examples = 256, 909, 4
    hw = float(RELAX_ASSET_OVERRIDES["half_width"])

    no_track, no_center, no_valid, no_count = _relax_track_batch(
        seed, batch, {"relax_smooth_passes": 0, "relax_smooth_spacing_iters": 0})
    yes_track, yes_center, yes_valid, yes_count = _relax_track_batch(seed, batch)

    # Rank valid WITHOUT-tail tracks by curvature noise; take the noisiest four.
    scores: list[tuple[float, int]] = []
    for e in range(batch):
        n = int(no_count[e])
        if not no_valid[e] or n < 8 or not np.isfinite(no_center[e, :n]).all():
            continue
        scores.append((_dkappa_rms(no_center[e], n, hw), e))
    scores.sort(reverse=True)
    chosen = [e for _, e in scores[:n_examples]]
    no_dk = {e: s for s, e in scores}

    ncol = n_examples + 1
    fig, axes = plt.subplots(2, ncol, figsize=(2.35 * ncol, 5.0), dpi=175, facecolor="white")
    for col, e in enumerate(chosen):
        draw_track(axes[0, col], no_track, e)
        draw_track(axes[1, col], yes_track, e)
        n = int(yes_count[e])
        axes[0, col].set_title(f"dκ RMS = {no_dk[e]:.3f}", fontsize=9.5,
                               fontweight="bold", color="#b91c1c", pad=4)
        axes[1, col].set_title(f"dκ RMS = {_dkappa_rms(yes_center[e], n, hw):.3f}",
                               fontsize=9.5, fontweight="bold", color="#15803d", pad=4)

    # 5th column: zoom the most rippled border segment of the worst (col-0) track. The
    # window and axis limits are identical for both rows so the ripple difference is the
    # only change on screen.
    ez = chosen[0]
    nz = int(no_count[ez])
    kap = _menger_kappa(no_center[ez, :nz])
    dk = np.abs(kap - np.roll(kap, -1))
    # Smooth |dk| over a small window and centre the zoom on the busiest run of ripple.
    half = 11
    win_scores = np.array([dk[(np.arange(i - half, i + half + 1)) % nz].sum()
                           for i in range(nz)])
    idx = int(win_scores.argmax())
    no_center_b, no_outer, no_inner = _relax_borders_np(no_track, batch)
    yes_center_b, yes_outer, yes_inner = _relax_borders_np(yes_track, batch)
    seg = (np.arange(idx - half, idx + half + 1)) % nz
    pts = np.vstack([no_outer[ez, seg], no_inner[ez, seg],
                     yes_outer[ez, seg], yes_inner[ez, seg]])
    pts = pts[np.isfinite(pts).all(axis=1)]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin) * 0.5
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    pad = span * 1.15
    xlim, ylim = (cx - pad, cx + pad), (cy - pad, cy + pad)
    _draw_band_zoom(axes[0, ncol - 1], no_center_b, no_outer, no_inner, no_count,
                    ez, idx, half, xlim, ylim)
    _draw_band_zoom(axes[1, ncol - 1], yes_center_b, yes_outer, yes_inner, yes_count,
                    ez, idx, half, xlim, ylim)
    axes[0, ncol - 1].set_title("worst segment (zoom)", fontsize=9.5,
                                fontweight="bold", color="#111827", pad=4)

    _relax_row_label(axes[0, 0], "without tail\n(passes=0, polish=0)")
    _relax_row_label(axes[1, 0], "with tail\n(5 Taubin + 10 polish)")
    fig.suptitle("Post-solve smoothing tail: curvature noise removed at equal validity",
                 fontsize=14, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.10, 0.0, 1.0, 0.96), h_pad=0.7, w_pad=0.3)
    path = output_dir / "relaxation-smoothing.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_relaxation_before_after(output_dir: Path = OUT_DIR) -> Path:
    """Same seeds, top row raw constant-spacing centerlines (relax_enable=False, drawn
    centerline-only), bottom row the relaxed + inflated constant-width band."""
    from viz.plot_tracks import draw_track

    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    batch, seed, n_examples = 24, 424, 5
    raw_track, raw_center, raw_valid, raw_count = _relax_track_batch(
        seed, batch, {"relax_enable": False})
    rel_track, rel_center, rel_valid, rel_count = _relax_track_batch(seed, batch)

    # Prefer envs where the raw (unrelaxed) band fails the validity gate but the relaxed
    # one passes — the panels where relaxation is visibly load-bearing.
    def _finite(e: int) -> bool:
        pts = raw_center[e, : raw_count[e]]
        return raw_count[e] >= 4 and bool(np.isfinite(pts).all())

    chosen = [e for e in range(batch) if rel_valid[e] and not raw_valid[e] and _finite(e)]
    chosen += [e for e in range(batch) if rel_valid[e] and _finite(e) and e not in chosen]
    chosen = chosen[:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(2.4 * n_examples, 5.0),
                             dpi=170, facecolor="white")
    for col, e in enumerate(chosen):
        _draw_centerline_stage(axes[0, col], raw_center[e, : raw_count[e]],
                               color="#0f766e", title="", dots=True)
        draw_track(axes[1, col], rel_track, e)
    _relax_row_label(axes[0, 0], "raw constant-spacing\n(relax_enable=False)")
    _relax_row_label(axes[1, 0], "relaxed + inflated")
    fig.suptitle("XPBD relaxation: raw centerlines vs relaxed constant-width bands",
                 fontsize=14, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.09, 0.0, 1.0, 0.96), h_pad=0.7, w_pad=0.3)
    path = output_dir / "relaxation-before-after.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_relaxation_same_seeds(output_dir: Path = OUT_DIR) -> Path:
    """Identical seeds through the old solver (plain Jacobi, 150 sweeps) and the new
    default solver (Chebyshev, 50 sweeps): the visual same-fixed-point argument."""
    from viz.plot_tracks import draw_track

    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    batch, seed, n_examples = 24, 777, 6
    old_track, old_center, old_valid, old_count = _relax_track_batch(
        seed, batch, {**RELAX_OLD_SOLVER, "relax_iters": 150})
    new_track, new_center, new_valid, new_count = _relax_track_batch(seed, batch)

    chosen = [
        e for e in range(batch)
        if old_valid[e] and new_valid[e] and old_count[e] >= 4
        and np.isfinite(old_center[e, : old_count[e]]).all()
        and np.isfinite(new_center[e, : new_count[e]]).all()
    ][:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(2.4 * n_examples, 5.0),
                             dpi=170, facecolor="white")
    for col, e in enumerate(chosen):
        draw_track(axes[0, col], old_track, e)
        draw_track(axes[1, col], new_track, e)
    _relax_row_label(axes[0, 0], "old solver\n(150 plain sweeps)")
    _relax_row_label(axes[1, 0], "new solver\n(50 accelerated sweeps)")
    fig.suptitle("Same seeds, same fixed point: 150 plain-Jacobi vs 50 Chebyshev sweeps",
                 fontsize=14, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.09, 0.0, 1.0, 0.96), h_pad=0.7, w_pad=0.3)
    path = output_dir / "relaxation-same-seeds.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_relaxation_convergence(
    output_dir: Path = OUT_DIR,
    num_envs: int = 2048,
    sweeps: tuple[int, ...] = (25, 50, 75, 100, 150),
    seed: int = 0,
) -> "Path | None":
    """Measured yield vs sweep count, old solver vs new defaults (library defaults
    regime). This is a real CUDA measurement, not an illustration: it generates
    ``num_envs`` tracks per (solver, sweeps) point. Skipped (returns None) when no
    CUDA device is available — the committed asset must come from a GPU run."""
    import torch

    if not torch.cuda.is_available():
        print("relaxation-convergence.png SKIPPED: requires a CUDA device "
              "(the curve is measured, not illustrative)")
        return None
    from viz.plot_tracks import make_rng

    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()

    def _measure(iters: int, overrides: dict) -> float:
        cfg = TrackGenConfig(num_envs=num_envs, device="cuda",
                             relax_iters=iters, **overrides)
        rng = make_rng(num_envs, seed=seed, device="cuda")
        track = TrackGenerator(cfg, rng).generate()
        return float(wp.to_torch(track.valid).float().mean().item())

    old_yield = [_measure(s, dict(RELAX_OLD_SOLVER)) for s in sweeps]
    new_yield = [_measure(s, {}) for s in sweeps]
    for s, oy, ny in zip(sweeps, old_yield, new_yield):
        print(f"  sweeps={s:4d}  old={oy:.4f}  new={ny:.4f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=170, facecolor="white")
    ax.plot(sweeps, new_yield, color=RELAX_COLOR_NEW, lw=2.0, marker="o", ms=6,
            label='new solver (Chebyshev, relax_sep_every=20)', zorder=3)
    ax.plot(sweeps, old_yield, color=RELAX_COLOR_OLD, lw=2.0, marker="s", ms=6,
            label='old solver (relax_accel="none", relax_sep_every=40)', zorder=3)
    ax.annotate("new @ 50", (50, new_yield[sweeps.index(50)]),
                textcoords="offset points", xytext=(6, -14),
                fontsize=8.5, color="#111827")
    ax.annotate("old @ 150", (150, old_yield[-1]),
                textcoords="offset points", xytext=(-4, -14),
                fontsize=8.5, color="#111827", ha="right")
    ax.set_xlabel("relax_iters (Jacobi sweeps)", fontsize=10, color="#111827")
    ax.set_ylabel("valid-track yield", fontsize=10, color="#111827")
    ax.set_title(f"Relaxation yield vs sweep count — library defaults, E={num_envs}, CUDA",
                 fontsize=11.5, fontweight="bold", color="#111827", pad=10)
    ax.set_xticks(list(sweeps))
    ax.set_ylim(min(min(old_yield), min(new_yield)) - 0.03, 1.005)
    ax.grid(True, color="0.92", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.75")
    ax.tick_params(colors="#374151", labelsize=9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    path = output_dir / "relaxation-convergence.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_relaxation_assets(output_dir: Path = OUT_DIR) -> list[Path]:
    """Render the three docs/relaxation figures. The convergence curve needs CUDA and
    is skipped (with a printed note) on GPU-free machines; the other two are CPU."""
    paths = [
        render_relaxation_before_after(output_dir),
        render_relaxation_same_seeds(output_dir),
        render_relaxation_smoothing(output_dir),
    ]
    convergence = render_relaxation_convergence(output_dir)
    if convergence is not None:
        paths.append(convergence)
    return paths


def main() -> None:
    for path in render_readme_assets():
        print(path)


if __name__ == "__main__":
    main()
