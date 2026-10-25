"""Explain the shared sampling tail with actual public pipeline taps.

The drawings stage real before/after data; they are not elapsed XPBD iterations.
Canonical replay checks protect both the initial target-spacing samples and the
final same-count redistribution. No library buffers or results are changed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from track_gen import ops
from track_gen.exec import LaunchContext
from track_gen.testing.capture import result_arrays
from viz.racing_theme import get_palette, plotting_theme, save_figure, apply_plot_theme, geometry_digest

STAGES = (
    ("Raw curve: a fixed number of samples", "The generator returns 256 raw samples. This is a point-count choice."),
    ("Sample at the configured target spacing", "Samples use distance k × spacing along the raw loop; the closing remainder can be shorter."),
    ("Relax the positions", "XPBD changes the curve while keeping its point count. Arrows show net before/after motion."),
    ("Redistribute the same number of points", "Equal arc positions on the relaxed source curve use its new length divided by the count."),
)


def _closed(points):
    return np.concatenate([points, points[:1]])



@lru_cache(maxsize=4)
def capture(generator="polar", seed=7):
    """Capture one real road once; both visual themes reuse the same geometry."""
    from viz.render_generator_guide import sample_batch

    samples, gen = sample_batch(generator, seeds=[seed], return_generator=True)
    sample, cfg = samples[0], gen.config
    public = {name: value.numpy().tobytes() for name, value in result_arrays(gen.program.result).items()}
    n = int(gen.program.tap("cl.count").numpy()[0])
    raw = gen.program.tap("raw.xy").numpy().copy()
    target_wp = gen.program.tap("cl.xy", after="resample")
    relaxed_wp = gen.program.tap("cl.xy", after="relax")
    uniform_wp = gen.program.tap("cl.xy", after="uniform")
    target = target_wp.numpy()[:n].copy()
    relaxed = relaxed_wp.numpy()[:n].copy()
    uniform = uniform_wp.numpy()[:n].copy()
    ctx = LaunchContext.current("cpu")
    scratch = ops.ArcScratch.alloc(1, max(len(raw), cfg.N_max), "cpu", compact=False)
    replay = wp.empty(cfg.N_max, dtype=wp.vec2f, device="cpu")
    count = wp.empty(1, dtype=wp.int32, device="cpu")
    truncated = wp.empty(1, dtype=wp.int32, device="cpu")
    raw_wp = wp.array(raw, dtype=wp.vec2f, device="cpu")
    ops.resample_constant_spacing(ctx, raw_wp, len(raw), cfg.spacing, cfg.N_max,
                                  scratch, replay, count, truncated)
    assert replay.numpy().tobytes() == target_wp.numpy().tobytes()
    assert int(count.numpy()[0]) == n
    raw_arc_total = float(scratch.s.numpy()[len(raw)])
    uniform_scratch = ops.ArcScratch.alloc(1, cfg.N_max, "cpu", compact=False)
    uniform_replay = wp.empty(cfg.N_max, dtype=wp.vec2f, device="cpu")
    ops.resample_uniform(ctx, relaxed_wp, cfg.N_max, uniform_scratch, uniform_replay,
                         count=gen.program.tap("cl.count"))
    assert uniform_replay.numpy().tobytes() == uniform_wp.numpy().tobytes()
    relaxed_arc_total = float(uniform_scratch.s.numpy()[n])
    assert public == {name: value.numpy().tobytes() for name, value in result_arrays(gen.program.result).items()}
    np.testing.assert_array_equal(uniform, sample["center"][:, :2])
    data = dict(generator=generator, seed=int(seed), raw=raw, target=target,
                relaxed=relaxed, uniform=uniform, outer=sample["outer"], inner=sample["inner"],
                raw_count=len(raw), count=n, spacing=float(cfg.spacing),
                raw_arc_total=raw_arc_total, relaxed_arc_total=relaxed_arc_total,
                nominal_uniform_step=relaxed_arc_total / n,
                closing_remainder=raw_arc_total - (n-1)*float(cfg.spacing),
                truncated=bool(truncated.numpy()[0]), valid=bool(sample["valid"]),
                constant_spacing_replay_equal=True, uniform_replay_equal=True,
                public_results_unchanged=True)
    data["geometry_sha256"] = geometry_digest({key: data[key] for key in
        ("raw", "target", "relaxed", "uniform", "outer", "inner")})
    return data


def _style(ax, data, palette):
    points = np.concatenate([data["raw"], data["relaxed"], data["outer"][:, :2]])
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo+hi)/2
    radius = max(hi-lo)*.59
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           aspect="equal", xlabel="x", ylabel="y")
    ax.grid(color=palette.grid, alpha=.4, lw=.6)
    ax.spines[["top", "right"]].set_visible(False)


def draw_step(ax, ruler, data, stage, progress, theme="light"):
    """Draw actual samples and a ruler measuring distance along the source path."""
    p = get_palette(theme)
    _style(ax, data, p)
    curve = data["relaxed"] if stage >= 2 else data["raw"]
    ax.plot(*_closed(curve).T, color=p.muted, lw=1.1)
    key = ("raw", "target", "relaxed", "uniform")[stage]
    points = data[key]
    k = max(1, int(np.ceil(len(points)*progress)))
    if stage == 2:
        ax.plot(*_closed(data["target"]).T, color=p.orange, lw=1.3, alpha=.8,
                label="before XPBD")
        for a, b in zip(data["target"][::6], data["relaxed"][::6]):
            ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=p.orange, lw=1))
        ax.plot(*_closed(points).T, color=p.blue, lw=1.8, label="after XPBD")
        ax.legend(loc="upper right", fontsize=9)
    ax.scatter(*points[:k].T, s=13 if stage == 0 else 24, color=p.blue, zorder=4)
    subtitle = (f"{data['raw_count']} raw points", f"spacing target {data['spacing']:.3f} · {data['count']} points",
                f"same {data['count']} points · positions change",
                f"same {data['count']} points · nominal arc step ≈ {data['nominal_uniform_step']:.6f}")[stage]
    ax.set_title(subtitle, color=p.ink, fontsize=12, pad=12)
    ruler.set_xlim(-.012, 3.25*data["spacing"])
    ruler.set_ylim(-.65, 1.35)
    ruler.axis("off")
    ruler.set_title("Ruler: distance along the SOURCE path", fontsize=11, color=p.ink)
    if stage == 0:
        ruler.text(.5, .6, "Point count is fixed here;\nthe road-spacing target comes next.",
                   transform=ruler.transAxes, ha="center", va="center", color=p.ink)
    elif stage == 2:
        ruler.text(.5, .6, "XPBD can change length.\nA fixed count does not fix the spacing.",
                   transform=ruler.transAxes, ha="center", va="center", color=p.ink)
    else:
        step = data["spacing"] if stage == 1 else data["nominal_uniform_step"]
        values = np.arange(4)*step
        ruler.plot([0, values[-1]], [1, 1], color=p.muted)
        ruler.scatter(values, np.ones(4), color=p.blue, s=30)
        for i, value in enumerate(values):
            ruler.text(value, .75, f"{value:.3f}", ha="center", fontsize=10, color=p.ink)
            if i:
                ruler.text(value-step/2, 1.14, f"{step:.3f}", ha="center", fontsize=9, color=p.orange)
        closing = data["closing_remainder"] if stage == 1 else step
        text = (f"Closing remainder ≈ {closing:.4f}; it can be shorter."
                if stage == 1 else "The closing interval has the same nominal arc target.")
        ruler.text(.5, .17, text, transform=ruler.transAxes, ha="center", color=p.ink, fontsize=10)
    return dict(active_curve=points[:k].copy(), sample_count=len(points), key=key)


def render(out_dir, *, movies=True, theme="light", writer_factory=None):
    """Render a shared explanatory ending for the generator construction clips."""
    data = capture()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    p = get_palette(theme)
    footer = ("Actual Polar seed 7 pipeline taps · explanation stages, not elapsed solve time.\n"
              "Equal arc positions on a source curve do not imply equal straight-line chords across bends.")
    with plotting_theme(theme):
        fig = plt.figure(figsize=(12, 7.2))
        grid = fig.add_gridspec(2, 2, width_ratios=(1.6, 1), height_ratios=(1, 1),
                               left=.06, right=.96, top=.83, bottom=.15, wspace=.22, hspace=.35)
        ax = fig.add_subplot(grid[:, 0]); ruler = fig.add_subplot(grid[0, 1])
        road = fig.add_subplot(grid[1, 1]); _style(road, data, p)
        road.fill(*_closed(data["outer"][:, :2]).T, color=p.road)
        road.fill(*_closed(data["inner"][:, :2]).T, color=p.background)
        road.plot(*_closed(data["uniform"]).T, color=p.blue, lw=1.3)
        road.scatter(*data["uniform"].T, color=p.blue, s=7)
        road.set_title(f"Finished road · {data['count']} points", fontsize=11, color=p.green)
        title = fig.text(.5, .95, "", ha="center", fontsize=18, color=p.ink)
        text = fig.text(.5, .89, "", ha="center", fontsize=11, color=p.ink)
        fig.text(.5, .035, footer, ha="center", fontsize=10, color=p.ink)
        def frame(stage, progress):
            ax.clear(); ruler.clear()
            title.set_text(STAGES[stage][0]); text.set_text(STAGES[stage][1])
            draw_step(ax, ruler, data, stage, progress, theme)
            apply_plot_theme(fig, theme)
        if movies:
            if writer_factory is None:
                from viz.render_generator_guide import _writer
                writer_factory = _writer
            writer = writer_factory()
            with writer.saving(fig, str(out_dir/"pipeline-sampling.mp4"), dpi=120):
                for stage in range(4):
                    for progress in [*np.linspace(.03, 1., 24), *([1.]*12)]:
                        frame(stage, progress); writer.grab_frame()
        frame(3, 1.)
        save_figure(fig, out_dir/"pipeline-sampling.png", theme=theme, dpi=150)
        plt.close(fig)
    return dict(kind="shared sampling-tail explanation", theme=theme, geometry_sha256=data["geometry_sha256"],
                stage_titles=[title for title, _ in STAGES], frames=144 if movies else 0,
                duration_seconds=24 if movies else 0, raw_count=data["raw_count"], final_count=data["count"],
                spacing=data["spacing"], nominal_uniform_step=data["nominal_uniform_step"],
                constant_spacing_replay_equal=True, uniform_replay_equal=True,
                geometry_interpolated=False, elapsed_runtime=False,
                output="pipeline-sampling.mp4" if movies else "pipeline-sampling.png")
