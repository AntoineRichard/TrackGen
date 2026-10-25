"""Source-backed Polar construction drawings for the reader documentation.

The zero-jitter and radial-only views are explanatory counterfactuals from the
canonical sampler with identical effective seeds. They are not chronological
buffers from a kernel that draws both perturbations together. No runtime
sampling, interpolation or validity implementation is replaced here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


_CACHE = "_polar_storyboard"
_KEYS = ("slots", "radii", "angles", "spline", "resample", "normalize")


def _signature(sample):
    digest = hashlib.sha256()
    digest.update(json.dumps({key: sample[key] for key in ("seed", "epoch", "config")},
                             sort_keys=True).encode())
    for points in (sample["raw"], sample["construction"]["anchors"],
                   sample["construction"]["dense"]):
        digest.update(np.ascontiguousarray(points).tobytes())
    return digest.hexdigest()


def _same(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise RuntimeError(f"Polar storyboard replay differs from the supplied {label}")


def capture(sample):
    """Capture canonical reference views without changing supplied sample arrays.

    This maintenance renderer accepts CPU, epoch-zero samples, as produced by
    the generator-guide sampler. A one-environment replay verifies the supplied
    controls, dense/raw curves and finished geometry before its data are used.
    """
    if sample["config"]["generator"] != "polar":
        raise ValueError("Polar storyboard requires a polar sample")
    if sample["epoch"] != 0 or sample["config"]["device"] != "cpu":
        raise ValueError("Polar storyboard requires a CPU sample at epoch zero")
    signature = _signature(sample)
    cached = sample.get(_CACHE)
    if cached is not None and cached["signature"] == signature:
        return cached

    import warp as wp
    from track_gen import TrackGenConfig, TrackGenerator, ops
    from track_gen._src.warp_generate_polar import sample_polar_controls_per_env
    from track_gen.exec import LaunchContext
    from track_gen.testing.capture import result_arrays

    values = dict(sample["config"], num_envs=1, device="cpu", on_invalid="keep")
    cfg = TrackGenConfig(**values)
    seeds = wp.array(np.array([sample["seed"]], np.int32), dtype=wp.int32, device="cpu")
    gen = TrackGenerator(cfg, seeds=seeds)
    result = gen.generate()
    scratch = {key.rsplit("/", 1)[-1]: value
               for key, value in gen.program._gen_scratch.scratch.items()}
    controls = scratch["controls"].numpy().copy()
    dense_np = scratch["dense"].numpy().copy()
    raw = gen.program.tap("raw.xy").numpy().copy()
    count = int(result.count.numpy()[0])
    _same(controls, sample["construction"]["anchors"], "control knots")
    _same(dense_np, sample["construction"]["dense"], "dense curve")
    _same(raw, sample["raw"], "raw curve")
    if count != sample["count"] or bool(result.valid.numpy()[0]) != sample["valid"] \
            or int(result.fail_mask.numpy()[0]) != sample["fail"]:
        raise RuntimeError("Polar storyboard replay differs from supplied result status")
    for name in ("center", "inner", "outer"):
        _same(getattr(result, name).numpy()[:count, :2], sample[name], name)
    before = {name: arr.numpy().tobytes() for name, arr in result_arrays(result).items()}

    ctx = LaunchContext.current("cpu")
    params = gen.params["generator"]
    K, N, M = len(controls), len(raw), len(dense_np)
    zero = wp.zeros(1, dtype=wp.float32, device="cpu")
    regular = wp.empty(K, dtype=wp.vec2f, device="cpu")
    radial_only = wp.empty(K, dtype=wp.vec2f, device="cpu")
    replay_controls = wp.empty(K, dtype=wp.vec2f, device="cpu")
    effective_seed = gen.program.tap("seed")
    sample_polar_controls_per_env(ctx, effective_seed, K, zero, zero, regular)
    sample_polar_controls_per_env(ctx, effective_seed, K,
                                  params["polar_radial_jitter"], zero, radial_only)
    sample_polar_controls_per_env(ctx, effective_seed, K,
                                  params["polar_radial_jitter"],
                                  params["polar_angular_jitter"], replay_controls)
    _same(replay_controls.numpy(), controls, "replayed controls")

    arc = ops.ArcScratch.alloc(1, M, "cpu")
    pre_raw = wp.empty(N, dtype=wp.vec2f, device="cpu")
    raw_count = wp.empty(1, dtype=wp.int32, device="cpu")
    ops.arc_resample(ctx, scratch["dense"], M, N, arc, pre_raw, raw_count)
    normalized = wp.clone(pre_raw)
    ops.normalize_extent(ctx, normalized, N, ops.constants.TARGET_EXTENT*float(cfg.scale),
                         size=params["size"])
    _same(normalized.numpy(), raw, "arc-resampled and normalized curve")
    if before != {name: arr.numpy().tobytes() for name, arr in result_arrays(result).items()}:
        raise RuntimeError("Polar diagnostic capture changed the public result")

    base_angles = np.arange(K)*2*np.pi/K
    actual_angles = np.unwrap(np.arctan2(controls[:, 1].astype(float),
                                        controls[:, 0].astype(float)))
    # The first knot can fall just below angle zero. Unwrap retains that
    # negative offset rather than inventing a separate global rotation.
    angular_offsets = actual_angles - base_angles
    radial = float(np.clip(params["polar_radial_jitter"].numpy()[0], 0., .85))
    angular = float(np.clip(params["polar_angular_jitter"].numpy()[0], 0., .45))
    data = dict(signature=signature, regular=regular.numpy().copy(),
                radial_only=radial_only.numpy().copy(), controls=controls,
                dense=dense_np, raw_before_normalization=pre_raw.numpy().copy(), raw=raw,
                base_angles=base_angles, angular_offsets=angular_offsets,
                radii=np.linalg.norm(controls.astype(float), axis=1),
                radial_jitter=radial, angular_jitter=angular,
                slot_degrees=360./K, angular_bound_degrees=angular*360./K,
                final_count=count, final_valid=sample["valid"], final_fail=sample["fail"],
                references_are_counterfactual=True, controls_replay_equal=True,
                raw_replay_equal=True, public_results_unchanged=True)
    sample[_CACHE] = data
    return data


def construction_stages(sample):
    """Six explanatory operations, with no claim of measured execution time."""
    data = capture(sample)
    K, dense, raw = len(data["controls"]), len(data["dense"]), len(data["raw"])
    descriptions = [
        ("Start from ordered angular slots",
         f"{K} slots are {data['slot_degrees']:g}° apart on the base-radius 1 ring. No random global rotation."),
        ("Choose each knot's radius",
         f"Each knot draws a radius uniformly in [{1-data['radial_jitter']:.2g}, {1+data['radial_jitter']:.2g}]. "
         "This reference view keeps its original angle; the final knots use the same draws."),
        ("Offset angles within their slots",
         f"Each knot gets an independent uniform offset: ±{data['angular_jitter']:g} of a {data['slot_degrees']:g}° slot "
         f"(±{data['angular_bound_degrees']:g}°). Index order stays fixed."),
        ("Join knots with Catmull–Rom segments",
         "Four neighboring knots determine each segment. Reveal the actual dense curve; parameter samples are not evenly spaced by distance."),
        ("Resample to a fixed raw count",
         f"{dense} dense parameter samples → {raw} arc-length samples. These are the actual canonical resampling outputs, before normalization."),
        ("Normalize the raw curve's extent",
         f"Recenter the raw bounding box and scale its longest side. {raw} raw points → {data['final_count']} finished-road points; "
         "the shared spacing/relaxation guide explains the next steps."),
    ]
    return [dict(key=key, title=title, explanation=explanation,
                 progressive_curve=(key == "spline"))
            for key, (title, explanation) in zip(_KEYS, descriptions)]


def _prefix(points, progress):
    points = np.asarray(points)
    n = max(1, int(np.ceil(len(points)*np.clip(progress, 0., 1.))))
    return points[:n]


def _closed(points):
    return np.vstack((points, points[:1]))


def _style(ax, palette, *, output=False, small=False):
    ax.set_facecolor(palette.background)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(palette.border)
    ax.tick_params(colors=palette.muted, labelsize=8 if small else 9)
    units = "output units" if output else "construction units"
    ax.set_xlabel(f"x [{units}]", color=palette.ink, fontsize=8 if small else 10)
    ax.set_ylabel(f"y [{units}]", color=palette.ink, fontsize=8 if small else 10)


def _numbers(ax, points, palette):
    for i, point in enumerate(points):
        ax.annotate(str(i), point, xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=palette.ink, zorder=8)


def _note(ax, text, palette):
    ax.text(.02, .02, text, transform=ax.transAxes, va="bottom", color=palette.ink,
            fontsize=10, bbox=dict(facecolor=palette.panel, edgecolor=palette.border,
                                  boxstyle="round,pad=.35", alpha=.94), zorder=10)


def _ring_guides(ax, data, palette):
    from matplotlib.patches import Circle
    outer = 1. + data["radial_jitter"]
    for angle in data["base_angles"]:
        ax.plot([0., outer*np.cos(angle)], [0., outer*np.sin(angle)],
                color=palette.grid, lw=.9, ls="--", zorder=0)
    ax.add_patch(Circle((0, 0), 1., fill=False, color=palette.muted, lw=1.1, ls="--"))
    if data["radial_jitter"]:
        for radius in (1.-data["radial_jitter"], outer):
            ax.add_patch(Circle((0, 0), radius, fill=False, color=palette.grid, lw=.8, ls=":"))
    ax.scatter([0.], [0.], s=12, color=palette.muted, zorder=3)


def _limits(ax, points, pad=.13):
    points = np.concatenate([np.asarray(p).reshape(-1, 2) for p in points])
    points = points[np.isfinite(points).all(axis=1)]
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = .5*(lo+hi)
    span = max(float(np.ptp(points, axis=0).max()), .1)*(.5+pad)
    ax.set_xlim(center[0]-span, center[0]+span)
    ax.set_ylim(center[1]-span, center[1]+span)


def draw_step(ax, sample, stage, progress, theme="light"):
    """Draw a known-coordinate reveal; guides never stand in for new RNG draws."""
    from matplotlib.patches import Arc, Rectangle, Wedge
    from viz.racing_theme import get_palette

    data = capture(sample)
    palette = get_palette(theme)
    if not 0 <= stage < len(_KEYS):
        raise ValueError(f"Polar construction stage must be 0…{len(_KEYS)-1}")
    key = _KEYS[stage]
    _style(ax, palette, output=(key == "normalize"))
    radius = max(1.+data["radial_jitter"],
                 float(np.linalg.norm(data["dense"], axis=1).max()))*1.13
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius*1.3, radius)
    active = np.empty((0, 2), dtype=np.float32)

    if key in ("slots", "radii", "angles"):
        _ring_guides(ax, data, palette)
        ax.scatter(*data["regular"].T, s=24, facecolors="none", edgecolors=palette.muted,
                   linewidths=.9, zorder=4)
        if key == "slots":
            active = _prefix(data["regular"], progress)
            ax.scatter(*active.T, s=35, color=palette.blue, zorder=5)
            _numbers(ax, active, palette)
            _note(ax, f"base radius = 1\n{len(data['controls'])} slots · {data['slot_degrees']:g}° apart", palette)
        elif key == "radii":
            active = _prefix(data["radial_only"], progress)
            for start, end in zip(data["regular"], active):
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=palette.orange, lw=1.2), zorder=4)
            ax.scatter(*active.T, s=35, color=palette.blue, zorder=5)
            _numbers(ax, active, palette)
            _note(ax, f"r = 1 ± {data['radial_jitter']:g}\nreference: angle offsets = 0", palette)
        else:
            active = _prefix(data["controls"], progress)
            radial = data["radial_only"]
            ax.scatter(*radial.T, s=23, facecolors="none", edgecolors=palette.muted, zorder=4)
            for start, end in zip(radial, active):
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=palette.orange, lw=1.1), zorder=4)
            selected = len(active)-1
            angle = np.degrees(data["base_angles"][selected])
            offset = np.degrees(data["angular_offsets"][selected])
            r = float(np.linalg.norm(radial[selected]))
            bound = data["angular_bound_degrees"]
            ax.add_patch(Wedge((0, 0), r*1.04, angle-bound, angle+bound,
                               facecolor=palette.orange, edgecolor="none", alpha=.10, zorder=1))
            ax.add_patch(Arc((0, 0), 2*r, 2*r, theta1=min(angle, angle+offset),
                             theta2=max(angle, angle+offset), color=palette.orange, lw=2., zorder=5))
            ax.scatter(*active.T, s=35, color=palette.blue, zorder=6)
            _numbers(ax, active, palette)
            _note(ax, f"allowed offset: ±{bound:g}°\nknot {selected}: {offset:+.1f}° · order retained", palette)
    elif key == "spline":
        controls = data["controls"]
        ax.plot(*_closed(controls).T, color=palette.grid, ls="--", lw=1.)
        ax.scatter(*controls.T, s=22, color=palette.muted, zorder=4)
        active = _prefix(data["dense"], progress)
        segment = min((len(active)-1)//(len(data["dense"])//len(controls)), len(controls)-1)
        neighbors = np.array([(segment-1)%len(controls), segment,
                              (segment+1)%len(controls), (segment+2)%len(controls)])
        ax.plot(*controls[neighbors].T, color=palette.orange, lw=1., ls=":")
        ax.scatter(*controls[neighbors].T, s=42, facecolors="none", edgecolors=palette.orange,
                   linewidths=1.5, zorder=6)
        ax.plot(*(_closed(active) if progress >= 1 else active).T,
                color=palette.blue, lw=2., zorder=5)
        _numbers(ax, controls, palette)
        _note(ax, f"segment {segment} → {(segment+1)%len(controls)}\norange: its four neighboring knots", palette)
    elif key == "resample":
        dense = data["dense"]
        active = _prefix(data["raw_before_normalization"], progress)
        # Separate panels keep the actual whole curve visible while a zoom
        # distinguishes dense parameter samples from uniform-arc raw samples.
        ax.set_aspect("auto")
        ax.set_axis_off()
        main = ax.inset_axes([.00, .25, .66, .68])
        _style(main, palette)
        main.set_xlim(-radius, radius)
        main.set_ylim(-radius*1.3, radius)
        main.plot(*_closed(dense).T, color=palette.muted, lw=1., alpha=.65)
        main.scatter(*active.T, s=12, color=palette.blue, zorder=5)
        edge = np.roll(dense, -1, axis=0)-dense
        index = int(np.linalg.norm(edge, axis=1).argmax())
        center = .5*(dense[index]+dense[(index+1)%len(dense)])
        zoom_radius = radius*.11
        main.add_patch(Rectangle(center-zoom_radius, 2*zoom_radius, 2*zoom_radius,
                                  fill=False, edgecolor=palette.orange, lw=.9, ls="--"))
        inset = ax.inset_axes([.71, .41, .29, .40])
        _style(inset, palette, small=True)
        inset.plot(*_closed(dense).T, color=palette.muted, lw=.8)
        inset.scatter(*dense.T, s=8, color=palette.muted)
        inset.scatter(*active.T, s=21, color=palette.blue)
        inset.set_xlim(center[0]-zoom_radius, center[0]+zoom_radius)
        inset.set_ylim(center[1]-zoom_radius, center[1]+zoom_radius)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_xlabel("")
        inset.set_ylabel("")
        inset.set_title("Same region\ngrey: dense · blue: arc", color=palette.ink, fontsize=9)
        _note(ax, f"{len(dense)} dense parameter samples → {len(data['raw'])} uniform-arc raw samples", palette)
    else:
        ax.set_aspect("auto")
        ax.set_axis_off()
        before, after = data["raw_before_normalization"], data["raw"]
        active = _prefix(after, progress)
        for bounds, points, color, title, output in [
            ([.01, .29, .43, .62], before, palette.muted, "Before normalization", False),
            ([.58, .29, .42, .62], active, palette.blue, "Normalized raw", True),
        ]:
            panel = ax.inset_axes(bounds)
            _style(panel, palette, output=output, small=True)
            reference = before if not output else after
            _limits(panel, [reference])
            panel.plot(*_closed(reference).T, color=palette.grid, lw=.9)
            panel.plot(*(points if output and progress < 1 else _closed(points)).T,
                       color=color, lw=1.5)
            panel.scatter(*points.T, color=color, s=7)
            lo, hi = reference.min(axis=0), reference.max(axis=0)
            panel.add_patch(Rectangle(lo, *(hi-lo), fill=False, edgecolor=palette.orange,
                                       linewidth=1., linestyle="--"))
            panel.set_title(f"{title}\nlongest side {float(np.ptp(reference, axis=0).max()):.3f}",
                            color=palette.ink, fontsize=10)
        ax.annotate("", xy=(.56, .17), xytext=(.45, .17), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=palette.orange, lw=1.5))
        ax.text(.5, .02, f"{len(after)} raw points → {data['final_count']} finished points\n"
                "Next: shared spacing, XPBD and uniform resampling", transform=ax.transAxes,
                ha="center", va="bottom", color=palette.ink, fontsize=10)
    return dict(stage=key, active_curve=active.copy(), references_are_counterfactual=True,
                raw_count=len(data["raw"]), final_count=data["final_count"],
                base_radius=1., theme=palette.name)


def _previews(raw_ax, road_ax, sample, palette):
    from matplotlib.collections import PolyCollection
    raw = sample["raw"]
    raw_ax.plot(*_closed(raw).T, color=palette.blue, lw=1.2)
    raw_ax.scatter(*raw.T, s=3, color=palette.blue)
    raw_ax.set_title(f"Actual raw · {len(raw)} points", fontsize=10, color=palette.ink)
    inner, outer, center = (sample[key] for key in ("inner", "outer", "center"))
    cells = np.stack((inner, outer, np.roll(outer, -1, axis=0),
                      np.roll(inner, -1, axis=0)), axis=1)
    road_ax.add_collection(PolyCollection(cells, facecolors=palette.road, edgecolors="none"))
    for points in (inner, outer):
        road_ax.plot(*_closed(points).T, color=palette.muted, lw=.7)
    road_ax.plot(*_closed(center).T, color=palette.blue, lw=1.1)
    road_ax.scatter(*center.T, s=5, color=palette.blue)
    status = "valid" if sample["valid"] else "INVALID"
    road_ax.set_title(f"Finished road · seed {sample['seed']}\n{sample['count']} points · {status} · fail=0x{sample['fail']:x}",
                      fontsize=10, color=palette.green if sample["valid"] else palette.red)
    for ax in (raw_ax, road_ax):
        _style(ax, palette, output=True, small=True)
        _limits(ax, [raw, inner, outer])
    # Both previews use the same output coordinates; avoid repeating x labels
    # in the narrow gap above the finished-road status.
    raw_ax.set_xlabel("")
    raw_ax.tick_params(labelbottom=False)


def _writer():
    import imageio_ffmpeg
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMpegWriter(fps=6, codec="libx264", bitrate=1100,
                       extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def render(out_dir, sample, movies=True, theme="light", writer_factory=None):
    """Write a six-step themed poster/movie using the same captured geometry."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz.racing_theme import apply_plot_theme, plotting_theme, save_figure, video_canvas_size

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = capture(sample)
    stages = construction_stages(sample)
    footer = ("Construction explanation, not execution timing. Radius/angle reference views use the same seeded draws.\n"
              "No point morph is implied; the shared spacing/relaxation guide explains the finished road.")
    with plotting_theme(theme) as palette:
        fig = plt.figure(figsize=video_canvas_size(12, dpi=120))
        grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1), left=.045, right=.98,
                               bottom=.15, top=.79, wspace=.20, hspace=.25)
        ax = fig.add_subplot(grid[:, :2])
        _previews(fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2]), sample, palette)
        heading = fig.text(.34, .955, "", ha="center", color=palette.ink, fontsize=17)
        explanation = fig.text(.34, .875, "", ha="center", va="center", color=palette.ink, fontsize=11)
        fig.text(.5, .035, footer, ha="center", color=palette.ink, fontsize=9)

        def show(index, progress):
            stage = stages[index]
            heading.set_text(f"Polar · {index+1}/{len(stages)} · {stage['title']}")
            explanation.set_text(textwrap.fill(stage["explanation"], width=80))
            ax.clear()
            draw_step(ax, sample, index, progress, theme)
            apply_plot_theme(fig, theme)

        try:
            if movies:
                writer = (writer_factory or _writer)()
                with writer.saving(fig, str(out_dir / "generator-polar-construction.mp4"), dpi=120):
                    for i in range(len(stages)):
                        for progress in [*np.linspace(.03, 1., 18), *([1.]*6)]:
                            show(i, progress)
                            writer.grab_frame()
            # A landscape movie poster avoids a player-size jump when metadata
            # replaces the tall six-step diagram. This is a real movie frame.
            show(2, 1.)
            save_figure(fig, out_dir / "generator-polar-construction-poster.png", theme=theme, dpi=120)
        finally:
            plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 13.2))
        try:
            for i, (ax, stage) in enumerate(zip(axes.flat, stages)):
                draw_step(ax, sample, i, 1., theme)
                ax.set_title(textwrap.fill(f"{i+1}. {stage['title']}", width=40),
                             color=palette.ink, fontsize=12, pad=12)
            fig.suptitle(f"Polar · slots, radii, angular jitter and sampling\nseed {sample['seed']} · actual geometry",
                         color=palette.ink, fontsize=17, y=.985)
            fig.text(.5, .016, footer, ha="center", color=palette.ink, fontsize=10)
            fig.subplots_adjust(left=.08, right=.98, top=.92, bottom=.08, hspace=.42, wspace=.23)
            save_figure(fig, out_dir / "generator-polar-construction.png", theme=theme, dpi=140)
        finally:
            plt.close(fig)
    record = {key: sample[key] for key in ("seed", "epoch", "env", "config", "count", "valid", "fail")}
    record.update(raw_finite=bool(np.isfinite(sample["raw"]).all()),
                  real_geometry_finite=all(bool(np.isfinite(sample[k]).all())
                                           for k in ("center", "inner", "outer")))
    return dict(kind="progressive construction drawing of recorded geometry", sample=record,
                stage_titles=[stage["title"] for stage in stages], stages=len(stages),
                frames=144 if movies else 0, duration_seconds=24 if movies else 0,
                geometry_interpolated=False, elapsed_runtime=False, theme=theme,
                output="generator-polar-construction."+("mp4" if movies else "png"),
                poster="generator-polar-construction-poster.png", aspect_ratio="16:9",
                frame_size=[round(x*120) for x in video_canvas_size(12, dpi=120)],
                poster_frame_size=[round(x*120) for x in video_canvas_size(12, dpi=120)],
                raw_count=len(data["raw"]), final_count=data["final_count"],
                references_are_counterfactual=True, controls_replay_equal=data["controls_replay_equal"],
                raw_replay_equal=data["raw_replay_equal"], geometry_digest=data["signature"])
