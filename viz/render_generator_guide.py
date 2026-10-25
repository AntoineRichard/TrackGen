"""Render the six-family generator guide from current CPU output.

    MPLCONFIGDIR=/private/tmp/trackgen-mpl WARP_CACHE_PATH=/private/tmp/trackgen-warp-cache \
        python -m viz.render_generator_guide --out-dir /private/tmp/generator-guide

Matplotlib and imageio-ffmpeg are optional maintenance tools, never Sphinx dependencies.
Private scratch reads below belong only to this renderer. Construction reveals show
recorded buffers, not simulated time; the repulsive movie records actual loop states.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.animation import FFMpegWriter

from track_gen import TrackGenConfig, TrackGenerator

FAMILIES = ("bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive")
BLUE, ORANGE, INK = "#166baf", "#d26a18", "#243344"
ROAD, RED, GREEN = "#e3e9ee", "#b43a36", "#19704b"

# Every Params field has an explanation. Grouped comparisons change only the named
# bounds together (the labels explicitly say so); unrelated config stays identical.
SETTING_COVERAGE = {
    "bezier": dict.fromkeys(("min_num_points", "max_num_points", "min_point_distance",
                            "rad", "edgy", "handle_clamp_frac"), "generator-bezier-controls.png"),
    "hull": dict.fromkeys(("min_num_points", "max_num_points", "min_point_distance",
                          "hull_displacement"), "generator-hull-controls.png"),
    "polar": dict.fromkeys(("polar_num_knots", "polar_radial_jitter", "polar_angular_jitter"),
                           "generator-polar-controls.png"),
    "voronoi": dict.fromkeys(("voronoi_num_sites", "voronoi_control_points",
                             "voronoi_radial_variation", "voronoi_angular_jitter"),
                            "generator-voronoi-controls.png"),
    "checkpoint": dict.fromkeys(("checkpoint_count", "checkpoint_radius_min_frac",
                                "checkpoint_angle_jitter", "checkpoint_turn_rate",
                                "checkpoint_steer_gain", "checkpoint_lookahead_frac"),
                               "generator-checkpoint-controls.png"),
    "repulsive": dict.fromkeys(("repulsive_grow_mult_min", "repulsive_grow_mult_max",
                               "repulsive_domain_frac", "repulsive_domain_init_ratio",
                               "repulsive_obstacle_count_min", "repulsive_obstacle_count_max",
                               "repulsive_obstacle_radius_min_frac", "repulsive_obstacle_radius_max_frac",
                               "repulsive_ratchet_rate", "repulsive_deactivate_obstacles"),
                              "generator-repulsive-controls.png"),
}
for _name in FAMILIES:
    SETTING_COVERAGE[_name]["size"] = "configuration/tuning: tuning-size"
    if _name in ("bezier", "hull", "polar", "voronoi"):
        SETTING_COVERAGE[_name]["num_points_per_segment"] = "generator-" + _name + "-sampling.png"
SETTING_COVERAGE["voronoi"]["voronoi_site_layout"] = "generator-voronoi-layouts.png"
SETTING_COVERAGE["checkpoint"].update({
    "checkpoint_best_of_k": "generator-checkpoint-candidates.png",
    "checkpoint_clip_fallback": "generator-checkpoint-fallback.png",
})
for _field in ("repulsive_alpha", "repulsive_beta", "repulsive_tau", "repulsive_w_len",
               "repulsive_stages", "repulsive_settle_iters", "repulsive_resample_every",
               "repulsive_stall_window", "repulsive_stall_area_tol"):
    SETTING_COVERAGE["repulsive"][_field] = "generators/repulsive: Growth mechanics and schedule"
for _field in ("repulsive_alpha", "repulsive_beta", "repulsive_tau", "repulsive_w_len"):
    SETTING_COVERAGE["repulsive"][_field] = "generator-repulsive-numerics.png"
for _name, _fields in {
    "bezier": ("min_num_points", "max_num_points", "min_point_distance"),
    "hull": ("min_point_distance",),
    "voronoi": ("voronoi_num_sites",),
    "checkpoint": ("checkpoint_count", "checkpoint_steer_gain", "checkpoint_lookahead_frac"),
    "repulsive": ("repulsive_obstacle_count_min", "repulsive_obstacle_count_max",
                  "repulsive_obstacle_radius_min_frac", "repulsive_obstacle_radius_max_frac",
                  "repulsive_ratchet_rate"),
}.items():
    for _field in _fields:
        SETTING_COVERAGE[_name][_field] = f"generator-{_name}-controls-more.png"
SETTING_COVERAGE["repulsive"]["repulsive_deactivate_obstacles"] = "generator-repulsive-controls-obstacles.png"


def curve_delta(a, b) -> float:
    """Maximum pointwise displacement of equally resampled raw loops."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1).max())


def _generator_scratch(gen):
    # Configs use the generator binding's built Stage; explicit PipelineSpecs keep
    # stage scratch directly on Program. Both are read-only renderer observations.
    built = gen.program._gen_scratch
    if hasattr(built, "scratch"):
        return {k.rsplit("/", 1)[-1]: v for k, v in built.scratch.items()}
    return gen.program._scratch["generator"]


def _construction(gen, cfg, e):
    """Read the buffers that were actually used, with each buffer's real prefix."""
    sc = _generator_scratch(gen)
    E = cfg.num_envs

    def rows(name):
        return sc[name].numpy().reshape(E, -1, 2)[e].copy()

    name = cfg.generator
    out = {}
    if name == "bezier":
        n = int(sc["gen_count"].numpy()[e])
        out = {"anchors": rows("gen_ordered")[:n], "dense": rows("gen_dense")}
        out["tangents"] = rows("gen_tan")[:n]
        out["handle_scales"] = sc["gen_scale"].numpy().reshape(E, -1)[e, :n].copy()
        out["crossings"] = int(sc["gen_crossers"].numpy()[e])
    elif name == "hull":
        n = int(sc["count"].numpy()[e])
        out = {"anchors": rows("sorted")[:n], "augmented": rows("aug")[:2*n],
               "dense": rows("dense"), "crossings": int(sc["crossers"].numpy()[e])}
    elif name == "polar":
        out = {"anchors": rows("controls"), "dense": rows("dense")}
    elif name == "voronoi":
        out = {"sites": rows("sites"), "anchors": rows("selected"),
               "augmented": rows("aug"), "dense": rows("dense"),
               "crossings": int(sc["crossers"].numpy()[e])}
    elif name == "checkpoint":
        K, C, N = cfg.checkpoint_best_of_k, cfg.checkpoint_count, cfg.num_points
        out = {"candidates": rows("cand").reshape(K, N, 2),
               "walks": rows("path").reshape(K, N, 2),
               "checkpoints": rows("checkpoints").reshape(K, C, 2)}
        out["candidate_crossings"] = sc["cross"].numpy().reshape(E, K)[e].copy() if K > 1 else None
    elif name == "repulsive":
        out = {"obstacles": rows("obs_pts"),
               "reached": bool(sc["reached"].numpy()[e]),
               "frozen": bool(sc["frozen"].numpy()[e])}
    return out


def sample_batch(name, *, seeds, overrides=None, return_generator=False):
    """Generate an explicit seed vector at epoch zero; preserve failed geometry honestly."""
    seeds = [int(s) for s in seeds]
    values = dict(generator=name, device="cpu", num_envs=len(seeds), on_invalid="keep")
    values.update(overrides or {})
    cfg = TrackGenConfig(**values)
    gen = TrackGenerator(cfg, seeds=wp.array(seeds, dtype=wp.int32, device="cpu"))
    track = gen.generate()
    raw = gen.program.tap("raw.xy").numpy().reshape(len(seeds), cfg.num_points, 2)
    count, valid, fail = track.count.numpy(), track.valid.numpy(), track.fail_mask.numpy()
    arrays = {k: getattr(track, k).numpy().reshape(len(seeds), cfg.N_max, 3)[..., :2]
              for k in ("center", "inner", "outer")}
    output = []
    for e, seed in enumerate(seeds):
        n = int(count[e])
        sample = dict(seed=seed, epoch=0, env=e, config=dataclasses.asdict(cfg),
                      count=n, valid=bool(valid[e]), fail=int(fail[e]), raw=raw[e].copy(),
                      construction=_construction(gen, cfg, e))
        for key, arr in arrays.items():
            sample[key] = arr[e, :n].copy()
        output.append(sample)
    return (output, gen) if return_generator else output


def _record(sample):
    record = {k: sample[k] for k in ("seed", "epoch", "env", "config", "count", "valid", "fail")}
    record["raw_finite"] = bool(np.isfinite(sample["raw"]).all())
    record["real_geometry_finite"] = all(bool(np.isfinite(sample[k]).all()) for k in ("center","inner","outer"))
    if "crossings" in sample["construction"]:
        record["smooth_raw_crossings"] = sample["construction"]["crossings"]
        record["polygon_fallback_used"] = record["smooth_raw_crossings"] > 0
    return record


def _close(points):
    return np.vstack((points, points[:1]))


def _line(ax, pts, **kwargs):
    pts = np.asarray(pts)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > 1:
        ax.plot(*_close(pts).T, **kwargs)


def _road(ax, sample, *, raw=False):
    c, inner, outer = (sample[k] for k in ("center", "inner", "outer"))
    if len(c) >= 3 and all(np.isfinite(p).all() for p in (c, inner, outer)):
        # Independent quads preserve the band seam; no fill-rule hole tricks.
        from matplotlib.collections import PolyCollection
        quads = np.stack((inner, outer, np.roll(outer, -1, axis=0),
                          np.roll(inner, -1, axis=0)), axis=1)
        ax.add_collection(PolyCollection(quads, facecolors=ROAD, edgecolors="none"))
        _line(ax, inner, color="#718394", lw=.7)
        _line(ax, outer, color="#718394", lw=.7)
        _line(ax, c, color=ORANGE if raw else BLUE, lw=1.1, ls="--" if raw else "-")
    if raw:
        _line(ax, sample["raw"], color=BLUE, lw=1.8)


def _limits(ax, points, *, span=None):
    finite = np.concatenate([p[np.isfinite(p).all(axis=1)] for p in points])
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    mid = (lo + hi) * .5
    span = max(float(np.max(hi-lo)), .2) if span is None else span
    ax.set_xlim(mid[0]-.58*span, mid[0]+.58*span)
    ax.set_ylim(mid[1]-.58*span, mid[1]+.58*span)


def _style(ax, *, ticks=False):
    ax.set_aspect("equal")
    if ticks:
        ax.tick_params(labelsize=9, colors="#6b7785")
        ax.set_xlabel("x [coordinate units]", fontsize=10)
        ax.set_ylabel("y [coordinate units]", fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def _save(fig, path):
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def _label(sample):
    return f"seed {sample['seed']} · n={sample['count']} · " + (
        "valid" if sample["valid"] else f"FAILED (bits {sample['fail']})")


def render_examples(out_dir, examples):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    common_span = max(float(np.ptp(s[k],axis=0).max()) for s in examples for k in ("raw","outer"))
    for col, sample in enumerate(examples):
        for row in range(2):
            ax = axes[row, col]
            if row == 0:
                _line(ax, sample["raw"], color=BLUE, lw=2)
            else:
                _road(ax, sample)
            _limits(ax, [sample["raw"], sample["outer"]], span=common_span)
            _style(ax)
            label = (f"seed {sample['seed']} · raw N={len(sample['raw'])} · final valid"
                     if row == 0 else _label(sample))
            if row == 0 and sample["construction"].get("crossings", 0) > 0:
                label += "\npolygon fallback selected"
            ax.set_title(("Raw centerline" if row == 0 else "Finished road") + "\n" + label, fontsize=11)
    name = examples[0]["config"]["generator"]
    fig.suptitle(f"{name.capitalize()} · three selected seeds", fontsize=18, color=INK)
    fig.text(.5, .01, "All panels share one scale. Coordinates are arbitrary units; half-width = 0.1. Selected valid examples are not a yield estimate.",
             ha="center", fontsize=10, color=INK)
    fig.tight_layout(rect=(0,.035,1,.95))
    _save(fig, out_dir / f"generator-{name}-examples.png")


def render_gallery(out_dir, selections):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # All families have a shared viewport span, so the longer repulsive loop stays honest.
    for ax, name in zip(axes.flat, FAMILIES):
        sample = selections[name][0]
        _road(ax, sample)
        _limits(ax, [sample["outer"]], span=2.6)
        _style(ax)
        ax.set_title(name.capitalize(), fontsize=18, color=INK, fontweight="bold")
        ax.text(.5, -.02, _label(sample), transform=ax.transAxes, ha="center", fontsize=10)
        x0,y0 = ax.get_xlim()[0]+.15,ax.get_ylim()[0]+.18
        ax.plot([x0,x0+.5],[y0,y0],color=INK,lw=1.5)
        ax.text(x0+.25,y0+.04,"0.5 units",fontsize=9,ha="center",color=INK)
    fig.text(.5, .01, "Current CPU output · default geometry · equal coordinate scale · selected valid examples, not a yield comparison",
             ha="center", fontsize=11, color=INK)
    fig.tight_layout(rect=(0,.035,1,1))
    _save(fig, out_dir / "generators-gallery.png")


def _cases():
    """(short field label, list of overrides); empty overrides explicitly include defaults."""
    return {
        "bezier": [
            ("rad", [{"rad": .1}, {}, {"rad": .8}]),
            ("handle_clamp_frac", [{"handle_clamp_frac": .1}, {}, {"handle_clamp_frac": .8}]),
            ("edgy", [{"edgy": -2.}, {}, {"edgy": 2.}]),
            ("min_num_points", [{"min_num_points": 4}, {}, {"min_num_points": 13}]),
            ("max_num_points", [{"max_num_points": 9}, {}, {"max_num_points": 18}]),
            ("min_point_distance", [{"min_point_distance": .025}, {}, {"min_point_distance": .1}]),
        ],
        "hull": [
            ("hull_displacement", [{"hull_displacement": 0.}, {}, {"hull_displacement": .6}]),
            ("min_num_points", [{"min_num_points": 4}, {}, {"min_num_points": 13}]),
            ("max_num_points", [{"max_num_points": 9}, {}, {"max_num_points": 18}]),
            ("min_point_distance", [{"min_point_distance": .025}, {}, {"min_point_distance": .1}]),
        ],
        "polar": [
            ("polar_radial_jitter", [{"polar_radial_jitter": .1}, {}, {"polar_radial_jitter": .85}]),
            ("polar_angular_jitter", [{"polar_angular_jitter": 0.}, {}, {"polar_angular_jitter": .45}]),
            ("polar_num_knots", [{"polar_num_knots": 6}, {}, {"polar_num_knots": 20}]),
        ],
        "voronoi": [
            ("voronoi_radial_variation", [{"voronoi_radial_variation": 0.}, {}, {"voronoi_radial_variation": .85}]),
            ("voronoi_angular_jitter", [{"voronoi_angular_jitter": 0.}, {}, {"voronoi_angular_jitter": .5}]),
            ("voronoi_control_points", [{"voronoi_control_points": 8}, {}, {"voronoi_control_points": 28}]),
            ("voronoi_num_sites", [{"voronoi_num_sites": 64}, {}, {"voronoi_num_sites": 512}]),
        ],
        "checkpoint": [
            ("checkpoint_radius_min_frac", [{"checkpoint_radius_min_frac": .1}, {}, {"checkpoint_radius_min_frac": .85}]),
            ("checkpoint_angle_jitter", [{"checkpoint_angle_jitter": 0.}, {}, {"checkpoint_angle_jitter": .9}]),
            ("checkpoint_turn_rate", [{"checkpoint_turn_rate": .08}, {}, {"checkpoint_turn_rate": 1.2}]),
            ("checkpoint_count", [{"checkpoint_count": 6}, {}, {"checkpoint_count": 20}]),
            ("checkpoint_steer_gain", [{"checkpoint_steer_gain": .2}, {}, {"checkpoint_steer_gain": 1.}]),
            ("checkpoint_lookahead_frac", [{"checkpoint_lookahead_frac": .05}, {}, {"checkpoint_lookahead_frac": .35}]),
        ],
        "repulsive": [
            ("grow range", [{"repulsive_grow_mult_min": 2.5, "repulsive_grow_mult_max": 3.5}, {},
                            {"repulsive_grow_mult_min": 6., "repulsive_grow_mult_max": 7.}]),
            ("repulsive_domain_frac", [{"repulsive_domain_frac": .25}, {}, {"repulsive_domain_frac": .5}]),
            ("repulsive_domain_init_ratio", [{"repulsive_domain_init_ratio": 2.5}, {}, {"repulsive_domain_init_ratio": 6.}]),
            ("obstacle count range", [{"repulsive_obstacle_count_min": 2, "repulsive_obstacle_count_max": 4}, {},
                                      {"repulsive_obstacle_count_min": 16, "repulsive_obstacle_count_max": 20}]),
            ("obstacle radius range", [{"repulsive_obstacle_radius_min_frac": .005, "repulsive_obstacle_radius_max_frac": .01}, {},
                                       {"repulsive_obstacle_radius_min_frac": .06, "repulsive_obstacle_radius_max_frac": .08}]),
            ("repulsive_ratchet_rate", [{"repulsive_ratchet_rate": .006}, {}, {"repulsive_ratchet_rate": .025}]),
            ("repulsive_deactivate_obstacles", [{}, {"repulsive_deactivate_obstacles": False}]),
        ],
    }


def _value_label(name, overrides, cfg):
    if "range" in name:
        stem = {"grow range": "repulsive_grow_mult_", "obstacle count range": "repulsive_obstacle_count_",
                "obstacle radius range": "repulsive_obstacle_radius_"}[name]
        keys = (stem + "min", stem + "max") if name != "obstacle radius range" else (stem+"min_frac", stem+"max_frac")
        value = f"[{cfg[keys[0]]}, {cfg[keys[1]]}]"
    else:
        value = str(cfg[name])
    return name.replace("repulsive_", "").replace("checkpoint_", "").replace("voronoi_", "") + " = " + value + (" (default)" if not overrides else "")


def render_comparisons(out_dir, name, *, seed=23):
    records, panels = [], []
    cases = _cases()[name]
    if name in ("bezier", "hull", "polar", "voronoi"):
        cases += [("num_points_per_segment", [{"num_points_per_segment": 4}, {}, {"num_points_per_segment": 60}])]
    for field, values in cases:
        seed_search = [seed, 7, 41, 59]
        for candidate_seed in seed_search:
            samples = [sample_batch(name, seeds=[candidate_seed], overrides=v)[0] for v in values]
            delta = max(curve_delta(samples[0]["raw"], s["raw"]) for s in samples[1:])
            if delta > 1e-6:
                break
        else:
            raise AssertionError(f"vacuous {name}.{field} comparison over {seed_search}")
        records.append(dict(field=field, includes_default={} in values,
                            max_raw_delta_from_first=delta, seed_search=seed_search,
                            samples=[_record(s) for s in samples]))
        panels.append((field, values, samples))
    # Keep each published image at most three rows high at a ~736 px content width.
    regular = panels[:-1] if name in ("bezier", "hull", "polar", "voronoi") else panels
    groups = [(regular[:3], "controls")]
    if len(regular) > 3:
        groups.append((regular[3:], "controls-more"))
    if len(regular) > 6:
        groups[-1] = (regular[3:6], "controls-more")
        groups.append((regular[6:], "controls-obstacles"))
    if len(panels) > len(regular):
        groups.append(([panels[-1]], "sampling"))
    for rows, suffix in groups:
        fig, axes = plt.subplots(len(rows), 3, figsize=(12, max(4.8,3.8*len(rows))), squeeze=False)
        for row, (field, values, samples) in zip(axes, rows):
            points = [p for s in samples for p in (s["raw"], s["outer"])]
            if name == "repulsive":
                points += [s["construction"]["obstacles"] for s in samples]
            finite = np.concatenate([p[np.isfinite(p).all(axis=1)] for p in points])
            lo, hi = finite.min(axis=0), finite.max(axis=0)
            mid, span = (lo+hi)/2, max(np.ptp(finite, axis=0).max(), .2)
            for ax, overrides, sample in zip(row, values, samples):
                _road(ax, sample, raw=True)
                if name == "repulsive":
                    obs = sample["construction"]["obstacles"]
                    ax.scatter(*obs.T, s=2, color="#839485", alpha=.55, zorder=0)
                if field == "num_points_per_segment":
                    dense = sample["construction"]["dense"]
                    # Bezier/Hull construction coordinates already share output units.
                    if name in ("bezier", "hull"):
                        ax.scatter(*dense.T, s=5, color=BLUE, alpha=.6)
                ax.set_xlim(mid[0]-.58*span, mid[0]+.58*span)
                ax.set_ylim(mid[1]-.58*span, mid[1]+.58*span)
                _style(ax)
                ax.set_title(_value_label(field, overrides, sample["config"]), fontsize=11)
                ax.text(.5, -.025, _label(sample), transform=ax.transAxes, ha="center", fontsize=10,
                        color=GREEN if sample["valid"] else RED)
            for ax in row[len(samples):]:
                ax.set_visible(False)
        fig.suptitle(f"{name.capitalize()} · same seed, one control per row", fontsize=17, color=INK)
        fig.text(.5, .008, "Blue: generator centerline   Orange dashed: final centerline   Grey: road   n: final count; coordinate units.\nEqual axes within each row. Failed geometry is retained explicitly (on_invalid='keep'). Selected examples do not measure success rate.",
                 ha="center", fontsize=10, color=INK)
        fig.tight_layout(rect=(0,.14 if len(rows)==1 else .055,1,.95))
        _save(fig, out_dir / f"generator-{name}-{suffix}.png")
    return records


def render_construction(out_dir, sample, *, movies=True):
    name = sample["config"]["generator"]
    sc = sample["construction"]
    if name == "repulsive":
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    a = sc["checkpoints"][0] if name == "checkpoint" else sc["anchors"]
    ax = axes[0]
    if "sites" in sc:
        ax.scatter(*sc["sites"].T, s=6, color="#a7b3bf", label="sites")
    if name == "polar":
        for p in a:
            ax.plot([0,p[0]], [0,p[1]], color="#b8c7d4", lw=.7)
    _line(ax, a, color="#8795a1", lw=1)
    ax.scatter(*a.T, s=28, color=BLUE, zorder=3)
    if name == "bezier":
        cfg = sample["config"]
        chords = np.linalg.norm(np.roll(a,-1,axis=0)-a,axis=1)
        for i,p in enumerate(a):
            h = min(cfg["rad"]*chords[i], cfg["handle_clamp_frac"]*sc["handle_scales"][i])
            q = p + h*sc["tangents"][i]
            ax.plot([p[0],q[0]],[p[1],q[1]], color=ORANGE, lw=1.5)
            ax.scatter(*q, color=ORANGE, s=12)
    if name == "hull":
        aug = sc["augmented"]
        _line(ax, aug, color=ORANGE, lw=1)
        ax.scatter(*aug[1::2].T, s=18, color=ORANGE)
    if name == "checkpoint":
        axes[0].plot(*sc["walks"][0].T, color=ORANGE, lw=1)
    _limits(ax, [a] + ([sc["sites"]] if "sites" in sc else []) +
            ([sc["walks"][0]] if name == "checkpoint" else []))
    ax.set_title({"bezier":"Sampled corners + outgoing handles", "hull":"Sorted anchors + displaced midpoints",
                  "polar":"Sampled radial / angular knots", "voronoi":"Sites + selected anchors",
                  "checkpoint":"Candidate 0: checkpoints + open walk"}[name], fontsize=11)
    _line(axes[1], sample["raw"], color=BLUE, lw=2)
    crossings = sc.get("crossings",0)
    axes[1].set_title("Selected generator centerline\n" +
                      (f"polygon fallback: {crossings} smooth crossings" if crossings else
                       "after extent normalization, if used"), fontsize=11)
    _road(axes[2], sample)
    axes[2].set_title("Shared tail: resample → relax → road\n" + _label(sample), fontsize=11)
    for ax in axes[1:]:
        _limits(ax, [sample["raw"], sample["outer"]])
    for ax in axes:
        _style(ax)
    fig.suptitle(name.capitalize() + " construction · actual buffers, illustrative stage reveal", fontsize=16)
    fig.text(.5,.015,"Left panel uses construction coordinates; the last two use final coordinate units. This is a stage diagram, not a solver-time trace.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.92))
    if movies:
        artists = [list(ax.lines)+list(ax.collections) for ax in axes]
        line_data = {line: (np.array(line.get_xdata()), np.array(line.get_ydata()))
                     for ax in axes for line in ax.lines}
        from matplotlib.collections import PathCollection
        scatter_data = {artist: np.array(artist.get_offsets()) for ax in axes
                        for artist in ax.collections if isinstance(artist, PathCollection)}
        writer = _writer()
        with writer.saving(fig, str(out_dir / f"generator-{name}-construction.mp4"), dpi=120):
            for stage in range(3):
                for j, group in enumerate(artists):
                    for art in group:
                        art.set_visible(j <= stage)
                for frame in range(12):
                    # Reveal prefixes of recorded curves/anchors, never synthesize a
                    # shape between stages or imply these are solver iterations.
                    for art in artists[stage]:
                        if art in line_data:
                            x,y = line_data[art]
                            n = max(1, int(np.ceil(len(x)*(frame+1)/8)))
                            art.set_data(x[:n],y[:n])
                        elif art in scatter_data:
                            points = scatter_data[art]
                            n = max(1,int(np.ceil(len(points)*(frame+1)/8)))
                            art.set_offsets(points[:n])
                    writer.grab_frame()
        for group in artists:
            for art in group:
                art.set_visible(True)
    _save(fig, out_dir / f"generator-{name}-construction.png")


def _writer():
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMpegWriter(fps=6, codec="libx264", bitrate=1100,
                       extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def capture_repulsive(sample):
    """Observe the unmodified CPU growth loop; no interpolation, shortened budget or reimplementation."""
    from track_gen.exec.context import LaunchContext
    from track_gen._src.warp_generate_repulsive import _stage_schedule

    overrides = {k:v for k,v in sample["config"].items()
                 if k not in ("generator","num_envs","device","on_invalid")}
    batch, gen = sample_batch("repulsive", seeds=[sample["seed"]], overrides=overrides,
                              return_generator=True)
    cfg, sc = batch[0]["config"], _generator_scratch(gen)
    stages = cfg["repulsive_stages"]
    n_ratchet = int(np.ceil(np.log(cfg["repulsive_grow_mult_max"]) / np.log1p(cfg["repulsive_ratchet_rate"])))
    # Read the exact schedule from the source's constants, not an assumed subdivision ratio.
    from track_gen._src.warp_generate_repulsive import _SUBDIV_RATIO
    starts = _stage_schedule(stages, cfg["repulsive_ratchet_rate"], _SUBDIV_RATIO, n_ratchet)
    frames = []
    base = LaunchContext.current("cpu")

    class Observer:
        def __getattr__(self, key):
            return getattr(base, key)

        def bounded_loop(self, n_iters, body, **kwargs):
            def observed(ctx, it):
                if it == 0:
                    snapshot(0, stages[0])
                body(ctx, it)
                stage = max(i for i, start in enumerate(starts) if start <= it)
                if (it+1) % 4 == 0 or it+1 == n_iters or it in starts:
                    snapshot(it+1, stages[stage])
            base.bounded_loop(n_iters, observed, **kwargs)

    def snapshot(iteration, n):
        frames.append(dict(iteration=iteration, n=n, xy=sc["center"].numpy()[:n].copy(),
                           reached=bool(sc["reached"].numpy()[0]), frozen=bool(sc["frozen"].numpy()[0]),
                           target=float(sc["L_target"].numpy()[0])))

    expected = gen.program.tap("raw.xy").numpy().copy()
    if not np.array_equal(expected, sample["raw"]):
        raise AssertionError("capture configuration does not reproduce the selected sample")
    gen.program._gen_scratch.enqueue(Observer())
    actual = gen.program.tap("raw.xy").numpy()
    matches = bool(np.array_equal(expected, actual))
    if not matches:
        raise AssertionError("observing the growth changed the final generator output")
    frames.append(dict(iteration=frames[-1]["iteration"], n=len(actual), xy=actual.copy(),
                       reached=frames[-1]["reached"], frozen=frames[-1]["frozen"],
                       target=frames[-1]["target"], final=True))
    return frames, matches


def render_growth(out_dir, sample, *, movies=True):
    frames, matches = capture_repulsive(sample)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    obs = sample["construction"]["obstacles"]
    ax.scatter(*obs.T, color="#829582", s=5, alpha=.65)
    line, = ax.plot([], [], color=BLUE, lw=2)
    label = ax.text(.02, .02, "", transform=ax.transAxes, fontsize=11, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    _limits(ax, [obs] + [f["xy"] for f in frames])
    _style(ax, ticks=True)
    ax.set_title("Repulsive · captured growth states\nseed " + str(sample["seed"]) + " · fixed viewpoint", fontsize=15)
    fig.text(.5,.012,"Actual CPU states sampled every 4 iterations; time between frames is not runtime.\nGreen dots: sampled domain / obstacle rings. Validity is evaluated after the shared tail.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.055,1,1))

    def draw(frame):
        line.set_data(*_close(frame["xy"]).T)
        label.set_text(f"iteration {frame['iteration']} · N={frame['n']}\n"
                       f"target perimeter {frame['target']:.2f} · reached={frame['reached']} · frozen={frame['frozen']}" +
                       ("\nfinal generator resample" if frame.get("final") else ""))
    if movies:
        writer = _writer()
        with writer.saving(fig, str(out_dir / "generator-repulsive-growth.mp4"), dpi=120):
            for frame in frames:
                draw(frame)
                writer.grab_frame()
            for _ in range(12):
                writer.grab_frame()
    draw(frames[-1])
    _save(fig, out_dir / "generator-repulsive-growth.png")
    return dict(kind="captured solver states", iterations=sorted({f["iteration"] for f in frames}),
                seed=sample["seed"], final_matches_generator=matches,
                frame_count=len(frames), output="generator-repulsive-growth." + ("mp4" if movies else "png"))


def render_specials(out_dir):
    records = {}
    samples = [sample_batch("voronoi", seeds=[23], overrides={"voronoi_site_layout": layout})[0]
               for layout in ("ring", "void_ring", "clustered", "mixed")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, s in zip(axes.flat, samples):
        sc = s["construction"]
        ax.scatter(*sc["sites"].T, s=7, color="#a4b0ba")
        _line(ax, sc["anchors"], color=BLUE, lw=1.5)
        ax.scatter(*sc["anchors"].T, s=22, color=BLUE)
        all_sites = np.concatenate([s["construction"]["sites"] for s in samples])
        _limits(ax, [all_sites])
        _style(ax)
        ax.set_title(s["config"]["voronoi_site_layout"], fontsize=15)
    fig.suptitle("Voronoi site layouts · same seed 23 · 256 sites / 18 anchors",fontsize=16)
    fig.text(.5,.01,"Actual construction buffers before extent normalization. Sites grey; selected anchor cycle blue.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.95))
    _save(fig, out_dir / "generator-voronoi-layouts.png")
    records["layouts"] = [_record(s) for s in samples]

    sample = sample_batch("checkpoint", seeds=[23])[0]
    sc = sample["construction"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    best = int(np.argmin(sc["candidate_crossings"]))
    for k, ax in enumerate(axes.flat):
        pts = sc["candidates"][k]
        _line(ax, pts, color=BLUE if k == best else "#91a1b0", lw=2)
        _limits(ax, list(sc["candidates"]))
        _style(ax)
        ax.set_title(f"Candidate {k}\n{sc['candidate_crossings'][k]} proper crossings" + ("\nselected (first minimum)" if k == best else ""),fontsize=12)
    fig.suptitle("Checkpoint · best-of-4 selects among decorrelated candidates",fontsize=15)
    fig.text(.5,.01,"seed 23 · actual closed candidates before extent normalization · minimum crossing count is not final road validity",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.9))
    _save(fig, out_dir / "generator-checkpoint-candidates.png")
    records["candidates"] = dict(sample=_record(sample), crossings=sc["candidate_crossings"].tolist(), selected=best)

    # Deliberately select a real crossing; the search is bounded and selection is recorded.
    failing = sample_batch("checkpoint", seeds=list(range(48)), overrides={"checkpoint_best_of_k": 1})
    pair = None
    for s in failing:
        # Use the existing crossing primitive on raw output, not a plot-based assertion.
        from track_gen import ops
        from track_gen.exec.context import LaunchContext
        crossings = wp.zeros(1, dtype=wp.int32, device="cpu")
        points = wp.array(s["raw"], dtype=wp.vec2f, device="cpu")
        count = wp.array([len(s["raw"])], dtype=wp.int32, device="cpu")
        ops.self_intersections(LaunchContext.current("cpu"), points, len(s["raw"]), crossings, count=count)
        if int(crossings.numpy()[0]) > 0:
            clipped = sample_batch("checkpoint", seeds=[s["seed"]], overrides={
                "checkpoint_best_of_k": 1, "checkpoint_clip_fallback": True})[0]
            pair = [s, clipped]
            break
    if pair is None:
        raise AssertionError("no real checkpoint crossing found in declared seed search")
    fig, axes = plt.subplots(1,2,figsize=(10,4.6))
    for ax, s, title in zip(axes, pair, ("clip_fallback=False (default)", "clip_fallback=True")):
        _road(ax,s,raw=True)
        _limits(ax,[p for t in pair for p in (t["raw"],t["outer"])])
        _style(ax)
        ax.set_title(title + "\n" + _label(s),fontsize=12)
    fig.suptitle("Checkpoint · first-crossing clip keeps the longer sub-loop",fontsize=16)
    fig.text(.5,.01,"best_of_k=1 held fixed · blue raw, orange final, grey road · on_invalid='keep' · selected crossing, not a frequency claim",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.93))
    _save(fig,out_dir / "generator-checkpoint-fallback.png")
    records["clip_fallback"] = dict(search_seeds=list(range(48)), samples=[_record(s) for s in pair])

    # End-state sensitivity data for numerical controls, without inventing monotonic
    # fold/quality trends or timing comparisons between unlike solver schedules.
    baseline = sample_batch("repulsive",seeds=[23])[0]
    fig,axes = plt.subplots(2,2,figsize=(10,9))
    experiments = []
    for ax,(field,value) in zip(axes.flat,(("repulsive_alpha",2.5),("repulsive_beta",5.),
                                         ("repulsive_tau",.2),("repulsive_w_len",0.))):
        alternative = sample_batch("repulsive",seeds=[23],overrides={field:value})[0]
        delta = curve_delta(baseline["raw"],alternative["raw"])
        assert delta > 1e-6
        _line(ax,baseline["raw"],color=BLUE,lw=1.8,label="default")
        _line(ax,alternative["raw"],color=ORANGE,lw=1.5,label="changed")
        _limits(ax,[baseline["raw"],alternative["raw"]])
        _style(ax)
        ax.set_title(f"{field.removeprefix('repulsive_')}: {baseline['config'][field]} → {value}\n"
                     f"raw displacement max {delta:.3f} coordinate units",fontsize=12)
        ax.text(.5,-.02,f"changed: valid={alternative['valid']} · target reached={alternative['construction']['reached']}\n"
                f"growth frozen={alternative['construction']['frozen']}",transform=ax.transAxes,ha="center",fontsize=10)
        ax.legend(loc="upper right",frameon=False,fontsize=9)
        experiments.append(dict(field=field,max_raw_delta=delta,
                                samples=[_record(baseline),_record(alternative)],
                                changed_reached=alternative["construction"]["reached"],
                                changed_frozen=alternative["construction"]["frozen"]))
    fig.suptitle("Repulsive numerical controls · same seed 23 · one setting changed",fontsize=16)
    fig.text(.5,.01,"Blue default vs orange changed: actual final generator states, before the road tail.\nAxes match within each pair. These outcomes establish sensitivity, not a monotonic shape or runtime rule.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.95))
    _save(fig,out_dir / "generator-repulsive-numerics.png")
    records["numerical_controls"] = experiments
    return records


def render(out_dir: Path, *, movies=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parents[1] / "track_gen/_src"
    source_hashes = {name: hashlib.sha256((source_dir / ("warp_generate" +
                      ("" if name == "bezier" else "_"+name) + ".py")).read_bytes()).hexdigest()
                     for name in FAMILIES}
    manifest = dict(source=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
                                warp_version=wp.__version__, platform=platform.platform(), device="cpu",
                                python_version=platform.python_version(), numpy_version=np.__version__,
                                matplotlib_version=matplotlib.__version__, generator_source_sha256=source_hashes,
                                renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                    command="python -m viz.render_generator_guide --out-dir DIR" + ("" if movies else " --no-movies"),
                    selection=dict(policy="first three valid finite rows from the listed explicit seed pool",
                                   claims_yield=False, epoch=0, coordinates="arbitrary coordinate units"),
                    families={}, animations={}, setting_coverage=SETTING_COVERAGE)
    examples = {}
    for name in FAMILIES:
        seeds = [7,23,41,59,73,97] if name != "repulsive" else [7,23,41,59]
        pool = sample_batch(name,seeds=seeds)
        selected = [s for s in pool if s["valid"] and np.isfinite(s["center"]).all()][:3]
        if len(selected) != 3:
            raise AssertionError(f"not enough valid examples for {name}: {[s['fail'] for s in pool]}")
        examples[name] = selected
        render_examples(out_dir,selected)
        render_construction(out_dir,selected[0],movies=movies)
        comparisons = render_comparisons(out_dir,name)
        manifest["families"][name] = dict(seed_pool=seeds, examples=[_record(s) for s in selected],
                                          comparisons=comparisons)
        if name != "repulsive":
            manifest["animations"][name] = dict(kind="illustrative reveal of recorded construction buffers",
                                                sample=_record(selected[0]), stages=3,
                                                output=f"generator-{name}-construction." + ("mp4" if movies else "png"))
        print(f"rendered {name}",flush=True)
    render_gallery(out_dir,examples)
    manifest["specials"] = render_specials(out_dir)
    manifest["animations"]["repulsive"] = render_growth(out_dir,examples["repulsive"][0],movies=movies)
    manifest["assets"] = {p.name: p.stat().st_size for p in sorted(out_dir.glob("generator*")) if p.suffix in (".png",".mp4")}
    (out_dir / "generators-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=Path("docs/assets/racing"))
    parser.add_argument("--no-movies",action="store_true",help="render posters only (no ffmpeg needed)")
    args = parser.parse_args()
    render(args.out_dir,movies=not args.no_movies)


if __name__ == "__main__":
    main()
