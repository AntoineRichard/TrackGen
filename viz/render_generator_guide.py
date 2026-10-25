"""Render the six-family generator guide from current CPU output.

    MPLCONFIGDIR=/private/tmp/trackgen-mpl WARP_CACHE_PATH=/private/tmp/trackgen-warp-cache \
        python -m viz.render_generator_guide --out-dir /private/tmp/generator-guide

Matplotlib and imageio-ffmpeg are optional maintenance tools, never Sphinx dependencies.
Private scratch reads below belong only to this renderer. Construction drawings explain
operations using recorded buffers, not simulated time; the repulsive movie records actual loop states.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib.animation import FFMpegWriter

from track_gen import TrackGenConfig, TrackGenerator
from track_gen._version import __version__

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
                               "repulsive_ratchet_rate"),
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
for _field in ("repulsive_alpha", "repulsive_beta", "repulsive_tau",
               "repulsive_stages", "repulsive_settle_iters", "repulsive_resample_every",
               "repulsive_stall_window", "repulsive_stall_area_tol"):
    SETTING_COVERAGE["repulsive"][_field] = "generators/repulsive: Growth mechanics and schedule"
for _field in ("repulsive_alpha", "repulsive_beta", "repulsive_tau", "repulsive_resample_every"):
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
SETTING_COVERAGE["repulsive"]["size"] = "generator-repulsive-controls-obstacles.png"


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
        out = {"sampled": rows("gen_corners")[:n], "anchors": rows("gen_ordered")[:n], "dense": rows("gen_dense")}
        out["tangents"] = rows("gen_tan")[:n]
        out["handle_scales"] = sc["gen_scale"].numpy().reshape(E, -1)[e, :n].copy()
        out["crossings"] = int(sc["gen_crossers"].numpy()[e])
    elif name == "hull":
        n = int(sc["count"].numpy()[e])
        out = {"sampled": rows("points")[:n], "anchors": rows("sorted")[:n], "augmented": rows("aug")[:2*n],
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
               "ratchet_complete": bool(sc["reached"].numpy()[e]),
               "frozen": bool(sc["frozen"].numpy()[e]),
               "force_samples": repulsive_force_samples(gen, env=e),
               "generator_valid": bool(gen.program.tap("gen.valid").numpy()[e])}
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
    if sample["config"]["generator"] == "repulsive":
        sc = sample["construction"]
        record.update(generator_valid=sc["generator_valid"], ratchet_complete=sc["ratchet_complete"], frozen=sc["frozen"])
        barriers = sc["force_samples"]
        record["barriers"] = {key: barriers[key] for key in
                              ("domain_radius", "size", "half_width", "unscaled_margin")}
        record["barriers"]["disc_count"] = len(barriers["disc_radii"])
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


def _comparison_caption(sample):
    """Keep generator fallback selection separate from the final road status."""
    caption = _label(sample)
    crossings = sample.get("construction", {}).get("crossings", sample.get("smooth_raw_crossings"))
    if crossings is not None:
        caption += (f"\npolygon fallback ({crossings} crossings)" if crossings > 0
                    else "\nsmooth curve selected")
    return caption


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
            if sample["config"]["generator"] == "repulsive":
                draw_repulsive_forces(ax, sample["construction"]["force_samples"])
                radius = sample["construction"]["force_samples"]["domain_radius"]
                ax.set_xlim(-1.1*radius, 1.1*radius)
                ax.set_ylim(-1.1*radius, 1.1*radius)
            else:
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
    # Include Repulsive's full fixed domain at the same scale as the other panels.
    for ax, name in zip(axes.flat, FAMILIES):
        sample = selections[name][0]
        _road(ax, sample)
        if name == "repulsive":
            draw_repulsive_forces(ax, sample["construction"]["force_samples"], reservation=False)
        _limits(ax, [sample["outer"]], span=3.5)
        if name == "repulsive":
            ax.set_xlim(-.58*3.5, .58*3.5)
            ax.set_ylim(-.58*3.5, .58*3.5)
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
            ("grow range", [{"repulsive_grow_mult_min": 1.5, "repulsive_grow_mult_max": 2.}, {},
                            {"repulsive_grow_mult_min": 3., "repulsive_grow_mult_max": 3.5}]),
            ("repulsive_domain_frac", [{"repulsive_domain_frac": .25}, {}, {"repulsive_domain_frac": .5}]),
            ("repulsive_domain_init_ratio", [{"repulsive_domain_init_ratio": 2.5}, {}, {"repulsive_domain_init_ratio": 6.}]),
            ("obstacle count range", [{"repulsive_obstacle_count_min": 2, "repulsive_obstacle_count_max": 4}, {},
                                      {"repulsive_obstacle_count_min": 16, "repulsive_obstacle_count_max": 20}]),
            ("obstacle radius range", [{"repulsive_obstacle_radius_min_frac": .005, "repulsive_obstacle_radius_max_frac": .01}, {},
                                       {"repulsive_obstacle_radius_min_frac": .06, "repulsive_obstacle_radius_max_frac": .08}]),
            ("repulsive_ratchet_rate", [{"repulsive_ratchet_rate": .006}, {}, {"repulsive_ratchet_rate": .025}]),
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
                            samples=[dict(_record(s), caption=_comparison_caption(s)) for s in samples]))
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
                    draw_repulsive_forces(ax, sample["construction"]["force_samples"])
                if field == "num_points_per_segment":
                    dense = sample["construction"]["dense"]
                    # Bezier/Hull construction coordinates already share output units.
                    if name in ("bezier", "hull"):
                        ax.scatter(*dense.T, s=5, color=BLUE, alpha=.6)
                ax.set_xlim(mid[0]-.58*span, mid[0]+.58*span)
                ax.set_ylim(mid[1]-.58*span, mid[1]+.58*span)
                _style(ax)
                ax.set_title(_value_label(field, overrides, sample["config"]), fontsize=11)
                ax.text(.5, -.025, _comparison_caption(sample), transform=ax.transAxes, ha="center", fontsize=10,
                        color=GREEN if sample["valid"] else RED)
            for ax in row[len(samples):]:
                ax.set_visible(False)
        fig.suptitle(f"{name.capitalize()} · same seed, one control per row", fontsize=17, color=INK)
        fig.text(.5, .008, "Blue: generator centerline   Orange dashed: final centerline   Grey: road   n: final count; coordinate units.\nEqual axes within each row. Failed geometry is retained explicitly (on_invalid='keep'). Selected examples do not measure success rate.",
                 ha="center", fontsize=10, color=INK)
        fig.tight_layout(rect=(0,.14 if len(rows)==1 else .055,1,.95))
        _save(fig, out_dir / f"generator-{name}-{suffix}.png")
    return records


def bezier_controls(sample):
    """Both handles for each cubic, derived from the recorded tangent/scale buffers."""
    sc, cfg = sample["construction"], sample["config"]
    a = sc["anchors"]
    b = np.roll(a, -1, axis=0)
    chord = np.linalg.norm(b-a, axis=1)
    h0 = np.minimum(cfg["rad"]*chord, cfg["handle_clamp_frac"]*sc["handle_scales"])
    h1 = np.minimum(cfg["rad"]*chord,
                    cfg["handle_clamp_frac"]*np.roll(sc["handle_scales"], -1))
    return np.stack((a, a+h0[:, None]*sc["tangents"],
                     b-h1[:, None]*np.roll(sc["tangents"], -1, axis=0), b), axis=1)


def construction_stages(sample):
    """A drawing storyboard; stage order is algorithm order, never elapsed runtime."""
    name = sample["config"]["generator"]
    descriptions = {
        "bezier": [
            ("Sample corners", "Reveal the actual sampled grid corners; labels retain sampling order."),
            ("Order around the centroid", "Connect the angle-sorted corners to form the anchor cycle."),
            ("Draw each cubic with two handles", "Orange: active cubic control polygon. Blue: recorded dense curve samples."),
            ("Resample and choose the raw loop", "Arc-resample the curve; a proper crossing selects the corner-polygon fallback.")],
        "hull": [
            ("Sample anchors", "Reveal the actual sampled grid anchors; labels retain sampling order."),
            ("Order around the centroid", "The base cycle is angle-sorted; it is not a geometric convex hull."),
            ("Displace inserted midpoints", "Arrows run from ordinary edge midpoints to their actual displaced positions."),
            ("Join and check the curve", "Draw recorded Catmull–Rom samples, or the augmented polygon if fallback was selected.")],
        "polar": [
            ("Keep the angular slots ordered", "Dashed rays are the unjittered angular slots; their order is fixed."),
            ("Sample radial and angular jitter", "Reveal actual knots and their radii; labels show the closed traversal order."),
            ("Join the periodic spline", "Draw the recorded dense Catmull–Rom samples through the radial knots."),
            ("Arc-resample, then normalize", "Left: recorded dense samples. Right: fixed-count arc-resampling and XY normalization.")],
        "voronoi": [
            ("Sample the site field", "Reveal actual sampled sites; a Voronoi edge graph is not traversed."),
            ("Select sites in anchor order", "Numbered blue sites were chosen against angular/radial targets, avoiding reused sites."),
            ("Cut each anchor corner", "Orange: actual 25%/75% Chaikin points, connected in their recorded order."),
            ("Join and check the curve", "Draw recorded spline samples, or the corner-cut polygon if fallback was selected.")],
        "checkpoint": [
            ("Sample the selected candidate's targets", "Numbered radial targets guide the walk; the final route need not pass through them."),
            ("Trace the bounded-turn walk", "Reveal successive positions of the actual open walk; the orange dot is its current end."),
            ("Close each candidate", "Heading drift and positional-gap correction create closed candidates; no morph is implied."),
            ("Choose the fewest crossings", "Blue is the first minimum crossing score; selection precedes extent normalization.")],
    }
    progressive = {"bezier": 2, "hull": 3, "polar": 2, "voronoi": 3, "checkpoint": 1}
    return [dict(title=t, explanation=e, progressive_curve=(i == progressive[name]))
            for i, (t, e) in enumerate(descriptions[name])]


def _prefix(points, progress):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    return points[:max(1, int(np.ceil(len(points)*progress)))]


def draw_construction_step(ax, sample, stage, progress):
    """Draw real geometry prefixes, with one fixed construction-coordinate view."""
    name, sc = sample["config"]["generator"], sample["construction"]
    chosen = (int(np.argmin(sc["candidate_crossings"]))
              if name == "checkpoint" and sc["candidate_crossings"] is not None else 0)
    a = sc["checkpoints"][chosen] if name == "checkpoint" else sc["anchors"]
    view = [a]
    for key in ("dense", "augmented", "sites", "walks", "candidates", "sampled"):
        if key in sc:
            view.append(sc[key].reshape(-1, 2))
    if name == "bezier":
        view.append(bezier_controls(sample).reshape(-1, 2))
    _limits(ax, view)
    _style(ax)
    active = np.empty((0, 2))

    def points(p, *, color=BLUE, labels=False, size=25):
        ax.scatter(*p.T, s=size, color=color, zorder=4)
        if labels:
            for i, q in enumerate(p):
                ax.annotate(str(i), q, xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color=INK)

    def path(p, *, color=BLUE, closed=False, lw=2, **kw):
        p = _close(p) if closed else p
        ax.plot(*p.T, color=color, lw=lw, **kw)

    if name in ("bezier", "hull"):
        if stage == 0:
            points(_prefix(sc["sampled"], progress), labels=True)
        elif stage == 1:
            points(a, color="#9eabb5", size=18)
            active = _prefix(_close(a), progress)
            path(active)
            points(active[:len(a)], labels=True)
        elif name == "bezier" and stage == 2:
            controls = bezier_controls(sample)
            dense = sc["dense"][:len(a)*sample["config"]["num_points_per_segment"]]
            active = _prefix(dense, progress)
            path(_close(a), color="#b4c0c9", lw=.8)
            seg = min((len(active)-1)//sample["config"]["num_points_per_segment"], len(a)-1)
            c = controls[seg]
            path(c, color=ORANGE, lw=1.4)
            points(c[1:3], color=ORANGE)
            points(c[[0, 3]])
            for label, p in zip(("P0", "P1", "P2", "P3"), c):
                ax.annotate(label, p, xytext=(4, -12), textcoords="offset points", fontsize=9)
            path(active)
            points(active[-1:], color=ORANGE, size=38)
        elif name == "hull" and stage == 2:
            path(a, color="#a6b3be", closed=True, lw=1)
            points(a, size=16)
            mid = (a+np.roll(a, -1, axis=0))*.5
            moved = sc["augmented"][1::2]
            n = len(_prefix(moved, progress))
            for p, q in zip(mid[:n], moved[:n]):
                ax.annotate("", q, p, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
            points(mid, color="#b4c0c9", size=12)
            points(moved[:n], color=ORANGE)
            path(_prefix(_close(sc["augmented"]), progress), color=ORANGE, lw=1)
        else:
            points(a, color="#a6b3be", size=14)
            fallback = sc.get("crossings", 0) > 0
            curve = (sample["raw"] if name == "bezier" else
                     sc["augmented"] if fallback else sc["dense"])
            active = _prefix(_close(curve) if fallback and name == "hull" else curve, progress)
            path(active)
            if name == "bezier":
                points(active[::8], size=11)
    elif name == "polar":
        radius = np.linalg.norm(a, axis=1).max()
        if stage <= 1:
            for angle in np.arange(len(a))*2*np.pi/len(a):
                path(np.array([[0, 0], [radius*np.cos(angle), radius*np.sin(angle)]]),
                     color="#c2ccd4", lw=.8, ls="--")
        if stage == 0:
            angles = np.arange(len(a))*2*np.pi/len(a)
            points(_prefix(np.column_stack((radius*np.cos(angles), radius*np.sin(angles))), progress),
                   color="#8192a0", labels=True)
        elif stage == 1:
            p = _prefix(a, progress)
            for q in p:
                path(np.array([[0, 0], q]), color=ORANGE, lw=1)
            points(p, labels=True)
            path(p, lw=1)
        else:
            points(a, color="#8e9eab", size=18)
            active = _prefix(sc["dense"], progress)
            path(active)
            if stage == 3:
                points(active[::max(1, len(active)//48)], size=10)
    elif name == "voronoi":
        sites = _prefix(sc["sites"], progress) if stage == 0 else sc["sites"]
        points(sites, color="#b9c3cb", size=10)
        if stage == 1:
            p = _prefix(a, progress)
            points(p, labels=True, size=35)
            path(p, lw=1.6)
        elif stage == 2:
            path(a, color="#94a3af", closed=True, lw=1)
            points(a, color="#94a3af", size=18)
            p = _prefix(_close(sc["augmented"]), progress)
            path(p, color=ORANGE)
            points(p, color=ORANGE, size=14)
        elif stage == 3:
            points(sc["augmented"], color=ORANGE, size=12)
            active = _prefix(sc["dense"], progress)
            path(active)
    else:
        if stage <= 1:
            points(_prefix(a, progress) if stage == 0 else a, labels=True)
            path(a, color="#adb9c2", closed=True, lw=.8)
        if stage == 1:
            active = _prefix(sc["walks"][chosen], progress)
            path(active)
            points(active[-1:], color=ORANGE, size=45)
        elif stage >= 2:
            # Closure changes coordinate scale. Separate, explicitly enlarged views
            # keep actual candidate coordinates and one shared fixed camera, instead
            # of shrinking them against the much larger open-walk coordinate box.
            candidates = sc["candidates"]
            scores = sc["candidate_crossings"]
            columns = min(2, len(candidates))
            rows = int(np.ceil(len(candidates)/columns))
            for i, curve in enumerate(candidates):
                col, row = i % columns, i // columns
                inset = ax.inset_axes([col/columns+.025, 1-(row+1)/rows+.035,
                                       .9/columns, .62/rows])
                _limits(inset, [candidates.reshape(-1, 2)])
                _style(inset)
                prefix = _prefix(_close(curve), progress)
                color = BLUE if stage == 3 and i == chosen else "#aab7c1" if stage == 3 else plt.get_cmap("tab10")(i)
                inset.plot(*prefix.T, color=color, lw=2 if stage == 3 and i == chosen else 1.3)
                score = "not counted" if scores is None else str(int(scores[i]))
                inset.set_title(f"candidate {i}\n{score} crossings" +
                                ("\nselected: first minimum" if stage == 3 and i == chosen else ""), fontsize=8,
                                color=BLUE if stage == 3 and i == chosen else INK)
            ax.text(.5, .01, "Enlarged closed-candidate views · shared fixed axes",
                    ha="center", transform=ax.transAxes, fontsize=9)
            if stage == 3:
                active = _prefix(_close(candidates[chosen]), progress)
    return dict(active_curve=active)


def _construction_result_axes(raw_ax, road_ax, sample):
    _line(raw_ax, sample["raw"], color=BLUE, lw=1.8)
    raw_ax.set_title("Generator output: raw centerline\n" +
                     (f"polygon fallback ({sample['construction']['crossings']} crossings)"
                      if sample["construction"].get("crossings", 0) else "before shared road pipeline"), fontsize=11)
    _road(road_ax, sample)
    road_ax.set_title("Finished road after shared pipeline\n" + _label(sample), fontsize=11,
                      color=GREEN if sample["valid"] else RED)
    for ax in (raw_ax, road_ax):
        _limits(ax, [sample["raw"], sample["outer"]])
        _style(ax)


def render_construction(out_dir, sample, *, movies=True):
    """Family-specific progressive drawings and a still four-step storyboard."""
    name = sample["config"]["generator"]
    if name == "repulsive":
        return
    stages = construction_stages(sample)
    footer = ("Illustrative drawing of recorded geometry: stage order is algorithm order, not elapsed runtime.\n"
              "Construction view uses pre-normalization coordinates; raw/final views at right use output coordinates.")
    if movies:
        fig = plt.figure(figsize=(12, 7.2))
        grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1), left=.025, right=.98,
                               bottom=.13, top=.86, wspace=.14, hspace=.18)
        ax = fig.add_subplot(grid[:, :2])
        _construction_result_axes(fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2]), sample)
        heading = fig.text(.34, .95, "", ha="center", fontsize=17)
        explanation = fig.text(.34, .9, "", ha="center", fontsize=10)
        fig.text(.5, .035, footer, ha="center", fontsize=10)
        writer = _writer()
        with writer.saving(fig, str(out_dir / f"generator-{name}-construction.mp4"), dpi=120):
            for i, stage in enumerate(stages):
                heading.set_text(f"{name.capitalize()} · {i+1}/4 · {stage['title']}")
                explanation.set_text(textwrap.fill(stage["explanation"], width=93))
                for progress in [*np.linspace(.02, 1, 18), *([1]*6)]:
                    ax.clear()
                    draw_construction_step(ax, sample, i, progress)
                    if i == 3 and sample["construction"].get("crossings", 0):
                        ax.text(.02, .02, f"Polygon fallback: {sample['construction']['crossings']} smooth crossings",
                                transform=ax.transAxes, color=RED, fontsize=11)
                    writer.grab_frame()
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for i, stage in enumerate(stages):
        ax = axes[i//2, i%2]
        draw_construction_step(ax, sample, i, 1)
        ax.set_title(textwrap.fill(f"{i+1}. {stage['title']}", width=39), fontsize=11)
    _construction_result_axes(axes[0, 2], axes[1, 2], sample)
    fig.suptitle(f"{name.capitalize()} construction · actual geometry, illustrative drawing", fontsize=16)
    fig.text(.5, .025, footer, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .93))
    _save(fig, out_dir / f"generator-{name}-construction.png")
    return dict(kind="progressive construction drawing of recorded geometry", sample=_record(sample),
                stage_titles=[s["title"] for s in stages], stages=len(stages),
                frames=96 if movies else 0, duration_seconds=16 if movies else 0,
                geometry_interpolated=False, elapsed_runtime=False,
                output=f"generator-{name}-construction." + ("mp4" if movies else "png"))


def _writer():
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMpegWriter(fps=6, codec="libx264", bitrate=1100,
                       extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def repulsive_force_samples(gen, *, env):
    """Read analytic barriers and weighted force samples from the actual generation."""
    from track_gen._src.warp_generate_repulsive import _N_WALL, _N_DISC

    sc = _generator_scratch(gen)
    envs = sc["reached"].shape[0]
    metadata = gen.program.tap("gen.repulsive.barriers").numpy()[env]
    size = float(metadata["size"])
    xy = sc["obs_pts"].numpy().reshape(envs, -1, 2)[env].copy() * size
    weights = sc["obs_mw"].numpy().reshape(envs, -1)[env]
    used = np.isfinite(xy).all(axis=1) & (weights != 0)
    xy[~used] = np.nan
    wall = xy[:_N_WALL][used[:_N_WALL]]
    rings = [ring.copy() for ring in xy[_N_WALL:].reshape(-1, _N_DISC, 2)
             if np.isfinite(ring).all(axis=1).any()]
    centers = sc["disc_centers"].numpy().reshape(envs, -1, 2)[env] * size
    radii = sc["disc_radii"].numpy().reshape(envs, -1)[env] * size
    active = radii > 0
    return dict(wall=wall, inner_rings=rings, n_wall=int(_N_WALL), n_disc=int(_N_DISC),
                disc_centers=centers[active].copy(), disc_radii=radii[active].copy(),
                domain_radius=float(metadata["domain_radius"])*size, size=size,
                half_width=float(gen.config.half_width),
                unscaled_margin=float(sc["margin"].numpy()[env]))


def draw_repulsive_forces(ax, samples, *, reservation=True):
    """Draw filled analytic discs, the hard wall and the centerline width reservation."""
    from matplotlib.patches import Circle

    radius, margin = samples["domain_radius"], samples["half_width"]
    wall = Circle((0, 0), radius, fill=False, edgecolor="#65856f", lw=1.1,
                  label="Hard domain wall", zorder=0)
    ax.add_patch(wall)
    ax.plot(*samples["wall"].T, color="#65856f", ls="", marker=".", markersize=2, alpha=.6)
    inner, reserved = [], []
    for i, (center, r) in enumerate(zip(samples["disc_centers"], samples["disc_radii"])):
        disc = Circle(center, float(r), facecolor="#ead2ba", edgecolor=ORANGE, lw=.8,
                      label="Hard filled discs" if i == 0 else None, zorder=0)
        ax.add_patch(disc)
        inner.append(disc)
        if reservation:
            ring = Circle(center, float(r)+margin, fill=False, edgecolor=ORANGE,
                          lw=.65, ls=":", alpha=.8,
                          label="Centerline width reservation" if i == 0 else None, zorder=0)
            ax.add_patch(ring)
            reserved.append(ring)
    if reservation and radius > margin:
        ring = Circle((0, 0), radius-margin, fill=False, edgecolor="#65856f",
                      lw=.7, ls=":", alpha=.8, zorder=0)
        ax.add_patch(ring)
        reserved.append(ring)
    return dict(wall=wall, inner=inner, reservation=reserved)


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
    effective_size = float(gen.program.tap("gen.repulsive.barriers").numpy()[0]["size"])
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
        ratchet_complete = bool(sc["reached"].numpy()[0])
        xy = sc["center"].numpy()[:n].copy()*effective_size
        frames.append(dict(iteration=iteration, n=n, xy=xy,
                           ratchet_complete=ratchet_complete, frozen=bool(sc["frozen"].numpy()[0]),
                           target=float(sc["L_target"].numpy()[0])*effective_size,
                           target_final=float(sc["L_final"].numpy()[0])*effective_size,
                           perimeter=float(np.linalg.norm(np.roll(xy.astype(np.float64), -1, axis=0)-xy,
                                                          axis=1).sum()),
                           wall_forces_active=True, inner_forces_active=True,
                           hard_barriers_active=True))

    expected = gen.program.tap("raw.xy").numpy().copy()
    expected_valid = gen.program.tap("gen.valid").numpy().copy()
    expected_barriers = gen.program.tap("gen.repulsive.barriers").numpy().tobytes()
    if not np.array_equal(expected, sample["raw"]):
        raise AssertionError("capture configuration does not reproduce the selected sample")
    gen.program._gen_scratch.enqueue(Observer())
    actual = gen.program.tap("raw.xy").numpy()
    matches = bool(np.array_equal(expected, actual)
                   and np.array_equal(expected_valid, gen.program.tap("gen.valid").numpy())
                   and expected_barriers == gen.program.tap("gen.repulsive.barriers").numpy().tobytes())
    if not matches:
        raise AssertionError("observing the growth changed the final generator output")
    frames.append(dict(iteration=frames[-1]["iteration"], n=len(actual), xy=actual.copy(),
                       ratchet_complete=frames[-1]["ratchet_complete"], frozen=frames[-1]["frozen"],
                       target=frames[-1]["target"], target_final=frames[-1]["target_final"],
                       perimeter=float(np.linalg.norm(np.roll(actual.astype(np.float64), -1, axis=0)-actual,
                                                      axis=1).sum()),
                       final=True, wall_forces_active=True, inner_forces_active=True,
                       hard_barriers_active=True))
    return frames, matches


def render_growth(out_dir, sample, *, movies=True):
    frames, matches = capture_repulsive(sample)
    fig, (ax, final_ax) = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": (1.55, 1)})
    samples = sample["construction"]["force_samples"]
    draw_repulsive_forces(ax, samples)
    ax.legend(loc="upper left", fontsize=8, framealpha=.95)
    line, = ax.plot([], [], color=BLUE, lw=2)
    label = ax.text(.02, .02, "", transform=ax.transAxes, fontsize=10, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    radius = samples["domain_radius"]
    for axis in (ax, final_ax):
        axis.set_xlim(-1.12*radius, 1.12*radius)
        axis.set_ylim(-1.12*radius, 1.12*radius)
        _style(axis, ticks=True)
    _road(final_ax, sample, raw=True)
    draw_repulsive_forces(final_ax, samples)
    final_ax.set_title("Finished road after shared pipeline\n" + _label(sample), fontsize=11,
                       color=GREEN if sample["valid"] else RED)
    ax.set_title("Repulsive · actual raw growth states\nseed " + str(sample["seed"]) + " · fixed viewpoint", fontsize=14)
    fig.text(.5, .025, "Filled discs and outer wall are hard barriers; dotted curves reserve the absolute road half-width.\n"
             "Disc/wall soft forces remain enabled. Final validity also checks every four-vertex road-cell hull. Playback is not runtime.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, 1))

    def draw(frame):
        line.set_data(*_close(frame["xy"]).T)
        label.set_text(f"iteration {frame['iteration']} · N={frame['n']}\n"
                       f"perimeter {frame['perimeter']:.3f} · ratchet {frame['target']:.3f} / final {frame['target_final']:.3f}\n"
                       f"ratchet complete={frame['ratchet_complete']} · frozen={frame['frozen']}" +
                       ("\nfinal output after resample check" if frame.get("final") else ""))
    if movies:
        writer = _writer()
        with writer.saving(fig, str(out_dir / "generator-repulsive-growth.mp4"), dpi=120):
            for frame in frames:
                draw(frame)
                writer.grab_frame()
            for _ in range(12):
                writer.grab_frame()
    draw(frames[-1])
    view_bounds = dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
    _save(fig, out_dir / "generator-repulsive-growth.png")
    return dict(kind="captured solver states", iterations=sorted({f["iteration"] for f in frames}),
                seed=sample["seed"], final_matches_generator=matches, geometry_interpolated=False,
                frame_count=len(frames), encoded_frame_count=len(frames)+12 if movies else 0, fps=6,
                view_bounds=view_bounds, output="generator-repulsive-growth." + ("mp4" if movies else "png"),
                sample=_record(sample), soft_shaping_forces=True, hard_collision_obstacles=True,
                width_reservation=samples["half_width"], n_wall=samples["n_wall"], n_disc=samples["n_disc"],
                force_states=[{k: f.get(k, False) for k in ("iteration", "n", "target", "target_final",
                              "perimeter", "ratchet_complete", "frozen", "inner_forces_active", "wall_forces_active",
                              "hard_barriers_active", "final")} for f in frames])


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

    return records


def render_repulsive_numerics(out_dir):
    records = {}
    # End-state sensitivity data for numerical controls, without inventing monotonic
    # fold/quality trends or timing comparisons between unlike solver schedules.
    baseline = sample_batch("repulsive",seeds=[23])[0]
    fig,axes = plt.subplots(2,2,figsize=(10,9))
    experiments = []
    for ax,(field,value) in zip(axes.flat,(("repulsive_alpha",3.5),("repulsive_beta",6.5),
                                         ("repulsive_tau",.2),("repulsive_resample_every",10))):
        alternative = sample_batch("repulsive",seeds=[23],overrides={field:value})[0]
        delta = curve_delta(baseline["raw"],alternative["raw"])
        assert delta > 1e-6
        _line(ax,baseline["raw"],color=BLUE,lw=1.8,label="default")
        _line(ax,alternative["raw"],color=ORANGE,lw=1.5,label="changed")
        _limits(ax,[baseline["raw"],alternative["raw"],baseline["construction"]["force_samples"]["wall"]])
        draw_repulsive_forces(ax, baseline["construction"]["force_samples"])
        _style(ax)
        ax.set_title(f"{field.removeprefix('repulsive_')}: {baseline['config'][field]} → {value}\n"
                     f"raw displacement max {delta:.3f} coordinate units",fontsize=12)
        ax.text(.5,-.02,f"changed: {_label(alternative)} · raw valid={alternative['construction']['generator_valid']}\n"
                f"growth frozen={alternative['construction']['frozen']}",transform=ax.transAxes,ha="center",fontsize=10)
        ax.legend(handles=ax.lines[:2], loc="upper right", frameon=True, fontsize=9)
        experiments.append(dict(field=field,max_raw_delta=delta,
                                samples=[_record(baseline),_record(alternative)],
                                changed_ratchet_complete=alternative["construction"]["ratchet_complete"],
                                changed_frozen=alternative["construction"]["frozen"]))
    fig.suptitle("Repulsive numerical controls · same seed 23 · one setting changed",fontsize=16)
    fig.text(.5,.01,"Blue default vs orange changed: actual final generator states, before the road tail.\nAxes match within each pair. These outcomes establish sensitivity, not a monotonic shape or runtime rule.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.95))
    _save(fig,out_dir / "generator-repulsive-numerics.png")
    records["numerical_controls"] = experiments
    return records


def render_repulsive_barriers(out_dir):
    """Actual size-dependent layouts and a post-tail rejection, with failed output kept."""
    sizes = [sample_batch("repulsive", seeds=[7], overrides={"size": value})[0]
             for value in (.5, 1., 2.)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, sample in zip(axes, sizes):
        barriers = sample["construction"]["force_samples"]
        _road(ax, sample, raw=True)
        draw_repulsive_forces(ax, barriers)
        radius = barriers["domain_radius"]
        ax.set_xlim(-1.12*radius, 1.12*radius)
        ax.set_ylim(-1.12*radius, 1.12*radius)
        _style(ax, ticks=True)
        ax.set_title(f"size={sample['config']['size']} · half_width=.1\n" + _label(sample), fontsize=11)
        ax.text(.5, -.27, f"unscaled width reservation {barriers['unscaled_margin']:.2f}\n"
                f"active sampled discs: {len(barriers['disc_radii'])}",
                ha="center", transform=ax.transAxes, fontsize=10)
    fig.suptitle("Repulsive · absolute width changes the growth reservation and disc layout", fontsize=15)
    fig.text(.5, .015, "Same seed 7. Filled discs/wall: hard barriers; dotted: reserved centerline margin.\n"
             "Each view includes its full domain; tick values show size. Blue raw, orange dashed final, grey road; keep retains failures.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .19, 1, .94))
    _save(fig, out_dir / "generator-repulsive-controls-obstacles.png")

    pair = [sizes[1], sample_batch("repulsive", seeds=[7], overrides={
        "repulsive_grow_mult_min": 3., "repulsive_grow_mult_max": 3.5})[0]]
    if not pair[0]["valid"] or not pair[1]["construction"]["generator_valid"] or not (pair[1]["fail"] & 256):
        raise AssertionError("the recorded CPU post-tail rejection no longer reproduces")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, sample, title in zip(axes, pair, ("Default growth range [2, 3]", "Growth range [3, 3.5]")):
        _road(ax, sample, raw=True)
        barriers = sample["construction"]["force_samples"]
        draw_repulsive_forces(ax, barriers)
        radius = barriers["domain_radius"]
        ax.set_xlim(-1.12*radius, 1.12*radius)
        ax.set_ylim(-1.12*radius, 1.12*radius)
        _style(ax)
        ax.set_title(title + "\n" + _label(sample), fontsize=12,
                     color=GREEN if sample["valid"] else RED)
        ax.text(.5, -.04, f"raw valid={sample['construction']['generator_valid']} · "
                f"ratchet complete={sample['construction']['ratchet_complete']}\n"
                f"frozen={sample['construction']['frozen']}", transform=ax.transAxes, ha="center", fontsize=10)
    fig.suptitle("Repulsive · feasible raw growth can still fail the finished-road check", fontsize=15)
    fig.text(.5, .015, "Actual CPU seed 7, epoch 0, env 0. Blue raw; orange dashed final; grey finite-width road.\n"
             "Right: final road-cell hull meets a disc (fail bit 256). on_invalid='keep' preserves the rejected geometry for diagnosis.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .94))
    _save(fig, out_dir / "generator-repulsive-limits.png")
    return dict(width_reservation=[_record(s) for s in sizes], finished_band_limit=[_record(s) for s in pair])


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
        construction_animation = render_construction(out_dir,selected[0],movies=movies)
        comparisons = render_comparisons(out_dir,name)
        manifest["families"][name] = dict(seed_pool=seeds, examples=[_record(s) for s in selected],
                                          comparisons=comparisons)
        if name != "repulsive":
            manifest["animations"][name] = construction_animation
        print(f"rendered {name}",flush=True)
    render_gallery(out_dir,examples)
    manifest["specials"] = render_specials(out_dir) | render_repulsive_numerics(out_dir)
    manifest["specials"].update(render_repulsive_barriers(out_dir))
    manifest["animations"]["repulsive"] = render_growth(out_dir,examples["repulsive"][0],movies=movies)
    manifest["assets"] = {p.name: p.stat().st_size for p in sorted(out_dir.glob("generator*")) if p.suffix in (".png",".mp4")}
    (out_dir / "generators-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def render_construction_animations(out_dir: Path, source_manifest: Path):
    """Refresh only construction assets, preserving the other visual selections."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source_manifest.read_text())
    source_dir = Path(__file__).resolve().parents[1] / "track_gen/_src"
    source = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  warp_version=wp.__version__, numpy_version=np.__version__,
                  matplotlib_version=matplotlib.__version__, trackgen_version=__version__, device="cpu")
    for name in FAMILIES[:-1]:
        old = manifest["animations"][name]["sample"]
        overrides = _current_overrides(old["config"])
        pool = sample_batch(name, seeds=manifest["families"][name]["seed_pool"], overrides=overrides)
        sample = pool[old["env"]]
        if sample["seed"] != old["seed"]:
            raise AssertionError("construction seed-pool selection changed")
        record = render_construction(out_dir, sample)
        source_file = source_dir / ("warp_generate" + ("" if name == "bezier" else "_"+name) + ".py")
        record["source"] = dict(source, generator_source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest())
        record["command"] = ("python -m viz.render_generator_guide --construction-only "
                             "--source-manifest docs/assets/racing/generators-manifest.json --out-dir DIR")
        manifest["animations"][name] = record
        for suffix in (".png", ".mp4"):
            file = out_dir / ("generator-"+name+"-construction"+suffix)
            manifest["assets"][file.name] = file.stat().st_size
        print(f"rendered construction {name}", flush=True)
    (out_dir / "generators-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def render_repulsive_animation(out_dir: Path, source_manifest: Path):
    """Refresh only Repulsive's force-state overlay, keeping the selected configuration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source_manifest.read_text())
    old = manifest["animations"]["repulsive"].get("sample", manifest["families"]["repulsive"]["examples"][0])
    overrides = _current_overrides(old["config"])
    pool = sample_batch("repulsive", seeds=manifest["families"]["repulsive"]["seed_pool"], overrides=overrides)
    sample = pool[old["env"]]
    if sample["seed"] != old["seed"]:
        raise AssertionError("Repulsive seed-pool selection changed")
    record = render_growth(out_dir, sample)
    source_file = Path(__file__).resolve().parents[1] / "track_gen/_src/warp_generate_repulsive.py"
    record["source"] = dict(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        warp_version=wp.__version__, numpy_version=np.__version__,
        matplotlib_version=matplotlib.__version__, trackgen_version=__version__, device="cpu")
    record["command"] = ("python -m viz.render_generator_guide --repulsive-only "
                         "--source-manifest docs/assets/racing/generators-manifest.json --out-dir DIR")
    manifest["animations"]["repulsive"] = record
    for suffix in (".png", ".mp4"):
        file = out_dir / ("generator-repulsive-growth"+suffix)
        manifest["assets"][file.name] = file.stat().st_size
    (out_dir / "generators-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def _current_overrides(config):
    """Replay an older selection with only settings supported by the current config."""
    fields = {field.name for field in dataclasses.fields(TrackGenConfig)}
    return {key: value for key, value in config.items()
            if key in fields and key not in ("generator", "num_envs", "device", "on_invalid")}


def render_repulsive_assets(out_dir: Path, source_manifest: Path, *, movies=True):
    """Refresh all Repulsive figures plus its shared gallery, preserving other media."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(source_manifest.read_text())
    root = Path(__file__).resolve().parents[1]
    source_file = root / "track_gen/_src/warp_generate_repulsive.py"
    dirty_source = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "track_gen", "viz/render_generator_guide.py"], text=True)
    if dirty_source:
        raise RuntimeError("commit the resolved runtime and renderer before final asset rendering")
    source = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                  source_tree_dirty=False,
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  generator_source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                  repulsive_dependency_sha256={name: hashlib.sha256((source_file.parent / name).read_bytes()).hexdigest()
                      for name in ("repulsive_clearance.py", "repulsive_barrier_energy.py", "repulsive_tangent_energy.py")},
                  warp_version=wp.__version__, numpy_version=np.__version__,
                  matplotlib_version=matplotlib.__version__, trackgen_version=__version__, device="cpu")
    seeds = [7, 23, 41, 59]
    pool = sample_batch("repulsive", seeds=seeds)
    selected = [sample for sample in pool if sample["valid"] and np.isfinite(sample["center"]).all()][:3]
    if len(selected) != 3:
        raise AssertionError(f"not enough valid Repulsive examples: {[(s['seed'], s['fail']) for s in pool]}")
    render_examples(out_dir, selected)
    comparisons = render_comparisons(out_dir, "repulsive")
    manifest["families"]["repulsive"] = dict(seed_pool=seeds, examples=[_record(s) for s in selected],
                                               comparisons=comparisons, source=source)
    manifest["specials"].update(render_repulsive_numerics(out_dir))
    manifest["specials"].update(render_repulsive_barriers(out_dir))
    animation = render_growth(out_dir, selected[0], movies=movies)
    animation["source"] = source
    manifest["animations"]["repulsive"] = animation
    selections = {"repulsive": selected}
    for name in FAMILIES[:-1]:
        previous = manifest["families"][name]
        config = previous["examples"][0]["config"]
        batch = sample_batch(name, seeds=previous["seed_pool"], overrides=_current_overrides(config))
        selections[name] = [batch[previous["examples"][0]["env"]]]
    render_gallery(out_dir, selections)
    command = ("python -m viz.render_generator_guide --repulsive-assets-only "
               "--source-manifest docs/assets/racing/generators-manifest.json --out-dir DIR" +
               (" --no-movies" if not movies else ""))
    manifest["families"]["repulsive"]["command"] = command
    animation["command"] = command
    manifest["setting_coverage"] = SETTING_COVERAGE
    provenance = manifest.setdefault("asset_provenance", {})
    refreshed = [*out_dir.glob("generator-repulsive-*.png"), out_dir / "generators-gallery.png"]
    if movies:
        refreshed.extend(out_dir.glob("generator-repulsive-*.mp4"))
    for path in sorted(refreshed):
        if path.suffix in (".png", ".mp4"):
            manifest["assets"][path.name] = path.stat().st_size
            provenance[path.name] = dict(source=source, command=command)
    (out_dir / "generators-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=Path("docs/assets/racing"))
    parser.add_argument("--no-movies",action="store_true",help="render posters only (no ffmpeg needed)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--construction-only", action="store_true",
                        help="refresh the five construction movies/posters using recorded manifest selections")
    selection.add_argument("--repulsive-only", action="store_true",
                           help="refresh only Repulsive growth movie/poster")
    selection.add_argument("--repulsive-assets-only", action="store_true",
                           help="refresh all Repulsive assets and its shared gallery from committed source")
    parser.add_argument("--source-manifest", type=Path,
                        default=Path("docs/assets/racing/generators-manifest.json"))
    args = parser.parse_args()
    if args.repulsive_assets_only:
        render_repulsive_assets(args.out_dir, args.source_manifest, movies=not args.no_movies)
    elif args.repulsive_only:
        if args.no_movies:
            parser.error("--repulsive-only includes a movie; omit --no-movies")
        render_repulsive_animation(args.out_dir, args.source_manifest)
    elif args.construction_only:
        if args.no_movies:
            parser.error("--construction-only includes movies; omit --no-movies")
        render_construction_animations(args.out_dir, args.source_manifest)
    else:
        render(args.out_dir,movies=not args.no_movies)


if __name__ == "__main__":
    main()
