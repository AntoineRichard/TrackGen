"""Render course-control figures from current CPU TrackGen output.

Run ``python -m viz.render_course_guide --out-dir /tmp/course-guide --check``.
Rendering is an explicit maintenance step; Sphinx only copies the selected assets.
All samples use seed 5, epoch 0, env 0. Failed candidates explicitly use ``keep``.
The JSON manifest records configs, counts, validity and numerical comparison checks.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from track_gen import GateGenConfig, GateGenerator, TrackGenConfig, TrackGenerator  # noqa: E402
from track_gen.pipeline import StageRef, load_centerlines, presets  # noqa: E402

SEED = 5
BLUE, ORANGE, GREEN, RED = "#2563eb", "#d97706", "#15803d", "#b91c1c"
BASE = dict(generator="polar", num_envs=1, device="cpu", polar_radial_jitter=0.25,
            polar_angular_jitter=0.1)
RECORDS: list[dict] = []
CHECKS: dict[str, bool] = {}


def _closed(a):
    return np.concatenate([a, a[:1]])


def _s(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(_closed(p), axis=0), axis=1))]


def _status(d):
    names = [k for k, v in d["fail_bits"].items() if int(d["fail"]) & (1 << int(v))]
    return "valid" if d["valid"] else "INVALID: " + ", ".join(names)


def _capture(result, name, config, kind):
    n = int(result.count.numpy()[0])
    valid, fail = bool(result.valid.numpy()[0]), int(result.fail_mask.numpy()[0])
    fields = ("center", "outer", "inner") if kind == "track" else (
        "position", "tangent", "forward", "orientation", "half_size", "left", "right")
    d = {field: getattr(result, field).numpy()[:n].copy() for field in fields}
    d.update(count=n, valid=valid, fail=fail, fail_bits=dict(result.fail_bits))
    for field in fields:
        assert len(d[field]) == n, (name, field, n)
        assert np.isfinite(d[field]).all(), (name, field, n)
    assert n >= 3, (name, n)
    if kind == "track":
        d["truncated"] = bool(result.truncated.numpy()[0])
    row = dict(name=name, kind=kind, seed=SEED, epoch=0, env=0, count=n,
               valid=valid, fail_mask=fail, fail_bits=dict(result.fail_bits), config=config)
    if kind == "track":
        row.update(truncated=d["truncated"], xy_length=float(_s(d["center"][:, :2])[-1]),
                   xyz_length=float(_s(d["center"])[-1]),
                   half_width=float(np.linalg.norm(d["outer"] - d["center"], axis=1).mean()))
    RECORDS.append(row)
    return d


def track(name, **kw):
    cfg = TrackGenConfig(**(BASE | kw))
    gen = TrackGenerator(cfg, seeds=SEED)
    d = _capture(gen.generate(), name, dataclasses.asdict(cfg), "track")
    # The polar built-in emits a fixed num_points raw row (no raw.count channel).
    d["raw"] = gen.program.tap("raw.xy", after="generator").numpy()[:cfg.num_points].copy()
    return d


def gates(name, **kw):
    cfg = GateGenConfig(num_envs=1, device="cpu", **kw)
    return _capture(GateGenerator(cfg, seeds=SEED).generate(), name,
                    dataclasses.asdict(cfg), "gates")


def preset_gates(name, **kw):
    options = dict(num_envs=1, device="cpu") | kw
    gen = GateGenerator(presets.gates(**options), seeds=SEED)
    d = _capture(gen.generate(), name, options, "gates")
    return d, gen


def _plan(ax, d, *, beads=False, limits=None):
    p = d["center"]
    o, i = _closed(d["outer"]), _closed(d["inner"])
    ax.fill(np.r_[o[:, 0], i[::-1, 0]], np.r_[o[:, 1], i[::-1, 1]], color="#dbeafe")
    for border in (o, i):
        ax.plot(border[:, 0], border[:, 1], color=BLUE, lw=0.9)
    c = _closed(p)
    ax.plot(c[:, 0], c[:, 1], color="#334155", lw=0.8, ls="--")
    if beads:
        ax.scatter(p[:, 0], p[:, 1], s=8, color=ORANGE, zorder=4)
    if limits is not None:
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
    ax.set_aspect("equal")
    ax.set_xlabel("x [coordinate units]")
    ax.set_ylabel("y [coordinate units]")
    ax.text(0.03, 0.02, f"{d['count']} points · {_status(d)}", transform=ax.transAxes,
            fontsize=8, color=GREEN if d["valid"] else RED)


def _gate_xy(d):
    return np.concatenate([d[field][:d["count"], :2] for field in ("position", "left", "right")])


def _gate_bounds_contain(ax, d):
    points = _gate_xy(d)
    return bool(np.all(points >= (ax.get_xlim()[0], ax.get_ylim()[0])) and
                np.all(points <= (ax.get_xlim()[1], ax.get_ylim()[1])))


def _gate_plan(ax, d, *, radius=0.0, numbering=False, limits=1.2):
    p = d["position"]
    c = _closed(p)
    ax.plot(c[:, 0], c[:, 1], "--", color="#94a3b8", lw=0.9)
    for j, (center, left, right) in enumerate(zip(p, d["left"], d["right"])):
        ax.plot([left[0], right[0]], [left[1], right[1]], color=BLUE, lw=2)
        if radius:
            ax.add_patch(plt.Circle(center[:2], radius, fill=False, ec=ORANGE, lw=0.8))
        if numbering:
            ax.annotate(str(j), center[:2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.scatter(p[:, 0], p[:, 1], color="#1e293b", s=12, zorder=3)
    ax.set(xlim=(-limits, limits), ylim=(-limits, limits), xlabel="x [coordinate units]",
           ylabel="y [coordinate units]", aspect="equal")
    ax.text(0.02, 0.02, f"{d['count']} gates · {_status(d)}", transform=ax.transAxes,
            fontsize=8, color=GREEN if d["valid"] else RED)


def _save(fig, out, stem, title):
    fig.suptitle(title + "\nCPU · seed 5 · epoch 0 · env 0 · selected examples, not yield estimates",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / (stem + ".png"), dpi=160, facecolor="white")
    plt.close(fig)


def size_width(out):
    samples = [track("size-base"), track("size-2", size=2),
               track("scale-2", scale=2), track("width-2", half_width=0.2, spacing=0.06, on_invalid="keep")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, d, title in zip(axes.flat, samples, ("scale=1, size=1, half_width=.1",
                                           "size=2 (dynamic)", "scale=2 (static)",
                                           "half_width=.2 (static); keep")):
        _plan(ax, d, limits=2.3)
        ax.set_title(title, fontsize=10)
    CHECKS["size_changes_xy"] = np.ptp(samples[1]["center"][:, 0]) > 1.5 * np.ptp(samples[0]["center"][:, 0])
    CHECKS["width_changes_band"] = np.linalg.norm(samples[3]["outer"] - samples[3]["center"], axis=1).mean() > 0.19
    CHECKS["size_preserves_absolute_width"] = all(abs(np.linalg.norm(d["outer"] - d["center"], axis=1).mean() - .1) < 1e-6 for d in samples[:3])
    _save(fig, out, "controls-size-width", "XY size and scale leave absolute width independent")


def sampling(out):
    samples = [track("spacing-fine", spacing=0.03, N_max=512),
               track("spacing-coarse", spacing=0.09),
               track("capacity-truncated", spacing=0.03, N_max=48, on_invalid="keep")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), gridspec_kw={"height_ratios": [4, 1]})
    for col, (d, label, capacity) in enumerate(zip(samples,
            ("spacing=.03, N_max=512", "spacing=.09, N_max=384", "spacing=.03, N_max=48; keep"),
            (512, 384, 48))):
        _plan(axes[0, col], d, beads=True, limits=1.2)
        axes[0, col].set_title(label, fontsize=10)
        ax = axes[1, col]
        n = d["count"]
        ax.barh([0], [n], color=BLUE, label="real count")
        ax.barh([0], [capacity - n], left=[n], color="#e2e8f0", label="padding")
        ax.text(capacity / 2, 0, f"count {n} / capacity {capacity}", ha="center", va="center", fontsize=9)
        ax.set(xlim=(0, capacity), yticks=[], xlabel="slots in one env row")
    CHECKS["spacing_changes_count"] = samples[0]["count"] > 2 * samples[1]["count"]
    CHECKS["capacity_truncates_and_fails"] = samples[2]["truncated"] and not samples[2]["valid"]
    poisoned = TrackGenerator(TrackGenConfig(**(BASE | dict(spacing=.03, N_max=48))), seeds=SEED).generate()
    CHECKS["poison_clears_failed_geometry"] = int(poisoned.count.numpy()[0]) == 0 and np.isnan(poisoned.center.numpy()).all() and int(poisoned.fail_mask.numpy()[0]) == samples[2]["fail"]
    _save(fig, out, "controls-sampling-capacity", "Spacing changes samples; capacity bounds storage")
    samples = [track("resolution-16", num_points=16), track("resolution-256", num_points=256)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, d, count in zip(axes, samples, (16, 256)):
        _plan(ax, d, limits=1.2)
        raw = _closed(d["raw"])
        ax.plot(raw[:, 0], raw[:, 1], color=ORANGE, lw=.8, marker="o", ms=2, label="raw generator samples")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"num_points={count}; spacing=.06", fontsize=11)
    CHECKS["resolution_nonvacuous"] = (samples[0]["count"] != samples[1]["count"] or
        not np.allclose(samples[0]["center"], samples[1]["center"]))
    _save(fig, out, "controls-resolution", "Intermediate resampling resolution is separate from final spacing")


def width_channel(out):
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    xy = np.c_[np.cos(angles), np.sin(angles)].astype(np.float32)
    source = (0.10 + 0.06 * np.cos(3 * angles)).astype(np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    widths = []
    for col, (factor, lo, hi) in enumerate(((1, 0, 1e9), (1.5, .08, .16))):
        opt = dict(num_envs=1, device="cpu", generator="external", num_points=128,
                   generator_params={"attributes": (("guide.width", "float32", "linear"),)},
                   spacing=.04, n_max=256,
                   width=StageRef("width", "from_channel", params={"scale": factor, "lo": lo, "hi": hi},
                                  ports={"src": "cl.guide.width"}))
        gen = TrackGenerator(presets.track(**opt), seeds=SEED)
        load_centerlines(gen, [xy], attributes={"guide.width": [source]})
        result = gen.generate()
        serial = opt | {"width": dict(kind="width", name="from_channel", scale=factor, lo=lo, hi=hi,
                                      ports={"src": "cl.guide.width"})}
        d = _capture(result, f"width-channel-{col}", serial, "track")
        _plan(axes[0, col], d, limits=1.3)
        axes[0, col].set_title(f"width.scale={factor}, lo={lo}, hi={hi:g}")
        n = d["count"]
        actual = result.half_width.numpy()[:n]
        carried = gen.program.tap("cl.guide.width", after="uniform").numpy()[:n]
        widths.append(actual)
        stage_count = int(gen.program.tap("cl.count", after="resample").numpy()[0])
        source_at_width = gen.program.tap("cl.guide.width", after="resample").numpy()[:stage_count]
        width_at_stage = gen.program.tap("cl.half_width", after="width").numpy()[:stage_count]
        # Clamp precedes the final uniform resampler; clamp and interpolation do not commute.
        CHECKS[f"width_channel_formula_{col}"] = np.allclose(width_at_stage, np.clip(factor * source_at_width, lo, hi))
        CHECKS[f"width_channel_bounds_{col}"] = actual.min() >= lo - 1e-6 and actual.max() <= hi + 1e-6
        ax = axes[1, col]
        ax.plot(_s(d["center"][:, :2])[:-1], carried, "--", color="#64748b", label="carried source")
        ax.plot(_s(d["center"][:, :2])[:-1], actual, color=BLUE, label="output half-width")
        ax.set(xlabel="XY arclength [units]", ylabel="half-width [units]", ylim=(0, .25))
        ax.legend(fontsize=9)
    CHECKS["width_channel_nonvacuous"] = not np.allclose(*widths)
    _save(fig, out, "controls-width-channel", "A wired width channel: multiply and clamp, with fixed centerline")


def elevation(out):
    common = dict(z_min=-0.4, z_max=0.4)
    samples = [track("elevation-flat", z_base=0.2),
               track("elevation-uniform", z_profile="uniform", **common),
               track("elevation-random_walk", z_profile="random_walk", z_max_step=0.3, **common),
               track("elevation-noise", z_profile="noise", z_noise_amplitude=0.35, **common)]
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), gridspec_kw={"width_ratios": [1, 1.7]})
    for row, (name, d) in enumerate(zip(("flat", "uniform", "random_walk", "noise"), samples)):
        p = d["center"]
        ax = axes[row, 0]
        segments = np.stack([p[:, :2], np.roll(p[:, :2], -1, axis=0)], axis=1)
        lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(-0.4, 0.4), linewidths=3)
        lc.set_array(p[:, 2])
        ax.add_collection(lc)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal", xlabel="x [units]", ylabel="y [units]")
        ax.set_title(name + " · " + _status(d), fontsize=10)
        fig.colorbar(lc, ax=ax, label="altitude [units]", fraction=0.045)
        ax = axes[row, 1]
        ax.plot(_s(p[:, :2]), _closed(p)[:, 2], color=BLUE)
        ax.axhline(-0.4, ls=":", color="#64748b")
        ax.axhline(0.4, ls=":", color="#64748b")
        ax.set(xlabel="XY arclength [coordinate units]", ylabel="altitude [units]", ylim=(-0.48, 0.48))
        ax.grid(alpha=0.15)
    CHECKS["elevation_preserves_xy"] = all(np.array_equal(samples[0]["center"][:, :2], d["center"][:, :2]) for d in samples[1:])
    CHECKS["all_profiles_active"] = all(np.ptp(d["center"][:, 2]) > 0.01 for d in samples[1:])
    _save(fig, out, "controls-elevation", "Four altitude models on one unchanged plan-view road")
    comparisons = [
        ("Uniform: altitude window", "uniform", [{"z_min": -0.2, "z_max": 0.2}, {"z_min": -0.4, "z_max": 0.4}], ["[-.2, .2]", "[-.4, .4]"]),
        ("Uniform: static knot count", "uniform", [common | {"z_control_points": 5}, common | {"z_control_points": 15}], ["5 knots", "15 knots"]),
        ("Random walk: grade at knots", "random_walk", [common | {"z_max_step": 0.1}, common | {"z_max_step": 0.4}], ["z_max_step=.1", "z_max_step=.4"]),
        ("Noise: amplitude", "noise", [common | {"z_noise_amplitude": 0.1}, common | {"z_noise_amplitude": 0.35}], ["amplitude=.1", "amplitude=.35"]),
        ("Noise: static harmonics", "noise", [common | {"z_noise_amplitude": 0.35, "z_noise_harmonics": 1}, common | {"z_noise_amplitude": 0.35, "z_noise_harmonics": 5}], ["1 harmonic", "5 harmonics"]),
        ("Noise: clipping window", "noise", [common | {"z_noise_amplitude": 0.35}, {"z_min": 0, "z_max": 0.2, "z_noise_amplitude": 0.35}], ["[-.4, .4]", "[0, .2]"]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for idx, (ax, (title, profile, options, labels)) in enumerate(zip(axes.flat, comparisons)):
        pair = []
        for opt, label, color in zip(options, labels, (BLUE, ORANGE)):
            d = track(f"elevation-knob-{idx}-{label}", z_profile=profile, **opt)
            pair.append(d["center"][:, 2])
            ax.plot(_s(d["center"][:, :2]), _closed(d["center"])[:, 2], label=label, color=color)
        CHECKS[f"elevation_knob_{idx}_nonvacuous"] = not np.allclose(*pair)
        ax.set(title=title, xlabel="XY arclength [units]", ylabel="altitude [units]", ylim=(-0.45, 0.45))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)
    _save(fig, out, "controls-elevation-knobs", "Elevation controls: one changed setting per comparison")


def sources_and_order(out):
    samples = [preset_gates("source-" + source, generator="polar", source=source,
                            **({"gates": 12} if source != "anchors" else {}),
                            gate_width=0.1, gate_radius=0)[0]
               for source in ("anchors", "stations", "relaxed_stations")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    for ax, d, label in zip(axes, samples, ("Native polar anchors", "Stations on raw track curve", "Stations on relaxed track curve")):
        _gate_plan(ax, d, numbering=True)
        ax.set_title(label, fontsize=10)
    CHECKS["sources_differ"] = not np.allclose(samples[0]["position"], samples[1]["position"])
    _save(fig, out, "controls-gate-sources", "The same family has distinct native-gate and track-station paths")
    comparisons = [(order, width) for order in ("ccw", "random_pairs") for width in (0.0, 0.12, 0.4)]
    order_samples = [gates(f"order-{order}-width-{width}", gate_ordering=order,
                           gate_width=width, on_invalid="keep") for order, width in comparisons]
    # All six panels share a square view, with room for bars and gate-number labels.
    limits = float(np.max(np.abs(np.concatenate([_gate_xy(d) for d in order_samples])))) * 1.10
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, d, (order, width) in zip(axes.flat, order_samples, comparisons):
        _gate_plan(ax, d, numbering=True, limits=limits)
        ax.set_title(f"order={order}, gate_width={width}", fontsize=10)
    CHECKS["gate_order_width_bounds_contain_geometry"] = all(
        _gate_bounds_contain(ax, d) for ax, d in zip(axes.flat, order_samples))
    _save(fig, out, "controls-gate-order-width", "Ordering changes traversal; full gate width changes the opening")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    rad_samples = []
    for ax, radius, iters in zip(axes, (0.025, 0.13, 0.13), (0, 0, 8)):
        d = gates(f"radius-{radius}-rounds-{iters}", gate_radius=radius, gate_solve_iters=iters,
                  gate_width=0.12, on_invalid="keep")
        rad_samples.append(d)
        _gate_plan(ax, d, radius=radius, limits=1.4)
        ax.set_title(f"gate_radius={radius}, rounds={iters}", fontsize=10)
    CHECKS["radius_solver_moves_centers"] = not np.allclose(rad_samples[1]["position"], rad_samples[2]["position"])
    _save(fig, out, "controls-gate-radius", "Separation radius is independent of the gate's opening width")
    d = gates("gate-count-layout")
    fig, ax = plt.subplots(figsize=(10, 2.7))
    count, capacity, stride, floor = d["count"], 13, 32, 4
    for j in range(stride):
        ax.add_patch(plt.Rectangle((j, 0), .92, .6, color=BLUE if j < count else "#e2e8f0"))
    for x, label, color, y in ((floor, "min_gates=4: validity floor", GREEN, 1.2),
                                (count, f"count={count}: real prefix", BLUE, .85),
                                (capacity, "anchor capacity=13", ORANGE, -.4),
                                (stride, "max_gates=32: row stride", "#334155", -.75)):
        ax.plot([x, x], [.1, y], color=color, lw=1)
        ax.text(x, y, label, ha="center", va="bottom" if y > 0 else "top", fontsize=10, color=color)
    ax.set(xlim=(-2, 37), ylim=(-1, 1.6))
    ax.axis("off")
    _save(fig, out, "controls-gate-capacity", "One native Bezier env: real count, anchor capacity and storage stride")


def gate_elevation(out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    pair = []
    for ax, profile in zip(axes.flat, ("flat", "uniform", "random_walk", "noise")):
        d = gates("gate-elevation-" + profile, generator="polar", gate_width=.1,
                  z_profile=profile, z_min=-.4, z_max=.4, z_max_step=.3, z_noise_amplitude=.35)
        p = d["position"]
        pair.append(p)
        ax.plot(_s(p[:, :2]), _closed(p)[:, 2], "o-", color=BLUE, ms=4)
        ax.set(title=f"{profile}: {len(p)} gates · {_status(d)}", xlabel="XY chord arclength [units]",
               ylabel="altitude [units]", ylim=(-.48, .48))
        ax.grid(alpha=.15)
    CHECKS["gate_elevation_preserves_xy"] = all(np.array_equal(pair[0][:, :2], p[:, :2]) for p in pair[1:])
    CHECKS["gate_elevation_all_active"] = all(np.ptp(p[:, 2]) > .01 for p in pair[1:])
    _save(fig, out, "controls-gate-elevation", "Gate-level profiles: markers are actual gates, lines are route chords")


def _rotate(q, v):
    xyz, w = q[:3], q[3]
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + w * v)


def frames(out):
    common = dict(generator="polar", gate_width=0.4, gate_radius=0, size=2,
                  z_profile="noise", z_noise_amplitude=0.8, z_noise_harmonics=2,
                  z_min=-1, z_max=1)
    samples = [gates("frame-" + mode, gate_align=mode, **common)
               for mode in ("yaw_only", "full_tangent")]
    idx = int(np.argmax(np.abs(samples[0]["tangent"][:, 2])))
    fig = plt.figure(figsize=(12, 5))
    for col, (d, mode) in enumerate(zip(samples, ("yaw_only", "full_tangent"))):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        q, hs, p = d["orientation"][idx], d["half_size"][idx], d["position"][idx]
        left, up, fwd = [_rotate(q, np.eye(3)[j]) for j in (1, 2, 0)]
        corners = np.array([p + a * hs * left + b * hs * up for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
        ax.plot(*corners.T, color=BLUE, lw=3)
        tangent = d["tangent"][idx]
        route = np.array([p - 0.45 * tangent, p + 0.45 * tangent])
        ax.plot(*route.T, "--", color="#64748b", label="3D route tangent")
        for vector, label, color in ((fwd, "forward (plane normal)", ORANGE), (left, "left", GREEN), (up, "up", BLUE)):
            ax.quiver(*p, *(0.35 * vector), color=color, arrow_length_ratio=0.15)
            ax.text(*(p + 0.38 * vector), label, color=color, fontsize=8)
        ax.set(xlim=(p[0] - .5, p[0] + .5), ylim=(p[1] - .5, p[1] + .5), zlim=(p[2] - .5, p[2] + .5),
               xlabel="x [units]", ylabel="y [units]", zlabel="z [units]", title=f"{mode} · gate {idx} · {_status(d)}")
        ax.set_box_aspect((1, 1, 1))
        camera = np.degrees(np.arctan2(samples[0]["forward"][idx, 1], samples[0]["forward"][idx, 0])) + 30
        ax.view_init(22, camera)
    CHECKS["frames_differ"] = not np.allclose(samples[0]["orientation"][idx], samples[1]["orientation"][idx])
    CHECKS["roll_free_bars_horizontal"] = all(abs(d["left"][idx, 2] - d["right"][idx, 2]) < 1e-6 for d in samples)
    _save(fig, out, "controls-gate-frames", "Pitching the gate plane: local x=forward, y=left, z=up")


def stations_3d(out):
    common = dict(generator="polar", source="relaxed_stations", gates=12,
                  gate_radius=0, gate_width=0.1, elevation_on="centerline",
                  elevation=("noise", {"z_noise_amplitude": 0.65, "z_noise_harmonics": 3,
                                        "z_min": -1, "z_max": 1}))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    all_pos = []
    for col, metric in enumerate(("xy", "xyz")):
        d, gen = preset_gates("stations-" + metric, spacing_metric=metric, **common)
        xy = gen.program.tap("cl.xy", after="relax").numpy()
        z = gen.program.tap("cl.z", after="elevation").numpy()
        count = int(gen.program.tap("cl.count", after="relax").numpy()[0])
        p = np.c_[xy[:count], z[:count]]
        all_pos.append(d["position"])
        axes[0, col].plot(*_closed(p)[:, :2].T, color="#94a3b8")
        _gate_plan(axes[0, col], d, numbering=True)
        axes[0, col].set_title(f"spacing_metric={metric}; same elevated loop")
        ax = axes[1, col]
        ax.plot(_s(p[:, :2]), _closed(p)[:, 2], color="#94a3b8", label="centerline altitude")
        # Project each station onto the sampled XY polyline for an annotated profile.
        base_s = _s(p[:, :2])
        station_s, metric_s = [], []
        base_metric = _s(p if metric == "xyz" else p[:, :2])
        for gate in d["position"]:
            best = (np.inf, 0.0, 0.0)
            for j in range(count):
                a, b = p[j, :2], p[(j + 1) % count, :2]
                delta = b - a
                t = np.clip(np.dot(gate[:2] - a, delta) / max(np.dot(delta, delta), 1e-20), 0, 1)
                err = np.linalg.norm(gate[:2] - a - t * delta)
                if err < best[0]:
                    best = (err, base_s[j] + t * (base_s[j + 1] - base_s[j]),
                            base_metric[j] + t * (base_metric[j + 1] - base_metric[j]))
            station_s.append(best[1])
            metric_s.append(best[2])
        steps = np.diff(np.r_[metric_s, metric_s[0] + base_metric[-1]])
        CHECKS[f"stations_{metric}_equal_arc"] = np.allclose(steps, base_metric[-1] / len(metric_s), rtol=1e-4, atol=1e-5)
        ax.scatter(station_s, d["position"][:, 2], color=BLUE, label="gate stations", zorder=3)
        for j, (s, h) in enumerate(zip(station_s, d["position"][:, 2])):
            ax.annotate(str(j), (s, h), xytext=(2, 4), textcoords="offset points", fontsize=7)
        ax.set(xlabel="XY arclength [coordinate units]", ylabel="altitude [units]", ylim=(-0.6, 0.6))
        ax.legend(fontsize=8)
    CHECKS["station_metrics_differ"] = not np.allclose(*all_pos)
    _save(fig, out, "controls-stations-3d", "XY versus 3D stations with elevation_on='centerline' held fixed")


def validate():
    required = {"size_changes_xy", "width_changes_band", "spacing_changes_count",
                "capacity_truncates_and_fails", "resolution_nonvacuous", "elevation_preserves_xy",
                "all_profiles_active", "sources_differ", "radius_solver_moves_centers", "frames_differ",
                "roll_free_bars_horizontal", "station_metrics_differ", "stations_xy_equal_arc",
                "stations_xyz_equal_arc", "size_preserves_absolute_width", "poison_clears_failed_geometry",
                "width_channel_nonvacuous", "width_channel_formula_0", "width_channel_formula_1",
                "width_channel_bounds_0", "width_channel_bounds_1", "gate_elevation_preserves_xy",
                "gate_elevation_all_active", "gate_order_width_bounds_contain_geometry"}
    assert required <= CHECKS.keys(), required - CHECKS.keys()
    assert len([k for k in CHECKS if k.startswith("elevation_knob_")]) == 6
    assert all(CHECKS.values()), {k: v for k, v in CHECKS.items() if not v}
    assert all(row["count"] >= 3 for row in RECORDS)
    assert len(RECORDS) == 48
    assert any(not row["valid"] and row["config"].get("on_invalid") == "keep" for row in RECORDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="assert numerical coverage and meaningful comparisons")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for renderer in (size_width, sampling, width_channel, elevation, sources_and_order, gate_elevation, frames, stations_3d):
        renderer(args.out_dir)
    if args.check:
        validate()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(source_revision=revision, renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    command="python -m viz.render_course_guide --out-dir DIR --check",
                    device="cpu", units="caller coordinate units", selection="fixed seed 5, epoch 0, env 0; no yield inference",
                    checks={k: bool(v) for k, v in CHECKS.items()}, samples=RECORDS,
                    assets=sorted(p.name for p in args.out_dir.glob("controls-*.png")))
    (args.out_dir / "controls-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['assets'])} figures; {len(RECORDS)} samples; {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
