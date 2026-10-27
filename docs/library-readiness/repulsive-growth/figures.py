"""Verify source-array hashes/metrics and redraw the internal diagnostic figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import array_record, curve_metrics, road_disc_gaps

NAMES = ("baseline", "tau02", "tau01", "keep_obstacles")


def load_case(directory, name):
    record = json.loads((directory / (name + ".json")).read_text())
    with np.load(directory / record["array_archive"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, expected in record["arrays"].items():
        if array_record(arrays[key]) != expected:
            raise ValueError(f"{name}/{key}: source-array hash, dtype or shape changed")
    for key, summary in (("raw", "raw"), ("center", "final_road_center")):
        actual = curve_metrics(arrays[key], arrays["circles"], arrays["radii"],
                               record["domain_radius"])
        for field, value in actual.items():
            expected = record[summary][field]
            if value is None and expected is None:
                continue
            if not np.isclose(value, expected, rtol=1e-10, atol=1e-10):
                raise ValueError(f"{name}/{key}: metric {field} differs from source array")
    actual = road_disc_gaps(arrays["inner"], arrays["outer"], arrays["circles"], arrays["radii"])
    if not np.allclose(actual, record["road_disc_gaps"], rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name}: road-band overlap does not match returned arrays")
    for frame in record["frames"] + record["substeps"]:
        if array_record(arrays[frame["xy_key"]]) != frame["source_array"]:
            raise ValueError(f"{name}: frame source-array hash changed")
    return record, arrays


def draw(cases, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle

    def loop(ax, xy, **kwargs):
        xy = np.vstack((xy, xy[:1]))
        ax.plot(xy[:, 0], xy[:, 1], **kwargs)

    def rings(ax, record, arrays, active):
        for center, radius in zip(arrays["circles"], arrays["radii"]):
            ax.add_patch(Circle(center, radius,
                                facecolor="#cce4cf" if active else "#e2e2e2",
                                edgecolor="#417b48" if active else "#9a9a9a", lw=.9))
        ax.add_patch(Circle((0, 0), record["domain_radius"], fill=False,
                            edgecolor="#417b48", lw=.6, linestyle=":"))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    extent = max(float(np.abs(arrays[key]).max()) for _, arrays in cases.values()
                 for key in ("raw", "inner", "outer")) * 1.10
    titles = ("Current default", "Only tau = 0.2", "Only tau = 0.1", "Only retain disc forces")
    for column, (name, title) in enumerate(zip(NAMES, titles)):
        record, arrays = cases[name]
        active = not (record["config"]["repulsive_deactivate_obstacles"] and record["reached"])
        for ax in axes[:, column]:
            rings(ax, record, arrays, active)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
        loop(axes[0, column], arrays["raw"], color="#176ba2", lw=1.1, marker=".", markersize=1.6)
        raw = record["raw"]
        axes[0, column].set_title(title + "\nmax turn %.1f° / p95 %.1f°" %
                                  (raw["turn_max_deg"], raw["turn_p95_deg"]))
        axes[0, column].text(.5, -.05, "Disc gap %+.3f; wall gap %+.3f" %
                             (raw["disc_seg_gap"], raw["wall_center_gap"]),
                             ha="center", transform=axes[0, column].transAxes)
        inner, outer = arrays["inner"], arrays["outer"]
        quads = np.stack((inner, outer, np.roll(outer, -1, axis=0),
                          np.roll(inner, -1, axis=0)), axis=1)
        axes[1, column].add_collection(PolyCollection(quads, facecolor="#c9d6e2",
                                                       edgecolor="none", alpha=.85))
        loop(axes[1, column], arrays["center"], color="#176ba2", lw=1.1)
        overlaps = sum(gap < 0 for gap in record["road_disc_gaps"])
        axes[1, column].text(.5, -.06, "Road overlaps %d / %d discs\nvalid = %s; stop %d; frozen = %s" %
                             (overlaps, len(arrays["radii"]), record["valid"],
                              record["iterations"], record["frozen"]),
                             ha="center", transform=axes[1, column].transAxes)
    fig.suptitle("Repulsive seed 7: actual raw curve (top) and finished road (bottom)\n"
                 "Green: active forces; grey: disabled discs. All views share scale.\n"
                 "tau = 0.1 is smoother but escapes the active wall; it is not a safe fix.", fontsize=14)
    fig.subplots_adjust(left=.03, right=.98, top=.81, bottom=.12, hspace=.18, wspace=.14)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "tuning-comparison.png", dpi=170)
    plt.close(fig)

    record, arrays = cases["baseline"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, iteration in ((axes[0, 0], 48), (axes[0, 1], 192)):
        frame = next(frame for frame in record["frames"] if frame["iteration"] == iteration)
        rings(ax, record, arrays, frame["inner_forces_active"])
        extent = record["domain_radius"] * 1.08
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
        loop(ax, arrays[frame["xy_key"]], color="#176ba2", lw=1, marker=".", markersize=2)
        ax.set_title("Default iteration %d, N=%d\nmax turn %.1f° / p95 %.1f°" %
                     (iteration, frame["n"], frame["turn_max_deg"], frame["turn_p95_deg"]))
    for name, label in (("baseline", "Default tau=.4"), ("tau02", "tau=.2"), ("tau01", "tau=.1")):
        frames = cases[name][0]["frames"]
        iterations = [frame["iteration"] for frame in frames]
        axes[1, 0].plot(iterations, [frame["turn_p95_deg"] for frame in frames], label=label)
        axes[1, 1].plot(iterations, [frame["disc_seg_gap"] for frame in frames], label=label)
    reached = next(frame["iteration"] for frame in record["frames"] if frame["reached"])
    axes[1, 0].set(xlabel="Iteration", ylabel="95th percentile vertex turn [degrees]",
                   title="Jaggedness exists in recorded coordinates")
    axes[1, 1].set(xlabel="Iteration", ylabel="Minimum segment-to-disc gap",
                   title=f"Default disables inner forces at iteration {reached}")
    axes[1, 1].axhline(0, color="black", lw=.6)
    axes[1, 1].axvline(reached, color="#777", linestyle="--", lw=.8)
    for ax in axes[1]:
        ax.legend(fontsize=9)
    fig.suptitle("Default seed 7: roughness and disc penetration are solver behavior", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out_dir / "default-diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cases = {name: load_case(args.input_dir, name) for name in NAMES}
    if not args.verify_only:
        draw(cases, args.out_dir or args.input_dir)
    print("Verified source hashes, returned-array metrics, and road-band intersections for four cases.")


if __name__ == "__main__":
    main()
