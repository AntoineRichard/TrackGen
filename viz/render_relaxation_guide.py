"""Render isolated XPBD control comparisons from actual fixed-seed CPU output.

Run ``python -m viz.render_relaxation_guide --out-dir /tmp/tg-relaxation``.
The selected seeds explain differences, not population yield. These figures show
the final resampled/inflated view, rather than internal solver bead buffers.
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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import warp as wp

from track_gen import TrackGenConfig, TrackGenerator, __version__

BLUE, GREY, INK = "#2563a6", "#8293a5", "#263445"
ROOT = Path(__file__).resolve().parents[1]
REGIME = dict(half_width=.5, scale=10., spacing=.30, N_max=384,
              relax_smooth_passes=0, relax_smooth_spacing_iters=0)
EXPERIMENTS = (
    ("iterations", "relax_iters", (0, 10, 50), 14,
     "Main-solve budget: movement can remain after ten sweeps"),
    ("margin", "relax_margin", (0., .15, .5), 14,
     "Separation margin: more requested room can distort the result at a fixed budget"),
    ("smoothing", "relax_smooth_passes", (0, 5, 20), 0,
     "Taubin smoothing: rounded detail can change without a new separation solve"),
)


def _diagnostics(points, half_width):
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    incoming = points - np.roll(points, 1, axis=0)
    chord = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    cross = np.abs(incoming[:, 0]*edges[:, 1] - incoming[:, 1]*edges[:, 0])
    numerator = np.linalg.norm(incoming, axis=1)*lengths*np.linalg.norm(chord, axis=1)
    radius = np.divide(numerator, 2*cross, out=np.full_like(cross, np.inf),
                       where=cross > 1e-12)
    n = len(points)
    delta = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    band = max(1, int(np.rint(2*half_width/lengths.mean())))
    nonlocal_pairs = np.minimum(delta, n-delta) > band
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    return dict(min_local_radius=float(radius.min()), band=band,
                min_nonlocal_bead_distance=float(distance[nonlocal_pairs].min()),
                spacing_cv=float(lengths.std()/lengths.mean()))


def _sample(seed, field, value):
    settings = dict(REGIME)
    settings[field] = value
    config = TrackGenConfig(num_envs=1, device="cpu", on_invalid="keep", **settings)
    generator = TrackGenerator(config, seeds=seed)
    result = generator.generate()
    count = int(result.count.numpy()[0])
    sample = dict(seed=seed, epoch=0,
                  config=json.loads(json.dumps(dataclasses.asdict(config))), count=count,
                  valid=bool(result.valid.numpy()[0]),
                  fail_mask=int(result.fail_mask.numpy()[0]),
                  raw=generator.program.tap("raw.xy").numpy().copy())
    for name in ("center", "inner", "outer"):
        sample[name] = getattr(result, name).numpy()[:count, :2].copy()
    assert count >= 3 and all(np.isfinite(sample[k]).all()
                              for k in ("raw", "center", "inner", "outer"))
    sample["diagnostics"] = _diagnostics(sample["center"], config.half_width)
    return sample


def collect_comparisons():
    """Keep seeds and all settings fixed within each one-control comparison."""
    return [dict(name=name, field=field, title=title,
                 samples=[_sample(seed, field, value) for value in values])
            for name, field, values, seed, title in EXPERIMENTS]


def _closed(points):
    return np.vstack([points, points[0]])


def _road(ax, sample, baseline=None):
    ax.add_patch(Polygon(np.vstack([sample["outer"], sample["inner"][::-1]]),
                         closed=True, facecolor="#e4e9ee", edgecolor="none"))
    for name in ("inner", "outer"):
        ax.plot(*_closed(sample[name]).T, color=GREY, lw=.9)
    if baseline is not None:
        ax.plot(*_closed(baseline).T, color="#dc791e", ls="--", lw=1.4,
                label="0-pass centerline")
    ax.plot(*_closed(sample["center"]).T, color=BLUE, lw=1.3)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("x [distance units]", fontsize=9)
    ax.set_ylabel("y [distance units]", fontsize=9)


def _render_comparison(row, out_dir):
    samples = row["samples"]
    zoom = row["name"] == "smoothing"
    fig, axes = plt.subplots(2 if zoom else 1, 3,
                             figsize=(12, 8.4 if zoom else 5.1), squeeze=False)
    points = np.vstack([sample[name] for sample in samples for name in ("outer", "inner")])
    lo, hi = points.min(axis=0), points.max(axis=0)
    midpoint, span = (lo+hi)/2, max(hi-lo)*.58
    default = getattr(TrackGenConfig(), row["field"])
    max_delta = max(float(np.linalg.norm(sample["center"]-samples[0]["center"], axis=1).max())
                    for sample in samples[1:])
    assert max_delta > .01, f"vacuous {row['field']} comparison"
    if zoom:
        index = np.linalg.norm(samples[-1]["center"]-samples[0]["center"], axis=1).argmax()
        center = (samples[-1]["center"][index]+samples[0]["center"][index])/2
        radius = 1.1
    for column, sample in enumerate(samples):
        ax = axes[0, column]
        _road(ax, sample)
        ax.set_xlim(midpoint[0]-span, midpoint[0]+span)
        ax.set_ylim(midpoint[1]-span, midpoint[1]+span)
        value = sample["config"][row["field"]]
        default_label = " (default)" if value == default else ""
        ax.set_title(f"{row['field']}={value:g}{default_label}", fontsize=11)
        d = sample["diagnostics"]
        status = "valid" if sample["valid"] else "INVALID"
        ax.text(.01, -.32,
                f"seed {sample['seed']} · n={sample['count']} · {status}\n"
                f"fail=0x{sample['fail_mask']:02x} · min nonlocal bead d={d['min_nonlocal_bead_distance']:.3f}",
                transform=ax.transAxes, fontsize=9,
                color="#237344" if sample["valid"] else "#b83e35")
        if zoom:
            ax.add_patch(Rectangle(center-radius, 2*radius, 2*radius,
                                   fill=False, edgecolor="#dc791e", lw=1.2))
            detail = axes[1, column]
            _road(detail, sample, samples[0]["center"])
            detail.set_xlim(center[0]-radius, center[0]+radius)
            detail.set_ylim(center[1]-radius, center[1]+radius)
            detail.set_title(f"Same zoom · loop min radius={d['min_local_radius']:.3f}", fontsize=10)
    fig.suptitle(row["title"], fontsize=14)
    if zoom:
        footer = "Blue: current centerline; orange dashed: 0-pass baseline in the same fixed zoom.\n50 main sweeps; spacing polish disabled. Smoothing does not re-enforce separation."
    else:
        footer = "Blue: final centerline; grey: road borders. Both smoothing and spacing polish disabled.\nOne control changes; each output includes final uniform resampling + inflation."
    fig.text(.5, .025, footer+"\nSelected CPU examples; failed geometry explicitly kept. Validity uses sampled checks.",
             ha="center", fontsize=9, color=INK)
    fig.subplots_adjust(left=.06, right=.98, top=.86 if not zoom else .91,
                        bottom=.33 if not zoom else .15, wspace=.35, hspace=.65)
    asset = f"relaxation-{row['name']}.png"
    fig.savefig(out_dir / asset, dpi=160, facecolor="white")
    plt.close(fig)
    return dict(field=row["field"], asset=asset, max_center_displacement=max_delta,
                samples=[{k: v for k, v in sample.items()
                          if k not in ("center", "inner", "outer", "raw")} |
                         {"finite": True} for sample in samples])


def render_guide(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = collect_comparisons()
    source = dict(version=__version__, warp_version=wp.__version__,
                  numpy_version=np.__version__, device="cpu",
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                     text=True).strip(),
                  runtime_reference=subprocess.run(
                      ["git", "rev-parse", "--verify", "--quiet", f"v{__version__}^{{commit}}"],
                      cwd=ROOT, capture_output=True, text=True).stdout.strip() or None,
                  runtime_worktree_changed=subprocess.run(
                      ["git", "diff", "--quiet", "HEAD", "--", "track_gen"], cwd=ROOT).returncode != 0,
                  solver_sha256=hashlib.sha256((ROOT/"track_gen/_src/warp_relax.py").read_bytes()).hexdigest(),
                  runtime_source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                         for name in ("track_gen/seeds.py", "track_gen/ops/sampling.py",
                                                      "track_gen/_src/warp_generate.py",
                                                      "track_gen/_src/warp_pipeline.py")})
    manifest = dict(source=source, command="python -m viz.render_relaxation_guide --out-dir DIR",
                    selection=dict(seeds=[14, 0], claims_yield=False,
                                   method="selected for visible same-seed changes after inspecting seeds 0 through 23",
                                   display="final uniform-resampled and inflated output; not internal solver buffers"),
                    comparisons=[_render_comparison(row, out_dir) for row in comparisons])
    (out_dir / "relaxation-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    render_guide(args.out_dir)


if __name__ == "__main__":
    main()
