"""Compare first-stage generators on quality / diversity / speed (dev tool — NOT runtime).

Runs each registered generator over a fixed seed suite through the full pipeline, reads the
pre-relax centerline (scratch.cs_center) and post-relax Track in ONE pass, and computes
metrics host-side in numpy. Characterizes; never gates.

    .venv/bin/python -m benchmarks.compare_generators            # all generators, cpu, E=4096
    .venv/bin/python -m benchmarks.compare_generators --cuda --E 8192
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src import generator_registry
from benchmarks import track_metrics as tm

_COMPACTNESS_DEGENERATE_THRESHOLD = 0.90
_COMPACTNESS_DEGENERATE_RATE_MAX = 0.25


def _real_points(flat_xy: np.ndarray, e: int, n_max: int, count: int) -> np.ndarray:
    """Slice env e's first `count` real points from a flat [E*n_max, 2] numpy array."""
    base = e * n_max
    return flat_xy[base:base + count]


def run_generator(name, seed_base, E, base_config) -> dict:
    cfg = dataclasses.replace(base_config, generator=name, num_envs=E)
    n_max = int(cfg.N_max)
    rng = PerEnvSeededRNG(seeds=int(seed_base), num_envs=E, device=str(cfg.device))
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)
    if "cuda" in str(cfg.device):
        wp.synchronize()

    # Read back once. cs_center is the pre-relax constant-spacing centerline (XPBD writes a
    # separate `relaxed` buffer, so cs_center survives the run). Track.center is post-relax.
    pre = wp.to_torch(gen._scratch.cs_center).cpu().numpy()
    post = wp.to_torch(track.center).cpu().numpy()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(track.count).cpu().numpy().astype(int)

    lengths, compactness, peak_k, lap_times, chicanes, straights = [], [], [], [], [], []
    compactness_degenerate = 0
    pre_self_int = 0
    disp_sum, disp_pts = 0.0, 0
    for e in range(E):
        c = int(count[e])
        if c < 4:
            continue
        post_e = _real_points(post, e, n_max, c)
        pre_e = _real_points(pre, e, n_max, c)
        if not np.isfinite(post_e).all():
            continue
        if np.isfinite(pre_e).all() and tm.self_intersects(pre_e):
            pre_self_int += 1
        if valid[e]:
            lengths.append(tm.perimeter(post_e))
            cpt = tm.compactness(post_e)
            compactness.append(cpt)
            if cpt > _COMPACTNESS_DEGENERATE_THRESHOLD:
                compactness_degenerate += 1
            rl = tm.racing_line_proxy(post_e)
            peak_k.append(rl["peak_curvature"])
            lap_times.append(rl["lap_time"])
            chicanes.append(tm.chicane_count(post_e))
            straights.append(tm.straight_fraction(post_e))
        if np.isfinite(pre_e).all():
            disp_sum += float(np.linalg.norm(post_e - pre_e, axis=1).sum())
            disp_pts += c

    # Warm timing of generate() alone.
    for _ in range(2):
        gen.generate(E)
    if "cuda" in str(cfg.device):
        wp.synchronize()
    reps = 5
    t0 = time.time()
    for _ in range(reps):
        gen.generate(E)
    if "cuda" in str(cfg.device):
        wp.synchronize()
    gen_ms = (time.time() - t0) / reps * 1e3

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    mean_compactness = _mean(compactness)
    comp_arr = np.array(compactness) if compactness else np.array([float("nan")])
    degenerate_rate = (
        compactness_degenerate / len(compactness) if compactness else float("nan")
    )
    shape_variety_pass = float(
        bool(compactness)
        and degenerate_rate < _COMPACTNESS_DEGENERATE_RATE_MAX
        and mean_compactness < _COMPACTNESS_DEGENERATE_THRESHOLD
    )

    return {
        "generator": name,
        "yield": float(valid.mean()),
        "pre_relax_self_intersection_rate": pre_self_int / E,
        "xpbd_displacement": (disp_sum / disp_pts) if disp_pts else float("nan"),
        "mean_length": _mean(lengths),
        "mean_compactness": mean_compactness,
        "compactness_p10": float(np.percentile(comp_arr, 10)),
        "compactness_p50": float(np.percentile(comp_arr, 50)),
        "compactness_p90": float(np.percentile(comp_arr, 90)),
        "compactness_degenerate_rate": degenerate_rate,
        "shape_variety_pass": shape_variety_pass,
        "mean_chicanes": _mean(chicanes),
        "straight_frac": _mean(straights),
        "peak_curvature": _mean(peak_k),
        "lap_time": _mean(lap_times),
        "gen_ms_per_call": gen_ms,
    }


def compare(names=None, seed_base=0, E=4096, base_config=None) -> list:
    if base_config is None:
        base_config = TrackGenConfig(num_envs=E)
    if names is None:
        names = generator_registry.available()
    return [run_generator(n, seed_base, E, base_config) for n in names]


def format_table(rows) -> str:
    cols = ["generator", "yield", "pre_relax_self_intersection_rate", "xpbd_displacement",
            "mean_length", "mean_compactness", "compactness_p50", "compactness_degenerate_rate",
            "shape_variety_pass", "mean_chicanes", "straight_frac",
            "peak_curvature", "lap_time", "gen_ms_per_call"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        cells = [str(r["generator"])] + [f"{r[c]:.4g}" for c in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--generators", nargs="*", default=None)
    a = ap.parse_args()
    cfg = TrackGenConfig(device="cuda" if a.cuda else "cpu", num_envs=a.E)
    rows = compare(a.generators, seed_base=a.seed, E=a.E, base_config=cfg)
    print(format_table(rows))
