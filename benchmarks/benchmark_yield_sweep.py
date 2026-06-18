# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""E=8192 end-to-end pipeline study: what lifts relaxed-valid yield, and at what cost?

Sweeps the pure-Warp pipeline (generate_tracks_warp) at E=8192 over four levers, all in
the physically-anchored 1 m-track / ~20 m-box regime (half_width=0.5, scale=10 -> the
~0.68-yield proportions):

  * chain links   (num_points)          -- relax/validation resolution
  * relax iters   (relax_iters)          -- convergence budget
  * regen attempts(max_regen_iters)      -- generation re-draw budget
  * XPBD step size (relax_sep_relax / relax_spc_relax, relax_margin) -- over-relaxation

Reports per config: relaxed-valid yield (at the working resolution), end-to-end
wall-clock (s/call, warmed + synchronized), and peak GPU memory. Sequential by design
(one process owns the GPU; parallel runs would corrupt timings).

    .venv/bin/python -m benchmarks.benchmark_yield_sweep

NOTE: "yield" is resolution-relative -- the thickness gate's curvature term is sampled at
num_points, so yields at different link counts are NOT directly comparable (see the
report / analysis). Iter, regen and PBD-step sweeps are at fixed 256 links, so those
comparisons are fair.
"""
from __future__ import annotations

import time

import torch
import warp as wp

wp.init()

from track_gen.types import TrackGenConfig
from track_gen import warp_pipeline as wpl

E = 8192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF_WIDTH = 0.5      # 1 m track width
SCALE = 10.0          # ~20 m box  (1m/20m proportions -> ~0.68 baseline yield)
REPS = 2              # E=8192 timing is low-variance


def bench(links=256, iters=150, regen=10, sr=1.0, pr=1.0, br=1.5, margin=0.15, seed=0):
    cfg = TrackGenConfig(
        num_envs=E, num_points=links, half_width=HALF_WIDTH, scale=SCALE,
        relax_iters=iters, max_regen_iters=regen,
        relax_sep_relax=sr, relax_spc_relax=pr, relax_bend_relax=br, relax_margin=margin,
        device=DEVICE,
    )
    seeds = (torch.arange(E, dtype=torch.int32) + seed).to(DEVICE)

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    wpl.generate_tracks_warp(cfg, seeds)  # warmup
    sync()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(REPS):
        track = wpl.generate_tracks_warp(cfg, seeds)
    sync()
    sec = (time.time() - t0) / REPS
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else float("nan")
    return {"links": links, "iters": iters, "regen": regen, "sr": sr, "pr": pr,
            "margin": margin, "yield": track.valid.float().mean().item(),
            "sec": sec, "peak_mb": peak}


# Each dict overrides bench() defaults. Baseline first, then one lever varied at a time.
CONFIGS = [
    {},                                              # baseline 256/150/10, steps 1.0, margin .15
    # chain-links (iters=150)
    {"links": 128}, {"links": 384}, {"links": 512},
    # relax-iters (links=256)
    {"iters": 50}, {"iters": 300}, {"iters": 600},
    # regen attempts (links=256)
    {"regen": 20}, {"regen": 40},
    # XPBD step size (over-relaxation), links=256, iters=150
    {"sr": 1.5, "pr": 1.5}, {"sr": 2.0, "pr": 2.0}, {"sr": 2.5, "pr": 2.5},
    {"margin": 0.30}, {"sr": 2.0, "pr": 2.0, "margin": 0.30},
    {"sr": 2.0, "pr": 2.0, "iters": 50},             # big steps + few iters (speed/yield)
    # combined high-link probe
    {"links": 512, "iters": 300, "regen": 20},
]


def main():
    print(f"# E={E}  device={DEVICE}  half_width={HALF_WIDTH} scale={SCALE} "
          f"(1m track / ~20m box)  reps={REPS}")
    rows = []
    for ov in CONFIGS:
        r = bench(**ov)
        rows.append(r)
        print(f"ROW links={r['links']:>4} iters={r['iters']:>4} regen={r['regen']:>3} "
              f"sr={r['sr']:.1f} pr={r['pr']:.1f} margin={r['margin']:.2f}  "
              f"yield={r['yield']:.3f}  sec={r['sec']:.3f}  peak_mb={r['peak_mb']:.0f}", flush=True)

    print("\n=== TABLE ===")
    hdr = f"{'links':>6}{'iters':>6}{'regen':>6}{'sr':>5}{'pr':>5}{'margin':>7}{'yield':>8}{'sec':>8}{'mb':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['links']:>6}{r['iters']:>6}{r['regen']:>6}{r['sr']:>5.1f}{r['pr']:>5.1f}"
              f"{r['margin']:>7.2f}{r['yield']:>8.3f}{r['sec']:>8.3f}{r['peak_mb']:>7.0f}")


if __name__ == "__main__":
    main()
