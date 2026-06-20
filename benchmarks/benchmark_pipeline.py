"""End-to-end benchmark of the pure-Warp track-generation pipeline.

Drives ``TrackGenerator.generate()`` (generation -> resample -> XPBD relax ->
inflate, ALL NVIDIA Warp kernels) at scale and reports validity yield, wall-clock,
and peak GPU memory. With ``--graph`` it also captures the WHOLE pipeline as one
CUDA graph and times replay (the deployable, GPU-resident path).

    .venv/bin/python -m benchmarks.benchmark_pipeline                # auto device, E=8192
    .venv/bin/python -m benchmarks.benchmark_pipeline --E 2048 --cpu
    .venv/bin/python -m benchmarks.benchmark_pipeline --graph        # + CUDA-graph replay
"""
from __future__ import annotations

import argparse
import time

import torch
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG


def run_pipeline_benchmark(E=8192, N=256, half_width=0.03, scale=1.0, device="cuda",
                           relax_iters=150, max_regen_iters=10, seed=0, graph=False, reps=3):
    """Benchmark the end-to-end Warp pipeline. Returns a list of metric rows."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    cfg = TrackGenConfig(device=device, num_envs=E, num_points=N, half_width=half_width,
                         scale=scale, relax_iters=relax_iters, max_regen_iters=max_regen_iters)
    rng = PerEnvSeededRNG(seeds=seed, num_envs=E, device=device)
    rows = []

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # --- eager path ---
    gen = TrackGenerator(cfg, rng)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen.generate(E)  # warmup (kernel compile / module load)
    _sync()
    t0 = time.time()
    for _ in range(reps):
        track = gen.generate(E)
    _sync()
    eager_s = (time.time() - t0) / reps
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else float("nan")
    rows.append({"mode": "eager", "device": device, "E": E, "N": N,
                 "valid_frac": wp.to_torch(track.valid).float().mean().item(),
                 "seconds": eager_s, "peak_gpu_mb": peak_mb, "capture_s": float("nan")})

    # --- single-CUDA-graph replay (cuda only) ---
    if graph and device == "cuda":
        # The TrackGenerator auto-captures on first generate() and replays on subsequent calls.
        rng2 = PerEnvSeededRNG(seeds=seed, num_envs=E, device=device)
        gen2 = TrackGenerator(cfg, rng2)
        t0 = time.time()
        captured_track = gen2.generate(E)  # first call: captures graph
        _sync()
        capture_s = time.time() - t0
        gen2.generate(E)  # warmup replay
        _sync()
        t0 = time.time()
        for _ in range(reps):
            rt = gen2.generate(E)
        _sync()
        replay_s = (time.time() - t0) / reps
        rows.append({"mode": "graph_replay", "device": device, "E": E, "N": N,
                     "valid_frac": wp.to_torch(rt.valid).float().mean().item(),
                     "seconds": replay_s, "peak_gpu_mb": float("nan"), "capture_s": capture_s})
    return rows


def _print_table(rows):
    cols = ["mode", "device", "E", "valid_frac", "seconds", "peak_gpu_mb", "capture_s"]
    print("  ".join(f"{c:>13}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>13.4g}" if isinstance(r[c], float) else f"{str(r[c]):>13}"
                        for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=8192)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--half_width", type=float, default=0.03)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--graph", action="store_true", help="also capture + time a single CUDA graph")
    a = ap.parse_args()
    rows = run_pipeline_benchmark(E=a.E, N=a.N, half_width=a.half_width,
                                  device="cpu" if a.cpu else "cuda", graph=a.graph)
    _print_table(rows)
