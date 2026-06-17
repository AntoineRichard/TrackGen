# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the three relaxation backends on a large batch (default E=8192).

Reports per backend: validity yield (thickness + zero border/centerline crossings),
wall-clock, peak GPU memory, and shape-quality metrics (displacement, clearance CV,
max curvature). Runs on GPU (primary) and CPU (fallback). Run directly:

    .venv/bin/python -m benchmarks.benchmark_relaxation            # auto device, E=8192
    .venv/bin/python -m benchmarks.benchmark_relaxation --E 2048 --cpu
"""
from __future__ import annotations
import argparse, time
import torch

from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry, inflation
from track_gen.generators import BezierCenterlineGenerator


class _Cfg:
    """Minimal stand-in exposing the fields relaxation._band reads."""
    def __init__(self, half_width, relax_band=None):
        self.half_width = half_width
        self.relax_band = relax_band


def _gen_simple_tracks(E, N, scale, device, seed):
    """Generate E simple, arc-length-uniform Bezier centerlines (needs warp RNG)."""
    import warp as wp; wp.init()
    from track_gen.rng_utils import PerEnvSeededRNG
    kept = []
    s = seed
    while len(kept) < E:
        B = min(2048, 2 * (E - len(kept)) + 256)
        seeds = torch.arange(B, dtype=torch.int32) + s
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=B, device=device)
        rng.set_seeds(seeds, ids=torch.arange(B, dtype=torch.int32))
        cfg = TrackGenConfig(device=device, num_envs=B, scale=scale, num_points=N,
                             max_regen_iters=20, relax_enable=False)
        cl = BezierCenterlineGenerator(cfg, rng).generate(torch.arange(B, device=device))
        res = inflation._resample_stage(cl, cfg)            # arc-length uniform [B,N,2]
        ok = torch.isfinite(res.center).all(dim=(1, 2)) & cl.valid
        for e in torch.where(ok)[0].tolist():
            kept.append(res.center[e])
            if len(kept) >= E:
                break
        s += B
    return torch.stack(kept[:E], dim=0).to(device)


def _quality(center0, relaxed, half_width):
    band = relaxation._band(center0, _Cfg(half_width=half_width))
    th = geometry.thickness(relaxed, band)
    target = 0.98 * half_width
    # Inflate at constant width to count border crossings (orientation is irrelevant
    # for crossing counts, so the plain +/- Nrm offset suffices).
    _, Nrm = geometry.tangents_normals(relaxed)
    outer = relaxed + half_width * Nrm
    inner = relaxed - half_width * Nrm
    border_x = geometry.self_intersections(outer) + geometry.self_intersections(inner)
    center_x = geometry.self_intersections(relaxed)
    valid = (th >= target) & (border_x == 0) & (center_x == 0)
    disp = torch.linalg.norm(relaxed - center0, dim=-1).mean(dim=1)
    kappa = geometry.menger_curvature(relaxed)
    crad = 1.0 / kappa.clamp_min(1e-12)
    hc = torch.minimum(crad, 0.5 * geometry.separation_min(relaxed, band).unsqueeze(1))
    cv = (hc.std(dim=1) / hc.mean(dim=1).clamp_min(1e-12))
    return {
        "valid_frac": valid.float().mean().item(),
        "thickness_med": th.median().item(),
        "disp_med": disp.median().item(),
        "clearance_cv_med": cv.median().item(),
        "kmax_med": kappa.amax(dim=1).median().item(),
    }


def run_benchmark(E=8192, N=256, half_width=0.03, scale=1.0, device="cuda",
                  solvers=("xpbd", "energy", "tp_sobolev"), chunk=None,
                  relax_iters=150, energy_steps=800, tp_iters=100, seed=20, smooth=False):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    center0 = _gen_simple_tracks(E, N, scale, device, seed)
    rows = []
    for solver in solvers:
        cfg = TrackGenConfig(device=device, num_envs=E, num_points=N, half_width=half_width,
                             relax_solver=solver, relax_chunk_size=chunk, relax_iters=relax_iters,
                             energy_steps=energy_steps, tp_iters=tp_iters, smooth_finish=smooth)
        if device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        relaxed = relaxation.relax(center0, cfg)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else float("nan")
        q = _quality(center0, relaxed, half_width)
        rows.append({"solver": solver, "device": device, "E": E, "N": N,
                     "seconds": seconds, "peak_gpu_mb": peak_mb, **q})
    return rows


def _print_table(rows):
    cols = ["solver", "device", "valid_frac", "seconds", "peak_gpu_mb",
            "thickness_med", "disp_med", "clearance_cv_med", "kmax_med"]
    print("  ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>14.4g}" if isinstance(r[c], float) else f"{str(r[c]):>14}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=8192)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--half_width", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smooth", action="store_true")
    a = ap.parse_args()
    rows = run_benchmark(E=a.E, N=a.N, half_width=a.half_width, chunk=a.chunk,
                         device="cpu" if a.cpu else "cuda", smooth=a.smooth)
    _print_table(rows)
