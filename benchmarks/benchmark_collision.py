"""Benchmark of ``track_gen.collision``: exact segments backend vs baked sdf.

Generates a real track batch, places ``B`` random oriented boxes per env near
the centerline, and reports:

- ``query()`` wall-clock for both backends — eager, and CUDA-graph replay on
  cuda (the deployable, GPU-resident path);
- the sdf ``bake()`` cost and grid memory (paid once per track regeneration);
- sdf accuracy against the exact segments backend on the same boxes
  (distance error, OOB-flag agreement, nearest/normal deviation).

    .venv/bin/python -m benchmarks.benchmark_collision                # auto device, E=8192
    .venv/bin/python -m benchmarks.benchmark_collision --E 2048 --cpu
    .venv/bin/python -m benchmarks.benchmark_collision --res 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import warp as wp

from track_gen._src import runtime
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig
from track_gen.collision import CollisionChecker


def _make_boxes(track, E, B, n_max, seed, device):
    """Random oriented boxes near each env's centerline (mixed in/out of band)."""
    r = np.random.default_rng(seed)
    center = track.center.numpy().reshape(E, n_max, 2)
    count = track.count.numpy().astype(np.int64)
    idx = r.integers(0, np.maximum(count, 1)[:, None], size=(E, B))
    pos = center[np.arange(E)[:, None], idx] + r.normal(0.0, 0.05, (E, B, 2))
    pos = np.nan_to_num(pos, nan=0.0).astype(np.float32)  # invalid envs: dummy origin
    yaw = r.uniform(0.0, 2.0 * np.pi, E * B).astype(np.float32)
    he = r.uniform(0.005, 0.05, (E * B, 2)).astype(np.float32)
    return (wp.array(pos.reshape(-1, 2), dtype=wp.vec2f, device=device),
            wp.array(yaw, dtype=wp.float32, device=device),
            wp.array(he, dtype=wp.vec2f, device=device))


def _time_eager(checker, pos, yaw, he, iters, warmup):
    for _ in range(warmup):
        checker.query(pos, yaw, he)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        checker.query(pos, yaw, he)
    wp.synchronize()
    return (time.perf_counter() - t0) / iters


def _time_graph(checker, pos, yaw, he, iters, device):
    """Capture one query into a CUDA graph and time its replay."""
    prev = runtime._CAPTURING
    runtime._CAPTURING = True
    try:
        checker.query(pos, yaw, he)  # warmup: modules loaded before capture
        wp.synchronize()
        with wp.ScopedCapture(device=device) as cap:
            checker.query(pos, yaw, he)
    finally:
        runtime._CAPTURING = prev
    wp.capture_launch(cap.graph)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    return (time.perf_counter() - t0) / iters


def run_collision_benchmark(E=8192, B=8, device="cuda", sdf_resolution=128,
                            iters=200, warmup=20, seed=42):
    """Benchmark both collision backends on one generated batch.

    Returns a dict of metrics (times in seconds; errors in world units).
    Accuracy stats are computed over valid envs only and are NaN when the
    batch contains no valid env.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    is_cuda = device == "cuda"

    cfg = TrackGenConfig(device=device, num_envs=E)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seed, num_envs=E, device=device))
    t0 = time.perf_counter()
    track = gen.generate()
    gen_s = time.perf_counter() - t0

    n_max = track.outer.shape[0] // E
    dev = str(track.outer.device)
    pos, yaw, he = _make_boxes(track, E, B, n_max, seed, dev)

    seg = CollisionChecker(track, max_boxes=B, method="segments")
    sdf = CollisionChecker(track, max_boxes=B, method="sdf",
                           sdf_resolution=sdf_resolution)
    t0 = time.perf_counter()
    sdf.bake()
    wp.synchronize()
    bake_s = time.perf_counter() - t0

    R = sdf_resolution
    m = {
        "device": device, "E": E, "B": B, "boxes": E * B, "sdf_resolution": R,
        "gen_s": gen_s, "bake_s": bake_s,
        "sdf_mem_mb": E * R * R * 5 / 1e6,  # float32 phi + int8 bid per texel
        "seg_eager_s": _time_eager(seg, pos, yaw, he, iters, warmup),
        "sdf_eager_s": _time_eager(sdf, pos, yaw, he, iters, warmup),
        "seg_graph_s": _time_graph(seg, pos, yaw, he, iters, dev) if is_cuda else None,
        "sdf_graph_s": _time_graph(sdf, pos, yaw, he, iters, dev) if is_cuda else None,
    }

    # --- accuracy: sdf vs exact on the same boxes, valid envs only ---
    exact = seg.query(pos, yaw, he).clone()
    approx = sdf.query(pos, yaw, he)
    mask = np.repeat(track.valid.numpy().astype(bool), B)
    if mask.any():
        d_ex = exact.distance.numpy()[mask]
        d_ap = approx.distance.numpy()[mask]
        err = np.abs(d_ap - d_ex)
        m["oob_rate"] = float(exact.oob.numpy()[mask].mean())
        m["flag_agreement"] = float(
            (exact.oob.numpy()[mask] == approx.oob.numpy()[mask]).mean())
        m["dist_err_mean"] = float(err.mean())
        m["dist_err_p99"] = float(np.percentile(err, 99))
        m["dist_err_max"] = float(err.max())
    else:
        m["oob_rate"] = m["flag_agreement"] = float("nan")
        m["dist_err_mean"] = m["dist_err_p99"] = m["dist_err_max"] = float("nan")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--E", type=int, default=8192)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--res", type=int, default=128, help="sdf grid resolution")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    m = run_collision_benchmark(E=args.E, B=args.B, sdf_resolution=args.res,
                                iters=args.iters, seed=args.seed,
                                device="cpu" if args.cpu else "cuda")

    def ms(x):
        return "     -" if x is None else f"{x * 1e3:9.3f} ms"

    print(f"\ndevice {m['device']}  E {m['E']}  B {m['B']}  "
          f"boxes/query {m['boxes']}  sdf {m['sdf_resolution']}^2 "
          f"({m['sdf_mem_mb']:.0f} MB)")
    print(f"generate(): {m['gen_s']:.2f}s   sdf bake(): {m['bake_s'] * 1e3:.1f} ms "
          f"(paid per regeneration)")
    print(f"\n{'backend':<10} {'eager/query':>13} {'graph/query':>13}")
    print(f"{'segments':<10} {ms(m['seg_eager_s']):>13} {ms(m['seg_graph_s']):>13}")
    print(f"{'sdf':<10} {ms(m['sdf_eager_s']):>13} {ms(m['sdf_graph_s']):>13}")
    print(f"\noob rate {m['oob_rate']:.3f}   flag agreement {m['flag_agreement']:.4f}")
    print(f"sdf |distance error| vs exact: mean {m['dist_err_mean']:.5f}  "
          f"p99 {m['dist_err_p99']:.5f}  max {m['dist_err_max']:.5f}")


if __name__ == "__main__":
    main()
