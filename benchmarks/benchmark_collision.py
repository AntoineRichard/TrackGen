"""Benchmark of ``track_gen.collision``: exact segments backend vs baked sdf.

Generates a real track batch, places ``B`` random oriented boxes per env near
the centerline, and reports:

- ``query()`` time for both backends — eager (wall-clock), and on cuda the
  GPU time per CUDA-graph replay, from events (the deployable, GPU-resident
  path);
- the sdf ``bake()`` cost and grid memory (paid once per track regeneration);
- sdf accuracy against the exact segments backend on the same boxes
  (distance error, OOB-flag agreement, nearest/normal deviation).

``--racing`` switches to the racing near-edge mode: bezier tracks at
``scale=17``, ``half_width=0.5``, ``N_max=512`` (seeds 1000000 + env), a
0.46 x 0.34 m car box placed near the edges (lateral offset U(-0.8, 0.8) m,
heading error U(-0.6, 0.6) rad) and R=256 grids. It reports the query time
per CUDA-graph replay (GPU events) and the sdf OOB-flag agreement with the
exact backend, with the false negatives and false positives counted.

    .venv/bin/python -m benchmarks.benchmark_collision                # auto device, E=8192
    .venv/bin/python -m benchmarks.benchmark_collision --E 2048 --cpu
    .venv/bin/python -m benchmarks.benchmark_collision --res 64
    .venv/bin/python -m benchmarks.benchmark_collision --racing       # E=4096, B=1, R=256
    .venv/bin/python -m benchmarks.benchmark_collision --racing --E 100 --B 1024 --no-timing
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import warp as wp

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen.collision import CollisionChecker


def _make_boxes(track, E, B, n_max, seed, device):
    """Random oriented boxes near each env's centerline (mixed in/out of band)."""
    r = np.random.default_rng(seed)
    center = track.center.numpy().reshape(E, n_max, 3)[..., :2]
    count = track.count.numpy().astype(np.int64)
    idx = r.integers(0, np.maximum(count, 1)[:, None], size=(E, B))
    pos = center[np.arange(E)[:, None], idx] + r.normal(0.0, 0.05, (E, B, 2))
    pos = np.nan_to_num(pos, nan=0.0).astype(np.float32)  # invalid envs: dummy origin
    pos3 = np.concatenate([pos.reshape(-1, 2),
                           np.zeros((E * B, 1), np.float32)], axis=1)
    yaw = r.uniform(0.0, 2.0 * np.pi, E * B)
    quat = np.zeros((E * B, 4), np.float32)
    quat[:, 2] = np.sin(0.5 * yaw)
    quat[:, 3] = np.cos(0.5 * yaw)
    he = r.uniform(0.005, 0.05, (E * B, 2)).astype(np.float32)
    return (wp.array(pos3, dtype=wp.vec3f, device=device),
            wp.array(quat, dtype=wp.quatf, device=device),
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


def _time_graph(query, iters, warmup, device):
    """Capture ``query()`` into a CUDA graph and time its replays with GPU
    events around each one: (median, p95) seconds per replay."""
    query()  # warmup: modules loaded before capture
    wp.synchronize()
    with wp.ScopedCapture(device=device) as cap:
        query()
    stream = wp.get_stream(device)
    for _ in range(warmup):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    events = [(wp.Event(device, enable_timing=True),
               wp.Event(device, enable_timing=True)) for _ in range(iters)]
    for a, b in events:
        stream.record_event(a)
        wp.capture_launch(cap.graph)
        stream.record_event(b)
    wp.synchronize()
    per = np.array([wp.get_event_elapsed_time(a, b) for a, b in events]) * 1e-3
    return float(np.median(per)), float(np.percentile(per, 95))


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
    wp.synchronize()  # utilities do not host-block: drain construction work first
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
        "seg_graph_s": None,
        "sdf_graph_s": None,
    }
    if is_cuda:  # GPU time per replay (median), from events
        m["seg_graph_s"] = _time_graph(lambda: seg.query(pos, yaw, he),
                                       iters, warmup, dev)[0]
        m["sdf_graph_s"] = _time_graph(lambda: sdf.query(pos, yaw, he),
                                       iters, warmup, dev)[0]

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


RACING = dict(generator="bezier", scale=17.0, half_width=0.5, N_max=512)
RACING_HALF_EXTENTS = (0.23, 0.17)  # a 0.46 x 0.34 m car box
RACING_SEED0 = 1_000_000


def racing_tracks(E, device, seed0=RACING_SEED0):
    """Racing-config track batch, env ``e`` seeded with ``seed0 + e``."""
    seeds = wp.array(np.arange(seed0, seed0 + E, dtype=np.int32),
                     dtype=wp.int32, device=device)
    cfg = TrackGenConfig(num_envs=E, device=device, **RACING)
    return TrackGenerator(cfg, PerEnvSeededRNG(seeds, E, device)).generate()


def near_edge_boxes(track, E, B, rng, half_extents=RACING_HALF_EXTENTS,
                    lateral=0.8, heading_err=0.6):
    """``B`` boxes per env near the edges: arc length s ~ U(0, L) along the
    centerline, lateral offset ~ U(-lateral, lateral), heading = segment
    heading + U(-heading_err, heading_err). Returns (position, orientation,
    half_extents) wp.arrays on the track's device."""
    n_max = track.center.shape[0] // E
    center = track.center.numpy().reshape(E, n_max, 3)[..., :2].astype(np.float64)
    count = track.count.numpy().astype(np.int64)
    xy = np.zeros((E, B, 2))
    yaw = np.zeros((E, B))
    for e in range(E):
        c = center[e, :max(count[e], 0)]
        if len(c) < 3:  # degenerate env: boxes at the origin, masked out later
            continue
        seg = np.roll(c, -1, axis=0) - c
        ln = np.linalg.norm(seg, axis=1)
        arc = np.cumsum(ln) - ln
        s = rng.uniform(0.0, ln.sum(), B)
        i = np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(c) - 1)
        hd = np.arctan2(seg[i, 1], seg[i, 0])
        p = c[i] + ((s - arc[i]) / np.maximum(ln[i], 1e-12))[:, None] * seg[i]
        right = np.stack([np.sin(hd), -np.cos(hd)], axis=1)
        xy[e] = p + rng.uniform(-lateral, lateral, B)[:, None] * right
        yaw[e] = hd + rng.uniform(-heading_err, heading_err, B)
    n = E * B
    pos = np.zeros((n, 3), np.float32)
    pos[:, :2] = xy.reshape(-1, 2)
    quat = np.zeros((n, 4), np.float32)
    quat[:, 2] = np.sin(0.5 * yaw.reshape(-1))
    quat[:, 3] = np.cos(0.5 * yaw.reshape(-1))
    he = np.tile(np.asarray(half_extents, np.float32), (n, 1))
    dev = str(track.outer.device)
    return (wp.array(pos, dtype=wp.vec3f, device=dev),
            wp.array(quat, dtype=wp.quatf, device=dev),
            wp.array(he, dtype=wp.vec2f, device=dev))


def run_racing_benchmark(E=4096, B=1, device="cuda", sdf_resolution=256,
                         iters=1000, warmup=20, seed=0, timing=True,
                         seed0=RACING_SEED0, fp_tol=2.0e-3):
    """Racing near-edge mode: sdf vs exact segments on car boxes near the edges.

    Returns a dict: ``agreement`` (OOB-flag agreement over the boxes of valid
    envs), ``fn`` / ``fp`` (sdf misses / extra hits against the exact flags),
    ``fp_beyond`` (false positives whose exact clearance exceeds ``fp_tol``
    metres), ``fp_max_clearance``, ``contract_violations`` (sdf boxes where
    ``oob != (distance < 0)``), the median grid cell size, and, on cuda when
    ``timing``, the per-replay graph times ``sdf_graph_s`` /
    ``seg_graph_s`` (median) and their p95.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    track = racing_tracks(E, device, seed0)
    dev = str(track.outer.device)
    pos, quat, he = near_edge_boxes(track, E, B, np.random.default_rng(seed))
    seg = CollisionChecker(track, max_boxes=B, method="segments",
                           position=pos, orientation=quat, half_extents=he)
    sdf = CollisionChecker(track, max_boxes=B, method="sdf",
                           sdf_resolution=sdf_resolution,
                           position=pos, orientation=quat, half_extents=he)
    exact = seg.query().clone()
    approx = sdf.query()
    wp.synchronize()

    valid = track.valid.numpy().astype(bool)
    mask = np.repeat(valid, B)
    o_ex = exact.oob.numpy()[mask].astype(bool)
    o_ap = approx.oob.numpy()[mask].astype(bool)
    d_ex = exact.distance.numpy()[mask]
    d_ap = approx.distance.numpy()[mask]
    fp = ~o_ex & o_ap
    m = {
        "device": device, "E": E, "B": B, "sdf_resolution": sdf_resolution,
        "boxes": int(mask.sum()), "valid_envs": int(valid.sum()),
        "cell_median": float(np.median(_sdf_extent(track)[valid] / sdf_resolution)),
        "oob_rate": float(o_ex.mean()),
        "agreement": float((o_ex == o_ap).mean()),
        "fn": int(np.sum(o_ex & ~o_ap)),
        "fp": int(fp.sum()),
        "fp_beyond": int(np.sum(fp & (d_ex > fp_tol))),
        "fp_max_clearance": float(d_ex[fp].max()) if fp.any() else 0.0,
        "contract_violations": int(np.sum(o_ap != (d_ap < 0.0))),
        "sdf_graph_s": None, "sdf_graph_p95_s": None,
        "seg_graph_s": None, "seg_graph_p95_s": None,
    }
    if timing and wp.get_device(dev).is_cuda:
        m["sdf_graph_s"], m["sdf_graph_p95_s"] = _time_graph(
            sdf.query, iters, warmup, dev)
        m["seg_graph_s"], m["seg_graph_p95_s"] = _time_graph(
            seg.query, min(iters, 200), warmup, dev)
    return m


def _sdf_extent(track) -> np.ndarray:
    """``[E]`` larger side of each env's SDF grid box, from public data: the documented
    auto rule (``sdf_padding=None``) pads the plan-view AABB of the env's outer boundary
    (its first ``count`` points) by ``0.1 *`` its larger extent on each side."""
    E = int(track.count.shape[0])
    outer = track.outer.numpy().reshape(E, -1, 3)
    count = track.count.numpy()
    out = np.full(E, 2.0, np.float64)       # the box of a degenerate env: (-1, -1)-(1, 1)
    for e in range(E):
        n = int(min(max(int(count[e]), 0), outer.shape[1]))
        if n < 3:
            continue
        xy = outer[e, :n, :2]
        ext = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
        out[e] = ext * (1.0 + 2.0 * 0.1)
    return out


def _print_racing(m):
    def us(x):
        return "-" if x is None else f"{x * 1e6:.1f} us"

    print(f"\nracing near-edge: device {m['device']}  E {m['E']}  B {m['B']}  "
          f"sdf {m['sdf_resolution']}^2 (median cell {m['cell_median'] * 1e3:.0f} mm)")
    print(f"boxes {m['boxes']} on {m['valid_envs']} valid envs, exact oob rate "
          f"{m['oob_rate']:.3f}")
    print(f"sdf flag agreement {m['agreement'] * 100:.3f}%  FN {m['fn']}  FP {m['fp']} "
          f"(beyond 2 mm: {m['fp_beyond']}, max exact clearance "
          f"{m['fp_max_clearance'] * 1e3:.2f} mm)  contract violations "
          f"{m['contract_violations']}")
    print(f"graph replay per query: sdf {us(m['sdf_graph_s'])} (p95 "
          f"{us(m['sdf_graph_p95_s'])})  segments {us(m['seg_graph_s'])} (p95 "
          f"{us(m['seg_graph_p95_s'])})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--E", type=int, default=None,
                    help="envs (default 8192; 4096 with --racing)")
    ap.add_argument("--B", type=int, default=None,
                    help="boxes per env (default 8; 1 with --racing)")
    ap.add_argument("--res", type=int, default=None,
                    help="sdf grid resolution (default 128; 256 with --racing)")
    ap.add_argument("--iters", type=int, default=None,
                    help="timed iterations (default 200; 1000 with --racing)")
    ap.add_argument("--seed", type=int, default=None,
                    help="box placement seed (default 42; 0 with --racing)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--racing", action="store_true",
                    help="racing near-edge mode (see the module docstring)")
    ap.add_argument("--no-timing", action="store_true",
                    help="--racing only: accuracy without the graph timings")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"

    def pick(value, default, racing_default):
        if value is not None:
            return value
        return racing_default if args.racing else default

    E = pick(args.E, 8192, 4096)
    B = pick(args.B, 8, 1)
    res = pick(args.res, 128, 256)
    iters = pick(args.iters, 200, 1000)
    seed = pick(args.seed, 42, 0)
    if args.racing:
        _print_racing(run_racing_benchmark(
            E=E, B=B, device=device, sdf_resolution=res, iters=iters,
            seed=seed, timing=not args.no_timing))
        return

    m = run_collision_benchmark(E=E, B=B, sdf_resolution=res,
                                iters=iters, seed=seed, device=device)

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
