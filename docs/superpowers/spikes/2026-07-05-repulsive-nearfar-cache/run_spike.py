"""Driver for the near/far-cache spike experiments.

    PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 \
        docs/superpowers/spikes/2026-07-05-repulsive-nearfar-cache/run_spike.py <cmd> [dev]

cmd in {frontier, perf, determ, smoke}. dev defaults to cuda.
"""
from __future__ import annotations
import sys, time
import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src import generator_registry as reg
from track_gen._src import warp_generate_repulsive as rep
from track_gen._src.track_generator import TrackGenerator
from track_gen import PerEnvSeededRNG

import nearfar


# --- maxnbr heuristic per cutoff (candidate slots per vertex) ---
MAXNBR = {8: 64, 16: 112, 32: 200}


def _compactness(pts):
    nxt = np.roll(pts, -1, axis=1)
    per = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs((pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1))
    return 4.0 * np.pi * area / np.maximum(per * per, 1e-12)


def _register(gen_fn):
    reg.register(reg.GeneratorSpec(name="repulsive", alloc_scratch=rep.repulsive_alloc_scratch,
                                   generate=gen_fn, capturable=False))


def run_yield(K, cutoff, E=64, seed=11, dev="cuda"):
    """Full pipeline through the tail -> (yield, compactness median, overflow)."""
    if K == 1:
        gen_fn = rep.generate_repulsive_warp
    else:
        gen_fn = nearfar.make_generate(K, float(cutoff), MAXNBR[cutoff])
    _register(gen_fn)
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seed, num_envs=E, device=dev))
    t = gen.generate(E)
    valid = t.valid.numpy().astype(bool)
    N = int(cfg.num_points)
    gc = gen._scratch.gen_centerline.numpy().reshape(E, N, 2)
    cmp = _compactness(gc)
    ovf = nearfar.last_overflow(gen._scratch.gen) if K > 1 else 0
    return int(valid.sum()), float(np.median(cmp)), ovf


def _alloc_growth(cfg, dev):
    E, N = int(cfg.num_envs), int(cfg.num_points)
    scratch = rep.repulsive_alloc_scratch(cfg)
    out = wp.empty(E * N, dtype=wp.vec2f, device=dev)
    valid = wp.empty(E, dtype=wp.int32, device=dev)
    seeds = PerEnvSeededRNG(seeds=11, num_envs=E, device=dev).seeds_warp
    return scratch, out, valid, seeds


def time_growth(gen_fn, cfg, dev, reps=3, warmup=2):
    scratch, out, valid, seeds = _alloc_growth(cfg, dev)
    for _ in range(warmup):
        gen_fn(seeds, cfg, out, valid, scratch)
    wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gen_fn(seeds, cfg, out, valid, scratch)
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), out, scratch


def cmd_smoke(dev):
    print("== smoke: K=1 must reproduce production baseline exactly ==")
    y, c, o = run_yield(1, 16, dev=dev)
    print(f"K=1 (production path): yield={y}/64 compactness={c:.5f}")
    y, c, o = run_yield(4, 16, dev=dev)
    print(f"K=4 cutoff=16: yield={y}/64 compactness={c:.5f} overflow={o}")


def cmd_frontier(dev):
    print("== Experiment 1: K/cutoff quality frontier (E=64 seed=11, through tail) ==")
    print(f"{'K':>3} {'cutoff':>7} {'maxnbr':>7} {'yield':>7} {'compact':>9} {'overflow':>9}")
    # ground truth
    y, c, o = run_yield(1, 16, dev=dev)
    print(f"{1:>3} {'--':>7} {'--':>7} {y:>5}/64 {c:>9.5f} {'--':>9}   <- K=1 ground truth")
    for cutoff in (8, 16, 32):
        for K in (2, 4, 8, 16):
            y, c, o = run_yield(K, cutoff, dev=dev)
            print(f"{K:>3} {cutoff:>7} {MAXNBR[cutoff]:>7} {y:>5}/64 {c:>9.5f} {o:>9}")


def _one_growth(gen_fn, cfg, dev, sc, out, valid, seeds):
    t0 = time.perf_counter()
    gen_fn(seeds, cfg, out, valid, sc)
    wp.synchronize()
    return time.perf_counter() - t0


def cmd_perf(dev):
    print("== Experiment 2: growth-phase wall-clock, best config vs K=1 baseline ==")
    print("   (alternating A/B, min of 5, post-warmup -- controls thermal/clock drift)")
    best_K, best_cut = int(_getarg("K", 8)), int(_getarg("cut", 8))
    print(f"best split config: K={best_K} cutoff={best_cut} maxnbr={MAXNBR[best_cut]}")
    print(f"{'E':>6} {'baseK1(s)':>11} {'split(s)':>11} {'speedup':>8}")
    for E in (64, 1024, 8192):
        cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
        sc_b, out_b, val_b, seeds = _alloc_growth(cfg, dev)
        sc_s, out_s, val_s, _ = _alloc_growth(cfg, dev)
        gen_fn = nearfar.make_generate(best_K, float(best_cut), MAXNBR[best_cut])
        for _ in range(2):
            rep.generate_repulsive_warp(seeds, cfg, out_b, val_b, sc_b)
            gen_fn(seeds, cfg, out_s, val_s, sc_s)
        wp.synchronize()
        tb, ts = [], []
        for _ in range(5):
            tb.append(_one_growth(rep.generate_repulsive_warp, cfg, dev, sc_b, out_b, val_b, seeds))
            ts.append(_one_growth(gen_fn, cfg, dev, sc_s, out_s, val_s, seeds))
        b, s = min(tb), min(ts)
        print(f"{E:>6} {b:>11.4f} {s:>11.4f} {b/s:>7.2f}x")


def cmd_determ(dev):
    print("== Experiment 3: byte-determinism of the split (same seed, twice) ==")
    best_K, best_cut = int(_getarg("K", 8)), int(_getarg("cut", 8))
    gen_fn = nearfar.make_generate(best_K, float(best_cut), MAXNBR[best_cut])
    cfg = TrackGenConfig(generator="repulsive", num_envs=64, device=dev)
    _, out1, _ = time_growth(gen_fn, cfg, dev, reps=1, warmup=0)
    a = out1.numpy().copy()
    _, out2, _ = time_growth(gen_fn, cfg, dev, reps=1, warmup=0)
    b = out2.numpy().copy()
    identical = np.array_equal(a, b)
    print(f"K={best_K} cutoff={best_cut}: byte-identical across two runs = {identical}")
    if not identical:
        print(f"  max abs diff = {np.abs(a-b).max():.3e}")


def cmd_stages(dev):
    print("== coarse vs final+settle wall-clock split (where the speedup can live) ==")
    nearfar.PROFILE = True
    for E in (1024, 8192):
        cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
        for label, K, cut in [("K=1 baseline", 1, 8), ("K=8 cut=8", 8, 8)]:
            gf = nearfar.make_generate(K, float(cut), MAXNBR[cut])
            nearfar._STAGE_LOG.clear()
            time_growth(gf, cfg, dev, reps=1, warmup=1)
            d = dict(nearfar._STAGE_LOG[-2:])
            tot = d["coarse"] + d["final+settle"]
            print(f"E={E:>5} {label:<14}: coarse={d['coarse']*1e3:7.1f}ms  "
                  f"final+settle={d['final+settle']*1e3:7.1f}ms  final%={d['final+settle']/tot*100:3.0f}%")
    nearfar.PROFILE = False


_ARGS = {}


def _getarg(k, default):
    return _ARGS.get(k, default)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    dev = "cuda"
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=")
            _ARGS[k] = v
        else:
            dev = a
    {"smoke": cmd_smoke, "frontier": cmd_frontier, "perf": cmd_perf,
     "determ": cmd_determ, "stages": cmd_stages}[cmd](dev)
