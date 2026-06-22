"""Benchmark: K=2 best-of-K + single-crossing CLIP vs plain best-of-K (DEV ONLY).

Tests the user's bet: instead of paying K=8 (8x compute) to brute-force self-intersection to
zero, use a cheaper K=2 best-of-K + one-shot loop-removal clip. Measures post-process
self-intersection rate, an O(N^2)-pass COST PROXY (the Warp cost driver), shape preservation
(compactness + straight_fraction), and the KEY diagnostic: among the ~21% single-candidate
crossers, what fraction have exactly ONE crossing (one-shot clippable) vs 2 vs >=3.

Run: /home/antoiner/Documents/TrackGen/.venv/bin/python track_gen/_experimental/checkpoint_clip_bench.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from track_gen._experimental import checkpoint_proto as cp  # noqa: E402
from benchmarks.track_metrics import (  # noqa: E402
    compactness, self_intersects, straight_fraction,
)

N_SEEDS = 1000
cfg = cp.DEFAULTS.copy()


# ---- per-config evaluators (each returns the final centerline + the O(N^2) passes it spent) ----

def eval_bestofK(seed: int, K: int):
    """Best-of-K ALONE: K candidates, keep fewest-crossings (early-out on first zero).

    Cost proxy: K self-intersection PASSES (one per candidate). We report the WORST-CASE K (no
    early-out) so the column is seed-independent and conservative; the runtime early-outs.
    """
    best, best_sc = None, None
    for k in range(K):
        cand = cp.generate_candidate(seed * K + k, cfg)
        sc = cp._self_intersections_count(cand)
        if best_sc is None or sc < best_sc:
            best, best_sc = cand, sc
            if best_sc == 0:
                break
    return best, K


def eval_Kclip(seed: int, K: int):
    """K + one-shot clip: clip each candidate, keep fewest residual crossings.

    Cost proxy: 2K passes (K clip-find passes + K residual-count passes), worst-case (no early-out).
    """
    best, _ = cp.generate_centerline_clip(seed, cfg, K), None
    return best[0], 2 * K


CONFIGS = [
    ("K=1",        lambda s: eval_bestofK(s, 1)),
    ("K=2",        lambda s: eval_bestofK(s, 2)),
    ("K=4",        lambda s: eval_bestofK(s, 4)),
    ("K=8",        lambda s: eval_bestofK(s, 8)),
    ("K=1+clip",   lambda s: eval_Kclip(s, 1)),
    ("K=2+clip",   lambda s: eval_Kclip(s, 2)),
    ("K=4+clip",   lambda s: eval_Kclip(s, 4)),
]


def run():
    print(f"=== checkpoint clip benchmark — {N_SEEDS} seeds, closure={cfg['closure']}, "
          f"C={cfg['checkpoint_count']}, N={cfg['num_points']} ===\n")
    print(f"{'config':>9} | {'SI rate':>8} | {'cost (N^2 passes/env)':>21} | "
          f"{'comp p50':>8} | {'straight':>8} | {'wall ms/1k':>10}")
    print("-" * 80)
    rows = {}
    for name, fn in CONFIGS:
        si = np.zeros(N_SEEDS, dtype=bool)
        comp = np.empty(N_SEEDS)
        strt = np.empty(N_SEEDS)
        passes = 0
        t0 = time.perf_counter()
        for s in range(N_SEEDS):
            pts, cost = fn(s)
            passes += cost
            si[s] = self_intersects(pts)
            comp[s] = compactness(pts)
            strt[s] = straight_fraction(pts)
        wall = (time.perf_counter() - t0) * 1000.0 * (1000.0 / N_SEEDS)
        cost_per_env = passes / N_SEEDS
        rows[name] = dict(si=si.mean(), cost=cost_per_env,
                          comp=float(np.median(comp)), strt=float(np.mean(strt)), wall=wall)
        print(f"{name:>9} | {si.mean():>8.4f} | {cost_per_env:>21.1f} | "
              f"{np.median(comp):>8.3f} | {np.mean(strt):>8.3f} | {wall:>10.0f}")

    # ---- KEY diagnostic: crossing-count histogram among raw single-candidate crossers ----
    print("\n=== crossing-count distribution among raw K=1 single-candidate CROSSERS ===")
    counts = np.array([cp._self_intersections_count(cp.generate_candidate(s, cfg))
                       for s in range(N_SEEDS)])
    crossers = counts[counts > 0]
    total = len(counts)
    n_cross = len(crossers)
    print(f"single-candidate crossing rate: {n_cross}/{total} = {n_cross / total:.4f}")
    if n_cross:
        eq1 = int(np.sum(crossers == 1))
        eq2 = int(np.sum(crossers == 2))
        ge3 = int(np.sum(crossers >= 3))
        print(f"  of the {n_cross} crossers:")
        print(f"    exactly 1 crossing (one-shot clippable -> simple): "
              f"{eq1} = {eq1 / n_cross:.4f} of crossers ({eq1 / total:.4f} of all)")
        print(f"    exactly 2 crossings:                               "
              f"{eq2} = {eq2 / n_cross:.4f} of crossers")
        print(f"    >=3 crossings:                                     "
              f"{ge3} = {ge3 / n_cross:.4f} of crossers")
        print(f"  max crossings on any crosser: {int(crossers.max())}")
    return rows


if __name__ == "__main__":
    run()
