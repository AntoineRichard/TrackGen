"""Quick ground-truth baseline: production repulsive generator, E=64 seed 11, cuda.

Confirms the current stable result (task says 64/64 @ compactness 0.15093) and times
the growth phase so the spike has a like-for-like reference.
"""
from __future__ import annotations
import sys, time
import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen import PerEnvSeededRNG


def compactness(pts):
    nxt = np.roll(pts, -1, axis=1)
    per = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs((pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1))
    return 4.0 * np.pi * area / np.maximum(per * per, 1e-12)


def run(E, seed, dev="cuda"):
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seed, num_envs=E, device=dev))
    t = gen.generate(E)
    valid = t.valid.numpy().astype(bool)
    N = int(cfg.num_points)
    gc = gen._scratch.gen_centerline.numpy().reshape(E, N, 2)
    cmp = compactness(gc)
    return gen, cfg, valid, cmp


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    gen, cfg, valid, cmp = run(64, 11, dev)
    print(f"[baseline E=64 seed=11 {dev}] yield={valid.sum()}/64  "
          f"compactness median={np.median(cmp):.5f} mean={cmp.mean():.5f}")
