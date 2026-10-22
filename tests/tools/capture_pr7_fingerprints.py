"""Byte-identity fingerprints of the built-in generators (roadmap PR7).

``compute()`` runs every built-in track and gate generator on CPU through the public
facades only (names that exist at 1077b18 and after PR7) and returns the sha256 of each
output field's bytes. The committed fingerprints were captured at 1077b18, before the
built-ins were ported to ``track_gen.ops``; ``tests/test_generator_fingerprints.py``
recomputes them on the current tree. Float results depend on the platform and the Warp
version, so the file is keyed by :func:`platform_key` and the test skips on a key it does
not hold.

Capture (from the repository root, against a checkout of the reference tree)::

    PYTHONPATH=<reference tree> python tests/tools/capture_pr7_fingerprints.py

It adds or replaces this platform's entry in ``tests/goldens/pr7_fingerprints.json``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import warp as wp

FINGERPRINTS = Path(__file__).resolve().parents[1] / "goldens" / "pr7_fingerprints.json"

E = 8
# Includes seeds whose seed*salt overflows int32 (the wrap-around keying).
SEEDS = [0, 1, 7, 99991, 2147483647, 1000003, 123456789, 46341]
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                "valid", "count", "winding", "truncated")
GATE_FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left",
               "right", "valid", "count")
TRACK_CASES = {
    "bezier": {},
    "bezier_style": dict(style_sampling=True, rad_range=(0.1, 0.35), scale_range=(0.8, 1.2),
                         handle_clamp_frac_range=(0.3, 0.9)),
    "hull": {},
    "polar": {},
    "voronoi": {},
    "checkpoint": {},
    "repulsive": dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384,
                      half_width=0.1),
}
# Configurations where some envs self-cross and take the polygon fallback (asserted).
FALLBACK_E = 32
FALLBACK_CASES = {
    "bezier_fallback": dict(generator="bezier", rad=0.49),
    "voronoi_fallback": dict(generator="voronoi", voronoi_radial_variation=0.85,
                             voronoi_angular_jitter=0.6, voronoi_num_sites=64),
    "hull_fallback": dict(generator="hull", hull_displacement=0.9),
}
GATE_CASES = ("bezier", "hull", "polar", "voronoi", "checkpoint")


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine()}-warp{wp.__version__}"


def _sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def compute() -> dict:
    from track_gen._src.gate_generator import GateGenerator
    from track_gen._src.rng_utils import PerEnvSeededRNG
    from track_gen._src.track_generator import TrackGenerator
    from track_gen._src.types import GateGenConfig, TrackGenConfig

    wp.config.quiet = True
    out = {}

    def seeds(n):
        return wp.array(np.asarray(SEEDS[:n], np.int32), dtype=wp.int32, device="cpu")

    for case, kw in TRACK_CASES.items():
        gen = case.split("_")[0]
        n = 2 if gen == "repulsive" else E
        cfg = TrackGenConfig(generator=gen, device="cpu", num_envs=n, **kw)
        track = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds(n), num_envs=n,
                                                    device="cpu")).generate()
        for f in TRACK_FIELDS:
            out[f"track/{case}/{f}"] = _sha(getattr(track, f).numpy())
    fb_seeds = wp.array(np.arange(FALLBACK_E, dtype=np.int32) * 7919 + 3, dtype=wp.int32,
                        device="cpu")
    for case, kw in FALLBACK_CASES.items():
        cfg = TrackGenConfig(device="cpu", num_envs=FALLBACK_E, **kw)
        tg = TrackGenerator(cfg, PerEnvSeededRNG(seeds=fb_seeds, num_envs=FALLBACK_E,
                                                 device="cpu"))
        track = tg.generate()
        g = tg._scratch.gen
        crossers = getattr(g, "gen_crossers", None)
        crossers = g.crossers if crossers is None else crossers
        assert (crossers.numpy() > 0).any(), f"{case}: no env took the polygon fallback"
        for f in TRACK_FIELDS:
            out[f"track/{case}/{f}"] = _sha(getattr(track, f).numpy())
    for gen in GATE_CASES:
        for ordering in ("ccw", "raw" if gen in ("polar", "voronoi", "checkpoint")
                         else "random_pairs"):
            gcfg = GateGenConfig(generator=gen, device="cpu", num_envs=E, gate_width=0.05,
                                 gate_ordering=ordering)
            seq = GateGenerator(gcfg, PerEnvSeededRNG(seeds=seeds(E), num_envs=E,
                                                      device="cpu")).generate()
            for f in GATE_FIELDS:
                out[f"gates/{gen}_{ordering}/{f}"] = _sha(getattr(seq, f).numpy())
    return out


def load() -> dict:
    return json.loads(FINGERPRINTS.read_text()) if FINGERPRINTS.exists() else {}


if __name__ == "__main__":
    data = load()
    data[platform_key()] = compute()
    FINGERPRINTS.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(data[platform_key()])} fingerprints for {platform_key()} "
          f"-> {FINGERPRINTS}")
