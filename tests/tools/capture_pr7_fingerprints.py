"""Track and gate field fingerprints for the 0.3 RNG baseline.

Use tests.tools.capture_03_goldens to capture and merge platform fragments together.
The historical filename identifies its coverage family; values use the current keying.
"""
from __future__ import annotations

from tests.tools._golden_config import keep_config

import hashlib
import json
import platform
import sys
import warnings
from collections.abc import Mapping
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

    "hull": {},
    "polar": {},
    "voronoi": {},
    "checkpoint": {},
    "repulsive": dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384,
                      half_width=0.1),
}
# Configurations where some envs self-cross and take the polygon fallback (asserted).
FALLBACK_E = 128
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


def _crossers(g):
    """The polygon-fallback crossing counts of a generator's private scratch: a legacy
    scratch group (``gen_crossers`` / ``crossers``, 1077b18 .. 0.2.0.dev4) or, since
    roadmap PR13c, the Stage's built scratch (``scratch["scratch:generator/<slot>"]``)."""
    scratch = getattr(g, "scratch", None)
    if isinstance(scratch, Mapping):
        for name in ("gen_crossers", "crossers"):
            arr = scratch.get(f"scratch:generator/{name}")
            if arr is not None:
                return arr
    crossers = getattr(g, "gen_crossers", None)
    return g.crossers if crossers is None else crossers


def compute() -> dict:
    with warnings.catch_warnings():
        # Roadmap PR13d: bezier's style_sampling (a fingerprinted case) is deprecated.
        warnings.filterwarnings("ignore", message="bezier's style_sampling is deprecated",
                                category=DeprecationWarning)
        return _compute()


def _compute() -> dict:
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
        cfg = keep_config(TrackGenConfig, generator=gen, device="cpu", num_envs=n, **kw)
        track = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds(n), num_envs=n,
                                                    device="cpu")).generate()
        for f in TRACK_FIELDS:
            out[f"track/{case}/{f}"] = _sha(getattr(track, f).numpy())
    fb_seeds = wp.array(np.arange(FALLBACK_E, dtype=np.int32) * 7919 + 3, dtype=wp.int32,
                        device="cpu")
    for case, kw in FALLBACK_CASES.items():
        cfg = keep_config(TrackGenConfig, device="cpu", num_envs=FALLBACK_E, **kw)
        tg = TrackGenerator(cfg, PerEnvSeededRNG(seeds=fb_seeds, num_envs=FALLBACK_E,
                                                 device="cpu"))
        track = tg.generate()
        crossers = _crossers(tg._scratch.gen)
        assert (crossers.numpy() > 0).any(), f"{case}: no env took the polygon fallback"
        for f in TRACK_FIELDS:
            out[f"track/{case}/{f}"] = _sha(getattr(track, f).numpy())
    for gen in GATE_CASES:
        for ordering in ("ccw", "raw" if gen in ("polar", "voronoi", "checkpoint")
                         else "random_pairs"):
            gcfg = keep_config(GateGenConfig, generator=gen, device="cpu", num_envs=E, gate_width=0.05,
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
