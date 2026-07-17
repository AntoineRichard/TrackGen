"""Capture pre-vec3f golden outputs (cpu, fixed seeds) for the migration regression.

Run once BEFORE the vec3f migration; the .npz is committed and never regenerated.
"""
from pathlib import Path

import numpy as np

from track_gen._src.gate_generator import GateGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import GateGenConfig, TrackGenConfig

GOLDEN = str(Path(__file__).resolve().parents[1] / "goldens" / "pre_vec3f.npz")
GENERATORS = ("bezier", "hull", "polar", "voronoi", "checkpoint")
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen",
                "length", "valid", "count", "winding")
GATE_FIELDS = ("position", "tangent", "left", "right", "valid", "count")
# Fields captured as vec2f pre-migration that are vec3f (z = 0) post-migration;
# the golden test compares their xy columns exactly and checks z separately.
VEC3_TRACK = ("outer", "center", "inner", "tangent", "normal")
VEC3_GATES = ("position", "tangent", "left", "right")


def capture() -> dict:
    out = {}
    for gen in GENERATORS:
        cfg = TrackGenConfig(generator=gen, device="cpu", num_envs=8)
        rng = PerEnvSeededRNG(seeds=1234, num_envs=8, device="cpu")
        track = TrackGenerator(cfg, rng).generate()
        for f in TRACK_FIELDS:
            out[f"track/{gen}/{f}"] = getattr(track, f).numpy().copy()
        gcfg = GateGenConfig(generator=gen, device="cpu", num_envs=8,
                             gate_width=0.05)
        grng = PerEnvSeededRNG(seeds=1234, num_envs=8, device="cpu")
        seq = GateGenerator(gcfg, grng).generate()
        for f in GATE_FIELDS:
            out[f"gates/{gen}/{f}"] = getattr(seq, f).numpy().copy()
    return out


if __name__ == "__main__":
    np.savez_compressed(GOLDEN, **capture())
    print(f"wrote {GOLDEN}")
