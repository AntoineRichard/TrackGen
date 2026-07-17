"""Generate bake-off inputs from the REAL pipeline, once per spacing factor.

For each f in {0.3, 0.45, 0.6}: run TrackGenerator twice with identical seeds —
relax_enable=False (raw constant-spacing centerlines = spike input) and
relax_enable=True (warp XPBD output = parity reference) — and cache both.
Torch is dev-only here (spike dir); the runtime stays warp-native.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch
import warp as wp

E, SEED, HW = 64, 20, 0.1
FACTORS = [0.3, 0.45, 0.6]
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _track_to_torch(track, E):
    n_max = track.center.shape[0] // E
    center = wp.to_torch(track.center).view(E, n_max, 2).to("cpu").clone()
    count = wp.to_torch(track.count).to("cpu").clone().long()
    valid = wp.to_torch(track.valid).to("cpu").clone().bool()
    return center, count, valid


def generate_for_factor(f: float) -> dict:
    from track_gen import TrackGenConfig, TrackGenerator, PerEnvSeededRNG
    spacing = f * HW
    out = {"f": f, "half_width": HW, "spacing": spacing, "seed": SEED, "E": E}
    for relax, key in [(False, "0"), (True, "warp")]:
        cfg = TrackGenConfig(device="cpu", num_envs=E, half_width=HW,
                             spacing=spacing, relax_enable=relax)
        rng = PerEnvSeededRNG(seeds=SEED, num_envs=E, device="cpu")
        track = TrackGenerator(cfg, rng).generate()
        center, count, valid = _track_to_torch(track, E)
        out[f"center_{key}"] = center
        out[f"count_{key}"] = count
        out[f"valid_{key}"] = valid
    assert torch.equal(out["count_0"], out["count_warp"]), \
        "relax must not change per-env point counts"
    return out


def main():
    wp.init()
    os.makedirs(CACHE_DIR, exist_ok=True)
    for f in FACTORS:
        d = generate_for_factor(f)
        n_fin = torch.isfinite(d["center_0"]).all(dim=-1).any(dim=-1).sum().item()
        path = os.path.join(CACHE_DIR, f"inputs_f{int(round(f*100)):03d}.pt")
        torch.save(d, path)
        print(f"f={f}: saved {path}  finite_envs={n_fin}/{E}  "
              f"count[min={d['count_0'].min()},max={d['count_0'].max()}]  "
              f"warp_valid={d['valid_warp'].sum().item()}/{E}")


if __name__ == "__main__":
    main()
