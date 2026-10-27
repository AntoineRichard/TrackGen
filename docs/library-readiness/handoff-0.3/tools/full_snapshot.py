"""Every registered track / gate generator's outputs at its default config on one device,
two calls each (the first builds the self-graph, the second replays it), for a byte-identity
diff between trees on the SAME machine.

    python full_snapshot.py cuda:0 out.npz
    python full_snapshot.py --compare a.npz b.npz
"""
import sys
import warnings

import numpy as np

TRACK = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
         "count", "winding")
GATE = ("position", "tangent", "forward", "orientation", "half_size", "left", "right",
        "valid", "count")


def snapshot(device, path):
    import warp as wp
    wp.config.quiet = True
    warnings.simplefilter("ignore")
    from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig,
                           TrackGenerator)
    from track_gen.plugin import registry
    wp.init()
    E, out = 8, {}
    for g in sorted(registry.available("generator")):
        if g == "external":
            continue                     # ingestion: no seeded output
        gen = TrackGenerator(TrackGenConfig(generator=g, device=device, num_envs=E),
                             PerEnvSeededRNG(seeds=1234, num_envs=E, device=device))
        for call in range(2):
            t = gen.generate()
            wp.synchronize()
            for f in TRACK:
                out[f"track/{g}/{call}/{f}"] = getattr(t, f).numpy().copy()
    for g in sorted(registry.available("gate_generator")):
        gen = GateGenerator(GateGenConfig(generator=g, device=device, num_envs=E,
                                          gate_width=0.05),
                            PerEnvSeededRNG(seeds=1234, num_envs=E, device=device))
        for call in range(2):
            s = gen.generate()
            wp.synchronize()
            for f in GATE:
                out[f"gates/{g}/{call}/{f}"] = getattr(s, f).numpy().copy()
    np.savez(path, **out)
    print(f"saved {len(out)} arrays to {path}")


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    missing = sorted(set(a.files) - set(b.files))
    new = sorted(set(b.files) - set(a.files))
    if missing:
        print("MISSING in b:", missing)
    if new:
        print(f"{len(new)} new in b (not compared), e.g. {new[:3]}")
    common = sorted(set(a.files) & set(b.files))
    bad = [k for k in common if a[k].dtype != b[k].dtype or a[k].tobytes() != b[k].tobytes()]
    print(f"{len(common)} arrays compared; byte mismatches: {bad}")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    snapshot(sys.argv[1], sys.argv[2])
