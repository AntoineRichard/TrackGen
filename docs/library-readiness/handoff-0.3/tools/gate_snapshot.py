"""Gate-pipeline byte snapshot over generator x ordering x z_profile x solve_iters x align,
using only the public GateGenConfig/GateGenerator API (works on pre- and post-PR11 trees).

    python gate_snapshot.py cuda:0 out.npz
    python gate_snapshot.py --compare a.npz b.npz
"""
import itertools
import sys

import numpy as np

FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left", "right",
          "valid", "count", "generation")


def snapshot(device, path):
    import warp as wp
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.plugin import registry

    wp.init()
    E = 8
    out, skipped = {}, []
    gens = sorted(registry.available("gate_generator"))
    for g, order, z, iters, align in itertools.product(
            gens, ("raw", "ccw", "random_pairs"), ("flat", "uniform", "random_walk", "noise"),
            (0, 8), ("yaw_only", "full_tangent")):
        key = f"{g}/{order}/{z}/{iters}/{align}"
        try:
            cfg = GateGenConfig(generator=g, device=device, num_envs=E, gate_ordering=order,
                                z_profile=z, gate_solve_iters=iters, gate_align=align,
                                gate_width=0.05)
            gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=1234, num_envs=E, device=device))
        except ValueError as exc:
            skipped.append(f"{key}: {exc}"[:120])
            continue
        for call in range(2):
            s = gen.generate()
            wp.synchronize()
            for f in FIELDS:
                a = getattr(s, f, None)
                if a is not None:
                    out[f"{key}/{call}/{f}"] = a.numpy().copy()
        # masked regeneration: advance the epochs of envs {1, 5} only
        mask = wp.array(np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.int32), device=device)
        s = gen.generate(advance=True, mask=mask)
        wp.synchronize()
        for f in FIELDS:
            a = getattr(s, f, None)
            if a is not None:
                out[f"{key}/2/{f}"] = a.numpy().copy()
    np.savez(path, **out)
    print(f"{len(out)} arrays, {len(out) // len(FIELDS)} configs*calls, {len(skipped)} skipped")
    for s in skipped[:12]:
        print("  skip", s)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    ka, kb = set(A.files), set(B.files)
    if ka != kb:
        print("key mismatch:", sorted(ka ^ kb)[:10], len(ka ^ kb))
    bad = [k for k in sorted(ka & kb) if A[k].tobytes() != B[k].tobytes()]
    print(f"{len(ka & kb)} common arrays, {len(bad)} byte mismatches")
    for k in bad[:20]:
        print("  MISMATCH", k)
    return 1 if bad or ka != kb else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    snapshot(sys.argv[1], sys.argv[2])
