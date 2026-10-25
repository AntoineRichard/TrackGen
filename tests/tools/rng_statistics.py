"""Capture and compare geometry distributions across the RNG transition.

Run this same script against the old and new trees on one machine::

    PYTHONPATH=<tree> python rng_statistics.py capture old.npz --device cuda:0
    PYTHONPATH=<new-tree> python rng_statistics.py capture new.npz --device cuda:0
    python rng_statistics.py compare old.npz new.npz

The default samples two disjoint blocks of 4096 seeds. No output hashes are used:
the gate compares yield and empirical distributions, allowing independent draws.

The default CPU test instead checks a small, committed rc1 summary. Regenerate it
only from an explicitly selected, validated runtime tree::

    PYTHONPATH=<rc1-tree> python rng_statistics.py reference rng_statistics.json \
        --source-revision 4664986f6097b62e4f6557da925de0f9210c63fe

That reference records every config and seed. It covers all six track generators,
all five built-in gate paths, independent random-pair ordering, and noise elevation.
Yield uses a four-sigma binomial bound (at least one environment at the endpoints);
10/50/90% quantiles allow 5% relative drift plus 1e-5 for zero/last-ulp differences.
This is a distribution regression guard, not another byte fingerprint.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math

import numpy as np


def metrics(result, *, gates: bool) -> dict[str, np.ndarray]:
    """Shape statistics over valid closed polygons; yield includes every env."""
    valid = result.valid.numpy().astype(bool)
    counts = result.count.numpy()
    points = (result.position if gates else result.center).numpy().reshape(len(valid), -1, 3)
    out = {"valid": valid.astype(np.float64), "count": counts[valid].astype(np.float64)}
    rows = {name: [] for name in ("length", "compactness", "curvature", "z_mean", "z_std")}
    for p, count, keep in zip(points, counts, valid):
        if not keep:
            continue
        p = p[:int(count)].astype(np.float64)
        if len(p) < 3 or not np.isfinite(p).all():
            raise AssertionError("valid result has fewer than three finite points")
        edges = np.roll(p, -1, axis=0) - p
        lengths = np.linalg.norm(edges, axis=1)
        perimeter = float(lengths.sum())
        xy = p[:, :2]
        area = abs(float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                                - xy[:, 1] * np.roll(xy[:, 0], -1)))) * 0.5
        xy_edges = edges[:, :2]
        previous = np.roll(xy_edges, 1, axis=0)
        angles = np.arctan2(previous[:, 0] * xy_edges[:, 1]
                           - previous[:, 1] * xy_edges[:, 0],
                           np.sum(previous * xy_edges, axis=1))
        rows["length"].append(perimeter)
        rows["compactness"].append(4.0 * math.pi * area / max(perimeter ** 2, 1e-30))
        rows["curvature"].append(float(np.abs(angles).sum()) / max(perimeter, 1e-30))
        rows["z_mean"].append(float(p[:, 2].mean()))
        rows["z_std"].append(float(p[:, 2].std()))
    out.update({key: np.asarray(values) for key, values in rows.items()})
    return out


_SHAPE_METRICS = ("length", "compactness", "curvature", "count", "z_mean", "z_std")
_QUANTILES = (0.1, 0.5, 0.9)


def summarize(sample) -> dict:
    """Summarize the per-case arrays produced by ``metrics`` or ``capture``."""
    cases = {key.rsplit("/", 1)[0] for key in sample}
    result = {}
    for case in sorted(cases):
        valid = np.asarray(sample[f"{case}/valid"])
        if not len(valid) or not np.isin(valid, [0, 1]).all():
            raise AssertionError(f"{case}: expected nonempty binary valid mask")
        count = int(valid.sum())
        if not count:
            raise AssertionError(f"{case}: no valid samples")
        quantiles = {}
        for metric in _SHAPE_METRICS:
            values = np.asarray(sample[f"{case}/{metric}"])
            if len(values) != count or not np.isfinite(values).all():
                raise AssertionError(f"{case}/{metric}: expected finite valid-only statistics")
            quantiles[metric] = np.quantile(values, _QUANTILES).tolist()
        result[case] = {"samples": len(valid), "valid": count, "yield": count / len(valid),
                        "quantiles": quantiles}
    return result


def compare_summaries(expected, observed, *, yield_sigma=4.0,
                      quantile_rtol=0.05, quantile_atol=1e-5) -> None:
    """Reject material yield/shape drift while tolerating CPU/Warp roundoff."""
    if set(expected) != set(observed):
        raise AssertionError("summary case keys differ")
    failures = []
    for case, want in expected.items():
        got = observed[case]
        n = want["samples"]
        if n <= 0 or got["samples"] != n:
            raise AssertionError(f"{case}: sample counts differ")
        p, q = want["yield"], got["yield"]
        if not 0 <= p <= 1 or not 0 <= q <= 1:
            raise AssertionError(f"{case}: yield must be finite and in [0, 1]")
        # One environment of continuity avoids a zero-width bound at p=0 or 1.
        limit = max(1 / n, yield_sigma * math.sqrt(p * (1 - p) / n))
        if abs(p - q) > limit:
            failures.append(f"{case}/yield: {q:.6g} vs {p:.6g} (limit {limit:.6g})")
        if set(want["quantiles"]) != set(got["quantiles"]):
            raise AssertionError(f"{case}: quantile metric keys differ")
        for metric, values in want["quantiles"].items():
            a, b = np.asarray(values), np.asarray(got["quantiles"][metric])
            if (a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all()
                    or not np.isfinite(b).all()):
                raise AssertionError(f"{case}/{metric}: expected three finite quantiles")
            if not np.allclose(b, a, rtol=quantile_rtol, atol=quantile_atol):
                failures.append(f"{case}/{metric}: {b.tolist()} vs {a.tolist()}")
    if failures:
        raise AssertionError("distribution drift:\n" + "\n".join(failures))


def reference_cases() -> list[dict]:
    """Full configs for the bounded default-lane sample; seeds include zero."""
    from track_gen import GateGenConfig, TrackGenConfig

    cases = []
    for gates, names in ((False, ("bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive")),
                         (True, ("bezier", "hull", "polar", "voronoi", "checkpoint"))):
        for name in names:
            options = {}
            # O(N²) growth needs a smaller batch and mesh, but remains in CI.
            n = 64 if name == "repulsive" else 1024
            if name == "repulsive":
                options = dict(num_points=64, repulsive_stages=(16, 32, 64))
            cfg = (GateGenConfig if gates else TrackGenConfig)(
                generator=name, device="cpu", num_envs=n, **options)
            case = f"{'gates' if gates else 'track'}/{name}/flat/default"
            cases.append(dict(name=case, gates=gates, seeds=dict(start=0, step=1, count=n),
                              config=dataclasses.asdict(cfg)))
    cfg = TrackGenConfig(device="cpu", num_envs=1024, generator="bezier", z_profile="noise",
                         z_min=-0.4, z_max=0.4, z_noise_amplitude=0.2)
    cases.append(dict(name="track/bezier/noise/default", gates=False,
                      seeds=dict(start=0, step=1, count=1024), config=dataclasses.asdict(cfg)))
    cfg = GateGenConfig(device="cpu", num_envs=1024, generator="bezier", gate_ordering="random_pairs")
    cases.append(dict(name="gates/bezier/flat/random_pairs", gates=True,
                      seeds=dict(start=0, step=1, count=1024), config=dataclasses.asdict(cfg)))
    return cases


def sample_cases(cases) -> dict[str, np.ndarray]:
    """Run the recorded CPU configurations with their explicit per-env seeds."""
    import warp as wp
    from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG,
                           TrackGenConfig, TrackGenerator)

    out = {}
    for case in cases:
        gates, config, block = case["gates"], case["config"], case["seeds"]
        if config["device"] != "cpu" or block["count"] != config["num_envs"]:
            raise AssertionError("reference samples require matching CPU config and seed count")
        values = block["start"] + block["step"] * np.arange(block["count"], dtype=np.int32)
        seeds = wp.array(values, dtype=wp.int32, device="cpu")
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=len(values), device="cpu")
        cfg = (GateGenConfig if gates else TrackGenConfig)(**config)
        gen = (GateGenerator if gates else TrackGenerator)(cfg, rng)
        result = gen.generate()
        for metric, array in metrics(result, gates=gates).items():
            out[f"{case['name']}/{metric}"] = array
        del result, gen, rng, seeds
    return out


def write_reference(path: str, *, source_revision: str) -> None:
    """Capture a reviewable reference from the runtime selected by PYTHONPATH."""
    import platform
    from pathlib import Path
    import warp as wp
    import track_gen

    cases = reference_cases()
    reference = dict(format=1, source_revision=source_revision,
                     track_gen_version=track_gen.__version__, warp_version=wp.__version__,
                     platform=f"{platform.system().lower()}-{platform.machine().lower()}",
                     device="cpu", quantile_levels=list(_QUANTILES),
                     tolerances=dict(yield_sigma=4.0, quantile_rtol=0.05, quantile_atol=1e-5),
                     cases=cases, summary=summarize(sample_cases(cases)))
    Path(path).write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")


def capture(path: str, *, device: str, samples: int, blocks: int) -> None:
    import warp as wp
    from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG,
                           TrackGenConfig, TrackGenerator)

    wp.config.quiet = True
    wp.init()
    cases = [(False, name, {}) for name in
             ("bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive")]
    cases += [(True, name, {}) for name in ("bezier", "hull", "polar", "voronoi", "checkpoint")]
    # Exercise every elevation RNG and the independent gate ordering stream.
    for gates in (False, True):
        for profile in ("uniform", "random_walk", "noise"):
            cases.append((gates, "bezier", dict(z_profile=profile, z_min=-0.2, z_max=0.4,
                                              z_max_step=0.1, z_noise_amplitude=0.2)))
    for name in ("bezier", "hull"):
        cases.append((True, name, dict(gate_ordering="random_pairs")))
    out = {}
    for gates, name, options in cases:
        case = "/".join(("gates" if gates else "track", name,
                         options.get("z_profile", "flat"), options.get("gate_ordering", "default")))
        cfg_type, facade = (GateGenConfig, GateGenerator) if gates else (TrackGenConfig, TrackGenerator)
        kw = dict(options)
        if "on_invalid" in {f.name for f in dataclasses.fields(cfg_type)}:
            kw["on_invalid"] = "keep"
        collected = {}
        for block in range(blocks):
            # Include seed zero and a far-away block, without int32 overflow.
            values = np.arange(samples, dtype=np.int32) + block * 1_000_003
            seeds = wp.array(values, dtype=wp.int32, device=device)
            rng = PerEnvSeededRNG(seeds=seeds, num_envs=samples, device=device)
            cfg = cfg_type(generator=name, device=device, num_envs=samples, **kw)
            gen = facade(cfg, rng)
            result = gen.generate()
            for metric, array in metrics(result, gates=gates).items():
                collected.setdefault(metric, []).append(array)
            del result, gen, rng, seeds
        for metric, arrays in collected.items():
            out[f"{case}/{metric}"] = np.concatenate(arrays)
        print(f"{case}: yield={out[case + '/valid'].mean():.5f}", flush=True)
        # Preserve partial evidence if a later, expensive generator fails.
        np.savez(path, **out)
    print(f"saved {len(out)} distributions to {path}", flush=True)


def compare(old_path: str, new_path: str) -> dict:
    from scipy.stats import ks_2samp

    with np.load(old_path) as old, np.load(new_path) as new:
        if set(old.files) != set(new.files):
            raise AssertionError("distribution keys differ")
        report, failures = {}, []
        # Family-wise alpha: 0.001 over all empirical-distribution comparisons.
        # DKW two-sample bound; discrete counts are compared by the same CDF distance.
        alpha = 0.001 / len(old.files)
        for key in sorted(old.files):
            a, b = old[key], new[key]
            if not len(a) or not len(b):
                raise AssertionError(f"{key}: no valid samples")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise AssertionError(f"{key}: nonfinite statistics")
            if key.endswith("/valid"):
                pooled = float((a.sum() + b.sum()) / (len(a) + len(b)))
                limit = 0.01 + 6 * math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
                difference = abs(float(a.mean() - b.mean()))
            else:
                limit = math.sqrt(-0.5 * math.log(alpha / 2) * (1 / len(a) + 1 / len(b)))
                difference = float(ks_2samp(a, b).statistic)
            ok = difference <= limit
            report[key] = dict(old_n=len(a), new_n=len(b), old_mean=float(a.mean()),
                               new_mean=float(b.mean()), distance=difference, limit=limit, passed=ok)
            if not ok:
                failures.append(key)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise AssertionError(f"distribution drift: {failures}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("path")
    cap.add_argument("--device", default="cpu")
    cap.add_argument("--samples", type=int, default=4096)
    cap.add_argument("--blocks", type=int, default=2)
    cmp = sub.add_parser("compare")
    cmp.add_argument("old")
    cmp.add_argument("new")
    ref = sub.add_parser("reference")
    ref.add_argument("path")
    ref.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.path, device=args.device, samples=args.samples, blocks=args.blocks)
    elif args.command == "compare":
        compare(args.old, args.new)
    else:
        write_reference(args.path, source_revision=args.source_revision)
