"""Record the original CPU solver and returned road without editing runtime code.

Run from the checkout root; --help and --list-cases do not import Warp.
Generated NPZ arrays are local evidence, not files to commit.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

from metrics import array_record, circles_from_rings, curve_metrics, road_disc_gaps, tp_wall_energy, late_motion

ROOT = Path(__file__).resolve().parents[3]
RC2_REVISION = "a42759a3f999a8668a200cddc3f04e0508478dd1"
ORIGINAL_DOCS_REVISION = "4aec44a8f30d7ef998bb37ce6cb86e6ad2d12bd2"
CASES = {
    "baseline": (7, {}),
    "tau02": (7, {"repulsive_tau": .2}),
    "tau01": (7, {"repulsive_tau": .1}),
    "keep_obstacles": (7, {"repulsive_deactivate_obstacles": False}),
    "strict_stall": (7, {"repulsive_stall_area_tol": .001, "repulsive_settle_iters": 200}),
    "tau01_keep": (7, {"repulsive_tau": .1, "repulsive_deactivate_obstacles": False}),
    "tau01_slow_keep": (7, {"repulsive_tau": .1, "repulsive_ratchet_rate": .003,
                            "repulsive_deactivate_obstacles": False}),
    "seed23": (23, {}), "seed41": (41, {}), "scale2": (7, {"scale": 2.}),
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--case", action="append", choices=CASES,
                        help="repeat to request individual cases; default: baseline")
    result.add_argument("--out-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "trackgen-repulsive-growth")
    result.add_argument("--list-cases", action="store_true")
    return result


def source_record():
    paths = ("track_gen/_src/warp_generate_repulsive.py", "track_gen/_src/warp_pipeline.py",
             "track_gen/stages/track.py", "track_gen/_src/builtin_params.py")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    return {"checkout_revision": revision, "reference_runtime_revision": RC2_REVISION,
            "original_docs_revision": ORIGINAL_DOCS_REVISION,
            "file_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
            "platform": platform.system() + "-" + platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__}


def run_case(name, out_dir):
    sys.path.insert(0, str(ROOT))
    import warp as wp
    from track_gen import TrackGenConfig, TrackGenerator
    from track_gen.exec.context import LaunchContext
    from track_gen._src.warp_generate_repulsive import _RepulsiveScalars, _stage_schedule, _SUBDIV_RATIO

    seed, overrides = CASES[name]
    cfg = TrackGenConfig(generator="repulsive", device="cpu", num_envs=1,
                         on_invalid="keep", **overrides)
    gen = TrackGenerator(cfg, seeds=wp.array([seed], dtype=wp.int32, device="cpu"))
    track = gen.generate()
    built = gen.program._gen_scratch
    scratch = ({k.rsplit("/", 1)[-1]: v for k, v in built.scratch.items()}
               if hasattr(built, "scratch") else gen.program._scratch["generator"])
    ordinary_raw = gen.program.tap("raw.xy").numpy().reshape(cfg.num_points, 2).copy()
    count = int(track.count.numpy()[0])
    arrays = {key: getattr(track, key).numpy().reshape(cfg.N_max, 3)[:count, :2].copy()
              for key in ("center", "inner", "outer")}
    arrays.update(raw=ordinary_raw, obs=scratch["obs_pts"].numpy().copy(),
                  obs_mw=scratch["obs_mw"].numpy().copy())
    scalars = _RepulsiveScalars(cfg)
    centers, radii, slots = circles_from_rings(arrays["obs"], arrays["obs_mw"],
                                               scalars.n_wall, scalars.n_disc)
    arrays.update(circles=centers, radii=radii)
    n_ratchet = int(np.ceil(np.log(cfg.repulsive_grow_mult_max) /
                            np.log1p(cfg.repulsive_ratchet_rate)))
    starts = _stage_schedule(cfg.repulsive_stages, cfg.repulsive_ratchet_rate,
                             _SUBDIV_RATIO, n_ratchet)
    frames, substeps = [], []
    base = LaunchContext.current("cpu")

    def snapshot(iteration, n, kind="end"):
        key = f"xy_{iteration}" if kind == "end" else f"{kind}_{iteration}"
        xy = scratch["center"].numpy()[:n].copy()
        arrays[key] = xy
        reached = bool(scratch["reached"].numpy()[0])
        result = {"iteration": iteration, "n": n, "xy_key": key, "kind": kind,
                  "source_array": array_record(xy), "reached": reached,
                  "frozen": bool(scratch["frozen"].numpy()[0]),
                  "target": float(scratch["L_target"].numpy()[0]),
                  # md_out is meaningful only after a final-resolution stall check.
                  "md": (float(scratch["md_out"].numpy()[0])
                         if kind == "end" and n == cfg.repulsive_stages[-1]
                         and iteration > starts[-1]
                         and iteration % cfg.repulsive_stall_window == 0 else None),
                  "wall_forces_active": True,
                  "inner_forces_active": not (cfg.repulsive_deactivate_obstacles and reached)}
        result.update(curve_metrics(xy, centers, radii, scalars.r_dom))
        if name == "baseline" and kind == "end" and 177 <= iteration <= 192:
            result["tp_wall_energy"] = tp_wall_energy(xy, arrays["obs"][:scalars.n_wall],
                                                      scalars.r_dom)
        return result

    class Observer:
        iteration = 0
        n = cfg.repulsive_stages[0]

        def __getattr__(self, key):
            return getattr(base, key)

        def launch(self, kernel, dim, inputs=None, **kwargs):
            result = base.launch(kernel, dim, inputs, **kwargs)
            if kernel.key in ("_step_k", "_rescale_k", "_copy_unfrozen_k"):
                substeps.append(snapshot(self.iteration + 1, self.n, kernel.key))
            return result

        def bounded_loop(self, n_iters, body, **kwargs):
            def observe(ctx, iteration):
                self.iteration = iteration
                self.n = cfg.repulsive_stages[max(i for i, start in enumerate(starts)
                                                   if start <= iteration)]
                if iteration == 0:
                    frames.append(snapshot(0, cfg.repulsive_stages[0]))
                body(self, iteration)
                frames.append(snapshot(iteration + 1, self.n))
            base.bounded_loop(n_iters, observe, **kwargs)

    built.enqueue(Observer())
    observed_raw = gen.program.tap("raw.xy").numpy().reshape(cfg.num_points, 2).copy()
    if not np.array_equal(ordinary_raw, observed_raw):
        raise AssertionError("observing the solver changed its raw output")
    arrays["observed_raw"] = observed_raw
    source = source_record()
    source["warp"] = wp.__version__
    record = {"name": name, "seed": seed, "epoch": 0, "env": 0, "source": source,
              "overrides": overrides, "config": dataclasses.asdict(cfg),
              "valid": bool(track.valid.numpy()[0]), "fail": int(track.fail_mask.numpy()[0]),
              "count": count, "iterations": frames[-1]["iteration"],
              "reached": frames[-1]["reached"], "frozen": frames[-1]["frozen"],
              "stage_starts": starts, "domain_radius": scalars.r_dom,
              "n_wall": scalars.n_wall, "n_disc": scalars.n_disc,
              "disc_slots": slots.tolist(), "observed_matches_ordinary_raw": True,
              "array_archive": name + ".npz",
              "arrays": {key: array_record(value) for key, value in arrays.items()},
              "raw": curve_metrics(ordinary_raw, centers, radii, scalars.r_dom),
              "final_road_center": curve_metrics(arrays["center"], centers, radii, scalars.r_dom),
              "road_disc_gaps": road_disc_gaps(arrays["inner"], arrays["outer"], centers, radii),
              "road_wall_gap": float(scalars.r_dom - max(np.linalg.norm(arrays[key], axis=1).max()
                                                        for key in ("inner", "outer"))),
              "frames": frames, "substeps": substeps}
    if name == "baseline":
        record["late_motion"] = late_motion(arrays)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / record["array_archive"], **arrays)
    (out_dir / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("name", "valid", "fail", "iterations",
                                                  "raw", "final_road_center")}))


def main():
    args = parser().parse_args()
    if args.list_cases:
        print(json.dumps(CASES, indent=2))
        return
    for name in args.case or ["baseline"]:
        run_case(name, args.out_dir)


if __name__ == "__main__":
    main()
