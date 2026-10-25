"""Per-stage-boundary digests of the 0.2.0.dev3 gate pipeline (roadmap PR11).

PR11 decomposes ``warp_gate._run_gate_pipeline`` into the stages of a gate
:class:`~track_gen.pipeline.Program`. This tool drives the LIVE dd181f0 path (the
``GateGenerator`` facade, with the boundary functions of ``warp_gate`` /
``warp_zprofile`` wrapped: they are looked up at call time) and records, per case of
:data:`CASES`, the array visible at every stage boundary under the channel name the
Program gives it (:data:`TAPS`). ``tests/goldens/pr11_gate_stages.json`` keeps ONE digest
per case (:func:`digest`: the sha256 over the sorted ``"tap:sha256"`` lines) and, for
:data:`LAUNCH_CASES`, the launch sequence of one ``generate()`` (``[op, kernel, dim,
scalars]``, :func:`launch_rows`). Float results depend on the platform and the Warp
version, so the file is keyed by :func:`platform_key` (the PR7 key).

``tests/_legacy_gate_pr11.py`` is the frozen reference (the HEAD code PR11 edits or
deletes, the unchanged kernels imported live); ``tests/test_gate_pipeline_parity.py``
checks that the reference reproduces these digests and that every Program tap equals
the reference array.

Capture (from the repository root, against a checkout of dd181f0)::

    PYTHONPATH=<dd181f0 tree> python tests/tools/capture_pr11_gate_stages.py

It adds or replaces this platform's entry. ``--diff`` prints, per case, the digest of
every tap (to locate a mismatch) instead of writing.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import warp as wp

GOLDEN = Path(__file__).resolve().parents[1] / "goldens" / "pr11_gate_stages.json"

E = 4
SEEDS = (0, 1, 7, 99991)

#: The nine GateSequence fields (the PR7 fingerprint set).
FINAL_FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left",
                "right", "valid", "count")

#: Every generator x the orderings it supports at dd181f0.
GEN_ORDERINGS = (("bezier", "ccw"), ("bezier", "random_pairs"), ("hull", "ccw"),
                 ("hull", "random_pairs"), ("polar", "ccw"), ("polar", "raw"),
                 ("voronoi", "ccw"), ("voronoi", "raw"), ("checkpoint", "ccw"),
                 ("checkpoint", "raw"))
NORMALIZING = ("polar", "voronoi", "checkpoint")
Z_PROFILES = {
    "flat": {},
    "uniform": dict(z_profile="uniform", z_max=1.5),
    "random_walk": dict(z_profile="random_walk", z_max=1.5, z_max_step=0.3),
    "noise": dict(z_profile="noise", z_base=0.1, z_max=1.5, z_noise_amplitude=0.5,
                  z_noise_harmonics=4),
}
SOLVE_ITERS = (0, 8)
ALIGNS = ("yaw_only", "full_tangent")
WIDTHS = (0.0, 0.04)
#: The capacity of each built-in under the default GateGenConfig (min_gates extra case).
CAPACITY = {"bezier": 13, "hull": 13, "polar": 12, "voronoi": 18, "checkpoint": 12}


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine()}-warp{wp.__version__}"


def _cases() -> dict:
    cases = {}
    for gen, order in GEN_ORDERINGS:
        base = dict(generator=gen, gate_ordering=order)
        for zname, zkw in Z_PROFILES.items():
            for iters in SOLVE_ITERS:
                for align in ALIGNS:
                    for width in WIDTHS:
                        cases[f"{gen}/{order}/{zname}/i{iters}/{align}/w{width}"] = dict(
                            base, **zkw, gate_solve_iters=iters, gate_align=align,
                            gate_width=width)
        extra = dict(base, gate_solve_iters=8, gate_width=0.04)
        cases[f"{gen}/{order}/extra/radius0"] = dict(extra, gate_radius=0.0)
        cases[f"{gen}/{order}/extra/scale1.7"] = dict(extra, scale=1.7)
        cases[f"{gen}/{order}/extra/min_gates"] = dict(extra, min_gates=CAPACITY[gen])
        cases[f"{gen}/{order}/extra/grade"] = dict(extra, **Z_PROFILES["uniform"],
                                                   z_valid_grade=0.8)
    return cases


#: case id -> GateGenConfig keyword arguments (num_envs / device are E / "cpu").
CASES = _cases()
#: Historical toy cases, retained as independent numerical parity references.
LEGACY_CASES = {
    "legacy/toy.ring": dict(generator="toy.ring", gate_ordering="raw"),
    "legacy/toy.ring/noise": dict(generator="toy.ring", gate_ordering="raw",
                                  **Z_PROFILES["noise"], gate_width=0.04),
    "legacy/tests.zero-count": dict(generator="tests.zero-count", min_gates=2,
                                    max_gates=2),
}
#: The cases whose launch sequence is recorded: width 0 (the default), yaw_only, flat
#: and noise, solve iterations 0 and 8, every generator x ordering.
LAUNCH_CASES = tuple(f"{gen}/{order}/{z}/i{it}/yaw_only/w0.0"
                     for gen, order in GEN_ORDERINGS for z in ("flat", "noise")
                     for it in SOLVE_ITERS)


def zero_count_spec():
    """A legacy gate generator that writes nothing (``count`` stays 0)."""
    import warp as wp

    from track_gen.plugin import Stage

    class ZeroCount(Stage):
        name = "tests.zero-count"
        kind = "gate_generator"
        output = "anchors"
        writes = {"xy": "anchors.xy", "count": "anchors.count"}
        orderings = frozenset({"ccw"})

        def capacity(self, p, layout):
            return {"gates": 2}

        def build(self, b, p):
            xy = b.write("xy", wp.vec2f, per="anchors")
            count = b.write("count", wp.int32)

            def enqueue(ctx):
                ctx.fill(xy, wp.vec2f(0.0, 0.0))
                ctx.fill(count, 0)
            return enqueue

    return ZeroCount


def register_legacy(reg) -> None:
    """Add the legacy toys of :data:`LEGACY_CASES` to ``reg``."""
    from tests._plugin_toys import TOY_RING

    reg.add(TOY_RING)
    reg.add(zero_count_spec())


def _sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def tap_digests(taps: dict) -> dict:
    return {k: _sha(v) for k, v in sorted(taps.items())}


def digest(taps: dict) -> str:
    """One digest per case: the sha256 over the sorted ``"tap:sha256"`` lines."""
    lines = "\n".join(f"{k}:{v}" for k, v in tap_digests(taps).items())
    return hashlib.sha256(lines.encode()).hexdigest()


def launch_rows(trace) -> list:
    """``[op, kernel, dim, scalars]`` of each record, JSON-normalised."""
    rows = [[r.op, r.kernel, list(r.dim), [[k, v] for k, v in r.scalars]] for r in trace]
    return json.loads(json.dumps(rows))


def _config(kw: dict):
    from track_gen import GateGenConfig

    return GateGenConfig(num_envs=E, device="cpu", **kw)


def _rng():
    from track_gen import PerEnvSeededRNG

    return PerEnvSeededRNG(seeds=wp.array(np.asarray(SEEDS, np.int32), dtype=wp.int32,
                                          device="cpu"), num_envs=E, device="cpu")


def capture_head(kw: dict, *, legacy: bool = False) -> dict:
    """The taps of one ``generate()`` of the LIVE dd181f0 ``GateGenerator`` (CPU)."""
    from track_gen import GateGenerator
    from track_gen._src import warp_gate, warp_zprofile

    taps: dict = {}

    def snap(a):
        return a.numpy().copy()

    orig = {}

    def patch(mod, name, fn):
        orig[(mod, name)] = getattr(mod, name)
        setattr(mod, name, fn)

    def order_points(seeds_wp, src, src_stride, count, max_gates, ordering, keys, dst,
                     ctx=None):
        taps["anchors.xy"] = snap(src)
        taps["anchors.count"] = snap(count)
        orig[(warp_gate, "order_points")](seeds_wp, src, src_stride, count, max_gates,
                                          ordering, keys, dst, ctx=ctx)
        taps["gates.xy@order"] = snap(dst)

    def normalize_positions(position, max_gates, count, target_extent, ctx=None):
        orig[(warp_gate, "normalize_positions")](position, max_gates, count,
                                                 target_extent, ctx=ctx)
        taps["gates.xy@normalize"] = snap(position)

    def finish_ordered_gates(*args, **kwargs):
        orig[(warp_gate, "finish_ordered_gates")](*args, **kwargs)
        out = args[7] if len(args) > 7 else kwargs["out"]
        taps["gates.count"] = snap(out.count)

    def relax_gate_spheres(position, max_gates, count, target_distance, iterations,
                           ctx=None):
        if legacy and "gates.xy@generator" not in taps:
            taps["gates.xy@generator"] = snap(position)
        orig[(warp_gate, "relax_gate_spheres")](position, max_gates, count,
                                                target_distance, iterations, ctx=ctx)
        if float(target_distance) > 0.0 and int(iterations) > 0:
            taps["gates.xy@gate_relax"] = snap(position)

    def gate_cum_perim(pos2, count, max_gates, cum, perim, ctx=None):
        if legacy and "gates.xy@generator" not in taps:
            taps["gates.xy@generator"] = snap(pos2)
        orig[(warp_zprofile, "gate_cum_perim")](pos2, count, max_gates, cum, perim,
                                                ctx=ctx)
        taps["gates.cum"] = snap(cum)
        taps["gates.perim"] = snap(perim)

    def apply_z_profile(config, seeds_wp, count, stride, cum, perim, z, ctx=None):
        orig[(warp_zprofile, "apply_z_profile")](config, seeds_wp, count, stride, cum,
                                                 perim, z, ctx=ctx)
        taps["gates.z"] = snap(z)

    def tangents_from_positions(position, tangent, max_gates, count, ctx=None):
        taps["gates.position@lift"] = snap(position)
        orig[(warp_gate, "tangents_from_positions")](position, tangent, max_gates, count,
                                                     ctx=ctx)
        taps["gates.tangent@tangents"] = snap(tangent)

    def finalize_gate_sequence(gates, config, fallbacks=None, ctx=None):
        orig[(warp_gate, "finalize_gate_sequence")](gates, config, fallbacks=fallbacks,
                                                    ctx=ctx)
        for f in FINAL_FIELDS:
            taps[f] = snap(getattr(gates, f))
        taps["fallbacks"] = snap(fallbacks)

    gen = GateGenerator(_config(kw), _rng())          # warm-up runs unwrapped
    for mod, name, fn in ((warp_gate, "order_points", order_points),
                          (warp_gate, "normalize_positions", normalize_positions),
                          (warp_gate, "finish_ordered_gates", finish_ordered_gates),
                          (warp_gate, "relax_gate_spheres", relax_gate_spheres),
                          (warp_zprofile, "gate_cum_perim", gate_cum_perim),
                          (warp_zprofile, "apply_z_profile", apply_z_profile),
                          (warp_gate, "tangents_from_positions", tangents_from_positions),
                          (warp_gate, "finalize_gate_sequence", finalize_gate_sequence)):
        patch(mod, name, fn)
    try:
        gen.generate()
    finally:
        for (mod, name), fn in orig.items():
            setattr(mod, name, fn)
    if kw["generator"] == "tests.zero-count":
        # The staging buffer is uninitialised (wp.empty) and never written: only the
        # finalized fields are defined.
        taps = {k: v for k, v in taps.items() if k in FINAL_FIELDS or k == "fallbacks"}
    return taps


def capture_launches(kw: dict) -> list:
    from track_gen import GateGenerator
    from track_gen.testing import trace_launches

    gen = GateGenerator(_config(kw), _rng())
    with trace_launches() as trace:
        gen.generate()
    return launch_rows(trace)


def compute() -> dict:
    from track_gen.plugin import registry

    wp.config.quiet = True
    digests, per_tap = {}, {}
    for case, kw in CASES.items():
        taps = capture_head(kw)
        digests[case] = digest(taps)
        per_tap[case] = tap_digests(taps)
    with registry.scoped():
        register_legacy(registry)
        for case, kw in LEGACY_CASES.items():
            taps = capture_head(kw, legacy=True)
            digests[case] = digest(taps)
            per_tap[case] = tap_digests(taps)
    launches = {case: capture_launches(CASES[case]) for case in LAUNCH_CASES}
    return {"digests": digests, "launches": launches}, per_tap


def load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


if __name__ == "__main__":
    data, per_tap = compute()
    if "--diff" in sys.argv[1:]:
        print(json.dumps(per_tap, indent=1, sort_keys=True))
        sys.exit(0)
    golden = load()
    golden[platform_key()] = data
    GOLDEN.write_text(json.dumps(golden, indent=0, sort_keys=True,
                                 separators=(",", ":")) + "\n")
    print(f"wrote {len(data['digests'])} digests and {len(data['launches'])} launch "
          f"sequences for {platform_key()} -> {GOLDEN}")
