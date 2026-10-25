"""Launch-trace and state digests of the 0.x ``Course(CourseConfig)`` path (roadmap PR15).

PR15 rebuilds ``Course`` as the composition root (``Course(generator, components)``, the
presets, the regeneration body over a component list). A ``Course(CourseConfig)`` must keep
its outputs AND its launch sequences: this tool drives ten configurations (the PR14
variants plus the defaulted capacities: the plan bound and the elevated readback) through
construction, a bind and 40 steps with every call of the 0.x surface interleaved (a masked
advancing regeneration, a respawn, a plain and an advance-all regeneration, both reseeds,
``regenerate(mask=inputs["regen_mask"])``, ``enqueue_generate``), on CPU, through names that
exist at d58ac55 and after PR15 only. Per call it records the sha256 of the launch trace's
topology (kernel, dims, scalars, every array argument by ROLE: its label in
``buffers_of(generator, rng, _regen_mask, _regen_flag, _reset_all_mask, _reseed_buf,
*_components, *bound arrays, masks)``), and after every step the sha256 of every state
buffer (result, checkpoints, step outputs, posts, sampler truncation, heightfield, course
line, progress and localizer state, seeds and epochs). ``tests/goldens/pr15_course.json``
keeps them, keyed by the PR7 platform key (float results depend on the platform and the
Warp version), plus a platform-independent ``"topology"`` entry: the launch-trace
digests (``construct``, ``calls``) and ``max_checkpoints``, whose kernels, dims, scalars
(host-computed) and buffer roles do not depend on float results.
``tests/test_course_composition.py`` recomputes them on the current tree: the launches
on every platform (this platform's entry when present, else ``"topology"``), the states
where this platform has an entry.

Capture (from the repository root, against a checkout of the base, d58ac55, made BEFORE
``track_gen/_src/course.py`` is edited)::

    git archive d58ac55 track_gen tests | tar -x -C /tmp/base
    cp tests/tools/make_pr15_course_goldens.py /tmp/base/tests/tools/
    PYTHONPATH=/tmp/base python /tmp/base/tests/tools/make_pr15_course_goldens.py

then merge the new platform entry of ``/tmp/base/tests/goldens/pr15_course.json`` into
this tree's (the tool writes beside itself). It adds or replaces this platform's entry,
writes ``"topology"`` when the file has none, and reports a platform whose launches
differ from it (its own entry then takes precedence there).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import warp as wp

GOLDEN = Path(__file__).resolve().parents[1] / "goldens" / "pr15_course.json"
#: The platform-independent entry: the launch digests of every case.
TOPOLOGY = "topology"
#: The digests of a case that ``TOPOLOGY`` keeps.
TOPOLOGY_KEYS = ("construct", "calls", "max_checkpoints")

DEV = "cpu"
E = 6
STEPS = 40

#: name -> (mode, CourseConfig keywords): the PR14 E8 variants plus the defaulted
#: capacities (the plan bound, 70 at spacing 0.5; the elevated readback).
CASES = {
    "track[segments]": ("track", dict(collision="segments", checkpoint_spacing=0.5,
                                      max_checkpoints=64)),
    "track[sdf+heightfield]": ("track", dict(collision="sdf", sdf_resolution=16,
                                             heightfield_resolution=16,
                                             checkpoint_spacing=0.5, max_checkpoints=64)),
    "track[segments,2 boxes]": ("track", dict(collision="segments", max_boxes=2,
                                              checkpoint_spacing=0.4, max_checkpoints=80)),
    "track[no collision]": ("track", dict(collision=None, checkpoint_spacing=0.5,
                                          max_checkpoints=64)),
    "track[plan bound]": ("track", dict(collision="segments", checkpoint_spacing=0.5)),
    "track[elevated readback]": ("track-elev", dict(collision="segments",
                                                    checkpoint_spacing=0.5)),
    "gates[posts]": ("gates", dict(post_radius=0.03, localize_window=3)),
    "gates[posts,2 boxes]": ("gates", dict(post_radius=0.03, max_boxes=2)),
    "gates[frames]": ("gates", dict(frame_collision=True, agent_radius=0.05,
                                    frame_thickness=0.02, frame_depth=0.02)),
    "gates[none]": ("gates", dict()),
}


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine()}-warp{wp.__version__}"


def _sha(obj) -> str:
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


def _sha_bytes(items) -> str:
    h = hashlib.sha256()
    for k, v in items:
        h.update(k.encode())
        h.update(v)
    return h.hexdigest()[:16]


def _i32(a) -> "wp.array":
    return wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=DEV)


def _gen_cfg(mode: str):
    from track_gen import GateGenConfig, TrackGenConfig

    if mode == "track":
        return TrackGenConfig(num_envs=E, device=DEV)
    if mode == "track-elev":
        return TrackGenConfig(num_envs=E, device=DEV, z_profile="uniform", z_min=0.0,
                              z_max=2.0)
    return GateGenConfig(num_envs=E, device=DEV, gate_width=0.3)


def _state(c) -> dict:
    """``name -> bytes`` of every state buffer of course ``c`` (0.x names only)."""
    from track_gen.testing.capture import result_arrays

    out = {}

    def add(prefix, obj):
        for k, v in result_arrays(obj).items():
            out[f"{prefix}.{k}"] = np.ascontiguousarray(v.numpy()).tobytes()

    add("result", c.result)
    add("checkpoints", c.checkpoints)
    out["epoch"] = c.seeds.epoch.numpy().tobytes()
    out["seeds"] = c.seeds.seeds.numpy().tobytes()
    sr = c._step_result
    if sr is not None:
        for part in ("events", "contacts", "frame"):
            obj = getattr(sr, part)
            if obj is not None:
                add(part, obj)
        out["valid"] = sr.valid.numpy().tobytes()
    if c._posts is not None:
        out["posts"] = c._posts.numpy().tobytes()
    if c.checkpoint_sampler is not None:
        out["sampler.truncated"] = c.checkpoint_sampler.truncated.numpy().tobytes()
    if c.heightfield is not None:
        add("heightfield", c.heightfield._hf)
    if c.course_line is not None:
        add("line", c.course_line.centerline)
    p = c.progress
    for k in ("_prev_pos", "_next", "_laps", "_progress", "_seen"):
        out[f"progress.{k}"] = getattr(p, k).numpy().tobytes()
    if c.localizer is not None:
        out["localizer._last"] = c.localizer._last.numpy().tobytes()
        out["localizer._seen"] = c.localizer._seen.numpy().tobytes()
    return out


def _roles(c, extra=()):
    """Every buffer a launch of ``c`` may touch, labelled by role."""
    from track_gen.testing.trace import buffers_of

    objs = [c.generator, c.rng, c._regen_mask, c._regen_flag, c._reset_all_mask,
            c._reseed_buf] + list(c._components) + list(extra)
    names = ["gen", "rng", "regen_mask", "regen_flag", "reset_all", "reseed"] + \
        [f"comp{i}" for i in range(len(c._components))] + \
        [f"x{i}" for i in range(len(extra))]
    # The private inflation view now aliases the elevation's parameter arrays. Keep
    # their original latch roles so that adding a second alias changes no trace digest.
    gen = c.generator
    dyn = getattr(getattr(getattr(gen, "_scratch", None), "inflate", None), "z_dyn", None)
    if dyn:
        pointers = {int(a.ptr) for a in dyn.values()}
        for i, (_, work) in enumerate(gen._program._pairs):
            if int(work.ptr) in pointers:
                objs.insert(0, work)
                names.insert(0, f"gen._program._pairs[{i}][1]")
    return buffers_of(*objs, root=names)


def _topology(trace, roles) -> "tuple[str, list]":
    """``"<launches>:<sha>"`` of a trace's topology, and the topology itself."""
    topo = trace.topology(roles, scalars=True)
    return f"{len(topo)}:{_sha(topo)}", topo


def _bind(c, B: int):
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    if c._needs_boxes():
        ori = wp.array(np.tile(np.float32([0, 0, 0, 1]), (E * B, 1)), dtype=wp.quatf,
                       device=DEV)
        he = wp.array(np.full((E * B, 2), 0.03, np.float32), dtype=wp.vec2f, device=DEV)
        bp = wp.zeros(E * B, dtype=wp.vec3f, device=DEV) if B > 1 else None
        c.bind(pos, ori, he, bp)
        return pos, bp, [pos, ori, he] + ([bp] if bp is not None else [])
    c.bind(position=pos)
    return pos, None, [pos]


def _positions(c, s: int) -> np.ndarray:
    if c._cfg.mode == "track":
        pts, cnt = c.result.center, c.result.count
    else:
        pts, cnt = c.course_line.centerline.center, c.course_line.centerline.count
    n = int(pts.shape[0]) // E
    p = pts.numpy().reshape(E, n, 3)
    k = cnt.numpy()
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = int(min(k[e], n))
        if m >= 3:
            out[e] = p[e, (s * 7 + 5 * e) % m] + np.float32([0.01 * ((s + e) % 3), 0.0, 0.0])
    return out


def _calls(s: int, seeds_arr) -> list:
    calls = [("step", lambda c: c.step())]
    if s == 9:
        calls.append(("regenerate(mask, advance)",
                      lambda c: c.regenerate(mask=c._g_m1, advance=True)))
    if s == 17:
        calls.append(("reset(mask)", lambda c: c.reset(c._g_m2)))
    if s == 23:
        calls.append(("regenerate()", lambda c: c.regenerate()))
    if s == 25:
        calls.append(("regenerate(advance)", lambda c: c.regenerate(advance=True)))
    if s == 29:
        calls.append(("generate(seeds=int)", lambda c: c.generate(seeds=77)))
    if s == 31:
        calls.append(("generate(seeds=array)", lambda c: c.generate(seeds=seeds_arr)))
    if s == 33:
        calls.append(("regenerate(inputs mask)", lambda c: c.regenerate(
            mask=c.inputs["regen_mask"], advance=True)))
    if s == 35:
        calls.append(("enqueue_generate(mask)", lambda c: c.enqueue_generate(
            mask=c._g_m1, advance=True)))
    return calls


def run_case(name: str) -> "tuple[dict, dict]":
    """``(digests, topologies)`` of one case: ``digests`` = ``{"construct", "calls":
    {call: digest}, "state": [per-step digest], "final": {field: digest},
    "max_checkpoints"}``; ``topologies`` = ``call -> topology`` (diagnostics)."""
    from track_gen.course import Course, CourseConfig
    from track_gen.testing import trace_launches

    mode, kw = CASES[name]
    cfg = CourseConfig(mode="gates" if mode == "gates" else "track", gen=_gen_cfg(mode),
                       seeds=11, **kw)
    with trace_launches() as tc:
        c = Course(cfg)
    B = int(kw.get("max_boxes", 1))
    pos, bp, bound = _bind(c, B)
    c._g_m1 = _i32([1, 0, 1, 0, 0, 1])
    c._g_m2 = _i32([0, 1, 1, 0, 1, 0])
    seeds_arr = _i32(np.arange(E) + 300)
    extra = bound + [c._g_m1, c._g_m2, seeds_arr]
    topos = {}
    d, topos["construct"] = _topology(tc, _roles(c, bound))
    out = {"construct": d, "calls": {}, "state": [],
           "max_checkpoints": getattr(c.checkpoint_sampler, "max_checkpoints", None)}
    for s in range(STEPS):
        x = _positions(c, s)
        wp.copy(pos, wp.array(x, dtype=wp.vec3f, device=DEV))
        if bp is not None:
            wp.copy(bp, wp.array(np.repeat(x, B, axis=0), dtype=wp.vec3f, device=DEV))
        for label, fn in _calls(s, seeds_arr):
            with trace_launches() as t:
                fn(c)
            key = f"{s:02d} {label}"
            out["calls"][key], topos[key] = _topology(t, _roles(c, extra))
        st = _state(c)
        out["state"].append(_sha_bytes(sorted(st.items())))
    out["final"] = {k: _sha_bytes([(k, v)]) for k, v in sorted(_state(c).items())}
    return out, topos


def compute() -> dict:
    """``case -> digests`` on the current tree (CPU)."""
    wp.config.quiet = True
    wp.init()
    return {name: run_case(name)[0] for name in CASES}


def load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


def topology_of(digests: dict) -> dict:
    """``case -> {construct, calls, max_checkpoints}`` of :func:`compute`'s output."""
    return {name: {k: d[k] for k in TOPOLOGY_KEYS} for name, d in digests.items()}


def main() -> None:
    data = load()
    data[platform_key()] = compute()
    topo = topology_of(data[platform_key()])
    if TOPOLOGY not in data:
        data[TOPOLOGY] = topo
    elif data[TOPOLOGY] != topo:
        print(f"note: the launches on {platform_key()} differ from the {TOPOLOGY!r} entry "
              f"in {sorted(k for k in topo if topo[k] != data[TOPOLOGY].get(k))}; this "
              f"platform's own entry is used here")
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} [{platform_key()}]: {sorted(data[platform_key()])}")


if __name__ == "__main__":
    main()
