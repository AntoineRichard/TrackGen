"""Digests of the runtime utilities' outputs for VALID envs (roadmap PR14).

PR14 changes what the utilities write for an env whose ``valid[e] == 0`` (the sentinel
table: an invalid env reads like a poisoned one) and adds a generation-aware self-reset,
the ``out=`` forms and the component adapters. For every VALID env the outputs must stay
byte-identical. This tool runs every utility (and both ``Course`` modes) on CPU, through
the legacy API only (names that exist at 9267388 and after PR14), at EXPLICIT capacities,
on batches where some envs are marked invalid with their geometry kept (``valid = 0`` on a
clone: what ``on_invalid="keep"`` leaves), and records the sha256 of each output field's
rows of the valid envs (per step for the stateful ones). ``tests/goldens/pr14_utilities.json``
keeps them, keyed by the PR7 platform key (float results depend on the platform and the
Warp version); ``tests/test_components.py`` recomputes them on the current tree.

Capture (from the repository root, against a checkout of the base, 9267388)::

    PYTHONPATH=<9267388 tree> python tests/tools/make_pr14_goldens.py

It adds or replaces this platform's entry.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import warp as wp

GOLDEN = Path(__file__).resolve().parents[1] / "goldens" / "pr14_utilities.json"

DEV = "cpu"
E = 6
#: Envs marked invalid (geometry kept) on the clones the utilities read.
INVALID = (1, 4)
STEPS = 20


#: The base's CheckpointSet fields (PR14 appends ``valid`` / ``generation``).
_CHECKPOINT_FIELDS = ("position", "left", "right", "tangent", "up_half", "count")


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine()}-warp{wp.__version__}"


def _sha(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        a = np.ascontiguousarray(c)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _valid_rows(arr: "wp.array", valid: np.ndarray):
    """The rows of the valid envs of a flat ``[E * k]`` array (``k`` slots per env)."""
    host = np.ascontiguousarray(arr.numpy())
    k = host.shape[0] // E
    rows = host.reshape((E, k) + host.shape[1:])
    return rows[valid.astype(bool)]


class _Digest:
    """``field -> list of valid-env row chunks`` (one per step), hashed at the end."""

    def __init__(self) -> None:
        self.chunks: dict = {}

    def add(self, prefix: str, obj, valid: np.ndarray) -> None:
        import dataclasses

        if isinstance(obj, wp.array):
            items = {"": obj}
        elif isinstance(obj, dict):
            items = obj
        else:
            items = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)
                     if isinstance(getattr(obj, f.name), wp.array)}
        for name, arr in items.items():
            key = f"{prefix}.{name}" if name else prefix
            self.chunks.setdefault(key, []).append(_valid_rows(arr, valid))

    def digests(self) -> dict:
        return {k: _sha(v) for k, v in sorted(self.chunks.items())}


def _i32(a) -> "wp.array":
    return wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=DEV)


def _keep_invalid(result):
    """A clone of ``result`` with the envs of :data:`INVALID` marked invalid, geometry
    kept (``on_invalid="keep"``)."""
    k = result.clone()
    v = k.valid.numpy()
    v[list(INVALID)] = 0
    wp.copy(k.valid, _i32(v))
    return k


def _points(src, s: int, *, gates: bool = False) -> np.ndarray:
    """One deterministic point per env on the geometry (a point slightly off the line,
    walking around it), from the host copy."""
    pts = (src.position if gates else src.center).numpy()
    n = pts.shape[0] // E
    p = pts.reshape(E, n, 3)
    cnt = src.count.numpy()
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = int(min(cnt[e], n))
        if m >= 1:
            a = p[e, (s * 3 + 2 * e) % m]
            b = p[e, (s * 3 + 2 * e + 1) % m]
            out[e] = a + (b - a) * np.float32(0.4) + np.float32([0.01 * ((s + e) % 3 - 1),
                                                                   0.005 * (e % 2), 0.0])
    return out


def _boxes(n: int, s: int):
    yaw = np.float32(0.3 * s)
    q = np.tile(np.float32([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]), (n, 1))
    he = np.tile(np.float32([0.03, 0.015]), (n, 1))
    return (wp.array(q, dtype=wp.quatf, device=DEV),
            wp.array(he, dtype=wp.vec2f, device=DEV))


def _track(**kw):
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator

    gen = TrackGenerator(TrackGenConfig(num_envs=E, device=DEV, **kw),
                         PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV))
    return gen.generate()


def _gates():
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

    gen = GateGenerator(GateGenConfig(num_envs=E, device=DEV, gate_width=0.1),
                        PerEnvSeededRNG(seeds=5, num_envs=E, device=DEV))
    return gen.generate()


def _track_utilities(track) -> dict:
    """Every track utility on ``track`` (a keep-invalid clone)."""
    from track_gen.checkpoints import CheckpointSampler
    from track_gen.collision import CollisionChecker
    from track_gen.heightfield import HeightFieldBaker
    from track_gen.localize import TrackLocalizer, curvature, speed_profile
    from track_gen.progress import ProgressTracker
    from track_gen.props import PropSampler

    valid = track.valid.numpy()
    d = _Digest()
    # Samplers (explicit capacities).
    for boundary in ("outer", "inner"):
        for mode in ("points", "segments"):
            ps = PropSampler(track, 0.15, boundary=boundary, mode=mode, max_props=96)
            d.add(f"props[{boundary},{mode}]", ps.sample(), valid)
    cs = CheckpointSampler(track, 0.5, max_checkpoints=48)
    cps = cs.sample()
    d.add("checkpoints", {k: getattr(cps, k) for k in _CHECKPOINT_FIELDS}, valid)
    d.add("checkpoints.sampler", {"truncated": cs.truncated, "step": cs.step}, valid)
    # Bakes.
    hf = HeightFieldBaker(track, 16)
    d.add("heightfield", hf.bake(), valid)
    sdf = CollisionChecker(track, max_boxes=2, method="sdf", sdf_resolution=16)
    sdf.bake()
    d.add("sdf", {"lo": sdf._sdf_lo, "hi": sdf._sdf_hi, "phi": sdf._sdf_phi,
                  "bid": sdf._sdf_bid}, valid)
    seg = CollisionChecker(track, max_boxes=2, method="segments")
    # Per-generation helpers.
    k0 = curvature(track, window=0)
    k2 = curvature(track, window=2)
    d.add("curvature[w0]", k0, valid)
    d.add("curvature[w2]", k2, valid)
    d.add("speed_profile", speed_profile(track, 4.0, 2.0, 3.0, 5.0, kappa=k2), valid)
    d.add("speed_profile[own kappa]", speed_profile(track, 3.0, 1.0, 2.0, 4.0, window=1),
          valid)
    # Stateful / per-step utilities along a trajectory.
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    boxes = wp.zeros(2 * E, dtype=wp.vec3f, device=DEV)
    tracker = ProgressTracker(cps, position=pos)
    cold = TrackLocalizer(track, position=pos)
    warm = TrackLocalizer(track, warm_window=3, position=pos)
    for s in range(STEPS):
        p = _points(track, s)
        wp.copy(pos, wp.array(p, dtype=wp.vec3f, device=DEV))
        b = np.repeat(p, 2, axis=0)
        b[1::2] += np.float32([0.04, -0.03, 0.0])
        if s % 7 == 3:
            b[::4] = np.nan                       # inactive slots
        wp.copy(boxes, wp.array(b, dtype=wp.vec3f, device=DEV))
        q, he = _boxes(2 * E, s)
        d.add("progress", tracker.update(), valid)
        d.add("localize[cold]", cold.query(), valid)
        d.add("localize[warm]", warm.query(), valid)
        d.add("collision[segments]", seg.query(boxes, q, he), valid)
        d.add("collision[sdf]", sdf.query(boxes, q, he), valid)
        if s == 11:
            m = _i32([1, 0, 1, 0, 1, 0])
            tracker.reset(m)
            warm.reset(m)
    return d.digests()


def _gate_utilities(seq) -> dict:
    from track_gen.checkpoints import CheckpointSet
    from track_gen._src.collision_frames import FrameChecker
    from track_gen._src.course_line import CourseLine
    from track_gen.collision import DiscChecker
    from track_gen.localize import TrackLocalizer
    from track_gen.progress import ProgressTracker

    valid = seq.valid.numpy()
    d = _Digest()
    line = CourseLine(seq, 8)
    line.refresh()
    d.add("course_line", {k: getattr(line.centerline, k) for k in
                          ("center", "tangent", "arclen", "length", "count", "valid")},
          valid)
    G = int(seq.position.shape[0]) // E
    left, right = seq.left.numpy(), seq.right.numpy()
    posts_h = np.empty((2 * E * G, 3), np.float32)
    posts_h[0::2], posts_h[1::2] = left, right
    posts = wp.array(posts_h, dtype=wp.vec3f, device=DEV)
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    q, he = _boxes(E, 0)
    discs = DiscChecker(posts, radius=0.04, max_boxes=1, num_envs=E, position=pos,
                        orientation=q, half_extents=he)
    tracker = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    frames = FrameChecker(seq, E, radius=0.05, frame_thickness=0.02, frame_depth=0.02)
    frames.bind_inputs(pos)
    frames.bind_window(tracker.next_checkpoint_state)
    loc = TrackLocalizer(line.centerline, warm_window=2, position=pos)
    for s in range(STEPS):
        p = _points(seq, s, gates=True)
        wp.copy(pos, wp.array(p, dtype=wp.vec3f, device=DEV))
        d.add("discs", discs.query(), valid)
        d.add("progress[gates]", tracker.update(), valid)
        d.add("frames", frames.query(), valid)
        d.add("localize[line,warm]", loc.query(), valid)
    return d.digests()


def _course_cases() -> dict:
    """Both Course modes at explicit capacities over a trajectory with steps, a respawn
    and masked advancing regenerations (rows of the envs valid at each step)."""
    from track_gen import GateGenConfig, TrackGenConfig
    from track_gen.course import Course, CourseConfig

    cases = {
        "course[track,segments,hf]": CourseConfig(
            mode="track", gen=TrackGenConfig(num_envs=E, device=DEV), seeds=3,
            collision="segments", heightfield_resolution=16, checkpoint_spacing=0.5,
            max_checkpoints=48),
        "course[track,sdf,boxes2]": CourseConfig(
            mode="track", gen=TrackGenConfig(num_envs=E, device=DEV), seeds=4,
            collision="sdf", sdf_resolution=16, max_boxes=2, checkpoint_spacing=0.4,
            max_checkpoints=64),
        "course[gates,posts,warm]": CourseConfig(
            mode="gates", gen=GateGenConfig(num_envs=E, device=DEV, gate_width=0.1),
            seeds=7, post_radius=0.03, localize_window=3),
        "course[gates,frames]": CourseConfig(
            mode="gates", gen=GateGenConfig(num_envs=E, device=DEV, gate_width=0.1),
            seeds=8, frame_collision=True, agent_radius=0.05, frame_thickness=0.02,
            frame_depth=0.02),
    }
    out = {}
    for name, cfg in cases.items():
        c = Course(cfg)
        B = int(cfg.max_boxes)
        pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
        bp = wp.zeros(E * B, dtype=wp.vec3f, device=DEV) if B > 1 else None
        q, he = _boxes(E * B, 1)
        if cfg.collision is not None or cfg.post_radius > 0.0:
            c.bind(position=pos, orientation=q, half_extents=he, box_position=bp)
        else:
            c.bind(position=pos)
        d = _Digest()
        gates = cfg.mode == "gates"
        for s in range(STEPS):
            p = _points(c.result, s, gates=gates)
            wp.copy(pos, wp.array(p, dtype=wp.vec3f, device=DEV))
            if bp is not None:
                wp.copy(bp, wp.array(np.repeat(p, B, axis=0), dtype=wp.vec3f, device=DEV))
            sr = c.step()
            valid = c.result.valid.numpy()
            d.add("events", sr.events, valid)
            if sr.contacts is not None:
                d.add("contacts", sr.contacts, valid)
            if sr.frame is not None:
                d.add("frame", sr.frame, valid)
            if not gates:
                d.add("checkpoints", {k: getattr(c.checkpoints, k)
                                      for k in _CHECKPOINT_FIELDS}, valid)
            if s == 5:
                c.reset(_i32([0, 1, 1, 0, 0, 1]))
            if s == 9:
                c.regenerate(mask=_i32([1, 0, 0, 1, 0, 0]), advance=True)
            if s == 14:
                c.regenerate(mask=_i32([0, 0, 1, 0, 1, 1]), advance=True)
        out[name] = d.digests()
    return out


def compute() -> dict:
    """``case -> {field: sha256}`` on the current tree (CPU)."""
    wp.config.quiet = True
    wp.init()
    out = {}
    for name, kw in (("track[flat]", {}),
                     ("track[uniform z]", dict(z_profile="uniform", z_min=0.0,
                                               z_max=0.6))):
        out[name] = _track_utilities(_keep_invalid(_track(**kw)))
    out["gates"] = _gate_utilities(_keep_invalid(_gates()))
    out.update(_course_cases())
    return out


def load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


def main() -> None:
    data = load()
    data[platform_key()] = compute()
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} [{platform_key()}]: {sorted(data[platform_key()])}")


if __name__ == "__main__":
    main()
