"""Sphere-vs-gate-frame collision for 3D gate courses.

Each square gate is four thin oriented boxes in the gate's local frame
(x=forward/depth, y=left, z=up): posts at y = +/-(hs + t/2) spanning the
opening height plus corners, bars at z = +/-(hs + t/2). The agent is a
sphere; a hit is sphere-vs-box penetration against any frame member of the
gates inside a fixed index window around the CURRENT target checkpoint
(prev, target, and the next ``window - 2`` gates) — fixed-size, so the
query is one kernel, allocation-free and capture-safe.
"""
from dataclasses import dataclass

import warp as wp

from .collision_geom import _is_nan3
from .runtime import _check_arr, _init, _sync
from .types import GateSequence


@dataclass
class FrameContact:
    """Per-env frame-collision result; overwritten in place per query.

    hit : [E] int32 — 1 iff the sphere penetrates any frame box this query.
    depth : [E] float32 — max penetration depth, 0.0 when no hit.
    """

    hit: wp.array
    depth: wp.array

    def clone(self) -> "FrameContact":
        return FrameContact(hit=wp.clone(self.hit), depth=wp.clone(self.depth))


@wp.func
def _sd_box(p: wp.vec3f, half: wp.vec3f) -> float:
    # signed distance point -> origin-centered AABB
    q = wp.vec3f(wp.abs(p[0]) - half[0], wp.abs(p[1]) - half[1],
                 wp.abs(p[2]) - half[2])
    outside = wp.length(wp.vec3f(wp.max(q[0], 0.0), wp.max(q[1], 0.0),
                                 wp.max(q[2], 0.0)))
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    return outside + inside


@wp.kernel
def _frame_query_k(
    gate_pos: wp.array(dtype=wp.vec3f),
    gate_quat: wp.array(dtype=wp.quatf),
    gate_hs: wp.array(dtype=wp.float32),
    gate_count: wp.array(dtype=wp.int32),
    gate_valid: wp.array(dtype=wp.int32),
    max_gates: int,
    next_cp: wp.array(dtype=wp.int32),
    window: int,
    position: wp.array(dtype=wp.vec3f),
    radius: float,
    thick: float,
    depth_x: float,
    out_hit: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    out_hit[e] = 0
    out_depth[e] = 0.0
    n = gate_count[e]
    if n > max_gates:
        n = max_gates
    if gate_valid[e] == 0 or n < 1:
        return
    base = e * max_gates
    p = position[e]
    if _is_nan3(p) == 1:
        return
    g0 = next_cp[e] - 1
    worst = float(0.0)
    hit = int(0)
    for k in range(window):
        gi = ((g0 + k) % n + n) % n
        c = gate_pos[base + gi]
        q = gate_quat[base + gi]
        hs = gate_hs[base + gi]
        if c[0] != c[0] or hs != hs:
            continue
        lp = wp.quat_rotate_inv(q, p - c)
        hd = 0.5 * depth_x
        ht = 0.5 * thick
        span = hs + thick          # posts cover the corners
        # posts: centers (0, +/-(hs+ht), 0), half (hd, ht, span)
        # bars:  centers (0, 0, +/-(hs+ht)), half (hd, span, ht)
        for m in range(4):
            off = wp.vec3f(0.0, 0.0, 0.0)
            half = wp.vec3f(hd, ht, span)
            if m == 0:
                off = wp.vec3f(0.0, hs + ht, 0.0)
            elif m == 1:
                off = wp.vec3f(0.0, -(hs + ht), 0.0)
            elif m == 2:
                off = wp.vec3f(0.0, 0.0, hs + ht)
                half = wp.vec3f(hd, span, ht)
            else:
                off = wp.vec3f(0.0, 0.0, -(hs + ht))
                half = wp.vec3f(hd, span, ht)
            d = _sd_box(lp - off, half) - radius
            if d < 0.0:
                hit = int(1)
                if -d > worst:
                    worst = -d
    out_hit[e] = hit
    out_depth[e] = worst


class FrameChecker:
    """See module docstring. Mirrors DiscChecker's bind/query lifecycle."""

    def __init__(self, seq: GateSequence, num_envs: int, radius: float,
                 frame_thickness: float, frame_depth: float,
                 window: int = 4) -> None:
        _init()
        for name, v in (("radius", radius), ("frame_thickness", frame_thickness),
                        ("frame_depth", frame_depth)):
            if not (float(v) > 0.0):
                raise ValueError(f"{name} must be > 0, got {v!r}")
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")
        E = int(num_envs)
        stride = int(seq.position.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"gate batch layout invalid: {stride} slots for {E} envs")
        self._seq = seq
        self._E = E
        self._G = stride // E
        self._radius = float(radius)
        self._thick = float(frame_thickness)
        self._depth = float(frame_depth)
        self._window = int(window)
        self._device = str(seq.position.device)
        self._pos: "wp.array | None" = None
        self._next: "wp.array | None" = None
        self._contact = FrameContact(
            hit=wp.zeros(E, dtype=wp.int32, device=self._device),
            depth=wp.zeros(E, dtype=wp.float32, device=self._device))

    def bind_inputs(self, position: wp.array) -> None:
        _check_arr("position", position, (self._E,), wp.vec3f, self._device)
        self._pos = position

    def bind_window(self, next_cp: wp.array) -> None:
        """Bind the [E] int32 current-target buffer (ProgressTracker state)."""
        _check_arr("next_cp", next_cp, (self._E,), wp.int32, self._device)
        self._next = next_cp

    def query(self) -> FrameContact:
        if self._pos is None or self._next is None:
            raise RuntimeError("call bind_inputs() and bind_window() first")
        s = self._seq
        wp.launch(_frame_query_k, dim=self._E,
                  inputs=[s.position, s.orientation, s.half_size, s.count,
                          s.valid, self._G, self._next, self._window,
                          self._pos, self._radius, self._thick, self._depth,
                          self._contact.hit, self._contact.depth],
                  device=self._device)
        _sync(self._device)
        return self._contact
