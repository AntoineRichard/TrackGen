"""Independent numpy mirror of ProgressTracker semantics (tests only)."""
from __future__ import annotations

import numpy as np


def plane_pass(prev, pos, fwd, l, r, v_half):
    """Mirror of collision_geom._plane_pass: swept segment vs bounded gate
    plane. +1 forward pass, -1 backward crossing inside the opening, else 0."""
    prev = np.asarray(prev, float)
    pos = np.asarray(pos, float)
    fwd = np.asarray(fwd, float)
    l = np.asarray(l, float)
    r = np.asarray(r, float)
    mid = 0.5 * (l + r)
    d0 = float(np.dot(prev - mid, fwd))
    d1 = float(np.dot(pos - mid, fwd))
    crossing = 0
    if d0 < 0.0 and d1 >= 0.0:
        crossing = 1
    if d0 > 0.0 and d1 <= 0.0:
        crossing = -1
    if crossing == 0:
        return 0
    t = d0 / (d0 - d1)
    pi = prev + (pos - prev) * t
    u_axis = r - l
    u_len = float(np.linalg.norm(u_axis))
    if u_len < 1.0e-12:
        return 0
    u_axis = u_axis / u_len
    u = float(np.dot(pi - mid, u_axis))
    v_axis = np.cross(u_axis, fwd)
    vl = float(np.linalg.norm(v_axis))
    v_axis = v_axis / vl if vl >= 1.0e-12 else v_axis * 0.0
    v = float(np.dot(pi - mid, v_axis))
    if abs(u) <= 0.5 * u_len and abs(v) <= v_half:
        return crossing
    return 0


class ProgressOracle:
    """One env's worth of checkpoints; mirrors the kernel's update order."""

    def __init__(self, positions, lefts, rights, tangents, up_halfs):
        self.p = np.asarray(positions, float)
        self.l = np.asarray(lefts, float)
        self.r = np.asarray(rights, float)
        self.t = np.asarray(tangents, float)
        self.v = np.asarray(up_halfs, float)
        self.n = len(self.p)
        self.reset()

    def reset(self):
        self.prev = None
        self.next = 0
        self.laps = 0
        self.progress = 0

    def update(self, pos):
        pos = np.asarray(pos, float)
        ev = {"passed": 0, "checkpoint_passed": -1, "wrong_way": 0,
              "wrong_checkpoint": -1}
        g = self.next
        if self.prev is not None and self.n >= 1:
            c = plane_pass(self.prev, pos, self.t[g], self.l[g], self.r[g],
                           self.v[g])
            if c == 1:
                ev["passed"] = 1
                ev["checkpoint_passed"] = g
            elif c == -1:
                ev["wrong_way"] = 1
            for i in range(self.n):
                if i != g and plane_pass(self.prev, pos, self.t[i], self.l[i],
                                         self.r[i], self.v[i]) != 0:
                    ev["wrong_checkpoint"] = i
                    break
        if ev["passed"]:
            self.next = (g + 1) % self.n
            self.progress += 1
            if self.next == 0:
                self.laps += 1
        self.prev = pos
        ev["next_checkpoint"] = self.next
        ev["laps"] = self.laps
        ev["progress"] = self.progress
        ev["dist_to_next"] = float(np.linalg.norm(self.p[self.next] - pos))
        return ev
