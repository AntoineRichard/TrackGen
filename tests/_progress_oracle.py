"""Independent numpy mirror of ProgressTracker semantics (tests only)."""
from __future__ import annotations

import numpy as np


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def segs_cross(a, b, c, d):
    """Strict proper intersection of segments ab and cd."""
    ab, cd = b - a, d - c
    o1, o2 = _cross(ab, c - a), _cross(ab, d - a)
    o3, o4 = _cross(cd, a - c), _cross(cd, b - c)
    return (((o1 > 0) and (o2 < 0)) or ((o1 < 0) and (o2 > 0))) and \
           (((o3 > 0) and (o4 < 0)) or ((o3 < 0) and (o4 > 0)))


class ProgressOracle:
    """One env's worth of checkpoints; mirrors the kernel's update order."""

    def __init__(self, positions, lefts, rights, tangents):
        self.p = np.asarray(positions, float)
        self.l = np.asarray(lefts, float)
        self.r = np.asarray(rights, float)
        self.t = np.asarray(tangents, float)
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
            if segs_cross(self.prev, pos, self.l[g], self.r[g]):
                if np.dot(pos - self.prev, self.t[g]) > 0:
                    ev["passed"] = 1
                    ev["checkpoint_passed"] = g
                else:
                    ev["wrong_way"] = 1
            for i in range(self.n):
                if i != g and segs_cross(self.prev, pos, self.l[i], self.r[i]):
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
