"""Pure-numpy geometric/racing metrics for comparing generators (dev tool — not runtime).

Each function takes one env's REAL points as ``pts`` ([n, 2], a closed loop; the segment
n-1 -> 0 closes it). Batched aggregation lives in compare_generators.py.
"""
from __future__ import annotations

import numpy as np


def _seg_vectors(pts: np.ndarray) -> np.ndarray:
    return np.roll(pts, -1, axis=0) - pts  # pts[i+1] - pts[i], wrapping


def perimeter(pts: np.ndarray) -> float:
    return float(np.linalg.norm(_seg_vectors(pts), axis=1).sum())


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) * 0.5)


def compactness(pts: np.ndarray) -> float:
    p = perimeter(pts)
    if p <= 0:
        return 0.0
    return float(4.0 * np.pi * polygon_area(pts) / (p * p))


def turn_angles(pts: np.ndarray) -> np.ndarray:
    v = _seg_vectors(pts)                       # outgoing edge at each vertex
    v_prev = np.roll(v, 1, axis=0)              # incoming edge
    a_prev = np.arctan2(v_prev[:, 1], v_prev[:, 0])
    a_cur = np.arctan2(v[:, 1], v[:, 0])
    d = a_cur - a_prev
    return (d + np.pi) % (2 * np.pi) - np.pi    # wrap to [-pi, pi)


def curvature(pts: np.ndarray) -> np.ndarray:
    seg_len = np.linalg.norm(_seg_vectors(pts), axis=1)
    mean_adj = 0.5 * (seg_len + np.roll(seg_len, 1))
    mean_adj = np.where(mean_adj > 1e-9, mean_adj, 1e-9)
    return np.abs(turn_angles(pts)) / mean_adj


def self_intersects(pts: np.ndarray) -> bool:
    n = len(pts)
    a = pts
    b = np.roll(pts, -1, axis=0)

    def _ccw(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    for i in range(n):
        for j in range(i + 1, n):
            if (i + 1) % n == j or (j + 1) % n == i:
                continue  # skip shared-endpoint / adjacent edges
            d1 = _ccw(a[i], b[i], a[j])
            d2 = _ccw(a[i], b[i], b[j])
            d3 = _ccw(a[j], b[j], a[i])
            d4 = _ccw(a[j], b[j], b[i])
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return True
    return False


def racing_line_proxy(pts: np.ndarray, a_lat_max: float = 1.0) -> dict:
    k = curvature(pts)
    seg_len = np.linalg.norm(_seg_vectors(pts), axis=1)
    k_safe = np.where(k > 1e-6, k, 1e-6)
    v = np.sqrt(a_lat_max / k_safe)             # friction-circle cornering speed
    lap_time = float(np.sum(seg_len / np.where(v > 1e-9, v, 1e-9)))
    return {
        "peak_curvature": float(k.max()),
        "integral_kappa2": float(np.sum(k * k * seg_len)),
        "lap_time": lap_time,
    }
