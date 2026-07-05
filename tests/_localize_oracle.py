"""Independent numpy mirrors of TrackLocalizer / speed_profile (tests only)."""
from __future__ import annotations

import numpy as np


def project(center, arclen, length, p):
    """Brute-force projection of p onto a closed polyline.

    Args:
        center: [m, 2] real centerline points (no NaN padding).
        arclen: [m] cumulative arc length at each point (arclen[0] == 0).
        length: total loop perimeter (closes segment m-1 -> 0).
        p: [2] query point.

    Returns:
        (s, n, segment) with the kernel's conventions: s in [0, length)
        (wrapped at the seam), n positive on the (t.y, -t.x) side of the
        segment tangent — the RIGHT of the direction of travel; which
        boundary that is depends on the loop's winding — and segment the
        argmin index (lowest index on exact ties, matching the kernel's
        strict <).
    """
    center = np.asarray(center, np.float64)
    arclen = np.asarray(arclen, np.float64)
    p = np.asarray(p, np.float64)
    m = len(center)
    best = (np.inf, 0, 0.0)
    for i in range(m):
        a = center[i]
        b = center[(i + 1) % m]
        ab = b - a
        denom = max(float(ab @ ab), 1e-12)
        u = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
        q = a + u * ab
        d2 = float((p - q) @ (p - q))
        if d2 < best[0]:
            best = (d2, i, u)
    _, i, u = best
    a = center[i]
    ab = center[(i + 1) % m] - a
    q = a + u * ab
    seg_start = float(arclen[i])
    seg_end = float(arclen[i + 1]) if i + 1 < m else float(length)
    s = seg_start + u * (seg_end - seg_start)
    if s >= float(length):
        s -= float(length)  # u == 1 on the closing segment: wrap the seam
    t = ab / max(float(np.linalg.norm(ab)), 1e-8)
    n = float((p - q) @ np.array([t[1], -t[0]]))
    return s, n, i


def speed_profile(kappa, seg_len, a_lat_max, a_accel, a_brake, v_cap):
    """One env's speed profile from per-point curvature + segment lengths.

    Mirrors the kernel: steady-state lateral limit capped at v_cap, then two
    wrap laps of a forward acceleration pass and a backward braking pass.
    seg_len[i] is the length of segment i -> (i+1) % m.
    """
    kappa = np.abs(np.asarray(kappa, np.float64))
    seg_len = np.asarray(seg_len, np.float64)
    m = len(kappa)
    v = np.full(m, float(v_cap))
    mask = kappa > 1e-9
    v[mask] = np.minimum(np.sqrt(a_lat_max / kappa[mask]), v_cap)
    for _ in range(2):
        for i in range(m):
            j = (i + 1) % m
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_accel * seg_len[i]))
        for i in range(m - 1, -1, -1):
            j = (i + 1) % m
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * a_brake * seg_len[i]))
    return v
