"""Independent numpy reference implementation of the collision semantics.

Small, readable loops (slow but only used on tiny test batches). Mirrors the
spec: band = inside outer AND outside inner; OOB iff any corner outside band
or any boundary segment intersects the box; distance = +min box-boundary
distance inside, -(deepest corner penetration) when OOB.
"""
from __future__ import annotations

import numpy as np


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def box_corners(pos, yaw, he):
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    return np.asarray(pos)[None, :] + (signs * np.asarray(he)[None, :]) @ rot(yaw).T


def point_in_poly(p, poly):
    x, y = p
    xs, ys = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xhit = xs + (y - ys) * (x2 - xs) / (y2 - ys)
    return int(np.count_nonzero(cond & (x < xhit))) % 2 == 1


def point_seg_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    cp = a + t * ab
    return float(np.linalg.norm(p - cp)), cp


def point_polyline_dist(p, poly):
    best_d, best_cp = np.inf, None
    m = len(poly)
    for i in range(m):
        d, cp = point_seg_dist(p, poly[i], poly[(i + 1) % m])
        if d < best_d:
            best_d, best_cp = d, cp
    return best_d, best_cp


def point_box_dist(p, pos, yaw, he):
    q = rot(yaw).T @ (np.asarray(p) - np.asarray(pos))
    dx = max(abs(q[0]) - he[0], 0.0)
    dy = max(abs(q[1]) - he[1], 0.0)
    return float(np.hypot(dx, dy))


def seg_hits_box(a, b, pos, yaw, he):
    R = rot(yaw)
    al = R.T @ (a - np.asarray(pos))
    bl = R.T @ (b - np.asarray(pos))
    d = bl - al
    tmin, tmax = 0.0, 1.0
    for ax in range(2):
        if abs(d[ax]) < 1e-12:
            if al[ax] < -he[ax] or al[ax] > he[ax]:
                return False
        else:
            t1 = (-he[ax] - al[ax]) / d[ax]
            t2 = (he[ax] - al[ax]) / d[ax]
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
            if tmin > tmax:
                return False
    return True


def box_contact(inner, outer, pos, yaw, he):
    """Reference contact result for one box vs one env's polylines."""
    pos = np.asarray(pos, float)
    he = np.asarray(he, float)
    corners = box_corners(pos, yaw, he)
    crossed = False
    best = (np.inf, None, -1)  # (dist, boundary point, boundary id)
    for bnd, poly in ((0, inner), (1, outer)):
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            if seg_hits_box(a, b, pos, yaw, he):
                crossed = True
                cand = (0.0, point_seg_dist(pos, a, b)[1], bnd)
            else:
                cand = (point_box_dist(a, pos, yaw, he), a, bnd)
                for c in corners:
                    d, cp = point_seg_dist(c, a, b)
                    if d < cand[0]:
                        cand = (d, cp, bnd)
            if cand[0] < best[0]:
                best = cand
    inside = True
    worst_pen = 0.0
    for c in corners:
        pen = 0.0
        if point_in_poly(c, inner):        # in the hole
            inside = False
            pen = point_polyline_dist(c, inner)[0]
        if not point_in_poly(c, outer):    # outside the outer loop
            inside = False
            pen = max(pen, point_polyline_dist(c, outer)[0])
        worst_pen = max(worst_pen, pen)
    oob = (not inside) or crossed
    return {"oob": int(oob),
            "distance": -worst_pen if oob else best[0],
            "nearest": best[1],
            "boundary": best[2]}
