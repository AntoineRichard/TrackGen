"""Shared pure-Warp geometry helpers for collision queries.

Leaf module (imports only warp): used by the segments backend in
``collision.py`` and the SDF backend in ``collision_sdf.py``. All helpers are
``@wp.func`` device functions; nothing here launches kernels.
"""
from __future__ import annotations

import warp as wp


@wp.func
def _is_nan2(v: wp.vec2f) -> int:
    # NaN != NaN; avoids a torch/older-warp dependency for the NaN probe.
    if v[0] != v[0] or v[1] != v[1]:
        return int(1)
    return int(0)


@wp.func
def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    return v / wp.max(wp.length(v), 1.0e-8)


@wp.func
def _rot2(yaw: float, v: wp.vec2f) -> wp.vec2f:
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    return wp.vec2f(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def _box_corner(center: wp.vec2f, ux: wp.vec2f, uy: wp.vec2f,
                he: wp.vec2f, k: int) -> wp.vec2f:
    # CCW corner order in the box frame: (+,+), (-,+), (-,-), (+,-).
    sx = 1.0
    sy = 1.0
    if k == 1 or k == 2:
        sx = -1.0
    if k == 2 or k == 3:
        sy = -1.0
    return center + ux * (sx * he[0]) + uy * (sy * he[1])


@wp.func
def _pick4(c0: wp.vec2f, c1: wp.vec2f, c2: wp.vec2f, c3: wp.vec2f,
           k: int) -> wp.vec2f:
    if k == 0:
        return c0
    if k == 1:
        return c1
    if k == 2:
        return c2
    return c3


@wp.func
def _closest_on_seg(p: wp.vec2f, a: wp.vec2f, b: wp.vec2f) -> wp.vec2f:
    ab = b - a
    denom = wp.dot(ab, ab)
    t = 0.0
    if denom > 1.0e-12:
        t = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
    return a + ab * t


@wp.func
def _crossing(p: wp.vec2f, a: wp.vec2f, b: wp.vec2f) -> int:
    """1 if the +x ray from p crosses segment ab (half-open rule), else 0."""
    if (a[1] > p[1]) != (b[1] > p[1]):
        x_hit = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        if p[0] < x_hit:
            return int(1)
    return int(0)


@wp.func
def _point_to_local_box_dist(q: wp.vec2f, he: wp.vec2f) -> float:
    """Distance from a box-local point to the solid AABB [-he, he]; 0 inside."""
    dx = wp.max(wp.abs(q[0]) - he[0], 0.0)
    dy = wp.max(wp.abs(q[1]) - he[1], 0.0)
    return wp.sqrt(dx * dx + dy * dy)


@wp.func
def _seg_hits_aabb(a: wp.vec2f, b: wp.vec2f, he: wp.vec2f) -> int:
    """1 if segment ab (box-local coords) intersects the solid AABB [-he, he].

    Liang-Barsky slab clip of the parametric segment; covers endpoint-inside,
    pass-through, and corner-clip cases.
    """
    d = b - a
    tmin = 0.0
    tmax = 1.0
    for axis in range(2):
        av = a[axis]
        dv = d[axis]
        hv = he[axis]
        if wp.abs(dv) < 1.0e-12:
            if av < -hv or av > hv:
                return int(0)
        else:
            t1 = (-hv - av) / dv
            t2 = (hv - av) / dv
            tmin = wp.max(tmin, wp.min(t1, t2))
            tmax = wp.min(tmax, wp.max(t1, t2))
            if tmin > tmax:
                return int(0)
    return int(1)


@wp.func
def _cross2(a: wp.vec2f, b: wp.vec2f) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _segs_cross(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f) -> int:
    """1 iff segments ab and cd properly intersect (strict crossing).

    Collinear overlap, endpoint touching, and degenerate (zero-length)
    segments all return 0 — a width-0 gate can never be crossed.
    """
    ab = b - a
    cd = d - c
    o1 = _cross2(ab, c - a)
    o2 = _cross2(ab, d - a)
    o3 = _cross2(cd, a - c)
    o4 = _cross2(cd, b - c)
    hit_ab = (o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)
    hit_cd = (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    if hit_ab and hit_cd:
        return int(1)
    return int(0)
