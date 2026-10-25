"""Analytic circular barriers for Repulsive proposals and the finished road band.

Finished cells use their four-vertex convex hull as a conservative certificate
for ruled or triangulated strips, including concave and folded cells. This can
reject a disc outside the precise strip image but never cancels an occupied lobe.
The metadata is a private per-environment channel; its array fields refer to fixed
sampler storage, with the uniform output transform applied by the consumer.
"""
from __future__ import annotations

import warp as wp


@wp.struct
class RepulsiveBarriers:
    disc_centers: wp.array(dtype=wp.vec2f)
    disc_radii: wp.array(dtype=wp.float32)
    offset: int
    count: int
    domain_center: wp.vec2f
    domain_radius: float
    size: float


@wp.func
def _tolerance(radius: float) -> wp.float64:
    # A conservative length tolerance, never permission to penetrate a barrier.
    return wp.float64(1.0e-5) * wp.max(wp.float64(1.0), wp.abs(wp.float64(radius)))


@wp.func
def _segment_distance_sq(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f) -> wp.float64:
    # Cast before subtracting and keep the projected offset relative to c. Adding
    # it to a large float32 world position can erase a small disc penetration.
    relative = wp.vec2d(a) - wp.vec2d(c)
    ab = wp.vec2d(b) - wp.vec2d(a)
    denom = wp.dot(ab, ab)
    t = wp.float64(0.0)
    if denom > wp.float64(0.0):
        t = wp.clamp(-wp.dot(relative, ab) / denom, wp.float64(0.0), wp.float64(1.0))
    d = relative + t * ab
    return wp.dot(d, d)


@wp.func
def segment_disc_clear(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, radius: float) -> bool:
    limit = wp.float64(radius) + _tolerance(radius)
    d2 = _segment_distance_sq(a, b, c)
    return wp.isfinite(d2) and d2 > limit * limit


@wp.func
def _cross(a: wp.vec2d, b: wp.vec2d) -> wp.float64:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _triangle_contains(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f,
                       p: wp.vec2f) -> bool:
    ad = wp.vec2d(a)
    bd = wp.vec2d(b)
    cd = wp.vec2d(c)
    area = _cross(bd - ad, cd - ad)
    if area == wp.float64(0.0):
        return False
    # Orient the edges relative to the query center, without float32 products
    # or reconstruction of a world-space projected point.
    ap = ad - wp.vec2d(p)
    bp = bd - wp.vec2d(p)
    cp = cd - wp.vec2d(p)
    x = _cross(ap, bp)
    y = _cross(bp, cp)
    z = _cross(cp, ap)
    zero = wp.float64(0.0)
    return (x >= zero and y >= zero and z >= zero) or (x <= zero and y <= zero and z <= zero)


@wp.func
def swept_segment_disc_clear(a: wp.vec2f, b: wp.vec2f, newa: wp.vec2f,
                             newb: wp.vec2f, c: wp.vec2f, radius: float) -> bool:
    # Every linear segment sweep lies in this convex hull. All six chords plus the
    # four triangles cover it, including degeneracies and endpoint permutations.
    clear = segment_disc_clear(a, b, c, radius)
    clear = clear and segment_disc_clear(a, newa, c, radius)
    clear = clear and segment_disc_clear(a, newb, c, radius)
    clear = clear and segment_disc_clear(b, newa, c, radius)
    clear = clear and segment_disc_clear(b, newb, c, radius)
    clear = clear and segment_disc_clear(newa, newb, c, radius)
    inside = _triangle_contains(a, b, newa, c) or _triangle_contains(a, b, newb, c)
    inside = inside or _triangle_contains(a, newa, newb, c) or _triangle_contains(b, newa, newb, c)
    return clear and not inside


@wp.func
def road_cell_disc_clear(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f,
                         d: wp.vec2f, center: wp.vec2f, radius: float) -> bool:
    # A ruled strip can fold over itself even when its four-edge polygon is
    # concave and its even/odd interior misses the occupied lobe. Every bilinear
    # or triangulated cell lies in its four-vertex convex hull. Certify that hull
    # conservatively; ordinary convex cells are unchanged.
    return swept_segment_disc_clear(a, d, b, c, center, radius)


@wp.kernel
def publish_barriers_k(centers: wp.array(dtype=wp.vec2f),
                       radii: wp.array(dtype=wp.float32), count: int,
                       domain_radius: float, size: wp.array(dtype=wp.float32),
                       out: wp.array(dtype=RepulsiveBarriers)):
    e = wp.tid()
    m = RepulsiveBarriers()
    m.disc_centers = centers
    m.disc_radii = radii
    m.offset = e * count
    m.count = count
    m.domain_center = wp.vec2f(0.0, 0.0)
    m.domain_radius = domain_radius
    m.size = size[e]
    out[e] = m


@wp.kernel
def _finished_band_validity_k(outer: wp.array(dtype=wp.vec2f),
                               inner: wp.array(dtype=wp.vec2f),
                               count: wp.array(dtype=wp.int32),
                               barriers: wp.array(dtype=RepulsiveBarriers), stride: int,
                               valid: wp.array(dtype=wp.int32),
                               fail: wp.array(dtype=wp.int32)):
    e = wp.tid()
    m = barriers[e]
    n = count[e]
    scale = m.size
    disc_bad = int(0)
    wall_bad = int(0)
    radius = m.domain_radius * scale
    center = m.domain_center * scale
    limit = wp.float64(radius) - _tolerance(radius)
    if not wp.isfinite(scale) or scale <= 0.0 or not wp.isfinite(radius) or limit <= 0.0:
        wall_bad = 1
    if n < 3 or n > stride:
        wall_bad = 1
    else:
        for i in range(n):
            j = (i + 1) % n
            a = outer[e * stride + i]
            b = outer[e * stride + j]
            c = inner[e * stride + j]
            d = inner[e * stride + i]
            # The circular domain is convex: containing every cell vertex
            # contains the entire convex-hull certificate, including a fold.
            da = wp.vec2d(a) - wp.vec2d(center)
            dd = wp.vec2d(d) - wp.vec2d(center)
            da2 = wp.dot(da, da)
            dd2 = wp.dot(dd, dd)
            if not wp.isfinite(da2) or not wp.isfinite(dd2) or da2 > limit * limit or dd2 > limit * limit:
                wall_bad = 1
            for k in range(m.count):
                r = m.disc_radii[m.offset + k] * scale
                if r > 0.0:
                    p = m.disc_centers[m.offset + k] * scale
                    if not road_cell_disc_clear(a, b, c, d, p, r):
                        disc_bad = 1
                elif not wp.isfinite(r) or r < 0.0:
                    disc_bad = 1
    own = (disc_bad << 8) | (wall_bad << 9)
    fail[e] = fail[e] | own
    if own != 0:
        valid[e] = 0


def finished_band_validity_inplace(ctx, outer, inner, count, barriers, stride,
                                   valid, fail) -> None:
    """OR exact finite-width barrier failures into the existing validity outputs."""
    ctx.launch(_finished_band_validity_k, int(count.shape[0]),
               [outer, inner, count, barriers, int(stride), valid, fail])
