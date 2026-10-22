"""``@wp.func`` building blocks for plugin kernels.

Each is a ``wp.Function``: call it from inside your own ``@wp.kernel`` (or another
``@wp.func``), e.g. ``from track_gen.ops import funcs`` then ``funcs.catmull_rom(...)``
in the kernel body. They are the device functions the built-in generators use: the
same operations in the same order. On CPU a plugin kernel that composes them reproduces
the built-in arithmetic bit for bit. On CUDA (``wp.config.fuse_fp``, on by default)
the compiler may contract a multiply and an add into one FMA, and which pairs it
contracts depends on the code around the inlined call, so a different surrounding
kernel can round differently in the last bit.

The three crossing predicates differ on purpose: :func:`segments_cross` is the strict
sign test used by gate validity; the polygon crossing counter
(:func:`track_gen.ops.self_intersections`) adds a scale-relative collinearity
tolerance on top of :func:`ccw`.
"""
from __future__ import annotations

import warp as wp

from .hash import hash32

__all__ = ["catmull_rom", "ccw", "cross2", "hash32", "nan0", "safe_normalize2",
           "segments_cross"]


@wp.func
def ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
    """Orientation of ``(o, p, q)``: ``cross(p - o, q - o)``, i.e. twice the signed
    triangle area; ``> 0`` when ``o -> p -> q`` turns counter-clockwise."""
    # Returns (q.y-o.y)*(p.x-o.x) - (p.y-o.y)*(q.x-o.x)
    return (qy - oy) * (px - ox) - (py - oy) * (qx - ox)


@wp.func
def nan0(x: float) -> float:
    """``x`` if finite, else ``0.0`` (``nan_to_num``; a no-op on finite inputs)."""
    # NaN/inf -> 0.0 guard (nan_to_num equivalent) for the border self-intersection
    # check. wp.where(cond, if_true, if_false) is the Warp 1.14 non-deprecated select
    # primitive. A NO-OP for finite inputs.
    return wp.where(wp.isfinite(x), x, 0.0)


@wp.func
def cross2(a: wp.vec2f, b: wp.vec2f) -> float:
    """2-D cross product ``a.x*b.y - a.y*b.x``."""
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    """``v / max(|v|, 1e-8)``: unit vector with a length floor.

    An all-NaN vector stays all-NaN; a NaN length is floored to ``1e-8``, so a vector
    with one NaN component keeps the other, scaled by ``1e8`` (e.g. ``(nan, 3)`` ->
    ``(nan, 3e8)``)."""
    # Mirrors geometry.safe_normalize: v / clamp_min(||v||, 1e-8). wp.max floors the
    # length at 1e-8 and also maps a NaN length to 1e-8, so only the NaN components of
    # v come out NaN (an all-NaN vector, e.g. a pruned corner, stays all-NaN).
    return v / wp.max(wp.length(v), 1.0e-8)


@wp.func
def catmull_rom(p0: wp.vec2f, p1: wp.vec2f, p2: wp.vec2f, p3: wp.vec2f, u: float) -> wp.vec2f:
    """Uniform Catmull-Rom (tension 0.5) segment ``p1 -> p2`` at ``u`` in ``[0, 1]``
    (``u = 0`` gives ``p1``, ``u = 1`` gives ``p2``)."""
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


@wp.func
def segments_cross(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f) -> int:
    """1 if segments ``ab`` and ``cd`` properly cross (strict sign test: touching
    or collinear overlap is not a crossing), else 0."""
    ab = b - a
    cd = d - c
    ac = c - a
    ad = d - a
    ca = a - c
    cb = b - c
    o1 = cross2(ab, ac)
    o2 = cross2(ab, ad)
    o3 = cross2(cd, ca)
    o4 = cross2(cd, cb)
    hit_ab = (o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)
    hit_cd = (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    if hit_ab and hit_cd:
        return int(1)
    return int(0)
