"""SDF backend kernels: per-env signed-distance bake + boundary-id grids.

Bake is a brute-force O(E * R^2 * N) scan (GPU-oriented; CPU bakes are only
for small tests). phi stores signed distance to the band boundary, positive
inside the drivable band; bid stores which boundary (0 inner / 1 outer) is
nearest at each texel. Grids cover the per-env track AABB expanded by the
configured padding; queries outside the grid clamp to edge texels, which stay
negative there, so far-out boxes still read as OOB.
"""
from __future__ import annotations

import warp as wp

from .collision_geom import _closest_on_seg, _crossing

_BIG = 1.0e30


@wp.kernel
def _track_aabb_k(
    outer: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    padding: float,   # explicit per-side padding; <= 0 selects auto mode
    pad_frac: float,  # auto padding as a fraction of the larger AABB extent
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
):
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        lo[e] = wp.vec2f(-1.0, -1.0)
        hi[e] = wp.vec2f(1.0, 1.0)
        return
    base = e * n_max
    mnx = float(_BIG)
    mny = float(_BIG)
    mxx = float(-_BIG)
    mxy = float(-_BIG)
    for i in range(m):
        p = outer[base + i]
        mnx = wp.min(mnx, p[0])
        mny = wp.min(mny, p[1])
        mxx = wp.max(mxx, p[0])
        mxy = wp.max(mxy, p[1])
    pad = padding
    if pad <= 0.0:
        pad = pad_frac * wp.max(mxx - mnx, mxy - mny)
    lo[e] = wp.vec2f(mnx - pad, mny - pad)
    hi[e] = wp.vec2f(mxx + pad, mxy + pad)


@wp.kernel
def _sdf_bake_k(
    inner: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    res: int,
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
    phi: wp.array(dtype=wp.float32),
    bid: wp.array(dtype=wp.int8),
):
    t = wp.tid()
    cells = res * res
    e = t // cells
    rem = t - e * cells
    gy = rem // res
    gx = rem - gy * res

    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        phi[t] = wp.nan
        bid[t] = wp.int8(-1)
        return

    l = lo[e]
    h = hi[e]
    p = wp.vec2f(
        l[0] + (float(gx) + 0.5) / float(res) * (h[0] - l[0]),
        l[1] + (float(gy) + 0.5) / float(res) * (h[1] - l[1]),
    )

    base = e * n_max
    d_in = _BIG
    d_out = _BIG
    cn_in = int(0)
    cn_out = int(0)
    for i in range(m):
        i2 = i + 1
        if i2 == m:
            i2 = 0
        a = inner[base + i]
        b = inner[base + i2]
        cp = _closest_on_seg(p, a, b)
        d_in = wp.min(d_in, wp.length(p - cp))
        cn_in = cn_in + _crossing(p, a, b)
        a = outer[base + i]
        b = outer[base + i2]
        cp = _closest_on_seg(p, a, b)
        d_out = wp.min(d_out, wp.length(p - cp))
        cn_out = cn_out + _crossing(p, a, b)

    d = wp.min(d_in, d_out)
    inside = int(0)
    if cn_out % 2 == 1 and cn_in % 2 == 0:
        inside = int(1)
    if inside == 1:
        phi[t] = d
    else:
        phi[t] = -d
    if d_in <= d_out:
        bid[t] = wp.int8(0)
    else:
        bid[t] = wp.int8(1)
