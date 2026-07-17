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

from .collision_geom import (
    _box_corner,
    _closest_on_seg,
    _crossing,
    _is_nan3,
    _pick4,
    _quat_yaw,
    _rot2,
    _safe_normalize2,
)
from .runtime import _BIG


@wp.kernel
def _track_aabb_k(
    outer: wp.array(dtype=wp.vec3f),
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
        p3 = outer[base + i]
        mnx = wp.min(mnx, p3[0])
        mny = wp.min(mny, p3[1])
        mxx = wp.max(mxx, p3[0])
        mxy = wp.max(mxy, p3[1])
    pad = padding
    if pad <= 0.0:
        pad = pad_frac * wp.max(mxx - mnx, mxy - mny)
    lo[e] = wp.vec2f(mnx - pad, mny - pad)
    hi[e] = wp.vec2f(mxx + pad, mxy + pad)


@wp.kernel
def _sdf_bake_k(
    inner: wp.array(dtype=wp.vec3f),
    outer: wp.array(dtype=wp.vec3f),
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
        a3 = inner[base + i]
        b3 = inner[base + i2]
        a = wp.vec2f(a3[0], a3[1])
        b = wp.vec2f(b3[0], b3[1])
        cp = _closest_on_seg(p, a, b)
        d_in = wp.min(d_in, wp.length(p - cp))
        cn_in = cn_in + _crossing(p, a, b)
        a3 = outer[base + i]
        b3 = outer[base + i2]
        a = wp.vec2f(a3[0], a3[1])
        b = wp.vec2f(b3[0], b3[1])
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


@wp.func
def _grid_coord(pv: float, lov: float, hiv: float, res: int) -> float:
    f = (pv - lov) / (hiv - lov) * float(res) - 0.5
    return wp.clamp(f, 0.0, float(res) - 1.0)


@wp.func
def _sample_phi(phi: wp.array(dtype=wp.float32), base: int, res: int,
                lo: wp.vec2f, hi: wp.vec2f, p: wp.vec2f) -> float:
    fx = _grid_coord(p[0], lo[0], hi[0], res)
    fy = _grid_coord(p[1], lo[1], hi[1], res)
    x0 = int(fx)
    y0 = int(fy)
    x1 = wp.min(x0 + 1, res - 1)
    y1 = wp.min(y0 + 1, res - 1)
    tx = fx - float(x0)
    ty = fy - float(y0)
    v00 = phi[base + y0 * res + x0]
    v10 = phi[base + y0 * res + x1]
    v01 = phi[base + y1 * res + x0]
    v11 = phi[base + y1 * res + x1]
    return wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty)


@wp.kernel
def _box_query_sdf_k(
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
    phi: wp.array(dtype=wp.float32),
    bid: wp.array(dtype=wp.int8),
    res: int,
    max_boxes: int,
    position: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_extents: wp.array(dtype=wp.vec2f),
    out_oob: wp.array(dtype=wp.int32),
    out_distance: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
    out_normal: wp.array(dtype=wp.vec2f),
    out_boundary: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos3 = position[t]
    if _is_nan3(pos3) == 1:
        out_oob[t] = 0
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return
    # Planar OOB semantics: project the box pose to xy.
    pos = wp.vec2f(pos3[0], pos3[1])

    l = lo[e]
    h = hi[e]
    base = e * res * res
    yw = _quat_yaw(orientation[t])
    he = half_extents[t]
    ux = _rot2(yw, wp.vec2f(1.0, 0.0))
    uy = _rot2(yw, wp.vec2f(0.0, 1.0))
    c0 = _box_corner(pos, ux, uy, he, 0)
    c1 = _box_corner(pos, ux, uy, he, 1)
    c2 = _box_corner(pos, ux, uy, he, 2)
    c3 = _box_corner(pos, ux, uy, he, 3)

    phimin = _sample_phi(phi, base, res, l, h, pos)
    pmin = pos
    for k in range(4):
        ck = _pick4(c0, c1, c2, c3, k)
        v = _sample_phi(phi, base, res, l, h, ck)
        if v < phimin:
            phimin = v
            pmin = ck

    # Degenerate env (count < 3): the whole grid is NaN-baked, so the center
    # sample is NaN. Emit the same conservative result as the segments
    # backend (oob=1, NaN geometry) instead of relying on clamp(NaN) index
    # behavior further down.
    if phimin != phimin:
        out_oob[t] = 1
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    oob = int(0)
    if phimin < 0.0:
        oob = int(1)

    # Central-difference gradient of phi at the argmin sample; phi increases
    # into the band, so normalize(grad) already points inward.
    hx = (h[0] - l[0]) / float(res)
    hy = (h[1] - l[1]) / float(res)
    gxv = (_sample_phi(phi, base, res, l, h, pmin + wp.vec2f(hx, 0.0))
           - _sample_phi(phi, base, res, l, h, pmin - wp.vec2f(hx, 0.0))) / (2.0 * hx)
    gyv = (_sample_phi(phi, base, res, l, h, pmin + wp.vec2f(0.0, hy))
           - _sample_phi(phi, base, res, l, h, pmin - wp.vec2f(0.0, hy))) / (2.0 * hy)
    n = _safe_normalize2(wp.vec2f(gxv, gyv))
    nearest = pmin - phimin * n

    fx = _grid_coord(nearest[0], l[0], h[0], res)
    fy = _grid_coord(nearest[1], l[1], h[1], res)
    xi = wp.min(int(fx + 0.5), res - 1)
    yi = wp.min(int(fy + 0.5), res - 1)

    out_oob[t] = oob
    out_distance[t] = phimin
    out_nearest[t] = nearest
    out_normal[t] = n
    out_boundary[t] = int(bid[base + yi * res + xi])
