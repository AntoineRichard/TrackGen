"""SDF backend kernels: per-env signed-distance bake + boundary-id grids.

Bake is a brute-force O(E * R^2 * N) scan (GPU-oriented; CPU bakes are only
for small tests). phi stores signed distance to the band boundary, positive
inside the drivable band; bid stores which boundary (0 inner / 1 outer) is
nearest at each texel. Grids cover the per-env track AABB expanded by the
configured padding; queries outside the grid clamp to edge texels, which stay
negative there, so far-out boxes still read as OOB.

Box query: phimin is the exact minimum of the bilinear (edge-clamped) phi
over the box OUTLINE plus the box centre. Along one side the interpolant
restricted to one grid cell is a quadratic in the side parameter, so each
side is split at every grid line it crosses (at most 2R + 1 pieces) and each
piece is minimized in closed form. An edge that bulges into a side between
four on-road corners (the inner edge at a tight bend) is therefore seen;
sampling only the corners missed it. A boundary loop lying entirely inside
the box without touching its outline or centre is still not seen.
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
def _grid_coord_raw(pv: float, lov: float, hiv: float, res: int) -> float:
    # Continuous texel coordinate (texel centres at integers), unclamped.
    return (pv - lov) / (hiv - lov) * float(res) - 0.5


@wp.func
def _grid_coord(pv: float, lov: float, hiv: float, res: int) -> float:
    return wp.clamp(_grid_coord_raw(pv, lov, hiv, res), 0.0, float(res) - 1.0)


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


@wp.func
def _grid_crossings(fa: float, fb: float, res: int) -> wp.vec3i:
    """Grid lines k in [0, res-1] strictly between raw coords fa and fb, in
    travel order: (first k, count, step), count <= res.

    The coords are clamped to [-1, res] before the int conversion: the
    result is unchanged and a far-away box cannot overflow int.
    """
    a = wp.clamp(fa, -1.0, float(res))
    b = wp.clamp(fb, -1.0, float(res))
    if fb > fa:
        k0 = wp.max(int(wp.floor(a)) + 1, 0)
        k1 = wp.min(int(wp.ceil(b)) - 1, res - 1)
        return wp.vec3i(k0, wp.max(k1 - k0 + 1, 0), 1)
    if fb < fa:
        k0 = wp.min(int(wp.ceil(a)) - 1, res - 1)
        k1 = wp.max(int(wp.floor(b)) + 1, 0)
        return wp.vec3i(k0, wp.max(k0 - k1 + 1, 0), -1)
    return wp.vec3i(0, 0, 1)


@wp.func
def _axis_piece(f0: float, f1: float, fm: float, res: int) -> wp.vec3f:
    """One axis of a side piece whose raw coord goes f0 -> f1 (midpoint fm):
    (cell index, offset in the cell at u=0, slope in u).

    The slope is 0 where _sample_phi's edge clamp is active, so the piece
    reproduces _sample_phi's interpolant, clamp included. Requires res >= 2.
    """
    fc = wp.clamp(fm, 0.0, float(res) - 1.0)
    x0 = wp.min(int(fc), res - 2)
    if fm < 0.0 or fm > float(res) - 1.0:
        return wp.vec3f(float(x0), fc - float(x0), 0.0)
    return wp.vec3f(float(x0), f0 - float(x0), f1 - f0)


@wp.func
def _side_min(phi: wp.array(dtype=wp.float32), base: int, res: int,
              lo: wp.vec2f, hi: wp.vec2f, a: wp.vec2f, b: wp.vec2f,
              best: wp.vec3f) -> wp.vec3f:
    """Fold the minimum of the bilinear phi over the open side (a, b) into
    best = (phimin, x, y); the corners a and b are the caller's candidates.

    The side is split at every grid line it crosses. On one piece the field
    is phi(u) = q0 + q1 u + q2 u^2 for u in [0, 1], so its minimum is at the
    piece start or at the vertex u* = -q1 / (2 q2) when q2 > 0. A side whose
    raw grid coords or their span overflow float (a box ~1e36 away) is
    skipped.
    """
    fax = _grid_coord_raw(a[0], lo[0], hi[0], res)
    fay = _grid_coord_raw(a[1], lo[1], hi[1], res)
    dx = _grid_coord_raw(b[0], lo[0], hi[0], res) - fax
    dy = _grid_coord_raw(b[1], lo[1], hi[1], res) - fay
    if not wp.isfinite(wp.vec4f(fax, fay, dx, dy)):
        return best
    fbx = fax + dx
    fby = fay + dy
    phimin = best[0]
    pmin = wp.vec2f(best[1], best[2])
    rx = _grid_crossings(fax, fbx, res)
    ry = _grid_crossings(fay, fby, res)
    i = int(0)
    j = int(0)
    t0 = float(0.0)
    for _it in range(rx[1] + ry[1] + 1):
        # Piece [t0, t1]: up to the next x or y grid line, or the side end.
        tx = float(2.0)
        ty = float(2.0)
        if i < rx[1]:
            tx = (float(rx[0] + rx[2] * i) - fax) / dx
        if j < ry[1]:
            ty = (float(ry[0] + ry[2] * j) - fay) / dy
        t1 = wp.max(wp.min(wp.min(tx, ty), 1.0), t0)
        if tx <= ty:
            i = i + 1
        else:
            j = j + 1
        tm = 0.5 * (t0 + t1)
        px = _axis_piece(fax + dx * t0, fax + dx * t1, fax + dx * tm, res)
        py = _axis_piece(fay + dy * t0, fay + dy * t1, fay + dy * tm, res)
        x0 = int(px[0])
        y0 = int(py[0])
        v00 = phi[base + y0 * res + x0]
        v10 = phi[base + y0 * res + x0 + 1]
        v01 = phi[base + (y0 + 1) * res + x0]
        v11 = phi[base + (y0 + 1) * res + x0 + 1]
        a1 = v10 - v00
        a2 = v01 - v00
        a3 = v00 - v10 - v01 + v11
        q0 = v00 + a1 * px[1] + a2 * py[1] + a3 * px[1] * py[1]
        q1 = a1 * px[2] + a2 * py[2] + a3 * (px[1] * py[2] + py[1] * px[2])
        q2 = a3 * px[2] * py[2]
        # t0 == 0 is corner a, already sampled by the caller.
        if t0 > 0.0 and q0 < phimin:
            phimin = q0
            pmin = a + (b - a) * t0
        if q2 > 0.0:
            us = -q1 / (2.0 * q2)
            if us > 0.0 and us < 1.0:
                v = q0 + q1 * us + q2 * us * us
                if v < phimin:
                    phimin = v
                    pmin = a + (b - a) * (t0 + (t1 - t0) * us)
        t0 = t1
    return wp.vec3f(phimin, pmin[0], pmin[1])


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

    # Non-finite half-extents or yaw make every corner non-finite. Such a
    # box is treated conservatively, like a degenerate env (oob=1, NaN
    # geometry), and the grid is not sampled: a NaN corner would otherwise
    # become an int(NaN) texel index.
    if not (wp.isfinite(c0) and wp.isfinite(c1) and wp.isfinite(c2)
            and wp.isfinite(c3)):
        out_oob[t] = 1
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    # Candidates: the centre and the corners, then the interior of each side.
    phimin = _sample_phi(phi, base, res, l, h, pos)
    pmin = pos
    for k in range(4):
        ck = _pick4(c0, c1, c2, c3, k)
        v = _sample_phi(phi, base, res, l, h, ck)
        if v < phimin:
            phimin = v
            pmin = ck
    best = wp.vec3f(phimin, pmin[0], pmin[1])
    best = _side_min(phi, base, res, l, h, c0, c1, best)
    best = _side_min(phi, base, res, l, h, c1, c2, best)
    best = _side_min(phi, base, res, l, h, c2, c3, best)
    best = _side_min(phi, base, res, l, h, c3, c0, best)
    phimin = best[0]
    pmin = wp.vec2f(best[1], best[2])

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

    # Central-difference gradient of phi at the argmin point (the centre, a
    # corner or a point inside a side); phi increases into the band, so
    # normalize(grad) already points inward.
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
