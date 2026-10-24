"""Frozen copies of the twelve utility kernels roadmap PR14 edits, at the base 9267388.

Generated from ``git show 9267388:<file>``: every body is VERBATIM; the only edit is the
rename ``<name>`` -> ``_base<name>``. PR14 adds the invalid-env sentinels (a ``valid``
array, and ``has_valid`` where it is optional), the generation-aware self-reset of
``_progress_update_k`` / ``_localize_k`` and the owned ``valid`` / ``generation`` of the
checkpoint set and of the course line. ``tests/test_components.py`` launches these next
to the edited kernels on the same inputs and checks that every VALID env's outputs are
byte-identical (on every device; ``tests/test_components_cuda.py`` on ``cuda:0``). The
helper ``wp.func`` s PR14 does not edit are imported from the library. Do not edit the
bodies.
"""
# ruff: noqa
# fmt: off
from __future__ import annotations

import warp as wp

from track_gen._src.checkpoints import _seg_at_arc
from track_gen._src.collision_geom import (_box_corner, _closest_on_seg, _crossing,
                                           _is_nan3, _pick4, _plane_pass,
                                           _point_to_local_box_dist, _quat_yaw, _rot2,
                                           _safe_normalize2, _safe_normalize3,
                                           _seg_hits_aabb)
from track_gen._src.course_line import _cr_point
from track_gen._src.runtime import _BIG

#: kernel name (in the library) -> its frozen base copy.
BASE_KERNELS: dict = {}


# track_gen/_src/progress.py:96 @ 9267388
@wp.kernel
def _base_progress_update_k(
    cp_position: wp.array(dtype=wp.vec3f),
    cp_left: wp.array(dtype=wp.vec3f),
    cp_right: wp.array(dtype=wp.vec3f),
    cp_tangent: wp.array(dtype=wp.vec3f),
    cp_up_half: wp.array(dtype=wp.float32),
    cp_count: wp.array(dtype=wp.int32),
    max_cp: int,
    position: wp.array(dtype=wp.vec3f),
    prev_pos: wp.array(dtype=wp.vec3f),
    next_cp: wp.array(dtype=wp.int32),
    laps: wp.array(dtype=wp.int32),
    progress: wp.array(dtype=wp.int32),
    out_passed: wp.array(dtype=wp.int32),
    out_cp_passed: wp.array(dtype=wp.int32),
    out_next: wp.array(dtype=wp.int32),
    out_laps: wp.array(dtype=wp.int32),
    out_progress: wp.array(dtype=wp.int32),
    out_wrong_way: wp.array(dtype=wp.int32),
    out_wrong_cp: wp.array(dtype=wp.int32),
    out_dist: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    pos = position[e]
    n = cp_count[e]
    base = e * max_cp

    passed = int(0)
    cp_passed = int(-1)
    wway = int(0)
    wcp = int(-1)

    if n < 1:
        prev_pos[e] = pos
        out_passed[e] = 0
        out_cp_passed[e] = -1
        out_next[e] = next_cp[e]
        out_laps[e] = laps[e]
        out_progress[e] = progress[e]
        out_wrong_way[e] = 0
        out_wrong_cp[e] = -1
        out_dist[e] = wp.nan
        return

    g = next_cp[e]
    if g >= n or g < 0:
        # Defensive clamp: course regenerated without reset (documented as
        # caller error, but never index out of the real range).
        g = 0

    prev = prev_pos[e]
    if _is_nan3(prev) == 0:
        c = _plane_pass(prev, pos, cp_tangent[base + g],
                        cp_left[base + g], cp_right[base + g],
                        cp_up_half[base + g])
        if c == 1:
            passed = int(1)
            cp_passed = g
        elif c == -1:
            wway = int(1)
        # Wrong-checkpoint scan vs the ORIGINAL target g: a double-jump
        # advances g and flags the second crossing in this same update.
        for i in range(n):
            if i != g and wcp == -1:
                if _plane_pass(prev, pos, cp_tangent[base + i],
                               cp_left[base + i], cp_right[base + i],
                               cp_up_half[base + i]) != 0:
                    wcp = i

    ng = g
    lp = laps[e]
    pr = progress[e]
    if passed == 1:
        ng = g + 1
        pr = pr + 1
        if ng == n:
            ng = 0
            lp = lp + 1

    prev_pos[e] = pos
    next_cp[e] = ng
    laps[e] = lp
    progress[e] = pr

    out_passed[e] = passed
    out_cp_passed[e] = cp_passed
    out_next[e] = ng
    out_laps[e] = lp
    out_progress[e] = pr
    out_wrong_way[e] = wway
    out_wrong_cp[e] = wcp
    out_dist[e] = wp.length(cp_position[base + ng] - pos)


# track_gen/_src/localize.py:102 @ 9267388
@wp.kernel
def _base_localize_k(
    center: wp.array(dtype=wp.vec3f),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    warm_window: int,
    position: wp.array(dtype=wp.vec3f),
    last_seg: wp.array(dtype=wp.int32),
    out_s: wp.array(dtype=wp.float32),
    out_n: wp.array(dtype=wp.float32),
    out_n_up: wp.array(dtype=wp.float32),
    out_seg: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    p = position[e]
    m = count[e]
    if m > n_max:
        m = n_max
    if _is_nan3(p) == 1 or m < 3:
        # NaN position pauses the env; degenerate track has no frame. Both
        # drop the warm-start memory so the next real query does a full scan.
        out_s[e] = wp.nan
        out_n[e] = wp.nan
        out_n_up[e] = wp.nan
        out_seg[e] = -1
        last_seg[e] = -1
        return

    base = e * n_max

    # Window selection: full ring by default; the 2W+1 segments around the
    # previous result when warm-started. Exact iff the nearest segment lies
    # inside the window (see the class docstring for the motion contract).
    start = int(0)
    span = m
    last = last_seg[e]
    if warm_window > 0 and last >= 0 and last < m:
        span = 2 * warm_window + 1
        if span > m:
            span = m
        start = last - warm_window

    best_d2 = float(_BIG)
    best_i = int(0)
    best_u = float(0.0)
    for k in range(span):
        i = start + k
        i = ((i % m) + m) % m
        j = i + 1
        if j == m:
            j = 0
        a = center[base + i]
        ab = center[base + j] - a
        denom = wp.max(wp.dot(ab, ab), 1.0e-12)
        u = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
        q = a + ab * u
        d2 = wp.dot(p - q, p - q)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_u = u

    j = best_i + 1
    if j == m:
        j = 0
    a = center[base + best_i]
    ab = center[base + j] - a
    q = a + ab * best_u

    # Arc length from the track's own cumulative table (the closing segment
    # ends at the perimeter), so s is consistent with Track.arclen/length.
    seg_start = arclen[base + best_i]
    seg_end = length[e]
    if best_i + 1 < m:
        seg_end = arclen[base + best_i + 1]

    # Signed offsets in the roll-free frame at the foot point: right_hat =
    # t x up_world points to the RIGHT of the direction of travel (increasing
    # s) — (t.y, -t.x, 0) for planar tangents. Which boundary that is depends
    # on the loop's winding — see the TrackFrame docstring.
    t = _safe_normalize3(ab)
    right_hat = wp.cross(t, wp.vec3f(0.0, 0.0, 1.0))   # (t.y, -t.x, 0)
    rl = wp.length(right_hat)
    if rl < 1.0e-6:
        right_hat = wp.vec3f(1.0, 0.0, 0.0)            # vertical segment guard
    else:
        right_hat = right_hat / rl
    up_hat = wp.cross(right_hat, t)

    s = seg_start + best_u * (seg_end - seg_start)
    if s >= length[e]:
        s = s - length[e]  # u == 1 on the closing segment: wrap the seam
    out_s[e] = s
    out_n[e] = wp.dot(p - q, right_hat)
    out_n_up[e] = wp.dot(p - q, up_hat)
    out_seg[e] = best_i
    last_seg[e] = best_i


# track_gen/_src/localize.py:375 @ 9267388
@wp.kernel
def _base_curvature_raw_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    out_kappa: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    e = tid // n_max
    i = tid - e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3 or i >= m:
        out_kappa[tid] = wp.nan
        return
    base = e * n_max
    ip = i - 1
    if ip < 0:
        ip = m - 1
    j = i + 1
    if j == m:
        j = 0
    v1 = center[base + i] - center[base + ip]
    v2 = center[base + j] - center[base + i]
    # Discrete curvature: turn angle at point i over the mean incident arc.
    # The +z cross component keeps the documented CCW-positive sign and is
    # bit-identical to the legacy 2D scalar cross for planar (z = 0) tracks.
    cr = wp.cross(v1, v2)
    ang = wp.atan2(cr[2], wp.dot(v1, v2))
    ds = 0.5 * (wp.length(v1) + wp.length(v2))
    out_kappa[tid] = ang / wp.max(ds, 1.0e-9)


# track_gen/_src/localize.py:409 @ 9267388
@wp.kernel
def _base_smooth_wrap_k(
    raw: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    window: int,
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    e = tid // n_max
    i = tid - e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3 or i >= m:
        out[tid] = wp.nan
        return
    base = e * n_max
    span = 2 * window + 1
    if span > m:
        span = m  # tiny loops: average each point once, not double-counted
    acc = float(0.0)
    for k in range(span):
        idx = i - window + k
        idx = ((idx % m) + m) % m
        acc = acc + raw[base + idx]
    out[tid] = acc / float(span)


# track_gen/_src/localize.py:515 @ 9267388
@wp.kernel
def _base_speed_profile_k(
    kappa: wp.array(dtype=wp.float32),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    a_lat_max: float,
    a_accel: float,
    a_brake: float,
    v_cap: float,
    out_v: wp.array(dtype=wp.float32),
):
    # One thread per env; the accel/brake passes are inherently sequential.
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    if m < 3:
        for i in range(n_max):
            out_v[base + i] = wp.nan
        return

    # Steady state: lateral-acceleration limit, capped at v_cap. NaN kappa
    # on a real point (defensive) is treated as straight.
    for i in range(m):
        k = wp.abs(kappa[base + i])
        v = v_cap
        if k == k and k > 1.0e-9:
            v = wp.min(wp.sqrt(a_lat_max / k), v_cap)
        out_v[base + i] = v
    for i in range(m, n_max):
        out_v[base + i] = wp.nan

    # Two wrap laps so the closed loop converges: a forward acceleration
    # pass, then a backward braking pass, both over segment i -> i+1 whose
    # length comes from the track's cumulative arc table.
    for lap in range(2):
        for i in range(m):
            j = i + 1
            if j == m:
                j = 0
            seg_end = length[e]
            if i + 1 < m:
                seg_end = arclen[base + i + 1]
            ds = seg_end - arclen[base + i]
            vi = out_v[base + i]
            cap = wp.sqrt(vi * vi + 2.0 * a_accel * ds)
            if cap < out_v[base + j]:
                out_v[base + j] = cap
        for r in range(m):
            i = m - 1 - r
            j = i + 1
            if j == m:
                j = 0
            seg_end = length[e]
            if i + 1 < m:
                seg_end = arclen[base + i + 1]
            ds = seg_end - arclen[base + i]
            vj = out_v[base + j]
            cap = wp.sqrt(vj * vj + 2.0 * a_brake * ds)
            if cap < out_v[base + i]:
                out_v[base + i] = cap


# track_gen/_src/props.py:98 @ 9267388
@wp.kernel
def _base_scan_boundary_k(
    points: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    spacing: float,
    max_props: int,
    cum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.int32),
    out_step: wp.array(dtype=wp.float32),
    out_truncated: wp.array(dtype=wp.int32),
):
    """Also launched by ``_src/checkpoints.py`` (CheckpointSampler) — keep the
    signature and snap semantics in sync. Arc lengths are the true 3D segment
    lengths of the (possibly lifted) polyline; on the flat z = 0 pipeline this
    is identical to the xy length (z contributes 0 to every sum)."""
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        out_count[e] = 0
        out_step[e] = wp.nan
        out_truncated[e] = 0
        return
    base = e * n_max
    s = float(0.0)
    cum[base] = 0.0
    prev = points[base]
    for i in range(1, m):
        p = points[base + i]
        d = p - prev
        s = s + wp.length(d)
        cum[base + i] = s
        prev = p
    dc = points[base] - prev  # closing edge back to point 0
    perim = s + wp.length(dc)

    # Snap in float first: float->int32 conversion of out-of-range values
    # (absurdly small spacing) differs between CPU (INT_MIN) and CUDA
    # (saturating), so clamp against max_props before the cast.
    nf = wp.round(perim / spacing)
    trunc = int(0)
    n = int(0)
    if nf > float(max_props):
        n = max_props
        trunc = int(1)
    else:
        n = int(nf)
        if n < 3:
            n = 3
    out_count[e] = n
    out_step[e] = perim / float(n)
    out_truncated[e] = trunc


# track_gen/_src/checkpoints.py:135 @ 9267388
@wp.kernel
def _base_place_checkpoints_k(
    center: wp.array(dtype=wp.vec3f),
    inner: wp.array(dtype=wp.vec3f),
    outer: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    cum: wp.array(dtype=wp.float32),
    cp_count: wp.array(dtype=wp.int32),
    cp_step: wp.array(dtype=wp.float32),
    max_cp: int,
    out_position: wp.array(dtype=wp.vec3f),
    out_left: wp.array(dtype=wp.vec3f),
    out_right: wp.array(dtype=wp.vec3f),
    out_tangent: wp.array(dtype=wp.vec3f),
    out_up_half: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    e = tid // max_cp
    k = tid - e * max_cp

    n = cp_count[e]
    if k >= n:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        out_position[tid] = nan3
        out_left[tid] = nan3
        out_right[tid] = nan3
        out_tangent[tid] = nan3
        out_up_half[tid] = wp.nan
        return

    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    step = cp_step[e]
    # Reconstructs the scan's perimeter (step = perim/n); max queried arc is
    # (n-1)*step < perim, so the closing segment is never overrun.
    perim = step * float(n)

    it = _seg_at_arc(cum, base, m, perim, float(k) * step)
    i = int(it[0])
    t = it[1]
    j = i + 1
    if j == m:
        j = 0
    # Index-aligned lerp: the same segment/parameter on all three polylines
    # gives the road cross-section at this checkpoint.
    out_position[tid] = center[base + i] + (center[base + j] - center[base + i]) * t
    out_left[tid] = inner[base + i] + (inner[base + j] - inner[base + i]) * t
    out_right[tid] = outer[base + i] + (outer[base + j] - outer[base + i]) * t
    out_tangent[tid] = _safe_normalize3(center[base + j] - center[base + i])
    # Track cross-sections are vertically unbounded for the crossing test.
    out_up_half[tid] = _BIG


# track_gen/_src/collision.py:92 @ 9267388
@wp.kernel
def _base_box_query_segments_k(
    inner: wp.array(dtype=wp.vec3f),
    outer: wp.array(dtype=wp.vec3f),
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
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

    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        # Degenerate/invalid track: conservative OOB, NaN geometry.
        out_oob[t] = 1
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    yw = _quat_yaw(orientation[t])
    he = half_extents[t]
    ux = _rot2(yw, wp.vec2f(1.0, 0.0))
    uy = _rot2(yw, wp.vec2f(0.0, 1.0))
    c0 = _box_corner(pos, ux, uy, he, 0)
    c1 = _box_corner(pos, ux, uy, he, 1)
    c2 = _box_corner(pos, ux, uy, he, 2)
    c3 = _box_corner(pos, ux, uy, he, 3)

    base = e * n_max

    # Per-corner crossing counts and min distances to each boundary polyline.
    cn_in = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    cn_out = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    dc_in = wp.vec4f(_BIG, _BIG, _BIG, _BIG)
    dc_out = wp.vec4f(_BIG, _BIG, _BIG, _BIG)

    crossed = int(0)
    best_d = _BIG
    best_pt = wp.vec2f(0.0, 0.0)
    best_bnd = int(0)
    best_i = int(0)

    for j in range(2 * m):
        bnd = int(0)
        i = j
        if j >= m:
            bnd = int(1)
            i = j - m
        i2 = i + 1
        if i2 == m:
            i2 = 0
        a = wp.vec2f(0.0, 0.0)
        b = wp.vec2f(0.0, 0.0)
        if bnd == 0:
            a3 = inner[base + i]
            b3 = inner[base + i2]
            a = wp.vec2f(a3[0], a3[1])
            b = wp.vec2f(b3[0], b3[1])
        else:
            a3 = outer[base + i]
            b3 = outer[base + i2]
            a = wp.vec2f(a3[0], a3[1])
            b = wp.vec2f(b3[0], b3[1])

        # Box<->segment distance candidates: box corners vs the segment ...
        cand_d = _BIG
        cand_pt = a
        for k in range(4):
            ck = _pick4(c0, c1, c2, c3, k)
            cr = float(_crossing(ck, a, b))
            cp = _closest_on_seg(ck, a, b)
            dk = wp.length(ck - cp)
            if bnd == 0:
                cn_in[k] = cn_in[k] + cr
                dc_in[k] = wp.min(dc_in[k], dk)
            else:
                cn_out[k] = cn_out[k] + cr
                dc_out[k] = wp.min(dc_out[k], dk)
            if dk < cand_d:
                cand_d = dk
                cand_pt = cp

        # ... plus the segment start vertex vs the solid box (the end vertex is
        # the next segment's start, so the closed loop covers every vertex).
        al = _rot2(-yw, a - pos)
        bl = _rot2(-yw, b - pos)
        d_end = _point_to_local_box_dist(al, he)
        if d_end < cand_d:
            cand_d = d_end
            cand_pt = a
        if _seg_hits_aabb(al, bl, he) == 1:
            crossed = int(1)
            cand_d = 0.0
            cand_pt = _closest_on_seg(pos, a, b)

        if cand_d < best_d:
            best_d = cand_d
            best_pt = cand_pt
            best_bnd = bnd
            best_i = i

    inside = int(1)
    worst_pen = 0.0
    for k in range(4):
        pen_k = 0.0
        if int(cn_in[k]) % 2 == 1:   # corner inside the inner hole
            inside = int(0)
            pen_k = dc_in[k]
        if int(cn_out[k]) % 2 == 0:  # corner outside the outer loop
            inside = int(0)
            pen_k = wp.max(pen_k, dc_out[k])
        worst_pen = wp.max(worst_pen, pen_k)

    oob = int(0)
    if inside == 0 or crossed == 1:
        oob = int(1)

    dist = best_d
    if oob == 1:
        dist = -worst_pen

    # Normal of the argmin segment, oriented into the band via the
    # index-aligned centerline point.
    i2 = best_i + 1
    if i2 == m:
        i2 = 0
    sa = wp.vec2f(0.0, 0.0)
    sb = wp.vec2f(0.0, 0.0)
    if best_bnd == 0:
        sa3 = inner[base + best_i]
        sb3 = inner[base + i2]
        sa = wp.vec2f(sa3[0], sa3[1])
        sb = wp.vec2f(sb3[0], sb3[1])
    else:
        sa3 = outer[base + best_i]
        sb3 = outer[base + i2]
        sa = wp.vec2f(sa3[0], sa3[1])
        sb = wp.vec2f(sb3[0], sb3[1])
    seg = sb - sa
    nrm = wp.vec2f(-seg[1], seg[0])
    cpt3 = center[base + best_i]
    cpt = wp.vec2f(cpt3[0], cpt3[1])
    if wp.length(nrm) < 1.0e-8:
        nrm = cpt - best_pt
    nrm = _safe_normalize2(nrm)
    if wp.dot(cpt - best_pt, nrm) < 0.0:
        nrm = -nrm

    out_oob[t] = oob
    out_distance[t] = dist
    out_nearest[t] = best_pt
    out_normal[t] = nrm
    out_boundary[t] = best_bnd


# track_gen/_src/collision_sdf.py:37 @ 9267388
@wp.kernel
def _base_track_aabb_k(
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


# track_gen/_src/collision_sdf.py:73 @ 9267388
@wp.kernel
def _base_sdf_bake_k(
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


# track_gen/_src/collision_discs.py:82 @ 9267388
@wp.kernel
def _base_box_query_discs_k(
    discs: wp.array(dtype=wp.vec3f),
    disc_count: wp.array(dtype=wp.int32),
    d_max: int,
    radius: float,
    max_boxes: int,
    position: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_extents: wp.array(dtype=wp.vec2f),
    out_hit: wp.array(dtype=wp.int32),
    out_disc: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos3 = position[t]
    if _is_nan3(pos3) == 1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return
    # Planar collision semantics: project the box pose to xy.
    pos = wp.vec2f(pos3[0], pos3[1])

    yw = _quat_yaw(orientation[t])
    he = half_extents[t]
    nd = disc_count[e]
    if nd > d_max:
        nd = d_max
    base = e * d_max

    best = int(-1)
    best_pen = float(0.0)
    for i in range(nd):
        c3 = discs[base + i]
        if _is_nan3(c3) == 0:
            c = wp.vec2f(c3[0], c3[1])
            q = _rot2(-yw, c - pos)
            pen = radius - _point_to_local_box_dist(q, he)
            if pen >= 0.0 and (best == -1 or pen > best_pen):
                best = i
                best_pen = pen

    if best == -1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return

    c3 = discs[base + best]
    c = wp.vec2f(c3[0], c3[1])
    # Closest point of the box to the disc center, then step back onto the
    # disc boundary toward it (approximate when c is deep inside the box).
    q = _rot2(-yw, c - pos)
    qcl = wp.vec2f(wp.clamp(q[0], -he[0], he[0]), wp.clamp(q[1], -he[1], he[1]))
    wcl = pos + _rot2(yw, qcl)
    out_hit[t] = 1
    out_disc[t] = best
    out_depth[t] = best_pen
    out_nearest[t] = c + _safe_normalize2(wcl - c) * radius


# track_gen/_src/course_line.py:39 @ 9267388
@wp.kernel
def _base_sample_line_k(
    gate_pos: wp.array(dtype=wp.vec3f),
    gate_count: wp.array(dtype=wp.int32),
    gate_valid: wp.array(dtype=wp.int32),
    max_gates: int,
    spg: int,
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    n_max = max_gates * spg
    e = tid // n_max
    j = tid - e * n_max
    n = gate_count[e]
    if n > max_gates:
        n = max_gates
    m = n * spg
    degenerate = n < 3 or gate_valid[e] == 0
    if j == 0:
        if degenerate:
            count[e] = 0
            valid[e] = 0
        else:
            count[e] = m
            valid[e] = gate_valid[e]
    if j >= m or degenerate:
        center[tid] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        return
    gbase = e * max_gates
    i = j // spg
    t = float(j - i * spg) / float(spg)
    i0 = ((i - 1) % n + n) % n
    i2 = (i + 1) % n
    i3 = (i + 2) % n
    center[tid] = _cr_point(gate_pos[gbase + i0], gate_pos[gbase + i],
                            gate_pos[gbase + i2], gate_pos[gbase + i3], t)


BASE_KERNELS["_progress_update_k"] = _base_progress_update_k
BASE_KERNELS["_localize_k"] = _base_localize_k
BASE_KERNELS["_curvature_raw_k"] = _base_curvature_raw_k
BASE_KERNELS["_smooth_wrap_k"] = _base_smooth_wrap_k
BASE_KERNELS["_speed_profile_k"] = _base_speed_profile_k
BASE_KERNELS["_scan_boundary_k"] = _base_scan_boundary_k
BASE_KERNELS["_place_checkpoints_k"] = _base_place_checkpoints_k
BASE_KERNELS["_box_query_segments_k"] = _base_box_query_segments_k
BASE_KERNELS["_track_aabb_k"] = _base_track_aabb_k
BASE_KERNELS["_sdf_bake_k"] = _base_sdf_bake_k
BASE_KERNELS["_box_query_discs_k"] = _base_box_query_discs_k
BASE_KERNELS["_sample_line_k"] = _base_sample_line_k
