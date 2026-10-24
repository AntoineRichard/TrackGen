"""Per-env geometry of closed 2-D polygons (flat per-env layout).

Every op takes a :class:`~track_gen.exec.LaunchContext` first, launches only through
it, never host-synchronizes and never allocates. Env ``e`` owns
``[e*stride, (e+1)*stride)`` of every flat point array; ``count[e]`` (when given)
is the number of real points at the start of that row.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .check import _check_1d, _check_float, _check_int, _check_min, check_array
from .constants import SIZE_BOUNDS
from .fill import _fill_i32_k
from .funcs import ccw as _ccw
from .funcs import nan0 as _nan0

if TYPE_CHECKING:  # pragma: no cover
    from ..exec.context import LaunchContext

__all__ = ["angle_sort", "normalize_extent", "self_intersections", "thickness",
           "turning_number"]


# ---------------------------------------------------------------------------------------
# Device functions and kernels (moved from track_gen/_src at 1077b18: warp_pipeline.py
# for crossings/thickness/turning, warp_generate_polar.py for the dense extent
# normalisation, warp_gate.py for the counted normalisation and the angle sort). The only
# edits are the count clamps of the three per-env polygon kernels, a no-op for every
# count <= stride.
# ---------------------------------------------------------------------------------------

@wp.func
def _self_intersections_func(poly: wp.array(dtype=wp.vec2f), base: int, N: int) -> int:
    # Proper-crossing double-loop count for the env whose points start at `base`.
    # Each coordinate is read through _nan0 (NaN->0 guard), which is a no-op on finite
    # inputs and reproduces the validity border check's nan_to_num(outer/inner, nan=0.0).
    count = int(0)
    for i in range(N):
        for j in range(i + 1, N):
            diff = j - i
            circ_dist = wp.min(diff, N - diff)
            if circ_dist <= 1:
                continue
            Ai = poly[base + i]
            Bi = poly[base + (i + 1) % N]
            Aj = poly[base + j]
            Bj = poly[base + (j + 1) % N]
            aix = _nan0(Ai[0]); aiy = _nan0(Ai[1])
            bix = _nan0(Bi[0]); biy = _nan0(Bi[1])
            ajx = _nan0(Aj[0]); ajy = _nan0(Aj[1])
            bjx = _nan0(Bj[0]); bjy = _nan0(Bj[1])
            d1 = _ccw(ajx, ajy, bjx, bjy, aix, aiy)
            d2 = _ccw(ajx, ajy, bjx, bjy, bix, biy)
            d3 = _ccw(aix, aiy, bix, biy, ajx, ajy)
            d4 = _ccw(aix, aiy, bix, biy, bjx, bjy)
            # Scale-relative collinearity tolerance (matches geometry.SELF_X_REL): a
            # straddle counts only if both endpoints clear the other segment's line by
            # more than SELF_X_REL * (that segment length). |d| = len * perp-offset, so
            # the threshold eps = REL * len^2. Kills the float32 sign-flip false positives
            # on near-collinear/straight segments without missing genuine crossings.
            lj2 = (bjx - ajx) * (bjx - ajx) + (bjy - ajy) * (bjy - ajy)
            li2 = (bix - aix) * (bix - aix) + (biy - aiy) * (biy - aiy)
            ej = float(1.0e-3) * lj2
            ei = float(1.0e-3) * li2
            seg_ij = (d1 > ej and d2 < -ej) or (d1 < -ej and d2 > ej)
            seg_ji = (d3 > ei and d4 < -ei) or (d3 < -ei and d4 > ei)
            if seg_ij and seg_ji:
                count = count + 1
    return count


@wp.kernel
def _self_intersections_by_i_k(
    poly: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    # One thread per segment start i. This preserves the same pair predicate as
    # _self_intersections_func, but avoids serializing every O(N^2) pair check onto
    # one thread per environment. Crossing counts are rare, so atomics are cheap.
    t = wp.tid()
    e = t // n_max
    i = t % n_max
    N = wp.min(count[e], n_max)     # clamp: an oversized count stays inside the row
    if i >= N:
        return

    base = e * n_max
    Ai = poly[base + i]
    Bi = poly[base + (i + 1) % N]
    aix = _nan0(Ai[0]); aiy = _nan0(Ai[1])
    bix = _nan0(Bi[0]); biy = _nan0(Bi[1])

    for j in range(i + 1, N):
        diff = j - i
        circ_dist = wp.min(diff, N - diff)
        if circ_dist <= 1:
            continue
        Aj = poly[base + j]
        Bj = poly[base + (j + 1) % N]
        ajx = _nan0(Aj[0]); ajy = _nan0(Aj[1])
        bjx = _nan0(Bj[0]); bjy = _nan0(Bj[1])
        d1 = _ccw(ajx, ajy, bjx, bjy, aix, aiy)
        d2 = _ccw(ajx, ajy, bjx, bjy, bix, biy)
        d3 = _ccw(aix, aiy, bix, biy, ajx, ajy)
        d4 = _ccw(aix, aiy, bix, biy, bjx, bjy)
        lj2 = (bjx - ajx) * (bjx - ajx) + (bjy - ajy) * (bjy - ajy)
        li2 = (bix - aix) * (bix - aix) + (biy - aiy) * (biy - aiy)
        ej = float(1.0e-3) * lj2
        ei = float(1.0e-3) * li2
        seg_ij = (d1 > ej and d2 < -ej) or (d1 < -ej and d2 > ej)
        seg_ji = (d3 > ei and d4 < -ei) or (d3 < -ei and d4 > ei)
        if seg_ij and seg_ji:
            wp.atomic_add(out, e, int(1))


@wp.func
def _thickness_func(pts: wp.array(dtype=wp.vec2f), base: int, N: int, band: int) -> float:
    # thickness = min(rad_min, 0.5 * sep_min) for the env whose points start at `base`.
    # sep_min = min pairwise distance over pairs with circ_dist > band;
    # rad_min  = 1 / max Menger curvature.
    # --- sep_min: min dist over pairs with circ_dist > band ---
    sep_min = float(1.0e30)
    for i in range(N):
        for j in range(i + 1, N):
            diff = j - i
            circ_dist = wp.min(diff, N - diff)
            if circ_dist > band:
                pi = pts[base + i]
                pj = pts[base + j]
                d = wp.length(pi - pj)
                sep_min = wp.min(sep_min, d)

    # --- rad_min: 1 / max Menger curvature ---
    kappa_max = float(0.0)
    for i in range(N):
        xp = pts[base + (i + N - 1) % N]
        xc = pts[base + i]
        xn = pts[base + (i + 1) % N]
        a = xc - xp
        bb = xn - xc
        cc = xn - xp
        cross = a[0] * bb[1] - a[1] * bb[0]
        area = 0.5 * wp.abs(cross)
        denom = wp.max(wp.length(a) * wp.length(bb) * wp.length(cc), float(1.0e-12))
        kappa = 4.0 * area / denom
        kappa_max = wp.max(kappa_max, kappa)
    rad_min = 1.0 / wp.max(kappa_max, float(1.0e-12))

    return wp.min(rad_min, 0.5 * sep_min)


@wp.kernel
def _thickness_k(
    pts: wp.array(dtype=wp.vec2f),
    band: wp.array(dtype=wp.int32),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Delegates to _thickness_func with this env's band (count
    # clamped to the row so an oversized count stays inside it).
    e = wp.tid()
    out[e] = _thickness_func(pts, e * n_max, wp.min(count[e], n_max), band[e])


@wp.func
def _turning_func(c: wp.array(dtype=wp.vec2f), base: int, N: int) -> float:
    # Signed total turning of the closed polygon whose points start at `base`.
    # Edge angle theta_i = atan2(d_i.y, d_i.x) for raw edge d_i = c[(i+1)%N] - c[i]
    # (atan2 is scale-invariant, so no normalization is needed; the zero-length
    # edge gives atan2(0,0)=0, matching the oracle's safe_normalize-then-atan2).
    # Per edge, dtheta = theta_i - theta_{i-1} wrapped into (-pi, pi] via
    # atan2(sin, cos); the sum over all edges is the turning number.
    total = float(0.0)
    for i in range(N):
        di = c[base + (i + 1) % N] - c[base + i]
        ip = (i + N - 1) % N
        dp = c[base + (ip + 1) % N] - c[base + ip]
        theta_i = wp.atan2(di[1], di[0])
        theta_prev = wp.atan2(dp[1], dp[0])
        dth = theta_i - theta_prev
        total = total + wp.atan2(wp.sin(dth), wp.cos(dth))
    return total


@wp.kernel
def _turning_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    # One thread per env e. Delegates to _turning_func over [e*n_max, e*n_max+count[e])
    # (count clamped to the row).
    e = wp.tid()
    out[e] = _turning_func(c, e * n_max, wp.min(count[e], n_max))


@wp.kernel
def _normalize_extent_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
):
    # One thread per env e. Center by bbox and scale so each env's longest bbox dimension
    # exactly matches target_extent after arc resampling.
    e = wp.tid()
    b = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(N):
        p = points[b + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(N):
        p = points[b + i]
        points[b + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


@wp.kernel
def _normalize_extent_count_k(
    position: wp.array(dtype=wp.vec2f),
    max_gates: int,
    count: wp.array(dtype=wp.int32),
    target_extent: float,
):
    e = wp.tid()
    base = e * max_gates
    cnt = count[e]
    if cnt < 0:
        cnt = 0
    if cnt > max_gates:
        cnt = max_gates

    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            min_x = wp.min(min_x, p[0])
            max_x = wp.max(max_x, p[0])
            min_y = wp.min(min_y, p[1])
            max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            position[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)
        else:
            position[base + i] = wp.vec2f(wp.nan, wp.nan)


_C_SIZE_LO = wp.constant(wp.float32(SIZE_BOUNDS[0]))
_C_SIZE_HI = wp.constant(wp.float32(SIZE_BOUNDS[1]))


@wp.kernel
def _normalize_extent_sized_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
    size_e: wp.array(dtype=wp.float32),
):
    # Roadmap PR13d: _normalize_extent_k to the extent target_extent * size[e] (the
    # per-env size, clamped to SIZE_BOUNDS by comparisons, folded into the finite scalar
    # target before the division; the per-point arithmetic is the scalar kernel's
    # verbatim, so size 1 is bit-exact and no FMA contraction changes).
    e = wp.tid()
    size = size_e[e]
    size = wp.where(size < _C_SIZE_LO, _C_SIZE_LO, size)
    size = wp.where(size > _C_SIZE_HI, _C_SIZE_HI, size)
    target_extent = target_extent * size
    b = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(N):
        p = points[b + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(N):
        p = points[b + i]
        points[b + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


@wp.kernel
def _angle_sort_k(
    src: wp.array(dtype=wp.vec2f),
    src_stride: int,
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    keys: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.vec2f),
    m_floor: int,
):
    # One thread per env e. Orders the env's FIRST m = clamp(count[e], m_floor,
    # min(src_stride, max_gates)) points ascending by the centroid-relative angle
    # key = atan2(dx, dy) (X FIRST) about the f64 centroid of those m points, with a
    # stable insertion sort; dst slots [m, max_gates) are written NaN. The sort reads
    # only behind its write frontier, so uninitialised keys/dst scratch is never consumed.
    # m_floor = 0 is the gate ordering (count <= 0 -> all NaN); m_floor = 1 reproduces
    # the legacy bezier/hull sort (count < 1 sorts the first point).
    e = wp.tid()
    src_base = e * src_stride
    dst_base = e * max_gates
    m = count[e]
    if m < m_floor:
        m = m_floor
    if m > src_stride:
        m = src_stride
    if m > max_gates:
        m = max_gates

    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(max_gates):
        if i < m:
            p = src[src_base + i]
            sx = sx + wp.float64(p[0])
            sy = sy + wp.float64(p[1])

    cx = wp.float32(0.0)
    cy = wp.float32(0.0)
    if m > 0:
        cx = wp.float32(sx / wp.float64(m))
        cy = wp.float32(sy / wp.float64(m))

    for i in range(max_gates):
        if i < m:
            p = src[src_base + i]
            key = wp.atan2(p[0] - cx, p[1] - cy)
            j = i - 1
            while j >= 0 and keys[dst_base + j] > key:
                keys[dst_base + j + 1] = keys[dst_base + j]
                dst[dst_base + j + 1] = dst[dst_base + j]
                j = j - 1
            keys[dst_base + j + 1] = key
            dst[dst_base + j + 1] = p
        else:
            dst[dst_base + i] = wp.vec2f(wp.nan, wp.nan)


# ---------------------------------------------------------------------------------------
# Private launchers (no validation; the legacy _src wrappers call these).
# ---------------------------------------------------------------------------------------

def _self_intersections_launch(ctx, poly, count, stride, out) -> None:
    E = count.shape[0]
    ctx.launch(_fill_i32_k, E, [out, 0])
    ctx.launch(_self_intersections_by_i_k, E * stride, [poly, stride, count, out])


def _turning_launch(ctx, poly, count, stride, out) -> None:
    ctx.launch(_turning_k, count.shape[0], [poly, stride, count, out])


def _thickness_launch(ctx, poly, count, stride, band, out) -> None:
    ctx.launch(_thickness_k, count.shape[0], [poly, band, stride, count, out])


def _normalize_extent_launch(ctx, pts, stride, target_extent, E, count=None) -> None:
    if count is None:
        ctx.launch(_normalize_extent_k, E, [pts, stride, float(target_extent)])
    else:
        ctx.launch(_normalize_extent_count_k, E,
                   [pts, stride, count, float(target_extent)])


def _normalize_extent_sized_launch(ctx, pts, stride, target_extent, E, size) -> None:
    # Roadmap PR13d: the per-env size twin (no count variant: no built-in needs one).
    ctx.launch(_normalize_extent_sized_k, E, [pts, stride, float(target_extent), size])


def _angle_sort_launch(ctx, src, stride, count, out_stride, keys, out, m_floor) -> None:
    ctx.launch(_angle_sort_k, count.shape[0],
               [src, stride, count, out_stride, keys, out, int(m_floor)])


# ---------------------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------------------

def _env_count(src, stride: int, dev) -> int:
    """``E = src.shape[0] // stride`` after validating ``src`` (1-D ``vec2f`` on ``dev``)."""
    n = _check_1d("src", src, wp.vec2f, dev)
    if n % stride:
        raise ValueError(f"src length {n} is not a multiple of stride {stride}")
    return n // stride


def _poly_args(ctx, src, stride, count):
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    E = _env_count(src, stride, dev)
    check_array("count", count, (E,), wp.int32, dev)
    return dev, E, stride


def self_intersections(ctx: "LaunchContext", src: "wp.array", stride: int,
                       out: "wp.array", *, count: "wp.array") -> None:
    """Number of proper self-crossings of each closed polygon.

    For env ``e`` the polygon is its first ``N = min(count[e], stride)`` points
    (closed: the last edge returns to point 0). ``out[e]`` counts the pairs of edges
    ``i < j`` with circular distance ``> 1`` (adjacent edges never count) that
    cross properly: each segment's endpoints must straddle the other's line by more
    than ``1e-3 * len^2`` of that segment (a scale-relative collinearity tolerance
    that rejects float32 sign flips on near-collinear edges). Coordinates are read
    through :func:`~track_gen.ops.funcs.nan0` (NaN reads as 0). ``N <= 3`` gives 0.
    Deterministic (integer atomics).

    Two launches (zero ``out``; ``E*stride`` threads); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` polygons; ``E`` is ``src.shape[0] // stride``.
        stride: points per env in ``src`` (``>= 1``).
        out: ``[E]`` ``int32`` crossing counts.
        count: ``[E]`` ``int32`` points per env (keyword-only, required; clamped to
            ``stride``).

    Raises:
        ValueError: a wrong type, shape, dtype or device (host-side, before any
            launch).
    """
    dev, E, stride = _poly_args(ctx, src, stride, count)
    check_array("out", out, (E,), wp.int32, dev)
    _self_intersections_launch(ctx, src, count, stride, out)


def turning_number(ctx: "LaunchContext", src: "wp.array", stride: int,
                   out: "wp.array", *, count: "wp.array") -> None:
    """Signed total turning (radians) of each closed polygon.

    Sum over the ``N = min(count[e], stride)`` vertices of the exterior angle between
    consecutive edges, each wrapped into ``(-pi, pi]``: about ``+2*pi`` for a simple
    counter-clockwise loop, ``-2*pi`` clockwise. A zero-length edge has angle 0.
    ``N <= 0`` gives 0; a NaN point makes the env's result NaN. One launch (``E``
    threads); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` polygons; ``E`` is ``src.shape[0] // stride``.
        stride: points per env in ``src`` (``>= 1``).
        out: ``[E]`` ``float32`` output.
        count: ``[E]`` ``int32`` points per env (keyword-only, required; clamped to
            ``stride``).

    Raises:
        ValueError: a wrong type, shape, dtype or device.
    """
    dev, E, stride = _poly_args(ctx, src, stride, count)
    check_array("out", out, (E,), wp.float32, dev)
    _turning_launch(ctx, src, count, stride, out)


def thickness(ctx: "LaunchContext", src: "wp.array", stride: int, band: "wp.array",
              out: "wp.array", *, count: "wp.array") -> None:
    """Thickness of each closed polygon: ``min(1 / max Menger curvature,
    0.5 * min |p_i - p_j|)`` over vertex pairs with circular index distance
    ``> band[e]``.

    ``band`` is the neighbour half-window excluded from the distance term, in points.
    The built-in validity gate uses ``band = max(1, round(2 * half_width / l0))`` for a
    track of half-width ``half_width`` whose segments have mean length ``l0``: pairs
    closer than a track width along the loop are skipped.

    Degenerate cases, with ``N = min(count[e], stride)``: the curvature maximum is
    floored at ``1e-12`` and the distance term starts at ``1e30``, so a polygon with
    no curvature and no vertex pair beyond the band (e.g. ``N < 3`` with
    ``band[e] >= 1``) returns ``1e12``. NaN points are not guarded: a row containing
    one gives an unspecified, generally finite, result (the running ``min``/``max``
    mostly skip NaN distances and curvatures instead of propagating them). One launch
    (``E`` threads, ``O(N^2)`` work per env); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` polygons; ``E`` is ``src.shape[0] // stride``.
        stride: points per env in ``src`` (``>= 1``).
        band: ``[E]`` ``int32`` excluded neighbour half-window per env.
        out: ``[E]`` ``float32`` output.
        count: ``[E]`` ``int32`` points per env (keyword-only, required; clamped to
            ``stride``).

    Raises:
        ValueError: a wrong type, shape, dtype or device.
    """
    dev, E, stride = _poly_args(ctx, src, stride, count)
    check_array("band", band, (E,), wp.int32, dev)
    check_array("out", out, (E,), wp.float32, dev)
    _thickness_launch(ctx, src, count, stride, band, out)


def normalize_extent(ctx: "LaunchContext", src: "wp.array", stride: int,
                     target_extent: float, *, count: "wp.array | None" = None,
                     size: "wp.array | None" = None) -> None:
    """Centre each env's bounding box at the origin and scale it uniformly so its
    longest side equals ``target_extent`` (in place).

    The extent is floored at ``1e-8`` before dividing. With ``count=None`` all
    ``stride`` points of each env are used (rows are expected to be finite; a NaN
    coordinate stays NaN). With ``count``, only the first
    ``clamp(count[e], 0, stride)`` points are used and the remaining slots are
    written NaN. One launch (``E`` threads); no allocation, no host sync.

    ``size`` (roadmap PR13d) is a per-env size multiplier of the target: env ``e``'s
    longest side becomes ``target_extent * size[e]``, each value clamped at use to
    :data:`track_gen.ops.constants.SIZE_BOUNDS` (by comparisons: a NaN passes). The
    size multiplies the finite scalar target before the division (no per-point
    arithmetic changes), so a size-1 array gives the ``size=None`` bytes exactly, on CPU
    and CUDA; ``size=None`` runs the unchanged kernels.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` points, modified in place; ``E`` is
            ``src.shape[0] // stride``.
        stride: points per env (``>= 1``).
        target_extent: longest bbox side after scaling (finite), e.g.
            ``ops.constants.TARGET_EXTENT * config.scale``.
        count: optional ``[E]`` ``int32`` real points per env (keyword-only).
        size: optional ``[E]`` ``float32`` per-env size multiplier of the target
            (keyword-only), on the op's device; not combinable with ``count``.

    Raises:
        ValueError: a wrong type, shape, dtype or device, ``src.shape[0]`` not a
            multiple of ``stride``, or ``count`` and ``size`` both given.
    """
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    target_extent = _check_float("target_extent", target_extent)
    E = _env_count(src, stride, dev)
    if count is not None and size is not None:
        raise ValueError("normalize_extent: size= with count= is not supported (no "
                         "counted variant of the per-env size)")
    if count is not None:
        check_array("count", count, (E,), wp.int32, dev)
    if size is not None:
        check_array("size", size, (E,), wp.float32, dev)
        _normalize_extent_sized_launch(ctx, src, stride, target_extent, E, size)
        return
    _normalize_extent_launch(ctx, src, stride, target_extent, E, count=count)


def angle_sort(ctx: "LaunchContext", src: "wp.array", stride: int, keys: "wp.array",
               out: "wp.array", *, count: "wp.array",
               out_stride: "int | None" = None) -> None:
    """Sort each env's first points by angle around their centroid.

    With ``m = clamp(count[e], 0, min(stride, out_stride))``, env ``e``'s output
    row is its first ``m`` source points ordered ascending by
    ``atan2(x - cx, y - cy)`` (x FIRST: the built-in generators' ordering), where
    ``(cx, cy)`` is the float64 centroid of those ``m`` points; the sort is stable.
    Slots ``[m, out_stride)`` are written NaN, so ``count[e] <= 0`` gives an all-NaN
    row. One launch (``E`` threads); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` points; ``E`` is ``src.shape[0] // stride``.
        stride: points per env in ``src`` (``>= 1``).
        keys: ``[>= E*out_stride]`` ``float32`` scratch.
        out: ``[E*out_stride]`` ``vec2f`` output.
        count: ``[E]`` ``int32`` points to sort per env (keyword-only, required).
        out_stride: points per env in ``out`` (default ``stride``).

    Raises:
        ValueError: a wrong type, shape, dtype or device.
    """
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    out_stride = stride if out_stride is None else _check_int("out_stride", out_stride, 1)
    E = _env_count(src, stride, dev)
    check_array("count", count, (E,), wp.int32, dev)
    check_array("out", out, (E * out_stride,), wp.vec2f, dev)
    _check_min("keys", keys, E * out_stride, wp.float32, dev)
    _angle_sort_launch(ctx, src, stride, count, out_stride, keys, out, 0)
