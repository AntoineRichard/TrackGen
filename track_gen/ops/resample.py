"""Arc-length resampling of closed 2-D loops (flat per-env layout).

Every op takes a :class:`~track_gen.exec.LaunchContext` first, launches only through
it, never host-synchronizes and never allocates: the working buffers come from an
:class:`ArcScratch` the caller allocates once (``ArcScratch.alloc``, typically in a
generator's ``alloc_scratch``).

Layout: env ``e`` owns ``[e*stride, (e+1)*stride)`` of every flat array. Cumulative
arc length is accumulated in float64 and stored as float32. Output slots past an
env's point count are written ``(nan, nan)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .check import (_check_1d, _check_float, _check_int, _check_min, _check_no_alias,
                    check_array)

if TYPE_CHECKING:  # pragma: no cover
    from ..exec.context import LaunchContext

__all__ = ["ArcScratch", "arc_resample", "resample_constant_spacing", "resample_uniform"]


# ---------------------------------------------------------------------------------------
# Kernels (moved from track_gen/_src/warp_pipeline.py at 1077b18; the only edits are the
# count clamps of the uniform resample, which change nothing for a count <= stride).
# ---------------------------------------------------------------------------------------

@wp.kernel
def _resample_scan_k(
    c: wp.array(dtype=wp.vec2f),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
):
    # One thread per env e. Segment lengths seg[e*n_max+i]=|c[i+1]-c[i]|
    # (i+1 wraps via %count[e]) and cumulative arc s[e*(n_max+1)+0]=0,
    # s[..+i+1]=s[..+i]+seg[i] for i in range(count[e]).
    # Running sum in float64 to limit accumulation drift.
    # PARITY: when count[e]==n_max for all e, produces identical output to the
    # former fixed-N kernel (same float64 accumulation order, same modular indexing).
    # count[e] is clamped to n_max so an oversized count cannot index past the env's row
    # (a no-op for every valid count).
    e = wp.tid()
    cn = wp.min(count[e], n_max)
    b = e * n_max
    es = e * (n_max + 1)
    s[es] = float(0.0)
    acc = wp.float64(0.0)
    for i in range(cn):
        d = c[b + (i + 1) % cn] - c[b + i]
        l = wp.length(d)
        seg[b + i] = l
        acc = acc + wp.float64(l)
        s[es + i + 1] = wp.float32(acc)


@wp.kernel
def _resample_lookup_k(
    c: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    n_max: int,
    count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per output slot t; e = t // n_max, k = t % n_max.
    # k >= count[e] -> NaN pad (padding region).
    # Otherwise: target tk = k * total / count[e], linear scan then lerp.
    # PARITY: when count[e]==n_max for all e, result is bit-identical to the
    # former fixed-N kernel (same formula, same scan bounds, same wrap modulus).
    t = wp.tid()
    e = t // n_max
    k = t % n_max
    cn = wp.min(count[e], n_max)    # see _resample_scan_k
    if k >= cn:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return
    eb = e * n_max
    es = e * (n_max + 1)
    total = s[es + cn]
    tk = float(k) * total / float(cn)
    idx = int(0)
    while idx < cn - 1 and s[es + idx + 1] < tk:
        idx = idx + 1
    s0 = s[es + idx]
    segl = wp.max(seg[eb + idx], float(1.0e-12))
    frac = wp.clamp((tk - s0) / segl, float(0.0), float(1.0))
    p0 = c[eb + idx]
    p1 = c[eb + (idx + 1) % cn]
    out[t] = p0 + frac * (p1 - p0)


@wp.kernel
def _cs_scan_k(c: wp.array(dtype=wp.vec2f), N: int, spacing: wp.float32, n_max: int,
               seg: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32),
               count: wp.array(dtype=wp.int32), truncated: wp.array(dtype=wp.int32)):
    # One thread per env e. Closed-loop seg lengths + cumulative arc s (len N+1),
    # then count = floor(total/spacing)+1, capped so target (count-1)*spacing < total
    # and to n_max. Mirrors geometry._resample_one's spacing branch. truncated[e] records
    # whether the UNCLAMPED count exceeded n_max (the loop's tail was dropped); the
    # validity gate fails such envs closed.
    e = wp.tid()
    b = e * N
    es = e * (N + 1)
    s[es] = float(0.0)
    acc = wp.float64(0.0)
    for i in range(N):
        d = c[b + (i + 1) % N] - c[b + i]
        l = wp.length(d)
        seg[b + i] = l
        acc = acc + wp.float64(l)
        s[es + i + 1] = wp.float32(acc)
    total = wp.float32(acc)
    if not (total > 0.0):      # catches total <= 0 AND NaN (never-accepted envs)
        count[e] = 0
        truncated[e] = 0
        return
    k = int(wp.floor(total / spacing)) + 1
    while k > 1 and wp.float32(k - 1) * spacing >= total:
        k = k - 1
    count[e] = wp.min(wp.max(k, 1), n_max)
    truncated[e] = wp.where(k > n_max, int(1), int(0))


@wp.kernel
def _cs_lookup_k(c: wp.array(dtype=wp.vec2f), seg: wp.array(dtype=wp.float32),
                 s: wp.array(dtype=wp.float32), N: int, spacing: wp.float32, n_max: int,
                 count: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
    # One thread per OUTPUT slot t (dim = E*n_max). k >= count[e] -> NaN pad.
    t = wp.tid()
    e = t // n_max
    k = t % n_max
    if k >= count[e]:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return
    eb = e * N
    es = e * (N + 1)
    target = wp.float32(k) * spacing
    idx = int(0)
    while idx < N - 1 and s[es + idx + 1] < target:
        idx = idx + 1
    s0 = s[es + idx]
    segl = wp.max(seg[eb + idx], float(1.0e-12))
    frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))
    p0 = c[eb + idx]
    p1 = c[eb + (idx + 1) % N]
    out[t] = p0 + frac * (p1 - p0)


@wp.kernel
def _arc_scan_k(
    dense: wp.array(dtype=wp.vec2f),
    M: int,
    num: int,
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
):
    # One thread per env e. NaN-aware generalization of _resample_scan_k: first
    # COMPACT the real (both-components-finite) points IN ORDER (dropping interior
    # NaN too, like the oracle's pe[isfinite.all(-1)]), then build the closed-loop
    # segment lengths and cumulative arc length over those R real points.
    e = wp.tid()
    db = e * M
    rb = e * M
    es = e * (M + 1)

    # Compaction: walk i=0..M-1, append every finite point in order.
    r = int(0)
    for i in range(M):
        p = dense[db + i]
        if wp.isfinite(p[0]) and wp.isfinite(p[1]):
            real_pts[rb + r] = p
            r = r + 1
    count_r[e] = r
    # count_out: R>=2 -> num, else 0.
    count_out[e] = wp.where(r >= 2, num, 0)

    # R >= 2: closed-loop arc length over real_pts[0..R-1]. seg[j] = |real[(j+1)%R]
    # - real[j]| (j=R-1 is the wrap |real[0]-real[R-1]|); s[0]=0, s[j+1]=s[j]+seg[j].
    # Accumulate in float64 to limit accumulation drift.
    if r >= 2:
        s[es] = float(0.0)
        acc = wp.float64(0.0)
        for j in range(r):
            nxt = real_pts[rb + (j + 1) % r]
            l = wp.length(nxt - real_pts[rb + j])
            seg[rb + j] = l
            acc = acc + wp.float64(l)
            s[es + j + 1] = wp.float32(acc)
    # R < 2: leave seg/s untouched; _arc_lookup_k emits NaN for this env.


@wp.kernel
def _arc_lookup_k(
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    M: int,
    num: int,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per OUTPUT point t (dim = E*num); e = env index, k = point-within-env.
    # R = count_r[e]. R < 2 -> NaN (matches _resample_one's degenerate full-NaN row).
    # Else: linear-scan searchsorted(right=False).clamp(max=R-1) over the closed
    # real-loop arc length, then lerp. Mirrors _resample_lookup_k with variable R and
    # the wrap p1 = real[0] for idx == R-1 ((idx+1)%R).
    t = wp.tid()
    e = t // num
    k = t % num
    rb = e * M
    es = e * (M + 1)

    r = count_r[e]
    if r < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    total = s[es + r]
    target = float(k) * total / float(num)

    # Linear scan: first j in [0, R-1] with s[es+j+1] >= target, clamped to R-1.
    idx = int(0)
    while idx < r - 1 and s[es + idx + 1] < target:
        idx = idx + 1

    s0 = s[es + idx]
    segl = wp.max(seg[rb + idx], float(1.0e-12))
    frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))

    p0 = real_pts[rb + idx]
    p1 = real_pts[rb + (idx + 1) % r]   # closed[idx+1]: real[0] when idx == R-1
    out[t] = p0 + frac * (p1 - p0)


@wp.kernel
def _arc_scan_selected_k(
    dense: wp.array(dtype=wp.vec2f),
    active: wp.array(dtype=wp.int32),
    M: int,
    num: int,
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
):
    # Same as _arc_scan_k for active envs. Inactive envs write only their counts,
    # allowing downstream lookup to skip them and leave output rows untouched.
    e = wp.tid()
    if active[e] <= 0:
        count_r[e] = int(0)
        count_out[e] = int(0)
        return

    db = e * M
    rb = e * M
    es = e * (M + 1)

    r = int(0)
    for i in range(M):
        p = dense[db + i]
        if wp.isfinite(p[0]) and wp.isfinite(p[1]):
            real_pts[rb + r] = p
            r = r + 1
    count_r[e] = r
    count_out[e] = wp.where(r >= 2, num, 0)

    if r >= 2:
        s[es] = float(0.0)
        acc = wp.float64(0.0)
        for j in range(r):
            nxt = real_pts[rb + (j + 1) % r]
            l = wp.length(nxt - real_pts[rb + j])
            seg[rb + j] = l
            acc = acc + wp.float64(l)
            s[es + j + 1] = wp.float32(acc)


@wp.kernel
def _arc_lookup_selected_k(
    real_pts: wp.array(dtype=wp.vec2f),
    seg: wp.array(dtype=wp.float32),
    s: wp.array(dtype=wp.float32),
    count_r: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    M: int,
    num: int,
    out: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // num
    if active[e] <= 0:
        return

    k = t % num
    rb = e * M
    es = e * (M + 1)
    r = count_r[e]
    if r < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    total = s[es + r]
    target = float(k) * total / float(num)
    idx = int(0)
    while idx < r - 1 and s[es + idx + 1] < target:
        idx = idx + 1

    s0 = s[es + idx]
    segl = wp.max(seg[rb + idx], float(1.0e-12))
    frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))
    p0 = real_pts[rb + idx]
    p1 = real_pts[rb + (idx + 1) % r]
    out[t] = p0 + frac * (p1 - p0)


# ---------------------------------------------------------------------------------------
# Scratch
# ---------------------------------------------------------------------------------------

class ArcScratch:
    """Working buffers for the resample ops (wraps existing arrays; never allocates).

    For ``E`` envs and an input stride ``stride`` (points per env of the op's ``src``):

    * ``real``: ``[>= E*stride]`` ``vec2f`` -- compacted real (finite) points
      (:func:`arc_resample` only).
    * ``seg``: ``[>= E*stride]`` ``float32`` -- per-segment lengths.
    * ``s``: ``[>= E*(stride+1)]`` ``float32`` -- cumulative arc length.
    * ``count_real``: ``[>= E]`` ``int32`` -- real-point count per env
      (:func:`arc_resample` only; readable after the call).

    :func:`resample_uniform` and :func:`resample_constant_spacing` only use ``seg``
    and ``s``; ``real``/``count_real`` may be ``None`` for them.

    Build one per generator in ``alloc_scratch`` with :meth:`alloc`, or wrap buffers
    you already own (views/slices included) with the constructor.
    """

    __slots__ = ("real", "seg", "s", "count_real")

    def __init__(self, real, seg, s, count_real) -> None:
        self.real = real
        self.seg = seg
        self.s = s
        self.count_real = count_real

    @classmethod
    def alloc(cls, E: int, stride: int, device, *, compact: bool = True) -> "ArcScratch":
        """Allocate scratch for ``E`` envs of ``stride`` input points each (construction
        time only: this allocates).

        Args:
            E: number of envs.
            stride: input points per env (the ``stride`` of the op's ``src``).
            device: Warp device.
            compact: also allocate ``real`` and ``count_real`` (needed by
                :func:`arc_resample`); ``False`` leaves them ``None``.
        """
        E, M = int(E), int(stride)
        if E < 0 or M < 1:
            raise ValueError(
                f"ArcScratch.alloc: need E >= 0 and stride >= 1, got E={E}, stride={M}")
        dev = str(wp.get_device(device))
        return cls(
            real=wp.empty(E * M, dtype=wp.vec2f, device=dev) if compact else None,
            seg=wp.empty(E * M, dtype=wp.float32, device=dev),
            s=wp.empty(E * (M + 1), dtype=wp.float32, device=dev),
            count_real=wp.empty(E, dtype=wp.int32, device=dev) if compact else None,
        )

    def __repr__(self) -> str:
        def n(a):
            return None if a is None else int(a.shape[0])
        return (f"ArcScratch(real={n(self.real)}, seg={n(self.seg)}, s={n(self.s)}, "
                f"count_real={n(self.count_real)})")


def _check_scratch(scratch, E: int, M: int, device, *, compact: bool) -> None:
    if not isinstance(scratch, ArcScratch):
        raise ValueError(f"scratch must be an ArcScratch, got {type(scratch).__name__}")
    if compact:
        _check_min("scratch.real", scratch.real, E * M, wp.vec2f, device)
        _check_min("scratch.count_real", scratch.count_real, E, wp.int32, device)
    _check_min("scratch.seg", scratch.seg, E * M, wp.float32, device)
    _check_min("scratch.s", scratch.s, E * (M + 1), wp.float32, device)


# ---------------------------------------------------------------------------------------
# Private launchers (no validation). The legacy warp_pipeline wrappers call these, so
# their historical shape tolerance is unchanged.
# ---------------------------------------------------------------------------------------

def _arc_resample_launch(ctx, src, M, num, scratch, out, count_out, E, active=None) -> None:
    if active is None:
        ctx.launch(_arc_scan_k, E,
                   [src, M, num, scratch.real, scratch.seg, scratch.s,
                    scratch.count_real, count_out])
        ctx.launch(_arc_lookup_k, E * num,
                   [scratch.real, scratch.seg, scratch.s, scratch.count_real, M, num, out])
    else:
        ctx.launch(_arc_scan_selected_k, E,
                   [src, active, M, num, scratch.real, scratch.seg, scratch.s,
                    scratch.count_real, count_out])
        ctx.launch(_arc_lookup_selected_k, E * num,
                   [scratch.real, scratch.seg, scratch.s, scratch.count_real, active,
                    M, num, out])


def _resample_uniform_launch(ctx, src, count, n, seg, s, out) -> None:
    E = count.shape[0]
    ctx.launch(_resample_scan_k, E, [src, n, count, seg, s])
    ctx.launch(_resample_lookup_k, E * n, [src, seg, s, n, count, out])


def _constant_spacing_launch(ctx, src, N, spacing, n_max, seg, s, out, count_out,
                             truncated_out) -> None:
    E = count_out.shape[0]
    ctx.launch(_cs_scan_k, E, [src, N, float(spacing), n_max,
               seg, s, count_out, truncated_out])
    ctx.launch(_cs_lookup_k, E * n_max, [src, seg, s, N,
               float(spacing), n_max, count_out, out])


# ---------------------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------------------

def _env_count(src, stride: int, dev) -> int:
    """``E = src.shape[0] // stride`` after validating ``src`` (1-D ``vec2f`` on ``dev``)."""
    n = _check_1d("src", src, wp.vec2f, dev)
    if n % stride:
        raise ValueError(f"src length {n} is not a multiple of stride {stride}")
    return n // stride


def _check_out_not_scratch(out, scratch, *, compact: bool) -> None:
    """``out`` must not share memory with a scratch buffer the lookup reads."""
    names = (("real", "seg", "s") if compact else ("seg", "s"))
    for name in names:
        _check_no_alias("out", out, f"scratch.{name}", getattr(scratch, name))


def arc_resample(ctx: "LaunchContext", src: "wp.array", stride: int, num: int,
                 scratch: ArcScratch, out: "wp.array", count_out: "wp.array", *,
                 active: "wp.array | None" = None) -> None:
    """NaN-aware arc-length resample of closed loops: ``stride`` points in, ``num`` out.

    Per env, the points whose x AND y are finite are compacted in order (interior
    NaN rows are dropped too) into ``R`` real points. With ``R >= 2`` the env's
    output row is ``num`` points uniformly spaced by arc length along the CLOSED
    loop through them (point 0 is the first real point; the last segment wraps back
    to it) and ``count_out[e] = num``. With ``R < 2`` the row is all NaN and
    ``count_out[e] = 0``. ``scratch.count_real[e]`` receives ``R``.

    With ``active``, only envs with ``active[e] > 0`` are resampled; every other
    env keeps its ``out`` row bit for bit and gets ``count_out[e] =
    scratch.count_real[e] = 0``.

    Two launches (``E`` scan threads, ``E*num`` lookup threads); no allocation, no
    host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` input loops (NaN rows allowed); ``E`` is
            ``src.shape[0] // stride``.
        stride: input points per env (``>= 1``).
        num: output points per env (``>= 1``).
        scratch: :class:`ArcScratch` with ``real >= E*stride``, ``seg >= E*stride``,
            ``s >= E*(stride+1)``, ``count_real >= E``.
        out: ``[E*num]`` ``vec2f`` output; must not alias a scratch buffer (it may
            alias ``src``: the lookup reads only the scratch).
        count_out: ``[E]`` ``int32`` output.
        active: optional ``[E]`` ``int32`` env mask.

    Raises:
        ValueError: a wrong type, shape, dtype or device, ``stride``/``num`` below 1,
            or ``out`` aliasing the scratch (host-side, before any launch).
    """
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    num = _check_int("num", num, 1)
    E = _env_count(src, stride, dev)
    check_array("count_out", count_out, (E,), wp.int32, dev)
    check_array("out", out, (E * num,), wp.vec2f, dev)
    if active is not None:
        check_array("active", active, (E,), wp.int32, dev)
    _check_scratch(scratch, E, stride, dev, compact=True)
    _check_out_not_scratch(out, scratch, compact=True)
    _arc_resample_launch(ctx, src, stride, num, scratch, out, count_out, E, active=active)


def resample_uniform(ctx: "LaunchContext", src: "wp.array", stride: int,
                     scratch: ArcScratch, out: "wp.array", *, count: "wp.array") -> None:
    """Count-aware arc-length-uniform resample of closed loops (``stride`` in and out).

    Env ``e``'s first ``count[e]`` points (which must be finite; no NaN
    compaction) are treated as a closed loop and resampled to ``count[e]`` points
    uniformly spaced by arc length, starting at point 0. Slots ``>= count[e]`` of the
    row are written NaN; ``count[e] <= 0`` gives an all-NaN row and a count above
    ``stride`` is clamped to ``stride``.

    Two launches (``E`` scan threads, ``E*stride`` lookup threads); no allocation, no
    host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` input loops; ``E`` is ``src.shape[0] // stride``.
        stride: points per env of ``src`` and ``out`` (``>= 1``).
        scratch: :class:`ArcScratch` with ``seg >= E*stride`` and
            ``s >= E*(stride+1)`` (``real``/``count_real`` unused, may be ``None``).
        out: ``[E*stride]`` ``vec2f`` output; must not alias ``src`` or the scratch.
        count: ``[E]`` ``int32`` real points per env (keyword-only, required).

    Raises:
        ValueError: a wrong type, shape, dtype or device, ``stride < 1``, or ``out``
            aliasing ``src`` or the scratch (host-side, before any launch).
    """
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    E = _env_count(src, stride, dev)
    check_array("count", count, (E,), wp.int32, dev)
    check_array("out", out, (E * stride,), wp.vec2f, dev)
    _check_scratch(scratch, E, stride, dev, compact=False)
    _check_no_alias("out", out, "src", src)
    _check_out_not_scratch(out, scratch, compact=False)
    _resample_uniform_launch(ctx, src, count, stride, scratch.seg, scratch.s, out)


def resample_constant_spacing(ctx: "LaunchContext", src: "wp.array", stride: int,
                              spacing: float, out_stride: int, scratch: ArcScratch,
                              out: "wp.array", count_out: "wp.array",
                              truncated_out: "wp.array") -> None:
    """Resample fully-real closed loops of ``stride`` points to a constant ``spacing``.

    Per env: ``total`` = closed-loop perimeter; ``count = floor(total/spacing) + 1``
    trimmed so ``(count-1)*spacing < total``, then capped at ``out_stride``. Point
    ``k`` sits at arc length ``k*spacing`` from point 0. ``truncated_out[e] = 1`` iff
    the uncapped count exceeded ``out_stride`` (the loop's tail was dropped). A loop
    whose perimeter is not ``> 0`` (zero or NaN) gets ``count_out[e] = 0`` and
    ``truncated_out[e] = 0``. Slots ``>= count_out[e]`` are written NaN.

    Two launches; no allocation, no host sync (the truncation flag stays on device).

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` input loops (every point real); ``E`` is
            ``src.shape[0] // stride``.
        stride: input points per env (``>= 1``).
        spacing: target spacing (finite, ``> 0``).
        out_stride: output row capacity per env (``>= 1``).
        scratch: :class:`ArcScratch` with ``seg >= E*stride`` and ``s >= E*(stride+1)``.
        out: ``[E*out_stride]`` ``vec2f`` output; must not alias ``src`` or the scratch.
        count_out: ``[E]`` ``int32`` output.
        truncated_out: ``[E]`` ``int32`` output (0/1).

    Raises:
        ValueError: a wrong type, shape, dtype or device, a bad scalar, or ``out``
            aliasing ``src`` or the scratch (host-side, before any launch).
    """
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    out_stride = _check_int("out_stride", out_stride, 1)
    spacing = _check_float("spacing", spacing, positive=True)
    E = _env_count(src, stride, dev)
    check_array("count_out", count_out, (E,), wp.int32, dev)
    check_array("truncated_out", truncated_out, (E,), wp.int32, dev)
    check_array("out", out, (E * out_stride,), wp.vec2f, dev)
    _check_scratch(scratch, E, stride, dev, compact=False)
    _check_no_alias("out", out, "src", src)
    _check_out_not_scratch(out, scratch, compact=False)
    _constant_spacing_launch(ctx, src, stride, spacing, out_stride, scratch.seg, scratch.s,
                             out, count_out, truncated_out)
