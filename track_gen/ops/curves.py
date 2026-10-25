"""Dense evaluation of closed curves through per-env control points.

Every op takes a :class:`~track_gen.exec.LaunchContext` first, launches only through
it, never host-synchronizes and never allocates. Env ``e``'s controls are
``src[e*stride : (e+1)*stride]``; segment ``i`` of a closed curve runs from control
``i`` to control ``i+1`` (mod ``K``) and is sampled ``per_seg`` times at
``u = s / (per_seg - endpoint)``, ``s = 0 .. per_seg-1``:

* ``endpoint=False``: ``u`` in ``[0, 1)``; consecutive segments do not repeat the
  shared control and the loop closes through the resampler's wrap edge.
* ``endpoint=True``: ``u`` in ``[0, 1]``; each segment ends exactly on the next
  control (which the next segment repeats).

Output ``out[e*stride*per_seg + i*per_seg + s]``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .check import _check_1d, _check_int, check_array
from .funcs import catmull_rom

if TYPE_CHECKING:  # pragma: no cover
    from ..exec.context import LaunchContext

__all__ = ["closed_catmull_rom", "closed_polyline"]


# ---------------------------------------------------------------------------------------
# Kernels (track_gen/_src at 1077b18: warp_generate_voronoi._catmull_rom_dense_k /
# _assemble_polygon_selected_k and warp_generate_hull._catmull_rom_k /
# _assemble_polygon_selected_k, with the per-segment parameter divisor made
# `per_seg - ep`, the env mask made optional (`masked`) and the count clamped to the
# stride; none of these touches the float arithmetic).
# ---------------------------------------------------------------------------------------

@wp.func
def _ctrl_at(ctrl: wp.array(dtype=wp.vec2f), base: int, i: int, count: int) -> wp.vec2f:
    return ctrl[base + (i % count)]


@wp.kernel
def _catmull_rom_dense_k(
    ctrl: wp.array(dtype=wp.vec2f),
    stride: int,
    per_seg: int,
    ep: int,
    out: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    per_env = stride * per_seg
    e = t // per_env
    rem = t % per_env
    i = rem // per_seg
    s = rem % per_seg
    base = e * stride
    p0 = _ctrl_at(ctrl, base, i - 1 + stride, stride)
    p1 = _ctrl_at(ctrl, base, i, stride)
    p2 = _ctrl_at(ctrl, base, i + 1, stride)
    p3 = _ctrl_at(ctrl, base, i + 2, stride)
    u = float(s) / float(per_seg - ep)
    out[t] = catmull_rom(p0, p1, p2, p3, u)


@wp.kernel
def _catmull_rom_count_k(
    ctrl: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    stride: int,
    per_seg: int,
    ep: int,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per dense sample t (dim = E * stride * per_seg). Segments i >= cnt
    # (and every segment when cnt < 2) write NaN and drop out via the arc-resample.
    t = wp.tid()
    per_env = stride * per_seg
    e = t // per_env
    rem = t % per_env
    i = rem // per_seg
    s = rem % per_seg
    b = e * stride
    cnt = wp.min(count[e], stride)      # an oversized count stays inside the row

    if i >= cnt or cnt < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    p0 = _ctrl_at(ctrl, b, i - 1 + cnt, cnt)
    p1 = _ctrl_at(ctrl, b, i, cnt)
    p2 = _ctrl_at(ctrl, b, i + 1, cnt)
    p3 = _ctrl_at(ctrl, b, i + 2, cnt)

    u = float(s) / float(per_seg - ep)
    out[t] = catmull_rom(p0, p1, p2, p3, u)


@wp.kernel
def _polyline_k(
    ctrl: wp.array(dtype=wp.vec2f),
    stride: int,
    per_seg: int,
    ep: int,
    masked: int,
    active: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Straight closed polyline; with masked != 0 only envs with active[e] > 0 are
    # written (inactive rows are left untouched). `active` may be None when masked == 0.
    t = wp.tid()
    per_env = stride * per_seg
    e = t // per_env
    if masked != 0:
        if active[e] <= 0:
            return
    rem = t % per_env
    i = rem // per_seg
    s = rem % per_seg
    base = e * stride
    p0 = ctrl[base + i]
    p1 = ctrl[base + (i + 1) % stride]
    u = float(s) / float(per_seg - ep)
    out[t] = (1.0 - u) * p0 + u * p1


@wp.kernel
def _polyline_count_k(
    ctrl: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    stride: int,
    per_seg: int,
    ep: int,
    masked: int,
    active: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Count-aware straight closed polyline; masking as in _polyline_k.
    t = wp.tid()
    per_env = stride * per_seg
    e = t // per_env
    if masked != 0:
        if active[e] <= 0:
            return

    rem = t % per_env
    i = rem // per_seg
    s = rem % per_seg
    b = e * stride
    cnt = wp.min(count[e], stride)      # an oversized count stays inside the row
    cm = wp.max(cnt, 1)

    if i >= cnt or cnt < 2:
        out[t] = wp.vec2f(wp.nan, wp.nan)
        return

    c0 = ctrl[b + i]
    c1 = ctrl[b + (i + 1) % cm]
    u = float(s) / float(per_seg - ep)
    out[t] = (1.0 - u) * c0 + u * c1


# ---------------------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------------------

def _curve_args(ctx, src, stride, per_seg, out, count, endpoint):
    dev = ctx.device
    stride = _check_int("stride", stride, 1)
    per_seg = _check_int("per_seg", per_seg, 2 if endpoint else 1)
    n = _check_1d("src", src, wp.vec2f, dev)
    if n % stride:
        raise ValueError(f"src length {n} is not a multiple of stride {stride}")
    E = n // stride
    check_array("out", out, (E * stride * per_seg,), wp.vec2f, dev)
    if count is not None:
        check_array("count", count, (E,), wp.int32, dev)
    return dev, E, stride, per_seg, int(bool(endpoint))


def closed_catmull_rom(ctx: "LaunchContext", src: "wp.array", stride: int, per_seg: int,
                       out: "wp.array", *, count: "wp.array | None" = None,
                       endpoint: bool = False) -> None:
    """Sample the closed uniform Catmull-Rom spline (tension 0.5) through each env's
    controls.

    ``out[e, i*per_seg + s] = catmull_rom(p[i-1], p[i], p[i+1], p[i+2]; u)`` with
    indices mod ``K`` and ``u = s / (per_seg - endpoint)`` (see the module docstring).
    ``count=None``: ``K = stride`` for every env. With ``count``:
    ``K = min(count[e], stride)``; segments ``i >= K`` (all of them when ``K < 2``)
    are written NaN, which the NaN-aware :func:`~track_gen.ops.arc_resample` drops.
    One launch (``E*stride*per_seg`` threads); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` controls; ``E`` is ``src.shape[0] // stride``
            (must divide exactly).
        stride: controls per env (``>= 1``).
        per_seg: samples per segment (``>= 1``; ``>= 2`` with ``endpoint=True``).
        out: ``[E*stride*per_seg]`` ``vec2f`` output.
        count: optional ``[E]`` ``int32`` controls per env (keyword-only).
        endpoint: include ``u = 1`` in each segment.

    Raises:
        ValueError: a wrong type, shape, dtype or device, or a bad scalar.
    """
    dev, E, stride, per_seg, ep = _curve_args(ctx, src, stride, per_seg, out, count,
                                              endpoint)
    dim = E * stride * per_seg
    if count is None:
        ctx.launch(_catmull_rom_dense_k, dim, [src, stride, per_seg, ep, out])
    else:
        ctx.launch(_catmull_rom_count_k, dim, [src, count, stride, per_seg, ep, out])


def closed_polyline(ctx: "LaunchContext", src: "wp.array", stride: int, per_seg: int,
                    out: "wp.array", *, count: "wp.array | None" = None,
                    endpoint: bool = False, active: "wp.array | None" = None) -> None:
    """Sample the closed straight polyline through each env's controls.

    ``out[e, i*per_seg + s] = (1-u)*p[i] + u*p[i+1]`` (index mod ``K``) with
    ``u = s / (per_seg - endpoint)``. ``count=None``: ``K = stride``; with ``count``,
    ``K = min(count[e], stride)`` and segments ``i >= K`` (all when ``K < 2``) are
    written NaN. Same layout as :func:`closed_catmull_rom`. With ``active``, only envs
    with ``active[e] > 0`` are written and every other row is left untouched: the
    drop-in "straight" fallback for the envs whose smooth curve self-crossed. One
    launch (``E*stride*per_seg`` threads); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        src: ``[E*stride]`` ``vec2f`` controls; ``E`` is ``src.shape[0] // stride``.
        stride: controls per env (``>= 1``).
        per_seg: samples per segment (``>= 1``; ``>= 2`` with ``endpoint=True``).
        out: ``[E*stride*per_seg]`` ``vec2f`` output.
        count: optional ``[E]`` ``int32`` controls per env (keyword-only).
        endpoint: include ``u = 1`` in each segment.
        active: optional ``[E]`` ``int32`` env mask (keyword-only).

    Raises:
        ValueError: a wrong type, shape, dtype or device, or a bad scalar.
    """
    dev, E, stride, per_seg, ep = _curve_args(ctx, src, stride, per_seg, out, count,
                                              endpoint)
    masked = int(active is not None)
    if masked:
        check_array("active", active, (E,), wp.int32, dev)
    dim = E * stride * per_seg
    if count is None:
        ctx.launch(_polyline_k, dim, [src, stride, per_seg, ep, masked, active, out])
    else:
        ctx.launch(_polyline_count_k, dim,
                   [src, count, stride, per_seg, ep, masked, active, out])
