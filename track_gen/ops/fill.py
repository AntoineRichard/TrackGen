"""Typed, stream-explicit fills.

``wp.array.zero_()`` / ``fill_()`` issue a memset on Warp's *current* stream, which is the
wrong stream once launches are stream-explicit. These kernels run on the context's stream
instead. One concrete kernel per dtype (no generic overloads, so nothing is compiled
lazily inside a caller's capture).
"""
from __future__ import annotations

import warp as wp

from .check import _check_1d


@wp.kernel
def _fill_i32_k(arr: wp.array(dtype=wp.int32), v: wp.int32):
    arr[wp.tid()] = v


@wp.kernel
def _fill_u32_k(arr: wp.array(dtype=wp.uint32), v: wp.uint32):
    arr[wp.tid()] = v


@wp.kernel
def _fill_f32_k(arr: wp.array(dtype=wp.float32), v: wp.float32):
    arr[wp.tid()] = v


@wp.kernel
def _fill_vec2f_k(arr: wp.array(dtype=wp.vec2f), v: wp.vec2f):
    arr[wp.tid()] = v


@wp.kernel
def _fill_vec3f_k(arr: wp.array(dtype=wp.vec3f), v: wp.vec3f):
    arr[wp.tid()] = v


@wp.kernel
def _fill_quatf_k(arr: wp.array(dtype=wp.quatf), v: wp.quatf):
    arr[wp.tid()] = v


_KERNELS = {
    wp.int32: (_fill_i32_k, wp.int32),
    wp.uint32: (_fill_u32_k, wp.uint32),
    wp.float32: (_fill_f32_k, wp.float32),
    wp.vec2f: (_fill_vec2f_k, wp.vec2f),
    wp.vec3f: (_fill_vec3f_k, wp.vec3f),
    wp.quatf: (_fill_quatf_k, wp.quatf),
}


def fill(ctx, arr: "wp.array", value) -> None:
    """Set every element of the 1-D array ``arr`` to ``value`` on ``ctx``'s stream.

    One launch of ``arr.shape[0]`` threads; no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        arr: ``[n]`` (1-D) ``wp.array`` of ``int32``, ``uint32``, ``float32``,
            ``vec2f``, ``vec3f`` or ``quatf`` on ``ctx.device``.
        value: a scalar; for ``vec2f``/``vec3f`` a vector of that type, a sequence of
            its components, or one scalar broadcast to every component; for ``quatf``
            a ``wp.quatf`` or a sequence of its four components (no broadcast).

    Raises:
        TypeError: unsupported dtype.
        ValueError: ``arr`` is not a ``wp.array``, not 1-D or not on ``ctx.device``,
            or a scalar ``value`` for a ``quatf`` array.
    """
    if not isinstance(arr, wp.array):
        raise ValueError(f"arr must be a wp.array, got {type(arr).__name__}")
    entry = _KERNELS.get(arr.dtype)
    if entry is None:
        raise TypeError(f"ops.fill: unsupported dtype {getattr(arr.dtype, '__name__', arr.dtype)}")
    if arr.ndim != 1:
        raise ValueError(f"ops.fill: expected a 1-D array, got ndim={arr.ndim}")
    _check_1d("arr", arr, arr.dtype, ctx.device)
    kernel, ctor = entry
    if ctor is wp.quatf and not isinstance(value, ctor) and not hasattr(value, "__len__"):
        raise ValueError("ops.fill: a quatf value needs four components, got a scalar")
    if ctor in (wp.vec2f, wp.vec3f, wp.quatf):
        v = value if isinstance(value, ctor) else ctor(*value) if hasattr(value, "__len__") else ctor(value)
    else:
        v = ctor(value)
    ctx.launch(kernel, arr.shape[0], [arr, v])
