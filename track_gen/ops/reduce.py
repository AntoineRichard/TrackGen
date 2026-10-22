"""Device-side reductions to a one-element flag (no host readback).

The flags feed device-side control flow, e.g.
:meth:`track_gen.exec.LaunchContext.skip_unless`, so they never leave the device.
"""
from __future__ import annotations

import warp as wp

from .check import _device_str
from .fill import fill

__all__ = ["any_set"]


@wp.kernel
def _any_set_k(mask: wp.array(dtype=wp.int32), flag: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        wp.atomic_max(flag, 0, 1)


def _check(name: str, arr, ndim: int = 1) -> None:
    if not isinstance(arr, wp.array):
        raise TypeError(f"{name} must be a wp.array, got {type(arr).__name__}")
    if arr.dtype is not wp.int32:
        got = getattr(arr.dtype, "__name__", str(arr.dtype))
        raise TypeError(f"{name} must have dtype int32, got {got}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got ndim={arr.ndim}")


def any_set(ctx, mask: "wp.array", flag_out: "wp.array") -> None:
    """``flag_out[0] = 1`` if any ``mask[e] != 0``, else ``0`` (on ``ctx``'s stream).

    Two launches, both capturable and allocation-free: zero ``flag_out``, then one
    thread per element raises it with an integer ``atomic_max`` (order-independent,
    so the result is deterministic).

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        mask: 1-D ``int32`` array (e.g. an ``[E]`` per-env mask).
        flag_out: ``[1]`` ``int32`` array on the same device.

    Raises:
        TypeError: ``mask`` or ``flag_out`` is not an ``int32`` ``wp.array``.
        ValueError: a wrong shape, or the arrays and ``ctx`` on different devices
            (host-side, before any launch).
    """
    _check("mask", mask)
    _check("flag_out", flag_out)
    if flag_out.shape[0] != 1:
        raise ValueError(f"flag_out must have shape (1,), got {tuple(flag_out.shape)}")
    if str(mask.device) != str(flag_out.device) or str(mask.device) != _device_str(ctx.device):
        raise ValueError(
            f"mask ({mask.device}), flag_out ({flag_out.device}) and the launch context "
            f"({ctx.device}) must be on the same device")
    fill(ctx, flag_out, 0)
    if mask.shape[0] > 0:
        ctx.launch(_any_set_k, mask.shape[0], [mask, flag_out])
