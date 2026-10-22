"""Host-side argument validation shared by the ops and the library facades.

:func:`check_array` is the one array validator of the library (the facades' and
utilities' ``_check_arr`` is this function). The private helpers below are what the
ops use for scratch buffers (a minimum length instead of an exact shape) and scalar
arguments. All of them run on the host before any launch and never touch device data.
"""
from __future__ import annotations

import warp as wp

__all__ = ["check_array"]

_I32_MIN = -(2 ** 31)
_I32_MAX = 2 ** 31 - 1


def _device_str(device) -> str:
    """Canonical device string of ``device``: ``"cuda"`` -> ``"cuda:0"`` (the current
    CUDA device), a ``wp.Device`` -> its alias. Host-only, no sync. An identifier Warp
    does not know (e.g. ``"cuda:0"`` on a CPU-only host) is returned as given, so the
    mismatch error still names it."""
    try:
        return str(wp.get_device(device))
    except Exception:  # noqa: BLE001 - unknown identifier: compare it verbatim
        return str(device)


def _check_device(name: str, arr, device) -> None:
    got = str(arr.device)
    if got == str(device):          # fast path: already canonical and equal
        return
    want = _device_str(device)
    if got != want:
        raise ValueError(f"{name} must be on device {want!r}, got {got!r}")


def check_array(name: str, arr, shape: tuple, dtype, device) -> None:
    """Validate a ``wp.array``'s type, shape, dtype, and device.

    Raises ``ValueError`` naming ``name`` on the first mismatch. Shared body
    behind the family's ``_validate_inputs`` / ``_validate_position`` /
    ``_check_arr`` wrappers; ``shape`` is the expected shape tuple. ``device`` is
    canonicalised before the comparison, so ``"cuda"`` matches an array on
    ``cuda:0`` when that is the current CUDA device.

    Args:
        name: argument name used in the error message.
        arr: the value to check.
        shape: the exact expected shape, e.g. ``(E * N,)``.
        dtype: the expected Warp dtype (compared by identity, e.g. ``wp.vec2f``).
        device: the expected device (a ``wp.Device`` or device string, e.g.
            ``"cuda"``, ``"cuda:0"`` or ``"cpu"``).

    Raises:
        ValueError: ``arr`` is not a ``wp.array``, or its shape, dtype or device
            differs from the expected one.
    """
    if not isinstance(arr, wp.array):
        raise ValueError(f"{name} must be a wp.array, got {type(arr).__name__}")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(arr.shape)}")
    if arr.dtype is not dtype:
        want = getattr(dtype, "__name__", str(dtype))
        got = getattr(arr.dtype, "__name__", str(arr.dtype))
        raise ValueError(f"{name} must have dtype {want}, got {got}")
    _check_device(name, arr, device)


def _check_min(name: str, arr, min_len: int, dtype, device) -> None:
    """Validate a 1-D scratch ``wp.array``: type, dtype, device and ``len >= min_len``."""
    if not isinstance(arr, wp.array):
        raise ValueError(f"{name} must be a wp.array, got {type(arr).__name__}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(arr.shape)}")
    if arr.dtype is not dtype:
        want = getattr(dtype, "__name__", str(dtype))
        got = getattr(arr.dtype, "__name__", str(arr.dtype))
        raise ValueError(f"{name} must have dtype {want}, got {got}")
    _check_device(name, arr, device)
    if arr.shape[0] < int(min_len):
        raise ValueError(
            f"{name} must hold at least {int(min_len)} elements, got {arr.shape[0]}")


def _check_1d(name: str, arr, dtype, device) -> int:
    """Validate a 1-D ``wp.array`` of ``dtype`` on ``device``; return its length."""
    _check_min(name, arr, 0, dtype, device)
    return int(arr.shape[0])


def _check_int(name: str, value, lo: int = _I32_MIN, hi: int = _I32_MAX) -> int:
    """``value`` as a Python int in ``[lo, hi]`` (int32 range by default)."""
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    v = int(value.__index__())
    if v < lo or v > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}")
    return v


def _check_float(name: str, value, *, positive: bool = False) -> float:
    """``value`` as a finite Python float (strictly positive with ``positive=True``)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float, got {type(value).__name__}") from None
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {v}")
    if positive and not v > 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _bytes(arr) -> int:
    return int(arr.shape[0]) * wp.types.type_size_in_bytes(arr.dtype)


def _check_no_alias(name_a: str, a, name_b: str, b) -> None:
    """Raise ``ValueError`` when the 1-D arrays ``a`` and ``b`` share memory."""
    pa, pb = a.ptr, b.ptr
    if not pa or not pb:
        return
    if pa < pb + _bytes(b) and pb < pa + _bytes(a):
        raise ValueError(f"{name_a} must not alias {name_b}")
