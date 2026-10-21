"""Shared Warp runtime helpers for the whole library.

One home for the pieces every module duplicated: the ``_BIG`` sentinel, the
lazy ``wp.init()`` guard, the debug-only host sync, and the array validator.
``warp_pipeline``, ``warp_gate`` and ``warp_relax`` re-export ``_init`` /
``_sync`` from here, so ONE guard and ONE sync policy cover generators and
utilities alike.

Host syncs: pipeline wrappers, generator bodies and utilities never
host-block (the TrackGenerator / GateGenerator / Course ``generate()``
facades still do on CUDA, around their self-capture). Every launch is
stream-ordered, so results are ready for anything that consumes them on the
same stream, for
``wp.array.numpy()`` (a blocking copy), and for torch on its legacy default
stream (Warp streams are created blocking). A consumer on a different stream
(e.g. a torch side stream) must order itself explicitly, e.g. with
``wp.synchronize_device()`` or a recorded ``wp.Event``. ``_sync`` remains as a
debugging aid only: with ``TRACKGEN_DEBUG_SYNC=1`` it synchronizes the stream
after each library call, so asynchronous CUDA errors surface at the call that
caused them.

``_CAPTURING`` is a deprecated, always-False compatibility name (see
:func:`set_capturing`). ``_CAPTURE_LOCK`` below is shared with the generators
until their self-capture moves to per-generator streams: at most one
self-capture/replay may be in flight process-wide regardless of which family
started it.
"""
from __future__ import annotations

import os
import threading
import warnings

import warp as wp

_BIG = 1.0e30

_INITED = False

# Deprecated: no longer read or written by the library. Kept defined because
# out-of-tree code imports it; set_capturing() no longer changes it.
_CAPTURING = False

# Process-wide mutex serializing every CUDA graph self-capture AND replay
# (TrackGenerator, GateGenerator, Course). Their captures record the device's
# current stream; a concurrent capture, replay, or plain kernel launch from
# another thread lands on that same stream and corrupts the recording (CUDA
# errors 401/900, or an async illegal-memory-access 700 that poisons the
# context). CPU paths have nothing to protect (eager execution);
# Course.generate() still takes the lock uniformly there — uncontended,
# harmless.
_CAPTURE_LOCK = threading.Lock()


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _is_capturing(device, stream: "wp.Stream | None" = None) -> bool:
    """True when ``stream`` (default: ``device``'s current stream) is capturing.

    ``Stream.is_capturing`` sees every capture on that stream (Warp, torch,
    external). ``Device.is_capturing`` adds Warp-registered captures on other
    streams of the device, under which Warp's own ``synchronize_*`` raise.
    Always False on the CPU device.
    """
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return False
    s = stream if stream is not None else dev.stream
    return bool(s.is_capturing or dev.is_capturing)


def _sync(device, stream: "wp.Stream | None" = None) -> None:
    """Debug-only host sync; a no-op unless ``TRACKGEN_DEBUG_SYNC=1``.

    With the variable set, blocks the host on ``stream`` (default: the
    device's current stream) so asynchronous CUDA errors surface at the
    library call that caused them. It never blocks on the CPU device, while
    the stream is capturing, or under a Warp-registered capture on the
    device. A foreign capture on ANOTHER stream (e.g. ``torch.cuda.graph`` on
    a side stream) is not detectable and may be invalidated by the sync, so
    leave the variable unset when capturing that way.
    """
    if os.environ.get("TRACKGEN_DEBUG_SYNC") != "1":
        return
    dev = wp.get_device(device)
    if dev.is_cuda and not _is_capturing(dev, stream):
        wp.synchronize_stream(stream if stream is not None else dev.stream)


def set_capturing(flag: bool) -> None:
    """Deprecated no-op, kept for compatibility. Removed in 1.0.

    Utilities no longer host-synchronize, so ``query()`` / ``update()`` /
    ``reset()`` / ``sample()`` / ``bake()`` / ``refresh()`` and
    ``Course.step()`` / ``reset()`` can be recorded into a CUDA graph without
    any flag.
    """
    warnings.warn(
        "track_gen.set_capturing() is deprecated and has no effect: the "
        "utilities no longer host-synchronize, so they can be recorded into a "
        "CUDA graph without it. It will be removed in 1.0.",
        DeprecationWarning,
        stacklevel=2,
    )


def _check_arr(name: str, arr, shape: tuple, dtype, device) -> None:
    """Validate a wp.array's type, shape, dtype, and device.

    Raises ``ValueError`` naming ``name`` on the first mismatch. Shared body
    behind the family's ``_validate_inputs`` / ``_validate_position`` /
    ``_check_arr`` wrappers; ``shape`` is the expected shape tuple and
    ``device`` the expected (already canonicalized) device string.
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
    if str(arr.device) != str(device):
        raise ValueError(
            f"{name} must be on device {str(device)!r}, got "
            f"{str(arr.device)!r}")
