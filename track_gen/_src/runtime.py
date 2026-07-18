"""Shared Warp runtime helpers for the utility family.

One home for the pieces every runtime utility (collision, collision_discs,
props, checkpoints, progress, course) duplicated: the ``_BIG`` sentinel, the
lazy ``wp.init()`` guard, the capture-aware host sync, the single process-wide
capture flag, and the array validator. Consolidating them means ONE flag
governs the whole family and ONE validator defines the error-message contract.

The generator-owned capture flags in ``warp_pipeline`` / ``warp_gate`` are
deliberately separate (pipeline-internal) and are NOT routed through here.
``_CAPTURE_LOCK`` below IS shared with the generators, by design: there is one
CUDA device stream, so at most one graph capture/replay may be in flight
process-wide regardless of which family started it.
"""
from __future__ import annotations

import threading

import warp as wp

_BIG = 1.0e30

_INITED = False
_CAPTURING = False

# Process-wide mutex serializing every CUDA graph capture AND replay (TrackGenerator,
# GateGenerator, Course). Captures record the device's current stream; a concurrent
# capture, replay, or plain kernel launch from another thread lands on that same stream
# and corrupts the recording (CUDA errors 401/900, or an async illegal-memory-access 700
# that poisons the context). The _CAPTURING flags above are also only mutated while this
# lock is held, which makes their save/restore idiom thread-safe. CPU paths have nothing
# to protect (eager execution); Course.generate() still takes the lock uniformly there —
# uncontended, harmless.
_CAPTURE_LOCK = threading.Lock()


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    """Host-block on a cuda device unless a capture is in progress.

    Reads the module-global ``_CAPTURING`` so a single flag suppresses every
    utility's post-launch synchronize during CUDA-graph capture.
    """
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


def set_capturing(flag: bool) -> None:
    """Toggle the family-wide capture flag.

    While ``True``, no utility (collision, discs, props, checkpoints,
    progress, course) performs a host synchronize, so ``query()`` /
    ``update()`` / ``sample()`` / ``bake()`` are safe to record into a
    CUDA graph. The flag is PROCESS-WIDE: exactly one capturing sim loop at
    a time is supported by design. Restore it (``set_capturing(False)`` or to
    the saved previous value) after the capture region.
    """
    global _CAPTURING
    _CAPTURING = bool(flag)


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
