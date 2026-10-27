"""Shared Warp runtime helpers for the whole library.

One home for the pieces every module duplicated: the ``_BIG`` sentinel, the
lazy ``wp.init()`` guard, the debug-only host sync, and the array validator.
``warp_pipeline``, ``warp_gate`` and ``warp_relax`` re-export ``_init`` /
``_sync`` from here, so ONE guard and ONE sync policy cover generators and
utilities alike.

Host syncs: pipeline wrappers, generator bodies and utilities never
host-block (the TrackGenerator / GateGenerator / Course ``generate()``
facades block on their call's stream only with ``sync=True``, their default,
and never while that stream is capturing). Every launch is
stream-ordered, so results are ready for anything that consumes them on the
same stream, for
``wp.array.numpy()`` (a blocking copy), and for torch on its legacy default
stream (Warp streams are created blocking). A consumer on a different stream
(e.g. a torch side stream) must order itself explicitly, e.g. with
``wp.synchronize_device()`` or a recorded ``wp.Event``. ``_sync`` remains as a
debugging aid only: with ``TRACKGEN_DEBUG_SYNC=1`` it synchronizes the stream
after each library call, so asynchronous CUDA errors surface at the call that
caused them.

"""
from __future__ import annotations

import copy
import dataclasses
import os

import warp as wp

# The array validator is the public track_gen.ops.check_array (one body for the ops and
# the facades/utilities); _check_arr stays importable from here under its old name.
from ..ops.check import check_array as _check_arr  # noqa: F401

_BIG = 1.0e30

_INITED = False

def _snapshot_config(config):
    """A deep copy of a config dataclass that SHARES its Warp handles.

    ``copy.deepcopy`` cannot copy a ``wp.Device`` (or ``wp.Stream``): they hold
    ctypes pointers ("ctypes objects containing pointers cannot be pickled").
    Such field values are kept by reference; everything else is copied, so later
    mutation of the original (or of the snapshot) cannot leak across.
    """
    memo = {}
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        for f in dataclasses.fields(config):
            v = getattr(config, f.name, None)
            if isinstance(v, (wp.Device, wp.Stream)):
                memo[id(v)] = v
    return copy.deepcopy(config, memo)


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
