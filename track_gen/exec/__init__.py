"""Execution layer: how track_gen emits device work (see :mod:`track_gen.exec.context`).

``LaunchContext`` pins a call to one device and one stream so the work can be recorded
into a caller's CUDA graph (Warp ``ScopedCapture``, ``capture_begin(external=True)``, or
``torch.cuda.graph``) and replayed correctly. ``GraphRunner`` is the execution policy
of one producer (record into the caller's capture / eager / self-graph replay).
"""
from .context import LaunchContext, is_capturing, resolve, resolve_stream
from .errors import CaptureError, LegacyPluginStreamError, NotCapturableError
from .runner import GraphRunner

__all__ = [
    "CaptureError",
    "GraphRunner",
    "LaunchContext",
    "LegacyPluginStreamError",
    "NotCapturableError",
    "is_capturing",
    "resolve",
    "resolve_stream",
]
