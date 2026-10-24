"""Execution layer: how track_gen emits device work (see :mod:`track_gen.exec.context`).

``LaunchContext`` pins a call to one device and one stream so the work can be recorded
into a caller's CUDA graph (Warp ``ScopedCapture``, ``capture_begin(external=True)``, or
``torch.cuda.graph``) and replayed correctly. ``GraphRunner`` is the execution policy
of one producer (record into the caller's capture / eager / self-graph replay, a
pipeline with HOST stages segment by segment). ``HostOps`` (``ctx.host()``) is how a
HOST stage reaches the host. ``unconditional()`` keeps CUDA conditional nodes out of the
captures recorded in its block. ``RebindAfterCaptureWarning`` flags a bound input replaced
after a capture recorded the method that reads it (roadmap PR14).
"""
from .context import (LaunchContext, conditional, is_capturing, resolve, resolve_stream,
                      unconditional)
from .errors import (CaptureError, LegacyPluginStreamError, NotCapturableError,
                     RebindAfterCaptureWarning)
from .host import HostOps
from .runner import GraphRunner

__all__ = [
    "CaptureError",
    "GraphRunner",
    "HostOps",
    "LaunchContext",
    "LegacyPluginStreamError",
    "NotCapturableError",
    "RebindAfterCaptureWarning",
    "is_capturing",
    "resolve",
    "resolve_stream",
    "unconditional",
    "conditional",
]
