"""Execution layer: how track_gen emits device work (see :mod:`track_gen.exec.context`).

``LaunchContext`` pins a call to one device and one stream so the work can be recorded
into a caller's CUDA graph (Warp ``ScopedCapture``, ``capture_begin(external=True)``, or
``torch.cuda.graph``) and replayed correctly.
"""
from .context import LaunchContext, is_capturing, resolve, resolve_stream

__all__ = ["LaunchContext", "is_capturing", "resolve", "resolve_stream"]
