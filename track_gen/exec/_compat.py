"""The ONLY track_gen module that touches Warp internals.

Everything here is defensive: a Warp release that renames an internal makes
these helpers degrade (return ``None`` / do nothing) instead of breaking a
launch. Verified against Warp 1.14 and 1.18:

- ``warp._src.context.runtime.captures`` maps a capture id to the
  ``wp.Graph`` Warp registered for an active capture (``wp.launch`` finds the
  graph the same way, by the stream's capture id).
- ``runtime.core.wp_cuda_stream_get_capture_id(handle)`` returns that id.
- ``runtime.core.wp_cuda_stream_synchronize(handle)`` blocks the host on ONE
  stream without the ``device.is_capturing`` guard of
  ``wp.synchronize_stream``, which raises while ANY Warp-registered capture is
  active on the device, from any thread.
- ``runtime.core.wp_cuda_thread_exchange_capture_mode(mode)`` (Warp >= 1.18)
  swaps the calling thread's CUDA stream-capture interaction mode.
- ``Device.captures`` maps the ``wp.Stream`` a Warp capture was started on to
  its ``wp.Graph`` while that capture is open.
- ``wp.Graph`` defines no ``__slots__``.
"""
from __future__ import annotations

__all__ = ["registered_graph", "anchor", "capture_stream", "stream_synchronize"]

_OWNERS_ATTR = "_track_gen_owners"

# cudaStreamCaptureModeRelaxed (== wp.CaptureMode.RELAXED).
_CAPTURE_MODE_RELAXED = 2


def _runtime():
    """Warp's runtime singleton (``None`` before ``wp.init()``)."""
    try:
        from warp._src import context as _wctx  # noqa: PLC0415 - current layout
    except ImportError:  # pragma: no cover - older layout
        import warp.context as _wctx  # noqa: PLC0415
    return getattr(_wctx, "runtime", None)


def registered_graph(stream) -> "object | None":
    """The ``wp.Graph`` Warp registered for ``stream``'s active capture, if any.

    Returns ``None`` when ``stream`` is ``None`` or not capturing, when the
    capture was not started through Warp (e.g. a plain ``torch.cuda.graph``),
    or when a Warp internal is missing.
    """
    if stream is None:
        return None
    try:
        rt = _runtime()
        captures = rt.captures
        if not captures or not stream.is_capturing:
            return None
        return captures.get(rt.core.wp_cuda_stream_get_capture_id(stream.cuda_stream))
    except AttributeError:
        return None


def anchor(stream, *owners) -> None:
    """Keep ``owners`` alive for as long as the Warp graph capturing ``stream``.

    A ``wp.Graph`` retains only the module executables its kernels need, not
    the buffers the recorded launches read and write. Anchoring the producer
    on the graph ties the buffers' lifetime to the caller's graph, so dropping
    the producer while the graph is still replayed cannot free them. A no-op
    for captures Warp did not register (torch graphs: keep the producer alive
    yourself).
    """
    g = registered_graph(stream)
    if g is None:
        return
    try:
        held = g.__dict__.setdefault(_OWNERS_ATTR, [])
    except AttributeError:  # pragma: no cover - a future Graph with __slots__
        return
    for owner in owners:
        if owner is not None and not any(o is owner for o in held):
            held.append(owner)


def capture_stream(device, handle: int) -> "object | None":
    """The ``wp.Stream`` an open Warp capture on ``device`` was started on, if
    its CUDA handle is ``handle``; ``None`` otherwise (or if an internal is
    missing).

    Lets a caller reuse the live wrapper of a capturing handle instead of
    creating a second ``wp.Stream`` for it (see
    :func:`track_gen.exec.context._torch_stream_as_warp`).
    """
    try:
        for s in list(device.captures):
            if int(s.cuda_stream or 0) == handle:
                return s
    except (AttributeError, RuntimeError, TypeError):
        return None
    return None


def stream_synchronize(stream) -> None:
    """Block the host on ``stream`` (``None`` or a capturing stream: no-op).

    Calls the native stream synchronize directly. ``wp.synchronize_stream``
    raises whenever a Warp capture is registered anywhere on the device, e.g.
    another thread's self-capture on its own private stream. A stream
    synchronize is still a "potentially unsafe" call for CUDA while the calling
    thread has an active capture, or while another thread runs a GLOBAL-mode
    capture (``torch.cuda.graph``'s default): it would invalidate that capture.
    Where Warp exposes it (>= 1.18), the calling thread's capture mode is
    switched to RELAXED for the duration of the call, which makes synchronizing
    an unrelated, non-capturing stream legal under any capture.
    """
    if stream is None or stream.is_capturing:
        return
    core = getattr(_runtime(), "core", None)
    sync = getattr(core, "wp_cuda_stream_synchronize", None)
    if sync is None:  # pragma: no cover - internal renamed: fall back to the public API
        import warp as wp  # noqa: PLC0415
        wp.synchronize_stream(stream)
        return
    exchange = getattr(core, "wp_cuda_thread_exchange_capture_mode", None)
    if exchange is None:  # Warp < 1.18
        sync(stream.cuda_stream)
        return
    prev = exchange(_CAPTURE_MODE_RELAXED)
    try:
        sync(stream.cuda_stream)
    finally:
        if prev is not None and prev >= 0:
            exchange(prev)
