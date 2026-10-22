"""The ONLY track_gen module that touches Warp internals.

Everything here is defensive: a Warp release that renames an internal makes
these helpers degrade (return ``None`` / do nothing) instead of breaking a
launch. Verified against Warp 1.18 (the floor):

- ``warp._src.context.runtime.captures`` maps a capture id to the
  ``wp.Graph`` Warp registered for an active capture (``wp.launch`` finds the
  graph the same way, by the stream's capture id).
- ``runtime.core.wp_cuda_stream_get_capture_id(handle)`` returns that id.
- ``runtime.core.wp_cuda_stream_synchronize(handle)`` blocks the host on ONE
  stream without the ``device.is_capturing`` guard of
  ``wp.synchronize_stream``, which raises while ANY Warp-registered capture is
  active on the device, from any thread.
- ``runtime.core.wp_cuda_thread_exchange_capture_mode(mode)``
  swaps the calling thread's CUDA stream-capture interaction mode.
- ``Device.captures`` maps the ``wp.Stream`` a Warp capture was started on to
  its ``wp.Graph`` while that capture is open.
- ``wp.Graph`` defines no ``__slots__``.
- ``Device.captures`` is keyed by stream OBJECT (``wp.Stream`` defines no
  ``__eq__``/``__hash__``): ``wp.capture_if`` / ``capture_end`` look the open
  capture up with the exact object it was started on, so a second wrapper of the
  same handle is not found (:func:`registered_stream` recovers that object).
- ``wp.capture_if`` outside a registered capture reads the condition back on
  the host (pinned copy + ``synchronize_stream``); callers must only use it on a
  registered stream (see :func:`conditional_ok`).
"""
from __future__ import annotations

__all__ = ["registered_graph", "anchor", "capture_stream", "registered_stream",
           "conditional_ok", "conditional_allowed", "set_conditional_allowed",
           "stream_synchronize"]

_OWNERS_ATTR = "_track_gen_owners"
# Set on the wp.Graph of a registration that must not receive conditional nodes
# (track_gen.interop.torch.registered() without conditional=True).
_NO_CONDITIONAL_ATTR = "_track_gen_no_conditional"

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


def registered_stream(stream) -> "object | None":
    """The exact ``wp.Stream`` object Warp registered an open capture under for
    ``stream``'s CUDA handle, or ``None``.

    ``None`` when ``stream`` is ``None``, when no Warp-registered capture is open
    on that handle (e.g. a plain ``torch.cuda.graph``), or when an internal is
    missing. The returned object is the key Warp's capture APIs need
    (``wp.capture_if(..., stream=key)``); it may differ from ``stream`` itself
    when ``stream`` is another wrapper of the same handle.
    """
    if stream is None:
        return None
    try:
        handle = int(stream.cuda_stream or 0)
        device = stream.device
    except (AttributeError, TypeError):
        return None
    if not handle:
        return None
    return capture_stream(device, handle)


def conditional_ok(device) -> bool:
    """Whether ``device`` can record CUDA conditional graph nodes
    (``wp.capture_if`` inside a capture): a CUDA device, with Warp built against
    CUDA Toolkit >= 12.4 and a driver >= 12.4. ``False`` on any error."""
    try:
        if not getattr(device, "is_cuda", False):
            return False
        import warp as wp  # noqa: PLC0415
        return bool(wp.is_conditional_graph_supported())
    except Exception:  # pragma: no cover - defensive: never break a launch over a probe
        return False


def set_conditional_allowed(key, allowed: bool) -> None:
    """Allow or forbid conditional nodes in the open Warp capture registered under
    the stream object ``key`` (a flag on its ``wp.Graph``, which Warp keeps across
    the pause/resume of a ``capture_if``). A no-op if nothing is registered under
    ``key`` or an internal is missing."""
    try:
        g = key.device.captures.get(key)
        if g is not None:
            g.__dict__[_NO_CONDITIONAL_ATTR] = not allowed
    except (AttributeError, TypeError):  # pragma: no cover - internals changed
        return


def conditional_allowed(key) -> bool:
    """Whether the open Warp capture registered under the stream object ``key``
    accepts conditional nodes (``False`` if none is registered, or it was marked
    by :func:`set_conditional_allowed`)."""
    try:
        g = key.device.captures.get(key)
    except (AttributeError, TypeError):
        return False
    return g is not None and not getattr(g, _NO_CONDITIONAL_ATTR, False)


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
