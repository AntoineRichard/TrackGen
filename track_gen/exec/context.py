"""Launch context: the one path through which track_gen emits device work.

Every launch, copy and fill on a library hot path goes through a
:class:`LaunchContext`, which pins the call to ONE device and ONE stream. That
is what makes the library recordable into a caller's CUDA graph: the work lands
on the stream the caller is capturing, never on whatever Warp's process-global
current stream happens to be.

Stream resolution for a public call (:func:`resolve_stream`), highest priority
first:

1. an explicit ``stream=`` argument (a ``wp.Stream``, or ``"torch"`` for
   torch's current stream on the device);
2. the ambient context set by :meth:`LaunchContext.scope` (used by the
   orchestrators so nested helpers and legacy plugins inherit the stream);
3. a caller-provided default (e.g. a stream bound at construction);
4. torch's current stream when torch is already imported and capturing on
   this device (inside ``torch.cuda.graph(g)`` the capture stream is current,
   so library calls made there are recorded into ``g``);
5. Warp's current stream for the device, evaluated at call time.

On the CPU device the stream is always ``None``.
"""
from __future__ import annotations

import contextlib
import contextvars
import sys
import threading
from typing import Callable, Iterator, Sequence

import warp as wp

from .._src import runtime as _runtime
from . import _compat

__all__ = ["LaunchContext", "resolve_stream", "resolve", "is_capturing"]

_AMBIENT: contextvars.ContextVar["LaunchContext | None"] = contextvars.ContextVar(
    "track_gen_launch_context", default=None)


# One wp.Stream per torch stream handle that is neither Warp's current stream nor
# the stream of an open Warp capture, kept for the life of the process (see
# _torch_stream_as_warp). Bounded by the handles torch hands out: its per-device
# stream pools and graph capture stream, plus any external streams.
_TORCH_WRAPPERS: "dict[tuple[int, int], wp.Stream]" = {}
_TORCH_WRAPPERS_LOCK = threading.Lock()


def _torch_stream_as_warp(dev, torch_stream) -> "wp.Stream":
    """A long-lived ``wp.Stream`` for ``torch_stream`` (never a throwaway wrapper).

    ``wp.stream_from_torch`` registers the handle in Warp's native stream table,
    and the wrapper unregisters it when it is garbage-collected. Unregistering is
    not reference-counted: it erases the handle's capture state and cached event
    for the whole process, and clears Warp's native current stream if it has that
    handle. A wrapper created per call and dropped after it would therefore break
    a Warp capture open on the same handle (``capture_end`` then fails with
    "unknown capture stream") or leave Warp's current stream NULL. So reuse, in
    order: Warp's current stream when it has the handle; the stream an open Warp
    capture on ``dev`` was started on; otherwise one wrapper per handle cached for
    the process (it holds the torch stream, so the handle is never recycled).
    """
    h = int(torch_stream.cuda_stream or 0)
    cur = wp.get_stream(dev)
    if int(cur.cuda_stream or 0) == h:
        return cur
    s = _compat.capture_stream(dev, h)
    if s is not None:
        return s
    key = (int(dev.ordinal), h)
    with _TORCH_WRAPPERS_LOCK:
        s = _TORCH_WRAPPERS.get(key)
        if s is None:
            s = _TORCH_WRAPPERS[key] = wp.stream_from_torch(torch_stream)
    return s


def _torch_capturing(dev) -> "object | None":
    """torch's current stream (a ``torch.cuda.Stream``) on ``dev`` if torch is
    imported AND that stream is capturing in the calling thread; else ``None``."""
    torch = sys.modules.get("torch")
    if torch is None or not dev.is_cuda:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        if torch.cuda.current_device() != dev.ordinal:
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        return torch.cuda.current_stream(dev.ordinal)
    except Exception:  # pragma: no cover - defensive: never let interop break a launch
        return None


def _torch_capturing_stream(dev) -> "wp.Stream | None":
    """torch's current stream on ``dev`` as a ``wp.Stream``, if torch is imported
    AND capturing there (see :func:`_torch_stream_as_warp`)."""
    ts = _torch_capturing(dev)
    if ts is None:
        return None
    try:
        return _torch_stream_as_warp(dev, ts)
    except Exception:  # pragma: no cover - defensive: never let interop break a launch
        return None


def resolve_stream(device, stream=None, default=None) -> "wp.Stream | None":
    """Resolve the single stream a public call runs on (see module docstring).

    Args:
        device: a Warp device or device string.
        stream: ``None``, a ``wp.Stream``, or ``"torch"``.
        default: stream to use when ``stream`` is ``None`` and no ambient
            context applies (e.g. a stream bound at construction).
    """
    dev = wp.get_device(device)
    if stream is not None and not isinstance(stream, (wp.Stream, str)):
        raise TypeError(f"stream must be a wp.Stream, 'torch' or None; got {type(stream).__name__}")
    if isinstance(stream, str) and stream != "torch":
        raise ValueError(f"stream must be a wp.Stream, 'torch' or None; got {stream!r}")
    if not dev.is_cuda:
        return None
    if isinstance(stream, wp.Stream):
        if stream.device.alias != dev.alias:
            raise ValueError(
                f"stream belongs to device {stream.device.alias!r}, not {dev.alias!r}")
        return stream
    if stream == "torch":
        import torch  # noqa: PLC0415 - only when explicitly requested
        return _torch_stream_as_warp(dev, torch.cuda.current_stream(dev.ordinal))
    if default is not None:
        return default
    adopted = _torch_capturing_stream(dev)
    return adopted if adopted is not None else wp.get_stream(dev)


def is_capturing(stream: "wp.Stream | None") -> bool:
    """True when ``stream`` is capturing (any capture: Warp, torch, external)."""
    return bool(stream is not None and stream.is_capturing)


class LaunchContext:
    """One device + one stream; the only way library code emits device work.

    Construct with :meth:`for_call` at a public API boundary and pass it down
    (``ctx=``) to every helper. Helpers accept ``ctx=None`` and resolve it with
    :func:`resolve`, which prefers the ambient context.
    """

    __slots__ = ("device", "stream")

    def __init__(self, device, stream: "wp.Stream | None") -> None:
        self.device = wp.get_device(device)
        self.stream = stream if self.device.is_cuda else None

    @classmethod
    def for_call(cls, device, stream=None, *, default=None) -> "LaunchContext":
        """Context for a public call on ``device``.

        With no explicit ``stream``, an ambient context on the same device (set
        by an enclosing :meth:`scope`) is reused; otherwise the stream is
        resolved by :func:`resolve_stream`.
        """
        dev = wp.get_device(device)
        if stream is None:
            amb = _AMBIENT.get()
            if amb is not None and amb.device.alias == dev.alias:
                return amb
        return cls(dev, resolve_stream(dev, stream, default))

    @classmethod
    def current(cls, device) -> "LaunchContext":
        """Context pinned to Warp's CURRENT stream on ``device`` (no adoption).

        For code that must stay on the stream Warp allocates and zero-fills
        on (e.g. ``PerEnvSeededRNG`` construction, the facades'
        backward-compatible ``_run()``), regardless of ambient or torch state.
        """
        dev = wp.get_device(device)
        return cls(dev, wp.get_stream(dev) if dev.is_cuda else None)

    # -- state ------------------------------------------------------------------
    @property
    def capturing(self) -> bool:
        """Whether this context's stream is capturing. Informational only:
        enqueue code must not branch on it."""
        return is_capturing(self.stream)

    def __repr__(self) -> str:
        s = None if self.stream is None else hex(self.stream.cuda_stream)
        return f"LaunchContext(device={self.device.alias!r}, stream={s})"

    # -- emission ---------------------------------------------------------------
    def launch(self, kernel, dim, inputs: Sequence = (), outputs: Sequence = (), *,
               block_dim: int = 256, max_blocks: int = 0) -> None:
        wp.launch(kernel, dim=dim, inputs=list(inputs), outputs=list(outputs),
                  device=self.device, stream=self.stream,
                  block_dim=block_dim, max_blocks=max_blocks)

    def copy(self, dst: "wp.array", src: "wp.array", *, dst_offset: int = 0,
             src_offset: int = 0, count: int = 0) -> None:
        wp.copy(dst, src, dest_offset=dst_offset, src_offset=src_offset, count=count,
                stream=self.stream)

    def fill(self, arr: "wp.array", value) -> None:
        """Fill ``arr`` with ``value`` on this stream (never ``arr.zero_()``/
        ``fill_()``, whose memset targets Warp's current stream)."""
        from ..ops.fill import fill  # noqa: PLC0415 - avoid import cycle
        fill(self, arr, value)

    def skip_unless(self, flag: "wp.array", body: "Callable[[LaunchContext], None]") -> None:
        """Run ``body(self)``, skipped on device when ``flag[0] == 0`` where that is
        possible; otherwise run it unconditionally.

        CONTRACT: ``body`` must be an exact no-op when ``flag[0] == 0`` (it must
        leave every buffer bitwise unchanged), because whether the skip happens
        depends on the capture, not on the data:

        - inside a capture that Warp registered on this context's stream (Warp
          ``ScopedCapture``, ``capture_begin(external=True)``, or
          ``track_gen.interop.torch.registered(conditional=True)`` under
          ``torch.cuda.graph``) on a device with conditional-node support (CUDA
          Toolkit and driver >= 12.4): ``body`` is recorded under a CUDA
          conditional node that reads ``flag[0]`` at replay time
          (``wp.capture_if``);
        - anywhere else (eager calls, the CPU device, an unregistered
          ``torch.cuda.graph`` capture or one registered without
          ``conditional=True``, a stream that only joined a capture registered on
          another stream, an older driver): ``body`` runs (or is recorded)
          unconditionally.

        A graph holding a conditional node cannot be used as a CUDA child graph
        (e.g. ``wp.capture_if(c, on_true=that_graph)``). ``wp.capture_if`` also
        ends and restarts the stream's capture around the node, which gives it a
        new CUDA capture id: an external capturer that tracks its capture by id
        (torch's caching allocator under ``torch.cuda.graph``) loses track of it
        for the rest of the capture. Hence the opt-in under torch.

        Never takes Warp's eager ``capture_if`` path, which reads the flag back on
        the host and synchronizes the stream.

        Args:
            flag: ``[1]`` ``int32`` array on this context's device (e.g. written by
                :func:`track_gen.ops.any_set` earlier on the same stream).
            body: ``body(ctx)`` emitting its work through ``ctx`` (this context).

        Raises:
            TypeError: ``flag`` is not a ``[1]`` int32 ``wp.array``.
            ValueError: ``flag`` is not on this context's device (``wp.capture_if``
                would miss the capture and read the flag on the host).
        """
        if not isinstance(flag, wp.array) or flag.dtype is not wp.int32 \
                or tuple(flag.shape) != (1,):
            raise TypeError("skip_unless: flag must be a [1] int32 wp.array")
        if str(flag.device) != str(self.device):
            raise ValueError(
                f"skip_unless: flag is on {flag.device}, but the launch context is on "
                f"{self.device}: the flag must be on the context's device")
        if self.capturing and _compat.conditional_ok(self.device):
            key = _compat.registered_stream(self.stream)
            if key is not None and _compat.conditional_allowed(key):
                wp.capture_if(flag, on_true=lambda: body(self), stream=key)
                return
        body(self)

    def sync(self) -> None:
        """Debug-only host sync on this stream (``TRACKGEN_DEBUG_SYNC=1``)."""
        _runtime._sync(self.device, self.stream)

    @contextlib.contextmanager
    def scope(self) -> Iterator["LaunchContext"]:
        """Make this the ambient context for nested ``ctx=None`` helpers and
        legacy plugins (context-variable scoped: thread- and task-local)."""
        token = _AMBIENT.set(self)
        try:
            yield self
        finally:
            _AMBIENT.reset(token)


def resolve(ctx: "LaunchContext | None", device) -> LaunchContext:
    """``ctx`` if given, else :meth:`LaunchContext.for_call` on ``device``."""
    return ctx if ctx is not None else LaunchContext.for_call(device)
