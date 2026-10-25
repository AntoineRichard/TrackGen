"""Host access for HOST stages (:class:`HostOps`, ``ctx.host()``) and the per-slot context.

A HOST stage (``Stage.capture = CaptureClass.HOST``) may read device data back and run
host control flow; it runs eagerly on every call, between the graph launches of the
rest of its pipeline (segmented execution, see :class:`~track_gen.exec.GraphRunner`).
It reaches the host through :meth:`LaunchContext.host() <track_gen.exec.LaunchContext.host>`,
which returns a :class:`HostOps` bound to the call's stream: every copy is
stream-ordered on ``ctx.stream`` and goes through pinned host buffers the stage
allocated at build (:meth:`StageBuilder.host_buffer
<track_gen.plugin.StageBuilder.host_buffer>`), and every wait is a stream synchronize
(never a device-wide one). Nothing is allocated per call.

The Program runs every stage's ``enqueue`` inside a per-slot context (a private context
variable): the stage id, its declared capture class and, for a HOST slot, its build-time
``[1]`` int32 host buffer. ``ctx.host()`` reads it (only a HOST slot gets host
operations), and so do the ``ctx.bounded_loop`` declaration check and its host poll.
"""
from __future__ import annotations

import contextvars
import dataclasses

import numpy as np
import warp as wp

from . import _compat
from .errors import CaptureError

__all__ = ["HostOps"]

# The one sanctioned eager readback of the library: the host poll of a
# ``LaunchContext.bounded_loop`` window (lowering (a)). The conformance kit's host-sync
# guard lets a sync / readback through only when it comes from this function AND the
# window driver set the token below around the call.
SANCTIONED_SITES = frozenset({("track_gen.exec.host", "_poll_i32")})
_POLLING: contextvars.ContextVar[bool] = contextvars.ContextVar("track_gen_polling",
                                                                default=False)
# The host operations of a HOST stage (``HostOps``, reachable only through ``ctx.host()``
# inside a HOST stage's enqueue): the guard lets the syncs and readbacks these methods
# make themselves through (``"HostOps.<m>"`` on Python >= 3.11, ``"<m>"`` before), so a
# HOST stage's OWN syncs, readbacks and allocations are still caught.
_HOSTOPS_METHODS = ("read_i32", "readback", "upload", "sync")
HOSTOPS_SITES = frozenset({("track_gen.exec.host", f"HostOps.{m}")
                           for m in _HOSTOPS_METHODS} |
                          {("track_gen.exec.host", m) for m in _HOSTOPS_METHODS})


@dataclasses.dataclass(frozen=True)
class _SlotInfo:
    """The stage a Program is running (set around each stage's ``enqueue``)."""

    stage_id: str
    label: str                       # e.g. "relax 'acme.smooth' (acme.stages.Smooth)"
    capture: object                  # its CaptureClass; None: a legacy generator spec
    device: str                      # the device alias
    i32: "wp.array | None" = None    # HOST: build-time [1] int32 host buffer (pinned, CUDA)
    # The forced host poll's pinned [1] buffer on CUDA, created on first use (the
    # conformance kit's private _force_lowering("poll") seam only: never on a hot path).
    _lazy: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    @property
    def is_host(self) -> bool:
        return getattr(self.capture, "name", None) == "HOST"

    @property
    def is_static(self) -> bool:
        return getattr(self.capture, "name", None) == "STATIC"


_SLOT: contextvars.ContextVar["_SlotInfo | None"] = contextvars.ContextVar(
    "track_gen_slot", default=None)


class _SlotEnqueue:
    """One stage's ``enqueue`` run inside its slot context (emits nothing itself)."""

    __slots__ = ("info", "fn")

    def __init__(self, info: _SlotInfo, fn) -> None:
        self.info = info
        self.fn = fn

    def __call__(self, ctx) -> None:
        token = _SLOT.set(self.info)
        try:
            self.fn(ctx)
        finally:
            _SLOT.reset(token)

    def __repr__(self) -> str:
        return f"<slot {self.info.stage_id!r}: {self.fn!r}>"


def _current_slot(ctx) -> "_SlotInfo | None":
    """The slot running on ``ctx``'s device, if any."""
    slot = _SLOT.get()
    if slot is None or slot.device != ctx.device.alias:
        return None
    return slot


def _slot_poll_buffer(ctx) -> "wp.array | None":
    """The current slot's ``[1]`` int32 host buffer for a host poll on ``ctx``'s device:
    a HOST slot's build-time one; on CUDA, any other slot's pinned one, allocated on the
    first forced poll (the kit's ``_force_lowering("poll")``; production CUDA never
    polls). ``None`` outside a Program slot."""
    slot = _current_slot(ctx)
    if slot is None:
        return None
    if slot.i32 is not None or not ctx.device.is_cuda:
        return slot.i32
    buf = slot._lazy.get("poll")
    if buf is None:
        buf = slot._lazy["poll"] = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)
    return buf


def _poll_i32(ctx, arr) -> int:
    """``arr[0]`` (a ``[1]`` int32 on ``ctx.device``) on the host: the ONE sanctioned eager
    readback (``bounded_loop`` lowering (a)). Called only by the window driver, inside
    the polling token. CPU: a direct read (CPU launches are synchronous). CUDA: a copy
    on the call's stream into the slot's build-time pinned buffer, then a wait on that
    stream only."""
    if ctx.device.is_cpu:
        return int(arr.numpy()[0])
    buf = _slot_poll_buffer(ctx)
    if buf is None:
        raise CaptureError("bounded_loop: the host-poll lowering on CUDA needs the "
                           "poll buffer of a Program slot (run it inside one)")
    wp.copy(buf, arr, count=1, stream=ctx.stream)
    _compat.stream_synchronize(ctx.stream)
    return int(buf.numpy()[0])


def _dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype))


class HostOps:
    """Host operations of a HOST stage, bound to one call's stream (``ctx.host()``).

    Every copy runs on ``ctx.stream`` through a pinned host buffer the stage allocated
    at build (:meth:`StageBuilder.host_buffer
    <track_gen.plugin.StageBuilder.host_buffer>`), and every wait blocks on that
    stream only, so a HOST stage never allocates per call and never waits on the whole
    device. On the CPU device copies are synchronous and :meth:`sync` does nothing (the
    same code path runs).

    Obtain it with ``h = ctx.host()`` inside the ``enqueue`` of a stage declaring
    ``capture = CaptureClass.HOST``::

        def build(self, b, p):
            seeds_h = b.host_buffer(wp.int32, b.E)
            xy_h = b.host_buffer(wp.vec2f, b.E * p.points)
            xy = b.write("xy", wp.vec2f, per="raw")

            def enqueue(ctx):
                h = ctx.host()
                seeds = h.readback(b.seed, via=seeds_h)        # numpy view of seeds_h
                pts = my_host_algorithm(seeds)                  # [E * points, 2] float32
                h.upload(xy, pts, via=xy_h)
            return enqueue
    """

    __slots__ = ("_ctx", "_slot")

    def __init__(self, ctx, slot: _SlotInfo) -> None:
        if not isinstance(slot, _SlotInfo) or not slot.is_host:
            raise TypeError("HostOps is created by ctx.host() inside a running HOST stage "
                            "(declare capture = CaptureClass.HOST); do not construct it")
        self._ctx = ctx
        self._slot = slot

    def __repr__(self) -> str:
        return f"HostOps(stage={self._slot.stage_id!r}, ctx={self._ctx!r})"

    @property
    def stage_id(self) -> str:
        """The id of the HOST stage these operations belong to."""
        return self._slot.stage_id

    # -- checks -----------------------------------------------------------------------
    def _device_array(self, name: str, arr) -> None:
        if not isinstance(arr, wp.array):
            raise TypeError(f"{name} must be a wp.array, got {type(arr).__name__}")
        if str(arr.device) != str(self._ctx.device):
            raise ValueError(f"{name} is on {arr.device}, but the launch context is on "
                             f"{self._ctx.device}")
        if not arr.is_contiguous:
            raise ValueError(f"{name} must be contiguous")

    def _via(self, via, dtype, n: int) -> None:
        if not isinstance(via, wp.array):
            raise TypeError(f"via must be a wp.array from StageBuilder.host_buffer(), got "
                            f"{type(via).__name__}")
        if not via.device.is_cpu:
            raise ValueError(f"via must be a host buffer (StageBuilder.host_buffer()), got "
                             f"an array on {via.device}")
        if self._ctx.device.is_cuda and not via.pinned:
            raise ValueError("via must be a pinned host buffer on a CUDA device: allocate "
                             "it once, at build, with StageBuilder.host_buffer() (a "
                             "pageable array makes every copy synchronous)")
        if via.dtype is not dtype:
            raise TypeError(f"via has dtype {_dtype_name(via.dtype)}, the device array "
                            f"{_dtype_name(dtype)}")
        if not via.is_contiguous:
            raise ValueError("via must be contiguous")
        if int(via.size) < int(n):
            raise ValueError(f"via holds {int(via.size)} elements, {int(n)} needed")

    # -- operations ---------------------------------------------------------------------
    def sync(self) -> None:
        """Block the host until the work queued on ``ctx.stream`` finished (CPU: no-op).
        A stream synchronize in relaxed capture mode, never a device-wide one."""
        ctx = self._ctx
        if ctx.device.is_cuda:
            _compat.stream_synchronize(ctx.stream)

    def read_i32(self, arr: "wp.array", index: int = 0) -> int:
        """``arr[index]`` of an int32 device array, through the slot's build-time pinned
        ``[1]`` buffer (copy on ``ctx.stream``, then :meth:`sync`).

        Raises:
            TypeError: ``arr`` is not an int32 ``wp.array``, or ``index`` not an int.
            ValueError: ``arr`` on another device, not contiguous, or ``index`` out of
                range.
        """
        self._device_array("arr", arr)
        if arr.dtype is not wp.int32:
            raise TypeError(f"arr must have dtype int32, got {_dtype_name(arr.dtype)}")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"index must be an int, got {type(index).__name__}")
        if not 0 <= int(index) < int(arr.size):
            raise ValueError(f"index {int(index)} is out of range for {int(arr.size)} "
                             f"elements")
        buf = self._slot.i32
        wp.copy(buf, arr, src_offset=int(index), count=1, stream=self._ctx.stream)
        self.sync()
        return int(buf.numpy()[0])

    def readback(self, src: "wp.array", *, via: "wp.array") -> np.ndarray:
        """Copy ``src`` (a device array) into the host buffer ``via`` on ``ctx.stream``,
        wait, and return a numpy VIEW of ``via`` shaped like ``src`` (vector dtypes add a
        trailing component axis). The view is valid until ``via`` is used again.

        Raises:
            TypeError: a non-array, or ``via`` of another dtype.
            ValueError: ``src`` on another device or not contiguous, ``via`` not a host
                buffer (not pinned, on CUDA) or too small.
        """
        self._device_array("src", src)
        n = int(src.size)
        self._via(via, src.dtype, n)
        if n:
            wp.copy(via, src, count=n, stream=self._ctx.stream)
        self.sync()
        host = via.numpy()
        comp = tuple(host.shape[via.ndim:])
        flat = host.reshape((int(via.size),) + comp)
        return flat[:n].reshape(tuple(int(s) for s in src.shape) + comp)

    def upload(self, dst: "wp.array", data, *, via: "wp.array") -> None:
        """Write ``data`` (anything numpy converts to ``via``'s dtype, ``dst.size``
        elements: e.g. an ``[n, 2]`` float array for a ``vec2f`` ``dst``) into the host
        buffer ``via``, copy it to the device array ``dst`` on ``ctx.stream`` and wait,
        so ``via`` is reusable on return.

        Raises:
            TypeError: a non-array, or ``via`` of another dtype.
            ValueError: ``dst`` on another device or not contiguous, ``via`` not a host
                buffer (not pinned, on CUDA) or too small, ``data`` of the wrong size.
        """
        self._device_array("dst", dst)
        n = int(dst.size)
        self._via(via, dst.dtype, n)
        host = via.numpy()
        comp = tuple(host.shape[via.ndim:])
        width = int(np.prod(comp, dtype=np.int64)) if comp else 1
        values = np.asarray(data, dtype=host.dtype)
        if values.size != n * width:
            raise ValueError(f"data holds {values.size} values; dst needs {n} "
                             f"{_dtype_name(dst.dtype)} ({n * width} values)")
        flat = host.reshape((int(via.size),) + comp)
        flat[:n] = values.reshape((n,) + comp)
        if n:
            wp.copy(dst, via, count=n, stream=self._ctx.stream)
        self.sync()
