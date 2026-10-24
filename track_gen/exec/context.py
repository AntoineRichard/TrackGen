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
import dataclasses
import numbers
import sys
import threading
import warnings
from typing import Callable, Iterator, Sequence

import warp as wp

from .._src import runtime as _runtime
from . import _compat
from . import host as _host
from .errors import CaptureError, NotCapturableError, RebindAfterCaptureWarning

__all__ = ["LaunchContext", "resolve_stream", "resolve", "is_capturing", "unconditional",
           "conditional"]

_AMBIENT: contextvars.ContextVar["LaunchContext | None"] = contextvars.ContextVar(
    "track_gen_launch_context", default=None)

# Set by unconditional(): skip_unless records no conditional node in the block.
_UNCONDITIONAL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "track_gen_unconditional", default=False)
# Set by conditional(): a Warp registration of a capture on torch's current stream that
# nobody marked may record conditional nodes (the caller vouches torch is not capturing).
_CONDITIONAL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "track_gen_conditional", default=False)


@contextlib.contextmanager
def unconditional(enabled: bool = True) -> Iterator[None]:
    """Record no CUDA conditional node in this block (thread- and task-local).

    Inside the block :meth:`LaunchContext.skip_unless` records its body unconditionally
    in every capture, so ``Course.regenerate(mask=...)`` and every
    :meth:`LaunchContext.bounded_loop` window (the ``repulsive`` generator, a plugin's
    ``BOUNDED`` stage) take lowering (c) (every window recorded), with the same
    results. Use it around a recording into a Warp-registered capture
    (``wp.ScopedCapture``, ``wp.capture_begin``) that must not get conditional nodes:

    - a graph you will use as a CUDA child graph (``wp.capture_if(c, on_true=graph)``,
      a graph node of another graph): a graph holding a conditional node cannot be one;
    - a capture whose stream has an outstanding fork (a side stream that joined the
      capture and has not been joined back): ``wp.capture_if`` ends and restarts the
      stream's capture around each node, which CUDA refuses with an unjoined fork
      (``cudaErrorStreamCaptureUnjoined``), invalidating your capture.

    A producer's own graph captured in the block (its first CUDA call outside a
    capture, e.g. ``gen.graph``) is recorded unconditionally too, and stays so. Eager
    calls are unchanged. ``unconditional(False)`` restores the default inside an
    enclosing block.
    """
    token = _UNCONDITIONAL.set(bool(enabled))
    try:
        yield
    finally:
        _UNCONDITIONAL.reset(token)


@contextlib.contextmanager
def conditional(enabled: bool = True) -> Iterator[None]:
    """Let your own Warp capture of torch's current stream record CUDA conditional nodes
    (thread- and task-local).

    :meth:`LaunchContext.skip_unless` (a masked ``Course.regenerate``, every
    :meth:`LaunchContext.bounded_loop` window) records no conditional node in a Warp
    registration nobody opted in whose stream is torch's current stream while that
    stream is capturing: Warp does not record who began the capture, and a hand-made
    ``capture_begin(external=True)`` of a running ``torch.cuda.graph`` must not get one
    (``wp.capture_if`` renews the capture id torch's caching allocator tracks). That
    also catches YOUR OWN ``wp.ScopedCapture`` of torch's current stream (e.g. under
    ``torch.cuda.stream(t)`` + ``wp.ScopedStream(wp.stream_from_torch(t))``), whose
    windows are then recorded unconditionally: same bytes, no device-side skip. Inside
    this block such a capture records conditional nodes; use it only when torch is NOT
    the capturer (under ``torch.cuda.graph`` use
    ``track_gen.interop.torch.registered(conditional=True)`` instead).
    :func:`unconditional` wins over it; the library's own graphs (a producer's
    ``graph``) are unaffected (Warp captures them, marked). ``conditional(False)``
    restores the default inside an enclosing block.
    """
    token = _CONDITIONAL.set(bool(enabled))
    try:
        yield
    finally:
        _CONDITIONAL.reset(token)

# Private seams of ctx.bounded_loop (the conformance kit, GraphRunner.prepare and tests;
# not public API): a forced lowering ("poll" / "unroll"), and the open loop probes.
_LOWERING: contextvars.ContextVar["str | None"] = contextvars.ContextVar(
    "track_gen_bounded_lowering", default=None)
_LOOP_PROBES: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "track_gen_bounded_probes", default=())
_LOWERINGS = (None, "poll", "unroll")


@contextlib.contextmanager
def _force_lowering(mode: "str | None") -> Iterator[None]:
    """Force the lowering of every ``bounded_loop`` in the block: ``"unroll"`` (every
    window runs, no check), ``"poll"`` (the eager host poll, also on CUDA: needs a
    Program slot, whose pinned poll buffer it allocates on first use; refused under
    any capture) or ``None`` (the default choice)."""
    if mode not in _LOWERINGS:
        raise ValueError(f"lowering must be one of {_LOWERINGS}, got {mode!r}")
    token = _LOWERING.set(mode)
    try:
        yield
    finally:
        _LOWERING.reset(token)


@dataclasses.dataclass(frozen=True)
class _LoopRecord:
    """One ``bounded_loop`` call with ``active`` (seen by an open :func:`_probe_loops`)."""

    stage_id: "str | None"       # the Program slot running it (None: no slot)
    n_iters: int
    first_check: int
    check_every: int
    active: "wp.array" = dataclasses.field(repr=False, compare=False)
    lowering: str                # "poll" | "conditional" | "unroll"
    windows: int                 # windows after first_check
    ran: "int | None"            # windows executed (None: decided on device)


@contextlib.contextmanager
def _probe_loops() -> Iterator[list]:
    """Collect a :class:`_LoopRecord` for every ``bounded_loop(active=...)`` call made in
    the block (also inside nested probes)."""
    records: list = []
    token = _LOOP_PROBES.set(_LOOP_PROBES.get() + (records,))
    try:
        yield records
    finally:
        _LOOP_PROBES.reset(token)


def _host_blocked(ctx) -> bool:
    """Whether host work on ``ctx`` would read stale data or be recorded: ``ctx``'s own
    stream is capturing (CUDA), or a CPU (APIC) capture is open on ``ctx``'s device."""
    return ctx.capturing or _compat.cpu_capture_active(ctx.device)


def _bounded_mode(ctx) -> str:
    """The lowering family of one ``bounded_loop`` call, from the capture state only:
    ``"poll"`` (a), ``"unroll"`` (forced (c)) or ``"graph"`` (each window through
    :meth:`LaunchContext.skip_unless`: (b) inside a Warp-registered capture with
    conditional nodes, (c) anywhere else)."""
    forced = _LOWERING.get()
    if forced == "unroll":
        return "unroll"
    if forced == "poll":
        if _host_blocked(ctx) or (ctx.device.is_cuda and (
                _compat.any_registered_capture(ctx.device)
                or _torch_capturing(ctx.device) is not None)):
            raise CaptureError("bounded_loop: the host-poll lowering was forced while a "
                               "capture is open (it is eager-only)")
        if ctx.device.is_cuda and _host._slot_poll_buffer(ctx) is None:
            raise CaptureError("bounded_loop: the host-poll lowering on CUDA needs the "
                               "poll buffer of a Program slot (run it inside one)")
        return "poll"
    if ctx.device.is_cpu and not _compat.cpu_capture_active(ctx.device):
        return "poll"                            # (a): CPU eager, nothing recording
    return "graph"                               # (b) or (c); never a host sync


def _run_range(body, lo: int, hi: int, ctx) -> None:
    for it in range(lo, hi):
        body(ctx, it)


def _run_windows(ctx, mode: str, body, active, n_iters: int, first_check: int,
                 check_every: int) -> "tuple[str, int, int | None]":
    """Run the windows ``[lo, min(lo + check_every, n_iters))``, ``lo = first_check,
    first_check + check_every, ...`` under ``mode``; returns ``(lowering, windows,
    ran)``."""
    bounds = [(lo, min(lo + check_every, n_iters))
              for lo in range(first_check, n_iters, check_every)]
    if mode == "poll":
        ran = 0
        for lo, hi in bounds:
            token = _host._POLLING.set(True)
            try:
                value = _host._poll_i32(ctx, active)
            finally:
                _host._POLLING.reset(token)
            if value == 0:
                break
            _run_range(body, lo, hi, ctx)
            ran += 1
        return "poll", len(bounds), ran
    if mode == "unroll":
        for lo, hi in bounds:
            _run_range(body, lo, hi, ctx)
        return "unroll", len(bounds), len(bounds)
    conditional = False
    for lo, hi in bounds:
        recorded = ctx.skip_unless(
            active, lambda c, _lo=lo, _hi=hi: _run_range(body, _lo, _hi, c))
        conditional = conditional or recorded
    if conditional:
        return "conditional", len(bounds), None
    return "unroll", len(bounds), len(bounds)


def _check_count(name: str, value, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"bounded_loop: {name} must be an int, got {type(value).__name__}")
    if int(value) < low:
        raise ValueError(f"bounded_loop: {name} must be >= {low}, got {int(value)}")
    return int(value)


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

    Besides launches, copies and fills it offers device-side control flow that stays
    recordable into any capture (:meth:`skip_unless`, :meth:`bounded_loop`) and, inside
    a HOST stage only, host operations on its stream (:meth:`host`).
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

    def skip_unless(self, flag: "wp.array", body: "Callable[[LaunchContext], None]") -> bool:
        """Run ``body(self)``, skipped on device when ``flag[0] == 0`` where that is
        possible; otherwise run it unconditionally. Returns ``True`` when ``body`` was
        recorded under a conditional node, ``False`` when it ran (or was recorded)
        unconditionally.

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
          ``conditional=True``, an unmarked Warp registration whose stream is torch's
          current, capturing stream in this thread (a hand-made
          ``capture_begin(external=True)`` of torch's capture, or your own
          ``ScopedCapture`` of torch's current stream unless inside
          :func:`conditional`), a stream that only joined a capture registered on another
          stream,
          an older driver, inside :func:`unconditional`): ``body`` runs (or is
          recorded) unconditionally.

        A graph holding a conditional node cannot be used as a CUDA child graph
        (e.g. ``wp.capture_if(c, on_true=that_graph)``). ``wp.capture_if`` also
        ends and restarts the stream's capture around the node, which CUDA refuses
        while a side stream that joined the capture has not been joined back (an
        unjoined fork invalidates the capture), and which gives the capture a new CUDA
        capture id: an external capturer that tracks its capture by id (torch's
        caching allocator under ``torch.cuda.graph``) loses track of it for the rest
        of the capture. Hence the opt-in under torch, and :func:`unconditional` to opt
        out of a Warp-registered capture.

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
        self._check_flag("skip_unless", "flag", flag)
        key = self._conditional_key()
        if key is not None:
            wp.capture_if(flag, on_true=lambda: body(self), stream=key)
            return True
        body(self)
        return False

    def _conditional_key(self):
        """The registered stream object under which :meth:`skip_unless` records a
        conditional node, or ``None`` (record unconditionally)."""
        if _UNCONDITIONAL.get() or not self.capturing or \
                not _compat.conditional_ok(self.device):
            return None
        key = _compat.registered_stream(self.stream)
        if key is None or not _compat.conditional_allowed(key):
            return None
        if _compat.conditional_marked(key) is None and not _CONDITIONAL.get():
            # A registration nobody opted in (a Warp ScopedCapture / capture_begin): when
            # its stream is torch's current, capturing stream it may be torch's running
            # capture (a hand-made capture_begin(external=True)), and Warp does not record
            # who began it: keep torch's capture id intact, as registered() does by
            # default. conditional() vouches it is not; the library's own self-captures
            # are marked allowed (GraphRunner._capture_once).
            ts = _torch_capturing(self.device)
            if ts is not None and \
                    int(getattr(ts, "cuda_stream", 0) or 0) == int(key.cuda_stream or 0):
                return None
        return key

    def _check_flag(self, what: str, name: str, flag) -> None:
        """``flag`` is a ``[1]`` int32 ``wp.array`` on this context's device."""
        if not isinstance(flag, wp.array) or flag.dtype is not wp.int32 \
                or tuple(flag.shape) != (1,):
            raise TypeError(f"{what}: {name} must be a [1] int32 wp.array")
        if str(flag.device) != str(self.device):
            raise ValueError(
                f"{what}: {name} is on {flag.device}, but the launch context is on "
                f"{self.device}: the {name} must be on the context's device")

    def bounded_loop(self, n_iters: int, body: "Callable[[LaunchContext, int], None]", *,
                     active: "wp.array | None" = None, first_check: int = 0,
                     check_every: int = 1) -> None:
        """Run ``body(ctx, it)`` for ``it in range(n_iters)`` (``it`` a static Python int),
        stopping early once ``active[0] == 0`` where the capture state allows it.

        Iterations ``[0, min(first_check, n_iters))`` always run. Then come the windows
        ``[lo, min(lo + check_every, n_iters))`` for ``lo = first_check, first_check +
        check_every, ...``: each is preceded by a check of ``active[0]`` and may be
        skipped, with every later one, once that check reads 0. Without ``active``
        every iteration runs (a plain unrolled loop).

        How a window check is lowered depends on where the call runs, never on data:

        - (a) a host poll (one 4-byte read) that stops the loop: the CPU device, eager,
          with no CPU graph capture open on it;
        - (b) a CUDA conditional node per window (``wp.capture_if``, via
          :meth:`skip_unless`): inside a Warp-registered capture of this context's
          stream with conditional-node support (CUDA >= 12.4), e.g. a producer's own
          graph, a Warp ``ScopedCapture`` or ``track_gen.interop.torch.registered(
          conditional=True)``;
        - (c) every window runs (or is recorded) unconditionally: eager calls on CUDA
          (they never host-sync), plain ``torch.cuda.graph`` captures, older drivers,
          CPU graph captures, recordings inside :func:`unconditional`.

        CONTRACT (checked by the conformance kit, ``bounded.absorbing``): ``active`` is
        a ``[1]`` int32 array on this context's device that the stage writes on the
        device before the first check (its ``enqueue``, before the loop, when
        ``first_check == 0``; otherwise usually the iteration ``lo - 1`` that precedes
        each window), in every generation; once a check reads ``active[0] == 0``, every
        remaining iteration leaves everything read after the loop bitwise unchanged (an
        ABSORBING freeze); the body emits only through the context it is given and
        mutates no Python state read after the loop. The three lowerings then give
        bitwise identical results; how many iterations actually execute is
        unspecified (``n_iters`` is a bound). A stage whose loop passes ``active``
        declares ``capture = CaptureClass.BOUNDED``.

        Under lowering (b) each window is the body graph of a CUDA conditional node,
        which may hold kernel launches, memsets and device-to-device copies only: a
        window may launch kernels and ``ctx.copy`` / ``ctx.fill`` between device arrays
        on ``ctx``'s stream, and nothing else (no copy to or from host or pinned memory,
        no event, no other stream). Anything else fails inside ``wp.capture_if`` while
        the caller's capture is being recorded. ``check_stage(..., devices=["cuda:0"])``
        (``bounded.lowering``) records every window that way.

        Args:
            n_iters: The iteration bound (``>= 0``).
            body: ``body(ctx, it)``, emitting its work through ``ctx``.
            active: ``[1]`` int32 flag on this context's device (``None``: no checks).
            first_check: The first iteration that may be skipped (``>= 0``).
            check_every: Iterations per window (``>= 1``).

        Raises:
            TypeError: a non-int count (``bool`` included), or ``active`` not a ``[1]``
                int32 ``wp.array``.
            ValueError: a negative count, ``check_every < 1``, ``active`` on another
                device, or ``active`` passed inside a stage that declares
                ``capture = CaptureClass.STATIC``.
        """
        n_iters = _check_count("n_iters", n_iters, 0)
        first_check = _check_count("first_check", first_check, 0)
        check_every = _check_count("check_every", check_every, 1)
        slot = None
        if active is not None:
            self._check_flag("bounded_loop", "active", active)
            slot = _host._current_slot(self)
            if slot is not None and slot.is_static:
                raise ValueError(
                    f"{slot.label} calls ctx.bounded_loop(active=...) but declares "
                    f"capture=CaptureClass.STATIC: declare capture = CaptureClass.BOUNDED "
                    f"(its loop may stop early; the conformance kit checks that with "
                    f"bounded.absorbing)")
        pre = min(first_check, n_iters)
        _run_range(body, 0, pre, self)
        if pre == n_iters and active is None:
            return
        if active is None:
            _run_range(body, pre, n_iters, self)
            return
        if pre == n_iters:
            lowering, windows, ran = "unroll", 0, 0
        else:
            mode = _bounded_mode(self)
            lowering, windows, ran = _run_windows(self, mode, body, active, n_iters,
                                                  first_check, check_every)
        probes = _LOOP_PROBES.get()
        if probes:
            record = _LoopRecord(None if slot is None else slot.stage_id, n_iters,
                                 first_check, check_every, active, lowering, windows, ran)
            for records in probes:
                records.append(record)

    def host(self) -> "_host.HostOps":
        """The :class:`~track_gen.exec.HostOps` of the HOST stage running on this
        context: readbacks, uploads and waits on this context's stream through the
        stage's build-time pinned host buffers.

        Raises:
            CaptureError: not inside the ``enqueue`` of a pipeline stage on this device.
            ValueError: inside the ``enqueue`` of a stage that does not declare
                ``capture = CaptureClass.HOST`` (the error a wrong declared capture class
                gives, as for ``bounded_loop(active=...)`` in a STATIC stage and
                ``StageBuilder.host_buffer``), or of a legacy generator spec.
            NotCapturableError: this context's stream is capturing, or a CPU graph
                capture is open on the device (host work would be recorded, or read
                stale data).
        """
        slot = _host._current_slot(self)
        if slot is None:
            raise CaptureError("ctx.host() is available only inside a HOST stage's "
                               "enqueue (Stage.capture = CaptureClass.HOST)")
        if not slot.is_host:
            if slot.capture is None:
                raise ValueError(
                    f"{slot.label} calls ctx.host(), which a legacy GeneratorSpec cannot "
                    f"use: port it to a Stage declaring capture = CaptureClass.HOST")
            raise ValueError(
                f"{slot.label} calls ctx.host() but declares capture=CaptureClass."
                f"{slot.capture.name}: declare capture = CaptureClass.HOST (it then runs "
                f"eagerly, outside every capture)")
        if _host_blocked(self):
            raise NotCapturableError(slot.label, stage_ids=(slot.stage_id,))
        return _host.HostOps(self, slot)

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


def _call_context(device, stream=None, ctx: "LaunchContext | None" = None
                  ) -> LaunchContext:
    """The context of a utility's hot call (roadmap PR14): ``ctx`` when given (it must be
    a :class:`LaunchContext` on ``device``), else ``LaunchContext.for_call(device,
    stream)``.

    Raises:
        TypeError: both ``stream=`` and ``ctx=`` are given, or ``ctx`` is not a
            ``LaunchContext``.
        ValueError: ``ctx`` is on another device.
    """
    if ctx is None:
        return LaunchContext.for_call(device, stream)
    if stream is not None:
        raise TypeError("pass stream= or ctx=, not both (ctx= already pins the stream)")
    if not isinstance(ctx, LaunchContext):
        raise TypeError(f"ctx must be a track_gen.exec.LaunchContext, got "
                        f"{type(ctx).__name__}")
    dev = wp.get_device(device)
    if ctx.device.alias != dev.alias:
        raise ValueError(f"ctx is on {ctx.device.alias!r}, but this utility's arrays are "
                         f"on {dev.alias!r}")
    return ctx


# -- the rebind-after-capture warning (roadmap PR14, a documented G1 exception) ----------
#
# A utility with bound inputs keeps ``_recorded`` (False until one of its hot methods that
# reads them ran while ``_host_blocked(ctx)``: the stream was capturing, or a CPU graph
# capture was open). The probe changes no emission; once armed it stops probing. A bind
# that replaces a pointer while armed warns (once per public call: nested binds report to
# the outermost one through _REBIND_HITS) and clears the flag.
_REBIND_HITS: contextvars.ContextVar["list | None"] = contextvars.ContextVar(
    "track_gen_rebind_hits", default=None)


def _probe_recorded(owner, ctx) -> None:
    """Arm ``owner._recorded`` when ``ctx`` is recording (one probe per call until armed;
    host flag only: no emission depends on it)."""
    if not owner._recorded and _host_blocked(ctx):
        owner._recorded = True


def _rebind_open() -> "tuple[object, list]":
    """Open a rebind scope: ``(token, hits)``; ``token`` is ``None`` inside an enclosing
    scope (whose owner warns)."""
    hits = _REBIND_HITS.get()
    if hits is not None:
        return None, hits
    hits = []
    return _REBIND_HITS.set(hits), hits


def _rebind_close(token, hits: list, stacklevel: int) -> None:
    """Close a scope opened by :func:`_rebind_open`; the outermost one warns once (at
    ``stacklevel`` counted from its caller's frame)."""
    if token is None:
        return
    _REBIND_HITS.reset(token)
    if hits:
        warnings.warn(_rebind_text(*hits[0]), RebindAfterCaptureWarning,
                      stacklevel=stacklevel + 1)


def _rebind_text(owner: str, method: str) -> str:
    return (f"{owner}: rebinding after {method}() was recorded into a capture: that graph "
            f"keeps reading the arrays bound when it was recorded; record it again after "
            f"rebinding (this warning also fires once on that recipe; filter "
            f"track_gen.exec.RebindAfterCaptureWarning if you re-record)")


def _note_rebind(owner, old, new, method: str, stacklevel: int) -> None:
    """A bind of ``owner`` replaces the arrays ``old`` (a tuple, or ``None``: first bind)
    with ``new``: when ``owner._recorded`` is armed and a pointer changes, report it
    (warn at ``stacklevel`` from the caller of the bind, or to an enclosing scope) and
    clear the flag."""
    if not getattr(owner, "_recorded", False) or old is None:
        return
    if all(int(a.ptr) == int(b.ptr) for a, b in zip(old, new)):
        return
    owner._recorded = False
    hits = _REBIND_HITS.get()
    item = (type(owner).__name__, method)
    if hits is not None:
        hits.append(item)
        return
    warnings.warn(_rebind_text(*item), RebindAfterCaptureWarning, stacklevel=stacklevel + 1)
