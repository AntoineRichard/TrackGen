"""``GraphRunner``: the execution policy for one producer.

A producer (``TrackGenerator``, ``GateGenerator``, ``Course``) owns ONE
enqueue body ``enqueue(ctx)`` that emits all of its device work through a
:class:`~track_gen.exec.LaunchContext`: no allocation, no host sync, no
branch on device data or on ``ctx.capturing``. The runner decides how that
body runs for each call:

==================  ====================================================
RECORD              the resolved stream is capturing (Warp
                    ``ScopedCapture``, ``capture_begin(external=True)``,
                    ``torch.cuda.graph``): the body is enqueued on that
                    stream so it is recorded into the CALLER's graph. No
                    self-capture, no ``wp.capture_launch`` (a graph launch
                    cannot itself be recorded), no sync.
EAGER               CPU, ``policy="eager"``, or a non-capturable producer
                    outside any capture: the body runs on the resolved
                    stream.
SELF-GRAPH          CUDA outside any capture: the body is captured ONCE
                    per cache key (``run(key=...)``; ``None`` ->
                    :attr:`GraphRunner.graph`; at most
                    ``MAX_KEYED_GRAPHS`` other keys, further ones run
                    eagerly) on a private stream, then every call with
                    that key replays it with
                    ``wp.capture_launch(graph, stream=resolved)``.
SEGMENTED           CUDA outside any capture, for a body made of parts
                    (a pipeline with HOST stages, ``Program.segments``):
                    every part that is not HOST is captured once per key
                    (its own graph), then each call runs the HOST parts
                    eagerly and replays the other parts' graphs, in
                    order, on the resolved stream. Such a producer
                    refuses a caller's capture (it is not capturable).
==================  ====================================================

The EAGER row never host-syncs, except inside the HOST stages of a body that has them
(their ``ctx.host()`` readbacks, uploads and waits on the call's stream): an iterative
(BOUNDED) stage runs its full iteration bound there on CUDA
(``LaunchContext.bounded_loop``); its self-graph skips the converged windows on device.

How many times the Python body runs (plugin authors: bodies must be pure
enqueues, see G1 in the target architecture):

==========================  ==================  =================
mode                        Python body runs    kernels execute
==========================  ==================  =================
``prepare()``               once                once
self-graph capture (once)   once                not at record time
each self-graph replay      0 times             once
eager call                  once                once
caller's recording          once                per caller replay
==========================  ==================  =================

Buffer contents persist across replays; host scalars and Python state freeze
at record time.

"""
from __future__ import annotations

import dataclasses
import threading
import warnings
import weakref
from typing import Callable

import warp as wp

from . import _compat
from .context import LaunchContext, _force_lowering, _torch_capturing
from .errors import CaptureError, NotCapturableError

__all__ = ["GraphRunner", "ensure_not_capturing"]

_POLICIES = ("auto", "eager")

# At most this many self-graphs per runner for non-None keys (e.g. one per mask
# pointer). Calls with a further new key run eagerly (one warning per runner), so a
# caller passing a fresh mask array per call cannot grow the cache without bound.
MAX_KEYED_GRAPHS = 16

@dataclasses.dataclass(frozen=True)
class _BodyPart:
    """One part of a segmented body (a ``Program`` segment)."""

    enqueue: Callable
    host: bool = False          # run eagerly on every call (a HOST segment)


class _SegmentedBody(tuple):
    """A body made of :class:`_BodyPart` s; calling it runs every part in order on the
    same context (eager runs, ``prepare``)."""

    __slots__ = ()

    def __new__(cls, parts=()):
        return super().__new__(cls, tuple(parts))

    def __call__(self, ctx) -> None:
        for part in self:
            part.enqueue(ctx)


def _is_current(stream, dev) -> bool:
    """Whether ``stream`` has the handle of Warp's current stream on ``dev``."""
    return stream is not None and \
        int(stream.cuda_stream or 0) == int(wp.get_stream(dev).cuda_stream or 0)


def ensure_not_capturing(device, *, default_stream=None, label: str = "") -> LaunchContext:
    """Raise :class:`~track_gen.exec.CaptureError` if construction-time work on
    ``device`` would land in a capture; return the resolved context otherwise.

    Construction allocates and zero-fills on Warp's CURRENT stream and runs a
    real warm-up pass on the resolved stream, so it is refused when either of
    them is capturing, when torch is capturing in the calling thread, or (CPU) when
    a CPU graph capture (``wp.ScopedCapture("cpu")``) is open on the device: it
    would record the warm-up instead of running it. Call it BEFORE the first
    allocation, so a refused construction leaves the caller's capture untouched.
    """
    dev = wp.get_device(device)
    ctx = LaunchContext.for_call(dev, None, default=default_stream)
    capturing = ctx.capturing
    if not capturing and dev.is_cuda:
        capturing = wp.get_stream(dev).is_capturing or _torch_capturing(dev) is not None
    elif not capturing:
        capturing = _compat.cpu_capture_active(dev)
    if capturing:
        what = repr(label) if label else "this producer"
        kind = "CUDA graph capture" if dev.is_cuda else \
            'CPU graph capture (wp.ScopedCapture("cpu"), which records launches without ' \
            'running them)'
        raise CaptureError(
            f"{what} must be constructed (prepared) outside any {kind}: "
            "construction allocates on Warp's current stream and runs one real warm-up "
            "pass. Construct it before beginning the capture.")
    return ctx


def _check_default_stream(device, default_stream) -> None:
    """The ``default_stream`` rules of :class:`GraphRunner` (host-only): ``None`` or a
    ``wp.Stream`` of ``device`` (a CPU device accepts any ``wp.Stream`` and ignores
    it). Raises ``TypeError`` / ``ValueError``."""
    if default_stream is None:
        return
    if not isinstance(default_stream, wp.Stream):
        raise TypeError(
            "stream must be a wp.Stream or None; got "
            f"{type(default_stream).__name__}")
    dev = wp.get_device(device)
    if dev.is_cuda and default_stream.device.alias != dev.alias:
        raise ValueError(
            f"stream belongs to device {default_stream.device.alias!r}, "
            f"not {dev.alias!r}")


class GraphRunner:
    """Execution policy (record / eager / self-graph) for one producer.

    Args:
        device: the producer's Warp device.
        capturable: whether the enqueue body obeys the graph-capture contract.
            ``False`` runs eagerly outside a capture and raises
            :class:`~track_gen.exec.NotCapturableError` (before any launch)
            inside one.
        default_stream: stream used when a call passes none and no ambient
            context applies (see :func:`~track_gen.exec.resolve_stream`).
        owner: object anchored on a caller's Warp graph when the body is
            recorded into it, so the buffers the graph references outlive
            the producer. Held weakly by the runner.
        name: label used in error messages.

    A body passed to :meth:`run` may also be a segmented body (one part per
    ``Program`` segment, built by the Program): see the SEGMENTED row of the module
    docstring.

    A runner is single-writer: running the same producer concurrently from
    two threads is a race on its output buffers (use one producer per
    concurrent caller). Different producers are independent: each self-graph
    is captured on the producer's own private stream, so no process-wide lock
    is taken.
    """

    def __init__(self, device, *, capturable: bool,
                 default_stream: "wp.Stream | None" = None, owner=None,
                 name: str = "", _host_ids=()) -> None:
        dev = wp.get_device(device)
        _check_default_stream(dev, default_stream)
        self._device = dev
        self._capturable = bool(capturable)
        self._host_ids = tuple(str(sid) for sid in _host_ids)
        self._default_stream = default_stream if dev.is_cuda else None
        self._name = str(name)
        if owner is None:
            self._owner_ref = None
        else:
            try:
                self._owner_ref = weakref.ref(owner)
            except TypeError:  # not weak-referenceable: hold it strongly
                self._owner_ref = lambda _o=owner: _o
        # Self-graphs by key (``None`` = the producer's default body). One graph per
        # key, captured on first use and replayed by every later call with that key.
        self._graphs: "dict[object, wp.Graph]" = {}
        # Segmented bodies by key: one graph per part (None at a HOST part).
        self._seg_graphs: "dict[object, tuple]" = {}
        self._keyed_limit_warned = False
        self._capture_stream: "wp.Stream | None" = None
        self._prepared = False
        self._lock = threading.RLock()

    # -- state ------------------------------------------------------------------
    @property
    def graph(self) -> "wp.Graph | None":
        """The default-key self-graph, once captured (first CUDA call with
        ``key=None`` outside a capture); ``None`` on CPU, for a non-capturable
        producer (also a segmented one: its parts have one graph each), or before
        that call."""
        return self._graphs.get(None)

    def graph_for(self, key=None) -> "wp.Graph | None":
        """The whole-body self-graph cached for ``key`` (see :meth:`run`), if captured;
        ``None`` for the key of a segmented body (see :meth:`segment_graphs_for`)."""
        return self._graphs.get(key)

    @property
    def graph_keys(self) -> tuple:
        """The keys of every self-graph captured so far (``None`` = default),
        segmented bodies included (their graphs: :meth:`segment_graphs_for`)."""
        return tuple(self._graphs) + tuple(k for k in self._seg_graphs
                                           if k not in self._graphs)

    def segment_graphs_for(self, key=None) -> "tuple | None":
        """The per-part self-graphs of the segmented body cached for ``key``, one per
        part in order (``None`` at a HOST part), or ``None`` before they were captured
        (and for a body that is not segmented)."""
        return self._seg_graphs.get(key)

    @property
    def device(self):
        """The producer's Warp device (a ``warp.context.Device``)."""
        return self._device

    @property
    def capturable(self) -> bool:
        """Whether the body may be recorded into a caller's capture (the constructor's
        ``capturable``); ``False`` runs eagerly and raises
        :class:`~track_gen.exec.NotCapturableError` inside a capture."""
        return self._capturable


    @property
    def default_stream(self) -> "wp.Stream | None":
        """The stream used when a call passes none and no ambient context applies
        (``None`` on CPU, or when the constructor got none)."""
        return self._default_stream

    @property
    def name(self) -> str:
        """The label used in error messages (the constructor's ``name``)."""
        return self._name

    @property
    def prepared(self) -> bool:
        """Whether :meth:`prepare` ran (one real pass outside any capture)."""
        return self._prepared

    def context(self, stream=None) -> LaunchContext:
        """The launch context of one call (explicit ``stream`` > ambient >
        ``default_stream`` > torch capture stream > Warp's current stream)."""
        return LaunchContext.for_call(self._device, stream, default=self._default_stream)

    def ensure_not_capturing(self) -> LaunchContext:
        """:func:`ensure_not_capturing` for this runner's device and default
        stream: call it before the producer's first allocation."""
        return ensure_not_capturing(self._device, default_stream=self._default_stream,
                                    label=self._name)


    # -- execution --------------------------------------------------------------
    def prepare(self, enqueue) -> None:
        """Run one real warm-up outside capture, loading every hot-path kernel.

        Wait on allocations made on Warp's current stream, then synchronize the
        resolved stream. Bounded loops use every window during warm-up.
        """
        ctx = self.context()
        ensure_not_capturing(self._device, default_stream=self._default_stream,
                             label=self._name)
        if ctx.stream is not None and not _is_current(ctx.stream, self._device):
            ctx.stream.wait_stream(wp.get_stream(self._device))
        with _force_lowering("unroll"):
            self._run_eager(enqueue, ctx)
        self._sync(ctx)
        self._prepared = True

    def run(self, enqueue, *, stream=None, policy: str = "auto",
            sync: bool = False, key=None) -> LaunchContext:
        """Run the body on the resolved stream under record/eager/self-graph policy.

        ``key`` identifies a stable body and its buffer pointers. Non-default keys
        require prior preparation; at most MAX_KEYED_GRAPHS are cached. A segmented
        body runs HOST parts eagerly and replays the capturable parts in order.
        """
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
        ctx = self.context(stream)
        segmented = isinstance(enqueue, _SegmentedBody)
        self.check_record(ctx)
        if ctx.capturing:
            with ctx.scope():
                enqueue(ctx)
            owner = self._owner()
            if owner is not None:
                _compat.anchor(ctx.stream, owner)
            return ctx
        if not self._device.is_cuda or policy == "eager" \
                or (not self._capturable and not segmented):
            self._run_eager(enqueue, ctx)
        elif segmented:
            self._run_segmented(enqueue, ctx, key)
        else:
            graph = self._graphs.get(key)
            if graph is None and key is not None and self._keyed_full():
                self._run_eager(enqueue, ctx)
            else:
                if graph is None:
                    graph = self._capture(enqueue, key)
                wp.capture_launch(graph, stream=ctx.stream)
        if sync:
            self._sync(ctx)
        return ctx

    def _run_segmented(self, body: "_SegmentedBody", ctx: LaunchContext, key) -> None:
        """Replay capturable parts and run HOST parts on the call's stream."""
        graphs = self._seg_graphs.get(key)
        if graphs is None and key is not None and self._keyed_full():
            self._run_eager(body, ctx)
            return
        if graphs is None:
            graphs = self._capture_parts(body, key)
        for part, graph in zip(body, graphs):
            if part.host:
                self._run_eager(part.enqueue, ctx)
            else:
                wp.capture_launch(graph, stream=ctx.stream)

    def check_record(self, ctx: LaunchContext) -> None:
        """Raise where :meth:`run` would refuse ``ctx`` before any launch: ``ctx`` is
        capturing and this producer cannot be recorded there, or a CPU graph capture
        (``wp.ScopedCapture("cpu")``) is open on the device and the producer is not
        capturable. A no-op otherwise. Lets a caller validate before issuing its own
        launches ahead of :meth:`run` (e.g. a reseed).

        Raises:
            NotCapturableError: the producer is not capturable (it names its HOST
                stages).
        """
        if not self._capturable and _compat.cpu_capture_active(self._device):
            # A CPU graph capture records every CPU launch without executing it: a
            # non-capturable body's host work would read stale data into it.
            raise NotCapturableError(self._name, stage_ids=self._host_ids)
        if not ctx.capturing:
            return
        if not self._capturable:
            raise NotCapturableError(self._name, stage_ids=self._host_ids)

    # -- internals --------------------------------------------------------------
    def _label(self) -> str:
        return repr(self._name) if self._name else "this producer"

    def _owner(self):
        return None if self._owner_ref is None else self._owner_ref()

    def _keyed_full(self) -> bool:
        """Whether the keyed self-graph cache is full (warns once when it is)."""
        keys = set(self._graphs) | set(self._seg_graphs)
        if sum(k is not None for k in keys) < MAX_KEYED_GRAPHS:
            return False
        if not self._keyed_limit_warned:
            self._keyed_limit_warned = True
            warnings.warn(
                f"{self._label()}: {MAX_KEYED_GRAPHS} self-graphs are cached for "
                "distinct keys (e.g. mask arrays); calls with a new key now run eagerly. "
                "Pass a stable mask array (e.g. inputs['regen_mask']) instead of a new "
                "one per call.", RuntimeWarning, stacklevel=4)
        return True

    def _sync(self, ctx: LaunchContext) -> None:
        """Block the host on ``ctx.stream`` unless that would touch a capture."""
        if ctx.stream is None or ctx.capturing:
            return
        if _torch_capturing(self._device) is not None:
            # This thread is inside torch.cuda.graph (the call resolved another
            # stream): a host sync here could invalidate torch's capture.
            return
        _compat.stream_synchronize(ctx.stream)


    def _run_eager(self, enqueue, ctx: LaunchContext) -> None:
        """Run the body on the resolved launch context."""
        with ctx.scope():
            enqueue(ctx)


    def _capture(self, enqueue, key=None) -> "wp.Graph":
        """Capture the self-graph of ``key`` once (thread-safe; retried once on
        failure)."""
        if not self._prepared:
            if key is not None:
                raise RuntimeError(
                    f"{self._label()}: prepare() must run (with the default body) before "
                    f"a self-graph is captured for key {key!r}")
            self.prepare(enqueue)
        with self._lock:
            cached = self._graphs.get(key)
            if cached is not None:
                return cached
            try:
                graph = self._capture_once(enqueue)
            except CaptureError:
                raise
            except RuntimeError:
                # A capture on a blocking stream is invalidated when another thread
                # touches the legacy default stream (e.g. torch on stream 0) during
                # the one-time capture window. The failed capture was ended by
                # ScopedCapture; one retry, then the error propagates.
                graph = self._capture_once(enqueue)
            self._graphs[key] = graph
            return graph

    def _capture_parts(self, body: "_SegmentedBody", key=None) -> tuple:
        """Capture every non-HOST part of ``body`` once for ``key`` (thread-safe, before
        anything is launched; each part retried once on failure)."""
        if not self._prepared:
            if key is not None:
                raise RuntimeError(
                    f"{self._label()}: prepare() must run (with the default body) before "
                    f"a self-graph is captured for key {key!r}")
            self.prepare(body)
        with self._lock:
            cached = self._seg_graphs.get(key)
            if cached is not None:
                return cached
            graphs = []
            for part in body:
                if part.host:
                    graphs.append(None)
                    continue
                try:
                    g = self._capture_once(part.enqueue)
                except CaptureError:
                    raise
                except RuntimeError:
                    # See _capture: one retry of a capture invalidated by another thread.
                    g = self._capture_once(part.enqueue)
                graphs.append(g)
            graphs = tuple(graphs)
            self._seg_graphs[key] = graphs
            return graphs

    def _capture_once(self, enqueue) -> "wp.Graph":
        dev = self._device
        s = self._capture_stream
        if s is None:
            s = self._capture_stream = wp.Stream(dev)
        ctx = LaunchContext(dev, s)
        with wp.ScopedCapture(device=dev, stream=s, force_module_load=False) as cap, \
                ctx.scope():
            # This capture is Warp's own (never torch's): allow
            # conditional nodes explicitly, so ctx.skip_unless / bounded_loop windows do
            # not fall back to the torch-capture heuristic (unconditional() still wins).
            key = _compat.registered_stream(s)
            _compat.set_conditional_allowed(key, True)
            # The library's own capture: a course's default-stream guard (roadmap PR15)
            # never mistakes it for a caller's capture of Warp's current stream.
            _compat.mark_library_capture(key)
            enqueue(ctx)
        return cap.graph
