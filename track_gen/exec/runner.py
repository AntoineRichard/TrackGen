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
==================  ====================================================

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

Legacy-ABI plugins (no ``accepts_ctx``) launch on Warp's CURRENT stream,
which is shared by every thread. A producer containing one therefore:

- captures its (default-key) self-graph on that stream, under the process-wide
  ``runtime._CAPTURE_LOCK``, and runs any other key eagerly;
- outside a capture, runs its whole body on that stream when the call
  resolved another one, ordered after and before the resolved stream with
  events (so the plugin's launches and the library's stages stay ordered);
- is recorded into a caller's capture only when the capture stream IS Warp's
  current stream.

While a legacy producer may still self-capture on Warp's current stream, every
other producer whose call resolves to that same stream takes the lock too and
decides its mode under it (otherwise it could see the legacy capture as "the
caller's capture" and record itself into it). Calls on any other stream stay
lock-free.
"""
from __future__ import annotations

import contextlib
import threading
import warnings
import weakref

import warp as wp

from .._src import runtime as _runtime
from . import _compat
from .context import LaunchContext, _torch_capturing
from .errors import CaptureError, LegacyPluginStreamError, NotCapturableError

__all__ = ["GraphRunner", "ensure_not_capturing"]

_POLICIES = ("auto", "eager")

# At most this many self-graphs per runner for non-None keys (e.g. one per mask
# pointer). Calls with a further new key run eagerly (one warning per runner), so a
# caller passing a fresh mask array per call cannot grow the cache without bound.
MAX_KEYED_GRAPHS = 16

# Capturable legacy-plugin runners on CUDA that have not captured their
# self-graph yet: each may still open a capture on Warp's shared current stream.
_PENDING_LEGACY: "weakref.WeakSet[GraphRunner]" = weakref.WeakSet()


def _legacy_capture_pending(alias: str) -> bool:
    """Whether a legacy runner on device ``alias`` may still self-capture on
    Warp's current stream (conservatively ``True`` on a concurrent update)."""
    if not _PENDING_LEGACY:
        return False
    try:
        return any(r._device.alias == alias for r in list(_PENDING_LEGACY))
    except RuntimeError:  # set changed size during iteration
        return True


def _is_current(stream, dev) -> bool:
    """Whether ``stream`` has the handle of Warp's current stream on ``dev``."""
    return stream is not None and \
        int(stream.cuda_stream or 0) == int(wp.get_stream(dev).cuda_stream or 0)


def ensure_not_capturing(device, *, default_stream=None, label: str = "") -> LaunchContext:
    """Raise :class:`~track_gen.exec.CaptureError` if construction-time work on
    ``device`` would land in a capture; return the resolved context otherwise.

    Construction allocates and zero-fills on Warp's CURRENT stream and runs a
    real warm-up pass on the resolved stream, so it is refused when either of
    them is capturing, or when torch is capturing in the calling thread. Call
    it BEFORE the first allocation, so a refused construction leaves the
    caller's capture untouched.
    """
    dev = wp.get_device(device)
    ctx = LaunchContext.for_call(dev, None, default=default_stream)
    capturing = ctx.capturing
    if not capturing and dev.is_cuda:
        capturing = wp.get_stream(dev).is_capturing or _torch_capturing(dev) is not None
    if capturing:
        what = repr(label) if label else "this producer"
        raise CaptureError(
            f"{what} must be constructed (prepared) outside any CUDA graph capture: "
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
        legacy_stream: the body contains a legacy-ABI plugin (no
            ``accepts_ctx``) whose own launches go to Warp's CURRENT stream
            (see the module docstring for what that changes).
            Recording into a caller's capture requires the capture stream to
            be that current stream
            (:class:`~track_gen.exec.LegacyPluginStreamError` otherwise).
        default_stream: stream used when a call passes none and no ambient
            context applies (see :func:`~track_gen.exec.resolve_stream`).
        owner: object anchored on a caller's Warp graph when the body is
            recorded into it, so the buffers the graph references outlive
            the producer. Held weakly by the runner.
        name: label used in error messages.

    A runner is single-writer: running the same producer concurrently from
    two threads is a race on its output buffers (use one producer per
    concurrent caller). Different producers are independent: each self-graph
    is captured on the producer's own private stream, so no process-wide lock
    is taken (legacy-plugin producers, and calls on Warp's current stream
    while one of them may still self-capture, excepted).
    """

    def __init__(self, device, *, capturable: bool, legacy_stream: bool = False,
                 default_stream: "wp.Stream | None" = None, owner=None,
                 name: str = "") -> None:
        dev = wp.get_device(device)
        _check_default_stream(dev, default_stream)
        self._device = dev
        self._capturable = bool(capturable)
        self._legacy_stream = bool(legacy_stream)
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
        self._keyed_limit_warned = False
        self._capture_stream: "wp.Stream | None" = None
        self._prepared = False
        self._lock = threading.RLock()
        if self._legacy_stream and self._capturable and dev.is_cuda:
            _PENDING_LEGACY.add(self)

    # -- state ------------------------------------------------------------------
    @property
    def graph(self) -> "wp.Graph | None":
        """The default-key self-graph, once captured (first CUDA call with
        ``key=None`` outside a capture); ``None`` on CPU, for a non-capturable
        producer, or before that call."""
        return self._graphs.get(None)

    def graph_for(self, key=None) -> "wp.Graph | None":
        """The self-graph cached for ``key`` (see :meth:`run`), if captured."""
        return self._graphs.get(key)

    @property
    def graph_keys(self) -> tuple:
        """The keys of every self-graph captured so far (``None`` = default)."""
        return tuple(self._graphs)

    @property
    def device(self):
        return self._device

    @property
    def capturable(self) -> bool:
        return self._capturable

    @property
    def legacy_stream(self) -> bool:
        return self._legacy_stream

    @property
    def default_stream(self) -> "wp.Stream | None":
        return self._default_stream

    @property
    def name(self) -> str:
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

    def serial(self, ctx: LaunchContext):
        """The lock a call on ``ctx`` must hold while it decides its mode and
        runs (a context manager; a no-op for most calls).

        ``runtime._CAPTURE_LOCK`` (reentrant) for a legacy-plugin producer on
        CUDA, and for any call resolving to Warp's current stream while a
        legacy producer on the device may still self-capture there. Callers
        that issue their own launches ahead of :meth:`run` (e.g. a reseed)
        hold it around both.
        """
        dev = self._device
        if not dev.is_cuda:
            return contextlib.nullcontext()
        if self._legacy_stream:
            return _runtime._CAPTURE_LOCK
        if _legacy_capture_pending(dev.alias) and _is_current(ctx.stream, dev):
            return _runtime._CAPTURE_LOCK
        return contextlib.nullcontext()

    # -- execution --------------------------------------------------------------
    def prepare(self, enqueue) -> None:
        """Run ``enqueue`` once, for real, outside any capture, and block until
        it finished.

        Launching for real loads every module / kernel overload / block-dim
        variant the hot path uses (``force_module_load`` and ``load_module``
        miss launch-time overloads), so later recordings never load modules
        while capturing. The pass leaves a valid result in the producer's
        buffers.

        When the resolved stream is not Warp's current stream (a
        ``default_stream`` or an ambient scope), it first waits on the current
        stream, where the producer's buffers were just allocated and zeroed.
        The host then blocks on the resolved stream (a cold, construction-time
        cost), so the result is complete for any later stream or readback.

        Raises:
            CaptureError: if the resolved stream or Warp's current stream is
                capturing, or torch is capturing in the calling thread.
        """
        ctx = self.context()
        with self.serial(ctx):
            ensure_not_capturing(self._device, default_stream=self._default_stream,
                                 label=self._name)
            if ctx.stream is not None and self._eager_context(ctx) is ctx:
                current = wp.get_stream(self._device)
                if not _is_current(ctx.stream, self._device):
                    ctx.stream.wait_stream(current)
            self._run_eager(enqueue, ctx)
            self._sync(ctx)
        self._prepared = True

    def run(self, enqueue, *, stream=None, policy: str = "auto",
            sync: bool = False, key=None) -> LaunchContext:
        """Run ``enqueue`` once under the execution policy; returns the call's
        resolved :class:`LaunchContext`.

        Args:
            enqueue: the producer's body, ``enqueue(ctx) -> None``.
            stream: optional ``wp.Stream`` (or ``"torch"``) for this call.
            policy: ``"auto"`` (record / self-graph / eager as above) or
                ``"eager"`` (record when capturing, otherwise run the body
                directly; never self-graph).
            sync: block the host on the resolved stream afterwards; skipped
                when recording, and while torch is capturing in the calling
                thread.
            key: hashable self-graph cache key identifying WHICH body
                ``enqueue`` is (e.g. ``(advance, mask.ptr)``). The self-graph
                of a key is captured on its first SELF-GRAPH call and replayed
                by every later call with an equal key, so equal keys must mean
                identical bodies (same launches, same buffer pointers).
                ``None`` (default) is the producer's default body
                (:attr:`graph`). A non-``None`` key requires :meth:`prepare` to
                have run (a keyed body may not be idempotent, so it is never
                used as a warm-up pass). Ignored when recording or eager. A
                legacy-plugin producer runs non-``None`` keys eagerly (it
                self-captures on Warp's shared current stream, once). At most
                :data:`MAX_KEYED_GRAPHS` non-``None`` keys get a self-graph;
                calls with a further new key run eagerly (with a one-time
                ``RuntimeWarning``).
        """
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
        ctx = self.context(stream)
        with self.serial(ctx):
            # Decided under the lock: a legacy self-capture on Warp's current stream
            # must never be mistaken for the caller's capture.
            if ctx.capturing:
                # RECORD: the caller owns the graph. Every check raises BEFORE any launch.
                self.check_record(ctx)
                with ctx.scope():
                    enqueue(ctx)
                owner = self._owner()
                if owner is not None:
                    _compat.anchor(ctx.stream, owner)
                return ctx
            if not self._device.is_cuda or policy == "eager" or not self._capturable \
                    or (key is not None and self._legacy_stream):
                # EAGER (a legacy producer self-graphs its default body only: another
                # capture on Warp's shared current stream would reopen the window in
                # which other calls on that stream must serialize, see serial()).
                self._run_eager(enqueue, ctx)
            else:
                # SELF-GRAPH: capture once (private stream), replay on the caller's stream.
                g = self._graphs.get(key)
                if g is None and key is not None and self._keyed_full():
                    self._run_eager(enqueue, ctx)
                else:
                    if g is None:
                        g = self._capture(enqueue, key)
                    wp.capture_launch(g, stream=ctx.stream)
            if sync:
                self._sync(ctx)
        return ctx

    def check_record(self, ctx: LaunchContext) -> None:
        """Raise if ``ctx`` is capturing and this producer cannot be recorded
        there; a no-op outside a capture. Lets a caller validate before
        issuing its own launches ahead of :meth:`run` (e.g. a reseed).

        Raises:
            NotCapturableError: the producer is not capturable.
            LegacyPluginStreamError: a legacy-plugin producer and the capture
                stream is not Warp's current stream.
        """
        if not ctx.capturing:
            return
        if not self._capturable:
            raise NotCapturableError(self._name)
        if self._legacy_stream and not _is_current(ctx.stream, self._device):
            raise LegacyPluginStreamError(self._name)

    # -- internals --------------------------------------------------------------
    def _label(self) -> str:
        return repr(self._name) if self._name else "this producer"

    def _owner(self):
        return None if self._owner_ref is None else self._owner_ref()

    def _keyed_full(self) -> bool:
        """Whether the keyed self-graph cache is full (warns once when it is)."""
        if sum(k is not None for k in self._graphs) < MAX_KEYED_GRAPHS:
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

    def _eager_context(self, ctx: LaunchContext) -> LaunchContext:
        """The context an eager body actually runs on: ``ctx``, or Warp's current
        stream for a legacy plugin whose call resolved another stream (its own
        launches go to the current stream, so the whole body must)."""
        if not self._legacy_stream or ctx.stream is None or \
                _is_current(ctx.stream, self._device):
            return ctx
        current = wp.get_stream(self._device)
        if current.is_capturing:
            raise LegacyPluginStreamError(
                self._name,
                detail="Here the call resolved a stream that is not capturing while "
                       "Warp's current stream is: the plugin's own launches would be "
                       "recorded into that capture.")
        return LaunchContext(self._device, current)

    def _run_eager(self, enqueue, ctx: LaunchContext) -> None:
        """Run the body for real on ``ctx`` (or on Warp's current stream for a
        legacy plugin, ordered after and before ``ctx.stream`` with events)."""
        run_ctx = self._eager_context(ctx)
        if run_ctx is ctx:
            with ctx.scope():
                enqueue(ctx)
            return
        current = run_ctx.stream
        current.wait_stream(ctx.stream)
        with run_ctx.scope():
            enqueue(run_ctx)
        ctx.stream.wait_stream(current)

    def _capture_serial(self):
        """Lock held for a self-capture: the process-wide one for legacy plugins
        (they record Warp's shared current stream), none otherwise (private
        stream)."""
        if self._legacy_stream and self._device.is_cuda:
            return _runtime._CAPTURE_LOCK
        return contextlib.nullcontext()

    def _capture(self, enqueue, key=None) -> "wp.Graph":
        """Capture the self-graph of ``key`` once (thread-safe; retried once on
        failure)."""
        if not self._prepared:
            if key is not None:
                raise RuntimeError(
                    f"{self._label()}: prepare() must run (with the default body) before "
                    f"a self-graph is captured for key {key!r}")
            self.prepare(enqueue)
        with self._capture_serial(), self._lock:
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
            _PENDING_LEGACY.discard(self)
            return graph

    def _capture_once(self, enqueue) -> "wp.Graph":
        dev = self._device
        if self._legacy_stream:
            # Legacy plugins launch on Warp's current stream: record that stream.
            s = wp.get_stream(dev)
        else:
            s = self._capture_stream
            if s is None:
                s = self._capture_stream = wp.Stream(dev)
        ctx = LaunchContext(dev, s)
        with wp.ScopedCapture(device=dev, stream=s, force_module_load=False) as cap, \
                ctx.scope():
            enqueue(ctx)
        return cap.graph
