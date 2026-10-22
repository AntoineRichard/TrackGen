"""The one place the conformance kit patches Warp: a context-local call interposer.

The contract guards (:mod:`track_gen.testing.contract`) and the launch tracer
(:mod:`track_gen.testing.trace`) are *listeners*. While at least one listener is active,
the Warp entry points below are replaced by thin wrappers (installed once, reference
counted, restored when the last listener exits). A wrapper looks up the listeners of the
CURRENT context (a ``contextvars.ContextVar``): calls made from another thread, or from
a context with no listener, go straight to Warp. Each listener sees a :class:`Call` and
may record it or raise; then Warp's real function runs. Calls Warp makes while a sync,
readback, allocation, launch, copy or memset is executing (``wp.array(data)`` copying
its data in) are not reported again.

A module-level entry point (``wp.launch``, ``wp.synchronize``, ...) is replaced wherever
a loaded module binds it: on ``warp`` itself, in the Warp module that defines it
(``warp._src.context.launch``) and in any module that imported it by name
(``from warp import launch``). Classes are not replaced: their ``__init__`` (or the
method) is, so ``isinstance`` and subclassing keep working while a guard is active.

Intercepted (when the running Warp has them):

- host syncs: ``wp.synchronize``, ``wp.synchronize_device``, ``wp.synchronize_stream``,
  ``wp.synchronize_event``; host readbacks: ``wp.array.numpy``, ``wp.array.list`` and
  the ``wp.array.__array_interface__`` a CPU array hands ``np.asarray`` / ``np.array``;
- allocations: Warp's array-allocation chokepoints (``wp.array._init_new`` and
  ``wp.array._init_from_data``), so ``wp.empty`` / ``zeros`` / ``full`` / ``ones`` /
  ``clone`` / ``empty_like`` / ``zeros_like`` / ``array(data)`` and any other path that
  allocates are all seen; when a Warp release lacks them, the public allocation
  functions are patched instead; the native objects ``wp.HashGrid`` (and
  ``HashGrid.reserve``), ``wp.Mesh``, ``wp.Volume`` and ``wp.Bvh``;
- work: ``wp.launch``, ``wp.launch_tiled``, ``wp.Launch.launch`` (a launch recorded with
  ``record_cmd=True``), ``wp.copy``, ``wp.array.assign``, ``wp.array.zero_``,
  ``wp.array.fill_``;
- capture and stream scoping: ``wp.ScopedCapture``, ``wp.capture_begin``,
  ``wp.capture_end``, ``wp.capture_launch``, ``wp.capture_if``, ``wp.capture_while``,
  ``wp.ScopedStream``.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import sys
import threading
import types
from typing import Any, Iterator, Mapping

import warp as wp

__all__ = ["Call", "Site", "listening"]

# Call.op values
SYNC, READBACK, ALLOC, LAUNCH, COPY, MEMSET, CAPTURE, SCOPED_STREAM = (
    "sync", "readback", "alloc", "launch", "copy", "memset", "capture", "scoped_stream")

_LISTENERS: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "track_gen_testing_listeners", default=())
_INSIDE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "track_gen_testing_inside", default=False)

# The listeners between their entry and their exit. A context can still list a listener
# that exited elsewhere (a guard entered in one asyncio task and exited in another): it
# is ignored there.
_ACTIVE: set = set()

_LOCK = threading.Lock()
_INSTALLS = 0
_SAVED: list = []          # (class, attr, original in its __dict__ or _MISSING, installed)
_MARK = "_track_gen_testing_real"
_MISSING = object()


@dataclasses.dataclass(frozen=True)
class Site:
    """A source location: ``module.function:line``.

    Attributes:
        module: the module's ``__name__``.
        function: the function's qualified name (``"Class.method"`` for a method on
            Python >= 3.11).
        line: the line number of the call.
    """

    module: str
    function: str
    line: int

    def __str__(self) -> str:
        return f"{self.module}.{self.function}:{self.line}"


def _site(frame) -> Site:
    code = frame.f_code
    name = getattr(code, "co_qualname", code.co_name)
    return Site(str(frame.f_globals.get("__name__", "?")), name, int(frame.f_lineno))


def is_warp_module(module: str) -> bool:
    return module == "warp" or module.startswith("warp.")


@dataclasses.dataclass(frozen=True)
class Call:
    """One intercepted Warp call, as a listener sees it.

    Attributes:
        api: the Warp API, e.g. ``"wp.launch"``, ``"wp.array.numpy"``; for an
            allocation, the outermost Warp function the caller called (``"wp.empty"``,
            ``"wp.clone"``, ``"wp.array"``, ...).
        op: the kind of call (``"sync"``, ``"readback"``, ``"alloc"``, ``"launch"``,
            ``"copy"``, ``"memset"``, ``"capture"``, ``"scoped_stream"``).
        args, kwargs: the arguments as passed (``args[0]`` is the object for a method,
            a property or ``__init__``).
        caller: the direct caller (may be inside Warp).
        origin: the first caller outside the ``warp`` package.
        emitter: for a launch, copy or memset, the first caller outside ``warp``,
            ``track_gen.exec.context`` and ``track_gen.ops.fill`` (the code that asked
            a ``LaunchContext`` to launch / copy / fill); ``origin`` otherwise.
    """

    api: str
    op: str
    args: tuple
    kwargs: Mapping[str, Any]
    caller: Site
    origin: Site
    emitter: Site

    @property
    def site(self) -> Site:
        """Where to attribute the call: the direct caller unless it is Warp itself."""
        return self.origin if is_warp_module(self.caller.module) else self.caller


def is_recorded_command(call: Call) -> bool:
    """``wp.launch(..., record_cmd=True)``: returns a ``wp.Launch`` and launches nothing
    (its ``launch()`` is seen when it runs)."""
    if call.api not in ("wp.launch", "wp.launch_tiled"):
        return False
    return bool(call.kwargs.get("record_cmd",
                                call.args[10] if len(call.args) > 10 else False))


def _alloc_api(frame) -> "tuple[str, Any]":
    """``(api, origin frame)`` of an allocation: walk out of Warp, naming the outermost
    Warp function on the way (``array.__init__`` -> ``wp.array``)."""
    api, f = "wp.array", frame
    while f is not None and is_warp_module(str(f.f_globals.get("__name__", ""))):
        code = f.f_code
        name = getattr(code, "co_qualname", code.co_name)
        if name not in ("array._init_new", "array._init_from_data", "_init_new",
                        "_init_from_data"):
            api = "wp.array" if name in ("array.__init__", "__init__") else f"wp.{name}"
        f = f.f_back
    return api, f


def _origin(frame):
    f = frame
    while f is not None and is_warp_module(str(f.f_globals.get("__name__", ""))):
        f = f.f_back
    return f if f is not None else frame


_EMISSION = ("track_gen.exec.context", "track_gen.ops.fill")


def _emitter(frame):
    f = frame
    while f is not None:
        module = str(f.f_globals.get("__name__", ""))
        if not is_warp_module(module) and module not in _EMISSION:
            return f
        f = f.f_back
    return frame


# Ops whose real call runs only Warp code: what Warp does inside them is not reported
# again. The capture ops run caller code (wp.capture_if's on_true, ...), so calls made
# inside them stay visible.
_LEAF_OPS = frozenset({SYNC, READBACK, ALLOC, LAUNCH, COPY, MEMSET})


def _wrap(real, api: str, op: str):
    """A function wrapper of ``real`` that reports each call to the active listeners."""
    leaf = op in _LEAF_OPS
    walk_alloc = op == ALLOC and api == "wp.array"

    def wrapper(*args, **kwargs):
        listeners = _LISTENERS.get()
        if not listeners or _INSIDE.get():
            return real(*args, **kwargs)
        listeners = [x for x in listeners if x in _ACTIVE]
        if not listeners:
            return real(*args, **kwargs)
        frame = sys._getframe(1)
        if walk_alloc:
            name, origin = _alloc_api(frame)
        else:
            name, origin = api, _origin(frame)
        origin = origin if origin is not None else frame
        emitter = _emitter(origin) if op in (LAUNCH, COPY, MEMSET) else origin
        call = Call(name, op, args, kwargs, _site(frame), _site(origin), _site(emitter))
        for listener in listeners:
            listener._on_call(call)
        if not leaf:
            return real(*args, **kwargs)
        token = _INSIDE.set(True)
        try:
            return real(*args, **kwargs)
        finally:
            _INSIDE.reset(token)

    try:
        wrapper.__name__ = real.__name__
        wrapper.__qualname__ = getattr(real, "__qualname__", real.__name__)
        wrapper.__doc__ = real.__doc__
        wrapper.__wrapped__ = real
    except (AttributeError, TypeError):  # pragma: no cover - exotic callables
        pass
    setattr(wrapper, _MARK, real)
    return wrapper


class _Property(property):
    """A property whose getter reports to the listeners (it can carry the mark)."""


def _wrap_attr(real, api: str, op: str):
    if isinstance(real, property):
        prop = _Property(_wrap(real.fget, api, op), real.fset, real.fdel, real.__doc__)
        setattr(prop, _MARK, real)
        return prop
    return _wrap(real, api, op)


def _unwrap(value):
    """The original of one of our wrappers (also a stale one, left in place by a
    foreign patch that was restored after us); ``value`` otherwise."""
    if isinstance(value, (types.FunctionType, _Property)):
        return value.__dict__.get(_MARK, value)
    return value


_PUBLIC_ALLOCATORS = ("empty", "zeros", "ones", "full", "clone", "empty_like",
                      "zeros_like", "ones_like", "full_like")


def _targets() -> list:
    """``(owner, attribute, api, op)`` of every entry point to patch (present ones). An
    owner is the ``warp`` module (a module-level function, replaced wherever a module
    binds it) or a class (its attribute is replaced)."""
    out = []
    for name in ("synchronize", "synchronize_device", "synchronize_stream",
                 "synchronize_event"):
        out.append((wp, name, f"wp.{name}", SYNC))
    out += [(wp.array, "numpy", "wp.array.numpy", READBACK),
            (wp.array, "list", "wp.array.list", READBACK),
            (wp.array, "__array_interface__", "wp.array.__array_interface__", READBACK)]
    if hasattr(wp.array, "_init_new") and hasattr(wp.array, "_init_from_data"):
        out += [(wp.array, "_init_new", "wp.array", ALLOC),
                (wp.array, "_init_from_data", "wp.array", ALLOC)]
    else:  # pragma: no cover - a Warp release without the chokepoints
        out += [(wp, name, f"wp.{name}", ALLOC) for name in _PUBLIC_ALLOCATORS]
    for name in ("HashGrid", "Mesh", "Volume", "Bvh"):
        cls = getattr(wp, name, None)
        if isinstance(cls, type):
            out.append((cls, "__init__", f"wp.{name}", ALLOC))
    if isinstance(getattr(wp, "HashGrid", None), type):
        out.append((wp.HashGrid, "reserve", "wp.HashGrid.reserve", ALLOC))
    out += [(wp, "launch", "wp.launch", LAUNCH),
            (wp, "launch_tiled", "wp.launch_tiled", LAUNCH),
            (wp, "copy", "wp.copy", COPY),
            (wp.array, "assign", "wp.array.assign", COPY),
            (wp.array, "zero_", "wp.array.zero_", MEMSET),
            (wp.array, "fill_", "wp.array.fill_", MEMSET)]
    if isinstance(getattr(wp, "Launch", None), type):
        out.append((wp.Launch, "launch", "wp.Launch.launch", LAUNCH))
    for name in ("capture_begin", "capture_end", "capture_launch", "capture_if",
                 "capture_while"):
        out.append((wp, name, f"wp.{name}", CAPTURE))
    out += [(wp.ScopedCapture, "__init__", "wp.ScopedCapture", CAPTURE),
            (wp.ScopedStream, "__init__", "wp.ScopedStream", SCOPED_STREAM)]
    return [t for t in out if hasattr(t[0], t[1])]


def _rebind_modules(replace) -> None:
    """Rebind every global of every loaded module for which ``replace(value)`` returns
    a new value. Only plain functions are considered."""
    for module in list(sys.modules.values()):
        if not isinstance(module, types.ModuleType):
            continue
        d = getattr(module, "__dict__", None)
        if not isinstance(d, dict):  # pragma: no cover - exotic module objects
            continue
        hits = []
        for key, value in list(d.items()):   # list(): another thread may import
            if type(value) is types.FunctionType:
                new = replace(value)
                if new is not None:
                    hits.append((key, new))
        for key, new in hits:
            d[key] = new


def _install() -> None:
    global _INSTALLS
    with _LOCK:
        if _INSTALLS == 0:
            functions = {}                   # id(real) -> (real, wrapper)
            for owner, attr, api, op in _targets():
                if isinstance(owner, types.ModuleType):
                    real = _unwrap(getattr(owner, attr))
                    functions[id(real)] = (real, _wrap(real, api, op))
                    continue
                own = vars(owner).get(attr, _MISSING)
                real = _unwrap(own if own is not _MISSING else getattr(owner, attr))
                wrapper = _wrap_attr(real, api, op)
                setattr(owner, attr, wrapper)
                _SAVED.append((owner, attr, _MISSING if own is _MISSING else real,
                               wrapper))

            def to_wrapper(value):
                hit = functions.get(id(_unwrap(value)))
                return hit[1] if hit is not None and hit[0] is _unwrap(value) else None

            _rebind_modules(to_wrapper)
        _INSTALLS += 1


def _uninstall() -> None:
    global _INSTALLS
    with _LOCK:
        _INSTALLS -= 1
        if _INSTALLS:
            return
        for owner, attr, original, wrapper in reversed(_SAVED):
            # Someone patched over us meanwhile (and will restore our wrapper, which
            # is a pass-through without listeners and is unwrapped by the next
            # install): leave their value in place.
            if vars(owner).get(attr, _MISSING) is wrapper:
                if original is _MISSING:
                    delattr(owner, attr)
                else:
                    setattr(owner, attr, original)
        _SAVED.clear()

        def to_original(value):
            real = _unwrap(value)
            return real if real is not value else None

        _rebind_modules(to_original)


@contextlib.contextmanager
def listening(listener) -> Iterator[None]:
    """Make ``listener`` (an object with ``_on_call(call)``) active in the current
    context for the duration of the block."""
    _install()
    try:
        token = _LISTENERS.set(_LISTENERS.get() + (listener,))
        _ACTIVE.add(listener)
    except BaseException:  # pragma: no cover - ContextVar.set does not fail
        _uninstall()
        raise
    try:
        yield
    finally:
        _ACTIVE.discard(listener)
        try:
            _LISTENERS.reset(token)
        except ValueError:
            # Exited in another context than it was entered in (another asyncio
            # task, say): drop this listener there (_ACTIVE covers the other one).
            _LISTENERS.set(tuple(x for x in _LISTENERS.get() if x is not listener))
        finally:
            _uninstall()
