"""Record the device work a call emits, and compare it across calls.

:func:`~track_gen.testing.trace_launches` records every ``wp.launch`` /
``wp.launch_tiled`` / ``wp.Launch.launch`` and ``wp.copy`` / ``wp.array.assign`` issued
in a block (by any code outside the ``warp`` package, in the current context) as a
:class:`LaunchRecord`: the kernel, the launch dimensions, the stream handle passed, the
pointer of every array argument and the value of every other argument (the fields of a
``wp.struct`` argument included). Two properties
of a capturable hot path follow from that record:

- **static topology** (:meth:`~track_gen.testing.LaunchTrace.diff`,
  :func:`~track_gen.testing.assert_same_topology`): a CUDA graph replays the launches it
  recorded, so two calls with the same inputs must emit the same launches, in the same
  order, with the same dimensions, buffers and scalar arguments; a launch sequence that
  depends on a Python counter, on data read back or on an allocation is not captured
  faithfully;
- **owned buffers** (:meth:`~track_gen.testing.LaunchTrace.foreign_pointers`,
  :func:`~track_gen.testing.assert_pointers_within`): every pointer must fall inside a
  buffer the producer owns (its result, scratch, seed and parameter buffers, collected
  with :func:`~track_gen.testing.buffers_of`): a buffer allocated per call, or shared
  through a module global, breaks a recorded graph or two producers running side by
  side. Ownership is also checked from the other side by the conformance kit: two
  producers built alike must launch on disjoint buffers.

Example (on CPU, where every generation runs eagerly)::

    from track_gen import TrackGenerator
    from track_gen.pipeline import presets
    from track_gen.testing import (assert_pointers_within, assert_same_topology,
                                   buffers_of, trace_launches)

    gen = TrackGenerator(presets.track(num_envs=64, device="cpu"))
    with trace_launches() as first:
        gen.generate()
    with trace_launches() as second:
        gen.generate()
    owned = buffers_of(gen, root="gen")
    assert_same_topology(first, second, buffers=owned)
    assert_pointers_within(first, owned)
"""
from __future__ import annotations

import bisect
import dataclasses
import types
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import warp as wp

from . import _interpose
from ._interpose import Site

__all__ = [
    "BufferSet",
    "LaunchRecord",
    "LaunchTrace",
    "assert_pointers_within",
    "assert_same_topology",
    "buffers_of",
    "trace_launches",
]


@dataclasses.dataclass(frozen=True)
class LaunchRecord:
    """One recorded launch or copy.

    Attributes:
        op: ``"launch"`` or ``"copy"``.
        kernel: ``"<warp module>.<kernel key>"`` for a launch, ``"wp.copy"`` for a copy
            (``"wp.array.assign"`` for an ``assign``).
        dim: the launch dimensions (a copy: ``(count or dest.size,)``).
        stream: the CUDA handle of the ``stream=`` argument; ``None`` when no stream
            (or ``None``) was passed, e.g. on CPU.
        arrays: ``(argument index, pointer)`` of every array argument (inputs then
            outputs; a copy: ``dest`` is 0, ``src`` is 1); a null array has pointer 0.
            A field of a ``wp.struct`` argument is keyed ``"<index>.<field>"``
            (``"3.staging"``).
        scalars: ``(argument index, value)`` of every other argument (a copy: the
            ``dest_offset``, ``src_offset`` and ``count``), keyed the same way.
        site: the code that emitted it (the caller of ``LaunchContext.launch`` /
            ``copy`` / ``fill`` when it went through one, else the caller).
    """

    op: str
    kernel: str
    dim: tuple
    stream: "int | None"
    arrays: tuple
    scalars: tuple
    site: Site

    def __str__(self) -> str:
        return f"{self.op} {self.kernel} dim={self.dim} from {self.site}"


def _dim(dim) -> tuple:
    if isinstance(dim, (int, np.integer)):
        return (int(dim),)
    try:
        return tuple(int(d) for d in dim)
    except TypeError:
        return (repr(dim),)


def _pointer(value) -> "int | None":
    """The device pointer of an array-like argument, else ``None``."""
    if isinstance(value, wp.array):
        return int(value.ptr or 0)
    data = getattr(value, "data", None)          # wp.indexedarray and friends
    if isinstance(data, wp.array):
        return int(data.ptr or 0)
    return None


def _float(value: float) -> Any:
    # NaN != NaN: a NaN argument is recorded as the string "nan", so two identical
    # launches compare equal.
    return "nan" if value != value else value


def _scalar(value) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, np.generic):
        return _scalar(value.item())
    inner = getattr(value, "value", None)        # wp.float32(1.0), wp.uint32(7), ...
    if isinstance(inner, (bool, int, float)):
        return _scalar(inner)
    try:                                          # vectors, matrices, quaternions
        return tuple(_scalar(v) for v in value)
    except TypeError:
        return f"<{type(value).__name__}>"


def _struct_fields(value) -> "list[str] | None":
    """The field names of a ``wp.struct`` instance, else ``None``."""
    cls = type(value)
    fields = getattr(getattr(cls, "_cls", None), "vars", None)
    if isinstance(fields, Mapping) and _interpose.is_warp_module(cls.__module__ or ""):
        return list(fields)
    return None


def _flatten(key, value, arrays: list, scalars: list, depth: int = 0) -> None:
    """Append ``value`` (a kernel argument) to ``arrays`` / ``scalars``; a struct is
    flattened into its fields, keyed ``"<key>.<field>"``."""
    ptr = _pointer(value)
    if ptr is not None:
        arrays.append((key, ptr))
        return
    fields = _struct_fields(value) if depth < 8 else None
    if fields is None:
        scalars.append((key, _scalar(value)))
        return
    for name in fields:
        _flatten(f"{key}.{name}", getattr(value, name, None), arrays, scalars, depth + 1)


def _stream_handle(stream) -> "int | None":
    if stream is None:
        return None
    handle = getattr(stream, "cuda_stream", None)
    return int(handle or 0) if handle is not None else None


def _record(call) -> LaunchRecord:
    args, kw = call.args, call.kwargs
    if call.op == _interpose.LAUNCH:
        if call.api == "wp.Launch.launch":       # cmd.launch(stream=None)
            cmd = args[0]
            kernel = getattr(cmd, "kernel", None)
            bounds = getattr(cmd, "bounds", None)
            dim = getattr(bounds, "shape", bounds)
            values = cmd.fwd_args                # Python-side args of the recorded launch
            stream = args[1] if len(args) > 1 else kw.get("stream")
            values = list(values)
        else:
            kernel = args[0] if args else kw.get("kernel")
            dim = args[1] if len(args) > 1 else kw.get("dim")
            inputs = args[2] if len(args) > 2 else kw.get("inputs", ())
            outputs = args[3] if len(args) > 3 else kw.get("outputs", ())
            values = list(inputs or ()) + list(outputs or ())
            stream = args[7] if len(args) > 7 else kw.get("stream")
        module = getattr(getattr(kernel, "module", None), "name", None)
        key = getattr(kernel, "key", None) or getattr(kernel, "__name__", repr(kernel))
        name = f"{module}.{key}" if module else str(key)
        arrays, scalars = [], []
        for i, value in enumerate(values):
            _flatten(i, value, arrays, scalars)
        return LaunchRecord("launch", name, _dim(dim), _stream_handle(stream),
                            tuple(arrays), tuple(scalars), call.emitter)
    names = ("dest", "src", "dest_offset", "src_offset", "count", "stream")
    vals = dict(zip(names, args))
    vals.update(kw)
    dest, src = vals.get("dest"), vals.get("src")
    count = int(vals.get("count", 0) or 0)
    size = int(getattr(dest, "size", 0) or 0)
    arrays = tuple((i, p) for i, p in ((0, _pointer(dest)), (1, _pointer(src)))
                   if p is not None)
    scalars = ((2, int(vals.get("dest_offset", 0) or 0)),
               (3, int(vals.get("src_offset", 0) or 0)), (4, count))
    kernel = "wp.array.assign" if call.api == "wp.array.assign" else "wp.copy"
    return LaunchRecord("copy", kernel, (count or size,),
                        _stream_handle(vals.get("stream")), arrays, scalars, call.emitter)


# ---------------------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------------------

class BufferSet:
    """A set of device buffers (address ranges with labels) a pointer can be checked
    against. Build one with :func:`buffers_of`, or :meth:`add` arrays yourself."""

    def __init__(self) -> None:
        self._entries: list = []       # (start, end, label), sorted by start
        self._starts: list = []

    def add(self, array, label: str = "") -> "BufferSet":
        """Add ``array``'s ``[ptr, ptr + capacity)`` range (a null array is ignored)."""
        ptr = int(getattr(array, "ptr", 0) or 0)
        if not ptr:
            return self
        size = int(getattr(array, "capacity", 0) or 0)
        if not size:
            size = int(getattr(array, "size", 0) or 0) * \
                int(wp.types.type_size_in_bytes(array.dtype))
        entry = (ptr, ptr + max(size, 1), label or f"<{array.dtype.__name__}[{array.shape}]>")
        i = bisect.bisect_left(self._starts, ptr)
        self._starts.insert(i, ptr)
        self._entries.insert(i, entry)
        return self

    def locate(self, ptr: int) -> "tuple[str, int] | None":
        """``(label, byte offset)`` of the buffer holding ``ptr`` (the innermost one
        when buffers overlap), or ``None``."""
        ptr = int(ptr)
        i = bisect.bisect_right(self._starts, ptr)
        best = None
        while i > 0:
            i -= 1
            start, end, label = self._entries[i]
            if start <= ptr < end and (best is None or end - start < best[2]):
                best = (label, ptr - start, end - start)
        return None if best is None else (best[0], best[1])

    def __contains__(self, ptr) -> bool:
        return self.locate(ptr) is not None

    def __len__(self) -> int:
        return len(self._entries)

    def labels(self) -> list[str]:
        """The label of every buffer in the set, in address order."""
        return [e[2] for e in self._entries]

    def __repr__(self) -> str:
        return f"BufferSet({len(self)} buffers)"


_SKIP_TYPES = (str, bytes, bytearray, int, float, complex, bool, type(None), type,
               types.ModuleType, types.BuiltinFunctionType, types.CodeType, np.ndarray,
               np.generic, range)


def _shared_specs() -> tuple:
    """The registry's spec types: one spec object serves every generator built from
    it, so nothing reached through it belongs to one producer."""
    try:
        from ..plugin import GeneratorSpec  # noqa: PLC0415
    except ImportError:  # pragma: no cover - a partial install
        return ()
    return (GeneratorSpec,)


def buffers_of(*objects, root: "str | Sequence[str] | None" = None,
               max_depth: int = 10) -> BufferSet:
    """Every ``wp.array`` reachable from ``objects``: through attributes
    (``__dict__`` and ``__slots__``), dataclass fields, ``wp.struct`` fields, mappings,
    sequences, the variables a function closes over and the object a bound method is
    bound to, up to ``max_depth`` levels. Warp objects other than arrays and structs
    (devices, streams, graphs, kernels), modules, types, a function's globals, numpy
    data and registry specs (``GeneratorSpec``, shared by every
    generator built from them) are not entered.

    ``buffers_of(gen)`` collects a ``TrackGenerator``'s / ``GateGenerator``'s result,
    scratch, seed, parameter and input buffers.

    Args:
        root: the label prefix of each object (default ``"obj0"``, ``"obj1"``, ...).
    """
    if isinstance(root, str):
        roots = [root] * len(objects) if len(objects) == 1 else \
            [f"{root}{i}" for i in range(len(objects))]
    elif root is None:
        roots = [f"obj{i}" for i in range(len(objects))]
    else:
        roots = list(root)
    out = BufferSet()
    seen: set = set()
    skip = _SKIP_TYPES + _shared_specs()

    def walk(obj, path: str, depth: int) -> None:
        if isinstance(obj, wp.array):
            if id(obj) not in seen:
                seen.add(id(obj))
                out.add(obj, path)
            return
        if depth > max_depth or isinstance(obj, skip) or id(obj) in seen:
            return
        fields = _struct_fields(obj)
        if fields is not None:
            seen.add(id(obj))
            for name in fields:
                walk(getattr(obj, name, None), f"{path}.{name}", depth + 1)
            return
        module = type(obj).__module__ or ""
        if _interpose.is_warp_module(module) or module.startswith("torch"):
            data = getattr(obj, "data", None)     # wp.indexedarray
            if isinstance(data, wp.array):
                walk(data, f"{path}.data", depth + 1)
            return
        seen.add(id(obj))
        if isinstance(obj, types.MethodType):
            walk(obj.__self__, f"{path}.__self__", depth + 1)
            return
        if isinstance(obj, types.FunctionType):
            names = obj.__code__.co_freevars
            for name, cell in zip(names, obj.__closure__ or ()):
                try:
                    value = cell.cell_contents
                except ValueError:                # an empty cell
                    continue
                walk(value, f"{path}.<{name}>", depth + 1)
            return
        if isinstance(obj, Mapping):
            for k, v in list(obj.items()):
                walk(v, f"{path}[{k!r}]", depth + 1)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for i, v in enumerate(list(obj)):
                walk(v, f"{path}[{i}]", depth + 1)
            return
        attrs = {}
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            attrs.update(d)
        for cls in type(obj).__mro__:
            for slot in getattr(cls, "__slots__", ()) or ():
                if isinstance(slot, str) and slot not in attrs and \
                        slot not in ("__dict__", "__weakref__"):
                    try:
                        attrs[slot] = getattr(obj, slot)
                    except AttributeError:
                        pass
        for k, v in attrs.items():
            walk(v, f"{path}.{k}", depth + 1)

    for obj, name in zip(objects, roots):
        walk(obj, name, 0)
    return out


# ---------------------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------------------

class LaunchTrace:
    """The launches and copies recorded in a :func:`trace_launches` block.

    A sequence of :class:`LaunchRecord` (``len``, iteration, indexing). Calls Warp makes
    internally (e.g. the copy inside ``wp.clone``) are left out unless
    ``include_warp=True``.
    """

    def __init__(self, *, include_warp: bool = False) -> None:
        self.include_warp = bool(include_warp)
        self.records: list[LaunchRecord] = []
        self._cm = None

    def __enter__(self) -> "LaunchTrace":
        if self._cm is not None:
            raise RuntimeError("this LaunchTrace is already recording")
        self.records = []
        self._cm = _interpose.listening(self)
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        cm, self._cm = self._cm, None
        cm.__exit__(exc_type, exc, tb)

    def _on_call(self, call) -> None:
        if call.op not in (_interpose.LAUNCH, _interpose.COPY):
            return
        if not self.include_warp and _interpose.is_warp_module(call.caller.module):
            return
        if _interpose.is_recorded_command(call):  # launched later, by cmd.launch()
            return
        self.records.append(_record(call))

    # -- sequence -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[LaunchRecord]:
        return iter(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def __repr__(self) -> str:
        return f"LaunchTrace({len(self.records)} records)"

    def kernels(self) -> list[str]:
        """The kernel (or ``"wp.copy"``) of every record, in order."""
        return [r.kernel for r in self.records]

    # -- topology -------------------------------------------------------------------
    def topology(self, buffers: "BufferSet | None" = None, *, scalars: bool = True,
                 streams: bool = False, ignore: Iterable[int] = ()) -> list[tuple]:
        """One comparable key per record: ``(op, kernel, dim, arrays[, scalars]
        [, stream])``. With ``buffers``, a pointer inside one of them becomes
        ``(label, offset)`` and any other pointer ``"<foreign>"``; pointers inside the
        ranges of ``ignore`` (an iterable of arrays, e.g. buffers allocated during the
        call, already reported elsewhere) become ``"<allocated>"``."""
        extra = BufferSet()
        for arr in ignore:
            extra.add(arr, "<allocated>")
        keys = []
        for r in self.records:
            arrays = []
            for i, ptr in r.arrays:
                if ptr and len(extra) and extra.locate(ptr) is not None:
                    where = "<allocated>"
                elif buffers is not None and ptr:
                    loc = buffers.locate(ptr)
                    where = loc if loc is not None else "<foreign>"
                else:
                    where = ptr
                arrays.append((i, where))
            key = (r.op, r.kernel, r.dim, tuple(arrays))
            if scalars:
                key += (r.scalars,)
            if streams:
                key += (r.stream,)
            keys.append(key)
        return keys

    def diff(self, other: "LaunchTrace", *, buffers: "BufferSet | None" = None,
             scalars: bool = True, streams: bool = False, ignore: Iterable = (),
             limit: int = 5) -> list[str]:
        """Human-readable differences between this trace and ``other`` (``[]`` when
        their topologies are identical; see :meth:`topology`)."""
        ignore = list(ignore)
        a = self.topology(buffers, scalars=scalars, streams=streams, ignore=ignore)
        b = other.topology(buffers, scalars=scalars, streams=streams, ignore=ignore)
        out = []
        if len(a) != len(b):
            out.append(f"{len(a)} launches/copies vs {len(b)}")
        for i, (ka, kb) in enumerate(zip(a, b)):
            if ka == kb:
                continue
            ra, rb = self.records[i], other.records[i]
            what = []
            if ra.kernel != rb.kernel:
                what.append(f"kernel {ra.kernel} vs {rb.kernel}")
            if ra.dim != rb.dim:
                what.append(f"dim {ra.dim} vs {rb.dim}")
            if ka[3] != kb[3]:
                what.append("buffers")
            if scalars and ka[4] != kb[4]:
                what.append(f"scalar args {_short(ra.scalars)} vs {_short(rb.scalars)}")
            if streams and ka[-1] != kb[-1]:
                what.append(f"stream {ra.stream} vs {rb.stream}")
            out.append(f"#{i} {ra.kernel} (from {ra.site}): " + ", ".join(what))
            if len(out) >= limit:
                break
        if len(a) != len(b) and len(out) < limit:
            longer, n = (self, len(b)) if len(a) > len(b) else (other, len(a))
            extra = longer.records[n:n + 3]
            if extra:
                out.append("extra: " + "; ".join(str(r) for r in extra))
        return out

    def same_topology(self, other: "LaunchTrace", **kwargs) -> bool:
        """Whether :meth:`diff` finds no difference."""
        return not self.diff(other, **kwargs)

    # -- pointers --------------------------------------------------------------------
    def foreign_pointers(self, buffers: BufferSet, *, ignore: Iterable = ()
                         ) -> list[tuple[LaunchRecord, int, int]]:
        """``(record, argument index, pointer)`` of every non-null pointer outside
        ``buffers`` (and outside the arrays in ``ignore``)."""
        extra = BufferSet()
        for arr in ignore:
            extra.add(arr, "<allocated>")
        out = []
        for r in self.records:
            for i, ptr in r.arrays:
                if ptr and buffers.locate(ptr) is None and extra.locate(ptr) is None:
                    out.append((r, i, ptr))
        return out


def _short(values: tuple, n: int = 4) -> str:
    text = ", ".join(f"{i}={v!r}" for i, v in values[:n])
    return "(" + text + (", ..." if len(values) > n else "") + ")"


def trace_launches(*, include_warp: bool = False) -> LaunchTrace:
    """A :class:`LaunchTrace` to use as a context manager (it yields itself). Calls Warp
    makes internally, such as the copy inside ``wp.clone``, are left out unless
    ``include_warp=True``::

        from track_gen import TrackGenerator
        from track_gen.pipeline import presets
        from track_gen.testing import trace_launches

        gen = TrackGenerator(presets.track(num_envs=64, device="cpu"))
        with trace_launches() as trace:
            gen.generate()
        print(len(trace.records), "launches and copies")
    """
    return LaunchTrace(include_warp=include_warp)


def assert_same_topology(a: LaunchTrace, b: LaunchTrace, *,
                         buffers: "BufferSet | None" = None, scalars: bool = True,
                         streams: bool = False) -> None:
    """Raise ``AssertionError`` when two traces differ (see :meth:`LaunchTrace.diff`)."""
    diff = a.diff(b, buffers=buffers, scalars=scalars, streams=streams)
    if diff:
        raise AssertionError("launch topology differs between the two calls:\n  "
                             + "\n  ".join(diff))


def assert_pointers_within(trace: LaunchTrace, buffers: BufferSet) -> None:
    """Raise ``AssertionError`` when a launch or copy touches a buffer outside
    ``buffers``."""
    foreign = trace.foreign_pointers(buffers)
    if foreign:
        lines = [f"{r.kernel} argument {i} -> {hex(p)} (from {r.site})"
                 for r, i, p in foreign[:8]]
        raise AssertionError(f"{len(foreign)} pointer(s) outside the allowed buffers:\n  "
                             + "\n  ".join(lines))
