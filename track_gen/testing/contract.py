"""Hot-path contract guards: no host sync, no allocation, stream-explicit work.

A capturable generator's hot path (its ``generate`` / a Stage's ``enqueue``, and every
library call it makes) must be recordable into a caller's CUDA graph. On a GPU that is
checked by capturing it (:mod:`track_gen.testing.capture`). Without one, the guards here
check the *call shape* instead, on any device:

- :func:`no_host_sync`: no ``wp.synchronize*``, no host readback
  (``wp.array.numpy()`` / ``wp.array.list()``, ``np.asarray`` of a CPU array) and no
  ``wp.copy`` between two devices;
- :func:`no_allocation`: no Warp array allocation (``wp.empty`` / ``zeros`` / ``full``
  / ``ones`` / ``clone`` / ``array(data)`` / ``empty_like`` / ``zeros_like`` / ..., seen
  at Warp's allocation chokepoint, so every path is covered) and no native Warp object
  (``wp.HashGrid`` and ``HashGrid.reserve``, ``wp.Mesh``, ``wp.Volume``, ``wp.Bvh``);
- :func:`stream_explicit`: every ``wp.launch`` / ``wp.launch_tiled`` /
  ``wp.Launch.launch`` / ``wp.copy`` issued from the probed modules passes ``stream=``
  (also when it reaches Warp through a Warp utility the probed code called), and none
  of them calls ``wp.array.assign()`` / ``zero_()`` / ``fill_()`` (their copy or memset
  targets Warp's current stream) or enters ``wp.ScopedStream``. The check sees that a
  ``stream=`` is passed, not which stream: ``stream=None`` (what a CPU
  ``LaunchContext`` passes) and a launch through a context other than the call's are
  only caught by recording into a real capture on CUDA
  (:mod:`track_gen.testing.capture`);
- ``capture`` (in :func:`hot_path_guard`): no nested ``wp.ScopedCapture`` /
  ``capture_begin`` / ``capture_end`` / ``capture_launch`` / ``capture_if`` /
  ``capture_while`` (a graph launch cannot be recorded into another capture).

:func:`hot_path_guard` combines all four. Each returns a :class:`HotPathGuard`, a
context manager that yields itself::

    from track_gen.testing.contract import hot_path_guard

    gen = TrackGenerator(cfg, rng)          # construction may allocate and sync
    with hot_path_guard(modules=("track_gen", "acme_tracks")):
        gen.generate()                      # raises HotPathViolation on the first breach

With ``strict=True`` (the default) a violating call raises :class:`HotPathViolation`
before it runs, and the block also raises on exit if a violation was recorded (so a
broad ``except`` in the code under test cannot hide one). With ``strict=False`` the guard
only records: every call goes through and :attr:`HotPathGuard.violations` lists them.

The guards patch Warp's entry points (wherever a module binds them, so ``from warp
import launch`` is seen too) for as long as one is active, and dispatch on a context
variable: calls made from other threads, or from a context without a guard, go straight
to Warp and are not seen. While :func:`no_host_sync` is active the
``TRACKGEN_DEBUG_SYNC`` debug sync is switched off (it is a deliberate, opt-in host
sync); the switch is the environment variable, so it is off for the whole process until
the guard exits.

Guard a call that runs eagerly: on CPU every generation does; on CUDA use
``enqueue()`` (a CUDA ``generate()`` replays the generator's own graph, which the
``capture`` check reports as a ``wp.capture_launch``).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Iterable, Sequence

import warp as wp

from . import _interpose
from ._interpose import Site

__all__ = [
    "CATEGORIES",
    "HotPathGuard",
    "HotPathViolation",
    "Site",
    "Violation",
    "hot_path_guard",
    "no_allocation",
    "no_host_sync",
    "stream_explicit",
]

#: Violation categories, in report order.
CATEGORIES = ("host_sync", "allocation", "stream", "capture")

_DEBUG_SYNC_ENV = "TRACKGEN_DEBUG_SYNC"

_HINTS = {
    "host_sync": "A capturable hot path must not synchronize or read device data back: "
                 "a CUDA graph capture can contain neither.",
    "allocation": "A capturable hot path must not allocate: allocate every buffer once, "
                  "in alloc_scratch or, for a Stage, in build through the StageBuilder.",
    "stream": "Emit device work through the LaunchContext (ctx.launch, ctx.copy, "
              "ctx.fill, or wp.launch / wp.copy with stream=ctx.stream) so it lands on "
              "the stream being captured, not on Warp's current stream.",
    "capture": "A hot path must not open, end or launch a CUDA graph itself: a graph "
               "launch cannot be recorded into the caller's capture.",
}


class HotPathViolation(AssertionError):
    """A guarded hot path broke the contract (raised by a strict guard)."""


@dataclasses.dataclass(frozen=True)
class Violation:
    """One contract breach recorded by a :class:`HotPathGuard`.

    Attributes:
        category: ``"host_sync"``, ``"allocation"``, ``"stream"`` or ``"capture"``.
        api: the Warp call, e.g. ``"wp.array.numpy()"``, ``"wp.empty()"``,
            ``"wp.launch without stream="``.
        site: where it was called from (the first frame outside Warp for a call Warp
            made on the caller's behalf, e.g. an allocation).
        key: ``(api, module, function)`` of the direct caller, the form
            :func:`no_host_sync`'s ``allow=`` takes.
        array: for an allocation, the ``wp.array`` it created (kept alive by the
            guard, see :meth:`HotPathGuard.allocated`); ``None`` otherwise.
    """

    category: str
    api: str
    site: Site
    key: tuple = ()
    array: "object | None" = dataclasses.field(default=None, compare=False, repr=False)

    def __str__(self) -> str:
        return f"{self.api} from {self.site}"


def _in_modules(module: str, prefixes: Sequence[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _stream_given(call) -> bool:
    args, kwargs = call.args, call.kwargs
    if "stream" in kwargs:
        return True
    if call.api == "wp.Launch.launch":         # cmd.launch(stream=None)
        return len(args) > 1
    if call.api == "wp.array.assign":          # no stream parameter
        return False
    # positional: wp.launch / wp.launch_tiled(kernel, dim, inputs, outputs, adj_inputs,
    # adj_outputs, device, stream, ...); wp.copy(dest, src, dest_offset, src_offset,
    # count, stream)
    return len(args) > (7 if call.op == _interpose.LAUNCH else 5)


def _copy_devices(call) -> "tuple[str, str] | None":
    """``(src device, dest device)`` of a copy between two Warp arrays on different
    devices, else ``None``."""
    names = ("dest", "src")
    vals = dict(zip(names, call.args))
    vals.update({k: v for k, v in call.kwargs.items() if k in names})
    dest, src = vals.get("dest"), vals.get("src")
    if not (isinstance(dest, wp.array) and isinstance(src, wp.array)):
        return None
    a, b = str(src.device), str(dest.device)
    return (a, b) if a != b else None


def _api_text(api: str) -> str:
    return api if api.endswith("__") else f"{api}()"


class HotPathGuard:
    """Records (and, when strict, refuses) hot-path contract breaches in a block.

    The type :func:`hot_path_guard` / :func:`no_host_sync` / :func:`no_allocation` /
    :func:`stream_explicit` return (build one with them); use it as a context manager
    (it yields itself). A guard can be entered again after it exited; its record is
    reset on every entry.

    Args:
        host_sync: check host syncs and readbacks.
        allocation: check Warp array allocations.
        stream: check that work from ``modules`` is stream-explicit.
        capture: check for nested graph captures and graph launches.
        modules: dotted module prefixes whose launches, copies, memsets and stream
            scopes the ``stream`` check covers (``"acme"`` covers ``acme`` and
            ``acme.kernels``). Host syncs, allocations and captures are checked from
            every caller.
        allow: ``(api, module, function)`` keys of host syncs / readbacks to let
            through and record in :attr:`allowed` instead (``api`` without the call
            parentheses, e.g. ``("wp.array.numpy", "acme.gen", "generate")``; the
            function is the direct caller's qualified name, ``"Class.method"`` for a
            method on Python >= 3.11).
        strict: raise :class:`HotPathViolation` at the first breach (default), or
            only record.
    """

    def __init__(self, *, host_sync: bool = True, allocation: bool = True,
                 stream: bool = True, capture: bool = True,
                 modules: Iterable[str] = ("track_gen",),
                 allow: Iterable[tuple] = (), strict: bool = True) -> None:
        self.checks = frozenset(c for c, on in zip(
            CATEGORIES, (host_sync, allocation, stream, capture)) if on)
        if isinstance(modules, str):
            modules = (modules,)
        self.modules = tuple(str(m) for m in modules)
        self.allow = frozenset(tuple(k) for k in allow)
        self.strict = bool(strict)
        #: Every breach recorded in the last (or current) block, in call order.
        self.violations: list[Violation] = []
        #: The ``allow=`` keys that were hit (let through), in call order.
        self.allowed: list[tuple] = []
        self._cm = None
        self._env: "str | None" = None

    # -- context manager --------------------------------------------------------
    def __enter__(self) -> "HotPathGuard":
        if self._cm is not None:
            raise RuntimeError("this HotPathGuard is already active")
        self.violations = []
        self.allowed = []
        if "host_sync" in self.checks:
            self._env = os.environ.pop(_DEBUG_SYNC_ENV, None)
        self._cm = _interpose.listening(self)
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        cm, self._cm = self._cm, None
        try:
            cm.__exit__(exc_type, exc, tb)
        finally:
            if self._env is not None:
                os.environ[_DEBUG_SYNC_ENV] = self._env
                self._env = None
        if self.strict and exc_type is None and self.violations:
            raise HotPathViolation(self.report())

    # -- listener ---------------------------------------------------------------
    def _on_call(self, call) -> None:
        op, caller = call.op, call.caller
        if op in (_interpose.SYNC, _interpose.READBACK):
            if "host_sync" not in self.checks:
                return
            if call.api == "wp.array.__array_interface__" and not getattr(
                    getattr(call.args[0], "device", None), "is_cpu", False):
                return                          # a CUDA array refuses it (no readback)
            self._host_sync(call, _api_text(call.api))
        elif op == _interpose.ALLOC:
            if "allocation" in self.checks:
                arr = call.args[0] if call.args else None
                self._record(Violation("allocation", f"{call.api}()", call.site,
                                       (call.api, call.site.module, call.site.function),
                                       array=arr))
        elif op in (_interpose.LAUNCH, _interpose.COPY):
            if _interpose.is_recorded_command(call):
                return
            devices = _copy_devices(call) if op == _interpose.COPY else None
            if devices is not None and "host_sync" in self.checks:
                self._host_sync(call, f"{call.api} from {devices[0]} to {devices[1]}")
            # A Warp utility the probed code called (wp.utils.array_cast, ...) launches
            # on its behalf: attribute it to the first caller outside Warp.
            site = call.site
            if "stream" in self.checks and _in_modules(site.module, self.modules) \
                    and not _stream_given(call):
                what = f"{call.api}()" if call.api == "wp.array.assign" else \
                    f"{call.api} without stream="
                self._record(Violation("stream", what, site,
                                       (call.api, site.module, site.function)))
        elif op == _interpose.MEMSET:
            if "stream" in self.checks and _in_modules(caller.module, self.modules):
                self._record(Violation("stream", f"{call.api}()", caller,
                                       (call.api, caller.module, caller.function)))
        elif op == _interpose.SCOPED_STREAM:
            if "stream" in self.checks and _in_modules(caller.module, self.modules):
                self._record(Violation("stream", call.api, caller,
                                       (call.api, caller.module, caller.function)))
        elif op == _interpose.CAPTURE:
            if "capture" in self.checks and \
                    not _interpose.is_warp_module(caller.module):
                self._record(Violation("capture", f"{call.api}()", caller,
                                       (call.api, caller.module, caller.function)))

    def _host_sync(self, call, what: str) -> None:
        caller = call.caller
        key = (call.api, caller.module, caller.function)
        if key in self.allow:
            self.allowed.append(key)
            return
        self._record(Violation("host_sync", what, call.site, key))

    def _record(self, violation: Violation) -> None:
        self.violations.append(violation)
        if self.strict:
            raise HotPathViolation(
                f"{violation.category.replace('_', ' ')} on a hot path: {violation}. "
                f"{_HINTS[violation.category]}")

    # -- results ----------------------------------------------------------------
    @property
    def ok(self) -> bool:
        """No breach was recorded."""
        return not self.violations

    def of(self, category: str) -> list[Violation]:
        """The recorded breaches of one category."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}; one of {CATEGORIES}")
        return [v for v in self.violations if v.category == category]

    def allocated(self) -> list:
        """The arrays allocated inside the block (kept alive by the guard)."""
        return [v.array for v in self.violations
                if v.category == "allocation" and isinstance(v.array, wp.array)]

    def report(self) -> str:
        """Every recorded breach, grouped by category, one per line."""
        if not self.violations:
            return "no hot-path contract breach"
        lines = []
        for category in CATEGORIES:
            found = self.of(category)
            if found:
                lines.append(f"{category.replace('_', ' ')} ({len(found)}). "
                             f"{_HINTS[category]}")
                lines += [f"  {v}" for v in found]
        return "\n".join(lines)

    def raise_for_violations(self) -> None:
        """Raise :class:`HotPathViolation` listing every recorded breach, if any."""
        if self.violations:
            raise HotPathViolation(self.report())

    def __repr__(self) -> str:
        state = "active" if self._cm is not None else "idle"
        return (f"HotPathGuard(checks={sorted(self.checks)}, modules={self.modules}, "
                f"violations={len(self.violations)}, {state})")


def hot_path_guard(*, host_sync: bool = True, allocation: bool = True,
                   stream: bool = True, capture: bool = True,
                   modules: Iterable[str] = ("track_gen",),
                   allow: Iterable[tuple] = (), strict: bool = True) -> HotPathGuard:
    """Every hot-path check at once (see :class:`HotPathGuard` for the arguments)."""
    return HotPathGuard(host_sync=host_sync, allocation=allocation, stream=stream,
                        capture=capture, modules=modules, allow=allow, strict=strict)


def no_host_sync(*, allow: Iterable[tuple] = (), strict: bool = True) -> HotPathGuard:
    """No ``wp.synchronize*`` and no ``wp.array.numpy()`` / ``list()`` in the block.

    ``allow``: ``(api, module, function)`` keys let through (and recorded in
    :attr:`HotPathGuard.allowed`), e.g. a known readback scheduled for removal.
    """
    return HotPathGuard(host_sync=True, allocation=False, stream=False, capture=False,
                        allow=allow, strict=strict)


def no_allocation(*, strict: bool = True) -> HotPathGuard:
    """No Warp array allocation in the block."""
    return HotPathGuard(host_sync=False, allocation=True, stream=False, capture=False,
                        strict=strict)


def stream_explicit(modules: Iterable[str] = ("track_gen",), *,
                    strict: bool = True) -> HotPathGuard:
    """Every ``wp.launch`` / ``wp.copy`` from ``modules`` passes ``stream=``, and
    ``modules`` call no ``wp.array.zero_()`` / ``fill_()`` and enter no
    ``wp.ScopedStream``."""
    return HotPathGuard(host_sync=False, allocation=False, stream=True, capture=False,
                        modules=modules, strict=strict)
