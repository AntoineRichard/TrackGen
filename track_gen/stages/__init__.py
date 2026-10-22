"""TrackGen's built-in track stages and the channel catalogue (roadmap PR10a/PR10b).

The stages of the four kinds open to plugins (:data:`OPEN_KINDS`: ``relax``, ``width``,
``elevation``, ``validity``) live in :data:`track_gen.plugin.registry`, next to the
generators: :func:`register_builtins` adds TrackGen's own (``relax.xpbd``,
``relax.none``, ``width.constant``, ``elevation.flat`` / ``uniform`` / ``random_walk`` /
``noise``, ``validity.default``) when the registry loads its built-ins. The other kinds
(``resample``, ``frames``, ``offset``, ``lift``) are built-in only: the planner resolves
them from the private catalogue here, keyed ``(kind, name)`` and filled lazily (once,
under a lock) by :func:`register`.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Iterator

from ..plugin.stage import _OPEN_KINDS
from .channels import BUILTIN_FAIL_BITS, CHANNELS, FAIL, SEED, Channel

__all__ = ["BUILTIN_FAIL_BITS", "CHANNELS", "Channel", "FAIL", "OPEN_KINDS", "SEED",
           "available", "get", "register", "register_builtins", "scoped"]

#: The track-pipeline kinds third parties register stages of (through
#: ``track_gen.plugin.registry``).
OPEN_KINDS = _OPEN_KINDS

_RESERVED_KINDS = ("generator", "gate_generator")


def _check_builtin_stage(cls) -> None:
    """The rules of a catalogue stage: a concrete Stage subclass of a kind that is
    neither a generator kind nor open to plugins (those live in the plugin registry),
    STATIC, no ``output`` / ``provides``, a valid Params class, catalogue channels only,
    and a port in both ``reads`` and ``writes`` naming one channel."""
    from ..plugin.params import _check_params_class  # noqa: PLC0415
    from ..plugin.stage import CaptureClass, Stage  # noqa: PLC0415

    if not (isinstance(cls, type) and issubclass(cls, Stage)):
        raise TypeError(f"not a Stage subclass: {cls!r}")
    label = cls.__name__
    if getattr(cls, "__abstractmethods__", None):
        raise TypeError(f"Stage {label} is abstract (implement "
                        f"{sorted(cls.__abstractmethods__)})")
    kind, name = getattr(cls, "kind", None), getattr(cls, "name", None)
    if not isinstance(kind, str) or not kind or kind in _RESERVED_KINDS:
        raise ValueError(f"Stage {label}.kind must be a non-generator kind, got {kind!r}")
    if kind in OPEN_KINDS:
        raise ValueError(f"Stage {label}: kind {kind!r} is open to plugins: register it "
                         f"with track_gen.plugin.registry.add")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Stage {label}.name must be a non-empty string, got {name!r}")
    _check_params_class(cls.Params, owner=f"Stage {label}")
    if cls.capture is not CaptureClass.STATIC:
        raise ValueError(f"Stage {label}: a built-in stage is STATIC")
    if cls.output is not None or frozenset(cls.provides):
        raise ValueError(f"Stage {label}: output / provides are not used by track stages")
    reads, writes = dict(cls.reads), dict(cls.writes)
    bad = sorted(str(c) for c in (*reads.values(), *writes.values()) if c not in CHANNELS)
    if bad:
        raise ValueError(f"Stage {label}: unknown channels {bad}; channels: "
                         f"{sorted(CHANNELS)}")
    if SEED in writes.values():
        raise ValueError(f"Stage {label} writes {SEED!r}: only the Program's prologue does")
    clash = sorted(p for p in set(reads) & set(writes) if reads[p] != writes[p])
    if clash:
        raise ValueError(f"Stage {label}: ports {clash} are both read and write ports of "
                         f"different channels")


class _Catalogue:
    """``(kind, name) -> Stage class``, filled lazily by :func:`register`."""

    def __init__(self) -> None:
        self._table: dict = {}
        self._loaded = False
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                register(self)
                self._loaded = True

    def add(self, cls, *, replace: bool = False) -> None:
        """Add a built-in stage class (checked by :func:`_check_builtin_stage`)."""
        _check_builtin_stage(cls)
        with self._lock:
            key = (cls.kind, cls.name)
            if key in self._table and not replace:
                raise ValueError(f"{cls.kind} stage {cls.name!r} is already in the "
                                 f"catalogue")
            self._table[key] = cls

    def get(self, kind: str, name: str):
        """The stage class, or ``None``."""
        self._ensure_loaded()
        return self._table.get((kind, name))

    def available(self, kind: "str | None" = None) -> list:
        """The kinds (``kind=None``) or the names of one kind, sorted."""
        self._ensure_loaded()
        if kind is None:
            return sorted({k for k, _ in self._table})
        return sorted(n for k, n in self._table if k == kind)

    @contextlib.contextmanager
    def scoped(self) -> Iterator["_Catalogue"]:
        """Restore the catalogue on exit (tests that inject a toy stage)."""
        self._ensure_loaded()
        with self._lock:
            saved = dict(self._table)
        try:
            yield self
        finally:
            with self._lock:
                self._table = saved


_catalogue = _Catalogue()
get = _catalogue.get
available = _catalogue.available
scoped = _catalogue.scoped


def register(catalogue) -> None:
    """Add the built-in stages of the kinds NOT open to plugins (resample, frames,
    offset, lift) to ``catalogue`` (anything with ``add(cls)``)."""
    from .track import CATALOGUE_STAGES  # noqa: PLC0415

    for cls in CATALOGUE_STAGES:
        catalogue.add(cls)


def register_builtins(reg) -> None:
    """Add TrackGen's relax / width / elevation / validity stages to the plugin
    registry ``reg`` (called by its built-in load, before any third-party plugin)."""
    from .track import REGISTRY_STAGES  # noqa: PLC0415

    for cls in REGISTRY_STAGES:
        reg.add(cls)
