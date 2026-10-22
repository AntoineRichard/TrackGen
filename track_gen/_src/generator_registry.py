"""Track-generator spec (canonical home of :class:`GeneratorSpec`) and the deprecated
module-level registry API.

A generator is a (name, alloc_scratch, generate) triple. ``config.generator`` selects
one by name. ``TrackGenerator`` resolves it once at construction through the public
registry, :data:`track_gen.plugin.registry` (kind ``"generator"``); the orchestrator
calls the resolved ``generate``. Built-in generators register explicitly from
``track_gen._src.builtin_plugins``; third parties use the ``"track_gen.plugins"`` entry
point (see :mod:`track_gen.plugin`). See docs/contributing/writing-a-generator.rst for
the contract every generator implements.

Backward-compatibility shims (kept working, deprecated quietly): ``register(spec)``
(silent overwrite; a ``DeprecationWarning`` when it replaces an existing name),
``get(name)``, ``available()``, ``_ensure_loaded()`` and ``GENERATORS`` (a read-only
live view of the registry's generator kind) all delegate to
:data:`track_gen.plugin.registry`.

This module imports nothing from the package at load time (leaf): the registry is
imported lazily, so ``track_gen.plugin`` can re-export ``GeneratorSpec`` from here.
"""
from __future__ import annotations

import dataclasses
import types
from typing import Callable, ClassVar


@dataclasses.dataclass(frozen=True)
class GeneratorSpec:
    """One registered generator.

    name:          the ``config.generator`` string that selects it. Third-party names
                   should be dotted (``"package.name"``).
    alloc_scratch: ``(config) -> scratch`` — allocate this generator's PRIVATE working
                   buffers ONCE (fixed shapes from config, on config.device). The
                   generation OUTPUT buffers (centerline, valid) are owned by the
                   orchestrator and passed to ``generate``; they are NOT part of this.
    generate:      ``(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None`` —
                   write the closed centerline ([E*num_points] vec2f) into out_centerline
                   and per-env validity ([E] int32) into out_valid_wp, using scratch.
                   Pure Warp, in-place, zero-alloc, no host sync WHEN ``capturable=True``.
    capturable:    whether ``generate`` obeys the CUDA-graph-capture contract (pure Warp,
                   no host sync, no per-call allocation). Defaults ``True`` so existing
                   three-arg registrations are unchanged and stay on the capture-then-replay
                   CUDA path. A ``capturable=False`` generator may use host-side loop control
                   and per-call allocation; ``TrackGenerator`` runs it EAGERLY on CUDA every
                   call (the same code path as ``cpu``), never building a ``wp.Graph`` for it.
    accepts_ctx:   ``True`` when ``generate`` accepts a ``ctx=`` keyword (a
                   ``track_gen.exec.LaunchContext``) and emits all of its device work through
                   it; the orchestrator then calls ``generate(..., ctx=ctx)``. Legacy plugins
                   leave it ``False`` and the orchestrator runs ``generate`` inside
                   ``ctx.scope()``, so the ``warp_pipeline`` helpers they call (which default
                   ``ctx=None`` to the ambient context) inherit the call's stream; their own
                   direct ``wp.launch`` calls still go to Warp's current stream.
    params:        optional :class:`track_gen.plugin.Params` subclass (STATIC fields
                   only) declaring this generator's own parameters. Users set them with
                   ``TrackGenConfig(generator=name, generator_params=...)`` (an instance
                   or a mapping of its fields; ``None`` resolves
                   ``params.from_legacy(config)``). ``generate`` / ``alloc_scratch`` read
                   the resolved instance as ``config.generator_params``, and
                   ``registry.describe`` reports its JSON schema. ``None`` (the default):
                   the generator takes no ``generator_params``.
    """

    #: Registry kind ``track_gen.plugin.registry.add(spec)`` infers for this spec type.
    registry_kind: ClassVar[str] = "generator"

    name: str
    alloc_scratch: Callable
    generate: Callable
    capturable: bool = True
    accepts_ctx: bool = False
    params: "type | None" = None


def _registry():
    from ..plugin._registry import registry

    return registry


def register(spec: GeneratorSpec) -> None:
    """Deprecated: use ``track_gen.plugin.registry.add(spec)``.

    Keeps the old semantics (an existing name is overwritten), implemented as
    ``add(spec, replace=True)`` with a ``DeprecationWarning`` when it replaces one.
    """
    _registry()._add_legacy(spec, "generator")


def _ensure_loaded() -> None:
    _registry()._ensure_loaded()


def get(name: str) -> "GeneratorSpec | type":
    """Deprecated alias of ``track_gen.plugin.registry.get("generator", name)``.

    The value may be a :class:`~track_gen.plugin.Stage` CLASS (a Stage plugin), which
    has no ``generate`` / ``alloc_scratch`` / ``capturable``: read capture support from
    ``registry.describe("generator", name).capturable``.
    """
    return _registry().get("generator", name)


def available() -> list[str]:
    """Deprecated alias of ``track_gen.plugin.registry.available("generator")``."""
    return _registry().available("generator")


def __getattr__(name: str):
    # GENERATORS: read-only live view of the registry's generator kind (loads it). Its
    # values may be Stage classes (no generate / alloc_scratch / capturable): enumerators
    # read registry.describe("generator", name).capturable instead.
    if name == "GENERATORS":
        reg = _registry()
        reg._ensure_loaded()
        return types.MappingProxyType(reg._table("generator"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
