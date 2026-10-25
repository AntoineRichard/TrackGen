"""Gate-generator spec (canonical home of :class:`GateGeneratorSpec`) and the
deprecated module-level registry API.

``GateGenerator`` resolves ``config.generator`` once at construction through the public
registry, :data:`track_gen.plugin.registry` (kind ``"gate_generator"``). Built-in gate
generators register explicitly from ``track_gen._src.builtin_plugins``; third parties
use the ``"track_gen.plugins"`` entry point (see :mod:`track_gen.plugin`). Loading is
lazy, so importing ``track_gen`` stays cheap.

Backward-compatibility shims (kept working, deprecated quietly): ``register(spec)``
(the old rule: a name registered from another module raises, now as
``DuplicateSpecError`` (a ``ValueError``); a spec from the same module overwrites it, with
a ``DeprecationWarning``), ``get(name)``, ``available()``, ``_ensure_loaded()`` and
``GATE_GENERATORS`` (a read-only live view of the registry's gate-generator kind) all
delegate to :data:`track_gen.plugin.registry`.
"""
from __future__ import annotations

import dataclasses
import types
from typing import Callable, ClassVar


@dataclasses.dataclass(frozen=True)
class GateGeneratorSpec:
    """One registered native gate generator (the legacy gate generator ABI).

    Deprecated: write a Stage with ``kind="gate_generator"`` and
    ``output="anchors"`` (:ref:`writing-a-gate-generator`). A
    ``GateGeneratorSpec`` runs through an adapter stage until 1.0, and binding one
    emits a ``DeprecationWarning``. Its ``max_gates`` callable is read only as the
    adapter's capacity (the most gates ``generate`` writes, checked against
    ``GateGenConfig.min_gates`` / ``max_gates``).

    capturable: whether ``generate`` obeys the CUDA-graph-capture contract (pure Warp,
                no host sync, no per-call allocation). Defaults ``True``. A
                ``capturable=False`` generator may use host-side loop control and
                per-call allocation; ``GateGenerator`` runs it EAGERLY on CUDA every
                call (the same code path as ``cpu``), never building a ``wp.Graph``.
                Mirrors ``GeneratorSpec.capturable``.
    accepts_ctx: ``True`` when ``generate`` accepts a ``ctx=`` keyword (a
                ``track_gen.exec.LaunchContext``) and emits all of its device work
                through it; the gate Program's legacy adapter stage
                (``track_gen.stages.gates._LegacyGateSlot``) then calls
                ``generate(seeds_wp, config, out, scratch, ctx=ctx)``. Legacy plugins
                leave it ``False`` and the adapter runs ``generate`` inside
                ``ctx.scope()``, so the ``warp_gate`` / ``warp_pipeline`` helpers they
                call (which default ``ctx=None`` to the ambient context) inherit the
                call's stream; their own direct ``wp.launch`` calls still go to Warp's
                current stream. Mirrors ``GeneratorSpec.accepts_ctx``.
    params:     optional :class:`track_gen.plugin.Params` subclass (static fields only)
                set through ``GateGenConfig.generator_params``; read as
                ``config.generator_params`` by ``generate`` / ``alloc_scratch`` /
                ``max_gates``. Mirrors ``GeneratorSpec.params``.
    """

    #: Registry kind ``track_gen.plugin.registry.add(spec)`` infers for this spec type.
    registry_kind: ClassVar[str] = "gate_generator"

    name: str
    alloc_scratch: Callable
    generate: Callable
    max_gates: Callable
    supported_orderings: frozenset[str]
    capturable: bool = True
    accepts_ctx: bool = False
    params: "type | None" = None


def _registry():
    from ..plugin._registry import registry

    return registry


def register(spec: GateGeneratorSpec) -> None:
    """Deprecated: use ``track_gen.plugin.registry.add(spec)``.

    Keeps the old gate-registry rule: an existing name registered from another module
    raises :class:`~track_gen.plugin.DuplicateSpecError` (a ``ValueError``, as before);
    a spec from the same module overwrites it, with a ``DeprecationWarning``.
    """
    _registry()._add_legacy(spec, "gate_generator", same_module_only=True)


def _ensure_loaded() -> None:
    _registry()._ensure_loaded()


def available() -> list[str]:
    """Deprecated alias of ``track_gen.plugin.registry.available("gate_generator")``."""
    return _registry().available("gate_generator")


def get(name: str) -> GateGeneratorSpec:
    """Deprecated alias of ``track_gen.plugin.registry.get("gate_generator", name)``."""
    return _registry().get("gate_generator", name)


def __getattr__(name: str):
    # GATE_GENERATORS: read-only live view of the registry's gate-generator kind.
    if name == "GATE_GENERATORS":
        reg = _registry()
        reg._ensure_loaded()
        return types.MappingProxyType(reg._table("gate_generator"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
