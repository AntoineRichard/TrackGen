"""Registry of native gate sequence generators.

Gate generator modules are optional at this stage of the implementation plan. The
registry therefore probes for future modules before importing them; if a module
exists, its import is allowed to fail normally so real implementation errors are
visible.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
from typing import Callable


@dataclasses.dataclass(frozen=True)
class GateGeneratorSpec:
    """One registered native gate generator."""

    name: str
    alloc_scratch: Callable
    generate: Callable
    max_gates: Callable
    supported_orderings: frozenset[str]


GATE_GENERATORS: dict[str, GateGeneratorSpec] = {}
_LOADED = False

_OPTIONAL_GATE_GENERATOR_MODULES = (
    "track_gen._src.warp_generate_gates",
    "track_gen._src.warp_gate_generate",
    "track_gen._src.warp_gate_generate_polar",
    "track_gen._src.warp_gate_generate_hull",
    "track_gen._src.warp_gate_generate_voronoi",
    "track_gen._src.warp_gate_generate_checkpoint",
    "track_gen._src.warp_gate_bezier",
    "track_gen._src.warp_gate_polar",
    "track_gen._src.warp_gate_hull",
    "track_gen._src.warp_gate_voronoi",
    "track_gen._src.warp_gate_checkpoint",
)


def register(spec: GateGeneratorSpec) -> None:
    if spec.name in GATE_GENERATORS:
        raise ValueError(f"gate generator {spec.name!r} is already registered")
    GATE_GENERATORS[spec.name] = spec


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return

    for module_name in _OPTIONAL_GATE_GENERATOR_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            importlib.import_module(module_name)

    _LOADED = True


def available() -> list[str]:
    _ensure_loaded()
    return sorted(GATE_GENERATORS)


def get(name: str) -> GateGeneratorSpec:
    _ensure_loaded()
    if name not in GATE_GENERATORS:
        raise ValueError(
            f"unknown gate generator {name!r}; available: {sorted(GATE_GENERATORS)}"
        )
    return GATE_GENERATORS[name]
