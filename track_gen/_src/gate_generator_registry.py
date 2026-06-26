"""Registry of shipped native gate sequence generators.

The registry loads the gate generator modules lazily so importing ``track_gen`` stays
cheap, while construction and discovery surface real import errors from shipped modules.
"""
from __future__ import annotations

import dataclasses
import importlib
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

_GATE_GENERATOR_MODULES = (
    "track_gen._src.warp_generate_gates",
    "track_gen._src.warp_generate_polar_gates",
    "track_gen._src.warp_generate_voronoi_gates",
    "track_gen._src.warp_generate_checkpoint_gates",
)


def _spec_module(spec: GateGeneratorSpec) -> str | None:
    for attr in ("generate", "alloc_scratch", "max_gates"):
        module = getattr(getattr(spec, attr), "__module__", None)
        if module:
            return module
    return None


def register(spec: GateGeneratorSpec) -> None:
    existing = GATE_GENERATORS.get(spec.name)
    if existing is not None:
        existing_module = _spec_module(existing)
        spec_module = _spec_module(spec)
        if existing_module != spec_module:
            raise ValueError(
                f"gate generator {spec.name!r} is already registered by "
                f"{existing_module!r}"
            )
    GATE_GENERATORS[spec.name] = spec


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return

    for module_name in _GATE_GENERATOR_MODULES:
        module = importlib.import_module(module_name)
        register_specs = getattr(module, "register_specs", None)
        if register_specs is not None:
            register_specs()

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
