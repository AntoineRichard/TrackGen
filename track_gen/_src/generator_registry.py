"""Registry of first-stage centerline generators.

A generator is a (name, alloc_scratch, generate) triple. ``config.generator`` selects
one by name. ``TrackGenerator`` resolves it once at construction; the orchestrator calls
the resolved ``generate``. See docs/generator-contract.md for the contract every generator
implements.

This module imports nothing from the package at load time (leaf). Generator modules are
imported lazily in ``_ensure_loaded`` so each self-registers exactly once, with no import
cycle (warp_generate imports warp_pipeline, not this module's body).
"""
from __future__ import annotations

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class GeneratorSpec:
    """One registered generator.

    name:          the ``config.generator`` string that selects it.
    alloc_scratch: ``(config) -> scratch`` — allocate this generator's PRIVATE working
                   buffers ONCE (fixed shapes from config, on config.device). The
                   generation OUTPUT buffers (centerline, valid) are owned by the
                   orchestrator and passed to ``generate``; they are NOT part of this.
    generate:      ``(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None`` —
                   write the closed centerline ([E*num_points] vec2f) into out_centerline
                   and per-env validity ([E] int32) into out_valid_wp, using scratch.
                   Pure Warp, in-place, graph-capturable, zero-alloc, no host sync.
    """
    name: str
    alloc_scratch: Callable
    generate: Callable


GENERATORS: dict[str, GeneratorSpec] = {}
_LOADED = False


def register(spec: GeneratorSpec) -> None:
    GENERATORS[spec.name] = spec


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # Importing each generator module runs its module-level register(...) call.
    # Add one import line per new generator (the only shared touch-point).
    from . import warp_generate  # noqa: F401  (registers "bezier")
    from . import warp_generate_polar  # noqa: F401  (registers "polar")
    from . import warp_generate_hull  # noqa: F401  (registers "hull")
    from . import warp_generate_voronoi  # noqa: F401  (registers "voronoi")
    from . import warp_generate_checkpoint  # noqa: F401  (registers "checkpoint")
    _LOADED = True


def get(name: str) -> GeneratorSpec:
    _ensure_loaded()
    if name not in GENERATORS:
        raise ValueError(
            f"unknown generator {name!r}; available: {sorted(GENERATORS)}"
        )
    return GENERATORS[name]


def available() -> list[str]:
    _ensure_loaded()
    return sorted(GENERATORS)
