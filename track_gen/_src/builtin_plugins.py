"""Explicit registration of the generators and stages shipped with track_gen.

:data:`track_gen.plugin.registry` calls :func:`register` once, on its first lookup and
before any third-party entry point, so the built-ins never need package metadata (a
source checkout works) and always win name collisions unless a plugin passes
``replace=True``. Importing a generator module has no registration side effect: each
one exposes a ``register(reg)`` function, called from here.
"""
from __future__ import annotations


def register(reg) -> None:
    """Register every built-in track generator, gate generator and track-pipeline stage
    (relax / width / elevation / validity) into ``reg``."""
    from . import (
        warp_generate,
        warp_generate_checkpoint,
        warp_generate_checkpoint_gates,
        warp_generate_gates,
        warp_generate_hull,
        warp_generate_polar,
        warp_generate_polar_gates,
        warp_generate_repulsive,
        warp_generate_voronoi,
        warp_generate_voronoi_gates,
    )

    # Track generators (kind "generator").
    warp_generate.register(reg)              # "bezier"
    warp_generate_polar.register(reg)        # "polar"
    warp_generate_hull.register(reg)         # "hull"
    warp_generate_voronoi.register(reg)      # "voronoi"
    warp_generate_checkpoint.register(reg)   # "checkpoint"
    warp_generate_repulsive.register(reg)    # "repulsive"
    # Roadmap PR13b: ingestion of caller-provided centerlines, built from the public
    # plugin surface only (the same register(reg) a third party writes).
    from ..stages import external

    external.register(reg)                   # "external"

    # Gate generators (kind "gate_generator").
    warp_generate_gates.register(reg)              # "bezier", "hull"
    warp_generate_polar_gates.register(reg)        # "polar"
    warp_generate_voronoi_gates.register(reg)      # "voronoi"
    warp_generate_checkpoint_gates.register(reg)   # "checkpoint"

    # The track-pipeline stages open to plugins (kinds "relax", "width", "elevation",
    # "validity"; roadmap PR10b).
    from ..stages import register_builtins

    register_builtins(reg)
