"""Gate-native generators backed by point-family samplers.

The anchors come from the same public ``track_gen.ops`` samplers (and RNG salts) the
bezier/hull track generators use, so a gate layout reproduces the track generator's
corner/point draw for the same seed.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from .runtime import _init


class PointGateScratch:
    __slots__ = ("count", "points", "used", "keys")

    def __init__(self, count, points, used, keys):
        self.count = count
        self.points = points
        self.used = used
        self.keys = keys


def _point_gate_alloc_scratch(config):
    from . import warp_gate

    _init()
    E = int(config.num_envs)
    P = int(config.max_num_points)
    dev = str(config.device)
    count, keys = warp_gate.alloc_order_scratch(config)
    return PointGateScratch(
        count=count,
        points=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        used=wp.empty(E * P, dtype=wp.int32, device=dev),
        keys=keys,
    )


def generate_bezier_gates(seeds_wp, config, out, scratch,
                          ctx: "LaunchContext | None" = None) -> None:
    """Bezier corner sampling -> ordered gate anchors in ``out``.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    from . import warp_gate, warp_generate

    ctx = _resolve_ctx(ctx, str(out.position.device))
    P = int(config.max_num_points)
    G = int(config.max_gates)
    num_cells, cell_size = ops.grid_cells(config.min_point_distance)
    ops.sample_counts(ctx, seeds_wp, warp_generate.CORNER_COUNT_SALT, 0,
                      int(config.min_num_points), int(config.max_num_points), scratch.count)
    ops.sample_grid_points(ctx, seeds_wp, warp_generate.CORNER_SALT, 0, P, num_cells,
                           float(cell_size), float(config.scale), scratch.used,
                           scratch.points)
    warp_gate.finish_ordered_gates(
        seeds_wp,
        scratch.points,
        P,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out,
        ctx=ctx,
    )


def generate_hull_gates(seeds_wp, config, out, scratch,
                        ctx: "LaunchContext | None" = None) -> None:
    """Hull random-point sampling -> ordered gate anchors in ``out``.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    from . import warp_gate, warp_generate_hull as hull

    ctx = _resolve_ctx(ctx, str(out.position.device))
    P = int(config.max_num_points)
    G = int(config.max_gates)
    num_cells, cell_size = ops.grid_cells(config.min_point_distance)
    ops.sample_counts(ctx, seeds_wp, hull.POINT_COUNT_SALT, hull.POINT_SALT_OFFSET,
                      int(config.min_num_points), int(config.max_num_points), scratch.count)
    ops.sample_grid_points(ctx, seeds_wp, hull.POINT_SALT, hull.POINT_SALT_OFFSET, P,
                           num_cells, float(cell_size), float(config.scale), scratch.used,
                           scratch.points)
    warp_gate.finish_ordered_gates(
        seeds_wp,
        scratch.points,
        P,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out,
        ctx=ctx,
    )


def _max_point_gates(config) -> int:
    return int(config.max_num_points)


def register(reg) -> None:
    """Register the built-in ``"bezier"`` and ``"hull"`` gate generators into ``reg``
    (called by ``track_gen._src.builtin_plugins``; importing this module registers
    nothing)."""
    from .gate_generator_registry import GateGeneratorSpec

    reg.add(GateGeneratorSpec(
        name="bezier",
        alloc_scratch=_point_gate_alloc_scratch,
        generate=generate_bezier_gates,
        max_gates=_max_point_gates,
        supported_orderings=frozenset({"ccw", "random_pairs"}),
        accepts_ctx=True,
    ))

    reg.add(GateGeneratorSpec(
        name="hull",
        alloc_scratch=_point_gate_alloc_scratch,
        generate=generate_hull_gates,
        max_gates=_max_point_gates,
        supported_orderings=frozenset({"ccw", "random_pairs"}),
        accepts_ctx=True,
    ))
