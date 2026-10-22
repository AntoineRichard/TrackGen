"""Gate-native generators backed by point-family samplers."""
from __future__ import annotations

import warp as wp

from ..exec.context import LaunchContext, resolve as _resolve_ctx


class PointGateScratch:
    __slots__ = ("count", "points", "used", "keys")

    def __init__(self, count, points, used, keys):
        self.count = count
        self.points = points
        self.used = used
        self.keys = keys


def _point_gate_alloc_scratch(config):
    from . import warp_gate

    warp_gate._init()
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
    warp_generate.corner_count_sample_inplace(seeds_wp, 0, config, scratch.count, ctx=ctx)
    warp_generate.corner_sample_inplace(seeds_wp, 0, config, scratch.points, scratch.used,
                                        ctx=ctx)
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
    from . import warp_gate, warp_generate_hull

    ctx = _resolve_ctx(ctx, str(out.position.device))
    P = int(config.max_num_points)
    G = int(config.max_gates)
    warp_generate_hull.point_count_sample_inplace(seeds_wp, config, scratch.count, ctx=ctx)
    warp_generate_hull.point_sample_inplace(seeds_wp, config, scratch.points, scratch.used,
                                            ctx=ctx)
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
