"""Gate-native generators backed by point-family samplers."""
from __future__ import annotations

import warp as wp


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
    G = int(config.max_gates)
    dev = str(config.device)
    return PointGateScratch(
        count=wp.empty(E, dtype=wp.int32, device=dev),
        points=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        used=wp.empty(E * P, dtype=wp.int32, device=dev),
        keys=wp.empty(E * G, dtype=wp.float32, device=dev),
    )


def generate_bezier_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate, warp_generate

    P = int(config.max_num_points)
    G = int(config.max_gates)
    warp_generate.corner_count_sample_inplace(seeds_wp, 0, config, scratch.count)
    warp_generate.corner_sample_inplace(seeds_wp, 0, config, scratch.points, scratch.used)
    warp_gate.order_points(
        seeds_wp,
        scratch.points,
        P,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out.position,
    )
    warp_gate.tangents_from_positions(out.position, out.tangent, G, scratch.count)
    wp.copy(out.count, scratch.count)


def generate_hull_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate, warp_generate_hull

    P = int(config.max_num_points)
    G = int(config.max_gates)
    warp_generate_hull.point_count_sample_inplace(seeds_wp, config, scratch.count)
    warp_generate_hull.point_sample_inplace(seeds_wp, config, scratch.points, scratch.used)
    warp_gate.order_points(
        seeds_wp,
        scratch.points,
        P,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out.position,
    )
    warp_gate.tangents_from_positions(out.position, out.tangent, G, scratch.count)
    wp.copy(out.count, scratch.count)


from . import gate_generator_registry as _registry  # noqa: E402

_registry.register(_registry.GateGeneratorSpec(
    name="bezier",
    alloc_scratch=_point_gate_alloc_scratch,
    generate=generate_bezier_gates,
    max_gates=lambda config: int(config.max_num_points),
    supported_orderings=frozenset({"ccw", "random_pairs"}),
))

_registry.register(_registry.GateGeneratorSpec(
    name="hull",
    alloc_scratch=_point_gate_alloc_scratch,
    generate=generate_hull_gates,
    max_gates=lambda config: int(config.max_num_points),
    supported_orderings=frozenset({"ccw", "random_pairs"}),
))
