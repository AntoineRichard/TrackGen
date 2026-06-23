"""Gate-native polar control-knot generation."""
from __future__ import annotations

import warp as wp

from .warp_generate_polar import (
    _BASE_RADIUS,
    _BEZIER_EXTENT,
    _polar_controls_k,
    _polar_num_knots,
)


class PolarGateScratch:
    __slots__ = ("controls", "count", "keys")

    def __init__(self, controls, count, keys):
        self.controls = controls
        self.count = count
        self.keys = keys


def polar_gate_alloc_scratch(config):
    from . import warp_gate

    warp_gate._init()
    E = int(config.num_envs)
    K = _polar_num_knots(config)
    G = int(config.max_gates)
    dev = str(config.device)
    return PolarGateScratch(
        controls=wp.empty(E * K, dtype=wp.vec2f, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        keys=wp.empty(E * G, dtype=wp.float32, device=dev),
    )


def generate_polar_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate

    E = int(config.num_envs)
    K = _polar_num_knots(config)
    G = int(config.max_gates)
    dev = str(out.position.device)

    radial_default = 0.60 * float(getattr(config, "amplitude", 1.0))
    radial_jitter = min(
        max(float(getattr(config, "polar_radial_jitter", radial_default)), 0.0),
        0.85,
    )
    angular_jitter = min(
        max(float(getattr(config, "polar_angular_jitter", 0.30)), 0.0),
        0.45,
    )
    target_extent = float(config.scale) * _BEZIER_EXTENT

    wp.launch(warp_gate._fill_i32_k, dim=E, inputs=[scratch.count, K], device=dev)
    wp.launch(
        _polar_controls_k,
        dim=E,
        inputs=[
            seeds_wp,
            K,
            radial_jitter,
            angular_jitter,
            _BASE_RADIUS,
            scratch.controls,
        ],
        device=dev,
    )
    warp_gate.order_points(
        seeds_wp,
        scratch.controls,
        K,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out.position,
    )
    warp_gate.normalize_positions(out.position, G, scratch.count, target_extent)
    warp_gate.tangents_from_positions(out.position, out.tangent, G, scratch.count)
    wp.copy(out.count, scratch.count)


def register_specs() -> None:
    from . import gate_generator_registry as _registry

    _registry.register(_registry.GateGeneratorSpec(
        name="polar",
        alloc_scratch=polar_gate_alloc_scratch,
        generate=generate_polar_gates,
        max_gates=_polar_num_knots,
        supported_orderings=frozenset({"ccw", "raw"}),
    ))


register_specs()
