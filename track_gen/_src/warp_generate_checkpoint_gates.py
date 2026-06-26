"""Gate-native checkpoint sampling generation."""
from __future__ import annotations

import warp as wp

from .warp_generate_checkpoint import (
    _BASE_RADIUS,
    _BEZIER_EXTENT,
    _sample_checkpoints_k,
)


class CheckpointGateScratch:
    __slots__ = ("checkpoints", "count", "keys")

    def __init__(self, checkpoints, count, keys):
        self.checkpoints = checkpoints
        self.count = count
        self.keys = keys


def checkpoint_gate_alloc_scratch(config):
    from . import warp_gate

    warp_gate._init()
    E = int(config.num_envs)
    C = int(config.checkpoint_count)
    dev = str(config.device)
    count, keys = warp_gate.alloc_order_scratch(config)
    return CheckpointGateScratch(
        checkpoints=wp.empty(E * C, dtype=wp.vec2f, device=dev),
        count=count,
        keys=keys,
    )


def generate_checkpoint_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate

    E = int(config.num_envs)
    C = int(config.checkpoint_count)
    G = int(config.max_gates)
    dev = str(out.position.device)

    radius_min_frac = float(config.checkpoint_radius_min_frac)
    angle_jitter = float(config.checkpoint_angle_jitter)
    target_extent = float(config.scale) * _BEZIER_EXTENT

    wp.launch(warp_gate._fill_i32_k, dim=E, inputs=[scratch.count, C], device=dev)
    wp.launch(
        _sample_checkpoints_k,
        dim=E,
        inputs=[
            seeds_wp,
            1,
            C,
            radius_min_frac,
            angle_jitter,
            _BASE_RADIUS,
            scratch.checkpoints,
        ],
        device=dev,
    )
    warp_gate.finish_ordered_gates(
        seeds_wp,
        scratch.checkpoints,
        C,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out,
        normalize_extent=target_extent,
    )


def register_specs() -> None:
    from . import gate_generator_registry as _registry

    _registry.register(_registry.GateGeneratorSpec(
        name="checkpoint",
        alloc_scratch=checkpoint_gate_alloc_scratch,
        generate=generate_checkpoint_gates,
        max_gates=lambda config: int(config.checkpoint_count),
        supported_orderings=frozenset({"ccw", "raw"}),
    ))


register_specs()
