"""Gate-native polar control-knot generation."""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.constants import TARGET_EXTENT
from .runtime import _init
from .warp_generate_polar import polar_jitters, polar_num_knots, sample_polar_controls


class PolarGateScratch:
    __slots__ = ("controls", "count", "keys")

    def __init__(self, controls, count, keys):
        self.controls = controls
        self.count = count
        self.keys = keys


def polar_gate_alloc_scratch(config):
    from . import warp_gate

    _init()
    E = int(config.num_envs)
    K = polar_num_knots(config)
    dev = str(config.device)
    count, keys = warp_gate.alloc_order_scratch(config)
    return PolarGateScratch(
        controls=wp.empty(E * K, dtype=wp.vec2f, device=dev),
        count=count,
        keys=keys,
    )


def generate_polar_gates(seeds_wp, config, out, scratch,
                         ctx: "LaunchContext | None" = None) -> None:
    """Polar control knots -> ordered gate anchors in ``out``.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    from . import warp_gate

    K = polar_num_knots(config)
    G = int(config.max_gates)
    dev = str(out.position.device)
    ctx = _resolve_ctx(ctx, dev)

    # Same clamped jitters as the polar track generator.
    radial_jitter, angular_jitter = polar_jitters(config)
    target_extent = float(config.scale) * TARGET_EXTENT

    ops.fill(ctx, scratch.count, K)
    sample_polar_controls(ctx, seeds_wp, K, radial_jitter, angular_jitter, scratch.controls)
    warp_gate.finish_ordered_gates(
        seeds_wp,
        scratch.controls,
        K,
        scratch.count,
        G,
        str(config.gate_ordering),
        scratch.keys,
        out,
        normalize_extent=target_extent,
        ctx=ctx,
    )


def register(reg) -> None:
    """Register the built-in ``"polar"`` gate generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .gate_generator_registry import GateGeneratorSpec

    reg.add(GateGeneratorSpec(
        name="polar",
        alloc_scratch=polar_gate_alloc_scratch,
        generate=generate_polar_gates,
        max_gates=polar_num_knots,
        supported_orderings=frozenset({"ccw", "raw"}),
        accepts_ctx=True,
    ))
