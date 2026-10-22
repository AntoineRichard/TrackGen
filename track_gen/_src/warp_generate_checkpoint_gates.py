"""Gate-native checkpoint sampling generation."""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.constants import TARGET_EXTENT
from .runtime import _init
from .warp_generate_checkpoint import sample_checkpoint_rings


class CheckpointGateScratch:
    __slots__ = ("checkpoints", "count", "keys")

    def __init__(self, checkpoints, count, keys):
        self.checkpoints = checkpoints
        self.count = count
        self.keys = keys


def checkpoint_gate_alloc_scratch(config):
    from . import warp_gate

    _init()
    E = int(config.num_envs)
    C = int(config.checkpoint_count)
    dev = str(config.device)
    count, keys = warp_gate.alloc_order_scratch(config)
    return CheckpointGateScratch(
        checkpoints=wp.empty(E * C, dtype=wp.vec2f, device=dev),
        count=count,
        keys=keys,
    )


def generate_checkpoint_gates(seeds_wp, config, out, scratch,
                              ctx: "LaunchContext | None" = None) -> None:
    """Checkpoint sampling -> ordered gate anchors in ``out``.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    from . import warp_gate

    C = int(config.checkpoint_count)
    G = int(config.max_gates)
    dev = str(out.position.device)
    ctx = _resolve_ctx(ctx, dev)

    radius_min_frac = float(config.checkpoint_radius_min_frac)
    angle_jitter = float(config.checkpoint_angle_jitter)
    target_extent = float(config.scale) * TARGET_EXTENT

    ops.fill(ctx, scratch.count, C)
    # One candidate ring per env (K = 1): the track generator's checkpoint draw.
    sample_checkpoint_rings(ctx, seeds_wp, 1, C, radius_min_frac, angle_jitter,
                            scratch.checkpoints)
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
        ctx=ctx,
    )


def register(reg) -> None:
    """Register the built-in ``"checkpoint"`` gate generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .builtin_params import CheckpointGateParams
    from .gate_generator_registry import GateGeneratorSpec

    reg.add(GateGeneratorSpec(
        name="checkpoint",
        alloc_scratch=checkpoint_gate_alloc_scratch,
        generate=generate_checkpoint_gates,
        max_gates=lambda config: int(config.checkpoint_count),
        supported_orderings=frozenset({"ccw", "raw"}),
        accepts_ctx=True,
        params=CheckpointGateParams,
    ))
