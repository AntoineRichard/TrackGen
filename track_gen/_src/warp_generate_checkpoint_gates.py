"""Gate-native checkpoint sampling (a ``gate_generator`` Stage, roadmap PR11).

One ring of ``checkpoint_count`` radial checkpoints (the checkpoint track generator's
draw), in loop order (``anchors.xy``); the gate pipeline orders them (``order.ccw`` by
default, ``order.raw`` keeps the loop order; ``order.random_pairs`` is excluded) and
normalizes their extent (``preferred["normalize"] = "extent"``).
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import CheckpointGateParams
from .warp_generate_checkpoint import sample_checkpoint_rings


class GateCheckpoint(Stage):
    """``"checkpoint"``: the checkpoint track generator's ring as gate anchors."""

    kind = "gate_generator"
    name = "checkpoint"
    output = "anchors"
    capture = CaptureClass.STATIC
    Params = CheckpointGateParams
    writes = {"xy": "anchors.xy", "count": "anchors.count"}
    excluded_orderings = frozenset({"random_pairs"})
    preferred = {"order": "ccw", "normalize": "extent"}

    def capacity(self, params, layout):
        C = int(params.checkpoint_count)
        return {"gates": C, "min_gates": C}

    def build(self, b, p):
        count = b.write("count", wp.int32)
        checkpoints = b.write("xy", wp.vec2f, per="anchors")
        C = int(p.checkpoint_count)
        radius_min_frac = float(p.checkpoint_radius_min_frac)
        angle_jitter = float(p.checkpoint_angle_jitter)
        seed = b.seed

        def enqueue(ctx):
            ops.fill(ctx, count, C)
            # One candidate ring per env (K = 1): the track generator's checkpoint draw.
            sample_checkpoint_rings(ctx, seed, 1, C, radius_min_frac, angle_jitter,
                                    checkpoints)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"checkpoint"`` gate generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    reg.add(GateCheckpoint)
