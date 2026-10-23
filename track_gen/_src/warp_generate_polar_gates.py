"""Gate-native polar control-knot generation (a ``gate_generator`` Stage, roadmap PR11).

The ``polar_num_knots`` control knots of the polar track generator, in loop order
(``anchors.xy``); the gate pipeline orders them (``order.ccw`` by default, ``order.raw``
keeps the loop order; ``order.random_pairs`` is excluded) and normalizes their extent
(``preferred["normalize"] = "extent"``).
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import PolarGateParams
from .warp_generate_polar import polar_jitters, polar_num_knots, sample_polar_controls


class GatePolar(Stage):
    """``"polar"``: the polar track generator's control knots as gate anchors."""

    kind = "gate_generator"
    name = "polar"
    output = "anchors"
    capture = CaptureClass.STATIC
    Params = PolarGateParams
    writes = {"xy": "anchors.xy", "count": "anchors.count"}
    excluded_orderings = frozenset({"random_pairs"})
    preferred = {"order": "ccw", "normalize": "extent"}

    def capacity(self, params, layout):
        K = polar_num_knots(params)
        return {"gates": K, "min_gates": K}

    def build(self, b, p):
        count = b.write("count", wp.int32)
        controls = b.write("xy", wp.vec2f, per="anchors")
        K = polar_num_knots(p)
        # Same clamped jitters as the polar track generator.
        radial_jitter, angular_jitter = polar_jitters(p)
        seed = b.seed

        def enqueue(ctx):
            ops.fill(ctx, count, K)
            sample_polar_controls(ctx, seed, K, radial_jitter, angular_jitter, controls)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"polar"`` gate generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    reg.add(GatePolar)
