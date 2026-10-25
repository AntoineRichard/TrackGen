"""Gate-native Voronoi anchor sites (a ``gate_generator`` Stage, roadmap PR11).

``voronoi_control_points`` anchor sites selected from the Voronoi site field (the
voronoi track generator's draw), in loop order (``anchors.xy``); the gate pipeline
orders them (``order.ccw`` by default, ``order.raw`` keeps the loop order;
``order.random_pairs`` is excluded) and normalizes their extent
(``preferred["normalize"] = "extent"``). The radial variation and angular jitter are
dynamic (roadmap PR13c): per-env arrays read by
:func:`~track_gen._src.warp_generate_voronoi.select_voronoi_anchors_per_env`, clamped
to the declared bounds at use.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import VoronoiGateParams
from .warp_generate_voronoi import (
    BOX_SIZE,
    sample_voronoi_sites,
    select_voronoi_anchors_per_env,
    voronoi_layout_mode,
)


class GateVoronoi(Stage):
    """``"voronoi"``: the voronoi track generator's anchor sites as gate anchors."""

    kind = "gate_generator"
    name = "voronoi"
    output = "anchors"
    capture = CaptureClass.STATIC
    Params = VoronoiGateParams
    writes = {"xy": "anchors.xy", "count": "anchors.count"}
    excluded_orderings = frozenset({"random_pairs"})
    preferred = {"order": "ccw", "normalize": "extent"}

    def capacity(self, params, layout):
        K = int(params.voronoi_control_points)
        return {"gates": K, "min_gates": K}

    def build(self, b, p):
        count = b.write("count", wp.int32)
        selected = b.write("xy", wp.vec2f, per="anchors")
        S = int(p.voronoi_num_sites)
        K = int(p.voronoi_control_points)
        sites = b.scratch("sites", wp.vec2f, b.E * S)
        used = b.scratch("used", wp.int32, b.E * S)
        layout_mode = voronoi_layout_mode(p.voronoi_site_layout)
        radial_variation = b.param("voronoi_radial_variation")
        angular_jitter = b.param("voronoi_angular_jitter")
        seed = b.seed

        def enqueue(ctx):
            ops.fill(ctx, count, K)
            sample_voronoi_sites(ctx, seed, S, layout_mode, BOX_SIZE, sites)
            select_voronoi_anchors_per_env(ctx, seed, sites, S, K, layout_mode, BOX_SIZE,
                                           radial_variation, angular_jitter, selected, used)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"voronoi"`` gate generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    reg.add(GateVoronoi)
