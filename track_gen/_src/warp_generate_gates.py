"""Gate-native generators backed by point-family samplers.

The anchors come from the same public ``track_gen.ops`` samplers (and RNG salts) the
bezier/hull track generators use, so a gate layout reproduces the track generator's
corner/point draw for the same seed. Both are ``gate_generator`` Stages
(``output="anchors"``, roadmap PR11): they write an UNORDERED anchor cloud
(``anchors.xy``, ``max_num_points`` rows per env, and ``anchors.count``); the gate
pipeline orders it (``order.ccw`` by default; ``order.raw`` is excluded: a cloud has no
loop order). Roadmap PR13d: their dynamic ``size`` multiplies the anchor grid after
``scale`` (``ops.sample_grid_points(..., size=)``); ``min_num_points`` stays static (it is
the gate count capacity, ``min_gates``).
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import PointGateParams


class _PointGates(Stage):
    kind = "gate_generator"
    output = "anchors"
    capture = CaptureClass.STATIC
    Params = PointGateParams
    writes = {"xy": "anchors.xy", "count": "anchors.count"}
    excluded_orderings = frozenset({"raw"})
    preferred = {"order": "ccw"}

    # (count salt, count offset, point salt, point offset)
    def _salts(self) -> "tuple[int, int, int, int]":
        raise NotImplementedError

    def capacity(self, params, layout):
        return {"gates": int(params.max_num_points),
                "min_gates": int(params.min_num_points)}

    def build(self, b, p):
        count = b.write("count", wp.int32)
        points = b.write("xy", wp.vec2f, per="anchors")
        P = int(p.max_num_points)
        used = b.scratch("used", wp.int32, b.E * P)
        num_cells, cell_size = ops.grid_cells(p.min_point_distance)
        count_salt, count_offset, point_salt, point_offset = self._salts()
        lo, hi = int(p.min_num_points), int(p.max_num_points)
        scale = float(b.scale)
        seed = b.seed
        size = b.param("size")

        def enqueue(ctx):
            ops.sample_counts(ctx, seed, count_salt, count_offset, lo, hi, count)
            ops.sample_grid_points(ctx, seed, point_salt, point_offset, P, num_cells,
                                   float(cell_size), scale, used, points, size=size)
        return enqueue


class GateBezier(_PointGates):
    """``"bezier"``: the bezier track generator's corner draw as gate anchors."""

    name = "bezier"

    def _salts(self):
        from . import warp_generate

        return warp_generate.CORNER_COUNT_SALT, 0, warp_generate.CORNER_SALT, 0


class GateHull(_PointGates):
    """``"hull"``: the hull track generator's point draw as gate anchors."""

    name = "hull"

    def _salts(self):
        from . import warp_generate_hull as hull

        return (hull.POINT_COUNT_SALT, hull.POINT_SALT_OFFSET, hull.POINT_SALT,
                hull.POINT_SALT_OFFSET)


def register(reg) -> None:
    """Register the built-in ``"bezier"`` and ``"hull"`` gate generators into ``reg``
    (called by ``track_gen._src.builtin_plugins``; importing this module registers
    nothing)."""
    reg.add(GateBezier)
    reg.add(GateHull)
