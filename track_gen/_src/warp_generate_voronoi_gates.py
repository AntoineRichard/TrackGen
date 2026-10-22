"""Gate-native Voronoi anchor-site generation."""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.constants import TARGET_EXTENT
from .runtime import _init
from .warp_generate_voronoi import (
    BOX_SIZE,
    sample_voronoi_sites,
    select_voronoi_anchors,
    voronoi_layout_mode,
)


class VoronoiGateScratch:
    __slots__ = ("sites", "used", "selected", "count", "keys")

    def __init__(self, sites, used, selected, count, keys):
        self.sites = sites
        self.used = used
        self.selected = selected
        self.count = count
        self.keys = keys


def voronoi_gate_alloc_scratch(config):
    from . import warp_gate

    _init()
    E = int(config.num_envs)
    S = int(config.voronoi_num_sites)
    K = int(config.voronoi_control_points)
    dev = str(config.device)
    count, keys = warp_gate.alloc_order_scratch(config)
    return VoronoiGateScratch(
        sites=wp.empty(E * S, dtype=wp.vec2f, device=dev),
        used=wp.empty(E * S, dtype=wp.int32, device=dev),
        selected=wp.empty(E * K, dtype=wp.vec2f, device=dev),
        count=count,
        keys=keys,
    )


def generate_voronoi_gates(seeds_wp, config, out, scratch,
                           ctx: "LaunchContext | None" = None) -> None:
    """Voronoi anchor sites -> ordered gate anchors in ``out``.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    from . import warp_gate

    S = int(config.voronoi_num_sites)
    K = int(config.voronoi_control_points)
    G = int(config.max_gates)
    dev = str(out.position.device)
    ctx = _resolve_ctx(ctx, dev)

    layout_mode = voronoi_layout_mode(getattr(config, "voronoi_site_layout", "void_ring"))
    radial_variation = float(getattr(config, "voronoi_radial_variation", 0.62))
    angular_jitter = float(getattr(config, "voronoi_angular_jitter", 0.08))
    target_extent = float(config.scale) * TARGET_EXTENT

    ops.fill(ctx, scratch.count, K)
    sample_voronoi_sites(ctx, seeds_wp, S, layout_mode, BOX_SIZE, scratch.sites)
    select_voronoi_anchors(ctx, seeds_wp, scratch.sites, S, K, layout_mode, BOX_SIZE,
                           radial_variation, angular_jitter, scratch.selected, scratch.used)
    warp_gate.finish_ordered_gates(
        seeds_wp,
        scratch.selected,
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
    """Register the built-in ``"voronoi"`` gate generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .builtin_params import VoronoiGateParams
    from .gate_generator_registry import GateGeneratorSpec

    reg.add(GateGeneratorSpec(
        name="voronoi",
        alloc_scratch=voronoi_gate_alloc_scratch,
        generate=generate_voronoi_gates,
        max_gates=lambda config: int(config.voronoi_control_points),
        supported_orderings=frozenset({"ccw", "raw"}),
        accepts_ctx=True,
        params=VoronoiGateParams,
    ))
