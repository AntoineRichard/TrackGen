"""Gate-native Voronoi anchor-site generation."""
from __future__ import annotations

import warp as wp

from .warp_generate_voronoi import (
    _TARGET_EXTENT,
    _sample_sites_k,
    _select_anchor_sites_k,
    _voronoi_layout_mode,
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

    warp_gate._init()
    E = int(config.num_envs)
    S = int(config.voronoi_num_sites)
    K = int(config.voronoi_control_points)
    G = int(config.max_gates)
    dev = str(config.device)
    return VoronoiGateScratch(
        sites=wp.empty(E * S, dtype=wp.vec2f, device=dev),
        used=wp.empty(E * S, dtype=wp.int32, device=dev),
        selected=wp.empty(E * K, dtype=wp.vec2f, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        keys=wp.empty(E * G, dtype=wp.float32, device=dev),
    )


def generate_voronoi_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate

    E = int(config.num_envs)
    S = int(config.voronoi_num_sites)
    K = int(config.voronoi_control_points)
    G = int(config.max_gates)
    dev = str(out.position.device)

    layout_mode = _voronoi_layout_mode(getattr(config, "voronoi_site_layout", "void_ring"))
    radial_variation = float(getattr(config, "voronoi_radial_variation", 0.62))
    angular_jitter = float(getattr(config, "voronoi_angular_jitter", 0.08))
    box_size = 2.0
    target_extent = float(config.scale) * _TARGET_EXTENT

    wp.launch(warp_gate._fill_i32_k, dim=E, inputs=[scratch.count, K], device=dev)
    wp.launch(
        _sample_sites_k,
        dim=E,
        inputs=[seeds_wp, S, layout_mode, box_size, scratch.sites],
        device=dev,
    )
    wp.launch(
        _select_anchor_sites_k,
        dim=E,
        inputs=[
            seeds_wp,
            scratch.sites,
            S,
            K,
            layout_mode,
            box_size,
            radial_variation,
            angular_jitter,
            scratch.selected,
            scratch.used,
        ],
        device=dev,
    )
    warp_gate.order_points(
        seeds_wp,
        scratch.selected,
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
        name="voronoi",
        alloc_scratch=voronoi_gate_alloc_scratch,
        generate=generate_voronoi_gates,
        max_gates=lambda config: int(config.voronoi_control_points),
        supported_orderings=frozenset({"ccw", "raw"}),
    ))


register_specs()
