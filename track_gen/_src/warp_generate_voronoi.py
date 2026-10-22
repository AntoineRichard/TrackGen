"""Pure-Warp Voronoi/graph-cycle centerline generator.

Registered as ``config.generator="voronoi"``.

This is the production distillation of the host-side Voronoi/random-geometric prototype:
sample a bounded field of cell sites, choose a fixed number of angular anchor targets,
snap each target to a nearby unused site, smooth the resulting graph-cycle polyline, and
arc-length resample it into the standard centerline buffer.

The implementation deliberately avoids exact Voronoi ridge traversal. Exact Delaunay /
Voronoi construction and dynamic cycle-basis search are not a good fit for the runtime
contract. The fixed site field still gives the useful "right number of cells" control,
while the anchor-snapping cycle is static-shape, allocation-free, and CUDA-graph safe.

The spline evaluation, arc resample, extent normalisation, self-intersection count,
polygon fallback and fill are the public ``track_gen.ops`` primitives; the site field and
anchor selection are exposed as :func:`sample_voronoi_sites` /
:func:`select_voronoi_anchors` (shared with the voronoi gate generator).
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops.constants import TARGET_EXTENT
from .runtime import _init

# Legacy private name (value only): the loop is normalised to config.scale * TARGET_EXTENT.
_TARGET_EXTENT = TARGET_EXTENT

#: Side of the square site field (centred at the origin) the built-in generator samples.
BOX_SIZE = 2.0

_SITE_SALT = 8171
_ANCHOR_SALT = 3911

# NOTE: _LAYOUT_RING ("ring") is the uniform box-fill baseline — it does NOT sample an
# annulus. Only _LAYOUT_VOID_RING ("void_ring") sets use_annulus; the "ring" name is kept
# for config compatibility but produces uniformly-filled sites, not a ring distribution.
_LAYOUT_RING = 0
_LAYOUT_VOID_RING = 1
_LAYOUT_CLUSTERED = 2
_LAYOUT_MIXED = 3


@wp.func
def _clamp_f(v: float, lo: float, hi: float) -> float:
    return wp.min(wp.max(v, lo), hi)


@wp.func
def _angle_delta(a: float, b: float) -> float:
    d = a - b
    return wp.atan2(wp.sin(d), wp.cos(d))


@wp.kernel
def _sample_sites_k(
    seeds: wp.array(dtype=wp.int32),
    S: int,
    layout_mode: int,
    box_size: float,
    out_sites: wp.array(dtype=wp.vec2f),
):
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _SITE_SALT + 19)
    base = e * S
    half = 0.5 * box_size
    two_pi = 2.0 * wp.pi
    cluster_phase = wp.randf(state) * two_pi
    cluster_radial = 0.24 * box_size + 0.10 * box_size * wp.randf(state)

    for i in range(S):
        use_annulus = int(0)
        if layout_mode == _LAYOUT_VOID_RING:
            use_annulus = int(1)
        elif layout_mode == _LAYOUT_MIXED:
            use_annulus = int(wp.randf(state) < 0.65)

        if layout_mode == _LAYOUT_CLUSTERED:
            cluster = int(wp.randf(state) * 6.0)
            if cluster > 5:
                cluster = int(5)
            ctheta = cluster_phase + two_pi * float(cluster) / 6.0
            cr = cluster_radial * (0.82 + 0.30 * wp.randf(state))
            center = wp.vec2f(cr * wp.cos(ctheta), cr * wp.sin(ctheta))
            theta = wp.randf(state) * two_pi
            noise_r = 0.025 * box_size + 0.080 * box_size * wp.randf(state)
            p = center + wp.vec2f(noise_r * wp.cos(theta), noise_r * wp.sin(theta))
            if wp.randf(state) < 0.22:
                use_annulus = int(1)
            else:
                out_sites[base + i] = wp.vec2f(_clamp_f(p[0], -half, half),
                                               _clamp_f(p[1], -half, half))
                continue

        if use_annulus:
            theta = wp.randf(state) * two_pi
            r_min = 0.14 * box_size
            r_max = 0.52 * box_size
            r = wp.sqrt(r_min * r_min + wp.randf(state) * (r_max * r_max - r_min * r_min))
            out_sites[base + i] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))
        else:
            x = (2.0 * wp.randf(state) - 1.0) * half
            y = (2.0 * wp.randf(state) - 1.0) * half
            out_sites[base + i] = wp.vec2f(x, y)


@wp.kernel
def _select_anchor_sites_k(
    seeds: wp.array(dtype=wp.int32),
    sites: wp.array(dtype=wp.vec2f),
    S: int,
    K: int,
    layout_mode: int,
    box_size: float,
    radial_variation: float,
    angular_jitter: float,
    selected: wp.array(dtype=wp.vec2f),
    used: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    sbase = e * S
    kbase = e * K
    state = wp.rand_init(seeds[e] * _ANCHOR_SALT + 31)
    two_pi = 2.0 * wp.pi
    sector = two_pi / float(K)
    rotation = wp.randf(state) * two_pi
    p0 = wp.randf(state) * two_pi
    p1 = wp.randf(state) * two_pi
    p2 = wp.randf(state) * two_pi
    p3 = wp.randf(state) * two_pi

    for j in range(S):
        used[sbase + j] = 0

    layout_boost = float(1.0)
    if layout_mode == _LAYOUT_VOID_RING:
        layout_boost = 1.10
    elif layout_mode == _LAYOUT_CLUSTERED:
        layout_boost = 1.16
    elif layout_mode == _LAYOUT_MIXED:
        layout_boost = 1.06
    amp = _clamp_f(radial_variation * layout_boost, 0.0, 0.85)

    for i in range(K):
        jitter = (2.0 * wp.randf(state) - 1.0) * angular_jitter
        theta = rotation + sector * float(i) + jitter
        profile = (
            0.62 * wp.sin(2.0 * theta + p0)
            + 0.44 * wp.cos(3.0 * theta + p1)
            + 0.26 * wp.sin(5.0 * theta + p2)
            + 0.18 * wp.cos(7.0 * theta + p3)
        ) / 1.50
        target_r = 0.34 * box_size * (1.0 + amp * profile)
        target_r = _clamp_f(target_r, 0.16 * box_size, 0.52 * box_size)

        best = int(0)
        best_cost = float(1.0e30)
        for j in range(S):
            site = sites[sbase + j]
            site_r = wp.length(site)
            site_theta = wp.atan2(site[1], site[0])
            angle_cost = wp.abs(_angle_delta(site_theta, theta)) / wp.max(sector, 1.0e-6)
            radius_cost = wp.abs(site_r - target_r) / wp.max(0.13 * box_size, 1.0e-6)
            used_cost = wp.where(used[sbase + j] > 0, 1000.0, 0.0)
            cost = angle_cost + radius_cost + used_cost
            if cost < best_cost:
                best_cost = cost
                best = j

        used[sbase + best] = 1
        selected[kbase + i] = sites[sbase + best]


@wp.kernel
def _chaikin_once_k(
    selected: wp.array(dtype=wp.vec2f),
    K: int,
    aug: wp.array(dtype=wp.vec2f),
):
    e = wp.tid()
    kbase = e * K
    abase = e * (2 * K)
    for i in range(K):
        p0 = selected[kbase + i]
        p1 = selected[kbase + (i + 1) % K]
        aug[abase + 2 * i] = 0.75 * p0 + 0.25 * p1
        aug[abase + 2 * i + 1] = 0.25 * p0 + 0.75 * p1


class VoronoiScratch:
    """Private working buffers for the standard ``"voronoi"`` generator."""

    __slots__ = (
        "sites", "used", "selected", "aug", "dense", "crossers",
        "arc_real", "arc_seg", "arc_s", "arc_cr", "arc_co",
    )

    def __init__(
        self,
        sites,
        used,
        selected,
        aug,
        dense,
        crossers,
        arc_real,
        arc_seg,
        arc_s,
        arc_cr,
        arc_co,
    ) -> None:
        self.sites = sites
        self.used = used
        self.selected = selected
        self.aug = aug
        self.dense = dense
        self.crossers = crossers
        self.arc_real = arc_real
        self.arc_seg = arc_seg
        self.arc_s = arc_s
        self.arc_cr = arc_cr
        self.arc_co = arc_co


def voronoi_layout_mode(name: str) -> int:
    """Kernel code of a ``voronoi_site_layout`` name (``"ring"``, ``"void_ring"``,
    ``"clustered"`` or ``"mixed"``); raises ``ValueError`` for any other name."""
    layouts = {
        "ring": _LAYOUT_RING,
        "void_ring": _LAYOUT_VOID_RING,
        "clustered": _LAYOUT_CLUSTERED,
        "mixed": _LAYOUT_MIXED,
    }
    try:
        return layouts[str(name)]
    except KeyError as exc:
        raise ValueError(
            "voronoi_site_layout must be one of "
            f"{sorted(layouts)}, got {name!r}"
        ) from exc


# Legacy private name (kept like warp_generate_polar._polar_num_knots: the gate generator
# imported it at 1077b18).
_voronoi_layout_mode = voronoi_layout_mode


def _voronoi_shape(config) -> tuple[int, int, int, int]:
    S = int(getattr(config, "voronoi_num_sites", 256))
    K = int(getattr(config, "voronoi_control_points", 18))
    npseg = int(config.num_points_per_segment)
    AP = 2 * K
    M = AP * npseg
    return S, K, AP, M


def sample_voronoi_sites(ctx: "LaunchContext", seeds: wp.array, S: int, layout_mode: int,
                         box_size: float, out_sites: wp.array) -> None:
    """Sample ``S`` Voronoi cell sites per env into ``out_sites`` (``[E*S]`` ``vec2f``).

    ``layout_mode`` is a :func:`voronoi_layout_mode` code; sites lie in the
    ``box_size``-wide square (or the annulus of the ring layouts) centred at the origin,
    drawn from ``wp.rand_init(seeds[e] * 8171 + 19)``. One launch (``E = seeds.shape[0]``
    threads); no allocation, no host sync, no validation.
    """
    ctx.launch(_sample_sites_k, seeds.shape[0],
               [seeds, int(S), int(layout_mode), float(box_size), out_sites])


def select_voronoi_anchors(ctx: "LaunchContext", seeds: wp.array, sites: wp.array, S: int,
                           K: int, layout_mode: int, box_size: float,
                           radial_variation: float, angular_jitter: float,
                           out_selected: wp.array, used: wp.array) -> None:
    """Snap ``K`` angular anchor targets per env to distinct nearby sites.

    For anchor ``i`` (angle ``rotation + 2*pi*i/K + jitter``, target radius from a
    seeded low-order radial profile scaled by ``radial_variation``) the cheapest unused
    site of the env's ``S`` ``sites`` is written to ``out_selected[e*K + i]``; ``used``
    (``[E*S]`` ``int32`` scratch) marks taken sites. Keyed
    ``wp.rand_init(seeds[e] * 3911 + 31)``. One launch (``E = seeds.shape[0]``
    threads); no allocation, no host sync, no validation.
    """
    ctx.launch(_select_anchor_sites_k, seeds.shape[0], [
        seeds, sites, int(S), int(K), int(layout_mode), float(box_size),
        float(radial_variation), float(angular_jitter), out_selected, used,
    ])


def voronoi_alloc_scratch(config):
    """Allocate private scratch for the standard Voronoi generator."""
    _init()
    E = int(config.num_envs)
    S, K, AP, M = _voronoi_shape(config)
    dev = str(config.device)
    return VoronoiScratch(
        sites=wp.empty(E * S, dtype=wp.vec2f, device=dev),
        used=wp.empty(E * S, dtype=wp.int32, device=dev),
        selected=wp.empty(E * K, dtype=wp.vec2f, device=dev),
        aug=wp.empty(E * AP, dtype=wp.vec2f, device=dev),
        dense=wp.empty(E * M, dtype=wp.vec2f, device=dev),
        crossers=wp.empty(E, dtype=wp.int32, device=dev),
        arc_real=wp.empty(E * M, dtype=wp.vec2f, device=dev),
        arc_seg=wp.empty(E * M, dtype=wp.float32, device=dev),
        arc_s=wp.empty(E * (M + 1), dtype=wp.float32, device=dev),
        arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        arc_co=wp.empty(E, dtype=wp.int32, device=dev),
    )


def generate_voronoi_warp(
    seeds_wp: wp.array,
    config,
    out_centerline: wp.array,
    out_valid_wp: wp.array,
    scratch,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Generate Voronoi-inspired graph-cycle centerlines into owned output buffers.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_voronoi_warp requires scratch"

    E = int(config.num_envs)
    N = int(config.num_points)
    S, K, AP, M = _voronoi_shape(config)
    npseg = int(config.num_points_per_segment)
    layout_mode = voronoi_layout_mode(getattr(config, "voronoi_site_layout", "void_ring"))
    radial_variation = float(getattr(config, "voronoi_radial_variation", 0.62))
    angular_jitter = float(getattr(config, "voronoi_angular_jitter", 0.08))
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    sample_voronoi_sites(ctx, seeds_wp, S, layout_mode, BOX_SIZE, scratch.sites)
    select_voronoi_anchors(ctx, seeds_wp, scratch.sites, S, K, layout_mode, BOX_SIZE,
                           radial_variation, angular_jitter, scratch.selected, scratch.used)
    ctx.launch(_chaikin_once_k, E, [scratch.selected, K, scratch.aug])
    # Closed Catmull-Rom over the 2K Chaikin points (endpoint-excluded, u = s/npseg).
    ops.closed_catmull_rom(ctx, scratch.aug, AP, npseg, scratch.dense)

    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)

    # Straight-polyline fallback for the envs whose smooth loop self-crosses; the second
    # normalisation re-applies the bbox rescale to every env.
    ops.self_intersections(ctx, out_centerline, N, scratch.crossers, count=scratch.arc_co)
    ops.closed_polyline(ctx, scratch.aug, AP, npseg, scratch.dense, active=scratch.crossers)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co,
                     active=scratch.crossers)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def register(reg) -> None:
    """Register the built-in ``"voronoi"`` track generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .generator_registry import GeneratorSpec

    reg.add(GeneratorSpec(
        name="voronoi",
        alloc_scratch=voronoi_alloc_scratch,
        generate=generate_voronoi_warp,
        accepts_ctx=True,
    ))
