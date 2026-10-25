"""Independent scalar numeric reference with current stream initialization."""
from __future__ import annotations
from track_gen._src import warp_generate_voronoi as _implementation
from track_gen._src.runtime import _init
from track_gen.exec.context import resolve as _resolve_ctx

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

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
    state = _rng.init(seeds[e], wp.uint32(_ANCHOR_SALT))
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

def _generate(
    seeds_wp: wp.array,
    config,
    out_centerline: wp.array,
    out_valid_wp: wp.array,
    scratch,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Generate Voronoi-inspired graph-cycle centerlines into owned output buffers.

    The legacy entry point (frozen signature; the knobs of ``config`` as host scalars):
    the registered ``"voronoi"`` generator is the :class:`Voronoi` Stage on the same
    private core (roadmap PR13c).

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_voronoi_warp requires scratch"
    _enqueue_voronoi(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)

def _enqueue_voronoi(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, dyn) -> None:
    """Private numeric reference of the pre-0.3 scalar path."""
    E = int(config.num_envs)
    N = int(config.num_points)
    S, K, AP, M = _voronoi_shape(config)
    npseg = int(config.num_points_per_segment)
    layout_mode = voronoi_layout_mode(getattr(config, 'voronoi_site_layout', 'void_ring'))
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)
    sample_voronoi_sites(ctx, seeds_wp, S, layout_mode, BOX_SIZE, scratch.sites)
    radial_variation = float(getattr(config, 'voronoi_radial_variation', 0.62))
    angular_jitter = float(getattr(config, 'voronoi_angular_jitter', 0.08))
    select_voronoi_anchors(ctx, seeds_wp, scratch.sites, S, K, layout_mode, BOX_SIZE, radial_variation, angular_jitter, scratch.selected, scratch.used)
    ctx.launch(_chaikin_once_k, E, [scratch.selected, K, scratch.aug])
    ops.closed_catmull_rom(ctx, scratch.aug, AP, npseg, scratch.dense)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)
    ops.self_intersections(ctx, out_centerline, N, scratch.crossers, count=scratch.arc_co)
    ops.closed_polyline(ctx, scratch.aug, AP, npseg, scratch.dense, active=scratch.crossers)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co, active=scratch.crossers)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def _alloc_scratch(config):
    return _implementation._alloc_voronoi_scratch(config)
