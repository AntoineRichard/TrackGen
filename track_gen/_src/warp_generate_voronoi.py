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
polygon fallback and fill are public ``track_gen.ops`` primitives. The site field and
per-env anchor selection are :func:`sample_voronoi_sites` and
:func:`select_voronoi_anchors_per_env`, shared with the voronoi gate generator.
The :class:`Voronoi` Stage reads dynamic radial variation and angular jitter arrays.
"""
from __future__ import annotations

import warp as wp

from ..ops import rng as _rng

from .. import ops
from ..exec.context import LaunchContext
from ..ops.constants import TARGET_EXTENT
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import VoronoiParams, _declared_bounds
from .builtin_stage import check_dyn, default_alloc, run_view
from .runtime import _init

_TARGET_EXTENT = TARGET_EXTENT

#: Side of the square site field (centred at the origin) the built-in generator samples.
BOX_SIZE = 2.0

_SITE_SALT = 0x877BE6DF
_ANCHOR_SALT = 0xC9F3730A

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
    state = _rng.init(seeds[e], wp.uint32(_SITE_SALT))
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




# The declared bounds of VoronoiParams' radial variation / angular jitter (the twin's
# clamp, float32; VoronoiGateParams declares the same ones).
_RV_LO = wp.constant(wp.float32(_declared_bounds(VoronoiParams,
                                                 "voronoi_radial_variation")[0]))
_RV_HI = wp.constant(wp.float32(_declared_bounds(VoronoiParams,
                                                 "voronoi_radial_variation")[1]))
_AJ_LO = wp.constant(wp.float32(_declared_bounds(VoronoiParams,
                                                 "voronoi_angular_jitter")[0]))
_AJ_HI = wp.constant(wp.float32(_declared_bounds(VoronoiParams,
                                                 "voronoi_angular_jitter")[1]))


@wp.kernel
def _select_anchor_sites_dyn_k(
    seeds: wp.array(dtype=wp.int32),
    sites: wp.array(dtype=wp.vec2f),
    S: int,
    K: int,
    layout_mode: int,
    box_size: float,
    rv_e: wp.array(dtype=wp.float32),
    aj_e: wp.array(dtype=wp.float32),
    selected: wp.array(dtype=wp.vec2f),
    used: wp.array(dtype=wp.int32),
):
    # _select_anchor_sites_k with the radial variation / angular jitter read per env
    # (roadmap PR13c): each loaded once and clamped to its declared bounds by comparisons
    # (NaN and every in-range value pass bit for bit), then the scalar kernel's body.
    e = wp.tid()
    radial_variation = rv_e[e]
    radial_variation = wp.where(radial_variation < _RV_LO, _RV_LO, radial_variation)
    radial_variation = wp.where(radial_variation > _RV_HI, _RV_HI, radial_variation)
    angular_jitter = aj_e[e]
    angular_jitter = wp.where(angular_jitter < _AJ_LO, _AJ_LO, angular_jitter)
    angular_jitter = wp.where(angular_jitter > _AJ_HI, _AJ_HI, angular_jitter)
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
    drawn from ``ops.rng.init`` with the fixed ``generator.voronoi/sites`` salt. One launch (``E = seeds.shape[0]``
    threads); no allocation, no host sync, no validation.
    """
    ctx.launch(_sample_sites_k, seeds.shape[0],
               [seeds, int(S), int(layout_mode), float(box_size), out_sites])




def select_voronoi_anchors_per_env(ctx: "LaunchContext", seeds: wp.array,
                                   sites: wp.array, S: int, K: int, layout_mode: int,
                                   box_size: float, radial_variation: wp.array,
                                   angular_jitter: wp.array, out_selected: wp.array,
                                   used: wp.array) -> None:
    """the scalar control draw with per-env knobs: ``[E]``
    ``float32`` arrays, env ``e`` selecting with ``radial_variation[e]`` /
    ``angular_jitter[e]``, each clamped at use to the voronoi generator's declared
    bounds (``[0, 0.85]`` / ``[0, pi]``, compared in float32; a NaN passes). With every
    value inside its bounds, env ``e`` equals the scalar control draw at the
    scalars. One launch (``E = seeds.shape[0]`` threads); no allocation, no host sync.
    """
    ctx.launch(_select_anchor_sites_dyn_k, seeds.shape[0], [
        seeds, sites, int(S), int(K), int(layout_mode), float(box_size),
        radial_variation, angular_jitter, out_selected, used,
    ])


#: The dynamic fields of the Voronoi Stage (roadmap PR13c), in Params order.
_DYNAMIC = ("voronoi_radial_variation", "voronoi_angular_jitter", "size")




def _alloc_voronoi_scratch(config, alloc=None):
    """the private scratch allocator with every buffer from ``alloc(name, dtype, n)``
    (named after its :class:`VoronoiScratch` slot; default ``wp.empty`` on
    ``config.device``). The Stage passes its ``StageBuilder.scratch``."""
    _init()
    alloc = default_alloc(config) if alloc is None else alloc
    E = int(config.num_envs)
    S, K, AP, M = _voronoi_shape(config)
    return VoronoiScratch(
        sites=alloc("sites", wp.vec2f, E * S),
        used=alloc("used", wp.int32, E * S),
        selected=alloc("selected", wp.vec2f, E * K),
        aug=alloc("aug", wp.vec2f, E * AP),
        dense=alloc("dense", wp.vec2f, E * M),
        crossers=alloc("crossers", wp.int32, E),
        arc_real=alloc("arc_real", wp.vec2f, E * M),
        arc_seg=alloc("arc_seg", wp.float32, E * M),
        arc_s=alloc("arc_s", wp.float32, E * (M + 1)),
        arc_cr=alloc("arc_cr", wp.int32, E),
        arc_co=alloc("arc_co", wp.int32, E),
    )




def _enqueue_voronoi(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                     dyn) -> None:
    """Enqueue the generator with resolved per-environment parameter arrays."""
    check_dyn(dyn, _DYNAMIC, "voronoi")
    E = int(config.num_envs)
    N = int(config.num_points)
    S, K, AP, M = _voronoi_shape(config)
    npseg = int(config.num_points_per_segment)
    layout_mode = voronoi_layout_mode(getattr(config, "voronoi_site_layout", "void_ring"))
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    sample_voronoi_sites(ctx, seeds_wp, S, layout_mode, BOX_SIZE, scratch.sites)
    select_voronoi_anchors_per_env(ctx, seeds_wp, scratch.sites, S, K, layout_mode,
                                   BOX_SIZE, dyn["voronoi_radial_variation"],
                                   dyn["voronoi_angular_jitter"], scratch.selected,
                                   scratch.used)
    ctx.launch(_chaikin_once_k, E, [scratch.selected, K, scratch.aug])
    # Closed Catmull-Rom over the 2K Chaikin points (endpoint-excluded, u = s/npseg).
    ops.closed_catmull_rom(ctx, scratch.aug, AP, npseg, scratch.dense)

    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent, size=dyn["size"])

    # Straight-polyline fallback for the envs whose smooth loop self-crosses; the second
    # normalisation re-applies the bbox rescale to every env.
    ops.self_intersections(ctx, out_centerline, N, scratch.crossers, count=scratch.arc_co)
    ops.closed_polyline(ctx, scratch.aug, AP, npseg, scratch.dense, active=scratch.crossers)
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co,
                     active=scratch.crossers)
    ops.normalize_extent(ctx, out_centerline, N, target_extent, size=dyn["size"])
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


class Voronoi(Stage):
    """``"voronoi"``: the Voronoi / graph-cycle track generator
     as a ``kind="generator"`` Stage.
    ``voronoi_radial_variation`` / ``voronoi_angular_jitter`` are dynamic: per-env
    arrays read by :func:`select_voronoi_anchors_per_env`, clamped to ``[0, 0.85]`` /
    ``[0, pi]`` at use."""

    kind = "generator"
    name = "voronoi"
    output = "centerline"
    capture = CaptureClass.STATIC
    Params = VoronoiParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}

    def build(self, b, p):
        cfg = run_view(p, b)
        scratch = _alloc_voronoi_scratch(cfg, b.scratch)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        dyn = {k: b.param(k) for k in _DYNAMIC}
        seed = b.seed

        def enqueue(ctx):
            _enqueue_voronoi(seed, cfg, xy, valid, scratch, ctx, dyn)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"voronoi"`` track generator (the :class:`Voronoi` Stage)
    into ``reg`` (called by ``track_gen._src.builtin_plugins``; importing this module
    registers nothing)."""
    reg.add(Voronoi)
