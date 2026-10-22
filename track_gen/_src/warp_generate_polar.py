"""Pure-Warp periodic polar control-knot centerline generation.

This implements generator method #3 from ``docs/pre-relaxation-generator-methods.md``
using the primary representation identified there and in the SOTA notes: sample sorted
polar control points, then fit/evaluate a periodic cubic spline. Each environment draws a
fixed number of radial knots around the origin, applies bounded angular jitter that keeps
the knot order monotone, evaluates a closed Catmull-Rom spline through those controls,
arc-length resamples it to ``config.num_points``, and normalizes the final bbox to the same
coordinate range as the bezier baseline.

The old Fourier radial function path was valid but collapsed toward high-compactness
near-circles. This module deliberately emphasizes random radial knots and local variation;
XPBD/inflation remain responsible for the final constant-width validity gate.

The arc resample, extent normalisation and fill are the public ``track_gen.ops``
primitives; the control-knot draw is exposed as :func:`sample_polar_controls` (shared
with the polar gate generator). The dense spline keeps its own one-thread-per-env kernel
(see :data:`_polar_spline_dense_k`).
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs
from ..ops.constants import TARGET_EXTENT
from .runtime import _init

# Independent RNG streams for controls. The constants are distinct from bezier/hull salts.
_CONTROL_SALT = 7919

#: Radius of the control ring before normalisation. The curve is normalised after
#: resampling, so the base radius only sets a stable pre-scale.
BASE_RADIUS = 1.0

# Legacy private names (values only): the loop is normalised to ``config.scale *
# TARGET_EXTENT`` (the bezier baseline's typical longest bbox side).
_BEZIER_EXTENT = TARGET_EXTENT
_BASE_RADIUS = BASE_RADIUS


@wp.kernel
def _polar_controls_k(
    seeds: wp.array(dtype=wp.int32),
    K: int,
    radial_jitter: float,
    angular_jitter: float,
    base_radius: float,
    controls: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Build K sorted polar control points. The angular jitter is a
    # fraction of one knot spacing and is clamped by Python to < 0.5, so the fixed index
    # order remains the sorted angular order without a dynamic sort.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _CONTROL_SALT + 17)
    b = e * K
    two_pi = 2.0 * wp.pi

    for i in range(K):
        radial_delta = (2.0 * wp.randf(state) - 1.0) * radial_jitter
        angle_delta = (2.0 * wp.randf(state) - 1.0) * angular_jitter
        r = base_radius * wp.max(1.0 + radial_delta, 0.1)
        theta = two_pi * (float(i) + angle_delta) / float(K)
        controls[b + i] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


# Legacy private name: the spline basis is ops.funcs.catmull_rom (the same function body).
_catmull_rom = funcs.catmull_rom


@wp.func
def _control_at(controls: wp.array(dtype=wp.vec2f), b: int, i: int, K: int) -> wp.vec2f:
    return controls[b + (i % K)]


@wp.kernel
def _polar_spline_dense_k(
    controls: wp.array(dtype=wp.vec2f),
    K: int,
    samples_per_segment: int,
    dense: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Evaluate a closed uniform Catmull-Rom spline through the polar
    # controls. Output stride per env is K * samples_per_segment.
    #
    # Kept verbatim from 1077b18 instead of ops.closed_catmull_rom (one thread per sample):
    # the values are bitwise equal on CPU, but on CUDA (fuse_fp) the loop structure decides
    # which multiply-adds the compiler contracts into FMAs (loop-invariant terms are
    # hoisted out of the inner loop), so the per-sample kernel is not guaranteed to round
    # the same. The merge is deferred to roadmap PR16.
    e = wp.tid()
    cb = e * K
    M = K * samples_per_segment
    db = e * M

    for i in range(K):
        p0 = _control_at(controls, cb, i - 1 + K, K)
        p1 = _control_at(controls, cb, i, K)
        p2 = _control_at(controls, cb, i + 1, K)
        p3 = _control_at(controls, cb, i + 2, K)
        for s in range(samples_per_segment):
            # Endpoint-excluded within each segment; the next segment starts at p2. The
            # final dense point closes back to the first through arc-resample's wrap edge.
            u = float(s) / float(samples_per_segment)
            dense[db + i * samples_per_segment + s] = _catmull_rom(p0, p1, p2, p3, u)


class PolarScratch:
    """Private working buffers for the polar generator (one alloc per generator).

    controls:  [E*K] vec2f - sorted polar control knots.
    dense:     [E*K*npseg] vec2f - periodic cubic samples through controls.
    arc_*:     scratch for NaN-aware arc-length resampling to config.num_points.
    """

    __slots__ = ("controls", "dense", "arc_real", "arc_seg", "arc_s", "arc_cr", "arc_co")

    def __init__(self, controls, dense, arc_real, arc_seg, arc_s, arc_cr, arc_co) -> None:
        self.controls = controls
        self.dense = dense
        self.arc_real = arc_real
        self.arc_seg = arc_seg
        self.arc_s = arc_s
        self.arc_cr = arc_cr
        self.arc_co = arc_co


def polar_num_knots(config) -> int:
    """Number of polar control knots ``K`` for ``config`` (at least 4).

    Reads ``config.polar_num_knots``, falling back to the old polar/Fourier knob
    ``num_harmonics`` (then 12) for config-like objects that have not adopted it.
    """
    return max(int(getattr(config, "polar_num_knots", getattr(config, "num_harmonics", 12))), 4)


# Legacy private name (kept: older plugins and the gate generator imported it).
_polar_num_knots = polar_num_knots


def polar_jitters(config) -> "tuple[float, float]":
    """``(radial_jitter, angular_jitter)`` of the polar control draw for ``config``.

    Defaults are deliberately stronger than the old low-pass Fourier path so compactness
    does not cluster near one. The clamps keep the radius positive (radial jitter
    ``<= 0.85``) and the angular order monotone (angular jitter ``<= 0.45`` of a knot
    spacing).
    """
    radial_default = 0.60 * float(getattr(config, "amplitude", 1.0))
    radial_jitter = min(max(float(getattr(config, "polar_radial_jitter", radial_default)), 0.0), 0.85)
    angular_jitter = min(max(float(getattr(config, "polar_angular_jitter", 0.30)), 0.0), 0.45)
    return radial_jitter, angular_jitter


def sample_polar_controls(ctx: "LaunchContext", seeds: wp.array, K: int,
                          radial_jitter: float, angular_jitter: float, out: wp.array, *,
                          base_radius: float = BASE_RADIUS) -> None:
    """Draw ``K`` angularly sorted polar control knots per env into ``out``.

    Knot ``i`` of env ``e`` sits at angle ``2*pi*(i + d_ang)/K`` and radius
    ``base_radius * max(1 + d_rad, 0.1)`` with ``d_rad, d_ang`` uniform in
    ``[-radial_jitter, radial_jitter]`` / ``[-angular_jitter, angular_jitter]``, from
    ``wp.rand_init(seeds[e] * 7919 + 17)``. With ``angular_jitter < 0.5`` the index order
    is the angular order. One launch (``E = seeds.shape[0]`` threads); no allocation, no
    host sync, no validation.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        seeds: ``[E]`` ``int32`` per-env seeds.
        K: knots per env.
        radial_jitter: relative radius jitter (see :func:`polar_jitters`).
        angular_jitter: angle jitter as a fraction of one knot spacing.
        out: ``[E*K]`` ``vec2f`` control knots.
        base_radius: ring radius before jitter.
    """
    ctx.launch(_polar_controls_k, seeds.shape[0],
               [seeds, int(K), float(radial_jitter), float(angular_jitter),
                float(base_radius), out])


def polar_alloc_scratch(config):
    """Allocate the polar generator's PRIVATE working scratch (one alloc per generator)."""
    _init()
    E = int(config.num_envs)
    K = polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    dev = str(config.device)
    return PolarScratch(
        controls=wp.empty(E * K, dtype=wp.vec2f, device=dev),
        dense=wp.empty(E * M, dtype=wp.vec2f, device=dev),
        arc_real=wp.empty(E * M, dtype=wp.vec2f, device=dev),
        arc_seg=wp.empty(E * M, dtype=wp.float32, device=dev),
        arc_s=wp.empty(E * (M + 1), dtype=wp.float32, device=dev),
        arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        arc_co=wp.empty(E, dtype=wp.int32, device=dev),
    )


def generate_polar_warp(seeds_wp: wp.array, config,
                        out_centerline: wp.array, out_valid_wp: wp.array,
                        scratch, ctx: "LaunchContext | None" = None) -> None:
    """Periodic polar-knot centerline generation - in-place owned path.

    Draws per-env radial/angle control knots, evaluates a periodic cubic spline, arc-length
    resamples it to ``config.num_points``, normalizes the output scale, and marks the
    generation stage valid. Pure Warp, zero per-call allocation, graph-capturable.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_polar_warp requires scratch"

    N = int(config.num_points)
    K = polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    radial_jitter, angular_jitter = polar_jitters(config)
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    sample_polar_controls(ctx, seeds_wp, K, radial_jitter, angular_jitter, scratch.controls)
    # Closed Catmull-Rom through the controls, endpoint-excluded within each segment (the
    # next segment starts at p2; the loop closes through the arc-resample's wrap edge).
    # Same values as ops.closed_catmull_rom(ctx, controls, K, npseg, dense) on CPU; the
    # per-env loop kernel is kept for CUDA byte identity (see _polar_spline_dense_k).
    ctx.launch(_polar_spline_dense_k, int(config.num_envs),
               [scratch.controls, K, npseg, scratch.dense])
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def register(reg) -> None:
    """Register the built-in ``"polar"`` track generator into ``reg`` (called by
    ``track_gen._src.builtin_plugins``; importing this module registers nothing)."""
    from .builtin_params import PolarParams
    from .generator_registry import GeneratorSpec

    reg.add(GeneratorSpec(
        name="polar",
        alloc_scratch=polar_alloc_scratch,
        generate=generate_polar_warp,
        accepts_ctx=True,
        params=PolarParams,
    ))
