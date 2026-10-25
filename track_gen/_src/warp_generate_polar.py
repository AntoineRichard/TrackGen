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

Roadmap PR13c: the registered ``"polar"`` generator is the :class:`Polar` Stage, whose
jitters are dynamic (per-env arrays, :func:`sample_polar_controls_per_env`, clamped to
the declared bounds, the former host clamp, at use); ``generate_polar_warp`` /
``polar_alloc_scratch`` / ``sample_polar_controls`` keep their signatures and the scalar
kernel on the same private core, :func:`_enqueue_polar`.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs
from ..ops.constants import TARGET_EXTENT
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import PolarParams, _declared_bounds
from .builtin_stage import (DeprecatedParamsAlias, check_dyn, check_no_size,
                            default_alloc, run_view)
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


# The declared bounds of PolarParams' jitters (the twin's clamp, float32; the polar gate
# generator's PolarGateParams declares the same ones).
_RADIAL_LO = wp.constant(wp.float32(_declared_bounds(PolarParams, "polar_radial_jitter")[0]))
_RADIAL_HI = wp.constant(wp.float32(_declared_bounds(PolarParams, "polar_radial_jitter")[1]))
_ANGULAR_LO = wp.constant(wp.float32(
    _declared_bounds(PolarParams, "polar_angular_jitter")[0]))
_ANGULAR_HI = wp.constant(wp.float32(
    _declared_bounds(PolarParams, "polar_angular_jitter")[1]))


@wp.kernel
def _polar_controls_dyn_k(
    seeds: wp.array(dtype=wp.int32),
    K: int,
    radial_e: wp.array(dtype=wp.float32),
    angular_e: wp.array(dtype=wp.float32),
    base_radius: float,
    controls: wp.array(dtype=wp.vec2f),
):
    # _polar_controls_k with the jitters read per env (roadmap PR13c): each loaded once and
    # clamped to its declared bounds by comparisons (== polar_jitters()'s host clamp after
    # float32 rounding; NaN passes like Python's min(max(nan, lo), hi)), then the scalar
    # kernel's body verbatim.
    e = wp.tid()
    radial_jitter = radial_e[e]
    radial_jitter = wp.where(radial_jitter < _RADIAL_LO, _RADIAL_LO, radial_jitter)
    radial_jitter = wp.where(radial_jitter > _RADIAL_HI, _RADIAL_HI, radial_jitter)
    angular_jitter = angular_e[e]
    angular_jitter = wp.where(angular_jitter < _ANGULAR_LO, _ANGULAR_LO, angular_jitter)
    angular_jitter = wp.where(angular_jitter > _ANGULAR_HI, _ANGULAR_HI, angular_jitter)
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


def sample_polar_controls_per_env(ctx: "LaunchContext", seeds: wp.array, K: int,
                                  radial_jitter: wp.array, angular_jitter: wp.array,
                                  out: wp.array, *,
                                  base_radius: float = BASE_RADIUS) -> None:
    """:func:`sample_polar_controls` with per-env jitters (roadmap PR13c): ``[E]``
    ``float32`` arrays, env ``e`` drawing with ``radial_jitter[e]`` /
    ``angular_jitter[e]``, each clamped at use to the polar generator's declared bounds
    (``[0, 0.85]`` / ``[0, 0.45]``, compared in float32; a NaN passes). With every
    value inside its bounds, env ``e`` equals :func:`sample_polar_controls` at the scalar
    ``float(radial_jitter[e])`` / ``float(angular_jitter[e])``. One launch
    (``E = seeds.shape[0]`` threads); no allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        seeds: ``[E]`` ``int32`` per-env seeds.
        K: knots per env.
        radial_jitter: ``[E]`` ``float32`` relative radius jitter.
        angular_jitter: ``[E]`` ``float32`` angle jitter (fraction of a knot spacing).
        out: ``[E*K]`` ``vec2f`` control knots.
        base_radius: ring radius before jitter.
    """
    ctx.launch(_polar_controls_dyn_k, seeds.shape[0],
               [seeds, int(K), radial_jitter, angular_jitter, float(base_radius), out])


#: The dynamic fields of the Polar Stage (roadmap PR13c), in Params order.
_DYNAMIC = ("polar_radial_jitter", "polar_angular_jitter", "size")


def polar_alloc_scratch(config):
    """Allocate the polar generator's PRIVATE working scratch (one alloc per generator)."""
    return _alloc_polar_scratch(config)


def _alloc_polar_scratch(config, alloc=None):
    """:func:`polar_alloc_scratch` with every buffer from ``alloc(name, dtype, n)``
    (named after its :class:`PolarScratch` slot; default ``wp.empty`` on
    ``config.device``). The Stage passes its ``StageBuilder.scratch``."""
    _init()
    alloc = default_alloc(config) if alloc is None else alloc
    E = int(config.num_envs)
    K = polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    return PolarScratch(
        controls=alloc("controls", wp.vec2f, E * K),
        dense=alloc("dense", wp.vec2f, E * M),
        arc_real=alloc("arc_real", wp.vec2f, E * M),
        arc_seg=alloc("arc_seg", wp.float32, E * M),
        arc_s=alloc("arc_s", wp.float32, E * (M + 1)),
        arc_cr=alloc("arc_cr", wp.int32, E),
        arc_co=alloc("arc_co", wp.int32, E),
    )


def generate_polar_warp(seeds_wp: wp.array, config,
                        out_centerline: wp.array, out_valid_wp: wp.array,
                        scratch, ctx: "LaunchContext | None" = None) -> None:
    """Periodic polar-knot centerline generation - in-place owned path.

    Draws per-env radial/angle control knots, evaluates a periodic cubic spline, arc-length
    resamples it to ``config.num_points``, normalizes the output scale, and marks the
    generation stage valid. Pure Warp, zero per-call allocation, graph-capturable.

    The legacy entry point (frozen signature; :func:`polar_jitters` of ``config`` as host
    scalars): the registered ``"polar"`` generator is the :class:`Polar` Stage on the same
    private core (roadmap PR13c).

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    check_no_size(config, "generate_polar_warp")
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_polar_warp requires scratch"
    _enqueue_polar(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)


def _enqueue_polar(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                   dyn) -> None:
    """The polar generator's launches on ``ctx``: ``dyn=None`` (the legacy path:
    :func:`polar_jitters` of ``config``, :func:`sample_polar_controls`) or the jitters'
    and ``size``'s ``[E]`` arrays (the Stage: :func:`sample_polar_controls_per_env`, and
    the normalization to ``scale * TARGET_EXTENT * size[e]``, roadmap PR13d)."""
    check_dyn(dyn, _DYNAMIC, "polar")
    N = int(config.num_points)
    K = polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    if dyn is None:
        radial_jitter, angular_jitter = polar_jitters(config)
        sample_polar_controls(ctx, seeds_wp, K, radial_jitter, angular_jitter,
                              scratch.controls)
    else:
        sample_polar_controls_per_env(ctx, seeds_wp, K, dyn["polar_radial_jitter"],
                                      dyn["polar_angular_jitter"], scratch.controls)
    # Closed Catmull-Rom through the controls, endpoint-excluded within each segment (the
    # next segment starts at p2; the loop closes through the arc-resample's wrap edge).
    # Same values as ops.closed_catmull_rom(ctx, controls, K, npseg, dense) on CPU; the
    # per-env loop kernel is kept for CUDA byte identity (see _polar_spline_dense_k).
    ctx.launch(_polar_spline_dense_k, int(config.num_envs),
               [scratch.controls, K, npseg, scratch.dense])
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    if dyn is None:
        ops.normalize_extent(ctx, out_centerline, N, target_extent)
    else:   # roadmap PR13d: the per-env size multiplies the target extent
        ops.normalize_extent(ctx, out_centerline, N, target_extent, size=dyn["size"])
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


class Polar(Stage):
    """``"polar"``: the periodic polar-knot track generator (:func:`generate_polar_warp`)
    as a ``kind="generator"`` Stage (roadmap PR13c). ``polar_radial_jitter`` /
    ``polar_angular_jitter`` are dynamic: per-env arrays read by
    :func:`sample_polar_controls_per_env`, clamped to ``[0, 0.85]`` / ``[0, 0.45]`` at
    use."""

    kind = "generator"
    name = "polar"
    output = "centerline"
    capture = CaptureClass.STATIC
    Params = PolarParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}
    #: Deprecated (0.2.0.dev4's ``GeneratorSpec.params``): use ``Params``.
    params = DeprecatedParamsAlias()

    def build(self, b, p):
        cfg = run_view(p, b)
        scratch = _alloc_polar_scratch(cfg, b.scratch)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        dyn = {k: b.param(k) for k in _DYNAMIC}
        seed = b.seed

        def enqueue(ctx):
            _enqueue_polar(seed, cfg, xy, valid, scratch, ctx, dyn)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"polar"`` track generator (the :class:`Polar` Stage) into
    ``reg`` (called by ``track_gen._src.builtin_plugins``; importing this module
    registers nothing)."""
    reg.add(Polar)
