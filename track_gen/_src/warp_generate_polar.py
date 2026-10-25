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
primitives. The control-knot draw uses :func:`sample_polar_controls_per_env`, shared
with the polar gate generator. The dense spline keeps its own one-thread-per-env
kernel for CUDA byte identity (see :data:`_polar_spline_dense_k`). The :class:`Polar`
Stage reads dynamic jitter arrays, clamped to their declared bounds at use.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext
from ..ops import funcs
from ..ops.constants import TARGET_EXTENT
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import PolarParams, _declared_bounds
from .builtin_stage import check_dyn, default_alloc, run_view
from .runtime import _init

# Independent RNG streams for controls. The constants are distinct from bezier/hull salts.
_CONTROL_SALT = 7919

#: Radius of the control ring before normalisation. The curve is normalised after
#: resampling, so the base radius only sets a stable pre-scale.
BASE_RADIUS = 1.0





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
            dense[db + i * samples_per_segment + s] = funcs.catmull_rom(p0, p1, p2, p3, u)


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

    Reads the polar generator's declared knot count.
    """
    return max(int(config.polar_num_knots), 4)








def sample_polar_controls_per_env(ctx: "LaunchContext", seeds: wp.array, K: int,
                                  radial_jitter: wp.array, angular_jitter: wp.array,
                                  out: wp.array, *,
                                  base_radius: float = BASE_RADIUS) -> None:
    """the scalar control draw with per-env jitters (roadmap PR13c): ``[E]``
    ``float32`` arrays, env ``e`` drawing with ``radial_jitter[e]`` /
    ``angular_jitter[e]``, each clamped at use to the polar generator's declared bounds
    (``[0, 0.85]`` / ``[0, 0.45]``, compared in float32; a NaN passes). With every
    value inside its bounds, env ``e`` equals the scalar control draw at the scalar
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




def _alloc_polar_scratch(config, alloc=None):
    """the private scratch allocator with every buffer from ``alloc(name, dtype, n)``
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




def _enqueue_polar(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                   dyn) -> None:
    """Enqueue the generator with resolved per-environment parameter arrays."""
    check_dyn(dyn, _DYNAMIC, "polar")
    N = int(config.num_points)
    K = polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)

    sample_polar_controls_per_env(ctx, seeds_wp, K, dyn["polar_radial_jitter"],
                                  dyn["polar_angular_jitter"], scratch.controls)
    # Closed Catmull-Rom through the controls, endpoint-excluded within each segment (the
    # next segment starts at p2; the loop closes through the arc-resample's wrap edge).
    # Same values as ops.closed_catmull_rom(ctx, controls, K, npseg, dense) on CPU; the
    # per-env loop kernel is kept for CUDA byte identity (see _polar_spline_dense_k).
    ctx.launch(_polar_spline_dense_k, int(config.num_envs),
               [scratch.controls, K, npseg, scratch.dense])
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent, size=dyn["size"])
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


class Polar(Stage):
    """``"polar"``: the periodic polar-knot track generator
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
