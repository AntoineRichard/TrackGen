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
"""
from __future__ import annotations

import warp as wp

from . import warp_pipeline as _pipe

# Target longest-bbox extent (in units of config.scale) the generated loop is normalized
# to. Chosen to match the bezier baseline's typical per-env longest bbox dimension
# (~1.44 at scale=1.0), so half_width / spacing / relax see comparable coordinates.
_BEZIER_EXTENT = 1.44

# Independent RNG streams for controls. The constants are distinct from bezier/hull salts.
_CONTROL_SALT = 7919

# The curve is normalized after resampling, so base radius only sets a stable pre-scale.
_BASE_RADIUS = 1.0


@wp.func
def _catmull_rom(p0: wp.vec2f, p1: wp.vec2f, p2: wp.vec2f, p3: wp.vec2f, u: float) -> wp.vec2f:
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


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


@wp.kernel
def _normalize_centerline_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
):
    # One thread per env e. Center by bbox and scale so each env's longest bbox dimension
    # exactly matches target_extent after arc resampling.
    e = wp.tid()
    b = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(N):
        p = points[b + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(N):
        p = points[b + i]
        points[b + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


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


def _polar_num_knots(config) -> int:
    # Fall back to the old polar/Fourier knob for config-like objects outside the
    # dataclass that have not adopted polar_num_knots yet.
    return max(int(getattr(config, "polar_num_knots", getattr(config, "num_harmonics", 12))), 4)


def polar_alloc_scratch(config):
    """Allocate the polar generator's PRIVATE working scratch (one alloc per generator)."""
    _pipe._init()
    E = int(config.num_envs)
    K = _polar_num_knots(config)
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
                        scratch) -> None:
    """Periodic polar-knot centerline generation - in-place owned path.

    Draws per-env radial/angle control knots, evaluates a periodic cubic spline, arc-length
    resamples it to ``config.num_points``, normalizes the output scale, and marks the
    generation stage valid. Pure Warp, zero per-call allocation, graph-capturable.
    """
    _pipe._init()
    assert scratch is not None, "generate_polar_warp requires scratch"

    E = int(config.num_envs)
    N = int(config.num_points)
    K = _polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    dev = str(out_centerline.device)

    # Defaults are deliberately stronger than the old low-pass Fourier path so compactness
    # does not cluster near one. Clamp keeps radius positive and angular order monotone.
    radial_default = 0.60 * float(getattr(config, "amplitude", 1.0))
    radial_jitter = min(max(float(getattr(config, "polar_radial_jitter", radial_default)), 0.0), 0.85)
    angular_jitter = min(max(float(getattr(config, "polar_angular_jitter", 0.30)), 0.0), 0.45)
    target_extent = float(config.scale) * _BEZIER_EXTENT

    wp.launch(_polar_controls_k, dim=E,
              inputs=[seeds_wp, K, radial_jitter, angular_jitter, _BASE_RADIUS,
                      scratch.controls],
              device=dev)
    wp.launch(_polar_spline_dense_k, dim=E,
              inputs=[scratch.controls, K, npseg, scratch.dense],
              device=dev)
    _pipe._arc_resample_inplace(scratch.dense, M, N,
                                scratch.arc_real, scratch.arc_seg, scratch.arc_s,
                                scratch.arc_cr, scratch.arc_co, out_centerline, dev)
    wp.launch(_normalize_centerline_k, dim=E,
              inputs=[out_centerline, N, target_extent], device=dev)
    wp.launch(_pipe._fill_i32_k, dim=E, inputs=[out_valid_wp, 1], device=dev)
    _pipe._sync(dev)


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="polar",
    alloc_scratch=polar_alloc_scratch,
    generate=generate_polar_warp,
))
