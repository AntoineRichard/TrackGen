"""Private pre-0.3 scalar numeric reference; never imported by the library."""
from __future__ import annotations
from track_gen._src import warp_generate_polar as _implementation
from track_gen._src.runtime import _init
from track_gen.exec.context import resolve as _resolve_ctx

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

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

def _generate(seeds_wp: wp.array, config,
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
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_polar_warp requires scratch"
    _enqueue_polar(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)

def _enqueue_polar(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, dyn) -> None:
    """Private numeric reference of the pre-0.3 scalar path."""
    N = int(config.num_points)
    K = polar_num_knots(config)
    npseg = int(config.num_points_per_segment)
    M = K * npseg
    target_extent = float(config.scale) * TARGET_EXTENT
    arc = ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr)
    radial_jitter, angular_jitter = polar_jitters(config)
    sample_polar_controls(ctx, seeds_wp, K, radial_jitter, angular_jitter, scratch.controls)
    ctx.launch(_polar_spline_dense_k, int(config.num_envs), [scratch.controls, K, npseg, scratch.dense])
    ops.arc_resample(ctx, scratch.dense, M, N, arc, out_centerline, scratch.arc_co)
    ops.normalize_extent(ctx, out_centerline, N, target_extent)
    ops.fill(ctx, out_valid_wp, 1)
    ctx.sync()


def _alloc_scratch(config):
    return _implementation._alloc_polar_scratch(config)
