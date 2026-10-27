"""Independent scalar elevation reference with current stream initialization."""
from __future__ import annotations
from track_gen._src import warp_zprofile as _implementation

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

@wp.kernel
def _z_uniform_k(seeds: wp.array(dtype=wp.int32),
                 count: wp.array(dtype=wp.int32), max_gates: int,
                 z_min: float, z_max: float,
                 z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    if i >= count[e]:
        z[t] = 0.0
        return
    state = _rng.init_at(seeds[e], wp.uint32(_P_UNIFORM), i)
    z[t] = z_min + wp.randf(state) * (z_max - z_min)

@wp.kernel
def _z_walk_k(seeds: wp.array(dtype=wp.int32),
              count: wp.array(dtype=wp.int32), max_gates: int,
              cum: wp.array(dtype=wp.float32),
              perim: wp.array(dtype=wp.float32),
              z_base: float, z_min: float, z_max: float, max_grade: float,
              z: wp.array(dtype=wp.float32)):
    e = wp.tid()
    base = e * max_gates
    n = count[e]
    if n > max_gates:
        n = max_gates
    if n < 1:
        return
    state = _rng.init(seeds[e], wp.uint32(_P_WALK))
    acc = float(0.0)
    for i in range(n):
        if i > 0:
            ds = cum[base + i] - cum[base + i - 1]
            acc = acc + (2.0 * wp.randf(state) - 1.0) * max_grade * ds
        z[base + i] = acc
    # Brownian bridge: subtract the linear drift so the closing step
    # (n-1 -> 0) carries no accumulated offset, then rebase and clamp.
    p_e = perim[e]
    drift = acc
    for i in range(n):
        frac = 0.0
        if p_e > 1.0e-9:
            frac = cum[base + i] / p_e
        z[base + i] = wp.clamp(z_base + z[base + i] - drift * frac,
                               z_min, z_max)
    for i in range(n, max_gates):
        z[base + i] = 0.0

@wp.kernel
def _z_noise_k(seeds: wp.array(dtype=wp.int32),
               count: wp.array(dtype=wp.int32), max_gates: int,
               cum: wp.array(dtype=wp.float32),
               perim: wp.array(dtype=wp.float32),
               z_base: float, z_min: float, z_max: float,
               amplitude: float, harmonics: int,
               z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    n = count[e]
    if n > max_gates:
        n = max_gates
    if i >= n:
        z[t] = 0.0
        return
    base = e * max_gates
    p_e = perim[e]
    frac = 0.0
    if p_e > 1.0e-9:
        frac = cum[base + i] / p_e
    acc = float(0.0)
    norm = float(0.0)
    for k in range(harmonics):
        state = _rng.init_at(seeds[e], wp.uint32(_P_NOISE), k)
        a = wp.randf(state)                       # harmonic amplitude in [0,1)
        phase = wp.randf(state) * _TWO_PI
        w = 1.0 / float(k + 1)                    # 1/f-ish spectrum
        acc = acc + a * w * wp.sin(_TWO_PI * float(k + 1) * frac + phase)
        norm = norm + w
    zz = z_base + amplitude * acc / wp.max(norm, 1.0e-9)
    z[t] = wp.clamp(zz, z_min, z_max)

def _apply(config, seeds_wp: wp.array, count: wp.array, stride: int,
                    cum: wp.array, perim: wp.array, z: wp.array,
                    ctx: "LaunchContext | None" = None) -> None:
    """Fill ``z`` ``[E*stride]`` from the configured profile.

    Pure profile-select + launches, capture-safe (no alloc/sync). ``cum``
    and ``perim`` must already hold the caller's plan-view arc
    parameterization (gate path: ``gate_cum_perim``; track path: the
    resampler's own arc tables). Reads only the shared ``z_*`` attribute
    names off ``config``, so both ``GateGenConfig`` and ``TrackGenConfig``
    work through this one entry point.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    ctx = _resolve_ctx(ctx, str(config.device))
    _apply_z_profile(config, seeds_wp, count, stride, cum, perim, z, ctx, None)

def _apply_knots(config, seeds_wp: wp.array, count: wp.array,
                          stride: int, arclen: wp.array, perim: wp.array,
                          knot_cum: wp.array, knot_count: wp.array,
                          knot_z: wp.array, z: wp.array,
                          ctx: "LaunchContext | None" = None) -> None:
    """Knot-based altitude: sample K control knots, then interpolate per point.

    Three launches, capture-safe (no alloc, no sync): build the analytic knot
    arc table, run the configured profile over the K knots via
    :func:`apply_z_profile` (``stride = K``), then evaluate the periodic
    monotone cubic at every point's arc fraction. Because the knots carry the
    profile's own clamp to ``[z_min, z_max]`` and the interpolant never leaves
    the interval between adjacent knots, the per-point result needs no
    additional clamping.

    Intended for ``z_profile`` in ``{"uniform", "random_walk"}``; ``"noise"``
    is analytically smooth and should call :func:`apply_z_profile` directly.

    Reads ``config.z_control_points``, so ``config`` must be a
    :class:`TrackGenConfig`; a ``GateGenConfig`` caller raises a bare
    ``AttributeError``.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    ctx = _resolve_ctx(ctx, str(config.device))
    _apply_z_profile_knots(config, seeds_wp, count, stride, arclen, perim, knot_cum,
                           knot_count, knot_z, z, ctx, None)

def _apply_z_profile(config, seeds_wp, count, stride, cum, perim, z, ctx, dyn) -> None:
    """:func:`apply_z_profile` on a resolved ``ctx``: ``dyn=None`` (the legacy path: the
    ``z_*`` host scalars of ``config``, the scalar kernels) or exactly the profile's
    dynamic fields mapped to their ``[E]`` float32 arrays (the elevation stages: the
    ``_dyn`` twins; ``z_base`` / ``z_noise_harmonics`` stay ``config`` reads)."""
    profile = config.z_profile
    _check_dyn(profile, dyn)
    E = int(config.num_envs)
    S = int(stride)
    if dyn:
        if profile == "uniform":
            ctx.launch(_z_uniform_dyn_k, E * S,
                       [seeds_wp, count, S, dyn["z_min"], dyn["z_max"], z])
        elif profile == "random_walk":
            ctx.launch(_z_walk_dyn_k, E,
                       [seeds_wp, count, S, cum, perim, float(config.z_base),
                        dyn["z_min"], dyn["z_max"], dyn["z_max_step"], z])
        elif profile == "noise":
            ctx.launch(_z_noise_dyn_k, E * S,
                       [seeds_wp, count, S, cum, perim, float(config.z_base),
                        dyn["z_min"], dyn["z_max"], dyn["z_noise_amplitude"],
                        int(config.z_noise_harmonics), z])
        return
    if profile == "flat":
        ctx.launch(_z_flat_k, E * S,
                   [count, S, float(config.z_base), z])
    elif profile == "uniform":
        ctx.launch(_z_uniform_k, E * S,
                   [seeds_wp, count, S, float(config.z_min),
                    float(config.z_max), z])
    elif profile == "random_walk":
        ctx.launch(_z_walk_k, E,
                   [seeds_wp, count, S, cum, perim,
                    float(config.z_base), float(config.z_min),
                    float(config.z_max), float(config.z_max_step), z])
    elif profile == "noise":
        ctx.launch(_z_noise_k, E * S,
                   [seeds_wp, count, S, cum, perim,
                    float(config.z_base), float(config.z_min),
                    float(config.z_max),
                    float(config.z_noise_amplitude),
                    int(config.z_noise_harmonics), z])
    else:
        # Config validation rejects unknown names at construction, but configs are
        # mutable; fail loudly rather than silently running some other profile.
        raise ValueError(
            f"unknown z_profile {profile!r}; expected one of "
            f"'flat', 'uniform', 'random_walk', 'noise'")

def _apply_z_profile_knots(config, seeds_wp, count, stride, arclen, perim, knot_cum,
                           knot_count, knot_z, z, ctx, dyn) -> None:
    """:func:`apply_z_profile_knots` on a resolved ``ctx``, the knot profile run by
    :func:`_apply_z_profile` with ``dyn`` (``None``: the legacy scalar path)."""
    _check_dyn(config.z_profile, dyn)
    E = int(config.num_envs)
    K = int(config.z_control_points)
    ctx.launch(_knot_tables_k, E * K,
               [perim, K, knot_cum, knot_count])
    _apply_z_profile(config, seeds_wp, knot_count, K, knot_cum, perim, knot_z, ctx, dyn)
    ctx.launch(_pchip_eval_k, E * int(stride),
               [arclen, perim, count, int(stride), knot_z, K, z])

