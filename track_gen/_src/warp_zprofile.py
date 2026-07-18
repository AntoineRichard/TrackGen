"""Pluggable altitude (Z) profiles for gate courses and tracks.

The Z profiler is the vertical counterpart of the XY generator: it runs on
the ordered/resampled 2D anchors AFTER ordering/relaxation and BEFORE the 3D
lift. It offers two parameterizations of where altitude is *decided*:
per-anchor, where every slot draws its own value (gates, whose anchors are
already sparse and knot-like, and the analytically smooth ``noise`` profile),
and knot-based, where a small arc-length-spaced set of control knots is
sampled and a periodic monotone cubic interpolates between them (tracks under
``uniform``/``random_walk``, whose dense resampled points would otherwise turn
a per-point draw into jitter). All kernels are fixed-shape, allocation-free
and capture-safe. Profiles are closed-loop consistent: flat trivially,
random_walk via a Brownian-bridge drift subtraction, noise via periodic
harmonics in normalized arc length, and the knot interpolant via wrap-around
knot indexing. Padding slots get z = 0 (the lift NaNs them via pos2 anyway).

Profiles are parameterized purely by ``(cum, perim)`` — a per-slot plan-view
cumulative arc length and a per-env closed-loop perimeter — never by the raw
2D positions directly. The gate path derives ``(cum, perim)`` from ordered
gate anchors via ``gate_cum_perim`` (which wraps ``_cum_chords_k``); the
track path passes the constant-spacing resampler's own 2D arc tables. Either
caller can then drive the shared profile-select + launch logic in
``apply_z_profile``.
"""
import warp as wp

_P_UNIFORM = 15679
_P_WALK = 15683
_P_NOISE = 15731
_TWO_PI = 6.2831853071795864


@wp.kernel
def _cum_chords_k(
    pos2: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    max_gates: int,
    cum: wp.array(dtype=wp.float32),
    perim: wp.array(dtype=wp.float32),
):
    # cum[base+i] = plan-view arc length from gate 0 to gate i; the closing
    # chord (n-1 -> 0) is NOT included in cum. perim[e] is the full closed-loop
    # plan-view perimeter, cum[n-1] plus that closing chord.
    e = wp.tid()
    base = e * max_gates
    n = count[e]
    if n > max_gates:
        n = max_gates
    acc = float(0.0)
    for i in range(max_gates):
        if i < n:
            if i > 0:
                acc = acc + wp.length(pos2[base + i] - pos2[base + i - 1])
            cum[base + i] = acc
        else:
            cum[base + i] = 0.0
    if n >= 1:
        perim[e] = acc + wp.length(pos2[base] - pos2[base + n - 1])
    else:
        perim[e] = 0.0


@wp.kernel
def _z_flat_k(count: wp.array(dtype=wp.int32), max_gates: int, z_base: float,
              z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // max_gates
    i = t - e * max_gates
    if i < count[e]:
        z[t] = z_base
    else:
        z[t] = 0.0


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
    state = wp.rand_init(seeds[e] * _P_UNIFORM + i)
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
    state = wp.rand_init(seeds[e] * _P_WALK)
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
        state = wp.rand_init(seeds[e] * _P_NOISE + k)
        a = wp.randf(state)                       # harmonic amplitude in [0,1)
        phase = wp.randf(state) * _TWO_PI
        w = 1.0 / float(k + 1)                    # 1/f-ish spectrum
        acc = acc + a * w * wp.sin(_TWO_PI * float(k + 1) * frac + phase)
        norm = norm + w
    zz = z_base + amplitude * acc / wp.max(norm, 1.0e-9)
    z[t] = wp.clamp(zz, z_min, z_max)


def alloc_z_scratch(num_envs: int, stride: int, device):
    """``(cum, perim, z)`` float32 scratch, zero-initialized.

    ``cum`` and ``z`` are ``[num_envs * stride]`` (per real slot); ``perim``
    is ``[num_envs]`` (per closed loop). The gate caller passes
    ``(E, max_gates)``; the track caller passes ``(E, n_max)``.
    """
    E, S = int(num_envs), int(stride)
    dev = str(device)
    return (wp.zeros(E * S, dtype=wp.float32, device=dev),
            wp.zeros(E, dtype=wp.float32, device=dev),
            wp.zeros(E * S, dtype=wp.float32, device=dev))


def gate_cum_perim(pos2: wp.array, count: wp.array, max_gates: int,
                   cum: wp.array, perim: wp.array) -> None:
    """Derive plan-view ``(cum, perim)`` from ordered gate anchors.

    Wraps ``_cum_chords_k`` for the gate path: one thread per env, walking the
    ordered anchor chain to accumulate cumulative chord length and close the
    loop into a perimeter. Capture-safe (launch only, no alloc/sync).
    """
    E = int(perim.shape[0])
    dev = str(pos2.device)
    wp.launch(_cum_chords_k, dim=E,
              inputs=[pos2, count, int(max_gates), cum, perim], device=dev)


def apply_z_profile(config, seeds_wp: wp.array, count: wp.array, stride: int,
                    cum: wp.array, perim: wp.array, z: wp.array) -> None:
    """Fill ``z`` ``[E*stride]`` from the configured profile.

    Pure profile-select + launches, capture-safe (no alloc/sync). ``cum``
    and ``perim`` must already hold the caller's plan-view arc
    parameterization (gate path: ``gate_cum_perim``; track path: the
    resampler's own arc tables). Reads only the shared ``z_*`` attribute
    names off ``config``, so both ``GateGenConfig`` and ``TrackGenConfig``
    work through this one entry point.
    """
    E = int(config.num_envs)
    S = int(stride)
    dev = str(config.device)
    profile = config.z_profile
    if profile == "flat":
        wp.launch(_z_flat_k, dim=E * S,
                  inputs=[count, S, float(config.z_base), z], device=dev)
    elif profile == "uniform":
        wp.launch(_z_uniform_k, dim=E * S,
                  inputs=[seeds_wp, count, S, float(config.z_min),
                          float(config.z_max), z], device=dev)
    elif profile == "random_walk":
        wp.launch(_z_walk_k, dim=E,
                  inputs=[seeds_wp, count, S, cum, perim,
                          float(config.z_base), float(config.z_min),
                          float(config.z_max), float(config.z_max_step), z],
                  device=dev)
    else:  # "noise" — config validation guarantees membership
        wp.launch(_z_noise_k, dim=E * S,
                  inputs=[seeds_wp, count, S, cum, perim,
                          float(config.z_base), float(config.z_min),
                          float(config.z_max),
                          float(config.z_noise_amplitude),
                          int(config.z_noise_harmonics), z], device=dev)


@wp.func
def _pchip_tangent(d_prev: float, d_next: float) -> float:
    """Monotonicity-preserving knot tangent (uniform knot spacing).

    Zero at a local extremum (``d_prev * d_next <= 0``) — this is what kills
    overshoot — otherwise the secant average, magnitude-capped at
    ``3 * min(|d_prev|, |d_next|)``, the standard Fritsch-Carlson sufficient
    condition for a monotone cubic segment.
    """
    if d_prev * d_next <= 0.0:
        return 0.0
    avg = 0.5 * (d_prev + d_next)
    lim = 3.0 * wp.min(wp.abs(d_prev), wp.abs(d_next))
    if avg > lim:
        return lim
    if avg < -lim:
        return -lim
    return avg


@wp.kernel
def _knot_tables_k(
    perim: wp.array(dtype=wp.float32),
    K: int,
    knot_cum: wp.array(dtype=wp.float32),
    knot_count: wp.array(dtype=wp.int32),
):
    # Knots sit at uniform arc fractions, so their cumulative table is
    # analytic: knot k is at k * perim / K. All K slots are real.
    t = wp.tid()               # dim = E * K
    e = t // K
    k = t - e * K
    knot_cum[t] = float(k) * perim[e] / float(K)
    if k == 0:
        knot_count[e] = K


@wp.kernel
def _pchip_eval_k(
    arclen: wp.array(dtype=wp.float32),
    perim: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    knot_z: wp.array(dtype=wp.float32),
    K: int,
    z: wp.array(dtype=wp.float32),
):
    """Periodic monotone cubic through the K knots, sampled at each point's
    plan-view arc fraction. Padding slots (``i >= count[e]``) get 0.0, matching
    the per-point profile kernels."""
    t = wp.tid()               # dim = E * n_max
    e = t // n_max
    i = t - e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if i >= m:
        z[t] = 0.0
        return

    kbase = e * K
    P = perim[e]
    if P <= 1.0e-9:
        z[t] = knot_z[kbase]
        return

    h = P / float(K)
    x = (arclen[t] / P) * float(K)
    k = int(x)
    if k < 0:
        k = 0
    if k > K - 1:
        k = K - 1
    u = wp.clamp(x - float(k), 0.0, 1.0)

    kp = (k + K - 1) % K
    k1 = (k + 1) % K
    k2 = (k + 2) % K
    z0 = knot_z[kbase + k]
    z1 = knot_z[kbase + k1]
    d_prev = (z0 - knot_z[kbase + kp]) / h
    d_here = (z1 - z0) / h
    d_next = (knot_z[kbase + k2] - z1) / h
    m0 = _pchip_tangent(d_prev, d_here)
    m1 = _pchip_tangent(d_here, d_next)

    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    z[t] = h00 * z0 + h10 * h * m0 + h01 * z1 + h11 * h * m1


def alloc_knot_scratch(num_envs: int, control_points: int, device):
    """``(knot_cum, knot_count, knot_z)`` knot-stage scratch, zero-initialized.

    ``knot_cum``/``knot_z`` are ``[E * K]`` float32; ``knot_count`` is ``[E]``
    int32. Allocated once by the pipeline; never allocated on the hot path.
    """
    E, K = int(num_envs), int(control_points)
    dev = str(device)
    return (wp.zeros(E * K, dtype=wp.float32, device=dev),
            wp.zeros(E, dtype=wp.int32, device=dev),
            wp.zeros(E * K, dtype=wp.float32, device=dev))


def apply_z_profile_knots(config, seeds_wp: wp.array, count: wp.array,
                          stride: int, arclen: wp.array, perim: wp.array,
                          knot_cum: wp.array, knot_count: wp.array,
                          knot_z: wp.array, z: wp.array) -> None:
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
    """
    E = int(config.num_envs)
    K = int(config.z_control_points)
    dev = str(config.device)
    wp.launch(_knot_tables_k, dim=E * K,
              inputs=[perim, K, knot_cum, knot_count], device=dev)
    apply_z_profile(config, seeds_wp, knot_count, K, knot_cum, perim, knot_z)
    wp.launch(_pchip_eval_k, dim=E * int(stride),
              inputs=[arclen, perim, count, int(stride), knot_z, K, z],
              device=dev)
