"""Pluggable per-point altitude (Z) profiles for gate courses and tracks.

The Z profiler is the vertical counterpart of the XY generator: it runs on
the ordered/resampled 2D anchors AFTER ordering/relaxation and BEFORE the 3D
lift. All kernels are fixed-shape, allocation-free and capture-safe. Profiles
are closed-loop consistent: flat trivially, random_walk via a Brownian-bridge
drift subtraction, noise via periodic harmonics in normalized arc length.
Padding slots get z = 0 (the lift NaNs them via pos2 anyway).

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
