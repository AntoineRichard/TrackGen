"""Pluggable per-gate altitude (Z) profiles for gate courses.

The Z profiler is the vertical counterpart of the XY generator: it runs on
the ordered 2D anchors AFTER ordering/relaxation and BEFORE the 3D lift.
All kernels are fixed-shape, allocation-free and capture-safe. Profiles are
closed-loop consistent: flat trivially, random_walk via a Brownian-bridge
drift subtraction, noise via periodic harmonics in normalized arc length.
Padding slots get z = 0 (the lift NaNs them via pos2 anyway).
"""
import warp as wp

from .types import GateGenConfig

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
):
    # cum[base+i] = plan-view arc length from gate 0 to gate i; the closing
    # chord (n-1 -> 0) is NOT included in cum but callers can recover the
    # perimeter as cum[n-1] + |p0 - p_{n-1}|.
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
              pos2: wp.array(dtype=wp.vec2f),
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
    perim = cum[base + n - 1] + wp.length(pos2[base] - pos2[base + n - 1])
    drift = acc
    for i in range(n):
        frac = 0.0
        if perim > 1.0e-9:
            frac = cum[base + i] / perim
        z[base + i] = wp.clamp(z_base + z[base + i] - drift * frac,
                               z_min, z_max)
    for i in range(n, max_gates):
        z[base + i] = 0.0


@wp.kernel
def _z_noise_k(seeds: wp.array(dtype=wp.int32),
               count: wp.array(dtype=wp.int32), max_gates: int,
               cum: wp.array(dtype=wp.float32),
               pos2: wp.array(dtype=wp.vec2f),
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
    perim = cum[base + n - 1] + wp.length(pos2[base] - pos2[base + n - 1])
    frac = 0.0
    if perim > 1.0e-9:
        frac = cum[base + i] / perim
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


def alloc_z_scratch(config: GateGenConfig):
    """(cum, z) float32 scratch, both [E * max_gates], zero-initialized."""
    E, G = int(config.num_envs), int(config.max_gates)
    dev = str(config.device)
    return (wp.zeros(E * G, dtype=wp.float32, device=dev),
            wp.zeros(E * G, dtype=wp.float32, device=dev))


def apply_z_profile(config: GateGenConfig, seeds_wp: wp.array,
                    pos2: wp.array, count: wp.array,
                    cum: wp.array, z: wp.array) -> None:
    """Fill z [E*G] from the configured profile. Capture-safe, no sync."""
    E, G = int(config.num_envs), int(config.max_gates)
    dev = str(config.device)
    profile = config.z_profile
    if profile != "flat":
        wp.launch(_cum_chords_k, dim=E, inputs=[pos2, count, G, cum],
                  device=dev)
    if profile == "flat":
        wp.launch(_z_flat_k, dim=E * G,
                  inputs=[count, G, float(config.z_base), z], device=dev)
    elif profile == "uniform":
        wp.launch(_z_uniform_k, dim=E * G,
                  inputs=[seeds_wp, count, G, float(config.z_min),
                          float(config.z_max), z], device=dev)
    elif profile == "random_walk":
        wp.launch(_z_walk_k, dim=E,
                  inputs=[seeds_wp, count, G, cum, pos2,
                          float(config.z_base), float(config.z_min),
                          float(config.z_max), float(config.z_max_step), z],
                  device=dev)
    else:  # "noise" — config validation guarantees membership
        wp.launch(_z_noise_k, dim=E * G,
                  inputs=[seeds_wp, count, G, cum, pos2,
                          float(config.z_base), float(config.z_min),
                          float(config.z_max),
                          float(config.z_noise_amplitude),
                          int(config.z_noise_harmonics), z], device=dev)
