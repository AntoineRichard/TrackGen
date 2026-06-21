"""Pure-Warp periodic polar-spline centerline generation (generator method #3).

Owns ONE generation strategy: per env build a smooth periodic radial function

    r(theta) = base * (1 + sum_{k=1..K} a_k * sin(k*theta + phi_k))

with low-frequency harmonics (K = ``config.num_harmonics``), per-harmonic decay
``a_k ~ amp / k**decay_p`` (``decay_p = config.decay_p``, ``amp`` from
``config.amplitude``) and random per-env phases ``phi_k`` drawn ONCE per env from
the Warp RNG and reused across all angles. The amplitude budget is normalised so
``sum_k |a_k| <= AMP_CAP < 1``; hence ``r(theta) >= base*(1 - AMP_CAP) > 0``
ALWAYS, so the polar graph ``(r cos theta, r sin theta)`` is a star-shaped simple
loop BY CONSTRUCTION — no sorting, no closure solve, no self-crossing fallback.

The curve is evaluated at N = ``config.num_points`` FIXED angles
``theta_i = 2*pi*i/N`` (endpoint excluded so the loop closes), centred at the
origin and isotropically rescaled so each env's longest bounding-box dimension
matches the bezier baseline's typical extent (``config.scale * _BEZIER_EXTENT``),
so the downstream resample / ``half_width`` / XPBD relaxation see the same
coordinate range the bezier generator produces.

Convention (shared with ``warp_pipeline`` / ``warp_generate``): one env per row;
flat arrays ``[E*N]`` of ``wp.vec2f`` and ``[E]`` per-env scalars. ``generate`` is
pure Warp, in-place, zero per-call allocation (all scratch from ``alloc_scratch``;
``out_centerline``/``out_valid_wp`` are orchestrator-owned), CUDA-graph-capturable
(no host sync, no per-env Python branching, fixed-bound loops), and deterministic
in ``(seeds_wp[e], config)`` via ``wp.rand_init``.
"""
from __future__ import annotations

import warp as wp

from . import warp_pipeline as _pipe

# Fraction of `base` the harmonic perturbation is allowed to reach. Keeping the
# summed amplitude strictly below 1 guarantees r(theta) > 0 for all theta, so the
# polar loop is simple by construction. 0.6 -> r in [0.4*base, 1.6*base]: visibly
# curvy lobes while staying non-self-intersecting.
_AMP_CAP = 0.6

# Target longest-bbox extent (in units of config.scale) the centred/rescaled polar
# loop is normalised to. Chosen to match the bezier baseline's typical per-env
# longest bbox dimension (~1.44 at scale=1.0), so half_width / spacing / relax see
# the same coordinate range either generator produces.
_BEZIER_EXTENT = 1.44

# Distinct large odd RNG multiplier for the polar phase stream, decorrelated from
# the bezier count (6151) / corner (9781) streams so the same per-env seed maps to
# an independent rand_init state here.
_PHASE_SALT = 7919

# `base` (mean radius) is arbitrary: the curve is rescaled to target_extent after
# evaluation, so any positive base produces the same final geometry. Fixed at 1.0.
_BASE_RADIUS = 1.0


@wp.kernel
def _polar_phase_k(
    seeds: wp.array(dtype=wp.int32),
    K: int,
    phases: wp.array(dtype=wp.float32),
):
    # One thread per env e. Draws this env's K harmonic phases phi_k ~ U[0, 2*pi)
    # ONCE (the same phases are reused at every angle by _polar_centerline_k), from
    # an independent rand_init(seeds[e] * _PHASE_SALT) stream. Deterministic in the
    # per-env seed; K is a fixed config bound so the loop is graph-capturable.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _PHASE_SALT)
    eb = e * K
    two_pi = 2.0 * wp.pi
    for k in range(K):
        phases[eb + k] = two_pi * wp.randf(state)


@wp.kernel
def _polar_centerline_k(
    phases: wp.array(dtype=wp.float32),
    N: int,
    K: int,
    decay_p: float,
    amplitude: float,
    base: float,
    target_extent: float,
    raw: wp.array(dtype=wp.vec2f),
    out: wp.array(dtype=wp.vec2f),
    valid: wp.array(dtype=wp.int32),
):
    # One thread per env e. Builds env e's periodic polar loop from its pre-drawn
    # phases, centres it at the origin and rescales its longest bbox dimension to
    # target_extent, writing N points into out[e*N + i] and valid[e] = 1.
    #
    # Amplitudes a_k = amp / k**decay_p are renormalised by `gain` so sum_k |a_k| ==
    # _AMP_CAP (when the raw sum is positive); this caps the perturbation below
    # `base`, guaranteeing r(theta) > 0 and a simple loop. Two passes over the N
    # fixed angles (bbox, then centre+scale) — both fixed-bound -> graph-capturable.
    e = wp.tid()
    eb = e * N
    pb = e * K
    two_pi = 2.0 * wp.pi

    # gain so sum_k |a_k| == _AMP_CAP: raw_sum = sum_k amp / k**p.
    raw_sum = float(0.0)
    for k in range(1, K + 1):
        raw_sum = raw_sum + amplitude / wp.pow(float(k), decay_p)
    gain = float(0.0)
    if raw_sum > 1.0e-12:
        gain = _AMP_CAP / raw_sum

    # First pass: evaluate raw polar points and accumulate the bbox.
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)
    for i in range(N):
        theta = two_pi * float(i) / float(N)
        pert = float(0.0)
        for k in range(1, K + 1):
            phi = phases[pb + (k - 1)]
            ak = gain * amplitude / wp.pow(float(k), decay_p)
            pert = pert + ak * wp.sin(float(k) * theta + phi)
        r = base * (1.0 + pert)
        px = r * wp.cos(theta)
        py = r * wp.sin(theta)
        raw[eb + i] = wp.vec2f(px, py)
        min_x = wp.min(min_x, px)
        max_x = wp.max(max_x, px)
        min_y = wp.min(min_y, py)
        max_y = wp.max(max_y, py)

    # Centre at origin and isotropically rescale so the longest bbox dim == target.
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    s = target_extent / wp.max(extent, 1.0e-8)
    for i in range(N):
        p = raw[eb + i]
        out[eb + i] = wp.vec2f((p[0] - cx) * s, (p[1] - cy) * s)

    valid[e] = 1


class PolarScratch:
    """Private working buffers for the polar generator (one alloc per generator).

    phases:    [E*K] float32 — per-env harmonic phases (K = config.num_harmonics).
    raw:       [E*N] vec2f — unscaled polar points (N = config.num_points), read back
               in the second pass of _polar_centerline_k to centre + rescale.

    The generation OUTPUT buffers (centerline, valid) are orchestrator-owned and
    passed into ``generate``; they are NOT part of this scratch.
    """

    __slots__ = ("phases", "raw")

    def __init__(self, phases: "wp.array", raw: "wp.array") -> None:
        self.phases = phases
        self.raw = raw


def polar_alloc_scratch(config):
    """Allocate the polar generator's PRIVATE working scratch (one alloc per generator)."""
    _pipe._init()
    E = int(config.num_envs)
    N = int(config.num_points)
    K = max(int(config.num_harmonics), 1)  # >=1 so the [E*K] buffer is never empty
    dev = str(config.device)
    return PolarScratch(
        phases=wp.empty(E * K, dtype=wp.float32, device=dev),
        raw=wp.empty(E * N, dtype=wp.vec2f, device=dev),
    )


def generate_polar_warp(seeds_wp: wp.array, config,
                        out_centerline: wp.array, out_valid_wp: wp.array,
                        scratch) -> None:
    """Periodic polar-spline centerline generation — in-place owned path.

    Draws per-env harmonic phases, evaluates r(theta) at N fixed angles, centres
    and rescales each loop to the bezier coordinate range, and writes the closed
    centerline into ``out_centerline`` ([E*num_points] vec2f) and per-env validity
    (always 1 — the loop is simple by construction; inflate runs the real gate)
    into ``out_valid_wp`` ([E] int32). Pure Warp, zero per-call allocation,
    graph-capturable.

    Args:
        seeds_wp:       [E] int32 wp.array per-env base seeds.
        config:         TrackGenConfig (uses num_points, num_harmonics, decay_p,
                        amplitude, scale).
        out_centerline: [E*num_points] vec2f wp.array — written in-place.
        out_valid_wp:   [E] int32 wp.array — filled with 1.
        scratch:        the PolarScratch returned by ``polar_alloc_scratch``.
    """
    _pipe._init()
    assert scratch is not None, "generate_polar_warp requires scratch"
    E = int(config.num_envs)
    N = int(config.num_points)
    K = max(int(config.num_harmonics), 1)
    dev = str(out_centerline.device)
    target_extent = float(config.scale) * _BEZIER_EXTENT

    wp.launch(_polar_phase_k, dim=E,
              inputs=[seeds_wp, K, scratch.phases], device=dev)
    wp.launch(_polar_centerline_k, dim=E,
              inputs=[scratch.phases, N, K, float(config.decay_p),
                      float(config.amplitude), _BASE_RADIUS, target_extent,
                      scratch.raw, out_centerline, out_valid_wp],
              device=dev)
    _pipe._sync(dev)


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="polar",
    alloc_scratch=polar_alloc_scratch,
    generate=generate_polar_warp,
))
