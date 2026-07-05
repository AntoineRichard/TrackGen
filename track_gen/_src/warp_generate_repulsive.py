"""Repulsive-growth centerline generator (``generator="repulsive"``).

Grows each env's closed centerline from a small circle under a hard ratcheting
length constraint while a tangent-point (TP) energy keeps the curve self-avoiding,
confined to a disc domain seeded with per-env random disc obstacles. Coarse-to-fine
``N = 64 -> 128 -> 256`` with an area-based stall-stop. The result is paper-quality
serpentine tracks (Henrich et al., *Generating Race Tracks With Repulsive Curves*;
Yu/Schumacher/Crane, *Repulsive Curves* SIGGRAPH 2021).

This is the FIRST non-CUDA-graph-capturable generator: the growth loop records a
fresh ``wp.Tape`` per iteration (autodiff) and reads back an area-convergence scalar
every window to drive stall-stop -- both illegal inside a capture region. It registers
with ``capturable=False`` and ``TrackGenerator`` runs it EAGERLY on CUDA (see the cost
warning on ``TrackGenConfig.repulsive_grow_mult_min``).

Ported from the validated spike ``docs/superpowers/spikes/2026-07-05-repulsive-growth-
phase1/grow_warp.py`` (proven 64/64 through the standard tail). Production changes:
(a) obstacle layout is a seed-driven Warp kernel (not host torch rejection sampling);
(b) scratch is allocated once, coarse stages slice the ``[0:E*N_stage]`` prefix;
(c) the stage upsample uses ``warp_pipeline.arc_length_resample_inplace``.

**torch-free:** imports only ``warp`` + ``numpy`` (host precompute uses ``numpy.fft``
for the Sobolev circulant rows and ``numpy.log1p`` for the stage schedule).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import warp_pipeline as _pipe

# --- Obstacle layout constants (mirror the spike's sample_obstacles reference layout) ---
# Independent RNG salt, distinct from the site/anchor/checkpoint/control salts.
_OBSTACLE_SALT = 6247
_N_WALL = 96          # points on the domain-wall ring (weight 1.0)
_N_DISC = 12          # points per inner-disc ring (weight 0.25)
_C_FRAC = 0.90        # inner-disc radial-band outer bound as a fraction of r_dom
_WALL_WEIGHT = 1.0
_DISC_WEIGHT = 0.25

# --- World-scale anchor (design §5, amended) ---
# All absolute sizes anchor to ``config.scale`` like every other generator (polar/voronoi
# anchor to the bezier extent 1.44). ``_DOMAIN_SCALE_REF`` is the world-scale reference
# LENGTH in config.scale units: at scale=1.0 it equals the spike's build_setup median
# bezier perimeter, so the domain radius / init radius / obstacle radii / target lengths
# are numerically identical to the spike's validated 64/64 regime.
#
# NOTE (amended from the spec's §5): the design cited a bezier-coupled, per-batch-measured
# ``_BEZIER_PERIMETER_REF ≈ 5.05``. That is REPLACED here by a fixed scale-anchored
# constant (no bezier coupling, no per-batch measurement, no final rescale). The exact
# value is the spike's *actual* build_setup median at scale=1 (E=64, the validated config),
# which measures 4.9029 today -- the design's 5.05 was an earlier/approximate figure that
# has drifted with the bezier defaults (exactly the risk the design flagged). Using the
# spike-derived value keeps the generator IDENTICAL to the validated ground truth.
_DOMAIN_SCALE_REF = 4.9029


class _RepulsiveScalars:
    """Host-side derived scalars for one config (geometry + obstacle layout bounds)."""

    __slots__ = (
        "E", "P_ref", "r_dom", "r_init", "M", "n_wall", "n_disc",
        "k_min", "k_max", "r_frac_lo", "r_frac_hi", "c_frac",
    )

    def __init__(self, config) -> None:
        self.E = int(config.num_envs)
        self.P_ref = _DOMAIN_SCALE_REF * float(config.scale)
        self.r_dom = float(config.repulsive_domain_frac) * self.P_ref
        self.r_init = self.r_dom / float(config.repulsive_domain_init_ratio)
        self.n_wall = _N_WALL
        self.n_disc = _N_DISC
        self.k_min = int(config.repulsive_obstacle_count_min)
        self.k_max = int(config.repulsive_obstacle_count_max)
        # M sizes the obstacle buffer at the max possible disc count (K_max = k_max).
        self.M = self.n_wall + self.k_max * self.n_disc
        self.r_frac_lo = float(config.repulsive_obstacle_radius_min_frac)
        self.r_frac_hi = float(config.repulsive_obstacle_radius_max_frac)
        self.c_frac = _C_FRAC


# ===========================================================================
# Seed-driven, device-side obstacle layout (replaces the spike's host torch
# rejection sampling). Deterministic per (seed, env); no rejection loop.
# ===========================================================================

@wp.kernel
def _sample_obstacles_k(
    seeds: wp.array(dtype=wp.int32),
    r_dom: float, r_init: float,
    r_frac_lo: float, r_frac_hi: float, c_frac: float,
    k_min: int, k_max: int, n_wall: int, n_disc: int, m_obs: int,
    nan_val: float,
    obs_pts: wp.array(dtype=wp.vec2f), obs_mw: wp.array(dtype=wp.float32),
):
    # One thread per env. Writes a domain-wall ring (weight 1.0) plus k inner-disc rings
    # (weight 0.25) at angular-stratified positions with an analytic radial band -- no
    # rejection loop. Unused / skipped disc columns are NaN-padded with weight 0 so the
    # obstacle-energy kernel (which guards on ``mw != 0``) never reads them.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _OBSTACLE_SALT + 23)
    two_pi = 2.0 * wp.pi
    base = e * m_obs

    # --- domain-wall ring at r_dom (weight 1.0; mass = ring arc spacing) ---
    r_wall = r_dom
    wall_mw = (two_pi * r_wall / float(n_wall)) * _WALL_WEIGHT
    for iw in range(n_wall):
        ang = float(iw) * two_pi / float(n_wall)
        obs_pts[base + iw] = wp.vec2f(r_wall * wp.cos(ang), r_wall * wp.sin(ang))
        obs_mw[base + iw] = wall_mw

    # --- k inner discs, one per 2*pi/k wedge, per-env phase ---
    k = wp.randi(state, k_min, k_max + 1)
    phase = wp.randf(state) * two_pi
    for j in range(k_max):
        col = base + n_wall + j * n_disc
        if j < k:
            r_disc = (r_frac_lo + (r_frac_hi - r_frac_lo) * wp.randf(state)) * r_dom
            # Radial band: the ring's inner edge clears the init circle by 0.05*r_dom (else
            # the p=beta-alpha energy blows up at t=0); the outer bound c_frac*r_dom stays
            # inside the wall. If the band is empty (tight config) the column is left unused.
            lo = r_init + r_disc + 0.05 * r_dom
            hi = c_frac * r_dom
            rad = lo + (hi - lo) * wp.randf(state)
            active = float(1.0)
            if lo >= hi:
                active = float(0.0)
            ang = phase + float(j) * two_pi / float(k)
            cx = rad * wp.cos(ang)
            cy = rad * wp.sin(ang)
            disc_mw = (two_pi * r_disc / float(n_disc)) * _DISC_WEIGHT
            for idx in range(n_disc):
                a = float(idx) * two_pi / float(n_disc)
                if active == 1.0:
                    obs_pts[col + idx] = wp.vec2f(cx + r_disc * wp.cos(a),
                                                  cy + r_disc * wp.sin(a))
                    obs_mw[col + idx] = disc_mw
                else:
                    obs_pts[col + idx] = wp.vec2f(nan_val, nan_val)
                    obs_mw[col + idx] = 0.0
        else:
            for idx in range(n_disc):
                obs_pts[col + idx] = wp.vec2f(nan_val, nan_val)
                obs_mw[col + idx] = 0.0


def _sample_obstacles_inplace(seeds_wp: wp.array, config,
                              obs_pts: wp.array, obs_mw: wp.array,
                              device: str) -> None:
    """Launch the seed-driven obstacle kernel in place. obs_pts/obs_mw are [E*M]."""
    _pipe._init()
    s = _RepulsiveScalars(config)
    wp.launch(_sample_obstacles_k, dim=s.E,
              inputs=[seeds_wp, s.r_dom, s.r_init, s.r_frac_lo, s.r_frac_hi, s.c_frac,
                      s.k_min, s.k_max, s.n_wall, s.n_disc, s.M, float("nan"),
                      obs_pts, obs_mw],
              device=device)
    _pipe._sync(device)
