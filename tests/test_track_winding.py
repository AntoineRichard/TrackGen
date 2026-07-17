"""Track.winding sign + Track.normal direction on generated tracks.

Pins the winding-dependent conventions on FULL-pipeline tracks of both
windings (bezier winds CW, polar winds CCW). Mirrors the empirical style of
tests/test_localize.py::test_sign_convention_on_generated_tracks: the annulus
fixture cannot exercise this because its hand-set fields do not come from the
pipeline.
"""
from __future__ import annotations

import numpy as np

import pytest

pytest.importorskip("warp")


def _shoelace(poly):
    """Signed area of a closed polygon: positive for CCW winding."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _generate(generator, E, seeds):
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    cfg = TrackGenConfig(num_envs=E, device="cpu", generator=generator)
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
    return TrackGenerator(cfg, rng).generate()


def test_winding_matches_shoelace_sign_on_generated_tracks():
    # Track.winding is +1 CCW / -1 CW, the sign of the centerline's signed
    # area. bezier winds CW (-1), polar winds CCW (+1).
    checked = 0
    for generator, expect in (("bezier", -1.0), ("polar", 1.0)):
        E = 3
        track = _generate(generator, E, seeds=7)
        valid = track.valid.numpy()
        counts = track.count.numpy()
        winding = track.winding.numpy()
        n_max = track.center.shape[0] // E
        center = track.center.numpy().reshape(E, n_max, 3)[..., :2]
        for e in range(E):
            if not valid[e]:
                continue
            m = int(counts[e])
            ref = 1.0 if _shoelace(center[e, :m]) > 0.0 else -1.0
            assert winding[e] == ref, \
                f"{generator} env {e}: winding {winding[e]} vs shoelace {ref}"
            assert winding[e] == expect, \
                f"{generator} env {e}: unexpected winding {winding[e]}"
            checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"


def test_normal_direction_is_winding_conditional_on_generated_tracks():
    # Track.normal is the LEFT normal, so which boundary it faces depends on
    # winding: it points toward OUTER for CW loops (dot(normal, outer-center)
    # > 0) and toward INNER for CCW loops (dot < 0). Equivalently the sign of
    # dot(normal, outer-center) is always the OPPOSITE of winding.
    checked = 0
    for generator in ("bezier", "polar"):
        E = 3
        track = _generate(generator, E, seeds=7)
        valid = track.valid.numpy()
        counts = track.count.numpy()
        winding = track.winding.numpy()
        n_max = track.center.shape[0] // E
        center = track.center.numpy().reshape(E, n_max, 3)[..., :2]
        outer = track.outer.numpy().reshape(E, n_max, 3)[..., :2]
        normal = track.normal.numpy().reshape(E, n_max, 3)[..., :2]
        for e in range(E):
            if not valid[e]:
                continue
            m = int(counts[e])
            for i in range(0, m, max(1, m // 8)):
                d = float(np.dot(normal[e, i], outer[e, i] - center[e, i]))
                assert d * winding[e] < 0.0, \
                    (f"{generator} env {e} pt {i}: dot(normal, outer-center)"
                     f"={d} not opposite winding {winding[e]}")
            checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"
