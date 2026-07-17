"""TrackLocalizer correctness: numpy oracle, warm/cold equivalence, resets."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests import _localize_oracle as oracle
from tests._collision_fixtures import make_annulus_track
from track_gen.localize import TrackLocalizer


def _positions(pts, device="cpu"):
    pts = np.asarray(pts, np.float32)
    z = np.zeros((pts.shape[0], 1), np.float32)
    return wp.array(np.concatenate([pts, z], axis=1), dtype=wp.vec3f,
                    device=device)


def _real_env(track, e, n_max):
    """(center, arclen, length) of env e without the NaN tail, as float64."""
    m = int(track.count.numpy()[e])
    center = track.center.numpy().reshape(-1, 3)[e * n_max:e * n_max + m, :2]
    arclen = track.arclen.numpy()[e * n_max:e * n_max + m]
    return center.astype(np.float64), arclen.astype(np.float64), \
        float(track.length.numpy()[e])


def _wrap_dist(a, b, length):
    d = abs(a - b)
    return min(d, length - d)


def test_generated_tracks_projection_oracle():
    """Random query points vs brute-force projection on REAL tracks (cpu)."""
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.center.shape[0] // E
    loc = TrackLocalizer(track)
    rng = np.random.default_rng(0)
    center_all = track.center.numpy().reshape(E, n_max, 3)[..., :2]

    checked = 0
    for trial in range(8):
        pts = np.zeros((E, 2), np.float32)
        for e in range(E):
            i = int(rng.integers(0, counts[e]))
            pts[e] = center_all[e, i] + rng.normal(0.0, 0.15, 2)
        f = loc.query(_positions(pts))
        s, n, seg = f.s.numpy(), f.n.numpy(), f.segment.numpy()
        n_up = f.n_up.numpy()
        for e in range(E):
            if not valid[e]:
                continue
            c, al, L = _real_env(track, e, n_max)
            s_ref, n_ref, seg_ref = oracle.project(c, al, L, pts[e])
            assert 0.0 <= float(s[e]) < L, \
                f"trial {trial} env {e}: s {s[e]} outside [0, {L})"
            assert _wrap_dist(float(s[e]), s_ref, L) < 1e-3, \
                f"trial {trial} env {e}: s {s[e]} vs {s_ref}"
            np.testing.assert_allclose(n[e], n_ref, atol=1e-4,
                                       err_msg=f"trial {trial} env {e}")
            # Planar track, z = 0 query: vertical offset is exactly zero.
            assert n_up[e] == 0.0, f"trial {trial} env {e}: n_up {n_up[e]}"
            # Near a shared vertex two adjacent segments tie to float
            # precision; accept either as long as the arc length agrees.
            assert seg[e] == seg_ref or \
                _wrap_dist(float(s[e]), s_ref, L) < 1e-3
            checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"


def test_warm_start_matches_cold_scan_on_trajectory():
    # Small per-step motion (well inside the warm window) must make warm and
    # cold scans pick the same segment, hence bitwise-identical results.
    E = 2
    track = make_annulus_track(E=E, n=256)
    cold = TrackLocalizer(track)
    warm = TrackLocalizer(track, warm_window=8)
    rng = np.random.default_rng(7)
    theta = np.zeros(E)
    for _ in range(60):
        theta = theta + np.deg2rad(rng.uniform(0.5, 3.0, E))  # < 3 segments
        r = 1.0 + rng.uniform(-0.25, 0.25, E)
        pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        fc = cold.query(_positions(pts)).clone()
        fw = warm.query(_positions(pts))
        np.testing.assert_array_equal(fw.segment.numpy(), fc.segment.numpy())
        np.testing.assert_array_equal(fw.s.numpy(), fc.s.numpy())
        np.testing.assert_array_equal(fw.n.numpy(), fc.n.numpy())


def test_reset_recovers_from_teleport():
    E = 2
    track = make_annulus_track(E=E, n=256)
    cold = TrackLocalizer(track)
    warm = TrackLocalizer(track, warm_window=4)
    a = _positions([[1.1, 0.0]] * E)
    b = _positions([[-1.1, 0.05]] * E)  # opposite side: outside the window
    warm.query(a)
    warm.reset(wp.full(E, 1, dtype=wp.int32, device="cpu"))
    fw = warm.query(b)
    fc = cold.query(b)
    np.testing.assert_array_equal(fw.segment.numpy(), fc.segment.numpy())
    np.testing.assert_array_equal(fw.s.numpy(), fc.s.numpy())


def test_nan_position_drops_warm_memory():
    # NaN pause re-arms the full scan, so a subsequent far-away point is
    # localized exactly even without an explicit reset.
    E = 1
    track = make_annulus_track(E=E, n=256)
    cold = TrackLocalizer(track)
    warm = TrackLocalizer(track, warm_window=4)
    warm.query(_positions([[1.1, 0.0]]))
    warm.query(_positions([[np.nan, np.nan]]))
    far = _positions([[-1.05, -0.2]])
    fw = warm.query(far)
    fc = cold.query(far)
    np.testing.assert_array_equal(fw.segment.numpy(), fc.segment.numpy())
    np.testing.assert_array_equal(fw.s.numpy(), fc.s.numpy())


def test_sees_track_buffer_updates_after_reset():
    # The localizer aliases the Track buffers; writing new geometry into the
    # SAME buffers (as TrackGenerator.generate() does) plus a reset must be
    # reflected in queries.
    track = make_annulus_track(E=1, n=64)
    loc = TrackLocalizer(track, warm_window=4)
    p = _positions([[1.2, 0.0]])
    np.testing.assert_allclose(loc.query(p).n.numpy(), [0.2], atol=1e-2)
    bigger = make_annulus_track(E=1, n=64, r_center=3.0)  # same shapes
    for name in ("center", "arclen", "length"):
        wp.copy(getattr(track, name), getattr(bigger, name))
    loc.reset(wp.full(1, 1, dtype=wp.int32, device="cpu"))
    np.testing.assert_allclose(loc.query(p).n.numpy(), [-1.8], atol=1e-2)


def _shoelace(poly):
    """Signed area of a closed polygon: positive for CCW winding."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _generate(generator, E, seeds):
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    cfg = TrackGenConfig(num_envs=E, device="cpu", generator=generator)
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
    return TrackGenerator(cfg, rng).generate()


def test_sign_convention_on_generated_tracks():
    # n is positive to the RIGHT of the direction of travel, so which
    # boundary is positive depends on the loop's winding: bezier loops wind
    # CW (outer on the LEFT -> toward-outer n < 0), polar loops wind CCW
    # (outer on the RIGHT -> toward-outer n > 0). The annulus fixture cannot
    # test this: its hand-set normals do not come from the pipeline.
    checked = 0
    for generator, winding in (("bezier", -1.0), ("polar", 1.0)):
        E = 2
        track = _generate(generator, E, seeds=7)
        valid = track.valid.numpy()
        counts = track.count.numpy()
        n_max = track.center.shape[0] // E
        center = track.center.numpy().reshape(E, n_max, 3)[..., :2]
        outer = track.outer.numpy().reshape(E, n_max, 3)[..., :2]
        inner = track.inner.numpy().reshape(E, n_max, 3)[..., :2]
        loc = TrackLocalizer(track)
        for boundary, toward in (("outer", outer), ("inner", inner)):
            pts = np.full((E, 2), np.nan, np.float32)
            expect = np.zeros(E)
            for e in range(E):
                if not valid[e]:
                    continue
                m = int(counts[e])
                w = 1.0 if _shoelace(center[e, :m]) > 0.0 else -1.0
                assert w == winding, \
                    f"{generator} env {e}: unexpected winding"
                i = m // 3
                # Halfway from the centerline toward the boundary: on-road,
                # unambiguous side.
                pts[e] = center[e, i] + 0.5 * (toward[e, i] - center[e, i])
                expect[e] = w if boundary == "outer" else -w
            n = loc.query(_positions(pts)).n.numpy()
            for e in range(E):
                if expect[e] == 0.0:
                    continue
                assert n[e] * expect[e] > 0.0, \
                    f"{generator} env {e}: toward {boundary} gave n={n[e]}"
                checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"


def test_warm_matches_cold_two_laps_on_generated_tracks():
    # Heterogeneous per-env counts, two full laps, wrap seam crossed
    # forward: warm and cold must stay bitwise identical throughout (points
    # hug the centerline, so no pinch ambiguity is in play).
    E = 4
    track = _generate("bezier", E, seeds=123)
    valid = track.valid.numpy().astype(bool)
    counts = track.count.numpy()
    assert len(set(counts[valid].tolist())) > 1, \
        "want heterogeneous counts — pick another seed"
    n_max = track.center.shape[0] // E
    center = track.center.numpy().reshape(E, n_max, 3)[..., :2]
    cold = TrackLocalizer(track)
    warm = TrackLocalizer(track, warm_window=8)
    rng = np.random.default_rng(3)
    steps = int(counts[valid].max()) + 4  # 2 points/step -> >= 2 laps each
    for step in range(steps):
        pts = np.zeros((E, 2), np.float32)
        for e in range(E):
            i = (2 * step) % int(counts[e])
            pts[e] = center[e, i] + rng.normal(0.0, 0.02, 2)
        pos = _positions(pts)
        fc = cold.query(pos).clone()
        fw = warm.query(pos)
        np.testing.assert_array_equal(fw.segment.numpy()[valid],
                                      fc.segment.numpy()[valid])
        np.testing.assert_array_equal(fw.s.numpy()[valid],
                                      fc.s.numpy()[valid])
        np.testing.assert_array_equal(fw.n.numpy()[valid],
                                      fc.n.numpy()[valid])
