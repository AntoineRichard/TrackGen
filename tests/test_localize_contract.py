"""TrackLocalizer construction/query contracts: validation, reuse, clone."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._collision_fixtures import make_annulus_track
from track_gen.localize import TrackFrame, TrackLocalizer


def _positions(pts, device="cpu"):
    return wp.array(np.asarray(pts, np.float32), dtype=wp.vec2f, device=device)


def test_constructor_validation():
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="warm_window"):
        TrackLocalizer(track, warm_window=0)
    with pytest.raises(ValueError, match="warm_window"):
        TrackLocalizer(track, warm_window=-3)


def test_query_validation():
    track = make_annulus_track(E=2, n=64)
    loc = TrackLocalizer(track)
    with pytest.raises(ValueError, match="position"):
        loc.query(_positions([[1.0, 0.0]]))  # wrong shape ([1] for E=2)
    with pytest.raises(ValueError, match="position"):
        loc.query(wp.zeros(2, dtype=wp.float32))  # wrong dtype
    with pytest.raises(ValueError, match="position"):
        loc.query(np.zeros((2, 2), np.float32))  # not a wp.array
    with pytest.raises(ValueError, match="mask"):
        loc.reset(wp.zeros(3, dtype=wp.int32))  # wrong shape


def test_bound_mode_contracts():
    track = make_annulus_track(E=2, n=64)
    pos = _positions([[1.0, 0.0], [0.0, 1.0]])
    free = TrackLocalizer(track)
    bound = TrackLocalizer(track, position=pos)
    r_free = free.query(pos).clone()
    r_bound = bound.query()
    np.testing.assert_array_equal(r_bound.s.numpy(), r_free.s.numpy())
    np.testing.assert_array_equal(r_bound.n.numpy(), r_free.n.numpy())
    np.testing.assert_array_equal(r_bound.segment.numpy(),
                                  r_free.segment.numpy())
    with pytest.raises(ValueError, match="bound"):
        bound.query(pos)
    with pytest.raises(ValueError, match="not bound"):
        free.query()


def test_query_returns_same_instance_and_clone_detaches():
    track = make_annulus_track(E=1, n=64)
    loc = TrackLocalizer(track)
    f1 = loc.query(_positions([[1.2, 0.0]]))
    assert isinstance(f1, TrackFrame)
    snap = f1.clone()
    n_before = float(f1.n.numpy()[0])
    f2 = loc.query(_positions([[0.9, 0.0]]))
    assert f2 is f1
    assert float(f1.n.numpy()[0]) != n_before
    np.testing.assert_allclose(float(snap.n.numpy()[0]), n_before)


def test_sign_convention_and_arc_length_on_annulus():
    # CCW annulus, r_center=1: n follows Track.normal — positive toward the
    # OUTER boundary (radius > 1), negative toward the inner one.
    track = make_annulus_track(E=3, n=256)
    length = track.length.numpy()
    theta = np.deg2rad(np.array([0.0, 130.0, 275.0]))
    r = np.array([1.2, 0.85, 1.0])
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    f = TrackLocalizer(track).query(_positions(pts))
    s, n = f.s.numpy(), f.n.numpy()
    np.testing.assert_allclose(n, [0.2, -0.15, 0.0], atol=1e-2)
    np.testing.assert_allclose(s, theta / (2.0 * np.pi) * length, atol=2e-2)
    assert np.all(f.segment.numpy() >= 0)


def test_nan_position_yields_nan_frame():
    track = make_annulus_track(E=2, n=64)
    loc = TrackLocalizer(track, warm_window=4)
    pts = np.array([[np.nan, np.nan], [1.0, 0.0]], np.float32)
    f = loc.query(_positions(pts))
    assert np.isnan(f.s.numpy()[0]) and np.isnan(f.n.numpy()[0])
    assert int(f.segment.numpy()[0]) == -1
    assert np.isfinite(f.s.numpy()[1])
    assert int(f.segment.numpy()[1]) >= 0


def test_degenerate_track_yields_nan_frame():
    track = make_annulus_track(E=2, n=64, counts=[2, 64])  # env 0: count < 3
    f = TrackLocalizer(track).query(_positions([[1.0, 0.0], [1.0, 0.0]]))
    assert np.isnan(f.s.numpy()[0]) and int(f.segment.numpy()[0]) == -1
    assert np.isfinite(f.s.numpy()[1])
