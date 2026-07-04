"""Sanity tests for the annulus fixture and the numpy oracle (tests of tests)."""
from __future__ import annotations

import numpy as np

from tests._collision_fixtures import annulus_polylines, make_annulus_track, make_boxes
from tests import _collision_oracle as oracle


def test_annulus_track_layout():
    E, n = 3, 128
    track = make_annulus_track(E=E, n=n, counts=[128, 100, 64])
    N_max = n + 8
    assert track.outer.shape == (E * N_max,)
    counts = track.count.numpy()
    assert list(counts) == [128, 100, 64]
    outer = track.outer.numpy().reshape(E, N_max, 2)
    # Real points on radius 1.3, NaN tail past count[e].
    r = np.linalg.norm(outer[1, :100], axis=1)
    np.testing.assert_allclose(r, 1.3, atol=1e-5)
    assert np.all(np.isnan(outer[1, 100:]))


def test_oracle_inside_box_clearance():
    track = make_annulus_track(E=1, n=512)
    inner, outer = annulus_polylines(track, 0, 512 + 8)
    # Axis-aligned box at (1.1, 0): outer gap = 1.3 - |far corner|.
    res = oracle.box_contact(inner, outer, (1.1, 0.0), 0.0, (0.05, 0.05))
    far = np.hypot(1.15, 0.05)
    assert res["oob"] == 0
    np.testing.assert_allclose(res["distance"], 1.3 - far, atol=2e-3)
    assert res["boundary"] == 1
    np.testing.assert_allclose(np.linalg.norm(res["nearest"]), 1.3, atol=2e-3)


def test_oracle_in_hole_and_crossing():
    track = make_annulus_track(E=1, n=512)
    inner, outer = annulus_polylines(track, 0, 512 + 8)
    hole = oracle.box_contact(inner, outer, (0.0, 0.0), 0.3, (0.1, 0.05))
    assert hole["oob"] == 1 and hole["distance"] < 0
    cross = oracle.box_contact(inner, outer, (1.3, 0.0), 0.0, (0.1, 0.1))
    assert cross["oob"] == 1
    # Deepest corner (1.4, +-0.1): penetration = |corner| - 1.3.
    pen = np.hypot(1.4, 0.1) - 1.3
    np.testing.assert_allclose(cross["distance"], -pen, atol=2e-3)


def test_make_boxes_nan_padding():
    pos, yaw, he = make_boxes(2, 4, {(0, 0): (1.0, 0.0, 0.1, 0.05, 0.02)})
    p = pos.numpy().reshape(-1, 2)
    assert not np.any(np.isnan(p[0]))
    assert np.all(np.isnan(p[1:]))
    assert yaw.numpy()[0] == np.float32(0.1)
