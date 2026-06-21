import numpy as np
import pytest
from benchmarks import track_metrics as m


def _circle(n=256, r=2.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(s=3.0, n_per_side=50):
    side = np.linspace(0, s, n_per_side, endpoint=False)
    top = np.stack([side, np.full_like(side, s)], 1)
    right = np.stack([np.full_like(side, s), s - side], 1)
    bottom = np.stack([s - side, np.zeros_like(side)], 1)
    left = np.stack([np.zeros_like(side), side], 1)
    return np.concatenate([top, right, bottom, left])


def _figure_eight(n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.sin(t), np.sin(t) * np.cos(t)], axis=1)


def test_perimeter_and_area_of_circle():
    c = _circle(n=512, r=2.0)
    assert m.perimeter(c) == pytest.approx(2 * np.pi * 2.0, rel=1e-3)
    assert m.polygon_area(c) == pytest.approx(np.pi * 2.0 ** 2, rel=1e-2)


def test_compactness_circle_near_one_square_less():
    assert m.compactness(_circle(512)) == pytest.approx(1.0, abs=1e-2)
    assert m.compactness(_square()) == pytest.approx(np.pi / 4, rel=5e-2)


def test_curvature_of_circle_is_constant_inverse_radius():
    r = 2.0
    k = m.curvature(_circle(n=512, r=r))
    assert np.allclose(k, 1.0 / r, rtol=5e-2)


def test_self_intersection_detection():
    assert not m.self_intersects(_circle(64))
    assert m.self_intersects(_figure_eight(200))


def test_racing_line_proxy_keys_and_circle_values():
    out = m.racing_line_proxy(_circle(n=512, r=2.0), a_lat_max=1.0)
    assert set(out) == {"peak_curvature", "integral_kappa2", "lap_time"}
    # circle: constant curvature 1/r -> constant speed v=sqrt(a_lat/k); lap_time = perim/v
    assert out["peak_curvature"] == pytest.approx(0.5, rel=5e-2)
    assert out["lap_time"] > 0
