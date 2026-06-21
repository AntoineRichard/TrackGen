import numpy as np
from track_gen._experimental import grammar_proto as gp


def test_close_and_integrate_returns_closed_loop():
    # a hand-built curvature profile must integrate to a CLOSED loop (edges sum ~0).
    N = 256
    kappa = np.zeros(N)
    kappa[:64] = 0.05          # one arc span; the rest straight-ish
    pts = gp.close_and_integrate(kappa)
    assert pts.shape == (N, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    assert np.linalg.norm(edges.sum(axis=0)) < 1e-6   # closed: edge vectors sum to zero


def test_generate_centerline_is_deterministic_and_finite():
    a = gp.generate_centerline(7, gp.DEFAULTS)
    b = gp.generate_centerline(7, gp.DEFAULTS)
    assert np.array_equal(a, b)                        # deterministic in seed
    assert np.isfinite(a).all()
    assert a.shape[0] == gp.DEFAULTS["num_points"]
