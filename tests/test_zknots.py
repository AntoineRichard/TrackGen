"""Knot sampling + periodic monotone-cubic interpolation for track elevation."""
import numpy as np
import pytest
import warp as wp

from track_gen._src import warp_zprofile
from track_gen._src.types import TrackGenConfig

E, K, N = 2, 4, 240


def _eval(knots, perim=1.0, n=N):
    """Evaluate _pchip_eval_k on one env at n uniformly-spaced arc positions."""
    kz = np.tile(np.asarray(knots, np.float32), (E, 1)).reshape(-1)
    arc = np.tile((np.arange(n, dtype=np.float32) / n) * perim, (E, 1)).reshape(-1)
    dev = "cpu"
    out = wp.zeros(E * n, dtype=wp.float32, device=dev)
    wp.launch(
        warp_zprofile._pchip_eval_k, dim=E * n,
        inputs=[wp.array(arc, dtype=wp.float32, device=dev),
                wp.array(np.full(E, perim, np.float32), dtype=wp.float32, device=dev),
                wp.array(np.full(E, n, np.int32), dtype=wp.int32, device=dev),
                n,
                wp.array(kz, dtype=wp.float32, device=dev),
                len(knots),
                out],
        device=dev)
    return out.numpy().reshape(E, n)[0]


def test_interpolates_knots_exactly():
    # knot j sits at arc fraction j/K -> sample index j*n/K
    knots = [0.0, 1.0, 0.5, 2.0]
    z = _eval(knots)
    for j, kv in enumerate(knots):
        assert abs(z[j * N // K] - kv) < 1e-5, f"knot {j}"


def test_no_overshoot_on_alternating_knots():
    # The classic overshoot trap: Catmull-Rom would exceed [0, 1] here.
    knots = [0.0, 1.0, 0.0, 1.0]
    z = _eval(knots)
    assert z.max() <= 1.0 + 1e-5
    assert z.min() >= 0.0 - 1e-5


def test_no_overshoot_on_monotone_run():
    knots = [0.0, 1.0, 2.0, 3.0]
    z = _eval(knots)
    assert z.max() <= 3.0 + 1e-5 and z.min() >= 0.0 - 1e-5


def test_flat_knots_give_flat_curve():
    z = _eval([1.25, 1.25, 1.25, 1.25])
    np.testing.assert_allclose(z, 1.25, atol=1e-6)


def test_periodic_closure_is_smooth():
    # Wrapping past the last knot must return toward knot 0 continuously:
    # the value just before the seam is close to the value just after it.
    knots = [0.0, 1.0, 0.5, 2.0]
    z = _eval(knots, n=1000)
    assert abs(z[-1] - z[0]) < 3.0 * abs(z[1] - z[0]) + 1e-4


def test_padding_slots_are_zero():
    dev = "cpu"
    n = 8
    out = wp.zeros(n, dtype=wp.float32, device=dev)
    wp.launch(
        warp_zprofile._pchip_eval_k, dim=n,
        inputs=[wp.array(np.linspace(0, 1, n, dtype=np.float32), dtype=wp.float32, device=dev),
                wp.array(np.array([1.0], np.float32), dtype=wp.float32, device=dev),
                wp.array(np.array([5], np.int32), dtype=wp.int32, device=dev),  # count=5 of 8
                n,
                wp.array(np.array([0.0, 1.0, 0.0, 1.0], np.float32), dtype=wp.float32, device=dev),
                4,
                out],
        device=dev)
    assert (out.numpy()[5:] == 0.0).all()


def test_knot_pipeline_bounds_and_smoothness():
    """End-to-end through apply_z_profile_knots: uniform draws at K knots,
    interpolated over many points, stays in bounds and stays smooth."""
    Kc, S = 8, 200
    cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="uniform",
                         z_min=0.5, z_max=1.5, z_control_points=Kc)
    dev = "cpu"
    perim_np = np.full(E, 4.0, np.float32)
    arc_np = np.tile(np.linspace(0.0, 4.0, S, endpoint=False, dtype=np.float32), (E, 1)).reshape(-1)
    knot_cum, knot_count, knot_z = warp_zprofile.alloc_knot_scratch(E, Kc, dev)
    z = wp.zeros(E * S, dtype=wp.float32, device=dev)
    warp_zprofile.apply_z_profile_knots(
        cfg, wp.array(np.array([3, 11], np.int32), dtype=wp.int32, device=dev),
        wp.array(np.full(E, S, np.int32), dtype=wp.int32, device=dev), S,
        wp.array(arc_np, dtype=wp.float32, device=dev),
        wp.array(perim_np, dtype=wp.float32, device=dev),
        knot_cum, knot_count, knot_z, z)
    zz = z.numpy().reshape(E, S)
    kz = knot_z.numpy().reshape(E, Kc)
    for e in range(E):
        assert (zz[e] >= 0.5 - 1e-5).all() and (zz[e] <= 1.5 + 1e-5).all()
        # no overshoot beyond the sampled knots
        assert zz[e].max() <= kz[e].max() + 1e-5
        assert zz[e].min() >= kz[e].min() - 1e-5
        # smooth: far fewer direction changes than points
        turns = int(np.sum(np.diff(np.sign(np.diff(zz[e]))) != 0))
        assert turns <= Kc, f"{turns} direction changes for K={Kc}"


def test_config_validation():
    with pytest.raises(ValueError, match="z_control_points"):
        TrackGenConfig(device="cpu", num_envs=1, z_control_points=2)
    assert TrackGenConfig(device="cpu", num_envs=1).z_control_points == 10
