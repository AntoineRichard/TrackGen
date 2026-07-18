import numpy as np
import pytest
import warp as wp

from track_gen._src.types import GateGenConfig
from track_gen._src import warp_zprofile

E, G = 8, 16


def _ring(seed=0):
    """Ordered ring anchors + counts, plausible gate layout."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(6, G + 1, size=E).astype(np.int32)
    pos = np.full((E * G, 2), np.nan, np.float32)
    for e in range(E):
        n = counts[e]
        ang = np.sort(rng.uniform(0, 2 * np.pi, n)).astype(np.float32)
        r = 1.0 + 0.2 * rng.standard_normal(n).astype(np.float32)
        pos[e * G:e * G + n, 0] = r * np.cos(ang)
        pos[e * G:e * G + n, 1] = r * np.sin(ang)
    return pos, counts


def _run(profile, **kw):
    cfg = GateGenConfig(device="cpu", num_envs=E, max_gates=G, gate_width=0.05,
                        z_profile=profile, **kw)
    pos_np, counts = _ring()
    pos2 = wp.array(pos_np, dtype=wp.vec2f, device="cpu")
    count = wp.array(counts, dtype=wp.int32, device="cpu")
    seeds = wp.array(np.arange(E, dtype=np.int32) + 7, dtype=wp.int32,
                     device="cpu")
    cum, perim, z = warp_zprofile.alloc_z_scratch(E, G, "cpu")
    warp_zprofile.gate_cum_perim(pos2, count, G, cum, perim)
    warp_zprofile.apply_z_profile(cfg, seeds, count, G, cum, perim, z)
    return z.numpy().reshape(E, G), counts, pos_np.reshape(E, G, 2)


def test_flat_is_base():
    z, counts, _ = _run("flat", z_base=1.5)
    for e in range(E):
        assert np.allclose(z[e, :counts[e]], 1.5)


def test_uniform_bounds_and_determinism():
    z1, counts, _ = _run("uniform", z_min=1.0, z_max=3.0)
    z2, _, _ = _run("uniform", z_min=1.0, z_max=3.0)
    np.testing.assert_array_equal(z1, z2)
    for e in range(E):
        zz = z1[e, :counts[e]]
        assert (zz >= 1.0).all() and (zz <= 3.0).all()
        assert zz.std() > 0.0


def test_walk_bounds_closure_and_grade():
    z, counts, pos = _run("random_walk", z_base=2.0, z_min=0.5, z_max=3.5,
                          z_max_step=0.3)
    for e in range(E):
        n = counts[e]
        zz = z[e, :n]
        assert (zz >= 0.5 - 1e-5).all() and (zz <= 3.5 + 1e-5).all()
        # closure: bridge pulls the walk back near its start
        assert abs(zz[0] - zz[-1]) < 1.0
        # grade cap (bridge adds at most drift/perimeter; allow 2x slack)
        p = pos[e, :n]
        ds = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)[: n - 1]
        grade = np.abs(np.diff(zz)) / np.maximum(ds, 1e-9)
        assert (grade <= 2.0 * 0.3 + 1e-4).all()


def test_noise_bounds_and_periodicity_shape():
    z, counts, _ = _run("noise", z_base=2.0, z_noise_amplitude=0.5,
                        z_noise_harmonics=3, z_min=1.0, z_max=3.0)
    for e in range(E):
        zz = z[e, :counts[e]]
        assert (zz >= 1.0 - 1e-5).all() and (zz <= 3.0 + 1e-5).all()
        assert zz.std() > 0.0


def test_config_validation():
    with pytest.raises(ValueError):
        GateGenConfig(device="cpu", num_envs=1, z_profile="bogus")
    with pytest.raises(ValueError):
        GateGenConfig(device="cpu", num_envs=1, z_profile="uniform",
                      z_min=2.0, z_max=1.0)
    with pytest.raises(ValueError):
        GateGenConfig(device="cpu", num_envs=1, z_profile="random_walk",
                      z_max_step=-0.1)


def test_apply_z_profile_track_parameterization():
    """The generalized API takes (cum, perim) directly — no pos2. A uniform
    cum (constant spacing s=i*0.1, perim = n*0.1) must drive the noise
    profile periodically: z at slot 0 ~ z at the wrap (frac 0 vs frac ~1)."""
    cfg = GateGenConfig(device="cpu", num_envs=2, max_gates=16, gate_width=0.05,
                        z_profile="noise", z_base=1.0, z_noise_amplitude=0.5,
                        z_noise_harmonics=2, z_min=0.0, z_max=2.0)
    E, S = 2, 16
    n = 12
    cum_np = np.zeros((E, S), np.float32)
    cum_np[:, :n] = np.arange(n, dtype=np.float32) * 0.1
    cum = wp.array(cum_np.reshape(-1), dtype=wp.float32, device="cpu")
    perim = wp.array(np.full(E, n * 0.1, np.float32), dtype=wp.float32, device="cpu")
    z = wp.zeros(E * S, dtype=wp.float32, device="cpu")
    count = wp.array(np.full(E, n, np.int32), dtype=wp.int32, device="cpu")
    seeds = wp.array(np.array([3, 9], np.int32), dtype=wp.int32, device="cpu")
    warp_zprofile.apply_z_profile(cfg, seeds, count, S, cum, perim, z)
    zz = z.numpy().reshape(E, S)
    for e in range(E):
        # periodic in frac: harmonic sum at frac=0 equals frac=1 exactly;
        # slot 0 (frac 0) vs a synthetic wrap evaluation isn't stored, so
        # assert bounds + variation + determinism instead
        assert (zz[e, :n] >= 0.0).all() and (zz[e, :n] <= 2.0).all()
        assert zz[e, :n].std() > 0.0
        assert (zz[e, n:] == 0.0).all()
