import warp as wp
from track_gen._src.rng_utils import PerEnvSeededRNG


def _rng(num_envs=4, seed=0):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device="cpu")


def test_uniform_3d_block_has_no_index_collisions():
    # A (3,5) block has 15 distinct (j,k) cells. With the correct row-major stride
    # (j*shape[2]+k) each cell seeds a distinct PCG state -> 15 distinct floats per env.
    # The stride bug (j*shape[1]+k) collides cells (e.g. (1,0) and (0,3) both map to 3),
    # producing duplicate values within a single draw.
    rng = _rng(num_envs=2, seed=7)
    out = wp.to_torch(rng.sample_uniform_warp(0.0, 1.0, (3, 5)))  # (2,3,5)
    for e in range(out.shape[0]):
        flat = out[e].reshape(-1)
        assert flat.unique().numel() == flat.numel(), "duplicate RNG values within a 3D draw"


def test_partial_ids_sample_preserves_untouched_env_states():
    import numpy as np
    rng = _rng(num_envs=4, seed=11)
    # numpy, not wp.to_torch: torch cannot wrap uint32 arrays (warp >= 1.14 surfaces
    # the underlying dtype instead of silently converting on the cpu path).
    before = rng.states_warp.numpy().copy()
    ids = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    rng.sample_uniform_warp(0.0, 1.0, (1,), ids=ids)  # touch only env 0
    after = rng.states_warp.numpy()
    # Envs 1..3 were not sampled; their states must be unchanged (not zeroed).
    assert (after[1:] == before[1:]).all(), "partial-ids sample corrupted untouched env states"
    assert (after[1:] != 0).any(), "untouched env states were zeroed"


def test_uniform_and_normal_accept_python_int_bounds():
    rng = _rng(num_envs=3, seed=3)
    u = wp.to_torch(rng.sample_uniform_warp(0, 1, (2,)))  # int bounds, must not raise
    assert u.shape == (3, 2)
    assert (u >= 0.0).all() and (u < 1.0).all()
    n = wp.to_torch(rng.sample_normal_warp(0, 1, (2,)))  # int mean/std, must not raise
    assert n.shape == (3, 2)


def test_quaternion_draws_are_not_all_identical_within_a_block():
    # Axis and angle previously shared a seed (angle a deterministic function of the axis).
    # After decorrelation, a (6,) block of quats for one env must not be all-identical.
    rng = _rng(num_envs=2, seed=5)
    q = wp.to_torch(rng.sample_quaternion_warp((6,)))  # (2,6,4)
    for e in range(q.shape[0]):
        block = q[e]
        first = block[0]
        assert not bool((block == first).all()), "quaternion block is degenerate/identical"
