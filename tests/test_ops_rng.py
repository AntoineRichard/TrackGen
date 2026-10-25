"""Hashed RNG initialization and opt-in sampler keying (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import ops
from track_gen.exec import LaunchContext
from track_gen.ops import sampling
from track_gen.testing.contract import hot_path_guard


DEV = "cpu"
_SEEDS = [0, 1, 7, -5, 2147483647, -2147483648, 99991, 123456789]
# seed, salt, index, key, one-argument state, two-argument state. These cover
# signed endpoints, high-bit salts, and index zero's distinct two-argument path.
_VECTORS = [
    (0, 0, 0, 0, 129708002, 817759070),
    (0, 0xFFFFFFFF, 0, 32599865, 2092523766, 1141823924),
    (1, 1, 1, 0, 129708002, 3911864714),
    (-5, 0xB739AFFB, -1, 1680781984, 3522466341, 3151677779),
    (-2147483648, 0x55E558EB, 2147483647, 949483309, 1957872625, 1248791822),
    (2147483647, 0x80000000, -2147483648, 32599865, 2092523766, 3211308703),
]


def _rng():
    assert hasattr(ops, "rng"), "track_gen.ops must export the rng module"
    return ops.rng


def _i32(values):
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=DEV)


def _empty(n, dtype):
    return wp.empty(n, dtype=dtype, device=DEV)


def _same(a, b):
    np.testing.assert_array_equal(a.numpy().view(np.uint8), b.numpy().view(np.uint8))


def _linearized_seeds(salt, index):
    # Warp native/rand.h: rand_init(key, i) = rand_pcg(key + rand_pcg(i)).
    # Feeding that sum to the existing linear sampler with salt=1, offset=0
    # isolates the new sampler dispatch/keying from its unchanged numeric body.
    mask = 0xFFFFFFFF
    b = ((index & mask) * 747796405 + 2891336453) & mask
    c = (((b >> ((b >> 28) + 4)) ^ b) * 277803737) & mask
    offset_state = (c >> 22) ^ c
    words = [(ops.hash32_host((seed & mask) ^ salt) & 0x7FFFFFFF) + offset_state
             for seed in _SEEDS]
    return wp.array(np.asarray([word & mask for word in words], np.uint32).view(np.int32),
                    dtype=wp.int32, device=DEV)


@pytest.mark.parametrize("seed,salt,index,key,state,state_at", _VECTORS)
def test_rng_host_frozen_vectors(seed, salt, index, key, state, state_at):
    rng = _rng()
    assert rng.key_host(seed, salt) == key
    assert rng.init_host(seed, salt) == state
    assert rng.init_at_host(seed, salt, index) == state_at


def test_rng_device_matches_warp_formula_and_host_twins():
    rng = _rng()
    assert isinstance(rng.init, wp.Function)
    assert isinstance(rng.init_at, wp.Function)

    @wp.kernel
    def init_k(seeds: wp.array(dtype=wp.int32), salts: wp.array(dtype=wp.uint32),
               indices: wp.array(dtype=wp.int32), actual: wp.array(dtype=wp.uint32),
               expected: wp.array(dtype=wp.uint32)):
        e = wp.tid()
        key = wp.int32(ops.hash32(wp.uint32(seeds[e]) ^ salts[e]) & wp.uint32(0x7FFFFFFF))
        actual[2 * e] = rng.init(seeds[e], salts[e])
        actual[2 * e + 1] = rng.init_at(seeds[e], salts[e], indices[e])
        expected[2 * e] = wp.rand_init(key)
        expected[2 * e + 1] = wp.rand_init(key, indices[e])

    seeds = _i32([v[0] for v in _VECTORS])
    salts = wp.array([v[1] for v in _VECTORS], dtype=wp.uint32, device=DEV)
    indices = _i32([v[2] for v in _VECTORS])
    actual = _empty(2 * len(_VECTORS), wp.uint32)
    expected = _empty(2 * len(_VECTORS), wp.uint32)
    ctx = LaunchContext(DEV, None)
    ctx.launch(init_k, len(_VECTORS), [seeds, salts, indices, actual, expected])
    _same(actual, expected)
    np.testing.assert_array_equal(actual.numpy(), np.asarray([x for v in _VECTORS
                                                            for x in v[4:]], np.uint32))


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("offset", [0, -5])
def test_hashed_count_variants_keep_linear_numeric_body(dynamic, offset):
    E = len(_SEEDS)
    ctx = LaunchContext(DEV, None)
    seeds, reference_seeds = _i32(_SEEDS), _linearized_seeds(0xB739AFFB, offset)
    lo = _i32([0, 2, 3, 4, 5, 6, 12, -1]) if dynamic else 3
    actual, expected = _empty(E, wp.int32), _empty(E, wp.int32)
    sampling._sample_counts_launch(ctx, reference_seeds, 1, 0, lo, 11, expected)
    sampling._sample_counts_launch(ctx, seeds, 0xB739AFFB, offset, lo, 11, actual,
                                   keying="hashed")
    _same(actual, expected)
    with hot_path_guard():
        sampling._sample_counts_launch(ctx, seeds, 0xB739AFFB, offset, lo, 11, actual,
                                       keying="hashed")


@pytest.mark.parametrize("per_env", [False, True])
@pytest.mark.parametrize("sized", [False, True])
@pytest.mark.parametrize("offset", [0, -2])
def test_hashed_grid_variants_keep_linear_numeric_body(per_env, sized, offset):
    E, P = len(_SEEDS), 12
    ctx = LaunchContext(DEV, None)
    seeds, reference_seeds = _i32(_SEEDS), _linearized_seeds(0xC2C74E45, offset)
    scale = (wp.array([0.5, 1.0, 1.5, 2.0] * 2, dtype=wp.float32, device=DEV)
             if per_env else 1.25)
    size = (wp.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.nan, -1.0],
                     dtype=wp.float32, device=DEV) if sized else None)
    actual, expected = _empty(E * P, wp.vec2f), _empty(E * P, wp.vec2f)
    used, reference_used = _empty(E * P, wp.int32), _empty(E * P, wp.int32)
    sampling._sample_grid_points_launch(ctx, reference_seeds, 1, 0, P, 3, 0.2, scale,
                                       reference_used, expected, size=size)
    sampling._sample_grid_points_launch(ctx, seeds, 0xC2C74E45, offset, P, 3, 0.2, scale,
                                       used, actual, size=size, keying="hashed")
    _same(actual, expected)
    _same(used, reference_used)
    with hot_path_guard():
        sampling._sample_grid_points_launch(ctx, seeds, 0xC2C74E45, offset, P, 3, 0.2,
                                           scale, used, actual, size=size, keying="hashed")


@pytest.mark.parametrize("salt", [0, 0x80000000, 0xFFFFFFFF, np.uint32(0xC2C74E45)])
def test_public_hashed_samplers_accept_uint32_salts(salt):
    E, P = len(_SEEDS), 5
    ctx = LaunchContext(DEV, None)
    seeds = _i32(_SEEDS)
    actual_count, expected_count = _empty(E, wp.int32), _empty(E, wp.int32)
    actual_grid, expected_grid = _empty(E * P, wp.vec2f), _empty(E * P, wp.vec2f)
    used, reference_used = _empty(E * P, wp.int32), _empty(E * P, wp.int32)
    reference_seeds = _linearized_seeds(int(salt), 0)
    ops.sample_counts(ctx, reference_seeds, 1, 0, 3, 11, expected_count)
    ops.sample_counts(ctx, seeds, salt, 0, 3, 11, actual_count, keying="hashed")
    ops.sample_grid_points(ctx, reference_seeds, 1, 0, P, 3, 0.2, 1.0,
                           reference_used, expected_grid)
    ops.sample_grid_points(ctx, seeds, salt, 0, P, 3, 0.2, 1.0, used, actual_grid,
                           keying="hashed")
    _same(actual_count, expected_count)
    _same(actual_grid, expected_grid)
    _same(used, reference_used)
    with hot_path_guard():
        ops.sample_counts(ctx, seeds, salt, 0, 3, 11, actual_count, keying="hashed")
        ops.sample_grid_points(ctx, seeds, salt, 0, P, 3, 0.2, 1.0, used, actual_grid,
                               keying="hashed")


@pytest.mark.parametrize("op", ["sample_counts", "sample_grid_points"])
@pytest.mark.parametrize("over,match", [
    ({"keying": "unknown"}, "keying"),
    ({"keying": None}, "keying"),
    ({"keying": "hashed", "salt": -1}, "salt"),
    ({"keying": "hashed", "salt": 2 ** 32}, "salt"),
    ({"keying": "hashed", "salt": 1.5}, "salt"),
    ({"keying": "hashed", "salt": True}, "salt"),
    ({"keying": "hashed", "offset": 2 ** 31}, "offset"),
    ({"keying": "linear", "salt": 2 ** 31}, "salt"),
])
def test_sampler_keying_validation_precedes_launch(op, over, match):
    class NoLaunch:
        device = DEV

        def launch(self, *args, **kwargs):
            pytest.fail("invalid arguments launched a kernel")

    args = dict(seeds=_i32([0]), salt=1, offset=0)
    if op == "sample_counts":
        args.update(lo=2, hi=5, out=_empty(1, wp.int32))
    else:
        args.update(num=3, num_cells=3, cell_size=0.2, scale=1.0,
                    used=_empty(3, wp.int32), out=_empty(3, wp.vec2f))
    args.update(over)
    with pytest.raises(ValueError, match=match):
        getattr(ops, op)(NoLaunch(), **args)


def test_hashed_grid_separates_seed_zero_streams():
    ctx = LaunchContext(DEV, None)
    seeds = _i32([0])
    out_a, out_b = _empty(8, wp.vec2f), _empty(8, wp.vec2f)
    used = _empty(8, wp.int32)
    ops.sample_grid_points(ctx, seeds, 0xC2C74E45, 0, 8, 4, 0.25, 1.0, used, out_a,
                           keying="hashed")
    ops.sample_grid_points(ctx, seeds, 0xC60040A4, 0, 8, 4, 0.25, 1.0, used, out_b,
                           keying="hashed")
    assert not np.array_equal(out_a.numpy(), out_b.numpy())
