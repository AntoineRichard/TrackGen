"""Device-side reseed, masked regeneration and in-graph auto-reset: CPU checks (PR4).

What runs here without a GPU:

- epoch 0 is byte-identical to the PR3 code path (the rng's seeds fed straight to the
  pipeline) for every registered track and gate generator;
- ``generate(advance=True, mask=m)``: masked envs change, unmasked envs are bitwise
  identical, ``generation`` equals the epoch, and ``(seed, epoch)`` reproduces an env;
- ``SeedState`` (bound seeds are never written, ``set`` on bound raises, ``reset_epoch``),
  the effective-seed kernel vs its host twin, ``ops.hash32`` and ``ops.any_set``;
- the rng fixes: ``set_seeds_warp`` mirrors the sampling double-buffer, and every
  ``sample_*_warp(out=...)`` equals the allocating call and allocates nothing;
- ``Course.regenerate(mask=...)``: masked envs get new courses and a runtime reset,
  unmasked envs keep everything, an all-zero mask is an exact no-op;
- policy on the fake CUDA device (``tests/_fake_cuda.py``): one self-graph per
  ``(advance, mask)`` key, ``LaunchContext.skip_unless`` lowering (conditional node only
  inside a Warp-registered capture), ``_compat.registered_stream`` / ``conditional_ok``
  and ``track_gen.interop.torch``.

The GPU side (caller graphs replayed N times, conditional skips, torch capture, the RL
loop) lives in ``tests/test_inplace_regen_cuda.py``.
"""
from __future__ import annotations

import dataclasses
import gc
import subprocess
import sys
import types
import weakref

import numpy as np
import pytest
import warp as wp

from tests._fake_cuda import FakeCuda, FakeGraph, FakeSeedState, FakeStream, patch_facade_seeds
from tests.test_caller_capture_cpu import _FACADES, _fake_rng
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import gate_generator_registry, generator_registry, warp_gate, warp_pipeline
from track_gen.course import Course, CourseConfig
from track_gen.exec import CaptureError, GraphRunner, LaunchContext, _compat
from track_gen.exec import runner as runner_mod
from track_gen.interop import torch as tg_torch
from track_gen.ops import any_set, hash32, hash32_host
from track_gen.seeds import SeedState, effective_seed

E = 4
DEV = "cpu"

_TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                 "valid", "count", "winding", "truncated")
_GATE_FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left",
                "right", "valid", "count")


@pytest.fixture(autouse=True)
def _fresh_legacy_registry(monkeypatch):
    monkeypatch.setattr(runner_mod, "_PENDING_LEGACY", weakref.WeakSet())


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def _i32(values) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=DEV)


def _rng(seed=3, n=E):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _track_kw(g: str) -> dict:
    # repulsive at its default size is slow on CPU; a small schedule exercises the same code
    return dict(num_points=64, repulsive_stages=(16, 32, 64)) if g == "repulsive" else {}


def _track_cfg(g="bezier", n=E):
    return TrackGenConfig(generator=g, num_envs=n, device=DEV, **_track_kw(g))


def _gate_cfg(g="bezier", n=E, **kw):
    return GateGenConfig(generator=g, num_envs=n, device=DEV, gate_width=0.2, **kw)


def _make(which: str, g: str, rng):
    if which == "track":
        gen = TrackGenerator(_track_cfg(g), rng)
        return gen, gen._track, _TRACK_FIELDS
    gen = GateGenerator(_gate_cfg(g), rng)
    return gen, gen._gate_sequence, _GATE_FIELDS


def _env_bytes(arr: wp.array, n=E) -> list[bytes]:
    """Per-env byte slices of an env-major flat array (NaN-safe comparisons)."""
    a = np.ascontiguousarray(arr.numpy()).reshape(n, -1)
    return [a[e].tobytes() for e in range(n)]


def _all_bytes(obj, fields) -> dict[str, bytes]:
    return {f: getattr(obj, f).numpy().tobytes() for f in fields}


def _assert_bytes_equal(a, b, fields, label=""):
    for f in fields:
        assert getattr(a, f).numpy().tobytes() == getattr(b, f).numpy().tobytes(), \
            f"{label}: field {f} differs"


GENERATORS = ([("track", g) for g in generator_registry.available()]
              + [("gates", g) for g in gate_generator_registry.available()])
GEN_IDS = [f"{w}-{g}" for w, g in GENERATORS]


# ---------------------------------------------------------------------------------------
# hashing and the effective seed
# ---------------------------------------------------------------------------------------

@wp.kernel
def _hash_k(x: wp.array(dtype=wp.uint32), out: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    out[i] = hash32(x[i])


def test_hash32_is_murmur3_fmix32_on_device_and_host():
    assert hash32_host(0) == 0
    assert hash32_host(1) == 0x514E28B7               # MurmurHash3 fmix32(1)
    xs = np.array([0, 1, 2, 3, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x9E3779B9, 123456789],
                  dtype=np.uint32)
    out = wp.empty(len(xs), dtype=wp.uint32, device=DEV)
    wp.launch(_hash_k, len(xs), [wp.array(xs, dtype=wp.uint32, device=DEV), out], device=DEV)
    assert out.numpy().tolist() == [hash32_host(int(x)) for x in xs]
    assert len(set(out.numpy().tolist())) == len(xs)  # a bijection: no collisions here


def test_effective_seed_kernel_matches_its_host_twin():
    seeds = np.array([0, 1, -1, 2**31 - 1, -(2**31), 7, 123456, -98765], np.int32)
    for epochs in ([0] * 8, [1] * 8, list(range(1, 9)),
                   [2**31 - 1, -1, 100000, 3, 9, 0, 1, -5]):
        st = SeedState(8, DEV, wp.array(seeds, dtype=wp.int32, device=DEV))
        st.epoch.assign(np.asarray(epochs, np.int32))
        out = wp.empty(8, dtype=wp.int32, device=DEV)
        st.effective(out)
        expect = [effective_seed(int(s), ep) for s, ep in zip(seeds, epochs)]
        assert out.numpy().tolist() == expect, epochs
    # epoch 0 is the identity (byte-identical keying); other epochs give distinct,
    # non-negative int32 seeds
    assert [effective_seed(s, 0) for s in (0, -5, 2**31 - 1)] == [0, -5, 2**31 - 1]
    hashed = {effective_seed(5, ep) for ep in range(1, 200)}
    assert len(hashed) == 199 and all(0 <= h < 2**31 for h in hashed)
    assert effective_seed(5, 1) != effective_seed(6, 1)


# ---------------------------------------------------------------------------------------
# SeedState
# ---------------------------------------------------------------------------------------

def test_seed_state_owned_expansions_match_the_rng():
    for k in (0, 41, 2**31 - 2, -7):
        st = SeedState(E, DEV, k)
        assert not st.bound
        np.testing.assert_array_equal(st.seeds.numpy(), _rng(k).seeds_warp.numpy())
        np.testing.assert_array_equal(st.epoch.numpy(), np.zeros(E, np.int32))
    np.testing.assert_array_equal(SeedState(E, DEV).seeds.numpy(), np.arange(E))
    st = SeedState(E, "cpu", 3)
    assert st.num_envs == E and st.device == "cpu" and "owned" in repr(st)


def test_seed_state_binds_caller_arrays_and_rngs_by_pointer():
    arr = _i32([5, 6, 7, 8])
    st = SeedState(E, DEV, arr)
    assert st.bound and st.seeds is arr and "bound" in repr(st)
    rng = _rng(9)
    st2 = SeedState(E, DEV, rng)
    assert st2.bound and st2.seeds is rng.seeds_warp
    rng.set_seeds_warp(_i32([1, 2, 3, 4]), None)          # reseeding the rng is seen
    np.testing.assert_array_equal(st2.seeds.numpy(), [1, 2, 3, 4])


def test_bound_seeds_are_never_written():
    arr = _i32([5, -6, 2**31 - 1, 8])
    before = arr.numpy().copy()
    st = SeedState(E, DEV, arr)
    out = wp.empty(E, dtype=wp.int32, device=DEV)
    m = _i32([1, 0, 1, 0])
    for _ in range(3):
        st.advance()
        st.advance(m)
        st.effective(out)
    st.reset_epoch(m)
    st.reset_epoch()
    with pytest.raises(RuntimeError, match="bound"):
        st.set(_i32([0, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="bound"):
        st.set(_i32([0]), _i32([1]))
    np.testing.assert_array_equal(arr.numpy(), before)
    # a generator built on the rng never writes the rng's seeds either
    rng = _rng(4)
    seeds_before = rng.seeds_warp.numpy().copy()
    gen = TrackGenerator(_track_cfg(), rng)
    gen.generate(advance=True, mask=m)
    gen.enqueue(advance=True)
    np.testing.assert_array_equal(rng.seeds_warp.numpy(), seeds_before)


def test_seed_state_copy_owns_its_seeds():
    arr = _i32([5, 6, 7, 8])
    st = SeedState(E, DEV, arr, copy=True)
    assert not st.bound and st.seeds is not arr
    np.testing.assert_array_equal(st.seeds.numpy(), arr.numpy())
    st.set(_i32([1, 1, 1, 1]))
    np.testing.assert_array_equal(arr.numpy(), [5, 6, 7, 8])
    rng = _rng(2)
    st2 = SeedState(E, DEV, rng, copy=True)
    assert not st2.bound and st2.seeds is not rng.seeds_warp


def test_seed_state_advance_reset_and_set():
    st = SeedState(E, DEV, 10)
    st.advance()
    st.advance(_i32([1, 0, 1, 0]))
    st.advance(_i32([0, 0, 5, 0]))                       # any non-zero value selects
    np.testing.assert_array_equal(st.epoch.numpy(), [2, 1, 3, 1])
    st.reset_epoch(_i32([0, 1, 0, 0]))
    np.testing.assert_array_equal(st.epoch.numpy(), [2, 0, 3, 1])
    st.set(_i32([100, 101]), _i32([3, 0]))               # scatter + zero those epochs
    np.testing.assert_array_equal(st.seeds.numpy(), [101, 11, 12, 100])
    np.testing.assert_array_equal(st.epoch.numpy(), [0, 0, 3, 0])
    st.set(_i32([7, 8, 9, 10]))
    np.testing.assert_array_equal(st.seeds.numpy(), [7, 8, 9, 10])
    np.testing.assert_array_equal(st.epoch.numpy(), [0, 0, 0, 0])
    st.advance()
    st.reset_epoch()
    np.testing.assert_array_equal(st.epoch.numpy(), [0, 0, 0, 0])
    st.set(wp.empty(0, dtype=wp.int32, device=DEV), wp.empty(0, dtype=wp.int32, device=DEV))


def test_seed_state_validation():
    with pytest.raises(ValueError, match="num_envs"):
        SeedState(0, DEV)
    with pytest.raises(TypeError, match="seeds"):
        SeedState(E, DEV, 1.5)
    with pytest.raises(TypeError, match="seeds"):
        SeedState(E, DEV, True)
    with pytest.raises(ValueError, match="shape"):
        SeedState(E, DEV, _i32([1, 2, 3]))
    with pytest.raises(ValueError, match="dtype"):
        SeedState(E, DEV, wp.zeros(E, dtype=wp.float32, device=DEV))
    with pytest.raises(ValueError, match="PerEnvSeededRNG with num_envs=4"):
        SeedState(E, DEV, _rng(3, n=E + 1))
    with pytest.raises(TypeError, match="default_stream"):
        SeedState(E, DEV, default_stream="torch")
    st = SeedState(E, DEV)
    strided = wp.zeros(2 * E, dtype=wp.int32, device=DEV)[::2]   # [E], same ptr as [:E]
    for bad in (_i32([1, 0]), wp.zeros(E, dtype=wp.float32, device=DEV),
                wp.zeros((E, 1), dtype=wp.int32, device=DEV), np.zeros(E, np.int32),
                strided):
        with pytest.raises(ValueError, match="mask"):
            st.advance(bad)
        with pytest.raises(ValueError, match="mask"):
            st.reset_epoch(bad)
        with pytest.raises(ValueError, match="mask"):
            st.effective(wp.empty(E, dtype=wp.int32, device=DEV), bad)
    with pytest.raises(ValueError, match="out"):
        st.effective(_i32([0, 0]))
    with pytest.raises(ValueError, match="seeds"):
        st.set(_i32([1, 2]), _i32([0]))               # one seed per id
    with pytest.raises(ValueError, match="ids"):
        st.set(_i32([1]), wp.zeros(1, dtype=wp.float32, device=DEV))
    with pytest.raises(TypeError, match="ctx"):
        st.advance(stream="torch", ctx=LaunchContext.for_call(DEV))
    np.testing.assert_array_equal(st.epoch.numpy(), np.zeros(E, np.int32))


# ---------------------------------------------------------------------------------------
# ops.any_set
# ---------------------------------------------------------------------------------------

def test_any_set():
    ctx = LaunchContext.for_call(DEV)
    flag = wp.full(1, 7, dtype=wp.int32, device=DEV)     # stale value: any_set zeroes first
    any_set(ctx, _i32([0, 0, 0, 0]), flag)
    assert flag.numpy().tolist() == [0]
    any_set(ctx, _i32([0, 0, 3, 0]), flag)
    assert flag.numpy().tolist() == [1]
    any_set(ctx, _i32([-1, 1, 1, 1]), flag)
    assert flag.numpy().tolist() == [1]
    any_set(ctx, wp.empty(0, dtype=wp.int32, device=DEV), flag)
    assert flag.numpy().tolist() == [0]
    with pytest.raises(TypeError):
        any_set(ctx, wp.zeros(4, dtype=wp.float32, device=DEV), flag)
    with pytest.raises(ValueError):
        any_set(ctx, _i32([1, 0]), wp.zeros(2, dtype=wp.int32, device=DEV))
    with pytest.raises(TypeError):
        any_set(ctx, np.zeros(4, np.int32), flag)


# ---------------------------------------------------------------------------------------
# Facades: epoch 0 is the PR3 code path, byte for byte
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("which,g", GENERATORS, ids=GEN_IDS)
def test_epoch0_is_byte_identical_to_the_pr3_seed_copy_path(which, g):
    """PR3's body copied the rng's seeds into the pipeline's seed buffer; PR4 writes the
    effective seeds, which equal the seeds at epoch 0. Run the PR3 body (the rng's seeds
    fed straight to the pipeline) into separate buffers and compare every field."""
    for seed in (5, 1234):
        rng = _rng(seed)
        gen, out, fields = _make(which, g, rng)
        gen.generate()
        ctx = LaunchContext.for_call(DEV)
        if which == "track":
            ref, scratch = warp_pipeline._inflate_warp_alloc(
                gen._config, generator_spec=gen._generator_spec)
            warp_pipeline._run_pipeline(gen._config, rng.seeds_warp, out=ref, scratch=scratch,
                                        generator_spec=gen._generator_spec, ctx=ctx)
        else:
            ref, scratch = warp_gate._gate_warp_alloc(gen._config,
                                                      generator_spec=gen._generator_spec)
            warp_gate._run_gate_pipeline(gen._config, rng.seeds_warp, out=ref,
                                         scratch=scratch,
                                         generator_spec=gen._generator_spec, ctx=ctx)
        _assert_bytes_equal(out, ref, fields, f"{which}-{g} seed {seed}")
        np.testing.assert_array_equal(out.generation.numpy(), np.zeros(E, np.int32))
        np.testing.assert_array_equal(gen._seed_buf.numpy(), rng.seeds_warp.numpy())


# ---------------------------------------------------------------------------------------
# Facades: advance / mask / generation / (seed, epoch) reproduction
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("which,g", GENERATORS, ids=GEN_IDS)
def test_masked_advance_changes_only_masked_envs(which, g):
    gen, out, fields = _make(which, g, _rng(7))
    before = {f: _env_bytes(getattr(out, f)) for f in fields}
    geo = "center" if which == "track" else "position"
    m = _i32([1, 0, 1, 0])
    assert gen.generate(advance=True, mask=m) is out
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), [1, 0, 1, 0])
    np.testing.assert_array_equal(out.generation.numpy(), [1, 0, 1, 0])
    after = {f: _env_bytes(getattr(out, f)) for f in fields}
    for e in range(E):
        if m.numpy()[e]:
            assert after[geo][e] != before[geo][e], f"masked env {e} kept its {geo}"
        else:
            for f in fields:
                assert after[f][e] == before[f][e], f"unmasked env {e}: {f} changed"
    # the complement, through enqueue(): every env now at epoch 1
    gen.enqueue(advance=True, mask=_i32([0, 1, 0, 1]))
    np.testing.assert_array_equal(out.generation.numpy(), [1, 1, 1, 1])
    final = {f: _env_bytes(getattr(out, f)) for f in fields}
    for e in (0, 2):
        for f in fields:
            assert final[f][e] == after[f][e], f"env {e} changed on the complement advance"
    # a plain generate() recomputes the same batch (no advance, epochs kept)
    gen.generate()
    assert {f: _env_bytes(getattr(out, f)) for f in fields} == final
    np.testing.assert_array_equal(out.generation.numpy(), [1, 1, 1, 1])


@pytest.mark.parametrize("which,g", GENERATORS, ids=GEN_IDS)
def test_seed_epoch_pair_reproduces_an_env(which, g):
    """(seed, epoch) fully determines an env: two generators driven to the same epochs
    along different advance histories agree bitwise, and both equal a fresh generator
    whose rng seeds ARE the effective seeds (at epoch 0)."""
    a, out_a, fields = _make(which, g, _rng(21))
    for m in ([1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]):  # epochs -> [3, 1, 0, 2]
        a.generate(advance=True, mask=_i32(m))
    b, out_b, _ = _make(which, g, _rng(21))
    b.enqueue(advance=True)                                # [1, 1, 1, 1]
    b.enqueue(advance=True, mask=_i32([1, 0, 0, 1]))       # [2, 1, 1, 2]
    env2 = {f: _env_bytes(getattr(out_b, f))[2] for f in fields}
    b.seeds.reset_epoch(_i32([0, 0, 1, 0]))                # [2, 1, 0, 2]
    b.generate(advance=True, mask=_i32([1, 0, 0, 0]))      # [3, 1, 0, 2]
    # env 2's epoch reset happened outside a generation: a masked one that does not
    # select env 2 keeps the effective seed its track was built from ...
    assert {f: _env_bytes(getattr(out_b, f))[2] for f in fields} == env2
    np.testing.assert_array_equal(out_b.generation.numpy(), [3, 1, 1, 2])
    b.generate()                                           # ... a full one applies it
    np.testing.assert_array_equal(a.seeds.epoch.numpy(), [3, 1, 0, 2])
    np.testing.assert_array_equal(b.seeds.epoch.numpy(), [3, 1, 0, 2])
    _assert_bytes_equal(out_a, out_b, fields + ("generation",), "two histories")
    eff = [effective_seed(21 + e, ep) for e, ep in enumerate([3, 1, 0, 2])]
    c, out_c, _ = _make(which, g, PerEnvSeededRNG(seeds=_i32(eff), num_envs=E, device=DEV))
    _assert_bytes_equal(out_a, out_c, fields, "effective-seed twin")
    np.testing.assert_array_equal(out_c.generation.numpy(), np.zeros(E, np.int32))


@pytest.mark.parametrize("which", ["track", "gates"])
def test_facade_inputs_and_mask_validation(which):
    gen, out, fields = _make(which, "bezier", _rng(3))
    inputs = gen.inputs
    m = inputs["regen_mask"]
    assert gen.inputs["regen_mask"] is m and m.dtype is wp.int32 and m.shape == (E,)
    np.testing.assert_array_equal(m.numpy(), np.zeros(E, np.int32))
    with pytest.raises(TypeError):
        inputs["regen_mask"] = m                          # read-only mapping
    m.assign(np.array([0, 1, 0, 0], np.int32))
    gen.generate(advance=True, mask=m)
    np.testing.assert_array_equal(out.generation.numpy(), [0, 1, 0, 0])
    snap = _all_bytes(out, fields + ("generation",))
    launches: list = []
    real = wp.launch
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "launch", lambda *a, **k: launches.append(a) or real(*a, **k))
        for bad in (_i32([1, 0]), wp.zeros(E, dtype=wp.float32, device=DEV),
                    wp.zeros((E, 1), dtype=wp.int32, device=DEV),
                    wp.zeros(2 * E, dtype=wp.int32, device=DEV)[::2]):
            with pytest.raises(ValueError, match="mask"):
                gen.generate(advance=True, mask=bad)
            with pytest.raises(ValueError, match="mask"):
                gen.enqueue(advance=True, mask=bad)
        with pytest.raises(ValueError, match="advance=True"):
            gen.generate(mask=m)
        with pytest.raises(ValueError, match="advance=True"):
            gen.enqueue(mask=m)
    assert launches == []                                 # refused before any launch
    assert _all_bytes(out, fields + ("generation",)) == snap


def test_generation_field_is_optional_and_cloned():
    gen = TrackGenerator(_track_cfg(), _rng(3))
    gen.generate(advance=True, mask=_i32([0, 0, 1, 0]))
    snap = gen._track.clone()
    assert snap.generation is not gen._track.generation
    np.testing.assert_array_equal(snap.generation.numpy(), [0, 0, 1, 0])
    hand_built = dataclasses.replace(gen._track, generation=None)
    assert hand_built.clone().generation is None
    gseq = GateGenerator(_gate_cfg(), _rng(3))._gate_sequence
    assert gseq.clone().generation is not None
    assert dataclasses.replace(gseq, generation=None).clone().generation is None


def test_advance_paths_do_not_allocate(monkeypatch):
    gens = [TrackGenerator(_track_cfg(), _rng(1)), GateGenerator(_gate_cfg(), _rng(1))]
    m = _i32([1, 0, 0, 1])
    _forbid_allocation(monkeypatch)
    for gen in gens:
        gen.generate(advance=True)
        gen.generate(advance=True, mask=m)
        gen.enqueue(advance=True, mask=gen.inputs["regen_mask"])
        gen.seeds.advance(m)
        gen.seeds.reset_epoch(m)


# ---------------------------------------------------------------------------------------
# PerEnvSeededRNG: the set_seeds_warp mirror fix and out= sampling
# ---------------------------------------------------------------------------------------

def _forbid_allocation(mp):
    def trip(name):
        def call(*a, **k):
            raise AssertionError(f"wp.{name} called on an allocation-free path")
        return call
    for name in ("empty", "zeros", "full", "ones", "clone", "empty_like", "zeros_like",
                 "full_like", "ones_like"):
        mp.setattr(wp, name, trip(name))


def test_reseed_survives_a_later_partial_ids_sample():
    rng = _rng(11)
    rng.sample_uniform_warp(0.0, 1.0, (3,))               # move the states on
    new = _i32([50, 60, 70, 80])
    rng.set_seeds_warp(new, None)
    reseeded = rng.states_warp.numpy().copy()
    np.testing.assert_array_equal(reseeded, PerEnvSeededRNG(
        seeds=_i32([50, 60, 70, 80]), num_envs=E, device=DEV).states_warp.numpy())
    rng.sample_uniform_warp(0.0, 1.0, (1,), ids=_i32([0]))  # touch env 0 only
    after = rng.states_warp.numpy()
    np.testing.assert_array_equal(after[1:], reseeded[1:],
                                  err_msg="a partial-ids sample reverted the reseed")
    # a partial reseed survives a sample of another env too
    rng.set_seeds_warp(_i32([99]), _i32([2]))
    env2 = rng.states_warp.numpy()[2]
    rng.sample_normal_warp(0.0, 1.0, (1,), ids=_i32([1]))
    assert rng.states_warp.numpy()[2] == env2
    np.testing.assert_array_equal(rng.seeds_warp.numpy(), [50, 60, 99, 80])


_SAMPLERS = ("uniform", "uniform_t", "sign_int", "sign_float", "integer", "integer_t",
             "normal", "normal_t", "poisson", "poisson_t", "quaternion")


def _sampler_call(name: str, shape: tuple, n: int):
    """``(method name, positional args)`` of one sampler case; the tensorized bounds
    (``*_t``) are ``[n]`` arrays, built here (outside any allocation tripwire)."""
    def f32(v):
        return wp.array(np.full(n, v, np.float32), dtype=wp.float32, device=DEV)

    def i32(v):
        return wp.array(np.full(n, v, np.int32), dtype=wp.int32, device=DEV)

    return {
        "uniform": ("sample_uniform_warp", (-1.0, 2.0, shape)),
        "uniform_t": ("sample_uniform_warp", (f32(-1.0), f32(2.0), shape)),
        "sign_int": ("sample_sign_warp", ("int", shape)),
        "sign_float": ("sample_sign_warp", ("float", shape)),
        "integer": ("sample_integer_warp", (0, 10, shape)),
        "integer_t": ("sample_integer_warp", (i32(0), i32(10), shape)),
        "normal": ("sample_normal_warp", (0.5, 2.0, shape)),
        "normal_t": ("sample_normal_warp", (f32(0.5), f32(2.0), shape)),
        "poisson": ("sample_poisson_warp", (3.0, shape)),
        "poisson_t": ("sample_poisson_warp", (f32(3.0), shape)),
        "quaternion": ("sample_quaternion_warp", (shape,)),
    }[name]


@pytest.mark.parametrize("ids", [None, [1, 3]], ids=["all", "partial"])
@pytest.mark.parametrize("shape", [(1,), (3,), (2, 3)], ids=str)
@pytest.mark.parametrize("name", _SAMPLERS)
def test_sample_out_equals_the_allocating_call_and_allocates_nothing(name, shape, ids):
    a, b = _rng(7), _rng(7)
    sel = None if ids is None else _i32(ids)
    meth, args = _sampler_call(name, shape, E if ids is None else len(ids))
    ref = getattr(a, meth)(*args, ids=sel)
    out = wp.empty(ref.shape, dtype=ref.dtype, device=DEV)
    with pytest.MonkeyPatch.context() as mp:
        _forbid_allocation(mp)
        got = getattr(b, meth)(*args, ids=sel, out=out)
    assert got is out
    assert out.numpy().tobytes() == ref.numpy().tobytes()
    np.testing.assert_array_equal(a.states_warp.numpy(), b.states_warp.numpy())
    # and again: the states moved on identically
    ref2 = getattr(a, meth)(*args, ids=sel)
    assert getattr(b, meth)(*args, ids=sel, out=out) is out
    assert out.numpy().tobytes() == ref2.numpy().tobytes()


@pytest.mark.parametrize("name", _SAMPLERS)
def test_sample_out_is_validated_before_any_launch(name):
    r = _rng(7)
    meth, args = _sampler_call(name, (3,), E)
    ref = getattr(_rng(7), meth)(*args)
    states = r.states_warp.numpy().copy()
    other = wp.float32 if ref.dtype is not wp.float32 else wp.int32
    for bad in (wp.empty((E, 2), dtype=ref.dtype, device=DEV),
                wp.empty(ref.shape, dtype=other, device=DEV),
                wp.empty(E * 3, dtype=ref.dtype, device=DEV),
                np.zeros(ref.shape)):
        with pytest.raises(ValueError, match="out"):
            getattr(r, meth)(*args, out=bad)
    np.testing.assert_array_equal(r.states_warp.numpy(), states)


# ---------------------------------------------------------------------------------------
# Course: masked regeneration on CPU
# ---------------------------------------------------------------------------------------

_COURSES = {
    "track-seg-hf": ("track", dict(collision="segments", heightfield_resolution=16,
                                   checkpoint_spacing=0.4)),
    "track-sdf": ("track", dict(collision="sdf", sdf_resolution=16, checkpoint_spacing=0.4)),
    "gates-posts": ("gates", dict(post_radius=0.02, localize_window=4)),
    "gates-frames": ("gates", dict(frame_collision=True, agent_radius=0.05,
                                   frame_thickness=0.02, frame_depth=0.02)),
}


def _course(variant: str, seeds=9) -> Course:
    mode, kw = _COURSES[variant]
    gen = TrackGenConfig(num_envs=E, device=DEV) if mode == "track" else _gate_cfg()
    c = Course(CourseConfig(mode=mode, gen=gen, seeds=seeds, **kw))
    pos = wp.array(np.zeros((E, 3), np.float32), dtype=wp.vec3f, device=DEV)
    if c._needs_boxes():
        ori = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (E, 1)).astype(np.float32),
                       dtype=wp.quatf, device=DEV)
        he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
        c.bind(position=pos, orientation=ori, half_extents=he)
    else:
        c.bind(position=pos)
    c.step()                                  # step results exist (and must stay put)
    return c


def _dc(obj, prefix: str) -> dict:
    return {prefix + f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)
            if isinstance(getattr(obj, f.name), wp.array)}


def _attrs(obj, prefix: str) -> dict:
    return {prefix + k: v for k, v in vars(obj).items() if isinstance(v, wp.array)}


def _course_buffers(c: Course) -> dict:
    """Every output and state buffer of the course (not scratch or internal flags)."""
    out = {"epoch": c.seeds.epoch, "seed_buf": c.generator._seed_buf}
    out.update(_dc(c.result, "result."))
    out.update(_dc(c.checkpoints, "cp."))
    for name in ("checkpoint_sampler", "collision", "localizer", "progress"):
        obj = getattr(c, name)
        if obj is not None:
            out.update(_attrs(obj, name + "."))
    out.update(_dc(c.progress._events, "events."))
    if c._step_result.contacts is not None:
        out.update(_dc(c._step_result.contacts, "contacts."))
    if c._step_result.frame is not None:
        out.update(_dc(c._step_result.frame, "frame."))
    if c.heightfield is not None:
        out.update(_dc(c.heightfield._hf, "hf."))
    if c.course_line is not None:
        out.update(_dc(c.course_line.track, "line."))
    if c._posts is not None:
        out["posts"] = c._posts
    return out


def _snap(c: Course) -> dict:
    return {k: v.numpy().copy() for k, v in _course_buffers(c).items()}


def _per_env(a: np.ndarray):
    return a.reshape(E, -1) if a.shape and a.shape[0] % E == 0 else None


def _poison_runtime(c: Course) -> None:
    """Stale per-env runtime state a regeneration must (or must not) reset."""
    p = c.progress
    p._next.assign(np.array([2, 1, 3, 1], np.int32))
    p._laps.assign(np.array([1, 2, 0, 1], np.int32))
    p._progress.assign(np.array([5, 6, 7, 8], np.int32))
    p._prev_pos.assign(np.full((E, 3), 0.25, np.float32))
    if c.localizer is not None:
        c.localizer._last.assign(np.array([3, 1, 2, 4], np.int32))


@pytest.mark.parametrize("variant", sorted(_COURSES))
def test_course_masked_regenerate(variant):
    c = _course(variant)
    _poison_runtime(c)
    before = _snap(c)
    m = _i32([1, 0, 0, 1])
    assert c.regenerate(mask=m, advance=True) is c.result
    after = _snap(c)
    geo = "result.center" if c._cfg.mode == "track" else "result.position"
    np.testing.assert_array_equal(after["epoch"], [1, 0, 0, 1])
    np.testing.assert_array_equal(after["result.generation"], [1, 0, 0, 1])
    for e in range(E):
        if e in (0, 3):
            assert after[geo].reshape(E, -1)[e].tobytes() != \
                before[geo].reshape(E, -1)[e].tobytes(), f"masked env {e}: same course"
            assert after["progress._next"][e] == 0 and after["progress._laps"][e] == 0
            assert after["progress._progress"][e] == 0
            assert np.isnan(after["progress._prev_pos"][e]).all()
            if c.localizer is not None:
                assert after["localizer._last"][e] == -1
            continue
        for k, a in after.items():
            pa, pb = _per_env(a), _per_env(before[k])
            if pa is not None:
                assert pa[e].tobytes() == pb[e].tobytes(), f"unmasked env {e}: {k} changed"
    # an eager twin driven identically is bitwise equal
    twin = _course(variant)
    _poison_runtime(twin)
    twin.enqueue_generate(mask=_i32([1, 0, 0, 1]), advance=True)
    twin_snap = _snap(twin)
    for k in after:
        assert after[k].tobytes() == twin_snap[k].tobytes(), f"twin differs: {k}"


@pytest.mark.parametrize("variant", sorted(_COURSES))
def test_course_zero_mask_is_an_exact_noop(variant):
    c = _course(variant)
    c.regenerate(mask=_i32([0, 1, 0, 0]), advance=True)   # epochs no longer all zero
    _poison_runtime(c)
    before = _snap(c)
    zero = _i32([0, 0, 0, 0])
    for kw in (dict(advance=True), dict(advance=False)):
        c.regenerate(mask=zero, **kw)
        c.enqueue_generate(mask=c.inputs["regen_mask"], **kw)
        after = _snap(c)
        for k in before:
            assert after[k].tobytes() == before[k].tobytes(), f"{kw}: {k} changed"


@pytest.mark.parametrize("variant", ["track-seg-hf", "gates-posts"])
def test_course_out_of_band_seed_changes_wait_for_the_env_mask(variant):
    """A reseed (rng.set_seeds_warp, a write to the seed array) or an epoch reset made
    OUTSIDE a regeneration must not leak into a masked regeneration that does not select
    those envs: an all-zero mask stays an exact no-op (so the always-run lowering agrees
    with the conditional skip), and a partial mask changes only the masked envs. The
    change reaches an env when it is masked, and every env at a full regenerate()."""
    c = _course(variant)
    c.regenerate(mask=_i32([1, 1, 0, 0]), advance=True)            # epochs [1, 1, 0, 0]
    _poison_runtime(c)
    before = _snap(c)
    c.seeds.reset_epoch()                                          # out of band
    c.rng.set_seeds_warp(_i32([1234]), _i32([2]))                  # out of band
    for kw in (dict(advance=True), dict(advance=False)):
        c.regenerate(mask=_i32([0, 0, 0, 0]), **kw)
        after = _snap(c)
        for k in before:
            if k != "epoch":
                assert after[k].tobytes() == before[k].tobytes(), f"{kw}: {k} changed"
    np.testing.assert_array_equal(c.seeds.epoch.numpy(), np.zeros(E, np.int32))
    np.testing.assert_array_equal(c.result.generation.numpy(), [1, 1, 0, 0])
    # a partial mask: only env 3 changes; envs 0-2 keep course, generation and state
    c.regenerate(mask=_i32([0, 0, 0, 1]), advance=True)
    after = _snap(c)
    np.testing.assert_array_equal(after["result.generation"], [1, 1, 0, 1])
    for k, a in after.items():
        pa, pb = _per_env(a), _per_env(before[k])
        if pa is not None and k != "epoch":
            for e in (0, 1, 2):
                assert pa[e].tobytes() == pb[e].tobytes(), f"env {e}: {k} changed"
    # a full regeneration applies every pending change: it equals a course built on the
    # same seeds and driven to the same epochs, then fully regenerated
    c.regenerate()
    twin = _course(variant)
    twin.rng.set_seeds_warp(_i32([1234]), _i32([2]))
    twin.regenerate(mask=_i32([0, 0, 0, 1]), advance=True)
    twin.regenerate()
    fields = _TRACK_FIELDS if c._cfg.mode == "track" else _GATE_FIELDS
    _assert_bytes_equal(c.result, twin.result, fields + ("generation",), "full regenerate")


@pytest.mark.parametrize("variant", ["track-seg-hf", "gates-posts"])
def test_course_mask_without_advance_respawns_on_the_same_courses(variant):
    c = _course(variant)
    _poison_runtime(c)
    before = _snap(c)
    c.regenerate(mask=_i32([0, 1, 0, 0]))
    after = _snap(c)
    geo = "result.center" if c._cfg.mode == "track" else "result.position"
    assert after[geo].tobytes() == before[geo].tobytes()
    np.testing.assert_array_equal(after["epoch"], np.zeros(E, np.int32))
    np.testing.assert_array_equal(after["progress._progress"], [5, 0, 7, 8])


@pytest.mark.parametrize("variant", ["track-sdf", "gates-frames"])
def test_course_whole_batch_advance_and_reseed(variant):
    c = _course(variant)
    _poison_runtime(c)
    c.regenerate(advance=True)                 # every env advances; FULL runtime reset
    np.testing.assert_array_equal(c.seeds.epoch.numpy(), np.ones(E, np.int32))
    np.testing.assert_array_equal(c.result.generation.numpy(), np.ones(E, np.int32))
    np.testing.assert_array_equal(c.progress._progress.numpy(), np.zeros(E, np.int32))
    twin = _course(variant)
    twin.regenerate(mask=_i32([1, 1, 1, 1]), advance=True)
    geo = "center" if c._cfg.mode == "track" else "position"
    assert getattr(c.result, geo).numpy().tobytes() == \
        getattr(twin.result, geo).numpy().tobytes()
    # generate(seeds=...) zeroes the epochs: it reproduces a fresh course with those seeds
    c.generate(seeds=77)
    np.testing.assert_array_equal(c.seeds.epoch.numpy(), np.zeros(E, np.int32))
    fresh = _course(variant, seeds=77)
    fields = _TRACK_FIELDS if c._cfg.mode == "track" else _GATE_FIELDS
    _assert_bytes_equal(c.result, fresh.result, fields + ("generation",), "reseed")
    c.regenerate(advance=True)
    c.generate(seeds=_i32([1, 2, 3, 4]))
    np.testing.assert_array_equal(c.seeds.epoch.numpy(), np.zeros(E, np.int32))
    _assert_bytes_equal(c.result, _course(variant, seeds=_i32([1, 2, 3, 4])).result,
                        fields, "array reseed")


def test_course_inputs_and_mask_validation():
    c = _course("track-seg-hf")
    m = c.inputs["regen_mask"]
    assert c.inputs["regen_mask"] is m and m.shape == (E,) and m.dtype is wp.int32
    np.testing.assert_array_equal(m.numpy(), np.zeros(E, np.int32))
    assert c.seeds is c.generator.seeds and c.seeds.seeds is c.rng.seeds_warp
    before = _snap(c)
    launches: list = []
    real = wp.launch
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "launch", lambda *a, **k: launches.append(a) or real(*a, **k))
        for bad in (_i32([1, 0]), wp.zeros(E, dtype=wp.float32, device=DEV),
                    wp.zeros((E, 1), dtype=wp.int32, device=DEV),
                    wp.zeros(2 * E, dtype=wp.int32, device=DEV)[::2]):
            for call in (c.regenerate, c.enqueue_generate):
                with pytest.raises(ValueError, match="mask"):
                    call(mask=bad, advance=True)
    assert launches == []
    after = _snap(c)
    assert all(after[k].tobytes() == before[k].tobytes() for k in before)
    m.assign(np.array([0, 0, 1, 0], np.int32))
    c.regenerate(mask=m, advance=True)
    np.testing.assert_array_equal(c.result.generation.numpy(), [0, 0, 1, 0])


@pytest.mark.parametrize("variant", ["track-sdf", "gates-posts"])
def test_course_masked_paths_do_not_allocate(variant, monkeypatch):
    c = _course(variant)
    m = _i32([0, 1, 1, 0])
    _forbid_allocation(monkeypatch)
    c.regenerate(mask=m, advance=True)
    c.regenerate(mask=c.inputs["regen_mask"], advance=True)
    c.regenerate(advance=True)
    c.enqueue_generate(mask=m, advance=True)
    c.generate(seeds=5)


# ---------------------------------------------------------------------------------------
# Fake CUDA device: keyed self-graphs, skip_unless lowering, _compat, interop.torch
# ---------------------------------------------------------------------------------------

class _Body:
    def __init__(self) -> None:
        self.ctxs: list = []

    def __call__(self, ctx) -> None:
        self.ctxs.append(ctx)


def test_runner_caches_one_self_graph_per_key(monkeypatch):
    fake = FakeCuda(monkeypatch)
    r = GraphRunner("cuda:0", capturable=True, name="toy")
    plain, keyed = _Body(), _Body()
    r.prepare(plain)
    assert r.graph is None and r.graph_keys == ()
    r.run(plain)
    g0 = r.graph
    assert g0 is not None and r.graph_for(None) is g0 and len(plain.ctxs) == 2
    r.run(keyed, key=(True, 0x10))
    g1 = r.graph_for((True, 0x10))
    assert g1 is not None and g1 is not g0 and len(keyed.ctxs) == 1
    assert keyed.ctxs[0].stream is fake.captures[1].stream     # recorded, not run
    r.run(keyed, key=(True, 0x10))
    r.run(plain)
    assert len(keyed.ctxs) == 1 and len(plain.ctxs) == 2      # replays only
    assert fake.launched == [(g0, fake.current), (g1, fake.current), (g1, fake.current),
                             (g0, fake.current)]
    assert r.graph is g0 and set(r.graph_keys) == {None, (True, 0x10)}
    # recording into a caller's capture ignores the key: no capture, no launch
    fake.current.is_capturing = True
    r.run(keyed, key=(True, 0x20))
    assert len(keyed.ctxs) == 2 and r.graph_for((True, 0x20)) is None
    assert len(fake.captures) == 2 and len(fake.launched) == 4


def test_runner_bounds_its_keyed_self_graphs(monkeypatch):
    """A new key per call (e.g. a fresh mask array each time) stops capturing once
    MAX_KEYED_GRAPHS keyed graphs are cached: further new keys run eagerly, with one
    warning; cached keys and the default body still replay."""
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(runner_mod, "MAX_KEYED_GRAPHS", 2)
    r = GraphRunner("cuda:0", capturable=True, name="toy")
    plain, keyed = _Body(), _Body()
    r.prepare(plain)
    r.run(plain)
    r.run(keyed, key=(True, 0x10))
    r.run(keyed, key=(True, 0x20))
    assert len(fake.captures) == 3 and len(keyed.ctxs) == 2
    with pytest.warns(RuntimeWarning, match="stable mask"):
        r.run(keyed, key=(True, 0x30))
    assert len(fake.captures) == 3 and len(keyed.ctxs) == 3
    assert keyed.ctxs[-1].stream is fake.current                # ran eagerly
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # warned once only
        r.run(keyed, key=(True, 0x40))
        r.run(keyed, key=(True, 0x10))                          # cached: replay
        r.run(plain)
    assert len(fake.captures) == 3 and len(keyed.ctxs) == 4
    assert set(r.graph_keys) == {None, (True, 0x10), (True, 0x20)}
    assert [g for g, _ in fake.launched][-2:] == [r.graph_for((True, 0x10)), r.graph]


def test_legacy_runner_self_graphs_only_its_default_body(monkeypatch):
    """A legacy-plugin producer self-captures on Warp's SHARED current stream, once: a
    keyed body runs eagerly there instead (a second capture would reopen the window in
    which other calls on that stream must serialize)."""
    fake = FakeCuda(monkeypatch)
    r = GraphRunner("cuda:0", capturable=True, legacy_stream=True, name="legacy")
    plain, keyed = _Body(), _Body()
    r.prepare(plain)
    r.run(plain)
    assert r.graph is not None and len(fake.captures) == 1
    for _ in range(2):
        r.run(keyed, key=(True, 0x40))
    assert len(keyed.ctxs) == 2 and all(c.stream is fake.current for c in keyed.ctxs)
    assert len(fake.captures) == 1 and r.graph_keys == (None,)
    assert fake.launched == [(r.graph, fake.current)]


def test_runner_keyed_capture_requires_prepare(monkeypatch):
    FakeCuda(monkeypatch)
    r = GraphRunner("cuda:0", capturable=True, name="toy")
    body = _Body()
    with pytest.raises(RuntimeError, match="prepare"):
        r.run(body, key=("advance",))
    assert body.ctxs == [] and r.graph_keys == ()


def _fake_facade_with_seeds(monkeypatch, make):
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(wp, "empty", lambda *a, **k: object())
    monkeypatch.setattr(wp, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(wp, "copy", lambda *a, **k: pytest.fail("unexpected wp.copy"))
    prologues, advances = patch_facade_seeds(monkeypatch)
    cfg, runs, result, cls = make(monkeypatch)
    gen = cls(cfg, _fake_rng())
    return fake, gen, runs, prologues, advances


@pytest.mark.parametrize("make", _FACADES)
def test_facade_self_graph_per_advance_and_mask(monkeypatch, make):
    fake, gen, runs, prologues, advances = _fake_facade_with_seeds(monkeypatch, make)
    assert len(runs) == 1 and advances == []                 # prepare: plain body
    gen.generate()
    g0 = gen.graph
    m = types.SimpleNamespace(ptr=0x1000)
    gen.generate(advance=True, mask=m)
    (cap0, cap1) = fake.captures
    assert advances == [(m, cap1.stream)]                     # recorded, in the graph
    assert len(runs) == 3 and gen.graph is g0
    gen.generate(advance=True, mask=types.SimpleNamespace(ptr=0x1000))
    assert len(runs) == 3 and len(fake.captures) == 2         # same pointer: replay
    gen.generate(advance=True)                                # all envs: another key
    assert len(fake.captures) == 3 and advances[-1] == (None, fake.captures[2].stream)
    assert [g for g, _ in fake.launched] == [g0, cap1.graph, cap1.graph,
                                             fake.captures[2].graph]
    side = FakeStream(fake.device)
    gen.enqueue(advance=True, mask=m, stream=side)             # eager, never self-graphs
    assert advances[-1] == (m, side) and len(fake.captures) == 3
    fake.current.is_capturing = True                          # caller's capture: RECORD
    gen.generate(advance=True, mask=m)
    assert advances[-1] == (m, fake.current) and len(fake.captures) == 3
    assert len(fake.launched) == 4 and prologues[-1] is fake.current
    with pytest.raises(ValueError, match="advance=True"):
        gen.generate(mask=m)
    assert len(advances) == 4 and len(runs) == 6


@pytest.mark.parametrize("make", _FACADES)
def test_facade_seed_state_defaults_to_the_constructor_stream(monkeypatch, make):
    """gen.seeds.advance() / reset_epoch() with no stream= run on the stream the
    facade was built with, the stream gen.generate() runs on, so the two are ordered."""
    from track_gen._src import gate_generator, track_generator

    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(wp, "empty", lambda *a, **k: object())
    monkeypatch.setattr(wp, "zeros", lambda *a, **k: object())
    got: list = []
    for mod in (track_generator, gate_generator):
        monkeypatch.setattr(mod, "SeedState",
                            lambda *a, **k: got.append(k) or FakeSeedState([], []))
    cfg, runs, _, cls = make(monkeypatch)
    s = FakeStream(fake.device)
    cls(cfg, _fake_rng(), stream=s)
    assert got == [{"default_stream": s}] and runs[0].stream is s


def test_seed_state_default_stream_resolution(monkeypatch):
    st = SeedState(E, DEV)
    assert st._ctx(None, None).stream is None                  # CPU: never a stream
    fake = FakeCuda(monkeypatch)
    s, other = FakeStream(fake.device), FakeStream(fake.device)
    st._device, st._default_stream = "cuda:0", s                # as built on CUDA
    assert st._ctx(None, None).stream is s
    assert st._ctx(other, None).stream is other                 # explicit stream wins
    st._default_stream = None
    assert st._ctx(None, None).stream is fake.current


def _skip_setup(monkeypatch, *, capturing: bool, registered: bool, supported: bool):
    fake = FakeCuda(monkeypatch)
    fake.device.captures = {}
    calls: list = []

    def capture_if(condition, on_true=None, on_false=None, stream=None, **kw):
        calls.append(stream)
        on_true()

    monkeypatch.setattr(wp, "capture_if", capture_if)
    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: supported)
    s = FakeStream(fake.device, capturing=capturing)
    key = FakeStream(fake.device)                     # the object Warp registered ...
    key.cuda_stream = s.cuda_stream                   # ... for the same handle
    if registered:
        fake.device.captures[key] = FakeGraph(key)
    return LaunchContext("cuda:0", s), key, calls


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_skip_unless_lowering(monkeypatch, capturing, registered, supported):
    ctx, key, calls = _skip_setup(monkeypatch, capturing=capturing, registered=registered,
                                  supported=supported)
    flag = wp.zeros(1, dtype=wp.int32, device=DEV)
    flag.device = ctx.device                          # a [1] int32 flag on the fake GPU
    seen: list = []
    ctx.skip_unless(flag, seen.append)
    assert seen == [ctx]                              # body ran (or was recorded) once
    conditional = capturing and registered and supported
    assert calls == ([key] if conditional else [])    # the registered KEY object
    if registered:                                    # a registration that opted out
        _compat.set_conditional_allowed(key, False)
        ctx.skip_unless(flag, seen.append)
        assert len(seen) == 2 and len(calls) == int(conditional)
        _compat.set_conditional_allowed(key, True)
        ctx.skip_unless(flag, seen.append)
        assert len(seen) == 3 and len(calls) == 2 * int(conditional)
    n = len(seen)
    for bad in (wp.zeros(2, dtype=wp.int32, device=DEV),
                wp.zeros(1, dtype=wp.float32, device=DEV), np.zeros(1, np.int32)):
        with pytest.raises(TypeError, match="flag"):
            ctx.skip_unless(bad, seen.append)
    # a flag on another device would make wp.capture_if miss the capture and read the
    # flag on the host: refused before anything is recorded
    with pytest.raises(ValueError, match="device"):
        ctx.skip_unless(wp.zeros(1, dtype=wp.int32, device=DEV), seen.append)
    assert len(seen) == n and len(calls) == (2 if registered else 1) * int(conditional)


def test_skip_unless_on_cpu_runs_the_body_and_never_reads_the_flag(monkeypatch):
    monkeypatch.setattr(wp, "capture_if", lambda *a, **k: pytest.fail("capture_if on CPU"))
    ctx = LaunchContext.for_call(DEV)
    seen: list = []
    ctx.skip_unless(wp.zeros(1, dtype=wp.int32, device=DEV), seen.append)
    assert seen == [ctx]
    fake = FakeCuda(monkeypatch)
    flag = wp.zeros(1, dtype=wp.int32, device=DEV)
    flag.device = fake.device
    with pytest.raises(ValueError, match="device"):            # GPU flag, CPU context
        ctx.skip_unless(flag, seen.append)
    assert seen == [ctx]


def test_compat_registered_stream_and_conditional_ok(monkeypatch):
    fake = FakeCuda(monkeypatch)
    assert _compat.registered_stream(None) is None
    s = FakeStream(fake.device, capturing=True)
    assert _compat.registered_stream(s) is None                # device has no captures
    fake.device.captures = {}
    assert _compat.registered_stream(s) is None
    key = FakeStream(fake.device)
    key.cuda_stream = s.cuda_stream
    fake.device.captures[key] = FakeGraph(key)
    assert _compat.registered_stream(s) is key and _compat.registered_stream(key) is key
    assert _compat.registered_stream(types.SimpleNamespace()) is None
    assert _compat.conditional_allowed(key) and not _compat.conditional_allowed(s)
    _compat.set_conditional_allowed(key, False)
    assert not _compat.conditional_allowed(key)
    _compat.set_conditional_allowed(s, False)                  # nothing registered: no-op
    assert _compat.conditional_allowed(types.SimpleNamespace()) is False
    assert _compat.conditional_ok(wp.get_device("cpu")) is False
    for supported in (True, False):
        monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda v=supported: v)
        assert _compat.conditional_ok(fake.device) is supported

    def boom():
        raise RuntimeError("no driver")
    monkeypatch.setattr(wp, "is_conditional_graph_supported", boom)
    assert _compat.conditional_ok(fake.device) is False


class _RecordingCapture:
    """A Warp ScopedCapture stand-in that registers its stream like Warp does."""

    opened: list = []

    def __init__(self, device=None, stream=None, force_module_load=None, external=False,
                 capture_mode=None, **kw) -> None:
        self.device, self.stream, self.external = device, stream, external
        self.capture_mode = capture_mode
        self.graph = None

    def __enter__(self):
        _RecordingCapture.opened.append(self)
        self.device.captures[self.stream] = self._g = FakeGraph(self.stream)
        return self

    def __exit__(self, *exc):
        del self.device.captures[self.stream]
        self.graph = self._g
        return False


def test_interop_torch_registered(monkeypatch):
    fake = FakeCuda(monkeypatch)
    fake.device.captures = {}
    _RecordingCapture.opened = []
    monkeypatch.setattr(wp, "ScopedCapture", _RecordingCapture)
    idle = FakeStream(fake.device)
    with pytest.raises(CaptureError, match="torch.cuda.graph"):
        with tg_torch.registered(idle):
            pass
    s = FakeStream(fake.device, capturing=True)
    graph = types.SimpleNamespace()
    with tg_torch.registered(s, graph=graph) as ws:
        assert ws is s and _compat.registered_stream(s) is s
        # default: registered for anchoring only, no conditional nodes (torch's
        # allocator tracks its capture by id, which a capture_if would change)
        assert not _compat.conditional_allowed(s)
        # nested: already registered -> the registered object, no second capture
        alias = FakeStream(fake.device, capturing=True)
        alias.cuda_stream = s.cuda_stream
        with tg_torch.registered(alias, conditional=True) as inner:
            assert inner is s and not _compat.conditional_allowed(s)
        # a side stream that joined this capture (same capture id): yielded as is,
        # never registered a second time
        side = FakeStream(fake.device, capturing=True)
        monkeypatch.setattr(_compat, "registered_graph",
                            lambda st: fake.device.captures.get(s) if st is side else None)
        with tg_torch.registered(side) as joined:
            assert joined is side and _compat.registered_stream(side) is None
    (cap,) = _RecordingCapture.opened
    assert cap.external and cap.stream is s and cap.capture_mode == wp.CaptureMode.GLOBAL
    assert _compat.registered_stream(s) is None              # unregistered on exit
    assert graph._track_gen_owners == [cap.graph]            # retained on the torch graph
    with tg_torch.registered(s, capture_mode="thread_local"):
        pass
    assert _RecordingCapture.opened[-1].capture_mode == wp.CaptureMode.THREAD_LOCAL
    with tg_torch.registered(s, conditional=True) as ws:      # explicit opt-in
        assert ws is s and _compat.conditional_allowed(s)
    with pytest.raises(ValueError, match="capture_mode"):
        with tg_torch.registered(s, capture_mode="loose"):
            pass


def test_interop_torch_retain_keeps_owners_alive():
    class _Graph:
        pass

    class _Owner:
        pass

    g, owner = _Graph(), _Owner()
    alive = weakref.ref(owner)
    tg_torch.retain(g, owner, None, owner)
    tg_torch.retain(g, owner)
    assert g._track_gen_owners == [owner]
    del owner
    gc.collect()
    assert alive() is not None
    del g
    gc.collect()
    assert alive() is None
    with pytest.raises(TypeError, match="__dict__"):
        tg_torch.retain(object(), _Owner())


def test_new_modules_do_not_import_torch():
    code = "\n".join([
        "import sys",
        "import track_gen, track_gen.seeds, track_gen.ops, track_gen.interop",
        "import track_gen.interop.torch",
        "from track_gen.interop import torch as t",
        "assert callable(t.registered) and callable(t.retain)",
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))",
        "assert not leaked, leaked",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
