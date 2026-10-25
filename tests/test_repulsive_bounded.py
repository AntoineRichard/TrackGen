"""The repulsive generator as a BOUNDED Stage (roadmap PR12), on CPU.

Its growth loop is a ``ctx.bounded_loop`` with ``active`` = any env not frozen, refreshed
by every stall check, checked every stall window from ``first_check`` (the first window
start at the final stage). Because the per-env freeze is absorbing, the CPU host poll
(lowering (a), which stops at the first window after every env froze) and the unrolled
lowering (c) write the same bytes, env ``e`` is the same alone or inside a larger batch
in either lowering, a CPU graph capture (recorded unrolled) replays the eager batch,
and the legacy entry ``generate_repulsive_warp`` (also with a generate-time config whose
stall window differs from the scratch's) matches the full loop.
"""
from __future__ import annotations
from tests._reference_generators import generate_raw

import random

import numpy as np
import pytest
import warp as wp

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import warp_generate_repulsive as rep
from track_gen.exec.context import _force_lowering, _probe_loops
from track_gen.plugin import CaptureClass, registry

DEV = "cpu"
_SMALL = dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384, half_width=0.1)
# A loose stall tolerance: every env freezes soon after reaching its length, so the poll
# skips windows. Which envs freeze, and when, is chaotic in float rounding and so differs
# between platforms (the CPU trajectories on x86 and arm64 are not bit-equal): the tests
# that need an early exit find their seed block at runtime (:func:`_early_exit_seeds`).
_FREEZE = dict(_SMALL, repulsive_stall_area_tol=0.5)
_SEED_BASES = (11, 111, 1011, 2011, 3011)
_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
           "count", "winding", "truncated")


def _gen(seeds, **kw) -> TrackGenerator:
    seeds = np.asarray(seeds, np.int32)
    E = int(seeds.shape[0])
    return TrackGenerator(
        TrackGenConfig(generator="repulsive", device=DEV, num_envs=E, **kw),
        PerEnvSeededRNG(seeds=wp.array(seeds, dtype=wp.int32, device=DEV), num_envs=E,
                        device=DEV))


def _rows(gen) -> dict:
    """field -> [E, ...] host copy (raw centerline included)."""
    E = gen.num_envs
    out = {f: getattr(gen._track, f).numpy().reshape(E, -1).copy() for f in _FIELDS}
    out["raw"] = gen._scratch.gen_centerline.numpy().reshape(E, -1).copy()
    return out


def _run(gen, mode=None):
    with _force_lowering(mode), _probe_loops() as recs:
        gen.enqueue()
    (rec,) = recs
    return _rows(gen), rec


def _early_exit_seeds(E: int, kw: dict) -> np.ndarray:
    """An ``[E]`` seed block whose every env freezes before the last window ON THIS
    PLATFORM (the eager poll stops early), searched over :data:`_SEED_BASES`."""
    tried = []
    for base in _SEED_BASES:
        seeds = np.arange(base, base + E, dtype=np.int32)
        _, rec = _run(_gen(seeds, **kw))
        if rec.ran < rec.windows:
            return seeds
        tried.append((base, rec.ran, rec.windows))
    pytest.skip(f"no seed block freezes every env early on this platform: {tried}")


def _same(a: dict, b: dict, rows_a=slice(None), rows_b=slice(None)) -> bool:
    return all(np.ascontiguousarray(a[k][rows_a]).tobytes() ==
               np.ascontiguousarray(b[k][rows_b]).tobytes() for k in a)


# ---------------------------------------------------------------------------------------
# Registration and schedule
# ---------------------------------------------------------------------------------------

def test_repulsive_is_a_bounded_stage():
    assert registry.get("generator", "repulsive") is rep.Repulsive
    info = registry.describe("generator", "repulsive")
    assert info.capture is CaptureClass.BOUNDED and info.capturable
    assert info.origin == "track_gen._src.warp_generate_repulsive" and info.builtin
    assert info.params_schema["properties"]["repulsive_stall_window"]["default"] == 16


def test_the_schedule_at_the_defaults():
    s = rep._alloc_repulsive_scratch(TrackGenConfig(generator="repulsive", device=DEV,
                                                   num_envs=1))
    assert (s.n_iters, s.stage_starts) == (269, [0, 59, 117])
    F = rep._first_check(s.stage_starts, 16)
    assert F == 128 and len(range(F, s.n_iters, 16)) == 9
    assert s.active.shape == (1,) and s.active.dtype is wp.int32
    _, rec = _run(_gen([0, 1], **_SMALL))            # the small config: same schedule
    assert (rec.n_iters, rec.first_check, rec.check_every, rec.windows) == (269, 128, 16, 9)
    assert rec.stage_id == "generator" and rec.lowering == "poll"


def test_windows_align_with_stall_checks_and_hold_no_stage_transition():
    """Claims 1-2 of the design (property-checked): every window start ``lo`` follows a
    stall-check iteration ``lo - 1`` at the final stage, the first check is the first
    stall check, and every stage transition happens before ``first_check``."""
    rng = random.Random(0)
    for _ in range(3000):
        growth = rng.uniform(0.001, 0.2)
        gmax = rng.uniform(1.01, 20.0)
        settle = rng.randint(1, 80)
        stages = sorted(rng.sample(range(4, 1024, 4), rng.randint(1, 5)))
        W = rng.randint(1, 64)
        n_ratchet = int(np.ceil(np.log(gmax) / np.log1p(growth)))
        n_iters = int(np.ceil(n_ratchet * 1.6)) + settle
        starts = rep._stage_schedule(stages, growth, rep._SUBDIV_RATIO, n_ratchet)
        F = rep._first_check(starts, W)
        assert all(t < F for t in starts)
        checks = [it for it in range(n_iters)
                  if it >= starts[-1] and (it + 1) % W == 0]
        for lo in range(F, n_iters, W):
            assert lo - 1 in checks
        assert (checks[0] == F - 1) if checks else F >= n_iters


# ---------------------------------------------------------------------------------------
# Lowerings (a) == (c)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("config", ["small", "default"])
def test_the_host_poll_and_the_unrolled_loop_write_the_same_bytes(config):
    kw = _FREEZE if config == "small" else dict(repulsive_stall_area_tol=0.5)
    gen = _gen(_early_exit_seeds(8, kw), **kw)
    a, rec_a = _run(gen, None)                        # CPU eager default: the poll
    c, rec_c = _run(gen, "unroll")
    assert rec_a.lowering == "poll" and rec_c.lowering == "unroll"
    assert rec_a.ran < rec_a.windows, rec_a           # (a) skipped at least one window
    assert rec_c.ran == rec_c.windows == 9
    assert _same(a, c)


def test_the_legacy_entry_equals_the_stage():
    seeds = np.arange(11, 19, dtype=np.int32)
    gen = _gen(seeds, **_SMALL)
    stage, _ = _run(gen)
    cfg = TrackGenConfig(generator="repulsive", device=DEV, num_envs=8, **_SMALL)
    s = rep._alloc_repulsive_scratch(cfg)
    out = wp.zeros(8 * 64, dtype=wp.vec2f, device=DEV)
    valid = wp.zeros(8, dtype=wp.int32, device=DEV)
    generate_raw("repulsive", wp.array(seeds, dtype=wp.int32, device=DEV), cfg, out,
                                valid, s)
    assert out.numpy().reshape(8, -1).tobytes() == stage["raw"].tobytes()
    assert (valid.numpy() == 1).all()


@pytest.mark.parametrize("stall_window", [48, 8])
def test_the_legacy_entry_with_a_mismatched_stall_window(stall_window):
    """The scratch is allocated for one config, ``generate_repulsive_warp`` called with
    another whose stall window differs: ``first_check`` comes from the generate-time
    window (the one that drives the stall checks), and the prologue resets ``active``, so
    the poll stays exact after a generation that froze every env."""
    E = 8
    seeds = wp.array(_early_exit_seeds(E, _FREEZE), dtype=wp.int32, device=DEV)
    alloc_cfg = TrackGenConfig(generator="repulsive", device=DEV, num_envs=E, **_FREEZE)
    gen_cfg = TrackGenConfig(generator="repulsive", device=DEV, num_envs=E,
                             repulsive_stall_window=stall_window, **_FREEZE)
    s = rep._alloc_repulsive_scratch(alloc_cfg)
    out = wp.zeros(E * 64, dtype=wp.vec2f, device=DEV)
    valid = wp.zeros(E, dtype=wp.int32, device=DEV)
    generate_raw("repulsive", seeds, alloc_cfg, out, valid, s)
    assert s.frozen.numpy().all() and int(s.active.numpy()[0]) == 0   # all frozen
    results = {}
    for mode in (None, "unroll"):
        with _force_lowering(mode), _probe_loops() as recs:
            generate_raw("repulsive", seeds, gen_cfg, out, valid, s)
        results[mode] = out.numpy().tobytes()
        (rec,) = recs
        assert rec.first_check == rep._first_check(s.stage_starts, stall_window)
        assert rec.check_every == stall_window and rec.stage_id is None
    assert results[None] == results["unroll"]


# ---------------------------------------------------------------------------------------
# Batch independence, CPU graph capture
# ---------------------------------------------------------------------------------------

def _inside(small_seeds, big_seeds, kw):
    small, big = _gen(small_seeds, **kw), _gen(big_seeds, **kw)
    rows = [int(np.flatnonzero(big_seeds == s)[0]) for s in small_seeds]
    for mode in (None, "unroll"):
        a, _ = _run(small, mode)
        b, _ = _run(big, mode)
        assert _same(a, b, slice(None), rows), f"lowering {mode or 'poll'}"


def test_env_e_is_the_same_inside_a_larger_batch_small_config():
    small = np.arange(11, 19, dtype=np.int32)
    big = np.concatenate([np.arange(1000, 1030), small, np.arange(2000, 2026)])
    _inside(small, big.astype(np.int32), _SMALL)


@pytest.mark.slow
def test_env_e_is_the_same_inside_a_larger_batch_default_config():
    small = np.arange(11, 19, dtype=np.int32)
    big = np.concatenate([np.arange(1000, 1030), small, np.arange(2000, 2026)])
    _inside(small, big.astype(np.int32), {})


def test_a_cpu_graph_capture_records_the_loop_unrolled_and_replays_the_eager_batch():
    gen = _gen([0, 1, 7, 99991], **_SMALL)
    gen.generate()                                    # every env frozen: a stale active
    ref = _rows(gen)
    gen._track.center.fill_(-5.0)
    with wp.ScopedCapture(device=DEV) as cap, _probe_loops() as recs:
        gen.generate()
    assert [r.lowering for r in recs] == ["unroll"]
    gen._track.center.fill_(-5.0)
    wp.capture_launch(cap.graph)
    assert _same(_rows(gen), ref)
