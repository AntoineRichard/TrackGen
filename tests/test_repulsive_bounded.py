"""Repulsive's bounded schedule, absorbing freeze and CPU lowering parity.

Actual Stage/core calls use fixed seeds; geometric validity does not require early
freeze. A separate deterministic fixture drives the real Repulsive step, freeze
and active kernels with controlled metrics, exercising early exit and mixed
frozen/unfrozen rows without searching for a platform-dependent equilibrium.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import warp as wp

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import warp_generate_repulsive as rep
from track_gen.exec import LaunchContext
from track_gen.exec.context import _force_lowering, _probe_loops
from track_gen.plugin import CaptureClass, registry

DEV = "cpu"
_SMALL = dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384, half_width=0.1)
_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
           "count", "winding", "truncated", "fail_mask", "half_width")


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
    assert (s.n_iters, s.stage_starts) == (189, [0, 59, 93])
    F = rep._first_check(s.stage_starts, 16)
    assert F == 96 and len(range(F, s.n_iters, 16)) == 6
    assert s.active.shape == (1,) and s.active.dtype is wp.int32
    _, rec = _run(_gen([0, 1], **_SMALL))            # the small config: same schedule
    assert (rec.n_iters, rec.first_check, rec.check_every, rec.windows) == (189, 96, 16, 6)
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
    kw = _SMALL if config == "small" else {}
    gen = _gen([0, 1, 7, 99991], **kw)
    a, rec_a = _run(gen, None)
    c, rec_c = _run(gen, "unroll")
    assert rec_a.lowering == "poll" and rec_c.lowering == "unroll"
    assert rec_a.ran <= rec_a.windows
    assert rec_c.ran == rec_c.windows == 6
    assert _same(a, c)


@wp.kernel
def _controlled_metrics_k(it: int, ready_at: wp.array(dtype=wp.int32), n: int,
                          g: wp.array(dtype=wp.vec2d),
                          stationarity: wp.array(dtype=wp.float32)):
    # The only synthetic part is the solver input. Each row has one stationary
    # window; later metrics become large again to challenge absorbing freeze.
    e, i = wp.tid()
    g[e * n + i] = wp.vec2d(wp.float64(it + 1), wp.float64(i + 1))
    if i == 0:
        stationary = it >= ready_at[e] and it < ready_at[e] + 4
        stationarity[e] = wp.where(stationary, float(0.0), float(1.0))


class _ControlledFreeze:
    """Stationary square geometry, real Repulsive kernels and a bounded loop.

    Changed gradients make a missing frozen guard in _step_k observable even
    though the shape itself is held fixed. No generated trajectory is mocked.
    """
    n = 4

    def __init__(self, ready_at=(0, 4, 8)):
        self.e = len(ready_at)
        square = np.asarray([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], np.float32)
        self.center = wp.array(np.tile(square, (self.e, 1)), dtype=wp.vec2f, device=DEV)
        self.ready_at = wp.array(np.asarray(ready_at, np.int32), dtype=wp.int32, device=DEV)
        self.g = wp.zeros(self.e * self.n, dtype=wp.vec2d, device=DEV)
        self.gmean = wp.zeros(self.e, dtype=wp.vec2d, device=DEV)
        self.energy = wp.full(self.e, 100., dtype=wp.float64, device=DEV)
        self.gmax = wp.full(self.e, 32., dtype=wp.float32, device=DEV)
        self.reached = wp.ones(self.e, dtype=wp.int32, device=DEV)
        self.target = wp.full(self.e, 8., dtype=wp.float64, device=DEV)
        self.area_initial = wp.full(self.e, 4., dtype=wp.float64, device=DEV)
        self.area_prev = wp.zeros(self.e, dtype=wp.float64, device=DEV)
        self.frozen = wp.zeros(self.e, dtype=wp.int32, device=DEV)
        self.md = wp.zeros(self.e, dtype=wp.float64, device=DEV)
        self.movement = wp.zeros(self.e, dtype=wp.float32, device=DEV)
        self.stationarity = wp.zeros(self.e, dtype=wp.float32, device=DEV)
        self.blocked = wp.zeros(self.e, dtype=wp.int32, device=DEV)
        self.active = wp.zeros(1, dtype=wp.int32, device=DEV)
        self.proposal = wp.zeros(self.e * self.n, dtype=wp.vec2f, device=DEV)

    def enqueue(self):
        ctx = LaunchContext.for_call(DEV)
        ctx.fill(self.frozen, 0)
        ctx.fill(self.active, 1)
        ctx.copy(self.area_prev, self.area_initial)
        ctx.fill(self.proposal, wp.vec2f(-17., 19.))

        def body(c, it):
            c.launch(_controlled_metrics_k, (self.e, self.n),
                     [it, self.ready_at, self.n, self.g, self.stationarity])
            c.launch(rep._step_k, (self.e, self.n),
                     [self.center, self.g, self.gmean, self.energy, self.gmax,
                      .01, 10., self.n, self.frozen, self.proposal])
            if (it + 1) % 4 == 0:
                c.launch(rep._freeze_update_k, self.e,
                         [self.center, self.n, self.reached, .01, self.area_prev,
                          self.frozen, self.md, self.target, self.movement,
                          self.stationarity, self.blocked, 10.])
                c.fill(self.active, 0)
                c.launch(rep._active_k, self.e, [self.frozen, self.active])

        ctx.bounded_loop(20, body, active=self.active, first_check=4, check_every=4)

    def rows(self):
        return {name: getattr(self, name).numpy().copy()
                for name in ('center', 'proposal', 'frozen', 'active', 'area_prev', 'md')}


def test_controlled_repulsive_freeze_skips_windows_and_is_absorbing_in_a_mixed_batch():
    results = {}
    for mode in (None, "unroll"):
        fixture = _ControlledFreeze()
        with _force_lowering(mode), _probe_loops() as recs:
            fixture.enqueue()
        (rec,) = recs
        results[mode] = fixture.rows()
        assert (rec.windows, rec.ran) == (4, 2 if mode is None else 4)
        assert results[mode]['frozen'].tolist() == [1, 1, 1]
        assert results[mode]['active'].tolist() == [0]
        # tau*min(n*R^2/E,R/gmax) = .003125; rows freeze after it3/7/11.
        square = fixture.center.numpy().reshape(3, 4, 2)
        expected = square.astype(np.float64)
        for row, last_it in enumerate((3, 7, 11)):
            expected[row, :, 0] -= .003125 * (last_it + 1)
            expected[row, :, 1] -= .003125 * np.arange(1, 5)
        np.testing.assert_array_equal(fixture.proposal.numpy().reshape(3, 4, 2),
                                      expected.astype(np.float32))
    assert _same(results[None], results["unroll"])
    # Each row's freeze time and bytes are independent of the other rows.
    for row, ready_at in enumerate((0, 4, 8)):
        fixture = _ControlledFreeze((ready_at,))
        fixture.enqueue()
        assert fixture.proposal.numpy().tobytes() == results[None]['proposal'][row * 4:(row + 1) * 4].tobytes()


def test_controlled_repulsive_freeze_replays_a_captured_unrolled_loop():
    fixture = _ControlledFreeze()
    fixture.enqueue()
    expected = fixture.rows()
    fixture.active.fill_(0)
    fixture.frozen.fill_(1)
    with wp.ScopedCapture(device=DEV) as cap, _probe_loops() as recs:
        fixture.enqueue()
    assert [r.lowering for r in recs] == ["unroll"]
    fixture.proposal.fill_(-5.)
    wp.capture_launch(cap.graph)
    assert _same(fixture.rows(), expected)


def _core_scratch(cfg):
    """Fresh solver storage with the same known road-width reservation as the Stage."""
    scratch = rep._alloc_repulsive_scratch(cfg)
    scratch.half_width = float(cfg.half_width)
    return scratch


def _core(seed, cfg, out, valid, scratch, size):
    rep._generate_repulsive(seed, cfg, out, valid, scratch,
                            LaunchContext.for_call(DEV), size)


def test_standalone_current_core_equals_the_stage_raw_and_validity():
    gen = _gen(np.arange(11, 19, dtype=np.int32), **_SMALL)
    stage, _ = _run(gen)
    cfg = gen._config
    scratch = _core_scratch(cfg)
    out = wp.zeros(8 * 64, dtype=wp.vec2f, device=DEV)
    valid = wp.zeros(8, dtype=wp.int32, device=DEV)
    size = wp.full(8, float(cfg.size), dtype=wp.float32, device=DEV)
    _core(wp.clone(gen._seed_buf), cfg, out, valid, scratch, size)
    assert out.numpy().reshape(8, -1).tobytes() == stage["raw"].tobytes()
    assert valid.numpy().tobytes() == gen.program.tap("gen.valid").numpy().tobytes()


@pytest.mark.parametrize("stall_window", [48, 8])
def test_current_core_with_a_mismatched_stall_window(stall_window):
    """Call-time cadence and a reset flag work even with stale frozen scratch."""
    e = 4
    seeds = wp.array(np.asarray([11, 12, 13, 14], np.int32), dtype=wp.int32, device=DEV)
    alloc_cfg = TrackGenConfig(generator="repulsive", device=DEV, num_envs=e, **_SMALL)
    gen_cfg = TrackGenConfig(generator="repulsive", device=DEV, num_envs=e,
                             repulsive_stall_window=stall_window, **_SMALL)
    scratch = _core_scratch(alloc_cfg)
    out = wp.zeros(e * 64, dtype=wp.vec2f, device=DEV)
    valid = wp.zeros(e, dtype=wp.int32, device=DEV)
    size = wp.ones(e, dtype=wp.float32, device=DEV)
    results = {}
    for mode in (None, "unroll"):
        scratch.active.fill_(0)
        scratch.frozen.fill_(1)
        with _force_lowering(mode), _probe_loops() as recs:
            _core(seeds, gen_cfg, out, valid, scratch, size)
        results[mode] = (out.numpy().tobytes(), valid.numpy().tobytes())
        (rec,) = recs
        assert rec.first_check == rep._first_check(scratch.stage_starts, stall_window)
        assert rec.check_every == stall_window and rec.stage_id is None
        assert rec.ran > 0
    assert results[None] == results["unroll"]


def test_repulsive_has_no_obsolete_frozen_numeric_oracle():
    from tests._reference_generators import spec

    with pytest.raises(ValueError, match="Repulsive has no frozen numeric oracle"):
        spec("repulsive")


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
    gen.generate()                                    # reuse populated solver scratch
    ref = _rows(gen)
    gen._track.center.fill_(-5.0)
    with wp.ScopedCapture(device=DEV) as cap, _probe_loops() as recs:
        gen.generate()
    assert [r.lowering for r in recs] == ["unroll"]
    gen._track.center.fill_(-5.0)
    wp.capture_launch(cap.graph)
    assert _same(_rows(gen), ref)
