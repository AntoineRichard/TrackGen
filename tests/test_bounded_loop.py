"""``LaunchContext.bounded_loop``, ``ops.all_set``, ``HostOps`` and the BOUNDED contract
(roadmap PR12), on CPU.

- Semantics: the static ``it``, the unconditional prefix ``[0, first_check)``, the
  windows of ``check_every`` iterations (a short last one), the check before each window
  (host poll on CPU: stops at the first 0 and stays stopped), validation.
- Lowering policy (from the capture state only, never data): CPU eager polls; CUDA eager
  runs every window with no host sync (fake device); inside a Warp-registered capture
  with conditional nodes each window is one ``wp.capture_if``; anything else unrolls; the
  forced seams; CPU graph captures (Warp APIC) never poll.
- The declaration check (a STATIC stage passing ``active=`` is refused), the private loop
  probe, ``GraphRunner.prepare`` forcing the unrolled lowering.
- The no-host-sync guard: the loop's host poll is the one sanctioned readback; a
  plugin's own sync (inside a loop body, or a direct call of the poll) is not. The kit's
  hot-path checks run the loops unrolled (a late-window sync / allocation is caught).
- ``HostOps`` (``ctx.host()``) and ``StageBuilder.host_buffer`` on CPU.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import warp as wp

from tests import _bounded_toys as T
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen import ops
from track_gen.exec import CaptureError, GraphRunner, HostOps, LaunchContext, NotCapturableError
from track_gen.exec import _compat
from track_gen.exec import context as context_mod
from track_gen.exec import host as host_mod
from track_gen.exec.context import _force_lowering, _probe_loops
from track_gen.pipeline import presets
from track_gen.plugin import CaptureClass, GeneratorSpec, Stage, registry
from track_gen.testing import check_generator, check_stage, no_host_sync
from track_gen.testing.contract import HotPathGuard
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 8


@pytest.fixture(autouse=True)
def _toys():
    T.register_bounded_toys(registry)
    T.register_broken_bounded_toys(registry)


@wp.kernel
def _set_k(flag: wp.array(dtype=wp.int32), value: int):
    flag[0] = value


def _ctx() -> LaunchContext:
    return LaunchContext.for_call(DEV)


def _flag(value: int = 1) -> wp.array:
    return wp.full(1, value, dtype=wp.int32, device=DEV)


class _Loop:
    """A loop body that records its iterations and writes ``active = 0`` (through the
    context) on the iterations in ``stop_at``."""

    def __init__(self, active=None, stop_at=()) -> None:
        self.its: list = []
        self.active = active
        self.stop_at = set(stop_at)

    def __call__(self, ctx, it) -> None:
        assert isinstance(it, int)
        self.its.append(it)
        if self.active is not None and it in self.stop_at:
            ctx.launch(_set_k, 1, [self.active, 0])


@pytest.fixture
def polls(monkeypatch):
    """Count the host polls (the real poll still runs)."""
    seen: list = []
    real = host_mod._poll_i32

    def counting(ctx, arr):
        value = real(ctx, arr)
        seen.append(value)
        return value

    monkeypatch.setattr(host_mod, "_poll_i32", counting)
    return seen


# ---------------------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------------------

def test_without_active_every_iteration_runs_in_order(polls):
    body = _Loop()
    _ctx().bounded_loop(7, body, first_check=3, check_every=2)
    assert body.its == list(range(7)) and polls == []


def test_zero_iterations():
    body = _Loop()
    with _probe_loops() as recs:
        _ctx().bounded_loop(0, body)
        _ctx().bounded_loop(0, body, active=_flag())
    assert body.its == []
    assert [(r.windows, r.ran) for r in recs] == [(0, 0)]


@pytest.mark.parametrize("n,first", [(5, 5), (5, 9)])
def test_no_window_when_first_check_reaches_n_iters(polls, n, first):
    active = _flag(0)                       # never read: there is no window
    body = _Loop(active)
    with _probe_loops() as recs:
        _ctx().bounded_loop(n, body, active=active, first_check=first, check_every=2)
    assert body.its == list(range(n)) and polls == []
    assert recs[0].windows == 0


def test_first_check_zero_checks_before_iteration_0(polls):
    body = _Loop()
    _ctx().bounded_loop(6, body, active=_flag(0), check_every=2)
    assert body.its == [] and polls == [0]
    body = _Loop()
    _ctx().bounded_loop(6, body, active=_flag(1), check_every=2)
    assert body.its == list(range(6)) and polls == [0, 1, 1, 1]


def test_windows_with_a_short_last_one(polls):
    body = _Loop()
    with _probe_loops() as recs:
        _ctx().bounded_loop(10, body, active=_flag(1), first_check=2, check_every=3)
    assert body.its == list(range(10))
    assert polls == [1, 1, 1]                # windows [2,5), [5,8), [8,10)
    (rec,) = recs
    assert (rec.lowering, rec.windows, rec.ran) == ("poll", 3, 3)
    assert (rec.n_iters, rec.first_check, rec.check_every) == (10, 2, 3)
    assert rec.stage_id is None


def test_check_every_one(polls):
    active = _flag(1)
    body = _Loop(active, stop_at={4})
    _ctx().bounded_loop(9, body, active=active, first_check=2, check_every=1)
    assert body.its == [0, 1, 2, 3, 4] and polls == [1, 1, 1, 0]


def test_stops_at_the_first_zero_and_stays_stopped(polls):
    active = _flag(1)
    body = _Loop(active, stop_at={5})
    with _probe_loops() as recs:
        _ctx().bounded_loop(20, body, active=active, first_check=4, check_every=4)
    # windows [4,8) ran (5 wrote 0), the check before [8,12) read 0: nothing after
    assert body.its == list(range(8)) and polls == [1, 0]
    assert (recs[0].windows, recs[0].ran) == (4, 1)


def test_unrolled_lowering_runs_every_window(polls):
    active = _flag(1)
    body = _Loop(active, stop_at={5})
    with _force_lowering("unroll"), _probe_loops() as recs:
        _ctx().bounded_loop(20, body, active=active, first_check=4, check_every=4)
    assert body.its == list(range(20)) and polls == []
    assert (recs[0].lowering, recs[0].ran) == ("unroll", 4)


@pytest.mark.parametrize("kwargs,exc,match", [
    (dict(n_iters=True), TypeError, "n_iters must be an int"),
    (dict(n_iters=2.0), TypeError, "n_iters must be an int"),
    (dict(n_iters=-1), ValueError, "n_iters must be >= 0"),
    (dict(first_check=-2), ValueError, "first_check must be >= 0"),
    (dict(first_check=False), TypeError, "first_check must be an int"),
    (dict(check_every=0), ValueError, "check_every must be >= 1"),
    (dict(active=wp.zeros(1, dtype=wp.float32, device=DEV)), TypeError,
     r"active must be a \[1\] int32 wp.array"),
    (dict(active=wp.zeros(2, dtype=wp.int32, device=DEV)), TypeError,
     r"active must be a \[1\] int32 wp.array"),
    (dict(active=np.zeros(1, np.int32)), TypeError, r"active must be a \[1\] int32"),
])
def test_validation_before_any_iteration(kwargs, exc, match):
    body = _Loop()
    args = dict(n_iters=4, first_check=1, check_every=1)
    args.update(kwargs)
    n = args.pop("n_iters")
    with pytest.raises(exc, match=match):
        _ctx().bounded_loop(n, body, **args)
    assert body.its == []


def test_active_on_another_device_is_refused():
    ctx = LaunchContext.__new__(LaunchContext)
    ctx.device = _NamedDevice("cuda:7")
    ctx.stream = None
    body = _Loop()
    with pytest.raises(ValueError, match="bounded_loop: active is on cpu, but the launch "
                                         "context is on cuda:7"):
        ctx.bounded_loop(4, body, active=_flag())
    assert body.its == []


class _NamedDevice:
    is_cuda, is_cpu = True, False

    def __init__(self, alias: str) -> None:
        self.alias = alias

    def __str__(self) -> str:
        return self.alias


def test_skip_unless_returns_whether_it_recorded_a_conditional(monkeypatch):
    ran: list = []
    assert _ctx().skip_unless(_flag(), lambda c: ran.append(c)) is False and len(ran) == 1


# ---------------------------------------------------------------------------------------
# Lowering policy
# ---------------------------------------------------------------------------------------

class _FakeCtx:
    """What ``_bounded_mode`` reads of a context."""

    def __init__(self, cuda: bool, capturing: bool = False) -> None:
        self.device = types.SimpleNamespace(is_cuda=cuda, is_cpu=not cuda,
                                            alias="cuda:0" if cuda else "cpu")
        self.capturing = capturing


@pytest.fixture
def quiet_probes(monkeypatch):
    """No capture anywhere, no torch, no slot buffer (individual tests flip them)."""
    state = types.SimpleNamespace(cpu=False, registered=False, torch=None, buffer=None)
    monkeypatch.setattr(_compat, "cpu_capture_active", lambda dev: state.cpu)
    monkeypatch.setattr(_compat, "any_registered_capture", lambda dev: state.registered)
    monkeypatch.setattr(context_mod, "_torch_capturing", lambda dev: state.torch)
    monkeypatch.setattr(host_mod, "_slot_poll_buffer", lambda ctx: state.buffer)
    return state


def test_default_lowering_per_capture_state(quiet_probes):
    mode = context_mod._bounded_mode
    assert mode(_FakeCtx(cuda=False)) == "poll"              # (a): CPU eager
    assert mode(_FakeCtx(cuda=True)) == "graph"              # CUDA eager: (c), no sync
    assert mode(_FakeCtx(cuda=True, capturing=True)) == "graph"
    quiet_probes.cpu = True
    assert mode(_FakeCtx(cuda=False)) == "graph"             # CPU graph capture: (c)


def test_forced_lowerings(quiet_probes):
    mode = context_mod._bounded_mode
    with _force_lowering("unroll"):
        assert mode(_FakeCtx(cuda=False)) == "unroll"
        assert mode(_FakeCtx(cuda=True, capturing=True)) == "unroll"
    with _force_lowering("poll"):
        assert mode(_FakeCtx(cuda=False)) == "poll"
        with pytest.raises(CaptureError, match="poll buffer of a Program slot"):
            mode(_FakeCtx(cuda=True))
        quiet_probes.buffer = object()
        assert mode(_FakeCtx(cuda=True)) == "poll"
        for flip in ("capturing", "registered", "torch", "cpu"):
            ctx = _FakeCtx(cuda=flip != "cpu", capturing=flip == "capturing")
            quiet_probes.registered = flip == "registered"
            quiet_probes.torch = object() if flip == "torch" else None
            quiet_probes.cpu = flip == "cpu"
            with pytest.raises(CaptureError, match="capture is open"):
                mode(ctx)
    with pytest.raises(ValueError, match="lowering must be one of"):
        with _force_lowering("conditional"):
            pass


def test_cuda_eager_windows_run_inline_without_a_poll(quiet_probes, monkeypatch, polls):
    """The window driver in "graph" mode on an eager context: every window runs through
    skip_unless, which runs it inline (no poll, no capture_if)."""
    seen: list = []

    class _Ctx:
        def skip_unless(self, flag, body):
            seen.append("skip_unless")
            body(self)
            return False

    body = _Loop()
    lowering, windows, ran = context_mod._run_windows(_Ctx(), "graph", body, _flag(0), 10,
                                                      4, 3)
    assert (lowering, windows, ran) == ("unroll", 2, 2)
    assert body.its == list(range(4, 10)) and polls == [] and seen == ["skip_unless"] * 2


def test_registered_capture_records_one_capture_if_per_window(monkeypatch):
    """Inside a Warp-registered capture with conditional nodes each window is recorded
    under ``wp.capture_if(active, stream=<registered key>)`` (the bodies inside
    ``on_true``); without conditional support, or when not allowed, windows unroll."""
    key = object()
    calls: list = []

    class _Capturing(LaunchContext):
        __slots__ = ()

        @property
        def capturing(self):
            return True

    def capture_if(flag, on_true=None, stream=None):
        calls.append(("capture_if", stream))
        on_true()

    monkeypatch.setattr(_compat, "cpu_capture_active", lambda dev: True)   # "graph"
    monkeypatch.setattr(_compat, "conditional_ok", lambda dev: True)
    monkeypatch.setattr(_compat, "registered_stream", lambda stream: key)
    monkeypatch.setattr(_compat, "conditional_allowed", lambda k: k is key)
    monkeypatch.setattr(wp, "capture_if", capture_if)
    ctx = _Capturing(DEV, None)
    body = _Loop()
    with _probe_loops() as recs:
        ctx.bounded_loop(10, lambda c, it: (calls.append(it), body(c, it)),
                         active=_flag(1), first_check=4, check_every=3)
    assert calls == [0, 1, 2, 3, ("capture_if", key), 4, 5, 6, ("capture_if", key), 7, 8, 9]
    assert (recs[0].lowering, recs[0].windows, recs[0].ran) == ("conditional", 2, None)
    monkeypatch.setattr(_compat, "conditional_ok", lambda dev: False)
    del calls[:]
    with _probe_loops() as recs:
        ctx.bounded_loop(10, lambda c, it: calls.append(it), active=_flag(1),
                         first_check=4, check_every=3)
    assert calls == list(range(10)) and recs[0].lowering == "unroll"


def test_a_cpu_graph_capture_never_polls(polls):
    """Warp records every CPU launch of a ``wp.ScopedCapture("cpu")`` without running it
    (a poll would read a stale flag): the loop is recorded unrolled; its replay equals
    the eager run."""
    assert not _compat.cpu_capture_active(DEV)
    out = wp.zeros(1, dtype=wp.int32, device=DEV)
    active = _flag(1)

    @wp.kernel
    def _count_k(out: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
                 it: int):
        if active[0] != 0:
            out[0] = out[0] + 1
            if it == 5:
                active[0] = 0

    def body(ctx, it):
        ctx.launch(_count_k, 1, [out, active, it])

    def run(ctx):
        ctx.fill(out, 0)
        ctx.fill(active, 1)
        ctx.bounded_loop(16, body, active=active, first_check=4, check_every=4)

    run(_ctx())
    eager = out.numpy().copy()
    assert polls == [1, 0] and eager.tolist() == [6]
    del polls[:]
    with wp.ScopedCapture(device=DEV) as cap:
        assert _compat.cpu_capture_active(DEV) and _compat.cpu_capture_active("cpu")
        with _probe_loops() as recs:
            run(_ctx())
    assert not _compat.cpu_capture_active(DEV)
    assert polls == [] and recs[0].lowering == "unroll" and recs[0].ran == 3
    out.fill_(-1)
    wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(out.numpy(), eager)


def test_prepare_forces_the_unrolled_lowering():
    seen: list = []

    def body(ctx):
        seen.append(context_mod._LOWERING.get())

    r = GraphRunner(DEV, capturable=True)
    r.prepare(body)
    r.run(body)
    assert seen == ["unroll", None]


# ---------------------------------------------------------------------------------------
# The BOUNDED declaration and the loop probe inside a Program
# ---------------------------------------------------------------------------------------

def _spec(relax, n=E):
    return presets.track(num_envs=n, device=DEV, relax=relax)


def _gen(relax, seeds=11, n=E):
    return TrackGenerator(_spec(relax, n), PerEnvSeededRNG(seeds=seeds, num_envs=n,
                                                           device=DEV))


def test_a_static_stage_passing_active_is_refused_at_its_first_run():
    with pytest.raises(ValueError, match=r"relax 'toy.undeclared_loop' "
                                         r"\(tests._bounded_toys.UndeclaredLoop\) calls "
                                         r"ctx.bounded_loop\(active=...\) but declares "
                                         r"capture=CaptureClass.STATIC: declare capture = "
                                         r"CaptureClass.BOUNDED"):
        _gen(T.UNDECLARED)
    report = check_stage("relax", T.UNDECLARED, devices=[DEV])
    assert report.failed_names == ["construct"]
    assert "declare capture = CaptureClass.BOUNDED" in report.result("construct").message


class _StaticLoop(Stage):
    """A STATIC relax with a plain unrolled loop (no ``active``): fine."""

    kind = "relax"
    name = "test.static_loop"
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        x_in, out = b.read("xy"), b.write("xy", wp.vec2f, per="cl")

        def enqueue(ctx):
            ctx.copy(out, x_in)
            ctx.bounded_loop(3, lambda c, it: None)
        return enqueue


def test_a_static_stage_may_run_a_loop_without_active():
    registry.add(_StaticLoop)
    gen = _gen("test.static_loop")
    assert gen.program.capture_class is CaptureClass.STATIC


def test_the_probe_names_the_slot_running_the_loop():
    gen = _gen(T.CONVERGING)
    with _probe_loops() as recs:
        gen.generate()
    (rec,) = recs
    assert rec.stage_id == "relax" and rec.lowering == "poll"
    assert (rec.n_iters, rec.first_check, rec.check_every, rec.windows) == (64, 8, 8, 7)
    assert rec.ran < rec.windows                       # the kit's batch converges early
    assert gen.program.capture_class is CaptureClass.BOUNDED and gen.program.capturable


# A legacy GeneratorSpec (it cannot declare a capture class) running a bounded loop: not
# checked at build, probed by the kit (applicability comes from what the unit does).

@wp.kernel
def _grow_init_k(seeds: wp.array(dtype=wp.int32), n: int, out: wp.array(dtype=wp.vec2f),
                 steps: wp.array(dtype=wp.int32), target: wp.array(dtype=wp.int32),
                 valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    t = 6.2831853 * float(i) / float(n)
    out[e * n + i] = wp.vec2f(0.4 * wp.cos(t), 0.4 * wp.sin(t))
    if i == 0:
        state = wp.rand_init(seeds[e])
        steps[e] = 0
        target[e] = wp.randi(state, 2, 9)
        valid[e] = 1


@wp.kernel
def _grow_step_k(n: int, out: wp.array(dtype=wp.vec2f), steps: wp.array(dtype=wp.int32),
                 target: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    if steps[e] >= target[e]:
        return
    out[e * n + i] = out[e * n + i] * 1.02


@wp.kernel
def _grow_count_k(steps: wp.array(dtype=wp.int32), target: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if steps[e] < target[e]:
        steps[e] = steps[e] + 1


@wp.kernel
def _grow_active_k(steps: wp.array(dtype=wp.int32), target: wp.array(dtype=wp.int32),
                   active: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if steps[e] < target[e]:
        wp.atomic_max(active, 0, 1)


def _grow_alloc(config):
    E_, N = int(config.num_envs), int(config.num_points)
    dev = str(config.device)
    return types.SimpleNamespace(steps=wp.zeros(E_, dtype=wp.int32, device=dev),
                                 target=wp.zeros(E_, dtype=wp.int32, device=dev),
                                 active=wp.zeros(1, dtype=wp.int32, device=dev), n=N, E=E_)


def _grow_generate(seeds, config, out_centerline, out_valid_wp, scratch, ctx=None):
    s = scratch

    def body(c, it):
        c.launch(_grow_step_k, (s.E, s.n), [s.n, out_centerline, s.steps, s.target])
        c.launch(_grow_count_k, s.E, [s.steps, s.target])
        if (it + 1) % 4 == 0:
            ops.fill(c, s.active, 0)
            c.launch(_grow_active_k, s.E, [s.steps, s.target, s.active])

    ctx.launch(_grow_init_k, (s.E, s.n), [seeds, s.n, out_centerline, s.steps, s.target,
                                         out_valid_wp])
    ctx.bounded_loop(16, body, active=s.active, first_check=4, check_every=4)


def test_a_legacy_spec_bounded_loop_is_unchecked_and_probed():
    registry.add(GeneratorSpec(name="test.legacy_grow", alloc_scratch=_grow_alloc,
                               generate=_grow_generate, accepts_ctx=True))
    gen = TrackGenerator(TrackGenConfig(generator="test.legacy_grow", device=DEV,
                                        num_envs=E, num_points=64),
                         PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV))
    with _probe_loops() as recs:
        gen.generate()
    assert [r.stage_id for r in recs] == ["generator"] and recs[0].ran < recs[0].windows
    report = check_generator("test.legacy_grow", devices=[DEV], config={"num_points": 64},
                             skip=("diversity",))
    absorbing = report.result("bounded.absorbing")
    assert absorbing.passed and not absorbing.skipped, absorbing
    assert absorbing.details["skipped"]["A"] >= 1


# ---------------------------------------------------------------------------------------
# ops.all_set
# ---------------------------------------------------------------------------------------

def _i32(values):
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


@pytest.mark.parametrize("mask,want", [([1, 2, -1, 5], 1), ([0, 0, 0], 0), ([1, 0, 1], 0),
                                       ([3], 1), ([0], 0), ([], 1)])
def test_all_set(mask, want):
    flag = _flag(7)                                   # stale: all_set writes it first
    ops.all_set(_ctx(), _i32(mask), flag)
    assert flag.numpy().tolist() == [want]


def test_all_set_errors_mirror_any_set():
    flag = _flag()
    for fn in (ops.any_set, ops.all_set):
        with pytest.raises(TypeError, match="mask must have dtype int32"):
            fn(_ctx(), wp.zeros(3, dtype=wp.float32, device=DEV), flag)
        with pytest.raises(TypeError, match="mask must be a wp.array"):
            fn(_ctx(), np.zeros(3, np.int32), flag)
        with pytest.raises(ValueError, match=r"flag_out must have shape \(1,\)"):
            fn(_ctx(), _i32([1, 0]), wp.zeros(2, dtype=wp.int32, device=DEV))


# ---------------------------------------------------------------------------------------
# The no-host-sync contract and the kit
# ---------------------------------------------------------------------------------------

def test_the_host_poll_is_the_one_sanctioned_readback():
    gen = _gen(T.CONVERGING)
    with no_host_sync() as guard:
        gen.generate()
    assert guard.ok and guard.allowed == []
    assert guard.sanctioned and set(guard.sanctioned) == {
        ("wp.array.numpy", "track_gen.exec.host", "_poll_i32")}
    with _force_lowering("unroll"), no_host_sync() as guard:
        gen.generate()
    assert guard.ok and guard.sanctioned == []


def test_a_plugin_sync_inside_a_loop_body_is_caught():
    gen = _gen(T.SYNCING)
    guard = HotPathGuard(host_sync=True, allocation=False, stream=False, capture=False,
                         strict=False)
    with guard:
        gen.generate()
    found = guard.of("host_sync")
    assert found and {v.site.module for v in found} == {"tests._bounded_toys"}
    assert guard.sanctioned                   # the library's own polls still pass


def test_a_direct_call_of_the_poll_is_not_sanctioned():
    gen = _gen(T.POLL_CALLER)
    guard = HotPathGuard(host_sync=True, allocation=False, stream=False, capture=False,
                         strict=False)
    with guard:
        gen.generate()
    assert [v.key for v in guard.of("host_sync")] == [
        ("wp.array.numpy", "track_gen.exec.host", "_poll_i32")]
    assert guard.sanctioned == []


def test_kit_hot_path_unrolls_the_loops():
    """A sync / an allocation emitted only in the last iteration, which the CPU host poll
    never reaches for the kit's seeds, is still caught (the guarded calls unroll)."""
    report = check_stage("relax", T.LATE_ALLOC, devices=[DEV])
    assert report.failed_names == ["hot_path.allocation"], str(report)
    assert "tests._bounded_toys.LateAllocSmooth._late" in \
        report.result("hot_path.allocation").message
    report = check_stage("relax", T.LATE_SYNC, devices=[DEV])
    assert report.failed_names == ["hot_path.host_sync"], str(report)
    report = check_stage("relax", T.SYNCING, devices=[DEV])
    assert report.failed_names == ["hot_path.host_sync"], str(report)
    ok = check_stage("relax", T.CONVERGING, devices=[DEV])
    ok.raise_for_failures()
    assert ok.result("hot_path.host_sync").details["sanctioned"] > 0
    assert not ok.result("bounded.absorbing").skipped


# ---------------------------------------------------------------------------------------
# HostOps and StageBuilder.host_buffer
# ---------------------------------------------------------------------------------------

def _host_slot(capture=CaptureClass.HOST):
    return host_mod._SlotInfo("relax", "relax 'test.h' (tests.H)", capture, "cpu",
                              wp.zeros(1, dtype=wp.int32, device="cpu"))


def _in_slot(info, fn, ctx=None):
    out: list = []
    host_mod._SlotEnqueue(info, lambda c: out.append(fn(c)))(ctx or _ctx())
    return out[0]


def test_host_ops_on_cpu():
    src = _i32([4, 5, 6, 7])
    via = wp.zeros(6, dtype=wp.int32, device="cpu")
    pts = wp.zeros(3, dtype=wp.vec2f, device=DEV)
    via2 = wp.zeros(4, dtype=wp.vec2f, device="cpu")

    def fn(ctx):
        h = ctx.host()
        assert isinstance(h, HostOps) and h.stage_id == "relax"
        h.sync()                                          # CPU: a no-op
        got = [h.read_i32(src), h.read_i32(src, 3)]
        view = h.readback(src, via=via)
        assert view.tolist() == [4, 5, 6, 7] and np.shares_memory(view, via.numpy())
        h.upload(pts, [[1, 2], [3, 4], [5, 6]], via=via2)
        back = h.readback(pts, via=via2)
        assert back.shape == (3, 2) and back.tolist() == [[1, 2], [3, 4], [5, 6]]
        for call, exc, match in [
            (lambda: h.read_i32(pts), TypeError, "arr must have dtype int32"),
            (lambda: h.read_i32(src, 4), ValueError, "index 4 is out of range"),
            (lambda: h.read_i32(src, True), TypeError, "index must be an int"),
            (lambda: h.readback(src, via=wp.zeros(2, dtype=wp.int32, device="cpu")),
             ValueError, "via holds 2 elements, 4 needed"),
            (lambda: h.readback(src, via=via2), TypeError, "via has dtype vec2f"),
            (lambda: h.readback(src, via=np.zeros(4, np.int32)), TypeError,
             "via must be a wp.array"),
            (lambda: h.upload(pts, [1.0, 2.0], via=via2), ValueError,
             "data holds 2 values; dst needs 3 vec2f"),
            (lambda: h.upload(np.zeros(3), [1], via=via2), TypeError,
             "dst must be a wp.array"),
        ]:
            with pytest.raises(exc, match=match):
                call()
        return got

    assert _in_slot(_host_slot(), fn) == [4, 7]
    np.testing.assert_array_equal(pts.numpy(), [[1, 2], [3, 4], [5, 6]])


def test_ctx_host_outside_a_host_slot_raises():
    with pytest.raises(CaptureError, match="only inside a HOST stage's enqueue"):
        _ctx().host()
    # Inside a stage that declares another class: the declared-class error (ValueError,
    # as bounded_loop(active=) in a STATIC stage and host_buffer), naming the stage.
    for cap in (CaptureClass.STATIC, CaptureClass.BOUNDED):
        with pytest.raises(ValueError, match=rf"relax 'test.h' \(tests.H\) calls "
                                             rf"ctx.host\(\) but declares capture="
                                             rf"CaptureClass.{cap.name}: declare capture = "
                                             rf"CaptureClass.HOST"):
            _in_slot(_host_slot(cap), lambda c: c.host())
    with pytest.raises(ValueError, match=r"which a legacy GeneratorSpec cannot use"):
        _in_slot(_host_slot(None), lambda c: c.host())


def _skip_without_pinned_memory() -> None:
    """These tests fake a CUDA device but allocate real pinned host memory, which needs a
    CUDA driver: skip on a CPU-only host whose Warp cannot pin (Warp on macOS falls back
    to pageable memory and runs them)."""
    try:
        wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)
    except RuntimeError as exc:
        pytest.skip(f"pinned host memory is unavailable here ({exc})")


def test_host_ops_require_a_pinned_via_on_cuda(monkeypatch):
    _skip_without_pinned_memory()
    """On CUDA a ``via`` must be pinned (a ``StageBuilder.host_buffer``): a pageable
    host array (e.g. allocated per call) is refused."""
    from tests._fake_cuda import FakeCuda

    fake = FakeCuda(monkeypatch)
    ops_ = HostOps(LaunchContext("cuda:0", fake.current), _host_slot())
    with pytest.raises(ValueError, match="via must be a pinned host buffer"):
        ops_._via(wp.zeros(4, dtype=wp.int32, device="cpu"), wp.int32, 4)
    ops_._via(wp.zeros(4, dtype=wp.int32, device="cpu", pinned=True), wp.int32, 4)
    pinned = wp.zeros(8, dtype=wp.int32, device="cpu", pinned=True)
    ops_._via(pinned[2:6], wp.int32, 4)                     # a view of one is fine
    # On CPU any host array is accepted (copies are synchronous there).
    HostOps(_ctx(), _host_slot())._via(wp.zeros(4, dtype=wp.int32, device="cpu"),
                                       wp.int32, 4)


def test_only_host_slots_get_a_build_time_host_buffer(monkeypatch):
    """A HOST slot gets its ``[1]`` buffer at build; a BOUNDED slot none (production never
    polls on CUDA): the forced CUDA poll allocates a pinned one on first use, once."""
    _skip_without_pinned_memory()
    gen = _gen(T.CONVERGING)
    infos = {e.info.stage_id: e.info for e in gen.program._enqueues}
    assert all(i.i32 is None for i in infos.values())
    host = TrackGenerator(presets.track(num_envs=E, device=DEV, relax=T.HOST_IDENTITY),
                          PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    infos = {e.info.stage_id: e.info for e in host.program._enqueues}
    assert infos["relax"].i32 is not None and infos["generator"].i32 is None
    from tests._fake_cuda import FakeCuda

    fake = FakeCuda(monkeypatch)
    slot = host_mod._SlotInfo("relax", "relax 'test.b' (tests.B)", CaptureClass.BOUNDED,
                              "cuda:0")
    ctx = LaunchContext("cuda:0", fake.current)
    got = []
    host_mod._SlotEnqueue(slot, lambda c: got.extend(
        [host_mod._slot_poll_buffer(c), host_mod._slot_poll_buffer(c)]))(ctx)
    assert got[0] is got[1] and got[0].pinned and tuple(got[0].shape) == (1,)
    assert host_mod._slot_poll_buffer(ctx) is None             # outside a slot


def test_ctx_host_refuses_a_capturing_context(monkeypatch):
    class _Capturing(LaunchContext):
        __slots__ = ()

        @property
        def capturing(self):
            return True

    with pytest.raises(NotCapturableError, match=r"HOST stage\(s\) \['relax'\]") as info:
        _in_slot(_host_slot(), lambda c: c.host(), ctx=_Capturing(DEV, None))
    assert info.value.stage_ids == ("relax",)
    with wp.ScopedCapture(device=DEV):
        with pytest.raises(NotCapturableError, match="relax 'test.h'"):
            _in_slot(_host_slot(), lambda c: c.host())


class _StaticHostBuffer(Stage):
    kind = "relax"
    name = "test.static_host_buffer"
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        b.host_buffer(wp.int32, 4)
        return lambda ctx: None


def test_host_buffer_is_for_host_stages_at_build_only():
    registry.add(_StaticHostBuffer)
    with pytest.raises(ValueError, match=r"StageBuilder.host_buffer\(\) is for HOST "
                                         r"stages: stage 'relax' \(_StaticHostBuffer\) has "
                                         r"capture class STATIC"):
        _gen("test.static_host_buffer")
    kept: list = []
    bufs: list = []

    class _Late(T.HostIdentity):
        name = "test.late_host_buffer"

        def build(self, b, p):
            kept.append(b)
            real = b.host_buffer
            b.host_buffer = lambda *a: bufs.append(real(*a)) or bufs[-1]
            return super().build(b, p)

    registry.add(_Late)
    gen = _gen("test.late_host_buffer")
    (buf,) = bufs
    assert buf.device.is_cpu and buf.dtype is wp.vec2f
    assert kept[0]._capture is CaptureClass.HOST           # the planner's resolved class
    with pytest.raises(RuntimeError, match=r"StageBuilder.host_buffer\(\) after build"):
        kept[0].host_buffer(wp.int32, 1)
    # The kit's direct call of a stage's enqueue (stage.isolation) still runs it in its
    # slot: a HOST stage gets its HostOps.
    i = gen.program._stage_ids.index("relax")
    gen.program._enqueues[i](_ctx())
