"""Caller-capturable generation: execution model checks that run without a GPU (PR3).

``TrackGenerator`` / ``GateGenerator`` / ``Course`` route every generation through one
enqueue body and :class:`track_gen.exec.GraphRunner`, which picks per call:

- RECORD into the caller's capture when the resolved stream is capturing (never a
  self-capture, never ``wp.capture_launch``, never a sync);
- EAGER on CPU, for ``policy="eager"`` (``enqueue()``) and for non-capturable generators;
- SELF-GRAPH on CUDA otherwise: capture once on a private stream, replay on the call's.

Construction runs the body once for real (``prepare``). The policy is checked here on a
fake CUDA device (``tests/_fake_cuda.py``); the real CPU paths are checked for counts,
results, allocation-freedom and the config snapshot. The GPU side (real captures,
poisoned replays, torch adoption) lives in ``tests/test_caller_capture_cuda.py``.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import sys
import threading
import types
import weakref

import numpy as np
import pytest
import warp as wp

from tests._fake_cuda import FakeCuda, FakeGraph, FakeStream, patch_facade_seeds
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import course as course_mod
from track_gen._src import gate_generator_registry, generator_registry, runtime, warp_gate, warp_pipeline
from track_gen.course import Course, CourseConfig
from track_gen.exec import (
    CaptureError,
    GraphRunner,
    LaunchContext,
    LegacyPluginStreamError,
    NotCapturableError,
    _compat,
    resolve_stream,
)
from track_gen.exec import context as context_mod
from track_gen.exec import runner as runner_mod

E = 4


@pytest.fixture(autouse=True)
def _fresh_legacy_registry(monkeypatch):
    """Isolate the pending-legacy-capture registry (a legacy runner kept alive by an
    earlier test's traceback must not make this test's calls take the lock)."""
    monkeypatch.setattr(runner_mod, "_PENDING_LEGACY", weakref.WeakSet())


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

class _Body:
    """A fake enqueue body recording the context of every run."""

    def __init__(self, fail: int = 0, exc=RuntimeError) -> None:
        self.ctxs: list = []
        self.capturing: list = []
        self._fail, self._exc = fail, exc

    def __call__(self, ctx) -> None:
        self.ctxs.append(ctx)
        self.capturing.append(ctx.capturing)
        # nested ctx=None helpers inherit the call's context
        assert LaunchContext.for_call(ctx.device) is ctx
        if self._fail:
            self._fail -= 1
            raise self._exc("simulated capture invalidation")

    @property
    def runs(self) -> int:
        return len(self.ctxs)


class _Owner:
    pass


def _flat(arr) -> np.ndarray:
    return np.nan_to_num(np.asarray(arr.numpy(), dtype=np.float64).reshape(-1))


def _track_cfg(**kw) -> TrackGenConfig:
    kw.setdefault("num_envs", E)
    return TrackGenConfig(device="cpu", **kw)


def _gate_cfg(**kw) -> GateGenConfig:
    kw.setdefault("num_envs", E)
    kw.setdefault("gate_width", 0.2)
    return GateGenConfig(device="cpu", **kw)


def _rng(seed=3, n=E, device="cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=device)


_TRACK_FIELDS = ("center", "outer", "inner", "tangent", "normal", "arclen", "length",
                 "valid", "count")
_GATE_FIELDS = ("position", "tangent", "orientation", "left", "right", "valid", "count")


def _assert_same(a, b, fields) -> None:
    for f in fields:
        np.testing.assert_array_equal(getattr(a, f).numpy(), getattr(b, f).numpy(),
                                      err_msg=f)


def _counting(monkeypatch, registry_mod, table: str, name: str, **replace):
    """Replace a registered spec by a copy whose generate() counts its calls."""
    base = registry_mod.get(name)
    calls: list = []

    def generate(*args, **kwargs):
        calls.append(1)
        return base.generate(*args, **kwargs)

    spec = dataclasses.replace(base, generate=generate, **replace)
    monkeypatch.setitem(getattr(registry_mod, table), name, spec)
    return calls


@pytest.fixture
def no_alloc(monkeypatch):
    """Tripwire: record every OWNING wp.array construction (views pass ``ptr=``)."""
    hits: list = []
    real_init = wp.array.__init__
    armed = [False]

    def init(self, *args, **kwargs):
        if armed[0] and kwargs.get("ptr") is None and len(args) < 6:
            hits.append((args[:1], {k: v for k, v in kwargs.items() if k != "data"}))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(wp.array, "__init__", init)

    class _Ctl:
        def __enter__(self):
            armed[0] = True
            return hits

        def __exit__(self, *exc):
            armed[0] = False
            return False

    return _Ctl()


def test_alloc_tripwire_sees_allocations_but_not_views(no_alloc):
    base = wp.zeros(8, dtype=wp.int32, device="cpu")
    with no_alloc as hits:
        base.reshape((2, 4))
        base[2:]
        assert hits == []
        wp.empty(3, dtype=wp.int32, device="cpu")
        wp.array(np.arange(3), dtype=wp.int32, device="cpu")
    assert len(hits) == 2


# ---------------------------------------------------------------------------------------
# GraphRunner on the CPU device
# ---------------------------------------------------------------------------------------

def test_runner_cpu_is_always_eager():
    body = _Body()
    r = GraphRunner("cpu", capturable=True, name="toy")
    r.prepare(body)
    assert r.prepared and body.runs == 1
    for policy in ("auto", "eager"):
        ctx = r.run(body, policy=policy)
        assert ctx.stream is None and not ctx.capturing
    assert body.runs == 3
    assert r.graph is None
    with pytest.raises(ValueError, match="policy"):
        r.run(body, policy="graph")
    assert body.runs == 3


def test_runner_rejects_bad_default_stream(monkeypatch):
    with pytest.raises(TypeError):
        GraphRunner("cpu", capturable=True, default_stream="torch")
    fake = FakeCuda(monkeypatch)
    other = FakeStream(type(fake.device)("cuda:1"))
    with pytest.raises(ValueError, match="cuda:1"):
        GraphRunner("cuda:0", capturable=True, default_stream=other)


def test_runner_holds_its_owner_weakly():
    owner = _Owner()
    r = GraphRunner("cpu", capturable=True, owner=owner)
    ref = weakref.ref(owner)
    assert r._owner() is owner
    del owner
    gc.collect()
    assert ref() is None and r._owner() is None


# ---------------------------------------------------------------------------------------
# GraphRunner policy on a fake CUDA device
# ---------------------------------------------------------------------------------------

def test_prepare_refuses_a_capturing_stream(monkeypatch):
    fake = FakeCuda(monkeypatch)
    fake.current.is_capturing = True
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True)
    with pytest.raises(CaptureError, match="outside any CUDA graph capture"):
        r.prepare(body)
    assert body.runs == 0 and not r.prepared


def test_self_graph_captures_once_on_a_private_stream_and_replays(monkeypatch):
    fake = FakeCuda(monkeypatch)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True, name="toy")
    r.prepare(body)
    assert body.runs == 1 and body.ctxs[0].stream is fake.current
    assert fake.current.waited_on == []        # already the allocation stream
    assert r.graph is None and fake.captures == []

    ctx = r.run(body)
    assert body.runs == 2                      # recorded once, not executed
    assert body.capturing == [False, True]
    (cap,) = fake.captures
    private = cap.stream
    assert private is not fake.current and fake.created == [private]
    assert body.ctxs[1].stream is private
    assert cap.force_module_load is False
    assert isinstance(r.graph, FakeGraph) and r.graph.stream is private
    assert fake.launched == [(r.graph, fake.current)]   # replayed on the call's stream
    assert ctx.stream is fake.current and not ctx.capturing

    r.run(body)
    r.run(body)
    assert body.runs == 2                      # replays never re-run the body
    assert len(fake.captures) == 1 and fake.created == [private]
    assert fake.launched == [(r.graph, fake.current)] * 3


def test_self_graph_replays_on_the_resolved_stream(monkeypatch):
    fake = FakeCuda(monkeypatch)
    default = FakeStream(fake.device)
    explicit = FakeStream(fake.device)
    r = GraphRunner("cuda:0", capturable=True, default_stream=default)
    body = _Body()
    r.prepare(body)
    assert body.ctxs[0].stream is default
    # construction allocated on Warp's current stream: the warm-up pass waits on it
    assert default.waited_on == [fake.current]
    r.run(body)
    r.run(body, stream=explicit)
    assert [s for _, s in fake.launched] == [default, explicit]
    assert default.waited_on == [fake.current] and explicit.waited_on == []


def test_self_capture_without_prepare_prepares_first(monkeypatch):
    FakeCuda(monkeypatch)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True)
    r.run(body)
    assert r.prepared and body.capturing == [False, True]


def test_record_into_a_caller_capture(monkeypatch):
    fake = FakeCuda(monkeypatch)
    owner = _Owner()
    graph = types.SimpleNamespace()
    monkeypatch.setattr(_compat, "registered_graph",
                        lambda s: graph if s is fake.current else None)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True, owner=owner, name="toy")
    r.prepare(body)
    fake.current.is_capturing = True           # the caller is capturing this stream
    for policy in ("auto", "eager"):
        ctx = r.run(body, policy=policy)
        assert ctx.capturing and ctx.stream is fake.current
    assert body.runs == 3 and body.capturing[1:] == [True, True]
    assert fake.captures == [] and fake.launched == [] and r.graph is None
    # the producer is anchored on the caller's graph, once
    assert graph._track_gen_owners == [owner]


def test_non_capturable_raises_before_any_launch_and_runs_eagerly_outside(monkeypatch):
    fake = FakeCuda(monkeypatch)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=False, name="toy-host")
    r.prepare(body)
    r.run(body)
    assert body.runs == 2 and fake.captures == [] and fake.launched == []
    assert r.graph is None
    fake.current.is_capturing = True
    for policy in ("auto", "eager"):
        with pytest.raises(NotCapturableError, match="toy-host"):
            r.run(body, policy=policy)
    assert body.runs == 2


def test_legacy_plugin_records_only_on_the_current_stream(monkeypatch):
    fake = FakeCuda(monkeypatch)
    body = _Body()
    side = FakeStream(fake.device)
    r = GraphRunner("cuda:0", capturable=True, legacy_stream=True, name="legacy")
    r.prepare(body)
    side.is_capturing = True
    with pytest.raises(LegacyPluginStreamError, match="ScopedStream"):
        r.run(body, stream=side)
    assert body.runs == 1
    fake.current.is_capturing = True
    r.run(body)                                # capture stream == current stream: fine
    assert body.runs == 2 and body.ctxs[-1].stream is fake.current


def _lock_free_in_another_thread() -> bool:
    """Whether another thread can take ``runtime._CAPTURE_LOCK`` right now."""
    out: list = []

    def probe():
        got = runtime._CAPTURE_LOCK.acquire(blocking=False)
        if got:
            runtime._CAPTURE_LOCK.release()
        out.append(got)

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return out[0]


def test_legacy_plugin_self_captures_on_the_current_stream_under_the_lock(monkeypatch):
    fake = FakeCuda(monkeypatch)
    held = []

    def body(ctx):
        if ctx.capturing:
            out = [_lock_free_in_another_thread()]
            held.append(not out[0])

    r = GraphRunner("cuda:0", capturable=True, legacy_stream=True)
    r.prepare(body)
    r.run(body)
    (cap,) = fake.captures
    assert cap.stream is fake.current and fake.created == []
    assert held == [True]                      # serialized process-wide while recording


def test_ctx_aware_self_capture_takes_no_global_lock(monkeypatch):
    FakeCuda(monkeypatch)
    free = []

    def body(ctx):
        if ctx.capturing:
            out = [_lock_free_in_another_thread()]
            free.append(out[0])

    r = GraphRunner("cuda:0", capturable=True)
    r.prepare(body)
    r.run(body)
    assert free == [True]


def test_failed_self_capture_is_retried_once(monkeypatch):
    fake = FakeCuda(monkeypatch)
    r = GraphRunner("cuda:0", capturable=True)
    r.prepare(_Body())
    body = _Body(fail=1)
    r.run(body)
    assert body.runs == 2 and len(fake.captures) == 2
    assert r.graph is fake.captures[-1].graph
    assert fake.launched == [(r.graph, fake.current)]

    fake2 = FakeCuda(monkeypatch)
    r2 = GraphRunner("cuda:0", capturable=True)
    r2.prepare(_Body())
    with pytest.raises(RuntimeError, match="simulated"):
        r2.run(_Body(fail=2))
    assert r2.graph is None and len(fake2.captures) == 2 and fake2.launched == []

    fake3 = FakeCuda(monkeypatch)
    r3 = GraphRunner("cuda:0", capturable=True)
    r3.prepare(_Body())
    with pytest.raises(CaptureError):
        r3.run(_Body(fail=1, exc=CaptureError))     # library errors are not retried
    assert len(fake3.captures) == 1


def test_eager_policy_never_self_graphs(monkeypatch):
    fake = FakeCuda(monkeypatch)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True)
    r.prepare(body)
    r.run(body, policy="eager")
    assert body.runs == 2 and body.capturing == [False, False]
    assert fake.captures == [] and fake.launched == [] and r.graph is None


def test_prepare_blocks_on_the_resolved_stream(monkeypatch):
    fake = FakeCuda(monkeypatch)
    side = FakeStream(fake.device)
    GraphRunner("cuda:0", capturable=True).prepare(_Body())
    GraphRunner("cuda:0", capturable=True, default_stream=side).prepare(_Body())
    # construction never leaves its warm-up in flight for a later call on another stream
    assert fake.synced == [fake.current, side]


def test_run_syncs_only_when_asked_and_never_while_recording(monkeypatch):
    fake = FakeCuda(monkeypatch)
    r = GraphRunner("cuda:0", capturable=True)
    r.prepare(_Body())
    del fake.synced[:]
    r.run(_Body())
    r.run(_Body(), policy="eager")
    assert fake.synced == []
    r.run(_Body(), sync=True)
    r.run(_Body(), policy="eager", sync=True)
    assert fake.synced == [fake.current] * 2
    fake.current.is_capturing = True
    r.run(_Body(), sync=True)
    assert fake.synced == [fake.current] * 2


def _holds_lock(cm) -> bool:
    return cm is runtime._CAPTURE_LOCK


def test_serial_locks_current_stream_calls_only_while_a_legacy_capture_is_pending(monkeypatch):
    fake = FakeCuda(monkeypatch)
    side = FakeStream(fake.device)
    r = GraphRunner("cuda:0", capturable=True)
    on_cur, on_side = r.context(), r.context(side)
    assert not _holds_lock(r.serial(on_cur))             # no legacy producer: lock-free
    assert not _holds_lock(GraphRunner("cpu", capturable=True, legacy_stream=True)
                           .serial(LaunchContext("cpu", None)))
    legacy = GraphRunner("cuda:0", capturable=True, legacy_stream=True)
    assert _holds_lock(legacy.serial(on_side))           # a legacy producer: always
    # it may self-capture on Warp's current stream: calls resolving there serialize
    assert _holds_lock(r.serial(on_cur)) and not _holds_lock(r.serial(on_side))
    legacy.prepare(_Body())
    legacy.run(_Body())
    assert legacy.graph is not None                      # captured once: nothing pending
    assert not _holds_lock(r.serial(on_cur))
    GraphRunner("cuda:0", capturable=False, legacy_stream=True)   # never self-captures
    assert not _holds_lock(r.serial(on_cur))
    pending = GraphRunner("cuda:0", capturable=True, legacy_stream=True)
    assert _holds_lock(r.serial(on_cur))
    del pending
    gc.collect()
    assert not _holds_lock(r.serial(on_cur))             # a dead runner never captures


def test_current_stream_call_waits_for_a_legacy_self_capture(monkeypatch):
    """A legacy producer self-captures Warp's shared current stream; another thread's
    call resolving that stream must not mistake it for its caller's capture (it would
    record itself into the legacy graph and return without running)."""
    fake = FakeCuda(monkeypatch)
    in_capture, release = threading.Event(), threading.Event()

    def legacy_body(ctx):
        if ctx.capturing:
            in_capture.set()
            assert release.wait(10)

    legacy = GraphRunner("cuda:0", capturable=True, legacy_stream=True, name="legacy")
    legacy.prepare(legacy_body)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True, name="toy")
    r.prepare(body)
    errors: list = []

    def call(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    t_legacy = threading.Thread(target=call, args=(legacy.run, legacy_body))
    t_legacy.start()
    try:
        assert in_capture.wait(10)
        assert fake.current.is_capturing                 # the legacy capture is open
        t_other = threading.Thread(target=call, args=(r.run, body), kwargs={"sync": True})
        t_other.start()
        t_other.join(0.3)
        assert t_other.is_alive(), "decided its mode while the legacy capture was open"
    finally:
        release.set()
    t_legacy.join(10)
    t_other.join(10)
    assert not errors, errors
    # Not recorded into the legacy graph: its own self-capture on a private stream,
    # then a replay and a sync on the (no longer capturing) current stream.
    assert body.runs == 2 and body.ctxs[1].stream is not fake.current
    assert r.graph is not None and (r.graph, fake.current) in fake.launched
    assert fake.synced[-1] is fake.current


def test_legacy_plugin_eager_body_runs_on_the_current_stream(monkeypatch):
    """A legacy plugin's own launches go to Warp's current stream: when the call resolved
    another stream, the WHOLE body runs on the current stream, ordered after the resolved
    stream and before it (prepare and eager calls alike)."""
    fake = FakeCuda(monkeypatch)
    side = FakeStream(fake.device)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True, legacy_stream=True, default_stream=side,
                    name="legacy")
    r.prepare(body)
    assert body.ctxs[0].stream is fake.current
    assert fake.current.waited_on == [side] and side.waited_on == [fake.current]
    assert fake.synced == [side]
    r.run(body, policy="eager", sync=True)
    assert body.ctxs[1].stream is fake.current
    assert fake.current.waited_on == [side] * 2 and side.waited_on == [fake.current] * 2
    assert fake.synced == [side] * 2
    r.run(body, stream=fake.current, policy="eager")     # already current: no detour
    assert body.ctxs[2].stream is fake.current and len(side.waited_on) == 2
    fake.current.is_capturing = True                     # the plugin would be recorded there
    with pytest.raises(LegacyPluginStreamError, match="current stream is"):
        r.run(body, policy="eager")
    assert body.runs == 3


def test_ctx_aware_eager_body_stays_on_the_resolved_stream(monkeypatch):
    fake = FakeCuda(monkeypatch)
    side = FakeStream(fake.device)
    body = _Body()
    r = GraphRunner("cuda:0", capturable=True, default_stream=side)
    r.prepare(body)
    r.run(body, policy="eager")
    assert [c.stream for c in body.ctxs] == [side, side]
    assert fake.current.waited_on == []


# ---------------------------------------------------------------------------------------
# _compat
# ---------------------------------------------------------------------------------------

def _fake_runtime(captures=None, synced=None):
    core = types.SimpleNamespace(
        wp_cuda_stream_get_capture_id=lambda handle: handle * 10,
        wp_cuda_stream_synchronize=(synced.append if synced is not None else None),
    )
    return types.SimpleNamespace(captures=captures or {}, core=core)


def test_compat_stream_synchronize(monkeypatch):
    synced: list = []
    monkeypatch.setattr(_compat, "_runtime", lambda: _fake_runtime(synced=synced))
    _compat.stream_synchronize(None)
    _compat.stream_synchronize(types.SimpleNamespace(is_capturing=True, cuda_stream=7))
    assert synced == []
    _compat.stream_synchronize(types.SimpleNamespace(is_capturing=False, cuda_stream=7))
    assert synced == [7]


def test_compat_stream_synchronize_relaxes_the_thread_capture_mode(monkeypatch):
    calls: list = []

    def sync(handle):
        calls.append(("sync", handle))
        if handle == 13:
            raise RuntimeError("boom")

    core = types.SimpleNamespace(
        wp_cuda_stream_synchronize=sync,
        wp_cuda_thread_exchange_capture_mode=lambda m: calls.append(("mode", m)) or 1)
    monkeypatch.setattr(_compat, "_runtime", lambda: types.SimpleNamespace(core=core))
    _compat.stream_synchronize(types.SimpleNamespace(is_capturing=False, cuda_stream=7))
    assert calls == [("mode", 2), ("sync", 7), ("mode", 1)]   # RELAXED, then restored
    del calls[:]
    with pytest.raises(RuntimeError, match="boom"):
        _compat.stream_synchronize(types.SimpleNamespace(is_capturing=False, cuda_stream=13))
    assert calls == [("mode", 2), ("sync", 13), ("mode", 1)]  # restored on error too


def test_compat_capture_stream():
    a, b = FakeStream(), FakeStream()
    a.cuda_stream, b.cuda_stream = 5, 6
    dev = types.SimpleNamespace(captures={a: object(), b: object()})
    assert _compat.capture_stream(dev, 6) is b
    assert _compat.capture_stream(dev, 7) is None
    assert _compat.capture_stream(types.SimpleNamespace(), 6) is None   # missing internal


def test_compat_registered_graph_and_anchor(monkeypatch):
    g = types.SimpleNamespace()
    capturing = types.SimpleNamespace(is_capturing=True, cuda_stream=3)
    idle = types.SimpleNamespace(is_capturing=False, cuda_stream=3)
    monkeypatch.setattr(_compat, "_runtime", lambda: _fake_runtime(captures={30: g}))
    assert _compat.registered_graph(None) is None
    assert _compat.registered_graph(idle) is None
    assert _compat.registered_graph(capturing) is g
    assert _compat.registered_graph(
        types.SimpleNamespace(is_capturing=True, cuda_stream=4)) is None   # not Warp's
    a, b = _Owner(), _Owner()
    _compat.anchor(capturing, a)
    _compat.anchor(capturing, a, b, None)
    assert g._track_gen_owners == [a, b]
    _compat.anchor(idle, a)                    # no registered graph: no-op
    monkeypatch.setattr(_compat, "_runtime", lambda: types.SimpleNamespace())
    assert _compat.registered_graph(capturing) is None   # missing internals: degrade


def test_compat_registered_graph_on_the_real_runtime():
    # CPU-only process: no capture is registered.
    assert _compat.registered_graph(types.SimpleNamespace(is_capturing=True,
                                                          cuda_stream=1)) is None


# ---------------------------------------------------------------------------------------
# torch stream adoption (fake torch module)
# ---------------------------------------------------------------------------------------

def _fake_torch(monkeypatch, handle: int, capturing: bool = True):
    """Install a fake ``torch`` whose current stream on device 0 has ``handle``."""
    state = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=handle),
                                  capturing=capturing)
    cuda = types.SimpleNamespace(
        is_initialized=lambda: True,
        current_device=lambda: 0,
        is_current_stream_capturing=lambda: state.capturing,
        current_stream=lambda device=None: state.stream,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return state


def test_torch_adoption_never_creates_a_throwaway_wrapper(monkeypatch):
    """A wp.Stream wrapper unregisters its handle process-wide when collected (not
    refcounted), which would break a Warp capture open on the same handle and NULL
    Warp's current stream. Adoption must reuse a live wrapper, or cache one."""
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(context_mod, "_TORCH_WRAPPERS", {})
    made: list = []

    def stream_from_torch(ts):
        s = FakeStream(fake.device)
        s.cuda_stream = ts.cuda_stream
        made.append(s)
        return s

    monkeypatch.setattr(wp, "stream_from_torch", stream_from_torch)
    torch = _fake_torch(monkeypatch, fake.current.cuda_stream)
    # torch's capturing stream IS Warp's current stream (the documented recipe)
    assert resolve_stream("cuda:0") is fake.current
    assert resolve_stream("cuda:0", "torch") is fake.current
    # a Warp capture is open on another wrapper of the handle: reuse that wrapper
    mine = FakeStream(fake.device)
    fake.device.captures = {mine: object()}
    torch.stream = types.SimpleNamespace(cuda_stream=mine.cuda_stream)
    assert resolve_stream("cuda:0") is mine
    assert made == []
    # otherwise ONE wrapper per handle, kept for the process
    fake.device.captures = {}
    first = resolve_stream("cuda:0")
    assert made == [first] and first.cuda_stream == mine.cuda_stream
    assert resolve_stream("cuda:0") is first
    assert resolve_stream("cuda:0", "torch") is first
    torch.capturing = False
    assert resolve_stream("cuda:0") is fake.current      # not capturing: no adoption
    assert made == [first]


def test_sync_is_skipped_while_torch_captures_in_this_thread(monkeypatch):
    """A producer bound to its own stream, called inside torch.cuda.graph without
    stream="torch", runs on its stream; a host sync from the capturing thread could
    invalidate torch's capture, so it is skipped."""
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(context_mod, "_TORCH_WRAPPERS", {})
    side = FakeStream(fake.device)
    r = GraphRunner("cuda:0", capturable=True, default_stream=side)
    r.prepare(_Body())
    assert fake.synced == [side]
    _fake_torch(monkeypatch, 0xDEAD)                     # torch captures its own stream
    ctx = r.run(_Body(), sync=True)
    assert ctx.stream is side and fake.launched == [(r.graph, side)]
    assert fake.synced == [side]


def test_construction_refuses_a_torch_capture_in_this_thread(monkeypatch):
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(context_mod, "_TORCH_WRAPPERS", {})
    side = FakeStream(fake.device)
    _fake_torch(monkeypatch, 0xBEEF)
    body = _Body()
    with pytest.raises(CaptureError):
        GraphRunner("cuda:0", capturable=True, default_stream=side).prepare(body)
    assert body.runs == 0


# ---------------------------------------------------------------------------------------
# Facades on a fake CUDA device (policy only: the pipelines are faked)
# ---------------------------------------------------------------------------------------

# (stream, capturing) of one pipeline run, captured at call time
_Run = types.SimpleNamespace


def _fake_track_facade(monkeypatch, capturable=True, accepts_ctx=True):
    monkeypatch.setattr(generator_registry, "GENERATORS", {})
    monkeypatch.setattr(generator_registry, "_LOADED", True)
    generator_registry.register(generator_registry.GeneratorSpec(
        name="fake", alloc_scratch=lambda config: object(),
        generate=lambda *a, **k: None, capturable=capturable, accepts_ctx=accepts_ctx))
    runs: list = []
    track = types.SimpleNamespace(generation=None)
    monkeypatch.setattr(warp_pipeline, "_inflate_warp_alloc",
                        lambda config, generator_spec=None: (track, object()))
    monkeypatch.setattr(warp_pipeline, "_run_pipeline",
                        lambda config, seed, out, scratch, generator_spec=None, ctx=None:
                        runs.append(_Run(stream=ctx.stream, capturing=ctx.capturing)))
    cfg = TrackGenConfig(generator="fake", device="cuda", num_envs=1)
    return cfg, runs, track, TrackGenerator


def _fake_gate_facade(monkeypatch, capturable=True, accepts_ctx=True):
    monkeypatch.setattr(gate_generator_registry, "GATE_GENERATORS", {})
    monkeypatch.setattr(gate_generator_registry, "_LOADED", True)
    gate_generator_registry.register(gate_generator_registry.GateGeneratorSpec(
        name="fake", alloc_scratch=lambda config: object(),
        generate=lambda *a, **k: None, max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}), capturable=capturable,
        accepts_ctx=accepts_ctx))
    runs: list = []
    seq = types.SimpleNamespace(generation=None)
    monkeypatch.setattr(warp_gate, "_gate_warp_alloc",
                        lambda config, generator_spec: (seq, (object(),) * 6))
    monkeypatch.setattr(warp_gate, "_run_gate_pipeline",
                        lambda config, seed, out, scratch, generator_spec, ctx=None:
                        runs.append(_Run(stream=ctx.stream, capturing=ctx.capturing)))
    cfg = GateGenConfig(generator="fake", device="cuda", num_envs=1)
    return cfg, runs, seq, GateGenerator


_FACADES = [_fake_track_facade, _fake_gate_facade]


def _fake_rng():
    return types.SimpleNamespace(seeds_warp=types.SimpleNamespace(shape=(1,)))


def _setup_fake_facade(monkeypatch, make, **kw):
    """A facade on the fake device. ``copies`` records the stream of every seed prologue
    (the effective-seed launch that opens every generator body, PR4; a seed copy in PR3)."""
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(wp, "empty", lambda *a, **k: object())
    monkeypatch.setattr(wp, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(wp, "copy", lambda *a, **k: pytest.fail("unexpected wp.copy"))
    copies, _ = patch_facade_seeds(monkeypatch)
    cfg, runs, result, cls = make(monkeypatch, **kw)
    gen = cls(cfg, _fake_rng())
    return fake, gen, runs, copies, result


@pytest.mark.parametrize("make", _FACADES)
def test_facade_prepares_then_self_graphs_then_replays(monkeypatch, make):
    fake, gen, runs, copies, result = _setup_fake_facade(monkeypatch, make)
    # construction: ONE real pass (seed prologue + pipeline) on the resolved stream
    assert len(runs) == 1 and runs[0].stream is fake.current and not runs[0].capturing
    assert copies == [fake.current]
    assert gen.graph is None and fake.captures == []
    assert fake.synced == [fake.current]               # construction waits for the warm-up

    assert gen.generate(1) is result
    assert len(runs) == 2 and runs[1].capturing        # recorded into the self-graph
    assert copies[1] is runs[1].stream                 # seed prologue is inside the graph
    assert gen.graph is not None and gen._graph is gen.graph
    assert fake.launched == [(gen.graph, fake.current)]
    assert fake.synced == [fake.current] * 2           # sync=True default

    gen.generate(sync=False)
    assert len(runs) == 2 and len(fake.launched) == 2
    assert fake.synced == [fake.current] * 2


@pytest.mark.parametrize("make", _FACADES)
def test_facade_records_into_a_caller_capture(monkeypatch, make):
    fake, gen, runs, copies, result = _setup_fake_facade(monkeypatch, make)
    fake.current.is_capturing = True                   # caller: ScopedCapture / torch graph
    assert gen.generate() is result
    assert gen.enqueue() is result
    assert len(runs) == 3 and all(c.capturing for c in runs[1:])
    assert all(c.stream is fake.current for c in runs[1:])
    assert copies[1:] == [fake.current, fake.current]
    assert fake.captures == [] and fake.launched == []
    assert fake.synced == [fake.current]               # construction only, never recording
    assert gen.graph is None                           # recording never self-captures


@pytest.mark.parametrize("make", _FACADES)
def test_facade_enqueue_is_eager_outside_a_capture(monkeypatch, make):
    fake, gen, runs, copies, result = _setup_fake_facade(monkeypatch, make)
    side = FakeStream(fake.device)
    assert gen.enqueue(stream=side) is result
    assert len(runs) == 2 and runs[1].stream is side and not runs[1].capturing
    assert fake.captures == [] and fake.launched == []
    assert fake.synced == [fake.current]               # construction only: enqueue never syncs
    assert gen.graph is None


@pytest.mark.parametrize("make", _FACADES)
def test_facade_non_capturable(monkeypatch, make):
    fake, gen, runs, copies, result = _setup_fake_facade(monkeypatch, make,
                                                         capturable=False)
    gen.generate()
    assert len(runs) == 2 and fake.captures == [] and fake.launched == []
    assert fake.synced == [fake.current] * 2 and gen.graph is None
    fake.current.is_capturing = True
    with pytest.raises(NotCapturableError, match="'fake'"):
        gen.generate()
    with pytest.raises(NotCapturableError):
        gen.enqueue()
    assert len(runs) == 2 and len(copies) == 2         # nothing was launched


@pytest.mark.parametrize("make", _FACADES)
def test_facade_legacy_plugin_stream_rule(monkeypatch, make):
    fake, gen, runs, copies, result = _setup_fake_facade(monkeypatch, make,
                                                         accepts_ctx=False)
    gen.generate()
    (cap,) = fake.captures
    assert cap.stream is fake.current                  # legacy: current stream
    side = FakeStream(fake.device)
    side.is_capturing = True
    with pytest.raises(LegacyPluginStreamError):
        gen.generate(stream=side)
    assert len(runs) == 2


@pytest.mark.parametrize("make", _FACADES)
def test_facade_construction_inside_a_capture_raises(monkeypatch, make):
    fake = FakeCuda(monkeypatch)
    monkeypatch.setattr(wp, "empty", lambda *a, **k: object())
    monkeypatch.setattr(wp, "copy", lambda *a, **k: None)
    cfg, runs, _, cls = make(monkeypatch)
    fake.current.is_capturing = True
    with pytest.raises(CaptureError):
        cls(cfg, _fake_rng())
    assert runs == []


@pytest.mark.parametrize("make", _FACADES)
def test_facade_construction_refuses_a_capturing_current_stream_before_allocating(
        monkeypatch, make):
    """The caller captures Warp's current stream while the generator would resolve a
    side stream: construction allocates and zero-fills on the CURRENT stream, so it
    must refuse before allocating anything (nothing lands in the caller's graph)."""
    fake = FakeCuda(monkeypatch)
    allocs: list = []
    monkeypatch.setattr(wp, "empty", lambda *a, **k: allocs.append("empty") or object())
    monkeypatch.setattr(wp, "copy", lambda *a, **k: None)
    cfg, runs, _, cls = make(monkeypatch)
    for mod, name in ((warp_pipeline, "_inflate_warp_alloc"), (warp_gate, "_gate_warp_alloc")):
        real = getattr(mod, name)
        monkeypatch.setattr(mod, name,
                            lambda *a, _r=real, _n=name, **k: allocs.append(_n) or _r(*a, **k))
    side = FakeStream(fake.device)
    fake.current.is_capturing = True
    with pytest.raises(CaptureError):
        cls(cfg, _fake_rng(), stream=side)
    assert runs == [] and allocs == []


def test_course_construction_refuses_a_capture_before_allocating(monkeypatch):
    fake = FakeCuda(monkeypatch)
    fake.current.is_capturing = True
    monkeypatch.setattr(course_mod, "PerEnvSeededRNG",
                        lambda *a, **k: pytest.fail("allocated under a capture"))
    cfg = CourseConfig(mode="track", gen=TrackGenConfig(device="cuda", num_envs=E),
                       collision="segments", checkpoint_spacing=0.6)
    with pytest.raises(CaptureError, match="course"):
        Course(cfg)


# ---------------------------------------------------------------------------------------
# Facades on the real CPU device
# ---------------------------------------------------------------------------------------

def test_track_generator_body_runs_once_at_construction_then_once_per_call(monkeypatch):
    calls = _counting(monkeypatch, generator_registry, "GENERATORS", "bezier")
    gen = TrackGenerator(_track_cfg(), _rng())
    assert len(calls) == 1
    gen.generate()
    assert len(calls) == 2
    gen.generate(stream=None, sync=False)
    gen.enqueue()
    assert len(calls) == 4
    assert gen.graph is None and gen._graph is None


def test_gate_generator_body_runs_once_at_construction_then_once_per_call(monkeypatch):
    calls = _counting(monkeypatch, gate_generator_registry, "GATE_GENERATORS", "bezier")
    gen = GateGenerator(_gate_cfg(), _rng())
    assert len(calls) == 1
    gen.generate()
    gen.generate(sync=False)
    gen.enqueue()
    assert len(calls) == 4
    assert gen.graph is None


def test_track_generator_construction_leaves_a_valid_batch():
    gen = TrackGenerator(_track_cfg(), _rng(11))
    snap = gen._track.clone()
    twin = TrackGenerator(_track_cfg(), _rng(11))
    _assert_same(snap, twin.generate(), _TRACK_FIELDS)
    _assert_same(gen.generate(), snap, _TRACK_FIELDS)


@pytest.mark.parametrize("call", ["generate", "generate_nosync", "enqueue"])
def test_track_generator_cpu_calls_agree(call):
    gen = TrackGenerator(_track_cfg(), _rng(5))
    ref = TrackGenerator(_track_cfg(), _rng(9)).generate().clone()
    gen._rng.set_seeds_warp(wp.array(9 + np.arange(E), dtype=wp.int32, device="cpu"), None)
    out = {"generate": lambda: gen.generate(),
           "generate_nosync": lambda: gen.generate(stream=None, sync=False),
           "enqueue": lambda: gen.enqueue()}[call]()
    assert out is gen._track
    _assert_same(out, ref, _TRACK_FIELDS)        # every call reads the rng's live seeds


@pytest.mark.parametrize("call", ["generate", "generate_nosync", "enqueue"])
def test_gate_generator_cpu_calls_agree(call):
    gen = GateGenerator(_gate_cfg(), _rng(5))
    ref = GateGenerator(_gate_cfg(), _rng(9)).generate().clone()
    gen._rng.set_seeds_warp(wp.array(9 + np.arange(E), dtype=wp.int32, device="cpu"), None)
    out = {"generate": lambda: gen.generate(),
           "generate_nosync": lambda: gen.generate(sync=False),
           "enqueue": lambda: gen.enqueue()}[call]()
    assert out is gen._gate_sequence
    _assert_same(out, ref, _GATE_FIELDS)


def test_cpu_generation_paths_do_not_allocate(no_alloc):
    tgen = TrackGenerator(_track_cfg(), _rng())
    ggen = GateGenerator(_gate_cfg(), _rng())
    with no_alloc as hits:
        tgen.generate()
        tgen.generate(sync=False)
        tgen.enqueue()
        ggen.generate()
        ggen.enqueue()
    assert hits == []


@pytest.mark.parametrize("cls,cfg", [(TrackGenerator, _track_cfg), (GateGenerator, _gate_cfg)])
def test_facade_stream_argument_must_be_a_wp_stream(cls, cfg):
    with pytest.raises(TypeError, match="wp.Stream"):
        cls(cfg(), _rng(), stream="torch")
    assert cls(cfg(), _rng(), stream=None).generate() is not None


@pytest.mark.parametrize("which", ["track", "gate"])
def test_config_is_a_deep_snapshot(which):
    make_cfg, cls, fields = ((_track_cfg, TrackGenerator, _TRACK_FIELDS) if which == "track"
                             else (_gate_cfg, GateGenerator, _GATE_FIELDS))
    cfg = make_cfg()
    gen = cls(cfg, _rng(21))
    ref = cls(make_cfg(), _rng(21)).generate().clone()
    # Mutating the caller's object after construction has no effect ...
    cfg.scale = 3.0
    cfg.num_envs = E + 3
    # ... nor does mutating the object the property hands out (a fresh deep copy).
    leaked = gen.config
    leaked.scale = 123.0
    assert leaked is not gen.config and leaked != gen.config
    assert gen.config == make_cfg()
    _assert_same(gen.generate(), ref, fields)


@pytest.mark.parametrize("which", ["track", "gate"])
def test_config_with_a_device_object(which):
    """``device=wp.get_device(...)`` (a ctypes-backed object deepcopy cannot copy) is
    accepted: the snapshot shares the Device and copies everything else."""
    dev = wp.get_device("cpu")
    if which == "track":
        cfg, ref_cfg, cls, fields = (TrackGenConfig(device=dev, num_envs=E), _track_cfg(),
                                     TrackGenerator, _TRACK_FIELDS)
    else:
        cfg, ref_cfg, cls, fields = (GateGenConfig(device=dev, num_envs=E, gate_width=0.2),
                                     _gate_cfg(), GateGenerator, _GATE_FIELDS)
    gen = cls(cfg, _rng(21))
    snap = gen.config
    assert snap.device is dev and snap == cfg and snap is not cfg
    cfg.num_envs = E + 1
    assert gen.config.num_envs == E
    _assert_same(gen.generate(), cls(ref_cfg, _rng(21)).generate(), fields)


# ---------------------------------------------------------------------------------------
# Course on the real CPU device
# ---------------------------------------------------------------------------------------

def _course(mode="track", seeds=7, **kw):
    if mode == "track":
        kw.setdefault("collision", "segments")
        kw.setdefault("checkpoint_spacing", 0.6)
        kw.setdefault("max_checkpoints", 64)
        cfg = CourseConfig(mode="track", gen=_track_cfg(), seeds=seeds, **kw)
    else:
        kw.setdefault("post_radius", 0.02)
        cfg = CourseConfig(mode="gates", gen=_gate_cfg(), seeds=seeds, **kw)
    return Course(cfg)


def _assert_course_same(a: Course, b: Course) -> None:
    if a._cfg.mode == "track":
        _assert_same(a.result, b.result, _TRACK_FIELDS)
        np.testing.assert_array_equal(a.checkpoints.count.numpy(), b.checkpoints.count.numpy())
        np.testing.assert_array_equal(_flat(a.checkpoints.position),
                                      _flat(b.checkpoints.position))
    else:
        _assert_same(a.result, b.result, _GATE_FIELDS)
        np.testing.assert_array_equal(_flat(a.course_line.track.center),
                                      _flat(b.course_line.track.center))
        np.testing.assert_array_equal(_flat(a._posts), _flat(b._posts))
    np.testing.assert_array_equal(a.progress._progress.numpy(), b.progress._progress.numpy())


@pytest.mark.parametrize("mode,kw", [
    ("track", dict(collision="segments")),
    ("track", dict(collision="sdf", sdf_resolution=16, heightfield_resolution=16)),
    ("gates", dict(post_radius=0.02)),
    ("gates", dict(post_radius=0.0, frame_collision=True, agent_radius=0.05,
                   frame_thickness=0.02, frame_depth=0.02)),
])
def test_course_subtools_exist_right_after_construction(mode, kw):
    c = _course(mode, **kw)
    assert c.progress is not None and c.checkpoints is not None
    assert c.collision is not None
    if mode == "track":
        assert c.checkpoint_sampler is not None and c.result is c.generator._track
        assert (kw.get("heightfield_resolution") is None) == (c.heightfield is None)
    else:
        assert c.course_line is not None and c.localizer is not None
        assert c.result is c.generator._gate_sequence
    assert c.graph is None                    # CPU: never self-graphed
    mask = wp.full(E, 1, dtype=wp.int32, device="cpu")
    c.reset(mask)                             # usable before any explicit generate()


@pytest.mark.parametrize("mode", ["track", "gates"])
def test_course_construction_matches_a_fresh_generator(mode):
    c = _course(mode, seeds=13)
    if mode == "track":
        _assert_same(c.result, TrackGenerator(_track_cfg(), _rng(13)).generate(),
                     _TRACK_FIELDS)
    else:
        _assert_same(c.result, GateGenerator(_gate_cfg(), _rng(13)).generate(),
                     _GATE_FIELDS)


@pytest.mark.parametrize("mode", ["track", "gates"])
def test_course_regenerate_and_reseeds_match_a_fresh_course(mode):
    c = _course(mode, seeds=7)
    c.progress._progress.fill_(5)             # stale state a regeneration must clear
    assert c.regenerate() is c.result
    _assert_course_same(c, _course(mode, seeds=7))

    assert c.generate(seeds=1234) is c.result
    _assert_course_same(c, _course(mode, seeds=1234))

    seeds = wp.array(np.array([5, 77, 3, 900], np.int32), dtype=wp.int32, device="cpu")
    c.generate(seeds=seeds)
    _assert_course_same(c, _course(mode, seeds=seeds))

    c.generate(seeds=2**31 - 2)               # int32 wrap-around like PerEnvSeededRNG
    _assert_course_same(c, _course(mode, seeds=2**31 - 2))

    c.generate(sync=False, stream=None)
    _assert_course_same(c, _course(mode, seeds=2**31 - 2))
    c.enqueue_generate()
    _assert_course_same(c, _course(mode, seeds=2**31 - 2))


def test_course_body_counts(monkeypatch):
    calls = _counting(monkeypatch, generator_registry, "GENERATORS", "bezier")
    c = _course("track")
    # generator prepare + course prepare (one regeneration body each)
    assert len(calls) == 2
    c.regenerate()
    c.generate(seeds=3)
    c.enqueue_generate()
    assert len(calls) == 5


@pytest.mark.parametrize("mode", ["track", "gates"])
def test_course_regeneration_does_not_allocate(mode, no_alloc):
    c = _course(mode)
    seeds = wp.array(np.arange(E, dtype=np.int32) + 40, dtype=wp.int32, device="cpu")
    with no_alloc as hits:
        c.regenerate()
        c.generate()
        c.generate(seeds=99)
        c.generate(seeds=seeds)
        c.enqueue_generate()
    assert hits == []


def test_course_non_capturable_generate_raises_before_reseeding(monkeypatch):
    """With a non-capturable generator, generate(seeds=...) under a caller capture must
    raise BEFORE the reseed launches (nothing recorded into the caller's graph)."""
    class _Capturing(LaunchContext):
        __slots__ = ()

        @property
        def capturing(self):
            return True

    c = _course("track")
    launches: list = []
    monkeypatch.setattr(c._runner, "_capturable", False)
    monkeypatch.setattr(c._runner, "context", lambda stream=None: _Capturing("cpu", None))
    monkeypatch.setattr(wp, "launch", lambda *a, **k: launches.append(a))
    seeds = wp.array(np.arange(E, dtype=np.int32), dtype=wp.int32, device="cpu")
    for s in (3, seeds):
        with pytest.raises(NotCapturableError, match="course"):
            c.generate(seeds=s)
    assert launches == []


@pytest.mark.parametrize("mode", ["track", "gates"])
def test_course_with_a_device_object(mode):
    dev = wp.get_device("cpu")
    if mode == "track":
        cfg = CourseConfig(mode="track", gen=TrackGenConfig(device=dev, num_envs=E), seeds=7,
                           collision="segments", checkpoint_spacing=0.6, max_checkpoints=64)
    else:
        cfg = CourseConfig(mode="gates",
                           gen=GateGenConfig(device=dev, num_envs=E, gate_width=0.2),
                           seeds=7, post_radius=0.02)
    c = Course(cfg)
    _assert_course_same(c, _course(mode, seeds=7))
    c.generate(seeds=99)
    _assert_course_same(c, _course(mode, seeds=99))


def test_course_reseed_and_regeneration_hold_the_runner_lock_together(monkeypatch):
    """generate(seeds=...) issues its reseed launches inside the same serialization as
    the regeneration (GraphRunner.serial), never between two lock windows."""
    c = _course("track")
    depth = [0]
    held: list = []

    @contextlib.contextmanager
    def serial(ctx):
        depth[0] += 1
        try:
            yield
        finally:
            depth[0] -= 1

    real_set = c.rng.set_seeds_warp
    real_run = c._runner.run

    def set_seeds_warp(*a, **k):
        held.append(("reseed", depth[0] > 0))
        return real_set(*a, **k)

    def run(*a, **k):
        held.append(("run", depth[0] > 0))
        return real_run(*a, **k)

    monkeypatch.setattr(c._runner, "serial", serial)
    monkeypatch.setattr(c._runner, "run", run)
    monkeypatch.setattr(c.rng, "set_seeds_warp", set_seeds_warp)
    c.generate(seeds=5)
    c.generate(seeds=wp.array(np.arange(E, dtype=np.int32) + 5, dtype=wp.int32, device="cpu"))
    assert held == [("reseed", True), ("run", True)] * 2
    _assert_course_same(c, _course("track", seeds=5))
