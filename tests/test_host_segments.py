"""HOST stages and segmented execution (roadmap PR12), on CPU (and a fake CUDA device for
the runner's policy).

- Planner rule 6: consecutive stages of equal host-ness form one segment
  (``Program.segments``, ``capture_class``, ``host_stage_ids``); a legacy
  ``capturable=False`` generator is a HOST segment followed by a capturable tail; gate
  pipelines segment the same way.
- ``PipelineSpec.require_caller_capturable``: HOST stages refused at build, host-side,
  naming every one; the flag round-trips through ``to_dict`` / ``from_dict``.
- Eager execution of a HOST pipeline (``ctx.host()``): deterministic, equal to running
  ``enqueue_segment(i)`` one by one, masked regeneration exact.
- Refusals before any launch: an (emulated or real CPU) caller capture of a Program with
  HOST stages raises ``NotCapturableError`` naming them, ahead of the "not prepared"
  ``CaptureError`` (also ``Course.generate(seeds=...)`` ahead of its reseed, and
  construction inside a CPU graph capture); a non-HOST segment stays recordable.
- The runner's SEGMENTED mode on a fake CUDA device: one capture per non-HOST part, HOST
  parts eager between graph launches, keyed caches, and the eager / RECORD rows.
- The conformance kit on the BOUNDED / HOST toys.
"""
from __future__ import annotations

import dataclasses
import threading
import types
import warnings
import weakref

import numpy as np
import pytest
import warp as wp

from tests import _bounded_toys as T
from tests._fake_cuda import FakeCuda, FakeStream
from tests._plugin_toys import TOY_CIRCLE, toy_circle_alloc_scratch, toy_circle_generate
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import runtime
from track_gen._src.gate_generator import GateGenerator
from track_gen.exec import CaptureError, GraphRunner, LaunchContext, NotCapturableError
from track_gen.exec import runner as runner_mod
from track_gen.exec.runner import _BodyPart, _SegmentedBody
from track_gen.course import Course, CourseConfig
from track_gen.pipeline import PipelineSpec, PlanError, Program, Segment, build, presets
from track_gen.plugin import CaptureClass, GeneratorSpec, registry
from track_gen.testing import check_gate_generator, check_generator, check_stage
from track_gen.testing.contract import no_allocation
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 8
_TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                 "valid", "count", "winding", "truncated", "generation", "fail_mask")


@pytest.fixture(autouse=True)
def _toys(monkeypatch):
    T.register_bounded_toys(registry)
    T.register_broken_bounded_toys(registry)


def _rng(seeds=11, n=E):
    return PerEnvSeededRNG(seeds=seeds, num_envs=n, device=DEV)


def _bytes(result, fields=_TRACK_FIELDS) -> dict:
    return {f: np.ascontiguousarray(getattr(result, f).numpy()).tobytes()
            for f in fields if getattr(result, f, None) is not None}


def _counted(base_name: str, calls: list, kind: str = "generator") -> str:
    """A subclass of the registered Stage ``base_name`` whose enqueue counts its runs."""
    base = registry.get(kind, base_name)
    name = f"{base_name}_counted"

    class Counted(base):
        def build(self, b, p):
            body = super().build(b, p)

            def enqueue(ctx):
                calls.append(bool(ctx.capturing))
                body(ctx)
            return enqueue

    Counted.name = name
    registry.add(Counted)
    return name


def _legacy_host_spec(name="test.legacy_host") -> str:
    registry.add(GeneratorSpec(name=name, alloc_scratch=toy_circle_alloc_scratch,
                               generate=toy_circle_generate, capturable=False))
    return name


# ---------------------------------------------------------------------------------------
# Planner rule 6: segments
# ---------------------------------------------------------------------------------------

def test_a_host_generator_is_a_host_segment_followed_by_the_tail():
    p = build(presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE), seeds=3)
    head, tail = p.segments
    assert isinstance(head, Segment) and head.index == 0 and head.host
    assert head.capture_class is CaptureClass.HOST and head.stage_ids == ("generator",)
    assert not tail.host and tail.capture_class is CaptureClass.STATIC
    assert tail.stage_ids == p._stage_ids[1:]
    assert p.capture_class is CaptureClass.HOST and p.host_stage_ids == ("generator",)
    assert not p.capturable and p.graph is None and p.segment_graphs() is None


def test_a_host_stage_in_the_middle_makes_three_segments():
    p = build(presets.track(num_envs=E, device=DEV, relax=T.HOST_IDENTITY), seeds=3)
    assert [s.host for s in p.segments] == [False, True, False]
    assert p.segments[1].stage_ids == ("relax",)
    assert sum(len(s.stage_ids) for s in p.segments) == len(p._stage_ids)
    assert p.host_stage_ids == ("relax",)


def test_bounded_and_static_stages_share_one_segment():
    p = build(presets.track(num_envs=E, device=DEV, relax=T.CONVERGING), seeds=3)
    (seg,) = p.segments
    assert seg.capture_class is CaptureClass.BOUNDED and not seg.host
    assert p.capture_class is CaptureClass.BOUNDED and p.capturable
    rep = TrackGenerator(TrackGenConfig(generator="repulsive", device=DEV, num_envs=2,
                                        num_points=64, repulsive_stages=(16, 32, 64),
                                        N_max=384, half_width=0.1), _rng(n=2))
    assert rep.program.capture_class is CaptureClass.BOUNDED
    assert len(rep.program.segments) == 1 and rep.program.capturable


def test_a_legacy_capturable_false_generator_is_a_host_segment():
    name = _legacy_host_spec()
    gen = TrackGenerator(TrackGenConfig(generator=name, device=DEV, num_envs=E), _rng())
    p = gen.program
    assert [s.host for s in p.segments] == [True, False]
    assert p.host_stage_ids == ("generator",) and not gen._capturable
    registry.add(TOY_CIRCLE)                      # the same generator, capturable
    ref = TrackGenerator(TrackGenConfig(generator=TOY_CIRCLE.name, device=DEV, num_envs=E),
                         _rng())
    assert len(ref.program.segments) == 1
    assert _bytes(gen.generate()) == _bytes(ref.generate())     # outputs unchanged


def test_a_host_gate_generator_segments_the_gate_pipeline():
    gen = GateGenerator(presets.gates(num_envs=E, device=DEV, generator=T.HOST_RING),
                        _rng())
    p = gen.program
    assert [s.host for s in p.segments] == [True, False]
    assert p.host_stage_ids == ("generator",)
    seq = gen.generate()
    assert (seq.count.numpy() == 8).all() and seq.valid.numpy().any()


# ---------------------------------------------------------------------------------------
# PipelineSpec.require_caller_capturable
# ---------------------------------------------------------------------------------------

def test_require_caller_capturable_names_every_host_stage_before_allocating():
    spec = presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE,
                         relax=T.HOST_IDENTITY)
    strict = dataclasses.replace(spec, require_caller_capturable=True)
    with no_allocation(), pytest.raises(
            PlanError, match=r"PipelineSpec.require_caller_capturable=True, but stage\(s\) "
                             r"'generator' \(generator.toy.host_circle\), 'relax' "
                             r"\(relax.toy.host_identity\) are HOST \(they read device data "
                             r"back or run host control flow, so they run eagerly and "
                             r"refuse a caller's capture\): use STATIC / BOUNDED stages or "
                             r"drop require_caller_capturable"):
        build(strict)
    gates = dataclasses.replace(presets.gates(num_envs=E, device=DEV,
                                              generator=T.HOST_RING),
                                require_caller_capturable=True)
    with pytest.raises(PlanError, match=r"stage\(s\) 'generator' "
                                        r"\(gate_generator.toy.host_ring\) is HOST"):
        build(gates)
    legacy = _legacy_host_spec()
    with pytest.raises(PlanError, match=r"stage\(s\) 'generator' \(generator "
                                        r"'test.legacy_host', GeneratorSpec "
                                        r"capturable=False\) is HOST"):
        build(PipelineSpec(presets.track(num_envs=E, device=DEV).batch,
                           (dataclasses.replace(
                               presets.track(num_envs=E, device=DEV).stages[0],
                               name=legacy),) +
                           presets.track(num_envs=E, device=DEV).stages[1:],
                           require_caller_capturable=True))


def test_require_caller_capturable_accepts_static_and_bounded_stages():
    spec = dataclasses.replace(presets.track(num_envs=E, device=DEV, relax=T.CONVERGING),
                               require_caller_capturable=True)
    p = build(spec, seeds=3)
    assert p.capturable and p.spec.require_caller_capturable is True


def test_require_caller_capturable_round_trips():
    spec = presets.track(num_envs=E, device=DEV)
    data = spec.to_dict()
    assert "require_caller_capturable" not in data           # unchanged without it
    assert PipelineSpec.from_dict(data).require_caller_capturable is False
    strict = dataclasses.replace(spec, require_caller_capturable=True)
    data = strict.to_dict()
    assert data["require_caller_capturable"] is True
    assert PipelineSpec.from_dict(data) == strict
    assert strict.replace("relax", strict.stages[2]).require_caller_capturable is True
    with pytest.raises(TypeError, match="require_caller_capturable must be a bool"):
        PipelineSpec(spec.batch, spec.stages, require_caller_capturable=1)


# ---------------------------------------------------------------------------------------
# Eager execution
# ---------------------------------------------------------------------------------------

def test_a_host_generator_runs_eagerly_and_deterministically():
    calls: list = []
    name = _counted(T.HOST_CIRCLE, calls)
    spec = presets.track(num_envs=E, device=DEV, generator=name)
    gen = TrackGenerator(spec, _rng())
    assert calls == [False]                                  # construction
    first = _bytes(gen.generate())
    assert calls == [False, False]
    assert _bytes(gen.generate()) == first
    twin = TrackGenerator(spec, _rng())
    assert _bytes(twin.generate()) == first
    assert gen.generate().valid.numpy().all()
    # enqueue_segment(i) one by one == one generation
    other = TrackGenerator(spec, _rng())
    p, ctx = other.program, LaunchContext.for_call(DEV)
    for seg in p.segments:
        p.enqueue_segment(seg.index, ctx)
    assert _bytes(other._track) == first
    with pytest.raises(IndexError, match="no segment 2"):
        p.enqueue_segment(2, ctx)


def test_masked_regeneration_of_a_host_pipeline_is_exact():
    spec = presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE)
    gen = TrackGenerator(spec, _rng())
    before = {f: getattr(gen.generate(), f).numpy().copy() for f in ("center", "valid")}
    mask = wp.array(np.array([1, 0, 1, 0, 0, 1, 0, 0], np.int32), dtype=wp.int32,
                    device=DEV)
    out = gen.generate(advance=True, mask=mask)
    twin = TrackGenerator(spec, _rng())
    want = twin.generate(advance=True)
    m = mask.numpy().astype(bool)
    N = gen.n_max
    center = out.center.numpy().reshape(E, N, 3)
    np.testing.assert_array_equal(np.nan_to_num(center[~m], nan=7),
                                  np.nan_to_num(before["center"].reshape(E, N, 3)[~m], nan=7))
    np.testing.assert_array_equal(np.nan_to_num(center[m], nan=7),
                                  np.nan_to_num(want.center.numpy().reshape(E, N, 3)[m],
                                                nan=7))
    np.testing.assert_array_equal(out.generation.numpy(), mask.numpy())


# ---------------------------------------------------------------------------------------
# Refusals before any launch
# ---------------------------------------------------------------------------------------

@pytest.fixture
def emulated_capture(monkeypatch):
    """Every LaunchContext reports capturing (a caller's capture, emulated on CPU)."""
    def start():
        monkeypatch.setattr(LaunchContext, "capturing", property(lambda self: True))
    return start


def _state(gen) -> dict:
    out = _bytes(gen._track)
    out["epoch"] = gen.seeds.epoch.numpy().tobytes()
    out["seed"] = gen.program.seed.numpy().tobytes()
    return out


def test_an_emulated_caller_capture_of_a_host_program_raises_before_any_launch(
        emulated_capture):
    calls: list = []
    name = _counted(T.HOST_CIRCLE, calls)
    gen = TrackGenerator(presets.track(num_envs=E, device=DEV, generator=name), _rng())
    gen.generate()
    before, n = _state(gen), len(calls)
    p = gen.program
    emulated_capture()
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as t:
        for call in (gen.generate, gen.enqueue, lambda: p.run(advance=True),
                     lambda: p.enqueue(ctx), lambda: p.enqueue_segment(0, ctx)):
            with pytest.raises(NotCapturableError, match=r"'toy.host_circle_counted' "
                                                         r"cannot be recorded into a graph "
                                                         r"capture: HOST stage\(s\) "
                                                         r"\['generator'\]") as info:
                call()
            assert info.value.stage_ids == ("generator",)
    assert len(t) == 0 and len(calls) == n
    assert _state(gen) == before


def test_a_real_cpu_graph_capture_of_a_host_program_raises(monkeypatch):
    calls: list = []
    name = _counted(T.HOST_CIRCLE, calls)
    gen = TrackGenerator(presets.track(num_envs=E, device=DEV, generator=name), _rng())
    before, n = _state(gen), len(calls)
    p, ctx = gen.program, LaunchContext.for_call(DEV)
    with wp.ScopedCapture(device=DEV) as cap:
        for call in (gen.generate, gen.enqueue, lambda: p.run(),
                     lambda: p.enqueue(ctx), lambda: p.enqueue_segment(0, ctx)):
            with pytest.raises(NotCapturableError, match=r"\['generator'\]"):
                call()
    assert len(calls) == n
    wp.capture_launch(cap.graph)               # the recording holds no launch
    assert _state(gen) == before and len(calls) == n


def _host_course() -> Course:
    return Course(CourseConfig(mode="track", seeds=7, collision="segments",
                               checkpoint_spacing=0.6, max_checkpoints=64,
                               gen=TrackGenConfig(generator=T.HOST_CIRCLE, num_envs=E,
                                                  device=DEV)))


def test_course_reseed_refuses_a_cpu_graph_capture_before_any_launch():
    """``Course.generate(seeds=k)`` over a HOST generator inside ``wp.ScopedCapture("cpu")``
    raises before its reseed launches: the caller's graph holds nothing of it."""
    course = _host_course()
    seeds0 = course.rng.seeds_warp.numpy().copy()
    epoch0 = course.seeds.epoch.numpy().copy()
    for call in (lambda: course.generate(seeds=123), course.regenerate,
                 course.enqueue_generate):
        with trace_launches() as t, wp.ScopedCapture(device=DEV) as cap:
            with pytest.raises(NotCapturableError, match=r"\['generator'\]"):
                call()
        assert len(t) == 0, [r.kernel for r in t]
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(course.rng.seeds_warp.numpy(), seeds0)
        np.testing.assert_array_equal(course.seeds.epoch.numpy(), epoch0)
    course.generate(seeds=123)                     # outside the capture: reseeds
    assert course.rng.seeds_warp.numpy().tolist() == list(range(123, 123 + E))


def test_construction_inside_a_cpu_graph_capture_raises_before_allocating():
    rng = _rng()
    spec = presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE)
    with wp.ScopedCapture(device=DEV) as cap:
        for make in (lambda: build(spec, seeds=3), lambda: TrackGenerator(spec, rng),
                     lambda: TrackGenerator(TrackGenConfig(device=DEV, num_envs=E), rng)):
            with no_allocation(), pytest.raises(CaptureError,
                                                match="outside any CPU graph capture"):
                make()
    wp.capture_launch(cap.graph)                   # nothing was recorded
    TrackGenerator(spec, rng).generate()           # and outside it, it builds


def test_the_host_refusal_comes_before_the_unprepared_one(emulated_capture):
    host = build(presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE),
                 seeds=3, prepare=False)
    static = build(presets.track(num_envs=E, device=DEV), seeds=3, prepare=False)
    emulated_capture()
    with pytest.raises(NotCapturableError, match=r"\['generator'\]"):
        host.run()
    with pytest.raises(CaptureError, match="prepare=False") as info:
        static.run()
    assert type(info.value) is CaptureError
    ctx = LaunchContext.for_call(DEV)
    with pytest.raises(CaptureError, match="prepare=False") as info:
        host.enqueue_segment(1, ctx)           # the tail is recordable, once prepared
    assert type(info.value) is CaptureError


def test_a_non_host_segment_is_recordable_and_carries_prologue_or_commit(
        emulated_capture):
    p = build(presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE), seeds=3)
    ref = build(presets.track(num_envs=E, device=DEV, generator=T.HOST_CIRCLE), seeds=3)
    emulated_capture()
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as t:
        p.enqueue_segment(1, ctx)              # the tail: no prologue, the commit
    kernels = [r.kernel for r in t]
    assert not any(str(k).endswith("_effective_seed_k") for k in kernels)
    assert t[-1].op == "copy"                  # the generation write closes the last one
    with trace_launches() as t_all:
        ref.enqueue_segment(1, ctx)
    assert len(t_all) == len(t)


# ---------------------------------------------------------------------------------------
# The runner's SEGMENTED mode (fake CUDA device)
# ---------------------------------------------------------------------------------------

class _Log:
    def __init__(self) -> None:
        self.events: list = []

    def part(self, name: str):
        def run(ctx):
            self.events.append((name, ctx.stream, bool(ctx.capturing)))
        return run


def _segmented(log: _Log) -> _SegmentedBody:
    return _SegmentedBody([_BodyPart(log.part("head"), host=True),
                           _BodyPart(log.part("tail"))])


def test_segmented_self_graph_captures_the_tail_once_and_runs_the_head_eagerly(
        monkeypatch):
    fake = FakeCuda(monkeypatch)
    log = _Log()
    body = _segmented(log)
    r = GraphRunner("cuda:0", capturable=False, name="toy", _host_ids=("generator",))
    r.prepare(body)
    assert [e[0] for e in log.events] == ["head", "tail"] and fake.captures == []
    del log.events[:]
    r.run(body)
    (cap,) = fake.captures
    assert cap.stream is not fake.current and cap.stream in fake.created  # private stream
    assert log.events == [("tail", cap.stream, True), ("head", fake.current, False)]
    graphs = r.segment_graphs_for()
    assert graphs[0] is None and graphs[1] is cap.graph
    assert fake.launched == [(cap.graph, fake.current)]
    assert r.graph is None and r.graph_for(None) is None and r.graph_keys == (None,)
    del log.events[:]
    r.run(body)                                          # no new capture
    assert len(fake.captures) == 1 and log.events == [("head", fake.current, False)]
    assert fake.launched == [(cap.graph, fake.current)] * 2
    key = (True, 1234)
    r.run(body, key=key)
    assert len(fake.captures) == 2 and r.segment_graphs_for(key)[1] is not cap.graph
    assert set(r.graph_keys) == {None, key} and r.graph_for(key) is None


def test_self_captures_are_marked_to_allow_conditional_nodes(monkeypatch):
    """The library's own self-capture marks its Warp registration as allowing conditional
    nodes (while it is open), so ``skip_unless`` / ``bounded_loop`` windows recorded in it
    never fall back to the torch-capture heuristic, even when the captured stream is
    torch's current stream (a legacy head records Warp's current stream)."""
    from track_gen.exec import _compat

    fake = FakeCuda(monkeypatch)
    marks = []
    monkeypatch.setattr(_compat, "registered_stream", lambda s: s)
    monkeypatch.setattr(_compat, "set_conditional_allowed",
                        lambda key, allowed: marks.append((key, allowed, key.is_capturing)))
    log = _Log()
    body = _segmented(log)
    r = GraphRunner("cuda:0", capturable=False, name="toy", _host_ids=("generator",))
    r.prepare(body)
    r.run(body)
    (cap,) = fake.captures
    assert marks == [(cap.stream, True, True)]


def test_hostops_comes_only_from_ctx_host_and_the_kit_sees_a_forged_one():
    """``HostOps`` cannot be built by hand, and a forged one in a STATIC stage does not
    get the HOST stage's sanction: the kit's ``hot_path.host_sync`` still fails."""
    from track_gen.exec import HostOps
    from track_gen.exec import host as host_mod
    from track_gen.plugin import Stage

    with pytest.raises(TypeError, match=r"ctx\.host\(\)"):
        HostOps(LaunchContext.for_call("cpu"), None)

    class Forger(Stage):
        kind = "relax"
        name = "probe.forger"
        reads = {"xy": "cl.xy"}
        writes = {"xy": "cl.xy"}

        def build(self, b, p):
            x_in = b.read("xy")
            out = b.write("xy", wp.vec2f, per="cl")
            via = wp.empty(x_in.shape, dtype=wp.vec2f, device="cpu")   # build time

            def enqueue(ctx):
                ctx.copy(out, x_in)
                h = object.__new__(HostOps)              # bypasses ctx.host()
                h._ctx, h._slot = ctx, host_mod._SLOT.get()
                h.readback(x_in, via=via)
            return enqueue

    registry.add(Forger)
    report = check_stage("relax", "probe.forger", devices=["cpu"], num_envs=4)
    sync = [r for r in report if r.name == "hot_path.host_sync"]
    assert sync and not sync[0].passed, str(report)


def test_segmented_eager_rows_and_record_refusal(monkeypatch):
    fake = FakeCuda(monkeypatch)
    log = _Log()
    body = _segmented(log)
    r = GraphRunner("cuda:0", capturable=False, name="toy", _host_ids=("generator",))
    r.prepare(body)
    del log.events[:]
    r.run(body, policy="eager")
    assert [e[0] for e in log.events] == ["head", "tail"] and fake.captures == []
    for i in range(runner_mod.MAX_KEYED_GRAPHS):
        r._seg_graphs[(True, i)] = (None, object())
    del log.events[:]
    with pytest.warns(RuntimeWarning, match="calls with a new key now run eagerly"):
        r.run(body, key=(True, 999))
    assert [e[0] for e in log.events] == ["head", "tail"] and fake.captures == []
    fake.current.is_capturing = True
    del log.events[:]
    with pytest.raises(NotCapturableError, match=r"'toy' .*\['generator'\]") as info:
        r.run(body)
    assert info.value.stage_ids == ("generator",) and log.events == []








# ---------------------------------------------------------------------------------------
# The conformance kit on the toys
# ---------------------------------------------------------------------------------------

def test_kit_leaky_smoother_fails_absorbing_and_batch_independence():
    report = check_stage("relax", T.LEAKY, devices=[DEV])
    assert report.failed_names == ["batch_independence", "bounded.absorbing"], str(report)
    msg = report.result("bounded.absorbing").message
    assert "with lowering (a) (batch A) the loop stopped at window 5/7 (active[0] == 0)" \
        in msg, msg
    assert "the outputs differ from the unrolled lowering (c): env 0 (cl.xy@relax" in msg
    assert "an absorbing freeze" in msg


def test_kit_batch_a_converges_in_several_windows():
    """The kit's batch A (seeds 11..18) converges in at least two different windows, so
    the leaky toy's batch dependence is not seed luck."""
    history: list = []

    class Recorded(T.ConvergingSmooth):
        name = "test.recorded_smooth"

        def _late(self, ctx, it, iters, active, running) -> None:
            if (it + 1) % 8 == 0:
                history.append(running.numpy().copy())

    registry.add(Recorded)
    gen = TrackGenerator(presets.track(num_envs=E, device=DEV, relax=Recorded.name),
                         _rng(11))
    from track_gen.exec.context import _force_lowering

    del history[:]
    with _force_lowering("unroll"):
        gen.generate()
    stopped = np.array(history)                       # [checks, E]
    window = [int(np.argmax(stopped[:, e] == 0)) for e in range(E)]
    assert stopped[-1].sum() == 0 and len(set(window)) >= 2, window


def test_kit_host_toys_pass_and_their_host_ops_are_sanctioned():
    """A HOST Stage waives only the topology / buffers checks: its ``ctx.host()``
    operations pass ``hot_path.host_sync`` / ``hot_path.allocation`` (sanctioned), while a
    legacy ``capturable=False`` spec (no ``ctx.host()``) also waives those two."""
    for report in (check_generator(T.HOST_CIRCLE, devices=[DEV]),
                   check_gate_generator(T.HOST_RING, devices=[DEV])):
        report.raise_for_failures()
        host = report.result("host.require_caller_capturable")
        assert host.passed and not host.skipped
        assert host.details["host_stage_ids"] == ["generator"]
        for check in ("hot_path.host_sync", "hot_path.allocation"):
            r = report.result(check)
            assert r.passed and not r.skipped, r
        assert report.result("hot_path.host_sync").details["host_ops"] > 0
        for check in ("hot_path.topology", "hot_path.buffers"):
            r = report.result(check)
            assert r.skipped and "HOST stage(s) ['generator'] run eagerly" in r.message, r
        assert report.result("bounded.absorbing").skipped
    report = check_stage("relax", T.HOST_IDENTITY, devices=[DEV])
    report.raise_for_failures()
    assert report.result("host.require_caller_capturable").details["host_stage_ids"] == \
        ["relax"]
    assert not report.result("hot_path.host_sync").skipped
    legacy = check_generator(_legacy_host_spec(), devices=[DEV])
    legacy.raise_for_failures()
    assert legacy.result("host.require_caller_capturable").passed
    for check in ("hot_path.host_sync", "hot_path.allocation", "hot_path.topology",
                  "hot_path.buffers"):
        r = legacy.result(check)
        assert r.skipped and "capturable=False" in r.message, r


def test_kit_fails_a_host_stage_that_reaches_the_host_outside_host_ops():
    report = check_stage("relax", T.SNEAKY_HOST, devices=[DEV])
    assert set(report.failed_names) == {"hot_path.host_sync", "hot_path.allocation"}, \
        str(report)
    sync = report.result("hot_path.host_sync").message
    assert "wp.array.numpy() from tests._bounded_toys.SneakyHost.build.<locals>.enqueue" \
        in sync and "wp.synchronize_device()" in sync, sync
    assert "track_gen.exec.host" not in sync          # HostOps never shows up
    alloc = report.result("hot_path.allocation").message
    assert "wp.array() from tests._bounded_toys.SneakyHost.build.<locals>.enqueue" in alloc
    assert "ctx.host() (HostOps)" in alloc, alloc




def test_warnings_are_not_raised_by_segmented_cpu_runs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gen = TrackGenerator(presets.track(num_envs=E, device=DEV, relax=T.HOST_IDENTITY),
                             _rng())
        gen.generate()
        gen.generate(advance=True, mask=gen.inputs["regen_mask"])
