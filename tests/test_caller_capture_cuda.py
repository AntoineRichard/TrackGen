"""CUDA-only: generation recorded into a CALLER's CUDA graph (roadmap PR3).

``TrackGenerator`` / ``GateGenerator`` ``generate()`` / ``enqueue()`` and ``Course``
``regenerate()`` / ``enqueue_generate()`` / ``generate(seeds=)`` must record into the
caller's capture (a Warp ``ScopedCapture`` on any stream, or ``torch.cuda.graph``
adoption) and replay correctly, while keeping the self-graph convenience outside any
capture.

Method (as in tests/test_stream_capture_cuda.py; the helpers are the conformance kit's,
:mod:`track_gen.testing.capture`): record on a NON-current side stream,
poison the outputs with FINITE sentinels (outputs are NaN-padded and
``assert_array_equal`` treats NaN == NaN, so a NaN poison would hide a missed write),
replay, and compare with an eager twin. Every recording also runs with

- the Warp mempool disabled, so any allocation inside the capture fails loudly
  (Warp's default CUDA allocator cannot allocate while a capture is active);
- ``wp.capture_launch`` patched to raise: a graph launch cannot be recorded into
  another capture (CUDA error 900), so a facade must re-enqueue its work.

Construction prepares every producer (one real eager pass), so every kernel the hot
path uses is loaded outside the captures; ``step()`` / ``reset()`` are warmed up
explicitly before they are recorded.
"""
from __future__ import annotations

import contextlib
import gc
import threading
import weakref

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

import track_gen.exec as tg_exec  # noqa: E402
from track_gen import GateGenConfig, PerEnvSeededRNG, TrackGenConfig  # noqa: E402
from track_gen.plugin import CaptureClass, GeneratorSpec, registry  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen.course import Course, CourseConfig  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402

DEV = "cuda:0"
E = 8
_KINDS = ("track", "gate")
_COURSE_VARIANTS = ("track-segments", "track-sdf-heightfield", "gates-posts", "gates-frames")


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _arrays(obj) -> dict:
    """name -> wp.array for every array field of a result dataclass (None fields skipped)."""
    return tg_capture.result_arrays(obj)


# Finite, dtype-aware sentinel on Warp's current stream; all arrays of the given results,
# then a device sync.
_poison = tg_capture.poison
_poison_all = tg_capture.poison_all


def _snapshot(obj) -> dict:
    return {k: a.numpy().copy() for k, a in _arrays(obj).items()}


def _assert_same(got, ref, label: str) -> None:
    ref_arrays = _arrays(ref)
    assert ref_arrays, label
    for name, arr in ref_arrays.items():
        np.testing.assert_array_equal(getattr(got, name).numpy(), arr.numpy(),
                                      err_msg=f"{label}: {name}")


def _differs(a, b) -> bool:
    return any(not np.array_equal(x.numpy(), getattr(b, k).numpy(), equal_nan=True)
               for k, x in _arrays(a).items())


def _mempool_disabled(device=DEV):
    """The Warp mempool disabled on ``device`` for the block."""
    return tg_capture.mempool_disabled(device)


# ``wp.capture_launch`` (also track_gen.exec.runner's reference) raises for the block,
# and the block fails on exit if it was called (a swallowed AssertionError included).
_capture_launch_forbidden = tg_capture.capture_launch_forbidden


def _recording(stream):
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch forbidden.

    Context managers exit in reverse order: the patch is lifted before ``capture_end``
    and the mempool is restored after it.
    """
    return tg_capture.recording(DEV, stream)


def _side_stream() -> wp.Stream:
    return tg_capture.side_stream(DEV)


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _seed_array(base: int) -> wp.array:
    return wp.array((base + np.arange(E)).astype(np.int32), dtype=wp.int32, device=DEV)


def _reseed(rng, base: int) -> None:
    """Reseed outside any capture (same expansion as ``PerEnvSeededRNG(seeds=base)``)."""
    seeds = _seed_array(base)
    rng.set_seeds_warp(seeds, None)
    wp.synchronize()


def _marker():
    """A tiny copy the caller records itself, so its graph is never empty."""
    src = wp.array(np.array([1, 2, 3], np.int32), dtype=wp.int32, device=DEV)
    dst = wp.zeros(3, dtype=wp.int32, device=DEV)
    wp.synchronize()
    return dst, src


def _make(kind: str, seeds=11, *, generator=None, stream=None):
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV)
    kw = {} if generator is None else {"generator": generator}
    extra = {} if stream is None else {"stream": stream}
    if kind == "track":
        gen = TrackGenerator(TrackGenConfig(device=DEV, num_envs=E, **kw), rng, **extra)
    else:
        gen = GateGenerator(GateGenConfig(device=DEV, num_envs=E, gate_width=0.05, **kw),
                            rng, **extra)
    return gen, rng


def _eager_ref(kind: str, seeds: int, *, generator=None):
    """A fresh twin built with ``seeds``, run eagerly (``enqueue`` never self-graphs)."""
    twin, _ = _make(kind, seeds, generator=generator)
    out = twin.enqueue()
    wp.synchronize()
    assert twin.graph is None, "enqueue() must never self-graph"
    return out


def _counting(base_generate, calls: list, *, host_readback: bool = False):
    """Wrap a shipped generator body: count Python body runs (optionally host-sync)."""
    def generate(*args, **kwargs):
        calls.append(1)
        if host_readback:
            # A blocking host readback: what makes a generator non-capturable.
            args[0].numpy()
        return base_generate(*args, **kwargs)
    return generate


# ---------------------------------------------------------------------------------------
# Generator facades: record into the caller's Warp capture on a side stream
# ---------------------------------------------------------------------------------------

def _record_replay_reseed(kind: str, generator: str) -> None:
    gen, rng = _make(kind, 11, generator=generator)
    ref = _eager_ref(kind, 11, generator=generator)
    s = _side_stream()
    wp.synchronize()

    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert gen.graph is None, "recording into the caller's graph must not self-capture"

    # Replay into poisoned outputs: the caller's graph regenerates the whole batch.
    _poison_all(out)
    _replay(cap.graph, s)
    _assert_same(out, ref, f"{kind}/{generator}: replay")

    # Reseed OUTSIDE the graph: the seed copy is part of the recording, so the same
    # graph must now produce the batch of a fresh twin built with the new seeds.
    new = _eager_ref(kind, 101, generator=generator)
    assert _differs(ref, new), "reseeding must change the batch (vacuous test otherwise)"
    _reseed(rng, 101)
    _poison_all(out)
    _replay(cap.graph, s)
    _assert_same(out, new, f"{kind}/{generator}: replay after reseed")


@pytest.mark.parametrize("generator", registry.available("generator"))
def test_track_generate_records_into_caller_capture(generator):
    """Every registered track generator: ``generate(stream=s)`` inside the caller's
    capture is recorded (no allocation, no capture_launch, no self-capture), replays to
    the eager twin, and the replay reads the rng's live seeds."""
    if not registry.describe("generator", generator).capturable:
        pytest.skip(f"{generator!r} is registered capturable=False")
    _record_replay_reseed("track", generator)


@pytest.mark.parametrize("generator", registry.available("gate_generator"))
def test_gate_generate_records_into_caller_capture(generator):
    """Every registered gate generator: as the track test."""
    if not registry.describe("gate_generator", generator).capturable:
        pytest.skip(f"{generator!r} is registered capturable=False")
    _record_replay_reseed("gate", generator)


@pytest.mark.parametrize("kind", _KINDS)
def test_enqueue_records_into_caller_capture_and_never_self_graphs(kind):
    """``enqueue()`` runs eagerly outside a capture (no graph), and is recorded into the
    caller's capture inside one."""
    gen, _ = _make(kind, 11)
    ref = _eager_ref(kind, 11)
    out = gen.enqueue()
    wp.synchronize()
    assert gen.graph is None
    _assert_same(out, ref, f"{kind}: eager enqueue")

    s = _side_stream()
    with _recording(s) as cap:
        assert gen.enqueue(stream=s) is out
    assert gen.graph is None
    _poison_all(out)
    _replay(cap.graph, s)
    _assert_same(out, ref, f"{kind}: recorded enqueue")


@pytest.mark.parametrize("kind", _KINDS)
def test_generate_is_adopted_by_torch_cuda_graph(kind):
    """Inside ``torch.cuda.graph(g)``, ``generate()`` without ``stream=`` adopts torch's
    capture stream and records into ``g``; replays read the rng's live seeds."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    gen, rng = _make(kind, 11)
    ref = _eager_ref(kind, 11)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with _mempool_disabled():
        with torch.cuda.graph(g):
            with _capture_launch_forbidden():
                out = gen.generate()         # no stream=: adopts torch's capture stream
    torch.cuda.synchronize()
    assert gen.graph is None

    def replay():
        g.replay()
        torch.cuda.synchronize()

    _poison_all(out)
    replay()
    _assert_same(out, ref, f"{kind}: torch replay")

    new = _eager_ref(kind, 101)
    _reseed(rng, 101)
    _poison_all(out)
    replay()
    _assert_same(out, new, f"{kind}: torch replay after reseed")


@pytest.mark.parametrize("kind", _KINDS)
def test_constructor_stream_is_recorded_when_capturing(kind):
    """A generator built with ``stream=s`` records into a capture on ``s`` when called
    with no ``stream=`` (the constructor stream is the call's default stream)."""
    s = _side_stream()
    gen, _ = _make(kind, 11, stream=s)       # prepared on s
    ref = _eager_ref(kind, 11)
    with _recording(s) as cap:
        out = gen.generate()
    assert gen.graph is None
    _poison_all(out)
    _replay(cap.graph, s)
    _assert_same(out, ref, f"{kind}: constructor-stream recording")


# ---------------------------------------------------------------------------------------
# Self-graph path (outside any capture) and Python body counts
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _KINDS)
def test_self_graph_outside_any_capture_matches_eager_twin(kind):
    """``graph`` is None after construction, set by the first CUDA ``generate()``, then
    captured once and replayed (on any stream, sync or not) to the eager twin's batch."""
    gen, rng = _make(kind, 11)
    assert gen.graph is None, "construction prepares eagerly; no graph yet"
    out = gen.generate()
    g = gen.graph
    assert g is not None
    _assert_same(out, _eager_ref(kind, 11), f"{kind}: first self-graph call")

    s = _side_stream()
    for i, base in enumerate((101, 202, 303)):
        ref = _eager_ref(kind, base)
        _reseed(rng, base)
        _poison_all(out)
        if i == 0:
            res = gen.generate()
        elif i == 1:
            res = gen.generate(stream=s, sync=False)   # stream-ordered, not blocking
            wp.synchronize()
        else:
            res = gen.generate(stream=s)               # replayed on a side stream
        assert res is out
        assert gen.graph is g, "the self-graph is captured once, then replayed"
        _assert_same(out, ref, f"{kind}: self-graph replay {i}")


def test_python_body_runs_once_per_recording_and_never_on_replay():
    """prepare runs the body once; the self-graph records it once; self-graph replays
    run no Python; a caller recording made AFTER the self-graph exists re-enqueues the
    body (never launches the stored graph) and the caller's replays run no Python."""
    calls = []
    base = registry.get("generator", "polar")
    registry.add(GeneratorSpec(name="toy.count", alloc_scratch=base.alloc_scratch,
                               generate=_counting(base.generate, calls),
                               capturable=True, accepts_ctx=True))
    gen, _ = _make("track", 11, generator="toy.count")
    ref = _eager_ref("track", 11, generator="polar")
    assert len(calls) == 1, "construction runs exactly one real pass"
    assert gen.graph is None

    out = gen.generate()
    assert len(calls) == 2, "the first CUDA generate() records the self-graph once"
    g = gen.graph
    assert g is not None
    _assert_same(out, ref, "toy.count: first call")
    for _ in range(3):
        gen.generate()
    assert len(calls) == 2, "self-graph replays must not run the Python body"
    assert gen.graph is g

    s = _side_stream()
    with _recording(s) as cap:
        assert gen.generate(stream=s) is out
    assert len(calls) == 3, "a caller recording runs the body exactly once"
    assert gen.graph is g
    for k in range(2):
        _poison_all(out)
        _replay(cap.graph, s)
        _assert_same(out, ref, f"toy.count: caller replay {k}")
    assert len(calls) == 3, "caller replays must not run the Python body"


# ---------------------------------------------------------------------------------------
# Non-capturable and legacy-plugin generators
# ---------------------------------------------------------------------------------------

def _register_host_toys(calls: list) -> None:
    """Register a non-capturable "toy.host" track AND gate generator (wrapping polar /
    bezier, so their outputs equal those shipped generators); undone by the per-test
    registry scope (tests/conftest.py)."""
    tbase = registry.get("generator", "polar")
    registry.add(GeneratorSpec(name="toy.host", alloc_scratch=tbase.alloc_scratch,
                               generate=_counting(tbase.generate, calls, host_readback=True),
                               capturable=False, accepts_ctx=True))
    gbase = registry.get("gate_generator", "bezier")   # an anchors Stage (roadmap PR11)

    class HostGates(gbase):
        name = "toy.host"
        capture = CaptureClass.HOST

        def build(self, b, p):
            body = super().build(b, p)
            seed = b.seed

            def enqueue(ctx):
                calls.append(1)
                seed.numpy()   # a blocking host readback: what makes it non-capturable
                body(ctx)
            return enqueue

    registry.add(HostGates)


@pytest.mark.parametrize("kind", _KINDS)
def test_non_capturable_generator_refuses_caller_capture_before_any_launch(kind):
    """A capturable=False generator runs eagerly outside a capture (never self-graphs);
    inside one, ``generate()`` / ``enqueue()`` raise NotCapturableError naming it before
    any launch: its body never runs, and replaying the caller's graph runs only the
    caller's own work (outputs and seed buffer untouched)."""
    calls = []
    _register_host_toys(calls)
    gen, _ = _make(kind, 11, generator="toy.host")
    assert len(calls) == 1, "construction runs exactly one real pass"
    ref = _eager_ref(kind, 11, generator="polar" if kind == "track" else "bezier")

    out = gen.generate()
    wp.synchronize()
    assert len(calls) == 2, "outside a capture the body runs eagerly once per call"
    assert gen.graph is None, "a non-capturable generator never self-graphs"
    _assert_same(out, ref, f"{kind}: eager toy.host")

    s = _side_stream()
    marker, marker_src = _marker()
    with _recording(s) as cap:
        wp.copy(marker, marker_src, stream=s)     # the caller's own recorded work
        with pytest.raises(tg_exec.NotCapturableError, match="toy.host"):
            gen.generate(stream=s)
        with pytest.raises(tg_exec.NotCapturableError, match="toy.host"):
            gen.enqueue(stream=s)
    assert len(calls) == 2, "the generator body ran under the caller's capture"
    assert gen.graph is None

    _poison_all(out)
    poisoned = _snapshot(out)
    _replay(cap.graph, s)                         # the capture stayed valid
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    for name, arr in poisoned.items():
        np.testing.assert_array_equal(getattr(out, name).numpy(), arr,
                                      err_msg=f"{kind}: {name} written by the caller's graph")


def test_course_with_non_capturable_generator_refuses_caller_capture():
    """Course over a capturable=False generator: ``regenerate()`` /
    ``enqueue_generate()`` / ``generate(seeds=)`` raise NotCapturableError under a
    capture before any launch (not even the reseed is recorded); outside a capture
    ``generate(seeds=)`` runs eagerly and never self-graphs."""
    calls = []
    _register_host_toys(calls)

    def cfg(generator, seeds):
        return CourseConfig(mode="track",
                            gen=TrackGenConfig(generator=generator, num_envs=E, device=DEV),
                            seeds=seeds, collision="segments", checkpoint_spacing=0.6,
                            max_checkpoints=64)

    course = Course(cfg("toy.host", 11))
    n = len(calls)
    assert n >= 1
    s = _side_stream()
    marker, marker_src = _marker()
    with _recording(s) as cap:
        wp.copy(marker, marker_src, stream=s)
        with pytest.raises(tg_exec.NotCapturableError, match="toy.host"):
            course.regenerate(stream=s)
        with pytest.raises(tg_exec.NotCapturableError, match="toy.host"):
            course.enqueue_generate(stream=s)
        with pytest.raises(tg_exec.NotCapturableError, match="toy.host"):
            course.generate(seeds=3, stream=s)
    assert len(calls) == n, "the generator body ran under the caller's capture"
    assert course.graph is None

    _poison_all(course.result)
    poisoned = _snapshot(course.result)
    _replay(cap.graph, s)
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    np.testing.assert_array_equal(course.rng.seeds_warp.numpy(), 11 + np.arange(E),
                                  err_msg="the int reseed was recorded before the raise")
    for name, arr in poisoned.items():
        np.testing.assert_array_equal(getattr(course.result, name).numpy(), arr,
                                      err_msg=f"result.{name} written by the caller's graph")

    res = course.generate(seeds=3)
    assert len(calls) == n + 1
    assert course.graph is None, "a non-capturable course never self-graphs"
    _assert_same(res, Course(cfg("polar", 3)).result, "course over toy.host, seeds=3")


@wp.kernel
def _copy_seeds_k(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    e = wp.tid()
    dst[e] = src[e]


def _register_legacy_toy(calls: list) -> None:
    """Register "toy.legacy": a legacy-ABI track plugin (no accepts_ctx) that, like a real
    one, issues its OWN ``wp.launch`` with no ``stream=`` (so on Warp's CURRENT stream):
    it copies the seeds into a private buffer, then runs polar on that copy (polar's
    helpers follow the ambient context the orchestrator sets). Its output therefore
    equals polar's only when that stream-less launch is ordered with the rest of the
    body. Undone by the per-test registry scope (tests/conftest.py)."""
    base = registry.get("generator", "polar")

    def alloc_scratch(config):
        seeds = wp.zeros(int(config.num_envs), dtype=wp.int32, device=str(config.device))
        return base.alloc_scratch(config), seeds

    def legacy_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch):
        calls.append(1)
        base_scratch, seed_copy = scratch
        wp.launch(_copy_seeds_k, dim=seeds_wp.shape[0], inputs=[seeds_wp, seed_copy],
                  device=seeds_wp.device)             # no stream=: Warp's current stream
        base.generate(seed_copy, config, out_centerline=out_centerline,
                      out_valid_wp=out_valid_wp, scratch=base_scratch)

    registry.add(GeneratorSpec(name="toy.legacy", alloc_scratch=alloc_scratch,
                               generate=legacy_generate))


def test_legacy_plugin_records_only_on_warps_current_stream():
    """A legacy-ABI plugin (no accepts_ctx) launches on Warp's current stream, so
    recording it on another capturing stream raises LegacyPluginStreamError before any
    launch; with the capture stream made current it records and replays correctly."""
    calls = []
    _register_legacy_toy(calls)
    gen, _ = _make("track", 11, generator="toy.legacy")
    ref = _eager_ref("track", 11, generator="polar")
    out = gen.enqueue()
    wp.synchronize()
    _assert_same(out, ref, "legacy plugin: eager")
    n = len(calls)

    s = _side_stream()
    marker, marker_src = _marker()
    with _recording(s) as refused:
        wp.copy(marker, marker_src, stream=s)
        with pytest.raises(tg_exec.LegacyPluginStreamError):
            gen.generate(stream=s)
    assert len(calls) == n, "the legacy body ran on a stream it cannot follow"
    _poison_all(out)
    poisoned = _snapshot(out)
    _replay(refused.graph, s)
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    for name, arr in poisoned.items():
        np.testing.assert_array_equal(getattr(out, name).numpy(), arr, err_msg=name)

    with wp.ScopedStream(s):                      # the capture stream is now current
        with _recording(s) as cap:
            assert gen.generate() is out
    assert len(calls) == n + 1
    _poison_all(out)
    _replay(cap.graph, s)
    _assert_same(out, ref, "legacy plugin: recorded on the current stream")

    out2 = gen.generate()                         # legacy self-graph (current stream)
    assert gen.graph is not None
    _assert_same(out2, ref, "legacy plugin: self-graph")


def test_legacy_plugin_off_the_current_stream_outside_a_capture():
    """Outside a capture, a legacy plugin whose call resolves a side stream (constructor
    ``stream=``, explicit ``stream=``) must still have its own stream-less launches
    ordered with the library's stages: the whole body runs on Warp's current stream,
    bracketed by events with the side stream (construction, eager ``enqueue()``), and
    the self-graph replays on the side stream."""
    calls = []
    _register_legacy_toy(calls)
    s = _side_stream()
    gen, rng = _make("track", 11, generator="toy.legacy", stream=s)   # prepared "on" s
    assert len(calls) == 1
    _assert_same(gen._track, _eager_ref("track", 11, generator="polar"),
                 "legacy plugin: construction with stream=side")
    for i, base in enumerate((101, 202, 303, 404)):
        ref = _eager_ref("track", base, generator="polar")
        _reseed(rng, base)
        _poison_all(gen._track)
        if i == 0:
            out = gen.enqueue()                   # eager on the constructor stream
        elif i == 1:
            out = gen.enqueue(stream=s)           # eager, explicit side stream
        elif i == 2:
            out = gen.generate()                  # self-graph captured on the current stream
            assert gen.graph is not None
        else:
            out = gen.generate(stream=s, sync=False)   # replayed on the side stream
        wp.synchronize()
        _assert_same(out, ref, f"legacy plugin off the current stream: call {i}")


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    return torch


@pytest.mark.parametrize("generator", ["polar", "toy.legacy"])
def test_documented_torch_recipe_leaves_the_wrapper_registered(generator):
    """The documented recipe: Warp's current stream = ``wp.stream_from_torch(s)``, then
    ``torch.cuda.graph(g, stream=s)``. Adoption must return THAT wrapper, never a
    throwaway one whose collection would unregister the handle process-wide (Warp's
    native current stream would turn NULL and a later Warp capture on the wrapper would
    fail with "unknown stream"). A second, Warp capture on the same wrapper must work."""
    torch = _torch_cuda()
    if generator == "toy.legacy":
        _register_legacy_toy([])
    gen, _ = _make("track", 11, generator=generator)
    ref = _eager_ref("track", 11, generator="polar")
    ts = torch.cuda.Stream(device=DEV)
    ws = wp.stream_from_torch(ts)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with wp.ScopedStream(ws):
        with _mempool_disabled():
            with torch.cuda.graph(g, stream=ts):
                with _capture_launch_forbidden():
                    out = gen.generate()          # adopted: Warp's current stream itself
        with _recording(ws) as cap:               # a second capture on the same wrapper
            assert gen.generate(stream=ws) is out
    torch.cuda.synchronize()
    assert gen.graph is None

    _poison_all(out)
    g.replay()
    torch.cuda.synchronize()
    _assert_same(out, ref, f"{generator}: torch recipe replay")
    _poison_all(out)
    _replay(cap.graph, ws)
    _assert_same(out, ref, f"{generator}: second capture on the same wrapper")


@pytest.mark.parametrize("current", [True, False], ids=["warp-current", "warp-not-current"])
def test_warp_capture_on_a_torch_current_stream(current):
    """A plain Warp capture on ``wp.stream_from_torch(ts)`` while ``ts`` is torch's current
    stream: torch reports it capturing, so ``generate()`` without ``stream=`` adopts it.
    It must reuse the capturing wrapper (Warp's current stream, or the stream the open
    capture was started on), so ``capture_end`` still finds the stream registered."""
    torch = _torch_cuda()
    gen, _ = _make("track", 11)
    ref = _eager_ref("track", 11)
    ts = torch.cuda.Stream(device=DEV)
    ws = wp.stream_from_torch(ts)
    torch.cuda.synchronize()
    scope = wp.ScopedStream(ws) if current else contextlib.nullcontext()
    with scope, torch.cuda.stream(ts):
        with _recording(ws) as cap:
            out = gen.generate()
    assert gen.graph is None
    _poison_all(out)
    _replay(cap.graph, ws)
    _assert_same(out, ref, "Warp capture on a torch-current stream")


# ---------------------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------------------

def _course_cfg(variant: str, seeds) -> CourseConfig:
    if variant == "track-segments":
        return CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=DEV),
                            seeds=seeds, collision="segments", checkpoint_spacing=0.6,
                            max_checkpoints=64)
    if variant == "track-sdf-heightfield":
        return CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=DEV),
                            seeds=seeds, collision="sdf", sdf_resolution=32,
                            heightfield_resolution=16, checkpoint_spacing=0.6,
                            max_checkpoints=64)
    if variant == "gates-posts":
        return CourseConfig(mode="gates",
                            gen=GateGenConfig(num_envs=E, device=DEV, gate_width=0.2),
                            seeds=seeds, post_radius=0.02)
    assert variant == "gates-frames", variant
    # 3D gates; scale=3.0 keeps 0.2-wide gates valid (see tests/test_course_gates_3d.py).
    gcfg = GateGenConfig(num_envs=E, device=DEV, gate_width=0.2, scale=3.0,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    return CourseConfig(mode="gates", gen=gcfg, seeds=seeds, frame_collision=True,
                        agent_radius=0.05, frame_thickness=0.05, frame_depth=0.05)


def _bound_course(variant: str, seeds=5):
    course = Course(_course_cfg(variant, seeds))
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    if variant == "gates-frames":
        course.bind(pos)
    else:
        orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                          dtype=wp.quatf, device=DEV)
        he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
        course.bind(position=pos, orientation=orient, half_extents=he)
    wp.synchronize()
    return course, pos


def _positions(course, k: int) -> np.ndarray:
    """[E, 3] finite agent positions on the course's current batch for step ``k``."""
    res = course.result
    pts_arr = res.center if hasattr(res, "center") else res.position
    n_max = pts_arr.shape[0] // E
    pts = np.nan_to_num(pts_arr.numpy().reshape(E, n_max, 3), nan=0.0)
    counts = res.count.numpy()
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = max(min(int(counts[e]), n_max), 1)
        out[e] = pts[e, (k * 3) % m]
    return out


def _course_arrays(course, step_result=None) -> dict:
    """Every buffer one regeneration (+ step + reset) rewrites, by name."""
    out = {}
    objs = [("result", course.result), ("checkpoints", course.checkpoints)]
    if step_result is not None:
        objs += [("events", step_result.events), ("contacts", step_result.contacts),
                 ("frame", step_result.frame)]
    if course.heightfield is not None:
        objs.append(("heightfield", course.heightfield._hf))
    for label, obj in objs:
        for k, a in _arrays(obj).items():
            out[f"{label}.{k}"] = a
    p = course.progress
    for k in ("_prev_pos", "_next", "_laps", "_progress"):   # fully reset by a regeneration
        out[f"progress.{k}"] = getattr(p, k)
    if course._posts is not None:
        out["posts"] = course._posts
    return out


def _assert_course_same(got: dict, ref: dict, label: str) -> None:
    assert got.keys() == ref.keys(), label
    for name in ref:
        np.testing.assert_array_equal(got[name].numpy(), ref[name].numpy(),
                                      err_msg=f"{label}: {name}")


@pytest.mark.parametrize("variant", _COURSE_VARIANTS)
def test_course_regenerate_step_reset_in_one_caller_graph(variant):
    """ONE caller graph holds ``regenerate()`` + ``step()`` + ``reset(mask)``; every
    replay (new seeds, positions and mask written outside the graph) equals an eager
    twin doing the same calls: generation, refresh (checkpoints / sdf + heightfield bake
    / posts / line), progress reset, step events, contacts, frame and masked reset."""
    course, pos_c = _bound_course(variant, seeds=5)
    twin, pos_t = _bound_course(variant, seeds=5)
    mask_c = wp.zeros(E, dtype=wp.int32, device=DEV)
    mask_t = wp.zeros(E, dtype=wp.int32, device=DEV)
    s = _side_stream()
    # Warm-up outside the capture (generation was prepared at construction).
    course.step(stream=s)
    course.reset(mask_c, stream=s)
    twin.step()
    twin.reset(mask_t)
    wp.synchronize()

    with _recording(s) as cap:
        res = course.regenerate(stream=s)
        sr_c = course.step(stream=s)
        course.reset(mask_c, stream=s)
    assert res is course.result
    assert course.graph is None, "recording must not self-capture"
    assert course.generator.graph is None

    for k in range(4):
        if k:   # vary the batch: the graph reads the rng's seeds at replay time
            _reseed(course.rng, 100 * k)
            _reseed(twin.rng, 100 * k)
        twin.enqueue_generate()
        wp.synchronize()
        p = wp.array(_positions(twin, k), dtype=wp.vec3f, device=DEV)
        m = wp.array(((np.arange(E) + k) % 2).astype(np.int32), dtype=wp.int32, device=DEV)
        for dst, src in ((pos_c, p), (pos_t, p), (mask_c, m), (mask_t, m)):
            wp.copy(dst, src)
        wp.synchronize()
        sr_t = twin.step()
        twin.reset(mask_t)
        wp.synchronize()

        got = _course_arrays(course, sr_c)
        for arr in got.values():
            _poison(arr)
        wp.synchronize()
        _replay(cap.graph, s)
        _assert_course_same(got, _course_arrays(twin, sr_t), f"{variant} replay {k}")
    assert twin.graph is None, "enqueue_generate() must never self-graph"


@pytest.mark.parametrize("seed_kind", ["int", "array"])
@pytest.mark.parametrize("variant", ["track-segments", "gates-posts"])
def test_course_generate_with_seeds_records_into_caller_capture(variant, seed_kind):
    """``generate(seeds=...)`` inside a capture records the reseed (no allocation) and
    the regeneration on the same stream. An int is baked into the graph (every replay
    reseeds with it); a wp.array is read at replay time. Each replay equals a fresh
    Course built with the expected seeds."""
    course, _ = _bound_course(variant, seeds=5)
    s = _side_stream()
    seed_buf = _seed_array(77)
    wp.synchronize()
    with _recording(s) as cap:
        res = course.generate(seeds=77 if seed_kind == "int" else seed_buf, stream=s)
    assert res is course.result
    assert course.graph is None
    np.testing.assert_array_equal(course.rng.seeds_warp.numpy(), 5 + np.arange(E),
                                  err_msg="the recording executed the reseed")

    for base in (91, 123):
        if seed_kind == "array":
            wp.copy(seed_buf, _seed_array(base))  # read by the next replay
            expected = base
        else:
            _reseed(course.rng, base)              # overwritten by the baked int
            expected = 77
        wp.synchronize()
        got = _course_arrays(course)
        for arr in got.values():
            _poison(arr)
        wp.synchronize()
        _replay(cap.graph, s)
        np.testing.assert_array_equal(course.rng.seeds_warp.numpy(), expected + np.arange(E))
        fresh = Course(_course_cfg(variant, expected))
        wp.synchronize()
        _assert_course_same(got, _course_arrays(fresh),
                            f"{variant}/{seed_kind}: replay with seeds {expected}")


@pytest.mark.parametrize("variant", ["track-segments", "gates-posts"])
def test_course_regenerate_self_graph_matches_eager_twin(variant):
    """Outside a capture ``regenerate()`` captures ONE graph (generator + refresh: the
    generator keeps no graph of its own) on the first CUDA call and replays it,
    ``generate(seeds=)`` reuses it, and every call equals an eager twin."""
    course, _ = _bound_course(variant, seeds=5)
    twin, _ = _bound_course(variant, seeds=5)
    assert course.graph is None
    course.regenerate()
    g = course.graph
    assert g is not None
    assert course.generator.graph is None, "generator + refresh must be ONE graph"
    twin.enqueue_generate()
    wp.synchronize()
    _assert_course_same(_course_arrays(course), _course_arrays(twin), f"{variant}: first")

    s = _side_stream()
    for i, base in enumerate((31, 47, 63)):
        got = _course_arrays(course)
        if i < 2:
            _reseed(course.rng, base)
        for arr in got.values():
            _poison(arr)
        wp.synchronize()
        if i == 0:
            res = course.regenerate()
        elif i == 1:
            res = course.regenerate(stream=s, sync=False)
            wp.synchronize()
        else:
            res = course.generate(seeds=base)
        assert res is course.result
        assert course.graph is g, "the course graph is captured once, then replayed"
        _reseed(twin.rng, base)
        twin.enqueue_generate()
        wp.synchronize()
        _assert_course_same(got, _course_arrays(twin), f"{variant}: self-graph call {i}")
    assert twin.graph is None


# ---------------------------------------------------------------------------------------
# Lifetime and lock-free concurrency
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["generator", "course"])
def test_caller_graph_keeps_recorded_producer_alive(target):
    """A Warp graph keeps only module execs alive, not buffers: recording anchors the
    producer on the caller's registered graph, so dropping every other reference
    neither frees its scratch nor breaks the replay."""
    s = _side_stream()
    if target == "generator":
        producer, _ = _make("track", 11)
        ref = _eager_ref("track", 11)
        with _recording(s) as cap:
            arrays = _arrays(producer.generate(stream=s))
    else:
        producer, _ = _bound_course("track-segments", seeds=5)
        ref = Course(_course_cfg("track-segments", 5)).result
        wp.synchronize()
        with _recording(s) as cap:
            arrays = _arrays(producer.regenerate(stream=s))
    alive = weakref.ref(producer)
    del producer
    gc.collect()
    assert alive() is not None, "the caller's wp.Graph must keep the recorded producer alive"

    for a in arrays.values():
        _poison(a)
    wp.synchronize()
    _replay(cap.graph, s)
    for name, a in _arrays(ref).items():
        np.testing.assert_array_equal(arrays[name].numpy(), a.numpy(), err_msg=name)


def test_generate_blocks_while_another_thread_holds_a_capture():
    """No process-wide lock, no device-wide sync: while thread B holds a Warp capture
    open (Warp then reports the device as capturing, and wp.synchronize* raise from any
    thread), thread A's generate() self-captures, replays and blocks on its own stream.
    Thread A must not touch the legacy default stream meanwhile (``.numpy()`` does, and
    would invalidate B's capture), so every readback happens after B ends."""
    gen_a, _ = _make("track", 11)
    gen_b, _ = _make("track", 11)
    ref = _eager_ref("track", 11)
    s_b = _side_stream()
    wp.synchronize()

    recorded, release = threading.Event(), threading.Event()
    box = {}

    def capture_in_b():
        try:
            with wp.ScopedCapture(device=DEV, stream=s_b) as cap:
                box["out"] = gen_b.generate(stream=s_b)
                recorded.set()
                if not release.wait(timeout=120):
                    raise TimeoutError("thread A never released the capture")
            box["graph"] = cap.graph
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            box["error"] = exc
        finally:
            recorded.set()

    t = threading.Thread(target=capture_in_b, daemon=True)
    t.start()
    try:
        assert recorded.wait(timeout=120), "thread B never recorded"
        assert "error" not in box, box.get("error")
        assert wp.get_device(DEV).is_capturing, "thread B's capture is registered device-wide"
        out_a = gen_a.generate()     # self-capture (first call) + replay + blocking sync
        assert gen_a.graph is not None
        assert gen_a.generate() is out_a   # replay + blocking sync
    finally:
        release.set()
        t.join(timeout=120)
    assert not t.is_alive()
    assert "error" not in box, box.get("error")

    wp.synchronize()
    _assert_same(out_a, ref, "thread A (self-graph during B's capture)")
    out_b = box["out"]
    _poison_all(out_b)
    _replay(box["graph"], s_b)
    _assert_same(out_b, ref, "thread B (recorded alongside A)")
