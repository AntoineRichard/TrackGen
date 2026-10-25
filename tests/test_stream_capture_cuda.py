"""CUDA-only: library work lands on the stream the caller is capturing (roadmap PR2).

Two ways a caller selects the stream:
- explicitly, ``stream=s`` on a per-call method, or an enclosing ``LaunchContext`` scope;
- implicitly, inside ``torch.cuda.graph(g)``: torch's capture stream is current there and
  the library adopts it.

Each test captures on a NON-current side stream, poisons the outputs, replays the graph
and compares against an eager twin. Work that ignored the stream would run eagerly on
Warp's current stream at capture time instead of being recorded, so the replay would
leave the poison in place and the test would fail.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from track_gen import GateGenConfig, PerEnvSeededRNG, TrackGenConfig  # noqa: E402
from track_gen.plugin import registry  # noqa: E402
from track_gen._src import warp_pipeline  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen.course import Course, CourseConfig  # noqa: E402
from track_gen.exec import LaunchContext  # noqa: E402

DEV = "cuda:0"
E = 8
_TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                 "valid", "count", "winding")
_GATE_FIELDS = ("position", "tangent", "left", "right", "valid", "count")


def _poison(arr):
    # Finite sentinels: outputs are NaN-padded and assert_array_equal treats NaN == NaN,
    # so a NaN poison would hide a missed write in the padding.
    if arr.dtype is wp.int32:
        arr.fill_(-7)
    elif arr.dtype is wp.float32:
        arr.fill_(-12345.0)
    else:
        arr.fill_(arr.dtype(-12345.0))


# ---------------------------------------------------------------------------------------
# Pipelines recorded on a side stream through an explicit LaunchContext.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("generator", registry.available("generator"))
def test_track_pipeline_records_on_side_stream(generator):
    cfg = TrackGenConfig(generator=generator, device=DEV, num_envs=E)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV))
    twin = TrackGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV))
    ref = twin.generate()
    wp.synchronize()

    s = wp.Stream(DEV)
    ctx = LaunchContext(DEV, s)
    wp.copy(gen._seed_buf, gen._rng.seeds_warp)
    wp.synchronize()

    def run():
        warp_pipeline._run_pipeline(gen._config, gen._seed_buf, out=gen._track,
                                    scratch=gen._scratch,
                                    generator_spec=gen._generator_spec, ctx=ctx)

    run()                                   # warm-up on s (module load outside capture)
    wp.synchronize()
    with wp.ScopedCapture(device=DEV, stream=s) as cap:
        run()
    for f in _TRACK_FIELDS:
        _poison(getattr(gen._track, f))
    wp.synchronize()
    wp.capture_launch(cap.graph, stream=s)
    wp.synchronize()
    for f in _TRACK_FIELDS:
        np.testing.assert_array_equal(getattr(gen._track, f).numpy(),
                                      getattr(ref, f).numpy(), err_msg=f)


@pytest.mark.parametrize("generator", registry.available("gate_generator"))
def test_gate_pipeline_records_on_side_stream(generator):
    cfg = GateGenConfig(generator=generator, device=DEV, num_envs=E, gate_width=0.05)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV))
    twin = GateGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV))
    ref = twin.generate()
    wp.synchronize()

    s = wp.Stream(DEV)
    ctx = LaunchContext(DEV, s)
    wp.copy(gen._seed_buf, gen._rng.seeds_warp)
    wp.synchronize()

    def run():
        # PR11: the gate pipeline is a Program (its stages, the seed prologue and the
        # commit) emitted on ctx.
        gen.program.enqueue(ctx)

    run()
    wp.synchronize()
    with wp.ScopedCapture(device=DEV, stream=s) as cap:
        run()
    for f in _GATE_FIELDS:
        _poison(getattr(gen._gate_sequence, f))
    wp.synchronize()
    wp.capture_launch(cap.graph, stream=s)
    wp.synchronize()
    for f in _GATE_FIELDS:
        np.testing.assert_array_equal(getattr(gen._gate_sequence, f).numpy(),
                                      getattr(ref, f).numpy(), err_msg=f)


# ---------------------------------------------------------------------------------------
# Course.step(): explicit side stream, and torch.cuda.graph adoption.
# ---------------------------------------------------------------------------------------

def _course(mode, seeds=5):
    if mode == "track":
        cfg = CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=DEV),
                           seeds=seeds, collision="segments", checkpoint_spacing=0.6,
                           max_checkpoints=64)
    else:
        cfg = CourseConfig(mode="gates",
                           gen=GateGenConfig(num_envs=E, device=DEV, gate_width=0.2),
                           seeds=seeds, post_radius=0.02)
    course = Course(cfg)
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                      dtype=wp.quatf, device=DEV)
    he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
    course.bind(position=pos, orientation=orient, half_extents=he)
    course.generate()
    return course, pos


def _positions(course, step):
    n_max = course.result.center.shape[0] // E if hasattr(course.result, "center") \
        else course.result.position.shape[0] // E
    pts = (course.result.center if hasattr(course.result, "center")
           else course.result.position).numpy().reshape(E, n_max, 3)
    pts = np.nan_to_num(pts, nan=0.0)
    counts = course.result.count.numpy()
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = max(int(counts[e]), 1)
        out[e] = pts[e, (step * 3) % m]
    return out


def _events(course):
    ev = course._step_result.events
    return ev.passed.numpy().copy(), ev.progress.numpy().copy(), ev.laps.numpy().copy()


def _replay_matches_eager(course_c, pos_c, course_e, pos_e, replay, steps=10):
    for k in range(steps):
        arr = wp.array(_positions(course_e, k), dtype=wp.vec3f, device=DEV)
        wp.copy(pos_c, arr)
        wp.copy(pos_e, arr)
        wp.synchronize()
        _poison(course_c._step_result.events.passed)       # poison (dtype-aware)
        _poison(course_c._step_result.events.progress)
        wp.synchronize()
        replay()
        wp.synchronize()
        course_e.step()
        wp.synchronize()
        for a, b in zip(_events(course_c), _events(course_e)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["track", "gates"])
def test_course_step_records_on_explicit_side_stream(mode):
    course_c, pos_c = _course(mode)
    course_e, pos_e = _course(mode)
    s = wp.Stream(DEV)
    course_c.step(stream=s)                 # warm-up
    wp.synchronize()
    mask = wp.full(E, 1, dtype=wp.int32, device=DEV)
    course_c.reset(mask, stream=s)
    course_e.reset(wp.full(E, 1, dtype=wp.int32, device=DEV))
    wp.synchronize()
    with wp.ScopedCapture(device=DEV, stream=s) as cap:
        course_c.step(stream=s)
    _replay_matches_eager(course_c, pos_c, course_e, pos_e,
                          lambda: wp.capture_launch(cap.graph, stream=s))


@pytest.mark.parametrize("mode", ["track", "gates"])
def test_course_step_is_adopted_by_torch_cuda_graph(mode):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    course_c, pos_c = _course(mode)
    course_e, pos_e = _course(mode)
    course_c.step()                          # warm-up (module load outside capture)
    wp.synchronize()
    course_c.reset(wp.full(E, 1, dtype=wp.int32, device=DEV))
    course_e.reset(wp.full(E, 1, dtype=wp.int32, device=DEV))
    wp.synchronize()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        course_c.step()                      # no stream=: adopts torch's capture stream
    torch.cuda.synchronize()

    def replay():
        g.replay()
        torch.cuda.synchronize()

    _replay_matches_eager(course_c, pos_c, course_e, pos_e, replay)
