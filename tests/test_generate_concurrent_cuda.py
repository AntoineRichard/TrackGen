"""CUDA-only: concurrent generate() calls must not corrupt graph capture.

``TrackGenerator.generate()`` / ``GateGenerator.generate()`` / ``Course.generate()`` on
cuda self-capture their graph on the first call and replay it afterwards. Unsynchronized
concurrent calls (e.g. the param explorer's two ``app.load`` events firing on one page
load) used to race the capture window on the device's shared stream, yielding
"Cannot synchronize device ... while graph capture is active", CUDA errors 401/900 at
``wp_cuda_graph_begin_capture`` / ``wp_cuda_graph_launch``, or an async illegal-access
700 that poisons the context. There is no process-wide lock any more: each producer
captures on its OWN private stream, replays on the call's stream, and ``sync=True``
blocks on that one stream with a raw stream synchronize (legal while another thread
captures, unlike ``wp.synchronize*``, which raise while any Warp capture is registered on
the device). This test drives the original failure mode.

Each scenario spawns threads that generate concurrently on cuda:0 with FRESH generator
instances (construction = warm-up pass; first generate = capture + replay, the racy
path) and asserts every thread completes without raising and reproduces the
single-threaded reference valid-count for the same seed (generation is deterministic
under a fixed rng).

The worker threads do no host readback and no device-wide sync while their peers may be
capturing: reading back through the legacy default stream (``wp.array.numpy()``, torch
on its default stream) while ANY blocking stream of the context is capturing is illegal
in CUDA and invalidates that capture, whoever started it. Results are read back by the
main thread after every worker joined.

Capture needs a real GPU, so the whole module is skipped without CUDA.
"""
from __future__ import annotations

import threading

import pytest

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
]

import warp as wp  # noqa: E402
from tests._warp_compare import to_t  # noqa: E402
from track_gen._src.course import Course, CourseConfig  # noqa: E402
from track_gen.exec import _compat  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.types import GateGenConfig, TrackGenConfig  # noqa: E402

_E = 256
_DEV = "cuda:0"


def _valid_count(result) -> int:
    """Host readback; main thread only, after every worker joined."""
    return int(to_t(result.valid).sum().item())


def _gen_tracks():
    """Fresh TrackGenerator (warm-up + capture), one batch; returns the Track.

    ``generate()`` (``sync=True``) blocks on its own stream only."""
    cfg = TrackGenConfig(num_envs=_E, relax_iters=20, device=_DEV)
    rng = PerEnvSeededRNG(seeds=0, num_envs=_E, device=_DEV)
    gen = TrackGenerator(cfg, rng)
    return gen.generate(_E)


def _gen_gates():
    """Fresh GateGenerator (warm-up + capture), one batch; returns the GateSequence."""
    cfg = GateGenConfig(num_envs=_E, device=_DEV)
    rng = PerEnvSeededRNG(seeds=0, num_envs=_E, device=_DEV)
    gen = GateGenerator(cfg, rng)
    return gen.generate()


def _make_gates_3d_course(seed: int) -> tuple[Course, wp.array]:
    """Fresh gates-mode 3D Course, CONSTRUCTED + bound but NOT generated.

    Mirrors the frame-collision fixture from ``tests/test_course_gates_3d.py``
    (gate_width=0.2 REQUIRES scale=3.0 to keep the seeds=11 batch valid).
    Returns ``(course, pos)`` so the caller drives the FIRST generate() itself
    (the one-time regeneration-graph capture) — the racy first-call path.
    """
    gcfg = GateGenConfig(device=_DEV, num_envs=_E, gate_width=0.2, scale=3.0,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    course = Course(CourseConfig(mode="gates", gen=gcfg, seeds=seed,
                                 frame_collision=True, agent_radius=0.05,
                                 frame_thickness=0.05, frame_depth=0.05))
    pos = wp.zeros(_E, dtype=wp.vec3f, device=_DEV)
    course.bind(pos)
    return course, pos


def _warm_gates_3d_course() -> Course:
    """Fresh gates-mode 3D Course, warmed SERIALLY: bind, first generate()
    (regeneration-graph capture), first step() (StepResult alloc).

    Serial warmup gives the replay test already-warmed courses; the racy
    first-call construction path is covered by
    ``test_concurrent_gates_3d_fresh_construction``.
    """
    course, _ = _make_gates_3d_course(11)
    course.generate()
    course.step()
    wp.synchronize()
    return course


def _replay_course(course: Course):
    """Replay-path generate() (no reseed: zero-alloc graph launch) + step(),
    then a block on the course's stream; returns the result."""
    course.generate()
    course.step()
    _compat.stream_synchronize(wp.get_stream(_DEV))
    return course.result


def _run_concurrent(workers) -> tuple[list, dict]:
    """Run (key, fn) workers in threads behind a common barrier; collect results/errors."""
    errors: list = []
    results: dict = {}
    barrier = threading.Barrier(len(workers))

    def _wrap(key, fn):
        try:
            barrier.wait()  # maximize capture-window overlap across threads
            results[key] = fn()
        except Exception as exc:  # noqa: BLE001 — the race manifests as varied exc types
            errors.append((key, exc))

    threads = [
        threading.Thread(target=_wrap, args=(key, fn)) for key, fn in workers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors, results


def test_concurrent_tracks_and_gates() -> None:
    """The param-explorer page-load scenario: tracks + gates capture concurrently."""
    wp.init()
    ref_tracks = _valid_count(_gen_tracks())  # single-threaded reference, same seeds
    ref_gates = _valid_count(_gen_gates())
    errors, results = _run_concurrent([("tracks", _gen_tracks), ("gates", _gen_gates)])
    assert not errors, f"concurrent generate raised: {errors}"
    assert _valid_count(results["tracks"]) == ref_tracks
    assert _valid_count(results["gates"]) == ref_gates


def test_concurrent_tracks_and_tracks() -> None:
    """Two overlapping page loads: two fresh TrackGenerators capture concurrently."""
    wp.init()
    ref = _valid_count(_gen_tracks())
    errors, results = _run_concurrent(
        [("tracks0", _gen_tracks), ("tracks1", _gen_tracks)]
    )
    assert not errors, f"concurrent generate raised: {errors}"
    assert _valid_count(results["tracks0"]) == ref
    assert _valid_count(results["tracks1"]) == ref


def test_concurrent_gates_3d_replay() -> None:
    """Two already-warmed gates-mode 3D Courses drive the REPLAY path concurrently on cuda.

    Construction/warmup is serialized (see ``_warm_gates_3d_course``); the
    concurrent phase drives the replay path — lock-free launches of each
    course's regeneration graph, plus the per-step localize / frame-collision /
    progress kernels — with full thread overlap.
    """
    wp.init()
    c0 = _warm_gates_3d_course()
    c1 = _warm_gates_3d_course()
    ref = int(to_t(c0.result.valid).sum().item())
    assert ref > 0, "warmed gates-3d course produced no valid envs"
    errors, results = _run_concurrent(
        [("course0", lambda: _replay_course(c0)),
         ("course1", lambda: _replay_course(c1))]
    )
    assert not errors, f"concurrent gates-3d course raised: {errors}"
    assert _valid_count(results["course0"]) == ref  # deterministic under the fixed seed
    assert _valid_count(results["course1"]) == ref


def test_concurrent_gates_3d_fresh_construction() -> None:
    """Two threads each build a FRESH gates-mode 3D Course and run their
    FIRST generate() + step() — the one-time regeneration-graph captures and
    the replays racing across threads — but only across the portions covered
    by the concurrency contract.

    The contract only guarantees that generate()-time device work (capture
    on the course's private stream, replay, stream sync) is safe under
    concurrent calls; construction (readback-based sizing through the legacy
    default stream) is OUT of contract while a peer may be capturing. Two
    ``threading.Barrier(2)`` waits pin this test to the contract: one after
    construction (so construction-time readbacks never overlap a peer's
    capture), and one after generate() (so step() runs once both captures
    ended), while still driving both threads' generate() calls concurrently
    — the contract-covered overlap this test exists to exercise.

    The ~25% CUDA-700 rate is what the ORIGINAL cold-compile root-cause
    measurement saw (fresh kernel cache widening the capture window, back
    when every capture recorded the shared current stream). The fix is
    mechanism-justified rather than reproduced empirically: each
    producer now captures on its own private stream, so a peer's launches
    can never land in its recording."""
    errors: list = []
    barrier = threading.Barrier(2)

    def worker(seed: int) -> None:
        try:
            course, pos = _make_gates_3d_course(seed)   # builds config+Course+bind, NO generate
            barrier.wait(timeout=60)  # construction done; only generate() overlaps a peer's capture
            course.generate()
            barrier.wait(timeout=60)  # generate() done on both sides: no capture in flight
            course.step()
        except Exception as exc:  # noqa: BLE001 — capture for main-thread assert
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(11,))
    t2 = threading.Thread(target=worker, args=(37,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors, errors
