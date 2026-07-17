"""CUDA-only: concurrent generate() calls must not corrupt graph capture.

``TrackGenerator.generate()`` and ``GateGenerator.generate()`` on cuda perform CUDA
graph capture + replay on the device's shared stream, coordinated by module-global
``_CAPTURING`` flags. Unsynchronized concurrent calls (e.g. the param explorer's two
``app.load`` events firing on one page load) used to race the capture window, yielding
"Cannot synchronize device ... while graph capture is active", CUDA errors 401/900 at
``wp_cuda_graph_begin_capture`` / ``wp_cuda_graph_launch``, or an async illegal-access
700 that poisons the context. ``runtime._CAPTURE_LOCK`` now serializes every capture,
replay, and eager-cuda launch; this test drives the original failure mode.

Each scenario spawns threads that generate concurrently on cuda:0 with FRESH generator
instances (first call = warmup + capture, the racy path) and asserts every thread
completes without raising and reproduces the single-threaded reference valid-count for
the same seed (generation is deterministic under a fixed rng).

Capture needs a real GPU, so the whole module is skipped without CUDA.
"""
from __future__ import annotations

import threading

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda"),
]

import warp as wp  # noqa: E402
from tests._warp_compare import to_t  # noqa: E402
from track_gen._src.course import Course, CourseConfig  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.types import GateGenConfig, TrackGenConfig  # noqa: E402

_E = 256
_DEV = "cuda:0"


def _gen_tracks() -> int:
    """Fresh TrackGenerator (warmup + capture), one batch; returns the valid count."""
    cfg = TrackGenConfig(num_envs=_E, relax_iters=20, device=_DEV)
    rng = PerEnvSeededRNG(seeds=0, num_envs=_E, device=_DEV)
    gen = TrackGenerator(cfg, rng)
    track = gen.generate(_E)
    wp.synchronize()
    return int(to_t(track.valid).sum().item())


def _gen_gates() -> int:
    """Fresh GateGenerator (warmup + capture), one batch; returns the valid count."""
    cfg = GateGenConfig(num_envs=_E, device=_DEV)
    rng = PerEnvSeededRNG(seeds=0, num_envs=_E, device=_DEV)
    gen = GateGenerator(cfg, rng)
    gates = gen.generate()
    wp.synchronize()
    return int(to_t(gates.valid).sum().item())


def _warm_gates_3d_course() -> Course:
    """Fresh gates-mode 3D Course, warmed SERIALLY: bind, first generate()
    (subtool allocation + graph A/B capture), first step() (StepResult alloc).

    Mirrors the frame-collision fixture from ``tests/test_course_gates_3d.py``
    (gate_width=0.2 REQUIRES scale=3.0 to keep the seeds=11 batch valid).

    Warmup is deliberately single-threaded: ``Course._build_subtools`` and the
    first-call allocations run OUTSIDE ``runtime._CAPTURE_LOCK``, so two
    threads racing their first generate() can drop a mempool alloc/free into
    the other thread's capture window (CUDA 700 at ``wp_free_device_async`` or
    "Failed to allocate", ~25% on an RTX 5000 Ada). That allocation-window gap
    is a known library limitation documented in the Task 10 report; this test
    pins down the pattern that IS guaranteed safe — and is the production
    pattern the lock protects: concurrent replay-path generate() + step() on
    already-warmed courses.
    """
    gcfg = GateGenConfig(device=_DEV, num_envs=_E, gate_width=0.2, scale=3.0,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    course = Course(CourseConfig(mode="gates", gen=gcfg, seeds=11,
                                 frame_collision=True, agent_radius=0.05,
                                 frame_thickness=0.05, frame_depth=0.05))
    pos = wp.zeros(_E, dtype=wp.vec3f, device=_DEV)
    course.bind(pos)
    course.generate()
    course.step()
    wp.synchronize()
    return course


def _replay_course(course: Course) -> int:
    """Replay-path generate() (no reseed: zero-alloc graph launch) + step();
    returns the valid count."""
    course.generate()
    course.step()
    wp.synchronize()
    return int(to_t(course.result.valid).sum().item())


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
    ref_tracks = _gen_tracks()  # single-threaded reference, same seeds
    ref_gates = _gen_gates()
    errors, results = _run_concurrent([("tracks", _gen_tracks), ("gates", _gen_gates)])
    assert not errors, f"concurrent generate raised: {errors}"
    assert results["tracks"] == ref_tracks
    assert results["gates"] == ref_gates


def test_concurrent_tracks_and_tracks() -> None:
    """Two overlapping page loads: two fresh TrackGenerators capture concurrently."""
    wp.init()
    ref = _gen_tracks()
    errors, results = _run_concurrent(
        [("tracks0", _gen_tracks), ("tracks1", _gen_tracks)]
    )
    assert not errors, f"concurrent generate raised: {errors}"
    assert results["tracks0"] == ref
    assert results["tracks1"] == ref


def test_concurrent_gates_3d_courses() -> None:
    """Two gates-mode 3D Courses run generate() + step() concurrently on cuda.

    Construction/warmup is serialized (see ``_warm_gates_3d_course``); the
    concurrent phase drives the replay path — locked graph launches for the
    gate pipeline and the facade's refresh graph, plus the per-step localize /
    frame-collision / progress kernels — with full thread overlap.
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
    assert results["course0"] == ref  # deterministic under the fixed seed
    assert results["course1"] == ref
