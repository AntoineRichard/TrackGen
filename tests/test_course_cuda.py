"""CUDA-only: facade Graph B refresh replay + user-captured step()."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda"),
]

import warp as wp  # noqa: E402
from track_gen import TrackGenConfig  # noqa: E402
from track_gen.course import Course, CourseConfig, set_capturing  # noqa: E402

DEV = "cuda:0"
E = 8


def _cfg(seeds=5):
    return CourseConfig(mode="track",
                        gen=TrackGenConfig(num_envs=E, device=DEV),
                        seeds=seeds, collision="segments",
                        checkpoint_spacing=0.6, max_checkpoints=64)


def _bound_course(seeds=5):
    course = Course(_cfg(seeds))
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                      dtype=wp.quatf, device=DEV)
    he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
    course.bind(position=pos, orientation=orient, half_extents=he)
    return course, pos


def test_graph_b_refresh_replay_recomputes():
    from track_gen.checkpoints import CheckpointSampler
    course, _ = _bound_course()
    course.generate()
    assert course._refresh_graph is not None      # captured on first generate
    course.generate(seeds=901)                    # replay path
    # Poisoned-replay proof: trash the checkpoint buffers and progress state,
    # regenerate with new seeds; the replayed refresh must recompute both.
    course.checkpoints.position.fill_(12345.0)
    course.progress._progress.fill_(-7)
    course.generate(seeds=902)
    ref = CheckpointSampler(course.result, 0.6, max_checkpoints=64).sample()
    np.testing.assert_allclose(course.checkpoints.position.numpy(),
                               ref.position.numpy(), rtol=1e-5, equal_nan=True)
    assert (course.progress._progress.numpy() == 0).all()


def test_user_captured_step_matches_eager_twin():
    course_c, pos_c = _bound_course(seeds=5)
    course_e, pos_e = _bound_course(seeds=5)      # identical seeds -> same tracks
    course_c.generate()
    course_e.generate()

    set_capturing(True)
    try:
        course_c.step()                            # warmup
        wp.synchronize()
        all_mask = wp.full(E, 1, dtype=wp.int32, device=DEV)
        course_c.reset(all_mask)
        course_e.reset(wp.full(E, 1, dtype=wp.int32, device=DEV))
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            course_c.step()
    finally:
        set_capturing(False)

    n_max = course_c.result.outer.shape[0] // E
    center = np.nan_to_num(
        course_c.result.center.numpy().reshape(E, n_max, 3), nan=0.0)
    counts = course_c.result.count.numpy()
    for s in range(12):
        step_pos = np.zeros((E, 3), np.float32)
        for e in range(E):
            m = max(int(counts[e]), 1)
            step_pos[e] = center[e, (s * 4) % m]
        arr = wp.array(step_pos, dtype=wp.vec3f, device=DEV)
        wp.copy(pos_c, arr)
        wp.copy(pos_e, arr)
        course_c._step_result.events.passed.fill_(-7)   # poison
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ev_e = course_e.step().events
        np.testing.assert_array_equal(
            course_c._step_result.events.passed.numpy(), ev_e.passed.numpy())
        np.testing.assert_array_equal(
            course_c._step_result.events.progress.numpy(),
            ev_e.progress.numpy())
        np.testing.assert_array_equal(
            course_c._step_result.contacts.oob.numpy(),
            course_e._step_result.contacts.oob.numpy())


def test_cuda_alias_device_binds_and_steps():
    """Regression: device='cuda' (the alias, not 'cuda:0') must construct,
    bind cuda:0 buffers, generate, and step without a device-mismatch raise."""
    cfg = CourseConfig(mode="track",
                       gen=TrackGenConfig(num_envs=E, device="cuda"),
                       seeds=3, collision="segments",
                       checkpoint_spacing=0.6, max_checkpoints=64)
    course = Course(cfg)
    pos = wp.zeros(E, dtype=wp.vec3f, device="cuda:0")
    orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                      dtype=wp.quatf, device="cuda:0")
    he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f,
                  device="cuda:0")
    course.bind(position=pos, orientation=orient, half_extents=he)
    course.generate()
    result = course.step()
    assert result.events is not None
    assert result.contacts is not None
