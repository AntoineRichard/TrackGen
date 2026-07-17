"""Gates-mode Course facade: from_gates progress + post collision."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import GateGenConfig

E = 4


def _course(post_radius=0.02, **kw):
    from track_gen.course import Course, CourseConfig
    cfg = CourseConfig(mode="gates",
                       gen=GateGenConfig(num_envs=E, device="cpu",
                                         gate_width=0.1),
                       seeds=21, post_radius=post_radius, **kw)
    return Course(cfg)


def _buffers():
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                      dtype=wp.quatf, device="cpu")
    he = wp.array(np.full((E, 2), 0.01, np.float32), dtype=wp.vec2f, device="cpu")
    return pos, orient, he


def test_gate_pass_through_facade():
    course = _course()
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    seq = course.generate()
    valid = seq.valid.numpy().astype(bool)
    e = int(np.argmax(valid))
    G = seq.position.shape[0] // E
    g0 = seq.position.numpy().reshape(E, G, 3)[e, 0]
    t0 = seq.tangent.numpy().reshape(E, G, 3)[e, 0]

    def put(p):
        arr = np.zeros((E, 3), np.float32)
        arr[e] = p
        wp.copy(pos, wp.array(arr, dtype=wp.vec3f, device="cpu"))

    put(g0 - 0.2 * t0)
    course.step()
    put(g0 + 0.2 * t0)
    ev = course.step().events
    assert int(ev.passed.numpy()[e]) == 1
    assert int(ev.checkpoint_passed.numpy()[e]) == 0
    assert int(ev.next_checkpoint.numpy()[e]) == 1


def test_post_collision_and_rebuild_on_regenerate():
    course = _course()
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    seq = course.generate()
    valid = seq.valid.numpy().astype(bool)
    e = int(np.argmax(valid))
    G = seq.position.shape[0] // E
    left0 = seq.left.numpy().reshape(E, G, 3)[e, 0]

    arr = np.zeros((E, 3), np.float32)
    arr[e] = left0
    wp.copy(pos, wp.array(arr, dtype=wp.vec3f, device="cpu"))
    res = course.step()
    assert int(res.contacts.hit.numpy()[e]) == 1
    assert int(res.contacts.disc.numpy()[e]) == 0      # gate 0 left post

    course.generate(seeds=888)                          # posts must follow
    seq2 = course.result
    left0b = seq2.left.numpy().reshape(E, G, 3)
    e2 = int(np.argmax(seq2.valid.numpy()))
    arr = np.zeros((E, 3), np.float32)
    arr[e2] = left0b[e2, 0]
    wp.copy(pos, wp.array(arr, dtype=wp.vec3f, device="cpu"))
    res = course.step()
    assert int(res.contacts.hit.numpy()[e2]) == 1       # NEW gate's post hits
    assert int(course.progress._progress.numpy().sum()) == 0  # full reset


def test_progress_only_gates_bundle():
    course = _course(post_radius=0.0)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    course.generate()
    res = course.step()
    assert res.contacts is None and course.collision is None


def test_checkpoints_alias_gates_zero_copy():
    course = _course()
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    seq = course.generate()
    assert course.checkpoints.position.ptr == seq.position.ptr
    assert course.checkpoints.count.ptr == seq.count.ptr
