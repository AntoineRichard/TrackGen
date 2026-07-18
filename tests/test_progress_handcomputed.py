"""Hand-computed progress fixtures. These assert LITERAL expected events —
deliberately not routed through tests/_progress_oracle.py, which mirrors
the kernel's _plane_pass and therefore cannot catch a shared bug.

Single-gate synthetic (same shape as tests/test_progress_gate_plane.py):
gate at the origin, PLANAR tangent and forward both (1, 0, 0) (old/new
semantics agree here — the tilted-tangent distinction is the sibling
file's concern), posts at y = +/-1 (half-opening 1 on the u axis),
half_size (up_half) = 1. Each fixture is two update() calls: the first
arms prev_pos, the second is the literal case under test.
"""
import numpy as np
import warp as wp

from track_gen._src.checkpoints import CheckpointSet
from track_gen._src.progress import ProgressTracker
from track_gen._src.types import GateSequence

E, G = 1, 1


def _planar_gate():
    """One gate at origin: tangent == forward == (1, 0, 0), half_size 1,
    posts at y = +/-1."""
    dev = "cpu"
    return GateSequence(
        position=wp.array(np.array([[0, 0, 0]], np.float32), dtype=wp.vec3f, device=dev),
        tangent=wp.array(np.array([[1, 0, 0]], np.float32), dtype=wp.vec3f, device=dev),
        forward=wp.array(np.array([[1, 0, 0]], np.float32), dtype=wp.vec3f, device=dev),
        orientation=wp.array(np.array([[0, 0, 0, 1]], np.float32), dtype=wp.quatf, device=dev),
        half_size=wp.array(np.array([1.0], np.float32), dtype=wp.float32, device=dev),
        left=wp.array(np.array([[0, 1, 0]], np.float32), dtype=wp.vec3f, device=dev),
        right=wp.array(np.array([[0, -1, 0]], np.float32), dtype=wp.vec3f, device=dev),
        valid=wp.array(np.array([1], np.int32), device=dev),
        count=wp.array(np.array([1], np.int32), device=dev),
    )


def _step(tracker, pos, xyz):
    p = pos.numpy()
    p[0] = xyz
    wp.copy(pos, wp.array(p, dtype=wp.vec3f, device="cpu"))
    return tracker.update()


def test_forward_pass():
    # (-0.5,0,0) -> (0.5,0,0): straight through the opening. n=1 so the
    # single checkpoint wraps to itself and laps increments.
    seq = _planar_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 0.0, 0.0])
    ev = _step(tr, pos, [0.5, 0.0, 0.0])
    assert int(ev.passed.numpy()[0]) == 1
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    assert int(ev.laps.numpy()[0]) == 1
    # dist_to_next: next wraps to the same (only) gate at the origin;
    # |pos - (0,0,0)| = |(0.5,0,0)| = 0.5
    np.testing.assert_allclose(ev.dist_to_next.numpy()[0], 0.5, atol=1e-6)


def test_backward_crossing():
    # (0.5,0,0) -> (-0.5,0,0): crosses the plane backward inside the opening.
    seq = _planar_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [0.5, 0.0, 0.0])
    ev = _step(tr, pos, [-0.5, 0.0, 0.0])
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_way.numpy()[0]) == 1
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    # target unchanged (still gate 0, at the origin); |(-0.5,0,0)| = 0.5
    np.testing.assert_allclose(ev.dist_to_next.numpy()[0], 0.5, atol=1e-6)


def test_edge_touch_lands_on_plane():
    # (-0.5,0,0) -> (0,0,0): the endpoint lands EXACTLY on the plane
    # (d1 == 0), which the kernel's d1 >= 0 counts as a pass.
    seq = _planar_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 0.0, 0.0])
    ev = _step(tr, pos, [0.0, 0.0, 0.0])
    assert int(ev.passed.numpy()[0]) == 1
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    # wraps to the same gate at the origin; |(0,0,0) - (0,0,0)| = 0
    np.testing.assert_allclose(ev.dist_to_next.numpy()[0], 0.0, atol=1e-6)


def test_outside_u_opening():
    # (-0.5,1.5,0) -> (0.5,1.5,0): crosses the plane at y=1.5, outside the
    # u half-opening of 1 (posts at y = +/-1) -> no event of any kind.
    seq = _planar_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 1.5, 0.0])
    ev = _step(tr, pos, [0.5, 1.5, 0.0])
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    # no advance: target still gate 0 at the origin; |(0.5,1.5,0)| = sqrt(2.5)
    np.testing.assert_allclose(ev.dist_to_next.numpy()[0], np.sqrt(2.5), atol=1e-6)


def test_outside_v_half():
    # (-0.5,0,1.5) -> (0.5,0,1.5): crosses the plane in-u (u=0) but at
    # v=1.5 > up_half=1 -> no event.
    seq = _planar_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 0.0, 1.5])
    ev = _step(tr, pos, [0.5, 0.0, 1.5])
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    # no advance: target still gate 0 at the origin; |(0.5,0,1.5)| = sqrt(2.5)
    np.testing.assert_allclose(ev.dist_to_next.numpy()[0], np.sqrt(2.5), atol=1e-6)


def test_nan_pause():
    # (-0.5,0,0) -> (nan,nan,nan): a NaN endpoint pauses the env; no
    # crossing/wrong-way event can fire and dist_to_next is NaN.
    seq = _planar_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 0.0, 0.0])
    ev = _step(tr, pos, [np.nan, np.nan, np.nan])
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    assert np.isnan(ev.dist_to_next.numpy()[0])
