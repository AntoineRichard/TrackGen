"""Pass detection uses the gate's PHYSICAL plane (pose forward), not the
3D spline tangent. On sloped yaw-only gates the two differ; these fixtures
pin the aligned semantics with hand-computed geometry."""
import numpy as np
import warp as wp

from track_gen._src.checkpoints import CheckpointSet
from track_gen._src.progress import ProgressTracker
from track_gen._src.types import GateSequence

E, G = 1, 1
SQ2 = np.float32(1.0 / np.sqrt(2.0))


def _sloped_yaw_only_gate():
    """One gate at origin: spline tangent pitched 45 deg ((s2,0,s2)), pose
    yaw-only upright (forward +x, identity quat), half_size 1, posts at
    y = +/-1."""
    dev = "cpu"
    return GateSequence(
        position=wp.array(np.array([[0, 0, 0]], np.float32), dtype=wp.vec3f, device=dev),
        tangent=wp.array(np.array([[SQ2, 0, SQ2]], np.float32), dtype=wp.vec3f, device=dev),
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


def test_upright_crossing_passes_despite_tilted_tangent():
    # prev (-0.5, 0, 0.9) -> pos (0.5, 0, 0.9): crosses the UPRIGHT plane
    # x = 0 at (0, 0, 0.9), inside the opening (|u|=0, |v|=0.9 <= 1).
    # Against the tilted tangent plane (normal (s2,0,s2)) both endpoints
    # are on the positive side (d = (x+z)/sqrt(2): 0.283 and 0.990) — the
    # OLD semantics saw no crossing at all.
    seq = _sloped_yaw_only_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [-0.5, 0.0, 0.9])          # arms prev_pos
    ev = _step(tr, pos, [0.5, 0.0, 0.9])
    assert int(ev.passed.numpy()[0]) == 1
    assert int(ev.checkpoint_passed.numpy()[0]) == 0


def test_tilted_only_crossing_no_longer_passes():
    # prev (0.2, 0, -0.5) -> pos (0.8, 0, -0.5): x stays positive so the
    # upright plane x = 0 is never crossed; but x + z changes sign
    # (-0.3 -> 0.3), i.e. the OLD tilted-tangent plane WAS crossed at
    # (0.5, 0, -0.5) with |u|=0, old-v = 0.707 <= 1 — the old semantics
    # counted a pass here. Aligned semantics: no event of any kind.
    seq = _sloped_yaw_only_gate()
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tr = ProgressTracker(CheckpointSet.from_gates(seq), position=pos)
    _step(tr, pos, [0.2, 0.0, -0.5])
    ev = _step(tr, pos, [0.8, 0.0, -0.5])
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_way.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
