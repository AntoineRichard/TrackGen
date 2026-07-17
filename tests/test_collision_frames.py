"""Sphere-vs-gate-frame collision (FrameChecker) unit tests."""
import numpy as np
import warp as wp

from track_gen._src.collision_frames import FrameChecker
from track_gen._src.types import GateSequence

E, G = 2, 4


def _square_gate_seq():
    """One valid env with a single gate at origin, facing +x, half_size 1."""
    n = E * G
    nan3 = np.full(3, np.nan, np.float32)
    pos = np.tile(nan3, (n, 1)); tan = pos.copy()
    left = pos.copy(); right = pos.copy()
    quat = np.tile(np.full(4, np.nan, np.float32), (n, 1))
    hs = np.full(n, np.nan, np.float32)
    pos[0] = (0, 0, 0); tan[0] = (1, 0, 0)
    quat[0] = (0, 0, 0, 1)              # identity: fwd=+x, left=+y, up=+z
    hs[0] = 1.0
    left[0] = (0, 1, 0); right[0] = (0, -1, 0)
    dev = "cpu"
    return GateSequence(
        position=wp.array(pos, dtype=wp.vec3f, device=dev),
        tangent=wp.array(tan, dtype=wp.vec3f, device=dev),
        orientation=wp.array(quat, dtype=wp.quatf, device=dev),
        half_size=wp.array(hs, dtype=wp.float32, device=dev),
        left=wp.array(left, dtype=wp.vec3f, device=dev),
        right=wp.array(right, dtype=wp.vec3f, device=dev),
        valid=wp.array(np.array([1, 0], np.int32), device=dev),
        count=wp.array(np.array([1, 0], np.int32), device=dev),
    )


def _query_at(p):
    seq = _square_gate_seq()
    chk = FrameChecker(seq, num_envs=E, radius=0.1, frame_thickness=0.1,
                       frame_depth=0.1)
    pos = wp.array(np.array([p, [0, 0, 0]], np.float32), dtype=wp.vec3f,
                   device="cpu")
    chk.bind_inputs(pos)
    chk.bind_window(wp.zeros(E, dtype=wp.int32, device="cpu"))
    return chk.query().hit.numpy()[0]


def test_through_opening_no_hit():
    assert _query_at([0.0, 0.0, 0.0]) == 0      # dead center
    assert _query_at([0.0, 0.5, 0.5]) == 0      # inside opening


def test_post_and_bar_hits():
    assert _query_at([0.0, 1.05, 0.0]) == 1     # left post
    assert _query_at([0.0, -1.05, 0.0]) == 1    # right post
    assert _query_at([0.0, 0.0, 1.05]) == 1     # top bar
    assert _query_at([0.0, 0.0, -1.05]) == 1    # bottom bar


def test_far_away_no_hit():
    assert _query_at([5.0, 5.0, 5.0]) == 0
