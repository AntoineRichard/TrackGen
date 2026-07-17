"""Analytic course tests for track_gen.progress (hand-built 4-gate ring)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

wp.init()

E = 1
M = 4


def _ring_checkpoints(device="cpu"):
    """4 checkpoints at angles 0/90/180/270 deg; crossing segments radial
    [0.3, 1.3]; tangents CCW. Agent paths on the unit circle cross them."""
    from track_gen.checkpoints import CheckpointSet
    ang = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    zeros = np.zeros((len(ang), 1), np.float32)
    radial = np.concatenate(
        [np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32), zeros],
        axis=1)
    tang = np.concatenate(
        [np.stack([-np.sin(ang), np.cos(ang)], axis=1).astype(np.float32), zeros],
        axis=1)

    def v3(a):
        return wp.array(a, dtype=wp.vec3f, device=device)

    return CheckpointSet(
        position=v3(radial * 1.0),
        left=v3(radial * 0.3),
        right=v3(radial * 1.3),
        tangent=v3(tang),
        up_half=wp.array(np.full(len(ang), 1.0e30, np.float32),
                         dtype=wp.float32, device=device),
        count=wp.array(np.array([M], np.int32), dtype=wp.int32, device=device),
    )


def _pos(deg):
    a = np.deg2rad(deg)
    return wp.array(np.array([[np.cos(a), np.sin(a), 0.0]], np.float32),
                    dtype=wp.vec3f, device="cpu")


def test_ccw_lap_event_trace():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    trace = []
    for k in range(10):  # angles -22.5 + 45k: crossings at k = 1,3,5,7,9
        ev = tracker.update(_pos(-22.5 + 45.0 * k))
        trace.append((int(ev.passed.numpy()[0]), int(ev.next_checkpoint.numpy()[0]),
                      int(ev.laps.numpy()[0]), int(ev.progress.numpy()[0])))
    assert trace == [
        (0, 0, 0, 0),  # first update: init only
        (1, 1, 0, 1),  # crossed gate 0
        (0, 1, 0, 1),
        (1, 2, 0, 2),  # gate 1
        (0, 2, 0, 2),
        (1, 3, 0, 3),  # gate 2
        (0, 3, 0, 3),
        (1, 0, 1, 4),  # gate 3 -> lap complete
        (0, 0, 1, 4),
        (1, 1, 1, 5),  # gate 0 again on lap 2
    ]


def test_dist_to_next_matches_geometry():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(-22.5))
    ev = tracker.update(_pos(22.5))   # passed gate 0, next = gate 1 at (0,1)
    p = np.array([np.cos(np.deg2rad(22.5)), np.sin(np.deg2rad(22.5))])
    expected = np.linalg.norm(p - np.array([0.0, 1.0]))
    np.testing.assert_allclose(float(ev.dist_to_next.numpy()[0]), expected,
                               rtol=1e-5)


def test_wrong_way_and_wrong_checkpoint():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(22.5))
    ev = tracker.update(_pos(-22.5))  # backward through gate 0
    assert int(ev.wrong_way.numpy()[0]) == 1
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.next_checkpoint.numpy()[0]) == 0  # no advance

    tracker2 = ProgressTracker(_ring_checkpoints())
    tracker2.update(_pos(80.0))
    ev = tracker2.update(_pos(170.0))  # crosses gate 1 (90 deg), target is 0
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == 1


def test_double_jump_advances_one_and_flags_second():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(-10.0))
    ev = tracker.update(_pos(100.0))  # one step across gates 0 AND 1
    assert int(ev.passed.numpy()[0]) == 1
    assert int(ev.checkpoint_passed.numpy()[0]) == 0
    assert int(ev.next_checkpoint.numpy()[0]) == 1
    assert int(ev.wrong_checkpoint.numpy()[0]) == 1  # the skipped gate 1


def test_reset_mask_no_spurious_crossing():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(-22.5))
    tracker.update(_pos(22.5))
    assert int(tracker._progress.numpy()[0]) == 1
    mask = wp.array(np.array([1], np.int32), dtype=wp.int32, device="cpu")
    tracker.reset(mask)
    # Teleport across the whole course: first post-reset update is inert.
    ev = tracker.update(_pos(200.0))
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    assert int(ev.next_checkpoint.numpy()[0]) == 0
    assert int(ev.laps.numpy()[0]) == 0
    assert int(ev.progress.numpy()[0]) == 0


def test_reset_mask_validation():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    # Wrong dtype exercises the shared runtime._check_arr validator path
    # (same body that now also enforces the device check).
    with pytest.raises(ValueError, match="mask"):
        tracker.reset(wp.zeros(E, dtype=wp.float32, device="cpu"))
    with pytest.raises(ValueError, match="mask"):
        tracker.reset(wp.zeros(E + 1, dtype=wp.int32, device="cpu"))


def test_bound_mode_equivalence_and_errors():
    from track_gen.progress import ProgressTracker
    buf = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    bound = ProgressTracker(_ring_checkpoints(), position=buf)
    free = ProgressTracker(_ring_checkpoints())
    with pytest.raises(ValueError, match="bound"):
        bound.update(_pos(0.0))       # arg while bound
    with pytest.raises(ValueError, match="position"):
        free.update()                 # no-arg while unbound
    for k in range(6):
        p = _pos(-22.5 + 45.0 * k)
        wp.copy(buf, p)
        ev_b = bound.update()
        ev_f = free.update(p)
        assert int(ev_b.passed.numpy()[0]) == int(ev_f.passed.numpy()[0])
        assert int(ev_b.next_checkpoint.numpy()[0]) == int(ev_f.next_checkpoint.numpy()[0])
        np.testing.assert_allclose(ev_b.dist_to_next.numpy(),
                                   ev_f.dist_to_next.numpy(), rtol=1e-6)


def test_import_surface():
    import track_gen
    from track_gen.progress import ProgressEvents, ProgressTracker  # noqa: F401
    assert "progress" in track_gen.__all__


def test_constructor_rejects_mismatched_checkpoint_set():
    from track_gen.checkpoints import CheckpointSet
    from track_gen.progress import ProgressTracker
    good = _ring_checkpoints()
    bad = CheckpointSet(position=good.position, left=wp.zeros(2, dtype=wp.vec3f, device="cpu"),
                        right=good.right, tangent=good.tangent,
                        up_half=good.up_half, count=good.count)
    with pytest.raises(ValueError, match="left"):
        ProgressTracker(bad)


def test_zero_checkpoint_env_is_inert():
    from track_gen.checkpoints import CheckpointSet
    from track_gen.progress import ProgressTracker
    cps = _ring_checkpoints()
    wp.copy(cps.count, wp.zeros(1, dtype=wp.int32, device="cpu"))
    tracker = ProgressTracker(cps)
    tracker.update(_pos(0.0))
    ev = tracker.update(_pos(90.0))   # motion that would cross gates if active
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    assert np.isnan(float(ev.dist_to_next.numpy()[0]))


def test_bind_after_construction_and_rebind():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    buf = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tracker.bind(buf)
    wp.copy(buf, _pos(-22.5))
    tracker.update()                     # bound mode now works
    wp.copy(buf, _pos(22.5))
    ev = tracker.update()
    assert int(ev.passed.numpy()[0]) == 1
    buf2 = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    tracker.bind(buf2)                   # rebinding replaces
    wp.copy(buf2, _pos(67.5))
    tracker.update()                     # reads buf2, no error
    with pytest.raises(ValueError, match="position"):
        tracker.bind(wp.zeros(E + 1, dtype=wp.vec3f, device="cpu"))
