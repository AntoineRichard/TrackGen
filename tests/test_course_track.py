"""Track-mode Course facade: lifecycle, refresh coherence, step/reset."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import TrackGenConfig

E = 4
SPACING = 0.6


def _course(collision="segments", **kw):
    from track_gen.course import Course, CourseConfig
    cfg = CourseConfig(mode="track",
                       gen=TrackGenConfig(num_envs=E, device="cpu"),
                       seeds=7, collision=collision,
                       checkpoint_spacing=SPACING, max_checkpoints=64, **kw)
    return Course(cfg)


def _buffers():
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                      dtype=wp.quatf, device="cpu")
    he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device="cpu")
    return pos, orient, he


def _drive(course, pos_buf, n_steps=40):
    """Walk each valid env along its own centerline; returns final events."""
    track = course.result
    n_max = track.outer.shape[0] // E
    center = np.nan_to_num(track.center.numpy().reshape(E, n_max, 3), nan=0.0)
    counts = track.count.numpy()
    ev = None
    for s in range(n_steps):
        step_pos = np.zeros((E, 3), np.float32)
        for e in range(E):
            m = max(int(counts[e]), 1)
            step_pos[e] = center[e, (s * 3) % m]
        wp.copy(pos_buf, wp.array(step_pos, dtype=wp.vec3f, device="cpu"))
        ev = course.step().events
    return ev


def test_lifecycle_errors():
    from track_gen.course import Course, CourseConfig
    course = _course()
    mask = wp.zeros(E, dtype=wp.int32, device="cpu")
    with pytest.raises(RuntimeError, match="generate"):
        course.step()
    with pytest.raises(RuntimeError, match="generate"):
        course.reset(mask)
    course.generate()
    with pytest.raises(RuntimeError, match="bind"):
        course.step()


def test_import_surface():
    import track_gen
    from track_gen.course import Course, CourseConfig, StepResult  # noqa: F401
    assert "course" in track_gen.__all__


def test_end_to_end_generate_step_reset():
    course = _course()
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    track = course.generate()
    assert track is course.result
    assert course.progress is not None and course.collision is not None
    assert course.checkpoints is course.checkpoint_sampler._set

    ev = _drive(course, pos)
    prog = ev.progress.numpy()
    valid = track.valid.numpy().astype(bool)
    assert prog[valid].sum() > 0, "driving the centerline must pass checkpoints"

    res = course.step()
    assert res is course.step()          # same StepResult instance
    assert res.contacts is not None
    # Boxes on the centerline are inside the band.
    oob = res.contacts.oob.numpy()
    assert not oob[valid].any()

    # Per-env reset: only masked envs are cleared.
    before = course.progress._progress.numpy().copy()
    mask_np = np.zeros(E, np.int32)
    victim = int(np.argmax(before * valid))
    mask_np[victim] = 1
    course.reset(wp.array(mask_np, dtype=wp.int32, device="cpu"))
    after = course.progress._progress.numpy()
    assert after[victim] == 0
    keep = [e for e in range(E) if e != victim]
    np.testing.assert_array_equal(after[keep], before[keep])


def test_regenerate_refreshes_everything():
    course = _course()
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    course.generate()
    counts1 = course.checkpoints.count.numpy().copy()
    # 40 steps (like _drive's default): envs whose bead count is divisible by the
    # 3-bead stride only ever visit a third of the beads, so shorter drives make
    # progress depend on checkpoint/bead phase luck of the generated shapes.
    _drive(course, pos, n_steps=40)
    assert course.progress._progress.numpy().sum() > 0

    track2 = course.generate(seeds=999)          # new courses for everyone
    assert track2 is course.result               # in-place fixed-batch contract
    counts2 = course.checkpoints.count.numpy()
    assert (counts1 != counts2).any(), "new geometry should change checkpoint counts"
    assert course.progress._progress.numpy().sum() == 0   # full reset
    assert np.isnan(course.progress._prev_pos.numpy()).all()


def test_sdf_mode_rebakes_on_regenerate():
    course = _course(collision="sdf", sdf_resolution=64)
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    track = course.generate()
    valid = track.valid.numpy().astype(bool)
    e = int(np.argmax(valid))
    n_max = track.outer.shape[0] // E

    def probe(kind):
        center = np.nan_to_num(track.center.numpy().reshape(E, n_max, 3), nan=0.0)
        p = np.zeros((E, 3), np.float32)
        p[e] = center[e, 0] if kind == "inside" else np.array([50.0, 50.0, 0.0])
        wp.copy(pos, wp.array(p, dtype=wp.vec3f, device="cpu"))
        return int(course.step().contacts.oob.numpy()[e])

    assert probe("inside") == 0
    assert probe("far") == 1
    course.generate(seeds=777)
    # Fresh bake: the NEW track's centerline reads inside, far still outside.
    assert probe("inside") == 0
    assert probe("far") == 1


def test_progress_only_bundle():
    course = _course(collision=None)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    course.generate()
    res = course.step()
    assert res.contacts is None
    assert course.collision is None


def test_generate_without_seeds_is_deterministic():
    course = _course(collision=None)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    t1 = course.generate()
    c1 = t1.center.numpy().copy()
    course.generate()                      # no reseed -> identical batch
    np.testing.assert_array_equal(course.result.center.numpy(), c1)
    course.generate(seeds=4242)            # reseed -> different batch
    assert not np.array_equal(course.result.center.numpy(), c1)


def test_int_reseed_is_deterministic():
    """generate(seeds=k) reseeds deterministically: two int reseeds with the
    same k produce identical batches (the int-seed build must match a fresh
    PerEnvSeededRNG's seed + arange expansion)."""
    course = _course(collision=None)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    course.generate(seeds=1234)
    a = course.result.center.numpy().copy()
    course.generate(seeds=999)               # advance to a different batch
    assert not np.array_equal(course.result.center.numpy(), a)
    course.generate(seeds=1234)              # same int reseed -> identical
    np.testing.assert_array_equal(course.result.center.numpy(), a)


def test_seed_array_validation():
    import warp as wp
    course = _course(collision=None)
    pos, _, _ = _buffers()
    course.bind(position=pos)
    course.generate()
    with pytest.raises(ValueError, match="seeds"):
        course.generate(seeds=wp.zeros(E + 1, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="seeds"):
        course.generate(seeds=wp.zeros(E, dtype=wp.float32, device="cpu"))
    from track_gen.course import Course, CourseConfig
    from track_gen import TrackGenConfig
    with pytest.raises(ValueError, match="seeds"):
        Course(CourseConfig(mode="track",
                            gen=TrackGenConfig(num_envs=E, device="cpu"),
                            seeds=wp.zeros(E + 2, dtype=wp.int32, device="cpu"),
                            checkpoint_spacing=SPACING))


def test_bind_validates_eagerly_before_generate():
    course = _course(collision="segments")
    bad = wp.zeros(E + 3, dtype=wp.vec3f, device="cpu")
    _, orient, he = _buffers()
    with pytest.raises(ValueError, match="position"):
        course.bind(position=bad, orientation=orient, half_extents=he)


def test_facade_matches_manual_wiring():
    from track_gen.progress import ProgressTracker
    course = _course(collision="segments")
    pos, orient, he = _buffers()
    course.bind(position=pos, orientation=orient, half_extents=he)
    track = course.generate()
    # Twin tracker on the SAME checkpoint set, driven with the same buffer.
    twin = ProgressTracker(course.checkpoints, position=pos)
    all_mask = wp.array(np.ones(E, np.int32), dtype=wp.int32, device="cpu")
    course.reset(all_mask)
    n_max = track.outer.shape[0] // E
    center = np.nan_to_num(track.center.numpy().reshape(E, n_max, 3), nan=0.0)
    counts = track.count.numpy()
    for s in range(25):
        step_pos = np.zeros((E, 3), np.float32)
        for e in range(E):
            m = max(int(counts[e]), 1)
            step_pos[e] = center[e, (s * 3) % m]
        wp.copy(pos, wp.array(step_pos, dtype=wp.vec3f, device="cpu"))
        ev_f = course.step().events
        ev_t = twin.update()
        np.testing.assert_array_equal(ev_f.passed.numpy(), ev_t.passed.numpy())
        np.testing.assert_array_equal(ev_f.next_checkpoint.numpy(),
                                      ev_t.next_checkpoint.numpy())
        np.testing.assert_array_equal(ev_f.progress.numpy(), ev_t.progress.numpy())
