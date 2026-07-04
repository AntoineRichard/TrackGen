"""CourseConfig validation matrix + set_capturing flag propagation."""
from __future__ import annotations

import pytest

from track_gen import GateGenConfig, TrackGenConfig


def _track_cfg(**kw):
    from track_gen.course import CourseConfig
    base = dict(mode="track", gen=TrackGenConfig(num_envs=2, device="cpu"),
                checkpoint_spacing=0.6)
    base.update(kw)
    return CourseConfig(**base)


def _gates_cfg(**kw):
    from track_gen.course import CourseConfig
    base = dict(mode="gates",
                gen=GateGenConfig(num_envs=2, device="cpu", gate_width=0.1))
    base.update(kw)
    return CourseConfig(**base)


def test_valid_constructions():
    _track_cfg()                                   # progress-only track
    _track_cfg(collision="segments")
    _track_cfg(collision="sdf", sdf_resolution=64)
    _track_cfg(collision="sdf")                    # sdf_resolution defaults to 128
    _gates_cfg()                                   # progress-only gates
    _gates_cfg(post_radius=0.02)


def test_mode_and_gen_type_agreement():
    from track_gen.course import CourseConfig
    with pytest.raises(ValueError, match="mode"):
        CourseConfig(mode="drone", gen=TrackGenConfig(num_envs=1, device="cpu"))
    with pytest.raises(ValueError, match="TrackGenConfig"):
        CourseConfig(mode="track",
                     gen=GateGenConfig(num_envs=1, device="cpu"),
                     checkpoint_spacing=0.5)
    with pytest.raises(ValueError, match="GateGenConfig"):
        CourseConfig(mode="gates", gen=TrackGenConfig(num_envs=1, device="cpu"))


def test_inapplicable_options_raise():
    with pytest.raises(ValueError, match="post_radius"):
        _track_cfg(post_radius=0.02)
    with pytest.raises(ValueError, match="collision"):
        _gates_cfg(collision="segments")
    with pytest.raises(ValueError, match="sdf_resolution"):
        _gates_cfg(sdf_resolution=64)
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        _gates_cfg(checkpoint_spacing=0.5)
    with pytest.raises(ValueError, match="max_checkpoints"):
        _gates_cfg(max_checkpoints=32)
    with pytest.raises(ValueError, match="sdf_resolution"):
        _track_cfg(collision="segments", sdf_resolution=64)  # sdf-only knob


def test_numeric_validation():
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        _track_cfg(checkpoint_spacing=0.0)
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        _track_cfg(checkpoint_spacing=float("nan"))
    with pytest.raises(ValueError, match="checkpoint_spacing"):
        from track_gen.course import CourseConfig
        CourseConfig(mode="track", gen=TrackGenConfig(num_envs=1, device="cpu"))
    with pytest.raises(ValueError, match="max_boxes"):
        _track_cfg(max_boxes=0)
    with pytest.raises(ValueError, match="collision"):
        _track_cfg(collision="bvh")
    with pytest.raises(ValueError, match="post_radius"):
        _gates_cfg(post_radius=float("nan"))
    with pytest.raises(ValueError, match="gate_width"):
        from track_gen.course import CourseConfig
        CourseConfig(mode="gates",
                     gen=GateGenConfig(num_envs=1, device="cpu", gate_width=0.0))


def test_set_capturing_propagates():
    from track_gen._src import checkpoints as cps_mod
    from track_gen._src import collision as col_mod
    from track_gen._src import collision_discs as discs_mod
    from track_gen._src import course as course_mod
    from track_gen._src import progress as prog_mod
    from track_gen.course import set_capturing
    set_capturing(True)
    try:
        assert course_mod._CAPTURING and col_mod._CAPTURING \
            and discs_mod._CAPTURING and cps_mod._CAPTURING \
            and prog_mod._CAPTURING
    finally:
        set_capturing(False)
    assert not (course_mod._CAPTURING or col_mod._CAPTURING
                or discs_mod._CAPTURING or cps_mod._CAPTURING
                or prog_mod._CAPTURING)


def test_device_is_canonicalized():
    """Course stores the Warp-canonical device string (matches
    str(arr.device) used in bind/seed validation)."""
    import warp as wp

    from track_gen.course import Course
    cfg = _track_cfg(collision="segments")
    course = Course(cfg)
    assert course._device == str(wp.get_device(cfg.gen.device))
