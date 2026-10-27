"""The removed configuration and capture surface fails at the API boundary."""
import dataclasses

import pytest

from track_gen import GateGenConfig, TrackGenConfig
from track_gen.course import Course
from track_gen._src.course_line import CourseLine
from track_gen.stages.params import GateValidityParams
from track_gen.tools import Progress


@pytest.mark.parametrize("field", [
    "energy_steps", "energy_lr", "energy_w_sep", "energy_w_len", "energy_w_bend",
    "energy_w_anchor", "tp_iters", "tp_tau", "tp_alpha", "tp_beta", "smooth_finish",
    "smooth_finish_iters", "smooth_finish_tau", "relax_chunk_size", "relax_use_warp",
    "max_regen_iters", "min_angle", "num_harmonics", "decay_p", "amplitude",
    "num_centerline_samples", "output_mode", "style_sampling", "rad_range",
    "scale_range", "handle_clamp_frac_range",
])
def test_removed_track_fields_reject_even_old_default_values(field):
    assert field not in {f.name for f in dataclasses.fields(TrackGenConfig)}
    with pytest.raises(TypeError, match=field):
        TrackGenConfig(**{field: None})


@pytest.mark.parametrize("field", ["num_points_per_segment", "rad", "edgy",
                                    "handle_clamp_frac", "hull_displacement"])
def test_removed_inert_gate_config_fields(field):
    assert field not in {f.name for f in dataclasses.fields(GateGenConfig)}


@pytest.mark.parametrize("field", ["gate_width", "check_crossing"])
def test_removed_inert_gate_validity_params(field):
    with pytest.raises(TypeError, match=field):
        GateValidityParams(**{field: 0})


def test_capture_flags_and_course_line_alias_are_gone():
    import track_gen
    from track_gen._src import runtime

    assert not hasattr(track_gen, "set_capturing")
    assert not hasattr(Course, "set_capturing")
    assert not hasattr(runtime, "set_capturing")
    assert not hasattr(runtime, "_CAPTURING")
    assert not hasattr(CourseLine, "track")


def test_progress_requires_source_spelling():
    with pytest.raises(TypeError, match="checkpoints"):
        Progress(checkpoints="custom")
    assert Progress(source="custom").source == "custom"


def test_progress_explicit_none_selects_default_checkpoints():
    assert Progress(source=None).source == "checkpoints"
