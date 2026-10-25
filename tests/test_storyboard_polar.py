"""The Polar explanation must retain actual knots and both raw sampling stages."""
from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from viz.render_generator_guide import sample_batch


@pytest.fixture(scope="module")
def sample():
    return sample_batch("polar", seeds=[7])[0]


def _storyboard():
    name = "viz.storyboard_polar"
    assert importlib.util.find_spec(name) is not None, "Polar needs a dedicated sampling storyboard"
    return importlib.import_module(name)


def test_reference_views_use_the_true_base_ring_and_preserve_actual_controls(sample):
    story = _storyboard()
    data = story.capture(sample)
    np.testing.assert_allclose(np.linalg.norm(data["regular"], axis=1), 1., atol=2e-7)
    np.testing.assert_array_equal(data["regular"][0], [1., 0.])
    np.testing.assert_array_equal(data["controls"], sample["construction"]["anchors"])
    angles = np.unwrap(np.arctan2(data["radial_only"][:, 1], data["radial_only"][:, 0]))
    np.testing.assert_allclose(angles, np.arange(12)*2*np.pi/12, atol=4e-7)
    np.testing.assert_allclose(np.linalg.norm(data["radial_only"], axis=1),
                               np.linalg.norm(data["controls"], axis=1), atol=2e-7)
    assert data["references_are_counterfactual"]
    assert data["controls_replay_equal"]


def test_arc_samples_are_not_dense_parameter_samples_and_preview_uses_actual_counts(sample):
    data = _storyboard().capture(sample)
    assert data["dense"].shape == (360, 2)
    assert data["raw_before_normalization"].shape == (256, 2)
    np.testing.assert_array_equal(data["raw"], sample["raw"])
    assert data["raw_replay_equal"]
    assert data["final_count"] == sample["count"] == 75
    assert data["regular"][0, 0] != pytest.approx(np.linalg.norm(data["controls"], axis=1).max())
    assert story_keys(sample) == ["slots", "radii", "angles", "spline", "resample", "normalize"]


def story_keys(sample):
    return [stage["key"] for stage in _storyboard().construction_stages(sample)]


def _draw(story, ax, sample, stage, progress=1., theme="light"):
    assert hasattr(story, "draw_step"), "Polar sampling phases need a dedicated drawing"
    return story.draw_step(ax, sample, stage, progress, theme=theme)


def test_radius_and_angle_frames_show_separate_canonical_views(sample):
    import matplotlib.pyplot as plt

    story = _storyboard()
    data = story.capture(sample)
    fig, axes = plt.subplots(1, 2)
    radius = _draw(story, axes[0], sample, 1)
    angle = _draw(story, axes[1], sample, 2)
    np.testing.assert_array_equal(radius["active_curve"], data["radial_only"])
    np.testing.assert_array_equal(angle["active_curve"], data["controls"])
    assert np.linalg.norm(radius["active_curve"]-angle["active_curve"], axis=1).max() > .01
    plt.close(fig)


def test_resampling_frame_reveals_actual_arc_samples_instead_of_dense_prefix(sample):
    import matplotlib.pyplot as plt

    story = _storyboard()
    data = story.capture(sample)
    fig, ax = plt.subplots()
    early = _draw(story, ax, sample, 4, .25)
    limits = ax.get_xlim(), ax.get_ylim()
    ax.clear()
    final = _draw(story, ax, sample, 4)
    np.testing.assert_array_equal(early["active_curve"], data["raw_before_normalization"][:64])
    np.testing.assert_array_equal(final["active_curve"], data["raw_before_normalization"])
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert not np.array_equal(final["active_curve"], data["dense"][:256])
    plt.close(fig)


def test_light_dark_frames_retain_identical_actual_geometry(sample):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from viz.racing_theme import get_palette

    story = _storyboard()
    fig, axes = plt.subplots(1, 2)
    light = _draw(story, axes[0], sample, 3, .37, "light")
    dark = _draw(story, axes[1], sample, 3, .37, "dark")
    np.testing.assert_array_equal(light["active_curve"], dark["active_curve"])
    assert axes[0].get_xlim() == axes[1].get_xlim()
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert axes[0].get_facecolor() == to_rgba(get_palette("light").background)
    assert axes[1].get_facecolor() == to_rgba(get_palette("dark").background)
    plt.close(fig)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_static_storyboard_uses_six_steps_and_actual_output_counts(sample, tmp_path, theme):
    import matplotlib.image as mpimg
    from matplotlib.colors import to_rgb
    from viz.racing_theme import get_palette

    story = _storyboard()
    assert hasattr(story, "render"), "Polar needs a themed six-stage poster and movie renderer"
    record = story.render(tmp_path, sample, movies=False, theme=theme)
    assert record["stages"] == 6
    assert record["raw_count"] == 256 and record["final_count"] == sample["count"]
    assert record["references_are_counterfactual"] and record["controls_replay_equal"]
    assert record["raw_replay_equal"] and not record["geometry_interpolated"]
    assert record["frames"] == 0 and record["duration_seconds"] == 0
    assert record.get("poster") == "generator-polar-construction-poster.png"
    poster = mpimg.imread(tmp_path / record["poster"])
    height, width = poster.shape[:2]
    assert width * 9 == height * 16
    assert record['aspect_ratio'] == '16:9'
    assert record['frame_size'] == record['poster_frame_size'] == [width, height]
    pixels = mpimg.imread(tmp_path / "generator-polar-construction.png")
    np.testing.assert_allclose(pixels[0, 0, :3], to_rgb(get_palette(theme).background), atol=1/255)
