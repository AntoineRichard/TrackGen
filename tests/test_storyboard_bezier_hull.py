"""The point-family movies expose actual sampling and whole-loop fallback."""
from __future__ import annotations

import importlib
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pytest
import warp as wp

from viz import render_generator_guide as guide
from viz.racing_theme import get_palette, plotting_theme


def _story():
    name = 'viz.storyboard_bezier_hull'
    assert importlib.util.find_spec(name) is not None, 'Point-selection storyboard is missing'
    return importlib.import_module(name)


@pytest.fixture(scope='module', params=['bezier', 'hull'])
def example(request):
    return guide.sample_batch(request.param, seeds=[7])[0]


def _crossings(points):
    from track_gen.exec import LaunchContext
    from track_gen import ops
    x = wp.array(points, dtype=wp.vec2f, device='cpu')
    count = wp.array([len(points)], dtype=wp.int32, device='cpu')
    out = wp.empty(1, dtype=wp.int32, device='cpu')
    ops.self_intersections(LaunchContext.current('cpu'), x, len(points), out, count=count)
    return int(out.numpy()[0])


def test_capture_retains_actual_grid_offsets_and_selected_output(example):
    data = _story().capture(example)
    assert data['observation_preserves_outputs']
    n = len(data['sampled'])
    assert len(data['grid_ids']) == n == len(example['construction']['sampled'])
    np.testing.assert_array_equal(data['sampled'], example['construction']['sampled'])
    np.testing.assert_array_equal(data['raw'], example['raw'])
    np.testing.assert_array_equal(data['final_center'], example['center'])
    np.testing.assert_allclose(data['grid_origins'] + data['offset_vectors'], data['sampled'], atol=1e-15)
    # At these defaults, real offsets span several .1 grid pitches, rather than
    # staying inside the chosen cell. The movie must not invent within-cell jitter.
    assert np.max(np.abs(data['offset_vectors'])) > 4 * data['grid_pitch']
    assert data['grid_cells_per_side'] == 10
    assert np.max(np.abs(data['offset_vectors'])) < data['offset_bound'] + 1e-6
    assert _story().capture(example) is data


@pytest.mark.parametrize('name,seed', [('bezier', 97), ('hull', 23)])
def test_actual_rejected_spline_and_whole_polygon_survive_capture(name, seed):
    sample = guide.sample_batch(name, seeds=[seed])[0]
    data = _story().capture(sample)
    assert data['smooth_crossings'] == _crossings(data['smooth_raw']) == 2
    assert _crossings(data['raw']) == 0
    assert not np.array_equal(data['smooth_raw'], data['raw'])
    assert len(data['crossing_points']) == 2
    assert sample['valid']
    if name == 'hull':
        # Hull overwrites dense with the fallback; the rejected smooth data must
        # be captured before that overwrite rather than guessed afterward.
        assert not np.array_equal(data['smooth_dense'], sample['construction']['dense'])
        np.testing.assert_array_equal(data['polygon_vertices'], sample['construction']['augmented'])
    else:
        np.testing.assert_array_equal(data['polygon_vertices'], sample['construction']['anchors'])


def test_progressive_curve_and_view_are_identical_between_themes(example):
    story = _story()
    stages = story.construction_stages(example)
    stage = next(i for i, item in enumerate(stages) if item['progressive_curve'])
    curves = []
    for theme in ('light', 'dark'):
        with plotting_theme(theme):
            fig, ax = plt.subplots()
            early = story.draw_step(ax, example, stage, .2, theme=theme)
            limits = ax.get_xlim(), ax.get_ylim()
            prefix = early['active_curve'].copy()
            ax.clear()
            last = story.draw_step(ax, example, stage, 1., theme=theme)
            np.testing.assert_array_equal(prefix, last['active_curve'][:len(prefix)])
            assert 1 < len(prefix) < len(last['active_curve'])
            assert limits == (ax.get_xlim(), ax.get_ylim())
            assert ax.get_facecolor() == pytest.approx(plt.matplotlib.colors.to_rgba(get_palette(theme).background))
            curves.append(last['active_curve'])
            plt.close(fig)
    np.testing.assert_array_equal(*curves)


def test_posters_keep_actual_sample_and_fallback_selection(example, tmp_path):
    story = _story()
    for theme in ('light', 'dark'):
        target = tmp_path / theme
        record = story.render(target, example, movies=False, theme=theme)
        name = example['config']['generator']
        assert (target / f'generator-{name}-construction.png').is_file()
        assert record['sample']['seed'] == 7
        assert record['fallback_example']['seed'] == (97 if name == 'bezier' else 23)
        assert record['fallback_example']['proper_crossings'] == 2
        assert record['fallback_example']['automatic_whole_loop']
        assert record['raw_count'] == len(example['raw'])
        assert record['final_count'] == example['count']
        assert record['theme'] == theme
        assert record['geometry_interpolated'] is False
        assert record['elapsed_runtime'] is False


def test_explicit_generator_params_drive_grid_and_handles_instead_of_flat_defaults():
    sample = guide.sample_batch('bezier', seeds=[7], overrides={
        'generator_params': dict(rad=.12, handle_clamp_frac=.25,
                                 min_point_distance=.1, num_points_per_segment=9)})[0]
    data = _story().capture(sample)
    assert data['grid_cells_per_side'] == 5
    assert data['grid_pitch'] == pytest.approx(.2)
    u = np.linspace(0., 1., 9)[:, None]
    for i, (p0, p1, p2, p3) in enumerate(data['controls']):
        expected = ((1-u)**3*p0 + 3*u*(1-u)**2*p1
                    + 3*u*u*(1-u)*p2 + u**3*p3)
        np.testing.assert_allclose(data['smooth_dense'][i*9:(i+1)*9], expected,
                                   atol=3e-7, rtol=0.)
