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


def test_posters_keep_declared_input_and_separate_fallback_selection(example, tmp_path):
    story = _story()
    for theme in ('light', 'dark'):
        target = tmp_path / theme
        record = story.render(target, example, movies=False, theme=theme)
        name = example['config']['generator']
        assert (target / f'generator-{name}-construction.png').is_file()
        selected = story.select_construction_sample(example)
        assert record['sample']['seed'] == 7
        assert example['seed'] == 7
        assert record['fallback_example']['seed'] == (97 if name == 'bezier' else 23)
        assert record['fallback_example']['proper_crossings'] == 2
        assert record['fallback_example']['automatic_whole_loop']
        assert record['raw_count'] == len(example['raw'])
        assert record['final_count'] == selected['count']
        assert record['illustrative_inputs']
        assert not record['observation_preserves_outputs']
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


def test_bezier_illustration_maps_ten_actual_draws_and_restores_ordinary_output():
    story = _story()
    default = guide.sample_batch('bezier', seeds=[7])[0]
    before = default['raw'].tobytes(), default['center'].tobytes()
    chosen = story.select_construction_sample(default)
    assert chosen['seed'] == 7
    assert chosen['config']['min_num_points'] == chosen['config']['max_num_points'] == 10
    assert chosen['config']['min_point_distance'] == .0625
    data = story.capture(chosen)
    assert data['illustrative_inputs'] and data['production_sampler_unchanged']
    assert not data['observation_preserves_outputs']
    assert data['ordinary_outputs_restored'] and data['illustration_replay_equal']
    assert len(set(data['grid_ids'])) == 10
    assert data['grid_cells_per_side'] == 8 and data['grid_pitch'] == .125
    assert ((data['offset_vectors'] >= 0) & (data['offset_vectors'] < data['grid_pitch'])).all()
    expected = (data['grid_origins'].astype(np.float32)
                + (data['source_sampler_noise'] + np.float32(.5))*np.float32(.125))
    np.testing.assert_array_equal(data['sampled'], expected)
    np.testing.assert_array_equal(data['source_sampler_points'],
                                  data['grid_origins'].astype(np.float32)+data['source_sampler_noise'])
    assert data['final_valid'] and data['final_fail'] == 0
    assert before == (default['raw'].tobytes(), default['center'].tobytes())
    again = guide.sample_batch('bezier', seeds=[7])[0]
    assert before == (again['raw'].tobytes(), again['center'].tobytes())
    assert default['config']['min_point_distance'] == .05
    assert 'example_selection' not in default
    assert not np.array_equal(chosen['raw'], default['raw'])
    u = np.linspace(0., 1., data['parameters']['num_points_per_segment'])[:, None]
    for i, (p0, p1, p2, p3) in enumerate(data['controls']):
        evaluated = (1-u)**3*p0+3*u*(1-u)**2*p1+3*u*u*(1-u)*p2+u**3*p3
        np.testing.assert_allclose(data['smooth_dense'][i*len(u):(i+1)*len(u)], evaluated,
                                   atol=3e-7, rtol=0.)


def test_illustrative_bezier_poster_labels_input_mapping_and_real_pipeline(tmp_path):
    from PIL import Image
    story = _story()
    default = guide.sample_batch('bezier', seeds=[7])[0]
    record = story.render(tmp_path, default, movies=False)
    selection = record['example_selection']
    assert selection['selected_seed'] == 7
    assert selection['distinct_cells'] == 10 and selection['sampled_inside_cells']
    assert selection['settings'] == {'min_num_points': 10, 'max_num_points': 10,
                                     'min_point_distance': .0625}
    assert record['illustrative_inputs'] and record['production_sampler_unchanged']
    assert 'original_noise + 0.5' in record['sampler_remapping']
    assert not record['observation_preserves_outputs']
    assert 'seed_search' not in selection and 'first_matching_seed' not in selection
    chosen = story.select_construction_sample(default)
    explanations = ' '.join(x['explanation'] for x in story.construction_stages(chosen))
    assert 'illustration' in explanations.lower()
    assert '8 × 8' in explanations and '64' in explanations and '10' in explanations
    with Image.open(tmp_path/'generator-bezier-construction.png') as image:
        assert image.width*9 == image.height*16
        assert record['frame_size'] == record['poster_frame_size'] == [image.width, image.height]
        assert record['aspect_ratio'] == '16:9'


def test_point_family_movie_and_poster_use_the_same_native_widescreen_canvas(example):
    story = _story()
    selected = story.select_construction_sample(example)
    fig, *_ = story._frame(selected, 'light')
    width, height = (np.rint(fig.get_size_inches() * 120)).astype(int)
    assert width * 9 == height * 16
    assert width % 32 == 0 and height % 2 == 0
    plt.close(fig)


def test_ten_point_sampler_replay_matches_actual_generator_prefix():
    from track_gen import ops
    from track_gen.exec import LaunchContext
    from track_gen.seeds import SeedState
    from track_gen._src.warp_generate import CORNER_SALT

    seeds = wp.array(np.arange(64, dtype=np.int32), dtype=wp.int32, device='cpu')
    context = LaunchContext.current('cpu')
    effective = wp.empty(64, dtype=wp.int32, device='cpu')
    state = SeedState(64, 'cpu', seeds)
    state.effective(effective, ctx=context)
    used = wp.empty(64*10, dtype=wp.int32, device='cpu')
    points = wp.empty(64*10, dtype=wp.vec2f, device='cpu')
    size = wp.ones(64, dtype=wp.float32, device='cpu')
    ops.sample_grid_points(context, effective, CORNER_SALT, 0, 10, 8, .125, 1.,
                           used, points, size=size)
    _, generator = guide.sample_batch('bezier', seeds=range(64), overrides={
        'min_num_points': 10, 'max_num_points': 10, 'min_point_distance': .0625},
        return_generator=True)
    scratch = guide._generator_scratch(generator)
    np.testing.assert_array_equal(effective.numpy(), generator.program.seed.numpy())
    np.testing.assert_array_equal(used.numpy(), scratch['gen_used'].numpy())
    np.testing.assert_array_equal(points.numpy(), scratch['gen_corners'].numpy())
    assert np.all(scratch['gen_count'].numpy() == 10)

    illustration = _story().capture(_story().select_construction_sample(
        guide.sample_batch('bezier', seeds=[7])[0]))
    np.testing.assert_array_equal(illustration['source_sampler_points'],
                                  scratch['gen_corners'].numpy().reshape(64, 10, 2)[7])
    np.testing.assert_array_equal(illustration['grid_ids'],
                                  scratch['gen_used'].numpy().reshape(64, 10)[7])


def test_hull_illustration_bounds_anchor_jitter_but_retains_real_midpoint_and_road_stages():
    from track_gen import ops
    from track_gen.exec import LaunchContext

    story = _story()
    default = guide.sample_batch('hull', seeds=[7])[0]
    before = default['raw'].tobytes(), default['center'].tobytes()
    chosen = story.select_construction_sample(default)
    data = story.capture(chosen)
    assert chosen['illustrative_inputs'] and data['production_sampler_unchanged']
    assert chosen['config']['min_num_points'] == chosen['config']['max_num_points'] == 10
    assert chosen['config']['min_point_distance'] == .0625
    assert len(set(data['grid_ids'])) == 10
    assert data['grid_cells_per_side'] == 8 and data['grid_pitch'] == .125
    assert ((data['offset_vectors'] >= 0) & (data['offset_vectors'] < .125)).all()
    np.testing.assert_array_equal(data['sampled'], data['grid_origins'].astype(np.float32)
        +(data['source_sampler_noise']+np.float32(.5))*np.float32(.125))
    assert data['illustration_replay_equal'] and data['ordinary_outputs_restored']
    assert not data['observation_preserves_outputs']
    assert data['final_valid'] and data['final_fail'] == 0
    assert data['smooth_crossings'] == 0
    assert before == (default['raw'].tobytes(), default['center'].tobytes())
    again = guide.sample_batch('hull', seeds=[7])[0]
    assert before == (again['raw'].tobytes(), again['center'].tobytes())
    np.testing.assert_array_equal(story.capture(default)['sampled'],
                                  default['construction']['sampled'])

    # The calm anchor mapping must not flatten Hull's separate midpoint layer.
    np.testing.assert_array_equal(data['polygon_vertices'][::2], data['ordered'])
    moved = data['polygon_vertices'][1::2]-data['midpoints']
    radial = data['midpoints']-data['ordered'].astype(np.float64).mean(axis=0)
    assert np.linalg.norm(moved, axis=1).max() > .01
    assert np.all(np.linalg.norm(moved, axis=1) <= .15*np.linalg.norm(radial, axis=1)+1e-7)
    assert np.max(np.abs(moved[:, 0]*radial[:, 1]-moved[:, 1]*radial[:, 0])) < 1e-7
    # Replaying the public Catmull-Rom op from the captured actual augmented loop
    # must reproduce the entire dense candidate, not a drawing approximation.
    context = LaunchContext.current('cpu')
    augmented = wp.array(data['polygon_vertices'], dtype=wp.vec2f, device='cpu')
    count = wp.array([20], dtype=wp.int32, device='cpu')
    dense = wp.empty(600, dtype=wp.vec2f, device='cpu')
    ops.closed_catmull_rom(context, augmented, 20, 30, dense, count=count, endpoint=True)
    np.testing.assert_array_equal(dense.numpy(), data['smooth_dense'])
    scratch = ops.ArcScratch(wp.empty(600, dtype=wp.vec2f, device='cpu'),
        wp.empty(600, dtype=wp.float32, device='cpu'),
        wp.empty(601, dtype=wp.float32, device='cpu'),
        wp.empty(1, dtype=wp.int32, device='cpu'))
    raw = wp.empty(256, dtype=wp.vec2f, device='cpu')
    out_count = wp.empty(1, dtype=wp.int32, device='cpu')
    ops.arc_resample(context, dense, 600, 256, scratch, raw, out_count)
    np.testing.assert_array_equal(raw.numpy(), data['raw'])
    assert data['fallback']['seed'] == 23
    assert data['fallback']['observation_preserves_outputs']
    assert data['fallback']['smooth_crossings'] == 2


def test_hull_illustration_uses_its_own_canonical_sampler_draws_and_salt():
    story = _story()
    _, generator = guide.sample_batch('hull', seeds=[7], overrides={
        'min_num_points': 10, 'max_num_points': 10, 'min_point_distance': .0625},
        return_generator=True)
    scratch = guide._generator_scratch(generator)
    chosen = story.select_construction_sample(guide.sample_batch('hull', seeds=[7])[0])
    data = story.capture(chosen)
    np.testing.assert_array_equal(data['source_sampler_points'], scratch['points'].numpy())
    np.testing.assert_array_equal(data['grid_ids'], scratch['used'].numpy())
    np.testing.assert_array_equal(data['source_sampler_points'],
        data['grid_origins'].astype(np.float32)+data['source_sampler_noise'])
    from track_gen._src.warp_generate_hull import _midpoint_displace_dyn_k
    from track_gen.exec import LaunchContext
    ordered = wp.array(data['ordered'], dtype=wp.vec2f, device='cpu')
    count = wp.array([10], dtype=wp.int32, device='cpu')
    displacement = wp.array([.15], dtype=wp.float32, device='cpu')
    augmented = wp.empty(20, dtype=wp.vec2f, device='cpu')
    augmented_count = wp.empty(1, dtype=wp.int32, device='cpu')
    LaunchContext.current('cpu').launch(_midpoint_displace_dyn_k, 1,
        [ordered, count, generator.program.seed, 10, displacement, augmented, augmented_count])
    np.testing.assert_array_equal(augmented.numpy(), data['polygon_vertices'])
    assert augmented_count.numpy()[0] == 20
    bezier = story.capture(story.select_construction_sample(
        guide.sample_batch('bezier', seeds=[7])[0]))
    assert not np.array_equal(data['source_sampler_noise'], bezier['source_sampler_noise'])


def test_hull_illustration_labels_anchor_mapping_and_midpoint_displacement_separately(tmp_path):
    story = _story()
    default = guide.sample_batch('hull', seeds=[7])[0]
    record = story.render(tmp_path, default, movies=False)
    assert record['illustrative_inputs'] and record['production_sampler_unchanged']
    assert 'real Hull/road stages' in record['kind']
    assert record['example_selection']['distinct_cells'] == 10
    chosen = story.select_construction_sample(default)
    stages = story.construction_stages(chosen)
    assert len(stages) == 7
    assert 'cell-bounded' in stages[1]['title'].lower()
    assert 'midpoint' in stages[3]['title'].lower()
    assert 'real' in stages[3]['explanation'].lower()
    assert 'Catmull' in stages[4]['title']
    assert '23' in stages[5]['explanation']
    fig, ax, heading, explanation = story._frame(chosen, 'light')
    assert any('Hull curve' == x.get_title() for x in fig.axes)
    plt.close(fig)
