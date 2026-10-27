"""The shared-tail animation must distinguish count, spacing and source arcs."""
import numpy as np
import matplotlib.pyplot as plt

from viz import storyboard_spacing as story


def test_real_pipeline_taps_and_canonical_replays_explain_the_same_valid_road():
    data = story.capture()
    assert data['valid'] and not data['truncated']
    assert data['raw_count'] == 256 and data['count'] == 75
    assert data['constant_spacing_replay_equal'] and data['uniform_replay_equal']
    assert data['public_results_unchanged']
    assert 0 < data['closing_remainder'] < data['spacing']
    assert abs(data['nominal_uniform_step'] - data['spacing']) > 1e-5
    assert np.max(np.linalg.norm(data['relaxed'] - data['target'], axis=1)) > 1e-4
    assert len(data['target']) == len(data['relaxed']) == len(data['uniform'])


def test_both_themes_draw_the_same_actual_prefix_without_modifying_capture_data():
    data = story.capture()
    original = {key: data[key].tobytes() for key in ('raw', 'target', 'relaxed', 'uniform')}
    for theme in ('light', 'dark'):
        for stage, key in enumerate(('raw', 'target', 'relaxed', 'uniform')):
            fig, (ax, ruler) = plt.subplots(1, 2)
            first = story.draw_step(ax, ruler, data, stage, .2, theme)
            ax.clear(); ruler.clear()
            last = story.draw_step(ax, ruler, data, stage, 1., theme)
            np.testing.assert_array_equal(first['active_curve'], last['active_curve'][:len(first['active_curve'])])
            np.testing.assert_array_equal(last['active_curve'], data[key])
            plt.close(fig)
    assert original == {key: data[key].tobytes() for key in original}
