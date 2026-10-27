"""Generator movies show construction and raw output, with matching real-frame posters."""
from contextlib import contextmanager
from io import BytesIO
import re

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from PIL import Image
import pytest

from viz import render_generator_guide as guide
from viz.racing_theme import plotting_theme


@pytest.fixture(scope='module', params=guide.FAMILIES)
def movie_sample(request):
    return guide.sample_batch(request.param, seeds=[7])[0]


def _axes(fig):
    def descend(axis):
        yield axis
        for child in axis.child_axes:
            yield from descend(child)
    return [child for axis in fig.axes for child in descend(axis)]


def _pixels(fig, dpi):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    brands = fig.findobj(lambda artist: artist.get_gid() == 'trackgen-circuit-brand')
    assert len(brands) == 1 and brands[0].get_visible()
    brand_box = brands[0].get_window_extent(renderer)
    assert brand_box.width > 0 and brand_box.height > 0
    for text in fig.texts:
        if text.get_visible() and text.get_position()[1] < .12 and text.get_text():
            assert not brand_box.overlaps(text.get_window_extent(renderer)), text.get_text()
    buffer = BytesIO()
    fig.savefig(buffer, dpi=dpi)
    return np.array(Image.open(buffer).convert('RGBA'))


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_generator_movie_has_only_raw_geometry_and_poster_matches_encoded_scene(
        movie_sample, theme, tmp_path, monkeypatch):
    sample = movie_sample
    name = sample['config']['generator']
    before = {key: sample[key].tobytes() for key in ('raw', 'center', 'inner', 'outer')}
    selected = sample
    if name in ('bezier', 'hull'):
        from viz import storyboard_bezier_hull
        selected = storyboard_bezier_hull.select_construction_sample(sample)
    expected_raw = np.vstack((selected['raw'], selected['raw'][:1]))
    poster_frame = {'bezier': 48, 'hull': 42, 'polar': 72,
                    'voronoi': 96, 'checkpoint': 96}.get(name)

    class Writer:
        frames = 0
        seen_headings = set()

        @contextmanager
        def saving(self, fig, path, dpi):
            self.figure, self.dpi = fig, dpi
            yield
            if name == 'repulsive':
                self.poster_pixels = _pixels(fig, dpi)
                np.testing.assert_array_equal(
                    np.asarray(fig.axes[0].lines[-1].get_xydata()), expected_raw)

        def grab_frame(self, **kwargs):
            fig = self.figure
            self.frames += 1
            assert len(fig.axes) == (1 if name == 'repulsive' else 2), (
                name, [axis.get_title() for axis in fig.axes])
            axes = _axes(fig)
            assert not any(isinstance(collection, PolyCollection)
                           for axis in axes for collection in axis.collections)
            visible = '\n'.join([text.get_text() for text in fig.texts] +
                                [axis.get_title() for axis in axes] +
                                [text.get_text() for axis in axes for text in axis.texts])
            assert not re.search(r'\b(finished|beads)\b', visible, flags=re.IGNORECASE)
            self.seen_headings.update(text.get_text() for text in fig.texts)
            if name == 'repulsive':
                radius = sample['construction']['force_samples']['domain_radius']
                np.testing.assert_allclose(fig.axes[0].get_xlim(), [-1.12*radius, 1.12*radius])
                np.testing.assert_allclose(fig.axes[0].get_ylim(), [-1.12*radius, 1.12*radius])
            else:
                construction, preview = fig.axes
                assert construction.get_subplotspec().get_position(fig).width > .55
                assert construction.get_subplotspec().get_position(fig).width > 1.8 * preview.get_subplotspec().get_position(fig).width
                assert any(np.array_equal(line.get_xydata(), expected_raw)
                           for line in preview.lines), 'Raw preview must retain actual generator samples'
            if self.frames == poster_frame:
                self.poster_pixels = _pixels(fig, self.dpi)

    writer = Writer()
    monkeypatch.setattr(guide, '_writer', lambda: writer)
    try:
        with plotting_theme(theme):
            if name == 'repulsive':
                record = guide.render_growth(tmp_path, sample, movies=True)
                assert record['final_matches_generator']
                assert record['encoded_frame_count'] == writer.frames
                poster = tmp_path / 'generator-repulsive-growth.png'
            else:
                record = guide.render_construction(tmp_path, sample, movies=True, theme=theme)
                assert writer.frames == (96 if name in ('voronoi', 'checkpoint') else 144)
                assert record['frames'] == writer.frames
                for title in record['stage_titles']:
                    assert any(title in heading for heading in writer.seen_headings)
                poster = tmp_path / record.get('poster', f'generator-{name}-construction.png')
        actual = np.array(Image.open(poster).convert('RGBA'))
        assert actual.shape[1] * 9 == actual.shape[0] * 16
        np.testing.assert_array_equal(actual, writer.poster_pixels)
        assert before == {key: sample[key].tobytes() for key in before}
    finally:
        plt.close('all')
