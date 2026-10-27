"""Theme styling changes presentation while retaining every plotted coordinate."""
from __future__ import annotations

import importlib
import importlib.util

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pytest


def _theme():
    name = 'viz.racing_theme'
    assert importlib.util.find_spec(name) is not None, 'Published media need explicit theme palettes'
    return importlib.import_module(name)


def test_dark_export_keeps_geometry_and_semantic_color_roles(tmp_path):
    theme = _theme()
    p = theme.get_palette('dark')
    xy = np.array([[0., 0.], [1., .4], [2., 0.]])
    fig, ax = plt.subplots()
    line, = ax.plot(*xy.T, color='#166baf', label='centerline')
    white, = ax.plot([0., 2.], [.1, .1], color='#f8fafc')
    road = Polygon(xy, facecolor='#e3e9ee')
    ax.add_patch(road)
    dots = ax.scatter(*xy.T, color='#d26a18')
    title = ax.set_title('Actual coordinates', color='#243344')
    note=ax.text(.1,.2,'Measured distance',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    ax.legend(facecolor='white')
    before = line.get_xydata().copy(), dots.get_offsets().copy(), road.get_xy().copy()
    theme.save_figure(fig, tmp_path/'dark.png', theme='dark', facecolor='white')
    np.testing.assert_array_equal(line.get_xydata(), before[0])
    np.testing.assert_array_equal(dots.get_offsets(), before[1])
    np.testing.assert_array_equal(road.get_xy(), before[2])
    assert colors.to_hex(line.get_color()) == p.blue
    assert colors.to_hex(title.get_color()) == p.ink
    assert colors.to_hex(road.get_facecolor()) == p.road
    assert colors.to_hex(note.get_bbox_patch().get_facecolor()) == p.panel
    assert colors.to_hex(white.get_color()) == '#f8fafc'  # a white road marking stays visible
    from PIL import Image
    pixel = np.asarray(Image.open(tmp_path/'dark.png').convert('RGB'))[0, 0]
    np.testing.assert_array_equal(pixel, np.rint(np.array(colors.to_rgb(p.background))*255))
    # Reapplying before another movie frame must not drift already themed colors.
    first = line.get_color(), title.get_color(), road.get_facecolor()
    theme.apply_plot_theme(fig, 'dark')
    assert (line.get_color(), title.get_color(), road.get_facecolor()) == first
    plt.close(fig)


def test_light_export_preserves_the_existing_figure_bytes(tmp_path):
    theme = _theme()
    fig, ax = plt.subplots()
    ax.plot([0., 1.], [1., 0.], color='#2563eb')
    ax.set_title('Existing light figure', color='#23334b')
    fig.savefig(tmp_path/'reference.png', facecolor='white')
    theme.save_figure(fig, tmp_path/'light.png', theme='light', facecolor='white')
    assert (tmp_path/'light.png').read_bytes() == (tmp_path/'reference.png').read_bytes()
    plt.close(fig)


def test_plotting_context_restores_defaults_and_rejects_unknown_theme():
    theme = _theme()
    before = matplotlib.rcParams['figure.facecolor']
    with theme.plotting_theme('dark'):
        assert theme.get_palette().name == 'dark'
        assert matplotlib.rcParams['figure.facecolor'] == '#131416'
    assert matplotlib.rcParams['figure.facecolor'] == before
    assert theme.get_palette().name == 'light'
    with pytest.raises(ValueError, match='theme'):
        theme.get_palette('inverted')


def test_geometry_digest_is_sensitive_to_data_but_not_dictionary_order():
    theme = _theme()
    x = np.array([[1., 2.], [3., 4.]], np.float32)
    first = theme.geometry_digest({'xy': x, 'valid': [1, 0]})
    assert first == theme.geometry_digest({'valid': [1, 0], 'xy': x.copy()})
    x[0, 0] += .01
    assert first != theme.geometry_digest({'xy': x, 'valid': [1, 0]})


def test_dark_three_dimensional_panes_do_not_keep_light_defaults(tmp_path):
    theme=_theme();p=theme.get_palette('dark')
    fig=plt.figure();ax=fig.add_subplot(111,projection='3d')
    ax.plot([0,1],[1,0],[0,1],color=p.blue)
    theme.save_figure(fig,tmp_path/'3d.png',theme='dark')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        assert colors.to_hex(axis.pane.get_facecolor())==p.panel
    plt.close(fig)


def test_same_theme_nested_storyboards_keep_outer_geometry_audit(tmp_path):
    theme=_theme()
    def draw(path):
        fig,ax=plt.subplots();ax.plot([0,1],[1,0])
        theme.save_figure(fig,path);plt.close(fig)
    with theme.plotting_theme('dark'):
        draw(tmp_path/'outer.png')
        with theme.plotting_theme('dark'):
            draw(tmp_path/'poster.png')
        assert set(theme.rendered_geometry())=={'outer.png','poster.png'}
        with theme.plotting_theme('light'):
            draw(tmp_path/'separate.png')
            assert set(theme.rendered_geometry())=={'separate.png'}
        assert set(theme.rendered_geometry())=={'outer.png','poster.png'}


@pytest.mark.parametrize('artist_kind', ['quiver', 'scatter'])
def test_geometry_audit_covers_3d_collections_before_draw(artist_kind, monkeypatch):
    theme = _theme()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    if artist_kind == 'quiver':
        artist = ax.quiver(0., 0., 0., 1., 1., 1.)
    else:
        artist = ax.scatter([0., 1.], [0., 1.], [0., 1.])

    def no_draw(*args, **kwargs):
        raise AssertionError('Coordinate audits must not force rendering')

    monkeypatch.setattr(fig.canvas, 'draw', no_draw)
    before = theme.figure_geometry_digest(fig)
    if artist_kind == 'quiver':
        artist.remove()
        ax.quiver(0., 0., 0., .25, .25, .25)
    else:
        artist.set_3d_properties([1., 0.], zdir='z')
    assert theme.figure_geometry_digest(fig) != before
    plt.close(fig)


@pytest.mark.parametrize('width,dpi,expected', [(14,120,(1696,954)),(12,120,(1440,810)),(12,100,(1216,684))])
def test_video_canvas_is_natively_exact_widescreen_with_encoder_safe_dimensions(tmp_path,width,dpi,expected):
    from PIL import Image
    theme=_theme()
    assert hasattr(theme,'video_canvas_size'), 'Movie canvases need a shared native 16:9 sizing rule'
    size=theme.video_canvas_size(width,dpi)
    fig=plt.figure(figsize=size,dpi=dpi)
    fig.text(.5,.5,'Native canvas',ha='center')
    path=tmp_path/'frame.png';fig.savefig(path)
    actual=Image.open(path).size
    assert actual==expected==fig.canvas.get_width_height()
    assert actual[0]*9==actual[1]*16
    assert actual[0]%32==actual[1]%2==0
    assert actual[0]>=width*dpi and actual[0]-width*dpi<32
    plt.close(fig)


@pytest.mark.parametrize('width,dpi', [(0,120),(12,0),(-1,100),(float('nan'),100),(12,float('inf'))])
def test_video_canvas_rejects_nonpositive_or_nonfinite_dimensions(width,dpi):
    theme=_theme()
    assert hasattr(theme,'video_canvas_size')
    with pytest.raises(ValueError,match='positive.*finite|finite.*positive'):
        theme.video_canvas_size(width,dpi)


@pytest.mark.parametrize('depth',[1,2])
def test_geometry_audit_traverses_child_and_grandchild_axes_without_drawing(depth,monkeypatch):
    theme=_theme();fig,ax=plt.subplots();ax.set(xlim=(0,1),ylim=(0,1))
    child=ax
    for _ in range(depth):
        child=child.inset_axes([.1,.1,.7,.7]);child.set(xlim=(0,1),ylim=(0,1))
    line,=child.plot([0.,1.],[0.,1.])
    def no_draw(*args,**kwargs):
        raise AssertionError('Coordinate audits must not force rendering')
    monkeypatch.setattr(fig.canvas,'draw',no_draw)
    before=theme.figure_geometry_digest(fig)
    line.set_ydata([1.,0.])
    after=theme.figure_geometry_digest(fig)
    assert after!=before
    ax.child_axes.append(child)
    assert theme.figure_geometry_digest(fig)==after  # one identity, even through two paths
    plt.close(fig)


@pytest.mark.parametrize('name', ['light', 'dark'])
def test_video_branding_reserves_footer_and_preserves_scientific_geometry(name):
    theme = _theme()
    assert hasattr(theme, 'add_video_branding'), 'Movies need the reusable Circuit brand artist'
    fig, ax = plt.subplots(figsize=theme.video_canvas_size(12), dpi=120)
    ax.plot([0., 1.], [1., 0.])
    before = theme.figure_geometry_digest(fig)
    mark = theme.add_video_branding(fig, name)
    fig.canvas.draw()
    bounds = mark.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    assert .82 <= bounds.x0 < bounds.x1 <= .99
    assert .01 <= bounds.y0 < bounds.y1 <= .10
    assert len(fig.axes) == 1
    assert theme.figure_geometry_digest(fig) == before
    rgba = np.asarray(mark.get_array())
    assert rgba.shape == (32, 140, 4)
    assert rgba[0, 0, 3] == 0  # transparent around the original art
    expected = np.rint(np.array(colors.to_rgba(theme.get_palette(name).blue))*255).astype(np.uint8)
    assert np.any(np.all(rgba == expected, axis=-1))
    pixels = np.asarray(fig.canvas.buffer_rgba())
    x0, y0, x1, y1 = map(round, mark.get_window_extent().extents)
    crop = pixels[pixels.shape[0]-y1:pixels.shape[0]-y0, x0:x1]
    assert np.unique(crop.reshape(-1, 4), axis=0).shape[0] > 1
    plt.close(fig)
