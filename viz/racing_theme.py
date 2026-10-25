"""Presentation palettes for optional documentation renderers.

Only Matplotlib artists and export colors change. Geometry, sampled values,
selection, axes limits and scalar colormaps are retained. The default light path
is an identity operation so existing maintenance scripts keep their output.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np


@dataclass(frozen=True)
class Palette:
    name: str
    background: str
    ink: str
    muted: str
    grid: str
    blue: str
    orange: str
    road: str
    red: str
    green: str
    purple: str
    teal: str
    yellow: str
    panel: str
    panel2: str
    border: str
    white: str = '#ffffff'


_PALETTES = {
    'light': Palette('light', '#ffffff', '#243344', '#8293a5', '#dce2e9',
                     '#166baf', '#d26a18', '#e3e9ee', '#b43a36', '#19704b',
                     '#7c3aed', '#087f8c', '#b88713', '#f4f7fa', '#edf3f8', '#b9c5d1'),
    'dark': Palette('dark', '#131416', '#e7e9ed', '#abb8c7', '#38434f',
                    '#65b8ff', '#ffad66', '#2a3b4b', '#ff8f87', '#7bd9a7',
                    '#c4a7ff', '#73d6db', '#edca72', '#1f2833', '#273442', '#71859b'),
}
_ACTIVE = ContextVar('trackgen_plot_theme', default='light')
_RENDERED = ContextVar('trackgen_rendered_geometry', default=None)


def get_palette(theme=None) -> Palette:
    name = _ACTIVE.get() if theme is None else theme
    try:
        return _PALETTES[name]
    except (KeyError, TypeError):
        raise ValueError(f"theme must be 'light' or 'dark', got {name!r}") from None


@contextmanager
def plotting_theme(theme='light'):
    """Set drawing defaults without leaking style into another renderer."""
    palette = get_palette(theme)
    parent = _RENDERED.get() if _ACTIVE.get() == palette.name else None
    token = _ACTIVE.set(palette.name)
    record_token = _RENDERED.set({} if parent is None else parent)
    settings = {
        'figure.facecolor': palette.background, 'axes.facecolor': palette.background,
        'savefig.facecolor': palette.background, 'text.color': palette.ink,
        'axes.labelcolor': palette.ink, 'axes.edgecolor': palette.border,
        'xtick.color': palette.muted, 'ytick.color': palette.muted,
        'grid.color': palette.grid, 'legend.facecolor': palette.panel,
        'legend.edgecolor': palette.border,
        'axes.prop_cycle': mpl.cycler(color=[palette.blue, palette.orange, palette.green,
                                             palette.red, palette.purple, palette.teal]),
    }
    try:
        with mpl.rc_context(settings):
            yield palette
    finally:
        _ACTIVE.reset(token)
        _RENDERED.reset(record_token)


# These are semantic colors already used by the documentation's renderers.
# Unlisted saturated colors and scalar-image colormaps retain their data meaning.
_ALIASES = {
    'ink': ('#243344', '#263445', '#23334b', '#111827', '#334155', '#344253', '#374151'),
    'muted': ('#8293a5', '#9daab8', '#667585', '#657086', '#6b7280', '#738499', '#64748b', '#5d6874', '#94a3b8'),
    'blue': ('#166baf', '#2563a6', '#2563eb', '#1f77b4', '#1f78b4', '#3b82f6'),
    'orange': ('#d26a18', '#d97706', '#dc791e', '#f97316', '#ff7f0e', '#e69f00'),
    'red': ('#b43a36', '#b83e35', '#dc2626', '#b91c1c', '#d62728', '#d7191c', '#ef4444'),
    'green': ('#19704b', '#237344', '#15803d', '#1a9641', '#16a34a', '#2ca02c'),
    'purple': ('#7c3aed', '#7570b3', '#8b5cf6', '#4c1d95'),
    'teal': ('#087f8c', '#0f766e', '#0891b2'),
    'yellow': ('#b76a0a', '#b88713', '#ca8a04'),
    'road': ('#e3e9ee', '#e4e9ee', '#e2eaf1', '#dbeafe', '#e5efff', '#ede9fe'),
    'panel': ('#f0f1f3', '#f4f7fa', '#f8fafc', '#e2f3ef'),
}
_ROLES = {value: role for role, values in _ALIASES.items() for value in values}


def _color(value, role, palette):
    try:
        rgba = colors.to_rgba(value)
    except (ValueError, TypeError):
        return value
    if rgba[3] == 0:
        return value
    code = colors.to_hex(rgba, keep_alpha=False)
    final_colors = set(palette.__dict__.values()) - {palette.name, palette.white}
    if code in final_colors:
        return rgba
    # White markings on dark road bands are data, not canvas backgrounds.
    if role != 'face' and code in ('#ffffff', '#f8fafc'):
        return rgba
    mapped = _ROLES.get(code)
    if mapped is not None:
        target = getattr(palette, mapped)
    else:
        rgb = np.asarray(rgba[:3])
        neutral = float(rgb.max() - rgb.min()) < .055
        luminance = float(rgb.mean())
        if not neutral:
            return rgba
        if role == 'face':
            target = palette.panel if luminance > .95 else palette.road
        elif luminance < .38:
            target = palette.ink
        elif luminance < .88:
            target = palette.muted
        else:
            target = palette.grid
    return (*colors.to_rgb(target), rgba[3])


def apply_plot_theme(fig, theme=None):
    """Apply a palette to a Figure's vector artists, preserving every data array.

    Use a fresh Figure for each palette. Reapplication for successive movie frames
    in the same palette is idempotent. Images/colorbars retain their scalar cmap.
    """
    palette = get_palette(theme)
    if palette.name == 'light':
        return palette
    fig.set_facecolor(palette.background)
    for ax in fig.axes:
        ax.set_facecolor(palette.background)
    for artist in fig.findobj(Artist):
        if isinstance(artist, Text):
            artist.set_color(_color(artist.get_color(), 'text', palette))
            box = artist.get_bbox_patch()
            if box is not None:
                box.set_facecolor(_color(box.get_facecolor(), 'face', palette))
                box.set_edgecolor(_color(box.get_edgecolor(), 'edge', palette))
        elif isinstance(artist, Line2D):
            artist.set_color(_color(artist.get_color(), 'line', palette))
            artist.set_markerfacecolor(_color(artist.get_markerfacecolor(), 'line', palette))
            artist.set_markeredgecolor(_color(artist.get_markeredgecolor(), 'line', palette))
        elif isinstance(artist, Patch):
            # Figure/Axes canvas patches were assigned their background above.
            if artist is fig.patch or any(artist is ax.patch for ax in fig.axes):
                continue
            artist.set_facecolor(_color(artist.get_facecolor(), 'face', palette))
            artist.set_edgecolor(_color(artist.get_edgecolor(), 'edge', palette))
        elif isinstance(artist, Collection):
            for getter, setter, role in ((artist.get_facecolors, artist.set_facecolors, 'face'),
                                          (artist.get_edgecolors, artist.set_edgecolors, 'edge')):
                rgba = getter()
                if len(rgba):
                    setter([_color(value, role, palette) for value in rgba])
    for ax in fig.axes:
        if hasattr(ax, 'zaxis'):
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((*colors.to_rgb(palette.panel), .65))
        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_color(palette.grid)
        for spine in ax.spines.values():
            spine.set_edgecolor(palette.border)
    return palette


def save_figure(fig, path, *, theme=None, **kwargs):
    """Save a true themed PNG/SVG; never transform already rendered pixels."""
    records = _RENDERED.get()
    if records is not None:
        records[Path(path).name] = figure_geometry_digest(fig)
    palette = apply_plot_theme(fig, theme)
    if palette.name == 'dark':
        kwargs['facecolor'] = palette.background
    fig.savefig(path, **kwargs)


def geometry_digest(value) -> str:
    """Hash capture data independently of dictionary order or presentation theme."""
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, np.ndarray):
            digest.update(b'array\0' + item.dtype.str.encode() + b'\0')
            digest.update(json.dumps(item.shape).encode() + b'\0')
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                add(str(key)); add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b'list\0')
            for child in item: add(child)
        elif isinstance(item, np.generic):
            add(item.item())
        elif isinstance(item, Path):
            add(str(item))
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=True).encode() + b'\0')
    add(value)
    return digest.hexdigest()


def figure_geometry_digest(fig) -> str:
    """Hash the plotted data before styling, excluding colors and pixel layout."""
    axes = []
    for ax in fig.axes:
        row = dict(limits=[ax.get_xlim(), ax.get_ylim()], lines=[], collections=[],
                   patches=[], images=[])
        if hasattr(ax, 'get_zlim'):
            row['limits'].append(ax.get_zlim())
        for line in ax.lines:
            coords = line.get_data_3d() if hasattr(line, 'get_data_3d') else line.get_data()
            row['lines'].append([np.asarray(x) for x in coords])
        for collection in ax.collections:
            item = dict(offsets=np.asarray(collection.get_offsets()),
                        paths=[path.vertices for path in collection.get_paths()])
            if hasattr(collection, 'get_segments'):
                item['segments'] = collection.get_segments()
            # Matplotlib's public 2D getters omit 3D coordinates before projection.
            # Read the stored data directly so auditing never needs a canvas draw.
            for name in ('_segments3d', '_offsets3d'):
                if hasattr(collection, name):
                    item[name[1:]] = [np.asarray(part) for part in getattr(collection, name)]
            for name in ('_faces', '_vec'):
                if hasattr(collection, name):
                    item[name[1:]] = np.asarray(getattr(collection, name))
            for name in ('U', 'V'):
                if hasattr(collection, name): item[name] = np.asarray(getattr(collection, name))
            row['collections'].append(item)
        for patch in ax.patches:
            row['patches'].append([patch.get_path().vertices, patch.get_patch_transform().get_matrix()])
        row['images'] = [np.asarray(image.get_array()) for image in ax.images]
        axes.append(row)
    return geometry_digest(axes)


def rendered_geometry() -> dict:
    """Per-file plotted-coordinate hashes for the current plotting_theme context."""
    return dict(_RENDERED.get() or {})


def theme_outputs(out_dir, theme):
    """Paired jobs use light/dark subfolders; single jobs keep the supplied path."""
    out = Path(out_dir)
    if theme == 'both':
        return [('light', out/'light'), ('dark', out/'dark')]
    return [(get_palette(theme).name, out)]


def render_variants(out_dir, theme, render):
    """Render one/pair of palettes and verify identical plotted coordinates."""
    manifests = {}
    reference = None
    for name, target in theme_outputs(out_dir, theme):
        target.mkdir(parents=True, exist_ok=True)
        with plotting_theme(name):
            result = render(target)
            records = rendered_geometry()
        if reference is None:
            reference = records
        elif records != reference:
            changed = sorted(key for key in set(records) | set(reference)
                             if records.get(key) != reference.get(key))
            raise AssertionError(f'Theme changed plotted geometry: {changed}')
        manifests[name] = result
    return manifests
