"""Source-backed explanatory movies for Bezier and Hull construction.

Ordinary endpoints and spline candidates are recorded from the real generator.
The labelled Bezier and Hull illustrations remap only their anchor inputs, then
record the actual curve and road stages. Prefixes explain operation order, not elapsed runtime.
"""
from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import numpy as np
import warp as wp

from .racing_theme import (apply_plot_theme, get_palette, plotting_theme, save_figure,
                           video_canvas_size)


_CACHE = '_bezier_hull_storyboard'
_FALLBACK_SEEDS = {'bezier': 97, 'hull': 23}
_BEZIER_EXAMPLE_SEED = 7
_BEZIER_EXAMPLE_SETTINGS = dict(min_num_points=10, max_num_points=10, min_point_distance=.0625)


@lru_cache(maxsize=1)
def _illustrative_bezier_example():
    data = _capture_one('bezier', _BEZIER_EXAMPLE_SEED, _BEZIER_EXAMPLE_SETTINGS,
                        illustrative_inputs=True)
    distinct = len(set(data['grid_ids']))
    inside = bool(((data['offset_vectors'] >= 0)
                   & (data['offset_vectors'] < data['grid_pitch'])).all())
    if distinct != 10 or not inside or not data['final_valid']:
        raise AssertionError('the declared Bezier illustration no longer meets its input/road checks')
    selection = dict(selected_seed=_BEZIER_EXAMPLE_SEED,
                     settings=dict(_BEZIER_EXAMPLE_SETTINGS), distinct_cells=distinct,
                     sampled_inside_cells=inside, grid_cells_per_side=data['grid_cells_per_side'],
                     grid_pitch=data['grid_pitch'], illustrative_inputs=True,
                     production_sampler_unchanged=True,
                     sampler_remapping='origin + (original_noise + 0.5) * pitch; exact canonical '
                                       'seed-7 cell IDs and RNG draws, remapped only for this illustration.')
    sample = dict(seed=_BEZIER_EXAMPLE_SEED, epoch=0, env=0, config=data['config'],
                  count=data['final_count'], valid=data['final_valid'], fail=data['final_fail'],
                  raw=data['raw'], center=data['final_center'], inner=data['final_inner'],
                  outer=data['final_outer'], illustrative_inputs=True,
                  construction=dict(sampled=data['sampled'], anchors=data['ordered'],
                                    dense=data['smooth_dense'], tangents=data['tangents'],
                                    handle_scales=data['handle_scales'], crossings=data['smooth_crossings']),
                  example_selection=selection)
    sample[_CACHE] = _with_fallback(data)
    return sample


@lru_cache(maxsize=1)
def _illustrative_hull_example():
    settings = dict(min_num_points=10, max_num_points=10, min_point_distance=.0625)
    data = _capture_one('hull', 7, settings, illustrative_inputs=True)
    distinct = len(set(data['grid_ids']))
    inside = bool(((data['offset_vectors'] >= 0)
                   & (data['offset_vectors'] < data['grid_pitch'])).all())
    if distinct != 10 or not inside or not data['final_valid']:
        raise AssertionError('the declared Hull illustration no longer meets its input/road checks')
    selection = dict(selected_seed=7, settings=settings, distinct_cells=distinct,
                     sampled_inside_cells=inside, grid_cells_per_side=data['grid_cells_per_side'],
                     grid_pitch=data['grid_pitch'], illustrative_inputs=True,
                     production_sampler_unchanged=True,
                     sampler_remapping='origin + (original_noise + 0.5) * pitch; exact canonical '
                                       'Hull seed-7 cell IDs and RNG draws, remapped only for this '
                                       'illustration; midpoint displacement is unchanged.')
    sample = dict(seed=7, epoch=0, env=0, config=data['config'],
                  count=data['final_count'], valid=data['final_valid'], fail=data['final_fail'],
                  raw=data['raw'], center=data['final_center'], inner=data['final_inner'],
                  outer=data['final_outer'], illustrative_inputs=True,
                  construction=dict(sampled=data['sampled'], anchors=data['ordered'],
                                    dense=data['smooth_dense'], augmented=data['polygon_vertices'],
                                    crossings=data['smooth_crossings']), example_selection=selection)
    sample[_CACHE] = _with_fallback(data)
    return sample


def select_construction_sample(sample):
    """Use labelled cell-bounded anchor inputs for the two grid-family movies.

    Only render() calls this selection boundary. capture() and draw_step() continue
    to expose the actual sample supplied by their caller, including default jitter.
    """
    name = sample['config']['generator']
    if name == 'bezier':
        return _illustrative_bezier_example()
    if name == 'hull':
        return _illustrative_hull_example()
    return sample


def _finite(points):
    points = np.asarray(points)
    return points[np.isfinite(points).all(axis=1)]


def _close(points):
    return np.concatenate([points, points[:1]])


def _prefix(points, progress):
    points = _finite(points)
    return points[:max(1, int(np.ceil(len(points) * np.clip(progress, 0., 1.))))]


def _crossing_locations(points, expected):
    """Locate the reported proper crossing pairs for the explanatory red markers.

    This plotting calculation uses the production predicate's relative tolerance;
    the recorded canonical count remains authoritative and must agree.
    """
    points = np.asarray(points, np.float64)
    out, pairs = [], []
    cross = lambda a, b: a[0]*b[1] - a[1]*b[0]
    for i in range(len(points)):
        a, b = points[i], points[(i+1) % len(points)]
        r = b-a
        for j in range(i+1, len(points)):
            if min(j-i, len(points)-(j-i)) <= 1:
                continue
            c, d = points[j], points[(j+1) % len(points)]
            s = d-c
            u, v = cross(s, a-c), cross(s, b-c)
            w, z = cross(r, c-a), cross(r, d-a)
            ej, ei = 1e-3*np.dot(s, s), 1e-3*np.dot(r, r)
            if (((u > ej and v < -ej) or (u < -ej and v > ej))
                    and ((w > ei and z < -ei) or (w < -ei and z > ei))):
                out.append(a + cross(c-a, s)/cross(r, s)*r)
                pairs.append([a, b, c, d])
    if len(out) != expected:
        raise AssertionError('plot crossing locations disagree with the recorded generator count')
    return np.asarray(out).reshape(-1, 2), np.asarray(pairs).reshape(-1, 4, 2)


def _capture_one(name, seed, config, *, illustrative_inputs=False):
    from track_gen import TrackGenConfig, TrackGenerator
    from track_gen.exec import LaunchContext
    from track_gen.ops.geometry import _self_intersections_by_i_k, _angle_sort_k
    from track_gen.ops.sampling import _sample_grid_points_launch
    from track_gen._src.warp_generate import CORNER_SALT
    from track_gen._src.warp_generate_hull import POINT_SALT, POINT_SALT_OFFSET
    from track_gen.testing.capture import result_arrays

    fields = {field.name for field in dataclasses.fields(TrackGenConfig)}
    options = {key: value for key, value in config.items() if key in fields}
    options.update(generator=name, num_envs=1, device='cpu', on_invalid='keep')
    cfg = TrackGenConfig(**options)
    seeds = wp.array([seed], dtype=wp.int32, device='cpu')
    gen = TrackGenerator(cfg, seeds=seeds)
    track = gen.generate()
    stage = next(ref for ref in gen.program.spec.stages if ref.slot_id == 'generator')
    parameters = dataclasses.asdict(stage.params)
    for field in ('size', 'min_num_points', 'rad', 'handle_clamp_frac', 'hull_displacement'):
        if field in parameters:
            value = gen.params['generator'].in_effect(field).numpy()[0]
            parameters[field] = int(value) if field == 'min_num_points' else float(value)
    scratch = gen.program._gen_scratch.scratch
    scratch = {key.rsplit('/', 1)[-1]: value for key, value in scratch.items()}
    n = int(scratch['gen_count' if name == 'bezier' else 'count'].numpy()[0])
    before = {key: value.numpy().tobytes() for key, value in result_arrays(track).items()}
    ordinary_raw = gen.program.tap('raw.xy').numpy().copy()
    valid = gen.program.tap('gen.valid').numpy().tobytes()
    observed, remapping, lifetimes = {}, {}, []
    base = LaunchContext.current('cpu')
    if illustrative_inputs:
        if name not in ('bezier', 'hull'):
            raise ValueError('illustrative input mapping requires a grid-family teaching clip')
        point_key = 'gen_corners' if name == 'bezier' else 'points'
        id_key = 'gen_used' if name == 'bezier' else 'used'
        sampler_salt = CORNER_SALT if name == 'bezier' else POINT_SALT
        sampler_offset = 0 if name == 'bezier' else POINT_SALT_OFFSET
        candidate = scratch['gen_rs'] if name == 'bezier' else gen.program.tap('raw.xy')
        noise = wp.empty(cfg.max_num_points, dtype=wp.vec2f, device='cpu')
        noise_ids = wp.empty(cfg.max_num_points, dtype=wp.int32, device='cpu')

    class Observer:
        def __getattr__(self, key):
            return getattr(base, key)

        def launch(self, kernel, dim, inputs=(), outputs=(), **kwargs):
            if (illustrative_inputs and kernel is _angle_sort_k
                    and inputs[0].ptr == scratch[point_key].ptr):
                ids = scratch[id_key].numpy().copy()
                canonical = scratch[point_key].numpy().copy()
                # Reuse the sampler itself with its grid term zeroed, so the exact
                # original RNG noise is retained rather than recovered by subtraction.
                _sample_grid_points_launch(base, gen.program.seed, sampler_salt, sampler_offset,
                    cfg.max_num_points, 8, 0., cfg.scale, noise_ids, noise,
                    size=gen.params['generator'].in_effect('size'))
                np.testing.assert_array_equal(ids, noise_ids.numpy())
                draws = noise.numpy().copy()
                origins = np.stack([ids % 8, ids // 8], axis=1).astype(np.float32)*np.float32(.125)
                np.testing.assert_array_equal(origins+draws, canonical)
                mapped = origins+(draws+np.float32(.5))*np.float32(.125)
                mapped_wp = wp.array(mapped, dtype=wp.vec2f, device='cpu')
                lifetimes.append(mapped_wp)
                base.copy(scratch[point_key], mapped_wp)
                remapping.update(source_sampler_points=canonical, source_sampler_noise=draws)
            if (kernel is _self_intersections_by_i_k
                    and (not illustrative_inputs or inputs[0].ptr == candidate.ptr)):
                if observed:
                    raise AssertionError('expected one generator crossing check')
                observed['smooth_raw'] = inputs[0].numpy().reshape(1, cfg.num_points, 2)[0].copy()
                key = 'gen_dense' if name == 'bezier' else 'dense'
                observed['smooth_dense'] = _finite(scratch[key].numpy().reshape(1, -1, 2)[0]).copy()
            return base.launch(kernel, dim, inputs, outputs, **kwargs)

    if illustrative_inputs:
        # Only this maintenance CPU enqueue sees the mapping. All subsequent
        # curve and road stages are the real bound Program, including validity.
        gen.program.enqueue(Observer())
        illustrated = {key: value.numpy().tobytes() for key, value in result_arrays(track).items()}
        first_raw = gen.program.tap('raw.xy').numpy().copy()
        observed.clear()
        gen.program.enqueue(Observer())
        assert illustrated == {key: value.numpy().tobytes() for key, value in result_arrays(track).items()}
        np.testing.assert_array_equal(first_raw, gen.program.tap('raw.xy').numpy())
    else:
        # Only the bound generator is rerun. Hull's rejected dense candidate would
        # otherwise be overwritten by its polygon fallback.
        gen.program._gen_scratch.enqueue(Observer())
        assert ordinary_raw.tobytes() == gen.program.tap('raw.xy').numpy().tobytes()
        assert valid == gen.program.tap('gen.valid').numpy().tobytes()
        assert before == {key: value.numpy().tobytes() for key, value in result_arrays(track).items()}
    if not observed:
        raise AssertionError('the generator crossing check was not observed')
    raw = gen.program.tap('raw.xy').numpy().reshape(1, cfg.num_points, 2)[0].copy()

    def points(key, count):
        return scratch[key].numpy().reshape(1, -1, 2)[0, :count].copy()

    sampled = points('gen_corners' if name == 'bezier' else 'points', n)
    ordered = points('gen_ordered' if name == 'bezier' else 'sorted', n)
    ids = scratch['gen_used' if name == 'bezier' else 'used'].numpy()[:n].copy()
    distance = parameters['min_point_distance']
    cells, pitch = int(1./(2.*distance)), 2.*distance
    size = np.float32(parameters['size'])
    origins = np.stack([ids % cells, ids // cells], axis=1).astype(np.float32)
    origins = ((origins*np.float32(pitch))*np.float32(cfg.scale))*size
    crossings = int(scratch['gen_crossers' if name == 'bezier' else 'crossers'].numpy()[0])
    locations, crossing_pairs = _crossing_locations(observed['smooth_raw'], crossings)
    count = int(track.count.numpy()[0])
    data = dict(name=name, seed=int(seed), config=dataclasses.asdict(cfg), parameters=parameters, sampled=sampled,
                ordered=ordered, grid_ids=ids, grid_origins=origins.astype(np.float64),
                grid_cells_per_side=cells, grid_pitch=pitch*float(cfg.scale)*float(size),
                grid_scale=float(cfg.scale)*float(size),
                offset_vectors=sampled.astype(np.float64)-origins,
                offset_bound=.5*float(cfg.scale)*float(size),
                raw=raw, smooth_crossings=crossings, crossing_points=locations,
                crossing_pairs=crossing_pairs, final_count=count,
                final_valid=bool(track.valid.numpy()[0]), final_fail=int(track.fail_mask.numpy()[0]),
                final_center=track.center.numpy().reshape(1, cfg.N_max, 3)[0, :count, :2].copy(),
                final_inner=track.inner.numpy().reshape(1, cfg.N_max, 3)[0, :count, :2].copy(),
                final_outer=track.outer.numpy().reshape(1, cfg.N_max, 3)[0, :count, :2].copy(),
                observation_preserves_outputs=not illustrative_inputs,
                illustrative_inputs=illustrative_inputs, production_sampler_unchanged=True, **observed)
    if name == 'hull':
        data['polygon_vertices'] = points('aug', 2*n)
        data['midpoints'] = .5*(ordered + np.roll(ordered, -1, axis=0))
    else:
        tangents, scales = points('gen_tan', n), scratch['gen_scale'].numpy()[:n].copy()
        following = np.roll(ordered, -1, axis=0)
        chord = np.linalg.norm(following-ordered, axis=1)
        h0 = np.minimum(parameters['rad']*chord, parameters['handle_clamp_frac']*scales)
        h1 = np.minimum(parameters['rad']*chord, parameters['handle_clamp_frac']*np.roll(scales, -1))
        data['tangents'], data['handle_scales'] = tangents.copy(), scales.copy()
        data['controls'] = np.stack([ordered, ordered+h0[:, None]*tangents,
                                     following-h1[:, None]*np.roll(tangents, -1, axis=0),
                                     following], axis=1)
        data['polygon_vertices'] = ordered.copy()
    if illustrative_inputs:
        data.update(remapping, illustration_replay_equal=True)
        gen.generate()
        assert before == {key: value.numpy().tobytes() for key, value in result_arrays(track).items()}
        np.testing.assert_array_equal(ordinary_raw, gen.program.tap('raw.xy').numpy())
        data['ordinary_outputs_restored'] = True
    return data


def capture(sample):
    """Cache actual grid, candidate and selected geometry without changing outputs."""
    if _CACHE in sample:
        return sample[_CACHE]
    name = sample['config']['generator']
    if name not in _FALLBACK_SEEDS:
        raise ValueError('this storyboard is only for bezier and hull')
    data = _capture_one(name, sample['seed'], sample['config'])
    np.testing.assert_array_equal(data['raw'], sample['raw'])
    np.testing.assert_array_equal(data['final_center'], sample['center'])
    if data['final_count'] != sample['count'] or data['final_valid'] != sample['valid']:
        raise AssertionError('standalone observation changed the selected example')
    sample[_CACHE] = _with_fallback(data)
    return sample[_CACHE]


def _with_fallback(data):
    name = data['name']
    fallback = _capture_one(name, _FALLBACK_SEEDS[name], {})
    if fallback['smooth_crossings'] <= 0:
        raise AssertionError('the declared default fallback example no longer crosses')
    data['fallback'] = fallback
    geometry = [data['sampled'], data['raw'], data['polygon_vertices'],
                fallback['smooth_raw'], fallback['raw']]
    if name == 'bezier':
        geometry.append(data['controls'].reshape(-1, 2))
    grid_end = data['grid_cells_per_side']*data['grid_pitch']
    geometry.append(np.array([[0., 0.], [grid_end, grid_end]]))
    joined = np.concatenate([_finite(x) for x in geometry])
    low, high = joined.min(axis=0), joined.max(axis=0)
    center = .5*(low+high)
    half = .56*float((high-low).max())
    data['limits'] = (center-half, center+half)
    return data


def construction_stages(sample):
    """Paced explanations, with a separate midpoint stage for Hull."""
    name = sample['config']['generator']
    stages = [
        ('Choose grid addresses', 'Draw the corner count, then sample grid addresses. Distinct addresses are best effort; they are not a clearance guarantee.'),
        ('Add the actual random offsets', 'Orange arrows end at the recorded points. Each offset is drawn in [-0.5, 0.5) before scale and size, not inside one grid cell.'),
        ('Order around the centroid', 'Keep every active point and sort its angle around the centroid. The blue cycle is not a geometric convex hull.'),
    ]
    if name == 'bezier':
        stages.append(('Join corners with two cubic handles', 'Incoming and outgoing directions set each tangent. Handle lengths obey rad and the shorter-edge clamp; blue is the recorded dense cubic.'))
    else:
        stages.extend([
            ('Displace inserted midpoints', 'Keep the original anchors. Move one midpoint per edge inward or outward along its centroid-relative radius.'),
            ('Join the augmented loop with a spline', 'The recorded closed Catmull-Rom samples pass through original anchors and displaced midpoints.'),
        ])
    stages.extend([
        ('If it crosses, replace the whole loop', f'A separate default seed {_FALLBACK_SEEDS[name]} shows the automatic branch: reject the smooth loop and choose its entire polygon. No local spline edit or user toggle.'),
        ('Produce fixed-count raw samples', 'The generator stops at num_points raw samples. The shared spacing, XPBD and uniform stages determine the finished road and bead count.'),
    ])
    if sample.get('illustrative_inputs') and name == 'bezier':
        stages = [
            ('Select 10 grid cells', 'This construction illustration selects 10 addresses from an 8 × 8 grid of 64 cells.'),
            ('Add cell-bounded jitter', 'Orange arrows keep the illustrated points inside their selected cells.'),
            ('Order around the centroid', 'Sort all 10 points by their angle around the centroid.'),
            ('Join corners with cubic handles', 'Tangent directions and two handles join each pair of corners with a cubic.'),
            ('Separate real example: whole-loop fallback', 'Default seed 97 rejects its crossing smooth loop and selects its entire polygon.'),
            ('Resample, then build the road', 'The Bezier stage writes 256 raw samples; the shared road pipeline produces the finished beads.'),
        ]
    if sample.get('illustrative_inputs') and name == 'hull':
        stages = [
            ('Select 10 grid cells', 'This construction illustration selects 10 addresses from an 8 × 8 grid of 64 cells.'),
            ('Add cell-bounded anchor jitter', 'Orange arrows keep the illustrated anchors inside their selected cells.'),
            ('Order around the centroid', 'Sort every anchor by angle; no point is discarded and no geometric convex hull is computed.'),
            ('Displace inserted midpoints', 'Real Hull displacement moves one midpoint per edge radially; the original anchors stay fixed.'),
            ('Join with a Catmull–Rom spline', 'The real closed spline passes through the anchors and displaced midpoints.'),
            ('Separate real example: whole-loop fallback', 'Default seed 23 rejects its crossing smooth loop and selects its entire augmented polygon.'),
            ('Resample, then build the road', 'The Hull stage writes 256 raw samples; the shared road pipeline produces the finished beads.'),
        ]
    smooth_stage = 3 if name == 'bezier' else 4
    return [dict(title=title, explanation=explanation, progressive_curve=i == smooth_stage)
            for i, (title, explanation) in enumerate(stages)]


def _style(ax, p, limits):
    for child in list(ax.child_axes):
        child.remove()
    ax.set_facecolor(p.background)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(limits[0][0], limits[1][0])
    ax.set_ylim(limits[0][1], limits[1][1])
    ax.tick_params(colors=p.muted, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('x [generator coordinates]', color=p.ink, fontsize=10)
    ax.set_ylabel('y [generator coordinates]', color=p.ink, fontsize=10)


def _line(ax, points, color, *, closed=False, **kw):
    points = _close(points) if closed else points
    ax.plot(points[:, 0], points[:, 1], color=color, **kw)


def _scatter(ax, points, color, *, size=28, marker='o'):
    ax.scatter(points[:, 0], points[:, 1], s=size, color=color, marker=marker, zorder=5)


def _grid(ax, data, p):
    end = data['grid_cells_per_side']*data['grid_pitch']
    ticks = np.arange(data['grid_cells_per_side']+1)*data['grid_pitch']
    for x in ticks:
        ax.plot([x, x], [0., end], color=p.grid, lw=.6, zorder=0)
        ax.plot([0., end], [x, x], color=p.grid, lw=.6, zorder=0)
    pitch = (f"{data['grid_pitch']:g}" if data['name'] == 'bezier' or data.get('illustrative_inputs')
             else f"{data['grid_pitch']:.2f}")
    label = (f"{len(data['sampled'])} corners · {data['grid_cells_per_side']} × "
             f"{data['grid_cells_per_side']} grid · pitch {pitch}" if data.get('illustrative_inputs')
             else f"{len(data['sampled'])} corners from [{data['parameters']['min_num_points']}, {data['parameters']['max_num_points']}] · "
                  f"{data['grid_cells_per_side']} × {data['grid_cells_per_side']} addresses · pitch {pitch}")
    ax.text(0., 1.02, label, transform=ax.transAxes, va='bottom', color=p.ink, fontsize=11)


def draw_step(ax, sample, stage, progress, theme='light'):
    """Draw recorded data; theme and progressive reveal never alter coordinates."""
    data, p = capture(sample), get_palette(theme)
    _style(ax, p, data['limits'])
    name = data['name']
    smooth_stage = 3 if name == 'bezier' else 4
    fallback_stage, raw_stage = smooth_stage+1, smooth_stage+2
    active = np.empty((0, 2))
    if stage in (0, 1):
        _grid(ax, data, p)
        count = len(_prefix(data['sampled'], progress))
        if stage == 0:
            for point in data['grid_origins'][:count]:
                ax.add_patch(Rectangle(point, data['grid_pitch'], data['grid_pitch'],
                                       facecolor=p.orange, alpha=.15, edgecolor=p.orange))
            _scatter(ax, data['grid_origins'][:count], p.blue, marker='s')
            if count:
                q = data['grid_origins'][count-1]
                ax.annotate(f"address {int(data['grid_ids'][count-1])}", q,
                            xytext=(7, 8), textcoords='offset points', color=p.ink, fontsize=11)
        else:
            _scatter(ax, data['grid_origins'], p.muted, size=17, marker='s')
            for origin, point in zip(data['grid_origins'][:count], data['sampled'][:count]):
                ax.annotate('', point, origin,
                            arrowprops=dict(arrowstyle='->', color=p.orange, lw=1.3))
            _scatter(ax, data['sampled'][:count], p.blue)
            if count:
                i = count-1
                dx, dy = data['offset_vectors'][i]
                note = ('Illustrative inputs: mapped draws stay inside the displayed cells.'
                        if sample.get('example_selection') else
                        f"Offsets may cross grid-cell boundaries (bound ±{data['offset_bound']:.2f}).")
                ax.text(.02, .02, f"Point {i}: Δx={dx:+.3f}, Δy={dy:+.3f}\n" + note,
                        transform=ax.transAxes, color=p.ink, fontsize=10)
    elif stage == 2:
        center = data['ordered'].astype(np.float64).mean(axis=0)
        _scatter(ax, data['sampled'], p.muted, size=19)
        for point in data['ordered']:
            _line(ax, np.stack([center, point]), p.grid, lw=.7)
        _scatter(ax, center[None], p.orange, size=65, marker='+')
        ax.annotate('centroid', center, xytext=(8, 8), textcoords='offset points', color=p.ink)
        active = _prefix(_close(data['ordered']), progress)
        _line(ax, active, p.blue, lw=2)
        for i, point in enumerate(data['ordered'][:min(len(active), len(data['ordered']))]):
            ax.annotate(str(i), point, xytext=(4, 4), textcoords='offset points', color=p.ink, fontsize=9)
    elif name == 'hull' and stage == 3:
        _line(ax, data['ordered'], p.muted, closed=True, lw=1)
        _scatter(ax, data['ordered'], p.blue, size=22)
        _scatter(ax, data['midpoints'], p.muted, size=17, marker='s')
        moved = data['polygon_vertices'][1::2]
        count = len(_prefix(moved, progress))
        center = data['ordered'].astype(np.float64).mean(axis=0)
        for old, new in zip(data['midpoints'][:count], moved[:count]):
            _line(ax, np.stack([center, new]), p.grid, lw=.6, ls=':')
            ax.annotate('', new, old, arrowprops=dict(arrowstyle='->', color=p.orange, lw=1.7))
        _scatter(ax, moved[:count], p.orange, marker='D')
        active = _prefix(_close(data['polygon_vertices']), progress)
        _line(ax, active, p.orange, lw=1)
    elif stage == smooth_stage:
        _line(ax, data['polygon_vertices'], p.muted, closed=True, lw=.8)
        active = _prefix(data['smooth_dense'], progress)
        _line(ax, active, p.blue, lw=2.1)
        if name == 'bezier':
            segment = min((len(active)-1)//data['parameters']['num_points_per_segment'], len(data['controls'])-1)
            controls = data['controls'][segment]
            _line(ax, controls, p.orange, lw=1.4)
            _scatter(ax, controls[1:3], p.orange, size=38)
            for label, point in zip(('P0', 'P1', 'P2', 'P3'), controls):
                ax.annotate(label, point, xytext=(4, -12), textcoords='offset points', color=p.ink, fontsize=10)
            ax.text(.02, .02, 'Handle = min(rad × chord, clamp × shorter incident edge)',
                    transform=ax.transAxes, color=p.ink, fontsize=10)
        else:
            _scatter(ax, data['ordered'], p.blue, size=20)
            _scatter(ax, data['polygon_vertices'][1::2], p.orange, size=20, marker='D')
    elif stage == fallback_stage:
        fallback = data['fallback']
        _line(ax, fallback['smooth_raw'], p.red, closed=True, lw=1.5, ls='--',
              label='rejected smooth candidate')
        _scatter(ax, fallback['crossing_points'], p.red, size=75, marker='x')
        if progress >= .35:
            active = _prefix(_close(fallback['raw']), (progress-.35)/.65)
            _line(ax, active, p.orange, lw=2.3, label='selected whole-loop polygon')
            _scatter(ax, fallback['polygon_vertices'], p.orange, size=18, marker='s')
        ax.text(0., 1.02, f"Default seed {fallback['seed']} · {fallback['smooth_crossings']} proper crossing pairs · whole-loop branch",
                transform=ax.transAxes, va='bottom', color=p.ink, fontsize=11)
        ax.legend(loc='lower left', fontsize=9)
        pair_points = fallback['crossing_pairs'].reshape(-1, 2)
        lo, hi = pair_points.min(axis=0), pair_points.max(axis=0)
        mid = .5*(lo+hi)
        half = max(float((hi-lo).max())*.7, .012)
        inset = ax.inset_axes([.65, .04, .32, .32])
        _line(inset, fallback['smooth_raw'], p.red, closed=True, lw=1.2, ls='--')
        if progress >= .35:
            _line(inset, fallback['raw'], p.orange, closed=True, lw=1.5)
        _scatter(inset, fallback['crossing_points'], p.red, size=35, marker='x')
        inset.set(xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half),
                  xticks=[], yticks=[], facecolor=p.panel)
        inset.set_aspect('equal')
        inset.set_title('Crossing detail', color=p.ink, fontsize=9)
        for spine in inset.spines.values():
            spine.set_edgecolor(p.border)
    elif stage == raw_stage:
        active = _prefix(data['raw'], progress)
        _line(ax, data['raw'], p.muted, closed=True, lw=.8)
        _scatter(ax, active, p.blue, size=11)
        ax.text(0., 1.02, f"Primary seed {data['seed']} · {len(data['raw'])} fixed-count raw samples → {data['final_count']} finished beads",
                transform=ax.transAxes, va='bottom', color=p.ink, fontsize=11)
        ax.text(.02, .02, 'num_points controls the raw count; final spacing belongs to the shared pipeline.',
                transform=ax.transAxes, color=p.ink, fontsize=10)
    else:
        raise ValueError(f'unknown storyboard stage {stage}')
    return dict(active_curve=active, theme=theme, seed=data['seed'])


def _result_axes(raw_ax, road_ax, data, p):
    limits = data['limits']
    for ax in (raw_ax, road_ax):
        _style(ax, p, limits)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
    _line(raw_ax, data['raw'], p.blue, closed=True, lw=2)
    raw_title = (("Bézier curve" if data['name'] == 'bezier' else "Hull curve")
                 if data.get('illustrative_inputs') else
                 f"Primary seed {data['seed']}\nraw n={len(data['raw'])}")
    raw_ax.set_title(raw_title, color=p.ink, fontsize=12)
    outer, inner = data['final_outer'], data['final_inner']
    cells = np.stack([outer, np.roll(outer, -1, axis=0), np.roll(inner, -1, axis=0), inner], axis=1)
    road_ax.add_collection(PolyCollection(cells, facecolors=p.road, edgecolors='none'))
    for points in (outer, inner):
        _line(road_ax, points, p.muted, closed=True, lw=.7)
    _line(road_ax, data['final_center'], p.blue, closed=True, lw=1.5)
    _scatter(road_ax, data['final_center'], p.blue, size=7)
    road_title = (f"Finished road · n={data['final_count']} · "
                  f"{'valid' if data['final_valid'] else 'invalid'}" if data.get('illustrative_inputs')
                  else f"Finished road · n={data['final_count']}\n"
                       f"{'valid' if data['final_valid'] else 'invalid'}")
    road_ax.set_title(road_title, color=p.green if data['final_valid'] else p.red, fontsize=12)


def _frame(sample, theme):
    p, data = get_palette(theme), capture(sample)
    fig = plt.figure(figsize=video_canvas_size(14, dpi=120), facecolor=p.background)
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, .9), left=.06, right=.97,
                           bottom=.15, top=.80, hspace=.25, wspace=.22)
    ax = fig.add_subplot(grid[:, :2])
    _result_axes(fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2]), data, p)
    heading = fig.text(.5, .95, '', ha='center', color=p.ink, fontsize=18)
    explanation = fig.text(.5, .875, '', ha='center', va='center', color=p.ink, fontsize=12)
    if sample.get('illustrative_inputs'):
        fig.text(.5, .915, 'Cell-bounded jitter for clarity',
                 ha='center', color=p.ink, fontsize=12)
        footer = ''
    else:
        footer = ('Recorded geometry; staged explanation, not elapsed runtime.\n'
                  'Shared tail: fixed-distance targets → XPBD → same-count uniform redistribution (see the spacing guide).')
    if footer:
        fig.text(.5, .055, footer, ha='center', color=p.ink, fontsize=10)
    return fig, ax, heading, explanation


def render(out_dir, sample, movies=True, theme='light', writer_factory=None):
    """Render a 24-second explanation and one readable representative poster."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = select_construction_sample(sample)
    data, stages = capture(sample), construction_stages(sample)
    name = data['name']
    frames, fps = 144, 6
    with plotting_theme(theme):
        if movies:
            if writer_factory is None:
                from .render_generator_guide import _writer
                writer_factory = _writer
            fig, ax, heading, explanation = _frame(sample, theme)
            writer = writer_factory()
            allocations = np.full(len(stages), frames//len(stages), dtype=int)
            allocations[:frames % len(stages)] += 1
            with writer.saving(fig, str(out_dir/f'generator-{name}-construction.mp4'), dpi=120):
                for i, (stage, count) in enumerate(zip(stages, allocations)):
                    if sample.get('illustrative_inputs'):
                        heading.set_text(('Bézier' if name == 'bezier' else 'Hull')
                                         + ' construction illustration · 10 points · 8 × 8 grid')
                        explanation.set_text(f"{i+1}/{len(stages)} · {stage['title']}\n"
                                             + textwrap.fill(stage['explanation'], width=112))
                    else:
                        heading.set_text(f"{name.capitalize()} · {i+1}/{len(stages)} · {stage['title']}")
                        explanation.set_text(textwrap.fill(stage['explanation'], width=112))
                    for progress in [*np.linspace(.02, 1., int(count)-5), *([1.]*5)]:
                        ax.clear()
                        draw_step(ax, sample, i, progress, theme=theme)
                        apply_plot_theme(fig, theme)
                        writer.grab_frame()
            plt.close(fig)
        fig, ax, heading, explanation = _frame(sample, theme)
        heading.set_text(("Bézier" if name == "bezier" else "Hull")
                         + " construction illustration · 10 points · 8 × 8 grid"
                         if sample.get("illustrative_inputs") else f"{name.capitalize()} · grid addresses and actual offsets")
        explanation.set_text(textwrap.fill(stages[1]['explanation'], width=112))
        draw_step(ax, sample, 1, 1., theme=theme)
        save_figure(fig, out_dir/f'generator-{name}-construction.png', theme=theme, dpi=120)
        plt.close(fig)
    from .render_generator_guide import _record
    record = dict(kind=(f'construction illustration with remapped inputs and real {"Bezier" if name == "bezier" else "Hull"}/road stages'
                        if sample.get('illustrative_inputs') else
                        'progressive construction drawing of recorded geometry'), sample=_record(sample),
                stage_titles=[stage['title'] for stage in stages], stages=len(stages),
                frames=frames if movies else 0, duration_seconds=frames/fps if movies else 0,
                frame_size=[int(round(x*120)) for x in video_canvas_size(14, dpi=120)],
                poster_frame_size=[int(round(x*120)) for x in video_canvas_size(14, dpi=120)],
                aspect_ratio='16:9',
                geometry_interpolated=False, elapsed_runtime=False, theme=theme,
                raw_count=len(data['raw']), final_count=data['final_count'],
                observation_preserves_outputs=data['observation_preserves_outputs'],
                effective_generator_params=data['parameters'],
                fallback_example=dict(seed=data['fallback']['seed'],
                                      proper_crossings=data['fallback']['smooth_crossings'],
                                      automatic_whole_loop=True, settings='current defaults'),
                output=f'generator-{name}-construction.' + ('mp4' if movies else 'png'))
    if sample.get('example_selection'):
        record['example_selection'] = dict(sample['example_selection'])
    if sample.get('illustrative_inputs'):
        record.update(illustrative_inputs=True, production_sampler_unchanged=True,
                      sampler_remapping=sample['example_selection']['sampler_remapping'],
                      illustration_replay_equal=data['illustration_replay_equal'],
                      ordinary_outputs_restored=data['ordinary_outputs_restored'],
                      source_sampler_points=data['source_sampler_points'].tolist(),
                      source_sampler_noise=data['source_sampler_noise'].tolist(),
                      illustrated_sampled_points=data['sampled'].tolist(),
                      grid_ids=data['grid_ids'].tolist())
    return record
