"""Analytic annulus tests for track_gen.props (boundary prop sampling)."""
from __future__ import annotations

import numpy as np

from tests._collision_fixtures import annulus_polylines, make_annulus_track

N = 512
N_MAX = N + 8
RO = 1.3  # outer boundary radius of the default annulus fixture
RI = 0.7


def _outer_perimeter(track, e=0):
    _, outer = annulus_polylines(track, e, N_MAX)
    seg = np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1)
    return float(seg.sum())


def test_import_surface():
    import track_gen
    from track_gen.props import PropSampler, PropSet  # noqa: F401
    assert "props" in track_gen.__all__
    assert track_gen.props.PropSampler is PropSampler


def test_points_mode_snapped_count_and_step():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    spacing = 0.1
    sampler = PropSampler(track, spacing=spacing, boundary="outer", mode="points")
    props = sampler.sample()
    perim = _outer_perimeter(track)
    n_expected = int(round(perim / spacing))
    assert int(props.count.numpy()[0]) == n_expected
    np.testing.assert_allclose(props.step.numpy()[0], perim / n_expected, rtol=1e-5)
    assert int(props.truncated.numpy()[0]) == 0


def test_points_mode_positions_on_circle_uniform_gaps():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    # On the outer circle (polyline tolerance).
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), RO, atol=2e-3)
    # Uniform angular gaps that close the ring: n gaps of 2*pi/n each.
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    gaps = np.diff(np.concatenate([ang, ang[:1]]))
    gaps = np.mod(gaps, 2 * np.pi)
    np.testing.assert_allclose(gaps, 2 * np.pi / n, atol=2e-3)


def test_points_mode_tangent_yaw_length():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    tang = props.tangent.numpy().reshape(-1, 2)[:n]
    yaw = props.yaw.numpy()[:n]
    length = props.length.numpy()[:n]
    # Unit tangents, perpendicular to the radial direction (circle tangent).
    np.testing.assert_allclose(np.linalg.norm(tang, axis=1), 1.0, atol=1e-5)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert np.abs((tang * radial).sum(axis=1)).max() < 0.02
    np.testing.assert_allclose(yaw, np.arctan2(tang[:, 1], tang[:, 0]), atol=1e-6)
    np.testing.assert_allclose(length, props.step.numpy()[0], rtol=1e-5)


def test_nan_padding_past_count():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    assert sampler._M > n
    pos = props.position.numpy().reshape(-1, 2)
    assert np.all(np.isnan(pos[n:]))
    assert np.all(np.isnan(props.yaw.numpy()[n:]))
    assert np.all(np.isnan(props.length.numpy()[n:]))


def test_segments_mode_chords_and_closure():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="segments")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    tang = props.tangent.numpy().reshape(-1, 2)[:n]
    length = props.length.numpy()[:n]
    # Chord across one arc step of the circle: 2*R*sin(pi/n).
    np.testing.assert_allclose(length, 2 * RO * np.sin(np.pi / n), atol=2e-3)
    # Chord midpoints sit slightly inside the circle: radius R*cos(pi/n).
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1),
                               RO * np.cos(np.pi / n), atol=2e-3)
    # Ring closure: each chord's end == next chord's start (wraps at n-1 -> 0).
    starts = pos - tang * (length[:, None] / 2.0)
    ends = pos + tang * (length[:, None] / 2.0)
    np.testing.assert_allclose(ends, np.roll(starts, -1, axis=0), atol=1e-4)


def test_inner_boundary_sampling():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    props = PropSampler(track, spacing=0.1, boundary="inner", mode="points").sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), RI, atol=2e-3)


def test_truncation_flag_and_closed_ring():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points",
                          max_props=10)
    props = sampler.sample()
    assert int(props.count.numpy()[0]) == 10
    assert int(props.truncated.numpy()[0]) == 1
    perim = _outer_perimeter(track)
    np.testing.assert_allclose(props.step.numpy()[0], perim / 10, rtol=1e-5)
    # Still a closed uniform ring at the coarser effective spacing.
    pos = props.position.numpy().reshape(-1, 2)[:10]
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    gaps = np.mod(np.diff(np.concatenate([ang, ang[:1]])), 2 * np.pi)
    np.testing.assert_allclose(gaps, 2 * np.pi / 10, atol=5e-3)


def test_degenerate_env_zero_count_nan():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=2, n=N, counts=[N, 2])  # env 1 degenerate
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points",
                          max_props=128)
    props = sampler.sample()
    counts = props.count.numpy()
    assert counts[0] > 0 and counts[1] == 0
    assert np.isnan(props.step.numpy()[1])
    M = sampler._M
    pos = props.position.numpy().reshape(-1, 2)
    assert np.all(np.isnan(pos[M:2 * M]))  # env 1 slots all NaN


def test_max_props_auto_derivation():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    spacing = 0.1
    sampler = PropSampler(track, spacing=spacing, boundary="outer", mode="points")
    perim = _outer_perimeter(track)
    assert sampler._M == max(3, int(np.ceil(1.5 * perim / spacing)))
    assert int(sampler.sample().truncated.numpy()[0]) == 0


def test_max_props_derivation_requires_valid_env():
    import warp as wp
    import pytest
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    wp.copy(track.valid, wp.zeros(1, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="max_props"):
        PropSampler(track, spacing=0.1)
    # Explicit max_props still works with no valid env.
    PropSampler(track, spacing=0.1, max_props=64)


def test_constructor_validation():
    import pytest
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="spacing"):
        PropSampler(track, spacing=0.0)
    with pytest.raises(ValueError, match="boundary"):
        PropSampler(track, spacing=0.1, boundary="center")
    with pytest.raises(ValueError, match="mode"):
        PropSampler(track, spacing=0.1, mode="walls")
    with pytest.raises(ValueError, match="max_props"):
        PropSampler(track, spacing=0.1, max_props=2)
