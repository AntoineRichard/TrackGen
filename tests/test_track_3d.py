import numpy as np
import pytest
import warp as wp

from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

E = 8


def _gen(**kw):
    cfg = TrackGenConfig(device="cpu", num_envs=E, **kw)
    rng = PerEnvSeededRNG(seeds=1234, num_envs=E, device="cpu")
    return TrackGenerator(cfg, rng).generate(), cfg


def _env(track, cfg, e):
    n_max = track.center.shape[0] // E
    m = int(track.count.numpy()[e])
    sl = slice(e * n_max, e * n_max + m)
    return (track.center.numpy()[sl], track.outer.numpy()[sl],
            track.inner.numpy()[sl], track.arclen.numpy()[sl],
            float(track.length.numpy()[e]))


def test_walk_profile_lifts_level_cross_sections():
    track, cfg = _gen(z_profile="random_walk", z_base=1.0, z_min=0.2,
                      z_max=2.0, z_max_step=0.3)
    valid = track.valid.numpy()
    assert valid.any()
    for e in np.flatnonzero(valid):
        ctr, out, inn, _, _ = _env(track, cfg, e)
        assert (ctr[:, 2] >= 0.2 - 1e-5).all() and (ctr[:, 2] <= 2.0 + 1e-5).all()
        assert ctr[:, 2].std() > 0.0
        np.testing.assert_allclose(out[:, 2], ctr[:, 2], atol=0.0)  # level
        np.testing.assert_allclose(inn[:, 2], ctr[:, 2], atol=0.0)


def test_3d_arclen_consistency():
    track, cfg = _gen(z_profile="noise", z_base=1.0, z_noise_amplitude=0.5,
                      z_min=0.0, z_max=2.0)
    for e in np.flatnonzero(track.valid.numpy()):
        ctr, _, _, arc, length = _env(track, cfg, e)
        chords = np.linalg.norm(np.diff(ctr, axis=0), axis=1)
        np.testing.assert_allclose(np.diff(arc), chords, atol=1e-4)
        closing = np.linalg.norm(ctr[0] - ctr[-1])
        np.testing.assert_allclose(length, arc[-1] + closing, atol=1e-4)
        # a hilly loop is strictly longer in 3D than in plan view
        plan = np.linalg.norm(np.diff(ctr[:, :2], axis=0), axis=1).sum()
        assert arc[-1] > plan


def test_walk_closes_loop():
    track, cfg = _gen(z_profile="random_walk", z_base=1.0, z_min=0.2,
                      z_max=2.0, z_max_step=0.3)
    for e in np.flatnonzero(track.valid.numpy()):
        ctr, _, _, _, _ = _env(track, cfg, e)
        assert abs(ctr[0, 2] - ctr[-1, 2]) < 0.5


def test_grade_validity_and_disable():
    _, _ = _gen()  # warm
    t_off, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0)
    t_on, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0,
                   z_valid_grade=0.5)
    assert t_on.valid.numpy().sum() < max(1, t_off.valid.numpy().sum())


def test_flat_matches_goldens_locally():
    track, cfg = _gen()
    for e in np.flatnonzero(track.valid.numpy()):
        ctr, _, _, _, _ = _env(track, cfg, e)
        assert (ctr[:, 2] == 0.0).all()


def test_standalone_inflate_rejects_nonflat_without_seeds():
    from track_gen._src import warp_pipeline
    cfg = TrackGenConfig(device="cpu", num_envs=1, z_profile="uniform",
                         z_min=0.5, z_max=1.0)
    center = wp.zeros(cfg.num_points, dtype=wp.vec2f, device="cpu")
    with pytest.raises(ValueError, match="seeds"):
        warp_pipeline.inflate_warp(center, cfg)


def test_track_frames3_padding_is_nan_and_degenerate_row_zeroed():
    """Regression for the review finding that _track_frames3_k's non-flat path
    left tangent/arclen padding slots [count[e], n_max) as uninitialized
    wp.empty() garbage instead of NaN-filling them (violating the types.py
    Track NaN-padding contract), and that the m < 3 branch returned without
    NaN-filling the row or zeroing length[e].
    """
    from track_gen._src import warp_pipeline

    # Part 1: a normal non-flat batch. Every env's padding slots (i >=
    # count[e]) must be all-NaN in both tangent and arclen.
    track, _ = _gen(z_profile="noise", z_base=1.0, z_noise_amplitude=0.5,
                    z_min=0.0, z_max=2.0)
    n_max = track.center.shape[0] // E
    tangent = track.tangent.numpy().reshape(E, n_max, 3)
    arclen = track.arclen.numpy().reshape(E, n_max)
    count = track.count.numpy()
    any_padded = False
    for e in range(E):
        m = int(count[e])
        if m < n_max:
            any_padded = True
            assert np.isnan(tangent[e, m:]).all(), f"env {e} tangent padding not NaN"
            assert np.isnan(arclen[e, m:]).all(), f"env {e} arclen padding not NaN"
    assert any_padded, "test batch has no padded envs -- widen num_points/spacing"

    # Part 2: count < 3 (degenerate row). The normal generator doesn't reliably
    # produce this, so construct it directly via the standalone inflate path:
    # a 3-env batch where env 1 has only 2 real points (the other two envs are
    # ordinary hexagons, so the shared kernels don't choke on a wholly
    # degenerate batch).
    E2, N = 3, 6
    cfg2 = TrackGenConfig(device="cpu", num_envs=E2, z_profile="uniform",
                         z_min=0.0, z_max=1.0)
    pts = np.zeros((E2, N, 2), dtype=np.float32)
    ang = 2.0 * np.pi * np.arange(N) / N
    for k in (0, 2):
        pts[k, :, 0] = 10.0 * np.cos(ang)
        pts[k, :, 1] = 10.0 * np.sin(ang)
    pts[1, 0] = [0.0, 0.0]
    pts[1, 1] = [1.0, 0.0]
    # pts[1, 2:] are unread: resample/frame/arclength kernels only iterate
    # i in [0, count[e]) for their env.
    center = wp.array(pts.reshape(E2 * N, 2), dtype=wp.vec2f, device="cpu")
    count_wp = wp.array([N, 2, N], dtype=wp.int32, device="cpu")
    seeds_wp = wp.array([1, 2, 3], dtype=wp.int32, device="cpu")

    out = warp_pipeline.inflate_warp(center, cfg2, count=count_wp, seeds=seeds_wp)
    assert int(out.count.numpy()[1]) == 2
    assert float(out.length.numpy()[1]) == 0.0
    tan1 = out.tangent.numpy().reshape(E2, N, 3)[1]
    arc1 = out.arclen.numpy().reshape(E2, N)[1]
    assert np.isnan(tan1).all()
    assert np.isnan(arc1).all()


def _turns(z):
    """Number of direction changes in an elevation series."""
    return int(np.sum(np.diff(np.sign(np.diff(z))) != 0))


def test_smoothness_is_independent_of_resample_density():
    """THE regression gate. Elevation direction changes must track
    z_control_points, not the resampled point count: doubling the density
    must not roughly double the bumps (which is what per-point profiling did).
    """
    counts, turns = [], []
    # 0.035 is about the finest feasible spacing at the default half_width=0.1:
    # below it the thickness gate rejects every track (flat profile included, so
    # this is a plan-view constraint, nothing to do with elevation). It still
    # gives ~1.7x the point count of 0.06, which is the density contrast the
    # assertions below need.
    for spacing in (0.06, 0.035):
        cfg = TrackGenConfig(device="cpu", num_envs=E, spacing=spacing,
                             z_profile="random_walk", z_base=1.0, z_min=0.2,
                             z_max=2.0, z_max_step=0.3, z_control_points=8)
        rng = PerEnvSeededRNG(seeds=1234, num_envs=E, device="cpu")
        track = TrackGenerator(cfg, rng).generate()
        e = int(np.flatnonzero(track.valid.numpy())[0])
        n_max = track.center.shape[0] // E
        m = int(track.count.numpy()[e])
        z = track.center.numpy().reshape(E, n_max, 3)[e, :m, 2]
        counts.append(m)
        turns.append(_turns(z))
    assert counts[1] > 1.5 * counts[0], f"densities not distinct: {counts}"
    assert turns[0] <= 8 and turns[1] <= 8, f"too many turns: {turns}"
    assert abs(turns[1] - turns[0]) <= 2, f"turns track density: {turns}"


def test_uniform_profile_is_smooth_not_jitter():
    cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="uniform",
                         z_min=0.5, z_max=1.5, z_control_points=10)
    rng = PerEnvSeededRNG(seeds=1234, num_envs=E, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    n_max = track.center.shape[0] // E
    for e in np.flatnonzero(track.valid.numpy()):
        m = int(track.count.numpy()[e])
        z = track.center.numpy().reshape(E, n_max, 3)[e, :m, 2]
        assert (z >= 0.5 - 1e-5).all() and (z <= 1.5 + 1e-5).all()
        assert _turns(z) <= 10, f"env {e}: {_turns(z)} turns"
        assert z.std() > 0.0


def test_xpbd_solves_in_2d_elevation_applies_after():
    """INVARIANT: relaxation runs before elevation exists, so the plan-view
    geometry must be bit-identical between a flat and a hilly config."""
    def gen(**kw):
        cfg = TrackGenConfig(device="cpu", num_envs=E, **kw)
        rng = PerEnvSeededRNG(seeds=99, num_envs=E, device="cpu")
        return TrackGenerator(cfg, rng).generate()

    flat = gen()
    hilly = gen(z_profile="random_walk", z_base=1.0, z_min=0.2, z_max=2.0,
                z_max_step=0.3)
    np.testing.assert_array_equal(flat.count.numpy(), hilly.count.numpy())
    np.testing.assert_array_equal(flat.valid.numpy(), hilly.valid.numpy())
    for name in ("center", "outer", "inner"):
        a = getattr(flat, name).numpy()[:, :2]
        b = getattr(hilly, name).numpy()[:, :2]
        np.testing.assert_array_equal(a, b, err_msg=f"{name} xy moved")
    assert getattr(hilly, "center").numpy()[:, 2].std() > 0.0  # z really varies


def test_no_overshoot_beyond_extremes():
    cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="random_walk",
                         z_base=1.0, z_min=0.6, z_max=1.4, z_max_step=0.4,
                         z_control_points=6)
    rng = PerEnvSeededRNG(seeds=7, num_envs=E, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    n_max = track.center.shape[0] // E
    for e in np.flatnonzero(track.valid.numpy()):
        m = int(track.count.numpy()[e])
        z = track.center.numpy().reshape(E, n_max, 3)[e, :m, 2]
        assert z.max() <= 1.4 + 1e-5 and z.min() >= 0.6 - 1e-5


def test_noise_profile_unchanged_by_control_points():
    """noise stays analytic per-point: z_control_points must not affect it."""
    def gen(K):
        cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="noise",
                             z_base=1.0, z_noise_amplitude=0.4, z_min=0.0,
                             z_max=2.0, z_control_points=K)
        rng = PerEnvSeededRNG(seeds=21, num_envs=E, device="cpu")
        return TrackGenerator(cfg, rng).generate().center.numpy()[:, 2]
    np.testing.assert_array_equal(gen(4), gen(20))
