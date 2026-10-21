"""N_max truncation fails closed.

The constant-spacing resample needs ``k = floor(perimeter / spacing) + 1`` points per env
but has only ``N_max`` slots. A truncated env used to be clamped silently: the dropped arc
became one long chord, relaxation smoothed it into a plausible loop, and the track passed
validity. The only diagnostic was a host readback that never ran on the CUDA graph path.

Now the scan writes a per-env ``truncated`` flag on device (from the unclamped k), the
validity gate ANDs ``!truncated``, and the flag is exposed as ``Track.truncated``.
"""
import warnings

import numpy as np
import pytest
import warp as wp

from track_gen._src import warp_pipeline as wpl
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import TrackGenConfig

DEVS = [
    "cpu",
    pytest.param("cuda", marks=[
        pytest.mark.cuda,
        pytest.mark.skipif(not wp.is_cuda_available(), reason="cuda"),
    ]),
]

_N = 512  # dense input points per synthetic loop


def _ngon_radius(perimeter: float) -> float:
    # A regular N-gon of circumradius r has perimeter 2*N*r*sin(pi/N).
    return perimeter / (2.0 * _N * np.sin(np.pi / _N))


def _loops_wp(perimeters, dev):
    ang = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    rows = []
    for p in perimeters:
        if p is None:  # never-accepted env: all-NaN centerline
            rows.append(np.full((_N, 2), np.nan))
        else:
            r = _ngon_radius(p)
            rows.append(np.stack([r * np.cos(ang), r * np.sin(ang)], -1))
    return wp.array(np.concatenate(rows).astype(np.float32), dtype=wp.vec2f, device=dev)


@pytest.mark.parametrize("dev", DEVS)
def test_resample_flags_truncation_from_unclamped_count(dev):
    n_max, spacing = 64, 0.1
    perims = [
        1.1 * n_max * spacing,              # needs ~71 points -> truncated
        0.5 * n_max * spacing,              # fits comfortably
        (n_max - 1) * spacing + 0.05,       # needs exactly n_max points -> fits
        None,                               # NaN env -> count 0, not "truncated"
    ]
    E = len(perims)
    out = wp.empty(E * n_max, dtype=wp.vec2f, device=dev)
    count = wp.empty(E, dtype=wp.int32, device=dev)
    seg = wp.empty(E * _N, dtype=wp.float32, device=dev)
    s = wp.empty(E * (_N + 1), dtype=wp.float32, device=dev)
    truncated = wp.full(E, -1, dtype=wp.int32, device=dev)

    wpl.resample_constant_spacing(_loops_wp(perims, dev), spacing, n_max,
                                  out_wp=out, count_wp=count, seg_wp=seg, s_wp=s,
                                  truncated_wp=truncated)

    assert truncated.numpy().tolist() == [1, 0, 0, 0]
    cnt = count.numpy()
    assert cnt[0] == n_max and cnt[2] == n_max  # both full; only one lost points
    assert 0 < cnt[1] < n_max
    assert cnt[3] == 0


def test_inplace_resample_does_no_host_readback():
    """The in-place (pipeline) path never reads count back to warn; the flag is the
    diagnostic. The allocating convenience path still warns on truncation."""
    n_max, spacing = 64, 0.1
    src = _loops_wp([1.1 * n_max * spacing], "cpu")
    out = wp.empty(n_max, dtype=wp.vec2f, device="cpu")
    count = wp.empty(1, dtype=wp.int32, device="cpu")
    truncated = wp.empty(1, dtype=wp.int32, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        wpl.resample_constant_spacing(src, spacing, n_max, out_wp=out, count_wp=count,
                                      truncated_wp=truncated)
    assert int(truncated.numpy()[0]) == 1

    with pytest.warns(RuntimeWarning, match="N_max"):
        wpl.resample_constant_spacing(src, spacing, n_max, count_wp=count)  # out_wp allocated


def _gen(dev, E, n_max, seed=1234):
    cfg = TrackGenConfig(num_envs=E, device=dev, N_max=n_max)
    return TrackGenerator(cfg, PerEnvSeededRNG(seeds=seed, num_envs=E, device=dev))


@pytest.mark.parametrize("dev", DEVS)
def test_truncated_tracks_are_flagged_and_invalid(dev):
    """Force truncation with a small N_max (default bezier needs ~57-95 points at the
    default spacing): every env is flagged and fails validity, on the first call and,
    on CUDA, on graph replay."""
    E, n_max = 8, 32
    gen = _gen(dev, E, n_max)
    for _ in range(2):  # cuda: capture + replay
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            track = gen.generate()
        assert track.truncated is not None
        assert track.truncated.shape == (E,) and track.truncated.dtype == wp.int32
        np.testing.assert_array_equal(track.truncated.numpy(), np.ones(E, np.int32))
        np.testing.assert_array_equal(track.valid.numpy(), np.zeros(E, np.int32))
        np.testing.assert_array_equal(track.count.numpy(), np.full(E, n_max, np.int32))


def test_truncation_flag_is_per_env():
    """In a mixed batch only the envs that needed more than N_max points are flagged
    and invalidated; the others are unaffected."""
    E = 8
    roomy = _gen("cpu", E, 384).generate()
    need = roomy.count.numpy().copy()  # unclamped point counts (all < 384)
    assert not roomy.truncated.numpy().any()
    n_max = int(np.median(need))
    expected = (need > n_max).astype(np.int32)
    assert 0 < expected.sum() < E  # the batch really is mixed

    track = _gen("cpu", E, n_max).generate()
    np.testing.assert_array_equal(track.truncated.numpy(), expected)
    assert not track.valid.numpy()[expected == 1].any()
    np.testing.assert_array_equal(track.valid.numpy()[expected == 0],
                                  roomy.valid.numpy()[expected == 0])


def test_clone_copies_truncated():
    track = _gen("cpu", 4, 32).generate()
    snap = track.clone()
    assert snap.truncated is not track.truncated
    np.testing.assert_array_equal(snap.truncated.numpy(), track.truncated.numpy())
    track.truncated.zero_()
    assert snap.truncated.numpy().all()
