import pytest
import torch

pytest.importorskip("warp")

from track_gen import PerEnvSeededRNG
from track_gen._src.track_generator import Track, TrackGenConfig, TrackGenerator


def _make_rng(num_envs, device="cpu"):
    import warp as wp
    wp.init()
    return PerEnvSeededRNG(seeds=0, num_envs=num_envs, device=device)


def test_bezier_path_returns_track_with_aligned_boundaries():
    # constant_spacing output: boundary arrays are [E, N_max, 2] NaN-padded, with a
    # per-env real-point count in track.count (real points live in [:count[e]]). The
    # second dim is N_max (NOT num_points / not a per-env count), so we set N_max
    # explicitly for a deterministic shape assertion.
    E, N_max = 4, 128
    cfg = TrackGenConfig(
        generator="bezier", num_envs=E, num_points=64, N_max=N_max, device="cpu"
    )
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)

    assert isinstance(track, Track)
    # All boundary arrays share the padded [E, N_max, 2] shape.
    assert track.outer.shape == (E, N_max, 2)
    assert track.center.shape == (E, N_max, 2)
    assert track.inner.shape == (E, N_max, 2)
    assert track.valid.shape == (E,)
    assert track.valid.dtype == torch.bool

    # count is a per-env real-point count in [1, N_max].
    assert track.count.shape == (E,)
    assert torch.all(track.count >= 1)
    assert torch.all(track.count <= N_max)

    # Boundaries are index-aligned: outer[e], center[e], inner[e] share one
    # cross-section normal, so their finite masks must agree per env, and the finite
    # region must be exactly the first count[e] rows (real points finite, padding NaN).
    for e in range(E):
        c = int(track.count[e])
        finite_center = torch.isfinite(track.center[e]).all(dim=-1)
        finite_outer = torch.isfinite(track.outer[e]).all(dim=-1)
        finite_inner = torch.isfinite(track.inner[e]).all(dim=-1)
        # Aligned: all three boundaries finite at exactly the same slots.
        assert torch.equal(finite_center, finite_outer)
        assert torch.equal(finite_center, finite_inner)
        # Real points are the first count[e] rows; the rest are NaN-padded.
        assert torch.all(finite_center[:c])
        assert not torch.any(finite_center[c:])


def test_fourier_generator_rejected():
    # The Fourier generator was not ported to Warp; the pure-Warp facade supports
    # generator="bezier" only and rejects "fourier" at construction. (The
    # FourierCenterlineGenerator class itself remains available as a torch primitive.)
    cfg = TrackGenConfig(generator="fourier", num_envs=4, num_points=64, device="cpu")
    rng = _make_rng(4)
    with pytest.raises(ValueError):
        TrackGenerator(cfg, rng)


def test_unknown_generator_raises():
    cfg = TrackGenConfig(generator="spline", num_envs=2, device="cpu")
    rng = _make_rng(2)
    with pytest.raises(ValueError):
        TrackGenerator(cfg, rng)
