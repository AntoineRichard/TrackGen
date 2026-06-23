import pytest
import torch

pytest.importorskip("warp")

from track_gen import PerEnvSeededRNG
from track_gen._src.track_generator import Track, TrackGenConfig, TrackGenerator
from tests._warp_compare import to_t


def _make_rng(num_envs, device="cpu"):
    import warp as wp
    wp.init()
    return PerEnvSeededRNG(seeds=0, num_envs=num_envs, device=device)


def test_bezier_path_returns_track_with_aligned_boundaries():
    # constant_spacing output: boundary arrays are wp.array with E*N_max vec2f elements
    # (flat [E*N_max] storage). The second dim is N_max (NOT num_points / not a per-env
    # count), so we set N_max explicitly for a deterministic shape assertion.
    E, N_max = 4, 128
    cfg = TrackGenConfig(
        generator="bezier", num_envs=E, num_points=64, N_max=N_max, device="cpu"
    )
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)

    assert isinstance(track, Track)
    # All boundary arrays are wp.array with flat E*N_max vec2f storage; reshape to [E,N,2]
    # via to_t() for shape assertions.
    outer_t = to_t(track.outer).view(E, N_max, 2)
    center_t = to_t(track.center).view(E, N_max, 2)
    inner_t = to_t(track.inner).view(E, N_max, 2)
    assert outer_t.shape == (E, N_max, 2)
    assert center_t.shape == (E, N_max, 2)
    assert inner_t.shape == (E, N_max, 2)
    valid_t = to_t(track.valid).bool()
    assert valid_t.shape == (E,)
    assert valid_t.dtype == torch.bool

    # count is a per-env real-point count in [1, N_max].
    count_t = to_t(track.count)
    assert count_t.shape == (E,)
    assert torch.all(count_t >= 1)
    assert torch.all(count_t <= N_max)

    # Boundaries are index-aligned: outer[e], center[e], inner[e] share one
    # cross-section normal, so their finite masks must agree per env, and the finite
    # region must be exactly the first count[e] rows (real points finite, padding NaN).
    for e in range(E):
        c = int(count_t[e])
        finite_center = torch.isfinite(center_t[e]).all(dim=-1)
        finite_outer = torch.isfinite(outer_t[e]).all(dim=-1)
        finite_inner = torch.isfinite(inner_t[e]).all(dim=-1)
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


def test_generate_reuses_output_buffers():
    """TrackGenerator.generate() returns the same Track instance on every call.

    The same wp.array pointers (stable device addresses) are reused, so callers
    can register persistent tensor views before the first generate and know those
    views will reflect updated values after each subsequent generate call.
    """
    import warp as wp

    E = 8
    cfg = TrackGenConfig(num_envs=E, num_points=64, N_max=128, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    t1 = gen.generate(E)
    # Record the device pointer of the center buffer.
    p1 = t1.center.ptr
    # Register a persistent torch view that shares the same memory.
    view = wp.to_torch(t1.center)

    # Clobber the shared buffer via the view so we can detect in-place overwrite.
    view.fill_(-999.0)

    # Second generate call — must return the identical Track instance AND write new
    # values into the same buffer (overwriting the sentinel we just wrote).
    t2 = gen.generate(E)

    assert t2 is t1, "generate() must return the same Track object every call"
    assert t2.center.ptr == p1, "center buffer pointer must be stable across calls"
    # The torch view (registered before t2) must alias t2.center (same memory).
    # Both are views of the same wp.array buffer, so their data_ptr must match.
    view2 = wp.to_torch(t2.center)
    assert view.data_ptr() == view2.data_ptr(), \
        "pre-registered torch view must share the same buffer as t2.center"
    # In-place write proof: generate() must have overwritten the -999 sentinel.
    assert not torch.all(view == -999.0), \
        "generate() must overwrite the shared buffer in place (sentinel not cleared)"


def test_relax_solver_energy_raises():
    """TrackGenerator must reject relax_solver != 'xpbd' at construction time."""
    cfg = TrackGenConfig(relax_solver="energy", num_envs=4)
    rng = _make_rng(4)
    with pytest.raises(AssertionError):
        TrackGenerator(cfg, rng)


def test_smooth_finish_raises():
    """TrackGenerator must reject smooth_finish=True at construction time."""
    cfg = TrackGenConfig(smooth_finish=True, num_envs=4)
    rng = _make_rng(4)
    with pytest.raises(AssertionError):
        TrackGenerator(cfg, rng)


def test_generate_wrong_batch_raises():
    """generate() must raise ValueError when num_or_ids count != num_envs."""
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)
    with pytest.raises(ValueError):
        gen.generate(E + 1)


def test_generate_rejects_sequence_ids():
    """generate() is fixed-batch; explicit env-id selection is not supported."""
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)
    with pytest.raises(TypeError, match="does not accept explicit environment ids"):
        gen.generate([0, 1, 2, 3])
