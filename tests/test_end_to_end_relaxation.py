import torch
import pytest
from tests._oracle import geometry


@pytest.fixture
def warp_rng():
    pytest.importorskip("warp")
    import warp as wp; wp.init()
    from track_gen._src.rng_utils import PerEnvSeededRNG

    def make(E, seed=20):
        seeds = torch.arange(E, dtype=torch.int32) + seed
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
        rng.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
        return rng
    return make


def test_xpbd_pipeline_makes_constant_width_tracks_valid(warp_rng):
    from track_gen._src.types import TrackGenConfig
    from track_gen._src.track_generator import TrackGenerator
    E = 32
    # constant_spacing is the only mode; pin spacing/N_max for determinism (spacing
    # would otherwise auto-resolve to 0.6*half_width = 0.018 anyway).
    cfg = TrackGenConfig(generator="bezier", device="cpu", num_envs=E, scale=1.0,
                         half_width=0.03, num_points=256, output_mode="constant_spacing",
                         spacing=0.018, N_max=256,
                         relax_solver="xpbd", relax_iters=200, relax_bend_relax=1.5,
                         relax_margin=0.15, max_regen_iters=20)
    track = TrackGenerator(cfg, warp_rng(E)).generate(E)
    # Relaxed + constant-width inflation: a large majority must be valid (was ~3% before).
    assert track.valid.float().mean().item() >= 0.9
    # Count-aware output: per-env real-point count is in [1, N_max]; the centerline is
    # finite over the real prefix [:count[e]] and NaN-padded beyond it.
    assert track.count.min().item() >= 1
    assert track.count.max().item() <= cfg.N_max
    # Width is constant (== half_width) at every real point, masking out the NaN padding.
    w = torch.linalg.norm(track.outer - track.center, dim=-1)  # [E, N_max], NaN past count
    real = torch.isfinite(w)
    # The finite (real) width slots are exactly the per-env real points.
    assert int(real.sum().item()) == int(track.count.sum().item())
    for e in range(E):
        c = int(track.count[e].item())
        # Real prefix is finite; padding beyond count is NaN.
        assert torch.isfinite(track.center[e, :c]).all()
        if c < cfg.N_max:
            assert torch.isnan(track.center[e, c:]).all()
        # Constant width over this env's real points.
        we = w[e, :c]
        assert torch.allclose(we, torch.full_like(we, 0.03), atol=1e-3)
