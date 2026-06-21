import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from benchmarks import compare_generators as cg


def _gen(seed, E=64):
    cfg = TrackGenConfig(generator="grammar", device="cpu", num_envs=E, half_width=0.1, relax_iters=40)
    return TrackGenerator(cfg, PerEnvSeededRNG(seeds=seed, num_envs=E, device="cpu")).generate(E)


def test_grammar_e2e_finite_n_points():
    E = 64
    t = _gen(0, E)
    center = wp.to_torch(t.center).cpu().numpy().reshape(E, -1, 2)
    count = wp.to_torch(t.count).cpu().numpy().astype(int)
    valid = wp.to_torch(t.valid).cpu().numpy().astype(bool)
    assert valid.mean() > 0.5
    for e in range(E):
        c = int(count[e])
        assert 1 <= c <= center.shape[1]
        assert np.isfinite(center[e, :c]).all()


def test_grammar_is_deterministic_within_device():
    a = wp.to_torch(_gen(7).center).cpu().numpy()
    b = wp.to_torch(_gen(7).center).cpu().numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_grammar_adds_net_new_features_vs_other_generators():
    # The whole point: grammar makes sustained STRAIGHTS the star-shaped generators
    # structurally cannot (they have no kappa=0 spans). straight_fraction is the feature
    # metric that captures this and survives relaxation; assert grammar clearly leads it.
    #
    # NOTE: do NOT assert mean_chicanes > others. Post-relax, chicane_count counts turn-angle
    # SIGN REVERSALS on the dense relaxed centerline (i.e. wiggliness). The star-shaped
    # generators wander, so they score HIGHER on it; grammar is net-winding (mostly one
    # direction with a few deliberate chicanes), so it scores LOWER by design. The metric is
    # anti-correlated with grammar's character, so it is the wrong gate (measured: grammar
    # chicane_count ~9.8 vs bezier ~13.2 / polar ~12.7 at E=512, hw=0.1).
    cfg = TrackGenConfig(device="cpu", num_envs=128, half_width=0.1, relax_iters=40)
    rows = {r["generator"]: r for r in cg.compare(["bezier", "polar", "hull", "grammar"],
                                                  seed_base=0, E=128, base_config=cfg)}
    g = rows["grammar"]
    others_straight = max(rows[k]["straight_frac"] for k in ("bezier", "polar", "hull"))
    assert g["straight_frac"] > others_straight, (g["straight_frac"], others_straight)
    assert g["shape_variety_pass"]  # not degenerate (median compactness < 0.65)
