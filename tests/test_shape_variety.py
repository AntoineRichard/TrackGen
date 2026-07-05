"""Shape-variety regression gate — 'no silent circles'.

Every registered first-stage generator must produce a NON-degenerate (non-circular) shape
distribution. polar once shipped degenerate (compactness ~0.97 — visually circles) while its
yield / self-intersection / curvature metrics all looked perfect, because a circle aces those.
This gate judges geometry directly: the median post-relax compactness over a seed batch must
be clearly below a circle's 1.0. A regression that makes any generator collapse toward circles
(or a new generator that ships degenerate) fails here.
"""
import numpy as np
import pytest
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src import generator_registry
from benchmarks import track_metrics as tm

pytestmark = pytest.mark.slow

# A circle is 1.0; the healthy generators sit at compactness median ~0.4-0.56. 0.85 leaves
# wide margin for legitimately-smooth generators (e.g. polar) while still catching the
# near-circular degeneracy (median ~0.95+).
_MEDIAN_COMPACTNESS_MAX = 0.85

# Per-generator config overrides for the loop below. The host-driven repulsive optimizer is
# O(E*N^2) per iteration and ~1000x slower than the point-family generators at the default
# config, so it runs here on a reduced budget (small stages/num_points, fewer envs) — enough
# to assess non-degeneracy cheaply (~1 s vs many minutes). ``num_envs`` selects the batch size;
# the rest are TrackGenConfig kwargs. Generators absent from the table keep the default E=256
# full config, byte-for-byte as before.
_GEN_OVERRIDES = {
    "repulsive": dict(num_envs=32, num_points=64, repulsive_stages=(16, 32, 64), N_max=384),
}


def test_no_registered_generator_is_degenerate():
    wp.init()
    for g in generator_registry.available():
        over = dict(_GEN_OVERRIDES.get(g, {}))
        E = int(over.pop("num_envs", 256))
        cfg = TrackGenConfig(generator=g, device="cpu", num_envs=E, half_width=0.1,
                             relax_iters=40, **over)
        track = TrackGenerator(cfg, PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")).generate(E)
        center = wp.to_torch(track.center).cpu().numpy().reshape(E, -1, 2)
        count = wp.to_torch(track.count).cpu().numpy().astype(int)
        valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
        comp = np.array([
            tm.compactness(center[e, :count[e]])
            for e in range(E)
            if valid[e] and count[e] >= 4 and np.isfinite(center[e, :count[e]]).all()
        ])
        assert comp.size > 0, f"{g!r}: no valid tracks to assess"
        p50 = float(np.median(comp))
        assert p50 < _MEDIAN_COMPACTNESS_MAX, (
            f"{g!r} is degenerate / near-circular: median compactness {p50:.3f} "
            f">= {_MEDIAN_COMPACTNESS_MAX} (1.0 == circle — the polar-0.97 signature)"
        )
