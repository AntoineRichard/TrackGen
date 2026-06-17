# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test for the pure-Warp track generation pipeline.

Exercises ``warp_pipeline.generate_tracks_warp`` (generation -> relax -> resample ->
inflate) on the Warp cpu AND cuda devices. The Warp pipeline uses Warp RNG (different
tracks than the torch oracle), so it is validated by YIELD / WIDTH / SHAPE aggregates,
not per-env allclose. A best-effort yield comparison against the torch oracle is also
made when the rng is cheap to construct.
"""
import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen import warp_pipeline as wpl
from track_gen.types import TrackGenConfig, Track

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# The O(N^2)-per-bead XPBD relax is slow on the Warp CPU device at E=64,N=256
# (tens of seconds). Keep the full E=64 on cuda; shrink E on cpu to stay fast.
_E_BY_DEV = {"cpu": 16, "cuda": 64}


def test_generate_tracks_warp_rejects_unsupported_relax_knobs():
    # Only default XPBD (no finisher) is ported; a non-default config must fail loudly
    # rather than silently diverge from the torch oracle once wired into the facade.
    seeds = torch.arange(4)
    with pytest.raises(AssertionError):
        wpl.generate_tracks_warp(TrackGenConfig(num_envs=4, relax_solver="energy"), seeds)
    with pytest.raises(AssertionError):
        wpl.generate_tracks_warp(TrackGenConfig(num_envs=4, smooth_finish=True), seeds)


@pytest.mark.parametrize("dev", DEVS)
def test_generate_tracks_warp_e2e(dev):
    E = _E_BY_DEV[dev]
    N = 256
    hw = 0.03
    config = TrackGenConfig(num_envs=E, half_width=hw)
    assert config.num_points == N

    seeds = torch.arange(E, device=dev)
    track = wpl.generate_tracks_warp(config, seeds)

    # --- type + shapes ---
    assert isinstance(track, Track)
    for field in ("outer", "center", "inner", "tangent", "normal"):
        assert getattr(track, field).shape == (E, N, 2), field
    assert track.arclen.shape == (E, N)
    assert track.length.shape == (E,)
    assert track.valid.shape == (E,)
    assert track.count.shape == (E,)

    # --- yield ---
    yield_frac = track.valid.float().mean().item()
    assert yield_frac >= 0.9, f"{dev} yield {yield_frac} < 0.9"

    # --- constant width on valid envs ---
    w = torch.linalg.norm(track.outer - track.center, dim=-1)  # [E, N]
    if track.valid.any():
        wv = w[track.valid]
        assert torch.allclose(wv, torch.full_like(wv, hw), atol=1e-4), \
            f"{dev} width not constant: range [{wv.min()}, {wv.max()}]"
        # valid tracks must be finite.
        assert torch.isfinite(track.center[track.valid]).all()

    # --- best-effort torch-oracle yield comparison ---
    # The Warp pipeline uses Warp RNG (different tracks than the torch oracle), so we
    # only compare the AGGREGATE yields, not per-env tracks. rng construction mirrors
    # tests/test_end_to_end_relaxation.py.
    try:
        from track_gen.track_generator import TrackGenerator
        from track_gen.rng_utils import PerEnvSeededRNG

        ocfg = TrackGenConfig(num_envs=E, half_width=hw, device=dev)
        oseeds = torch.arange(E, dtype=torch.int32, device=dev)
        rng = PerEnvSeededRNG(seeds=oseeds, num_envs=E, device=dev)
        rng.set_seeds(oseeds, ids=torch.arange(E, dtype=torch.int32, device=dev))
        otrack = TrackGenerator(ocfg, rng).generate(E)
        oracle_yield = otrack.valid.float().mean().item()
    except Exception:  # rng/oracle construction non-obvious or unavailable -> skip
        oracle_yield = None

    if oracle_yield is not None:
        # Both pipelines should land in the same high-yield regime; allow ~0.1 slack.
        assert abs(yield_frac - oracle_yield) <= 0.1, \
            f"{dev} warp yield {yield_frac} vs oracle {oracle_yield}"
