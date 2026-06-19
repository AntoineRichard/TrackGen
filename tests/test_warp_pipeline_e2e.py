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

from track_gen._src import warp_pipeline as wpl
from track_gen._src.types import TrackGenConfig, Track

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
    # constant_spacing output: arrays are [E, N_max, 2] NaN-padded with a per-env real
    # point count in track.count (real points live in center[e, :count[e]]). N_max is the
    # padded width, NOT a per-env point count, so it is set explicitly for a deterministic
    # shape; spacing is auto 0.6*half_width.
    N_max = 256
    hw = 0.03
    config = TrackGenConfig(num_envs=E, half_width=hw, N_max=N_max)

    seeds = torch.arange(E, device=dev)
    track = wpl.generate_tracks_warp(config, seeds)

    # --- type + shapes (all arrays padded to N_max; count varies per env) ---
    assert isinstance(track, Track)
    for field in ("outer", "center", "inner", "tangent", "normal"):
        assert getattr(track, field).shape == (E, N_max, 2), field
    assert track.arclen.shape == (E, N_max)
    assert track.length.shape == (E,)
    assert track.valid.shape == (E,)
    assert track.count.shape == (E,)

    # --- yield ---
    yield_frac = track.valid.float().mean().item()
    assert yield_frac >= 0.9, f"{dev} yield {yield_frac} < 0.9"

    # --- count is a sane per-env real-point count in (0, N_max] for valid envs ---
    if track.valid.any():
        cv = track.count[track.valid]
        assert (cv > 0).all() and (cv <= N_max).all(), \
            f"{dev} count out of range: {cv.min()}..{cv.max()} (N_max={N_max})"

    # --- constant width on valid envs, count-aware ---
    # The real points are center[e, :count[e]]; everything from count[e] onward is NaN
    # padding. Mask to the finite real points per env before comparing the width to hw.
    w = torch.linalg.norm(track.outer - track.center, dim=-1)  # [E, N_max], NaN-padded
    real = torch.isfinite(track.center).all(dim=-1)  # [E, N_max] real-point mask
    for e in torch.nonzero(track.valid, as_tuple=False).flatten().tolist():
        cnt = int(track.count[e].item())
        # The finite mask must exactly cover the first count[e] points and nothing else.
        assert bool(real[e, :cnt].all()), f"{dev} env {e}: real points must be finite"
        assert not bool(real[e, cnt:].any()), \
            f"{dev} env {e}: padding past count[e]={cnt} must be NaN"
        wv = w[e, :cnt]
        assert torch.allclose(wv, torch.full_like(wv, hw), atol=1e-4), \
            f"{dev} env {e} width not constant: range [{wv.min()}, {wv.max()}]"

    # --- best-effort torch-oracle yield comparison ---
    # The Warp pipeline uses Warp RNG (different tracks than the torch oracle), so we
    # only compare the AGGREGATE yields, not per-env tracks. rng construction mirrors
    # tests/test_end_to_end_relaxation.py.
    try:
        from track_gen._src.track_generator import TrackGenerator
        from track_gen._src.rng_utils import PerEnvSeededRNG

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
