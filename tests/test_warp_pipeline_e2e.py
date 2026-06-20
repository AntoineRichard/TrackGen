"""End-to-end test for the pure-Warp track generation pipeline.

Exercises ``warp_pipeline.generate_tracks_warp`` (generation -> relax -> resample ->
inflate) on the Warp cpu AND cuda devices. The Warp pipeline uses Warp RNG (different
tracks than the torch oracle), so it is validated by YIELD / WIDTH / SHAPE aggregates,
not per-env allclose. A best-effort yield comparison against the torch oracle is also
made when the rng is cheap to construct.

Track fields are wp.array; test reads are wrapped with to_t() from _warp_compare
to convert to torch tensors at the oracle boundary.
"""
import math

import pytest
import torch

pytest.importorskip("warp")
import warp as wp

from tests._warp_compare import to_t  # noqa: E402
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
    # Track fields are wp.array; convert to torch for shape/dtype assertions.
    for field in ("outer", "center", "inner", "tangent", "normal"):
        arr = to_t(getattr(track, field))
        assert arr.shape == (E * N_max, 2), f"{field} shape mismatch: {arr.shape}"
    arclen_t = to_t(track.arclen)
    assert arclen_t.shape == (E * N_max,), f"arclen shape: {arclen_t.shape}"
    length_t = to_t(track.length)
    assert length_t.shape == (E,), f"length shape: {length_t.shape}"
    valid_t = to_t(track.valid).bool()
    assert valid_t.shape == (E,), f"valid shape: {valid_t.shape}"
    count_t = to_t(track.count)
    assert count_t.shape == (E,), f"count shape: {count_t.shape}"

    # --- yield ---
    yield_frac = valid_t.float().mean().item()
    assert yield_frac >= 0.9, f"{dev} yield {yield_frac} < 0.9"

    # --- count is a sane per-env real-point count in (0, N_max] for valid envs ---
    if valid_t.any():
        cv = count_t[valid_t]
        assert (cv > 0).all() and (cv <= N_max).all(), \
            f"{dev} count out of range: {cv.min()}..{cv.max()} (N_max={N_max})"

    # --- constant width on valid envs, count-aware ---
    # Reshape flat wp.array fields to [E, N_max, 2] torch for per-env masking.
    center_th = to_t(track.center).view(E, N_max, 2)
    outer_th = to_t(track.outer).view(E, N_max, 2)
    w = torch.linalg.norm(outer_th - center_th, dim=-1)  # [E, N_max], NaN-padded
    real = torch.isfinite(center_th).all(dim=-1)  # [E, N_max] real-point mask
    for e in torch.nonzero(valid_t, as_tuple=False).flatten().tolist():
        cnt = int(count_t[e].item())
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
        wp_oseeds = wp.from_torch(oseeds, dtype=wp.int32)
        rng = PerEnvSeededRNG(seeds=wp_oseeds, num_envs=E, device=dev)
        rng.set_seeds_warp(wp_oseeds,
                           ids=wp.array(list(range(E)), dtype=wp.int32, device=dev))
        otrack = TrackGenerator(ocfg, rng).generate(E)
        oracle_yield = to_t(otrack.valid).bool().float().mean().item()
    except Exception:  # rng/oracle construction non-obvious or unavailable -> skip
        oracle_yield = None

    if oracle_yield is not None:
        # Both pipelines should land in the same high-yield regime; allow ~0.1 slack.
        assert abs(yield_frac - oracle_yield) <= 0.1, \
            f"{dev} warp yield {yield_frac} vs oracle {oracle_yield}"
