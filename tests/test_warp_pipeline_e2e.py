"""End-to-end test for the pure-Warp track generation pipeline.

Exercises the pipeline (generation -> relax -> resample -> inflate) via
``TrackGenerator`` on the Warp cpu AND cuda devices. The Warp pipeline uses Warp
RNG (different tracks than the oracle), so it is validated by YIELD / WIDTH / SHAPE
aggregates, not per-env allclose.

Track fields are wp.array; test reads are wrapped with to_t() from _warp_compare
to convert to tensors at the oracle boundary.
"""
import math

import pytest
import torch

pytest.importorskip("warp")
import warp as wp

from tests._warp_compare import to_t  # noqa: E402
from track_gen._src.types import TrackGenConfig, Track
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# The O(N^2)-per-bead XPBD relax is slow on the Warp CPU device at E=64,N=256
# (tens of seconds). Keep the full E=64 on cuda; shrink E on cpu to stay fast.
_E_BY_DEV = {"cpu": 16, "cuda": 64}


@pytest.mark.parametrize("dev", DEVS)
def test_generate_tracks_e2e(dev):
    E = _E_BY_DEV[dev]
    # constant_spacing output: arrays are [E*N_max] flat NaN-padded with a per-env real
    # point count in track.count. N_max is the padded width; spacing is auto 0.6*half_width.
    # With per-env seed diversity the half_width=0.03 regime spans counts ~112..468, so
    # N_max=512 sits above the max -> no truncation and the NaN-padding tail is exercised.
    N_max = 512
    hw = 0.03
    config = TrackGenConfig(num_envs=E, half_width=hw, N_max=N_max, device=dev)

    rng = PerEnvSeededRNG(seeds=42, num_envs=E, device=dev)
    gen = TrackGenerator(config, rng)
    track = gen.generate(E)

    # --- type + shapes (all arrays padded to N_max; count varies per env) ---
    assert isinstance(track, Track)
    for field in ("outer", "center", "inner", "tangent", "normal"):
        arr = to_t(getattr(track, field))
        assert arr.shape == (E * N_max, 3), f"{field} shape mismatch: {arr.shape}"
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
    center_th = to_t(track.center).view(E, N_max, 3)[..., :2]
    outer_th = to_t(track.outer).view(E, N_max, 3)[..., :2]
    w = torch.linalg.norm(outer_th - center_th, dim=-1)  # [E, N_max], NaN-padded
    real = torch.isfinite(center_th).all(dim=-1)  # [E, N_max] real-point mask
    for e in torch.nonzero(valid_t, as_tuple=False).flatten().tolist():
        cnt = int(count_t[e].item())
        assert bool(real[e, :cnt].all()), f"{dev} env {e}: real points must be finite"
        assert not bool(real[e, cnt:].any()), \
            f"{dev} env {e}: padding past count[e]={cnt} must be NaN"
        wv = w[e, :cnt]
        assert torch.allclose(wv, torch.full_like(wv, hw), atol=1e-4), \
            f"{dev} env {e} width not constant: range [{wv.min()}, {wv.max()}]"
