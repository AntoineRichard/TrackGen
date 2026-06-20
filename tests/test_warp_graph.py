"""CUDA-only: automatic CUDA graph capture in TrackGenerator.

``TrackGenerator.generate()`` on a CUDA device captures the pure-Warp pipeline
(generation -> resample -> relax -> inflate) into a ``wp.Graph`` on the first call via
``wp.ScopedCapture``, then replays it on every subsequent call. Seeds are written into a
pre-allocated seed buffer in place before each replay.

This test constructs a ``TrackGenerator`` on ``cuda:0``, calls ``generate()`` twice (the
first captures, the second replays), and asserts:
  - Replayed Track == eager (positions allclose ~1e-4; valid/count exact; stable .ptr).
  - Buffers are reused: the same Track instance is returned on every call.

Capture needs a real GPU, so the whole module is skipped without CUDA.

Track fields are wp.array; test reads are wrapped with to_t() from _warp_compare.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")

import warp as wp  # noqa: E402
from tests._warp_compare import to_t  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src import warp_pipeline as wpp  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402


def _cfg(E: int) -> TrackGenConfig:
    # Modest iters keep the test fast; the capture mechanism is independent of count.
    # constant_spacing is the only supported mode; N_max large enough for every env.
    return TrackGenConfig(
        num_envs=E, device="cuda:0", output_mode="constant_spacing",
        spacing=0.6, N_max=256,
        relax_solver="xpbd", smooth_finish=False,
        relax_iters=20, max_regen_iters=4,
    )


def _make_rng(E: int, seed: int) -> PerEnvSeededRNG:
    return PerEnvSeededRNG(seeds=seed, num_envs=E, device="cuda:0")


def _track_allclose(got, ref, atol=1e-4):
    """Assert got == ref field-by-field (Track with wp.array fields)."""
    E_Nmax = to_t(got.valid).shape[0]   # E
    center_flat_size = to_t(got.center).shape[0]  # E*N_max
    N_max = center_flat_size // E_Nmax

    assert torch.equal(to_t(got.valid), to_t(ref.valid)), "valid mask differs"
    assert torch.equal(to_t(got.count), to_t(ref.count)), "count differs"
    got_center = wp.to_torch(got.center).view(E_Nmax, N_max, 2)
    ref_center = wp.to_torch(ref.center).view(E_Nmax, N_max, 2)
    assert torch.equal(torch.isnan(got_center), torch.isnan(ref_center)), "NaN pattern differs"
    for name in ("center", "outer", "inner", "tangent", "normal", "arclen"):
        if name == "arclen":
            a = torch.nan_to_num(wp.to_torch(getattr(got, name)).view(E_Nmax, N_max))
            b = torch.nan_to_num(wp.to_torch(getattr(ref, name)).view(E_Nmax, N_max))
        else:
            a = torch.nan_to_num(wp.to_torch(getattr(got, name)).view(E_Nmax, N_max, 2))
            b = torch.nan_to_num(wp.to_torch(getattr(ref, name)).view(E_Nmax, N_max, 2))
        assert torch.allclose(a, b, atol=atol), \
            f"{name} mismatch, max err {(a - b).abs().max().item():.3e}"
    la = torch.nan_to_num(to_t(got.length))
    lb = torch.nan_to_num(to_t(ref.length))
    assert torch.allclose(la, lb, atol=1e-3), \
        f"length mismatch, max err {(la - lb).abs().max().item():.3e}"


def test_autocapture_replay_matches_eager():
    """First generate() captures; second generate() replays; result == eager."""
    E = 64
    cfg = _cfg(E)

    # Build an eager reference with the free-func (same seeds as the generator will use).
    rng = _make_rng(E, seed=42)
    seeds_wp = rng.seeds_warp                          # [E] int32 wp.array on cuda:0
    seeds_t = wp.to_torch(seeds_wp).to(torch.int64)   # to_torch returns int32; cast for compat

    ref = wpp.generate_tracks_warp(cfg, seeds_t)
    torch.cuda.synchronize()

    # Construct TrackGenerator with the SAME rng (same seeds).
    gen = TrackGenerator(cfg, rng)

    # First call: captures the graph (internally), replays, returns self._track.
    track_a = gen.generate(E)
    torch.cuda.synchronize()

    # track_a should equal the eager reference (same seeds).
    _track_allclose(track_a, ref)

    # Second call: replays (graph already captured). The rng seeds haven't changed so
    # the result must be bit-identical to the first call.
    ptr_before = track_a.center.ptr
    track_b = gen.generate(E)
    torch.cuda.synchronize()

    # Stable pointers: same Track instance returned.
    assert track_b is track_a, "generate() must return the same Track instance"
    assert track_b.center.ptr == ptr_before, "center.ptr changed between calls"

    # Replay matches reference.
    _track_allclose(track_b, ref)


def test_eager_path_unaffected():
    """The public eager API (free func) is unaffected by capture machinery."""
    assert wpp._CAPTURING is False
    E = 16
    cfg = _cfg(E)
    rng = _make_rng(E, seed=7)
    seeds_t = wp.to_torch(rng.seeds_warp).to(torch.int64)
    t = wpp.generate_tracks_warp(cfg, seeds_t)
    torch.cuda.synchronize()
    center_t = wp.to_torch(t.center)
    assert center_t.shape[0] == E * cfg.N_max, \
        f"center flat size mismatch: {center_t.shape[0]} != {E * cfg.N_max}"
    assert wpp._CAPTURING is False
