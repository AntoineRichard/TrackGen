# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA-only: end-to-end CUDA graph capture of the pure-Warp track pipeline.

``generate_tracks_warp_graph`` captures the ENTIRE ``generate_tracks_warp`` pipeline
(generation -> band/L0 torch glue -> XPBD relax -> resample -> inflate) as ONE CUDA
graph and replays it with new seeds copied into a static buffer. The whole pipeline --
torch ops AND every Warp kernel launch -- is unified onto torch's internal capture
stream, so a single ``torch.cuda.graph`` records all of it. This test proves
replay(new_seeds) == the eager ``generate_tracks_warp(config, new_seeds)`` (positions
allclose to atol 1e-4; valid/count via ``torch.equal``).

Capture needs a real GPU, so the whole module is skipped without CUDA.
"""
from __future__ import annotations

import pytest
import torch

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")

from track_gen.types import TrackGenConfig  # noqa: E402
from track_gen import warp_pipeline as wpp  # noqa: E402


def _cfg(E: int) -> TrackGenConfig:
    # Modest iters keep the test fast; the capture mechanism is independent of count.
    # constant_spacing is the only supported mode; fix spacing + N_max so the captured
    # graph's static N_max buffer is deterministic (and large enough for every env).
    return TrackGenConfig(
        num_envs=E, device="cuda:0", output_mode="constant_spacing",
        spacing=0.6, N_max=256,
        relax_solver="xpbd", smooth_finish=False,
        relax_iters=20, max_regen_iters=4,
    )


def _seeds(E: int, base: int) -> torch.Tensor:
    g = torch.Generator(device="cuda:0").manual_seed(base)
    return torch.randint(0, 2**31 - 1, (E,), device="cuda:0", dtype=torch.int64, generator=g)


def _track_allclose(got, ref, atol=1e-4):
    # NaN rows (invalid envs) appear in both; compare with NaNs zeroed so positions
    # agree where finite and the NaN pattern is identical.
    assert torch.equal(got.valid, ref.valid), "valid mask differs"
    assert torch.equal(got.count, ref.count), "count differs"
    assert torch.equal(torch.isnan(got.center), torch.isnan(ref.center)), "NaN pattern differs"
    for name in ("center", "outer", "inner", "tangent", "normal", "arclen"):
        a = torch.nan_to_num(getattr(got, name))
        b = torch.nan_to_num(getattr(ref, name))
        assert torch.allclose(a, b, atol=atol), \
            f"{name} mismatch, max err {(a - b).abs().max().item():.3e}"
    la = torch.nan_to_num(got.length)
    lb = torch.nan_to_num(ref.length)
    assert torch.allclose(la, lb, atol=1e-3), \
        f"length mismatch, max err {(la - lb).abs().max().item():.3e}"


def test_graph_replay_matches_eager_new_seeds():
    """Capture once, replay with TWO different seed sets, each == the eager pipeline."""
    E = 64
    cfg = _cfg(E)
    seeds_a = _seeds(E, 0)
    seeds_b = _seeds(E, 12345)

    ref_a = wpp.generate_tracks_warp(cfg, seeds_a)
    ref_b = wpp.generate_tracks_warp(cfg, seeds_b)
    torch.cuda.synchronize()

    # Capture with seeds_a as the template (warmup uses the template's contents).
    captured = wpp.generate_tracks_warp_graph(cfg, seeds_a)

    # Replay with the SAME seeds first (sanity: graph re-runs deterministically).
    out_a = captured.replay(seeds_a)
    _track_allclose(out_a, ref_a)

    # Replay with NEW seeds: proves the graph actually re-executes the buffer contents.
    out_b = captured.replay(seeds_b)
    _track_allclose(out_b, ref_b)

    # And back to A (replay is reusable / not one-shot).
    out_a2 = captured.replay(seeds_a)
    _track_allclose(out_a2, ref_a)


def test_eager_path_still_syncs_unchanged():
    """The public eager API is unaffected by the capture machinery (no _CAPTURING leak)."""
    assert wpp._CAPTURING is False
    E = 16
    cfg = _cfg(E)
    t = wpp.generate_tracks_warp(cfg, _seeds(E, 7))
    torch.cuda.synchronize()
    # constant_spacing output is NaN-padded to [E, N_max, 2] (N_max, NOT num_points).
    assert t.center.shape == (E, cfg.N_max, 2)
    # Flag must be restored to False after a (separate) capture, not left set.
    assert wpp._CAPTURING is False
