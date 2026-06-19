import math

import pytest
import torch

pytest.importorskip("warp")
from track_gen._src import warp_pipeline as wpl
from tests._oracle import geometry

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_batch(dev):
    """Fixed [E, M, 2] batch with varied real counts to exercise NaN compaction.

    env 0: a full clean jittered-circle loop (M real points).
    env 1: a clean loop with a NaN-padded TAIL (R = M-7 real, rest NaN).
    env 2: a clean loop with INTERIOR NaN points (proves in-order compaction).
    env 3: ALL NaN (R = 0 -> count 0, all-NaN output row).
    env 4: exactly 1 real point (R = 1 -> count 0, all-NaN output row).
    """
    M = 40
    E = 5
    torch.manual_seed(7)
    t = torch.linspace(0, 2 * math.pi, M + 1, device=dev)[:-1]
    base = torch.stack([2.0 * torch.cos(t), 1.5 * torch.sin(t)], dim=-1)  # [M, 2]
    jitter = 0.05 * torch.randn(E, M, 2, device=dev)
    pts = base.unsqueeze(0).repeat(E, 1, 1) + jitter

    nan = float("nan")
    # env 1: NaN tail (last 7 points).
    pts[1, M - 7:, :] = nan
    # env 2: interior NaN at scattered indices.
    pts[2, [3, 4, 17, 28, 31], :] = nan
    # env 3: all NaN.
    pts[3, :, :] = nan
    # env 4: exactly one real point (index 0), rest NaN.
    pts[4, 1:, :] = nan
    return pts


@pytest.mark.parametrize("dev", DEVS)
def test_arc_resample_matches_oracle(dev):
    num = 64
    pts = _make_batch(dev)

    ref_r, ref_c = geometry.arc_length_resample(pts, num=num)
    got_r, got_c = wpl.arc_length_resample_warp(pts, num)

    # Count EXACT: 64 for R>=2 envs (0, 1, 2), 0 for R<2 envs (3, 4).
    assert torch.equal(got_c.cpu(), ref_c.cpu()), (got_c.tolist(), ref_c.tolist())
    assert ref_c.cpu().tolist() == [num, num, num, 0, 0]

    # NaN masks match exactly (R<2 envs all-NaN, others fully finite).
    assert torch.equal(torch.isnan(got_r), torch.isnan(ref_r))

    # Values match within float32-accumulation drift. The Warp scan accumulates the
    # closed-loop arc length in float64 (vs torch.cumsum), so the residual is FP sqrt
    # rounding (~1e-4 on scale-2 coords), not algorithmic; 5e-4 covers it cpu+cuda.
    assert torch.allclose(got_r, ref_r, atol=5e-4, equal_nan=True), \
        (got_r - ref_r)[~torch.isnan(ref_r)].abs().max().item()


@pytest.mark.parametrize("dev", DEVS)
def test_arc_resample_clean_loop_uniform(dev):
    # A single clean loop resampled to a different count must be ~uniformly spaced.
    M = 50
    num = 137
    t = torch.linspace(0, 2 * math.pi, M + 1, device=dev)[:-1]
    c = torch.stack([2.0 * torch.cos(t), 2.0 * torch.sin(t)], dim=-1).unsqueeze(0)
    out, count = wpl.arc_length_resample_warp(c, num)
    assert int(count.item()) == num
    assert out.shape == (1, num, 2)
    seg = torch.linalg.norm(torch.diff(out, dim=1, append=out[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-2
