"""Verify the pure-Warp ``assemble`` (vertex tangents + cubic Bezier) wrapper
matches the torch oracle ``BezierCenterlineGenerator._assemble_centerline``.

Builds FIXED corner sets (no RNG) so the comparison is deterministic, with a
``count`` vector that is full (== P) for some envs and short (P - 3) for others,
exercising the NaN-pruning path. Runs on the Warp ``cpu`` device and on ``cuda``
when a GPU is present.
"""
import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen._src import warp_pipeline as wpl
from tests._oracle.generators import BezierCenterlineGenerator
from track_gen._src.types import TrackGenConfig

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fixed_corners(P: int, device) -> torch.Tensor:
    """Three envs of distinct jittered-polygon vertices, shape [3, P, 2]."""
    envs = []
    for e in range(3):
        ang = torch.arange(P, dtype=torch.float32) * (2.0 * math.pi / P)
        # Distinct radii/phase per env so the three loops differ.
        r = 1.0 + 0.1 * e
        phase = 0.37 * (e + 1)
        x = r * torch.cos(ang + phase) + 0.05 * torch.cos(3.0 * ang)
        y = r * torch.sin(ang + phase) + 0.05 * torch.sin(2.0 * ang)
        envs.append(torch.stack([x, y], dim=1))
    return torch.stack(envs, dim=0).to(device)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("edgy", [0.0, 1.0])  # 0.0 -> p=0.5 bisector; 1.0 -> asymmetric blend
def test_assemble_matches_oracle(dev, edgy):
    config = TrackGenConfig()
    config.edgy = edgy  # exercise both symmetric (p=0.5) and asymmetric tangent blends
    config.device = dev  # oracle precomputes its Bernstein basis on config.device
    P = config.max_num_points
    npseg = config.num_points_per_segment

    corners = _fixed_corners(P, dev)            # [3, P, 2]
    E = corners.shape[0]
    # Full count for env 0, short (P-3) for envs 1 and 2 -> exercises NaN pruning.
    count = torch.tensor([P, P - 3, P - 3], dtype=torch.long, device=corners.device)

    # Oracle: prune the same way the wrapper folds in (rows >= count -> NaN),
    # then assemble. rng is unused by _assemble_centerline.
    gen = BezierCenterlineGenerator(config, rng=None)
    row = torch.arange(P, device=corners.device)
    keep = (row < count[:, None]).unsqueeze(-1)               # [E, P, 1]
    pruned = torch.where(keep, corners, torch.full_like(corners, float("nan")))
    ref = gen._assemble_centerline(pruned)

    got = wpl.assemble(corners, count, config)

    assert got.shape == (E, P * npseg, 2)
    # NaN must land in EXACTLY the same positions as the oracle.
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    assert torch.allclose(got, ref, atol=1e-4, equal_nan=True)
