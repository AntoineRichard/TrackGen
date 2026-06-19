import math

import pytest
import torch

pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from tests._oracle import geometry, inflation
from track_gen.types import TrackGenConfig

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

N = 256


def _circle(n=N, r=2.0, dev="cpu"):
    t = torch.linspace(0, 2 * math.pi, n + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0)


def _fig8(n=N, s=1.0, dev="cpu"):
    t = torch.linspace(0, 2 * math.pi, n + 1, device=dev)[:-1]
    return torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1).unsqueeze(0)


def _thin_ellipse(n=N, dev="cpu"):
    t = torch.linspace(0, 2 * math.pi, n + 1, device=dev)[:-1]
    return torch.stack([2.0 * torch.cos(t), 0.04 * torch.sin(t)], dim=-1).unsqueeze(0)


def _build_centers(dev):
    # circle -> VALID, figure-eight -> INVALID (turning ~0 + crossings),
    # thin/folded ellipse -> INVALID (thickness too small).
    return torch.cat([_circle(N, 2.0, dev), _fig8(N, 1.0, dev), _thin_ellipse(N, dev)], dim=0)


@pytest.mark.parametrize("dev", DEVS)
def test_validity_matches_oracle(dev):
    center = _build_centers(dev)
    E = center.shape[0]
    config = TrackGenConfig(half_width=0.1, num_points=N)

    # Build oracle inputs from inflation's own stages (no use of our code here).
    T, Nrm, kappa = inflation._frame_curvature_stage(center)
    w = inflation._width_stage(center, kappa, config)
    outer, inner = inflation._offset_stage(center, Nrm, w)
    count = torch.full((E,), N, dtype=torch.long, device=dev)
    gen_valid = torch.ones(E, dtype=torch.bool, device=dev)

    ref = inflation._validity_stage(center, w, count, gen_valid, config, outer, inner)
    got = wpl.validity(center, w, count, gen_valid, config, outer, inner)

    assert torch.equal(got.cpu(), ref.cpu())
    assert got.cpu().tolist() == [True, False, False]


@pytest.mark.parametrize("dev", DEVS)
def test_validity_no_border_matches_oracle(dev):
    # With outer/inner omitted both paths skip the border check (border_ok all True),
    # so validity reduces to gen/turning/width/no-nan/thickness. Matches the oracle's
    # outer=None/inner=None defaults.
    center = _build_centers(dev)
    E = center.shape[0]
    config = TrackGenConfig(half_width=0.1, num_points=N)
    count = torch.full((E,), N, dtype=torch.long, device=dev)
    gen_valid = torch.ones(E, dtype=torch.bool, device=dev)
    w = torch.full((E, N), float(config.half_width), device=dev)

    ref = inflation._validity_stage(center, w, count, gen_valid, config)
    got = wpl.validity(center, w, count, gen_valid, config)
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.mark.parametrize("dev", DEVS)
def test_turning_number_matches_oracle(dev):
    c = _circle(N, 2.0, dev)
    assert torch.allclose(wpl.turning_number(c).cpu(),
                          geometry.turning_number(c).cpu(), atol=1e-4)
    # circle turning number is approximately +/- 2*pi.
    assert abs(abs(wpl.turning_number(c).item()) - 2.0 * math.pi) <= 1e-3

    torch.manual_seed(0)
    rand = torch.randn(4, N, 2, device=dev) * 0.8
    assert torch.allclose(wpl.turning_number(rand).cpu(),
                          geometry.turning_number(rand).cpu(), atol=1e-4)
