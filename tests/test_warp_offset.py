import math
import pytest
import torch

pytest.importorskip("warp")
from track_gen._src import warp_pipeline as wpl
from tests._oracle import geometry, inflation

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _loop(n=256, r=2.0, dev="cpu"):
    t = torch.linspace(0, 2 * math.pi, n + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], -1).unsqueeze(0)


@pytest.mark.parametrize("dev", DEVS)
def test_offset_matches_torch(dev):
    torch.manual_seed(0)
    c = torch.randn(5, 256, 2, device=dev) * torch.linspace(0.5, 3.0, 5, device=dev).view(5, 1, 1)
    _, Nrm = geometry.tangents_normals(c)
    hw = 0.1
    outer, inner = wpl.offset(c, Nrm, hw)
    w = torch.full((5, 256), hw, device=dev)
    o_ref, i_ref = inflation._offset_stage(c, Nrm, w)
    assert torch.allclose(outer, o_ref, atol=1e-5)
    assert torch.allclose(inner, i_ref, atol=1e-5)


@pytest.mark.parametrize("dev", DEVS)
def test_offset_outer_bigger_on_circle(dev):
    c = _loop(256, 3.0, dev)
    _, Nrm = geometry.tangents_normals(c)
    outer, inner = wpl.offset(c, Nrm, 0.5)
    assert geometry.polygon_area(outer).abs()[0] > geometry.polygon_area(inner).abs()[0]
