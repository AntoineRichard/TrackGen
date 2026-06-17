import math
import pytest
import torch

pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from track_gen import geometry

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _loop(n=256, r=2.0, dev="cpu"):
    t = torch.linspace(0, 2 * math.pi, n + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0)


@pytest.mark.parametrize("dev", DEVS)
def test_frame_circle_properties(dev):
    c = _loop(256, 2.0, dev)
    T, Nrm, kap = wpl.frame_curvature(c)
    assert torch.allclose(torch.linalg.norm(T, dim=-1), torch.ones(1, 256, device=dev), atol=1e-4)
    assert torch.allclose((T * Nrm).sum(-1), torch.zeros(1, 256, device=dev), atol=1e-5)
    assert torch.allclose(kap, torch.full((1, 256), 0.5, device=dev), atol=1e-2)  # 1/r


@pytest.mark.parametrize("dev", DEVS)
def test_frame_matches_torch_oracle(dev):
    torch.manual_seed(0)
    c = (torch.randn(5, 256, 2, device=dev) * torch.linspace(0.5, 3.0, 5, device=dev).view(5, 1, 1))
    T, Nrm, kap = wpl.frame_curvature(c)
    T_ref, Nrm_ref = geometry.tangents_normals(c)
    kap_ref = geometry.menger_curvature(c)
    assert torch.allclose(T, T_ref, atol=1e-5)
    assert torch.allclose(Nrm, Nrm_ref, atol=1e-5)
    assert torch.allclose(kap, kap_ref, atol=1e-4)
