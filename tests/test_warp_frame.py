import math
import pytest
import torch

pytest.importorskip("warp")
import warp as wp
from track_gen._src import warp_pipeline as wpl
from tests._oracle import geometry

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _loop(n=256, r=2.0, dev="cpu"):
    t = torch.linspace(0, 2 * math.pi, n + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0)


def _call_frame(c, count=None):
    """Allocate out/scratch buffers and call the in-place frame_curvature wrapper."""
    E, n_max, _ = c.shape
    flat = E * n_max
    dev = str(c.device)
    cf = wp.from_torch(c.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    if count is None:
        count_t = torch.full((E,), n_max, dtype=torch.int32, device=c.device)
    else:
        count_t = count.to(torch.int32).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_T = wp.zeros(flat, dtype=wp.vec2f, device=dev)
    out_Nrm = wp.zeros(flat, dtype=wp.vec2f, device=dev)
    kappa = wp.zeros(flat, dtype=wp.float32, device=dev)
    wpl.frame_curvature(cf, out_T, out_Nrm, kappa, cnt_wp)
    T = wp.to_torch(out_T).view(E, n_max, 2)
    Nrm = wp.to_torch(out_Nrm).view(E, n_max, 2)
    kap = wp.to_torch(kappa).view(E, n_max)
    return T, Nrm, kap


@pytest.mark.parametrize("dev", DEVS)
def test_frame_circle_properties(dev):
    c = _loop(256, 2.0, dev)
    T, Nrm, kap = _call_frame(c)
    assert torch.allclose(torch.linalg.norm(T, dim=-1), torch.ones(1, 256, device=dev), atol=1e-4)
    assert torch.allclose((T * Nrm).sum(-1), torch.zeros(1, 256, device=dev), atol=1e-5)
    assert torch.allclose(kap, torch.full((1, 256), 0.5, device=dev), atol=1e-2)  # 1/r


@pytest.mark.parametrize("dev", DEVS)
def test_frame_matches_torch_oracle(dev):
    torch.manual_seed(0)
    c = (torch.randn(5, 256, 2, device=dev) * torch.linspace(0.5, 3.0, 5, device=dev).view(5, 1, 1))
    T, Nrm, kap = _call_frame(c)
    T_ref, Nrm_ref = geometry.tangents_normals(c)
    kap_ref = geometry.menger_curvature(c)
    assert torch.allclose(T, T_ref, atol=1e-5)
    assert torch.allclose(Nrm, Nrm_ref, atol=1e-5)
    assert torch.allclose(kap, kap_ref, atol=1e-4)
