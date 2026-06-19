import math
import pytest
import torch

pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from tests._oracle.relaxation import _resample_uniform

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
def test_resample_matches_torch(dev):
    torch.manual_seed(0)
    c = (torch.randn(8, 200, 2, device=dev)
         * torch.linspace(0.3, 2.0, 8, device=dev).view(8, 1, 1))
    got = wpl.resample_uniform(c, 200)
    ref = _resample_uniform(c, 200)
    # The Warp scan is pure-Warp (float32 sqrt + float64-accumulated arc length); it
    # differs from torch's cumsum only by FP rounding (~1e-4 on scale-2 coords —
    # geometrically negligible), not algorithmically. 5e-4 covers that delta cpu+cuda.
    assert torch.allclose(got, ref, atol=5e-4), (got - ref).abs().max().item()


@pytest.mark.parametrize("dev", DEVS)
def test_resample_circle_uniform(dev):
    t = torch.linspace(0, 2 * math.pi, 300 + 1, device=dev)[:-1]
    c = torch.stack([2 * torch.cos(t), 2 * torch.sin(t)], -1).unsqueeze(0)
    # input N must equal output for this kernel; resample 300->300
    out = wpl.resample_uniform(c, 300)
    seg = torch.linalg.norm(torch.diff(out, dim=1, append=out[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-2
