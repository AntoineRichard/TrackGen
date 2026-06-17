import pytest
import torch

pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl


def test_warp_runs_on_cpu_device():
    # The pure-Warp pipeline must run on the Warp CPU device so CI works GPU-free.
    out = wpl._smoke_double(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warp_runs_on_cuda_device():
    out = wpl._smoke_double(torch.tensor([1.0, 2.0, 3.0], device="cuda"))
    assert torch.allclose(out.cpu(), torch.tensor([2.0, 4.0, 6.0]))
