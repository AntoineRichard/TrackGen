import math
import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen._src import warp_pipeline as wpl
from tests._oracle.relaxation import _resample_uniform

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _call_resample_uniform(center: torch.Tensor, n: int,
                            count: torch.Tensor | None = None) -> torch.Tensor:
    """Test helper: allocate wp.array buffers, call in-place resample_uniform, return torch.

    This wraps the new strict-in-place API for use in standalone tests (buffers allocated
    per call here, which is fine for test-side use; the no-alloc rule only applies to the
    runtime generate() path).
    """
    E, n_max, _ = center.shape
    assert n == n_max
    dev = str(center.device)
    flat = E * n_max

    if count is None:
        count_t = torch.full((E,), n_max, dtype=torch.int32, device=center.device)
    else:
        count_t = count.to(torch.int32).contiguous()

    center_wp = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    out_wp = wp.empty(flat, dtype=wp.vec2f, device=dev)
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)

    wpl.resample_uniform(center_wp, out_wp, n, cnt_wp, device=dev)
    wp.synchronize() if "cuda" in dev else None
    return wp.to_torch(out_wp).view(E, n, 2)


@pytest.mark.parametrize("dev", DEVS)
def test_resample_matches_torch(dev):
    torch.manual_seed(0)
    c = (torch.randn(8, 200, 2, device=dev)
         * torch.linspace(0.3, 2.0, 8, device=dev).view(8, 1, 1))
    got = _call_resample_uniform(c, 200)
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
    out = _call_resample_uniform(c, 300)
    seg = torch.linalg.norm(torch.diff(out, dim=1, append=out[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-2
