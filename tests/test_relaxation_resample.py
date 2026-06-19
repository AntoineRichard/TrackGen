import math
import torch
from tests._oracle import relaxation


def _resample_uniform_reference(center, n):
    """Per-env reference (the pre-vectorization implementation) — ground truth."""
    E = center.shape[0]
    closed = torch.cat([center, center[:, :1]], dim=1)
    seg = torch.linalg.norm(closed[:, 1:] - closed[:, :-1], dim=-1)
    s = torch.cat([torch.zeros(E, 1, dtype=center.dtype), torch.cumsum(seg, dim=1)], dim=1)
    total = s[:, -1:]
    targets = torch.arange(n, dtype=center.dtype)[None] * (total / n)
    out = torch.empty_like(center)
    for e in range(E):
        idx = torch.searchsorted(s[e, 1:], targets[e], right=False).clamp(max=seg.shape[1] - 1)
        frac = ((targets[e] - s[e, idx]) / seg[e, idx].clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
        out[e] = closed[e, idx] + frac * (closed[e, idx + 1] - closed[e, idx])
    return out


def test_batched_resample_matches_per_env_reference():
    torch.manual_seed(0)
    # Mixed batch of random closed loops of varying scale.
    E, n = 17, 200
    center = torch.randn(E, n, 2) * torch.linspace(0.2, 3.0, E).view(E, 1, 1)
    got = relaxation._resample_uniform(center, n)
    ref = _resample_uniform_reference(center, n)
    assert got.shape == (E, n, 2)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_batched_resample_circle_is_arc_uniform():
    t = torch.linspace(0, 2 * math.pi, 300 + 1)[:-1]
    circle = torch.stack([2.0 * torch.cos(t), 2.0 * torch.sin(t)], dim=-1).unsqueeze(0)
    out = relaxation._resample_uniform(circle, 128)
    seg = torch.linalg.norm(torch.diff(out, dim=1, append=out[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-3          # uniform spacing
    r = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(r, torch.full_like(r, 2.0), atol=1e-2)  # stays on the circle
