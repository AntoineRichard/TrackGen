import math

import torch

from tests._oracle.geometry import vertex_tangents


def _regular_polygon(n: int, radius: float = 1.0) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)  # [1, n, 2]


def test_regular_polygon_tangents_are_unit_length():
    pts = _regular_polygon(6)
    t = vertex_tangents(pts, p=0.5)
    assert t.shape == pts.shape
    norms = torch.linalg.norm(t, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_p_half_is_symmetric_between_in_and_out_edges():
    pts = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64
    )
    t = vertex_tangents(pts, p=0.5)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.allclose(
        t[0, 1], torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float64), atol=1e-6
    )


def test_p_extremes_recover_pure_edge_directions():
    pts = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64
    )
    t_out = vertex_tangents(pts, p=1.0)  # pure out-edge at vertex 1 -> +y
    t_in = vertex_tangents(pts, p=0.0)  # pure in-edge at vertex 1 -> +x
    assert torch.allclose(t_out[0, 1], torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(t_in[0, 1], torch.tensor([1.0, 0.0], dtype=torch.float64), atol=1e-6)
