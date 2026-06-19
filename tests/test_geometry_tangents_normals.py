import math

import torch

from tests._oracle.geometry import tangents_normals


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_tangent_is_unit_everywhere():
    pts = _circle(64, 3.0)
    T, Nrm = tangents_normals(pts)
    assert T.shape == pts.shape
    assert Nrm.shape == pts.shape
    norms = torch.linalg.norm(T, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_tangent_and_normal_are_orthogonal_everywhere():
    pts = _circle(64, 3.0)
    T, Nrm = tangents_normals(pts)
    dot = (T * Nrm).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-6)


def test_normal_is_unit_and_is_left_rotation_of_tangent():
    pts = _circle(32, 1.0)
    T, Nrm = tangents_normals(pts)
    nrm_norms = torch.linalg.norm(Nrm, dim=-1)
    assert torch.allclose(nrm_norms, torch.ones_like(nrm_norms), atol=1e-6)
    assert torch.allclose(Nrm[..., 0], -T[..., 1], atol=1e-6)
    assert torch.allclose(Nrm[..., 1], T[..., 0], atol=1e-6)
