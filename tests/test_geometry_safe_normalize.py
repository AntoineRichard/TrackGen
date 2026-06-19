import torch

from tests._oracle.geometry import safe_normalize


def test_unit_vectors_have_norm_one():
    v = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [-5.0, 0.0]]])  # [1, 3, 2]
    out = safe_normalize(v)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    assert torch.allclose(out[0, 0], torch.tensor([0.6, 0.8]), atol=1e-6)


def test_zero_vector_stays_finite_and_zero():
    v = torch.zeros((1, 1, 2))
    out = safe_normalize(v)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_shape_is_preserved():
    v = torch.randn((4, 7, 2))
    out = safe_normalize(v)
    assert out.shape == v.shape
