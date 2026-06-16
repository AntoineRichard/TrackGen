import torch
import pytest

from track_gen.generators import Centerline, CenterlineGenerator


def test_centerline_holds_tensors():
    E, M_max = 4, 7
    points = torch.zeros((E, M_max, 2))
    valid = torch.ones((E,), dtype=torch.bool)
    cl = Centerline(points=points, valid=valid)
    assert cl.points.shape == (E, M_max, 2)
    assert cl.valid.shape == (E,)
    assert cl.valid.dtype == torch.bool


def test_fake_generator_satisfies_protocol():
    class FakeGen(CenterlineGenerator):
        def generate(self, ids):
            E = len(ids)
            return Centerline(
                points=torch.zeros((E, 5, 2)),
                valid=torch.ones((E,), dtype=torch.bool),
            )

    gen = FakeGen()
    assert isinstance(gen, CenterlineGenerator)
    ids = torch.arange(3)
    out = gen.generate(ids)
    assert isinstance(out, Centerline)
    assert out.points.shape == (3, 5, 2)
    assert out.valid.tolist() == [True, True, True]


def test_abstract_generator_cannot_instantiate():
    with pytest.raises(TypeError):
        CenterlineGenerator()
