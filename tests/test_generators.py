import math
import types

import torch
import pytest

from track_gen.generators import Centerline, CenterlineGenerator
from track_gen.generators import BezierCenterlineGenerator


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


def _bezier_config(**overrides):
    cfg = types.SimpleNamespace(
        min_num_points=9,
        max_num_points=13,
        num_points_per_segment=30,
        min_point_distance=0.05,
        min_angle=(12.5 / 180) * math.pi,
        rad=0.2,
        edgy=0.0,
        scale=1.0,
        device="cpu",
        num_envs=4,
        max_regen_iters=20,
        turning_tol=0.35,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_bezier_init_sets_derived_params():
    cfg = _bezier_config(min_point_distance=0.05, edgy=0.0)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    # num_cells = int(1 / (2 * 0.05)) = int(10.0) = 10
    assert gen.num_cells == 10
    # p = atan(0)/pi + 0.5 = 0.5
    assert gen.p == pytest.approx(0.5)


def test_bezier_init_basis_shapes():
    cfg = _bezier_config(num_points_per_segment=30)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    for basis in (gen.bernstein_0, gen.bernstein_1, gen.bernstein_2, gen.bernstein_3):
        assert basis.shape == (30,)
        assert basis.dtype == torch.float32
    total = gen.bernstein_0 + gen.bernstein_1 + gen.bernstein_2 + gen.bernstein_3
    assert torch.allclose(total, torch.ones(30), atol=1e-5)


def test_bezier_init_p_increases_with_edgy():
    low = BezierCenterlineGenerator(_bezier_config(edgy=0.0), rng=None).p
    high = BezierCenterlineGenerator(_bezier_config(edgy=5.0), rng=None).p
    assert high > low
