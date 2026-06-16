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


def _make_rng(num_envs, seed=1234, device="cpu"):
    import warp as wp  # noqa: F401  (governed by importorskip in each test)
    wp.init()
    from track_gen.rng_utils import PerEnvSeededRNG

    seeds = torch.arange(num_envs, dtype=torch.int32) + seed
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=device)
    rng.set_seeds(seeds, ids=torch.arange(num_envs, dtype=torch.int32))
    return rng


def test_cell_sampling_shape():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu")
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E))
    ids = torch.arange(E)
    pts = gen._sample_corner_points(ids)
    assert pts.shape == (E, cfg.max_num_points, 2)
    assert torch.isfinite(pts).all()


def test_cell_sampling_reproducible():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    gen_a = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=7))
    gen_b = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=7))
    pts_a = gen_a._sample_corner_points(ids)
    pts_b = gen_b._sample_corner_points(ids)
    assert torch.allclose(pts_a, pts_b)


def test_cell_indices_distinct_per_env():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu")
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E))
    ids = torch.arange(E)
    idxs = gen._sample_cell_indices(ids)  # [E, max_num_points] cell ids
    assert idxs.shape == (E, cfg.max_num_points)
    for e in range(E):
        assert len(torch.unique(idxs[e])) == cfg.max_num_points  # no duplicate cells


def test_cell_sampling_env_independence():
    pytest.importorskip("warp")
    E = 3
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    base = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=100))
    base_pts = base._sample_corner_points(ids)
    import warp as wp  # noqa: F401
    from track_gen.rng_utils import PerEnvSeededRNG

    seeds = torch.tensor([100, 101, 999], dtype=torch.int32)
    rng2 = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
    rng2.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
    gen2 = BezierCenterlineGenerator(cfg, rng=rng2)
    pts2 = gen2._sample_corner_points(ids)
    assert torch.allclose(base_pts[0], pts2[0])
    assert torch.allclose(base_pts[1], pts2[1])
    assert not torch.allclose(base_pts[2], pts2[2])


def test_prune_corners_shape_and_count():
    pytest.importorskip("warp")
    E = 6
    cfg = _bezier_config(num_envs=E, device="cpu", min_num_points=9, max_num_points=13)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E))
    ids = torch.arange(E)
    raw = gen._sample_corner_points(ids)
    pruned, count = gen._prune_corners(raw, ids)
    assert pruned.shape == (E, cfg.max_num_points, 2)
    assert count.shape == (E,)
    assert (count >= cfg.min_num_points).all()
    assert (count <= cfg.max_num_points).all()


def test_prune_corners_pads_with_nan():
    pytest.importorskip("warp")
    E = 6
    cfg = _bezier_config(num_envs=E, device="cpu", min_num_points=4, max_num_points=13)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=42))
    ids = torch.arange(E)
    raw = gen._sample_corner_points(ids)
    pruned, count = gen._prune_corners(raw, ids)
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(pruned[e, :c]).all()
        if c < cfg.max_num_points:
            assert torch.isnan(pruned[e, c:]).all()
        finite_rows = torch.isfinite(pruned[e]).all(dim=1).sum().item()
        assert finite_rows == c


def test_prune_corners_reproducible():
    pytest.importorskip("warp")
    E = 5
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    a = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=3))
    b = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=3))
    pa, ca = a._prune_corners(a._sample_corner_points(ids), ids)
    pb, cb = b._prune_corners(b._sample_corner_points(ids), ids)
    assert torch.equal(ca, cb)
    assert torch.equal(torch.isnan(pa), torch.isnan(pb))
