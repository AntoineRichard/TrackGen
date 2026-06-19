import math
import types

import torch
import pytest

from tests._oracle.generators import Centerline, CenterlineGenerator
from tests._oracle.generators import BezierCenterlineGenerator


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


def _square_corners(E=2):
    sq = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return sq.unsqueeze(0).expand(E, 4, 2).contiguous()


def test_assemble_centerline_shape():
    cfg = _bezier_config(num_points_per_segment=30)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    corners = _square_corners(E=3)  # P=4
    dense = gen._assemble_centerline(corners)
    assert dense.shape == (3, 4 * 30, 2)


def test_assemble_centerline_is_closed_loop():
    cfg = _bezier_config(num_points_per_segment=30, rad=0.2, edgy=0.0)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    corners = _square_corners(E=1)
    dense = gen._assemble_centerline(corners)
    assert torch.isfinite(dense).all()
    gap = torch.linalg.norm(dense[0, -1] - dense[0, 0])
    seg_step = torch.linalg.norm(dense[0, 1] - dense[0, 0])
    assert gap <= 3.0 * seg_step + 1e-4


def test_assemble_centerline_nan_corner_propagates():
    cfg = _bezier_config(num_points_per_segment=30)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    # Use a hexagon: vertex_tangents makes a pruned corner poison its tangent
    # plus its two neighbours' tangents (4 consecutive cubic segments). With a
    # 4-corner square that is ALL segments, so nothing finite survives; with
    # >=6 corners at least one fully-finite segment remains, which is what lets
    # us assert that the NaN propagates *locally* rather than destroying the
    # whole dense polyline.
    ang = torch.arange(6, dtype=torch.float32) * (2.0 * torch.pi / 6.0)
    corners = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).unsqueeze(0)  # [1, 6, 2]
    corners[0, 2] = float("nan")  # prune the 3rd corner
    dense = gen._assemble_centerline(corners)
    assert torch.isnan(dense[0]).any()
    assert torch.isfinite(dense[0]).any()


def test_corner_angles_clamped_no_nan():
    cfg = _bezier_config()
    gen = BezierCenterlineGenerator(cfg, rng=None)
    corners = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # repeated corner
    ang = gen._corner_angles(corners)
    assert ang.shape == (1, 4)
    assert torch.isfinite(ang).all()


def test_generate_returns_centerline():
    pytest.importorskip("warp")
    E = 8
    cfg = _bezier_config(num_envs=E, device="cpu")
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=11))
    ids = torch.arange(E)
    cl = gen.generate(ids)
    from tests._oracle.generators import Centerline

    assert isinstance(cl, Centerline)
    M_max = cfg.max_num_points * cfg.num_points_per_segment
    assert cl.points.shape == (E, M_max, 2)
    assert cl.valid.shape == (E,)
    assert cl.valid.dtype == torch.bool


def test_generate_reproducible():
    pytest.importorskip("warp")
    E = 6
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    a = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=5)).generate(ids)
    b = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=5)).generate(ids)
    assert torch.equal(torch.isnan(a.points), torch.isnan(b.points))
    fin = torch.isfinite(a.points)
    assert torch.allclose(a.points[fin], b.points[fin])
    assert torch.equal(a.valid, b.valid)


def test_generate_pathological_flags_invalid_without_hang():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu", min_angle=3.10, max_regen_iters=3)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=1))
    ids = torch.arange(E)
    cl = gen.generate(ids)  # must return, not hang
    assert cl.valid.shape == (E,)
    assert (~cl.valid).all()


def test_generate_accepts_pruned_variable_count_tracks():
    pytest.importorskip("warp")
    # With a wide [min,max] count window, many envs draw < max corners. The
    # NaN-aware gates must still accept geometrically-good pruned tracks, so
    # not every valid env has exactly max_num_points corners.
    E = 32
    cfg = _bezier_config(num_envs=E, device="cpu", min_num_points=6, max_num_points=13)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=99))
    ids = torch.arange(E)
    cl = gen.generate(ids)
    # At least one valid env exists and at least one valid env has a NaN tail
    # (i.e. a pruned, variable-count track was accepted).
    valid_idx = torch.where(cl.valid)[0]
    assert valid_idx.numel() > 0
    has_nan_tail = torch.tensor(
        [bool(torch.isnan(cl.points[e]).any()) for e in valid_idx.tolist()]
    )
    assert has_nan_tail.any(), "no pruned variable-count track was accepted"


from tests._oracle.generators import FourierCenterlineGenerator


def _fourier_config(**overrides):
    cfg = types.SimpleNamespace(
        num_harmonics=3,
        decay_p=2.0,
        amplitude=1.0,
        scale=10.0,
        num_centerline_samples=256,
        device="cpu",
        num_envs=4,
        turning_tol=0.5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_fourier_generate_shape_and_closed():
    pytest.importorskip("warp")
    E = 4
    cfg = _fourier_config(num_envs=E, num_centerline_samples=256)
    gen = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=21))
    ids = torch.arange(E)
    cl = gen.generate(ids)
    assert cl.points.shape == (E, 256, 2)
    assert torch.isfinite(cl.points).all()
    for e in range(E):
        gap = torch.linalg.norm(cl.points[e, -1] - cl.points[e, 0])
        step = torch.linalg.norm(cl.points[e, 1] - cl.points[e, 0])
        assert gap <= 3.0 * step + 1e-4


def test_fourier_mean_centered_and_scaled():
    pytest.importorskip("warp")
    E = 3
    cfg = _fourier_config(num_envs=E, scale=10.0)
    gen = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=8))
    cl = gen.generate(torch.arange(E))
    for e in range(E):
        center = cl.points[e].mean(dim=0)
        assert torch.allclose(center, torch.zeros(2), atol=1e-4)
        bbox = cl.points[e].amax(dim=0) - cl.points[e].amin(dim=0)
        assert bbox.amax().item() == pytest.approx(10.0, abs=1e-3)


def test_fourier_reproducible_and_independent():
    pytest.importorskip("warp")
    E = 4
    cfg = _fourier_config(num_envs=E)
    ids = torch.arange(E)
    a = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=2)).generate(ids)
    b = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=2)).generate(ids)
    assert torch.allclose(a.points, b.points)
    import warp as wp  # noqa: F401
    from track_gen.rng_utils import PerEnvSeededRNG

    seeds = torch.tensor([2, 3, 4, 999], dtype=torch.int32)
    c = FourierCenterlineGenerator(cfg, rng=PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")).generate(ids)
    assert torch.allclose(a.points[0], c.points[0])
    assert not torch.allclose(a.points[3], c.points[3])


def test_fourier_low_k_turning_is_loop():
    pytest.importorskip("warp")
    from tests._oracle.geometry import turning_number

    E = 4
    cfg = _fourier_config(num_envs=E, num_harmonics=1, decay_p=2.0)
    cl = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=14)).generate(torch.arange(E))
    turn = turning_number(cl.points)
    assert torch.allclose(turn.abs(), torch.full((E,), 2.0 * math.pi), atol=cfg.turning_tol)
    assert cl.valid.dtype == torch.bool


def test_module_exposes_both_generators():
    from tests._oracle.generators import (
        BezierCenterlineGenerator,
        Centerline,
        CenterlineGenerator,
        FourierCenterlineGenerator,
    )

    assert issubclass(BezierCenterlineGenerator, CenterlineGenerator)
    assert issubclass(FourierCenterlineGenerator, CenterlineGenerator)
    assert Centerline.__dataclass_fields__.keys() == {"points", "valid"}
