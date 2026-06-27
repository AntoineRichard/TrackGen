import pytest
import torch

from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen._src import gate_generator_registry as reg
from tests._warp_compare import to_t


def _make_rng(num_envs: int, seed: int = 0, device: str = "cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


def _assert_generated_gate_fields_are_finite(gates, num_envs: int, max_gates: int):
    fields = [
        to_t(gates.position).view(num_envs, max_gates, 2),
        to_t(gates.tangent).view(num_envs, max_gates, 2),
        to_t(gates.normal).view(num_envs, max_gates, 2),
        to_t(gates.left).view(num_envs, max_gates, 2),
        to_t(gates.right).view(num_envs, max_gates, 2),
    ]
    count = to_t(gates.count)
    for e in range(num_envs):
        c = int(count[e])
        for field in fields:
            assert torch.isfinite(field[e, :c]).all()
            assert torch.isnan(field[e, c:]).all()


def test_gate_generator_requires_rng():
    cfg = GateGenConfig()
    with pytest.raises(ValueError, match="random number generator"):
        GateGenerator(cfg, None)


def test_gate_generator_unknown_generator_raises_before_warp_gate_import():
    cfg = GateGenConfig(generator="does-not-exist")
    rng = PerEnvSeededRNG(seeds=0, num_envs=cfg.num_envs, device=cfg.device)
    with pytest.raises(ValueError, match="unknown gate generator"):
        GateGenerator(cfg, rng)


def test_gate_generator_unsupported_ordering_raises(monkeypatch):
    monkeypatch.setattr(reg, "GATE_GENERATORS", {})
    monkeypatch.setattr(reg, "_LOADED", True)
    reg.register(reg.GateGeneratorSpec(
        name="fake",
        alloc_scratch=lambda config: object(),
        generate=lambda seeds_wp, config, out, scratch: None,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    ))
    cfg = GateGenConfig(generator="fake", gate_ordering="raw")
    rng = PerEnvSeededRNG(seeds=0, num_envs=cfg.num_envs, device=cfg.device)
    with pytest.raises(ValueError, match="does not support gate_ordering"):
        GateGenerator(cfg, rng)


@pytest.mark.parametrize(
    ("cfg_kwargs", "producible"),
    [
        ({"generator": "bezier", "max_num_points": 13}, 13),
        ({"generator": "hull", "max_num_points": 13}, 13),
        ({"generator": "polar", "polar_num_knots": 12}, 12),
        ({"generator": "voronoi", "voronoi_control_points": 12, "voronoi_num_sites": 12}, 12),
        ({"generator": "checkpoint", "checkpoint_count": 12}, 12),
    ],
)
def test_gate_generator_rejects_unreachable_min_gates(cfg_kwargs, producible):
    cfg = GateGenConfig(min_gates=20, max_gates=32, gate_radius=0.0, **cfg_kwargs)
    with pytest.raises(ValueError, match=rf"min_gates.*{producible}"):
        GateGenerator(cfg, _make_rng(cfg.num_envs, seed=13))


def test_gate_generator_rejects_too_small_max_gates(monkeypatch):
    monkeypatch.setattr(reg, "GATE_GENERATORS", {})
    monkeypatch.setattr(reg, "_LOADED", True)
    reg.register(reg.GateGeneratorSpec(
        name="fake",
        alloc_scratch=lambda config: object(),
        generate=lambda seeds_wp, config, out, scratch: None,
        max_gates=lambda config: 64,
        supported_orderings=frozenset({"ccw"}),
    ))
    cfg = GateGenConfig(generator="fake", max_gates=32)
    rng = PerEnvSeededRNG(seeds=0, num_envs=cfg.num_envs, device=cfg.device)
    with pytest.raises(ValueError, match="max_gates"):
        GateGenerator(cfg, rng)


def test_cuda_capture_sets_gate_and_pipeline_capture_flags(monkeypatch):
    from track_gen._src import gate_generator as gate_generator_mod
    from track_gen._src import warp_gate, warp_pipeline

    monkeypatch.setattr(reg, "GATE_GENERATORS", {})
    monkeypatch.setattr(reg, "_LOADED", True)
    reg.register(reg.GateGeneratorSpec(
        name="fake-cuda",
        alloc_scratch=lambda config: object(),
        generate=lambda seeds_wp, config, out, scratch: None,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    ))

    monkeypatch.setattr(warp_gate, "_CAPTURING", False)
    monkeypatch.setattr(warp_pipeline, "_CAPTURING", False)
    monkeypatch.setattr(
        warp_gate, "_gate_warp_alloc", lambda config, generator_spec: (object(), object())
    )

    calls = []

    def fake_run_gate_pipeline(config, seed_buf_wp, out, scratch, generator_spec):
        assert warp_gate._CAPTURING is True
        assert warp_pipeline._CAPTURING is True
        calls.append("run")

    class FakeCapture:
        def __init__(self, device):
            self.device = device
            self.graph = object()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    launched = []
    monkeypatch.setattr(warp_gate, "_run_gate_pipeline", fake_run_gate_pipeline)
    monkeypatch.setattr(gate_generator_mod.wp, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(gate_generator_mod.wp, "copy", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate_generator_mod.wp, "synchronize", lambda: None)
    monkeypatch.setattr(gate_generator_mod.wp, "ScopedCapture", FakeCapture)
    monkeypatch.setattr(
        gate_generator_mod.wp, "capture_launch", lambda graph: launched.append(graph)
    )

    cfg = GateGenConfig(generator="fake-cuda", device="cuda", num_envs=1)
    # seeds_warp.shape[0] must match num_envs (checked at construction).
    rng = type("Rng", (), {"seeds_warp": type("Seeds", (), {"shape": (1,)})()})()
    gen = GateGenerator(cfg, rng)

    assert gen.generate(1) is gen._gate_sequence

    assert len(calls) == 4
    assert launched == [gen._graph]
    assert warp_gate._CAPTURING is False
    assert warp_pipeline._CAPTURING is False


@pytest.mark.parametrize("generator", ["bezier", "hull"])
@pytest.mark.parametrize("ordering", ["ccw", "random_pairs"])
def test_point_family_gate_generators_emit_finite_native_gates(generator, ordering):
    E, G = 8, 32
    cfg = GateGenConfig(
        generator=generator,
        gate_ordering=ordering,
        num_envs=E,
        max_gates=G,
        device="cpu",
        gate_radius=0.0,
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=31)).generate(E)
    position = to_t(gates.position).view(E, G, 2)
    count = to_t(gates.count)
    valid = to_t(gates.valid).bool()

    assert valid.all()
    assert torch.all(count >= cfg.min_gates)
    _assert_generated_gate_fields_are_finite(gates, E, G)
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(position[e]).all(dim=-1).sum().item() == c


@pytest.mark.parametrize(
    ("generator", "orderings"),
    [
        ("polar", ["ccw", "raw"]),
        ("voronoi", ["ccw", "raw"]),
        ("checkpoint", ["ccw", "raw"]),
    ],
)
def test_structured_gate_generators_emit_finite_native_gates(generator, orderings):
    E, G = 6, 32
    for ordering in orderings:
        cfg = GateGenConfig(
            generator=generator,
            gate_ordering=ordering,
            num_envs=E,
            max_gates=G,
            device="cpu",
            gate_radius=0.0,
        )
        gates = GateGenerator(cfg, _make_rng(E, seed=71)).generate(E)
        count = to_t(gates.count)
        valid = to_t(gates.valid).bool()
        assert valid.all()
        _assert_generated_gate_fields_are_finite(gates, E, G)
        for e in range(E):
            c = int(count[e])
            assert c >= cfg.min_gates


def test_polar_gate_generator_capacity_uses_clamped_knot_count():
    cfg = GateGenConfig(generator="polar", polar_num_knots=3, max_gates=3, min_gates=2)
    rng = PerEnvSeededRNG(seeds=0, num_envs=cfg.num_envs, device=cfg.device)
    with pytest.raises(ValueError, match="max_gates"):
        GateGenerator(cfg, rng)


@pytest.mark.parametrize(
    ("cfg_kwargs", "required"),
    [
        ({"generator": "polar", "polar_num_knots": 3, "max_gates": 4}, 4),
        (
            {
                "generator": "voronoi",
                "voronoi_control_points": 7,
                "voronoi_num_sites": 7,
                "max_gates": 7,
            },
            7,
        ),
        ({"generator": "checkpoint", "checkpoint_count": 5, "max_gates": 5}, 5),
    ],
)
def test_structured_gate_generators_succeed_at_tight_capacity(cfg_kwargs, required):
    cfg = GateGenConfig(num_envs=3, gate_radius=0.0, **cfg_kwargs)
    gates = GateGenerator(cfg, _make_rng(cfg.num_envs, seed=83)).generate(cfg.num_envs)
    count = to_t(gates.count)
    assert torch.equal(count, torch.full_like(count, required))


@pytest.mark.parametrize("generator", ["polar", "voronoi", "checkpoint"])
def test_structured_gate_generators_reject_random_pairs_ordering(generator):
    cfg = GateGenConfig(generator=generator, gate_ordering="random_pairs")
    with pytest.raises(ValueError, match="does not support gate_ordering"):
        GateGenerator(cfg, _make_rng(cfg.num_envs, seed=97))


@pytest.mark.parametrize("generator", ["bezier", "hull"])
def test_point_family_gate_generators_reject_too_small_max_gates(generator):
    cfg = GateGenConfig(
        generator=generator,
        num_envs=1,
        max_gates=8,
        max_num_points=13,
        device="cpu",
    )
    with pytest.raises(ValueError, match="max_gates"):
        GateGenerator(cfg, _make_rng(1, seed=7))


def test_gate_generator_independent_instances_with_same_seed_are_deterministic():
    E, G = 4, 32
    cfg = GateGenConfig(
        generator="bezier",
        gate_ordering="random_pairs",
        num_envs=E,
        max_gates=G,
        device="cpu",
        gate_radius=0.0,
    )
    gates_a = GateGenerator(cfg, _make_rng(E, seed=211)).generate(E)
    gates_b = GateGenerator(cfg, _make_rng(E, seed=211)).generate(E)

    for name in ("position", "tangent", "normal", "left", "right"):
        a = to_t(getattr(gates_a, name))
        b = to_t(getattr(gates_b, name))
        assert torch.allclose(a, b, equal_nan=True)
    assert torch.equal(to_t(gates_a.count), to_t(gates_b.count))
    assert torch.equal(to_t(gates_a.valid), to_t(gates_b.valid))


def test_gate_generator_cpu_reuses_output_instance_and_buffers():
    E, G = 2, 32
    cfg = GateGenConfig(
        generator="bezier",
        gate_ordering="ccw",
        num_envs=E,
        max_gates=G,
        device="cpu",
        gate_radius=0.0,
    )
    gen = GateGenerator(cfg, _make_rng(E, seed=41))

    first = gen.generate(E)
    ptr = first.position.ptr
    second = gen.generate()

    assert second is first
    assert second.position.ptr == ptr
    assert to_t(second.valid).bool().all()
    _assert_generated_gate_fields_are_finite(second, E, G)


def test_gate_generator_sphere_solve_repairs_overlapping_native_points(monkeypatch):
    import types
    import warp as wp

    def alloc_scratch(config):
        nan = float("nan")
        return types.SimpleNamespace(
            position=wp.array(
                [
                    wp.vec2f(0.0, 0.0),
                    wp.vec2f(0.01, 0.0),
                    wp.vec2f(nan, nan),
                    wp.vec2f(nan, nan),
                ],
                dtype=wp.vec2f,
                device=str(config.device),
            ),
            count=wp.array([2], dtype=wp.int32, device=str(config.device)),
        )

    def generate(seeds_wp, config, out, scratch):
        wp.copy(out.position, scratch.position)
        wp.copy(out.count, scratch.count)

    monkeypatch.setattr(reg, "GATE_GENERATORS", {})
    monkeypatch.setattr(reg, "_LOADED", True)
    reg.register(reg.GateGeneratorSpec(
        name="overlap",
        alloc_scratch=alloc_scratch,
        generate=generate,
        max_gates=lambda config: 2,
        supported_orderings=frozenset({"raw"}),
    ))

    base_kwargs = dict(
        generator="overlap",
        gate_ordering="raw",
        num_envs=1,
        min_gates=2,
        max_gates=4,
        gate_radius=0.1,
        device="cpu",
    )

    raw_cfg = GateGenConfig(gate_solve_iters=0, **base_kwargs)
    raw_gates = GateGenerator(raw_cfg, _make_rng(1, seed=53)).generate()
    raw_position = to_t(raw_gates.position).view(1, 4, 2)
    raw_distance = torch.linalg.norm(raw_position[0, 1] - raw_position[0, 0])
    assert to_t(raw_gates.valid).bool().tolist() == [False]
    assert raw_distance < 0.2

    solved_cfg = GateGenConfig(gate_solve_iters=1, **base_kwargs)
    gates = GateGenerator(solved_cfg, _make_rng(1, seed=53)).generate()

    position = to_t(gates.position).view(1, 4, 2)
    tangent = to_t(gates.tangent).view(1, 4, 2)
    distance = torch.linalg.norm(position[0, 1] - position[0, 0])
    assert to_t(gates.valid).bool().tolist() == [True]
    assert distance >= 0.2 - 1e-6
    assert torch.isfinite(tangent[0, :2]).all()
    assert torch.isnan(position[0, 2:]).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
@pytest.mark.parametrize("generator", ["bezier", "hull"])
def test_point_family_gate_generators_cuda_capture_reuses_output(generator):
    E, G = 2, 32
    cfg = GateGenConfig(
        generator=generator,
        gate_ordering="ccw",
        num_envs=E,
        max_gates=G,
        device="cuda:0",
        gate_radius=0.0,
    )
    gen = GateGenerator(cfg, _make_rng(E, seed=41, device="cuda:0"))

    first = gen.generate(E)
    ptr = first.position.ptr
    second = gen.generate(E)
    torch.cuda.synchronize()

    assert second is first
    assert second.position.ptr == ptr

    position = to_t(second.position).view(E, G, 2)
    count = to_t(second.count)
    valid = to_t(second.valid).bool()
    assert valid.all()
    _assert_generated_gate_fields_are_finite(second, E, G)
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(position[e]).all(dim=-1).sum().item() == c


def test_gate_generator_invalidates_zero_count_from_native_generator(monkeypatch):
    monkeypatch.setattr(reg, "GATE_GENERATORS", {})
    monkeypatch.setattr(reg, "_LOADED", True)
    reg.register(reg.GateGeneratorSpec(
        name="zero-count",
        alloc_scratch=lambda config: object(),
        generate=lambda seeds_wp, config, out, scratch: None,
        max_gates=lambda config: 2,
        supported_orderings=frozenset({"ccw"}),
    ))

    E, G = 3, 2
    cfg = GateGenConfig(
        generator="zero-count",
        num_envs=E,
        min_gates=2,
        max_gates=G,
        device="cpu",
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=101)).generate()

    count = to_t(gates.count)
    assert torch.equal(count, torch.zeros_like(count))
    assert not to_t(gates.valid).bool().any()
    _assert_generated_gate_fields_are_finite(gates, E, G)


def test_gate_generator_invalidates_large_gate_radius():
    E = 4
    cfg = GateGenConfig(
        generator="checkpoint",
        gate_ordering="raw",
        num_envs=E,
        max_gates=32,
        gate_radius=50.0,
        gate_solve_iters=0,
        device="cpu",
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=5)).generate(E)
    assert not to_t(gates.valid).bool().any()


def test_generate_wrong_batch_raises():
    cfg = GateGenConfig(num_envs=4, device="cpu")
    gen = GateGenerator(cfg, _make_rng(4))
    with pytest.raises(ValueError):
        gen.generate(5)


def test_generate_rejects_sequence_ids():
    cfg = GateGenConfig(num_envs=4, device="cpu")
    gen = GateGenerator(cfg, _make_rng(4))
    with pytest.raises(TypeError, match="does not accept explicit environment ids"):
        gen.generate([0, 1, 2, 3])


def test_distinct_per_env_seeds_produce_diverse_gates():
    # A single int seed expands to seed + arange(E) per-env seeds. Guard against an
    # env-collapse regression where every environment generates the identical track,
    # which would silently destroy batch diversity for RL training.
    E, G = 8, 32
    cfg = GateGenConfig(
        generator="bezier", gate_ordering="ccw", num_envs=E, max_gates=G,
        device="cpu", gate_radius=0.0,
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=17)).generate(E)
    position = to_t(gates.position).view(E, G, 2)
    count = to_t(gates.count)

    diverse = False
    for e in range(1, E):
        c0, ce = int(count[0]), int(count[e])
        if c0 != ce or not torch.allclose(
            position[0, :c0], position[e, :ce], equal_nan=True
        ):
            diverse = True
            break
    assert diverse, "distinct per-env seeds must yield at least two differing envs"


def test_different_global_seeds_produce_different_gates():
    E, G = 4, 32
    cfg = GateGenConfig(
        generator="bezier", gate_ordering="ccw", num_envs=E, max_gates=G,
        device="cpu", gate_radius=0.0,
    )
    a = GateGenerator(cfg, _make_rng(E, seed=1)).generate(E)
    b = GateGenerator(cfg, _make_rng(E, seed=999)).generate(E)
    assert not torch.allclose(to_t(a.position), to_t(b.position), equal_nan=True)


def test_clone_is_isolated_from_in_place_regenerate():
    # Exercises the GateSequence aliasing contract: generate() overwrites the same
    # instance in place, while clone() snapshots a fully-owned copy. Mutating the rng
    # seeds between calls is what makes the in-place overwrite observable.
    import numpy as np
    import warp as wp

    E, G = 4, 32
    cfg = GateGenConfig(
        generator="bezier", gate_ordering="ccw", num_envs=E, max_gates=G,
        device="cpu", gate_radius=0.0,
    )
    seeds = wp.array(np.arange(E) + 1, dtype=wp.int32, device="cpu")
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
    gen = GateGenerator(cfg, rng)

    first = gen.generate(E)
    snapshot = first.clone()
    snapshot_pos = to_t(snapshot.position).clone()

    # clone() is a value-preserving deep copy, not an alias.
    assert snapshot.position.ptr != first.position.ptr
    assert torch.allclose(to_t(first.position), snapshot_pos, equal_nan=True)

    # Change the per-env seeds and regenerate: the shared instance is overwritten...
    wp.copy(rng.seeds_warp, wp.array(np.arange(E) + 9999, dtype=wp.int32, device="cpu"))
    second = gen.generate(E)
    assert second is first
    assert not torch.allclose(to_t(second.position), snapshot_pos, equal_nan=True)
    # ...but the earlier clone snapshot is untouched.
    assert torch.allclose(to_t(snapshot.position), snapshot_pos, equal_nan=True)


def _segments_properly_cross(a, b, c, d):
    """Independent mirror of warp_gate._proper_segment_intersection (strict crossing)."""
    def cross(o, p, q):
        return float((p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]))

    def opp(x, y):
        return (x > 0.0 and y < 0.0) or (x < 0.0 and y > 0.0)

    o1, o2 = cross(a, b, c), cross(a, b, d)
    o3, o4 = cross(c, d, a), cross(c, d, b)
    return opp(o1, o2) and opp(o3, o4)


def test_gate_width_flows_through_real_generator_without_crossing_bars():
    # The gate_width collision path is only otherwise tested on hand-built sequences.
    # Run a positive width end-to-end through a real generator and verify both the
    # endpoint geometry and (independently) the non-crossing guarantee for valid envs.
    E, G, width = 6, 32, 0.03
    cfg = GateGenConfig(
        generator="checkpoint", gate_ordering="ccw", num_envs=E, max_gates=G,
        device="cpu", gate_radius=0.02, gate_width=width,
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=123)).generate(E)
    pos = to_t(gates.position).view(E, G, 2)
    nrm = to_t(gates.normal).view(E, G, 2)
    left = to_t(gates.left).view(E, G, 2)
    right = to_t(gates.right).view(E, G, 2)
    count = to_t(gates.count)
    valid = to_t(gates.valid).bool()

    assert valid.any(), "expected at least one valid env with gate_width > 0"
    for e in range(E):
        c = int(count[e])
        assert torch.allclose(left[e, :c], pos[e, :c] + 0.5 * width * nrm[e, :c], atol=1e-5)
        assert torch.allclose(right[e, :c], pos[e, :c] - 0.5 * width * nrm[e, :c], atol=1e-5)
        if valid[e]:
            for i in range(c):
                for j in range(i + 1, c):
                    assert not _segments_properly_cross(
                        left[e, i], right[e, i], left[e, j], right[e, j]
                    ), f"valid env {e} has crossing gate bars {i},{j}"


def test_gate_generator_rejects_rng_env_count_mismatch():
    cfg = GateGenConfig(num_envs=4, device="cpu")
    rng = PerEnvSeededRNG(seeds=0, num_envs=3, device="cpu")  # wrong env count
    with pytest.raises(ValueError, match="seeds"):
        GateGenerator(cfg, rng)
