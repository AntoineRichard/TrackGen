import pytest
import torch

from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen._src import gate_generator_registry as reg
from tests._warp_compare import to_t


def _make_rng(num_envs: int, seed: int = 0, device: str = "cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


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
        max_gates=lambda config: 1,
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
    rng = type("Rng", (), {"seeds_warp": object()})()
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
        min_gate_distance=0.0,
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=31)).generate(E)
    position = to_t(gates.position).view(E, G, 2)
    tangent = to_t(gates.tangent).view(E, G, 2)
    count = to_t(gates.count)
    valid = to_t(gates.valid).bool()

    assert valid.all()
    assert torch.all(count >= cfg.min_gates)
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(position[e, :c]).all()
        assert torch.isfinite(tangent[e, :c]).all()
        assert torch.isnan(position[e, c:]).all()
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
            min_gate_distance=0.0,
        )
        gates = GateGenerator(cfg, _make_rng(E, seed=71)).generate(E)
        position = to_t(gates.position).view(E, G, 2)
        tangent = to_t(gates.tangent).view(E, G, 2)
        count = to_t(gates.count)
        valid = to_t(gates.valid).bool()
        assert valid.all()
        for e in range(E):
            c = int(count[e])
            assert c >= cfg.min_gates
            assert torch.isfinite(position[e, :c]).all()
            assert torch.isfinite(tangent[e, :c]).all()
            assert torch.isnan(position[e, c:]).all()


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
        min_gate_distance=0.0,
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
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(position[e, :c]).all()
        assert torch.isfinite(position[e]).all(dim=-1).sum().item() == c
