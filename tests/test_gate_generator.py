import pytest

from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen._src import gate_generator_registry as reg


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
