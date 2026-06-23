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
