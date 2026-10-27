"""Diagnostic producers retain raw geometry unless their API explicitly requests poison."""
from __future__ import annotations

import numpy as np
import pytest

from track_gen import GateGenConfig, GateGenerator, TrackGenConfig, TrackGenerator
from track_gen.pipeline import presets
from track_gen.plugin import registry
from track_gen.testing.conformance import _DeviceRun, _StageUnderTest


@pytest.mark.parametrize("kind", ["generator", "gate_generator"])
@pytest.mark.parametrize("policy", [None, "poison"])
def test_generator_diagnostics_default_to_keep_and_honor_explicit_poison(kind, policy):
    options = {"N_max": 8} if kind == "generator" else {
        "gate_radius": 100.0, "gate_solve_iters": 0}
    if policy is not None:
        options["on_invalid"] = policy
    seeds = np.array([5, 6], np.int32)
    unit = _DeviceRun(kind, "bezier", None, "cpu", 2, seeds, seeds + 100,
                      options, registry.describe(kind, "bezier"), (), (), frozenset())
    result = unit.make(seeds).generate()
    assert result.valid.numpy().tolist() == [0, 0]
    assert (result.count.numpy() == 0).all() if policy == "poison" else \
        (result.count.numpy() > 0).all()


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("policy", [None, "poison"])
def test_stage_diagnostics_default_to_keep_and_honor_explicit_poison(gate, policy):
    kind, name = ("order", "ccw") if gate else ("relax", "none")
    options = {"source": "anchors", "gate_radius": 100.0, "gate_solve_iters": 0} \
        if gate else {"n_max": 8}
    if policy is not None:
        options["on_invalid"] = policy
    stage = _StageUnderTest(kind, name, None, options, registry.get(kind, name))
    facade = GateGenerator if gate else TrackGenerator
    result = facade(stage.spec(2, "cpu"), seeds=5).generate()
    assert result.valid.numpy().tolist() == [0, 0]
    assert (result.count.numpy() == 0).all() if policy == "poison" else \
        (result.count.numpy() > 0).all()


@pytest.mark.parametrize("source", ["track", "gates", "track_config", "gate_config",
                                      "track_spec", "gate_spec"])
def test_component_diagnostics_force_keep_for_every_source_form(source):
    from track_gen.testing.components import _producer

    if source == "track_config":
        source = TrackGenConfig(N_max=8)
    elif source == "gate_config":
        source = GateGenConfig(gate_radius=100.0, gate_solve_iters=0)
    elif source == "track_spec":
        source = presets.track(num_envs=2, device="cpu", n_max=8)
    elif source == "gate_spec":
        source = presets.gates(num_envs=2, device="cpu", gate_radius=100.0,
                                gate_solve_iters=0)
    producer = _producer(source, 2, "cpu", np.array([5, 6], np.int32))
    assert producer.program.on_invalid == "keep"
    assert (producer.generate().count.numpy() > 0).all()
