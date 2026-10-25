import dataclasses
import importlib

import numpy as np
import pytest

from track_gen.plugin import registry as reg
from track_gen.plugin import DuplicateSpecError


def test_gate_registry_available_returns_a_list():
    names = reg.available("gate_generator")
    assert isinstance(names, list)


def test_unknown_gate_generator_raises_with_available_list():
    with pytest.raises(ValueError) as e:
        reg.get("gate_generator", "does-not-exist")
    assert "available" in str(e.value)




def test_reloading_gate_generator_module_keeps_specs_registered():
    # Importing (or re-importing) a generator module has no registration side effect.
    from track_gen._src import warp_generate_gates

    assert "bezier" in reg.available("gate_generator")
    before = {name: reg.get("gate_generator", name) for name in ("bezier", "hull")}
    importlib.reload(warp_generate_gates)

    names = reg.available("gate_generator")
    assert "bezier" in names
    assert "hull" in names
    assert {name: reg.get("gate_generator", name) for name in ("bezier", "hull")} == before


def test_reloading_structured_gate_generator_modules_keeps_specs_registered():
    from track_gen._src import warp_generate_checkpoint_gates
    from track_gen._src import warp_generate_polar_gates
    from track_gen._src import warp_generate_voronoi_gates

    names = reg.available("gate_generator")
    assert "checkpoint" in names
    assert "polar" in names
    assert "voronoi" in names

    importlib.reload(warp_generate_checkpoint_gates)
    importlib.reload(warp_generate_polar_gates)
    importlib.reload(warp_generate_voronoi_gates)

    names = reg.available("gate_generator")
    assert "checkpoint" in names
    assert "polar" in names
    assert "voronoi" in names


def test_fresh_registry_repopulates_already_imported_gate_modules():
    # Built-ins are registered explicitly on (re)load, whether or not their modules
    # were imported before: nothing depends on import order or import side effects.
    from track_gen._src import warp_generate_gates  # noqa: F401
    from track_gen._src import warp_generate_checkpoint_gates  # noqa: F401
    from track_gen._src import warp_generate_polar_gates  # noqa: F401
    from track_gen._src import warp_generate_voronoi_gates  # noqa: F401
    from track_gen.plugin import registry

    assert "bezier" in reg.available("gate_generator")
    with registry.scoped(fresh=True):
        assert not registry._loaded and registry._specs["gate_generator"] == {}
        names = reg.available("gate_generator")
        assert "bezier" in names
        assert "checkpoint" in names
        assert "hull" in names
        assert "polar" in names
        assert "voronoi" in names


def test_all_standard_gate_generators_registered():
    # Superset: installed plugins may add more gate generators.
    # PR11: the built-ins are gate_generator Stages (output="anchors"); their orderings
    # are exclusions (so a third-party order runs with every built-in), which resolve to
    # 0.2.0.dev3's supported lists while only the built-in orders are registered.
    from track_gen.plugin import Stage, registry
    from track_gen.stages.gates import _supported_orderings

    assert {"bezier", "checkpoint", "hull", "polar", "voronoi"} <= set(reg.available("gate_generator"))
    for name in ("bezier", "checkpoint", "hull", "polar", "voronoi"):
        spec = reg.get("gate_generator", name)
        assert spec.name == name
        assert issubclass(spec, Stage)
        assert spec.kind == "gate_generator" and spec.output == "anchors"
        info = registry.describe("gate_generator", name)
        assert info.orderings is None and info.output == "anchors"

    for name in ("bezier", "hull"):
        assert registry.describe("gate_generator", name).excluded_orderings == \
            frozenset({"raw"})
        assert _supported_orderings(registry, name) == ["ccw", "random_pairs"]
    for name in ("checkpoint", "polar", "voronoi"):
        assert registry.describe("gate_generator", name).excluded_orderings == \
            frozenset({"random_pairs"})
        assert _supported_orderings(registry, name) == ["ccw", "raw"]




def test_shipped_gate_generators_are_capturable():
    from track_gen.plugin import registry

    for name in reg.available("gate_generator"):
        assert registry.describe("gate_generator", name).capturable is True, name


_GATE_DEVS = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]




@pytest.mark.parametrize("dev", _GATE_DEVS)
def test_host_gate_stage_runs_eagerly(plugin_registry, dev):
    import warp as wp

    from tests._plugin_toys import TOY_RING
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.plugin import CaptureClass

    if dev == "cuda" and not wp.is_cuda_available():
        pytest.skip("cuda")
    calls = []

    class HostRing(TOY_RING):
        capture = CaptureClass.HOST

        def build(self, b, p):
            enqueue = super().build(b, p)

            def counted(ctx):
                calls.append(1)
                enqueue(ctx)
            return counted

    plugin_registry.add(HostRing)
    cfg = GateGenConfig(generator="toy.ring", gate_ordering="raw", device=dev,
                        num_envs=4, gate_width=0.05)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=1234, num_envs=4, device=dev))
    assert len(calls) == 1
    first = gen.generate().position.numpy().copy()
    second = gen.generate().position.numpy().copy()
    assert gen.graph is None and len(calls) == 3
    np.testing.assert_array_equal(first, second)
