import dataclasses
import importlib

import numpy as np
import pytest

from tests.conftest import override_spec
from track_gen._src import gate_generator_registry as reg
from track_gen.plugin import DuplicateSpecError


def test_gate_registry_available_returns_a_list():
    names = reg.available()
    assert isinstance(names, list)


def test_unknown_gate_generator_raises_with_available_list():
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "available" in str(e.value)


def test_duplicate_gate_generator_registration_from_different_module_raises(plugin_registry):
    def first_generate(seeds_wp, config, out, scratch):
        return None

    def second_generate(seeds_wp, config, out, scratch):
        return None

    second_generate.__module__ = "other_gate_module"

    spec = reg.GateGeneratorSpec(
        name="tests.fake",
        alloc_scratch=lambda config: object(),
        generate=first_generate,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    )
    duplicate = reg.GateGeneratorSpec(
        name="tests.fake",
        alloc_scratch=lambda config: object(),
        generate=second_generate,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    )
    plugin_registry.add(spec)
    with pytest.raises(DuplicateSpecError, match="already registered") as e:
        plugin_registry.add(duplicate)
    assert isinstance(e.value, ValueError)
    assert __name__ in str(e.value) and "other_gate_module" in str(e.value)
    assert reg.get("tests.fake") is spec
    # registry.add: another spec object from the same module is a duplicate too (no
    # module-keyed exemption); only replace=True overrides. The identical object again
    # is a no-op.
    same_module = dataclasses.replace(spec, max_gates=lambda config: 8)
    with pytest.raises(DuplicateSpecError):
        plugin_registry.add(same_module)
    plugin_registry.add(spec)
    assert reg.get("tests.fake") is spec
    # The deprecated reg.register keeps the old gate rule: another module's spec raises
    # (a ValueError, as before); the same module's overwrites, with a DeprecationWarning.
    with pytest.raises(ValueError, match="already registered"):
        reg.register(duplicate)
    assert reg.get("tests.fake") is spec
    with pytest.warns(DeprecationWarning):
        reg.register(same_module)
    assert reg.get("tests.fake") is same_module


def test_reloading_gate_generator_module_keeps_specs_registered():
    # Importing (or re-importing) a generator module has no registration side effect.
    from track_gen._src import warp_generate_gates

    assert "bezier" in reg.available()
    before = {name: reg.get(name) for name in ("bezier", "hull")}
    importlib.reload(warp_generate_gates)

    names = reg.available()
    assert "bezier" in names
    assert "hull" in names
    assert {name: reg.get(name) for name in ("bezier", "hull")} == before


def test_reloading_structured_gate_generator_modules_keeps_specs_registered():
    from track_gen._src import warp_generate_checkpoint_gates
    from track_gen._src import warp_generate_polar_gates
    from track_gen._src import warp_generate_voronoi_gates

    names = reg.available()
    assert "checkpoint" in names
    assert "polar" in names
    assert "voronoi" in names

    importlib.reload(warp_generate_checkpoint_gates)
    importlib.reload(warp_generate_polar_gates)
    importlib.reload(warp_generate_voronoi_gates)

    names = reg.available()
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

    assert "bezier" in reg.available()
    with registry.scoped(fresh=True):
        assert not registry._loaded and registry._specs["gate_generator"] == {}
        names = reg.available()
        assert "bezier" in names
        assert "checkpoint" in names
        assert "hull" in names
        assert "polar" in names
        assert "voronoi" in names


def test_all_standard_gate_generators_registered():
    # Superset: installed plugins may add more gate generators.
    assert {"bezier", "checkpoint", "hull", "polar", "voronoi"} <= set(reg.available())
    for name in ("bezier", "checkpoint", "hull", "polar", "voronoi"):
        spec = reg.get(name)
        assert spec.name == name
        assert callable(spec.alloc_scratch)
        assert callable(spec.generate)
        assert callable(spec.max_gates)

    assert reg.get("bezier").supported_orderings == frozenset({"ccw", "random_pairs"})
    assert reg.get("hull").supported_orderings == frozenset({"ccw", "random_pairs"})
    for name in ("checkpoint", "polar", "voronoi"):
        assert reg.get(name).supported_orderings == frozenset({"ccw", "raw"})


def test_gate_generator_spec_capturable_defaults_true():
    # Mirrors GeneratorSpec: a spec registered without the flag stays on the
    # capture-then-replay CUDA path.
    spec = reg.GateGeneratorSpec(
        name="x",
        alloc_scratch=lambda config: None,
        generate=lambda *a: None,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    )
    assert spec.capturable is True


def test_shipped_gate_generators_are_capturable():
    for name in reg.available():
        assert reg.get(name).capturable is True, name


_GATE_DEVS = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("dev", _GATE_DEVS)
def test_noncapturable_gate_generator_runs_eagerly(plugin_registry, dev):
    """A capturable=False gate generator never gets a wp.Graph: its Python body runs
    once at construction (the warm-up pass) and then exactly once per generate() on
    every device (a self-graphed path would run it once more to record on the first
    CUDA call and 0x on every replay)."""
    import dataclasses

    import warp as wp

    from track_gen._src.gate_generator import GateGenerator
    from track_gen._src.rng_utils import PerEnvSeededRNG
    from track_gen._src.types import GateGenConfig

    if dev == "cuda" and not wp.is_cuda_available():
        pytest.skip("cuda")

    base = reg.get("bezier")
    calls = []

    def counting_generate(*args, **kwargs):
        calls.append(1)
        return base.generate(*args, **kwargs)

    eager = dataclasses.replace(base, generate=counting_generate, capturable=False)
    override_spec(plugin_registry, eager)

    E = 4
    cfg = GateGenConfig(generator="bezier", device=dev, num_envs=E, gate_width=0.05)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=1234, num_envs=E, device=dev))
    assert len(calls) == 1                   # construction: one real warm-up pass
    first = gen.generate().position.numpy().copy()
    second = gen.generate().position.numpy().copy()

    assert gen.graph is None
    assert len(calls) == 3
    np.testing.assert_array_equal(first, second)
