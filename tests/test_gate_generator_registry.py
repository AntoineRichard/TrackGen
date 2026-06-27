import importlib

import pytest

from track_gen._src import gate_generator_registry as reg


def test_gate_registry_available_returns_a_list():
    names = reg.available()
    assert isinstance(names, list)


def test_unknown_gate_generator_raises_with_available_list():
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "available" in str(e.value)


def test_duplicate_gate_generator_registration_from_different_module_raises(monkeypatch):
    monkeypatch.setattr(reg, "GATE_GENERATORS", {})
    monkeypatch.setattr(reg, "_LOADED", True)

    def first_generate(seeds_wp, config, out, scratch):
        return None

    def second_generate(seeds_wp, config, out, scratch):
        return None

    second_generate.__module__ = "other_gate_module"

    spec = reg.GateGeneratorSpec(
        name="fake",
        alloc_scratch=lambda config: object(),
        generate=first_generate,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    )
    duplicate = reg.GateGeneratorSpec(
        name="fake",
        alloc_scratch=lambda config: object(),
        generate=second_generate,
        max_gates=lambda config: 4,
        supported_orderings=frozenset({"ccw"}),
    )
    reg.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(duplicate)


def test_reloading_gate_generator_module_keeps_specs_registered():
    from track_gen._src import warp_generate_gates

    assert "bezier" in reg.available()
    importlib.reload(warp_generate_gates)

    names = reg.available()
    assert "bezier" in names
    assert "hull" in names


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


def test_reloading_registry_repopulates_already_imported_gate_modules():
    from track_gen._src import warp_generate_gates  # noqa: F401
    from track_gen._src import warp_generate_checkpoint_gates  # noqa: F401
    from track_gen._src import warp_generate_polar_gates  # noqa: F401
    from track_gen._src import warp_generate_voronoi_gates  # noqa: F401

    assert "bezier" in reg.available()
    importlib.reload(reg)

    names = reg.available()
    assert "bezier" in names
    assert "checkpoint" in names
    assert "hull" in names
    assert "polar" in names
    assert "voronoi" in names


def test_all_standard_gate_generators_registered():
    assert reg.available() == ["bezier", "checkpoint", "hull", "polar", "voronoi"]
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
