import pytest

from track_gen._src import gate_generator_registry as reg


def test_gate_registry_available_returns_a_list():
    names = reg.available()
    assert isinstance(names, list)


def test_unknown_gate_generator_raises_with_available_list():
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "available" in str(e.value)


def test_bezier_and_hull_gate_generators_registered():
    names = reg.available()
    assert "bezier" in names
    assert "hull" in names
    for name in ("bezier", "hull"):
        spec = reg.get(name)
        assert spec.name == name
        assert callable(spec.alloc_scratch)
        assert callable(spec.generate)
        assert callable(spec.max_gates)
        assert spec.supported_orderings == frozenset({"ccw", "random_pairs"})
