from track_gen._src import generator_registry as reg


def test_bezier_is_registered():
    assert "bezier" in reg.available()
    spec = reg.get("bezier")
    assert spec.name == "bezier"
    assert callable(spec.alloc_scratch) and callable(spec.generate)


def test_unknown_generator_raises_with_available_list():
    import pytest
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "bezier" in str(e.value)  # error lists what IS available
