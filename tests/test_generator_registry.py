from track_gen._src import generator_registry as reg


def test_bezier_is_registered():
    assert "bezier" in reg.available()
    spec = reg.get("bezier")
    assert spec.name == "bezier"
    assert callable(spec.alloc_scratch) and callable(spec.generate)


def test_generatorspec_has_capturable_default_true():
    # A three-arg GeneratorSpec (the existing registration shape) defaults capturable=True.
    spec = reg.GeneratorSpec(name="x", alloc_scratch=lambda c: None, generate=lambda *a: None)
    assert spec.capturable is True


def test_existing_generators_are_capturable():
    for name in ("bezier", "polar", "hull", "voronoi", "checkpoint"):
        assert reg.get(name).capturable is True


def test_unknown_generator_raises_with_available_list():
    import pytest
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "bezier" in str(e.value)  # error lists what IS available


def test_composite_scratch_fallthrough_supports_non_slotted_generator_scratch():
    from types import SimpleNamespace

    from track_gen._src.warp_pipeline import InflateScratch, _Scratch

    scratch = _Scratch(
        gen=SimpleNamespace(gen_private="gen_private"),
        inflate=InflateScratch(
            area_a="area_a",
            area_b="area_b",
            kappa="kappa",
            w="w",
        ),
    )

    assert scratch.gen_private == "gen_private"
    assert scratch.kappa == "kappa"


def test_track_generator_uses_constructor_resolved_generator():
    from track_gen._src.rng_utils import PerEnvSeededRNG
    from track_gen._src.track_generator import TrackGenerator
    from track_gen._src.types import TrackGenConfig

    cfg = TrackGenConfig(num_envs=1, device="cpu")
    rng = PerEnvSeededRNG(seeds=0, num_envs=1, device="cpu")
    gen = TrackGenerator(cfg, rng)
    original = reg.get("bezier")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing TrackGenerator used a later registry entry")

    try:
        reg.register(reg.GeneratorSpec(
            name="bezier",
            alloc_scratch=original.alloc_scratch,
            generate=fail_if_called,
        ))
        gen.generate(1)
    finally:
        reg.register(original)
