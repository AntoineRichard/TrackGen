import pytest

from tests.conftest import override_spec
from track_gen._src import generator_registry as reg


def test_bezier_is_registered():
    # Roadmap PR13c: the built-in track generators are generator Stages (like repulsive
    # since PR12); the frozen legacy entry points stay importable.
    from track_gen._src import builtin_params, warp_generate
    from track_gen.plugin.stage import _is_stage

    assert "bezier" in reg.available()
    spec = reg.get("bezier")
    assert _is_stage(spec) and spec.name == "bezier" and spec.kind == "generator"
    assert spec.Params is builtin_params.BezierParams
    assert callable(warp_generate.bezier_alloc_scratch)
    assert callable(warp_generate.generate_centerline_warp)


def test_generatorspec_has_capturable_default_true():
    # A three-arg GeneratorSpec (the existing registration shape) defaults capturable=True.
    spec = reg.GeneratorSpec(name="x", alloc_scratch=lambda c: None, generate=lambda *a: None)
    assert spec.capturable is True


def test_existing_generators_are_capturable():
    from track_gen.plugin import registry

    # repulsive is a BOUNDED Stage (roadmap PR12): capturable, reported by describe().
    for name in ("bezier", "polar", "hull", "voronoi", "checkpoint", "repulsive"):
        assert registry.describe("generator", name).capturable is True, name


def test_unknown_generator_raises_with_available_list():
    import pytest
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "bezier" in str(e.value)  # error lists what IS available


def test_composite_scratch_has_no_flat_name_fallthrough():
    """Sub-group buffers are reached only through their group: a flat name never
    resolves into .gen/.relax/.inflate (a generator field could otherwise shadow
    the pipeline's buffer of the same name)."""
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

    assert scratch.gen.gen_private == "gen_private"
    assert scratch.inflate.kappa == "kappa"
    with pytest.raises(AttributeError):
        scratch.gen_private
    with pytest.raises(AttributeError):
        scratch.kappa


def test_generator_scratch_names_do_not_shadow_inflate_buffers(plugin_registry):
    """A generator scratch exposing fields named like the inflate buffers (w, kappa,
    area_a, area_b, z) must not be read or written by the pipeline tail."""
    import dataclasses
    from types import SimpleNamespace

    import numpy as np
    import warp as wp

    from track_gen._src import warp_generate
    from track_gen._src.rng_utils import PerEnvSeededRNG
    from track_gen._src.track_generator import TrackGenerator
    from track_gen._src.types import TrackGenConfig

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    flat = E * int(cfg.N_max)
    # The legacy bezier spec over the frozen public functions (the registered "bezier"
    # is a Stage since roadmap PR13c).
    base = reg.GeneratorSpec(name="bezier", alloc_scratch=warp_generate.bezier_alloc_scratch,
                             generate=warp_generate.generate_centerline_warp,
                             accepts_ctx=True)
    sentinel = -7.0
    decoys = {}

    def alloc_with_decoys(config):
        gs = base.alloc_scratch(config)
        fields = {k: getattr(gs, k) for k in warp_generate.GenScratch.__slots__}
        for name, n in (("w", flat), ("kappa", flat), ("z", flat), ("area_a", E), ("area_b", E)):
            decoys[name] = wp.full(n, sentinel, dtype=wp.float32, device="cpu")
            fields[name] = decoys[name]
        return SimpleNamespace(**fields)

    decoy_spec = dataclasses.replace(base, alloc_scratch=alloc_with_decoys)
    ref = TrackGenerator(cfg, PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")).generate()
    ref_center = ref.center.numpy().copy()
    ref_valid = ref.valid.numpy().copy()

    override_spec(plugin_registry, decoy_spec)
    got = TrackGenerator(cfg, PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")).generate()

    assert set(decoys) == {"w", "kappa", "z", "area_a", "area_b"}, "decoy spec unused"
    for name, arr in decoys.items():
        assert (arr.numpy() == sentinel).all(), f"pipeline wrote generator field {name!r}"
    np.testing.assert_array_equal(got.center.numpy(), ref_center)
    np.testing.assert_array_equal(got.valid.numpy(), ref_valid)


def test_track_generator_uses_constructor_resolved_generator(plugin_registry):
    from track_gen._src.rng_utils import PerEnvSeededRNG
    from track_gen._src.track_generator import TrackGenerator
    from track_gen._src.types import TrackGenConfig

    from track_gen._src import warp_generate

    cfg = TrackGenConfig(num_envs=1, device="cpu")
    rng = PerEnvSeededRNG(seeds=0, num_envs=1, device="cpu")
    gen = TrackGenerator(cfg, rng)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing TrackGenerator used a later registry entry")

    override_spec(plugin_registry, reg.GeneratorSpec(
        name="bezier",
        alloc_scratch=warp_generate.bezier_alloc_scratch,
        generate=fail_if_called,
    ))
    gen.generate(1)
