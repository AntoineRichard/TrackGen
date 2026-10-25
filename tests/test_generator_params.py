"""Built-in generator Params: the flat-field mapping (``from_legacy`` /
``_apply_legacy``), byte-identity of the three ways to configure a built-in, config
conflicts and round trips, gate capacity, legacy specs and the built-in schemas
(roadmap PR8). CPU only."""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from tests._plugin_toys import (
    TOY_CIRCLE,
    TOY_LOBED,
    ToyLobedParams,
    ToyWobbleParams,
)
from track_gen import (
    GateGenConfig,
    GateGenerator,
    PerEnvSeededRNG,
    TrackGenConfig,
    TrackGenerator,
)
from track_gen._src import builtin_params as bp
from track_gen._src.generator_binding import bind_generator
from track_gen.plugin import GeneratorSpec, registry

# Roadmap PR13d: bezier's style_sampling is deprecated (one DeprecationWarning per built
# generator); this module still exercises it.
pytestmark = pytest.mark.filterwarnings(
    "ignore:bezier's style_sampling is deprecated:DeprecationWarning")

E = 4
DEV = "cpu"
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                "valid", "count", "winding", "truncated", "generation")
GATE_FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left",
               "right", "valid", "count", "generation")

TRACK_PARAMS = {
    "bezier": bp.BezierParams, "hull": bp.HullParams, "polar": bp.PolarParams,
    "voronoi": bp.VoronoiParams, "checkpoint": bp.CheckpointParams,
    "repulsive": bp.RepulsiveParams,
}
GATE_PARAMS = {
    "bezier": bp.PointGateParams, "hull": bp.PointGateParams, "polar": bp.PolarGateParams,
    "voronoi": bp.VoronoiGateParams, "checkpoint": bp.CheckpointGateParams,
}
ALL_SPECS = [("generator", n, P) for n, P in TRACK_PARAMS.items()] + \
            [("gate_generator", n, P) for n, P in GATE_PARAMS.items()]
# Roadmap PR13c: the dynamic knobs of each built-in (every other field is static).
# Roadmap PR13d: + size (every built-in track generator, the bezier / hull gate
# generators) and bezier / hull's min_num_points (int32).
DYNAMIC = {
    ("generator", "bezier"): {"min_num_points", "rad", "handle_clamp_frac", "size"},
    ("generator", "hull"): {"min_num_points", "hull_displacement", "size"},
    ("generator", "polar"): {"polar_radial_jitter", "polar_angular_jitter", "size"},
    ("generator", "voronoi"): {"voronoi_radial_variation", "voronoi_angular_jitter",
                               "size"},
    ("generator", "checkpoint"): {"checkpoint_radius_min_frac", "checkpoint_angle_jitter",
                                  "checkpoint_turn_rate", "size"},
    ("generator", "repulsive"): {"size"},
    ("gate_generator", "bezier"): {"size"},
    ("gate_generator", "hull"): {"size"},
    ("gate_generator", "polar"): {"polar_radial_jitter", "polar_angular_jitter"},
    ("gate_generator", "voronoi"): {"voronoi_radial_variation", "voronoi_angular_jitter"},
    ("gate_generator", "checkpoint"): {"checkpoint_radius_min_frac",
                                       "checkpoint_angle_jitter"},
}


def _rng(n=E, seed=5):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _other(v):
    """A value different from ``v`` of the same shape (from_legacy does not validate)."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 0.01
    if isinstance(v, str):
        return "mixed" if v != "mixed" else "ring"
    if isinstance(v, tuple):
        return tuple(x * 2 for x in v)
    if v is None:
        return (0.1, 0.2)
    raise AssertionError(v)


# ---------------------------------------------------------------------------------------
# The flat-field mapping, for every built-in
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,name,P", ALL_SPECS, ids=[f"{k}-{n}" for k, n, _ in ALL_SPECS])
def test_builtin_params_class_maps_the_flat_fields(kind, name, P):
    from track_gen.plugin._registry import _params_class

    # The track built-ins are generator Stages since PR13c, the gate built-ins
    # gate_generator Stages since PR11 (Params=P).
    assert _params_class(registry.get(kind, name)) is P
    Config = TrackGenConfig if kind == "generator" else GateGenConfig
    defaults = Config()
    dynamic = DYNAMIC.get((kind, name), set())
    # Defaults equal the config defaults; every field is static but the PR13c knobs.
    for f in dataclasses.fields(P):
        assert getattr(P(), f.name) == getattr(defaults, f.name), f.name
        assert f.metadata["tg"] == ("dynamic" if f.name in dynamic else "static"), f.name
    assert P.from_legacy(defaults) == P()
    # from_legacy round-trips non-default flat values (raw, no casts) ...
    changed = {f.name: _other(getattr(defaults, f.name)) for f in dataclasses.fields(P)}
    cfg = Config(**changed)
    params = P.from_legacy(cfg)
    for k, v in changed.items():
        assert getattr(params, k) == v and type(getattr(params, k)) is type(v), k
    # ... and _apply_legacy writes them back onto a config.
    target = Config()
    params._apply_legacy(target)
    for k, v in changed.items():
        assert getattr(target, k) == v, k
    # The schema is JSON: the PR13c / PR13d knobs dynamic (float32; min_num_points
    # int32), every other field static.
    schema = registry.describe(kind, name).params_schema
    assert list(schema["properties"]) == [f.name for f in dataclasses.fields(P)]
    for field, prop in schema["properties"].items():
        if field in dynamic:
            dtype = "int32" if field == "min_num_points" else "float32"
            assert prop.get("x-dynamic") is True and prop["x-dtype"] == dtype, field
        else:
            assert prop.get("x-static") is True, field
    json.dumps(schema)


def test_ownership_partitions_the_generator_fields():
    # The core fields are no generator's Params; min_angle (dead) is in none.
    for P in list(TRACK_PARAMS.values()) + list(GATE_PARAMS.values()):
        names = {f.name for f in dataclasses.fields(P)}
        assert not names & {"scale", "num_points", "num_envs", "device", "generator",
                            "min_angle", "generator_params"}


# ---------------------------------------------------------------------------------------
# Byte identity: flat knobs == dict generator_params == P.from_legacy(flat) instance
# ---------------------------------------------------------------------------------------

TRACK_CASES = {
    "bezier": ({}, dict(min_num_points=7, max_num_points=11, rad=0.3, edgy=0.2)),
    "bezier_style": ({}, dict(style_sampling=True, rad_range=(0.1, 0.35),
                              scale_range=(0.8, 1.2), handle_clamp_frac_range=(0.3, 0.9))),
    "bezier_fallback": ({}, dict(rad=0.49, min_num_points=10)),
    "hull": ({}, dict(hull_displacement=0.25, min_num_points=8)),
    "polar": ({}, dict(polar_num_knots=9, polar_radial_jitter=0.4)),
    "voronoi": ({}, dict(voronoi_control_points=14, voronoi_site_layout="mixed")),
    "checkpoint": ({}, dict(checkpoint_count=10, checkpoint_best_of_k=2)),
    "repulsive": (dict(num_points=64, N_max=384, half_width=0.1),
                  dict(repulsive_stages=(16, 32, 64))),
}


def _track(cfg, n, crossers=False):
    tg = TrackGenerator(cfg, _rng(n))
    track = tg.generate()
    if crossers:            # some env took the polygon fallback (as in the PR7 fingerprints)
        # The built-in Stage's private scratch (roadmap PR13c: "scratch:generator/<slot>")
        scratch = tg._scratch.gen.scratch
        c = scratch.get("scratch:generator/gen_crossers",
                        scratch.get("scratch:generator/crossers"))
        assert (c.numpy() > 0).any()
    return {f: getattr(track, f).numpy().tobytes() for f in TRACK_FIELDS}


@pytest.mark.parametrize("case", sorted(TRACK_CASES))
def test_three_ways_to_configure_a_builtin_are_byte_identical(case):
    gen = case.split("_")[0]
    core, knobs = TRACK_CASES[case]
    n = 2 if gen == "repulsive" else 32 if case.endswith("_fallback") else E
    flat = TrackGenConfig(generator=gen, device=DEV, num_envs=n, **core, **knobs)
    ref = _track(flat, n, crossers=case.endswith("_fallback"))
    as_dict = TrackGenConfig(generator=gen, device=DEV, num_envs=n, **core,
                             generator_params=dict(knobs))
    assert _track(as_dict, n) == ref
    inst = TRACK_PARAMS[gen].from_legacy(flat)
    as_inst = TrackGenConfig(generator=gen, device=DEV, num_envs=n, **core,
                             generator_params=inst)
    assert _track(as_inst, n) == ref
    # And the knobs matter (the comparison is not vacuous).
    if gen != "repulsive":
        assert _track(TrackGenConfig(generator=gen, device=DEV, num_envs=n, **core), n) \
            != ref


GATE_CASES = {
    "bezier": dict(min_num_points=6, max_num_points=9),
    "hull": dict(min_point_distance=0.08),
    "polar": dict(polar_num_knots=9, polar_angular_jitter=0.2),
    "voronoi": dict(voronoi_control_points=10, voronoi_site_layout="ring"),
    "checkpoint": dict(checkpoint_count=8, checkpoint_angle_jitter=0.3),
}


def _gates(cfg):
    seq = GateGenerator(cfg, _rng()).generate()
    return {f: getattr(seq, f).numpy().tobytes() for f in GATE_FIELDS}


@pytest.mark.parametrize("gen", sorted(GATE_CASES))
def test_three_ways_to_configure_a_gate_builtin_are_byte_identical(gen):
    knobs = GATE_CASES[gen]
    base = dict(generator=gen, device=DEV, num_envs=E, gate_width=0.05)
    flat = GateGenConfig(**base, **knobs)
    ref = _gates(flat)
    assert _gates(GateGenConfig(**base, generator_params=dict(knobs))) == ref
    inst = GATE_PARAMS[gen].from_legacy(flat)
    assert _gates(GateGenConfig(**base, generator_params=inst)) == ref
    assert _gates(GateGenConfig(**base)) != ref


def test_gate_capacity_sees_generator_params():
    cfg = GateGenConfig(generator="polar", device=DEV, num_envs=E, gate_ordering="raw",
                        generator_params={"polar_num_knots": 9})
    gen = GateGenerator(cfg, _rng())
    assert gen.generate().count.numpy().tolist() == [9] * E
    assert gen.params["generator"].params == bp.PolarGateParams(polar_num_knots=9)
    assert gen.config.generator_params == bp.PolarGateParams(polar_num_knots=9)
    assert gen._config.polar_num_knots == 9            # the run snapshot is overlaid
    assert gen.config.polar_num_knots == 12            # the user config is as passed
    with pytest.raises(ValueError, match="max_gates=8 is too small"):
        GateGenerator(GateGenConfig(generator="voronoi", device=DEV, num_envs=E,
                                    max_gates=8, generator_params={
                                        "voronoi_control_points": 10}), _rng())


# ---------------------------------------------------------------------------------------
# Config semantics
# ---------------------------------------------------------------------------------------

def test_flat_and_generator_params_conflict():
    with pytest.raises(ValueError, match=r"fields \['polar_num_knots'\] both configure "
                                         r"generator 'polar' with different values"):
        TrackGenerator(TrackGenConfig(generator="polar", num_envs=E, polar_num_knots=9,
                                      generator_params={"polar_num_knots": 10}), _rng())
    # Equal values, a flat field at its default, and another generator's knob are fine.
    TrackGenerator(TrackGenConfig(generator="polar", num_envs=E, polar_num_knots=9,
                                  generator_params={"polar_num_knots": 9}), _rng())
    TrackGenerator(TrackGenConfig(generator="polar", num_envs=E, voronoi_control_points=7,
                                  generator_params={"polar_num_knots": 9}), _rng())
    # A list flat value equals a tuple param value.
    cfg = TrackGenConfig(generator="repulsive", num_envs=1, num_points=64,
                         repulsive_stages=[16, 32, 64],
                         generator_params={"repulsive_stages": (16, 32, 64)})
    binding = bind_generator("generator", cfg, registry=registry)
    assert binding.run_config.repulsive_stages == (16, 32, 64)


def test_a_config_subclass_default_conflicts_with_partial_generator_params():
    @dataclasses.dataclass
    class MyCfg(TrackGenConfig):
        polar_num_knots: int = 9

    with pytest.raises(ValueError, match=r"fields \['polar_num_knots'\] both configure"):
        TrackGenerator(MyCfg(generator="polar", num_envs=E,
                             generator_params={"polar_radial_jitter": 0.5}), _rng())
    # Alone, the subclass default is the flat value; equal values do not conflict.
    binding = bind_generator("generator", MyCfg(generator="polar", num_envs=E),
                             registry=registry)
    assert binding.params.polar_num_knots == 9
    bind_generator("generator", MyCfg(generator="polar", num_envs=E,
                                      generator_params={"polar_num_knots": 9}),
                   registry=registry)


def test_a_json_mapping_resolves_like_the_instance():
    cfg = TrackGenConfig(generator="repulsive", num_envs=1, num_points=64,
                         generator_params={"repulsive_stages": [16, 32, 64]})
    binding = bind_generator("generator", cfg, registry=registry)
    assert binding.params.repulsive_stages == (16, 32, 64)
    hash(binding.params)


def test_generator_params_is_the_complete_parameter_set():
    cfg = TrackGenConfig(generator="voronoi", num_envs=E,
                         generator_params={"voronoi_control_points": 12})
    binding = bind_generator("generator", cfg, registry=registry)
    assert binding.params == bp.VoronoiParams(voronoi_control_points=12)
    assert binding.run_config.voronoi_control_points == 12
    assert binding.run_config.voronoi_num_sites == 256          # left-out keys: defaults
    assert binding.user_config.voronoi_control_points == 18     # as passed
    assert binding.user_config.generator_params == binding.params


def test_switching_generators_requires_resetting_generator_params():
    cfg = TrackGenConfig(generator="bezier", num_envs=E,
                         generator_params=bp.BezierParams(rad=0.3))
    with pytest.raises(TypeError, match="must be a PolarParams"):
        TrackGenerator(dataclasses.replace(cfg, generator="polar"), _rng())
    TrackGenerator(dataclasses.replace(cfg, generator="polar", generator_params=None),
                   _rng())


def test_unknown_generator_error_comes_before_params():
    with pytest.raises(ValueError, match="unknown generator 'nope'"):
        TrackGenerator(TrackGenConfig(generator="nope", num_envs=E,
                                      generator_params={"x": 1}), _rng())


def test_config_round_trips_and_params_are_resolved():
    cfg = TrackGenConfig(generator="checkpoint", num_envs=E,
                         generator_params={"checkpoint_count": 9})
    gen = TrackGenerator(cfg, _rng())
    assert cfg.generator_params == {"checkpoint_count": 9}          # caller's object kept
    assert gen.config.generator_params == bp.CheckpointParams(checkpoint_count=9)
    again = TrackGenerator(gen.config, _rng())
    a, b = gen.generate(), again.generate()
    for f in TRACK_FIELDS:
        assert getattr(a, f).numpy().tobytes() == getattr(b, f).numpy().tobytes(), f
    # Flat configs: the reported Params are from_legacy of the flat fields; the hull
    # displacement is dynamic since roadmap PR13c (latched; its live array holds the
    # flat value and set() writes it); since roadmap PR13d min_num_points (int32) and
    # size too, while max_num_points (the corner capacity) stays static.
    flat = TrackGenConfig(generator="hull", num_envs=E, hull_displacement=0.3)
    gen = TrackGenerator(flat, _rng())
    assert gen.params["generator"].params == bp.HullParams.from_legacy(flat)
    assert gen.config == flat                                         # generator_params None
    assert gen.config.generator_params is None
    assert list(gen.params["generator"]) == ["min_num_points", "hull_displacement", "size"]
    live = gen.params["generator"]["hull_displacement"]
    assert live.numpy().tolist() == [np.float32(0.3)] * E
    gen.params["generator"].set("hull_displacement", 0.2)
    assert live.numpy().tolist() == [np.float32(0.2)] * E
    gen.params["generator"].set("min_num_points", 5)
    assert gen.params["generator"]["min_num_points"].numpy().tolist() == [5] * E
    with pytest.raises(KeyError, match="static"):
        gen.params["generator"].set("max_num_points", 12)
    assert gen._latch is not None


# ---------------------------------------------------------------------------------------
# Legacy GeneratorSpec plugins
# ---------------------------------------------------------------------------------------

def test_legacy_spec_without_params():
    registry.add(TOY_CIRCLE)
    gen = TrackGenerator(TrackGenConfig(generator="toy.circle", num_envs=E), _rng())
    assert gen.generate().valid.numpy().tolist() == [1] * E
    assert gen.params["generator"].params is None
    assert gen._config.generator_params is None
    assert registry.describe("generator", "toy.circle").params_schema == {}
    with pytest.raises(ValueError, match="generator 'toy.circle' takes no generator_params"):
        TrackGenerator(TrackGenConfig(generator="toy.circle", num_envs=E,
                                      generator_params={"lobes": 3}), _rng())
    TrackGenerator(TrackGenConfig(generator="toy.circle", num_envs=E, generator_params={}),
                   _rng())


def test_legacy_spec_with_static_params_reads_config_generator_params():
    registry.add(TOY_LOBED)
    schema = registry.describe("generator", "toy.lobed").params_schema
    assert schema["properties"]["lobes"]["minimum"] == 2
    gen = TrackGenerator(TrackGenConfig(generator="toy.lobed", num_envs=E), _rng())
    assert gen._config.generator_params == ToyLobedParams()
    a = gen.generate().center.numpy().copy()
    gen3 = TrackGenerator(TrackGenConfig(generator="toy.lobed", num_envs=E,
                                         generator_params={"lobes": 3}), _rng())
    assert gen3._config.generator_params == ToyLobedParams(lobes=3)
    assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(gen3.generate().center.numpy()))
    with pytest.raises(ValueError, match=r"ToyLobedParams.lobes must be in \[2, 12\]"):
        TrackGenerator(TrackGenConfig(generator="toy.lobed", num_envs=E,
                                      generator_params={"lobes": 1}), _rng())


def test_a_duck_typed_legacy_spec_with_an_unrelated_params_attribute_runs():
    class Duck:
        registry_kind = "generator"
        name = "acme.duck"
        capturable = True
        accepts_ctx = True
        alloc_scratch = staticmethod(TOY_CIRCLE.alloc_scratch)
        generate = staticmethod(TOY_CIRCLE.generate)

        def __init__(self):
            self.params = {"radius": 1.0}                 # not a Params class

    registry.add(Duck())
    gen = TrackGenerator(TrackGenConfig(generator="acme.duck", num_envs=E), _rng())
    assert gen.generate().valid.numpy().tolist() == [1] * E
    assert gen.params["generator"].params is None
    assert registry.describe("generator", "acme.duck").params_schema == {}


def test_legacy_spec_with_a_dynamic_field_is_rejected_at_add():
    spec = GeneratorSpec(name="acme.dyn", alloc_scratch=lambda c: None,
                         generate=lambda *a, **k: None, params=ToyWobbleParams)
    with pytest.raises(TypeError, match="dynamic fields need a Stage"):
        registry.add(spec)


def test_a_config_subclass_plugin_keeps_working():
    # The 0.x pattern: a TrackGenConfig subclass adds the plugin's knobs.
    @dataclasses.dataclass
    class AcmeConfig(TrackGenConfig):
        acme_radius: float = 0.6

    seen = []

    def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None):
        seen.append(config.acme_radius)
        TOY_CIRCLE.generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=ctx)

    registry.add(GeneratorSpec(name="acme.sub", alloc_scratch=lambda c: None,
                               generate=generate, accepts_ctx=True))
    gen = TrackGenerator(AcmeConfig(generator="acme.sub", num_envs=E, acme_radius=0.5),
                         _rng())
    assert gen.generate().valid.numpy().tolist() == [1] * E
    assert seen and set(seen) == {0.5}
    assert isinstance(gen.config, AcmeConfig)


# ---------------------------------------------------------------------------------------
# Built-in schemas
# ---------------------------------------------------------------------------------------

def test_builtin_schemas():
    props = registry.describe("generator", "voronoi").params_schema["properties"]
    assert props["voronoi_site_layout"]["enum"] == ["ring", "void_ring", "clustered", "mixed"]
    assert props["voronoi_site_layout"]["type"] == "string"
    assert props["voronoi_control_points"]["minimum"] == 6
    assert props["voronoi_control_points"]["type"] == "integer"
    dyn = DYNAMIC[("generator", "voronoi")]
    assert all(p.get("x-static") is True for k, p in props.items() if k not in dyn)
    assert (props["size"]["minimum"], props["size"]["maximum"]) == (0.1, 10.0)  # PR13d
    assert props["voronoi_angular_jitter"]["x-dynamic"] is True          # roadmap PR13c
    assert (props["voronoi_angular_jitter"]["minimum"],
            props["voronoi_angular_jitter"]["maximum"]) == (0.0, math.pi)
    gate = registry.describe("gate_generator", "voronoi").params_schema["properties"]
    assert gate["voronoi_control_points"]["minimum"] == 3
    rep = registry.describe("generator", "repulsive").params_schema["properties"]
    assert rep["repulsive_stages"]["type"] == "array"
    assert rep["repulsive_stages"]["default"] == [64, 128, 256]
    bez = registry.describe("generator", "bezier").params_schema["properties"]
    assert bez["rad_range"]["type"] == ["array", "null"]
    assert bez["min_point_distance"]["maximum"] == 0.5
    polar = registry.describe("generator", "polar").params_schema["properties"]
    # The generator's former clamp is the declared bound (roadmap PR13c).
    assert (polar["polar_radial_jitter"]["minimum"],
            polar["polar_radial_jitter"]["maximum"]) == (0.0, 0.85)
