import dataclasses
import math

import pytest
import torch

from track_gen import GateGenConfig, PerEnvSeededRNG, TrackGenerator
from track_gen._src.types import Track, TrackGenConfig


def test_config_defaults_instantiate():
    cfg = TrackGenConfig()
    # generator + batching
    assert cfg.generator == "bezier"
    assert cfg.on_invalid == "poison"
    assert cfg.num_envs == 1
    assert cfg.device == "cpu"
    # Bezier params
    assert cfg.min_num_points == 9
    assert cfg.max_num_points == 13
    assert cfg.num_points_per_segment == 30
    assert cfg.min_point_distance == 0.05
    assert cfg.rad == 0.4
    assert cfg.handle_clamp_frac == 0.4  # kept == rad so the clamp doesn't bind every segment
    assert cfg.edgy == 0.0
    assert cfg.scale == 1.0
    # Polar / Fourier params (reconciled names)
    assert cfg.polar_num_knots == 12
    assert cfg.polar_radial_jitter == 0.60
    assert cfg.polar_angular_jitter == 0.30
    # Hull params
    assert cfg.hull_displacement == 0.15
    # Voronoi / graph-cycle params
    assert cfg.voronoi_num_sites == 256
    assert cfg.voronoi_site_layout == "void_ring"
    assert cfg.voronoi_control_points == 18
    assert cfg.voronoi_radial_variation == 0.62
    assert cfg.voronoi_angular_jitter == 0.08
    # Width params
    assert cfg.half_width == 0.1
    # Output params
    assert cfg.num_points == 256
    # constant_spacing is the only supported output mode.
    # spacing defaults to None -> __post_init__ auto-couples it to 0.6*half_width.
    assert math.isclose(cfg.spacing, 0.6 * cfg.half_width)
    assert cfg.N_max == 384
    # Robustness params
    assert cfg.turning_tol == 0.1
    assert cfg.w_floor == 1e-3


def test_config_overrides_round_trip():
    cfg = TrackGenConfig(generator="polar", num_envs=32, num_points=128,
                         half_width=0.25, spacing=0.05, N_max=512, w_floor=1e-2)
    assert dataclasses.asdict(TrackGenConfig(**dataclasses.asdict(cfg))) == dataclasses.asdict(cfg)


def test_repulsive_config_defaults():
    # Supported defaults reserve room for the complete road band.
    cfg = TrackGenConfig()
    assert cfg.repulsive_grow_mult_min == 2.0
    assert cfg.repulsive_grow_mult_max == 3.0
    assert cfg.repulsive_domain_frac == 0.35
    assert cfg.repulsive_domain_init_ratio == 4.0
    assert cfg.repulsive_obstacle_count_min == 8
    assert cfg.repulsive_obstacle_count_max == 12
    assert cfg.repulsive_ratchet_rate == 0.012
    assert cfg.repulsive_stages == (64, 128, 256)
    # A repulsive config with stages[-1] == num_points also constructs clean.
    TrackGenConfig(generator="repulsive", num_points=256, repulsive_stages=(64, 128, 256))


# ---------------------------------------------------------------------------------------
# Generator-specific rules: moved from __post_init__ to the generators' Params.validate()
# (roadmap PR8). Same messages and conditions; they now fire at TrackGenerator /
# GateGenerator construction, and only for the selected generator.
# ---------------------------------------------------------------------------------------

# (owner generator, bad flat fields, match, a generator that does not own the knob)
TRACK_RULES = [
    ("voronoi", dict(voronoi_num_sites=12, voronoi_control_points=18), "voronoi_num_sites",
     "bezier"),
    ("voronoi", dict(voronoi_site_layout="bad-layout"), "voronoi_site_layout", "bezier"),
    ("voronoi", dict(voronoi_control_points=5), "voronoi_control_points", "polar"),
    ("voronoi", dict(voronoi_radial_variation=-0.1), "voronoi_radial_variation", "bezier"),
    ("voronoi", dict(voronoi_angular_jitter=-0.1), "voronoi_angular_jitter", "bezier"),
    ("voronoi", dict(num_points_per_segment=1), "num_points_per_segment", "checkpoint"),
    ("polar", dict(num_points_per_segment=1), "num_points_per_segment", "checkpoint"),
    ("bezier", dict(min_num_points=1), "min_num_points", "polar"),
    ("bezier", dict(min_num_points=13, max_num_points=9), "max_num_points", "polar"),
    ("bezier", dict(min_point_distance=0.0), "min_point_distance", "polar"),
    ("bezier", dict(min_point_distance=0.6), "min_point_distance", "polar"),
    ("bezier", dict(num_points_per_segment=1), "num_points_per_segment", "checkpoint"),
    ("hull", dict(min_num_points=1), "min_num_points", "voronoi"),
    ("hull", dict(min_point_distance=0.6), "min_point_distance", "voronoi"),
    # grow-mult must genuinely grow: min >= 1 and max > 1 (a multiplier <= 1 zeroes the
    # growth budget -> an empty growth loop -> a mis-strided centerline).
    ("repulsive", dict(repulsive_grow_mult_min=0.9), "repulsive_grow_mult_min", "bezier"),
    ("repulsive", dict(repulsive_grow_mult_min=1.0, repulsive_grow_mult_max=1.0),
     "repulsive_grow_mult_max", "bezier"),
    # settle_iters / resample_every / stall_window >= 1 (an empty growth loop, or a
    # ZeroDivisionError at the `(it+1) % k` guards mid-generate).
    ("repulsive", dict(repulsive_settle_iters=0), "repulsive_settle_iters", "bezier"),
    ("repulsive", dict(repulsive_resample_every=0), "repulsive_resample_every", "bezier"),
    ("repulsive", dict(repulsive_stall_window=0), "repulsive_stall_window", "bezier"),
    ("repulsive", dict(repulsive_grow_mult_min=5.5, repulsive_grow_mult_max=4.5),
     "repulsive_grow_mult", "bezier"),
    ("repulsive", dict(repulsive_obstacle_radius_min_frac=0.05,
                       repulsive_obstacle_radius_max_frac=0.02),
     "repulsive_obstacle_radius", "bezier"),
    ("repulsive", dict(repulsive_obstacle_count_min=0), "repulsive_obstacle_count_min",
     "bezier"),
    ("repulsive", dict(repulsive_obstacle_count_min=12, repulsive_obstacle_count_max=8),
     "repulsive_obstacle_count", "bezier"),
    ("repulsive", dict(repulsive_domain_init_ratio=1.0), "repulsive_domain_init_ratio",
     "bezier"),
    ("repulsive", dict(repulsive_domain_frac=0.0), "repulsive_domain_frac", "bezier"),
    ("repulsive", dict(repulsive_ratchet_rate=0.0), "repulsive_ratchet_rate", "bezier"),
    ("repulsive", dict(repulsive_alpha=0.0), "repulsive_alpha", "bezier"),
    ("repulsive", dict(repulsive_beta=-1.0), "repulsive_beta", "bezier"),
    ("repulsive", dict(repulsive_stages=(128, 128, 256)), "repulsive_stages", "bezier"),
    ("repulsive", dict(repulsive_stages=(64, 130, 256)), "repulsive_stages", "bezier"),
    # For generator="repulsive", stages[-1] must equal num_points (a mismatched last
    # stage is allowed for every other generator).
    ("repulsive", dict(num_points=256, repulsive_stages=(64, 128, 200)),
     r"repulsive_stages\[-1\] must equal num_points", "bezier"),
    ("checkpoint", dict(checkpoint_count=2), "checkpoint_count", "bezier"),
    ("checkpoint", dict(checkpoint_best_of_k=0), "checkpoint_best_of_k", "bezier"),
    ("checkpoint", dict(checkpoint_radius_min_frac=1.0), "checkpoint_radius_min_frac",
     "bezier"),
]

GATE_RULES = [
    ("bezier", dict(min_point_distance=0.0), "min_point_distance", "polar"),
    ("hull", dict(min_num_points=1), "min_num_points", "polar"),
    ("bezier", dict(min_num_points=13, max_num_points=9), "max_num_points", "polar"),
    ("voronoi", dict(voronoi_control_points=2), "voronoi_control_points", "bezier"),
    ("voronoi", dict(voronoi_num_sites=4, voronoi_control_points=8), "voronoi_num_sites",
     "bezier"),
    ("voronoi", dict(voronoi_radial_variation=-0.1), "voronoi_radial_variation", "bezier"),
    ("voronoi", dict(voronoi_angular_jitter=-0.1), "voronoi_angular_jitter", "bezier"),
    ("voronoi", dict(voronoi_site_layout="spiral"), "voronoi_site_layout", "bezier"),
    ("checkpoint", dict(checkpoint_count=2), "checkpoint_count", "bezier"),
    ("checkpoint", dict(checkpoint_radius_min_frac=-0.1), "checkpoint_radius_min_frac",
     "bezier"),
]


def _rng(n=1):
    from track_gen import PerEnvSeededRNG

    return PerEnvSeededRNG(seeds=0, num_envs=n, device="cpu")


@pytest.mark.parametrize("owner,bad,match,other", TRACK_RULES,
                         ids=[f"{r[0]}-{r[2]}-{i}" for i, r in enumerate(TRACK_RULES)])
def test_track_generator_rules_fire_at_construction_for_the_owner(owner, bad, match, other):
    from track_gen import TrackGenerator

    # The bare config no longer validates generator knobs ...
    cfg = TrackGenConfig(generator=owner, num_envs=1, device="cpu", **bad)
    # ... the owning generator raises the legacy message at construction ...
    with pytest.raises(ValueError, match=match):
        TrackGenerator(cfg, _rng())
    # ... and so does the dict form of generator_params (same Params.validate()).
    flat = {k: v for k, v in bad.items() if k not in ("num_points",)}
    core = {k: v for k, v in bad.items() if k in ("num_points",)}
    with pytest.raises(ValueError, match=match):
        TrackGenerator(TrackGenConfig(generator=owner, num_envs=1, device="cpu",
                                      generator_params=flat, **core), _rng())


@pytest.mark.parametrize("owner,bad,match,other", TRACK_RULES[:3] + TRACK_RULES[-3:],
                         ids=lambda v: str(v) if isinstance(v, str) else None)
def test_track_rules_ignore_a_knob_of_a_generator_that_is_not_selected(owner, bad, match,
                                                                        other):
    from track_gen import TrackGenerator

    gen = TrackGenerator(TrackGenConfig(generator=other, num_envs=1, device="cpu", **bad),
                         _rng())
    assert gen.config.generator == other


@pytest.mark.parametrize("owner,bad,match,other", GATE_RULES,
                         ids=[f"{r[0]}-{r[2]}-{i}" for i, r in enumerate(GATE_RULES)])
def test_gate_generator_rules_fire_at_construction_for_the_owner(owner, bad, match, other):
    from track_gen import GateGenerator
    from track_gen._src.types import GateGenConfig

    cfg = GateGenConfig(generator=owner, num_envs=1, device="cpu", **bad)
    with pytest.raises(ValueError, match=match):
        GateGenerator(cfg, _rng())
    # A generator that does not own the knob accepts (and ignores) it.
    GateGenerator(GateGenConfig(generator=other, num_envs=1, device="cpu", **bad), _rng())


def test_repulsive_stage_rule_still_only_binds_the_repulsive_generator():
    from track_gen import TrackGenerator

    # A non-repulsive generator accepts a mismatched last stage.
    TrackGenerator(TrackGenConfig(generator="bezier", num_envs=1, num_points=256,
                                  repulsive_stages=(64, 128, 200)), _rng())


def test_track_construct_from_tensors_field_shapes():
    import warp as wp
    from track_gen.results import Layout
    wp.init()
    E, N = 4, 16
    track = Track(
        outer=wp.zeros(E * N, dtype=wp.vec3f, device="cpu"),
        center=wp.zeros(E * N, dtype=wp.vec3f, device="cpu"),
        inner=wp.zeros(E * N, dtype=wp.vec3f, device="cpu"),
        tangent=wp.zeros(E * N, dtype=wp.vec3f, device="cpu"),
        normal=wp.zeros(E * N, dtype=wp.vec3f, device="cpu"),
        arclen=wp.zeros(E * N, dtype=wp.float32, device="cpu"),
        length=wp.zeros(E, dtype=wp.float32, device="cpu"),
        valid=wp.zeros(E, dtype=wp.int32, device="cpu"),
        count=wp.zeros(E, dtype=wp.int32, device="cpu"),
        winding=wp.zeros(E, dtype=wp.float32, device="cpu"),
        truncated=wp.zeros(E, dtype=wp.int32, device="cpu"),
        generation=wp.zeros(E, dtype=wp.int32, device="cpu"),
        fail_mask=wp.zeros(E, dtype=wp.int32, device="cpu"),
        half_width=wp.zeros(E * N, dtype=wp.float32, device="cpu"),
        layout=Layout(E, N, "cpu"),
    )
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (E * N,)
        assert arr.dtype == wp.vec3f
    assert track.arclen.shape == (E * N,)
    assert track.arclen.dtype == wp.float32
    assert track.length.shape == (E,)
    assert track.length.dtype == wp.float32
    assert track.valid.shape == (E,)
    assert track.count.shape == (E,)


def test_relaxation_defaults():
    from track_gen._src.types import TrackGenConfig
    cfg = TrackGenConfig()
    assert cfg.relax_enable is True
    assert cfg.relax_solver == "xpbd"
    assert cfg.relax_bend_relax == 1.5
    assert cfg.relax_margin == 0.15


def test_deprecated_width_clamp_fields_removed():
    from track_gen._src.types import TrackGenConfig
    cfg = TrackGenConfig()
    for dead in ("alpha", "clamp_self_distance", "self_distance_margin",
                 "self_distance_band", "self_distance_decimation"):
        assert not hasattr(cfg, dead), f"{dead} should be removed"


def test_types_module_has_no_intra_package_imports():
    # types.py must not import the heavier _src siblings (which would create import
    # cycles), so the public dataclasses stay cheap. Warp is a core dep (Track fields
    # are wp.array), so "import warp" is explicitly allowed.
    import track_gen._src.types as t

    src = open(t.__file__).read()
    for forbidden in ("from .track_generator", "from .warp_pipeline",
                      "from .warp_relax", "from .rng_utils", "from .rng_kernels"):
        assert forbidden not in src, f"types.py must not contain '{forbidden}'"


def test_gate_config_defaults_instantiate():
    from track_gen._src.types import GateGenConfig

    cfg = GateGenConfig()
    assert cfg.generator == "bezier"
    assert cfg.on_invalid == "poison"
    assert cfg.device == "cpu"
    assert cfg.num_envs == 1
    assert cfg.min_gates == 4
    assert cfg.max_gates == 32
    assert cfg.gate_radius == 0.025
    assert cfg.gate_solve_iters == 8
    assert cfg.gate_width == 0.0
    assert cfg.gate_ordering is None        # the generator's default order (PR11)
    assert cfg.max_num_points == 13
    assert cfg.polar_num_knots == 12
    assert cfg.voronoi_control_points == 18
    assert cfg.checkpoint_count == 12


def test_gate_config_validates_basic_bounds():
    from track_gen._src.types import GateGenConfig

    with pytest.raises(ValueError, match="min_gates"):
        GateGenConfig(min_gates=1)
    with pytest.raises(ValueError, match="max_gates"):
        GateGenConfig(min_gates=6, max_gates=5)
    with pytest.raises(ValueError, match="gate_radius"):
        GateGenConfig(gate_radius=-1.0)
    with pytest.raises(ValueError, match="gate_solve_iters"):
        GateGenConfig(gate_solve_iters=-1)
    with pytest.raises(ValueError, match="gate_width"):
        GateGenConfig(gate_width=-1.0)
    # PR11: gate_ordering names any registered order stage (or None: the generator's
    # default order): the config checks its type only; an unknown name is refused by
    # GateGenerator, through the registry.
    for bad in ("", 3):
        with pytest.raises(ValueError, match="gate_ordering must be a non-empty string"):
            GateGenConfig(gate_ordering=bad)
    assert GateGenConfig(gate_ordering=None).gate_ordering is None
    from track_gen import GateGenerator

    with pytest.raises(ValueError, match=r"GateGenConfig.gate_ordering: unknown order "
                                         r"'spiral'; available: \['ccw', 'random_pairs', "
                                         r"'raw'\]"):
        GateGenerator(GateGenConfig(gate_ordering="spiral"))
    # The voronoi_* rules moved to the gate generator (GATE_RULES): config time is clean.
    GateGenConfig(voronoi_radial_variation=-0.1)
    GateGenConfig(voronoi_angular_jitter=-0.1)


def test_gate_config_validates_num_envs_and_point_family_inputs():
    from track_gen._src.types import GateGenConfig

    with pytest.raises(ValueError, match="num_envs"):
        GateGenConfig(num_envs=0)
    # min_point_distance feeds a 1/(d*2) cell count in the shared corner sampler; a
    # non-positive value must fail at generator construction (GATE_RULES), not as a
    # downstream ZeroDivisionError. The bare config no longer checks it.
    GateGenConfig(min_point_distance=0.0)
    GateGenConfig(min_num_points=1)
    GateGenConfig(min_num_points=13, max_num_points=9)


def test_track_config_z_defaults_instantiate():
    cfg = TrackGenConfig(device="cpu", num_envs=1)
    assert cfg.z_profile == "flat"
    assert cfg.z_base == 0.0
    assert cfg.z_min == 0.0
    assert cfg.z_max == 0.0
    assert cfg.z_max_step == 0.0
    assert cfg.z_noise_amplitude == 0.0
    assert cfg.z_noise_harmonics == 3
    assert cfg.z_valid_grade == 0.0


def test_track_config_z_validates():
    # z_profile names any registered elevation stage (roadmap PR10b): the config checks
    # its type only; an unknown name is refused by TrackGenerator, through the registry.
    cfg = TrackGenConfig(device="cpu", num_envs=1, z_profile="bogus")
    with pytest.raises(ValueError, match=r"TrackGenConfig.z_profile: unknown elevation "
                                         r"'bogus'; available: \['flat', 'noise', "
                                         r"'random_walk', 'uniform'\]"):
        TrackGenerator(cfg, PerEnvSeededRNG(seeds=1, num_envs=1, device="cpu"))
    for bad in ("", None, 3):
        with pytest.raises(ValueError, match="z_profile must be a non-empty string"):
            TrackGenConfig(device="cpu", num_envs=1, z_profile=bad)
    with pytest.raises(ValueError, match="z_min"):
        TrackGenConfig(device="cpu", num_envs=1, z_min=2.0, z_max=1.0)
    # PR11: a gate z_profile names any registered elevation stage too (same rule).
    from track_gen import GateGenerator

    gcfg = GateGenConfig(device="cpu", num_envs=1, z_profile="bogus")
    with pytest.raises(ValueError, match=r"GateGenConfig.z_profile: unknown elevation "
                                         r"'bogus'; available: \['flat', 'noise', "
                                         r"'random_walk', 'uniform'\]"):
        GateGenerator(gcfg)
    for bad in ("", None, 3):
        with pytest.raises(ValueError, match="z_profile must be a non-empty string"):
            GateGenConfig(device="cpu", num_envs=1, z_profile=bad)


def test_track_config_validates_sampler_and_buffer_invariants():
    from track_gen._src.types import TrackGenConfig
    import pytest
    with pytest.raises(ValueError, match="num_envs"):
        TrackGenConfig(num_envs=0)
    with pytest.raises(ValueError, match="half_width"):
        TrackGenConfig(half_width=0.0)
    # The point-family / num_points_per_segment rules moved to the generators
    # (TRACK_RULES): the bare config no longer checks them.
    TrackGenConfig(min_num_points=1)
    TrackGenConfig(min_point_distance=0.6)
    TrackGenConfig(num_points_per_segment=1)
    with pytest.raises(ValueError, match="spacing"):
        TrackGenConfig(spacing=0.0)


def test_track_config_defaults_still_valid():
    from track_gen._src.types import TrackGenConfig
    cfg = TrackGenConfig()
    assert cfg.num_points <= cfg.N_max
    assert cfg.spacing == 0.6 * cfg.half_width


def test_gate_sequence_construct_from_warp_arrays_and_clone():
    import warp as wp
    from track_gen.results import Layout
    from track_gen._src.types import GateSequence

    wp.init()
    E, G = 2, 8
    gates = GateSequence(
        position=wp.zeros(E * G, dtype=wp.vec3f, device="cpu"),
        tangent=wp.zeros(E * G, dtype=wp.vec3f, device="cpu"),
        forward=wp.zeros(E * G, dtype=wp.vec3f, device="cpu"),
        orientation=wp.zeros(E * G, dtype=wp.quatf, device="cpu"),
        half_size=wp.zeros(E * G, dtype=wp.float32, device="cpu"),
        left=wp.zeros(E * G, dtype=wp.vec3f, device="cpu"),
        right=wp.zeros(E * G, dtype=wp.vec3f, device="cpu"),
        valid=wp.zeros(E, dtype=wp.int32, device="cpu"),
        count=wp.zeros(E, dtype=wp.int32, device="cpu"),
        generation=wp.zeros(E, dtype=wp.int32, device="cpu"),
        fail_mask=wp.zeros(E, dtype=wp.int32, device="cpu"),
        layout=Layout(E, G, "cpu"),
    )
    assert gates.position.shape == (E * G,)
    assert gates.tangent.dtype == wp.vec3f
    assert gates.forward.dtype == wp.vec3f
    assert gates.orientation.dtype == wp.quatf
    assert gates.half_size.dtype == wp.float32
    assert not hasattr(gates, "normal")
    clone = gates.clone()
    assert clone is not gates
    assert clone.position.ptr != gates.position.ptr
    assert clone.forward.ptr != gates.forward.ptr
    assert clone.orientation.ptr != gates.orientation.ptr
    assert clone.half_size.ptr != gates.half_size.ptr


# ---------------------------------------------------------------------------------------
# generator_params type check and dead-field deprecations (roadmap PR8)
# ---------------------------------------------------------------------------------------

def test_generator_params_type_is_checked_at_config_construction():
    from track_gen._src.types import GateGenConfig

    for cls in (TrackGenConfig, GateGenConfig):
        for bad in (3, "polar_num_knots=9", [("polar_num_knots", 9)], TrackGenConfig):
            with pytest.raises(TypeError, match="generator_params must be a Params"):
                cls(generator_params=bad)
        assert cls(generator_params={"x": 1}).generator_params == {"x": 1}  # resolved later
        assert cls().generator_params is None
