import math

import pytest
import torch

from track_gen._src.types import Track, TrackGenConfig


def test_config_defaults_instantiate():
    cfg = TrackGenConfig()
    # generator + batching
    assert cfg.generator == "bezier"
    assert cfg.num_envs == 1
    assert cfg.device == "cpu"
    # Bezier params
    assert cfg.min_num_points == 9
    assert cfg.max_num_points == 13
    assert cfg.num_points_per_segment == 30
    assert cfg.min_point_distance == 0.05
    assert math.isclose(cfg.min_angle, (12.5 / 180) * math.pi)
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
    assert cfg.num_harmonics == 5
    assert cfg.decay_p == 2
    assert cfg.amplitude == 1.0
    assert cfg.num_centerline_samples == 256
    # Width params
    assert cfg.half_width == 0.1
    # Output params
    assert cfg.num_points == 256
    # constant_spacing is the only supported output mode.
    assert cfg.output_mode == "constant_spacing"
    # spacing defaults to None -> __post_init__ auto-couples it to 0.6*half_width.
    assert math.isclose(cfg.spacing, 0.6 * cfg.half_width)
    assert cfg.N_max == 384
    # Robustness params
    assert cfg.max_regen_iters == 10
    assert cfg.turning_tol == 0.1
    assert cfg.w_floor == 1e-3


def test_config_overrides_round_trip():
    cfg = TrackGenConfig(
        generator="fourier",
        num_envs=32,
        num_points=128,
        half_width=0.25,
        output_mode="constant_spacing",
        spacing=0.05,
        N_max=512,
        decay_p=3,
        num_centerline_samples=512,
        w_floor=1e-2,
    )
    assert cfg.generator == "fourier"
    assert cfg.num_envs == 32
    assert cfg.num_points == 128
    assert cfg.output_mode == "constant_spacing"
    assert cfg.spacing == 0.05
    assert cfg.N_max == 512
    assert cfg.decay_p == 3
    assert cfg.num_centerline_samples == 512
    assert cfg.w_floor == 1e-2


def test_voronoi_static_shape_config_validation():
    with pytest.raises(ValueError, match="voronoi_num_sites"):
        TrackGenConfig(voronoi_num_sites=12, voronoi_control_points=18)
    with pytest.raises(ValueError, match="voronoi_site_layout"):
        TrackGenConfig(voronoi_site_layout="bad-layout")
    with pytest.raises(ValueError, match="voronoi_control_points"):
        TrackGenConfig(voronoi_control_points=5)


def test_repulsive_config_defaults_and_validation():
    # Defaults construct clean (the spike's tuned config).
    cfg = TrackGenConfig()
    assert cfg.repulsive_grow_mult_min == 4.5
    assert cfg.repulsive_grow_mult_max == 5.5
    assert cfg.repulsive_domain_frac == 0.35
    assert cfg.repulsive_domain_init_ratio == 4.0
    assert cfg.repulsive_obstacle_count_min == 8
    assert cfg.repulsive_obstacle_count_max == 12
    assert cfg.repulsive_ratchet_rate == 0.012
    assert cfg.repulsive_stages == (64, 128, 256)
    # A repulsive config with stages[-1] == num_points also constructs clean.
    TrackGenConfig(generator="repulsive", num_points=256, repulsive_stages=(64, 128, 256))

    # grow-mult must genuinely grow: min >= 1 and max > 1. A multiplier <= 1 zeroes the growth
    # budget (n_ratchet <= 0 -> empty growth loop -> centerline left mis-strided at the coarse
    # stage), so both are rejected at config time.
    with pytest.raises(ValueError, match="repulsive_grow_mult_min"):
        TrackGenConfig(repulsive_grow_mult_min=0.9)
    with pytest.raises(ValueError, match="repulsive_grow_mult_max"):
        TrackGenConfig(repulsive_grow_mult_min=1.0, repulsive_grow_mult_max=1.0)
    # settle_iters / resample_every / stall_window must be >= 1 (empty growth loop, or a
    # ZeroDivisionError at the `(it+1) % k` guards mid-generate).
    with pytest.raises(ValueError, match="repulsive_settle_iters"):
        TrackGenConfig(repulsive_settle_iters=0)
    with pytest.raises(ValueError, match="repulsive_resample_every"):
        TrackGenConfig(repulsive_resample_every=0)
    with pytest.raises(ValueError, match="repulsive_stall_window"):
        TrackGenConfig(repulsive_stall_window=0)
    # Ordered range: max < min raises.
    with pytest.raises(ValueError, match="repulsive_grow_mult"):
        TrackGenConfig(repulsive_grow_mult_min=5.5, repulsive_grow_mult_max=4.5)
    with pytest.raises(ValueError, match="repulsive_obstacle_radius"):
        TrackGenConfig(repulsive_obstacle_radius_min_frac=0.05,
                       repulsive_obstacle_radius_max_frac=0.02)
    # Counts >= 1.
    with pytest.raises(ValueError, match="repulsive_obstacle_count_min"):
        TrackGenConfig(repulsive_obstacle_count_min=0)
    with pytest.raises(ValueError, match="repulsive_obstacle_count"):
        TrackGenConfig(repulsive_obstacle_count_min=12, repulsive_obstacle_count_max=8)
    # domain_init_ratio > 1.
    with pytest.raises(ValueError, match="repulsive_domain_init_ratio"):
        TrackGenConfig(repulsive_domain_init_ratio=1.0)
    # domain_frac > 0.
    with pytest.raises(ValueError, match="repulsive_domain_frac"):
        TrackGenConfig(repulsive_domain_frac=0.0)
    # ratchet_rate > 0.
    with pytest.raises(ValueError, match="repulsive_ratchet_rate"):
        TrackGenConfig(repulsive_ratchet_rate=0.0)
    # alpha, beta > 0.
    with pytest.raises(ValueError, match="repulsive_alpha"):
        TrackGenConfig(repulsive_alpha=0.0)
    with pytest.raises(ValueError, match="repulsive_beta"):
        TrackGenConfig(repulsive_beta=-1.0)
    # stages strictly increasing.
    with pytest.raises(ValueError, match="repulsive_stages"):
        TrackGenConfig(repulsive_stages=(128, 128, 256))
    # stages must be positive multiples of 4.
    with pytest.raises(ValueError, match="repulsive_stages"):
        TrackGenConfig(repulsive_stages=(64, 130, 256))
    # For generator="repulsive", stages[-1] must equal num_points.
    with pytest.raises(ValueError, match="repulsive_stages"):
        TrackGenConfig(generator="repulsive", num_points=256, repulsive_stages=(64, 128, 200))
    # But for a non-repulsive generator, a mismatched last stage is allowed.
    TrackGenConfig(generator="bezier", num_points=256, repulsive_stages=(64, 128, 200))


def test_track_construct_from_tensors_field_shapes():
    import warp as wp
    wp.init()
    E, N = 4, 16
    track = Track(
        outer=wp.zeros(E * N, dtype=wp.vec2f),
        center=wp.zeros(E * N, dtype=wp.vec2f),
        inner=wp.zeros(E * N, dtype=wp.vec2f),
        tangent=wp.zeros(E * N, dtype=wp.vec2f),
        normal=wp.zeros(E * N, dtype=wp.vec2f),
        arclen=wp.zeros(E * N, dtype=wp.float32),
        length=wp.zeros(E, dtype=wp.float32),
        valid=wp.zeros(E, dtype=wp.int32),
        count=wp.zeros(E, dtype=wp.int32),
        winding=wp.zeros(E, dtype=wp.float32),
    )
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (E * N,)
        assert arr.dtype == wp.vec2f
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
    assert cfg.energy_steps == 800
    assert cfg.tp_iters == 100
    assert cfg.smooth_finish is False


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
    assert cfg.device == "cpu"
    assert cfg.num_envs == 1
    assert cfg.min_gates == 4
    assert cfg.max_gates == 32
    assert cfg.gate_radius == 0.025
    assert cfg.gate_solve_iters == 8
    assert cfg.gate_width == 0.0
    assert cfg.gate_ordering == "ccw"
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
    with pytest.raises(ValueError, match="gate_ordering"):
        GateGenConfig(gate_ordering="spiral")
    with pytest.raises(ValueError, match="voronoi_radial_variation"):
        GateGenConfig(voronoi_radial_variation=-0.1)
    with pytest.raises(ValueError, match="voronoi_angular_jitter"):
        GateGenConfig(voronoi_angular_jitter=-0.1)


def test_gate_config_validates_num_envs_and_point_family_inputs():
    from track_gen._src.types import GateGenConfig

    with pytest.raises(ValueError, match="num_envs"):
        GateGenConfig(num_envs=0)
    # min_point_distance feeds a 1/(d*2) cell count in the shared corner sampler; a
    # non-positive value must fail at construction, not as a downstream ZeroDivisionError.
    with pytest.raises(ValueError, match="min_point_distance"):
        GateGenConfig(min_point_distance=0.0)
    with pytest.raises(ValueError, match="min_num_points"):
        GateGenConfig(min_num_points=1)
    with pytest.raises(ValueError, match="max_num_points"):
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
    with pytest.raises(ValueError, match="z_profile"):
        TrackGenConfig(device="cpu", num_envs=1, z_profile="bogus")
    with pytest.raises(ValueError, match="z_min"):
        TrackGenConfig(device="cpu", num_envs=1, z_min=2.0, z_max=1.0)


def test_track_config_validates_sampler_and_buffer_invariants():
    from track_gen._src.types import TrackGenConfig
    import pytest
    with pytest.raises(ValueError, match="num_envs"):
        TrackGenConfig(num_envs=0)
    with pytest.raises(ValueError, match="half_width"):
        TrackGenConfig(half_width=0.0)
    with pytest.raises(ValueError, match="min_num_points"):
        TrackGenConfig(min_num_points=1)
    with pytest.raises(ValueError, match="max_num_points"):
        TrackGenConfig(min_num_points=13, max_num_points=9)
    with pytest.raises(ValueError, match="min_point_distance"):
        TrackGenConfig(min_point_distance=0.0)
    with pytest.raises(ValueError, match="min_point_distance"):
        TrackGenConfig(min_point_distance=0.6)
    with pytest.raises(ValueError, match="num_points_per_segment"):
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
    from track_gen._src.types import GateSequence

    wp.init()
    E, G = 2, 8
    gates = GateSequence(
        position=wp.zeros(E * G, dtype=wp.vec3f),
        tangent=wp.zeros(E * G, dtype=wp.vec3f),
        forward=wp.zeros(E * G, dtype=wp.vec3f),
        orientation=wp.zeros(E * G, dtype=wp.quatf),
        half_size=wp.zeros(E * G, dtype=wp.float32),
        left=wp.zeros(E * G, dtype=wp.vec3f),
        right=wp.zeros(E * G, dtype=wp.vec3f),
        valid=wp.zeros(E, dtype=wp.int32),
        count=wp.zeros(E, dtype=wp.int32),
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
