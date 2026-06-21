import math

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
