import math

import torch

from track_gen.types import Track, TrackGenConfig


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
    # Fourier params (reconciled names)
    assert cfg.num_harmonics == 5
    assert cfg.decay_p == 2
    assert cfg.amplitude == 1.0
    assert cfg.num_centerline_samples == 256
    # Width params
    assert cfg.half_width == 0.1
    # Output params
    assert cfg.num_points == 256
    # "fixed" was dropped; constant_spacing is the only supported mode.
    assert cfg.output_mode == "constant_spacing"
    # spacing defaults to None -> __post_init__ auto-couples it to 0.6*half_width.
    assert math.isclose(cfg.spacing, 0.6 * cfg.half_width)
    assert cfg.N_max == 256
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
    E, N = 4, 16
    track = Track(
        outer=torch.zeros(E, N, 2),
        center=torch.zeros(E, N, 2),
        inner=torch.zeros(E, N, 2),
        tangent=torch.zeros(E, N, 2),
        normal=torch.zeros(E, N, 2),
        arclen=torch.zeros(E, N),
        length=torch.zeros(E),
        valid=torch.ones(E, dtype=torch.bool),
        count=torch.full((E,), N, dtype=torch.long),
    )
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (E, N, 2)
    assert track.arclen.shape == (E, N)
    assert track.length.shape == (E,)
    assert track.valid.shape == (E,)
    assert track.valid.dtype == torch.bool
    assert track.count.shape == (E,)


def test_relaxation_defaults():
    from track_gen.types import TrackGenConfig
    cfg = TrackGenConfig()
    assert cfg.relax_enable is True
    assert cfg.relax_solver == "xpbd"
    assert cfg.relax_bend_relax == 1.5
    assert cfg.relax_margin == 0.15
    assert cfg.energy_steps == 800
    assert cfg.tp_iters == 100
    assert cfg.smooth_finish is False


def test_deprecated_width_clamp_fields_removed():
    from track_gen.types import TrackGenConfig
    cfg = TrackGenConfig()
    for dead in ("alpha", "clamp_self_distance", "self_distance_margin",
                 "self_distance_band", "self_distance_decimation"):
        assert not hasattr(cfg, dead), f"{dead} should be removed"


def test_types_module_has_no_intra_package_imports():
    # The leaf must not import generators/inflation/track_generator/rng_utils.
    import track_gen.types as t

    src = open(t.__file__).read()
    for forbidden in ("from .generators", "from .inflation", "from .track_generator", "from .rng_utils", "import warp"):
        assert forbidden not in src, f"types.py must not contain '{forbidden}'"
