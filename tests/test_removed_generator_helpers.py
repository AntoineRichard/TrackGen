"""The 0.3 generator implementation exposes no frozen compatibility helpers."""
import importlib
import inspect

import pytest


_REMOVED = {
    "warp_generate": ("generate_centerline_warp", "bezier_alloc_scratch", "style_sample_inplace",
                      "assemble_inplace", "corner_count_sample_inplace", "corner_sample_inplace",
                      "ccw_sort_inplace", "_safe_normalize2", "_style_sample_k", "_assemble_k",
                      "_assemble_style_k"),
    "warp_generate_hull": ("generate_hull_warp", "hull_alloc_scratch", "point_count_sample_inplace",
                           "point_sample_inplace", "_midpoint_displace_k"),
    "warp_generate_polar": ("generate_polar_warp", "polar_alloc_scratch", "sample_polar_controls",
                            "_polar_controls_k", "_polar_num_knots", "_catmull_rom"),
    "warp_generate_voronoi": ("generate_voronoi_warp", "voronoi_alloc_scratch",
                              "select_voronoi_anchors", "_select_anchor_sites_k", "_voronoi_layout_mode"),
    "warp_generate_checkpoint": ("generate_checkpoint_warp", "checkpoint_alloc_scratch",
                                 "sample_checkpoint_rings", "_sample_checkpoints_k", "_steer_k"),
    "warp_generate_repulsive": ("generate_repulsive_warp", "repulsive_alloc_scratch", "_safe_dir"),
    "warp_zprofile": ("apply_z_profile", "apply_z_profile_knots", "_z_uniform_k", "_z_walk_k",
                       "_z_noise_k"),
    "warp_gate": ("_CAPTURING", "_init", "_fill_f32_k", "alloc_order_scratch", "order_points", "finish_ordered_gates", "__getattr__",
                   "_safe_normalize2", "_sync", "_fill_i32_k", "_fill_vec2_k", "_fill_vec3_k", "_fill_quat_k"),
    "warp_pipeline": ("inflate_warp", "_CAPTURING", "_init", "__getattr__", "generate_centerline_warp", "GenScratch", "_sync",
                       "_fill_f32_k", "_fill_i32_k", "_ccw", "_nan0", "_self_intersections_func",
                       "_turning_func", "_thickness_func", "_arc_resample_inplace",
                       "_arc_resample_selected_inplace", "self_intersections_inplace",
                       "arc_length_resample_inplace", "arc_length_resample_selected_inplace",
                       "resample_uniform", "resample_constant_spacing", "turning_number_inplace"),
}


@pytest.mark.parametrize("module,names", _REMOVED.items())
def test_frozen_generator_helpers_are_absent(module, names):
    mod = importlib.import_module("track_gen._src." + module)
    assert not [name for name in names if hasattr(mod, name)]


@pytest.mark.parametrize("module,cls", [("warp_generate", "Bezier"),
    ("warp_generate_hull", "Hull"), ("warp_generate_polar", "Polar"),
    ("warp_generate_voronoi", "Voronoi"), ("warp_generate_checkpoint", "Checkpoint")])
def test_builtin_params_alias_is_absent(module, cls):
    assert not hasattr(getattr(importlib.import_module("track_gen._src." + module), cls), "params")


def test_bezier_style_params_are_absent():
    from track_gen._src.builtin_params import BezierParams
    assert not {"style_sampling", "rad_range", "scale_range", "handle_clamp_frac_range"}.intersection(
        BezierParams.__dataclass_fields__)


@pytest.mark.parametrize("function", ["xpbd_solve_inplace", "band_l0_inplace"])
def test_relaxation_has_no_ignored_capturing_argument(function):
    from track_gen._src import warp_relax
    assert "capturing" not in inspect.signature(getattr(warp_relax, function)).parameters
