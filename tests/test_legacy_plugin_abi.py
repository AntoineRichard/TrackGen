"""The private helpers legacy (pre-``track_gen.ops``) generator plugins call keep working.

Out-of-tree plugins written against the ``_pipe.*`` helpers keep their names, parameter
lists (names, kinds, defaults) and behaviour after PR7 moved the implementations into
``track_gen.ops``. The pinned parameter lists below were recorded with
``inspect.signature`` at HEAD 1077b18 (annotations intentionally not pinned). Each
pinned wrapper is also run on a tiny CPU batch with ``ctx=None`` inside a
``LaunchContext.scope()``, the way ``_run_pipeline`` runs a legacy plugin.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import warp as wp

from track_gen._src import runtime
from track_gen._src import warp_generate
from track_gen._src import warp_generate_hull
from track_gen._src import warp_pipeline as _pipe
from track_gen._src.types import TrackGenConfig
from track_gen.exec import LaunchContext

DEV = "cpu"
_REQ = "<required>"

_ARC = ["points_wp", "M", "num", "real_wp", "seg_wp", "s_wp", "count_r_wp", "count_out_wp",
        "out_wp", "dev", ("ctx", None)]
_ARC_SEL = ["points_wp", "active_wp", "M", "num", "real_wp", "seg_wp", "s_wp", "count_r_wp",
            "count_out_wp", "out_wp", "dev", ("ctx", None)]

# (module, name) -> parameter list at HEAD 1077b18; a bare string is a required
# POSITIONAL_OR_KEYWORD parameter, a (name, default) tuple an optional one.
PINNED = {
    (_pipe, "_init"): [],
    (_pipe, "_sync"): ["device", ("stream", None)],
    (_pipe, "_arc_resample_inplace"): _ARC,
    (_pipe, "_arc_resample_selected_inplace"): _ARC_SEL,
    (_pipe, "self_intersections_inplace"): ["poly_wp", "count_wp", "out_wp", "n_max",
                                            ("ctx", None)],
    (_pipe, "arc_length_resample_inplace"): _ARC,
    (_pipe, "arc_length_resample_selected_inplace"): _ARC_SEL,
    (_pipe, "resample_uniform"): ["center_wp", "out_wp", "n", "count_wp", ("seg_wp", None),
                                  ("s_wp", None), ("device", "cpu"), ("ctx", None)],
    (_pipe, "resample_constant_spacing"): ["center", "spacing", "n_max", ("out_wp", None),
                                           ("count_wp", None), ("seg_wp", None),
                                           ("s_wp", None), ("truncated_wp", None),
                                           ("ctx", None)],
    (_pipe, "turning_number_inplace"): ["center_wp", "n_max", "count_wp", "out_wp",
                                        ("ctx", None)],
    (warp_generate, "corner_count_sample_inplace"): ["seeds_wp", "attempt", "config",
                                                     "out_count", ("ctx", None)],
    (warp_generate, "corner_sample_inplace"): ["seeds_wp", "attempt", "config",
                                               "out_corners", "used_scratch",
                                               ("scale_per_env", None), ("ctx", None)],
    (warp_generate, "ccw_sort_inplace"): ["corners_wp", "count_wp", "keys_scratch", "out_wp",
                                          "P", ("ctx", None)],
    (warp_generate_hull, "point_count_sample_inplace"): ["seeds_wp", "config", "out_count",
                                                         ("ctx", None)],
    (warp_generate_hull, "point_sample_inplace"): ["seeds_wp", "config", "out_points",
                                                   "used_scratch", ("ctx", None)],
}


def _params(fn):
    out = []
    for p in inspect.signature(fn).parameters.values():
        out.append((p.name, p.kind, _REQ if p.default is inspect.Parameter.empty
                    else p.default))
    return out


def _expected(spec):
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    return [(s, kind, _REQ) if isinstance(s, str) else (s[0], kind, s[1]) for s in spec]


@pytest.mark.parametrize("key", list(PINNED), ids=lambda k: f"{k[0].__name__}.{k[1]}")
def test_pinned_signatures(key):
    mod, name = key
    fn = getattr(mod, name)
    assert callable(fn)
    assert _params(fn) == _expected(PINNED[key])


def test_init_and_sync_are_the_runtime_helpers():
    assert _pipe._init is runtime._init
    assert _pipe._sync is runtime._sync
    _pipe._init()
    _pipe._sync(DEV)


def test_fill_i32_kernel_abi():
    k = _pipe._fill_i32_k
    assert isinstance(k, wp.Kernel)
    arr_arg, v_arg = k.adj.args
    assert arr_arg.label == "arr" and isinstance(arr_arg.type, wp.array)
    assert arr_arg.type.dtype is wp.int32 and arr_arg.type.ndim == 1
    assert v_arg.label == "v" and v_arg.type is wp.int32
    arr = wp.empty(4, dtype=wp.int32, device=DEV)
    wp.launch(k, 4, [arr, 5], device=DEV)
    np.testing.assert_array_equal(arr.numpy(), [5, 5, 5, 5])


def test_pipe_ccw_is_a_device_function_usable_in_a_plugin_kernel():
    assert isinstance(_pipe._ccw, wp.Function)

    @wp.kernel
    def plugin_k(out: wp.array(dtype=wp.float32)):
        out[0] = _pipe._ccw(0.0, 0.0, 1.0, 0.0, 1.0, 1.0)

    out = wp.empty(1, dtype=wp.float32, device=DEV)
    wp.launch(plugin_k, 1, [out], device=DEV)
    assert out.numpy()[0] == 1.0


def test_pipe_keeps_the_historical_kernel_names():
    # Kernels moved into track_gen.ops are re-exported under their old names.
    for name in ("_ccw", "_nan0", "_self_intersections_func", "_turning_func",
                 "_thickness_func"):
        assert isinstance(getattr(_pipe, name), wp.Function), name
    for name in ("_self_intersections_by_i_k", "_thickness_k", "_turning_k", "_resample_scan_k",
                 "_resample_lookup_k", "_cs_scan_k", "_cs_lookup_k", "_arc_scan_k",
                 "_arc_lookup_k", "_arc_scan_selected_k", "_arc_lookup_selected_k",
                 "_fill_f32_k", "_fill_i32_k"):
        assert isinstance(getattr(_pipe, name), wp.Kernel), name


def _v2(a):
    return wp.array(np.asarray(a, np.float32).reshape(-1, 2), dtype=wp.vec2f, device=DEV)


def _i32(a):
    return wp.array(np.asarray(a, np.int32).reshape(-1), dtype=wp.int32, device=DEV)


def _circle(n, r=1.0):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], -1).astype(np.float32)


def test_pinned_wrappers_run_with_ambient_context():
    """Every pinned wrapper runs with ctx=None under an ambient LaunchContext (the legacy
    plugin path) and produces sane results."""
    E, M, num, n = 2, 12, 8, 10
    dense = np.stack([_circle(M), _circle(M, 2.0)])
    dense[1, 4] = np.nan
    dense_wp = _v2(dense)
    real = wp.empty(E * M, dtype=wp.vec2f, device=DEV)
    seg = wp.empty(E * M, dtype=wp.float32, device=DEV)
    s = wp.empty(E * (M + 1), dtype=wp.float32, device=DEV)
    cr = wp.empty(E, dtype=wp.int32, device=DEV)
    co = wp.empty(E, dtype=wp.int32, device=DEV)
    out = wp.empty(E * num, dtype=wp.vec2f, device=DEV)
    loops = _v2(np.stack([_circle(n), _circle(n, 3.0)]))
    cnt = _i32([n, n])
    cfg = TrackGenConfig(device=DEV, num_envs=E, min_num_points=3, max_num_points=6)
    P = int(cfg.max_num_points)
    seeds = _i32([3, 4])
    ctx = LaunchContext(DEV, None)
    with ctx.scope():
        _pipe._arc_resample_inplace(dense_wp, M, num, real, seg, s, cr, co, out, DEV)
        np.testing.assert_array_equal(cr.numpy(), [M, M - 1])
        np.testing.assert_array_equal(co.numpy(), [num, num])
        _pipe.arc_length_resample_inplace(dense_wp, M, num, real, seg, s, cr, co, out, DEV)
        _pipe._arc_resample_selected_inplace(dense_wp, _i32([0, 1]), M, num, real, seg, s,
                                             cr, co, out, DEV)
        np.testing.assert_array_equal(co.numpy(), [0, num])
        _pipe.arc_length_resample_selected_inplace(dense_wp, _i32([1, 1]), M, num, real,
                                                   seg, s, cr, co, out, DEV)
        x = wp.empty(E, dtype=wp.int32, device=DEV)
        _pipe.self_intersections_inplace(loops, cnt, x, n)
        np.testing.assert_array_equal(x.numpy(), [0, 0])
        rs = wp.empty(E * n, dtype=wp.vec2f, device=DEV)
        _pipe.resample_uniform(loops, rs, n, cnt, device=DEV)
        assert np.isfinite(rs.numpy()).all()
        tn = wp.empty(E, dtype=wp.float32, device=DEV)
        _pipe.turning_number_inplace(loops, n, cnt, tn)
        np.testing.assert_allclose(tn.numpy(), 2 * np.pi, atol=1e-4)
        cso, csc = _pipe.resample_constant_spacing(loops, 0.5, 64, count_wp=None,
                                                   out_wp=wp.empty(E * 64, dtype=wp.vec2f,
                                                                   device=DEV))
        assert (csc.numpy() > 0).all()
        count = wp.empty(E, dtype=wp.int32, device=DEV)
        warp_generate.corner_count_sample_inplace(seeds, 0, cfg, count)
        assert ((count.numpy() >= 3) & (count.numpy() <= P)).all()
        corners = wp.empty(E * P, dtype=wp.vec2f, device=DEV)
        used = wp.empty(E * P, dtype=wp.int32, device=DEV)
        warp_generate.corner_sample_inplace(seeds, 0, cfg, corners, used)
        warp_generate.corner_sample_inplace(
            seeds, 0, cfg, corners, used,
            scale_per_env=wp.array(np.ones(E, np.float32), dtype=wp.float32, device=DEV))
        keys = wp.empty(E * P, dtype=wp.float32, device=DEV)
        srt = wp.empty(E * P, dtype=wp.vec2f, device=DEV)
        warp_generate.ccw_sort_inplace(corners, count, keys, srt, P)
        warp_generate_hull.point_count_sample_inplace(seeds, cfg, count)
        warp_generate_hull.point_sample_inplace(seeds, cfg, corners, used)
        assert np.isfinite(corners.numpy()).all()


@pytest.mark.parametrize("E", [1, 3])
def test_corner_sample_accepts_any_per_env_scale_array(E):
    # At 1077b18 scale_per_env went straight to the per-env kernel, so any [E] array
    # Warp's launch accepts worked (a numpy float32 array included, also for E == 1).
    cfg = TrackGenConfig(device=DEV, num_envs=E, min_num_points=3, max_num_points=6)
    P = int(cfg.max_num_points)
    seeds = _i32(np.arange(E) * 17 + 3)
    scale = np.linspace(0.5, 1.5, E).astype(np.float32)
    outs = []
    for sc in (scale, wp.array(scale, dtype=wp.float32, device=DEV)):
        corners = wp.empty(E * P, dtype=wp.vec2f, device=DEV)
        used = wp.empty(E * P, dtype=wp.int32, device=DEV)
        warp_generate.corner_sample_inplace(seeds, 0, cfg, corners, used, scale_per_env=sc)
        outs.append(corners.numpy())
    np.testing.assert_array_equal(outs[0].view(np.uint32), outs[1].view(np.uint32))


def test_legacy_private_aliases_still_resolve():
    # Private names out-of-tree plugins imported at 1077b18 that have a same-arithmetic
    # public replacement keep resolving (the removed kernels are listed in the CHANGELOG).
    from track_gen import ops
    from track_gen._src import (warp_gate, warp_generate_polar, warp_generate_repulsive,
                                warp_generate_voronoi)
    assert warp_generate_voronoi._voronoi_layout_mode is warp_generate_voronoi.voronoi_layout_mode
    assert warp_generate_polar._polar_num_knots is warp_generate_polar.polar_num_knots
    assert warp_generate_polar._catmull_rom is ops.funcs.catmull_rom
    for fn in (warp_gate._safe_normalize2, warp_generate._safe_normalize2,
               _pipe._safe_normalize2, warp_generate_repulsive._safe_dir):
        assert fn is ops.funcs.safe_normalize2
    assert isinstance(warp_generate_polar._polar_spline_dense_k, wp.Kernel)


# ---------------------------------------------------------------------------------------
# GenScratch moved to warp_generate (bezier's private scratch); the old name warns.
# ---------------------------------------------------------------------------------------

def test_genscratch_lives_in_warp_generate():
    assert warp_generate.GenScratch.__module__ == "track_gen._src.warp_generate"
    gs = warp_generate.bezier_alloc_scratch(TrackGenConfig(device=DEV, num_envs=2))
    assert isinstance(gs, warp_generate.GenScratch)
    assert "GenScratch" not in vars(_pipe)


def test_pipe_genscratch_attribute_warns_and_aliases():
    with pytest.warns(DeprecationWarning, match="warp_generate.GenScratch"):
        cls = _pipe.GenScratch
    assert cls is warp_generate.GenScratch


def test_pipe_genscratch_from_import_warns():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        from track_gen._src.warp_pipeline import GenScratch  # noqa: F401
    assert GenScratch is warp_generate.GenScratch


def test_other_generation_names_still_forward_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _pipe.generate_centerline_warp is warp_generate.generate_centerline_warp
        assert _pipe.corner_sample_inplace is warp_generate.corner_sample_inplace
    with pytest.raises(AttributeError):
        _pipe.no_such_name  # noqa: B018


# ---------------------------------------------------------------------------------------
# The warp_gate helpers of 0.1 gate plugins (deprecated shims since roadmap PR11)
# ---------------------------------------------------------------------------------------

from tests import _legacy_gate_pr11 as _gate_ref  # noqa: E402
from track_gen._src import warp_gate  # noqa: E402

# name -> parameter list at dd181f0 (0.2.0.dev3), as PINNED.
GATE_PINNED = {
    "alloc_order_scratch": ["config"],
    "order_points": ["seeds_wp", "src", "src_stride", "count", "max_gates", "ordering",
                     "keys", "dst", ("ctx", None)],
    "finish_ordered_gates": ["seeds_wp", "src", "src_stride", "count", "max_gates",
                             "ordering", "keys", "out", ("normalize_extent", None),
                             ("ctx", None)],
}
_K_RING = 7


@pytest.mark.parametrize("name", list(GATE_PINNED))
def test_gate_helper_signatures_are_pinned(name):
    fn = getattr(warp_gate, name)
    assert callable(fn)
    assert _params(fn) == _expected(GATE_PINNED[name])


def test_gate_staging_name_warns_and_aliases():
    from track_gen.stages.gates import _GateStaging

    with pytest.warns(DeprecationWarning, match="_GateStaging .*deprecated"):
        cls = warp_gate._GateStaging
    assert cls is _GateStaging
    with pytest.raises(AttributeError):
        warp_gate.no_such_name  # noqa: B018


@pytest.mark.parametrize("ordering", ["raw", "ccw", "random_pairs"])
def test_gate_helpers_match_the_frozen_dd181f0_code(ordering):
    """``finish_ordered_gates`` (order + extent normalization + count copy) and
    ``order_points`` reproduce the frozen 0.2.0.dev3 ordering bit for bit."""
    import types

    E, A, G = 3, 9, 12
    rs = np.random.default_rng(5)
    src = _v2(rs.uniform(-1.0, 1.0, (E * A, 2)))
    count = _i32([9, 6, 0])
    seeds = _i32([3, 4, 5])
    cfg = types.SimpleNamespace(num_envs=E, max_gates=G, device=DEV)
    with pytest.deprecated_call():
        cnt_scratch, keys = warp_gate.alloc_order_scratch(cfg)
    assert (cnt_scratch.shape, keys.shape) == ((E,), (E * G,))
    ref_ctx = LaunchContext(DEV, None)
    want = wp.zeros(E * G, dtype=wp.vec2f, device=DEV)
    ref_keys = wp.zeros(E * G, dtype=wp.float32, device=DEV)
    _gate_ref._order_points(ref_ctx, seeds, src, A, count, G, ordering, ref_keys, want)
    got = wp.zeros(E * G, dtype=wp.vec2f, device=DEV)
    with pytest.deprecated_call():
        warp_gate.order_points(seeds, src, A, count, G, ordering, keys, got)
    assert got.numpy().tobytes() == want.numpy().tobytes()
    from track_gen.ops.geometry import _normalize_extent_launch

    _normalize_extent_launch(ref_ctx, want, G, 1.5, E, count=count)
    staging = types.SimpleNamespace(position=wp.zeros(E * G, dtype=wp.vec2f, device=DEV),
                                    count=wp.zeros(E, dtype=wp.int32, device=DEV))
    with pytest.deprecated_call():
        warp_gate.finish_ordered_gates(seeds, src, A, count, G, ordering, keys, staging,
                                       normalize_extent=1.5)
    assert staging.position.numpy().tobytes() == want.numpy().tobytes()
    np.testing.assert_array_equal(staging.count.numpy(), count.numpy())
    with pytest.raises(ValueError, match="unsupported gate ordering 'spiral'"), \
            pytest.deprecated_call():
        warp_gate.order_points(seeds, src, A, count, G, "spiral", keys, got)


@wp.kernel
def _ring_points_k(seeds: wp.array(dtype=wp.int32), K: int, out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // K
    i = t - e * K
    state = wp.rand_init(seeds[e], i)
    theta = 6.2831853 * float(i) / float(K) + 0.2 * wp.randf(state)
    r = 1.0 + 0.3 * wp.randf(state)
    out[t] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


def _v01_alloc_scratch(config):
    import types

    warp_gate._init()
    count, keys = warp_gate.alloc_order_scratch(config)
    return types.SimpleNamespace(
        points=wp.empty(int(config.num_envs) * _K_RING, dtype=wp.vec2f,
                        device=str(config.device)), count=count, keys=keys)


def _v01_generate(seeds_wp, config, out, scratch) -> None:
    # A 0.1-ABI gate plugin, modelled on the 0.1 built-in polar gate generator: no ctx=,
    # direct wp.launch calls, then the warp_gate ordering helper.
    E, G, dev = int(config.num_envs), int(config.max_gates), str(out.position.device)
    wp.launch(warp_gate._fill_i32_k, dim=E, inputs=[scratch.count, _K_RING], device=dev)
    wp.launch(_ring_points_k, dim=E * _K_RING, inputs=[seeds_wp, _K_RING, scratch.points],
              device=dev)
    warp_gate.finish_ordered_gates(seeds_wp, scratch.points, _K_RING, scratch.count, G,
                                   str(config.gate_ordering), scratch.keys, out,
                                   normalize_extent=float(config.scale) * 1.5)


@pytest.mark.parametrize("ordering", ["raw", "ccw", "random_pairs"])
def test_a_v01_gate_plugin_still_runs(ordering, plugin_registry):
    """A ``GateGeneratorSpec`` written to the 0.1 ABI (``alloc_order_scratch`` in
    ``alloc_scratch``, ``finish_ordered_gates`` in ``generate``) runs through the legacy
    adapter, under ``DeprecationWarning``s, byte-identical to the frozen 0.2.0.dev3 gate
    pipeline running the same plugin."""
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen._src.gate_generator_registry import GateGeneratorSpec

    spec = GateGeneratorSpec(name="legacy01.ring", alloc_scratch=_v01_alloc_scratch,
                             generate=_v01_generate, max_gates=lambda config: _K_RING,
                             supported_orderings=frozenset({"ccw", "raw", "random_pairs"}))
    plugin_registry.add(spec)
    E = 4
    cfg = GateGenConfig(generator="legacy01.ring", gate_ordering=ordering, num_envs=E,
                        device=DEV, max_gates=8, z_profile="uniform", z_max=0.5)
    seeds = [11, 12, 13, 14]
    rng = PerEnvSeededRNG(seeds=_i32(seeds), num_envs=E, device=DEV)
    with pytest.deprecated_call():
        gen = GateGenerator(cfg, rng)
        seq = gen.generate()
    with pytest.deprecated_call():
        want = _gate_ref.run_reference_legacy(cfg, spec, seeds)
    assert (seq.count.numpy() == _K_RING).all() and seq.valid.numpy().any()
    for f in _gate_ref.FINAL_FIELDS:
        assert getattr(seq, f).numpy().tobytes() == want[f].tobytes(), f
