"""Roadmap PR13c: dynamic built-in knobs (CPU; the CUDA twin is
``tests/test_dynamic_knobs_cuda.py``).

The five built-in track generators are generator Stages whose shape knobs are dynamic
float32 params (bezier ``rad`` / ``handle_clamp_frac``, ``hull_displacement``, the polar
jitters, the voronoi radial variation / angular jitter, the checkpoint radius / angle
jitter / turn rate; the gate generators' copies of the polar / voronoi / checkpoint
ones); the elevation profiles' ``z_min`` / ``z_max`` / ``z_max_step`` /
``z_noise_amplitude`` too. Each is read on the device by an array-reading twin of its
scalar kernel (clamped to the declared bounds by comparisons); the frozen legacy entry
points keep the scalar kernels. Covered here (design section 9.2):

1. the declared dynamic sets, bounds, schemas, Stage classes and the deprecated
   ``.params`` alias;
2. each twin == its scalar kernel on generator-produced inputs (uniform, per-env,
   out-of-range == at the bound, ``-0.0``, NaN; polar == ``polar_jitters()`` + scalar;
   bezier with ``(-inf, inf)`` bounds == ``_assemble_style_k``);
3. every generator: per-env knobs == scalar runs env by env (non-vacuous);
4. each Stage's raw output == its frozen ``generate_*_warp``;
5. the gate generators per env (both orderings);
6. the elevation profiles per env (track, gates, relaxed stations);
7. ``randomize`` on built-in knobs (presets.track / presets.gates, masked advance) and
   ``PipelineSpec.with_randomize``;
8. the cross-field draw rules and the inert-window warnings;
9. the launch deltas (the rule of ``tests/_pr13c_delta.py`` against the legacy
   composition, the design section 4 counts, plain and masked);
10. bounds (float32) and value types;
11. the poison sentinels of the built-ins' static views and the ``dyn`` key checks;
12. ``style_sampling`` (bytes == the frozen legacy path; self-drawn knobs);
13. the generated CUDA source of every scalar / twin pair (locked until PR16);
14. the one-generation latch lag of a write recorded after segment 0;
15. the legacy surfaces (the PR9 composition, frozen signatures, a ``GeneratorSpec``
    with dynamic Params refused);
16. the kit's ``params.per_env`` on toys reading ``m[0]`` / writing their param array.
"""
from __future__ import annotations

import ast
import collections
import dataclasses
import inspect
import math
import re
import textwrap
import types
import warnings

import numpy as np
import pytest
import warp as wp

from tests._pr13c_delta import (ELEVATION_LATCHES, GATE_GENERATOR_LATCHES,
                                GENERATOR_LATCHES, LATCH_ALL, LATCH_MASKED,
                                pr13c_sequence)
from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig,
                       TrackGenerator)
from track_gen import plugin
from track_gen._src import builtin_params as bp
from track_gen._src import warp_generate as wg_bezier
from track_gen._src import warp_generate_checkpoint as wg_checkpoint
from track_gen._src import warp_generate_hull as wg_hull
from track_gen._src import warp_generate_polar as wg_polar
from track_gen._src import warp_generate_voronoi as wg_voronoi
from track_gen._src import warp_pipeline, warp_zprofile
from track_gen._src.builtin_stage import run_view
from track_gen.exec import LaunchContext
from track_gen.pipeline import PlanError, StageRef, build, legacy, presets
from track_gen.plugin import GeneratorSpec, registry
from track_gen.plugin.params import _DynamicField, _fields, _static_view
from track_gen.plugin.stage import _is_stage
from track_gen.stages.track import _zcfg
from track_gen.testing import check_generator
from track_gen.testing.capture import result_arrays
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 8
#: The fingerprint seeds (int32 extremes included); at epoch 0 the effective seed is the
#: base seed.
SEEDS = np.array([0, 1, 7, 99991, 2147483647, 1000003, 123456789, 46341], np.int32)
SEEDS_B = np.arange(E, dtype=np.int32) * 7919 + 3
INF = math.inf

#: The dynamic set (design section 2.2): (kind, name) -> {field: (lo, hi)}.
DYNAMIC = {
    ("generator", "bezier"): {"rad": (0.0, 1.0), "handle_clamp_frac": (0.0, None)},
    ("generator", "hull"): {"hull_displacement": (0.0, 1.0)},
    ("generator", "polar"): {"polar_radial_jitter": (0.0, 0.85),
                             "polar_angular_jitter": (0.0, 0.45)},
    ("generator", "voronoi"): {"voronoi_radial_variation": (0.0, 0.85),
                               "voronoi_angular_jitter": (0.0, math.pi)},
    ("generator", "checkpoint"): {"checkpoint_radius_min_frac": (0.0, 1.0),
                                  "checkpoint_angle_jitter": (0.0, 1.0),
                                  "checkpoint_turn_rate": (0.0, math.pi)},
    ("gate_generator", "polar"): {"polar_radial_jitter": (0.0, 0.85),
                                  "polar_angular_jitter": (0.0, 0.45)},
    ("gate_generator", "voronoi"): {"voronoi_radial_variation": (0.0, 0.85),
                                    "voronoi_angular_jitter": (0.0, math.pi)},
    ("gate_generator", "checkpoint"): {"checkpoint_radius_min_frac": (0.0, 1.0),
                                       "checkpoint_angle_jitter": (0.0, 1.0)},
    ("elevation", "uniform"): {"z_min": (None, None), "z_max": (None, None)},
    ("elevation", "random_walk"): {"z_min": (None, None), "z_max": (None, None),
                                   "z_max_step": (0.0, None)},
    ("elevation", "noise"): {"z_min": (None, None), "z_max": (None, None),
                             "z_noise_amplitude": (0.0, None)},
}
#: Units without any dynamic field.
STATIC_ONLY = [("generator", "repulsive"), ("gate_generator", "bezier"),
               ("gate_generator", "hull"), ("elevation", "flat")]

#: Per generator: each knob's four values inside its effective range (rad below the
#: default clamp 0.4, voronoi_angular_jitter in its useful [0, 0.5]).
GEN_KNOBS = {
    "bezier": {"rad": (0.1, 0.2, 0.3, 0.35), "handle_clamp_frac": (0.2, 0.3, 0.5, 0.9)},
    "hull": {"hull_displacement": (0.05, 0.1, 0.3, 0.5)},
    "polar": {"polar_radial_jitter": (0.2, 0.4, 0.7, 0.85),
              "polar_angular_jitter": (0.05, 0.15, 0.35, 0.45)},
    "voronoi": {"voronoi_radial_variation": (0.1, 0.3, 0.5, 0.8),
                "voronoi_angular_jitter": (0.0, 0.15, 0.3, 0.45)},
    "checkpoint": {"checkpoint_radius_min_frac": (0.1, 0.2, 0.5, 0.7),
                   "checkpoint_angle_jitter": (0.1, 0.3, 0.75, 0.9),
                   "checkpoint_turn_rate": (0.25, 0.35, 0.5, 0.8)},
}
#: The non-default static settings of each generator (tests/test_generator_params.py).
GEN_STATIC = {
    "bezier": dict(min_num_points=7, max_num_points=11, edgy=0.2),
    "hull": dict(min_num_points=8),
    "polar": dict(polar_num_knots=9),
    "voronoi": dict(voronoi_control_points=14, voronoi_site_layout="mixed"),
    "checkpoint": dict(checkpoint_count=10, checkpoint_best_of_k=2),
}
GATE_KNOBS = {
    "polar": {"polar_radial_jitter": (0.1, 0.3, 0.45, 0.8),
              "polar_angular_jitter": (0.05, 0.15, 0.4, 0.45)},
    "checkpoint": {"checkpoint_radius_min_frac": (0.1, 0.2, 0.5, 0.7),
                   "checkpoint_angle_jitter": (0.1, 0.3, 0.75, 0.9)},
    "voronoi": {"voronoi_radial_variation": (0.1, 0.3, 0.5, 0.8),
                "voronoi_angular_jitter": (0.0, 0.15, 0.3, 0.45)},
}
#: The elevation profiles (flat config fields / stage params) and their per-env values
#: (no combination crosses the window).
ELEVATIONS = {
    "uniform": (dict(z_min=-0.2, z_max=0.3),
                {"z_min": (-0.4, -0.3, -0.2, -0.1), "z_max": (0.2, 0.3, 0.5, 0.7)}),
    "random_walk": (dict(z_base=0.05, z_min=-0.2, z_max=0.3, z_max_step=0.4,
                         z_control_points=7),
                    {"z_min": (-0.4, -0.3, -0.2, -0.1), "z_max": (0.2, 0.3, 0.5, 0.7),
                     "z_max_step": (0.1, 0.2, 0.5, 0.8)}),
    "noise": (dict(z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.2,
                   z_noise_harmonics=4),
              {"z_min": (-0.5, -0.3, -0.2, -0.1), "z_max": (0.15, 0.25, 0.4, 0.6),
               "z_noise_amplitude": (0.05, 0.15, 0.3, 0.6)}),
}
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                "valid", "count", "winding", "truncated")
_LEGACY = {
    "bezier": (wg_bezier, "bezier_alloc_scratch", "generate_centerline_warp"),
    "hull": (wg_hull, "hull_alloc_scratch", "generate_hull_warp"),
    "polar": (wg_polar, "polar_alloc_scratch", "generate_polar_warp"),
    "voronoi": (wg_voronoi, "voronoi_alloc_scratch", "generate_voronoi_warp"),
    "checkpoint": (wg_checkpoint, "checkpoint_alloc_scratch", "generate_checkpoint_warp"),
}


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _seeds(seeds=SEEDS) -> wp.array:
    return wp.array(np.asarray(seeds, np.int32), dtype=wp.int32, device=DEV)


def _rng(seeds=SEEDS) -> PerEnvSeededRNG:
    return PerEnvSeededRNG(seeds=_seeds(seeds), num_envs=len(seeds), device=DEV)


def _f32(values) -> wp.array:
    return wp.array(np.asarray(values, np.float32), dtype=wp.float32, device=DEV)


def _per_env(values, shift: int = 0) -> np.ndarray:
    """Env ``e`` takes ``values[(e + shift) % len(values)]``, as float32."""
    return np.array([values[(e + shift) % len(values)] for e in range(E)], np.float32)


def _snap(result) -> dict:
    return {k: np.array(a.numpy(), copy=True) for k, a in result_arrays(result).items()}


def _row(arr: np.ndarray, e: int) -> bytes:
    return np.ascontiguousarray(arr.reshape(E, -1)[e]).tobytes()


def _assert_row(got: dict, ref: dict, e: int, label: str) -> None:
    bad = [k for k in ref if _row(got[k], e) != _row(ref[k], e)]
    assert not bad, f"{label}: env {e} differs in {bad}"


def _differs(a: dict, b: dict) -> bool:
    return any(x.tobytes() != b[k].tobytes() for k, x in a.items())


def _tcfg(**kw) -> TrackGenConfig:
    return TrackGenConfig(device=DEV, num_envs=E, **kw)


def _gcfg(**kw) -> GateGenConfig:
    return GateGenConfig(device=DEV, num_envs=E, **kw)


def _set_all(table, sid: str, sets: dict) -> list:
    """``table[sid].set(field, [E] array)`` for each field; keeps the sources alive."""
    keep = []
    for f, arr in sets.items():
        src = _f32(arr)
        keep.append(src)
        table[sid].set(f, src)
    return keep


def _check_per_env(make, sid: str, knobs: dict, label: str, *, vacuity=True) -> None:
    """``make(**scalars)`` builds a generator; its per-env run (every knob of ``knobs``
    set per env, each its own permutation) equals, env by env, a run built with that
    env's values as scalars; the live arrays read back unwritten; every combination
    moves the batch (non-vacuity)."""
    gen = make()
    sets = {f: _per_env(v, k) for k, (f, v) in enumerate(knobs.items())}
    keep = _set_all(gen.params, sid, sets)
    got = _snap(gen.generate())
    for f, arr in sets.items():
        assert gen.params[sid][f].numpy().tobytes() == arr.tobytes(), \
            f"{label}: the stage wrote its live {f} array"
    del keep
    default = _snap(make().generate())
    refs: dict = {}
    for e in range(E):
        combo = tuple((f, float(arr[e])) for f, arr in sets.items())
        if combo not in refs:
            refs[combo] = _snap(make(**dict(combo)).generate())
            if vacuity:
                assert _differs(refs[combo], default), \
                    f"{label}: {combo} does not change the batch (vacuous)"
        _assert_row(got, refs[combo], e, f"{label} {dict(combo)}")


# ---------------------------------------------------------------------------------------
# 1. Declarations
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("unit", sorted(DYNAMIC) + STATIC_ONLY,
                         ids=[f"{k}.{n}" for k, n in sorted(DYNAMIC) + STATIC_ONLY])
def test_the_dynamic_set_and_its_bounds(unit):
    kind, name = unit
    cls = registry.get(kind, name).Params
    want = DYNAMIC.get(unit, {})
    got = {f.name: (f.lo, f.hi) for f in _fields(cls) if f.kind == "dynamic"}
    assert got == want, f"{kind}.{name}: {got}"
    for f in _fields(cls):
        if f.kind == "dynamic":
            assert f.dtype is wp.float32, f.name
    schema = registry.describe(kind, name).params_schema
    for field, prop in schema.get("properties", {}).items():
        if field in want:
            lo, hi = want[field]
            assert prop["x-dynamic"] is True and prop["x-dtype"] == "float32", field
            assert prop.get("minimum") == lo and prop.get("maximum") == hi, (field, prop)
            # The bounds compare in float32 (a read-back value at a bound is in range).
            assert prop.get("x-bounds") == ("float32" if (lo, hi) != (None, None)
                                             else None), (field, prop)
        else:
            assert prop.get("x-static") is True, field


def test_checkpoint_steering_internals_and_edgy_stay_static():
    static = {f.name for f in _fields(bp.CheckpointParams) if f.kind == "static"}
    assert {"checkpoint_steer_gain", "checkpoint_lookahead_frac", "checkpoint_count",
            "checkpoint_best_of_k", "checkpoint_clip_fallback"} <= static
    bez = {f.name: f.kind for f in _fields(bp.BezierParams)}
    assert bez["edgy"] == "static" and bez["style_sampling"] == "static"
    assert all(f.kind == "static" for f in _fields(bp.RepulsiveParams))


@pytest.mark.parametrize("name", sorted(GEN_KNOBS))
def test_the_builtins_are_stage_classes_with_a_deprecated_params_alias(name):
    cls = registry.get("generator", name)
    assert _is_stage(cls) and cls.kind == "generator" and cls.output == "centerline"
    assert cls.capture is plugin.CaptureClass.STATIC
    with pytest.warns(DeprecationWarning,
                      match=rf"registry\.get\('generator', '{name}'\)\.params is "
                            rf"deprecated: use \.Params \(removed in 1\.0\)") as rec:
        assert cls.params is cls.Params
    assert rec[0].filename == __file__, "the warning is attributed to the caller"
    info = registry.describe("generator", name)
    assert info.builtin and info.capturable and info.accepts_ctx


def test_no_params_alias_on_the_other_stages():
    for kind, name in (("generator", "repulsive"), ("gate_generator", "polar"),
                       ("elevation", "noise")):
        assert not hasattr(registry.get(kind, name), "params"), (kind, name)


# ---------------------------------------------------------------------------------------
# 2. Each twin == its scalar kernel, on generator-produced inputs
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Twin:
    module: object
    scalar: str
    twin: str
    config: dict
    knobs: dict                         # scalar argument index -> (field, lo, hi)
    bounds_after: "int | None" = None   # _assemble_dyn_k: 4 bound scalars after this arg


_ZU = dict(z_profile="uniform", z_min=-0.2, z_max=0.3, z_valid_grade=0.5)
_ZW = dict(z_profile="random_walk", z_base=0.05, z_min=-0.2, z_max=0.3, z_max_step=0.4,
           z_valid_grade=0.8, z_control_points=7)
_ZN = dict(z_profile="noise", z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.2,
           z_noise_harmonics=4, z_valid_grade=0.6)

TWINS = {
    "assemble": _Twin(wg_bezier, "_assemble_k", "_assemble_dyn_k", dict(generator="bezier"),
                      {6: ("rad", 0.0, 1.0), 7: ("handle_clamp_frac", 0.0, None)},
                      bounds_after=7),
    "midpoint_displace": _Twin(wg_hull, "_midpoint_displace_k", "_midpoint_displace_dyn_k",
                               dict(generator="hull"), {4: ("hull_displacement", 0.0, 1.0)}),
    "polar_controls": _Twin(wg_polar, "_polar_controls_k", "_polar_controls_dyn_k",
                            dict(generator="polar"),
                            {2: ("polar_radial_jitter", 0.0, 0.85),
                             3: ("polar_angular_jitter", 0.0, 0.45)}),
    "sample_checkpoints": _Twin(wg_checkpoint, "_sample_checkpoints_k",
                                "_sample_checkpoints_dyn_k", dict(generator="checkpoint"),
                                {3: ("checkpoint_radius_min_frac", 0.0, 1.0),
                                 4: ("checkpoint_angle_jitter", 0.0, 1.0)}),
    "steer": _Twin(wg_checkpoint, "_steer_k", "_steer_dyn_k", dict(generator="checkpoint"),
                   {4: ("checkpoint_turn_rate", 0.0, math.pi)}),
    "select_anchor_sites": _Twin(wg_voronoi, "_select_anchor_sites_k",
                                 "_select_anchor_sites_dyn_k", dict(generator="voronoi"),
                                 {6: ("voronoi_radial_variation", 0.0, 0.85),
                                  7: ("voronoi_angular_jitter", 0.0, math.pi)}),
    "z_uniform": _Twin(warp_zprofile, "_z_uniform_k", "_z_uniform_dyn_k", _ZU,
                       {3: ("z_min", None, None), 4: ("z_max", None, None)}),
    "z_walk": _Twin(warp_zprofile, "_z_walk_k", "_z_walk_dyn_k", _ZW,
                    {6: ("z_min", None, None), 7: ("z_max", None, None),
                     8: ("z_max_step", 0.0, None)}),
    "z_noise": _Twin(warp_zprofile, "_z_noise_k", "_z_noise_dyn_k", _ZN,
                     {6: ("z_min", None, None), 7: ("z_max", None, None),
                      8: ("z_noise_amplitude", 0.0, None)}),
}


def _legacy_track(cfg, seeds=SEEDS):
    """The PR9 composition with the frozen legacy spec (the scalar kernels) for ``cfg``
    on fresh buffers; returns ``(track, scratch)``."""
    track, scratch = warp_pipeline._inflate_warp_alloc(cfg)
    assert not _is_stage(scratch.generator_spec)
    warp_pipeline._run_pipeline(cfg, _seeds(seeds), out=track, scratch=scratch,
                                ctx=LaunchContext.for_call(DEV))
    return track, scratch


def _capture(monkeypatch, config: dict, kernel) -> tuple:
    """``(dim, args)`` of the first launch of ``kernel`` in the PR9 composition of
    ``TrackGenConfig(**config)`` (array arguments cloned right before it)."""
    real = LaunchContext.launch
    got: list = []

    def spy(self, k, dim, inputs=(), outputs=(), **kw):
        if k is kernel and not got:
            got.append((dim, [wp.clone(a) if isinstance(a, wp.array) else a
                              for a in (*inputs, *outputs)]))
        return real(self, k, dim, inputs, outputs, **kw)

    with monkeypatch.context() as m:
        m.setattr(LaunchContext, "launch", spy)
        _legacy_track(_tcfg(**config))
    assert got, f"the PR9 composition of {config} never launched {kernel.key}"
    return got[0]


def _launch(kernel, dim, args) -> list:
    fresh = [wp.clone(a) if isinstance(a, wp.array) else a for a in args]
    wp.launch(kernel, dim=dim, inputs=fresh, device=DEV)
    return [np.array(a.numpy(), copy=True) if isinstance(a, wp.array) else None
            for a in fresh]


class _Harness:
    def __init__(self, pair: _Twin, monkeypatch) -> None:
        self.pair = pair
        self.scalar_k = getattr(pair.module, pair.scalar)
        self.twin_k = getattr(pair.module, pair.twin)
        self.dim, self.args = _capture(monkeypatch, pair.config, self.scalar_k)
        self.arrays = [i for i, a in enumerate(self.args) if isinstance(a, wp.array)]

    def scalar(self, values: dict) -> list:
        args = list(self.args)
        for i, v in values.items():
            args[i] = float(v)
        out = _launch(self.scalar_k, self.dim, args)
        return [out[i] for i in self.arrays]

    def twin(self, arrays: dict, bounds=None) -> list:
        knob_arrays = {i: _f32(arrays.get(i, np.full(E, float(self.args[i]), np.float32)))
                       for i in self.pair.knobs}
        args = [knob_arrays.get(i, a) for i, a in enumerate(self.args)]
        after = self.pair.bounds_after
        if after is not None:
            if bounds is None:
                bounds = []
                for i in sorted(self.pair.knobs):
                    _, lo, hi = self.pair.knobs[i]
                    bounds += [-INF if lo is None else lo, INF if hi is None else hi]
            args = args[:after + 1] + [float(b) for b in bounds] + args[after + 1:]
        out = _launch(self.twin_k, self.dim, args)
        shift = (lambda i: i if after is None or i <= after else i + 4)
        for i, arr in knob_arrays.items():
            assert out[shift(i)].tobytes() == arr.numpy().tobytes(), "a knob array written"
        return [out[shift(i)] for i in self.arrays]


def _clamped(v, lo, hi) -> float:
    x = np.float32(v)
    if lo is not None and x < np.float32(lo):
        x = np.float32(lo)
    if hi is not None and x > np.float32(hi):
        x = np.float32(hi)
    return float(x)


def _same(a: list, b: list) -> bool:
    return all(x.tobytes() == y.tobytes() for x, y in zip(a, b))


def _rows_equal(a: list, b: list, e: int) -> bool:
    return all(_row(x, e) == _row(y, e) for x, y in zip(a, b) if x.shape[0] % E == 0)


@pytest.mark.parametrize("pair", sorted(TWINS))
def test_each_twin_equals_its_scalar_kernel(pair, monkeypatch):
    h = _Harness(TWINS[pair], monkeypatch)
    for i, (field, lo, hi) in h.pair.knobs.items():
        d = float(h.args[i])
        if lo is not None and hi is not None:
            inside = [lo + 0.2 * (hi - lo), lo + 0.7 * (hi - lo), lo, hi]
        elif lo is not None:
            inside = [lo + 0.15, lo + 0.9, lo, 1.0e9]
        else:
            inside = [d - 0.35, d + 0.45, -50.0, 50.0]
        inside = [d] + inside + [-0.0, math.nan]
        # (a) uniform in-range arrays (and -0.0, NaN) == the scalar kernel at the value
        for v in inside:
            assert _same(h.twin({i: np.full(E, v, np.float32)}), h.scalar({i: v})), \
                f"{pair}.{field} = {v}"
        # (b) out of range == the scalar kernel at the bound
        outside = ([lo - 0.3, -INF, lo - 1.0e-3] if lo is not None else []) + \
            ([hi + 0.4, INF, hi + 1.0e-3] if hi is not None else [])
        for v in outside:
            assert _same(h.twin({i: np.full(E, v, np.float32)}),
                         h.scalar({i: _clamped(v, lo, hi)})), f"{pair}.{field} = {v}"
        # (c) per env (interior, bound, out-of-range values) == row by row the scalar
        # kernel at each env's clamped value
        vals = (inside[1:5] + outside)[:E] if outside else inside[:E]
        per_env = _per_env(vals)
        got = h.twin({i: per_env})
        for e in range(E):
            want = h.scalar({i: _clamped(per_env[e], lo, hi)})
            assert _rows_equal(got, want, e), f"{pair}.{field} per env: env {e}"


def test_polar_twin_clamp_equals_polar_jitters(monkeypatch):
    """The comparison clamp of ``_polar_controls_dyn_k`` == ``polar_jitters()``'s host
    clamp + the scalar kernel, on the bounds' float32 neighbours, signed zeros,
    subnormals, infinities and NaN (design E8a)."""
    h = _Harness(TWINS["polar_controls"], monkeypatch)
    vals = set()
    for b in (0.85, 0.45, 0.0, 0.6, 0.3):
        f = np.float32(b)
        lo, hi = np.nextafter(f, np.float32(-INF)), np.nextafter(f, np.float32(INF))
        vals |= {b, float(f), float(lo), float(hi), math.nextafter(b, 1.0),
                 math.nextafter(b, -1.0)}
    vals |= {-0.0, 1e-45, -1e-45, 1e-39, -1e-39, INF, -INF, math.nan, 1e30, -1e30}
    vals = sorted(vals, key=lambda v: (math.isnan(v), v))
    for radial in vals:
        for angular in (0.3, 0.45, 0.4500000001, 0.46, -0.0, -1e-46, math.nan):
            rj, aj = wg_polar.polar_jitters(types.SimpleNamespace(
                polar_radial_jitter=radial, polar_angular_jitter=angular))
            want = h.scalar({2: rj, 3: aj})
            got = h.twin({2: np.full(E, radial, np.float32),
                          3: np.full(E, angular, np.float32)})
            assert _same(got, want), (radial, angular)


def test_assemble_twin_unbounded_equals_the_style_kernel(monkeypatch):
    """``_assemble_dyn_k`` with ``(-inf, inf)`` bounds (a knob ``style_sampling`` draws)
    == ``_assemble_style_k`` on out-of-range per-env draws (their ranges were never
    bounded)."""
    h = _Harness(TWINS["assemble"], monkeypatch)
    rad = np.array([-0.3, 1.7, 0.2, math.nan, 0.5, -0.0, 3.0, 0.4], np.float32)
    clamp = np.array([-0.5, 2.0, 0.3, 0.0, INF, 0.7, -0.0, 1e9], np.float32)
    style = list(h.args)
    style[6], style[7] = _f32(rad), _f32(clamp)
    want = [x for i, x in enumerate(_launch(wg_bezier._assemble_style_k, h.dim, style))
            if i in h.arrays]
    got = h.twin({6: rad, 7: clamp}, bounds=(-INF, INF, -INF, INF))
    assert _same(got, want)


# ---------------------------------------------------------------------------------------
# 3. / 4. Generators: per env == scalar runs; Stage == the frozen legacy function
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("statics", ["defaults", "non_defaults"])
@pytest.mark.parametrize("name", sorted(GEN_KNOBS))
def test_generator_per_env_knobs_equal_scalar_runs(name, statics):
    base = dict(generator=name, **(GEN_STATIC[name] if statics == "non_defaults" else {}))

    def make(**scalars):
        return TrackGenerator(_tcfg(**base, **scalars), _rng())

    _check_per_env(make, "generator", GEN_KNOBS[name], f"{name}/{statics}")


_ALL_KNOBS = [("generator", g, f, v[-1]) for g, kn in GEN_KNOBS.items() for f, v in kn.items()]


@pytest.mark.parametrize("sid,name,field,value", _ALL_KNOBS,
                         ids=[f"{g}.{f}" for _, g, f, _ in _ALL_KNOBS])
def test_each_generator_knob_moves_the_batch(sid, name, field, value):
    """Non-vacuity, one knob at a time: a ``set`` of the knob changes the batch, and
    the run equals one constructed with that value."""
    gen = TrackGenerator(_tcfg(generator=name), _rng())
    default = _snap(gen.generate())
    gen.params[sid].set(field, value)
    moved = _snap(gen.generate())
    assert _differs(moved, default), f"{name}.{field}={value} does not move the batch"
    built = _snap(TrackGenerator(_tcfg(generator=name, **{field: value}), _rng()).generate())
    assert not _differs(moved, built)


@pytest.mark.parametrize("variant", ["default", "non_default"])
@pytest.mark.parametrize("name", sorted(GEN_KNOBS))
def test_stage_raw_output_equals_the_frozen_legacy_function(name, variant):
    kw = dict(generator=name)
    if variant == "non_default":
        kw.update(GEN_STATIC[name])
        kw.update({f: v[2] for f, v in GEN_KNOBS[name].items()})
    cfg = _tcfg(**kw)
    gen = TrackGenerator(cfg, _rng())
    gen.generate()
    raw = gen.program.tap("raw.xy").numpy().tobytes()
    valid = gen.program.tap("gen.valid").numpy().tobytes()
    mod, alloc, generate = _LEGACY[name]
    scratch = getattr(mod, alloc)(cfg)
    out = wp.empty(E * int(cfg.num_points), dtype=wp.vec2f, device=DEV)
    ok = wp.empty(E, dtype=wp.int32, device=DEV)
    getattr(mod, generate)(_seeds(), cfg, out, ok, scratch, ctx=LaunchContext.for_call(DEV))
    assert out.numpy().tobytes() == raw and ok.numpy().tobytes() == valid


# ---------------------------------------------------------------------------------------
# 5. / 6. Gate generators and elevation profiles per env
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("ordering", ["ccw", "raw"])
@pytest.mark.parametrize("name", sorted(GATE_KNOBS))
def test_gate_generator_per_env_knobs_equal_scalar_runs(name, ordering):
    def make(**scalars):
        return GateGenerator(_gcfg(generator=name, gate_ordering=ordering, **scalars),
                             _rng())

    _check_per_env(make, "generator", GATE_KNOBS[name], f"gates {name}/{ordering}")


@pytest.mark.parametrize("where", ["track", "gates", "stations"])
@pytest.mark.parametrize("profile", sorted(ELEVATIONS))
def test_elevation_per_env_knobs_equal_scalar_runs(profile, where):
    params, knobs = ELEVATIONS[profile]

    if where == "track":
        def make(**scalars):
            return TrackGenerator(_tcfg(z_profile=profile, **dict(params, **scalars)),
                                  _rng())
    elif where == "gates":
        gate_params = {k: v for k, v in params.items() if k != "z_control_points"}

        def make(**scalars):
            return GateGenerator(_gcfg(z_profile=profile, gate_width=0.04,
                                       **dict(gate_params, **scalars)), _rng())
    else:
        def make(**scalars):
            return GateGenerator(presets.gates(
                num_envs=E, device=DEV, generator="bezier", source="relaxed_stations",
                gates=12, elevation=(profile, dict(params, **scalars)),
                elevation_on="centerline"), _rng())

    _check_per_env(make, "elevation", knobs, f"elevation.{profile}/{where}")


# ---------------------------------------------------------------------------------------
# 7. randomize on built-in knobs
# ---------------------------------------------------------------------------------------

_TRACK_RANDOMIZE = {
    "bezier": (dict(generator="bezier"), {"generator.rad": (0.1, 0.4)}),
    "bezier.rad+clamp": (dict(generator="bezier"),
                         {"generator.rad": (0.1, 0.6),
                          "generator.handle_clamp_frac": (0.15, 0.9)}),
    "hull": (dict(generator="hull"), {"generator.hull_displacement": (0.05, 0.5)}),
    "polar": (dict(generator="polar"), {"generator.polar_radial_jitter": (0.2, 0.8),
                                        "generator.polar_angular_jitter": (0.05, 0.45)}),
    "voronoi": (dict(generator="voronoi"),
                {"generator.voronoi_radial_variation": (0.2, 0.85),
                 "generator.voronoi_angular_jitter": (0.0, 0.3)}),
    "checkpoint": (dict(generator="checkpoint"),
                   {"generator.checkpoint_radius_min_frac": (0.1, 0.6),
                    "generator.checkpoint_angle_jitter": (0.2, 0.9),
                    "generator.checkpoint_turn_rate": (0.25, 0.8)}),
    "elevation.uniform": (dict(elevation=("uniform", {"z_min": -0.2, "z_max": 0.3})),
                          {"elevation.z_min": (-0.5, -0.2), "elevation.z_max": (0.3, 0.8)}),
    "elevation.random_walk": (dict(elevation=("random_walk", dict(ELEVATIONS["random_walk"][0]))),
                              {"elevation.z_max_step": (0.1, 0.8)}),
    "elevation.noise": (dict(elevation=("noise", dict(ELEVATIONS["noise"][0]))),
                        {"elevation.z_noise_amplitude": (0.05, 0.5)}),
}
_GATE_RANDOMIZE = {
    "polar": (dict(generator="polar"), {"generator.polar_radial_jitter": (0.2, 0.8),
                                        "generator.polar_angular_jitter": (0.05, 0.45)}),
    "checkpoint": (dict(generator="checkpoint"),
                   {"generator.checkpoint_radius_min_frac": (0.1, 0.6),
                    "generator.checkpoint_angle_jitter": (0.2, 0.9)}),
    "voronoi": (dict(generator="voronoi"),
                {"generator.voronoi_radial_variation": (0.2, 0.85),
                 "generator.voronoi_angular_jitter": (0.0, 0.3)}),
    "polar+elevation.noise": (dict(generator="polar", elevation=(
        "noise", {"z_base": 0.1, "z_min": -0.5, "z_max": 1.5, "z_noise_amplitude": 0.5})),
        {"generator.polar_radial_jitter": (0.2, 0.8),
         "elevation.z_noise_amplitude": (0.1, 0.6)}),
}


def _assert_randomized(facade, preset, kw: dict, randomize: dict) -> None:
    spec = preset(num_envs=E, device=DEV, randomize=randomize, **kw)
    for key in randomize:
        sid = key.rsplit(".", 1)[0]
        assert spec.ids.index(f"randomize.{key}") < spec.ids.index(sid), spec.ids
    gen = facade(spec, _rng())
    twin = facade(preset(num_envs=E, device=DEV, **kw), _rng())
    keys = [tuple(k.rsplit(".", 1)) for k in randomize]

    def sync_twin(**adv):
        for sid, f in keys:
            twin.params[sid].set(f, gen.params[sid][f])
        return twin.generate(**adv)

    got = _snap(gen.generate())
    draws = {k: gen.params[sid][f].numpy().copy() for k, (sid, f) in zip(randomize, keys)}
    for k, (lo, hi) in randomize.items():
        d = draws[k]
        assert np.all((d >= np.float32(lo)) & (d <= np.float32(hi))), (k, d)
        assert len(set(d.tolist())) > 1, f"{k}: the draws do not vary per env"
    assert not _differs(got, _snap(sync_twin())), "randomized != its twin at the draws"
    plain = _snap(facade(preset(num_envs=E, device=DEV, **kw), _rng()).generate())
    assert _differs(got, plain), "the draws do not move the batch (vacuous)"
    # A masked advance: masked envs redraw (new effective seed), the others keep their
    # rows and draws.
    mask_np = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
    mask = wp.array(mask_np, dtype=wp.int32, device=DEV)
    got2 = _snap(gen.generate(advance=True, mask=mask))
    for k, (sid, f) in zip(randomize, keys):
        now = gen.params[sid][f].numpy()
        keep = mask_np == 0
        assert now[keep].tobytes() == draws[k][keep].tobytes(), f"{k}: unmasked redrawn"
        assert (now[~keep] != draws[k][~keep]).any(), f"{k}: masked envs not redrawn"
    for e in np.flatnonzero(mask_np == 0):
        _assert_row(got2, got, int(e), "unmasked env kept")
    twin2 = _snap(sync_twin(advance=True, mask=wp.clone(mask)))
    assert not _differs(got2, twin2), "masked advance: randomized != its twin"


@pytest.mark.parametrize("case", sorted(_TRACK_RANDOMIZE))
def test_track_randomize_on_builtin_knobs(case):
    kw, randomize = _TRACK_RANDOMIZE[case]
    _assert_randomized(TrackGenerator, presets.track, kw, randomize)


@pytest.mark.parametrize("case", sorted(_GATE_RANDOMIZE))
def test_gate_randomize_on_builtin_knobs(case):
    kw, randomize = _GATE_RANDOMIZE[case]
    _assert_randomized(GateGenerator, presets.gates, kw, randomize)


def test_randomize_plan_rules_on_builtin_knobs():
    with pytest.raises(PlanError, match=r"^generator\.edgy is a static field \(build a "
                                        r"new generator to change it"):
        build(presets.track(num_envs=E, device=DEV, randomize={"generator.edgy": (0.0, 1.0)}))
    with pytest.raises(PlanError, match=r"the draw range \[0\.2, 0\.9\] lies outside "
                                        r"generator\.polar_radial_jitter's declared range "
                                        r"\[0\.0, 0\.85\]"):
        build(presets.track(num_envs=E, device=DEV, generator="polar",
                            randomize={"generator.polar_radial_jitter": (0.2, 0.9)}))
    # A range whose hi is the float32 value of the declared bound (a read-back value) is
    # accepted (bounds compare in float32).
    hi = float(np.float32(0.85))
    assert hi > 0.85
    build(presets.track(num_envs=E, device=DEV, generator="polar",
                        randomize={"generator.polar_radial_jitter": (0.2, hi)}))
    pi32 = float(np.float32(math.pi))
    build(presets.track(num_envs=E, device=DEV, generator="checkpoint",
                        randomize={"generator.checkpoint_turn_rate": (0.1, pi32)}))


def test_with_randomize_is_the_presets_insertion():
    rnd = {"generator.polar_radial_jitter": (0.2, 0.8), "elevation.z_max": (0.3, 0.8)}
    elev = ("uniform", {"z_min": -0.2, "z_max": 0.3})
    want = presets.track(num_envs=E, device=DEV, generator="polar", elevation=elev,
                         randomize=rnd)
    base = presets.track(num_envs=E, device=DEV, generator="polar", elevation=elev)
    assert base.with_randomize(rnd) == want
    assert base.with_randomize(None) == base
    refs = [StageRef("randomize", "uniform", params={"stage": "generator",
                                                    "field": "polar_radial_jitter",
                                                    "lo": 0.2, "hi": 0.8}, id="r")]
    assert base.with_randomize(refs).ids[0] == "r"
    with pytest.raises(ValueError, match=r"names stage 'nope', which this pipeline does "
                                         r"not have"):
        base.with_randomize({"nope.x": (0.0, 1.0)})
    with pytest.raises(TypeError, match=r"must be a \(lo, hi\) pair"):
        base.with_randomize({"generator.rad": 0.3})
    # A config-built pipeline gets randomizers the same way.
    cfg = _tcfg(generator="polar")
    spec = legacy.to_pipeline_spec(cfg).with_randomize(
        {"generator.polar_radial_jitter": (0.2, 0.8)})
    gen = TrackGenerator(spec, _rng())
    assert gen.params["generator"].drawn == {
        "polar_radial_jitter": "randomize.generator.polar_radial_jitter"}
    # A field already randomized is refused (not a duplicate-id PlanError at build).
    for again in ({"generator.polar_radial_jitter": (0.3, 0.7)},
                  [StageRef("randomize", "uniform",
                            params={"stage": "generator", "field": "polar_radial_jitter",
                                    "lo": 0.3, "hi": 0.7}, id="r2")]):
        with pytest.raises(ValueError, match=r"randomize: 'generator\.polar_radial_jitter' "
                                             r"is already drawn by stage "
                                             r"'randomize\.generator\.polar_radial_jitter'; "
                                             r"replace it, spec\.replace\("):
            spec.with_randomize(again)
    # ... and the documented replacement builds.
    rid = "randomize.generator.polar_radial_jitter"
    build(spec.replace(rid, StageRef("randomize", "uniform",
                                     params={"stage": "generator",
                                             "field": "polar_radial_jitter",
                                             "lo": 0.3, "hi": 0.7})))
    # The gate route: a GateGenConfig's spec (to_pipeline_spec refuses the config).
    gcfg = _gcfg(generator="polar")
    with pytest.raises(TypeError, match=r"legacy\.to_pipeline_spec takes a TrackGenConfig; "
                                        r"use legacy\.to_gate_pipeline_spec"):
        legacy.to_pipeline_spec(gcfg)
    gates = GateGenerator(legacy.to_gate_pipeline_spec(gcfg).with_randomize(
        {"generator.polar_radial_jitter": (0.2, 0.8)}), _rng())
    assert dict(gates.params["generator"].drawn) == {
        "polar_radial_jitter": "randomize.generator.polar_radial_jitter"}


# ---------------------------------------------------------------------------------------
# 8. Cross-field draw rules and inert windows
# ---------------------------------------------------------------------------------------

_WINDOW = ("uniform", {"z_min": -0.2, "z_max": 0.3})


@pytest.mark.parametrize("randomize,match", [
    ({"elevation.z_min": (-0.5, 0.0), "elevation.z_max": (-0.1, 0.5)},
     r"stage 'elevation' \(elevation\.uniform\): the z_min draws \[-0\.5, 0\.0\] and "
     r"z_max draws \[-0\.1, 0\.5\] can cross"),
    ({"elevation.z_min": (-0.5, 0.4)}, r"the z_min draws \[-0\.5, 0\.4\] can exceed "
                                       r"z_max=0\.3"),
    ({"elevation.z_max": (-0.3, 0.5)}, r"the z_max draws \[-0\.3, 0\.5\] can fall below "
                                       r"z_min=-0\.2"),
])
def test_crossing_elevation_randomizers_are_plan_errors(randomize, match):
    with pytest.raises(PlanError, match=match):
        build(presets.track(num_envs=E, device=DEV, elevation=_WINDOW, randomize=randomize))


def test_non_crossing_elevation_randomizers_build():
    build(presets.track(num_envs=E, device=DEV, elevation=_WINDOW,
                        randomize={"elevation.z_min": (-0.5, -0.2),
                                   "elevation.z_max": (-0.2, 0.8)}))
    build(presets.track(num_envs=E, device=DEV, elevation=_WINDOW,
                        randomize={"elevation.z_max": (-0.2, 0.8)}))


def test_rad_drawn_above_the_clamp_warns():
    with pytest.warns(UserWarning, match=r"generator\.rad is drawn up to 0\.6, above "
                                         r"handle_clamp_frac=0\.4: rad has no effect") as rec:
        build(presets.track(num_envs=E, device=DEV, randomize={"generator.rad": (0.1, 0.6)}))
    assert rec[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        build(presets.track(num_envs=E, device=DEV, randomize={"generator.rad": (0.1, 0.4)}))
        build(presets.track(num_envs=E, device=DEV,
                            randomize={"generator.rad": (0.1, 0.6),
                                       "generator.handle_clamp_frac": (0.15, 0.9)}))
    # A drawn clamp entirely below the rad draws: rad is inert in every env.
    match = r"generator\.rad is drawn up to 0\.9, above the handle_clamp_frac draws' hi 0\.2"
    with pytest.warns(UserWarning, match=match):
        build(presets.track(num_envs=E, device=DEV,
                            randomize={"generator.rad": (0.5, 0.9),
                                       "generator.handle_clamp_frac": (0.1, 0.2)}))
    spec = legacy.to_pipeline_spec(_tcfg(style_sampling=True,
                                         handle_clamp_frac_range=(0.1, 0.2)))
    with pytest.warns(UserWarning, match=match):
        build(spec.with_randomize({"generator.rad": (0.5, 0.9)}))
    # A style rad range alone stays silent (style_sampling never warned).
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        build(legacy.to_pipeline_spec(_tcfg(style_sampling=True, rad_range=(0.5, 0.9))))


def _max_abs_z(spec) -> float:
    gen = TrackGenerator(spec, _rng())
    return float(np.nanmax(np.abs(gen.generate().center.numpy()[:, 2])))


@pytest.mark.parametrize("profile,knob", [("noise", "z_noise_amplitude"),
                                          ("random_walk", "z_max_step")])
def test_inert_window_warns_at_construction(profile, knob):
    with pytest.warns(UserWarning, match=rf"elevation\.{profile}: {knob}=0\.2 has no effect "
                                         rf"in the window z_min == z_max == 0\.0"):
        build(presets.track(num_envs=E, device=DEV, elevation=(profile, {knob: 0.2})))
    # The knob drawn by a randomizer in the default closed window: inert, and warned
    # when the pipeline is built (the window, not the initial value, decides).
    spec = presets.track(num_envs=E, device=DEV, elevation=profile,
                         randomize={f"elevation.{knob}": (0.1, 0.5)})
    with pytest.warns(UserWarning, match=rf"elevation\.{profile}: {knob} drawn in "
                                         rf"\[0\.1, 0\.5\] has no effect in the window "
                                         rf"z_min == z_max == 0\.0"):
        assert _max_abs_z(spec) == 0.0
    with pytest.warns(UserWarning, match=rf"elevation\.{profile}: {knob} drawn"):
        build(presets.gates(num_envs=E, device=DEV, generator="polar", elevation=profile,
                            randomize={f"elevation.{knob}": (0.1, 0.6)}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        build(presets.track(num_envs=E, device=DEV,
                            elevation=(profile, {knob: 0.2, "z_min": -0.3, "z_max": 0.3})))
        build(presets.track(num_envs=E, device=DEV, elevation=(profile, None)))
        # A window opened by z_min / z_max randomizers is open: no warning, and hills.
        spec = presets.track(num_envs=E, device=DEV, elevation=(profile, {knob: 0.2}),
                             randomize={"elevation.z_min": (-0.5, -0.2),
                                        "elevation.z_max": (0.2, 0.5)})
        assert _max_abs_z(spec) > 0.0
        spec = presets.track(num_envs=E, device=DEV, elevation=(profile, {knob: 0.2}),
                             randomize={"elevation.z_max": (0.2, 0.5)})
        assert _max_abs_z(spec) > 0.0
        # A drawn knob in an open window.
        build(presets.track(num_envs=E, device=DEV,
                            elevation=(profile, {"z_min": -0.3, "z_max": 0.3}),
                            randomize={f"elevation.{knob}": (0.1, 0.5)}))
        # A drawn knob that cannot be positive: nothing to warn about.
        build(presets.track(num_envs=E, device=DEV, elevation=profile,
                            randomize={f"elevation.{knob}": (0.0, 0.0)}))


# ---------------------------------------------------------------------------------------
# 9. Launch deltas (design section 4)
# ---------------------------------------------------------------------------------------

_STYLE3 = dict(style_sampling=True, rad_range=(0.1, 0.35), scale_range=(0.8, 1.2),
               handle_clamp_frac_range=(0.3, 0.9))
#: case -> (TrackGenConfig kwargs, the 0.2.0.dev5 PR13b count, the PR13c count).
LAUNCH_TABLE = {
    "bezier": (dict(generator="bezier"), 152, 154),
    "bezier_style_all": (_STYLE3, 153, 153),
    "bezier_style_rad": (dict(style_sampling=True, rad_range=(0.1, 0.35)), 153, 154),
    "bezier_style_clamp": (dict(style_sampling=True, handle_clamp_frac_range=(0.3, 0.9)),
                           153, 154),
    "bezier_style_none": (dict(style_sampling=True, scale_range=(0.8, 1.2)), 153, 155),
    "hull": (dict(generator="hull"), 151, 152),
    "voronoi": (dict(generator="voronoi"), 151, 153),
    "polar": (dict(generator="polar"), 143, 145),
    "checkpoint": (dict(generator="checkpoint"), 146, 149),
    "checkpoint_k1_clip": (dict(generator="checkpoint", checkpoint_best_of_k=1,
                                checkpoint_clip_fallback=True), 146, 149),
    "z_uniform": (dict(_ZU), 156, 160),
    "z_random_walk": (dict(_ZW), 156, 161),
    "z_noise": (dict(_ZN), 154, 159),
}


def _rows(trace) -> list:
    return [(r.op, r.kernel, r.dim, r.scalars) for r in trace]


def _style_fixups(rows: list, cfg) -> list:
    """The style path's records under PR13c: a ``None``-range knob's draw gets the
    ``0.0`` placeholders, and the legacy ``_assemble_style_k`` is ``_assemble_dyn_k``
    with ``(-inf, inf)`` bounds for a ranged knob, the declared ones otherwise."""
    if not cfg.style_sampling:
        return rows
    out = []
    rad, clamp = cfg.rad_range is not None, cfg.handle_clamp_frac_range is not None
    for op, kernel, dim, scalars in rows:
        name = kernel.rsplit(".", 1)[-1]
        if name == "_style_sample_k":
            zero = ({1, 2} if not rad else set()) | ({5, 6} if not clamp else set())
            scalars = tuple((i, 0.0 if i in zero else v) for i, v in scalars)
        elif name == "_assemble_style_k":
            kernel = kernel[:-len(name)] + "_assemble_dyn_k"
            scalars = tuple(scalars) + (
                (8, -INF if rad else 0.0), (9, INF if rad else 1.0),
                (10, -INF if clamp else 0.0), (11, INF))
        out.append((op, kernel, dim, scalars))
    return out


@pytest.mark.parametrize("case", sorted(LAUNCH_TABLE))
def test_launch_delta_against_the_legacy_composition(case):
    """One generation's trace == the legacy (scalar) PR9 composition's under the PR13c
    rule (latches after the effective seed, knob kernels renamed to their twins), with
    the design section 4 counts; masked: the latches are masked, right after the
    effective seed, before the hold (+3: advance, hold, commit)."""
    kw, head, pr13c = LAUNCH_TABLE[case]
    cfg = _tcfg(**kw)
    gen = TrackGenerator(cfg, _rng())
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as program:
        gen.program.enqueue(ctx)
    got = _rows(program)
    gen_latches = GENERATOR_LATCHES[cfg.generator]
    if cfg.style_sampling:
        gen_latches = (cfg.rad_range is None) + (cfg.handle_clamp_frac_range is None)
    n_latch = gen_latches + ELEVATION_LATCHES[cfg.z_profile]
    track, scratch = warp_pipeline._inflate_warp_alloc(cfg)
    seed = wp.empty(E, dtype=wp.int32, device=DEV)
    with trace_launches() as pr9:
        gen.seeds.effective(seed, None, ctx=ctx)
        warp_pipeline._run_pipeline(cfg, seed, out=track, scratch=scratch, ctx=ctx)
        gen.seeds._write_generation(track.generation, None, ctx)
    legacy_rows = _rows(pr9)
    if head is not None:
        assert len(legacy_rows) == head, f"{case}: HEAD emitted {len(legacy_rows)}"
        assert len(got) == pr13c == head + n_latch, (case, len(got))
    want = _style_fixups(pr13c_sequence(legacy_rows, n_latch, E), cfg)
    assert len(got) == len(want), f"{case}: {len(got)} vs {len(want)}"
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    assert first is None, f"{case}: record {first}: {got[first]} vs {want[first]}"
    assert [r[1] for r in got[1:1 + n_latch]] == [LATCH_ALL] * n_latch
    # masked
    mask = wp.array(np.arange(E) % 2, dtype=wp.int32, device=DEV)
    with trace_launches() as masked:
        gen.program.enqueue(ctx, advance=True, mask=mask)
    names = [r.kernel.rsplit(".", 1)[-1] for r in masked]
    assert len(masked) == len(got) + 3, case
    assert names[1] == "_effective_seed_masked_k", names[:4]
    assert [r.kernel for r in list(masked)[2:2 + n_latch]] == [LATCH_MASKED] * n_latch
    assert names[2 + n_latch] == "_hold_k" and names.count(LATCH_ALL.rsplit(".", 1)[1]) == 0


_GATE_TABLE = {
    "bezier/ccw/flat": 14, "polar/ccw/flat": 16, "checkpoint/ccw/flat": 16,
    "voronoi/ccw/flat": 17, "polar/raw/flat": 16, "hull/ccw/flat": 14,
}


@pytest.mark.parametrize("case", sorted(_GATE_TABLE) + ["polar/ccw/noise",
                                                        "voronoi/raw/noise",
                                                        "bezier/ccw/uniform",
                                                        "checkpoint/ccw/random_walk"])
def test_gate_launch_deltas(case):
    """The gate pipelines: the generator's latches (polar / checkpoint / voronoi: 2) and
    the elevation's (uniform 2, random_walk 3, noise 3) follow the effective seed; the
    knob kernels are the twins; the default pipeline keeps its 14 records; a case of
    the PR11 golden equals its recorded count plus the latches."""
    name, ordering, z = case.split("/")
    zkw = {"flat": {}, "uniform": dict(z_profile="uniform", z_max=1.5),
           "random_walk": dict(z_profile="random_walk", z_max=1.5, z_max_step=0.3),
           "noise": dict(z_profile="noise", z_base=0.1, z_max=1.5, z_noise_amplitude=0.5,
                         z_noise_harmonics=4)}[z]
    gen = GateGenerator(_gcfg(generator=name, gate_ordering=ordering, **zkw), _rng())
    with trace_launches() as t:
        gen.generate()
    names = [r.kernel.rsplit(".", 1)[-1] for r in t]
    n = GATE_GENERATOR_LATCHES[name] + ELEVATION_LATCHES[z]
    assert names[0] == "_effective_seed_k"
    assert names[1:1 + n] == ["_latch_all_f32_k"] * n and names[1 + n] != "_latch_all_f32_k"
    if case in _GATE_TABLE:
        assert len(t) == _GATE_TABLE[case]
    twin = {"polar": "_polar_controls_dyn_k", "checkpoint": "_sample_checkpoints_dyn_k",
            "voronoi": "_select_anchor_sites_dyn_k"}.get(name)
    if twin is not None:
        assert twin in names and twin.replace("_dyn_k", "_k") not in names
    if z != "flat":
        zk = {"uniform": "_z_uniform", "random_walk": "_z_walk", "noise": "_z_noise"}[z]
        assert f"{zk}_dyn_k" in names and f"{zk}_k" not in names


def _launch_cases():
    from tests.tools import capture_pr11_gate_stages as tool

    return tool.LAUNCH_CASES


@pytest.mark.parametrize("case", _launch_cases())
def test_gate_launch_counts_are_the_pr11_golden_plus_the_latches(case):
    from tests.tools import capture_pr11_gate_stages as tool

    golden = tool.load().get(tool.platform_key())
    if golden is None:
        pytest.skip(f"no dd181f0 gate-stage launches for {tool.platform_key()}")
    name, z = case.split("/")[0], case.split("/")[2]
    gen = GateGenerator(GateGenConfig(num_envs=tool.E, device=DEV, **tool.CASES[case]),
                        PerEnvSeededRNG(seeds=_seeds(SEEDS[:tool.E]), num_envs=tool.E,
                                        device=DEV))
    with trace_launches() as t:
        gen.generate()
    n = GATE_GENERATOR_LATCHES[name] + ELEVATION_LATCHES[z]
    assert len(t) == len(golden["launches"][case]) + n


def test_repulsive_and_the_default_gates_gain_no_latch():
    gen = TrackGenerator(_tcfg(generator="repulsive", num_points=64,
                               repulsive_stages=(16, 32, 64)), _rng())
    with trace_launches() as t:
        gen.generate()
    assert "_latch_all_f32_k" not in [r.kernel.rsplit(".", 1)[-1] for r in t]
    gates = GateGenerator(_gcfg(), _rng())
    with trace_launches() as t:
        gates.generate()
    assert len(t) == 14


# ---------------------------------------------------------------------------------------
# 10. Bounds and value types
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw,match", [
    (dict(generator="bezier", rad=1.2), r"BezierParams\.rad must be in \[0\.0, 1\.0\], "
                                        r"got 1\.2"),
    (dict(generator="bezier", rad=-0.1), r"BezierParams\.rad must be in \[0\.0, 1\.0\]"),
    (dict(generator="bezier", handle_clamp_frac=-0.5),
     r"BezierParams\.handle_clamp_frac must be in \[0\.0, inf\], got -0\.5"),
    (dict(generator="hull", hull_displacement=1.5),
     r"HullParams\.hull_displacement must be in \[0\.0, 1\.0\], got 1\.5"),
    # The polar break names the 0.2.0.dev4 clamp and the value giving its output.
    (dict(generator="polar", polar_radial_jitter=0.9),
     r"^polar_radial_jitter must be in \[0, 0\.85\], got 0\.9 \(0\.2\.0\.dev4 clamped it "
     r"to that range: set polar_radial_jitter=0\.85 for the same output\)$"),
    (dict(generator="polar", polar_angular_jitter=0.5),
     r"^polar_angular_jitter must be in \[0, 0\.45\], got 0\.5 \(0\.2\.0\.dev4 clamped "
     r"it to that range: set polar_angular_jitter=0\.45 for the same output\)$"),
    (dict(generator="polar", polar_radial_jitter=-0.1),
     r"^polar_radial_jitter must be in \[0, 0\.85\], got -0\.1 \(.*set "
     r"polar_radial_jitter=0\.0 for the same output\)$"),
    (dict(generator="polar", polar_radial_jitter=True),
     r"PolarParams\.polar_radial_jitter is dynamic and must be a real number, got True"),
    (dict(generator="voronoi", voronoi_radial_variation=0.9),
     r"VoronoiParams\.voronoi_radial_variation must be in \[0\.0, 0\.85\], got 0\.9"),
    (dict(generator="voronoi", voronoi_angular_jitter=3.5),
     r"VoronoiParams\.voronoi_angular_jitter must be in \[0\.0, 3\.14159"),
    (dict(generator="checkpoint", checkpoint_angle_jitter=1.2),
     r"CheckpointParams\.checkpoint_angle_jitter must be in \[0\.0, 1\.0\], got 1\.2"),
    (dict(generator="checkpoint", checkpoint_turn_rate=4.0),
     r"CheckpointParams\.checkpoint_turn_rate must be in \[0\.0, 3\.14159"),
    # The legacy validate() messages come first.
    (dict(generator="voronoi", voronoi_radial_variation=-0.1),
     r"^voronoi_radial_variation must be >= 0, got -0\.1$"),
    (dict(generator="checkpoint", checkpoint_radius_min_frac=1.0),
     r"^checkpoint_radius_min_frac must be in \[0, 1\), got 1\.0$"),
    # Bools are rejected.
    (dict(generator="bezier", rad=True),
     r"BezierParams\.rad is dynamic and must be a real number, got True"),
    (dict(generator="hull", hull_displacement="abc"),
     r"HullParams\.hull_displacement is dynamic and must be a real number, got 'abc'"),
])
def test_newly_rejected_values_raise_at_construction(kw, match):
    with pytest.raises(ValueError, match=match):
        TrackGenerator(_tcfg(**kw), _rng())


def test_gate_bounds_and_elevation_bounds():
    with pytest.raises(ValueError, match=r"^polar_radial_jitter must be in \[0, 0\.85\], "
                                         r"got 0\.95 \(0\.2\.0\.dev4 clamped it to that "
                                         r"range: set polar_radial_jitter=0\.85"):
        GateGenerator(_gcfg(generator="polar", polar_radial_jitter=0.95), _rng())
    # ... and the dev4 clamp value itself gives the same gates as before.
    GateGenerator(_gcfg(generator="polar", polar_angular_jitter=0.45), _rng())
    gen = TrackGenerator(presets.track(num_envs=E, device=DEV,
                                       elevation=("noise", ELEVATIONS["noise"][0])), _rng())
    with pytest.raises(ValueError, match=r"ElevationNoiseParams\.z_noise_amplitude must be "
                                         r"in \[0\.0, inf\], got -0\.1"):
        gen.params["elevation"].set("z_noise_amplitude", -0.1)


def test_float32_bounds_accept_read_back_values():
    hi = float(np.float32(0.85))
    gen = TrackGenerator(_tcfg(generator="polar", polar_radial_jitter=hi), _rng())
    gen.params["generator"].set("polar_radial_jitter", hi)
    pi32 = float(np.float32(math.pi))
    gen = TrackGenerator(_tcfg(generator="checkpoint", checkpoint_turn_rate=pi32), _rng())
    gen.params["generator"].set("checkpoint_turn_rate", pi32)
    with pytest.raises(ValueError, match=r"must be in \[0\.0, 0\.85\]"):
        TrackGenerator(_tcfg(generator="polar"), _rng()).params["generator"].set(
            "polar_radial_jitter", float(np.nextafter(np.float32(0.85), np.float32(1))))
    # A value within half an ulp above the bound rounds to it: accepted.
    TrackGenerator(_tcfg(generator="polar", polar_radial_jitter=math.nextafter(0.85, 1.0)),
                   _rng())


def test_float_coercion_keeps_the_legacy_float_acceptance():
    ref = _snap(TrackGenerator(_tcfg(handle_clamp_frac=1e9), _rng()).generate())
    got = _snap(TrackGenerator(_tcfg(handle_clamp_frac="1e9"), _rng()).generate())
    assert not _differs(got, ref)
    as_dict = TrackGenerator(_tcfg(generator_params={"handle_clamp_frac": "1.0e9"}), _rng())
    assert as_dict.params["generator"].params.handle_clamp_frac == 1e9
    assert not _differs(_snap(as_dict.generate()), ref)
    # A flat dynamic value compares with generator_params after the same coercion.
    both = TrackGenerator(_tcfg(handle_clamp_frac="1e9",
                                generator_params={"handle_clamp_frac": 1e9}), _rng())
    assert not _differs(_snap(both.generate()), ref)
    with pytest.raises(ValueError, match=r"both configure generator 'bezier'"):
        TrackGenerator(_tcfg(handle_clamp_frac="0.5",
                             generator_params={"handle_clamp_frac": 1e9}), _rng())
    ref = _snap(TrackGenerator(_tcfg(rad=0.3), _rng()).generate())
    gen = TrackGenerator(_tcfg(rad=np.array(0.3)), _rng())
    assert type(gen.params["generator"].params.rad) is float
    assert not _differs(_snap(gen.generate()), ref)
    torch = pytest.importorskip("torch")
    t = torch.tensor(0.3)
    ref = _snap(TrackGenerator(_tcfg(rad=float(t)), _rng()).generate())
    assert not _differs(_snap(TrackGenerator(_tcfg(rad=t), _rng()).generate()), ref)


def _torch_scalar(v):
    torch = pytest.importorskip("torch")
    return torch.tensor(v)


@pytest.mark.parametrize("make", [np.float32, lambda v: np.array(v, np.float32),
                                  _torch_scalar], ids=["np32", "np32_0d", "torch"])
def test_a_float32_flat_knob_does_not_conflict_with_generator_params(make):
    """0.2.0.dev4 compared a flat knob with its default / the generator_params value
    with ``!=`` (NumPy 2 compares ``np.float32(0.4) != 0.4`` in float32: equal); the
    comparison is in float32, the value the array holds."""
    # At the default, with generator_params setting another field.
    TrackGenerator(_tcfg(rad=make(0.4), generator_params={"max_num_points": 12}), _rng())
    TrackGenerator(_tcfg(generator="polar", polar_radial_jitter=make(0.6),
                         generator_params={"polar_num_knots": 9}), _rng())
    GateGenerator(_gcfg(generator="polar", polar_radial_jitter=make(0.6),
                        generator_params={"polar_num_knots": 9}), _rng())
    # Equal to the generator_params value.
    TrackGenerator(_tcfg(rad=make(0.3), generator_params={"rad": 0.3}), _rng())
    GateGenerator(_gcfg(generator="polar", polar_angular_jitter=make(0.2),
                        generator_params={"polar_angular_jitter": 0.2}), _rng())
    # A real difference still conflicts.
    with pytest.raises(ValueError, match=r"both configure generator 'bezier'"):
        TrackGenerator(_tcfg(rad=make(0.3), generator_params={"rad": 0.35}), _rng())


def _torch_bool(v):
    torch = pytest.importorskip("torch")
    return torch.tensor(v)


@pytest.mark.parametrize("make", [np.bool_, np.array, _torch_bool],
                         ids=["np_bool", "np_bool_0d", "torch_bool"])
def test_bool_like_values_are_rejected_like_bools(make):
    with pytest.raises(ValueError, match=r"HullParams\.hull_displacement is dynamic and "
                                         r"must be a real number, got"):
        TrackGenerator(_tcfg(generator="hull", hull_displacement=make(True)), _rng())


# ---------------------------------------------------------------------------------------
# 11. Poison sentinels and the dyn key sets
# ---------------------------------------------------------------------------------------

def _builder(E_=E):
    return types.SimpleNamespace(E=E_, raw=256, scale=1.0, device=wp.get_device(DEV))


_POISON_UNITS = [("generator", g) for g in sorted(GEN_KNOBS)] + \
    [("gate_generator", g) for g in sorted(GATE_KNOBS)] + \
    [("elevation", p) for p in sorted(ELEVATIONS)]


@pytest.mark.parametrize("unit", _POISON_UNITS, ids=[f"{k}.{n}" for k, n in _POISON_UNITS])
def test_the_static_view_poisons_every_dynamic_name(unit):
    kind, name = unit
    cls = registry.get(kind, name).Params
    view = _static_view(cls())
    assert isinstance(view, cls)
    dyn = DYNAMIC[unit]
    ns = run_view(view, _builder()) if kind != "elevation" else \
        _zcfg(types.SimpleNamespace(E=E, layout=types.SimpleNamespace(device=DEV)), name, view)
    for f in dyn:
        v = getattr(ns, f)
        assert isinstance(v, _DynamicField), f
        d = 0.6
        for read in (lambda: float(getattr(ns, f, 0.0)), lambda: v * 2.0,
                     lambda: min(max(v, 0.0), 0.85), lambda: bool(v), lambda: int(v),
                     lambda: np.float32(v), lambda: -v,
                     # equality, membership and hashing pick no branch silently
                     lambda: v == d, lambda: v != d, lambda: d == v, lambda: v in (d,),
                     lambda: hash(v), lambda: {d: 1}.get(v),
                     lambda: getattr(ns, f, 0.0) == 0.0):
            with pytest.raises(TypeError, match=rf"{cls.__name__}\.{f} is dynamic: a "
                                                rf"built-in stage reads it on the device"):
                read()
    # The legacy host reads of each built-in raise on the view (never a default).
    probes = {
        "polar": lambda: wg_polar.polar_jitters(ns),
        "hull": lambda: float(getattr(ns, "hull_displacement", 0.15)),
        "voronoi": lambda: float(getattr(ns, "voronoi_radial_variation", 0.62)),
        "bezier": lambda: float(getattr(ns, "handle_clamp_frac", 1.0e9)),
        "checkpoint": lambda: float(ns.checkpoint_turn_rate),
    }
    if name in probes and kind == "generator":
        with pytest.raises(TypeError):
            probes[name]()
    # A launch with a sentinel as a float argument raises too.
    seeds = _seeds()
    out = wp.empty(E * 12, dtype=wp.vec2f, device=DEV)
    with pytest.raises(Exception):
        wp.launch(wg_polar._polar_controls_k, E, [seeds, 12, getattr(ns, next(iter(dyn))),
                                                  0.3, 1.0, out], device=DEV)


def test_the_legacy_branch_of_a_core_raises_on_the_stage_view():
    """A missed ``dyn`` routing (the core's legacy branch run on the Stage's view)
    raises instead of reading a default."""
    view = _static_view(bp.PolarParams())
    cfg = run_view(view, _builder())
    scratch = wg_polar._alloc_polar_scratch(cfg)
    out = wp.empty(E * 256, dtype=wp.vec2f, device=DEV)
    ok = wp.empty(E, dtype=wp.int32, device=DEV)
    with pytest.raises(TypeError, match=r"PolarParams\.polar_radial_jitter is dynamic"):
        wg_polar._enqueue_polar(_seeds(), cfg, out, ok, scratch,
                                LaunchContext.for_call(DEV), None)


_CORES = {
    "bezier": (wg_bezier._enqueue_bezier, ("rad", "handle_clamp_frac")),
    "hull": (wg_hull._enqueue_hull, ("hull_displacement",)),
    "polar": (wg_polar._enqueue_polar, ("polar_radial_jitter", "polar_angular_jitter")),
    "voronoi": (wg_voronoi._enqueue_voronoi, ("voronoi_radial_variation",
                                               "voronoi_angular_jitter")),
    "checkpoint": (wg_checkpoint._enqueue_checkpoint, ("checkpoint_radius_min_frac",
                                                        "checkpoint_angle_jitter",
                                                        "checkpoint_turn_rate")),
}


@pytest.mark.parametrize("name", sorted(_CORES))
def test_the_private_cores_check_the_dyn_key_set(name):
    core, fields = _CORES[name]
    arr = _f32(np.zeros(E))
    partial = {f: arr for f in fields[:-1]}
    extra = dict({f: arr for f in fields}, bogus=arr)
    for dyn in (partial, extra):
        with pytest.raises(ValueError, match=rf"{name}: dyn must map exactly the dynamic "
                                             rf"fields"):
            core(None, None, None, None, None, None, dyn)
    zcfg = types.SimpleNamespace(z_profile="noise")
    for dyn in ({"z_min": arr, "z_max": arr},
                {"z_min": arr, "z_max": arr, "z_noise_amplitude": arr, "z_max_step": arr}):
        with pytest.raises(ValueError, match=r"elevation 'noise': dyn must map exactly"):
            warp_zprofile._apply_z_profile(zcfg, None, None, 1, None, None, None, None, dyn)
        with pytest.raises(ValueError, match=r"elevation 'noise': dyn must map exactly"):
            warp_zprofile._apply_z_profile_knots(zcfg, None, None, 1, None, None, None,
                                                 None, None, None, None, dyn)


# ---------------------------------------------------------------------------------------
# 12. style_sampling
# ---------------------------------------------------------------------------------------

_RANGES = {"rad_range": (0.1, 0.35), "scale_range": (0.8, 1.2),
           "handle_clamp_frac_range": (0.3, 0.9)}
_STYLE_COMBOS = [{k: (v if (m >> i) & 1 else None) for i, (k, v) in enumerate(_RANGES.items())}
                 for m in range(8)]


@pytest.mark.parametrize("seeds", ["fingerprint", "spread"])
@pytest.mark.parametrize("combo", range(8))
def test_style_sampling_bytes_equal_the_legacy_path(combo, seeds):
    s = SEEDS if seeds == "fingerprint" else SEEDS_B
    ranges = _STYLE_COMBOS[combo]
    cfg = _tcfg(style_sampling=True, **ranges)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # no deprecation, no warning
        gen = TrackGenerator(cfg, _rng(s))
    gen.generate()
    scratch = wg_bezier.bezier_alloc_scratch(cfg)
    out = wp.empty(E * 256, dtype=wp.vec2f, device=DEV)
    ok = wp.empty(E, dtype=wp.int32, device=DEV)
    wg_bezier.generate_centerline_warp(_seeds(s), cfg, out, ok, scratch,
                                       ctx=LaunchContext.for_call(DEV))
    assert gen.program.tap("raw.xy").numpy().tobytes() == out.numpy().tobytes()
    table = gen.params["generator"]
    for field, rng, legacy_draw in (("rad", "rad_range", scratch.gen_style_rad),
                                    ("handle_clamp_frac", "handle_clamp_frac_range",
                                     scratch.gen_style_clamp)):
        if ranges[rng] is not None:
            # A ranged knob is drawn by the style kernel into its (unlatched) array:
            # drawn by the stage itself (its own id).
            assert table.drawn[field] == "generator"
            assert table[field].numpy().tobytes() == legacy_draw.numpy().tobytes()
            with pytest.raises(ValueError, match=rf"generator\.{field} is drawn every "
                                                 rf"generation by the generator's "
                                                 rf"style_sampling \({rng}\); set "
                                                 rf"{rng}=None to set it"):
                table.set(field, 0.3)
        else:
            assert field not in table.drawn
            assert table[field].numpy().tolist() == \
                [float(np.float32(getattr(cfg, field)))] * E


def test_style_sampling_composes_with_set_and_randomize():
    # A None-range knob reads its dynamic array: set(v) == a style run built with v.
    cfg = dict(style_sampling=True, rad_range=None, scale_range=(0.8, 1.2))
    gen = TrackGenerator(_tcfg(**cfg), _rng())
    gen.params["generator"].set("rad", 0.25)
    got = _snap(gen.generate())
    want = _snap(TrackGenerator(_tcfg(rad=0.25, **cfg), _rng()).generate())
    assert not _differs(got, want)
    # A ranged knob cannot be randomized.
    spec = legacy.to_pipeline_spec(_tcfg(style_sampling=True, rad_range=(0.1, 0.35)))
    with pytest.raises(PlanError, match=r"generator\.rad is drawn by the generator's "
                                        r"style_sampling \(rad_range\): set rad_range=None "
                                        r"to randomize it"):
        build(spec.with_randomize({"generator.rad": (0.1, 0.3)}))
    # ... but the other one can, and scale_range keeps working alongside.
    spec = legacy.to_pipeline_spec(_tcfg(style_sampling=True, rad_range=(0.1, 0.35),
                                         scale_range=(0.8, 1.2)))
    gen = TrackGenerator(spec.with_randomize({"generator.handle_clamp_frac": (0.3, 0.9)}),
                         _rng())
    gen.generate()
    # drawn maps each field to the id of the stage drawing it (the stage's own id for a
    # style draw): every value is a key of gen.params.
    drawn = dict(gen.params["generator"].drawn)
    assert drawn == {"rad": "generator",
                     "handle_clamp_frac": "randomize.generator.handle_clamp_frac"}
    assert all(w in gen.params for w in drawn.values())
    for writer in set(drawn.values()) - {"generator"}:          # a curriculum helper
        gen.params[writer].set("hi", 0.8)
    r = repr(gen.params["generator"])
    assert "rad=<drawn by the generator's style_sampling (rad_range)>" in r, r
    assert "handle_clamp_frac=<drawn by 'randomize.generator.handle_clamp_frac'>" in r, r


# ---------------------------------------------------------------------------------------
# 13. The generated CUDA source of each scalar / twin pair (locked until PR16)
# ---------------------------------------------------------------------------------------

#: twin -> the ops it adds to its scalar kernel's forward body: the knob loads
#: (address / load / copy), the comparison clamps' selects (where; the comparisons are
#: C operators) and, for the per-slot steer twin, the env index (floordiv).
_CUDA_EXTRAS = {
    "_assemble_dyn_k": {"address": 2, "load": 2, "copy": 2, "where": 4},
    "_midpoint_displace_dyn_k": {"address": 1, "load": 1, "copy": 1, "where": 2},
    "_polar_controls_dyn_k": {"address": 2, "load": 2, "copy": 2, "where": 4},
    "_sample_checkpoints_dyn_k": {"address": 2, "load": 2, "copy": 2, "where": 4},
    "_steer_dyn_k": {"floordiv": 1, "address": 1, "load": 1, "copy": 1, "where": 2},
    "_select_anchor_sites_dyn_k": {"address": 2, "load": 2, "copy": 2, "where": 4},
    "_z_uniform_dyn_k": {"address": 2, "load": 2, "copy": 2},
    "_z_walk_dyn_k": {"address": 3, "load": 3, "copy": 3, "where": 1},
    "_z_noise_dyn_k": {"address": 3, "load": 3, "copy": 3, "where": 1},
}
_ARITH = ("mul", "add", "sub", "div", "min", "max", "clamp", "neg", "sin", "cos", "atan2",
          "sqrt", "length", "randf", "rand_init", "floordiv", "mod", "abs", "where")


def _cuda_ops(kernel) -> list:
    from warp._src import codegen as G  # noqa: PLC0415 - Warp's private codegen
    from warp._src import context as C  # noqa: PLC0415

    mod = kernel.module
    opts = dict(mod.options)
    opts["enable_backward"] = False
    C.ModuleBuilder(mod, opts).build_kernel(kernel)
    text = G.codegen_kernel(kernel, "cuda", opts)
    return re.findall(r"wp::(\w+)\(", text.split("// forward", 1)[-1])


@pytest.mark.parametrize("pair", sorted(TWINS))
def test_cuda_source_of_each_twin_is_its_scalar_kernels(pair):
    """Warp's generated CUDA forward body of the twin has its scalar kernel's op
    sequence once the knob loads, the clamp selects and the env index are set aside
    (same expression graph: same FMA contraction), and no min / max was added. Uses
    Warp's private codegen entry points: updated with a Warp floor bump, never
    skipped."""
    p = TWINS[pair]
    a = _cuda_ops(getattr(p.module, p.scalar))
    b = _cuda_ops(getattr(p.module, p.twin))
    extra = collections.Counter(b) - collections.Counter(a)
    missing = collections.Counter(a) - collections.Counter(b)
    assert dict(extra) == _CUDA_EXTRAS[p.twin] and not missing, (dict(extra), dict(missing))
    drop = set(_CUDA_EXTRAS[p.twin])
    assert [o for o in a if o not in drop] == [o for o in b if o not in drop]
    assert [o for o in a if o in _ARITH and o not in drop] == \
        [o for o in b if o in _ARITH and o not in drop]


def _kernel_ast(kernel) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel.func)))
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef), kernel.key
    return fn


class _Rename(ast.NodeTransformer):
    def __init__(self, names: dict) -> None:
        self.names = names

    def visit_Name(self, node):
        if node.id in self.names:
            return ast.copy_location(ast.Name(id=self.names[node.id], ctx=node.ctx), node)
        return node


def _twin_body(pair: _Twin, scalar: ast.FunctionDef, twin: ast.FunctionDef) -> list:
    """The twin's body with its prelude dropped (each knob's load ``x = arr[e]`` from a
    knob array argument, the comparison clamps ``x = wp.where(x < LO, LO, x)`` /
    ``wp.where(x > HI, HI, x)`` and, when the scalar kernel has none, the env index
    ``e = ...``) and each loaded local renamed to the scalar kernel's parameter name."""
    knob_arrays = {twin.args.args[i].arg: scalar.args.args[i].arg for i in pair.knobs}
    scalar_assigns_e = any(isinstance(n, ast.Name) and n.id == "e" and
                           isinstance(n.ctx, ast.Store) for n in ast.walk(scalar))
    rename, body = {}, []
    for stmt in twin.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0].id, stmt.value
            if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and \
                    value.value.id in knob_arrays and \
                    isinstance(value.slice, ast.Name) and value.slice.id == "e":
                rename[target] = knob_arrays[value.value.id]
                continue
            if target in rename and isinstance(value, ast.Call) and \
                    ast.unparse(value.func) == "wp.where" and len(value.args) == 3:
                cond, a, b = value.args
                if isinstance(cond, ast.Compare) and len(cond.ops) == 1 and \
                        isinstance(cond.ops[0], (ast.Lt, ast.Gt)) and \
                        ast.unparse(cond.left) == target == ast.unparse(b) and \
                        ast.dump(cond.comparators[0]) == ast.dump(a):
                    continue
            if target == "e" and not scalar_assigns_e:
                continue
        body.append(stmt)
    module = ast.Module(body=body, type_ignores=[])
    return _Rename(rename).visit(module).body


@pytest.mark.parametrize("pair", sorted(TWINS))
def test_the_source_of_each_twin_is_its_scalar_kernels_body(pair):
    """Beyond the CUDA op sequence: once its prelude (knob loads, comparison clamps,
    env index) is dropped and the loaded locals take the scalar parameter names, the
    twin's body IS the scalar kernel's body, statement for statement (operands,
    operand order, constants and indices included). Warp's codegen is a function of
    the AST, so the generated arithmetic, hence nvcc's FMA contraction, is the same."""
    p = TWINS[pair]
    scalar = _kernel_ast(getattr(p.module, p.scalar))
    twin = _kernel_ast(getattr(p.module, p.twin))
    want, got = scalar.body, _twin_body(p, scalar, twin)
    diff = [f"scalar: {ast.unparse(a)}\n  twin: {ast.unparse(b)}"
            for a, b in zip(want, got) if ast.dump(a) != ast.dump(b)]
    assert not diff and len(want) == len(got), "\n".join(diff) or (len(want), len(got))
    # The signatures agree outside the knobs (and bezier's four bound scalars).
    after = p.bounds_after
    targs = list(twin.args.args)
    if after is not None:
        targs = targs[:after + 1] + targs[after + 5:]
    assert len(targs) == len(scalar.args.args)
    for i, (a, b) in enumerate(zip(scalar.args.args, targs)):
        if i not in p.knobs:
            assert (a.arg, ast.dump(a.annotation)) == (b.arg, ast.dump(b.annotation)), i


# ---------------------------------------------------------------------------------------
# 14. Segmented Programs: a write after segment 0 applies one generation later
# ---------------------------------------------------------------------------------------

def test_a_write_recorded_after_segment_0_applies_at_the_next_generation(plugin_registry):
    from tests._plugin_toys import register_stage_toys

    register_stage_toys(plugin_registry)
    noise = dict(z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.3)

    def spec(z_max):
        return presets.track(num_envs=E, device=DEV, generator="toy.wobble_host",
                             elevation=("noise", dict(noise, z_max=z_max)))

    def z_of(z_max):
        p = build(spec(z_max), seeds=_seeds())
        return p.result.center.numpy().copy()

    program = build(spec(0.4), seeds=_seeds())
    assert len(program.segments) == 2 and program.segments[0].host
    ctx = LaunchContext.for_call(DEV)
    zmax = program.params["elevation"]

    def generation(write_between=None):
        program.enqueue_segment(0, ctx)
        if write_between is not None:
            zmax.set("z_max", write_between)
        program.enqueue_segment(1, ctx)
        return program.result.center.numpy().copy()

    at_04, at_02 = z_of(0.4), z_of(0.2)
    assert at_04.tobytes() != at_02.tobytes()
    # A write between segment 0 and the tail: this generation still uses the old value.
    assert generation(write_between=0.2).tobytes() == at_04.tobytes()
    assert generation().tobytes() == at_02.tobytes()            # ... applied one later
    # A write before segment 0 applies at once.
    zmax.set("z_max", 0.4)
    assert generation().tobytes() == at_04.tobytes()


# ---------------------------------------------------------------------------------------
# 15. Legacy surfaces
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(GEN_KNOBS))
def test_the_pr9_composition_with_the_legacy_spec_equals_the_trackgenerator(name):
    kw = dict(generator=name, **{f: v[1] for f, v in GEN_KNOBS[name].items()})
    cfg = _tcfg(**kw)
    track, scratch = _legacy_track(cfg)
    assert scratch.generator_spec is warp_pipeline._legacy_generator_spec(name)
    got = _snap(TrackGenerator(cfg, _rng()).generate())
    for f in TRACK_FIELDS:
        assert got[f].tobytes() == getattr(track, f).numpy().tobytes(), f


_FROZEN = {
    (wg_bezier, "generate_centerline_warp"): "(seeds_wp, config, out_centerline, "
                                             "out_valid_wp, scratch, ctx=None)",
    (wg_hull, "generate_hull_warp"): "(seeds_wp, config, out_centerline, out_valid_wp, "
                                     "scratch, ctx=None)",
    (wg_polar, "generate_polar_warp"): "(seeds_wp, config, out_centerline, out_valid_wp, "
                                       "scratch, ctx=None)",
    (wg_voronoi, "generate_voronoi_warp"): "(seeds_wp, config, out_centerline, "
                                           "out_valid_wp, scratch, ctx=None)",
    (wg_checkpoint, "generate_checkpoint_warp"): "(seeds_wp, config, out_centerline, "
                                                 "out_valid_wp, scratch, ctx=None)",
    (wg_bezier, "bezier_alloc_scratch"): "(config)",
    (wg_hull, "hull_alloc_scratch"): "(config)",
    (wg_polar, "polar_alloc_scratch"): "(config)",
    (wg_voronoi, "voronoi_alloc_scratch"): "(config)",
    (wg_checkpoint, "checkpoint_alloc_scratch"): "(config)",
    (warp_zprofile, "apply_z_profile"): "(config, seeds_wp, count, stride, cum, perim, z, "
                                        "ctx=None)",
    (warp_zprofile, "apply_z_profile_knots"): "(config, seeds_wp, count, stride, arclen, "
                                              "perim, knot_cum, knot_count, knot_z, z, "
                                              "ctx=None)",
    (wg_polar, "sample_polar_controls"): "(ctx, seeds, K, radial_jitter, angular_jitter, "
                                         "out, *, base_radius=1.0)",
    (wg_checkpoint, "sample_checkpoint_rings"): "(ctx, seeds, K, C, radius_min_frac, "
                                                "angle_jitter, out, *, base_radius=1.0)",
    (wg_voronoi, "select_voronoi_anchors"): "(ctx, seeds, sites, S, K, layout_mode, "
                                            "box_size, radial_variation, angular_jitter, "
                                            "out_selected, used)",
    (wg_polar, "sample_polar_controls_per_env"): "(ctx, seeds, K, radial_jitter, "
                                                 "angular_jitter, out, *, base_radius=1.0)",
    (wg_checkpoint, "sample_checkpoint_rings_per_env"): "(ctx, seeds, K, C, "
                                                        "radius_min_frac, angle_jitter, "
                                                        "out, *, base_radius=1.0)",
    (wg_voronoi, "select_voronoi_anchors_per_env"): "(ctx, seeds, sites, S, K, "
                                                    "layout_mode, box_size, "
                                                    "radial_variation, angular_jitter, "
                                                    "out_selected, used)",
}


def test_the_legacy_signatures_are_frozen():
    bad = []
    for (mod, name), want in _FROZEN.items():
        sig = inspect.signature(getattr(mod, name))
        bare = sig.replace(parameters=[p.replace(annotation=inspect.Parameter.empty)
                                       for p in sig.parameters.values()],
                           return_annotation=inspect.Signature.empty)
        if str(bare) != want:
            bad.append(f"{mod.__name__}.{name}{bare}")
    assert not bad, bad


def test_a_generator_spec_with_dynamic_params_is_refused():
    with pytest.raises(TypeError, match=r"BezierParams\.rad .* is dynamic: dynamic fields "
                                        r"need a Stage"):
        registry.add(GeneratorSpec(name="acme.legacy_bezier",
                                   alloc_scratch=wg_bezier.bezier_alloc_scratch,
                                   generate=wg_bezier.generate_centerline_warp,
                                   accepts_ctx=True, params=bp.BezierParams))


# ---------------------------------------------------------------------------------------
# 16. The kit's params.per_env
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class _LobeParams(plugin.Params):
    m: float = plugin.dynamic(0.1, lo=0.0, hi=0.3, doc="lobe amplitude")


@wp.kernel
def _lobe_m0_k(seed: wp.array(dtype=wp.int32), m: wp.array(dtype=wp.float32), n: int,
               out: wp.array(dtype=wp.vec2f)):
    # BROKEN on purpose: reads m[0] for every env.
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = wp.rand_init(seed[e], i)
    th = 6.2831853 * float(i) / float(n)
    r = 1.0 + m[0] * wp.sin(5.0 * th) + 0.01 * wp.randf(st)
    out[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th))


@wp.kernel
def _lobe_writes_k(seed: wp.array(dtype=wp.int32), m: wp.array(dtype=wp.float32), n: int,
                   out: wp.array(dtype=wp.vec2f)):
    # BROKEN on purpose: writes its (latched) param array.
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = wp.rand_init(seed[e], i)
    th = 6.2831853 * float(i) / float(n)
    r = 1.0 + m[e] * wp.sin(5.0 * th) + 0.01 * wp.randf(st)
    out[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th))
    if i == 0:
        m[e] = m[e] * 0.5


@wp.kernel
def _lobe_ok_k(seed: wp.array(dtype=wp.int32), m: wp.array(dtype=wp.float32), n: int,
               out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = wp.rand_init(seed[e], i)
    th = 6.2831853 * float(i) / float(n)
    r = 1.0 + m[e] * wp.sin(5.0 * th) + 0.01 * wp.randf(st)
    out[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th))


def _lobe_stage(name: str, kernel):
    class Lobe(plugin.Stage):
        kind = "generator"
        output = "centerline"
        Params = _LobeParams
        writes = {"xy": "raw.xy", "valid": "gen.valid"}

        def build(self, b, p):
            xy = b.write("xy", wp.vec2f, per="raw")
            ok = b.write("valid", wp.int32)
            m, n, seed = b.param("m"), b.raw, b.seed

            def enqueue(ctx):
                ctx.launch(kernel, b.E * n, [seed, m, n, xy])
                ctx.fill(ok, 1)
            return enqueue

    Lobe.name = name
    Lobe.__name__ = Lobe.__qualname__ = name.replace(".", "_")
    return Lobe


@pytest.mark.parametrize("name,kernel,fails", [
    ("acme.lobe_ok", _lobe_ok_k, None),
    ("acme.lobe_m0", _lobe_m0_k, "must be read as param[e], never param[0]"),
    ("acme.lobe_writes", _lobe_writes_k, "the stage wrote its param array(s) ['m']"),
])
def test_the_kit_params_per_env_check(plugin_registry, name, kernel, fails):
    plugin_registry.add(_lobe_stage(name, kernel))
    report = check_generator(name, devices=[DEV])
    per_env = report.result("params.per_env", DEV)
    assert not per_env.skipped
    if fails is None:
        assert report.ok, str(report)
        assert per_env.details["fields"] == ["m"]
    else:
        assert report.failed_names == ["params.per_env"], str(report)
        assert fails in per_env.message, per_env.message


def test_the_kit_params_per_env_skips_units_without_dynamic_params():
    report = check_generator("repulsive", devices=[DEV], num_envs=2,
                             config=dict(num_points=64, repulsive_stages=(16, 32, 64)),
                             skip=("hot_path", "statelessness", "batch_independence",
                                   "determinism", "diversity", "output"))
    per_env = report.result("params.per_env", DEV)
    assert per_env.skipped and per_env.message == "skipped: no dynamic parameters"
