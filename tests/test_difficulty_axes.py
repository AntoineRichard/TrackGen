"""Roadmap PR13d: per-env size and the remaining difficulty axes (CPU; the CUDA twin is
``tests/test_difficulty_axes_cuda.py``).

- ``size`` (float32, default 1, bounds ``ops.constants.SIZE_BOUNDS``): a per-env factor
  on ``scale`` applied where each generator applies it (bezier / hull and their gate
  generators: last in the corner grid; polar / voronoi / checkpoint: the normalization
  target; the polar / voronoi / checkpoint gate pipelines: ``normalize.size``;
  repulsive: width-aware constrained growth followed by its sized output copy). No size on ``external``.
- ``min_num_points`` (bezier / hull track generators): a dynamic int32 lower count
  bound, clamped to ``[2, max_num_points]`` at use.
- ``gate_frames.gate_width`` dynamic; ``validity.gate_default`` checks bar crossings per
  env (its ``gate_width`` / ``check_crossing`` inert).
- ``StageParams.in_effect`` / ``StageParams.bounds``.

Covered here (design section 9.2): 1. declarations, the flat ``size`` and the public ops;
2. each twin == its original (generator-produced inputs, discriminating NaN payloads for
the guarded copy); 3. the generated CUDA source of each twin (on CPU); 4. per env ==
per-env scalar runs; 5. repulsive's dynamic size consistency; 6. size vs scale; 7. randomize
on the new axes and the plan rules (hints, truncation and composed-draw warnings); 8. the
launch deltas of every pipeline (``tests/_pr13d_delta.py``, plain and masked); 9.
``in_effect`` / ``bounds``; 11.
bounds and types; 12. poison; 13. capacity (truncation fails closed, the checkpoint
bound); 14. legacy surfaces; 15. the kit; 16. ``presets(size=)``; 17. hints; 18. posed
gates.
"""
from __future__ import annotations
from tests._reference_generators import warp_pipeline as _scalar_pipeline
from tests._reference_generators import warp_generate as _bezier_reference

import collections
import inspect
import math
import re
import warnings

import numpy as np
import pytest
import warp as wp

from tests._pr13d_delta import (GENERATOR_LATCHES, assert_latch_block, gate_latches,
                                latch_dtypes, pr13d_sequence)
from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig,
                       TrackGenerator, ops)
from track_gen._src import builtin_params as bp
from track_gen._src import warp_gate
from track_gen._src import warp_generate as wg_bezier
from track_gen._src import warp_generate_checkpoint as wg_checkpoint
from track_gen._src import warp_generate_hull as wg_hull
from track_gen._src import warp_generate_polar as wg_polar
from track_gen._src import warp_generate_repulsive as wg_repulsive
from track_gen._src import warp_generate_voronoi as wg_voronoi
from track_gen._src import warp_pipeline
from track_gen.exec import LaunchContext
from track_gen.ops import geometry as ops_geometry
from track_gen.ops import sampling as ops_sampling
from track_gen.ops.constants import SIZE_BOUNDS
from track_gen.pipeline import PipelineSpec, PlanError, StageRef, build, legacy, presets
from track_gen.plugin import GeneratorSpec, registry
from track_gen.plugin.params import _DynamicField, _fields, _schema, _static_view
from track_gen.stages import gates as st_gates
from track_gen.stages import params as sp
from track_gen.stages.external import ExternalParams
from track_gen.testing import check_gate_generator, check_generator
from track_gen.testing.capture import result_arrays
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 8
#: The fingerprint seeds (int32 extremes included).
SEEDS = np.array([0, 1, 7, 99991, 2147483647, 1000003, 123456789, 46341], np.int32)
#: Per-env sizes: interior values, 1 (== the default rows) and two clamped ones.
SIZES = (1.0, 0.7, 1.3, 1.0, 0.45, 2.2, 20.0, 0.01)
REP = dict(num_points=64, repulsive_stages=(16, 32, 64))


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _seeds(seeds=SEEDS) -> wp.array:
    return wp.array(np.asarray(seeds, np.int32), dtype=wp.int32, device=DEV)


def _rng(seeds=SEEDS) -> PerEnvSeededRNG:
    return PerEnvSeededRNG(seeds=_seeds(seeds), num_envs=len(seeds), device=DEV)


def _f32(values) -> wp.array:
    return wp.array(np.asarray(values, np.float32), dtype=wp.float32, device=DEV)


def _i32(values) -> wp.array:
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


def _tcfg(**kw) -> TrackGenConfig:
    kw.setdefault("num_envs", E)
    return TrackGenConfig(device=DEV, **kw)


def _gcfg(**kw) -> GateGenConfig:
    kw.setdefault("num_envs", E)
    return GateGenConfig(device=DEV, **kw)


def _snap(result) -> dict:
    return {k: np.array(a.numpy(), copy=True) for k, a in result_arrays(result).items()}


def _row(arr: np.ndarray, e: int, n: int = E) -> bytes:
    return np.ascontiguousarray(arr.reshape(n, -1)[e]).tobytes()


def _rows_differ(a: dict, b: dict, e: int, n: int = E) -> list:
    return [k for k in b if _row(a[k], e, n) != _row(b[k], e, n)]


def _differs(a: dict, b: dict) -> bool:
    return any(x.tobytes() != b[k].tobytes() for k, x in a.items())


def _clamp(v, lo, hi):
    """``v`` clamped by comparisons in float32, as the kernels do (a NaN passes)."""
    x = np.float32(v)
    if lo is not None and x < np.float32(lo):
        x = np.float32(lo)
    if hi is not None and x > np.float32(hi):
        x = np.float32(hi)
    return float(x)


def _clamp_size(v) -> float:
    return _clamp(v, *SIZE_BOUNDS)


def _check_per_env(make, sid: str, field: str, values, label: str, *, used=None,
                   n: int = E, dtype=np.float32) -> None:
    """``make(v)`` builds a generator (``v=None``: the default); one built with the
    defaults and ``field`` set per env to ``values`` equals, env by env, a run BUILT with
    that env's (clamped: ``used(v)``) value; a size-1 env equals the default run; the
    batch moves (non-vacuity)."""
    used = (lambda v: v) if used is None else used
    gen = make(None)
    gen.params[sid].set(field, wp.array(np.asarray(values, dtype), device=DEV))
    got = _snap(gen.generate())
    default = _snap(make(None).generate())
    refs: dict = {}
    moved = 0
    for e, v in enumerate(values):
        u = used(v)
        if u not in refs:
            refs[u] = _snap(make(u).generate())
        bad = _rows_differ(got, refs[u], e, n)
        assert not bad, f"{label}: env {e} ({field}={v!r}, used {u!r}) differs in {bad}"
        moved += bool(_rows_differ(got, default, e, n))
    assert moved, f"{label}: no env moved (vacuous)"


def _capture(monkeypatch, make, kernel) -> tuple:
    """``(dim, args)`` of the first launch of ``kernel`` while ``make()`` runs (array
    arguments cloned right before it: the generator-produced inputs)."""
    real = LaunchContext.launch
    got: list = []

    def spy(self, k, dim, inputs=(), outputs=(), **kw):
        if k is kernel and not got:
            got.append((dim, [wp.clone(a) if isinstance(a, wp.array) else a
                              for a in (*inputs, *outputs)]))
        return real(self, k, dim, inputs, outputs, **kw)

    with monkeypatch.context() as m:
        m.setattr(LaunchContext, "launch", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make()
    assert got, f"{kernel.key} was never launched"
    return got[0]


def _launch(kernel, dim, args) -> list:
    """Launch on fresh copies of the array arguments; their contents after the launch."""
    fresh = [wp.clone(a) if isinstance(a, wp.array) else a for a in args]
    wp.launch(kernel, dim=dim, inputs=fresh, device=DEV)
    return [np.array(a.numpy(), copy=True) if isinstance(a, wp.array) else None
            for a in fresh]


def _arrays(out: list, skip=()) -> list:
    return [a for i, a in enumerate(out) if a is not None and i not in skip]


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x.tobytes() == y.tobytes() for x, y in zip(a, b))


def _same_rows(a: list, b: list, e: int) -> bool:
    return all(_row(x, e) == _row(y, e) for x, y in zip(a, b) if x.shape[0] % E == 0)


# ---------------------------------------------------------------------------------------
# 1. Declarations, the flat size, the public ops
# ---------------------------------------------------------------------------------------

#: The Params classes that declare size (design section 1.1) and some that must not.
SIZE_OWNERS = (bp.BezierParams, bp.HullParams, bp.PolarParams, bp.VoronoiParams,
               bp.CheckpointParams, bp.RepulsiveParams, bp.PointGateParams,
               sp.NormalizeExtentParams)
NO_SIZE = (ExternalParams, bp.PolarGateParams, bp.VoronoiGateParams,
           bp.CheckpointGateParams, sp.GateFramesParams, sp.GateValidityParams)


@pytest.mark.parametrize("cls", SIZE_OWNERS, ids=lambda c: c.__name__)
def test_size_is_the_last_field_of_every_size_owner(cls):
    last = _fields(cls)[-1]
    assert last.name == "size" and last.kind == "dynamic" and last.dtype is wp.float32
    assert (last.default, last.lo, last.hi) == (1.0, *SIZE_BOUNDS) == (1.0, 0.1, 10.0)
    prop = _schema(cls)["properties"]["size"]
    assert prop["x-dynamic"] is True and prop["x-dtype"] == "float32"
    assert (prop["minimum"], prop["maximum"], prop["x-bounds"]) == (0.1, 10.0, "float32")


def test_the_size_owners_and_the_other_axes():
    for cls in NO_SIZE:
        assert "size" not in {f.name for f in _fields(cls)}, cls.__name__
    # Every built-in generator but external declares size; the polar / voronoi /
    # checkpoint gate generators are re-normalized: their size is normalize.extent's.
    for name in registry.available("generator"):
        info = registry.describe("generator", name)
        if info.builtin:
            assert ("size" in info.params_schema["properties"]) == (name != "external"), name
    for name in ("bezier", "hull", "polar", "voronoi", "checkpoint"):
        props = registry.describe("gate_generator", name).params_schema["properties"]
        assert ("size" in props) == (name in ("bezier", "hull")), name
    from track_gen import stages

    assert stages.get("normalize", "extent").Params is sp.NormalizeExtentParams
    assert ops.constants.SIZE_BOUNDS == (0.1, 10.0) and "SIZE_BOUNDS" in \
        ops.constants.__all__
    # min_num_points: a dynamic int32 on the bezier / hull track generators only.
    for cls in (bp.BezierParams, bp.HullParams):
        f = {x.name: x for x in _fields(cls)}["min_num_points"]
        assert (f.kind, f.dtype, f.default, f.lo, f.hi) == ("dynamic", wp.int32, 9, 2, None)
        assert {x.name: x for x in _fields(cls)}["max_num_points"].kind == "static"
        assert _schema(cls)["properties"]["min_num_points"]["x-dtype"] == "int32"
    assert {x.name: x.kind for x in _fields(bp.PointGateParams)}["min_num_points"] == \
        "static"
    # gate_width: dynamic on gate_frames.default, static (inert) on validity.gate_default
    kinds = {x.name: (x.kind, x.dtype, x.lo) for x in _fields(sp.GateFramesParams)}
    assert kinds["gate_width"] == ("dynamic", wp.float32, 0.0)
    v = {x.name: x.kind for x in _fields(sp.GateValidityParams)}
    assert "gate_width" not in v and "check_crossing" not in v


def test_the_flat_size_round_trips_and_reaches_its_stage():
    cfg = _tcfg(size=1.5)
    assert TrackGenConfig().size == 1.0 and GateGenConfig().size == 1.0
    spec = legacy.to_pipeline_spec(cfg)
    assert spec.stages[0].params.size == 1.5
    gen = TrackGenerator(cfg, _rng())
    assert gen.config.size == 1.5 and gen.params["generator"].params.size == 1.5
    assert gen.params["generator"]["size"].numpy().tolist() == [1.5] * E
    # bezier / hull gates: generator.size; polar / voronoi / checkpoint: normalize.size
    for name in ("bezier", "hull"):
        g = GateGenerator(_gcfg(generator=name, size=1.5), _rng())
        assert g.params["generator"]["size"].numpy().tolist() == [1.5] * E
    for name in ("polar", "voronoi", "checkpoint"):
        gspec = legacy.to_gate_pipeline_spec(_gcfg(generator=name, size=1.5))
        norm = next(r for r in gspec.stages if r.slot_id == "normalize")
        assert dict(norm.params) == {"size": 1.5}
        g = GateGenerator(gspec, _rng())
        assert g.params["normalize"]["size"].numpy().tolist() == [1.5] * E
        assert "size" not in g.params["generator"]
    # The defaults add no size entry to a spec dict (presets) nor normalize params.
    for d in (presets.track(num_envs=E, device=DEV).to_dict(),
              presets.gates(num_envs=E, device=DEV, generator="polar").to_dict()):
        assert "'size'" not in repr(d) and '"size"' not in repr(d)
    norm = next(r for r in legacy.to_gate_pipeline_spec(_gcfg(generator="polar")).stages
                if r.slot_id == "normalize")
    assert norm.params is None


def test_public_ops_take_a_per_env_size():
    ctx = LaunchContext.for_call(DEV)
    seeds, P = _seeds(), 13
    nc, cell = ops.grid_cells(0.05)
    used = wp.empty(E * P, dtype=wp.int32, device=DEV)

    def grid(scale, size=None):
        out = wp.empty(E * P, dtype=wp.vec2f, device=DEV)
        with trace_launches() as t:
            ops.sample_grid_points(ctx, seeds, 9781, 0, P, nc, cell, scale, used, out,
                                   size=size)
        return out.numpy().copy(), [r.kernel.rsplit(".", 1)[-1] for r in t]

    plain, k0 = grid(1.0)
    ones, k1 = grid(1.0, _f32(np.ones(E)))
    assert k0 == ["_fill_i32_k", "_sample_grid_points_h_k"]
    assert k1 == ["_fill_i32_k", "_sample_grid_points_sized_h_k"]
    assert ones.tobytes() == plain.tobytes()
    sized, _ = grid(1.0, _f32(SIZES))
    for e, v in enumerate(SIZES):
        ref, _ = grid(_clamp_size(v))
        assert _row(sized, e) == _row(ref, e), (e, v)
    per_scale = _f32(np.linspace(0.8, 1.2, E))
    _, k2 = grid(per_scale, _f32(SIZES))
    assert k2 == ["_fill_i32_k", "_sample_grid_points_scaled_sized_h_k"]
    a, _ = grid(per_scale)
    b, _ = grid(per_scale, _f32(np.ones(E)))
    assert a.tobytes() == b.tobytes()
    for bad, match in ((_f32(np.ones(E - 1)), r"size must have shape \(8,\)"),
                       (_i32(np.ones(E)), r"size must have dtype float32"),
                       (np.ones(E, np.float32), r"size must be a wp\.array")):
        with pytest.raises(ValueError, match=match):
            ops.sample_grid_points(ctx, seeds, 9781, 0, P, nc, cell, 1.0, used,
                                   wp.empty(E * P, dtype=wp.vec2f, device=DEV), size=bad)
    # normalize_extent: the target extent times size[e]
    N = 16
    t = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    pts = np.concatenate([np.stack([np.cos(t) * (1 + 0.1 * e), 0.6 * np.sin(t)], -1)
                          for e in range(E)]).astype(np.float32)

    def norm(target, **kw):
        a = wp.array(pts, dtype=wp.vec2f, device=DEV)
        with trace_launches() as tr:
            ops.normalize_extent(ctx, a, N, target, **kw)
        return a.numpy().copy(), [r.kernel.rsplit(".", 1)[-1] for r in tr]

    n0, kn0 = norm(1.44)
    n1, kn1 = norm(1.44, size=_f32(np.ones(E)))
    assert (kn0, kn1) == (["_normalize_extent_k"], ["_normalize_extent_sized_k"])
    assert n0.tobytes() == n1.tobytes()
    ns, _ = norm(1.44, size=_f32(SIZES))
    for e, v in enumerate(SIZES):
        ref, _ = norm(float(np.float32(1.44) * np.float32(_clamp_size(v))))
        assert _row(ns, e) == _row(ref, e), (e, v)
    with pytest.raises(ValueError, match=r"size= with count= is not supported"):
        ops.normalize_extent(ctx, wp.array(pts, dtype=wp.vec2f, device=DEV), N, 1.44,
                             count=_i32([N] * E), size=_f32(np.ones(E)))
    with pytest.raises(ValueError, match=r"size must have shape"):
        ops.normalize_extent(ctx, wp.array(pts, dtype=wp.vec2f, device=DEV), N, 1.44,
                             size=_f32(np.ones(3)))


# ---------------------------------------------------------------------------------------
# 2. Each twin == its original, on generator-produced inputs
# ---------------------------------------------------------------------------------------

def _make_track(**kw):
    return lambda: TrackGenerator(_tcfg(**kw), _rng())


def _make_gates(**kw):
    return lambda: GateGenerator(_gcfg(**kw), _rng())


def test_the_count_twin_equals_the_scalar_kernel(monkeypatch):
    dim, args = _capture(monkeypatch, _make_track(), ops_sampling._sample_counts_dyn_h_k)
    hi = int(args[4])
    assert hi == 13

    def scalar(lo):
        a = list(args)
        a[3] = int(lo)
        return _launch(ops_sampling._sample_counts_h_k, dim, a)[5]

    def twin(values):
        a = list(args)
        a[3] = _i32(values)
        out = _launch(ops_sampling._sample_counts_dyn_h_k, dim, a)
        assert out[3].tobytes() == np.asarray(values, np.int32).tobytes(), "lo written"
        return out[5]

    for v in (2, 5, 9, 13):
        assert twin([v] * E).tobytes() == scalar(v).tobytes(), v
    for v, want in ((1, 2), (0, 2), (-5, 2), (-2 ** 31, 2), (14, hi), (40, hi),
                    (2 ** 31 - 1, hi)):
        assert twin([v] * E).tobytes() == scalar(want).tobytes(), v
    values = [2, 13, 9, 0, 40, 7, 5, 11]
    got = twin(values)
    for e, v in enumerate(values):
        assert got[e] == scalar(min(max(v, 2), hi))[e], (e, v)
    # raising min never lowers an env's count (the same single randf)
    lo9, lo11 = twin([9] * E), twin([11] * E)
    assert np.all(lo11 >= lo9) and np.all((lo9 >= 9) & (lo9 <= hi))


@pytest.mark.parametrize("variant", ["plain", "scaled"])
def test_the_sized_grid_twins_equal_the_original(monkeypatch, variant):
    style = variant == "scaled"
    twin_k = ops_sampling._sample_grid_points_scaled_sized_h_k if style else \
        ops_sampling._sample_grid_points_sized_h_k
    orig_k = ops_sampling._sample_grid_points_scaled_h_k if style else \
        ops_sampling._sample_grid_points_h_k
    if style:
        dim = E
        P = 13
        nc, cell = ops.grid_cells(0.05)
        args = [_seeds(), 9781, 0, nc, nc * nc, float(cell), _f32(np.linspace(0.8, 1.2, E)), P,
                wp.empty(E * P, dtype=wp.int32, device=DEV),
                wp.empty(E * P, dtype=wp.vec2f, device=DEV),
                _f32(np.ones(E))]
    else:
        dim, args = _capture(monkeypatch, _make_track(), twin_k)
    base = args[:10]

    def twin(size):
        out = _launch(twin_k, dim, base + [_f32(size)])
        assert out[10].tobytes() == np.asarray(size, np.float32).tobytes(), "size written"
        return out[8:10]                                  # used, out

    want = _launch(orig_k, dim, base)[8:10]
    assert _same(twin(np.ones(E)), want), "size 1 != the original"
    got = twin(SIZES)
    if not style:
        assert float(base[6]) == 1.0                      # scale 1: size s == scale s
        for e, v in enumerate(SIZES):
            a = list(base)
            a[6] = _clamp_size(v)
            assert _same_rows(got, _launch(orig_k, dim, a)[8:10], e), (e, v)
    else:
        # per env: the original's (x * cs + nx) * sc, then one float32 multiply by size
        ref = want[1].reshape(E, -1, 2) * np.array([_clamp_size(v) for v in SIZES],
                                                   np.float32)[:, None, None]
        assert got[1].reshape(E, -1, 2).tobytes() == ref.astype(np.float32).tobytes()
        assert got[0].tobytes() == want[0].tobytes()


@pytest.mark.parametrize("where", ["track", "gates"])
def test_the_sized_normalize_twins_equal_the_original(monkeypatch, where):
    if where == "track":
        twin_k, orig_k = ops_geometry._normalize_extent_sized_k, \
            ops_geometry._normalize_extent_k
        make, t_idx, n_base, outs = _make_track(generator="polar"), 2, 3, (0,)
    else:
        twin_k, orig_k = st_gates._normalize_extent_to_sized_k, \
            st_gates._normalize_extent_to_k
        make, t_idx, n_base, outs = _make_gates(generator="polar"), 3, 5, (4,)
    dim, args = _capture(monkeypatch, make, twin_k)
    base = args[:n_base]
    target = float(base[t_idx])

    def twin(size):
        out = _launch(twin_k, dim, base + [_f32(size)])
        return [out[i] for i in outs]

    def orig(t):
        a = list(base)
        a[t_idx] = t
        out = _launch(orig_k, dim, a)
        return [out[i] for i in outs]

    assert _same(twin(np.ones(E)), orig(target)), "size 1 != the original"
    got = twin(SIZES)
    for e, v in enumerate(SIZES):
        t = float(np.float32(target) * np.float32(_clamp_size(v)))
        assert _same_rows(got, orig(t), e), (e, v)


def test_the_frame_twin_equals_the_scalar_kernel(monkeypatch):
    dim, args = _capture(monkeypatch, _make_gates(gate_width=0.3),
                         warp_gate._finalize_frame_dyn_k)
    outs = [0, 1, 2, 3, 4, 5, 6, 11]

    def scalar(w):
        a = list(args)
        a[9] = float(w)
        return [_launch(warp_gate._finalize_frame_k, dim, a)[i] for i in outs]

    def twin(values):
        a = list(args)
        a[9] = _f32(values)
        out = _launch(warp_gate._finalize_frame_dyn_k, dim, a)
        assert out[9].tobytes() == np.asarray(values, np.float32).tobytes()
        return [out[i] for i in outs]

    for v in (0.3, 0.0, 0.05, 1.0, -0.0, math.nan):
        assert _same(twin([v] * E), scalar(v)), v
    for v in (-0.2, -1e-30, -math.inf):
        assert _same(twin([v] * E), scalar(0.0)), v
    values = [0.0, 0.2, 0.5, -0.1, 0.05, 0.3, 1.0, 0.0]
    got = twin(values)
    G = dim[0] // E if isinstance(dim, tuple) else dim // E
    for e, v in enumerate(values):
        want = scalar(max(v, 0.0))
        for g, w in zip(got, want):
            n = g.shape[0] // E
            if n == G:                                    # per-gate arrays
                assert _row(g, e) == _row(w, e), (e, v)
        assert got[-1][e] == want[-1][e]                  # fallbacks


def _validity(seq, G: int, toggle=None) -> tuple:
    """valid / fail of the PR11 kernel (``toggle``: its crossing argument) or of the
    bars twin (``toggle=None``) on ``seq``'s arrays."""
    E_ = seq.count.shape[0]
    valid = wp.empty(E_, dtype=wp.int32, device=DEV)
    fail = wp.empty(E_, dtype=wp.int32, device=DEV)
    ins = [seq.position, seq.tangent, seq.forward, seq.orientation, seq.half_size,
           seq.left, seq.right, seq.count, G, 4, 0.05]
    if toggle is None:
        wp.launch(warp_gate._finalize_validity_bars_k, E_, ins + [0.0, valid, fail],
                  device=DEV)
    else:
        wp.launch(warp_gate._finalize_validity_k, E_, ins + [float(toggle), 0.0, valid,
                                                             fail], device=DEV)
    return valid.numpy().copy(), fail.numpy().copy()


@pytest.mark.parametrize("name", ["bezier", "hull", "polar", "voronoi", "checkpoint"])
def test_the_validity_twin_decides_as_the_pr11_kernel(name):
    """On real gate outputs with per-env bar widths 0 / 0.2 / 0.5: the twin's valid and
    fail bits equal the PR11 kernel's with the crossing check ON everywhere, and with it
    OFF on the zero-width envs (design A2)."""
    n = 64
    seeds = np.arange(n, dtype=np.int32) * 7919 + 13
    gen = GateGenerator(_gcfg(generator=name, num_envs=n, on_invalid="keep"), _rng(seeds))
    w = np.array([0.0, 0.2, 0.5, 0.0], np.float32)[np.arange(n) % 4]
    gen.params["gate_frames"].set("gate_width", _f32(w))
    seq = gen.generate()
    G = seq.position.shape[0] // n
    v_t, f_t = _validity(seq, G)
    v_on, f_on = _validity(seq, G, 1.0)
    v_off, f_off = _validity(seq, G, 0.0)
    assert v_t.tolist() == v_on.tolist() and f_t.tolist() == f_on.tolist()
    z = w == 0.0
    assert f_t[z].tolist() == f_off[z].tolist()
    # the pipeline's own bits are the twin's, and the crossing check bites at 0.5
    assert seq.fail_mask.numpy().tolist() == f_t.tolist()
    assert (f_t[w > 0] & 16).any(), "no bar crossing in the probe (vacuous)"


def test_the_validity_twin_on_posed_like_half_sizes():
    """Per-gate half sizes {0, +, -, NaN} with left / right by the frames' formula: the
    twin decides as the PR11 kernel with the crossing check on (NaN counts as a bar)."""
    n = 64
    gen = GateGenerator(_gcfg(num_envs=n, on_invalid="keep"),
                        _rng(np.arange(n, dtype=np.int32) * 31 + 5))
    seq = gen.generate()
    G = seq.position.shape[0] // n
    r = np.random.default_rng(0)
    hs = r.choice(np.array([0.0, 0.0, 0.1, -0.15, 0.3, np.nan], np.float32), size=n * G)
    half = (n // 2) * G
    hs[:half] = np.where(np.isnan(hs[:half]), 0.0, hs[:half]) * \
        (np.arange(half) // G % 3 == 0)
    pos, tan = seq.position.numpy(), seq.tangent.numpy()
    la = np.stack([-tan[:, 1], tan[:, 0], np.zeros(n * G, np.float32)], 1)
    seq.half_size.assign(hs)
    seq.left.assign((pos + hs[:, None] * la).astype(np.float32))
    seq.right.assign((pos - hs[:, None] * la).astype(np.float32))
    v_t, f_t = _validity(seq, G)
    v_on, f_on = _validity(seq, G, 1.0)
    assert v_t.tolist() == v_on.tolist() and f_t.tolist() == f_on.tolist()
    assert (f_t & 16).any(), "no crossing in the probe (vacuous)"


#: Discriminating payloads (design A3): signalling NaNs (an unguarded multiply quiets
#: them), quiet NaNs, the all-ones NaN, subnormals, infinities, finite values.
_PAYLOADS = np.array([[0x7F800001, 0xFF800123], [0x7FC00001, 0xFFC00000],
                      [0x7FFFFFFF, 0x3F800000], [0x00000003, 0x80000003],
                      [0x7F800000, 0xFF800000], [0x3F800000, 0xBF9DF3B6],
                      [0x00000000, 0x80000000], [0x42F6E979, 0xC2F6E979]], np.uint32)


@wp.kernel
def _unguarded_k(src: wp.array(dtype=wp.vec2f), n: int, size_e: wp.array(dtype=wp.float32),
                 out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    out[t] = src[t] * size_e[t // n]


def test_the_sized_copy_keeps_every_payload():
    """``_copy_sized_k`` at size 1 is a bitwise copy of any pattern; at size 1.3 a finite
    value is ``f32(v * s)`` and a non-finite one is kept bitwise; the negative control
    (an unguarded multiply quiets the signalling NaNs on CPU) shows the payloads
    discriminate."""
    n = _PAYLOADS.shape[0]
    src = wp.array(_PAYLOADS.view(np.float32).copy(), dtype=wp.vec2f, device=DEV)

    def run(kernel, size):
        out = wp.zeros(n, dtype=wp.vec2f, device=DEV)
        wp.launch(kernel, n, [src, n, _f32([size]), out], device=DEV)
        return out.numpy().view(np.uint32).reshape(n, 2)

    assert (run(wg_repulsive._copy_sized_k, 1.0) == _PAYLOADS).all()
    control = run(_unguarded_k, 1.0)
    assert (control[0] != _PAYLOADS[0]).any(), "the payloads do not discriminate"
    vals = _PAYLOADS.view(np.float32)
    fin = np.isfinite(vals).all(axis=1)                  # a point is finite or not
    assert fin.any() and (~fin).any()
    # size 1.3 and the clamped sizes (20 -> 10, 0.01 -> 0.1): a finite point is
    # f32(v * s), a non-finite one is kept bitwise (both components)
    for s, used in ((1.3, 1.3), (20.0, 10.0), (0.01, 0.1)):
        out = run(wg_repulsive._copy_sized_k, s)
        with np.errstate(invalid="ignore"):
            ref = (vals * np.float32(used)).astype(np.float32).view(np.uint32)
        assert (out[fin] == ref[fin]).all(), s
        assert (out[~fin] == _PAYLOADS[~fin]).all(), s


# ---------------------------------------------------------------------------------------
# 3. The generated CUDA source of each twin (on CPU; Warp's private codegen entry points:
#    updated with a Warp floor bump, never skipped)
# ---------------------------------------------------------------------------------------

_STMT = re.compile(r"^\s*(var_\w+)\s*=\s*(.*);\s*$")
_COMMENT = re.compile(r"^\s*//\s?(.*?)\s*<L \d+>\s*$")


def _cuda_forward(kernel) -> str:
    from warp._src import codegen as G  # noqa: PLC0415 - Warp's private codegen
    from warp._src import context as C  # noqa: PLC0415

    mod = kernel.module
    opts = dict(mod.options)
    opts["enable_backward"] = False
    C.ModuleBuilder(mod, opts).build_kernel(kernel)
    text = G.codegen_kernel(kernel, "cuda", opts)
    return text.split("// forward", 1)[-1].split("// reverse", 1)[0]


def _blocks(text: str) -> list:
    """``[(python source line, [(out, op, args, rhs)])]`` in order: each SSA statement of
    the generated forward body under the source line Warp emitted it for."""
    out, cur = [], ("<prologue>", [])
    for line in text.splitlines():
        m = _COMMENT.match(line)
        if m:
            out.append(cur)
            cur = (m.group(1).strip(), [])
            continue
        m = _STMT.match(line)
        if m:
            rhs = m.group(2).strip()
            mo = re.match(r"wp::(\w+)(?:<[^>]*>)?\(", rhs)
            op = mo.group(1) if mo else ("expr:" + re.sub(r"var_\w+", "v", rhs))
            cur[1].append((m.group(1), op, re.findall(r"var_\w+", rhs), rhs))
    out.append(cur)
    return out


def _flat(blocks) -> list:
    return [(o, op, args) for _, ss in blocks for o, op, args, _ in ss]


def _reduce(blocks, knob: str, replaces: "str | None") -> "tuple[list, list]":
    """The twin's statements reduced to its original's: the knob prelude (``knob =
    <arr>[e]`` and its ``wp.where`` clamps) dropped; a knob that REPLACES an argument
    renamed to it; a knob that MULTIPLIES contracted (``mul(X, K) -> X``). Returns
    ``(statements, contracted)`` with ``(out, op, producer of X, consumers)`` per
    contraction."""
    pre = re.compile(rf"^{knob}\s*=\s*(\w+_e\[e\]|wp\.where\()")
    kept, k_var = [], None
    for src, ss in blocks:
        if pre.match(src):
            k_var = ss[-1][0]
            continue
        kept.extend(ss)
    assert k_var is not None, "no knob prelude"
    alias = {k_var: f"var_{replaces}"} if replaces else {}
    prod = {o: op for o, op, _, _ in kept}
    users = collections.defaultdict(list)
    for o, op, args, _ in kept:
        for a in args:
            users[a].append(op)
    out, contracted = [], []
    for o, op, args, rhs in kept:
        args = [alias.get(a, a) for a in args]
        if replaces is None and k_var in args:
            data = [a for a in args if a != k_var]
            assert op == "mul" and len(set(data)) == 1, (o, op, rhs)
            alias[o] = data[0]
            contracted.append((o, op, prod.get(data[0], "argument"), users[o]))
            continue
        out.append((o, op, args))
    return out, contracted


def _canonical(stmts) -> list:
    """The op sequence with the SSA names renumbered by first definition."""
    ren, seq = {}, []
    for o, op, args in stmts:
        seq.append((op, tuple(ren.get(a, a if not re.fullmatch(r"var_\d+", a) else "?")
                              for a in args)))
        ren[o] = f"s{len(ren)}"
    return seq


def _fma_edges(stmts) -> set:
    """``mul -> add/sub`` edges (the pattern nvcc's ``--fmad=true`` contracts), each end
    named by its op and occurrence index."""
    count, name, prod = collections.Counter(), {}, {}
    for o, op, _ in stmts:
        count[op] += 1
        name[o] = (op, count[op])
        prod[o] = op
    return {(name[a], name[o]) for o, op, args in stmts if op in ("add", "sub")
            for a in args if prod.get(a) == "mul"}


#: (original, twin, knob local, the argument it replaces or None (it multiplies),
#: statements, mul -> add/sub edges) -- design section 1.4, E3.
#: Hashed sampling uses one init_at call instead of mul/add/rand_init: two fewer
#: statements and one fewer integer mul -> add edge than the frozen linear body.
_CUDA_PAIRS = {
    "counts": (ops_sampling._sample_counts_h_k, ops_sampling._sample_counts_dyn_h_k, "lo",
               "lo", 12, 0),
    "grid": (ops_sampling._sample_grid_points_h_k, ops_sampling._sample_grid_points_sized_h_k,
             "size", None, 48, 5),
    "grid_scaled": (ops_sampling._sample_grid_points_scaled_h_k,
                    ops_sampling._sample_grid_points_scaled_sized_h_k, "size", None, 51, 5),
    "normalize": (ops_geometry._normalize_extent_k, ops_geometry._normalize_extent_sized_k,
                  "size", None, 43, 5),
    "normalize_to": (st_gates._normalize_extent_to_k, st_gates._normalize_extent_to_sized_k,
                     "size", None, 60, 6),
    "frame": (warp_gate._finalize_frame_k, warp_gate._finalize_frame_dyn_k, "gate_width",
              "gate_width", 82, 7),
}


@pytest.mark.parametrize("pair", sorted(_CUDA_PAIRS))
def test_the_cuda_source_of_each_twin_reduces_to_its_original(pair):
    """Once the knob prelude is dropped and the knob renamed (replacing) or contracted
    (multiplying), the twin's generated CUDA forward body IS its original's: the same op
    sequence (operands included, up to SSA renumbering) and the same ``mul -> add/sub``
    edges; every contracted multiply takes a ``mul`` result or an argument and feeds no
    add / sub, so nvcc forms the same FMAs and size 1 stays bit-exact."""
    orig, twin, knob, replaces, n_stmts, n_edges = _CUDA_PAIRS[pair]
    a = _flat(_blocks(_cuda_forward(orig)))
    b, contracted = _reduce(_blocks(_cuda_forward(twin)), knob, replaces)
    assert len(a) == n_stmts
    assert _canonical(a) == _canonical(b)
    assert _fma_edges(a) == _fma_edges(b) and len(_fma_edges(a)) == n_edges
    if replaces is None:
        assert contracted, "no contracted multiply"
    for _, op, producer, consumers in contracted:
        assert producer in ("mul", "argument"), producer
        assert not set(consumers) & {"add", "sub"}, consumers


def test_the_cuda_source_of_the_validity_twin_is_the_pr11_kernel():
    """``_finalize_validity_bars_k`` == ``_finalize_validity_k`` once the ``bars``
    bookkeeping (its init, the ``hs != 0.0`` flag and their loop merges: integer selects,
    no float arithmetic) is dropped and its loop guard ``bars != 0`` is matched with the
    original's ``gate_width > 0.0``: same op sequence, same ``mul -> add/sub`` edges."""
    a_blocks = _blocks(_cuda_forward(warp_gate._finalize_validity_k))
    b_blocks = _blocks(_cuda_forward(warp_gate._finalize_validity_bars_k))
    tainted, dropped, b = set(), [], []
    guard_b = guard_a = None
    for src, ss in b_blocks:
        for o, op, args, rhs in ss:
            if src == "if bars != 0:":
                assert guard_b is None and op.startswith("expr:")
                guard_b = o
                continue
            if src in ("bars = int(0)", "if hs != 0.0:", "bars = int(1)") or \
                    tainted & set(args):
                tainted.add(o)
                dropped.append(op)
                continue
            b.append((o, op, args))
    a = []
    for src, ss in a_blocks:
        for o, op, args, rhs in ss:
            if src == "if gate_width > 0.0:":
                assert guard_a is None and op.startswith("expr:")
                guard_a = o
                continue
            a.append((o, op, args))
    assert guard_a is not None and guard_b is not None
    assert set(dropped) <= {"int", "where", "expr:(v != v)"}, dropped
    assert _canonical(a) == _canonical(b)
    assert _fma_edges(a) == _fma_edges(b)


def test_the_cuda_source_of_the_sized_copy_has_one_guarded_multiply():
    """``_copy_sized_k`` (no original: it replaces a memcpy): exactly one ``mul``, whose
    operands are the finite-gated value ``where(fin, v, vec2(0, 0))`` and the clamped
    size, consumed only by the final ``where`` whose TRUE arm is the loaded ``v`` and
    whose condition is ``size == 1 || !fin``: no select fold can turn the size-1 copy
    into a multiply of a NaN (design section 1.3, A3)."""
    blocks = _blocks(_cuda_forward(wg_repulsive._copy_sized_k))
    stmts = [(o, op, args, rhs) for _, ss in blocks for o, op, args, rhs in ss]
    by_out = {o: (op, args, rhs) for o, op, args, rhs in stmts}
    muls = [s for s in stmts if s[1] == "mul"]
    assert len(muls) == 1
    m_out, _, (w, size), _ = muls[0]
    w_op, w_args, _ = by_out[w]
    assert w_op == "where"
    fin, v, zero = w_args
    assert by_out[zero][0] == "vec_t" and by_out[v][0] == "copy"
    assert by_out[by_out[v][1][0]][0] == "load"                   # v = src[t]
    assert by_out[size][0] == "where"                              # the clamped size
    users = [s for s in stmts if m_out in s[2]]
    assert len(users) == 1 and users[0][1] == "where"
    cond, true_arm, false_arm = users[0][2]
    assert (true_arm, false_arm) == (v, m_out)
    # the condition: (size == 1.0) || !fin (a short-circuit: cond assigned twice)
    srcs = {a for d in stmts if d[0] == cond for a in d[2]} - {cond}
    eq = [x for x in stmts if x[0] in srcs and x[1] == "expr:(v == v)"]
    assert len(eq) == 1 and size in eq[0][2]
    assert [x for x in stmts if x[0] in srcs and x[1] == "unot"] and \
        all(x[2] == [fin] for x in stmts if x[0] in srcs and x[1] == "unot")


# ---------------------------------------------------------------------------------------
# 4. Per env == per-env scalar runs
# ---------------------------------------------------------------------------------------

def _track_maker(name: str, field: str = "size", n: int = E, seeds=SEEDS, gp=None,
                 **kw):
    """``make(v)``: a TrackGenerator built with ``generator_params={**gp, field: v}``
    (the truncation warning silenced)."""
    def make(v):
        params = dict(gp or {})
        if v is not None:
            params[field] = v
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"generator\.size.*truncates")
            return TrackGenerator(_tcfg(generator=name, num_envs=n,
                                        generator_params=params or None, **kw),
                                  _rng(seeds))
    return make


@pytest.mark.parametrize("name", ["bezier", "hull", "polar", "voronoi", "checkpoint"])
def test_size_per_env_equals_scalar_runs(name):
    _check_per_env(_track_maker(name), "generator", "size", SIZES, f"{name}.size",
                   used=_clamp_size)




def test_size_per_env_on_repulsive():
    seeds = SEEDS[:4]
    make = _track_maker("repulsive", n=4, seeds=seeds, num_points=64,
                        gp={"repulsive_stages": (16, 32, 64)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # 20 -> 10 truncates (warned)
        _check_per_env(make, "generator", "size", (1.0, 0.7, 1.3, 20.0), "repulsive.size",
                       used=_clamp_size, n=4)


@pytest.mark.parametrize("name", ["bezier", "hull"])
def test_min_num_points_per_env_equals_scalar_runs(name):
    """Incl. out-of-range array values, clamped to [2, max_num_points] at use (the scalar
    run is built with the clamped value: construction refuses the others)."""
    values = (9, 2, 13, 5, 40, 1, 11, -3)
    _check_per_env(_track_maker(name, "min_num_points"), "generator", "min_num_points",
                   values, f"{name}.min_num_points", used=lambda v: min(max(v, 2), 13),
                   dtype=np.int32)


@pytest.mark.parametrize("name", ["bezier", "hull"])
def test_size_per_env_on_point_gate_generators(name):
    def make(v):
        return GateGenerator(_gcfg(generator=name, size=1.0 if v is None else v), _rng())

    _check_per_env(make, "generator", "size", SIZES, f"gates {name}.size", used=_clamp_size)


@pytest.mark.parametrize("name", ["polar", "voronoi", "checkpoint"])
def test_normalize_size_per_env_on_renormalized_gate_generators(name):
    def make(v):
        return GateGenerator(_gcfg(generator=name, size=1.0 if v is None else v), _rng())

    _check_per_env(make, "normalize", "size", SIZES, f"gates {name} normalize.size",
                   used=_clamp_size)


@pytest.mark.parametrize("name", ["bezier", "polar"])
def test_gate_width_per_env_equals_pipelines_built_with_each_width(name):
    """``gate_frames.gate_width`` per env: every GateSequence field, the valid flag and
    the fail bits (bar crossings included) equal a pipeline BUILT with that width (the
    PR13c static toggle on: the per-env crossing check decides the same); a negative
    array value is clamped to 0."""
    values = (0.0, 0.2, 0.5, 0.0, 0.1, 0.3, -0.2, 0.05)

    def make(v):
        return GateGenerator(_gcfg(generator=name, gate_width=0.0 if v is None else v),
                             _rng())

    _check_per_env(make, "gate_frames", "gate_width", values, f"gates {name}.gate_width",
                   used=lambda v: max(v, 0.0))
    # non-vacuity of the crossing check: bars of 0.5 fail it somewhere
    gen = make(0.5)
    assert (gen.generate().fail_mask.numpy() & 16).any() or name == "polar"


# ---------------------------------------------------------------------------------------
# 5. Repulsive: dynamic size consistently reserves absolute road width
# ---------------------------------------------------------------------------------------

def test_repulsive_dynamic_size_matches_a_fresh_constructor():
    """The same live size produces the same constrained geometry regardless of the
    initial constructor size. Width reservation makes growth size-dependent."""
    seeds = SEEDS[:4]
    gen = TrackGenerator(_tcfg(generator="repulsive", num_envs=4, **REP), _rng(seeds))
    gen.generate()
    r1 = gen.program.tap("raw.xy").numpy().copy()
    assert np.isfinite(r1).all()
    differs = False
    for size in (0.7, 1.3, 2.0):
        gen.params["generator"].set("size", size)
        actual = _snap(gen.generate())
        raw = gen.program.tap("raw.xy").numpy().copy()
        fresh = TrackGenerator(_tcfg(generator="repulsive", num_envs=4, size=size, **REP),
                               _rng(seeds))
        expected = _snap(fresh.generate())
        assert raw.tobytes() == fresh.program.tap("raw.xy").numpy().tobytes()
        assert actual.keys() == expected.keys()
        assert all(actual[key].tobytes() == expected[key].tobytes() for key in actual)
        differs |= raw.tobytes() != (r1 * np.float32(size)).astype(np.float32).tobytes()
    assert differs, "absolute road width must participate in Repulsive growth"


def test_repulsive_physically_infeasible_size_fails_closed():
    gen = TrackGenerator(_tcfg(generator="repulsive", num_envs=4, size=0.1, **REP),
                         _rng(SEEDS[:4]))
    result = gen.generate()
    assert not result.valid.numpy().any()
    assert (result.fail_mask.numpy() != 0).all()
    assert result.count.numpy().tolist() == [0] * 4
    assert np.isnan(result.center.numpy()).all()


# ---------------------------------------------------------------------------------------
# 6. size vs scale
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["bezier", "hull"])
def test_size_equals_scale_on_the_corner_grid(name):
    """At ``scale=1`` the corner grid's ``((x*cs + nx) * 1) * s`` IS ``(x*cs + nx) * s``:
    a size-s run equals a scale-s run bitwise (raw loop and Track)."""
    for s in (0.7, 1.6):
        a = TrackGenerator(_tcfg(generator=name, size=s), _rng())
        b = TrackGenerator(_tcfg(generator=name, scale=s), _rng())
        assert a.program.tap("raw.xy").numpy().tobytes() == \
            b.program.tap("raw.xy").numpy().tobytes(), s
        assert not _differs(_snap(a.generate()), _snap(b.generate())), s


@pytest.mark.parametrize("name", ["polar", "voronoi", "checkpoint"])
def test_size_is_scale_within_rounding_on_the_normalized_generators(name):
    """The normalization target is ``f32(scale * TE) * size`` (vs ``f32(s * TE)``): the
    raw loops agree within 4 ulp, with equal valid / count (design E5)."""
    for s in (0.7, 1.6):
        a = TrackGenerator(_tcfg(generator=name, size=s), _rng())
        b = TrackGenerator(_tcfg(generator=name, scale=s), _rng())
        ra, rb = a.program.tap("raw.xy").numpy(), b.program.tap("raw.xy").numpy()
        np.testing.assert_array_max_ulp(ra, rb, maxulp=4)
        ta, tb = a.generate(), b.generate()
        assert ta.valid.numpy().tolist() == tb.valid.numpy().tolist()
        assert ta.count.numpy().tolist() == tb.count.numpy().tolist()


# ---------------------------------------------------------------------------------------
# 7. randomize on the new axes == eager twins; the plan rules
# ---------------------------------------------------------------------------------------

def _assert_randomized(facade, preset, kw: dict, randomize: dict, n: int = E) -> None:
    """A randomized pipeline == its unrandomized twin whose fields are set to the draws
    (``in_effect``), over two generations and a masked advance (unmasked envs keep rows
    and draws, masked ones redraw); the draws vary per env and move the batch."""
    seeds = SEEDS[:n]
    spec = preset(num_envs=n, device=DEV, randomize=randomize, **kw)
    gen = facade(spec, _rng(seeds))
    twin = facade(preset(num_envs=n, device=DEV, **kw), _rng(seeds))
    keys = [tuple(k.rsplit(".", 1)) for k in randomize]

    def sync_twin(**adv):
        for sid, f in keys:
            twin.params[sid].set(f, gen.params[sid].in_effect(f))
        return twin.generate(**adv)

    got = _snap(gen.generate())
    draws = {k: gen.params[sid][f].numpy().copy() for k, (sid, f) in zip(randomize, keys)}
    for k, (lo, hi) in randomize.items():
        d = draws[k]
        assert np.all((d >= np.float32(lo)) & (d <= np.float32(hi))), (k, d)
        assert len(set(d.tolist())) > 1, f"{k}: the draws do not vary per env"
        sid, f = k.rsplit(".", 1)
        assert gen.params[sid].in_effect(f) is gen.params[sid][f]      # drawn: itself
    assert not _differs(got, _snap(sync_twin())), "randomized != its twin at the draws"
    plain = _snap(facade(preset(num_envs=n, device=DEV, **kw), _rng(seeds)).generate())
    assert _differs(got, plain), "the draws do not move the batch (vacuous)"
    got = _snap(gen.generate(advance=True))
    assert not _differs(got, _snap(sync_twin(advance=True))), "second generation"
    draws = {k: gen.params[sid][f].numpy().copy() for k, (sid, f) in zip(randomize, keys)}
    mask_np = (np.arange(n) % 2 == 0).astype(np.int32)
    mask = wp.array(mask_np, dtype=wp.int32, device=DEV)
    got2 = _snap(gen.generate(advance=True, mask=mask))
    keep = mask_np == 0
    for k, (sid, f) in zip(randomize, keys):
        now = gen.params[sid][f].numpy()
        assert now[keep].tobytes() == draws[k][keep].tobytes(), f"{k}: unmasked redrawn"
        assert (now[~keep] != draws[k][~keep]).any(), f"{k}: masked envs not redrawn"
    for e in np.flatnonzero(keep):
        assert not _rows_differ(got2, got, int(e), n), f"unmasked env {e} moved"
    assert not _differs(got2, _snap(sync_twin(advance=True, mask=wp.clone(mask))))


_RANDOMIZE_CASES = {
    "track.bezier.size": (TrackGenerator, presets.track, {}, {"generator.size": (0.7, 1.6)}),
    "track.polar.size": (TrackGenerator, presets.track, {"generator": "polar"},
                         {"generator.size": (0.7, 1.6)}),
    "track.repulsive.size": (TrackGenerator, presets.track,
                             {"generator": "repulsive", "num_points": 64,
                              "generator_params": {"repulsive_stages": (16, 32, 64)}},
                             {"generator.size": (0.7, 1.4)}),
    "gates.bezier.size": (GateGenerator, presets.gates, {},
                          {"generator.size": (0.7, 1.6)}),
    "gates.polar.normalize.size": (GateGenerator, presets.gates, {"generator": "polar"},
                                   {"normalize.size": (0.7, 1.6)}),
    "gates.bezier.gate_width": (GateGenerator, presets.gates, {},
                                {"gate_frames.gate_width": (0.0, 0.5)}),
}


@pytest.mark.parametrize("case", sorted(_RANDOMIZE_CASES))
def test_randomize_on_the_new_axes_equals_eager_twins(case):
    facade, preset, kw, randomize = _RANDOMIZE_CASES[case]
    n = 4 if "repulsive" in case else E
    _assert_randomized(facade, preset, kw, randomize, n=n)


def test_a_randomized_bar_width_is_crossing_checked():
    """``gate_frames.gate_width`` drawn on a default validity builds (no PlanError) and
    every env with a bar is crossing-checked: the bits equal a twin built per width."""
    spec = presets.gates(num_envs=E, device=DEV,
                         randomize={"gate_frames.gate_width": (0.3, 0.6)})
    gen = GateGenerator(spec, _rng())
    seq = _snap(gen.generate())
    assert (seq["fail_mask"] & 16).any(), "no crossing among the drawn bars (vacuous)"
    widths = gen.params["gate_frames"].in_effect("gate_width").numpy()
    for e, w in enumerate(widths):
        ref = _snap(GateGenerator(_gcfg(gate_width=float(w)), _rng()).generate())
        assert not _rows_differ(seq, ref, e), (e, w)


def test_the_randomize_plan_rules_on_the_new_axes():
    with pytest.raises(PlanError, match=r"^generator\.min_num_points is int32: "
                                        r"randomize\.uniform draws float32 fields in 0\.x"):
        build(presets.track(num_envs=E, device=DEV,
                            randomize={"generator.min_num_points": (2.0, 9.0)}))
    for rng_ in ((0.05, 1.0), (1.0, 12.0)):
        with pytest.raises(PlanError, match=r"lies outside generator\.size's declared "
                                            r"range \[0\.1, 10\.0\]"):
            build(presets.track(num_envs=E, device=DEV, randomize={"generator.size": rng_}))
    with pytest.raises(PlanError, match=r"has no param 'size'.*\(external ingests your "
                                        r"geometry as given: scale centerline\.xy in your "
                                        r"graph\)"):
        build(presets.track(num_envs=E, device=DEV, generator="external",
                            randomize={"generator.size": (0.8, 1.2)}))
    with pytest.raises(PlanError, match=r"has no param 'size'.*\(this gate generator is "
                                        r"re-normalized by normalize\.extent: use "
                                        r"normalize\.size, whose initial value is "
                                        r"GateGenConfig\.size / presets\.gates\(size=\)\)"):
        build(presets.gates(num_envs=E, device=DEV, generator="polar",
                            randomize={"generator.size": (0.8, 1.2)}))
    with pytest.raises(PlanError, match=r"generator\.min_num_points is a static field.*"
                                        r"\(on gate generators it is the gate count "
                                        r"capacity \(min_gates\): not per env in 0\.x\)"):
        build(presets.gates(num_envs=E, device=DEV,
                            randomize={"generator.min_num_points": (2.0, 9.0)}))


def test_the_build_time_truncation_warning():
    rep = dict(generator="repulsive", num_points=64,
               generator_params={"repulsive_stages": (16, 32, 64)})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        build(presets.track(num_envs=2, device=DEV, randomize={"generator.size": (0.8, 4.0)},
                            **rep))
    trunc = [w for w in rec if "truncates at n_max" in str(w.message)]
    assert len(trunc) == 1 and trunc[0].filename == __file__
    msg = str(trunc[0].message)
    # repulsive's nominal perimeter is exact from its params: 3.0 * 2 pi * r_init
    nominal = 3.0 * 2 * math.pi * 0.35 * 4.9029 / 4.0
    need = math.ceil(4.0 * nominal / 0.06) + 1
    assert re.search(rf"generator\.size drawn up to 4\.0 truncates at n_max=384 "
                     rf"\(nominal raw perimeter ~{4.0 * nominal:.1f} > \(n_max - 1\) \* "
                     rf"spacing = 22\.98\): raise n_max to >= {need} or lower the range",
                     msg), msg
    # ... a spec with that n_max does not warn; nor does bezier's (0.8, 1.2); a
    # construction value above the threshold does.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        build(presets.track(num_envs=2, device=DEV, n_max=need,
                            randomize={"generator.size": (0.8, 4.0)}, **rep))
        build(presets.track(num_envs=E, device=DEV,
                            randomize={"generator.size": (0.8, 1.2)}))
        build(presets.track(num_envs=E, device=DEV, size=1.0))
    with pytest.warns(UserWarning, match=r"generator\.size=4\.0 truncates at n_max=384"):
        build(presets.track(num_envs=E, device=DEV, size=4.0))
    with pytest.warns(UserWarning, match=r"generator\.size=3\.0 truncates"):
        TrackGenerator(_tcfg(size=3.0, **{k: v for k, v in rep.items()
                                          if k != "generator_params"},
                             repulsive_stages=(16, 32, 64)), _rng())


# ---------------------------------------------------------------------------------------
# 8. The launch deltas of every pipeline (design section 6.2, tests/_pr13d_delta.py)
# ---------------------------------------------------------------------------------------

_ZU = dict(z_profile="uniform", z_min=-0.2, z_max=0.3, z_valid_grade=0.5)
_ZW = dict(z_profile="random_walk", z_base=0.05, z_min=-0.2, z_max=0.3, z_max_step=0.4,
           z_valid_grade=0.8, z_control_points=7)
_ZN = dict(z_profile="noise", z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.2,
           z_noise_harmonics=4, z_valid_grade=0.6)
_I, _F = "i32", "f32"
#: case -> (TrackGenConfig kwargs, PR13c count, its latches, PR13d count, the PR13d
#: latch block): the non-latch records are unchanged (PR13c - its latches == PR13d -
#: the block).
_TRACK_DELTAS = {
    "bezier": ({}, 154, 2, 156, [_I, _F, _F, _F]),
    "hull": (dict(generator="hull"), 152, 1, 154, [_I, _F, _F]),
    "polar": (dict(generator="polar"), 145, 2, 146, [_F, _F, _F]),
    "voronoi": (dict(generator="voronoi"), 153, 2, 154, [_F, _F, _F]),
    "checkpoint": (dict(generator="checkpoint"), 149, 3, 150, [_F] * 4),
    "checkpoint_k1_clip": (dict(generator="checkpoint", checkpoint_best_of_k=1,
                                checkpoint_clip_fallback=True), 149, 3, 150, [_F] * 4),
    "bezier_z_uniform": (_ZU, 160, 4, 162, [_I, _F, _F, _F, _F, _F]),
    "bezier_z_random_walk": (_ZW, 161, 5, 163, [_I, _F, _F, _F, _F, _F, _F]),
    "bezier_z_noise": (_ZN, 159, 5, 161, [_I, _F, _F, _F, _F, _F, _F]),
}
_TRACK_TWINS = {"bezier": ("_sample_counts_dyn_h_k", "_sample_grid_points_sized_h_k"),
                "hull": ("_sample_counts_dyn_h_k", "_sample_grid_points_sized_h_k"),
                "polar": ("_normalize_extent_sized_k",),
                "voronoi": ("_normalize_extent_sized_k", "_normalize_extent_sized_k"),
                "checkpoint": ("_normalize_extent_sized_k",)}


def _names(trace) -> list:
    return [r.kernel.rsplit(".", 1)[-1] for r in trace]


def _rows(trace) -> list:
    return [(r.op, r.kernel.replace("tests._reference_generators.", "track_gen._src."), r.dim, r.scalars) for r in trace]


def _traces(program) -> tuple:
    ctx = LaunchContext.for_call(DEV)
    E_ = program.layout.num_envs
    with trace_launches() as plain:
        program.enqueue(ctx)
    mask = wp.array((np.arange(E_) % 2 == 0).astype(np.int32), dtype=wp.int32, device=DEV)
    with trace_launches() as masked:
        program.enqueue(ctx, advance=True, mask=mask)
    return plain, masked


@pytest.mark.parametrize("case", sorted(_TRACK_DELTAS))
def test_the_track_launch_deltas(case):
    kw, pr13c, pr13c_latches, pr13d, block = _TRACK_DELTAS[case]
    gen = TrackGenerator(_tcfg(on_invalid="keep", **kw), _rng())
    plain, masked = _traces(gen.program)
    assert len(plain) == pr13d
    assert pr13d - len(block) == pr13c - pr13c_latches     # only latches were added
    assert assert_latch_block(gen.program, plain, masked) == block
    assert len(masked) == len(plain) + 3
    names = _names(plain)
    name = kw.get("generator", "bezier")
    twins = list(_TRACK_TWINS[name])
    assert [n for n in names if n in set(twins)] == twins, names
    for orig in ("_sample_counts_h_k", "_sample_grid_points_h_k", "_sample_grid_points_scaled_h_k",
                 "_normalize_extent_k"):
        assert orig not in names, orig
    # The generator's latches lead the block (the elevation's follow).
    gl = list(GENERATOR_LATCHES[name])
    assert block[:len(gl)] == gl


def test_the_track_launch_deltas_against_the_legacy_composition():
    """The rule (``pr13d_sequence``) on HEAD's legacy (scalar) composition reproduces the
    default bezier trace record for record (``tests/test_dynamic_knobs.py`` runs it on
    every built-in)."""
    gen = TrackGenerator(_tcfg(on_invalid="keep"), _rng())
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as program:
        gen.program.enqueue(ctx)
    cfg = gen.config
    track, scratch = _scalar_pipeline._inflate_warp_alloc(cfg)
    seed = wp.empty(E, dtype=wp.int32, device=DEV)
    with trace_launches() as pr9:
        gen.seeds.effective(seed, None, ctx=ctx)
        _scalar_pipeline._run_pipeline(cfg, seed, out=track, scratch=scratch, ctx=ctx)
        gen.seeds._write_generation(track.generation, None, ctx)
    assert len(pr9) == 152
    want = pr13d_sequence(_rows(pr9), latch_dtypes(gen.program), E, legacy=True)
    assert _rows(program) == want


def test_external_gains_no_launch():
    gen = TrackGenerator(presets.track(num_envs=E, device=DEV, generator="external",
                                      on_invalid="keep"),
                         _rng())
    plain, masked = _traces(gen.program)
    assert len(plain) == 20
    assert assert_latch_block(gen.program, plain, masked) == ["i32"]    # centerline.count


#: case -> (presets.gates kwargs, PR13c count, PR13d count).
_GATE_DELTAS = {
    "bezier/ccw/flat": (dict(), 14, 16),
    "hull/ccw/flat": (dict(generator="hull"), 14, 16),
    "polar/ccw/flat": (dict(generator="polar"), 16, 18),
    "polar/raw/flat": (dict(generator="polar", order="raw"), 16, 18),
    "checkpoint/ccw/flat": (dict(generator="checkpoint"), 16, 18),
    "voronoi/ccw/flat": (dict(generator="voronoi"), 17, 19),
    "bezier/ccw/noise": (dict(elevation=("noise", {"z_base": 0.1, "z_max": 1.5,
                                                   "z_noise_amplitude": 0.5})), 18, 20),
}


@pytest.mark.parametrize("case", sorted(_GATE_DELTAS))
def test_the_gate_launch_deltas(case):
    kw, pr13c, pr13d = _GATE_DELTAS[case]
    name, _, z = case.split("/")
    gen = GateGenerator(presets.gates(num_envs=E, device=DEV, on_invalid="keep", **kw), _rng())
    plain, masked = _traces(gen.program)
    assert len(plain) == pr13d == pr13c + 2
    assert assert_latch_block(gen.program, plain, masked) == gate_latches(name, z)
    names = _names(plain)
    assert "_finalize_frame_dyn_k" in names and "_finalize_frame_k" not in names
    v = [r for r in plain if r.kernel.endswith("._finalize_validity_bars_k")]
    assert len(v) == 1 and "_finalize_validity_k" not in names
    G = int(gen._gate_sequence.position.shape[0] // E)
    # the PR11 record minus the toggle (scalar 11): z_valid_grade moves to index 11
    assert v[0].scalars == ((8, G), (9, 4), (10, 0.05), (11, 0.0)), v[0].scalars
    if name in ("bezier", "hull"):
        assert "_sample_counts_h_k" in names and "_sample_grid_points_sized_h_k" in names
    else:
        assert "_normalize_extent_to_sized_k" in names
        assert "_normalize_extent_to_k" not in names


def test_the_stations_launch_delta():
    """A stations source's track generator latches its four fields (bezier) and
    gate_frames its width: 29 -> 32 (+3: size, min_num_points and the width over PR13c's
    rad / clamp)."""
    gen = GateGenerator(presets.gates(num_envs=E, device=DEV, source="stations",
                                     on_invalid="keep"), _rng())
    plain, masked = _traces(gen.program)
    assert len(plain) == 32
    assert assert_latch_block(gen.program, plain, masked) == [_I, _F, _F, _F, _F]
    names = _names(plain)
    assert "_sample_counts_dyn_h_k" in names and "_sample_grid_points_sized_h_k" in names


# ---------------------------------------------------------------------------------------
# 9. StageParams.in_effect / bounds
# ---------------------------------------------------------------------------------------

def test_in_effect_of_a_latched_field_under_set_and_masked_regeneration():
    n = 4
    gen = TrackGenerator(_tcfg(num_envs=n), _rng(SEEDS[:n]))
    sp_ = gen.params["generator"]
    view = sp_.in_effect("size")
    assert view is not sp_["size"] and view.dtype is wp.float32 and view.shape == (n,)
    assert view.numpy().tolist() == [1.0] * n
    sp_.set("size", 1.3)
    assert sp_.in_effect("size").numpy().tolist() == [1.0] * n     # not latched yet
    before = _snap(gen._track)
    mask = wp.array([1, 0, 1, 0], dtype=wp.int32, device=DEV)
    after = _snap(gen.generate(advance=True, mask=mask))
    got = sp_.in_effect("size")
    assert got is view                                              # stable pointer
    assert got.numpy().tolist() == [np.float32(1.3), 1.0, np.float32(1.3), 1.0]
    for e in (1, 3):
        assert not _rows_differ(after, before, e, n), e             # their rows: size 1
    # the regenerated rows are a size-1.3 run's at their effective seeds
    twin = TrackGenerator(_tcfg(num_envs=n, size=1.3), _rng(SEEDS[:n]))
    twin_rows = _snap(twin.generate(advance=True, mask=wp.clone(mask)))
    for e in (0, 2):
        assert not _rows_differ(after, twin_rows, e, n), e
    # an int32 field
    sp_.set("min_num_points", 11)
    gen.generate(advance=True, mask=mask)
    assert sp_.in_effect("min_num_points").numpy().tolist() == [11, 9, 11, 9]
    assert sp_.in_effect("min_num_points").dtype is wp.int32


def test_in_effect_of_drawn_fields_and_of_a_randomizers_bounds():
    n = 4
    spec = presets.track(num_envs=n, device=DEV, randomize={"generator.size": (0.8, 1.6)})
    gen = TrackGenerator(spec, _rng(SEEDS[:n]))
    sp_, rnd = gen.params["generator"], gen.params["randomize.generator.size"]
    assert sp_.in_effect("size") is sp_["size"]                     # the drawn array
    d0 = sp_.in_effect("size").numpy().copy()
    rnd.set("hi", 1.9)
    mask = wp.array([1, 0, 1, 0], dtype=wp.int32, device=DEV)
    gen.generate(advance=True, mask=mask)
    d1 = sp_.in_effect("size").numpy()
    assert d1[[1, 3]].tobytes() == d0[[1, 3]].tobytes()             # unmasked: same bits
    assert (d1[[0, 2]] != d0[[0, 2]]).all()
    assert rnd.in_effect("hi").numpy().tolist() == [np.float32(1.9), np.float32(1.6),
                                                    np.float32(1.9), np.float32(1.6)]
    assert rnd.bounds("hi") == rnd.bounds("lo") == (0.1, 10.0)


def test_in_effect_out_and_its_checks():
    n = 4
    gen = TrackGenerator(_tcfg(num_envs=n), _rng(SEEDS[:n]))
    sp_ = gen.params["generator"]
    sp_.set("size", _f32([1.2, 0.9, 1.0, 2.0]))
    gen.generate()
    out = wp.zeros(n, dtype=wp.float32, device=DEV)
    assert sp_.in_effect("size", out=out) is out
    assert out.numpy().tobytes() == sp_.in_effect("size").numpy().tobytes()
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as t:
        sp_.in_effect("size", out=out, stream=ctx.stream)
    assert [r.op for r in t] == ["copy"]                            # one capturable copy
    out_i = wp.zeros(n, dtype=wp.int32, device=DEV)
    sp_.in_effect("min_num_points", out=out_i)
    assert out_i.numpy().tolist() == [9] * n
    for bad, match in ((wp.zeros(n + 1, dtype=wp.float32, device=DEV), r"shape \(4,\)"),
                       (wp.zeros(n, dtype=wp.int32, device=DEV), r"float32 array"),
                       (np.zeros(n, np.float32), r"must be a wp\.array"),
                       (wp.zeros((n, 2), dtype=wp.float32, device=DEV)[:, 0],
                        r"contiguous")):
        with pytest.raises(ValueError, match=match):
            sp_.in_effect("size", out=bad)
    with pytest.raises(KeyError, match=r"'max_num_points' is a static parameter"):
        sp_.in_effect("max_num_points")
    with pytest.raises(KeyError, match=r"no dynamic parameter 'nope'"):
        sp_.bounds("nope")


def test_bounds_and_out_of_range_writes():
    """``in_effect`` returns array writes as written; the rows equal scalar runs at
    ``clamp(in_effect, *bounds)`` (design C5)."""
    n = 4
    gen = TrackGenerator(_tcfg(num_envs=n), _rng(SEEDS[:n]))
    sp_ = gen.params["generator"]
    assert sp_.bounds("size") == (0.1, 10.0)
    assert sp_.bounds("min_num_points") == (2, 13)
    assert sp_.bounds("rad") == (0.0, 1.0) and sp_.bounds("handle_clamp_frac") == (0.0, None)
    hull = TrackGenerator(_tcfg(num_envs=n, generator="hull", max_num_points=11),
                          _rng(SEEDS[:n]))
    assert hull.params["generator"].bounds("min_num_points") == (2, 11)
    gates = GateGenerator(_gcfg(num_envs=n), _rng(SEEDS[:n]))
    assert gates.params["gate_frames"].bounds("gate_width") == (0.0, None)
    sizes, counts = [20.0, 1.0, 0.01, 1.0], [40, 9, 1, 9]
    sp_.set("size", _f32(sizes))
    sp_.set("min_num_points", _i32(counts))
    got = _snap(gen.generate())
    ie_s, ie_c = sp_.in_effect("size").numpy(), sp_.in_effect("min_num_points").numpy()
    assert ie_s.tolist() == [np.float32(v) for v in sizes] and ie_c.tolist() == counts
    lo, hi = sp_.bounds("size")
    clo, chi = sp_.bounds("min_num_points")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"generator\.size.*truncates")
        for e in range(n):
            s = float(np.clip(ie_s[e], lo, hi))
            c = int(np.clip(ie_c[e], clo, chi))
            ref = _snap(TrackGenerator(_tcfg(num_envs=n, size=s, min_num_points=c),
                                       _rng(SEEDS[:n])).generate())
            assert not _rows_differ(got, ref, e, n), (e, s, c)


def test_in_effect_timing_in_segmented_programs(plugin_registry):
    """(a) a latched field of a stage after a HOST segment: a write enqueued after segment
    0 is seen by ``in_effect`` (and used) only from the next generation; (b) a drawn
    field whose randomizer sits behind the HOST segment: after segment 0 ``in_effect``
    holds the previous generation's draws, after the generation the new ones."""
    from tests._bounded_toys import HOST_RING, register_bounded_toys
    from tests._plugin_toys import register_stage_toys

    register_stage_toys(plugin_registry)
    register_bounded_toys(plugin_registry)
    ctx = LaunchContext.for_call(DEV)
    noise = dict(z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.3)
    program = build(presets.track(num_envs=E, device=DEV, generator="toy.wobble_host",
                                  elevation=("noise", noise)), seeds=_seeds())
    assert len(program.segments) == 2 and program.segments[0].host
    elev = program.params["elevation"]
    program.enqueue_segment(0, ctx)
    elev.set("z_max", 0.2)
    assert elev.in_effect("z_max").numpy().tolist() == [np.float32(0.4)] * E
    program.enqueue_segment(1, ctx)
    assert elev.in_effect("z_max").numpy().tolist() == [np.float32(0.4)] * E
    program.enqueue_segment(0, ctx)
    assert elev.in_effect("z_max").numpy().tolist() == [np.float32(0.2)] * E
    program.enqueue_segment(1, ctx)
    # (b) normalize.size drawn behind a HOST gate generator
    spec = presets.gates(num_envs=E, device=DEV, generator=HOST_RING)
    spec = spec.insert_after("order", StageRef("normalize", "extent"))
    spec = spec.with_randomize({"normalize.size": (0.7, 1.6)})
    gates = build(spec, seeds=_seeds())
    assert gates.segments[0].host and len(gates.segments) == 2
    norm = gates.params["normalize"]
    prev = norm.in_effect("size").numpy().copy()                    # the warm-up's draws
    assert norm.in_effect("size") is norm["size"]
    gates.enqueue_segment(0, ctx, advance=True)
    assert norm.in_effect("size").numpy().tobytes() == prev.tobytes()
    gates.enqueue_segment(1, ctx)
    now = norm.in_effect("size").numpy()
    assert (now != prev).any()
    assert ((now >= np.float32(0.7)) & (now <= np.float32(1.6))).all()


# ---------------------------------------------------------------------------------------
# 11. Bounds and value types
# ---------------------------------------------------------------------------------------

def test_size_bounds_and_types():
    for v in (20.0, 0.05, -1.0):
        with pytest.raises(ValueError, match=rf"BezierParams\.size must be in "
                                             rf"\[0\.1, 10\.0\], got {re.escape(repr(v))}"):
            TrackGenerator(_tcfg(size=v), _rng())
    with pytest.raises(ValueError, match=r"PolarParams\.size is dynamic and must be a real "
                                         r"number, got True"):
        TrackGenerator(_tcfg(generator="polar", size=True), _rng())
    with pytest.raises(ValueError, match=r"NormalizeExtentParams\.size must be in"):
        GateGenerator(_gcfg(generator="polar", size=11.0), _rng())
    gen = TrackGenerator(_tcfg(), _rng())
    sp_ = gen.params["generator"]
    with pytest.raises(ValueError, match=r"BezierParams\.size must be in \[0\.1, 10\.0\]"):
        sp_.set("size", 10.5)
    sp_.set("size", float(np.float32(10.0)))
    sp_.set("size", math.nextafter(10.0, 11.0))                 # rounds to f32(10)
    with pytest.raises(ValueError, match=r"must be in \[0\.1, 10\.0\]"):
        sp_.set("size", float(np.nextafter(np.float32(10.0), np.float32(11.0))))
    sp_.set("size", 0.1)
    # a YAML-style string is coerced like the other dynamic floats
    assert TrackGenerator(_tcfg(size="1.5"), _rng()).params["generator"].params.size == 1.5


def test_min_num_points_values_and_types():
    ref = _snap(TrackGenerator(_tcfg(min_num_points=7), _rng()).generate())
    for v in (7.0, np.int64(7), np.int32(7), "7"):
        gen = TrackGenerator(_tcfg(min_num_points=v), _rng())
        assert type(gen.params["generator"].params.min_num_points) is int, v
        assert not _differs(_snap(gen.generate()), ref), v
        gen = TrackGenerator(_tcfg(generator_params={"min_num_points": v}), _rng())
        assert not _differs(_snap(gen.generate()), ref), v
    # a flat value equal after int() does not conflict with generator_params
    TrackGenerator(_tcfg(min_num_points=7.0, generator_params={"min_num_points": 7}), _rng())
    with pytest.raises(ValueError, match=r"min_num_points must be >= 2, got True"):
        TrackGenerator(_tcfg(min_num_points=True), _rng())
    with pytest.raises(ValueError, match=r"max_num_points must be >= min_num_points"):
        TrackGenerator(_tcfg(min_num_points=14), _rng())
    sp_ = TrackGenerator(_tcfg(), _rng()).params["generator"]
    with pytest.raises(ValueError, match=r"BezierParams\.min_num_points is dynamic and must "
                                         r"be an int, got 9\.5"):
        sp_.set("min_num_points", 9.5)
    with pytest.raises(ValueError, match=r"must be an int, got True"):
        sp_.set("min_num_points", True)
    with pytest.raises(ValueError, match=r"BezierParams\.min_num_points must be in "
                                         r"\[2, 13\], at most max_num_points \(the static "
                                         r"corner capacity\), got 40"):
        sp_.set("min_num_points", 40)
    # every refusal names the merged range bounds() reports (not the declared [2, inf])
    assert sp_.bounds("min_num_points") == (2, 13)
    with pytest.raises(ValueError, match=r"BezierParams\.min_num_points must be in "
                                         r"\[2, 13\], at most max_num_points \(the static "
                                         r"corner capacity\), got 1"):
        sp_.set("min_num_points", 1)
    with pytest.raises(ValueError, match=r"must be an int, got 40\.5"):
        sp_.set("min_num_points", 40.5)
    sp_.set("min_num_points", 13)
    with pytest.raises(ValueError, match=r"expected a int32 array"):
        sp_.set("min_num_points", _f32(np.full(E, 9.0)))


def test_gate_width_values():
    with pytest.raises(ValueError, match=r"^gate_width must be >= 0, got -0\.1$"):
        GateGenerator(_gcfg(gate_width=-0.1), _rng())
    gen = GateGenerator(_gcfg(), _rng())
    with pytest.raises(ValueError, match=r"GateFramesParams\.gate_width must be in "
                                         r"\[0\.0, inf\], got -0\.1"):
        gen.params["gate_frames"].set("gate_width", -0.1)
    gen.params["gate_frames"].set("gate_width", 0.3)
    seq = gen.generate()
    assert np.nanmax(seq.half_size.numpy()) == np.float32(0.15)


# ---------------------------------------------------------------------------------------
# 12. Poison: the static views raise on the new dynamic names
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls,field", [(bp.BezierParams, "size"),
                                       (bp.BezierParams, "min_num_points"),
                                       (bp.HullParams, "min_num_points"),
                                       (bp.RepulsiveParams, "size"),
                                       (bp.PointGateParams, "size"),
                                       (sp.NormalizeExtentParams, "size"),
                                       (sp.GateFramesParams, "gate_width")])
def test_the_static_views_poison_the_new_dynamic_names(cls, field):
    view = _static_view(cls())
    v = getattr(view, field)
    assert isinstance(v, _DynamicField)
    for read in (lambda: float(v), lambda: int(v), lambda: v * 2, lambda: v == 1,
                 lambda: bool(v)):
        with pytest.raises(TypeError, match=rf"{cls.__name__}\.{field} is dynamic"):
            read()


def test_the_private_cores_check_the_new_dyn_keys():
    arr = _f32(np.ones(E))
    for core, name, fields in (
            (wg_bezier._enqueue_bezier, "bezier", ("min_num_points", "rad",
                                                   "handle_clamp_frac", "size")),
            (wg_hull._enqueue_hull, "hull", ("min_num_points", "hull_displacement", "size")),
            (wg_polar._enqueue_polar, "polar", ("polar_radial_jitter",
                                                "polar_angular_jitter", "size"))):
        partial = {f: arr for f in fields if f != "size"}
        with pytest.raises(ValueError, match=rf"{name}: dyn must map exactly the dynamic "
                                             rf"fields.*missing \['size'\]"):
            core(None, None, None, None, None, None, partial)


# ---------------------------------------------------------------------------------------
# 13. Capacity: truncation fails closed; the checkpoint ring bound
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("on_invalid", ["keep", "poison"])
def test_a_truncating_size_fails_closed(on_invalid):
    """Repulsive at size 5 with the default ``n_max`` (its target perimeters 5.4-8.1 at
    size 1 vs ``(n_max - 1) * spacing = 22.98``): every env truncated and invalid, its
    rows kept or poisoned as ``on_invalid`` says (PR5)."""
    n = 4
    spec = presets.track(num_envs=n, device=DEV, generator="repulsive", num_points=64,
                         generator_params={"repulsive_stages": (16, 32, 64)}, size=5.0,
                         on_invalid=on_invalid)
    with pytest.warns(UserWarning, match=r"generator\.size=5\.0 truncates at n_max=384"):
        gen = TrackGenerator(spec, _rng(SEEDS[:n]))
    t = gen.generate()
    assert t.truncated.numpy().tolist() == [1] * n
    assert t.valid.numpy().tolist() == [0] * n
    center = t.center.numpy()
    if on_invalid == "poison":
        assert np.isnan(center).all() and t.count.numpy().tolist() == [0] * n
    else:
        assert np.isfinite(center).any() and (t.count.numpy() > 0).all()


def test_a_relaxed_stations_source_fails_closed_on_a_truncated_loop():
    spec = presets.gates(num_envs=E, device=DEV, source="relaxed_stations", gates=12,
                         n_max=128, size=2.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"generator\.size.*truncates")
        gen = GateGenerator(spec, _rng())
    seq = gen.generate()
    truncated = gen.program.tap("cl.truncated").numpy().astype(bool)
    assert truncated.any(), "no truncated loop (vacuous)"
    assert (seq.count.numpy()[truncated] == 0).all()
    assert (seq.valid.numpy()[truncated] == 0).all()


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "default"])
def test_the_checkpoint_bound_holds_under_a_size_curriculum(explicit):
    """A Course whose ``max_checkpoints`` is ``ceil(1.5 * n_max * spacing /
    checkpoint_spacing)`` never clips a valid track's checkpoint ring as its size grows
    from 1 to 2.5 (the readback default of 0.2.0.dev5, sized at construction, would).
    Roadmap PR14: that bound IS the default at flat elevation (``max_checkpoints=None``)."""
    from track_gen.course import Course, CourseConfig

    n, spacing_cp = 16, 0.5
    cfg = _tcfg(num_envs=n)
    bound = math.ceil(1.5 * cfg.N_max * cfg.resolved_spacing / spacing_cp)
    course = Course(CourseConfig(mode="track", gen=cfg, seeds=_seeds(np.arange(n) * 7 + 1),
                                 checkpoint_spacing=spacing_cp,
                                 max_checkpoints=bound if explicit else None))
    assert course.checkpoint_sampler.max_checkpoints == bound
    for s in (1.0, 1.5, 2.0, 2.5):
        course.generator.params["generator"].set("size", s)
        course.regenerate()
        valid = course.result.valid.numpy().astype(bool)
        assert valid.any()
        assert (course.checkpoint_sampler.truncated.numpy()[valid] == 0).all(), s


# ---------------------------------------------------------------------------------------
# 14. Legacy surfaces
# ---------------------------------------------------------------------------------------



def test_configs_and_specs_without_size_load():
    d = dict(generator="hull", num_envs=E, device=DEV, min_num_points=5)
    assert TrackGenConfig(**d).size == 1.0
    data = presets.track(num_envs=E, device=DEV, generator_params={"rad": 0.3}).to_dict()
    assert "size" not in repr(data)
    gen = TrackGenerator(PipelineSpec.from_dict(data), _rng())
    assert gen.params["generator"]["size"].numpy().tolist() == [1.0] * E
    gdata = presets.gates(num_envs=E, device=DEV, generator="checkpoint").to_dict()
    g = GateGenerator(PipelineSpec.from_dict(gdata), _rng())
    assert g.params["normalize"]["size"].numpy().tolist() == [1.0] * E


def test_a_generator_spec_with_bezier_params_is_refused():
    with pytest.raises(TypeError, match=r"BezierParams\.min_num_points .* is dynamic: "
                                        r"dynamic fields need a Stage"):
        registry.add(GeneratorSpec(name="acme.legacy_bezier2",
                                   alloc_scratch=_bezier_reference._alloc_scratch,
                                   generate=_bezier_reference._generate,
                                   params=bp.BezierParams))


def test_a_flat_size_no_stage_applies_raises(plugin_registry):
    from tests._plugin_toys import register_stage_toys, register_toys

    register_stage_toys(plugin_registry)
    register_toys(plugin_registry)          # the legacy toy.circle / toy.ring specs
    with pytest.raises(ValueError, match=r"TrackGenConfig\.size=2\.0: generator 'external' "
                                         r"declares no size \(external ingests your "
                                         r"geometry as given: scale centerline\.xy in your "
                                         r"graph\)"):
        TrackGenerator(_tcfg(generator="external", size=2.0), _rng())
    with pytest.raises(ValueError, match=r"TrackGenConfig\.size=2\.0: generator "
                                         r"'toy\.wobble' declares no size \(a per-env size "
                                         r"is a generator field"):
        legacy.to_pipeline_spec(_tcfg(generator="toy.wobble", size=2.0))
    # a legacy spec cannot declare a dynamic size: the remedy is to port it to a Stage
    with pytest.raises(ValueError, match=r"TrackGenConfig\.size=2\.0: generator "
                                         r"'toy\.circle' declares no size \(a legacy "
                                         r"GeneratorSpec has no per-env "
                                         r"size: port it to a Stage"):
        legacy.to_pipeline_spec(_tcfg(generator="toy.circle", size=2.0))
    TrackGenerator(_tcfg(generator="toy.wobble", size=1.0), _rng())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)    # the legacy gate spec
        with pytest.raises(ValueError, match=r"GateGenConfig\.size=2\.0: gate generator "
                                             r"'toy\.ring' declares no size \(a per-env size "):
            legacy.to_gate_pipeline_spec(_gcfg(generator="toy.ring", size=2.0))
        legacy.to_gate_pipeline_spec(_gcfg(generator="toy.ring"))
    # a gate generator without size whose gates normalize.extent re-normalizes: routed
    gspec = legacy.to_gate_pipeline_spec(_gcfg(generator="voronoi", size=2.0))
    assert dict(next(r for r in gspec.stages if r.slot_id == "normalize").params) == \
        {"size": 2.0}


# ---------------------------------------------------------------------------------------
# 15. The kit: params.per_env runs for repulsive and the point gate generators
# ---------------------------------------------------------------------------------------

_KIT_SKIP = ("hot_path", "statelessness", "batch_independence", "determinism", "output")


@pytest.mark.parametrize("kind,name", [("generator", "repulsive"),
                                       ("gate_generator", "bezier"),
                                       ("gate_generator", "hull"),
                                       ("generator", "hull")])
def test_the_kit_runs_params_per_env_on_the_new_fields(kind, name):
    if kind == "generator":
        config = dict(REP) if name == "repulsive" else None
        report = check_generator(name, devices=[DEV], num_envs=4, config=config,
                                 skip=_KIT_SKIP + ("diversity",))
    else:
        report = check_gate_generator(name, devices=[DEV], num_envs=4,
                                      skip=_KIT_SKIP + ("diversity",))
    per_env = report.result("params.per_env", DEV)
    assert not per_env.skipped and per_env.passed, str(report)
    want = {"repulsive": ["size"], "bezier": ["size"]}.get(name, ["size"])
    if (kind, name) == ("generator", "hull"):
        want = ["hull_displacement", "min_num_points", "size"]
    assert per_env.details["fields"] == want


# ---------------------------------------------------------------------------------------
# 16. presets(size=)
# ---------------------------------------------------------------------------------------

def test_presets_size_routes_to_the_stage_that_holds_it():
    assert presets.track(num_envs=E, device=DEV, size=1.0) == \
        presets.track(num_envs=E, device=DEV)
    assert presets.gates(num_envs=E, device=DEV, generator="polar", size=1.0).to_dict() == \
        presets.gates(num_envs=E, device=DEV, generator="polar").to_dict()
    t = presets.track(num_envs=E, device=DEV, size=1.5)
    assert dict(t.stages[0].params) == {"size": 1.5}
    t = presets.track(num_envs=E, device=DEV, generator="hull", size=1.5,
                      generator_params={"hull_displacement": 0.2})
    assert dict(t.stages[0].params) == {"hull_displacement": 0.2, "size": 1.5}
    t = presets.track(num_envs=E, device=DEV, size=1.5,
                      generator_params=bp.BezierParams(rad=0.3))
    assert t.stages[0].params == bp.BezierParams(rad=0.3, size=1.5)
    g = presets.gates(num_envs=E, device=DEV, size=1.5)
    assert dict(g.stages[0].params) == {"size": 1.5}
    g = presets.gates(num_envs=E, device=DEV, generator="polar", size=1.5)
    assert g.stages[0].params is None
    assert dict(next(r for r in g.stages if r.slot_id == "normalize").params) == \
        {"size": 1.5}
    for source in ("stations", "relaxed_stations"):
        g = presets.gates(num_envs=E, device=DEV, source=source, size=1.5)
        assert g.stages[0].kind == "generator" and dict(g.stages[0].params) == {"size": 1.5}
    # they build, and equal the flat route
    a = _snap(TrackGenerator(presets.track(num_envs=E, device=DEV, size=1.5), _rng())
              .generate())
    b = _snap(TrackGenerator(_tcfg(size=1.5), _rng()).generate())
    assert not _differs(a, b)
    a = _snap(GateGenerator(presets.gates(num_envs=E, device=DEV, generator="polar",
                                          size=1.5), _rng()).generate())
    b = _snap(GateGenerator(_gcfg(generator="polar", size=1.5), _rng()).generate())
    assert not _differs(a, b)


def test_presets_size_errors():
    with pytest.raises(ValueError, match=r"size=1\.5 and the generator's params set "
                                         r"size=2\.0: give size once"):
        presets.track(num_envs=E, device=DEV, size=1.5, generator_params={"size": 2.0})
    with pytest.raises(ValueError, match=r"give size once"):
        presets.track(num_envs=E, device=DEV, size=1.5,
                      generator_params=bp.BezierParams(size=2.0))
    with pytest.raises(ValueError, match=r"give size once"):
        presets.gates(num_envs=E, device=DEV, size=1.5, generator_params={"size": 1.0})
    with pytest.raises(ValueError, match=r"size=1\.5: generator 'external' declares no "
                                         r"size \(external ingests your geometry as given: "
                                         r"scale centerline\.xy in your graph\)"):
        presets.track(num_envs=E, device=DEV, generator="external", size=1.5)


# ---------------------------------------------------------------------------------------
# 17. Hints
# ---------------------------------------------------------------------------------------

def test_the_hints_of_the_size_and_count_keys():
    ext = TrackGenerator(presets.track(num_envs=E, device=DEV, generator="external"),
                         _rng())
    with pytest.raises(KeyError, match=r"no dynamic parameter 'size'.*\(external ingests "
                                       r"your geometry as given: scale centerline\.xy in "
                                       r"your graph\)"):
        ext.params["generator"].set("size", 2.0)
    polar = GateGenerator(_gcfg(generator="polar"), _rng())
    with pytest.raises(KeyError, match=r"no dynamic parameter 'size'.*\(this gate generator "
                                       r"is re-normalized by normalize\.extent: use "
                                       r"normalize\.size, whose initial value is "
                                       r"GateGenConfig\.size"):
        polar.params["generator"]["size"]
    bez = GateGenerator(_gcfg(), _rng())
    with pytest.raises(KeyError, match=r"'min_num_points' is a static parameter.*\(on gate "
                                       r"generators it is the gate count capacity "
                                       r"\(min_gates\): not per env in 0\.x\)"):
        bez.params["generator"].set("min_num_points", 5)
    # no hint where none applies
    with pytest.raises(KeyError) as info:
        TrackGenerator(_tcfg(), _rng()).params["generator"]["nope"]
    assert "(" not in str(info.value).split("dynamic fields:")[-1].split("]")[-1]


# ---------------------------------------------------------------------------------------
# 18. Posed gates: check_crossing is inert
# ---------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------
# 19. Review fixes: size routing for plugins, NaN sizes, the knob-aware truncation
#     warning, the inert validity width, hints, Course's gate-width floor
# ---------------------------------------------------------------------------------------

def _plugin_size_toys():
    """A convention track generator (non-flat Params declaring ``size``, applied through
    ``ops.normalize_extent(size=)``), a convention anchors gate generator (non-flat
    Params declaring ``size``, the bezier gates' kernels), and a bezier gate generator
    whose gates ``normalize.extent`` re-normalizes (flat ``PointGateParams``)."""
    import dataclasses

    from tests._plugin_toys import ToyWobble, ToyWobbleParams, _wobble_k
    from track_gen import plugin
    from track_gen._src.warp_generate_gates import GateBezier

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class SizedWobbleParams(ToyWobbleParams):
        size: float = plugin.dynamic(1.0, lo=SIZE_BOUNDS[0], hi=SIZE_BOUNDS[1])

    class SizedWobble(ToyWobble):
        name = "acme.sized_wobble"
        Params = SizedWobbleParams

        def build(self, b, p):
            xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
            amp, salt, size = b.param("amp"), b.salt("shape"), b.param("size")
            n, lobes = self.RAW, int(p.lobes)
            radius, extent = 0.6 * b.scale, 1.2 * float(b.scale)

            def enqueue(ctx):
                ctx.launch(_wobble_k, b.E * n, [b.seed, salt, amp, lobes, n, radius, xy])
                ops.normalize_extent(ctx, xy, n, extent, size=size)
                ctx.fill(ok, 1)
            return enqueue

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class SizedGateParams(plugin.Params):
        min_num_points: int = plugin.static(9, lo=2)
        max_num_points: int = plugin.static(13)
        min_point_distance: float = plugin.static(0.05)
        size: float = plugin.dynamic(1.0, lo=SIZE_BOUNDS[0], hi=SIZE_BOUNDS[1])

    class SizedGates(GateBezier):
        name = "acme.sized_gates"
        Params = SizedGateParams

    class NormBez(GateBezier):
        name = "acme.normbez"
        preferred = {"order": "ccw", "normalize": "extent"}

    return SizedWobble, SizedGates, NormBez


def test_a_convention_plugin_receives_the_flat_size(plugin_registry):
    """A plugin Stage whose Params declare ``size`` (and whose ``from_legacy`` is the
    default ``cls()``) gets ``TrackGenConfig.size`` / ``GateGenConfig.size``, as the
    built-ins' ``from_legacy`` maps it; it equals ``presets(size=)``."""
    wobble, sized_gates, _ = _plugin_size_toys()
    plugin_registry.add(wobble)
    plugin_registry.add(sized_gates)
    gen = TrackGenerator(_tcfg(generator="acme.sized_wobble", size=2.0), _rng())
    assert gen.params["generator"].params.size == 2.0
    flat = _snap(gen.generate())
    preset = _snap(TrackGenerator(presets.track(num_envs=E, device=DEV,
                                                generator="acme.sized_wobble", size=2.0),
                                  _rng()).generate())
    one = _snap(TrackGenerator(_tcfg(generator="acme.sized_wobble"), _rng()).generate())
    assert not _differs(flat, preset)
    assert _differs(flat, one), "size 2 changed nothing (vacuous)"
    # an explicit generator_params size wins (the flat one must then agree or be 1)
    gp = TrackGenerator(_tcfg(generator="acme.sized_wobble",
                              generator_params={"size": 2.0}), _rng())
    assert not _differs(_snap(gp.generate()), flat)
    # an anchors gate generator with non-flat Params: the same bytes as the built-in
    # bezier gates (same kernels and salts) at that size
    gates = GateGenerator(_gcfg(generator="acme.sized_gates", size=2.0), _rng())
    assert gates.params["generator"].params.size == 2.0
    ref = _snap(GateGenerator(_gcfg(size=2.0), _rng()).generate())
    assert not _differs(_snap(gates.generate()), ref)


def test_a_static_size_field_is_never_written(plugin_registry):
    """A plugin ``size`` that does not follow the convention (here a static ``int``, an
    arena cell count) is the plugin's own: neither the flat ``size`` nor
    ``presets.track(size=)`` writes it; both raise, and a size of 1 builds."""
    import dataclasses

    from tests._plugin_toys import ToyWobble, ToyWobbleParams
    from track_gen import plugin

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ArenaParams(ToyWobbleParams):
        size: int = plugin.static(64)

    class Arena(ToyWobble):
        name = "acme.arena"
        Params = ArenaParams

    plugin_registry.add(Arena)
    for size in (2, 2.0):
        with pytest.raises(ValueError, match=r"TrackGenConfig\.size=2(\.0)?: generator "
                                             r"'acme\.arena''s Params do not map the flat "
                                             r"size \(its size is 64\); its size field is "
                                             r"not the per-env size"):
            legacy.to_pipeline_spec(_tcfg(generator="acme.arena", size=size))
        with pytest.raises(ValueError, match=r"generator 'acme\.arena' declares no size "
                                             r"\(a per-env size is a generator field"):
            presets.track(num_envs=E, device=DEV, generator="acme.arena", size=size)
    gen = TrackGenerator(_tcfg(generator="acme.arena"), _rng())
    assert gen.params["generator"].params.size == 64

    # A static float size that happens to equal the flat size still does not apply it.
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class FloatArenaParams(ToyWobbleParams):
        size: float = plugin.static(2.0)

    class FloatArena(ToyWobble):
        name = "acme.float_arena"
        Params = FloatArenaParams

    plugin_registry.add(FloatArena)
    with pytest.raises(ValueError, match=r"TrackGenConfig\.size=2\.0: generator "
                                         r"'acme\.float_arena''s Params do not map the flat "
                                         r"size \(its size is 2\.0\); its size field is "
                                         r"not the per-env size"):
        legacy.to_pipeline_spec(_tcfg(generator="acme.float_arena", size=2.0))


def test_a_renormalized_gate_generator_with_a_static_size_of_its_own(plugin_registry):
    """A re-normalized gate generator whose ``size`` is static (not the convention) keeps
    it: ``presets.gates(size=)`` routes the per-env size to ``normalize.size`` whatever
    the generator's params hold, and binding never resets the value its ``from_legacy``
    chose."""
    import dataclasses

    from track_gen import plugin
    from track_gen._src.generator_binding import bind_generator
    from track_gen._src.warp_generate_gates import GateBezier

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class OwnSizeParams(plugin.Params):
        min_num_points: int = plugin.static(9, lo=2)
        max_num_points: int = plugin.static(13)
        min_point_distance: float = plugin.static(0.05)
        size: int = plugin.static(64)

        @classmethod
        def from_legacy(cls, cfg):
            return cls(size=32)

    class OwnSize(GateBezier):
        name = "acme.own_size"
        Params = OwnSizeParams
        preferred = {"order": "ccw", "normalize": "extent"}

    plugin_registry.add(OwnSize)
    for gp in (None, {"size": 64}, OwnSizeParams()):
        spec = presets.gates(num_envs=E, device=DEV, generator="acme.own_size",
                             generator_params=gp, size=2.0)
        norm = next(r for r in spec.stages if r.slot_id == "normalize")
        assert dict(norm.params) == {"size": 2.0}, gp
    binding = bind_generator("gate_generator", _gcfg(generator="acme.own_size"),
                             registry=plugin_registry)
    assert binding.params.size == 32
    assert binding.run_config.generator_params.size == 32


def test_the_normalize_size_hint_qualifies_the_gate_route(plugin_registry):
    """A ``normalize.size`` randomizer on a gate pipeline without a normalize stage points
    to ``generator.size`` only where the generator declares one."""
    import dataclasses

    from track_gen import plugin
    from track_gen._src.warp_generate_gates import GateBezier

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class NoSizeParams(plugin.Params):
        min_num_points: int = plugin.static(9, lo=2)
        max_num_points: int = plugin.static(13)
        min_point_distance: float = plugin.static(0.05)

    class NoSize(GateBezier):
        name = "acme.no_size"
        Params = NoSizeParams

    plugin_registry.add(NoSize)
    with pytest.raises(ValueError, match=r"holds its size: use generator\.size, where it "
                                         r"declares one"):
        presets.gates(num_envs=E, device=DEV, generator="acme.no_size",
                      randomize={"normalize.size": (1.0, 2.0)})


def test_a_renormalized_gate_generator_that_declares_size(plugin_registry):
    """Its gates are rescaled by normalize.extent, so the size goes to normalize.size
    (presets and GateGenConfig), and a generator.size randomizer or construction value
    on it warns that it has no effect."""
    _, _, normbez = _plugin_size_toys()
    plugin_registry.add(normbez)
    spec = presets.gates(num_envs=E, device=DEV, generator="acme.normbez", size=2.0)
    norm = next(r for r in spec.stages if r.slot_id == "normalize")
    assert dict(norm.params) == {"size": 2.0}
    assert "size" not in dict(next(r for r in spec.stages if r.slot_id == "generator")
                              .params or {})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        two = _snap(GateGenerator(spec, _rng()).generate())
        cfg = GateGenerator(_gcfg(generator="acme.normbez", size=2.0), _rng())
        assert cfg.params["generator"].params.size == 1.0     # the erased size reset
        assert not _differs(_snap(cfg.generate()), two)
    one = _snap(GateGenerator(presets.gates(num_envs=E, device=DEV,
                                            generator="acme.normbez"), _rng()).generate())
    assert _differs(two, one), "size 2 changed nothing (vacuous)"
    for kw in ({"randomize": {"generator.size": (1.0, 2.0)}},
               {"generator_params": {"size": 2.0}}):
        with pytest.warns(UserWarning, match=r"has no effect on gate generator "
                                             r"'acme\.normbez': this gate generator is "
                                             r"re-normalized by normalize\.extent: use "
                                             r"normalize\.size"):
            build(presets.gates(num_envs=E, device=DEV, generator="acme.normbez", **kw))
    # a size in the generator params and size= on the preset: given twice
    with pytest.raises(ValueError, match=r"give size once"):
        presets.gates(num_envs=E, device=DEV, generator="acme.normbez", size=2.0,
                      generator_params={"size": 3.0})


def test_a_nan_size_builds():
    """A NaN size is a legal dynamic value (PR13c policy: those envs fail validity): no
    route crashes or blames the flat field."""
    nan = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for make in (
                lambda: TrackGenerator(presets.track(num_envs=E, device=DEV, size=nan),
                                       _rng()),
                lambda: TrackGenerator(_tcfg(size=nan), _rng()),
                lambda: TrackGenerator(_tcfg(generator_params={"size": nan}), _rng()),
                lambda: TrackGenerator(_tcfg(generator_params=bp.BezierParams(size=nan)),
                                       _rng()),
                lambda: TrackGenerator(presets.track(
                    num_envs=E, device=DEV, generator="repulsive", num_points=64,
                    generator_params={"size": nan, "repulsive_stages": (16, 32, 64)}),
                    _rng())):
            assert make().generate().valid.numpy().tolist() == [0] * E
        build(presets.track(num_envs=E, device=DEV,
                            randomize={"generator.size": (0.5, nan)}))


def _trunc_warnings(spec) -> list:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        build(spec)
    return [str(w.message) for w in rec if "truncates at n_max" in str(w.message)]


def test_the_truncation_warning_tracks_the_loop_length_knobs():
    """The nominal loop is the measured LONGEST default loop (it fires from the
    documented "truncates from" size), scaled by the corner capacity, hull's displacement
    (its randomizer's hi); a ring generator with a
    non-default shape knob is not checked."""
    def track(**kw):
        return presets.track(num_envs=E, device=DEV, **kw)

    assert not _trunc_warnings(track(size=2.6))                # 8.68 * 2.6 < 22.98
    assert _trunc_warnings(track(size=2.7))
    # the corner-count recipe: a larger max_num_points (x (20 / 13) ** 0.6)
    assert not _trunc_warnings(track(size=2.5))
    w = _trunc_warnings(track(size=2.5, generator_params={"max_num_points": 20}))
    assert len(w) == 1 and f"~{8.68 * 2.5 * (20 / 13) ** 0.6:.1f} >" in w[0], w
    # hull_displacement: its value, or its randomizer's hi
    assert not _trunc_warnings(track(generator="hull", size=2.5))
    nominal = 8.30 * 2.5 * (1.0 + 0.6 * 0.85)
    for kw in ({"generator_params": {"hull_displacement": 1.0}},
               {"randomize": {"generator.hull_displacement": (0.1, 1.0)}}):
        w = _trunc_warnings(track(generator="hull", size=2.5, **kw))
        assert len(w) == 1 and f"~{nominal:.1f} >" in w[0], (kw, w)
    # the longest loop of 8192 envs: sizes that truncate a 256-env default-seed batch
    # (hull at 3.1, voronoi at 4.8) warn
    assert _trunc_warnings(track(generator="hull", size=3.1))
    assert _trunc_warnings(track(generator="voronoi", size=4.8))
    # a ring generator whose shape knobs differ from the measured defaults: not checked
    assert _trunc_warnings(track(generator="polar", size=4.0))
    assert not _trunc_warnings(track(generator="polar", size=4.0,
                                     generator_params={"polar_num_knots": 24}))
    assert not _trunc_warnings(track(generator="polar", size=4.0, randomize={
        "generator.polar_radial_jitter": (0.0, 0.85)}))


def test_the_hints_of_the_review():
    with pytest.raises(PlanError, match=r"has no param 'gate_width'"):
        build(presets.gates(num_envs=E, device=DEV,
                            randomize={"validity.gate_width": (0.0, 0.3)}))
    with pytest.raises(ValueError, match=r"names stage 'normalize', which this pipeline "
                                         r"does not have.*; this gate generator is not "
                                         r"re-normalized and holds its size: use "
                                         r"generator\.size"):
        presets.gates(num_envs=E, device=DEV, randomize={"normalize.size": (0.8, 1.2)})
    with pytest.raises(PlanError, match=r"has no param 'scale'.*\(the core scale is static "
                                        r"\(TrackGenConfig\.scale / BatchSpec\.scale\); the "
                                        r"per-env multiplier is size\)"):
        build(presets.track(num_envs=E, device=DEV,
                            randomize={"generator.scale": (0.8, 1.2)}))
    with pytest.raises(ValueError, match=r"unknown generator_params for generator 'bezier': "
                                         r"\['scale'\].*; scale: the core scale is static"):
        TrackGenerator(_tcfg(generator_params={"scale": 2.0}), _rng())
    with pytest.raises(ValueError, match=r"unknown generator_params for gate generator "
                                         r"'polar': \['size'\].*; size: this gate generator "
                                         r"is re-normalized by normalize\.extent: use "
                                         r"normalize\.size, whose initial value is "
                                         r"GateGenConfig\.size"):
        GateGenerator(_gcfg(generator="polar", generator_params={"size": 2.0}), _rng())




def test_course_gates_mode_refuses_a_zero_gate_width():
    from track_gen.course import Course, CourseConfig

    course = Course(CourseConfig(mode="gates", gen=_gcfg(gate_width=0.3), seeds=1))
    frames = course.generator.params["gate_frames"]
    lo, hi = frames.bounds("gate_width")
    assert 0.0 < lo == 2.0 ** -149 and hi is None
    for v in (0.0, -0.1):
        with pytest.raises(ValueError, match=r"GateFramesParams\.gate_width must be in "
                                             r"\[.*, inf\], > 0 in Course gates mode \(a "
                                             r"width-0 gate has a degenerate crossing "
                                             r"segment and can never be passed\), got"):
            frames.set("gate_width", v)
    frames.set("gate_width", 0.2)
    # a GateGenerator outside a Course keeps the declared bound
    assert GateGenerator(_gcfg(), _rng()).params["gate_frames"].bounds("gate_width") == \
        (0.0, None)
