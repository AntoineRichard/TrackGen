"""CUDA-only: per-env size and the remaining difficulty axes (roadmap PR13d, design 9.3).

PR13d adds four per-env difficulty axes, each a DYNAMIC Params field read by an
array-reading TWIN of an existing kernel (the value loaded once per thread and clamped
to its bounds by comparisons, so a NaN passes; the original kernel's body verbatim):

- ``size`` (float32, default 1, bounds ``ops.constants.SIZE_BOUNDS``): a per-env factor
  applied where each generator applies ``scale`` (bezier / hull and the bezier / hull
  gate generators: last in the corner grid, ``_sample_grid_points_sized_k`` /
  ``_scaled_sized_k``; polar / voronoi / checkpoint: the normalization target,
  ``_normalize_extent_sized_k``; the polar / voronoi / checkpoint gate pipelines:
  ``normalize.size``, ``_normalize_extent_to_sized_k``; repulsive: a finite-gated
  multiply of its output copy, ``_copy_sized_k``);
- ``min_num_points`` (int32, bezier / hull track generators): the corner count's lower
  bound, clamped to ``[2, max_num_points]`` (``_sample_counts_dyn_k``);
- ``gate_frames.gate_width`` (float32, ``>= 0``): ``_finalize_frame_dyn_k``, and
  ``validity.gate_default`` checks bar crossings PER ENV (``_finalize_validity_bars_k``:
  the crossing loop runs for an env with a real gate of non-zero ``half_size``);
- ``StageParams.in_effect(field, out=, stream=)``: the ``[E]`` values the stage READ for
  the current rows (a latched field's work copy, a drawn field's array), and
  ``StageParams.bounds(field)``: the clamp a built-in applies where it reads them.

The CPU proofs are tests/test_difficulty_axes.py; this module proves the claims on
``cuda:0``, where FMA contraction, ``fminf`` / ``fmaxf`` and NaN canonicalization could
break them:

1. Each twin == its original kernel, launched on the inputs a real generation handed it
   (captured right before the launch): a size-1 (or default) array is bitwise the
   original; in-range values == the original at ``scale * s`` / ``target * s`` / the
   scalar knob; out-of-range values == the original at the bound; NaN passes the clamp;
   per env == row by row the original at each env's clamped value; the knob array is
   never written. The validity twin == the toggle-ON PR11 kernel on real gate outputs of
   all five gate generators with per-env bar widths (and == the toggle-OFF kernel on
   zero-width envs), and on posed-like per-gate half sizes ``{0, -0, +, -, NaN, tiny}``.
   ``_copy_sized_k`` meets the design's DEVICE contract (section 1.3): finite vectors
   bitwise at size 1 and ``f32(v * s)`` otherwise, non-finite vectors kept unscaled
   (``±inf`` bitwise, a NaN stays NaN); and, a HARD assertion (design 1.3: the device
   test is the final check of the finite-gated form), every NaN payload (signalling and
   non-default quiet NaNs) is kept bitwise at size 1 and on every non-finite vector. If
   that assertion fails on a device, the design's fallback contract applies: relax it to
   ``isnan`` in the same change and record the device in repulsive.rst and the
   CHANGELOG.
2. Per env == per-env scalar runs: ``set(field, [E] array)`` then ``enqueue()`` equals,
   env by env and on every result field, separate runs BUILT with each scalar value
   (``size`` on the five track generators, repulsive, the
   bezier / hull / stations gate generators; ``normalize.size`` on the polar /
   checkpoint / voronoi gates; ``min_num_points`` on bezier / hull, clamped values
   included); size-1 envs of a mixed array equal the default run; the traces show the
   twins (not the originals); ``in_effect`` returns the array as written and ``bounds``
   the declared / per-instance range. Out-of-range size arrays are clamped at use.
   Repulsive's raw loop at size ``s`` is ``f32(raw(1) * s)`` (the finite-gated copy end
   to end).
3. The per-env crossing check: a point-gate pipeline (``gate_width`` 0, validity built
   without a width) with ``gate_frames.gate_width`` set per env equals, env by env and
   on every GateSequence field (``valid`` / ``fail_mask`` included), pipelines BUILT
   with each width (the PR13c "check on" configuration); bars cross and fail.
4. ``presets.track(randomize={"generator.size": ...})`` (each generator, repulsive
   included) and ``presets.gates(randomize=...)`` on ``generator.size``,
   ``normalize.size`` and ``gate_frames.gate_width``, recorded with
   ``generate(stream=s, advance=True)`` and ``in_effect(out=...)`` into ONE caller
   graph: every replay redraws inside ``[lo, hi]``; the recorded ``in_effect`` copy is
   the drawn array; the batch equals the randomized eager twin and a NON-randomized eager
   twin whose fields were ``set`` to the recorded ``in_effect`` arrays (crossing bits
   included), and differs from the default.
5. ``in_effect`` under masked regeneration in a caller graph: a drawn field (redrawn for
   every env, unmasked envs redraw the same bits) and a latched field written live
   between replays (masked envs take the new value, unmasked envs keep the value their
   rows were built with); unmasked rows keep every byte; the current rows equal a twin
   advanced alike.
6. A curriculum in the caller's graph (Warp capture and a registered
   ``torch.cuda.graph``): the caller's kernels write ``size``, the int32
   ``min_num_points`` and ``gate_frames.gate_width`` (values out of bounds included)
   before ``generate``; ``in_effect(out=)`` recorded after it reads the values as
   written; ``clamp(in_effect, *bounds(field))`` are the values used: every replay
   equals the eager twin set to them (written gate bars crossing-checked per env).
7. The conformance kit on ``cuda:0`` runs ``params.per_env`` (not skipped) and passes
   for repulsive and the bezier / hull gate generators.

Method (as in tests/test_dynamic_knobs_cuda.py; the helpers are
:mod:`track_gen.testing.capture`'s): record on a NON-current side stream with the Warp
mempool disabled (an allocation inside the capture fails loudly) and
``wp.capture_launch`` replaced by a tripwire; poison the outputs (and drawn / copied
param arrays) with FINITE, dtype-aware sentinels before each replay (outputs are
NaN-padded and ``assert_array_equal`` treats NaN == NaN); compare with eager twins
(``enqueue()``, never a graph). Construction prepares the default body; the masked body
is warmed up with an all-zero mask (an exact no-op) and the test's own kernels on
scratch buffers, so no module loads while capturing. Every array is allocated on
``cuda:0`` explicitly. PR13d names (``SIZE_BOUNDS``, the twins) are resolved inside the
tests, so the CPU lane collects this module.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import math
import warnings
from collections.abc import Mapping

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from track_gen import GateGenerator, PerEnvSeededRNG, TrackGenConfig, TrackGenerator  # noqa: E402
from track_gen._src import warp_pipeline  # noqa: E402
from tests._reference_generators import spec as _reference_generator_spec  # noqa: E402
from track_gen._src.generator_binding import _StageGenerator  # noqa: E402
from track_gen.exec import LaunchContext  # noqa: E402
from track_gen.pipeline import GATE_FAIL_BITS, presets  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import check_gate_generator, check_generator  # noqa: E402
from track_gen.testing.trace import trace_launches  # noqa: E402

DEV = "cuda:0"
E = 8
#: Per-env base seeds (the fingerprint seeds: int32 extremes included).
SEEDS = np.array([0, 1, 7, 99991, 2147483647, 1000003, 123456789, 46341], np.int32)
_ZERO = np.zeros(E, np.int32)
_ONES = np.ones(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
_MODES = ("warp", "torch")           # caller capture: Warp ScopedCapture / torch.cuda.graph
#: validity.gate_default's crossing bit in ``GateSequence.fail_mask``.
_CROSSING = 1 << GATE_FAIL_BITS["crossing"]
#: Repulsive's small config (tests/test_bounded_host_cuda.py): presets kwargs, and the
#: flat TrackGenConfig fields of the conformance kit.
_REPULSIVE_SMALL = dict(num_points=64, generator_params={"repulsive_stages": (16, 32, 64)})
_REPULSIVE_KIT = dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384,
                      half_width=0.1)


# ---------------------------------------------------------------------------------------
# Test kernels (one Warp module: _load_test_kernels() loads it outside any capture)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _pick_f32_k(counter: wp.array(dtype=wp.int32), steps: int,
                table: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    # The caller's in-graph curriculum: row k = counter[0] % steps of a [steps, E] table.
    e = wp.tid()
    n = out.shape[0]
    k = counter[0] % steps
    out[e] = table[k * n + e]


@wp.kernel
def _pick_i32_k(counter: wp.array(dtype=wp.int32), steps: int,
                table: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    n = out.shape[0]
    k = counter[0] % steps
    out[e] = table[k * n + e]


def _load_test_kernels() -> None:
    """Load this module's kernels outside any capture (scratch buffers only)."""
    c = wp.zeros(1, dtype=wp.int32, device=DEV)
    tf = wp.zeros(2, dtype=wp.float32, device=DEV)
    ti = wp.zeros(2, dtype=wp.int32, device=DEV)
    of = wp.zeros(2, dtype=wp.float32, device=DEV)
    oi = wp.zeros(2, dtype=wp.int32, device=DEV)
    wp.launch(_bump_k, 1, inputs=[c], device=DEV)
    wp.launch(_pick_f32_k, 2, inputs=[c, 1, tf], outputs=[of], device=DEV)
    wp.launch(_pick_i32_k, 2, inputs=[c, 1, ti], outputs=[oi], device=DEV)
    wp.synchronize()


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _size_bounds() -> "tuple[float, float]":
    """``ops.constants.SIZE_BOUNDS`` (resolved at run time: a PR13d name)."""
    from track_gen.ops.constants import SIZE_BOUNDS  # noqa: PLC0415

    return float(SIZE_BOUNDS[0]), float(SIZE_BOUNDS[1])


def _kernel(path: str):
    """The kernel ``"module:name"`` (asserting it exists)."""
    mod, name = path.split(":")
    k = getattr(importlib.import_module(mod), name, None)
    assert k is not None, f"{mod}.{name} is missing (roadmap PR13d)"
    return k


@contextlib.contextmanager
def _quiet():
    """Silence the build-time size truncation heuristic, which is not under test here."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*truncat", category=UserWarning)
        yield


def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True)
            for k, a in tg_capture.result_arrays(obj).items()}


def _rows(host: dict, envs) -> dict:
    return {k: v.reshape(E, -1)[envs] for k, v in host.items()}


def _stitch(per_env: list) -> dict:
    """The batch whose env ``e`` is env ``e`` of ``per_env[e]`` (host snapshots)."""
    return {name: np.stack([s[name].reshape(E, -1)[e] for e, s in enumerate(per_env)])
            .reshape(per_env[0][name].shape) for name in per_env[0]}


def _assert_bits(got: np.ndarray, want: np.ndarray, label: str, *,
                 nan_ok: bool = False) -> None:
    """Same dtype, shape and BYTES (signed zeros and NaN payloads count). ``nan_ok``:
    where ``want`` is NaN, ``got`` must be NaN with any payload (the design's device
    contract for host-computed expectations); every other element stays bitwise."""
    assert got.dtype == want.dtype and got.shape == want.shape, \
        f"{label}: {got.dtype}{got.shape} vs {want.dtype}{want.shape}"
    if nan_ok and want.dtype.kind == "f":
        wn, gn = np.isnan(want), np.isnan(got)
        if not np.array_equal(wn, gn):
            bad = np.argwhere(wn != gn)[:4].tolist()
            raise AssertionError(f"{label}: NaN positions differ (first {bad}); got "
                                 f"{got[wn != gn][:4]}, want {want[wn != gn][:4]}")
        got = np.where(wn, np.float32(0.0), got).astype(want.dtype)
        want = np.where(wn, np.float32(0.0), want).astype(want.dtype)
    if got.tobytes() != want.tobytes():
        np.testing.assert_array_equal(got, want, err_msg=label)
        raise AssertionError(f"{label}: differs bitwise (signed zeros / NaN payloads)")


def _assert_bitwise(got: dict, ref: dict, label: str, fields=None) -> None:
    """``got[f]`` and ``ref[f]`` have the same dtype, shape and BYTES for every field
    (``fields``, default every field of ``ref``)."""
    names = sorted(ref) if fields is None else list(fields)
    assert names, f"{label}: nothing to compare"
    missing = [n for n in names if n not in got]
    assert not missing, f"{label}: missing {missing}"
    for n in names:
        _assert_bits(got[n], ref[n], f"{label}: {n}")


def _assert_host(got: dict, ref: dict, label: str) -> None:
    assert ref and got.keys() == ref.keys(), f"{label}: {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{label}: {name}")


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in b)


def _side_stream() -> wp.Stream:
    return tg_capture.side_stream(DEV)


def _recording(stream):
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch
    forbidden."""
    return tg_capture.recording(DEV, stream)


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _dev(values, dtype=wp.float32) -> wp.array:
    """A device copy of ``values``, complete before any other stream reads it."""
    np_dtype = np.float32 if dtype is wp.float32 else np.int32
    arr = wp.array(np.asarray(values, np_dtype), dtype=dtype, device=DEV)
    wp.synchronize()
    return arr


def _write(dst: wp.array, values) -> None:
    """Write host values into a device buffer, outside any capture (read at replay)."""
    dst.assign(np.asarray(values, dst.numpy().dtype))
    wp.synchronize()


def _per_env(values, dtype=np.float32) -> np.ndarray:
    """``values`` cycled over the ``E`` envs."""
    return np.resize(np.asarray(values, dtype), E).astype(dtype)


def _key(v, dtype=np.float32) -> bytes:
    """A dict key that tells ``-0.0`` from ``0.0``."""
    return np.asarray(v, dtype).tobytes()


def _clamp(v, lo, hi) -> np.float32:
    """The twins' comparison clamp on the host: NaN and in-range values (signed zeros
    included) pass bit for bit; ``None``: an open side."""
    x = np.float32(v)
    if lo is not None and x < np.float32(lo):
        x = np.float32(lo)
    if hi is not None and x > np.float32(hi):
        x = np.float32(hi)
    return x


def _clamp_int(v, lo, hi) -> int:
    x = int(v)
    if lo is not None and x < int(lo):
        x = int(lo)
    if hi is not None and x > int(hi):
        x = int(hi)
    return x


def _used(values: np.ndarray, lo, hi) -> np.ndarray:
    """``clamp(values, lo, hi)`` elementwise, in the dtype of ``values``."""
    if values.dtype == np.int32:
        return np.array([_clamp_int(v, lo, hi) for v in values], np.int32)
    return np.array([_clamp(v, lo, hi) for v in values], np.float32)


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    return torch


def _caller_graph(mode: str, body):
    """Record ``body(stream)`` into ONE caller graph and return its ``replay()``.

    ``"warp"``: a Warp ``ScopedCapture`` on a non-current side stream. ``"torch"``: a
    ``torch.cuda.graph`` capture registered with Warp through
    ``track_gen.interop.torch.registered`` (``body`` gets the registered stream).
    Both with the mempool disabled and ``wp.capture_launch`` forbidden."""
    if mode == "warp":
        s = _side_stream()
        wp.synchronize()
        with _recording(s) as cap:
            body(s)

        def replay(_g=cap.graph, _s=s):
            _replay(_g, _s)
        return replay
    assert mode == "torch", mode
    torch = _torch_cuda()
    from track_gen.interop.torch import registered  # noqa: PLC0415

    ordinal = wp.get_device(DEV).ordinal
    with torch.cuda.device(ordinal):
        g = torch.cuda.CUDAGraph()
        wp.synchronize()
        torch.cuda.synchronize(ordinal)
        with tg_capture.mempool_disabled(DEV):
            with torch.cuda.graph(g):
                with registered(graph=g) as ws, tg_capture.capture_launch_forbidden():
                    body(ws)
        torch.cuda.synchronize(ordinal)

    def replay(_g=g):
        with torch.cuda.device(ordinal):
            _g.replay()
            torch.cuda.synchronize(ordinal)
        wp.synchronize()
    return replay


def _rng(seeds=SEEDS) -> PerEnvSeededRNG:
    return PerEnvSeededRNG(seeds=wp.array(np.asarray(seeds, np.int32), dtype=wp.int32,
                                          device=DEV), num_envs=len(seeds), device=DEV)


def _with_stage_params(spec, sid: str, params: dict):
    """``spec`` with ``params`` merged into stage ``sid``'s params."""
    ref = spec.stages[spec.ids.index(sid)]
    cur = ref.params
    if cur is None:
        merged = dict(params)
    elif isinstance(cur, Mapping):
        merged = {**dict(cur), **params}
    else:
        merged = dataclasses.replace(cur, **params)
    return spec.replace(sid, dataclasses.replace(ref, params=merged))


def _merge_gp(kw: dict, gp: "dict | None") -> dict:
    kw = dict(kw)
    if gp:
        kw["generator_params"] = {**dict(kw.get("generator_params") or {}), **gp}
    return kw


def _track_spec(kw: dict, randomize=None, *, gp=None, stage_params=None):
    spec = presets.track(num_envs=E, device=DEV, randomize=randomize, **_merge_gp(kw, gp))
    for sid, params in (stage_params or {}).items():
        spec = _with_stage_params(spec, sid, params)
    return spec


def _gate_spec(kw: dict, randomize=None, *, gp=None, stage_params=None):
    spec = presets.gates(num_envs=E, device=DEV, randomize=randomize, **_merge_gp(kw, gp))
    for sid, params in (stage_params or {}).items():
        spec = _with_stage_params(spec, sid, params)
    return spec


#: kind -> (facade, spec builder)
_FACTORY = {"track": (TrackGenerator, _track_spec), "gates": (GateGenerator, _gate_spec)}


def _build(cls, spec, rng=None):
    with _quiet():
        return cls(spec, _rng() if rng is None else rng)


def _eager(cls, spec) -> dict:
    """A fresh generator for ``spec`` at :data:`SEEDS`, one eager ``enqueue()``."""
    gen = _build(cls, spec)
    out = gen.enqueue()
    wp.synchronize()
    assert gen.graph is None, "enqueue() must never self-graph"
    return _snapshot(out)


def _ref_spec(kind: str, kw: dict, sid: str, field: str, value):
    """The spec BUILT with the scalar ``value`` of ``sid.field``."""
    _, spec_of = _FACTORY[kind]
    if sid == "generator":
        return spec_of(kw, gp={field: value})
    if (sid, field) == ("gate_frames", "gate_width"):
        return spec_of(dict(kw, gate_width=float(value)))
    return spec_of(kw, stage_params={sid: {field: value}})


def _expected_bounds(gen, sid: str, field: str):
    if field == "size":
        return _size_bounds()
    if field == "min_num_points":
        return (2, int(gen.params[sid].params.max_num_points))
    if field == "gate_width":
        return (0.0, None)
    raise AssertionError(f"no expected bounds for {sid}.{field}")


def _launched(names: list, kernel: str) -> bool:
    return any(n.rsplit(".", 1)[-1] == kernel for n in names)


def _prepare_masked(gen, mask: wp.array) -> None:
    """Warm up ``gen``'s masked body with an all-zero ``mask``: an exact no-op (nothing
    committed, no epoch advanced)."""
    out = gen.program.result
    base = _snapshot(out)
    epoch = gen.seeds.epoch.numpy().copy()
    np.testing.assert_array_equal(mask.numpy(), 0, err_msg="the warm-up mask is zero")
    assert gen.enqueue(advance=True, mask=mask) is out
    wp.synchronize()
    _assert_host(_snapshot(out), base, "an all-zero mask must change nothing")
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch)


def _keys(randomize: dict) -> list:
    """``[(stage id, field, lo, hi)]`` of a ``randomize=`` mapping (split at the LAST
    dot, as ``PipelineSpec.with_randomize``)."""
    out = []
    for key, (lo, hi) in randomize.items():
        sid, field = key.rsplit(".", 1)
        out.append((sid, field, lo, hi))
    return out


def _check_spec(spec, randomize: dict, label: str) -> None:
    ids = list(spec.ids)
    for key in randomize:
        rid, target = f"randomize.{key}", key.rsplit(".", 1)[0]
        assert rid in ids and target in ids and ids.index(rid) < ids.index(target), \
            f"{label}: {rid} must run before {target}: {ids}"


# ---------------------------------------------------------------------------------------
# 1. Each twin == its original kernel, on generator-produced inputs
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Pair:
    original: str              # "module:kernel": the original (scalar) kernel
    twin: str                  # "module:kernel": its PR13d twin
    how: str                   # "replace": the scalar at `index` becomes an [E] array;
                               # "append": an [E] float32 size array is appended;
                               # "drop": the scalar at `index` (a toggle) is removed
    index: "int | None"        # replace / drop: the original's argument index
    scalar: "int | None"       # append: the original's scalar the size multiplies
    source: str                # the generation whose launch provides the inputs


_SAMPLING = "track_gen.ops.sampling"
_GEOMETRY = "track_gen.ops.geometry"
_GATES = "track_gen.stages.gates"
_WARP_GATE = "track_gen._src.warp_gate"
_PAIRS = {
    "sample_counts": _Pair(f"{_SAMPLING}:_sample_counts_k", f"{_SAMPLING}:_sample_counts_dyn_k",
                           "replace", 3, None, "track:bezier"),
    "sample_grid_points": _Pair(f"{_SAMPLING}:_sample_grid_points_k",
                                f"{_SAMPLING}:_sample_grid_points_sized_k",
                                "append", None, 6, "track:bezier"),
    "sample_grid_points_scaled": _Pair(f"{_SAMPLING}:_sample_grid_points_scaled_k",
                                       f"{_SAMPLING}:_sample_grid_points_scaled_sized_k",
                                       "append", None, 6, "ops:scaled-grid"),
    "normalize_extent": _Pair(f"{_GEOMETRY}:_normalize_extent_k",
                              f"{_GEOMETRY}:_normalize_extent_sized_k",
                              "append", None, 2, "track:polar"),
    "normalize_extent_to": _Pair(f"{_GATES}:_normalize_extent_to_k",
                                 f"{_GATES}:_normalize_extent_to_sized_k",
                                 "append", None, 3, "gates:polar"),
    "finalize_frame": _Pair(f"{_WARP_GATE}:_finalize_frame_k",
                            f"{_WARP_GATE}:_finalize_frame_dyn_k",
                            "replace", 9, None, "gates:bezier"),
    "finalize_validity": _Pair(f"{_WARP_GATE}:_finalize_validity_k",
                               f"{_WARP_GATE}:_finalize_validity_bars_k",
                               "drop", 11, None, "gates:bezier"),
}
#: The PR9 composition configs whose (frozen legacy) generator launches the originals.
_TRACK_SOURCES = {"bezier": dict(generator="bezier"),
                  "polar": dict(generator="polar")}


def _run_pr9(kw: dict) -> None:
    """The PR9 composition (``warp_pipeline._run_pipeline``) for
    ``TrackGenConfig(**kw)`` at :data:`SEEDS`: the frozen legacy generator spec, i.e. the
    ORIGINAL kernels."""
    with _quiet():
        cfg = TrackGenConfig(device=DEV, num_envs=E, **kw)
        track, scratch = warp_pipeline._inflate_warp_alloc(
            cfg, generator_spec=_reference_generator_spec(cfg.generator))
    spec = scratch.generator_spec
    assert not isinstance(spec, _StageGenerator) and callable(getattr(spec, "generate", None)), \
        f"the PR9 composition must run the legacy spec of {cfg.generator!r}, got {spec!r}"
    seed = wp.array(SEEDS, dtype=wp.int32, device=DEV)
    warp_pipeline._run_pipeline(cfg, seed, out=track, scratch=scratch,
                                ctx=LaunchContext.for_call(DEV))
    wp.synchronize()


def _source(source: str, widths=None):
    """``(run, generator)``: ``run()`` performs the generation whose launches are
    captured (the generator is built BEFORE, so its construction launches are not)."""
    kind, name = source.split(":")
    if kind == "ops":
        from track_gen import ops

        assert name == "scaled-grid" and widths is None
        points_per_env = 13
        cells, cell_size = ops.grid_cells(0.05)
        seed = _dev(SEEDS, wp.int32)
        scale = _dev(np.linspace(0.8, 1.2, E, dtype=np.float32))
        used = wp.empty(E * points_per_env, dtype=wp.int32, device=DEV)
        points = wp.empty(E * points_per_env, dtype=wp.vec2f, device=DEV)
        ctx = LaunchContext.for_call(DEV)

        def run():
            ops.sample_grid_points(ctx, seed, 9781, 0, points_per_env, cells,
                                   float(cell_size), scale, used, points)
            wp.synchronize()
        return run, None
    if kind == "track":
        assert widths is None
        kw = _TRACK_SOURCES[name]
        return (lambda: _run_pr9(kw)), None
    gen = _build(GateGenerator, _gate_spec(dict(generator=name)))
    if widths is not None:
        gen.params["gate_frames"].set("gate_width", _dev(_per_env(widths)))
        wp.synchronize()

    def run():
        gen.enqueue()
        wp.synchronize()
    return run, gen


def _clone_all(args) -> list:
    """Clone every array argument; an array passed twice gets ONE clone (aliasing kept)."""
    memo: dict = {}
    out = []
    for a in args:
        if isinstance(a, wp.array):
            k = (a.ptr, tuple(a.shape), a.dtype)
            if k not in memo:
                memo[k] = wp.clone(a)
            out.append(memo[k])
        else:
            out.append(a)
    return out


def _spy_launches(monkeypatch, kernels, run) -> dict:
    """Run ``run()`` with ``LaunchContext.launch`` spied: ``id(kernel) -> (kernel, dim,
    args)`` of the FIRST launch of each of ``kernels`` (inputs then outputs, every array
    cloned right before the launch: the inputs a real generation handed it)."""
    real = LaunchContext.launch
    wanted = {id(k) for k in kernels}
    got: dict = {}

    def spy(self, kernel, dim, inputs=(), outputs=(), **kw):
        if id(kernel) in wanted and id(kernel) not in got:
            wp.synchronize_device(DEV)              # the work before it has completed
            args = _clone_all([*inputs, *outputs])
            wp.synchronize_device(DEV)
            got[id(kernel)] = (kernel, dim, args)
        return real(self, kernel, dim, inputs, outputs, **kw)

    with monkeypatch.context() as m:
        m.setattr(LaunchContext, "launch", spy)
        run()
    wp.synchronize()
    return got


def _canonical(pair: _Pair, from_original: bool, args: list) -> list:
    """The ORIGINAL kernel's argument list, from a launch of either kernel (the live
    PR13d Program launches the twin; the PR9 composition the original)."""
    args = list(args)
    if from_original:
        return args
    if pair.how == "append":
        return args[:-1]
    if pair.how == "replace":
        knob = args[pair.index].numpy()
        assert knob.tobytes() == np.full_like(knob, knob[0]).tobytes(), \
            f"{pair.twin}: a non-uniform knob was captured: {knob}"
        args[pair.index] = knob[0].item()
        return args
    assert pair.how == "drop", pair.how
    return args[:pair.index] + [1.0] + args[pair.index:]


def _run(kernel, dim, args) -> list:
    """Launch ``kernel`` on fresh copies of ``args`` (aliasing kept); return the host copy
    of every argument after the launch (``None`` for a scalar)."""
    fresh = _clone_all(args)
    wp.synchronize()
    wp.launch(kernel, dim=dim, inputs=fresh, device=DEV)
    wp.synchronize()
    return [np.array(a.numpy(), copy=True) if isinstance(a, wp.array) else None
            for a in fresh]


class _Harness:
    """One pair's captured inputs (in the original's argument order) and launches of
    both kernels on fresh copies of them."""

    def __init__(self, pair: _Pair, orig_k, twin_k, dim, args: list) -> None:
        self.pair, self.orig_k, self.twin_k, self.dim, self.args = pair, orig_k, twin_k, dim, args
        self.arrays = [i for i, a in enumerate(args) if isinstance(a, wp.array)]
        targs = self.twin_k.adj.args
        n = len(args) + {"append": 1, "replace": 0, "drop": -1}[pair.how]
        assert len(targs) == n, \
            f"{pair.twin}: {[a.label for a in targs]}: expected {n} arguments (design 6.1)"
        if pair.how == "append":
            assert wp.types.is_array(targs[-1].type) and targs[-1].type.dtype is wp.float32, \
                f"{pair.twin}: the last argument must be the [E] float32 size"
        elif pair.how == "replace":
            assert wp.types.is_array(targs[pair.index].type), \
                f"{pair.twin}: argument {pair.index} must be the [E] knob array"

    def orig(self, overrides=None) -> dict:
        """The original kernel (scalar arguments overridden by index)."""
        a = list(self.args)
        for i, v in (overrides or {}).items():
            a[i] = v
        host = _run(self.orig_k, self.dim, a)
        return {i: host[i] for i in self.arrays}

    def twin(self, knob: "wp.array | None" = None) -> dict:
        """The twin with ``knob`` (``None`` for a dropped toggle), its outputs indexed
        as the original's arguments; asserts the knob array was not written."""
        a = list(self.args)
        how, i = self.pair.how, self.pair.index
        if how == "append":
            a.append(knob)
        elif how == "replace":
            a[i] = knob
        else:
            del a[i]
        host = _run(self.twin_k, self.dim, a)
        if how == "drop":
            res = {j: host[j if j < i else j - 1] for j in self.arrays}
        else:
            res = {j: host[j] for j in self.arrays}
            kh = host[-1] if how == "append" else host[i]
            assert kh.tobytes() == knob.numpy().tobytes(), \
                f"{self.pair.twin} wrote its knob array"
        return res


def _capture(monkeypatch, keys, *, source=None, widths=None) -> "tuple[dict, object]":
    """``({key: _Harness}, generator)`` for the pairs ``keys``, all captured from ONE
    generation of ``source`` (default: the pairs' own)."""
    pairs = {k: _PAIRS[k] for k in keys}
    srcs = {source or p.source for p in pairs.values()}
    assert len(srcs) == 1, srcs
    run, gen = _source(srcs.pop(), widths)
    kernels = {k: (_kernel(p.original), _kernel(p.twin)) for k, p in pairs.items()}
    got = _spy_launches(monkeypatch, [k for pr in kernels.values() for k in pr], run)
    out = {}
    for key, p in pairs.items():
        orig_k, twin_k = kernels[key]
        hit = got.get(id(orig_k)) or got.get(id(twin_k))
        assert hit is not None, \
            f"{source or p.source} launched neither {p.original} nor {p.twin}"
        kernel, dim, args = hit
        out[key] = _Harness(p, orig_k, twin_k, dim, _canonical(p, kernel is orig_k, args))
    return out, gen


def _cmp(got: dict, want: dict, label: str, *, env=None, nan_ok: bool = False) -> None:
    """Every array argument of ``want`` (row ``env`` of a per-env array when given)."""
    for i in want:
        g, w = got[i], want[i]
        if env is not None and g.shape[0] % E == 0:
            g, w = g.reshape(E, -1)[env], w.reshape(E, -1)[env]
        _assert_bits(g, w, f"{label}: argument {i}", nan_ok=nan_ok)


def test_count_twin_equals_the_original_on_generator_inputs(monkeypatch):
    """``_sample_counts_dyn_k`` (bezier / hull ``min_num_points`` per env) against
    ``_sample_counts_k`` on the inputs the PR9 composition handed it: a uniform array at
    the captured ``lo`` (9), at 2, at ``hi`` and interior == the original at that ``lo``;
    below 2 (0, -5, int32 min) == the original at 2; above ``hi`` (int32 max) == the
    original at ``hi``; per env == row by row the original at each env's clamped value;
    the array is never written; the count moves with ``lo``."""
    h = _capture(monkeypatch, ["sample_counts"])[0]["sample_counts"]
    lo0, hi = int(h.args[3]), int(h.args[4])
    assert 2 <= lo0 <= hi, (lo0, hi)
    for v in (lo0, 2, hi, 5, hi - 1, 1, 0, -5, hi + 1, 40, -2 ** 31, 2 ** 31 - 1):
        c = _clamp_int(v, 2, hi)
        _cmp(h.twin(_dev(np.full(E, v), wp.int32)), h.orig({3: c}),
             f"sample_counts: lo = {v} (uniform) vs the original at {c}")
    per = np.array([lo0, 2, hi, 1, 40, -5, 5, 0], np.int32)
    got = h.twin(_dev(per, wp.int32))
    for e in range(E):
        c = _clamp_int(per[e], 2, hi)
        _cmp(got, h.orig({3: c}), f"sample_counts: per env {per.tolist()} at env {e}", env=e)
    out_i = h.arrays[-1]
    assert not np.array_equal(h.orig({3: 2})[out_i], h.orig({3: hi})[out_i]), \
        "the count does not depend on lo (the comparison would be vacuous)"


_SIZES_IN = (1.0, 0.7, 1.3, 0.1, 10.0, 2.5, 0.45)
_SIZES_OUT = (0.01, -1.0, -0.0, 0.0, -math.inf, 20.0, math.inf, 10.001)
_SIZES_PER_ENV = (1.0, 0.7, 1.3, 0.01, 20.0, 2.5, math.nan, 0.45)


@pytest.mark.parametrize("pair", ["sample_grid_points", "sample_grid_points_scaled"])
def test_grid_twin_equals_the_original_on_generator_inputs(pair, monkeypatch):
    """The sized grid twins against scalar and per-env-scale public sampling inputs: env ``e``'s points are
    ``f32(original * clamp(size[e]))`` bitwise (size 1: the original's bytes; the scalar
    grid at scale 1 also == the original launched at ``scale = clamp(size)``), the cell
    history ``used`` is unchanged, out-of-range sizes act as the bound, a NaN size passes
    the clamp (NaN points), per env row by row; the size array is never written."""
    h = _capture(monkeypatch, [pair])[0][pair]
    si = h.pair.scalar
    scalar_scale = not isinstance(h.args[si], wp.array)
    if scalar_scale:
        assert float(h.args[si]) == 1.0, f"{pair}: captured scale {h.args[si]} (want 1)"
    out_i = h.arrays[-1]
    base = h.orig()
    lo, hi = _size_bounds()

    def check(got: dict, s: float, label: str, env=None) -> None:
        c = _clamp(s, lo, hi)
        want = dict(base)
        with np.errstate(over="ignore", invalid="ignore"):
            want[out_i] = (base[out_i] * np.float32(c)).astype(np.float32)
        _cmp(got, want, f"{label}: f32(original * {c})", env=env, nan_ok=bool(np.isnan(c)))
        if scalar_scale and not np.isnan(c):
            _cmp(got, h.orig({si: float(c)}), f"{label}: the original at scale={c}", env=env)

    for s in _SIZES_IN + _SIZES_OUT + (math.nan,):
        got = h.twin(_dev(np.full(E, s, np.float32)))
        check(got, s, f"{pair}: size = {s!r} (uniform)")
        if s == 1.0:
            _cmp(got, base, f"{pair}: size 1 vs the original's bytes")
    per = _per_env(_SIZES_PER_ENV)
    got = h.twin(_dev(per))
    for e in range(E):
        check(got, float(per[e]), f"{pair}: per env {per.tolist()} at env {e}", env=e)


@pytest.mark.parametrize("pair", ["normalize_extent", "normalize_extent_to"])
def test_normalize_twin_equals_the_original_on_generator_inputs(pair, monkeypatch):
    """The sized normalization twins (polar / voronoi / checkpoint track generators, in
    place; the polar / voronoi / checkpoint gate pipelines' ``normalize.extent``, out of
    place) against the originals on the inputs a real generation handed them: size 1 ==
    the original's bytes; env ``e`` == the original at the target ``f32(target *
    clamp(size[e]))`` (the twin's product), out-of-range sizes at the bound, a NaN size
    at a NaN target; per env row by row; the size array is never written."""
    h = _capture(monkeypatch, [pair])[0][pair]
    ti = h.pair.scalar
    target = np.float32(h.args[ti])
    assert np.isfinite(target) and target > 0, target
    base = h.orig()
    lo, hi = _size_bounds()

    def at(s: float) -> dict:
        return h.orig({ti: float(np.float32(target * _clamp(s, lo, hi)))})

    for s in _SIZES_IN + _SIZES_OUT + (math.nan,):
        got = h.twin(_dev(np.full(E, s, np.float32)))
        _cmp(got, at(s), f"{pair}: size = {s!r} (uniform) vs the original at target*size")
        if s == 1.0:
            _cmp(got, base, f"{pair}: size 1 vs the original's bytes")
    per = _per_env(_SIZES_PER_ENV)
    got = h.twin(_dev(per))
    for e in range(E):
        _cmp(got, at(float(per[e])), f"{pair}: per env {per.tolist()} at env {e}", env=e)
    assert _differs({0: at(1.3)[h.arrays[-1]]}, {0: base[h.arrays[-1]]}), \
        f"{pair}: the size does not move the points (vacuous)"


def test_frame_twin_equals_the_original_on_generator_inputs(monkeypatch):
    """``_finalize_frame_dyn_k`` (``gate_frames.gate_width`` per env) against
    ``_finalize_frame_k`` on the inputs a bezier gate generation handed it: a uniform
    array (0, -0.0, interior, 1e9, +inf: no upper bound, NaN) == the original at that
    width; negative widths (-0.3, -1e-3, -inf) == the original at 0; per env row by row;
    the width array is never written; the width moves the bars."""
    h = _capture(monkeypatch, ["finalize_frame"])[0]["finalize_frame"]
    assert float(h.args[9]) == 0.0, f"captured gate_width {h.args[9]} (point gates: 0)"
    for v in (0.0, -0.0, 0.1, 0.3, 1.0, 1.0e9, math.inf, math.nan, -0.3, -1.0e-3, -math.inf):
        c = float(_clamp(v, 0.0, None))
        _cmp(h.twin(_dev(np.full(E, v, np.float32))), h.orig({9: c}),
             f"finalize_frame: gate_width = {v!r} (uniform) vs the original at {c!r}")
    per = _per_env((0.0, 0.2, -0.3, 0.5, -0.0, math.nan, -math.inf, 0.05))
    got = h.twin(_dev(per))
    for e in range(E):
        c = float(_clamp(per[e], 0.0, None))
        _cmp(got, h.orig({9: c}), f"finalize_frame: per env {per.tolist()} at env {e}", env=e)
    hs = 4                                               # half_size
    assert _differs({0: h.orig({9: 0.3})[hs]}, {0: h.orig({9: 0.0})[hs]}), \
        "finalize_frame: the width does not move the bars (vacuous)"


_GATE_GENERATORS = ("bezier", "hull", "polar", "voronoi", "checkpoint")
#: Generators whose bars at width 0.5 cross on nearly every course (design A2: valid
#: 0.039 / 0.023 / 0.039), so 8 envs surely show the crossing bit.
_CROSSING_GENERATORS = ("bezier", "hull", "voronoi")
_BAR_WIDTHS = (0.0, 0.2, 0.5, 0.0, 0.35, 0.0, 0.5, 0.1)


@pytest.mark.parametrize("generator", _GATE_GENERATORS)
def test_validity_twin_equals_the_toggle_on_kernel_on_real_gates(generator, monkeypatch):
    """``_finalize_validity_bars_k`` against the PR11 ``_finalize_validity_k`` on the
    gates a real generation handed it, ``gate_frames.gate_width`` set per env (0 / 0.2 /
    0.35 / 0.5 / 0.1): ``valid`` and ``fail`` equal the toggle-ON kernel in every env
    and the toggle-OFF kernel in the zero-width envs (skipping the crossing loop there
    changes no decision); the Program's own GateSequence carries the twin's bits; for
    bezier / hull / voronoi the crossing check matters (toggle ON != OFF, the crossing bit
    is set somewhere)."""
    hs, gen = _capture(monkeypatch, ["finalize_validity"], source=f"gates:{generator}",
                       widths=_BAR_WIDTHS)
    h = hs["finalize_validity"]
    valid_i, fail_i = h.arrays[-2], h.arrays[-1]
    count = h.args[7].numpy()
    G = int(h.args[8])
    half = h.args[4].numpy().reshape(E, G)
    widths = _per_env(_BAR_WIDTHS)
    for e in range(E):                                   # the per-env widths reached it
        real = half[e, :max(0, min(int(count[e]), G))]
        np.testing.assert_array_equal(real, np.float32(0.5) * widths[e],
                                      err_msg=f"{generator}: env {e} half sizes")
    on, off, tw = h.orig({11: 1.0}), h.orig({11: 0.0}), h.twin(None)
    _cmp(tw, on, f"{generator}: bars twin vs the toggle-ON kernel")
    zero = np.flatnonzero(widths == 0.0)
    for e in zero:
        _cmp(tw, off, f"{generator}: bars twin vs the toggle-OFF kernel (zero width)", env=e)
    seq = _snapshot(gen.program.result)
    np.testing.assert_array_equal(seq["valid"], tw[valid_i],
                                  err_msg=f"{generator}: GateSequence.valid vs the twin")
    np.testing.assert_array_equal(seq["fail_mask"], tw[fail_i],
                                  err_msg=f"{generator}: GateSequence.fail_mask vs the twin")
    if generator in _CROSSING_GENERATORS:
        assert np.any(on[fail_i] & _CROSSING) and _differs({0: on[fail_i]}, {0: off[fail_i]}), \
            f"{generator}: no bar crossing at widths {widths.tolist()} (vacuous)"


def _posed_half_sizes(G: int) -> np.ndarray:
    """``[E, G]`` per-gate half sizes: point gates, positive, signed zero, negative,
    NaN, both signs, a tiny non-zero bar."""
    rows = ([0.0], [0.15], [0.0, 0.15, -0.0], [0.0, -0.15], [0.0, math.nan, 0.0, 0.15],
            [-0.2], [0.3, -0.3, 0.0, math.nan], [0.0, 1.0e-30])
    return np.stack([np.resize(np.asarray(r, np.float32), G) for r in rows])


def test_validity_twin_on_posed_like_half_sizes(monkeypatch):
    """The validity twin on frames built by ``_finalize_frame_posed_k`` (left / right by
    the frames' formula) from a real bezier gate generation with per-gate half sizes
    ``{0, -0.0, +, -, NaN, 1e-30}``: ``valid`` / ``fail`` == the toggle-ON kernel in every
    env, == the toggle-OFF kernel where every real gate has a zero half size (``-0.0``
    included); the bars cross somewhere and NaN bars fail the finite bit."""
    hs, _ = _capture(monkeypatch, ["finalize_frame", "finalize_validity"])
    fr, va = hs["finalize_frame"], hs["finalize_validity"]
    posed = _kernel(f"{_GATES}:_finalize_frame_posed_k")
    G = int(fr.args[8])
    count = fr.args[7].numpy()
    pattern = _posed_half_sizes(G)
    outs = [wp.clone(a) for a in fr.args[:7]]
    fallbacks = wp.zeros(E, dtype=wp.int32, device=DEV)
    hs_in = _dev(pattern.ravel())
    roll = wp.zeros(E * G, dtype=wp.float32, device=DEV)
    pos_in, tan_in = wp.clone(fr.args[12]), wp.clone(fr.args[13])
    wp.synchronize()
    wp.launch(posed, dim=E * G, inputs=[*outs, fr.args[7], G, int(fr.args[10]), fallbacks,
                                        pos_in, tan_in, hs_in, roll], device=DEV)
    wp.synchronize()
    real = [pattern[e, :max(0, min(int(count[e]), G))] for e in range(E)]
    half = outs[4].numpy().reshape(E, G)
    for e in range(E):
        _assert_bits(half[e, :len(real[e])], real[e], f"posed-like: env {e} half sizes",
                     nan_ok=True)
    va = _Harness(va.pair, va.orig_k, va.twin_k, va.dim, outs + list(va.args[7:]))
    valid_i, fail_i = va.arrays[-2], va.arrays[-1]
    on, off, tw = va.orig({11: 1.0}), va.orig({11: 0.0}), va.twin(None)
    _cmp(tw, on, "posed-like: bars twin vs the toggle-ON kernel")
    zero = [e for e in range(E) if np.all(real[e] == 0.0)]
    assert 0 in zero and len(zero) < E, zero
    for e in zero:
        _cmp(tw, off, f"posed-like: env {e} (zero half sizes) vs the toggle-OFF kernel",
             env=e)
    assert np.any(on[fail_i] & _CROSSING), f"no crossing bar: {on[fail_i].tolist()}"
    finite_bit = 1 << GATE_FAIL_BITS["finite"]
    for e in range(E):
        if np.any(np.isnan(real[e])):
            assert tw[fail_i][e] & finite_bit, f"env {e}: a NaN bar must fail 'finite'"


#: Discriminating payloads (design 9.2 (2)), two per vec2: signalling NaNs, subnormals,
#: quiet NaNs (default and non-default payloads), finite, infinities, a finite + a NaN /
#: an infinity in one vector (the vector is non-finite), signed zero, max / min normal.
_PAYLOADS = np.array([[0x7F800001, 0xFF800123], [0x00000003, 0x80000003],
                      [0x7FC00001, 0xFFC00000], [0x3F800000, 0xBF9DF3B6],
                      [0x7F800000, 0xFF800000], [0x7FFFFFFF, 0x00000000],
                      [0x80000000, 0x3E99999A], [0x7F7FFFFF, 0x00800000],
                      [0x41200000, 0x7F800001], [0xC0490FDB, 0x7F800000]], np.uint32)
#: The env sizes of the copy launch: 1, interior, below / above the bounds, NaN.
_COPY_SIZES = (1.0, 1.3, 0.01, 20.0, math.nan)


def _copy_sized() -> "tuple[np.ndarray, np.ndarray]":
    """Launch ``_copy_sized_k`` (repulsive's final ``rs_out -> raw.xy`` copy) on
    :data:`_PAYLOADS` repeated for each env of :data:`_COPY_SIZES`; returns ``(src,
    out)`` as ``[envs * N, 2]`` float32 host arrays. The argument roles are read from the
    kernel's signature (two vec2f arrays: source and output; one float32 array: the
    sizes; one int: the per-env point count)."""
    k = _kernel("track_gen._src.warp_generate_repulsive:_copy_sized_k")
    n, envs = len(_PAYLOADS), len(_COPY_SIZES)
    src_host = np.ascontiguousarray(np.tile(_PAYLOADS, (envs, 1))).view(np.float32)
    src = wp.array(src_host, dtype=wp.vec2f, device=DEV)
    sizes = _dev(np.asarray(_COPY_SIZES, np.float32))
    out = wp.zeros(envs * n, dtype=wp.vec2f, device=DEV)
    tg_capture.poison(out)

    def role(a) -> str:
        if wp.types.is_array(a.type) and a.type.dtype is wp.vec2f:
            return "vec2"
        if wp.types.is_array(a.type) and a.type.dtype is wp.float32:
            return "size"
        return "int" if a.type is wp.int32 else "other"

    roles = [role(a) for a in k.adj.args]
    assert sorted(roles) == ["int", "size", "vec2", "vec2"], \
        f"_copy_sized_k{[(a.label, a.type) for a in k.adj.args]}: expected (src, N, size, out)"
    vec_labels = [a.label for a, r in zip(k.adj.args, roles) if r == "vec2"]
    out_label = next((lb for lb in vec_labels if lb in ("out", "dst", "dest")), vec_labels[1])
    args = []
    for a, r in zip(k.adj.args, roles):
        if r == "vec2":
            args.append(out if a.label == out_label else src)
        elif r == "size":
            args.append(sizes)
        else:
            args.append(n)
    wp.synchronize()
    wp.launch(k, dim=envs * n, inputs=args, device=DEV)
    wp.synchronize()
    assert sizes.numpy().tobytes() == np.asarray(_COPY_SIZES, np.float32).tobytes()
    assert src.numpy().tobytes() == src_host.tobytes(), "_copy_sized_k wrote its source"
    return src_host, out.numpy().reshape(-1, 2)


def test_copy_sized_meets_the_device_contract():
    """``_copy_sized_k`` on ``cuda:0`` (design 1.3, the contract that holds on a device):
    a finite vector at size 1 is copied bitwise (subnormals and ``-0.0`` included: no
    flush-to-zero), at any other size it is ``f32(v * clamp(size))`` (the bounds 0.1 /
    10; a NaN size gives NaN); a vector with a non-finite component is kept UNSCALED at
    every size: its finite component and ``±inf`` bitwise, a NaN stays NaN (``isnan``
    here; the payloads are asserted bitwise by
    :func:`test_copy_sized_keeps_nan_payloads_bitwise`)."""
    src, out = _copy_sized()
    n = len(_PAYLOADS)
    lo, hi = _size_bounds()
    want = np.empty_like(src)
    for e, s in enumerate(_COPY_SIZES):
        c = _clamp(s, lo, hi)
        v = src[e * n:(e + 1) * n]
        fin = np.all(np.isfinite(v), axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (v * np.float32(c)).astype(np.float32)
        w = np.where(fin[:, None], v if c == np.float32(1.0) else scaled, v)
        want[e * n:(e + 1) * n] = w
    for e, s in enumerate(_COPY_SIZES):
        rows = slice(e * n, (e + 1) * n)
        _assert_bits(out[rows], want[rows].astype(np.float32),
                     f"_copy_sized_k: size {s!r} (clamped {_clamp(s, lo, hi)!r})", nan_ok=True)


def test_copy_sized_keeps_nan_payloads_bitwise():
    """At size 1 every pattern (signalling NaNs ``0x7f800001`` / ``0xff800123``, quiet
    ``0x7fc00001`` / ``0xffc00000`` / ``0x7fffffff``, subnormals, ``±inf``) is copied
    bit for bit; at every other size a non-finite vector is kept bit for bit. A HARD
    assertion (design 1.3 / 9.3 (1)): the finite-gated form exists so that no NVVM
    select fold can quiet a payload (design A3's CUDA source). If it fails on a device,
    the fallback contract applies (finite bitwise, a NaN stays NaN): relax this test to
    ``isnan`` in the same change and record the device in repulsive.rst and the
    CHANGELOG."""
    src, out = _copy_sized()
    n = len(_PAYLOADS)
    _assert_bits(out[:n], src[:n], "_copy_sized_k: size 1 payloads")
    for e in range(1, len(_COPY_SIZES)):
        rows = slice(e * n, (e + 1) * n)
        fin = np.all(np.isfinite(src[rows]), axis=1)
        _assert_bits(out[rows][~fin], src[rows][~fin],
                     f"_copy_sized_k: size {_COPY_SIZES[e]!r} non-finite payloads")


# ---------------------------------------------------------------------------------------
# 2. Per env == per-env scalar runs (every generator, gates, repulsive)
# ---------------------------------------------------------------------------------------

_SIZES = (1.0, 0.7, 1.3, 1.0, 0.45, 2.2, 1.6, 0.9)
#: Repulsive truncates from ~1.6 at the default n_max (design 1.5): stay below it.
_SIZES_REPULSIVE = (1.0, 0.7, 1.3, 1.0, 0.85, 1.5, 1.2, 0.9)
_COUNTS = (9, 2, 13, 11, 1, 40, 5, -3)
_GRID_SIZED = (("_sample_grid_points_sized_k",), ("_sample_grid_points_k",))
_GATE_TWINS = (("_finalize_frame_dyn_k", "_finalize_validity_bars_k"),
               ("_finalize_frame_k", "_finalize_validity_k"))
_NORM_TRACK = (("_normalize_extent_sized_k",), ("_normalize_extent_k",))
_NORM_GATES = (("_normalize_extent_to_sized_k",), ("_normalize_extent_to_k",))
_COUNT_TWINS = (("_sample_counts_dyn_k",), ("_sample_counts_k",))


def _gates_twins(twins):
    return twins[0] + _GATE_TWINS[0], twins[1] + _GATE_TWINS[1]


#: case -> (kind, presets kwargs, stage id, field, per-env values, (twins, originals)).
_PER_ENV = {
    "track.bezier.size": ("track", dict(generator="bezier"), "generator", "size", _SIZES,
                          _GRID_SIZED),
    "track.hull.size": ("track", dict(generator="hull"), "generator", "size", _SIZES,
                        _GRID_SIZED),
    "track.polar.size": ("track", dict(generator="polar"), "generator", "size", _SIZES,
                         _NORM_TRACK),
    "track.voronoi.size": ("track", dict(generator="voronoi"), "generator", "size", _SIZES,
                           _NORM_TRACK),
    "track.checkpoint.size": ("track", dict(generator="checkpoint"), "generator", "size",
                              _SIZES, _NORM_TRACK),
    "track.repulsive.size": ("track", dict(generator="repulsive", **_REPULSIVE_SMALL),
                             "generator", "size", _SIZES_REPULSIVE,
                             (("_copy_sized_k",), ())),
    "track.bezier.min_num_points": ("track", dict(generator="bezier"), "generator",
                                    "min_num_points", _COUNTS, _COUNT_TWINS),
    "track.hull.min_num_points": ("track", dict(generator="hull"), "generator",
                                  "min_num_points", _COUNTS, _COUNT_TWINS),
    "gates.bezier.size": ("gates", dict(generator="bezier"), "generator", "size", _SIZES,
                          _gates_twins(_GRID_SIZED)),
    "gates.hull.size": ("gates", dict(generator="hull"), "generator", "size", _SIZES,
                        _gates_twins(_GRID_SIZED)),
    "gates.stations.size": ("gates", dict(generator="bezier", source="stations"),
                            "generator", "size", _SIZES, _gates_twins(_GRID_SIZED)),
    "gates.polar.normalize.size": ("gates", dict(generator="polar"), "normalize", "size",
                                   _SIZES, _gates_twins(_NORM_GATES)),
    "gates.checkpoint.normalize.size": ("gates", dict(generator="checkpoint"), "normalize",
                                        "size", _SIZES, _gates_twins(_NORM_GATES)),
    "gates.voronoi.normalize.size": ("gates", dict(generator="voronoi"), "normalize",
                                     "size", _SIZES, _gates_twins(_NORM_GATES)),
}


@pytest.mark.parametrize("case", list(_PER_ENV))
def test_per_env_axis_equals_scalar_runs_env_by_env(case):
    """``gen.params[stage].set(field, [E] array)`` then ``enqueue()`` on ``cuda:0``: env
    ``e`` equals env ``e`` of a separate run BUILT with the scalar ``clamp(v_e)``
    (``bounds(field)``: ``SIZE_BOUNDS``; ``(2, max_num_points)`` for the corner count),
    bitwise on every result field; an env at the initial value equals the default run;
    the trace shows the twin kernels and not the originals; the live array reads back
    unwritten and ``in_effect`` returns it as written; every value moves the batch."""
    kind, kw, sid, field, values, (twins, originals) = _PER_ENV[case]
    cls, spec_of = _FACTORY[kind]
    is_int = field == "min_num_points"
    np_dt, wp_dt = (np.int32, wp.int32) if is_int else (np.float32, wp.float32)
    per_env = _per_env(values, np_dt)
    gen = _build(cls, spec_of(kw))
    sp = gen.params[sid]
    lo, hi = sp.bounds(field)
    assert (lo, hi) == _expected_bounds(gen, sid, field), f"{case}: bounds {lo, hi}"
    initial = sp[field].numpy().copy()
    assert initial.dtype == np_dt and np.all(initial == initial[0]), initial
    src = _dev(per_env, wp_dt)
    sp.set(field, src)
    wp.synchronize()
    with trace_launches() as t:
        out = gen.enqueue()
    wp.synchronize()
    got = _snapshot(out)
    names = [r.kernel for r in t]
    for k in twins:
        assert _launched(names, k), f"{case}: the per-env run must launch {k}"
    for k in originals:
        assert not _launched(names, k), f"{case}: the per-env run launched the original {k}"
    _assert_bits(sp[field].numpy(), per_env, f"{case}: the stage wrote its live array")
    _assert_bits(sp.in_effect(field).numpy(), per_env,
                 f"{case}: in_effect must return the values read, as written")

    default = _eager(cls, spec_of(kw))
    used = _used(per_env, lo, hi)
    refs = {}
    for c in used:
        k = _key(c, np_dt)
        if k in refs:
            continue
        if k == _key(initial[0], np_dt):
            refs[k] = default
            continue
        refs[k] = _eager(cls, _ref_spec(kind, kw, sid, field, c.item()))
        assert _differs(refs[k], default), \
            f"{case}: {field}={c} does not change the batch (the comparison would be vacuous)"
    _assert_bitwise(got, _stitch([refs[_key(c, np_dt)] for c in used]),
                    f"{case}: per-env {per_env.tolist()} (used {used.tolist()}) vs the "
                    f"scalar runs env by env")


#: case -> (kind, presets kwargs, stage id): out-of-range size arrays.
_SIZE_CLAMPED = {
    "track.bezier": ("track", dict(generator="bezier"), "generator"),
    "track.polar": ("track", dict(generator="polar"), "generator"),
    "gates.polar.normalize": ("gates", dict(generator="polar"), "normalize"),
}


@pytest.mark.parametrize("case", list(_SIZE_CLAMPED))
def test_out_of_range_size_arrays_are_clamped_at_use(case):
    """A per-env ``size`` array below / above ``SIZE_BOUNDS`` (``±inf`` included):
    ``set`` accepts it (arrays are clamped at use, scalars are checked), the live array
    and ``in_effect`` keep the values as written, and env ``e`` equals a run whose size
    was ``set`` to the scalar ``clamp(v_e, *bounds("size"))``, bitwise on every field."""
    kind, kw, sid = _SIZE_CLAMPED[case]
    cls, spec_of = _FACTORY[kind]
    per_env = _per_env((0.01, 20.0, -1.0, math.inf, -math.inf, 0.1, 10.0, 1.0))
    gen = _build(cls, spec_of(kw))
    sp = gen.params[sid]
    lo, hi = sp.bounds("size")
    assert (lo, hi) == _size_bounds()
    sp.set("size", _dev(per_env))
    wp.synchronize()
    got = _snapshot(gen.enqueue())
    wp.synchronize()
    _assert_bits(sp["size"].numpy(), per_env, f"{case}: the live array keeps what was set")
    _assert_bits(sp.in_effect("size").numpy(), per_env, f"{case}: in_effect (as written)")
    refs = {}
    for c in _used(per_env, lo, hi):
        if _key(c) not in refs:
            ref = _build(cls, spec_of(kw))
            ref.params[sid].set("size", float(c))
            wp.synchronize()
            refs[_key(c)] = _snapshot(ref.enqueue())
            wp.synchronize()
    _assert_bitwise(got, _stitch([refs[_key(c)] for c in _used(per_env, lo, hi)]),
                    f"{case}: per-env {per_env.tolist()} vs the scalar bound runs")


def test_repulsive_size_scales_its_raw_loop_exactly():
    """Repulsive's ``size`` multiplies its final copy only (the growth does not see it):
    the raw loop (``raw.xy``) of env ``e`` at size ``s_e`` is ``f32(raw(1) * s_e)`` for a
    finite vector and the size-1 vector otherwise (a NaN stays NaN: ``isnan`` on CUDA),
    and size-1 envs are the default bytes. The device check of ``_copy_sized_k`` end to
    end."""
    kw = dict(generator="repulsive", **_REPULSIVE_SMALL)
    base = _build(TrackGenerator, _track_spec(kw))
    base.enqueue()
    wp.synchronize()
    raw1 = base.program.tap("raw.xy").numpy().reshape(E, -1, 2).copy()
    sizes = _per_env(_SIZES_REPULSIVE)
    gen = _build(TrackGenerator, _track_spec(kw))
    gen.params["generator"].set("size", _dev(sizes))
    wp.synchronize()
    gen.enqueue()
    wp.synchronize()
    raws = gen.program.tap("raw.xy").numpy().reshape(E, -1, 2)
    assert np.isfinite(raw1).any(), "repulsive produced no finite raw point"
    for e in range(E):
        v = raw1[e]
        fin = np.all(np.isfinite(v), axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (v * np.float32(sizes[e])).astype(np.float32)
        want = np.where(fin[:, None], v if sizes[e] == np.float32(1.0) else scaled, v)
        _assert_bits(raws[e], want.astype(np.float32),
                     f"repulsive: env {e} raw.xy at size {sizes[e]}", nan_ok=True)


# ---------------------------------------------------------------------------------------
# 3. The per-env crossing check: point gates + gate_width set per env
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("generator", ["bezier", "hull", "polar"])
def test_gate_width_set_per_env_on_point_gates_matches_builds_with_each_width(generator):
    """A pipeline BUILT with point gates (``presets.gates()``: ``gate_frames`` and
    ``validity.gate_default`` at width 0) with ``gate_frames.gate_width`` set per env:
    env ``e`` equals env ``e`` of ``presets.gates(gate_width=w_e)`` (the configuration
    whose validity checked crossings in PR13c), bitwise on every GateSequence field
    (``half_size``, bars, ``valid``, ``fail_mask``): the crossing check runs per env on
    ``cuda:0`` (``_finalize_validity_bars_k``, not the toggle kernel). Zero-width envs
    never carry the crossing bit; for bezier / hull bars cross and fail
    (``valid == (fail_mask == 0)``)."""
    kw = dict(generator=generator)
    gen = _build(GateGenerator, _gate_spec(kw))
    sp = gen.params["gate_frames"]
    assert sp.bounds("gate_width") == (0.0, None), sp.bounds("gate_width")
    assert float(sp["gate_width"].numpy()[0]) == 0.0, "point gates at construction"
    widths = _per_env(_BAR_WIDTHS)
    sp.set("gate_width", _dev(widths))
    wp.synchronize()
    with trace_launches() as t:
        out = gen.enqueue()
    wp.synchronize()
    names = [r.kernel for r in t]
    assert _launched(names, "_finalize_frame_dyn_k") and \
        _launched(names, "_finalize_validity_bars_k"), names
    assert not _launched(names, "_finalize_validity_k"), "the static toggle kernel ran"
    got = _snapshot(out)
    default = _eager(GateGenerator, _gate_spec(kw))
    refs = {}
    for w in widths:
        if _key(w) not in refs:
            refs[_key(w)] = default if w == 0.0 else \
                _eager(GateGenerator, _gate_spec(dict(kw, gate_width=float(w))))
    _assert_bitwise(got, _stitch([refs[_key(w)] for w in widths]),
                    f"{generator}: gate_width per env {widths.tolist()} vs builds with each")
    fail = got["fail_mask"]
    np.testing.assert_array_equal(got["valid"], (fail == 0).astype(got["valid"].dtype))
    assert not np.any(fail[widths == 0.0] & _CROSSING), "a zero-width env crossed"
    if generator in ("bezier", "hull"):
        assert np.any(fail[widths > 0.0] & _CROSSING), \
            f"{generator}: no crossing at {widths.tolist()} (the check would be vacuous)"
    assert _differs(got, default), f"{generator}: the widths do not change the batch"


# ---------------------------------------------------------------------------------------
# 4. randomize on the new axes in a caller graph (in_effect(out=) recorded)
# ---------------------------------------------------------------------------------------

_TRACK_RANDOMIZE = {
    "bezier": (dict(generator="bezier"), {"generator.size": (0.7, 1.6)}),
    "hull": (dict(generator="hull"), {"generator.size": (0.7, 1.6)}),
    "polar": (dict(generator="polar"), {"generator.size": (0.7, 1.6)}),
    "voronoi": (dict(generator="voronoi"), {"generator.size": (0.7, 1.6)}),
    "checkpoint": (dict(generator="checkpoint"), {"generator.size": (0.7, 1.6)}),
    "repulsive": (dict(generator="repulsive", **_REPULSIVE_SMALL),
                  {"generator.size": (0.7, 1.4)}),
    "bezier.size+rad": (dict(generator="bezier"),
                        {"generator.size": (0.7, 1.6), "generator.rad": (0.1, 0.35)}),
}
#: presets.gates cases: (kwargs, randomize, the crossing bit must show).
_GATE_RANDOMIZE = {
    "bezier.generator.size": (dict(generator="bezier"), {"generator.size": (0.7, 1.6)},
                              False),
    "hull.generator.size": (dict(generator="hull"), {"generator.size": (0.7, 1.6)}, False),
    "polar.normalize.size": (dict(generator="polar"), {"normalize.size": (0.7, 1.6)},
                             False),
    "checkpoint.normalize.size": (dict(generator="checkpoint"),
                                  {"normalize.size": (0.7, 1.6)}, False),
    "voronoi.normalize.size": (dict(generator="voronoi"), {"normalize.size": (0.7, 1.6)},
                               False),
    "bezier.gate_width": (dict(generator="bezier"),
                          {"gate_frames.gate_width": (0.2, 0.5)}, True),
    "polar.gate_width": (dict(generator="polar"), {"gate_frames.gate_width": (0.2, 0.5)},
                         False),
    "bezier.size+gate_width": (dict(generator="bezier"),
                               {"generator.size": (0.7, 1.6),
                                "gate_frames.gate_width": (0.2, 0.5)}, True),
}


def _assert_randomized_replays(kind: str, kw: dict, randomize: dict, label: str, *,
                               replays: int = 3, expect_crossing: bool = False) -> None:
    """ONE caller graph holding ``generate(stream=s, advance=True)`` of the randomized
    pipeline followed by ``in_effect(field, out=buf, stream=s)`` of every drawn field;
    every replay into a poisoned result, poisoned drawn arrays and poisoned ``buf``s:
    ``buf`` == the drawn array, draws in ``[lo, hi]``, distinct, redrawn for every env and
    equal to the randomized eager twin's; the batch equals that twin's and the
    non-randomized twin's ``set`` to the recorded ``buf``s, and differs from the
    default."""
    cls, spec_of = _FACTORY[kind]
    spec = spec_of(kw, randomize)
    _check_spec(spec, randomize, label)
    keys = _keys(randomize)
    gen, twin = _build(cls, spec), _build(cls, spec)
    plain, default = _build(cls, spec_of(kw)), _build(cls, spec_of(kw))
    out = gen.program.result
    drawn = {(sid, f): gen.params[sid][f] for sid, f, _, _ in keys}
    for (sid, f), arr in drawn.items():
        assert gen.params[sid].drawn.get(f) == f"randomize.{sid}.{f}", gen.params[sid]
        assert gen.params[sid].in_effect(f).ptr == arr.ptr, \
            f"{label}: in_effect({f!r}) of a drawn field must be the drawn array"
        assert plain.params[sid].in_effect(f).ptr != plain.params[sid][f].ptr, \
            f"{label}: in_effect({f!r}) of a latched field must be the work copy"
    ie = {k: wp.zeros(E, dtype=wp.float32, device=DEV) for k in drawn}
    prev = {k: a.numpy().copy() for k, a in drawn.items()}
    for (sid, f), v in prev.items():
        np.testing.assert_array_equal(v, twin.params[sid][f].numpy(),
                                      err_msg=f"{label}: construction draws {sid}.{f}")

    def body(stream):
        assert gen.generate(stream=stream, advance=True) is out
        for (sid, f), buf in ie.items():
            assert gen.params[sid].in_effect(f, out=buf, stream=stream) is buf

    replay = _caller_graph("warp", body)
    assert gen.graph is None, f"{label}: recording into the caller's graph self-captured"
    for k, v in prev.items():
        np.testing.assert_array_equal(drawn[k].numpy(), v, err_msg=f"{label}: recording drew")
        np.testing.assert_array_equal(ie[k].numpy(), 0.0, err_msg=f"{label}: recording copied")

    crossing = False
    for r in range(replays):
        step = f"{label}: replay {r}"
        tg_capture.poison_all(out)
        for k in drawn:
            tg_capture.poison(drawn[k])
            tg_capture.poison(ie[k])
        wp.synchronize()
        replay()
        twin.enqueue(advance=True)
        for (sid, f), buf in ie.items():
            plain.params[sid].set(f, buf)
        wp.synchronize()
        plain.enqueue(advance=True)
        default.enqueue(advance=True)
        wp.synchronize()
        d = {k: a.numpy().copy() for k, a in drawn.items()}
        for sid, f, lo, hi in keys:
            v = d[(sid, f)]
            _assert_bits(ie[(sid, f)].numpy(), v, f"{step}: in_effect(out=) of {sid}.{f}")
            assert np.all((v >= np.float32(lo)) & (v <= np.float32(hi))), \
                f"{step}: {sid}.{f} draws {v} outside [{lo}, {hi}]"
            assert len(np.unique(v)) == E, f"{step}: {sid}.{f} draws not distinct: {v}"
            assert np.all(v != prev[(sid, f)]), f"{step}: {sid}.{f} not redrawn"
            np.testing.assert_array_equal(v, twin.params[sid][f].numpy(),
                                          err_msg=f"{step}: {sid}.{f} draws vs the twin")
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result), f"{step}: vs the randomized twin")
        _assert_host(got, _snapshot(plain.program.result),
                     f"{step}: vs the twin set to the recorded in_effect arrays")
        assert _differs(got, _snapshot(default.program.result)), \
            f"{step}: the draws do not change the batch (vacuous)"
        if "fail_mask" in got:
            crossing |= bool(np.any(got["fail_mask"] & _CROSSING))
        prev = d
    if expect_crossing:
        assert crossing, f"{label}: no drawn bar crossed in {replays} replays (vacuous)"


@pytest.mark.parametrize("case", list(_TRACK_RANDOMIZE))
def test_track_randomize_size_in_a_caller_graph(case):
    """``presets.track(randomize={"generator.size": ...})`` (each generator, repulsive
    as a BOUNDED stage with its small config, and size with ``rad``), recorded into ONE
    caller graph with ``in_effect(out=)``: every replay == the eager twins (see
    :func:`_assert_randomized_replays`)."""
    kw, randomize = _TRACK_RANDOMIZE[case]
    _assert_randomized_replays("track", kw, randomize, case)


@pytest.mark.parametrize("case", list(_GATE_RANDOMIZE))
def test_gate_randomize_size_and_bar_width_in_a_caller_graph(case):
    """``presets.gates(randomize=...)`` on ``generator.size`` (bezier / hull),
    ``normalize.size`` (polar / checkpoint / voronoi) and ``gate_frames.gate_width`` on
    a default validity (no ``PlanError``; the drawn bars are crossing-checked per env),
    recorded into ONE caller graph: every replay == the eager twins, crossing bits
    included."""
    kw, randomize, crossing = _GATE_RANDOMIZE[case]
    _assert_randomized_replays("gates", kw, randomize, case, expect_crossing=crossing)


# ---------------------------------------------------------------------------------------
# 5. in_effect under masked regeneration in a caller graph
# ---------------------------------------------------------------------------------------

#: Live values of the latched field written before each replay (in range).
_RAD_ROWS = ((0.1, 0.2, 0.3, 0.35, 0.15, 0.25, 0.05, 0.3),
             (0.3, 0.1, 0.25, 0.2, 0.35, 0.05, 0.15, 0.2),
             (0.2,) * E,
             (0.05, 0.35, 0.1, 0.3, 0.2, 0.15, 0.25, 0.1))
_WIDTH_ROWS = ((0.0, 0.2, 0.4, 0.1, 0.3, 0.0, 0.25, 0.15),
               (0.3, 0.0, 0.1, 0.5, 0.0, 0.2, 0.35, 0.05),
               (0.2,) * E,
               (0.0,) * E)
_MASKS = (_M1, 1 - _M1, _ZERO, _ONES)
#: case -> (kind, kwargs, randomize (one drawn field), (latched stage, field), live rows)
_MASKED = {
    "track.bezier": ("track", dict(generator="bezier"), {"generator.size": (0.7, 1.6)},
                     ("generator", "rad"), _RAD_ROWS),
    "gates.polar": ("gates", dict(generator="polar"), {"normalize.size": (0.7, 1.6)},
                    ("gate_frames", "gate_width"), _WIDTH_ROWS),
}


@pytest.mark.parametrize("case", list(_MASKED))
def test_in_effect_under_masked_regeneration_in_a_caller_graph(case):
    """ONE caller graph holds ``[copy mask_src -> regen_mask] -> generate(advance=True,
    mask=regen_mask) -> in_effect(drawn, out=) -> in_effect(latched, out=)``; before each
    replay the latched field's LIVE array is written (outside the graph). Per replay
    (masks mixed, complementary, zero, all):

    - the drawn field's recorded ``in_effect`` is the drawn array: masked envs redrawn,
      unmasked envs redraw the same bits (same effective seed, same latched bounds);
    - the latched field's recorded ``in_effect`` (== the zero-copy view, == the twin's)
      is the new live value on masked envs and the value the env's CURRENT rows were
      built with on unmasked envs; the live array is never written;
    - unmasked rows keep every byte (the poison); masked rows equal the randomized twin's
      and the rows of a non-randomized twin set to the recorded ``in_effect`` values and
      the live row; the current rows (unmasked from before, masked new) equal the twin's
      full batch."""
    kind, kw, randomize, (lsid, lfield), rows = _MASKED[case]
    cls, spec_of = _FACTORY[kind]
    (dsid, dfield, dlo, dhi), = _keys(randomize)
    spec = spec_of(kw, randomize)
    gen, twin, plain = _build(cls, spec), _build(cls, spec), _build(cls, spec_of(kw))
    out = gen.program.result
    m, tm, pm = (g.inputs["regen_mask"] for g in (gen, twin, plain))
    for g, mm in ((gen, m), (twin, tm), (plain, pm)):
        _prepare_masked(g, mm)
    drawn = gen.params[dsid][dfield]
    assert gen.params[dsid].in_effect(dfield).ptr == drawn.ptr
    live = gen.params[lsid][lfield]
    view = gen.params[lsid].in_effect(lfield)
    assert view.ptr != live.ptr, f"{case}: in_effect({lfield!r}) must be the work copy"
    prev_l = view.numpy().copy()
    np.testing.assert_array_equal(prev_l, live.numpy(),
                                  err_msg=f"{case}: construction latched the initial value")
    prev_d = drawn.numpy().copy()
    ie_d = wp.zeros(E, dtype=wp.float32, device=DEV)
    ie_l = wp.zeros(E, dtype=wp.float32, device=DEV)
    mask_src = _dev(_ZERO, wp.int32)
    # The rows each env currently holds (host copy): the poison below replaces the bytes
    # an unmasked env keeps, so the history lives here.
    history = _snapshot(out)

    def body(stream):
        wp.copy(m, mask_src, stream=stream)
        assert gen.generate(stream=stream, advance=True, mask=m) is out
        gen.params[dsid].in_effect(dfield, out=ie_d, stream=stream)
        gen.params[lsid].in_effect(lfield, out=ie_l, stream=stream)

    replay = _caller_graph("warp", body)
    assert gen.graph is None
    np.testing.assert_array_equal(drawn.numpy(), prev_d, err_msg=f"{case}: recording drew")
    np.testing.assert_array_equal(ie_l.numpy(), 0.0, err_msg=f"{case}: recording copied")

    for r, (mk, row) in enumerate(zip(_MASKS, rows)):
        label = f"{case}: replay {r} mask {mk.tolist()}"
        row = np.asarray(row, np.float32)
        src = _dev(row)
        gen.params[lsid].set(lfield, src)
        twin.params[lsid].set(lfield, src)
        wp.synchronize()
        _write(mask_src, mk)
        tg_capture.poison_all(out)
        tg_capture.poison(ie_d)
        tg_capture.poison(ie_l)
        wp.synchronize()
        poison = _snapshot(out)
        replay()
        _write(tm, mk)
        twin.enqueue(advance=True, mask=tm)
        plain.params[dsid].set(dfield, ie_d)
        plain.params[lsid].set(lfield, src)
        wp.synchronize()
        _write(pm, mk)
        plain.enqueue(advance=True, mask=pm)
        wp.synchronize()
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)

        d = drawn.numpy().copy()
        _assert_bits(ie_d.numpy(), d, f"{label}: in_effect(out=) of the drawn {dfield}")
        np.testing.assert_array_equal(d[unmasked], prev_d[unmasked],
                                      err_msg=f"{label}: unmasked envs redraw the same bits")
        assert np.all(d[masked] != prev_d[masked]), f"{label}: masked envs not redrawn"
        assert np.all((d >= np.float32(dlo)) & (d <= np.float32(dhi))), d
        _assert_bits(twin.params[dsid].in_effect(dfield).numpy(), d, f"{label}: twin draws")

        want_l = np.where(mk != 0, row, prev_l).astype(np.float32)
        _assert_bits(ie_l.numpy(), want_l,
                     f"{label}: in_effect(out=) of the latched {lfield} (the rows' values)")
        _assert_bits(view.numpy(), want_l, f"{label}: the zero-copy in_effect view")
        _assert_bits(twin.params[lsid].in_effect(lfield).numpy(), want_l,
                     f"{label}: the twin's in_effect")
        _assert_bits(live.numpy(), row, f"{label}: the stage wrote the live array")

        cur = _snapshot(out)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env's rows changed")
        _assert_host(_rows(cur, masked), _rows(_snapshot(twin.program.result), masked),
                     f"{label}: masked vs the randomized twin")
        _assert_host(_rows(cur, masked), _rows(_snapshot(plain.program.result), masked),
                     f"{label}: masked vs the twin set to the in_effect values")
        sel = (mk != 0)
        history = {k: np.where(sel[:, None], cur[k].reshape(E, -1),
                               history[k].reshape(E, -1)).reshape(cur[k].shape)
                   for k in cur}
        _assert_host(history, _snapshot(twin.program.result),
                     f"{label}: the current rows (masked new, unmasked kept) vs the twin's")
        prev_d, prev_l = d, want_l


# ---------------------------------------------------------------------------------------
# 6. A curriculum written in the caller's graph (out-of-bounds values included)
# ---------------------------------------------------------------------------------------

_SIZE_STEPS = ((1.0, 0.7, 1.3, 0.05, 2.0, 0.9, 25.0, 1.1),
               (0.6, 1.4, -1.0, 1.2, math.inf, 0.8, 1.0, 1.5),
               (1.2, 1.2, 1.2, -math.inf, 1.2, 1.2, 1.2, 0.01))
_COUNT_STEPS = ((9, 2, 13, 1, 40, 5, 7, 11),
                (0, 13, 9, 12, -4, 2, 100, 6),
                (3, 3, 3, 3, 3, 3, 3, 3))
_WIDTH_STEPS = ((0.0, 0.2, -0.3, 0.5, 0.1, 0.0, 0.35, 0.05),
                (0.4, 0.0, 0.1, -math.inf, 0.3, 0.2, 0.0, 0.5),
                (0.25,) * E)
#: case -> (kind, kwargs, ((stage id, field, [steps, E] schedule), ...))
_CURRICULUM = {
    "track.bezier": ("track", dict(generator="bezier"),
                     (("generator", "size", _SIZE_STEPS),
                      ("generator", "min_num_points", _COUNT_STEPS))),
    "gates.bezier": ("gates", dict(generator="bezier"),
                     (("generator", "size", _SIZE_STEPS),
                      ("gate_frames", "gate_width", _WIDTH_STEPS))),
}


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case", list(_CURRICULUM))
def test_curriculum_written_inside_the_caller_graph(case, mode):
    """ONE caller graph holds ``[the caller's kernels write each field's live array from
    row k = step % 3 of a device schedule] -> generate(advance=True) -> in_effect(field,
    out=) -> [step += 1]`` (``size``, the int32 ``min_num_points``,
    ``gate_frames.gate_width``; values below / above the bounds and ``±inf``
    included). Recording executes nothing. Every replay: the live arrays and the recorded
    ``in_effect`` copies hold row k AS WRITTEN (values read); ``clamp(in_effect,
    *bounds(field))`` are the values used: the batch equals the eager twin ``set`` to
    them and advanced alike, and differs from the default twin's; the written gate bars
    cross on some env (checked per env, from a point-gate build)."""
    kind, kw, writes = _CURRICULUM[case]
    cls, spec_of = _FACTORY[kind]
    _load_test_kernels()
    gen, twin, default = (_build(cls, spec_of(kw)) for _ in range(3))
    out = gen.program.result
    plan = []
    for sid, field, schedule in writes:
        is_int = field == "min_num_points"
        np_dt, wp_dt = (np.int32, wp.int32) if is_int else (np.float32, wp.float32)
        tab = np.asarray(schedule, np_dt)
        live = gen.params[sid][field]
        assert live.dtype is wp_dt, (sid, field, live.dtype)
        lo, hi = gen.params[sid].bounds(field)
        assert (lo, hi) == _expected_bounds(gen, sid, field), (sid, field, lo, hi)
        assert (lo is not None and np.any(tab < lo)) or (hi is not None and np.any(tab > hi)), \
            f"{sid}.{field}: the schedule must exercise the clamp"
        plan.append(dict(sid=sid, field=field, tab=tab, live=live, lo=lo, hi=hi, wp_dt=wp_dt,
                         dev_tab=_dev(tab.ravel(), wp_dt),
                         ie=wp.zeros(E, dtype=wp_dt, device=DEV),
                         kernel=_pick_i32_k if is_int else _pick_f32_k,
                         live0=live.numpy().copy()))
    steps = plan[0]["tab"].shape[0]
    assert all(p["tab"].shape == (steps, E) for p in plan)
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)

    def body(stream):
        for p in plan:
            wp.launch(p["kernel"], dim=E, inputs=[counter, steps, p["dev_tab"]],
                      outputs=[p["live"]], device=DEV, stream=stream)
        assert gen.generate(stream=stream, advance=True) is out
        for p in plan:
            gen.params[p["sid"]].in_effect(p["field"], out=p["ie"], stream=stream)
        wp.launch(_bump_k, dim=1, inputs=[counter], device=DEV, stream=stream)

    replay = _caller_graph(mode, body)
    assert gen.graph is None
    np.testing.assert_array_equal(counter.numpy(), [0], err_msg="the recording executed")
    for p in plan:
        np.testing.assert_array_equal(p["live"].numpy(), p["live0"],
                                      err_msg="the recording wrote")

    crossing = False
    for r in range(steps + 1):
        k = r % steps
        label = f"{case}/{mode}: replay {r} (step {k})"
        tg_capture.poison_all(out)
        for p in plan:
            tg_capture.poison(p["ie"])
        wp.synchronize()
        replay()
        np.testing.assert_array_equal(counter.numpy(), [r + 1], err_msg=label)
        keep = []
        for p in plan:
            row = p["tab"][k]
            _assert_bits(p["live"].numpy(), row, f"{label}: live {p['field']}")
            ie = p["ie"].numpy()
            _assert_bits(ie, row, f"{label}: in_effect(out=) of {p['field']} (as written)")
            used = _used(ie, p["lo"], p["hi"])
            _assert_bits(used, _used(row, p["lo"], p["hi"]), f"{label}: used {p['field']}")
            arr = _dev(used, p["wp_dt"])
            keep.append(arr)
            twin.params[p["sid"]].set(p["field"], arr)
        wp.synchronize()
        twin.enqueue(advance=True)
        default.enqueue(advance=True)
        wp.synchronize()
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result),
                     f"{label}: vs the twin set to clamp(in_effect, *bounds)")
        assert _differs(got, _snapshot(default.program.result)), \
            f"{label}: the in-graph writes do not change the batch (vacuous)"
        if "fail_mask" in got:
            crossing |= bool(np.any(got["fail_mask"] & _CROSSING))
    if kind == "gates":
        assert crossing, f"{case}/{mode}: no written bar crossed (the per-env check is vacuous)"


# ---------------------------------------------------------------------------------------
# 7. The conformance kit on CUDA: params.per_env for the newly dynamic built-ins
# ---------------------------------------------------------------------------------------

_KIT = (("generator", "repulsive"), ("gate_generator", "bezier"), ("gate_generator", "hull"))


@pytest.mark.parametrize("kind,name", _KIT, ids=[f"{k}.{n}" for k, n in _KIT])
def test_newly_dynamic_builtins_pass_the_kit_with_params_per_env_on_cuda(kind, name):
    """``check_generator("repulsive")`` (small config, E=8) and ``check_gate_generator``
    for the bezier / hull gate generators on ``cuda:0``: every check passes,
    ``params.per_env`` RAN (repulsive and the point gate generators gained ``size`` in
    PR13d: no longer skipped) and passed, and the caller capture replayed to the eager
    twin."""
    with _quiet():
        if kind == "generator":
            report = check_generator(name, devices=[DEV], config=_REPULSIVE_KIT, num_envs=E)
        else:
            report = check_gate_generator(name, devices=[DEV])
    report.raise_for_failures()
    per_env = report.result("params.per_env", DEV)
    assert not per_env.skipped and per_env.passed, f"{kind}.{name}: {report}"
    caller = report.result("capture.caller", DEV)
    assert not caller.skipped and caller.details.get("warp") == "ok", f"{kind}.{name}: {report}"
