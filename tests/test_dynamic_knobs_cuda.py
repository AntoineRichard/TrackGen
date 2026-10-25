"""CUDA-only: the dynamic built-in knobs (roadmap PR13c, design section 9.3).

PR13c makes the built-in curriculum knobs DYNAMIC Params fields (per-env ``[E]`` float32
arrays with declared bounds): bezier ``rad`` / ``handle_clamp_frac``, hull
``hull_displacement``, polar ``polar_radial_jitter`` / ``polar_angular_jitter``, voronoi
``voronoi_radial_variation`` / ``voronoi_angular_jitter``, checkpoint
``checkpoint_radius_min_frac`` / ``checkpoint_angle_jitter`` / ``checkpoint_turn_rate``
(the polar / voronoi / checkpoint gate generators share their ring knobs), and the
elevation ``z_min`` / ``z_max`` / ``z_max_step`` / ``z_noise_amplitude``. A Stage reads
each one through an array-reading TWIN of a scalar kernel: the knob is loaded once per
thread, clamped to its declared bounds with comparisons (no ``fminf`` / ``fmaxf``), then
the scalar kernel's body runs verbatim. The scalar kernels stay for the frozen legacy
entry points (``generate_*_warp``, ``apply_z_profile*``, the PR9 composition
``warp_pipeline._run_pipeline`` and the frozen PR11 gate reference). The CPU proofs are
tests/test_dynamic_knobs.py; this module proves the same claims on ``cuda:0``, where
FMA contraction and ``fminf`` / ``fmaxf`` could break them:

1. Each twin == its scalar kernel (the nine pairs of design section 3.1), launched on
   inputs a real generation produced (captured from the PR9 composition right before
   the scalar launch): uniform arrays at the default, interior values, the bounds,
   ``-0.0`` and NaN are bitwise equal to the scalar launch; out-of-range arrays equal
   the scalar launch at the bound; a per-env array equals, row by row, the scalar
   launch at each env's (clamped) value; the knob arrays are never written;
   ``_assemble_dyn_k`` with ``(-inf, inf)`` bounds equals ``_assemble_style_k``.
2. Per env == scalar, every knob of the five generators and of elevation uniform /
   random_walk / noise: ``set(field, [E] array)`` on a ``TrackGenerator`` then
   ``enqueue()`` equals, env by env and on every result field, separate Stage runs
   built with the scalar ``v_e``; each scalar Stage run equals the PR9 composition with
   the legacy spec (the SCALAR kernels) bitwise on every Track field, and the traces
   show the twin on the Stage side and the scalar kernel on the legacy side; every
   value moves the batch (non-vacuous); the live arrays read back unwritten.
3. Arrays outside the declared bounds (below ``lo``, ``-inf``, above ``hi``, ``+inf``)
   and ``-0.0`` are clamped AT USE: env ``e`` equals a run whose knob was ``set`` to the
   scalar bound (or ``-0.0``).
4. Gates end to end: the polar / checkpoint / voronoi gate generators, both orderings,
   flat and with noise / random_walk elevation, with every generator and elevation knob
   set per env: env ``e`` equals ``tests/_legacy_gate_pr11.run_reference`` (the frozen
   dd181f0 pipeline, scalar helpers) run with the scalar values of env ``e``, on every
   GateSequence field; the default batch equals the reference first (a pre-existing
   CUDA divergence is reported as such).
5. ``presets.track(randomize={...})`` on built-in generator knobs (each generator, inside
   each knob's effective range) and on elevation knobs (uniform ``z_min`` + ``z_max``,
   random_walk ``z_max_step``, noise ``z_noise_amplitude``, and a generator + elevation
   pipeline), recorded with ``generate(stream=s, advance=True)`` into ONE caller graph:
   every replay redraws every env inside ``[lo, hi]``, equal to the randomized eager
   twin (draws and Track) and to a NON-randomized eager twin whose knobs were ``set`` to
   the replay's drawn arrays (the draw reaches the stage in the same generation), and
   differs from the default twin (the draws matter). A masked advance recorded in the
   graph redraws the masked envs only and keeps every unmasked Track byte and draw.
6. The same for ``presets.gates(randomize={...})`` (polar, checkpoint, voronoi, and
   polar + ``elevation.z_noise_amplitude``).
7. A curriculum in the caller's graph: the caller's kernel writes
   ``params["generator"][knob]`` and ``params["elevation"]["z_max"]`` from a device
   schedule indexed by an in-graph step counter (values below ``lo`` / above ``hi``
   included), then ``generate(advance=True)``; every replay (Warp capture, and a
   registered ``torch.cuda.graph`` when torch has CUDA) equals the eager twin given the
   clamped values, the live arrays keep the unclamped values, and the batch differs
   from the default twin's.
8. The conformance kit on ``cuda:0`` runs ``params.per_env`` (not skipped) and passes
   for the five track generators, the three gate generators and the three dynamic
   elevations.

Method (as in tests/test_results_cuda.py and tests/test_params_cuda.py; the helpers are
:mod:`track_gen.testing.capture`'s): record on a NON-current side stream with the Warp
mempool disabled (an allocation inside the capture fails loudly) and
``wp.capture_launch`` replaced by a tripwire (a graph launch cannot be recorded into
another capture); poison the outputs (and the drawn param arrays) with FINITE,
dtype-aware sentinels before each replay (outputs are NaN-padded and
``assert_array_equal`` treats NaN == NaN, so a NaN poison would hide a missed write);
compare with eager twins (``enqueue()``, never a graph). Construction prepares the
default body; the masked body is warmed up with an all-zero mask (an exact no-op) and the
test's own kernels on scratch buffers, so no module loads while capturing. Every array is
allocated on ``cuda:0`` explicitly. The CPU lane only collects this module.
"""
from __future__ import annotations

import dataclasses
import importlib
import math

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from tests import _legacy_gate_pr11 as gate_ref  # noqa: E402
from track_gen import (  # noqa: E402
    GateGenConfig,
    GateGenerator,
    PerEnvSeededRNG,
    TrackGenConfig,
    TrackGenerator,
)
from tests._reference_generators import warp_pipeline  # noqa: E402
from track_gen._src.generator_binding import _StageGenerator  # noqa: E402
from track_gen.exec import LaunchContext  # noqa: E402
from track_gen.pipeline import presets  # noqa: E402
from track_gen.plugin import registry  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import check_gate_generator, check_generator, check_stage  # noqa: E402
from track_gen.testing.trace import trace_launches  # noqa: E402

DEV = "cuda:0"
E = 8
#: Per-env base seeds (the fingerprint seeds: int32 extremes included). At epoch 0 the
#: effective seed is the base seed, so the frozen references take them as they are.
SEEDS = np.array([0, 1, 7, 99991, 2147483647, 1000003, 123456789, 46341], np.int32)
#: The Track fields the PR9 composition writes (tests/test_pipeline_parity.py).
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                "valid", "count", "winding", "truncated")
#: The GateSequence fields of the frozen PR11 reference.
GATE_FIELDS = gate_ref.FINAL_FIELDS
_ZERO = np.zeros(E, np.int32)
_ONES = np.ones(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
_MODES = ("warp", "torch")           # caller capture: Warp ScopedCapture / torch.cuda.graph

# The 3D TrackGenConfig cases of tests/test_pipeline_parity.py (flat legacy fields).
_ZU = dict(z_profile="uniform", z_min=-0.2, z_max=0.3, z_valid_grade=0.5)
_ZW = dict(z_profile="random_walk", z_base=0.05, z_min=-0.2, z_max=0.3, z_max_step=0.4,
           z_valid_grade=0.8, z_control_points=7)
_ZN = dict(z_profile="noise", z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.2,
           z_noise_harmonics=4, z_valid_grade=0.6)
# The same profiles as presets.track(elevation=(name, params)) selections.
_ZU_P = {"z_min": -0.2, "z_max": 0.3}
_ZW_P = {"z_base": 0.05, "z_min": -0.2, "z_max": 0.3, "z_max_step": 0.4,
         "z_control_points": 7}
_ZN_P = {"z_base": 0.1, "z_min": -0.3, "z_max": 0.4, "z_noise_amplitude": 0.2,
         "z_noise_harmonics": 4}
# Gate noise (a wide symmetric window, so the amplitude is not clipped away).
_GZN_P = {"z_base": 0.1, "z_min": -0.5, "z_max": 1.5, "z_noise_amplitude": 0.5,
          "z_noise_harmonics": 4}

#: The dynamic set (design section 2.2): owner ``(kind, name)`` -> its dynamic fields.
DYNAMIC = {
    ("generator", "bezier"): ("rad", "handle_clamp_frac"),
    ("generator", "hull"): ("hull_displacement",),
    ("generator", "polar"): ("polar_radial_jitter", "polar_angular_jitter"),
    ("generator", "voronoi"): ("voronoi_radial_variation", "voronoi_angular_jitter"),
    ("generator", "checkpoint"): ("checkpoint_radius_min_frac", "checkpoint_angle_jitter",
                                  "checkpoint_turn_rate"),
    ("gate_generator", "polar"): ("polar_radial_jitter", "polar_angular_jitter"),
    ("gate_generator", "voronoi"): ("voronoi_radial_variation", "voronoi_angular_jitter"),
    ("gate_generator", "checkpoint"): ("checkpoint_radius_min_frac",
                                       "checkpoint_angle_jitter"),
    ("elevation", "uniform"): ("z_min", "z_max"),
    ("elevation", "random_walk"): ("z_min", "z_max", "z_max_step"),
    ("elevation", "noise"): ("z_min", "z_max", "z_noise_amplitude"),
}


# ---------------------------------------------------------------------------------------
# Test kernels (one Warp module: _load_test_kernels() loads it outside any capture)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _curriculum_k(counter: wp.array(dtype=wp.int32), steps: int,
                  knob_table: wp.array(dtype=wp.float32),
                  zmax_table: wp.array(dtype=wp.float32),
                  knob: wp.array(dtype=wp.float32), zmax: wp.array(dtype=wp.float32)):
    # The caller's in-graph curriculum: step k = counter[0] % steps selects row k of a
    # [steps, E] schedule for the generator knob and for the elevation's z_max.
    e = wp.tid()
    n = knob.shape[0]
    k = counter[0] % steps
    knob[e] = knob_table[k * n + e]
    zmax[e] = zmax_table[k * n + e]


def _load_test_kernels() -> None:
    """Load this module's kernels outside any capture (scratch buffers only)."""
    c = wp.zeros(1, dtype=wp.int32, device=DEV)
    tab = wp.zeros(2, dtype=wp.float32, device=DEV)
    a = wp.zeros(2, dtype=wp.float32, device=DEV)
    b = wp.zeros(2, dtype=wp.float32, device=DEV)
    wp.launch(_bump_k, 1, inputs=[c], device=DEV)
    wp.launch(_curriculum_k, 2, inputs=[c, 1, tab, tab], outputs=[a, b], device=DEV)
    wp.synchronize()


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True)
            for k, a in tg_capture.result_arrays(obj).items()}


def _rows(host: dict, envs) -> dict:
    return {k: v.reshape(E, -1)[envs] for k, v in host.items()}


def _stitch(per_env: list) -> dict:
    """The batch whose env ``e`` is env ``e`` of ``per_env[e]`` (host snapshots)."""
    return {name: np.stack([s[name].reshape(E, -1)[e] for e, s in enumerate(per_env)])
            .reshape(per_env[0][name].shape) for name in per_env[0]}


def _assert_bitwise(got: dict, ref: dict, label: str, fields=None) -> None:
    """``got[f]`` and ``ref[f]`` have the same dtype, shape and BYTES for every field
    (``fields``, default every field of ``ref``): signed zeros and NaN payloads count."""
    names = sorted(ref) if fields is None else list(fields)
    assert names, f"{label}: nothing to compare"
    missing = [n for n in names if n not in got]
    assert not missing, f"{label}: missing {missing}"
    bad = [n for n in names
           if got[n].dtype != ref[n].dtype or got[n].shape != ref[n].shape
           or got[n].tobytes() != ref[n].tobytes()]
    if bad:
        np.testing.assert_array_equal(got[bad[0]], ref[bad[0]],
                                      err_msg=f"{label}: {bad[0]} (of {bad})")
        raise AssertionError(f"{label}: {bad} differ bitwise (signed zeros / NaN payloads)")


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


def _per_env(values, shift: int = 0) -> np.ndarray:
    """``values`` cycled over the ``E`` envs (rolled by ``shift``), as float32."""
    v = np.asarray(values, np.float32)
    v = np.resize(v, E).astype(np.float32) if v.ndim else np.full(E, v, np.float32)
    return np.roll(v, shift).astype(np.float32)


def _f32key(v) -> bytes:
    """A dict key that tells ``-0.0`` from ``0.0``."""
    return np.float32(v).tobytes()


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


def _declared(kind: str, name: str, field: str) -> "tuple[float | None, float | None]":
    """The declared ``(lo, hi)`` of a dynamic field (its Params schema; ``None``: an
    unbounded side). Asserts the field is a dynamic float32 field."""
    prop = registry.describe(kind, name).params_schema["properties"][field]
    assert prop.get("x-dynamic") and prop.get("x-dtype") == "float32", \
        f"{kind}.{name}.{field} is not a dynamic float32 field: {prop}"
    return prop.get("minimum"), prop.get("maximum")


def _clamp(v, lo, hi) -> np.float32:
    """The twins' comparison clamp on the host: NaN and in-range values (signed zeros
    included) pass bit for bit."""
    x = np.float32(v)
    if lo is not None and x < np.float32(lo):
        x = np.float32(lo)
    if hi is not None and x > np.float32(hi):
        x = np.float32(hi)
    return x


def _rng(seeds=SEEDS) -> PerEnvSeededRNG:
    return PerEnvSeededRNG(seeds=wp.array(np.asarray(seeds, np.int32), dtype=wp.int32,
                                          device=DEV), num_envs=len(seeds), device=DEV)


def _track_cfg(kw: dict) -> TrackGenConfig:
    return TrackGenConfig(device=DEV, num_envs=E, **kw)


def _stage_track(kw: dict, sets=(), *, trace=None) -> "tuple[TrackGenerator, dict]":
    """``TrackGenerator(TrackGenConfig(**kw))`` at :data:`SEEDS` (the built-in Stage,
    hence the twins), each ``(stage id, field, value)`` of ``sets`` written with
    ``StageParams.set`` (a scalar, or an ``[E]`` array), then one eager ``enqueue()``
    at epoch 0 (traced into ``trace`` when given). Returns ``(gen, snapshot)``."""
    gen = TrackGenerator(_track_cfg(kw), _rng())
    keep = []
    for sid, field, value in sets:
        src = _dev(value) if isinstance(value, np.ndarray) else float(value)
        keep.append(src)
        gen.params[sid].set(field, src)
    wp.synchronize()
    if trace is None:
        out = gen.enqueue()
    else:
        with trace_launches() as t:
            out = gen.enqueue()
        trace.extend(r.kernel for r in t)
    wp.synchronize()
    assert gen.graph is None, "enqueue() must never self-graph"
    return gen, _snapshot(out)


def _legacy_track(kw: dict, *, trace=None) -> dict:
    """The PR9 composition (``warp_pipeline._run_pipeline`` -> ``_inflate_warp``) for
    ``TrackGenConfig(**kw)`` on fresh buffers at :data:`SEEDS`: the frozen legacy
    generator spec (``generate_*_warp``) and ``apply_z_profile*``, i.e. the SCALAR
    kernels. Returns the :data:`TRACK_FIELDS` arrays."""
    cfg = _track_cfg(kw)
    track, scratch = warp_pipeline._inflate_warp_alloc(cfg)
    spec = scratch.generator_spec
    assert not isinstance(spec, _StageGenerator) and callable(getattr(spec, "generate", None)), \
        f"the PR9 composition must run the legacy spec of {cfg.generator!r}, got {spec!r}"
    seed = wp.array(SEEDS, dtype=wp.int32, device=DEV)
    ctx = LaunchContext.for_call(DEV)
    if trace is None:
        warp_pipeline._run_pipeline(cfg, seed, out=track, scratch=scratch, ctx=ctx)
    else:
        with trace_launches() as t:
            warp_pipeline._run_pipeline(cfg, seed, out=track, scratch=scratch, ctx=ctx)
        trace.extend(r.kernel for r in t)
    wp.synchronize()
    return {f: np.array(getattr(track, f).numpy(), copy=True) for f in TRACK_FIELDS}


def _launched(names: list, kernel: str) -> bool:
    return any(n == kernel or n.endswith("." + kernel) for n in names)


# ---------------------------------------------------------------------------------------
# 1. Each twin == its scalar kernel, on generator-produced inputs
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Twin:
    module: str                       # track_gen._src.<module>
    scalar: str                       # the scalar kernel (frozen legacy path)
    twin: str                         # its array-reading twin (Stage path)
    config: dict                      # TrackGenConfig whose PR9 run launches `scalar`
    owner: tuple                      # (kind, name) declaring the knobs
    knobs: dict                       # scalar argument index -> dynamic field
    bounds_after: "int | None" = None  # _assemble_dyn_k: 4 bound scalars after this arg


_TWINS = {
    "assemble": _Twin("warp_generate", "_assemble_k", "_assemble_dyn_k",
                      dict(generator="bezier"), ("generator", "bezier"),
                      {6: "rad", 7: "handle_clamp_frac"}, bounds_after=7),
    "midpoint_displace": _Twin("warp_generate_hull", "_midpoint_displace_k",
                               "_midpoint_displace_dyn_k", dict(generator="hull"),
                               ("generator", "hull"), {4: "hull_displacement"}),
    "polar_controls": _Twin("warp_generate_polar", "_polar_controls_k",
                            "_polar_controls_dyn_k", dict(generator="polar"),
                            ("generator", "polar"),
                            {2: "polar_radial_jitter", 3: "polar_angular_jitter"}),
    "sample_checkpoints": _Twin("warp_generate_checkpoint", "_sample_checkpoints_k",
                                "_sample_checkpoints_dyn_k", dict(generator="checkpoint"),
                                ("generator", "checkpoint"),
                                {3: "checkpoint_radius_min_frac",
                                 4: "checkpoint_angle_jitter"}),
    "steer": _Twin("warp_generate_checkpoint", "_steer_k", "_steer_dyn_k",
                   dict(generator="checkpoint"), ("generator", "checkpoint"),
                   {4: "checkpoint_turn_rate"}),
    "select_anchor_sites": _Twin("warp_generate_voronoi", "_select_anchor_sites_k",
                                 "_select_anchor_sites_dyn_k", dict(generator="voronoi"),
                                 ("generator", "voronoi"),
                                 {6: "voronoi_radial_variation",
                                  7: "voronoi_angular_jitter"}),
    "z_uniform": _Twin("warp_zprofile", "_z_uniform_k", "_z_uniform_dyn_k",
                       dict(generator="bezier", **_ZU), ("elevation", "uniform"),
                       {3: "z_min", 4: "z_max"}),
    "z_walk": _Twin("warp_zprofile", "_z_walk_k", "_z_walk_dyn_k",
                    dict(generator="bezier", **_ZW), ("elevation", "random_walk"),
                    {6: "z_min", 7: "z_max", 8: "z_max_step"}),
    "z_noise": _Twin("warp_zprofile", "_z_noise_k", "_z_noise_dyn_k",
                     dict(generator="bezier", **_ZN), ("elevation", "noise"),
                     {6: "z_min", 7: "z_max", 8: "z_noise_amplitude"}),
}


def _capture_scalar_launch(monkeypatch, config: dict, kernel) -> "tuple[object, list]":
    """Run the PR9 composition for ``TrackGenConfig(**config)`` and return ``(dim,
    args)`` of the FIRST launch of ``kernel`` (inputs then outputs), every array
    argument cloned right before that launch: the inputs a real generation hands the
    scalar kernel."""
    real = LaunchContext.launch
    got: list = []

    def spy(self, k, dim, inputs=(), outputs=(), **kw):
        if k is kernel and not got:
            wp.synchronize_device(DEV)              # the work before it has completed
            args = [wp.clone(a) if isinstance(a, wp.array) else a
                    for a in (*inputs, *outputs)]
            wp.synchronize_device(DEV)
            got.append((dim, args))
        return real(self, k, dim, inputs, outputs, **kw)

    with monkeypatch.context() as m:
        m.setattr(LaunchContext, "launch", spy)
        _legacy_track(config)
    assert got, f"the PR9 composition of {config} never launched {kernel.key}"
    return got[0]


def _run(kernel, dim, args) -> list:
    """Launch ``kernel`` on fresh copies of ``args`` (arrays cloned); return the host
    copy of every argument after the launch (``None`` for a scalar)."""
    fresh = [wp.clone(a) if isinstance(a, wp.array) else a for a in args]
    wp.synchronize()
    wp.launch(kernel, dim=dim, inputs=fresh, device=DEV)
    wp.synchronize()
    return [np.array(a.numpy(), copy=True) if isinstance(a, wp.array) else None
            for a in fresh]


class _TwinHarness:
    """The captured scalar launch of one pair, and launches of both kernels on it."""

    def __init__(self, pair: _Twin, monkeypatch) -> None:
        mod = importlib.import_module(f"track_gen._src.{pair.module}")
        self.pair = pair
        self.scalar_k = getattr(importlib.import_module(f"tests._reference_generators.{pair.module}"), pair.scalar)
        self.twin_k = getattr(mod, pair.twin, None)
        assert self.twin_k is not None, f"track_gen._src.{pair.module}.{pair.twin} missing"
        self.dim, self.args = _capture_scalar_launch(monkeypatch, pair.config,
                                                     self.scalar_k)
        self.bounds = {i: _declared(*pair.owner, f) for i, f in pair.knobs.items()}
        self.arrays = [i for i, a in enumerate(self.args) if isinstance(a, wp.array)]
        n_extra = 4 if pair.bounds_after is not None else 0
        labels = [(a.label, a.type) for a in self.twin_k.adj.args]
        assert len(labels) == len(self.args) + n_extra, \
            f"{pair.twin}{labels}: expected {len(self.args) + n_extra} arguments"
        for i in pair.knobs:
            j = self._twin_index(i)
            assert wp.types.is_array(labels[j][1]), \
                f"{pair.twin} argument {j} ({labels[j][0]}) must be the [E] float32 array"
        for i, a in enumerate(self.args):
            if i not in pair.knobs and not isinstance(a, wp.array):
                assert not wp.types.is_array(labels[self._twin_index(i)][1]), (pair, i)

    def _twin_index(self, i: int) -> int:
        after = self.pair.bounds_after
        return i if after is None or i <= after else i + 4

    def default(self, i: int) -> float:
        return float(self.args[i])

    def scalar(self, values: dict) -> list:
        """The scalar kernel with knob ``i`` at the float ``values[i]`` (the captured
        value for a knob not in ``values``)."""
        args = list(self.args)
        for i, v in values.items():
            args[i] = float(v)
        out = _run(self.scalar_k, self.dim, args)
        return [out[i] for i in self.arrays]

    def twin(self, arrays: dict, bounds=None) -> list:
        """The twin with knob ``i`` read from the ``[E]`` float32 array ``arrays[i]``
        (a knob not in ``arrays``: its captured value in every env); ``bounds`` are the
        bezier twin's ``(rad_lo, rad_hi, clamp_lo, clamp_hi)`` (default: declared)."""
        knob_arrays = {}
        for i in self.pair.knobs:
            v = arrays.get(i, np.full(E, self.default(i), np.float32))
            knob_arrays[i] = _dev(np.asarray(v, np.float32))
        args = [knob_arrays.get(i, a) for i, a in enumerate(self.args)]
        if self.pair.bounds_after is not None:
            if bounds is None:
                bounds = []
                for i in sorted(self.pair.knobs):
                    lo, hi = self.bounds[i]
                    bounds += [-math.inf if lo is None else float(lo),
                               math.inf if hi is None else float(hi)]
            at = self.pair.bounds_after + 1
            args = args[:at] + [float(b) for b in bounds] + args[at:]
        out = _run(self.twin_k, self.dim, args)
        for i, arr in knob_arrays.items():
            assert out[self._twin_index(i)].tobytes() == arr.numpy().tobytes(), \
                f"{self.pair.twin} wrote its knob array {self.pair.knobs[i]}"
        return [out[self._twin_index(i)] for i in self.arrays]

    def assert_same(self, got: list, want: list, label: str) -> None:
        for i, g, w in zip(self.arrays, got, want):
            _assert_bitwise({"a": g}, {"a": w}, f"{label}: argument {i}")

    def assert_rows(self, got: list, want: list, e: int, label: str) -> None:
        for i, g, w in zip(self.arrays, got, want):
            if g.shape[0] % E == 0:
                g, w = g.reshape(E, -1)[e], w.reshape(E, -1)[e]
            _assert_bitwise({"a": g}, {"a": w}, f"{label}: env {e} argument {i}")


def _in_range_values(lo, hi, d: float) -> list:
    """The default, two interior values, the declared bounds, ``-0.0`` (when 0 is the
    lower bound or there is none) and NaN."""
    if lo is not None and hi is not None:
        span = float(hi) - float(lo)
        vals = [lo + 0.2 * span, lo + 0.7 * span, lo, hi]
    elif lo is not None:
        vals = [lo + 0.15, lo + 0.9, lo, lo + 1.0e9]
    elif hi is not None:
        vals = [hi - 0.15, hi - 0.9, hi, hi - 1.0e9]
    else:
        vals = [d - 0.35, d + 0.45, -50.0, 50.0]
    if lo is None or float(lo) == 0.0:
        vals.append(-0.0)
    return [d] + [float(v) for v in vals] + [math.nan]


def _out_of_range_values(lo, hi) -> list:
    """Values outside each declared side, the two sides interleaved."""
    below = [] if lo is None else [float(lo) - 0.3, -math.inf, float(lo) - 1.0e-3]
    above = [] if hi is None else [float(hi) + 0.4, math.inf, float(hi) + 1.0e-3]
    if below and above:
        return [v for pair in zip(below, above) for v in pair]
    return below + above


@pytest.mark.parametrize("pair", list(_TWINS))
def test_twin_kernel_equals_its_scalar_kernel_on_generator_inputs(pair, monkeypatch):
    """Each twin of design section 3.1 against its scalar kernel, on the inputs the PR9
    composition handed the scalar kernel in a real generation (captured just before
    the launch): (a) a uniform array at the default, two interior values, the declared
    bounds, ``-0.0`` and NaN == the scalar launch, bitwise; (b) out-of-range uniform
    arrays (below ``lo``, ``-inf``, above ``hi``, ``+inf``) == the scalar launch at the
    bound; (c) a per-env array mixing interior and out-of-range values == row by row
    the scalar launch at each env's clamped value; every knob array reads back
    unwritten. For ``_assemble_dyn_k`` also (d) ``(-inf, inf)`` bounds (a style-drawn
    knob) == ``_assemble_style_k`` on the same out-of-range per-env draws."""
    h = _TwinHarness(_TWINS[pair], monkeypatch)
    knobs = h.pair.knobs
    # (a) + (b): one knob at a time, the others at their captured value.
    for i, field in knobs.items():
        lo, hi = h.bounds[i]
        for v in _in_range_values(lo, hi, h.default(i)):
            label = f"{pair}: {field} = {v!r} (uniform)"
            h.assert_same(h.twin({i: np.full(E, v, np.float32)}), h.scalar({i: v}), label)
        for v in _out_of_range_values(lo, hi):
            label = f"{pair}: {field} = {v!r} (out of range) vs the bound"
            h.assert_same(h.twin({i: np.full(E, v, np.float32)}),
                          h.scalar({i: float(_clamp(v, lo, hi))}), label)
    # (c): every knob per env at once (each its own permutation of the envs): the
    # default, two interior values, -0.0 (lower bound 0 or none), NaN, then out-of-range
    # values below and above.
    per_env = {}
    for k, i in enumerate(sorted(knobs)):
        lo, hi = h.bounds[i]
        inside = _in_range_values(lo, hi, h.default(i))
        signed_zero = [-0.0] if lo is None or float(lo) == 0.0 else []
        vals = inside[:3] + signed_zero + [math.nan] + _out_of_range_values(lo, hi)
        per_env[i] = _per_env(vals[:E], shift=k)
    got = h.twin(per_env)
    for e in range(E):
        at = {i: float(_clamp(v[e], *h.bounds[i])) for i, v in per_env.items()}
        h.assert_rows(got, h.scalar(at), e, f"{pair}: per env {at}")



# ---------------------------------------------------------------------------------------
# 2. Per env == scalar, every knob; the Stage (twins) == the PR9 composition (scalars)
# ---------------------------------------------------------------------------------------

#: case -> (TrackGenConfig kwargs, stage id, field, four in-range values, (scalar
#: kernel, twin kernel)). Values lie inside each knob's EFFECTIVE range (rad below the
#: default handle_clamp_frac 0.4, voronoi_angular_jitter in its useful [0, 0.5]) and keep
#: the legacy validate() rules (z_min <= z_max, checkpoint_radius_min_frac < 1).
_KNOBS = {
    "bezier.rad": (dict(generator="bezier"), "generator", "rad",
                   (0.1, 0.2, 0.3, 0.35), ("_assemble_k", "_assemble_dyn_k")),
    "bezier.handle_clamp_frac": (dict(generator="bezier"), "generator",
                                 "handle_clamp_frac", (0.1, 0.25, 0.6, 1.5),
                                 ("_assemble_k", "_assemble_dyn_k")),
    "hull.hull_displacement": (dict(generator="hull"), "generator", "hull_displacement",
                               (0.05, 0.3, 0.5, 0.8),
                               ("_midpoint_displace_k", "_midpoint_displace_dyn_k")),
    "polar.polar_radial_jitter": (dict(generator="polar"), "generator",
                                  "polar_radial_jitter", (0.1, 0.3, 0.45, 0.8),
                                  ("_polar_controls_k", "_polar_controls_dyn_k")),
    "polar.polar_angular_jitter": (dict(generator="polar"), "generator",
                                   "polar_angular_jitter", (0.05, 0.15, 0.4, 0.45),
                                   ("_polar_controls_k", "_polar_controls_dyn_k")),
    "voronoi.voronoi_radial_variation": (dict(generator="voronoi"), "generator",
                                         "voronoi_radial_variation", (0.1, 0.3, 0.5, 0.8),
                                         ("_select_anchor_sites_k",
                                          "_select_anchor_sites_dyn_k")),
    "voronoi.voronoi_angular_jitter": (dict(generator="voronoi"), "generator",
                                       "voronoi_angular_jitter", (0.0, 0.15, 0.3, 0.45),
                                       ("_select_anchor_sites_k",
                                        "_select_anchor_sites_dyn_k")),
    "checkpoint.checkpoint_radius_min_frac": (dict(generator="checkpoint"), "generator",
                                              "checkpoint_radius_min_frac",
                                              (0.1, 0.2, 0.5, 0.7),
                                              ("_sample_checkpoints_k",
                                               "_sample_checkpoints_dyn_k")),
    "checkpoint.checkpoint_angle_jitter": (dict(generator="checkpoint"), "generator",
                                           "checkpoint_angle_jitter", (0.1, 0.3, 0.75, 0.9),
                                           ("_sample_checkpoints_k",
                                            "_sample_checkpoints_dyn_k")),
    "checkpoint.checkpoint_turn_rate": (dict(generator="checkpoint"), "generator",
                                        "checkpoint_turn_rate", (0.2, 0.3, 0.6, 0.9),
                                        ("_steer_k", "_steer_dyn_k")),
    "uniform.z_min": (dict(generator="bezier", **_ZU), "elevation", "z_min",
                      (-0.5, -0.35, -0.1, 0.0), ("_z_uniform_k", "_z_uniform_dyn_k")),
    "uniform.z_max": (dict(generator="bezier", **_ZU), "elevation", "z_max",
                      (0.1, 0.2, 0.5, 0.8), ("_z_uniform_k", "_z_uniform_dyn_k")),
    "random_walk.z_min": (dict(generator="bezier", **_ZW), "elevation", "z_min",
                          (-0.15, -0.1, -0.05, 0.0), ("_z_walk_k", "_z_walk_dyn_k")),
    "random_walk.z_max": (dict(generator="bezier", **_ZW), "elevation", "z_max",
                          (0.1, 0.15, 0.2, 0.25), ("_z_walk_k", "_z_walk_dyn_k")),
    "random_walk.z_max_step": (dict(generator="bezier", **_ZW), "elevation", "z_max_step",
                               (0.1, 0.2, 0.6, 1.0), ("_z_walk_k", "_z_walk_dyn_k")),
    "noise.z_min": (dict(generator="bezier", **_ZN), "elevation", "z_min",
                    (0.05, 0.1, 0.12, 0.15), ("_z_noise_k", "_z_noise_dyn_k")),
    "noise.z_max": (dict(generator="bezier", **_ZN), "elevation", "z_max",
                    (0.0, 0.05, 0.1, 0.15), ("_z_noise_k", "_z_noise_dyn_k")),
    "noise.z_noise_amplitude": (dict(generator="bezier", **_ZN), "elevation",
                                "z_noise_amplitude", (0.05, 0.1, 0.3, 0.5),
                                ("_z_noise_k", "_z_noise_dyn_k")),
}


@pytest.mark.parametrize("case", list(_KNOBS))
def test_per_env_knob_equals_scalar_runs_env_by_env(case):
    """``gen.params[stage].set(field, [E] array)`` then ``enqueue()``: env ``e`` equals
    env ``e`` of a separate Stage run built with the scalar ``v_e``, on every result
    field, bitwise; the live array reads back unwritten. Each scalar Stage run equals
    the PR9 composition with the legacy spec (the SCALAR kernels) on every Track field,
    bitwise (twin == scalar kernel end to end on CUDA, after resample, relax, frames,
    offset, lift and validity); the Stage launched the twin and the legacy path the
    scalar kernel. Non-vacuity: every value changes the batch."""
    kw, sid, field, values, (scalar_k, twin_k) = _KNOBS[case]
    owner = ("generator", kw["generator"]) if sid == "generator" else \
        ("elevation", kw["z_profile"])
    assert field in DYNAMIC[owner], (owner, field)
    _declared(*owner, field)
    default_kw = dict(kw)
    default_v = float(getattr(_track_cfg(default_kw), field))
    _, default = _stage_track(default_kw)

    refs = {}
    for k, v in enumerate(values):
        vkw = dict(kw, **{field: float(v)})
        stage_trace: list = []
        legacy_trace: list = []
        _, refs[_f32key(v)] = _stage_track(vkw, trace=stage_trace if k == 0 else None)
        legacy = _legacy_track(vkw, trace=legacy_trace if k == 0 else None)
        _assert_bitwise(refs[_f32key(v)], legacy,
                        f"{case}={v}: the Stage (twin) vs the PR9 composition (scalar)",
                        fields=TRACK_FIELDS)
        if k == 0:
            assert _launched(stage_trace, twin_k) and not _launched(stage_trace, scalar_k), \
                f"{case}: the Stage must launch {twin_k}, not {scalar_k}"
            assert _launched(legacy_trace, scalar_k) and not _launched(legacy_trace, twin_k), \
                f"{case}: the PR9 composition must launch {scalar_k}, not {twin_k}"
        if float(np.float32(v)) != float(np.float32(default_v)):
            assert _differs(refs[_f32key(v)], default), \
                f"{case}={v} does not change the batch (the comparison would be vacuous)"

    per_env = _per_env(values)
    gen, got = _stage_track(kw, [(sid, field, per_env)])
    np.testing.assert_array_equal(gen.params[sid][field].numpy().view(np.uint32),
                                  per_env.view(np.uint32),
                                  err_msg=f"{case}: the stage wrote its live array")
    _assert_bitwise(got, _stitch([refs[_f32key(v)] for v in per_env]),
                    f"{case}: per-env {per_env.tolist()} vs the scalar runs env by env")


# ---------------------------------------------------------------------------------------
# 3. Out-of-range arrays are clamped at use
# ---------------------------------------------------------------------------------------

#: The bounded knobs: case -> (TrackGenConfig kwargs, stage id, field, owner).
_BOUNDED = {
    "bezier.rad": (dict(generator="bezier"), "generator", "rad"),
    "bezier.handle_clamp_frac": (dict(generator="bezier"), "generator", "handle_clamp_frac"),
    "hull.hull_displacement": (dict(generator="hull"), "generator", "hull_displacement"),
    "polar.polar_radial_jitter": (dict(generator="polar"), "generator",
                                  "polar_radial_jitter"),
    "polar.polar_angular_jitter": (dict(generator="polar"), "generator",
                                   "polar_angular_jitter"),
    "voronoi.voronoi_radial_variation": (dict(generator="voronoi"), "generator",
                                         "voronoi_radial_variation"),
    "voronoi.voronoi_angular_jitter": (dict(generator="voronoi"), "generator",
                                       "voronoi_angular_jitter"),
    "checkpoint.checkpoint_radius_min_frac": (dict(generator="checkpoint"), "generator",
                                              "checkpoint_radius_min_frac"),
    "checkpoint.checkpoint_angle_jitter": (dict(generator="checkpoint"), "generator",
                                           "checkpoint_angle_jitter"),
    "checkpoint.checkpoint_turn_rate": (dict(generator="checkpoint"), "generator",
                                        "checkpoint_turn_rate"),
    "random_walk.z_max_step": (dict(generator="bezier", **_ZW), "elevation", "z_max_step"),
    "noise.z_noise_amplitude": (dict(generator="bezier", **_ZN), "elevation",
                                "z_noise_amplitude"),
}


@pytest.mark.parametrize("case", list(_BOUNDED))
def test_out_of_range_per_env_values_are_clamped_at_use(case):
    """A per-env array holding values below ``lo``, ``-inf``, above ``hi``, ``+inf``
    (each declared side), ``-0.0`` and the bounds themselves: ``set`` accepts the array
    (arrays are clamped at use, scalars are checked), the live array keeps the values
    as written, and env ``e`` equals a run whose knob was ``set`` to the scalar
    ``clamp(v_e)`` (``-0.0`` stays ``-0.0``: the comparison clamp passes it bit for
    bit), bitwise on every result field; the batch differs from the default one."""
    kw, sid, field = _BOUNDED[case]
    owner = ("generator", kw["generator"]) if sid == "generator" else \
        ("elevation", kw["z_profile"])
    lo, hi = _declared(*owner, field)
    assert lo is not None or hi is not None, f"{case} declares no bound"
    vals = []
    if lo is not None:
        vals += [float(lo) - 0.25, -math.inf, float(lo)]
    if hi is not None:
        vals += [float(hi) + 0.5, math.inf, float(hi)]
    if lo is not None and float(lo) == 0.0:
        vals.append(-0.0)
    per_env = _per_env(vals)
    gen, got = _stage_track(kw, [(sid, field, per_env)])
    np.testing.assert_array_equal(gen.params[sid][field].numpy().view(np.uint32),
                                  per_env.view(np.uint32),
                                  err_msg=f"{case}: the live array must keep what was set")
    refs = {}
    for v in per_env:
        c = _clamp(v, lo, hi)
        if _f32key(c) not in refs:
            refs[_f32key(c)] = _stage_track(kw, [(sid, field, float(c))])[1]
    _assert_bitwise(got, _stitch([refs[_f32key(_clamp(v, lo, hi))] for v in per_env]),
                    f"{case}: per-env {per_env.tolist()} vs the scalar bound runs")
    assert _differs(got, _stage_track(kw)[1]), \
        f"{case}: the clamped values do not change the batch (vacuous)"


# ---------------------------------------------------------------------------------------
# 4. Gates end to end vs the frozen PR11 reference (scalar helpers)
# ---------------------------------------------------------------------------------------

#: Gate generator -> its dynamic knobs and four in-range values each.
_GATE_KNOBS = {
    "polar": {"polar_radial_jitter": (0.1, 0.3, 0.45, 0.8),
              "polar_angular_jitter": (0.05, 0.15, 0.4, 0.45)},
    "checkpoint": {"checkpoint_radius_min_frac": (0.1, 0.2, 0.5, 0.7),
                   "checkpoint_angle_jitter": (0.1, 0.3, 0.75, 0.9)},
    "voronoi": {"voronoi_radial_variation": (0.1, 0.3, 0.5, 0.8),
                "voronoi_angular_jitter": (0.0, 0.15, 0.3, 0.45)},
}
#: Gate elevation -> (GateGenConfig kwargs, its dynamic knobs set per env).
_GATE_ELEVATIONS = {
    "flat": ({}, {}),
    "noise": (dict(z_profile="noise", z_base=0.1, z_min=-0.5, z_max=1.5,
                   z_noise_amplitude=0.5, z_noise_harmonics=4, gate_align="full_tangent",
                   gate_solve_iters=0),
              {"z_noise_amplitude": (0.1, 0.3, 0.8, 1.2), "z_max": (0.1, 0.15, 0.2, 0.3)}),
    "random_walk": (dict(z_profile="random_walk", z_max=1.5, z_max_step=0.3),
                    {"z_max_step": (0.1, 0.2, 0.5, 0.8), "z_min": (-0.5, -0.2, 0.1, 0.3)}),
}
_GATE_CASES = [(g, o, z) for g in _GATE_KNOBS for o in ("ccw", "raw")
               for z in _GATE_ELEVATIONS]


def _gate_cfg(kw: dict) -> GateGenConfig:
    return GateGenConfig(num_envs=E, device=DEV, **kw)


def _gate_reference(kw: dict) -> dict:
    """``tests/_legacy_gate_pr11.run_reference`` (the frozen dd181f0 pipeline: the
    scalar ``sample_polar_controls`` / ``sample_checkpoint_rings`` /
    ``select_voronoi_anchors`` and ``apply_z_profile``) at :data:`SEEDS`."""
    taps = gate_ref.run_reference(_gate_cfg(kw), SEEDS)
    wp.synchronize()
    return {f: np.asarray(taps[f]) for f in GATE_FIELDS}


@pytest.mark.parametrize("gen_name,ordering,elev", _GATE_CASES,
                         ids=[f"{g}-{o}-{z}" for g, o, z in _GATE_CASES])
def test_gate_generator_per_env_knobs_match_the_frozen_pr11_reference(gen_name, ordering,
                                                                      elev):
    """A ``GateGenerator(GateGenConfig)`` on ``cuda:0``: at the default knobs its batch
    equals the frozen PR11 reference (precondition, reported separately); then every
    dynamic knob of the gate generator (and of the elevation) is set per env (each knob
    its own permutation of four in-range values) and ``enqueue()``: env ``e`` equals
    env ``e`` of the reference run with ``e``'s values as scalars, bitwise on every
    GateSequence field; every value combination moves the batch."""
    zkw, zknobs = _GATE_ELEVATIONS[elev]
    base = dict(generator=gen_name, gate_ordering=ordering, **zkw)
    gkn = _GATE_KNOBS[gen_name]
    for f in gkn:
        assert f in DYNAMIC[("gate_generator", gen_name)], f
        _declared("gate_generator", gen_name, f)
    if zknobs:
        for f in zknobs:
            assert f in DYNAMIC[("elevation", zkw["z_profile"])], f
            _declared("elevation", zkw["z_profile"], f)
    label = f"{gen_name}/{ordering}/{elev}"

    gen = GateGenerator(_gate_cfg(base), _rng())
    out = gen.enqueue()
    wp.synchronize()
    default_ref = _gate_reference(base)
    _assert_bitwise(_snapshot(out), default_ref,
                    f"{label}: defaults: the gate Program vs the frozen PR11 reference "
                    f"(pre-existing CUDA divergence, not a PR13c knob)", fields=GATE_FIELDS)

    sets = [("generator", f, _per_env(v, shift=k)) for k, (f, v) in enumerate(gkn.items())]
    sets += [("elevation", f, _per_env(v, shift=k + 1))
             for k, (f, v) in enumerate(zknobs.items())]
    keep = []
    for sid, f, arr in sets:
        src = _dev(arr)
        keep.append(src)
        gen.params[sid].set(f, src)
    wp.synchronize()
    out = gen.enqueue()
    wp.synchronize()
    got = _snapshot(out)
    for sid, f, arr in sets:
        np.testing.assert_array_equal(gen.params[sid][f].numpy().view(np.uint32),
                                      arr.view(np.uint32),
                                      err_msg=f"{label}: {sid}.{f} live array was written")

    refs: dict = {}
    per_env = []
    for e in range(E):
        combo = tuple((f, float(arr[e])) for _, f, arr in sets)
        if combo not in refs:
            refs[combo] = _gate_reference(dict(base, **dict(combo)))
            assert _differs(refs[combo], default_ref), \
                f"{label}: {combo} does not change the batch (vacuous)"
        per_env.append(refs[combo])
    _assert_bitwise(got, _stitch(per_env),
                    f"{label}: per-env knobs vs the PR11 reference env by env",
                    fields=GATE_FIELDS)


# ---------------------------------------------------------------------------------------
# 5. / 6. randomize.uniform on built-in knobs in a caller graph
# ---------------------------------------------------------------------------------------

#: presets.track cases: (preset kwargs, randomize ranges inside each knob's effective
#: range). Crossing rules hold (hi(z_min) <= lo(z_max)), rad stays <= the clamp unless
#: the clamp is drawn too.
_TRACK_RANDOMIZE = {
    "bezier.rad": (dict(generator="bezier"), {"generator.rad": (0.1, 0.4)}),
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
    "elevation.uniform": (dict(generator="bezier", elevation=("uniform", _ZU_P)),
                          {"elevation.z_min": (-0.5, -0.2), "elevation.z_max": (0.3, 0.8)}),
    "elevation.random_walk": (dict(generator="bezier", elevation=("random_walk", _ZW_P)),
                              {"elevation.z_max_step": (0.1, 0.8)}),
    "elevation.noise": (dict(generator="bezier", elevation=("noise", _ZN_P)),
                        {"elevation.z_noise_amplitude": (0.05, 0.5)}),
    "polar+elevation.z_max": (dict(generator="polar", elevation=("uniform", _ZU_P)),
                              {"generator.polar_radial_jitter": (0.2, 0.8),
                               "elevation.z_max": (0.3, 0.8)}),
}
#: presets.gates cases.
_GATE_RANDOMIZE = {
    "polar": (dict(generator="polar"), {"generator.polar_radial_jitter": (0.2, 0.8),
                                        "generator.polar_angular_jitter": (0.05, 0.45)}),
    "checkpoint": (dict(generator="checkpoint"),
                   {"generator.checkpoint_radius_min_frac": (0.1, 0.6),
                    "generator.checkpoint_angle_jitter": (0.2, 0.9)}),
    "voronoi": (dict(generator="voronoi"),
                {"generator.voronoi_radial_variation": (0.2, 0.85),
                 "generator.voronoi_angular_jitter": (0.0, 0.3)}),
    "polar+elevation.noise": (dict(generator="polar", elevation=("noise", _GZN_P),
                                   gate_align="full_tangent", gate_solve_iters=0),
                              {"generator.polar_radial_jitter": (0.2, 0.8),
                               "elevation.z_noise_amplitude": (0.1, 0.6)}),
}


def _track_spec(kw: dict, randomize):
    return presets.track(num_envs=E, device=DEV, randomize=randomize, **kw)


def _gate_spec(kw: dict, randomize):
    return presets.gates(num_envs=E, device=DEV, randomize=randomize, **kw)


def _keys(randomize: dict) -> list:
    """``[(stage id, field, lo, hi)]`` of a ``randomize=`` mapping."""
    out = []
    for key, (lo, hi) in randomize.items():
        sid, field = key.split(".", 1)
        out.append((sid, field, lo, hi))
    return out


def _drawn(gen, keys) -> dict:
    return {(sid, f): np.array(gen.params[sid][f].numpy(), copy=True)
            for sid, f, _, _ in keys}


def _check_spec(spec, randomize: dict, label: str) -> None:
    ids = list(spec.ids)
    for key in randomize:
        rid, target = f"randomize.{key}", key.split(".", 1)[0]
        assert rid in ids and target in ids and ids.index(rid) < ids.index(target), \
            f"{label}: {rid} must run before {target}: {ids}"


def _set_drawn(twin, gen, keys) -> None:
    """The non-randomized twin's knobs ``set`` to ``gen``'s drawn arrays (device
    copies on the twin's stream)."""
    for sid, f, _, _ in keys:
        twin.params[sid].set(f, gen.params[sid][f])
    wp.synchronize()


def _assert_randomized_replays(cls, kw: dict, randomize: dict, spec_of, label: str,
                               replays: int = 3) -> None:
    """ONE caller graph holding ``generate(stream=s, advance=True)`` of the randomized
    pipeline; every replay into a poisoned result AND poisoned drawn arrays: draws in
    ``[lo, hi]`` (float32), distinct across envs, redrawn for every env, equal to the
    randomized eager twin's, and the batch equals that twin's, the non-randomized twin
    ``set`` to the drawn arrays, and differs from the default twin's."""
    spec = spec_of(kw, randomize)
    _check_spec(spec, randomize, label)
    keys = _keys(randomize)
    gen = cls(spec, seeds=11)
    twin = cls(spec, seeds=11)
    plain = cls(spec_of(kw, None), seeds=11)
    default = cls(spec_of(kw, None), seeds=11)
    out = gen.program.result
    arrays = [gen.params[sid][f] for sid, f, _, _ in keys]
    prev = _drawn(gen, keys)
    for k, v in prev.items():
        np.testing.assert_array_equal(v, _drawn(twin, keys)[k],
                                      err_msg=f"{label}: construction draws {k}")

    s = _side_stream()
    with _recording(s) as cap:
        assert gen.generate(stream=s, advance=True) is out
    assert gen.graph is None, f"{label}: recording into the caller's graph self-captured"
    for k, v in _drawn(gen, keys).items():
        np.testing.assert_array_equal(v, prev[k], err_msg=f"{label}: the recording drew")

    for r in range(replays):
        step = f"{label}: replay {r}"
        tg_capture.poison_all(out)
        for a in arrays:
            tg_capture.poison(a)
        wp.synchronize()
        _replay(cap.graph, s)
        twin.enqueue(advance=True)
        _set_drawn(plain, gen, keys)
        plain.enqueue(advance=True)
        default.enqueue(advance=True)
        wp.synchronize()
        d = _drawn(gen, keys)
        dt = _drawn(twin, keys)
        for sid, f, lo, hi in keys:
            v = d[(sid, f)]
            assert np.all((v >= np.float32(lo)) & (v <= np.float32(hi))), \
                f"{step}: {sid}.{f} draws {v} outside [{lo}, {hi}]"
            assert len(np.unique(v)) == E, f"{step}: {sid}.{f} draws not distinct: {v}"
            assert np.all(v != prev[(sid, f)]), f"{step}: {sid}.{f} not redrawn"
            np.testing.assert_array_equal(v, dt[(sid, f)],
                                          err_msg=f"{step}: {sid}.{f} draws vs the twin")
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result), f"{step}: vs the randomized twin")
        _assert_host(got, _snapshot(plain.program.result),
                     f"{step}: vs the twin set to the drawn values")
        assert _differs(got, _snapshot(default.program.result)), \
            f"{step}: the draws do not change the batch (vacuous)"
        prev = d


@pytest.mark.parametrize("case", list(_TRACK_RANDOMIZE))
def test_track_randomize_on_builtin_knobs_in_a_caller_graph(case):
    """``presets.track(randomize={...})`` on built-in generator / elevation knobs,
    recorded into ONE caller graph: every replay == the eager twins (see
    :func:`_assert_randomized_replays`)."""
    kw, randomize = _TRACK_RANDOMIZE[case]
    _assert_randomized_replays(TrackGenerator, kw, randomize, _track_spec, case)


@pytest.mark.parametrize("case", list(_GATE_RANDOMIZE))
def test_gate_randomize_on_builtin_knobs_in_a_caller_graph(case):
    """``presets.gates(randomize={...})`` on the gate generators' ring knobs (and a gate
    elevation), recorded into ONE caller graph: every replay == the eager twins."""
    kw, randomize = _GATE_RANDOMIZE[case]
    _assert_randomized_replays(GateGenerator, kw, randomize, _gate_spec, case)


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


def test_track_randomize_masked_replay_in_a_caller_graph():
    """ONE caller graph holds ``[copy mask_src -> regen_mask] -> generate(advance=True,
    mask=regen_mask)`` of a polar pipeline drawing ``generator.polar_radial_jitter`` and
    ``elevation.z_max``. Each replay into a poisoned Track: masked envs get new draws
    of both knobs, equal to the randomized twin's, and the twin's rows (and the rows of
    a non-randomized twin set to the draws); unmasked envs redraw their previous values
    bit for bit and keep every Track byte (the poison)."""
    kw, randomize = _TRACK_RANDOMIZE["polar+elevation.z_max"]
    keys = _keys(randomize)
    spec = _track_spec(kw, randomize)
    gen = TrackGenerator(spec, seeds=11)
    twin = TrackGenerator(spec, seeds=11)
    plain = TrackGenerator(_track_spec(kw, None), seeds=11)
    out = gen.program.result
    m, tm, pm = gen.inputs["regen_mask"], twin.inputs["regen_mask"], plain.inputs["regen_mask"]
    for g, mm in ((gen, m), (twin, tm), (plain, pm)):
        _prepare_masked(g, mm)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(m, mask_src, stream=s)
        assert gen.generate(stream=s, advance=True, mask=m) is out
    prev = _drawn(gen, keys)
    for r, mk in enumerate((_M1, 1 - _M1, _ZERO, _ONES)):
        label = f"replay {r} mask {mk.tolist()}"
        _write(mask_src, mk)
        tg_capture.poison_all(out)
        poison = _snapshot(out)
        _replay(cap.graph, s)
        _write(tm, mk)
        twin.enqueue(advance=True, mask=tm)
        _set_drawn(plain, gen, keys)
        _write(pm, mk)
        plain.enqueue(advance=True, mask=pm)
        wp.synchronize()
        d, dt = _drawn(gen, keys), _drawn(twin, keys)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        for k in d:
            np.testing.assert_array_equal(d[k][unmasked], prev[k][unmasked],
                                          err_msg=f"{label}: unmasked draws of {k}")
            assert np.all(d[k][masked] != prev[k][masked]), f"{label}: {k} not redrawn"
            np.testing.assert_array_equal(d[k], dt[k], err_msg=f"{label}: {k} vs twin")
        cur = _snapshot(out)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env's Track changed")
        _assert_host(_rows(cur, masked), _rows(_snapshot(twin.program.result), masked),
                     f"{label}: masked vs the randomized twin")
        _assert_host(_rows(cur, masked), _rows(_snapshot(plain.program.result), masked),
                     f"{label}: masked vs the twin set to the drawn values")
        prev = d


# ---------------------------------------------------------------------------------------
# 7. A curriculum written in the caller's graph
# ---------------------------------------------------------------------------------------

#: generator -> (knob, [steps, E] schedule; values outside the declared bounds included).
_CURRICULUM = {
    "bezier": ("rad", ((0.1, 0.2, 0.3, 0.35, -0.25, 0.15, 0.25, 0.05),
                       (0.35, 0.3, -1.0, 0.2, 0.1, 0.05, 0.4, 0.3),
                       (0.2, 0.2, 0.2, 0.2, -np.inf, 0.2, 0.2, 0.2))),
    "polar": ("polar_radial_jitter", ((0.1, 0.3, 1.2, 0.5, 0.7, 0.2, 0.4, 0.6),
                                      (-0.1, 0.8, 0.85, 0.3, 2.0, 0.45, 0.6, 0.0),
                                      (0.6, 0.6, 0.6, np.inf, 0.6, 0.6, 0.6, 0.6))),
}
#: elevation.uniform z_max schedule (z_min = -0.2 static; z_max has no declared bound).
_ZMAX_STEPS = ((0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65),
               (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.35, 0.45),
               (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("gen_name", list(_CURRICULUM))
def test_curriculum_written_inside_the_caller_graph(gen_name, mode):
    """ONE caller graph holds ``[the caller's kernel writes params["generator"][knob]
    and params["elevation"]["z_max"] from row k = step % 3 of a device schedule] ->
    generate(advance=True) -> [step += 1]``. Recording executes nothing. Every replay
    (no host write in between): the live arrays hold row k as written (out-of-bounds
    values included: the stage never clamps or writes them), the batch equals the eager
    twin ``set`` to row k CLAMPED to the declared bounds and advanced alike, and differs
    from the default twin's (the writes matter)."""
    field, knob_steps = _CURRICULUM[gen_name]
    lo, hi = _declared("generator", gen_name, field)
    knob_tab = np.asarray(knob_steps, np.float32)
    zmax_tab = np.asarray(_ZMAX_STEPS, np.float32)
    steps = knob_tab.shape[0]
    assert knob_tab.shape == zmax_tab.shape == (steps, E)
    out_of_bounds = [(lo is not None and v < np.float32(lo)) or
                     (hi is not None and v > np.float32(hi)) for v in knob_tab.ravel()]
    assert any(out_of_bounds), "the schedule must exercise the clamp"
    _load_test_kernels()

    spec = presets.track(num_envs=E, device=DEV, generator=gen_name,
                         elevation=("uniform", _ZU_P))
    gen = TrackGenerator(spec, seeds=11)
    twin = TrackGenerator(spec, seeds=11)
    default = TrackGenerator(spec, seeds=11)
    out = gen.program.result
    knob_live = gen.params["generator"][field]
    zmax_live = gen.params["elevation"]["z_max"]
    knob0 = knob_live.numpy().copy()
    zmax0 = zmax_live.numpy().copy()
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    knob_dev, zmax_dev = _dev(knob_tab.ravel()), _dev(zmax_tab.ravel())

    def body(stream):
        wp.launch(_curriculum_k, dim=E, inputs=[counter, steps, knob_dev, zmax_dev],
                  outputs=[knob_live, zmax_live], device=DEV, stream=stream)
        assert gen.generate(stream=stream, advance=True) is out
        wp.launch(_bump_k, dim=1, inputs=[counter], device=DEV, stream=stream)

    replay = _caller_graph(mode, body)
    assert gen.graph is None
    np.testing.assert_array_equal(counter.numpy(), [0], err_msg="the recording executed")
    np.testing.assert_array_equal(knob_live.numpy(), knob0, err_msg="the recording wrote")
    np.testing.assert_array_equal(zmax_live.numpy(), zmax0, err_msg="the recording wrote")

    for r in range(steps + 1):
        k = r % steps
        label = f"{gen_name}/{mode}: replay {r} (step {k})"
        tg_capture.poison_all(out)
        replay()
        np.testing.assert_array_equal(counter.numpy(), [r + 1], err_msg=label)
        np.testing.assert_array_equal(knob_live.numpy().view(np.uint32),
                                      knob_tab[k].view(np.uint32),
                                      err_msg=f"{label}: live {field}")
        np.testing.assert_array_equal(zmax_live.numpy().view(np.uint32),
                                      zmax_tab[k].view(np.uint32),
                                      err_msg=f"{label}: live z_max")
        clamped = np.array([_clamp(v, lo, hi) for v in knob_tab[k]], np.float32)
        a, b = _dev(clamped), _dev(zmax_tab[k])
        twin.params["generator"].set(field, a)
        twin.params["elevation"].set("z_max", b)
        wp.synchronize()
        twin.enqueue(advance=True)
        default.enqueue(advance=True)
        wp.synchronize()
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result),
                     f"{label}: vs the twin set to the clamped schedule")
        assert _differs(got, _snapshot(default.program.result)), \
            f"{label}: the in-graph writes do not change the batch (vacuous)"


# ---------------------------------------------------------------------------------------
# 8. The conformance kit on CUDA: params.per_env
# ---------------------------------------------------------------------------------------

_KIT = (
    [("generator", g, None) for g in ("bezier", "hull", "polar", "voronoi", "checkpoint")]
    + [("gate_generator", g, None) for g in ("polar", "checkpoint", "voronoi")]
    + [("elevation", "uniform", _ZU_P), ("elevation", "random_walk", _ZW_P),
       ("elevation", "noise", _ZN_P)]
)


@pytest.mark.parametrize("kind,name,params", _KIT,
                         ids=[f"{k}.{n}" for k, n, _ in _KIT])
def test_builtin_passes_the_kit_with_params_per_env_on_cuda(kind, name, params):
    """``check_generator`` / ``check_gate_generator`` / ``check_stage("elevation",
    ...)`` on ``cuda:0``: every check passes, ``params.per_env`` ran (not skipped: the
    built-in has dynamic fields) and passed, and the caller capture replayed to the
    eager twin."""
    if kind == "generator":
        report = check_generator(name, devices=[DEV])
    elif kind == "gate_generator":
        report = check_gate_generator(name, devices=[DEV])
    else:
        report = check_stage(kind, name, params=params, devices=[DEV])
    report.raise_for_failures()
    per_env = report.result("params.per_env", DEV)
    assert not per_env.skipped and per_env.passed, f"{kind}.{name}: {report}"
    caller = report.result("capture.caller", DEV)
    assert not caller.skipped and caller.details.get("warp") == "ok", f"{kind}.{name}: {report}"
