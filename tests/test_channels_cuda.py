"""CUDA-only: per-point channels, variable half-width, ``Track.half_width`` /
``attributes`` and 3D station spacing under caller capture (roadmap PR13a).

The toys of ``tests/_channel_toys.py`` (public plugin surface only, the fake
``acme-channels`` distribution, installed through a fake entry point) drive the new paths:

- generator ``acme.ribbon``: a smooth closed loop that writes carried families
  ``raw.acme.theta`` (f32, ``angle``), ``raw.acme.surface_id`` (int32, ``nearest``),
  ``raw.acme.lane`` (int32, ``previous``), ``raw.acme.grip`` (f32, ``linear``) and
  ``raw.acme.xy_copy`` (vec2f, ``linear``, a copy of ``raw.xy``), all exported, plus
  ``raw.acme.hw`` (f32, ``linear``, not exported: ``hw_mean[e] + hw_amp[e] *
  sin(...)``, both dynamic); ``acme.ribbon_plain`` is the same ``raw.xy`` without
  families;
- generator ``acme.helix_loop`` (provides ``z``) and ``acme.pinch`` (provides
  ``half_width``); width stages ``acme.wave`` / ``acme.rescale``; gate generator
  ``acme.tagged`` (``anchors.acme.tag``).

The variable-width track is ``acme.ribbon`` + ``width.from_channel`` wired to its carried
``cl.acme.hw`` through ``StageRef.ports`` (per-bead XPBD radii, the per-point offset
band, the per-pair thickness and the real ``w_floor`` check).

What this module proves on ``cuda:0`` (design section 9.4):

1. Caller capture (Warp ``ScopedCapture``, and torch default / registered modes through
   ``testing.capture``) of attribute + variable-width + ``on_invalid="poison"`` track
   Programs and of gate Programs carrying channels (the helix-loop ``xyz`` stations, a
   relaxed-stations ``xyz`` pipeline with a centerline elevation, ribbon stations with
   ``GateSequence.attributes``, ``acme.tagged``): every replay equals the eager twin,
   before and after a reseed; ``hw_mean`` / ``hw_amp`` / ``scale`` written inside the
   caller graph take effect on the replay (and move the poisoned set).
2. Masked regeneration recorded with ``inputs["regen_mask"]`` (the mask rewritten
   in-graph): unmasked envs keep geometry, ``half_width`` and every attribute byte for
   byte, also when the pipeline overwrote the very result buffers; masked envs equal the
   eager twin at the same (seed, epoch).
3. GPU validations of what the design claims exact on CPU (section 4.3): the
   table-writing lookups and the ``_stations_ex_k`` xy branch leave the geometry
   unchanged (ribbon vs ribbon_plain), a linear carry of ``raw.xy`` reproduces the
   resampled points, the per-point offset / validity kernels reproduce the scalar ones
   at a constant-valued width; and ``allclose`` ones: the XPBD variant vs the scalar
   solve, the ``xyz`` stations vs the CPU kernel.
4. ``build(..., stream=s)`` with ``s`` not Warp's current stream (which is held busy):
   the build-time ``width.constant`` fill is ordered before the first generation.
5. Carried padding on recycled (dirty) device memory: NaN / -1 beyond ``count`` in every
   ``Track`` attribute, and pure padding in every anchors / gates attribute of an env
   the stations skip, on the first generation and after envs flip valid <-> invalid.
6. The conformance kit (``check_stage`` / ``check_generator`` /
   ``check_gate_generator``) on the toys with ``devices=["cuda:0"]``.

Method (as in tests/test_caller_capture_cuda.py, tests/test_stages_cuda.py and
tests/test_gate_stages_cuda.py; the helpers are :mod:`track_gen.testing.capture`'s):
record on a NON-current side stream, poison the outputs with FINITE, dtype-aware
sentinels (outputs are NaN / -1 padded, poisoned envs are NaN / -1, and
``assert_array_equal`` treats NaN == NaN, so a NaN poison would hide a missed write),
replay, and compare with an eager twin: a fresh generator of the same spec run with
``enqueue()`` (never a graph). Every recording runs with the Warp mempool disabled (an
allocation inside the capture fails loudly) and ``wp.capture_launch`` replaced by a
tripwire (a graph launch cannot be recorded into another capture). Construction prepares
the default body; masked bodies are warmed up with an all-zero mask (an exact no-op) and
the test's own kernels on scratch buffers, so no module loads while capturing. Per-env
parameter values written in-graph are compared with rows stitched from twins built with
uniform values, so the per-env latch under test is never also the reference. The CPU
lane only collects this module (the CPU proofs are the PR13a CPU suite, design 9.3).
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from tests._channel_toys import acme_channels_entry_point  # noqa: E402
from tests._plugin_toys import install_entry_points  # noqa: E402
from track_gen import GateGenerator, PerEnvSeededRNG, TrackGenerator  # noqa: E402
from track_gen.pipeline import BUILTIN_FAIL_BITS, StageRef, build, presets  # noqa: E402
from track_gen.plugin import POLICY_ENV, registry  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import (check_gate_generator, check_generator,  # noqa: E402
                               check_stage)
from track_gen.testing.trace import trace_launches  # noqa: E402

DEV = "cuda:0"
E = 8
G_ST = 12                       # stations per env (presets.gates(gates=...))
# Stations on the helix loop (z = height * sin(3 theta)): 12 would put every station at
# a symmetry point of the 3D speed profile, where xy and xyz spacing coincide.
G_XYZ = 10

RIBBON = "acme.ribbon"
RIBBON_PLAIN = "acme.ribbon_plain"
HELIX_LOOP = "acme.helix_loop"
PINCH = "acme.pinch"
WAVE = "acme.wave"
RESCALE = "acme.rescale"
TAGGED = "acme.tagged"
_TOYS = (("generator", RIBBON), ("generator", RIBBON_PLAIN), ("generator", HELIX_LOOP),
         ("generator", PINCH), ("width", WAVE), ("width", RESCALE),
         ("gate_generator", TAGGED))

#: The cl image of ribbon's ``raw.acme.hw`` (the source of ``width.from_channel``).
HW_SRC = "cl.acme.hw"
#: ribbon's exported attributes: key -> dtype (the family dtype of every image).
RIBBON_ATTRS = {"acme.theta": wp.float32, "acme.surface_id": wp.int32,
                "acme.lane": wp.int32, "acme.grip": wp.float32, "acme.xy_copy": wp.vec2f}
#: acme.tagged's exported anchors family (``anchors.acme.tag`` -> ``gates.acme.tag``).
TAG_KEY = "acme.tag"
_ATTR = "attributes/"           # testing.capture.result_arrays' prefix of an attribute

_ZERO = np.zeros(E, np.int32)
_ONES = np.ones(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
_W_FLOOR_BIT = BUILTIN_FAIL_BITS["w_floor"]
_THICKNESS_BIT = BUILTIN_FAIL_BITS["thickness"]
_TRACK_POINT_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen",
                       "half_width")
# A non-flat centerline elevation (the PR11 3D variant's profile).
_NOISE = ("noise", {"z_base": 0.5, "z_max": 1.5, "z_noise_amplitude": 0.5,
                    "z_noise_harmonics": 4})
# The XPBD variant vs the scalar solve at a constant-valued width: allclose on CUDA
# (design 4.3; on CPU it is exact at the default (0.1, 0.15) pair). A one-ulp change of
# an active constraint can move a bead by a few 1e-3 (E3d's worst case, 6.1e-3, came from
# targets differing throughout), so the bound is 5 % of the half-width.
_XPBD_ATOL = 5.0e-3


@pytest.fixture
def acme(monkeypatch):
    """The ``acme-channels`` toy distribution, discovered through a fake entry point in a
    fresh registry scope (restored afterwards, as is ``TRACK_GEN_PLUGINS``)."""
    monkeypatch.delenv(POLICY_ENV, raising=False)
    install_entry_points(monkeypatch, acme_channels_entry_point())
    with registry.scoped(fresh=True):
        for kind, name in _TOYS:
            assert not registry.describe(kind, name).builtin, (kind, name)
        yield registry


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True)
            for k, a in tg_capture.result_arrays(obj).items()}


def _attr_keys(host: dict) -> list:
    return sorted(k for k in host if k.startswith(_ATTR))


def _geometry(host: dict) -> dict:
    """Every result array except the attributes."""
    return {k: v for k, v in host.items() if not k.startswith(_ATTR)}


def _pad(values: np.ndarray):
    """The padding of a carried family: -1 for int32, NaN for floats (every component)."""
    return -1 if np.issubdtype(values.dtype, np.integer) else np.nan


def _rows(host: dict, envs) -> dict:
    return {k: v.reshape(E, -1)[envs] for k, v in host.items()}


def _assert_host(got: dict, ref: dict, label: str) -> None:
    assert ref and got.keys() == ref.keys(), f"{label}: {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{label}: {name}")


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def _stitch(per_env: list) -> dict:
    """The batch whose env ``e`` is env ``e`` of ``per_env[e]`` (host snapshots)."""
    return {name: np.stack([s[name].reshape(E, -1)[e] for e, s in enumerate(per_env)])
            .reshape(per_env[0][name].shape) for name in per_env[0]}


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


def _per_env(values) -> np.ndarray:
    v = np.asarray(values, np.float32)
    return np.resize(v, E).astype(np.float32) if v.ndim else np.full(E, v, np.float32)


def _reseed(rng, base: int) -> None:
    """Reseed outside any capture (same expansion as ``PerEnvSeededRNG(seeds=base)``)."""
    rng.set_seeds_warp(wp.array((base + np.arange(E)).astype(np.int32), dtype=wp.int32,
                                device=DEV), None)
    wp.synchronize()


def _make(spec, seeds=11):
    """``TrackGenerator`` / ``GateGenerator`` (by ``spec.batch.max_gates``) of ``spec``;
    returns ``(gen, rng)``."""
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV)
    cls = TrackGenerator if spec.batch.max_gates is None else GateGenerator
    return cls(spec, rng), rng


def _eager(spec, seeds=11) -> dict:
    """A fresh twin of ``spec`` run eagerly (``enqueue`` never self-graphs)."""
    twin, _ = _make(spec, seeds)
    out = twin.enqueue()
    wp.synchronize()
    assert twin.graph is None, "enqueue() must never self-graph"
    return _snapshot(out)


def _stage_id(spec, kind: str, name: str) -> str:
    ids = [r.slot_id for r in spec.stages if r.kind == kind and r.name == name]
    assert len(ids) == 1, f"{kind}.{name} in {spec.ids}"
    return ids[0]


def _has(spec, kind: str, name: "str | None" = None) -> bool:
    return any(r.kind == kind and (name is None or r.name == name) for r in spec.stages)


def _set(gen, stage_id: str, field: str, values, *, stream=None) -> None:
    """Write a dynamic field's live array: a scalar ``set`` for uniform values, an
    ``[E]`` array ``set`` otherwise."""
    v = _per_env(values)
    sp = gen.params[stage_id]
    src = float(v[0]) if np.all(v == v[0]) else _dev(v)
    sp.set(field, src, stream=stream)
    wp.synchronize()                       # src stays alive until the copy completed


def _assert_valid_is_fail_mask_zero(host: dict, label: str) -> None:
    assert "fail_mask" in host, f"{label}: fail_mask is None"
    np.testing.assert_array_equal(host["valid"], (host["fail_mask"] == 0).astype(np.int32),
                                  err_msg=f"{label}: valid != (fail_mask == 0)")


def _poisoned(keep: dict) -> dict:
    """The ``on_invalid="poison"`` Track of a ``"keep"`` snapshot, derived independently
    of the commit kernels: every env with ``fail_mask != 0`` gets ``count = 0``, NaN on
    every per-point field (``half_width`` included), the family padding (NaN / -1) on
    every attribute row, ``length = winding = valid = 0``; ``fail_mask``, ``truncated``
    and ``generation`` keep their values; valid envs are unchanged."""
    assert "half_width" in keep and _attr_keys(keep), sorted(keep)
    out = {k: v.copy() for k, v in keep.items()}
    bad = keep["fail_mask"] != 0
    for name in _TRACK_POINT_FIELDS:
        out[name].reshape(E, -1)[bad] = np.nan
    for name in _attr_keys(keep):
        out[name].reshape(E, -1)[bad] = _pad(out[name])
    out["count"][bad] = 0
    out["length"][bad] = 0.0
    out["winding"][bad] = 0.0
    out["valid"][bad] = 0
    return out


def _assert_padding(arr: np.ndarray, count, label: str) -> None:
    """Every slot ``>= count[e]`` of ``arr`` (``[E*S]`` or ``[E*S, k]``) holds the family
    padding: NaN (every component) for floats, -1 for int32."""
    rows = arr.reshape(E, int(arr.shape[0]) // E, -1)
    for e in range(E):
        tail = rows[e, int(count[e]):]
        if np.issubdtype(arr.dtype, np.integer):
            assert np.all(tail == -1), f"{label}: env {e}: int padding {np.unique(tail)}"
        else:
            assert np.all(np.isnan(tail)), \
                f"{label}: env {e}: float padding {tail[~np.isnan(tail)][:4]}"


def _kernels(trace) -> set:
    """The short names (``<warp module>.<key>`` -> ``<key>``) of a trace's kernels."""
    return {k.rsplit(".", 1)[-1] for k in trace.kernels()}


def _traced_kernels(gen) -> set:
    """The kernels one eager generation of ``gen`` launches (same seeds and epoch: the
    batch is unchanged)."""
    with trace_launches() as tr:
        gen.enqueue()
        wp.synchronize()
    return _kernels(tr)


def _assert_gpu_exact(got, want, label: str) -> None:
    """A GPU validation the design expects to be exact (section 4.3: the expression is
    the CPU-proven one). On a mismatch the message says whether the documented fallback
    (``allclose`` plus a note on the pipeline page) would hold."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape == want.shape and np.array_equal(got, want, equal_nan=True):
        return
    close = diff = None
    if got.shape == want.shape:
        close = bool(np.allclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True))
        with np.errstate(invalid="ignore"):
            d = np.abs(got.astype(np.float64) - want.astype(np.float64))
        diff = float(np.nanmax(d)) if np.any(~np.isnan(d)) else None
    raise AssertionError(
        f"{label}: not byte-equal on {DEV} (shapes {got.shape} / {want.shape}, max |diff| "
        f"{diff}, allclose(rtol=1e-5, atol=1e-6): {close}). Design 4.3 expects this "
        f"exact on the GPU; its documented fallback is allclose plus a note, if nvcc "
        f"contracts the two expressions differently")


@wp.kernel
def _spin_k(n: int, out: wp.array(dtype=wp.float32)):
    # One thread, a dependent chain of n sines: holds a stream busy for a while.
    acc = float(0.0)
    for _i in range(n):
        acc = wp.sin(acc + 1.0)
    out[0] = acc


@contextlib.contextmanager
def _dirty_pool():
    """Recycled device memory for the block: the mempool keeps (does not release)
    blocks freed just before, which held a finite non-zero pattern (4242.0: neither NaN
    nor -1 in any dtype's view), so a buffer allocated uninitialised (``wp.empty``) in the
    block very likely starts with stale bytes, not zeros. Best effort: allocations on
    the current stream reuse the freed blocks."""
    if not wp.is_mempool_enabled(DEV):
        yield
        return
    was = wp.get_mempool_release_threshold(DEV)
    wp.set_mempool_release_threshold(DEV, 1 << 30)
    try:
        junk = [wp.full(n, 4242.0, dtype=wp.float32, device=DEV)
                for n in (1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18)
                for _ in range(8)]
        wp.synchronize()
        del junk
        wp.synchronize()
        yield
    finally:
        wp.set_mempool_release_threshold(DEV, was)


# ---------------------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------------------

def _track(**kw):
    return presets.track(num_envs=E, device=DEV, **kw)


def _gates(**kw):
    return presets.gates(num_envs=E, device=DEV, **kw)


def _helix_gates(metric: str, device: "str | None" = None):
    """Stations on the z-providing helix loop, spaced by ``metric``."""
    return presets.gates(num_envs=E, device=DEV if device is None else device,
                         generator=HELIX_LOOP, gates=G_XYZ, spacing_metric=metric)


def _from_channel(scale=None) -> StageRef:
    """``width.from_channel`` wired to ribbon's carried ``cl.acme.hw``."""
    return StageRef("width", "from_channel",
                    params=None if scale is None else {"scale": float(scale)},
                    ports={"src": HW_SRC})


def _ribbon_spec(on_invalid: str = "keep", *, hw_mean=None, hw_amp=None, scale=None, **kw):
    """ribbon (every family) + ``width.from_channel`` on ``cl.acme.hw`` (the variable
    width path) + ``on_invalid``; ``None`` leaves a field at its default."""
    gp = {k: float(v) for k, v in (("hw_mean", hw_mean), ("hw_amp", hw_amp))
          if v is not None}
    return _track(generator=RIBBON, generator_params=gp or None, width=_from_channel(scale),
                  on_invalid=on_invalid, **kw)


def _ribbon_defaults() -> "tuple[float, float]":
    """ribbon's default ``(hw_mean, hw_amp)``."""
    p = registry.get("generator", RIBBON).Params()
    return float(p.hw_mean), float(p.hw_amp)


def _pinch_amp() -> float:
    """An ``hw_amp`` that pinches ribbon's width source below 0 over a third of the loop:
    ``width.from_channel`` clamps it to ``lo = 0``, so the final points there fail the
    ``w_floor`` check (bit 4) whatever the resampling."""
    return 2.0 * _ribbon_defaults()[0]


# ---------------------------------------------------------------------------------------
# 1. The result surface the capture helpers see
# ---------------------------------------------------------------------------------------

def test_result_arrays_cover_half_width_and_every_attribute(acme):
    """The kit's ``result_arrays`` (what ``poison_all``, every snapshot below and
    ``assert_caller_capturable`` see) includes ``Track.half_width`` and every exported
    attribute as ``attributes/<key>`` (the same arrays, ``[E*S]``, the family dtype,
    keys == ``Program.attributes``), and ``poison_all`` fills them with FINITE
    sentinels, distinguishable from the NaN / -1 padding."""
    gen, _ = _make(_ribbon_spec())
    track = gen.program.result
    arrays = tg_capture.result_arrays(track)
    S = int(track.center.shape[0]) // E
    assert arrays.get("half_width") is track.half_width, sorted(arrays)
    assert set(track.attributes) == set(RIBBON_ATTRS), sorted(track.attributes)
    assert set(gen.program.attributes) == set(RIBBON_ATTRS), sorted(gen.program.attributes)
    for key, dtype in RIBBON_ATTRS.items():
        arr = track.attributes[key]
        assert arrays.get(_ATTR + key) is arr, (key, sorted(arrays))
        assert arr.dtype is dtype and arr.shape == (E * S,), (key, arr.dtype, arr.shape)
        assert str(arr.device) == DEV, (key, arr.device)
        assert gen.program.attributes[key].dtype is dtype, gen.program.attributes[key]
    tg_capture.poison_all(track)
    host = _snapshot(track)
    for name in ["half_width"] + [_ATTR + k for k in RIBBON_ATTRS]:
        v = host[name]
        assert np.all(np.isfinite(v)) and not np.any(v == -1), f"{name}: poison not finite"


# ---------------------------------------------------------------------------------------
# 2. Caller capture: track Programs
# ---------------------------------------------------------------------------------------

def _record_replay_reseed(spec, label: str, *, seeds=(11, 101), check=None):
    """Build the generator of ``spec``, record ONE ``generate(stream=s)`` into the
    caller's capture (no allocation, no capture_launch, no self-graph; the recording
    executes nothing), then replay it into poisoned outputs three times: at the build
    seeds, after a reseed to ``seeds[1]``, and back. Each replay equals the eager twin
    of those seeds and has ``valid == (fail_mask == 0)``; ``check(gen, host, label)``
    runs after each. Returns ``(gen, out, cap, stream, rng)``."""
    gen, rng = _make(spec, seeds[0])
    refs = {s: _eager(spec, s) for s in seeds}
    assert _differs(refs[seeds[0]], refs[seeds[1]]), \
        f"{label}: reseeding must change the batch (vacuous test otherwise)"
    base = _snapshot(gen.program.result)
    _assert_host(base, refs[seeds[0]], f"{label}: construction batch vs eager twin")

    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert out is gen.program.result, label
    assert gen.graph is None, \
        f"{label}: recording into the caller's graph must not self-capture"
    _assert_host(_snapshot(out), base, f"{label}: the recording executed the pipeline")

    for k, seed in enumerate((seeds[0], seeds[1], seeds[0])):
        if k:
            _reseed(rng, seed)
        step = f"{label}: replay {k} (seeds {seed}..)"
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        got = _snapshot(out)
        _assert_host(got, refs[seed], step)
        _assert_valid_is_fail_mask_zero(got, step)
        if check is not None:
            check(gen, got, step)
    assert gen.graph is None
    return gen, out, cap, s, rng


def _assert_kit_capturable(gen, label: str) -> None:
    """The conformance kit's recording (Warp capture, plus the torch default and
    registered modes when torch with CUDA is available) replays to the eager twin before
    and after a reseed."""
    status = tg_capture.assert_caller_capturable(gen)
    assert status["warp"] == "ok", f"{label}: {status}"
    assert set(status) == set(tg_capture.CAPTURE_MODES), status
    assert gen.graph is None, f"{label}: caller recordings never self-capture"


def _check_variable_width(gen, host, label) -> None:
    """``Track.half_width`` of the replay is the band the replay built: ``|outer -
    center| == |center - inner| == half_width`` in the plan view on every real point
    (1e-5), NaN padding, and it is not constant on some env (the variable path)."""
    count = host["count"]
    hw = host["half_width"].reshape(E, -1)
    c = host["center"].reshape(E, -1, 3)[..., :2]
    o = host["outer"].reshape(E, -1, 3)[..., :2]
    i = host["inner"].reshape(E, -1, 3)[..., :2]
    varied = False
    for e in range(E):
        m = int(count[e])
        assert np.all(np.isnan(hw[e, m:])), f"{label}: env {e}: half_width padding"
        if m == 0:
            continue
        np.testing.assert_allclose(np.linalg.norm(o[e, :m] - c[e, :m], axis=-1), hw[e, :m],
                                   atol=1e-5, err_msg=f"{label}: env {e}: outer band")
        np.testing.assert_allclose(np.linalg.norm(c[e, :m] - i[e, :m], axis=-1), hw[e, :m],
                                   atol=1e-5, err_msg=f"{label}: env {e}: inner band")
        varied |= bool(np.ptp(hw[e, :m]) > 1e-4)
    assert varied, f"{label}: half_width is constant on every env (vacuous)"


# ribbon's attribute ranges on real points: key -> (lo, hi) (xy_copy: the loop's extent
# at scale 1, semi-axes 1.0 / 0.8; theta: the angle carry's closed range).
_RIBBON_RANGES = {"acme.theta": (-np.pi, np.pi), "acme.surface_id": (0, 3),
                  "acme.lane": (0, 3), "acme.grip": (-1.0, 1.0),
                  "acme.xy_copy": (-1.0, 1.0)}


def _check_attributes(gen, host, label) -> None:
    """Every ribbon attribute of a result: family padding beyond ``count``; finite
    values within the family's range on real points (no stale bytes)."""
    count = host["count"]
    for key in _attr_keys(host):
        v = host[key]
        _assert_padding(v, count, f"{label}: {key}")
        lo, hi = _RIBBON_RANGES[key[len(_ATTR):]]
        rows = v.reshape(E, int(v.shape[0]) // E, -1)
        for e in range(E):
            real = rows[e, :int(count[e])].astype(np.float64)
            assert np.all(np.isfinite(real)), f"{label}: {key} env {e}: non-finite"
            assert np.all((real >= lo - 1e-6) & (real <= hi + 1e-6)), \
                f"{label}: {key} env {e}: outside [{lo}, {hi}]: {real.min()}..{real.max()}"


# label -> (spec builder, check after each replay)
_TRACK_CASES = {
    "ribbon-from_channel-keep": (lambda: _ribbon_spec("keep"), _check_attributes),
    "ribbon-from_channel-poison": (lambda: _ribbon_spec("poison"), _check_attributes),
    "pinch-provides-half_width": (lambda: _track(generator=PINCH), _check_variable_width),
    "wave-width": (lambda: _track(width=WAVE), _check_variable_width),
    "helix_loop-provides-z": (lambda: _track(generator=HELIX_LOOP), None),
}


@pytest.mark.parametrize("case", list(_TRACK_CASES))
def test_track_with_channels_records_into_caller_capture(acme, case):
    """Track Programs on the new paths: ribbon's five exported attributes + a per-point
    width from ``width.from_channel`` (``on_invalid`` keep and poison), a generator that
    provides its half-widths (no width stage), a plugin width before relax, and a
    generator that provides z (no elevation stage, ``lift.elevated``). Recorded into
    the caller's graph, every replay into poisoned outputs (attributes and
    ``half_width`` included) equals the eager twin before and after a reseed, the
    band follows the replay's own half-width, and the conformance kit's recording
    (torch modes included) agrees."""
    build_spec, check = _TRACK_CASES[case]
    spec = build_spec()
    if case.startswith("ribbon"):
        assert _ref_ports(spec, "width") == {"src": HW_SRC}, spec.stages
    if case.startswith("pinch"):
        assert not _has(spec, "width"), spec.ids
    if case.startswith("helix"):
        assert not _has(spec, "elevation") and _has(spec, "lift", "elevated"), spec.ids

    def checks(gen, host, label):
        if case.startswith("ribbon"):
            _check_variable_width(gen, host, label)
        if check is not None:
            check(gen, host, label)
        if case.startswith("helix"):
            z = np.nan_to_num(host["center"].reshape(E, -1, 3)[..., 2])
            assert np.any(z != 0.0), f"{label}: the provided z did not reach the Track"

    gen, *_ = _record_replay_reseed(spec, case, check=checks)
    _assert_kit_capturable(gen, case)


def _ref_ports(spec, stage_id: str) -> dict:
    refs = [r for r in spec.stages if r.slot_id == stage_id]
    assert len(refs) == 1, f"{stage_id!r} in {spec.ids}"
    return dict(refs[0].ports)


def test_width_and_attribute_params_written_inside_the_caller_graph_with_poison(acme):
    """ONE caller graph holds ``[set hw_mean, hw_amp (ribbon) and scale
    (width.from_channel) from device sources] -> generate()`` on the poison Program.
    Recording executes nothing. Before each replay the sources are rewritten outside the
    graph; the replay copies them into the live arrays, latches them, and equals the
    poison twins built with those values (per-env rows stitched from uniform twins),
    which are the keep twins with the invalid envs poisoned (geometry, ``half_width``
    and every attribute). The ``w_floor`` bit, and so the poisoned set, follows
    ``hw_amp`` on device replay by replay: set exactly where the width source pinches
    below zero (clamped to 0 <= w_floor)."""
    spec = _ribbon_spec("poison")
    gen, _ = _make(spec)
    gsid = _stage_id(spec, "generator", RIBBON)
    wsid = _stage_id(spec, "width", "from_channel")
    fields = {"hw_mean": gsid, "hw_amp": gsid, "scale": wsid}
    defaults = {f: float(getattr(gen.params[sid].params, f)) for f, sid in fields.items()}
    src = {f: _dev(_per_env(v)) for f, v in defaults.items()}
    for f, sid in fields.items():                      # warm-up: a no-op write of each
        gen.params[sid].set(f, src[f])
    wp.synchronize()
    base = _snapshot(gen.program.result)

    s = _side_stream()
    with _recording(s) as cap:
        for f, sid in fields.items():
            gen.params[sid].set(f, src[f], stream=s)
        out = gen.generate(stream=s)
    assert gen.graph is None
    for f, sid in fields.items():
        np.testing.assert_array_equal(gen.params[sid][f].numpy(), _per_env(defaults[f]),
                                      err_msg=f"the recording executed the {f} write")
    _assert_host(_snapshot(out), base, "the recording executed the pipeline")

    cache: dict = {}

    def twin(on_invalid, hw_mean, hw_amp, scale) -> dict:
        key = (on_invalid, float(hw_mean), float(hw_amp), float(scale))
        if key not in cache:
            cache[key] = _eager(_ribbon_spec(on_invalid, hw_mean=hw_mean, hw_amp=hw_amp,
                                             scale=scale))
        return cache[key]

    pinch = _pinch_amp()
    split = np.array([0.0, pinch] * (E // 2), np.float32)
    schedule = (
        {"hw_mean": defaults["hw_mean"], "hw_amp": split, "scale": 1.0},
        {"hw_mean": 0.8 * defaults["hw_mean"], "hw_amp": 0.0, "scale": [1.0, 1.25]},
        {"hw_mean": defaults["hw_mean"], "hw_amp": split[::-1].copy(),
         "scale": defaults["scale"]},
    )
    seen = []
    for k, step in enumerate(schedule):
        label = f"replay {k} {step}"
        values = {f: _per_env(v) for f, v in step.items()}
        for f in fields:
            _write(src[f], values[f])
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        for f, sid in fields.items():
            np.testing.assert_array_equal(gen.params[sid][f].numpy(), values[f],
                                          err_msg=f"{label}: live {f}")
        got = _snapshot(out)
        per = list(zip(values["hw_mean"], values["hw_amp"], values["scale"]))
        _assert_host(got, _stitch([twin("poison", *p) for p in per]), f"{label}: vs twin")
        _assert_host(got, _poisoned(_stitch([twin("keep", *p) for p in per])),
                     f"{label}: vs keep twin, poisoned")
        _assert_valid_is_fail_mask_zero(got, label)
        floor = (got["fail_mask"] >> _W_FLOOR_BIT) & 1
        np.testing.assert_array_equal(floor, (values["hw_amp"] == np.float32(pinch))
                                      .astype(floor.dtype),
                                      err_msg=f"{label}: w_floor bit where the source "
                                              f"pinches")
        bad = got["fail_mask"] != 0
        np.testing.assert_array_equal(got["count"][bad], 0, err_msg=label)
        for key in _attr_keys(got) + ["half_width"]:
            rows = got[key].reshape(E, -1)[bad]
            pad = _pad(got[key])
            assert np.all(rows == pad) if pad == -1 else np.all(np.isnan(rows)), \
                f"{label}: {key}: a poisoned env is not pure padding"
        seen.append(got)
    assert _differs(seen[0], seen[1]) and _differs(seen[1], seen[2])
    # The attributes themselves move with the replay (not only the geometry).
    assert any(not np.array_equal(seen[0][k], seen[1][k], equal_nan=True)
               for k in _attr_keys(seen[0]) + ["half_width"])


# ---------------------------------------------------------------------------------------
# 3. Caller capture: gate Programs carrying channels
# ---------------------------------------------------------------------------------------

def _check_carried_z(gen, host, label) -> None:
    """``carry.z`` by anchor index (``order.raw``): ``position.z == anchors.z`` bit for
    bit on every real gate of the replay, and the gates are not flat."""
    tap = gen.program.tap
    A = int(tap("anchors.xy").shape[0]) // E
    G = int(host["position"].shape[0]) // E
    z = tap("anchors.z").numpy().reshape(E, A)
    pos = host["position"].reshape(E, G, 3)
    for e in range(E):
        m = int(host["count"][e])
        np.testing.assert_array_equal(pos[e, :m, 2], z[e, :m],
                                      err_msg=f"{label}: env {e}: position.z")
    assert np.any(np.nan_to_num(pos[..., 2]) != 0.0), f"{label}: the gates are flat"


def _check_gate_attributes(keys):
    def check(gen, host, label):
        assert set(_attr_keys(host)) == {_ATTR + k for k in keys}, sorted(host)
        count = host["count"]
        for key in _attr_keys(host):
            _assert_padding(host[key], count, f"{label}: {key}")
    return check


def _relaxed_xyz(metric: str = "xyz"):
    """bezier ``relaxed_stations`` with a non-flat elevation on the relaxed centerline
    (``elevation_on="centerline"``, the ``xyz`` default) and ``metric`` spacing."""
    return _gates(generator="bezier", source="relaxed_stations", gates=G_ST,
                  elevation=_NOISE, spacing_metric=metric, elevation_on="centerline")


# label -> (spec builder, the xyz stations stage name or None, the same pipeline spaced
#           "xy" (None: no metric), carried attributes, check)
_GATE_CASES = {
    "helix_loop-stations-xyz": (
        lambda: _helix_gates("xyz"), "raw", lambda: _helix_gates("xy"), (),
        _check_carried_z),
    "bezier-relaxed-xyz-centerline-noise": (
        _relaxed_xyz, "relaxed", lambda: _relaxed_xyz("xy"), (), _check_carried_z),
    "ribbon-stations-attributes": (
        lambda: _gates(generator=RIBBON, gates=G_ST), None, None, tuple(RIBBON_ATTRS),
        None),
    "tagged-anchors-attribute": (
        lambda: _gates(generator=TAGGED), None, None, (TAG_KEY,), None),
}


@pytest.mark.parametrize("case", list(_GATE_CASES))
def test_gate_pipeline_with_channels_records_into_caller_capture(acme, case):
    """Gate Programs on the new paths: ``stations.raw`` with ``spacing_metric="xyz"`` on a
    z-providing loop (``_stations_ex_k``'s 3D branch, f64 accumulation, then
    ``carry.z``), ``relaxed_stations`` + ``xyz`` with the elevation run on the relaxed
    centerline before the stations (no gate-level ``gate_arclen`` / elevation), ribbon
    stations carrying every exported family to ``GateSequence.attributes``
    (``carry.attributes`` after ``order.raw``), and an anchors generator's own
    ``anchors.acme.tag``. Every replay into poisoned outputs equals the eager twin
    before and after a reseed, carried z / attributes reach the GateSequence of THAT
    replay, and the conformance kit's recording (torch modes included) agrees."""
    build_spec, stations, xy_spec, keys, check = _GATE_CASES[case]
    spec = build_spec()
    if stations is not None:
        sid = _stage_id(spec, "stations", stations)
        assert _ref_params(spec, sid).get("spacing_metric") == "xyz", spec.stages
        assert "gate_arclen" not in spec.ids, spec.ids
        assert _has(spec, "carry", "z"), spec.ids
        assert _stage_id(spec, "order", "raw")
        a, b = _eager(spec), _eager(xy_spec())
        assert not np.allclose(a["position"], b["position"], atol=1e-4, equal_nan=True), \
            f"{case}: xy and xyz stations coincide (the 3D metric is not exercised)"
    if keys:
        assert _has(spec, "carry", "attributes"), spec.ids
        check = _check_gate_attributes(keys)
    gen, *_ = _record_replay_reseed(spec, case, check=check)
    if keys:
        assert set(gen.program.result.attributes) == set(keys), \
            sorted(gen.program.result.attributes)
    _assert_kit_capturable(gen, case)


def _ref_params(spec, stage_id: str) -> dict:
    refs = [r for r in spec.stages if r.slot_id == stage_id]
    assert len(refs) == 1, f"{stage_id!r} in {spec.ids}"
    p = refs[0].params
    if p is None:
        return {}
    return dict(p) if hasattr(p, "keys") else \
        {f: getattr(p, f) for f in getattr(type(p), "__dataclass_fields__", {})}


# ---------------------------------------------------------------------------------------
# 4. Masked regeneration under capture
# ---------------------------------------------------------------------------------------

def _prepared(spec, hw_amp) -> TrackGenerator:
    """A generator of ``spec`` with ribbon's ``hw_amp`` live array set to ``hw_amp``
    and its masked body warmed up with an all-zero mask, which must be an exact no-op
    (nothing committed, no epoch advanced)."""
    g, _ = _make(spec)
    _set(g, _stage_id(spec, "generator", RIBBON), "hw_amp", hw_amp)
    base = _snapshot(g.program.result)
    m = g.inputs["regen_mask"]
    np.testing.assert_array_equal(m.numpy(), _ZERO, err_msg="regen_mask starts zeroed")
    assert g.enqueue(advance=True, mask=m) is g.program.result
    wp.synchronize()
    _assert_host(_snapshot(g.program.result), base, "an all-zero mask must change nothing")
    np.testing.assert_array_equal(g.seeds.epoch.numpy(), _ZERO)
    return g


def _new_batch(cur: dict, old: dict, e: int) -> bool:
    """Whether env ``e``'s geometry changed (an env poisoned in both counts as new: its
    NaN rows cannot differ)."""
    both_poisoned = cur["count"][e] == 0 == old["count"][e] and \
        cur["fail_mask"][e] != 0 and old["fail_mask"][e] != 0 and \
        np.isnan(cur["center"].reshape(E, -1)[e]).all()
    return both_poisoned or not np.array_equal(cur["center"].reshape(E, -1)[e],
                                               old["center"].reshape(E, -1)[e],
                                               equal_nan=True)


@pytest.mark.parametrize("on_invalid", ["keep", "poison"])
def test_masked_regeneration_in_caller_graph_keeps_unmasked_width_and_attributes(
        acme, on_invalid):
    """ONE caller graph holds ``[copy mask_src -> inputs["regen_mask"]] ->
    generate(advance=True, mask=regen_mask)`` on the ribbon + ``width.from_channel``
    Program (``hw_amp`` pinching every other env, so with ``"poison"`` some masked envs
    are poisoned and some kept). Before every replay the WHOLE Track is poisoned
    (``half_width``, every attribute and ``generation`` too); after it:

    - envs with ``mask[e] == 0`` still hold the poison byte for byte, geometry,
      ``half_width`` and every attribute (an all-zero mask changes no byte at all);
    - masked envs equal an eager twin driven with the same masks (same seed and epoch),
      with ``generation == epoch``, and got a new batch;
    - with ``"poison"`` the masked invalid envs are the ``"keep"`` twin's rows poisoned
      (NaN half_width, NaN / -1 attributes);
    - after an all-ones mask every env is committed and the Track equals the twin."""
    pinch = _pinch_amp()
    split = np.array([0.0, pinch] * (E // 2), np.float32)
    spec = _ribbon_spec(on_invalid)
    gen, twin = _prepared(spec, split), _prepared(spec, split)
    keep_twin = _prepared(_ribbon_spec("keep"), split) if on_invalid == "poison" else None
    out = gen.program.result
    m = gen.inputs["regen_mask"]
    twins = [t for t in (twin, keep_twin) if t is not None]
    base = _snapshot(out)
    assert "half_width" in base and len(_attr_keys(base)) == len(RIBBON_ATTRS), sorted(base)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(m, mask_src, stream=s)
        assert gen.generate(stream=s, advance=True, mask=m) is out
    assert gen.graph is None, "recording into the caller's graph must not self-capture"
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO,
                                  err_msg="the recording executed the epoch advance")
    _assert_host(_snapshot(out), base, "the recording executed the pipeline")

    schedule = [_M1, _M1, _ZERO, 1 - _M1, _ONES]
    epoch = np.zeros(E, np.int32)
    committed = {k: v.copy() for k, v in base.items()}
    poisoned_masked = kept_masked = 0
    for k, mk in enumerate(schedule):
        label = f"{on_invalid}: replay {k} mask {mk.tolist()}"
        _write(mask_src, mk)
        tg_capture.poison_all(out)
        poison = _snapshot(out)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(m.numpy(), mk, err_msg=f"{label}: in-graph mask copy")
        for t in twins:
            _write(t.inputs["regen_mask"], mk)
            t.enqueue(advance=True, mask=t.inputs["regen_mask"])
        wp.synchronize()
        ref = _snapshot(twin.program.result)
        epoch += mk
        cur = _snapshot(out)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env's Track changed")
        _assert_host(_rows(cur, masked), _rows(ref, masked),
                     f"{label}: masked envs vs twin")
        np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(cur["generation"][masked], epoch[masked],
                                      err_msg=label)
        _assert_valid_is_fail_mask_zero(_rows(cur, masked), label)
        if keep_twin is not None:
            keep = _poisoned(_snapshot(keep_twin.program.result))
            _assert_host(_rows(cur, masked), _rows(keep, masked),
                         f"{label}: masked envs vs keep twin, poisoned")
            bad = cur["fail_mask"][masked] != 0
            np.testing.assert_array_equal(cur["count"][masked][bad], 0, err_msg=label)
            poisoned_masked += int(bad.sum())
            kept_masked += int((~bad).sum())
        for e in masked:
            assert _new_batch(cur, committed, e), f"{label}: masked env {e}: no new batch"
            for name, v in cur.items():
                committed[name].reshape(E, -1)[e] = v.reshape(E, -1)[e]
    _assert_host(cur, ref, "after an all-ones mask every env is committed")
    if on_invalid == "poison":
        assert poisoned_masked and kept_masked, (poisoned_masked, kept_masked)


@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _shift_vec2_k(counter: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    a[t] = a[t] + wp.vec2f(1.0e-3 * float(counter[0]), 0.0)


@wp.kernel
def _shift_f32_k(counter: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.float32)):
    t = wp.tid()
    a[t] = a[t] + 1.0e-3 * float(counter[0])


@wp.kernel
def _shift_i32_k(counter: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.int32)):
    t = wp.tid()
    a[t] = a[t] + counter[0]


_SHIFT = {wp.vec2f: _shift_vec2_k, wp.float32: _shift_f32_k, wp.int32: _shift_i32_k}


def _load_shift_kernels() -> None:
    """Load the test kernels outside any capture (on scratch buffers)."""
    c = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.launch(_bump_k, 1, inputs=[c], device=DEV)
    for dtype, k in _SHIFT.items():
        wp.launch(k, 4, inputs=[c, wp.zeros(4, dtype=dtype, device=DEV)], device=DEV)
    wp.synchronize()


def _perturb_uniform(gen) -> wp.array:
    """Wrap the uniform resample's enqueue (test-only): after it, bump a device counter
    and shift, for EVERY env, by ``counter``-dependent amounts: the uniform's ``cl.xy``
    (a work buffer the later stages read), its ``cl.half_width`` (read by offset, which
    writes ``Track.half_width``) and the final version of every exported attribute,
    which IS the Track's array (the pipeline overwrites the result buffers of unmasked
    envs too). Returns the counter."""
    program = gen.program
    track = program.result
    i = list(program._stage_ids).index("uniform")
    base = program._enqueues[i]
    targets = [program.tap("cl.xy", after="uniform"), program.tap("cl.half_width")]
    for key, arr in track.attributes.items():
        final = program.tap("cl." + key)
        assert final.ptr == arr.ptr, f"tap('cl.{key}') must be Track.attributes[{key!r}]"
        assert program._writers("cl." + key)[-1] == "uniform", program._writers("cl." + key)
        targets.append(arr)
    assert program._writers("cl.half_width")[-1] == "uniform", \
        program._writers("cl.half_width")
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.synchronize()

    def enqueue(ctx):
        base(ctx)
        ctx.launch(_bump_k, 1, [counter])
        for arr in targets:
            ctx.launch(_SHIFT[arr.dtype], int(arr.shape[0]), [counter, arr])

    program._enqueues[i] = enqueue
    return counter


def test_masked_regeneration_restores_attributes_the_pipeline_overwrote(acme):
    """The uniform resample's output is perturbed differently on every generation (a
    device counter bumped in-graph), INCLUDING the final attribute versions, which are
    the Track's own arrays: every generation overwrites the attribute rows of unmasked
    envs with values that differ from replay to replay. A recorded masked regeneration
    still leaves every unmasked env's Track rows (geometry, ``half_width``, every
    attribute) bitwise unchanged (the commit restores the held bytes, it never relies
    on recomputing them identically), while the masked envs equal an eager twin
    perturbed the same way, and differ from an unperturbed one (the shift reached the
    result buffers)."""
    spec = _ribbon_spec("keep")
    zero = np.zeros(E, np.float32)
    gen, twin, plain = (_prepared(spec, zero) for _ in range(3))
    out = gen.program.result
    m = gen.inputs["regen_mask"]
    _load_shift_kernels()
    counter, counter_t = _perturb_uniform(gen), _perturb_uniform(twin)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(m, mask_src, stream=s)
        gen.generate(stream=s, advance=True, mask=m)
    assert int(counter.numpy()[0]) == 0, "the recording executed the pipeline"

    tap = gen.program.tap
    for k, mk in enumerate([_M1, _M1, 1 - _M1]):
        label = f"replay {k} mask {mk.tolist()}"
        before = _snapshot(out)
        tap_before = tap("cl.xy").numpy().reshape(E, -1).copy()
        _write(mask_src, mk)
        _replay(cap.graph, s)
        for t in (twin, plain):
            _write(t.inputs["regen_mask"], mk)
            t.enqueue(advance=True, mask=t.inputs["regen_mask"])
        wp.synchronize()
        assert int(counter.numpy()[0]) == int(counter_t.numpy()[0]) == k + 1, label
        cur = _snapshot(out)
        tap_now = tap("cl.xy").numpy().reshape(E, -1)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        recomputed = [e for e in unmasked
                      if not np.array_equal(tap_now[e], tap_before[e], equal_nan=True)]
        live = [e for e in unmasked if before["count"][e] > 0]
        assert live and recomputed == live, \
            f"{label}: the unmasked recomputation did not change (vacuous): {recomputed}"
        _assert_host(_rows(cur, unmasked), _rows(before, unmasked),
                     f"{label}: an unmasked env's Track changed")
        _assert_host(_rows(cur, masked), _rows(_snapshot(twin.program.result), masked),
                     f"{label}: masked envs vs the perturbed twin")
        unperturbed = _rows(_snapshot(plain.program.result), masked)
        mrows = _rows(cur, masked)
        for key in _attr_keys(cur) + ["half_width"]:
            assert not np.array_equal(mrows[key], unperturbed[key], equal_nan=True), \
                f"{label}: {key}: the perturbation did not reach the result buffer " \
                f"(vacuous)"


# ---------------------------------------------------------------------------------------
# 5. GPU validations (design 4.3): expected exact, and allclose where stated
# ---------------------------------------------------------------------------------------

def test_families_never_change_track_geometry_on_cuda(acme):
    """ribbon (five exported families + ``raw.acme.hw``, so both resamplers run their
    table-writing lookups ``_cs_lookup_tables_k`` / ``_resample_lookup_tables_k`` and the
    carries) vs ribbon_plain (the same ``raw.xy``, no family: the plain lookups) at the
    default constant width: ``cl.xy`` after each resampler and every Track array but
    the attributes are byte-equal on CUDA (the variant's body is the plain lookup plus
    three stores)."""
    a, _ = _make(_track(generator=RIBBON))
    b, _ = _make(_track(generator=RIBBON_PLAIN))
    ka, kb = _traced_kernels(a), _traced_kernels(b)
    assert {"_cs_lookup_tables_k", "_resample_lookup_tables_k"} <= ka, sorted(ka)
    assert not {"_cs_lookup_tables_k", "_resample_lookup_tables_k"} & kb, sorted(kb)
    assert {"_cs_lookup_k", "_resample_lookup_k"} <= kb, sorted(kb)
    for sid in ("resample", "uniform"):
        _assert_gpu_exact(a.program.tap("cl.xy", after=sid).numpy(),
                          b.program.tap("cl.xy", after=sid).numpy(), f"cl.xy after {sid}")
    ha, hb = _snapshot(a.program.result), _snapshot(b.program.result)
    assert len(_attr_keys(ha)) == len(RIBBON_ATTRS) and not _attr_keys(hb), \
        (sorted(ha), sorted(hb))
    ga, gb = _geometry(ha), _geometry(hb)
    assert ga.keys() == gb.keys(), sorted(set(ga) ^ set(gb))
    for name in ga:
        _assert_gpu_exact(ga[name], gb[name], f"Track.{name}")
    assert ha["valid"].any(), f"no valid env (vacuous): {ha['fail_mask'].tolist()}"


def test_linear_carry_reproduces_the_resampled_points_on_cuda(acme):
    """A ``linear`` carry of ``raw.xy`` itself (``raw.acme.xy_copy``) through the
    resamplers' tables reproduces the resampled points byte for byte on CUDA:
    ``cl.acme.xy_copy@resample == cl.xy@resample`` (constant spacing), and without relax
    (beads do not move) ``Track.attributes["acme.xy_copy"] == cl.xy@uniform`` on every
    real point, NaN padding beyond ``count`` (the carry expression is the lookups'
    own)."""
    gen, _ = _make(_track(generator=RIBBON, relax=None))
    tap = gen.program.tap
    _assert_gpu_exact(tap("cl.acme.xy_copy", after="resample").numpy(),
                      tap("cl.xy", after="resample").numpy(), "xy_copy after resample")
    track = gen.program.result
    got = track.attributes["acme.xy_copy"].numpy().reshape(E, -1, 2)
    want = tap("cl.xy", after="uniform").numpy().reshape(E, -1, 2)
    count = track.count.numpy()
    assert count.max() > 0
    for e in range(E):
        m = int(count[e])
        _assert_gpu_exact(got[e, :m], want[e, :m], f"env {e}: xy_copy after uniform")
    _assert_padding(track.attributes["acme.xy_copy"].numpy(), count, "xy_copy")


def test_stations_carry_and_xy_branch_match_pr11_on_cuda(acme):
    """Stations gate pipelines over ribbon (carrying its families: ``_stations_ex_k`` +
    ``resample_channels`` + ``carry.attributes``) vs ribbon_plain (PR11's
    ``_stations_k``): ``anchors.xy`` / ``anchors.count`` and every GateSequence array but
    the attributes are byte-equal on CUDA (the xy branch is PR11's code), and the linear
    carry of ``raw.xy`` through the anchors tables reproduces ``anchors.xy`` byte for
    byte (padding included)."""
    a, _ = _make(_gates(generator=RIBBON, gates=G_ST))
    b, _ = _make(_gates(generator=RIBBON_PLAIN, gates=G_ST))
    ka, kb = _traced_kernels(a), _traced_kernels(b)
    assert "_stations_ex_k" in ka and "_stations_k" not in ka, sorted(ka)
    assert "_stations_k" in kb and "_stations_ex_k" not in kb, sorted(kb)
    ta, tb = a.program.tap, b.program.tap
    for ch in ("anchors.xy", "anchors.count"):
        _assert_gpu_exact(ta(ch).numpy(), tb(ch).numpy(), ch)
    _assert_gpu_exact(ta("anchors.acme.xy_copy").numpy(), ta("anchors.xy").numpy(),
                      "anchors.acme.xy_copy vs anchors.xy")
    ha, hb = _snapshot(a.program.result), _snapshot(b.program.result)
    assert set(_attr_keys(ha)) == {_ATTR + k for k in RIBBON_ATTRS}, sorted(ha)
    ga, gb = _geometry(ha), _geometry(hb)
    assert ga.keys() == gb.keys(), sorted(set(ga) ^ set(gb))
    for name in ga:
        _assert_gpu_exact(ga[name], gb[name], f"GateSequence.{name}")
    assert ha["valid"].any(), f"no valid env (vacuous): {ha['fail_mask'].tolist()}"


@pytest.mark.parametrize("hw", [0.1, 0.25])
def test_constant_valued_variant_width_is_exact_without_relax_on_cuda(acme, hw):
    """``acme.wave`` with ``hw_amp = 0`` writes the constant ``hw`` through the PER-POINT
    path (no constant declaration: ``_offset_build_w_k`` / ``_offset_assign_w_k`` /
    ``_validity_w_k`` and the uniform's lerp of a constant-valued channel), vs
    ``width.constant(hw)`` (the scalar kernels). Without relax the centerline is
    identical, so every Track array (band, ``half_width``, ``valid`` and every fail
    bit, thickness included) is byte-equal on CUDA. At ``hw = 0.25`` some envs fail the
    thickness check, so the per-pair / per-point flags are exercised."""
    var, _ = _make(_track(relax=None, half_width=hw,
                          width=(WAVE, {"hw_mean": hw, "hw_amp": 0.0})))
    const, _ = _make(_track(relax=None, half_width=hw))
    kv, kc = _traced_kernels(var), _traced_kernels(const)
    assert {"_offset_build_w_k", "_validity_w_k"} <= kv, sorted(kv)
    assert not {"_offset_build_w_k", "_validity_w_k"} & kc, sorted(kc)
    assert {"_offset_build_k", "_validity_k"} <= kc, sorted(kc)
    hv, hc = _snapshot(var.program.result), _snapshot(const.program.result)
    assert hv.keys() == hc.keys(), sorted(set(hv) ^ set(hc))
    for name in hc:
        _assert_gpu_exact(hv[name], hc[name], f"hw={hw}: Track.{name}")
    count = hc["count"]
    for e in range(E):
        m = int(count[e])
        np.testing.assert_array_equal(hc["half_width"].reshape(E, -1)[e, :m],
                                      np.float32(hw), err_msg=f"env {e}: half_width")
    if hw == 0.25:
        thick = (hc["fail_mask"] >> _THICKNESS_BIT) & 1
        assert thick.any(), f"no env fails thickness (vacuous): {hc['fail_mask'].tolist()}"


def test_xpbd_variant_matches_the_scalar_solve_on_cuda(acme):
    """The variable-width XPBD (``_xpbd_radius_k`` + the ``_w`` solver kernels) on a
    constant-valued width (``acme.wave``, ``hw_amp = 0``) vs the scalar solve
    (``width.constant``) at the default ``(half_width, margin)``: the same beads
    (``cl.count``) and relaxed positions within :data:`_XPBD_ATOL` on CUDA (exact on
    CPU; ``allclose`` here, design 4.3)."""
    hw = 0.1
    var, _ = _make(_track(half_width=hw, width=(WAVE, {"hw_mean": hw, "hw_amp": 0.0})))
    const, _ = _make(_track(half_width=hw))
    kv, kc = _traced_kernels(var), _traced_kernels(const)
    assert "_xpbd_radius_k" in kv and "_xpbd_radius_k" not in kc, (sorted(kv), sorted(kc))
    cv, cc = var.program.tap("cl.count").numpy(), const.program.tap("cl.count").numpy()
    np.testing.assert_array_equal(cv, cc, err_msg="cl.count")
    xv = var.program.tap("cl.xy", after="relax").numpy().reshape(E, -1, 2)
    xc = const.program.tap("cl.xy", after="relax").numpy().reshape(E, -1, 2)
    assert cc.max() > 0, cc.tolist()
    for e in range(E):
        m = int(cc[e])
        np.testing.assert_allclose(xv[e, :m], xc[e, :m], rtol=0.0, atol=_XPBD_ATOL,
                                   err_msg=f"env {e}: relaxed beads, variant vs scalar")


def test_stations_xyz_on_cuda_matches_the_cpu_kernel(acme):
    """``spacing_metric="xyz"`` stations on the helix loop (``_stations_ex_k``'s 3D
    branch, f64 accumulation) agree between ``cuda:0`` and ``cpu`` (where
    the CPU suite pins equal 3D spacing to 1e-6): the same gate counts and
    positions within 1e-5 (the generators' transcendentals differ by ulps between the
    devices), while the ``xy`` metric places the stations elsewhere (non-vacuous)."""
    def gates_on(device, metric):
        spec = _helix_gates(metric, device=device)
        rng = PerEnvSeededRNG(seeds=11, num_envs=E, device=device)
        gen = GateGenerator(spec, rng)
        out = gen.enqueue()
        wp.synchronize()
        return gen, _snapshot(out)

    gc, hc = gates_on(DEV, "xyz")
    gp, hp = gates_on("cpu", "xyz")
    _, hxy = gates_on(DEV, "xy")
    np.testing.assert_array_equal(hc["count"], hp["count"], err_msg="gate counts")
    assert hc["valid"].any(), hc["fail_mask"].tolist()
    ac = gc.program.tap("anchors.xy").numpy()
    ap = gp.program.tap("anchors.xy").numpy()
    np.testing.assert_allclose(ac, ap, rtol=0.0, atol=1e-5, equal_nan=True,
                               err_msg="anchors.xy cuda vs cpu")
    np.testing.assert_allclose(gc.program.tap("anchors.z").numpy(),
                               gp.program.tap("anchors.z").numpy(), rtol=0.0, atol=1e-5,
                               equal_nan=True, err_msg="anchors.z cuda vs cpu")
    np.testing.assert_allclose(hc["position"], hp["position"], rtol=0.0, atol=1e-5,
                               equal_nan=True, err_msg="GateSequence.position cuda vs cpu")
    assert not np.allclose(hc["position"], hxy["position"], atol=1e-4, equal_nan=True), \
        "xy and xyz stations coincide (vacuous)"


# ---------------------------------------------------------------------------------------
# 6. Build on a non-current stream: the width.constant fill is ordered
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("prepare", [True, False], ids=["prepare", "lazy"])
def test_constant_width_fill_is_ordered_before_a_side_stream_build(prepare):
    """``width.constant`` launches nothing: its version is ``wp.full(hw)`` allocated at
    build on Warp's CURRENT stream. ``build(..., stream=s)`` with ``s`` another stream,
    while the current stream is held busy (a long kernel queued before the build) on
    recycled memory: the first generation on ``s`` (``prepare``, or the first
    ``run(stream=s)``) must wait for that fill (``GraphRunner.prepare``'s stream wait).
    ``tap("cl.half_width", after="width")`` is ``hw`` on every slot, and the Track (built
    from XPBD, offset and validity reading it on ``s``) equals the eager twin built on
    the current stream, with ``Track.half_width == hw`` on real points, NaN padding."""
    hw = 0.09
    spec = _track(half_width=hw)
    ref = _eager(spec)
    s = _side_stream()
    busy = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(_spin_k, 1, inputs=[16, busy], device=DEV)         # load the module
    wp.synchronize()
    with _dirty_pool():
        wp.launch(_spin_k, 1, inputs=[1 << 22, busy], device=DEV)   # current stream busy
        program = build(spec, seeds=PerEnvSeededRNG(seeds=11, num_envs=E, device=DEV),
                        stream=s, prepare=prepare)
        if not prepare:
            program.run(stream=s)
        wp.synchronize()
    fill = program.tap("cl.half_width", after="width").numpy()
    np.testing.assert_array_equal(fill, np.full(fill.shape, np.float32(hw)),
                                  err_msg="the width.constant version")
    got = _snapshot(program.result)
    _assert_host(got, ref, "side-stream build vs eager twin")
    count = got["count"]
    hwr = got["half_width"].reshape(E, -1)
    for e in range(E):
        m = int(count[e])
        np.testing.assert_array_equal(hwr[e, :m], np.float32(hw), err_msg=f"env {e}")
        assert np.all(np.isnan(hwr[e, m:])), f"env {e}: half_width padding"


# ---------------------------------------------------------------------------------------
# 7. Carried padding on recycled device memory
# ---------------------------------------------------------------------------------------

#: ribbon's dynamic ``invalid`` flag per env (1: the generator writes ``gen.valid = 0``).
_FLIP = np.array([1, 0] * (E // 2), np.float32)


def _assert_rows_padding(arr: np.ndarray, bad, label: str) -> None:
    rows = arr.reshape(E, -1)[bad]
    ok = np.all(rows == -1) if np.issubdtype(arr.dtype, np.integer) \
        else np.all(np.isnan(rows))
    assert ok, f"{label}: an env the carriers skipped is not pure padding"


def test_carried_padding_is_exact_on_dirty_cuda_memory(acme):
    """Built on recycled device memory (stale non-zero bytes in every ``wp.empty``
    buffer), with ribbon's ``invalid`` flag (``gen.valid = 0``) set for EVERY env on the
    first generation (construction), then for every other env, then for the others
    (after a reseed: envs that HAD real tables become invalid):

    - the Track: every attribute holds exactly NaN / -1 beyond ``count`` and finite,
      in-range values on real points, on every generation; the generator bit is set
      exactly on the flagged envs;
    - the ribbon stations gate Program: a flagged env has no gates and pure padding in
      every carried anchors image (``tap("anchors.acme.*")``) and GateSequence
      attribute (the stations' early exit writes the table sentinels on every exit
      path, the carries branch on ``i0 < 0``), so neither stale memory nor a previous
      generation's tables leak into a skipped env; the other envs get all their
      stations, with in-range attributes."""
    gp = {"invalid": 1.0}
    with _dirty_pool():
        tgen, trng = _make(_track(generator=RIBBON, generator_params=gp))
        ggen, grng = _make(_gates(generator=RIBBON, generator_params=gp, gates=G_ST))
    schedule = (np.ones(E, np.float32), _FLIP, 1.0 - _FLIP)
    for gen, rng, kind in ((tgen, trng, "track"), (ggen, grng, "gates")):
        sid = _stage_id(gen.program.spec, "generator", RIBBON)
        for k, flags in enumerate(schedule):
            label = f"{kind} generation {k} invalid={flags.tolist()}"
            if k:
                _set(gen, sid, "invalid", flags)
                _reseed(rng, 100 + k)
                gen.generate()
                wp.synchronize()
            host = _snapshot(gen.program.result)
            bad = flags != 0
            assert len(_attr_keys(host)) == len(RIBBON_ATTRS), sorted(host)
            _check_attributes(gen, host, label)
            if kind == "track":
                gen_bit = (host["fail_mask"] >> BUILTIN_FAIL_BITS["generator"]) & 1
                np.testing.assert_array_equal(gen_bit, bad.astype(gen_bit.dtype),
                                              err_msg=f"{label}: generator bit")
                continue
            np.testing.assert_array_equal(host["count"][bad], 0, err_msg=f"{label}: count")
            np.testing.assert_array_equal(host["count"][~bad], G_ST, err_msg=label)
            for key in RIBBON_ATTRS:
                _assert_rows_padding(gen.program.tap(f"anchors.{key}").numpy(), bad,
                                     f"{label}: anchors.{key}")
                _assert_rows_padding(host[_ATTR + key], bad, f"{label}: gates {key}")


# ---------------------------------------------------------------------------------------
# 8. The conformance kit on CUDA
# ---------------------------------------------------------------------------------------

def _assert_kit_captured(report, label: str) -> None:
    report.raise_for_failures()
    caller = report.result("capture.caller", DEV)
    assert not caller.skipped and caller.details.get("warp") == "ok", f"{label}: {report}"
    assert not report.result("capture.self_graph", DEV).skipped, f"{label}: {report}"


_KIT_WIDTHS = {
    "from_channel": dict(ports={"src": HW_SRC}, upstream={"generator": RIBBON}),
    WAVE: {},
    RESCALE: {},
}


@pytest.mark.parametrize("name", list(_KIT_WIDTHS))
def test_width_stage_passes_check_stage_on_cuda(acme, name):
    """``check_stage`` on ``cuda:0`` for the width registry (``width.from_channel`` wired
    by ``ports=`` to ribbon's ``cl.acme.hw``, and the plugin width stages): every check
    passes (``width.values`` included, not skipped), the stage recorded into a caller
    capture (and the torch modes when available) replays to the eager twin, and the
    Program's own self-graph replays the eager batch."""
    report = check_stage("width", name, devices=[DEV], **_KIT_WIDTHS[name])
    _assert_kit_captured(report, f"width.{name}")
    assert not report.result("width.values", DEV).skipped, str(report)


@pytest.mark.parametrize("name", [RIBBON, HELIX_LOOP, PINCH])
def test_toy_generator_passes_check_generator_on_cuda(acme, name):
    """``check_generator`` on ``cuda:0`` for the family-writing / providing toys: every
    check passes and the caller / self-graph recordings replay to the eager batch;
    ribbon's ``attributes.written`` / ``attributes.carried`` run (not skipped)."""
    report = check_generator(name, devices=[DEV])
    _assert_kit_captured(report, name)
    if name == RIBBON:
        for check in ("attributes.written", "attributes.carried"):
            assert not report.result(check, DEV).skipped, str(report)


def test_toy_gate_generator_passes_check_gate_generator_on_cuda(acme):
    """``check_gate_generator`` on ``cuda:0`` for ``acme.tagged`` (an anchors plugin
    family carried by index): every check passes, ``attributes.*`` run, and the caller /
    self-graph recordings replay to the eager batch."""
    report = check_gate_generator(TAGGED, devices=[DEV])
    _assert_kit_captured(report, TAGGED)
    for check in ("attributes.written", "attributes.carried"):
        assert not report.result(check, DEV).skipped, str(report)
