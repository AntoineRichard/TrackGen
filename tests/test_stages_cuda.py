"""CUDA-only: plugin pipeline stages under caller capture, the masked commit and
``on_invalid="poison"`` (roadmap PR10b).

The toy plugins of ``tests/_stage_toys.py`` (public plugin surface only) are installed
through a fake entry point (the registry's real discovery path) and selected through
:func:`track_gen.pipeline.presets.track`:

- relax ``acme.taubin``: an in-place rewrite of ``cl.xy`` (read the previous version,
  ping-pong through ``b.temp_like``, end in the ``b.write`` buffer); dynamic ``lam``;
- elevation ``acme.sinusoid_hills``: ``cl.z`` from the plan-view arc-length tables and a
  salted per-env phase; dynamic ``amplitude``;
- validity ``acme.max_curvature``: ORs its own fail bit (``b.fail_bit()``, 16..30) into
  ``valid.fail`` where ``max(cl.kappa) > kappa_max[e]``; dynamic ``kappa_max``.

Method (as in tests/test_caller_capture_cuda.py and tests/test_inplace_regen_cuda.py;
the helpers are :mod:`track_gen.testing.capture`'s): record on a NON-current side
stream, poison the outputs with FINITE, dtype-aware sentinels (outputs are NaN-padded,
poisoned envs are NaN, and ``assert_array_equal`` treats NaN == NaN, so a NaN poison
would hide a missed write), replay, and compare with an eager twin: a fresh
``TrackGenerator`` of the same spec run with ``enqueue()`` (never a graph). Every
recording runs with the Warp mempool disabled (an allocation inside the capture fails
loudly) and ``wp.capture_launch`` replaced by a tripwire (a graph launch cannot be
recorded into another capture). Construction prepares the default body; the masked
body is warmed up with an all-zero mask (an exact no-op) and the test's own kernels on
scratch buffers, so no module loads while capturing.

The masked-commit tests poison the WHOLE Track (``generation`` included) before every
masked replay: the commit decides what each env's Track holds, and an env with
``mask[e] == 0`` must end the replay with exactly the bytes it began it with, whatever
the stages recomputed for it. Its rows must therefore still hold the poison, not merely
equal values (which a bitwise-deterministic recomputation would also give); a second
test makes the recomputation differ on every replay.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from tests._plugin_toys import install_entry_points  # noqa: E402
from tests._stage_toys import acme_entry_point  # noqa: E402
from track_gen import PerEnvSeededRNG, TrackGenerator  # noqa: E402
from track_gen.pipeline import presets  # noqa: E402
from track_gen.plugin import POLICY_ENV, registry  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import check_stage  # noqa: E402

DEV = "cuda:0"
E = 8

TAUBIN = "acme.taubin"
HILLS = "acme.sinusoid_hills"
KMAX = "acme.max_curvature"
# (kind, name, dynamic field, a uniform value, per-env values): every value changes the
# batch (asserted), neighbours differ so a write that leaks across envs shows up.
TOYS = (
    ("relax", TAUBIN, "lam", 0.3, [0.3, 0.8]),
    ("elevation", HILLS, "amplitude", 1.0, [0.5, 2.0]),
    ("validity", KMAX, "kappa_max", 0.0, [0.0, 1.0e6]),
)
_IDS = [name for _, name, *_ in TOYS]

_ZERO = np.zeros(E, np.int32)
_ONES = np.ones(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
# kappa_max per env for the poison / masked tests: 0 fails every non-empty loop, 1e6
# none, so of the envs _M1 selects, 0 and 2 are always poisoned while 5 and 7 are kept
# unless a built-in check fails them.
_KAPPA_SPLIT = np.array([0.0, 1.0e6] * (E // 2), np.float32)
_POINT_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen")


@pytest.fixture(autouse=True)
def acme(monkeypatch):
    """The ``acme-stages`` toy distribution, discovered through a fake entry point in a
    fresh registry scope (restored afterwards, as is ``TRACK_GEN_PLUGINS``)."""
    monkeypatch.delenv(POLICY_ENV, raising=False)
    install_entry_points(monkeypatch, acme_entry_point())
    with registry.scoped(fresh=True):
        for kind, name, *_ in TOYS:
            assert not registry.describe(kind, name).builtin, (kind, name)
        yield registry


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True)
            for k, a in tg_capture.result_arrays(obj).items()}


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


def _spec(**kw):
    return presets.track(num_envs=E, device=DEV, **kw)


def _sel(name: str, field: str, value):
    """A preset stage selection: the toy with its defaults, or ``field=value``."""
    return name if value is None else (name, {field: float(value)})


def _toy_spec(kind: str, name: str, field: "str | None" = None, value=None, **kw):
    """The default preset with ONE toy in its slot (a validity toy after the default
    checks)."""
    sel = _sel(name, field, value)
    if kind == "validity":
        return _spec(checks=("default", sel), **kw)
    return _spec(**{kind: sel}, **kw)


# Taubin smoothing has no self-separation term: at the default half-width (0.1) every
# loop fails the built-in thickness check, which would make every env invalid and the
# poison tests vacuous. A thinner band on a coarser step leaves some envs valid.
_ALL_TOYS_BAND = {"half_width": 0.03, "spacing": 0.06}


def _all_toys(on_invalid: str = "keep", *, lam=None, amplitude=None, kappa_max=None):
    """The preset with all three toys (a thinner band, :data:`_ALL_TOYS_BAND`) and
    ``on_invalid``."""
    return _spec(relax=_sel(TAUBIN, "lam", lam),
                 elevation=_sel(HILLS, "amplitude", amplitude),
                 checks=("default", _sel(KMAX, "kappa_max", kappa_max)),
                 on_invalid=on_invalid, **_ALL_TOYS_BAND)


def _stage_id(spec, kind: str, name: str) -> str:
    ids = [r.slot_id for r in spec.stages if r.kind == kind and r.name == name]
    assert len(ids) == 1, f"{kind}.{name} in {spec.ids}"
    return ids[0]


def _make(spec, seeds=11) -> TrackGenerator:
    return TrackGenerator(spec, PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV))


def _eager(spec, seeds=11) -> dict:
    """A fresh twin of ``spec`` run eagerly (``enqueue`` never self-graphs)."""
    twin = _make(spec, seeds)
    out = twin.enqueue()
    wp.synchronize()
    assert twin.graph is None, "enqueue() must never self-graph"
    return _snapshot(out)


class _Twins:
    """Eager-twin snapshots of ``make(value)`` (cached per value), and per-env
    stitching: env ``e`` of :meth:`rows` comes from the twin built with ``values[e]``,
    so the per-env latch under test is not also the reference."""

    def __init__(self, make) -> None:
        self._make = make
        self._cache: dict = {}

    def uniform(self, value) -> dict:
        key = float(np.float32(value))
        if key not in self._cache:
            self._cache[key] = _eager(self._make(key))
        return self._cache[key]

    def rows(self, values) -> dict:
        return _stitch([self.uniform(v) for v in _per_env(values)])


def _set(gen, stage_id: str, field: str, values, *, stream=None) -> None:
    """Write a dynamic field's live array: a scalar ``set`` for uniform values, an
    ``[E]`` array ``set`` otherwise."""
    v = _per_env(values)
    sp = gen.params[stage_id]
    src = float(v[0]) if np.all(v == v[0]) else _dev(v)
    sp.set(field, src, stream=stream)
    wp.synchronize()                       # src stays alive until the copy completed


def _max_kappa(gen) -> np.ndarray:
    """Per env: what ``acme.max_curvature`` compares with ``kappa_max``, the max of the
    final ``cl.kappa`` over the real points, NaN ignored (``-inf`` when none)."""
    tap = gen.program.tap
    kappa = tap("cl.kappa").numpy().reshape(E, -1)
    count = tap("cl.count").numpy()
    out = np.full(E, -np.inf)
    for e in range(E):
        k = kappa[e, :count[e]]
        k = k[~np.isnan(k)]
        if k.size:
            out[e] = float(k.max())
    return out


def _split_threshold(kmax: np.ndarray) -> float:
    """A float32 ``kappa_max`` strictly between two per-env maxima around the median, so
    the bit is set for some envs and clear for others."""
    finite = np.unique(kmax[np.isfinite(kmax)].astype(np.float32))
    assert finite.size >= 2, f"too few finite loops to split: {kmax.tolist()}"
    k = finite.size // 2
    thr = np.float32(0.5 * (float(finite[k - 1]) + float(finite[k])))
    assert finite[k - 1] < thr < finite[k], (finite.tolist(), thr)
    return float(thr)


def _poisoned(keep: dict) -> dict:
    """The ``on_invalid="poison"`` Track of a ``"keep"`` snapshot, derived independently
    of the commit kernel: every env with ``fail_mask != 0`` gets ``count = 0``, NaN on
    every per-point field, ``length = winding = 0`` and ``valid = 0``; ``fail_mask``,
    ``truncated`` and ``generation`` keep their values; valid envs are unchanged."""
    out = {k: v.copy() for k, v in keep.items()}
    bad = keep["fail_mask"] != 0
    for name in _POINT_FIELDS:
        out[name].reshape(E, -1)[bad] = np.nan
    out["count"][bad] = 0
    out["length"][bad] = 0.0
    out["winding"][bad] = 0.0
    out["valid"][bad] = 0
    return out


def _assert_valid_is_fail_mask_zero(host: dict, label: str) -> None:
    np.testing.assert_array_equal(host["valid"], (host["fail_mask"] == 0).astype(np.int32),
                                  err_msg=f"{label}: valid != (fail_mask == 0)")


# ---------------------------------------------------------------------------------------
# Each toy: the conformance kit's capture checks, and presets -> caller capture
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,name", [t[:2] for t in TOYS], ids=_IDS)
def test_toy_stage_passes_check_stage_on_cuda(kind, name):
    """``check_stage`` on ``cuda:0``: every check passes, the toy recorded into a Warp
    caller capture (and the torch modes when available) replays to the eager twin, and
    the Program's own self-graph replays the eager batch. The relax toy runs on the
    thinner band (:data:`_ALL_TOYS_BAND`), so ``output.accepted`` is not skipped."""
    upstream = dict(_ALL_TOYS_BAND) if kind == "relax" else None
    report = check_stage(kind, name, upstream=upstream, devices=[DEV])
    report.raise_for_failures()
    caller = report.result("capture.caller", DEV)
    assert not caller.skipped and caller.details.get("warp") == "ok", str(report)
    assert not report.result("capture.self_graph", DEV).skipped, str(report)
    if kind != "validity":
        assert not report.result("output.accepted", DEV).skipped, str(report)
    else:
        assert not report.result("validity.bits", DEV).skipped, str(report)


@pytest.mark.parametrize("kind,name", [t[:2] for t in TOYS], ids=_IDS)
def test_toy_stage_selected_through_presets_records_into_caller_capture(kind, name):
    """``TrackGenerator(presets.track(<slot>=toy), rng)``: the toy is in the recorded
    program and changes the batch (non-vacuous), ``generate(stream=s)`` records into
    the caller's capture (no allocation, no capture_launch, no self-graph) and every
    replay into poisoned outputs equals the eager twin, before and after a reseed."""
    if kind == "validity":
        probe = _make(_toy_spec(kind, name))
        thr = _split_threshold(_max_kappa(probe))
        spec = _toy_spec(kind, name, "kappa_max", thr)
    else:
        spec = _toy_spec(kind, name)
    gen = _make(spec)
    sid = _stage_id(spec, kind, name)
    snap = _snapshot(gen.program.result)
    _assert_valid_is_fail_mask_zero(snap, name)
    tap = gen.program.tap
    if kind == "relax":
        # A rewrite: the relax version is new, the version it read is left intact.
        assert not np.array_equal(tap("cl.xy", after=sid).numpy(),
                                  tap("cl.xy", after="resample").numpy(), equal_nan=True)
        assert _differs(snap, _eager(_spec())), "acme.taubin must change the batch"
    elif kind == "elevation":
        z = np.nan_to_num(snap["center"].reshape(E, -1, 3)[..., 2])
        assert np.any(z != 0.0), "acme.sinusoid_hills must lift the centerline"
        assert _differs(snap, _eager(_spec())), "acme.sinusoid_hills must change the batch"
    else:
        bit = gen.program.fail_bits[sid]
        assert 16 <= bit <= 30 and snap["fail_mask"].dtype == np.int32, bit
        own = (snap["fail_mask"] >> bit) & 1
        want = (_max_kappa(gen) > np.float32(thr)).astype(own.dtype)
        np.testing.assert_array_equal(own, want, err_msg="bit set where max kappa > kappa_max")
        assert own.any() and not own.all(), f"vacuous split: {own.tolist()}"
    status = tg_capture.assert_caller_capturable(gen)
    assert status["warp"] == "ok", status
    assert gen.graph is None, "caller recordings never self-capture"


@pytest.mark.parametrize("kind,name,field,uniform,per_env", TOYS, ids=_IDS)
def test_dynamic_stage_param_written_between_replays_reaches_the_next_replay(
        kind, name, field, uniform, per_env):
    """ONE recording of ``generate(stream=s)``. Between replays the toy's dynamic field
    is written in place (``gen.params[stage_id].set``: a scalar, then an ``[E]`` array,
    then back to the default). The recording executed nothing (live array and Track
    unchanged); each write lands in the live array (stable pointer) and the next replay
    latches it and equals the eager twin(s) built with those values (per-env values:
    rows stitched from scalar twins). Nothing is re-recorded (no self-graph)."""
    spec = _toy_spec(kind, name)
    gen = _make(spec)
    sid = _stage_id(spec, kind, name)
    live = gen.params[sid][field]
    ptr = live.ptr
    default = float(getattr(gen.params[sid].params, field))
    twins = _Twins(lambda v: _toy_spec(kind, name, field, v))
    base = _snapshot(gen.program.result)
    _assert_host(base, twins.uniform(default), f"{name}: construction batch")

    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert out is gen.program.result and gen.graph is None
    np.testing.assert_array_equal(live.numpy(), _per_env(default),
                                  err_msg="the recording executed a write")
    _assert_host(_snapshot(out), base, f"{name}: the recording executed the pipeline")

    schedule = (_per_env(uniform), _per_env(per_env), _per_env(default))
    assert _differs(twins.rows(schedule[0]), twins.uniform(default)), \
        f"{field}={uniform} must change the batch (vacuous otherwise)"
    assert _differs(twins.rows(schedule[1]), twins.rows(schedule[0])), \
        f"per-env {field} must change the batch (vacuous otherwise)"
    for k, values in enumerate(schedule):
        label = f"{name}: replay {k} {field}={values.tolist()}"
        _set(gen, sid, field, values)
        np.testing.assert_array_equal(live.numpy(), values, err_msg=f"{label}: live")
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        got = _snapshot(out)
        _assert_host(got, twins.rows(values), label)
        _assert_valid_is_fail_mask_zero(got, label)
    assert gen.params[sid][field] is live and live.ptr == ptr, "stable live array"
    assert gen.graph is None


# ---------------------------------------------------------------------------------------
# All three toys + on_invalid="poison"
# ---------------------------------------------------------------------------------------

def test_all_toys_with_poison_record_into_caller_capture():
    """One Program with the three toys and ``on_invalid="poison"`` (``kappa_max``
    between two per-env maxima, so some envs are poisoned and some are not): eagerly it
    is exactly the ``"keep"`` batch with the invalid envs poisoned (``count = 0``, NaN
    geometry, ``length = winding = valid = 0``, ``fail_mask`` / ``truncated`` kept), and
    recorded into a caller's capture every replay into poisoned outputs equals the
    eager twin, before and after a reseed."""
    thr = _split_threshold(_max_kappa(_make(_all_toys())))
    spec = _all_toys("poison", kappa_max=thr)
    gen = _make(spec)
    snap = _snapshot(gen.program.result)
    bad = snap["fail_mask"] != 0
    assert bad.any() and not bad.all(), f"vacuous poison split: {snap['fail_mask'].tolist()}"
    np.testing.assert_array_equal(snap["count"][bad], 0)
    assert np.isnan(snap["center"].reshape(E, -1)[bad]).all()
    keep = _eager(_all_toys("keep", kappa_max=thr))
    _assert_host(snap, _poisoned(keep), "poison == keep with the invalid envs poisoned")
    _assert_valid_is_fail_mask_zero(snap, "poison")

    status = tg_capture.assert_caller_capturable(gen)
    assert status["warp"] == "ok", status
    assert gen.graph is None


def test_all_toys_params_written_inside_the_caller_graph_with_poison():
    """ONE caller graph holds ``[set lam, amplitude, kappa_max from device sources] ->
    generate()`` (the in-graph curriculum) on the three-toy poison Program. Recording
    executes nothing. Before each replay the sources are rewritten outside the graph;
    the replay copies them into the live arrays, latches them, and equals the poison
    twins built with those values (per-env ``kappa_max`` stitched), which are the keep
    twins with the invalid envs poisoned: the poisoned set follows ``kappa_max`` on
    device, replay by replay."""
    spec = _all_toys("poison")
    gen = _make(spec)
    sids = {"lam": _stage_id(spec, "relax", TAUBIN),
            "amplitude": _stage_id(spec, "elevation", HILLS),
            "kappa_max": _stage_id(spec, "validity", KMAX)}
    defaults = {f: float(getattr(gen.params[sid].params, f)) for f, sid in sids.items()}
    src = {f: _dev(_per_env(v)) for f, v in defaults.items()}
    for f, sid in sids.items():                      # warm-up: a no-op write of each
        gen.params[sid].set(f, src[f])
    wp.synchronize()
    base = _snapshot(gen.program.result)

    s = _side_stream()
    with _recording(s) as cap:
        for f, sid in sids.items():
            gen.params[sid].set(f, src[f], stream=s)
        out = gen.generate(stream=s)
    assert gen.graph is None
    for f, sid in sids.items():
        np.testing.assert_array_equal(gen.params[sid][f].numpy(), _per_env(defaults[f]),
                                      err_msg=f"the recording executed the {f} write")
    _assert_host(_snapshot(out), base, "the recording executed the pipeline")

    cache: dict = {}

    def twin(on_invalid, lam, amplitude, kappa_max) -> dict:
        key = (on_invalid, float(lam), float(amplitude), float(kappa_max))
        if key not in cache:
            cache[key] = _eager(_all_toys(on_invalid, lam=lam, amplitude=amplitude,
                                          kappa_max=kappa_max))
        return cache[key]

    schedule = (
        {"lam": 0.3, "amplitude": 1.0, "kappa_max": _KAPPA_SPLIT},
        {"lam": defaults["lam"], "amplitude": 0.5, "kappa_max": 1.0e6},
        {"lam": 0.8, "amplitude": defaults["amplitude"], "kappa_max": _KAPPA_SPLIT[::-1]},
    )
    seen = []
    for k, step in enumerate(schedule):
        label = f"replay {k} {step}"
        values = {f: _per_env(v) for f, v in step.items()}
        for f in sids:
            _write(src[f], values[f])
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        for f, sid in sids.items():
            np.testing.assert_array_equal(gen.params[sid][f].numpy(), values[f],
                                          err_msg=f"{label}: live {f}")
        got = _snapshot(out)
        lam, amp = float(values["lam"][0]), float(values["amplitude"][0])
        per = [(lam, amp, float(km)) for km in values["kappa_max"]]
        _assert_host(got, _stitch([twin("poison", *p) for p in per]), f"{label}: vs twin")
        _assert_host(got, _poisoned(_stitch([twin("keep", *p) for p in per])),
                     f"{label}: vs keep twin, poisoned")
        _assert_valid_is_fail_mask_zero(got, label)
        seen.append(got)
    # The check's own bit follows kappa_max replay by replay (every loop is non-empty),
    # and each poisoned env is empty.
    bit = gen.program.fail_bits[sids["kappa_max"]]
    for g, step in zip(seen, schedule):
        own = (g["fail_mask"] >> bit) & 1
        np.testing.assert_array_equal(own, (_per_env(step["kappa_max"]) == 0.0).astype(own.dtype))
        np.testing.assert_array_equal(g["count"][g["fail_mask"] != 0], 0)
    assert _differs(seen[0], seen[1]) and _differs(seen[1], seen[2])


# ---------------------------------------------------------------------------------------
# The masked commit under capture
# ---------------------------------------------------------------------------------------

def _prepared(spec) -> TrackGenerator:
    """A generator of ``spec`` with ``kappa_max = _KAPPA_SPLIT`` written to its live
    array and its masked body warmed up with an all-zero mask, which must be an exact
    no-op (nothing committed, no epoch advanced)."""
    g = _make(spec)
    _set(g, _stage_id(spec, "validity", KMAX), "kappa_max", _KAPPA_SPLIT)
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
def test_masked_commit_in_caller_graph_writes_only_the_masked_envs(on_invalid):
    """ONE caller graph holds ``[copy mask_src -> inputs["regen_mask"]] ->
    generate(advance=True, mask=regen_mask)`` (the mask is produced in-graph). Before
    every replay the WHOLE Track is poisoned (``generation`` too); after it:

    - envs with ``mask[e] == 0`` still hold the poison byte for byte: the commit left
      them exactly as the replay found them (an all-zero mask changes no byte at all);
    - masked envs equal an eager twin driven with the same masks (same seed and
      epoch), with ``generation == epoch``, and got a new batch;
    - with ``"poison"`` the masked invalid envs are poisoned exactly as the ``"keep"``
      twin with its invalid envs poisoned, while unmasked ones are untouched;
    - after an all-ones mask every env is committed and the Track equals the twin."""
    spec = _all_toys(on_invalid)
    gen, twin = _prepared(spec), _prepared(spec)
    keep_twin = _prepared(_all_toys("keep")) if on_invalid == "poison" else None
    out = gen.program.result
    m = gen.inputs["regen_mask"]
    twins = [t for t in (twin, keep_twin) if t is not None]
    base = _snapshot(out)
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
        _assert_host(_rows(cur, masked), _rows(ref, masked), f"{label}: masked envs vs twin")
        np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(cur["generation"][masked], epoch[masked], err_msg=label)
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
def _shift_k(counter: wp.array(dtype=wp.int32), xy: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    xy[t] = xy[t] + wp.vec2f(1.0e-3 * float(counter[0]), 0.0)


def _perturb_relax(gen) -> wp.array:
    """Wrap the relax stage's enqueue (test-only): after it, bump a device counter and
    shift the relax version of ``cl.xy`` of EVERY env by ``1e-3 * counter``, so each
    generation recomputes every env differently. Returns the counter."""
    program = gen.program
    i = list(program._stage_ids).index("relax")
    base = program._enqueues[i]
    xy = program.tap("cl.xy", after="relax")
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.synchronize()

    def enqueue(ctx):
        base(ctx)
        ctx.launch(_bump_k, 1, [counter])
        ctx.launch(_shift_k, int(xy.shape[0]), [counter, xy])

    program._enqueues[i] = enqueue
    return counter


def test_masked_commit_under_capture_does_not_rely_on_recomputing_unmasked_envs():
    """The relax stage's output is perturbed differently on every generation (a device
    counter bumped in-graph), so the pipeline's recomputation of an unmasked env
    DIFFERS from replay to replay (its ``tap("cl.xy")`` row changes). A recorded masked
    regeneration still leaves every unmasked env's Track rows bitwise unchanged (the
    commit keeps their bytes, it never relies on recomputing them identically), while
    the masked envs equal an eager twin perturbed the same way (the in-graph counter is
    read at replay time)."""
    spec = _all_toys("keep")
    gen, twin = _prepared(spec), _prepared(spec)
    out = gen.program.result
    m = gen.inputs["regen_mask"]
    scratch = wp.zeros(4, dtype=wp.vec2f, device=DEV)         # load the test kernels
    c0 = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.launch(_bump_k, 1, inputs=[c0], device=DEV)
    wp.launch(_shift_k, 4, inputs=[c0, scratch], device=DEV)
    wp.synchronize()
    counter, counter_t = _perturb_relax(gen), _perturb_relax(twin)
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
        _write(twin.inputs["regen_mask"], mk)
        twin.enqueue(advance=True, mask=twin.inputs["regen_mask"])
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
